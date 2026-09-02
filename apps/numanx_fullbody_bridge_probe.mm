#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/NeuronCultureArtifacts.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/mrnx_bridge_v1.h"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"
#include "metalrobo/numanx_human_io_gpu.h"
#include "numi/matter/shared.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#ifndef MRNX_FULLBODY_RIGID
#error MRNX_FULLBODY_RIGID is required
#endif
#ifndef MRNX_FULLBODY_MUSCLE
#error MRNX_FULLBODY_MUSCLE is required
#endif
#ifndef MRNX_FULLBODY_SUPPORT_CONTACT
#error MRNX_FULLBODY_SUPPORT_CONTACT is required
#endif
#ifndef MRNX_METALROBO_METALLIB
#error MRNX_METALROBO_METALLIB is required
#endif
#ifndef MRNX_MATTER_METALLIB
#error MRNX_MATTER_METALLIB is required
#endif
#ifndef MRNX_MATTER_MATERIAL
#error MRNX_MATTER_MATERIAL is required
#endif

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint64_t kStartMicros = 1'000u;
constexpr std::uint64_t kDurationMicros = 2'000u;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void mixU32(std::uint64_t& hash, const std::uint32_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixU64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixFloat(std::uint64_t& hash, const float value) noexcept {
    mixU32(hash, std::bit_cast<std::uint32_t>(value));
}

std::vector<std::uint8_t> readPayloadBytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "provenance payload did not open");
    std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    require(!bytes.empty() && !input.bad(),
            "provenance payload did not read exactly");
    return bytes;
}

void qualifyHumanSupportKKT(id<MTLDevice> device) {
    NSError* libraryError = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:MRNX_MATTER_METALLIB]]
                       error:&libraryError];
    require(library != nil, "Matter support qualification library failed");
    const auto pipeline = [&](NSString* name) {
        id<MTLFunction> function = [library newFunctionWithName:
            [@"numi_matter_metal::" stringByAppendingString:name]];
        NSError* error = nil;
        id<MTLComputePipelineState> result = function == nil ? nil :
            [device newComputePipelineStateWithFunction:function error:&error];
        require(result != nil, "Matter support qualification pipeline failed");
        return result;
    };
    id<MTLComputePipelineState> checkpointPipeline =
        pipeline(@"nm_human_support_checkpoint");
    id<MTLComputePipelineState> evaluatePipeline =
        pipeline(@"nm_human_support_evaluate");
    id<MTLComputePipelineState> residualPipeline =
        pipeline(@"nm_human_support_accumulate_rigid_residual");
    id<MTLComputePipelineState> operatorPipeline =
        pipeline(@"nm_fgmres_apply_human_support");
    id<MTLComputePipelineState> commitPipeline =
        pipeline(@"nm_human_support_commit");
    id<MTLComputePipelineState> rollbackPipeline =
        pipeline(@"nm_human_support_rollback");

    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = 1u;
    dispatch.rigidGeneralizedCapacity = 1u;
    NMHumanSupportDispatchGPU support{};
    support.contactCount = 1u;
    support.articulatedNv = 1u;
    support.bodyCount = 1u;
    support.bodyStride = 1u;
    support.groundPointAndTimestep.w = 0.01f;
    support.groundNormal.y = 1.0f;
    NMHumanSupportContactGPU contact{};
    contact.identity = {0u, 7u, 0u, 0u};
    contact.localPoint.y = -0.01f;
    contact.frictionSlopAndStabilization = {0.5f, 0.02f, 0.2f, 0.0f};
    MRBodyStateGPU body{};
    body.orientation.w = 1.0f;
    const float generalized = -1.0f;
    const std::array<float, 3u> jacobian{0.0f, 1.0f, 0.0f};
    const nm_float4 zero4{};
    const NMMatterStatusGPU success{};
    NMMatterStatusGPU failure{};
    failure.code = NM_STATUS_CONTACT_FAILURE;
    NMFGMRESStateGPU fgmres{};
    nm_float4 direction{};
    direction.x = 2.0f;

    const auto buffer = [&](const void* bytes, const NSUInteger length) {
        id<MTLBuffer> result = [device newBufferWithBytes:bytes length:length
            options:MTLResourceStorageModeShared];
        require(result != nil, "Matter support qualification buffer failed");
        return result;
    };
    id<MTLBuffer> contacts = buffer(&contact, sizeof(contact));
    id<MTLBuffer> bodies = buffer(&body, sizeof(body));
    id<MTLBuffer> generalizedBuffer = buffer(&generalized, sizeof(generalized));
    id<MTLBuffer> jacobians = buffer(jacobian.data(), sizeof(jacobian));
    id<MTLBuffer> acceptedHistory = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> candidateHistory = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> checkpointHistory = buffer(&zero4, sizeof(zero4));
    const NMHumanSupportConsequenceGPU zeroConsequence{};
    id<MTLBuffer> acceptedConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));
    id<MTLBuffer> candidateConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));
    id<MTLBuffer> checkpointConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));
    const NMContactSampleGPU zeroSample{};
    id<MTLBuffer> samples = buffer(&zeroSample, sizeof(zeroSample));
    id<MTLBuffer> successStatus = buffer(&success, sizeof(success));
    id<MTLBuffer> failureStatus = buffer(&failure, sizeof(failure));
    id<MTLBuffer> residual = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> directionBuffer = buffer(&direction, sizeof(direction));
    id<MTLBuffer> work = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> fgmresState = buffer(&fgmres, sizeof(fgmres));
    id<MTLBuffer> committedHistory = buffer(&zero4, sizeof(zero4));
    id<MTLBuffer> committedConsequence =
        buffer(&zeroConsequence, sizeof(zeroConsequence));

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(queue != nil && command != nil,
            "Matter support qualification command failed");
    const auto encodeOne = [&](id<MTLComputePipelineState> state,
                               const auto& bind) {
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        require(encoder != nil, "Matter support qualification encoder failed");
        [encoder setComputePipelineState:state];
        bind(encoder);
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
    };
    const std::uint32_t count = 1u;
    encodeOne(checkpointPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&count length:sizeof(count) atIndex:1u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:2u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:3u];
        [encoder setBuffer:checkpointHistory offset:0u atIndex:4u];
        [encoder setBuffer:acceptedConsequence offset:0u atIndex:5u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:6u];
        [encoder setBuffer:checkpointConsequence offset:0u atIndex:7u];
    });
    encodeOne(evaluatePipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:contacts offset:0u atIndex:2u];
        [encoder setBuffer:bodies offset:0u atIndex:3u];
        [encoder setBuffer:generalizedBuffer offset:0u atIndex:4u];
        [encoder setBuffer:jacobians offset:0u atIndex:5u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:6u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:7u];
        [encoder setBuffer:samples offset:0u atIndex:8u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:9u];
        [encoder setBuffer:successStatus offset:0u atIndex:10u];
    });
    encodeOne(residualPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:samples offset:0u atIndex:2u];
        [encoder setBuffer:jacobians offset:0u atIndex:3u];
        [encoder setBuffer:residual offset:0u atIndex:4u];
    });
    encodeOne(operatorPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&support length:sizeof(support) atIndex:1u];
        [encoder setBuffer:directionBuffer offset:0u atIndex:2u];
        [encoder setBuffer:work offset:0u atIndex:3u];
        [encoder setBuffer:samples offset:0u atIndex:4u];
        [encoder setBuffer:jacobians offset:0u atIndex:5u];
        [encoder setBuffer:fgmresState offset:0u atIndex:6u];
    });
    encodeOne(commitPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&count length:sizeof(count) atIndex:1u];
        [encoder setBuffer:successStatus offset:0u atIndex:2u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:3u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:4u];
        [encoder setBuffer:acceptedConsequence offset:0u atIndex:5u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:6u];
    });
    id<MTLBlitCommandEncoder> snapshot = [command blitCommandEncoder];
    require(snapshot != nil, "Matter support qualification snapshot failed");
    [snapshot copyFromBuffer:acceptedHistory sourceOffset:0u
        toBuffer:committedHistory destinationOffset:0u size:sizeof(nm_float4)];
    [snapshot copyFromBuffer:acceptedConsequence sourceOffset:0u
        toBuffer:committedConsequence destinationOffset:0u
        size:sizeof(NMHumanSupportConsequenceGPU)];
    [snapshot endEncoding];
    encodeOne(rollbackPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&count length:sizeof(count) atIndex:1u];
        [encoder setBuffer:failureStatus offset:0u atIndex:2u];
        [encoder setBuffer:acceptedHistory offset:0u atIndex:3u];
        [encoder setBuffer:candidateHistory offset:0u atIndex:4u];
        [encoder setBuffer:checkpointHistory offset:0u atIndex:5u];
        [encoder setBuffer:acceptedConsequence offset:0u atIndex:6u];
        [encoder setBuffer:candidateConsequence offset:0u atIndex:7u];
        [encoder setBuffer:checkpointConsequence offset:0u atIndex:8u];
    });
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "Matter support qualification GPU command failed");

    const auto& committed =
        *static_cast<const NMHumanSupportConsequenceGPU*>(
            committedConsequence.contents);
    const auto& history =
        *static_cast<const nm_float4*>(committedHistory.contents);
    const auto& residualValue =
        *static_cast<const nm_float4*>(residual.contents);
    const auto& operatorValue =
        *static_cast<const nm_float4*>(work.contents);
    const auto& rolledHistory =
        *static_cast<const nm_float4*>(acceptedHistory.contents);
    const auto& rolledConsequence =
        *static_cast<const NMHumanSupportConsequenceGPU*>(
            acceptedConsequence.contents);
    require(committed.identity.x == 0u && committed.identity.y == 7u &&
                committed.identity.w ==
                    NM_HUMAN_SUPPORT_CONSEQUENCE_VERSION &&
                (committed.identity.z & NM_CONTACT_VALID) != 0u &&
                committed.impulseAndNormal.w > 1.0f &&
                history.w == committed.impulseAndNormal.w &&
                residualValue.x > 1.0f && operatorValue.x < -1.0f,
            "Matter support J^T lambda or J^T D J evidence is wrong");
    require(rolledHistory.x == 0.0f && rolledHistory.y == 0.0f &&
                rolledHistory.z == 0.0f && rolledHistory.w == 0.0f &&
                rolledConsequence.identity.x == 0u &&
                rolledConsequence.identity.y == 0u &&
                rolledConsequence.identity.z == 0u &&
                rolledConsequence.identity.w == 0u,
            "Matter support checkpoint rollback did not restore exact bytes");
}

void mixBytes(
    std::uint64_t& hash,
    const void* raw,
    const std::size_t byteCount
) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    for (std::size_t index = 0u; index < byteCount; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

std::uint64_t fullBodySourceFingerprint(
    const std::vector<std::uint8_t>& rigid,
    const std::vector<std::uint8_t>& muscle,
    const std::vector<std::uint8_t>& support
) noexcept {
    constexpr char domain[] = "mrnx.fullbody.source.v1";
    std::uint64_t hash = kFnvOffset;
    mixBytes(hash, domain, sizeof(domain) - 1u);
    for (const auto* payload : {&rigid, &muscle, &support}) {
        mixU64(hash, static_cast<std::uint64_t>(payload->size()));
        mixBytes(hash, payload->data(), payload->size());
    }
    return hash == 0u ? kFnvOffset : hash;
}

std::uint64_t motorOutputFingerprint(
    const MRNumanXBrainMotorOutputHeaderGPU& header,
    const float* excitation
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION);
    mixU32(hash, header.formatVersion);
    mixU32(hash, header.flags);
    mixU64(hash, header.timestampMicroseconds);
    mixU64(hash, header.brainGeneration);
    mixU64(hash, header.profileFingerprint);
    mixU64(hash, header.protectiveCommandFingerprint);
    mixU32(hash, header.muscleCount);
    mixU32(hash, header.environmentIdentifier);
    mixFloat(hash, header.motorInhibition);
    mixFloat(hash, header.autonomicArousal);
    mixU32(hash, header.actuatorCommandKind);
    mixU32(hash, header.reserved);
    mixFloat(hash, header.outputMinimum);
    mixFloat(hash, header.outputMaximum);
    for (std::uint32_t index = 0u; index < header.muscleCount; ++index) {
        mixFloat(hash, excitation[index]);
    }
    return hash;
}

std::uint64_t readyGateFingerprint(
    const MRNumanXBrainMotorReadyGateGPU& gate
) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&gate);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u; index < 152u; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

std::uint64_t timingFingerprint(
    const mrnx_candidate_timing_v1& timing
) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&timing);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u;
         index < offsetof(mrnx_candidate_timing_v1, timing_fingerprint);
         ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

mrnx_metal_range_v1 range(
    id<MTLBuffer> buffer,
    const std::uint32_t elementType,
    const std::uint32_t elementBytes
) noexcept {
    mrnx_metal_range_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.metal_buffer = (__bridge void*)buffer;
    result.gpu_address = buffer.gpuAddress;
    result.byte_count = buffer.length;
    result.element_type = elementType;
    result.element_byte_count = elementBytes;
    return result;
}

struct Completion {
    std::atomic<std::uint32_t> count{0u};
    std::atomic<std::uint32_t> status{0u};
    mrnx_prepared_v1* prepared = nullptr;
    mrnx_candidate_v1* candidate = nullptr;
    mrnx_root_v1 root{};
};

void settled(
    void* raw,
    mrnx_prepared_v1* prepared,
    mrnx_candidate_v1* candidate,
    const mrnx_completion_v1* completion,
    const mrnx_root_v1* root
) {
    auto* capture = static_cast<Completion*>(raw);
    if (capture == nullptr || completion == nullptr) return;
    if (prepared != nullptr) mrnx_bridge_v1_prepared_retain(prepared);
    if (candidate != nullptr) mrnx_bridge_v1_candidate_retain(candidate);
    capture->prepared = prepared;
    capture->candidate = candidate;
    if (root != nullptr) capture->root = *root;
    capture->status.store(completion->status, std::memory_order_release);
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
}

void waitForCompletion(Completion& completion) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(10);
    while (completion.count.load(std::memory_order_acquire) == 0u &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(
        completion.count.load(std::memory_order_acquire) == 1u,
        "full-body root did not settle exactly once");
}

int run() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        qualifyHumanSupportKKT(device);
        const auto culturePack = metalrobo::makePotterReferenceCulture(
            1000u, 50000u, 2056u);
        metalrobo::CompiledNeuronCulture compiledCulture;
        require(metalrobo::compileNeuronCulture(
                    culturePack, compiledCulture).succeeded(),
                "full-body culture pack did not compile");
        const auto culturePath = std::filesystem::temp_directory_path() /
            ("numanx-fullbody-culture-" + std::to_string(getpid()) +
             ".nculture");
        require(metalrobo::writeCompiledNeuronCulture(
                    compiledCulture, culturePath).succeeded(),
                "full-body culture artifact publication failed");
        const auto culturePathString = culturePath.string();
        mrnx_runtime_config_v2 config{};
        config.abi_version = MRNX_RUNTIME_CONFIG_ABI_V2;
        config.struct_size = sizeof(config);
        config.metal_device = (__bridge void*)device;
        config.rigid_payload_path = MRNX_FULLBODY_RIGID;
        config.muscle_payload_path = MRNX_FULLBODY_MUSCLE;
        config.support_contact_payload_path = MRNX_FULLBODY_SUPPORT_CONTACT;
        config.visual_pack_path = MRNX_FULLBODY_VISUAL_PACK;
        config.vision_profile_path = MRNX_FULLBODY_VISION_PROFILE;
        config.metalrobo_metallib_path = MRNX_METALROBO_METALLIB;
        config.matter_metallib_path = MRNX_MATTER_METALLIB;
        config.matter_material_path = MRNX_MATTER_MATERIAL;
        config.timestep_microseconds = kDurationMicros;
        config.maximum_retained_bytes = 1024ull * 1024ull * 1024ull;
        config.transaction_slot_count = 2u;
        config.culture_pack_path = culturePathString.c_str();
        config.culture_window_ticks = 100u;
        config.culture_current_per_newton = 4.0f;
        auto mismatchedSupport = config;
        mismatchedSupport.support_contact_payload_path = MRNX_FULLBODY_MUSCLE;
        mrnx_runtime_info_v1 mismatchedInfo{};
        mismatchedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
        mismatchedInfo.struct_size = sizeof(mismatchedInfo);
        mrnx_runtime_v1* mismatchedRuntime =
            mrnx_bridge_v1_runtime_create_v2(
                &mismatchedSupport, &mismatchedInfo);
        require(
            mismatchedRuntime == nullptr &&
                mismatchedInfo.status == MRNX_RUNTIME_ASSET_FAILURE_V1,
            "mismatched source support-contact authority was admitted");
        mrnx_runtime_info_v1 info{};
        mrnx_runtime_v1* runtime = mrnx_bridge_v1_runtime_create_v2(
            &config, &info);
        if (runtime == nullptr || info.status != MRNX_RUNTIME_READY_V1) {
            std::fprintf(
                stderr, "full-body runtime status=%u\n", info.status);
        }
        require(runtime != nullptr && info.status == MRNX_RUNTIME_READY_V1,
                "full-body runtime creation failed");
        require(info.q_coordinate_count == 129u && info.dof_count == 128u &&
                    info.muscle_count == 416u && info.body_count == 157u &&
                    info.accepted_state_proof_program_fingerprint != 0u &&
                    info.model_source_fingerprint != 0u,
                "full-body runtime provenance is wrong");
        const auto rigidPayload = readPayloadBytes(MRNX_FULLBODY_RIGID);
        const auto musclePayload = readPayloadBytes(MRNX_FULLBODY_MUSCLE);
        const auto supportPayload =
            readPayloadBytes(MRNX_FULLBODY_SUPPORT_CONTACT);
        require(
            info.model_source_fingerprint == fullBodySourceFingerprint(
                rigidPayload, musclePayload, supportPayload),
            "runtime model fingerprint does not bind exact source payloads");
        auto mutatedSupportPayload = supportPayload;
        require(
            mutatedSupportPayload.size() > 56u,
            "support payload is too small for its NHCNT1 header");
        mutatedSupportPayload[56u] ^= 1u;
        require(
            info.model_source_fingerprint != fullBodySourceFingerprint(
                rigidPayload, musclePayload, mutatedSupportPayload),
            "support-contact payload mutation did not change model identity");

        id<MTLBuffer> headerBuffer = [device
            newBufferWithLength:sizeof(MRNumanXBrainMotorOutputHeaderGPU)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> excitationBuffer = [device
            newBufferWithLength:416u * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> autonomicBuffer = [device
            newBufferWithLength:MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> activeBuffer = [device
            newBufferWithLength:
                MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> gateBuffer = [device
            newBufferWithLength:sizeof(MRNumanXBrainMotorReadyGateGPU)
                       options:MTLResourceStorageModeShared];
        id<MTLSharedEvent> readyEvent = [device newSharedEvent];
        require(headerBuffer != nil && excitationBuffer != nil &&
                    autonomicBuffer != nil && activeBuffer != nil &&
                    gateBuffer != nil && readyEvent != nil,
                "fixture Metal resources unavailable");
        auto* excitation = static_cast<float*>(excitationBuffer.contents);
        for (std::uint32_t index = 0u; index < 416u; ++index) {
            excitation[index] = 0.05f +
                0.1f * static_cast<float>(index % 7u) / 6.0f;
        }
        std::memset(autonomicBuffer.contents, 0, autonomicBuffer.length);
        std::memset(activeBuffer.contents, 0, activeBuffer.length);

        mrnx_physical_root_request_v1 request{};
        request.abi_version = MRNX_BRIDGE_ABI_V1;
        request.struct_size = sizeof(request);
        request.root.format_version =
            MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION;
        request.root.environment_identifier = 0u;
        request.root.episode_identifier = 1u;
        request.root.control_step_identifier = 1u;
        request.root.parameter_version_fingerprint = 2u;
        request.root.base_brain_generation = 0u;
        request.root.base_physics_generation = 0u;
        request.root.committed_timestamp_microseconds = kStartMicros;
        request.root.target_timestamp_microseconds =
            kStartMicros + kDurationMicros;
        request.root.shadow_generation = 1u;
        request.root.random_counter_generation = 3u;
        MRNumanXBrainJointTransactionToken nativeRoot{};
        std::memcpy(&nativeRoot, &request.root, sizeof(nativeRoot));
        request.root.transaction_fingerprint =
            metalrobo::metalNumanXBrainJointTransactionFingerprint(nativeRoot);

        request.substep.transaction_fingerprint =
            request.root.transaction_fingerprint;
        request.substep.substep_index = 0u;
        request.substep.attempt_index = 0u;
        request.substep.start_timestamp_microseconds = kStartMicros;
        request.substep.duration_microseconds = kDurationMicros;
        request.substep.candidate_timestamp_microseconds =
            kStartMicros + kDurationMicros;
        request.substep.shadow_generation = request.root.shadow_generation;
        request.substep.random_counter_generation =
            request.root.random_counter_generation;
        MRNumanXBrainJointSubstepToken nativeSubstep{};
        std::memcpy(&nativeSubstep, &request.substep, sizeof(nativeSubstep));
        request.substep.substep_fingerprint =
            metalrobo::metalNumanXBrainJointSubstepFingerprint(nativeSubstep);

        request.candidate.format_version =
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION;
        request.candidate.flags =
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
        request.candidate.transaction_fingerprint =
            request.root.transaction_fingerprint;
        request.candidate.substep_fingerprint =
            request.substep.substep_fingerprint;
        request.candidate.accepted_brain_timestamp_microseconds =
            kStartMicros;
        request.candidate.brain_generation = request.root.shadow_generation;
        request.candidate.motor_profile_fingerprint = 4u;
        request.candidate.motor_output_header_gpu_address =
            headerBuffer.gpuAddress;
        request.candidate.muscle_excitation_gpu_address =
            excitationBuffer.gpuAddress;
        request.candidate.random_counter_generation =
            request.root.random_counter_generation;
        request.candidate.motor_output_header_byte_count =
            headerBuffer.length;
        request.candidate.muscle_excitation_byte_count =
            excitationBuffer.length;
        request.candidate.muscle_count = 416u;
        request.candidate.environment_identifier = 0u;
        request.candidate.autonomic_command_gpu_address =
            autonomicBuffer.gpuAddress;
        request.candidate.autonomic_command_byte_count =
            autonomicBuffer.length;
        request.candidate.autonomic_command_count = 1u;
        request.candidate.active_sensing_command_gpu_address =
            activeBuffer.gpuAddress;
        request.candidate.active_sensing_command_byte_count =
            activeBuffer.length;
        request.candidate.active_sensing_command_count = 1u;
        request.candidate.actuator_command_kind =
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
        request.candidate.species_template_fingerprint = 5u;
        request.candidate.compiled_species_template_fingerprint = 6u;
        MRNumanXBrainMotorCandidate nativeCandidate{};
        std::memcpy(
            &nativeCandidate, &request.candidate, sizeof(nativeCandidate));
        request.candidate.candidate_fingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                nativeCandidate);

        auto* header = static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
            headerBuffer.contents);
        *header = {};
        header->formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION;
        header->flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
        header->timestampMicroseconds = kStartMicros;
        header->brainGeneration = request.root.shadow_generation;
        header->profileFingerprint =
            request.candidate.motor_profile_fingerprint;
        header->protectiveCommandFingerprint = 7u;
        header->muscleCount = 416u;
        header->environmentIdentifier = 0u;
        header->motorInhibition = 0.1f;
        header->autonomicArousal = 0.2f;
        header->actuatorCommandKind =
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
        header->outputMinimum = 0.0f;
        header->outputMaximum = 1.0f;
        header->outputFingerprint = motorOutputFingerprint(
            *header, excitation);

        auto* gate = static_cast<MRNumanXBrainMotorReadyGateGPU*>(
            gateBuffer.contents);
        *gate = {};
        gate->abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION;
        gate->structBytes = sizeof(*gate);
        gate->status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
        gate->environment = 0u;
        gate->substepIndex = 0u;
        gate->attemptIndex = 0u;
        gate->muscleCount = 416u;
        gate->actuatorCommandKind =
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
        gate->controlStep = request.root.control_step_identifier;
        gate->transactionFingerprint = request.root.transaction_fingerprint;
        gate->substepFingerprint = request.substep.substep_fingerprint;
        gate->candidateFingerprint = request.candidate.candidate_fingerprint;
        gate->motorOutputFingerprint = header->outputFingerprint;
        gate->motorProfileFingerprint =
            request.candidate.motor_profile_fingerprint;
        gate->brainGeneration = request.root.shadow_generation;
        gate->acceptedBrainTimestampMicroseconds = kStartMicros;
        gate->randomCounterGeneration =
            request.root.random_counter_generation;
        gate->speciesTemplateFingerprint =
            request.candidate.species_template_fingerprint;
        gate->compiledSpeciesTemplateFingerprint =
            request.candidate.compiled_species_template_fingerprint;
        gate->brainProgramFingerprint = 8u;
        gate->fastProgramFingerprint = 9u;
        gate->decisionGateFingerprint = 10u;
        gate->gateFingerprint = readyGateFingerprint(*gate);

        request.motor_header = range(
            headerBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.muscle_excitation = range(
            excitationBuffer, MRNX_ELEMENT_FLOAT32_V1, sizeof(float));
        request.autonomic_command = range(
            autonomicBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.active_sensing_command = range(
            activeBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.motor_ready_gate = range(
            gateBuffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
        request.motor_ready.abi_version = MRNX_BRIDGE_ABI_V1;
        request.motor_ready.struct_size = sizeof(request.motor_ready);
        request.motor_ready.shared_event = (__bridge void*)readyEvent;
        request.motor_ready.value = 1u;
        request.motor_ready.device_registry_id = device.registryID;

        std::memcpy(&nativeRoot, &request.root, sizeof(nativeRoot));
        std::memcpy(&nativeSubstep, &request.substep, sizeof(nativeSubstep));
        std::memcpy(
            &nativeCandidate, &request.candidate, sizeof(nativeCandidate));
        require(request.root.transaction_fingerprint ==
                    metalrobo::metalNumanXBrainJointTransactionFingerprint(
                        nativeRoot),
                "fixture root fingerprint mismatch");
        require(request.substep.substep_fingerprint ==
                    metalrobo::metalNumanXBrainJointSubstepFingerprint(
                        nativeSubstep),
                "fixture substep fingerprint mismatch");
        require(request.candidate.candidate_fingerprint ==
                    metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                        nativeCandidate),
                "fixture candidate fingerprint mismatch");

        auto overflow = request;
        overflow.root.control_step_identifier =
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) + 1u;
        require(!mrnx_bridge_v1_runtime_begin_physical_root(
                    runtime, &overflow, nullptr, &settled),
                "UInt64 control-step overflow was admitted");
        Completion completion{};
        if (!mrnx_bridge_v1_runtime_begin_physical_root(
                runtime, &request, &completion, &settled)) {
            mrnx_runtime_info_v1 failedInfo{};
            failedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            failedInfo.struct_size = sizeof(failedInfo);
            (void)mrnx_bridge_v1_runtime_copy_info(runtime, &failedInfo);
            throw std::runtime_error(
                "full-body physical root was not armed status=" +
                std::to_string(failedInfo.status) + " stage=" +
                std::to_string(failedInfo.request_failure_stage));
        }
        require(completion.count.load(std::memory_order_acquire) == 0u,
                "physical root ignored the unsignaled motor-ready event");
        readyEvent.signaledValue = 1u;
        waitForCompletion(completion);
        if (completion.status.load(std::memory_order_acquire) !=
                MRNX_COMPLETION_READY_V1 ||
            completion.prepared == nullptr || completion.candidate == nullptr ||
            completion.root.q_coordinate_count != 129u ||
            completion.root.dof_count != 128u) {
            mrnx_runtime_info_v1 failedInfo{};
            failedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            failedInfo.struct_size = sizeof(failedInfo);
            (void)mrnx_bridge_v1_runtime_copy_info(runtime, &failedInfo);
            throw std::runtime_error(
                "real full-body physical root did not prepare status=" +
                std::to_string(completion.status.load(
                    std::memory_order_acquire)) + " stage=" +
                std::to_string(failedInfo.request_failure_stage) +
                " prepared=" +
                std::to_string(completion.prepared != nullptr) +
                " candidate=" +
                std::to_string(completion.candidate != nullptr) +
                " nq=" +
                std::to_string(completion.root.q_coordinate_count) +
                " nv=" + std::to_string(completion.root.dof_count));
        }
        mrnx_wire_lease_v1 physicalGate{};
        physicalGate.abi_version = MRNX_BRIDGE_ABI_V1;
        physicalGate.struct_size = sizeof(physicalGate);
        require(mrnx_bridge_v1_prepared_copy_physical_gate(
                    completion.prepared, &physicalGate) &&
                    physicalGate.record.byte_count == 64u &&
                    physicalGate.ready.value != 0u,
                "prepared physical gate is unavailable");
        mrnx_culture_prepared_view_v1 culturePrepared{};
        culturePrepared.abi_version = MRNX_CULTURE_PREPARED_VIEW_ABI_V1;
        culturePrepared.struct_size = sizeof(culturePrepared);
        require(mrnx_bridge_v1_prepared_copy_culture_view(
                    completion.prepared, &culturePrepared) &&
                    culturePrepared.culture_fingerprint ==
                        compiledCulture.fingerprint() &&
                    culturePrepared.accepted_generation == 0u &&
                    culturePrepared.prepared_generation == 1u &&
                    culturePrepared.source_root_fingerprint ==
                        completion.root.transaction_fingerprint &&
                    culturePrepared.receipt_fingerprint != 0u &&
                    culturePrepared.ready.value != 0u,
                "prepared culture receipt is unavailable or stale");
        mrnx_candidate_view_v1 sensor{};
        sensor.abi_version = MRNX_BRIDGE_ABI_V1;
        sensor.struct_size = sizeof(sensor);
        mrnx_candidate_channel_v1 proprioception{};
        proprioception.abi_version = MRNX_BRIDGE_ABI_V1;
        proprioception.struct_size = sizeof(proprioception);
        mrnx_candidate_channel_v1 interoception{};
        interoception.abi_version = MRNX_BRIDGE_ABI_V1;
        interoception.struct_size = sizeof(interoception);
        mrnx_candidate_channel_v1 supplemental[5]{};
        for (auto& channel : supplemental) {
            channel.abi_version = MRNX_BRIDGE_ABI_V1;
            channel.struct_size = sizeof(channel);
        }
        mrnx_candidate_timing_v1 timing{};
        timing.abi_version = MRNX_BRIDGE_ABI_V1;
        timing.struct_size = sizeof(timing);
        require(mrnx_bridge_v1_candidate_copy_view(
                    completion.candidate, &sensor) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 0u, &proprioception) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 1u, &interoception) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 2u, &supplemental[0]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 3u, &supplemental[1]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 4u, &supplemental[2]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 5u, &supplemental[3]) &&
                    mrnx_bridge_v1_candidate_copy_channel(
                        completion.candidate, 6u, &supplemental[4]) &&
                    mrnx_bridge_v1_candidate_copy_timing(
                        completion.candidate, &timing) &&
                    sensor.channel_count == 7u &&
                    sensor.accepted_brain_generation == 1u &&
                    sensor.key.sensor_generation == 1u &&
                    proprioception.modality ==
                        MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1 &&
                    proprioception.receptor_count == 416u &&
                    proprioception.feature_dimension == 10u &&
                    proprioception.receptor_timestamp_microseconds ==
                        kStartMicros &&
                    interoception.modality ==
                        MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1 &&
                    interoception.receptor_count == 416u &&
                    interoception.feature_dimension == 6u &&
                    interoception.receptor_timestamp_microseconds ==
                        kStartMicros &&
                    supplemental[0].modality ==
                        MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1 &&
                    supplemental[0].receptor_count == 128u &&
                    supplemental[0].feature_dimension == 7u &&
                    supplemental[1].modality ==
                        MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1 &&
                    supplemental[1].receptor_count == 1u &&
                    supplemental[1].feature_dimension == 22u &&
                    supplemental[2].modality ==
                        MRNX_CANDIDATE_MODALITY_AUDITION_V1 &&
                    supplemental[2].receptor_count == 24u &&
                    supplemental[2].feature_dimension == 8u &&
                    supplemental[3].modality ==
                        MRNX_CANDIDATE_MODALITY_VISION_V1 &&
                    supplemental[3].receptor_count == 64u * 48u &&
                    supplemental[3].feature_dimension == 8u &&
                    supplemental[4].modality ==
                        MRNX_CANDIDATE_MODALITY_TOUCH_V1 &&
                    supplemental[4].receptor_count == 10u &&
                    supplemental[4].feature_dimension == 7u &&
                    timing.capture_timestamp_microseconds == kStartMicros &&
                    timing.delivery_timestamp_microseconds ==
                        kStartMicros + kDurationMicros &&
                    timing.latency_microseconds == kDurationMicros &&
                    timing.sample_interval_microseconds == kDurationMicros &&
                    timing.timing_fingerprint == timingFingerprint(timing),
                "causal HumanIO candidate is not the exact full-body view");
        __unsafe_unretained id<MTLBuffer> touchValues =
            (__bridge id<MTLBuffer>)supplemental[4].values.metal_buffer;
        __unsafe_unretained id<MTLBuffer> touchValidity =
            (__bridge id<MTLBuffer>)supplemental[4].validity.metal_buffer;
        require(touchValues != nil && touchValidity != nil &&
                    supplemental[4].values.byte_count ==
                        10u * 7u * sizeof(float) &&
                    supplemental[4].validity.byte_count ==
                        10u * sizeof(std::uint32_t),
                "touch consequence buffers have the wrong exact layout");
        id<MTLBuffer> touchReadback = [device
            newBufferWithLength:supplemental[4].values.byte_count
                + supplemental[4].validity.byte_count +
                    sizeof(MRNumanXAcceptedPhysicsStateTokenGPU)
                         options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> readbackQueue = [device newCommandQueue];
        id<MTLCommandBuffer> readbackCommand =
            [readbackQueue commandBuffer];
        id<MTLBlitCommandEncoder> readback =
            [readbackCommand blitCommandEncoder];
        require(touchReadback != nil && readbackQueue != nil &&
                    readbackCommand != nil && readback != nil,
                "touch consequence diagnostic readback allocation failed");
        [readback copyFromBuffer:touchValues
                    sourceOffset:supplemental[4].values.byte_offset
                        toBuffer:touchReadback destinationOffset:0u
                            size:supplemental[4].values.byte_count];
        [readback copyFromBuffer:touchValidity
                    sourceOffset:supplemental[4].validity.byte_offset
                        toBuffer:touchReadback
               destinationOffset:supplemental[4].values.byte_count
                            size:supplemental[4].validity.byte_count];
        __unsafe_unretained id<MTLBuffer> physicalTokenBuffer =
            (__bridge id<MTLBuffer>)physicalGate.record.metal_buffer;
        [readback copyFromBuffer:physicalTokenBuffer
                    sourceOffset:physicalGate.record.byte_offset
                        toBuffer:touchReadback
               destinationOffset:supplemental[4].values.byte_count +
                    supplemental[4].validity.byte_count
                            size:sizeof(
                                MRNumanXAcceptedPhysicsStateTokenGPU)];
        [readback endEncoding];
        [readbackCommand commit];
        [readbackCommand waitUntilCompleted];
        require(readbackCommand.status == MTLCommandBufferStatusCompleted,
                "touch consequence diagnostic readback failed");
        const auto* touch = static_cast<const float*>(touchReadback.contents);
        const auto* validity = reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::uint8_t*>(touchReadback.contents) +
            supplemental[4].values.byte_count);
        const auto* acceptedToken =
            reinterpret_cast<const MRNumanXAcceptedPhysicsStateTokenGPU*>(
                static_cast<const std::uint8_t*>(touchReadback.contents) +
                supplemental[4].values.byte_count +
                supplemental[4].validity.byte_count);
        require(acceptedToken->transactionFingerprint ==
                    completion.root.transaction_fingerprint &&
                    acceptedToken->substepFingerprint != 0u &&
                    acceptedToken->physicsStateFingerprint != 0u &&
                    acceptedToken->physicsGeneration == 1u &&
                    acceptedToken->environmentIdentifier == 0u &&
                    acceptedToken->reserved == 0u &&
                    acceptedToken->tokenFingerprint != 0u,
                "accepted token did not bind the proved Human/Matter state");
        bool observedSupportGeometry = false;
        for (std::uint32_t row = 0u; row < 10u; ++row) {
            const float separation = touch[row * 7u + 3u];
            const float normalForce = touch[row * 7u + 4u];
            const float frictionForce = touch[row * 7u + 5u];
            const float slipSpeed = touch[row * 7u + 6u];
            const bool rowValid = validity[row] ==
                    MR_NUMANX_HUMAN_TOUCH_VALIDITY_ALL &&
                std::isfinite(separation) &&
                std::isfinite(normalForce) && normalForce >= 0.0f &&
                std::isfinite(frictionForce) && frictionForce >= 0.0f &&
                std::isfinite(slipSpeed) && slipSpeed >= 0.0f;
            if (!rowValid) {
                std::fprintf(stderr,
                    "support_row=%u validity=%u separation=%g normal=%g "
                    "friction=%g slip=%g\n", row, validity[row], separation,
                    normalForce, frictionForce, slipSpeed);
            }
            require(rowValid, "Matter support consequence is invalid");
            observedSupportGeometry = observedSupportGeometry ||
                std::abs(separation) < 0.1f;
        }
        require(observedSupportGeometry,
                "NHCNT rows produced no solver-owned support geometry");
        mrnx_aggregate_snapshot_v1 aggregate{};
        aggregate.abi_version = MRNX_BRIDGE_ABI_V1;
        aggregate.struct_size = sizeof(aggregate);
        require(!mrnx_bridge_v1_runtime_copy_aggregate_snapshot(
                    runtime, &aggregate),
                "unpublished full-body root escaped the aggregate reader");
        mrnx_aggregate_snapshot_v2 aggregateV2{};
        aggregateV2.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V2;
        aggregateV2.struct_size = sizeof(aggregateV2);
        mrnx_aggregate_snapshot_v3 aggregateV3{};
        aggregateV3.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V3;
        aggregateV3.struct_size = sizeof(aggregateV3);
        mrnx_aggregate_snapshot_v4 aggregateV4{};
        aggregateV4.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V4;
        aggregateV4.struct_size = sizeof(aggregateV4);
        require(!mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v2(
                    runtime, &aggregateV2) &&
                    !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v3(
                        runtime, &aggregateV3) &&
                    !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v4(
                        runtime, &aggregateV4),
                "unpublished root escaped an extensible aggregate reader");
        require(mrnx_bridge_v1_quarantine_timeout(completion.prepared),
                "prepared-only qualification did not quarantine on timeout");
        mrnx_bridge_v1_candidate_drop(completion.candidate);
        mrnx_bridge_v1_prepared_drop(completion.prepared);
        mrnx_bridge_v1_runtime_drop(runtime);
        std::filesystem::remove(culturePath);
        std::printf(
            "numanx_fullbody_bridge_probe=pass bodies=%u nq=%u nv=%u "
            "muscles=%u motor_wait=ordered physical=prepared "
            "sensor=unpublished timeout=quarantined\n",
            info.body_count, info.q_coordinate_count, info.dof_count,
            info.muscle_count);
        return 0;
    }
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
