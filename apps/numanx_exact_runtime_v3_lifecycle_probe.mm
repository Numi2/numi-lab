#define main numanx_fullbody_bridge_probe_embedded_main
#include "numanx_fullbody_bridge_probe.mm"
#undef main

#include "metalrobo/NumanXExactTransaction.hpp"
#include "metalrobo/mrnx_human_behavior_v1.h"
#include "metalrobo/numanx_human_matter_gpu.h"

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace exact_runtime_probe {

constexpr std::uint64_t kTimestepNanoseconds = 12'500u;
constexpr std::uint64_t kInitialTimestampNanoseconds = 1'000'000u;
constexpr std::uint64_t kBrainProgramFingerprint =
    0x425241494e50524full;
constexpr std::uint64_t kBrainShadowBaseFingerprint =
    0x425241494e534844ull;
constexpr std::uint64_t kFastProgramFingerprint =
    0x4641535450524f47ull;
constexpr std::uint64_t kFastGateBaseFingerprint =
    0x4641535447415445ull;
constexpr std::uint64_t kJointReceiptBaseFingerprint =
    0x4a4f494e54524350ull;
constexpr std::uint64_t kJointCommitBaseFingerprint =
    0x4a4f494e54434d54ull;
constexpr std::size_t kMotorHeaderByteOffset = 256u;
constexpr std::size_t kExcitationByteOffset = 512u;
constexpr std::size_t kAutonomicByteOffset = 768u;
constexpr std::size_t kActiveSensingByteOffset = 1'024u;
constexpr std::size_t kReadyGateByteOffset = 1'280u;

template <typename Capture>
void waitForCapture(
    Capture& capture,
    const char* message,
    const unsigned timeoutSeconds = 30u
) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(timeoutSeconds);
    while (capture.count.load(std::memory_order_acquire) == 0u &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(capture.count.load(std::memory_order_acquire) == 1u, message);
}

std::uint64_t recordFingerprint(const void* raw) noexcept {
    if (raw == nullptr) return 0u;
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u; index < 120u; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

std::uint64_t witnessFingerprint(
    const MRNumanXHumanMatterBrainCommitWitnessGPU& witness
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, witness.magic);
    mixU32(hash, witness.abiVersion);
    mixU32(hash, witness.structBytes);
    mixU32(hash, witness.status);
    mixU32(hash, witness.decision);
    mixU32(hash, witness.environment);
    mixU32(hash, witness.stepIndex);
    mixU32(hash, witness.substepIndex);
    mixU32(hash, witness.transactionSlot);
    mixU32(hash, witness.physicsSubstepCount);
    mixU32(hash, witness.controlStep);
    mixU32(hash, witness.reserved0);
    mixU64(hash, witness.programFingerprint);
    mixU64(hash, witness.transactionFingerprint);
    mixU64(hash, witness.linearizationEpoch);
    mixU64(hash, witness.slotGeneration);
    mixU64(hash, witness.physicsTokenFingerprint);
    mixU64(hash, witness.brainProgramFingerprint);
    mixU64(hash, witness.brainShadowStateFingerprint);
    mixU64(hash, witness.reserved1[0]);
    mixU64(hash, witness.reserved1[1]);
    return hash == 0u ? kFnvOffset : hash;
}

template <typename Record>
Record copyRecord(const mrnx_metal_range_v1& range) {
    static_assert(std::is_trivially_copyable_v<Record>);
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)range.metal_buffer;
    require(buffer != nil && buffer.contents != nullptr &&
                range.byte_offset <= buffer.length &&
                sizeof(Record) <= buffer.length - range.byte_offset &&
                range.byte_count >= sizeof(Record),
            "bridge record range is unavailable");
    Record result{};
    std::memcpy(
        &result,
        static_cast<const std::uint8_t*>(buffer.contents) +
            range.byte_offset,
        sizeof(result));
    return result;
}

template <typename Record>
Record readbackRecord(
    id<MTLDevice> device,
    const mrnx_metal_range_v1& range,
    const mrnx_event_point_v1& ready
) {
    static_assert(std::is_trivially_copyable_v<Record>);
    __unsafe_unretained id<MTLBuffer> source =
        (__bridge id<MTLBuffer>)range.metal_buffer;
    __unsafe_unretained id<MTLSharedEvent> event =
        (__bridge id<MTLSharedEvent>)ready.shared_event;
    require(device != nil && source != nil && event != nil &&
                source.device == device && range.byte_offset <= source.length &&
                sizeof(Record) <= source.length - range.byte_offset &&
                range.byte_count >= sizeof(Record) && ready.value != 0u,
            "GPU-only bridge record range is unavailable");
    id<MTLBuffer> destination = [device
        newBufferWithLength:sizeof(Record)
                   options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(destination != nil && queue != nil && command != nil,
            "GPU-only bridge record readback allocation failed");
    [command encodeWaitForEvent:event value:ready.value];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    require(blit != nil, "GPU-only bridge record blit allocation failed");
    [blit copyFromBuffer:source
            sourceOffset:range.byte_offset
                toBuffer:destination
       destinationOffset:0u
                    size:sizeof(Record)];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(
        command.status == MTLCommandBufferStatusCompleted &&
            command.error == nil && destination.contents != nullptr,
        "GPU-only bridge record readback failed");
    Record result{};
    std::memcpy(&result, destination.contents, sizeof(result));
    return result;
}

template <typename Record>
void storeRecord(const mrnx_metal_range_v1& range, const Record& record) {
    static_assert(std::is_trivially_copyable_v<Record>);
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)range.metal_buffer;
    require(buffer != nil && buffer.contents != nullptr &&
                range.byte_offset <= buffer.length &&
                sizeof(Record) <= buffer.length - range.byte_offset &&
                range.byte_count >= sizeof(Record),
            "bridge record range is not writable");
    std::memcpy(
        static_cast<std::uint8_t*>(buffer.contents) + range.byte_offset,
        &record,
        sizeof(record));
}

struct ExactRequestResources {
    __strong id<MTLBuffer> header = nil;
    __strong id<MTLBuffer> excitation = nil;
    __strong id<MTLBuffer> autonomic = nil;
    __strong id<MTLBuffer> activeSensing = nil;
    __strong id<MTLBuffer> readyGate = nil;
    __strong id<MTLSharedEvent> readyEvent = nil;
    mrnx_physical_root_request_v3 request{};
};

mrnx_metal_range_v1 sliceRange(
    id<MTLBuffer> buffer,
    const std::size_t byteOffset,
    const std::size_t byteCount,
    const std::uint32_t elementType,
    const std::uint32_t elementBytes
) noexcept {
    auto result = range(buffer, elementType, elementBytes);
    result.gpu_address = buffer.gpuAddress + byteOffset;
    result.byte_offset = byteOffset;
    result.byte_count = byteCount;
    return result;
}

bool rangeMatchesSlice(
    const mrnx_metal_range_v1& descriptor,
    id<MTLBuffer> buffer,
    const std::size_t byteOffset,
    const std::size_t byteCount
) noexcept {
    return buffer != nil && byteOffset != 0u &&
        descriptor.metal_buffer == (__bridge void*)buffer &&
        descriptor.byte_offset == byteOffset &&
        descriptor.byte_count == byteCount &&
        descriptor.gpu_address == buffer.gpuAddress + byteOffset &&
        byteOffset <= buffer.length &&
        byteCount <= buffer.length - byteOffset;
}

ExactRequestResources makeRequest(
    id<MTLDevice> device,
    const std::uint64_t controlStep,
    const std::uint64_t baseBrainGeneration,
    const std::uint64_t basePhysicsGeneration,
    const std::uint64_t committedTimestampNanoseconds
) {
    constexpr std::size_t excitationBytes =
        MRNX_FULL_BODY_MUSCLE_COUNT * sizeof(float);
    constexpr std::size_t autonomicBytes =
        MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT;
    constexpr std::size_t activeSensingBytes =
        MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
    ExactRequestResources result;
    result.header = [device
        newBufferWithLength:kMotorHeaderByteOffset +
            sizeof(MRNumanXBrainMotorOutputHeaderGPUV2)
                   options:MTLResourceStorageModeShared];
    result.excitation = [device
        newBufferWithLength:kExcitationByteOffset + excitationBytes
                   options:MTLResourceStorageModeShared];
    result.autonomic = [device
        newBufferWithLength:kAutonomicByteOffset + autonomicBytes
                   options:MTLResourceStorageModeShared];
    result.activeSensing = [device
        newBufferWithLength:kActiveSensingByteOffset + activeSensingBytes
                   options:MTLResourceStorageModeShared];
    result.readyGate = [device
        newBufferWithLength:kReadyGateByteOffset +
            sizeof(MRNumanXBrainMotorReadyGateGPUV2)
                   options:MTLResourceStorageModeShared];
    result.readyEvent = [device newSharedEvent];
    require(result.header != nil && result.excitation != nil &&
                result.autonomic != nil && result.activeSensing != nil &&
                result.readyGate != nil && result.readyEvent != nil,
            "exact request Metal resources are unavailable");

    std::memset(
        result.header.contents, 0xa5,
        static_cast<std::size_t>(result.header.length));
    std::memset(
        result.excitation.contents, 0xa5,
        static_cast<std::size_t>(result.excitation.length));
    std::memset(
        result.autonomic.contents, 0xa5,
        static_cast<std::size_t>(result.autonomic.length));
    std::memset(
        result.activeSensing.contents, 0xa5,
        static_cast<std::size_t>(result.activeSensing.length));
    std::memset(
        result.readyGate.contents, 0xa5,
        static_cast<std::size_t>(result.readyGate.length));

    auto* excitation = reinterpret_cast<float*>(
        static_cast<std::uint8_t*>(result.excitation.contents) +
        kExcitationByteOffset);
    for (std::uint32_t index = 0u;
         index < MRNX_FULL_BODY_MUSCLE_COUNT; ++index) {
        excitation[index] = 0.05f +
            0.1f * static_cast<float>(index % 7u) / 6.0f;
    }
    std::memset(
        static_cast<std::uint8_t*>(result.autonomic.contents) +
            kAutonomicByteOffset,
        0, autonomicBytes);
    std::memset(
        static_cast<std::uint8_t*>(result.activeSensing.contents) +
            kActiveSensingByteOffset,
        0, activeSensingBytes);

    auto& request = result.request;
    request.abi_version = MRNX_PHYSICAL_ROOT_REQUEST_ABI_V3;
    request.struct_size = sizeof(request);
    request.root.format_version = MRNX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
    request.root.environment_identifier = 0u;
    request.root.episode_identifier = 1u;
    request.root.control_step_identifier = controlStep;
    request.root.parameter_version_fingerprint = 2u;
    request.root.base_brain_generation = baseBrainGeneration;
    request.root.base_physics_generation = basePhysicsGeneration;
    request.root.committed_timestamp_nanoseconds =
        committedTimestampNanoseconds;
    request.root.target_timestamp_nanoseconds =
        committedTimestampNanoseconds + kTimestepNanoseconds;
    request.root.shadow_generation = baseBrainGeneration + 1u;
    request.root.random_counter_generation = controlStep + 2u;
    request.root.clock_domain =
        MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    request.root.clock_quantum_nanoseconds =
        MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    MRNumanXBrainJointTransactionTokenV2 nativeRoot{};
    std::memcpy(&nativeRoot, &request.root, sizeof(nativeRoot));
    request.root.transaction_fingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(nativeRoot);

    request.substep.transaction_fingerprint =
        request.root.transaction_fingerprint;
    request.substep.substep_index = 0u;
    request.substep.attempt_index = 0u;
    request.substep.start_timestamp_nanoseconds =
        request.root.committed_timestamp_nanoseconds;
    request.substep.duration_nanoseconds = kTimestepNanoseconds;
    request.substep.candidate_timestamp_nanoseconds =
        request.root.target_timestamp_nanoseconds;
    request.substep.shadow_generation = request.root.shadow_generation;
    request.substep.random_counter_generation =
        request.root.random_counter_generation;
    request.substep.clock_domain = request.root.clock_domain;
    request.substep.clock_quantum_nanoseconds =
        request.root.clock_quantum_nanoseconds;
    MRNumanXBrainJointSubstepTokenV2 nativeSubstep{};
    std::memcpy(&nativeSubstep, &request.substep, sizeof(nativeSubstep));
    request.substep.substep_fingerprint =
        metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(nativeSubstep);

    request.candidate.format_version =
        MRNX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
    request.candidate.flags =
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
    request.candidate.transaction_fingerprint =
        request.root.transaction_fingerprint;
    request.candidate.substep_fingerprint =
        request.substep.substep_fingerprint;
    request.candidate.accepted_brain_timestamp_nanoseconds =
        committedTimestampNanoseconds;
    request.candidate.brain_generation = request.root.shadow_generation;
    request.candidate.motor_profile_fingerprint = 4u;
    request.candidate.motor_output_header_gpu_address =
        result.header.gpuAddress + kMotorHeaderByteOffset;
    request.candidate.muscle_excitation_gpu_address =
        result.excitation.gpuAddress + kExcitationByteOffset;
    request.candidate.random_counter_generation =
        request.root.random_counter_generation;
    request.candidate.motor_output_header_byte_count =
        sizeof(MRNumanXBrainMotorOutputHeaderGPUV2);
    request.candidate.muscle_excitation_byte_count =
        excitationBytes;
    request.candidate.muscle_count = MRNX_FULL_BODY_MUSCLE_COUNT;
    request.candidate.environment_identifier = 0u;
    request.candidate.autonomic_command_gpu_address =
        result.autonomic.gpuAddress + kAutonomicByteOffset;
    request.candidate.autonomic_command_byte_count =
        autonomicBytes;
    request.candidate.autonomic_command_count = 1u;
    request.candidate.active_sensing_command_gpu_address =
        result.activeSensing.gpuAddress + kActiveSensingByteOffset;
    request.candidate.active_sensing_command_byte_count =
        activeSensingBytes;
    request.candidate.active_sensing_command_count = 1u;
    request.candidate.actuator_command_kind =
        MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
    request.candidate.clock_domain =
        MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    request.candidate.species_template_fingerprint = 5u;
    request.candidate.compiled_species_template_fingerprint = 6u;
    MRNumanXBrainMotorCandidateV2 nativeCandidate{};
    std::memcpy(
        &nativeCandidate, &request.candidate, sizeof(nativeCandidate));
    request.candidate.candidate_fingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
            nativeCandidate);

    auto* header = static_cast<MRNumanXBrainMotorOutputHeaderGPUV2*>(
        static_cast<void*>(
            static_cast<std::uint8_t*>(result.header.contents) +
            kMotorHeaderByteOffset));
    *header = {};
    header->formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
    header->flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
    header->timestampNanoseconds = committedTimestampNanoseconds;
    header->brainGeneration = request.root.shadow_generation;
    header->profileFingerprint =
        request.candidate.motor_profile_fingerprint;
    header->protectiveCommandFingerprint = 7u;
    header->muscleCount = MRNX_FULL_BODY_MUSCLE_COUNT;
    header->environmentIdentifier = 0u;
    header->motorInhibition = 0.1f;
    header->autonomicArousal = 0.2f;
    header->actuatorCommandKind =
        MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
    header->clockDomain =
        MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    header->outputMinimum = 0.0f;
    header->outputMaximum = 1.0f;
    header->outputFingerprint =
        metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
            *header, excitation, MRNX_FULL_BODY_MUSCLE_COUNT);

    auto* gate = static_cast<MRNumanXBrainMotorReadyGateGPUV2*>(
        static_cast<void*>(
            static_cast<std::uint8_t*>(result.readyGate.contents) +
            kReadyGateByteOffset));
    *gate = {};
    gate->abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION_V2;
    gate->structBytes = sizeof(*gate);
    gate->status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
    gate->environment = 0u;
    gate->substepIndex = 0u;
    gate->attemptIndex = 0u;
    gate->muscleCount = MRNX_FULL_BODY_MUSCLE_COUNT;
    gate->actuatorCommandKind =
        MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
    gate->controlStep = controlStep;
    gate->transactionFingerprint = request.root.transaction_fingerprint;
    gate->substepFingerprint = request.substep.substep_fingerprint;
    gate->candidateFingerprint = request.candidate.candidate_fingerprint;
    gate->motorOutputFingerprint = header->outputFingerprint;
    gate->motorProfileFingerprint =
        request.candidate.motor_profile_fingerprint;
    gate->brainGeneration = request.root.shadow_generation;
    gate->acceptedBrainTimestampNanoseconds = committedTimestampNanoseconds;
    gate->randomCounterGeneration =
        request.root.random_counter_generation;
    gate->speciesTemplateFingerprint =
        request.candidate.species_template_fingerprint;
    gate->compiledSpeciesTemplateFingerprint =
        request.candidate.compiled_species_template_fingerprint;
    gate->brainProgramFingerprint = kBrainProgramFingerprint;
    gate->fastProgramFingerprint = kFastProgramFingerprint;
    gate->decisionGateFingerprint = 10u;
    gate->clockDomain =
        MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    gate->clockQuantumNanoseconds =
        MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    gate->gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(*gate);

    request.motor_header = sliceRange(
        result.header, kMotorHeaderByteOffset, sizeof(*header),
        MRNX_ELEMENT_BRAIN_MOTOR_OUTPUT_HEADER_V2,
        sizeof(MRNumanXBrainMotorOutputHeaderGPUV2));
    request.muscle_excitation = sliceRange(
        result.excitation, kExcitationByteOffset, excitationBytes,
        MRNX_ELEMENT_FLOAT32_V1, sizeof(float));
    request.autonomic_command = sliceRange(
        result.autonomic, kAutonomicByteOffset, autonomicBytes,
        MRNX_ELEMENT_RAW_BYTES_V1, 1u);
    request.active_sensing_command = sliceRange(
        result.activeSensing, kActiveSensingByteOffset, activeSensingBytes,
        MRNX_ELEMENT_RAW_BYTES_V1, 1u);
    request.motor_ready_gate = sliceRange(
        result.readyGate, kReadyGateByteOffset, sizeof(*gate),
        MRNX_ELEMENT_BRAIN_MOTOR_READY_GATE_V2,
        sizeof(MRNumanXBrainMotorReadyGateGPUV2));
    request.motor_ready.abi_version = MRNX_BRIDGE_ABI_V1;
    request.motor_ready.struct_size = sizeof(request.motor_ready);
    request.motor_ready.shared_event = (__bridge void*)result.readyEvent;
    request.motor_ready.value = 1u;
    request.motor_ready.device_registry_id = device.registryID;

    require(
        rangeMatchesSlice(
            request.motor_header, result.header,
            kMotorHeaderByteOffset, sizeof(*header)) &&
            rangeMatchesSlice(
                request.muscle_excitation, result.excitation,
                kExcitationByteOffset, excitationBytes) &&
            rangeMatchesSlice(
                request.autonomic_command, result.autonomic,
                kAutonomicByteOffset, autonomicBytes) &&
            rangeMatchesSlice(
                request.active_sensing_command, result.activeSensing,
                kActiveSensingByteOffset, activeSensingBytes) &&
            rangeMatchesSlice(
                request.motor_ready_gate, result.readyGate,
                kReadyGateByteOffset, sizeof(*gate)) &&
            request.candidate.motor_output_header_gpu_address ==
                request.motor_header.gpu_address &&
            request.candidate.muscle_excitation_gpu_address ==
                request.muscle_excitation.gpu_address &&
            request.candidate.autonomic_command_gpu_address ==
                request.autonomic_command.gpu_address &&
            request.candidate.active_sensing_command_gpu_address ==
                request.active_sensing_command.gpu_address,
        "exact request did not preserve all five nonzero Metal slices");

    require(
        metalrobo::metalNumanXBrainJointTransactionV2Valid(nativeRoot) ==
            false,
        "unfingerprinted exact root unexpectedly validated");
    std::memcpy(&nativeRoot, &request.root, sizeof(nativeRoot));
    std::memcpy(&nativeSubstep, &request.substep, sizeof(nativeSubstep));
    std::memcpy(
        &nativeCandidate, &request.candidate, sizeof(nativeCandidate));
    require(
        metalrobo::metalNumanXBrainJointTransactionV2Valid(nativeRoot) &&
            metalrobo::metalNumanXBrainJointSubstepV2Valid(
                nativeRoot, nativeSubstep) &&
            metalrobo::metalNumanXBrainMotorCandidateV2Valid(
                nativeRoot, nativeSubstep, nativeCandidate) &&
            metalrobo::metalNumanXBrainMotorOutputV2Valid(
                nativeCandidate, *header, excitation,
                MRNX_FULL_BODY_MUSCLE_COUNT) &&
            metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                nativeRoot, nativeSubstep, nativeCandidate, *header, *gate),
        "exact request fixture failed canonical CPU validation");
    return result;
}

mrnx_wire_lease_v1 makeWire(
    const mrnx_root_v1& root,
    id<MTLBuffer> buffer,
    id<MTLSharedEvent> event,
    const std::uint64_t value
) noexcept {
    mrnx_wire_lease_v1 wire{};
    wire.abi_version = MRNX_BRIDGE_ABI_V1;
    wire.struct_size = sizeof(wire);
    wire.root = root;
    wire.record = range(buffer, MRNX_ELEMENT_RAW_BYTES_V1, 1u);
    wire.ready.abi_version = MRNX_BRIDGE_ABI_V1;
    wire.ready.struct_size = sizeof(wire.ready);
    wire.ready.shared_event = (__bridge void*)event;
    wire.ready.value = value;
    wire.ready.device_registry_id = buffer.device.registryID;
    return wire;
}

struct ProposalCapture {
    std::atomic<std::uint32_t> count{0u};
    std::atomic<std::uint32_t> status{0u};
    mrnx_proposal_view_v1 view{};
};

void proposalSettled(
    void* raw,
    const mrnx_completion_v1* completion,
    const mrnx_proposal_view_v1* proposal
) {
    auto* capture = static_cast<ProposalCapture*>(raw);
    if (capture == nullptr || completion == nullptr || proposal == nullptr)
        return;
    capture->view = *proposal;
    capture->status.store(completion->status, std::memory_order_release);
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
}

struct ApplyCapture {
    std::atomic<std::uint32_t> count{0u};
    std::atomic<std::uint32_t> status{0u};
    mrnx_applied_view_v1 view{};
};

void applySettled(
    void* raw,
    const mrnx_completion_v1* completion,
    const mrnx_applied_view_v1* applied
) {
    auto* capture = static_cast<ApplyCapture*>(raw);
    if (capture == nullptr || completion == nullptr || applied == nullptr)
        return;
    capture->view = *applied;
    capture->status.store(completion->status, std::memory_order_release);
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
}

struct LatchCapture {
    std::uint64_t expectedGeneration = 0u;
    std::atomic<std::uint32_t> count{0u};
};

bool latchBrainGeneration(
    void* raw,
    const std::uint64_t publishingBrainGeneration
) {
    auto* capture = static_cast<LatchCapture*>(raw);
    if (capture == nullptr ||
        publishingBrainGeneration != capture->expectedGeneration) {
        return false;
    }
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
    return true;
}

struct BlockingLatchCapture {
    std::uint64_t expectedGeneration = 0u;
    std::atomic<std::uint32_t> count{0u};
    std::mutex mutex;
    std::condition_variable condition;
    bool release = false;
};

bool blockingLatchBrainGeneration(
    void* raw,
    const std::uint64_t publishingBrainGeneration
) {
    auto* capture = static_cast<BlockingLatchCapture*>(raw);
    if (capture == nullptr ||
        publishingBrainGeneration != capture->expectedGeneration) {
        return false;
    }
    std::unique_lock lock(capture->mutex);
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
    capture->condition.notify_all();
    capture->condition.wait(lock, [&] { return capture->release; });
    return true;
}

MRNumanXHumanMatterBrainCommitWitnessGPU makeWitness(
    const mrnx_root_v1& root,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const std::uint64_t controlStep
) noexcept {
    MRNumanXHumanMatterBrainCommitWitnessGPU witness{};
    witness.magic = MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_MAGIC;
    witness.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_ABI_VERSION;
    witness.structBytes =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_BYTES;
    witness.status =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_COMPLETE;
    witness.decision = MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    witness.environment = root.environment;
    witness.stepIndex = root.step_index;
    witness.substepIndex = root.substep_index;
    witness.transactionSlot = root.transaction_slot;
    witness.physicsSubstepCount = root.physics_substep_count;
    witness.controlStep = root.control_step;
    witness.programFingerprint = root.program_fingerprint;
    witness.transactionFingerprint = root.transaction_fingerprint;
    witness.linearizationEpoch = root.linearization_epoch;
    witness.slotGeneration = root.slot_generation;
    witness.physicsTokenFingerprint = token.tokenFingerprint;
    witness.brainProgramFingerprint = kBrainProgramFingerprint;
    witness.brainShadowStateFingerprint =
        kBrainShadowBaseFingerprint + controlStep;
    witness.witnessFingerprint = witnessFingerprint(witness);
    return witness;
}

MRNumanXHumanMatterBrainCommitPreflightGPU makePreflight(
    const mrnx_root_v1& root,
    const mrnx_physical_root_request_v3& request,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const std::uint64_t controlStep
) noexcept {
    MRNumanXHumanMatterBrainCommitPreflightGPU preflight{};
    preflight.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION;
    preflight.structBytes = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES;
    preflight.status = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS;
    preflight.environment = root.environment;
    preflight.controlStep = root.control_step;
    preflight.substepIndex = root.substep_index;
    preflight.physicsSubstepCount = root.physics_substep_count;
    preflight.transactionSlot = root.transaction_slot;
    preflight.ownerProgramFingerprint = root.program_fingerprint;
    preflight.transactionFingerprint = root.transaction_fingerprint;
    preflight.linearizationEpoch = root.linearization_epoch;
    preflight.slotGeneration = root.slot_generation;
    preflight.substepFingerprint = request.substep.substep_fingerprint;
    preflight.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    preflight.fastTargetGeneration = proposal.brainProgramFingerprint + 1u;
    preflight.cognitiveTargetGeneration =
        proposal.brainProgramFingerprint + 2u;
    preflight.jointReceiptFingerprint =
        kJointReceiptBaseFingerprint + controlStep;
    preflight.fastProgramFingerprint = kFastProgramFingerprint;
    preflight.brainProgramFingerprint = proposal.brainProgramFingerprint;
    preflight.preflightFingerprint = recordFingerprint(&preflight);
    return preflight;
}

MRNumanXHumanMatterBrainAckGPU makeAck(
    const mrnx_root_v1& root,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterBrainCommitPreflightGPU& preflight,
    const std::uint64_t controlStep
) noexcept {
    MRNumanXHumanMatterBrainAckGPU ack{};
    ack.abiVersion = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION;
    ack.status = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT;
    ack.decision = MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    ack.code = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS;
    ack.programFingerprint = root.program_fingerprint;
    ack.transactionFingerprint = root.transaction_fingerprint;
    ack.linearizationEpoch = root.linearization_epoch;
    ack.slotGeneration = root.slot_generation;
    ack.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    ack.proposalFingerprint = proposal.proposalFingerprint;
    ack.preflightFingerprint = preflight.preflightFingerprint;
    ack.fastGateFingerprint = kFastGateBaseFingerprint + controlStep;
    ack.brainWitnessFingerprint = proposal.brainWitnessFingerprint;
    ack.brainProgramFingerprint = proposal.brainProgramFingerprint;
    ack.environment = root.environment;
    ack.stepIndex = root.step_index;
    ack.substepIndex = root.substep_index;
    ack.transactionSlot = root.transaction_slot;
    ack.physicsSubstepCount = root.physics_substep_count;
    ack.controlStep = root.control_step;
    ack.ackFingerprint = recordFingerprint(&ack);
    return ack;
}

bool snapshotInternallyConsistent(
    const mrnx_aggregate_snapshot_v5& snapshot
) noexcept {
    return snapshot.abi_version == MRNX_AGGREGATE_SNAPSHOT_ABI_V5 &&
        snapshot.struct_size == sizeof(snapshot) &&
        snapshot.publication_epoch != 0u &&
        snapshot.brain_generation == snapshot.publication.brain_generation &&
        snapshot.physics_generation != 0u &&
        snapshot.sensor_generation == snapshot.sensor.key.sensor_generation &&
        snapshot.root.transaction_fingerprint ==
            snapshot.publication.transaction_fingerprint &&
        snapshot.sensor.candidate_publication_fingerprint ==
            snapshot.publication.candidate_publication_fingerprint &&
        snapshot.sensor_packet.candidate_publication_fingerprint ==
            snapshot.publication.candidate_publication_fingerprint &&
        snapshot.sensor.candidate_publication_fingerprint ==
            snapshot.sensor_packet.candidate_publication_fingerprint &&
        snapshot.sensor_packet.accepted_physics_token_fingerprint ==
            snapshot.publication.accepted_physics_token_fingerprint &&
        snapshot.timing.delivery_timestamp_nanoseconds ==
            snapshot.publication.committed_timestamp_nanoseconds &&
        snapshot.publication.publication_fingerprint ==
            metalrobo::metalNumanXExactPublicationV2Fingerprint(
                snapshot.publication) &&
        snapshot.sensor.channel_count == snapshot.sensor_packet.channel_count &&
        snapshot.sensor.channel_count > 0u &&
        snapshot.sensor.channel_count <= MRNX_MAX_SENSOR_CHANNELS_V2;
}

bool legacyAggregateReadersRejectExact(mrnx_runtime_v1* runtime) noexcept {
    mrnx_aggregate_snapshot_v1 v1{};
    v1.abi_version = MRNX_BRIDGE_ABI_V1;
    v1.struct_size = sizeof(v1);
    mrnx_aggregate_snapshot_v2 v2{};
    v2.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V2;
    v2.struct_size = sizeof(v2);
    mrnx_aggregate_snapshot_v3 v3{};
    v3.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V3;
    v3.struct_size = sizeof(v3);
    mrnx_aggregate_snapshot_v4 v4{};
    v4.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V4;
    v4.struct_size = sizeof(v4);
    return !mrnx_bridge_v1_runtime_copy_aggregate_snapshot(runtime, &v1) &&
        !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v2(runtime, &v2) &&
        !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v3(runtime, &v3) &&
        !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v4(runtime, &v4);
}

struct RootOutcome {
    mrnx_aggregate_snapshot_v5 aggregate{};
    mrnx_publication_v2 publication{};
};

RootOutcome executeAcceptedRoot(
    mrnx_runtime_v1* runtime,
    id<MTLDevice> device,
    const std::uint64_t controlStep,
    const std::uint64_t baseBrainGeneration,
    const std::uint64_t basePhysicsGeneration,
    const std::uint64_t committedTimestampNanoseconds,
    const bool checkMixedFamilyEntry
) {
    auto resources = makeRequest(
        device, controlStep, baseBrainGeneration, basePhysicsGeneration,
        committedTimestampNanoseconds);
    if (checkMixedFamilyEntry) {
        mrnx_physical_root_request_v1 legacy{};
        legacy.abi_version = MRNX_BRIDGE_ABI_V1;
        legacy.struct_size = sizeof(legacy);
        Completion rejected{};
        require(
            !mrnx_bridge_v1_runtime_begin_physical_root(
                runtime, &legacy, &rejected, &settled) &&
                rejected.count.load(std::memory_order_acquire) == 0u,
            "exact runtime admitted the legacy request family");
        mrnx_physical_root_request_v2 legacyV2{};
        legacyV2.abi_version = MRNX_PHYSICAL_ROOT_REQUEST_ABI_V2;
        legacyV2.struct_size = sizeof(legacyV2);
        require(
            !mrnx_bridge_v1_runtime_begin_physical_root_v2(
                runtime, &legacyV2, &rejected, &settled) &&
                rejected.count.load(std::memory_order_acquire) == 0u,
            "exact runtime admitted the frozen all-v1 request-v2 family");
    }

    Completion physical{};
    const bool began = mrnx_bridge_v1_runtime_begin_physical_root_v3(
        runtime, &resources.request, &physical, &settled);
    if (!began) {
        mrnx_runtime_info_v1 failedInfo{};
        failedInfo.abi_version = MRNX_BRIDGE_ABI_V1;
        failedInfo.struct_size = sizeof(failedInfo);
        (void)mrnx_bridge_v1_runtime_copy_info(runtime, &failedInfo);
        std::fprintf(
            stderr,
            "exact request-v3 arm failed status=%u stage=%u\n",
            failedInfo.status, failedInfo.request_failure_stage);
    }
    require(began, "exact request-v3 root was not armed");
    require(physical.count.load(std::memory_order_acquire) == 0u,
            "exact root ignored its unsignaled Brain ready event");
    resources.readyEvent.signaledValue = 1u;
    waitForCompletion(physical, 60u);
    require(
        physical.status.load(std::memory_order_acquire) ==
                MRNX_COMPLETION_READY_V1 &&
            physical.prepared != nullptr && physical.candidate != nullptr &&
            physical.root.transaction_fingerprint ==
                resources.request.root.transaction_fingerprint &&
            physical.root.control_step == controlStep,
        "exact physical/HumanIO root did not reach READY");

    mrnx_candidate_view_v1 sensor{};
    sensor.abi_version = MRNX_BRIDGE_ABI_V1;
    sensor.struct_size = sizeof(sensor);
    mrnx_candidate_timing_v2 timing{};
    timing.abi_version = MRNX_CANDIDATE_TIMING_ABI_V2;
    timing.struct_size = sizeof(timing);
    mrnx_exact_inbound_authority_v2 authority{};
    authority.abi_version = MRNX_EXACT_INBOUND_AUTHORITY_ABI_V2;
    authority.struct_size = sizeof(authority);
    mrnx_exact_sensor_packet_v2 packet{};
    packet.abi_version = MRNX_EXACT_SENSOR_PACKET_ABI_V2;
    packet.struct_size = sizeof(packet);
    require(
        mrnx_bridge_v1_candidate_copy_view(physical.candidate, &sensor) &&
            mrnx_bridge_v1_candidate_copy_timing_v2(
                physical.candidate, &timing) &&
            mrnx_bridge_v1_candidate_copy_inbound_authority_v2(
                physical.candidate, &authority) &&
            mrnx_bridge_v1_candidate_copy_sensor_packet_v2(
                physical.candidate, &packet) &&
            sensor.channel_count == packet.channel_count &&
            sensor.channel_count == 7u &&
            timing.capture_timestamp_nanoseconds ==
                committedTimestampNanoseconds &&
            timing.delivery_timestamp_nanoseconds ==
                committedTimestampNanoseconds + kTimestepNanoseconds &&
            authority.transaction_fingerprint ==
                resources.request.root.transaction_fingerprint &&
            authority.motor_candidate_fingerprint ==
                resources.request.candidate.candidate_fingerprint &&
            packet.transaction_fingerprint ==
                resources.request.root.transaction_fingerprint &&
            packet.accepted_brain_generation ==
                resources.request.root.shadow_generation &&
            sensor.candidate_publication_fingerprint ==
                packet.candidate_publication_fingerprint,
        "exact candidate receipts are incomplete or stale");
    require(
        metalrobo::metalNumanXExactCandidateTimingV2Valid(timing) &&
            metalrobo::metalNumanXExactInboundAuthorityV2Valid(authority),
        "exact candidate timing/authority fingerprints are invalid");
    const auto* requestHeader =
        reinterpret_cast<const MRNumanXBrainMotorOutputHeaderGPUV2*>(
            static_cast<const std::uint8_t*>(resources.header.contents) +
            resources.request.motor_header.byte_offset);
    const auto* requestGate =
        reinterpret_cast<const MRNumanXBrainMotorReadyGateGPUV2*>(
            static_cast<const std::uint8_t*>(resources.readyGate.contents) +
            resources.request.motor_ready_gate.byte_offset);
    mrnx_brain_motor_output_header_v2 requestOutputWire{};
    std::memcpy(
        &requestOutputWire, requestHeader, sizeof(requestOutputWire));
    mrnx_brain_motor_ready_gate_v2 requestGateWire{};
    std::memcpy(&requestGateWire, requestGate, sizeof(requestGateWire));
    require(
        rangeMatchesSlice(
            resources.request.motor_header, resources.header,
            kMotorHeaderByteOffset, sizeof(*requestHeader)) &&
            rangeMatchesSlice(
                resources.request.muscle_excitation, resources.excitation,
                kExcitationByteOffset,
                MRNX_FULL_BODY_MUSCLE_COUNT * sizeof(float)) &&
            rangeMatchesSlice(
                resources.request.autonomic_command, resources.autonomic,
                kAutonomicByteOffset,
                MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT) &&
            rangeMatchesSlice(
                resources.request.active_sensing_command,
                resources.activeSensing, kActiveSensingByteOffset,
                MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT) &&
            rangeMatchesSlice(
                resources.request.motor_ready_gate, resources.readyGate,
                kReadyGateByteOffset, sizeof(*requestGate)) &&
            metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                authority, resources.request.root,
                resources.request.substep, resources.request.candidate,
                requestOutputWire, requestGateWire) &&
            authority.inbound_authority_fingerprint ==
                metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
                    authority) &&
            packet.inbound_authority_fingerprint ==
                authority.inbound_authority_fingerprint,
        "returned exact authority did not bind the five sliced inputs");

    std::array<mrnx_candidate_channel_v2, MRNX_MAX_SENSOR_CHANNELS_V2>
        channels{};
    std::uint32_t previousModality = 0u;
    for (std::uint32_t index = 0u; index < sensor.channel_count; ++index) {
        channels[index].abi_version = MRNX_CANDIDATE_CHANNEL_ABI_V2;
        channels[index].struct_size = sizeof(channels[index]);
        require(
            mrnx_bridge_v1_candidate_copy_channel_v2(
                physical.candidate, index, &channels[index]) &&
                channels[index].modality > previousModality &&
                channels[index].receptor_timestamp_nanoseconds ==
                    committedTimestampNanoseconds &&
                metalrobo::metalNumanXExactCandidateChannelV2Valid(
                    channels[index], timing),
            "exact candidate channel order or identity is invalid");
        previousModality = channels[index].modality;
    }
    require(
        packet.channel_set_fingerprint ==
            metalrobo::metalNumanXExactCandidateChannelSetV2Fingerprint(
                channels.data(), sensor.channel_count) &&
            packet.inbound_authority_fingerprint ==
                authority.inbound_authority_fingerprint &&
            packet.candidate_publication_fingerprint ==
                metalrobo::metalNumanXExactSensorPacketV2Fingerprint(packet),
        "exact sensor packet does not close over its channels");

    mrnx_candidate_timing_v1 legacyTiming{};
    legacyTiming.abi_version = MRNX_BRIDGE_ABI_V1;
    legacyTiming.struct_size = sizeof(legacyTiming);
    mrnx_candidate_channel_v1 legacyChannel{};
    legacyChannel.abi_version = MRNX_BRIDGE_ABI_V1;
    legacyChannel.struct_size = sizeof(legacyChannel);
    require(
        !mrnx_bridge_v1_candidate_copy_timing(
            physical.candidate, &legacyTiming) &&
            !mrnx_bridge_v1_candidate_copy_channel(
                physical.candidate, 0u, &legacyChannel),
        "exact candidate down-converted through a legacy reader");

    mrnx_wire_lease_v1 physicalGate{};
    physicalGate.abi_version = MRNX_BRIDGE_ABI_V1;
    physicalGate.struct_size = sizeof(physicalGate);
    require(
        mrnx_bridge_v1_prepared_copy_physical_gate(
            physical.prepared, &physicalGate) &&
            physicalGate.record.byte_count ==
                sizeof(MRNumanXAcceptedPhysicsStateTokenGPUV2),
        "exact physical accepted-token gate is unavailable");
    const auto acceptedToken =
        readbackRecord<MRNumanXAcceptedPhysicsStateTokenGPUV2>(
            device, physicalGate.record, physicalGate.ready);
    require(
        acceptedToken.transactionFingerprint ==
                physical.root.transaction_fingerprint &&
            acceptedToken.acceptedTimestampNanoseconds ==
                timing.delivery_timestamp_nanoseconds &&
            acceptedToken.physicsGeneration == basePhysicsGeneration + 1u &&
            acceptedToken.clockDomain ==
                MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
            acceptedToken.clockQuantumNanoseconds ==
                MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
            acceptedToken.tokenFingerprint ==
                packet.accepted_physics_token_fingerprint &&
            acceptedToken.tokenFingerprint ==
                metalrobo::
                    metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(
                        acceptedToken),
        "exact accepted physics token is invalid");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> witnessBuffer = [device
        newBufferWithLength:sizeof(MRNumanXHumanMatterBrainCommitWitnessGPU)
                   options:MTLResourceStorageModeShared];
    id<MTLSharedEvent> witnessReady = [device newSharedEvent];
    require(queue != nil && witnessBuffer != nil && witnessReady != nil,
            "Brain witness resources are unavailable");
    *static_cast<MRNumanXHumanMatterBrainCommitWitnessGPU*>(
        witnessBuffer.contents) =
        makeWitness(physical.root, acceptedToken, controlStep);
    witnessReady.signaledValue = 1u;
    const auto witnessWire = makeWire(
        physical.root, witnessBuffer, witnessReady, 1u);
    id<MTLCommandBuffer> proposalCommand = [queue commandBuffer];
    ProposalCapture proposalCapture{};
    require(
        proposalCommand != nil &&
            mrnx_bridge_v1_submit_proposal(
                physical.prepared, (__bridge void*)proposalCommand,
                &witnessWire, &proposalCapture, &proposalSettled),
        "exact root proposal was not encoded");
    [proposalCommand commit];
    waitForCapture(
        proposalCapture, "exact root proposal did not settle exactly once");
    require(
        proposalCapture.status.load(std::memory_order_acquire) ==
                MRNX_COMPLETION_READY_V1 &&
            proposalCommand.status == MTLCommandBufferStatusCompleted,
        "exact root proposal did not become READY");
    const auto proposal = copyRecord<MRNumanXHumanMatterProposalGPU>(
        proposalCapture.view.proposal);
    if (!(proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
          proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
          proposal.code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS &&
          proposal.transactionFingerprint ==
              physical.root.transaction_fingerprint &&
          proposal.physicsTokenFingerprint ==
              acceptedToken.tokenFingerprint &&
          proposal.candidatePublicationFingerprint ==
              sensor.candidate_publication_fingerprint &&
          proposal.candidatePublicationFingerprint ==
              packet.candidate_publication_fingerprint &&
          proposal.proposalFingerprint == recordFingerprint(&proposal))) {
        std::fprintf(
            stderr,
            "proposal status=%u decision=%u code=%u tx=%016llx/%016llx "
            "token=%016llx/%016llx candidate=%016llx/%016llx "
            "fp=%016llx/%016llx\n",
            proposal.status, proposal.decision, proposal.code,
            static_cast<unsigned long long>(
                proposal.transactionFingerprint),
            static_cast<unsigned long long>(
                physical.root.transaction_fingerprint),
            static_cast<unsigned long long>(
                proposal.physicsTokenFingerprint),
            static_cast<unsigned long long>(acceptedToken.tokenFingerprint),
            static_cast<unsigned long long>(
                proposal.candidatePublicationFingerprint),
            static_cast<unsigned long long>(
                packet.candidate_publication_fingerprint),
            static_cast<unsigned long long>(proposal.proposalFingerprint),
            static_cast<unsigned long long>(recordFingerprint(&proposal)));
    }
    require(
        proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
            proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
            proposal.code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS &&
            proposal.transactionFingerprint ==
                physical.root.transaction_fingerprint &&
            proposal.physicsTokenFingerprint ==
                acceptedToken.tokenFingerprint &&
            proposal.candidatePublicationFingerprint ==
                sensor.candidate_publication_fingerprint &&
            proposal.candidatePublicationFingerprint ==
                packet.candidate_publication_fingerprint &&
            proposal.proposalFingerprint == recordFingerprint(&proposal),
        "exact root proposal failed ACCEPT integrity");

    id<MTLBuffer> preflightBuffer = [device
        newBufferWithLength:
            sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU)
                   options:MTLResourceStorageModeShared];
    id<MTLSharedEvent> preflightReady = [device newSharedEvent];
    require(preflightBuffer != nil && preflightReady != nil,
            "Brain preflight resources are unavailable");
    const auto preflight = makePreflight(
        physical.root, resources.request, proposal, controlStep);
    *static_cast<MRNumanXHumanMatterBrainCommitPreflightGPU*>(
        preflightBuffer.contents) = preflight;
    preflightReady.signaledValue = 1u;
    const auto preflightWire = makeWire(
        physical.root, preflightBuffer, preflightReady, 1u);
    require(
        mrnx_bridge_v1_reserve_application(
            physical.prepared, &preflightWire),
        "exact root application reservation was rejected");

    id<MTLBuffer> ackBuffer = [device
        newBufferWithLength:sizeof(MRNumanXHumanMatterBrainAckGPU)
                   options:MTLResourceStorageModeShared];
    id<MTLSharedEvent> ackReady = [device newSharedEvent];
    require(ackBuffer != nil && ackReady != nil,
            "Brain ACK resources are unavailable");
    const auto ack = makeAck(
        physical.root, proposal, preflight, controlStep);
    *static_cast<MRNumanXHumanMatterBrainAckGPU*>(ackBuffer.contents) = ack;
    ackReady.signaledValue = 1u;
    const auto ackWire = makeWire(
        physical.root, ackBuffer, ackReady, 1u);
    id<MTLCommandBuffer> applyCommand = [queue commandBuffer];
    ApplyCapture applyCapture{};
    require(
        applyCommand != nil &&
            mrnx_bridge_v1_submit_apply(
                physical.prepared, (__bridge void*)applyCommand,
                &ackWire, &applyCapture, &applySettled),
        "exact root apply was not encoded");
    [applyCommand commit];
    waitForCapture(
        applyCapture, "exact root apply did not settle exactly once");
    require(
        applyCapture.status.load(std::memory_order_acquire) ==
                MRNX_COMPLETION_ACCEPTED_PENDING_PUBLICATION_V1 &&
            applyCapture.view.command_disposition ==
                MRNX_COMMAND_ACCEPTED_PENDING_PUBLICATION_V1 &&
            applyCommand.status == MTLCommandBufferStatusCompleted,
        "exact root apply was not accepted-but-private");
    const auto applied = copyRecord<MRNumanXHumanMatterAppliedOutcomeGPU>(
        applyCapture.view.applied);
    const auto finalToken =
        copyRecord<MRNumanXAcceptedPhysicsStateTokenGPUV2>(
            applyCapture.view.final_token);
    require(
        applied.status ==
                MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED &&
            applied.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
            applied.physicsTokenFingerprint ==
                acceptedToken.tokenFingerprint &&
            applied.proposalFingerprint == proposal.proposalFingerprint &&
            applied.ackFingerprint == ack.ackFingerprint &&
            applied.appliedFingerprint == recordFingerprint(&applied) &&
            std::memcmp(
                &finalToken, &acceptedToken, sizeof(finalToken)) == 0,
        "exact root apply did not preserve the accepted token and decision");

    mrnx_publication_v1 legacyPublication{};
    legacyPublication.abi_version = MRNX_BRIDGE_ABI_V1;
    legacyPublication.struct_size = sizeof(legacyPublication);
    legacyPublication.joint_commit_fingerprint =
        kJointCommitBaseFingerprint + controlStep;
    legacyPublication.brain_generation = packet.accepted_brain_generation;
    LatchCapture legacyLatch{legacyPublication.brain_generation};
    require(
        !mrnx_bridge_v1_reserve_publication(
            physical.prepared, &legacyPublication) &&
            mrnx_bridge_v1_release_accepted(
                physical.prepared, &legacyPublication, &legacyLatch,
                &latchBrainGeneration) == MRNX_PUBLICATION_REJECTED_V1 &&
            legacyLatch.count.load(std::memory_order_acquire) == 0u,
        "exact root admitted a legacy publication operation");

    mrnx_publication_v2 publication{};
    publication.abi_version = MRNX_PUBLICATION_ABI_V2;
    publication.struct_size = sizeof(publication);
    publication.clock_domain =
        MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    publication.clock_quantum_nanoseconds =
        MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    publication.transaction_fingerprint =
        physical.root.transaction_fingerprint;
    publication.accepted_physics_token_fingerprint =
        acceptedToken.tokenFingerprint;
    publication.candidate_publication_fingerprint =
        packet.candidate_publication_fingerprint;
    publication.joint_commit_fingerprint =
        kJointCommitBaseFingerprint + controlStep;
    publication.brain_generation = packet.accepted_brain_generation;
    publication.committed_timestamp_nanoseconds =
        acceptedToken.acceptedTimestampNanoseconds;
    publication.publication_fingerprint =
        metalrobo::metalNumanXExactPublicationV2Fingerprint(publication);
    require(
        metalrobo::metalNumanXExactPublicationV2Valid(
            acceptedToken, publication) &&
            publication.candidate_publication_fingerprint ==
                sensor.candidate_publication_fingerprint &&
            publication.candidate_publication_fingerprint ==
                proposal.candidatePublicationFingerprint &&
            mrnx_bridge_v1_reserve_publication_v2(
                physical.prepared, &publication),
        "exact joint publication reservation was rejected");

    auto fence =
        copyRecord<MRNumanXHumanMatterJointPublicationFenceGPU>(
            proposalCapture.view.publication_fence);
    require(
        fence.abiVersion ==
                MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2 &&
            fence.status == MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING &&
            fence.transactionFingerprint ==
                publication.transaction_fingerprint &&
            fence.physicsTokenFingerprint ==
                publication.accepted_physics_token_fingerprint &&
            fence.jointCommitFingerprint ==
                publication.joint_commit_fingerprint &&
            fence.brainGeneration == publication.brain_generation &&
            fence.fenceFingerprint == recordFingerprint(&fence),
        "exact publication reservation did not install a canonical PENDING fence");
    fence.status = MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED;
    fence.fenceFingerprint = recordFingerprint(&fence);
    storeRecord(proposalCapture.view.publication_fence, fence);
    if (checkMixedFamilyEntry) {
        BlockingLatchCapture latch{};
        latch.expectedGeneration = publication.brain_generation;
        std::atomic<std::uint32_t> releaseDisposition{
            MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1};
        std::thread publisher([&] {
            releaseDisposition.store(
                mrnx_bridge_v1_release_accepted_v2(
                    physical.prepared, &publication, &latch,
                    &blockingLatchBrainGeneration),
                std::memory_order_release);
        });
        bool latchEntered = false;
        {
            std::unique_lock lock(latch.mutex);
            latchEntered = latch.condition.wait_for(
                lock, std::chrono::seconds(10), [&] {
                    return latch.count.load(std::memory_order_acquire) == 1u;
                });
        }
        // Reaching the borrowed latch proves accepted release claimed the
        // writer outcome under prepared->mutex before this timeout call.
        const bool timeoutRejected = latchEntered &&
            !mrnx_bridge_v1_quarantine_timeout(physical.prepared);
        {
            const std::lock_guard lock(latch.mutex);
            latch.release = true;
        }
        latch.condition.notify_all();
        publisher.join();
        require(
            latchEntered && timeoutRejected &&
                releaseDisposition.load(std::memory_order_acquire) ==
                    MRNX_PUBLICATION_RELEASED_V1 &&
                latch.count.load(std::memory_order_acquire) == 1u,
            "exact publication and timeout did not serialize at the writer gate");
    } else {
        LatchCapture latch{publication.brain_generation};
        require(
            mrnx_bridge_v1_release_accepted_v2(
                physical.prepared, &publication, &latch,
                &latchBrainGeneration) == MRNX_PUBLICATION_RELEASED_V1 &&
                latch.count.load(std::memory_order_acquire) == 1u,
            "exact accepted root did not release jointly");
    }

    // Successful release crosses the retained private HumanIO callbacks and
    // terminalizes the borrowed candidate. The old handle must fail closed;
    // continuity is proven below by the jointly published v5 aggregate.
    mrnx_candidate_view_v1 terminalSensor{};
    terminalSensor.abi_version = MRNX_BRIDGE_ABI_V1;
    terminalSensor.struct_size = sizeof(terminalSensor);
    mrnx_exact_sensor_packet_v2 terminalPacket{};
    terminalPacket.abi_version = MRNX_EXACT_SENSOR_PACKET_ABI_V2;
    terminalPacket.struct_size = sizeof(terminalPacket);
    require(
        !mrnx_bridge_v1_candidate_copy_view(
            physical.candidate, &terminalSensor) &&
            !mrnx_bridge_v1_candidate_copy_sensor_packet_v2(
                physical.candidate, &terminalPacket),
        "released exact candidate remained readable after terminal release");
    require(
        legacyAggregateReadersRejectExact(runtime),
        "exact published root leaked through a legacy aggregate reader");
    RootOutcome outcome{};
    outcome.aggregate.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V5;
    outcome.aggregate.struct_size = sizeof(outcome.aggregate);
    require(
        mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v5(
            runtime, &outcome.aggregate) &&
            snapshotInternallyConsistent(outcome.aggregate) &&
            outcome.aggregate.publication_epoch == controlStep &&
            outcome.aggregate.root.transaction_fingerprint ==
                physical.root.transaction_fingerprint &&
            outcome.aggregate.publication.publication_fingerprint ==
                publication.publication_fingerprint &&
            outcome.aggregate.sensor.candidate_publication_fingerprint ==
                sensor.candidate_publication_fingerprint &&
            outcome.aggregate.sensor_packet.
                    candidate_publication_fingerprint ==
                packet.candidate_publication_fingerprint &&
            outcome.aggregate.physics_generation ==
                acceptedToken.physicsGeneration,
        "exact aggregate v5 did not publish one coherent root");
    outcome.publication = publication;

    mrnx_bridge_v1_candidate_drop(physical.candidate);
    mrnx_bridge_v1_prepared_drop(physical.prepared);
    return outcome;
}

int runLifecycle() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        mrnx_runtime_config_v2 base{};
        base.abi_version = MRNX_RUNTIME_CONFIG_ABI_V2;
        base.struct_size = sizeof(base);
        base.metal_device = (__bridge void*)device;
        base.rigid_payload_path = MRNX_FULLBODY_RIGID;
        base.muscle_payload_path = MRNX_FULLBODY_MUSCLE;
        base.support_contact_payload_path = MRNX_FULLBODY_SUPPORT_CONTACT;
        base.visual_pack_path = MRNX_FULLBODY_VISUAL_PACK;
        base.vision_profile_path = MRNX_FULLBODY_VISION_PROFILE;
        base.metalrobo_metallib_path = MRNX_METALROBO_METALLIB;
        base.matter_metallib_path = MRNX_MATTER_METALLIB;
        base.matter_material_path = MRNX_MATTER_MATERIAL;
        base.timestep_microseconds = 0u;
        base.maximum_retained_bytes = 1024ull * 1024ull * 1024ull;
        base.transaction_slot_count = 2u;

        mrnx_runtime_info_v1 info{};
        mrnx_runtime_v1* runtime = makeExactRuntime(base, info);
        require(
            runtime != nullptr && info.status == MRNX_RUNTIME_READY_V1 &&
                info.device_registry_id == device.registryID &&
                info.accepted_state_proof_program_fingerprint != 0u,
            "exact runtime v8 construction failed");

        const char* behaviorMetricPath =
            std::getenv("MRNX_EXACT_BEHAVIOR_METRIC");
        const char* behaviorMetricSHA256 =
            std::getenv("MRNX_EXACT_BEHAVIOR_METRIC_SHA256");
        const bool behaviorRequired =
            std::getenv("MRNX_REQUIRE_EXACT_BEHAVIOR_METRIC") != nullptr;
        const bool behaviorRequested = behaviorMetricPath != nullptr ||
            behaviorMetricSHA256 != nullptr;
        require(
            !behaviorRequired || behaviorRequested,
            "required exact behavior metric was not supplied");
        require(
            !behaviorRequested ||
                (behaviorMetricPath != nullptr && behaviorMetricPath[0] != '\0' &&
                 behaviorMetricSHA256 != nullptr &&
                 behaviorMetricSHA256[0] != '\0'),
            "exact behavior metric path and SHA-256 must be supplied together");
        if (behaviorRequested) {
            require(
                mrnx_bridge_v1_runtime_behavior_attach(
                    runtime, behaviorMetricPath, behaviorMetricSHA256,
                    kInitialTimestampNanoseconds),
                "source-bound behavior metric did not attach to exact runtime");
        }

        mrnx_aggregate_snapshot_v5 unpublished{};
        unpublished.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V5;
        unpublished.struct_size = sizeof(unpublished);
        require(
            !mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v5(
                runtime, &unpublished),
            "unpublished exact runtime exposed aggregate v5");

        const auto first = executeAcceptedRoot(
            runtime, device, 1u, 0u, 0u,
            kInitialTimestampNanoseconds, true);

        std::atomic<bool> readerStop{false};
        std::atomic<std::uint64_t> readerCount{0u};
        std::atomic<bool> readerFailure{false};
        std::thread reader([&] {
            while (!readerStop.load(std::memory_order_acquire)) {
                mrnx_aggregate_snapshot_v5 snapshot{};
                snapshot.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V5;
                snapshot.struct_size = sizeof(snapshot);
                if (!mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v5(
                        runtime, &snapshot) ||
                    !snapshotInternallyConsistent(snapshot) ||
                    (snapshot.publication_epoch != 1u &&
                     snapshot.publication_epoch != 2u)) {
                    readerFailure.store(true, std::memory_order_release);
                    break;
                }
                readerCount.fetch_add(1u, std::memory_order_relaxed);
            }
        });

        RootOutcome second{};
        try {
            second = executeAcceptedRoot(
                runtime, device, 2u, first.aggregate.brain_generation,
                first.aggregate.physics_generation,
                first.publication.committed_timestamp_nanoseconds, false);
        } catch (...) {
            readerStop.store(true, std::memory_order_release);
            reader.join();
            throw;
        }
        readerStop.store(true, std::memory_order_release);
        reader.join();
        require(
            !readerFailure.load(std::memory_order_acquire) &&
                readerCount.load(std::memory_order_acquire) != 0u &&
                second.aggregate.publication_epoch == 2u &&
                second.aggregate.brain_generation == 2u &&
                second.aggregate.physics_generation == 2u &&
                second.aggregate.sensor_generation == 2u &&
                second.aggregate.root.control_step == 2u &&
                second.publication.committed_timestamp_nanoseconds ==
                    kInitialTimestampNanoseconds +
                        2u * kTimestepNanoseconds,
            "second exact root did not continue atomically from the first");

        mrnx_exact_clock_info_v1 clock{};
        clock.abi_version = MRNX_EXACT_CLOCK_INFO_ABI_V1;
        clock.struct_size = sizeof(clock);
        require(
            mrnx_bridge_v1_runtime_copy_exact_clock(runtime, &clock) &&
                clock.timestep_nanoseconds == kTimestepNanoseconds &&
                clock.clock_quantum_nanoseconds == 1u &&
                clock.published_timestamp_nanoseconds ==
                    second.publication.committed_timestamp_nanoseconds &&
                clock.publication_epoch == 2u,
            "exact clock did not advance with the second publication");

        std::string behaviorJSON;
        if (behaviorRequested) {
            const std::size_t required =
                mrnx_bridge_v1_runtime_behavior_flush_json(runtime, nullptr, 0u);
            require(required > 1u, "exact behavior telemetry did not flush");
            std::vector<char> bytes(required, '\0');
            require(
                mrnx_bridge_v1_runtime_behavior_flush_json(
                    runtime, bytes.data(), bytes.size()) == required &&
                    bytes.back() == '\0',
                "exact behavior telemetry flush failed");
            std::vector<char> replayBytes(required, '\0');
            require(
                mrnx_bridge_v1_runtime_behavior_flush_json(
                    runtime, replayBytes.data(), replayBytes.size()) ==
                        required &&
                    replayBytes.back() == '\0' && replayBytes == bytes,
                "repeated exact behavior telemetry flush changed bytes");
            behaviorJSON.assign(bytes.data());
            NSData* behaviorData = [NSData
                dataWithBytes:behaviorJSON.data()
                length:behaviorJSON.size()];
            NSError* behaviorError = nil;
            id parsed = [NSJSONSerialization
                JSONObjectWithData:behaviorData
                options:0
                error:&behaviorError];
            require(
                behaviorError == nil &&
                    [parsed isKindOfClass:[NSDictionary class]],
                "exact behavior telemetry is not a JSON object");
            NSDictionary* document = (NSDictionary*)parsed;
            const auto unsignedField = [&](NSString* key,
                                           const std::uint64_t expected) {
                id value = document[key];
                require(
                    [value isKindOfClass:[NSNumber class]] &&
                        CFGetTypeID((__bridge CFTypeRef)value) !=
                            CFBooleanGetTypeID() &&
                        [value unsignedLongLongValue] == expected,
                    "exact behavior telemetry numeric field mismatch");
            };
            const auto booleanField = [&](NSString* key,
                                          const bool expected) {
                id value = document[key];
                require(
                    value != nil &&
                        CFGetTypeID((__bridge CFTypeRef)value) ==
                            CFBooleanGetTypeID() &&
                        [value boolValue] == expected,
                    "exact behavior telemetry Boolean field mismatch");
            };
            const auto measuredBooleanField = [&](NSString* key) {
                id value = document[key];
                require(
                    value != nil &&
                        CFGetTypeID((__bridge CFTypeRef)value) ==
                            CFBooleanGetTypeID(),
                    "exact behavior telemetry measured Boolean is missing");
            };
            require(
                [document[@"schema"]
                    isEqual:@"numi.human.accepted-metric-snapshot.v1"] &&
                    [document[@"metric_program_sha256"]
                        isEqual:[NSString
                            stringWithUTF8String:behaviorMetricSHA256]] &&
                    [document[@"native_audit_coverage"]
                        isEqual:@"unavailable"] &&
                    [document[@"forbidden_contact_coverage"]
                        isEqual:@"unavailable"] &&
                    [document[@"generic_taskpack_lowering"]
                        isEqual:@"source_bound_metric_program"] &&
                    document[@"accepted_root_proof_sha256"] == [NSNull null],
                "exact behavior telemetry source or evidence boundary mismatch");
            unsignedField(@"accepted_root_count", 2u);
            unsignedField(@"rejected_attempt_count", 0u);
            unsignedField(@"completed_attempt_count", 2u);
            unsignedField(@"metric_sample_count", 2u);
            unsignedField(@"step_ns", kTimestepNanoseconds);
            unsignedField(
                @"initial_timestamp_ns", kInitialTimestampNanoseconds);
            unsignedField(
                @"end_ns",
                kInitialTimestampNanoseconds + 2u * kTimestepNanoseconds);
            measuredBooleanField(@"initial_posture_valid");
            measuredBooleanField(@"initial_settled");
            booleanField(@"full_behavior_qualified", false);
            booleanField(@"finalized", true);
        } else {
            require(
                mrnx_bridge_v1_runtime_behavior_flush_json(
                    runtime, nullptr, 0u) == 0u,
                "unattached exact runtime exposed behavior telemetry");
        }

        mrnx_bridge_v1_runtime_drop(runtime);
        std::printf(
            "numanx_exact_runtime_v3_lifecycle_probe=pass "
            "roots=2 publications=2 aggregate=v5 clock=nanoseconds "
            "proof_program=v2 legacy_downconversion=rejected "
            "timeout_race=serialized behavior_metric=%s "
            "full_behavior_qualified=false "
            "audit_coverage=unavailable contact_coverage=unavailable "
            "accepted_root_proof=unavailable "
            "concurrent_reads=%llu "
            "final_timestamp_ns=%llu\n",
            behaviorRequested ? "source_bound_exact_clock" : "not_attached",
            static_cast<unsigned long long>(readerCount.load()),
            static_cast<unsigned long long>(
                second.publication.committed_timestamp_nanoseconds));
        return 0;
    }
}

} // namespace exact_runtime_probe

int main() {
    try {
        return exact_runtime_probe::runLifecycle();
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "numanx_exact_runtime_v3_lifecycle_probe: %s\n",
            error.what());
        return 1;
    }
}
