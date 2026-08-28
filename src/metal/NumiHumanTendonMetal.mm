#include "metalrobo/NumiHumanTendonMetal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>

namespace metalrobo {
namespace {

constexpr NSUInteger kThreadsPerThreadgroup = 128u;

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(reinterpret_cast<const void*>(&defaultMetallibPath), &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory / "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() / "shaders/MetalRobo.metallib",
        };
        for (const auto& candidate : candidates) {
            if (regularFile(candidate)) return candidate.string();
        }
    }
    const std::filesystem::path configured{METALROBO_DEFAULT_METALLIB};
    return regularFile(configured) ? configured.string() : std::string{};
}

std::string nsString(NSString* value) {
    return value != nil && value.UTF8String != nullptr
        ? std::string{value.UTF8String} : std::string{};
}

std::string describeError(NSError* error) {
    if (error == nil) return "unknown Metal error";
    std::string result = nsString(error.localizedDescription);
    if (result.empty()) result = nsString(error.description);
    return result.empty() ? "unknown Metal error" : result;
}

NumiHumanTendonMetalDiagnostics reject(
    NumiHumanTendonMetalDiagnostics diagnostics,
    const NumiHumanTendonMetalStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

} // namespace

NumiHumanTendonDiagnostics makeNumiHumanTendonMetalProgram(
    const NumiHumanTendonPayload& payload,
    NumiHumanTendonMetalProgram& program
) {
    program = {};
    if (payload.payloadAbi != 2u || payload.muscleCount == 0u ||
        payload.bindings.size() != 2u * payload.muscleCount ||
        payload.envelopes.empty()) {
        return {NumiHumanTendonStatus::invalidPayload, MR_INVALID_INDEX};
    }
    program.bindings.reserve(payload.bindings.size());
    for (std::size_t index = 0u; index < payload.bindings.size(); ++index) {
        const NumiHumanTendonBinding& source = payload.bindings[index];
        if (source.muscleIndex >= payload.muscleCount || source.endpointOrdinal > 1u ||
            source.bodyIndex >= payload.bodyCount ||
            (source.mode != NumiHumanTendonAttachmentMode::sourceSitePoint &&
             source.mode != NumiHumanTendonAttachmentMode::registeredBoneDistributedEnvelope)) {
            return {NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(index)};
        }
        MRNumiHumanTendonBindingGPU destination{};
        destination.muscleIndex = source.muscleIndex;
        destination.endpointOrdinal = source.endpointOrdinal;
        destination.bodyIndex = source.bodyIndex;
        destination.mode = static_cast<std::uint32_t>(source.mode);
        destination.envelopeIndex = source.triangleIndex;
        destination.boneStableId = source.boneStableId;
        destination.sourceLocalPoint = {
            static_cast<float>(source.resolvedLocalPoint[0]),
            static_cast<float>(source.resolvedLocalPoint[1]),
            static_cast<float>(source.resolvedLocalPoint[2]), 0.0f,
        };
        program.bindings.push_back(destination);
    }
    program.envelopes.reserve(payload.envelopes.size());
    for (std::size_t index = 0u; index < payload.envelopes.size(); ++index) {
        const NumiHumanTendonEnvelope& source = payload.envelopes[index];
        if (source.bodyIndex >= payload.bodyCount || source.boneStableId == 0u) {
            return {NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(index)};
        }
        MRNumiHumanTendonEnvelopeGPU destination{};
        destination.bodyIndex = source.bodyIndex;
        destination.boneStableId = source.boneStableId;
        destination.sourceTriangleIndex = source.sourceTriangleIndex;
        destination.nodeCount = 4u;
        for (std::size_t node = 0u; node < 4u; ++node) {
            destination.localNodes[node] = {
                static_cast<float>(source.localNodes[node][0]),
                static_cast<float>(source.localNodes[node][1]),
                static_cast<float>(source.localNodes[node][2]), 0.0f,
            };
            for (std::size_t row = 0u; row < 3u; ++row) {
                destination.forceMapRows[3u * node + row] = {
                    static_cast<float>(source.forceMaps[node][row][0]),
                    static_cast<float>(source.forceMaps[node][row][1]),
                    static_cast<float>(source.forceMaps[node][row][2]), 0.0f,
                };
            }
        }
        destination.metrics = {
            static_cast<float>(source.surfaceDistance),
            static_cast<float>(source.patchRadius),
            static_cast<float>(source.forceAmplification),
            static_cast<float>(source.l2ForceAmplification),
        };
        program.envelopes.push_back(destination);
    }
    return {};
}

NumiHumanTendonMetalDiagnostics runMetalNumiHumanTendonTransfer(
    const NumiHumanTendonMetalProgram& program,
    const NumiHumanTendonMetalInput& input,
    NumiHumanTendonMetalResult& result,
    const NumiHumanTendonMetalConfig& config
) {
    NumiHumanTendonMetalDiagnostics diagnostics;
    if (program.bindings.empty() || program.bindings.size() % 2u != 0u ||
        program.envelopes.empty()) {
        return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::invalidProgram,
                      "NHTENDON2 Metal program is empty or incomplete");
    }
    const std::size_t muscleCount = program.bindings.size() / 2u;
    if (input.environmentCount == 0u || input.dofCount == 0u ||
        input.bodyPoseStride == 0u || input.pointJacobianStride == 0u ||
        input.bodyJacobianPointOffset == MR_INVALID_INDEX ||
        input.dofCount > std::numeric_limits<std::uint32_t>::max() ||
        input.environmentCount > std::numeric_limits<std::uint32_t>::max() ||
        input.bodyPoseStride > std::numeric_limits<std::uint32_t>::max() ||
        input.pointJacobianStride > std::numeric_limits<std::uint32_t>::max() ||
        input.muscleResults.size() != input.environmentCount * muscleCount ||
        input.bodyPoses.size() != input.environmentCount * input.bodyPoseStride ||
        input.pointJacobians.size() != input.environmentCount * input.pointJacobianStride ||
        input.pointJacobianStride % (3u * input.dofCount) != 0u) {
        return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::invalidInput,
                      "NHTENDON2 Metal input dimensions are inconsistent");
    }
    const std::size_t pointCount = input.pointJacobianStride / (3u * input.dofCount);
    if (input.bodyJacobianPointOffset > pointCount ||
        4u * input.bodyPoseStride > pointCount - input.bodyJacobianPointOffset) {
        return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::invalidInput,
                      "NHTENDON2 Metal body-Jacobian probe block is incomplete");
    }
    const std::size_t resultCount = input.environmentCount * program.bindings.size();
    if (resultCount > std::numeric_limits<NSUInteger>::max() /
                          sizeof(MRNumiHumanTendonTransferResultGPU) ||
        resultCount > std::numeric_limits<NSUInteger>::max() /
                          (input.dofCount * sizeof(float))) {
        return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::invalidInput,
                      "NHTENDON2 Metal output dimensions overflow NSUInteger");
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::metalUnavailable,
                          "no Apple Metal device is available");
        }
        diagnostics.deviceName = nsString(device.name);
        const std::string metallibPath = config.metallibPath.empty()
            ? defaultMetallibPath() : config.metallibPath;
        if (metallibPath.empty()) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::metallibUnavailable,
                          "MetalRobo metallib is unavailable");
        }
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:metallibPath.c_str()]]
            error:&error];
        if (library == nil) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::pipelineFailure,
                          "cannot load MetalRobo metallib: " + describeError(error));
        }
        id<MTLFunction> function = [library newFunctionWithName:@"mr_numi_human_tendon_transfer"];
        error = nil;
        id<MTLComputePipelineState> pipeline = function == nil ? nil :
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < kThreadsPerThreadgroup) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::pipelineFailure,
                          "cannot create NHTENDON2 Metal pipeline: " + describeError(error));
        }
        MRNumiHumanTendonTransferDispatchGPU dispatch{};
        dispatch.abiVersion = MR_NUMI_HUMAN_TENDON_TRANSFER_GPU_ABI_VERSION;
        dispatch.endpointCount = static_cast<mr_u32>(program.bindings.size());
        dispatch.envelopeCount = static_cast<mr_u32>(program.envelopes.size());
        dispatch.muscleCount = static_cast<mr_u32>(muscleCount);
        dispatch.environmentCount = static_cast<mr_u32>(input.environmentCount);
        dispatch.dofCount = static_cast<mr_u32>(input.dofCount);
        dispatch.bodyPoseStride = static_cast<mr_u32>(input.bodyPoseStride);
        dispatch.articulationFirstBody = input.articulationFirstBody;
        dispatch.pointJacobianStride = static_cast<mr_u32>(input.pointJacobianStride);
        dispatch.bodyJacobianPointOffset = input.bodyJacobianPointOffset;
        dispatch.bodyJacobianPointStride = 4u;
        const auto inputBuffer = [device](const void* bytes, const NSUInteger length) {
            return [device newBufferWithBytes:bytes length:length
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> dispatchBuffer = inputBuffer(&dispatch, sizeof(dispatch));
        id<MTLBuffer> bindingBuffer = inputBuffer(
            program.bindings.data(), program.bindings.size() * sizeof(program.bindings.front()));
        id<MTLBuffer> envelopeBuffer = inputBuffer(
            program.envelopes.data(), program.envelopes.size() * sizeof(program.envelopes.front()));
        id<MTLBuffer> muscleBuffer = inputBuffer(
            input.muscleResults.data(), input.muscleResults.size() * sizeof(input.muscleResults.front()));
        id<MTLBuffer> poseBuffer = inputBuffer(
            input.bodyPoses.data(), input.bodyPoses.size() * sizeof(input.bodyPoses.front()));
        id<MTLBuffer> jacobianBuffer = inputBuffer(
            input.pointJacobians.data(), input.pointJacobians.size() * sizeof(float));
        id<MTLBuffer> resultBuffer = [device newBufferWithLength:
            resultCount * sizeof(MRNumiHumanTendonTransferResultGPU)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> correctionBuffer = [device newBufferWithLength:
            resultCount * input.dofCount * sizeof(float)
            options:MTLResourceStorageModeShared];
        if (dispatchBuffer == nil || bindingBuffer == nil || envelopeBuffer == nil ||
            muscleBuffer == nil || poseBuffer == nil || jacobianBuffer == nil ||
            resultBuffer == nil || correctionBuffer == nil) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::metalUnavailable,
                          "NHTENDON2 Metal buffer allocation failed");
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (queue == nil || command == nil || encoder == nil) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::commandFailure,
                          "NHTENDON2 Metal command allocation failed");
        }
        [encoder setComputePipelineState:pipeline];
        const std::array buffers{dispatchBuffer, bindingBuffer, envelopeBuffer, muscleBuffer,
                                 poseBuffer, jacobianBuffer, resultBuffer, correctionBuffer};
        for (NSUInteger index = 0u; index < buffers.size(); ++index) {
            [encoder setBuffer:buffers[index] offset:0u atIndex:index];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(
            (resultCount + kThreadsPerThreadgroup - 1u) / kThreadsPerThreadgroup, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerThreadgroup, 1u, 1u)];
        [encoder endEncoding];
        const auto start = std::chrono::steady_clock::now();
        diagnostics.dispatched = true;
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::commandFailure,
                          "NHTENDON2 Metal command failed: " + describeError(command.error));
        }
        std::vector<MRNumiHumanTendonTransferResultGPU> transfers(resultCount);
        std::vector<float> corrections(resultCount * input.dofCount);
        std::memcpy(transfers.data(), resultBuffer.contents,
                    transfers.size() * sizeof(transfers.front()));
        std::memcpy(corrections.data(), correctionBuffer.contents,
                    corrections.size() * sizeof(corrections.front()));
        for (std::size_t index = 0u; index < transfers.size(); ++index) {
            const auto& transfer = transfers[index];
            if (transfer.status != MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS ||
                transfer.environment != index / program.bindings.size() ||
                transfer.bindingIndex != index % program.bindings.size() ||
                !finite(transfer.terminalWorldForce) ||
                !finite(transfer.residualsAndForce) ||
                !std::all_of(std::begin(transfer.nodalWorldForces),
                             std::end(transfer.nodalWorldForces),
                             [](const mr_float4 value) { return finite(value); })) {
                return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::gpuFailure,
                              "NHTENDON2 Metal kernel rejected binding " + std::to_string(index));
            }
        }
        if (!std::all_of(corrections.begin(), corrections.end(),
                         [](const float value) { return std::isfinite(value); })) {
            return reject(std::move(diagnostics), NumiHumanTendonMetalStatus::gpuFailure,
                          "NHTENDON2 Metal kernel produced a non-finite generalized correction");
        }
        result.transfers = std::move(transfers);
        result.generalizedCorrections = std::move(corrections);
        diagnostics.published = true;
    }
    return diagnostics;
}

const char* numiHumanTendonMetalStatusName(const NumiHumanTendonMetalStatus status) noexcept {
    switch (status) {
    case NumiHumanTendonMetalStatus::success: return "success";
    case NumiHumanTendonMetalStatus::invalidProgram: return "invalid_program";
    case NumiHumanTendonMetalStatus::invalidInput: return "invalid_input";
    case NumiHumanTendonMetalStatus::metallibUnavailable: return "metallib_unavailable";
    case NumiHumanTendonMetalStatus::metalUnavailable: return "metal_unavailable";
    case NumiHumanTendonMetalStatus::pipelineFailure: return "pipeline_failure";
    case NumiHumanTendonMetalStatus::commandFailure: return "command_failure";
    case NumiHumanTendonMetalStatus::gpuFailure: return "gpu_failure";
    }
    return "unknown";
}

} // namespace metalrobo
