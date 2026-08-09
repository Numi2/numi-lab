#import <Metal/Metal.h>

#include "metalrobo/MetalMeasuredSurfaceMechanics.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace metalrobo {

struct MetalMeasuredSurfaceMechanics::State {
    explicit State(CompiledMeasuredSurfaceBinding compiled)
        : binding(std::move(compiled)) {}

    CompiledMeasuredSurfaceBinding binding;
    mutable std::mutex mutex;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLComputePipelineState> preparePipeline = nil;
    __strong id<MTLComputePipelineState> commitPipeline = nil;
    __strong id<MTLBuffer> actions = nil;
    __strong id<MTLBuffer> positions = nil;
    __strong id<MTLBuffer> triangles = nil;
    __strong id<MTLBuffer> vertexParts = nil;
    __strong id<MTLBuffer> acceptedStates = nil;
    __strong id<MTLBuffer> candidateStates = nil;
    __strong id<MTLBuffer> acceptedEvidence = nil;
    __strong id<MTLBuffer> candidateEvidence = nil;
    MetalMeasuredSurfaceStats stats;
    bool initialized = false;
};

namespace {

template <typename T>
id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const T* values,
    const std::size_t count,
    NSString* label
) {
    if (count == 0u) return nil;
    id<MTLBuffer> buffer = [device newBufferWithBytes:values
        length:count * sizeof(T)
        options:MTLResourceStorageModeShared];
    buffer.label = label;
    return buffer;
}

bool initialize(
    MetalMeasuredSurfaceMechanics::State& state,
    const MetalWorldDeviceMechanicsPass& pass
) {
    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)pass.metalLibrary;
    if (command == nil || library == nil || command.device == nil) return false;
    if (state.initialized) {
        return state.device == command.device &&
            state.stats.environmentCapacity == pass.environmentCount;
    }
    id<MTLDevice> device = command.device;
    id<MTLFunction> prepare = [library newFunctionWithName:
        @"mr_step_measured_surface_mechanics"];
    id<MTLFunction> commit = [library newFunctionWithName:
        @"mr_commit_measured_surface_mechanics"];
    if (prepare == nil || commit == nil) return false;
    NSError* error = nil;
    id<MTLComputePipelineState> preparePipeline =
        [device newComputePipelineStateWithFunction:prepare error:&error];
    if (preparePipeline == nil ||
        preparePipeline.maxTotalThreadsPerThreadgroup < 256u) return false;
    error = nil;
    id<MTLComputePipelineState> commitPipeline =
        [device newComputePipelineStateWithFunction:commit error:&error];
    if (commitPipeline == nil) return false;
    const auto& robot = state.binding.robot;
    id<MTLBuffer> actions = makeSharedBuffer(
        device, robot.gpuActions.data(), robot.gpuActions.size(),
        @"MeasuredSurface immutable actions");
    id<MTLBuffer> positions = makeSharedBuffer(
        device, robot.pack.frameMajorPositions.data(),
        robot.pack.frameMajorPositions.size(),
        @"MeasuredSurface immutable frame-major positions");
    id<MTLBuffer> triangles = makeSharedBuffer(
        device, robot.pack.triangleIndices.data(),
        robot.pack.triangleIndices.size(),
        @"MeasuredSurface immutable triangles");
    id<MTLBuffer> vertexParts = makeSharedBuffer(
        device, robot.vertexComponents.data(), robot.vertexComponents.size(),
        @"MeasuredSurface immutable vertex components");
    const NSUInteger stateBytes = std::max<NSUInteger>(
        sizeof(MRMeasuredSurfaceStateGPU) * pass.environmentCount, 1u);
    const NSUInteger evidenceBytes = std::max<NSUInteger>(
        sizeof(MRMeasuredSurfaceEvidenceGPU) * pass.environmentCount, 1u);
    id<MTLBuffer> acceptedStates = [device newBufferWithLength:stateBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> candidateStates = [device newBufferWithLength:stateBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> acceptedEvidence = [device newBufferWithLength:evidenceBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> candidateEvidence = [device newBufferWithLength:evidenceBytes
        options:MTLResourceStorageModePrivate];
    if (actions == nil || positions == nil || triangles == nil ||
        vertexParts == nil || acceptedStates == nil || candidateStates == nil ||
        acceptedEvidence == nil || candidateEvidence == nil) return false;
    acceptedStates.label = @"MeasuredSurface accepted state";
    candidateStates.label = @"MeasuredSurface candidate state";
    acceptedEvidence.label = @"MeasuredSurface accepted evidence";
    candidateEvidence.label = @"MeasuredSurface candidate evidence";
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return false;
    [blit fillBuffer:acceptedStates range:NSMakeRange(0u, stateBytes) value:0u];
    [blit fillBuffer:candidateStates range:NSMakeRange(0u, stateBytes) value:0u];
    [blit fillBuffer:acceptedEvidence range:NSMakeRange(0u, evidenceBytes) value:0u];
    [blit fillBuffer:candidateEvidence range:NSMakeRange(0u, evidenceBytes) value:0u];
    [blit endEncoding];
    state.device = device;
    state.preparePipeline = preparePipeline;
    state.commitPipeline = commitPipeline;
    state.actions = actions;
    state.positions = positions;
    state.triangles = triangles;
    state.vertexParts = vertexParts;
    state.acceptedStates = acceptedStates;
    state.candidateStates = candidateStates;
    state.acceptedEvidence = acceptedEvidence;
    state.candidateEvidence = candidateEvidence;
    state.stats.bufferAllocationCount = 8u;
    state.stats.immutableBytes = actions.length + positions.length +
        triangles.length + vertexParts.length;
    state.stats.persistentBytes = acceptedStates.length + acceptedEvidence.length;
    state.stats.transientBytes = candidateStates.length + candidateEvidence.length;
    state.stats.environmentCapacity = pass.environmentCount;
    state.stats.threadgroupWidth = 256u;
    state.initialized = true;
    return true;
}

MRCompiledMeasuredSurfaceDispatchGPU dispatchFor(
    const MetalMeasuredSurfaceMechanics::State& state,
    const MetalWorldDeviceMechanicsPass& pass
) {
    MRCompiledMeasuredSurfaceDispatchGPU dispatch{};
    dispatch.environmentCount = pass.environmentCount;
    dispatch.qStride = pass.nq;
    dispatch.vStride = pass.nv;
    dispatch.bodyStride = pass.bodyCount;
    dispatch.qOffset = state.binding.qOffset;
    dispatch.vOffset = state.binding.vOffset;
    dispatch.bodyIndex = state.binding.bodyIndex;
    dispatch.localBodyIndex = 0u;
    dispatch.actionCount = pass.actionCount;
    dispatch.actionHistoryStride = pass.actionHistoryStride;
    dispatch.filterSlot = pass.actionFilterSlot;
    dispatch.firstAction = state.binding.firstAction;
    dispatch.threadsPerThreadgroup = 256u;
    dispatch.timestepAndWindX = {pass.timestepSeconds, 0.0f, 0.0f, 0.0f};
    return dispatch;
}

bool encodePrepare(void* context, const MetalWorldDeviceMechanicsPass& pass) {
    auto& state = *static_cast<MetalMeasuredSurfaceMechanics::State*>(context);
    std::scoped_lock lock(state.mutex);
    if (!initialize(state, pass) || pass.q == nullptr || pass.v == nullptr ||
        pass.actionHistory == nullptr || pass.resetMasks == nullptr ||
        pass.bodyWrenches == nullptr || pass.environmentStatuses == nullptr ||
        pass.actionCount < state.binding.firstAction +
            state.binding.robot.gpuModel.actionCount ||
        state.binding.qOffset + 7u > pass.nq ||
        state.binding.vOffset + 6u > pass.nv ||
        state.binding.bodyIndex >= pass.bodyCount ||
        !(pass.timestepSeconds > 0.0f)) return false;
    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return false;
    encoder.label = @"MeasuredSurface device-resident mechanics";
    [encoder setComputePipelineState:state.preparePipeline];
    [encoder setBytes:&state.binding.robot.gpuModel
        length:sizeof(MRMeasuredSurfaceModelGPU) atIndex:0u];
    [encoder setBuffer:state.actions offset:0u atIndex:1u];
    [encoder setBuffer:state.positions offset:0u atIndex:2u];
    [encoder setBuffer:state.triangles offset:0u atIndex:3u];
    [encoder setBuffer:state.vertexParts offset:0u atIndex:4u];
    MRCompiledMeasuredSurfaceDispatchGPU dispatch = dispatchFor(state, pass);
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:5u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.actionHistory offset:0u atIndex:6u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.resetMasks offset:0u atIndex:7u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.q offset:0u atIndex:8u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.v offset:0u atIndex:9u];
    [encoder setBuffer:state.acceptedStates offset:0u atIndex:10u];
    [encoder setBuffer:state.candidateStates offset:0u atIndex:11u];
    [encoder setBuffer:state.candidateEvidence offset:0u atIndex:12u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.bodyWrenches offset:0u atIndex:13u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.environmentStatuses offset:0u atIndex:14u];
    MRMetalWorldPassGPU worldPass{
        pass.controlStep, pass.physicsSubstep, 0u, 0u};
    [encoder setBytes:&worldPass length:sizeof(worldPass) atIndex:15u];
    [encoder setThreadgroupMemoryLength:256u * sizeof(mr_float4) atIndex:0u];
    [encoder setThreadgroupMemoryLength:256u * sizeof(mr_float4) atIndex:1u];
    [encoder setThreadgroupMemoryLength:256u * sizeof(float) * 2u atIndex:2u];
    [encoder setThreadgroupMemoryLength:256u * sizeof(std::uint32_t) atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(pass.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    ++state.stats.encodedPrepareCount;
    return true;
}

bool encodeCommit(void* context, const MetalWorldDeviceMechanicsPass& pass) {
    auto& state = *static_cast<MetalMeasuredSurfaceMechanics::State*>(context);
    std::scoped_lock lock(state.mutex);
    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    if (!state.initialized || command == nil ||
        state.device != command.device || pass.environmentStatuses == nullptr) {
        return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return false;
    encoder.label = @"MeasuredSurface transactional commit";
    [encoder setComputePipelineState:state.commitPipeline];
    [encoder setBuffer:state.acceptedStates offset:0u atIndex:0u];
    [encoder setBuffer:state.candidateStates offset:0u atIndex:1u];
    [encoder setBuffer:state.acceptedEvidence offset:0u atIndex:2u];
    [encoder setBuffer:state.candidateEvidence offset:0u atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.environmentStatuses offset:0u atIndex:4u];
    MRCompiledMeasuredSurfaceDispatchGPU dispatch = dispatchFor(state, pass);
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:5u];
    const NSUInteger width = std::min<NSUInteger>(
        state.commitPipeline.maxTotalThreadsPerThreadgroup, 128u);
    [encoder dispatchThreads:MTLSizeMake(pass.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    ++state.stats.encodedCommitCount;
    return true;
}

} // namespace

MetalMeasuredSurfaceMechanics::MetalMeasuredSurfaceMechanics(
    CompiledMeasuredSurfaceBinding binding
) : state_(std::make_unique<State>(std::move(binding))) {
    if (!state_->binding.valid()) {
        throw std::invalid_argument(
            "Metal measured-surface mechanics requires a valid compiled binding");
    }
}

MetalMeasuredSurfaceMechanics::~MetalMeasuredSurfaceMechanics() = default;
MetalMeasuredSurfaceMechanics::MetalMeasuredSurfaceMechanics(
    MetalMeasuredSurfaceMechanics&&) noexcept = default;
MetalMeasuredSurfaceMechanics& MetalMeasuredSurfaceMechanics::operator=(
    MetalMeasuredSurfaceMechanics&&) noexcept = default;

MetalWorldDeviceMechanicsProgram
MetalMeasuredSurfaceMechanics::program() noexcept {
    return {
        .context = state_.get(),
        .prepare = encodePrepare,
        .commit = encodeCommit,
        .fingerprint = state_->binding.fingerprint,
    };
}

const CompiledMeasuredSurfaceBinding&
MetalMeasuredSurfaceMechanics::binding() const noexcept {
    return state_->binding;
}

MetalMeasuredSurfaceStats MetalMeasuredSurfaceMechanics::stats() const noexcept {
    std::scoped_lock lock(state_->mutex);
    return state_->stats;
}

} // namespace metalrobo
