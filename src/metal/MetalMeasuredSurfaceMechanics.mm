#import <Metal/Metal.h>

#include "metalrobo/MetalMeasuredSurfaceMechanics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
    __strong id<MTLComputePipelineState> preparePresentationPipeline = nil;
    __strong id<MTLComputePipelineState> clearPresentationPipeline = nil;
    __strong id<MTLComputePipelineState> rasterPresentationPipeline = nil;
    __strong id<MTLComputePipelineState> compositePresentationPipeline = nil;
    __strong id<MTLBuffer> actions = nil;
    __strong id<MTLBuffer> positions = nil;
    __strong id<MTLBuffer> triangles = nil;
    __strong id<MTLBuffer> vertexParts = nil;
    __strong id<MTLBuffer> acceptedStates = nil;
    __strong id<MTLBuffer> candidateStates = nil;
    __strong id<MTLBuffer> checkpointStates = nil;
    __strong id<MTLBuffer> acceptedEvidence = nil;
    __strong id<MTLBuffer> candidateEvidence = nil;
    __strong id<MTLBuffer> checkpointEvidence = nil;
    __strong id<MTLBuffer> presentationVertices = nil;
    std::size_t presentationVertexCapacity = 0u;
    MetalMeasuredSurfacePresentationStyle presentationStyle{};
    bool presentationConfigured = false;
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
    id<MTLFunction> preparePresentation = [library newFunctionWithName:
        @"mr_prepare_measured_surface_presentation"];
    id<MTLFunction> clearPresentation = [library newFunctionWithName:
        @"mr_clear_measured_surface_presentation_winners"];
    id<MTLFunction> rasterPresentation = [library newFunctionWithName:
        @"mr_raster_measured_surface_presentation"];
    id<MTLFunction> compositePresentation = [library newFunctionWithName:
        @"mr_composite_measured_surface_presentation"];
    if (prepare == nil || commit == nil ||
        preparePresentation == nil || clearPresentation == nil ||
        rasterPresentation == nil || compositePresentation == nil) return false;
    NSError* error = nil;
    id<MTLComputePipelineState> preparePipeline =
        [device newComputePipelineStateWithFunction:prepare error:&error];
    if (preparePipeline == nil ||
        preparePipeline.maxTotalThreadsPerThreadgroup < 256u) return false;
    error = nil;
    id<MTLComputePipelineState> commitPipeline =
        [device newComputePipelineStateWithFunction:commit error:&error];
    if (commitPipeline == nil) return false;
    error = nil;
    id<MTLComputePipelineState> preparePresentationPipeline =
        [device newComputePipelineStateWithFunction:preparePresentation
                                               error:&error];
    error = nil;
    id<MTLComputePipelineState> clearPresentationPipeline =
        [device newComputePipelineStateWithFunction:clearPresentation
                                               error:&error];
    error = nil;
    id<MTLComputePipelineState> rasterPresentationPipeline =
        [device newComputePipelineStateWithFunction:rasterPresentation
                                               error:&error];
    error = nil;
    id<MTLComputePipelineState> compositePresentationPipeline =
        [device newComputePipelineStateWithFunction:compositePresentation
                                               error:&error];
    if (preparePresentationPipeline == nil ||
        clearPresentationPipeline == nil ||
        rasterPresentationPipeline == nil ||
        compositePresentationPipeline == nil) return false;
    const auto& robot = state.binding.robot;
    id<MTLBuffer> stagedActions = makeSharedBuffer(
        device, robot.gpuActions.data(), robot.gpuActions.size(),
        @"MeasuredSurface staging actions");
    id<MTLBuffer> stagedPositions = makeSharedBuffer(
        device, robot.pack.frameMajorPositions.data(),
        robot.pack.frameMajorPositions.size(),
        @"MeasuredSurface staging frame-major positions");
    id<MTLBuffer> stagedTriangles = makeSharedBuffer(
        device, robot.pack.triangleIndices.data(),
        robot.pack.triangleIndices.size(),
        @"MeasuredSurface staging triangles");
    id<MTLBuffer> stagedVertexParts = makeSharedBuffer(
        device, robot.vertexComponents.data(), robot.vertexComponents.size(),
        @"MeasuredSurface staging vertex components");
    if (stagedActions == nil || stagedPositions == nil ||
        stagedTriangles == nil || stagedVertexParts == nil) return false;
    const auto makePrivate = [device](id<MTLBuffer> staging, NSString* label) {
        id<MTLBuffer> buffer = [device newBufferWithLength:staging.length
            options:MTLResourceStorageModePrivate];
        buffer.label = label;
        return buffer;
    };
    id<MTLBuffer> actions = makePrivate(
        stagedActions, @"MeasuredSurface immutable actions");
    id<MTLBuffer> positions = makePrivate(
        stagedPositions, @"MeasuredSurface immutable frame-major positions");
    id<MTLBuffer> triangles = makePrivate(
        stagedTriangles, @"MeasuredSurface immutable triangles");
    id<MTLBuffer> vertexParts = makePrivate(
        stagedVertexParts, @"MeasuredSurface immutable vertex components");
    const NSUInteger stateBytes = std::max<NSUInteger>(
        sizeof(MRMeasuredSurfaceStateGPU) * pass.environmentCount, 1u);
    const NSUInteger evidenceBytes = std::max<NSUInteger>(
        sizeof(MRMeasuredSurfaceEvidenceGPU) * pass.environmentCount, 1u);
    id<MTLBuffer> acceptedStates = [device newBufferWithLength:stateBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> candidateStates = [device newBufferWithLength:stateBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> checkpointStates = [device newBufferWithLength:stateBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> acceptedEvidence = [device newBufferWithLength:evidenceBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> candidateEvidence = [device newBufferWithLength:evidenceBytes
        options:MTLResourceStorageModePrivate];
    id<MTLBuffer> checkpointEvidence = [device newBufferWithLength:evidenceBytes
        options:MTLResourceStorageModePrivate];
    if (actions == nil || positions == nil || triangles == nil ||
        vertexParts == nil || acceptedStates == nil || candidateStates == nil ||
        checkpointStates == nil || acceptedEvidence == nil ||
        candidateEvidence == nil || checkpointEvidence == nil) return false;
    acceptedStates.label = @"MeasuredSurface accepted state";
    candidateStates.label = @"MeasuredSurface candidate state";
    checkpointStates.label = @"MeasuredSurface control-step checkpoint state";
    acceptedEvidence.label = @"MeasuredSurface accepted evidence";
    candidateEvidence.label = @"MeasuredSurface candidate evidence";
    checkpointEvidence.label = @"MeasuredSurface control-step checkpoint evidence";
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return false;
    [blit copyFromBuffer:stagedActions sourceOffset:0u
        toBuffer:actions destinationOffset:0u size:actions.length];
    [blit copyFromBuffer:stagedPositions sourceOffset:0u
        toBuffer:positions destinationOffset:0u size:positions.length];
    [blit copyFromBuffer:stagedTriangles sourceOffset:0u
        toBuffer:triangles destinationOffset:0u size:triangles.length];
    [blit copyFromBuffer:stagedVertexParts sourceOffset:0u
        toBuffer:vertexParts destinationOffset:0u size:vertexParts.length];
    [blit fillBuffer:acceptedStates range:NSMakeRange(0u, stateBytes) value:0u];
    [blit fillBuffer:candidateStates range:NSMakeRange(0u, stateBytes) value:0u];
    [blit fillBuffer:checkpointStates range:NSMakeRange(0u, stateBytes) value:0u];
    [blit fillBuffer:acceptedEvidence range:NSMakeRange(0u, evidenceBytes) value:0u];
    [blit fillBuffer:candidateEvidence range:NSMakeRange(0u, evidenceBytes) value:0u];
    [blit fillBuffer:checkpointEvidence range:NSMakeRange(0u, evidenceBytes) value:0u];
    [blit endEncoding];
    state.device = device;
    state.preparePipeline = preparePipeline;
    state.commitPipeline = commitPipeline;
    state.preparePresentationPipeline = preparePresentationPipeline;
    state.clearPresentationPipeline = clearPresentationPipeline;
    state.rasterPresentationPipeline = rasterPresentationPipeline;
    state.compositePresentationPipeline = compositePresentationPipeline;
    state.actions = actions;
    state.positions = positions;
    state.triangles = triangles;
    state.vertexParts = vertexParts;
    state.acceptedStates = acceptedStates;
    state.candidateStates = candidateStates;
    state.checkpointStates = checkpointStates;
    state.acceptedEvidence = acceptedEvidence;
    state.candidateEvidence = candidateEvidence;
    state.checkpointEvidence = checkpointEvidence;
    state.stats.bufferAllocationCount = 14u;
    state.stats.immutableBytes = actions.length + positions.length +
        triangles.length + vertexParts.length;
    state.stats.persistentBytes = acceptedStates.length +
        checkpointStates.length + acceptedEvidence.length +
        checkpointEvidence.length;
    state.stats.transientBytes = candidateStates.length + candidateEvidence.length;
    state.stats.controlStepCheckpointBytes =
        checkpointStates.length + checkpointEvidence.length;
    state.stats.threadgroupBytes =
        8u * (2u * sizeof(mr_float4) + sizeof(float) * 2u +
              sizeof(std::uint32_t)) +
        2u * sizeof(MRMeasuredSurfaceStateGPU);
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
    dispatch.reserved0 = state.binding.robot.gpuModel.frameCount;
    dispatch.timestepAndWindX = {
        pass.timestepSeconds,
        pass.windVelocity.x,
        pass.windVelocity.y,
        pass.windVelocity.z,
    };
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
        pass.taskStates == nullptr ||
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
    [encoder setBuffer:state.checkpointStates offset:0u atIndex:12u];
    [encoder setBuffer:state.acceptedEvidence offset:0u atIndex:13u];
    [encoder setBuffer:state.candidateEvidence offset:0u atIndex:14u];
    [encoder setBuffer:state.checkpointEvidence offset:0u atIndex:15u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.bodyWrenches offset:0u atIndex:16u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.environmentStatuses offset:0u atIndex:17u];
    MRMetalWorldPassGPU worldPass{
        pass.controlStep, pass.physicsSubstep, 0u, 0u};
    [encoder setBytes:&worldPass length:sizeof(worldPass) atIndex:18u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(mr_float4) atIndex:0u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(mr_float4) atIndex:1u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(float) * 2u atIndex:2u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(std::uint32_t) atIndex:3u];
    [encoder setThreadgroupMemoryLength:
        2u * sizeof(MRMeasuredSurfaceStateGPU) atIndex:4u];
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
    [encoder setBuffer:state.checkpointStates offset:0u atIndex:2u];
    [encoder setBuffer:state.acceptedEvidence offset:0u atIndex:3u];
    [encoder setBuffer:state.candidateEvidence offset:0u atIndex:4u];
    [encoder setBuffer:state.checkpointEvidence offset:0u atIndex:5u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.environmentStatuses offset:0u atIndex:6u];
    MRCompiledMeasuredSurfaceDispatchGPU dispatch = dispatchFor(state, pass);
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:7u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.taskStates offset:0u atIndex:8u];
    const NSUInteger width = std::min<NSUInteger>(
        state.commitPipeline.maxTotalThreadsPerThreadgroup, 128u);
    [encoder dispatchThreads:MTLSizeMake(pass.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    ++state.stats.encodedCommitCount;
    return true;
}

bool encodePresentation(
    void* context,
    const MetalHybridDevicePresentationPass& pass
) {
    auto& state = *static_cast<MetalMeasuredSurfaceMechanics::State*>(context);
    std::scoped_lock lock(state.mutex);
    const auto& encoder = pass.encoder;
    const std::uint32_t environments = pass.liveState.environmentCount;
    const std::uint32_t bodyCount = pass.liveState.bodyCount;
    const auto& robot = state.binding.robot;
    const std::size_t bodyBytesPerEnvironment =
        static_cast<std::size_t>(bodyCount) * sizeof(MRBodyStateGPU);
    if (!state.initialized || !state.presentationConfigured ||
        state.device == nil || !encoder.valid() || environments == 0u ||
        environments > state.stats.environmentCapacity || bodyCount == 0u ||
        state.binding.bodyIndex >= bodyCount ||
        pass.liveState.currentBodyStates == nullptr ||
        pass.liveState.previousBodyStates == nullptr ||
        pass.instanceHeaders == nullptr || pass.sensorInstances == nullptr ||
        pass.cameraStates == nullptr || pass.meshWinners == nullptr ||
        pass.outputs.rgb == nullptr || pass.outputs.depth == nullptr ||
        pass.outputs.validity == nullptr ||
        bodyBytesPerEnvironment == 0u ||
        pass.liveState.currentBodyOffset % bodyBytesPerEnvironment != 0u ||
        pass.liveState.previousBodyOffset % bodyBytesPerEnvironment != 0u ||
        pass.liveState.previousBodyOffset /
                bodyBytesPerEnvironment !=
            pass.liveState.currentBodyOffset / bodyBytesPerEnvironment) {
        return false;
    }
    const std::size_t sourceEnvironment =
        pass.liveState.currentBodyOffset / bodyBytesPerEnvironment;
    if (sourceEnvironment > state.stats.environmentCapacity ||
        environments > state.stats.environmentCapacity - sourceEnvironment) {
        return false;
    }
    const std::size_t vertexCount = robot.gpuModel.vertexCount;
    if (vertexCount == 0u ||
        environments > std::numeric_limits<std::size_t>::max() / vertexCount) {
        return false;
    }
    const std::size_t requiredVertices = environments * vertexCount;
    if (requiredVertices > state.presentationVertexCapacity) {
        if (requiredVertices > std::numeric_limits<std::size_t>::max() /
                sizeof(MRMeasuredSurfaceVisualVertexGPU)) {
            return false;
        }
        id<MTLBuffer> vertices = [state.device
            newBufferWithLength:requiredVertices *
                sizeof(MRMeasuredSurfaceVisualVertexGPU)
            options:MTLResourceStorageModePrivate];
        if (vertices == nil) return false;
        vertices.label = @"MeasuredSurface accepted visual vertices";
        state.presentationVertices = vertices;
        state.presentationVertexCapacity = requiredVertices;
        ++state.stats.bufferAllocationCount;
    }

    MRMeasuredSurfacePresentationGPU presentation{};
    presentation.counts = {
        environments,
        bodyCount,
        static_cast<std::uint32_t>(sourceEnvironment),
        state.binding.bodyIndex,
    };
    presentation.topology = {
        robot.gpuModel.vertexCount,
        robot.gpuModel.triangleCount,
        pass.uniforms.render.x,
        0u,
    };
    presentation.identity = {
        state.presentationStyle.semanticId,
        state.presentationStyle.instanceId,
        state.binding.bodyIndex,
        pass.outputs.outputMask,
    };
    presentation.localTranslationAndScale =
        state.binding.visualOriginFromBodyCenterOfMass;
    presentation.baseColorAndOpacity =
        state.presentationStyle.baseColorAndOpacity;
    presentation.material = {
        state.presentationStyle.perceptualRoughness,
        state.presentationStyle.metallic,
        pass.uniforms.exposure.z,
        0.0f,
    };
    const std::size_t bandPixels =
        pass.uniforms.band.z == 0u
        ? static_cast<std::size_t>(pass.uniforms.image.x) *
            pass.uniforms.band.y
        : static_cast<std::size_t>(pass.uniforms.image.y) *
            pass.uniforms.band.y;
    if (bandPixels == 0u ||
        environments > std::numeric_limits<std::size_t>::max() / bandPixels) {
        return false;
    }
    const std::size_t presentationPixels = environments * bandPixels;
    const std::size_t presentationTriangles =
        static_cast<std::size_t>(environments) *
        robot.gpuModel.triangleCount;

    encoder.setLabel(encoder.context, "MeasuredSurface accepted presentation");
    encoder.setPipeline(
        encoder.context,
        (__bridge void*)state.preparePresentationPipeline
    );
    encoder.setBytes(encoder.context, &robot.gpuModel,
        sizeof(robot.gpuModel), 0u);
    encoder.setBuffer(encoder.context, (__bridge void*)state.positions, 0u, 1u);
    encoder.setBuffer(encoder.context, (__bridge void*)state.vertexParts, 0u, 2u);
    encoder.setBuffer(encoder.context, (__bridge void*)state.acceptedStates, 0u, 3u);
    encoder.setBuffer(encoder.context, (__bridge void*)state.checkpointStates, 0u, 4u);
    encoder.setBuffer(encoder.context, pass.liveState.currentBodyStates,
        pass.liveState.currentBodyOffset, 5u);
    encoder.setBuffer(encoder.context, pass.liveState.previousBodyStates,
        pass.liveState.previousBodyOffset, 6u);
    encoder.setBuffer(encoder.context,
        (__bridge void*)state.presentationVertices, 0u, 7u);
    encoder.setBytes(encoder.context, &presentation, sizeof(presentation), 8u);
    encoder.dispatchThreads(encoder.context, requiredVertices,
        std::min<std::size_t>(
            state.preparePresentationPipeline.maxTotalThreadsPerThreadgroup,
            256u));

    encoder.setPipeline(encoder.context,
        (__bridge void*)state.clearPresentationPipeline);
    encoder.setBuffer(encoder.context, pass.meshWinners, 0u, 0u);
    encoder.setBytes(encoder.context, &pass.uniforms,
        sizeof(pass.uniforms), 1u);
    encoder.dispatchThreads(encoder.context, presentationPixels,
        std::min<std::size_t>(
            state.clearPresentationPipeline.maxTotalThreadsPerThreadgroup,
            256u));

    encoder.setPipeline(encoder.context,
        (__bridge void*)state.rasterPresentationPipeline);
    encoder.setBuffer(encoder.context,
        (__bridge void*)state.presentationVertices, 0u, 0u);
    encoder.setBuffer(encoder.context, (__bridge void*)state.triangles, 0u, 1u);
    encoder.setBuffer(encoder.context, pass.instanceHeaders, 0u, 2u);
    encoder.setBuffer(encoder.context, pass.sensorInstances, 0u, 3u);
    encoder.setBuffer(encoder.context, pass.cameraStates, 0u, 4u);
    encoder.setBuffer(encoder.context, pass.meshWinners, 0u, 5u);
    encoder.setBytes(encoder.context, &presentation, sizeof(presentation), 6u);
    encoder.setBytes(encoder.context, &pass.uniforms,
        sizeof(pass.uniforms), 7u);
    encoder.dispatchThreads(encoder.context, presentationTriangles,
        std::min<std::size_t>(
            state.rasterPresentationPipeline.maxTotalThreadsPerThreadgroup,
            256u));

    encoder.setPipeline(encoder.context,
        (__bridge void*)state.compositePresentationPipeline);
    encoder.setBuffer(encoder.context,
        (__bridge void*)state.presentationVertices, 0u, 0u);
    encoder.setBuffer(encoder.context, (__bridge void*)state.triangles, 0u, 1u);
    encoder.setBuffer(encoder.context, pass.instanceHeaders, 0u, 2u);
    encoder.setBuffer(encoder.context, pass.sensorInstances, 0u, 3u);
    encoder.setBuffer(encoder.context, pass.cameraStates, 0u, 4u);
    encoder.setBuffer(encoder.context, pass.meshWinners, 0u, 5u);
    encoder.setBuffer(encoder.context, pass.outputs.rgb, 0u, 6u);
    encoder.setBuffer(encoder.context, pass.outputs.depth, 0u, 7u);
    encoder.setBuffer(encoder.context, pass.outputs.segmentation, 0u, 8u);
    encoder.setBuffer(encoder.context, pass.outputs.identities, 0u, 9u);
    encoder.setBuffer(encoder.context, pass.outputs.normals, 0u, 10u);
    encoder.setBuffer(encoder.context, pass.outputs.motion, 0u, 11u);
    encoder.setBuffer(encoder.context, pass.outputs.validity, 0u, 12u);
    encoder.setBuffer(encoder.context, pass.lights, 0u, 13u);
    encoder.setBytes(encoder.context, &presentation, sizeof(presentation), 14u);
    encoder.setBytes(encoder.context, &pass.uniforms,
        sizeof(pass.uniforms), 15u);
    encoder.dispatchThreads(encoder.context, presentationPixels,
        std::min<std::size_t>(
            state.compositePresentationPipeline.maxTotalThreadsPerThreadgroup,
            256u));
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

MetalHybridDevicePresentationProgram
MetalMeasuredSurfaceMechanics::presentationProgram(
    MetalMeasuredSurfacePresentationStyle style
) noexcept {
    if (state_ == nullptr || style.semanticId == 0u ||
        style.semanticId == MR_INVALID_INDEX || style.instanceId == 0u ||
        style.instanceId == MR_INVALID_INDEX ||
        !std::isfinite(style.baseColorAndOpacity.x) ||
        !std::isfinite(style.baseColorAndOpacity.y) ||
        !std::isfinite(style.baseColorAndOpacity.z) ||
        !std::isfinite(style.baseColorAndOpacity.w) ||
        !std::isfinite(style.perceptualRoughness) ||
        !std::isfinite(style.metallic)) {
        return {};
    }
    std::scoped_lock lock(state_->mutex);
    style.baseColorAndOpacity.w = 1.0f;
    style.perceptualRoughness = std::clamp(
        style.perceptualRoughness, 0.045f, 1.0f);
    style.metallic = std::clamp(style.metallic, 0.0f, 1.0f);
    state_->presentationStyle = style;
    state_->presentationConfigured = true;
    std::uint64_t fingerprint = state_->binding.robot.fingerprint ^
        0x6d65617375726564ull ^
        (static_cast<std::uint64_t>(style.semanticId) << 32u) ^
        style.instanceId;
    if (fingerprint == 0u) fingerprint = 1u;
    return {
        .context = state_.get(),
        .encode = encodePresentation,
        .fingerprint = fingerprint,
    };
}

MetalMeasuredSurfaceInspection
MetalMeasuredSurfaceMechanics::inspectAccepted() const {
    std::scoped_lock lock(state_->mutex);
    if (!state_->initialized || state_->device == nil ||
        state_->acceptedStates == nil || state_->acceptedEvidence == nil) {
        throw std::logic_error(
            "measured-surface state is unavailable before its first completed submission");
    }
    id<MTLCommandQueue> queue = [state_->device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBuffer> states = [state_->device
        newBufferWithLength:state_->acceptedStates.length
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> evidence = [state_->device
        newBufferWithLength:state_->acceptedEvidence.length
        options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (queue == nil || command == nil || states == nil || evidence == nil ||
        blit == nil) {
        throw std::runtime_error(
            "cannot allocate measured-surface inspection resources");
    }
    [blit copyFromBuffer:state_->acceptedStates sourceOffset:0u
        toBuffer:states destinationOffset:0u size:states.length];
    [blit copyFromBuffer:state_->acceptedEvidence sourceOffset:0u
        toBuffer:evidence destinationOffset:0u size:evidence.length];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(
            "measured-surface inspection command did not complete");
    }
    MetalMeasuredSurfaceInspection inspection;
    inspection.acceptedStates.resize(state_->stats.environmentCapacity);
    inspection.acceptedEvidence.resize(state_->stats.environmentCapacity);
    std::memcpy(inspection.acceptedStates.data(), states.contents,
        inspection.acceptedStates.size() * sizeof(MRMeasuredSurfaceStateGPU));
    std::memcpy(inspection.acceptedEvidence.data(), evidence.contents,
        inspection.acceptedEvidence.size() * sizeof(MRMeasuredSurfaceEvidenceGPU));
    return inspection;
}

} // namespace metalrobo
