#import <Metal/Metal.h>

#include "metalrobo/NumiHumanExtensorHoodMetal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

std::uint64_t appendFingerprint(
    std::uint64_t hash, const void* bytes, const std::size_t count
) {
    constexpr std::uint64_t prime = 1099511628211ull;
    const auto* cursor = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0u; index < count; ++index) {
        hash ^= cursor[index];
        hash *= prime;
    }
    return hash;
}

bool finite4(const mr_float4& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

} // namespace

struct NumiHumanExtensorHoodMetalAdapter::State {
    std::vector<MRNumiHumanExtensorHoodRayGPU> rays;
    std::vector<MRNumiHumanExtensorHoodNodeGPU> nodes;
    std::vector<MRNumiHumanExtensorHoodElementGPU> elements;
    std::vector<MRNumiHumanExtensorHoodInputGPU> inputs;
    std::vector<MRMujocoMuscleRouteCutGPU> routeCuts;
    NumiHumanExtensorHoodMetalConfiguration configuration;
    std::uint32_t environmentCount = 0u;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t lastStepIndex = 0u;
    std::uint64_t fingerprint = 0u;
    std::string message;

    __strong id<MTLDevice> device = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> solvePipeline = nil;
    __strong id<MTLComputePipelineState> routeCutPipeline = nil;
    __strong id<MTLComputePipelineState> assemblePipeline = nil;
    __strong id<MTLComputePipelineState> applyPipeline = nil;
    __strong id<MTLComputePipelineState> commitPipeline = nil;
    __strong id<MTLBuffer> rayBuffer = nil;
    __strong id<MTLBuffer> nodeBuffer = nil;
    __strong id<MTLBuffer> elementBuffer = nil;
    __strong id<MTLBuffer> inputBuffer = nil;
    __strong id<MTLBuffer> routeCutBuffer = nil;
    __strong id<MTLBuffer> suffixJacobianBuffer = nil;
    __strong id<MTLBuffer> nodeResultBuffer = nil;
    __strong id<MTLBuffer> rayResultBuffer = nil;
    __strong id<MTLBuffer> historyBuffer = nil;
    __strong id<MTLBuffer> correctionBuffer = nil;
};

NumiHumanExtensorHoodMetalAdapter::NumiHumanExtensorHoodMetalAdapter() = default;
NumiHumanExtensorHoodMetalAdapter::~NumiHumanExtensorHoodMetalAdapter() = default;
NumiHumanExtensorHoodMetalAdapter::NumiHumanExtensorHoodMetalAdapter(
    NumiHumanExtensorHoodMetalAdapter&&) noexcept = default;
NumiHumanExtensorHoodMetalAdapter&
NumiHumanExtensorHoodMetalAdapter::operator=(
    NumiHumanExtensorHoodMetalAdapter&&) noexcept = default;

bool NumiHumanExtensorHoodMetalAdapter::initialize(
    const NumiHumanExtensorHoodMetalSource& source,
    const NumiHumanExtensorHoodMetalConfiguration& configuration
) {
    if (source.environmentCount == 0u || source.rays.size() != 8u ||
        source.nodes.empty() || source.elements.empty() ||
        source.inputs.empty() || configuration.maximumIterations == 0u ||
        configuration.maximumLineSearchSteps == 0u ||
        !(configuration.forceToleranceNewtons > 0.0f) ||
        !(configuration.minimumLengthMeters > 0.0f) ||
        !(configuration.diagonalRegularization > 0.0f) ||
        !(configuration.armijoFraction > 0.0f &&
          configuration.armijoFraction < 1.0f) ||
        !(configuration.foundationStiffnessNewtonsPerMeter > 0.0f) ||
        !std::isfinite(configuration.forceToleranceNewtons) ||
        !std::isfinite(configuration.minimumLengthMeters) ||
        !std::isfinite(configuration.diagonalRegularization) ||
        !std::isfinite(configuration.armijoFraction) ||
        !std::isfinite(configuration.foundationStiffnessNewtonsPerMeter)) {
        return false;
    }
    std::uint32_t expectedNodeOffset = 0u;
    std::uint32_t expectedElementOffset = 0u;
    std::uint32_t expectedInputOffset = 0u;
    for (std::size_t rayIndex = 0u; rayIndex < source.rays.size(); ++rayIndex) {
        const auto& ray = source.rays[rayIndex];
        const std::uint32_t expectedSide = static_cast<std::uint32_t>(
            rayIndex / 4u);
        const std::uint32_t expectedDigit = static_cast<std::uint32_t>(
            2u + rayIndex % 4u);
        const std::uint32_t expectedNodes = expectedDigit == 5u ? 12u : 10u;
        const std::uint32_t expectedElements = expectedDigit == 5u ? 14u : 12u;
        const std::uint32_t expectedInputs = expectedDigit == 5u ? 5u : 4u;
        if (ray.nodes.x != expectedNodeOffset ||
            ray.nodes.y != expectedNodes || ray.nodes.z != expectedSide ||
            ray.nodes.w != expectedDigit ||
            ray.elements.x != expectedElementOffset ||
            ray.elements.y != expectedElements ||
            ray.elements.z != expectedInputOffset ||
            ray.elements.w != expectedInputs ||
            ray.nodes.x > source.nodes.size() ||
            ray.nodes.y > source.nodes.size() - ray.nodes.x ||
            ray.elements.x > source.elements.size() ||
            ray.elements.y > source.elements.size() - ray.elements.x ||
            ray.elements.z > source.inputs.size() ||
            ray.elements.w > source.inputs.size() - ray.elements.z) {
            return false;
        }
        std::uint32_t fixedCount = 0u;
        for (std::uint32_t local = 0u; local < ray.nodes.y; ++local) {
            const auto& node = source.nodes[ray.nodes.x + local];
            if ((node.flags & ~MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u ||
                node.role != local || node.bodyIndex == MR_INVALID_INDEX ||
                !finite4(node.localPoint) || node.localPoint.w != 0.0f) {
                return false;
            }
            if ((node.flags & MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u)
                ++fixedCount;
        }
        if (fixedCount != 3u ||
            (source.nodes[ray.nodes.x].flags &
             MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) == 0u ||
            (source.nodes[ray.nodes.x + 1u].flags &
             MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) == 0u ||
            (source.nodes[ray.nodes.x + 2u].flags &
             MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) == 0u) return false;
        for (std::uint32_t local = 0u; local < ray.elements.y; ++local) {
            const auto& element = source.elements[ray.elements.x + local];
            if (element.nodeA < ray.nodes.x ||
                element.nodeA >= ray.nodes.x + ray.nodes.y ||
                element.nodeB < ray.nodes.x ||
                element.nodeB >= ray.nodes.x + ray.nodes.y ||
                element.nodeA == element.nodeB || element.reserved0 != 0u ||
                !finite4(element.material) ||
                !(element.material.x >= configuration.minimumLengthMeters) ||
                !(element.material.y > 0.0f) ||
                !(element.material.z > 0.0f) || element.material.w != 0.0f) {
                return false;
            }
        }
        for (std::uint32_t local = 0u; local < ray.elements.w; ++local) {
            const auto& input = source.inputs[ray.elements.z + local];
            if (input.nodeIndex < ray.nodes.x ||
                input.nodeIndex >= ray.nodes.x + ray.nodes.y ||
                input.muscleIndex == MR_INVALID_INDEX ||
                input.proximalBodyIndex == MR_INVALID_INDEX ||
                input.routeNodeOrdinal == MR_INVALID_INDEX ||
                input.targetRouteNodeOrdinal == MR_INVALID_INDEX ||
                input.routeNodeOrdinal >= input.targetRouteNodeOrdinal ||
                input.reserved0 != 0u || input.reserved1 != 0u ||
                input.reserved2 != 0u ||
                source.nodes[input.nodeIndex].sourceSiteIndex == MR_INVALID_INDEX ||
                !finite4(input.proximalLocalPoint) ||
                input.proximalLocalPoint.w != 0.0f ||
                (source.nodes[input.nodeIndex].flags &
                 MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED) != 0u) return false;
        }
        expectedNodeOffset += ray.nodes.y;
        expectedElementOffset += ray.elements.y;
        expectedInputOffset += ray.elements.w;
    }
    if (expectedNodeOffset != source.nodes.size() ||
        expectedElementOffset != source.elements.size() ||
        expectedInputOffset != source.inputs.size()) return false;

    auto candidate = std::make_unique<State>();
    candidate->rays.assign(source.rays.begin(), source.rays.end());
    candidate->nodes.assign(source.nodes.begin(), source.nodes.end());
    candidate->elements.assign(source.elements.begin(), source.elements.end());
    candidate->inputs.assign(source.inputs.begin(), source.inputs.end());
    candidate->routeCuts.reserve(source.inputs.size());
    for (const auto& input : source.inputs) {
        candidate->routeCuts.push_back({
            input.muscleIndex, input.routeNodeOrdinal, 0u, 0u,
        });
    }
    candidate->configuration = configuration;
    candidate->environmentCount = source.environmentCount;
    std::uint64_t fingerprint = 1469598103934665603ull;
    fingerprint = appendFingerprint(
        fingerprint, candidate->rays.data(),
        candidate->rays.size() * sizeof(candidate->rays.front()));
    fingerprint = appendFingerprint(
        fingerprint, candidate->nodes.data(),
        candidate->nodes.size() * sizeof(candidate->nodes.front()));
    fingerprint = appendFingerprint(
        fingerprint, candidate->elements.data(),
        candidate->elements.size() * sizeof(candidate->elements.front()));
    fingerprint = appendFingerprint(
        fingerprint, candidate->inputs.data(),
        candidate->inputs.size() * sizeof(candidate->inputs.front()));
    fingerprint = appendFingerprint(
        fingerprint, &candidate->environmentCount,
        sizeof(candidate->environmentCount));
    candidate->fingerprint = fingerprint == 0u ? 1u : fingerprint;
    candidate->message = "initialized";
    state_ = std::move(candidate);
    return true;
}

bool NumiHumanExtensorHoodMetalAdapter::encodePreDynamics(
    const MetalNumiHumanTendonLoadPass& pass
) {
    @autoreleasepool {
        if (state_ == nullptr || pass.commandBuffer == nullptr ||
            pass.mujocoMuscles == nullptr || pass.mujocoSites == nullptr ||
            pass.mujocoWraps == nullptr ||
            pass.mujocoRouteNodes == nullptr || pass.mujocoResults == nullptr ||
            pass.generalizedForces == nullptr || pass.bodyPoses == nullptr ||
            pass.pointJacobians == nullptr ||
            pass.environmentCount != state_->environmentCount ||
            pass.environmentCount == 0u || pass.muscleCount == 0u ||
            pass.siteCount == 0u || pass.wrapCount == 0u ||
            pass.routeNodeCount == 0u ||
            pass.dofCount == 0u || pass.bodyPoseStride == 0u ||
            pass.generalizedForceStride < pass.dofCount ||
            pass.pointJacobianStride == 0u ||
            pass.bodyJacobianPointOffset == MR_INVALID_INDEX ||
            pass.stepIndex >= MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS) {
            if (state_ != nullptr) state_->message = "invalid borrowed hood pass";
            return false;
        }
        for (const auto& node : state_->nodes) {
            if (node.bodyIndex < pass.articulationFirstBody ||
                node.bodyIndex - pass.articulationFirstBody >=
                    pass.bodyPoseStride ||
                (node.sourceSiteIndex != MR_INVALID_INDEX &&
                 node.sourceSiteIndex >= pass.siteCount)) {
                state_->message = "hood node/body binding drifted";
                return false;
            }
        }
        for (const auto& input : state_->inputs) {
            if (input.muscleIndex >= pass.muscleCount ||
                input.proximalBodyIndex < pass.articulationFirstBody ||
                input.proximalBodyIndex - pass.articulationFirstBody >=
                    pass.bodyPoseStride) {
                state_->message = "hood muscle/proximal binding drifted";
                return false;
            }
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> muscles = (__bridge id<MTLBuffer>)pass.mujocoMuscles;
        id<MTLBuffer> sites = (__bridge id<MTLBuffer>)pass.mujocoSites;
        id<MTLBuffer> wraps = (__bridge id<MTLBuffer>)pass.mujocoWraps;
        id<MTLBuffer> routes = (__bridge id<MTLBuffer>)pass.mujocoRouteNodes;
        id<MTLBuffer> results = (__bridge id<MTLBuffer>)pass.mujocoResults;
        id<MTLBuffer> generalized =
            (__bridge id<MTLBuffer>)pass.generalizedForces;
        id<MTLBuffer> bodyPoses = (__bridge id<MTLBuffer>)pass.bodyPoses;
        id<MTLBuffer> pointJacobians =
            (__bridge id<MTLBuffer>)pass.pointJacobians;
        if (command == nil || muscles == nil || sites == nil || wraps == nil ||
            routes == nil ||
            results == nil || generalized == nil || bodyPoses == nil ||
            pointJacobians == nil) {
            state_->message = "borrowed hood Metal objects are unavailable";
            return false;
        }
        const std::uint64_t muscleRowElements =
            static_cast<std::uint64_t>(pass.environmentCount) *
            pass.muscleCount * pass.dofCount;
        const std::uint64_t reducedEnd =
            static_cast<std::uint64_t>(pass.generalizedForceOffset) +
            static_cast<std::uint64_t>(pass.environmentCount - 1u) *
                pass.generalizedForceStride + pass.dofCount;
        if (pass.generalizedForceOffset < muscleRowElements ||
            std::max(muscleRowElements, reducedEnd) >
                generalized.length / sizeof(float)) {
            state_->message = "hood generalized-force arena is undersized";
            return false;
        }
        id<MTLDevice> device = generalized.device;
        if (device == nil || muscles.device.registryID != device.registryID ||
            sites.device.registryID != device.registryID ||
            wraps.device.registryID != device.registryID ||
            routes.device.registryID != device.registryID ||
            results.device.registryID != device.registryID ||
            bodyPoses.device.registryID != device.registryID ||
            pointJacobians.device.registryID != device.registryID) {
            state_->message = "hood borrowed buffers do not share one device";
            return false;
        }
        if (state_->device == nil) {
            state_->device = device;
            NSError* error = nil;
            const std::filesystem::path metallib =
                state_->configuration.metallib.empty()
                    ? std::filesystem::path(METALROBO_DEFAULT_METALLIB)
                    : state_->configuration.metallib;
            NSString* path = [NSString stringWithUTF8String:
                metallib.string().c_str()];
            state_->library = path == nil ? nil : [device
                newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            if (state_->library == nil) {
                state_->message = "hood Metal library did not load";
                return false;
            }
            const auto pipeline = [&](const char* name) {
                id<MTLFunction> function = [state_->library newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
                if (function == nil) return static_cast<id<MTLComputePipelineState>>(nil);
                return [device newComputePipelineStateWithFunction:function error:&error];
            };
            state_->solvePipeline = pipeline(
                "mr_numi_human_solve_extensor_hood");
            state_->routeCutPipeline = pipeline(
                "mr_mujoco_muscle_route_suffix_jacobian");
            state_->assemblePipeline = pipeline(
                "mr_numi_human_assemble_extensor_hood_correction");
            state_->applyPipeline = pipeline(
                "mr_numi_human_apply_extensor_hood_correction");
            state_->commitPipeline = pipeline(
                "mr_numi_human_commit_extensor_hood_audit");
            state_->rayBuffer = [device newBufferWithBytes:state_->rays.data()
                length:state_->rays.size() * sizeof(state_->rays.front())
                options:MTLResourceStorageModeShared];
            state_->nodeBuffer = [device newBufferWithBytes:state_->nodes.data()
                length:state_->nodes.size() * sizeof(state_->nodes.front())
                options:MTLResourceStorageModeShared];
            state_->elementBuffer = [device
                newBufferWithBytes:state_->elements.data()
                length:state_->elements.size() * sizeof(state_->elements.front())
                options:MTLResourceStorageModeShared];
            state_->inputBuffer = [device newBufferWithBytes:state_->inputs.data()
                length:state_->inputs.size() * sizeof(state_->inputs.front())
                options:MTLResourceStorageModeShared];
            state_->routeCutBuffer = [device
                newBufferWithBytes:state_->routeCuts.data()
                length:state_->routeCuts.size() *
                    sizeof(state_->routeCuts.front())
                options:MTLResourceStorageModeShared];
            state_->suffixJacobianBuffer = [device newBufferWithLength:
                static_cast<NSUInteger>(state_->environmentCount) *
                    state_->routeCuts.size() * pass.dofCount * sizeof(float)
                options:MTLResourceStorageModeShared];
            state_->nodeResultBuffer = [device newBufferWithLength:
                static_cast<NSUInteger>(state_->environmentCount) *
                    state_->nodes.size() *
                    sizeof(MRNumiHumanExtensorHoodNodeResultGPU)
                options:MTLResourceStorageModeShared];
            state_->rayResultBuffer = [device newBufferWithLength:
                static_cast<NSUInteger>(state_->environmentCount) *
                    state_->rays.size() *
                    sizeof(MRNumiHumanExtensorHoodRayResultGPU)
                options:MTLResourceStorageModeShared];
            state_->historyBuffer = [device newBufferWithLength:
                static_cast<NSUInteger>(state_->environmentCount) *
                    MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS *
                    state_->rays.size() *
                    sizeof(MRNumiHumanExtensorHoodRayResultGPU)
                options:MTLResourceStorageModeShared];
            state_->correctionBuffer = [device newBufferWithLength:
                static_cast<NSUInteger>(state_->environmentCount) *
                    pass.dofCount * sizeof(float)
                options:MTLResourceStorageModeShared];
            if (state_->solvePipeline == nil || state_->routeCutPipeline == nil ||
                state_->assemblePipeline == nil ||
                state_->applyPipeline == nil || state_->commitPipeline == nil ||
                state_->rayBuffer == nil || state_->nodeBuffer == nil ||
                state_->elementBuffer == nil || state_->inputBuffer == nil ||
                state_->routeCutBuffer == nil ||
                state_->suffixJacobianBuffer == nil ||
                state_->nodeResultBuffer == nil ||
                state_->rayResultBuffer == nil || state_->historyBuffer == nil ||
                state_->correctionBuffer == nil) {
                state_->message = "hood pipeline or buffer creation failed";
                return false;
            }
            std::memset(state_->historyBuffer.contents, 0,
                        state_->historyBuffer.length);
        } else if (state_->device.registryID != device.registryID ||
                   state_->correctionBuffer.length <
                       static_cast<NSUInteger>(pass.environmentCount) *
                           pass.dofCount * sizeof(float) ||
                   state_->suffixJacobianBuffer.length <
                       static_cast<NSUInteger>(pass.environmentCount) *
                           state_->routeCuts.size() * pass.dofCount *
                           sizeof(float)) {
            state_->message = "hood device or DoF layout changed";
            return false;
        }

        MRNumiHumanExtensorHoodDispatchGPU dispatch{};
        dispatch.abiVersion = MR_NUMI_HUMAN_EXTENSOR_HOOD_GPU_ABI_VERSION;
        dispatch.environmentCount = pass.environmentCount;
        dispatch.rayCount = static_cast<mr_u32>(state_->rays.size());
        dispatch.nodeCount = static_cast<mr_u32>(state_->nodes.size());
        dispatch.elementCount = static_cast<mr_u32>(state_->elements.size());
        dispatch.inputCount = static_cast<mr_u32>(state_->inputs.size());
        dispatch.muscleCount = pass.muscleCount;
        dispatch.siteCount = pass.siteCount;
        dispatch.routeNodeCount = pass.routeNodeCount;
        dispatch.dofCount = pass.dofCount;
        dispatch.bodyPoseStride = pass.bodyPoseStride;
        dispatch.articulationFirstBody = pass.articulationFirstBody;
        dispatch.pointJacobianStride = pass.pointJacobianStride;
        dispatch.bodyJacobianPointOffset = pass.bodyJacobianPointOffset;
        dispatch.generalizedForceStride = pass.generalizedForceStride;
        dispatch.generalizedForceOffset = pass.generalizedForceOffset;
        dispatch.stepIndex = pass.stepIndex;
        dispatch.maximumIterations = state_->configuration.maximumIterations;
        dispatch.maximumLineSearchSteps =
            state_->configuration.maximumLineSearchSteps;
        dispatch.wrapCount = pass.wrapCount;
        dispatch.solver = {
            state_->configuration.forceToleranceNewtons,
            state_->configuration.minimumLengthMeters,
            state_->configuration.diagonalRegularization,
            state_->configuration.armijoFraction,
        };
        dispatch.foundation = {
            state_->configuration.foundationStiffnessNewtonsPerMeter,
            0.0f, 0.0f, 0.0f,
        };
        const auto encode = [&](id<MTLComputePipelineState> pipeline,
                                const NSUInteger count, const auto& bind) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (encoder == nil) return false;
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            bind(encoder);
            const NSUInteger width = std::max<NSUInteger>(1u, std::min(
                count, std::min<NSUInteger>(
                    pipeline.maxTotalThreadsPerThreadgroup, 64u)));
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
            [encoder endEncoding];
            return true;
        };
        const NSUInteger rayCount = static_cast<NSUInteger>(
            pass.environmentCount) * state_->rays.size();
        const NSUInteger routeCutCount = static_cast<NSUInteger>(
            pass.environmentCount) * state_->routeCuts.size();
        const NSUInteger dofCount = static_cast<NSUInteger>(
            pass.environmentCount) * pass.dofCount;
        MRMujocoMuscleReferenceDispatchGPU routeDispatch{};
        routeDispatch.abiVersion =
            MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION;
        routeDispatch.muscleCount = pass.muscleCount;
        routeDispatch.siteCount = pass.siteCount;
        routeDispatch.wrapCount = pass.wrapCount;
        routeDispatch.routeNodeCount = pass.routeNodeCount;
        routeDispatch.environmentCount = pass.environmentCount;
        routeDispatch.bodyPoseStride = pass.bodyPoseStride;
        routeDispatch.articulationFirstBody = pass.articulationFirstBody;
        routeDispatch.dofCount = pass.dofCount;
        routeDispatch.pointJacobianStride = pass.pointJacobianStride;
        routeDispatch.bodyJacobianPointOffset =
            pass.bodyJacobianPointOffset;
        routeDispatch.bodyJacobianPointStride = 4u;
        MRMujocoMuscleRouteCutDispatchGPU routeCutDispatch{};
        routeCutDispatch.abiVersion =
            MR_MUJOCO_MUSCLE_ROUTE_CUT_GPU_ABI_VERSION;
        routeCutDispatch.cutCount = static_cast<mr_u32>(
            state_->routeCuts.size());
        id<MTLComputeCommandEncoder> routeCutEncoder =
            [command computeCommandEncoder];
        if (routeCutEncoder == nil) {
            state_->message = "hood source-route cut encoder is unavailable";
            return false;
        }
        [routeCutEncoder setComputePipelineState:state_->routeCutPipeline];
        [routeCutEncoder setBuffer:bodyPoses offset:0u atIndex:0u];
        [routeCutEncoder setBuffer:pointJacobians offset:0u atIndex:1u];
        [routeCutEncoder setBytes:&routeDispatch
                           length:sizeof(routeDispatch) atIndex:2u];
        [routeCutEncoder setBuffer:muscles offset:0u atIndex:3u];
        [routeCutEncoder setBuffer:sites offset:0u atIndex:4u];
        [routeCutEncoder setBuffer:wraps offset:0u atIndex:5u];
        [routeCutEncoder setBuffer:routes offset:0u atIndex:6u];
        [routeCutEncoder setBytes:&routeCutDispatch
                           length:sizeof(routeCutDispatch) atIndex:7u];
        [routeCutEncoder setBuffer:state_->routeCutBuffer
                            offset:0u atIndex:8u];
        [routeCutEncoder setBuffer:state_->suffixJacobianBuffer
                            offset:0u atIndex:9u];
        const NSUInteger routeCutWidth = std::max<NSUInteger>(1u, std::min(
            routeCutCount, std::min<NSUInteger>(
                state_->routeCutPipeline.maxTotalThreadsPerThreadgroup, 64u)));
        [routeCutEncoder dispatchThreads:MTLSizeMake(routeCutCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(routeCutWidth, 1u, 1u)];
        [routeCutEncoder endEncoding];
        if (!encode(state_->solvePipeline, rayCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->rayBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:state_->nodeBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:state_->elementBuffer offset:0u atIndex:3u];
                    [encoder setBuffer:state_->inputBuffer offset:0u atIndex:4u];
                    [encoder setBuffer:muscles offset:0u atIndex:5u];
                    [encoder setBuffer:sites offset:0u atIndex:6u];
                    [encoder setBuffer:routes offset:0u atIndex:7u];
                    [encoder setBuffer:results offset:0u atIndex:8u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:9u];
                    [encoder setBuffer:state_->nodeResultBuffer offset:0u atIndex:10u];
                    [encoder setBuffer:state_->rayResultBuffer offset:0u atIndex:11u];
                    [encoder setBuffer:wraps offset:0u atIndex:12u];
                }) ||
            !encode(state_->assemblePipeline, dofCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->rayBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:state_->nodeBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:state_->inputBuffer offset:0u atIndex:3u];
                    [encoder setBuffer:muscles offset:0u atIndex:4u];
                    [encoder setBuffer:sites offset:0u atIndex:5u];
                    [encoder setBuffer:routes offset:0u atIndex:6u];
                    [encoder setBuffer:results offset:0u atIndex:7u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:8u];
                    [encoder setBuffer:pointJacobians offset:0u atIndex:9u];
                    [encoder setBuffer:state_->nodeResultBuffer offset:0u atIndex:10u];
                    [encoder setBuffer:state_->rayResultBuffer offset:0u atIndex:11u];
                    [encoder setBuffer:state_->correctionBuffer offset:0u atIndex:12u];
                    [encoder setBuffer:state_->suffixJacobianBuffer
                                offset:0u atIndex:13u];
                }) ||
            !encode(state_->applyPipeline, dofCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->correctionBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:generalized offset:0u atIndex:2u];
                })) {
            state_->message = "hood pre-dynamics encoding failed";
            return false;
        }
        state_->lastStepIndex = pass.stepIndex;
        state_->message = "pre-dynamics encoded";
        return true;
    }
}

bool NumiHumanExtensorHoodMetalAdapter::encodePostValidation(
    const MetalNumiHumanTendonLoadPass& pass
) {
    @autoreleasepool {
        if (state_ == nullptr || state_->device == nil ||
            state_->commitPipeline == nil || pass.commandBuffer == nullptr ||
            pass.standStatuses == nullptr ||
            pass.environmentCount != state_->environmentCount ||
            pass.stepIndex != state_->lastStepIndex ||
            pass.stepIndex >= MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS) {
            if (state_ != nullptr) state_->message = "invalid hood commit pass";
            return false;
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> statuses = (__bridge id<MTLBuffer>)pass.standStatuses;
        if (command == nil || statuses == nil ||
            statuses.device.registryID != state_->device.registryID) {
            state_->message = "hood stand status is unavailable";
            return false;
        }
        MRNumiHumanExtensorHoodDispatchGPU dispatch{};
        dispatch.abiVersion = MR_NUMI_HUMAN_EXTENSOR_HOOD_GPU_ABI_VERSION;
        dispatch.environmentCount = pass.environmentCount;
        dispatch.rayCount = static_cast<mr_u32>(state_->rays.size());
        dispatch.nodeCount = static_cast<mr_u32>(state_->nodes.size());
        dispatch.elementCount = static_cast<mr_u32>(state_->elements.size());
        dispatch.inputCount = static_cast<mr_u32>(state_->inputs.size());
        dispatch.muscleCount = pass.muscleCount;
        dispatch.siteCount = pass.siteCount;
        dispatch.routeNodeCount = pass.routeNodeCount;
        dispatch.dofCount = pass.dofCount;
        dispatch.bodyPoseStride = pass.bodyPoseStride;
        dispatch.articulationFirstBody = pass.articulationFirstBody;
        dispatch.pointJacobianStride = pass.pointJacobianStride;
        dispatch.bodyJacobianPointOffset = pass.bodyJacobianPointOffset;
        dispatch.generalizedForceStride = pass.generalizedForceStride;
        dispatch.generalizedForceOffset = pass.generalizedForceOffset;
        dispatch.stepIndex = pass.stepIndex;
        dispatch.maximumIterations = state_->configuration.maximumIterations;
        dispatch.maximumLineSearchSteps =
            state_->configuration.maximumLineSearchSteps;
        dispatch.wrapCount = pass.wrapCount;
        dispatch.solver = {
            state_->configuration.forceToleranceNewtons,
            state_->configuration.minimumLengthMeters,
            state_->configuration.diagonalRegularization,
            state_->configuration.armijoFraction,
        };
        dispatch.foundation = {
            state_->configuration.foundationStiffnessNewtonsPerMeter,
            0.0f, 0.0f, 0.0f,
        };
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            state_->message = "hood audit encoder is unavailable";
            return false;
        }
        [encoder setComputePipelineState:state_->commitPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBuffer:statuses offset:0u atIndex:1u];
        [encoder setBuffer:state_->rayResultBuffer offset:0u atIndex:2u];
        [encoder setBuffer:state_->historyBuffer offset:0u atIndex:3u];
        const NSUInteger count = static_cast<NSUInteger>(pass.environmentCount) *
            state_->rays.size();
        const NSUInteger width = std::max<NSUInteger>(1u, std::min(
            count, std::min<NSUInteger>(
                state_->commitPipeline.maxTotalThreadsPerThreadgroup, 64u)));
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        [encoder endEncoding];
        ++state_->encodedPassCount;
        state_->message = "two-phase hood transaction encoded";
        return true;
    }
}

void NumiHumanExtensorHoodMetalAdapter::abort(void* commandBuffer) noexcept {
    if (state_ != nullptr && commandBuffer != nullptr) {
        ++state_->abortCount;
        state_->message = "enclosing command buffer aborted";
    }
}

MetalNumiHumanTendonLoadProgram
NumiHumanExtensorHoodMetalAdapter::program() noexcept {
    MetalNumiHumanTendonLoadProgram result{};
    if (state_ == nullptr || state_->fingerprint == 0u) return result;
    result.context = this;
    result.encodePreDynamics = [](
        void* context, const MetalNumiHumanTendonLoadPass& pass) {
        return context != nullptr &&
            static_cast<NumiHumanExtensorHoodMetalAdapter*>(context)->
                encodePreDynamics(pass);
    };
    result.encodePostValidation = [](
        void* context, const MetalNumiHumanTendonLoadPass& pass) {
        return context != nullptr &&
            static_cast<NumiHumanExtensorHoodMetalAdapter*>(context)->
                encodePostValidation(pass);
    };
    result.abort = [](void* context, void* commandBuffer) {
        if (context != nullptr) {
            static_cast<NumiHumanExtensorHoodMetalAdapter*>(context)->abort(
                commandBuffer);
        }
    };
    result.fingerprint = state_->fingerprint;
    return result;
}

NumiHumanExtensorHoodMetalDiagnostics
NumiHumanExtensorHoodMetalAdapter::diagnostics() const noexcept {
    NumiHumanExtensorHoodMetalDiagnostics output{};
    if (state_ == nullptr) return output;
    output.initialized = state_->fingerprint != 0u;
    output.encodedPassCount = state_->encodedPassCount;
    output.abortCount = state_->abortCount;
    output.fingerprint = state_->fingerprint;
    output.message = state_->message;
    if (state_->rayResultBuffer != nil) {
        const auto* results = static_cast<
            const MRNumiHumanExtensorHoodRayResultGPU*>(
                state_->rayResultBuffer.contents);
        const std::size_t count = static_cast<std::size_t>(
            state_->environmentCount) * state_->rays.size();
        for (std::size_t index = 0u; index < count; ++index) {
            if (results[index].status == MR_NUMI_HUMAN_EXTENSOR_HOOD_SUCCESS)
                ++output.successfulRayCount;
            else if (output.firstFailingRay == MR_INVALID_INDEX) {
                output.firstFailingRay = results[index].rayIndex;
                output.firstFailureStatus = results[index].status;
                output.firstFailureCompletedIterations =
                    results[index].completedIterations;
            }
            output.maximumFreeNodeResidualNewtons = std::max(
                output.maximumFreeNodeResidualNewtons,
                results[index].forceClosureAndMaximumResidual.w);
            output.maximumForceClosureResidualNewtons = std::max(
                output.maximumForceClosureResidualNewtons,
                std::hypot(
                    results[index].forceClosureAndMaximumResidual.x,
                    results[index].forceClosureAndMaximumResidual.y,
                    results[index].forceClosureAndMaximumResidual.z));
            output.maximumMomentClosureResidualNewtonMeters = std::max(
                output.maximumMomentClosureResidualNewtonMeters,
                std::hypot(
                    results[index].momentClosureAndMaximumTension.x,
                    results[index].momentClosureAndMaximumTension.y,
                    results[index].momentClosureAndMaximumTension.z));
            output.maximumTensionNewtons = std::max(
                output.maximumTensionNewtons,
                results[index].momentClosureAndMaximumTension.w);
            output.maximumFreeNodeDisplacementMeters = std::max(
                output.maximumFreeNodeDisplacementMeters,
                results[index].energyAndCounts.z);
            output.maximumEngineeringStrain = std::max(
                output.maximumEngineeringStrain,
                results[index].energyAndCounts.w);
        }
    }
    if (output.successfulRayCount == state_->rays.size() &&
        state_->nodeResultBuffer != nil) {
        const auto* nodes = static_cast<
            const MRNumiHumanExtensorHoodNodeResultGPU*>(
                state_->nodeResultBuffer.contents);
        output.solvedNodePositions.reserve(state_->nodes.size());
        for (std::size_t index = 0u; index < state_->nodes.size(); ++index)
            output.solvedNodePositions.push_back(nodes[index].position);
    }
    if (state_->historyBuffer != nil && state_->encodedPassCount > 0u) {
        const auto* history = static_cast<
            const MRNumiHumanExtensorHoodRayResultGPU*>(
                state_->historyBuffer.contents);
        for (std::uint32_t environment = 0u;
             environment < state_->environmentCount; ++environment) {
            const std::size_t base =
                (static_cast<std::size_t>(environment) *
                     MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS +
                 state_->lastStepIndex) * state_->rays.size();
            for (std::size_t ray = 0u; ray < state_->rays.size(); ++ray) {
                if (history[base + ray].transaction.x == 1.0f)
                    ++output.acceptedRayCount;
            }
        }
    }
    if (state_->correctionBuffer != nil) {
        const auto* corrections = static_cast<const float*>(
            state_->correctionBuffer.contents);
        const std::size_t count = state_->correctionBuffer.length / sizeof(float);
        for (std::size_t index = 0u; index < count; ++index) {
            if (std::isfinite(corrections[index])) {
                output.maximumGeneralizedCorrection = std::max(
                    output.maximumGeneralizedCorrection,
                    std::abs(corrections[index]));
            }
        }
    }
    return output;
}

} // namespace metalrobo
