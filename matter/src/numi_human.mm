#include "numi/matter/numi_human.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace numi::matter {
namespace {

std::uint64_t appendFingerprint(
    std::uint64_t fingerprint,
    const void* bytes,
    const std::size_t count
) noexcept {
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0u; index < count; ++index) {
        fingerprint ^= values[index];
        fingerprint *= 1099511628211ull;
    }
    return fingerprint;
}

bool finiteScale(const nm_float4& scale) noexcept {
    return std::isfinite(scale.x) && std::isfinite(scale.y) &&
        std::isfinite(scale.z) && std::isfinite(scale.w);
}

} // namespace

struct NumiHumanTendonFEMLoadAdapter::State {
    Runtime* runtime = nullptr;
    std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads;
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors;
    std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU> replacements;
    std::filesystem::path metallib;
    std::uint32_t endpointCount = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint64_t fingerprint = 0u;
    std::string message;

    __strong id<MTLDevice> device = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> provisionalStatusPipeline = nil;
    __strong id<MTLComputePipelineState> statusPipeline = nil;
    __strong id<MTLComputePipelineState> forcePipeline = nil;
    __strong id<MTLComputePipelineState> targetPipeline = nil;
    __strong id<MTLComputePipelineState> reactionPipeline = nil;
    __strong id<MTLBuffer> nodeLoadBuffer = nil;
    __strong id<MTLBuffer> nodeAnchorBuffer = nil;
    __strong id<MTLBuffer> replacementBuffer = nil;
    __strong id<MTLBuffer> externalForceBuffer = nil;
    __strong id<MTLBuffer> kinematicTargetBuffer = nil;
    __strong id<MTLBuffer> worldStatusBuffer = nil;
};

NumiHumanTendonFEMLoadAdapter::NumiHumanTendonFEMLoadAdapter() = default;
NumiHumanTendonFEMLoadAdapter::~NumiHumanTendonFEMLoadAdapter() = default;
NumiHumanTendonFEMLoadAdapter::NumiHumanTendonFEMLoadAdapter(
    NumiHumanTendonFEMLoadAdapter&&
) noexcept = default;
NumiHumanTendonFEMLoadAdapter& NumiHumanTendonFEMLoadAdapter::operator=(
    NumiHumanTendonFEMLoadAdapter&&
) noexcept = default;

bool NumiHumanTendonFEMLoadAdapter::initialize(
    Runtime& runtime,
    const NumiHumanTendonFEMLoadSource& source,
    const NumiHumanTendonFEMLoadConfiguration& configuration
) {
    if (!runtime.valid() || source.nodeLoads.empty() ||
        source.nodeAnchors.size() != source.nodeLoads.size() ||
        source.endpointReplacements.empty() ||
        source.endpointCount == 0u || source.environmentCount == 0u ||
        !std::isfinite(source.productionForceOwnerFraction) ||
        !(source.productionForceOwnerFraction > 0.0f) ||
        source.productionForceOwnerFraction > 0.25f ||
        configuration.metallib.empty() ||
        !std::filesystem::is_regular_file(configuration.metallib)) {
        return false;
    }
    std::vector<double> endpointScales(source.endpointCount, 0.0);
    std::uint32_t anchorCount = 0u;
    for (const NMNumiHumanTendonFEMNodeLoadGPU& load : source.nodeLoads) {
        if (!finiteScale(load.scale) || load.reserved0 != 0u ||
            load.reserved1 != 0u ||
            (load.flags & ~NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE) != 0u) {
            return false;
        }
        if ((load.flags & NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE) == 0u) {
            if (load.endpointIndex != NM_INVALID_INDEX || load.scale.x != 0.0f ||
                load.scale.y != 0.0f || load.scale.z != 0.0f ||
                load.scale.w != 0.0f) {
                return false;
            }
            continue;
        }
        if (load.endpointIndex >= source.endpointCount || load.scale.x < 0.0f) {
            return false;
        }
        endpointScales[load.endpointIndex] += load.scale.x;
    }
    for (const NMNumiHumanTendonFEMNodeAnchorGPU& anchor :
         source.nodeAnchors) {
        if (!finiteScale(anchor.localPoint) || anchor.reserved0 != 0u ||
            anchor.reserved1 != 0u ||
            (anchor.flags &
                ~NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) != 0u ||
            anchor.localPoint.w != 0.0f) {
            return false;
        }
        if ((anchor.flags &
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) == 0u) {
            if (anchor.bodyIndex != NM_INVALID_INDEX ||
                anchor.localPoint.x != 0.0f || anchor.localPoint.y != 0.0f ||
                anchor.localPoint.z != 0.0f) {
                return false;
            }
        } else {
            if (anchor.bodyIndex == NM_INVALID_INDEX) return false;
            ++anchorCount;
        }
    }
    std::vector<bool> loadEndpoints(source.endpointCount, false);
    std::vector<bool> anchorEndpoints(source.endpointCount, false);
    for (const NMNumiHumanTendonFEMEndpointReplacementGPU& replacement :
         source.endpointReplacements) {
        if (replacement.flags !=
                NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE ||
            replacement.reserved0 != 0u ||
            replacement.loadEndpointIndex >= source.endpointCount ||
            replacement.anchorEndpointIndex >= source.endpointCount ||
            replacement.loadEndpointIndex == replacement.anchorEndpointIndex ||
            !finiteScale(replacement.forceOwnerFraction) ||
            replacement.forceOwnerFraction.x <= 0.0f ||
            replacement.forceOwnerFraction.x > 0.25f ||
            replacement.forceOwnerFraction.y != 0.0f ||
            replacement.forceOwnerFraction.z != 0.0f ||
            replacement.forceOwnerFraction.w != 0.0f ||
            std::abs(replacement.forceOwnerFraction.x -
                source.productionForceOwnerFraction) > 1.0e-6f ||
            loadEndpoints[replacement.loadEndpointIndex] ||
            anchorEndpoints[replacement.anchorEndpointIndex]) {
            return false;
        }
        loadEndpoints[replacement.loadEndpointIndex] = true;
        anchorEndpoints[replacement.anchorEndpointIndex] = true;
        if (std::abs(endpointScales[replacement.loadEndpointIndex] -
                replacement.forceOwnerFraction.x) > 1.0e-6) {
            return false;
        }
    }
    if (anchorCount == 0u || std::any_of(
            endpointScales.begin(), endpointScales.end(),
            [](const double scale) {
                return !std::isfinite(scale) || scale > 0.250001;
            }
        ) || std::none_of(
            endpointScales.begin(), endpointScales.end(),
            [](const double scale) { return scale > 0.0; }
        )) {
        return false;
    }
    for (std::size_t endpointIndex = 0u;
         endpointIndex < endpointScales.size();
         ++endpointIndex) {
        if ((endpointScales[endpointIndex] > 0.0) !=
            loadEndpoints[endpointIndex]) {
            return false;
        }
    }
    auto candidate = std::make_unique<State>();
    candidate->runtime = &runtime;
    candidate->nodeLoads.assign(source.nodeLoads.begin(), source.nodeLoads.end());
    candidate->nodeAnchors.assign(
        source.nodeAnchors.begin(), source.nodeAnchors.end()
    );
    candidate->replacements.assign(
        source.endpointReplacements.begin(), source.endpointReplacements.end()
    );
    candidate->metallib = configuration.metallib;
    candidate->endpointCount = source.endpointCount;
    candidate->environmentCount = source.environmentCount;
    std::uint64_t fingerprint = 1469598103934665603ull;
    const std::uint64_t runtimeFingerprint = runtime.deviceProgramFingerprint();
    fingerprint = appendFingerprint(
        fingerprint, &runtimeFingerprint, sizeof(runtimeFingerprint)
    );
    fingerprint = appendFingerprint(
        fingerprint, &candidate->endpointCount, sizeof(candidate->endpointCount)
    );
    fingerprint = appendFingerprint(
        fingerprint, &candidate->environmentCount,
        sizeof(candidate->environmentCount)
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->nodeLoads.data(),
        candidate->nodeLoads.size() * sizeof(candidate->nodeLoads.front())
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->nodeAnchors.data(),
        candidate->nodeAnchors.size() * sizeof(candidate->nodeAnchors.front())
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->replacements.data(),
        candidate->replacements.size() * sizeof(candidate->replacements.front())
    );
    candidate->fingerprint = fingerprint == 0u ? 1u : fingerprint;
    candidate->message = "initialized";
    state_ = std::move(candidate);
    return true;
}

bool NumiHumanTendonFEMLoadAdapter::encodePreDynamics(
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    @autoreleasepool {
        if (state_ == nullptr || state_->runtime == nullptr ||
            pass.commandBuffer == nullptr || pass.bindings == nullptr ||
            pass.transfers == nullptr || pass.generalizedForces == nullptr ||
            pass.bodyPoses == nullptr || pass.pointJacobians == nullptr ||
            pass.environmentCount != state_->environmentCount ||
            pass.endpointCount != state_->endpointCount ||
            pass.environmentCount == 0u || pass.endpointCount == 0u ||
            pass.dofCount == 0u || pass.muscleCount == 0u ||
            pass.generalizedForceStride < pass.dofCount ||
            pass.pointJacobianStride == 0u ||
            pass.bodyJacobianPointOffset == MR_INVALID_INDEX ||
            pass.bodyPoseStride == 0u ||
            pass.stepIndex == std::numeric_limits<std::uint32_t>::max()) {
            if (state_ != nullptr)
                state_->message = "invalid borrowed Human pre-dynamics pass";
            return false;
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> bindings = (__bridge id<MTLBuffer>)pass.bindings;
        id<MTLBuffer> transfers = (__bridge id<MTLBuffer>)pass.transfers;
        id<MTLBuffer> generalizedForces =
            (__bridge id<MTLBuffer>)pass.generalizedForces;
        id<MTLBuffer> bodyPoses = (__bridge id<MTLBuffer>)pass.bodyPoses;
        id<MTLBuffer> pointJacobians =
            (__bridge id<MTLBuffer>)pass.pointJacobians;
        if (command == nil || bindings == nil || transfers == nil ||
            generalizedForces == nil || bodyPoses == nil ||
            pointJacobians == nil) {
            state_->message = "borrowed Human Metal objects are unavailable";
            return false;
        }
        id<MTLDevice> device = transfers.device;
        if (device == nil || bindings.device.registryID != device.registryID ||
            generalizedForces.device.registryID != device.registryID ||
            bodyPoses.device.registryID != device.registryID ||
            pointJacobians.device.registryID != device.registryID) {
            state_->message = "borrowed Human buffers do not share one Metal device";
            return false;
        }
        if (state_->device == nil) {
            state_->device = device;
            NSError* error = nil;
            NSString* path = [NSString
                stringWithUTF8String:state_->metallib.string().c_str()];
            state_->library = path == nil
                ? nil
                : [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                      error:&error];
            if (state_->library == nil) {
                state_->message = "Numi Matter metallib did not load for Human tendon/FEM assembly";
                return false;
            }
            const auto pipeline = [&](const char* name) {
                const std::string qualified =
                    std::string("numi_matter_metal::") + name;
                id<MTLFunction> function = [state_->library
                    newFunctionWithName:[NSString
                        stringWithUTF8String:qualified.c_str()]];
                if (function == nil) {
                    state_->message = std::string("missing Metal function ") + name;
                    return static_cast<id<MTLComputePipelineState>>(nil);
                }
                error = nil;
                id<MTLComputePipelineState> result = [device
                    newComputePipelineStateWithFunction:function
                                                   error:&error];
                if (result == nil) {
                    const char* description = error == nil
                        ? "unknown Metal error"
                        : error.localizedDescription.UTF8String;
                    state_->message = std::string("failed to compile pipeline ") +
                        name + ": " + (description == nullptr
                            ? "unknown Metal error"
                            : description);
                }
                return result;
            };
            state_->provisionalStatusPipeline = pipeline(
                "nm_numi_human_prepare_provisional_status");
            state_->statusPipeline = pipeline("nm_numi_human_adapt_stand_status");
            state_->forcePipeline = pipeline(
                "nm_numi_human_assemble_tendon_fem_loads");
            state_->targetPipeline = pipeline(
                "nm_numi_human_assemble_fem_kinematic_targets");
            state_->reactionPipeline = pipeline(
                "nm_numi_human_apply_fem_anchor_reactions");
            const NSUInteger nodeLoadBytes = static_cast<NSUInteger>(
                state_->nodeLoads.size() * sizeof(state_->nodeLoads.front())
            );
            const NSUInteger nodeAnchorBytes = static_cast<NSUInteger>(
                state_->nodeAnchors.size() * sizeof(state_->nodeAnchors.front())
            );
            const NSUInteger replacementBytes = static_cast<NSUInteger>(
                state_->replacements.size() * sizeof(state_->replacements.front())
            );
            const NSUInteger externalForceBytes = static_cast<NSUInteger>(
                state_->environmentCount * state_->nodeLoads.size() *
                sizeof(nm_float4)
            );
            const NSUInteger statusBytes = static_cast<NSUInteger>(
                state_->environmentCount * sizeof(MRMetalWorldStatusGPU)
            );
            state_->nodeLoadBuffer = [device
                newBufferWithBytes:state_->nodeLoads.data()
                length:nodeLoadBytes
                options:MTLResourceStorageModeShared];
            state_->nodeAnchorBuffer = [device
                newBufferWithBytes:state_->nodeAnchors.data()
                length:nodeAnchorBytes
                options:MTLResourceStorageModeShared];
            state_->replacementBuffer = [device
                newBufferWithBytes:state_->replacements.data()
                length:replacementBytes
                options:MTLResourceStorageModeShared];
            state_->externalForceBuffer = [device
                newBufferWithLength:externalForceBytes
                options:MTLResourceStorageModePrivate];
            state_->kinematicTargetBuffer = [device
                newBufferWithLength:externalForceBytes
                options:MTLResourceStorageModePrivate];
            state_->worldStatusBuffer = [device
                newBufferWithLength:statusBytes
                options:MTLResourceStorageModeShared];
            if (state_->provisionalStatusPipeline == nil ||
                state_->statusPipeline == nil || state_->forcePipeline == nil ||
                state_->targetPipeline == nil || state_->reactionPipeline == nil) {
                if (state_->message == "initialized") {
                    state_->message = "Human tendon/FEM pipeline is unavailable";
                }
                return false;
            }
            if (state_->nodeLoadBuffer == nil || state_->nodeAnchorBuffer == nil ||
                state_->replacementBuffer == nil) {
                state_->message = "Human tendon/FEM immutable mapping buffer is unavailable";
                return false;
            }
            if (state_->externalForceBuffer == nil ||
                state_->kinematicTargetBuffer == nil) {
                state_->message = "Human tendon/FEM force/target buffer is unavailable";
                return false;
            }
            if (state_->worldStatusBuffer == nil) {
                state_->message = "Human tendon/FEM world-status buffer is unavailable";
                return false;
            }
        } else if (state_->device.registryID != device.registryID) {
            state_->message = "Human tendon/FEM adapter changed Metal devices";
            return false;
        }

        const NMNumiHumanTendonFEMLoadDispatchGPU dispatch{
            .abiVersion = NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION,
            .environmentCount = state_->environmentCount,
            .femNodeCount = static_cast<std::uint32_t>(state_->nodeLoads.size()),
            .endpointCount = state_->endpointCount,
            .transferStride = pass.endpointCount,
            .stepIndex = pass.stepIndex,
            .replacementCount = static_cast<std::uint32_t>(
                state_->replacements.size()),
            .dofCount = pass.dofCount,
            .bodyPoseStride = pass.bodyPoseStride,
            .articulationFirstBody = pass.articulationFirstBody,
            .pointJacobianStride = pass.pointJacobianStride,
            .bodyJacobianPointOffset = pass.bodyJacobianPointOffset,
            .generalizedForceStride = pass.generalizedForceStride,
            .generalizedForceOffset = pass.generalizedForceOffset,
            .reserved0 = 0u,
            .reserved1 = 0u,
        };
        const auto encodeKernel = [&](id<MTLComputePipelineState> pipeline,
                                      const NSUInteger count,
                                      const auto& bind) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (encoder == nil) return false;
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            bind(encoder);
            const NSUInteger width = std::min(
                count,
                std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256u)
            );
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(std::max<NSUInteger>(width, 1u), 1u, 1u)];
            [encoder endEncoding];
            return true;
        };
        if (!encodeKernel(
                state_->provisionalStatusPipeline, state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeLoadBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:state_->replacementBuffer offset:0u atIndex:3u];
                    [encoder setBuffer:bindings offset:0u atIndex:4u];
                    [encoder setBuffer:transfers offset:0u atIndex:5u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:6u];
                    [encoder setBuffer:state_->worldStatusBuffer
                                offset:0u atIndex:7u];
                }
            ) || !encodeKernel(
                state_->forcePipeline,
                state_->environmentCount * state_->nodeLoads.size(),
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeLoadBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:transfers offset:0u atIndex:2u];
                    [encoder setBuffer:state_->externalForceBuffer
                                offset:0u atIndex:3u];
                }
            ) || !encodeKernel(
                state_->targetPipeline,
                state_->environmentCount * state_->nodeAnchors.size(),
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
                    [encoder setBuffer:state_->kinematicTargetBuffer
                                offset:0u atIndex:3u];
                }
            )) {
            state_->message = "Human tendon/FEM assembly kernel encoding failed";
            return false;
        }

        EncodeRequest request{};
        request.commandBuffer = pass.commandBuffer;
        request.environmentStatuses =
            (__bridge void*)state_->worldStatusBuffer;
        request.femExternalForces =
            (__bridge void*)state_->externalForceBuffer;
        request.femExternalForceCount = static_cast<std::uint32_t>(
            state_->environmentCount * state_->nodeLoads.size()
        );
        request.femKinematicTargets =
            (__bridge void*)state_->kinematicTargetBuffer;
        request.femKinematicTargetCount = request.femExternalForceCount;
        request.controlStep = pass.stepIndex;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = state_->runtime->timestepSeconds();
        request.phase = EncodePhase::preDynamics;
        const auto pre = state_->runtime->encode(request);
        if (!pre.encoded) {
            state_->message = "Human tendon/FEM Matter pre-dynamics failed: " + pre.message;
            return false;
        }
        id<MTLBuffer> reactions = (__bridge id<MTLBuffer>)
            state_->runtime->femConstraintReactionBuffer();
        id<MTLBuffer> matterStatuses =
            (__bridge id<MTLBuffer>)state_->runtime->statusBuffer();
        if (reactions == nil || matterStatuses == nil ||
            !encodeKernel(
                state_->reactionPipeline,
                state_->environmentCount * pass.dofCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:state_->replacementBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:bindings offset:0u atIndex:3u];
                    [encoder setBuffer:transfers offset:0u atIndex:4u];
                    [encoder setBuffer:reactions offset:0u atIndex:5u];
                    [encoder setBuffer:matterStatuses offset:0u atIndex:6u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:7u];
                    [encoder setBuffer:pointJacobians offset:0u atIndex:8u];
                    [encoder setBuffer:generalizedForces offset:0u atIndex:9u];
                }
            )) {
            state_->message = "Human tendon/FEM anchor-reaction encoding failed";
            return false;
        }
        state_->message = "pre-dynamics encoded";
        return true;
    }
}

bool NumiHumanTendonFEMLoadAdapter::encodePostValidation(
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    @autoreleasepool {
        if (state_ == nullptr || state_->runtime == nullptr ||
            state_->device == nil || state_->statusPipeline == nil ||
            pass.commandBuffer == nullptr || pass.standStatuses == nullptr ||
            pass.environmentCount != state_->environmentCount ||
            pass.endpointCount != state_->endpointCount ||
            pass.stepIndex == std::numeric_limits<std::uint32_t>::max()) {
            if (state_ != nullptr)
                state_->message = "invalid borrowed Human post-validation pass";
            return false;
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> standStatuses =
            (__bridge id<MTLBuffer>)pass.standStatuses;
        if (command == nil || standStatuses == nil ||
            standStatuses.device.registryID != state_->device.registryID) {
            state_->message = "borrowed Human stand status is unavailable";
            return false;
        }
        const NMNumiHumanTendonFEMLoadDispatchGPU dispatch{
            .abiVersion = NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION,
            .environmentCount = state_->environmentCount,
            .femNodeCount = static_cast<std::uint32_t>(state_->nodeLoads.size()),
            .endpointCount = state_->endpointCount,
            .transferStride = pass.endpointCount,
            .stepIndex = pass.stepIndex,
            .replacementCount = static_cast<std::uint32_t>(
                state_->replacements.size()),
            .dofCount = pass.dofCount,
            .bodyPoseStride = pass.bodyPoseStride,
            .articulationFirstBody = pass.articulationFirstBody,
            .pointJacobianStride = pass.pointJacobianStride,
            .bodyJacobianPointOffset = pass.bodyJacobianPointOffset,
            .generalizedForceStride = pass.generalizedForceStride,
            .generalizedForceOffset = pass.generalizedForceOffset,
            .reserved0 = 0u,
            .reserved1 = 0u,
        };
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            state_->message = "Human tendon/FEM status encoder is unavailable";
            return false;
        }
        [encoder setComputePipelineState:state_->statusPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBuffer:standStatuses offset:0u atIndex:1u];
        [encoder setBuffer:state_->worldStatusBuffer offset:0u atIndex:2u];
        const NSUInteger count = state_->environmentCount;
        const NSUInteger width = std::min<NSUInteger>(
            count, std::min<NSUInteger>(
                state_->statusPipeline.maxTotalThreadsPerThreadgroup, 256u));
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(
                std::max<NSUInteger>(width, 1u), 1u, 1u)];
        [encoder endEncoding];

        EncodeRequest request{};
        request.commandBuffer = pass.commandBuffer;
        request.environmentStatuses = (__bridge void*)state_->worldStatusBuffer;
        request.femExternalForces = (__bridge void*)state_->externalForceBuffer;
        request.femExternalForceCount = static_cast<std::uint32_t>(
            state_->environmentCount * state_->nodeLoads.size());
        request.femKinematicTargets =
            (__bridge void*)state_->kinematicTargetBuffer;
        request.femKinematicTargetCount = request.femExternalForceCount;
        request.controlStep = pass.stepIndex;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = state_->runtime->timestepSeconds();
        request.phase = EncodePhase::postCommit;
        const auto post = state_->runtime->encode(request);
        if (!post.encoded) {
            state_->message =
                "Human tendon/FEM Matter post-commit failed: " + post.message;
            return false;
        }
        ++state_->encodedPassCount;
        state_->message = "two-way transaction encoded";
        return true;
    }
}

void NumiHumanTendonFEMLoadAdapter::abort(void* commandBuffer) noexcept {
    if (state_ != nullptr && state_->runtime != nullptr &&
        commandBuffer != nullptr) {
        ++state_->abortCount;
        state_->runtime->cancel(commandBuffer);
    }
}

metalrobo::MetalNumiHumanTendonLoadProgram
NumiHumanTendonFEMLoadAdapter::program() noexcept {
    metalrobo::MetalNumiHumanTendonLoadProgram result{};
    if (state_ == nullptr || state_->fingerprint == 0u) return result;
    result.context = this;
    result.encodePreDynamics = [](void* context,
                                  const metalrobo::MetalNumiHumanTendonLoadPass& pass) {
        return context != nullptr &&
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->
                encodePreDynamics(pass);
    };
    result.encodePostValidation = [](
        void* context,
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    ) {
        return context != nullptr &&
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->
                encodePostValidation(pass);
    };
    result.abort = [](void* context, void* commandBuffer) {
        if (context != nullptr) {
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->abort(
                commandBuffer
            );
        }
    };
    result.fingerprint = state_->fingerprint;
    return result;
}

NumiHumanTendonFEMLoadDiagnostics
NumiHumanTendonFEMLoadAdapter::diagnostics() const noexcept {
    NumiHumanTendonFEMLoadDiagnostics result{};
    if (state_ == nullptr) return result;
    result.initialized = state_->runtime != nullptr && state_->fingerprint != 0u;
    result.encodedPassCount = state_->encodedPassCount;
    result.abortCount = state_->abortCount;
    result.fingerprint = state_->fingerprint;
    result.message = state_->message;
    return result;
}

} // namespace numi::matter
