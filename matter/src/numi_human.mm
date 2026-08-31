#include "numi/matter/numi_human.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
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

float scaleComponent(
    const nm_float4& scale, const std::uint32_t index
) noexcept {
    switch (index) {
        case 0u: return scale.x;
        case 1u: return scale.y;
        case 2u: return scale.z;
        case 3u: return scale.w;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

} // namespace

struct NumiHumanTendonFEMLoadAdapter::State {
    Runtime* runtime = nullptr;
    std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads;
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors;
    std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU> replacements;
    std::vector<NMNumiHumanFEMContactSampleGPU> contactSamples;
    std::vector<NMNumiHumanFEMContactContributionGPU> contactContributions;
    std::vector<NMIncidenceRangeGPU> contactRanges;
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
    __strong id<MTLComputePipelineState> forceAuditPipeline = nil;
    __strong id<MTLComputePipelineState> contactForcePipeline = nil;
    __strong id<MTLComputePipelineState> targetPipeline = nil;
    __strong id<MTLComputePipelineState> reactionPipeline = nil;
    __strong id<MTLBuffer> nodeLoadBuffer = nil;
    __strong id<MTLBuffer> nodeAnchorBuffer = nil;
    __strong id<MTLBuffer> replacementBuffer = nil;
    __strong id<MTLBuffer> contactSampleBuffer = nil;
    __strong id<MTLBuffer> contactContributionBuffer = nil;
    __strong id<MTLBuffer> contactRangeBuffer = nil;
    __strong id<MTLBuffer> externalForceBuffer = nil;
    __strong id<MTLBuffer> externalForceAuditBuffer = nil;
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
    const bool passiveAttachmentOnly = source.endpointReplacements.empty();
    const bool hasContact = !source.contactSamples.empty();
    if (!runtime.valid() || source.nodeLoads.empty() ||
        source.nodeAnchors.size() != source.nodeLoads.size() ||
        source.endpointCount == 0u || source.environmentCount == 0u ||
        !std::isfinite(source.productionForceOwnerFraction) ||
        (passiveAttachmentOnly
            ? source.productionForceOwnerFraction != 0.0f
            : !(source.productionForceOwnerFraction > 0.0f)) ||
        source.productionForceOwnerFraction > 1.0f ||
        configuration.metallib.empty() ||
        !std::filesystem::is_regular_file(configuration.metallib) ||
        (hasContact != !source.contactContributions.empty()) ||
        (hasContact != !source.contactRanges.empty()) ||
        (hasContact &&
            (source.contactRanges.size() != source.nodeLoads.size() ||
             source.contactSamples.size() >
                 std::numeric_limits<std::uint32_t>::max() ||
             source.contactContributions.size() >
                 std::numeric_limits<std::uint32_t>::max() ||
             source.contactSamples.size() >
                 source.contactContributions.size() / 4u))) {
        return false;
    }
    if (hasContact) {
        std::vector<std::array<std::uint32_t, 4u>> roleCounts(
            source.contactSamples.size());
        std::uint64_t nextContribution = 0u;
        for (std::uint32_t node = 0u;
             node < source.contactRanges.size(); ++node) {
            const NMIncidenceRangeGPU range = source.contactRanges[node];
            if (range.first != nextContribution ||
                static_cast<std::uint64_t>(range.first) + range.count >
                    source.contactContributions.size()) return false;
            for (std::uint32_t local = 0u; local < range.count; ++local) {
                const auto& contribution = source.contactContributions[
                    range.first + local];
                if (contribution.sampleIndex >= source.contactSamples.size() ||
                    contribution.role > 3u || contribution.reserved0 != 0u ||
                    contribution.reserved1 != 0u) return false;
                const auto& sample =
                    source.contactSamples[contribution.sampleIndex];
                std::uint32_t owner = sample.slaveNode;
                switch (contribution.role) {
                    case 1u: owner = sample.masterNode0; break;
                    case 2u: owner = sample.masterNode1; break;
                    case 3u: owner = sample.masterNode2; break;
                    default: break;
                }
                if (owner != node ||
                    ++roleCounts[contribution.sampleIndex][contribution.role] !=
                        1u) return false;
            }
            nextContribution += range.count;
        }
        if (nextContribution != source.contactContributions.size() ||
            source.contactContributions.size() !=
                4u * source.contactSamples.size()) return false;
        for (std::uint32_t index = 0u;
             index < source.contactSamples.size(); ++index) {
            const auto& sample = source.contactSamples[index];
            const auto& roles = roleCounts[index];
            const nm_float4 bary =
                sample.barycentricAndReferenceSeparation;
            const nm_float4 normal = sample.normalAndArea;
            const nm_float4 stiffness = sample.stiffness;
            const double normalLength = std::sqrt(
                normal.x * normal.x + normal.y * normal.y +
                normal.z * normal.z);
            if (sample.slaveNode >= source.nodeLoads.size() ||
                sample.masterNode0 >= source.nodeLoads.size() ||
                sample.masterNode1 >= source.nodeLoads.size() ||
                sample.masterNode2 >= source.nodeLoads.size() ||
                sample.slaveNode == sample.masterNode0 ||
                sample.slaveNode == sample.masterNode1 ||
                sample.slaveNode == sample.masterNode2 ||
                sample.masterNode0 == sample.masterNode1 ||
                sample.masterNode0 == sample.masterNode2 ||
                sample.masterNode1 == sample.masterNode2 ||
                !finiteScale(bary) || bary.x < 0.0f || bary.y < 0.0f ||
                bary.z < 0.0f ||
                std::abs(bary.x + bary.y + bary.z - 1.0f) > 1.0e-5f ||
                !finiteScale(normal) ||
                std::abs(normalLength - 1.0) > 1.0e-5 ||
                normal.w <= 0.0f || !finiteScale(stiffness) ||
                stiffness.x <= 0.0f || stiffness.y != 0.0f ||
                stiffness.z != 0.0f || stiffness.w != 0.0f ||
                std::any_of(roles.begin(), roles.end(),
                            [](const std::uint32_t count) {
                                return count != 1u;
                            })) return false;
        }
    }
    std::vector<double> endpointSignedScales(source.endpointCount, 0.0);
    std::vector<double> endpointAbsoluteScales(source.endpointCount, 0.0);
    std::uint32_t anchorCount = 0u;
    for (const NMNumiHumanTendonFEMNodeLoadGPU& load : source.nodeLoads) {
        if (!finiteScale(load.scale)) return false;
        for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
            const std::uint32_t endpoint = load.endpointIndex[slot];
            const float scale = scaleComponent(load.scale, slot);
            if ((endpoint == NM_INVALID_INDEX) != (scale == 0.0f) ||
                (endpoint != NM_INVALID_INDEX && endpoint >= source.endpointCount)) {
                return false;
            }
            for (std::uint32_t previous = 0u; previous < slot; ++previous) {
                if (endpoint != NM_INVALID_INDEX &&
                    load.endpointIndex[previous] == endpoint) {
                    return false;
                }
            }
            if (endpoint != NM_INVALID_INDEX) {
                endpointSignedScales[endpoint] += scale;
                endpointAbsoluteScales[endpoint] += std::abs(scale);
            }
        }
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
    std::vector<bool> expectedLoadedEndpoints(source.endpointCount, false);
    for (const NMNumiHumanTendonFEMEndpointReplacementGPU& replacement :
         source.endpointReplacements) {
        constexpr std::uint32_t replacementFlagMask =
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE |
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW |
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE;
        const bool fullMuscleRow = (replacement.flags &
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW) != 0u;
        const bool distalForceCouple = (replacement.flags &
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE) != 0u;
        if ((replacement.flags &
                NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE) == 0u ||
            (replacement.flags & ~replacementFlagMask) != 0u ||
            (distalForceCouple && !fullMuscleRow) ||
            replacement.reserved0 != 0u ||
            replacement.loadEndpointIndex >= source.endpointCount ||
            replacement.anchorEndpointIndex >= source.endpointCount ||
            replacement.loadEndpointIndex == replacement.anchorEndpointIndex ||
            !finiteScale(replacement.forceOwnerFraction) ||
            replacement.forceOwnerFraction.x <= 0.0f ||
            replacement.forceOwnerFraction.x > 1.0f ||
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
        expectedLoadedEndpoints[replacement.loadEndpointIndex] = true;
        if (distalForceCouple) {
            expectedLoadedEndpoints[replacement.anchorEndpointIndex] = true;
        }
        const double owner = replacement.forceOwnerFraction.x;
        if (std::abs(
                endpointSignedScales[replacement.loadEndpointIndex] - owner) >
                1.0e-6 ||
            std::abs(
                endpointAbsoluteScales[replacement.loadEndpointIndex] - owner) >
                1.0e-6 ||
            std::abs(endpointSignedScales[
                replacement.anchorEndpointIndex]) > 1.0e-6 ||
            std::abs(endpointAbsoluteScales[
                replacement.anchorEndpointIndex] -
                (distalForceCouple ? 2.0 * owner : 0.0)) > 1.0e-6) {
            return false;
        }
    }
    if (anchorCount == 0u || std::any_of(
            endpointAbsoluteScales.begin(), endpointAbsoluteScales.end(),
            [](const double scale) {
                return !std::isfinite(scale) || scale > 2.000001;
            }
        ) || (passiveAttachmentOnly
            ? std::any_of(
                endpointAbsoluteScales.begin(), endpointAbsoluteScales.end(),
                [](const double scale) { return scale != 0.0; })
            : std::none_of(
                endpointAbsoluteScales.begin(), endpointAbsoluteScales.end(),
                [](const double scale) { return scale > 0.0; }))) {
        return false;
    }
    for (std::size_t endpointIndex = 0u;
         endpointIndex < endpointAbsoluteScales.size();
         ++endpointIndex) {
        if ((endpointAbsoluteScales[endpointIndex] > 0.0) !=
            expectedLoadedEndpoints[endpointIndex]) {
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
    candidate->contactSamples.assign(
        source.contactSamples.begin(), source.contactSamples.end());
    candidate->contactContributions.assign(
        source.contactContributions.begin(),
        source.contactContributions.end());
    candidate->contactRanges.assign(
        source.contactRanges.begin(), source.contactRanges.end());
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
    fingerprint = appendFingerprint(
        fingerprint, candidate->contactSamples.data(),
        candidate->contactSamples.size() *
            sizeof(NMNumiHumanFEMContactSampleGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->contactContributions.data(),
        candidate->contactContributions.size() *
            sizeof(NMNumiHumanFEMContactContributionGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->contactRanges.data(),
        candidate->contactRanges.size() * sizeof(NMIncidenceRangeGPU));
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
        const std::uint64_t muscleRowElements =
            static_cast<std::uint64_t>(pass.environmentCount) *
            pass.muscleCount * pass.dofCount;
        const std::uint64_t reducedForceEnd =
            static_cast<std::uint64_t>(pass.generalizedForceOffset) +
            static_cast<std::uint64_t>(pass.environmentCount - 1u) *
                pass.generalizedForceStride +
            pass.dofCount;
        const std::uint64_t requiredForceElements = std::max(
            muscleRowElements, reducedForceEnd
        );
        if (pass.generalizedForceOffset < muscleRowElements ||
            requiredForceElements > generalizedForces.length / sizeof(float)) {
            state_->message =
                "borrowed Human generalized-force workspace is undersized";
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
            state_->forceAuditPipeline = pipeline(
                "nm_numi_human_audit_tendon_fem_loads");
            if (!state_->contactSamples.empty()) {
                state_->contactForcePipeline = pipeline(
                    "nm_numi_human_assemble_internal_fem_contact_loads");
            }
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
            state_->replacementBuffer = replacementBytes == 0u
                ? [device
                    newBufferWithLength:sizeof(
                        NMNumiHumanTendonFEMEndpointReplacementGPU)
                    options:MTLResourceStorageModeShared]
                : [device
                    newBufferWithBytes:state_->replacements.data()
                    length:replacementBytes
                    options:MTLResourceStorageModeShared];
            if (!state_->contactSamples.empty()) {
                state_->contactSampleBuffer = [device
                    newBufferWithBytes:state_->contactSamples.data()
                    length:state_->contactSamples.size() *
                        sizeof(NMNumiHumanFEMContactSampleGPU)
                    options:MTLResourceStorageModeShared];
                state_->contactContributionBuffer = [device
                    newBufferWithBytes:state_->contactContributions.data()
                    length:state_->contactContributions.size() *
                        sizeof(NMNumiHumanFEMContactContributionGPU)
                    options:MTLResourceStorageModeShared];
                state_->contactRangeBuffer = [device
                    newBufferWithBytes:state_->contactRanges.data()
                    length:state_->contactRanges.size() *
                        sizeof(NMIncidenceRangeGPU)
                    options:MTLResourceStorageModeShared];
            }
            state_->externalForceBuffer = [device
                newBufferWithLength:externalForceBytes
                options:MTLResourceStorageModePrivate];
            state_->externalForceAuditBuffer = [device
                newBufferWithLength:state_->environmentCount * sizeof(nm_float4)
                options:MTLResourceStorageModeShared];
            state_->kinematicTargetBuffer = [device
                newBufferWithLength:externalForceBytes
                options:MTLResourceStorageModePrivate];
            state_->worldStatusBuffer = [device
                newBufferWithLength:statusBytes
                options:MTLResourceStorageModeShared];
            if (state_->provisionalStatusPipeline == nil ||
                state_->statusPipeline == nil || state_->forcePipeline == nil ||
                state_->forceAuditPipeline == nil ||
                (!state_->contactSamples.empty() &&
                 state_->contactForcePipeline == nil) ||
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
            if (!state_->contactSamples.empty() &&
                (state_->contactSampleBuffer == nil ||
                 state_->contactContributionBuffer == nil ||
                 state_->contactRangeBuffer == nil)) {
                state_->message =
                    "Human FEM contact immutable mapping buffer is unavailable";
                return false;
            }
            if (state_->externalForceBuffer == nil ||
                state_->externalForceAuditBuffer == nil ||
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
            .muscleCount = pass.muscleCount,
            .contactSampleCount = static_cast<std::uint32_t>(
                state_->contactSamples.size()),
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
            )) {
            state_->message = "Human tendon/FEM assembly kernel encoding failed";
            return false;
        }
        if (!state_->contactSamples.empty()) {
            id<MTLBuffer> acceptedNodes = (__bridge id<MTLBuffer>)
                state_->runtime->femAcceptedNodeBuffer();
            const std::uint64_t expectedAcceptedNodeBytes =
                static_cast<std::uint64_t>(state_->environmentCount) *
                state_->nodeLoads.size() * sizeof(NMFEMNodeStateGPU);
            if (acceptedNodes == nil ||
                acceptedNodes.device.registryID != state_->device.registryID ||
                acceptedNodes.length < expectedAcceptedNodeBytes ||
                !encodeKernel(
                    state_->contactForcePipeline,
                    state_->environmentCount * state_->nodeLoads.size(),
                    [&](id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:state_->contactSampleBuffer
                                    offset:0u atIndex:1u];
                        [encoder setBuffer:state_->contactContributionBuffer
                                    offset:0u atIndex:2u];
                        [encoder setBuffer:state_->contactRangeBuffer
                                    offset:0u atIndex:3u];
                        [encoder setBuffer:acceptedNodes offset:0u atIndex:4u];
                        [encoder setBuffer:state_->externalForceBuffer
                                    offset:0u atIndex:5u];
                    })) {
                state_->message =
                    "Human internal FEM contact kernel encoding failed";
                return false;
            }
        }
        if (!encodeKernel(
                state_->forceAuditPipeline, state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->externalForceBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:state_->externalForceAuditBuffer
                                offset:0u atIndex:2u];
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
            pass.dofCount == 0u || pass.muscleCount == 0u ||
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
            .muscleCount = pass.muscleCount,
            .contactSampleCount = static_cast<std::uint32_t>(
                state_->contactSamples.size()),
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
    result.contactSampleCount = static_cast<std::uint32_t>(
        state_->contactSamples.size());
    if (state_->externalForceAuditBuffer != nil) {
        const auto* audits = static_cast<const nm_float4*>(
            state_->externalForceAuditBuffer.contents);
        double resultantX = 0.0;
        double resultantY = 0.0;
        double resultantZ = 0.0;
        for (std::uint32_t environment = 0u;
             environment < state_->environmentCount; ++environment) {
            const nm_float4 audit = audits[environment];
            if (!finiteScale(audit) || audit.w < 0.0f) {
                result.assembledExternalForceL1Newtons =
                    std::numeric_limits<double>::quiet_NaN();
                result.assembledExternalForceResultantNewtons =
                    std::numeric_limits<double>::quiet_NaN();
                break;
            }
            resultantX += audit.x;
            resultantY += audit.y;
            resultantZ += audit.z;
            result.assembledExternalForceL1Newtons += audit.w;
            result.assembledExternalForceResultantNewtons = std::sqrt(
                resultantX * resultantX + resultantY * resultantY +
                resultantZ * resultantZ);
        }
    }
    result.message = state_->message;
    return result;
}

} // namespace numi::matter
