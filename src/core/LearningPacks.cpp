#include "metalrobo/LearningPacks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

namespace metalrobo {
namespace {

constexpr std::array<char, 8u> kMagic{
    'M', 'R', 'L', 'E', 'A', 'R', 'N', '\0',
};
constexpr std::uint32_t kTaskKind = 1u;
constexpr std::uint32_t kPolicyKind = 2u;
constexpr std::uint32_t kPolicyRolloutKind = 3u;
constexpr std::uint32_t kMotionKind = 4u;
constexpr std::uint32_t kInteractionKind = 5u;
constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
constexpr std::uint64_t kMaximumPayloadBytes =
    std::numeric_limits<std::uint32_t>::max();
// Darwin write(2) does not accept an arbitrarily large count even though the
// public API uses size_t. Keep each transfer comfortably below the signed
// 32-bit boundary while preserving one durable, atomic pack publication.
constexpr std::size_t kMaximumIOChunkBytes = 64u * 1024u * 1024u;

struct LearningPackFileHeader {
    std::array<char, 8u> magic = kMagic;
    std::uint32_t formatVersion = 0u;
    std::uint32_t kind = 0u;
    std::uint64_t payloadBytes = 0u;
    std::uint64_t contentHash = 0u;
};

static_assert(
    std::is_trivially_copyable_v<LearningPackFileHeader>
);
static_assert(sizeof(LearningPackFileHeader) == 32u);
static_assert(std::endian::native == std::endian::little);

LearningPackResult fail(
    const LearningPackStatus status,
    std::string message
) {
    return {
        .status = status,
        .message = std::move(message),
    };
}

bool countFits(const std::size_t count) noexcept {
    return count <= std::numeric_limits<std::uint32_t>::max();
}

bool stringFits(const std::string_view value) noexcept {
    return countFits(value.size());
}

LearningPackResult validateTaskArtifact(
    const TaskPack& pack
) {
    if (pack.id.empty() || !stringFits(pack.id) ||
        pack.actions.empty() ||
        pack.actorFrame.empty() ||
        pack.actorHistoryLength == 0u ||
        pack.criticHistoryLength == 0u ||
        pack.maximumEpisodeSteps == 0u ||
        pack.curriculumLevelCount == 0u) {
        return fail(
            LearningPackStatus::invalidPack,
            "TaskPack identity, dimensions, or episode limits are invalid"
        );
    }
    const auto probability = [](const float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
    };
    if (!probability(pack.visual.fullDropoutProbability) ||
        !probability(pack.visual.pixelDropoutProbability) ||
        !probability(pack.visual.edgeFlickerProbability) ||
        !std::isfinite(pack.visual.depthJitterMeters) ||
        pack.visual.depthJitterMeters < 0.0f ||
        !std::isfinite(pack.visual.depthNoiseSigmaMeters) ||
        pack.visual.depthNoiseSigmaMeters < 0.0f) {
        return fail(
            LearningPackStatus::invalidPack,
            "TaskPack visual corruption is invalid"
        );
    }
    if (!pack.threat.protectedGroup.empty() &&
        (!(pack.threat.activationSpeed > 0.0f) ||
         !(pack.threat.horizonSeconds > 0.0f) ||
         !(pack.threat.safetyMargin >= 0.0f) ||
         !(pack.threat.cbfAlpha > 0.0f) ||
         !(pack.threat.stepOverMaximumHeight > 0.0f) ||
         !(pack.threat.sidestepMaximumHeight >
             pack.threat.stepOverMaximumHeight) ||
         !(pack.threat.leanMaximumHeight >
             pack.threat.sidestepMaximumHeight) ||
         !(pack.threat.urgencySeconds > 0.0f) ||
         !(pack.threat.desiredVelocityHorizonSeconds > 0.0f) ||
         !(pack.threat.projectionEpsilon > 0.0f))) {
        return fail(
            LearningPackStatus::invalidPack,
            "TaskPack threat program is invalid"
        );
    }
    const bool projectileBallistics =
        pack.pushes.projectileHorizontalSpeedUpper > 0.0f;
    if (!std::isfinite(
            pack.pushes.projectileTargetHorizontalRadius
        ) ||
        pack.pushes.projectileTargetHorizontalRadius < 0.0f ||
        !std::isfinite(pack.pushes.projectileHorizontalSpeedLower) ||
        !std::isfinite(pack.pushes.projectileHorizontalSpeedUpper) ||
        !std::isfinite(pack.pushes.projectileTargetHeightLower) ||
        !std::isfinite(pack.pushes.projectileTargetHeightUpper) ||
        pack.pushes.projectileHorizontalSpeedLower < 0.0f ||
        pack.pushes.projectileHorizontalSpeedUpper <
            pack.pushes.projectileHorizontalSpeedLower ||
        (projectileBallistics &&
         !(pack.pushes.projectileHorizontalSpeedLower > 0.0f)) ||
        pack.pushes.projectileTargetHeightUpper <
            pack.pushes.projectileTargetHeightLower) {
        return fail(
            LearningPackStatus::invalidPack,
            "TaskPack projectile ballistics are invalid"
        );
    }
    if (!countFits(pack.actions.size()) ||
        !countFits(pack.actorFrame.size()) ||
        !countFits(pack.critic.size()) ||
        !countFits(pack.contactGroups.size()) ||
        !countFits(pack.jointGroups.size()) ||
        !countFits(pack.rewards.size()) ||
        !countFits(pack.terminations.size()) ||
        !countFits(pack.randomization.size()) ||
        !countFits(pack.terrain.sampleOffsets.size()) ||
        !countFits(pack.terrain.resetTranslations.size()) ||
        !countFits(pack.visual.frameOffsets.size()) ||
        !countFits(pack.motion.trackedBodies.size())) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack table count exceeds the 32-bit artifact boundary"
        );
    }
    const auto validObservation = [](const auto& value) {
        return stringFits(value.target);
    };
    if (!std::all_of(
            pack.actorFrame.begin(),
            pack.actorFrame.end(),
            validObservation
        ) ||
        !std::all_of(
            pack.critic.begin(),
            pack.critic.end(),
            validObservation
        )) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack observation semantic exceeds the 32-bit artifact boundary"
        );
    }
    for (const TaskActionBinding& value : pack.actions) {
        if (!stringFits(value.joint)) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack action semantic exceeds the 32-bit artifact boundary"
            );
        }
    }
    for (const TaskContactGroup& value : pack.contactGroups) {
        const bool hasSupportPatch =
            value.supportPatchWidth != 0u ||
            value.supportPatchHeight != 0u;
        const std::uint64_t supportPatchCellCount =
            static_cast<std::uint64_t>(value.supportPatchWidth) *
            value.supportPatchHeight;
        if (!stringFits(value.id) ||
            !stringFits(value.referenceBody) ||
            !countFits(value.bodies.size()) ||
            !std::all_of(
                value.bodies.begin(),
                value.bodies.end(),
                stringFits
            )) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack contact-group table exceeds the 32-bit artifact boundary"
            );
        }
        if (hasSupportPatch &&
            (!value.support ||
             value.supportPatchWidth == 0u ||
             value.supportPatchHeight == 0u ||
             supportPatchCellCount > 64u ||
             !std::isfinite(value.supportPatchBounds.x) ||
             !std::isfinite(value.supportPatchBounds.y) ||
             !std::isfinite(value.supportPatchBounds.z) ||
             !std::isfinite(value.supportPatchBounds.w) ||
             !(value.supportPatchBounds.z >
                 value.supportPatchBounds.x) ||
             !(value.supportPatchBounds.w >
                 value.supportPatchBounds.y))) {
            return fail(
                LearningPackStatus::invalidPack,
                "TaskPack support patch is invalid"
            );
        }
    }
    for (const TaskJointGroup& value : pack.jointGroups) {
        if (!stringFits(value.id) ||
            !countFits(value.joints.size()) ||
            !std::all_of(
                value.joints.begin(),
                value.joints.end(),
                stringFits
            )) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack joint-group table exceeds the 32-bit artifact boundary"
            );
        }
    }
    const auto semanticFits = [](const auto& value) {
        if constexpr (requires { value.target; }) {
            return stringFits(value.sourceGroup) &&
                stringFits(value.target);
        }
        return stringFits(value.sourceGroup);
    };
    if (!std::all_of(
            pack.rewards.begin(),
            pack.rewards.end(),
            semanticFits
        ) ||
        !std::all_of(
            pack.terminations.begin(),
            pack.terminations.end(),
            semanticFits
        ) ||
        !std::all_of(
            pack.randomization.begin(),
            pack.randomization.end(),
            [](const auto& value) {
                return stringFits(value.target);
            }
        ) ||
        !stringFits(pack.terrain.body) ||
        !stringFits(pack.threat.protectedGroup) ||
        !stringFits(pack.motion.anchorBody) ||
        !std::all_of(
            pack.motion.trackedBodies.begin(),
            pack.motion.trackedBodies.end(),
            stringFits
        )) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack operator semantic exceeds the 32-bit artifact boundary"
        );
    }
    return {};
}

LearningPackResult validatePolicyArtifact(
    const PolicyPack& pack
) {
    if (pack.id.empty() || !stringFits(pack.id) ||
        pack.revision == 0u ||
        pack.layers.empty() ||
        !std::isfinite(pack.observationClip) ||
        !(pack.observationClip > 0.0f) ||
        !std::isfinite(pack.actionClip) ||
        !(pack.actionClip > 0.0f)) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyPack identity, revision, layers, or clipping is invalid"
        );
    }
    if (!countFits(pack.layers.size()) ||
        !countFits(pack.criticLayers.size()) ||
        !countFits(pack.observationMean.size()) ||
        !countFits(
            pack.observationInverseStandardDeviation.size()
        ) ||
        !countFits(pack.criticObservationMean.size()) ||
        !countFits(
            pack.criticObservationInverseStandardDeviation.size()
        ) ||
        !countFits(
            pack.actionLogStandardDeviation.size()
        ) ||
        !countFits(pack.actionBias.size()) ||
        !countFits(pack.actionScale.size())) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyPack table count exceeds the 32-bit artifact boundary"
        );
    }
    const auto validLayerTable = [](
        const std::span<const PolicyDenseLayer> layers
    ) {
        return std::all_of(
            layers.begin(),
            layers.end(),
            [](const PolicyDenseLayer& layer) {
                const std::uint64_t expectedWeights =
                    static_cast<std::uint64_t>(
                        layer.inputCount
                    ) * layer.outputCount;
                const auto finiteValues = [](
                    const std::span<const float> values
                ) {
                    return std::all_of(
                        values.begin(),
                        values.end(),
                        [](const float value) {
                            return std::isfinite(value);
                        }
                    );
                };
                return layer.inputCount != 0u &&
                    layer.outputCount != 0u &&
                    static_cast<std::uint32_t>(
                        layer.activation
                    ) <= MR_POLICY_ACTIVATION_SILU &&
                    expectedWeights == layer.weights.size() &&
                    layer.bias.size() == layer.outputCount &&
                    countFits(layer.weights.size()) &&
                    countFits(layer.bias.size()) &&
                    finiteValues(layer.weights) &&
                    finiteValues(layer.bias);
            }
        );
    };
    const auto finiteValues = [](
        const std::span<const float> values
    ) {
        return std::all_of(
            values.begin(),
            values.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        );
    };
    if (!validLayerTable(pack.layers) ||
        !validLayerTable(pack.criticLayers) ||
        (!pack.actionLogStandardDeviation.empty() &&
         pack.criticLayers.empty()) ||
        !finiteValues(pack.observationMean) ||
        !finiteValues(
            pack.observationInverseStandardDeviation
        ) ||
        !finiteValues(pack.criticObservationMean) ||
        !finiteValues(
            pack.criticObservationInverseStandardDeviation
        ) ||
        !finiteValues(pack.actionLogStandardDeviation) ||
        !finiteValues(pack.actionBias) ||
        !finiteValues(pack.actionScale)) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyPack dense-layer shape or stochastic contract is invalid"
        );
    }
    return {};
}

LearningPackResult validateMotionArtifact(
    const MotionPack& pack
) {
    if (pack.id.empty() || pack.sourceRepository.empty() ||
        pack.sourceRevision.empty() || pack.license.empty() ||
        pack.anchorBody.empty() || pack.trackedBodies.empty() ||
        pack.featureCount != pack.trackedBodies.size() * 9u ||
        pack.clips.empty() || !stringFits(pack.id) ||
        !stringFits(pack.sourceRepository) ||
        !stringFits(pack.sourceRevision) ||
        !stringFits(pack.license) || !stringFits(pack.anchorBody) ||
        !countFits(pack.trackedBodies.size()) ||
        !countFits(pack.clips.size())) {
        return fail(
            LearningPackStatus::invalidPack,
            "MotionPack identity, provenance, or feature contract is invalid"
        );
    }
    const auto finite = [](const std::span<const float> values) {
        return std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        });
    };
    if (!std::all_of(
            pack.trackedBodies.begin(),
            pack.trackedBodies.end(),
            stringFits
        ) ||
        !std::all_of(
            pack.clips.begin(),
            pack.clips.end(),
            [&](const MotionClip& clip) {
                return !clip.id.empty() && stringFits(clip.id) &&
                    std::isfinite(clip.framesPerSecond) &&
                    clip.framesPerSecond > 0.0f &&
                    countFits(clip.features.size()) &&
                    clip.features.size() >=
                        2u * pack.featureCount &&
                    clip.features.size() % pack.featureCount == 0u &&
                    finite(clip.features);
            }
        )) {
        return fail(
            LearningPackStatus::invalidPack,
            "MotionPack tracked bodies or clip samples are invalid"
        );
    }
    return {};
}

LearningPackResult validateInteractionArtifact(
    const InteractionPack& pack
) {
    if (pack.id.empty() || pack.sourceRepository.empty() ||
        pack.sourceRevision.empty() || pack.license.empty() ||
        pack.coordinateFrame != kInteractionCoordinateFrame ||
        pack.jointNames.empty() || pack.contactTracks.empty() ||
        pack.clips.empty() || !stringFits(pack.id) ||
        !stringFits(pack.sourceRepository) ||
        !stringFits(pack.sourceRevision) ||
        !stringFits(pack.license) ||
        !stringFits(pack.coordinateFrame) ||
        !countFits(pack.jointNames.size()) ||
        !countFits(pack.contactTracks.size()) ||
        !countFits(pack.clips.size())) {
        return fail(
            LearningPackStatus::invalidPack,
            "InteractionPack identity, provenance, or coordinate contract is invalid"
        );
    }
    const auto uniqueStrings = [](const auto& values) {
        for (std::size_t index = 0u; index < values.size(); ++index) {
            if (values[index].empty() || !stringFits(values[index]) ||
                std::find(
                    values.begin(),
                    values.begin() +
                        static_cast<std::ptrdiff_t>(index),
                    values[index]
                ) != values.begin() +
                    static_cast<std::ptrdiff_t>(index)) {
                return false;
            }
        }
        return true;
    };
    if (!uniqueStrings(pack.jointNames)) {
        return fail(
            LearningPackStatus::invalidPack,
            "InteractionPack joint semantics are empty or duplicated"
        );
    }
    std::vector<std::string> contactIds;
    contactIds.reserve(pack.contactTracks.size());
    for (const InteractionContactTrack& track : pack.contactTracks) {
        if (track.id.empty() || track.taskContactGroup.empty() ||
            track.counterpart.empty() || !stringFits(track.id) ||
            !stringFits(track.taskContactGroup) ||
            !stringFits(track.counterpart) ||
            std::find(contactIds.begin(), contactIds.end(), track.id) !=
                contactIds.end()) {
            return fail(
                LearningPackStatus::invalidPack,
                "InteractionPack contact semantics are invalid or duplicated"
            );
        }
        contactIds.push_back(track.id);
    }
    const auto finiteValues = [](const std::span<const float> values) {
        return std::all_of(
            values.begin(),
            values.end(),
            [](const float value) { return std::isfinite(value); }
        );
    };
    std::vector<std::string> clipIds;
    clipIds.reserve(pack.clips.size());
    constexpr std::uint32_t validSampleFlags =
        interactionSamplePredicted |
        interactionSamplePhysicsCertified;
    for (const InteractionClip& clip : pack.clips) {
        const std::uint64_t frameCount = clip.frameCount;
        const std::uint64_t contactSampleCount =
            frameCount * pack.contactTracks.size();
        const std::uint64_t contactValueCount =
            contactSampleCount * kInteractionContactFeatureCount;
        const std::uint64_t rootValueCount =
            frameCount * kInteractionRootTargetCount;
        const std::uint64_t jointValueCount =
            frameCount * pack.jointNames.size();
        if (clip.id.empty() || clip.desiredOutcome.empty() ||
            !stringFits(clip.id) ||
            !stringFits(clip.desiredOutcome) ||
            std::find(clipIds.begin(), clipIds.end(), clip.id) !=
                clipIds.end() ||
            !std::isfinite(clip.framesPerSecond) ||
            !(clip.framesPerSecond > 0.0f) ||
            clip.frameCount < 2u ||
            rootValueCount != clip.rootTargets.size() ||
            jointValueCount != clip.jointTargets.size() ||
            contactSampleCount != clip.contactModes.size() ||
            contactSampleCount != clip.contactFeatureMasks.size() ||
            contactSampleCount != clip.contactSampleFlags.size() ||
            contactSampleCount != clip.contactConfidence.size() ||
            contactValueCount != clip.contactTargets.size() ||
            contactValueCount != clip.contactTolerances.size() ||
            !countFits(clip.rootTargets.size()) ||
            !countFits(clip.jointTargets.size()) ||
            !countFits(clip.contactModes.size()) ||
            !countFits(clip.contactTargets.size()) ||
            !finiteValues(clip.rootTargets) ||
            !finiteValues(clip.jointTargets) ||
            !finiteValues(clip.contactConfidence) ||
            !finiteValues(clip.contactTargets) ||
            !finiteValues(clip.contactTolerances)) {
            return fail(
                LearningPackStatus::invalidPack,
                "InteractionPack clip dimensions or samples are invalid"
            );
        }
        for (std::uint32_t frame = 0u; frame < clip.frameCount; ++frame) {
            const std::size_t root =
                static_cast<std::size_t>(frame) *
                kInteractionRootTargetCount;
            const float x = clip.rootTargets[root + 3u];
            const float y = clip.rootTargets[root + 4u];
            const float z = clip.rootTargets[root + 5u];
            const float w = clip.rootTargets[root + 6u];
            const float norm = x * x + y * y + z * z + w * w;
            if (std::abs(norm - 1.0f) > 1.0e-3f) {
                return fail(
                    LearningPackStatus::invalidPack,
                    "InteractionPack root quaternion is not normalized"
                );
            }
        }
        for (std::size_t sample = 0u;
             sample < clip.contactModes.size();
             ++sample) {
            if (clip.contactModes[sample] >
                    static_cast<std::uint32_t>(
                        InteractionContactMode::release
                    ) ||
                (clip.contactFeatureMasks[sample] &
                    ~kInteractionContactFeatureMask) != 0u ||
                clip.contactSampleFlags[sample] == 0u ||
                (clip.contactSampleFlags[sample] &
                    ~validSampleFlags) != 0u ||
                clip.contactConfidence[sample] < 0.0f ||
                clip.contactConfidence[sample] > 1.0f) {
                return fail(
                    LearningPackStatus::invalidPack,
                    "InteractionPack contact mode, mask, provenance, or confidence is invalid"
                );
            }
            const std::size_t featureBase =
                sample * kInteractionContactFeatureCount;
            for (std::uint32_t feature = 0u;
                 feature < kInteractionContactFeatureCount;
                 ++feature) {
                const bool valid =
                    (clip.contactFeatureMasks[sample] &
                     (1u << feature)) != 0u;
                const float target =
                    clip.contactTargets[featureBase + feature];
                const float tolerance =
                    clip.contactTolerances[featureBase + feature];
                if ((valid && !(tolerance > 0.0f)) ||
                    (!valid && tolerance < 0.0f) ||
                    (valid && feature >= 8u && target < 0.0f)) {
                    return fail(
                        LearningPackStatus::invalidPack,
                        "InteractionPack contact target or tolerance is invalid"
                    );
                }
            }
        }
        clipIds.push_back(clip.id);
    }
    return {};
}

template <typename Pack>
LearningPackResult validatePolicyRolloutArtifact(const Pack& pack) {
    if (pack.id.empty() || !stringFits(pack.id) ||
        pack.taskFingerprint == 0u ||
        pack.policyFingerprint == 0u ||
        pack.policyRevision == 0u ||
        pack.environmentCount == 0u ||
        pack.controlStepCount == 0u ||
        pack.actorObservationCount == 0u ||
        pack.criticObservationCount == 0u ||
        pack.actionCount == 0u) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyRolloutPack identity, fingerprints, or dimensions are invalid"
        );
    }
    const auto multiply = [](
        const std::uint64_t left,
        const std::uint64_t right,
        std::uint64_t& output
    ) {
        if (right != 0u &&
            left >
                std::numeric_limits<std::uint64_t>::max() /
                    right) {
            return false;
        }
        output = left * right;
        return true;
    };
    std::uint64_t samples = 0u;
    std::uint64_t actorElements = 0u;
    std::uint64_t criticElements = 0u;
    std::uint64_t actionElements = 0u;
    std::uint64_t motionElements = 0u;
    if (!multiply(
            pack.environmentCount,
            pack.controlStepCount,
            samples
        ) ||
        !multiply(
            samples,
            pack.actorObservationCount,
            actorElements
        ) ||
        !multiply(
            samples,
            pack.criticObservationCount,
            criticElements
        ) ||
        !multiply(
            samples,
            pack.actionCount,
            actionElements
        ) ||
        !multiply(
            samples,
            pack.motionFeatureCount,
            motionElements
        ) ||
        samples >
            std::numeric_limits<std::size_t>::max() ||
        actorElements >
            std::numeric_limits<std::size_t>::max() ||
        criticElements >
            std::numeric_limits<std::size_t>::max() ||
        actionElements >
            std::numeric_limits<std::size_t>::max() ||
        motionElements >
            std::numeric_limits<std::size_t>::max() ||
        pack.actorObservations.size() != actorElements ||
        pack.criticObservations.size() != criticElements ||
        pack.motionFeatures.size() != motionElements ||
        (!pack.teacherActions.empty() &&
         pack.teacherActions.size() != actionElements) ||
        pack.latents.size() != actionElements ||
        pack.logProbabilities.size() != samples ||
        pack.values.size() != samples ||
        pack.bootstrapValues.size() !=
            pack.environmentCount ||
        pack.transitions.size() != samples) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyRolloutPack tensors do not match its declared dimensions"
        );
    }
    const auto finiteValues = [](
        const std::span<const float> values
    ) {
        return std::all_of(
            values.begin(),
            values.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        );
    };
    if (!finiteValues(pack.actorObservations) ||
        !finiteValues(pack.criticObservations) ||
        !finiteValues(pack.motionFeatures) ||
        !finiteValues(pack.teacherActions) ||
        !finiteValues(pack.latents) ||
        !finiteValues(pack.logProbabilities) ||
        !finiteValues(pack.values) ||
        !finiteValues(pack.bootstrapValues) ||
        !std::all_of(
            pack.transitions.begin(),
            pack.transitions.end(),
            [&](const MRTaskTransitionGPU& transition) {
                return transition.policyRevision ==
                        pack.policyRevision &&
                    std::isfinite(
                        transition.rewardAndState.x
                    ) &&
                    std::isfinite(
                        transition.rewardAndState.y
                    ) &&
                    std::isfinite(
                        transition.rewardAndState.z
                    ) &&
                    std::isfinite(
                        transition.rewardAndState.w
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.x
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.y
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.z
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.w
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.x
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.y
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.z
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.w
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown0.x
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown0.y
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown0.z
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown0.w
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown1.x
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown1.y
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown1.z
                    ) &&
                    std::isfinite(
                        transition.dodgeRewardBreakdown1.w
                    ) &&
                    std::isfinite(
                        transition.timeoutBootstrapValue
                    ) &&
                    std::isfinite(
                        transition.episodeTrackingScore
                    ) &&
                    transition.termination.x <= 1u &&
                    transition.termination.y <= 1u &&
                    transition.termination.z <= 1u;
            }
        )) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyRolloutPack contains non-finite or inconsistent samples"
        );
    }
    std::uint64_t payloadBytes =
        8u + pack.id.size() +
        3u * sizeof(std::uint64_t) +
        6u * sizeof(std::uint32_t) +
        9u * sizeof(std::uint64_t);
    if (payloadBytes > kMaximumPayloadBytes) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyRolloutPack payload exceeds the 32-bit artifact boundary"
        );
    }
    const auto addTable = [&]<typename Table>(
        const Table& values
    ) {
        using Value = typename Table::value_type;
        const std::uint64_t available =
            kMaximumPayloadBytes - payloadBytes;
        if (values.size() > available / sizeof(Value)) {
            return false;
        }
        payloadBytes +=
            static_cast<std::uint64_t>(values.size()) *
            sizeof(Value);
        return true;
    };
    if (!addTable(pack.actorObservations) ||
        !addTable(pack.criticObservations) ||
        !addTable(pack.motionFeatures) ||
        !addTable(pack.teacherActions) ||
        !addTable(pack.latents) ||
        !addTable(pack.logProbabilities) ||
        !addTable(pack.values) ||
        !addTable(pack.bootstrapValues) ||
        !addTable(pack.transitions)) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyRolloutPack payload exceeds the 32-bit artifact boundary"
        );
    }
    return {};
}

std::uint64_t contentHash(
    const std::span<const std::byte> values
) {
    std::uint64_t hash = kFNVOffset;
    for (const std::byte value : values) {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= kFNVPrime;
    }
    return hash == 0u ? 1u : hash;
}

class Writer {
public:
    explicit Writer(const std::size_t reservedBytes = 0u) {
        data_.reserve(reservedBytes);
    }

    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes =
            reinterpret_cast<const std::byte*>(&value);
        data_.insert(data_.end(), bytes, bytes + sizeof(T));
    }

    void string(const std::string_view value) {
        const std::uint64_t count = value.size();
        pod(count);
        const auto* bytes =
            reinterpret_cast<const std::byte*>(
                value.data()
            );
        data_.insert(data_.end(), bytes, bytes + value.size());
    }

    template <typename T>
    void vector(const std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint64_t count = values.size();
        pod(count);
        if (!values.empty()) {
            const auto* bytes =
                reinterpret_cast<const std::byte*>(
                    values.data()
                );
            data_.insert(
                data_.end(),
                bytes,
                bytes + values.size() * sizeof(T)
            );
        }
    }

    template <typename T>
    void vector(const std::vector<T>& values) {
        vector(std::span<const T>{values});
    }

    void strings(const std::vector<std::string>& values) {
        const std::uint64_t count = values.size();
        pod(count);
        for (const std::string& value : values) {
            string(value);
        }
    }

    [[nodiscard]] const std::vector<std::byte>& data()
        const noexcept {
        return data_;
    }

private:
    std::vector<std::byte> data_;
};

class Reader {
public:
    explicit Reader(const std::span<const std::byte> data)
        : data_(data) {}

    template <typename T>
    bool pod(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (sizeof(T) > data_.size() - cursor_) {
            return false;
        }
        std::memcpy(
            &value,
            data_.data() + cursor_,
            sizeof(T)
        );
        cursor_ += sizeof(T);
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > data_.size() - cursor_ ||
            count > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        value.assign(
            reinterpret_cast<const char*>(
                data_.data() + cursor_
            ),
            static_cast<std::size_t>(count)
        );
        cursor_ += static_cast<std::size_t>(count);
        return true;
    }

    template <typename T>
    bool vector(std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > std::numeric_limits<std::uint32_t>::max() ||
            count >
                (data_.size() - cursor_) / sizeof(T)) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        const std::size_t bytes =
            values.size() * sizeof(T);
        if (bytes != 0u) {
            std::memcpy(
                values.data(),
                data_.data() + cursor_,
                bytes
            );
        }
        cursor_ += bytes;
        return true;
    }

    bool strings(std::vector<std::string>& values) {
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        for (std::string& value : values) {
            if (!string(value)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool finished() const noexcept {
        return cursor_ == data_.size();
    }

private:
    std::span<const std::byte> data_;
    std::size_t cursor_ = 0u;
};

template <typename Enum>
void writeEnum(Writer& writer, const Enum value) {
    writer.pod(static_cast<std::uint32_t>(value));
}

template <typename Enum>
bool readEnum(Reader& reader, Enum& value) {
    std::uint32_t raw = 0u;
    if (!reader.pod(raw)) {
        return false;
    }
    value = static_cast<Enum>(raw);
    return true;
}

void writeObservation(
    Writer& writer,
    const TaskObservationOperatorSpec& value
) {
    writeEnum(writer, value.source);
    writer.string(value.target);
    writer.pod(value.component);
    writer.pod(value.scale);
    writer.pod(value.offset);
    writer.pod(value.noiseAmplitude);
    writer.pod(value.biasLower);
    writer.pod(value.biasUpper);
    writer.pod(
        static_cast<std::uint8_t>(
            value.normalizeVector3
        )
    );
}

bool readObservation(
    Reader& reader,
    TaskObservationOperatorSpec& value
) {
    std::uint8_t normalize = 0u;
    if (!readEnum(reader, value.source) ||
        !reader.string(value.target) ||
        !reader.pod(value.component) ||
        !reader.pod(value.scale) ||
        !reader.pod(value.offset) ||
        !reader.pod(value.noiseAmplitude) ||
        !reader.pod(value.biasLower) ||
        !reader.pod(value.biasUpper) ||
        !reader.pod(normalize) ||
        normalize > 1u) {
        return false;
    }
    value.normalizeVector3 = normalize != 0u;
    return true;
}

template <typename T, typename Function>
void writeRichVector(
    Writer& writer,
    const std::vector<T>& values,
    Function&& function
) {
    writer.pod(static_cast<std::uint64_t>(values.size()));
    for (const T& value : values) {
        function(writer, value);
    }
}

template <typename T, typename Function>
bool readRichVector(
    Reader& reader,
    std::vector<T>& values,
    Function&& function
) {
    std::uint64_t count = 0u;
    if (!reader.pod(count) ||
        count > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    values.resize(static_cast<std::size_t>(count));
    for (T& value : values) {
        if (!function(reader, value)) {
            return false;
        }
    }
    return true;
}

std::vector<std::byte> serializeTask(
    const TaskPack& pack
) {
    Writer writer;
    writer.string(pack.id);
    writer.pod(pack.capacities);
    writeRichVector(
        writer,
        pack.actions,
        [](Writer& target, const TaskActionBinding& value) {
            target.string(value.joint);
            target.pod(value.scale);
            target.pod(value.responseTimeSeconds);
        }
    );
    writeRichVector(writer, pack.actorFrame, writeObservation);
    writer.pod(pack.actorHistoryLength);
    writeRichVector(writer, pack.critic, writeObservation);
    writer.pod(pack.criticHistoryLength);
    writer.pod(static_cast<std::uint8_t>(
        pack.criticIncludesCleanHistory
    ));
    writeRichVector(
        writer,
        pack.contactGroups,
        [](Writer& target, const TaskContactGroup& value) {
            target.string(value.id);
            target.strings(value.bodies);
            target.pod(static_cast<std::uint8_t>(
                value.support
            ));
            target.pod(static_cast<std::uint8_t>(
                value.forbidden
            ));
            target.string(value.referenceBody);
            target.pod(value.localReference);
            target.pod(value.gaitPhaseOffsetRadians);
            target.pod(value.stanceFraction);
            target.pod(value.supportPatchBounds);
            target.pod(value.supportPatchWidth);
            target.pod(value.supportPatchHeight);
        }
    );
    writeRichVector(
        writer,
        pack.jointGroups,
        [](Writer& target, const TaskJointGroup& value) {
            target.string(value.id);
            target.strings(value.joints);
        }
    );
    writeRichVector(
        writer,
        pack.rewards,
        [](Writer& target, const TaskRewardOperatorSpec& value) {
            writeEnum(target, value.operation);
            target.string(value.sourceGroup);
            target.string(value.target);
            target.pod(value.weight);
            target.pod(value.parameters);
        }
    );
    writeRichVector(
        writer,
        pack.terminations,
        [](Writer& target,
           const TaskTerminationOperatorSpec& value) {
            writeEnum(target, value.operation);
            target.string(value.sourceGroup);
            target.pod(value.reason);
            target.pod(value.priority);
            target.pod(value.threshold);
            target.pod(value.failurePenalty);
        }
    );
    writeRichVector(
        writer,
        pack.randomization,
        [](Writer& target,
           const TaskRandomizationOperatorSpec& value) {
            writeEnum(target, value.operation);
            target.string(value.target);
            target.pod(value.component);
            target.pod(value.minimumCurriculumLevel);
            target.pod(value.parameters);
        }
    );
    writer.pod(pack.commands.lower);
    writer.pod(pack.commands.upper);
    writer.pod(pack.commands.limitLower);
    writer.pod(pack.commands.limitUpper);
    writer.pod(pack.commands.curriculumStep);
    writer.pod(pack.commands.standingProbability);
    writer.pod(pack.commands.minimumEpisodeSurvivalFraction);
    writer.pod(pack.commands.minimumDurationSeconds);
    writer.pod(pack.commands.maximumDurationSeconds);
    writer.pod(pack.pushes.maximumVelocity);
    writer.pod(pack.pushes.minimumIntervalSeconds);
    writer.pod(pack.pushes.maximumIntervalSeconds);
    writer.pod(pack.pushes.projectileStandingProbability);
    writer.pod(pack.pushes.projectileTargetHorizontalRadius);
    writer.pod(pack.pushes.projectileHorizontalSpeedLower);
    writer.pod(pack.pushes.projectileHorizontalSpeedUpper);
    writer.pod(pack.pushes.projectileTargetHeightLower);
    writer.pod(pack.pushes.projectileTargetHeightUpper);
    writer.string(pack.terrain.body);
    writer.vector(pack.terrain.sampleOffsets);
    writer.vector(pack.terrain.resetTranslations);
    writer.pod(pack.visual.width);
    writer.pod(pack.visual.height);
    writer.vector(pack.visual.frameOffsets);
    writer.pod(pack.visual.nearDepthMeters);
    writer.pod(pack.visual.farDepthMeters);
    writer.pod(pack.visual.fullDropoutProbability);
    writer.pod(pack.visual.pixelDropoutProbability);
    writer.pod(pack.visual.depthJitterMeters);
    writer.pod(pack.visual.depthNoiseSigmaMeters);
    writer.pod(pack.visual.edgeFlickerProbability);
    writer.pod(pack.visual.curriculumCorruptionGain);
    writer.pod(static_cast<std::uint8_t>(
        pack.visual.includeDerivedFeatures ? 1u : 0u
    ));
    writer.string(pack.threat.protectedGroup);
    writer.pod(pack.threat.activationSpeed);
    writer.pod(pack.threat.horizonSeconds);
    writer.pod(pack.threat.safetyMargin);
    writer.pod(pack.threat.cbfAlpha);
    writer.pod(pack.threat.stepOverMaximumHeight);
    writer.pod(pack.threat.sidestepMaximumHeight);
    writer.pod(pack.threat.leanMaximumHeight);
    writer.pod(pack.threat.urgencySeconds);
    writer.pod(pack.threat.desiredVelocityHorizonSeconds);
    writer.pod(pack.threat.projectionEpsilon);
    writer.string(pack.motion.anchorBody);
    writer.strings(pack.motion.trackedBodies);
    writer.pod(pack.maximumEpisodeSteps);
    writer.pod(pack.maximumActionDelaySteps);
    writer.pod(pack.maximumObservationDelaySteps);
    writer.pod(pack.curriculumLevelCount);
    writer.pod(pack.baseHeightTarget);
    writer.pod(pack.gaitPeriodSeconds);
    writer.pod(pack.clearanceTarget);
    writer.pod(pack.successTrackingThreshold);
    writer.pod(pack.supportForceThreshold);
    return writer.data();
}

bool deserializeTask(
    const std::span<const std::byte> payload,
    TaskPack& pack
) {
    Reader reader{payload};
    std::uint8_t cleanHistory = 0u;
    std::uint8_t visualDerivedFeatures = 0u;
    if (!reader.string(pack.id) ||
        !reader.pod(pack.capacities) ||
        !readRichVector(
            reader,
            pack.actions,
            [](Reader& source, TaskActionBinding& value) {
                return source.string(value.joint) &&
                    source.pod(value.scale) &&
                    source.pod(value.responseTimeSeconds);
            }
        ) ||
        !readRichVector(
            reader,
            pack.actorFrame,
            readObservation
        ) ||
        !reader.pod(pack.actorHistoryLength) ||
        !readRichVector(reader, pack.critic, readObservation) ||
        !reader.pod(pack.criticHistoryLength) ||
        !reader.pod(cleanHistory) ||
        cleanHistory > 1u ||
        !readRichVector(
            reader,
            pack.contactGroups,
            [](Reader& source, TaskContactGroup& value) {
                std::uint8_t support = 0u;
                std::uint8_t forbidden = 0u;
                if (!source.string(value.id) ||
                    !source.strings(value.bodies) ||
                    !source.pod(support) ||
                    !source.pod(forbidden) ||
                    support > 1u || forbidden > 1u ||
                    !source.string(value.referenceBody) ||
                    !source.pod(value.localReference) ||
                    !source.pod(
                        value.gaitPhaseOffsetRadians
                    ) ||
                    !source.pod(value.stanceFraction) ||
                    !source.pod(value.supportPatchBounds) ||
                    !source.pod(value.supportPatchWidth) ||
                    !source.pod(value.supportPatchHeight)) {
                    return false;
                }
                value.support = support != 0u;
                value.forbidden = forbidden != 0u;
                return true;
            }
        ) ||
        !readRichVector(
            reader,
            pack.jointGroups,
            [](Reader& source, TaskJointGroup& value) {
                return source.string(value.id) &&
                    source.strings(value.joints);
            }
        ) ||
        !readRichVector(
            reader,
            pack.rewards,
            [](Reader& source,
               TaskRewardOperatorSpec& value) {
                return readEnum(source, value.operation) &&
                    source.string(value.sourceGroup) &&
                    source.string(value.target) &&
                    source.pod(value.weight) &&
                    source.pod(value.parameters);
            }
        ) ||
        !readRichVector(
            reader,
            pack.terminations,
            [](Reader& source,
               TaskTerminationOperatorSpec& value) {
                return readEnum(source, value.operation) &&
                    source.string(value.sourceGroup) &&
                    source.pod(value.reason) &&
                    source.pod(value.priority) &&
                    source.pod(value.threshold) &&
                    source.pod(value.failurePenalty);
            }
        ) ||
        !readRichVector(
            reader,
            pack.randomization,
            [](Reader& source,
               TaskRandomizationOperatorSpec& value) {
                return readEnum(source, value.operation) &&
                    source.string(value.target) &&
                    source.pod(value.component) &&
                    source.pod(value.minimumCurriculumLevel) &&
                    source.pod(value.parameters);
            }
        ) ||
        !reader.pod(pack.commands.lower) ||
        !reader.pod(pack.commands.upper) ||
        !reader.pod(pack.commands.limitLower) ||
        !reader.pod(pack.commands.limitUpper) ||
        !reader.pod(pack.commands.curriculumStep) ||
        !reader.pod(pack.commands.standingProbability) ||
        !reader.pod(
            pack.commands.minimumEpisodeSurvivalFraction
        ) ||
        !reader.pod(pack.commands.minimumDurationSeconds) ||
        !reader.pod(pack.commands.maximumDurationSeconds) ||
        !reader.pod(pack.pushes.maximumVelocity) ||
        !reader.pod(pack.pushes.minimumIntervalSeconds) ||
        !reader.pod(pack.pushes.maximumIntervalSeconds) ||
        !reader.pod(pack.pushes.projectileStandingProbability) ||
        !reader.pod(pack.pushes.projectileTargetHorizontalRadius) ||
        !reader.pod(pack.pushes.projectileHorizontalSpeedLower) ||
        !reader.pod(pack.pushes.projectileHorizontalSpeedUpper) ||
        !reader.pod(pack.pushes.projectileTargetHeightLower) ||
        !reader.pod(pack.pushes.projectileTargetHeightUpper) ||
        !reader.string(pack.terrain.body) ||
        !reader.vector(pack.terrain.sampleOffsets) ||
        !reader.vector(pack.terrain.resetTranslations) ||
        !reader.pod(pack.visual.width) ||
        !reader.pod(pack.visual.height) ||
        !reader.vector(pack.visual.frameOffsets) ||
        !reader.pod(pack.visual.nearDepthMeters) ||
        !reader.pod(pack.visual.farDepthMeters) ||
        !reader.pod(pack.visual.fullDropoutProbability) ||
        !reader.pod(pack.visual.pixelDropoutProbability) ||
        !reader.pod(pack.visual.depthJitterMeters) ||
        !reader.pod(pack.visual.depthNoiseSigmaMeters) ||
        !reader.pod(pack.visual.edgeFlickerProbability) ||
        !reader.pod(pack.visual.curriculumCorruptionGain) ||
        !reader.pod(visualDerivedFeatures) ||
        !reader.string(pack.threat.protectedGroup) ||
        !reader.pod(pack.threat.activationSpeed) ||
        !reader.pod(pack.threat.horizonSeconds) ||
        !reader.pod(pack.threat.safetyMargin) ||
        !reader.pod(pack.threat.cbfAlpha) ||
        !reader.pod(pack.threat.stepOverMaximumHeight) ||
        !reader.pod(pack.threat.sidestepMaximumHeight) ||
        !reader.pod(pack.threat.leanMaximumHeight) ||
        !reader.pod(pack.threat.urgencySeconds) ||
        !reader.pod(pack.threat.desiredVelocityHorizonSeconds) ||
        !reader.pod(pack.threat.projectionEpsilon) ||
        !reader.string(pack.motion.anchorBody) ||
        !reader.strings(pack.motion.trackedBodies) ||
        !reader.pod(pack.maximumEpisodeSteps) ||
        !reader.pod(pack.maximumActionDelaySteps) ||
        !reader.pod(pack.maximumObservationDelaySteps) ||
        !reader.pod(pack.curriculumLevelCount) ||
        !reader.pod(pack.baseHeightTarget) ||
        !reader.pod(pack.gaitPeriodSeconds) ||
        !reader.pod(pack.clearanceTarget) ||
        !reader.pod(pack.successTrackingThreshold) ||
        !reader.pod(pack.supportForceThreshold) ||
        !reader.finished()) {
        return false;
    }
    pack.criticIncludesCleanHistory = cleanHistory != 0u;
    pack.visual.includeDerivedFeatures = visualDerivedFeatures != 0u;
    return true;
}

std::vector<std::byte> serializePolicy(
    const PolicyPack& pack
) {
    Writer writer;
    writer.string(pack.id);
    writer.pod(pack.revision);
    writer.vector(pack.observationMean);
    writer.vector(
        pack.observationInverseStandardDeviation
    );
    writeRichVector(
        writer,
        pack.layers,
        [](Writer& target, const PolicyDenseLayer& layer) {
            target.pod(layer.inputCount);
            target.pod(layer.outputCount);
            writeEnum(target, layer.activation);
            target.vector(layer.weights);
            target.vector(layer.bias);
        }
    );
    writer.vector(pack.criticObservationMean);
    writer.vector(
        pack.criticObservationInverseStandardDeviation
    );
    writeRichVector(
        writer,
        pack.criticLayers,
        [](Writer& target, const PolicyDenseLayer& layer) {
            target.pod(layer.inputCount);
            target.pod(layer.outputCount);
            writeEnum(target, layer.activation);
            target.vector(layer.weights);
            target.vector(layer.bias);
        }
    );
    writer.vector(pack.actionLogStandardDeviation);
    writer.vector(pack.actionBias);
    writer.vector(pack.actionScale);
    writer.pod(pack.observationClip);
    writer.pod(pack.actionClip);
    return writer.data();
}

bool deserializePolicy(
    const std::span<const std::byte> payload,
    PolicyPack& pack
) {
    Reader reader{payload};
    return reader.string(pack.id) &&
        reader.pod(pack.revision) &&
        reader.vector(pack.observationMean) &&
        reader.vector(
            pack.observationInverseStandardDeviation
        ) &&
        readRichVector(
            reader,
            pack.layers,
            [](Reader& source, PolicyDenseLayer& layer) {
                return source.pod(layer.inputCount) &&
                    source.pod(layer.outputCount) &&
                    readEnum(source, layer.activation) &&
                    source.vector(layer.weights) &&
                    source.vector(layer.bias);
            }
        ) &&
        reader.vector(pack.criticObservationMean) &&
        reader.vector(
            pack.criticObservationInverseStandardDeviation
        ) &&
        readRichVector(
            reader,
            pack.criticLayers,
            [](Reader& source, PolicyDenseLayer& layer) {
                return source.pod(layer.inputCount) &&
                    source.pod(layer.outputCount) &&
                    readEnum(source, layer.activation) &&
                    source.vector(layer.weights) &&
                    source.vector(layer.bias);
            }
        ) &&
        reader.vector(pack.actionLogStandardDeviation) &&
        reader.vector(pack.actionBias) &&
        reader.vector(pack.actionScale) &&
        reader.pod(pack.observationClip) &&
        reader.pod(pack.actionClip) &&
        reader.finished();
}

template <typename Pack>
std::vector<std::byte> serializePolicyRollout(const Pack& pack) {
    const std::size_t payloadBytes =
        8u + pack.id.size() +
        3u * sizeof(std::uint64_t) +
        6u * sizeof(std::uint32_t) +
        9u * sizeof(std::uint64_t) +
        pack.actorObservations.size() * sizeof(float) +
        pack.criticObservations.size() * sizeof(float) +
        pack.motionFeatures.size() * sizeof(float) +
        pack.teacherActions.size() * sizeof(float) +
        pack.latents.size() * sizeof(float) +
        pack.logProbabilities.size() * sizeof(float) +
        pack.values.size() * sizeof(float) +
        pack.bootstrapValues.size() * sizeof(float) +
        pack.transitions.size() *
            sizeof(MRTaskTransitionGPU);
    Writer writer{payloadBytes};
    writer.string(pack.id);
    writer.pod(pack.taskFingerprint);
    writer.pod(pack.policyFingerprint);
    writer.pod(pack.policyRevision);
    writer.pod(pack.environmentCount);
    writer.pod(pack.controlStepCount);
    writer.pod(pack.actorObservationCount);
    writer.pod(pack.criticObservationCount);
    writer.pod(pack.actionCount);
    writer.pod(pack.motionFeatureCount);
    writer.vector(pack.actorObservations);
    writer.vector(pack.criticObservations);
    writer.vector(pack.motionFeatures);
    writer.vector(pack.teacherActions);
    writer.vector(pack.latents);
    writer.vector(pack.logProbabilities);
    writer.vector(pack.values);
    writer.vector(pack.bootstrapValues);
    writer.vector(pack.transitions);
    return writer.data();
}

bool deserializePolicyRollout(
    const std::span<const std::byte> payload,
    PolicyRolloutPack& pack
) {
    Reader reader{payload};
    return reader.string(pack.id) &&
        reader.pod(pack.taskFingerprint) &&
        reader.pod(pack.policyFingerprint) &&
        reader.pod(pack.policyRevision) &&
        reader.pod(pack.environmentCount) &&
        reader.pod(pack.controlStepCount) &&
        reader.pod(pack.actorObservationCount) &&
        reader.pod(pack.criticObservationCount) &&
        reader.pod(pack.actionCount) &&
        reader.pod(pack.motionFeatureCount) &&
        reader.vector(pack.actorObservations) &&
        reader.vector(pack.criticObservations) &&
        reader.vector(pack.motionFeatures) &&
        reader.vector(pack.teacherActions) &&
        reader.vector(pack.latents) &&
        reader.vector(pack.logProbabilities) &&
        reader.vector(pack.values) &&
        reader.vector(pack.bootstrapValues) &&
        reader.vector(pack.transitions) &&
        reader.finished();
}

std::vector<std::byte> serializeMotion(
    const MotionPack& pack
) {
    Writer writer{0u};
    writer.string(pack.id);
    writer.string(pack.sourceRepository);
    writer.string(pack.sourceRevision);
    writer.string(pack.license);
    writer.string(pack.anchorBody);
    writer.strings(pack.trackedBodies);
    writer.pod(pack.featureCount);
    writeRichVector(
        writer,
        pack.clips,
        [](Writer& output, const MotionClip& clip) {
            output.string(clip.id);
            output.pod(clip.framesPerSecond);
            output.vector(clip.features);
        }
    );
    return writer.data();
}

bool deserializeMotion(
    const std::span<const std::byte> payload,
    MotionPack& pack
) {
    Reader reader{payload};
    return reader.string(pack.id) &&
        reader.string(pack.sourceRepository) &&
        reader.string(pack.sourceRevision) &&
        reader.string(pack.license) &&
        reader.string(pack.anchorBody) &&
        reader.strings(pack.trackedBodies) &&
        reader.pod(pack.featureCount) &&
        readRichVector(
            reader,
            pack.clips,
            [](Reader& source, MotionClip& clip) {
                return source.string(clip.id) &&
                    source.pod(clip.framesPerSecond) &&
                    source.vector(clip.features);
            }
        ) && reader.finished();
}

std::vector<std::byte> serializeInteraction(
    const InteractionPack& pack
) {
    Writer writer{0u};
    writer.string(pack.id);
    writer.string(pack.sourceRepository);
    writer.string(pack.sourceRevision);
    writer.string(pack.license);
    writer.string(pack.coordinateFrame);
    writer.strings(pack.jointNames);
    writeRichVector(
        writer,
        pack.contactTracks,
        [](Writer& output, const InteractionContactTrack& track) {
            output.string(track.id);
            output.string(track.taskContactGroup);
            output.string(track.counterpart);
        }
    );
    writeRichVector(
        writer,
        pack.clips,
        [](Writer& output, const InteractionClip& clip) {
            output.string(clip.id);
            output.string(clip.desiredOutcome);
            output.pod(clip.framesPerSecond);
            output.pod(clip.frameCount);
            const std::uint32_t loop = clip.loop ? 1u : 0u;
            output.pod(loop);
            output.vector(clip.rootTargets);
            output.vector(clip.jointTargets);
            output.vector(clip.contactModes);
            output.vector(clip.contactFeatureMasks);
            output.vector(clip.contactSampleFlags);
            output.vector(clip.contactConfidence);
            output.vector(clip.contactTargets);
            output.vector(clip.contactTolerances);
        }
    );
    return writer.data();
}

bool deserializeInteraction(
    const std::span<const std::byte> payload,
    InteractionPack& pack
) {
    Reader reader{payload};
    return reader.string(pack.id) &&
        reader.string(pack.sourceRepository) &&
        reader.string(pack.sourceRevision) &&
        reader.string(pack.license) &&
        reader.string(pack.coordinateFrame) &&
        reader.strings(pack.jointNames) &&
        readRichVector(
            reader,
            pack.contactTracks,
            [](Reader& source, InteractionContactTrack& track) {
                return source.string(track.id) &&
                    source.string(track.taskContactGroup) &&
                    source.string(track.counterpart);
            }
        ) &&
        readRichVector(
            reader,
            pack.clips,
            [](Reader& source, InteractionClip& clip) {
                std::uint32_t loop = 0u;
                if (!source.string(clip.id) ||
                    !source.string(clip.desiredOutcome) ||
                    !source.pod(clip.framesPerSecond) ||
                    !source.pod(clip.frameCount) ||
                    !source.pod(loop) || loop > 1u ||
                    !source.vector(clip.rootTargets) ||
                    !source.vector(clip.jointTargets) ||
                    !source.vector(clip.contactModes) ||
                    !source.vector(clip.contactFeatureMasks) ||
                    !source.vector(clip.contactSampleFlags) ||
                    !source.vector(clip.contactConfidence) ||
                    !source.vector(clip.contactTargets) ||
                    !source.vector(clip.contactTolerances)) {
                    return false;
                }
                clip.loop = loop != 0u;
                return true;
            }
        ) && reader.finished();
}

bool writeAll(
    const int descriptor,
    const void* bytes,
    const std::size_t byteCount,
    int& errorNumber
) noexcept {
    const auto* cursor =
        static_cast<const std::byte*>(bytes);
    std::size_t remaining = byteCount;
    while (remaining != 0u) {
        const std::size_t chunkBytes =
            std::min(remaining, kMaximumIOChunkBytes);
        const ssize_t written = ::write(
            descriptor,
            cursor,
            chunkBytes
        );
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            errorNumber = errno;
            return false;
        }
        if (written == 0) {
            errorNumber = EIO;
            return false;
        }
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
    }
    return true;
}

LearningPackResult writePack(
    const std::vector<std::byte>& payload,
    const std::uint32_t kind,
    const std::uint32_t version,
    const std::filesystem::path& path
) {
    if (path.empty() ||
        payload.size() > kMaximumPayloadBytes) {
        return fail(
            payload.size() > kMaximumPayloadBytes
                ? LearningPackStatus::capacityOverflow
                : LearningPackStatus::ioFailure,
            "learning pack path is empty or payload exceeds the 32-bit artifact boundary"
        );
    }
    try {
        LearningPackFileHeader header;
        header.formatVersion = version;
        header.kind = kind;
        header.payloadBytes = payload.size();
        header.contentHash = contentHash(payload);
        std::string temporaryTemplate =
            path.string() + ".tmp.XXXXXX";
        std::vector<char> temporaryCharacters(
            temporaryTemplate.begin(),
            temporaryTemplate.end()
        );
        temporaryCharacters.push_back('\0');
        const int descriptor =
            ::mkstemp(temporaryCharacters.data());
        if (descriptor < 0) {
            return fail(
                LearningPackStatus::ioFailure,
                "could not create sibling learning-pack temporary"
            );
        }
        const std::filesystem::path temporary{
            temporaryCharacters.data()
        };
        int writeError = 0;
        const bool wrote =
            writeAll(
                descriptor,
                &header,
                sizeof(header),
                writeError
            ) &&
            (payload.empty() ||
             writeAll(
                 descriptor,
                 payload.data(),
                 payload.size(),
                 writeError
             )) &&
            (::fsync(descriptor) == 0 ||
             (writeError = errno, false));
        const bool closed =
            ::close(descriptor) == 0 ||
            (writeError = writeError == 0 ? errno : writeError, false);
        if (!wrote || !closed) {
            const std::string error =
                std::generic_category().message(
                    writeError == 0 ? EIO : writeError
                );
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(
                LearningPackStatus::ioFailure,
                "could not durably write learning pack: " + error
            );
        }
        if (::rename(
                temporary.c_str(),
                path.c_str()
            ) != 0) {
            const std::string error =
                std::generic_category().message(errno);
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(
                LearningPackStatus::ioFailure,
                "could not atomically publish learning pack: " +
                    error
            );
        }
        return {
            .status = LearningPackStatus::success,
            .contentHash = header.contentHash,
        };
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "learning pack allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::ioFailure,
            error.what()
        );
    }
}

template <typename Pack, typename Deserialize>
LearningPackResult readPack(
    const std::filesystem::path& path,
    const std::uint32_t expectedKind,
    const std::uint32_t expectedVersion,
    Pack& output,
    Deserialize&& deserialize
) {
    try {
        std::error_code sizeError;
        const std::uintmax_t fileBytes =
            std::filesystem::file_size(path, sizeError);
        if (sizeError ||
            fileBytes < sizeof(LearningPackFileHeader) ||
            fileBytes >
                sizeof(LearningPackFileHeader) +
                    kMaximumPayloadBytes) {
            return fail(
                LearningPackStatus::ioFailure,
                "learning pack is missing or has an invalid length"
            );
        }
        std::ifstream stream(path, std::ios::binary);
        LearningPackFileHeader header;
        if (!stream.read(
                reinterpret_cast<char*>(&header),
                sizeof(header)
            ) ||
            header.magic != kMagic ||
            header.kind != expectedKind) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack header or kind is invalid"
            );
        }
        if (header.formatVersion != expectedVersion) {
            return fail(
                LearningPackStatus::unsupportedVersion,
                "learning pack wire version is unsupported"
            );
        }
        if (header.payloadBytes !=
            fileBytes - sizeof(header)) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack payload length is inconsistent"
            );
        }
        std::vector<std::byte> payload(
            static_cast<std::size_t>(header.payloadBytes)
        );
        if (!payload.empty() &&
            !stream.read(
                reinterpret_cast<char*>(payload.data()),
                static_cast<std::streamsize>(payload.size())
            )) {
            return fail(
                LearningPackStatus::ioFailure,
                "could not read learning pack payload"
            );
        }
        if (contentHash(payload) != header.contentHash) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack content fingerprint does not match"
            );
        }
        Pack candidate;
        if (!deserialize(payload, candidate)) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack payload is malformed"
            );
        }
        output = std::move(candidate);
        return {
            .status = LearningPackStatus::success,
            .contentHash = header.contentHash,
        };
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "learning pack allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::ioFailure,
            error.what()
        );
    }
}

} // namespace

std::uint64_t learningPackContentHash(
    const std::span<const std::byte> payload
) noexcept {
    return contentHash(payload);
}

LearningPackResult writeTaskPack(
    const TaskPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validateTaskArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializeTask(pack),
            kTaskKind,
            kTaskPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readTaskPack(
    const std::filesystem::path& path,
    TaskPack& output
) {
    return readPack(
        path,
        kTaskKind,
        kTaskPackFormatVersion,
        output,
        deserializeTask
    );
}

LearningPackResult writePolicyPack(
    const PolicyPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validatePolicyArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializePolicy(pack),
            kPolicyKind,
            kPolicyPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readPolicyPack(
    const std::filesystem::path& path,
    PolicyPack& output
) {
    return readPack(
        path,
        kPolicyKind,
        kPolicyPackFormatVersion,
        output,
        deserializePolicy
    );
}

LearningPackResult writePolicyRolloutPack(
    const PolicyRolloutPack& pack,
    const std::filesystem::path& path
) {
    return writePolicyRolloutPack(
        PolicyRolloutPackView{
            .id = pack.id,
            .taskFingerprint = pack.taskFingerprint,
            .policyFingerprint = pack.policyFingerprint,
            .policyRevision = pack.policyRevision,
            .environmentCount = pack.environmentCount,
            .controlStepCount = pack.controlStepCount,
            .actorObservationCount =
                pack.actorObservationCount,
            .criticObservationCount =
                pack.criticObservationCount,
            .actionCount = pack.actionCount,
            .motionFeatureCount = pack.motionFeatureCount,
            .actorObservations = pack.actorObservations,
            .criticObservations = pack.criticObservations,
            .motionFeatures = pack.motionFeatures,
            .teacherActions = pack.teacherActions,
            .latents = pack.latents,
            .logProbabilities = pack.logProbabilities,
            .values = pack.values,
            .bootstrapValues = pack.bootstrapValues,
            .transitions = pack.transitions,
        },
        path
    );
}

LearningPackResult writePolicyRolloutPack(
    const PolicyRolloutPackView& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validatePolicyRolloutArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializePolicyRollout(pack),
            kPolicyRolloutKind,
            kPolicyRolloutPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyRolloutPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readPolicyRolloutPack(
    const std::filesystem::path& path,
    PolicyRolloutPack& output
) {
    PolicyRolloutPack staged;
    LearningPackResult result = readPack(
        path,
        kPolicyRolloutKind,
        kPolicyRolloutPackFormatVersion,
        staged,
        deserializePolicyRollout
    );
    if (!result.succeeded()) {
        return result;
    }
    result = validatePolicyRolloutArtifact(staged);
    if (!result.succeeded()) {
        result.status = LearningPackStatus::corruptPayload;
        return result;
    }
    output = std::move(staged);
    return result;
}

LearningPackResult writeMotionPack(
    const MotionPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validateMotionArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializeMotion(pack),
            kMotionKind,
            kMotionPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "MotionPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readMotionPack(
    const std::filesystem::path& path,
    MotionPack& output
) {
    MotionPack staged;
    LearningPackResult result = readPack(
        path,
        kMotionKind,
        kMotionPackFormatVersion,
        staged,
        deserializeMotion
    );
    if (!result.succeeded()) {
        return result;
    }
    const LearningPackResult validation =
        validateMotionArtifact(staged);
    if (!validation.succeeded()) {
        return validation;
    }
    output = std::move(staged);
    return result;
}

LearningPackResult writeInteractionPack(
    const InteractionPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validateInteractionArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializeInteraction(pack),
            kInteractionKind,
            kInteractionPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "InteractionPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readInteractionPack(
    const std::filesystem::path& path,
    InteractionPack& output
) {
    InteractionPack staged;
    LearningPackResult result = readPack(
        path,
        kInteractionKind,
        kInteractionPackFormatVersion,
        staged,
        deserializeInteraction
    );
    if (!result.succeeded()) {
        return result;
    }
    const LearningPackResult validation =
        validateInteractionArtifact(staged);
    if (!validation.succeeded()) {
        LearningPackResult corrupt = validation;
        corrupt.status = LearningPackStatus::corruptPayload;
        return corrupt;
    }
    output = std::move(staged);
    return result;
}

bool validInteractionPack(
    const InteractionPack& pack
) noexcept {
    try {
        return validateInteractionArtifact(pack).succeeded();
    } catch (...) {
        return false;
    }
}

const char* learningPackStatusName(
    const LearningPackStatus status
) noexcept {
    switch (status) {
    case LearningPackStatus::success:
        return "success";
    case LearningPackStatus::invalidPack:
        return "invalid_pack";
    case LearningPackStatus::ioFailure:
        return "io_failure";
    case LearningPackStatus::unsupportedVersion:
        return "unsupported_version";
    case LearningPackStatus::corruptPayload:
        return "corrupt_payload";
    case LearningPackStatus::capacityOverflow:
        return "capacity_overflow";
    case LearningPackStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
