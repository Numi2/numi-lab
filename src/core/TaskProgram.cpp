#include "metalrobo/TaskProgram.hpp"

#include "metalrobo/MetalWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {

static_assert(
    kInteractionContactFeatureCount ==
        MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT
);

struct CompiledTaskProgram::Storage {
    std::uint64_t fingerprint = 0u;
    std::uint64_t worldFingerprint = 0u;
    TaskProgramLayout layout{};
    MRTaskProgramHeaderGPU header{};
    std::vector<MRTaskActionBindingGPU> actionBindings;
    std::vector<MRTaskObservationOperatorGPU> actorOperators;
    std::vector<MRTaskObservationOperatorGPU> criticOperators;
    std::vector<MRTaskContactGroupGPU> contactGroups;
    std::vector<std::uint32_t> contactMembers;
    std::vector<float> contactMemberRadii;
    std::vector<MRTaskIndexGroupGPU> jointGroups;
    std::vector<std::uint32_t> jointMembers;
    std::vector<MRTaskRewardOperatorGPU> rewardOperators;
    std::vector<MRTaskTerminationOperatorGPU> terminationOperators;
    std::vector<MRTaskRandomizationOperatorGPU>
        randomizationOperators;
    std::vector<MRTaskImpactEventGPU> impactEvents;
    std::vector<std::uint32_t> motionBodies;
    std::vector<float> interactionRootTargets;
    std::vector<float> interactionJointTargets;
    std::vector<MRTaskInteractionContactGPU> interactionContacts;
    std::vector<MRTaskInteractionSampleGPU> interactionSamples;
    std::vector<float> interactionContactTargets;
    std::vector<float> interactionContactTolerances;
    std::vector<MRTaskBiasSpecGPU> biasSpecs;
    std::vector<mr_float4> terrainSampleOffsets;
    std::vector<mr_float4> terrainResetTranslations;
    std::array<mr_float4, 3u> commandCurriculum{};
    std::vector<std::byte> arena;
};

namespace {

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

class Hash {
public:
    void bytes(const void* data, const std::size_t size) {
        const auto* values =
            static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= values[index];
            value_ *= kFNVPrime;
        }
    }

    template <typename T>
    void scalar(const T& value) {
        bytes(&value, sizeof(value));
    }

    void string(const std::string_view value) {
        scalar<std::uint64_t>(value.size());
        bytes(value.data(), value.size());
    }

    template <typename T>
    void span(const std::span<const T> values) {
        scalar<std::uint64_t>(values.size());
        if (!values.empty()) {
            bytes(values.data(), values.size_bytes());
        }
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_ = kFNVOffset;
};

TaskCompileDiagnostics reject(
    const TaskCompileStatus status,
    std::string element,
    std::string message
) {
    return {
        .status = status,
        .element = std::move(element),
        .message = std::move(message),
    };
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

template <typename T>
bool narrowCount(const std::size_t value, T& output) {
    if (value > std::numeric_limits<T>::max()) {
        return false;
    }
    output = static_cast<T>(value);
    return true;
}

std::uint32_t uniqueIndex(
    const std::vector<std::string>& names,
    const std::string_view target,
    bool& ambiguous
) {
    std::uint32_t result = MR_INVALID_INDEX;
    ambiguous = false;
    for (std::size_t index = 0u; index < names.size(); ++index) {
        if (names[index] != target) {
            continue;
        }
        if (result != MR_INVALID_INDEX) {
            ambiguous = true;
            return MR_INVALID_INDEX;
        }
        result = static_cast<std::uint32_t>(index);
    }
    return result;
}

std::uint32_t namedGroup(
    const std::span<const std::string> ids,
    const std::string_view target
) {
    for (std::size_t index = 0u; index < ids.size(); ++index) {
        if (ids[index] == target) {
            return static_cast<std::uint32_t>(index);
        }
    }
    return MR_INVALID_INDEX;
}

bool uniqueNonempty(const std::span<const std::string> values) {
    for (std::size_t index = 0u; index < values.size(); ++index) {
        if (values[index].empty() ||
            std::find(
                values.begin(),
                values.begin() +
                    static_cast<std::ptrdiff_t>(index),
                values[index]
            ) !=
                values.begin() +
                    static_cast<std::ptrdiff_t>(index)) {
            return false;
        }
    }
    return true;
}

std::uint32_t actionIndexForJoint(
    const std::vector<std::uint32_t>& actionJoints,
    const std::uint32_t joint
) {
    const auto found = std::find(
        actionJoints.begin(),
        actionJoints.end(),
        joint
    );
    return found == actionJoints.end()
        ? MR_INVALID_INDEX
        : static_cast<std::uint32_t>(
              found - actionJoints.begin()
          );
}

bool inRange(
    const std::uint32_t value,
    const std::uint32_t first,
    const std::uint32_t count
) {
    return value >= first &&
        static_cast<std::uint64_t>(value - first) < count;
}

bool orderedRange(const mr_float4 values) {
    return values.x <= values.y;
}

bool integerStepRange(
    const mr_float4 values,
    const std::uint32_t maximum
) {
    return orderedRange(values) &&
        values.x >= 0.0f &&
        values.y <= static_cast<float>(maximum) &&
        std::floor(values.x) == values.x &&
        std::floor(values.y) == values.y;
}

struct CapacityField {
    std::string_view name;
    std::uint32_t MetalWorldCapacityProfile::* value;
};

constexpr std::array kCapacityFields{
    CapacityField{
        "candidatePairs",
        &MetalWorldCapacityProfile::candidatePairs,
    },
    CapacityField{
        "rawContacts",
        &MetalWorldCapacityProfile::rawContacts,
    },
    CapacityField{
        "manifolds",
        &MetalWorldCapacityProfile::manifolds,
    },
    CapacityField{
        "constraintBlocks",
        &MetalWorldCapacityProfile::constraintBlocks,
    },
    CapacityField{
        "constraintRows",
        &MetalWorldCapacityProfile::constraintRows,
    },
    CapacityField{
        "islands",
        &MetalWorldCapacityProfile::islands,
    },
    CapacityField{
        "hardConvexPairs",
        &MetalWorldCapacityProfile::hardConvexPairs,
    },
    CapacityField{
        "meshTriangleCandidates",
        &MetalWorldCapacityProfile::meshTriangleCandidates,
    },
    CapacityField{
        "solverTiles",
        &MetalWorldCapacityProfile::solverTiles,
    },
    CapacityField{
        "spillRows",
        &MetalWorldCapacityProfile::spillRows,
    },
    CapacityField{
        "ccdCandidates",
        &MetalWorldCapacityProfile::ccdCandidates,
    },
    CapacityField{
        "ccdEvents",
        &MetalWorldCapacityProfile::ccdEvents,
    },
    CapacityField{
        "endpointRuntimeRecords",
        &MetalWorldCapacityProfile::endpointRuntimeRecords,
    },
    CapacityField{
        "articulationPointQueries",
        &MetalWorldCapacityProfile::articulationPointQueries,
    },
    CapacityField{
        "rodCandidatePairs",
        &MetalWorldCapacityProfile::rodCandidatePairs,
    },
    CapacityField{
        "rodRawContacts",
        &MetalWorldCapacityProfile::rodRawContacts,
    },
    CapacityField{
        "rodManifolds",
        &MetalWorldCapacityProfile::rodManifolds,
    },
    CapacityField{
        "rodCCDEvents",
        &MetalWorldCapacityProfile::rodCCDEvents,
    },
    CapacityField{
        "qualityGeneralizedVelocities",
        &MetalWorldCapacityProfile::qualityGeneralizedVelocities,
    },
    CapacityField{
        "qualityRows",
        &MetalWorldCapacityProfile::qualityRows,
    },
    CapacityField{
        "qualityKrylovVectors",
        &MetalWorldCapacityProfile::qualityKrylovVectors,
    },
    CapacityField{
        "qualityDirectTiles",
        &MetalWorldCapacityProfile::qualityDirectTiles,
    },
    CapacityField{
        "dynamicNodes",
        &MetalWorldCapacityProfile::dynamicNodes,
    },
    CapacityField{
        "islandNodeReferences",
        &MetalWorldCapacityProfile::islandNodeReferences,
    },
    CapacityField{
        "islandConstraintReferences",
        &MetalWorldCapacityProfile::islandConstraintReferences,
    },
    CapacityField{
        "rodFactorBlocks",
        &MetalWorldCapacityProfile::rodFactorBlocks,
    },
    CapacityField{
        "operatorVelocityElements",
        &MetalWorldCapacityProfile::operatorVelocityElements,
    },
};

} // namespace

bool CompiledTaskProgram::valid() const noexcept {
    return storage_ != nullptr &&
        storage_->fingerprint != 0u &&
        storage_->worldFingerprint != 0u &&
        storage_->header.taskFingerprint ==
            storage_->fingerprint &&
        storage_->header.worldFingerprint ==
            storage_->worldFingerprint &&
        storage_->layout.actionCount != 0u &&
        storage_->layout.actorFrameSize != 0u &&
        storage_->layout.actorHistoryLength != 0u;
}

std::uint64_t CompiledTaskProgram::fingerprint() const noexcept {
    return valid() ? storage_->fingerprint : 0u;
}

std::uint64_t CompiledTaskProgram::worldFingerprint() const noexcept {
    return valid() ? storage_->worldFingerprint : 0u;
}

const TaskProgramLayout& CompiledTaskProgram::layout() const noexcept {
    static const TaskProgramLayout empty{};
    return valid() ? storage_->layout : empty;
}

const MRTaskProgramHeaderGPU&
CompiledTaskProgram::header() const noexcept {
    static const MRTaskProgramHeaderGPU empty{};
    return valid() ? storage_->header : empty;
}

std::span<const MRTaskActionBindingGPU>
CompiledTaskProgram::actionBindings() const noexcept {
    return valid()
        ? std::span<const MRTaskActionBindingGPU>{
              storage_->actionBindings
          }
        : std::span<const MRTaskActionBindingGPU>{};
}

std::span<const MRTaskObservationOperatorGPU>
CompiledTaskProgram::actorOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskObservationOperatorGPU>{
              storage_->actorOperators
          }
        : std::span<const MRTaskObservationOperatorGPU>{};
}

std::span<const MRTaskObservationOperatorGPU>
CompiledTaskProgram::criticOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskObservationOperatorGPU>{
              storage_->criticOperators
          }
        : std::span<const MRTaskObservationOperatorGPU>{};
}

std::span<const MRTaskContactGroupGPU>
CompiledTaskProgram::contactGroups() const noexcept {
    return valid()
        ? std::span<const MRTaskContactGroupGPU>{
              storage_->contactGroups
          }
        : std::span<const MRTaskContactGroupGPU>{};
}

std::span<const std::uint32_t>
CompiledTaskProgram::contactMembers() const noexcept {
    return valid()
        ? std::span<const std::uint32_t>{
              storage_->contactMembers
          }
        : std::span<const std::uint32_t>{};
}

std::span<const float>
CompiledTaskProgram::contactMemberRadii() const noexcept {
    return valid()
        ? std::span<const float>{storage_->contactMemberRadii}
        : std::span<const float>{};
}

std::span<const MRTaskIndexGroupGPU>
CompiledTaskProgram::jointGroups() const noexcept {
    return valid()
        ? std::span<const MRTaskIndexGroupGPU>{
              storage_->jointGroups
          }
        : std::span<const MRTaskIndexGroupGPU>{};
}

std::span<const std::uint32_t>
CompiledTaskProgram::jointMembers() const noexcept {
    return valid()
        ? std::span<const std::uint32_t>{
              storage_->jointMembers
          }
        : std::span<const std::uint32_t>{};
}

std::span<const MRTaskRewardOperatorGPU>
CompiledTaskProgram::rewardOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskRewardOperatorGPU>{
              storage_->rewardOperators
          }
        : std::span<const MRTaskRewardOperatorGPU>{};
}

std::span<const MRTaskTerminationOperatorGPU>
CompiledTaskProgram::terminationOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskTerminationOperatorGPU>{
              storage_->terminationOperators
          }
        : std::span<const MRTaskTerminationOperatorGPU>{};
}

std::span<const MRTaskRandomizationOperatorGPU>
CompiledTaskProgram::randomizationOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskRandomizationOperatorGPU>{
              storage_->randomizationOperators
          }
        : std::span<const MRTaskRandomizationOperatorGPU>{};
}

std::span<const MRTaskImpactEventGPU>
CompiledTaskProgram::impactEvents() const noexcept {
    return valid()
        ? std::span<const MRTaskImpactEventGPU>{
              storage_->impactEvents
          }
        : std::span<const MRTaskImpactEventGPU>{};
}

std::span<const std::uint32_t>
CompiledTaskProgram::motionBodies() const noexcept {
    return valid()
        ? std::span<const std::uint32_t>{storage_->motionBodies}
        : std::span<const std::uint32_t>{};
}

std::span<const float>
CompiledTaskProgram::interactionRootTargets() const noexcept {
    return valid()
        ? std::span<const float>{storage_->interactionRootTargets}
        : std::span<const float>{};
}

std::span<const float>
CompiledTaskProgram::interactionJointTargets() const noexcept {
    return valid()
        ? std::span<const float>{storage_->interactionJointTargets}
        : std::span<const float>{};
}

std::span<const MRTaskInteractionContactGPU>
CompiledTaskProgram::interactionContacts() const noexcept {
    return valid()
        ? std::span<const MRTaskInteractionContactGPU>{
              storage_->interactionContacts
          }
        : std::span<const MRTaskInteractionContactGPU>{};
}

std::span<const MRTaskInteractionSampleGPU>
CompiledTaskProgram::interactionSamples() const noexcept {
    return valid()
        ? std::span<const MRTaskInteractionSampleGPU>{
              storage_->interactionSamples
          }
        : std::span<const MRTaskInteractionSampleGPU>{};
}

std::span<const float>
CompiledTaskProgram::interactionContactTargets() const noexcept {
    return valid()
        ? std::span<const float>{storage_->interactionContactTargets}
        : std::span<const float>{};
}

std::span<const float>
CompiledTaskProgram::interactionContactTolerances() const noexcept {
    return valid()
        ? std::span<const float>{
              storage_->interactionContactTolerances
          }
        : std::span<const float>{};
}

std::span<const MRTaskBiasSpecGPU>
CompiledTaskProgram::biasSpecs() const noexcept {
    return valid()
        ? std::span<const MRTaskBiasSpecGPU>{
              storage_->biasSpecs
          }
        : std::span<const MRTaskBiasSpecGPU>{};
}

std::span<const mr_float4>
CompiledTaskProgram::terrainSampleOffsets() const noexcept {
    return valid()
        ? std::span<const mr_float4>{
              storage_->terrainSampleOffsets
          }
        : std::span<const mr_float4>{};
}

std::span<const mr_float4>
CompiledTaskProgram::terrainResetTranslations() const noexcept {
    return valid()
        ? std::span<const mr_float4>{
              storage_->terrainResetTranslations
          }
        : std::span<const mr_float4>{};
}

std::span<const std::byte>
CompiledTaskProgram::arena() const noexcept {
    return valid()
        ? std::span<const std::byte>{storage_->arena}
        : std::span<const std::byte>{};
}

TaskCompileDiagnostics compileTaskProgram(
    const TaskPack& pack,
    const CompiledWorld& world,
    CompiledTaskProgram& output
) {
    static const InteractionPack noInteraction;
    return compileTaskProgram(
        pack,
        noInteraction,
        {},
        world,
        output
    );
}

TaskCompileDiagnostics compileTaskProgram(
    const TaskPack& pack,
    const InteractionPack& interactions,
    const std::string_view clipId,
    const CompiledWorld& world,
    CompiledTaskProgram& output
) {
    if (!world.valid()) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world",
            "task compilation requires a valid compiled world"
        );
    }
    const InteractionClip* interactionClip = nullptr;
    if (!clipId.empty()) {
        if (!validInteractionPack(interactions)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "interaction",
                "selected InteractionPack is structurally invalid"
            );
        }
        for (const InteractionClip& candidate : interactions.clips) {
            if (candidate.id != clipId) {
                continue;
            }
            if (interactionClip != nullptr) {
                return reject(
                    TaskCompileStatus::ambiguousSemantic,
                    std::string{clipId},
                    "InteractionPack clip identity is ambiguous"
                );
            }
            interactionClip = &candidate;
        }
        if (interactionClip == nullptr) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                std::string{clipId},
                "InteractionPack clip does not exist"
            );
        }
    }
    const EngineModel& model = world.model();
    if (model.bodyNames.size() != model.bodies.size() ||
        model.jointNames.size() != model.joints.size() ||
        !uniqueNonempty(model.bodyNames) ||
        !uniqueNonempty(model.jointNames)) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world.semantics",
            "compiled task worlds require canonical body and joint names"
        );
    }
    if (world.articulationIndex() >=
        model.articulations.size()) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world.articulation",
            "compiled world has no selected articulation"
        );
    }
    for (const CapacityField& field : kCapacityFields) {
        const std::uint32_t requested =
            pack.capacities.*(field.value);
        const std::uint32_t compiled =
            world.capacities().*(field.value);
        if (requested != 0u && requested != compiled) {
            return reject(
                TaskCompileStatus::invalidWorld,
                "capacities." + std::string{field.name},
                "compiled world does not match the TaskPack capacity contract"
            );
        }
    }
    const MRArticulationGPU& articulation =
        model.articulations[world.articulationIndex()];
    if (articulation.rootType != MR_ROOT_FLOATING ||
        articulation.nq < 7u ||
        articulation.nv < 6u) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world.articulation",
            "locomotion task programs require a floating root"
        );
    }
    const auto countFits = [](const std::size_t count) {
        return count <
            std::numeric_limits<std::uint32_t>::max();
    };
    if (!countFits(pack.actions.size()) ||
        !countFits(pack.actorFrame.size()) ||
        !countFits(pack.actorCurrent.size()) ||
        !countFits(pack.critic.size()) ||
        !countFits(pack.contactGroups.size()) ||
        !countFits(pack.jointGroups.size()) ||
        !countFits(pack.rewards.size()) ||
        !countFits(pack.terminations.size()) ||
        !countFits(pack.randomization.size()) ||
        !countFits(pack.terrain.sampleOffsets.size()) ||
        !countFits(pack.terrain.resetTranslations.size())) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "task",
            "task table count exceeds the 32-bit GPU ABI"
        );
    }
    std::uint64_t contactMemberCount = 0u;
    for (const TaskContactGroup& group : pack.contactGroups) {
        contactMemberCount += group.bodies.size();
    }
    std::uint64_t jointMemberCount = 0u;
    for (const TaskJointGroup& group : pack.jointGroups) {
        jointMemberCount += group.joints.size();
    }
    const std::uint64_t observationOperatorCount =
        static_cast<std::uint64_t>(pack.actorFrame.size()) +
        pack.actorCurrent.size() +
        pack.critic.size();
    if (contactMemberCount >=
            std::numeric_limits<std::uint32_t>::max() ||
        jointMemberCount >=
            std::numeric_limits<std::uint32_t>::max() ||
        observationOperatorCount >=
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "task",
            "task member or observation count exceeds the 32-bit GPU ABI"
        );
    }
    if (pack.id.empty() || pack.actions.empty() ||
        pack.actorFrame.empty() ||
        pack.actorHistoryLength == 0u ||
        pack.criticHistoryLength == 0u ||
        pack.maximumEpisodeSteps == 0u ||
        pack.difficultyBandCount == 0u ||
        !finite(pack.interactionStudentAuthority) ||
        pack.interactionStudentAuthority < 0.0f ||
        pack.interactionStudentAuthority > 1.0f ||
        !finite(pack.interactionResetPhaseFraction) ||
        pack.interactionResetPhaseFraction < 0.0f ||
        pack.interactionResetPhaseFraction > 1.0f ||
        !finite(pack.baseHeightTarget) ||
        !finite(pack.gaitPeriodSeconds) ||
        !(pack.gaitPeriodSeconds > 0.0f) ||
        !finite(pack.clearanceTarget) ||
        !finite(pack.successTrackingThreshold) ||
        !finite(pack.supportForceThreshold) ||
        !(pack.supportForceThreshold >= 0.0f) ||
        !finite(pack.commands.lower) ||
        !finite(pack.commands.upper) ||
        !finite(pack.commands.limitLower) ||
        !finite(pack.commands.limitUpper) ||
        !finite(pack.commands.difficultyStep) ||
        !finite(pack.commands.standingProbability) ||
        pack.commands.standingProbability < 0.0f ||
        pack.commands.standingProbability > 1.0f ||
        !finite(pack.commands.difficultySamplingExponent) ||
        !(pack.commands.difficultySamplingExponent > 0.0f) ||
        !finite(pack.commands.minimumDurationSeconds) ||
        !finite(pack.commands.maximumDurationSeconds) ||
        !(pack.commands.minimumDurationSeconds > 0.0f) ||
        pack.commands.maximumDurationSeconds <
            pack.commands.minimumDurationSeconds ||
        !finite(pack.pushes.maximumVelocity) ||
        pack.pushes.maximumVelocity < 0.0f ||
        !finite(pack.pushes.minimumIntervalSeconds) ||
        !finite(pack.pushes.maximumIntervalSeconds) ||
        !finite(pack.pushes.projectileStandingProbability) ||
        pack.pushes.projectileStandingProbability < 0.0f ||
        pack.pushes.projectileStandingProbability > 1.0f ||
        !finite(pack.pushes.projectileTargetHorizontalRadius) ||
        pack.pushes.projectileTargetHorizontalRadius < 0.0f ||
        !finite(pack.pushes.projectileHorizontalSpeedLower) ||
        !finite(pack.pushes.projectileHorizontalSpeedUpper) ||
        !finite(pack.pushes.projectileTargetHeightLower) ||
        !finite(pack.pushes.projectileTargetHeightUpper) ||
        pack.pushes.projectileHorizontalSpeedLower < 0.0f ||
        pack.pushes.projectileHorizontalSpeedUpper <
            pack.pushes.projectileHorizontalSpeedLower ||
        (pack.pushes.projectileHorizontalSpeedUpper > 0.0f &&
         !(pack.pushes.projectileHorizontalSpeedLower > 0.0f)) ||
        pack.pushes.projectileTargetHeightUpper <
            pack.pushes.projectileTargetHeightLower ||
        !(pack.pushes.minimumIntervalSeconds > 0.0f) ||
        pack.pushes.maximumIntervalSeconds <
            pack.pushes.minimumIntervalSeconds ||
        pack.maximumObservationDelaySteps >=
            pack.actorHistoryLength) {
        return reject(
            TaskCompileStatus::invalidPack,
            "task",
            "task identity, dimensions, timing, or scalar parameters are invalid"
        );
    }
    const bool hasVisualProgram = pack.visual.width != 0u ||
        pack.visual.height != 0u ||
        !pack.visual.frameOffsets.empty();
    if (hasVisualProgram &&
        (pack.visual.width == 0u || pack.visual.height == 0u ||
         pack.visual.frameOffsets.empty() ||
         pack.visual.frameOffsets.size() > 4u ||
         pack.visual.frameOffsets.front() != 0u ||
         !std::ranges::is_sorted(pack.visual.frameOffsets) ||
         std::adjacent_find(
             pack.visual.frameOffsets.begin(),
             pack.visual.frameOffsets.end()
         ) !=
             pack.visual.frameOffsets.end() ||
         !finite(pack.visual.nearDepthMeters) ||
         !finite(pack.visual.farDepthMeters) ||
         !(pack.visual.nearDepthMeters > 0.0f) ||
         !(pack.visual.farDepthMeters >
           pack.visual.nearDepthMeters) ||
         !finite(pack.visual.fullDropoutProbability) ||
         pack.visual.fullDropoutProbability < 0.0f ||
         pack.visual.fullDropoutProbability > 1.0f ||
         !finite(pack.visual.pixelDropoutProbability) ||
         pack.visual.pixelDropoutProbability < 0.0f ||
         pack.visual.pixelDropoutProbability > 1.0f ||
         !finite(pack.visual.depthJitterMeters) ||
         pack.visual.depthJitterMeters < 0.0f ||
         !finite(pack.visual.depthNoiseSigmaMeters) ||
         pack.visual.depthNoiseSigmaMeters < 0.0f ||
         !finite(pack.visual.edgeFlickerProbability) ||
         pack.visual.edgeFlickerProbability < 0.0f ||
         pack.visual.edgeFlickerProbability > 1.0f ||
         !finite(pack.visual.difficultyCorruptionGain) ||
         pack.visual.difficultyCorruptionGain < 0.0f)) {
        return reject(
            TaskCompileStatus::invalidPack,
            "visual",
            "visual depth requires dimensions, one to four unique sorted offsets beginning at zero, and a finite positive range"
        );
    }
    const std::uint64_t visualPixelComponentCount =
        static_cast<std::uint64_t>(pack.visual.width) *
        pack.visual.height * pack.visual.frameOffsets.size();
    const std::uint64_t visualFeatureComponentCount =
        pack.visual.includeDerivedFeatures
        ? MR_TASK_MASKED_DEPTH_FEATURE_COUNT
        : 0u;
    const std::uint64_t visualComponentCount =
        visualPixelComponentCount + visualFeatureComponentCount;
    if (visualComponentCount >
        std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "visual",
            "visual observation layout exceeds the 32-bit task ABI"
        );
    }
    const std::array commandLower{
        pack.commands.lower.x,
        pack.commands.lower.y,
        pack.commands.lower.z,
    };
    const std::array commandUpper{
        pack.commands.upper.x,
        pack.commands.upper.y,
        pack.commands.upper.z,
    };
    const std::array commandLimitLower{
        pack.commands.limitLower.x,
        pack.commands.limitLower.y,
        pack.commands.limitLower.z,
    };
    const std::array commandLimitUpper{
        pack.commands.limitUpper.x,
        pack.commands.limitUpper.y,
        pack.commands.limitUpper.z,
    };
    const std::array commandCurriculumStep{
        pack.commands.difficultyStep.x,
        pack.commands.difficultyStep.y,
        pack.commands.difficultyStep.z,
    };
    for (std::size_t component = 0u;
         component < commandLower.size();
         ++component) {
        if (commandLower[component] >
                commandUpper[component] ||
            commandLimitLower[component] >
                commandLower[component] ||
            commandUpper[component] >
                commandLimitUpper[component] ||
            commandCurriculumStep[component] < 0.0f) {
            return reject(
                TaskCompileStatus::invalidPack,
                "commands",
                "command range, limits, or difficulty step are invalid"
            );
        }
    }

    auto staged = std::make_shared<CompiledTaskProgram::Storage>();
    staged->worldFingerprint = world.fingerprint();

    std::vector<std::uint32_t> actionJoints;
    actionJoints.reserve(pack.actions.size());
    staged->actionBindings.reserve(pack.actions.size());
    for (std::size_t actionIndex = 0u;
         actionIndex < pack.actions.size();
         ++actionIndex) {
        const TaskActionBinding& binding =
            pack.actions[actionIndex];
        bool ambiguous = false;
        const std::uint32_t jointIndex = uniqueIndex(
            model.jointNames,
            binding.joint,
            ambiguous
        );
        if (ambiguous) {
            return reject(
                TaskCompileStatus::ambiguousSemantic,
                binding.joint,
                "action joint identity is ambiguous"
            );
        }
        if (jointIndex == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                binding.joint,
                "action joint does not exist in the compiled world"
            );
        }
        if (std::find(
                actionJoints.begin(),
                actionJoints.end(),
                jointIndex
            ) != actionJoints.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                binding.joint,
                "a joint may have only one action binding"
            );
        }
        const auto dofFound = std::find_if(
            model.dofs.begin(),
            model.dofs.end(),
            [jointIndex](const MRDofPropertiesGPU& dof) {
                return dof.jointIndex == jointIndex;
            }
        );
        const std::size_t jointDofCount = std::count_if(
            model.dofs.begin(),
            model.dofs.end(),
            [jointIndex](const MRDofPropertiesGPU& dof) {
                return dof.jointIndex == jointIndex;
            }
        );
        const MRJointDescriptorGPU& joint =
            model.joints[jointIndex];
        if (dofFound == model.dofs.end() ||
            jointDofCount != 1u ||
            !inRange(
                jointIndex,
                articulation.firstJoint,
                articulation.jointCount
            ) ||
            joint.nq != 1u ||
            joint.nv != 1u ||
            dofFound->articulationIndex !=
                world.articulationIndex() ||
            (dofFound->flags & MR_DOF_FLAG_ACTUATED) == 0u ||
            dofFound->qIndex == MR_INVALID_INDEX ||
            dofFound->vIndex == MR_INVALID_INDEX ||
            !inRange(
                dofFound->qIndex,
                articulation.qOffset,
                articulation.nq
            ) ||
            !inRange(
                dofFound->vIndex,
                articulation.vOffset,
                articulation.nv
            ) ||
            !finite(binding.scale) ||
            !(binding.scale > 0.0f) ||
            !finite(binding.responseTimeSeconds) ||
            binding.responseTimeSeconds < 0.0f) {
            return reject(
                TaskCompileStatus::invalidPack,
                binding.joint,
                "action binding requires one actuated scalar position DoF"
            );
        }
        const std::uint32_t dofIndex =
            static_cast<std::uint32_t>(
                dofFound - model.dofs.begin()
            );
        actionJoints.push_back(jointIndex);
        staged->actionBindings.push_back({
            {
                static_cast<std::uint32_t>(actionIndex),
                dofIndex,
                dofFound->qIndex,
                dofFound->vIndex,
            },
            {
                binding.scale,
                dofFound->limits.x,
                dofFound->limits.y,
                binding.responseTimeSeconds,
            },
            {
                dofFound->drive.x,
                dofFound->drive.y,
                0.0f,
                0.0f,
            },
        });
    }

    std::vector<std::string> contactGroupIds;
    contactGroupIds.reserve(pack.contactGroups.size());
    staged->contactGroups.reserve(pack.contactGroups.size());
    std::uint32_t contactMetricCount = 0u;
    for (const TaskContactGroup& group : pack.contactGroups) {
        const bool hasSupportPatch =
            group.supportPatchWidth != 0u ||
            group.supportPatchHeight != 0u;
        const std::uint64_t supportPatchCellCount =
            static_cast<std::uint64_t>(group.supportPatchWidth) *
            group.supportPatchHeight;
        if (group.id.empty() || group.bodies.empty() ||
            !finite(group.localReference) ||
            !finite(group.gaitPhaseOffsetRadians) ||
            !finite(group.stanceFraction) ||
            group.stanceFraction < 0.0f ||
            group.stanceFraction > 1.0f ||
            std::find(
                contactGroupIds.begin(),
                contactGroupIds.end(),
                group.id
            ) != contactGroupIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                group.id,
                "contact group identity, members, or parameters are invalid"
            );
        }
        if (hasSupportPatch &&
            (!group.support ||
             group.supportPatchWidth == 0u ||
             group.supportPatchHeight == 0u ||
             supportPatchCellCount > 64u ||
             !finite(group.supportPatchBounds) ||
             !(group.supportPatchBounds.z >
                 group.supportPatchBounds.x) ||
             !(group.supportPatchBounds.w >
                 group.supportPatchBounds.y))) {
            return reject(
                TaskCompileStatus::invalidPack,
                group.id,
                "support patch requires a support group, finite ordered bounds, and 1 to 64 cells"
            );
        }
        if (group.support && group.forbidden) {
            return reject(
                TaskCompileStatus::invalidPack,
                group.id,
                "a contact group cannot be both support and forbidden"
            );
        }
        const std::uint32_t memberOffset =
            static_cast<std::uint32_t>(
                staged->contactMembers.size()
            );
        std::vector<std::uint32_t> resolvedMembers;
        for (const std::string& bodyName : group.bodies) {
            bool ambiguous = false;
            const std::uint32_t body = uniqueIndex(
                model.bodyNames,
                bodyName,
                ambiguous
            );
            if (ambiguous) {
                return reject(
                    TaskCompileStatus::ambiguousSemantic,
                    bodyName,
                    "contact body identity is ambiguous"
                );
            }
            if (body == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    bodyName,
                    "contact body does not exist in the compiled world"
                );
            }
            if (!inRange(
                    body,
                    articulation.firstBody,
                    articulation.bodyCount
                )) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    bodyName,
                    "contact groups may contain only selected-articulation bodies"
                );
            }
            if (std::find(
                    resolvedMembers.begin(),
                    resolvedMembers.end(),
                    body
                ) == resolvedMembers.end()) {
                resolvedMembers.push_back(body);
            }
        }
        staged->contactMembers.insert(
            staged->contactMembers.end(),
            resolvedMembers.begin(),
            resolvedMembers.end()
        );
        for (const std::uint32_t body : resolvedMembers) {
            float radius = 0.0f;
            for (const MRShapeGPU& shape : model.shapes) {
                if (shape.bodyIndex != body ||
                    (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u) {
                    continue;
                }
                const float offset = std::sqrt(
                    shape.localPosition.x * shape.localPosition.x +
                    shape.localPosition.y * shape.localPosition.y +
                    shape.localPosition.z * shape.localPosition.z
                );
                radius = std::max(
                    radius,
                    offset + shape.contactRestAndBoundingRadius.z
                );
            }
            if (radius < 0.0f || !finite(radius)) {
                return reject(
                    TaskCompileStatus::invalidWorld,
                    model.bodyNames[body],
                    "contact-group body has an invalid collision envelope"
                );
            }
            staged->contactMemberRadii.push_back(radius);
        }
        std::uint32_t referenceBody = resolvedMembers.front();
        if (!group.referenceBody.empty()) {
            bool ambiguous = false;
            referenceBody = uniqueIndex(
                model.bodyNames,
                group.referenceBody,
                ambiguous
            );
            if (ambiguous) {
                return reject(
                    TaskCompileStatus::ambiguousSemantic,
                    group.referenceBody,
                    "contact reference body identity is ambiguous"
                );
            }
            if (referenceBody == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    group.referenceBody,
                    "contact reference body does not exist"
                );
            }
            if (!inRange(
                    referenceBody,
                    articulation.firstBody,
                    articulation.bodyCount
                )) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    group.referenceBody,
                    "contact reference body is outside the selected articulation"
                );
            }
        }
        std::uint32_t flags = 0u;
        if (group.support) {
            flags |= MR_TASK_CONTACT_SUPPORT;
        }
        if (group.forbidden) {
            flags |= MR_TASK_CONTACT_FORBIDDEN;
        }
        const std::uint32_t metricOffset =
            contactMetricCount;
        const std::uint32_t supportCellCount =
            static_cast<std::uint32_t>(supportPatchCellCount);
        const std::uint32_t baseMetricWidth =
            group.support
            ? 6u + supportCellCount
            : group.forbidden ? 1u : 0u;
        constexpr std::uint32_t localWrenchWidth = 6u;
        const std::uint32_t metricWidth =
            baseMetricWidth + localWrenchWidth;
        if (contactMetricCount >
            std::numeric_limits<std::uint32_t>::max() -
                metricWidth) {
            return reject(
                TaskCompileStatus::arithmeticOverflow,
                group.id,
                "contact metric layout exceeds the 32-bit GPU ABI"
            );
        }
        contactMetricCount += metricWidth;
        const mr_float4 centerOfMass =
            model.bodies[referenceBody].centerOfMass;
        staged->contactGroups.push_back({
            {
                memberOffset,
                static_cast<std::uint32_t>(
                    resolvedMembers.size()
                ),
                flags,
                metricOffset,
            },
            {
                referenceBody,
                metricOffset + baseMetricWidth,
                0u,
                0u,
            },
            group.localReference,
            {
                -centerOfMass.x,
                -centerOfMass.y,
                -centerOfMass.z,
                0.0f,
            },
            {
                group.gaitPhaseOffsetRadians,
                group.stanceFraction,
                0.0f,
                0.0f,
            },
            group.supportPatchBounds,
            {
                group.supportPatchWidth,
                group.supportPatchHeight,
                supportCellCount,
                metricOffset + 6u,
            },
        });
        contactGroupIds.push_back(group.id);
    }

    std::vector<std::string> jointGroupIds;
    jointGroupIds.reserve(pack.jointGroups.size());
    staged->jointGroups.reserve(pack.jointGroups.size());
    for (const TaskJointGroup& group : pack.jointGroups) {
        if (group.id.empty() || group.joints.empty() ||
            std::find(
                jointGroupIds.begin(),
                jointGroupIds.end(),
                group.id
            ) != jointGroupIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                group.id,
                "joint group identity or members are invalid"
            );
        }
        const std::uint32_t offset =
            static_cast<std::uint32_t>(
                staged->jointMembers.size()
            );
        for (const std::string& jointName : group.joints) {
            bool ambiguous = false;
            const std::uint32_t joint = uniqueIndex(
                model.jointNames,
                jointName,
                ambiguous
            );
            const std::uint32_t action =
                joint == MR_INVALID_INDEX
                ? MR_INVALID_INDEX
                : actionIndexForJoint(actionJoints, joint);
            if (ambiguous) {
                return reject(
                    TaskCompileStatus::ambiguousSemantic,
                    jointName,
                    "joint-group identity is ambiguous"
                );
            }
            if (action == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    jointName,
                    "joint group member has no action binding"
                );
            }
            if (std::find(
                    staged->jointMembers.begin() + offset,
                    staged->jointMembers.end(),
                    action
                ) ==
                staged->jointMembers.end()) {
                staged->jointMembers.push_back(action);
            }
        }
        staged->jointGroups.push_back({
            {
                offset,
                static_cast<std::uint32_t>(
                    staged->jointMembers.size() - offset
                ),
                0u,
                0u,
            },
        });
        jointGroupIds.push_back(group.id);
    }

    std::vector<std::string> interactionContactIds;
    if (interactionClip != nullptr) {
        staged->interactionRootTargets =
            interactionClip->rootTargets;
        std::vector<std::uint32_t> interactionJointIndices;
        interactionJointIndices.reserve(pack.actions.size());
        for (const TaskActionBinding& action : pack.actions) {
            const auto found = std::find(
                interactions.jointNames.begin(),
                interactions.jointNames.end(),
                action.joint
            );
            if (found == interactions.jointNames.end()) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    action.joint,
                    "InteractionPack does not provide every task action joint"
                );
            }
            interactionJointIndices.push_back(
                static_cast<std::uint32_t>(
                    found - interactions.jointNames.begin()
                )
            );
        }
        staged->interactionJointTargets.reserve(
            static_cast<std::size_t>(interactionClip->frameCount) *
            pack.actions.size()
        );
        for (std::uint32_t frame = 0u;
             frame < interactionClip->frameCount;
             ++frame) {
            const std::size_t sourceBase =
                static_cast<std::size_t>(frame) *
                interactions.jointNames.size();
            for (const std::uint32_t joint : interactionJointIndices) {
                staged->interactionJointTargets.push_back(
                    interactionClip->jointTargets[sourceBase + joint]
                );
            }
        }
        for (std::uint32_t frame = 0u;
             frame < interactionClip->frameCount;
             ++frame) {
            for (std::uint32_t action = 0u;
                 action < pack.actions.size();
                 ++action) {
                const MRTaskActionBindingGPU& binding =
                    staged->actionBindings[action];
                const MRDofPropertiesGPU& properties =
                    model.dofs[binding.indices.w];
                const std::size_t targetIndex =
                    static_cast<std::size_t>(frame) *
                        pack.actions.size() +
                    action;
                const float target =
                    staged->interactionJointTargets[targetIndex];
                constexpr float positionTolerance = 1.0e-4f;
                if ((properties.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                    (target < properties.limits.x - positionTolerance ||
                     target > properties.limits.y + positionTolerance)) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        pack.actions[action].joint,
                        "InteractionPack joint target exceeds the compiled mechanism limit"
                    );
                }
                if (frame == 0u ||
                    (properties.flags & MR_DOF_FLAG_VELOCITY_LIMIT) == 0u) {
                    continue;
                }
                const float previous = staged->interactionJointTargets[
                    targetIndex - pack.actions.size()
                ];
                const float requestedVelocity =
                    std::abs(target - previous) *
                    interactionClip->framesPerSecond;
                if (requestedVelocity >
                    properties.limits.z * (1.0f + 1.0e-5f)) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        pack.actions[action].joint,
                        "InteractionPack joint target exceeds the compiled mechanism velocity limit"
                    );
                }
            }
        }

        const auto usesInteractionContact = [&pack] {
            const auto observesContact = [](const auto& operations) {
                return std::ranges::any_of(
                    operations,
                    [](const TaskObservationOperatorSpec& operation) {
                        return operation.source ==
                                TaskObservationSource::interactionContactMode ||
                            operation.source ==
                                TaskObservationSource::interactionContactTarget ||
                            operation.source ==
                                TaskObservationSource::interactionContactValidity;
                    }
                );
            };
            return observesContact(pack.actorFrame) ||
                observesContact(pack.actorCurrent) ||
                observesContact(pack.critic) ||
                std::ranges::any_of(
                    pack.rewards,
                    [](const TaskRewardOperatorSpec& reward) {
                        return reward.operation ==
                            TaskRewardOperator::interactionContactTracking;
                    }
                );
        }();
        interactionContactIds.reserve(
            interactions.contactTracks.size()
        );
        if (usesInteractionContact &&
            world.sceneBodyIndices().size() != 1u) {
            return reject(
                TaskCompileStatus::invalidWorld,
                "interaction",
                "InteractionPack v1 requires one resolved support-surface scene body so contact outcomes cannot alias another counterpart"
            );
        }
        staged->interactionContacts.reserve(
            interactions.contactTracks.size()
        );
        for (std::uint32_t trackIndex = 0u;
             trackIndex < interactions.contactTracks.size();
             ++trackIndex) {
            const InteractionContactTrack& track =
                interactions.contactTracks[trackIndex];
            if (track.counterpart != pack.terrain.body) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    track.counterpart,
                    "InteractionPack v1 contact counterpart must match the TaskPack terrain body"
                );
            }
            const std::uint32_t contactGroup = namedGroup(
                contactGroupIds,
                track.taskContactGroup
            );
            if (contactGroup == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    track.taskContactGroup,
                    "InteractionPack contact track has no TaskPack contact group"
                );
            }
            const MRTaskContactGroupGPU& compiledGroup =
                staged->contactGroups[contactGroup];
            bool usesCompactField = false;
            for (std::uint32_t frame = 0u;
                 frame < interactionClip->frameCount;
                 ++frame) {
                const std::size_t sample =
                    static_cast<std::size_t>(frame) *
                        interactions.contactTracks.size() +
                    trackIndex;
                usesCompactField = usesCompactField ||
                    interactionClip->contactFeatureMasks[sample] != 0u;
            }
            if (usesCompactField &&
                ((compiledGroup.members.z &
                  MR_TASK_CONTACT_SUPPORT) == 0u ||
                 compiledGroup.supportPatch.x != 2u ||
                 compiledGroup.supportPatch.y != 2u ||
                 compiledGroup.supportPatch.z != 4u)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    track.taskContactGroup,
                    "InteractionPack compact contact fields require a 2x2 support patch; mode-only tracks may bind generic contact groups"
                );
            }
            staged->interactionContacts.push_back({{
                contactGroup,
                trackIndex,
                trackIndex * kInteractionContactFeatureCount,
                kInteractionContactFeatureCount,
            }});
            interactionContactIds.push_back(track.id);
        }
        const std::size_t contactSampleCount =
            static_cast<std::size_t>(interactionClip->frameCount) *
            interactions.contactTracks.size();
        staged->interactionSamples.reserve(contactSampleCount);
        for (std::size_t sample = 0u;
             sample < contactSampleCount;
             ++sample) {
            staged->interactionSamples.push_back({
                {
                    interactionClip->contactModes[sample],
                    interactionClip->contactFeatureMasks[sample],
                    interactionClip->contactSampleFlags[sample],
                    0u,
                },
                {
                    interactionClip->contactConfidence[sample],
                    0.0f,
                    0.0f,
                    0.0f,
                },
            });
        }
        staged->interactionContactTargets =
            interactionClip->contactTargets;
        staged->interactionContactTolerances =
            interactionClip->contactTolerances;
    }

    const auto compileObservations =
        [&](const std::vector<TaskObservationOperatorSpec>& specs,
            std::vector<MRTaskObservationOperatorGPU>& compiled)
        -> TaskCompileDiagnostics {
        compiled.reserve(specs.size());
        for (std::size_t operatorIndex = 0u;
             operatorIndex < specs.size();
             ++operatorIndex) {
            const TaskObservationOperatorSpec& spec =
                specs[operatorIndex];
            if (!finite(spec.scale) ||
                !finite(spec.offset) ||
                !finite(spec.noiseAmplitude) ||
                spec.noiseAmplitude < 0.0f ||
                !finite(spec.biasLower) ||
                !finite(spec.biasUpper) ||
                spec.biasLower > spec.biasUpper) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.target,
                    "observation transform or corruption is invalid"
                );
            }
            if (spec.normalizeVector3 &&
                spec.source !=
                    TaskObservationSource::projectedGravity) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.target,
                    "vector normalization is supported only for projected gravity"
                );
            }
            const std::uint32_t opcode =
                static_cast<std::uint32_t>(spec.source);
            std::uint32_t sourceIndex = MR_INVALID_INDEX;
            std::uint32_t componentLimit = 1u;
            switch (spec.source) {
            case TaskObservationSource::rootAngularVelocityLocal:
            case TaskObservationSource::projectedGravity:
            case TaskObservationSource::command:
            case TaskObservationSource::rootLinearVelocityLocal:
                componentLimit = 3u;
                break;
            case TaskObservationSource::jointPositionError:
            case TaskObservationSource::jointVelocity:
            case TaskObservationSource::previousAction: {
                bool ambiguous = false;
                const std::uint32_t joint = uniqueIndex(
                    model.jointNames,
                    spec.target,
                    ambiguous
                );
                sourceIndex =
                    joint == MR_INVALID_INDEX
                    ? MR_INVALID_INDEX
                    : actionIndexForJoint(actionJoints, joint);
                if (ambiguous) {
                    return reject(
                        TaskCompileStatus::ambiguousSemantic,
                        spec.target,
                        "observation joint identity is ambiguous"
                    );
                }
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation joint has no action binding"
                    );
                }
                break;
            }
            case TaskObservationSource::interactionJointPositionError: {
                if (interactionClip == nullptr) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.target,
                        "interaction joint observation requires a selected InteractionPack clip"
                    );
                }
                bool ambiguous = false;
                const std::uint32_t joint = uniqueIndex(
                    model.jointNames,
                    spec.target,
                    ambiguous
                );
                sourceIndex =
                    joint == MR_INVALID_INDEX
                    ? MR_INVALID_INDEX
                    : actionIndexForJoint(actionJoints, joint);
                if (ambiguous || sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        ambiguous
                            ? TaskCompileStatus::ambiguousSemantic
                            : TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "interaction joint observation has no task action binding"
                    );
                }
                break;
            }
            case TaskObservationSource::interactionContactMode:
            case TaskObservationSource::interactionContactTarget:
            case TaskObservationSource::interactionContactValidity:
                if (interactionClip == nullptr) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.target,
                        "interaction contact observation requires a selected InteractionPack clip"
                    );
                }
                sourceIndex = namedGroup(
                    interactionContactIds,
                    spec.target
                );
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "interaction contact track does not exist"
                    );
                }
                componentLimit =
                    spec.source ==
                        TaskObservationSource::interactionContactMode
                    ? 2u
                    : kInteractionContactFeatureCount;
                break;
            case TaskObservationSource::interactionPhase:
                if (interactionClip == nullptr) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        "interaction",
                        "interaction phase observation requires a selected InteractionPack clip"
                    );
                }
                componentLimit = 3u;
                break;
            case TaskObservationSource::interactionRootTrackingError:
                if (interactionClip == nullptr) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        "interaction",
                        "interaction root tracking observation requires a selected InteractionPack clip"
                    );
                }
                componentLimit = 12u;
                break;
            case TaskObservationSource::rootHeight:
                break;
            case TaskObservationSource::supportSense:
                componentLimit = 3u;
                if (std::none_of(
                        staged->contactGroups.begin(),
                        staged->contactGroups.end(),
                        [](const MRTaskContactGroupGPU& group) {
                            return (group.members.z &
                                    MR_TASK_CONTACT_SUPPORT) != 0u;
                        }
                    )) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        "support_sense",
                        "support-sense observation requires an authored support contact group"
                    );
                }
                break;
            case TaskObservationSource::supportPatch: {
                sourceIndex = namedGroup(
                    contactGroupIds,
                    spec.target
                );
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "support-patch observation contact group does not exist"
                    );
                }
                const MRTaskContactGroupGPU& group =
                    staged->contactGroups[sourceIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u ||
                    group.supportPatch.z == 0u) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.target,
                        "support-patch observation requires an authored support patch"
                    );
                }
                componentLimit = 9u + group.supportPatch.z;
                break;
            }
            case TaskObservationSource::gaitPhase:
                componentLimit = 2u;
                break;
            case TaskObservationSource::recoveryEvent:
                componentLimit = 4u;
                break;
            case TaskObservationSource::objectTrack: {
                bool ambiguous = false;
                const std::uint32_t body = uniqueIndex(
                    model.bodyNames,
                    spec.target,
                    ambiguous
                );
                if (ambiguous || body == MR_INVALID_INDEX) {
                    return reject(
                        ambiguous
                            ? TaskCompileStatus::ambiguousSemantic
                            : TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "tracked object identity is unresolved"
                    );
                }
                const auto sceneBody = std::find(
                    world.sceneBodyIndices().begin(),
                    world.sceneBodyIndices().end(),
                    body
                );
                if (sceneBody == world.sceneBodyIndices().end()) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.target,
                        "tracked object is not a scene body"
                    );
                }
                sourceIndex = static_cast<std::uint32_t>(
                    sceneBody - world.sceneBodyIndices().begin()
                );
                componentLimit = 7u;
                break;
            }
            case TaskObservationSource::maskedDepth:
                if (!hasVisualProgram) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        "visual",
                        "masked-depth observation requires a TaskPack visual program"
                    );
                }
                sourceIndex = 0u;
                componentLimit = static_cast<std::uint32_t>(
                    visualComponentCount
                );
                break;
            case TaskObservationSource::contactMetric: {
                sourceIndex = namedGroup(
                    contactGroupIds,
                    spec.target
                );
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation contact group does not exist"
                    );
                }
                const std::uint32_t flags =
                    staged->contactGroups[sourceIndex]
                        .members.z;
                componentLimit =
                    (flags & MR_TASK_CONTACT_SUPPORT) != 0u
                    ? 6u
                    : (flags & MR_TASK_CONTACT_FORBIDDEN) != 0u
                    ? 1u
                    : 0u;
                break;
            }
            case TaskObservationSource::contactWrenchLocal:
                sourceIndex = namedGroup(
                    contactGroupIds,
                    spec.target
                );
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation contact group does not exist"
                    );
                }
                componentLimit = 6u;
                break;
            case TaskObservationSource::terrainHeight:
                sourceIndex = spec.component;
                componentLimit = 1u;
                if (sourceIndex >=
                    pack.terrain.sampleOffsets.size()) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        "terrain",
                        "terrain observation sample index is out of range"
                    );
                }
                break;
            case TaskObservationSource::bodyParameterMean:
            case TaskObservationSource::controllerParameter:
                componentLimit = 4u;
                break;
            case TaskObservationSource::bodyParameter: {
                bool ambiguous = false;
                sourceIndex = uniqueIndex(
                    model.bodyNames,
                    spec.target,
                    ambiguous
                );
                if (ambiguous) {
                    return reject(
                        TaskCompileStatus::ambiguousSemantic,
                        spec.target,
                        "observation body identity is ambiguous"
                    );
                }
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation body does not exist"
                    );
                }
                componentLimit = 4u;
                break;
            }
            default:
                return reject(
                    TaskCompileStatus::unsupportedOperator,
                    spec.target,
                    "observation opcode is unsupported"
                );
            }
            if (spec.source !=
                    TaskObservationSource::terrainHeight &&
                spec.component >= componentLimit) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.target,
                    "observation component is out of range"
                );
            }
            std::uint32_t biasIndex = MR_INVALID_INDEX;
            if (spec.biasLower != 0.0f ||
                spec.biasUpper != 0.0f) {
                biasIndex = static_cast<std::uint32_t>(
                    staged->biasSpecs.size()
                );
                staged->biasSpecs.push_back({
                    {
                        spec.biasLower,
                        spec.biasUpper,
                        0.0f,
                        0.0f,
                    },
                    {4096u + biasIndex, 0u, 0u, 0u},
                });
            }
            compiled.push_back({
                {
                    opcode,
                    sourceIndex,
                    spec.component,
                    spec.normalizeVector3
                        ? MR_TASK_OBSERVATION_NORMALIZE_VECTOR3
                        : 0u,
                },
                {
                    spec.scale,
                    spec.offset,
                    spec.noiseAmplitude,
                    0.0f,
                },
                {
                    biasIndex,
                    1024u +
                        static_cast<std::uint32_t>(
                            operatorIndex
                        ),
                    0u,
                    0u,
                },
            });
        }
        return {};
    };

    if (std::ranges::any_of(
            pack.actorCurrent,
            [](const TaskObservationOperatorSpec& observation) {
                return observation.source ==
                        TaskObservationSource::maskedDepth ||
                    observation.source ==
                        TaskObservationSource::objectTrack ||
                    observation.normalizeVector3;
            }
        )) {
        return reject(
            TaskCompileStatus::invalidPack,
            "actor_current",
            "actorCurrent cannot contain renderer-owned sources or history-vector normalization"
        );
    }
    const std::size_t visualSuffixCount =
        static_cast<std::size_t>(visualComponentCount);
    if (visualSuffixCount > pack.actorFrame.size()) {
        return reject(
            TaskCompileStatus::invalidPack,
            "visual",
            "visual observation suffix exceeds the actor frame"
        );
    }
    const std::size_t temporalActorCount =
        pack.actorFrame.size() - visualSuffixCount;
    std::vector<TaskObservationOperatorSpec> orderedActor;
    orderedActor.reserve(
        pack.actorFrame.size() + pack.actorCurrent.size()
    );
    orderedActor.insert(
        orderedActor.end(),
        pack.actorFrame.begin(),
        pack.actorFrame.begin() + temporalActorCount
    );
    orderedActor.insert(
        orderedActor.end(),
        pack.actorCurrent.begin(),
        pack.actorCurrent.end()
    );
    orderedActor.insert(
        orderedActor.end(),
        pack.actorFrame.begin() + temporalActorCount,
        pack.actorFrame.end()
    );
    TaskCompileDiagnostics observationStatus =
        compileObservations(
            orderedActor,
            staged->actorOperators
        );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }
    if (hasVisualProgram) {
        if (staged->actorOperators.size() <= visualComponentCount) {
            return reject(
                TaskCompileStatus::invalidPack,
                "visual",
                "sparse masked-depth history requires temporal actor observations and a complete visual plane"
            );
        }
        const std::size_t first =
            staged->actorOperators.size() -
            static_cast<std::size_t>(visualComponentCount);
        for (std::uint32_t component = 0u;
             component < visualComponentCount;
             ++component) {
            const MRTaskObservationOperatorGPU& operation =
                staged->actorOperators[first + component];
            if (operation.source.x != MR_TASK_OBSERVE_MASKED_DEPTH ||
                operation.source.z != component) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "visual",
                    "masked-depth components must be the complete contiguous suffix of the actor frame"
                );
            }
        }
    }
    if (std::ranges::any_of(
            pack.critic,
            [](const TaskObservationOperatorSpec& observation) {
                return observation.source ==
                    TaskObservationSource::maskedDepth;
            }
        )) {
        return reject(
            TaskCompileStatus::invalidPack,
            "visual",
            "masked depth is a deployable actor source; critics use explicit supervisory operators"
        );
    }
    observationStatus = compileObservations(
        pack.critic,
        staged->criticOperators
    );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }

    staged->rewardOperators.reserve(pack.rewards.size());
    bool hasRecoveryDefinition = false;
    mr_float4 recoveryDefinition{};
    for (const TaskRewardOperatorSpec& reward : pack.rewards) {
        if (!finite(reward.weight) ||
            !finite(reward.parameters)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.sourceGroup,
                "reward weight or parameters are non-finite"
            );
        }
        std::uint32_t sourceIndex = MR_INVALID_INDEX;
        std::uint32_t targetIndex = MR_INVALID_INDEX;
        mr_float4 compiledParameters = reward.parameters;
        switch (reward.operation) {
        case TaskRewardOperator::jointGroupPostureSquared:
        case TaskRewardOperator::jointGroupPostureAbsolute:
            if (reward.sourceGroup.empty()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "reward",
                    "joint-posture reward requires a joint group"
                );
            }
            sourceIndex = namedGroup(
                jointGroupIds,
                reward.sourceGroup
            );
            break;
        case TaskRewardOperator::interactionJointTracking:
            if (interactionClip == nullptr) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "interaction",
                    "interaction joint tracking requires a selected InteractionPack clip"
                );
            }
            if (!reward.sourceGroup.empty()) {
                sourceIndex = namedGroup(
                    jointGroupIds,
                    reward.sourceGroup
                );
            }
            break;
        case TaskRewardOperator::interactionRootTracking:
        case TaskRewardOperator::interactionRootLinearVelocityError:
            if (interactionClip == nullptr) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "interaction",
                    "interaction root tracking requires a selected InteractionPack clip"
                );
            }
            break;
        case TaskRewardOperator::interactionContactTracking:
            if (interactionClip == nullptr ||
                reward.sourceGroup.empty() || reward.target.empty()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    reward.target,
                    "interaction contact tracking requires a selected clip, contact group, and contact track"
                );
            }
            sourceIndex = namedGroup(
                contactGroupIds,
                reward.sourceGroup
            );
            targetIndex = namedGroup(
                interactionContactIds,
                reward.target
            );
            if (sourceIndex == MR_INVALID_INDEX ||
                targetIndex == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    sourceIndex == MR_INVALID_INDEX
                        ? reward.sourceGroup
                        : reward.target,
                    "interaction contact reward semantic does not exist"
                );
            }
            if (staged->interactionContacts[targetIndex]
                    .binding.x != sourceIndex) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    reward.target,
                    "interaction contact track is bound to a different TaskPack contact group"
                );
            }
            break;
        case TaskRewardOperator::gaitContactMatch:
        case TaskRewardOperator::swingClearance:
        case TaskRewardOperator::footClearance:
        case TaskRewardOperator::supportSlip:
        case TaskRewardOperator::forbiddenContact:
        case TaskRewardOperator::supportContactCount:
        case TaskRewardOperator::bodyHeightExponential:
        case TaskRewardOperator::recoveryTiltProgress:
        case TaskRewardOperator::recoveryCompletion:
            if (!reward.sourceGroup.empty()) {
                sourceIndex = namedGroup(
                    contactGroupIds,
                    reward.sourceGroup
                );
            }
            break;
        case TaskRewardOperator::wholeBodyRecovery:
            if (reward.sourceGroup.empty() || reward.target.empty()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "whole_body_recovery",
                    "whole-body recovery requires assist and trunk contact groups"
                );
            }
            sourceIndex = namedGroup(
                contactGroupIds,
                reward.sourceGroup
            );
            targetIndex = namedGroup(
                contactGroupIds,
                reward.target
            );
            if (sourceIndex == MR_INVALID_INDEX ||
                targetIndex == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    sourceIndex == MR_INVALID_INDEX
                        ? reward.sourceGroup
                        : reward.target,
                    "whole-body recovery contact semantic does not exist"
                );
            }
            break;
        case TaskRewardOperator::linkClearanceBarrier: {
            sourceIndex = namedGroup(
                contactGroupIds,
                reward.sourceGroup
            );
            bool ambiguous = false;
            const std::uint32_t body = uniqueIndex(
                model.bodyNames,
                reward.target,
                ambiguous
            );
            const auto sceneBody = body == MR_INVALID_INDEX
                ? world.sceneBodyIndices().end()
                : std::find(
                      world.sceneBodyIndices().begin(),
                      world.sceneBodyIndices().end(),
                      body
                  );
            if (sourceIndex == MR_INVALID_INDEX ||
                ambiguous ||
                sceneBody == world.sceneBodyIndices().end()) {
                return reject(
                    ambiguous
                        ? TaskCompileStatus::ambiguousSemantic
                        : TaskCompileStatus::unresolvedSemantic,
                    sourceIndex == MR_INVALID_INDEX
                        ? reward.sourceGroup
                        : reward.target,
                    "link-clearance barrier requires a protected body group and a dynamic scene projectile"
                );
            }
            float projectileRadius = 0.0f;
            for (const MRShapeGPU& shape : model.shapes) {
                if (shape.bodyIndex == body &&
                    (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u) {
                    projectileRadius = std::max(
                        projectileRadius,
                        shape.contactRestAndBoundingRadius.z
                    );
                }
            }
            if (!(projectileRadius > 0.0f) ||
                !finite(projectileRadius)) {
                return reject(
                    TaskCompileStatus::invalidWorld,
                    reward.target,
                    "projectile has no finite collision envelope"
                );
            }
            targetIndex = static_cast<std::uint32_t>(
                sceneBody - world.sceneBodyIndices().begin()
            );
            // The GPU consumes the exact authored projectile envelope plus
            // the task margin. Per-link envelopes live in a parallel table.
            compiledParameters.y += projectileRadius;
            break;
        }
        case TaskRewardOperator::projectileMiss:
        case TaskRewardOperator::projectileSafeStillness:
        case TaskRewardOperator::projectileSafeActionRate:
        case TaskRewardOperator::jointCbfCorrection:
        case TaskRewardOperator::jointCbfBuffer:
        case TaskRewardOperator::projectilePredictedClearance:
            break;
        case TaskRewardOperator::projectileEvasion: {
            bool ambiguous = false;
            const std::uint32_t body = uniqueIndex(
                model.bodyNames,
                reward.target,
                ambiguous
            );
            const auto sceneBody = body == MR_INVALID_INDEX
                ? world.sceneBodyIndices().end()
                : std::find(
                      world.sceneBodyIndices().begin(),
                      world.sceneBodyIndices().end(),
                      body
                  );
            if (ambiguous ||
                sceneBody == world.sceneBodyIndices().end()) {
                return reject(
                    ambiguous
                        ? TaskCompileStatus::ambiguousSemantic
                        : TaskCompileStatus::unresolvedSemantic,
                    reward.target,
                    "projectile-evasion reward requires a dynamic scene projectile"
                );
            }
            targetIndex = static_cast<std::uint32_t>(
                sceneBody - world.sceneBodyIndices().begin()
            );
            break;
        }
        case TaskRewardOperator::objectGrasp:
        case TaskRewardOperator::objectLift:
        case TaskRewardOperator::objectPosition:
        case TaskRewardOperator::objectPlacement: {
            if (reward.target.empty() ||
                (reward.operation == TaskRewardOperator::objectGrasp &&
                 reward.sourceGroup.empty())) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    reward.target,
                    "object manipulation reward requires a dynamic scene target and grasp also requires a contact group"
                );
            }
            if (reward.operation == TaskRewardOperator::objectGrasp) {
                sourceIndex = namedGroup(
                    contactGroupIds,
                    reward.sourceGroup
                );
            }
            bool ambiguous = false;
            const std::uint32_t body = uniqueIndex(
                model.bodyNames,
                reward.target,
                ambiguous
            );
            const auto sceneBody = body == MR_INVALID_INDEX
                ? world.sceneBodyIndices().end()
                : std::find(
                      world.sceneBodyIndices().begin(),
                      world.sceneBodyIndices().end(),
                      body
                  );
            if (ambiguous ||
                sceneBody == world.sceneBodyIndices().end() ||
                body == MR_INVALID_INDEX ||
                model.bodies[body].motionType != MR_MOTION_DYNAMIC) {
                return reject(
                    ambiguous
                        ? TaskCompileStatus::ambiguousSemantic
                        : TaskCompileStatus::unresolvedSemantic,
                    reward.target,
                    "object manipulation target must resolve to one dynamic scene body"
                );
            }
            targetIndex = static_cast<std::uint32_t>(
                sceneBody - world.sceneBodyIndices().begin()
            );
            if (reward.operation == TaskRewardOperator::objectPlacement) {
                bool goalAmbiguous = false;
                const std::uint32_t goalBody = uniqueIndex(
                    model.bodyNames,
                    reward.sourceGroup,
                    goalAmbiguous
                );
                const auto goalScene = goalBody == MR_INVALID_INDEX
                    ? world.sceneBodyIndices().end()
                    : std::find(
                          world.sceneBodyIndices().begin(),
                          world.sceneBodyIndices().end(),
                          goalBody
                      );
                if (reward.sourceGroup.empty() || goalAmbiguous ||
                    goalScene == world.sceneBodyIndices().end()) {
                    return reject(
                        goalAmbiguous
                            ? TaskCompileStatus::ambiguousSemantic
                            : TaskCompileStatus::unresolvedSemantic,
                        reward.sourceGroup,
                        "object placement requires one named scene-body goal"
                    );
                }
                sourceIndex = static_cast<std::uint32_t>(
                    goalScene - world.sceneBodyIndices().begin()
                );
            }
            break;
        }
        case TaskRewardOperator::linearVelocityTracking:
        case TaskRewardOperator::yawVelocityTracking:
        case TaskRewardOperator::constant:
        case TaskRewardOperator::rootVerticalVelocitySquared:
        case TaskRewardOperator::rootRollPitchVelocitySquared:
        case TaskRewardOperator::tiltSquared:
        case TaskRewardOperator::
            projectedGravityHorizontalSquared:
        case TaskRewardOperator::rootHeightErrorSquared:
        case TaskRewardOperator::jointVelocitySquared:
        case TaskRewardOperator::jointAccelerationSquared:
        case TaskRewardOperator::actionRateSquared:
        case TaskRewardOperator::jointLimitViolationSquared:
        case TaskRewardOperator::jointLimitViolationAbsolute:
        case TaskRewardOperator::mechanicalPower:
        case TaskRewardOperator::rootHeightNormalized:
        case TaskRewardOperator::rootHeightProgress:
        case TaskRewardOperator::uprightness:
        case TaskRewardOperator::supportHeightExponential:
        case TaskRewardOperator::bodyUpExponential:
        case TaskRewardOperator::standingCompletion:
        case TaskRewardOperator::restoration:
            break;
        default:
            return reject(
                TaskCompileStatus::unsupportedOperator,
                reward.sourceGroup,
                "reward opcode is unsupported"
            );
        }
        if ((reward.operation ==
                 TaskRewardOperator::linearVelocityTracking ||
             reward.operation ==
                 TaskRewardOperator::yawVelocityTracking ||
             reward.operation ==
                 TaskRewardOperator::swingClearance ||
             reward.operation ==
                 TaskRewardOperator::footClearance) &&
            !(reward.parameters.x > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.sourceGroup,
                "tracking and clearance reward widths must be positive"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::interactionJointTracking &&
            !(reward.parameters.x > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.sourceGroup,
                "interaction joint tracking width must be positive"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::interactionContactTracking &&
            (reward.parameters.x < 0.0f ||
             reward.parameters.x > 1.0f ||
             !(reward.parameters.y > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "interaction contact tracking requires a mode blend in [0, 1] and positive field temperature"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::jointLimitViolationAbsolute &&
            (!(reward.parameters.x > 0.0f) ||
             reward.parameters.x > 1.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.sourceGroup,
                "soft joint-limit factor must be in (0, 1]"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::linkClearanceBarrier &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f) ||
             !(reward.parameters.z > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "link-clearance barrier requires positive alpha, safety radius, and constraint clip"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::projectileEvasion &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f) ||
             reward.parameters.z < 0.0f ||
             reward.parameters.z > 1.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "projectile evasion requires positive distance and velocity scales plus a position blend in [0, 1]"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::projectileSafeStillness &&
            !(reward.parameters.x > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "projectile_safe_stillness",
                "projectile-safe stillness requires a positive velocity scale"
            );
        }
        if ((reward.operation ==
                 TaskRewardOperator::jointCbfCorrection ||
             reward.operation ==
                 TaskRewardOperator::jointCbfBuffer ||
             reward.operation ==
                 TaskRewardOperator::projectilePredictedClearance) &&
            pack.threat.protectedGroup.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                "threat_reward",
                "threat-derived rewards require a compiled threat program"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::projectilePredictedClearance &&
            !(reward.parameters.x > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "projectile_predicted_clearance",
                "predicted projectile clearance requires a positive distance scale"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::bodyHeightExponential &&
            (reward.sourceGroup.empty() ||
             !(reward.parameters.x > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.sourceGroup,
                "body-height reward requires a group and positive target"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::supportHeightExponential &&
            !(reward.parameters.x > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "support_height",
                "support-height reward decay must be positive"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::standingCompletion &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f) ||
             reward.parameters.y > 1.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "standing_completion",
                "standing completion requires positive height and cosine in (0, 1]"
            );
        }
        if (reward.operation == TaskRewardOperator::restoration &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f) ||
             !(reward.parameters.z > 0.0f) ||
             reward.parameters.z > 1.0f ||
             !(reward.parameters.w > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                "restoration",
                "restoration requires positive joint, position, and speed tolerances plus an orientation cosine in (0, 1]"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::wholeBodyRecovery &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f) ||
             reward.parameters.y > 1.0f ||
             !(reward.parameters.z > 0.0f) ||
             !(reward.parameters.w > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                "whole_body_recovery",
                "whole-body recovery requires positive height, support radius, and speed scale plus upright cosine in (0, 1]"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::interactionRootTracking &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                "interaction_root_tracking",
                "interaction root tracking requires positive position and orientation widths"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::interactionRootLinearVelocityError &&
            !(reward.parameters.x > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "interaction_root_linear_velocity_error",
                "interaction root linear-velocity error requires a positive velocity scale"
            );
        }
        if (reward.operation == TaskRewardOperator::objectGrasp &&
            (reward.parameters.x < 1.0f ||
             !(reward.parameters.y > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "object grasp requires at least one distinct contact member and a positive force scale"
            );
        }
        if (reward.operation == TaskRewardOperator::objectLift &&
            !(reward.parameters.y > reward.parameters.x)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "object lift requires an upper height greater than its lower height"
            );
        }
        if (reward.operation == TaskRewardOperator::objectPosition &&
            !(reward.parameters.w > 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "object position requires a positive squared-error width"
            );
        }
        if (reward.operation == TaskRewardOperator::objectPlacement &&
            (!(reward.parameters.x > 0.0f) ||
             !(reward.parameters.y > 0.0f) ||
             !(reward.parameters.z > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.target,
                "object placement requires positive position, linear-speed, and angular-speed squared-error widths"
            );
        }
        if ((reward.operation ==
                 TaskRewardOperator::recoveryTiltProgress ||
             reward.operation ==
                 TaskRewardOperator::recoveryCompletion) &&
            (reward.sourceGroup.empty() ||
             !(reward.parameters.x > reward.parameters.y) ||
             reward.parameters.y < 0.0f ||
             !(reward.parameters.z > 0.0f))) {
            return reject(
                TaskCompileStatus::invalidPack,
                "recovery_event",
                "recovery event requires a contact group, activation tilt above a non-negative stable tilt, and positive stable duration"
            );
        }
        if (reward.operation ==
                TaskRewardOperator::recoveryTiltProgress ||
            reward.operation ==
                TaskRewardOperator::recoveryCompletion) {
            if (hasRecoveryDefinition &&
                (reward.parameters.x != recoveryDefinition.x ||
                 reward.parameters.y != recoveryDefinition.y ||
                 reward.parameters.z != recoveryDefinition.z)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "recovery_event",
                    "all recovery reward operators must share one event definition"
                );
            }
            hasRecoveryDefinition = true;
            recoveryDefinition = reward.parameters;
        }
        if (!reward.sourceGroup.empty() &&
            sourceIndex == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                reward.sourceGroup,
                "reward source group does not exist"
            );
        }
        staged->rewardOperators.push_back({
            {
                static_cast<std::uint32_t>(reward.operation),
                sourceIndex,
                targetIndex,
                0u,
            },
            {
                reward.weight,
                compiledParameters.x,
                compiledParameters.y,
                compiledParameters.z,
            },
            {
                compiledParameters.w,
                0.0f,
                0.0f,
                0.0f,
            },
        });
    }

    staged->terminationOperators.reserve(
        pack.terminations.size()
    );
    for (const TaskTerminationOperatorSpec& termination :
         pack.terminations) {
        if (!finite(termination.threshold) ||
            !finite(termination.failurePenalty) ||
            termination.failurePenalty > 0.0f ||
            termination.minimumDifficultyBand >=
                pack.difficultyBandCount ||
            (termination.maximumDifficultyBand != MR_INVALID_INDEX &&
             (termination.maximumDifficultyBand >=
                  pack.difficultyBandCount ||
              termination.maximumDifficultyBand <
                  termination.minimumDifficultyBand)) ||
            termination.reason ==
                MR_TASK_TERMINATION_CONTINUING ||
            termination.reason >
                MR_TASK_TERMINATION_PROJECTILE_CONTACT) {
            return reject(
                TaskCompileStatus::invalidPack,
                termination.sourceGroup,
                "termination threshold, failure penalty, or reason is invalid"
            );
        }
        std::uint32_t sourceIndex = MR_INVALID_INDEX;
        switch (termination.operation) {
        case TaskTerminationOperator::contactGroup:
        case TaskTerminationOperator::projectileContact:
            sourceIndex = namedGroup(
                contactGroupIds,
                termination.sourceGroup
            );
            if (sourceIndex == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    termination.sourceGroup,
                    "termination contact group does not exist"
                );
            }
            if (termination.operation ==
                    TaskTerminationOperator::projectileContact &&
                !(termination.threshold > 0.0f)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    termination.sourceGroup,
                    "projectile-contact termination requires a positive force threshold"
                );
            }
            break;
        case TaskTerminationOperator::minimumRootHeight:
        case TaskTerminationOperator::maximumTilt:
            break;
        default:
            return reject(
                TaskCompileStatus::unsupportedOperator,
                termination.sourceGroup,
                "termination opcode is unsupported"
            );
        }
        staged->terminationOperators.push_back({
            {
                static_cast<std::uint32_t>(
                    termination.operation
                ),
                sourceIndex,
                termination.reason,
                termination.priority,
            },
            {
                termination.threshold,
                termination.failurePenalty,
                0.0f,
                0.0f,
            },
            {
                termination.minimumDifficultyBand,
                termination.maximumDifficultyBand,
                0u,
                0u,
            },
        });
    }

    staged->randomizationOperators.reserve(
        pack.randomization.size()
    );
    for (std::size_t operatorIndex = 0u;
         operatorIndex < pack.randomization.size();
         ++operatorIndex) {
        const TaskRandomizationOperatorSpec& random =
            pack.randomization[operatorIndex];
        if (!finite(random.parameters) ||
            random.minimumDifficultyBand >=
                pack.difficultyBandCount) {
            return reject(
                TaskCompileStatus::invalidPack,
                random.target,
                "randomization parameters or difficulty band are invalid"
            );
        }
        std::uint32_t targetIndex = MR_INVALID_INDEX;
        std::uint32_t targetBodyIndex = MR_INVALID_INDEX;
        mr_float4 compiledParameters = random.parameters;
        bool impactEvent = false;
        switch (random.operation) {
        case TaskRandomizationOperator::bodyParameter:
            targetIndex = namedGroup(
                contactGroupIds,
                random.target
            );
            if (targetIndex == MR_INVALID_INDEX ||
                random.component >= 4u) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    random.target,
                    "body randomization requires a group and parameter component"
                );
            }
            if (!orderedRange(random.parameters) ||
                (random.component == 0u &&
                 !(random.parameters.x > 0.0f)) ||
                (random.component != 0u &&
                 random.parameters.x < 0.0f)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "body-parameter randomization range is physically invalid"
                );
            }
            break;
        case TaskRandomizationOperator::bodyPayload: {
            bool ambiguous = false;
            targetIndex = uniqueIndex(
                model.bodyNames,
                random.target,
                ambiguous
            );
            if (ambiguous) {
                return reject(
                    TaskCompileStatus::ambiguousSemantic,
                    random.target,
                    "payload body identity is ambiguous"
                );
            }
            if (targetIndex == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    random.target,
                    "payload body does not exist"
                );
            }
            if (!inRange(
                    targetIndex,
                    articulation.firstBody,
                    articulation.bodyCount
                ) ||
                !orderedRange(random.parameters)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "payload target or range is outside the selected articulation"
                );
            }
            const float mass =
                model.bodies[targetIndex]
                    .massAndInverseMass.x;
            if (!finite(mass) || !(mass > 0.0f)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "payload body has no positive authored mass"
                );
            }
            compiledParameters.z = 1.0f / mass;
            break;
        }
        case TaskRandomizationOperator::controllerParameter:
            if (random.component >= 4u ||
                random.component == 2u ||
                !orderedRange(random.parameters) ||
                random.parameters.x < 0.0f) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "controller",
                    "controller randomization component or range is invalid"
                );
            }
            break;
        case TaskRandomizationOperator::rootPosition:
            if (random.parameters.x < 0.0f ||
                random.parameters.y < 0.0f ||
                random.parameters.z < 0.0f) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "root",
                    "root-position amplitudes must be nonnegative"
                );
            }
            break;
        case TaskRandomizationOperator::rootYaw:
        case TaskRandomizationOperator::actionPosition:
        case TaskRandomizationOperator::velocity:
        case TaskRandomizationOperator::actionVelocity:
        case TaskRandomizationOperator::rootHeight:
            if (!orderedRange(random.parameters)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "randomization lower bound exceeds its upper bound"
                );
            }
            break;
        case TaskRandomizationOperator::rootOrientation: {
            const float squaredNorm =
                random.parameters.x * random.parameters.x +
                random.parameters.y * random.parameters.y +
                random.parameters.z * random.parameters.z +
                random.parameters.w * random.parameters.w;
            if (!finite(squaredNorm) || !(squaredNorm > 1.0e-8f)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "root",
                    "root orientation must be a finite nonzero quaternion"
                );
            }
            const float inverseNorm = 1.0f / std::sqrt(squaredNorm);
            compiledParameters = {
                random.parameters.x * inverseNorm,
                random.parameters.y * inverseNorm,
                random.parameters.z * inverseNorm,
                random.parameters.w * inverseNorm,
            };
            break;
        }
        case TaskRandomizationOperator::jointPosition: {
            bool ambiguous = false;
            const std::uint32_t joint = uniqueIndex(
                model.jointNames,
                random.target,
                ambiguous
            );
            const std::uint32_t action =
                joint == MR_INVALID_INDEX
                ? MR_INVALID_INDEX
                : actionIndexForJoint(actionJoints, joint);
            if (ambiguous || action == MR_INVALID_INDEX) {
                return reject(
                    ambiguous
                        ? TaskCompileStatus::ambiguousSemantic
                        : TaskCompileStatus::unresolvedSemantic,
                    random.target,
                    "joint reset target has no unique action binding"
                );
            }
            const MRTaskActionBindingGPU& binding =
                staged->actionBindings[action];
            if (!orderedRange(random.parameters) ||
                random.parameters.x < binding.parameters.y ||
                random.parameters.y > binding.parameters.z) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "joint reset range exceeds the authored joint limits"
                );
            }
            targetIndex = binding.indices.z;
            break;
        }
        case TaskRandomizationOperator::sceneBodyPosition:
        case TaskRandomizationOperator::sceneBodyVelocity:
        case TaskRandomizationOperator::sceneBodyLaunchStep:
        case TaskRandomizationOperator::sceneBodyEventImpact: {
            bool ambiguous = false;
            const std::uint32_t body = uniqueIndex(
                model.bodyNames,
                random.target,
                ambiguous
            );
            if (ambiguous || body == MR_INVALID_INDEX) {
                return reject(
                    ambiguous
                        ? TaskCompileStatus::ambiguousSemantic
                        : TaskCompileStatus::unresolvedSemantic,
                    random.target,
                    "scene randomization target is unresolved"
                );
            }
            const auto sceneBody = std::find(
                world.sceneBodyIndices().begin(),
                world.sceneBodyIndices().end(),
                body
            );
            if (sceneBody == world.sceneBodyIndices().end()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "scene randomization target is not a scene body"
                );
            }
            targetIndex = static_cast<std::uint32_t>(
                sceneBody - world.sceneBodyIndices().begin()
            );
            targetBodyIndex = body;
            if (random.operation ==
                    TaskRandomizationOperator::sceneBodyLaunchStep) {
                constexpr std::uint32_t maximumLaunchStep =
                    MR_BODY_STATE_LAUNCH_STEP_MASK >>
                    MR_BODY_STATE_LAUNCH_STEP_SHIFT;
                if (!integerStepRange(
                        random.parameters,
                        maximumLaunchStep
                    )) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        random.target,
                        "scene launch-step range exceeds its packed capacity"
                    );
                }
            } else if (random.operation ==
                       TaskRandomizationOperator::sceneBodyEventImpact) {
                if (random.component >= 256u ||
                    random.parameters.x < 0.0f ||
                    !(random.parameters.y > 0.0f) ||
                    !(random.parameters.z > 0.0f) ||
                    !(random.parameters.w > 0.0f)) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        random.target,
                        "event impact requires a byte-sized order, nonnegative stable tilt, and positive stable, flight, and height gates"
                    );
                }
                impactEvent = true;
            } else if (random.component >= 3u ||
                       !orderedRange(random.parameters)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "scene position or velocity randomization is invalid"
                );
            }
            break;
        }
        case TaskRandomizationOperator::actionDelay:
            if (!integerStepRange(
                    random.parameters,
                    pack.maximumActionDelaySteps
                )) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "action_delay",
                    "action delay range exceeds its compiled capacity"
                );
            }
            break;
        case TaskRandomizationOperator::observationDelay:
            if (!integerStepRange(
                    random.parameters,
                    pack.maximumObservationDelaySteps
                )) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "observation_delay",
                    "observation delay range exceeds its compiled capacity"
                );
            }
            break;
        default:
            return reject(
                TaskCompileStatus::unsupportedOperator,
                random.target,
                "randomization opcode is unsupported"
            );
        }
        if (impactEvent) {
            const auto duplicate = std::find_if(
                staged->impactEvents.begin(),
                staged->impactEvents.end(),
                [&random](const MRTaskImpactEventGPU& event) {
                    return event.binding.y == random.component;
                }
            );
            if (duplicate != staged->impactEvents.end()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "event impact sequence order is duplicated"
                );
            }
            float projectileRadius = 0.0f;
            for (const MRShapeGPU& shape : model.shapes) {
                if (shape.bodyIndex == targetBodyIndex &&
                    (shape.flags &
                     MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u) {
                    projectileRadius = std::max(
                        projectileRadius,
                        shape.contactRestAndBoundingRadius.z
                    );
                }
            }
            if (!(projectileRadius > 0.0f) ||
                !finite(projectileRadius)) {
                return reject(
                    TaskCompileStatus::invalidWorld,
                    random.target,
                    "event projectile has no finite collision envelope"
                );
            }
            staged->impactEvents.push_back({
                {
                    targetIndex,
                    random.component,
                    random.minimumDifficultyBand,
                    targetBodyIndex,
                },
                compiledParameters,
                {projectileRadius, 0.0f, 0.0f, 0.0f},
            });
        } else {
            staged->randomizationOperators.push_back({
                {
                    static_cast<std::uint32_t>(random.operation),
                    targetIndex,
                    random.component,
                    random.minimumDifficultyBand,
                },
                compiledParameters,
            });
        }
    }

    std::sort(
        staged->impactEvents.begin(),
        staged->impactEvents.end(),
        [](const MRTaskImpactEventGPU& lhs,
           const MRTaskImpactEventGPU& rhs) {
            return lhs.binding.y < rhs.binding.y;
        }
    );
    for (std::size_t index = 0u;
         index < staged->impactEvents.size();
         ++index) {
        if (staged->impactEvents[index].binding.y != index) {
            return reject(
                TaskCompileStatus::invalidPack,
                "event_impacts",
                "event impact sequence orders must be contiguous from zero"
            );
        }
        if (index != 0u &&
            staged->impactEvents[index].binding.z !=
                staged->impactEvents[0].binding.z) {
            return reject(
                TaskCompileStatus::invalidPack,
                "event_impacts",
                "one event impact sequence must share one minimum difficulty band"
            );
        }
    }

    std::uint32_t threatGroup = MR_INVALID_INDEX;
    if (!pack.threat.protectedGroup.empty()) {
        threatGroup = namedGroup(
            contactGroupIds,
            pack.threat.protectedGroup
        );
        if (threatGroup == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                pack.threat.protectedGroup,
                "threat program requires a protected contact group"
            );
        }
        if (staged->impactEvents.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                "threat",
                "threat program requires an event projectile sequence"
            );
        }
        const TaskThreatProgram& threat = pack.threat;
        if (!(threat.activationSpeed > 0.0f) ||
            !(threat.horizonSeconds > 0.0f) ||
            !(threat.safetyMargin >= 0.0f) ||
            !(threat.cbfAlpha > 0.0f) ||
            !(threat.stepOverMaximumHeight > 0.0f) ||
            !(threat.sidestepMaximumHeight >
                threat.stepOverMaximumHeight) ||
            !(threat.leanMaximumHeight >
                threat.sidestepMaximumHeight) ||
            !(threat.urgencySeconds > 0.0f) ||
            !(threat.desiredVelocityHorizonSeconds > 0.0f) ||
            !(threat.projectionEpsilon > 0.0f) ||
            !finite(threat.activationSpeed) ||
            !finite(threat.horizonSeconds) ||
            !finite(threat.safetyMargin) ||
            !finite(threat.cbfAlpha) ||
            !finite(threat.stepOverMaximumHeight) ||
            !finite(threat.sidestepMaximumHeight) ||
            !finite(threat.leanMaximumHeight) ||
            !finite(threat.urgencySeconds) ||
            !finite(threat.desiredVelocityHorizonSeconds) ||
            !finite(threat.projectionEpsilon)) {
            return reject(
                TaskCompileStatus::invalidPack,
                "threat",
                "threat timing, classification, and Joint-CBF parameters are invalid"
            );
        }
    }

    std::uint32_t motionAnchor = MR_INVALID_INDEX;
    if (!pack.motion.anchorBody.empty() ||
        !pack.motion.trackedBodies.empty()) {
        if (pack.motion.anchorBody.empty() ||
            pack.motion.trackedBodies.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                "motion",
                "motion prior requires an anchor and tracked bodies"
            );
        }
        bool ambiguous = false;
        motionAnchor = uniqueIndex(
            model.bodyNames,
            pack.motion.anchorBody,
            ambiguous
        );
        if (ambiguous || motionAnchor == MR_INVALID_INDEX) {
            return reject(
                ambiguous
                    ? TaskCompileStatus::ambiguousSemantic
                    : TaskCompileStatus::unresolvedSemantic,
                pack.motion.anchorBody,
                "motion-prior anchor body is unresolved"
            );
        }
        staged->motionBodies.reserve(
            pack.motion.trackedBodies.size()
        );
        for (const std::string& name : pack.motion.trackedBodies) {
            ambiguous = false;
            const std::uint32_t body = uniqueIndex(
                model.bodyNames,
                name,
                ambiguous
            );
            if (ambiguous || body == MR_INVALID_INDEX ||
                std::find(
                    staged->motionBodies.begin(),
                    staged->motionBodies.end(),
                    body
                ) != staged->motionBodies.end()) {
                return reject(
                    ambiguous
                        ? TaskCompileStatus::ambiguousSemantic
                        : TaskCompileStatus::unresolvedSemantic,
                    name,
                    "motion-prior tracked body is unresolved or duplicated"
                );
            }
            staged->motionBodies.push_back(body);
        }
        if (staged->motionBodies.size() >
            std::numeric_limits<std::uint32_t>::max() / 9u) {
            return reject(
                TaskCompileStatus::arithmeticOverflow,
                "motion",
                "motion-prior feature count exceeds uint32"
            );
        }
    }

    std::uint32_t terrainSceneBody = MR_INVALID_INDEX;
    std::uint32_t terrainShape = MR_INVALID_INDEX;
    std::uint32_t terrainGeometry = MR_INVALID_INDEX;
    if (!pack.terrain.body.empty() ||
        !pack.terrain.sampleOffsets.empty() ||
        !pack.terrain.resetTranslations.empty()) {
        if (pack.terrain.body.empty() ||
            pack.terrain.resetTranslations.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                "terrain",
                "terrain task requires a body and reset translations"
            );
        }
        bool ambiguous = false;
        const std::uint32_t terrainBody = uniqueIndex(
            model.bodyNames,
            pack.terrain.body,
            ambiguous
        );
        if (ambiguous) {
            return reject(
                TaskCompileStatus::ambiguousSemantic,
                pack.terrain.body,
                "terrain body identity is ambiguous"
            );
        }
        if (terrainBody == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                pack.terrain.body,
                "terrain body does not exist"
            );
        }
        const auto sceneBody = std::find(
            world.sceneBodyIndices().begin(),
            world.sceneBodyIndices().end(),
            terrainBody
        );
        if (sceneBody == world.sceneBodyIndices().end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                pack.terrain.body,
                "terrain body must be a compiled scene body"
            );
        }
        terrainSceneBody = static_cast<std::uint32_t>(
            sceneBody - world.sceneBodyIndices().begin()
        );
        for (std::uint32_t shapeIndex = 0u;
             shapeIndex < model.shapes.size();
             ++shapeIndex) {
            const MRShapeGPU& shape = model.shapes[shapeIndex];
            if (shape.bodyIndex == terrainBody &&
                (shape.shapeType == MR_SHAPE_PLANE ||
                 shape.shapeType == MR_SHAPE_HEIGHTFIELD)) {
                if (terrainShape != MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::ambiguousSemantic,
                        pack.terrain.body,
                        "terrain body has multiple plane or heightfield colliders"
                    );
                }
                terrainShape = shapeIndex;
                terrainGeometry =
                    shape.shapeType == MR_SHAPE_HEIGHTFIELD
                    ? shape.geometryOffset
                    : MR_INVALID_INDEX;
            }
        }
        if (terrainShape == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::invalidPack,
                pack.terrain.body,
                "terrain body has no plane or heightfield collider"
            );
        }
        for (const mr_float4 value :
             pack.terrain.sampleOffsets) {
            if (!finite(value)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "terrain.samples",
                    "terrain sample offset is non-finite"
                );
            }
        }
        for (const mr_float4 value :
             pack.terrain.resetTranslations) {
            if (!finite(value)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    "terrain.resets",
                    "terrain reset translation is non-finite"
                );
            }
        }
        staged->terrainSampleOffsets =
            pack.terrain.sampleOffsets;
        staged->terrainResetTranslations =
            pack.terrain.resetTranslations;
    }

    std::uint32_t actionCount = 0u;
    std::uint32_t actorOperatorCount = 0u;
    std::uint32_t criticOperatorCount = 0u;
    std::uint32_t biasCount = 0u;
    if (!narrowCount(staged->actionBindings.size(), actionCount) ||
        !narrowCount(
            staged->actorOperators.size(),
            actorOperatorCount
        ) ||
        !narrowCount(
            staged->criticOperators.size(),
            criticOperatorCount
        ) ||
        !narrowCount(staged->biasSpecs.size(), biasCount)) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "layout",
            "task operator count exceeds the 32-bit GPU ABI"
        );
    }
    const std::uint32_t currentActorObservationCount =
        static_cast<std::uint32_t>(pack.actorCurrent.size());
    const std::uint32_t directActorObservationCount =
        currentActorObservationCount +
        static_cast<std::uint32_t>(visualComponentCount);
    if (directActorObservationCount > actorOperatorCount) {
        return reject(
            TaskCompileStatus::internalFailure,
            "layout",
            "direct actor observation count exceeds compiled operators"
        );
    }
    const std::uint32_t actorFrameSize =
        actorOperatorCount - directActorObservationCount;
    const std::uint64_t temporalActorObservationSize =
        static_cast<std::uint64_t>(actorFrameSize) *
        pack.actorHistoryLength;
    const std::uint64_t actorObservationSize =
        temporalActorObservationSize +
        directActorObservationCount;
    const std::uint64_t criticObservationSize =
        (pack.criticIncludesCleanHistory
             ? temporalActorObservationSize
             : 0u) +
        static_cast<std::uint64_t>(criticOperatorCount) *
            pack.criticHistoryLength;
    if (actorObservationSize >
            std::numeric_limits<std::uint32_t>::max() ||
        criticObservationSize >
            std::numeric_limits<std::uint32_t>::max() ||
        pack.maximumActionDelaySteps >
            std::numeric_limits<std::uint32_t>::max() - 3u) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "layout",
            "task observation or delay layout overflows"
        );
    }
    staged->layout = {
        .actionCount = actionCount,
        .actorFrameSize = actorFrameSize,
        .actorHistoryLength = pack.actorHistoryLength,
        .actorObservationSize =
            static_cast<std::uint32_t>(actorObservationSize),
        .criticFrameSize = criticOperatorCount,
        .criticHistoryLength = pack.criticHistoryLength,
        .criticObservationSize =
            static_cast<std::uint32_t>(criticObservationSize),
        .contactMetricCount = contactMetricCount,
        .biasCount = biasCount,
        // Raw actions retain one preceding sample beyond the delay window;
        // the final slot is the independent filtered actuator target.
        .delayStateCount = pack.maximumActionDelaySteps + 3u,
        .motionFeatureCount = static_cast<std::uint32_t>(
            9u * staged->motionBodies.size()
        ),
        .interactionFrameCount = interactionClip == nullptr
            ? 0u
            : interactionClip->frameCount,
        .interactionContactCount = static_cast<std::uint32_t>(
            staged->interactionContacts.size()
        ),
    };

    staged->header.counts0 = {
        actionCount,
        actorFrameSize,
        criticOperatorCount,
        static_cast<std::uint32_t>(
            staged->contactGroups.size()
        ),
    };
    staged->header.counts1 = {
        static_cast<std::uint32_t>(
            staged->contactMembers.size()
        ),
        static_cast<std::uint32_t>(
            staged->jointGroups.size()
        ),
        static_cast<std::uint32_t>(
            staged->jointMembers.size()
        ),
        static_cast<std::uint32_t>(
            staged->rewardOperators.size()
        ),
    };
    staged->header.counts2 = {
        static_cast<std::uint32_t>(
            staged->terminationOperators.size()
        ),
        static_cast<std::uint32_t>(
            staged->randomizationOperators.size()
        ),
        biasCount,
        static_cast<std::uint32_t>(
            staged->terrainSampleOffsets.size()
        ),
    };
    staged->header.layout = {
        actorFrameSize,
        pack.actorHistoryLength,
        contactMetricCount,
        staged->layout.delayStateCount,
    };
    staged->header.root = {
        world.articulationIndex(),
        articulation.rootBody,
        articulation.qOffset,
        articulation.vOffset,
    };
    staged->header.terrain = {
        terrainSceneBody,
        terrainShape,
        terrainGeometry,
        static_cast<std::uint32_t>(
            staged->terrainResetTranslations.size()
        ),
    };
    const bool heightfieldTerrain =
        terrainShape != MR_INVALID_INDEX &&
        model.shapes[terrainShape].shapeType ==
            MR_SHAPE_HEIGHTFIELD;
    staged->header.schedule = {
        pack.maximumEpisodeSteps,
        pack.maximumObservationDelaySteps,
        pack.difficultyBandCount,
        heightfieldTerrain
            ? MR_TASK_PROGRAM_TERRAIN
            : 0u,
    };
    if (pack.criticIncludesCleanHistory) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY;
    }
    if (pack.visual.includeDerivedFeatures) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_MASKED_DEPTH_FEATURES;
    }
    if (threatGroup != MR_INVALID_INDEX) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_THREAT_TEACHER;
    }
    if (interactionClip != nullptr) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_INTERACTION_RESET;
        if (pack.interactionControlReference) {
            staged->header.schedule.w |=
                MR_TASK_PROGRAM_INTERACTION_REFERENCE;
        }
    }
    staged->header.locomotion = {
        pack.baseHeightTarget,
        pack.gaitPeriodSeconds,
        pack.clearanceTarget,
        pack.successTrackingThreshold,
    };
    staged->header.commandLower = pack.commands.lower;
    staged->header.commandLower.w =
        pack.commands.standingProbability;
    staged->header.commandUpper = pack.commands.upper;
    staged->header.commandUpper.w =
        pack.commands.difficultySamplingExponent;
    staged->commandCurriculum = {
        pack.commands.limitLower,
        pack.commands.limitUpper,
        pack.commands.difficultyStep,
    };
    staged->header.scheduleSeconds = {
        pack.commands.minimumDurationSeconds,
        pack.commands.maximumDurationSeconds,
        pack.pushes.minimumIntervalSeconds,
        pack.pushes.maximumIntervalSeconds,
    };
    staged->header.dynamics = {
        pack.pushes.maximumVelocity,
        pack.supportForceThreshold,
        pack.pushes.projectileStandingProbability,
        pack.pushes.projectileTargetHorizontalRadius,
    };
    staged->header.projectile = {
        pack.pushes.projectileHorizontalSpeedLower,
        pack.pushes.projectileHorizontalSpeedUpper,
        pack.pushes.projectileTargetHeightLower,
        pack.pushes.projectileTargetHeightUpper,
    };
    staged->header.projectileGravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
        pack.pushes.projectileTargetHorizontalRadius,
    };
    staged->header.counts3.x = static_cast<std::uint32_t>(
        staged->impactEvents.size()
    );
    staged->header.counts3.y = static_cast<std::uint32_t>(
        staged->contactMemberRadii.size()
    );
    staged->header.counts3.z = static_cast<std::uint32_t>(
        staged->interactionContacts.size()
    );
    staged->header.counts3.w = currentActorObservationCount;
    staged->header.articulation = {
        articulation.firstBody,
        articulation.bodyCount,
        MR_TASK_PROGRAM_ABI_VERSION,
        pack.criticHistoryLength,
    };
    const mr_float4 rootCenterOfMass =
        model.bodies[articulation.rootBody].centerOfMass;
    staged->header.rootReference = {
        -rootCenterOfMass.x,
        -rootCenterOfMass.y,
        -rootCenterOfMass.z,
        0.0f,
    };
    staged->header.visualLayout = {
        pack.visual.width,
        pack.visual.height,
        static_cast<std::uint32_t>(pack.visual.frameOffsets.size()),
        pack.visual.frameOffsets.empty()
            ? 0u
            : pack.visual.frameOffsets.back(),
    };
    staged->header.visualHistory = {
        pack.visual.frameOffsets.size() > 0u
            ? pack.visual.frameOffsets[0u] : 0u,
        pack.visual.frameOffsets.size() > 1u
            ? pack.visual.frameOffsets[1u] : 0u,
        pack.visual.frameOffsets.size() > 2u
            ? pack.visual.frameOffsets[2u] : 0u,
        pack.visual.frameOffsets.size() > 3u
            ? pack.visual.frameOffsets[3u] : 0u,
    };
    staged->header.visualRange = {
        pack.visual.nearDepthMeters,
        pack.visual.farDepthMeters,
        pack.visual.edgeFlickerProbability,
        pack.visual.difficultyCorruptionGain,
    };
    staged->header.visualCorruption = {
        pack.visual.fullDropoutProbability,
        pack.visual.pixelDropoutProbability,
        pack.visual.depthJitterMeters,
        pack.visual.depthNoiseSigmaMeters,
    };
    staged->header.threat = {
        threatGroup,
        threatGroup == MR_INVALID_INDEX ? 0u : 1u,
        0u,
        0u,
    };
    staged->header.threatTiming = {
        pack.threat.activationSpeed,
        pack.threat.horizonSeconds,
        pack.threat.safetyMargin,
        pack.threat.cbfAlpha,
    };
    staged->header.threatClassification = {
        pack.threat.stepOverMaximumHeight,
        pack.threat.sidestepMaximumHeight,
        pack.threat.leanMaximumHeight,
        0.0f,
    };
    staged->header.threatTeacher = {
        pack.threat.urgencySeconds,
        pack.threat.desiredVelocityHorizonSeconds,
        pack.threat.projectionEpsilon,
        0.0f,
    };
    staged->header.interaction = {
        interactionClip == nullptr ? 0u : interactionClip->frameCount,
        interactionClip == nullptr ? 0u : actionCount,
        static_cast<std::uint32_t>(
            staged->interactionContacts.size()
        ),
        interactionClip != nullptr && interactionClip->loop
            ? MR_TASK_INTERACTION_LOOP
            : 0u,
    };
    staged->header.interactionTiming = {
        interactionClip == nullptr
            ? 0.0f
            : interactionClip->framesPerSecond,
        interactionClip == nullptr
            ? 0.0f
            : static_cast<float>(interactionClip->frameCount - 1u) /
                interactionClip->framesPerSecond,
        // Authored student authority scales the learned control residual
        // around the guide during collection. Zero remains observation-only
        // shadow mode; autonomous evaluation disables the guide explicitly.
        interactionClip == nullptr
            ? 0.0f
            : pack.interactionStudentAuthority,
        interactionClip == nullptr
            ? 0.0f
            : pack.interactionResetPhaseFraction,
    };

    const auto appendArena =
        [&staged]<typename T>(
            const std::span<const T> values
        ) -> std::uint32_t {
            constexpr std::size_t alignment = 16u;
            const std::size_t aligned =
                (staged->arena.size() + alignment - 1u) /
                alignment * alignment;
            if (aligned >
                    std::numeric_limits<std::uint32_t>::max() ||
                values.size_bytes() >
                    std::numeric_limits<std::uint32_t>::max() -
                        aligned) {
                return MR_INVALID_INDEX;
            }
            staged->arena.resize(
                aligned + values.size_bytes(),
                std::byte{0}
            );
            if (!values.empty()) {
                std::memcpy(
                    staged->arena.data() + aligned,
                    values.data(),
                    values.size_bytes()
                );
            }
            return static_cast<std::uint32_t>(aligned);
        };
    staged->header.offsets0 = {
        appendArena(
            std::span<const MRTaskActionBindingGPU>{
                staged->actionBindings
            }
        ),
        appendArena(
            std::span<const MRTaskObservationOperatorGPU>{
                staged->actorOperators
            }
        ),
        appendArena(
            std::span<const MRTaskObservationOperatorGPU>{
                staged->criticOperators
            }
        ),
        appendArena(
            std::span<const MRTaskContactGroupGPU>{
                staged->contactGroups
            }
        ),
    };
    staged->header.offsets1 = {
        appendArena(
            std::span<const std::uint32_t>{
                staged->contactMembers
            }
        ),
        appendArena(
            std::span<const MRTaskIndexGroupGPU>{
                staged->jointGroups
            }
        ),
        appendArena(
            std::span<const std::uint32_t>{
                staged->jointMembers
            }
        ),
        appendArena(
            std::span<const MRTaskRewardOperatorGPU>{
                staged->rewardOperators
            }
        ),
    };
    staged->header.offsets2 = {
        appendArena(
            std::span<const MRTaskTerminationOperatorGPU>{
                staged->terminationOperators
            }
        ),
        appendArena(
            std::span<const MRTaskRandomizationOperatorGPU>{
                staged->randomizationOperators
            }
        ),
        appendArena(
            std::span<const MRTaskBiasSpecGPU>{
                staged->biasSpecs
            }
        ),
        appendArena(
            std::span<const mr_float4>{
                staged->terrainSampleOffsets
            }
        ),
    };
    staged->header.offsets3 = {
        appendArena(
            std::span<const mr_float4>{
                staged->terrainResetTranslations
            }
        ),
        appendArena(
            std::span<const mr_float4>{
                staged->commandCurriculum
            }
        ),
        appendArena(
            std::span<const MRTaskImpactEventGPU>{
                staged->impactEvents
            }
        ),
        appendArena(
            std::span<const float>{staged->contactMemberRadii}
        ),
    };
    const std::array offsets{
        staged->header.offsets0,
        staged->header.offsets1,
        staged->header.offsets2,
        staged->header.offsets3,
    };
    for (const mr_uint4 value : offsets) {
        if (value.x == MR_INVALID_INDEX ||
            value.y == MR_INVALID_INDEX ||
            value.z == MR_INVALID_INDEX ||
            value.w == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::arithmeticOverflow,
                "arena",
                "compiled task arena exceeds the 32-bit byte-offset ABI"
            );
        }
    }
    const std::uint32_t motionOffset = appendArena(
        std::span<const std::uint32_t>{staged->motionBodies}
    );
    if (motionOffset == MR_INVALID_INDEX) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "motion",
            "compiled motion-prior table exceeds the arena ABI"
        );
    }
    staged->header.motion = {
        motionAnchor,
        static_cast<std::uint32_t>(staged->motionBodies.size()),
        staged->layout.motionFeatureCount,
        motionOffset,
    };
    staged->header.interactionOffsets0 = {
        appendArena(
            std::span<const float>{staged->interactionRootTargets}
        ),
        appendArena(
            std::span<const float>{staged->interactionJointTargets}
        ),
        appendArena(
            std::span<const MRTaskInteractionContactGPU>{
                staged->interactionContacts
            }
        ),
        appendArena(
            std::span<const MRTaskInteractionSampleGPU>{
                staged->interactionSamples
            }
        ),
    };
    staged->header.interactionOffsets1 = {
        appendArena(
            std::span<const float>{
                staged->interactionContactTargets
            }
        ),
        appendArena(
            std::span<const float>{
                staged->interactionContactTolerances
            }
        ),
        0u,
        0u,
    };
    if (staged->header.interactionOffsets0.x == MR_INVALID_INDEX ||
        staged->header.interactionOffsets0.y == MR_INVALID_INDEX ||
        staged->header.interactionOffsets0.z == MR_INVALID_INDEX ||
        staged->header.interactionOffsets0.w == MR_INVALID_INDEX ||
        staged->header.interactionOffsets1.x == MR_INVALID_INDEX ||
        staged->header.interactionOffsets1.y == MR_INVALID_INDEX) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "interaction",
            "compiled interaction reference exceeds the arena ABI"
        );
    }

    Hash hash;
    hash.string(pack.id);
    hash.scalar(world.fingerprint());
    hash.scalar(staged->header);
    hash.span<std::byte>(staged->arena);
    for (const TaskActionBinding& action : pack.actions) {
        hash.string(action.joint);
    }
    for (const TaskContactGroup& group : pack.contactGroups) {
        hash.string(group.id);
        hash.string(group.referenceBody);
        for (const std::string& body : group.bodies) {
            hash.string(body);
        }
    }
    for (const TaskJointGroup& group : pack.jointGroups) {
        hash.string(group.id);
        for (const std::string& joint : group.joints) {
            hash.string(joint);
        }
    }
    if (interactionClip != nullptr) {
        hash.string(interactions.id);
        hash.string(interactions.sourceRepository);
        hash.string(interactions.sourceRevision);
        hash.string(interactions.license);
        hash.string(interactions.coordinateFrame);
        hash.string(interactionClip->id);
        hash.string(interactionClip->desiredOutcome);
        for (const std::string& joint : interactions.jointNames) {
            hash.string(joint);
        }
        for (const InteractionContactTrack& track :
             interactions.contactTracks) {
            hash.string(track.id);
            hash.string(track.taskContactGroup);
            hash.string(track.counterpart);
        }
    }
    staged->fingerprint = hash.finish();
    staged->header.taskFingerprint = staged->fingerprint;
    staged->header.worldFingerprint = staged->worldFingerprint;

    output.storage_ = std::move(staged);
    return {
        .status = TaskCompileStatus::success,
        .fingerprint = output.fingerprint(),
        .message = "compiled task program",
    };
}

const char* taskCompileStatusName(
    const TaskCompileStatus status
) noexcept {
    switch (status) {
    case TaskCompileStatus::success:
        return "success";
    case TaskCompileStatus::invalidWorld:
        return "invalid_world";
    case TaskCompileStatus::invalidPack:
        return "invalid_pack";
    case TaskCompileStatus::unresolvedSemantic:
        return "unresolved_semantic";
    case TaskCompileStatus::ambiguousSemantic:
        return "ambiguous_semantic";
    case TaskCompileStatus::unsupportedOperator:
        return "unsupported_operator";
    case TaskCompileStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case TaskCompileStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
