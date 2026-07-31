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
    std::vector<MRTaskIndexGroupGPU> jointGroups;
    std::vector<std::uint32_t> jointMembers;
    std::vector<MRTaskRewardOperatorGPU> rewardOperators;
    std::vector<MRTaskTerminationOperatorGPU> terminationOperators;
    std::vector<MRTaskRandomizationOperatorGPU>
        randomizationOperators;
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
    if (!world.valid()) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world",
            "task compilation requires a valid compiled world"
        );
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
        pack.curriculumLevelCount == 0u ||
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
        !finite(pack.commands.curriculumStep) ||
        !finite(pack.commands.standingProbability) ||
        pack.commands.standingProbability < 0.0f ||
        pack.commands.standingProbability > 1.0f ||
        !finite(
            pack.commands.minimumEpisodeSurvivalFraction
        ) ||
        pack.commands.minimumEpisodeSurvivalFraction < 0.0f ||
        pack.commands.minimumEpisodeSurvivalFraction > 1.0f ||
        !finite(pack.commands.minimumDurationSeconds) ||
        !finite(pack.commands.maximumDurationSeconds) ||
        !(pack.commands.minimumDurationSeconds > 0.0f) ||
        pack.commands.maximumDurationSeconds <
            pack.commands.minimumDurationSeconds ||
        !finite(pack.pushes.maximumVelocity) ||
        pack.pushes.maximumVelocity < 0.0f ||
        !finite(pack.pushes.minimumIntervalSeconds) ||
        !finite(pack.pushes.maximumIntervalSeconds) ||
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
        pack.commands.curriculumStep.x,
        pack.commands.curriculumStep.y,
        pack.commands.curriculumStep.z,
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
                "command range, limits, or curriculum step are invalid"
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
            !finite(binding.response) ||
            !(binding.response > 0.0f) ||
            binding.response > 1.0f) {
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
                binding.response,
            },
        });
    }

    std::vector<std::string> contactGroupIds;
    contactGroupIds.reserve(pack.contactGroups.size());
    staged->contactGroups.reserve(pack.contactGroups.size());
    std::uint32_t contactMetricCount = 0u;
    for (const TaskContactGroup& group : pack.contactGroups) {
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
        const std::uint32_t baseMetricWidth =
            group.support ? 6u : group.forbidden ? 1u : 0u;
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
            case TaskObservationSource::rootHeight:
                break;
            case TaskObservationSource::gaitPhase:
                componentLimit = 2u;
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

    TaskCompileDiagnostics observationStatus =
        compileObservations(
            pack.actorFrame,
            staged->actorOperators
        );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }
    observationStatus = compileObservations(
        pack.critic,
        staged->criticOperators
    );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }

    staged->rewardOperators.reserve(pack.rewards.size());
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
        case TaskRewardOperator::gaitContactMatch:
        case TaskRewardOperator::swingClearance:
        case TaskRewardOperator::footClearance:
        case TaskRewardOperator::supportSlip:
        case TaskRewardOperator::forbiddenContact:
            if (!reward.sourceGroup.empty()) {
                sourceIndex = namedGroup(
                    contactGroupIds,
                    reward.sourceGroup
                );
            }
            break;
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
                TaskRewardOperator::jointLimitViolationAbsolute &&
            (!(reward.parameters.x > 0.0f) ||
             reward.parameters.x > 1.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.sourceGroup,
                "soft joint-limit factor must be in (0, 1]"
            );
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
                0u,
                0u,
            },
            {
                reward.weight,
                reward.parameters.x,
                reward.parameters.y,
                reward.parameters.z,
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
            termination.reason ==
                MR_TASK_TERMINATION_CONTINUING ||
            termination.reason >
                MR_TASK_TERMINATION_PHYSICS_ERROR) {
            return reject(
                TaskCompileStatus::invalidPack,
                termination.sourceGroup,
                "termination threshold, failure penalty, or reason is invalid"
            );
        }
        std::uint32_t sourceIndex = MR_INVALID_INDEX;
        switch (termination.operation) {
        case TaskTerminationOperator::contactGroup:
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
            random.minimumCurriculumLevel >=
                pack.curriculumLevelCount) {
            return reject(
                TaskCompileStatus::invalidPack,
                random.target,
                "randomization parameters or curriculum gate are invalid"
            );
        }
        std::uint32_t targetIndex = MR_INVALID_INDEX;
        mr_float4 compiledParameters = random.parameters;
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
            if (!orderedRange(random.parameters)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    random.target,
                    "randomization lower bound exceeds its upper bound"
                );
            }
            break;
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
        staged->randomizationOperators.push_back({
            {
                static_cast<std::uint32_t>(random.operation),
                targetIndex,
                random.component,
                random.minimumCurriculumLevel,
            },
            compiledParameters,
        });
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
    std::uint32_t actorFrameSize = 0u;
    std::uint32_t criticOperatorCount = 0u;
    std::uint32_t biasCount = 0u;
    if (!narrowCount(staged->actionBindings.size(), actionCount) ||
        !narrowCount(
            staged->actorOperators.size(),
            actorFrameSize
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
    const std::uint64_t actorObservationSize =
        static_cast<std::uint64_t>(actorFrameSize) *
        pack.actorHistoryLength;
    const std::uint64_t criticObservationSize =
        (pack.criticIncludesCleanHistory
             ? actorObservationSize
             : 0u) +
        static_cast<std::uint64_t>(criticOperatorCount) *
            pack.criticHistoryLength;
    if (actorObservationSize >
            std::numeric_limits<std::uint32_t>::max() ||
        criticObservationSize >
            std::numeric_limits<std::uint32_t>::max() ||
        pack.maximumActionDelaySteps ==
            std::numeric_limits<std::uint32_t>::max()) {
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
        .delayStateCount = std::max(
            pack.maximumActionDelaySteps + 1u,
            2u
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
        pack.curriculumLevelCount,
        heightfieldTerrain
            ? MR_TASK_PROGRAM_TERRAIN
            : 0u,
    };
    if (pack.criticIncludesCleanHistory) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY;
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
        pack.commands.minimumEpisodeSurvivalFraction;
    staged->commandCurriculum = {
        pack.commands.limitLower,
        pack.commands.limitUpper,
        pack.commands.curriculumStep,
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
        0.0f,
        0.0f,
    };
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
        0u,
        0u,
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
