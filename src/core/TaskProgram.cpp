#include "metalrobo/TaskProgram.hpp"

#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/SensorProgram.hpp"
#include "metalrobo/runtime_abi_generated.h"

#include "SemanticTransform.hpp"

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
#include <unordered_set>
#include <utility>
#include <vector>

namespace metalrobo {

struct CompiledTaskProgram::Storage {
    std::uint64_t fingerprint = 0u;
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t sensorFingerprint = 0u;
    TaskProgramLayout layout{};
    MRTaskProgramHeaderGPU header{};
    std::vector<MRTaskActionBindingGPU> actionBindings;
    std::vector<MRTaskObservationOperatorGPU> actorOperators;
    std::vector<MRTaskObservationOperatorGPU> criticOperators;
    std::vector<MRTaskObservationOperatorGPU> signalSources;
    std::vector<MRTaskSignalOperatorGPU> signalOperators;
    std::vector<MRTaskCommandOperatorGPU> commandOperators;
    std::vector<std::string> commandIds;
    std::vector<MRTaskEventOperatorGPU> eventOperators;
    std::vector<std::string> eventIds;
    std::vector<MRTaskContactGroupGPU> contactGroups;
    std::vector<std::uint32_t> contactMembers;
    std::vector<MRTaskFrameGPU> frames;
    std::vector<MRTaskKinematicFrameGPU> kinematicFrames;
    std::vector<MRTaskGoalGPU> goals;
    std::vector<MRArticulatedPointImpulseGPU>
        kinematicPointQueries;
    std::vector<TaskKinematicCohort> kinematicCohorts;
    std::vector<MRTaskRewardOperatorGPU> rewardOperators;
    std::vector<MRTaskRecorderOperatorGPU> recorderOperators;
    std::vector<std::string> recorderIds;
    std::vector<MRTaskTerminationOperatorGPU> terminationOperators;
    std::vector<MRTaskRandomizationOperatorGPU>
        randomizationOperators;
    std::vector<MRTaskBiasSpecGPU> biasSpecs;
    std::vector<mr_float4> terrainSampleOffsets;
    std::vector<mr_float4> terrainResetTranslations;
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

using semantic_transform::normalizeQuaternion;

constexpr float kPi = 3.14159265358979323846f;

bool zeroVector4(const mr_float4 value) {
    return value.x == 0.0f && value.y == 0.0f &&
        value.z == 0.0f && value.w == 0.0f;
}

bool identityQuaternion(const mr_float4 value) {
    mr_float4 normalized{};
    return normalizeQuaternion(value, normalized) &&
        normalized.x == 0.0f && normalized.y == 0.0f &&
        normalized.z == 0.0f &&
        std::abs(normalized.w) == 1.0f;
}

std::uint64_t goalRandomIdentity(const std::string_view id) {
    Hash hash;
    hash.string("MetalRobo.TaskIR.goal-counter-key");
    hash.string(id);
    return hash.finish();
}

std::uint64_t commandRandomIdentity(const std::string_view id) {
    Hash hash;
    hash.string("MetalRobo.TaskIR.command-counter-key");
    hash.string(id);
    return hash.finish();
}

std::uint64_t eventRandomIdentity(const std::string_view id) {
    Hash hash;
    hash.string("MetalRobo.TaskIR.event-counter-key");
    hash.string(id);
    return hash.finish();
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
        "numiGeneralizedVelocities",
        &MetalWorldCapacityProfile::numiGeneralizedVelocities,
    },
    CapacityField{
        "numiRows",
        &MetalWorldCapacityProfile::numiRows,
    },
    CapacityField{
        "numiKrylovVectors",
        &MetalWorldCapacityProfile::numiKrylovVectors,
    },
    CapacityField{
        "numiDirectTiles",
        &MetalWorldCapacityProfile::numiDirectTiles,
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
    const std::uint64_t headerSensorFingerprint =
        storage_ == nullptr
        ? 0u
        : static_cast<std::uint64_t>(
              storage_->header.typedCounts.z
          ) |
              (static_cast<std::uint64_t>(
                   storage_->header.typedCounts.w
               ) << 32u);
    if (storage_ == nullptr ||
        storage_->fingerprint == 0u ||
        storage_->worldFingerprint == 0u ||
        storage_->header.taskFingerprint !=
            storage_->fingerprint ||
        storage_->header.worldFingerprint !=
            storage_->worldFingerprint ||
        headerSensorFingerprint !=
            storage_->sensorFingerprint ||
        storage_->layout.actionCount == 0u ||
        storage_->layout.actorFrameSize == 0u ||
        storage_->layout.actorHistoryLength == 0u ||
        storage_->layout.kinematicPointQueryCount !=
            storage_->kinematicPointQueries.size() ||
        storage_->layout.signalCount !=
            storage_->signalOperators.size() ||
        storage_->layout.commandCount !=
            storage_->commandOperators.size() ||
        static_cast<std::uint64_t>(
            storage_->layout.scalarStateCount
        ) !=
            static_cast<std::uint64_t>(
                storage_->layout.contactMetricCount
            ) + storage_->layout.commandCount ||
        storage_->commandOperators.size() !=
            storage_->commandIds.size() ||
        storage_->header.curriculum.w !=
            storage_->commandOperators.size() ||
        storage_->layout.eventCount !=
            storage_->eventOperators.size() ||
        storage_->eventOperators.size() !=
            storage_->eventIds.size() ||
        storage_->header.counts1.z !=
            storage_->eventOperators.size() ||
        storage_->layout.recorderCount !=
            storage_->recorderOperators.size() ||
        storage_->recorderOperators.size() !=
            storage_->recorderIds.size() ||
        storage_->header.counts1.y !=
            storage_->recorderOperators.size() ||
        storage_->header.curriculum.x == 0u ||
        storage_->header.curriculum.y == 0u ||
        (storage_->header.curriculum.z != MR_INVALID_INDEX &&
         storage_->header.curriculum.z >=
             storage_->signalOperators.size()) ||
        storage_->header.graphCounts.x !=
            storage_->signalOperators.size() ||
        storage_->header.graphCounts.y !=
            storage_->signalSources.size() ||
        storage_->header.graphCounts.z !=
            storage_->layout.spatialJacobianEnvironmentStride ||
        storage_->header.graphCounts.w !=
            storage_->layout.signalSensorScratchCount ||
        (storage_->layout.signalSensorScratchCount != 0u &&
         storage_->sensorFingerprint == 0u) ||
        storage_->kinematicFrames.size() !=
            storage_->frames.size()) {
        return false;
    }

    std::uint32_t expectedSensorScratch = 0u;
    for (const MRTaskObservationOperatorGPU& source :
         storage_->signalSources) {
        const bool sensorSource =
            source.source.x == MR_TASK_OBSERVE_SENSOR_VALUE ||
            source.source.x == MR_TASK_OBSERVE_SENSOR_VALIDITY;
        if (sensorSource) {
            if (source.auxiliary.w != expectedSensorScratch) {
                return false;
            }
            ++expectedSensorScratch;
        } else if (source.auxiliary.w != MR_INVALID_INDEX) {
            return false;
        }
    }
    if (expectedSensorScratch !=
        storage_->layout.signalSensorScratchCount) {
        return false;
    }

    std::uint64_t queryOffset = 0u;
    std::uint64_t pointPrefix = 0u;
    std::uint64_t jacobianPrefix = 0u;
    for (std::size_t owner = 0u;
         owner < storage_->kinematicCohorts.size();
         ++owner) {
        const TaskKinematicCohort& cohort =
            storage_->kinematicCohorts[owner];
        if (cohort.articulationIndex != owner ||
            cohort.queryOffset != queryOffset ||
            cohort.pointPrefix != pointPrefix ||
            cohort.jacobianPrefix != jacobianPrefix) {
            return false;
        }
        queryOffset += cohort.queryCount;
        pointPrefix += cohort.queryCount;
        jacobianPrefix += cohort.jacobianEnvironmentStride;
        if (queryOffset >
                storage_->kinematicPointQueries.size() ||
            pointPrefix >
                storage_->layout.kinematicPointQueryCount ||
            jacobianPrefix >
                storage_->layout
                    .spatialJacobianEnvironmentStride) {
            return false;
        }
    }
    if (queryOffset != storage_->kinematicPointQueries.size() ||
        pointPrefix !=
            storage_->layout.kinematicPointQueryCount ||
        jacobianPrefix !=
            storage_->layout.spatialJacobianEnvironmentStride) {
        return false;
    }
    for (std::size_t frameIndex = 0u;
         frameIndex < storage_->kinematicFrames.size();
         ++frameIndex) {
        const MRTaskKinematicFrameGPU& frame =
            storage_->kinematicFrames[frameIndex];
        if (frame.layout.x == MR_INVALID_INDEX) {
            if (frame.layout.y != MR_INVALID_INDEX ||
                frame.coordinates.y != MR_INVALID_INDEX) {
                return false;
            }
            continue;
        }
        if (frame.layout.x >= storage_->kinematicCohorts.size()) {
            return false;
        }
        const TaskKinematicCohort& cohort =
            storage_->kinematicCohorts[frame.layout.x];
        if (frame.layout.y >= cohort.queryCount ||
            frame.layout.z != cohort.jacobianPrefix ||
            frame.layout.w !=
                cohort.jacobianEnvironmentStride ||
            frame.coordinates.x == 0u ||
            frame.coordinates.y == MR_INVALID_INDEX) {
            return false;
        }
        const std::size_t queryIndex =
            static_cast<std::size_t>(cohort.queryOffset) +
            frame.layout.y;
        if (queryIndex >= storage_->kinematicPointQueries.size() ||
            storage_->kinematicPointQueries[queryIndex].bodyIndex !=
                storage_->frames[frameIndex].indices.x) {
            return false;
        }
    }

    const std::uint64_t frameOffset = storage_->header.offsets3.z;
    const std::uint64_t commandOffset =
        storage_->header.offsets3.y;
    const std::uint64_t commandEnd = commandOffset +
        storage_->commandOperators.size() *
            sizeof(MRTaskCommandOperatorGPU);
    const std::uint64_t recorderOffset =
        storage_->header.offsets1.y;
    const std::uint64_t recorderEnd = recorderOffset +
        storage_->recorderOperators.size() *
            sizeof(MRTaskRecorderOperatorGPU);
    const std::uint64_t eventOffset =
        storage_->header.offsets1.z;
    const std::uint64_t eventEnd = eventOffset +
        storage_->eventOperators.size() *
            sizeof(MRTaskEventOperatorGPU);
    const std::uint64_t rewardOffset =
        storage_->header.offsets1.w;
    const std::uint64_t rewardEnd = rewardOffset +
        storage_->rewardOperators.size() *
            sizeof(MRTaskRewardOperatorGPU);
    const std::uint64_t kinematicOffset =
        frameOffset +
        storage_->frames.size() * sizeof(MRTaskFrameGPU);
    const std::uint64_t goalOffset =
        kinematicOffset +
        storage_->kinematicFrames.size() *
            sizeof(MRTaskKinematicFrameGPU);
    const std::uint64_t goalEnd =
        goalOffset +
        storage_->goals.size() * sizeof(MRTaskGoalGPU);
    const std::uint64_t signalSourceOffset =
        storage_->header.offsets4.x;
    const std::uint64_t signalSourceEnd =
        signalSourceOffset +
        storage_->signalSources.size() *
            sizeof(MRTaskObservationOperatorGPU);
    const std::uint64_t signalOperatorOffset =
        storage_->header.offsets4.y;
    const std::uint64_t signalOperatorEnd =
        signalOperatorOffset +
        storage_->signalOperators.size() *
            sizeof(MRTaskSignalOperatorGPU);
    return recorderEnd <= storage_->arena.size() &&
        eventEnd <= storage_->arena.size() &&
        (storage_->eventOperators.empty() ||
         eventOffset >= recorderEnd) &&
        rewardEnd <= storage_->arena.size() &&
        rewardOffset >=
            (storage_->eventOperators.empty()
                 ? recorderEnd
                 : eventEnd) &&
        commandEnd <= storage_->arena.size() &&
        frameOffset >= commandEnd &&
        storage_->header.offsets3.w == goalOffset &&
        goalEnd <= storage_->arena.size() &&
        signalSourceOffset >= goalEnd &&
        signalSourceEnd <= storage_->arena.size() &&
        signalOperatorOffset >= signalSourceEnd &&
        signalOperatorEnd <= storage_->arena.size();
}

std::uint64_t CompiledTaskProgram::fingerprint() const noexcept {
    return valid() ? storage_->fingerprint : 0u;
}

std::uint64_t CompiledTaskProgram::worldFingerprint() const noexcept {
    return valid() ? storage_->worldFingerprint : 0u;
}

std::uint64_t CompiledTaskProgram::sensorFingerprint() const noexcept {
    return valid() ? storage_->sensorFingerprint : 0u;
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

std::span<const MRTaskObservationOperatorGPU>
CompiledTaskProgram::signalSources() const noexcept {
    return valid()
        ? std::span<const MRTaskObservationOperatorGPU>{
              storage_->signalSources
          }
        : std::span<const MRTaskObservationOperatorGPU>{};
}

std::span<const MRTaskSignalOperatorGPU>
CompiledTaskProgram::signalOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskSignalOperatorGPU>{
              storage_->signalOperators
          }
        : std::span<const MRTaskSignalOperatorGPU>{};
}

std::span<const MRTaskCommandOperatorGPU>
CompiledTaskProgram::commandOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskCommandOperatorGPU>{
              storage_->commandOperators
          }
        : std::span<const MRTaskCommandOperatorGPU>{};
}

std::span<const std::string>
CompiledTaskProgram::commandIds() const noexcept {
    return valid()
        ? std::span<const std::string>{storage_->commandIds}
        : std::span<const std::string>{};
}

std::span<const MRTaskEventOperatorGPU>
CompiledTaskProgram::eventOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskEventOperatorGPU>{
              storage_->eventOperators
          }
        : std::span<const MRTaskEventOperatorGPU>{};
}

std::span<const std::string>
CompiledTaskProgram::eventIds() const noexcept {
    return valid()
        ? std::span<const std::string>{storage_->eventIds}
        : std::span<const std::string>{};
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

std::span<const MRTaskFrameGPU>
CompiledTaskProgram::frames() const noexcept {
    return valid()
        ? std::span<const MRTaskFrameGPU>{storage_->frames}
        : std::span<const MRTaskFrameGPU>{};
}

std::span<const MRTaskGoalGPU>
CompiledTaskProgram::goals() const noexcept {
    return valid()
        ? std::span<const MRTaskGoalGPU>{storage_->goals}
        : std::span<const MRTaskGoalGPU>{};
}

std::span<const MRArticulatedPointImpulseGPU>
CompiledTaskProgram::kinematicPointQueries() const noexcept {
    return valid()
        ? std::span<const MRArticulatedPointImpulseGPU>{
              storage_->kinematicPointQueries
          }
        : std::span<const MRArticulatedPointImpulseGPU>{};
}

std::span<const TaskKinematicCohort>
CompiledTaskProgram::kinematicCohorts() const noexcept {
    return valid()
        ? std::span<const TaskKinematicCohort>{
              storage_->kinematicCohorts
          }
        : std::span<const TaskKinematicCohort>{};
}

std::span<const MRTaskRewardOperatorGPU>
CompiledTaskProgram::rewardOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskRewardOperatorGPU>{
              storage_->rewardOperators
          }
        : std::span<const MRTaskRewardOperatorGPU>{};
}

std::span<const MRTaskRecorderOperatorGPU>
CompiledTaskProgram::recorderOperators() const noexcept {
    return valid()
        ? std::span<const MRTaskRecorderOperatorGPU>{
              storage_->recorderOperators
          }
        : std::span<const MRTaskRecorderOperatorGPU>{};
}

std::span<const std::string>
CompiledTaskProgram::recorderIds() const noexcept {
    return valid()
        ? std::span<const std::string>{storage_->recorderIds}
        : std::span<const std::string>{};
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
    const CompiledSensorProgram& sensors,
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
        model.dofNames.size() != model.dofs.size() ||
        !uniqueNonempty(model.bodyNames) ||
        !uniqueNonempty(model.jointNames) ||
        !uniqueNonempty(model.dofNames)) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world.semantics",
            "compiled task worlds require canonical body, joint, and DoF names"
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
    if (sensors.valid() &&
        sensors.worldFingerprint() != world.fingerprint()) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "sensors",
            "compiled sensors do not match the task world"
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
    const bool floatingRoot =
        articulation.rootType == MR_ROOT_FLOATING &&
        articulation.nq >= 7u && articulation.nv >= 6u;
    const bool fixedRoot =
        articulation.rootType == MR_ROOT_FIXED;
    if (!floatingRoot && !fixedRoot) {
        return reject(
            TaskCompileStatus::invalidWorld,
            "world.articulation",
            "native task programs require a fixed or floating articulation root"
        );
    }
    if (fixedRoot) {
        const auto rootObservation = [](const TaskObservationSource source) {
            switch (source) {
            case TaskObservationSource::rootAngularVelocityLocal:
            case TaskObservationSource::projectedGravity:
            case TaskObservationSource::rootLinearVelocityLocal:
            case TaskObservationSource::rootHeight:
            case TaskObservationSource::terrainHeight:
                return true;
            default:
                return false;
            }
        };
        const bool requiresFloatingRoot =
            std::any_of(
                pack.actorFrame.begin(),
                pack.actorFrame.end(),
                [&](const TaskObservationOperatorSpec& operation) {
                    return rootObservation(operation.source);
                }
            ) ||
            std::any_of(
                pack.critic.begin(),
                pack.critic.end(),
                [&](const TaskObservationOperatorSpec& operation) {
                    return rootObservation(operation.source);
                }
            ) ||
            std::any_of(
                pack.signals.begin(),
                pack.signals.end(),
                [&](const TaskSignalSpec& signal) {
                    return (
                        signal.operation ==
                            TaskSignalOperator::source &&
                        rootObservation(signal.source.source)
                    ) ||
                        std::any_of(
                            signal.reductionSources.begin(),
                            signal.reductionSources.end(),
                            [&](const TaskObservationOperatorSpec& source) {
                                return rootObservation(source.source);
                            }
                        );
                }
            ) ||
            std::any_of(
                pack.randomization.begin(),
                pack.randomization.end(),
                [](const TaskRandomizationOperatorSpec& operation) {
                    return operation.operation ==
                            TaskRandomizationOperator::rootPosition ||
                        operation.operation ==
                            TaskRandomizationOperator::rootYaw;
                }
            );
        if (requiresFloatingRoot) {
            return reject(
                TaskCompileStatus::unsupportedOperator,
                "world.articulation.root",
                "the authored task uses a floating-root operator on a fixed-base articulation"
            );
        }
    }
    const auto countFits = [](const std::size_t count) {
        return count <
            std::numeric_limits<std::uint32_t>::max();
    };
    if (!countFits(pack.actions.size()) ||
        !countFits(pack.actorFrame.size()) ||
        !countFits(pack.critic.size()) ||
        !countFits(pack.contactGroups.size()) ||
        !countFits(pack.frames.size()) ||
        !countFits(pack.goals.size()) ||
        !countFits(pack.signals.size()) ||
        !countFits(pack.rewards.size()) ||
        !countFits(pack.recorders.size()) ||
        !countFits(pack.terminations.size()) ||
        !countFits(pack.randomization.size()) ||
        !countFits(pack.commands.values.size()) ||
        !countFits(pack.events.values.size()) ||
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
    std::uint64_t signalSourceCount = 0u;
    for (const TaskSignalSpec& signal : pack.signals) {
        const std::uint64_t added =
            signal.operation == TaskSignalOperator::source
            ? 1u
            : signal.reductionSources.size();
        if (added >= std::numeric_limits<std::uint32_t>::max() ||
            signalSourceCount >=
                std::numeric_limits<std::uint32_t>::max() - added) {
            return reject(
                TaskCompileStatus::arithmeticOverflow,
                signal.id,
                "SignalIR source cohort exceeds the 32-bit GPU ABI"
            );
        }
        signalSourceCount += added;
    }
    const std::uint64_t observationOperatorCount =
        static_cast<std::uint64_t>(pack.actorFrame.size()) +
        pack.critic.size() + signalSourceCount;
    if (contactMemberCount >=
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
        pack.recorders.size() >
            MR_TASK_TRANSITION_METRIC_COUNT ||
        pack.curriculum.levelCount == 0u ||
        pack.curriculum.evaluationWindowSteps == 0u ||
        !finite(pack.phase.periodSeconds) ||
        !(pack.phase.periodSeconds > 0.0f) ||
        !finite(pack.curriculum.successThreshold) ||
        !finite(
            pack.curriculum.minimumEpisodeSurvivalFraction
        ) ||
        pack.curriculum.minimumEpisodeSurvivalFraction < 0.0f ||
        pack.curriculum.minimumEpisodeSurvivalFraction > 1.0f ||
        !finite(pack.supportForceThreshold) ||
        !(pack.supportForceThreshold >= 0.0f) ||
        !finite(pack.commands.zeroProbability) ||
        pack.commands.zeroProbability < 0.0f ||
        pack.commands.zeroProbability > 1.0f ||
        !finite(pack.commands.minimumDurationSeconds) ||
        !finite(pack.commands.maximumDurationSeconds) ||
        !(pack.commands.minimumDurationSeconds > 0.0f) ||
        pack.commands.maximumDurationSeconds <
            pack.commands.minimumDurationSeconds ||
        !finite(pack.events.minimumIntervalSeconds) ||
        !finite(pack.events.maximumIntervalSeconds) ||
        !(pack.events.minimumIntervalSeconds > 0.0f) ||
        pack.events.maximumIntervalSeconds <
            pack.events.minimumIntervalSeconds ||
        pack.maximumObservationDelaySteps >=
            pack.actorHistoryLength) {
        return reject(
            TaskCompileStatus::invalidPack,
            "task",
            "task identity, dimensions, timing, or scalar parameters are invalid"
        );
    }
    std::vector<std::string> commandIds;
    commandIds.reserve(pack.commands.values.size());
    for (const TaskCommandSpec& command : pack.commands.values) {
        if (command.id.empty() ||
            std::find(
                commandIds.begin(),
                commandIds.end(),
                command.id
            ) != commandIds.end() ||
            !finite(command.lower) ||
            !finite(command.upper) ||
            !finite(command.limitLower) ||
            !finite(command.limitUpper) ||
            !finite(command.curriculumStep) ||
            command.lower > command.upper ||
            command.limitLower > command.lower ||
            command.upper > command.limitUpper ||
            command.curriculumStep < 0.0f) {
            return reject(
                TaskCompileStatus::invalidPack,
                command.id,
                "command identity, range, limits, or curriculum step are invalid"
            );
        }
        commandIds.push_back(command.id);
    }

    auto staged = std::make_shared<CompiledTaskProgram::Storage>();
    staged->worldFingerprint = world.fingerprint();
    staged->commandIds = commandIds;
    staged->commandOperators.reserve(pack.commands.values.size());
    std::unordered_set<std::uint64_t> commandRandomIdentities;
    for (const TaskCommandSpec& command : pack.commands.values) {
        const std::uint64_t randomIdentity =
            commandRandomIdentity(command.id);
        if (!commandRandomIdentities.insert(randomIdentity).second) {
            return reject(
                TaskCompileStatus::invalidPack,
                command.id,
                "task command ids collide in the 64-bit counter-RNG identity"
            );
        }
        staged->commandOperators.push_back({
            {
                command.lower,
                command.upper,
                command.limitLower,
                command.limitUpper,
            },
            {
                command.curriculumStep,
                0.0f,
                0.0f,
                0.0f,
            },
            {
                static_cast<std::uint32_t>(randomIdentity),
                static_cast<std::uint32_t>(randomIdentity >> 32u),
                0u,
                0u,
            },
        });
    }

    staged->eventIds.reserve(pack.events.values.size());
    staged->eventOperators.reserve(pack.events.values.size());
    std::unordered_set<std::uint64_t> eventRandomIdentities;
    for (const TaskEventSpec& event : pack.events.values) {
        if (event.id.empty() || event.target.empty() ||
            !finite(event.initialLower) ||
            !finite(event.initialUpper) ||
            !finite(event.finalLower) ||
            !finite(event.finalUpper) ||
            event.initialLower > event.initialUpper ||
            event.finalLower > event.finalUpper ||
            std::find(
                staged->eventIds.begin(),
                staged->eventIds.end(),
                event.id
            ) != staged->eventIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                event.id,
                "task event identity or range is invalid"
            );
        }
        if (event.operation !=
            TaskEventOperator::generalizedVelocityDelta) {
            return reject(
                TaskCompileStatus::unsupportedOperator,
                event.id,
                "task event operator is not supported"
            );
        }
        bool ambiguous = false;
        const std::uint32_t dofIndex = uniqueIndex(
            model.dofNames,
            event.target,
            ambiguous
        );
        if (ambiguous) {
            return reject(
                TaskCompileStatus::ambiguousSemantic,
                event.target,
                "task event generalized-velocity identity is ambiguous"
            );
        }
        if (dofIndex == MR_INVALID_INDEX ||
            dofIndex >= model.dofs.size()) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                event.target,
                "task event generalized-velocity coordinate does not exist"
            );
        }
        const MRDofPropertiesGPU& dof = model.dofs[dofIndex];
        if (dof.articulationIndex != world.articulationIndex() ||
            dof.vIndex == MR_INVALID_INDEX ||
            !inRange(
                dof.vIndex,
                articulation.vOffset,
                articulation.nv
            )) {
            return reject(
                TaskCompileStatus::invalidPack,
                event.target,
                "task event coordinate must belong to the selected articulation"
            );
        }
        const std::uint64_t randomIdentity =
            eventRandomIdentity(event.id);
        if (!eventRandomIdentities.insert(randomIdentity).second) {
            return reject(
                TaskCompileStatus::invalidPack,
                event.id,
                "task event ids collide in the 64-bit counter-RNG identity"
            );
        }
        staged->eventIds.push_back(event.id);
        staged->eventOperators.push_back({
            {
                MR_TASK_EVENT_GENERALIZED_VELOCITY_DELTA,
                dof.vIndex,
                static_cast<std::uint32_t>(randomIdentity),
                static_cast<std::uint32_t>(randomIdentity >> 32u),
            },
            {
                event.initialLower,
                event.initialUpper,
                0.0f,
                0.0f,
            },
            {
                event.finalLower,
                event.finalUpper,
                0.0f,
                0.0f,
            },
        });
    }

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
            !(binding.scale > 0.0f)) {
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
                0.0f,
            },
        });
    }

    std::vector<std::string> frameIds;
    frameIds.reserve(pack.frames.size());
    staged->frames.reserve(pack.frames.size());
    for (const TaskFrameSpec& frame : pack.frames) {
        const bool bodyAuthored = !frame.body.empty();
        const bool siteAuthored = !frame.site.empty();
        if (frame.id.empty() || bodyAuthored == siteAuthored ||
            !finite(frame.localPosition) ||
            std::find(
                frameIds.begin(),
                frameIds.end(),
                frame.id
            ) != frameIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                frame.id,
                "task frame requires a unique identity, finite local transform, and exactly one body or site source"
            );
        }
        std::uint32_t body = MR_INVALID_INDEX;
        mr_float4 localPosition = frame.localPosition;
        mr_float4 localOrientation{};
        if (!normalizeQuaternion(
                frame.localOrientation,
                localOrientation
            )) {
            return reject(
                TaskCompileStatus::invalidPack,
                frame.id,
                "task frame orientation is not a finite quaternion"
            );
        }
        const std::string& sourceIdentity = siteAuthored
            ? frame.site
            : frame.body;
        if (siteAuthored) {
            const auto found = std::find_if(
                model.sites.begin(),
                model.sites.end(),
                [&](const EngineSite& site) {
                    return site.id == frame.site;
                }
            );
            if (found == model.sites.end()) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    frame.site,
                    "task frame site does not exist"
                );
            }
            body = found->bodyIndex;
            if (!semantic_transform::compose(
                    found->localPosition,
                    found->localOrientation,
                    frame.localPosition,
                    localOrientation,
                    localPosition,
                    localOrientation
                )) {
                return reject(
                    TaskCompileStatus::invalidWorld,
                    frame.site,
                    "task frame site composition is non-finite"
                );
            }
        } else {
            bool ambiguous = false;
            body = uniqueIndex(
                model.bodyNames,
                frame.body,
                ambiguous
            );
            if (ambiguous) {
                return reject(
                    TaskCompileStatus::ambiguousSemantic,
                    frame.body,
                    "task frame body identity is ambiguous"
                );
            }
            if (body == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    frame.body,
                    "task frame body does not exist"
                );
            }
        }
        if (body >= model.bodies.size()) {
            return reject(
                TaskCompileStatus::invalidWorld,
                sourceIdentity,
                "task frame source has an invalid body index"
            );
        }
        const std::uint32_t articulationOwner =
            model.bodies[body].articulationIndex;
        std::uint32_t sourceKind =
            MR_TASK_FRAME_SOURCE_ARTICULATED_BODY;
        std::uint32_t sourceIndex = body;
        if (articulationOwner == MR_INVALID_INDEX) {
            const auto sceneBody = std::find(
                world.sceneBodyIndices().begin(),
                world.sceneBodyIndices().end(),
                body
            );
            if (sceneBody == world.sceneBodyIndices().end()) {
                return reject(
                    TaskCompileStatus::invalidWorld,
                    sourceIdentity,
                    "non-articulated task frame body is absent from the compiled scene-state layout"
                );
            }
            sourceKind = MR_TASK_FRAME_SOURCE_SCENE_BODY;
            sourceIndex = static_cast<std::uint32_t>(
                sceneBody - world.sceneBodyIndices().begin()
            );
        } else if (articulationOwner >=
                   model.articulations.size()) {
            return reject(
                TaskCompileStatus::invalidWorld,
                sourceIdentity,
                "task frame body has an invalid articulation owner"
            );
        }
        const mr_float4 centerOfMass =
            model.bodies[body].centerOfMass;
        staged->frames.push_back({
            {
                body,
                sourceKind,
                sourceIndex,
                articulationOwner,
            },
            {
                localPosition.x - centerOfMass.x,
                localPosition.y - centerOfMass.y,
                localPosition.z - centerOfMass.z,
                0.0f,
            },
            localOrientation,
        });
        frameIds.push_back(frame.id);
    }

    std::vector<std::string> goalIds;
    std::unordered_set<std::uint64_t> goalRandomIdentities;
    goalIds.reserve(pack.goals.size());
    goalRandomIdentities.reserve(pack.goals.size());
    staged->goals.reserve(pack.goals.size());
    for (const TaskGoalSpec& goal : pack.goals) {
        const std::uint32_t mode =
            static_cast<std::uint32_t>(goal.mode);
        const std::uint32_t playback =
            static_cast<std::uint32_t>(goal.playback);
        if (goal.id.empty() ||
            mode > MR_TASK_GOAL_TRAJECTORY ||
            playback > MR_TASK_GOAL_PLAYBACK_PING_PONG ||
            !finite(goal.position) ||
            !finite(goal.targetPosition) ||
            !finite(goal.positionOffsetLower) ||
            !finite(goal.positionOffsetUpper) ||
            !finite(goal.rotationVectorLower) ||
            !finite(goal.rotationVectorUpper) ||
            !finite(goal.durationSeconds) ||
            !finite(goal.phaseSeconds) ||
            std::find(
                goalIds.begin(),
                goalIds.end(),
                goal.id
            ) != goalIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "task goal identity, mode, pose, range, or timing is invalid"
            );
        }
        mr_float4 orientation{};
        mr_float4 targetOrientation{};
        if (!normalizeQuaternion(goal.orientation, orientation) ||
            !normalizeQuaternion(
                goal.targetOrientation,
                targetOrientation
            )) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "task goal orientations are not finite quaternions"
            );
        }
        const bool orderedPositionRange =
            goal.positionOffsetLower.x <=
                goal.positionOffsetUpper.x &&
            goal.positionOffsetLower.y <=
                goal.positionOffsetUpper.y &&
            goal.positionOffsetLower.z <=
                goal.positionOffsetUpper.z &&
            goal.positionOffsetLower.w == 0.0f &&
            goal.positionOffsetUpper.w == 0.0f;
        const bool orderedRotationRange =
            goal.rotationVectorLower.x <=
                goal.rotationVectorUpper.x &&
            goal.rotationVectorLower.y <=
                goal.rotationVectorUpper.y &&
            goal.rotationVectorLower.z <=
                goal.rotationVectorUpper.z &&
            goal.rotationVectorLower.w == 0.0f &&
            goal.rotationVectorUpper.w == 0.0f;
        const double maximumRotationX = std::max(
            std::abs(goal.rotationVectorLower.x),
            std::abs(goal.rotationVectorUpper.x)
        );
        const double maximumRotationY = std::max(
            std::abs(goal.rotationVectorLower.y),
            std::abs(goal.rotationVectorUpper.y)
        );
        const double maximumRotationZ = std::max(
            std::abs(goal.rotationVectorLower.z),
            std::abs(goal.rotationVectorUpper.z)
        );
        const double maximumRotationNorm = std::sqrt(
            maximumRotationX * maximumRotationX +
            maximumRotationY * maximumRotationY +
            maximumRotationZ * maximumRotationZ
        );
        if (!orderedPositionRange || !orderedRotationRange ||
            maximumRotationNorm > kPi + 1.0e-6) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "sampled goal ranges must be ordered, have zero w, and remain within the principal rotation-vector ball"
            );
        }
        const bool canonicalTarget =
            zeroVector4(goal.targetPosition) &&
            identityQuaternion(goal.targetOrientation);
        const bool canonicalPositionRange =
            zeroVector4(goal.positionOffsetLower) &&
            zeroVector4(goal.positionOffsetUpper);
        const bool canonicalRotationRange =
            zeroVector4(goal.rotationVectorLower) &&
            zeroVector4(goal.rotationVectorUpper);
        if (mode == MR_TASK_GOAL_FIXED &&
            (playback != MR_TASK_GOAL_PLAYBACK_CLAMP ||
             !canonicalTarget || !canonicalPositionRange ||
             !canonicalRotationRange ||
             goal.durationSeconds != 0.0f ||
             goal.phaseSeconds != 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "fixed goal contains sampled or trajectory fields"
            );
        }
        if (mode == MR_TASK_GOAL_SAMPLED_EPISODE &&
            (playback != MR_TASK_GOAL_PLAYBACK_CLAMP ||
             !canonicalTarget ||
             goal.durationSeconds != 0.0f ||
             goal.phaseSeconds != 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "episode-sampled goal contains trajectory fields"
            );
        }
        if (mode == MR_TASK_GOAL_TRAJECTORY &&
            (!canonicalPositionRange ||
             !canonicalRotationRange ||
             !(goal.durationSeconds > 0.0f) ||
             goal.phaseSeconds < 0.0f)) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "trajectory goal requires positive duration, non-negative phase, and no sampled ranges"
            );
        }
        const std::uint64_t randomIdentity =
            goalRandomIdentity(goal.id);
        if (!goalRandomIdentities.insert(randomIdentity).second) {
            return reject(
                TaskCompileStatus::invalidPack,
                goal.id,
                "task goal ids collide in the 64-bit counter-RNG identity"
            );
        }
        const bool trajectoryMode =
            mode == MR_TASK_GOAL_TRAJECTORY;
        staged->goals.push_back({
            {
                mode,
                playback,
                static_cast<std::uint32_t>(randomIdentity),
                static_cast<std::uint32_t>(randomIdentity >> 32u),
            },
            {
                goal.position.x,
                goal.position.y,
                goal.position.z,
                1.0f,
            },
            orientation,
            {
                trajectoryMode ? goal.targetPosition.x : 0.0f,
                trajectoryMode ? goal.targetPosition.y : 0.0f,
                trajectoryMode ? goal.targetPosition.z : 0.0f,
                1.0f,
            },
            trajectoryMode
                ? targetOrientation
                : mr_float4{0.0f, 0.0f, 0.0f, 1.0f},
            goal.positionOffsetLower,
            goal.positionOffsetUpper,
            goal.rotationVectorLower,
            goal.rotationVectorUpper,
            {
                goal.durationSeconds,
                goal.phaseSeconds,
                0.0f,
                0.0f,
            },
        });
        goalIds.push_back(goal.id);
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

    bool usesSensors = false;
    std::vector<bool> jacobianFrames(
        staged->frames.size(),
        false
    );
    const auto compileObservations =
        [&](const std::vector<TaskObservationOperatorSpec>& specs,
            std::vector<MRTaskObservationOperatorGPU>& compiled,
            const std::uint32_t requiredConsumer)
        -> TaskCompileDiagnostics {
        compiled.reserve(specs.size());
        for (std::size_t operatorIndex = 0u;
             operatorIndex < specs.size();
             ++operatorIndex) {
            const TaskObservationOperatorSpec& spec =
                specs[operatorIndex];
            if (!finite(spec.scale) ||
                !finite(spec.offset) ||
                !finite(spec.parameters) ||
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
            if (spec.source !=
                    TaskObservationSource::jointSoftLimitViolation &&
                !zeroVector4(spec.parameters)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.target,
                    "observation source contains unsupported parameters"
                );
            }
            const std::uint32_t opcode =
                static_cast<std::uint32_t>(spec.source);
            std::uint32_t sourceIndex = MR_INVALID_INDEX;
            std::uint32_t goalIndex = MR_INVALID_INDEX;
            std::uint32_t componentLimit = 1u;
            mr_float4 compiledParameters = spec.parameters;
            switch (spec.source) {
            case TaskObservationSource::rootAngularVelocityLocal:
            case TaskObservationSource::projectedGravity:
            case TaskObservationSource::rootLinearVelocityLocal:
                componentLimit = 3u;
                break;
            case TaskObservationSource::command:
                sourceIndex = namedGroup(
                    staged->commandIds,
                    spec.target
                );
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation command identity does not exist"
                    );
                }
                break;
            case TaskObservationSource::jointPositionError:
            case TaskObservationSource::jointVelocity:
            case TaskObservationSource::previousAction:
            case TaskObservationSource::jointAcceleration:
            case TaskObservationSource::actionDelta:
            case TaskObservationSource::jointSoftLimitViolation: {
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
                if (spec.source ==
                        TaskObservationSource::jointSoftLimitViolation) {
                    const float softFactor = spec.parameters.x;
                    if (!(softFactor > 0.0f) || softFactor > 1.0f ||
                        spec.parameters.y != 0.0f ||
                        spec.parameters.z != 0.0f ||
                        spec.parameters.w != 0.0f) {
                        return reject(
                            TaskCompileStatus::invalidPack,
                            spec.target,
                            "joint soft-limit source requires one factor in (0, 1]"
                        );
                    }
                    const std::uint32_t dofIndex =
                        staged->actionBindings[sourceIndex].indices.y;
                    if (dofIndex >= model.dofs.size()) {
                        return reject(
                            TaskCompileStatus::invalidWorld,
                            spec.target,
                            "joint soft-limit source resolved an invalid DoF"
                        );
                    }
                    const MRDofPropertiesGPU& dof = model.dofs[dofIndex];
                    const float center =
                        0.5f * (dof.limits.x + dof.limits.y);
                    const float halfRange =
                        0.5f * (dof.limits.y - dof.limits.x) *
                        softFactor;
                    compiledParameters = {
                        center - halfRange,
                        center + halfRange,
                        0.0f,
                        0.0f,
                    };
                }
                break;
            }
            case TaskObservationSource::rootHeight:
            case TaskObservationSource::mechanicalPower:
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
            case TaskObservationSource::desiredSupportContact: {
                sourceIndex = namedGroup(
                    contactGroupIds,
                    spec.target
                );
                if (sourceIndex == MR_INVALID_INDEX ||
                    (staged->contactGroups[sourceIndex].members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "desired-contact source requires a semantic support group"
                    );
                }
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
            case TaskObservationSource::framePositionWorld:
            case TaskObservationSource::frameOrientationWorld:
            case TaskObservationSource::frameGoalPositionError:
            case TaskObservationSource::frameGoalOrientationError:
            case TaskObservationSource::frameRelativePosition:
            case TaskObservationSource::frameRelativeOrientation:
            case TaskObservationSource::frameLinearVelocityWorld:
            case TaskObservationSource::frameLinearVelocityHeading:
            case TaskObservationSource::frameAngularVelocityWorld:
            case TaskObservationSource::frameRelativeLinearVelocity:
            case TaskObservationSource::frameRelativeAngularVelocity:
                sourceIndex = namedGroup(frameIds, spec.target);
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation task frame does not exist"
                    );
                }
                componentLimit = spec.source ==
                        TaskObservationSource::frameOrientationWorld
                    ? 4u
                    : spec.source ==
                          TaskObservationSource::frameLinearVelocityHeading
                    ? 2u
                    : 3u;
                if (spec.source ==
                        TaskObservationSource::frameGoalPositionError ||
                    spec.source ==
                        TaskObservationSource::frameGoalOrientationError) {
                    goalIndex = namedGroup(goalIds, spec.goal);
                    if (goalIndex == MR_INVALID_INDEX) {
                        return reject(
                            TaskCompileStatus::unresolvedSemantic,
                            spec.goal,
                            "observation task goal does not exist"
                        );
                    }
                } else if (
                    spec.source ==
                        TaskObservationSource::frameRelativePosition ||
                    spec.source ==
                        TaskObservationSource::frameRelativeOrientation ||
                    spec.source ==
                        TaskObservationSource::frameRelativeLinearVelocity ||
                    spec.source ==
                        TaskObservationSource::frameRelativeAngularVelocity
                ) {
                    goalIndex = namedGroup(
                        frameIds,
                        spec.reference
                    );
                    if (goalIndex == MR_INVALID_INDEX) {
                        return reject(
                            TaskCompileStatus::unresolvedSemantic,
                            spec.reference,
                            "observation reference frame does not exist"
                        );
                    }
                    if (!spec.goal.empty()) {
                        return reject(
                            TaskCompileStatus::invalidPack,
                            spec.goal,
                            "a frame-relative observation cannot bind a static goal"
                        );
                    }
                } else if (!spec.goal.empty() ||
                           !spec.reference.empty()) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        !spec.goal.empty()
                            ? spec.goal
                            : spec.reference,
                        "a direct frame observation cannot bind a goal or reference frame"
                    );
                }
                break;
            case TaskObservationSource::frameLinearJacobianWorld:
            case TaskObservationSource::frameAngularJacobianWorld: {
                sourceIndex = namedGroup(frameIds, spec.target);
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "Jacobian observation task frame does not exist"
                    );
                }
                if (staged->frames[sourceIndex].indices.y !=
                    MR_TASK_FRAME_SOURCE_ARTICULATED_BODY) {
                    return reject(
                        TaskCompileStatus::unsupportedOperator,
                        spec.target,
                        "Jacobian observations currently require an articulated task frame"
                    );
                }
                if (spec.coordinate.empty()) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.target,
                        "Jacobian observation requires a named generalized-velocity coordinate"
                    );
                }
                bool ambiguous = false;
                const std::uint32_t dofIndex = uniqueIndex(
                    model.dofNames,
                    spec.coordinate,
                    ambiguous
                );
                if (ambiguous) {
                    return reject(
                        TaskCompileStatus::ambiguousSemantic,
                        spec.coordinate,
                        "Jacobian coordinate identity is ambiguous"
                    );
                }
                if (dofIndex == MR_INVALID_INDEX ||
                    dofIndex >= model.dofs.size() ||
                    model.dofs[dofIndex].vIndex != dofIndex) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.coordinate,
                        "Jacobian generalized-velocity coordinate does not exist"
                    );
                }
                componentLimit = 3u;
                goalIndex = dofIndex;
                jacobianFrames[sourceIndex] = true;
                break;
            }
            case TaskObservationSource::sensorValue:
            case TaskObservationSource::sensorValidity: {
                if (!sensors.valid()) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "sensor observation requires a compiled SensorIR program"
                    );
                }
                sourceIndex = sensors.sensorIndex(spec.target);
                if (sourceIndex == MR_INVALID_INDEX) {
                    return reject(
                        TaskCompileStatus::unresolvedSemantic,
                        spec.target,
                        "observation sensor does not exist"
                    );
                }
                const MRSensorDescriptorGPU descriptor =
                    sensors.descriptors()[sourceIndex];
                if ((descriptor.identity.w & requiredConsumer) == 0u) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.target,
                        "observation sensor is not authorized for this task consumer"
                    );
                }
                if (!spec.goal.empty()) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.goal,
                        "sensor observations cannot bind a task goal"
                    );
                }
                if (!spec.reference.empty()) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        spec.reference,
                        "sensor observations cannot bind a reference frame"
                    );
                }
                componentLimit =
                    spec.source == TaskObservationSource::sensorValue
                    ? descriptor.output.y
                    : 6u;
                // auxiliary.z stores the environment-local SensorIR output
                // offset. Validity reads metadata at source.y but retains the
                // offset so both opcodes share one generated record layout.
                goalIndex = descriptor.output.x;
                usesSensors = true;
                break;
            }
            default:
                return reject(
                    TaskCompileStatus::unsupportedOperator,
                    spec.target,
                    "observation opcode is unsupported"
                );
            }
            const bool goalObservation =
                spec.source ==
                    TaskObservationSource::frameGoalPositionError ||
                spec.source ==
                    TaskObservationSource::frameGoalOrientationError;
            const bool relativeObservation =
                spec.source ==
                    TaskObservationSource::frameRelativePosition ||
                spec.source ==
                    TaskObservationSource::frameRelativeOrientation ||
                spec.source ==
                    TaskObservationSource::frameRelativeLinearVelocity ||
                spec.source ==
                    TaskObservationSource::frameRelativeAngularVelocity;
            const bool jacobianObservation =
                spec.source ==
                    TaskObservationSource::frameLinearJacobianWorld ||
                spec.source ==
                    TaskObservationSource::frameAngularJacobianWorld;
            if (!goalObservation && !spec.goal.empty()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.goal,
                    "observation source does not accept a static goal"
                );
            }
            if (!relativeObservation && !spec.reference.empty()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.reference,
                    "observation source does not accept a reference frame"
                );
            }
            if (!jacobianObservation && !spec.coordinate.empty()) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    spec.coordinate,
                    "observation source does not accept a generalized-velocity coordinate"
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
                    goalIndex,
                    0u,
                },
                compiledParameters,
            });
        }
        return {};
    };

    TaskCompileDiagnostics observationStatus =
        compileObservations(
            pack.actorFrame,
            staged->actorOperators,
            MR_WORLD_SENSOR_CONSUMER_ACTOR
        );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }
    observationStatus = compileObservations(
        pack.critic,
        staged->criticOperators,
        MR_WORLD_SENSOR_CONSUMER_CRITIC
    );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }

    std::vector<std::string> signalIds;
    std::vector<TaskObservationOperatorSpec> signalSourceSpecs;
    signalIds.reserve(pack.signals.size());
    signalSourceSpecs.reserve(
        static_cast<std::size_t>(signalSourceCount)
    );
    staged->signalOperators.reserve(pack.signals.size());
    for (const TaskSignalSpec& signal : pack.signals) {
        if (signal.id.empty() ||
            std::find(
                signalIds.begin(),
                signalIds.end(),
                signal.id
            ) != signalIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                signal.id,
                "SignalIR node identities must be non-empty and unique"
            );
        }
        if (!finite(signal.parameters)) {
            return reject(
                TaskCompileStatus::invalidPack,
                signal.id,
                "SignalIR parameters must be finite"
            );
        }

        std::uint32_t sourceIndex = MR_INVALID_INDEX;
        std::uint32_t leftIndex = MR_INVALID_INDEX;
        std::uint32_t rightIndex = MR_INVALID_INDEX;
        const auto resolveEarlier = [&](
            const std::string& operand
        ) -> std::uint32_t {
            return namedGroup(signalIds, operand);
        };
        const auto requireNoOperands = [&]() -> bool {
            return signal.left.empty() && signal.right.empty();
        };
        const auto requireUnary = [&]() -> bool {
            leftIndex = resolveEarlier(signal.left);
            return leftIndex != MR_INVALID_INDEX &&
                signal.right.empty();
        };
        const auto requireBinary = [&]() -> bool {
            leftIndex = resolveEarlier(signal.left);
            rightIndex = resolveEarlier(signal.right);
            return leftIndex != MR_INVALID_INDEX &&
                rightIndex != MR_INVALID_INDEX;
        };
        const auto truthOnly = [](
            const TaskObservationOperatorSpec& source
        ) {
            return source.noiseAmplitude == 0.0f &&
                source.biasLower == 0.0f &&
                source.biasUpper == 0.0f &&
                !source.normalizeVector3;
        };
        bool validShape = true;
        if (signal.operation != TaskSignalOperator::reduction &&
            !signal.reductionSources.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                signal.id,
                "only SignalIR reduction nodes may carry a source cohort"
            );
        }
        switch (signal.operation) {
        case TaskSignalOperator::source:
            validShape = requireNoOperands();
            if (!truthOnly(signal.source)) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    signal.id,
                    "SignalIR source leaves are truth-only and cannot request actor corruption"
                );
            }
            sourceIndex = static_cast<std::uint32_t>(
                signalSourceSpecs.size()
            );
            signalSourceSpecs.push_back(signal.source);
            break;
        case TaskSignalOperator::reduction: {
            validShape = requireNoOperands() &&
                !signal.reductionSources.empty() &&
                static_cast<std::uint32_t>(signal.transform) <=
                    MR_TASK_SIGNAL_TRANSFORM_SQUARE &&
                static_cast<std::uint32_t>(signal.reduction) <=
                    MR_TASK_SIGNAL_REDUCE_MAXIMUM;
            sourceIndex = static_cast<std::uint32_t>(
                signalSourceSpecs.size()
            );
            leftIndex = static_cast<std::uint32_t>(
                signal.reductionSources.size()
            );
            rightIndex =
                static_cast<std::uint32_t>(signal.transform) |
                (static_cast<std::uint32_t>(signal.reduction) << 8u);
            for (const TaskObservationOperatorSpec& source :
                 signal.reductionSources) {
                if (!truthOnly(source)) {
                    return reject(
                        TaskCompileStatus::invalidPack,
                        signal.id,
                        "SignalIR reduction sources are truth-only"
                    );
                }
                signalSourceSpecs.push_back(source);
            }
            break;
        }
        case TaskSignalOperator::constant:
            validShape = requireNoOperands();
            break;
        case TaskSignalOperator::add:
        case TaskSignalOperator::subtract:
        case TaskSignalOperator::multiply:
        case TaskSignalOperator::minimum:
        case TaskSignalOperator::maximum:
        case TaskSignalOperator::atan2:
            validShape = requireBinary();
            break;
        case TaskSignalOperator::safeDivide:
            validShape = requireBinary() &&
                signal.parameters.x > 0.0f;
            break;
        case TaskSignalOperator::absolute:
        case TaskSignalOperator::square:
        case TaskSignalOperator::squareRoot:
        case TaskSignalOperator::hyperbolicTangent:
        case TaskSignalOperator::lessThan:
        case TaskSignalOperator::greaterThan:
            validShape = requireUnary();
            break;
        case TaskSignalOperator::clamp:
        case TaskSignalOperator::insideBounds:
            validShape = requireUnary() &&
                signal.parameters.x <= signal.parameters.y;
            break;
        case TaskSignalOperator::exponentialTracking:
            validShape = requireUnary() &&
                signal.parameters.y > 0.0f;
            break;
        case TaskSignalOperator::exponentialDecay:
            validShape = requireUnary() &&
                signal.parameters.x > 0.0f;
            break;
        default:
            return reject(
                TaskCompileStatus::unsupportedOperator,
                signal.id,
                "SignalIR opcode is unsupported"
            );
        }
        if (!validShape) {
            return reject(
                TaskCompileStatus::invalidPack,
                signal.id,
                "SignalIR operands must reference earlier nodes and match the operator arity"
            );
        }
        staged->signalOperators.push_back({
            {
                static_cast<std::uint32_t>(signal.operation),
                sourceIndex,
                leftIndex,
                rightIndex,
            },
            signal.parameters,
        });
        signalIds.push_back(signal.id);
    }
    observationStatus = compileObservations(
        signalSourceSpecs,
        staged->signalSources,
        MR_WORLD_SENSOR_CONSUMER_TRUTH
    );
    if (!observationStatus.succeeded()) {
        return observationStatus;
    }
    std::uint32_t signalSensorScratchCount = 0u;
    for (MRTaskObservationOperatorGPU& source :
         staged->signalSources) {
        if (source.source.x == MR_TASK_OBSERVE_SENSOR_VALUE ||
            source.source.x == MR_TASK_OBSERVE_SENSOR_VALIDITY) {
            source.auxiliary.w = signalSensorScratchCount++;
        } else {
            source.auxiliary.w = MR_INVALID_INDEX;
        }
    }

    staged->kinematicFrames.assign(
        staged->frames.size(),
        MRTaskKinematicFrameGPU{
            {
                MR_INVALID_INDEX,
                MR_INVALID_INDEX,
                0u,
                0u,
            },
            {
                0u,
                MR_INVALID_INDEX,
                0u,
                0u,
            },
        }
    );
    staged->kinematicCohorts.resize(model.articulations.size());
    std::vector<std::uint32_t> ownerQueryCounts(
        model.articulations.size(),
        0u
    );
    for (std::size_t frameIndex = 0u;
         frameIndex < staged->frames.size();
         ++frameIndex) {
        if (!jacobianFrames[frameIndex]) {
            continue;
        }
        const std::uint32_t owner =
            staged->frames[frameIndex].indices.w;
        if (owner >= ownerQueryCounts.size() ||
            ownerQueryCounts[owner] ==
                std::numeric_limits<std::uint32_t>::max()) {
            return reject(
                TaskCompileStatus::arithmeticOverflow,
                frameIds[frameIndex],
                "semantic point-query cohort exceeds the 32-bit ABI"
            );
        }
        ++ownerQueryCounts[owner];
    }
    std::uint64_t pointPrefix = 0u;
    std::uint64_t jacobianPrefix = 0u;
    for (std::uint32_t owner = 0u;
         owner < model.articulations.size();
         ++owner) {
        const MRArticulationGPU& owned =
            model.articulations[owner];
        const std::uint64_t ownerJacobianStride =
            static_cast<std::uint64_t>(
                ownerQueryCounts[owner]
            ) * 6u * owned.nv;
        if (pointPrefix >
                std::numeric_limits<std::uint32_t>::max() ||
            pointPrefix + ownerQueryCounts[owner] >
                std::numeric_limits<std::uint32_t>::max() ||
            jacobianPrefix >
                std::numeric_limits<std::uint32_t>::max() ||
            ownerJacobianStride >
                std::numeric_limits<std::uint32_t>::max() ||
            jacobianPrefix + ownerJacobianStride >
                std::numeric_limits<std::uint32_t>::max()) {
            return reject(
                TaskCompileStatus::arithmeticOverflow,
                "kinematic_queries",
                "semantic spatial-Jacobian layout exceeds the 32-bit ABI"
            );
        }
        TaskKinematicCohort& cohort =
            staged->kinematicCohorts[owner];
        cohort.articulationIndex = owner;
        cohort.queryOffset = static_cast<std::uint32_t>(
            staged->kinematicPointQueries.size()
        );
        cohort.queryCount = ownerQueryCounts[owner];
        cohort.pointPrefix = static_cast<std::uint32_t>(
            pointPrefix
        );
        cohort.jacobianPrefix = static_cast<std::uint32_t>(
            jacobianPrefix
        );
        cohort.jacobianEnvironmentStride =
            static_cast<std::uint32_t>(ownerJacobianStride);
        std::uint32_t localQuery = 0u;
        for (std::size_t frameIndex = 0u;
             frameIndex < staged->frames.size();
             ++frameIndex) {
            if (!jacobianFrames[frameIndex] ||
                staged->frames[frameIndex].indices.w != owner) {
                continue;
            }
            staged->kinematicFrames[frameIndex] = {
                {
                    owner,
                    localQuery,
                    cohort.jacobianPrefix,
                    cohort.jacobianEnvironmentStride,
                },
                {
                    owned.nv,
                    owned.vOffset,
                    0u,
                    0u,
                },
            };
            MRArticulatedPointImpulseGPU query{};
            query.bodyIndex =
                staged->frames[frameIndex].indices.x;
            query.localPoint =
                staged->frames[frameIndex].localPosition;
            staged->kinematicPointQueries.push_back(query);
            ++localQuery;
        }
        pointPrefix += ownerQueryCounts[owner];
        jacobianPrefix += ownerJacobianStride;
    }
    staged->rewardOperators.reserve(pack.rewards.size());
    for (const TaskRewardOperatorSpec& reward : pack.rewards) {
        if (!finite(reward.weight)) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.signal,
                "reward weight is non-finite"
            );
        }
        if (static_cast<std::uint32_t>(reward.channel) >=
            MR_TASK_REWARD_CHANNEL_COUNT) {
            return reject(
                TaskCompileStatus::invalidPack,
                reward.signal,
                "reward channel exceeds the compact transition layout"
            );
        }
        const std::uint32_t sourceIndex = namedGroup(
            signalIds,
            reward.signal
        );
        if (sourceIndex == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                reward.signal,
                "SignalIR reward source does not exist"
            );
        }
        staged->rewardOperators.push_back({
            {
                sourceIndex,
                static_cast<std::uint32_t>(reward.channel),
                0u,
                0u,
            },
            {
                reward.weight,
                0.0f,
                0.0f,
                0.0f,
            },
        });
    }

    staged->recorderOperators.reserve(pack.recorders.size());
    staged->recorderIds.reserve(pack.recorders.size());
    for (std::size_t recorderIndex = 0u;
         recorderIndex < pack.recorders.size();
         ++recorderIndex) {
        const TaskRecorderSpec& recorder =
            pack.recorders[recorderIndex];
        if (recorder.id.empty() ||
            std::find(
                staged->recorderIds.begin(),
                staged->recorderIds.end(),
                recorder.id
            ) != staged->recorderIds.end()) {
            return reject(
                TaskCompileStatus::invalidPack,
                recorder.id,
                "recorder identity is empty or duplicated"
            );
        }
        const std::uint32_t sourceIndex = namedGroup(
            signalIds,
            recorder.signal
        );
        if (sourceIndex == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                recorder.signal,
                "recorder SignalIR source does not exist"
            );
        }
        staged->recorderOperators.push_back({{
            sourceIndex,
            static_cast<std::uint32_t>(recorderIndex),
            0u,
            0u,
        }});
        staged->recorderIds.push_back(recorder.id);
    }

    std::uint32_t curriculumSignalIndex = MR_INVALID_INDEX;
    if (!pack.curriculum.successSignal.empty()) {
        curriculumSignalIndex = namedGroup(
            signalIds,
            pack.curriculum.successSignal
        );
        if (curriculumSignalIndex == MR_INVALID_INDEX) {
            return reject(
                TaskCompileStatus::unresolvedSemantic,
                pack.curriculum.successSignal,
                "curriculum success SignalIR source does not exist"
            );
        }
    } else if (pack.curriculum.levelCount > 1u) {
        return reject(
            TaskCompileStatus::invalidPack,
            "curriculum",
            "a multi-level curriculum requires a success signal"
        );
    }

    staged->terminationOperators.reserve(
        pack.terminations.size()
    );
    for (const TaskTerminationOperatorSpec& termination :
         pack.terminations) {
        if (!finite(termination.threshold) ||
            !finite(termination.upperThreshold) ||
            !finite(termination.failurePenalty) ||
            termination.failurePenalty > 0.0f ||
            termination.reason ==
                MR_TASK_TERMINATION_CONTINUING ||
            termination.reason >
                MR_TASK_TERMINATION_GOAL_ERROR) {
            return reject(
                TaskCompileStatus::invalidPack,
                termination.sourceGroup,
                "termination threshold, failure penalty, or reason is invalid"
            );
        }
        std::uint32_t sourceIndex = MR_INVALID_INDEX;
        switch (termination.operation) {
        case TaskTerminationOperator::signalBelow:
        case TaskTerminationOperator::signalAbove:
        case TaskTerminationOperator::signalOutside:
            sourceIndex = namedGroup(
                signalIds,
                termination.signal
            );
            if (sourceIndex == MR_INVALID_INDEX) {
                return reject(
                    TaskCompileStatus::unresolvedSemantic,
                    termination.signal,
                    "SignalIR termination source does not exist"
                );
            }
            if (termination.operation ==
                    TaskTerminationOperator::signalOutside &&
                termination.threshold >
                    termination.upperThreshold) {
                return reject(
                    TaskCompileStatus::invalidPack,
                    termination.signal,
                    "SignalIR outside bounds are inverted"
                );
            }
            break;
        default:
            return reject(
                TaskCompileStatus::unsupportedOperator,
                termination.sourceGroup,
                "termination opcode is unsupported"
            );
        }
        const bool signalTermination =
            termination.operation ==
                TaskTerminationOperator::signalBelow ||
            termination.operation ==
                TaskTerminationOperator::signalAbove ||
            termination.operation ==
                TaskTerminationOperator::signalOutside;
        if (signalTermination &&
            !termination.sourceGroup.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                termination.signal,
                "SignalIR terminations cannot also bind a group"
            );
        }
        if (!signalTermination && !termination.signal.empty()) {
            return reject(
                TaskCompileStatus::invalidPack,
                termination.signal,
                "only SignalIR termination operators may bind a signal"
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
                termination.upperThreshold,
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
                pack.curriculum.levelCount) {
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
    const std::uint64_t scalarStateCount =
        static_cast<std::uint64_t>(contactMetricCount) +
        staged->commandOperators.size();
    if (scalarStateCount >
        std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "task.scalar_state",
            "task scalar-state stride exceeds the 32-bit GPU ABI"
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
        .kinematicPointQueryCount =
            static_cast<std::uint32_t>(
                staged->kinematicPointQueries.size()
            ),
        .spatialJacobianEnvironmentStride =
            static_cast<std::uint32_t>(jacobianPrefix),
        .signalCount = static_cast<std::uint32_t>(
            staged->signalOperators.size()
        ),
        .signalSensorScratchCount =
            signalSensorScratchCount,
        .commandCount = static_cast<std::uint32_t>(
            staged->commandOperators.size()
        ),
        .eventCount = static_cast<std::uint32_t>(
            staged->eventOperators.size()
        ),
        .scalarStateCount =
            static_cast<std::uint32_t>(scalarStateCount),
        .recorderCount = static_cast<std::uint32_t>(
            staged->recorderOperators.size()
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
            staged->recorderOperators.size()
        ),
        static_cast<std::uint32_t>(
            staged->eventOperators.size()
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
        0u,
        heightfieldTerrain
            ? MR_TASK_PROGRAM_TERRAIN
            : 0u,
    };
    staged->header.curriculum = {
        pack.curriculum.levelCount,
        pack.curriculum.evaluationWindowSteps,
        curriculumSignalIndex,
        static_cast<std::uint32_t>(
            staged->commandOperators.size()
        ),
    };
    if (pack.criticIncludesCleanHistory) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY;
    }
    if (floatingRoot) {
        staged->header.schedule.w |=
            MR_TASK_PROGRAM_FLOATING_ROOT;
    }
    staged->header.taskScalars = {
        pack.phase.periodSeconds,
        pack.curriculum.successThreshold,
        pack.curriculum.minimumEpisodeSurvivalFraction,
        pack.supportForceThreshold,
    };
    staged->header.commandSchedule = {
        pack.commands.zeroProbability,
        pack.commands.minimumDurationSeconds,
        pack.commands.maximumDurationSeconds,
        0.0f,
    };
    staged->header.eventSchedule = {
        0.0f,
        pack.events.minimumIntervalSeconds,
        pack.events.maximumIntervalSeconds,
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
    staged->sensorFingerprint = usesSensors
        ? sensors.fingerprint()
        : 0u;
    staged->header.typedCounts = {
        static_cast<std::uint32_t>(staged->frames.size()),
        static_cast<std::uint32_t>(staged->goals.size()),
        static_cast<std::uint32_t>(
            staged->sensorFingerprint
        ),
        static_cast<std::uint32_t>(
            staged->sensorFingerprint >> 32u
        ),
    };
    staged->header.graphCounts = {
        static_cast<std::uint32_t>(
            staged->signalOperators.size()
        ),
        static_cast<std::uint32_t>(
            staged->signalSources.size()
        ),
        staged->layout.spatialJacobianEnvironmentStride,
        staged->layout.signalSensorScratchCount,
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
        staged->recorderOperators.empty()
            ? 0u
            : appendArena(
                  std::span<const MRTaskRecorderOperatorGPU>{
                      staged->recorderOperators
                  }
              ),
        staged->eventOperators.empty()
            ? 0u
            : appendArena(
                  std::span<const MRTaskEventOperatorGPU>{
                      staged->eventOperators
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
    const std::uint32_t terrainResetOffset = appendArena(
        std::span<const mr_float4>{
            staged->terrainResetTranslations
        }
    );
    const std::uint32_t commandOperatorOffset = appendArena(
        std::span<const MRTaskCommandOperatorGPU>{
            staged->commandOperators
        }
    );
    const std::uint32_t frameOffset = appendArena(
        std::span<const MRTaskFrameGPU>{staged->frames}
    );
    const std::uint32_t kinematicFrameOffset = appendArena(
        std::span<const MRTaskKinematicFrameGPU>{
            staged->kinematicFrames
        }
    );
    const std::uint64_t expectedKinematicFrameOffset =
        static_cast<std::uint64_t>(frameOffset) +
        staged->frames.size() * sizeof(MRTaskFrameGPU);
    if (kinematicFrameOffset == MR_INVALID_INDEX ||
        expectedKinematicFrameOffset != kinematicFrameOffset) {
        return reject(
            TaskCompileStatus::arithmeticOverflow,
            "kinematic_frames",
            "compiled kinematic-frame table is not contiguous with task frames"
        );
    }
    const std::uint32_t goalOffset = appendArena(
        std::span<const MRTaskGoalGPU>{staged->goals}
    );
    staged->header.offsets3 = {
        terrainResetOffset,
        commandOperatorOffset,
        frameOffset,
        goalOffset,
    };
    staged->header.offsets4 = {
        appendArena(
            std::span<const MRTaskObservationOperatorGPU>{
                staged->signalSources
            }
        ),
        appendArena(
            std::span<const MRTaskSignalOperatorGPU>{
                staged->signalOperators
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
        staged->header.offsets4,
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
    for (const TaskCommandSpec& command : pack.commands.values) {
        hash.string(command.id);
    }
    for (const TaskEventSpec& event : pack.events.values) {
        hash.string(event.id);
        hash.string(event.target);
    }
    for (const TaskContactGroup& group : pack.contactGroups) {
        hash.string(group.id);
        hash.string(group.referenceBody);
        for (const std::string& body : group.bodies) {
            hash.string(body);
        }
    }
    for (const TaskFrameSpec& frame : pack.frames) {
        hash.string(frame.id);
        hash.string(frame.body);
        hash.string(frame.site);
    }
    for (const TaskGoalSpec& goal : pack.goals) {
        hash.string(goal.id);
    }
    for (const TaskSignalSpec& signal : pack.signals) {
        hash.string(signal.id);
        hash.string(signal.left);
        hash.string(signal.right);
        hash.string(signal.source.target);
        hash.string(signal.source.goal);
        hash.string(signal.source.reference);
        hash.string(signal.source.coordinate);
        for (const TaskObservationOperatorSpec& source :
             signal.reductionSources) {
            hash.string(source.target);
            hash.string(source.goal);
            hash.string(source.reference);
            hash.string(source.coordinate);
        }
    }
    for (const TaskObservationOperatorSpec& observation :
         pack.actorFrame) {
        hash.string(observation.target);
        hash.string(observation.goal);
        hash.string(observation.reference);
        hash.string(observation.coordinate);
    }
    for (const TaskObservationOperatorSpec& observation :
         pack.critic) {
        hash.string(observation.target);
        hash.string(observation.goal);
        hash.string(observation.reference);
        hash.string(observation.coordinate);
    }
    for (const TaskRewardOperatorSpec& reward : pack.rewards) {
        hash.string(reward.signal);
    }
    for (const TaskRecorderSpec& recorder : pack.recorders) {
        hash.string(recorder.id);
        hash.string(recorder.signal);
    }
    hash.string(pack.curriculum.successSignal);
    for (const TaskTerminationOperatorSpec& termination :
         pack.terminations) {
        hash.string(termination.signal);
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
