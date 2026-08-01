#pragma once

#include "metalrobo/MetalWorldCapacity.hpp"
#include "metalrobo/task_program_types.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

class CompiledWorld;
class CompiledSensorProgram;

enum class TaskObservationSource : std::uint32_t {
    rootAngularVelocityLocal =
        MR_TASK_OBSERVE_ROOT_ANGULAR_VELOCITY_LOCAL,
    projectedGravity = MR_TASK_OBSERVE_PROJECTED_GRAVITY,
    command = MR_TASK_OBSERVE_COMMAND,
    jointPositionError = MR_TASK_OBSERVE_JOINT_POSITION_ERROR,
    jointVelocity = MR_TASK_OBSERVE_JOINT_VELOCITY,
    previousAction = MR_TASK_OBSERVE_PREVIOUS_ACTION,
    rootLinearVelocityLocal =
        MR_TASK_OBSERVE_ROOT_LINEAR_VELOCITY_LOCAL,
    rootHeight = MR_TASK_OBSERVE_ROOT_HEIGHT,
    contactMetric = MR_TASK_OBSERVE_CONTACT_METRIC,
    terrainHeight = MR_TASK_OBSERVE_TERRAIN_HEIGHT,
    bodyParameterMean = MR_TASK_OBSERVE_BODY_PARAMETER_MEAN,
    bodyParameter = MR_TASK_OBSERVE_BODY_PARAMETER,
    controllerParameter = MR_TASK_OBSERVE_CONTROLLER_PARAMETER,
    contactWrenchLocal =
        MR_TASK_OBSERVE_CONTACT_WRENCH_LOCAL,
    framePositionWorld = MR_TASK_OBSERVE_FRAME_POSITION_WORLD,
    frameOrientationWorld =
        MR_TASK_OBSERVE_FRAME_ORIENTATION_WORLD,
    frameGoalPositionError =
        MR_TASK_OBSERVE_FRAME_GOAL_POSITION_ERROR,
    frameGoalOrientationError =
        MR_TASK_OBSERVE_FRAME_GOAL_ORIENTATION_ERROR,
    sensorValue = MR_TASK_OBSERVE_SENSOR_VALUE,
    sensorValidity = MR_TASK_OBSERVE_SENSOR_VALIDITY,
    frameRelativePosition =
        MR_TASK_OBSERVE_FRAME_RELATIVE_POSITION,
    frameRelativeOrientation =
        MR_TASK_OBSERVE_FRAME_RELATIVE_ORIENTATION,
    frameLinearVelocityWorld =
        MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_WORLD,
    frameAngularVelocityWorld =
        MR_TASK_OBSERVE_FRAME_ANGULAR_VELOCITY_WORLD,
    frameRelativeLinearVelocity =
        MR_TASK_OBSERVE_FRAME_RELATIVE_LINEAR_VELOCITY,
    frameRelativeAngularVelocity =
        MR_TASK_OBSERVE_FRAME_RELATIVE_ANGULAR_VELOCITY,
    frameLinearJacobianWorld =
        MR_TASK_OBSERVE_FRAME_LINEAR_JACOBIAN_WORLD,
    frameAngularJacobianWorld =
        MR_TASK_OBSERVE_FRAME_ANGULAR_JACOBIAN_WORLD,
};

enum class TaskRewardOperator : std::uint32_t {
    linearVelocityTracking =
        MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING,
    yawVelocityTracking = MR_TASK_REWARD_YAW_VELOCITY_TRACKING,
    jointAccelerationSquared =
        MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED,
    actionRateSquared = MR_TASK_REWARD_ACTION_RATE_SQUARED,
    mechanicalPower = MR_TASK_REWARD_MECHANICAL_POWER,
    gaitContactMatch = MR_TASK_REWARD_GAIT_CONTACT_MATCH,
    footClearance = MR_TASK_REWARD_FOOT_CLEARANCE,
    jointLimitViolationAbsolute =
        MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE,
    signal = MR_TASK_REWARD_SIGNAL,
};

enum class TaskRewardChannel : std::uint32_t {
    primary = MR_TASK_REWARD_CHANNEL_PRIMARY,
    stability = MR_TASK_REWARD_CHANNEL_STABILITY,
    velocity = MR_TASK_REWARD_CHANNEL_VELOCITY,
    acceleration = MR_TASK_REWARD_CHANNEL_ACCELERATION,
    control = MR_TASK_REWARD_CHANNEL_CONTROL,
    configuration = MR_TASK_REWARD_CHANNEL_CONFIGURATION,
    energy = MR_TASK_REWARD_CHANNEL_ENERGY,
    contact = MR_TASK_REWARD_CHANNEL_CONTACT,
};

enum class TaskTerminationOperator : std::uint32_t {
    signalBelow = MR_TASK_TERMINATE_SIGNAL_BELOW,
    signalAbove = MR_TASK_TERMINATE_SIGNAL_ABOVE,
    signalOutside = MR_TASK_TERMINATE_SIGNAL_OUTSIDE,
};

enum class TaskSignalOperator : std::uint32_t {
    source = MR_TASK_SIGNAL_SOURCE,
    constant = MR_TASK_SIGNAL_CONSTANT,
    add = MR_TASK_SIGNAL_ADD,
    subtract = MR_TASK_SIGNAL_SUBTRACT,
    multiply = MR_TASK_SIGNAL_MULTIPLY,
    minimum = MR_TASK_SIGNAL_MINIMUM,
    maximum = MR_TASK_SIGNAL_MAXIMUM,
    absolute = MR_TASK_SIGNAL_ABSOLUTE,
    square = MR_TASK_SIGNAL_SQUARE,
    squareRoot = MR_TASK_SIGNAL_SQUARE_ROOT,
    safeDivide = MR_TASK_SIGNAL_SAFE_DIVIDE,
    clamp = MR_TASK_SIGNAL_CLAMP,
    exponentialTracking = MR_TASK_SIGNAL_EXPONENTIAL_TRACKING,
    insideBounds = MR_TASK_SIGNAL_INSIDE_BOUNDS,
    exponentialDecay = MR_TASK_SIGNAL_EXPONENTIAL_DECAY,
    atan2 = MR_TASK_SIGNAL_ATAN2,
};

enum class TaskRandomizationOperator : std::uint32_t {
    rootPosition = MR_TASK_RANDOMIZE_ROOT_POSITION,
    rootYaw = MR_TASK_RANDOMIZE_ROOT_YAW,
    actionPosition = MR_TASK_RANDOMIZE_ACTION_POSITION,
    velocity = MR_TASK_RANDOMIZE_VELOCITY,
    bodyParameter = MR_TASK_RANDOMIZE_BODY_PARAMETER,
    bodyPayload = MR_TASK_RANDOMIZE_BODY_PAYLOAD,
    controllerParameter =
        MR_TASK_RANDOMIZE_CONTROLLER_PARAMETER,
    actionDelay = MR_TASK_RANDOMIZE_ACTION_DELAY,
    observationDelay = MR_TASK_RANDOMIZE_OBSERVATION_DELAY,
    actionVelocity = MR_TASK_RANDOMIZE_ACTION_VELOCITY,
};

struct TaskActionBinding {
    std::string joint;
    float scale = 0.25f;
};

struct TaskObservationOperatorSpec {
    TaskObservationSource source =
        TaskObservationSource::rootAngularVelocityLocal;
    // Joint, contact-group, or body identity when the source requires one.
    std::string target;
    // Required only by frame-to-goal operators.
    std::string goal;
    // Required only by frame-to-frame operators.
    std::string reference;
    // Required only by frame-Jacobian operators. This is a semantic DoF
    // identity, resolved to one global generalized-velocity coordinate.
    std::string coordinate;
    std::uint32_t component = 0u;
    float scale = 1.0f;
    float offset = 0.0f;
    float noiseAmplitude = 0.0f;
    float biasLower = 0.0f;
    float biasUpper = 0.0f;
    bool normalizeVector3 = false;
};

// A named task frame is authored either in one body link frame or relative to
// one imported/authored model site. Exactly one source is required. Site
// transforms are composed at compilation, then translation is converted to
// the body's centre-of-mass frame; Metal receives only MRTaskFrameGPU records.
struct TaskFrameSpec {
    std::string id;
    std::string body;
    std::string site;
    mr_float4 localPosition{};
    mr_float4 localOrientation{0.0f, 0.0f, 0.0f, 1.0f};
};

enum class TaskGoalMode : std::uint32_t {
    fixed = MR_TASK_GOAL_FIXED,
    sampledEpisode = MR_TASK_GOAL_SAMPLED_EPISODE,
    trajectory = MR_TASK_GOAL_TRAJECTORY,
};

enum class TaskGoalPlayback : std::uint32_t {
    clamp = MR_TASK_GOAL_PLAYBACK_CLAMP,
    loop = MR_TASK_GOAL_PLAYBACK_LOOP,
    pingPong = MR_TASK_GOAL_PLAYBACK_PING_PONG,
};

struct TaskGoalSpec {
    std::string id;
    TaskGoalMode mode = TaskGoalMode::fixed;
    TaskGoalPlayback playback = TaskGoalPlayback::clamp;
    // Fixed pose, sampled-pose centre, or trajectory start pose.
    mr_float4 position{};
    mr_float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
    // Used only by trajectory mode.
    mr_float4 targetPosition{};
    mr_float4 targetOrientation{0.0f, 0.0f, 0.0f, 1.0f};
    // Used only by sampledEpisode. Position offsets are world-aligned;
    // rotation vectors are applied in the base goal's local frame.
    mr_float4 positionOffsetLower{};
    mr_float4 positionOffsetUpper{};
    mr_float4 rotationVectorLower{};
    mr_float4 rotationVectorUpper{};
    // Trajectory time is accepted episodeStep * controlPeriod + phase.
    float durationSeconds = 0.0f;
    float phaseSeconds = 0.0f;
};

struct TaskContactGroup {
    std::string id;
    std::vector<std::string> bodies;
    bool support = false;
    bool forbidden = false;
    std::string referenceBody;
    // Reference point in the body's COM-centred local frame; w is the
    // clearance radius subtracted from sampled terrain height.
    mr_float4 localReference{};
    float gaitPhaseOffsetRadians = 0.0f;
    float stanceFraction = 0.5f;
};

// One node in a topologically ordered scalar TaskIR graph. Operand names may
// reference only earlier nodes. Source nodes compile one truth-only semantic
// observation; other nodes must leave source at its default value.
struct TaskSignalSpec {
    std::string id;
    TaskSignalOperator operation = TaskSignalOperator::constant;
    TaskObservationOperatorSpec source;
    std::string left;
    std::string right;
    mr_float4 parameters{};
};

struct TaskRewardOperatorSpec {
    TaskRewardOperator operation =
        TaskRewardOperator::signal;
    std::string sourceGroup;
    // Required only by the generic SignalIR reward operator.
    std::string signal;
    TaskRewardChannel channel = TaskRewardChannel::primary;
    // Reward rate in units per second. The native task integrates every
    // weighted term over the control interval, keeping TaskPacks invariant
    // when the control frequency changes.
    float weight = 0.0f;
    mr_float4 parameters{};
};

struct TaskTerminationOperatorSpec {
    TaskTerminationOperator operation =
        TaskTerminationOperator::signalBelow;
    std::string sourceGroup;
    // Required only by SignalIR threshold operators.
    std::string signal;
    std::uint32_t reason = MR_TASK_TERMINATION_HEIGHT;
    std::uint32_t priority = 0u;
    float threshold = 0.0f;
    // Upper bound used only by signalOutside.
    float upperThreshold = 0.0f;
    // One-shot reward applied when this non-timeout termination wins the
    // priority reduction. This closes the early-termination loophole when
    // the task also contains per-step penalties.
    float failurePenalty = 0.0f;
};

struct TaskRandomizationOperatorSpec {
    TaskRandomizationOperator operation =
        TaskRandomizationOperator::rootPosition;
    std::string target;
    std::uint32_t component = 0u;
    // The operator is inactive below this global task-curriculum level.
    std::uint32_t minimumCurriculumLevel = 0u;
    mr_float4 parameters{};
};

struct TaskCommandProgram {
    // Initial range, hard range limits, and per-curriculum-level expansion.
    mr_float4 lower{};
    mr_float4 upper{};
    mr_float4 limitLower{};
    mr_float4 limitUpper{};
    mr_float4 curriculumStep{};
    float standingProbability = 0.0f;
    // Fraction of completed, non-physics episodes that must reach the time
    // limit during a curriculum window before command difficulty advances.
    float minimumEpisodeSurvivalFraction = 0.0f;
    float minimumDurationSeconds = 5.0f;
    float maximumDurationSeconds = 10.0f;
};

struct TaskPushProgram {
    float maximumVelocity = 0.0f;
    float minimumIntervalSeconds = 2.0f;
    float maximumIntervalSeconds = 5.0f;
};

struct TaskTerrainProgram {
    std::string body;
    std::vector<mr_float4> sampleOffsets;
    std::vector<mr_float4> resetTranslations;
};

// Authored, robot-independent task artifact. Names are resolved only by
// compileTaskProgram; the runtime consumes no strings.
struct TaskPack {
    std::string id;
    // Operational contact-graph budget for this task. Zero fields retain the
    // world's topology-derived envelope.
    MetalWorldCapacityProfile capacities;
    std::vector<TaskActionBinding> actions;
    std::vector<TaskObservationOperatorSpec> actorFrame;
    std::uint32_t actorHistoryLength = 1u;
    std::vector<TaskObservationOperatorSpec> critic;
    std::uint32_t criticHistoryLength = 1u;
    bool criticIncludesCleanHistory = true;
    std::vector<TaskContactGroup> contactGroups;
    std::vector<TaskFrameSpec> frames;
    std::vector<TaskGoalSpec> goals;
    std::vector<TaskSignalSpec> signals;
    std::vector<TaskRewardOperatorSpec> rewards;
    std::vector<TaskTerminationOperatorSpec> terminations;
    std::vector<TaskRandomizationOperatorSpec> randomization;
    TaskCommandProgram commands;
    TaskPushProgram pushes;
    TaskTerrainProgram terrain;
    std::uint32_t maximumEpisodeSteps = 1000u;
    std::uint32_t maximumActionDelaySteps = 0u;
    std::uint32_t maximumObservationDelaySteps = 0u;
    std::uint32_t curriculumLevelCount = 1u;
    float baseHeightTarget = 0.0f;
    float gaitPeriodSeconds = 0.8f;
    float clearanceTarget = 0.1f;
    // Episode-mean linear-velocity tracking required to advance the command
    // curriculum. Angular tracking is an independent reward/metric.
    float successTrackingThreshold = 0.8f;
    float supportForceThreshold = 1.0f;
};

enum class TaskCompileStatus : std::uint32_t {
    success = 0u,
    invalidWorld,
    invalidPack,
    unresolvedSemantic,
    ambiguousSemantic,
    unsupportedOperator,
    arithmeticOverflow,
    internalFailure,
};

struct TaskCompileDiagnostics {
    TaskCompileStatus status = TaskCompileStatus::success;
    std::uint64_t fingerprint = 0u;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == TaskCompileStatus::success;
    }
};

struct TaskProgramLayout {
    std::uint32_t actionCount = 0u;
    std::uint32_t actorFrameSize = 0u;
    std::uint32_t actorHistoryLength = 0u;
    std::uint32_t actorObservationSize = 0u;
    std::uint32_t criticFrameSize = 0u;
    std::uint32_t criticHistoryLength = 0u;
    std::uint32_t criticObservationSize = 0u;
    std::uint32_t contactMetricCount = 0u;
    std::uint32_t biasCount = 0u;
    std::uint32_t delayStateCount = 0u;
    std::uint32_t kinematicPointQueryCount = 0u;
    std::uint32_t spatialJacobianEnvironmentStride = 0u;
    std::uint32_t signalCount = 0u;
};

// Private execution metadata for one articulation cohort of semantic point
// queries. Counts and offsets are elements, never bytes. The query packet is
// immutable and broadcast across environments; result buffers are owner-major.
struct TaskKinematicCohort {
    std::uint32_t articulationIndex = 0u;
    std::uint32_t queryOffset = 0u;
    std::uint32_t queryCount = 0u;
    std::uint32_t pointPrefix = 0u;
    std::uint32_t jacobianPrefix = 0u;
    std::uint32_t jacobianEnvironmentStride = 0u;
};

class CompiledTaskProgram {
public:
    CompiledTaskProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t worldFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t sensorFingerprint() const noexcept;
    [[nodiscard]] const TaskProgramLayout& layout() const noexcept;
    [[nodiscard]] const MRTaskProgramHeaderGPU& header() const noexcept;
    [[nodiscard]] std::span<const MRTaskActionBindingGPU>
    actionBindings() const noexcept;
    [[nodiscard]] std::span<const MRTaskObservationOperatorGPU>
    actorOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskObservationOperatorGPU>
    criticOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskObservationOperatorGPU>
    signalSources() const noexcept;
    [[nodiscard]] std::span<const MRTaskSignalOperatorGPU>
    signalOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskContactGroupGPU>
    contactGroups() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    contactMembers() const noexcept;
    [[nodiscard]] std::span<const MRTaskFrameGPU>
    frames() const noexcept;
    [[nodiscard]] std::span<const MRTaskGoalGPU>
    goals() const noexcept;
    [[nodiscard]] std::span<const MRArticulatedPointImpulseGPU>
    kinematicPointQueries() const noexcept;
    [[nodiscard]] std::span<const TaskKinematicCohort>
    kinematicCohorts() const noexcept;
    [[nodiscard]] std::span<const MRTaskRewardOperatorGPU>
    rewardOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskTerminationOperatorGPU>
    terminationOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskRandomizationOperatorGPU>
    randomizationOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskBiasSpecGPU>
    biasSpecs() const noexcept;
    [[nodiscard]] std::span<const mr_float4>
    terrainSampleOffsets() const noexcept;
    [[nodiscard]] std::span<const mr_float4>
    terrainResetTranslations() const noexcept;
    [[nodiscard]] std::span<const std::byte>
    arena() const noexcept;

private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;

    friend TaskCompileDiagnostics compileTaskProgram(
        const TaskPack&,
        const CompiledWorld&,
        const CompiledSensorProgram&,
        CompiledTaskProgram&
    );
};

[[nodiscard]] TaskCompileDiagnostics compileTaskProgram(
    const TaskPack& pack,
    const CompiledWorld& world,
    const CompiledSensorProgram& sensors,
    CompiledTaskProgram& output
);

[[nodiscard]] const char* taskCompileStatusName(
    TaskCompileStatus status
) noexcept;

} // namespace metalrobo
