#pragma once

#include "metalrobo/InteractionPack.hpp"
#include "metalrobo/MetalWorldCapacity.hpp"
#include "metalrobo/task_program_types.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {

class CompiledWorld;

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
    gaitPhase = MR_TASK_OBSERVE_GAIT_PHASE,
    recoveryEvent = MR_TASK_OBSERVE_RECOVERY_EVENT,
    objectTrack = MR_TASK_OBSERVE_OBJECT_TRACK,
    maskedDepth = MR_TASK_OBSERVE_MASKED_DEPTH,
    supportSense = MR_TASK_OBSERVE_SUPPORT_SENSE,
    supportPatch = MR_TASK_OBSERVE_SUPPORT_PATCH,
    interactionJointPositionError =
        MR_TASK_OBSERVE_INTERACTION_JOINT_POSITION_ERROR,
    interactionContactMode =
        MR_TASK_OBSERVE_INTERACTION_CONTACT_MODE,
    interactionContactTarget =
        MR_TASK_OBSERVE_INTERACTION_CONTACT_TARGET,
    interactionPhase = MR_TASK_OBSERVE_INTERACTION_PHASE,
    interactionContactValidity =
        MR_TASK_OBSERVE_INTERACTION_CONTACT_VALIDITY,
    interactionRootTrackingError =
        MR_TASK_OBSERVE_INTERACTION_ROOT_TRACKING_ERROR,
};

enum class TaskRewardOperator : std::uint32_t {
    linearVelocityTracking =
        MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING,
    yawVelocityTracking = MR_TASK_REWARD_YAW_VELOCITY_TRACKING,
    constant = MR_TASK_REWARD_CONSTANT,
    rootVerticalVelocitySquared =
        MR_TASK_REWARD_ROOT_VERTICAL_VELOCITY_SQUARED,
    rootRollPitchVelocitySquared =
        MR_TASK_REWARD_ROOT_ROLL_PITCH_VELOCITY_SQUARED,
    tiltSquared = MR_TASK_REWARD_TILT_SQUARED,
    rootHeightErrorSquared =
        MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED,
    jointVelocitySquared = MR_TASK_REWARD_JOINT_VELOCITY_SQUARED,
    jointAccelerationSquared =
        MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED,
    actionRateSquared = MR_TASK_REWARD_ACTION_RATE_SQUARED,
    jointLimitViolationSquared =
        MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_SQUARED,
    mechanicalPower = MR_TASK_REWARD_MECHANICAL_POWER,
    jointGroupPostureSquared =
        MR_TASK_REWARD_JOINT_GROUP_POSTURE_SQUARED,
    gaitContactMatch = MR_TASK_REWARD_GAIT_CONTACT_MATCH,
    swingClearance = MR_TASK_REWARD_SWING_CLEARANCE,
    supportSlip = MR_TASK_REWARD_SUPPORT_SLIP,
    forbiddenContact = MR_TASK_REWARD_FORBIDDEN_CONTACT,
    jointGroupPostureAbsolute =
        MR_TASK_REWARD_JOINT_GROUP_POSTURE_ABSOLUTE,
    projectedGravityHorizontalSquared =
        MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED,
    footClearance = MR_TASK_REWARD_FOOT_CLEARANCE,
    jointLimitViolationAbsolute =
        MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE,
    rootHeightNormalized = MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED,
    rootHeightProgress = MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS,
    uprightness = MR_TASK_REWARD_UPRIGHTNESS,
    supportContactCount = MR_TASK_REWARD_SUPPORT_CONTACT_COUNT,
    bodyHeightExponential = MR_TASK_REWARD_BODY_HEIGHT_EXPONENTIAL,
    supportHeightExponential =
        MR_TASK_REWARD_SUPPORT_HEIGHT_EXPONENTIAL,
    bodyUpExponential = MR_TASK_REWARD_BODY_UP_EXPONENTIAL,
    standingCompletion = MR_TASK_REWARD_STANDING_COMPLETION,
    recoveryTiltProgress =
        MR_TASK_REWARD_RECOVERY_TILT_PROGRESS,
    recoveryCompletion = MR_TASK_REWARD_RECOVERY_COMPLETION,
    linkClearanceBarrier =
        MR_TASK_REWARD_LINK_CLEARANCE_BARRIER,
    projectileMiss = MR_TASK_REWARD_PROJECTILE_MISS,
    projectileEvasion = MR_TASK_REWARD_PROJECTILE_EVASION,
    projectileSafeStillness =
        MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS,
    projectileSafeActionRate =
        MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE,
    jointCbfCorrection = MR_TASK_REWARD_JOINT_CBF_CORRECTION,
    jointCbfBuffer = MR_TASK_REWARD_JOINT_CBF_BUFFER,
    projectilePredictedClearance =
        MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE,
    interactionJointTracking =
        MR_TASK_REWARD_INTERACTION_JOINT_TRACKING,
    interactionContactTracking =
        MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING,
    restoration = MR_TASK_REWARD_RESTORATION,
    interactionRootTracking =
        MR_TASK_REWARD_INTERACTION_ROOT_TRACKING,
    wholeBodyRecovery = MR_TASK_REWARD_WHOLE_BODY_RECOVERY,
    interactionRootLinearVelocityError =
        MR_TASK_REWARD_INTERACTION_ROOT_LINEAR_VELOCITY_ERROR,
    objectGrasp = MR_TASK_REWARD_OBJECT_GRASP,
    objectLift = MR_TASK_REWARD_OBJECT_LIFT,
    objectPosition = MR_TASK_REWARD_OBJECT_POSITION,
    objectPlacement = MR_TASK_REWARD_OBJECT_PLACEMENT,
};

enum class TaskTerminationOperator : std::uint32_t {
    minimumRootHeight = MR_TASK_TERMINATE_MINIMUM_ROOT_HEIGHT,
    maximumTilt = MR_TASK_TERMINATE_MAXIMUM_TILT,
    contactGroup = MR_TASK_TERMINATE_CONTACT_GROUP,
    projectileContact = MR_TASK_TERMINATE_PROJECTILE_CONTACT,
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
    rootHeight = MR_TASK_RANDOMIZE_ROOT_HEIGHT,
    rootOrientation = MR_TASK_RANDOMIZE_ROOT_ORIENTATION,
    jointPosition = MR_TASK_RANDOMIZE_JOINT_POSITION,
    sceneBodyPosition = MR_TASK_RANDOMIZE_SCENE_BODY_POSITION,
    sceneBodyVelocity = MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY,
    sceneBodyLaunchStep = MR_TASK_RANDOMIZE_SCENE_BODY_LAUNCH_STEP,
    sceneBodyEventImpact =
        MR_TASK_RANDOMIZE_SCENE_BODY_EVENT_IMPACT,
};

struct TaskActionBinding {
    std::string joint;
    float scale = 0.25f;
    // First-order target-filter time constant in seconds. Zero disables
    // filtering, so the physical response is independent of control rate.
    float responseTimeSeconds = 0.0f;
};

// Stable sources already produced by the native task transaction. TaskPack
// authors select which measurements are meaningful for their task; runtime
// construction never infers semantics from observation or reward operators.
enum class TaskOutcomeSource : std::uint32_t {
    trackingScore = 2u,
    rootHeight = 3u,
    tilt = 4u,
    contactReward = 8u,
    // The compiler assigns one of eight generic native outcome channels and
    // accumulates every matching reward-operator contribution into it.
    rewardContribution = 9u,
};

enum class TaskOutcomeDirection : std::uint32_t {
    neutral = 0u,
    higherIsBetter = 1u,
    lowerIsBetter = 2u,
};

struct TaskOutcomeSpec {
    std::string id;
    std::string unit;
    TaskOutcomeSource source = TaskOutcomeSource::trackingScore;
    TaskOutcomeDirection direction = TaskOutcomeDirection::neutral;
    TaskRewardOperator rewardOperation = TaskRewardOperator::constant;
};

struct CompiledTaskOutcomeSpec {
    std::string id;
    std::string unit;
    // MRTaskOutcomeSourceC numeric value. Reward contributions resolve to
    // generic channels 0...7 during compilation.
    std::uint32_t source = 0u;
    TaskOutcomeDirection direction = TaskOutcomeDirection::neutral;
};

struct TaskObservationOperatorSpec {
    TaskObservationSource source =
        TaskObservationSource::rootAngularVelocityLocal;
    // Joint, contact-group, body, or tracked scene-body identity when the
    // source requires one.
    std::string target;
    std::uint32_t component = 0u;
    float scale = 1.0f;
    float offset = 0.0f;
    float noiseAmplitude = 0.0f;
    float biasLower = 0.0f;
    float biasUpper = 0.0f;
    bool normalizeVector3 = false;
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
    // Optional spatial support field in the reference body's link frame,
    // relative to localReference. A zero width and height disable it.
    mr_float4 supportPatchBounds{};
    std::uint32_t supportPatchWidth = 0u;
    std::uint32_t supportPatchHeight = 0u;
};

struct TaskJointGroup {
    std::string id;
    std::vector<std::string> joints;
};

struct TaskRewardOperatorSpec {
    TaskRewardOperator operation =
        TaskRewardOperator::constant;
    std::string sourceGroup;
    // Dynamic scene-body identity for projectile-relative and rigid-object
    // manipulation operators.
    std::string target;
    // Reward rate in units per second. The native task integrates every
    // weighted term over the control interval, keeping TaskPacks invariant
    // when the control frequency changes.
    float weight = 0.0f;
    mr_float4 parameters{};
};

struct TaskTerminationOperatorSpec {
    TaskTerminationOperator operation =
        TaskTerminationOperator::minimumRootHeight;
    std::string sourceGroup;
    std::uint32_t reason = MR_TASK_TERMINATION_HEIGHT;
    std::uint32_t priority = 0u;
    float threshold = 0.0f;
    // One-shot reward applied when this non-timeout termination wins the
    // priority reduction. This closes the early-termination loophole when
    // the task also contains per-step penalties.
    float failurePenalty = 0.0f;
    // Inclusive reset-band applicability. The default preserves the former
    // all-band behavior. Band-local reset boundaries let one TaskPack train
    // overlapping recovery skills without treating a valid floor pose as a
    // failure merely because standing practice should restart after a fall.
    std::uint32_t minimumDifficultyBand = 0u;
    std::uint32_t maximumDifficultyBand = MR_INVALID_INDEX;
};

struct TaskRandomizationOperatorSpec {
    TaskRandomizationOperator operation =
        TaskRandomizationOperator::rootPosition;
    std::string target;
    std::uint32_t component = 0u;
    // The operator is inactive below this sampled difficulty band.
    std::uint32_t minimumDifficultyBand = 0u;
    mr_float4 parameters{};
};

struct TaskCommandProgram {
    // Initial range, hard range limits, and per-difficulty-band expansion.
    mr_float4 lower{};
    mr_float4 upper{};
    mr_float4 limitLower{};
    mr_float4 limitUpper{};
    mr_float4 difficultyStep{};
    float standingProbability = 0.0f;
    // Shapes deterministic episode sampling over every authored band. One is
    // uniform; values above one retain more easy episodes without excluding
    // difficult evidence.
    float difficultySamplingExponent = 2.0f;
    float minimumDurationSeconds = 5.0f;
    float maximumDurationSeconds = 10.0f;
};

struct TaskPushProgram {
    float maximumVelocity = 0.0f;
    float minimumIntervalSeconds = 2.0f;
    float maximumIntervalSeconds = 5.0f;
    // Fraction of episodes that hold every projectile as a standing anchor.
    float projectileStandingProbability = 0.0f;
    // Optional event-projectile ballistics. A positive speed range retargets
    // each launch at a sampled point around the live root while preserving
    // ordinary rigid-body flight under the compiled world's gravity.
    float projectileTargetHorizontalRadius = 0.0f;
    float projectileHorizontalSpeedLower = 0.0f;
    float projectileHorizontalSpeedUpper = 0.0f;
    float projectileTargetHeightLower = 0.0f;
    float projectileTargetHeightUpper = 0.0f;
};

struct TaskTerrainProgram {
    std::string body;
    std::vector<mr_float4> sampleOffsets;
    std::vector<mr_float4> resetTranslations;
};

struct TaskVisualProgram {
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::vector<std::uint32_t> frameOffsets;
    float nearDepthMeters = 0.1f;
    float farDepthMeters = 5.0f;
    float fullDropoutProbability = 0.0f;
    float pixelDropoutProbability = 0.0f;
    float depthJitterMeters = 0.0f;
    float depthNoiseSigmaMeters = 0.0f;
    float edgeFlickerProbability = 0.0f;
    // Additional corruption applied at the hardest difficulty band. Zero
    // preserves the authored observation distribution at every level; one
    // doubles dropout, jitter, noise, and edge flicker at the final level.
    float difficultyCorruptionGain = 0.0f;
    // Append confidence, bearing, elevation, nearness, and apparent area for
    // each sparse frame plus four temporal changes. Every value is reduced
    // from the corrupted masked-depth plane; this never exposes scene truth.
    bool includeDerivedFeatures = false;
};

// Privileged training-time threat analysis. The compiler resolves the
// protected semantic group; active projectile identity comes from the
// ordinary event sequence. Deployment actors never receive these values.
struct TaskThreatProgram {
    std::string protectedGroup;
    float activationSpeed = 0.5f;
    float horizonSeconds = 2.0f;
    float safetyMargin = 0.05f;
    float cbfAlpha = 2.0f;
    // Root-relative strike-height boundaries for the four generic evasion
    // classes: step-over, sidestep, lean, and duck.
    float stepOverMaximumHeight = 0.35f;
    float sidestepMaximumHeight = 0.75f;
    float leanMaximumHeight = 1.10f;
    // Joint-CBF urgency and desired-velocity construction.
    float urgencySeconds = 0.35f;
    float desiredVelocityHorizonSeconds = 0.20f;
    float projectionEpsilon = 1.0e-5f;
};

// Training-only tracked-link pose stream consumed by the motion prior. Each
// frame stores anchor-relative position and 6D orientation per body. It is a
// compact learner output, never an actor observation.
struct TaskMotionProgram {
    std::string anchorBody;
    std::vector<std::string> trackedBodies;
};

// Authored, robot-independent task artifact. Names are resolved only by
// compileTaskProgram; the runtime consumes no strings.
struct TaskPack {
    std::string id;
    // Operational contact-graph budget for this task. Zero fields retain the
    // world's topology-derived envelope.
    MetalWorldCapacityProfile capacities;
    std::vector<TaskActionBinding> actions;
    // Universal transaction outcomes are supplied by the compiler. This
    // table contains only task-authored physical/competence measurements.
    std::vector<TaskOutcomeSpec> outcomes;
    // Temporal proprioception retained in the actor history. Current task
    // intent belongs in actorCurrent so it is appended only once.
    std::vector<TaskObservationOperatorSpec> actorFrame;
    std::uint32_t actorHistoryLength = 1u;
    std::vector<TaskObservationOperatorSpec> actorCurrent;
    std::vector<TaskObservationOperatorSpec> critic;
    std::uint32_t criticHistoryLength = 1u;
    bool criticIncludesCleanHistory = true;
    std::vector<TaskContactGroup> contactGroups;
    std::vector<TaskJointGroup> jointGroups;
    std::vector<TaskRewardOperatorSpec> rewards;
    std::vector<TaskTerminationOperatorSpec> terminations;
    std::vector<TaskRandomizationOperatorSpec> randomization;
    TaskCommandProgram commands;
    TaskPushProgram pushes;
    TaskTerrainProgram terrain;
    TaskVisualProgram visual;
    TaskThreatProgram threat;
    TaskMotionProgram motion;
    std::uint32_t maximumEpisodeSteps = 1000u;
    std::uint32_t maximumActionDelaySteps = 0u;
    std::uint32_t maximumObservationDelaySteps = 0u;
    // All bands remain eligible on every episode; this is an authored
    // difficulty discretization, not an advancement ladder.
    std::uint32_t difficultyBandCount = 1u;
    // Fraction of the student's normalized balance/control residual added to
    // the InteractionPack mechanism-space motion reference.
    // Zero runs the student in shadow mode while retaining executed targets
    // as distillation labels; one grants the full residual action range.
    float interactionStudentAuthority = 0.1f;
    // Interaction reference-state curriculum strength. A value f samples
    // later phases for fraction f of resets, over [0, f] of the clip, while
    // the remaining 1-f resets retain the canonical frame-zero trajectory.
    // The solver remains authoritative for every subsequent transition.
    float interactionResetPhaseFraction = 0.0f;
    // Negative values preserve the legacy coupled fraction above. Runtime
    // configuration may independently choose how often a reset is sampled
    // away from frame zero and how far through the clip it may begin.
    float interactionResetPhaseProbability = -1.0f;
    float interactionResetMaximumPhase = -1.0f;
    // When false, an InteractionPack still supplies the initial physical
    // state but no longer supplies runtime joint targets or teacher actions.
    bool interactionControlReference = true;
    float baseHeightTarget = 0.0f;
    float gaitPeriodSeconds = 0.8f;
    float clearanceTarget = 0.1f;
    // Episode-mean linear-velocity tracking used as a physical outcome.
    // Angular tracking is an independent reward/metric.
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
    std::uint32_t motionFeatureCount = 0u;
    std::uint32_t interactionFrameCount = 0u;
    std::uint32_t interactionContactCount = 0u;
};

class CompiledTaskProgram {
public:
    CompiledTaskProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t worldFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t observationFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t actionFingerprint() const noexcept;
    [[nodiscard]] const TaskProgramLayout& layout() const noexcept;
    [[nodiscard]] const MRTaskProgramHeaderGPU& header() const noexcept;
    [[nodiscard]] std::span<const MRTaskActionBindingGPU>
    actionBindings() const noexcept;
    [[nodiscard]] std::span<const CompiledTaskOutcomeSpec>
    outcomes() const noexcept;
    [[nodiscard]] std::span<const MRTaskObservationOperatorGPU>
    actorOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskObservationOperatorGPU>
    criticOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskContactGroupGPU>
    contactGroups() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    contactMembers() const noexcept;
    [[nodiscard]] std::span<const float>
    contactMemberRadii() const noexcept;
    [[nodiscard]] std::span<const MRTaskIndexGroupGPU>
    jointGroups() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    jointMembers() const noexcept;
    [[nodiscard]] std::span<const MRTaskRewardOperatorGPU>
    rewardOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskTerminationOperatorGPU>
    terminationOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskRandomizationOperatorGPU>
    randomizationOperators() const noexcept;
    [[nodiscard]] std::span<const MRTaskImpactEventGPU>
    impactEvents() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    motionBodies() const noexcept;
    [[nodiscard]] std::span<const float>
    interactionRootTargets() const noexcept;
    [[nodiscard]] std::span<const float>
    interactionJointTargets() const noexcept;
    [[nodiscard]] std::span<const MRTaskInteractionContactGPU>
    interactionContacts() const noexcept;
    [[nodiscard]] std::span<const MRTaskInteractionSampleGPU>
    interactionSamples() const noexcept;
    [[nodiscard]] std::span<const float>
    interactionContactTargets() const noexcept;
    [[nodiscard]] std::span<const float>
    interactionContactTolerances() const noexcept;
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
        CompiledTaskProgram&
    );
    friend TaskCompileDiagnostics compileTaskProgram(
        const TaskPack&,
        const InteractionPack&,
        std::string_view,
        const CompiledWorld&,
        CompiledTaskProgram&
    );
};

[[nodiscard]] TaskCompileDiagnostics compileTaskProgram(
    const TaskPack& pack,
    const CompiledWorld& world,
    CompiledTaskProgram& output
);

// Selects one generated reference transactionally. Named joints are
// retargeted into TaskPack action order, the controller blends the reference
// with absolute student targets at the authored authority, and contact
// targets remain expected values compared against solver-resolved physical
// outcomes.
[[nodiscard]] TaskCompileDiagnostics compileTaskProgram(
    const TaskPack& pack,
    const InteractionPack& interactions,
    std::string_view clipId,
    const CompiledWorld& world,
    CompiledTaskProgram& output
);

[[nodiscard]] const char* taskCompileStatusName(
    TaskCompileStatus status
) noexcept;

} // namespace metalrobo
