#pragma once

#include "metalrobo/engine_types.h"

#define MR_TASK_PROGRAM_ABI_VERSION 22u
#define MR_TASK_TRANSITION_METRIC_COUNT 3u

#define MR_TASK_GOAL_FIXED 0u
#define MR_TASK_GOAL_SAMPLED_EPISODE 1u
#define MR_TASK_GOAL_TRAJECTORY 2u

#define MR_TASK_GOAL_PLAYBACK_CLAMP 0u
#define MR_TASK_GOAL_PLAYBACK_LOOP 1u
#define MR_TASK_GOAL_PLAYBACK_PING_PONG 2u

enum MRTaskProgramFlags : mr_u32 {
    MR_TASK_PROGRAM_TERRAIN = 1u << 0u,
    MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY = 1u << 1u,
    MR_TASK_PROGRAM_FLOATING_ROOT = 1u << 2u,
};

enum MRTaskObservationOpcode : mr_u32 {
    MR_TASK_OBSERVE_ROOT_ANGULAR_VELOCITY_LOCAL = 0u,
    MR_TASK_OBSERVE_PROJECTED_GRAVITY = 1u,
    MR_TASK_OBSERVE_COMMAND = 2u,
    MR_TASK_OBSERVE_JOINT_POSITION_ERROR = 3u,
    MR_TASK_OBSERVE_JOINT_VELOCITY = 4u,
    MR_TASK_OBSERVE_PREVIOUS_ACTION = 5u,
    MR_TASK_OBSERVE_ROOT_LINEAR_VELOCITY_LOCAL = 6u,
    MR_TASK_OBSERVE_ROOT_HEIGHT = 7u,
    MR_TASK_OBSERVE_CONTACT_METRIC = 8u,
    MR_TASK_OBSERVE_TERRAIN_HEIGHT = 9u,
    MR_TASK_OBSERVE_BODY_PARAMETER_MEAN = 10u,
    MR_TASK_OBSERVE_BODY_PARAMETER = 11u,
    MR_TASK_OBSERVE_CONTROLLER_PARAMETER = 12u,
    // Six-axis resultant contact wrench in the contact group's reference-body
    // frame: force xyz in newtons, torque xyz in newton-metres.
    MR_TASK_OBSERVE_CONTACT_WRENCH_LOCAL = 13u,
    MR_TASK_OBSERVE_FRAME_POSITION_WORLD = 14u,
    MR_TASK_OBSERVE_FRAME_ORIENTATION_WORLD = 15u,
    MR_TASK_OBSERVE_FRAME_GOAL_POSITION_ERROR = 16u,
    MR_TASK_OBSERVE_FRAME_GOAL_ORIENTATION_ERROR = 17u,
    // Named SensorIR scalar output. source.y is the compiled sensor index,
    // source.z is the channel, and auxiliary.z is the environment-local
    // output offset. The TaskIR fingerprint binds the exact SensorIR program.
    MR_TASK_OBSERVE_SENSOR_VALUE = 18u,
    // One scalar validity bit selected by source.z from
    // MRSensorSampleValidityFlags (valid, fresh, reset, stale, nonfinite).
    MR_TASK_OBSERVE_SENSOR_VALIDITY = 19u,
    // Target-frame position expressed in a second named frame. source.y is
    // the target frame and auxiliary.z is the reference frame.
    MR_TASK_OBSERVE_FRAME_RELATIVE_POSITION = 20u,
    // Tangent rotation vector of the target frame expressed relative to the
    // second named frame.
    MR_TASK_OBSERVE_FRAME_RELATIVE_ORIENTATION = 21u,
    // Linear velocity at the named frame origin in world axes.
    MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_WORLD = 22u,
    // Angular velocity of the named frame in world axes.
    MR_TASK_OBSERVE_FRAME_ANGULAR_VELOCITY_WORLD = 23u,
    // Time derivative of target-frame position in reference-frame axes.
    MR_TASK_OBSERVE_FRAME_RELATIVE_LINEAR_VELOCITY = 24u,
    // Target angular velocity relative to the reference, in reference axes.
    MR_TASK_OBSERVE_FRAME_RELATIVE_ANGULAR_VELOCITY = 25u,
    // One world-axis row of the linear Jacobian at a named frame origin,
    // with respect to one compiled generalized-velocity coordinate.
    MR_TASK_OBSERVE_FRAME_LINEAR_JACOBIAN_WORLD = 26u,
    // One world-axis row of the angular Jacobian at a named frame origin,
    // with respect to one compiled generalized-velocity coordinate.
    MR_TASK_OBSERVE_FRAME_ANGULAR_JACOBIAN_WORLD = 27u,
    // Linear velocity at a named frame origin expressed in the horizontal
    // heading frame derived from that frame's authored orientation. Only x/y
    // are defined; roll and pitch never leak into locomotion commands.
    MR_TASK_OBSERVE_FRAME_LINEAR_VELOCITY_HEADING = 28u,
    // Accepted joint velocity finite difference over one control interval.
    MR_TASK_OBSERVE_JOINT_ACCELERATION = 29u,
    // Difference between the newest and immediately preceding action samples.
    MR_TASK_OBSERVE_ACTION_DELTA = 30u,
    // Absolute violation outside compiler-resolved soft lower/upper bounds.
    MR_TASK_OBSERVE_JOINT_SOFT_LIMIT_VIOLATION = 31u,
    // Sum of absolute applied-effort times generalized velocity, in watts.
    MR_TASK_OBSERVE_MECHANICAL_POWER = 32u,
    // Desired contact bit for a semantic support group at the accepted phase.
    MR_TASK_OBSERVE_DESIRED_SUPPORT_CONTACT = 33u,
};

enum MRTaskObservationFlags : mr_u32 {
    MR_TASK_OBSERVATION_NORMALIZE_VECTOR3 = 1u << 0u,
};

enum MRTaskContactGroupFlags : mr_u32 {
    MR_TASK_CONTACT_SUPPORT = 1u << 0u,
    MR_TASK_CONTACT_FORBIDDEN = 1u << 1u,
};

enum MRTaskFrameSourceKind : mr_u32 {
    MR_TASK_FRAME_SOURCE_ARTICULATED_BODY = 0u,
    MR_TASK_FRAME_SOURCE_SCENE_BODY = 1u,
};

enum MRTaskRewardChannel : mr_u32 {
    MR_TASK_REWARD_CHANNEL_PRIMARY = 0u,
    MR_TASK_REWARD_CHANNEL_STABILITY = 1u,
    MR_TASK_REWARD_CHANNEL_VELOCITY = 2u,
    MR_TASK_REWARD_CHANNEL_ACCELERATION = 3u,
    MR_TASK_REWARD_CHANNEL_CONTROL = 4u,
    MR_TASK_REWARD_CHANNEL_CONFIGURATION = 5u,
    MR_TASK_REWARD_CHANNEL_ENERGY = 6u,
    MR_TASK_REWARD_CHANNEL_CONTACT = 7u,
    MR_TASK_REWARD_CHANNEL_COUNT = 8u,
};

enum MRTaskTerminationOpcode : mr_u32 {
    MR_TASK_TERMINATE_SIGNAL_BELOW = 0u,
    MR_TASK_TERMINATE_SIGNAL_ABOVE = 1u,
    MR_TASK_TERMINATE_SIGNAL_OUTSIDE = 2u,
};

// Statically shaped scalar TaskIR. Source leaves reuse the semantic
// observation compiler, but execute without actor noise or mutable bias.
enum MRTaskSignalOpcode : mr_u32 {
    MR_TASK_SIGNAL_SOURCE = 0u,
    MR_TASK_SIGNAL_CONSTANT = 1u,
    MR_TASK_SIGNAL_ADD = 2u,
    MR_TASK_SIGNAL_SUBTRACT = 3u,
    MR_TASK_SIGNAL_MULTIPLY = 4u,
    MR_TASK_SIGNAL_MINIMUM = 5u,
    MR_TASK_SIGNAL_MAXIMUM = 6u,
    MR_TASK_SIGNAL_ABSOLUTE = 7u,
    MR_TASK_SIGNAL_SQUARE = 8u,
    MR_TASK_SIGNAL_SQUARE_ROOT = 9u,
    MR_TASK_SIGNAL_SAFE_DIVIDE = 10u,
    MR_TASK_SIGNAL_CLAMP = 11u,
    MR_TASK_SIGNAL_EXPONENTIAL_TRACKING = 12u,
    MR_TASK_SIGNAL_INSIDE_BOUNDS = 13u,
    MR_TASK_SIGNAL_EXPONENTIAL_DECAY = 14u,
    MR_TASK_SIGNAL_ATAN2 = 15u,
    // Contiguous semantic-source cohort transformed and reduced in one node.
    MR_TASK_SIGNAL_REDUCTION = 16u,
    MR_TASK_SIGNAL_TANH = 17u,
    MR_TASK_SIGNAL_LESS_THAN = 18u,
    MR_TASK_SIGNAL_GREATER_THAN = 19u,
};

enum MRTaskSignalTransform : mr_u32 {
    MR_TASK_SIGNAL_TRANSFORM_IDENTITY = 0u,
    MR_TASK_SIGNAL_TRANSFORM_ABSOLUTE = 1u,
    MR_TASK_SIGNAL_TRANSFORM_SQUARE = 2u,
};

enum MRTaskSignalReduction : mr_u32 {
    MR_TASK_SIGNAL_REDUCE_SUM = 0u,
    MR_TASK_SIGNAL_REDUCE_MEAN = 1u,
    MR_TASK_SIGNAL_REDUCE_MINIMUM = 2u,
    MR_TASK_SIGNAL_REDUCE_MAXIMUM = 3u,
};

enum MRTaskTerminationReason : mr_u32 {
    MR_TASK_TERMINATION_CONTINUING = 0u,
    MR_TASK_TERMINATION_HEIGHT = 1u,
    MR_TASK_TERMINATION_TILT = 2u,
    MR_TASK_TERMINATION_CONTACT = 3u,
    MR_TASK_TERMINATION_TIMEOUT = 4u,
    MR_TASK_TERMINATION_PHYSICS_ERROR = 5u,
    MR_TASK_TERMINATION_GOAL_ERROR = 6u,
};

enum MRLearningPublicationCode : mr_u32 {
    MR_LEARNING_PUBLICATION_SUCCESS = 0u,
    MR_LEARNING_PUBLICATION_INVALID_FLOAT = 1u,
    MR_LEARNING_PUBLICATION_INVALID_TRANSITION = 2u,
};

enum MRLearningPublicationStream : mr_u32 {
    MR_LEARNING_STREAM_ACTOR_OBSERVATIONS = 0u,
    MR_LEARNING_STREAM_CRITIC_OBSERVATIONS = 1u,
    MR_LEARNING_STREAM_LATENTS = 2u,
    MR_LEARNING_STREAM_LOG_PROBABILITIES = 3u,
    MR_LEARNING_STREAM_VALUES = 4u,
    MR_LEARNING_STREAM_TRANSITIONS = 5u,
    MR_LEARNING_STREAM_NONE = MR_INVALID_INDEX,
};

enum MRTaskRandomizationOpcode : mr_u32 {
    MR_TASK_RANDOMIZE_ROOT_POSITION = 0u,
    MR_TASK_RANDOMIZE_ROOT_YAW = 1u,
    MR_TASK_RANDOMIZE_ACTION_POSITION = 2u,
    MR_TASK_RANDOMIZE_VELOCITY = 3u,
    MR_TASK_RANDOMIZE_BODY_PARAMETER = 4u,
    MR_TASK_RANDOMIZE_BODY_PAYLOAD = 5u,
    MR_TASK_RANDOMIZE_CONTROLLER_PARAMETER = 6u,
    MR_TASK_RANDOMIZE_ACTION_DELAY = 7u,
    MR_TASK_RANDOMIZE_OBSERVATION_DELAY = 8u,
    MR_TASK_RANDOMIZE_ACTION_VELOCITY = 9u,
};

// Per-submission dimensions and attribution. Every stride is in elements.
typedef struct MR_ALIGN16 MRTaskDispatchGPU {
    // environments, control steps, nq, nv.
    mr_uint4 counts;
    // action step, body state, shape, scene-body strides.
    mr_uint4 strides;
    // actor step, critic step, transition step, physics status stride.
    mr_uint4 outputs;
    // control dt, physics dt, publish final actor, publish terminal critic.
    mr_float4 timing;
    mr_u64 seed;
    mr_u64 policyRevision;
    mr_u64 taskFingerprint;
    mr_u64 worldFingerprint;
} MRTaskDispatchGPU;

// Immutable compiled tables and task-level constants.
typedef struct MR_ALIGN16 MRTaskProgramHeaderGPU {
    // actions, actor-frame operators, critic operators, contact groups.
    mr_uint4 counts0;
    // contact members, recorder operators, reserved, reward operators.
    mr_uint4 counts1;
    // termination operators, randomization operators, bias slots,
    // terrain-sample offsets.
    mr_uint4 counts2;
    // actor frame, history length, contact metric count, delay-state count.
    mr_uint4 layout;
    // articulation, root body, root q offset, root v offset.
    mr_uint4 root;
    // terrain scene-body local index, shape index, geometry index, profiles.
    mr_uint4 terrain;
    // max episode steps, max observation delay, reserved, flags.
    mr_uint4 schedule;
    // Curriculum levels, evaluation-window steps, success signal, and compact
    // command count. The signal is MR_INVALID_INDEX without promotion.
    mr_uint4 curriculum;
    // Phase period, curriculum success threshold, minimum episode-survival
    // fraction, and support-force threshold.
    mr_float4 taskScalars;
    // Cohort-zero probability and command duration min/max, seconds.
    mr_float4 commandSchedule;
    // Push velocity magnitude and interval min/max, seconds.
    mr_float4 eventSchedule;
    // Byte offsets in the immutable packed task arena:
    // action bindings, actor operators, critic operators, contact groups.
    mr_uint4 offsets0;
    // contact members, recorder operators, reserved, reward operators.
    mr_uint4 offsets1;
    // termination operators, randomization operators, bias specs, terrain
    // samples.
    mr_uint4 offsets2;
    // Terrain reset profiles, command operators, frames, and goals.
    mr_uint4 offsets3;
    mr_u64 taskFingerprint;
    mr_u64 worldFingerprint;
    // selected-articulation first body/count, ABI version, critic history.
    mr_uint4 articulation;
    // Root link-frame origin relative to the root COM, in root body axes.
    mr_float4 rootReference;
    // named frames, static SE(3) goals, SensorIR fingerprint low/high.
    mr_uint4 typedCounts;
    // SignalIR nodes, semantic source leaves, spatial-Jacobian stride, and
    // dense current-SensorIR semantic-source scratch count.
    mr_uint4 graphCounts;
    // SignalIR source operators and nodes; remaining offsets are reserved.
    mr_uint4 offsets4;
} MRTaskProgramHeaderGPU;

typedef struct MR_ALIGN16 MRTaskActionBindingGPU {
    // action index, global DoF index, q index, v index.
    mr_uint4 indices;
    // normalized scale, lower target, upper target, reserved.
    mr_float4 parameters;
} MRTaskActionBindingGPU;

typedef struct MR_ALIGN16 MRTaskObservationOperatorGPU {
    // opcode, resolved source index, component, flags.
    mr_uint4 source;
    // scale, offset, uniform-noise amplitude, bias slot as float bits unused.
    mr_float4 transform;
    // Bias index, deterministic noise channel, goal/reference/output offset,
    // and dense SensorIR scratch slot for SignalIR sources.
    mr_uint4 auxiliary;
    // Compiler-resolved source parameters. Their meaning is opcode-specific;
    // ordinary affine transforms and actor corruption remain above.
    mr_float4 parameters;
} MRTaskObservationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskSignalOperatorGPU {
    // Ordinary node: opcode, semantic-source index, left node, right node.
    // Reduction node: opcode, source offset/count, packed transform/reduction.
    mr_uint4 inputs;
    // Operator parameters. Their meaning is opcode-specific.
    mr_float4 parameters;
} MRTaskSignalOperatorGPU;

typedef struct MR_ALIGN16 MRTaskCommandOperatorGPU {
    // Initial lower/upper and hard lower/upper bounds.
    mr_float4 range;
    // Per-curriculum-level symmetric expansion and reserved values.
    mr_float4 curriculum;
} MRTaskCommandOperatorGPU;

typedef struct MR_ALIGN16 MRTaskContactGroupGPU {
    // member offset/count, flags, compact metric offset.
    mr_uint4 members;
    // reference body, compact local-wrench offset, reserved, reserved.
    mr_uint4 reference;
    // local reference xyz and clearance radius.
    mr_float4 localReference;
    // Reference link-frame origin relative to the body COM.
    mr_float4 kinematicReference;
    // gait phase offset, stance fraction, reserved, reserved.
    mr_float4 gait;
} MRTaskContactGroupGPU;

typedef struct MR_ALIGN16 MRTaskFrameGPU {
    // x = global body index, y = MRTaskFrameSourceKind,
    // z = source-layout index (global body-pose index for articulated bodies,
    // scene-local state index for scene bodies), w = articulation owner or
    // MR_INVALID_INDEX for scene bodies.
    mr_uint4 indices;
    // Authored frame origin relative to the body COM, in body axes.
    mr_float4 localPosition;
    // Authored frame-to-body quaternion xyzw.
    mr_float4 localOrientation;
} MRTaskFrameGPU;

typedef struct MR_ALIGN16 MRTaskGoalGPU {
    // Goal mode, playback mode, stable random identity low/high.
    mr_uint4 metadata;
    // Base pose. Sampled goals perturb it once per episode; trajectories use
    // it as the start pose.
    mr_float4 position;
    mr_float4 orientation;
    // Trajectory end pose. Canonical zero/identity for other modes.
    mr_float4 targetPosition;
    mr_float4 targetOrientation;
    // Episode-sampled world-position offset bounds.
    mr_float4 positionOffsetLower;
    mr_float4 positionOffsetUpper;
    // Episode-sampled local tangent rotation-vector bounds, radians.
    mr_float4 rotationVectorLower;
    mr_float4 rotationVectorUpper;
    // Trajectory duration and non-negative phase offset, seconds.
    mr_float4 timing;
} MRTaskGoalGPU;

typedef struct MR_ALIGN16 MRTaskRewardOperatorGPU {
    // Resolved SignalIR index, reward channel, reserved, reserved.
    mr_uint4 source;
    // weight and reserved values.
    mr_float4 parameters;
} MRTaskRewardOperatorGPU;

typedef struct MR_ALIGN16 MRTaskRecorderOperatorGPU {
    // Resolved SignalIR index, compact transition metric slot, reserved,
    // reserved. Recorder identity remains host metadata and is fingerprinted.
    mr_uint4 source;
} MRTaskRecorderOperatorGPU;

typedef struct MR_ALIGN16 MRTaskTerminationOperatorGPU {
    // opcode, resolved group/signal index, reason, priority.
    mr_uint4 source;
    // threshold, one-shot failure penalty, and reserved values.
    mr_float4 parameters;
} MRTaskTerminationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskRandomizationOperatorGPU {
    // opcode, resolved group/index, component, minimum curriculum level.
    mr_uint4 target;
    // lower, upper, auxiliary lower, auxiliary upper.
    mr_float4 parameters;
} MRTaskRandomizationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskBiasSpecGPU {
    // lower, upper, reserved, reserved.
    mr_float4 range;
    // deterministic random channel and reserved values.
    mr_uint4 metadata;
} MRTaskBiasSpecGPU;

typedef struct MR_ALIGN16 MRTaskStateGPU {
    // episode step, episode index, curriculum level, terrain level.
    mr_uint4 episode;
    // command steps, push steps, actuator delay, observation delay.
    mr_uint4 schedule;
    // initialized, pending reset, last termination, reserved.
    mr_uint4 status;
    // commanded x/y/yaw velocity and gait phase.
    mr_float4 commandAndPhase;
    // current mechanical power, reserved, episode return, curriculum metric.
    mr_float4 powerReturnMetric;
} MRTaskStateGPU;

// One compact task-wide curriculum controller remains device-resident across
// submissions. Its success metric is a compiler-bound SignalIR node; terrain
// difficulty remains environment-local.
typedef struct MR_ALIGN16 MRTaskCurriculumStateGPU {
    mr_u64 controlSteps;
    mr_u64 completedEpisodeCount;
    mr_u64 timeoutEpisodeCount;
    float metricSum;
    mr_u32 commandLevel;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRTaskCurriculumStateGPU;

typedef struct MR_ALIGN16 MRTaskTransitionGPU {
    // reward followed by three compiler-bound recorder metrics.
    mr_float4 rewardAndMetrics;
    // done, timeout, physics error, termination reason.
    mr_uint4 termination;
    // Generic authored reward-reporting channels 0 through 3.
    mr_float4 rewardBreakdown0;
    // Generic authored reward-reporting channels 4 through 7.
    mr_float4 rewardBreakdown1;
    mr_u64 policyRevision;
    // V of the accepted post-transition state for timeout bootstrapping.
    float timeoutBootstrapValue;
    // Episode mean of the compiled curriculum signal.
    float episodeMetric;
    // Task-wide curriculum and environment-local terrain levels.
    mr_uint4 taskProgress;
} MRTaskTransitionGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRTaskDispatchGPU) == 96u);
static_assert(sizeof(MRTaskProgramHeaderGPU) == 336u);
static_assert(sizeof(MRTaskActionBindingGPU) == 32u);
static_assert(sizeof(MRTaskObservationOperatorGPU) == 64u);
static_assert(sizeof(MRTaskSignalOperatorGPU) == 32u);
static_assert(sizeof(MRTaskCommandOperatorGPU) == 32u);
static_assert(sizeof(MRTaskContactGroupGPU) == 80u);
static_assert(sizeof(MRTaskFrameGPU) == 48u);
static_assert(sizeof(MRTaskGoalGPU) == 160u);
static_assert(sizeof(MRTaskRewardOperatorGPU) == 32u);
static_assert(sizeof(MRTaskRecorderOperatorGPU) == 16u);
static_assert(sizeof(MRTaskTerminationOperatorGPU) == 32u);
static_assert(sizeof(MRTaskRandomizationOperatorGPU) == 32u);
static_assert(sizeof(MRTaskBiasSpecGPU) == 32u);
static_assert(sizeof(MRTaskStateGPU) == 80u);
static_assert(sizeof(MRTaskCurriculumStateGPU) == 48u);
static_assert(sizeof(MRTaskTransitionGPU) == 96u);
#endif
#endif
