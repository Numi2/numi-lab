#pragma once

#include "metalrobo/engine_types.h"

#define MR_TASK_PROGRAM_ABI_VERSION 8u

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
};

enum MRTaskObservationFlags : mr_u32 {
    MR_TASK_OBSERVATION_NORMALIZE_VECTOR3 = 1u << 0u,
};

enum MRTaskContactGroupFlags : mr_u32 {
    MR_TASK_CONTACT_SUPPORT = 1u << 0u,
    MR_TASK_CONTACT_FORBIDDEN = 1u << 1u,
};

enum MRTaskRewardOpcode : mr_u32 {
    MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING = 0u,
    MR_TASK_REWARD_YAW_VELOCITY_TRACKING = 1u,
    MR_TASK_REWARD_CONSTANT = 2u,
    MR_TASK_REWARD_ROOT_VERTICAL_VELOCITY_SQUARED = 3u,
    MR_TASK_REWARD_ROOT_ROLL_PITCH_VELOCITY_SQUARED = 4u,
    MR_TASK_REWARD_TILT_SQUARED = 5u,
    MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED = 6u,
    MR_TASK_REWARD_JOINT_VELOCITY_SQUARED = 7u,
    MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED = 8u,
    MR_TASK_REWARD_ACTION_RATE_SQUARED = 9u,
    MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_SQUARED = 10u,
    MR_TASK_REWARD_MECHANICAL_POWER = 11u,
    MR_TASK_REWARD_JOINT_GROUP_POSTURE_SQUARED = 12u,
    MR_TASK_REWARD_GAIT_CONTACT_MATCH = 13u,
    MR_TASK_REWARD_SWING_CLEARANCE = 14u,
    MR_TASK_REWARD_SUPPORT_SLIP = 15u,
    MR_TASK_REWARD_FORBIDDEN_CONTACT = 16u,
    MR_TASK_REWARD_JOINT_GROUP_POSTURE_ABSOLUTE = 17u,
    MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED = 18u,
    MR_TASK_REWARD_FOOT_CLEARANCE = 19u,
    MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE = 20u,
    MR_TASK_REWARD_FRAME_POSITION_ERROR_SQUARED = 21u,
    MR_TASK_REWARD_FRAME_ORIENTATION_ERROR_SQUARED = 22u,
    MR_TASK_REWARD_FRAME_POSITION_TRACKING = 23u,
    MR_TASK_REWARD_FRAME_ORIENTATION_TRACKING = 24u,
};

enum MRTaskTerminationOpcode : mr_u32 {
    MR_TASK_TERMINATE_MINIMUM_ROOT_HEIGHT = 0u,
    MR_TASK_TERMINATE_MAXIMUM_TILT = 1u,
    MR_TASK_TERMINATE_CONTACT_GROUP = 2u,
    MR_TASK_TERMINATE_MAXIMUM_FRAME_POSITION_ERROR = 3u,
    MR_TASK_TERMINATE_MAXIMUM_FRAME_ORIENTATION_ERROR = 4u,
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
    // contact members, joint groups, joint members, reward operators.
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
    // max episode steps, max observation delay, curriculum levels, flags.
    mr_uint4 schedule;
    // Root-height target, periodic-task interval, clearance target, and
    // curriculum success threshold. Operators that consume these values are
    // validated by the compiler; fixed-base tasks need not read floating-root
    // state.
    mr_float4 taskScalars;
    // xyz command lower bound; w standing-command probability.
    mr_float4 commandLower;
    // xyz command upper bound; w minimum episode-survival fraction.
    mr_float4 commandUpper;
    // command duration min/max and push interval min/max, seconds.
    mr_float4 scheduleSeconds;
    // push velocity magnitude, contact force threshold, reserved, reserved.
    mr_float4 dynamics;
    // Byte offsets in the immutable packed task arena:
    // action bindings, actor operators, critic operators, contact groups.
    mr_uint4 offsets0;
    // contact members, joint groups, joint members, reward operators.
    mr_uint4 offsets1;
    // termination operators, randomization operators, bias specs, terrain
    // samples.
    mr_uint4 offsets2;
    // terrain reset profiles and reserved offsets.
    mr_uint4 offsets3;
    mr_u64 taskFingerprint;
    mr_u64 worldFingerprint;
    // selected-articulation first body/count, ABI version, critic history.
    mr_uint4 articulation;
    // Root link-frame origin relative to the root COM, in root body axes.
    mr_float4 rootReference;
    // named frames, static SE(3) goals, SensorIR fingerprint low/high.
    mr_uint4 typedCounts;
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
    // bias index, deterministic noise channel, reserved, reserved.
    mr_uint4 auxiliary;
} MRTaskObservationOperatorGPU;

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

typedef struct MR_ALIGN16 MRTaskIndexGroupGPU {
    // member offset/count, reserved, reserved.
    mr_uint4 members;
} MRTaskIndexGroupGPU;

typedef struct MR_ALIGN16 MRTaskFrameGPU {
    // Global body index and reserved values.
    mr_uint4 indices;
    // Authored frame origin relative to the body COM, in body axes.
    mr_float4 localPosition;
    // Authored frame-to-body quaternion xyzw.
    mr_float4 localOrientation;
} MRTaskFrameGPU;

typedef struct MR_ALIGN16 MRTaskGoalGPU {
    // Static goal mode and reserved values. Later sampled/trajectory modes
    // extend this persisted ABI deliberately rather than adding task shaders.
    mr_uint4 metadata;
    mr_float4 position;
    mr_float4 orientation;
} MRTaskGoalGPU;

typedef struct MR_ALIGN16 MRTaskRewardOperatorGPU {
    // opcode, resolved group/index, auxiliary index, flags.
    mr_uint4 source;
    // weight and three operator parameters.
    mr_float4 parameters;
} MRTaskRewardOperatorGPU;

typedef struct MR_ALIGN16 MRTaskTerminationOperatorGPU {
    // opcode, resolved group/index, reason, priority.
    mr_uint4 source;
    // threshold, one-shot failure penalty, and reserved values.
    mr_float4 parameters;
    // Resolved goal index and reserved values.
    mr_uint4 auxiliary;
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
    // current mechanical power, reserved, episode return, tracking score.
    mr_float4 airReturnTracking;
} MRTaskStateGPU;

// One compact task-wide curriculum controller remains device-resident across
// submissions. The command level is global, matching the authored Unitree
// curriculum, while terrain difficulty remains environment-local.
typedef struct MR_ALIGN16 MRTaskCurriculumStateGPU {
    mr_u64 controlSteps;
    mr_u64 completedEpisodeCount;
    mr_u64 timeoutEpisodeCount;
    float trackingScoreSum;
    mr_u32 commandLevel;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRTaskCurriculumStateGPU;

typedef struct MR_ALIGN16 MRTaskTransitionGPU {
    // reward, tracking score, root height, tilt.
    mr_float4 rewardAndState;
    // done, timeout, physics error, termination reason.
    mr_uint4 termination;
    // task, base, joint-velocity, and joint-acceleration contributions.
    mr_float4 rewardBreakdown0;
    // action-control, posture/limits, energy, and contact contributions.
    mr_float4 rewardBreakdown1;
    mr_u64 policyRevision;
    // V of the accepted post-transition state for timeout bootstrapping.
    float timeoutBootstrapValue;
    // Mean linear tracking score for a non-physics episode ending here.
    float episodeTrackingScore;
    // Global command curriculum and environment-local terrain levels.
    mr_uint4 taskProgress;
} MRTaskTransitionGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRTaskDispatchGPU) == 96u);
static_assert(sizeof(MRTaskProgramHeaderGPU) == 320u);
static_assert(sizeof(MRTaskActionBindingGPU) == 32u);
static_assert(sizeof(MRTaskObservationOperatorGPU) == 48u);
static_assert(sizeof(MRTaskContactGroupGPU) == 80u);
static_assert(sizeof(MRTaskIndexGroupGPU) == 16u);
static_assert(sizeof(MRTaskFrameGPU) == 48u);
static_assert(sizeof(MRTaskGoalGPU) == 48u);
static_assert(sizeof(MRTaskRewardOperatorGPU) == 32u);
static_assert(sizeof(MRTaskTerminationOperatorGPU) == 48u);
static_assert(sizeof(MRTaskRandomizationOperatorGPU) == 32u);
static_assert(sizeof(MRTaskBiasSpecGPU) == 32u);
static_assert(sizeof(MRTaskStateGPU) == 80u);
static_assert(sizeof(MRTaskCurriculumStateGPU) == 48u);
static_assert(sizeof(MRTaskTransitionGPU) == 96u);
#endif
#endif
