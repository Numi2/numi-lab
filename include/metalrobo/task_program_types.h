#pragma once

#include "metalrobo/engine_types.h"

#define MR_TASK_PROGRAM_ABI_VERSION 16u

enum MRTaskProgramFlags : mr_u32 {
    MR_TASK_PROGRAM_TERRAIN = 1u << 0u,
    MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY = 1u << 1u,
    MR_TASK_PROGRAM_RECOVERY_CURRICULUM = 1u << 2u,
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
    // Command-gated locomotion phase: sin/cos of the task phase, or zero
    // while the commanded xyz velocity magnitude is below 0.1.
    MR_TASK_OBSERVE_GAIT_PHASE = 14u,
    // Privileged recovery-event state: active, previous tilt, peak tilt,
    // and stable time. Intended for asymmetric critics, not deployed actors.
    MR_TASK_OBSERVE_RECOVERY_EVENT = 15u,
    // Deployable object-centric perception contract for one tracked scene
    // body: confidence, root-local position xyz, and root-local relative
    // velocity xyz. Simulation supplies the contract natively; deployment
    // may populate the same slots from an RGB-D perception provider.
    MR_TASK_OBSERVE_OBJECT_TRACK = 16u,
    // Deployable ball-only metric depth. Each scalar component addresses one
    // row-major pixel in one sparse temporal frame declared by TaskPack.
    // The native visual stage overwrites these zero-valued physics slots.
    MR_TASK_OBSERVE_MASKED_DEPTH = 17u,
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
    MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED = 21u,
    MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS = 22u,
    MR_TASK_REWARD_UPRIGHTNESS = 23u,
    MR_TASK_REWARD_SUPPORT_CONTACT_COUNT = 24u,
    MR_TASK_REWARD_BODY_HEIGHT_EXPONENTIAL = 25u,
    MR_TASK_REWARD_SUPPORT_HEIGHT_EXPONENTIAL = 26u,
    MR_TASK_REWARD_BODY_UP_EXPONENTIAL = 27u,
    MR_TASK_REWARD_STANDING_COMPLETION = 28u,
    MR_TASK_REWARD_RECOVERY_TILT_PROGRESS = 29u,
    MR_TASK_REWARD_RECOVERY_COMPLETION = 30u,
    // Training-only whole-body safety signal. source.y selects a semantic
    // protected-body group and source.z selects one dynamic scene projectile.
    // The value is the most negative per-link CBF constraint and is zero when
    // every protected link is clearing the projectile safely.
    MR_TASK_REWARD_LINK_CLEARANCE_BARRIER = 31u,
    // One-shot reward when an authored projectile event completes without a
    // protected-body contact.
    MR_TASK_REWARD_PROJECTILE_MISS = 32u,
    // Dense gross-evasion signal for one active dynamic projectile. The
    // reward combines saturated root-to-projectile distance with root
    // horizontal stillness; per-link CBF remains the fine safety signal.
    MR_TASK_REWARD_PROJECTILE_EVASION = 33u,
    // Safe-only regularizers. They are zero while an authored projectile is
    // live, so standing discipline never suppresses a genuine dodge.
    MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS = 34u,
    MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE = 35u,
};

enum MRTaskTerminationOpcode : mr_u32 {
    MR_TASK_TERMINATE_MINIMUM_ROOT_HEIGHT = 0u,
    MR_TASK_TERMINATE_MAXIMUM_TILT = 1u,
    MR_TASK_TERMINATE_CONTACT_GROUP = 2u,
    // Contact-group termination scoped to the active projectile flight. This
    // avoids treating ordinary support contact as a dodge failure.
    MR_TASK_TERMINATE_PROJECTILE_CONTACT = 3u,
};

enum MRTaskTerminationReason : mr_u32 {
    MR_TASK_TERMINATION_CONTINUING = 0u,
    MR_TASK_TERMINATION_HEIGHT = 1u,
    MR_TASK_TERMINATION_TILT = 2u,
    MR_TASK_TERMINATION_CONTACT = 3u,
    MR_TASK_TERMINATION_TIMEOUT = 4u,
    MR_TASK_TERMINATION_PHYSICS_ERROR = 5u,
    MR_TASK_TERMINATION_PROJECTILE_CONTACT = 6u,
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
    MR_TASK_RANDOMIZE_ROOT_HEIGHT = 10u,
    MR_TASK_RANDOMIZE_ROOT_ORIENTATION = 11u,
    MR_TASK_RANDOMIZE_JOINT_POSITION = 12u,
    MR_TASK_RANDOMIZE_SCENE_BODY_POSITION = 13u,
    MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY = 14u,
    MR_TASK_RANDOMIZE_SCENE_BODY_LAUNCH_STEP = 15u,
    // Event-driven physical impact sequence. The compiler resolves these into
    // a dedicated immutable table; reset randomization never executes them.
    MR_TASK_RANDOMIZE_SCENE_BODY_EVENT_IMPACT = 16u,
};

enum MRTaskImpactTransitionFlags : mr_u32 {
    MR_TASK_IMPACT_TOUCH = 1u << 0u,
    MR_TASK_IMPACT_RECOVERED = 1u << 1u,
    MR_TASK_IMPACT_MISSED = 1u << 2u,
    MR_TASK_IMPACT_SEQUENCE_ENABLED = 1u << 3u,
    // First solver contact between the active projectile and any link in the
    // compiled articulation. This is evidence, not a termination decision.
    MR_TASK_IMPACT_CONTACT = 1u << 4u,
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
    // Impact events, contact-member radii, and reserved table counts.
    mr_uint4 counts3;
    // actor frame, history length, contact metric count, delay-state count.
    mr_uint4 layout;
    // articulation, root body, root q offset, root v offset.
    mr_uint4 root;
    // terrain scene-body local index, shape index, geometry index, profiles.
    mr_uint4 terrain;
    // max episode steps, max observation delay, curriculum levels, flags.
    mr_uint4 schedule;
    // base height target, gait period, clearance target, success threshold.
    mr_float4 locomotion;
    // xyz command lower bound; w standing-command probability.
    mr_float4 commandLower;
    // xyz command upper bound; w minimum episode-survival fraction.
    mr_float4 commandUpper;
    // command duration min/max and push interval min/max, seconds.
    mr_float4 scheduleSeconds;
    // Push velocity, contact threshold, reserved, reserved.
    mr_float4 dynamics;
    // Projectile horizontal speed lower/upper and target height lower/upper.
    mr_float4 projectile;
    // Compiled world gravity xyz and horizontal target radius.
    mr_float4 projectileGravity;
    // Byte offsets in the immutable packed task arena:
    // action bindings, actor operators, critic operators, contact groups.
    mr_uint4 offsets0;
    // contact members, joint groups, joint members, reward operators.
    mr_uint4 offsets1;
    // termination operators, randomization operators, bias specs, terrain
    // samples.
    mr_uint4 offsets2;
    // Terrain reset profiles, command curriculum, impact events, and
    // contact-member radii.
    mr_uint4 offsets3;
    mr_u64 taskFingerprint;
    mr_u64 worldFingerprint;
    // selected-articulation first body/count, ABI version, critic history.
    mr_uint4 articulation;
    // Root link-frame origin relative to the root COM, in root body axes.
    mr_float4 rootReference;
    // Masked-depth width, height, sparse frame count, maximum frame offset.
    mr_uint4 visualLayout;
    // Sparse control-step offsets, newest first. Unused lanes are zero.
    mr_uint4 visualHistory;
    // Near depth, far depth, edge-flicker probability, reserved.
    mr_float4 visualRange;
    // Full/pixel dropout probabilities, depth jitter, noise sigma.
    mr_float4 visualCorruption;
} MRTaskProgramHeaderGPU;

typedef struct MR_ALIGN16 MRTaskActionBindingGPU {
    // action index, global DoF index, q index, v index.
    mr_uint4 indices;
    // normalized scale, lower target, upper target, response time seconds.
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
} MRTaskTerminationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskRandomizationOperatorGPU {
    // opcode, resolved group/index, component, minimum curriculum level.
    mr_uint4 target;
    // lower, upper, auxiliary lower, auxiliary upper.
    mr_float4 parameters;
} MRTaskRandomizationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskImpactEventGPU {
    // Scene-body local index, sequence order, minimum curriculum, global body.
    mr_uint4 binding;
    // Stable tilt, stable seconds, maximum flight seconds, minimum height.
    mr_float4 gate;
} MRTaskImpactEventGPU;

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
    // previous tilt, peak event tilt, stable time, event active flag.
    mr_float4 recovery;
    // detected events, completed recoveries, previous touch, packed impact
    // sequence state (including the per-throw contact latch).
    mr_uint4 recoveryStats;
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
static_assert(sizeof(MRTaskProgramHeaderGPU) == 416u);
static_assert(sizeof(MRTaskActionBindingGPU) == 32u);
static_assert(sizeof(MRTaskObservationOperatorGPU) == 48u);
static_assert(sizeof(MRTaskContactGroupGPU) == 80u);
static_assert(sizeof(MRTaskIndexGroupGPU) == 16u);
static_assert(sizeof(MRTaskRewardOperatorGPU) == 32u);
static_assert(sizeof(MRTaskTerminationOperatorGPU) == 32u);
static_assert(sizeof(MRTaskRandomizationOperatorGPU) == 32u);
static_assert(sizeof(MRTaskImpactEventGPU) == 32u);
static_assert(sizeof(MRTaskBiasSpecGPU) == 32u);
static_assert(sizeof(MRTaskStateGPU) == 112u);
static_assert(sizeof(MRTaskCurriculumStateGPU) == 48u);
static_assert(sizeof(MRTaskTransitionGPU) == 96u);
#endif
#endif
