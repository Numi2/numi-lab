#pragma once

#include "metalrobo/engine_types.h"

#define MR_TASK_PROGRAM_ABI_VERSION 32u
#define MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT 13u
#define MR_TASK_MASKED_DEPTH_FEATURE_COUNT 24u

enum MRTaskProgramFlags : mr_u32 {
    MR_TASK_PROGRAM_TERRAIN = 1u << 0u,
    MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY = 1u << 1u,
    MR_TASK_PROGRAM_THREAT_TEACHER = 1u << 3u,
    // The masked-depth suffix also contains compact features derived from
    // the corrupted sparse depth frames on device. No scene state enters
    // these slots.
    MR_TASK_PROGRAM_MASKED_DEPTH_FEATURES = 1u << 5u,
    // Action values become bounded residuals around the current retargeted
    // joint reference instead of offsets from the mechanism default pose.
    MR_TASK_PROGRAM_INTERACTION_REFERENCE = 1u << 6u,
    // Initialize the mechanism from the first InteractionPack frame. This is
    // intentionally independent of reference control so distilled policies
    // can be evaluated autonomously from the exact demonstrated state.
    MR_TASK_PROGRAM_INTERACTION_RESET = 1u << 7u,
};

enum MRTaskInteractionFlags : mr_u32 {
    MR_TASK_INTERACTION_LOOP = 1u << 0u,
};

enum MRTaskInteractionContactMode : mr_u32 {
    MR_TASK_INTERACTION_CONTACT_FREE = 0u,
    MR_TASK_INTERACTION_CONTACT_APPROACH = 1u,
    MR_TASK_INTERACTION_CONTACT_STICK = 2u,
    MR_TASK_INTERACTION_CONTACT_ROLL = 3u,
    MR_TASK_INTERACTION_CONTACT_SLIDE = 4u,
    MR_TASK_INTERACTION_CONTACT_RELEASE = 5u,
};

enum MRTaskThreatClass : mr_u32 {
    MR_TASK_THREAT_NONE = 0u,
    MR_TASK_THREAT_STEP_OVER = 1u,
    MR_TASK_THREAT_SIDESTEP = 2u,
    MR_TASK_THREAT_LEAN = 3u,
    MR_TASK_THREAT_DUCK = 4u,
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
    // Deployable ball-only metric depth. Initial scalar components address
    // row-major pixels in sparse temporal frames. When the visual feature
    // flag is set, 24 camera-derived summary components follow the pixels.
    // The native visual stage overwrites these zero-valued physics slots.
    MR_TASK_OBSERVE_MASKED_DEPTH = 17u,
    // Device-resident aggregate over every authored support contact group:
    // total normal load, phase-signed load balance, and maximum planar slip.
    // This is compact plantar/contact evidence, not inferred touch.
    MR_TASK_OBSERVE_SUPPORT_SENSE = 18u,
    // Authored support-patch observation. Components are local force xyz,
    // local torque xyz, local center of pressure xy, occupied area, followed
    // by row-major pressure cells. Spatial resolution is authored per contact
    // group and compiled into fixed tables.
    MR_TASK_OBSERVE_SUPPORT_PATCH = 19u,
    // Difference between the live joint position and the selected
    // InteractionPack clip's current joint target.
    MR_TASK_OBSERVE_INTERACTION_JOINT_POSITION_ERROR = 20u,
    // Expected contact (component 0) and confidence (component 1) for one
    // compiled contact track. Generated contact remains intent, not evidence.
    MR_TASK_OBSERVE_INTERACTION_CONTACT_MODE = 21u,
    // One of the 13 compact contact targets: wrench 6, CoP 2, area 1, and
    // row-major 2x2 pressure 4. Invalid target components publish zero.
    MR_TASK_OBSERVE_INTERACTION_CONTACT_TARGET = 22u,
    // Reference phase sin, cos, and normalized progress.
    MR_TASK_OBSERVE_INTERACTION_PHASE = 23u,
    // Per-feature validity for the compact interaction contact target. This
    // keeps an unknown generated field distinct from a valid zero target.
    MR_TASK_OBSERVE_INTERACTION_CONTACT_VALIDITY = 24u,
    // Root-link tracking state for generated motion: root-local position
    // error xyz, orientation error axis-angle xyz, linear-velocity error xyz,
    // and angular-velocity error xyz. This exposes the physical tracking
    // problem to an action policy without granting root actuation.
    MR_TASK_OBSERVE_INTERACTION_ROOT_TRACKING_ERROR = 25u,
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
    // Privileged training-only Joint-CBF supervision. The correction term
    // measures the closed-form projection distance from the actor's desired
    // joint velocity; the buffer term penalizes the predicted keep-out
    // violation. Neither changes the deployed actor contract.
    MR_TASK_REWARD_JOINT_CBF_CORRECTION = 36u,
    MR_TASK_REWARD_JOINT_CBF_BUFFER = 37u,
    // Training-only prospective whole-body clearance. The native threat
    // query supplies the minimum predicted link/projectile clearance and
    // time to closest approach; the deployed actor still consumes only its
    // authored sensor observations.
    MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE = 38u,
    // Exponential joint-position tracking against the selected interaction
    // reference. A named joint group may restrict the action set.
    MR_TASK_REWARD_INTERACTION_JOINT_TRACKING = 39u,
    // Contact-mode and validity-masked compact-field tracking. Solved contact
    // supplies the achieved values; the reference never overwrites physics.
    MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING = 40u,
    // Smooth whole-recovery objective plus strict per-step restoration truth.
    // Parameters are maximum joint RMS error, maximum root-XY error,
    // minimum root-orientation cosine, and maximum generalized-speed RMS.
    MR_TASK_REWARD_RESTORATION = 41u,
    // Smooth tracking of authored root position and orientation. Parameters
    // are positive position and orientation squared-error widths.
    MR_TASK_REWARD_INTERACTION_ROOT_TRACKING = 42u,
    // Dense, phase-readable physical get-up objective. source.y is the
    // hand/knee assist group and source.z is the trunk group. Parameters are
    // standing height, upright cosine, horizontal support radius, and quiet
    // generalized-speed scale. Feet and CoP come from authored support groups.
    MR_TASK_REWARD_WHOLE_BODY_RECOVERY = 43u,
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
    // Generic task-outcome evidence sharing the transition flag word. This is
    // emitted only when the standing-completion operator observes bilateral
    // solver support, sufficient root height, and uprightness together.
    MR_TASK_OUTCOME_STANDING = 1u << 31u,
    // Standing plus nominal posture, root pose, and stillness. The learner
    // still requires a sustained streak before accepting teacher actions.
    MR_TASK_OUTCOME_RESTORED = 1u << 30u,
    // Phase evidence for a get-up transition. These are observations of the
    // accepted solver state, never curriculum or promotion gates.
    MR_TASK_OUTCOME_RECOVERY_BRACE = 1u << 29u,
    MR_TASK_OUTCOME_TRUNK_CLEAR = 1u << 28u,
    MR_TASK_OUTCOME_FOOT_SUPPORT = 1u << 27u,
    MR_TASK_OUTCOME_SUPPORT_TRANSFER = 1u << 26u,
    MR_TASK_OUTCOME_RECOVERY_RISE = 1u << 25u,
    MR_TASK_OUTCOME_QUIET_STAND = 1u << 24u,
    // The policy contributed an additive residual to an authored interaction
    // target. PPO remains valid; absolute teacher imitation is not.
    MR_TASK_OUTCOME_POLICY_RESIDUAL = 1u << 23u,
    // An InteractionPack supplied the complete executed action. The sampled
    // student action did not control physics, so PPO attribution is invalid;
    // the transition remains eligible for outcome-weighted distillation.
    MR_TASK_OUTCOME_INTERACTION_TEACHER = 1u << 22u,
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
    // sampled difficulty-band lower bound, inclusive upper bound, reserved.
    // MR_INVALID_INDEX in y selects the compiled TaskPack upper bound.
    mr_uint4 sampling;
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
    // Impact events, contact-member radii, interaction tracks, and current
    // non-visual actor observations.
    mr_uint4 counts3;
    // actor frame, history length, contact metric count, delay-state count.
    mr_uint4 layout;
    // articulation, root body, root q offset, root v offset.
    mr_uint4 root;
    // terrain scene-body local index, shape index, geometry index, profiles.
    mr_uint4 terrain;
    // max episode steps, max observation delay, difficulty bands, flags.
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
    // Terrain reset profiles, command difficulty ranges, impact events, and
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
    // Protected contact-group index, enabled flag, and reserved lanes.
    mr_uint4 threat;
    // Activation speed, prediction horizon, safety margin, CBF alpha.
    mr_float4 threatTiming;
    // Root-relative step-over, sidestep, and lean height boundaries.
    mr_float4 threatClassification;
    // Urgency horizon, desired-velocity horizon, projection epsilon, reserved.
    mr_float4 threatTeacher;
    // Anchor global body, tracked-body count, feature count, arena byte offset.
    mr_uint4 motion;
    // Interaction frames, joint targets/frame, contact tracks, flags.
    mr_uint4 interaction;
    // Reference fps, duration seconds, reserved, reserved.
    mr_float4 interactionTiming;
    // Root targets, joint targets, contact descriptors, sample metadata.
    mr_uint4 interactionOffsets0;
    // Contact targets, tolerances, reserved, reserved.
    mr_uint4 interactionOffsets1;
} MRTaskProgramHeaderGPU;

typedef struct MR_ALIGN16 MRTaskActionBindingGPU {
    // action index, global DoF index, q index, v index.
    mr_uint4 indices;
    // normalized scale, lower target, upper target, response time seconds.
    mr_float4 parameters;
    // Authored drive stiffness, damping, reserved, reserved. Interaction
    // playback uses these values to preserve the reference joint velocity
    // through the implicit position-drive target without bypassing physics.
    mr_float4 drive;
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
    // Support-patch bounds relative to localReference: min x/y, max x/y.
    mr_float4 supportPatchBounds;
    // Patch width, height, cell count, first compact pressure-cell metric.
    mr_uint4 supportPatch;
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
    // Fourth operator parameter and reserved lanes. Most rewards leave this
    // zero; whole-state restoration requires the fourth tolerance.
    mr_float4 auxiliary;
} MRTaskRewardOperatorGPU;

typedef struct MR_ALIGN16 MRTaskTerminationOperatorGPU {
    // opcode, resolved group/index, reason, priority.
    mr_uint4 source;
    // threshold, one-shot failure penalty, and reserved values.
    mr_float4 parameters;
    // Inclusive minimum/maximum difficulty band; remaining lanes reserved.
    mr_uint4 schedule;
} MRTaskTerminationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskRandomizationOperatorGPU {
    // opcode, resolved group/index, component, minimum difficulty band.
    mr_uint4 target;
    // lower, upper, auxiliary lower, auxiliary upper.
    mr_float4 parameters;
} MRTaskRandomizationOperatorGPU;

typedef struct MR_ALIGN16 MRTaskImpactEventGPU {
    // Scene-body local index, sequence order, minimum difficulty, global body.
    mr_uint4 binding;
    // Stable tilt, stable seconds, maximum flight seconds, minimum height.
    mr_float4 gate;
    // Projectile collision-envelope radius and reserved values.
    mr_float4 projectile;
} MRTaskImpactEventGPU;

typedef struct MR_ALIGN16 MRTaskInteractionContactGPU {
    // Task contact-group index, pack track index, per-frame feature offset,
    // and fixed compact feature count.
    mr_uint4 binding;
} MRTaskInteractionContactGPU;

typedef struct MR_ALIGN16 MRTaskInteractionSampleGPU {
    // Contact mode, valid feature mask, provenance flags, reserved.
    mr_uint4 metadata;
    // Confidence in x; remaining lanes are reserved.
    mr_float4 confidence;
} MRTaskInteractionSampleGPU;

typedef struct MR_ALIGN16 MRTaskBiasSpecGPU {
    // lower, upper, reserved, reserved.
    mr_float4 range;
    // deterministic random channel and reserved values.
    mr_uint4 metadata;
} MRTaskBiasSpecGPU;

typedef struct MR_ALIGN16 MRTaskStateGPU {
    // episode step, episode index, difficulty band, terrain level.
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
    // Closest clearance, time to closest approach, root-relative strike
    // height, and current barrier value.
    mr_float4 threatGeometry;
    // Joint-CBF correction RMS, keep-out buffer violation, projected actor
    // margin, and urgency margin.
    mr_float4 threatTeacher;
    // Threatened global body, class, latched escape direction encoded as
    // {-1,+1} shifted to {0,2}, and active impact event index.
    mr_uint4 threatMetadata;
} MRTaskStateGPU;

// Compact task-wide physical evidence accumulated on device. It does not own
// a difficulty level or decide whether learning may proceed. Every authored
// difficulty band remains episode-sampleable; this record only publishes
// exposure-normalized outcomes for the evidence ledger.
typedef struct MR_ALIGN16 MRTaskEvidenceStateGPU {
    mr_u64 controlSteps;
    mr_u64 evidenceWindows;
    mr_u32 completedEpisodeCount;
    mr_u32 timeoutEpisodeCount;
    mr_u32 impactContactCount;
    mr_u32 impactCleanMissCount;
    mr_u32 balanceFailureCount;
    float trackingScoreSum;
    mr_u32 lastCompletedEpisodeCount;
    mr_u32 reserved;
    // Contact, clean-miss, and balance-failure rates per million environment
    // steps, followed by mean completed-episode tracking per million.
    mr_uint4 lastWindow;
} MRTaskEvidenceStateGPU;

typedef struct MR_ALIGN16 MRTaskTransitionGPU {
    // reward, tracking score, root height, tilt.
    mr_float4 rewardAndState;
    // done, timeout, physics error, termination reason.
    mr_uint4 termination;
    // task, base, joint-velocity, and joint-acceleration contributions.
    mr_float4 rewardBreakdown0;
    // action-control, posture/limits, energy, and contact contributions.
    mr_float4 rewardBreakdown1;
    // Dodge link-clearance, evasion, miss, and safe-stillness contributions.
    mr_float4 dodgeRewardBreakdown0;
    // Dodge safe-action-rate, CBF correction, CBF buffer, and predicted
    // clearance contributions.
    mr_float4 dodgeRewardBreakdown1;
    mr_u64 policyRevision;
    // V of the accepted post-transition state for timeout bootstrapping.
    float timeoutBootstrapValue;
    // Mean linear tracking score for a non-physics episode ending here.
    float episodeTrackingScore;
    // Episode difficulty band and terrain profile.
    mr_uint4 taskProgress;
} MRTaskTransitionGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRTaskDispatchGPU) == 112u);
static_assert(sizeof(MRTaskProgramHeaderGPU) == 560u);
static_assert(sizeof(MRTaskActionBindingGPU) == 48u);
static_assert(sizeof(MRTaskObservationOperatorGPU) == 48u);
static_assert(sizeof(MRTaskContactGroupGPU) == 112u);
static_assert(sizeof(MRTaskIndexGroupGPU) == 16u);
static_assert(sizeof(MRTaskRewardOperatorGPU) == 48u);
static_assert(sizeof(MRTaskTerminationOperatorGPU) == 48u);
static_assert(sizeof(MRTaskRandomizationOperatorGPU) == 32u);
static_assert(sizeof(MRTaskImpactEventGPU) == 48u);
static_assert(sizeof(MRTaskInteractionContactGPU) == 16u);
static_assert(sizeof(MRTaskInteractionSampleGPU) == 32u);
static_assert(sizeof(MRTaskBiasSpecGPU) == 32u);
static_assert(sizeof(MRTaskStateGPU) == 160u);
static_assert(sizeof(MRTaskEvidenceStateGPU) == 64u);
static_assert(sizeof(MRTaskTransitionGPU) == 128u);
#endif
#endif
