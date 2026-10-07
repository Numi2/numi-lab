#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_timed_root_force.h"

#define MR_NUMI_HUMAN_STAND_ABI_VERSION 15u
// Six spatial Jacobian rows plus three cached world-inertia products.
#define MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS 9u
#define MR_NUMI_HUMAN_STAND_MAX_BODIES 192u
#define MR_NUMI_HUMAN_STAND_MAX_DOFS 160u
// Prefix of the opt-in hybrid equality factor cache; its tail remains in the arena.
#define MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_ELEMENTS 7800u
#define MR_NUMI_HUMAN_STAND_MAX_Q 161u
#define MR_NUMI_HUMAN_STAND_MAX_CONTACTS 32u
#define MR_NUMI_HUMAN_STAND_MAX_STEPS 4096u
// Total accepted horizon across bounded submissions. This is not permission
// to encode this many steps in one command buffer.
#define MR_NUMI_HUMAN_STAND_MAX_HORIZON_STEPS 1000000u

enum MRNumiHumanStandStatusCode {
    MR_NUMI_HUMAN_STAND_SUCCESS = 0u,
    MR_NUMI_HUMAN_STAND_INVALID_DISPATCH = 1u,
    MR_NUMI_HUMAN_STAND_INVALID_MODEL = 2u,
    MR_NUMI_HUMAN_STAND_NONFINITE_INPUT = 3u,
    MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED = 4u,
    MR_NUMI_HUMAN_STAND_CONTACT_FAILED = 5u,
    MR_NUMI_HUMAN_STAND_NONFINITE_RESULT = 6u,
    MR_NUMI_HUMAN_STAND_TENDON_TRANSFER_FAILED = 7u,
    MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED = 8u,
    // A borrowed external-physics program rejected the candidate Human step.
    // failingIndex identifies the external object when one is available.
    MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED = 9u,
};

enum MRNumiHumanStandFlags {
    MR_NUMI_HUMAN_STAND_ENABLE_CONTACT = 1u << 0u,
    MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE = 1u << 1u,
    MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS = 1u << 2u,
    MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES = 1u << 3u,
    // Internal owner prefix: vState names private scratch initialized from
    // accepted v. Produce v_free without advancing q or completedSteps.
    MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY = 1u << 4u,
    MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM = 1u << 5u,
    // Diagnostic only: export final signed source-limit multipliers by DOF
    // after the coupled solve. The default stand path performs no such write.
    MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES = 1u << 6u,
    // Internal split owner phase. Assemble/factor the current source state and
    // leave q/v untouched for a following device-resident solve dispatch.
    MR_NUMI_HUMAN_STAND_PREPARE_ONLY = 1u << 7u,
    // Split preparation may publish spatial Jacobians/bias before a
    // grid-parallel mass assembly, then factor the exact assembled matrix.
    MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY = 1u << 8u,
    MR_NUMI_HUMAN_STAND_MASS_READY = 1u << 9u,
    // Split preparation: factor the assembled mass once, solve independent
    // response columns on a grid, then condition their shared constraints.
    MR_NUMI_HUMAN_STAND_FACTOR_ONLY = 1u << 10u,
    MR_NUMI_HUMAN_STAND_RESPONSES_READY = 1u << 11u,
    // Internal split-finish handshake. FREE_ONLY publishes the current
    // unconstrained velocity without advancing physical state; FREE_READY
    // consumes exactly that velocity in the next dispatch of the same owner.
    MR_NUMI_HUMAN_STAND_FREE_ONLY = 1u << 12u,
    MR_NUMI_HUMAN_STAND_FREE_READY = 1u << 13u,
    MR_NUMI_HUMAN_STAND_CPU_FINISH_READY = 1u << 14u,
    // Internal same-command-buffer handoff: projected mass responses were
    // solved against the current mass factor before this Metal dispatch.
    MR_NUMI_HUMAN_STAND_PROJECTED_RAW_READY = 1u << 15u,
    // Optional response-only R^T A R solve. The source equality factor and
    // ordered physical finish remain authoritative.
    MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES = 1u << 16u,
    // The selected source operator uses the upper triangle for sparse
    // Cholesky and the lower triangle for dense Cholesky.
    MR_NUMI_HUMAN_STAND_REDUCED_SOURCE_UPPER_TRIANGLE = 1u << 17u,
    // Read-only per-root evidence in status.flags: operator factored and a
    // projected response actually used the reduced path.
    MR_NUMI_HUMAN_STAND_REDUCED_PROJECTION_READY = 1u << 18u,
    MR_NUMI_HUMAN_STAND_REDUCED_PROJECTION_USED = 1u << 19u,
    // The six source spatial-J rows are already present in the first six
    // rows of spatial scratch; retain the existing point-J fallback by default.
    MR_NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS = 1u << 20u,
    // Optional accurate body-ordered reductions; the physical terms are unchanged.
    MR_NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM = 1u << 21u,
    // Status-only proof that the opt-in per-sweep equality diagnostic replay
    // was selected for this finish. This is never an input dispatch flag.
    MR_NUMI_HUMAN_STAND_DEFERRED_EQUALITY_DIAGNOSTICS_USED = 1u << 22u,
    // Status-only proof that equality factorization used the hybrid cache.
    MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_USED = 1u << 23u,
    // Accepted prior contact history matched the static source basis and was
    // consumed as an initial impulse guess in this finish.
    MR_NUMI_HUMAN_STAND_CONTACT_WARMSTART_HIT = 1u << 24u,
    // The accepted guess changed the cold source seed and was applied to v.
    MR_NUMI_HUMAN_STAND_CONTACT_WARMSTART_USED = 1u << 25u,
    // Status-only receipt that the cooperative projected-PGS residual met its
    // requested threshold before the configured sweep limit.
    MR_NUMI_HUMAN_STAND_PGS_EARLY_EXIT_USED = 1u << 26u,
    // Status-only proof that a bilateral base refinement used the validated
    // reduced-coordinate correction rather than the full Schur fallback.
    MR_NUMI_HUMAN_STAND_REDUCED_BASE_PROJECTION_USED = 1u << 27u,
    // Status-only receipt that the opt-in path initially omitted legacy
    // equality responses and the full Schur factor.
    MR_NUMI_HUMAN_STAND_REDUCED_EQUALITY_DATA_DEFERRED_USED = 1u << 28u,
    // Split dispatch option and fallback-stage markers. These are submitted
    // only after the initial status dispatch; the status receipt above is
    // published separately by the GPU owner.
    MR_NUMI_HUMAN_STAND_DEFERRED_EQUALITY_DATA = 1u << 29u,
    MR_NUMI_HUMAN_STAND_REBUILD_DEFERRED_EQUALITY_DATA = 1u << 30u,
    MR_NUMI_HUMAN_STAND_DEFERRED_EQUALITY_FALLBACK_USED = 1u << 30u,
    MR_NUMI_HUMAN_STAND_DEFERRED_RESPONSE_FALLBACK_ONLY = 1u << 31u,
};

// One source-authored support witness. The point-query index addresses the
// same current-pose Jacobian stream consumed by MyoSim force projection.
typedef struct MR_ALIGN16 MRNumiHumanStandContactGPU {
    mr_u32 bodyIndex;
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;

    // x = Coulomb friction, y = source activation and warm-start slop
    // metres (operator-level speculative admission may be wider),
    // z = normal stabilization fraction, w = source static normal
    // support force in N. A zero force retains cold-start contact.
    mr_float4 frictionSlopAndStabilization;
} MRNumiHumanStandContactGPU;

// Accepted split-finish contact impulses are retained only after the
// existing accepted-state reconciliation gate. A request identity is transient
// submission input and is kept outside this accepted record so diagnostics can
// compare exactly the committed history bytes.
typedef struct MR_ALIGN16 MRNumiHumanStandContactWarmStartGPU {
    mr_uint4 identity;
    // x = support-contact count, y = valid, z = last accepted completed
    // step count (the required predecessor for dispatch.stepIndex), w reserved.
    mr_uint4 metadata;
    // [normal, tangent0, tangent1] in the prior accepted root's basis.
    mr_float4 impulses[MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
} MRNumiHumanStandContactWarmStartGPU;

typedef struct MR_ALIGN16 MRNumiHumanStandContactWarmStartSlotGPU {
    MRNumiHumanStandContactWarmStartGPU accepted;
    mr_uint4 requestedIdentity;
    // x = request enabled, y/z/w reserved.
    mr_uint4 requestState;
} MRNumiHumanStandContactWarmStartSlotGPU;

typedef struct MR_ALIGN16 MRNumiHumanStandDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 articulationIndex;
    mr_u32 stepIndex;

    mr_u32 stepCount;
    mr_u32 bodyJacobianPointOffset;
    mr_u32 supportContactCount;
    mr_u32 flags;

    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 pointWorldStride;
    mr_u32 pointJacobianStride;

    mr_u32 bodyPoseStride;
    mr_u32 generalizedForceStride;
    mr_u32 generalizedForceOffset;
    mr_u32 contactIterationCount;

    mr_u32 tendonEndpointCount;
    mr_u32 tendonEnvelopeCount;
    mr_u32 tendonTransferStride;
    mr_u32 jointEqualityCount;

    // xyz = ground point, w = timestep seconds.
    mr_float4 groundPointAndTimestep;
    // xyz = normalized ground normal, w reserved.
    mr_float4 groundNormal;
    // xyz = target floating-root position, w reserved.
    mr_float4 targetRootPosition;
    // xyzw = target floating-root orientation.
    mr_float4 targetRootOrientation;
    // linear stiffness, linear damping, angular stiffness, angular damping.
    mr_float4 assistanceGains;
    // Immutable external disturbance, independent of root assistance/state.
    MRNumiHumanTimedRootForceGPU timedRootForce;
} MRNumiHumanStandDispatchGPU;

typedef struct MR_ALIGN16 MRNumiHumanStandStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 completedSteps;
    mr_u32 failingIndex;

    mr_u32 activeContactCount;
    mr_u32 maximumActiveContactCount;
    mr_u32 contactIterations;
    mr_u32 flags;

    // minimum plane gap, maximum penetration, total normal impulse,
    // maximum generalized acceleration.
    mr_float4 contactAndAcceleration;
    // minimum Cholesky pivot, maximum Cholesky pivot,
    // root-assistance force norm, root-assistance torque norm.
    mr_float4 factorAndAssistance;

    // Cumulative accepted endpoint transactions across completed steps.
    mr_u32 tendonTransferCount;
    mr_u32 tendonEnvelopeTransferCount;
    mr_u32 tendonPointTransferCount;
    mr_u32 tendonFailureCount;

    // Maximum force residual, source-point moment residual, generalized
    // wrench-equivalence correction, and represented actuator-force norm.
    mr_float4 tendonDiagnostics;

    // active row count, maximum active row count, failed row count, and the
    // articulation-local velocity DOF with the maximum pre-projection
    // acceleration. The final field is MR_INVALID_INDEX when no step ran.
    mr_uint4 jointEqualityCounts;
    // Maximum pre-projection position error, maximum constrained velocity
    // error, maximum absolute bilateral impulse, and sum absolute impulses.
    mr_float4 jointEqualityDiagnostics;
    // The final exact-coordinate projection can overwrite the post-solve
    // dependent position and velocity.  Keep that correction separate from
    // the constraint solve itself: x/y are the maximum/sum absolute position
    // overwrites (m or rad), and z/w are the maximum/sum absolute velocity
    // overwrites (m/s or rad/s), accumulated across accepted steps.
    mr_float4 jointEqualityProjectionDiagnostics;
    // Residuals immediately before the final exact equality projection.
    // x/y/z are the maximum normal-contact, source position-limit, and
    // equality target velocity residuals; w is their maximum. Contact and
    // limit rows use the pre-step Jacobian, gap, and source position.
    mr_float4 preProjectionPreStepConstraintDiagnostics;
    // The corresponding residuals immediately after the final exact equality
    // projection. This is a linearized diagnostic, not a post-step contact
    // query; comparing it with the preceding vector isolates that overwrite.
    mr_float4 postProjectionPreStepConstraintDiagnostics;
    // Maximum normal-contact impulse, maximum tangential-contact impulse,
    // maximum absolute source-limit impulse, and cumulative absolute
    // source-limit impulse across accepted steps. These are diagnostic-only:
    // they do not participate in the solve or warm start.
    mr_float4 constraintImpulseDiagnostics;
    // Source owners for x/y/z above: support-contact index, support-contact
    // index, source velocity DOF, then equality record with the maximum
    // absolute bilateral impulse. MR_INVALID_INDEX means no nonzero owner.
    mr_uint4 constraintImpulseOwners;

    // Signed generalized impulse work accumulated across accepted steps.
    // x/y split normal and tangential contact work, z is bilateral equality
    // work (including equality reactions induced by source limits), and w is
    // direct source-limit work. Each impulse update uses the trapezoidal row
    // velocity, 0.5 * delta_lambda * (velocity_before + velocity_after), so
    // the sum is the corresponding quadratic-energy change of the effective
    // response operator in joules. This is not by itself physical kinetic
    // energy when the response includes implicit passive terms.
    // The final exact coordinate projection is an overwrite, not an owned
    // impulse, and therefore remains outside this diagnostic.
    mr_float4 constraintImpulseWorkDiagnostics;
    // Sum of the absolute per-update work contributions for x/y/z/w above.
    // Keeping activity as well as the signed result prevents cancellation
    // across coupled sweeps or accepted steps from hiding a large reaction.
    mr_float4 constraintImpulseAbsoluteWorkDiagnostics;

    // Keep smooth dynamics and velocity-level constraint corrections separate.
    // x = maximum unconstrained force acceleration before contact/equalities/
    // limits; y = maximum constraint-induced delta-v before the exact equality
    // projection; z = maximum total pre-projection delta-v from the accepted
    // state; w = maximum final published delta-v after exact projection.
    mr_float4 velocityDiagnostics;
    // Articulation-local velocity DOFs owning x/y/z/w above. An invalid owner
    // means no accepted physical step populated that diagnostic.
    mr_uint4 velocityDiagnosticOwners;
} MRNumiHumanStandStatusGPU;

#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRNumiHumanStandContactGPU) == 32);
static_assert(sizeof(MRNumiHumanStandContactWarmStartGPU) == 544);
static_assert(sizeof(MRNumiHumanStandContactWarmStartSlotGPU) == 576);
static_assert(sizeof(MRNumiHumanStandDispatchGPU) == 192);
static_assert(sizeof(MRNumiHumanStandStatusGPU) == 272);
#endif
