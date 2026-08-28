#pragma once

// Typed device ABI for MyoSim's MuJoCo ``general`` muscle reference.  The
// source model uses this force law and sphere/cylinder spatial tendon wraps,
// so it intentionally does not share the OpenSim Millard sidecar ABI.

#include "metalrobo/engine_types.h"

#define MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION 4u
#define MR_MUJOCO_MUSCLE_ACTIVATION_GPU_ABI_VERSION 2u
#define MR_MUJOCO_MUSCLE_ACTIVE_FORCE_GPU_ABI_VERSION 1u

enum MRMujocoMuscleReferenceGPUStatus : mr_u32 {
    MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS = 0u,
    MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PROGRAM = 1u,
    MR_MUJOCO_MUSCLE_REFERENCE_INVALID_STATE = 2u,
    MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PATH = 3u,
    MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT = 4u,
};

enum MRMujocoMuscleRouteNodeType : mr_u32 {
    MR_MUJOCO_MUSCLE_ROUTE_SITE = 1u,
    MR_MUJOCO_MUSCLE_ROUTE_SPHERE = 2u,
    MR_MUJOCO_MUSCLE_ROUTE_CYLINDER = 3u,
};

typedef struct MR_ALIGN16 MRMujocoMuscleReferenceDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 muscleCount;
    mr_u32 siteCount;
    mr_u32 wrapCount;

    mr_u32 routeNodeCount;
    mr_u32 environmentCount;
    mr_u32 bodyPoseStride;
    mr_u32 articulationFirstBody;

    // Articulation-local generalized-coordinate count and the enclosing
    // articulated operator's per-environment Jacobian stride. The source
    // route kernel projects every current path segment through four supplied
    // point-Jacobian probes per body: COM, local +X, +Y, and +Z.
    mr_u32 dofCount;
    mr_u32 pointJacobianStride;
    mr_u32 bodyJacobianPointOffset;
    mr_u32 bodyJacobianPointStride;
    // x integration timestep used by NHMYO2 backward-Euler fibre equilibrium;
    // yzw required zero. A nonpositive x keeps the compliant state static.
    mr_float4 timestepSecondsAndReserved;
} MRMujocoMuscleReferenceDispatchGPU;

// Immutable source program. Every three parameter blocks retain the ten
// source MuJoCo gain/bias/dynamics values followed by explicit zero padding.
typedef struct MR_ALIGN16 MRMujocoMuscleGPU {
    // x route-node offset; y route-node count; zw reserved zero.
    mr_uint4 route;
    // x/y tendon length range; z acceleration scale; w reserved zero.
    mr_float4 lengthRangeAndAcceleration;
    // x/y control range; zw reserved zero. Retained source metadata.
    mr_float4 controlRange;
    mr_float4 gainParameters[3];
    mr_float4 biasParameters[3];
    mr_float4 dynamicParameters[3];
    // NHMYO2 compliant architecture. All-zero preserves the legacy inelastic
    // MuJoCo path. Otherwise: x L0, y LT, z tendon strain at normalized force
    // one, w tendon stiffness there.
    mr_float4 compliantArchitecture0;
    // x normalized force at toe end, y source-default curviness, z normalized
    // fibre damping, w offline static force-surface fit NRMSE.
    mr_float4 compliantArchitecture1;
} MRMujocoMuscleGPU;

typedef struct MR_ALIGN16 MRMujocoMuscleSiteGPU {
    mr_u32 bodyIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
    // xyz COM-relative Core body coordinate; w required zero.
    mr_float4 localPoint;
} MRMujocoMuscleSiteGPU;

typedef struct MR_ALIGN16 MRMujocoMuscleWrapGPU {
    mr_u32 bodyIndex;
    mr_u32 type;
    mr_u32 reserved0;
    mr_u32 reserved1;
    // xyz COM-relative Core body coordinate; w required zero.
    mr_float4 localCenter;
    // Row-major geometry-to-Core-body rotation; w of each row required zero.
    mr_float4 rotationRow0;
    mr_float4 rotationRow1;
    mr_float4 rotationRow2;
    // x radius metres; yzw required zero.
    mr_float4 radius;
} MRMujocoMuscleWrapGPU;

typedef struct MR_ALIGN16 MRMujocoMuscleRouteNodeGPU {
    mr_u32 type;
    // Site index for site nodes and wrap index otherwise.
    mr_u32 targetIndex;
    // Optional side-site index, or MR_INVALID_INDEX.
    mr_u32 sideSiteIndex;
    mr_u32 reserved0;
} MRMujocoMuscleRouteNodeGPU;

// State is environment-major. x excitation, y activation, z accepted fibre
// length (m; zero requests deterministic zero-velocity fibre/tendon
// equilibrium at the current path and activation), w fibre velocity (m/s).
// The reference pass publishes a candidate in its result and the sidecar step
// commits it together with activation.
typedef struct MR_ALIGN16 MRMujocoMuscleStateGPU {
    mr_float4 excitationAndActivation;
} MRMujocoMuscleStateGPU;

// One explicit device-side activation update follows the reference pass. It
// intentionally owns only the mutable activation sidecar: source force and
// tendon-path evaluation remain in MRMujocoMuscleReferenceDispatchGPU.
typedef struct MR_ALIGN16 MRMujocoMuscleActivationDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 stateCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
    // x timestep seconds; yzw required zero.
    mr_float4 timestepSecondsAndReserved;
} MRMujocoMuscleActivationDispatchGPU;

// Requests the activation-dependent component of each already-evaluated
// source row. This is used by the Human standing runtime because the imported
// passive bias is not an equilibrium preload at the registered v1 pose.
typedef struct MR_ALIGN16 MRMujocoMuscleActiveForceDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 muscleCount;
    mr_u32 environmentCount;
    mr_u32 dofCount;
} MRMujocoMuscleActiveForceDispatchGPU;

typedef struct MR_ALIGN16 MRMujocoMuscleResultGPU {
    mr_u32 status;
    mr_u32 environment;
    mr_u32 muscleIndex;
    mr_u32 appliedWrapCount;
    // x path length; y J(q)*v path velocity; z actuator force; w activation derivative.
    mr_float4 pathForceAndActivationDerivative;
    // Exact world-space d(length)/d(endpoint position) used by the owning
    // wrapped route evaluation: origin first, insertion second. w is zero.
    mr_float4 endpointLengthGradients[2];
    // x force represented by the currently published generalized-force row.
    // It is total source force after route evaluation and active-only force
    // after the optional Human active-force transform. yzw are zero.
    mr_float4 activeForceAndReserved;
    // x candidate fibre length; y candidate fibre velocity; z positive tendon
    // tension; w normalized damped-equilibrium residual.
    mr_float4 fiberStateTendonForceResidual;
} MRMujocoMuscleResultGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMujocoMuscleReferenceDispatchGPU) == 64u);
static_assert(sizeof(MRMujocoMuscleGPU) == 224u);
static_assert(sizeof(MRMujocoMuscleSiteGPU) == 32u);
static_assert(sizeof(MRMujocoMuscleWrapGPU) == 96u);
static_assert(sizeof(MRMujocoMuscleRouteNodeGPU) == 16u);
static_assert(sizeof(MRMujocoMuscleStateGPU) == 16u);
static_assert(sizeof(MRMujocoMuscleActivationDispatchGPU) == 32u);
static_assert(sizeof(MRMujocoMuscleActiveForceDispatchGPU) == 16u);
static_assert(sizeof(MRMujocoMuscleResultGPU) == 96u);
#endif
