#pragma once

// Typed device ABI for a source-materialized Millard 2012 equilibrium muscle
// reference. The program consumes the private pose, point-world, and analytic
// point-Jacobian streams emitted by the generic articulated operator in the
// same Metal command buffer. MetalWorld admits this program only for its
// bounded fixed-root FunctionBased dense-dynamics path.

#include "metalrobo/engine_types.h"

#define MR_MILLARD_REFERENCE_GPU_ABI_VERSION 2u
#define MR_MILLARD_ACTIVATION_GPU_ABI_VERSION 2u
#define MR_MILLARD_REFERENCE_MAX_WRAPS_PER_MUSCLE 16u

enum MRMillardReferenceGPUStatus : mr_u32 {
    MR_MILLARD_REFERENCE_SUCCESS = 0u,
    MR_MILLARD_REFERENCE_INVALID_PROGRAM = 1u,
    MR_MILLARD_REFERENCE_INVALID_STATE = 2u,
    MR_MILLARD_REFERENCE_INVALID_PATH = 3u,
    MR_MILLARD_REFERENCE_UNBRACKETED = 4u,
    MR_MILLARD_REFERENCE_NONFINITE_RESULT = 5u,
};

typedef struct MR_ALIGN16 MRMillardReferenceDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 muscleCount;
    mr_u32 pathPointCount;
    mr_u32 wrapCount;

    mr_u32 environmentCount;
    mr_u32 dofCount;
    mr_u32 pointWorldStride;
    mr_u32 pointJacobianStride;

    mr_u32 bodyPoseStride;
    mr_u32 articulationFirstBody;
    mr_u32 flags;
    mr_u32 reserved0;
} MRMillardReferenceDispatchGPU;

// Immutable source geometry/parameters for one Millard muscle. Path and wrap
// offsets address the companion pointer-free arrays below. The source curves
// are one record per muscle at the matching index.
typedef struct MR_ALIGN16 MRMillardMuscleGPU {
    // max isometric force, optimal fiber length, tendon slack length,
    // pennation angle at optimal fiber length.
    mr_float4 forceAndLengths;
    // fiber damping, default activation, minimum activation, reserved zero.
    mr_float4 dampingAndActivation;
    // path point offset/count and cylinder-wrap offset/count.
    mr_uint4 pathAndWrap;
    // Only bit zero is currently retained from the source payload
    // (ignore-tendon-compliance); the static source-reference solve still
    // reports the authored flag instead of silently changing parameters.
    mr_uint4 flags;
} MRMillardMuscleGPU;

// Mutable per-environment muscle input. The reference uses the supplied
// activation and normalized fiber velocity and solves fiber length from static
// tendon/fiber equilibrium on device. Unused lanes must be zero.
typedef struct MR_ALIGN16 MRMillardMuscleStateGPU {
    mr_float4 activationAndVelocity;
} MRMillardMuscleStateGPU;

// Per-submission activation-control contract. Excitations are packed
// [control step][environment][muscle]; the reference state remains packed
// [environment][muscle]. The source importer owns both time constants.
typedef struct MR_ALIGN16 MRMillardActivationDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 muscleCount;
    mr_u32 environmentCount;
    // Zero reads caller-packed [control][environment][muscle] excitations.
    // Bit zero reads the filtered native task action history instead, mapping
    // its conventional [-1, 1] action range to normalized excitation [0, 1].
    mr_u32 flags;
    // x control-period seconds; y activation tau; z deactivation tau;
    // w must be zero.
    mr_float4 timestepAndTimeConstants;
} MRMillardActivationDispatchGPU;

#define MR_MILLARD_ACTIVATION_FROM_NATIVE_TASK 1u

// A point query that must correspond exactly to an articulated-operator point
// query in the enclosing batch. The local coordinate itself remains in the
// operator point buffer, preventing a duplicate host-side kinematics path.
typedef struct MR_ALIGN16 MRMillardPathPointGPU {
    mr_u32 pointQueryIndex;
    mr_u32 bodyIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMillardPathPointGPU;

// The 22 scalar OpenSim source curve parameters are packed in source order:
// five active-force-length, eight force-velocity, five passive-fiber, and four
// tendon-force-length values. The final two lanes are required zero.
typedef struct MR_ALIGN16 MRMillardSourceCurveGPU {
    mr_float4 values[6];
} MRMillardSourceCurveGPU;

enum MRMillardPathWrapMethod : mr_u32 {
    MR_MILLARD_PATH_WRAP_HYBRID = 0u,
    MR_MILLARD_PATH_WRAP_MIDPOINT = 1u,
    MR_MILLARD_PATH_WRAP_AXIAL = 2u,
};

typedef struct MR_ALIGN16 MRMillardCylinderWrapGPU {
    mr_u32 bodyIndex;
    // OpenSim PathWrap range endpoints are source 1-based path-point indices;
    // -1 means use the corresponding end of the source path.
    mr_i32 startPoint;
    mr_i32 endPoint;
    mr_u32 method;
    mr_float4 center;
    // xyz uses the source BodyFixedXYZ cylinder rotation; w is radius.
    mr_float4 rotationAndRadius;
    // x is finite cylinder length; yzw are required zero.
    mr_float4 length;
} MRMillardCylinderWrapGPU;

// One static-equilibrium result per environment/muscle. generalizedForces is
// stored in a separate dense [environment][muscle][dof] output so callers can
// preserve individual actuator provenance before any downstream reduction.
typedef struct MR_ALIGN16 MRMillardMuscleResultGPU {
    mr_u32 status;
    mr_u32 environment;
    mr_u32 muscleIndex;
    mr_u32 appliedCylinderWrapCount;
    // path length, solved fiber length, tendon force, equilibrium residual.
    mr_float4 pathFiberTendonResidual;
} MRMillardMuscleResultGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMillardReferenceDispatchGPU) == 48u);
static_assert(alignof(MRMillardReferenceDispatchGPU) == 16u);
static_assert(sizeof(MRMillardMuscleGPU) == 64u);
static_assert(alignof(MRMillardMuscleGPU) == 16u);
static_assert(sizeof(MRMillardMuscleStateGPU) == 16u);
static_assert(sizeof(MRMillardActivationDispatchGPU) == 32u);
static_assert(alignof(MRMillardActivationDispatchGPU) == 16u);
static_assert(sizeof(MRMillardPathPointGPU) == 16u);
static_assert(sizeof(MRMillardSourceCurveGPU) == 96u);
static_assert(sizeof(MRMillardCylinderWrapGPU) == 64u);
static_assert(sizeof(MRMillardMuscleResultGPU) == 32u);
#endif
