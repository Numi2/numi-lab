#pragma once

#include "metalrobo/engine_types.h"

#define MR_NUMI_HUMAN_STAND_ABI_VERSION 2u
#define MR_NUMI_HUMAN_STAND_MAX_BODIES 192u
#define MR_NUMI_HUMAN_STAND_MAX_DOFS 160u
#define MR_NUMI_HUMAN_STAND_MAX_Q 161u
#define MR_NUMI_HUMAN_STAND_MAX_CONTACTS 32u
#define MR_NUMI_HUMAN_STAND_MAX_STEPS 4096u

enum MRNumiHumanStandStatusCode {
    MR_NUMI_HUMAN_STAND_SUCCESS = 0u,
    MR_NUMI_HUMAN_STAND_INVALID_DISPATCH = 1u,
    MR_NUMI_HUMAN_STAND_INVALID_MODEL = 2u,
    MR_NUMI_HUMAN_STAND_NONFINITE_INPUT = 3u,
    MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED = 4u,
    MR_NUMI_HUMAN_STAND_CONTACT_FAILED = 5u,
    MR_NUMI_HUMAN_STAND_NONFINITE_RESULT = 6u,
    MR_NUMI_HUMAN_STAND_TENDON_TRANSFER_FAILED = 7u,
};

enum MRNumiHumanStandFlags {
    MR_NUMI_HUMAN_STAND_ENABLE_CONTACT = 1u << 0u,
    MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE = 1u << 1u,
    MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS = 1u << 2u,
};

// One source-authored support witness. The point-query index addresses the
// same current-pose Jacobian stream consumed by MyoSim force projection.
typedef struct MR_ALIGN16 MRNumiHumanStandContactGPU {
    mr_u32 bodyIndex;
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;

    // x = Coulomb friction, y = activation slop metres,
    // z = normal stabilization fraction, w reserved.
    mr_float4 frictionSlopAndStabilization;
} MRNumiHumanStandContactGPU;

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
    mr_u32 reserved0;

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
} MRNumiHumanStandStatusGPU;

#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRNumiHumanStandContactGPU) == 32);
static_assert(sizeof(MRNumiHumanStandDispatchGPU) == 160);
static_assert(sizeof(MRNumiHumanStandStatusGPU) == 96);
#endif
