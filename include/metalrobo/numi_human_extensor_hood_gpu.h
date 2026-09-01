#pragma once

#include "metalrobo/engine_types.h"

#define MR_NUMI_HUMAN_EXTENSOR_HOOD_GPU_ABI_VERSION 1u
#define MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_RAY_NODES 12u
#define MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_RAY_ELEMENTS 14u
#define MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_RAY_INPUTS 5u
#define MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_FREE_DIMENSION 27u
#define MR_NUMI_HUMAN_EXTENSOR_HOOD_MAX_STEPS 4096u

enum MRNumiHumanExtensorHoodNodeFlags : mr_u32 {
    MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED = 1u << 0u,
};

enum MRNumiHumanExtensorHoodStatus : mr_u32 {
    MR_NUMI_HUMAN_EXTENSOR_HOOD_SUCCESS = 0u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_DISPATCH = 1u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_TOPOLOGY = 2u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_ROUTE_CUT = 3u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_INVALID_MUSCLE_RESULT = 4u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_SINGULAR_SYSTEM = 5u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_DID_NOT_CONVERGE = 6u,
    MR_NUMI_HUMAN_EXTENSOR_HOOD_NONFINITE_RESULT = 7u,
};

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 rayCount;
    mr_u32 nodeCount;

    mr_u32 elementCount;
    mr_u32 inputCount;
    mr_u32 muscleCount;
    mr_u32 siteCount;

    mr_u32 routeNodeCount;
    mr_u32 dofCount;
    mr_u32 bodyPoseStride;
    mr_u32 articulationFirstBody;

    mr_u32 pointJacobianStride;
    mr_u32 bodyJacobianPointOffset;
    mr_u32 generalizedForceStride;
    mr_u32 generalizedForceOffset;

    mr_u32 stepIndex;
    mr_u32 maximumIterations;
    mr_u32 maximumLineSearchSteps;
    mr_u32 wrapCount;

    // x force tolerance [N], y minimum length [m], z diagonal
    // regularization, w Armijo fraction.
    mr_float4 solver;
    // x body-relative fascial foundation stiffness [N/m], yzw zero.
    mr_float4 foundation;
} MRNumiHumanExtensorHoodDispatchGPU;

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodRayGPU {
    mr_uint4 nodes;    // offset, count, side, digit
    mr_uint4 elements; // offset, count, input offset, input count
} MRNumiHumanExtensorHoodRayGPU;

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodNodeGPU {
    mr_u32 bodyIndex;
    mr_u32 flags;
    mr_u32 role;
    mr_u32 sourceSiteIndex;
    // COM-relative point in the owning Core body. For a free node this is a
    // deterministic pose-dependent initializer, not a kinematic constraint.
    mr_float4 localPoint;
} MRNumiHumanExtensorHoodNodeGPU;

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodElementGPU {
    mr_u32 nodeA;
    mr_u32 nodeB;
    mr_u32 bundle;
    mr_u32 reserved0;
    // x rest length [m], y Young modulus [Pa], z area [m^2], w reserved zero.
    mr_float4 material;
} MRNumiHumanExtensorHoodElementGPU;

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodInputGPU {
    mr_u32 nodeIndex;
    mr_u32 muscleIndex;
    mr_u32 proximalBodyIndex;
    // First source route site whose outgoing share is replaced.
    mr_u32 routeNodeOrdinal;
    // Exact hood-input source site. Kept separate because the cut begins at
    // the preceding named site and may cross a wrap before reaching it.
    mr_u32 targetRouteNodeOrdinal;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_float4 proximalLocalPoint;
} MRNumiHumanExtensorHoodInputGPU;

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodNodeResultGPU {
    mr_float4 position;
    // Force applied to the owning rigid body. Fixed nodes contain the hood's
    // anchor load; loaded free nodes contain the force opposite the proximal
    // muscle load. Other nodes are zero.
    mr_float4 bodyForce;
} MRNumiHumanExtensorHoodNodeResultGPU;

typedef struct MR_ALIGN16 MRNumiHumanExtensorHoodRayResultGPU {
    mr_u32 status;
    mr_u32 environment;
    mr_u32 rayIndex;
    mr_u32 completedIterations;
    // xyz force closure [N], w maximum free-node residual [N].
    mr_float4 forceClosureAndMaximumResidual;
    // xyz moment closure about world origin [N m], w maximum tension [N].
    mr_float4 momentClosureAndMaximumTension;
    // x strain energy [J], y active element count, z maximum free-node
    // displacement [m], w maximum active engineering strain.
    mr_float4 energyAndCounts;
    // x is one only after the enclosing stand step accepted this ray.
    mr_float4 transaction;
} MRNumiHumanExtensorHoodRayResultGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRNumiHumanExtensorHoodDispatchGPU) == 112u);
static_assert(sizeof(MRNumiHumanExtensorHoodRayGPU) == 32u);
static_assert(sizeof(MRNumiHumanExtensorHoodNodeGPU) == 32u);
static_assert(sizeof(MRNumiHumanExtensorHoodElementGPU) == 32u);
static_assert(sizeof(MRNumiHumanExtensorHoodInputGPU) == 48u);
static_assert(sizeof(MRNumiHumanExtensorHoodNodeResultGPU) == 32u);
static_assert(sizeof(MRNumiHumanExtensorHoodRayResultGPU) == 80u);
#endif
