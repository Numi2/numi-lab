#pragma once

#include "metalrobo/gpu_types.h"

#define NUMI_PLASTIC_BAG_GPU_ABI 14u
#define PLASTIC_BAG_COIN_COUNT 10u
#define PLASTIC_BAG_REEDED_EDGE_FRICTION 0.65f

typedef struct MR_ALIGN16 PlasticBagNodeGPU {
    mr_float4 positionAndInverseMass;
    mr_float4 velocityAndMass;
    mr_float4 previousAndFixed;
    mr_float4 restPosition;
    mr_float4 contactPositionCorrection;
    // Kahan world-position rounding remainder (subtract from high position).
    mr_float4 positionRemainder;
} PlasticBagNodeGPU;

typedef struct MR_ALIGN16 PlasticBagEdgeGPU {
    mr_uint4 nodes;
    // x rest length, y XPBD compliance, z accumulated lambda, w strain viscosity time (s).
    mr_float4 material;
} PlasticBagEdgeGPU;

typedef struct MR_ALIGN16 PlasticBagHingeGPU {
    // edge endpoints, two opposite vertices; rest angle, compliance, lambda, viscosity time.
    mr_uint4 nodes;
    mr_float4 material;
} PlasticBagHingeGPU;

typedef struct MR_ALIGN16 PlasticBagEdgeScratchGPU {
    mr_float4 correctionA;
    mr_float4 correctionB;
} PlasticBagEdgeScratchGPU;

typedef struct MR_ALIGN16 PlasticBagNodeEdgeRefGPU {
    // x edge index, y endpoint (0=A, 1=B).
    mr_uint4 identity;
} PlasticBagNodeEdgeRefGPU;

typedef struct MR_ALIGN16 PlasticBagNodeSpanGPU {
    mr_uint4 range;
} PlasticBagNodeSpanGPU;

typedef struct MR_ALIGN16 PlasticBagTriangleGPU {
    mr_uint4 nodes;
    // Current and previous bounding spheres for swept coin-film candidate culling.
    mr_float4 centerAndRadius;
    mr_float4 previousCenterAndRadius;
} PlasticBagTriangleGPU;

typedef struct MR_ALIGN16 PlasticBagNodeTriangleRefGPU {
    mr_uint4 identity;
} PlasticBagNodeTriangleRefGPU;

typedef struct MR_ALIGN16 PlasticBagLapRowGPU {
    // Up to four axial neighbors and the center node.
    mr_uint4 nodes0;
    mr_uint4 nodes1;
    mr_float4 coefficients;
    mr_float4 coefficient4NormalX;
    mr_float4 normalYArea;
} PlasticBagLapRowGPU;

typedef struct MR_ALIGN16 PlasticBagNodeLapRefGPU {
    // x row index, y bitcast signed row coefficient, z/w unused.
    mr_uint4 identity;
} PlasticBagNodeLapRefGPU;

typedef struct MR_ALIGN16 PlasticBagCoinGPU {
    mr_float4 positionAndInverseMass;
    mr_float4 velocityAndMass;
    mr_float4 previousPosition;
    // Local +Y is the disk normal; quaternion is local-to-world (xyzw).
    mr_float4 orientation;
    mr_float4 angularVelocity;
    mr_float4 previousOrientation;
    mr_float4 predictedPosition;
    mr_float4 predictedOrientation;
    mr_float4 predictedVelocity;
    mr_float4 predictedAngularVelocity;
    mr_float4 positionRemainder;
} PlasticBagCoinGPU;

typedef struct MR_ALIGN16 PlasticBagContactGPU {
    mr_float4 correctionA;
    mr_float4 correctionB;
    mr_float4 correctionC;
    mr_float4 correctionCoin;
    // Coin/film position pass stores a world rotation correction here.
    mr_float4 angularVelocityChangeCoin;
    mr_float4 normalAndActive;
    mr_float4 barycentricAndDepth;
    // x selected triangle index; y/z/w reserved.
    mr_uint4 identity;
} PlasticBagContactGPU;

typedef struct MR_ALIGN16 PlasticBagSeamContactGPU {
    // x/y side-seam node indices; w is one when unilateral contact is active.
    mr_uint4 nodesAndActive;
    // x/y endpoint barycentrics, w pre-projection penetration depth.
    mr_float4 barycentricAndDepth;
    mr_float4 correctionA;
    mr_float4 correctionB;
    mr_float4 correctionCoin;
    // xyz point from the seam toward the pouch interior; w is reserved.
    mr_float4 normalAndPadding;
} PlasticBagSeamContactGPU;

typedef struct MR_ALIGN16 PlasticBagCoinPairGPU {
    mr_uint4 coins;
} PlasticBagCoinPairGPU;

typedef struct MR_ALIGN16 PlasticBagPairLoadGPU {
    mr_float4 normalImpulse;
    mr_float4 lambda[4];
} PlasticBagPairLoadGPU;

typedef struct MR_ALIGN16 PlasticBagCoinPairScratchGPU {
    mr_float4 correctionA;
    mr_float4 correctionB;
    mr_float4 angularVelocityChangeA;
    mr_float4 angularVelocityChangeB;
} PlasticBagCoinPairScratchGPU;

typedef struct MR_ALIGN16 PlasticBagCoinPairRefGPU {
    mr_uint4 identity;
} PlasticBagCoinPairRefGPU;

// Derived within one velocity phase only. Positions/orientations do not change
// during its interleaved pair/film/table impulse sweeps. Never checkpointed.
typedef struct MR_ALIGN16 PlasticBagPairVelocityGeometryGPU {
    mr_float4 normal;
    mr_float4 tangent1;
    mr_float4 tangent2;
    mr_float4 pointsAndGap[4];
    mr_uint4 validAndCount;
} PlasticBagPairVelocityGeometryGPU;

typedef struct MR_ALIGN16 PlasticBagConfigGPU {
    // x nodes, y edges, z triangles, w coins.
    mr_uint4 counts;
    // x substep dt, y coin radius, z coin half-thickness, w contact samples.
    mr_float4 scales;
    // xyz gravity, w bending rigidity.
    mr_float4 gravityAndBending;
    // x substep damping, y constraint relaxation, z contact relaxation, w reserved.
    mr_float4 solver;
    // x film half-thickness, y ground height, z candidate capacity, w angular-speed cap.
    mr_float4 geometry;
    // Membrane sweeps per coupled round, hinge sweeps, pair sweeps, reserved.
    mr_uint4 iterations;
} PlasticBagConfigGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(PlasticBagNodeGPU) == 96);
static_assert(sizeof(PlasticBagEdgeGPU) == 32);
static_assert(sizeof(PlasticBagHingeGPU) == 32);
static_assert(sizeof(PlasticBagEdgeScratchGPU) == 32);
static_assert(sizeof(PlasticBagNodeEdgeRefGPU) == 16);
static_assert(sizeof(PlasticBagNodeSpanGPU) == 16);
static_assert(sizeof(PlasticBagTriangleGPU) == 48);
static_assert(sizeof(PlasticBagNodeTriangleRefGPU) == 16);
static_assert(sizeof(PlasticBagLapRowGPU) == 80);
static_assert(sizeof(PlasticBagNodeLapRefGPU) == 16);
static_assert(sizeof(PlasticBagCoinGPU) == 176);
static_assert(sizeof(PlasticBagContactGPU) == 128);
static_assert(sizeof(PlasticBagSeamContactGPU) == 96);
static_assert(sizeof(PlasticBagCoinPairGPU) == 16);
static_assert(sizeof(PlasticBagCoinPairScratchGPU) == 64);
static_assert(sizeof(PlasticBagCoinPairRefGPU) == 16);
static_assert(sizeof(PlasticBagConfigGPU) == 96);
static_assert(sizeof(PlasticBagPairVelocityGeometryGPU) == 128);
#endif
