#pragma once

#include "metalrobo/gpu_types.h"

#define MR_MULTI_CONTACT_ABI_VERSION 2u
#define MR_MULTI_CONTACT_MAX_CONTACTS 32u
#define MR_MULTI_CONTACT_MAX_EQUALITY_ROWS 32u

enum MRMultiContactEndpointKindGPU : mr_u32 {
    MR_MULTI_CONTACT_ARTICULATED = 0u,
    MR_MULTI_CONTACT_SCENE_BODY = 1u,
    MR_MULTI_CONTACT_STATIC_WORLD = 2u,
};

enum MRMultiContactStatusCode : mr_u32 {
    MR_MULTI_CONTACT_SUCCESS = 0u,
    MR_MULTI_CONTACT_INVALID_DISPATCH = 1u,
    MR_MULTI_CONTACT_POINT_JACOBIAN_FAILED = 2u,
    MR_MULTI_CONTACT_INVERSE_MASS_FAILED = 3u,
    MR_MULTI_CONTACT_INVALID_CONTACT = 4u,
    MR_MULTI_CONTACT_QUALITY_FAILED = 5u,
    MR_MULTI_CONTACT_NONFINITE_RESULT = 6u,
    MR_MULTI_CONTACT_EQUALITY_FAILED = 7u,
};

enum MRMultiContactEqualityStatusCode : mr_u32 {
    MR_MULTI_CONTACT_EQUALITY_SUCCESS = 0u,
    MR_MULTI_CONTACT_EQUALITY_INVALID_DISPATCH = 1u,
    MR_MULTI_CONTACT_EQUALITY_INVERSE_MASS_FAILED = 2u,
    MR_MULTI_CONTACT_EQUALITY_INVALID_ROW = 3u,
    MR_MULTI_CONTACT_EQUALITY_FACTORIZATION_FAILED = 4u,
    MR_MULTI_CONTACT_EQUALITY_NONFINITE_RESULT = 5u,
    MR_MULTI_CONTACT_EQUALITY_RESIDUAL_FAILED = 6u,
};

typedef struct MR_ALIGN16 MRMultiContactDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 articulationCount;
    mr_u32 sceneBodyCount;

    mr_u32 articulatedNv;
    mr_u32 totalNv;
    mr_u32 contactCount;
    mr_u32 rowCount;

    mr_u32 inverseWorkCount;
    mr_u32 equalityRowCount;
    mr_u32 responseRowCount;
    mr_u32 reserved0;

    // symmetry tolerance, diagonal tolerance, reserved, reserved.
    mr_float4 tolerances;
    // timestep, maximum stabilization velocity, minimum time-constant ratio,
    // minimum regularization.
    mr_float4 equalityEvaluation0;
    // relative pivot floor, equality residual tolerance, reserved, reserved.
    mr_float4 equalityEvaluation1;
} MRMultiContactDispatchGPU;

// Per-environment contact geometry and exact-cone semantics. Topology lives
// in MRMultiContactEndpointsGPU and is immutable across a cloned batch.
typedef struct MR_ALIGN16 MRMultiContactGPU {
    mr_float4 localPointA;
    mr_float4 localPointB;
    mr_float4 normal;
    mr_float4 tangentU;
    mr_float4 tangentV;
    mr_float4 targetVelocity;
    mr_float4 regularization;
    mr_float4 warmImpulse;
    // x = friction coefficient; remaining lanes reserved.
    mr_float4 friction;
} MRMultiContactGPU;

typedef struct MR_ALIGN16 MRMultiContactEndpointsGPU {
    mr_u32 kindA;
    mr_u32 bodyA;
    mr_u32 sliceA;
    mr_u32 queryA;

    mr_u32 kindB;
    mr_u32 bodyB;
    mr_u32 sliceB;
    mr_u32 queryB;
} MRMultiContactEndpointsGPU;

// Describes one compact articulation-local point-Jacobian stream.
typedef struct MR_ALIGN16 MRMultiContactJacobianSliceGPU {
    mr_u32 articulationIndex;
    mr_u32 queryOffset;
    mr_u32 queryCount;
    mr_u32 jacobianOffset;

    mr_u32 jacobianEnvironmentStride;
    mr_u32 vOffset;
    mr_u32 nv;
    mr_u32 reserved0;
} MRMultiContactJacobianSliceGPU;

typedef struct MR_ALIGN16 MRMultiContactStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 failingContact;
    mr_u32 failingWork;

    mr_u32 pointStatusCode;
    mr_u32 inverseMassCode;
    mr_u32 qualityCode;
    mr_u32 activeContacts;

    // maximum impulse, maximum velocity change, minimum Delassus diagonal,
    // maximum Delassus asymmetry.
    mr_float4 diagnostics;
} MRMultiContactStatusGPU;

typedef struct MR_ALIGN16 MRMultiContactEqualityStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 failingRow;
    mr_u32 rowCount;

    // maximum equality residual, minimum Cholesky pivot, maximum equality
    // impulse, maximum null-space leakage.
    mr_float4 diagnostics;
} MRMultiContactEqualityStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMultiContactDispatchGPU) == 96u);
static_assert(sizeof(MRMultiContactGPU) == 144u);
static_assert(sizeof(MRMultiContactEndpointsGPU) == 32u);
static_assert(sizeof(MRMultiContactJacobianSliceGPU) == 32u);
static_assert(sizeof(MRMultiContactStatusGPU) == 48u);
static_assert(sizeof(MRMultiContactEqualityStatusGPU) == 32u);
#endif
