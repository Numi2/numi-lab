#pragma once

#include "metalrobo/gpu_types.h"

#define MR_GENERALIZED_CONSTRAINT_ABI_VERSION 1u
#define MR_GENERALIZED_CONSTRAINT_MAX_ROWS 64u

enum MRGeneralizedConstraintStatusCode : mr_u32 {
    MR_GENERALIZED_CONSTRAINT_SUCCESS = 0u,
    MR_GENERALIZED_CONSTRAINT_INVALID_DISPATCH = 1u,
    MR_GENERALIZED_CONSTRAINT_INVERSE_MASS_FAILED = 2u,
    MR_GENERALIZED_CONSTRAINT_NONFINITE_INPUT = 3u,
    MR_GENERALIZED_CONSTRAINT_INVALID_ROW = 4u,
    MR_GENERALIZED_CONSTRAINT_SINGULAR_ROW = 5u,
    MR_GENERALIZED_CONSTRAINT_DID_NOT_CONVERGE = 6u,
    MR_GENERALIZED_CONSTRAINT_NONFINITE_RESULT = 7u,
};

typedef struct MR_ALIGN16 MRGeneralizedConstraintDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 nv;
    mr_u32 rowCount;

    mr_u32 inverseWorkCount;
    mr_u32 solverIterations;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // timestep, penetration slop, maximum stabilization velocity, minimum
    // time-constant ratio.
    mr_float4 evaluation0;
    // minimum regularization, convergence tolerance, diagonal floor, unused.
    mr_float4 evaluation1;
} MRGeneralizedConstraintDispatchGPU;

typedef struct MR_ALIGN16 MRGeneralizedConstraintStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 iterations;
    mr_u32 failingRow;

    mr_u32 failingInverseWork;
    mr_u32 inverseMassCode;
    mr_u32 activeRows;
    mr_u32 reserved0;

    // maximum impulse delta, natural residual, diagonal minimum, diagonal
    // maximum.
    mr_float4 diagnostics;
} MRGeneralizedConstraintStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRGeneralizedConstraintDispatchGPU) == 64u);
static_assert(sizeof(MRGeneralizedConstraintStatusGPU) == 48u);
#endif
