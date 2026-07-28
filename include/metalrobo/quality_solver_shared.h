#pragma once

#include "metalrobo/gpu_types.h"

#define MR_METAL_QUALITY_SOLVER_ABI_VERSION 1u
#define MR_METAL_QUALITY_MAX_CONTACTS 32u
#define MR_METAL_QUALITY_MAX_DIMENSION \
    (3u * MR_METAL_QUALITY_MAX_CONTACTS)

enum MRMetalQualityStatusCode : mr_u32 {
    MR_METAL_QUALITY_SUCCESS = 0u,
    MR_METAL_QUALITY_INVALID_DISPATCH = 1u,
    MR_METAL_QUALITY_NONFINITE_INPUT = 2u,
    MR_METAL_QUALITY_LINEAR_SOLVE_FAILED = 3u,
    MR_METAL_QUALITY_DID_NOT_CONVERGE = 4u,
    MR_METAL_QUALITY_NONFINITE_RESULT = 5u,
};

typedef struct MR_ALIGN16 MRMetalQualityDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 problemCount;
    mr_u32 contactCount;
    mr_u32 dimension;

    mr_u32 matrixStride;
    mr_u32 vectorStride;
    mr_u32 maximumNewtonIterations;
    mr_u32 maximumCGIterations;

    mr_u32 maximumLineSearchIterations;
    mr_u32 flags;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // convergence tolerance, Armijo coefficient, normal-equation
    // regularization, minimum CG denominator.
    mr_float4 tolerances;
} MRMetalQualityDispatchGPU;

typedef struct MR_ALIGN16 MRMetalQualityStatusGPU {
    mr_u32 code;
    mr_u32 problemIndex;
    mr_u32 newtonIterations;
    mr_u32 cgIterations;

    mr_u32 lineSearchBacktracks;
    mr_u32 projectedGradientFallbacks;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // scaled natural residual, merit, cone violation, objective.
    mr_float4 diagnostics;
} MRMetalQualityStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMetalQualityDispatchGPU) == 64u);
static_assert(sizeof(MRMetalQualityStatusGPU) == 48u);
#endif
