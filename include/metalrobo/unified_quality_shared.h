#pragma once

#include "metalrobo/gpu_types.h"

// Product-cone, generalized-velocity quality solve. This ABI is independent
// from ConstraintIR: one adapter binds canonical IR blocks to these compact
// runtime records without changing ConstraintIR ABI v2.
#define MR_UNIFIED_QUALITY_ABI_VERSION 1u
#define MR_UNIFIED_QUALITY_MAX_BLOCK_DIMENSION 6u
#define MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES 128u
#define MR_UNIFIED_QUALITY_MAX_ROWS 768u
#define MR_UNIFIED_QUALITY_MAX_BLOCKS 128u
#define MR_UNIFIED_QUALITY_INVALID_INDEX 0xffffffffu

enum MRUnifiedQualityBlockKind : mr_u32 {
    MR_UNIFIED_QUALITY_SCALAR_INTERVAL = 0u,
    MR_UNIFIED_QUALITY_ELLIPTIC_CONE = 1u,
};

enum MRUnifiedQualityBlockFlags : mr_u32 {
    MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY = 1u << 0u,
    MR_UNIFIED_QUALITY_BLOCK_REPORT_FLOOR = 1u << 1u,
};

enum MRUnifiedQualitySolvePath : mr_u32 {
    MR_UNIFIED_QUALITY_PATH_DIRECT = 0u,
    MR_UNIFIED_QUALITY_PATH_PCG = 1u,
};

enum MRUnifiedQualityStatusCode : mr_u32 {
    MR_UNIFIED_QUALITY_SUCCESS = 0u,
    MR_UNIFIED_QUALITY_INVALID_DISPATCH = 1u,
    MR_UNIFIED_QUALITY_INVALID_BLOCK = 2u,
    MR_UNIFIED_QUALITY_NONFINITE_INPUT = 3u,
    MR_UNIFIED_QUALITY_FACTORIZATION_FAILED = 4u,
    MR_UNIFIED_QUALITY_PCG_FAILED = 5u,
    MR_UNIFIED_QUALITY_LINE_SEARCH_FAILED = 6u,
    MR_UNIFIED_QUALITY_DID_NOT_CONVERGE = 7u,
    MR_UNIFIED_QUALITY_NONFINITE_RESULT = 8u,
};

enum MRUnifiedQualityWorkQueueFlags : mr_u32 {
    MR_UNIFIED_QUALITY_QUEUE_PERSISTENT_WORKER = 1u << 0u,
};

// Invocation-local queue metadata. Packet order is produced by a stable
// environment scan; cursor claim order is deliberately excluded from all
// physical reductions.
typedef struct MR_ALIGN16 MRUnifiedQualityWorkQueueGPU {
    mr_u32 count;
    mr_u32 capacity;
    mr_u32 required;
    mr_u32 cursor;

    mr_u32 workerGroups;
    mr_u32 packetsProcessed;
    mr_u32 emptyPulls;
    mr_u32 flags;

    mr_uint4 firstFailingStableKey;
    // Layout-compatible x/y/z dispatch arguments plus active-count evidence.
    // Standalone Metal dispatches from this field without exposing count.
    mr_uint4 indirect;
} MRUnifiedQualityWorkQueueGPU;

// One immutable active quality problem. The stable key remains available for
// diagnostics even though every output is written to the disjoint problem
// slot selected by problemIndex.
typedef struct MR_ALIGN16 MRUnifiedQualityWorkPacketGPU {
    // problem index, island index, stable-key low, stable-key high.
    mr_uint4 identity;
    // event generation, phase, flags, reserved.
    mr_uint4 scheduling;
} MRUnifiedQualityWorkPacketGPU;

typedef struct MR_ALIGN16 MRUnifiedQualityBlockGPU {
    // Row offset, dimension, MRUnifiedQualityBlockKind, flags.
    mr_uint4 layout;
    // Canonical stable block key.
    mr_uint4 stableKey;

    // Cone coordinates are lambda_i / scale_i. scale_0 must be one.
    // Scalar blocks use scale0.x = 1 and ignore other entries.
    mr_float4 scale0;
    mr_float4 scale1;

    // Positive diagonal R in
    // 0.5 lambda' R lambda + (Jv+b)' lambda.
    mr_float4 regularization0;
    mr_float4 regularization1;

    // Scalar: x=lower, y=upper.
    // Cone: x=adhesion shift, y=maximum physical normal impulse
    // (zero means unbounded).
    mr_float4 boundsAndShift;
    mr_uint4 reserved;
} MRUnifiedQualityBlockGPU;

typedef struct MR_ALIGN16 MRUnifiedQualityDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 problemCount;
    mr_u32 generalizedVelocityCount;
    mr_u32 rowCount;

    mr_u32 blockCount;
    mr_u32 dynamicsStride;
    mr_u32 jacobianStride;
    mr_u32 vectorStride;

    mr_u32 maximumNewtonIterations;
    mr_u32 maximumPCGIterations;
    mr_u32 maximumLineSearchIterations;
    mr_u32 directMaximumGeneralizedVelocities;

    mr_u32 directMaximumRows;
    mr_u32 derivativeStride;
    mr_u32 hessianStride;
    // Zero means one topology shared by every problem. Nonzero addresses an
    // environment-major runtime block stream, as used by MetalWorld.
    mr_u32 blockStride;

    // optimality, feasibility, Armijo, contraction.
    mr_float4 tolerances;
    // compliance floor multiplier, minimum pivot, minimum PCG denominator,
    // regularization retry scale.
    mr_float4 numerics;
} MRUnifiedQualityDispatchGPU;

typedef struct MR_ALIGN16 MRUnifiedQualityStatusGPU {
    mr_u32 code;
    mr_u32 problemIndex;
    mr_u32 solvePath;
    mr_u32 failingBlock;

    mr_u32 newtonIterations;
    mr_u32 pcgIterations;
    mr_u32 lineSearchBacktracks;
    mr_u32 regularizationRetries;

    mr_uint4 firstFailingStableKey;

    // scaled optimality, cone/scalar feasibility, equality feasibility,
    // complementarity.
    mr_float4 certificates0;
    // dynamics backward error, Newton decrement, objective change,
    // final objective.
    mr_float4 certificates1;
    // effective compliance floor, regularization retry, minimum pivot,
    // maximum pivot.
    mr_float4 numerics;
} MRUnifiedQualityStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRUnifiedQualityWorkQueueGPU) == 64u);
static_assert(sizeof(MRUnifiedQualityWorkPacketGPU) == 32u);
static_assert(sizeof(MRUnifiedQualityBlockGPU) == 128u);
static_assert(sizeof(MRUnifiedQualityDispatchGPU) == 96u);
static_assert(sizeof(MRUnifiedQualityStatusGPU) == 96u);
#endif
