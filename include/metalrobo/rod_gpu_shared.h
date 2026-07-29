#pragma once

#include "metalrobo/gpu_types.h"

#define MR_ROD_GPU_ABI_VERSION 1u
#define MR_ROD_GPU_MAX_NODES 128u
#define MR_ROD_GPU_MAX_ATTACHMENTS 8u

enum {
    MR_ROD_GPU_SUCCESS = 0u,
    MR_ROD_GPU_INVALID_DISPATCH = 1u,
    MR_ROD_GPU_DEGENERATE_GEOMETRY = 2u,
    MR_ROD_GPU_NONFINITE_RESULT = 3u,
    MR_ROD_GPU_DID_NOT_CONVERGE = 4u,
};

typedef struct MR_ALIGN16 MRRodGPUDispatch {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 nodeCount;
    mr_u32 edgeCount;

    mr_u32 attachmentCount;
    mr_u32 solverIterations;
    mr_u32 stateNodeStride;
    mr_u32 stateEdgeStride;

    // xyz gravity, w timestep.
    mr_float4 gravityAndTimestep;
    // linear damping, twist damping, derivative step, tolerance.
    mr_float4 dampingDerivativeTolerance;
} MRRodGPUDispatch;

typedef struct MR_ALIGN16 MRRodGPUAttachment {
    // xyz target, w compliance.
    mr_float4 targetAndCompliance;
    // xyz velocity, w must be zero.
    mr_float4 velocity;
    mr_u32 nodeIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
} MRRodGPUAttachment;

typedef struct MR_ALIGN16 MRRodGPUStatus {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 iterations;
    mr_u32 failingIndex;

    // Maximum constraint error, maximum correction, reserved, reserved.
    mr_float4 diagnostics;
} MRRodGPUStatus;

#ifdef __cplusplus
static_assert(sizeof(MRRodGPUDispatch) == 64);
static_assert(alignof(MRRodGPUDispatch) == 16);
static_assert(sizeof(MRRodGPUAttachment) == 48);
static_assert(alignof(MRRodGPUAttachment) == 16);
static_assert(sizeof(MRRodGPUStatus) == 32);
static_assert(alignof(MRRodGPUStatus) == 16);
#endif
