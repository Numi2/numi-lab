#pragma once

#include "metalrobo/unified_quality_shared.h"

// Inspection only. No captured value is consumed by a physical solve.
#define MR_UNIFIED_QUALITY_FAILURE_MAGIC 0x51464c31u
#define MR_UNIFIED_QUALITY_FAILURE_VERSION 1u
#define MR_UNIFIED_QUALITY_FAILURE_MAX_BYTES (16u * 1024u * 1024u)

typedef struct MR_ALIGN16 MRUnifiedQualityFailurePassGPU {
    mr_u32 controlStep;
    mr_u32 physicsSubstep;
    mr_u32 recordBytes;
    mr_u32 reserved;
} MRUnifiedQualityFailurePassGPU;

typedef struct MR_ALIGN16 MRUnifiedQualityFailureHeaderGPU {
    mr_u32 valid;
    mr_u32 version;
    mr_u32 environment;
    mr_u32 controlStep;
    mr_u32 physicsSubstep;
    mr_u32 recordBytes;
    mr_u32 reserved0;
    mr_u32 reserved1;
    MRUnifiedQualityDispatchGPU dispatch;
    MRUnifiedQualityStatusGPU status;
    MRUnifiedQualityWorkPacketGPU packet;
} MRUnifiedQualityFailureHeaderGPU;

// Header, blocks, then packed FP32 arrays: A, J, bias, free velocity,
// warm velocity, warm impulse, failed velocity, impulse, row velocity,
// gradient. Stride is padded to 16 bytes. Only declared dimensions are copied.
static inline mr_u32 mrUnifiedQualityFailureRecordBytes(
    const mr_u32 nv, const mr_u32 rows, const mr_u32 blocks
) {
    if (nv == 0u || nv > MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES ||
        rows == 0u || rows > MR_UNIFIED_QUALITY_MAX_ROWS ||
        blocks == 0u || blocks > MR_UNIFIED_QUALITY_MAX_BLOCKS) return 0u;
    const mr_u32 bytes = 256u + 128u * blocks +
        4u * (nv * nv + rows * nv + 4u * nv + 4u * rows);
    return (bytes + 15u) & ~15u;
}

#ifndef __METAL_VERSION__
static_assert(sizeof(MRUnifiedQualityFailureHeaderGPU) == 256u);
static_assert(sizeof(MRUnifiedQualityFailurePassGPU) == 16u);
#endif
