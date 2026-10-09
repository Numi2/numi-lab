#pragma once
#include "metalrobo/engine_types.h"

// Fixed-world, finite, row-major height grid. Both triangles of every cell
// use the 10-to-01 diagonal: (00,10,01), (10,11,01). Heights never follow q.
typedef struct MR_ALIGN16 MRHumanRestingBedGPU {
    mr_uint4 counts; // nx, ny, enabled (0/1), ABI=1 when enabled
    mr_float4 originSpacing; // origin x/y, spacing x/y, metres
} MRHumanRestingBedGPU;
#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRHumanRestingBedGPU) == 32);
#endif
