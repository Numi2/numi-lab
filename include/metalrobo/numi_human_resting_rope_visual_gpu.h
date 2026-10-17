#pragma once
#include "metalrobo/numi_human_resting_rope_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"

// Presentation-only geometry for the accepted-state resting tether.
// The rope force remains owned exclusively by MRHumanRestingRopeGPU.
typedef struct MR_ALIGN16 MRHumanRestingRopeVisualGPU {
    // First world-bound mesh vertex, rope rings, sides per ring, harness ring start.
    mr_uint4 ranges;
    // Rope vertex count, harness major segments, minor segments, harness vertex count.
    mr_uint4 counts;
    // Rope radius, harness major radius, harness tube radius, visual-only slack sag cap.
    mr_float4 radiiAndSag;
} MRHumanRestingRopeVisualGPU;
#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRHumanRestingRopeVisualGPU) == 48u);
#endif
