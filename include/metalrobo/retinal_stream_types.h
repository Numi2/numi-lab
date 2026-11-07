#pragma once
#include "metalrobo/gpu_types.h"
#define MR_RETINAL_STREAM_ABI 1u
// A purely sensory stream. No identities, depth, transforms, rewards or
// object coordinates are inputs to these kernels.
typedef struct MR_ALIGN16 MRRetinalDispatchGPU {
    // Source width/height and output width/height; exact area reduction.
    mr_uint4 dimensions;
    // Environments, eyes, actor feature offset, actor stride.
    mr_uint4 layout;
    // Sample-rate numerator/denominator per accepted control tick, delay, ABI.
    mr_uint4 clock;
    // Output base element offset low/high, selected eye, reserved.
    mr_uint4 output;
    // Control tick duration seconds; other lanes reserved.
    mr_float4 timing;
} MRRetinalDispatchGPU;
typedef struct MR_ALIGN16 MRRetinalStateGPU {
    // Last observed accepted episode step/index, initialized, reserved.
    mr_uint4 timeline;
    // Visible frame valid, capture step/index, cumulative captures in epoch.
    mr_uint4 delivery;
    // Capture now, deliver now, clear visible, delivery ring slot.
    mr_uint4 work;
} MRRetinalStateGPU;
#ifndef __METAL_VERSION__
static_assert(sizeof(MRRetinalDispatchGPU)==80u);
static_assert(sizeof(MRRetinalStateGPU)==48u);
#endif
