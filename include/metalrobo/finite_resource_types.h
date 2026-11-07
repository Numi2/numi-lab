#pragma once
#include "metalrobo/gpu_types.h"

#define MR_FINITE_RESOURCE_ABI 3u
#define MR_FINITE_RESOURCE_MAX_PATCHES 8u

// Integer micrograms of nutritional payload, not a bulk-food mechanics model.
typedef struct MR_ALIGN16 MRFiniteResourcePatchGPU {
    // Accepted world body index, initial micrograms, bite micrograms, reserved.
    mr_uint4 binding;
    // Body-local intake-volume center xyz and radius in metres.
    mr_float4 centerRadius;
} MRFiniteResourcePatchGPU;

typedef struct MR_ALIGN16 MRFiniteResourceProgramGPU {
    // Mouth body, scalar jaw q index, patch count, crop capacity micrograms.
    mr_uint4 binding;
    // Mouth-body-local intake point xyz and closed-jaw angle in radians.
    mr_float4 mouthClosed;
    // Open-jaw threshold radians, reserved lanes.
    mr_float4 opening;
} MRFiniteResourceProgramGPU;

typedef struct MR_ALIGN16 MRFiniteResourceStateGPU {
    mr_uint4 remaining[2];
    // Crop micrograms, total consumed micrograms, bite count, initialized.
    mr_uint4 accounting;
    // Accepted control ticks low/high, open-jaw latch, reserved.
    mr_uint4 clock;
    // Previous accepted jaw angle; other lanes reserved.
    mr_float4 jaw;
} MRFiniteResourceStateGPU;

typedef struct MR_ALIGN16 MRFiniteResourceDispatchGPU {
    // Environments, body stride, q stride, body count.
    mr_uint4 counts;
    // q count, ABI, publish native task nutrition, reserved.
    mr_uint4 limits;
    // Reset-mask control-step stride, control-step index, reserved, reserved.
    mr_uint4 reset;
} MRFiniteResourceDispatchGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRFiniteResourcePatchGPU)==32u);
static_assert(sizeof(MRFiniteResourceProgramGPU)==48u);
static_assert(sizeof(MRFiniteResourceStateGPU)==80u);
static_assert(sizeof(MRFiniteResourceDispatchGPU)==48u);
#endif
