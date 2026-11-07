#pragma once
#include "metalrobo/gpu_types.h"

#define MR_FINITE_RESOURCE_ABI 3u
// Kernel argument layout is versioned independently from persistent state and
// program fingerprint bytes. Allocation adds two bindings without changing the
// legacy state layout. Its entry point is mr_finite_resource_commit_v4.
#define MR_FINITE_RESOURCE_KERNEL_ABI 4u
#define MR_FINITE_RESOURCE_MAX_PATCHES 8u

// A rendered mesh instance cues one stocked patch during the first specified
// accepted lifetime ticks. No selection/clock value is an actor feature.
typedef struct MR_ALIGN16 MRFiniteResourceCueGPU {
    // Global rendered mesh instance, resource patch, exclusive end tick, zero.
    mr_uint4 binding;
} MRFiniteResourceCueGPU;

// Optional allocation v1. Zero means the original all-sites, immediate,
// observable-proximity program. Nonzero selects exactly activePatchCount sites
// once per native lifetime; ordinary episode resets never reroll/refill it.
typedef struct MR_ALIGN16 MRFiniteResourceAllocationGPU {
    mr_u32 version;
    mr_u32 activePatchCount;
    mr_u32 availableAfterTicks;
    mr_u32 concealProximity;
} MRFiniteResourceAllocationGPU;

// SplitMix64 counter stream, with a resource-specific lane domain. Rejection
// sampling avoids modulo bias; at most eight sites are selected once per life.
inline mr_u64 mrResourceMix64(mr_u64 value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31u);
}
inline mr_u32 mrResourceAllocationMask(mr_u64 seed, mr_u32 environment,
                                     mr_u32 patches, mr_u32 active) {
    mr_u32 selected = 0u, draw = 0u;
    const mr_u64 key = seed ^ ((mr_u64(environment)+1u)*0xd2b74407b1ce6e93ULL)
        ^ 0x666f6f6473697465ULL;
    for (mr_u32 pick=0u; pick<active; ++pick) {
        const mr_u32 bound = patches-pick;
        const mr_u32 threshold = (mr_u32(0u)-bound)%bound;
        mr_u32 bits;
        do { bits = mr_u32(mrResourceMix64(key+mr_u64(draw++))); } while (bits<threshold);
        mr_u32 rank=bits%bound;
        for (mr_u32 patch=0u; patch<patches; ++patch) {
            if (selected & (1u<<patch)) continue;
            if (rank==0u) { selected |= 1u<<patch; break; }
            --rank;
        }
    }
    return selected;
}

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
    // Accepted control ticks low/high, open-jaw latch, allocation-v1 site mask.
    // Legacy programs retain zero in the mask lane.
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
static_assert(sizeof(MRFiniteResourceCueGPU)==16u);
static_assert(sizeof(MRFiniteResourceAllocationGPU)==16u);
static_assert(sizeof(MRFiniteResourcePatchGPU)==32u);
static_assert(sizeof(MRFiniteResourceProgramGPU)==48u);
static_assert(sizeof(MRFiniteResourceStateGPU)==80u);
static_assert(sizeof(MRFiniteResourceDispatchGPU)==48u);
#endif
