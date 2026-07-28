#pragma once

#include "metalrobo/engine_types.h"

#define MR_PARALLEL_ABA_SCHEDULE_ABI_VERSION 1u

enum {
    MR_PARALLEL_ABA_FIXED_ROOT = 1u << 0u,
    MR_PARALLEL_ABA_FLOATING_ROOT = 1u << 1u,
    MR_PARALLEL_ABA_SERIAL_CHAIN = 1u << 2u,
    MR_PARALLEL_ABA_BRANCHING = 1u << 3u,
};

// Immutable offsets into the flattened topology streams consumed by the
// level-scheduled Metal ABA graph. Every index is global to its typed stream.
typedef struct MR_ALIGN16 MRParallelABAArticulationGPU {
    mr_u32 abiVersion;
    mr_u32 articulationIndex;
    mr_u32 rootLocalBody;
    mr_u32 bodyCount;

    mr_u32 jointCount;
    mr_u32 flags;
    mr_u32 maximumDepth;
    mr_u32 maximumLevelWidth;

    mr_u32 forwardLevelOffset;
    mr_u32 forwardLevelCount;
    mr_u32 reverseLevelOffset;
    mr_u32 reverseLevelCount;

    mr_u32 bodyOrderOffset;
    mr_u32 parentLocalOffset;
    mr_u32 inboundJointOffset;
    mr_u32 childOffsetOffset;

    mr_u32 childIndexOffset;
    mr_u32 childIndexCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRParallelABAArticulationGPU;

// A frontier has disjoint body outputs. Reverse frontiers additionally point
// at deterministic parent-owned child reductions, avoiding floating atomics.
typedef struct MR_ALIGN16 MRParallelABALevelGPU {
    mr_u32 bodyOffset;
    mr_u32 bodyCount;
    mr_u32 parentReductionOffset;
    mr_u32 parentReductionCount;
} MRParallelABALevelGPU;

typedef struct MR_ALIGN16 MRParallelABAParentReductionGPU {
    mr_u32 parentLocalBody;
    mr_u32 firstChildIndex;
    mr_u32 childCount;
    mr_u32 stableOrdinal;
} MRParallelABAParentReductionGPU;

// One grid-Y work item. Base offsets make a single environment-major global
// q/v/state allocation addressable without rebasing buffers between
// articulations or exposing an intermediate count to the host.
typedef struct MR_ALIGN16 MRMultiABADispatchGPU {
    MRABADispatchGPU dispatch;

    mr_u32 qBase;
    mr_u32 vBase;
    mr_u32 effortBase;
    mr_u32 wrenchBase;

    mr_u32 accelerationBase;
    mr_u32 nextVBase;
    mr_u32 nextQBase;
    mr_u32 statusBase;
} MRMultiABADispatchGPU;

#ifdef __cplusplus
static_assert(sizeof(MRParallelABAArticulationGPU) == 80);
static_assert(alignof(MRParallelABAArticulationGPU) == 16);
static_assert(sizeof(MRParallelABALevelGPU) == 16);
static_assert(alignof(MRParallelABALevelGPU) == 16);
static_assert(sizeof(MRParallelABAParentReductionGPU) == 16);
static_assert(alignof(MRParallelABAParentReductionGPU) == 16);
static_assert(sizeof(MRMultiABADispatchGPU) == 80);
static_assert(alignof(MRMultiABADispatchGPU) == 16);
#endif
