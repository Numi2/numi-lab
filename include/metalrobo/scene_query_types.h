#pragma once

#include "metalrobo/gpu_types.h"

#define MR_SCENE_QUERY_ABI_VERSION 2u

enum MRSceneQueryFlags : mr_u32 {
    // Collision-disabled authored geometry is factual but excluded from
    // normal sensor queries unless explicitly requested.
    MR_SCENE_QUERY_INCLUDE_DISABLED = 1u << 0u,
    // Treat triangle meshes as two-sided even when their collision record is
    // authored one-sided. This is useful for range sensors mounted inside
    // thin shells.
    MR_SCENE_QUERY_FORCE_TWO_SIDED = 1u << 1u,
    // Orient the reported normal against the incident ray.
    MR_SCENE_QUERY_FACE_FORWARD_NORMAL = 1u << 2u,
};

// Device contract for assembling the complete environment-major rigid state
// from generalized articulation state and compact scene-body arrays.
typedef struct MR_ALIGN16 MRBodyStateMaterializeDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 bodyCount;
    mr_u32 articulationCount;

    mr_u32 sceneBodyCount;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 bodyStride;
} MRBodyStateMaterializeDispatchGPU;

// Device contract for arbitrary environment-major ray batches. Every stride
// is measured in records, not bytes.
typedef struct MR_ALIGN16 MRSceneQueryDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 rayCount;
    mr_u32 bodyCount;

    mr_u32 shapeCount;
    mr_u32 geometryCount;
    mr_u32 vertexCount;
    mr_u32 meshNodeCount;

    mr_u32 meshTriangleCount;
    mr_u32 convexFaceCount;
    mr_u32 rayStride;
    mr_u32 bodyStride;
} MRSceneQueryDispatchGPU;

// Environment-local shape projection reused by every ray in one query.
// centerAndRadius.w is a verified conservative radius, -1 for a valid plane,
// and -2 for an invalid transform.
typedef struct MR_ALIGN16 MRSceneQueryShapeStateGPU {
    mr_float4 centerAndRadius;
    mr_float4 rotation;
} MRSceneQueryShapeStateGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRBodyStateMaterializeDispatchGPU) == 32);
static_assert(sizeof(MRSceneQueryDispatchGPU) == 48);
static_assert(sizeof(MRSceneQueryShapeStateGPU) == 32);
#endif
