#pragma once

#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"

#define MR_NUMI_HUMAN_SUPPORT_GEOMETRY_ABI_VERSION 1u

// A source NHCNT row owns this support region. sourceGeometryIndex preserves
// the seed identity; pointQueryIndex selects the existing contact query slot
// whose position and weighted Jacobian are refreshed from the current skin.
typedef struct MR_ALIGN16 MRHumanRestingSupportRegionGPU {
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRHumanRestingSupportRegionGPU;

// One derived rest-space Voronoi label per registered skin vertex. Invalid is
// reserved for non-skin render vertices; every actual skin vertex must map to
// exactly one contact region before upload.
typedef struct MR_ALIGN16 MRHumanRestingSupportVertexRegionGPU {
    mr_u32 regionIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
} MRHumanRestingSupportVertexRegionGPU;

typedef struct MR_ALIGN16 MRHumanRestingSupportPositionGPU {
    mr_float4 high;
    mr_float4 low;
} MRHumanRestingSupportPositionGPU;

// Rotation's world-Z basis is cached once per body pose. Published contact
// positions and Jacobians continue through the canonical compensated LBS path.
typedef struct MR_ALIGN16 MRHumanRestingSupportRotationZBasisGPU {
    mr_float4 high;
    mr_float4 low;
} MRHumanRestingSupportRotationZBasisGPU;

typedef struct MR_ALIGN16 MRHumanRestingSupportDispatchGPU {
    // vertex count, region count, dof count, environment count
    mr_uint4 counts;
    // body-pose stride, body-probe offset, point-world stride, point-J stride
    mr_uint4 strides;
    // articulation first body, influence-buffer count, support row count, ABI
    mr_uint4 identity;
} MRHumanRestingSupportDispatchGPU;

// One-shot first-step audit output. It is written only when the debug path is
// compiled/explicitly enabled by the owner; it is not part of accepted state.
typedef struct MR_ALIGN16 MRHumanRestingSupportDebugGPU {
    mr_uint4 status; // selected vertex, region error, global error, nonfinite J count
    mr_float4 position; // current full-LBS high+low world point
    mr_float4 jacobian; // max absolute weighted-J entry, remaining lanes reserved
} MRHumanRestingSupportDebugGPU;

#if defined(__cplusplus)
static_assert(sizeof(MRHumanRestingSupportRegionGPU) == 16);
static_assert(sizeof(MRHumanRestingSupportVertexRegionGPU) == 16);
static_assert(sizeof(MRHumanRestingSupportPositionGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportRotationZBasisGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportDispatchGPU) == 48);
static_assert(sizeof(MRHumanRestingSupportDebugGPU) == 48);
#endif
