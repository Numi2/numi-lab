#pragma once

#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"

#define MR_NUMI_HUMAN_SUPPORT_GEOMETRY_ABI_VERSION 3u

// A source NHCNT row owns a role-0 floor region or role-1 moving boundary
// region. Role-2 derived floor rows preserve foot source-row lineage in
// reserved1. Roles 3/4 identify the registered organ floor parent and its
// derived spatial bins; sourceStableId/binIndex are carried separately below.
// pointQueryIndex selects the unique contact query refreshed from its source.
enum MRHumanRestingSupportRegionRole {
    MR_HUMAN_RESTING_SUPPORT_REGION_FLOOR = 0u,
    MR_HUMAN_RESTING_SUPPORT_REGION_BOUNDARY_ONLY = 1u,
    MR_HUMAN_RESTING_SUPPORT_REGION_DERIVED_FLOOR = 2u,
    MR_HUMAN_RESTING_SUPPORT_REGION_ORGAN_FLOOR = 3u,
    MR_HUMAN_RESTING_SUPPORT_REGION_DERIVED_ORGAN_FLOOR = 4u,
};

enum MRHumanRestingSupportLineageKind {
    MR_HUMAN_RESTING_SUPPORT_LINEAGE_NONE = 0u,
    MR_HUMAN_RESTING_SUPPORT_LINEAGE_ORGAN_SURFACE = 1u,
};

typedef struct MR_ALIGN16 MRHumanRestingSupportRegionGPU {
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRHumanRestingSupportRegionGPU;

// Per-region source lineage. Organ rows name the immutable registered
// anatomy ID and an explicit spatial bin; parentRegion ties all four bins to
// the one reserved NHCNT contact row. Other rows carry kind NONE.
typedef struct MR_ALIGN16 MRHumanRestingSupportLineageGPU {
    mr_u32 kind;
    mr_u32 sourceStableId;
    mr_u32 parentRegion;
    mr_u32 binIndex;
} MRHumanRestingSupportLineageGPU;

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

typedef struct MR_ALIGN16 MRHumanRestingSupportPredictionGPU {
    // xyz = global support-plane point; w = timestep seconds.
    mr_float4 groundPointAndTimestep;
    // x = source-free velocity element offset; y = buffer element count;
    // z = per-environment element stride; w = body count.
    mr_uint4 sourceVelocity;
} MRHumanRestingSupportPredictionGPU;

typedef struct MR_ALIGN16 MRHumanRestingSupportDispatchGPU {
    // vertex count, region count, dof count, environment count
    mr_uint4 counts;
    // body-pose stride, body-probe offset, point-world stride, point-J stride
    mr_uint4 strides;
    // articulation first body, influence-buffer count, support row count, ABI
    mr_uint4 identity;
    MRHumanRestingSupportPredictionGPU prediction;
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
static_assert(sizeof(MRHumanRestingSupportLineageGPU) == 16);
static_assert(sizeof(MRHumanRestingSupportVertexRegionGPU) == 16);
static_assert(sizeof(MRHumanRestingSupportPositionGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportRotationZBasisGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportPredictionGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportDispatchGPU) == 80);
static_assert(sizeof(MRHumanRestingSupportDebugGPU) == 48);
#endif
