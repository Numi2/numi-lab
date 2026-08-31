#pragma once

#include "numi/matter/shared.h"

#define NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION 4u

enum NMNumiHumanTendonFEMNodeLoadFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE = 1u << 0u,
};

enum NMNumiHumanTendonFEMNodeAnchorFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE = 1u << 0u,
};

enum NMNumiHumanTendonFEMEndpointReplacementFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE = 1u << 0u,
    // Replace the selected muscle's complete source generalized-force row,
    // then restore only loadEndpointIndex's body reaction. The continuum and
    // its solved anchors own every remaining route force, including forces
    // that the source model previously applied through internal wrap bodies.
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW = 1u << 1u,
    // In full-muscle-row mode, also consume anchorEndpointIndex as a signed
    // zero-resultant force couple: positive node weights load one attachment
    // and negative weights load the opposing attachment. This reconstructs a
    // massless two-segment tendon such as QAT -> patella -> PTL -> tibia while
    // the load endpoint remains the explicitly restored proximal reaction.
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE = 1u << 2u,
};

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMLoadDispatchGPU {
    nm_u32 abiVersion;
    nm_u32 environmentCount;
    nm_u32 femNodeCount;
    nm_u32 endpointCount;

    nm_u32 transferStride;
    nm_u32 stepIndex;
    nm_u32 replacementCount;
    nm_u32 dofCount;

    nm_u32 bodyPoseStride;
    nm_u32 articulationFirstBody;
    nm_u32 pointJacobianStride;
    nm_u32 bodyJacobianPointOffset;

    nm_u32 generalizedForceStride;
    nm_u32 generalizedForceOffset;
    nm_u32 muscleCount;
    nm_u32 reserved1;
} NMNumiHumanTendonFEMLoadDispatchGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMNodeLoadGPU {
    // Up to four terminal forces may contribute to one continuum node. An
    // unused slot has NM_INVALID_INDEX and a zero scale. This is required for
    // anatomical junctions such as the quadriceps tendon, where four source
    // muscles share one proximal traction surface without arbitrary spatial
    // partitioning.
    nm_u32 endpointIndex[4];

    // Component i is the signed fraction of endpointIndex[i]'s terminal world
    // force assigned to this node. Ordinary traction components are positive.
    // Negative components are admitted only for an explicitly declared distal
    // force couple and apply the equal-and-opposite attachment load.
    nm_float4 scale;
} NMNumiHumanTendonFEMNodeLoadGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMNodeAnchorGPU {
    nm_u32 bodyIndex;
    nm_u32 flags;
    nm_u32 reserved0;
    nm_u32 reserved1;

    // Prescribed attachment in COM-relative Human body coordinates.
    nm_float4 localPoint;
} NMNumiHumanTendonFEMNodeAnchorGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMEndpointReplacementGPU {
    nm_u32 loadEndpointIndex;
    nm_u32 anchorEndpointIndex;
    nm_u32 flags;
    nm_u32 reserved0;

    // x is the source-force fraction replaced by the continuum. In endpoint
    // mode this is the anchor endpoint J^T share. In full-muscle-row mode it
    // is the complete selected muscle row, with the load endpoint reaction
    // restored explicitly. yzw are reserved and must be zero.
    nm_float4 forceOwnerFraction;
} NMNumiHumanTendonFEMEndpointReplacementGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(NMNumiHumanTendonFEMLoadDispatchGPU) == 64u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeLoadGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeAnchorGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMEndpointReplacementGPU) == 32u);
#endif
