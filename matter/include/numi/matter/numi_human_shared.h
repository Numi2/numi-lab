#pragma once

#include "numi/matter/shared.h"

#define NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION 2u

enum NMNumiHumanTendonFEMNodeLoadFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE = 1u << 0u,
};

enum NMNumiHumanTendonFEMNodeAnchorFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE = 1u << 0u,
};

enum NMNumiHumanTendonFEMEndpointReplacementFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE = 1u << 0u,
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
    nm_u32 reserved0;
    nm_u32 reserved1;
} NMNumiHumanTendonFEMLoadDispatchGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMNodeLoadGPU {
    nm_u32 endpointIndex;
    nm_u32 flags;
    nm_u32 reserved0;
    nm_u32 reserved1;

    // x is the fraction of the terminal world force assigned to this node.
    // yzw are reserved and must be zero.
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

    // x is the source endpoint J^T fraction replaced by the continuum anchor
    // reaction. yzw are reserved and must be zero.
    nm_float4 forceOwnerFraction;
} NMNumiHumanTendonFEMEndpointReplacementGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(NMNumiHumanTendonFEMLoadDispatchGPU) == 64u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeLoadGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeAnchorGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMEndpointReplacementGPU) == 32u);
#endif
