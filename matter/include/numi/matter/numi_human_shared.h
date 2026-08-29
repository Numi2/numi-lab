#pragma once

#include "numi/matter/shared.h"

#define NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION 1u

enum NMNumiHumanTendonFEMNodeLoadFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE = 1u << 0u,
};

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMLoadDispatchGPU {
    nm_u32 abiVersion;
    nm_u32 environmentCount;
    nm_u32 femNodeCount;
    nm_u32 endpointCount;

    nm_u32 transferStride;
    nm_u32 stepIndex;
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

#ifndef __METAL_VERSION__
static_assert(sizeof(NMNumiHumanTendonFEMLoadDispatchGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeLoadGPU) == 32u);
#endif
