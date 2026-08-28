#pragma once

#include "metalrobo/engine_types.h"

#define MR_NUMI_HUMAN_TENDON_TRANSFER_GPU_ABI_VERSION 1u

enum MRNumiHumanTendonTransferGPUStatus : mr_u32 {
    MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS = 0u,
    MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_DISPATCH = 1u,
    MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_BINDING = 2u,
    MR_NUMI_HUMAN_TENDON_TRANSFER_INVALID_MUSCLE_RESULT = 3u,
    MR_NUMI_HUMAN_TENDON_TRANSFER_NONFINITE_RESULT = 4u,
};

enum MRNumiHumanTendonTransferMode : mr_u32 {
    MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT = 0u,
    MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE = 2u,
};

typedef struct MR_ALIGN16 MRNumiHumanTendonTransferDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 endpointCount;
    mr_u32 envelopeCount;
    mr_u32 muscleCount;

    mr_u32 environmentCount;
    mr_u32 dofCount;
    mr_u32 bodyPoseStride;
    mr_u32 articulationFirstBody;

    mr_u32 pointJacobianStride;
    mr_u32 bodyJacobianPointOffset;
    mr_u32 bodyJacobianPointStride;
    mr_u32 reserved0;
} MRNumiHumanTendonTransferDispatchGPU;

typedef struct MR_ALIGN16 MRNumiHumanTendonBindingGPU {
    mr_u32 muscleIndex;
    mr_u32 endpointOrdinal;
    mr_u32 bodyIndex;
    mr_u32 mode;
    mr_u32 envelopeIndex;
    mr_u32 boneStableId;
    mr_u32 reserved0;
    mr_u32 reserved1;
    // Authored MyoSim endpoint in COM-relative Core body coordinates.
    mr_float4 sourceLocalPoint;
} MRNumiHumanTendonBindingGPU;

typedef struct MR_ALIGN16 MRNumiHumanTendonEnvelopeGPU {
    mr_u32 bodyIndex;
    mr_u32 boneStableId;
    mr_u32 sourceTriangleIndex;
    mr_u32 nodeCount;
    mr_float4 localNodes[4];
    // Four row-major 3x3 maps, padded to float4 rows.
    mr_float4 forceMapRows[12];
    // x surface distance; y patch radius; z sampled force amplification;
    // w L2 force amplification.
    mr_float4 metrics;
} MRNumiHumanTendonEnvelopeGPU;

typedef struct MR_ALIGN16 MRNumiHumanTendonTransferResultGPU {
    mr_u32 status;
    mr_u32 environment;
    mr_u32 bindingIndex;
    mr_u32 envelopeIndex;
    mr_float4 terminalWorldForce;
    mr_float4 nodalWorldForces[4];
    // x force residual N; y source-point moment residual Nm;
    // z maximum absolute generalized correction; w represented actuator force.
    mr_float4 residualsAndForce;
} MRNumiHumanTendonTransferResultGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRNumiHumanTendonTransferDispatchGPU) == 48u);
static_assert(sizeof(MRNumiHumanTendonBindingGPU) == 48u);
static_assert(sizeof(MRNumiHumanTendonEnvelopeGPU) == 288u);
static_assert(sizeof(MRNumiHumanTendonTransferResultGPU) == 112u);
#endif
