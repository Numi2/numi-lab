#pragma once
#include "numi/matter/shared.h"

// NHEQ2: pinned MuJoCo 3.12 scalar-joint, classic acceleration semantics.
// Positive source compliance permits exact Schur elimination; this is never
// a tangent projection or a hard position correction.
#define NM_HUMAN_EQUALITY_ABI_VERSION 1u
#define NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC 1u
#define NM_HUMAN_EQUALITY_REFSAFE 1u

typedef struct NM_ALIGN16 NMHumanJointEqualityGPU {
    nm_uint4 indices;
    nm_float4 referencesAndCoefficients0;
    nm_float4 coefficients1;
    nm_float4 solref;
    nm_float4 solimp0;
    nm_float4 solimp1;
    nm_float4 sourceInverseWeights;
} NMHumanJointEqualityGPU;

typedef struct NM_ALIGN16 NMHumanEqualityDispatchGPU {
    nm_u32 count;
    nm_u32 qCount;
    nm_u32 dofCount;
    nm_u32 flags;
    nm_u32 policy;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u32 reserved2;
    nm_float4 time;
} NMHumanEqualityDispatchGPU;

#if !defined(__METAL_VERSION__)
static_assert(sizeof(NMHumanJointEqualityGPU) == 112u);
static_assert(sizeof(NMHumanEqualityDispatchGPU) == 48u);
#endif
