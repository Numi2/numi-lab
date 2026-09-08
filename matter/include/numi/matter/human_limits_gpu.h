#pragma once
#include "numi/matter/human_equality_gpu.h"

// NHLIM1 retains one source scalar joint. Both lower and upper inequalities
// are instantiated independently by the owning Metal step at q0/v0.
#define NM_HUMAN_LIMIT_ABI_VERSION 1u
typedef struct NM_ALIGN16 NMHumanJointLimitGPU {
    nm_uint4 indices; // local q, local v, source joint identity, reserved zero
    nm_float4 rangeMarginInverseWeight; // lower, upper, margin, dof_invweight0
    nm_float4 solref;
    nm_float4 solimp0;
    nm_float4 solimp1;
} NMHumanJointLimitGPU;

// count is source joint count; linearization has 2*count float4 rows/env.
// Policy and REFSAFE flags are shared with the pinned NHEQ2 scalar law.
typedef NMHumanEqualityDispatchGPU NMHumanLimitDispatchGPU;

#if !defined(__METAL_VERSION__)
static_assert(sizeof(NMHumanJointLimitGPU) == 80u);
static_assert(sizeof(NMHumanLimitDispatchGPU) == 48u);
#endif
