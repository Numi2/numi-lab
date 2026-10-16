#pragma once

#include "metalrobo/engine_types.h"

#define MR_NUMI_HUMAN_JOINT_EQUALITY_GPU_ABI_VERSION 1u
#define MR_NUMI_HUMAN_JOINT_EQUALITY_FIXED_MASTER MR_INVALID_INDEX

// Source MuJoCo scalar joint equality:
//   y - y0 = a0 + a1 (x-x0) + ... + a4 (x-x0)^4.
// Indices are articulation-local. A fixed master uses MR_INVALID_INDEX for
// both master indices. solref/solimp are retained source compliance metadata;
// the bounded Human solver may select a stricter stabilization policy but may
// not silently change the polynomial.
typedef struct MR_ALIGN16 MRNumiHumanJointEqualityGPU {
    // dependent q/v, master q/v.
    mr_uint4 indices;
    // dependent reference y0, master reference x0, a0, a1.
    mr_float4 referencesAndCoefficients0;
    // a2, a3, a4, reserved zero.
    mr_float4 coefficients1;
    // source solref[0:2], zw reserved zero.
    mr_float4 solref;
    // source solimp[0:4].
    mr_float4 solimp0;
    mr_float4 solimp1;
} MRNumiHumanJointEqualityGPU;

#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRNumiHumanJointEqualityGPU) == 96);
#endif
