#pragma once

#include "metalrobo/gpu_types.h"

#define MR_HYPER_POLICY_PROGRAM_ABI_VERSION 2u
#define MR_HYPER_POLICY_INVALID_INDEX 0xffffffffu

enum MRHyperPolicyProgramFlags : mr_u32 {
    MR_HYPER_POLICY_STOCHASTIC = 1u << 0u,
    MR_HYPER_POLICY_REFERENCE_RESIDUAL = 1u << 1u,
};

enum MRHyperPolicyPhaseFlags : mr_u32 {
    MR_HYPER_POLICY_PHASE_COMPLETE = 1u << 0u,
    MR_HYPER_POLICY_PHASE_GUARD_BLOCKED = 1u << 1u,
    MR_HYPER_POLICY_PHASE_INITIALIZED = 1u << 2u,
};

typedef struct MR_ALIGN16 MRHyperPolicyProgramHeaderGPU {
    // layers, actor observations, actions, generated coefficients.
    mr_uint4 counts0;
    // knots, reference frames, phase-signature width, maximum adapter rank.
    mr_uint4 counts1;
    // event guards, contact tracks, flags, reserved.
    mr_uint4 counts2;
    // layer table, actor mean, actor inverse stddev, coefficient limits.
    mr_uint4 offsets0;
    // action bias, action scale, knot phases, coefficient knots.
    mr_uint4 offsets1;
    // coefficient tangents, authority knots/tangents, phase-rate knots.
    mr_uint4 offsets2;
    // phase-rate tangents, reference phases/actions/signatures.
    mr_uint4 offsets3;
    // signature weights, reference contact masks, event table, log stddev.
    mr_uint4 offsets4;
    // task contact-group indices for each generated contact track; reserved.
    mr_uint4 offsets5;
    // observation clip, action clip, maximum phase advance, alignment blend.
    mr_float4 limits0;
    // Huber delta, control dt, interpolation epsilon, reserved.
    mr_float4 limits1;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
    mr_u64 sourceMotionFingerprint;
    mr_u64 revision;
    // ABI version and reserved values.
    mr_uint4 abi;
} MRHyperPolicyProgramHeaderGPU;

typedef struct MR_ALIGN16 MRHyperPolicyLayerGPU {
    // input width, output width, adapter rank, activation.
    mr_uint4 counts;
    // base weight, base bias, adapter-down, adapter-up byte offsets.
    mr_uint4 offsets0;
    // adapter-bias basis byte offset, first generated coefficient, reserved.
    mr_uint4 offsets1;
} MRHyperPolicyLayerGPU;

typedef struct MR_ALIGN16 MRHyperPolicyEventGuardGPU {
    // phase, confidence, reserved, reserved.
    mr_float4 phase;
    // required contact-on mask, contact-off mask, dwell steps, event kind.
    mr_uint4 contract;
} MRHyperPolicyEventGuardGPU;

// Persistent event/phase state. rootFrame stores the world-to-canonical yaw
// quaternion captured at reset, so runtime signatures match ARDY's heading-
// canonical reference without copying or actuating the generated root.
typedef struct MR_ALIGN16 MRHyperPolicyPhaseStateGPU {
    // phase, aligned reference phase, alignment cost, generated phase rate.
    mr_float4 phase;
    // reference index, event index, event dwell, flags.
    mr_uint4 state;
    // world-to-canonical heading quaternion xyzw.
    mr_float4 rootFrame;
} MRHyperPolicyPhaseStateGPU;

typedef struct MR_ALIGN16 MRHyperPolicySignatureDispatchGPU {
    // environments, actions, signature width, contact tracks.
    mr_uint4 counts;
    // q stride, v stride, compact-contact stride, reset-mask stride.
    mr_uint4 strides;
    // q base, v base, compact-contact base, reset-mask base.
    mr_uint4 offsets;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRHyperPolicySignatureDispatchGPU;

typedef struct MR_ALIGN16 MRHyperPolicyPhaseDispatchGPU {
    // environments, signature width, forward-search frames, contact tracks.
    mr_uint4 counts;
    // signature env stride, contact-mask env stride, phase-state base, reserved.
    mr_uint4 strides;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRHyperPolicyPhaseDispatchGPU;

typedef struct MR_ALIGN16 MRHyperPolicyLayerDispatchGPU {
    // environments, layer index, input width, output width.
    mr_uint4 counts;
    // input env stride, output env stride, rank workspace stride, reserved.
    mr_uint4 strides;
    // input base, output base, rank-workspace base, reserved.
    mr_uint4 offsets;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRHyperPolicyLayerDispatchGPU;

typedef struct MR_ALIGN16 MRHyperPolicyActionDispatchGPU {
    // environments, actions, control step, program flags.
    mr_uint4 counts;
    // mean env stride, action/latent step stride, scalar step stride,
    // previous-action env stride.
    mr_uint4 strides;
    // actor mean base, action base, latent base, log-probability base.
    mr_uint4 offsets;
    // previous-action base, action-lower base, action-upper base, rate base.
    mr_uint4 bounds;
    // reset-mask base, reset-mask stride, phase-trace base, reserved.
    mr_uint4 reset;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
    mr_u64 randomSeed;
} MRHyperPolicyActionDispatchGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRHyperPolicyProgramHeaderGPU) == 224u);
static_assert(sizeof(MRHyperPolicyLayerGPU) == 48u);
static_assert(sizeof(MRHyperPolicyEventGuardGPU) == 32u);
static_assert(sizeof(MRHyperPolicyPhaseStateGPU) == 48u);
static_assert(sizeof(MRHyperPolicySignatureDispatchGPU) == 64u);
static_assert(sizeof(MRHyperPolicyPhaseDispatchGPU) == 48u);
static_assert(sizeof(MRHyperPolicyLayerDispatchGPU) == 64u);
static_assert(sizeof(MRHyperPolicyActionDispatchGPU) == 112u);
#endif
#endif
