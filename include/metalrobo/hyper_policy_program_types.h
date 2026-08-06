#pragma once

#include "metalrobo/gpu_types.h"

// Shared C++/Metal ABI for a generated ARDY motion policy.  The hypernetwork
// never appears in this ABI: only authenticated base weights, adapter bases,
// generated phase-knot coefficients, references, and physical event guards are
// uploaded to the device.
#define MR_HYPER_POLICY_PROGRAM_ABI_VERSION 1u

#define MR_HYPER_POLICY_INVALID_INDEX 0xffffffffu

enum MRHyperPolicyProgramFlags : mr_u32 {
    MR_HYPER_POLICY_STOCHASTIC = 1u << 0u,
    MR_HYPER_POLICY_REFERENCE_RESIDUAL = 1u << 1u,
};

enum MRHyperPolicyPhaseFlags : mr_u32 {
    MR_HYPER_POLICY_PHASE_COMPLETE = 1u << 0u,
    MR_HYPER_POLICY_PHASE_GUARD_BLOCKED = 1u << 1u,
};

typedef struct MR_ALIGN16 MRHyperPolicyProgramHeaderGPU {
    // layers, actor observations, actions, generated coefficients.
    mr_uint4 counts0;
    // coefficient knots, reference frames, phase-signature width, maximum rank.
    mr_uint4 counts1;
    // event guards, contact tracks, program flags, reserved.
    mr_uint4 counts2;

    // layer table, observation mean, observation inverse stddev,
    // per-coefficient absolute limits.
    mr_uint4 offsets0;
    // action bias, action scale, knot phases, coefficient knots.
    mr_uint4 offsets1;
    // coefficient tangents, authority knots, authority tangents,
    // normalized phase-rate knots.
    mr_uint4 offsets2;
    // phase-rate tangents, reference phases, reference policy actions,
    // reference phase-alignment signatures.
    mr_uint4 offsets3;
    // signature weights, reference contact masks, event guards,
    // optional action log standard deviation.
    mr_uint4 offsets4;

    // observation clip, action clip, maximum phase advance per control step,
    // phase-alignment blend.
    mr_float4 limits0;
    // robust-alignment Huber delta, control dt, coefficient interpolation
    // epsilon, reserved.
    mr_float4 limits1;

    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
    mr_u64 sourceMotionFingerprint;
    mr_u64 revision;
    mr_uint4 abi;
} MRHyperPolicyProgramHeaderGPU;

typedef struct MR_ALIGN16 MRHyperPolicyLayerGPU {
    // input width, output width, rank, activation opcode.  Activation opcodes
    // match MRPolicyActivationOpcode.
    mr_uint4 counts;
    // base weight, base bias, adapter-down, adapter-up byte offsets.
    mr_uint4 offsets0;
    // adapter bias basis byte offset, coefficient vector offset, reserved,
    // reserved.
    mr_uint4 offsets1;
} MRHyperPolicyLayerGPU;

typedef struct MR_ALIGN16 MRHyperPolicyEventGuardGPU {
    // phase, confidence, reserved, reserved.
    mr_float4 phase;
    // required-on mask, required-off mask, minimum dwell steps, event kind.
    mr_uint4 contract;
} MRHyperPolicyEventGuardGPU;

typedef struct MR_ALIGN16 MRHyperPolicyPhaseStateGPU {
    // live phase, best aligned phase, alignment cost, generated phase rate.
    mr_float4 phase;
    // reference frame, next event, event dwell, flags.
    mr_uint4 state;
} MRHyperPolicyPhaseStateGPU;

typedef struct MR_ALIGN16 MRHyperPolicyPhaseDispatchGPU {
    // environments, phase signature width, forward-search frames,
    // measured contact track count.
    mr_uint4 counts;
    // signature environment stride, measured-contact environment stride,
    // phase-state base, reserved.
    mr_uint4 strides;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRHyperPolicyPhaseDispatchGPU;

typedef struct MR_ALIGN16 MRHyperPolicyLayerDispatchGPU {
    // environments, layer index, input width, output width.
    mr_uint4 counts;
    // input environment stride, output environment stride,
    // rank-workspace environment stride, reserved.
    mr_uint4 strides;
    // input base, output base, rank-workspace base, reserved.
    mr_uint4 offsets;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRHyperPolicyLayerDispatchGPU;

typedef struct MR_ALIGN16 MRHyperPolicyActionDispatchGPU {
    // environments, actions, control step, program flags.
    mr_uint4 counts;
    // actor-mean environment stride, action step stride, scalar step stride,
    // previous-action environment stride.
    mr_uint4 strides;
    // actor-mean base, action base, latent base, scalar base.
    mr_uint4 offsets;
    // previous-action base, action-lower base, action-upper base,
    // maximum-action-rate base.  The last three refer to separate bound
    // buffers, not the immutable arena.
    mr_uint4 bounds;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
    mr_u64 randomSeed;
    mr_u64 reserved;
} MRHyperPolicyActionDispatchGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRHyperPolicyProgramHeaderGPU) == 208u);
static_assert(sizeof(MRHyperPolicyLayerGPU) == 48u);
static_assert(sizeof(MRHyperPolicyEventGuardGPU) == 32u);
static_assert(sizeof(MRHyperPolicyPhaseStateGPU) == 32u);
static_assert(sizeof(MRHyperPolicyPhaseDispatchGPU) == 48u);
static_assert(sizeof(MRHyperPolicyLayerDispatchGPU) == 64u);
static_assert(sizeof(MRHyperPolicyActionDispatchGPU) == 96u);
#endif
#endif
