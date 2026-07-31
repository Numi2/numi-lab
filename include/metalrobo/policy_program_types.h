#pragma once

#include "metalrobo/gpu_types.h"

#define MR_POLICY_PROGRAM_ABI_VERSION 4u

enum MRPolicyActivationOpcode : mr_u32 {
    MR_POLICY_ACTIVATION_IDENTITY = 0u,
    MR_POLICY_ACTIVATION_RELU = 1u,
    MR_POLICY_ACTIVATION_TANH = 2u,
    MR_POLICY_ACTIVATION_ELU = 3u,
    MR_POLICY_ACTIVATION_SILU = 4u,
};

enum MRPolicyDenseFlags : mr_u32 {
    MR_POLICY_DENSE_NORMALIZE_INPUT = 1u << 0u,
};

enum MRPolicyProgramFlags : mr_u32 {
    MR_POLICY_PROGRAM_HAS_CRITIC = 1u << 0u,
    MR_POLICY_PROGRAM_STOCHASTIC = 1u << 1u,
};

typedef struct MR_ALIGN16 MRPolicyProgramHeaderGPU {
    // actor layers, critic layers, actor observation width,
    // critic observation width.
    mr_uint4 counts0;
    // action width, maximum hidden width, program flags, reserved.
    mr_uint4 counts1;
    // actor layer table, critic layer table, actor mean, actor inverse stddev.
    mr_uint4 offsets0;
    // critic mean, critic inverse stddev, action bias, action scale.
    mr_uint4 offsets1;
    // action log standard deviation and reserved byte offsets.
    mr_uint4 offsets2;
    // observation clip, action clip, reserved, reserved.
    mr_float4 limits;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
    mr_u64 revision;
    mr_u64 reserved;
    // ABI version and reserved values.
    mr_uint4 abi;
} MRPolicyProgramHeaderGPU;

typedef struct MR_ALIGN16 MRPolicyDenseLayerGPU {
    // input width, output width, activation opcode, flags.
    mr_uint4 counts;
    // weight and bias byte offsets in the immutable arena.
    mr_uint4 offsets;
} MRPolicyDenseLayerGPU;

// One encoded dense layer for one control step.
typedef struct MR_ALIGN16 MRPolicyDenseDispatchGPU {
    // environments, input width, output width, activation opcode.
    mr_uint4 counts;
    // input environment stride, output environment stride,
    // input base, output base.
    mr_uint4 strides;
    // weights, bias, observation mean, observation inverse stddev.
    mr_uint4 offsets0;
    // SIMD width, SIMDgroups per threadgroup, flags, reserved.
    mr_uint4 offsets1;
    // observation clip, action clip, reserved, reserved.
    mr_float4 limits;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRPolicyDenseDispatchGPU;

// One environment per thread finalizes the actor distribution. It preserves
// the diagonal-Gaussian sample and exact old-policy log probability for PPO
// while publishing transformed actions to the task graph.
typedef struct MR_ALIGN16 MRPolicySampleDispatchGPU {
    // environments, actions, control step, program flags.
    mr_uint4 counts;
    // action step stride, scalar step stride, actor-mean environment stride,
    // reserved.
    mr_uint4 strides;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRPolicySampleDispatchGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRPolicyProgramHeaderGPU) == 144u);
static_assert(sizeof(MRPolicyDenseLayerGPU) == 32u);
static_assert(sizeof(MRPolicyDenseDispatchGPU) == 96u);
static_assert(sizeof(MRPolicySampleDispatchGPU) == 48u);
#endif
#endif
