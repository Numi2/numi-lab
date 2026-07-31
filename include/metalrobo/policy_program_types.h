#pragma once

#include "metalrobo/gpu_types.h"

#define MR_POLICY_PROGRAM_ABI_VERSION 1u

enum MRPolicyActivationOpcode : mr_u32 {
    MR_POLICY_ACTIVATION_IDENTITY = 0u,
    MR_POLICY_ACTIVATION_RELU = 1u,
    MR_POLICY_ACTIVATION_TANH = 2u,
    MR_POLICY_ACTIVATION_ELU = 3u,
    MR_POLICY_ACTIVATION_SILU = 4u,
};

enum MRPolicyDenseFlags : mr_u32 {
    MR_POLICY_DENSE_NORMALIZE_INPUT = 1u << 0u,
    MR_POLICY_DENSE_TRANSFORM_OUTPUT = 1u << 1u,
    MR_POLICY_DENSE_CLAMP_OUTPUT = 1u << 2u,
};

typedef struct MR_ALIGN16 MRPolicyProgramHeaderGPU {
    // dense layers, observation width, action width, maximum hidden width.
    mr_uint4 counts;
    // layer table, observation mean, observation inverse stddev, action bias.
    mr_uint4 offsets0;
    // action scale and reserved byte offsets.
    mr_uint4 offsets1;
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
    // action bias, action scale, flags, reserved.
    mr_uint4 offsets1;
    // observation clip, action clip, reserved, reserved.
    mr_float4 limits;
    mr_u64 policyFingerprint;
    mr_u64 taskFingerprint;
} MRPolicyDenseDispatchGPU;

#ifndef __METAL_VERSION__
#ifdef __cplusplus
static_assert(sizeof(MRPolicyProgramHeaderGPU) == 112u);
static_assert(sizeof(MRPolicyDenseLayerGPU) == 32u);
static_assert(sizeof(MRPolicyDenseDispatchGPU) == 96u);
#endif
#endif
