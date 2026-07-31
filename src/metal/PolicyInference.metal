#include <metal_stdlib>

#include "metalrobo/policy_program_types.h"

using namespace metal;

namespace {

template <typename T>
inline device const T* policyTable(
    device const uchar* arena,
    const uint byteOffset
) {
    return reinterpret_cast<device const T*>(
        arena + byteOffset
    );
}

inline float policyActivation(
    const uint operation,
    const float value
) {
    switch (operation) {
    case MR_POLICY_ACTIVATION_IDENTITY:
        return value;
    case MR_POLICY_ACTIVATION_RELU:
        return max(value, 0.0f);
    case MR_POLICY_ACTIVATION_TANH:
        return tanh(value);
    case MR_POLICY_ACTIVATION_ELU:
        return value >= 0.0f ? value : exp(value) - 1.0f;
    case MR_POLICY_ACTIVATION_SILU:
        return value / (1.0f + exp(-value));
    default:
        return 0.0f;
    }
}

} // namespace

// Shared dense policy operator. Robot and joint layout never appear here;
// compiled dimensions and immutable byte offsets define each layer.
kernel void mr_policy_dense_layer(
    device const MRPolicyProgramHeaderGPU& program
        [[buffer(0)]],
    device const uchar* arena [[buffer(1)]],
    constant MRPolicyDenseDispatchGPU& dispatch
        [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    const uint flatOutput [[thread_position_in_grid]]
) {
    const uint outputElements =
        dispatch.counts.x * dispatch.counts.z;
    if (flatOutput >= outputElements ||
        program.abi.x != MR_POLICY_PROGRAM_ABI_VERSION ||
        dispatch.policyFingerprint !=
            program.policyFingerprint ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.counts.y == 0u ||
        dispatch.counts.z == 0u) {
        return;
    }

    const uint environment =
        flatOutput / dispatch.counts.z;
    const uint neuron =
        flatOutput - environment * dispatch.counts.z;
    device const float* weights = policyTable<float>(
        arena,
        dispatch.offsets0.x
    );
    device const float* bias = policyTable<float>(
        arena,
        dispatch.offsets0.y
    );
    const uint inputBase =
        dispatch.strides.z +
        environment * dispatch.strides.x;
    float value = bias[neuron];
    const uint weightBase = neuron * dispatch.counts.y;
    for (uint feature = 0u;
         feature < dispatch.counts.y;
         ++feature) {
        float sample = input[inputBase + feature];
        if ((dispatch.offsets1.z &
             MR_POLICY_DENSE_NORMALIZE_INPUT) != 0u) {
            device const float* mean = policyTable<float>(
                arena,
                dispatch.offsets0.z
            );
            device const float* inverseStd =
                policyTable<float>(
                    arena,
                    dispatch.offsets0.w
                );
            sample = clamp(
                (sample - mean[feature]) *
                    inverseStd[feature],
                -dispatch.limits.x,
                dispatch.limits.x
            );
        }
        value = fma(
            weights[weightBase + feature],
            sample,
            value
        );
    }
    value = policyActivation(
        dispatch.counts.w,
        value
    );
    if ((dispatch.offsets1.z &
         MR_POLICY_DENSE_TRANSFORM_OUTPUT) != 0u) {
        device const float* actionBias =
            policyTable<float>(
                arena,
                dispatch.offsets1.x
            );
        device const float* actionScale =
            policyTable<float>(
                arena,
                dispatch.offsets1.y
            );
        value =
            actionBias[neuron] +
            actionScale[neuron] * value;
    }
    if ((dispatch.offsets1.z &
         MR_POLICY_DENSE_CLAMP_OUTPUT) != 0u) {
        value = clamp(
            value,
            -dispatch.limits.y,
            dispatch.limits.y
        );
    }
    output[
        dispatch.strides.w +
        environment * dispatch.strides.y +
        neuron
    ] = value;
}
