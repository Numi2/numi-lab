#include <metal_stdlib>

#include "metalrobo/policy_program_types.h"
#include "metalrobo/task_program_types.h"

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

inline ulong policyMix64(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value =
        (value ^ (value >> 30u)) *
        0xbf58476d1ce4e5b9ul;
    value =
        (value ^ (value >> 27u)) *
        0x94d049bb133111ebul;
    return value ^ (value >> 31u);
}

inline float policyUniform01(const ulong key) {
    const uint sample =
        uint(policyMix64(key) >> 40u);
    return (float(sample) + 0.5f) *
        (1.0f / 16777216.0f);
}

inline float policyNormal(
    const ulong key,
    const uint action
) {
    const ulong channel =
        key ^ policyMix64(
            ulong(action) * 2ul
        );
    const float first = policyUniform01(channel);
    const float second = policyUniform01(
        channel ^ 0xd2b74407b1ce6e93ul
    );
    return sqrt(-2.0f * log(first)) *
        cos(6.2831853071795864769f * second);
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
    device const MRTaskTransitionGPU* transitions [[buffer(5)]],
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
    if ((dispatch.offsets1.z &
         MR_POLICY_DENSE_TIMEOUT_ONLY) != 0u &&
        transitions[
            dispatch.offsets1.x + environment
        ].termination.y == 0u) {
        return;
    }
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
    output[
        dispatch.strides.w +
        environment * dispatch.strides.y +
        neuron
    ] = value;
}

kernel void mr_policy_sample_and_score(
    device const MRPolicyProgramHeaderGPU& program
        [[buffer(0)]],
    device const uchar* arena [[buffer(1)]],
    constant MRPolicySampleDispatchGPU& dispatch
        [[buffer(2)]],
    device const MRTaskDispatchGPU& task [[buffer(3)]],
    device const MRTaskStateGPU* taskStates [[buffer(4)]],
    device const float* actorMean [[buffer(5)]],
    device float* actions [[buffer(6)]],
    device float* latents [[buffer(7)]],
    device float* logProbabilities [[buffer(8)]],
    device float* values [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        program.abi.x != MR_POLICY_PROGRAM_ABI_VERSION ||
        dispatch.policyFingerprint !=
            program.policyFingerprint ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        task.taskFingerprint != program.taskFingerprint ||
        dispatch.counts.y != program.counts1.x ||
        dispatch.counts.w != program.counts1.z) {
        return;
    }

    const uint actionBase =
        dispatch.counts.z * dispatch.strides.x +
        environment * dispatch.counts.y;
    const uint scalarIndex =
        dispatch.counts.z * dispatch.strides.y +
        environment;
    const uint meanBase =
        environment * dispatch.strides.z;
    device const float* actionBias =
        policyTable<float>(
            arena,
            program.offsets1.z
        );
    device const float* actionScale =
        policyTable<float>(
            arena,
            program.offsets1.w
        );
    const bool stochastic =
        (dispatch.counts.w &
         MR_POLICY_PROGRAM_STOCHASTIC) != 0u;
    device const float* logStandardDeviation =
        stochastic
        ? policyTable<float>(
              arena,
              program.offsets2.x
          )
        : nullptr;
    const MRTaskStateGPU state =
        taskStates[environment];
    ulong randomKey = task.seed;
    randomKey ^= policyMix64(
        (ulong(environment) << 32u) |
        ulong(state.episode.y)
    );
    randomKey ^= policyMix64(
        (ulong(state.episode.x) << 32u) |
        ulong(dispatch.counts.z)
    );
    randomKey ^= policyMix64(program.revision);

    constexpr float kHalfLogTwoPi =
        0.91893853320467274178f;
    float logProbability = 0.0f;
    for (uint action = 0u;
         action < dispatch.counts.y;
         ++action) {
        const float mean = actorMean[meanBase + action];
        float latent = mean;
        if (stochastic) {
            const float logStd =
                logStandardDeviation[action];
            const float normal =
                policyNormal(randomKey, action);
            latent = fma(exp(logStd), normal, mean);
            logProbability +=
                -0.5f * normal * normal -
                logStd -
                kHalfLogTwoPi;
        }
        latents[actionBase + action] = latent;
        actions[actionBase + action] = clamp(
            fma(
                actionScale[action],
                latent,
                actionBias[action]
            ),
            -program.limits.y,
            program.limits.y
        );
    }
    logProbabilities[scalarIndex] = logProbability;
    if ((dispatch.counts.w &
         MR_POLICY_PROGRAM_HAS_CRITIC) == 0u) {
        values[scalarIndex] = 0.0f;
    }
}
