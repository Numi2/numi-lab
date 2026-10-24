#include <metal_stdlib>
using namespace metal;
#include "../src/metal/fem_backtracking_failure_witness.metalinc"

struct NMBacktrackingWitnessCouponInput {
    float determinantValue;
    float currentJ;
    uint globalTet;
    uint constitutiveValid;
    float lowerJ;
    float upperJ;
};

kernel void nm_fem_backtracking_witness_coupon(
    device const NMBacktrackingWitnessCouponInput* inputs [[buffer(0)]],
    device float4* outputs [[buffer(1)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    const NMBacktrackingWitnessCouponInput input = inputs[index];
    const uint reason = nmFEMBacktrackingFailureReason(
        input.determinantValue, input.lowerJ, input.upperJ,
        input.constitutiveValid != 0u);
    const uint firstBad = reason != 0u ? input.globalTet : 0xffffffffu;
    const uint selectedTet = simd_min(firstBad);
    const float selectedTrialJ = simd_sum(
        firstBad == selectedTet ? input.determinantValue : 0.0f);
    const float selectedCurrentJ = simd_sum(
        firstBad == selectedTet ? input.currentJ : 0.0f);
    const uint selectedReason = simd_sum(
        firstBad == selectedTet ? reason : 0u);
    const float validMinimumJ = simd_min(
        reason == 0u ? input.determinantValue : INFINITY);
    if (lane == 0u) {
        outputs[simdgroup] = selectedTet == 0xffffffffu
            ? float4(validMinimumJ, 0.0f, 0.0f, 0.0f)
            : float4(as_type<float>(selectedTet), selectedTrialJ,
                     selectedCurrentJ, float(selectedReason));
        if (simdgroup == 0u) {
            const NMFEMBacktrackingFailureWitnessTuple noTrial =
                nmFEMBacktrackingEmptyFailureWitness();
            outputs[9] = float4(noTrial.trialJ, noTrial.currentJ,
                                noTrial.firstBad, float(noTrial.reason));
        }
    }
}
