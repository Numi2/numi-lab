#include "hyper_policy/HyperPolicy.00.metalinc"
#include "hyper_policy/HyperPolicy.01.metalinc"
#include "hyper_policy/HyperPolicy.03.metalinc"
#include "hyper_policy/HyperPolicy.02.metalinc"

#include "metalrobo/engine_types.h"

// Restore the full pre-policy state only for environments whose physical
// transaction was rejected. Accepted environments retain their new state.
kernel void mr_hyper_policy_restore_failed_step(
    device MRHyperPolicyPhaseStateGPU* phases [[buffer(0)]],
    device const MRHyperPolicyPhaseStateGPU* savedPhases [[buffer(1)]],
    device float* actions [[buffer(2)]],
    device const float* savedActions [[buffer(3)]],
    device const MRMetalWorldStatusGPU* world [[buffer(4)]],
    device const MRMetalWorldContactStatusGPU* contact [[buffer(5)]],
    constant uint4& counts [[buffer(6)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= counts.x ||
        (world[environment].code == MR_STEP_SUCCESS &&
         (counts.z == 0u || contact[environment].code == MR_STEP_SUCCESS))) return;
    phases[environment] = savedPhases[environment];
    for (uint action = 0u; action < counts.y; ++action) {
        const uint index = environment * counts.y + action;
        actions[index] = savedActions[index];
    }
}
