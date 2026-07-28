#include <metal_stdlib>

#include "metalrobo/engine_types.h"

using namespace metal;

namespace {

uint mapABAStatus(const uint code) {
    switch (code) {
    case MR_ABA_SUCCESS:
        return MR_STEP_SUCCESS;
    case MR_ABA_INVALID_DISPATCH:
    case MR_ABA_INVALID_MODEL:
    case MR_ABA_INVALID_QUATERNION:
        return MR_STEP_UNSUPPORTED;
    case MR_ABA_NONFINITE_INPUT:
        return MR_STEP_NONFINITE_INPUT;
    case MR_ABA_FACTORIZATION_FAILED:
        return MR_STEP_FACTORIZATION_FAILED;
    case MR_ABA_NONFINITE_RESULT:
        return MR_STEP_NONFINITE_RESULT;
    case MR_ABA_UNSUPPORTED_TOPOLOGY:
        return MR_STEP_UNSUPPORTED;
    default:
        return MR_STEP_UNSUPPORTED;
    }
}

} // namespace

// Transactional publication adapter for the MLX custom primitive. q/v inputs
// are the immutable control-step checkpoint. A failed microstep restores that
// checkpoint, zeros acceleration, and leaves the failure latched for every
// remaining encoded microstep.
kernel void mr_mlx_world_commit_aba(
    constant MRMLXWorldStepDispatchGPU& dispatch [[buffer(0)]],
    device const float* checkpointQ [[buffer(1)]],
    device const float* checkpointV [[buffer(2)]],
    device const float* candidateQ [[buffer(3)]],
    device const float* candidateV [[buffer(4)]],
    device const float* candidateAcceleration [[buffer(5)]],
    device const MRABAStatusGPU* abaStatuses [[buffer(6)]],
    device float* destinationQ [[buffer(7)]],
    device float* destinationV [[buffer(8)]],
    device float* publishedAcceleration [[buffer(9)]],
    device MRMLXWorldStepStatusGPU* statuses [[buffer(10)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }

    MRMLXWorldStepStatusGPU status{};
    if (dispatch.physicsSubstep != 0u) {
        status = statuses[environment];
    } else {
        status.code = MR_STEP_SUCCESS;
        status.abaCode = MR_ABA_SUCCESS;
        status.failingSubstep = MR_INVALID_INDEX;
        status.failingIndex = MR_INVALID_INDEX;
    }

    const bool validDispatch =
        dispatch.environmentCount > 0u &&
        dispatch.nq > 0u &&
        dispatch.nv > 0u &&
        dispatch.physicsSubsteps > 0u &&
        dispatch.physicsSubstep < dispatch.physicsSubsteps &&
        dispatch.reserved0 == 0u &&
        dispatch.reserved1 == 0u;
    const MRABAStatusGPU aba = abaStatuses[environment];
    const bool validABA =
        aba.environment == environment &&
        aba.articulationIndex == dispatch.articulationIndex &&
        aba.code <= MR_ABA_UNSUPPORTED_TOPOLOGY;
    if (status.code == MR_STEP_SUCCESS &&
        (!validDispatch || !validABA ||
         aba.code != MR_ABA_SUCCESS)) {
        status.code =
            validDispatch && validABA
            ? mapABAStatus(aba.code)
            : MR_STEP_UNSUPPORTED;
        status.abaCode =
            validABA ? aba.code : MR_ABA_INVALID_DISPATCH;
        status.failingSubstep = dispatch.physicsSubstep;
        status.failingIndex =
            validABA ? aba.failingIndex : MR_INVALID_INDEX;
    }

    const bool publishCandidate =
        status.code == MR_STEP_SUCCESS;
    const uint qBase = environment * dispatch.nq;
    const uint vBase = environment * dispatch.nv;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        destinationQ[qBase + coordinate] =
            publishCandidate
            ? candidateQ[qBase + coordinate]
            : checkpointQ[qBase + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        destinationV[vBase + coordinate] =
            publishCandidate
            ? candidateV[vBase + coordinate]
            : checkpointV[vBase + coordinate];
        publishedAcceleration[vBase + coordinate] =
            publishCandidate
            ? candidateAcceleration[vBase + coordinate]
            : 0.0f;
    }
    if (publishCandidate) {
        ++status.successfulSubsteps;
    }
    statuses[environment] = status;
}
