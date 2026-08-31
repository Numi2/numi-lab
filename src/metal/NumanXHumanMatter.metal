#include <metal_stdlib>

#include "metalrobo/numanx_human_matter_adapter_gpu.h"
#include "numi/matter/shared.h"

using namespace metal;

namespace {

constant ulong kFNVOffset = 14695981039346656037ul;
constant ulong kFNVPrime = 1099511628211ul;
constant uint kJointTransactionVersion = 1u;
constant uint kPhysicsStateDomain = 0x4e585053u; // "NXPS"

inline void fnvMixUInt(thread ulong& hash, const uint value) {
    for (uint byte = 0u; byte < 4u; ++byte) {
        hash ^= ulong((value >> (byte * 8u)) & 0xffu);
        hash *= kFNVPrime;
    }
}

inline void fnvMixULong(thread ulong& hash, const ulong value) {
    for (uint byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xfful;
        hash *= kFNVPrime;
    }
}

inline bool validDispatch(
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch
) {
    return dispatch.abiVersion ==
            MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION &&
        dispatch.environmentCount == 1u &&
        dispatch.expectedMatterCompletedMicrosteps != 0u &&
        dispatch.jointStatusStride != 0u &&
        dispatch.standStatusStride != 0u &&
        dispatch.matterOutcomeStride != 0u &&
        dispatch.acceptedTokenStrideBytes ==
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES &&
        dispatch.worldStatusStride != 0u &&
        dispatch.acceptedStateProofStride != 0u &&
        dispatch.flags == 0u && dispatch.reserved0 == 0u &&
        dispatch.physicsSubsteps == 1u &&
        dispatch.physicsSubstep == 0u &&
        dispatch.programFingerprint != 0u &&
        dispatch.transactionFingerprint != 0u &&
        dispatch.substepFingerprint != 0u &&
        dispatch.acceptedTimestampMicroseconds != 0u &&
        dispatch.physicsGeneration != 0u &&
        dispatch.linearizationEpoch != 0u &&
        dispatch.slotGeneration != 0u &&
        dispatch.matterSourcePhysicsFingerprint != 0u &&
        dispatch.matterDeviceProgramFingerprint != 0u;
}

inline ulong proofFingerprint(
    thread const MRNumanXAcceptedStateProofGPU& proof
) {
    ulong hash = kFNVOffset;
    fnvMixUInt(hash, proof.abiVersion);
    fnvMixUInt(hash, proof.structSize);
    fnvMixUInt(hash, proof.status);
    fnvMixUInt(hash, proof.environment);
    fnvMixULong(hash, proof.transactionFingerprint);
    fnvMixULong(hash, proof.substepFingerprint);
    fnvMixULong(hash, proof.acceptedTimestampMicroseconds);
    fnvMixULong(hash, proof.physicsGeneration);
    fnvMixULong(hash, proof.humanStateFingerprint);
    fnvMixULong(hash, proof.matterStateFingerprint);
    fnvMixULong(hash, proof.physicsStateFingerprint);
    fnvMixULong(hash, proof.matterSourcePhysicsFingerprint);
    fnvMixULong(hash, proof.matterDeviceProgramFingerprint);
    fnvMixULong(hash, proof.linearizationEpoch);
    fnvMixULong(hash, proof.slotGeneration);
    fnvMixULong(hash, proof.adapterProgramFingerprint);
    fnvMixULong(hash, proof.transactionPolicyFingerprint);
    return hash;
}

inline ulong combinedPhysicsStateFingerprint(
    thread const MRNumanXAcceptedStateProofGPU& proof,
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch
) {
    ulong hash = kFNVOffset;
    fnvMixUInt(hash, kPhysicsStateDomain);
    fnvMixUInt(hash, MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION);
    fnvMixULong(hash, proof.humanStateFingerprint);
    fnvMixULong(hash, proof.matterStateFingerprint);
    fnvMixULong(hash, dispatch.matterSourcePhysicsFingerprint);
    fnvMixULong(hash, dispatch.matterDeviceProgramFingerprint);
    fnvMixULong(hash, dispatch.stateProofProgramFingerprint);
    fnvMixULong(hash, dispatch.programFingerprint);
    fnvMixULong(hash, proof.transactionPolicyFingerprint);
    fnvMixULong(hash, dispatch.transactionFingerprint);
    fnvMixULong(hash, dispatch.substepFingerprint);
    fnvMixULong(hash, dispatch.physicsGeneration);
    fnvMixUInt(hash, proof.environment);
    return hash;
}

inline ulong acceptedTokenFingerprint(
    thread const MRNumanXAcceptedPhysicsStateTokenGPU& token
) {
    // This is the exact NumiBrain ABI relation.
    ulong hash = kFNVOffset;
    fnvMixUInt(hash, kJointTransactionVersion);
    fnvMixULong(hash, token.transactionFingerprint);
    fnvMixULong(hash, token.substepFingerprint);
    fnvMixULong(hash, token.physicsStateFingerprint);
    fnvMixULong(hash, token.acceptedTimestampMicroseconds);
    fnvMixULong(hash, token.physicsGeneration);
    fnvMixUInt(hash, token.environmentIdentifier);
    fnvMixUInt(hash, token.flags);
    fnvMixULong(hash, token.reserved);
    return hash;
}

inline bool validAcceptedStateProof(
    thread const MRNumanXAcceptedStateProofGPU& proof,
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch,
    const uint environment
) {
    const ulong expectedProofFingerprint = proofFingerprint(proof);
    const ulong expectedStateFingerprint =
        combinedPhysicsStateFingerprint(proof, dispatch);
    return validDispatch(dispatch) &&
        dispatch.stateProofProgramFingerprint != 0u &&
        proof.abiVersion == MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION &&
        proof.structSize == MR_NUMANX_ACCEPTED_STATE_PROOF_BYTES &&
        proof.status == MR_NUMANX_ACCEPTED_STATE_PROOF_VALID &&
        proof.environment == environment &&
        proof.transactionFingerprint == dispatch.transactionFingerprint &&
        proof.substepFingerprint == dispatch.substepFingerprint &&
        proof.acceptedTimestampMicroseconds ==
            dispatch.acceptedTimestampMicroseconds &&
        proof.physicsGeneration == dispatch.physicsGeneration &&
        proof.humanStateFingerprint != 0ul &&
        proof.matterStateFingerprint != 0ul &&
        proof.physicsStateFingerprint == expectedStateFingerprint &&
        proof.matterSourcePhysicsFingerprint ==
            dispatch.matterSourcePhysicsFingerprint &&
        proof.matterDeviceProgramFingerprint ==
            dispatch.matterDeviceProgramFingerprint &&
        proof.linearizationEpoch == dispatch.linearizationEpoch &&
        proof.slotGeneration == dispatch.slotGeneration &&
        proof.adapterProgramFingerprint == dispatch.programFingerprint &&
        proof.transactionPolicyFingerprint != 0ul &&
        proof.proofFingerprint != 0ul &&
        proof.proofFingerprint == expectedProofFingerprint;
}

} // namespace

kernel void numanx_human_matter_prepare_world_status(
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch [[buffer(0)]],
    device MRMetalWorldStatusGPU* worldStatuses [[buffer(1)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    MRMetalWorldStatusGPU status = {};
    status.code = validDispatch(dispatch)
        ? MR_STEP_SUCCESS
        : MR_STEP_UNSUPPORTED;
    status.environment = environment;
    status.controlStep = dispatch.controlStep;
    status.successfulSubsteps = dispatch.physicsSubstep;
    status.failingSubstep = MR_INVALID_INDEX;
    status.failingIndex = MR_INVALID_INDEX;
    worldStatuses[environment * dispatch.worldStatusStride] = status;
}

kernel void numanx_human_matter_map_human_status(
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumiHumanStandStatusGPU* standStatuses [[buffer(1)]],
    device MRMetalWorldStatusGPU* worldStatuses [[buffer(2)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const MRNumiHumanStandStatusGPU human =
        standStatuses[environment * dispatch.standStatusStride];
    // This is deliberately provisional. Matter must first reconcile every
    // success-surviving mutation into its prepared state; only the later proof
    // gate may turn the joint candidate into a physical-prepare token.
    const bool accepted = human.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        human.environment == environment &&
        human.completedSteps == 1u;
    MRMetalWorldStatusGPU status = {};
    status.code = accepted
        ? MR_STEP_SUCCESS
        : (human.code == MR_NUMI_HUMAN_STAND_SUCCESS
            ? MR_STEP_UNSUPPORTED
            : MR_STEP_FACTORIZATION_FAILED);
    status.environment = environment;
    status.controlStep = dispatch.controlStep;
    status.successfulSubsteps = accepted
        ? dispatch.physicsSubstep + 1u
        : dispatch.physicsSubstep;
    status.abaCode = human.code;
    status.failingSubstep = accepted
        ? MR_INVALID_INDEX
        : dispatch.physicsSubstep;
    status.failingIndex = accepted ? MR_INVALID_INDEX : human.failingIndex;
    status.diagnostics = human.factorAndAssistance;
    worldStatuses[environment * dispatch.worldStatusStride] = status;
}

kernel void numanx_human_matter_capture_outcome(
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch [[buffer(0)]],
    device const NMMatterStatusGPU* matterStatuses [[buffer(1)]],
    device MRNumanXCoupledMatterOutcomeGPU* outcomes [[buffer(2)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const NMMatterStatusGPU matter = matterStatuses[environment];
    MRNumanXCoupledMatterOutcomeGPU outcome = {};
    outcome.code = validDispatch(dispatch)
        ? matter.code
        : NM_STATUS_INVALID_DISPATCH;
    outcome.environment = environment;
    outcome.completedMicrosteps = matter.completedMicrosteps;
    outcomes[environment * dispatch.matterOutcomeStride] = outcome;
}

kernel void numanx_human_matter_write_prepared_token(
    constant MRNumanXHumanMatterAdapterDispatchGPU& dispatch [[buffer(0)]],
    device MRNumanXCoupledHumanStatusGPU* jointStatuses [[buffer(1)]],
    device const MRNumanXAcceptedStateProofGPU* proofs [[buffer(2)]],
    device uchar* acceptedTokens [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const uint tokenBase =
        environment * dispatch.acceptedTokenStrideBytes;
    device MRNumanXAcceptedPhysicsStateTokenGPU* token =
        reinterpret_cast<device MRNumanXAcceptedPhysicsStateTokenGPU*>(
            acceptedTokens + tokenBase);
    *token = {};

    device MRNumanXCoupledHumanStatusGPU& joint =
        jointStatuses[environment * dispatch.jointStatusStride];
    const MRNumanXAcceptedStateProofGPU proof =
        proofs[environment * dispatch.acceptedStateProofStride];
    const bool proofAvailable =
        proof.status != MR_NUMANX_ACCEPTED_STATE_PROOF_PENDING;
    const bool valid =
        validAcceptedStateProof(proof, dispatch, environment);
    if (!valid) {
        // Preserve a genuine Matter failure.  A staged/reconciled rejection
        // carrying Matter's success code is not a complete reason, however:
        // fail it explicitly at this final proof gate so no rejected joint
        // transaction can masquerade as Matter success.
        if (joint.decision == MR_NUMANX_COUPLED_HUMAN_ACCEPT ||
            (joint.decision == MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER &&
             joint.matterCode == dispatch.matterSuccessCode)) {
            joint.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
            joint.matterCode = proofAvailable
                ? MR_NUMANX_HUMAN_MATTER_ADAPTER_INVALID_STATE_PROOF
                : MR_NUMANX_HUMAN_MATTER_ADAPTER_MISSING_STATE_PROOF;
        }
        return;
    }
    if (joint.decision != MR_NUMANX_COUPLED_HUMAN_ACCEPT) return;

    MRNumanXAcceptedPhysicsStateTokenGPU accepted = {};
    accepted.transactionFingerprint = dispatch.transactionFingerprint;
    accepted.substepFingerprint = dispatch.substepFingerprint;
    accepted.physicsStateFingerprint = proof.physicsStateFingerprint;
    accepted.acceptedTimestampMicroseconds =
        dispatch.acceptedTimestampMicroseconds;
    accepted.physicsGeneration = dispatch.physicsGeneration;
    accepted.environmentIdentifier =
        dispatch.environmentIdentifierBase + environment;
    accepted.tokenFingerprint = acceptedTokenFingerprint(accepted);
    if (accepted.tokenFingerprint == 0ul) {
        joint.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
        joint.matterCode =
            MR_NUMANX_HUMAN_MATTER_ADAPTER_INVALID_STATE_PROOF;
        return;
    }
    *token = accepted;
}
