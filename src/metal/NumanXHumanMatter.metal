#include <metal_stdlib>

#include "metalrobo/numanx_human_matter_adapter_gpu.h"
#include "metalrobo/numanx_human_io_gpu.h"
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
    const bool familyValid =
        (dispatch.abiVersion ==
                MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION &&
         dispatch.reserved0 == 0u) ||
        (dispatch.abiVersion ==
                MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION &&
         dispatch.reserved0 == sizeof(
             MRNumanXHumanMatterAdapterDispatchGPUV2));
    return familyValid &&
        dispatch.environmentCount == 1u &&
        dispatch.expectedMatterCompletedMicrosteps != 0u &&
        dispatch.jointStatusStride != 0u &&
        dispatch.standStatusStride != 0u &&
        dispatch.matterOutcomeStride != 0u &&
        dispatch.acceptedTokenStrideBytes ==
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES &&
        dispatch.worldStatusStride != 0u &&
        dispatch.acceptedStateProofStride != 0u &&
        dispatch.flags == 0u &&
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

inline ulong exactInboundAuthorityFingerprint(
    thread const MRNumanXExactInboundAuthorityGPUV2& authority
) {
    ulong hash = kFNVOffset;
    fnvMixUInt(
        hash, MR_NUMANX_FINGERPRINT_DOMAIN_EXACT_INBOUND_AUTHORITY_V2);
    fnvMixUInt(hash, authority.abiVersion);
    fnvMixUInt(hash, authority.structSize);
    fnvMixUInt(hash, authority.clockDomain);
    fnvMixUInt(hash, authority.clockQuantumNanoseconds);
    fnvMixULong(hash, authority.acceptedBrainTimestampNanoseconds);
    fnvMixULong(hash, authority.brainGeneration);
    fnvMixULong(hash, authority.transactionFingerprint);
    fnvMixULong(hash, authority.substepFingerprint);
    fnvMixULong(hash, authority.motorCandidateFingerprint);
    fnvMixULong(hash, authority.motorOutputFingerprint);
    fnvMixULong(hash, authority.motorProfileFingerprint);
    fnvMixULong(hash, authority.motorReadyGateFingerprint);
    fnvMixULong(hash, authority.brainProgramFingerprint);
    fnvMixULong(hash, authority.fastProgramFingerprint);
    fnvMixULong(hash, authority.decisionGateFingerprint);
    return hash;
}

inline bool validExactDispatch(
    constant MRNumanXHumanMatterAdapterDispatchGPUV2& dispatch
) {
    return dispatch.abiVersion ==
            MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION &&
        dispatch.structSize == sizeof(dispatch) &&
        dispatch.environmentCount == 1u &&
        dispatch.expectedMatterCompletedMicrosteps != 0u &&
        dispatch.jointStatusStride != 0u &&
        dispatch.standStatusStride != 0u &&
        dispatch.matterOutcomeStride != 0u &&
        dispatch.acceptedTokenStrideBytes ==
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_V2_BYTES &&
        dispatch.worldStatusStride != 0u &&
        dispatch.acceptedStateProofStride != 0u &&
        dispatch.flags == 0u && dispatch.physicsSubsteps == 1u &&
        dispatch.physicsSubstep == 0u &&
        dispatch.programFingerprint != 0u &&
        dispatch.transactionFingerprint != 0u &&
        dispatch.substepFingerprint != 0u &&
        dispatch.acceptedTimestampNanoseconds != 0u &&
        dispatch.acceptedTimestampNanoseconds >
            dispatch.acceptedBrainTimestampNanoseconds &&
        dispatch.physicsGeneration != 0u &&
        dispatch.linearizationEpoch != 0u &&
        dispatch.slotGeneration != 0u &&
        dispatch.matterSourcePhysicsFingerprint != 0u &&
        dispatch.matterDeviceProgramFingerprint != 0u &&
        dispatch.stateProofProgramFingerprint != 0u &&
        dispatch.clockDomain ==
            MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        dispatch.clockQuantumNanoseconds ==
            MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        dispatch.inboundAuthorityByteCount ==
            sizeof(MRNumanXExactInboundAuthorityGPUV2) &&
        dispatch.reserved0 == 0u &&
        dispatch.motorCandidateFingerprint != 0u &&
        dispatch.brainGeneration != 0u &&
        dispatch.humanIOProgramFingerprint != 0u &&
        dispatch.inboundAuthorityGPUAddress != 0u &&
        dispatch.authorityRangeIdentityFingerprint != 0u;
}

inline bool validExactInboundAuthority(
    thread const MRNumanXExactInboundAuthorityGPUV2& authority,
    constant MRNumanXHumanMatterAdapterDispatchGPUV2& dispatch
) {
    const ulong expected = exactInboundAuthorityFingerprint(authority);
    return authority.abiVersion ==
            MR_NUMANX_EXACT_INBOUND_AUTHORITY_ABI_VERSION_V2 &&
        authority.structSize == sizeof(authority) &&
        authority.clockDomain == dispatch.clockDomain &&
        authority.clockQuantumNanoseconds ==
            dispatch.clockQuantumNanoseconds &&
        authority.acceptedBrainTimestampNanoseconds ==
            dispatch.acceptedBrainTimestampNanoseconds &&
        authority.brainGeneration == dispatch.brainGeneration &&
        authority.transactionFingerprint ==
            dispatch.transactionFingerprint &&
        authority.substepFingerprint == dispatch.substepFingerprint &&
        authority.motorCandidateFingerprint ==
            dispatch.motorCandidateFingerprint &&
        authority.motorOutputFingerprint != 0ul &&
        authority.motorProfileFingerprint != 0ul &&
        authority.motorReadyGateFingerprint != 0ul &&
        authority.brainProgramFingerprint != 0ul &&
        authority.fastProgramFingerprint != 0ul &&
        authority.decisionGateFingerprint != 0ul &&
        authority.inboundAuthorityFingerprint != 0ul &&
        authority.inboundAuthorityFingerprint == expected;
}

inline ulong exactPhysicsStateFingerprint(
    thread const MRNumanXAcceptedStateProofGPUV2& proof
) {
    ulong hash = kFNVOffset;
    fnvMixUInt(hash, MR_NUMANX_FINGERPRINT_DOMAIN_PHYSICS_STATE_V2);
    fnvMixUInt(hash, MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION);
    fnvMixUInt(hash, proof.clockDomain);
    fnvMixUInt(hash, proof.clockQuantumNanoseconds);
    fnvMixULong(hash, proof.humanStateFingerprint);
    fnvMixULong(hash, proof.matterStateFingerprint);
    fnvMixULong(hash, proof.matterSourcePhysicsFingerprint);
    fnvMixULong(hash, proof.matterDeviceProgramFingerprint);
    fnvMixULong(hash, proof.stateProofProgramFingerprint);
    fnvMixULong(hash, proof.adapterProgramFingerprint);
    fnvMixULong(hash, proof.transactionPolicyFingerprint);
    fnvMixULong(hash, proof.motorCandidateFingerprint);
    fnvMixULong(hash, proof.inboundAuthorityFingerprint);
    fnvMixULong(hash, proof.transactionFingerprint);
    fnvMixULong(hash, proof.substepFingerprint);
    fnvMixULong(hash, proof.acceptedTimestampNanoseconds);
    fnvMixULong(hash, proof.physicsGeneration);
    fnvMixUInt(hash, proof.environment);
    return hash;
}

inline ulong exactProofFingerprint(
    thread const MRNumanXAcceptedStateProofGPUV2& proof
) {
    ulong hash = kFNVOffset;
    fnvMixUInt(
        hash, MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_STATE_PROOF_V2);
    fnvMixUInt(hash, proof.abiVersion);
    fnvMixUInt(hash, proof.structSize);
    fnvMixUInt(hash, proof.status);
    fnvMixUInt(hash, proof.environment);
    fnvMixULong(hash, proof.transactionFingerprint);
    fnvMixULong(hash, proof.substepFingerprint);
    fnvMixULong(hash, proof.acceptedTimestampNanoseconds);
    fnvMixULong(hash, proof.physicsGeneration);
    fnvMixUInt(hash, proof.clockDomain);
    fnvMixUInt(hash, proof.clockQuantumNanoseconds);
    fnvMixULong(hash, proof.humanStateFingerprint);
    fnvMixULong(hash, proof.matterStateFingerprint);
    fnvMixULong(hash, proof.physicsStateFingerprint);
    fnvMixULong(hash, proof.matterSourcePhysicsFingerprint);
    fnvMixULong(hash, proof.matterDeviceProgramFingerprint);
    fnvMixULong(hash, proof.stateProofProgramFingerprint);
    fnvMixULong(hash, proof.adapterProgramFingerprint);
    fnvMixULong(hash, proof.transactionPolicyFingerprint);
    fnvMixULong(hash, proof.linearizationEpoch);
    fnvMixULong(hash, proof.slotGeneration);
    fnvMixULong(hash, proof.motorCandidateFingerprint);
    fnvMixULong(hash, proof.inboundAuthorityFingerprint);
    return hash;
}

inline bool validExactAcceptedStateProof(
    thread const MRNumanXAcceptedStateProofGPUV2& proof,
    thread const MRNumanXExactInboundAuthorityGPUV2& authority,
    constant MRNumanXHumanMatterAdapterDispatchGPUV2& dispatch
) {
    return validExactDispatch(dispatch) &&
        validExactInboundAuthority(authority, dispatch) &&
        proof.abiVersion ==
            MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION &&
        proof.structSize == sizeof(proof) &&
        proof.status == MR_NUMANX_ACCEPTED_STATE_PROOF_VALID &&
        proof.environment == dispatch.environmentIdentifierBase &&
        proof.transactionFingerprint == dispatch.transactionFingerprint &&
        proof.substepFingerprint == dispatch.substepFingerprint &&
        proof.acceptedTimestampNanoseconds ==
            dispatch.acceptedTimestampNanoseconds &&
        proof.physicsGeneration == dispatch.physicsGeneration &&
        proof.clockDomain == dispatch.clockDomain &&
        proof.clockQuantumNanoseconds ==
            dispatch.clockQuantumNanoseconds &&
        proof.humanStateFingerprint != 0ul &&
        proof.matterStateFingerprint != 0ul &&
        proof.physicsStateFingerprint ==
            exactPhysicsStateFingerprint(proof) &&
        proof.matterSourcePhysicsFingerprint ==
            dispatch.matterSourcePhysicsFingerprint &&
        proof.matterDeviceProgramFingerprint ==
            dispatch.matterDeviceProgramFingerprint &&
        proof.stateProofProgramFingerprint ==
            dispatch.stateProofProgramFingerprint &&
        proof.adapterProgramFingerprint == dispatch.programFingerprint &&
        proof.transactionPolicyFingerprint != 0ul &&
        proof.linearizationEpoch == dispatch.linearizationEpoch &&
        proof.slotGeneration == dispatch.slotGeneration &&
        proof.motorCandidateFingerprint ==
            dispatch.motorCandidateFingerprint &&
        proof.inboundAuthorityFingerprint ==
            authority.inboundAuthorityFingerprint &&
        proof.proofFingerprint != 0ul &&
        proof.proofFingerprint == exactProofFingerprint(proof);
}

inline ulong exactAcceptedTokenFingerprint(
    thread const MRNumanXAcceptedPhysicsStateTokenGPUV2& token
) {
    ulong hash = kFNVOffset;
    fnvMixUInt(
        hash, MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_PHYSICS_TOKEN_V2);
    fnvMixUInt(hash, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_VERSION_V2);
    fnvMixULong(hash, token.transactionFingerprint);
    fnvMixULong(hash, token.substepFingerprint);
    fnvMixULong(hash, token.physicsStateFingerprint);
    fnvMixULong(hash, token.acceptedTimestampNanoseconds);
    fnvMixULong(hash, token.physicsGeneration);
    fnvMixUInt(hash, token.environmentIdentifier);
    fnvMixUInt(hash, token.flags);
    fnvMixUInt(hash, token.clockDomain);
    fnvMixUInt(hash, token.clockQuantumNanoseconds);
    return hash;
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

kernel void numanx_human_matter_write_prepared_token_v2(
    constant MRNumanXHumanMatterAdapterDispatchGPUV2& dispatch [[buffer(0)]],
    device MRNumanXCoupledHumanStatusGPU* jointStatuses [[buffer(1)]],
    device const MRNumanXAcceptedStateProofGPUV2* proofs [[buffer(2)]],
    device uchar* acceptedTokens [[buffer(3)]],
    device const MRNumanXExactInboundAuthorityGPUV2* inboundAuthorities
        [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    // The exact path always dispatches one thread. Clear the only token before
    // inspecting environmentCount or any mutable input, so malformed dispatch
    // bytes cannot preserve a token from a reused slot.
    if (environment != 0u) return;
    device MRNumanXAcceptedPhysicsStateTokenGPUV2* token =
        reinterpret_cast<device MRNumanXAcceptedPhysicsStateTokenGPUV2*>(
            acceptedTokens);
    *token = {};
    if (dispatch.jointStatusStride == 0u) return;

    device MRNumanXCoupledHumanStatusGPU& joint = jointStatuses[0];
    const MRNumanXAcceptedStateProofGPUV2 proof = proofs[0];
    const MRNumanXExactInboundAuthorityGPUV2 authority =
        inboundAuthorities[0];
    const bool authorityAvailable =
        authority.abiVersion != 0u || authority.inboundAuthorityFingerprint != 0ul;
    const bool authorityValid =
        validExactInboundAuthority(authority, dispatch);
    const bool proofAvailable =
        proof.status != MR_NUMANX_ACCEPTED_STATE_PROOF_PENDING;
    const bool valid = authorityValid &&
        validExactAcceptedStateProof(proof, authority, dispatch);
    if (!valid) {
        if (joint.decision == MR_NUMANX_COUPLED_HUMAN_ACCEPT ||
            (joint.decision == MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER &&
             joint.matterCode == dispatch.matterSuccessCode)) {
            joint.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
            joint.matterCode = !authorityValid
                ? (authorityAvailable
                    ? MR_NUMANX_HUMAN_MATTER_ADAPTER_INVALID_INBOUND_AUTHORITY
                    : MR_NUMANX_HUMAN_MATTER_ADAPTER_MISSING_INBOUND_AUTHORITY)
                : (proofAvailable
                    ? MR_NUMANX_HUMAN_MATTER_ADAPTER_INVALID_STATE_PROOF
                    : MR_NUMANX_HUMAN_MATTER_ADAPTER_MISSING_STATE_PROOF);
        }
        return;
    }
    if (joint.decision != MR_NUMANX_COUPLED_HUMAN_ACCEPT) return;

    MRNumanXAcceptedPhysicsStateTokenGPUV2 accepted = {};
    accepted.transactionFingerprint = dispatch.transactionFingerprint;
    accepted.substepFingerprint = dispatch.substepFingerprint;
    accepted.physicsStateFingerprint = proof.physicsStateFingerprint;
    accepted.acceptedTimestampNanoseconds =
        dispatch.acceptedTimestampNanoseconds;
    accepted.physicsGeneration = dispatch.physicsGeneration;
    accepted.environmentIdentifier = proof.environment;
    accepted.clockDomain = dispatch.clockDomain;
    accepted.clockQuantumNanoseconds =
        dispatch.clockQuantumNanoseconds;
    accepted.tokenFingerprint = exactAcceptedTokenFingerprint(accepted);
    if (accepted.tokenFingerprint == 0ul) {
        joint.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
        joint.matterCode =
            MR_NUMANX_HUMAN_MATTER_ADAPTER_INVALID_STATE_PROOF;
        return;
    }
    *token = accepted;
}

// Pose-only scratch for Human support's initial-gap recovery. The owner has
// completed FK before this dispatch. Every record is written, including any
// prefix outside the articulation; no candidate or accepted arena is aliased.
kernel void numanx_human_matter_materialize_support_bodies(
    constant uint4& layout [[buffer(0)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(1)]],
    device MRBodyStateGPU* bodies [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    const uint environment = index / layout.w;
    const uint body = index % layout.w;
    if (environment >= layout.x) return;
    MRBodyStateGPU state = {};
    if (body >= layout.z) {
        const MRArticulatedBodyPoseGPU pose =
            poses[environment * layout.y + body - layout.z];
        state.position = pose.position;
        state.orientation = pose.orientation;
    }
    bodies[index] = state;
}
