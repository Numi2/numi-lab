#include <metal_stdlib>

#include "metalrobo/numanx_human_matter_gpu.h"
#include "metalrobo/mujoco_muscle_gpu.h"

using namespace metal;

namespace {

constant float kPivotFloor = 1.0e-10f;

inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline float invalidFloat() {
    return as_type<float>(0x7fc00000u);
}

inline ulong fnv1aByte(ulong hash, const uint byteValue) {
    return (hash ^ static_cast<ulong>(byteValue & 0xffu)) *
        1099511628211ul;
}

inline ulong fnv1aU32(ulong hash, const uint value) {
    for (uint shift = 0u; shift < 32u; shift += 8u)
        hash = fnv1aByte(hash, value >> shift);
    return hash;
}

inline ulong fnv1aU64(ulong hash, const ulong value) {
    for (uint shift = 0u; shift < 64u; shift += 8u)
        hash = fnv1aByte(hash, static_cast<uint>(value >> shift));
    return hash;
}

inline ulong nonzeroFingerprint(const ulong hash) {
    return hash == 0ul ? 14695981039346656037ul : hash;
}

// Exact NBAcceptedPhysicsStateToken relation. The record itself has no ABI
// word, so version 1 is mixed as the first u32 and the terminal word at byte
// 56 is excluded from the fold. This is timeline integrity/replay identity,
// not a cryptographic authenticator.
inline ulong acceptedTokenFingerprint(device const uchar* token) {
    device const ulong* words =
        reinterpret_cast<device const ulong*>(token);
    device const uint* words32 =
        reinterpret_cast<device const uint*>(token);
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, 1u);
    hash = fnv1aU64(hash, words[0]);
    hash = fnv1aU64(hash, words[1]);
    hash = fnv1aU64(hash, words[2]);
    hash = fnv1aU64(hash, words[3]);
    hash = fnv1aU64(hash, words[4]);
    hash = fnv1aU32(hash, words32[10]);
    hash = fnv1aU32(hash, words32[11]);
    hash = fnv1aU64(hash, words[6]);
    // The NBAcceptedPhysicsStateToken relation is the raw FNV result. A zero
    // hash is invalid at the call site; unlike versioned 128-byte records it
    // must not be remapped to the offset basis.
    return hash;
}

inline bool validAcceptedToken(
    device const uchar* token,
    const ulong expectedTransactionFingerprint,
    const ulong expectedTokenFingerprint
) {
    device const ulong* words =
        reinterpret_cast<device const ulong*>(token);
    return expectedTransactionFingerprint != 0ul &&
        expectedTokenFingerprint != 0ul &&
        words[0] == expectedTransactionFingerprint && words[6] == 0ul &&
        words[7] == expectedTokenFingerprint &&
        acceptedTokenFingerprint(token) == expectedTokenFingerprint;
}

inline bool zeroAcceptedToken(device const uchar* token) {
    bool zero = true;
    for (uint byte = 0u;
         byte < MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
         ++byte) {
        zero = zero && token[byte] == 0u;
    }
    return zero;
}

inline bool readyProposalRejectCode(const uint code) {
    return code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT;
}

inline uint appliedCodeForProposalReject(const uint code) {
    switch (code) {
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT;
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS:
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_BRAIN_REJECT;
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_TOKEN_MISMATCH;
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT;
    default:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER;
    }
}

inline ulong brainWitnessFingerprint(
    const MRNumanXHumanMatterBrainCommitWitnessGPU witness
) {
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, witness.magic);
    hash = fnv1aU32(hash, witness.abiVersion);
    hash = fnv1aU32(hash, witness.structBytes);
    hash = fnv1aU32(hash, witness.status);
    hash = fnv1aU32(hash, witness.decision);
    hash = fnv1aU32(hash, witness.environment);
    hash = fnv1aU32(hash, witness.stepIndex);
    hash = fnv1aU32(hash, witness.substepIndex);
    hash = fnv1aU32(hash, witness.transactionSlot);
    hash = fnv1aU32(hash, witness.physicsSubstepCount);
    hash = fnv1aU32(hash, witness.controlStep);
    hash = fnv1aU32(hash, witness.reserved0);
    hash = fnv1aU64(hash, witness.programFingerprint);
    hash = fnv1aU64(hash, witness.transactionFingerprint);
    hash = fnv1aU64(hash, witness.linearizationEpoch);
    hash = fnv1aU64(hash, witness.slotGeneration);
    hash = fnv1aU64(hash, witness.physicsTokenFingerprint);
    hash = fnv1aU64(hash, witness.brainProgramFingerprint);
    hash = fnv1aU64(hash, witness.brainShadowStateFingerprint);
    for (uint index = 0u; index < 2u; ++index)
        hash = fnv1aU64(hash, witness.reserved1[index]);
    return nonzeroFingerprint(hash);
}

inline ulong proposalFingerprint(
    const MRNumanXHumanMatterProposalGPU value
) {
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, value.abiVersion);
    hash = fnv1aU32(hash, value.status);
    hash = fnv1aU32(hash, value.decision);
    hash = fnv1aU32(hash, value.code);
    hash = fnv1aU64(hash, value.programFingerprint);
    hash = fnv1aU64(hash, value.transactionFingerprint);
    hash = fnv1aU64(hash, value.linearizationEpoch);
    hash = fnv1aU64(hash, value.slotGeneration);
    hash = fnv1aU64(hash, value.physicsTokenFingerprint);
    hash = fnv1aU64(hash, value.brainProgramFingerprint);
    hash = fnv1aU64(hash, value.brainShadowStateFingerprint);
    hash = fnv1aU64(hash, value.brainWitnessFingerprint);
    hash = fnv1aU64(hash, value.candidatePublicationFingerprint);
    hash = fnv1aU64(hash, value.humanIOIdentityFingerprint);
    hash = fnv1aU32(hash, value.environment);
    hash = fnv1aU32(hash, value.stepIndex);
    hash = fnv1aU32(hash, value.substepIndex);
    hash = fnv1aU32(hash, value.transactionSlot);
    hash = fnv1aU32(hash, value.physicsSubstepCount);
    hash = fnv1aU32(hash, value.controlStep);
    return nonzeroFingerprint(hash);
}

inline ulong brainAckFingerprint(
    const MRNumanXHumanMatterBrainAckGPU value
) {
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, value.abiVersion);
    hash = fnv1aU32(hash, value.status);
    hash = fnv1aU32(hash, value.decision);
    hash = fnv1aU32(hash, value.code);
    hash = fnv1aU64(hash, value.programFingerprint);
    hash = fnv1aU64(hash, value.transactionFingerprint);
    hash = fnv1aU64(hash, value.linearizationEpoch);
    hash = fnv1aU64(hash, value.slotGeneration);
    hash = fnv1aU64(hash, value.physicsTokenFingerprint);
    hash = fnv1aU64(hash, value.proposalFingerprint);
    hash = fnv1aU64(hash, value.preflightFingerprint);
    hash = fnv1aU64(hash, value.fastGateFingerprint);
    hash = fnv1aU64(hash, value.brainWitnessFingerprint);
    hash = fnv1aU64(hash, value.brainProgramFingerprint);
    hash = fnv1aU32(hash, value.environment);
    hash = fnv1aU32(hash, value.stepIndex);
    hash = fnv1aU32(hash, value.substepIndex);
    hash = fnv1aU32(hash, value.transactionSlot);
    hash = fnv1aU32(hash, value.physicsSubstepCount);
    hash = fnv1aU32(hash, value.controlStep);
    return nonzeroFingerprint(hash);
}

inline ulong applyActionFingerprint(
    const MRNumanXHumanMatterApplyActionGPU value
) {
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, value.abiVersion);
    hash = fnv1aU32(hash, value.status);
    hash = fnv1aU32(hash, value.decision);
    hash = fnv1aU32(hash, value.code);
    hash = fnv1aU64(hash, value.programFingerprint);
    hash = fnv1aU64(hash, value.transactionFingerprint);
    hash = fnv1aU64(hash, value.linearizationEpoch);
    hash = fnv1aU64(hash, value.slotGeneration);
    hash = fnv1aU64(hash, value.physicsTokenFingerprint);
    hash = fnv1aU64(hash, value.proposalFingerprint);
    hash = fnv1aU64(hash, value.ackFingerprint);
    hash = fnv1aU64(hash, value.preflightFingerprint);
    hash = fnv1aU64(hash, value.fastGateFingerprint);
    hash = fnv1aU64(hash, value.brainWitnessFingerprint);
    hash = fnv1aU32(hash, value.environment);
    hash = fnv1aU32(hash, value.stepIndex);
    hash = fnv1aU32(hash, value.substepIndex);
    hash = fnv1aU32(hash, value.transactionSlot);
    hash = fnv1aU32(hash, value.physicsSubstepCount);
    hash = fnv1aU32(hash, value.controlStep);
    return nonzeroFingerprint(hash);
}

inline ulong matterApplyOutcomeFingerprint(
    const MRNumanXHumanMatterMatterApplyOutcomeGPU value
) {
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, value.abiVersion);
    hash = fnv1aU32(hash, value.status);
    hash = fnv1aU32(hash, value.decision);
    hash = fnv1aU32(hash, value.code);
    hash = fnv1aU64(hash, value.programFingerprint);
    hash = fnv1aU64(hash, value.transactionFingerprint);
    hash = fnv1aU64(hash, value.linearizationEpoch);
    hash = fnv1aU64(hash, value.slotGeneration);
    hash = fnv1aU64(hash, value.physicsTokenFingerprint);
    hash = fnv1aU64(hash, value.proposalFingerprint);
    hash = fnv1aU64(hash, value.ackFingerprint);
    hash = fnv1aU64(hash, value.actionFingerprint);
    hash = fnv1aU64(hash, value.matterProgramFingerprint);
    hash = fnv1aU64(hash, value.reserved0);
    hash = fnv1aU32(hash, value.environment);
    hash = fnv1aU32(hash, value.stepIndex);
    hash = fnv1aU32(hash, value.substepIndex);
    hash = fnv1aU32(hash, value.transactionSlot);
    hash = fnv1aU32(hash, value.physicsSubstepCount);
    hash = fnv1aU32(hash, value.controlStep);
    return nonzeroFingerprint(hash);
}

inline ulong appliedOutcomeFingerprint(
    const MRNumanXHumanMatterAppliedOutcomeGPU value
) {
    ulong hash = 14695981039346656037ul;
    hash = fnv1aU32(hash, value.abiVersion);
    hash = fnv1aU32(hash, value.status);
    hash = fnv1aU32(hash, value.decision);
    hash = fnv1aU32(hash, value.code);
    hash = fnv1aU64(hash, value.programFingerprint);
    hash = fnv1aU64(hash, value.transactionFingerprint);
    hash = fnv1aU64(hash, value.linearizationEpoch);
    hash = fnv1aU64(hash, value.slotGeneration);
    hash = fnv1aU64(hash, value.physicsTokenFingerprint);
    hash = fnv1aU64(hash, value.proposalFingerprint);
    hash = fnv1aU64(hash, value.ackFingerprint);
    hash = fnv1aU64(hash, value.preflightFingerprint);
    hash = fnv1aU64(hash, value.fastGateFingerprint);
    hash = fnv1aU64(hash, value.matterApplyFingerprint);
    hash = fnv1aU32(hash, value.environment);
    hash = fnv1aU32(hash, value.stepIndex);
    hash = fnv1aU32(hash, value.substepIndex);
    hash = fnv1aU32(hash, value.transactionSlot);
    hash = fnv1aU32(hash, value.physicsSubstepCount);
    hash = fnv1aU32(hash, value.controlStep);
    return nonzeroFingerprint(hash);
}

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float4 quaternionMultiply(const float4 left, const float4 right) {
    return float4(
        left.w * right.x + left.x * right.w +
            left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z +
            left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y -
            left.y * right.x + left.z * right.w,
        left.w * right.w - dot(left.xyz, right.xyz)
    );
}

inline bool normalizedQuaternion(
    const float4 input,
    thread float4& output
) {
    const float normSquared = dot(input, input);
    if (!finite4(input) || !(normSquared > 1.0e-12f) ||
        !isfinite(normSquared)) return false;
    output = input * rsqrt(normSquared);
    return finite4(output);
}

inline float4 quaternionFromRotationVector(const float3 rotationVector) {
    const float angleSquared = dot(rotationVector, rotationVector);
    if (angleSquared < 1.0e-12f) {
        return normalize(float4(0.5f * rotationVector, 1.0f));
    }
    const float angle = sqrt(angleSquared);
    return normalize(float4(
        rotationVector * (sin(0.5f * angle) / angle),
        cos(0.5f * angle)
    ));
}

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross +
        cross(quaternion.xyz, doubledCross);
}

inline float3 worldInertiaMultiply(
    device const MRBodyPropertiesGPU& body,
    const float4 orientation,
    const float3 worldVector
) {
    const float3 local = quaternionRotate(
        quaternionConjugate(orientation), worldVector
    );
    const float3 localResult{
        dot(body.inertiaRow0.xyz, local),
        dot(body.inertiaRow1.xyz, local),
        dot(body.inertiaRow2.xyz, local),
    };
    return quaternionRotate(orientation, localResult);
}

inline bool validOwnerDispatch(
    constant MRNumanXHumanMatterDispatchGPU& dispatch
) {
    return dispatch.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        dispatch.environmentCount == 1u && dispatch.stepIndex == 0u &&
        dispatch.flags == MR_NUMANX_HUMAN_MATTER_HAS_PREPARED_TOKEN &&
        dispatch.transactionSlot <
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS &&
        dispatch.nv != 0u &&
        dispatch.nv <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        dispatch.nq == dispatch.nv + 1u &&
        dispatch.nq <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        dispatch.qStride == dispatch.nq &&
        dispatch.vStride == dispatch.nv &&
        dispatch.mujocoStateStride == dispatch.mujocoStateCount &&
        dispatch.generalizedForceStride == dispatch.nv &&
        dispatch.reactionStride >= dispatch.nv &&
        dispatch.reactionStride <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        dispatch.jointStatusStride != 0u &&
        dispatch.acceptedTokenStrideBytes ==
            MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_BYTES &&
        dispatch.ownerStatusStride != 0u &&
        dispatch.substepIndex == 0u &&
        dispatch.physicsSubstepCount ==
            MR_NUMANX_HUMAN_MATTER_PHYSICS_SUBSTEP_COUNT &&
        dispatch.timestepAndInverse.x > 0.0f &&
        dispatch.timestepAndInverse.y > 0.0f &&
        isfinite(dispatch.timestepAndInverse.x) &&
        isfinite(dispatch.timestepAndInverse.y) &&
        dispatch.timestepAndInverse.z == 0.0f &&
        dispatch.timestepAndInverse.w == 0.0f &&
        dispatch.programFingerprint != 0u &&
        dispatch.transactionFingerprint != 0u &&
        dispatch.linearizationEpoch != 0u &&
        dispatch.slotGeneration != 0u;
}

inline bool validProposalDispatch(
    constant MRNumanXHumanMatterProposalDispatchGPU& dispatch
) {
    const uint knownFlags =
        MR_NUMANX_HUMAN_MATTER_PROPOSAL_VALIDATE_BRAIN_WITNESS |
        MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCE_REJECT;
    const bool oneMode = dispatch.flags ==
            MR_NUMANX_HUMAN_MATTER_PROPOSAL_VALIDATE_BRAIN_WITNESS ||
        dispatch.flags == MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCE_REJECT;
    return dispatch.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        (dispatch.flags & ~knownFlags) == 0u && oneMode &&
        dispatch.environmentCount == 1u && dispatch.stepIndex == 0u &&
        dispatch.substepIndex == 0u &&
        dispatch.transactionSlot <
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS &&
        dispatch.ownerStatusStride != 0u &&
        (dispatch.flags ==
             MR_NUMANX_HUMAN_MATTER_PROPOSAL_VALIDATE_BRAIN_WITNESS
             ? dispatch.brainWitnessStride != 0u
             : dispatch.brainWitnessStride == 0u) &&
        dispatch.preparedTokenStrideBytes ==
            MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_BYTES &&
        dispatch.proposalStride != 0u &&
        dispatch.proposedTokenStrideBytes ==
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES &&
        dispatch.physicsSubstepCount ==
            MR_NUMANX_HUMAN_MATTER_PHYSICS_SUBSTEP_COUNT &&
        dispatch.reserved0 == 0u && dispatch.reserved1 == 0u &&
        dispatch.reserved2 == 0u && dispatch.programFingerprint != 0ul &&
        dispatch.transactionFingerprint != 0ul &&
        dispatch.linearizationEpoch != 0ul && dispatch.slotGeneration != 0ul &&
        dispatch.candidatePublicationFingerprint != 0ul &&
        dispatch.humanIOIdentityFingerprint != 0ul;
}

inline bool validApplyDispatch(
    constant MRNumanXHumanMatterApplyDispatchGPU& dispatch
) {
    const uint knownFlags = MR_NUMANX_HUMAN_MATTER_APPLY_VALIDATE_BRAIN_ACK |
        MR_NUMANX_HUMAN_MATTER_APPLY_FORCE_REJECT;
    const bool oneMode = dispatch.flags ==
            MR_NUMANX_HUMAN_MATTER_APPLY_VALIDATE_BRAIN_ACK ||
        dispatch.flags == MR_NUMANX_HUMAN_MATTER_APPLY_FORCE_REJECT;
    return dispatch.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        (dispatch.flags & ~knownFlags) == 0u && oneMode &&
        dispatch.environmentCount == 1u && dispatch.stepIndex == 0u &&
        dispatch.substepIndex == 0u &&
        dispatch.transactionSlot <
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS &&
        dispatch.nv != 0u &&
        dispatch.nv <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        dispatch.nq == dispatch.nv + 1u &&
        dispatch.nq <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        dispatch.qStride == dispatch.nq && dispatch.vStride == dispatch.nv &&
        dispatch.mujocoStateStride == dispatch.mujocoStateCount &&
        dispatch.ownerStatusStride != 0u &&
        (dispatch.flags == MR_NUMANX_HUMAN_MATTER_APPLY_VALIDATE_BRAIN_ACK
             ? dispatch.brainAckStride != 0u
             : dispatch.brainAckStride == 0u) &&
        dispatch.proposedTokenStrideBytes ==
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES &&
        dispatch.applyActionStride != 0u &&
        dispatch.matterOutcomeStride != 0u &&
        dispatch.appliedOutcomeStride != 0u &&
        dispatch.finalTokenStrideBytes ==
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES &&
        dispatch.physicsSubstepCount ==
            MR_NUMANX_HUMAN_MATTER_PHYSICS_SUBSTEP_COUNT &&
        dispatch.reserved0 == 0u && dispatch.reserved1 == 0u &&
        dispatch.reserved2 == 0u && dispatch.programFingerprint != 0ul &&
        dispatch.transactionFingerprint != 0ul &&
        dispatch.linearizationEpoch != 0ul && dispatch.slotGeneration != 0ul;
}

inline bool validCandidateDispatch(
    constant MRNumanXHumanMatterCandidateDispatchGPU& dispatch
) {
    const ulong bodyProbeCount =
        static_cast<ulong>(dispatch.bodyCount) * 4ul;
    const ulong combinedCount = bodyProbeCount +
        static_cast<ulong>(dispatch.candidatePointCount);
    const ulong candidateJacobianCount =
        static_cast<ulong>(dispatch.candidatePointCount) * 3ul *
        static_cast<ulong>(dispatch.deltaVelocityStride);
    const ulong privateJacobianCount = combinedCount * 3ul *
        static_cast<ulong>(dispatch.nv);
    const bool hasPointWorld =
        (dispatch.flags &
         MR_NUMANX_HUMAN_MATTER_CANDIDATE_HAS_POINT_WORLD) != 0u;
    return dispatch.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        (dispatch.flags &
         ~MR_NUMANX_HUMAN_MATTER_CANDIDATE_HAS_POINT_WORLD) == 0u &&
        dispatch.environmentCount == 1u &&
        dispatch.nv != 0u &&
        dispatch.nv <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        dispatch.nq == dispatch.nv + 1u &&
        dispatch.nq <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        dispatch.bodyCount != 0u &&
        dispatch.bodyCount <=
            MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_BODIES &&
        dispatch.bodyProbeCount == bodyProbeCount &&
        combinedCount <= MR_ARTICULATED_OPERATOR_MAX_POINTS &&
        dispatch.sourcePointStride >=
            dispatch.sourceBodyProbeOffset + dispatch.bodyProbeCount &&
        dispatch.combinedPointStride >= combinedCount &&
        dispatch.deltaVelocityStride >= dispatch.nv &&
        dispatch.deltaVelocityStride <=
            MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        dispatch.candidateQStride >= dispatch.nq &&
        dispatch.candidateQStride <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        dispatch.candidateBodyStride >=
            dispatch.articulationFirstBody + dispatch.bodyCount &&
        (dispatch.candidatePointCount == 0u ||
         (dispatch.candidatePointStride >= dispatch.candidatePointCount &&
          dispatch.candidatePointJacobianStride >=
              candidateJacobianCount)) &&
        (hasPointWorld
             ? dispatch.candidatePointWorldStride >=
                   dispatch.candidatePointCount
             : dispatch.candidatePointWorldStride == 0u) &&
        dispatch.privateBodyPoseStride >= dispatch.bodyCount &&
        dispatch.privatePointWorldStride >= combinedCount &&
        dispatch.privatePointJacobianStride >= privateJacobianCount &&
        dispatch.privateOperatorScratchStride >= dispatch.nv &&
        dispatch.transactionSlot <
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS &&
        dispatch.substepIndex == 0u &&
        dispatch.physicsSubstepCount ==
            MR_NUMANX_HUMAN_MATTER_PHYSICS_SUBSTEP_COUNT &&
        dispatch.reserved3 == 0u &&
        dispatch.timestepAndInverse.x > 0.0f &&
        dispatch.timestepAndInverse.y > 0.0f &&
        isfinite(dispatch.timestepAndInverse.x) &&
        isfinite(dispatch.timestepAndInverse.y) &&
        dispatch.timestepAndInverse.z == 0.0f &&
        dispatch.timestepAndInverse.w == 0.0f &&
        dispatch.programFingerprint != 0u &&
        dispatch.transactionFingerprint != 0u &&
        dispatch.linearizationEpoch != 0u &&
        dispatch.slotGeneration != 0u;
}

inline bool matchingOwner(
    device const MRNumanXHumanMatterOwnerStatusGPU& status,
    constant MRNumanXHumanMatterDispatchGPU& dispatch,
    const uint environment
) {
    return status.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        status.environment == environment &&
        status.stepIndex == dispatch.stepIndex &&
        status.substepIndex == dispatch.substepIndex &&
        status.transactionSlot == dispatch.transactionSlot &&
        status.physicsSubstepCount == dispatch.physicsSubstepCount &&
        status.controlStep == dispatch.controlStep &&
        status.qCoordinateCount == dispatch.nq &&
        status.dofCount == dispatch.nv &&
        status.programFingerprint == dispatch.programFingerprint &&
        status.transactionFingerprint == dispatch.transactionFingerprint &&
        status.linearizationEpoch == dispatch.linearizationEpoch &&
        status.slotGeneration == dispatch.slotGeneration;
}

inline void failOwner(
    device MRNumanXHumanMatterOwnerStatusGPU& status,
    const uint code
) {
    if (status.code == MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS) {
        status.code = code;
    }
    status.stage = MR_NUMANX_HUMAN_MATTER_STAGE_FAILED;
}

} // namespace

kernel void mr_numanx_human_matter_begin(
    constant MRNumanXHumanMatterDispatchGPU& dispatch [[buffer(0)]],
    device uchar* preparedTokens [[buffer(1)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(2)]],
    device MRNumanXHumanMatterAppliedOutcomeGPU* appliedOutcomes [[buffer(3)]],
    device uchar* finalTokens [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const uint ownerIndex = environment * dispatch.ownerStatusStride;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[ownerIndex];
    owner = {};
    owner.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    owner.stage = validOwnerDispatch(dispatch)
        ? MR_NUMANX_HUMAN_MATTER_STAGE_BEGUN
        : MR_NUMANX_HUMAN_MATTER_STAGE_FAILED;
    owner.environment = environment;
    owner.stepIndex = dispatch.stepIndex;
    owner.substepIndex = dispatch.substepIndex;
    owner.transactionSlot = dispatch.transactionSlot;
    owner.physicsSubstepCount = dispatch.physicsSubstepCount;
    owner.qCoordinateCount = dispatch.nq;
    owner.controlStep = dispatch.controlStep;
    owner.dofCount = dispatch.nv;
    owner.physicalCommandStatus =
        MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_UNKNOWN;
    owner.code = validOwnerDispatch(dispatch)
        ? MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS
        : MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_DISPATCH;
    owner.programFingerprint = dispatch.programFingerprint;
    owner.transactionFingerprint = dispatch.transactionFingerprint;
    owner.linearizationEpoch = dispatch.linearizationEpoch;
    owner.slotGeneration = dispatch.slotGeneration;
    const uint tokenBase =
        environment * dispatch.acceptedTokenStrideBytes;
    for (uint byte = 0u;
         byte < MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_BYTES; ++byte) {
        preparedTokens[tokenBase + byte] = 0u;
        finalTokens[tokenBase + byte] = 0u;
    }
    appliedOutcomes[environment] = {};
}

// Integrates the source checkpoint with v0+delta-v and builds the private
// combined query stream. This is the same exponential-map floating-root and
// scalar-coordinate integration used by Stand; no finite-difference or
// first-order pose path exists here.
kernel void mr_numanx_human_matter_prepare_candidate(
    constant MRNumanXHumanMatterCandidateDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const float* sourceQ [[buffer(3)]],
    device const float* sourceV [[buffer(4)]],
    device const float* deltaVelocity [[buffer(5)]],
    device float* candidateQ [[buffer(6)]],
    device const MRArticulatedPointImpulseGPU* sourcePoints [[buffer(7)]],
    device const MRArticulatedPointImpulseGPU* candidatePoints [[buffer(8)]],
    device MRArticulatedPointImpulseGPU* combinedPoints [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const uint candidateQBase =
        environment * dispatch.candidateQStride;
    const uint sourceQBase = environment * dispatch.nq;
    const uint sourceVBase = environment * dispatch.nv;
    const uint deltaBase =
        environment * dispatch.deltaVelocityStride;
    bool valid = validCandidateDispatch(dispatch);
    const MRArticulationGPU articulation =
        articulations[dispatch.articulationIndex];
    valid = valid && articulation.rootType == MR_ROOT_FLOATING &&
        articulation.nq == dispatch.nq &&
        articulation.nv == dispatch.nv &&
        articulation.bodyCount == dispatch.bodyCount &&
        articulation.firstBody == dispatch.articulationFirstBody;

    for (uint coordinate = 0u; coordinate < dispatch.nq; ++coordinate) {
        candidateQ[candidateQBase + coordinate] =
            sourceQ[sourceQBase + coordinate];
        valid = valid && isfinite(sourceQ[sourceQBase + coordinate]);
    }
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        valid = valid && isfinite(sourceV[sourceVBase + dof]) &&
            isfinite(deltaVelocity[deltaBase + dof]) &&
            isfinite(sourceV[sourceVBase + dof] +
                     deltaVelocity[deltaBase + dof]);
    }
    float4 sourceOrientation;
    valid = valid && normalizedQuaternion(
        float4(
            sourceQ[sourceQBase + 3u], sourceQ[sourceQBase + 4u],
            sourceQ[sourceQBase + 5u], sourceQ[sourceQBase + 6u]
        ),
        sourceOrientation
    );
    for (uint dof = 6u; dof < dispatch.nv; ++dof) {
        const MRDofPropertiesGPU properties =
            dofs[articulation.vOffset + dof];
        valid = valid &&
            properties.articulationIndex == dispatch.articulationIndex &&
            properties.vIndex == articulation.vOffset + dof &&
            properties.qIndex != MR_INVALID_INDEX &&
            properties.qIndex >= articulation.qOffset &&
            properties.qIndex < articulation.qOffset + dispatch.nq;
    }
    if (!valid) {
        for (uint coordinate = 0u; coordinate < dispatch.nq; ++coordinate)
            candidateQ[candidateQBase + coordinate] = invalidFloat();
        return;
    }

    const float timestep = dispatch.timestepAndInverse.x;
    const float3 linear{
        sourceV[sourceVBase + 0u] + deltaVelocity[deltaBase + 0u],
        sourceV[sourceVBase + 1u] + deltaVelocity[deltaBase + 1u],
        sourceV[sourceVBase + 2u] + deltaVelocity[deltaBase + 2u],
    };
    const float3 angular{
        sourceV[sourceVBase + 3u] + deltaVelocity[deltaBase + 3u],
        sourceV[sourceVBase + 4u] + deltaVelocity[deltaBase + 4u],
        sourceV[sourceVBase + 5u] + deltaVelocity[deltaBase + 5u],
    };
    candidateQ[candidateQBase + 0u] += timestep * linear.x;
    candidateQ[candidateQBase + 1u] += timestep * linear.y;
    candidateQ[candidateQBase + 2u] += timestep * linear.z;
    const float4 increment = quaternionFromRotationVector(
        timestep * angular
    );
    const float4 orientation = normalize(
        quaternionMultiply(increment, sourceOrientation)
    );
    candidateQ[candidateQBase + 3u] = orientation.x;
    candidateQ[candidateQBase + 4u] = orientation.y;
    candidateQ[candidateQBase + 5u] = orientation.z;
    candidateQ[candidateQBase + 6u] = orientation.w;
    for (uint dof = 6u; dof < dispatch.nv; ++dof) {
        const MRDofPropertiesGPU properties =
            dofs[articulation.vOffset + dof];
        const uint localQ = properties.qIndex - articulation.qOffset;
        candidateQ[candidateQBase + localQ] += timestep *
            (sourceV[sourceVBase + dof] +
             deltaVelocity[deltaBase + dof]);
    }

    const uint combinedBase =
        environment * dispatch.combinedPointStride;
    const uint sourcePointBase =
        environment * dispatch.sourcePointStride +
        dispatch.sourceBodyProbeOffset;
    for (uint point = 0u; point < dispatch.bodyProbeCount; ++point)
        combinedPoints[combinedBase + point] =
            sourcePoints[sourcePointBase + point];
    const uint candidatePointBase =
        environment * dispatch.candidatePointStride;
    for (uint point = 0u;
         point < dispatch.candidatePointCount; ++point)
        combinedPoints[combinedBase + dispatch.bodyProbeCount + point] =
            candidatePoints[candidatePointBase + point];
}

// Materializes collision-compatible Human body state plus the attachment
// suffix after the generic analytic kinematics/Jacobian kernel succeeds.
// Body velocities are exact J(q_candidate)*(v0+delta-v), with angular rows
// reconstructed analytically from the canonical COM/+axis probes.
kernel void mr_numanx_human_matter_materialize_candidate(
    constant MRNumanXHumanMatterCandidateDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(2)]],
    device const float* sourceV [[buffer(3)]],
    device const float* deltaVelocity [[buffer(4)]],
    device float* candidateQ [[buffer(5)]],
    device const MRArticulatedBodyPoseGPU* privateBodyPoses [[buffer(6)]],
    device const MRArticulatedPointWorldGPU* privatePointWorld [[buffer(7)]],
    device const float* privatePointJacobians [[buffer(8)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses [[buffer(9)]],
    device MRBodyStateGPU* candidateBodies [[buffer(10)]],
    device MRArticulatedPointWorldGPU* candidatePointWorld [[buffer(11)]],
    device float* candidatePointJacobians [[buffer(12)]],
    threadgroup uint& valid [[threadgroup(0)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    if (lane == 0u) {
        const MRArticulationGPU articulation =
            articulations[dispatch.articulationIndex];
        const MRArticulatedOperatorStatusGPU status =
            operatorStatuses[environment];
        valid = validCandidateDispatch(dispatch) &&
            articulation.rootType == MR_ROOT_FLOATING &&
            articulation.nq == dispatch.nq &&
            articulation.nv == dispatch.nv &&
            articulation.bodyCount == dispatch.bodyCount &&
            articulation.firstBody == dispatch.articulationFirstBody &&
            status.code == MR_ARTICULATED_OPERATOR_SUCCESS &&
            status.environment == environment &&
            status.articulationIndex == dispatch.articulationIndex &&
            status.bodyCount == dispatch.bodyCount &&
            status.nq == dispatch.nq && status.nv == dispatch.nv &&
            status.pointCount ==
                dispatch.bodyProbeCount + dispatch.candidatePointCount;
        if (valid == 0u) {
            candidateQ[environment * dispatch.candidateQStride] =
                invalidFloat();
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid == 0u) return;

    const uint bodyPoseBase =
        environment * dispatch.privateBodyPoseStride;
    const uint privateJacobianBase =
        environment * dispatch.privatePointJacobianStride;
    const uint bodyBase =
        environment * dispatch.candidateBodyStride;
    const uint sourceVBase = environment * dispatch.nv;
    const uint deltaBase =
        environment * dispatch.deltaVelocityStride;
    for (uint localBody = lane;
         localBody < dispatch.bodyCount; localBody += threadCount) {
        const uint probe = 4u * localBody;
        const uint probeBase =
            privateJacobianBase + probe * 3u * dispatch.nv;
        const float4 orientation =
            privateBodyPoses[bodyPoseBase + localBody].orientation;
        const float3 axisX = quaternionRotate(
            orientation, float3(1.0f, 0.0f, 0.0f));
        const float3 axisY = quaternionRotate(
            orientation, float3(0.0f, 1.0f, 0.0f));
        const float3 axisZ = quaternionRotate(
            orientation, float3(0.0f, 0.0f, 1.0f));
        float3 linear{0.0f};
        float3 angular{0.0f};
        for (uint dof = 0u; dof < dispatch.nv; ++dof) {
            const float velocity = sourceV[sourceVBase + dof] +
                deltaVelocity[deltaBase + dof];
            const float3 column{
                privatePointJacobians[probeBase + 0u * dispatch.nv + dof],
                privatePointJacobians[probeBase + 1u * dispatch.nv + dof],
                privatePointJacobians[probeBase + 2u * dispatch.nv + dof],
            };
            const float3 dx{
                privatePointJacobians[
                    probeBase + 3u * dispatch.nv + 0u * dispatch.nv + dof] -
                    column.x,
                privatePointJacobians[
                    probeBase + 3u * dispatch.nv + 1u * dispatch.nv + dof] -
                    column.y,
                privatePointJacobians[
                    probeBase + 3u * dispatch.nv + 2u * dispatch.nv + dof] -
                    column.z,
            };
            const float3 dy{
                privatePointJacobians[
                    probeBase + 6u * dispatch.nv + 0u * dispatch.nv + dof] -
                    column.x,
                privatePointJacobians[
                    probeBase + 6u * dispatch.nv + 1u * dispatch.nv + dof] -
                    column.y,
                privatePointJacobians[
                    probeBase + 6u * dispatch.nv + 2u * dispatch.nv + dof] -
                    column.z,
            };
            const float3 dz{
                privatePointJacobians[
                    probeBase + 9u * dispatch.nv + 0u * dispatch.nv + dof] -
                    column.x,
                privatePointJacobians[
                    probeBase + 9u * dispatch.nv + 1u * dispatch.nv + dof] -
                    column.y,
                privatePointJacobians[
                    probeBase + 9u * dispatch.nv + 2u * dispatch.nv + dof] -
                    column.z,
            };
            const float3 angularColumn = 0.5f * (
                cross(axisX, dx) + cross(axisY, dy) +
                cross(axisZ, dz)
            );
            linear += velocity * column;
            angular += velocity * angularColumn;
        }
        const uint globalBody =
            dispatch.articulationFirstBody + localBody;
        MRBodyStateGPU state{};
        state.position =
            privateBodyPoses[bodyPoseBase + localBody].position;
        state.orientation = orientation;
        state.linearVelocityAndInverseMass = float4(linear, 0.0f);
        state.angularVelocity = float4(angular, 0.0f);
        state.flagsAndIndices[0] = bodies[globalBody].motionType;
        state.flagsAndIndices[1] = dispatch.articulationIndex;
        state.flagsAndIndices[2] = globalBody;
        state.flagsAndIndices[3] = 0u;
        candidateBodies[bodyBase + globalBody] = state;
    }

    const uint privatePointBase =
        environment * dispatch.privatePointWorldStride +
        dispatch.bodyProbeCount;
    const uint privateAttachmentJacobianBase =
        privateJacobianBase +
        dispatch.bodyProbeCount * 3u * dispatch.nv;
    const uint candidatePointBase =
        environment * dispatch.candidatePointWorldStride;
    const uint candidateJacobianBase =
        environment * dispatch.candidatePointJacobianStride;
    if ((dispatch.flags &
         MR_NUMANX_HUMAN_MATTER_CANDIDATE_HAS_POINT_WORLD) != 0u) {
        for (uint point = lane;
             point < dispatch.candidatePointCount; point += threadCount)
            candidatePointWorld[candidatePointBase + point] =
                privatePointWorld[privatePointBase + point];
    }
    const uint logicalJacobianCount =
        dispatch.candidatePointCount * 3u * dispatch.nv;
    for (uint logicalIndex = lane; logicalIndex < logicalJacobianCount;
         logicalIndex += threadCount) {
        const uint row = logicalIndex / dispatch.nv;
        const uint dof = logicalIndex - row * dispatch.nv;
        candidatePointJacobians[
            candidateJacobianBase +
            row * dispatch.deltaVelocityStride + dof] =
            privatePointJacobians[
                privateAttachmentJacobianBase +
                row * dispatch.nv + dof];
    }
}

// Builds the exact frozen source-step effective tangent used by stand:
// A0 = J(q0)^T I J(q0) + armature + h*D, followed by the same deterministic
// lane-zero Cholesky order as mr_numi_human_stand_step.
kernel void mr_numanx_human_matter_source_factor(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(3)]],
    constant const MRNumiHumanStandDispatchGPU& stand [[buffer(4)]],
    constant const MRNumanXHumanMatterDispatchGPU& dispatch [[buffer(5)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(6)]],
    device const float* pointJacobians [[buffer(7)]],
    device float* spatialJacobians [[buffer(8)]],
    device float* factors [[buffer(9)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(10)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    device const MRWorldGPU& world = worlds[0];
    device const MRArticulationGPU& articulation =
        articulations[stand.articulationIndex];
    const uint bodyCount = articulation.bodyCount;
    const uint nv = articulation.nv;
    const uint bodyPoseBase = environment * stand.bodyPoseStride;
    const uint pointJacobianBase =
        environment * stand.pointJacobianStride;
    const uint spatialBase = environment * bodyCount * 6u * nv;
    const uint factorBase = environment * nv * nv;
    device float* factor = factors + factorBase;

    if (lane == 0u &&
        (!validOwnerDispatch(dispatch) ||
         !matchingOwner(owner, dispatch, environment) ||
         owner.stage != MR_NUMANX_HUMAN_MATTER_STAGE_BEGUN ||
         owner.code != MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS ||
         stand.abiVersion != MR_NUMI_HUMAN_STAND_ABI_VERSION ||
         stand.environmentCount != dispatch.environmentCount ||
         stand.stepIndex != 0u || stand.stepCount != 1u ||
         stand.qStride != dispatch.qStride ||
         stand.vStride != dispatch.vStride ||
         stand.articulationIndex >= world.articulationCount ||
         articulation.rootType != MR_ROOT_FLOATING ||
         articulation.nq != dispatch.nq || articulation.nv != dispatch.nv ||
         bodyCount == 0u ||
         bodyCount > MR_NUMI_HUMAN_STAND_MAX_BODIES ||
         stand.bodyPoseStride < bodyCount ||
         stand.bodyJacobianPointOffset > stand.pointWorldStride ||
         bodyCount >
             (stand.pointWorldStride - stand.bodyJacobianPointOffset) / 4u ||
         stand.pointJacobianStride /
             max(3u * nv, 1u) < stand.pointWorldStride ||
         (stand.flags & (MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
                         MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES)) != 0u ||
         stand.groundPointAndTimestep.w != dispatch.timestepAndInverse.x)) {
        failOwner(
            owner, MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_DISPATCH
        );
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (owner.stage != MR_NUMANX_HUMAN_MATTER_STAGE_BEGUN) return;

    const uint spatialElements = bodyCount * nv;
    for (uint index = lane; index < spatialElements; index += threadCount) {
        const uint localBody = index / nv;
        const uint dof = index - localBody * nv;
        const uint probe = stand.bodyJacobianPointOffset + 4u * localBody;
        const uint probeBase = pointJacobianBase + probe * 3u * nv;
        const float3 linear{
            pointJacobians[probeBase + 0u * nv + dof],
            pointJacobians[probeBase + 1u * nv + dof],
            pointJacobians[probeBase + 2u * nv + dof],
        };
        const float3 dx{
            pointJacobians[probeBase + 3u * nv + 0u * nv + dof] - linear.x,
            pointJacobians[probeBase + 3u * nv + 1u * nv + dof] - linear.y,
            pointJacobians[probeBase + 3u * nv + 2u * nv + dof] - linear.z,
        };
        const float3 dy{
            pointJacobians[probeBase + 6u * nv + 0u * nv + dof] - linear.x,
            pointJacobians[probeBase + 6u * nv + 1u * nv + dof] - linear.y,
            pointJacobians[probeBase + 6u * nv + 2u * nv + dof] - linear.z,
        };
        const float3 dz{
            pointJacobians[probeBase + 9u * nv + 0u * nv + dof] - linear.x,
            pointJacobians[probeBase + 9u * nv + 1u * nv + dof] - linear.y,
            pointJacobians[probeBase + 9u * nv + 2u * nv + dof] - linear.z,
        };
        const float4 orientation =
            bodyPoses[bodyPoseBase + localBody].orientation;
        const float3 axisX = quaternionRotate(
            orientation, float3(1.0f, 0.0f, 0.0f));
        const float3 axisY = quaternionRotate(
            orientation, float3(0.0f, 1.0f, 0.0f));
        const float3 axisZ = quaternionRotate(
            orientation, float3(0.0f, 0.0f, 1.0f));
        const float3 angular = 0.5f * (
            cross(axisX, dx) + cross(axisY, dy) + cross(axisZ, dz));
        const uint base = spatialBase + localBody * 6u * nv + dof;
        spatialJacobians[base + 0u * nv] = angular.x;
        spatialJacobians[base + 1u * nv] = angular.y;
        spatialJacobians[base + 2u * nv] = angular.z;
        spatialJacobians[base + 3u * nv] = linear.x;
        spatialJacobians[base + 4u * nv] = linear.y;
        spatialJacobians[base + 5u * nv] = linear.z;
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint matrixElements = nv * nv;
    for (uint index = lane; index < matrixElements; index += threadCount) {
        const uint row = index / nv;
        const uint column = index - row * nv;
        float value = 0.0f;
        for (uint localBody = 0u; localBody < bodyCount; ++localBody) {
            const uint globalBody = articulation.firstBody + localBody;
            device const MRBodyPropertiesGPU& body = bodies[globalBody];
            const uint base = spatialBase + localBody * 6u * nv;
            const float3 leftAngular{
                spatialJacobians[base + 0u * nv + row],
                spatialJacobians[base + 1u * nv + row],
                spatialJacobians[base + 2u * nv + row],
            };
            const float3 rightAngular{
                spatialJacobians[base + 0u * nv + column],
                spatialJacobians[base + 1u * nv + column],
                spatialJacobians[base + 2u * nv + column],
            };
            const float3 leftLinear{
                spatialJacobians[base + 3u * nv + row],
                spatialJacobians[base + 4u * nv + row],
                spatialJacobians[base + 5u * nv + row],
            };
            const float3 rightLinear{
                spatialJacobians[base + 3u * nv + column],
                spatialJacobians[base + 4u * nv + column],
                spatialJacobians[base + 5u * nv + column],
            };
            value += dot(
                leftAngular,
                worldInertiaMultiply(
                    body,
                    bodyPoses[bodyPoseBase + localBody].orientation,
                    rightAngular
                )
            ) + body.massAndInverseMass.x * dot(leftLinear, rightLinear);
        }
        if (row == column) {
            device const MRDofPropertiesGPU& dof =
                dofs[articulation.vOffset + row];
            value += dof.drive.z;
            if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
                value += dispatch.timestepAndInverse.x * dof.drive.y;
            }
        }
        factor[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane != 0u) return;

    for (uint row = 0u; row < nv; ++row) {
        float scale = 0.0f;
        for (uint column = 0u; column < nv; ++column) {
            scale = max(scale, abs(factor[row * nv + column]));
        }
        for (uint column = 0u; column <= row; ++column) {
            float value = factor[row * nv + column];
            for (uint inner = 0u; inner < column; ++inner) {
                value -= factor[row * nv + inner] *
                    factor[column * nv + inner];
            }
            if (row == column) {
                if (!(value > max(
                        kPivotFloor,
                        scale * 8.0f * 1.1920928955078125e-7f)) ||
                    !isfinite(value)) {
                    failOwner(
                        owner,
                        MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_DISPATCH
                    );
                    return;
                }
                factor[row * nv + row] = sqrt(value);
            } else {
                factor[row * nv + column] =
                    value / factor[column * nv + column];
            }
        }
    }
}

kernel void mr_numanx_human_matter_consume_reaction(
    constant MRNumanXHumanMatterDispatchGPU& dispatch [[buffer(0)]],
    device const float* matterReaction [[buffer(1)]],
    device const MRNumanXCoupledHumanStatusGPU* jointStatuses [[buffer(2)]],
    device float* generalizedForces [[buffer(3)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(4)]],
    threadgroup atomic_uint& invalid [[threadgroup(0)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    if (lane == 0u) {
        atomic_store_explicit(&invalid, 0u, memory_order_relaxed);
        const MRNumanXCoupledHumanStatusGPU joint =
            jointStatuses[environment * dispatch.jointStatusStride];
        if (!validOwnerDispatch(dispatch) ||
            !matchingOwner(owner, dispatch, environment) ||
            owner.stage != MR_NUMANX_HUMAN_MATTER_STAGE_BEGUN ||
            owner.code != MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS ||
            owner.reactionConsumed != 0u ||
            joint.abiVersion != MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
            joint.environment != environment ||
            joint.stepIndex != dispatch.stepIndex ||
            joint.decision != MR_NUMANX_COUPLED_HUMAN_PENDING ||
            joint.humanCode != 0u ||
            joint.matterCompletedMicrosteps != 0u) {
            atomic_store_explicit(&invalid, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint reactionBase = environment * dispatch.reactionStride;
    for (uint dof = lane; dof < dispatch.nv; dof += threadCount) {
        if (!isfinite(matterReaction[reactionBase + dof])) {
            atomic_store_explicit(&invalid, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&invalid, memory_order_relaxed) != 0u) {
        if (lane == 0u) {
            failOwner(
                owner,
                owner.stage == MR_NUMANX_HUMAN_MATTER_STAGE_BEGUN
                    ? MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_JOINT_STATUS
                    : MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_ORDER
            );
        }
        return;
    }
    const uint forceBase =
        environment * dispatch.generalizedForceStride +
        dispatch.generalizedForceOffset;
    for (uint dof = lane; dof < dispatch.nv; dof += threadCount) {
        generalizedForces[forceBase + dof] +=
            matterReaction[reactionBase + dof];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0u) {
        owner.reactionConsumed = 1u;
        owner.stage = MR_NUMANX_HUMAN_MATTER_STAGE_REACTION_CONSUMED;
    }
}

// First-command-buffer close.  A successful physical joint decision becomes
// PHYSICAL_PREPARED and keeps q/v/MyoSim plus all checkpoints quarantined.
// Reject, pending, malformed, and failed physical decisions restore
// immediately.  Neither path emits a final root decision or token.
kernel void mr_numanx_human_matter_prepare_physical(
    constant MRNumanXHumanMatterDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumanXCoupledHumanStatusGPU* jointStatuses [[buffer(1)]],
    device uchar* preparedTokens [[buffer(2)]],
    device float* q [[buffer(3)]],
    device float* v [[buffer(4)]],
    device MRMujocoMuscleStateGPU* mujocoStates [[buffer(5)]],
    device const float* qCheckpoint [[buffer(6)]],
    device const float* vCheckpoint [[buffer(7)]],
    device const MRMujocoMuscleStateGPU* mujocoCheckpoint [[buffer(8)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(9)]],
    threadgroup atomic_uint& restore [[threadgroup(0)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    const uint tokenBase =
        environment * dispatch.acceptedTokenStrideBytes;
    if (lane == 0u) {
        const MRNumanXCoupledHumanStatusGPU joint =
            jointStatuses[environment * dispatch.jointStatusStride];
        const device ulong* tokenFingerprint =
            reinterpret_cast<device ulong*>(
                preparedTokens + tokenBase +
                    MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_FINGERPRINT_OFFSET);
        const bool prepare = validOwnerDispatch(dispatch) &&
            matchingOwner(owner, dispatch, environment) &&
            owner.stage ==
                MR_NUMANX_HUMAN_MATTER_STAGE_REACTION_CONSUMED &&
            owner.code == MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS &&
            owner.reactionConsumed == 1u &&
            joint.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
            joint.environment == environment &&
            joint.stepIndex == dispatch.stepIndex &&
            joint.decision == MR_NUMANX_COUPLED_HUMAN_ACCEPT &&
            joint.humanCode == MR_NUMI_HUMAN_STAND_SUCCESS &&
            joint.humanCompletedSteps == 1u &&
            joint.matterCompletedMicrosteps != 0u &&
            *tokenFingerprint != 0u;
        atomic_store_explicit(
            &restore, prepare ? 0u : 1u, memory_order_relaxed);
        owner.preparedTokenPreserved = prepare ? 1u : 0u;
        owner.restored = prepare ? 0u : 1u;
        owner.stage = prepare
            ? MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_PREPARED
            : MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_REJECT_RESTORED;
        if (!prepare &&
            owner.code == MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS) {
            owner.code = MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_JOINT_STATUS;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&restore, memory_order_relaxed) == 0u) return;

    const uint qBase = environment * dispatch.qStride;
    for (uint coordinate = lane;
         coordinate < dispatch.nq; coordinate += threadCount) {
        q[qBase + coordinate] = qCheckpoint[qBase + coordinate];
    }
    const uint vBase = environment * dispatch.vStride;
    for (uint dof = lane; dof < dispatch.nv; dof += threadCount) {
        v[vBase + dof] = vCheckpoint[vBase + dof];
    }
    const uint stateBase = environment * dispatch.mujocoStateStride;
    for (uint state = lane;
         state < dispatch.mujocoStateCount; state += threadCount) {
        mujocoStates[stateBase + state] =
            mujocoCheckpoint[stateBase + state];
    }
    for (uint byte = lane;
         byte < MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_BYTES;
         byte += threadCount) {
        preparedTokens[tokenBase + byte] = 0u;
    }
}

// Last owner operation on the first borrowed command buffer. The owner event
// signal is encoded immediately after this kernel. If that command buffer
// fails before completing, the host completion path overwrites this field
// with FAILED before advancing the event only for liveness.
kernel void mr_numanx_human_matter_mark_physical_complete(
    constant MRNumanXHumanMatterDispatchGPU& dispatch [[buffer(0)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(1)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    const bool terminalStage =
        owner.stage == MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_PREPARED ||
        owner.stage ==
            MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_REJECT_RESTORED;
    const bool valid = validOwnerDispatch(dispatch) &&
        matchingOwner(owner, dispatch, environment) && terminalStage &&
        owner.physicalCommandStatus ==
            MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_UNKNOWN &&
        owner.qCoordinateCount == dispatch.nq &&
        owner.dofCount == dispatch.nv;
    owner.physicalCommandStatus = valid
        ? MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE
        : MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_FAILED;
    if (!valid) failOwner(
        owner, MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_ORDER);
}

// ABI4 phase 2a. This kernel is intentionally mutation-free with respect to
// Human q/v/MyoSim and Matter. It only emits an integrity-checked proposal and
// an immutable copy of the prepared 64-byte token.
kernel void mr_numanx_human_matter_propose_prepared(
    constant MRNumanXHumanMatterProposalDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumanXHumanMatterBrainCommitWitnessGPU*
        brainWitnesses [[buffer(1)]],
    device const uchar* preparedTokens [[buffer(2)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(3)]],
    device MRNumanXHumanMatterProposalGPU* proposals [[buffer(4)]],
    device uchar* proposedTokens [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    const bool dispatchValid = validProposalDispatch(dispatch);
    const bool ownerMatches = dispatchValid &&
        owner.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        owner.environment == environment && owner.stepIndex == dispatch.stepIndex &&
        owner.substepIndex == dispatch.substepIndex &&
        owner.transactionSlot == dispatch.transactionSlot &&
        owner.controlStep == dispatch.controlStep &&
        owner.dofCount != 0u &&
        owner.dofCount <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        owner.qCoordinateCount == owner.dofCount + 1u &&
        owner.qCoordinateCount <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        owner.programFingerprint == dispatch.programFingerprint &&
        owner.transactionFingerprint == dispatch.transactionFingerprint &&
        owner.linearizationEpoch == dispatch.linearizationEpoch &&
        owner.slotGeneration == dispatch.slotGeneration &&
        owner.physicsSubstepCount == dispatch.physicsSubstepCount;
    const bool physicalComplete = ownerMatches &&
        owner.physicalCommandStatus ==
            MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE;
    const bool physicalPrepared = physicalComplete &&
        owner.stage == MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_PREPARED &&
        owner.code == MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS &&
        owner.reactionConsumed == 1u && owner.restored == 0u &&
        owner.preparedTokenPreserved == 1u;
    const bool physicalRejected = physicalComplete &&
        owner.stage ==
            MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_REJECT_RESTORED &&
        owner.restored == 1u;
    const uint preparedBase = environment * dispatch.preparedTokenStrideBytes;
    const uint proposedBase = environment * dispatch.proposedTokenStrideBytes;
    const device ulong* tokenFingerprint =
        reinterpret_cast<const device ulong*>(
        preparedTokens + preparedBase +
            MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_FINGERPRINT_OFFSET);
    const ulong physicsTokenFingerprint = physicalPrepared
        ? *tokenFingerprint : 0ul;
    const bool preparedTokenValid = physicalPrepared &&
        validAcceptedToken(
            preparedTokens + preparedBase,
            dispatch.transactionFingerprint,
            physicsTokenFingerprint);

    bool accept = false;
    uint code = MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_OWNER;
    ulong brainProgram = 0ul;
    ulong brainShadow = 0ul;
    ulong brainWitness = 0ul;
    const bool forceReject = dispatchValid && dispatch.flags ==
        MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCE_REJECT;
    if (!physicalComplete) {
        code = MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_OWNER;
    } else if (physicalRejected) {
        code = MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT;
    } else if (forceReject) {
        code = MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT;
    } else if (preparedTokenValid) {
        const MRNumanXHumanMatterBrainCommitWitnessGPU witness =
            brainWitnesses[environment * dispatch.brainWitnessStride];
        bool reservedZero = true;
        for (uint index = 0u; index < 2u; ++index)
            reservedZero = reservedZero && witness.reserved1[index] == 0ul;
        const bool witnessValid =
            witness.magic ==
                MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_MAGIC &&
            witness.abiVersion ==
                MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_ABI_VERSION &&
            witness.structBytes ==
                MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_BYTES &&
            witness.status ==
                MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_COMPLETE &&
            (witness.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT ||
             witness.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT) &&
            witness.environment == environment &&
            witness.stepIndex == dispatch.stepIndex &&
            witness.substepIndex == dispatch.substepIndex &&
            witness.transactionSlot == dispatch.transactionSlot &&
            witness.physicsSubstepCount == dispatch.physicsSubstepCount &&
            witness.controlStep == dispatch.controlStep && witness.reserved0 == 0u &&
            witness.programFingerprint == dispatch.programFingerprint &&
            witness.transactionFingerprint == dispatch.transactionFingerprint &&
            witness.linearizationEpoch == dispatch.linearizationEpoch &&
            witness.slotGeneration == dispatch.slotGeneration &&
            witness.physicsTokenFingerprint == physicsTokenFingerprint &&
            witness.brainProgramFingerprint != 0ul &&
            witness.brainShadowStateFingerprint != 0ul &&
            witness.witnessFingerprint != 0ul && reservedZero &&
            witness.witnessFingerprint == brainWitnessFingerprint(witness);
        if (witnessValid) {
            accept = witness.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
            code = accept ? MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS
                          : MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT;
            brainProgram = witness.brainProgramFingerprint;
            brainShadow = witness.brainShadowStateFingerprint;
            brainWitness = witness.witnessFingerprint;
        } else {
            code = witness.physicsTokenFingerprint != physicsTokenFingerprint
                ? MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH
                : MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS;
        }
    } else if (physicalPrepared) {
        code = MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH;
    }

    MRNumanXHumanMatterProposalGPU proposal{};
    proposal.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    const bool proposalReady = physicalComplete &&
        code != MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_OWNER;
    proposal.status = proposalReady
        ? MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY
        : MR_NUMANX_HUMAN_MATTER_PROPOSAL_TERMINAL_NO_TOUCH;
    proposal.decision = !proposalReady
        ? MR_NUMANX_HUMAN_MATTER_ROOT_PENDING
        : (accept ? MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT
                  : MR_NUMANX_HUMAN_MATTER_ROOT_REJECT);
    proposal.code = code;
    proposal.programFingerprint = dispatch.programFingerprint;
    proposal.transactionFingerprint = dispatch.transactionFingerprint;
    proposal.linearizationEpoch = dispatch.linearizationEpoch;
    proposal.slotGeneration = dispatch.slotGeneration;
    proposal.physicsTokenFingerprint = accept
        ? physicsTokenFingerprint : 0ul;
    proposal.brainProgramFingerprint = accept ? brainProgram : 0ul;
    proposal.brainShadowStateFingerprint = accept ? brainShadow : 0ul;
    proposal.brainWitnessFingerprint = accept ? brainWitness : 0ul;
    proposal.candidatePublicationFingerprint =
        dispatch.candidatePublicationFingerprint;
    proposal.humanIOIdentityFingerprint =
        dispatch.humanIOIdentityFingerprint;
    proposal.environment = environment;
    proposal.stepIndex = dispatch.stepIndex;
    proposal.substepIndex = dispatch.substepIndex;
    proposal.transactionSlot = dispatch.transactionSlot;
    proposal.physicsSubstepCount = dispatch.physicsSubstepCount;
    proposal.controlStep = dispatch.controlStep;
    proposal.proposalFingerprint = proposalFingerprint(proposal);
    for (uint byte = 0u; byte < MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
         ++byte) {
        proposedTokens[proposedBase + byte] = accept
            ? preparedTokens[preparedBase + byte] : 0u;
    }
    proposals[environment * dispatch.proposalStride] = proposal;
    if (ownerMatches && physicalComplete) {
        owner.stage = MR_NUMANX_HUMAN_MATTER_STAGE_PROPOSED;
        owner.appliedOutcomePublished = 0u;
    }
}

// ABI4 phase 3a. Validates the immutable proposal and Brain ACK and writes only
// a private action. Matter consumes this action before either owner mutates its
// authoritative candidate/checkpoint state.
kernel void mr_numanx_human_matter_validate_apply(
    constant MRNumanXHumanMatterApplyDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumanXHumanMatterProposalGPU* proposals [[buffer(1)]],
    device const MRNumanXHumanMatterBrainAckGPU* brainAcks [[buffer(2)]],
    device const uchar* proposedTokens [[buffer(3)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(4)]],
    device MRNumanXHumanMatterApplyActionGPU* actions [[buffer(5)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) return;
    const MRNumanXHumanMatterProposalGPU proposal = proposals[environment];
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    const bool dispatchValid = validApplyDispatch(dispatch);
    const bool ownerMatches = dispatchValid &&
        owner.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        owner.environment == environment && owner.stepIndex == dispatch.stepIndex &&
        owner.substepIndex == dispatch.substepIndex &&
        owner.transactionSlot == dispatch.transactionSlot &&
        owner.controlStep == dispatch.controlStep &&
        owner.qCoordinateCount == dispatch.nq &&
        owner.dofCount == dispatch.nv &&
        owner.programFingerprint == dispatch.programFingerprint &&
        owner.transactionFingerprint == dispatch.transactionFingerprint &&
        owner.linearizationEpoch == dispatch.linearizationEpoch &&
        owner.slotGeneration == dispatch.slotGeneration &&
        owner.physicsSubstepCount == dispatch.physicsSubstepCount &&
        owner.stage == MR_NUMANX_HUMAN_MATTER_STAGE_PROPOSED &&
        owner.appliedOutcomePublished == 0u &&
        owner.physicalCommandStatus ==
            MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE;
    const bool proposalAcceptConsistent =
        (proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
         proposal.code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS &&
         proposal.physicsTokenFingerprint != 0ul &&
         proposal.brainProgramFingerprint != 0ul &&
         proposal.brainShadowStateFingerprint != 0ul &&
         proposal.brainWitnessFingerprint != 0ul);
    const bool proposalRejectConsistent =
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
        readyProposalRejectCode(proposal.code) &&
        proposal.physicsTokenFingerprint == 0ul &&
        proposal.brainProgramFingerprint == 0ul &&
        proposal.brainShadowStateFingerprint == 0ul &&
        proposal.brainWitnessFingerprint == 0ul;
    const bool proposalStatusConsistent = proposalAcceptConsistent ||
        proposalRejectConsistent;
    const bool proposalValid = ownerMatches &&
        proposal.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
        proposalStatusConsistent &&
        proposal.programFingerprint == dispatch.programFingerprint &&
        proposal.transactionFingerprint == dispatch.transactionFingerprint &&
        proposal.linearizationEpoch == dispatch.linearizationEpoch &&
        proposal.slotGeneration == dispatch.slotGeneration &&
        proposal.environment == environment &&
        proposal.stepIndex == dispatch.stepIndex &&
        proposal.substepIndex == dispatch.substepIndex &&
        proposal.transactionSlot == dispatch.transactionSlot &&
        proposal.physicsSubstepCount == dispatch.physicsSubstepCount &&
        proposal.controlStep == dispatch.controlStep &&
        proposal.candidatePublicationFingerprint != 0ul &&
        proposal.humanIOIdentityFingerprint != 0ul &&
        proposal.proposalFingerprint != 0ul &&
        proposal.proposalFingerprint == proposalFingerprint(proposal);
    device const uchar* proposedToken = proposedTokens +
        environment * dispatch.proposedTokenStrideBytes;
    const bool tokenMatches = proposalValid &&
        (proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
             ? zeroAcceptedToken(proposedToken)
             : validAcceptedToken(
                   proposedToken,
                   dispatch.transactionFingerprint,
                   proposal.physicsTokenFingerprint));
    const bool forceReject = dispatch.flags ==
        MR_NUMANX_HUMAN_MATTER_APPLY_FORCE_REJECT;
    bool accept = false;
    uint code = MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER;
    MRNumanXHumanMatterBrainAckGPU ack{};
    if (proposalValid && tokenMatches && !forceReject) {
        ack = brainAcks[environment * dispatch.brainAckStride];
        const bool ackAcceptConsistent =
            (ack.status == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT &&
             ack.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
             ack.code == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS &&
             proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
             ack.physicsTokenFingerprint ==
                 proposal.physicsTokenFingerprint &&
             ack.physicsTokenFingerprint != 0ul &&
             ack.preflightFingerprint != 0ul &&
             ack.fastGateFingerprint != 0ul &&
             ack.brainWitnessFingerprint ==
                 proposal.brainWitnessFingerprint &&
             ack.brainWitnessFingerprint != 0ul &&
             ack.brainProgramFingerprint ==
                 proposal.brainProgramFingerprint &&
             ack.brainProgramFingerprint != 0ul);
        const bool ackRejectConsistent =
            (ack.status == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT &&
             ack.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
             ack.code ==
                 MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT &&
             proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
             ack.physicsTokenFingerprint == 0ul &&
             ack.preflightFingerprint == 0ul &&
             ack.fastGateFingerprint == 0ul &&
             ack.brainWitnessFingerprint == 0ul &&
             ack.brainProgramFingerprint != 0ul);
        const bool ackStatusConsistent =
            ackAcceptConsistent || ackRejectConsistent;
        const bool ackValid =
            ack.abiVersion == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION &&
            ackStatusConsistent &&
            ack.programFingerprint == dispatch.programFingerprint &&
            ack.transactionFingerprint == dispatch.transactionFingerprint &&
            ack.linearizationEpoch == dispatch.linearizationEpoch &&
            ack.slotGeneration == dispatch.slotGeneration &&
            ack.proposalFingerprint == proposal.proposalFingerprint &&
            ack.environment == environment && ack.stepIndex == dispatch.stepIndex &&
            ack.substepIndex == dispatch.substepIndex &&
            ack.transactionSlot == dispatch.transactionSlot &&
            ack.physicsSubstepCount == dispatch.physicsSubstepCount &&
            ack.controlStep == dispatch.controlStep && ack.ackFingerprint != 0ul &&
            ack.ackFingerprint == brainAckFingerprint(ack);
        accept = ackValid && proposal.decision ==
            MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
            ack.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
        if (!ackValid) {
            code = MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_BRAIN_ACK;
        } else if (accept) {
            code = MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS;
        } else {
            code = appliedCodeForProposalReject(proposal.code);
        }
    } else if (forceReject && ownerMatches) {
        code = MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT;
    } else if (proposalValid && proposal.decision ==
                   MR_NUMANX_HUMAN_MATTER_ROOT_REJECT) {
        code = appliedCodeForProposalReject(proposal.code);
    } else if (!tokenMatches) {
        code = MR_NUMANX_HUMAN_MATTER_APPLIED_TOKEN_MISMATCH;
    }

    MRNumanXHumanMatterApplyActionGPU action{};
    action.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    const bool terminal = !ownerMatches || !proposalValid || !tokenMatches;
    action.status = terminal
        ? MR_NUMANX_HUMAN_MATTER_APPLY_TERMINAL_NO_TOUCH
        : (accept ? MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT
                  : MR_NUMANX_HUMAN_MATTER_APPLY_REJECT);
    action.decision = terminal ? MR_NUMANX_HUMAN_MATTER_ROOT_PENDING
        : (accept ? MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT
                  : MR_NUMANX_HUMAN_MATTER_ROOT_REJECT);
    action.code = code;
    action.programFingerprint = dispatch.programFingerprint;
    action.transactionFingerprint = dispatch.transactionFingerprint;
    action.linearizationEpoch = dispatch.linearizationEpoch;
    action.slotGeneration = dispatch.slotGeneration;
    action.physicsTokenFingerprint = forceReject
        ? 0ul : ack.physicsTokenFingerprint;
    action.proposalFingerprint = proposal.proposalFingerprint;
    action.ackFingerprint = ack.ackFingerprint;
    action.preflightFingerprint = ack.preflightFingerprint;
    action.fastGateFingerprint = ack.fastGateFingerprint;
    action.brainWitnessFingerprint = forceReject
        ? 0ul : ack.brainWitnessFingerprint;
    action.environment = environment;
    action.stepIndex = dispatch.stepIndex;
    action.substepIndex = dispatch.substepIndex;
    action.transactionSlot = dispatch.transactionSlot;
    action.physicsSubstepCount = dispatch.physicsSubstepCount;
    action.controlStep = dispatch.controlStep;
    action.actionFingerprint = applyActionFingerprint(action);
    actions[environment * dispatch.applyActionStride] = action;
}

// ABI4 phase 3b. Runs after the adapter/Matter apply encoder. Human mutation or
// restoration and the final 64-byte token occur first; the applied record is
// written last and remains quarantined until the host publication fence close.
kernel void mr_numanx_human_matter_complete_apply(
    constant MRNumanXHumanMatterApplyDispatchGPU& dispatch [[buffer(0)]],
    device const MRNumanXHumanMatterProposalGPU* proposals [[buffer(1)]],
    device const MRNumanXHumanMatterApplyActionGPU* actions [[buffer(2)]],
    device const MRNumanXHumanMatterMatterApplyOutcomeGPU*
        matterOutcomes [[buffer(3)]],
    device const uchar* proposedTokens [[buffer(4)]],
    device float* q [[buffer(5)]],
    device float* v [[buffer(6)]],
    device MRMujocoMuscleStateGPU* mujocoStates [[buffer(7)]],
    device const float* qCheckpoint [[buffer(8)]],
    device const float* vCheckpoint [[buffer(9)]],
    device const MRMujocoMuscleStateGPU* mujocoCheckpoint [[buffer(10)]],
    device MRNumanXHumanMatterOwnerStatusGPU* ownerStatuses [[buffer(11)]],
    device MRNumanXHumanMatterAppliedOutcomeGPU* appliedOutcomes [[buffer(12)]],
    device uchar* finalTokens [[buffer(13)]],
    threadgroup MRNumanXHumanMatterAppliedOutcomeGPU& pending [[threadgroup(0)]],
    threadgroup uint& tokenAction [[threadgroup(1)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumanXHumanMatterOwnerStatusGPU& owner =
        ownerStatuses[environment * dispatch.ownerStatusStride];
    const MRNumanXHumanMatterProposalGPU proposal = proposals[environment];
    const MRNumanXHumanMatterApplyActionGPU action =
        actions[environment * dispatch.applyActionStride];
    const MRNumanXHumanMatterMatterApplyOutcomeGPU matter =
        matterOutcomes[environment * dispatch.matterOutcomeStride];
    if (lane == 0u) {
        const bool ownerValid = validApplyDispatch(dispatch) &&
            owner.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
            owner.environment == environment && owner.stepIndex == dispatch.stepIndex &&
            owner.substepIndex == dispatch.substepIndex &&
            owner.transactionSlot == dispatch.transactionSlot &&
            owner.controlStep == dispatch.controlStep &&
            owner.qCoordinateCount == dispatch.nq &&
            owner.dofCount == dispatch.nv &&
            owner.programFingerprint == dispatch.programFingerprint &&
            owner.transactionFingerprint == dispatch.transactionFingerprint &&
            owner.linearizationEpoch == dispatch.linearizationEpoch &&
            owner.slotGeneration == dispatch.slotGeneration &&
            owner.physicsSubstepCount == dispatch.physicsSubstepCount &&
            owner.stage == MR_NUMANX_HUMAN_MATTER_STAGE_PROPOSED &&
            owner.appliedOutcomePublished == 0u &&
            owner.physicalCommandStatus ==
                MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE;
        const bool proposalAcceptConsistent =
            (proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
             proposal.code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS &&
             proposal.physicsTokenFingerprint != 0ul &&
             proposal.brainProgramFingerprint != 0ul &&
             proposal.brainShadowStateFingerprint != 0ul &&
             proposal.brainWitnessFingerprint != 0ul);
        const bool proposalRejectConsistent =
            proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
            readyProposalRejectCode(proposal.code) &&
            proposal.physicsTokenFingerprint == 0ul &&
            proposal.brainProgramFingerprint == 0ul &&
            proposal.brainShadowStateFingerprint == 0ul &&
            proposal.brainWitnessFingerprint == 0ul;
        const bool proposalStatusConsistent = proposalAcceptConsistent ||
            proposalRejectConsistent;
        const bool proposalValid = ownerValid &&
            proposal.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
            proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
            proposalStatusConsistent &&
            proposal.programFingerprint == dispatch.programFingerprint &&
            proposal.transactionFingerprint == dispatch.transactionFingerprint &&
            proposal.linearizationEpoch == dispatch.linearizationEpoch &&
            proposal.slotGeneration == dispatch.slotGeneration &&
            proposal.environment == environment &&
            proposal.stepIndex == dispatch.stepIndex &&
            proposal.substepIndex == dispatch.substepIndex &&
            proposal.transactionSlot == dispatch.transactionSlot &&
            proposal.physicsSubstepCount == dispatch.physicsSubstepCount &&
            proposal.controlStep == dispatch.controlStep &&
            proposal.candidatePublicationFingerprint != 0ul &&
            proposal.humanIOIdentityFingerprint != 0ul &&
            proposal.proposalFingerprint != 0ul &&
            proposal.proposalFingerprint == proposalFingerprint(proposal);
        device const uchar* proposedToken = proposedTokens +
            environment * dispatch.proposedTokenStrideBytes;
        const bool tokenValid = proposal.decision ==
                MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
            ? zeroAcceptedToken(proposedToken)
            : validAcceptedToken(
                  proposedToken,
                  dispatch.transactionFingerprint,
                  proposal.physicsTokenFingerprint);
        const bool actionAcceptConsistent =
            (action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT &&
             action.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
             action.code == MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS &&
             proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
             action.physicsTokenFingerprint ==
                 proposal.physicsTokenFingerprint &&
             action.ackFingerprint != 0ul &&
             action.preflightFingerprint != 0ul &&
             action.fastGateFingerprint != 0ul &&
             action.brainWitnessFingerprint ==
                 proposal.brainWitnessFingerprint);
        const bool actionForcedRejectConsistent =
            (action.status == MR_NUMANX_HUMAN_MATTER_APPLY_REJECT &&
             action.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
             action.code == MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT &&
             action.physicsTokenFingerprint == 0ul &&
             action.ackFingerprint == 0ul &&
             action.preflightFingerprint == 0ul &&
             action.fastGateFingerprint == 0ul &&
             action.brainWitnessFingerprint == 0ul);
        const bool actionZeroGateRejectConsistent =
            action.status == MR_NUMANX_HUMAN_MATTER_APPLY_REJECT &&
            action.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
            proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
            action.code == appliedCodeForProposalReject(proposal.code) &&
            action.physicsTokenFingerprint == 0ul &&
            action.ackFingerprint != 0ul &&
            action.preflightFingerprint == 0ul &&
            action.fastGateFingerprint == 0ul &&
            action.brainWitnessFingerprint == 0ul;
        const bool actionStatusConsistent = actionAcceptConsistent ||
            actionForcedRejectConsistent ||
            actionZeroGateRejectConsistent;
        const bool actionValid = proposalValid && tokenValid &&
            action.abiVersion ==
                MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
            actionStatusConsistent &&
            action.programFingerprint == dispatch.programFingerprint &&
            action.transactionFingerprint == dispatch.transactionFingerprint &&
            action.linearizationEpoch == dispatch.linearizationEpoch &&
            action.slotGeneration == dispatch.slotGeneration &&
            action.proposalFingerprint == proposal.proposalFingerprint &&
            action.environment == environment && action.stepIndex == dispatch.stepIndex &&
            action.substepIndex == dispatch.substepIndex &&
            action.transactionSlot == dispatch.transactionSlot &&
            action.physicsSubstepCount == dispatch.physicsSubstepCount &&
            action.controlStep == dispatch.controlStep &&
            action.actionFingerprint != 0ul &&
            action.actionFingerprint == applyActionFingerprint(action);
        const bool matterStatusConsistent =
            (matter.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT &&
             matter.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
             matter.code == MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS &&
             action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT) ||
            (matter.status == MR_NUMANX_HUMAN_MATTER_APPLY_REJECT &&
             matter.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
             ((action.status == MR_NUMANX_HUMAN_MATTER_APPLY_REJECT &&
               matter.code == action.code) ||
              (action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT &&
               matter.code ==
                   MR_NUMANX_HUMAN_MATTER_APPLIED_MATTER_REJECT)));
        const bool matterValid = actionValid &&
            matter.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
            matterStatusConsistent &&
            matter.programFingerprint == dispatch.programFingerprint &&
            matter.transactionFingerprint == dispatch.transactionFingerprint &&
            matter.linearizationEpoch == dispatch.linearizationEpoch &&
            matter.slotGeneration == dispatch.slotGeneration &&
            matter.physicsTokenFingerprint == action.physicsTokenFingerprint &&
            matter.proposalFingerprint == action.proposalFingerprint &&
            matter.ackFingerprint == action.ackFingerprint &&
            matter.actionFingerprint == action.actionFingerprint &&
            matter.matterProgramFingerprint != 0ul && matter.reserved0 == 0ul &&
            matter.environment == environment && matter.stepIndex == dispatch.stepIndex &&
            matter.substepIndex == dispatch.substepIndex &&
            matter.transactionSlot == dispatch.transactionSlot &&
            matter.physicsSubstepCount == dispatch.physicsSubstepCount &&
            matter.controlStep == dispatch.controlStep &&
            matter.outcomeFingerprint != 0ul &&
            matter.outcomeFingerprint == matterApplyOutcomeFingerprint(matter);
        const bool accept = matterValid &&
            action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT &&
            matter.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT;
        const bool terminal = !ownerValid || !proposalValid || !tokenValid ||
            !actionValid || !matterValid;
        pending = {};
        pending.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
        pending.status = terminal
            ? MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH
            : (accept ? MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED
                      : MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED);
        pending.decision = terminal ? MR_NUMANX_HUMAN_MATTER_ROOT_PENDING
            : (accept ? MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT
                      : MR_NUMANX_HUMAN_MATTER_ROOT_REJECT);
        pending.code = terminal
            ? (!ownerValid || !proposalValid || !tokenValid || !actionValid
                   ? MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER
                   : MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_MATTER_OUTCOME)
            : (accept ? MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS
                      : (action.status == MR_NUMANX_HUMAN_MATTER_APPLY_REJECT
                             ? action.code
                             : MR_NUMANX_HUMAN_MATTER_APPLIED_MATTER_REJECT));
        pending.programFingerprint = dispatch.programFingerprint;
        pending.transactionFingerprint = dispatch.transactionFingerprint;
        pending.linearizationEpoch = dispatch.linearizationEpoch;
        pending.slotGeneration = dispatch.slotGeneration;
        pending.physicsTokenFingerprint = terminal
            ? 0ul : action.physicsTokenFingerprint;
        pending.proposalFingerprint = action.proposalFingerprint;
        pending.ackFingerprint = action.ackFingerprint;
        pending.preflightFingerprint = action.preflightFingerprint;
        pending.fastGateFingerprint = action.fastGateFingerprint;
        pending.matterApplyFingerprint = matterValid
            ? matter.outcomeFingerprint : 0ul;
        pending.environment = environment;
        pending.stepIndex = dispatch.stepIndex;
        pending.substepIndex = dispatch.substepIndex;
        pending.transactionSlot = dispatch.transactionSlot;
        pending.physicsSubstepCount = dispatch.physicsSubstepCount;
        pending.controlStep = dispatch.controlStep;
        tokenAction = terminal ? 0u : (accept ? 1u : 2u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint actionCode = tokenAction;
    if (actionCode == 2u) {
        const uint qBase = environment * dispatch.qStride;
        for (uint coordinate = lane; coordinate < dispatch.nq;
             coordinate += threadCount)
            q[qBase + coordinate] = qCheckpoint[qBase + coordinate];
        const uint vBase = environment * dispatch.vStride;
        for (uint dof = lane; dof < dispatch.nv; dof += threadCount)
            v[vBase + dof] = vCheckpoint[vBase + dof];
        const uint stateBase = environment * dispatch.mujocoStateStride;
        for (uint state = lane; state < dispatch.mujocoStateCount;
             state += threadCount)
            mujocoStates[stateBase + state] = mujocoCheckpoint[stateBase + state];
    }
    const uint proposedBase = environment * dispatch.proposedTokenStrideBytes;
    const uint finalBase = environment * dispatch.finalTokenStrideBytes;
    if (actionCode != 0u) {
        for (uint byte = lane; byte < MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
             byte += threadCount) {
            finalTokens[finalBase + byte] = actionCode == 1u
                ? proposedTokens[proposedBase + byte] : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0u) {
        pending.appliedFingerprint = appliedOutcomeFingerprint(pending);
        appliedOutcomes[environment * dispatch.appliedOutcomeStride] = pending;
        if (actionCode == 1u) {
            owner.stage =
                MR_NUMANX_HUMAN_MATTER_STAGE_APPLIED_ACCEPT_QUARANTINED;
            owner.restored = 0u;
            owner.preparedTokenPreserved = 1u;
        } else if (actionCode == 2u) {
            owner.stage = MR_NUMANX_HUMAN_MATTER_STAGE_APPLIED_RESTORE;
            owner.restored = 1u;
            owner.preparedTokenPreserved = 0u;
        } else {
            owner.stage = MR_NUMANX_HUMAN_MATTER_STAGE_FAILED;
        }
        owner.appliedOutcomePublished = 1u;
    }
}
