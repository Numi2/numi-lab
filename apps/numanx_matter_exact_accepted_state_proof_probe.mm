#define main numanx_human_matter_adapter_probe_v2_embedded_main
#include "numanx_human_matter_adapter_probe.mm"
#undef main

#include "metalrobo/NumanXExactTransaction.hpp"
#include "numi/matter/accepted_state_apply_gpu.h"
#include "numi/matter/accepted_state_proof_gpu.h"

namespace matter_exact_proof_fixture {

using namespace adapter_fixture;

static_assert(sizeof(NMExactInboundAuthorityGPUV2) ==
              sizeof(mrnx_exact_inbound_authority_v2));
static_assert(sizeof(NMExactInboundAuthorityGPUV2) ==
              sizeof(MRNumanXExactInboundAuthorityGPUV2));
static_assert(sizeof(NMAcceptedStateProofGPUV2) ==
              sizeof(MRNumanXAcceptedStateProofGPUV2));

constexpr std::uint32_t kEnvironmentIdentifier = 0xf1234567u;
constexpr std::uint32_t kTransactionSlot = 0u;
constexpr std::uint32_t kControlStep = 37u;
constexpr std::uint64_t kAdapterProgramFingerprint =
    0x4e584d4154563201ull;
constexpr std::uint64_t kTransactionFingerprint =
    0x4e585452414e3201ull;
constexpr std::uint64_t kSubstepFingerprint =
    0x4e58535542533201ull;
constexpr std::uint64_t kAcceptedBrainTimestampNanoseconds = 1000000003ull;
constexpr std::uint64_t kAcceptedTimestampNanoseconds = 1000000019ull;
constexpr std::uint64_t kPhysicsGeneration = 7u;
constexpr std::uint64_t kLinearizationEpoch =
    0x4e584c494e320001ull;
constexpr std::uint64_t kSlotGeneration = 11u;
constexpr std::uint64_t kMotorCandidateFingerprint =
    0x4e584d4f544f5201ull;

static_assert(kAcceptedBrainTimestampNanoseconds % 1000u != 0u);
static_assert(kAcceptedTimestampNanoseconds % 1000u != 0u);
static_assert(kAcceptedTimestampNanoseconds >
              kAcceptedBrainTimestampNanoseconds);

enum class AuthorityMode {
    valid,
    corruptFingerprint,
    mismatchedIdentity,
};

struct MatterCandidateService {
    id<MTLCommandBuffer> commandBuffer = nil;
    id<MTLBuffer> rootTranslation = nil;
    id<MTLBuffer> bodyPositionLow = nil;
    id<MTLBuffer> borrowedCandidateRoot = nil;
    id<MTLBuffer> q = nil;
    id<MTLBuffer> body = nil;
    id<MTLBuffer> jacobian = nil;
    id<MTLBuffer> inverseStatus = nil;
    std::array<std::uint32_t, 4u> calls{};
};

bool encodeMatterCandidate(
    void* context,
    const numi::matter::CoupledCandidateQuery& query
) {
    auto* service = static_cast<MatterCandidateService*>(context);
    const std::uint32_t operation =
        static_cast<std::uint32_t>(query.operation);
    if (service == nullptr || service->commandBuffer == nil ||
        operation >= service->calls.size() ||
        query.generalizedVectorStride != kDofs || query.legacyHighOnly ||
        query.pointPositionLow != nullptr ||
        query.pointPositionLowGPUAddress != 0u ||
        query.pointPositionLowElementCount != 0u) {
        return false;
    }
    const bool kinematics = query.operation ==
        numi::matter::CoupledCandidateOperation::candidateKinematics;
    if (kinematics) {
        if (query.candidateRootTranslation == nullptr ||
            query.candidateBodyPositionLow == nullptr ||
            query.candidateRootTranslationElementCount != 1u ||
            query.candidateBodyPositionLowElementCount != kBodies) {
            return false;
        }
        service->borrowedCandidateRoot =
            (__bridge id<MTLBuffer>)query.candidateRootTranslation;
    } else {
        if (query.candidateBodies != nullptr ||
            query.candidateBodyStride != 0u ||
            query.candidateRootTranslation != nullptr ||
            query.candidateBodyPositionLow != nullptr ||
            query.candidateRootTranslationGPUAddress != 0u ||
            query.candidateBodyPositionLowGPUAddress != 0u ||
            query.candidateRootTranslationElementCount != 0u ||
            query.candidateBodyPositionLowElementCount != 0u) {
            return false;
        }
        if (query.operation == numi::matter::CoupledCandidateOperation::
                inverseMassPreconditioner &&
            (query.candidateQ != nullptr || query.candidateQStride != 0u)) {
            return false;
        }
    }

    id<MTLBlitCommandEncoder> blit =
        [service->commandBuffer blitCommandEncoder];
    if (blit == nil) return false;
    bool valid = false;
    switch (query.operation) {
    case numi::matter::CoupledCandidateOperation::candidateKinematics:
        valid = query.candidateQStride == kQ &&
            query.candidateBodyStride >= kBodies &&
            copy(blit, service->q, query.candidateQ,
                 kQ * sizeof(float)) &&
            copy(blit, service->body, query.candidateBodies,
                 query.candidateBodyStride * sizeof(MRBodyStateGPU)) &&
            copy(blit, service->rootTranslation,
                 query.candidateRootTranslation,
                 sizeof(MRCompensatedRootTranslationGPU)) &&
            copy(blit, service->bodyPositionLow,
                 query.candidateBodyPositionLow,
                 kBodies * sizeof(mr_float4));
        if (valid && query.pointCount != 0u) {
            valid = query.pointCount == kPoints &&
                query.pointJacobianStride == 3u * kDofs &&
                copy(blit, service->jacobian, query.pointJacobians,
                     3u * kDofs * sizeof(float));
        }
        break;
    case numi::matter::CoupledCandidateOperation::massAction:
    case numi::matter::CoupledCandidateOperation::publishCandidate:
        valid = copy(
            blit, (__bridge id<MTLBuffer>)query.input, query.output,
            kDofs * sizeof(float));
        break;
    case numi::matter::CoupledCandidateOperation::inverseMassPreconditioner:
        valid = query.statusStride == 1u &&
            copy(blit, (__bridge id<MTLBuffer>)query.input, query.output,
                 kDofs * sizeof(float)) &&
            copy(blit, service->inverseStatus, query.statuses,
                 sizeof(MRInverseMassStatusGPU));
        break;
    }
    [blit endEncoding];
    if (valid) ++service->calls[operation];
    return valid;
}

mrnx_exact_inbound_authority_v2 makeAuthority(const AuthorityMode mode) {
    mrnx_exact_inbound_authority_v2 authority{};
    authority.abi_version = MRNX_EXACT_INBOUND_AUTHORITY_ABI_V2;
    authority.struct_size = sizeof(authority);
    authority.clock_domain =
        MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    authority.clock_quantum_nanoseconds =
        MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    authority.accepted_brain_timestamp_nanoseconds =
        kAcceptedBrainTimestampNanoseconds;
    authority.brain_generation = 19u;
    authority.transaction_fingerprint = kTransactionFingerprint;
    authority.substep_fingerprint = kSubstepFingerprint;
    authority.motor_candidate_fingerprint = kMotorCandidateFingerprint;
    authority.motor_output_fingerprint = 0x4e584f5554505554ull;
    authority.motor_profile_fingerprint = 0x4e5850524f46494cull;
    authority.motor_ready_gate_fingerprint = 0x4e58524541445947ull;
    authority.brain_program_fingerprint = 0x4e58425241494e50ull;
    authority.fast_program_fingerprint = 0x4e58464153545052ull;
    authority.decision_gate_fingerprint = 0x4e58444543495349ull;
    authority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(authority);

    if (mode == AuthorityMode::corruptFingerprint) {
        authority.inbound_authority_fingerprint ^= 1u;
    } else if (mode == AuthorityMode::mismatchedIdentity) {
        authority.substep_fingerprint ^= 1u;
        authority.inbound_authority_fingerprint =
            metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
                authority);
    }
    return authority;
}

MRNumanXAcceptedStateProofGPUV2 metalroboProof(
    const NMAcceptedStateProofGPUV2& proof
) {
    MRNumanXAcceptedStateProofGPUV2 result{};
    std::memcpy(&result, &proof, sizeof(result));
    return result;
}

MRNumanXAcceptedPhysicsStateTokenGPUV2 makeExactToken(
    const MRNumanXAcceptedStateProofGPUV2& proof
) {
    MRNumanXAcceptedPhysicsStateTokenGPUV2 token{};
    token.transactionFingerprint = proof.transactionFingerprint;
    token.substepFingerprint = proof.substepFingerprint;
    token.physicsStateFingerprint = proof.physicsStateFingerprint;
    token.acceptedTimestampNanoseconds =
        proof.acceptedTimestampNanoseconds;
    token.physicsGeneration = proof.physicsGeneration;
    token.environmentIdentifier = proof.environment;
    token.flags = 0u;
    token.clockDomain = proof.clockDomain;
    token.clockQuantumNanoseconds = proof.clockQuantumNanoseconds;
    token.tokenFingerprint =
        metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(token);
    return token;
}

void requireLegacyApplyRejectsExactFamily(
    numi::matter::Runtime& matter,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const MRNumanXAcceptedStateProofGPUV2& proof
) {
    const MRNumanXAcceptedPhysicsStateTokenGPUV2 token =
        makeExactToken(proof);
    require(
        metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Valid(proof, token),
        "exact token fixture failed CPU validation");

    id<MTLBuffer> proposals = makeZeroBuffer(
        device, sizeof(NMOwnerProposalGPU), @"V2 family legacy proposal");
    id<MTLBuffer> acknowledgements = makeZeroBuffer(
        device, sizeof(NMOwnerBrainAckGPU), @"V2 family legacy ACK");
    id<MTLBuffer> actions = makeZeroBuffer(
        device, sizeof(NMOwnerApplyActionGPU), @"V2 family legacy action");
    id<MTLBuffer> outcomes = makeZeroBuffer(
        device, sizeof(NMMatterApplyOutcomeGPU), @"V2 family legacy outcome");
    id<MTLBuffer> proposedToken = makeBuffer(
        device, token, @"V2 exact token presented to V1 apply");
    id<MTLCommandBuffer> apply = [queue commandBuffer];
    require(apply != nil, "failed to allocate V1-family rejection CB");

    numi::matter::AcceptedStateApplyPass pass;
    pass.environmentCount = 1u;
    pass.environmentIdentifierBase = kEnvironmentIdentifier;
    pass.controlStep = kControlStep;
    pass.physicsSubstep = 0u;
    pass.physicsSubstepCount = 1u;
    pass.transactionSlot = kTransactionSlot;
    pass.commandBuffer = (__bridge void*)apply;
    pass.proposals = (__bridge void*)proposals;
    pass.brainAcks = (__bridge void*)acknowledgements;
    pass.applyActions = (__bridge void*)actions;
    pass.matterApplyOutcomes = (__bridge void*)outcomes;
    pass.proposedPhysicsStateTokens = (__bridge void*)proposedToken;
    pass.proposalsGPUAddress = proposals.gpuAddress;
    pass.brainAcksGPUAddress = acknowledgements.gpuAddress;
    pass.applyActionsGPUAddress = actions.gpuAddress;
    pass.matterApplyOutcomesGPUAddress = outcomes.gpuAddress;
    pass.proposedPhysicsStateTokensGPUAddress = proposedToken.gpuAddress;
    pass.proposalElementCount = 1u;
    pass.brainAckElementCount = 1u;
    pass.applyActionElementCount = 1u;
    pass.matterApplyOutcomeElementCount = 1u;
    pass.proposedPhysicsStateTokenBytes = sizeof(token);
    pass.proposalStride = 1u;
    pass.brainAckStride = 1u;
    pass.applyActionStride = 1u;
    pass.matterApplyOutcomeStride = 1u;
    pass.proposedPhysicsStateTokenStrideBytes = sizeof(token);
    pass.ownerProgramFingerprint = kAdapterProgramFingerprint;
    pass.transactionFingerprint = kTransactionFingerprint;
    pass.linearizationEpoch = kLinearizationEpoch;
    pass.slotGeneration = kSlotGeneration;

    require(!matter.applyPreparedState(pass),
        "legacy apply admitted the exact prepared family");
    const numi::matter::PreparedStateDispositionIdentity identity{
        .abiVersion = 1u,
        .structSize = sizeof(numi::matter::PreparedStateDispositionIdentity),
        .controlStep = kControlStep,
        .physicsSubstep = 0u,
        .physicsSubstepCount = 1u,
        .transactionSlot = kTransactionSlot,
        .reserved0 = 0u,
        .reserved1 = 0u,
        .ownerProgramFingerprint = kAdapterProgramFingerprint,
        .transactionFingerprint = kTransactionFingerprint,
        .linearizationEpoch = kLinearizationEpoch,
        .slotGeneration = kSlotGeneration,
    };
    require(matter.preparedStateDisposition(identity) ==
            numi::matter::PreparedStateDisposition::prepared,
        "rejected legacy apply changed exact prepared disposition");
}

void requireExactApplyAcceptsExactFamily(
    numi::matter::Runtime& matter,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const MRNumanXAcceptedStateProofGPUV2& proof
) {
    const MRNumanXAcceptedPhysicsStateTokenGPUV2 token =
        makeExactToken(proof);

    NMOwnerProposalGPU proposal{};
    proposal.abiVersion = NM_MATTER_OWNER_APPLY_ABI_VERSION;
    proposal.status = NM_OWNER_PROPOSAL_READY;
    proposal.decision = NM_OWNER_ROOT_ACCEPT;
    proposal.code = NM_OWNER_PROPOSAL_SUCCESS;
    proposal.programFingerprint = kAdapterProgramFingerprint;
    proposal.transactionFingerprint = kTransactionFingerprint;
    proposal.linearizationEpoch = kLinearizationEpoch;
    proposal.slotGeneration = kSlotGeneration;
    proposal.physicsTokenFingerprint = token.tokenFingerprint;
    proposal.brainProgramFingerprint = 0xb2000001u;
    proposal.brainShadowStateFingerprint = 0xb2000002u;
    proposal.brainWitnessFingerprint = 0xb2000003u;
    proposal.candidatePublicationFingerprint = 0xb2000004u;
    proposal.humanIOIdentityFingerprint = 0xb2000005u;
    proposal.environment = 0u;
    proposal.stepIndex = 0u;
    proposal.substepIndex = 0u;
    proposal.transactionSlot = kTransactionSlot;
    proposal.physicsSubstepCount = 1u;
    proposal.controlStep = kControlStep;
    proposal.proposalFingerprint = recordFingerprint(proposal);

    NMOwnerBrainAckGPU ack{};
    ack.abiVersion = NM_MATTER_OWNER_BRAIN_ACK_ABI_VERSION;
    ack.status = NM_OWNER_BRAIN_ACK_ACCEPT;
    ack.decision = NM_OWNER_ROOT_ACCEPT;
    ack.code = NM_OWNER_BRAIN_ACK_SUCCESS;
    ack.programFingerprint = kAdapterProgramFingerprint;
    ack.transactionFingerprint = kTransactionFingerprint;
    ack.linearizationEpoch = kLinearizationEpoch;
    ack.slotGeneration = kSlotGeneration;
    ack.physicsTokenFingerprint = token.tokenFingerprint;
    ack.proposalFingerprint = proposal.proposalFingerprint;
    ack.preflightFingerprint = 0xb2000010u;
    ack.fastGateFingerprint = 0xb2000011u;
    ack.brainWitnessFingerprint = proposal.brainWitnessFingerprint;
    ack.brainProgramFingerprint = proposal.brainProgramFingerprint;
    ack.environment = 0u;
    ack.stepIndex = 0u;
    ack.substepIndex = 0u;
    ack.transactionSlot = kTransactionSlot;
    ack.physicsSubstepCount = 1u;
    ack.controlStep = kControlStep;
    ack.ackFingerprint = recordFingerprint(ack);

    NMOwnerApplyActionGPU action{};
    action.abiVersion = NM_MATTER_OWNER_APPLY_ABI_VERSION;
    action.status = NM_OWNER_APPLY_ACCEPT;
    action.decision = NM_OWNER_ROOT_ACCEPT;
    action.code = NM_OWNER_APPLIED_SUCCESS;
    action.programFingerprint = kAdapterProgramFingerprint;
    action.transactionFingerprint = kTransactionFingerprint;
    action.linearizationEpoch = kLinearizationEpoch;
    action.slotGeneration = kSlotGeneration;
    action.physicsTokenFingerprint = token.tokenFingerprint;
    action.proposalFingerprint = proposal.proposalFingerprint;
    action.ackFingerprint = ack.ackFingerprint;
    action.preflightFingerprint = ack.preflightFingerprint;
    action.fastGateFingerprint = ack.fastGateFingerprint;
    action.brainWitnessFingerprint = ack.brainWitnessFingerprint;
    action.environment = 0u;
    action.stepIndex = 0u;
    action.substepIndex = 0u;
    action.transactionSlot = kTransactionSlot;
    action.physicsSubstepCount = 1u;
    action.controlStep = kControlStep;
    action.actionFingerprint = recordFingerprint(action);

    id<MTLBuffer> proposals = makeBuffer(
        device, proposal, @"V2 exact immutable proposal");
    id<MTLBuffer> acknowledgements = makeBuffer(
        device, ack, @"V2 exact immutable ACK");
    id<MTLBuffer> actions = makeBuffer(
        device, action, @"V2 exact immutable action");
    id<MTLBuffer> outcomes = makeZeroBuffer(
        device, sizeof(NMMatterApplyOutcomeGPU), @"V2 exact Matter outcome");
    id<MTLBuffer> proposedToken = makeBuffer(
        device, token, @"V2 exact immutable token");
    id<MTLCommandBuffer> apply = [queue commandBuffer];
    require(apply != nil, "failed to allocate exact-family apply CB");

    numi::matter::AcceptedStateApplyPassV2 pass;
    pass.environmentCount = 1u;
    pass.environmentIdentifierBase = kEnvironmentIdentifier;
    pass.controlStep = kControlStep;
    pass.physicsSubstep = 0u;
    pass.physicsSubstepCount = 1u;
    pass.transactionSlot = kTransactionSlot;
    pass.commandBuffer = (__bridge void*)apply;
    pass.proposals = (__bridge void*)proposals;
    pass.brainAcks = (__bridge void*)acknowledgements;
    pass.applyActions = (__bridge void*)actions;
    pass.matterApplyOutcomes = (__bridge void*)outcomes;
    pass.proposedPhysicsStateTokens = (__bridge void*)proposedToken;
    pass.proposalsGPUAddress = proposals.gpuAddress;
    pass.brainAcksGPUAddress = acknowledgements.gpuAddress;
    pass.applyActionsGPUAddress = actions.gpuAddress;
    pass.matterApplyOutcomesGPUAddress = outcomes.gpuAddress;
    pass.proposedPhysicsStateTokensGPUAddress = proposedToken.gpuAddress;
    pass.proposalElementCount = 1u;
    pass.brainAckElementCount = 1u;
    pass.applyActionElementCount = 1u;
    pass.matterApplyOutcomeElementCount = 1u;
    pass.proposedPhysicsStateTokenBytes = sizeof(token);
    pass.proposalStride = 1u;
    pass.brainAckStride = 1u;
    pass.applyActionStride = 1u;
    pass.matterApplyOutcomeStride = 1u;
    pass.proposedPhysicsStateTokenStrideBytes = sizeof(token);
    pass.ownerProgramFingerprint = kAdapterProgramFingerprint;
    pass.transactionFingerprint = kTransactionFingerprint;
    pass.linearizationEpoch = kLinearizationEpoch;
    pass.slotGeneration = kSlotGeneration;

    auto wrongFamily = pass;
    wrongFamily.tokenFamily = NM_MATTER_PREPARED_TOKEN_FAMILY_V1;
    require(!matter.applyPreparedStateV2(wrongFamily),
        "V2 apply admitted a legacy family tag");
    auto wrongClock = pass;
    wrongClock.clockQuantumNanoseconds = 2u;
    require(!matter.applyPreparedStateV2(wrongClock),
        "V2 apply admitted a non-exact clock quantum");
    require(matter.applyPreparedStateV2(pass),
        "exact prepared-state application was rejected on the host");

    const numi::matter::PreparedStateDispositionIdentity identity{
        .abiVersion = 1u,
        .structSize = sizeof(numi::matter::PreparedStateDispositionIdentity),
        .controlStep = kControlStep,
        .physicsSubstep = 0u,
        .physicsSubstepCount = 1u,
        .transactionSlot = kTransactionSlot,
        .reserved0 = 0u,
        .reserved1 = 0u,
        .ownerProgramFingerprint = kAdapterProgramFingerprint,
        .transactionFingerprint = kTransactionFingerprint,
        .linearizationEpoch = kLinearizationEpoch,
        .slotGeneration = kSlotGeneration,
    };
    require(matter.preparedStateDisposition(identity) ==
            numi::matter::PreparedStateDisposition::applying,
        "NotEnqueued exact apply did not reserve the prepared generation");
    finish(apply);
    require(matter.preparedStateDisposition(identity) ==
            numi::matter::PreparedStateDisposition::
                acceptedPendingPublication,
        "completed exact apply did not retain pending-publication authority");

    const NMMatterApplyOutcomeGPU outcome =
        value<NMMatterApplyOutcomeGPU>(outcomes);
    require(outcome.status == NM_OWNER_APPLY_ACCEPT &&
            outcome.decision == NM_OWNER_ROOT_ACCEPT &&
            outcome.code == NM_MATTER_APPLY_SUCCESS &&
            outcome.physicsTokenFingerprint == token.tokenFingerprint &&
            outcome.matterProgramFingerprint ==
                matter.acceptedStateProofProgramFingerprintV2() &&
            outcome.outcomeFingerprint == recordFingerprint(outcome),
        "GPU rejected or misidentified the exact accepted-token apply");

    numi::matter::PreparedStatePublicationBinding binding;
    binding.tokenFamily = NM_MATTER_PREPARED_TOKEN_FAMILY_V2;
    binding.physicsTokenFingerprint = token.tokenFingerprint;
    binding.brainProgramFingerprint = proposal.brainProgramFingerprint;
    binding.brainShadowStateFingerprint =
        proposal.brainShadowStateFingerprint;
    binding.brainWitnessFingerprint = proposal.brainWitnessFingerprint;
    binding.matterApplyFingerprint = outcome.outcomeFingerprint;
    binding.appliedDecisionFingerprint = 0xb2000020u;
    binding.jointCommitFingerprint = 0xb2000021u;
    binding.brainGeneration = 23u;
    auto wrongPublicationFamily = binding;
    wrongPublicationFamily.tokenFamily =
        NM_MATTER_PREPARED_TOKEN_FAMILY_V1;
    numi::matter::PreparedStatePublicationReservation rejected{};
    require(!matter.reservePublishedRoot(
                identity, wrongPublicationFamily, rejected),
        "exact prepared generation admitted a legacy publication family");

    numi::matter::PreparedStatePublicationReservation reservation{};
    require(matter.reservePublishedRoot(identity, binding, reservation),
        "exact publication reservation was rejected");
    numi::matter::PreparedStatePublicationFence fence;
    fence.abiVersion = NM_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2;
    fence.structBytes = sizeof(fence);
    fence.status = NM_JOINT_PUBLICATION_COMMITTED;
    fence.environment = 0u;
    fence.controlStep = kControlStep;
    fence.substepIndex = 0u;
    fence.physicsSubstepCount = 1u;
    fence.ownerProgramFingerprint = kAdapterProgramFingerprint;
    fence.transactionFingerprint = kTransactionFingerprint;
    fence.linearizationEpoch = kLinearizationEpoch;
    fence.slotGeneration = kSlotGeneration;
    fence.physicsTokenFingerprint = token.tokenFingerprint;
    fence.brainProgramFingerprint = proposal.brainProgramFingerprint;
    fence.brainShadowStateFingerprint =
        proposal.brainShadowStateFingerprint;
    fence.brainWitnessFingerprint = proposal.brainWitnessFingerprint;
    fence.appliedDecisionFingerprint =
        binding.appliedDecisionFingerprint;
    fence.jointCommitFingerprint = binding.jointCommitFingerprint;
    fence.brainGeneration = binding.brainGeneration;
    fence.fenceFingerprint = recordFingerprint(fence);
    require(matter.releasePublishedRoot(reservation, fence),
        "exact ABI2 committed publication fence was rejected");
    require(matter.preparedStateDisposition(identity) ==
            numi::matter::PreparedStateDisposition::resolved,
        "exact ABI2 publication did not resolve the prepared generation");
}

struct CaseResult {
    NMAcceptedStateProofGPUV2 proof{};
    std::uint64_t stateProofProgramFingerprint = 0u;
    std::array<std::uint32_t, 4u> candidateCalls{};
    bool authorityWasPrivate = false;
    bool legacyApplyRejected = false;
    bool exactApplyAccepted = false;
};

CaseResult runCase(
    id<MTLDevice> device,
    const AuthorityMode authorityMode,
    const bool verifyLegacyApply
) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "failed to create exact-proof command queue");
    auto world = compileAttachedWorld();
    numi::matter::RuntimeConfiguration runtimeConfig;
    runtimeConfig.metallib = NUMI_MATTER_METALLIB;
    runtimeConfig.environmentCount = 1u;
    runtimeConfig.captureEvents = false;
    runtimeConfig.captureDiagnostics = true;
    runtimeConfig.adaptiveTransfer = false;
    runtimeConfig.coupledCandidateCompensatedTranslation = true;
    runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        sizeof(MRMujocoMuscleStateGPU);

    numi::matter::Runtime matter;
    const auto initialized = matter.initialize(world, runtimeConfig);
    require(initialized.encoded,
        "Matter V2 proof Runtime init failed: " + initialized.message);
    const std::uint64_t proofProgram =
        matter.acceptedStateProofProgramFingerprintV2();
    require(proofProgram != 0u,
        "Matter V2 proof-program fingerprint is zero");
    require(proofProgram != matter.acceptedStateProofProgramFingerprint(),
        "V1 and V2 proof-program families share an identity");

    OwnerArenas arenas = makeOwnerArenas(device);
    std::array<float, kQ> candidateQ{};
    candidateQ[6] = 1.0f;
    auto* acceptedRoot =
        static_cast<MRCompensatedRootTranslationGPU*>(
            arenas.rootTranslation.contents);
    *acceptedRoot = mrCompensatedTranslationFromProjection(
        {candidateQ[0], candidateQ[1], candidateQ[2], 0.0f});

    MRBodyStateGPU body{};
    body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    body.linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
    body.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
    body.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    body.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    body.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    std::array<float, 3u * kDofs> jacobian{};
    jacobian[0u * kDofs] = 1.0f;
    jacobian[1u * kDofs + 1u] = 1.0f;
    jacobian[2u * kDofs + 2u] = 1.0f;
    MRInverseMassStatusGPU inverse{};
    inverse.code = MR_INVERSE_MASS_SUCCESS;
    inverse.environment = 0u;
    inverse.articulationIndex = 0u;
    inverse.failingIndex = MR_INVALID_INDEX;
    inverse.bodyCount = kBodies;
    inverse.nq = kQ;
    inverse.nv = kDofs;
    inverse.rhsCount = 1u;

    MatterCandidateService service;
    service.q = makeBuffer(device, candidateQ, @"V2 proof candidate q");
    service.rootTranslation = makeBuffer(
        device, *acceptedRoot, @"V2 proof candidate compensated root");
    service.bodyPositionLow = makeZeroBuffer(
        device, kBodies * sizeof(mr_float4),
        @"V2 proof body-position low");
    service.body = makeBuffer(device, body, @"V2 proof candidate body");
    service.jacobian = makeBuffer(
        device, jacobian, @"V2 proof candidate Jacobian");
    service.inverseStatus = makeBuffer(
        device, inverse, @"V2 proof inverse status");

    MRMetalWorldStatusGPU worldStatus{};
    worldStatus.code = MR_STEP_SUCCESS;
    worldStatus.environment = 0u;
    worldStatus.controlStep = kControlStep;
    worldStatus.successfulSubsteps = 1u;
    worldStatus.failingSubstep = MR_INVALID_INDEX;
    worldStatus.failingIndex = MR_INVALID_INDEX;
    id<MTLBuffer> worldStatuses = makeBuffer(
        device, worldStatus, @"V2 proof world status");
    id<MTLBuffer> reaction = makeZeroBuffer(
        device, kDofs * sizeof(float), @"V2 transient reaction");
    id<MTLBuffer> proofs = makeZeroBuffer(
        device, sizeof(NMAcceptedStateProofGPUV2), @"V2 accepted proof");

    const mrnx_exact_inbound_authority_v2 authority =
        makeAuthority(authorityMode);
    if (authorityMode == AuthorityMode::corruptFingerprint) {
        require(!metalrobo::metalNumanXExactInboundAuthorityV2Valid(authority),
            "corrupt-fingerprint authority remained CPU-valid");
    } else {
        require(metalrobo::metalNumanXExactInboundAuthorityV2Valid(authority),
            "self-consistent exact authority failed CPU validation");
    }
    id<MTLBuffer> authorityStaging = makeBuffer(
        device, authority, @"V2 authority fixture staging");
    constexpr MTLResourceOptions privateOptions =
        MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
    id<MTLBuffer> authorityPrivate = [device
        newBufferWithLength:sizeof(authority)
                   options:privateOptions];
    require(authorityPrivate != nil && authorityPrivate.gpuAddress != 0u &&
            authorityPrivate.storageMode == MTLStorageModePrivate,
        "failed to allocate private exact authority receipt");
    authorityPrivate.label = @"V2 private exact inbound authority";

    const auto makeRequest = [&](id<MTLCommandBuffer> commandBuffer,
                                 const numi::matter::EncodePhase phase) {
        numi::matter::EncodeRequest request;
        request.commandBuffer = (__bridge void*)commandBuffer;
        request.phase = phase;
        request.rigid.q = (__bridge void*)arenas.q;
        request.rigid.v = (__bridge void*)arenas.v;
        request.rigid.currentBodyCount = kBodies;
        request.rigid.currentBodyStride = kBodies;
        request.rigid.qStride = kQ;
        request.rigid.vStride = kDofs;
        request.environmentStatuses = (__bridge void*)worldStatuses;
        request.coupledCandidateContext = &service;
        request.encodeCoupledCandidate = &encodeMatterCandidate;
        request.controlStep = kControlStep;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.seed = 0x51u;
        request.timestepSeconds = matter.timestepSeconds();
        request.enablePreparedState = true;
        return request;
    };

    id<MTLCommandBuffer> prepare = [queue commandBuffer];
    require(prepare != nil, "failed to allocate exact prepare CB");
    id<MTLBlitCommandEncoder> authorityBlit =
        [prepare blitCommandEncoder];
    require(authorityBlit != nil,
        "failed to allocate exact-authority blit encoder");
    [authorityBlit copyFromBuffer:authorityStaging sourceOffset:0u
                         toBuffer:authorityPrivate destinationOffset:0u
                             size:sizeof(authority)];
    [authorityBlit endEncoding];

    service.commandBuffer = prepare;
    const auto pre = makeRequest(
        prepare, numi::matter::EncodePhase::preDynamics);
    const auto preResult = matter.encode(pre);
    require(preResult.encoded,
        "real V2 prepared preDynamics failed: " + preResult.message);
    const auto post = makeRequest(
        prepare, numi::matter::EncodePhase::postCommit);
    const auto prepared = matter.prepareAcceptedState(post);
    require(prepared.encoded,
        "real V2 prepareAcceptedState failed: " + prepared.message);

    id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)matter.statusBuffer();
    numi::matter::AcceptedStateProofPassV2 pass;
    pass.environmentCount = 1u;
    pass.environmentIdentifierBase = kEnvironmentIdentifier;
    pass.commandBuffer = (__bridge void*)prepare;
    pass.rootTranslation = (__bridge void*)arenas.rootTranslation;
    pass.q = (__bridge void*)arenas.q;
    pass.v = (__bridge void*)arenas.v;
    pass.mujocoStates = (__bridge void*)arenas.mujoco;
    pass.matterGeneralizedReaction = (__bridge void*)reaction;
    pass.environmentStatuses = (__bridge void*)worldStatuses;
    pass.matterStatuses = (__bridge void*)matterStatuses;
    pass.acceptedStateProofs = (__bridge void*)proofs;
    pass.inboundAuthority = (__bridge void*)authorityPrivate;
    pass.rootTranslationGPUAddress = arenas.rootTranslation.gpuAddress;
    pass.qGPUAddress = arenas.q.gpuAddress;
    pass.vGPUAddress = arenas.v.gpuAddress;
    pass.mujocoStatesGPUAddress = arenas.mujoco.gpuAddress;
    pass.matterGeneralizedReactionGPUAddress = reaction.gpuAddress;
    pass.environmentStatusesGPUAddress = worldStatuses.gpuAddress;
    pass.matterStatusesGPUAddress = matterStatuses.gpuAddress;
    pass.acceptedStateProofsGPUAddress = proofs.gpuAddress;
    pass.inboundAuthorityGPUAddress = authorityPrivate.gpuAddress;
    pass.rootTranslationElementCount = 1u;
    pass.qElementCount = kQ;
    pass.vElementCount = kDofs;
    pass.mujocoStateCount = 1u;
    pass.matterGeneralizedReactionElementCount = kDofs;
    pass.environmentStatusElementCount = 1u;
    pass.matterStatusElementCount = 1u;
    pass.acceptedStateProofElementCount = 1u;
    pass.inboundAuthorityByteCount = sizeof(authority);
    pass.rootTranslationStride = 1u;
    pass.qStride = kQ;
    pass.vStride = kDofs;
    pass.mujocoStateStride = 1u;
    pass.reactionStride = kDofs;
    pass.environmentStatusStride = 1u;
    pass.matterStatusStride = 1u;
    pass.acceptedStateProofStride = 1u;
    pass.qCoordinateCount = kQ;
    pass.dofCount = kDofs;
    pass.transactionSlot = kTransactionSlot;
    pass.programFingerprint = kAdapterProgramFingerprint;
    pass.stateProofProgramFingerprint = proofProgram;
    pass.transactionFingerprint = kTransactionFingerprint;
    pass.substepFingerprint = kSubstepFingerprint;
    pass.acceptedTimestampNanoseconds = kAcceptedTimestampNanoseconds;
    pass.physicsGeneration = kPhysicsGeneration;
    pass.linearizationEpoch = kLinearizationEpoch;
    pass.slotGeneration = kSlotGeneration;
    pass.matterSourcePhysicsFingerprint =
        matter.sourcePhysicsFingerprint();
    pass.matterDeviceProgramFingerprint =
        matter.deviceProgramFingerprint();
    pass.motorCandidateFingerprint = kMotorCandidateFingerprint;

    require(matter.encodeAcceptedStateProofV2(pass),
        "host rejected the structurally valid V2 proof pass");
    finish(prepare);

    const NMAcceptedStateProofGPUV2 proof =
        value<NMAcceptedStateProofGPUV2>(proofs);
    const MRNumanXAcceptedStateProofGPUV2 exactProof =
        metalroboProof(proof);
    if (authorityMode == AuthorityMode::valid) {
        require(proof.status == NM_ACCEPTED_STATE_PROOF_VALID,
            "valid exact authority did not produce a VALID proof");
        require(proof.environment == kEnvironmentIdentifier,
            "V2 proof lost the global environment identifier");
        require(proof.acceptedTimestampNanoseconds ==
                    kAcceptedTimestampNanoseconds &&
                proof.acceptedTimestampNanoseconds % 1000u != 0u,
            "V2 proof did not preserve the exact non-microsecond timestamp");
        require(proof.inboundAuthorityFingerprint ==
                    authority.inbound_authority_fingerprint &&
                proof.motorCandidateFingerprint ==
                    authority.motor_candidate_fingerprint,
            "V2 proof lost its exact HumanIO authority identity");
        require(metalrobo::metalNumanXExactAcceptedStateProofV2Valid(
                    exactProof),
            "GPU V2 proof failed the MetalRobo CPU exact validator");
        require(proof.physicsStateFingerprint ==
                    metalrobo::metalNumanXExactPhysicsStateV2Fingerprint(
                        exactProof) &&
                proof.proofFingerprint ==
                    metalrobo::
                        metalNumanXExactAcceptedStateProofV2Fingerprint(
                            exactProof),
            "GPU and CPU V2 proof fingerprints diverged");
        if (verifyLegacyApply) {
            requireLegacyApplyRejectsExactFamily(
                matter, device, queue, exactProof);
            requireExactApplyAcceptsExactFamily(
                matter, device, queue, exactProof);
        }
    } else {
        require(proof.status == NM_ACCEPTED_STATE_PROOF_REJECTED,
            "invalid exact authority did not produce a REJECTED proof");
        require(proof.humanStateFingerprint == 0u &&
                proof.matterStateFingerprint == 0u &&
                proof.physicsStateFingerprint == 0u &&
                proof.inboundAuthorityFingerprint == 0u &&
                proof.proofFingerprint == 0u &&
                !metalrobo::metalNumanXExactAcceptedStateProofV2Valid(
                    exactProof),
            "rejected V2 authority leaked accepted proof authority");
    }

    CaseResult result;
    result.proof = proof;
    result.stateProofProgramFingerprint = proofProgram;
    result.candidateCalls = service.calls;
    result.authorityWasPrivate =
        authorityPrivate.storageMode == MTLStorageModePrivate;
    result.legacyApplyRejected = verifyLegacyApply;
    result.exactApplyAccepted = verifyLegacyApply;
    return result;
}

} // namespace matter_exact_proof_fixture

int main() {
    using namespace matter_exact_proof_fixture;
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "exact Matter V2 probe requires Metal");
            const CaseResult positive = runCase(
                device, AuthorityMode::valid, true);
            const CaseResult corruptFingerprint = runCase(
                device, AuthorityMode::corruptFingerprint, false);
            const CaseResult mismatchedIdentity = runCase(
                device, AuthorityMode::mismatchedIdentity, false);
            require(positive.authorityWasPrivate &&
                    corruptFingerprint.authorityWasPrivate &&
                    mismatchedIdentity.authorityWasPrivate,
                "one or more exact receipts were not device-private");
            require(positive.legacyApplyRejected,
                "V1 apply-family rejection was not exercised");
            require(positive.exactApplyAccepted,
                "V2 apply-family acceptance was not exercised");
            require(positive.candidateCalls[0] != 0u &&
                    positive.candidateCalls[1] != 0u &&
                    positive.candidateCalls[2] != 0u &&
                    positive.candidateCalls[3] != 0u,
                "positive case did not execute the full real Matter candidate path");
            std::cout
                << "numanx_matter_exact_accepted_state_proof_probe=pass\n"
                << "device=" << stringValue(device.name) << "\n"
                << "proof_bytes=" << sizeof(NMAcceptedStateProofGPUV2)
                << "\n"
                << "authority_bytes=" << sizeof(NMExactInboundAuthorityGPUV2)
                << "\n"
                << "environment_identifier="
                << positive.proof.environment << "\n"
                << "accepted_brain_timestamp_nanoseconds="
                << kAcceptedBrainTimestampNanoseconds << "\n"
                << "accepted_timestamp_nanoseconds="
                << positive.proof.acceptedTimestampNanoseconds << "\n"
                << "proof_fingerprint="
                << positive.proof.proofFingerprint << "\n"
                << "proof_program_fingerprint="
                << positive.stateProofProgramFingerprint << "\n"
                << "authority_storage=private\n"
                << "negative_cases=2\n"
                << "legacy_v1_apply_exact_family=rejected\n"
                << "exact_v2_apply_exact_family=accepted\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr
                << "numanx_matter_exact_accepted_state_proof_probe=fail\n"
                << "error=" << exception.what() << '\n';
            return 1;
        }
    }
}
