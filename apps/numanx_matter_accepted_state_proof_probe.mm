#define main numanx_human_matter_adapter_probe_legacy_main
#include "numanx_human_matter_adapter_probe.mm"
#undef main

#include "../matter/src/accepted_state_proof_gpu.hpp"
#include "numi/matter/accepted_state_apply_gpu.h"

namespace proof_fixture {

using namespace adapter_fixture;

void mixByte(std::uint64_t& hash, const std::uint8_t byte) {
    hash ^= byte;
    hash *= 1099511628211ull;
}

void mixU32(std::uint64_t& hash, const std::uint32_t value) {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        mixByte(hash, static_cast<std::uint8_t>(value >> (8u * byte)));
    }
}

void mixU64(std::uint64_t& hash, const std::uint64_t value) {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        mixByte(hash, static_cast<std::uint8_t>(value >> (8u * byte)));
    }
}

std::uint64_t sourceTree(
    const std::span<const std::byte> bytes,
    const std::uint32_t source
) {
    constexpr std::uint32_t chunkDomain = 0x4e584348u;
    constexpr std::uint32_t treeDomain = 0x4e585452u;
    constexpr std::size_t chunkBytes =
        numi::matter::detail::kAcceptedStateProofChunkBytes;
    const std::size_t chunkCount = std::max<std::size_t>(
        1u, bytes.size() / chunkBytes + (bytes.size() % chunkBytes != 0u));
    std::vector<std::uint64_t> level(chunkCount);
    for (std::size_t chunk = 0u; chunk < chunkCount; ++chunk) {
        const std::size_t begin = chunk * chunkBytes;
        const std::size_t count = begin < bytes.size()
            ? std::min(chunkBytes, bytes.size() - begin) : 0u;
        std::uint64_t hash = 14695981039346656037ull;
        mixU32(hash, chunkDomain);
        mixU32(hash, source);
        mixU32(hash, static_cast<std::uint32_t>(chunk));
        mixU64(hash, count);
        for (std::size_t index = 0u; index < count; ++index) {
            mixByte(hash, std::to_integer<std::uint8_t>(bytes[begin + index]));
        }
        level[chunk] = hash;
    }
    std::uint32_t reductionLevel = 0u;
    while (level.size() > 1u) {
        std::vector<std::uint64_t> next((level.size() + 1u) / 2u);
        for (std::size_t node = 0u; node < next.size(); ++node) {
            const std::size_t leftIndex = node * 2u;
            const std::size_t rightIndex = leftIndex + 1u;
            const bool hasRight = rightIndex < level.size();
            std::uint64_t hash = 14695981039346656037ull;
            mixU32(hash, treeDomain);
            mixU32(hash, source);
            mixU32(hash, reductionLevel);
            mixU32(hash, hasRight ? 2u : 1u);
            mixU64(hash, level[leftIndex]);
            mixU64(hash, hasRight ? level[rightIndex] : 0u);
            next[node] = hash;
        }
        level = std::move(next);
        ++reductionLevel;
    }
    return level.front();
}

std::span<const std::byte> bytes(id<MTLBuffer> buffer) {
    return {
        static_cast<const std::byte*>(buffer.contents),
        static_cast<std::size_t>(buffer.length),
    };
}

std::uint64_t humanReference(const OwnerArenas& arenas) {
    std::uint64_t hash = 14695981039346656037ull;
    mixU32(hash, 0x4e584855u);
    mixU32(hash, numi::matter::detail::kAcceptedStateProofSchemaVersion);
    const auto fold = [&](const id<MTLBuffer> buffer,
                          const numi::matter::detail::AcceptedStateProofSource source) {
        const auto content = bytes(buffer);
        mixU32(hash, static_cast<std::uint32_t>(source));
        mixU64(hash, content.size());
        mixU64(hash, sourceTree(
            content, static_cast<std::uint32_t>(source)));
    };
    fold(arenas.q, numi::matter::detail::AcceptedStateProofSource::humanQ);
    fold(arenas.v, numi::matter::detail::AcceptedStateProofSource::humanV);
    fold(arenas.mujoco,
         numi::matter::detail::AcceptedStateProofSource::humanMujoco);
    return hash;
}

std::uint64_t proofFingerprint(const NMAcceptedStateProofGPU& proof) {
    std::uint64_t hash = 14695981039346656037ull;
    mixU32(hash, proof.abiVersion);
    mixU32(hash, proof.structSize);
    mixU32(hash, proof.status);
    mixU32(hash, proof.environment);
    mixU64(hash, proof.transactionFingerprint);
    mixU64(hash, proof.substepFingerprint);
    mixU64(hash, proof.acceptedTimestampMicroseconds);
    mixU64(hash, proof.physicsGeneration);
    mixU64(hash, proof.humanStateFingerprint);
    mixU64(hash, proof.matterStateFingerprint);
    mixU64(hash, proof.physicsStateFingerprint);
    mixU64(hash, proof.matterSourcePhysicsFingerprint);
    mixU64(hash, proof.matterDeviceProgramFingerprint);
    mixU64(hash, proof.linearizationEpoch);
    mixU64(hash, proof.slotGeneration);
    mixU64(hash, proof.adapterProgramFingerprint);
    mixU64(hash, proof.transactionPolicyFingerprint);
    return hash;
}

enum class FinalMode {
    applyAccept,
    applyReject,
    applyPending,
    applyAbortAndRetryAccept,
    applyPublicationMismatch,
};

template <typename T>
std::uint64_t recordFingerprint(const T& record) {
    static_assert(sizeof(T) == 128u);
    std::uint64_t hash = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&record);
    for (std::size_t index = 0u; index < 120u; ++index) {
        mixByte(hash, bytes[index]);
    }
    return hash == 0u ? 14695981039346656037ull : hash;
}

std::uint64_t publicationReservationFingerprint(
    const numi::matter::PreparedStatePublicationReservation& reservation
) {
    std::uint64_t hash = 14695981039346656037ull;
    mixU32(hash, reservation.abiVersion);
    mixU32(hash, reservation.structSize);
    mixU32(hash, reservation.transactionSlot);
    mixU32(hash, reservation.reserved0);
    mixU64(hash, reservation.transactionFingerprint);
    mixU64(hash, reservation.slotGeneration);
    mixU64(hash, reservation.reservationNonce);
    mixU64(hash, reservation.reserved1);
    mixU64(hash, reservation.reserved2);
    return hash == 0u ? 14695981039346656037ull : hash;
}

std::uint64_t publicationFactsFingerprint(
    const NMPreparedStatePublicationFactsGPU& facts
) {
    std::uint64_t hash = 14695981039346656037ull;
    mixU32(hash, facts.abiVersion);
    mixU32(hash, facts.status);
    mixU32(hash, facts.reserved0);
    mixU32(hash, facts.reserved1);
    mixU64(hash, facts.physicsTokenFingerprint);
    mixU64(hash, facts.brainProgramFingerprint);
    mixU64(hash, facts.brainShadowStateFingerprint);
    mixU64(hash, facts.brainWitnessFingerprint);
    mixU64(hash, facts.matterApplyFingerprint);
    return hash == 0u ? 14695981039346656037ull : hash;
}

struct ApplicationKernelResult {
    std::uint32_t action = NM_PREPARED_STATE_ACTION_PENDING;
    NMMatterStatusGPU status{};
    NMMatterApplyOutcomeGPU outcome{};
    NMPreparedStatePublicationFactsGPU facts{};
};

ApplicationKernelResult runApplicationKernel(
    id<MTLDevice> device,
    const bool forceRestore
) {
    require(device != nil, "Matter application kernel requires Metal");
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:NUMI_MATTER_METALLIB]]
                   error:&error];
    require(library != nil,
        "failed to load focused Matter metallib: " + errorValue(error));
    id<MTLFunction> function = [library newFunctionWithName:
        @"numi_matter_metal::nm_prepared_state_validate_application"];
    require(function != nil,
        "focused Matter application kernel is missing");
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil,
        "failed to build focused Matter application pipeline: " +
            errorValue(error));
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "failed to allocate focused Matter queue");

    NMPreparedStateApplyGPU pass{};
    pass.abiVersion = NM_MATTER_OWNER_APPLY_ABI_VERSION;
    pass.environmentCount = 1u;
    pass.proposalStride = 1u;
    pass.brainAckStride = 1u;
    pass.applyActionStride = 1u;
    pass.matterApplyOutcomeStride = 1u;
    pass.proposedTokenStrideBytes = 64u;
    pass.stepIndex = 0u;
    pass.substepIndex = 0u;
    pass.transactionSlot = 3u;
    pass.physicsSubstepCount = 1u;
    pass.controlStep = 37u;
    pass.forceRestore = forceRestore ? 1u : 0u;
    pass.ownerProgramFingerprint = 0x4f574e4552505247ull;
    pass.transactionFingerprint = 0x5452414e53414354ull;
    pass.linearizationEpoch = 0x4c494e4541523031ull;
    pass.slotGeneration = 9u;
    pass.matterProgramFingerprint = 0x4d41545445525047ull;

    const NMOwnerProposalGPU proposal{};
    const NMOwnerBrainAckGPU ack{};
    const NMOwnerApplyActionGPU applyAction{};
    const NMPreparedStateBindingGPU binding{};
    const std::array<std::uint8_t, 64u> proposedToken{};
    NMMatterStatusGPU status{};
    status.code = NM_STATUS_SUCCESS;
    status.environment = 0u;

    id<MTLBuffer> passBuffer = makeBuffer(
        device, pass, @"focused Matter apply pass");
    id<MTLBuffer> proposalBuffer = makeBuffer(
        device, proposal, @"focused pending proposal");
    id<MTLBuffer> ackBuffer = makeBuffer(
        device, ack, @"focused pending ACK");
    id<MTLBuffer> applyActionBuffer = makeBuffer(
        device, applyAction, @"focused pending owner action");
    id<MTLBuffer> bindingBuffer = makeBuffer(
        device, binding, @"focused pending proof binding");
    id<MTLBuffer> internalActionBuffer = makeZeroBuffer(
        device, sizeof(std::uint32_t), @"focused internal action");
    id<MTLBuffer> statusBuffer = makeBuffer(
        device, status, @"focused Matter status");
    id<MTLBuffer> outcomeBuffer = makeZeroBuffer(
        device, sizeof(NMMatterApplyOutcomeGPU), @"focused Matter outcome");
    id<MTLBuffer> proposedTokenBuffer = makeBuffer(
        device, proposedToken, @"focused proposed token");
    id<MTLBuffer> factsBuffer = makeZeroBuffer(
        device, sizeof(NMPreparedStatePublicationFactsGPU),
        @"focused publication facts");
    *static_cast<std::uint32_t*>(internalActionBuffer.contents) =
        std::numeric_limits<std::uint32_t>::max();
    std::memset(outcomeBuffer.contents, 0xa5, outcomeBuffer.length);
    std::memset(factsBuffer.contents, 0xa5, factsBuffer.length);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    require(commandBuffer != nil && encoder != nil,
        "failed to allocate focused Matter command encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:passBuffer offset:0u atIndex:0u];
    [encoder setBuffer:proposalBuffer offset:0u atIndex:1u];
    [encoder setBuffer:ackBuffer offset:0u atIndex:2u];
    [encoder setBuffer:applyActionBuffer offset:0u atIndex:3u];
    [encoder setBuffer:bindingBuffer offset:0u atIndex:4u];
    [encoder setBuffer:internalActionBuffer offset:0u atIndex:5u];
    [encoder setBuffer:statusBuffer offset:0u atIndex:6u];
    [encoder setBuffer:outcomeBuffer offset:0u atIndex:7u];
    [encoder setBuffer:proposedTokenBuffer offset:0u atIndex:8u];
    [encoder setBuffer:factsBuffer offset:0u atIndex:9u];
    [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [encoder endEncoding];
    finish(commandBuffer);

    ApplicationKernelResult result{};
    result.action = value<std::uint32_t>(internalActionBuffer);
    result.status = value<NMMatterStatusGPU>(statusBuffer);
    result.outcome = value<NMMatterApplyOutcomeGPU>(outcomeBuffer);
    result.facts = value<NMPreparedStatePublicationFactsGPU>(factsBuffer);
    return result;
}

void verifyApplicationKernelTerminalSemantics(id<MTLDevice> device) {
    const ApplicationKernelResult terminal =
        runApplicationKernel(device, false);
    require(terminal.action ==
                NM_PREPARED_STATE_ACTION_TERMINAL_NO_TOUCH &&
            terminal.status.code == NM_STATUS_SUCCESS &&
            terminal.outcome.status ==
                NM_OWNER_APPLY_TERMINAL_NO_TOUCH &&
            terminal.outcome.decision == NM_OWNER_ROOT_PENDING &&
            terminal.outcome.code == NM_MATTER_APPLY_INVALID_OUTCOME &&
            terminal.outcome.physicsTokenFingerprint == 0u &&
            terminal.outcome.outcomeFingerprint ==
                recordFingerprint(terminal.outcome) &&
            terminal.facts.abiVersion ==
                NM_MATTER_OWNER_APPLY_ABI_VERSION &&
            terminal.facts.status ==
                NM_PREPARED_STATE_PUBLICATION_FACTS_PENDING &&
            terminal.facts.reserved0 == 0u &&
            terminal.facts.reserved1 == 0u &&
            terminal.facts.physicsTokenFingerprint == 0u &&
            terminal.facts.brainProgramFingerprint == 0u &&
            terminal.facts.brainShadowStateFingerprint == 0u &&
            terminal.facts.brainWitnessFingerprint == 0u &&
            terminal.facts.matterApplyFingerprint == 0u &&
            terminal.facts.factsFingerprint ==
                publicationFactsFingerprint(terminal.facts),
        "terminal-no-touch did not preserve PENDING outcome/facts");

    const ApplicationKernelResult restored =
        runApplicationKernel(device, true);
    require(restored.action == NM_PREPARED_STATE_ACTION_RESTORE &&
            restored.status.code == NM_STATUS_RIGID_WORLD_FAILURE &&
            restored.status.failingIndex == NM_MATTER_APPLY_FORCED_REJECT &&
            restored.outcome.status == NM_OWNER_APPLY_REJECT &&
            restored.outcome.decision == NM_OWNER_ROOT_REJECT &&
            restored.outcome.code == NM_MATTER_APPLY_FORCED_REJECT &&
            restored.outcome.physicsTokenFingerprint == 0u &&
            restored.outcome.outcomeFingerprint ==
                recordFingerprint(restored.outcome) &&
            restored.facts.abiVersion ==
                NM_MATTER_OWNER_APPLY_ABI_VERSION &&
            restored.facts.status ==
                NM_PREPARED_STATE_PUBLICATION_FACTS_REJECTED &&
            restored.facts.reserved0 == 0u &&
            restored.facts.reserved1 == 0u &&
            restored.facts.physicsTokenFingerprint == 0u &&
            restored.facts.brainProgramFingerprint == 0u &&
            restored.facts.brainShadowStateFingerprint == 0u &&
            restored.facts.brainWitnessFingerprint == 0u &&
            restored.facts.matterApplyFingerprint == 0u &&
            restored.facts.factsFingerprint ==
                publicationFactsFingerprint(restored.facts),
        "explicit restore did not preserve REJECT outcome/facts");
}

struct MatterCandidateService {
    id<MTLCommandBuffer> commandBuffer = nil;
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
        query.generalizedVectorStride != kDofs) return false;
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
                 query.candidateBodyStride * sizeof(MRBodyStateGPU));
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

std::uint64_t tokenFingerprint(
    const MRNumanXAcceptedPhysicsStateTokenGPU& token
) {
    std::uint64_t hash = 14695981039346656037ull;
    mixU32(hash, 1u);
    mixU64(hash, token.transactionFingerprint);
    mixU64(hash, token.substepFingerprint);
    mixU64(hash, token.physicsStateFingerprint);
    mixU64(hash, token.acceptedTimestampMicroseconds);
    mixU64(hash, token.physicsGeneration);
    mixU32(hash, token.environmentIdentifier);
    mixU32(hash, token.flags);
    mixU64(hash, token.reserved);
    return hash;
}

template <typename T>
bool equalBytes(const std::vector<T>& left, const std::vector<T>& right) {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(
            left.data(), right.data(), left.size() * sizeof(T)) == 0);
}

bool equalAcceptedAuthority(
    const numi::matter::RuntimeStateSnapshot& left,
    const numi::matter::RuntimeStateSnapshot& right
) {
    return left.available && right.available &&
        equalBytes(left.particles, right.particles) &&
        equalBytes(left.femNodes, right.femNodes) &&
        equalBytes(left.femFields, right.femFields) &&
        equalBytes(left.femTopologyNodes, right.femTopologyNodes) &&
        equalBytes(left.femTopologyTetrahedra,
                   right.femTopologyTetrahedra) &&
        equalBytes(left.cohesiveFaces, right.cohesiveFaces) &&
        equalBytes(left.punctureChannels, right.punctureChannels) &&
        equalBytes(left.topologyStates, right.topologyStates) &&
        equalBytes(left.rigidGeneralizedCandidate,
                   right.rigidGeneralizedCandidate) &&
        equalBytes(left.learnedWeights, right.learnedWeights) &&
        left.learnedWeightRevision == right.learnedWeightRevision &&
        equalBytes(left.adaptive, right.adaptive) &&
        equalBytes(left.schedulers, right.schedulers) &&
        equalBytes(left.reactions, right.reactions) &&
        equalBytes(left.rigidStates, right.rigidStates) &&
        equalBytes(left.contactHistories, right.contactHistories) &&
        equalBytes(left.deformableContactHistories,
                   right.deformableContactHistories) &&
        equalBytes(left.particleMaterialState,
                   right.particleMaterialState) &&
        equalBytes(left.femMaterialState, right.femMaterialState) &&
        equalBytes(left.identification, right.identification) &&
        equalBytes(left.environmentParameters,
                   right.environmentParameters);
}

struct ProofResult {
    NMAcceptedStateProofGPU proof{};
    MRNumanXAcceptedPhysicsStateTokenGPU token{};
    std::uint64_t humanReference = 0u;
    std::size_t retainedBytes = 0u;
    std::size_t proofResidentBytes = 0u;
    std::array<std::uint32_t, 4u> candidateCalls{};
    bool authorityRestored = false;
    bool terminalNoTouch = false;
};

ProofResult runProof(
    const FinalMode mode,
    const bool mutateHuman,
    const bool mutateMatter,
    const bool checkUnsupportedFlags = false
) {
    constexpr std::uint32_t environmentIdentifier = 17u;
    constexpr std::uint32_t transactionSlot = 0u;
    constexpr std::uint32_t controlStep = 37u;
    constexpr std::uint64_t adapterOwnerProgram = 0xabc00001u;
    constexpr std::uint64_t transaction = 0xabc001u;
    constexpr std::uint64_t substep = 0xabc002u;
    constexpr std::uint64_t timestamp = 0xabc003u;
    constexpr std::uint64_t generation = 7u;
    constexpr std::uint64_t linearization = 0xabc004u;
    constexpr std::uint64_t slotGeneration = 11u;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "no Metal device is available");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "failed to create proof probe queue");
    auto world = compileAttachedWorld();
    numi::matter::RuntimeConfiguration runtimeConfig;
    runtimeConfig.metallib = NUMI_MATTER_METALLIB;
    runtimeConfig.environmentCount = 1u;
    runtimeConfig.captureEvents = false;
    runtimeConfig.captureDiagnostics = true;
    runtimeConfig.adaptiveTransfer = false;
    runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        sizeof(MRMujocoMuscleStateGPU);
    numi::matter::Runtime matter;
    const auto matterInit = matter.initialize(world, runtimeConfig);
    require(matterInit.encoded,
        "Matter proof Runtime init failed: " + matterInit.message);
    require(matter.acceptedStateProofProgramFingerprint() != 0u,
        "Matter proof program fingerprint is zero");
    if (mutateMatter) {
        auto mutation = matter.snapshot();
        require(mutation.available && mutation.femNodes.size() == 4u,
            "Matter mutation snapshot unavailable");
        mutation.femNodes[1].positionAndMass.x += 2.5e-4f;
        const auto restored = matter.restore(mutation);
        require(restored.encoded,
            "Matter mutation restore failed: " + restored.message);
    }
    const auto before = matter.snapshot();
    require(before.available, "pre-prepare authority snapshot unavailable");

    OwnerArenas arenas = makeOwnerArenas(device);
    std::array<float, kQ> candidateQ{};
    candidateQ[6] = 1.0f;
    if (mutateHuman) {
        static_cast<float*>(arenas.q.contents)[0] = 1.25e-3f;
        candidateQ[0] = 1.25e-3f;
    }
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
    service.q = makeBuffer(device, candidateQ, @"proof candidate q");
    service.body = makeBuffer(device, body, @"proof candidate body");
    service.jacobian = makeBuffer(
        device, jacobian, @"proof candidate Jacobian");
    service.inverseStatus = makeBuffer(
        device, inverse, @"proof inverse status");

    MRMetalWorldStatusGPU worldStatus{};
    worldStatus.code = MR_STEP_SUCCESS;
    worldStatus.environment = 0u;
    worldStatus.controlStep = controlStep;
    worldStatus.successfulSubsteps = 1u;
    worldStatus.failingSubstep = MR_INVALID_INDEX;
    worldStatus.failingIndex = MR_INVALID_INDEX;
    id<MTLBuffer> worldStatuses = makeBuffer(
        device, worldStatus, @"proof world status");
    id<MTLBuffer> reaction = makeZeroBuffer(
        device, kDofs * sizeof(float), @"transient generalized reaction");
    id<MTLBuffer> proofs = makeZeroBuffer(
        device, sizeof(NMAcceptedStateProofGPU), @"accepted state proof");

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
        request.controlStep = controlStep;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.seed = 0x51u;
        request.timestepSeconds = matter.timestepSeconds();
        request.enablePreparedState = true;
        return request;
    };

    if (checkUnsupportedFlags) {
        id<MTLCommandBuffer> unsupported = [queue commandBuffer];
        service.commandBuffer = unsupported;
        auto request = makeRequest(
            unsupported, numi::matter::EncodePhase::preDynamics);
        request.runIdentification = true;
        require(!matter.encode(request).encoded,
            "prepared identification transaction did not fail closed");
        request.runIdentification = false;
        request.resetMaskStepStride = 1u;
        require(!matter.encode(request).encoded,
            "prepared reset transaction did not fail closed");
        request.resetMaskStepStride = 0u;
        request.runAdaptiveTransfer = true;
        require(!matter.encode(request).encoded,
            "prepared adaptive-transfer transaction did not fail closed");
        request.runAdaptiveTransfer = false;
        request.physicsSubsteps = 2u;
        require(!matter.encode(request).encoded,
            "prepared multi-substep transaction did not fail closed");
    }

    id<MTLCommandBuffer> prepare = [queue commandBuffer];
    require(prepare != nil, "failed to create prepare command buffer");
    service.commandBuffer = prepare;
    const auto pre = makeRequest(
        prepare, numi::matter::EncodePhase::preDynamics);
    const auto preResult = matter.encode(pre);
    require(preResult.encoded,
        "real prepared preDynamics failed: " + preResult.message);
    const auto post = makeRequest(
        prepare, numi::matter::EncodePhase::postCommit);
    const auto prepared = matter.prepareAcceptedState(post);
    require(prepared.encoded,
        "real Matter prepareAcceptedState failed: " + prepared.message);

    id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)matter.statusBuffer();
    numi::matter::AcceptedStateProofPass proofPass;
    proofPass.environmentCount = 1u;
    proofPass.environmentIdentifierBase = environmentIdentifier;
    proofPass.commandBuffer = (__bridge void*)prepare;
    proofPass.q = (__bridge void*)arenas.q;
    proofPass.v = (__bridge void*)arenas.v;
    proofPass.mujocoStates = (__bridge void*)arenas.mujoco;
    proofPass.matterGeneralizedReaction = (__bridge void*)reaction;
    proofPass.environmentStatuses = (__bridge void*)worldStatuses;
    proofPass.matterStatuses = (__bridge void*)matterStatuses;
    proofPass.acceptedStateProofs = (__bridge void*)proofs;
    proofPass.qGPUAddress = arenas.q.gpuAddress;
    proofPass.vGPUAddress = arenas.v.gpuAddress;
    proofPass.mujocoStatesGPUAddress = arenas.mujoco.gpuAddress;
    proofPass.matterGeneralizedReactionGPUAddress = reaction.gpuAddress;
    proofPass.environmentStatusesGPUAddress = worldStatuses.gpuAddress;
    proofPass.matterStatusesGPUAddress = matterStatuses.gpuAddress;
    proofPass.acceptedStateProofsGPUAddress = proofs.gpuAddress;
    proofPass.qElementCount = kQ;
    proofPass.vElementCount = kDofs;
    proofPass.mujocoStateCount = 1u;
    proofPass.matterGeneralizedReactionElementCount = kDofs;
    proofPass.environmentStatusElementCount = 1u;
    proofPass.matterStatusElementCount = 1u;
    proofPass.acceptedStateProofElementCount = 1u;
    proofPass.qStride = kQ;
    proofPass.vStride = kDofs;
    proofPass.qCoordinateCount = kQ;
    proofPass.dofCount = kDofs;
    proofPass.mujocoStateStride = 1u;
    proofPass.reactionStride = kDofs;
    proofPass.environmentStatusStride = 1u;
    proofPass.matterStatusStride = 1u;
    proofPass.acceptedStateProofStride = 1u;
    proofPass.transactionSlot = transactionSlot;
    proofPass.programFingerprint = adapterOwnerProgram;
    proofPass.stateProofProgramFingerprint =
        matter.acceptedStateProofProgramFingerprint();
    proofPass.transactionFingerprint = transaction;
    proofPass.substepFingerprint = substep;
    proofPass.acceptedTimestampMicroseconds = timestamp;
    proofPass.physicsGeneration = generation;
    proofPass.linearizationEpoch = linearization;
    proofPass.slotGeneration = slotGeneration;
    proofPass.matterSourcePhysicsFingerprint =
        matter.sourcePhysicsFingerprint();
    proofPass.matterDeviceProgramFingerprint =
        matter.deviceProgramFingerprint();
    if (checkUnsupportedFlags) {
        auto aliasedInput = proofPass;
        aliasedInput.v = aliasedInput.q;
        aliasedInput.vGPUAddress = aliasedInput.qGPUAddress;
        require(!matter.encodeAcceptedStateProof(aliasedInput),
            "overlapping q/v proof ranges were admitted");
        auto aliasedOutput = proofPass;
        aliasedOutput.acceptedStateProofs = aliasedOutput.q;
        aliasedOutput.acceptedStateProofsGPUAddress =
            aliasedOutput.qGPUAddress;
        require(!matter.encodeAcceptedStateProof(aliasedOutput),
            "proof output aliasing a borrowed input was admitted");
        id<MTLBuffer> substitutedWorldStatus = makeBuffer(
            device, worldStatus, @"substituted proof world status");
        auto substitutedStatus = proofPass;
        substitutedStatus.environmentStatuses =
            (__bridge void*)substitutedWorldStatus;
        substitutedStatus.environmentStatusesGPUAddress =
            substitutedWorldStatus.gpuAddress;
        require(!matter.encodeAcceptedStateProof(substitutedStatus),
            "proof admitted a different Human status object than postCommit");
    }
    require(matter.encodeAcceptedStateProof(proofPass),
        "prepared accepted-state proof encoding failed");
    const numi::matter::PreparedStateDispositionIdentity dispositionIdentity{
        .abiVersion = 1u,
        .structSize = sizeof(
            numi::matter::PreparedStateDispositionIdentity),
        .controlStep = controlStep,
        .physicsSubstep = 0u,
        .physicsSubstepCount = 1u,
        .transactionSlot = transactionSlot,
        .reserved0 = 0u,
        .reserved1 = 0u,
        .ownerProgramFingerprint = adapterOwnerProgram,
        .transactionFingerprint = transaction,
        .linearizationEpoch = linearization,
        .slotGeneration = slotGeneration,
    };
    require(matter.preparedStateDisposition(dispositionIdentity) ==
            numi::matter::PreparedStateDisposition::prepared,
        "encoded proof did not expose exact PREPARED disposition");
    auto staleDispositionIdentity = dispositionIdentity;
    staleDispositionIdentity.slotGeneration ^= 1u;
    require(matter.preparedStateDisposition(staleDispositionIdentity) ==
            numi::matter::PreparedStateDisposition::unknown,
        "stale disposition identity did not fail closed");
    finish(prepare);
    require(matter.preparedStateDisposition(dispositionIdentity) ==
            numi::matter::PreparedStateDisposition::prepared,
        "completed prepare command did not retain PREPARED disposition");

    NMAcceptedStateProofGPU proof =
        value<NMAcceptedStateProofGPU>(proofs);
    require(proof.status == NM_ACCEPTED_STATE_PROOF_VALID,
        "prepared GPU content proof was not valid");
    const std::uint64_t serialHuman = humanReference(arenas);

    MRNumanXAcceptedPhysicsStateTokenGPU token{};
    token.transactionFingerprint = proof.transactionFingerprint;
    token.substepFingerprint = proof.substepFingerprint;
    token.physicsStateFingerprint = proof.physicsStateFingerprint;
    token.acceptedTimestampMicroseconds =
        proof.acceptedTimestampMicroseconds;
    token.physicsGeneration = proof.physicsGeneration;
    token.environmentIdentifier = environmentIdentifier;
    token.flags = 0u;
    token.reserved = 0u;
    token.tokenFingerprint = tokenFingerprint(token);
    {
        NMOwnerProposalGPU proposal{};
        proposal.abiVersion = NM_MATTER_OWNER_APPLY_ABI_VERSION;
        proposal.status = NM_OWNER_PROPOSAL_READY;
        proposal.decision = NM_OWNER_ROOT_ACCEPT;
        proposal.code = 0u;
        proposal.programFingerprint = adapterOwnerProgram;
        proposal.transactionFingerprint = transaction;
        proposal.linearizationEpoch = linearization;
        proposal.slotGeneration = slotGeneration;
        proposal.physicsTokenFingerprint = token.tokenFingerprint;
        proposal.brainProgramFingerprint = 0xb001u;
        proposal.brainShadowStateFingerprint = 0xb002u;
        proposal.brainWitnessFingerprint = 0xb003u;
        proposal.candidatePublicationFingerprint = 0xb004u;
        proposal.humanIOIdentityFingerprint = 0xb005u;
        proposal.environment = 0u;
        proposal.stepIndex = 0u;
        proposal.substepIndex = 0u;
        proposal.transactionSlot = transactionSlot;
        proposal.physicsSubstepCount = 1u;
        proposal.controlStep = controlStep;
        proposal.proposalFingerprint = recordFingerprint(proposal);

        NMOwnerBrainAckGPU ack{};
        ack.abiVersion = NM_MATTER_OWNER_BRAIN_ACK_ABI_VERSION;
        ack.status = mode == FinalMode::applyPending
            ? NM_OWNER_BRAIN_ACK_PENDING
            : (mode == FinalMode::applyReject
                ? NM_OWNER_BRAIN_ACK_REJECT : NM_OWNER_BRAIN_ACK_ACCEPT);
        ack.decision = mode == FinalMode::applyPending
            ? NM_OWNER_ROOT_PENDING
            : (mode == FinalMode::applyReject
                ? NM_OWNER_ROOT_REJECT : NM_OWNER_ROOT_ACCEPT);
        ack.code = 0u;
        ack.programFingerprint = adapterOwnerProgram;
        ack.transactionFingerprint = transaction;
        ack.linearizationEpoch = linearization;
        ack.slotGeneration = slotGeneration;
        ack.physicsTokenFingerprint = token.tokenFingerprint;
        ack.proposalFingerprint = proposal.proposalFingerprint;
        ack.preflightFingerprint = 0xa001u;
        ack.fastGateFingerprint = 0xa002u;
        ack.brainWitnessFingerprint = proposal.brainWitnessFingerprint;
        ack.brainProgramFingerprint = proposal.brainProgramFingerprint;
        ack.environment = 0u;
        ack.stepIndex = 0u;
        ack.substepIndex = 0u;
        ack.transactionSlot = transactionSlot;
        ack.physicsSubstepCount = 1u;
        ack.controlStep = controlStep;
        ack.ackFingerprint = recordFingerprint(ack);

        NMOwnerApplyActionGPU action{};
        action.abiVersion = NM_MATTER_OWNER_APPLY_ABI_VERSION;
        action.status = mode == FinalMode::applyPending
            ? NM_OWNER_APPLY_PENDING
            : (mode == FinalMode::applyReject
                ? NM_OWNER_APPLY_REJECT : NM_OWNER_APPLY_ACCEPT);
        action.decision = ack.decision;
        action.code = mode == FinalMode::applyReject
            ? NM_MATTER_APPLY_INVALID_BRAIN_ACK : 0u;
        action.programFingerprint = adapterOwnerProgram;
        action.transactionFingerprint = transaction;
        action.linearizationEpoch = linearization;
        action.slotGeneration = slotGeneration;
        action.physicsTokenFingerprint = token.tokenFingerprint;
        action.proposalFingerprint = proposal.proposalFingerprint;
        action.ackFingerprint = ack.ackFingerprint;
        action.preflightFingerprint = ack.preflightFingerprint;
        action.fastGateFingerprint = ack.fastGateFingerprint;
        action.brainWitnessFingerprint = ack.brainWitnessFingerprint;
        action.environment = 0u;
        action.stepIndex = 0u;
        action.substepIndex = 0u;
        action.transactionSlot = transactionSlot;
        action.physicsSubstepCount = 1u;
        action.controlStep = controlStep;
        action.actionFingerprint = recordFingerprint(action);

        id<MTLBuffer> proposals = makeBuffer(
            device, proposal, @"owner immutable proposal");
        id<MTLBuffer> acknowledgements = makeBuffer(
            device, ack, @"Brain immutable acknowledgement");
        id<MTLBuffer> actions = makeBuffer(
            device, action, @"owner immutable apply action");
        id<MTLBuffer> outcomes = makeZeroBuffer(
            device, sizeof(NMMatterApplyOutcomeGPU), @"Matter apply outcome");
        id<MTLBuffer> proposedTokens = makeBuffer(
            device, token, @"owner immutable proposed token");

        id<MTLCommandBuffer> apply = [queue commandBuffer];
        numi::matter::AcceptedStateApplyPass applyPass;
        applyPass.environmentCount = 1u;
        applyPass.environmentIdentifierBase = environmentIdentifier;
        applyPass.controlStep = controlStep;
        applyPass.physicsSubstep = 0u;
        applyPass.physicsSubstepCount = 1u;
        applyPass.transactionSlot = transactionSlot;
        applyPass.commandBuffer = (__bridge void*)apply;
        applyPass.proposals = (__bridge void*)proposals;
        applyPass.brainAcks = (__bridge void*)acknowledgements;
        applyPass.applyActions = (__bridge void*)actions;
        applyPass.matterApplyOutcomes = (__bridge void*)outcomes;
        applyPass.proposedPhysicsStateTokens =
            (__bridge void*)proposedTokens;
        applyPass.proposalsGPUAddress = proposals.gpuAddress;
        applyPass.brainAcksGPUAddress = acknowledgements.gpuAddress;
        applyPass.applyActionsGPUAddress = actions.gpuAddress;
        applyPass.matterApplyOutcomesGPUAddress = outcomes.gpuAddress;
        applyPass.proposedPhysicsStateTokensGPUAddress =
            proposedTokens.gpuAddress;
        applyPass.proposalElementCount = 1u;
        applyPass.brainAckElementCount = 1u;
        applyPass.applyActionElementCount = 1u;
        applyPass.matterApplyOutcomeElementCount = 1u;
        applyPass.proposedPhysicsStateTokenBytes = sizeof(token);
        applyPass.proposalStride = 1u;
        applyPass.brainAckStride = 1u;
        applyPass.applyActionStride = 1u;
        applyPass.matterApplyOutcomeStride = 1u;
        applyPass.proposedPhysicsStateTokenStrideBytes = sizeof(token);
        applyPass.ownerProgramFingerprint = adapterOwnerProgram;
        applyPass.transactionFingerprint = transaction;
        applyPass.linearizationEpoch = linearization;
        applyPass.slotGeneration = slotGeneration;

        if (checkUnsupportedFlags) {
            auto alias = applyPass;
            alias.matterApplyOutcomes = alias.applyActions;
            alias.matterApplyOutcomesGPUAddress =
                alias.applyActionsGPUAddress;
            require(!matter.applyPreparedState(alias),
                "apply admitted aliased action/outcome ranges");
            auto stale = applyPass;
            stale.slotGeneration ^= 1u;
            require(!matter.applyPreparedState(stale),
                "apply admitted a stale slot generation");
            auto wrongSlot = applyPass;
            wrongSlot.transactionSlot += 1u;
            require(!matter.applyPreparedState(wrongSlot),
                "apply admitted a wrong transaction slot");
        }
        require(matter.applyPreparedState(applyPass),
            "ABI4 prepared-state application was rejected on the host");
        require(matter.preparedStateDisposition(dispositionIdentity) ==
                numi::matter::PreparedStateDisposition::applying,
            "NotEnqueued ABI4 apply did not expose FINALIZING");
        id<MTLCommandBuffer> duplicate = [queue commandBuffer];
        auto duplicatePass = applyPass;
        duplicatePass.commandBuffer = (__bridge void*)duplicate;
        require(!matter.applyPreparedState(duplicatePass),
            "duplicate ABI4 apply reservation was admitted");

        if (mode == FinalMode::applyAbortAndRetryAccept) {
            matter.cancel((__bridge void*)apply);
            require(matter.preparedStateDisposition(dispositionIdentity) ==
                    numi::matter::PreparedStateDisposition::prepared,
                "aborted ABI4 apply did not return to PREPARED");
            id<MTLCommandBuffer> retry = [queue commandBuffer];
            auto retryPass = applyPass;
            retryPass.commandBuffer = (__bridge void*)retry;
            require(matter.applyPreparedState(retryPass),
                "ABI4 apply retry was rejected");
            finish(retry);
        } else {
            finish(apply);
        }

        const auto expectedDisposition = mode == FinalMode::applyPending
            ? numi::matter::PreparedStateDisposition::terminalNoTouch
            : (mode == FinalMode::applyReject
                ? numi::matter::PreparedStateDisposition::resolved
                : numi::matter::PreparedStateDisposition::
                    acceptedPendingPublication);
        require(matter.preparedStateDisposition(dispositionIdentity) ==
                expectedDisposition,
            "completed ABI4 apply exposed the wrong disposition");
        const NMMatterApplyOutcomeGPU matterOutcome =
            value<NMMatterApplyOutcomeGPU>(outcomes);
        require(matterOutcome.outcomeFingerprint ==
                    recordFingerprint(matterOutcome) &&
                matterOutcome.programFingerprint == adapterOwnerProgram &&
                matterOutcome.transactionFingerprint == transaction &&
                matterOutcome.slotGeneration == slotGeneration,
            "Matter apply outcome identity/FNV is invalid");
        require(std::memcmp(proposedTokens.contents, &token, sizeof(token)) == 0,
            "Matter mutated the immutable proposed token");

        if (mode != FinalMode::applyReject &&
            mode != FinalMode::applyPending) {
            require(!matter.snapshot().available,
                "accepted-pending Matter state leaked through snapshot");
            id<MTLCommandBuffer> forbiddenEncode = [queue commandBuffer];
            service.commandBuffer = forbiddenEncode;
            auto blockedRequest = makeRequest(
                forbiddenEncode, numi::matter::EncodePhase::preDynamics);
            blockedRequest.controlStep = controlStep + 1u;
            require(!matter.encode(blockedRequest).encoded,
                "accepted-pending Matter root admitted a new encode");
            numi::matter::PreparedStatePublicationBinding binding;
            binding.physicsTokenFingerprint = token.tokenFingerprint;
            binding.brainProgramFingerprint =
                proposal.brainProgramFingerprint;
            binding.brainShadowStateFingerprint =
                proposal.brainShadowStateFingerprint;
            binding.brainWitnessFingerprint =
                proposal.brainWitnessFingerprint;
            binding.matterApplyFingerprint =
                matterOutcome.outcomeFingerprint;
            binding.appliedDecisionFingerprint = 0xd001u;
            binding.jointCommitFingerprint = 0xd002u;
            binding.brainGeneration = 19u;
            auto staleIdentity = dispositionIdentity;
            staleIdentity.slotGeneration ^= 1u;
            numi::matter::PreparedStatePublicationReservation reservation;
            numi::matter::PreparedStatePublicationReservation rejected{};
            require(!matter.reservePublishedRoot(
                        staleIdentity, binding, rejected),
                "stale publication reservation was admitted");
            auto fabricatedBinding = binding;
            fabricatedBinding.physicsTokenFingerprint ^= 1u;
            require(!matter.reservePublishedRoot(
                        dispositionIdentity, fabricatedBinding, rejected),
                "caller-fabricated publication facts were admitted");
            fabricatedBinding = binding;
            fabricatedBinding.matterApplyFingerprint ^= 1u;
            require(!matter.reservePublishedRoot(
                        dispositionIdentity, fabricatedBinding, rejected),
                "stale Matter outcome publication binding was admitted");
            require(matter.reservePublishedRoot(
                        dispositionIdentity, binding, reservation),
                "exact publication reservation was rejected");
            require(!matter.reservePublishedRoot(
                        dispositionIdentity, binding, rejected),
                "duplicate publication reservation was admitted");

            numi::matter::PreparedStatePublicationFence fence;
            fence.abiVersion = NM_MATTER_PUBLICATION_FENCE_ABI_VERSION;
            fence.structBytes = sizeof(fence);
            fence.status = NM_JOINT_PUBLICATION_COMMITTED;
            fence.environment = 0u;
            fence.controlStep = controlStep;
            fence.substepIndex = 0u;
            fence.physicsSubstepCount = 1u;
            fence.ownerProgramFingerprint = adapterOwnerProgram;
            fence.transactionFingerprint = transaction;
            fence.linearizationEpoch = linearization;
            fence.slotGeneration = slotGeneration;
            fence.physicsTokenFingerprint =
                binding.physicsTokenFingerprint;
            fence.brainProgramFingerprint =
                binding.brainProgramFingerprint;
            fence.brainShadowStateFingerprint =
                binding.brainShadowStateFingerprint;
            fence.brainWitnessFingerprint =
                binding.brainWitnessFingerprint;
            fence.appliedDecisionFingerprint =
                binding.appliedDecisionFingerprint;
            fence.jointCommitFingerprint =
                binding.jointCommitFingerprint;
            fence.brainGeneration = binding.brainGeneration;
            if (mode == FinalMode::applyPublicationMismatch) {
                // A coherent/FNV-valid but caller-fabricated commit plan must
                // not release Matter's accepted root.
                fence.jointCommitFingerprint ^= 1u;
            }
            fence.fenceFingerprint = recordFingerprint(fence);
            auto staleReservation = reservation;
            staleReservation.slotGeneration ^= 1u;
            staleReservation.reservationFingerprint =
                publicationReservationFingerprint(staleReservation);
            require(!matter.releasePublishedRoot(staleReservation, fence),
                "stale publication capability was admitted");
            if (mode == FinalMode::applyPublicationMismatch) {
                require(!matter.releasePublishedRoot(
                            reservation, fence),
                    "fabricated publication plan released Matter root");
                require(matter.preparedStateDisposition(dispositionIdentity) ==
                        numi::matter::PreparedStateDisposition::terminalNoTouch,
                    "publication mismatch did not terminally quarantine");
            } else {
                require(matter.releasePublishedRoot(
                            reservation, fence),
                    "exact COMMITTED publication fence was rejected");
                require(matter.preparedStateDisposition(dispositionIdentity) ==
                        numi::matter::PreparedStateDisposition::resolved,
                    "published root did not resolve Matter quarantine");
                require(!matter.releasePublishedRoot(
                            reservation, fence),
                    "publication release was not one shot");
            }
        }

        ProofResult result;
        result.proof = value<NMAcceptedStateProofGPU>(proofs);
        result.token = (mode == FinalMode::applyReject ||
                        mode == FinalMode::applyPending)
            ? MRNumanXAcceptedPhysicsStateTokenGPU{} : token;
        result.humanReference = serialHuman;
        result.retainedBytes = matterInit.residentBytes;
        result.proofResidentBytes = matter.acceptedStateProofResidentBytes();
        result.candidateCalls = service.calls;
        const auto after = matter.snapshot();
        result.authorityRestored = equalAcceptedAuthority(before, after);
        result.terminalNoTouch =
            (mode == FinalMode::applyPending ||
             mode == FinalMode::applyPublicationMismatch) &&
            !after.available;
        return result;
    }
}

void requireZeroToken(const MRNumanXAcceptedPhysicsStateTokenGPU& token) {
    const MRNumanXAcceptedPhysicsStateTokenGPU zero{};
    require(std::memcmp(&token, &zero, sizeof(token)) == 0,
        "rejected apply retained a nonzero accepted token");
}

} // namespace proof_fixture

int main() {
    using proof_fixture::FinalMode;
    using proof_fixture::ProofResult;
    using proof_fixture::proofFingerprint;
    using proof_fixture::requireZeroToken;
    using proof_fixture::runProof;
    using proof_fixture::tokenFingerprint;
    using proof_fixture::verifyApplicationKernelTerminalSemantics;
    using adapter_fixture::require;
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "accepted-state proof probe requires Metal");
            verifyApplicationKernelTerminalSemantics(device);
            const ProofResult baseline = runProof(
                FinalMode::applyAccept, false, false, true);
            const ProofResult replay = runProof(
                FinalMode::applyAccept, false, false);
            const ProofResult humanMutation = runProof(
                FinalMode::applyAccept, true, false);
            const ProofResult matterMutation = runProof(
                FinalMode::applyAccept, false, true);
            const ProofResult rejected = runProof(
                FinalMode::applyReject, false, false);
            const ProofResult pending = runProof(
                FinalMode::applyPending, false, false);
            const ProofResult appliedRetried = runProof(
                FinalMode::applyAbortAndRetryAccept, false, false);
            const ProofResult publicationMismatch = runProof(
                FinalMode::applyPublicationMismatch, false, false);

            require(baseline.proof.status == NM_ACCEPTED_STATE_PROOF_VALID &&
                    baseline.token.physicsStateFingerprint ==
                        baseline.proof.physicsStateFingerprint &&
                    baseline.token.reserved == 0u &&
                    baseline.token.tokenFingerprint ==
                        tokenFingerprint(baseline.token),
                "valid prepared proof/token relation failed");
            require(baseline.proof.humanStateFingerprint ==
                    baseline.humanReference,
                "GPU chunk/tree Human hash differs from serial CPU reference");
            require(baseline.proof.proofFingerprint ==
                    proofFingerprint(baseline.proof),
                "Matter proof ABI4 FNV/layout parity failed");
            require(std::memcmp(&baseline.proof, &replay.proof,
                                sizeof(baseline.proof)) == 0 &&
                    std::memcmp(&baseline.token, &replay.token,
                                sizeof(baseline.token)) == 0,
                "identical fresh Runtime replay was not byte deterministic");
            require(humanMutation.proof.humanStateFingerprint !=
                    baseline.proof.humanStateFingerprint,
                "Human q mutation did not change Human content hash");
            require(matterMutation.proof.humanStateFingerprint ==
                    baseline.proof.humanStateFingerprint &&
                    matterMutation.proof.matterStateFingerprint !=
                        baseline.proof.matterStateFingerprint,
                "Matter accepted-state mutation did not change Matter hash");
            requireZeroToken(rejected.token);
            require(rejected.authorityRestored,
                "ABI4 REJECT did not restore accepted authority");
            requireZeroToken(pending.token);
            require(pending.terminalNoTouch && !pending.authorityRestored,
                "ABI4 PENDING did not retain terminal no-touch quarantine");
            require(baseline.token.tokenFingerprint != 0u &&
                    !baseline.authorityRestored,
                "ABI4 ACCEPT publication lifecycle did not preserve accepted authority");
            require(appliedRetried.token.tokenFingerprint ==
                    baseline.token.tokenFingerprint,
                "aborted ABI4 apply did not preserve exact ACCEPT retry");
            require(publicationMismatch.terminalNoTouch &&
                    !publicationMismatch.authorityRestored,
                "fabricated COMMITTED fence did not quarantine Matter root");
            for (const std::uint32_t calls : baseline.candidateCalls) {
                require(calls != 0u,
                    "real prepared Runtime skipped a coupled operation");
            }

            std::cout
                << "numanx_matter_accepted_state_proof_probe=pass\n"
                << "device=Apple_Metal\n"
                << "real_runtime_prepare_proof_apply_publication=pass\n"
                << "accepted_rigid_and_matter_content_mutation=pass\n"
                << "human_content_mutation=pass\n"
                << "chunk_tree_cpu_parity=pass\n"
                << "byte_replay=pass\n"
                << "borrowed_interval_aliases=fail_closed\n"
                << "pending=terminal_no_touch\n"
                << "stale_slot_generation=fail_closed\n"
                << "owner_abi4_global_control_step=bound\n"
                << "prepared_disposition_identity=pass\n"
                << "abi4_proposal_ack_apply_accept_reject=pass\n"
                << "terminal_outcome_and_facts=pending_no_touch\n"
                << "explicit_restore_outcome_and_facts=rejected\n"
                << "accepted_pending_publication=blocked_until_exact_fence\n"
                << "accepted_pending_snapshot_and_encode=blocked\n"
                << "publication_facts_fnv_binding=pass\n"
                << "fabricated_publication_fence=terminal_no_touch\n"
                << "aborted_uncommitted_apply_retry=pass\n"
                << "completed_slot_reuse=pass\n"
                << "identification_reset_adaptive=unsupported_fail_closed\n"
                << "multi_substep=unsupported_fail_closed\n"
                << "runtime_resident_bytes=" << baseline.retainedBytes << '\n'
                << "proof_resident_bytes=" << baseline.proofResidentBytes
                << '\n';
            return 0;
        } catch (const std::exception& exception) {
            std::cerr
                << "numanx_matter_accepted_state_proof_probe=fail\n"
                << "reason=" << exception.what() << '\n';
            return 1;
        }
    }
}
