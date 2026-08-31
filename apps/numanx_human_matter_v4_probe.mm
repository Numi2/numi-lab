#define main numanx_human_matter_candidate_probe_embedded_main
#include "numanx_human_matter_candidate_probe.mm"
#undef main

#include "metalrobo/MetalNumanXHumanIO.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace {

constexpr std::uint32_t kV4ControlStep = 37u;
constexpr std::uint64_t kBrainProgramFingerprint = 0x425241494e505247ull;
constexpr std::uint64_t kBrainShadowFingerprint = 0x425241494e534844ull;
constexpr std::uint64_t kFastGateFingerprint = 0x4641535447415445ull;
constexpr std::uint64_t kFastProgramFingerprint = 0x4641535450524f47ull;
constexpr std::uint64_t kMatterProgramFingerprint = 0x4d41545445525047ull;
constexpr std::uint64_t kJointCommitFingerprint = 0x4a4f494e54434d54ull;
constexpr std::uint64_t kBrainGeneration = 23u;
constexpr std::uint64_t kHumanIOCandidateKeyFingerprint =
    0x48494f43414e444bull;
constexpr std::uint64_t kHumanIOProgramFingerprint =
    0x48494f50524f4752ull;
constexpr std::uint64_t kHumanIOSensorFingerprint =
    0x48494f53454e534full;
constexpr std::uint64_t kHumanIOTransactionInstanceFingerprint =
    0x48494f5458494e53ull;
constexpr std::uint64_t kHumanIOCandidatePublicationFingerprint =
    0x48494f5055424341ull;
constexpr std::uint64_t kHumanIOSensorGeneration = 19u;

enum class V4Scenario {
    accept,
    physicalReject,
    forcedReject,
    physicalRejectForcedProposal,
    brainReject,
    invalidBrainWitness,
    tokenMismatch,
    staleAck,
    malformedPublicationRelease,
    proposalAbortRetry,
    proposalCallbackDrop,
};

bool isPhysicalRejectScenario(const V4Scenario scenario) noexcept {
    return scenario == V4Scenario::physicalReject ||
        scenario == V4Scenario::physicalRejectForcedProposal;
}

bool usesForcedProposal(const V4Scenario scenario) noexcept {
    return scenario == V4Scenario::forcedReject ||
        scenario == V4Scenario::physicalRejectForcedProposal;
}

std::uint64_t fnvByte(
    const std::uint64_t hash,
    const std::uint8_t value
) noexcept {
    return (hash ^ value) * 1099511628211ull;
}

std::uint64_t fnvU32(
    std::uint64_t hash,
    const std::uint32_t value
) noexcept {
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
        hash = fnvByte(
            hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

std::uint64_t fnvU64(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
        hash = fnvByte(
            hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

std::uint64_t nonzero(const std::uint64_t value) noexcept {
    return value == 0u ? 14695981039346656037ull : value;
}

std::uint64_t humanIOBindingFingerprint(
    const metalrobo::MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept {
    return metalrobo::metalNumanXHumanIOPublicationBindingFingerprint(
        binding);
}

template <typename Record>
std::uint64_t recordFingerprint(const Record& record) noexcept {
    static_assert(sizeof(Record) == 128u);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&record);
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t index = 0u; index < 120u; ++index) {
        hash = fnvByte(hash, bytes[index]);
    }
    return nonzero(hash);
}

std::uint64_t witnessFingerprint(
    const MRNumanXHumanMatterBrainCommitWitnessGPU& witness
) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = fnvU32(hash, witness.magic);
    hash = fnvU32(hash, witness.abiVersion);
    hash = fnvU32(hash, witness.structBytes);
    hash = fnvU32(hash, witness.status);
    hash = fnvU32(hash, witness.decision);
    hash = fnvU32(hash, witness.environment);
    hash = fnvU32(hash, witness.stepIndex);
    hash = fnvU32(hash, witness.substepIndex);
    hash = fnvU32(hash, witness.transactionSlot);
    hash = fnvU32(hash, witness.physicsSubstepCount);
    hash = fnvU32(hash, witness.controlStep);
    hash = fnvU32(hash, witness.reserved0);
    hash = fnvU64(hash, witness.programFingerprint);
    hash = fnvU64(hash, witness.transactionFingerprint);
    hash = fnvU64(hash, witness.linearizationEpoch);
    hash = fnvU64(hash, witness.slotGeneration);
    hash = fnvU64(hash, witness.physicsTokenFingerprint);
    hash = fnvU64(hash, witness.brainProgramFingerprint);
    hash = fnvU64(hash, witness.brainShadowStateFingerprint);
    hash = fnvU64(hash, witness.reserved1[0]);
    hash = fnvU64(hash, witness.reserved1[1]);
    return nonzero(hash);
}

std::uint64_t canonicalTokenFingerprint(
    const std::uint64_t* words
) noexcept {
    std::uint64_t hash = fnvU32(14695981039346656037ull, 1u);
    for (std::uint32_t index = 0u; index < 7u; ++index) {
        hash = fnvU64(hash, words[index]);
    }
    return nonzero(hash);
}

void initializeCanonicalToken(id<MTLBuffer> token) {
    auto* words = contents<std::uint64_t>(token);
    std::memset(words, 0, MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
    words[0] = kTransactionFingerprint;
    words[1] = 0x5355425354455046ull;
    words[2] = 0x5048595353544154ull;
    words[3] = 1u;
    words[4] = 1u;
    words[5] = 0u;
    words[6] = 0u;
    words[7] = canonicalTokenFingerprint(words);
}

bool sameLease(
    const metalrobo::MetalNumanXHumanMatterPrepareLease& left,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& right
) noexcept {
    return left.abiVersion == right.abiVersion &&
        left.structSize == right.structSize &&
        left.environmentCount == right.environmentCount &&
        left.transactionSlot == right.transactionSlot &&
        left.stepIndex == right.stepIndex &&
        left.substepIndex == right.substepIndex &&
        left.physicsSubstepCount == right.physicsSubstepCount &&
        left.controlStep == right.controlStep &&
        left.qCoordinateCount == right.qCoordinateCount &&
        left.dofCount == right.dofCount &&
        left.dofLayoutVersion == right.dofLayoutVersion &&
        left.reservedDofLayout == right.reservedDofLayout &&
        left.preparedPhysicsStateTokens == right.preparedPhysicsStateTokens &&
        left.proposals == right.proposals &&
        left.proposedPhysicsStateTokens ==
            right.proposedPhysicsStateTokens &&
        left.applyActions == right.applyActions &&
        left.matterApplyOutcomes == right.matterApplyOutcomes &&
        left.appliedOutcomes == right.appliedOutcomes &&
        left.finalAcceptedPhysicsStateTokens ==
            right.finalAcceptedPhysicsStateTokens &&
        left.publicationFences == right.publicationFences &&
        left.physicalPreparedEvent == right.physicalPreparedEvent &&
        left.physicalPreparedEventValue == right.physicalPreparedEventValue &&
        left.proposalEventValue == right.proposalEventValue &&
        left.appliedEventValue == right.appliedEventValue &&
        left.programFingerprint == right.programFingerprint &&
        left.transactionFingerprint == right.transactionFingerprint &&
        left.linearizationEpoch == right.linearizationEpoch &&
        left.slotGeneration == right.slotGeneration;
}

struct V4Audit {
    CandidateAudit candidate{};
    id<MTLDevice> device = nil;
    id<MTLBuffer> reactionSource = nil;
    id<MTLBuffer> matterOutcomeSource = nil;
    id<MTLBuffer> matterOutcomeTarget = nil;

    __unsafe_unretained id<MTLBuffer> liveQ = nil;
    __unsafe_unretained id<MTLBuffer> liveV = nil;
    __unsafe_unretained id<MTLBuffer> liveMujoco = nil;
    __unsafe_unretained id<MTLBuffer> checkpointQ = nil;
    __unsafe_unretained id<MTLBuffer> checkpointV = nil;
    __unsafe_unretained id<MTLBuffer> checkpointMujoco = nil;
    __unsafe_unretained id<MTLBuffer> ownerStatuses = nil;

    metalrobo::MetalNumanXHumanMatterPrepareLease lease{};
    std::atomic<bool> leaseActive{false};
    std::atomic<std::uint32_t> releaseCount{0u};
    std::atomic<std::uint32_t> applyCompletionCount{0u};
    std::atomic<std::uint32_t> applyTerminalStatus{0u};
    std::atomic<std::uint32_t> proposalCompletionCount{0u};
    std::atomic<std::uint32_t> proposalCompletionStatus{0u};
    std::atomic<bool> proposalCompletionReentrySafe{false};
    std::atomic<std::uint32_t> physicalCompletionCount{0u};
    std::atomic<std::uint32_t> physicalCompletionStatus{0u};
    std::atomic<bool> physicalCompletionReentrySafe{false};
    metalrobo::MetalNumanXHumanMatterPrepared* completionPrepared = nullptr;
    bool dropPreparedInProposalCompletion = false;
    std::uint32_t phaseCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t reserveApplicationCount = 0u;
    std::uint32_t encodeApplyCount = 0u;
    std::uint32_t abortApplyCount = 0u;
    std::uint32_t reservePublicationCount = 0u;
    std::uint32_t releasePublicationCount = 0u;
    std::uint32_t bindHumanIOCount = 0u;
    std::uint32_t reserveHumanIOCount = 0u;
    std::uint32_t publishHumanIOCount = 0u;
    std::uint32_t rejectHumanIOCount = 0u;
    metalrobo::MetalNumanXHumanIOCandidatePublicationBinding
        humanIOBinding{};
    bool humanIOPublicationReserved = false;
    const char* failure = nullptr;

    bool fail(const char* message) noexcept {
        if (failure == nullptr) failure = message;
        return false;
    }
};

bool sameHumanIOProgram(
    const metalrobo::MetalNumanXHumanIOCandidatePublicationProgram& left,
    const metalrobo::MetalNumanXHumanIOCandidatePublicationProgram& right
) noexcept {
    return left.abiVersion == right.abiVersion &&
        left.structSize == right.structSize && left.context == right.context &&
        left.reservePublishedRoot == right.reservePublishedRoot &&
        left.publishCandidate == right.publishCandidate &&
        left.rejectCandidate == right.rejectCandidate &&
        left.candidateKeyFingerprint == right.candidateKeyFingerprint &&
        left.transactionFingerprint == right.transactionFingerprint &&
        left.acceptedBrainGeneration == right.acceptedBrainGeneration &&
        left.sensorGeneration == right.sensorGeneration &&
        left.humanIOProgramFingerprint == right.humanIOProgramFingerprint &&
        left.sensorFingerprint == right.sensorFingerprint &&
        left.transactionInstanceFingerprint ==
            right.transactionInstanceFingerprint &&
        left.candidatePublicationFingerprint ==
            right.candidatePublicationFingerprint &&
        left.deviceRegistryID == right.deviceRegistryID &&
        left.identityFingerprint == right.identityFingerprint;
}

bool reserveHumanIOPublication(
    void* raw,
    const std::uint64_t candidateFingerprint,
    const metalrobo::MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    const auto& program = audit.lease.humanIOCandidate;
    if (!program.valid()) {
        return audit.fail("HumanIO publication program invalid");
    }
    if (audit.humanIOPublicationReserved) {
        return audit.fail("HumanIO publication already reserved");
    }
    if (candidateFingerprint != program.candidatePublicationFingerprint) {
        return audit.fail("HumanIO publication candidate fingerprint mismatch");
    }
    if (
        binding.abiVersion !=
            metalrobo::kMetalNumanXHumanIOPublicationABIVersion ||
        binding.structSize != sizeof(binding) ||
        binding.environmentCount != 1u || binding.transactionSlot != 0u ||
        binding.stepIndex != 0u || binding.substepIndex != 0u ||
        binding.physicsSubstepCount != 1u ||
        binding.controlStep != kV4ControlStep ||
        binding.ownerProgramFingerprint != kProgramFingerprint ||
        binding.transactionFingerprint != kTransactionFingerprint ||
        binding.linearizationEpoch != kLinearizationEpoch ||
        binding.slotGeneration != kSlotGeneration) {
        return audit.fail("HumanIO publication owner identity mismatch");
    }
    if (binding.physicsTokenFingerprint == 0u ||
        binding.proposalFingerprint == 0u || binding.ackFingerprint == 0u ||
        binding.appliedDecisionFingerprint == 0u ||
        binding.jointCommitFingerprint != kJointCommitFingerprint ||
        binding.brainGeneration != kBrainGeneration) {
        return audit.fail("HumanIO publication joint decision mismatch");
    }
    if (
        binding.candidateKeyFingerprint !=
            program.candidateKeyFingerprint ||
        binding.acceptedBrainGeneration !=
            program.acceptedBrainGeneration ||
        binding.sensorGeneration != program.sensorGeneration ||
        binding.humanIOProgramFingerprint !=
            program.humanIOProgramFingerprint ||
        binding.sensorFingerprint != program.sensorFingerprint ||
        binding.transactionInstanceFingerprint !=
            program.transactionInstanceFingerprint ||
        binding.candidatePublicationFingerprint !=
            program.candidatePublicationFingerprint ||
        binding.deviceRegistryID != program.deviceRegistryID ||
        binding.humanIOIdentityFingerprint != program.identityFingerprint) {
        return audit.fail("HumanIO publication candidate identity mismatch");
    }
    if (binding.bindingFingerprint != humanIOBindingFingerprint(binding)) {
        return audit.fail("HumanIO publication binding fingerprint mismatch");
    }
    audit.humanIOBinding = binding;
    audit.humanIOPublicationReserved = true;
    ++audit.reserveHumanIOCount;
    return true;
}

metalrobo::MetalNumanXHumanIOCandidatePublicationDisposition
publishHumanIOCandidate(
    void* raw,
    const std::uint64_t candidateFingerprint,
    const metalrobo::MetalNumanXHumanIOCandidatePublicationCommit& commit
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (!audit.humanIOPublicationReserved ||
        candidateFingerprint != kHumanIOCandidatePublicationFingerprint ||
        commit.abiVersion !=
            metalrobo::kMetalNumanXHumanIOPublicationABIVersion ||
        commit.structSize != sizeof(commit) ||
        commit.status != metalrobo::
            MetalNumanXHumanIOCandidatePublicationCommitStatus::committed ||
        commit.reserved0 != 0u ||
        commit.candidatePublicationFingerprint != candidateFingerprint ||
        commit.bindingFingerprint != audit.humanIOBinding.bindingFingerprint ||
        commit.jointCommitFingerprint != kJointCommitFingerprint ||
        commit.brainGeneration != kBrainGeneration ||
        commit.fenceFingerprint == 0u) {
        return metalrobo::
            MetalNumanXHumanIOCandidatePublicationDisposition::terminalNoTouch;
    }
    ++audit.publishHumanIOCount;
    audit.humanIOPublicationReserved = false;
    return metalrobo::
        MetalNumanXHumanIOCandidatePublicationDisposition::released;
}

metalrobo::MetalNumanXHumanIOCandidatePublicationDisposition
rejectHumanIOCandidate(
    void* raw,
    const std::uint64_t candidateFingerprint
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (candidateFingerprint != kHumanIOCandidatePublicationFingerprint ||
        !audit.lease.humanIOCandidate.valid() ||
        audit.humanIOPublicationReserved) {
        return metalrobo::
            MetalNumanXHumanIOCandidatePublicationDisposition::terminalNoTouch;
    }
    ++audit.rejectHumanIOCount;
    return metalrobo::
        MetalNumanXHumanIOCandidatePublicationDisposition::rejected;
}

metalrobo::MetalNumanXHumanIOCandidatePublicationProgram
makeHumanIOCandidateProgram(V4Audit& audit) noexcept {
    metalrobo::MetalNumanXHumanIOCandidatePublicationProgram program{};
    program.context = &audit;
    program.reservePublishedRoot = &reserveHumanIOPublication;
    program.publishCandidate = &publishHumanIOCandidate;
    program.rejectCandidate = &rejectHumanIOCandidate;
    program.candidateKeyFingerprint = kHumanIOCandidateKeyFingerprint;
    program.transactionFingerprint = kTransactionFingerprint;
    program.acceptedBrainGeneration = kBrainGeneration;
    program.sensorGeneration = kHumanIOSensorGeneration;
    program.humanIOProgramFingerprint = kHumanIOProgramFingerprint;
    program.sensorFingerprint = kHumanIOSensorFingerprint;
    program.transactionInstanceFingerprint =
        kHumanIOTransactionInstanceFingerprint;
    program.candidatePublicationFingerprint =
        kHumanIOCandidatePublicationFingerprint;
    program.deviceRegistryID = audit.device.registryID;
    program.identityFingerprint = program.computedIdentityFingerprint();
    return program;
}

bool bindHumanIOCandidate(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease,
    const metalrobo::MetalNumanXHumanIOCandidatePublicationProgram& candidate
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (!audit.leaseActive.load() ||
        audit.lease.humanIOCandidate.configured() ||
        !sameLease(audit.lease, lease) || !candidate.valid() ||
        !sameHumanIOProgram(lease.humanIOCandidate, candidate) ||
        candidate.deviceRegistryID != audit.device.registryID) {
        return audit.fail("post-physical HumanIO bind mismatch");
    }
    audit.lease = lease;
    ++audit.bindHumanIOCount;
    return true;
}

bool encodeOwner(void* raw, const Pass& pass) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    const auto phase = static_cast<std::uint32_t>(pass.phase);
    if (phase != audit.phaseCount || pass.environmentCount != 1u ||
        pass.stepIndex != 0u || pass.stepCount != 1u ||
        pass.substepIndex != 0u || pass.physicsSubstepCount != 1u ||
        pass.controlStep != kV4ControlStep || pass.transactionSlot != 0u ||
        pass.programFingerprint != kProgramFingerprint ||
        pass.transactionFingerprint != kTransactionFingerprint ||
        pass.linearizationEpoch != kLinearizationEpoch ||
        pass.slotGeneration != kSlotGeneration ||
        pass.encodeExactCandidate != nullptr ||
        pass.exactCandidateContext != nullptr) {
        return audit.fail("ABI4 owner phase identity mismatch");
    }
    ++audit.phaseCount;
    __unsafe_unretained id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    if (command == nil) return audit.fail("missing borrowed owner command");
    if (pass.phase == Phase::beginStep) {
        audit.liveQ = (__bridge id<MTLBuffer>)pass.q;
        audit.liveV = (__bridge id<MTLBuffer>)pass.v;
        audit.liveMujoco = (__bridge id<MTLBuffer>)pass.mujocoStates;
        audit.checkpointQ = (__bridge id<MTLBuffer>)pass.qCheckpoint;
        audit.checkpointV = (__bridge id<MTLBuffer>)pass.vCheckpoint;
        audit.checkpointMujoco =
            (__bridge id<MTLBuffer>)pass.mujocoStateCheckpoint;
        audit.ownerStatuses = (__bridge id<MTLBuffer>)pass.ownerStatuses;
        return true;
    }
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return audit.fail("owner callback blit allocation failed");
    if (pass.phase == Phase::preDynamics) {
        [blit copyFromBuffer:audit.reactionSource sourceOffset:0u
                    toBuffer:audit.candidate.reaction destinationOffset:0u
                         size:kNv * sizeof(float)];
    } else if (pass.phase == Phase::postDynamics) {
        [blit copyFromBuffer:audit.candidate.acceptJoint sourceOffset:0u
                    toBuffer:audit.candidate.joint destinationOffset:0u
                         size:sizeof(MRNumanXCoupledHumanStatusGPU)];
        [blit copyFromBuffer:audit.candidate.acceptToken sourceOffset:0u
                    toBuffer:audit.candidate.token destinationOffset:0u
                         size:MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_BYTES];
    } else {
        [blit endEncoding];
        return audit.fail("unknown owner phase");
    }
    [blit endEncoding];
    return true;
}

void abortOwner(void* raw, void*) noexcept {
    ++static_cast<V4Audit*>(raw)->abortCount;
}

bool acquireLease(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (audit.leaseActive.load() || lease.environmentCount != 1u ||
        lease.transactionSlot != 0u || lease.stepIndex != 0u ||
        lease.substepIndex != 0u || lease.physicsSubstepCount != 1u ||
        lease.controlStep != kV4ControlStep ||
        lease.slotGeneration != kSlotGeneration ||
        lease.preparedPhysicsStateTokens !=
            (__bridge void*)audit.candidate.token ||
        lease.matterApplyOutcomes !=
            (__bridge void*)audit.matterOutcomeTarget ||
        lease.proposals == nullptr || lease.proposedPhysicsStateTokens == nullptr ||
        lease.applyActions == nullptr || lease.appliedOutcomes == nullptr ||
        lease.finalAcceptedPhysicsStateTokens == nullptr ||
        lease.publicationFences == nullptr ||
        lease.physicalPreparedEvent == nullptr ||
        lease.physicalPreparedEventValue == 0u ||
        lease.proposalEventValue <= lease.physicalPreparedEventValue ||
        lease.appliedEventValue <= lease.proposalEventValue ||
        lease.humanIOCandidate.configured()) {
        return false;
    }
    audit.lease = lease;
    audit.leaseActive.store(true);
    return true;
}

metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition releaseLease(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease,
    void*,
    const bool completed
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    ++audit.releaseCount;
    if (!audit.leaseActive.load() || !sameLease(audit.lease, lease) ||
        !sameHumanIOProgram(
            audit.lease.humanIOCandidate, lease.humanIOCandidate) ||
        !completed) {
        return metalrobo::
            MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
    }
    const auto& applied = contents<MRNumanXHumanMatterAppliedOutcomeGPU>(
        (__bridge id<MTLBuffer>)lease.appliedOutcomes)[0];
    if (applied.status ==
        MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED) {
        return metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
            acceptedPendingPublication;
    }
    if (applied.status ==
        MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED) {
        if (lease.humanIOCandidate.rejectCandidate(
                lease.humanIOCandidate.context,
                lease.humanIOCandidate.candidatePublicationFingerprint) !=
            metalrobo::
                MetalNumanXHumanIOCandidatePublicationDisposition::rejected) {
            return metalrobo::
                MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
        }
        audit.leaseActive.store(false);
        return metalrobo::
            MetalNumanXHumanMatterPrepareLeaseDisposition::released;
    }
    return metalrobo::
        MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
}

bool reserveApplication(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease,
    const metalrobo::MetalNumanXHumanMatterProposalView& proposal,
    const metalrobo::MetalNumanXHumanMatterBrainPreflightView& preflight
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (!audit.leaseActive.load() || !sameLease(audit.lease, lease) ||
        !sameHumanIOProgram(
            audit.lease.humanIOCandidate, lease.humanIOCandidate) ||
        proposal.proposals != lease.proposals ||
        proposal.proposedPhysicsStateTokens !=
            lease.proposedPhysicsStateTokens ||
        preflight.brainCommitPreflights == nullptr ||
        preflight.preflightReadyEvent == nullptr ||
        preflight.preflightReadyEventValue == 0u ||
        preflight.environmentCount != 1u ||
        preflight.controlStep != kV4ControlStep ||
        preflight.slotGeneration != kSlotGeneration) {
        return audit.fail("application reservation identity mismatch");
    }
    ++audit.reserveApplicationCount;
    return true;
}

std::uint32_t appliedCodeForProposalReject(
    const std::uint32_t code
) noexcept {
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

MRNumanXHumanMatterApplyActionGPU expectedAction(
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterBrainAckGPU& ack
) noexcept {
    const bool ackAccept =
        ack.status == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT &&
        ack.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
        ack.code == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS &&
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
        ack.physicsTokenFingerprint == proposal.physicsTokenFingerprint &&
        ack.physicsTokenFingerprint != 0u &&
        ack.preflightFingerprint != 0u &&
        ack.fastGateFingerprint != 0u &&
        ack.brainWitnessFingerprint == proposal.brainWitnessFingerprint &&
        ack.brainWitnessFingerprint != 0u &&
        ack.brainProgramFingerprint == proposal.brainProgramFingerprint &&
        ack.brainProgramFingerprint != 0u;
    const bool ackReject =
        ack.status == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT &&
        ack.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
        ack.code == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT &&
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
        ack.physicsTokenFingerprint == 0u &&
        ack.preflightFingerprint == 0u &&
        ack.fastGateFingerprint == 0u &&
        ack.brainWitnessFingerprint == 0u &&
        ack.brainProgramFingerprint == kBrainProgramFingerprint;
    const bool ackIdentity =
        ack.abiVersion == MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION &&
        (ackAccept || ackReject) &&
        ack.programFingerprint == kProgramFingerprint &&
        ack.transactionFingerprint == kTransactionFingerprint &&
        ack.linearizationEpoch == kLinearizationEpoch &&
        ack.slotGeneration == kSlotGeneration &&
        ack.proposalFingerprint == proposal.proposalFingerprint &&
        ack.environment == 0u && ack.stepIndex == 0u &&
        ack.substepIndex == 0u && ack.transactionSlot == 0u &&
        ack.physicsSubstepCount == 1u && ack.controlStep == kV4ControlStep &&
        ack.ackFingerprint == recordFingerprint(ack);
    const bool ackValid = ackIdentity;
    const bool accept = ackValid &&
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
        ack.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    std::uint32_t code = MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_BRAIN_ACK;
    if (accept) {
        code = MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS;
    } else if (ackValid) {
        code = appliedCodeForProposalReject(proposal.code);
    }
    MRNumanXHumanMatterApplyActionGPU action{};
    action.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    action.status = accept ? MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT
                           : MR_NUMANX_HUMAN_MATTER_APPLY_REJECT;
    action.decision = accept ? MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT
                             : MR_NUMANX_HUMAN_MATTER_ROOT_REJECT;
    action.code = code;
    action.programFingerprint = kProgramFingerprint;
    action.transactionFingerprint = kTransactionFingerprint;
    action.linearizationEpoch = kLinearizationEpoch;
    action.slotGeneration = kSlotGeneration;
    action.physicsTokenFingerprint = ack.physicsTokenFingerprint;
    action.proposalFingerprint = proposal.proposalFingerprint;
    action.ackFingerprint = ack.ackFingerprint;
    action.preflightFingerprint = ack.preflightFingerprint;
    action.fastGateFingerprint = ack.fastGateFingerprint;
    action.brainWitnessFingerprint = ack.brainWitnessFingerprint;
    action.environment = 0u;
    action.stepIndex = 0u;
    action.substepIndex = 0u;
    action.transactionSlot = 0u;
    action.physicsSubstepCount = 1u;
    action.controlStep = kV4ControlStep;
    action.actionFingerprint = recordFingerprint(action);
    return action;
}

bool encodeApply(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease,
    const metalrobo::MetalNumanXHumanMatterApplyPass& pass
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (!audit.leaseActive.load() || !sameLease(audit.lease, lease) ||
        !sameHumanIOProgram(
            audit.lease.humanIOCandidate, lease.humanIOCandidate) ||
        pass.commandBuffer == nullptr || pass.brainAcks == nullptr ||
        pass.controlStep != kV4ControlStep ||
        pass.slotGeneration != kSlotGeneration) {
        return audit.fail("apply callback identity mismatch");
    }
    const auto proposal = contents<MRNumanXHumanMatterProposalGPU>(
        (__bridge id<MTLBuffer>)lease.proposals)[0];
    const auto ack = contents<MRNumanXHumanMatterBrainAckGPU>(
        (__bridge id<MTLBuffer>)pass.brainAcks)[0];
    const auto action = expectedAction(proposal, ack);
    MRNumanXHumanMatterMatterApplyOutcomeGPU outcome{};
    outcome.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    outcome.status = action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT
        ? MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT
        : MR_NUMANX_HUMAN_MATTER_APPLY_REJECT;
    outcome.decision = action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT
        ? MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT
        : MR_NUMANX_HUMAN_MATTER_ROOT_REJECT;
    outcome.code = action.code;
    outcome.programFingerprint = kProgramFingerprint;
    outcome.transactionFingerprint = kTransactionFingerprint;
    outcome.linearizationEpoch = kLinearizationEpoch;
    outcome.slotGeneration = kSlotGeneration;
    outcome.physicsTokenFingerprint = action.physicsTokenFingerprint;
    outcome.proposalFingerprint = action.proposalFingerprint;
    outcome.ackFingerprint = action.ackFingerprint;
    outcome.actionFingerprint = action.actionFingerprint;
    outcome.matterProgramFingerprint = kMatterProgramFingerprint;
    outcome.environment = 0u;
    outcome.stepIndex = 0u;
    outcome.substepIndex = 0u;
    outcome.transactionSlot = 0u;
    outcome.physicsSubstepCount = 1u;
    outcome.controlStep = kV4ControlStep;
    outcome.outcomeFingerprint = recordFingerprint(outcome);
    contents<MRNumanXHumanMatterMatterApplyOutcomeGPU>(
        audit.matterOutcomeSource)[0] = outcome;

    __unsafe_unretained id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return audit.fail("Matter apply blit allocation failed");
    [blit copyFromBuffer:audit.matterOutcomeSource sourceOffset:0u
                toBuffer:(__bridge id<MTLBuffer>)lease.matterApplyOutcomes
       destinationOffset:0u size:sizeof(outcome)];
    [blit endEncoding];
    ++audit.encodeApplyCount;
    return true;
}

void abortApply(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease&,
    const metalrobo::MetalNumanXHumanMatterApplyPass&
) noexcept {
    ++static_cast<V4Audit*>(raw)->abortApplyCount;
}

bool reservePublication(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease,
    const metalrobo::MetalNumanXHumanMatterPublicationReservationView& view
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (!audit.leaseActive.load() || !sameLease(audit.lease, lease) ||
        !sameHumanIOProgram(
            audit.lease.humanIOCandidate, lease.humanIOCandidate) ||
        view.proposal.proposals != lease.proposals ||
        view.appliedOutcomes != lease.appliedOutcomes ||
        view.finalAcceptedPhysicsStateTokens !=
            lease.finalAcceptedPhysicsStateTokens ||
        view.fence.publicationFences != lease.publicationFences ||
        view.jointCommitFingerprint != kJointCommitFingerprint ||
        view.brainGeneration != kBrainGeneration) {
        return audit.fail("publication reservation identity mismatch");
    }
    const auto& fence = contents<
        MRNumanXHumanMatterJointPublicationFenceGPU>(
            (__bridge id<MTLBuffer>)lease.publicationFences)[0];
    if (fence.status != MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING ||
        fence.fenceFingerprint != recordFingerprint(fence)) {
        return audit.fail("owner did not provide a valid PENDING fence");
    }
    if (!lease.humanIOCandidate.reservePublishedRoot(
            lease.humanIOCandidate.context,
            lease.humanIOCandidate.candidatePublicationFingerprint,
            view.humanIOBinding)) {
        return audit.fail("HumanIO candidate reservation was refused");
    }
    ++audit.reservePublicationCount;
    return true;
}

metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition releasePublication(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPrepareLease& lease,
    const metalrobo::MetalNumanXHumanMatterPublicationFenceView& fenceView
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (!audit.leaseActive.load() || !sameLease(audit.lease, lease) ||
        !sameHumanIOProgram(
            audit.lease.humanIOCandidate, lease.humanIOCandidate) ||
        fenceView.publicationFences != lease.publicationFences) {
        return metalrobo::
            MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
    }
    const auto& fence = contents<
        MRNumanXHumanMatterJointPublicationFenceGPU>(
            (__bridge id<MTLBuffer>)lease.publicationFences)[0];
    if (fence.status != MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED ||
        fence.fenceFingerprint != recordFingerprint(fence)) {
        return metalrobo::
            MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
    }
    metalrobo::MetalNumanXHumanIOCandidatePublicationCommit commit{};
    commit.candidatePublicationFingerprint =
        lease.humanIOCandidate.candidatePublicationFingerprint;
    commit.bindingFingerprint = audit.humanIOBinding.bindingFingerprint;
    commit.jointCommitFingerprint = fence.jointCommitFingerprint;
    commit.brainGeneration = fence.brainGeneration;
    commit.fenceFingerprint = fence.fenceFingerprint;
    if (lease.humanIOCandidate.publishCandidate(
            lease.humanIOCandidate.context,
            lease.humanIOCandidate.candidatePublicationFingerprint,
            commit) != metalrobo::
                MetalNumanXHumanIOCandidatePublicationDisposition::released) {
        return metalrobo::
            MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
    }
    ++audit.releasePublicationCount;
    audit.leaseActive.store(false);
    return metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::released;
}

void applyCompletion(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterApplyTerminalStatus status,
    const std::uint64_t generation
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    if (generation != kSlotGeneration) {
        audit.failure = "apply completion generation mismatch";
    }
    audit.applyTerminalStatus.store(static_cast<std::uint32_t>(status));
    audit.applyCompletionCount.fetch_add(1u);
}

void proposalCompletion(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterProposalCompletionStatus status,
    const std::uint64_t generation
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    bool safe = generation == kSlotGeneration &&
        audit.completionPrepared != nullptr;
    if (safe) {
        metalrobo::MetalNumanXHumanMatterPreparedView view{};
        safe = audit.completionPrepared->view(view) &&
            view.slotGeneration == generation &&
            !audit.completionPrepared->registerProposalCompletion(
                &audit, &proposalCompletion);
    }
    if (!safe) {
        audit.failure = "proposal completion callback was stale, locked, or duplicate-admissible";
    }
    if (safe && audit.dropPreparedInProposalCompletion) {
        *audit.completionPrepared =
            metalrobo::MetalNumanXHumanMatterPrepared{};
        audit.completionPrepared = nullptr;
    }
    audit.proposalCompletionStatus.store(
        static_cast<std::uint32_t>(status));
    audit.proposalCompletionReentrySafe.store(safe);
    audit.proposalCompletionCount.fetch_add(1u);
}

void physicalCompletion(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterPhysicalCompletionStatus status,
    const std::uint64_t generation
) noexcept {
    auto& audit = *static_cast<V4Audit*>(raw);
    bool safe = generation == kSlotGeneration &&
        audit.completionPrepared != nullptr;
    if (safe) {
        metalrobo::MetalNumanXHumanMatterPreparedView view{};
        safe = audit.completionPrepared->view(view) &&
            view.slotGeneration == generation &&
            !audit.completionPrepared->registerPhysicalCompletion(
                &audit, &physicalCompletion);
    }
    if (!safe) {
        audit.failure = "physical completion callback was stale, locked, or duplicate-admissible";
    }
    audit.physicalCompletionStatus.store(
        static_cast<std::uint32_t>(status));
    audit.physicalCompletionReentrySafe.store(safe);
    audit.physicalCompletionCount.fetch_add(1u);
}

void initializeV4Audit(V4Audit& audit, id<MTLDevice> device) {
    audit.device = device;
    initializeAudit(audit.candidate, device);
    initializeCanonicalToken(audit.candidate.acceptToken);
    audit.reactionSource = makeBuffer<float>(
        device, kNv, @"ABI4 staged reaction source");
    contents<float>(audit.reactionSource)[0] = 37.0f;
    audit.matterOutcomeSource = makeBuffer<
        MRNumanXHumanMatterMatterApplyOutcomeGPU>(
            device, 1u, @"ABI4 Matter outcome source");
    audit.matterOutcomeTarget = makeBuffer<
        MRNumanXHumanMatterMatterApplyOutcomeGPU>(
            device, 1u, @"ABI4 Matter outcome target");
}

metalrobo::MetalArticulatedOperatorInput makeV4Input(
    const metalrobo::EngineModel& model,
    const std::vector<MRArticulatedPointImpulseGPU>& points,
    V4Audit& audit
) {
    auto input = makeInput(model, points, audit.candidate);
    auto& program = input.stand.numanXHumanMatterProgram;
    program.capabilities &=
        ~metalrobo::MetalNumanXHumanMatterExactCandidateKinematics;
    program.accessFlags &=
        ~metalrobo::MetalNumanXHumanMatterMayEncodeExactCandidate;
    program.context = &audit;
    program.encode = &encodeOwner;
    program.abort = &abortOwner;
    program.acquirePrepareLease = &acquireLease;
    program.bindHumanIOCandidatePublication = &bindHumanIOCandidate;
    program.releasePrepareLease = &releaseLease;
    program.reservePreparedApplication = &reserveApplication;
    program.encodePreparedApply = &encodeApply;
    program.abortPreparedApply = &abortApply;
    program.reservePublishedRoot = &reservePublication;
    program.releasePublishedRoot = &releasePublication;
    program.matterApplyOutcomes = (__bridge void*)audit.matterOutcomeTarget;
    program.matterApplyOutcomesGPUAddress = audit.matterOutcomeTarget.gpuAddress;
    program.matterApplyOutcomeElementCount = 1u;
    program.matterApplyOutcomeStride = 1u;
    program.substepIndex = 0u;
    program.physicsSubstepCount = 1u;
    program.candidatePointCapacity = 0u;
    program.controlStep = kV4ControlStep;
    program.qCoordinateCount = model.articulations[0u].nq;
    program.dofCount = model.articulations[0u].nv;
    program.dofLayoutVersion =
        metalrobo::kMetalNumanXHumanMatterDofLayoutVersion;
    return input;
}

MRNumanXHumanMatterBrainCommitWitnessGPU makeWitness(
    const V4Scenario scenario,
    const std::uint64_t tokenFingerprint
) noexcept {
    if (isPhysicalRejectScenario(scenario) || usesForcedProposal(scenario)) {
        return {};
    }
    MRNumanXHumanMatterBrainCommitWitnessGPU witness{};
    witness.magic = MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_MAGIC;
    witness.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_ABI_VERSION;
    witness.structBytes =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_BYTES;
    witness.status = MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_COMPLETE;
    witness.decision = scenario == V4Scenario::brainReject
        ? MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
        : MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    witness.environment = 0u;
    witness.stepIndex = 0u;
    witness.substepIndex = 0u;
    witness.transactionSlot = 0u;
    witness.physicsSubstepCount = 1u;
    witness.controlStep = kV4ControlStep;
    witness.programFingerprint = kProgramFingerprint;
    witness.transactionFingerprint = kTransactionFingerprint;
    witness.linearizationEpoch = kLinearizationEpoch;
    witness.slotGeneration = kSlotGeneration;
    witness.physicsTokenFingerprint = tokenFingerprint;
    witness.brainProgramFingerprint = kBrainProgramFingerprint;
    witness.brainShadowStateFingerprint = kBrainShadowFingerprint;
    if (scenario == V4Scenario::tokenMismatch) {
        witness.physicsTokenFingerprint ^= 1u;
    }
    witness.witnessFingerprint = witnessFingerprint(witness);
    if (scenario == V4Scenario::invalidBrainWitness) {
        witness.witnessFingerprint ^= 1u;
    }
    return witness;
}

MRNumanXHumanMatterBrainCommitPreflightGPU makePreflight(
    const V4Scenario scenario,
    const MRNumanXHumanMatterProposalGPU& proposal
) noexcept {
    if (isPhysicalRejectScenario(scenario) ||
        scenario == V4Scenario::forcedReject) return {};
    MRNumanXHumanMatterBrainCommitPreflightGPU preflight{};
    preflight.abiVersion = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION;
    preflight.structBytes = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES;
    preflight.status = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS;
    preflight.environment = 0u;
    preflight.controlStep = kV4ControlStep;
    preflight.substepIndex = 0u;
    preflight.physicsSubstepCount = 1u;
    preflight.transactionSlot = 0u;
    preflight.ownerProgramFingerprint = kProgramFingerprint;
    preflight.transactionFingerprint = kTransactionFingerprint;
    preflight.linearizationEpoch = kLinearizationEpoch;
    preflight.slotGeneration = kSlotGeneration;
    preflight.substepFingerprint = 0x5355425354455050ull;
    preflight.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    preflight.fastTargetGeneration = 41u;
    preflight.cognitiveTargetGeneration = 43u;
    preflight.jointReceiptFingerprint = 0x5245434549505431ull;
    preflight.fastProgramFingerprint = kFastProgramFingerprint;
    preflight.brainProgramFingerprint = kBrainProgramFingerprint;
    preflight.preflightFingerprint = recordFingerprint(preflight);
    return preflight;
}

MRNumanXHumanMatterBrainAckGPU makeAck(
    const V4Scenario scenario,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterBrainCommitPreflightGPU& preflight
) noexcept {
    const bool reject = proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT;
    MRNumanXHumanMatterBrainAckGPU ack{};
    ack.abiVersion = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION;
    ack.status = reject ? MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT
                        : MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT;
    ack.decision = reject ? MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
                          : MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    ack.code = reject ? MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT
                      : MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS;
    ack.programFingerprint = kProgramFingerprint;
    ack.transactionFingerprint = kTransactionFingerprint;
    ack.linearizationEpoch = kLinearizationEpoch;
    ack.slotGeneration = kSlotGeneration;
    ack.physicsTokenFingerprint = reject
        ? 0u : proposal.physicsTokenFingerprint;
    ack.proposalFingerprint = proposal.proposalFingerprint;
    ack.preflightFingerprint = reject ? 0u : preflight.preflightFingerprint;
    ack.fastGateFingerprint = reject ? 0u : kFastGateFingerprint;
    ack.brainWitnessFingerprint = reject
        ? 0u : proposal.brainWitnessFingerprint;
    ack.brainProgramFingerprint = reject
        ? kBrainProgramFingerprint : proposal.brainProgramFingerprint;
    ack.environment = 0u;
    ack.stepIndex = 0u;
    ack.substepIndex = 0u;
    ack.transactionSlot = 0u;
    ack.physicsSubstepCount = 1u;
    ack.controlStep = scenario == V4Scenario::staleAck
        ? kV4ControlStep + 1u : kV4ControlStep;
    ack.ackFingerprint = recordFingerprint(ack);
    return ack;
}

template <typename T>
std::vector<std::uint8_t> bytes(id<MTLBuffer> buffer, const std::size_t count) {
    require(buffer != nil && buffer.contents != nullptr &&
                buffer.length >= count * sizeof(T),
            "snapshot range is invalid");
    const auto* begin = static_cast<const std::uint8_t*>(buffer.contents);
    return {begin, begin + count * sizeof(T)};
}

void waitForApplyCompletion(V4Audit& audit) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while (audit.applyCompletionCount.load() == 0u &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(audit.applyCompletionCount.load() == 1u,
            "apply completion callback did not arrive exactly once");
}

void waitForProposalCompletion(V4Audit& audit) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while (audit.proposalCompletionCount.load() == 0u &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(audit.proposalCompletionCount.load() == 1u,
            "proposal completion callback did not arrive exactly once");
    require(audit.proposalCompletionStatus.load() ==
                static_cast<std::uint32_t>(metalrobo::
                    MetalNumanXHumanMatterProposalCompletionStatus::ready) &&
                audit.proposalCompletionReentrySafe.load(),
            "proposal completion was not READY and reentrant-safe");
}

struct V4Result {
    MRNumanXHumanMatterProposalGPU proposal{};
    MRNumanXHumanMatterBrainAckGPU ack{};
    MRNumanXHumanMatterAppliedOutcomeGPU applied{};
    MRNumanXHumanMatterJointPublicationFenceGPU fence{};
    std::vector<std::uint8_t> finalToken;
    std::vector<std::uint8_t> q;
    std::vector<std::uint8_t> v;
    std::vector<std::uint8_t> mujoco;
};

V4Result runScenario(
    id<MTLDevice> device,
    const char* metallibPath,
    const V4Scenario scenario
) {
    metalrobo::EngineModel model = makeHumanModel();
    const auto points = bodyProbes();
    V4Audit audit;
    initializeV4Audit(audit, device);
    audit.dropPreparedInProposalCompletion =
        scenario == V4Scenario::proposalCallbackDrop;
    if (isPhysicalRejectScenario(scenario)) {
        auto& joint = contents<MRNumanXCoupledHumanStatusGPU>(
            audit.candidate.acceptJoint)[0];
        joint.decision = MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN;
        joint.humanCode = MR_NUMI_HUMAN_STAND_CONTACT_FAILED;
        std::memset(
            audit.candidate.acceptToken.contents,
            0,
            audit.candidate.acceptToken.length
        );
    }
    auto input = makeV4Input(model, points, audit);
    require(input.stand.numanXHumanMatterProgram.valid(),
            "ABI4 Human/Matter program is invalid");
    const metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = kTimestep,
        .metallibPath = metallibPath,
    };
    metalrobo::MetalArticulatedOperatorContext context(config);
    metalrobo::MetalArticulatedOperatorSubmission submission;
    const auto submit = context.submit(model, input, submission);
    require(submit.succeeded() && submit.dispatched && submission.valid(),
            "ABI4 physical prepare submission failed: " + submit.message);
    metalrobo::MetalNumanXHumanMatterPrepared prepared;
    require(submission.extractPreparedHumanMatter(prepared) &&
                !submission.valid() && prepared.valid(),
            "prepared capability did not adopt the submission");
    audit.completionPrepared = &prepared;
    metalrobo::MetalNumanXHumanMatterPreparedView view{};
    require(prepared.view(view) && view.controlStep == kV4ControlStep &&
                view.physicsSubstepCount == 1u,
            "ABI4 prepared view is malformed");
    const bool registerPhysicalAfterCompletion =
        scenario == V4Scenario::tokenMismatch;
    if (!registerPhysicalAfterCompletion) {
        require(prepared.registerPhysicalCompletion(
                    &audit, &physicalCompletion) &&
                    !prepared.registerPhysicalCompletion(
                        &audit, &physicalCompletion),
                "physical completion registration was not exact-once");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLSharedEvent> brainPrepared = [device newSharedEvent];
    id<MTLSharedEvent> preflightReady = [device newSharedEvent];
    id<MTLSharedEvent> ackReady = [device newSharedEvent];
    require(queue != nil && brainPrepared != nil &&
                preflightReady != nil && ackReady != nil,
            "ABI4 timeline allocation failed");

    id<MTLCommandBuffer> physicalWait = [queue commandBuffer];
    require(prepared.encodeWaitForPhysicalPrepare(
                (__bridge void*)physicalWait),
            "physical-prepare wait encoding failed");
    [physicalWait commit];
    [physicalWait waitUntilCompleted];
    require(physicalWait.status == MTLCommandBufferStatusCompleted,
            "physical-prepare wait command failed");
    if (registerPhysicalAfterCompletion) {
        require(prepared.registerPhysicalCompletion(
                    &audit, &physicalCompletion) &&
                    !prepared.registerPhysicalCompletion(
                        &audit, &physicalCompletion),
                "late physical completion did not notify synchronously once");
    }
    require(audit.physicalCompletionCount.load() == 1u &&
                audit.physicalCompletionStatus.load() ==
                    static_cast<std::uint32_t>(metalrobo::
                        MetalNumanXHumanMatterPhysicalCompletionStatus::ready) &&
                audit.physicalCompletionReentrySafe.load(),
            "physical completion was not settled, exact-once, and reentrant");

    const auto humanIO = makeHumanIOCandidateProgram(audit);
    require(humanIO.valid(), "synthetic HumanIO candidate program is invalid");
    auto wrongDevice = humanIO;
    wrongDevice.deviceRegistryID ^= 1u;
    wrongDevice.identityFingerprint =
        wrongDevice.computedIdentityFingerprint();
    require(wrongDevice.valid() &&
                !prepared.bindHumanIOCandidatePublication(wrongDevice) &&
                audit.bindHumanIOCount == 0u,
            "wrong-device HumanIO candidate was admitted or consumed");
    auto stale = humanIO;
    stale.transactionFingerprint ^= 1u;
    stale.identityFingerprint = stale.computedIdentityFingerprint();
    require(stale.valid() &&
                !prepared.bindHumanIOCandidatePublication(stale) &&
                audit.bindHumanIOCount == 0u,
            "stale-root HumanIO candidate was admitted or consumed");
    require(prepared.bindHumanIOCandidatePublication(humanIO) &&
                audit.bindHumanIOCount == 1u &&
                !prepared.bindHumanIOCandidatePublication(humanIO) &&
                audit.bindHumanIOCount == 1u,
            "HumanIO candidate bind was not exact-once");
    require(prepared.view(view) &&
                view.humanIOCandidateKeyFingerprint ==
                    kHumanIOCandidateKeyFingerprint &&
                view.acceptedBrainGeneration == kBrainGeneration &&
                view.humanIOSensorGeneration == kHumanIOSensorGeneration &&
                view.humanIOProgramFingerprint ==
                    kHumanIOProgramFingerprint &&
                view.humanIOSensorFingerprint ==
                    kHumanIOSensorFingerprint &&
                view.humanIOTransactionInstanceFingerprint ==
                    kHumanIOTransactionInstanceFingerprint &&
                view.humanIOCandidatePublicationFingerprint ==
                    kHumanIOCandidatePublicationFingerprint &&
                view.humanIODeviceRegistryID == device.registryID &&
                view.humanIOIdentityFingerprint ==
                    humanIO.identityFingerprint,
            "prepared view did not expose exact pointer-free HumanIO identity");

    const std::uint64_t tokenFingerprint =
        contents<std::uint64_t>(audit.candidate.acceptToken)[7];
    id<MTLBuffer> witness = makeBuffer<
        MRNumanXHumanMatterBrainCommitWitnessGPU>(
            device, 1u, @"ABI4 Brain witness");
    contents<MRNumanXHumanMatterBrainCommitWitnessGPU>(witness)[0] =
        makeWitness(scenario, tokenFingerprint);
    brainPrepared.signaledValue = 11u;

    const auto preparedQ = bytes<std::uint8_t>(
        audit.liveQ, audit.liveQ.length);
    const auto preparedV = bytes<std::uint8_t>(
        audit.liveV, audit.liveV.length);
    const auto preparedMujoco = bytes<std::uint8_t>(
        audit.liveMujoco, audit.liveMujoco.length);

    id<MTLCommandBuffer> proposalCommand = [queue commandBuffer];
    metalrobo::MetalNumanXHumanMatterProposalRequest proposalRequest{};
    const bool forceProposal = usesForcedProposal(scenario);
    proposalRequest.mode = forceProposal
        ? metalrobo::MetalNumanXHumanMatterProposalMode::forceReject
        : metalrobo::MetalNumanXHumanMatterProposalMode::validateBrainWitness;
    proposalRequest.commandBuffer = (__bridge void*)proposalCommand;
    if (!forceProposal) {
        proposalRequest.brainCommitWitnesses = (__bridge void*)witness;
        proposalRequest.brainPrepareCompleteEvent =
            (__bridge void*)brainPrepared;
        proposalRequest.brainPrepareCompleteEventValue = 11u;
        proposalRequest.brainCommitWitnessesGPUAddress = witness.gpuAddress;
        proposalRequest.brainCommitWitnessElementCount = 1u;
        proposalRequest.brainCommitWitnessStride = 1u;
    }
    proposalRequest.environmentCount = 1u;
    proposalRequest.transactionSlot = 0u;
    proposalRequest.stepIndex = 0u;
    proposalRequest.substepIndex = 0u;
    proposalRequest.physicsSubstepCount = 1u;
    proposalRequest.controlStep = kV4ControlStep;
    proposalRequest.programFingerprint = kProgramFingerprint;
    proposalRequest.transactionFingerprint = kTransactionFingerprint;
    proposalRequest.linearizationEpoch = kLinearizationEpoch;
    proposalRequest.slotGeneration = kSlotGeneration;
    const auto proposed = prepared.proposePrepared(proposalRequest);
    require(proposed.succeeded() && proposed.encoded,
            "mutation-free proposal was rejected: " + proposed.message);
    require(!prepared.bindHumanIOCandidatePublication(humanIO) &&
                audit.bindHumanIOCount == 1u,
            "after-proposal HumanIO rebinding was admitted");
    require(!prepared.proposePrepared(proposalRequest).succeeded(),
            "duplicate proposal was admitted");
    if (scenario == V4Scenario::proposalAbortRetry) {
        require(prepared.registerProposalCompletion(
                    &audit, &proposalCompletion) &&
                    prepared.abortProposal((__bridge void*)proposalCommand) &&
                    audit.proposalCompletionCount.load() == 0u,
                "registered uncommitted proposal did not abort without a callback");
        proposalCommand = [queue commandBuffer];
        require(proposalCommand != nil,
                "failed to allocate proposal retry command buffer");
        proposalRequest.commandBuffer = (__bridge void*)proposalCommand;
        const auto retried = prepared.proposePrepared(proposalRequest);
        require(retried.succeeded() && retried.encoded,
                "proposal retry after exact abort was rejected: " +
                    retried.message);
    }
    const bool registerAfterCompletion =
        scenario == V4Scenario::tokenMismatch;
    if (!registerAfterCompletion) {
        require(prepared.registerProposalCompletion(
                    &audit, &proposalCompletion) &&
                    !prepared.registerProposalCompletion(
                        &audit, &proposalCompletion),
                "proposal completion registration was not exact-once");
    }
    [proposalCommand commit];
    [proposalCommand waitUntilCompleted];
    require(proposalCommand.status == MTLCommandBufferStatusCompleted,
            "proposal command failed");
    if (registerAfterCompletion) {
        require(prepared.registerProposalCompletion(
                    &audit, &proposalCompletion) &&
                    !prepared.registerProposalCompletion(
                        &audit, &proposalCompletion),
                "already-completed proposal did not notify synchronously exactly once");
    }
    waitForProposalCompletion(audit);
    require(preparedQ == bytes<std::uint8_t>(audit.liveQ, audit.liveQ.length) &&
                preparedV == bytes<std::uint8_t>(audit.liveV, audit.liveV.length) &&
                preparedMujoco == bytes<std::uint8_t>(
                    audit.liveMujoco, audit.liveMujoco.length),
            "proposal mutated quarantined Human state");

    const auto proposal = contents<MRNumanXHumanMatterProposalGPU>(
        (__bridge id<MTLBuffer>)view.proposals)[0];
    require(proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
                proposal.candidatePublicationFingerprint ==
                    kHumanIOCandidatePublicationFingerprint &&
                proposal.humanIOIdentityFingerprint ==
                    humanIO.identityFingerprint &&
                proposal.proposalFingerprint == recordFingerprint(proposal),
            "proposal record failed integrity validation");
    if (scenario == V4Scenario::proposalCallbackDrop) {
        const auto terminalStats = context.stats();
        require(!prepared.valid() &&
                    terminalStats.submissionDestructorWaitCount == 0u &&
                    terminalStats.terminalSubmissionNonwaitingReapCount == 1u &&
                    terminalStats.completedSubmissionCount == 1u &&
                    terminalStats.hasInFlightSubmission &&
                    audit.leaseActive.load(),
                "proposal callback drop waited or prematurely released quarantined authority");
        V4Result dropped{};
        dropped.proposal = proposal;
        return dropped;
    }
    id<MTLBuffer> preflight = makeBuffer<
        MRNumanXHumanMatterBrainCommitPreflightGPU>(
            device, 1u, @"ABI4 Brain commit preflight");
    const auto preflightValue = makePreflight(scenario, proposal);
    contents<MRNumanXHumanMatterBrainCommitPreflightGPU>(preflight)[0] =
        preflightValue;
    preflightReady.signaledValue = 13u;
    metalrobo::MetalNumanXHumanMatterBrainPreflightView preflightView{};
    preflightView.brainCommitPreflights = (__bridge void*)preflight;
    preflightView.preflightReadyEvent = (__bridge void*)preflightReady;
    preflightView.brainCommitPreflightsGPUAddress = preflight.gpuAddress;
    preflightView.brainCommitPreflightElementCount = 1u;
    preflightView.preflightReadyEventValue = 13u;
    preflightView.brainCommitPreflightStride = 1u;
    preflightView.environmentCount = 1u;
    preflightView.transactionSlot = 0u;
    preflightView.stepIndex = 0u;
    preflightView.substepIndex = 0u;
    preflightView.physicsSubstepCount = 1u;
    preflightView.controlStep = kV4ControlStep;
    preflightView.programFingerprint = kProgramFingerprint;
    preflightView.transactionFingerprint = kTransactionFingerprint;
    preflightView.linearizationEpoch = kLinearizationEpoch;
    preflightView.slotGeneration = kSlotGeneration;
    require(prepared.reservePreparedApplication(preflightView) &&
                audit.reserveApplicationCount == 1u,
            "pre-ACK application reservation failed");

    id<MTLBuffer> ackBuffer = makeBuffer<MRNumanXHumanMatterBrainAckGPU>(
        device, 1u, @"ABI4 Brain ACK");
    const auto ack = makeAck(scenario, proposal, preflightValue);
    contents<MRNumanXHumanMatterBrainAckGPU>(ackBuffer)[0] = ack;
    ackReady.signaledValue = 17u;
    id<MTLCommandBuffer> applyCommand = [queue commandBuffer];
    metalrobo::MetalNumanXHumanMatterApplyRequest applyRequest{};
    applyRequest.commandBuffer = (__bridge void*)applyCommand;
    applyRequest.brainAcks = (__bridge void*)ackBuffer;
    applyRequest.brainAckEvent = (__bridge void*)ackReady;
    applyRequest.completionContext = &audit;
    applyRequest.completion = &applyCompletion;
    applyRequest.brainAckEventValue = 17u;
    applyRequest.brainAcksGPUAddress = ackBuffer.gpuAddress;
    applyRequest.brainAckElementCount = 1u;
    applyRequest.brainAckStride = 1u;
    applyRequest.environmentCount = 1u;
    applyRequest.transactionSlot = 0u;
    applyRequest.stepIndex = 0u;
    applyRequest.substepIndex = 0u;
    applyRequest.physicsSubstepCount = 1u;
    applyRequest.controlStep = kV4ControlStep;
    applyRequest.programFingerprint = kProgramFingerprint;
    applyRequest.transactionFingerprint = kTransactionFingerprint;
    applyRequest.linearizationEpoch = kLinearizationEpoch;
    applyRequest.slotGeneration = kSlotGeneration;
    const auto appliedDiagnostics = prepared.applyPrepared(applyRequest);
    require(appliedDiagnostics.succeeded() && appliedDiagnostics.encoded,
            "ABI4 apply was rejected: " + appliedDiagnostics.message);
    require(!prepared.applyPrepared(applyRequest).succeeded(),
            "duplicate apply was admitted");
    [applyCommand commit];
    [applyCommand waitUntilCompleted];
    require(applyCommand.status == MTLCommandBufferStatusCompleted,
            "ABI4 apply command failed");
    waitForApplyCompletion(audit);

    V4Result result{};
    result.proposal = proposal;
    result.ack = ack;
    result.applied = contents<MRNumanXHumanMatterAppliedOutcomeGPU>(
        (__bridge id<MTLBuffer>)view.appliedOutcomes)[0];
    result.finalToken = bytes<std::uint8_t>(
        (__bridge id<MTLBuffer>)view.finalAcceptedPhysicsStateTokens,
        MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
    result.q = bytes<std::uint8_t>(audit.liveQ, audit.liveQ.length);
    result.v = bytes<std::uint8_t>(audit.liveV, audit.liveV.length);
    result.mujoco = bytes<std::uint8_t>(
        audit.liveMujoco, audit.liveMujoco.length);
    require(result.applied.appliedFingerprint ==
                recordFingerprint(result.applied) &&
                (scenario == V4Scenario::staleAck
                    ? result.applied.matterApplyFingerprint == 0u
                    : result.applied.matterApplyFingerprint != 0u),
            "applied outcome failed integrity validation");

    if (scenario == V4Scenario::staleAck) {
        const auto zeroToken = std::vector<std::uint8_t>(
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES, 0u);
        metalrobo::MetalNumanXHumanMatterPublicationReservationRequest
            terminalReserve{};
        terminalReserve.environmentCount = 1u;
        terminalReserve.transactionSlot = 0u;
        terminalReserve.stepIndex = 0u;
        terminalReserve.substepIndex = 0u;
        terminalReserve.physicsSubstepCount = 1u;
        terminalReserve.controlStep = kV4ControlStep;
        terminalReserve.programFingerprint = kProgramFingerprint;
        terminalReserve.transactionFingerprint = kTransactionFingerprint;
        terminalReserve.linearizationEpoch = kLinearizationEpoch;
        terminalReserve.slotGeneration = kSlotGeneration;
        terminalReserve.jointCommitFingerprint = kJointCommitFingerprint;
        terminalReserve.brainGeneration = kBrainGeneration;
        metalrobo::MetalNumanXHumanMatterPublicationReleaseRequest
            terminalRelease{};
        terminalRelease.publicationFences = view.publicationFences;
        terminalRelease.publicationFencesGPUAddress =
            view.publicationFencesGPUAddress;
        terminalRelease.publicationFenceElementCount =
            view.publicationFenceElementCount;
        terminalRelease.publicationFenceStride = view.publicationFenceStride;
        terminalRelease.environmentCount = 1u;
        terminalRelease.transactionSlot = 0u;
        terminalRelease.stepIndex = 0u;
        terminalRelease.substepIndex = 0u;
        terminalRelease.physicsSubstepCount = 1u;
        terminalRelease.controlStep = kV4ControlStep;
        terminalRelease.programFingerprint = kProgramFingerprint;
        terminalRelease.transactionFingerprint = kTransactionFingerprint;
        terminalRelease.linearizationEpoch = kLinearizationEpoch;
        terminalRelease.slotGeneration = kSlotGeneration;
        terminalRelease.jointCommitFingerprint = kJointCommitFingerprint;
        terminalRelease.brainGeneration = kBrainGeneration;
        require(result.applied.status ==
                    MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH &&
                result.applied.decision ==
                    MR_NUMANX_HUMAN_MATTER_ROOT_PENDING &&
                result.applied.code ==
                    MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER &&
                result.finalToken == zeroToken &&
                result.q == preparedQ && result.v == preparedV &&
                result.mujoco == preparedMujoco &&
                prepared.valid() &&
                !prepared.proposePrepared(proposalRequest).succeeded() &&
                !prepared.applyPrepared(applyRequest).succeeded() &&
                !prepared.reservePreparedApplication(preflightView) &&
                !prepared.reservePublishedRoot(terminalReserve) &&
                prepared.releasePublishedRoot(terminalRelease) ==
                    metalrobo::
                        MetalNumanXHumanMatterPrepareLeaseDisposition::
                            terminalNoTouch &&
                audit.leaseActive.load() &&
                audit.rejectHumanIOCount == 0u &&
                audit.reserveHumanIOCount == 0u &&
                audit.publishHumanIOCount == 0u &&
                audit.releasePublicationCount == 0u &&
                audit.applyTerminalStatus.load() ==
                    static_cast<std::uint32_t>(metalrobo::
                        MetalNumanXHumanMatterApplyTerminalStatus::
                            terminalNoTouch),
            "stale ACK did not preserve exact terminal-no-touch quarantine");
        prepared = metalrobo::MetalNumanXHumanMatterPrepared{};
        const auto terminalStats = context.stats();
        require(terminalStats.submissionDestructorWaitCount == 0u &&
                    terminalStats.terminalSubmissionNonwaitingReapCount == 1u &&
                    terminalStats.hasInFlightSubmission,
                "late terminal quarantine cleanup performed a host wait or reaped authority");
        return result;
    }

    const bool expectedAccept = scenario == V4Scenario::accept ||
        scenario == V4Scenario::malformedPublicationRelease ||
        scenario == V4Scenario::proposalAbortRetry;
    if (!expectedAccept) {
        require(result.applied.status ==
                    MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED &&
                    result.q == bytes<std::uint8_t>(
                        audit.checkpointQ, audit.checkpointQ.length) &&
                    result.v == bytes<std::uint8_t>(
                        audit.checkpointV, audit.checkpointV.length) &&
                    result.mujoco == bytes<std::uint8_t>(
                        audit.checkpointMujoco,
                        audit.checkpointMujoco.length) &&
                    !prepared.valid() && !audit.leaseActive.load() &&
                    audit.bindHumanIOCount == 1u &&
                    audit.rejectHumanIOCount == 1u &&
                    audit.reserveHumanIOCount == 0u &&
                    audit.publishHumanIOCount == 0u,
                "rejected/stale ACK did not restore and release exactly once");
        require(audit.applyTerminalStatus.load() ==
                    static_cast<std::uint32_t>(metalrobo::
                        MetalNumanXHumanMatterApplyTerminalStatus::
                            rejectedReleased),
                "rejected apply did not report released terminal status");
        const auto terminalStats = context.stats();
        require(terminalStats.submissionDestructorWaitCount == 0u &&
                    terminalStats.terminalSubmissionNonwaitingReapCount == 1u &&
                    terminalStats.completedSubmissionCount == 1u &&
                    !terminalStats.hasInFlightSubmission,
                "rejected root reaped through a host wait");
        if (isPhysicalRejectScenario(scenario)) {
            require(result.proposal.code ==
                        MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT &&
                    result.proposal.physicsTokenFingerprint == 0u &&
                    result.proposal.brainProgramFingerprint == 0u &&
                    result.proposal.brainShadowStateFingerprint == 0u &&
                    result.proposal.brainWitnessFingerprint == 0u &&
                    result.ack.status ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT &&
                    result.ack.physicsTokenFingerprint == 0u &&
                    result.ack.preflightFingerprint == 0u &&
                    result.ack.fastGateFingerprint == 0u &&
                    result.ack.brainWitnessFingerprint == 0u &&
                    result.ack.brainProgramFingerprint ==
                        kBrainProgramFingerprint &&
                    result.applied.code ==
                        MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT &&
                    result.applied.physicsTokenFingerprint == 0u &&
                    result.applied.preflightFingerprint == 0u &&
                    result.applied.fastGateFingerprint == 0u,
                    "physical reject lost its exact zero-gate code/shape");
        } else if (scenario == V4Scenario::forcedReject) {
            require(result.proposal.code ==
                        MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT &&
                    result.proposal.physicsTokenFingerprint == 0u &&
                    result.proposal.brainProgramFingerprint == 0u &&
                    result.proposal.brainShadowStateFingerprint == 0u &&
                    result.proposal.brainWitnessFingerprint == 0u &&
                    result.applied.code ==
                        MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT,
                    "prepared force reject lost its exact cause mapping");
        } else if (scenario == V4Scenario::brainReject) {
            require(result.proposal.code ==
                        MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT &&
                    result.proposal.physicsTokenFingerprint == 0u &&
                    result.proposal.brainProgramFingerprint == 0u &&
                    result.proposal.brainShadowStateFingerprint == 0u &&
                    result.proposal.brainWitnessFingerprint == 0u &&
                    result.ack.status ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT &&
                    result.ack.code ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT &&
                    result.ack.physicsTokenFingerprint == 0u &&
                    result.ack.preflightFingerprint == 0u &&
                    result.ack.fastGateFingerprint == 0u &&
                    result.ack.brainWitnessFingerprint == 0u &&
                    result.ack.brainProgramFingerprint ==
                        kBrainProgramFingerprint &&
                    result.applied.code ==
                        MR_NUMANX_HUMAN_MATTER_APPLIED_BRAIN_REJECT,
                    "Brain reject lost its exact zero-gate cause mapping");
        } else if (scenario == V4Scenario::invalidBrainWitness) {
            require(result.proposal.code ==
                        MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS &&
                    result.proposal.physicsTokenFingerprint == 0u &&
                    result.proposal.brainProgramFingerprint == 0u &&
                    result.proposal.brainShadowStateFingerprint == 0u &&
                    result.proposal.brainWitnessFingerprint == 0u &&
                    result.ack.status ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT &&
                    result.ack.code ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT &&
                    result.ack.physicsTokenFingerprint == 0u &&
                    result.ack.preflightFingerprint == 0u &&
                    result.ack.fastGateFingerprint == 0u &&
                    result.ack.brainWitnessFingerprint == 0u &&
                    result.ack.brainProgramFingerprint ==
                        kBrainProgramFingerprint &&
                    result.applied.code ==
                        MR_NUMANX_HUMAN_MATTER_APPLIED_BRAIN_REJECT,
                    "invalid Brain witness lost its exact cause mapping");
        } else if (scenario == V4Scenario::tokenMismatch) {
            require(result.proposal.code ==
                        MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH &&
                    result.proposal.physicsTokenFingerprint == 0u &&
                    result.proposal.brainProgramFingerprint == 0u &&
                    result.proposal.brainShadowStateFingerprint == 0u &&
                    result.proposal.brainWitnessFingerprint == 0u &&
                    result.ack.status ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT &&
                    result.ack.code ==
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT &&
                    result.ack.physicsTokenFingerprint == 0u &&
                    result.ack.preflightFingerprint == 0u &&
                    result.ack.fastGateFingerprint == 0u &&
                    result.ack.brainWitnessFingerprint == 0u &&
                    result.ack.brainProgramFingerprint ==
                        kBrainProgramFingerprint &&
                    result.applied.code ==
                        MR_NUMANX_HUMAN_MATTER_APPLIED_TOKEN_MISMATCH,
                    "token mismatch lost its exact zero-gate cause mapping");
        }
        return result;
    }

    require(result.applied.status ==
                MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED &&
                result.q == preparedQ && result.v == preparedV &&
                result.mujoco == preparedMujoco && prepared.valid() &&
                audit.applyTerminalStatus.load() == static_cast<std::uint32_t>(
                    metalrobo::MetalNumanXHumanMatterApplyTerminalStatus::
                        acceptedPendingPublication),
            "accepted apply was not preserved and quarantined");
    const auto proposedToken = bytes<std::uint8_t>(
        (__bridge id<MTLBuffer>)view.proposedPhysicsStateTokens,
        MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
    require(result.finalToken == proposedToken,
            "final token differs from immutable proposed token");

    metalrobo::MetalNumanXHumanMatterPublicationReservationRequest reserve{};
    reserve.environmentCount = 1u;
    reserve.transactionSlot = 0u;
    reserve.stepIndex = 0u;
    reserve.substepIndex = 0u;
    reserve.physicsSubstepCount = 1u;
    reserve.controlStep = kV4ControlStep;
    reserve.programFingerprint = kProgramFingerprint;
    reserve.transactionFingerprint = kTransactionFingerprint;
    reserve.linearizationEpoch = kLinearizationEpoch;
    reserve.slotGeneration = kSlotGeneration;
    reserve.jointCommitFingerprint = kJointCommitFingerprint;
    reserve.brainGeneration = kBrainGeneration;
    require(
        prepared.reservePublishedRoot(reserve),
        audit.failure == nullptr
            ? "publication reservation was rejected"
            : std::string("publication reservation was rejected: ") +
                audit.failure);
    require(!prepared.reservePublishedRoot(reserve),
            "duplicate publication reservation was admitted");
    require(audit.reservePublicationCount == 1u,
            "publication reservation callback was not exact-once");
    auto& fence = contents<MRNumanXHumanMatterJointPublicationFenceGPU>(
        (__bridge id<MTLBuffer>)view.publicationFences)[0];
    fence.status = MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED;
    fence.fenceFingerprint = recordFingerprint(fence);
    result.fence = fence;
    metalrobo::MetalNumanXHumanMatterPublicationReleaseRequest release{};
    release.publicationFences = view.publicationFences;
    release.publicationFencesGPUAddress = view.publicationFencesGPUAddress;
    release.publicationFenceElementCount = view.publicationFenceElementCount;
    release.publicationFenceStride = view.publicationFenceStride;
    release.environmentCount = 1u;
    release.transactionSlot = 0u;
    release.stepIndex = 0u;
    release.substepIndex = 0u;
    release.physicsSubstepCount = 1u;
    release.controlStep = kV4ControlStep;
    release.programFingerprint = kProgramFingerprint;
    release.transactionFingerprint = kTransactionFingerprint;
    release.linearizationEpoch = kLinearizationEpoch;
    release.slotGeneration = kSlotGeneration;
    release.jointCommitFingerprint = kJointCommitFingerprint;
    release.brainGeneration = kBrainGeneration;
    if (scenario == V4Scenario::malformedPublicationRelease) {
        auto malformed = release;
        malformed.controlStep += 1u;
        require(prepared.releasePublishedRoot(malformed) == metalrobo::
                    MetalNumanXHumanMatterPrepareLeaseDisposition::
                        terminalNoTouch &&
                    prepared.valid() && audit.releasePublicationCount == 0u &&
                    audit.reserveHumanIOCount == 1u &&
                    audit.publishHumanIOCount == 0u &&
                    prepared.releasePublishedRoot(release) == metalrobo::
                        MetalNumanXHumanMatterPrepareLeaseDisposition::
                            terminalNoTouch,
                "malformed owning release remained retryable");
        prepared = metalrobo::MetalNumanXHumanMatterPrepared{};
        const auto terminalStats = context.stats();
        require(terminalStats.submissionDestructorWaitCount == 0u &&
                    terminalStats.terminalSubmissionNonwaitingReapCount == 1u &&
                    terminalStats.hasInFlightSubmission,
                "malformed publication quarantine cleanup performed a host wait or reaped authority");
        return result;
    }
    const auto publicationDisposition = prepared.releasePublishedRoot(release);
    require(publicationDisposition == metalrobo::
                MetalNumanXHumanMatterPrepareLeaseDisposition::released,
            "COMMITTED publication was not released");
    require(!prepared.valid() && !audit.leaseActive.load(),
            "released publication retained active ownership");
    require(audit.releasePublicationCount == 1u &&
                audit.bindHumanIOCount == 1u &&
                audit.reserveHumanIOCount == 1u &&
                audit.publishHumanIOCount == 1u &&
                audit.rejectHumanIOCount == 0u,
            "publication callbacks were not exact-once");
    require(contents<MRNumanXHumanMatterOwnerStatusGPU>(
                audit.ownerStatuses)[0].stage ==
                MR_NUMANX_HUMAN_MATTER_STAGE_ROOT_PUBLISHED,
            "COMMITTED publication did not stamp ROOT_PUBLISHED");
    const auto terminalStats = context.stats();
    require(terminalStats.submissionDestructorWaitCount == 0u &&
                terminalStats.terminalSubmissionNonwaitingReapCount == 1u &&
                terminalStats.completedSubmissionCount == 1u &&
                !terminalStats.hasInFlightSubmission,
            "accepted root reaped through a host wait");
    return result;
}

void verifyReplay(const V4Result& left, const V4Result& right) {
    require(std::memcmp(&left.proposal, &right.proposal,
                        sizeof(left.proposal)) == 0 &&
                std::memcmp(&left.ack, &right.ack, sizeof(left.ack)) == 0 &&
                std::memcmp(&left.applied, &right.applied,
                            sizeof(left.applied)) == 0 &&
                std::memcmp(&left.fence, &right.fence,
                            sizeof(left.fence)) == 0 &&
                left.finalToken == right.finalToken &&
                left.q == right.q && left.v == right.v &&
                left.mujoco == right.mujoco,
            "ABI4 byte replay diverged");
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2,
                    "usage: numanx_human_matter_v4_probe <metallib>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "ABI4 probe requires Apple Metal");
            const auto accepted = runScenario(
                device, argv[1], V4Scenario::accept);
            const auto replay = runScenario(
                device, argv[1], V4Scenario::accept);
            verifyReplay(accepted, replay);
            (void)runScenario(device, argv[1], V4Scenario::physicalReject);
            (void)runScenario(device, argv[1], V4Scenario::forcedReject);
            (void)runScenario(
                device, argv[1],
                V4Scenario::physicalRejectForcedProposal);
            (void)runScenario(device, argv[1], V4Scenario::brainReject);
            (void)runScenario(
                device, argv[1], V4Scenario::invalidBrainWitness);
            (void)runScenario(device, argv[1], V4Scenario::tokenMismatch);
            (void)runScenario(device, argv[1], V4Scenario::staleAck);
            (void)runScenario(
                device, argv[1], V4Scenario::malformedPublicationRelease);
            (void)runScenario(
                device, argv[1], V4Scenario::proposalAbortRetry);
            (void)runScenario(
                device, argv[1], V4Scenario::proposalCallbackDrop);
            std::cout
                << "PASS device=\"" << device.name.UTF8String
                << "\" abi=4 control_step=37 dofs=160 q=161"
                << " proposal=mutation_free"
                << " proposal_completion=pre_post_abort_reentry_drop"
                << " preflight_ack=bound"
                << " apply=matter_then_human"
                << " accept=quarantined_then_published"
                << " human_io=post_physical_bound_delayed_publish"
                << " physical_reject=zero_gate_exact_code"
                << " physical_plus_force=physical_precedence"
                << " prepared_force_reject=exact_code"
                << " brain_reject=zero_gate_exact_code"
                << " invalid_brain_witness=brain_reject_exact_code"
                << " token_mismatch=zero_gate_exact_code"
                << " reject=byte_restored"
                << " stale_ack=fail_closed"
                << " malformed_release=terminal_no_touch"
                << " root_published=stamped"
                << " replay=byte_identical\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "numanx_human_matter_v4_probe: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
