#define main numanx_human_matter_adapter_probe_embedded_main
#include "numanx_human_matter_adapter_probe.mm"
#undef main

#include "metalrobo/NumanXExactTransaction.hpp"
#include "numi/matter/accepted_state_apply_gpu.h"
#include "numi/matter/accepted_state_proof_gpu.h"

namespace exact_v2_lifecycle_fixture {

using namespace adapter_fixture;

constexpr std::uint64_t kStartNanoseconds = 1'000'000'003ull;
constexpr std::uint64_t kDurationNanoseconds = 4'166'667ull;
constexpr std::uint64_t kDeliveryNanoseconds =
    kStartNanoseconds + kDurationNanoseconds;
constexpr std::uint32_t kControlStep = 73u;
constexpr std::uint64_t kPhysicsGeneration = 17u;
constexpr std::uint64_t kLinearizationEpoch = 0x4e584c4332000001ull;
constexpr std::uint64_t kSlotGeneration = 23u;
constexpr std::uint64_t kBrainProgramFingerprint =
    0x4e58425241494e32ull;
constexpr std::uint64_t kBrainShadowFingerprint =
    0x4e58534841444f57ull;
constexpr std::uint64_t kFastProgramFingerprint =
    0x4e58464153545032ull;
constexpr std::uint64_t kFastGateFingerprint =
    0x4e58464153544732ull;
constexpr std::uint64_t kJointCommitFingerprint =
    0x4e584a4f494e5432ull;

static_assert(kStartNanoseconds % 1000u != 0u);
static_assert(kDeliveryNanoseconds % 1000u != 0u);
static_assert(sizeof(MRNumanXExactInboundAuthorityGPUV2) ==
              sizeof(mrnx_exact_inbound_authority_v2));
static_assert(sizeof(MRNumanXAcceptedPhysicsStateTokenGPUV2) ==
              MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);

bool encodeRuntimeProofV2(
    void* context,
    const metalrobo::MetalNumanXHumanMatterStateProofPassV2& source
) noexcept {
    auto* runtime = static_cast<numi::matter::Runtime*>(context);
    if (runtime == nullptr ||
        source.abiVersion !=
            MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION ||
        source.structSize != sizeof(source)) {
        return false;
    }
    numi::matter::AcceptedStateProofPassV2 pass{};
    pass.environmentCount = source.environmentCount;
    pass.environmentIdentifierBase = source.environmentIdentifierBase;
    pass.commandBuffer = source.commandBuffer;
    pass.rootTranslation = source.rootTranslation;
    pass.q = source.q;
    pass.v = source.v;
    pass.mujocoStates = source.mujocoStates;
    pass.matterGeneralizedReaction = source.matterGeneralizedReaction;
    pass.environmentStatuses = source.environmentStatuses;
    pass.matterStatuses = source.matterStatuses;
    pass.acceptedStateProofs = source.acceptedStateProofs;
    pass.inboundAuthority = source.inboundAuthority;
    pass.rootTranslationGPUAddress = source.rootTranslationGPUAddress;
    pass.qGPUAddress = source.qGPUAddress;
    pass.vGPUAddress = source.vGPUAddress;
    pass.mujocoStatesGPUAddress = source.mujocoStatesGPUAddress;
    pass.matterGeneralizedReactionGPUAddress =
        source.matterGeneralizedReactionGPUAddress;
    pass.environmentStatusesGPUAddress =
        source.environmentStatusesGPUAddress;
    pass.matterStatusesGPUAddress = source.matterStatusesGPUAddress;
    pass.acceptedStateProofsGPUAddress =
        source.acceptedStateProofsGPUAddress;
    pass.inboundAuthorityGPUAddress = source.inboundAuthorityGPUAddress;
    pass.rootTranslationElementCount = source.rootTranslationElementCount;
    pass.qElementCount = source.qElementCount;
    pass.vElementCount = source.vElementCount;
    pass.mujocoStateCount = source.mujocoStateCount;
    pass.matterGeneralizedReactionElementCount =
        source.matterGeneralizedReactionElementCount;
    pass.environmentStatusElementCount =
        source.environmentStatusElementCount;
    pass.matterStatusElementCount = source.matterStatusElementCount;
    pass.acceptedStateProofElementCount =
        source.acceptedStateProofElementCount;
    pass.inboundAuthorityByteCount = source.inboundAuthorityByteCount;
    pass.rootTranslationStride = source.rootTranslationStride;
    pass.qStride = source.qStride;
    pass.vStride = source.vStride;
    pass.mujocoStateStride = source.mujocoStateStride;
    pass.reactionStride = source.reactionStride;
    pass.environmentStatusStride = source.environmentStatusStride;
    pass.matterStatusStride = source.matterStatusStride;
    pass.acceptedStateProofStride = source.acceptedStateProofStride;
    pass.qCoordinateCount = source.qCoordinateCount;
    pass.dofCount = source.dofCount;
    pass.transactionSlot = source.transactionSlot;
    pass.clockDomain = source.clockDomain;
    pass.clockQuantumNanoseconds = source.clockQuantumNanoseconds;
    pass.reserved0 = source.reserved0;
    pass.programFingerprint = source.programFingerprint;
    pass.stateProofProgramFingerprint = source.stateProofProgramFingerprint;
    pass.transactionFingerprint = source.transactionFingerprint;
    pass.substepFingerprint = source.substepFingerprint;
    pass.acceptedTimestampNanoseconds = source.acceptedTimestampNanoseconds;
    pass.physicsGeneration = source.physicsGeneration;
    pass.linearizationEpoch = source.linearizationEpoch;
    pass.slotGeneration = source.slotGeneration;
    pass.matterSourcePhysicsFingerprint =
        source.matterSourcePhysicsFingerprint;
    pass.matterDeviceProgramFingerprint =
        source.matterDeviceProgramFingerprint;
    pass.motorCandidateFingerprint = source.motorCandidateFingerprint;
    return runtime->encodeAcceptedStateProofV2(pass);
}

numi::matter::CompiledWorld compileSettledLifecycleWorld() {
    const auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "lifecycle Matter material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 240.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.articulatedDofCapacity = kDofs;
    source.articulatedQCapacity = kQ;
    source.materials.push_back(parsed.material);
    numi::matter::ObjectSource object;
    object.name = "numanx_exact_v2_lifecycle_settled_fem";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.characteristicLength = 0.01;
    object.mixedFEM = false;
    object.femNodes = {
        {-0.005, -0.005, -0.005}, {0.005, -0.005, -0.005},
        {-0.005, 0.005, -0.005}, {-0.005, -0.005, 0.005},
    };
    object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    // The lifecycle normalizes the Human fixture to an origin/identity root.
    // Bind every tetrahedron node to that body at its exact rest location.
    // This retains real coupled-candidate ownership without free rigid modes
    // or an initial attachment strain.
    constexpr std::array<std::array<double, 3u>, 4u> localPoints{{
        {{-0.005, -0.005, -0.005}},
        {{0.005, -0.005, -0.005}},
        {{-0.005, 0.005, -0.005}},
        {{-0.005, -0.005, 0.005}},
    }};
    for (std::uint32_t node = 0u; node < localPoints.size(); ++node) {
        numi::matter::FEMHumanAttachmentSource attachment;
        attachment.node = node;
        attachment.bodyIndex = owner_fixture::kFirstBody;
        attachment.stableIdentifier = 0x4e584c32u + node;
        attachment.localPoint = localPoints[node];
        object.femHumanAttachments.push_back(attachment);
    }
    source.objects.push_back(std::move(object));
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(),
        "settled lifecycle Matter world did not compile");
    std::string error;
    require(numi::matter::validateCompiledWorldLayout(compiled.world, &error),
        "settled lifecycle Matter layout failed: " + error);
    return std::move(compiled.world);
}

struct HumanIOFixture {
    id<MTLBuffer> header = nil;
    id<MTLBuffer> excitation = nil;
    id<MTLBuffer> autonomic = nil;
    id<MTLBuffer> activeSensing = nil;
    id<MTLBuffer> readyGate = nil;
    id<MTLSharedEvent> readyEvent = nil;
    MRNumanXBrainJointTransactionTokenV2 root{};
    MRNumanXBrainJointSubstepTokenV2 substep{};
    MRNumanXBrainMotorCandidateV2 candidate{};
    MRNumanXBrainMotorOutputHeaderGPUV2 output{};
    MRNumanXBrainMotorReadyGateGPUV2 gate{};
    metalrobo::MetalNumanXHumanIOInputV2 input{};
};

HumanIOFixture makeHumanIOFixture(id<MTLDevice> device) {
    HumanIOFixture fixture;
    fixture.header = makeZeroBuffer(
        device, sizeof(fixture.output), @"lifecycle exact motor header");
    fixture.excitation = makeZeroBuffer(
        device, sizeof(float), @"lifecycle exact excitation");
    fixture.autonomic = makeZeroBuffer(
        device, MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT,
        @"lifecycle exact autonomic command");
    fixture.activeSensing = makeZeroBuffer(
        device, MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT,
        @"lifecycle exact active-sensing command");
    fixture.readyGate = makeZeroBuffer(
        device, sizeof(fixture.gate), @"lifecycle exact ready gate");
    fixture.readyEvent = [device newSharedEvent];
    require(fixture.readyEvent != nil,
        "failed to allocate exact HumanIO ready event");
    // Keep the owner fixture at its calibrated zero-activation root. The
    // lifecycle under test is authority transport, not a new motor policy.
    *static_cast<float*>(fixture.excitation.contents) = 0.0f;

    fixture.root.formatVersion =
        MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
    fixture.root.episodeIdentifier = 7u;
    fixture.root.controlStepIdentifier = kControlStep;
    fixture.root.parameterVersionFingerprint = 0x4e58504152414d32ull;
    fixture.root.baseBrainGeneration = 18u;
    fixture.root.basePhysicsGeneration = 16u;
    fixture.root.committedTimestampNanoseconds = kStartNanoseconds;
    fixture.root.targetTimestampNanoseconds = kDeliveryNanoseconds;
    fixture.root.shadowGeneration = 19u;
    fixture.root.randomCounterGeneration = 29u;
    fixture.root.clockDomain =
        MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    fixture.root.clockQuantumNanoseconds =
        MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    fixture.root.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(
            fixture.root);

    fixture.substep.transactionFingerprint =
        fixture.root.transactionFingerprint;
    fixture.substep.startTimestampNanoseconds = kStartNanoseconds;
    fixture.substep.durationNanoseconds = kDurationNanoseconds;
    fixture.substep.candidateTimestampNanoseconds = kDeliveryNanoseconds;
    fixture.substep.shadowGeneration = fixture.root.shadowGeneration;
    fixture.substep.randomCounterGeneration =
        fixture.root.randomCounterGeneration;
    fixture.substep.clockDomain = fixture.root.clockDomain;
    fixture.substep.clockQuantumNanoseconds =
        fixture.root.clockQuantumNanoseconds;
    fixture.substep.substepFingerprint =
        metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(
            fixture.substep);

    fixture.candidate.formatVersion =
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
    fixture.candidate.flags = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
    fixture.candidate.transactionFingerprint =
        fixture.root.transactionFingerprint;
    fixture.candidate.substepFingerprint =
        fixture.substep.substepFingerprint;
    fixture.candidate.acceptedBrainTimestampNanoseconds =
        kStartNanoseconds;
    fixture.candidate.brainGeneration = fixture.root.shadowGeneration;
    fixture.candidate.motorProfileFingerprint = 0x4e584d50524f4632ull;
    fixture.candidate.motorOutputHeaderGPUAddress =
        fixture.header.gpuAddress;
    fixture.candidate.muscleExcitationGPUAddress =
        fixture.excitation.gpuAddress;
    fixture.candidate.randomCounterGeneration =
        fixture.root.randomCounterGeneration;
    fixture.candidate.motorOutputHeaderByteCount = sizeof(fixture.output);
    fixture.candidate.muscleExcitationByteCount = sizeof(float);
    fixture.candidate.muscleCount = 1u;
    fixture.candidate.autonomicCommandGPUAddress =
        fixture.autonomic.gpuAddress;
    fixture.candidate.autonomicCommandByteCount =
        fixture.autonomic.length;
    fixture.candidate.autonomicCommandCount = 1u;
    fixture.candidate.activeSensingCommandGPUAddress =
        fixture.activeSensing.gpuAddress;
    fixture.candidate.activeSensingCommandByteCount =
        fixture.activeSensing.length;
    fixture.candidate.activeSensingCommandCount = 1u;
    fixture.candidate.actuatorCommandKind =
        MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
    fixture.candidate.clockDomain = fixture.root.clockDomain;
    fixture.candidate.speciesTemplateFingerprint =
        0x4e58535045434945ull;
    fixture.candidate.compiledSpeciesTemplateFingerprint =
        0x4e58434f4d50494cull;
    fixture.candidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
            fixture.candidate);

    fixture.output.formatVersion =
        MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
    fixture.output.flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
    fixture.output.timestampNanoseconds = kStartNanoseconds;
    fixture.output.brainGeneration = fixture.candidate.brainGeneration;
    fixture.output.profileFingerprint =
        fixture.candidate.motorProfileFingerprint;
    fixture.output.protectiveCommandFingerprint =
        0x4e5850524f544543ull;
    fixture.output.muscleCount = 1u;
    fixture.output.environmentIdentifier = 0u;
    fixture.output.autonomicArousal = 0.25f;
    fixture.output.actuatorCommandKind =
        fixture.candidate.actuatorCommandKind;
    fixture.output.clockDomain = fixture.root.clockDomain;
    fixture.output.outputMaximum = 1.0f;
    fixture.output.outputFingerprint =
        metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
            fixture.output,
            static_cast<const float*>(fixture.excitation.contents),
            1u);

    fixture.gate.abiVersion =
        MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION_V2;
    fixture.gate.structBytes = sizeof(fixture.gate);
    fixture.gate.status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
    fixture.gate.environment = 0u;
    fixture.gate.muscleCount = 1u;
    fixture.gate.actuatorCommandKind =
        fixture.candidate.actuatorCommandKind;
    fixture.gate.controlStep = kControlStep;
    fixture.gate.transactionFingerprint =
        fixture.root.transactionFingerprint;
    fixture.gate.substepFingerprint = fixture.substep.substepFingerprint;
    fixture.gate.candidateFingerprint = fixture.candidate.candidateFingerprint;
    fixture.gate.motorOutputFingerprint = fixture.output.outputFingerprint;
    fixture.gate.motorProfileFingerprint =
        fixture.candidate.motorProfileFingerprint;
    fixture.gate.brainGeneration = fixture.candidate.brainGeneration;
    fixture.gate.acceptedBrainTimestampNanoseconds = kStartNanoseconds;
    fixture.gate.randomCounterGeneration =
        fixture.candidate.randomCounterGeneration;
    fixture.gate.speciesTemplateFingerprint =
        fixture.candidate.speciesTemplateFingerprint;
    fixture.gate.compiledSpeciesTemplateFingerprint =
        fixture.candidate.compiledSpeciesTemplateFingerprint;
    fixture.gate.brainProgramFingerprint = kBrainProgramFingerprint;
    fixture.gate.fastProgramFingerprint = kFastProgramFingerprint;
    fixture.gate.decisionGateFingerprint = 0x4e58444543495332ull;
    fixture.gate.clockDomain = fixture.root.clockDomain;
    fixture.gate.clockQuantumNanoseconds =
        fixture.root.clockQuantumNanoseconds;
    fixture.gate.gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(
            fixture.gate);
    std::memcpy(fixture.header.contents, &fixture.output,
                sizeof(fixture.output));
    std::memcpy(fixture.readyGate.contents, &fixture.gate,
                sizeof(fixture.gate));
    fixture.readyEvent.signaledValue = 1u;

    fixture.input.root = fixture.root;
    fixture.input.substep = fixture.substep;
    fixture.input.candidate = fixture.candidate;
    fixture.input.motorOutputHeaderMetalBuffer =
        (__bridge void*)fixture.header;
    fixture.input.motorOutputHeaderByteCount = sizeof(fixture.output);
    fixture.input.motorOutputHeaderEnvironmentStride =
        sizeof(fixture.output);
    fixture.input.expectedMotorOutputHeaderGPUAddress =
        fixture.header.gpuAddress;
    fixture.input.excitationMetalBuffer = (__bridge void*)fixture.excitation;
    fixture.input.excitationByteCount = sizeof(float);
    fixture.input.excitationEnvironmentStride = 1u;
    fixture.input.expectedExcitationGPUAddress =
        fixture.excitation.gpuAddress;
    fixture.input.autonomicCommandMetalBuffer =
        (__bridge void*)fixture.autonomic;
    fixture.input.autonomicCommandByteCount = fixture.autonomic.length;
    fixture.input.expectedAutonomicCommandGPUAddress =
        fixture.autonomic.gpuAddress;
    fixture.input.activeSensingCommandMetalBuffer =
        (__bridge void*)fixture.activeSensing;
    fixture.input.activeSensingCommandByteCount =
        fixture.activeSensing.length;
    fixture.input.expectedActiveSensingCommandGPUAddress =
        fixture.activeSensing.gpuAddress;
    fixture.input.motorReadyGateMetalBuffer =
        (__bridge void*)fixture.readyGate;
    fixture.input.motorReadyGateByteCount = sizeof(fixture.gate);
    fixture.input.expectedMotorReadyGateGPUAddress =
        fixture.readyGate.gpuAddress;
    fixture.input.motorReadySharedEvent = (__bridge void*)fixture.readyEvent;
    fixture.input.motorReadySharedEventValue = 1u;
    fixture.input.environmentCount = 1u;
    fixture.input.muscleCount = 1u;
    fixture.input.stepCount = 1u;
    fixture.input.timestepNanoseconds = kDurationNanoseconds;
    fixture.input.receptorTimestampNanoseconds = kStartNanoseconds;
    fixture.input.candidateSensorGeneration = 31u;
    return fixture;
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
        hash = fnvByte(hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

std::uint64_t fnvU64(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
        hash = fnvByte(hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
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
    return hash == 0u ? 14695981039346656037ull : hash;
}

numi::matter::PreparedStateDispositionIdentity dispositionIdentity(
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program
) {
    numi::matter::PreparedStateDispositionIdentity result{};
    result.controlStep = transaction.controlStep;
    result.physicsSubstep = transaction.physicsSubstep;
    result.physicsSubstepCount = transaction.physicsSubsteps;
    result.transactionSlot = transaction.transactionSlot;
    result.ownerProgramFingerprint = program.fingerprint;
    result.transactionFingerprint = transaction.transactionFingerprint;
    result.linearizationEpoch = transaction.linearizationEpoch;
    result.slotGeneration = transaction.slotGeneration;
    return result;
}

template <typename Pass>
Pass makeApplyAdmissionPass(
    id<MTLCommandBuffer> command,
    id<MTLBuffer> proposals,
    id<MTLBuffer> acknowledgements,
    id<MTLBuffer> actions,
    id<MTLBuffer> outcomes,
    id<MTLBuffer> token,
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program
) {
    Pass pass{};
    pass.environmentCount = 1u;
    pass.environmentIdentifierBase = transaction.environmentIdentifierBase;
    pass.controlStep = transaction.controlStep;
    pass.physicsSubstep = transaction.physicsSubstep;
    pass.physicsSubstepCount = transaction.physicsSubsteps;
    pass.transactionSlot = transaction.transactionSlot;
    pass.commandBuffer = (__bridge void*)command;
    pass.proposals = (__bridge void*)proposals;
    pass.brainAcks = (__bridge void*)acknowledgements;
    pass.applyActions = (__bridge void*)actions;
    pass.matterApplyOutcomes = (__bridge void*)outcomes;
    pass.proposedPhysicsStateTokens = (__bridge void*)token;
    pass.proposalsGPUAddress = proposals.gpuAddress;
    pass.brainAcksGPUAddress = acknowledgements.gpuAddress;
    pass.applyActionsGPUAddress = actions.gpuAddress;
    pass.matterApplyOutcomesGPUAddress = outcomes.gpuAddress;
    pass.proposedPhysicsStateTokensGPUAddress = token.gpuAddress;
    pass.proposalElementCount = 1u;
    pass.brainAckElementCount = 1u;
    pass.applyActionElementCount = 1u;
    pass.matterApplyOutcomeElementCount = 1u;
    pass.proposedPhysicsStateTokenBytes = token.length;
    pass.proposalStride = 1u;
    pass.brainAckStride = 1u;
    pass.applyActionStride = 1u;
    pass.matterApplyOutcomeStride = 1u;
    pass.proposedPhysicsStateTokenStrideBytes =
        MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
    pass.ownerProgramFingerprint = program.fingerprint;
    pass.transactionFingerprint = transaction.transactionFingerprint;
    pass.linearizationEpoch = transaction.linearizationEpoch;
    pass.slotGeneration = transaction.slotGeneration;
    return pass;
}

void requireApplyAdmissionNegatives(
    numi::matter::Runtime& matter,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLBuffer> token,
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program
) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "failed to allocate family-negative command");
    id<MTLBuffer> proposals = makeZeroBuffer(
        device, sizeof(NMOwnerProposalGPU), @"lifecycle family proposal");
    id<MTLBuffer> acknowledgements = makeZeroBuffer(
        device, sizeof(NMOwnerBrainAckGPU), @"lifecycle family ACK");
    id<MTLBuffer> actions = makeZeroBuffer(
        device, sizeof(NMOwnerApplyActionGPU), @"lifecycle family action");
    id<MTLBuffer> outcomes = makeZeroBuffer(
        device, sizeof(NMMatterApplyOutcomeGPU), @"lifecycle family outcome");
    auto legacy = makeApplyAdmissionPass<numi::matter::AcceptedStateApplyPass>(
        command, proposals, acknowledgements, actions, outcomes, token,
        transaction, program);
    require(!matter.applyPreparedState(legacy),
        "legacy Matter apply admitted the exact prepared family");
    auto wrongClock =
        makeApplyAdmissionPass<numi::matter::AcceptedStateApplyPassV2>(
            command, proposals, acknowledgements, actions, outcomes, token,
            transaction, program);
    wrongClock.clockQuantumNanoseconds = 2u;
    require(!matter.applyPreparedStateV2(wrongClock),
        "exact Matter apply admitted a non-exact clock quantum");
    require(matter.preparedStateDisposition(
                dispositionIdentity(transaction, program)) ==
            numi::matter::PreparedStateDisposition::prepared,
        "apply-admission negatives changed prepared Matter authority");
}

MRNumanXHumanMatterBrainCommitWitnessGPU makeWitness(
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program,
    const std::uint64_t tokenFingerprint
) {
    MRNumanXHumanMatterBrainCommitWitnessGPU witness{};
    witness.magic = MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_MAGIC;
    witness.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_ABI_VERSION;
    witness.structBytes =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_BYTES;
    witness.status =
        MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_COMPLETE;
    witness.decision = MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    witness.environment = 0u;
    witness.stepIndex = 0u;
    witness.substepIndex = transaction.physicsSubstep;
    witness.transactionSlot = transaction.transactionSlot;
    witness.physicsSubstepCount = transaction.physicsSubsteps;
    witness.controlStep = transaction.controlStep;
    witness.programFingerprint = program.fingerprint;
    witness.transactionFingerprint = transaction.transactionFingerprint;
    witness.linearizationEpoch = transaction.linearizationEpoch;
    witness.slotGeneration = transaction.slotGeneration;
    witness.physicsTokenFingerprint = tokenFingerprint;
    witness.brainProgramFingerprint = kBrainProgramFingerprint;
    witness.brainShadowStateFingerprint = kBrainShadowFingerprint;
    witness.witnessFingerprint = witnessFingerprint(witness);
    return witness;
}

MRNumanXHumanMatterBrainCommitPreflightGPU makePreflight(
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program,
    const MRNumanXHumanMatterProposalGPU& proposal
) {
    MRNumanXHumanMatterBrainCommitPreflightGPU preflight{};
    preflight.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION;
    preflight.structBytes =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES;
    preflight.status = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS;
    preflight.environment = 0u;
    preflight.controlStep = transaction.controlStep;
    preflight.substepIndex = transaction.physicsSubstep;
    preflight.physicsSubstepCount = transaction.physicsSubsteps;
    preflight.transactionSlot = transaction.transactionSlot;
    preflight.ownerProgramFingerprint = program.fingerprint;
    preflight.transactionFingerprint = transaction.transactionFingerprint;
    preflight.linearizationEpoch = transaction.linearizationEpoch;
    preflight.slotGeneration = transaction.slotGeneration;
    preflight.substepFingerprint = transaction.substepFingerprint;
    preflight.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    preflight.fastTargetGeneration = 41u;
    preflight.cognitiveTargetGeneration = 43u;
    preflight.jointReceiptFingerprint = 0x4e584a4f494e5432ull;
    preflight.fastProgramFingerprint = kFastProgramFingerprint;
    preflight.brainProgramFingerprint = kBrainProgramFingerprint;
    preflight.preflightFingerprint = recordFingerprint(preflight);
    return preflight;
}

MRNumanXHumanMatterBrainAckGPU makeAck(
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterBrainCommitPreflightGPU& preflight
) {
    MRNumanXHumanMatterBrainAckGPU ack{};
    ack.abiVersion = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION;
    ack.status = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT;
    ack.decision = MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    ack.code = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS;
    ack.programFingerprint = program.fingerprint;
    ack.transactionFingerprint = transaction.transactionFingerprint;
    ack.linearizationEpoch = transaction.linearizationEpoch;
    ack.slotGeneration = transaction.slotGeneration;
    ack.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    ack.proposalFingerprint = proposal.proposalFingerprint;
    ack.preflightFingerprint = preflight.preflightFingerprint;
    ack.fastGateFingerprint = kFastGateFingerprint;
    ack.brainWitnessFingerprint = proposal.brainWitnessFingerprint;
    ack.brainProgramFingerprint = proposal.brainProgramFingerprint;
    ack.environment = 0u;
    ack.stepIndex = 0u;
    ack.substepIndex = transaction.physicsSubstep;
    ack.transactionSlot = transaction.transactionSlot;
    ack.physicsSubstepCount = transaction.physicsSubsteps;
    ack.controlStep = transaction.controlStep;
    ack.ackFingerprint = recordFingerprint(ack);
    return ack;
}

bool validExactTokenShape(
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const metalrobo::MetalNumanXHumanMatterTransactionV2& transaction,
    const std::uint64_t acceptedTimestampNanoseconds
) {
    return token.transactionFingerprint == transaction.transactionFingerprint &&
        token.substepFingerprint == transaction.substepFingerprint &&
        token.physicsStateFingerprint != 0u &&
        token.acceptedTimestampNanoseconds == acceptedTimestampNanoseconds &&
        token.physicsGeneration == transaction.physicsGeneration &&
        token.environmentIdentifier == transaction.environmentIdentifierBase &&
        token.flags == 0u &&
        token.clockDomain ==
            MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        token.clockQuantumNanoseconds ==
            MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        token.tokenFingerprint != 0u &&
        token.tokenFingerprint ==
            metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(
                token);
}

void runLifecycle(id<MTLDevice> device) {
    auto world = compileSettledLifecycleWorld();
    numi::matter::RuntimeConfiguration runtimeConfig;
    runtimeConfig.metallib = NUMI_MATTER_METALLIB;
    runtimeConfig.environmentCount = 1u;
    runtimeConfig.captureEvents = false;
    runtimeConfig.captureDiagnostics = true;
    runtimeConfig.adaptiveTransfer = false;
    runtimeConfig.coupledCandidateCompensatedTranslation = true;
    runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        sizeof(MRMujocoMuscleStateGPU);
    auto* matter = new numi::matter::Runtime;
    const auto matterInit = matter->initialize(world, runtimeConfig);
    require(matterInit.encoded,
        "exact lifecycle Matter initialization failed: " +
            matterInit.message);

    metalrobo::MetalNumanXHumanMatterConfig adapterConfig;
    adapterConfig.matterRuntime = matter;
    adapterConfig.coupledHumanMetallibPath = NUMANX_ADAPTER_METALLIB;
    adapterConfig.adapterMetallibPath = NUMANX_ADAPTER_METALLIB;
    adapterConfig.environmentCapacity = 1u;
    adapterConfig.pointCapacity = 4u;
    adapterConfig.transactionSlotCount = 1u;
    adapterConfig.stateProofProgram.context = matter;
    adapterConfig.stateProofProgram.encode = &encodeRuntimeProof;
    adapterConfig.stateProofProgram.fingerprint =
        matter->acceptedStateProofProgramFingerprint();
    adapterConfig.stateProofProgramV2.context = matter;
    adapterConfig.stateProofProgramV2.encode = &encodeRuntimeProofV2;
    adapterConfig.stateProofProgramV2.fingerprint =
        matter->acceptedStateProofProgramFingerprintV2();
    auto* adapter = new metalrobo::MetalNumanXHumanMatterContext(
        adapterConfig);
    const auto adapterInit = adapter->initialize();
    require(adapterInit.succeeded() &&
                adapterInit.acceptedStateProofAvailable &&
                adapterInit.acceptedStateProofV2Available,
        "exact lifecycle adapter initialization failed: " +
            adapterInit.message);

    auto* humanIO = new metalrobo::MetalNumanXHumanIOContext({
        .metallibPath = NUMANX_ADAPTER_METALLIB,
    });
    HumanIOFixture ioFixture = makeHumanIOFixture(device);
    metalrobo::MetalNumanXTransactionProgram humanIOProgram{};
    metalrobo::MetalNumanXHumanIOExactPreparedView exactHumanIO{};
    const auto ioPrepared = humanIO->prepare(
        ioFixture.input, humanIOProgram, exactHumanIO);
    require(ioPrepared.succeeded() && humanIOProgram.valid() &&
                exactHumanIO.valid(),
        "live exact HumanIO prepare failed: " + ioPrepared.message);
    require(((__bridge id<MTLBuffer>)
                exactHumanIO.authority.metalBuffer).storageMode ==
                    MTLStorageModePrivate &&
                exactHumanIO.authority.acceptedBrainTimestampNanoseconds ==
                    kStartNanoseconds &&
                exactHumanIO.sensor.deliveryTimestampNanoseconds ==
                    kDeliveryNanoseconds,
        "live HumanIO did not expose exact private receipt timing");

    auto model = owner_fixture::makeHumanModel();
    std::fill(model.defaultQ.begin(), model.defaultQ.end(), 0.0f);
    model.defaultQ[6u] = 1.0f;
    std::fill(model.defaultV.begin(), model.defaultV.end(), 0.0f);
    metalrobo::MetalNumanXHumanMatterTransactionV2 transaction{};
    transaction.environmentCount = 1u;
    transaction.environmentIdentifierBase = 0u;
    transaction.transactionSlot = 0u;
    transaction.controlStep = kControlStep;
    transaction.physicsSubstep = 0u;
    transaction.physicsSubsteps = 1u;
    transaction.expectedMatterCompletedMicrosteps = 1u;
    transaction.qCoordinateCount = model.articulations[0u].nq;
    transaction.dofCount = model.articulations[0u].nv;
    transaction.seed = 0x4e585345454432ull;
    transaction.transactionFingerprint =
        ioFixture.root.transactionFingerprint;
    transaction.substepFingerprint = ioFixture.substep.substepFingerprint;
    transaction.physicsGeneration = kPhysicsGeneration;
    transaction.linearizationEpoch = kLinearizationEpoch;
    transaction.slotGeneration = kSlotGeneration;

    auto wrongClockView = exactHumanIO;
    wrongClockView.authority.clockQuantumNanoseconds = 2u;
    wrongClockView.authority.rangeIdentityFingerprint = metalrobo::
        metalNumanXHumanIOExactAuthorityRangeIdentityFingerprint(
            wrongClockView.authority);
    require(!adapter->program(transaction, wrongClockView).valid(),
        "adapter admitted a non-exact HumanIO clock quantum");
    auto wrongFamilyView = exactHumanIO;
    wrongFamilyView.authority.abiVersion = 1u;
    wrongFamilyView.authority.rangeIdentityFingerprint = metalrobo::
        metalNumanXHumanIOExactAuthorityRangeIdentityFingerprint(
            wrongFamilyView.authority);
    require(!adapter->program(transaction, wrongFamilyView).valid(),
        "adapter admitted a legacy-tagged HumanIO authority family");
    auto wrongTokenIdentity = transaction;
    wrongTokenIdentity.substepFingerprint ^= 1u;
    require(!adapter->program(wrongTokenIdentity, exactHumanIO).valid(),
        "adapter admitted a mismatched exact substep identity");

    const auto humanMatterProgram = adapter->program(transaction, exactHumanIO);
    require(humanMatterProgram.valid() &&
                humanMatterProgram.tokenFamily == metalrobo::
                    MetalNumanXHumanMatterTokenFamily::exactNanosecondsV2 &&
                humanMatterProgram.humanIOProgramFingerprint ==
                    humanIOProgram.fingerprint,
        "exact HumanMatter program lost its HumanIO family binding");
    auto wrongProgramFamily = humanMatterProgram;
    wrongProgramFamily.tokenFamily = metalrobo::
        MetalNumanXHumanMatterTokenFamily::legacyMicrosecondsV1;
    require(!wrongProgramFamily.valid(),
        "exact HumanMatter program remained valid under a V1 token tag");

    owner_fixture::CandidateAudit ownerFixture;
    owner_fixture::initializeAudit(ownerFixture, device);
    const auto ownerPoints = owner_fixture::bodyProbes();
    auto input = owner_fixture::makeInput(
        model, ownerPoints, ownerFixture);
    input.stand.numanXTransactionProgram = humanIOProgram;
    input.stand.numanXHumanMatterProgram = humanMatterProgram;
    const float exactTimestepSeconds = static_cast<float>(
        static_cast<double>(kDurationNanoseconds) * 1.0e-9);
    require(exactTimestepSeconds == matter->timestepSeconds(),
        "integer-nanosecond HumanIO cadence differs from Matter cadence");
    auto* owner = new metalrobo::MetalArticulatedOperatorContext({
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = exactTimestepSeconds,
        .metallibPath = NUMANX_ADAPTER_METALLIB,
    });
    metalrobo::MetalArticulatedOperatorSubmission submission;
    const auto submitted = owner->submit(model, input, submission);
    require(submitted.succeeded() && submitted.dispatched &&
                submission.valid(),
        "owner rejected exact HumanIO+HumanMatter prepare: " +
            submitted.message);
    auto* prepared = new metalrobo::MetalNumanXHumanMatterPrepared;
    require(submission.extractPreparedHumanMatter(*prepared) &&
                prepared->valid(),
        "owner did not expose exact prepared lifecycle capability");
    metalrobo::MetalNumanXHumanMatterPreparedView view{};
    require(prepared->view(view) &&
                view.programFingerprint == humanMatterProgram.fingerprint &&
                view.transactionFingerprint ==
                    transaction.transactionFingerprint,
        "exact prepared view lost adapter identity");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "failed to allocate lifecycle command queue");
    id<MTLCommandBuffer> physicalWait = [queue commandBuffer];
    require(physicalWait != nil && prepared->encodeWaitForPhysicalPrepare(
                (__bridge void*)physicalWait),
        "failed to encode exact physical-prepare wait");
    finish(physicalWait);

    metalrobo::MetalNumanXHumanIOTransactionKey key{};
    metalrobo::MetalNumanXHumanIOSensorView sensor{};
    const auto pending = humanIO->pendingCandidate(key, sensor);
    require(pending.succeeded() && key.valid() &&
                key.programFingerprint == humanIOProgram.fingerprint &&
                sensor.deliveryTimestampNanoseconds == kDeliveryNanoseconds,
        "live exact HumanIO candidate did not settle with owner command");
    metalrobo::MetalNumanXHumanIOCandidatePublicationLease legacyLease{};
    const auto exactPublication = humanIO->reserveCandidatePublication(
        key, legacyLease);
    require(exactPublication.status == metalrobo::
                MetalNumanXHumanIOStatus::candidateUnavailable &&
                !legacyLease.valid(),
        "exact HumanIO escaped through the legacy publication ABI");

    auto wrongAuthority = exactHumanIO.authority;
    ++wrongAuthority.acceptedBrainTimestampNanoseconds;
    wrongAuthority.rangeIdentityFingerprint = metalrobo::
        metalNumanXHumanIOExactAuthorityRangeIdentityFingerprint(
            wrongAuthority);
    require(wrongAuthority.valid(),
        "mutated exact HumanIO authority was not internally canonical");
    metalrobo::MetalNumanXHumanIOCandidatePublicationLease wrongLease{};
    const auto wrongReservation = humanIO->reserveCandidatePublication(
        key, wrongAuthority, wrongLease);
    require(wrongReservation.status == metalrobo::
                MetalNumanXHumanIOStatus::incompatibleTransaction &&
                !wrongLease.valid(),
        "exact HumanIO admitted a different private authority receipt");

    metalrobo::MetalNumanXHumanIOCandidatePublicationLease publicationLease{};
    const auto publicationReservation = humanIO->reserveCandidatePublication(
        key, exactHumanIO.authority, publicationLease);
    require(publicationReservation.succeeded() &&
                !publicationReservation.published &&
                publicationLease.valid() &&
                publicationLease.program().valid() &&
                publicationLease.program().abiVersion == metalrobo::
                    kMetalNumanXHumanIOExactPublicationABIVersion &&
                publicationLease.view().abiVersion == metalrobo::
                    kMetalNumanXHumanIOExactPublicationABIVersion &&
                publicationLease.program().humanIOProgramFingerprint ==
                    humanIOProgram.fingerprint &&
                publicationLease.program().transactionFingerprint ==
                    transaction.transactionFingerprint &&
                publicationLease.program().acceptedBrainGeneration ==
                    ioFixture.candidate.brainGeneration &&
                publicationLease.view().sensor.deliveryTimestampNanoseconds ==
                    kDeliveryNanoseconds,
        "exact HumanIO did not issue an ABI2 authority-bound publication lease");
    const auto publicationProgram = publicationLease.program();
    const auto publicationSensor = publicationLease.view().sensor;
    auto downgradedPublicationProgram = publicationProgram;
    downgradedPublicationProgram.abiVersion = metalrobo::
        kMetalNumanXHumanIOPublicationABIVersion;
    downgradedPublicationProgram.identityFingerprint =
        downgradedPublicationProgram.computedIdentityFingerprint();
    require(downgradedPublicationProgram.valid() &&
                !prepared->bindHumanIOCandidatePublication(
                    downgradedPublicationProgram),
        "exact HumanMatter admitted an ABI1 publication downgrade");
    require(prepared->bindHumanIOCandidatePublication(publicationProgram) &&
                !prepared->bindHumanIOCandidatePublication(
                    publicationProgram) &&
                prepared->view(view) &&
                view.humanIOCandidateKeyFingerprint ==
                    publicationProgram.candidateKeyFingerprint &&
                view.acceptedBrainGeneration ==
                    publicationProgram.acceptedBrainGeneration &&
                view.humanIOSensorGeneration ==
                    publicationProgram.sensorGeneration &&
                view.humanIOProgramFingerprint ==
                    publicationProgram.humanIOProgramFingerprint &&
                view.humanIOSensorFingerprint ==
                    publicationProgram.sensorFingerprint &&
                view.humanIOTransactionInstanceFingerprint ==
                    publicationProgram.transactionInstanceFingerprint &&
                view.humanIOCandidatePublicationFingerprint ==
                    publicationProgram.candidatePublicationFingerprint &&
                view.humanIODeviceRegistryID == device.registryID &&
                view.humanIOIdentityFingerprint ==
                    publicationProgram.identityFingerprint,
        "exact HumanMatter did not retain the ABI2 HumanIO publication identity");
    metalrobo::MetalNumanXHumanIOSensorView unpublishedSensor{};
    require(humanIO->publishedView(unpublishedSensor).status == metalrobo::
                MetalNumanXHumanIOStatus::candidateUnavailable,
        "exact HumanIO sensor became visible before root publication");

    id<MTLBuffer> tokenReadback = makeZeroBuffer(
        device, sizeof(MRNumanXAcceptedPhysicsStateTokenGPUV2),
        @"lifecycle exact token readback");
    id<MTLCommandBuffer> tokenReadbackCommand = [queue commandBuffer];
    id<MTLBlitCommandEncoder> tokenReadbackBlit =
        [tokenReadbackCommand blitCommandEncoder];
    require(tokenReadbackCommand != nil && tokenReadbackBlit != nil,
        "failed to allocate exact token readback command");
    [tokenReadbackBlit
        copyFromBuffer:(__bridge id<MTLBuffer>)
            view.preparedPhysicsStateTokens
           sourceOffset:0u
               toBuffer:tokenReadback
      destinationOffset:0u
                   size:sizeof(MRNumanXAcceptedPhysicsStateTokenGPUV2)];
    [tokenReadbackBlit endEncoding];
    finish(tokenReadbackCommand);
    const auto token =
        value<MRNumanXAcceptedPhysicsStateTokenGPUV2>(tokenReadback);
    require(view.preparedPhysicsStateTokenByteCount == sizeof(token) &&
                view.finalAcceptedPhysicsStateTokenByteCount == sizeof(token) &&
                view.proposedPhysicsStateTokenByteCount == sizeof(token) &&
                validExactTokenShape(token, transaction, kDeliveryNanoseconds),
        "successful exact physical prepare did not expose a valid V2 token");

    metalrobo::MetalNumanXHumanMatterPhysicalOutcome outcome{};
    require(adapter->physicalOutcome(
                transaction.transactionSlot,
                transaction.transactionFingerprint,
                transaction.slotGeneration,
                outcome),
        "exact physical prepare did not expose its settled outcome");
    require(outcome.jointDecision == MR_NUMANX_COUPLED_HUMAN_ACCEPT &&
                outcome.humanCode == 0u &&
                outcome.matterCode == 0u &&
                outcome.worldCode == 0u &&
                outcome.matterCompletedMicrosteps == 1u,
        "exact physical prepare did not settle as an accepted joint candidate");

    requireApplyAdmissionNegatives(
        *matter, device, queue,
        (__bridge id<MTLBuffer>)view.preparedPhysicsStateTokens,
        transaction, humanMatterProgram);

    const auto witness = makeWitness(
        transaction, humanMatterProgram, token.tokenFingerprint);
    require(witness.witnessFingerprint == witnessFingerprint(witness),
        "exact Brain witness fingerprint is not canonical");
    id<MTLBuffer> witnessBuffer = makeBuffer(
        device, witness, @"lifecycle exact Brain witness");
    id<MTLSharedEvent> witnessReady = [device newSharedEvent];
    require(witnessReady != nil,
        "failed to allocate exact Brain witness event");
    witnessReady.signaledValue = 1u;
    id<MTLCommandBuffer> proposalCommand = [queue commandBuffer];
    require(proposalCommand != nil,
        "failed to allocate exact proposal command");
    metalrobo::MetalNumanXHumanMatterProposalRequest proposalRequest{};
    proposalRequest.commandBuffer = (__bridge void*)proposalCommand;
    proposalRequest.brainCommitWitnesses = (__bridge void*)witnessBuffer;
    proposalRequest.brainPrepareCompleteEvent = (__bridge void*)witnessReady;
    proposalRequest.brainPrepareCompleteEventValue = 1u;
    proposalRequest.brainCommitWitnessesGPUAddress = witnessBuffer.gpuAddress;
    proposalRequest.brainCommitWitnessElementCount = 1u;
    proposalRequest.brainCommitWitnessStride = 1u;
    proposalRequest.environmentCount = view.environmentCount;
    proposalRequest.transactionSlot = view.transactionSlot;
    proposalRequest.stepIndex = view.stepIndex;
    proposalRequest.substepIndex = view.substepIndex;
    proposalRequest.physicsSubstepCount = view.physicsSubstepCount;
    proposalRequest.controlStep = view.controlStep;
    proposalRequest.programFingerprint = view.programFingerprint;
    proposalRequest.transactionFingerprint = view.transactionFingerprint;
    proposalRequest.linearizationEpoch = view.linearizationEpoch;
    proposalRequest.slotGeneration = view.slotGeneration;

    proposalRequest.mode = metalrobo::
        MetalNumanXHumanMatterProposalMode::validateBrainWitness;
    const auto proposed = prepared->proposePrepared(proposalRequest);
    require(proposed.succeeded() && proposed.encoded,
        "exact mutation-free proposal was rejected: " + proposed.message);
    require(!prepared->proposePrepared(proposalRequest).succeeded(),
        "exact prepared generation admitted a duplicate proposal");
    finish(proposalCommand);
    __unsafe_unretained id<MTLSharedEvent> ownerEvent =
        (__bridge id<MTLSharedEvent>)view.physicalPreparedEvent;
    waitForSharedEventValue(
        ownerEvent, view.proposalEventValue,
        "exact proposal completion event did not advance");

    const auto proposal = value<MRNumanXHumanMatterProposalGPU>(
        (__bridge id<MTLBuffer>)view.proposals);
    const auto proposedToken =
        value<MRNumanXAcceptedPhysicsStateTokenGPUV2>(
            (__bridge id<MTLBuffer>)view.proposedPhysicsStateTokens);
    require(proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
                proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
                proposal.code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS &&
                proposal.programFingerprint == humanMatterProgram.fingerprint &&
                proposal.transactionFingerprint ==
                    transaction.transactionFingerprint &&
                proposal.linearizationEpoch == transaction.linearizationEpoch &&
                proposal.slotGeneration == transaction.slotGeneration &&
                proposal.physicsTokenFingerprint == token.tokenFingerprint &&
                proposal.brainProgramFingerprint ==
                    kBrainProgramFingerprint &&
                proposal.brainShadowStateFingerprint ==
                    kBrainShadowFingerprint &&
                proposal.brainWitnessFingerprint ==
                    witness.witnessFingerprint &&
                proposal.candidatePublicationFingerprint ==
                    publicationProgram.candidatePublicationFingerprint &&
                proposal.humanIOIdentityFingerprint ==
                    publicationProgram.identityFingerprint &&
                proposal.environment == 0u &&
                proposal.stepIndex == view.stepIndex &&
                proposal.substepIndex == transaction.physicsSubstep &&
                proposal.transactionSlot == transaction.transactionSlot &&
                proposal.physicsSubstepCount == transaction.physicsSubsteps &&
                proposal.controlStep == transaction.controlStep &&
                proposal.proposalFingerprint == recordFingerprint(proposal) &&
                validExactTokenShape(
                    proposedToken, transaction, kDeliveryNanoseconds) &&
                std::memcmp(
                    &proposedToken, &token, sizeof(proposedToken)) == 0,
        "exact proposal lost its Brain, HumanIO, or V2 token identity");

    const auto preflight = makePreflight(
        transaction, humanMatterProgram, proposal);
    require(preflight.preflightFingerprint == recordFingerprint(preflight),
        "exact Brain preflight fingerprint is not canonical");
    id<MTLBuffer> preflightBuffer = makeBuffer(
        device, preflight, @"lifecycle exact Brain preflight");
    id<MTLSharedEvent> preflightReady = [device newSharedEvent];
    require(preflightReady != nil,
        "failed to allocate exact Brain preflight event");
    preflightReady.signaledValue = 1u;
    metalrobo::MetalNumanXHumanMatterBrainPreflightView preflightView{};
    preflightView.brainCommitPreflights = (__bridge void*)preflightBuffer;
    preflightView.preflightReadyEvent = (__bridge void*)preflightReady;
    preflightView.brainCommitPreflightsGPUAddress = preflightBuffer.gpuAddress;
    preflightView.brainCommitPreflightElementCount = 1u;
    preflightView.preflightReadyEventValue = 1u;
    preflightView.brainCommitPreflightStride = 1u;
    preflightView.environmentCount = view.environmentCount;
    preflightView.transactionSlot = view.transactionSlot;
    preflightView.stepIndex = view.stepIndex;
    preflightView.substepIndex = view.substepIndex;
    preflightView.physicsSubstepCount = view.physicsSubstepCount;
    preflightView.controlStep = view.controlStep;
    preflightView.programFingerprint = view.programFingerprint;
    preflightView.transactionFingerprint = view.transactionFingerprint;
    preflightView.linearizationEpoch = view.linearizationEpoch;
    preflightView.slotGeneration = view.slotGeneration;
    auto stalePreflight = preflightView;
    ++stalePreflight.controlStep;
    require(!prepared->reservePreparedApplication(stalePreflight),
        "exact application reservation admitted a stale control step");
    require(prepared->reservePreparedApplication(preflightView) &&
                !prepared->reservePreparedApplication(preflightView),
        "exact application reservation was not accepted exactly once");

    const auto ack = makeAck(
        transaction, humanMatterProgram, proposal, preflight);
    require(ack.ackFingerprint == recordFingerprint(ack),
        "exact Brain ACK fingerprint is not canonical");
    id<MTLBuffer> ackBuffer = makeBuffer(
        device, ack, @"lifecycle exact Brain ACK");
    id<MTLSharedEvent> ackReady = [device newSharedEvent];
    require(ackReady != nil, "failed to allocate exact Brain ACK event");
    ackReady.signaledValue = 1u;
    OwnerApplyCompletionCapture applyCompletion;
    id<MTLCommandBuffer> applyCommand = [queue commandBuffer];
    require(applyCommand != nil,
        "failed to allocate exact apply command");
    metalrobo::MetalNumanXHumanMatterApplyRequest applyRequest{};
    applyRequest.mode = metalrobo::
        MetalNumanXHumanMatterApplyMode::validateBrainAck;
    applyRequest.commandBuffer = (__bridge void*)applyCommand;
    applyRequest.brainAcks = (__bridge void*)ackBuffer;
    applyRequest.brainAckEvent = (__bridge void*)ackReady;
    applyRequest.completionContext = &applyCompletion;
    applyRequest.completion = &captureOwnerApplyCompletion;
    applyRequest.brainAckEventValue = 1u;
    applyRequest.brainAcksGPUAddress = ackBuffer.gpuAddress;
    applyRequest.brainAckElementCount = 1u;
    applyRequest.brainAckStride = 1u;
    applyRequest.environmentCount = view.environmentCount;
    applyRequest.transactionSlot = view.transactionSlot;
    applyRequest.stepIndex = view.stepIndex;
    applyRequest.substepIndex = view.substepIndex;
    applyRequest.physicsSubstepCount = view.physicsSubstepCount;
    applyRequest.controlStep = view.controlStep;
    applyRequest.programFingerprint = view.programFingerprint;
    applyRequest.transactionFingerprint = view.transactionFingerprint;
    applyRequest.linearizationEpoch = view.linearizationEpoch;
    applyRequest.slotGeneration = view.slotGeneration;
    const auto appliedDiagnostics = prepared->applyPrepared(applyRequest);
    require(appliedDiagnostics.succeeded() && appliedDiagnostics.encoded,
        "exact V2 apply was rejected: " + appliedDiagnostics.message);
    require(!prepared->applyPrepared(applyRequest).succeeded(),
        "exact prepared generation admitted a duplicate apply");
    id<MTLBuffer> matterApplyReadback = makeZeroBuffer(
        device, sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU),
        @"lifecycle exact Matter apply readback");
    id<MTLBlitCommandEncoder> matterApplyBlit =
        [applyCommand blitCommandEncoder];
    require(matterApplyBlit != nil && copy(
                matterApplyBlit,
                (__bridge id<MTLBuffer>)view.matterApplyOutcomes,
                (__bridge void*)matterApplyReadback,
                sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU)),
        "failed to stage exact Matter apply outcome");
    [matterApplyBlit endEncoding];
    finish(applyCommand);
    waitForOwnerApplyCompletion(applyCompletion);
    require(applyCompletion.status.load(std::memory_order_acquire) ==
                static_cast<std::uint32_t>(metalrobo::
                    MetalNumanXHumanMatterApplyTerminalStatus::
                        acceptedPendingPublication) &&
                applyCompletion.slotGeneration.load(
                    std::memory_order_acquire) == view.slotGeneration,
        "exact apply completion lost accepted quarantine identity");

    const auto action = value<MRNumanXHumanMatterApplyActionGPU>(
        (__bridge id<MTLBuffer>)view.applyActions);
    const auto matterApply =
        value<MRNumanXHumanMatterMatterApplyOutcomeGPU>(
            matterApplyReadback);
    const auto applied = value<MRNumanXHumanMatterAppliedOutcomeGPU>(
        (__bridge id<MTLBuffer>)view.appliedOutcomes);
    const auto finalToken =
        value<MRNumanXAcceptedPhysicsStateTokenGPUV2>(
            (__bridge id<MTLBuffer>)view.finalAcceptedPhysicsStateTokens);
    require(action.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT &&
                action.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
                action.code == MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS &&
                action.physicsTokenFingerprint == token.tokenFingerprint &&
                action.proposalFingerprint == proposal.proposalFingerprint &&
                action.ackFingerprint == ack.ackFingerprint &&
                action.preflightFingerprint == preflight.preflightFingerprint &&
                action.fastGateFingerprint == kFastGateFingerprint &&
                action.brainWitnessFingerprint ==
                    witness.witnessFingerprint &&
                action.actionFingerprint == recordFingerprint(action) &&
                matterApply.status == MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT &&
                matterApply.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
                matterApply.code == MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS &&
                matterApply.physicsTokenFingerprint == token.tokenFingerprint &&
                matterApply.proposalFingerprint ==
                    proposal.proposalFingerprint &&
                matterApply.ackFingerprint == ack.ackFingerprint &&
                matterApply.actionFingerprint == action.actionFingerprint &&
                matterApply.matterProgramFingerprint ==
                    matter->acceptedStateProofProgramFingerprintV2() &&
                matterApply.outcomeFingerprint ==
                    recordFingerprint(matterApply) &&
                applied.status ==
                    MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED &&
                applied.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
                applied.code == MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS &&
                applied.physicsTokenFingerprint == token.tokenFingerprint &&
                applied.proposalFingerprint == proposal.proposalFingerprint &&
                applied.ackFingerprint == ack.ackFingerprint &&
                applied.preflightFingerprint == preflight.preflightFingerprint &&
                applied.fastGateFingerprint == kFastGateFingerprint &&
                applied.matterApplyFingerprint ==
                    matterApply.outcomeFingerprint &&
                applied.appliedFingerprint == recordFingerprint(applied) &&
                validExactTokenShape(
                    finalToken, transaction, kDeliveryNanoseconds) &&
                std::memcmp(
                    &finalToken, &proposedToken, sizeof(finalToken)) == 0 &&
                prepared->valid() && publicationLease.valid() &&
                matter->preparedStateDisposition(
                    dispositionIdentity(transaction, humanMatterProgram)) ==
                    numi::matter::PreparedStateDisposition::
                        acceptedPendingPublication,
        "exact apply did not retain one immutable accepted V2 root");

    const std::uint64_t publicationBrainGeneration =
        publicationProgram.acceptedBrainGeneration;
    metalrobo::MetalNumanXHumanMatterPublicationReservationRequest
        reservePublication{};
    reservePublication.environmentCount = view.environmentCount;
    reservePublication.transactionSlot = view.transactionSlot;
    reservePublication.stepIndex = view.stepIndex;
    reservePublication.substepIndex = view.substepIndex;
    reservePublication.physicsSubstepCount = view.physicsSubstepCount;
    reservePublication.controlStep = view.controlStep;
    reservePublication.programFingerprint = view.programFingerprint;
    reservePublication.transactionFingerprint = view.transactionFingerprint;
    reservePublication.linearizationEpoch = view.linearizationEpoch;
    reservePublication.slotGeneration = view.slotGeneration;
    reservePublication.jointCommitFingerprint = kJointCommitFingerprint;
    reservePublication.brainGeneration = publicationBrainGeneration;
    require(prepared->reservePublishedRoot(reservePublication) &&
                !prepared->reservePublishedRoot(reservePublication),
        "exact joint publication reservation was not accepted exactly once");

    id<MTLBuffer> fenceBuffer =
        (__bridge id<MTLBuffer>)view.publicationFences;
    require(fenceBuffer != nil && fenceBuffer.contents != nullptr,
        "exact publication fence is not host-visible");
    auto& fence = *static_cast<
        MRNumanXHumanMatterJointPublicationFenceGPU*>(fenceBuffer.contents);
    require(fence.abiVersion ==
                MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2 &&
                fence.structBytes ==
                    MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES &&
                fence.status == MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING &&
                fence.environment == 0u &&
                fence.controlStep == transaction.controlStep &&
                fence.substepIndex == transaction.physicsSubstep &&
                fence.physicsSubstepCount == transaction.physicsSubsteps &&
                fence.ownerProgramFingerprint ==
                    humanMatterProgram.fingerprint &&
                fence.transactionFingerprint ==
                    transaction.transactionFingerprint &&
                fence.linearizationEpoch == transaction.linearizationEpoch &&
                fence.slotGeneration == transaction.slotGeneration &&
                fence.physicsTokenFingerprint == token.tokenFingerprint &&
                fence.brainProgramFingerprint == kBrainProgramFingerprint &&
                fence.brainShadowStateFingerprint ==
                    kBrainShadowFingerprint &&
                fence.brainWitnessFingerprint == witness.witnessFingerprint &&
                fence.appliedDecisionFingerprint ==
                    applied.appliedFingerprint &&
                fence.jointCommitFingerprint == kJointCommitFingerprint &&
                fence.brainGeneration == publicationBrainGeneration &&
                fence.fenceFingerprint == recordFingerprint(fence) &&
                publicationLease.valid() &&
                humanIO->publishedView(unpublishedSensor).status == metalrobo::
                    MetalNumanXHumanIOStatus::candidateUnavailable,
        "exact PENDING publication fence exposed or misidentified the root");

    fence.status = MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED;
    fence.fenceFingerprint = recordFingerprint(fence);
    metalrobo::MetalNumanXHumanMatterPublicationReleaseRequest
        releasePublication{};
    releasePublication.publicationFences = view.publicationFences;
    releasePublication.publicationFencesGPUAddress =
        view.publicationFencesGPUAddress;
    releasePublication.publicationFenceElementCount =
        view.publicationFenceElementCount;
    releasePublication.publicationFenceStride = view.publicationFenceStride;
    releasePublication.environmentCount = view.environmentCount;
    releasePublication.transactionSlot = view.transactionSlot;
    releasePublication.stepIndex = view.stepIndex;
    releasePublication.substepIndex = view.substepIndex;
    releasePublication.physicsSubstepCount = view.physicsSubstepCount;
    releasePublication.controlStep = view.controlStep;
    releasePublication.programFingerprint = view.programFingerprint;
    releasePublication.transactionFingerprint = view.transactionFingerprint;
    releasePublication.linearizationEpoch = view.linearizationEpoch;
    releasePublication.slotGeneration = view.slotGeneration;
    releasePublication.jointCommitFingerprint = kJointCommitFingerprint;
    releasePublication.brainGeneration = publicationBrainGeneration;
    require(prepared->releasePublishedRoot(releasePublication) == metalrobo::
                MetalNumanXHumanMatterPrepareLeaseDisposition::released,
        "exact COMMITTED publication fence did not release the root");

    metalrobo::MetalNumanXHumanIOSensorView publishedSensor{};
    const auto publishedDiagnostics = humanIO->publishedView(publishedSensor);
    const auto publishedFinalToken =
        value<MRNumanXAcceptedPhysicsStateTokenGPUV2>(
            (__bridge id<MTLBuffer>)view.finalAcceptedPhysicsStateTokens);
    const auto ownerStats = owner->stats();
    require(!prepared->valid() && !publicationLease.valid() &&
                matter->preparedStateDisposition(
                    dispositionIdentity(transaction, humanMatterProgram)) ==
                    numi::matter::PreparedStateDisposition::resolved &&
                matter->snapshot().available &&
                publishedDiagnostics.succeeded() &&
                publishedDiagnostics.published &&
                publishedSensor.state ==
                    metalrobo::MetalNumanXHumanIOViewState::published &&
                publishedSensor.programFingerprint ==
                    humanIOProgram.fingerprint &&
                publishedSensor.transactionFingerprint ==
                    transaction.transactionFingerprint &&
                publishedSensor.sensorGeneration ==
                    publicationProgram.sensorGeneration &&
                publishedSensor.sensorFingerprint ==
                    publicationProgram.sensorFingerprint &&
                publishedSensor.transactionInstanceFingerprint ==
                    key.transactionInstanceFingerprint &&
                publishedSensor.commandBufferIdentity ==
                    key.commandBufferIdentity &&
                publishedSensor.acceptedBrainGeneration ==
                    publicationBrainGeneration &&
                publishedSensor.proprioceptionMetalBuffer ==
                    publicationSensor.proprioceptionMetalBuffer &&
                publishedSensor.validityMetalBuffer ==
                    publicationSensor.validityMetalBuffer &&
                publishedSensor.interoceptionMetalBuffer ==
                    publicationSensor.interoceptionMetalBuffer &&
                publishedSensor.interoceptionValidityMetalBuffer ==
                    publicationSensor.interoceptionValidityMetalBuffer &&
                publishedSensor.proprioceptionGPUAddress ==
                    publicationSensor.proprioceptionGPUAddress &&
                publishedSensor.validityGPUAddress ==
                    publicationSensor.validityGPUAddress &&
                publishedSensor.interoceptionGPUAddress ==
                    publicationSensor.interoceptionGPUAddress &&
                publishedSensor.interoceptionValidityGPUAddress ==
                    publicationSensor.interoceptionValidityGPUAddress &&
                publishedSensor.receptorTimestampMicroseconds == 0u &&
                publishedSensor.deliveryTimestampMicroseconds == 0u &&
                publishedSensor.latencyMicroseconds == 0u &&
                publishedSensor.stepTimeStrideMicroseconds == 0u &&
                publishedSensor.timestampQuantumNanoseconds ==
                    MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
                publishedSensor.receptorTimestampNanoseconds ==
                    kStartNanoseconds &&
                publishedSensor.deliveryTimestampNanoseconds ==
                    kDeliveryNanoseconds &&
                publishedSensor.latencyNanoseconds ==
                    kDurationNanoseconds &&
                publishedSensor.stepTimeStrideNanoseconds ==
                    kDurationNanoseconds &&
                validExactTokenShape(
                    publishedFinalToken, transaction, kDeliveryNanoseconds) &&
                std::memcmp(
                    &publishedFinalToken, &finalToken,
                    sizeof(publishedFinalToken)) == 0 &&
                ownerStats.completedSubmissionCount == 1u &&
                ownerStats.submissionDestructorWaitCount == 0u &&
                ownerStats.terminalSubmissionNonwaitingReapCount == 1u &&
                !ownerStats.hasInFlightSubmission,
        "exact COMMITTED root did not expose coherent resolved owner views");

    std::cout
        << "PASS exact_v2_lifecycle device="
        << device.name.UTF8String
        << " disposition=resolved"
        << " authority_storage=private"
        << " publication_abi=2"
        << " token=accepted_v2"
        << " final_token=published_v2"
        << " joint=" << outcome.jointDecision
        << " human=" << outcome.humanCode
        << " matter=" << outcome.matterCode
        << " world=" << outcome.worldCode
        << " matter_fgmres=" << outcome.matterFGMRESIterations
        << " negatives=family,clock,authority,duplicate"
        << " proposal=accepted"
        << " apply=accepted_quarantined"
        << " publication=committed\n";
}

} // namespace exact_v2_lifecycle_fixture

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            adapter_fixture::require(device != nil,
                "Metal device is unavailable");
            exact_v2_lifecycle_fixture::runLifecycle(device);
            return 0;
        } catch (const std::exception& exception) {
            std::cerr
                << "numanx_human_matter_exact_v2_lifecycle_probe: "
                << exception.what() << '\n';
            return 1;
        }
    }
}
