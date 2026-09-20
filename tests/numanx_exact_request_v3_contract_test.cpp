#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/mrnx_bridge_v1.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "numanx exact request-v3 contract: %s\n", message);
    std::exit(1);
}

void require(const bool condition, const char* message) {
    if (!condition) fail(message);
}

} // namespace

int main() {
    constexpr std::uint64_t rootGolden = 0x986252871c867014ull;
    constexpr std::uint64_t substepGolden = 0x339760742e9d5b13ull;
    constexpr std::uint64_t candidateGolden = 0xd545ffb84702f6ccull;
    constexpr std::uint64_t outputGolden = 0xd2997dd67ccf83f4ull;
    constexpr std::uint64_t gateGolden = 0x7a7c4daa7709eef0ull;
    constexpr std::uint64_t legacyRootGolden = 0xdd0df636db87727eull;
    constexpr std::uint64_t legacySubstepGolden = 0x5a84e593f19eab3eull;
    constexpr std::uint64_t legacyCandidateGolden = 0x5b9107814f5bf8c0ull;

    static_assert(sizeof(MRNumanXBrainJointTransactionTokenV2) == 96u);
    static_assert(sizeof(MRNumanXBrainJointSubstepTokenV2) == 72u);
    static_assert(sizeof(MRNumanXBrainMotorCandidateV2) == 152u);
    static_assert(sizeof(MRNumanXBrainMotorOutputHeaderGPUV2) == 80u);
    static_assert(sizeof(MRNumanXBrainMotorReadyGateGPUV2) == 160u);
    static_assert(alignof(mrnx_brain_motor_output_header_v2) == 16u);
    static_assert(alignof(mrnx_brain_motor_ready_gate_v2) == 16u);
    static_assert(sizeof(mrnx_physical_root_request_v2) == 600u);
    static_assert(std::is_same_v<
        decltype(mrnx_physical_root_request_v2{}.root),
        mrnx_brain_joint_transaction_v1>);
    static_assert(std::is_same_v<
        decltype(mrnx_physical_root_request_v2{}.candidate),
        mrnx_brain_motor_candidate_v1>);
    static_assert(sizeof(mrnx_physical_root_request_v3) == 600u);
    static_assert(std::is_same_v<
        decltype(mrnx_physical_root_request_v3{}.root),
        mrnx_brain_joint_transaction_v2>);
    static_assert(std::is_same_v<
        decltype(mrnx_physical_root_request_v3{}.candidate),
        mrnx_brain_motor_candidate_v2>);
    static_assert(offsetof(mrnx_physical_root_request_v3, candidate) == 176u);
    static_assert(offsetof(mrnx_physical_root_request_v3, motor_header) == 328u);
    static_assert(offsetof(
        mrnx_physical_root_request_v3, motor_ready_gate) == 520u);

    MRNumanXBrainJointTransactionTokenV2 root{};
    root.formatVersion = MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
    root.environmentIdentifier = 7u;
    root.episodeIdentifier = 23u;
    root.controlStepIdentifier = 17u;
    root.parameterVersionFingerprint = 0x123456789abcdef0ull;
    root.baseBrainGeneration = 9u;
    root.basePhysicsGeneration = 100u;
    root.committedTimestampNanoseconds = 12'500u;
    root.targetTimestampNanoseconds = 25'000u;
    root.shadowGeneration = 10u;
    root.randomCounterGeneration = 55u;
    root.clockDomain =
        MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    root.clockQuantumNanoseconds =
        MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    root.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(root);

    MRNumanXBrainJointSubstepTokenV2 substep{};
    substep.transactionFingerprint = root.transactionFingerprint;
    substep.startTimestampNanoseconds = root.committedTimestampNanoseconds;
    substep.durationNanoseconds = 12'500u;
    substep.candidateTimestampNanoseconds = root.targetTimestampNanoseconds;
    substep.shadowGeneration = root.shadowGeneration;
    substep.randomCounterGeneration = root.randomCounterGeneration;
    substep.clockDomain = root.clockDomain;
    substep.clockQuantumNanoseconds = root.clockQuantumNanoseconds;
    substep.substepFingerprint =
        metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(substep);

    MRNumanXBrainMotorCandidateV2 candidate{};
    candidate.formatVersion = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
    candidate.flags = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
    candidate.transactionFingerprint = root.transactionFingerprint;
    candidate.substepFingerprint = substep.substepFingerprint;
    candidate.acceptedBrainTimestampNanoseconds =
        substep.startTimestampNanoseconds;
    candidate.brainGeneration = root.shadowGeneration;
    candidate.motorProfileFingerprint = 0x4444u;
    candidate.motorOutputHeaderGPUAddress = 0x1000u;
    candidate.muscleExcitationGPUAddress = 0x2000u;
    candidate.randomCounterGeneration = root.randomCounterGeneration;
    candidate.motorOutputHeaderByteCount =
        sizeof(MRNumanXBrainMotorOutputHeaderGPUV2);
    candidate.muscleCount = 3u;
    candidate.muscleExcitationByteCount =
        candidate.muscleCount * sizeof(float);
    candidate.environmentIdentifier = root.environmentIdentifier;
    candidate.autonomicCommandGPUAddress = 0x3000u;
    candidate.autonomicCommandByteCount =
        MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT;
    candidate.autonomicCommandCount = 1u;
    candidate.activeSensingCommandGPUAddress = 0x4000u;
    candidate.activeSensingCommandByteCount =
        MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
    candidate.activeSensingCommandCount = 1u;
    candidate.actuatorCommandKind =
        MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
    candidate.clockDomain = root.clockDomain;
    candidate.speciesTemplateFingerprint = 0x5555u;
    candidate.compiledSpeciesTemplateFingerprint = 0x6666u;
    candidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(candidate);

    const std::array<float, 3u> outputs{0.25f, 0.5f, 0.75f};
    MRNumanXBrainMotorOutputHeaderGPUV2 header{};
    header.formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
    header.flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
    header.timestampNanoseconds = candidate.acceptedBrainTimestampNanoseconds;
    header.brainGeneration = candidate.brainGeneration;
    header.profileFingerprint = candidate.motorProfileFingerprint;
    header.protectiveCommandFingerprint = 0x7777u;
    header.muscleCount = candidate.muscleCount;
    header.environmentIdentifier = candidate.environmentIdentifier;
    header.motorInhibition = 0.125f;
    header.autonomicArousal = 0.25f;
    header.actuatorCommandKind = candidate.actuatorCommandKind;
    header.clockDomain = candidate.clockDomain;
    header.outputMinimum = 0.0f;
    header.outputMaximum = 1.0f;
    header.outputFingerprint =
        metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
            header, outputs.data(), outputs.size());

    MRNumanXBrainMotorReadyGateGPUV2 gate{};
    gate.abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION_V2;
    gate.structBytes = sizeof(gate);
    gate.status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
    gate.environment = candidate.environmentIdentifier;
    gate.muscleCount = candidate.muscleCount;
    gate.actuatorCommandKind = candidate.actuatorCommandKind;
    gate.controlStep = root.controlStepIdentifier;
    gate.transactionFingerprint = root.transactionFingerprint;
    gate.substepFingerprint = substep.substepFingerprint;
    gate.candidateFingerprint = candidate.candidateFingerprint;
    gate.motorOutputFingerprint = header.outputFingerprint;
    gate.motorProfileFingerprint = candidate.motorProfileFingerprint;
    gate.brainGeneration = candidate.brainGeneration;
    gate.acceptedBrainTimestampNanoseconds =
        candidate.acceptedBrainTimestampNanoseconds;
    gate.randomCounterGeneration = candidate.randomCounterGeneration;
    gate.speciesTemplateFingerprint = candidate.speciesTemplateFingerprint;
    gate.compiledSpeciesTemplateFingerprint =
        candidate.compiledSpeciesTemplateFingerprint;
    gate.brainProgramFingerprint = 0x8888u;
    gate.fastProgramFingerprint = 0x9999u;
    gate.decisionGateFingerprint = 0xaaaau;
    gate.clockDomain = root.clockDomain;
    gate.clockQuantumNanoseconds = root.clockQuantumNanoseconds;
    gate.gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(gate);

    require(root.transactionFingerprint == rootGolden,
            "v2 root fingerprint changed from the shared Brain golden");
    require(substep.substepFingerprint == substepGolden,
            "v2 substep fingerprint changed from the shared Brain golden");
    require(candidate.candidateFingerprint == candidateGolden,
            "v2 candidate fingerprint changed from the shared Brain golden");
    require(header.outputFingerprint == outputGolden,
            "v2 output fingerprint changed from the shared Brain golden");
    require(gate.gateFingerprint == gateGolden,
            "v2 ready-gate fingerprint changed from the shared Brain golden");

    require(metalrobo::metalNumanXBrainJointTransactionV2Valid(root),
            "coherent v2 root was rejected");
    require(metalrobo::metalNumanXBrainJointSubstepV2Valid(root, substep),
            "coherent v2 substep was rejected");
    require(metalrobo::metalNumanXBrainMotorCandidateV2Valid(
                root, substep, candidate),
            "coherent v2 motor candidate was rejected");
    require(metalrobo::metalNumanXBrainMotorOutputV2Valid(
                candidate, header, outputs.data(), outputs.size()),
            "coherent v2 motor output was rejected");
    require(metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                root, substep, candidate, header, gate),
            "coherent v2 ready gate was rejected");

    auto mixedRoot = root;
    mixedRoot.formatVersion = MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION;
    mixedRoot.clockDomain = 0u;
    mixedRoot.clockQuantumNanoseconds = 0u;
    mixedRoot.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(mixedRoot);
    require(!metalrobo::metalNumanXBrainJointTransactionV2Valid(mixedRoot),
            "v2 validator admitted a legacy root family");

    auto wrongQuantumRoot = root;
    wrongQuantumRoot.clockQuantumNanoseconds = 1'000u;
    wrongQuantumRoot.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(
            wrongQuantumRoot);
    require(!metalrobo::metalNumanXBrainJointTransactionV2Valid(
                wrongQuantumRoot),
            "v2 validator admitted a non-nanosecond root quantum");

    auto mixedSubstep = substep;
    mixedSubstep.clockDomain = 0u;
    mixedSubstep.clockQuantumNanoseconds = 0u;
    mixedSubstep.substepFingerprint =
        metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(mixedSubstep);
    require(!metalrobo::metalNumanXBrainJointSubstepV2Valid(
                root, mixedSubstep),
            "v2 validator admitted a mixed-domain substep");

    auto mixedCandidate = candidate;
    mixedCandidate.formatVersion = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION;
    mixedCandidate.clockDomain = 0u;
    mixedCandidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
            mixedCandidate);
    require(!metalrobo::metalNumanXBrainMotorCandidateV2Valid(
                root, substep, mixedCandidate),
            "v2 validator admitted a legacy motor candidate");
    require(!metalrobo::metalNumanXBrainMotorOutputV2Valid(
                mixedCandidate, header, outputs.data(), outputs.size()),
            "v2 output validator admitted a legacy candidate envelope");

    auto misalignedCandidate = candidate;
    misalignedCandidate.motorOutputHeaderGPUAddress += 8u;
    misalignedCandidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
            misalignedCandidate);
    require(!metalrobo::metalNumanXBrainMotorCandidateV2Valid(
                root, substep, misalignedCandidate),
            "v2 validator admitted a non-16-byte motor-header address");
    require(!metalrobo::metalNumanXBrainMotorOutputV2Valid(
                misalignedCandidate, header, outputs.data(), outputs.size()),
            "v2 output validator admitted a misaligned candidate envelope");

    auto invalidCommandCandidate = candidate;
    invalidCommandCandidate.actuatorCommandKind = 8u;
    invalidCommandCandidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
            invalidCommandCandidate);
    auto invalidCommandHeader = header;
    invalidCommandHeader.actuatorCommandKind = 8u;
    invalidCommandHeader.outputFingerprint =
        metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
            invalidCommandHeader, outputs.data(), outputs.size());
    require(!metalrobo::metalNumanXBrainMotorOutputV2Valid(
                invalidCommandCandidate, invalidCommandHeader,
                outputs.data(), outputs.size()),
            "v2 output validator admitted actuator command kind 8");

    auto mixedHeader = header;
    mixedHeader.formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION;
    mixedHeader.clockDomain = 0u;
    mixedHeader.outputFingerprint =
        metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
            mixedHeader, outputs.data(), outputs.size());
    require(!metalrobo::metalNumanXBrainMotorOutputV2Valid(
                candidate, mixedHeader, outputs.data(), outputs.size()),
            "v2 validator admitted a legacy motor-output header");

    auto gateForMixedHeader = gate;
    gateForMixedHeader.motorOutputFingerprint = mixedHeader.outputFingerprint;
    gateForMixedHeader.gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(
            gateForMixedHeader);
    require(!metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                root, substep, candidate, mixedHeader, gateForMixedHeader),
            "v2 ready gate admitted legacy output-header metadata");

    auto mixedGate = gate;
    mixedGate.abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION;
    mixedGate.clockDomain = 0u;
    mixedGate.clockQuantumNanoseconds = 0u;
    mixedGate.gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(mixedGate);
    require(!metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                root, substep, candidate, header, mixedGate),
            "v2 validator admitted a legacy ready gate");

    MRNumanXBrainJointTransactionToken legacyRoot{};
    legacyRoot.formatVersion = MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION;
    legacyRoot.environmentIdentifier = root.environmentIdentifier;
    legacyRoot.episodeIdentifier = root.episodeIdentifier;
    legacyRoot.controlStepIdentifier = root.controlStepIdentifier;
    legacyRoot.parameterVersionFingerprint = root.parameterVersionFingerprint;
    legacyRoot.committedTimestampMicroseconds =
        root.committedTimestampNanoseconds;
    legacyRoot.targetTimestampMicroseconds = root.targetTimestampNanoseconds;
    legacyRoot.shadowGeneration = root.shadowGeneration;
    legacyRoot.randomCounterGeneration = root.randomCounterGeneration;
    legacyRoot.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionFingerprint(legacyRoot);

    MRNumanXBrainJointSubstepToken legacySubstep{};
    legacySubstep.transactionFingerprint = legacyRoot.transactionFingerprint;
    legacySubstep.startTimestampMicroseconds =
        legacyRoot.committedTimestampMicroseconds;
    legacySubstep.durationMicroseconds = 12'500u;
    legacySubstep.candidateTimestampMicroseconds =
        legacyRoot.targetTimestampMicroseconds;
    legacySubstep.shadowGeneration = legacyRoot.shadowGeneration;
    legacySubstep.randomCounterGeneration =
        legacyRoot.randomCounterGeneration;
    legacySubstep.substepFingerprint =
        metalrobo::metalNumanXBrainJointSubstepFingerprint(legacySubstep);

    MRNumanXBrainMotorCandidate legacyCandidate{};
    legacyCandidate.formatVersion = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION;
    legacyCandidate.flags = candidate.flags;
    legacyCandidate.transactionFingerprint =
        legacyRoot.transactionFingerprint;
    legacyCandidate.substepFingerprint = legacySubstep.substepFingerprint;
    legacyCandidate.acceptedBrainTimestampMicroseconds =
        legacySubstep.startTimestampMicroseconds;
    legacyCandidate.brainGeneration = candidate.brainGeneration;
    legacyCandidate.motorProfileFingerprint =
        candidate.motorProfileFingerprint;
    legacyCandidate.motorOutputHeaderGPUAddress =
        candidate.motorOutputHeaderGPUAddress;
    legacyCandidate.muscleExcitationGPUAddress =
        candidate.muscleExcitationGPUAddress;
    legacyCandidate.randomCounterGeneration =
        candidate.randomCounterGeneration;
    legacyCandidate.motorOutputHeaderByteCount =
        sizeof(MRNumanXBrainMotorOutputHeaderGPU);
    legacyCandidate.muscleExcitationByteCount =
        candidate.muscleExcitationByteCount;
    legacyCandidate.muscleCount = candidate.muscleCount;
    legacyCandidate.environmentIdentifier = candidate.environmentIdentifier;
    legacyCandidate.autonomicCommandGPUAddress =
        candidate.autonomicCommandGPUAddress;
    legacyCandidate.autonomicCommandByteCount =
        candidate.autonomicCommandByteCount;
    legacyCandidate.autonomicCommandCount = candidate.autonomicCommandCount;
    legacyCandidate.activeSensingCommandGPUAddress =
        candidate.activeSensingCommandGPUAddress;
    legacyCandidate.activeSensingCommandByteCount =
        candidate.activeSensingCommandByteCount;
    legacyCandidate.activeSensingCommandCount =
        candidate.activeSensingCommandCount;
    legacyCandidate.actuatorCommandKind = candidate.actuatorCommandKind;
    legacyCandidate.speciesTemplateFingerprint =
        candidate.speciesTemplateFingerprint;
    legacyCandidate.compiledSpeciesTemplateFingerprint =
        candidate.compiledSpeciesTemplateFingerprint;
    legacyCandidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateFingerprint(legacyCandidate);

    require(legacyRoot.transactionFingerprint == legacyRootGolden,
            "v1 root fingerprint changed");
    require(legacySubstep.substepFingerprint == legacySubstepGolden,
            "v1 substep fingerprint changed");
    require(legacyCandidate.candidateFingerprint == legacyCandidateGolden,
            "v1 candidate fingerprint changed");
    require(root.transactionFingerprint != legacyRoot.transactionFingerprint &&
                substep.substepFingerprint != legacySubstep.substepFingerprint &&
                candidate.candidateFingerprint !=
                    legacyCandidate.candidateFingerprint,
            "v2 fingerprints were not domain-separated from v1");

    std::puts("numanx_exact_request_v3_contract=pass cpu_only=true");
    return 0;
}
