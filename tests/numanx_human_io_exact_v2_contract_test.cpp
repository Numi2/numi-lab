#include "metalrobo/MetalNumanXHumanIO.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "numanx HumanIO exact-v2 contract: %s\n", message);
    std::exit(1);
}

void require(const bool condition, const char* message) {
    if (!condition) fail(message);
}

} // namespace

int main() {
    static_assert(std::is_same_v<
        decltype(metalrobo::MetalNumanXHumanIOInputV2{}.root),
        MRNumanXBrainJointTransactionTokenV2>);
    static_assert(std::is_same_v<
        decltype(metalrobo::MetalNumanXHumanIOInputV2{}.substep),
        MRNumanXBrainJointSubstepTokenV2>);
    static_assert(std::is_same_v<
        decltype(metalrobo::MetalNumanXHumanIOInputV2{}.candidate),
        MRNumanXBrainMotorCandidateV2>);
    static_assert(sizeof(MRNumanXHumanMotorDispatchGPUV2) == 160u);
    static_assert(alignof(MRNumanXHumanMotorDispatchGPUV2) == 16u);
    static_assert(offsetof(
        MRNumanXHumanMotorDispatchGPUV2,
        acceptedBrainTimestampNanoseconds) == 104u);

    MRNumanXBrainJointTransactionTokenV2 root{};
    root.formatVersion = MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
    root.environmentIdentifier = 0u;
    root.episodeIdentifier = 1u;
    root.controlStepIdentifier = 1u;
    root.parameterVersionFingerprint = 0x101u;
    root.baseBrainGeneration = 0u;
    root.basePhysicsGeneration = 0u;
    root.committedTimestampNanoseconds = 12'500u;
    root.targetTimestampNanoseconds = 25'000u;
    root.shadowGeneration = 1u;
    // The first canonical Brain root starts at random-counter generation zero.
    root.randomCounterGeneration = 0u;
    root.clockDomain =
        MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    root.clockQuantumNanoseconds =
        MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    root.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(root);

    MRNumanXBrainJointSubstepTokenV2 substep{};
    substep.transactionFingerprint = root.transactionFingerprint;
    substep.substepIndex = 0u;
    substep.attemptIndex = 0u;
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
    candidate.motorProfileFingerprint = 0x202u;
    candidate.motorOutputHeaderGPUAddress = 0x1000u;
    candidate.muscleExcitationGPUAddress = 0x2000u;
    candidate.randomCounterGeneration = root.randomCounterGeneration;
    candidate.motorOutputHeaderByteCount =
        sizeof(MRNumanXBrainMotorOutputHeaderGPUV2);
    candidate.muscleExcitationByteCount = sizeof(float);
    candidate.muscleCount = 1u;
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
    candidate.speciesTemplateFingerprint = 0x303u;
    candidate.compiledSpeciesTemplateFingerprint = 0x404u;
    candidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(candidate);

    const std::array<float, 1u> outputs{0.5f};
    MRNumanXBrainMotorOutputHeaderGPUV2 header{};
    header.formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
    header.flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
    header.timestampNanoseconds = candidate.acceptedBrainTimestampNanoseconds;
    header.brainGeneration = candidate.brainGeneration;
    header.profileFingerprint = candidate.motorProfileFingerprint;
    header.protectiveCommandFingerprint = 0x505u;
    header.muscleCount = candidate.muscleCount;
    header.environmentIdentifier = candidate.environmentIdentifier;
    header.motorInhibition = 0.0f;
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
    gate.substepIndex = substep.substepIndex;
    gate.attemptIndex = substep.attemptIndex;
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
    gate.brainProgramFingerprint = 0x606u;
    gate.fastProgramFingerprint = 0x707u;
    gate.decisionGateFingerprint = 0x808u;
    gate.clockDomain = root.clockDomain;
    gate.clockQuantumNanoseconds = root.clockQuantumNanoseconds;
    gate.gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(gate);

    require(metalrobo::metalNumanXBrainMotorOutputV2Valid(
                candidate, header, outputs.data(), outputs.size()),
            "CPU mirror rejected the coherent exact motor output");
    require(metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                root, substep, candidate, header, gate),
            "CPU mirror rejected the coherent exact ready gate");

    std::array<std::byte, 1u> headerObject{};
    std::array<std::byte, 1u> excitationObject{};
    std::array<std::byte, 1u> autonomicObject{};
    std::array<std::byte, 1u> sensingObject{};
    std::array<std::byte, 1u> gateObject{};
    std::array<std::byte, 1u> eventObject{};

    metalrobo::MetalNumanXHumanIOInputV2 input{};
    input.root = root;
    input.substep = substep;
    input.candidate = candidate;
    input.motorOutputHeaderMetalBuffer = headerObject.data();
    input.motorOutputHeaderByteCount = sizeof(header);
    input.motorOutputHeaderEnvironmentStride = sizeof(header);
    input.expectedMotorOutputHeaderGPUAddress =
        candidate.motorOutputHeaderGPUAddress;
    input.excitationMetalBuffer = excitationObject.data();
    input.excitationByteCount = sizeof(float);
    input.excitationEnvironmentStride = 1u;
    input.expectedExcitationGPUAddress =
        candidate.muscleExcitationGPUAddress;
    input.autonomicCommandMetalBuffer = autonomicObject.data();
    input.autonomicCommandByteCount = candidate.autonomicCommandByteCount;
    input.expectedAutonomicCommandGPUAddress =
        candidate.autonomicCommandGPUAddress;
    input.activeSensingCommandMetalBuffer = sensingObject.data();
    input.activeSensingCommandByteCount =
        candidate.activeSensingCommandByteCount;
    input.expectedActiveSensingCommandGPUAddress =
        candidate.activeSensingCommandGPUAddress;
    input.motorReadyGateMetalBuffer = gateObject.data();
    input.motorReadyGateByteCount = sizeof(gate);
    input.expectedMotorReadyGateGPUAddress = 0x5000u;
    input.motorReadySharedEvent = eventObject.data();
    input.motorReadySharedEventValue = 1u;
    input.environmentCount = 1u;
    input.muscleCount = 1u;
    input.stepCount = 1u;
    input.timestepNanoseconds = substep.durationNanoseconds;
    input.receptorTimestampNanoseconds =
        candidate.acceptedBrainTimestampNanoseconds;
    input.candidateSensorGeneration = 1u;

    MRNumanXHumanMotorDispatchGPUV2 dispatch{};
    require(metalrobo::metalNumanXHumanIOBuildMotorDispatchV2(
                input, dispatch),
            "coherent exact HumanIO input did not build a GPU dispatch");
    MRNumanXHumanMotorDispatchGPUV2 expected{};
    expected.abiVersion = MR_NUMANX_HUMAN_MOTOR_DISPATCH_ABI_VERSION_V2;
    expected.environmentCount = 1u;
    expected.muscleCount = 1u;
    expected.excitationEnvironmentStride = 1u;
    expected.motorOutputFormatVersion =
        MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
    expected.motorCandidateFormatVersion =
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
    expected.motorCandidateFlags = candidate.flags;
    expected.actuatorCommandKind = candidate.actuatorCommandKind;
    expected.environmentIdentifierBase = candidate.environmentIdentifier;
    expected.headerEnvironmentStride = 1u;
    expected.substepIndex = substep.substepIndex;
    expected.attemptIndex = substep.attemptIndex;
    expected.clockDomain =
        MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    expected.clockQuantumNanoseconds = 1u;
    expected.controlStep = root.controlStepIdentifier;
    expected.transactionFingerprint = root.transactionFingerprint;
    expected.substepFingerprint = substep.substepFingerprint;
    expected.motorCandidateFingerprint = candidate.candidateFingerprint;
    expected.acceptedBrainGeneration = candidate.brainGeneration;
    expected.acceptedBrainTimestampNanoseconds = 12'500u;
    expected.motorProfileFingerprint = candidate.motorProfileFingerprint;
    expected.randomCounterGeneration = 0u;
    expected.speciesTemplateFingerprint =
        candidate.speciesTemplateFingerprint;
    expected.compiledSpeciesTemplateFingerprint =
        candidate.compiledSpeciesTemplateFingerprint;
    expected.expectedExcitationGPUAddress =
        candidate.muscleExcitationGPUAddress;
    expected.expectedMotorOutputHeaderGPUAddress =
        candidate.motorOutputHeaderGPUAddress;
    require(std::memcmp(&dispatch, &expected, sizeof(dispatch)) == 0,
            "exact dispatch did not match the complete canonical image");

    const auto requireRejected = [](
        const metalrobo::MetalNumanXHumanIOInputV2& rejectedInput,
        const char* message
    ) {
        MRNumanXHumanMotorDispatchGPUV2 output;
        std::memset(&output, 0xa5, sizeof(output));
        const MRNumanXHumanMotorDispatchGPUV2 before = output;
        require(!metalrobo::metalNumanXHumanIOBuildMotorDispatchV2(
                    rejectedInput, output) &&
                std::memcmp(&output, &before, sizeof(output)) == 0,
                message);
    };

    auto wrongQuantum = input;
    wrongQuantum.root.clockQuantumNanoseconds = 1'000u;
    wrongQuantum.root.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(
            wrongQuantum.root);
    requireRejected(
        wrongQuantum,
        "mixed clock quantum was admitted or mutated the output");

    auto staleTimestamp = input;
    ++staleTimestamp.receptorTimestampNanoseconds;
    requireRejected(
        staleTimestamp,
        "stale receptor timestamp was admitted or mutated the output");

    auto nonzeroAttempt = input;
    nonzeroAttempt.substep.attemptIndex = 1u;
    nonzeroAttempt.substep.substepFingerprint =
        metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(
            nonzeroAttempt.substep);
    nonzeroAttempt.candidate.substepFingerprint =
        nonzeroAttempt.substep.substepFingerprint;
    nonzeroAttempt.candidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
            nonzeroAttempt.candidate);
    requireRejected(
        nonzeroAttempt,
        "nonzero exact attempt was admitted or mutated the output");

    auto missingGate = input;
    missingGate.motorReadyGateMetalBuffer = nullptr;
    requireRejected(
        missingGate,
        "missing terminal gate was admitted or mutated the output");

    auto overlappingLease = input;
    overlappingLease.expectedMotorReadyGateGPUAddress =
        candidate.muscleExcitationGPUAddress;
    requireRejected(
        overlappingLease,
        "overlapping GPU ranges were admitted or mutated the output");

    auto mixedGate = gate;
    mixedGate.clockDomain = 0u;
    mixedGate.gateFingerprint =
        metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(mixedGate);
    require(!metalrobo::metalNumanXBrainMotorReadyGateV2Valid(
                root, substep, candidate, header, mixedGate),
            "CPU mirror admitted a mixed-domain ready gate");

    std::puts("numanx HumanIO exact-v2 contract: ok");
    return 0;
}
