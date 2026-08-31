#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanIO.hpp"

#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using metalrobo::MetalNumanXHumanIOContext;
using metalrobo::MetalNumanXHumanIOCandidatePublicationDisposition;
using metalrobo::MetalNumanXHumanIOCandidatePublicationLease;
using metalrobo::MetalNumanXHumanIODiagnostics;
using metalrobo::MetalNumanXHumanIOInput;
using metalrobo::MetalNumanXHumanIOSensorView;
using metalrobo::MetalNumanXHumanIOStatus;
using metalrobo::MetalNumanXHumanIOTransactionKey;
using metalrobo::MetalNumanXHumanIOCandidatePublicationBinding;
using metalrobo::MetalNumanXHumanIOCandidatePublicationCommit;
using metalrobo::MetalNumanXHumanIOCandidateCompletionStatus;
using metalrobo::MetalNumanXTransactionPass;
using metalrobo::MetalNumanXTransactionPhase;
using metalrobo::MetalNumanXTransactionProgram;

constexpr std::uint32_t kEnvironmentCount = 1u;
constexpr std::uint32_t kMuscleCount = 4u;
constexpr std::uint32_t kStepCount = 1u;
constexpr std::size_t kMuscleElementCount =
    kEnvironmentCount * kMuscleCount;
constexpr std::uint64_t kBrainTimestampMicroseconds = 4'000'001u;
constexpr std::uint64_t kBrainGeneration = 3001u;
constexpr std::uint64_t kMotorProfileFingerprint = 5001u;
constexpr std::uint32_t kEnvironmentIdentifierBase = 7000u;
constexpr std::uint64_t kSpeciesTemplateFingerprint = 8001u;
constexpr std::uint64_t kCompiledSpeciesTemplateFingerprint = 8002u;
constexpr std::uint64_t kRandomCounterGeneration = 9001u;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

struct ProbeFailure {
    int code = 1;
    std::string message;
};

struct BorrowedHumanBuffers {
    id<MTLBuffer> states = nil;
    id<MTLBuffer> results = nil;
    id<MTLBuffer> statuses = nil;
};

struct EncodedCandidate {
    MetalNumanXHumanIOTransactionKey key{};
    MetalNumanXHumanIOSensorView view{};
};

struct CandidateCompletionAudit {
    MetalNumanXHumanIOContext* context = nullptr;
    MetalNumanXHumanIOTransactionKey expectedKey{};
    std::atomic<std::uint32_t> count{0u};
    std::atomic<bool> reentrySafe{false};
    std::atomic<std::uint32_t> status{0u};
};

void candidateCompletion(
    void* raw,
    MetalNumanXHumanIOCandidateCompletionStatus status,
    const MetalNumanXHumanIOTransactionKey& key,
    const MetalNumanXHumanIOSensorView& view
) noexcept;

[[nodiscard]] bool closeEnough(const float a, const float b) noexcept {
    return std::fabs(a - b) <= 1.0e-6f;
}

void mixU32(std::uint64_t& hash, const std::uint32_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixU64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixFloat(std::uint64_t& hash, const float value) noexcept {
    mixU32(hash, std::bit_cast<std::uint32_t>(value));
}

[[nodiscard]] std::uint64_t motorOutputFingerprint(
    const MRNumanXBrainMotorOutputHeaderGPU& header,
    const float* excitations
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION);
    mixU32(hash, header.formatVersion);
    mixU32(hash, header.flags);
    mixU64(hash, header.timestampMicroseconds);
    mixU64(hash, header.brainGeneration);
    mixU64(hash, header.profileFingerprint);
    mixU64(hash, header.protectiveCommandFingerprint);
    mixU32(hash, header.muscleCount);
    mixU32(hash, header.environmentIdentifier);
    mixFloat(hash, header.motorInhibition);
    mixFloat(hash, header.autonomicArousal);
    mixU32(hash, header.actuatorCommandKind);
    mixU32(hash, header.reserved);
    mixFloat(hash, header.outputMinimum);
    mixFloat(hash, header.outputMaximum);
    for (std::uint32_t muscle = 0u; muscle < header.muscleCount; ++muscle) {
        mixFloat(hash, excitations[muscle]);
    }
    return hash;
}

[[nodiscard]] std::uint64_t motorReadyGateFingerprint(
    const MRNumanXBrainMotorReadyGateGPU& gate
) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&gate);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u; index < 152u; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

void initializeMotorReadyGate(
    const MetalNumanXHumanIOInput& input,
    const MRNumanXBrainMotorOutputHeaderGPU& header,
    MRNumanXBrainMotorReadyGateGPU& gate,
    const std::uint32_t status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS
) noexcept {
    gate = {};
    gate.abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION;
    gate.structBytes = MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT;
    gate.status = status;
    gate.environment = input.candidate.environmentIdentifier;
    gate.substepIndex = input.substep.substepIndex;
    gate.attemptIndex = input.substep.attemptIndex;
    gate.muscleCount = input.candidate.muscleCount;
    gate.actuatorCommandKind = input.candidate.actuatorCommandKind;
    gate.controlStep = input.root.controlStepIdentifier;
    gate.transactionFingerprint = input.root.transactionFingerprint;
    gate.substepFingerprint = input.substep.substepFingerprint;
    gate.candidateFingerprint = input.candidate.candidateFingerprint;
    gate.motorOutputFingerprint = status == MR_NUMANX_BRAIN_READY_GATE_SUCCESS
        ? header.outputFingerprint
        : 0u;
    gate.motorProfileFingerprint = input.candidate.motorProfileFingerprint;
    gate.brainGeneration = input.candidate.brainGeneration;
    gate.acceptedBrainTimestampMicroseconds =
        input.candidate.acceptedBrainTimestampMicroseconds;
    gate.randomCounterGeneration = input.candidate.randomCounterGeneration;
    gate.speciesTemplateFingerprint =
        input.candidate.speciesTemplateFingerprint;
    gate.compiledSpeciesTemplateFingerprint =
        input.candidate.compiledSpeciesTemplateFingerprint;
    gate.brainProgramFingerprint = 0x425241494e505247ull;
    gate.fastProgramFingerprint = 0x4641535450524f47ull;
    gate.decisionGateFingerprint = 0x4445434953494f4eull;
    gate.gateFingerprint = motorReadyGateFingerprint(gate);
}

void initializeMotorHeaders(
    MRNumanXBrainMotorOutputHeaderGPU* headers,
    const float* excitations,
    const std::uint64_t brainGeneration = kBrainGeneration
) noexcept {
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        auto& header = headers[environment];
        header = {};
        header.formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION;
        header.flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
        header.timestampMicroseconds = kBrainTimestampMicroseconds;
        header.brainGeneration = brainGeneration;
        header.profileFingerprint = kMotorProfileFingerprint;
        header.protectiveCommandFingerprint = 6001u + environment;
        header.muscleCount = kMuscleCount;
        header.environmentIdentifier =
            kEnvironmentIdentifierBase + environment;
        header.motorInhibition = 0.25f;
        header.autonomicArousal = 0.125f;
        header.actuatorCommandKind =
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
        header.outputMinimum = 0.0f;
        header.outputMaximum = 1.0f;
        header.outputFingerprint = motorOutputFingerprint(
            header,
            excitations +
                static_cast<std::size_t>(environment) * kMuscleCount
        );
    }
}

void refreshTransactionIdentity(
    MetalNumanXHumanIOInput& input,
    const std::uint64_t controlStepIdentifier,
    const std::uint64_t sensorGeneration
) noexcept {
    input.root.controlStepIdentifier = controlStepIdentifier;
    input.root.transactionFingerprint =
        metalrobo::metalNumanXBrainJointTransactionFingerprint(input.root);
    input.substep.transactionFingerprint =
        input.root.transactionFingerprint;
    input.substep.substepFingerprint =
        metalrobo::metalNumanXBrainJointSubstepFingerprint(input.substep);
    input.candidate.transactionFingerprint =
        input.root.transactionFingerprint;
    input.candidate.substepFingerprint = input.substep.substepFingerprint;
    input.candidate.candidateFingerprint =
        metalrobo::metalNumanXBrainMotorCandidateFingerprint(input.candidate);
    input.candidateSensorGeneration = sensorGeneration;
}

void initializeAuthoritativeInput(
    MetalNumanXHumanIOInput& input,
    id<MTLBuffer> motorHeaders,
    id<MTLBuffer> excitations,
    id<MTLBuffer> autonomicCommands,
    id<MTLBuffer> activeSensingCommands
) noexcept {
    input = {};
    input.root.formatVersion = MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION;
    input.root.environmentIdentifier = kEnvironmentIdentifierBase;
    input.root.episodeIdentifier = 101u;
    input.root.controlStepIdentifier = 201u;
    input.root.parameterVersionFingerprint = 301u;
    input.root.baseBrainGeneration = kBrainGeneration;
    input.root.basePhysicsGeneration = 401u;
    input.root.committedTimestampMicroseconds =
        kBrainTimestampMicroseconds - 1'000u;
    input.root.targetTimestampMicroseconds =
        kBrainTimestampMicroseconds + 2'000u;
    input.root.shadowGeneration = kBrainGeneration + 1u;
    input.root.randomCounterGeneration = kRandomCounterGeneration;

    input.substep.substepIndex = 0u;
    input.substep.attemptIndex = 0u;
    input.substep.startTimestampMicroseconds =
        kBrainTimestampMicroseconds;
    input.substep.durationMicroseconds = 2'000u;
    input.substep.candidateTimestampMicroseconds =
        kBrainTimestampMicroseconds + 2'000u;
    input.substep.shadowGeneration = input.root.shadowGeneration;
    input.substep.randomCounterGeneration = kRandomCounterGeneration;

    input.candidate.formatVersion =
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION;
    input.candidate.flags = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID;
    input.candidate.acceptedBrainTimestampMicroseconds =
        kBrainTimestampMicroseconds;
    input.candidate.brainGeneration = kBrainGeneration;
    input.candidate.motorProfileFingerprint = kMotorProfileFingerprint;
    input.candidate.motorOutputHeaderGPUAddress = motorHeaders.gpuAddress;
    input.candidate.muscleExcitationGPUAddress = excitations.gpuAddress;
    input.candidate.randomCounterGeneration = kRandomCounterGeneration;
    input.candidate.motorOutputHeaderByteCount =
        sizeof(MRNumanXBrainMotorOutputHeaderGPU);
    input.candidate.muscleExcitationByteCount =
        kMuscleElementCount * sizeof(float);
    input.candidate.muscleCount = kMuscleCount;
    input.candidate.environmentIdentifier = kEnvironmentIdentifierBase;
    input.candidate.autonomicCommandGPUAddress =
        autonomicCommands.gpuAddress;
    input.candidate.autonomicCommandByteCount =
        MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT;
    input.candidate.autonomicCommandCount = 1u;
    input.candidate.activeSensingCommandGPUAddress =
        activeSensingCommands.gpuAddress;
    input.candidate.activeSensingCommandByteCount =
        MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
    input.candidate.activeSensingCommandCount = 1u;
    input.candidate.actuatorCommandKind =
        MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
    input.candidate.speciesTemplateFingerprint =
        kSpeciesTemplateFingerprint;
    input.candidate.compiledSpeciesTemplateFingerprint =
        kCompiledSpeciesTemplateFingerprint;

    input.motorOutputHeaderMetalBuffer = (__bridge void*)motorHeaders;
    input.motorOutputHeaderByteCount =
        sizeof(MRNumanXBrainMotorOutputHeaderGPU);
    input.motorOutputHeaderEnvironmentStride =
        sizeof(MRNumanXBrainMotorOutputHeaderGPU);
    input.expectedMotorOutputHeaderGPUAddress = motorHeaders.gpuAddress;
    input.excitationMetalBuffer = (__bridge void*)excitations;
    input.excitationByteCount = kMuscleElementCount * sizeof(float);
    input.excitationEnvironmentStride = kMuscleCount;
    input.expectedExcitationGPUAddress = excitations.gpuAddress;
    input.autonomicCommandMetalBuffer = (__bridge void*)autonomicCommands;
    input.autonomicCommandByteCount =
        MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT;
    input.expectedAutonomicCommandGPUAddress = autonomicCommands.gpuAddress;
    input.activeSensingCommandMetalBuffer =
        (__bridge void*)activeSensingCommands;
    input.activeSensingCommandByteCount =
        MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
    input.expectedActiveSensingCommandGPUAddress =
        activeSensingCommands.gpuAddress;

    input.environmentCount = kEnvironmentCount;
    input.muscleCount = kMuscleCount;
    input.stepCount = kStepCount;
    input.timestepSeconds = 0.002f;
    input.receptorTimestampMicroseconds = kBrainTimestampMicroseconds;
    refreshTransactionIdentity(input, 201u, 4001u);
}

[[nodiscard]] ProbeFailure fail(const int code, std::string message) {
    return {code, std::move(message)};
}

[[nodiscard]] bool checkDiagnostics(
    const MetalNumanXHumanIODiagnostics& diagnostics,
    ProbeFailure& failure,
    const int code,
    const char* operation
) {
    if (diagnostics.succeeded()) {
        return true;
    }
    failure = fail(
        code,
        std::string(operation) + " failed [" +
            metalrobo::metalNumanXHumanIOStatusName(diagnostics.status) +
            "]: " + diagnostics.message
    );
    return false;
}

void candidateCompletion(
    void* raw,
    const MetalNumanXHumanIOCandidateCompletionStatus status,
    const MetalNumanXHumanIOTransactionKey& key,
    const MetalNumanXHumanIOSensorView& view
) noexcept {
    auto& audit = *static_cast<CandidateCompletionAudit*>(raw);
    bool safe = audit.context != nullptr && key.valid() &&
        key.transactionFingerprint ==
            audit.expectedKey.transactionFingerprint &&
        key.programFingerprint == audit.expectedKey.programFingerprint &&
        key.sensorFingerprint == audit.expectedKey.sensorFingerprint &&
        key.transactionInstanceFingerprint ==
            audit.expectedKey.transactionInstanceFingerprint &&
        key.sensorGeneration == audit.expectedKey.sensorGeneration &&
        key.commandBufferIdentity ==
            audit.expectedKey.commandBufferIdentity &&
        view.transactionInstanceFingerprint ==
            key.transactionInstanceFingerprint &&
        view.commandBufferIdentity == key.commandBufferIdentity;
    if (safe) {
        MetalNumanXHumanIOTransactionKey reenteredKey{};
        MetalNumanXHumanIOSensorView reenteredView{};
        const auto pending = audit.context->pendingCandidate(
            reenteredKey, reenteredView);
        const auto duplicate = audit.context->registerCandidateCompletion(
            key, &audit, &candidateCompletion);
        safe = pending.succeeded() && duplicate.status ==
                MetalNumanXHumanIOStatus::contextBusy &&
            reenteredKey.transactionInstanceFingerprint ==
                key.transactionInstanceFingerprint &&
            reenteredView.proprioceptionMetalBuffer ==
                view.proprioceptionMetalBuffer;
    }
    audit.status.store(static_cast<std::uint32_t>(status));
    audit.reentrySafe.store(safe);
    audit.count.fetch_add(1u);
}

[[nodiscard]] bool expectPrepareRejected(
    MetalNumanXHumanIOContext& context,
    const MetalNumanXHumanIOInput& input,
    const char* label,
    ProbeFailure& failure,
    const int code
) {
    MetalNumanXTransactionProgram program{};
    program.fingerprint = 0xfeedfaceu;
    MetalNumanXHumanIOSensorView view{};
    view.sensorGeneration = 0xabcdefu;
    const MetalNumanXHumanIODiagnostics diagnostics = context.prepare(
        input, program, view);
    if (diagnostics.status != MetalNumanXHumanIOStatus::invalidInput ||
        program.fingerprint != 0xfeedfaceu ||
        view.sensorGeneration != 0xabcdefu) {
        failure = fail(
            code,
            std::string(label) +
                " was not rejected before candidate/program publication"
        );
        return false;
    }
    return true;
}

void initializeHumanInputs(const BorrowedHumanBuffers& buffers) {
    auto* states = static_cast<MRMujocoMuscleStateGPU*>(
        buffers.states.contents
    );
    auto* results = static_cast<MRMujocoMuscleResultGPU*>(
        buffers.results.contents
    );
    auto* statuses = static_cast<MRNumiHumanStandStatusGPU*>(
        buffers.statuses.contents
    );
    std::memset(
        states,
        0,
        kMuscleElementCount * sizeof(MRMujocoMuscleStateGPU)
    );
    std::memset(
        results,
        0,
        kMuscleElementCount * sizeof(MRMujocoMuscleResultGPU)
    );
    std::memset(
        statuses,
        0,
        kEnvironmentCount * sizeof(MRNumiHumanStandStatusGPU)
    );
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        statuses[environment].code = MR_NUMI_HUMAN_STAND_SUCCESS;
        statuses[environment].environment = environment;
        statuses[environment].completedSteps = kStepCount;
        statuses[environment].failingIndex = MR_INVALID_INDEX;
        for (std::uint32_t muscle = 0u;
             muscle < kMuscleCount;
             ++muscle) {
            const std::size_t index =
                static_cast<std::size_t>(environment) * kMuscleCount +
                muscle;
            states[index].excitationAndActivation = {
                0.0f,
                0.5f,
                0.1f,
                0.0f,
            };
            results[index].status =
                MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS;
            results[index].environment = environment;
            results[index].muscleIndex = muscle;
            results[index].pathForceAndActivationDerivative = {
                1.0f,
                2.0f,
                3.0f,
                4.0f,
            };
            results[index].activeForceAndReserved = {
                5.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            results[index].fiberStateTendonForceResidual = {
                0.1f,
                0.0f,
                6.0f,
                0.01f,
            };
        }
    }
}

[[nodiscard]] MetalNumanXTransactionPass makePass(
    id<MTLCommandBuffer> commandBuffer,
    const BorrowedHumanBuffers& buffers,
    const MetalNumanXTransactionProgram& program,
    const MetalNumanXTransactionPhase phase,
    const float timestepSeconds
) {
    MetalNumanXTransactionPass pass{};
    pass.abiVersion = metalrobo::kMetalNumanXTransactionABIVersion;
    pass.structSize = sizeof(MetalNumanXTransactionPass);
    pass.accessFlags = metalrobo::MetalNumanXTransactionReadBorrowedState;
    if (phase == MetalNumanXTransactionPhase::beginStep) {
        pass.accessFlags |=
            metalrobo::MetalNumanXTransactionWriteMujocoExcitation;
    } else if (phase == MetalNumanXTransactionPhase::postDynamics) {
        pass.accessFlags |=
            metalrobo::MetalNumanXTransactionWriteStandFailure;
    }
    pass.commandBuffer = (__bridge void*)commandBuffer;
    pass.mujocoStates = (__bridge void*)buffers.states;
    pass.mujocoResults = (__bridge void*)buffers.results;
    pass.standStatuses = (__bridge void*)buffers.statuses;
    pass.phase = phase;
    pass.programFingerprint = program.fingerprint;
    pass.stepIndex = 0u;
    pass.stepCount = kStepCount;
    pass.timestepSeconds = timestepSeconds;
    pass.environmentCount = kEnvironmentCount;
    pass.mujocoMuscleCount = kMuscleCount;
    pass.mujocoStateElementCount = kMuscleElementCount;
    pass.mujocoStateStride = kMuscleCount;
    pass.mujocoResultElementCount = kMuscleElementCount;
    pass.mujocoResultStride = kMuscleCount;
    pass.standStatusElementCount = kEnvironmentCount;
    pass.standStatusStride = 1u;
    return pass;
}

[[nodiscard]] bool encodeAndComplete(
    id<MTLCommandQueue> queue,
    MetalNumanXHumanIOContext& context,
    const BorrowedHumanBuffers& buffers,
    const MetalNumanXTransactionProgram& program,
    const float timestepSeconds,
    EncodedCandidate& candidate,
    ProbeFailure& failure,
    const int codeBase,
    CandidateCompletionAudit* completionAudit = nullptr
) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        failure = fail(codeBase, "failed to create probe command buffer");
        return false;
    }
    for (const MetalNumanXTransactionPhase phase : {
             MetalNumanXTransactionPhase::beginStep,
             MetalNumanXTransactionPhase::preDynamics,
             MetalNumanXTransactionPhase::postDynamics,
         }) {
        const MetalNumanXTransactionPass pass = makePass(
            commandBuffer,
            buffers,
            program,
            phase,
            timestepSeconds
        );
        if (!program.encode(program.context, pass)) {
            program.abort(program.context, (__bridge void*)commandBuffer);
            failure = fail(
                codeBase + 1,
                "adapter rejected a valid probe phase encoding"
            );
            return false;
        }
    }
    if (completionAudit != nullptr) {
        MetalNumanXHumanIODiagnostics diagnostics =
            context.pendingCandidate(candidate.key, candidate.view);
        if (!checkDiagnostics(
                diagnostics,
                failure,
                codeBase + 2,
                "pre-completion pendingCandidate") ||
            !candidate.key.valid()) {
            return false;
        }
        completionAudit->context = &context;
        completionAudit->expectedKey = candidate.key;
        diagnostics = context.registerCandidateCompletion(
            candidate.key, completionAudit, &candidateCompletion);
        if (!diagnostics.succeeded() ||
            context.registerCandidateCompletion(
                candidate.key,
                completionAudit,
                &candidateCompletion).status !=
                MetalNumanXHumanIOStatus::contextBusy) {
            failure = fail(
                codeBase + 2,
                "pre-completion callback registration was not exact-once");
            return false;
        }
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        failure = fail(
            codeBase + 2,
            "probe command buffer did not complete successfully"
        );
        return false;
    }
    if (completionAudit != nullptr) {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(3);
        while (completionAudit->count.load() == 0u &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
    }
    if (completionAudit != nullptr &&
        (completionAudit->count.load() != 1u ||
         completionAudit->status.load() != static_cast<std::uint32_t>(
             MetalNumanXHumanIOCandidateCompletionStatus::succeeded) ||
         !completionAudit->reentrySafe.load())) {
        failure = fail(
            codeBase + 2,
            "pre-completion callback was not asynchronous, exact-once, and reentrant-safe");
        return false;
    }

    MetalNumanXHumanIODiagnostics diagnostics = context.pendingCandidate(
        candidate.key,
        candidate.view
    );
    if (!checkDiagnostics(
            diagnostics,
            failure,
            codeBase + 3,
            "pendingCandidate"
        )) {
        return false;
    }
    if (!candidate.key.valid() ||
        candidate.key.commandBufferIdentity !=
            reinterpret_cast<std::uintptr_t>(
                (__bridge void*)commandBuffer
            ) ||
        candidate.view.commandBufferIdentity !=
            candidate.key.commandBufferIdentity ||
        candidate.view.transactionInstanceFingerprint !=
            candidate.key.transactionInstanceFingerprint) {
        failure = fail(
            codeBase + 4,
            "candidate key/view did not preserve exact command identity"
        );
        return false;
    }
    return true;
}

[[nodiscard]] std::vector<std::uint8_t> copyPrivateBuffer(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    void* sourceObject,
    const std::size_t byteCount,
    ProbeFailure& failure,
    const int code
) {
    std::vector<std::uint8_t> output;
    __unsafe_unretained id<MTLBuffer> source =
        (__bridge id<MTLBuffer>)sourceObject;
    id<MTLBuffer> staging = [device
        newBufferWithLength:byteCount
                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder =
        [commandBuffer blitCommandEncoder];
    if (source == nil || staging == nil || commandBuffer == nil ||
        encoder == nil) {
        failure = fail(code, "failed to allocate replay readback staging");
        return output;
    }
    [encoder
        copyFromBuffer:source
           sourceOffset:0u
               toBuffer:staging
      destinationOffset:0u
                   size:byteCount];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        failure = fail(code, "replay readback blit did not complete");
        return output;
    }
    output.resize(byteCount);
    std::memcpy(output.data(), staging.contents, byteCount);
    return output;
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationBinding publicationBinding(
    const MetalNumanXHumanIOCandidatePublicationLease& lease,
    const std::uint32_t controlStep
) noexcept {
    const auto& program = lease.program();
    MetalNumanXHumanIOCandidatePublicationBinding binding{};
    binding.environmentCount = kEnvironmentCount;
    binding.transactionSlot = 0u;
    binding.stepIndex = 0u;
    binding.substepIndex = 0u;
    binding.physicsSubstepCount = 1u;
    binding.controlStep = controlStep;
    binding.ownerProgramFingerprint = 0x48494f010001ull;
    binding.transactionFingerprint = program.transactionFingerprint;
    binding.linearizationEpoch = 0x48494f010002ull;
    binding.slotGeneration = controlStep + 1u;
    binding.physicsTokenFingerprint = 0x48494f010003ull;
    binding.proposalFingerprint = 0x48494f010004ull;
    binding.ackFingerprint = 0x48494f010005ull;
    binding.appliedDecisionFingerprint = 0x48494f010006ull;
    binding.jointCommitFingerprint = 0x48494f010007ull;
    binding.brainGeneration = program.acceptedBrainGeneration;
    binding.candidateKeyFingerprint = program.candidateKeyFingerprint;
    binding.acceptedBrainGeneration = program.acceptedBrainGeneration;
    binding.humanIOProgramFingerprint =
        program.humanIOProgramFingerprint;
    binding.sensorFingerprint = program.sensorFingerprint;
    binding.transactionInstanceFingerprint =
        program.transactionInstanceFingerprint;
    binding.candidatePublicationFingerprint =
        program.candidatePublicationFingerprint;
    binding.deviceRegistryID = program.deviceRegistryID;
    binding.sensorGeneration = program.sensorGeneration;
    binding.humanIOIdentityFingerprint = program.identityFingerprint;
    binding.bindingFingerprint =
        metalrobo::metalNumanXHumanIOPublicationBindingFingerprint(binding);
    return binding;
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationCommit publicationCommit(
    const MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept {
    MetalNumanXHumanIOCandidatePublicationCommit commit{};
    commit.candidatePublicationFingerprint =
        binding.candidatePublicationFingerprint;
    commit.bindingFingerprint = binding.bindingFingerprint;
    commit.jointCommitFingerprint = binding.jointCommitFingerprint;
    commit.brainGeneration = binding.brainGeneration;
    commit.fenceFingerprint = 0x48494f010008ull;
    return commit;
}

[[nodiscard]] bool samePublishedIdentity(
    const MetalNumanXHumanIOSensorView& first,
    const MetalNumanXHumanIOSensorView& second
) noexcept {
    return first.proprioceptionMetalBuffer ==
            second.proprioceptionMetalBuffer &&
        first.validityMetalBuffer == second.validityMetalBuffer &&
        first.interoceptionMetalBuffer ==
            second.interoceptionMetalBuffer &&
        first.interoceptionValidityMetalBuffer ==
            second.interoceptionValidityMetalBuffer &&
        first.proprioceptionGPUAddress ==
            second.proprioceptionGPUAddress &&
        first.validityGPUAddress == second.validityGPUAddress &&
        first.interoceptionGPUAddress ==
            second.interoceptionGPUAddress &&
        first.interoceptionValidityGPUAddress ==
            second.interoceptionValidityGPUAddress &&
        first.motorOutputHeaderGPUAddress ==
            second.motorOutputHeaderGPUAddress &&
        first.sensorGeneration == second.sensorGeneration &&
        first.sensorFingerprint == second.sensorFingerprint &&
        first.transactionInstanceFingerprint ==
            second.transactionInstanceFingerprint &&
        first.commandBufferIdentity == second.commandBufferIdentity &&
        first.receptorTimestampMicroseconds ==
            second.receptorTimestampMicroseconds;
}

[[nodiscard]] int runProbe(const char* metallibPath) {
    @autoreleasepool {
        ProbeFailure failure{};
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (device == nil || queue == nil) {
            std::fprintf(stderr, "Metal device/queue unavailable\n");
            return 2;
        }

        id<MTLBuffer> excitation = [device
            newBufferWithLength:kMuscleElementCount * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> motorHeaders = [device
            newBufferWithLength:
                kEnvironmentCount *
                    sizeof(MRNumanXBrainMotorOutputHeaderGPU)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> autonomicCommands = [device
            newBufferWithLength:MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> activeSensingCommands = [device
            newBufferWithLength:
                MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> motorReadyGate = [device
            newBufferWithLength:MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT
                       options:MTLResourceStorageModeShared];
        id<MTLSharedEvent> motorReadyEvent = [device newSharedEvent];
        BorrowedHumanBuffers buffers{};
        buffers.states = [device
            newBufferWithLength:
                kMuscleElementCount * sizeof(MRMujocoMuscleStateGPU)
                       options:MTLResourceStorageModeShared];
        buffers.results = [device
            newBufferWithLength:
                kMuscleElementCount * sizeof(MRMujocoMuscleResultGPU)
                       options:MTLResourceStorageModeShared];
        buffers.statuses = [device
            newBufferWithLength:
                kEnvironmentCount * sizeof(MRNumiHumanStandStatusGPU)
                       options:MTLResourceStorageModeShared];
        if (excitation == nil || motorHeaders == nil ||
            autonomicCommands == nil || activeSensingCommands == nil ||
            motorReadyGate == nil || motorReadyEvent == nil ||
            buffers.states == nil ||
            buffers.results == nil || buffers.statuses == nil ||
            excitation.gpuAddress == 0u ||
            motorHeaders.gpuAddress == 0u ||
            autonomicCommands.gpuAddress == 0u ||
            activeSensingCommands.gpuAddress == 0u ||
            motorReadyGate.gpuAddress == 0u) {
            std::fprintf(stderr, "probe buffer allocation failed\n");
            return 3;
        }
        float* excitationValues = static_cast<float*>(
            excitation.contents
        );
        for (std::size_t index = 0u;
             index < kMuscleElementCount;
             ++index) {
            excitationValues[index] = 0.25f;
        }
        initializeMotorHeaders(
            static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
                motorHeaders.contents
            ),
            excitationValues
        );

        metalrobo::MetalNumanXHumanIOConfig config{};
        config.metallibPath = metallibPath;
        MetalNumanXHumanIOContext context(config);
        MetalNumanXHumanIOInput input{};
        initializeAuthoritativeInput(
            input,
            motorHeaders,
            excitation,
            autonomicCommands,
            activeSensingCommands
        );
        const auto bindMotorReadyLease = [&] (
            MetalNumanXHumanIOInput& candidateInput
        ) {
            candidateInput.motorReadyGateMetalBuffer =
                (__bridge void*)motorReadyGate;
            candidateInput.motorReadyGateByteOffset = 0u;
            candidateInput.motorReadyGateByteCount =
                MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT;
            candidateInput.expectedMotorReadyGateGPUAddress =
                motorReadyGate.gpuAddress;
            candidateInput.motorReadySharedEvent =
                (__bridge void*)motorReadyEvent;
            candidateInput.motorReadySharedEventValue = 1u;
        };

        const auto rejectInput = [&] (
            const MetalNumanXHumanIOInput& rejected,
            const char* label,
            const int code
        ) -> bool {
            return expectPrepareRejected(
                context, rejected, label, failure, code);
        };
        MetalNumanXHumanIOInput invalid = input;
        invalid.environmentCount = 2u;
        if (!rejectInput(invalid, "multi-environment candidate", 4)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.root.transactionFingerprint ^= 1u;
        if (!rejectInput(invalid, "wrong root fingerprint", 5)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        ++invalid.substep.randomCounterGeneration;
        invalid.substep.substepFingerprint =
            metalrobo::metalNumanXBrainJointSubstepFingerprint(
                invalid.substep);
        if (!rejectInput(invalid, "wrong substep random generation", 6)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.timestepSeconds = std::nextafter(
            input.timestepSeconds,
            std::numeric_limits<float>::infinity()
        );
        if (!rejectInput(invalid, "noncanonical timestep float", 7)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        ++invalid.receptorTimestampMicroseconds;
        if (!rejectInput(invalid, "wrong exact receptor timestamp", 7)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        ++invalid.candidate.acceptedBrainTimestampMicroseconds;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "wrong accepted brain timestamp", 7)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        ++invalid.candidate.brainGeneration;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "wrong candidate brain generation", 8)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.flags |=
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
        bindMotorReadyLease(invalid);
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(
                invalid,
                "decision-shadow candidate with base generation",
                8
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.substep.substepIndex = 1u;
        invalid.substep.substepFingerprint =
            metalrobo::metalNumanXBrainJointSubstepFingerprint(
                invalid.substep);
        invalid.candidate.flags |=
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
        bindMotorReadyLease(invalid);
        invalid.candidate.substepFingerprint =
            invalid.substep.substepFingerprint;
        invalid.candidate.brainGeneration = input.root.shadowGeneration;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(
                invalid,
                "decision-shadow candidate after substep zero",
                8
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.speciesTemplateFingerprint = 0u;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "zero species template", 9)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.compiledSpeciesTemplateFingerprint = 0u;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "zero compiled species template", 10)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.actuatorCommandKind = 2u;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "wrong Human actuator kind", 11)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.muscleExcitationByteCount += sizeof(float);
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "wrong somatic byte count", 12)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.muscleExcitationGPUAddress += sizeof(float);
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "wrong somatic lease address", 13)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.candidateFingerprint ^= 1u;
        if (!rejectInput(invalid, "wrong candidate fingerprint", 14)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.expectedAutonomicCommandGPUAddress += sizeof(float);
        if (!rejectInput(invalid, "wrong autonomic lease", 15)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.activeSensingCommandMetalBuffer = (__bridge void*)excitation;
        invalid.expectedActiveSensingCommandGPUAddress = excitation.gpuAddress;
        invalid.candidate.activeSensingCommandGPUAddress =
            excitation.gpuAddress;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "overlapping candidate leases", 16)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.motorOutputHeaderMetalBuffer = nullptr;
        if (!rejectInput(invalid, "null motor-header lease", 17)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.excitationMetalBuffer = nullptr;
        if (!rejectInput(invalid, "null somatic lease", 18)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.autonomicCommandMetalBuffer = nullptr;
        if (!rejectInput(invalid, "null autonomic lease", 19)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.activeSensingCommandMetalBuffer = nullptr;
        if (!rejectInput(invalid, "null active-sensing lease", 20)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.excitationByteOffset = 1u;
        if (!rejectInput(invalid, "misaligned somatic lease", 21)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.autonomicCommandByteOffset = autonomicCommands.length;
        invalid.expectedAutonomicCommandGPUAddress =
            autonomicCommands.gpuAddress + autonomicCommands.length;
        invalid.candidate.autonomicCommandGPUAddress =
            invalid.expectedAutonomicCommandGPUAddress;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "out-of-range autonomic lease", 22)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        invalid = input;
        invalid.candidate.activeSensingCommandByteCount +=
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
        invalid.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                invalid.candidate);
        if (!rejectInput(invalid, "wrong active-sensing size", 23)) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }

        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        for (id<MTLDevice> candidateDevice in devices) {
            if (candidateDevice.registryID == device.registryID) continue;
            id<MTLBuffer> foreign = [candidateDevice
                newBufferWithLength:
                    MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT
                           options:MTLResourceStorageModeShared];
            if (foreign == nil || foreign.gpuAddress == 0u) continue;
            invalid = input;
            invalid.autonomicCommandMetalBuffer = (__bridge void*)foreign;
            invalid.expectedAutonomicCommandGPUAddress = foreign.gpuAddress;
            invalid.candidate.autonomicCommandGPUAddress = foreign.gpuAddress;
            invalid.candidate.candidateFingerprint =
                metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                    invalid.candidate);
            if (!rejectInput(invalid, "wrong-device autonomic lease", 24)) {
                std::fprintf(stderr, "%s\n", failure.message.c_str());
                return failure.code;
            }
            break;
        }

        MetalNumanXHumanIOInput decisionShadow = input;
        decisionShadow.candidate.flags |=
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
        bindMotorReadyLease(decisionShadow);
        decisionShadow.candidate.brainGeneration =
            decisionShadow.root.shadowGeneration;
        decisionShadow.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                decisionShadow.candidate);
        initializeMotorHeaders(
            static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
                motorHeaders.contents),
            excitationValues,
            decisionShadow.root.shadowGeneration
        );
        MetalNumanXTransactionProgram decisionShadowProgram{};
        MetalNumanXHumanIOSensorView decisionShadowView{};
        MetalNumanXHumanIODiagnostics diagnostics = context.prepare(
            decisionShadow,
            decisionShadowProgram,
            decisionShadowView
        );
        if (!checkDiagnostics(
                diagnostics,
                failure,
                25,
                "decision-shadow prepare"
            ) || !decisionShadowProgram.valid() ||
            decisionShadowView.acceptedBrainGeneration !=
                decisionShadow.root.shadowGeneration) {
            if (failure.message.empty()) {
                failure = fail(
                    26,
                    "decision-shadow authority was not preserved"
                );
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        diagnostics = context.cancelPrepared(
            decisionShadow.root.transactionFingerprint,
            decisionShadowProgram.fingerprint
        );
        if (!diagnostics.succeeded()) {
            std::fprintf(stderr, "decision-shadow prepare did not cancel\n");
            return 27;
        }
        initializeMotorHeaders(
            static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
                motorHeaders.contents),
            excitationValues
        );

        MetalNumanXTransactionProgram program{};
        MetalNumanXHumanIOSensorView preparedView{};
        diagnostics = context.prepare(
            input,
            program,
            preparedView
        );
        if (!checkDiagnostics(
                diagnostics,
                failure,
                10,
                "initial prepare"
            ) || !program.valid() ||
            preparedView.transactionInstanceFingerprint != 0u ||
            preparedView.motorOutputHeaderGPUAddress !=
                motorHeaders.gpuAddress ||
            preparedView.receptorTimestampMicroseconds !=
                input.receptorTimestampMicroseconds ||
            preparedView.receptorTimeSeconds !=
                static_cast<double>(input.receptorTimestampMicroseconds) /
                    1'000'000.0 ||
            preparedView.deliveryTimeSeconds !=
                preparedView.receptorTimeSeconds + input.timestepSeconds ||
            preparedView.latencySeconds != input.timestepSeconds) {
            if (failure.message.empty()) {
                failure = fail(11, "initial program/view metadata is invalid");
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        initializeHumanInputs(buffers);
        EncodedCandidate acceptedCandidate{};
        if (!encodeAndComplete(
                queue,
                context,
                buffers,
                program,
                input.timestepSeconds,
                acceptedCandidate,
                failure,
                20
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        CandidateCompletionAudit synchronousCompletion{};
        synchronousCompletion.context = &context;
        synchronousCompletion.expectedKey = acceptedCandidate.key;
        diagnostics = context.registerCandidateCompletion(
            acceptedCandidate.key,
            &synchronousCompletion,
            &candidateCompletion);
        if (!diagnostics.succeeded() ||
            synchronousCompletion.count.load() != 1u ||
            synchronousCompletion.status.load() !=
                static_cast<std::uint32_t>(
                    MetalNumanXHumanIOCandidateCompletionStatus::succeeded) ||
            !synchronousCompletion.reentrySafe.load() ||
            context.registerCandidateCompletion(
                acceptedCandidate.key,
                &synchronousCompletion,
                &candidateCompletion).status !=
                MetalNumanXHumanIOStatus::contextBusy) {
            std::fprintf(
                stderr,
                "already-completed candidate callback was not synchronous, exact-once, and reentrant-safe\n");
            return 25;
        }
        const std::vector<std::uint8_t> acceptedPayload =
            copyPrivateBuffer(
                device,
                queue,
                acceptedCandidate.view.proprioceptionMetalBuffer,
                acceptedCandidate.view.proprioceptionByteCount,
                failure,
                25
            );
        const std::vector<std::uint8_t> acceptedValidity =
            copyPrivateBuffer(
                device,
                queue,
                acceptedCandidate.view.validityMetalBuffer,
                acceptedCandidate.view.validityByteCount,
                failure,
                26
            );
        const std::vector<std::uint8_t> acceptedInteroception =
            copyPrivateBuffer(
                device,
                queue,
                acceptedCandidate.view.interoceptionMetalBuffer,
                acceptedCandidate.view.interoceptionByteCount,
                failure,
                26
            );
        const std::vector<std::uint8_t> acceptedInteroceptionValidity =
            copyPrivateBuffer(
                device,
                queue,
                acceptedCandidate.view.interoceptionValidityMetalBuffer,
                acceptedCandidate.view.interoceptionValidityByteCount,
                failure,
                26
            );
        if (!failure.message.empty()) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        if (acceptedPayload.size() <
                MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT *
                    sizeof(float) ||
            acceptedValidity.size() < sizeof(std::uint32_t)) {
            std::fprintf(stderr, "candidate payload size is malformed\n");
            return 27;
        }
        const float* firstReceptor = reinterpret_cast<const float*>(
            acceptedPayload.data()
        );
        const auto* firstValidity =
            reinterpret_cast<const std::uint32_t*>(
                acceptedValidity.data()
            );
        const float* firstInteroception =
            reinterpret_cast<const float*>(acceptedInteroception.data());
        const auto* firstInteroceptionValidity =
            reinterpret_cast<const std::uint32_t*>(
                acceptedInteroceptionValidity.data());
        const float expectedFeatures[] = {
            0.25f,
            0.5f,
            0.1f,
            0.0f,
            1.0f,
            2.0f,
            5.0f,
            6.0f,
            4.0f,
            0.01f,
        };
        for (std::uint32_t feature = 0u;
             feature <
                MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
             ++feature) {
            if (!closeEnough(firstReceptor[feature], expectedFeatures[feature])) {
                std::fprintf(
                    stderr,
                    "feature %u mismatch: %.9g != %.9g\n",
                    feature,
                    firstReceptor[feature],
                    expectedFeatures[feature]
                );
                return 28;
            }
        }
        if (firstValidity[0] !=
                MR_NUMANX_HUMAN_PROPRIOCEPTION_VALIDITY_ALL ||
            acceptedInteroception.size() < sizeof(float) ||
            acceptedInteroceptionValidity.size() < sizeof(std::uint32_t) ||
            !closeEnough(firstInteroception[0], 0.375f) ||
            firstInteroceptionValidity[0] !=
                MR_NUMANX_HUMAN_INTEROCEPTION_VALIDITY_ALL) {
            std::fprintf(stderr, "valid candidate mask is not complete\n");
            return 29;
        }

        MetalNumanXHumanIOSensorView published{};
        MetalNumanXHumanIOCandidatePublicationLease publicationLease;
        diagnostics = context.reserveCandidatePublication(
            acceptedCandidate.key,
            publicationLease
        );
        if (!checkDiagnostics(
                diagnostics,
                failure,
                30,
                "valid candidate publication reservation"
            ) || diagnostics.published || !publicationLease.valid() ||
            !publicationLease.program().valid() ||
            publicationLease.view().proprioception.metalBuffer !=
                acceptedCandidate.view.proprioceptionMetalBuffer ||
            publicationLease.view().proprioception.byteOffset != 0u ||
            publicationLease.view().proprioception.byteCount !=
                acceptedCandidate.view.proprioceptionByteCount ||
            publicationLease.view().validity.metalBuffer !=
                acceptedCandidate.view.validityMetalBuffer ||
            publicationLease.view().validity.byteOffset != 0u ||
            publicationLease.view().validity.byteCount !=
                acceptedCandidate.view.validityByteCount ||
            publicationLease.view().interoception.metalBuffer !=
                acceptedCandidate.view.interoceptionMetalBuffer ||
            publicationLease.view().interoception.byteCount !=
                acceptedCandidate.view.interoceptionByteCount ||
            publicationLease.view().interoceptionValidity.metalBuffer !=
                acceptedCandidate.view.interoceptionValidityMetalBuffer ||
            publicationLease.view().interoceptionValidity.byteCount !=
                acceptedCandidate.view.interoceptionValidityByteCount) {
            if (failure.message.empty()) {
                failure = fail(
                    31,
                    "candidate reservation lost exact lease metadata"
                );
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        diagnostics = context.publishedView(published);
        if (diagnostics.status !=
            MetalNumanXHumanIOStatus::candidateUnavailable) {
            std::fprintf(stderr, "sensor candidate published before root close\n");
            return 32;
        }
        const auto publicationProgram = publicationLease.program();
        MetalNumanXHumanIOCandidatePublicationBinding binding =
            publicationBinding(publicationLease, 37u);
        auto staleBinding = binding;
        ++staleBinding.sensorGeneration;
        staleBinding.bindingFingerprint =
            metalrobo::metalNumanXHumanIOPublicationBindingFingerprint(
                staleBinding
            );
        if (publicationProgram.reservePublishedRoot(
                publicationProgram.context,
                publicationProgram.candidatePublicationFingerprint,
                staleBinding
            ) ||
            !publicationProgram.reservePublishedRoot(
                publicationProgram.context,
                publicationProgram.candidatePublicationFingerprint,
                binding
            ) ||
            publicationProgram.reservePublishedRoot(
                publicationProgram.context,
                publicationProgram.candidatePublicationFingerprint,
                binding
            )) {
            std::fprintf(stderr, "root binding was not exact and one-shot\n");
            return 33;
        }
        const MetalNumanXHumanIOCandidatePublicationCommit commit =
            publicationCommit(binding);
        if (publicationProgram.publishCandidate(
                publicationProgram.context,
                publicationProgram.candidatePublicationFingerprint,
                commit
            ) != MetalNumanXHumanIOCandidatePublicationDisposition::released ||
            publicationLease.valid()) {
            std::fprintf(stderr, "exact committed root did not publish sensor\n");
            return 34;
        }
        diagnostics = context.publishedView(published);
        if (!checkDiagnostics(
                diagnostics,
                failure,
                35,
                "published view after exact root close"
            ) || !diagnostics.published) {
            if (failure.message.empty()) {
                failure = fail(36, "root close did not expose sensor view");
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }

        // Replay the identical causal receptor inputs in the non-published
        // slot. Payload bytes must match even though provenance/generation and
        // command identity are new.
        refreshTransactionIdentity(input, 202u, 4002u);
        diagnostics = context.prepare(input, program, preparedView);
        if (!checkDiagnostics(
                diagnostics,
                failure,
                40,
                "replay prepare"
            ) || preparedView.proprioceptionMetalBuffer ==
                published.proprioceptionMetalBuffer) {
            if (failure.message.empty()) {
                failure = fail(41, "replay did not select non-published slot");
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        initializeHumanInputs(buffers);
        EncodedCandidate replayCandidate{};
        CandidateCompletionAudit asynchronousCompletion{};
        if (!encodeAndComplete(
                queue,
                context,
                buffers,
                program,
                input.timestepSeconds,
                replayCandidate,
                failure,
                42,
                &asynchronousCompletion
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        const std::vector<std::uint8_t> replayPayload =
            copyPrivateBuffer(
                device,
                queue,
                replayCandidate.view.proprioceptionMetalBuffer,
                replayCandidate.view.proprioceptionByteCount,
                failure,
                47
            );
        const std::vector<std::uint8_t> replayValidity =
            copyPrivateBuffer(
                device,
                queue,
                replayCandidate.view.validityMetalBuffer,
                replayCandidate.view.validityByteCount,
                failure,
                48
            );
        const std::vector<std::uint8_t> replayInteroception =
            copyPrivateBuffer(
                device,
                queue,
                replayCandidate.view.interoceptionMetalBuffer,
                replayCandidate.view.interoceptionByteCount,
                failure,
                48
            );
        const std::vector<std::uint8_t> replayInteroceptionValidity =
            copyPrivateBuffer(
                device,
                queue,
                replayCandidate.view.interoceptionValidityMetalBuffer,
                replayCandidate.view.interoceptionValidityByteCount,
                failure,
                48
            );
        if (!failure.message.empty() || replayPayload != acceptedPayload ||
            replayValidity != acceptedValidity ||
            replayInteroception != acceptedInteroception ||
            replayInteroceptionValidity != acceptedInteroceptionValidity) {
            if (failure.message.empty()) {
                failure = fail(49, "proprioception replay is not byte-identical");
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }

        auto wrongKey = replayCandidate.key;
        wrongKey.transactionInstanceFingerprint ^= 1u;
        MetalNumanXHumanIOCandidatePublicationLease wronglyReserved;
        diagnostics = context.reserveCandidatePublication(
            wrongKey,
            wronglyReserved
        );
        if (diagnostics.status !=
            MetalNumanXHumanIOStatus::incompatibleTransaction) {
            std::fprintf(stderr, "stale candidate key was not rejected\n");
            return 50;
        }
        MetalNumanXHumanIOCandidatePublicationLease rejectedLease;
        diagnostics = context.reserveCandidatePublication(
            replayCandidate.key,
            rejectedLease
        );
        if (!diagnostics.succeeded() || !rejectedLease.valid()) {
            std::fprintf(stderr, "valid reject candidate was not reserved\n");
            return 51;
        }
        diagnostics = context.reject(replayCandidate.key);
        if (diagnostics.status != MetalNumanXHumanIOStatus::contextBusy) {
            std::fprintf(stderr, "context bypassed move-only candidate lease\n");
            return 52;
        }
        const auto rejectedProgram = rejectedLease.program();
        if (rejectedProgram.rejectCandidate(
                rejectedProgram.context,
                rejectedProgram.candidatePublicationFingerprint
            ) != MetalNumanXHumanIOCandidatePublicationDisposition::rejected ||
            rejectedLease.valid()) {
            std::fprintf(stderr, "exact leased candidate was not rejected\n");
            return 53;
        }
        MetalNumanXHumanIOSensorView stable{};
        diagnostics = context.publishedView(stable);
        if (!diagnostics.succeeded() ||
            !samePublishedIdentity(stable, published)) {
            std::fprintf(stderr, "identity rejection changed published view\n");
            return 54;
        }
        const std::vector<std::uint8_t> identityStablePayload =
            copyPrivateBuffer(
                device,
                queue,
                stable.proprioceptionMetalBuffer,
                stable.proprioceptionByteCount,
                failure,
                55
            );
        const std::vector<std::uint8_t> identityStableValidity =
            copyPrivateBuffer(
                device,
                queue,
                stable.validityMetalBuffer,
                stable.validityByteCount,
                failure,
                56
            );
        if (!failure.message.empty() ||
            identityStablePayload != acceptedPayload ||
            identityStableValidity != acceptedValidity) {
            if (failure.message.empty()) {
                failure = fail(
                    57,
                    "identity rejection changed published payload bytes"
                );
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        // The candidate FNV authenticates metadata and addresses, not mutable
        // payload bytes. Mutating a finite command after host prepare must be
        // caught by same-timeline header+payload authentication.
        refreshTransactionIdentity(input, 203u, 4003u);
        diagnostics = context.prepare(input, program, preparedView);
        if (!checkDiagnostics(
                diagnostics,
                failure,
                58,
                "post-prepare payload-mutation prepare"
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        excitationValues[0] = 0.5f;
        initializeHumanInputs(buffers);
        EncodedCandidate corruptHeaderCandidate{};
        if (!encodeAndComplete(
                queue,
                context,
                buffers,
                program,
                input.timestepSeconds,
                corruptHeaderCandidate,
                failure,
                59
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        const auto* corruptHeaderStatuses =
            static_cast<const MRNumiHumanStandStatusGPU*>(
                buffers.statuses.contents
            );
        if (corruptHeaderStatuses[0].code !=
                MR_NUMI_HUMAN_STAND_INVALID_DISPATCH ||
            corruptHeaderStatuses[0].failingIndex != 0u) {
            std::fprintf(stderr, "post-prepare motor payload mutation was not rejected\n");
            return 64;
        }
        diagnostics = context.reject(corruptHeaderCandidate.key);
        if (!checkDiagnostics(
                diagnostics,
                failure,
                65,
                "corrupt-header explicit reject"
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        diagnostics = context.publishedView(stable);
        if (!diagnostics.succeeded() ||
            !samePublishedIdentity(stable, published)) {
            std::fprintf(stderr, "payload mutation changed prior publication\n");
            return 66;
        }
        excitationValues[0] = 0.25f;

        // An out-of-range motor is safely replaced by zero at beginStep, then
        // propagated into the owning stand status at postDynamics. It cannot
        // be accepted and cannot disturb the prior published generation.
        excitationValues[0] = 1.25f;
        refreshTransactionIdentity(input, 204u, 4004u);
        diagnostics = context.prepare(input, program, preparedView);
        if (!checkDiagnostics(
                diagnostics,
                failure,
                60,
                "invalid-motor prepare"
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        initializeHumanInputs(buffers);
        EncodedCandidate invalidCandidate{};
        if (!encodeAndComplete(
                queue,
                context,
                buffers,
                program,
                input.timestepSeconds,
                invalidCandidate,
                failure,
                61
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        const auto* statuses =
            static_cast<const MRNumiHumanStandStatusGPU*>(
                buffers.statuses.contents
            );
        if (statuses[0].code !=
                MR_NUMI_HUMAN_STAND_INVALID_DISPATCH ||
            statuses[0].failingIndex != 0u) {
            std::fprintf(stderr, "invalid motor was not propagated to Human\n");
            return 66;
        }

        MetalNumanXHumanIOCandidatePublicationLease invalidReservation;
        diagnostics = context.reserveCandidatePublication(
            invalidCandidate.key,
            invalidReservation
        );
        if (!diagnostics.succeeded() || !invalidReservation.valid()) {
            std::fprintf(stderr, "invalid motor candidate was not quarantined\n");
            return 67;
        }
        const auto invalidProgram = invalidReservation.program();
        if (invalidProgram.rejectCandidate(
                invalidProgram.context,
                invalidProgram.candidatePublicationFingerprint
            ) != MetalNumanXHumanIOCandidatePublicationDisposition::rejected) {
            std::fprintf(stderr, "owner-equivalent reject did not release sensor\n");
            return 68;
        }
        diagnostics = context.publishedView(stable);
        if (!diagnostics.succeeded() ||
            !samePublishedIdentity(stable, published)) {
            std::fprintf(stderr, "invalid motor changed prior publication\n");
            return 69;
        }
        const std::vector<std::uint8_t> motorStablePayload =
            copyPrivateBuffer(
                device,
                queue,
                stable.proprioceptionMetalBuffer,
                stable.proprioceptionByteCount,
                failure,
                70
            );
        const std::vector<std::uint8_t> motorStableValidity =
            copyPrivateBuffer(
                device,
                queue,
                stable.validityMetalBuffer,
                stable.validityByteCount,
                failure,
                71
            );
        if (!failure.message.empty() ||
            motorStablePayload != acceptedPayload ||
            motorStableValidity != acceptedValidity) {
            if (failure.message.empty()) {
                failure = fail(
                    72,
                    "invalid motor changed published payload bytes"
                );
            }
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        diagnostics = context.publishedView(stable);
        if (!diagnostics.succeeded() ||
            !samePublishedIdentity(stable, published)) {
            std::fprintf(stderr, "explicit reject changed prior publication\n");
            return 73;
        }

        // A reached liveness event is not motor authority. A terminal FAILURE
        // gate must reject the physical candidate even when all payload bytes
        // are otherwise valid and the event has already advanced.
        excitationValues[0] = 0.25f;
        refreshTransactionIdentity(input, 205u, 4005u);
        input.candidate.flags |=
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
        input.candidate.brainGeneration = input.root.shadowGeneration;
        input.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                input.candidate);
        bindMotorReadyLease(input);
        initializeMotorHeaders(
            static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
                motorHeaders.contents),
            excitationValues,
            input.root.shadowGeneration
        );
        initializeMotorReadyGate(
            input,
            *static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
                motorHeaders.contents),
            *static_cast<MRNumanXBrainMotorReadyGateGPU*>(
                motorReadyGate.contents),
            MR_NUMANX_BRAIN_READY_GATE_FAILURE
        );
        motorReadyEvent.signaledValue = 1u;
        diagnostics = context.prepare(input, program, preparedView);
        if (!diagnostics.succeeded()) {
            std::fprintf(stderr, "failed-gate candidate prepare failed\n");
            return 74;
        }
        initializeHumanInputs(buffers);
        EncodedCandidate failedGateCandidate{};
        if (!encodeAndComplete(
                queue,
                context,
                buffers,
                program,
                input.timestepSeconds,
                failedGateCandidate,
                failure,
                75
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        if (static_cast<const MRNumiHumanStandStatusGPU*>(
                buffers.statuses.contents)[0].code !=
                MR_NUMI_HUMAN_STAND_INVALID_DISPATCH) {
            std::fprintf(stderr, "liveness-only failed gate reached Human\n");
            return 79;
        }
        MetalNumanXHumanIOCandidatePublicationLease failedGateLease;
        diagnostics = context.reserveCandidatePublication(
            failedGateCandidate.key,
            failedGateLease
        );
        if (!diagnostics.succeeded() || !failedGateLease.valid() ||
            failedGateLease.program().rejectCandidate(
                failedGateLease.program().context,
                failedGateLease.program().candidatePublicationFingerprint
            ) != MetalNumanXHumanIOCandidatePublicationDisposition::rejected) {
            std::fprintf(stderr, "failed-gate candidate did not reject\n");
            return 80;
        }

        // A valid decision-shadow SUCCESS gate reaches the real Human GPU
        // admission path. Once its exact root reservation exists, a malformed
        // COMMITTED close is terminal and cannot be replaced later.
        refreshTransactionIdentity(input, 206u, 4006u);
        input.candidate.candidateFingerprint =
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(
                input.candidate);
        initializeMotorReadyGate(
            input,
            *static_cast<MRNumanXBrainMotorOutputHeaderGPU*>(
                motorHeaders.contents),
            *static_cast<MRNumanXBrainMotorReadyGateGPU*>(
                motorReadyGate.contents)
        );
        diagnostics = context.prepare(input, program, preparedView);
        if (!diagnostics.succeeded()) {
            std::fprintf(stderr, "terminal-close candidate prepare failed\n");
            return 81;
        }
        initializeHumanInputs(buffers);
        EncodedCandidate terminalCandidate{};
        if (!encodeAndComplete(
                queue,
                context,
                buffers,
                program,
                input.timestepSeconds,
                terminalCandidate,
                failure,
                82
            )) {
            std::fprintf(stderr, "%s\n", failure.message.c_str());
            return failure.code;
        }
        if (static_cast<const MRNumiHumanStandStatusGPU*>(
                buffers.statuses.contents)[0].code !=
                MR_NUMI_HUMAN_STAND_SUCCESS) {
            std::fprintf(stderr, "valid motor-ready gate did not reach Human\n");
            return 86;
        }
        MetalNumanXHumanIOCandidatePublicationLease terminalLease;
        diagnostics = context.reserveCandidatePublication(
            terminalCandidate.key,
            terminalLease
        );
        if (!diagnostics.succeeded() || !terminalLease.valid()) {
            std::fprintf(stderr, "terminal-close candidate reserve failed\n");
            return 87;
        }
        const auto terminalProgram = terminalLease.program();
        const auto terminalBinding = publicationBinding(terminalLease, 38u);
        if (!terminalProgram.reservePublishedRoot(
                terminalProgram.context,
                terminalProgram.candidatePublicationFingerprint,
                terminalBinding
            )) {
            std::fprintf(stderr, "terminal-close root binding failed\n");
            return 88;
        }
        auto malformedCommit = publicationCommit(terminalBinding);
        malformedCommit.jointCommitFingerprint ^= 1u;
        if (terminalProgram.publishCandidate(
                terminalProgram.context,
                terminalProgram.candidatePublicationFingerprint,
                malformedCommit
            ) != MetalNumanXHumanIOCandidatePublicationDisposition::
                    terminalNoTouch ||
            terminalProgram.publishCandidate(
                terminalProgram.context,
                terminalProgram.candidatePublicationFingerprint,
                publicationCommit(terminalBinding)
            ) != MetalNumanXHumanIOCandidatePublicationDisposition::
                    terminalNoTouch) {
            std::fprintf(stderr, "malformed close did not quarantine one-shot\n");
            return 89;
        }
        diagnostics = context.publishedView(stable);
        if (!diagnostics.succeeded() ||
            !samePublishedIdentity(stable, published)) {
            std::fprintf(stderr, "terminal close changed prior publication\n");
            return 90;
        }

        std::printf(
            "numanx_human_io_probe passed: device=%s generation=%llu "
            "payload_bytes=%zu validity_bytes=%zu replay=byte-identical "
            "motor_gate=success_and_liveness_failure publication=reserved_one_shot "
            "terminal=quarantined\n",
            device.name.UTF8String,
            static_cast<unsigned long long>(published.sensorGeneration),
            published.proprioceptionByteCount,
            published.validityByteCount
        );
        return 0;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || argv[1] == nullptr || argv[1][0] == '\0') {
        std::fprintf(
            stderr,
            "usage: metalrobo_numanx_human_io_probe /absolute/path/to/MetalRobo.metallib\n"
        );
        return 64;
    }
    return runProbe(argv[1]);
}
