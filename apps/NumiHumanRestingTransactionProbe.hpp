#pragma once

#include "NumiHumanRestingCoupling.hpp"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace numi::human::transaction_probe {

template <class T>
[[nodiscard]] inline bool sameBytes(
    const std::vector<T>& first,
    const std::vector<T>& second
) noexcept {
    return first.size() == second.size() &&
        (first.empty() || std::memcmp(
            first.data(), second.data(), first.size() * sizeof(T)) == 0);
}

template <class T>
[[nodiscard]] inline std::size_t firstDifferingByte(
    const std::vector<T>& first,
    const std::vector<T>& second
) noexcept {
    const auto* firstBytes = reinterpret_cast<const std::byte*>(first.data());
    const auto* secondBytes = reinterpret_cast<const std::byte*>(second.data());
    const std::size_t firstByteCount = first.size() * sizeof(T);
    const std::size_t secondByteCount = second.size() * sizeof(T);
    const std::size_t sharedSize = firstByteCount < secondByteCount
        ? firstByteCount : secondByteCount;
    for (std::size_t index = 0u; index < sharedSize; ++index) {
        if (firstBytes[index] != secondBytes[index]) return index;
    }
    return firstByteCount == secondByteCount ? std::string::npos : sharedSize;
}

[[nodiscard]] inline std::vector<std::byte> readBufferBytes(
    id<MTLBuffer> buffer,
    const char* label
) {
    need(buffer != nil && buffer.contents != nullptr,
         std::string("resting transaction probe missing ") + label);
    std::vector<std::byte> bytes(buffer.length);
    std::memcpy(bytes.data(), buffer.contents, bytes.size());
    return bytes;
}

inline void restoreBufferBytes(
    id<MTLBuffer> buffer,
    const std::vector<std::byte>& bytes,
    const char* label
) {
    need(buffer != nil && buffer.contents != nullptr &&
             buffer.length == bytes.size(),
         std::string("resting transaction probe cannot restore ") + label);
    std::memcpy(buffer.contents, bytes.data(), bytes.size());
}

struct CouplingMemoryImage {
    std::vector<std::byte> respirationAccepted;
    std::vector<std::byte> respirationCandidate;
    std::vector<std::byte> respirationExcitation;
    std::vector<std::byte> brainInput;
    std::vector<std::byte> brainAccepted;
    std::vector<std::byte> brainCandidate;
    std::vector<std::byte> brainOutput;
    std::vector<std::byte> bodyStatuses;
    std::vector<std::byte> presentationBodies;
    std::vector<std::byte> presentationRespiration;
    std::vector<std::byte> presentationCandidateBodies;
    std::vector<std::byte> presentationCandidateRespiration;
    std::vector<std::byte> presentationCommonCoordinates;
    std::vector<std::byte> presentationCandidateCommonCoordinates;
    std::vector<std::byte> acceptedCommonCoordinates;
    std::vector<std::byte> presentationFrameCommonCoordinates;
    bool commonAcceptedCoordinatesInitialized = false;
    std::uint32_t presentedStep = 0u;
};

[[nodiscard]] inline CouplingMemoryImage captureCouplingMemory(
    const NumiHumanRestingCoupling& coupling
) {
    CouplingMemoryImage image;
    image.respirationAccepted = readBufferBytes(
        coupling.physiology.respiration->accepted, "respiration accepted state");
    image.respirationCandidate = readBufferBytes(
        coupling.physiology.respiration->candidate, "respiration candidate state");
    image.respirationExcitation = readBufferBytes(
        coupling.physiology.respiration->excitation, "respiration excitation");
    image.brainInput = readBufferBytes(coupling.brain.input, "Brain input");
    image.brainAccepted = readBufferBytes(coupling.brain.accepted, "Brain accepted state");
    image.brainCandidate = readBufferBytes(coupling.brain.candidate, "Brain candidate state");
    image.brainOutput = readBufferBytes(coupling.brain.output, "Brain output");
    image.bodyStatuses = readBufferBytes(coupling.physiology.statuses, "Matter status mirror");
    image.presentationBodies = readBufferBytes(coupling.presentationBodies, "presented body state");
    image.presentationRespiration = readBufferBytes(
        coupling.presentationRespiration, "presented respiration state");
    image.presentationCandidateBodies = readBufferBytes(
        coupling.presentationCandidateBodies, "candidate presentation body state");
    image.presentationCandidateRespiration = readBufferBytes(
        coupling.presentationCandidateRespiration,
        "candidate presentation respiration state");
    image.presentationCommonCoordinates = readBufferBytes(
        coupling.presentationCommonCoordinates, "presented common coordinates");
    image.presentationCandidateCommonCoordinates = readBufferBytes(
        coupling.presentationCandidateCommonCoordinates,
        "candidate common coordinates");
    image.acceptedCommonCoordinates = readBufferBytes(
        coupling.acceptedCommonCoordinates, "accepted common coordinates");
    image.presentationFrameCommonCoordinates = readBufferBytes(
        coupling.presentationFrameCommonCoordinates, "captured frame common coordinates");
    image.commonAcceptedCoordinatesInitialized = coupling.commonAcceptedCoordinatesInitialized;
    image.presentedStep = coupling.presentedStep;
    return image;
}

inline void restoreCouplingMemory(
    NumiHumanRestingCoupling& coupling,
    const CouplingMemoryImage& image
) {
    restoreBufferBytes(coupling.physiology.respiration->accepted,
                       image.respirationAccepted, "respiration accepted state");
    restoreBufferBytes(coupling.physiology.respiration->candidate,
                       image.respirationCandidate, "respiration candidate state");
    restoreBufferBytes(coupling.physiology.respiration->excitation,
                       image.respirationExcitation, "respiration excitation");
    restoreBufferBytes(coupling.brain.input, image.brainInput, "Brain input");
    restoreBufferBytes(coupling.brain.accepted,
                       image.brainAccepted, "Brain accepted state");
    restoreBufferBytes(coupling.brain.candidate,
                       image.brainCandidate, "Brain candidate state");
    restoreBufferBytes(coupling.brain.output, image.brainOutput, "Brain output");
    restoreBufferBytes(coupling.physiology.statuses,
                       image.bodyStatuses, "Matter status mirror");
    restoreBufferBytes(coupling.presentationBodies,
                       image.presentationBodies, "presented body state");
    restoreBufferBytes(coupling.presentationRespiration,
                       image.presentationRespiration,
                       "presented respiration state");
    restoreBufferBytes(coupling.presentationCandidateBodies,
                       image.presentationCandidateBodies,
                       "candidate presentation body state");
    restoreBufferBytes(coupling.presentationCandidateRespiration,
                       image.presentationCandidateRespiration,
                       "candidate presentation respiration state");
    restoreBufferBytes(coupling.presentationCommonCoordinates,
                       image.presentationCommonCoordinates,
                       "presented common coordinates");
    restoreBufferBytes(coupling.presentationCandidateCommonCoordinates,
                       image.presentationCandidateCommonCoordinates,
                       "candidate common coordinates");
    restoreBufferBytes(coupling.acceptedCommonCoordinates,
                       image.acceptedCommonCoordinates, "accepted common coordinates");
    restoreBufferBytes(coupling.presentationFrameCommonCoordinates,
                       image.presentationFrameCommonCoordinates,
                       "captured frame common coordinates");
    coupling.commonAcceptedCoordinatesInitialized = image.commonAcceptedCoordinatesInitialized;
    coupling.presentedStep = image.presentedStep;
}

struct PhysicalSnapshot {
    __strong id<MTLBuffer> roots = nil;
    __strong id<MTLBuffer> q = nil;
    __strong id<MTLBuffer> v = nil;
    __strong id<MTLBuffer> muscles = nil;
    std::size_t rootCount = 0u;
    std::size_t qCount = 0u;
    std::size_t vCount = 0u;
    std::size_t muscleCount = 0u;
    std::vector<MRCompensatedRootTranslationGPU> rootValues;
    std::vector<float> qValues;
    std::vector<float> vValues;
    std::vector<MRMujocoMuscleStateGPU> muscleValues;
    std::uint64_t transaction = 0u;
    std::uint64_t generation = 0u;
    bool encoded = false;

    explicit PhysicalSnapshot(id<MTLDevice> device,
                              const metalrobo::EngineModel& model,
                              const metalrobo::MetalArticulatedOperatorInput& input)
        : rootCount(input.environmentCount),
          qCount(static_cast<std::size_t>(input.environmentCount) *
                 model.articulations[input.articulationIndex].nq),
          vCount(static_cast<std::size_t>(input.environmentCount) *
                 model.articulations[input.articulationIndex].nv),
          muscleCount(input.mujoco.muscles.size()) {
        roots = [device newBufferWithLength:rootCount * sizeof(MRCompensatedRootTranslationGPU)
                                    options:MTLResourceStorageModeShared];
        q = [device newBufferWithLength:qCount * sizeof(float)
                                options:MTLResourceStorageModeShared];
        v = [device newBufferWithLength:vCount * sizeof(float)
                                options:MTLResourceStorageModeShared];
        muscles = [device newBufferWithLength:muscleCount * sizeof(MRMujocoMuscleStateGPU)
                                      options:MTLResourceStorageModeShared];
        need(roots != nil && q != nil && v != nil && muscles != nil,
             "resting transaction physical snapshot allocation failed");
    }
};

[[nodiscard]] inline bool encodePhysicalSnapshot(
    void* context,
    const metalrobo::MetalArticulatedOperatorPhysicalStateObserverPass& pass
) noexcept {
    auto* snapshot = static_cast<PhysicalSnapshot*>(context);
    if (snapshot == nullptr || pass.abiVersion !=
            metalrobo::kMetalArticulatedOperatorPhysicalStateObserverABIVersion ||
        pass.environmentCount != 1u || pass.transactionFingerprint != snapshot->transaction ||
        pass.physicsGeneration != snapshot->generation ||
        pass.rootTranslationElementCount != snapshot->rootCount ||
        pass.qElementCount != snapshot->qCount || pass.vElementCount != snapshot->vCount ||
        pass.mujocoStateElementCount != snapshot->muscleCount ||
        pass.rootTranslations == nullptr || pass.q == nullptr || pass.v == nullptr ||
        pass.mujocoStates == nullptr || pass.commandBuffer == nullptr) {
        return false;
    }
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return false;
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)pass.rootTranslations sourceOffset:0u
                 toBuffer:snapshot->roots destinationOffset:0u
                     size:snapshot->rootCount * sizeof(MRCompensatedRootTranslationGPU)];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)pass.q sourceOffset:0u
                 toBuffer:snapshot->q destinationOffset:0u
                     size:snapshot->qCount * sizeof(float)];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)pass.v sourceOffset:0u
                 toBuffer:snapshot->v destinationOffset:0u
                     size:snapshot->vCount * sizeof(float)];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)pass.mujocoStates sourceOffset:0u
                 toBuffer:snapshot->muscles destinationOffset:0u
                     size:snapshot->muscleCount * sizeof(MRMujocoMuscleStateGPU)];
    [blit endEncoding];
    snapshot->encoded = true;
    return true;
}

[[nodiscard]] inline PhysicalSnapshot observePhysicalState(
    metalrobo::MetalArticulatedOperatorContext& context,
    PhysicalSnapshot& snapshot,
    const std::uint64_t transaction,
    const std::uint64_t generation
) {
    snapshot.transaction = transaction;
    snapshot.generation = generation;
    snapshot.encoded = false;
    std::string error;
    need(context.flushPhysicalStateObserver(
             &snapshot, &encodePhysicalSnapshot, error),
         "resting transaction physical-state observer failed: " + error);
    need(snapshot.encoded, "resting transaction observer was not encoded");
    snapshot.rootValues.resize(snapshot.rootCount);
    snapshot.qValues.resize(snapshot.qCount);
    snapshot.vValues.resize(snapshot.vCount);
    snapshot.muscleValues.resize(snapshot.muscleCount);
    std::memcpy(snapshot.rootValues.data(), snapshot.roots.contents,
                snapshot.rootValues.size() * sizeof(snapshot.rootValues.front()));
    std::memcpy(snapshot.qValues.data(), snapshot.q.contents,
                snapshot.qValues.size() * sizeof(float));
    std::memcpy(snapshot.vValues.data(), snapshot.v.contents,
                snapshot.vValues.size() * sizeof(float));
    std::memcpy(snapshot.muscleValues.data(), snapshot.muscles.contents,
                snapshot.muscleValues.size() * sizeof(snapshot.muscleValues.front()));
    return snapshot;
}

inline void readPhysicalSnapshotBuffers(PhysicalSnapshot& snapshot) {
    snapshot.rootValues.resize(snapshot.rootCount);
    snapshot.qValues.resize(snapshot.qCount);
    snapshot.vValues.resize(snapshot.vCount);
    snapshot.muscleValues.resize(snapshot.muscleCount);
    std::memcpy(snapshot.rootValues.data(), snapshot.roots.contents,
                snapshot.rootValues.size() * sizeof(snapshot.rootValues.front()));
    std::memcpy(snapshot.qValues.data(), snapshot.q.contents,
                snapshot.qValues.size() * sizeof(float));
    std::memcpy(snapshot.vValues.data(), snapshot.v.contents,
                snapshot.vValues.size() * sizeof(float));
    std::memcpy(snapshot.muscleValues.data(), snapshot.muscles.contents,
                snapshot.muscleValues.size() * sizeof(snapshot.muscleValues.front()));
}

[[nodiscard]] inline bool samePhysicalState(
    const PhysicalSnapshot& first,
    const PhysicalSnapshot& second
) noexcept {
    return sameBytes(first.rootValues, second.rootValues) &&
        sameBytes(first.qValues, second.qValues) &&
        sameBytes(first.vValues, second.vValues) &&
        sameBytes(first.muscleValues, second.muscleValues);
}

[[nodiscard]] inline bool sameMatterCirculation(
    const RuntimeStateSnapshot& first,
    const RuntimeStateSnapshot& second
) noexcept {
    return first.available && second.available &&
        first.controlStep == second.controlStep &&
        first.physicsSubstep == second.physicsSubstep &&
        sameBytes(first.vascularState, second.vascularState) &&
        sameBytes(first.vascularClock, second.vascularClock);
}

// Compare continuation-authoritative Matter state, omitting completion status
// and solver certificates that may describe the rejected attempt itself.
[[nodiscard]] inline bool sameMatterAcceptedState(
    const RuntimeStateSnapshot& first,
    const RuntimeStateSnapshot& second
) noexcept {
    return first.available && second.available &&
        first.sourcePhysicsFingerprint == second.sourcePhysicsFingerprint &&
        first.deviceProgramFingerprint == second.deviceProgramFingerprint &&
        first.controlStep == second.controlStep &&
        first.physicsSubstep == second.physicsSubstep &&
        first.identificationGeneration == second.identificationGeneration &&
        first.identificationCheckpoint == second.identificationCheckpoint &&
        first.identificationAdvanced == second.identificationAdvanced &&
        sameBytes(first.sutureProxyEdges, second.sutureProxyEdges) &&
        first.sutureProxyBindingRevision == second.sutureProxyBindingRevision &&
        first.coupledTimestepMultiplier == second.coupledTimestepMultiplier &&
        first.coupledTimestepDivisor == second.coupledTimestepDivisor &&
        first.fgmresIterationBudgetOverride == second.fgmresIterationBudgetOverride &&
        first.newtonIterationBudgetOverride == second.newtonIterationBudgetOverride &&
        sameBytes(first.particles, second.particles) &&
        sameBytes(first.femNodes, second.femNodes) &&
        sameBytes(first.femFields, second.femFields) &&
        sameBytes(first.vascularState, second.vascularState) &&
        sameBytes(first.vascularClock, second.vascularClock) &&
        sameBytes(first.femTopologyNodes, second.femTopologyNodes) &&
        sameBytes(first.femTopologyTetrahedra, second.femTopologyTetrahedra) &&
        sameBytes(first.cohesiveFaces, second.cohesiveFaces) &&
        sameBytes(first.punctureChannels, second.punctureChannels) &&
        sameBytes(first.topologyStates, second.topologyStates) &&
        first.allocationGeneration == second.allocationGeneration &&
        sameBytes(first.mpmActiveNodeIndices, second.mpmActiveNodeIndices) &&
        sameBytes(first.mpmNodeToActive, second.mpmNodeToActive) &&
        sameBytes(first.mpmActiveNodeCounts, second.mpmActiveNodeCounts) &&
        sameBytes(first.rigidGeneralizedCandidate, second.rigidGeneralizedCandidate) &&
        sameBytes(first.learnedWeights, second.learnedWeights) &&
        first.learnedWeightRevision == second.learnedWeightRevision &&
        sameBytes(first.adaptive, second.adaptive) &&
        sameBytes(first.schedulers, second.schedulers) &&
        sameBytes(first.reactions, second.reactions) &&
        sameBytes(first.rigidStates, second.rigidStates) &&
        sameBytes(first.contactHistories, second.contactHistories) &&
        sameBytes(first.humanSupportHistories, second.humanSupportHistories) &&
        sameBytes(first.humanSupportConsequences, second.humanSupportConsequences) &&
        sameBytes(first.deformableContactHistories, second.deformableContactHistories) &&
        first.materialStateStride == second.materialStateStride &&
        sameBytes(first.particleMaterialState, second.particleMaterialState) &&
        sameBytes(first.femMaterialState, second.femMaterialState) &&
        sameBytes(first.identification, second.identification) &&
        sameBytes(first.environmentParameters, second.environmentParameters);
}

[[nodiscard]] inline bool sameAcceptedCouplingMemory(
    const CouplingMemoryImage& first,
    const CouplingMemoryImage& second
) noexcept {
    return sameBytes(first.respirationAccepted, second.respirationAccepted) &&
        sameBytes(first.brainAccepted, second.brainAccepted) &&
        sameBytes(first.presentationBodies, second.presentationBodies) &&
        sameBytes(first.presentationRespiration, second.presentationRespiration) &&
        sameBytes(first.presentationCommonCoordinates,
                  second.presentationCommonCoordinates) &&
        sameBytes(first.acceptedCommonCoordinates,
                  second.acceptedCommonCoordinates) &&
        sameBytes(first.presentationFrameCommonCoordinates,
                  second.presentationFrameCommonCoordinates) &&
        first.commonAcceptedCoordinatesInitialized ==
            second.commonAcceptedCoordinatesInitialized &&
        first.presentedStep == second.presentedStep;
}

inline void requireAcceptedStep(
    const metalrobo::MetalArticulatedOperatorDiagnostics& diagnostics,
    const metalrobo::MetalArticulatedOperatorResult& result,
    const std::uint32_t endStep,
    const std::size_t qCount,
    const std::size_t vCount,
    const std::size_t muscleCount
) {
    need(diagnostics.succeeded() && diagnostics.dispatched && diagnostics.published &&
             diagnostics.completedStandSteps == endStep &&
             diagnostics.residentStateTransactionFingerprint != 0u &&
             diagnostics.residentStateGeneration != 0u &&
             result.standStatuses.size() == 1u &&
             result.standStatuses.front().code == MR_NUMI_HUMAN_STAND_SUCCESS &&
             result.standQ.size() == qCount && result.standV.size() == vCount &&
             result.standRootTranslations.size() == 1u &&
             result.mujocoActivationStates.size() == muscleCount,
         "resting transaction probe could not publish an accepted full body step: " +
             diagnostics.message);
}

inline void run(
    const metalrobo::EngineModel& model,
    const metalrobo::MetalArticulatedOperatorConfig& config,
    metalrobo::MetalArticulatedOperatorInput seedInput,
    NumiHumanRestingCoupling& coupling
) {
    auto& physiology = coupling.physiology;
    auto& respiration = *physiology.respiration;
    const CouplingMemoryImage originalMemory = captureCouplingMemory(coupling);
    const RuntimeStateSnapshot originalMatter = physiology.runtime.snapshot();
    need(originalMatter.available, "resting rejection probe lacks an initial Matter snapshot");
    const auto originalDispatch = respiration.dispatch;
    const auto originalDiagnosticReject = respiration.diagnosticRejectAtControlStep;
    __strong id<MTLBuffer> originalProbeRoots = coupling.transactionProbeRoots;
    __strong id<MTLBuffer> originalProbeQ = coupling.transactionProbeQ;
    __strong id<MTLBuffer> originalProbeV = coupling.transactionProbeV;
    __strong id<MTLBuffer> originalProbeMuscles = coupling.transactionProbeMuscles;
    const auto originalProbeCaptureStep = coupling.transactionProbeCaptureControlStep;
    const bool originalProbeCaptured = coupling.transactionProbeCaptured;
    const auto originalProbeCapturedStep = coupling.transactionProbeCapturedStep;
    respiration.dispatch.reject = 0u;

    auto restoreInitialCoupling = [&]() {
        const auto restored = physiology.runtime.restore(originalMatter);
        need(restored.encoded, "resting rejection probe could not restore Matter seed: " +
             restored.message);
        restoreCouplingMemory(coupling, originalMemory);
        respiration.dispatch = originalDispatch;
        respiration.diagnosticRejectAtControlStep = originalDiagnosticReject;
        coupling.transactionProbeRoots = originalProbeRoots;
        coupling.transactionProbeQ = originalProbeQ;
        coupling.transactionProbeV = originalProbeV;
        coupling.transactionProbeMuscles = originalProbeMuscles;
        coupling.transactionProbeCaptureControlStep = originalProbeCaptureStep;
        coupling.transactionProbeCaptured = originalProbeCaptured;
        coupling.transactionProbeCapturedStep = originalProbeCapturedStep;
    };

    auto program = coupling.program();
    auto makeInput = [&](bool resident, std::uint64_t transaction,
                         std::uint64_t generation, bool fullCollection) {
        auto input = seedInput;
        input.stand.numanXTransactionProgram = program;
        // The probe performs extra accepted/rejected body submissions before
        // the requested horizon. Do not reuse the horizon's borrowed tendon
        // publication consumer: its snapshots are the output audit for the
        // real run and must only reflect that run's accepted submissions.
        input.stand.tendonLoadProgram = {};
        input.stand.authoritativeStepCount = 2u;
        input.stand.stepCount = 1u;
        input.publishAcceptedResidentState = true;
        input.collectFullResultToHost = fullCollection;
        input.residentContinuation = resident
            ? metalrobo::MetalArticulatedOperatorResidentStateContinuation{
                  .previousTransactionFingerprint = transaction,
                  .previousPhysicsGeneration = generation,
              }
            : metalrobo::MetalArticulatedOperatorResidentStateContinuation{};
        input.stand.stepIndexOffset = resident ? 1u : 0u;
        if (resident) {
            input.q = {};
            input.v = {};
            input.rootTranslations = {};
            input.stand.v = {};
            input.mujoco.states = {};
        }
        return input;
    };
    auto makeSegmentInput = [&](bool resident, std::uint64_t transaction,
                                std::uint64_t generation, std::uint32_t offset,
                                std::uint32_t count, bool fullCollection) {
        auto input = makeInput(resident, transaction, generation, fullCollection);
        input.stand.authoritativeStepCount = 6u;
        input.stand.stepIndexOffset = offset;
        input.stand.stepCount = count;
        return input;
    };
    const std::size_t qCount = model.articulations[seedInput.articulationIndex].nq;
    const std::size_t vCount = model.articulations[seedInput.articulationIndex].nv;
    const std::size_t muscleCount = seedInput.mujoco.muscles.size();
    PhysicalSnapshot physical(coupling.physiology.device, model, seedInput);

    metalrobo::MetalArticulatedOperatorContext baselineContext(config);
    auto baselineFirstInput = makeInput(false, 0u, 0u, true);
    metalrobo::MetalArticulatedOperatorResult baselineFirst;
    const auto baselineFirstDiagnostics = baselineContext.run(
        model, baselineFirstInput, baselineFirst);
    requireAcceptedStep(baselineFirstDiagnostics, baselineFirst, 1u,
                        qCount, vCount, muscleCount);
    const auto baselineFirstMatter = physiology.runtime.snapshot();
    const auto baselineFirstMemory = captureCouplingMemory(coupling);
    const PhysicalSnapshot baselineFirstPhysical = observePhysicalState(
        baselineContext, physical,
        baselineFirstDiagnostics.residentStateTransactionFingerprint,
        baselineFirstDiagnostics.residentStateGeneration);
    auto baselineSecondInput = makeInput(
        true, baselineFirstDiagnostics.residentStateTransactionFingerprint,
        baselineFirstDiagnostics.residentStateGeneration, true);
    metalrobo::MetalArticulatedOperatorResult baselineSecond;
    const auto baselineSecondDiagnostics = baselineContext.run(
        model, baselineSecondInput, baselineSecond);
    requireAcceptedStep(baselineSecondDiagnostics, baselineSecond, 2u,
                        qCount, vCount, muscleCount);
    const auto baselineFinalMatter = physiology.runtime.snapshot();
    const auto baselineFinalMemory = captureCouplingMemory(coupling);
    const PhysicalSnapshot baselineFinalPhysical = observePhysicalState(
        baselineContext, physical,
        baselineSecondDiagnostics.residentStateTransactionFingerprint,
        baselineSecondDiagnostics.residentStateGeneration);

    restoreInitialCoupling();
    metalrobo::MetalArticulatedOperatorContext trialContext(config);
    auto trialFirstInput = makeInput(false, 0u, 0u, true);
    metalrobo::MetalArticulatedOperatorResult trialFirst;
    const auto trialFirstDiagnostics = trialContext.run(
        model, trialFirstInput, trialFirst);
    requireAcceptedStep(trialFirstDiagnostics, trialFirst, 1u,
                        qCount, vCount, muscleCount);
    const auto trialFirstMatter = physiology.runtime.snapshot();
    const auto trialFirstMemory = captureCouplingMemory(coupling);
    const PhysicalSnapshot trialFirstPhysical = observePhysicalState(
        trialContext, physical,
        trialFirstDiagnostics.residentStateTransactionFingerprint,
        trialFirstDiagnostics.residentStateGeneration);
    need(samePhysicalState(baselineFirstPhysical, trialFirstPhysical) &&
             sameMatterCirculation(baselineFirstMatter, trialFirstMatter) &&
             sameBytes(baselineFirstMemory.respirationAccepted,
                       trialFirstMemory.respirationAccepted) &&
             sameBytes(baselineFirstMemory.brainAccepted,
                       trialFirstMemory.brainAccepted) &&
             sameBytes(baselineFirstMemory.presentationCommonCoordinates,
                       trialFirstMemory.presentationCommonCoordinates) &&
             sameBytes(baselineFirstMemory.acceptedCommonCoordinates,
                       trialFirstMemory.acceptedCommonCoordinates) &&
             sameBytes(baselineFirstMemory.presentationFrameCommonCoordinates,
                       trialFirstMemory.presentationFrameCommonCoordinates) &&
             baselineFirstMemory.commonAcceptedCoordinatesInitialized ==
                 trialFirstMemory.commonAcceptedCoordinatesInitialized,
         "resting transaction predecessor did not replay from the same accepted seed");

    if (respiration.commonGeometryGateEnabled) {
        need(respiration.commonGeometryBoxes != nil &&
                 respiration.commonCandidateCoordinates != nil &&
                 respiration.commonGeometryBoxes.contents != nullptr &&
                 respiration.commonGeometryBoxes.length %
                     sizeof(MRHumanRestingCommonCoordinateBoxGPU) == 0u,
             "common-field domain rejection probe lacks certified-box buffers");
        const auto savedBoxes = readBufferBytes(
            respiration.commonGeometryBoxes, "common certified coordinate boxes");
        const std::size_t boxCount = savedBoxes.size() /
            sizeof(MRHumanRestingCommonCoordinateBoxGPU);
        need(boxCount > 0u, "common-field domain rejection probe has no boxes");
        std::vector<MRHumanRestingCommonCoordinateBoxGPU> impossibleBoxes(boxCount);
        for (auto& box : impossibleBoxes) {
            box.lower[0] = {0.95f, 0.95f, 0.95f, 0.95f};
            box.lower[1] = {0.95f, 0.0f, 0.0f, 0.0f};
            box.upper[0] = {0.951f, 0.951f, 0.951f, 0.951f};
            box.upper[1] = {0.951f, 0.0f, 0.0f, 0.0f};
        }
        std::memcpy(respiration.commonGeometryBoxes.contents, impossibleBoxes.data(),
                    impossibleBoxes.size() * sizeof(impossibleBoxes.front()));
        auto domainRejectedInput = makeInput(
            true, trialFirstDiagnostics.residentStateTransactionFingerprint,
            trialFirstDiagnostics.residentStateGeneration, false);
        metalrobo::MetalArticulatedOperatorResult domainRejectedResult;
        domainRejectedResult.standQ = {-902.0f};
        const auto domainRejectedDiagnostics = trialContext.run(
            model, domainRejectedInput, domainRejectedResult);
        const auto afterDomainMatter = physiology.runtime.snapshot();
        const auto afterDomainMemory = captureCouplingMemory(coupling);
        MRHumanRestingCommonCoordinatesGPU rejectedCoordinates{};
        need(afterDomainMemory.presentationCandidateCommonCoordinates.size() ==
                 sizeof(rejectedCoordinates),
             "common-field rejection candidate ABI size changed");
        std::memcpy(&rejectedCoordinates,
                    afterDomainMemory.presentationCandidateCommonCoordinates.data(),
                    sizeof(rejectedCoordinates));
        const PhysicalSnapshot afterDomainPhysical = observePhysicalState(
            trialContext, physical,
            trialFirstDiagnostics.residentStateTransactionFingerprint,
            trialFirstDiagnostics.residentStateGeneration);
        const bool domainRejected = !domainRejectedDiagnostics.succeeded() &&
            domainRejectedDiagnostics.dispatched &&
            !domainRejectedDiagnostics.published &&
            domainRejectedDiagnostics.firstStandGPUStatusCode !=
                MR_NUMI_HUMAN_STAND_SUCCESS &&
            domainRejectedResult.standQ == std::vector<float>{-902.0f} &&
            rejectedCoordinates.status.x ==
                MR_HUMAN_RESTING_COMMON_STATUS_OUTSIDE_CERTIFIED_DOMAIN;
        const bool acceptedStateUnchanged =
            samePhysicalState(trialFirstPhysical, afterDomainPhysical) &&
            sameMatterCirculation(trialFirstMatter, afterDomainMatter) &&
            sameBytes(trialFirstMemory.respirationAccepted,
                      afterDomainMemory.respirationAccepted) &&
            sameBytes(trialFirstMemory.brainAccepted,
                      afterDomainMemory.brainAccepted) &&
            sameBytes(trialFirstMemory.presentationBodies,
                      afterDomainMemory.presentationBodies) &&
            sameBytes(trialFirstMemory.presentationRespiration,
                      afterDomainMemory.presentationRespiration) &&
            sameBytes(trialFirstMemory.presentationCommonCoordinates,
                      afterDomainMemory.presentationCommonCoordinates) &&
            sameBytes(trialFirstMemory.acceptedCommonCoordinates,
                      afterDomainMemory.acceptedCommonCoordinates) &&
            sameBytes(trialFirstMemory.presentationFrameCommonCoordinates,
                      afterDomainMemory.presentationFrameCommonCoordinates) &&
            trialFirstMemory.commonAcceptedCoordinatesInitialized ==
                afterDomainMemory.commonAcceptedCoordinatesInitialized &&
            trialFirstMemory.presentedStep == afterDomainMemory.presentedStep;
        if (!acceptedStateUnchanged) {
            std::cerr << "resting_common_domain_rollback_diagnostic"
                      << " physical=" << samePhysicalState(trialFirstPhysical, afterDomainPhysical)
                      << " matter=" << sameMatterCirculation(trialFirstMatter, afterDomainMatter)
                      << " respiration=" << sameBytes(trialFirstMemory.respirationAccepted, afterDomainMemory.respirationAccepted)
                      << " brain=" << sameBytes(trialFirstMemory.brainAccepted, afterDomainMemory.brainAccepted)
                      << " presented_bodies=" << sameBytes(trialFirstMemory.presentationBodies, afterDomainMemory.presentationBodies)
                      << " presented_respiration=" << sameBytes(trialFirstMemory.presentationRespiration, afterDomainMemory.presentationRespiration)
                      << " presented_common=" << sameBytes(trialFirstMemory.presentationCommonCoordinates, afterDomainMemory.presentationCommonCoordinates)
                      << " accepted_common=" << sameBytes(trialFirstMemory.acceptedCommonCoordinates, afterDomainMemory.acceptedCommonCoordinates)
                      << " captured_common=" << sameBytes(trialFirstMemory.presentationFrameCommonCoordinates, afterDomainMemory.presentationFrameCommonCoordinates)
                      << " common_initialized=" << (trialFirstMemory.commonAcceptedCoordinatesInitialized == afterDomainMemory.commonAcceptedCoordinatesInitialized)
                      << " cursor=" << (trialFirstMemory.presentedStep == afterDomainMemory.presentedStep)
                      << " root_diff=" << firstDifferingByte(trialFirstPhysical.rootValues, afterDomainPhysical.rootValues)
                      << " q_diff=" << firstDifferingByte(trialFirstPhysical.qValues, afterDomainPhysical.qValues)
                      << " v_diff=" << firstDifferingByte(trialFirstPhysical.vValues, afterDomainPhysical.vValues)
                      << " muscle_diff=" << firstDifferingByte(trialFirstPhysical.muscleValues, afterDomainPhysical.muscleValues)
                      << " matter_state_diff=" << firstDifferingByte(trialFirstMatter.vascularState, afterDomainMatter.vascularState)
                      << " matter_clock_diff=" << firstDifferingByte(trialFirstMatter.vascularClock, afterDomainMatter.vascularClock)
                      << " respiration_diff=" << firstDifferingByte(trialFirstMemory.respirationAccepted, afterDomainMemory.respirationAccepted)
                      << " brain_diff=" << firstDifferingByte(trialFirstMemory.brainAccepted, afterDomainMemory.brainAccepted)
                      << " presentation_body_diff=" << firstDifferingByte(trialFirstMemory.presentationBodies, afterDomainMemory.presentationBodies)
                      << " presentation_resp_diff=" << firstDifferingByte(trialFirstMemory.presentationRespiration, afterDomainMemory.presentationRespiration)
                      << " presentation_common_diff=" << firstDifferingByte(trialFirstMemory.presentationCommonCoordinates, afterDomainMemory.presentationCommonCoordinates)
                      << " cursor_before=" << trialFirstMemory.presentedStep
                      << " cursor_after=" << afterDomainMemory.presentedStep << std::endl;
        }
        std::memcpy(respiration.commonGeometryBoxes.contents, savedBoxes.data(),
                    savedBoxes.size());
        restoreBufferBytes(coupling.presentationCandidateCommonCoordinates,
                           trialFirstMemory.presentationCandidateCommonCoordinates,
                           "common coordinates after domain rejection probe");
        need(domainRejected,
             "out-of-certified-domain common coordinates did not reject the actual body transaction");
        need(acceptedStateUnchanged,
             "common-coordinate domain rejection advanced body, circulation, controller history, or presentation");
        std::cout << "resting_common_domain_rejection=pass status="
                  << rejectedCoordinates.status.x
                  << " accepted_physical_and_circulation_unchanged=true"
                     " controller_history_and_common_presentation_unchanged=true"
                     " retry_predecessor_generation="
                  << trialFirstDiagnostics.residentStateGeneration << '\n';
    }

    auto rejectedInput = makeInput(
        true, trialFirstDiagnostics.residentStateTransactionFingerprint,
        trialFirstDiagnostics.residentStateGeneration, false);
    respiration.dispatch.reject = 1u;
    metalrobo::MetalArticulatedOperatorResult rejectedResult;
    rejectedResult.standQ = {-901.0f};
    const auto rejectedDiagnostics = trialContext.run(
        model, rejectedInput, rejectedResult);
    need(!rejectedDiagnostics.succeeded() && rejectedDiagnostics.dispatched &&
             !rejectedDiagnostics.published &&
             rejectedDiagnostics.firstStandGPUStatusCode ==
                 MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED &&
             rejectedResult.standQ == std::vector<float>{-901.0f},
         "forced respiratory dispatch rejection did not fail the actual body transaction: " +
             rejectedDiagnostics.message +
             " succeeded=" + std::to_string(rejectedDiagnostics.succeeded()) +
             " dispatched=" + std::to_string(rejectedDiagnostics.dispatched) +
             " published=" + std::to_string(rejectedDiagnostics.published) +
             " stand_status=" + std::to_string(rejectedDiagnostics.firstStandGPUStatusCode) +
             " expected_stand_status=" + std::to_string(MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED) +
             " result_q_count=" + std::to_string(rejectedResult.standQ.size()));
    const auto afterRejectedMatter = physiology.runtime.snapshot();
    const auto afterRejectedMemory = captureCouplingMemory(coupling);
    NMHumanRespirationState rejectedRespirationCandidate{};
    need(afterRejectedMemory.respirationCandidate.size() ==
             sizeof(rejectedRespirationCandidate),
         "resting rejection probe candidate respiration ABI size changed");
    std::memcpy(&rejectedRespirationCandidate,
                afterRejectedMemory.respirationCandidate.data(),
                sizeof(rejectedRespirationCandidate));
    need(rejectedRespirationCandidate.status.w != 0u &&
             !afterRejectedMatter.statuses.empty() &&
             afterRejectedMatter.statuses.front().code != NM_STATUS_SUCCESS,
         "respiratory reject flag did not fail the coupled Matter candidate");
    const PhysicalSnapshot afterRejectedPhysical = observePhysicalState(
        trialContext, physical,
        trialFirstDiagnostics.residentStateTransactionFingerprint,
        trialFirstDiagnostics.residentStateGeneration);
    const bool unchangedRoots = sameBytes(
        trialFirstPhysical.rootValues, afterRejectedPhysical.rootValues);
    const bool unchangedQ = sameBytes(
        trialFirstPhysical.qValues, afterRejectedPhysical.qValues);
    const bool unchangedV = sameBytes(
        trialFirstPhysical.vValues, afterRejectedPhysical.vValues);
    const bool unchangedMuscles = sameBytes(
        trialFirstPhysical.muscleValues, afterRejectedPhysical.muscleValues);
    const bool unchangedMatterState = sameBytes(
        trialFirstMatter.vascularState, afterRejectedMatter.vascularState);
    const bool unchangedMatterClock = sameBytes(
        trialFirstMatter.vascularClock, afterRejectedMatter.vascularClock);
    const bool unchangedMatterMetadata = trialFirstMatter.available &&
        afterRejectedMatter.available &&
        trialFirstMatter.controlStep == afterRejectedMatter.controlStep &&
        trialFirstMatter.physicsSubstep == afterRejectedMatter.physicsSubstep;
    const bool unchangedRespiration = sameBytes(
        trialFirstMemory.respirationAccepted,
        afterRejectedMemory.respirationAccepted);
    const bool unchangedBrain = sameBytes(
        trialFirstMemory.brainAccepted, afterRejectedMemory.brainAccepted);
    const bool unchangedPresentedBodies = sameBytes(
        trialFirstMemory.presentationBodies,
        afterRejectedMemory.presentationBodies);
    const bool unchangedPresentedRespiration = sameBytes(
        trialFirstMemory.presentationRespiration,
        afterRejectedMemory.presentationRespiration);
    const bool unchangedPresentedCommonCoordinates = sameBytes(
        trialFirstMemory.presentationCommonCoordinates,
        afterRejectedMemory.presentationCommonCoordinates);
    const bool unchangedAcceptedCommonCoordinates = sameBytes(
        trialFirstMemory.acceptedCommonCoordinates,
        afterRejectedMemory.acceptedCommonCoordinates);
    const bool unchangedCapturedCommonCoordinates = sameBytes(
        trialFirstMemory.presentationFrameCommonCoordinates,
        afterRejectedMemory.presentationFrameCommonCoordinates);
    const bool unchangedCommonInitialization =
        trialFirstMemory.commonAcceptedCoordinatesInitialized ==
        afterRejectedMemory.commonAcceptedCoordinatesInitialized;
    const bool unchangedPresentationCursor =
        trialFirstMemory.presentedStep == afterRejectedMemory.presentedStep;
    const bool rejectedStateUnchanged = unchangedRoots && unchangedQ &&
        unchangedV && unchangedMuscles && unchangedMatterState &&
        unchangedMatterClock && unchangedMatterMetadata && unchangedRespiration &&
        unchangedBrain && unchangedPresentedBodies && unchangedPresentedRespiration &&
        unchangedPresentedCommonCoordinates && unchangedAcceptedCommonCoordinates &&
        unchangedCapturedCommonCoordinates && unchangedCommonInitialization &&
        unchangedPresentationCursor;
    if (!rejectedStateUnchanged) {
        std::cerr << "resting_rejection_rollback_diagnostic"
                  << " roots=" << unchangedRoots
                  << " q=" << unchangedQ
                  << " v=" << unchangedV
                  << " muscles=" << unchangedMuscles
                  << " matter_state=" << unchangedMatterState
                  << " matter_clock=" << unchangedMatterClock
                  << " matter_metadata=" << unchangedMatterMetadata
                  << " respiration=" << unchangedRespiration
                  << " brain=" << unchangedBrain
                  << " presented_bodies=" << unchangedPresentedBodies
                  << " presented_respiration=" << unchangedPresentedRespiration
                  << " accepted_common=" << unchangedAcceptedCommonCoordinates
                  << " captured_common=" << unchangedCapturedCommonCoordinates
                  << " common_initialized=" << unchangedCommonInitialization
                  << " control_step=" << trialFirstMatter.controlStep << "/"
                  << afterRejectedMatter.controlStep
                  << " physics_substep=" << trialFirstMatter.physicsSubstep << "/"
                  << afterRejectedMatter.physicsSubstep
                  << " first_byte_roots=" << firstDifferingByte(
                         trialFirstPhysical.rootValues, afterRejectedPhysical.rootValues)
                  << " first_byte_q=" << firstDifferingByte(
                         trialFirstPhysical.qValues, afterRejectedPhysical.qValues)
                  << " first_byte_v=" << firstDifferingByte(
                         trialFirstPhysical.vValues, afterRejectedPhysical.vValues)
                  << " first_byte_muscles=" << firstDifferingByte(
                         trialFirstPhysical.muscleValues, afterRejectedPhysical.muscleValues)
                  << " first_byte_matter_state=" << firstDifferingByte(
                         trialFirstMatter.vascularState, afterRejectedMatter.vascularState)
                  << " first_byte_matter_clock=" << firstDifferingByte(
                         trialFirstMatter.vascularClock, afterRejectedMatter.vascularClock)
                  << " first_byte_respiration=" << firstDifferingByte(
                         trialFirstMemory.respirationAccepted,
                         afterRejectedMemory.respirationAccepted)
                  << " first_byte_brain=" << firstDifferingByte(
                         trialFirstMemory.brainAccepted,
                         afterRejectedMemory.brainAccepted)
                  << " first_byte_presented_bodies=" << firstDifferingByte(
                         trialFirstMemory.presentationBodies,
                         afterRejectedMemory.presentationBodies)
                  << " first_byte_presented_respiration=" << firstDifferingByte(
                         trialFirstMemory.presentationRespiration,
                         afterRejectedMemory.presentationRespiration)
                  << " first_byte_accepted_common=" << firstDifferingByte(
                         trialFirstMemory.acceptedCommonCoordinates,
                         afterRejectedMemory.acceptedCommonCoordinates)
                  << " first_byte_captured_common=" << firstDifferingByte(
                         trialFirstMemory.presentationFrameCommonCoordinates,
                         afterRejectedMemory.presentationFrameCommonCoordinates)
                  << '\n';
    }
    need(rejectedStateUnchanged,
         "rejected respiratory dispatch advanced body, circulation, controller history, or presentation");

    respiration.dispatch.reject = 0u;
    metalrobo::MetalArticulatedOperatorResult retryResult;
    rejectedInput.collectFullResultToHost = true;
    const auto retryDiagnostics = trialContext.run(
        model, rejectedInput, retryResult);
    requireAcceptedStep(retryDiagnostics, retryResult, 2u,
                        qCount, vCount, muscleCount);
    need(retryDiagnostics.residentStateGeneration == 2u,
         "resting retry did not consume the unchanged accepted predecessor generation");
    const auto retryMatter = physiology.runtime.snapshot();
    const auto retryMemory = captureCouplingMemory(coupling);
    const PhysicalSnapshot retryPhysical = observePhysicalState(
        trialContext, physical,
        retryDiagnostics.residentStateTransactionFingerprint,
        retryDiagnostics.residentStateGeneration);
    need(samePhysicalState(baselineFinalPhysical, retryPhysical) &&
             sameMatterCirculation(baselineFinalMatter, retryMatter) &&
             sameBytes(baselineFinalMemory.respirationAccepted,
                       retryMemory.respirationAccepted) &&
             sameBytes(baselineFinalMemory.brainAccepted,
                       retryMemory.brainAccepted) &&
             sameBytes(baselineFinalMemory.presentationBodies,
                       retryMemory.presentationBodies) &&
             sameBytes(baselineFinalMemory.presentationRespiration,
                       retryMemory.presentationRespiration) &&
             sameBytes(baselineFinalMemory.presentationCommonCoordinates,
                       retryMemory.presentationCommonCoordinates) &&
             sameBytes(baselineFinalMemory.acceptedCommonCoordinates,
                       retryMemory.acceptedCommonCoordinates) &&
             sameBytes(baselineFinalMemory.presentationFrameCommonCoordinates,
                       retryMemory.presentationFrameCommonCoordinates),
         "accepted respiratory retry did not replay the uninterrupted coupled baseline");

    restoreInitialCoupling();
    physiology.batch(0u, 2u);
    const RuntimeStateSnapshot cursorBaselineMatter =
        physiology.runtime.snapshot();
    const auto cursorBaselineRespiration = readBufferBytes(
        respiration.accepted, "multi-frame baseline respiration");
    const auto cursorBaselineBrain = readBufferBytes(
        coupling.brain.accepted, "multi-frame baseline Brain");
    restoreInitialCoupling();
    physiology.batch(0u, 1u);
    physiology.batch(1u, 3u, true, 2u);
    const RuntimeStateSnapshot cursorRejectedMatter =
        physiology.runtime.snapshot();
    const auto cursorRejectedRespiration = readBufferBytes(
        respiration.accepted, "multi-frame rejected respiration");
    const auto cursorRejectedBrain = readBufferBytes(
        coupling.brain.accepted, "multi-frame rejected Brain");
    need(sameMatterCirculation(cursorBaselineMatter, cursorRejectedMatter) &&
             sameBytes(cursorBaselineRespiration,
                       cursorRejectedRespiration) &&
             sameBytes(cursorBaselineBrain, cursorRejectedBrain),
         "same-command-buffer rejection did not retain exactly the accepted two-frame prefix");
    restoreInitialCoupling();
    respiration.dispatch.reject = 0u;

    // Exercise a rejected multi-root Human continuation as one submission.
    // The rejection is injected at global control step 4; step 5 is the inert
    // suffix. Raw physical buffers are copied by the coupling callback because
    // a failed submission deliberately does not publish a resident token.
    metalrobo::MetalArticulatedOperatorContext multiBaselineContext(config);
    auto multiBaseline2Input = makeSegmentInput(false, 0u, 0u, 0u, 2u, true);
    metalrobo::MetalArticulatedOperatorResult multiBaseline2Result;
    const auto multiBaseline2Diagnostics = multiBaselineContext.run(
        model, multiBaseline2Input, multiBaseline2Result);
    requireAcceptedStep(multiBaseline2Diagnostics, multiBaseline2Result, 2u,
                        qCount, vCount, muscleCount);
    const auto multiBaseline2Matter = physiology.runtime.snapshot();
    const auto multiBaseline2Memory = captureCouplingMemory(coupling);
    const PhysicalSnapshot multiBaseline2Physical = observePhysicalState(
        multiBaselineContext, physical,
        multiBaseline2Diagnostics.residentStateTransactionFingerprint,
        multiBaseline2Diagnostics.residentStateGeneration);

    auto multiBaseline4Input = makeSegmentInput(
        true, multiBaseline2Diagnostics.residentStateTransactionFingerprint,
        multiBaseline2Diagnostics.residentStateGeneration, 2u, 2u, true);
    metalrobo::MetalArticulatedOperatorResult multiBaseline4Result;
    const auto multiBaseline4Diagnostics = multiBaselineContext.run(
        model, multiBaseline4Input, multiBaseline4Result);
    requireAcceptedStep(multiBaseline4Diagnostics, multiBaseline4Result, 4u,
                        qCount, vCount, muscleCount);
    const auto multiBaseline4Matter = physiology.runtime.snapshot();
    const auto multiBaseline4Memory = captureCouplingMemory(coupling);
    const PhysicalSnapshot multiBaseline4Physical = observePhysicalState(
        multiBaselineContext, physical,
        multiBaseline4Diagnostics.residentStateTransactionFingerprint,
        multiBaseline4Diagnostics.residentStateGeneration);

    auto multiBaseline6Input = makeSegmentInput(
        true, multiBaseline4Diagnostics.residentStateTransactionFingerprint,
        multiBaseline4Diagnostics.residentStateGeneration, 4u, 2u, true);
    metalrobo::MetalArticulatedOperatorResult multiBaseline6Result;
    const auto multiBaseline6Diagnostics = multiBaselineContext.run(
        model, multiBaseline6Input, multiBaseline6Result);
    requireAcceptedStep(multiBaseline6Diagnostics, multiBaseline6Result, 6u,
                        qCount, vCount, muscleCount);
    const auto multiBaseline6Matter = physiology.runtime.snapshot();
    const auto multiBaseline6Memory = captureCouplingMemory(coupling);
    const PhysicalSnapshot multiBaseline6Physical = observePhysicalState(
        multiBaselineContext, physical,
        multiBaseline6Diagnostics.residentStateTransactionFingerprint,
        multiBaseline6Diagnostics.residentStateGeneration);

    restoreInitialCoupling();
    respiration.dispatch.reject = 0u;
    metalrobo::MetalArticulatedOperatorContext multiFailureContext(config);
    auto multiTrial2Input = makeSegmentInput(false, 0u, 0u, 0u, 2u, true);
    metalrobo::MetalArticulatedOperatorResult multiTrial2Result;
    const auto multiTrial2Diagnostics = multiFailureContext.run(
        model, multiTrial2Input, multiTrial2Result);
    requireAcceptedStep(multiTrial2Diagnostics, multiTrial2Result, 2u,
                        qCount, vCount, muscleCount);
    const auto multiTrial2Matter = physiology.runtime.snapshot();
    const auto multiTrial2Memory = captureCouplingMemory(coupling);
    const PhysicalSnapshot multiTrial2Physical = observePhysicalState(
        multiFailureContext, physical,
        multiTrial2Diagnostics.residentStateTransactionFingerprint,
        multiTrial2Diagnostics.residentStateGeneration);
    need(samePhysicalState(multiBaseline2Physical, multiTrial2Physical) &&
             sameMatterAcceptedState(multiBaseline2Matter, multiTrial2Matter) &&
             sameAcceptedCouplingMemory(multiBaseline2Memory, multiTrial2Memory),
         "multi-step Human transaction prefix did not replay its accepted two-root seed");

    PhysicalSnapshot multiFailedRawPhysical(
        coupling.physiology.device, model, seedInput);
    coupling.transactionProbeRoots = multiFailedRawPhysical.roots;
    coupling.transactionProbeQ = multiFailedRawPhysical.q;
    coupling.transactionProbeV = multiFailedRawPhysical.v;
    coupling.transactionProbeMuscles = multiFailedRawPhysical.muscles;
    coupling.transactionProbeCaptureControlStep = 5u;
    coupling.transactionProbeCaptured = false;
    coupling.transactionProbeCapturedStep = MR_INVALID_INDEX;
    respiration.diagnosticRejectAtControlStep = 4u;
    auto multiRejectedInput = makeSegmentInput(
        true, multiTrial2Diagnostics.residentStateTransactionFingerprint,
        multiTrial2Diagnostics.residentStateGeneration, 2u, 4u, false);
    metalrobo::MetalArticulatedOperatorResult multiRejectedResult;
    multiRejectedResult.standQ = {-903.0f};
    const auto multiRejectedDiagnostics = multiFailureContext.run(
        model, multiRejectedInput, multiRejectedResult);
    const bool capturedInertSuffix = coupling.transactionProbeCaptured &&
        coupling.transactionProbeCapturedStep == 5u;
    coupling.transactionProbeRoots = nil;
    coupling.transactionProbeQ = nil;
    coupling.transactionProbeV = nil;
    coupling.transactionProbeMuscles = nil;
    coupling.transactionProbeCaptureControlStep = MR_INVALID_INDEX;
    coupling.transactionProbeCaptured = false;
    coupling.transactionProbeCapturedStep = MR_INVALID_INDEX;
    respiration.diagnosticRejectAtControlStep = NM_INVALID_INDEX;
    const auto multiFailedMatter = physiology.runtime.snapshot();
    const auto multiFailedMemory = captureCouplingMemory(coupling);
    readPhysicalSnapshotBuffers(multiFailedRawPhysical);
    const bool rejectedAtGlobalStep4 = !multiRejectedDiagnostics.succeeded() &&
        multiRejectedDiagnostics.dispatched && !multiRejectedDiagnostics.published &&
        multiRejectedDiagnostics.firstStandGPUStatusCode ==
            MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED &&
        multiRejectedDiagnostics.completedStandSteps == 4u &&
        multiRejectedResult.standQ == std::vector<float>{-903.0f};
    const bool acceptedPrefixAtFour =
        samePhysicalState(multiBaseline4Physical, multiFailedRawPhysical) &&
        sameMatterAcceptedState(multiBaseline4Matter, multiFailedMatter) &&
        sameAcceptedCouplingMemory(multiBaseline4Memory, multiFailedMemory);
    if (!rejectedAtGlobalStep4 || !capturedInertSuffix || !acceptedPrefixAtFour) {
        std::cerr << "resting_multistep_prefix_diagnostic"
                  << " reject_step4=" << rejectedAtGlobalStep4
                  << " captured_suffix5=" << capturedInertSuffix
                  << " completed=" << multiRejectedDiagnostics.completedStandSteps
                  << " unpublished=" << !multiRejectedDiagnostics.published
                  << " physical=" << samePhysicalState(multiBaseline4Physical, multiFailedRawPhysical)
                  << " matter=" << sameMatterAcceptedState(multiBaseline4Matter, multiFailedMatter)
                  << " coupled=" << sameAcceptedCouplingMemory(multiBaseline4Memory, multiFailedMemory)
                  << " q_diff=" << firstDifferingByte(multiBaseline4Physical.qValues, multiFailedRawPhysical.qValues)
                  << " v_diff=" << firstDifferingByte(multiBaseline4Physical.vValues, multiFailedRawPhysical.vValues)
                  << " root_diff=" << firstDifferingByte(multiBaseline4Physical.rootValues, multiFailedRawPhysical.rootValues)
                  << " muscle_diff=" << firstDifferingByte(multiBaseline4Physical.muscleValues, multiFailedRawPhysical.muscleValues)
                  << " matter_vascular_diff=" << firstDifferingByte(multiBaseline4Matter.vascularState, multiFailedMatter.vascularState)
                  << " matter_clock_diff=" << firstDifferingByte(multiBaseline4Matter.vascularClock, multiFailedMatter.vascularClock)
                  << " respiration_diff=" << firstDifferingByte(multiBaseline4Memory.respirationAccepted, multiFailedMemory.respirationAccepted)
                  << " brain_diff=" << firstDifferingByte(multiBaseline4Memory.brainAccepted, multiFailedMemory.brainAccepted)
                  << " presentation_body_diff=" << firstDifferingByte(multiBaseline4Memory.presentationBodies, multiFailedMemory.presentationBodies)
                  << " presentation_resp_diff=" << firstDifferingByte(multiBaseline4Memory.presentationRespiration, multiFailedMemory.presentationRespiration)
                  << " presentation_common_diff=" << firstDifferingByte(multiBaseline4Memory.presentationCommonCoordinates, multiFailedMemory.presentationCommonCoordinates)
                  << " accepted_common_diff=" << firstDifferingByte(multiBaseline4Memory.acceptedCommonCoordinates, multiFailedMemory.acceptedCommonCoordinates)
                  << " frame_common_diff=" << firstDifferingByte(multiBaseline4Memory.presentationFrameCommonCoordinates, multiFailedMemory.presentationFrameCommonCoordinates)
                  << '\n';
    }
    need(rejectedAtGlobalStep4 && capturedInertSuffix,
         "multi-step Human rejection did not occur at global step 4 before an inert step-5 suffix: " +
             multiRejectedDiagnostics.message);
    need(acceptedPrefixAtFour,
         "failed multi-step Human submission did not retain one coherent accepted four-root physical and coupled prefix");

    // The failed continuation is terminal on its original context by contract.
    // Recreate the accepted two-root prefix in a fresh context, then replay the
    // same four-root continuation and compare against the uninterrupted run.
    restoreInitialCoupling();
    respiration.dispatch.reject = 0u;
    metalrobo::MetalArticulatedOperatorContext multiRetryContext(config);
    auto multiRetry2Input = makeSegmentInput(false, 0u, 0u, 0u, 2u, true);
    metalrobo::MetalArticulatedOperatorResult multiRetry2Result;
    const auto multiRetry2Diagnostics = multiRetryContext.run(
        model, multiRetry2Input, multiRetry2Result);
    requireAcceptedStep(multiRetry2Diagnostics, multiRetry2Result, 2u,
                        qCount, vCount, muscleCount);
    auto multiRetry6Input = makeSegmentInput(
        true, multiRetry2Diagnostics.residentStateTransactionFingerprint,
        multiRetry2Diagnostics.residentStateGeneration, 2u, 4u, true);
    metalrobo::MetalArticulatedOperatorResult multiRetry6Result;
    const auto multiRetry6Diagnostics = multiRetryContext.run(
        model, multiRetry6Input, multiRetry6Result);
    requireAcceptedStep(multiRetry6Diagnostics, multiRetry6Result, 6u,
                        qCount, vCount, muscleCount);
    const auto multiRetryMatter = physiology.runtime.snapshot();
    const auto multiRetryMemory = captureCouplingMemory(coupling);
    const PhysicalSnapshot multiRetryPhysical = observePhysicalState(
        multiRetryContext, physical,
        multiRetry6Diagnostics.residentStateTransactionFingerprint,
        multiRetry6Diagnostics.residentStateGeneration);
    need(samePhysicalState(multiBaseline6Physical, multiRetryPhysical) &&
             sameMatterAcceptedState(multiBaseline6Matter, multiRetryMatter) &&
             sameAcceptedCouplingMemory(multiBaseline6Memory, multiRetryMemory),
         "fresh-context multi-step replay did not reproduce the uninterrupted six-root coupled endpoint");

    restoreInitialCoupling();
    std::cout << "resting_integrated_rejection=pass rejected_after_accepted_predecessor=true"
                 " body_q_v_root_myo_unchanged=true circulation_and_clock_unchanged=true"
                 " respiration_brain_history_unchanged=true retry_matches_uninterrupted_replay=true"
                 " same_command_buffer_reject=pass accepted_prefix=2 rejected_step=2 inert_suffix=1"
                 " multistep_reject=pass accepted_prefix=4 rejected_global_step=4 inert_suffix_global_step=5"
                 " raw_failed_submission_snapshot=pass all_owners_match_uninterrupted_prefix=true"
                 " retry=fresh_context_reseed_and_replay_matches_six_root_baseline"
                 " same_context_retry=unsupported\n";
}

} // namespace numi::human::transaction_probe
