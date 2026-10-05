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
    respiration.dispatch.reject = 0u;

    auto restoreInitialCoupling = [&]() {
        const auto restored = physiology.runtime.restore(originalMatter);
        need(restored.encoded, "resting rejection probe could not restore Matter seed: " +
             restored.message);
        restoreCouplingMemory(coupling, originalMemory);
        respiration.dispatch = originalDispatch;
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
                       trialFirstMemory.brainAccepted),
         "resting transaction predecessor did not replay from the same accepted seed");

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
             rejectedDiagnostics.message);
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
    const bool rejectedStateUnchanged = unchangedRoots && unchangedQ &&
        unchangedV && unchangedMuscles && unchangedMatterState &&
        unchangedMatterClock && unchangedMatterMetadata && unchangedRespiration &&
        unchangedBrain && unchangedPresentedBodies && unchangedPresentedRespiration;
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
                       retryMemory.presentationRespiration),
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
    std::cout << "resting_integrated_rejection=pass rejected_after_accepted_predecessor=true"
                 " body_q_v_root_myo_unchanged=true circulation_and_clock_unchanged=true"
                 " respiration_brain_history_unchanged=true retry_matches_uninterrupted_replay=true"
                 " same_command_buffer_reject=pass accepted_prefix=2 rejected_step=2 inert_suffix=1\n";
}

} // namespace numi::human::transaction_probe
