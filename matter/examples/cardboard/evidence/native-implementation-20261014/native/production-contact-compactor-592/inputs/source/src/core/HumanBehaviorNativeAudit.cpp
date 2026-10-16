#include "metalrobo/HumanBehaviorNativeAudit.hpp"

#include "metalrobo/numanx_coupled_human_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"

#include <algorithm>
#include <cmath>

namespace metalrobo {
namespace {

[[nodiscard]] bool finitePhysicalDiagnostics(
    const MetalNumanXHumanMatterPhysicalOutcome& outcome
) noexcept {
    return std::all_of(
               outcome.humanContactAndAcceleration.begin(),
               outcome.humanContactAndAcceleration.end(),
               [](const float value) { return std::isfinite(value); }) &&
        std::all_of(
               outcome.humanFactorAndAssistance.begin(),
               outcome.humanFactorAndAssistance.end(),
               [](const float value) { return std::isfinite(value); }) &&
        std::all_of(
               outcome.matterDiagnostics.begin(),
               outcome.matterDiagnostics.end(),
               [](const float value) { return std::isfinite(value); });
}

[[nodiscard]] bool finiteBehaviorMetrics(
    const MRHumanBehaviorCandidateGPU& candidate
) noexcept {
    return std::isfinite(candidate.valueHigh.x) &&
        std::isfinite(candidate.valueHigh.y) &&
        std::isfinite(candidate.valueHigh.z) &&
        std::isfinite(candidate.valueHigh.w) &&
        std::isfinite(candidate.valueLow.x) &&
        std::isfinite(candidate.valueLow.y) &&
        std::isfinite(candidate.valueLow.z) &&
        std::isfinite(candidate.valueLow.w);
}

} // namespace

HumanBehaviorNativeAuditResult humanBehaviorNativeNonfiniteAudit(
    const bool physicalOwnerReady,
    const MetalNumanXHumanMatterPhysicalOutcome* const physicalOutcome,
    const MRHumanBehaviorCandidateGPU* const behaviorCandidate,
    const HumanBehaviorNativeAttemptIdentity& expected
) noexcept {
    HumanBehaviorNativeAuditResult result{};
    if (!physicalOwnerReady || physicalOutcome == nullptr ||
        behaviorCandidate == nullptr) {
        return result;
    }

    const auto& physical = *physicalOutcome;
    const auto& candidate = *behaviorCandidate;
    const bool successfulPhysicalOwners =
        physical.jointDecision == MR_NUMANX_COUPLED_HUMAN_ACCEPT &&
        physical.humanCode == MR_NUMI_HUMAN_STAND_SUCCESS &&
        physical.matterCode == 0u && physical.humanCompletedSteps == 1u &&
        physical.matterCompletedMicrosteps == 1u &&
        physical.worldCode == MR_STEP_SUCCESS &&
        physical.worldSuccessfulSubsteps == 1u &&
        physical.worldABACode == MR_ABA_SUCCESS;
    const bool exactBehaviorCandidate =
        candidate.abiVersion == MR_HUMAN_BEHAVIOR_ABI_VERSION &&
        candidate.status == 0u && candidate.postureValid <= 1u &&
        candidate.settled <= 1u &&
        candidate.programFingerprint == expected.behaviorProgramFingerprint &&
        candidate.transactionFingerprint == expected.transactionFingerprint &&
        candidate.linearizationEpoch == expected.linearizationEpoch &&
        candidate.slotGeneration == expected.slotGeneration &&
        candidate.physicsGeneration == expected.physicsGeneration &&
        candidate.acceptedTimestampNanoseconds ==
            expected.acceptedTimestampNanoseconds;
    if (!successfulPhysicalOwners || !exactBehaviorCandidate) return result;

    result.coveredMask = MR_HUMAN_BEHAVIOR_AUDIT_NONFINITE;
    if (!finitePhysicalDiagnostics(physical) ||
        !finiteBehaviorMetrics(candidate)) {
        result.violationMask = MR_HUMAN_BEHAVIOR_AUDIT_NONFINITE;
    }
    return result;
}

} // namespace metalrobo
