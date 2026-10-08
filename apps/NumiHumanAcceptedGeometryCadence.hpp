#pragma once

#include <cstdint>

namespace numiHumanRestingAcceptedGeometry {
enum class CaptureStepClass : std::uint8_t { invalid, initial, submission_endpoint, terminal };

constexpr CaptureStepClass classifyCaptureStep(
    const std::uint32_t step,
    const std::uint32_t totalAcceptedSteps,
    const std::uint32_t submissionSteps
) noexcept {
    if (totalAcceptedSteps == 0u || submissionSteps == 0u || step > totalAcceptedSteps)
        return CaptureStepClass::invalid;
    if (step == 0u) return CaptureStepClass::initial;
    if (step == totalAcceptedSteps) return CaptureStepClass::terminal;
    if ((step + 1u) % submissionSteps == 0u)
        return CaptureStepClass::submission_endpoint;
    return CaptureStepClass::invalid;
}

constexpr bool terminalSnapshotReady(
    const std::uint32_t expectedTerminalStep,
    const std::uint32_t completedAcceptedSteps,
    const bool standSucceeded,
    const std::uint32_t respirationStep,
    const bool respirationAccepted
) noexcept {
    return expectedTerminalStep > 0u &&
        completedAcceptedSteps == expectedTerminalStep &&
        standSucceeded && respirationStep == expectedTerminalStep &&
        respirationAccepted;
}
} // namespace numiHumanRestingAcceptedGeometry
