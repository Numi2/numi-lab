#pragma once

#include "metalrobo/mujoco_muscle_gpu.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace metalrobo {

inline constexpr std::size_t kNumiHumanDynamicStateMuscleCount = 416u;

enum class NumiHumanDynamicStateSeedStatus : std::uint32_t {
    success = 0u,
    dimensionMismatch,
    invalidActivation,
    invalidFiberLength,
};

struct NumiHumanDynamicStateSeedDiagnostics {
    NumiHumanDynamicStateSeedStatus status =
        NumiHumanDynamicStateSeedStatus::success;
    std::size_t failingIndex = 0u;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanDynamicStateSeedStatus::success;
    }
};

// Constructs the persistent Human muscle sidecar from the accepted static
// equilibrium. The accepted fibre length is transported explicitly; zero is
// reserved for a separate first-evaluation equilibrium solve and therefore is
// not admissible in this qualification path. Output is unchanged on failure.
[[nodiscard]] inline NumiHumanDynamicStateSeedDiagnostics
seedNumiHumanDynamicMuscleState(
    const std::span<const double> activation,
    const std::span<const double> acceptedFiberLengthMeters,
    std::vector<MRMujocoMuscleStateGPU>& output
) {
    NumiHumanDynamicStateSeedDiagnostics diagnostics;
    if (activation.size() != kNumiHumanDynamicStateMuscleCount ||
        acceptedFiberLengthMeters.size() !=
            kNumiHumanDynamicStateMuscleCount) {
        diagnostics.status =
            NumiHumanDynamicStateSeedStatus::dimensionMismatch;
        return diagnostics;
    }

    std::vector<MRMujocoMuscleStateGPU> candidate(
        kNumiHumanDynamicStateMuscleCount);
    constexpr double kFloatMaximum =
        static_cast<double>(std::numeric_limits<float>::max());
    for (std::size_t muscle = 0u; muscle < candidate.size(); ++muscle) {
        const double activationValue = activation[muscle];
        if (!std::isfinite(activationValue) || activationValue < 0.0 ||
            activationValue > 1.0) {
            diagnostics.status =
                NumiHumanDynamicStateSeedStatus::invalidActivation;
            diagnostics.failingIndex = muscle;
            return diagnostics;
        }
        const double fiberLength = acceptedFiberLengthMeters[muscle];
        if (!std::isfinite(fiberLength) || fiberLength <= 0.0 ||
            fiberLength > kFloatMaximum) {
            diagnostics.status =
                NumiHumanDynamicStateSeedStatus::invalidFiberLength;
            diagnostics.failingIndex = muscle;
            return diagnostics;
        }
        const float activationFP32 = static_cast<float>(activationValue);
        const float fiberLengthFP32 = static_cast<float>(fiberLength);
        if (!std::isfinite(activationFP32) || !std::isfinite(fiberLengthFP32) ||
            fiberLengthFP32 <= 0.0f) {
            diagnostics.status =
                NumiHumanDynamicStateSeedStatus::invalidFiberLength;
            diagnostics.failingIndex = muscle;
            return diagnostics;
        }
        candidate[muscle].excitationAndActivation = {
            activationFP32,
            activationFP32,
            fiberLengthFP32,
            0.0f,
        };
    }

    output = std::move(candidate);
    return diagnostics;
}

} // namespace metalrobo
