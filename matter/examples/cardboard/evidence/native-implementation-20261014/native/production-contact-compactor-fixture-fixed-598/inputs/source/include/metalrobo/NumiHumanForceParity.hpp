#pragma once

#include "metalrobo/engine_types.h"

#include <cmath>
#include <cstdint>
#include <span>

namespace metalrobo {

// Compares two unconstrained one-step velocity publications produced from the
// same q, v, timestep, mass operator, gravity, and damping but different
// generalized-force vectors. This converts force-law parity into the dynamic
// units that matter for low-inertia Human coordinates. It is diagnostic only:
// it does not include contact, equalities, limits, or their reaction impulses.
struct NumiHumanForceParityDiagnostics {
    bool valid = false;
    std::uint32_t maximumVelocityDeltaDof = MR_INVALID_INDEX;
    double maximumVelocityDelta = 0.0;
    double maximumAccelerationDelta = 0.0;
};

[[nodiscard]] inline NumiHumanForceParityDiagnostics
compareNumiHumanUnconstrainedVelocityParity(
    const std::span<const double> referenceVelocity,
    const std::span<const double> candidateVelocity,
    const double timestepSeconds
) noexcept {
    NumiHumanForceParityDiagnostics result;
    if (referenceVelocity.empty() ||
        referenceVelocity.size() != candidateVelocity.size() ||
        !std::isfinite(timestepSeconds) || !(timestepSeconds > 0.0)) {
        return result;
    }
    for (std::size_t dof = 0u; dof < referenceVelocity.size(); ++dof) {
        const double reference = referenceVelocity[dof];
        const double candidate = candidateVelocity[dof];
        if (!std::isfinite(reference) || !std::isfinite(candidate)) {
            return {};
        }
        const double delta = std::abs(candidate - reference);
        if (result.maximumVelocityDeltaDof == MR_INVALID_INDEX ||
            delta > result.maximumVelocityDelta) {
            result.maximumVelocityDelta = delta;
            result.maximumVelocityDeltaDof = static_cast<std::uint32_t>(dof);
        }
    }
    result.maximumAccelerationDelta =
        result.maximumVelocityDelta / timestepSeconds;
    if (!std::isfinite(result.maximumAccelerationDelta)) {
        return {};
    }
    result.valid = true;
    return result;
}

} // namespace metalrobo
