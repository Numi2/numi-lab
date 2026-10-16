#pragma once

#include "numi/matter/shared.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace numi::cardboard {

struct FreeForceResidualBounds {
    double l2Newtons = 0.0;
    double maximumNodeNewtons = 0.0;
    std::size_t freeNodeCount = 0u;
};

// The native FEM residual is an impulse (N s). Divide by this accepted step's
// dt and include only currently free nodes. This is a measured residual, not
// an estimate of force error.
inline std::optional<FreeForceResidualBounds>
calculateFreeForceResidualBounds(
    const std::span<const nm_float4> residuals,
    const std::span<const NMFEMNodeStateGPU> nodes,
    const std::size_t environmentCount,
    const std::size_t nodeCount,
    const std::size_t environment,
    const double dtSeconds) noexcept {
    if (!std::isfinite(dtSeconds) || dtSeconds <= 0.0 ||
        environmentCount == 0u || nodeCount == 0u ||
        environment >= environmentCount ||
        nodeCount > std::numeric_limits<std::size_t>::max() / environmentCount)
        return std::nullopt;
    const std::size_t total = environmentCount * nodeCount;
    if (residuals.size() < total || nodes.size() < total)
        return std::nullopt;
    const std::size_t base = environment * nodeCount;
    double sumSquared = 0.0;
    FreeForceResidualBounds result;
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        const float fixed = nodes[base + node].restAndFixed.w;
        if (!std::isfinite(fixed)) return std::nullopt;
        if (fixed != 0.0f) continue;
        const nm_float4 residual = residuals[base + node];
        if (!std::isfinite(residual.x) || !std::isfinite(residual.y) ||
            !std::isfinite(residual.z))
            return std::nullopt;
        const double fx = static_cast<double>(residual.x) / dtSeconds;
        const double fy = static_cast<double>(residual.y) / dtSeconds;
        const double fz = static_cast<double>(residual.z) / dtSeconds;
        const double magnitude = std::hypot(std::hypot(fx, fy), fz);
        if (!std::isfinite(magnitude)) return std::nullopt;
        sumSquared += magnitude * magnitude;
        if (!std::isfinite(sumSquared)) return std::nullopt;
        result.maximumNodeNewtons = std::max(result.maximumNodeNewtons, magnitude);
        ++result.freeNodeCount;
    }
    if (result.freeNodeCount == 0u) return std::nullopt;
    result.l2Newtons = std::sqrt(sumSquared);
    if (!std::isfinite(result.l2Newtons)) return std::nullopt;
    return result;
}

}  // namespace numi::cardboard
