#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t NUMI_HUMAN_CONTINUUM_INVALID_INDEX =
    0xffffffffu;

struct NumiHumanContinuumRigidPose {
    std::array<double, 3u> position{};
    // Unit quaternion in (x, y, z, w) order.
    std::array<double, 4u> orientation{0.0, 0.0, 0.0, 1.0};
};

struct NumiHumanContinuumBodyMap {
    std::uint32_t bodyIndex = NUMI_HUMAN_CONTINUUM_INVALID_INDEX;
    NumiHumanContinuumRigidPose referencePose;
    NumiHumanContinuumRigidPose targetPose;
};

struct NumiHumanContinuumMapConfig {
    double inverseDistanceExponent = 2.0;
    double minimumJacobian = 0.05;
    double maximumJacobian = 20.0;
};

enum class NumiHumanContinuumMapStatus : std::uint32_t {
    success = 0u,
    invalidInput,
    invalidTopology,
    disconnectedTopology,
    invalidDeformation,
};

struct NumiHumanContinuumMapDiagnostics {
    NumiHumanContinuumMapStatus status =
        NumiHumanContinuumMapStatus::success;
    std::uint32_t failingIndex = NUMI_HUMAN_CONTINUUM_INVALID_INDEX;
    std::uint32_t ownerCount = 0u;
    std::uint32_t anchorCount = 0u;
    double maximumAnchorResidualMeters = 0.0;
    double maximumDisplacementMeters = 0.0;
    double minimumJacobian = 1.0;
    double maximumJacobian = 1.0;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanContinuumMapStatus::success;
    }
};

struct NumiHumanContinuumMapResult {
    std::vector<std::array<double, 3u>> targetWorldPoints;
};

// Maps a tetrahedral continuum from its source anatomical pose to moved bone
// poses. Each fixed-node owner seeds a geodesic distance field on the tetra
// edge graph. Free nodes use normalized inverse-distance weights to blend the
// owner transforms; fixed nodes use their owner transform exactly. This gives
// a topology-aware, smooth initial field with exact moving entheses instead of
// forcing an entire multi-bone tissue through one bone's rigid transform.
[[nodiscard]] NumiHumanContinuumMapDiagnostics
mapNumiHumanContinuumToMovingEntheses(
    std::span<const std::array<double, 3u>> referenceWorldPoints,
    std::span<const std::array<std::uint32_t, 4u>> tetrahedra,
    std::span<const std::uint32_t> anchorBodyIndices,
    std::span<const NumiHumanContinuumBodyMap> bodyMaps,
    NumiHumanContinuumMapResult& result,
    const NumiHumanContinuumMapConfig& config = {}
);

[[nodiscard]] const char* numiHumanContinuumMapStatusName(
    NumiHumanContinuumMapStatus status
) noexcept;

} // namespace metalrobo
