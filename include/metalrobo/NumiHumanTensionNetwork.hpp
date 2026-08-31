#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// Static reference mechanics for tendon sheets represented as an explicit
// graph of axial collagen bundles. Elements carry tension only; they are not
// joint torques and do not bypass attachment reactions.
struct NumiHumanTensionNetworkNode {
    std::array<double, 3> position{};
    bool fixed = false;
};

struct NumiHumanTensionNetworkElement {
    std::uint32_t nodeA = 0u;
    std::uint32_t nodeB = 0u;
    double restLength = 0.0;
    double youngModulus = 0.0;
    double area = 0.0;
};

struct NumiHumanTensionNetworkLoad {
    std::uint32_t nodeIndex = 0u;
    std::array<double, 3> force{};
};

struct NumiHumanTensionNetworkConfig {
    std::uint32_t maximumIterations = 128u;
    std::uint32_t maximumLineSearchSteps = 24u;
    double forceTolerance = 1.0e-9;
    double minimumLength = 1.0e-9;
    double diagonalRegularization = 1.0e-10;
    double armijoFraction = 1.0e-4;
};

enum class NumiHumanTensionNetworkStatus : std::uint32_t {
    success = 0u,
    invalidTopology,
    invalidMaterial,
    invalidLoad,
    singularSystem,
    didNotConverge,
    nonfiniteResult,
};

struct NumiHumanTensionNetworkResult {
    std::vector<std::array<double, 3>> position;
    std::vector<double> elementTension;
    std::vector<std::array<double, 3>> nodeResidualForce;
    std::vector<std::array<double, 3>> fixedReactionForce;
    double strainEnergy = 0.0;
    double potentialEnergy = 0.0;
    double maximumFreeNodeResidual = 0.0;
    std::array<double, 3> forceClosureResidual{};
    std::array<double, 3> momentClosureResidual{};
    std::uint32_t activeElementCount = 0u;
    std::uint32_t completedIterations = 0u;
};

struct NumiHumanTensionNetworkDiagnostics {
    NumiHumanTensionNetworkStatus status =
        NumiHumanTensionNetworkStatus::success;
    std::uint32_t failingIndex = 0u;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanTensionNetworkStatus::success;
    }
};

[[nodiscard]] NumiHumanTensionNetworkDiagnostics
solveNumiHumanTensionNetwork(
    std::span<const NumiHumanTensionNetworkNode> nodes,
    std::span<const NumiHumanTensionNetworkElement> elements,
    std::span<const NumiHumanTensionNetworkLoad> loads,
    NumiHumanTensionNetworkResult& result,
    const NumiHumanTensionNetworkConfig& config = {}
);

[[nodiscard]] const char* numiHumanTensionNetworkStatusName(
    NumiHumanTensionNetworkStatus status
) noexcept;

} // namespace metalrobo
