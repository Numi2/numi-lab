#include "metalrobo/NumiHumanContinuumMap.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

void require(const bool condition, const std::string_view message) {
    if (condition) return;
    std::cerr << "numi_human_continuum_map_test=failed error=\""
              << message << "\"\n";
    std::exit(1);
}

bool near(const double actual, const double expected,
          const double tolerance = 1.0e-10) {
    return std::abs(actual - expected) <= tolerance;
}

metalrobo::NumiHumanContinuumBodyMap body(
    const std::uint32_t index,
    const std::array<double, 3u>& targetPosition,
    const std::array<double, 4u>& targetOrientation = {0.0, 0.0, 0.0, 1.0}
) {
    metalrobo::NumiHumanContinuumBodyMap result;
    result.bodyIndex = index;
    result.targetPose.position = targetPosition;
    result.targetPose.orientation = targetOrientation;
    return result;
}

} // namespace

int main() {
    const std::vector<std::array<double, 3u>> points{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}};
    const std::vector<std::array<std::uint32_t, 4u>> tetrahedra{
        {0u, 1u, 2u, 3u}};

    {
        const std::vector<std::uint32_t> anchors{
            10u, 20u,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX};
        const std::array bodies{
            body(10u, {0.0, 0.0, 0.0}),
            body(20u, {0.0, 0.1, 0.0})};
        metalrobo::NumiHumanContinuumMapResult result;
        const auto diagnostics =
            metalrobo::mapNumiHumanContinuumToMovingEntheses(
                points, tetrahedra, anchors, bodies, result);
        require(diagnostics.succeeded(), diagnostics.message);
        require(result.targetWorldPoints.size() == points.size(),
                "two-owner map lost nodes");
        require(near(result.targetWorldPoints[0u][1u], 0.0) &&
                    near(result.targetWorldPoints[1u][1u], 0.1),
                "moving entheses are not exact");
        require(result.targetWorldPoints[2u][1u] > 1.0 &&
                    result.targetWorldPoints[2u][1u] < 1.1,
                "free node did not blend owner motion");
        require(diagnostics.ownerCount == 2u &&
                    diagnostics.anchorCount == 2u &&
                    diagnostics.maximumAnchorResidualMeters == 0.0 &&
                    diagnostics.minimumJacobian > 0.0,
                "two-owner diagnostics are incomplete");
    }

    {
        const std::vector<std::uint32_t> anchors{
            30u, metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX};
        const double halfRootTwo = std::sqrt(0.5);
        const std::array bodies{body(
            30u, {2.0, -1.0, 0.5},
            {0.0, 0.0, halfRootTwo, halfRootTwo})};
        metalrobo::NumiHumanContinuumMapResult result;
        const auto diagnostics =
            metalrobo::mapNumiHumanContinuumToMovingEntheses(
                points, tetrahedra, anchors, bodies, result);
        require(diagnostics.succeeded(), diagnostics.message);
        require(near(result.targetWorldPoints[0u][0u], 2.0) &&
                    near(result.targetWorldPoints[0u][1u], -1.0) &&
                    near(result.targetWorldPoints[1u][0u], 2.0) &&
                    near(result.targetWorldPoints[1u][1u], 0.0),
                "single-owner rigid transform is incorrect");
        require(near(diagnostics.minimumJacobian, 1.0) &&
                    near(diagnostics.maximumJacobian, 1.0),
                "single-owner rigid map changed volume");
    }

    {
        const std::vector<std::array<double, 3u>> disconnectedPoints{
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
            {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0},
            {3.0, 1.0, 0.0}, {3.0, 0.0, 1.0}};
        const std::vector<std::array<std::uint32_t, 4u>> disconnectedTets{
            {0u, 1u, 2u, 3u}, {4u, 5u, 6u, 7u}};
        const std::vector<std::uint32_t> anchors{
            10u, metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            20u, metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX};
        const std::array bodies{
            body(10u, {0.0, 0.0, 0.0}),
            body(20u, {0.0, 0.0, 0.0})};
        metalrobo::NumiHumanContinuumMapResult result;
        const auto diagnostics =
            metalrobo::mapNumiHumanContinuumToMovingEntheses(
                disconnectedPoints, disconnectedTets, anchors, bodies,
                result);
        require(diagnostics.status ==
                    metalrobo::NumiHumanContinuumMapStatus::disconnectedTopology,
                "disconnected owner topology did not fail closed");
    }

    {
        const std::vector<std::uint32_t> anchors{
            999u, metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX};
        const std::array bodies{body(10u, {0.0, 0.0, 0.0})};
        metalrobo::NumiHumanContinuumMapResult result;
        const auto diagnostics =
            metalrobo::mapNumiHumanContinuumToMovingEntheses(
                points, tetrahedra, anchors, bodies, result);
        require(diagnostics.status ==
                    metalrobo::NumiHumanContinuumMapStatus::invalidInput,
                "unknown anchor owner did not fail closed");
    }

    std::cout << "numi_human_continuum_map_test=passed\n";
    return 0;
}
