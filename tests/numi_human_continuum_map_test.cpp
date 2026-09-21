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

double determinant(
    const std::vector<std::array<double, 3u>>& points
) {
    const auto& a = points[0u];
    const std::array<double, 3u> ab{{
        points[1u][0u] - a[0u], points[1u][1u] - a[1u],
        points[1u][2u] - a[2u]}};
    const std::array<double, 3u> ac{{
        points[2u][0u] - a[0u], points[2u][1u] - a[1u],
        points[2u][2u] - a[2u]}};
    const std::array<double, 3u> ad{{
        points[3u][0u] - a[0u], points[3u][1u] - a[1u],
        points[3u][2u] - a[2u]}};
    return ab[0u] * (ac[1u] * ad[2u] - ac[2u] * ad[1u]) -
        ab[1u] * (ac[0u] * ad[2u] - ac[2u] * ad[0u]) +
        ab[2u] * (ac[0u] * ad[1u] - ac[1u] * ad[0u]);
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

    {
        // The one-shot inverse-distance blend folds this tetrahedron, while
        // deterministic dyadic continuation reaches the exact same rigid
        // endpoint constraints in two orientation-preserving increments.
        const std::vector<std::uint32_t> anchors{
            10u, 20u,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX};
        const std::array bodies{
            body(10u, {0.0, 0.0, 0.0}),
            body(20u, {-0.45153725588341853,
                       0.448807078987282,
                       -2.51833563015108})};
        metalrobo::NumiHumanContinuumMapResult direct;
        const auto rejected =
            metalrobo::mapNumiHumanContinuumToMovingEntheses(
                points, tetrahedra, anchors, bodies, direct);
        require(rejected.jacobianGateFailure &&
                    rejected.failingIndex == 0u &&
                    rejected.failingJacobian < 0.0 &&
                    direct.targetWorldPoints.empty(),
                "synthetic direct inversion did not fail closed");

        metalrobo::NumiHumanContinuumMapResult continued;
        const auto accepted = metalrobo::
            mapNumiHumanContinuumToMovingEnthesesWithContinuation(
                points, tetrahedra, anchors, bodies, continued);
        require(accepted.succeeded() && accepted.substepCount == 2u &&
                    accepted.directMap.jacobianGateFailure &&
                    accepted.directMap.failingJacobian < 0.0 &&
                    continued.targetWorldPoints.size() == points.size() &&
                    determinant(continued.targetWorldPoints) > 0.0,
                "dyadic continuation did not repair a free-node fold");
        require(near(continued.targetWorldPoints[0u][0u], 0.0) &&
                    near(continued.targetWorldPoints[0u][1u], 0.0) &&
                    near(continued.targetWorldPoints[0u][2u], 0.0) &&
                    near(continued.targetWorldPoints[1u][0u],
                         0.5484627441165815) &&
                    near(continued.targetWorldPoints[1u][1u],
                         0.448807078987282) &&
                    near(continued.targetWorldPoints[1u][2u],
                         -2.51833563015108),
                "continued moving entheses are not exact");

        metalrobo::NumiHumanContinuumMapResult replay;
        const auto replayed = metalrobo::
            mapNumiHumanContinuumToMovingEnthesesWithContinuation(
                points, tetrahedra, anchors, bodies, replay);
        require(replayed.succeeded() && replayed.substepCount == 2u &&
                    replay.targetWorldPoints == continued.targetWorldPoints,
                "dyadic continuation replay is not deterministic");

        metalrobo::NumiHumanContinuumMapResult invalid;
        const auto nonDyadic = metalrobo::
            mapNumiHumanContinuumToMovingEnthesesWithContinuation(
                points, tetrahedra, anchors, bodies, invalid, {}, 3u);
        require(nonDyadic.finalMap.status ==
                    metalrobo::NumiHumanContinuumMapStatus::invalidInput &&
                    invalid.targetWorldPoints.empty(),
                "non-dyadic continuation cap did not fail closed");
        const auto oversized = metalrobo::
            mapNumiHumanContinuumToMovingEnthesesWithContinuation(
                points, tetrahedra, anchors, bodies, invalid, {}, 512u);
        require(oversized.finalMap.status ==
                    metalrobo::NumiHumanContinuumMapStatus::invalidInput &&
                    invalid.targetWorldPoints.empty(),
                "oversized continuation cap did not fail closed");
    }

    {
        // Each of two increments stays below the configured expansion bound,
        // but their compounded original-to-target Jacobian is 1.21. The
        // continuation must not use subdivision to evade the total gate.
        const std::vector<std::uint32_t> anchors{
            10u, 20u,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
            metalrobo::NUMI_HUMAN_CONTINUUM_INVALID_INDEX};
        const std::array bodies{
            body(10u, {0.0, 0.0, 0.0}),
            body(20u, {0.21, 0.0, 0.0})};
        metalrobo::NumiHumanContinuumMapConfig config;
        config.maximumJacobian = 1.15;
        metalrobo::NumiHumanContinuumMapResult result;
        const auto rejected = metalrobo::
            mapNumiHumanContinuumToMovingEnthesesWithContinuation(
                points, tetrahedra, anchors, bodies, result, config, 2u);
        require(!rejected.succeeded() &&
                    rejected.substepCount == 0u &&
                    rejected.directMap.jacobianGateFailure &&
                    rejected.finalMap.jacobianGateFailure &&
                    rejected.finalMap.failingIndex == 0u &&
                    near(rejected.finalMap.failingJacobian, 1.21) &&
                    rejected.finalMap.message.find(
                        "continuation final map violates") !=
                        std::string::npos &&
                    result.targetWorldPoints.empty(),
                "continuation admitted a compounded out-of-bounds map");
    }

    std::cout << "numi_human_continuum_map_test=passed\n";
    return 0;
}
