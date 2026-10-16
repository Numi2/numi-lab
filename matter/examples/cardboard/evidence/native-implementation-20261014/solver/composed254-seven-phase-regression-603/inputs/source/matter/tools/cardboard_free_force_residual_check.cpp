#include "cardboard_free_force_residual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>

namespace {
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
bool close(double a, double b) {
    return std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}
}

int main() {
    std::array<nm_float4, 6> residuals{};
    std::array<NMFEMNodeStateGPU, 6> nodes{};
    residuals[0] = {1.5f, 2.0f, 0.0f, 99.0f};  // (3,4,0) N at dt=.5
    residuals[1] = {1000.0f, 0.0f, 0.0f, 0.0f};
    residuals[2] = {0.0f, 0.0f, 6.0f, 0.0f};  // 12 N
    residuals[3] = {500.0f, 0.0f, 0.0f, 0.0f};
    residuals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    residuals[5] = {0.0f, 0.0f, 0.0f, 0.0f};
    nodes[1].restAndFixed.w = 1.0f;
    nodes[3].restAndFixed.w = 3.0f;

    auto env0 = numi::cardboard::calculateFreeForceResidualBounds(
        residuals, nodes, 2u, 3u, 0u, 0.5);
    auto env1 = numi::cardboard::calculateFreeForceResidualBounds(
        residuals, nodes, 2u, 3u, 1u, 0.5);
    check(env0.has_value() && env0->freeNodeCount == 2u, "env0 free-node selection");
    check(close(env0->l2Newtons, 13.0) && close(env0->maximumNodeNewtons, 12.0),
          "env0 impulse-to-force reduction");
    check(env1.has_value() && env1->freeNodeCount == 2u &&
          close(env1->l2Newtons, 0.0) && close(env1->maximumNodeNewtons, 0.0),
          "env1 stride and fixed-node masking");
    check(!numi::cardboard::calculateFreeForceResidualBounds(
              residuals, nodes, 2u, 3u, 2u, 0.5),
          "out-of-range environment must fail");
    check(!numi::cardboard::calculateFreeForceResidualBounds(
              residuals, nodes, 2u, 3u, 0u, 0.0),
          "zero dt must fail");
    check(!numi::cardboard::calculateFreeForceResidualBounds(
              std::span<const nm_float4>(residuals.data(), 5u), nodes, 2u, 3u, 0u, 0.5),
          "truncated residual buffer must fail");
    auto nonfinite = residuals;
    nonfinite[0].y = std::numeric_limits<float>::quiet_NaN();
    check(!numi::cardboard::calculateFreeForceResidualBounds(
              nonfinite, nodes, 2u, 3u, 0u, 0.5),
          "nonfinite free residual must fail");
    std::array<NMFEMNodeStateGPU, 6> allFixed{};
    for (auto& node : allFixed) node.restAndFixed.w = 1.0f;
    check(!numi::cardboard::calculateFreeForceResidualBounds(
              residuals, allFixed, 2u, 3u, 0u, 0.5),
          "no free nodes must fail");
    return 0;
}
