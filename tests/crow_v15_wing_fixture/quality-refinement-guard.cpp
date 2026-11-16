#include "metalrobo/MetalUnifiedQualitySolver.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

void runCase(bool hard, float bias, bool expectSuccess) {
    MRUnifiedQualityBlockGPU block{};
    block.layout = {0u, 1u, MR_UNIFIED_QUALITY_SCALAR_INTERVAL,
        hard ? MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY : 0u};
    block.stableKey = {501u, 0u, 0u, 0u};
    block.scale0 = block.scale1 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.regularization0 = {0.02f, 1.0f, 1.0f, 1.0f};
    block.regularization1 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.boundsAndShift = {-std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), 0.0f, 0.0f};
    const bool mixedGuard = hard && !expectSuccess;
    auto disabled = block;
    disabled.layout = {1u, 1u, MR_UNIFIED_QUALITY_SCALAR_INTERVAL, 0u};
    disabled.boundsAndShift = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::array blocks{block, disabled};
    const std::array mass{1.0f};
    const std::array jacobian{1.0f, 0.0f}, biases{bias, 0.0f}, zero{0.0f, 0.0f};
    metalrobo::MetalUnifiedQualityProblem problem;
    problem.problemCount = 1u;
    problem.generalizedVelocityCount = 1u;
    problem.rowCount = mixedGuard ? 2u : 1u;
    problem.blocks = std::span(blocks.data(), mixedGuard ? 2u : 1u);
    problem.dynamics = mass;
    problem.jacobian = std::span(jacobian.data(), mixedGuard ? 2u : 1u);
    problem.bias = std::span(biases.data(), mixedGuard ? 2u : 1u);
    problem.freeVelocity = std::span(zero.data(), 1u);
    problem.warmVelocity = std::span(zero.data(), 1u);
    problem.warmImpulses = std::span(zero.data(), mixedGuard ? 2u : 1u);
    metalrobo::MetalUnifiedQualityConfig config;
    config.maximumNewtonIterations = 40u;
    config.optimalityTolerance = 1.0e-5f;
    config.feasibilityTolerance = 2.0e-5f;
    config.directMaximumGeneralizedVelocities = 64u;
    metalrobo::MetalUnifiedQualityResult result;
    result.velocity = {123.0f};
    result.impulses = {456.0f};
    const auto diagnostics = metalrobo::solveMetalUnifiedQuality(problem, result, config);
    if (!expectSuccess) {
        check(!diagnostics.succeeded() && diagnostics.dispatched && !diagnostics.published &&
              diagnostics.firstGPUStatusCode == MR_UNIFIED_QUALITY_DID_NOT_CONVERGE,
              "refinement accepted an inaccurate hard equality");
        const auto equalityPosition = diagnostics.message.find("hard_equality=");
        check(equalityPosition != std::string::npos &&
              std::stof(diagnostics.message.substr(equalityPosition + 14u)) >
                  config.feasibilityTolerance,
              "hard-equality failure omitted its violation");
        check(result.velocity == std::vector<float>{123.0f} &&
              result.impulses == std::vector<float>{456.0f} && result.statuses.empty(),
              "failed hard equality changed the output");
        std::cout << "hard=" << hard << " bias=" << bias << " rejected="
                  << diagnostics.firstGPUStatusCode << " transactional=1\n";
    } else {
        check(diagnostics.succeeded() && result.statuses.size() == 1u,
              "valid control failed native dispatch");
        const auto& status = result.statuses[0];
        check(status.code == MR_UNIFIED_QUALITY_SUCCESS,
              "valid control was rejected");
        check(result.velocity.size() == 1u && result.impulses.size() == 1u,
              "valid control returned the wrong dimensions");
        const double expected = hard ? -double(bias) : -double(bias) / 1.02;
        check(std::abs(double(result.velocity[0]) - expected) < 2.0e-5,
              "valid control disagrees with analytic regularized solution");
        if (hard) {
            check(status.certificates0.z <= config.feasibilityTolerance,
                  "accepted hard equality exceeds tolerance");
            check(status.solvePath == MR_UNIFIED_QUALITY_PATH_EXACT_EQUALITY,
                  "hard control did not use exact equality path");
            check(std::abs(double(result.impulses[0]) - expected) < 2.0e-5,
                  "hard control did not report the matching reaction");
        }
        std::cout << "hard=" << hard << " bias=" << bias << " status=" << status.code
                  << " path=" << status.solvePath << " equality=" << status.certificates0.z << '\n';
    }
}

void runSingularCase(bool inconsistent) {
    std::array<MRUnifiedQualityBlockGPU, 2> blocks{};
    for (std::uint32_t i = 0; i < blocks.size(); ++i) {
        auto& b = blocks[i];
        b.layout = {i, 1u, MR_UNIFIED_QUALITY_SCALAR_INTERVAL,
                    MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY};
        b.scale0 = b.scale1 = {1, 1, 1, 1};
        b.regularization0 = b.regularization1 = {0.02f, 1, 1, 1};
        b.boundsAndShift = {-std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(), 0, 0};
    }
    const std::array A{1.0f, 0.0f, 0.0f, 1.0f}, J{1.0f, 0.0f, 1.0f, 0.0f};
    const std::array bias{0.1f, inconsistent ? 0.2f : 0.1f}, zero{0.0f, 0.0f};
    metalrobo::MetalUnifiedQualityProblem problem;
    problem.problemCount = 1; problem.generalizedVelocityCount = 2; problem.rowCount = 2;
    problem.blocks = blocks; problem.dynamics = A; problem.jacobian = J; problem.bias = bias;
    problem.freeVelocity = zero; problem.warmVelocity = zero; problem.warmImpulses = zero;
    metalrobo::MetalUnifiedQualityConfig config;
    config.maximumNewtonIterations = 40;
    metalrobo::MetalUnifiedQualityResult result;
    result.velocity = {123.0f}; result.impulses = {456.0f};
    const auto d = metalrobo::solveMetalUnifiedQuality(problem, result, config);
    check(!d.succeeded() && d.dispatched && !d.published &&
          d.firstGPUStatusCode != MR_UNIFIED_QUALITY_SUCCESS,
          "dependent equality rows were incorrectly published");
    check(result.velocity == std::vector<float>{123.0f} &&
          result.impulses == std::vector<float>{456.0f} && result.statuses.empty(),
          "singular equality failure changed output");
    std::cout << "singular=1 inconsistent=" << inconsistent
              << " rejected=" << d.firstGPUStatusCode << " transactional=1\n";
}
}

int main() {
    try {
        runCase(true, 0.08f, false);
        runCase(true, 0.08f, true);
        runCase(false, 0.08f, true);
        runCase(true, 0.0f, true);
        runSingularCase(false);
        runSingularCase(true);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "quality_refinement_guard=failed reason=" << error.what() << '\n';
        return 1;
    }
}
