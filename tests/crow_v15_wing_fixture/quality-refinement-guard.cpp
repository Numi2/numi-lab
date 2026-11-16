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
    const std::array blocks{block};
    const std::array mass{1.0f}, jacobian{1.0f}, biases{bias}, zero{0.0f};
    metalrobo::MetalUnifiedQualityProblem problem;
    problem.problemCount = 1u;
    problem.generalizedVelocityCount = 1u;
    problem.rowCount = 1u;
    problem.blocks = blocks;
    problem.dynamics = mass;
    problem.jacobian = jacobian;
    problem.bias = biases;
    problem.freeVelocity = zero;
    problem.warmVelocity = zero;
    problem.warmImpulses = zero;
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
        const double expected = -double(bias) / 1.02;
        check(std::abs(double(result.velocity[0]) - expected) < 2.0e-5,
              "valid control disagrees with analytic regularized solution");
        if (hard) {
            check(status.certificates0.z <= config.feasibilityTolerance,
                  "accepted hard equality exceeds tolerance");
        }
        std::cout << "hard=" << hard << " bias=" << bias << " status=" << status.code
                  << " path=" << status.solvePath << " equality=" << status.certificates0.z << '\n';
    }
}
}

int main() {
    try {
        runCase(true, 0.08f, false);
        runCase(false, 0.08f, true);
        runCase(true, 0.0f, true);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "quality_refinement_guard=failed reason=" << error.what() << '\n';
        return 1;
    }
}
