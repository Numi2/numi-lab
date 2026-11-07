#include "metalrobo/DiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

int main() {
    constexpr double radius = 0.000020, rest = 0.11758755785232791;
    constexpr double ea = 2.3818718058345718, length = 0.122;
    metalrobo::DiscreteRodMaterial mat;
    mat.radius = radius;
    mat.density = 1100;
    mat.youngModulus = ea / (std::numbers::pi * radius * radius);
    auto model = metalrobo::makeStraightSutureRod(2, rest, mat);
    auto state = metalrobo::makeDiscreteElasticRodDefaultState(model);
    state.positions[1][0] = length;
    auto before = state;
    std::array<metalrobo::DiscreteRodAttachment, 2> anchors{};
    anchors[0].nodeIndex = 0;
    anchors[1].nodeIndex = 1;
    anchors[1].targetPosition = {length, 0, 0};
    std::array<metalrobo::DiscreteRodAttachmentReaction, 2> reactions{};
    reactions[0].impulseOnTarget[0] = 41;
    reactions[1].impulseOnTarget[0] = 42;
    metalrobo::DiscreteElasticRodStepConfig cfg;
    cfg.timestep = 0.001;
    cfg.gravity = {0,0,0};
    cfg.linearDamping = 0;
    cfg.twistDamping = 0;
    cfg.solverIterations = 128;
    cfg.constraintTolerance = 1e-10;
    const auto rejected = metalrobo::stepDiscreteElasticRodCpu(model,state,anchors,cfg,reactions);
    if (rejected.status != metalrobo::DiscreteElasticRodStatus::didNotConverge ||
        state.positions != before.positions || state.velocities != before.velocities ||
        state.twists != before.twists || state.twistRates != before.twistRates ||
        reactions[0].impulseOnTarget[0] != 41 || reactions[1].impulseOnTarget[0] != 42) {
        std::cerr << "uncertified force was not rejected transactionally\n"; return 1;
    }
    cfg.solverIterations = 12000;
    const auto resolved = metalrobo::stepDiscreteElasticRodCpu(model,state,anchors,cfg,reactions);
    const double expected = ea/rest*(length-rest);
    const double error = std::abs(-reactions[1].averageForceOnTarget[0]-expected)/expected;
    if (!resolved.succeeded() || !std::isfinite(error) || error > .01) {
        std::cerr << "resolved pinned reaction mismatch " << error << "\n"; return 1;
    }
    auto restState = metalrobo::makeDiscreteElasticRodDefaultState(model);
    anchors[1].targetPosition = {rest,0,0};
    cfg.solverIterations = 128;
    const auto normal = metalrobo::stepDiscreteElasticRodCpu(model,restState,anchors,cfg,reactions);
    if (!normal.succeeded() || std::abs(reactions[1].averageForceOnTarget[0]) > 1e-8) {
        std::cerr << "force-free pinned motion failed\n"; return 1;
    }
    std::cout << "rejected iterations=" << rejected.iterations
              << " resolved iterations=" << resolved.iterations
              << " force_relative_error=" << error
              << " rest_iterations=" << normal.iterations << "\n";
}
