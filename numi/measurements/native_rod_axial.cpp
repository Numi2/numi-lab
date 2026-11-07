#include "metalrobo/DiscreteElasticRod.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 || std::filesystem::exists(argv[1])) {
        std::cerr << "usage: native_rod_axial NEW_OUTPUT.csv\n";
        return 2;
    }
    constexpr double a = 0.04886680223398625;
    constexpr double b = 0.020256155067237984;
    constexpr double dt = 0.001;
    constexpr double speed = 0.0005;
    constexpr double radius = 0.000020;
    constexpr double density = 1100.0;
    const double rest = 0.120 - a / (1000.0 * b);
    const double ea = 1000.0 * b * rest;
    metalrobo::DiscreteRodMaterial material;
    material.radius = radius;
    material.density = density;
    material.youngModulus = ea / (std::numbers::pi * radius * radius);
    material.poissonRatio = 0.35;
    const auto model = metalrobo::makeStraightSutureRod(2, rest, material);
    auto state = metalrobo::makeDiscreteElasticRodDefaultState(model);
    state.positions[1][0] = 0.122;
    state.velocities[1][0] = speed;
    std::array<metalrobo::DiscreteRodAttachment, 2> attachments{};
    attachments[0].nodeIndex = 0;
    attachments[1].nodeIndex = 1;
    attachments[1].targetVelocity = {speed, 0.0, 0.0};
    metalrobo::DiscreteElasticRodStepConfig config;
    config.timestep = dt;
    config.gravity = {0, 0, 0};
    config.linearDamping = 0;
    config.twistDamping = 0;
    config.solverIterations = 128;
    config.constraintTolerance = 1e-10;
    std::ofstream out(argv[1]);
    if (!out) return 2;
    out << std::setprecision(17) << "step,extension_mm,force_N,elastic_J,kinetic_J,work_J,status\n";
    std::array<metalrobo::DiscreteRodAttachmentReaction, 2> reaction{};
    double work = 0, oldForce = 0, oldTarget = 0, initialEnergy = 0;
    double maxForceRelative = 0;
    bool forcePassed = true;
    int accepted = 0;
    for (int i = 0; i <= 4000; ++i) {
        const double x_mm = 2.0 + i * dt * speed * 1000.0;
        const double target = 0.120 + x_mm / 1000.0;
        attachments[1].targetPosition = {target, 0, 0};
        auto d = metalrobo::stepDiscreteElasticRodCpu(model, state, attachments, config, reaction);
        if (!d.succeeded()) {
            out << i << ',' << x_mm << ",,,,,\"" << d.message << "\"\n";
            std::cerr << "step " << i << " failed: " << d.message << "\n";
            return 1;
        }
        metalrobo::DiscreteElasticRodEnergy e;
        auto ed = metalrobo::evaluateDiscreteElasticRodEnergy(model, state, e);
        if (!ed.succeeded()) {
            std::cerr << "energy evaluation failed at " << i << "\n";
            return 1;
        }
        double kinetic = 0;
        for (std::size_t n = 0; n < state.velocities.size(); ++n) {
            const auto& v = state.velocities[n];
            kinetic += 0.5 * model.nodeMasses[n] * (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        }
        const double force = -reaction[1].averageForceOnTarget[0];
        const double expected = a + b * x_mm;
        const double error = std::abs(force-expected)/std::abs(expected);
        maxForceRelative = std::max(maxForceRelative, error);
        forcePassed = forcePassed && std::isfinite(error) && error <= 0.01;
        if (i == 0) initialEnergy = e.total() + kinetic;
        else work += 0.5 * (oldForce + force) * (target - oldTarget);
        oldForce = force;
        oldTarget = target;
        ++accepted;
        out << i << ',' << x_mm << ',' << force << ',' << e.total() << ',' << kinetic << ',' << work << ",ok\n";
        if (i == 4000) {
            const double delta = e.total() + kinetic - initialEnergy;
            const double energyError = std::abs(work-delta)/std::abs(delta);
            std::cout << std::setprecision(17)
                << "accepted=" << accepted << " rest_m=" << rest << " EA_N=" << ea
                << " max_force_relative_error=" << maxForceRelative
                << " work_J=" << work << " energy_change_J=" << delta
                << " work_relative_error=" << energyError
                << " force_gate=" << forcePassed
                << " work_gate=" << (std::isfinite(energyError) && energyError <= 0.01)
                << "\n";
            return forcePassed && std::isfinite(energyError) && energyError <= 0.01 ? 0 : 1;
        }
    }
    return 1;
}
