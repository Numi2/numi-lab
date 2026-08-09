#include "metalrobo/MeasuredSurfaceRobot.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace metalrobo;
    MeasuredSurfaceRobotPack pack;
    pack.id = "deetjen-f03-surface-robot-v1";
    pack.datasetIdentifier = "deetjen-ob-2018-12-11-f03-complete-surface-v1";
    pack.manifestSHA256 = "ad42148aa9ee72d994d668ba16f8b6572cb8b192b77539fe66d97586ed9e1a13";
    pack.positionsSHA256 = "690b6dd2a24d593a512d799b7fe5f3f756ca7ae3ce1cd1cdc4bb12b2531567a6";
    pack.trianglesSHA256 = "9d832ff22ecedc15e47c454378146a1006ae7f6974512ce222994e2f12f43d61";
    pack.frameCount = 144u;
    pack.vertexCount = 2157u;
    pack.triangleCount = 3968u;
    pack.sampleRateHertz = 1000.0f;
    pack.actions = makeMeasuredSurfaceFlightActions();
    pack.components = {
        {MeasuredSurfaceComponent::body, 0u, 1443u, 0u, 2736u},
        {MeasuredSurfaceComponent::leftWing, 1443u, 297u, 2736u, 512u},
        {MeasuredSurfaceComponent::rightWing, 1740u, 297u, 3248u, 512u},
        {MeasuredSurfaceComponent::tail, 2037u, 120u, 3760u, 208u},
    };
    const auto robot = compileMeasuredSurfaceRobot(pack);
    if (robot.vertexComponents.size() != pack.vertexCount ||
        robot.triangleComponents.size() != pack.triangleCount || robot.fingerprint == 0u) {
        throw std::runtime_error("compiled measured-surface tables are incomplete");
    }
    MeasuredSurfaceActuatorState zeroState;
    const std::array<float, kMeasuredSurfaceActionCount> zeroTargets {};
    stepMeasuredSurfaceActuators(robot, zeroTargets, 0.001f, zeroState);
    for (std::uint32_t i = 0u; i < kMeasuredSurfaceActionCount; ++i) {
        if (zeroState.position[i] != 0.0f || zeroState.velocity[i] != 0.0f) {
            throw std::runtime_error("zero-action invariant failed");
        }
    }
    auto targets = zeroTargets;
    targets[5] = 0.8f;
    targets[13] = -0.8f;
    targets[20] = 0.35f;
    for (int step = 0; step < 250; ++step) {
        stepMeasuredSurfaceActuators(robot, targets, 0.001f, zeroState);
    }
    if (!(zeroState.position[5] > 0.7f && zeroState.position[13] < -0.7f &&
          zeroState.position[20] > 0.3f)) {
        throw std::runtime_error("bounded surface actuators did not track their targets");
    }
    const auto before = zeroState;
    auto invalid = targets;
    invalid[0] = std::nanf("");
    bool rejected = false;
    try {
        stepMeasuredSurfaceActuators(robot, invalid, 0.001f, zeroState);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected || zeroState.position != before.position || zeroState.velocity != before.velocity) {
        throw std::runtime_error("transactional actuator rollback failed");
    }
    std::cout << "MeasuredSurfaceRobotPack probe passed\n"
              << "  fingerprint: " << robot.fingerprint << '\n'
              << "  actions: " << kMeasuredSurfaceActionCount << '\n'
              << "  vertices: " << robot.vertexComponents.size() << '\n'
              << "  triangles: " << robot.triangleComponents.size() << '\n'
              << "  zero-action exact: true\n"
              << "  nonfinite rollback: true\n";
}
