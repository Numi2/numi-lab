#include "numi/matter/numi_human.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    numi::matter::CompiledWorld world;
    world.dispatch.environmentCount = 2u;
    world.dispatch.parameterCount = 4u;
    world.materials.resize(2u);
    world.materials[0u].parameterOffset = 0u;
    world.materials[0u].parameterCount = 2u;
    world.materials[1u].parameterOffset = 2u;
    world.materials[1u].parameterCount = 2u;
    world.parameters.resize(4u);
    for (auto& parameter : world.parameters)
        parameter.valueAndBounds = {1.0f, 1.0f, 1.2f, 0.0f};

    numi::matter::RuntimeStateSnapshot snapshot;
    snapshot.available = true;
    snapshot.deviceProgramFingerprint = 1u;
    snapshot.environmentParameters = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<numi::matter::NumiHumanFEMPrestressTarget> targets{
        {.materialIndex = 0u, .localParameterIndex = 1u,
         .neutralValue = 1.0f, .sourceValue = 1.034f},
        {.materialIndex = 1u, .localParameterIndex = 0u,
         .neutralValue = 1.0f, .sourceValue = 1.016f},
    };
    const auto staged = numi::matter::prepareNumiHumanFEMPrestressStage(
        world, targets, 0.5f, snapshot);
    require(staged.succeeded() && staged.appliedParameterCount == 4u,
            "valid prestress stage was rejected");
    const float first = 1.0f + 0.5f * (1.034f - 1.0f);
    const float second = 1.0f + 0.5f * (1.016f - 1.0f);
    require(snapshot.environmentParameters[1u] == first &&
                snapshot.environmentParameters[2u] == second &&
                snapshot.environmentParameters[5u] == first &&
                snapshot.environmentParameters[6u] == second,
            "prestress stage did not update every environment exactly");
    require(snapshot.environmentParameters[0u] == 1.0f &&
                snapshot.environmentParameters[3u] == 1.0f &&
                snapshot.environmentParameters[4u] == 1.0f &&
                snapshot.environmentParameters[7u] == 1.0f,
            "prestress stage mutated an unrelated parameter");

    const auto beforeRejected = snapshot.environmentParameters;
    const std::vector<numi::matter::NumiHumanFEMPrestressTarget> duplicate{
        targets[0u], targets[0u],
    };
    const auto rejectedDuplicate =
        numi::matter::prepareNumiHumanFEMPrestressStage(
            world, duplicate, 1.0f, snapshot);
    require(!rejectedDuplicate.succeeded() &&
                rejectedDuplicate.status ==
                    numi::matter::NumiHumanFEMPrestressStatus::duplicateParameter &&
                snapshot.environmentParameters == beforeRejected,
            "duplicate prestress target did not fail atomically");

    auto outOfBounds = targets;
    outOfBounds[0u].sourceValue = 1.3f;
    const auto rejectedBounds =
        numi::matter::prepareNumiHumanFEMPrestressStage(
            world, outOfBounds, 1.0f, snapshot);
    require(!rejectedBounds.succeeded() &&
                rejectedBounds.status ==
                    numi::matter::NumiHumanFEMPrestressStatus::invalidBounds &&
                snapshot.environmentParameters == beforeRejected,
            "out-of-bounds prestress target did not fail atomically");

    NMNumiHumanPassiveLigamentGPU ligament{};
    ligament.firstBodyIndex = 1u;
    ligament.secondBodyIndex = 2u;
    ligament.flags = NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE;
    ligament.material = {2.0e6f, 10.0f, 100.0e6f, 1.05f};
    ligament.reference = {0.03f, 1.0e-4f, 1.02f, 0.0f};
    numi::matter::NumiHumanPassiveLigamentFiberEvaluation fiber;
    require(numi::matter::evaluateNumiHumanPassiveLigamentFiber(
                ligament, 0.99 * 0.03 / 1.02, fiber) &&
                std::abs(fiber.effectiveStretch - 0.99) < 2.0e-7 &&
                fiber.fiberStressPascals == 0.0 &&
                fiber.tensionNewtons == 0.0,
            "passive ligament slack branch drifted");
    require(numi::matter::evaluateNumiHumanPassiveLigamentFiber(
                ligament, 0.03, fiber),
            "valid passive ligament source law was rejected");
    const double expectedStress = 2.0e6 * std::expm1(0.2);
    require(std::abs(fiber.effectiveStretch - 1.02) < 2.0e-7 &&
                std::abs(fiber.fiberStressPascals - expectedStress) /
                    expectedStress < 2.0e-6 &&
                std::abs(fiber.tensionNewtons - expectedStress * 1.0e-4) /
                    (expectedStress * 1.0e-4) < 2.0e-6,
            "passive ligament exponential branch drifted");
    const double linearLength = 0.03 * 1.06 / 1.02;
    require(numi::matter::evaluateNumiHumanPassiveLigamentFiber(
                ligament, linearLength, fiber),
            "passive ligament linear branch was rejected");
    const double c6 = 2.0e6 * std::expm1(0.5) - 100.0e6 * 1.05;
    const double expectedLinearStress = 100.0e6 * 1.06 + c6;
    require(std::abs(fiber.fiberStressPascals - expectedLinearStress) /
                    expectedLinearStress < 2.0e-5,
            "passive ligament linear branch drifted");
    const double transitionLength = 0.03 * 1.05 / 1.02;
    require(numi::matter::evaluateNumiHumanPassiveLigamentFiber(
                ligament, transitionLength, fiber),
            "passive ligament transition was rejected");
    const double expectedTransitionStress = 2.0e6 * std::expm1(0.5);
    require(std::abs(fiber.fiberStressPascals - expectedTransitionStress) /
                    expectedTransitionStress < 2.0e-5,
            "passive ligament source law lost stress continuity");

    std::cout
        << "numi_human_fem_prestress_stage=passed"
        << " environments=2 targets=2 atomic_rejection=verified"
        << " passive_ligament_fiber_branches=slack_exponential_linear"
        << " transition_continuity=verified\n";
    return 0;
}
