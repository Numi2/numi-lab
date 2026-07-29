#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalMultiArticulatedConstraints.hpp"
#include "metalrobo/MultiArticulatedWorld.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double norm(const std::array<double, 3>& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

} // namespace

int main() {
    try {
        metalrobo::DualPsmNeedleThreadWorldConfig config;
        config.threadNodeCount = 9u;
        config.threadLengthM = 0.12;
        metalrobo::HeterogeneousWorld world;
        const auto diagnostics =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    world,
                    config
                );
        require(
            diagnostics.succeeded(),
            "heterogeneous surgical cook failed: " +
                diagnostics.message
        );
        std::string reason;
        require(
            world.valid(&reason),
            "heterogeneous surgical world is invalid: " + reason
        );
        require(
            world.model.articulations.size() == 2u &&
                world.model.bodies.size() == 19u &&
                world.model.shapes.size() == 72u &&
                world.model.constraintProgram.blocks.size() == 14u &&
                world.sceneBodyIndices ==
                    std::vector<std::uint32_t>{18u} &&
                world.defaultSceneBodies.size() == 1u &&
                world.rods.size() == 1u &&
                world.rods[0].rigidBindings[0].bodyIndex == 0u,
            "heterogeneous owned streams changed"
        );

        metalrobo::CompiledMetalMultiArticulatedProgram program;
        const auto programDiagnostics =
            metalrobo::compileMetalMultiArticulatedProgram(
                world.model,
                program
            );
        require(
            programDiagnostics.succeeded() &&
                program.valid() &&
                program.rowCount() == 14u,
            "heterogeneous multi-articulation program did not compile"
        );

        std::vector<double> q(
            world.model.defaultQ.begin(),
            world.model.defaultQ.end()
        );
        std::vector<double> v(
            world.model.defaultV.begin(),
            world.model.defaultV.end()
        );
        std::vector<double> force(world.model.world.nv, 0.0);
        metalrobo::MultiArticulationFactorCache cache;
        metalrobo::MultiArticulatedWorldConfig stepConfig;
        stepConfig.dynamics.timestep =
            world.model.world.gravityAndTimestep.w;
        stepConfig.solverIterations = 128u;
        stepConfig.solverTolerance = 1.0e-8;
        stepConfig.constraintResidual.residualTolerance =
            1.0e-7;
        const auto step =
            metalrobo::stepMultiArticulatedWorldCpu(
                world.model,
                q,
                v,
                force,
                {},
                cache,
                stepConfig
            );
        require(
            step.succeeded(),
            "heterogeneous articulation program did not execute"
        );

        metalrobo::HeterogeneousRodProgram rod = world.rods[0];
        rod.defaultState.positions[4][1] += 0.005;
        metalrobo::DiscreteElasticRodStepConfig rodConfig;
        rodConfig.gravity = {0.0, 0.0, 0.0};
        rodConfig.solverIterations = 256u;
        rodConfig.constraintTolerance = 1.0e-5;
        std::array<
            metalrobo::DiscreteRodAttachmentReaction,
            1
        > reactions{};
        const auto rodStep =
            metalrobo::stepDiscreteElasticRodCpu(
                rod.model,
                rod.defaultState,
                rod.attachments,
                rodConfig,
                reactions
            );
        require(
            rodStep.succeeded(),
            "heterogeneous rod sidecar failed: " +
                rodStep.message
        );
        require(
            reactions[0].bodyIndex ==
                metalrobo::kDiscreteRodNoRigidBody &&
                norm(reactions[0].averageForceOnTarget) > 0.0,
            "heterogeneous rod reaction evidence is empty"
        );

        metalrobo::HeterogeneousWorld replay;
        const auto replayDiagnostics =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    replay,
                    config
                );
        require(
            replayDiagnostics.succeeded() &&
                replay.fingerprint == world.fingerprint &&
                replay.sceneBodyIndices == world.sceneBodyIndices,
            "heterogeneous cook replay changed"
        );

        metalrobo::HeterogeneousWorld sentinel = world;
        const std::uint64_t sentinelFingerprint =
            sentinel.fingerprint;
        auto rejectedConfig = config;
        rejectedConfig.threadNodeCount = 1u;
        const auto rejected =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    sentinel,
                    rejectedConfig
                );
        require(
            !rejected.succeeded() &&
                sentinel.fingerprint == sentinelFingerprint &&
                sentinel.valid(),
            "failed heterogeneous cook published partial state"
        );

        std::cout
            << "heterogeneous_world=ok"
            << " components="
            << world.componentInstanceIds.size()
            << " articulations="
            << world.model.articulations.size()
            << " scene_bodies="
            << world.defaultSceneBodies.size()
            << " colliders=" << world.model.shapes.size()
            << " constraint_rows=" << program.rowCount()
            << " rods=" << world.rods.size()
            << " fingerprint=" << world.fingerprint
            << " swage_force_n="
            << norm(reactions[0].averageForceOnTarget)
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "heterogeneous_world=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
