#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalDiscreteElasticRod.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double length(const std::array<double, 3>& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

std::array<double, 3> midpoint(
    const std::array<double, 3>& first,
    const std::array<double, 3>& second
) {
    return {
        0.5 * (first[0] + second[0]),
        0.5 * (first[1] + second[1]),
        0.5 * (first[2] + second[2]),
    };
}

} // namespace

int main() {
    try {
        constexpr std::size_t environmentCount = 4u;
        metalrobo::DualPsmNeedleThreadWorldConfig worldConfig;
        worldConfig.threadNodeCount = 9u;
        worldConfig.threadLengthM = 0.12;
        metalrobo::HeterogeneousWorld world;
        const auto worldDiagnostics =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    world,
                    worldConfig
                );
        require(
            worldDiagnostics.succeeded(),
            "heterogeneous needle-thread world did not cook"
        );
        const metalrobo::HeterogeneousRodProgram& thread =
            world.rods[0];
        const metalrobo::DiscreteElasticRodModel& model =
            thread.model;
        std::vector<metalrobo::DiscreteElasticRodState> states(
            environmentCount,
            thread.defaultState
        );
        std::vector<metalrobo::DiscreteElasticRodEnergy> before(
            environmentCount
        );
        std::vector<metalrobo::DiscreteRodAttachment> attachments(
            environmentCount
        );
        std::vector<MRBodyStateGPU> rigidBodies(
            environmentCount,
            world.defaultSceneBodies[0]
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            states[environment].positions[4][1] +=
                0.004 + 0.001 * environment;
            states[environment].positions.back()[0] +=
                0.002 + 0.0005 * environment;
            states[environment].twists.back() =
                0.25 + 0.05 * environment;
            const auto energyDiagnostics =
                metalrobo::evaluateDiscreteElasticRodEnergy(
                    model,
                    states[environment],
                    before[environment]
                );
            require(
                energyDiagnostics.succeeded() &&
                    before[environment].total() > 0.0,
                "initial rod energy is invalid"
            );
            attachments[environment] =
                thread.attachments[0];
        }
        const std::array<
            metalrobo::DiscreteRodRigidAttachmentBinding,
            1
        > rigidBindings{{
            thread.rigidBindings[0],
        }};

        metalrobo::MetalDiscreteElasticRodConfig config;
        config.step = thread.stepConfig;
        config.step.gravity = {0.0, 0.0, 0.0};
        config.step.solverIterations = 256u;
        config.step.constraintTolerance = 1.0e-5;
        const metalrobo::MetalDiscreteElasticRodInput input{
            .states = states,
            .attachmentCount = 1u,
            .attachments = attachments,
            .rigidBodyCount = 1u,
            .rigidBodies = rigidBodies,
            .rigidBindings = rigidBindings,
        };
        metalrobo::MetalDiscreteElasticRodResult result;
        const auto diagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                model,
                input,
                result,
                config
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"Metal rod solve failed: "} +
                diagnostics.message +
                " gpu_code=" +
                std::to_string(diagnostics.firstGPUStatusCode)
            );
        }
        require(
            diagnostics.dispatched &&
                diagnostics.published &&
                result.states.size() == environmentCount &&
                result.statuses.size() == environmentCount &&
                result.reactions.size() == environmentCount &&
                result.rigidBodies.size() == environmentCount,
            "Metal rod result was not published"
        );
        double maximumAfter = 0.0;
        double maximumReaction = 0.0;
        std::uint32_t maximumIterations = 0u;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            metalrobo::DiscreteElasticRodEnergy after;
            const auto energyDiagnostics =
                metalrobo::evaluateDiscreteElasticRodEnergy(
                    model,
                    result.states[environment],
                    after
                );
            require(
                energyDiagnostics.succeeded() &&
                    after.total() < before[environment].total(),
                "Metal rod projection did not reduce energy"
            );
            require(
                std::abs(
                    result.states[environment].
                        positions.front()[0] -
                    attachments[environment].targetPosition[0]
                ) <= 1.0e-7 &&
                    std::abs(
                        result.states[environment].
                            positions.front()[1] -
                        attachments[environment].
                            targetPosition[1]
                    ) <= 1.0e-7 &&
                    std::abs(
                        result.states[environment].
                            positions.front()[2] -
                        attachments[environment].
                            targetPosition[2]
                    ) <= 1.0e-7 &&
                    result.states[environment].
                            velocities.front() ==
                        attachments[environment].targetVelocity,
                "hard rod attachment was not enforced"
            );
            const auto& reaction = result.reactions[environment];
            maximumReaction = std::max(
                maximumReaction,
                length(reaction.averageForceOnTarget)
            );
            const MRBodyStateGPU& body =
                result.rigidBodies[environment];
            const double inverseMass =
                rigidBodies[environment].
                    linearVelocityAndInverseMass.w;
            require(
                reaction.nodeIndex == 0u &&
                    reaction.bodyIndex == 0u &&
                    reaction.finalPositionError <= 1.0e-5 &&
                    length(reaction.impulseOnTarget) > 0.0 &&
                    std::abs(
                        body.linearVelocityAndInverseMass.x -
                        inverseMass *
                            reaction.impulseOnTarget[0]
                    ) <= 2.0e-5 &&
                    std::abs(
                        body.linearVelocityAndInverseMass.y -
                        inverseMass *
                            reaction.impulseOnTarget[1]
                    ) <= 2.0e-5 &&
                    std::abs(
                        body.linearVelocityAndInverseMass.z -
                        inverseMass *
                            reaction.impulseOnTarget[2]
                    ) <= 2.0e-5,
                "needle did not receive the on-GPU rod reaction"
            );
            maximumAfter = std::max(
                maximumAfter,
                after.total()
            );
            maximumIterations = std::max(
                maximumIterations,
                result.statuses[environment].iterations
            );
        }

        metalrobo::MetalDiscreteElasticRodResult replay;
        const auto replayDiagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                model,
                input,
                replay,
                config
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.states.size() == result.states.size(),
            "Metal rod replay failed"
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            require(
                replay.states[environment].positions ==
                        result.states[environment].positions &&
                    replay.states[environment].velocities ==
                        result.states[environment].velocities &&
                    replay.states[environment].twists ==
                        result.states[environment].twists &&
                    replay.states[environment].twistRates ==
                        result.states[environment].twistRates &&
                    replay.reactions[environment].
                            impulseOnTarget ==
                        result.reactions[environment].
                            impulseOnTarget &&
                    replay.rigidBodies[environment].
                            linearVelocityAndInverseMass.x ==
                        result.rigidBodies[environment].
                            linearVelocityAndInverseMass.x &&
                    replay.rigidBodies[environment].
                            linearVelocityAndInverseMass.y ==
                        result.rigidBodies[environment].
                            linearVelocityAndInverseMass.y &&
                    replay.rigidBodies[environment].
                            linearVelocityAndInverseMass.z ==
                        result.rigidBodies[environment].
                            linearVelocityAndInverseMass.z,
                "Metal rod replay is not deterministic"
            );
        }

        metalrobo::DiscreteElasticRodModel collisionModel =
            metalrobo::makeStraightSutureRod(4u, 0.12);
        collisionModel.stretchStiffness.assign(
            collisionModel.stretchStiffness.size(),
            1.0e-6
        );
        collisionModel.bendStiffness.assign(
            collisionModel.bendStiffness.size(),
            1.0e-12
        );
        collisionModel.twistStiffness.assign(
            collisionModel.twistStiffness.size(),
            1.0e-12
        );
        metalrobo::DiscreteElasticRodState crossing =
            metalrobo::makeDiscreteElasticRodDefaultState(
                collisionModel
            );
        crossing.positions = {{
            {-0.02, 0.00, 0.0},
            { 0.02, 0.00, 0.0},
            { 0.00, -0.02, 0.0},
            { 0.00, 0.02, 0.0},
        }};
        std::vector<metalrobo::DiscreteElasticRodState>
            crossingStates(environmentCount, crossing);
        metalrobo::MetalDiscreteElasticRodConfig
            collisionConfig;
        collisionConfig.step.gravity = {0.0, 0.0, 0.0};
        collisionConfig.step.solverIterations = 256u;
        collisionConfig.step.constraintTolerance = 1.0e-6;
        collisionConfig.step.enableSelfCollision = true;
        metalrobo::MetalDiscreteElasticRodResult
            collisionResult;
        const auto collisionDiagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                collisionModel,
                {
                    .states = crossingStates,
                },
                collisionResult,
                collisionConfig
            );
        require(
            collisionDiagnostics.succeeded() &&
                collisionResult.states.size() ==
                    environmentCount,
            "Metal DER self-contact solve failed: " +
                collisionDiagnostics.message
        );
        auto cpuCrossing = crossing;
        const auto cpuCollisionDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                cpuCrossing,
                {},
                collisionConfig.step
            );
        require(
            cpuCollisionDiagnostics.succeeded(),
            "CPU DER self-contact oracle failed"
        );
        double maximumSelfContactError = 0.0;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            const auto& metalState =
                collisionResult.states[environment];
            const auto firstMidpoint = midpoint(
                metalState.positions[0],
                metalState.positions[1]
            );
            const auto secondMidpoint = midpoint(
                metalState.positions[2],
                metalState.positions[3]
            );
            const double separation = length({
                firstMidpoint[0] - secondMidpoint[0],
                firstMidpoint[1] - secondMidpoint[1],
                firstMidpoint[2] - secondMidpoint[2],
            });
            require(
                collisionResult.statuses[environment].
                        diagnostics.w > 0.0F &&
                    collisionResult.statuses[environment].
                        diagnostics.z >=
                            1.9F *
                                static_cast<float>(
                                    collisionModel.radius
                                ) &&
                    separation >=
                        0.95 * 2.0 * collisionModel.radius,
                "Metal DER capsule self-contact did not separate edges"
            );
            for (std::size_t node = 0u;
                 node < metalState.positions.size();
                 ++node) {
                maximumSelfContactError = std::max(
                    maximumSelfContactError,
                    std::abs(
                        metalState.positions[node][2] -
                        cpuCrossing.positions[node][2]
                    )
                );
            }
        }
        require(
            maximumSelfContactError <= 2.0e-6,
            "Metal DER self-contact normal response diverged "
            "from FP64 oracle"
        );

        auto invalidStates = states;
        invalidStates[2].positions[3][0] =
            std::numeric_limits<double>::quiet_NaN();
        metalrobo::MetalDiscreteElasticRodResult sentinel = result;
        const double sentinelValue =
            sentinel.states[0].positions[1][0];
        const metalrobo::MetalDiscreteElasticRodInput invalidInput{
            .states = invalidStates,
            .attachmentCount = 1u,
            .attachments = attachments,
            .rigidBodyCount = 1u,
            .rigidBodies = rigidBodies,
            .rigidBindings = rigidBindings,
        };
        const auto rejected =
            metalrobo::runMetalDiscreteElasticRod(
                model,
                invalidInput,
                sentinel,
                config
            );
        require(
            !rejected.succeeded() &&
                !rejected.dispatched &&
                sentinel.states[0].positions[1][0] ==
                    sentinelValue,
            "invalid Metal rod input changed published state"
        );

        std::cout
            << "metal_discrete_elastic_rod=ok"
            << " device=\"" << diagnostics.deviceName << "\""
            << " environments=" << environmentCount
            << " nodes=" << model.restPositions.size()
            << " iterations=" << maximumIterations
            << " max_after_energy=" << maximumAfter
            << " max_anchor_force_n=" << maximumReaction
            << " self_contacts="
            << collisionResult.statuses[0].diagnostics.w
            << " self_penetration="
            << collisionResult.statuses[0].diagnostics.z
            << " self_oracle_error="
            << maximumSelfContactError
            << " allocated_bytes=" << diagnostics.allocatedBytes
            << " elapsed_ms=" << diagnostics.elapsedMilliseconds
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "metal_discrete_elastic_rod=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
