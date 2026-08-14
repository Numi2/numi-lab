#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalDiscreteElasticRod.hpp"

#include <algorithm>
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

double dot(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

std::array<double, 3> subtract(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

double segmentDistance(
    const std::array<double, 3>& firstA,
    const std::array<double, 3>& firstB,
    const std::array<double, 3>& secondA,
    const std::array<double, 3>& secondB
) {
    const auto first = subtract(firstB, firstA);
    const auto second = subtract(secondB, secondA);
    const auto offset = subtract(firstA, secondA);
    const double aa = dot(first, first);
    const double bb = dot(second, second);
    const double ab = dot(first, second);
    const double ao = dot(first, offset);
    const double bo = dot(second, offset);
    const double denominator = aa * bb - ab * ab;
    double firstParameter = denominator >
            1.0e-14 * aa * bb
        ? std::clamp((ab * bo - ao * bb) / denominator, 0.0, 1.0)
        : 0.0;
    double secondNumerator = ab * firstParameter + bo;
    double secondParameter = 0.0;
    if (secondNumerator < 0.0) {
        firstParameter = std::clamp(-ao / aa, 0.0, 1.0);
    } else if (secondNumerator > bb) {
        secondParameter = 1.0;
        firstParameter = std::clamp((ab - ao) / aa, 0.0, 1.0);
    } else {
        secondParameter = secondNumerator / bb;
    }
    return length({
        firstA[0] + firstParameter * first[0] -
            secondA[0] - secondParameter * second[0],
        firstA[1] + firstParameter * first[1] -
            secondA[1] - secondParameter * second[1],
        firstA[2] + firstParameter * first[2] -
            secondA[2] - secondParameter * second[2],
    });
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
        // Match the shader's documented FP32 central-difference floor so the
        // comparison isolates solver semantics rather than FP64-vs-FP32
        // perturbation scale.
        collisionConfig.step.derivativeStep = 3.5e-4;
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
        const double cpuClearance = segmentDistance(
            cpuCrossing.positions[0],
            cpuCrossing.positions[1],
            cpuCrossing.positions[2],
            cpuCrossing.positions[3]
        );
        double maximumSelfContactError = 0.0;
        double firstMetalClearance = 0.0;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            const auto& metalState =
                collisionResult.states[environment];
            const double clearance = segmentDistance(
                metalState.positions[0],
                metalState.positions[1],
                metalState.positions[2],
                metalState.positions[3]
            );
            if (environment == 0u) {
                firstMetalClearance = clearance;
            }
            require(
                collisionResult.statuses[environment].
                        diagnostics.w > 0.0F &&
                    collisionResult.statuses[environment].
                        diagnostics.z >=
                            1.9F *
                                static_cast<float>(
                                    collisionModel.radius
                                ) &&
                    clearance >=
                        0.95 * 2.0 * collisionModel.radius,
                "Metal DER capsule self-contact did not separate edges"
            );
            // The exact zero-distance crossing has two sign-equivalent
            // separating normals. Compare the physical gap magnitude rather
            // than a coordinate whose sign is not an invariant.
            maximumSelfContactError = std::max(
                maximumSelfContactError,
                std::abs(clearance - cpuClearance)
            );
        }
        require(
            maximumSelfContactError <= 2.0e-6,
            "Metal DER self-contact separation diverged "
            "from FP64 oracle: maximum_error_m=" +
                std::to_string(maximumSelfContactError) +
                " cpu_clearance_m=" + std::to_string(cpuClearance) +
                " metal_clearance_m=" +
                std::to_string(firstMetalClearance)
        );

        metalrobo::DiscreteElasticRodModel toolRod =
            metalrobo::makeStraightSutureRod(5u, 0.12);
        std::vector<metalrobo::DiscreteElasticRodState>
            toolStates(
                environmentCount,
                metalrobo::makeDiscreteElasticRodDefaultState(
                    toolRod
                )
            );
        metalrobo::EngineModel toolModel =
            metalrobo::makeFreeSphereEngineModel();
        toolModel.shapes[1].dimensions.x = 0.002F;
        toolModel.shapes[1]
            .contactRestAndBoundingRadius = {
                2.0e-5F,
                0.0F,
                0.002F,
                0.0F,
            };
        std::string toolModelReason;
        require(
            toolModel.valid(&toolModelReason),
            "rod/tool model is invalid: " + toolModelReason
        );
        std::vector<MRBodyStateGPU> toolBodies(
            environmentCount * toolModel.bodies.size()
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            MRBodyStateGPU& floor =
                toolBodies[environment * 2u];
            floor.position = {0.0F, -1.0F, 0.0F, 1.0F};
            floor.orientation = {0.0F, 0.0F, 0.0F, 1.0F};
            floor.flagsAndIndices[0] = MR_MOTION_STATIC;
            floor.flagsAndIndices[1] = MR_INVALID_INDEX;
            floor.flagsAndIndices[2] = 0u;
            MRBodyStateGPU& sphere =
                toolBodies[environment * 2u + 1u];
            sphere.position = {
                0.06F,
                -0.0019F,
                0.0F,
                1.0F,
            };
            sphere.orientation = {0.0F, 0.0F, 0.0F, 1.0F};
            sphere.linearVelocityAndInverseMass = {
                0.0F,
                0.0F,
                0.0F,
                1.0F,
            };
            sphere.inverseInertiaWorldRow0 =
                toolModel.bodies[1].inverseInertiaRow0;
            sphere.inverseInertiaWorldRow1 =
                toolModel.bodies[1].inverseInertiaRow1;
            sphere.inverseInertiaWorldRow2 =
                toolModel.bodies[1].inverseInertiaRow2;
            sphere.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
            sphere.flagsAndIndices[1] = 0u;
            sphere.flagsAndIndices[2] = 1u;
        }
        std::vector<MRRodToolPairGPU> toolPairs;
        for (std::uint32_t edge = 0u;
             edge + 1u < toolRod.restPositions.size();
             ++edge) {
            toolPairs.push_back({
                .rodCollider = edge,
                .rigidCollider = 1u,
                .pairClass =
                    MR_COLLISION_PAIR_SPHERE_CAPSULE,
                .flags = MR_ROD_TOOL_PAIR_VALID,
            });
        }
        metalrobo::MetalDiscreteElasticRodConfig toolConfig;
        toolConfig.step.gravity = {0.0, 0.0, 0.0};
        toolConfig.step.solverIterations = 64u;
        toolConfig.step.constraintTolerance = 1.0e-6;
        toolConfig.tool.enabled = true;
        toolConfig.tool.outerIterations = 3u;
        toolConfig.tool.contactOffset = 2.0e-5F;
        toolConfig.tool.restitution = 0.0F;
        metalrobo::MetalDiscreteElasticRodResult toolResult;
        const auto toolDiagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                toolRod,
                {
                    .states = toolStates,
                    .rigidBodyCount = 2u,
                    .rigidBodies = toolBodies,
                    .toolModel = &toolModel,
                    .toolPairs = toolPairs,
                },
                toolResult,
                toolConfig
            );
        require(
            toolDiagnostics.succeeded() &&
                toolDiagnostics.toolContactCount > 0u &&
                toolResult.toolContactCounts.size() ==
                    environmentCount * toolPairs.size(),
            "Metal thread/tool contact graph failed: " +
                toolDiagnostics.message +
                " status=" +
                std::to_string(
                    static_cast<std::uint32_t>(toolDiagnostics.status)
                ) +
                " contacts=" +
                std::to_string(toolDiagnostics.toolContactCount) +
                " result_counts=" +
                std::to_string(toolResult.toolContactCounts.size()) +
                " rod_mid_y=" +
                std::to_string(
                    toolResult.states.empty()
                    ? 0.0
                    : toolResult.states[0].positions[2][1]
                ) +
                " sphere_y=" +
                std::to_string(
                    toolResult.rigidBodies.size() < 2u
                    ? 0.0
                    : static_cast<double>(
                          toolResult.rigidBodies[1].position.y
                      )
                )
        );
        double maximumMomentumError = 0.0;
        double maximumToolImpulse = 0.0;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::array<double, 3> rodMomentum{};
            for (std::size_t node = 0u;
                 node < toolRod.nodeMasses.size();
                 ++node) {
                for (std::size_t axis = 0u;
                     axis < 3u;
                     ++axis) {
                    rodMomentum[axis] +=
                        toolRod.nodeMasses[node] *
                        toolResult.states[environment]
                            .velocities[node][axis];
                }
            }
            const MRBodyStateGPU& sphere =
                toolResult.rigidBodies[
                    environment * 2u + 1u
                ];
            const std::array<double, 3> totalMomentum{
                rodMomentum[0] +
                    sphere.linearVelocityAndInverseMass.x,
                rodMomentum[1] +
                    sphere.linearVelocityAndInverseMass.y,
                rodMomentum[2] +
                    sphere.linearVelocityAndInverseMass.z,
            };
            maximumMomentumError = std::max(
                maximumMomentumError,
                length(totalMomentum)
            );
            for (std::size_t pair = 0u;
                 pair < toolPairs.size();
                 ++pair) {
                const std::uint32_t count =
                    toolResult.toolContactCounts[
                        environment * toolPairs.size() + pair
                    ];
                for (std::uint32_t slot = 0u;
                     slot < count;
                     ++slot) {
                    const auto& witness =
                        toolResult.toolContacts[
                            (
                                environment *
                                    toolPairs.size() +
                                pair
                            ) *
                                MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR +
                            slot
                        ];
                    maximumToolImpulse = std::max(
                        maximumToolImpulse,
                        static_cast<double>(
                            std::sqrt(
                                witness.impulses.x *
                                    witness.impulses.x +
                                witness.impulses.y *
                                    witness.impulses.y +
                                witness.impulses.z *
                                    witness.impulses.z
                            )
                        )
                    );
                }
            }
        }
        require(
            maximumToolImpulse > 0.0 &&
                maximumMomentumError <= 2.0e-5,
            "thread/tool impulse was not equal and opposite"
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
            << " tool_contacts="
            << toolDiagnostics.toolContactCount
            << " tool_impulse=" << maximumToolImpulse
            << " tool_momentum_error="
            << maximumMomentumError
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
