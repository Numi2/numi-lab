#include "metalrobo/DiscreteElasticRod.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double distance(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b
) {
    const double x = a[0] - b[0];
    const double y = a[1] - b[1];
    const double z = a[2] - b[2];
    return std::sqrt(x * x + y * y + z * z);
}

std::array<double, 3> midpoint(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b
) {
    return {
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    };
}

} // namespace

int main() {
    try {
        const metalrobo::DiscreteElasticRodModel model =
            metalrobo::makeStraightSutureRod(
                9u,
                0.12
            );
        metalrobo::DiscreteElasticRodState state =
            metalrobo::makeDiscreteElasticRodDefaultState(model);
        state.positions[4][1] = 0.008;
        state.positions.back()[0] += 0.006;
        state.twists.back() = 0.6;
        metalrobo::DiscreteElasticRodEnergy initial{};
        const auto initialDiagnostics =
            metalrobo::evaluateDiscreteElasticRodEnergy(
                model,
                state,
                initial
            );
        require(
            initialDiagnostics.succeeded() &&
                initial.stretch > 0.0 &&
                initial.bend > 0.0 &&
                initial.twist > 0.0,
            "deformed DER energy is incomplete"
        );

        metalrobo::DiscreteRodAttachment attachment{
            .nodeIndex = 0u,
            .targetPosition = model.restPositions[0],
            .targetVelocity = {0.0, 0.0, 0.0},
            .compliance = 0.0,
        };
        metalrobo::DiscreteElasticRodStepConfig config;
        config.gravity = {0.0, 0.0, 0.0};
        config.solverIterations = 96u;
        config.constraintTolerance = 1.0e-5;
        std::array<
            metalrobo::DiscreteRodAttachmentReaction,
            1
        > reactions{};
        const auto diagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                model,
                state,
                {&attachment, 1u},
                config,
                reactions
            );
        require(
            diagnostics.succeeded(),
            "implicit DER step failed: " +
                diagnostics.message
        );
        require(
            state.positions[0] == attachment.targetPosition &&
                distance(
                    state.positions[0],
                    state.positions[1]
                ) < 1.1 * model.restLengths[0],
            "hard attachment or stretch projection failed"
        );
        require(
            reactions[0].nodeIndex == 0u &&
                std::isfinite(
                    reactions[0].averageForceOnTarget[0]
                ) &&
                std::isfinite(
                    reactions[0].averageForceOnTarget[1]
                ) &&
                std::isfinite(
                    reactions[0].averageForceOnTarget[2]
                ) &&
                reactions[0].finalPositionError <= 1.0e-12,
            "DER attachment reaction evidence is invalid"
        );
        require(
            diagnostics.after.total() <
                initial.total(),
            "DER projection did not reduce elastic energy"
        );

        metalrobo::DiscreteElasticRodState replay =
            metalrobo::makeDiscreteElasticRodDefaultState(model);
        replay.positions[4][1] = 0.008;
        replay.positions.back()[0] += 0.006;
        replay.twists.back() = 0.6;
        const auto replayDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                model,
                replay,
                {&attachment, 1u},
                config
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.positions == state.positions &&
                replay.velocities == state.velocities &&
                replay.twists == state.twists &&
                replay.twistRates == state.twistRates,
            "DER replay is not deterministic"
        );

        metalrobo::DiscreteElasticRodState rejected = state;
        rejected.positions[3][0] =
            std::numeric_limits<double>::quiet_NaN();
        const auto sentinel = rejected;
        const auto failed =
            metalrobo::stepDiscreteElasticRodCpu(
                model,
                rejected,
                {&attachment, 1u},
                config
            );
        require(
            !failed.succeeded() &&
                std::isnan(rejected.positions[3][0]) &&
                rejected.positions[2] == sentinel.positions[2] &&
                rejected.twists == sentinel.twists,
            "DER non-finite rejection changed state"
        );

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
        metalrobo::DiscreteElasticRodStepConfig collisionConfig;
        collisionConfig.gravity = {0.0, 0.0, 0.0};
        collisionConfig.solverIterations = 256u;
        collisionConfig.constraintTolerance = 1.0e-6;
        collisionConfig.enableSelfCollision = true;
        const auto crossingInput = crossing;
        const auto collisionDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                crossing,
                {},
                collisionConfig
            );
        const double separatedMidpoints = distance(
            midpoint(
                crossing.positions[1],
                crossing.positions[0]
            ),
            midpoint(
                crossing.positions[2],
                crossing.positions[3]
            )
        );
        require(
            collisionDiagnostics.succeeded() &&
                collisionDiagnostics.projectedSelfContacts > 0u &&
                collisionDiagnostics.maximumSelfPenetration >=
                    1.9 * collisionModel.radius &&
                separatedMidpoints >=
                    0.95 * 2.0 * collisionModel.radius,
            "DER capsule self-contact did not separate crossing edges: " +
                collisionDiagnostics.message +
                " contacts=" +
                std::to_string(
                    collisionDiagnostics.projectedSelfContacts
                ) +
                " penetration=" +
                std::to_string(
                    collisionDiagnostics.maximumSelfPenetration
                ) +
                " separation=" +
                std::to_string(separatedMidpoints)
        );
        auto crossingReplay = crossingInput;
        const auto crossingReplayDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                crossingReplay,
                {},
                collisionConfig
            );
        require(
            crossingReplayDiagnostics.succeeded() &&
                crossingReplay.positions == crossing.positions &&
                crossingReplay.velocities == crossing.velocities,
            "DER self-contact replay is not deterministic"
        );

        std::cout
            << "discrete_elastic_rod=ok"
            << " nodes=" << model.restPositions.size()
            << " edges=" << model.restLengths.size()
            << " iterations=" << diagnostics.iterations
            << " stretch=" << diagnostics.after.stretch
            << " bend=" << diagnostics.after.bend
            << " twist=" << diagnostics.after.twist
            << " max_error="
            << diagnostics.maximumConstraintError
            << " anchor_force_n="
            << std::sqrt(
                reactions[0].averageForceOnTarget[0] *
                    reactions[0].averageForceOnTarget[0] +
                reactions[0].averageForceOnTarget[1] *
                    reactions[0].averageForceOnTarget[1] +
                reactions[0].averageForceOnTarget[2] *
                reactions[0].averageForceOnTarget[2]
            )
            << " self_contacts="
            << collisionDiagnostics.projectedSelfContacts
            << " self_penetration="
            << collisionDiagnostics.maximumSelfPenetration
            << " separated_midpoints="
            << separatedMidpoints
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "discrete_elastic_rod=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
