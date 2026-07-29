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
