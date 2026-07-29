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

} // namespace

int main() {
    try {
        constexpr std::size_t environmentCount = 4u;
        const metalrobo::DiscreteElasticRodModel model =
            metalrobo::makeStraightSutureRod(9u, 0.12);
        std::vector<metalrobo::DiscreteElasticRodState> states(
            environmentCount,
            metalrobo::makeDiscreteElasticRodDefaultState(model)
        );
        std::vector<metalrobo::DiscreteElasticRodEnergy> before(
            environmentCount
        );
        std::vector<metalrobo::DiscreteRodAttachment> attachments(
            environmentCount
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            states[environment].positions[4][1] =
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
            attachments[environment] = {
                .nodeIndex = 0u,
                .targetPosition = model.restPositions[0],
                .targetVelocity = {0.0, 0.0, 0.0},
                .compliance = 0.0,
            };
        }

        metalrobo::MetalDiscreteElasticRodConfig config;
        config.step.gravity = {0.0, 0.0, 0.0};
        config.step.solverIterations = 256u;
        config.step.constraintTolerance = 1.0e-5;
        const metalrobo::MetalDiscreteElasticRodInput input{
            .states = states,
            .attachmentCount = 1u,
            .attachments = attachments,
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
                result.statuses.size() == environmentCount,
            "Metal rod result was not published"
        );
        double maximumAfter = 0.0;
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
                result.states[environment].positions.front() ==
                    attachments[environment].targetPosition &&
                    result.states[environment].velocities.front() ==
                    attachments[environment].targetVelocity,
                "hard rod attachment was not enforced"
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
                        result.states[environment].twistRates,
                "Metal rod replay is not deterministic"
            );
        }

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
