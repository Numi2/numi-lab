#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/MetalMultiArticulatedConstraints.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct OwnedBlock {
    metalrobo::ConstraintIRStableKey key{};
    std::uint32_t type = MR_CONSTRAINT_BILATERAL;
    std::vector<metalrobo::ConstraintIREndpoint> endpoints;
    std::vector<metalrobo::ConstraintIRRow> rows;
    std::vector<float> warmImpulses;
};

metalrobo::ConstraintIRRow scalarRow() {
    metalrobo::ConstraintIRRow row{};
    row.timeConstant = 0.01f;
    row.dampingRatio = 1.0f;
    row.impulseLower = -metalrobo::kConstraintIRUnbounded;
    row.impulseUpper = metalrobo::kConstraintIRUnbounded;
    return row;
}

metalrobo::DualPsmWorld crossArticulationWorld() {
    metalrobo::DualPsmWorldConfig config;
    config.lockBases = false;
    config.coupleJaws = false;
    metalrobo::DualPsmWorld world =
        metalrobo::makeDualDvrkPsmWorld(config);

    std::vector<OwnedBlock> owned;
    owned.reserve(8u);
    for (std::uint32_t axis = 0u; axis < 6u; ++axis) {
        OwnedBlock block;
        block.key.words[0] = 0x4d554c54u;
        block.key.words[1] = 0u;
        block.key.words[2] = axis;
        block.key.words[3] = 0u;
        for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
            const std::uint32_t qIndex = axis < 3u
                ? world.metadata.qOffsets[arm] + axis
                : metalrobo::kConstraintIRInvalidIndex;
            block.endpoints.push_back(
                metalrobo::makeConstraintIRGeneralizedEndpoint(
                    world.metadata.articulationIndices[arm],
                    qIndex,
                    world.metadata.vOffsets[arm] + axis,
                    0u,
                    arm == 0u ? 1.0f : -1.0f
                )
            );
        }
        block.rows.push_back(scalarRow());
        block.warmImpulses.push_back(0.0f);
        owned.push_back(std::move(block));
    }

    OwnedBlock jawGear;
    jawGear.key.words[0] = 0x4d554c54u;
    jawGear.key.words[1] = 1u;
    jawGear.type = MR_CONSTRAINT_GEAR;
    for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
        const MRDofPropertiesGPU& dof = world.model.dofs[
            world.metadata.firstJawVelocity[arm]
        ];
        jawGear.endpoints.push_back(
            metalrobo::makeConstraintIRGeneralizedEndpoint(
                world.metadata.articulationIndices[arm],
                dof.qIndex,
                dof.vIndex,
                0u,
                1.0f
            )
        );
    }
    jawGear.rows.push_back(scalarRow());
    jawGear.warmImpulses.push_back(0.0f);
    owned.push_back(std::move(jawGear));

    OwnedBlock unilateral;
    unilateral.key.words[0] = 0x4d554c54u;
    unilateral.key.words[1] = 2u;
    unilateral.type = MR_CONSTRAINT_LIMIT;
    const MRDofPropertiesGPU& limitedDof = world.model.dofs[
        world.metadata.secondJawVelocity[0]
    ];
    unilateral.endpoints.push_back(
        metalrobo::makeConstraintIRGeneralizedEndpoint(
            world.metadata.articulationIndices[0],
            limitedDof.qIndex,
            limitedDof.vIndex,
            0u,
            1.0f
        )
    );
    metalrobo::ConstraintIRRow limitRow = scalarRow();
    limitRow.positionError = -5.0e-4f;
    limitRow.flags =
        metalrobo::constraintIRRowPositionStabilized |
        metalrobo::constraintIRRowUnilateral;
    limitRow.impulseLower = 0.0f;
    unilateral.rows.push_back(limitRow);
    unilateral.warmImpulses.push_back(0.0f);
    owned.push_back(std::move(unilateral));

    std::vector<metalrobo::ConstraintIRSourceBlock> sources;
    sources.reserve(owned.size());
    for (const OwnedBlock& block : owned) {
        sources.push_back({
            .key = block.key,
            .type = block.type,
            .flags = 0u,
            .islandIndex = 0u,
            .eventSlot = metalrobo::kConstraintIRInvalidIndex,
            .endpoints = block.endpoints,
            .rows = block.rows,
            .cone = std::nullopt,
            .warmImpulses = block.warmImpulses,
        });
    }
    const metalrobo::ConstraintIRCompilationResult compiled =
        metalrobo::compileConstraintIR(sources);
    require(
        compiled.succeeded(),
        "cross-articulation ConstraintIR compilation failed"
    );
    world.model.constraintProgram = compiled.ir;
    std::string reason;
    require(
        world.model.valid(&reason),
        reason.c_str()
    );
    return world;
}

} // namespace

int main() {
    try {
        constexpr std::size_t environmentCount = 4u;
        metalrobo::DualPsmWorld world =
            crossArticulationWorld();
        const metalrobo::EngineModel& model = world.model;
        const std::size_t rowCount =
            model.constraintProgram.rows.size();
        require(
            model.articulations.size() == 2u &&
                rowCount == 8u,
            "probe did not create the intended cross-articulation graph"
        );

        std::vector<float> q(
            environmentCount * model.world.nq
        );
        std::vector<float> freeVelocity(
            environmentCount * model.world.nv
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                model.defaultQ.begin(),
                model.defaultQ.end(),
                q.begin() + environment * model.world.nq
            );
            for (std::size_t dof = 0u;
                 dof < model.world.nv;
                 ++dof) {
                freeVelocity[
                    environment * model.world.nv + dof
                ] =
                    0.08f *
                    std::sin(float(
                        1u + 3u * environment + dof
                    ));
            }
        }

        metalrobo::MetalMultiArticulatedConstraintConfig config;
        config.evaluation.timestep = 1.0 / 1000.0;
        config.evaluation.minimumRegularization = 1.0e-7;
        config.solverIterations = 192u;
        config.convergenceTolerance = 5.0e-5f;
        const metalrobo::MetalMultiArticulatedConstraintInput input{
            .environmentCount = environmentCount,
            .q = q,
            .freeVelocity = freeVelocity,
        };
        metalrobo::MetalMultiArticulatedConstraintResult result;
        const auto diagnostics =
            metalrobo::solveMetalMultiArticulatedConstraints(
                model,
                input,
                result,
                config
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{
                    "Metal generalized constraint solve failed: "
                } + diagnostics.message +
                " gpu_code=" +
                std::to_string(
                    diagnostics.firstGPUStatusCode
                ) +
                " row=" +
                std::to_string(
                    diagnostics.firstFailingRow
                )
            );
        }
        require(
            diagnostics.dispatched &&
                diagnostics.published &&
                result.nextVelocity.size() ==
                    freeVelocity.size() &&
                result.impulses.size() ==
                    environmentCount * rowCount &&
                result.layout.inverseMassDispatches.size() >
                    model.articulations.size(),
            "generalized result did not exercise chunked inverse mass"
        );

        double maximumResidual = 0.0;
        double maximumCrossVelocity = 0.0;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            const auto free =
                metalrobo::computeConstraintIRGeneralizedVelocities(
                    model.constraintProgram,
                    std::span{
                        freeVelocity.data() +
                            environment * model.world.nv,
                        model.world.nv,
                    }
                );
            const auto post =
                metalrobo::computeConstraintIRGeneralizedVelocities(
                    model.constraintProgram,
                    std::span{
                        result.nextVelocity.data() +
                            environment * model.world.nv,
                        model.world.nv,
                    }
                );
            require(
                free.succeeded() && post.succeeded(),
                "failed to evaluate generalized velocity evidence"
            );
            const auto evaluated = metalrobo::evaluateConstraintIR(
                model.constraintProgram,
                {free.relativeVelocities, {}},
                config.evaluation
            );
            require(
                evaluated.succeeded(),
                "CPU semantic evaluator rejected probe IR"
            );
            metalrobo::ConstraintIRResidualConfig residualConfig;
            residualConfig.residualTolerance = 2.0e-4;
            residualConfig.impulseTolerance = 2.0e-5;
            const auto residual =
                metalrobo::evaluateConstraintIRResidual(
                    metalrobo::makeConstraintIREvaluationView(
                        evaluated.evaluated,
                        metalrobo::ConstraintIRConsumer::throughput
                    ),
                    post.relativeVelocities,
                    std::span{
                        result.impulses.data() +
                            environment * rowCount,
                        rowCount,
                    },
                    residualConfig
                );
            require(
                residual.succeeded(),
                "failed to evaluate generalized KKT evidence"
            );
            maximumResidual = std::max(
                maximumResidual,
                residual.maximumNaturalResidual
            );
            for (std::size_t row = 0u; row < 7u; ++row) {
                maximumCrossVelocity = std::max(
                    maximumCrossVelocity,
                    std::abs(
                        static_cast<double>(
                            post.relativeVelocities[row]
                        )
                    )
                );
            }
        }
        require(
            maximumResidual <= 2.0e-4 &&
                maximumCrossVelocity <= 2.0e-4,
            "cross-articulation constraints missed the KKT gate"
        );

        metalrobo::MetalMultiArticulatedConstraintResult replay;
        const auto replayDiagnostics =
            metalrobo::solveMetalMultiArticulatedConstraints(
                model,
                input,
                replay,
                config
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.nextVelocity == result.nextVelocity &&
                replay.impulses == result.impulses,
            "generalized Metal solve is not deterministic"
        );

        metalrobo::MetalMultiArticulatedConstraintResult sentinel =
            result;
        const std::size_t sentinelSize =
            sentinel.nextVelocity.size();
        const float sentinelValue =
            sentinel.nextVelocity.front();
        std::vector<float> invalidVelocity = freeVelocity;
        invalidVelocity.front() =
            std::numeric_limits<float>::quiet_NaN();
        auto invalidInput = input;
        invalidInput.freeVelocity = invalidVelocity;
        const auto rejected =
            metalrobo::solveMetalMultiArticulatedConstraints(
                model,
                invalidInput,
                sentinel,
                config
            );
        require(
            !rejected.succeeded() &&
                !rejected.dispatched &&
                sentinel.nextVelocity.size() == sentinelSize &&
                sentinel.nextVelocity.front() == sentinelValue,
            "generalized rejection was not transactional"
        );

        std::cout
            << "multi_articulated_constraints=ok"
            << " device=\"" << diagnostics.deviceName << "\""
            << " articulations=" << model.articulations.size()
            << " environments=" << environmentCount
            << " rows=" << rowCount
            << " inverse_packets="
            << result.layout.inverseMassDispatches.size() *
                environmentCount
            << " max_natural_residual=" << maximumResidual
            << " max_cross_velocity=" << maximumCrossVelocity
            << " elapsed_ms=" << diagnostics.elapsedMilliseconds
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "multi_articulated_constraints=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
