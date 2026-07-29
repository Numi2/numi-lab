#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/MetalMultiArticulatedConstraints.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <array>
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
        const metalrobo::DualPsmWorld surgicalWorld =
            crossArticulationWorld();
        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        const std::array components{
            metalrobo::EngineModelComponent{
                .model = &surgicalWorld.model,
                .instanceId = "dual_psm_cell",
            },
            metalrobo::EngineModelComponent{
                .model = &g1,
                .instanceId = "g1_observer",
            },
        };
        metalrobo::EngineModel model;
        const auto composeDiagnostics =
            metalrobo::composeEngineModels(components, model);
        require(
            composeDiagnostics.succeeded(),
            "failed to compose heterogeneous constraint probe"
        );
        const std::size_t rowCount =
            model.constraintProgram.rows.size();
        require(
            model.articulations.size() == 3u &&
                rowCount == 8u,
            "probe did not create dual PSM plus G1 constraint graph"
        );

        metalrobo::CompiledMetalMultiArticulatedProgram program;
        const auto compileDiagnostics =
            metalrobo::compileMetalMultiArticulatedProgram(
                model,
                program
            );
        require(
            compileDiagnostics.succeeded() &&
                program.valid() &&
                program.rowCount() == rowCount &&
                program.abaSchedule().articulations.size() == 3u &&
                program.rowChunkOffsets().size() > 1u &&
                program.fingerprint() != 0u,
            "heterogeneous Metal execution plan did not cook"
        );
        const std::uint64_t programFingerprint =
            program.fingerprint();
        metalrobo::CompiledMetalMultiArticulatedProgram replayProgram;
        const auto replayCompileDiagnostics =
            metalrobo::compileMetalMultiArticulatedProgram(
                model,
                replayProgram
            );
        require(
            replayCompileDiagnostics.succeeded() &&
                replayProgram.fingerprint() == programFingerprint &&
                std::ranges::equal(
                    replayProgram.generalizedJacobian(),
                    program.generalizedJacobian()
                ) &&
                std::ranges::equal(
                    replayProgram.rowChunkOffsets(),
                    program.rowChunkOffsets()
                ) &&
                std::ranges::equal(
                    replayProgram.rowChunkCounts(),
                    program.rowChunkCounts()
                ),
            "compiled execution plan is not deterministic"
        );
        metalrobo::EngineModel invalidModel = model;
        ++invalidModel.world.nv;
        const auto rejectedCompile =
            metalrobo::compileMetalMultiArticulatedProgram(
                invalidModel,
                replayProgram
            );
        require(
            !rejectedCompile.succeeded() &&
                replayProgram.fingerprint() == programFingerprint,
            "failed plan compilation modified the published plan"
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
        metalrobo::MetalMultiArticulatedConstraintContext context(
            program,
            config
        );
        metalrobo::MetalMultiArticulatedConstraintSubmission
            submission;
        metalrobo::MetalMultiArticulatedConstraintResult result;
        auto diagnostics = context.submit(input, submission);
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"Metal generalized submit failed: "} +
                diagnostics.message
            );
        }
        require(
            diagnostics.dispatched &&
                !diagnostics.published &&
                submission.valid(),
            "persistent context did not return an async ticket"
        );
        const auto submittedStats = context.stats();
        require(
            submittedStats.pipelineCreationCount == 3u &&
                submittedStats.immutableUploadCount == 1u &&
                submittedStats.submissionCount == 1u &&
                submittedStats.parallelABAFrontierCount ==
                    program.abaSchedule().levels.size() &&
                submittedStats.maximumABAFrontierWidth > 1u &&
                submittedStats.hasInFlightSubmission &&
                submittedStats.retainedBufferBytes > 0u,
            "persistent context did not retain its device program"
        );
        metalrobo::MetalMultiArticulatedConstraintSubmission
            busySubmission;
        const auto busyDiagnostics =
            context.submit(input, busySubmission);
        require(
            busyDiagnostics.status ==
                metalrobo::
                    MetalMultiArticulatedConstraintStatus::
                        contextBusy &&
                !busySubmission.valid(),
            "persistent arena admitted overlapping submissions"
        );
        diagnostics = submission.wait(result);
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"Metal generalized wait failed: "} +
                diagnostics.message +
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
        const auto completedStats = context.stats();
        require(
            diagnostics.dispatched &&
                diagnostics.published &&
                result.nextVelocity.size() ==
                    freeVelocity.size() &&
                result.impulses.size() ==
                    environmentCount * rowCount &&
                result.layout.inverseMassDispatches.size() >
                    model.articulations.size() &&
                completedStats.completedSubmissionCount == 1u &&
                !completedStats.hasInFlightSubmission,
            "persistent result did not exercise chunked inverse mass"
        );
        auto qualityConfig = config;
        qualityConfig.solverMode =
            metalrobo::MetalGeneralizedConstraintSolverMode::
                qualitySemismoothNewton;
        qualityConfig.solverIterations = 128u;
        qualityConfig.qualityCGIterations = 96u;
        qualityConfig.qualityLineSearchIterations = 16u;
        metalrobo::MetalMultiArticulatedConstraintContext
            qualityContext(program, qualityConfig);
        metalrobo::MetalMultiArticulatedConstraintResult
            qualityResult;
        const auto qualityDiagnostics =
            qualityContext.run(input, qualityResult);
        if (!qualityDiagnostics.succeeded() ||
            !qualityDiagnostics.published ||
            qualityResult.nextVelocity.size() !=
                freeVelocity.size() ||
            qualityResult.impulses.size() !=
                environmentCount * rowCount) {
            throw std::runtime_error(
                "Metal generalized quality solve failed: " +
                qualityDiagnostics.message +
                " code=" +
                std::to_string(
                    qualityDiagnostics.firstGPUStatusCode
                ) +
                " row=" +
                std::to_string(
                    qualityDiagnostics.firstFailingRow
                ) +
                " throughput_diag=[" +
                std::to_string(
                    result.statuses.front().diagnostics.z
                ) +
                "," +
                std::to_string(
                    result.statuses.front().diagnostics.w
                ) +
                "]"
            );
        }
        const MRArticulationGPU& observer =
            model.articulations.back();
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            for (std::uint32_t local = 0u;
                 local < observer.nv;
                 ++local) {
                const std::size_t index =
                    environment * model.world.nv +
                    observer.vOffset + local;
                require(
                    result.nextVelocity[index] ==
                        freeVelocity[index] &&
                    qualityResult.nextVelocity[index] ==
                        freeVelocity[index],
                    "uncoupled G1 velocity changed"
                );
            }
        }

        double maximumResidual = 0.0;
        double maximumQualityResidual = 0.0;
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
            const auto qualityPost =
                metalrobo::computeConstraintIRGeneralizedVelocities(
                    model.constraintProgram,
                    std::span{
                        qualityResult.nextVelocity.data() +
                            environment * model.world.nv,
                        model.world.nv,
                    }
                );
            require(
                free.succeeded() &&
                    post.succeeded() &&
                    qualityPost.succeeded(),
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
            const auto qualityResidual =
                metalrobo::evaluateConstraintIRResidual(
                    metalrobo::makeConstraintIREvaluationView(
                        evaluated.evaluated,
                        metalrobo::ConstraintIRConsumer::quality
                    ),
                    qualityPost.relativeVelocities,
                    std::span{
                        qualityResult.impulses.data() +
                            environment * rowCount,
                        rowCount,
                    },
                    residualConfig
                );
            require(
                qualityResidual.succeeded(),
                "failed to evaluate generalized quality KKT evidence"
            );
            maximumQualityResidual = std::max(
                maximumQualityResidual,
                qualityResidual.maximumNaturalResidual
            );
            for (std::size_t row = 0u;
                 row < rowCount;
                 ++row) {
                if ((model.constraintProgram.rows[row].flags &
                     metalrobo::constraintIRRowUnilateral) != 0u) {
                    continue;
                }
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
        if (maximumResidual > 2.0e-4 ||
            maximumQualityResidual > 2.0e-4 ||
            maximumCrossVelocity > 2.0e-4) {
            throw std::runtime_error(
                "cross-articulation constraints missed the KKT "
                "gate residual=" +
                std::to_string(maximumResidual) +
                " quality_residual=" +
                std::to_string(maximumQualityResidual) +
                " cross_velocity=" +
                std::to_string(maximumCrossVelocity)
            );
        }

        metalrobo::MetalMultiArticulatedConstraintResult replay;
        const auto replayDiagnostics = context.run(input, replay);
        const auto replayStats = context.stats();
        require(
            replayDiagnostics.succeeded() &&
                replay.nextVelocity == result.nextVelocity &&
                replay.impulses == result.impulses &&
                replayStats.pipelineCreationCount ==
                    completedStats.pipelineCreationCount &&
                replayStats.bufferAllocationCount ==
                    completedStats.bufferAllocationCount &&
                replayStats.immutableUploadCount ==
                    completedStats.immutableUploadCount &&
                replayStats.completedSubmissionCount == 2u,
            "persistent generalized replay rebuilt or diverged"
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
            context.run(invalidInput, sentinel);
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
            << " plan_fingerprint=" << programFingerprint
            << " aba_levels="
            << program.abaSchedule().levels.size()
            << " max_aba_frontier_width="
            << replayStats.maximumABAFrontierWidth
            << " row_chunks="
            << program.rowChunkOffsets().size()
            << " retained_bytes="
            << replayStats.retainedBufferBytes
            << " pipeline_creations="
            << replayStats.pipelineCreationCount
            << " buffer_allocations="
            << replayStats.bufferAllocationCount
            << " inverse_packets="
            << result.layout.inverseMassDispatches.size() *
                environmentCount
            << " max_natural_residual=" << maximumResidual
            << " max_quality_residual="
            << maximumQualityResidual
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
