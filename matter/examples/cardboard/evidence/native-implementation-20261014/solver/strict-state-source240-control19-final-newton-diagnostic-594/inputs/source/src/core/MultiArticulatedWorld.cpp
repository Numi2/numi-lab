#include "metalrobo/MultiArticulatedWorld.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const std::span<const double> values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

MultiArticulatedWorldDiagnostics fail(
    MultiArticulatedWorldDiagnostics diagnostics,
    const MultiArticulatedWorldStatus status,
    const std::uint32_t articulation =
        kConstraintIRInvalidIndex
) {
    diagnostics.status = status;
    diagnostics.firstFailingArticulation = articulation;
    return diagnostics;
}

bool validConfiguration(
    const MultiArticulatedWorldConfig& config
) {
    return
        finite(config.dynamics.timestep) &&
        config.dynamics.timestep > 0.0 &&
        config.solverIterations > 0u &&
        finite(config.solverTolerance) &&
        config.solverTolerance > 0.0;
}

bool solveFactor(
    const MultiArticulationFactor& factor,
    const std::span<const double> right,
    std::span<double> solution,
    double& relativeResidual
) {
    if (right.size() != factor.nv ||
        solution.size() != factor.nv ||
        factor.massCholeskyLower.size() !=
            static_cast<std::size_t>(factor.nv) * factor.nv) {
        return false;
    }
    std::vector<double> intermediate(factor.nv, 0.0);
    for (std::uint32_t row = 0u; row < factor.nv; ++row) {
        double value = right[row];
        for (std::uint32_t column = 0u;
             column < row;
             ++column) {
            value -=
                factor.massCholeskyLower[
                    row * factor.nv + column
                ] * intermediate[column];
        }
        const double diagonal =
            factor.massCholeskyLower[
                row * factor.nv + row
            ];
        if (!(diagonal > 0.0) || !finite(diagonal)) {
            return false;
        }
        intermediate[row] = value / diagonal;
    }
    for (std::uint32_t reverse = 0u;
         reverse < factor.nv;
         ++reverse) {
        const std::uint32_t row =
            factor.nv - 1u - reverse;
        double value = intermediate[row];
        for (std::uint32_t column = row + 1u;
             column < factor.nv;
             ++column) {
            value -=
                factor.massCholeskyLower[
                    column * factor.nv + row
                ] * solution[column];
        }
        solution[row] =
            value /
            factor.massCholeskyLower[
                row * factor.nv + row
            ];
        if (!finite(solution[row])) {
            return false;
        }
    }

    std::vector<double> transposeAction(factor.nv, 0.0);
    for (std::uint32_t column = 0u;
         column < factor.nv;
         ++column) {
        for (std::uint32_t row = column;
             row < factor.nv;
             ++row) {
            transposeAction[column] +=
                factor.massCholeskyLower[
                    row * factor.nv + column
                ] * solution[row];
        }
    }
    double maximumError = 0.0;
    double scale = 1.0;
    for (std::uint32_t row = 0u;
         row < factor.nv;
         ++row) {
        double action = 0.0;
        for (std::uint32_t column = 0u;
             column <= row;
             ++column) {
            action +=
                factor.massCholeskyLower[
                    row * factor.nv + column
                ] * transposeAction[column];
        }
        maximumError = std::max(
            maximumError,
            std::abs(action - right[row])
        );
        scale = std::max({
            scale,
            std::abs(action),
            std::abs(right[row]),
        });
    }
    relativeResidual = maximumError / scale;
    return finite(relativeResidual);
}

MultiArticulatedWorldStatus statusForDynamics(
    const ArticulatedDynamicsStatus status
) {
    switch (status) {
    case ArticulatedDynamicsStatus::success:
        return MultiArticulatedWorldStatus::success;
    case ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite:
        return MultiArticulatedWorldStatus::factorizationFailure;
    case ArticulatedDynamicsStatus::nonlinearSolveFailed:
        return MultiArticulatedWorldStatus::didNotConverge;
    case ArticulatedDynamicsStatus::nonfiniteResult:
        return MultiArticulatedWorldStatus::nonfiniteResult;
    case ArticulatedDynamicsStatus::unsupportedTopology:
        return MultiArticulatedWorldStatus::invalidModel;
    case ArticulatedDynamicsStatus::invalidModel:
    case ArticulatedDynamicsStatus::invalidDimensions:
    case ArticulatedDynamicsStatus::nonfiniteInput:
    case ArticulatedDynamicsStatus::invalidQuaternion:
    case ArticulatedDynamicsStatus::jointLimitViolation:
    case ArticulatedDynamicsStatus::bodySpeedLimitViolation:
        return MultiArticulatedWorldStatus::freeDynamicsFailure;
    }
    return MultiArticulatedWorldStatus::nonfiniteResult;
}

bool buildGeneralizedJacobian(
    const EngineModel& model,
    std::vector<double>& jacobian
) {
    const ConstraintIR& program = model.constraintProgram;
    jacobian.assign(
        program.rows.size() * model.world.nv,
        0.0
    );
    for (const ConstraintIRBlock& block : program.blocks) {
        if (block.type == MR_CONSTRAINT_CONTACT) {
            return false;
        }
        for (std::uint32_t local = 0u;
             local < block.endpointCount;
             ++local) {
            const ConstraintIREndpoint& endpoint =
                program.endpoints[
                    block.endpointOffset + local
                ];
            if (endpoint.jacobianKind !=
                    constraintIRJacobianGeneralized ||
                endpoint.objectIndex >= model.world.nv) {
                return false;
            }
            const std::uint32_t localRow =
                endpoint.flags &
                constraintIREndpointRowMask;
            jacobian[
                (block.rowOffset + localRow) *
                    model.world.nv +
                endpoint.objectIndex
            ] += endpoint.axis.x;
        }
    }
    return finite(jacobian);
}

} // namespace

MultiArticulatedWorldDiagnostics
buildMultiArticulationFactorCache(
    const EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> velocity,
    MultiArticulationFactorCache& output,
    const ArticulatedDynamicsConfig& config
) {
    MultiArticulatedWorldDiagnostics diagnostics;
    diagnostics.articulationCount =
        static_cast<std::uint32_t>(
            model.articulations.size()
        );
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::invalidModel
        );
    }
    if (q.size() != model.world.nq ||
        velocity.size() != model.world.nv) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::invalidDimensions
        );
    }
    if (!finite(q) || !finite(velocity)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::nonfiniteInput
        );
    }

    MultiArticulationFactorCache staged;
    staged.generation = output.generation + 1u;
    staged.factors.reserve(model.articulations.size());
    for (std::uint32_t articulationIndex = 0u;
         articulationIndex < model.articulations.size();
         ++articulationIndex) {
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        ArticulatedContactProblem factorProblem;
        const ArticulatedContactDiagnostics factorDiagnostics =
            buildArticulatedContactProblem(
                model,
                articulationIndex,
                q.subspan(
                    articulation.qOffset,
                    articulation.nq
                ),
                velocity.subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                {},
                factorProblem,
                config,
                false
            );
        if (!factorDiagnostics.succeeded()) {
            return fail(
                std::move(diagnostics),
                factorDiagnostics.status ==
                    ArticulatedContactStatus::
                        factorizationFailure
                    ? MultiArticulatedWorldStatus::
                          factorizationFailure
                    : MultiArticulatedWorldStatus::
                          freeDynamicsFailure,
                articulationIndex
            );
        }
        staged.factors.push_back({
            .articulationIndex = articulationIndex,
            .qOffset = articulation.qOffset,
            .nq = articulation.nq,
            .vOffset = articulation.vOffset,
            .nv = articulation.nv,
            .massCholeskyLower =
                std::move(
                    factorProblem.massCholeskyLower
                ),
        });
    }
    output = std::move(staged);
    return diagnostics;
}

MultiArticulatedWorldDiagnostics
applyMultiArticulationInverseMass(
    const EngineModel& model,
    const MultiArticulationFactorCache& factors,
    const std::span<const double> generalizedImpulse,
    const std::span<double> velocityDelta
) {
    MultiArticulatedWorldDiagnostics diagnostics;
    diagnostics.articulationCount =
        static_cast<std::uint32_t>(
            model.articulations.size()
        );
    if (generalizedImpulse.size() != model.world.nv ||
        velocityDelta.size() != model.world.nv ||
        factors.factors.size() !=
            model.articulations.size()) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::invalidDimensions
        );
    }
    if (!finite(generalizedImpulse)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::nonfiniteInput
        );
    }

    std::vector<double> staged(model.world.nv, 0.0);
    for (std::uint32_t index = 0u;
         index < factors.factors.size();
         ++index) {
        const MultiArticulationFactor& factor =
            factors.factors[index];
        const MRArticulationGPU& articulation =
            model.articulations[index];
        if (factor.articulationIndex != index ||
            factor.vOffset != articulation.vOffset ||
            factor.nv != articulation.nv) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedWorldStatus::invalidDimensions,
                index
            );
        }
        double residual = 0.0;
        if (!solveFactor(
                factor,
                generalizedImpulse.subspan(
                    factor.vOffset,
                    factor.nv
                ),
                std::span<double>(staged).subspan(
                    factor.vOffset,
                    factor.nv
                ),
                residual
            )) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedWorldStatus::
                    factorizationFailure,
                index
            );
        }
        diagnostics.maximumFactorResidual = std::max(
            diagnostics.maximumFactorResidual,
            residual
        );
    }
    std::ranges::copy(staged, velocityDelta.begin());
    return diagnostics;
}

MultiArticulatedWorldDiagnostics stepMultiArticulatedWorldCpu(
    const EngineModel& model,
    const std::span<double> q,
    const std::span<double> v,
    const std::span<const double> generalizedForce,
    const std::span<const ArticulatedBodyWrench> externalWrenches,
    MultiArticulationFactorCache& cache,
    const MultiArticulatedWorldConfig& config
) {
    MultiArticulatedWorldDiagnostics diagnostics;
    diagnostics.articulationCount =
        static_cast<std::uint32_t>(
            model.articulations.size()
        );
    diagnostics.constraintBlockCount =
        static_cast<std::uint32_t>(
            model.constraintProgram.blocks.size()
        );
    diagnostics.constraintRowCount =
        static_cast<std::uint32_t>(
            model.constraintProgram.rows.size()
        );
    if (!validConfiguration(config)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::
                invalidConfiguration
        );
    }
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::invalidModel
        );
    }
    if (q.size() != model.world.nq ||
        v.size() != model.world.nv ||
        generalizedForce.size() != model.world.nv ||
        (!externalWrenches.empty() &&
         externalWrenches.size() != model.bodies.size())) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::invalidDimensions
        );
    }
    if (!finite(q) || !finite(v) ||
        !finite(generalizedForce)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::nonfiniteInput
        );
    }

    std::vector<double> candidateQ(q.begin(), q.end());
    std::vector<double> freeVelocity(v.begin(), v.end());
    diagnostics.freeDynamics.resize(
        model.articulations.size()
    );
    for (std::uint32_t articulationIndex = 0u;
         articulationIndex < model.articulations.size();
         ++articulationIndex) {
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        std::vector<double> acceleration(
            articulation.nv,
            0.0
        );
        diagnostics.freeDynamics[articulationIndex] =
            computeArticulatedForwardDynamics(
                model,
                articulationIndex,
                std::span<const double>(q).subspan(
                    articulation.qOffset,
                    articulation.nq
                ),
                std::span<const double>(v).subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                generalizedForce.subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                externalWrenches,
                acceleration,
                config.dynamics
            );
        if (!diagnostics.freeDynamics[
                articulationIndex
            ].succeeded()) {
            return fail(
                std::move(diagnostics),
                statusForDynamics(
                    diagnostics.freeDynamics[
                        articulationIndex
                    ].status
                ),
                articulationIndex
            );
        }
        for (std::uint32_t local = 0u;
             local < articulation.nv;
             ++local) {
            freeVelocity[
                articulation.vOffset + local
            ] += config.dynamics.timestep *
                acceleration[local];
        }
    }

    MultiArticulationFactorCache stagedCache;
    MultiArticulatedWorldDiagnostics factorDiagnostics =
        buildMultiArticulationFactorCache(
            model,
            q,
            freeVelocity,
            stagedCache,
            config.dynamics
        );
    if (!factorDiagnostics.succeeded()) {
        factorDiagnostics.freeDynamics =
            std::move(diagnostics.freeDynamics);
        return factorDiagnostics;
    }

    std::vector<double> nextVelocity = freeVelocity;
    if (!model.constraintProgram.blocks.empty()) {
        std::vector<double> jacobian;
        if (!buildGeneralizedJacobian(model, jacobian)) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedWorldStatus::
                    unsupportedConstraint
            );
        }
        const std::size_t rowCount =
            model.constraintProgram.rows.size();
        std::vector<float> relative(rowCount, 0.0F);
        for (std::size_t row = 0u;
             row < rowCount;
             ++row) {
            double value = 0.0;
            for (std::uint32_t dof = 0u;
                 dof < model.world.nv;
                 ++dof) {
                value +=
                    jacobian[row * model.world.nv + dof] *
                    freeVelocity[dof];
            }
            if (!finite(value) ||
                std::abs(value) >
                    std::numeric_limits<float>::max()) {
                return fail(
                    std::move(diagnostics),
                    MultiArticulatedWorldStatus::
                        nonfiniteResult
                );
            }
            relative[row] = static_cast<float>(value);
        }
        ConstraintIREvaluationConfig evaluationConfig =
            config.constraintEvaluation;
        evaluationConfig.timestep =
            config.dynamics.timestep;
        const ConstraintIREvaluationResult evaluation =
            evaluateConstraintIR(
                model.constraintProgram,
                {relative, {}},
                evaluationConfig
            );
        diagnostics.constraintEvaluation =
            evaluation.diagnostics;
        if (!evaluation.succeeded()) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedWorldStatus::
                    constraintEvaluationFailure
            );
        }

        std::vector<double> responseColumns(
            rowCount * model.world.nv,
            0.0
        );
        std::vector<double> delassus(
            rowCount * rowCount,
            0.0
        );
        std::vector<double> impulseRhs(
            model.world.nv,
            0.0
        );
        std::vector<double> response(
            model.world.nv,
            0.0
        );
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            for (std::uint32_t dof = 0u;
                 dof < model.world.nv;
                 ++dof) {
                impulseRhs[dof] =
                    jacobian[
                        column * model.world.nv + dof
                    ];
            }
            const MultiArticulatedWorldDiagnostics action =
                applyMultiArticulationInverseMass(
                    model,
                    stagedCache,
                    impulseRhs,
                    response
                );
            if (!action.succeeded()) {
                return fail(
                    std::move(diagnostics),
                    action.status,
                    action.firstFailingArticulation
                );
            }
            diagnostics.maximumFactorResidual = std::max(
                diagnostics.maximumFactorResidual,
                action.maximumFactorResidual
            );
            for (std::uint32_t dof = 0u;
                 dof < model.world.nv;
                 ++dof) {
                responseColumns[
                    column * model.world.nv + dof
                ] = response[dof];
            }
            for (std::size_t row = 0u;
                 row < rowCount;
                 ++row) {
                double value = 0.0;
                for (std::uint32_t dof = 0u;
                     dof < model.world.nv;
                     ++dof) {
                    value +=
                        jacobian[
                            row * model.world.nv + dof
                        ] * response[dof];
                }
                delassus[row * rowCount + column] =
                    value;
            }
        }

        std::vector<float> impulses =
            evaluation.evaluated.warmImpulses;
        std::vector<double> physicalVelocity(
            relative.begin(),
            relative.end()
        );
        bool converged = false;
        for (std::uint32_t iteration = 0u;
             iteration < config.solverIterations;
             ++iteration) {
            double maximumDelta = 0.0;
            for (std::size_t row = 0u;
                 row < rowCount;
                 ++row) {
                const EvaluatedConstraintIRRow& semantics =
                    evaluation.evaluated.rows[row];
                const double diagonal =
                    delassus[row * rowCount + row] +
                    semantics.regularization;
                if (!(diagonal > 0.0) ||
                    !finite(diagonal)) {
                    return fail(
                        std::move(diagnostics),
                        MultiArticulatedWorldStatus::
                            solverFailure
                    );
                }
                const double oldImpulse = impulses[row];
                const double gradient =
                    physicalVelocity[row] -
                    semantics.targetVelocity +
                    semantics.regularization *
                        oldImpulse;
                const double candidate = std::clamp(
                    oldImpulse - gradient / diagonal,
                    static_cast<double>(
                        semantics.impulseLower
                    ),
                    static_cast<double>(
                        semantics.impulseUpper
                    )
                );
                const double delta =
                    candidate - oldImpulse;
                impulses[row] =
                    static_cast<float>(candidate);
                maximumDelta = std::max(
                    maximumDelta,
                    std::abs(delta)
                );
                for (std::size_t affected = 0u;
                     affected < rowCount;
                     ++affected) {
                    physicalVelocity[affected] +=
                        delassus[
                            affected * rowCount + row
                        ] * delta;
                }
            }
            diagnostics.solverIterations = iteration + 1u;
            diagnostics.maximumImpulseDelta =
                maximumDelta;
            if (maximumDelta <= config.solverTolerance) {
                converged = true;
                break;
            }
        }
        std::vector<float> postVelocity(
            physicalVelocity.begin(),
            physicalVelocity.end()
        );
        diagnostics.residual =
            evaluateConstraintIRResidual(
                makeConstraintIREvaluationView(
                    evaluation.evaluated,
                    ConstraintIRConsumer::quality
                ),
                postVelocity,
                impulses,
                config.constraintResidual
            );
        if (!converged ||
            !diagnostics.residual.succeeded() ||
            diagnostics.residual.maximumNaturalResidual >
                std::max(
                    config.solverTolerance,
                    config.constraintResidual
                        .residualTolerance
                )) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedWorldStatus::
                    didNotConverge
            );
        }

        std::ranges::fill(impulseRhs, 0.0);
        for (std::size_t row = 0u;
             row < rowCount;
             ++row) {
            for (std::uint32_t dof = 0u;
                 dof < model.world.nv;
                 ++dof) {
                impulseRhs[dof] +=
                    jacobian[
                        row * model.world.nv + dof
                    ] * impulses[row];
            }
        }
        const MultiArticulatedWorldDiagnostics action =
            applyMultiArticulationInverseMass(
                model,
                stagedCache,
                impulseRhs,
                response
            );
        if (!action.succeeded()) {
            return fail(
                std::move(diagnostics),
                action.status,
                action.firstFailingArticulation
            );
        }
        for (std::uint32_t dof = 0u;
             dof < model.world.nv;
             ++dof) {
            nextVelocity[dof] += response[dof];
        }
    }

    diagnostics.integration.resize(
        model.articulations.size()
    );
    for (std::uint32_t articulationIndex = 0u;
         articulationIndex < model.articulations.size();
         ++articulationIndex) {
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        diagnostics.integration[articulationIndex] =
            integrateArticulatedConfiguration(
                model,
                articulationIndex,
                std::span<double>(candidateQ).subspan(
                    articulation.qOffset,
                    articulation.nq
                ),
                std::span<const double>(
                    nextVelocity
                ).subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                config.dynamics
            );
        if (!diagnostics.integration[
                articulationIndex
            ].succeeded()) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedWorldStatus::
                    integrationFailure,
                articulationIndex
            );
        }
    }
    if (!finite(candidateQ) || !finite(nextVelocity)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedWorldStatus::nonfiniteResult
        );
    }

    std::ranges::copy(candidateQ, q.begin());
    std::ranges::copy(nextVelocity, v.begin());
    cache = std::move(stagedCache);
    return diagnostics;
}

const char* multiArticulatedWorldStatusName(
    const MultiArticulatedWorldStatus status
) noexcept {
    switch (status) {
    case MultiArticulatedWorldStatus::success:
        return "success";
    case MultiArticulatedWorldStatus::invalidConfiguration:
        return "invalid_configuration";
    case MultiArticulatedWorldStatus::invalidModel:
        return "invalid_model";
    case MultiArticulatedWorldStatus::invalidDimensions:
        return "invalid_dimensions";
    case MultiArticulatedWorldStatus::unsupportedConstraint:
        return "unsupported_constraint";
    case MultiArticulatedWorldStatus::nonfiniteInput:
        return "nonfinite_input";
    case MultiArticulatedWorldStatus::freeDynamicsFailure:
        return "free_dynamics_failure";
    case MultiArticulatedWorldStatus::factorizationFailure:
        return "factorization_failure";
    case MultiArticulatedWorldStatus::constraintEvaluationFailure:
        return "constraint_evaluation_failure";
    case MultiArticulatedWorldStatus::solverFailure:
        return "solver_failure";
    case MultiArticulatedWorldStatus::didNotConverge:
        return "did_not_converge";
    case MultiArticulatedWorldStatus::integrationFailure:
        return "integration_failure";
    case MultiArticulatedWorldStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
