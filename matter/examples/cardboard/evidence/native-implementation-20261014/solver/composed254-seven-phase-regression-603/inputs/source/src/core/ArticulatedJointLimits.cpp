#include "metalrobo/ArticulatedJointLimits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
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

bool validQualityConfig(const QualityContactSolverConfig& config) {
    return
        config.maximumIterations > 0u &&
        config.maximumLineSearchIterations > 0u &&
        config.kktTolerance > 0.0 &&
        config.armijoCoefficient > 0.0 &&
        config.armijoCoefficient < 0.5 &&
        config.factorizationPivotTolerance > 0.0 &&
        config.gaussNewtonRegularization > 0.0 &&
        finite(config.kktTolerance) &&
        finite(config.armijoCoefficient) &&
        finite(config.factorizationPivotTolerance) &&
        finite(config.gaussNewtonRegularization);
}

bool validConfig(const ArticulatedJointLimitConfig& config) {
    return
        config.timestep > 0.0 &&
        config.activationDistance >= 0.0 &&
        config.positionSlop >= 0.0 &&
        config.recoveryFraction >= 0.0 &&
        config.recoveryFraction <= 1.0 &&
        config.maximumRecoverySpeed >= 0.0 &&
        config.regularization > 0.0 &&
        config.maximumRows > 0u &&
        finite(config.timestep) &&
        finite(config.activationDistance) &&
        finite(config.positionSlop) &&
        finite(config.recoveryFraction) &&
        finite(config.maximumRecoverySpeed) &&
        finite(config.regularization) &&
        validQualityConfig(config.quality);
}

ArticulatedJointLimitDiagnostics diagnosticsFor(
    const std::uint32_t articulationIndex,
    const std::size_t dofCount = 0u,
    const std::size_t rowCount = 0u
) {
    ArticulatedJointLimitDiagnostics result;
    result.articulationIndex = articulationIndex;
    result.dofCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(
            dofCount,
            std::numeric_limits<std::uint32_t>::max()
        )
    );
    result.rowCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(
            rowCount,
            std::numeric_limits<std::uint32_t>::max()
        )
    );
    return result;
}

void summarizeRows(
    const ArticulatedJointLimitProblem& problem,
    ArticulatedJointLimitDiagnostics& diagnostics
) {
    diagnostics.lowerRowCount = 0u;
    diagnostics.upperRowCount = 0u;
    diagnostics.penetratingRowCount = 0u;
    diagnostics.minimumGap = 0.0;
    diagnostics.maximumPenetration = 0.0;
    bool first = true;
    for (const ArticulatedJointLimitRow& row : problem.rows) {
        if (row.side == ArticulatedJointLimitSide::lower) {
            ++diagnostics.lowerRowCount;
        } else {
            ++diagnostics.upperRowCount;
        }
        if (first) {
            diagnostics.minimumGap = row.gap;
            first = false;
        } else {
            diagnostics.minimumGap =
                std::min(diagnostics.minimumGap, row.gap);
        }
        if (row.gap < 0.0) {
            ++diagnostics.penetratingRowCount;
            diagnostics.maximumPenetration = std::max(
                diagnostics.maximumPenetration,
                -row.gap
            );
        }
    }
}

struct CholeskyFactor {
    std::vector<double> lower;
    std::size_t dimension = 0u;
    double minimumPivot = 0.0;
    double maximumPivot = 0.0;
    bool valid = false;
};

CholeskyFactor factorize(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    CholeskyFactor result;
    result.dimension = dimension;
    if (dimension == 0u ||
        matrix.size() != dimension * dimension ||
        !finite(matrix)) {
        return result;
    }
    double maximumDiagonal = 0.0;
    for (std::size_t index = 0u; index < dimension; ++index) {
        maximumDiagonal = std::max(
            maximumDiagonal,
            std::abs(matrix[index * dimension + index])
        );
    }
    const double pivotFloor =
        std::max(1.0, maximumDiagonal) * 1.0e-13;
    result.lower.assign(dimension * dimension, 0.0);
    result.minimumPivot = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * dimension + column];
            for (std::size_t inner = 0u;
                 inner < column;
                 ++inner) {
                value -=
                    result.lower[row * dimension + inner] *
                    result.lower[column * dimension + inner];
            }
            if (row == column) {
                if (!(value > pivotFloor) || !finite(value)) {
                    return result;
                }
                const double pivot = std::sqrt(value);
                result.lower[row * dimension + column] = pivot;
                result.minimumPivot =
                    std::min(result.minimumPivot, pivot);
                result.maximumPivot =
                    std::max(result.maximumPivot, pivot);
            } else {
                result.lower[row * dimension + column] =
                    value /
                    result.lower[column * dimension + column];
            }
        }
    }
    result.valid =
        finite(result.lower) &&
        finite(result.minimumPivot) &&
        finite(result.maximumPivot);
    return result;
}

CholeskyFactor retainedFactor(
    const ArticulatedJointLimitProblem& problem
) {
    CholeskyFactor result;
    result.dimension = problem.nv;
    result.lower = problem.massCholeskyLower;
    result.minimumPivot = std::numeric_limits<double>::infinity();
    result.maximumPivot = 0.0;
    if (result.dimension == 0u ||
        result.lower.size() !=
            result.dimension * result.dimension ||
        !finite(result.lower)) {
        return result;
    }
    for (std::size_t row = 0u; row < result.dimension; ++row) {
        for (std::size_t column = row + 1u;
             column < result.dimension;
             ++column) {
            if (result.lower[row * result.dimension + column] != 0.0) {
                return result;
            }
        }
        const double pivot =
            result.lower[row * result.dimension + row];
        if (!(pivot > 0.0) || !finite(pivot)) {
            return result;
        }
        result.minimumPivot = std::min(result.minimumPivot, pivot);
        result.maximumPivot = std::max(result.maximumPivot, pivot);
    }
    result.valid = true;
    return result;
}

bool solve(
    const CholeskyFactor& factor,
    const std::span<const double> right,
    std::vector<double>& solution
) {
    if (!factor.valid ||
        right.size() != factor.dimension ||
        !finite(right)) {
        return false;
    }
    const std::size_t dimension = factor.dimension;
    std::vector<double> intermediate(dimension, 0.0);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = right[row];
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            value -=
                factor.lower[row * dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value / factor.lower[row * dimension + row];
    }
    solution.assign(dimension, 0.0);
    for (std::size_t reverse = 0u;
         reverse < dimension;
         ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                factor.lower[column * dimension + row] *
                solution[column];
        }
        solution[row] =
            value / factor.lower[row * dimension + row];
    }
    return finite(solution);
}

double factorSolveResidual(
    const CholeskyFactor& factor,
    const std::span<const double> solution,
    const std::span<const double> right
) {
    const std::size_t dimension = factor.dimension;
    if (!factor.valid ||
        solution.size() != dimension ||
        right.size() != dimension) {
        return std::numeric_limits<double>::infinity();
    }
    std::vector<double> transposeAction(dimension, 0.0);
    for (std::size_t column = 0u;
         column < dimension;
         ++column) {
        for (std::size_t row = column;
             row < dimension;
             ++row) {
            transposeAction[column] +=
                factor.lower[row * dimension + column] *
                solution[row];
        }
    }
    double maximumResidual = 0.0;
    double scale = 1.0;
    for (std::size_t row = 0u; row < dimension; ++row) {
        double action = 0.0;
        for (std::size_t column = 0u;
             column <= row;
             ++column) {
            action +=
                factor.lower[row * dimension + column] *
                transposeAction[column];
        }
        maximumResidual = std::max(
            maximumResidual,
            std::abs(action - right[row])
        );
        scale = std::max({
            scale,
            std::abs(action),
            std::abs(right[row]),
        });
    }
    return maximumResidual / scale;
}

bool validRow(
    const ArticulatedJointLimitRow& row,
    const std::size_t nv
) {
    const bool lower =
        row.side == ArticulatedJointLimitSide::lower;
    const bool upper =
        row.side == ArticulatedJointLimitSide::upper;
    return
        (lower || upper) &&
        row.globalQIndex != MR_INVALID_INDEX &&
        row.globalVIndex != MR_INVALID_INDEX &&
        row.localQIndex != MR_INVALID_INDEX &&
        row.localVIndex < nv &&
        row.direction == (lower ? 1.0 : -1.0) &&
        finite(row.positionLimit) &&
        finite(row.gap) &&
        finite(row.freeNormalVelocity) &&
        finite(row.targetVelocity) &&
        row.regularization > 0.0 &&
        finite(row.regularization);
}

bool structurallyValid(
    const ArticulatedJointLimitProblem& problem
) {
    const std::size_t nv = problem.nv;
    const std::size_t rowCount = problem.rows.size();
    if (nv == 0u ||
        problem.freeVelocity.size() != nv ||
        problem.delassus.size() != rowCount * rowCount ||
        !finite(problem.freeVelocity) ||
        !finite(problem.delassus)) {
        return false;
    }
    if (rowCount == 0u) {
        return problem.massCholeskyLower.empty();
    }
    if (problem.massCholeskyLower.size() != nv * nv ||
        !retainedFactor(problem).valid) {
        return false;
    }
    std::uint64_t previousKey = 0u;
    bool hasPrevious = false;
    for (const ArticulatedJointLimitRow& row : problem.rows) {
        if (!validRow(row, nv) ||
            (hasPrevious && row.stableKey <= previousKey) ||
            row.freeNormalVelocity !=
                row.direction *
                    problem.freeVelocity[row.localVIndex]) {
            return false;
        }
        previousKey = row.stableKey;
        hasPrevious = true;
    }
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            const double left =
                problem.delassus[row * rowCount + column];
            const double right =
                problem.delassus[column * rowCount + row];
            const double scale =
                1.0 + std::max(std::abs(left), std::abs(right));
            if (std::abs(left - right) > 1.0e-12 * scale) {
                return false;
            }
        }
    }
    return true;
}

double targetVelocity(
    const double gap,
    const ArticulatedJointLimitConfig& config
) {
    if (gap < -config.positionSlop) {
        return std::min(
            config.maximumRecoverySpeed,
            config.recoveryFraction *
                (-gap - config.positionSlop) /
                config.timestep
        );
    }
    return
        -std::max(gap - config.positionSlop, 0.0) /
        config.timestep;
}

bool rowActive(
    const double gap,
    const double normalVelocity,
    const ArticulatedJointLimitConfig& config
) {
    const double predictedGap =
        gap + config.timestep * normalVelocity;
    return
        gap <= config.activationDistance ||
        predictedGap <= config.activationDistance;
}

void addRow(
    const MRDofPropertiesGPU& dof,
    const std::uint32_t localQ,
    const std::uint32_t localV,
    const ArticulatedJointLimitSide side,
    const double position,
    const double velocity,
    const ArticulatedJointLimitConfig& config,
    std::vector<ArticulatedJointLimitRow>& rows,
    ArticulatedJointLimitDiagnostics& diagnostics
) {
    const bool lower = side == ArticulatedJointLimitSide::lower;
    const double direction = lower ? 1.0 : -1.0;
    const double limit = lower
        ? static_cast<double>(dof.limits.x)
        : static_cast<double>(dof.limits.y);
    const double gap = lower
        ? position - limit
        : limit - position;
    const double normalVelocity = direction * velocity;
    if (!rowActive(gap, normalVelocity, config)) {
        return;
    }
    ArticulatedJointLimitRow row;
    row.stableKey =
        2u * static_cast<std::uint64_t>(dof.vIndex) +
        (lower ? 0u : 1u);
    row.globalQIndex = dof.qIndex;
    row.globalVIndex = dof.vIndex;
    row.localQIndex = localQ;
    row.localVIndex = localV;
    row.side = side;
    row.direction = direction;
    row.positionLimit = limit;
    row.gap = gap;
    row.freeNormalVelocity = normalVelocity;
    row.targetVelocity = targetVelocity(gap, config);
    row.regularization = config.regularization;
    rows.push_back(row);

    if (lower) {
        ++diagnostics.lowerRowCount;
    } else {
        ++diagnostics.upperRowCount;
    }
    if (gap < 0.0) {
        ++diagnostics.penetratingRowCount;
        diagnostics.maximumPenetration = std::max(
            diagnostics.maximumPenetration,
            -gap
        );
    }
    if (diagnostics.rowCount == 0u) {
        diagnostics.minimumGap = gap;
    } else {
        diagnostics.minimumGap =
            std::min(diagnostics.minimumGap, gap);
    }
    ++diagnostics.rowCount;
}

ArticulatedJointLimitStatus statusForSolverCode(
    const MRStepStatusCode code
) {
    if (code == MR_STEP_DID_NOT_CONVERGE) {
        return ArticulatedJointLimitStatus::didNotConverge;
    }
    if (code == MR_STEP_FACTORIZATION_FAILED) {
        return ArticulatedJointLimitStatus::factorizationFailure;
    }
    return ArticulatedJointLimitStatus::solverFailure;
}

} // namespace

ArticulatedJointLimitDiagnostics
compileArticulatedJointLimitRows(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> freeVelocity,
    std::vector<ArticulatedJointLimitRow>& rows,
    const ArticulatedJointLimitConfig& config
) {
    ArticulatedJointLimitDiagnostics diagnostics =
        diagnosticsFor(articulationIndex);
    if (!validConfig(config)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidConfiguration;
        return diagnostics;
    }
    if (articulationIndex >= model.articulations.size()) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidArticulation;
        return diagnostics;
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    diagnostics.dofCount = articulation.nv;
    if (q.size() != articulation.nq ||
        freeVelocity.size() != articulation.nv) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(q) || !finite(freeVelocity)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::nonfiniteInput;
        return diagnostics;
    }

    const std::size_t nv = articulation.nv;
    std::vector<ArticulatedJointLimitRow> staged;
    staged.reserve(
        std::min<std::size_t>(2u * nv, config.maximumRows)
    );
    for (std::uint32_t localV = 0u;
         localV < articulation.nv;
         ++localV) {
        const std::uint32_t globalV =
            articulation.vOffset + localV;
        if (globalV >= model.dofs.size()) {
            diagnostics.status =
                ArticulatedJointLimitStatus::invalidDofMetadata;
            return diagnostics;
        }
        const MRDofPropertiesGPU& dof = model.dofs[globalV];
        if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u) {
            continue;
        }
        if (dof.articulationIndex != articulationIndex ||
            dof.vIndex != globalV ||
            dof.qIndex == MR_INVALID_INDEX ||
            dof.qIndex < articulation.qOffset ||
            dof.qIndex - articulation.qOffset >= articulation.nq ||
            !finite(static_cast<double>(dof.limits.x)) ||
            !finite(static_cast<double>(dof.limits.y)) ||
            dof.limits.x > dof.limits.y) {
            diagnostics.status =
                ArticulatedJointLimitStatus::invalidDofMetadata;
            return diagnostics;
        }
        const std::uint32_t localQ =
            dof.qIndex - articulation.qOffset;
        addRow(
            dof,
            localQ,
            localV,
            ArticulatedJointLimitSide::lower,
            q[localQ],
            freeVelocity[localV],
            config,
            staged,
            diagnostics
        );
        addRow(
            dof,
            localQ,
            localV,
            ArticulatedJointLimitSide::upper,
            q[localQ],
            freeVelocity[localV],
            config,
            staged,
            diagnostics
        );
        if (staged.size() > config.maximumRows) {
            diagnostics.status =
                ArticulatedJointLimitStatus::capacityExceeded;
            return diagnostics;
        }
    }
    rows = std::move(staged);
    return diagnostics;
}

ArticulatedJointLimitDiagnostics
buildArticulatedJointLimitProblem(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> freeVelocity,
    ArticulatedJointLimitProblem& problem,
    const ArticulatedJointLimitConfig& config,
    const ArticulatedDynamicsConfig& dynamicsConfig
) {
    std::vector<ArticulatedJointLimitRow> rows;
    ArticulatedJointLimitDiagnostics diagnostics =
        compileArticulatedJointLimitRows(
            model,
            articulationIndex,
            q,
            freeVelocity,
            rows,
            config
        );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }

    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const std::size_t nv = articulation.nv;
    ArticulatedJointLimitProblem staged;
    staged.articulationIndex = articulationIndex;
    staged.nv = articulation.nv;
    staged.freeVelocity.assign(
        freeVelocity.begin(),
        freeVelocity.end()
    );
    staged.rows = std::move(rows);
    if (staged.rows.empty()) {
        problem = std::move(staged);
        return diagnostics;
    }

    std::vector<double> massMatrix(nv * nv, 0.0);
    const ArticulatedDynamicsDiagnostics massDiagnostics =
        computeArticulatedMassMatrix(
            model,
            articulationIndex,
            q,
            massMatrix,
            dynamicsConfig
        );
    diagnostics.dynamicsStatus = massDiagnostics.status;
    if (!massDiagnostics.succeeded()) {
        diagnostics.status =
            massDiagnostics.status ==
                ArticulatedDynamicsStatus::
                    massMatrixNotPositiveDefinite
            ? ArticulatedJointLimitStatus::factorizationFailure
            : ArticulatedJointLimitStatus::dynamicsFailure;
        return diagnostics;
    }

    const CholeskyFactor factor = factorize(massMatrix, nv);
    if (!factor.valid) {
        diagnostics.status =
            ArticulatedJointLimitStatus::factorizationFailure;
        return diagnostics;
    }
    diagnostics.minimumCholeskyPivot = factor.minimumPivot;
    diagnostics.maximumCholeskyPivot = factor.maximumPivot;
    staged.massCholeskyLower = factor.lower;

    const std::size_t rowCount = staged.rows.size();
    staged.delassus.assign(rowCount * rowCount, 0.0);
    std::vector<std::vector<double>> responses(
        rowCount,
        std::vector<double>(nv, 0.0)
    );
    for (std::size_t column = 0u;
         column < rowCount;
         ++column) {
        std::vector<double> generalizedImpulse(nv, 0.0);
        const ArticulatedJointLimitRow& row = staged.rows[column];
        generalizedImpulse[row.localVIndex] = row.direction;
        if (!solve(
                factor,
                generalizedImpulse,
                responses[column]
            )) {
            diagnostics.status =
                ArticulatedJointLimitStatus::factorizationFailure;
            return diagnostics;
        }
    }
    for (std::size_t rowIndex = 0u;
         rowIndex < rowCount;
         ++rowIndex) {
        const ArticulatedJointLimitRow& row =
            staged.rows[rowIndex];
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            staged.delassus[rowIndex * rowCount + column] =
                row.direction *
                responses[column][row.localVIndex];
        }
    }
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t column = row + 1u;
             column < rowCount;
             ++column) {
            const double left =
                staged.delassus[row * rowCount + column];
            const double right =
                staged.delassus[column * rowCount + row];
            diagnostics.maximumDelassusAsymmetry = std::max(
                diagnostics.maximumDelassusAsymmetry,
                std::abs(left - right)
            );
            const double symmetric = 0.5 * (left + right);
            staged.delassus[row * rowCount + column] = symmetric;
            staged.delassus[column * rowCount + row] = symmetric;
        }
    }
    if (!structurallyValid(staged)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::nonfiniteResult;
        return diagnostics;
    }
    problem = std::move(staged);
    return diagnostics;
}

ArticulatedJointLimitDiagnostics
applyArticulatedJointLimitImpulses(
    const ArticulatedJointLimitProblem& problem,
    const std::span<const double> impulses,
    const std::span<double> generalizedVelocity
) {
    ArticulatedJointLimitDiagnostics diagnostics =
        diagnosticsFor(
            problem.articulationIndex,
            problem.nv,
            problem.rows.size()
        );
    summarizeRows(problem, diagnostics);
    if (!structurallyValid(problem)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidProblem;
        return diagnostics;
    }
    if (impulses.size() != problem.rows.size() ||
        generalizedVelocity.size() != problem.nv) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(impulses) || !finite(generalizedVelocity)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::nonfiniteInput;
        return diagnostics;
    }
    if (std::ranges::any_of(impulses, [](const double impulse) {
            return impulse < 0.0;
        })) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidImpulse;
        return diagnostics;
    }
    if (problem.rows.empty()) {
        return diagnostics;
    }

    std::vector<double> generalizedImpulse(problem.nv, 0.0);
    for (std::size_t row = 0u;
         row < problem.rows.size();
         ++row) {
        const ArticulatedJointLimitRow& limit = problem.rows[row];
        generalizedImpulse[limit.localVIndex] +=
            limit.direction * impulses[row];
        diagnostics.maximumImpulse = std::max(
            diagnostics.maximumImpulse,
            impulses[row]
        );
    }
    const CholeskyFactor factor = retainedFactor(problem);
    std::vector<double> deltaVelocity;
    if (!solve(factor, generalizedImpulse, deltaVelocity)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::factorizationFailure;
        return diagnostics;
    }
    diagnostics.minimumCholeskyPivot = factor.minimumPivot;
    diagnostics.maximumCholeskyPivot = factor.maximumPivot;
    diagnostics.maximumFactorSolveResidual = factorSolveResidual(
        factor,
        deltaVelocity,
        generalizedImpulse
    );
    std::vector<double> candidate(
        generalizedVelocity.begin(),
        generalizedVelocity.end()
    );
    for (std::size_t dof = 0u; dof < problem.nv; ++dof) {
        candidate[dof] += deltaVelocity[dof];
    }
    if (!finite(candidate) ||
        !finite(diagnostics.maximumFactorSolveResidual)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::nonfiniteResult;
        return diagnostics;
    }
    std::ranges::copy(candidate, generalizedVelocity.begin());
    return diagnostics;
}

ArticulatedJointLimitSolution solveArticulatedJointLimits(
    const ArticulatedJointLimitProblem& problem,
    const ArticulatedJointLimitConfig& config,
    const std::span<const double> warmImpulses
) {
    ArticulatedJointLimitSolution solution;
    solution.diagnostics = diagnosticsFor(
        problem.articulationIndex,
        problem.nv,
        problem.rows.size()
    );
    ArticulatedJointLimitDiagnostics& diagnostics =
        solution.diagnostics;
    summarizeRows(problem, diagnostics);
    if (!validConfig(config)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidConfiguration;
        return solution;
    }
    if (!structurallyValid(problem)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidProblem;
        return solution;
    }
    const std::size_t rowCount = problem.rows.size();
    if (!warmImpulses.empty() &&
        warmImpulses.size() != rowCount) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidDimensions;
        return solution;
    }
    if ((!warmImpulses.empty() && !finite(warmImpulses)) ||
        std::ranges::any_of(
            warmImpulses,
            [](const double impulse) {
                return impulse < 0.0;
            }
        )) {
        diagnostics.status =
            ArticulatedJointLimitStatus::invalidWarmStart;
        return solution;
    }
    if (rowCount == 0u) {
        solution.generalizedVelocity = problem.freeVelocity;
        return solution;
    }

    const std::size_t fullDimension = 3u * rowCount;
    ContactSpaceConicProblem contactSpace;
    contactSpace.delassus.assign(
        fullDimension * fullDimension,
        0.0
    );
    contactSpace.freeContactVelocity.assign(
        fullDimension,
        0.0
    );
    contactSpace.contacts.resize(rowCount);
    for (std::size_t row = 0u; row < rowCount; ++row) {
        contactSpace.freeContactVelocity[3u * row] =
            problem.rows[row].freeNormalVelocity;
        ContactConicBlock& block = contactSpace.contacts[row];
        block.targetVelocity[0] =
            problem.rows[row].targetVelocity;
        block.regularization = {
            problem.rows[row].regularization,
            problem.rows[row].regularization,
            problem.rows[row].regularization,
        };
        block.friction = 0.0;
        if (!warmImpulses.empty()) {
            block.warmImpulse[0] = warmImpulses[row];
        }
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            contactSpace.delassus[
                (3u * row) * fullDimension +
                3u * column
            ] = problem.delassus[row * rowCount + column];
        }
    }

    const QualityContactSolution quality =
        solveQualityContactSpaceProblem(
            contactSpace,
            config.quality
        );
    diagnostics.solverCode = quality.code;
    diagnostics.iterations = quality.iterations;
    diagnostics.semismoothNewtonSteps =
        quality.semismoothNewtonSteps;
    diagnostics.fallbackSteps =
        quality.gaussNewtonFallbackSteps +
        quality.projectedGradientFallbackSteps;
    diagnostics.scaledKktCertificate =
        quality.scaledKktCertificate;
    if (!quality.converged()) {
        diagnostics.status = statusForSolverCode(quality.code);
        return solution;
    }
    if (quality.impulses.size() != fullDimension ||
        quality.velocity.size() != fullDimension ||
        !finite(quality.impulses) ||
        !finite(quality.velocity)) {
        diagnostics.status =
            ArticulatedJointLimitStatus::nonfiniteResult;
        return solution;
    }

    solution.impulses.assign(rowCount, 0.0);
    solution.constraintVelocity.assign(rowCount, 0.0);
    for (std::size_t row = 0u; row < rowCount; ++row) {
        solution.impulses[row] = quality.impulses[3u * row];
        solution.constraintVelocity[row] =
            quality.velocity[3u * row];
        diagnostics.maximumImpulse = std::max(
            diagnostics.maximumImpulse,
            solution.impulses[row]
        );
        const double dual =
            solution.constraintVelocity[row] -
            problem.rows[row].targetVelocity +
            problem.rows[row].regularization *
                solution.impulses[row];
        diagnostics.maximumPhysicalVelocityViolation = std::max(
            diagnostics.maximumPhysicalVelocityViolation,
            std::max(
                problem.rows[row].targetVelocity -
                    solution.constraintVelocity[row],
                0.0
            )
        );
        diagnostics.maximumDualViolation = std::max(
            diagnostics.maximumDualViolation,
            std::max(-dual, 0.0)
        );
        diagnostics.maximumComplementarityResidual = std::max(
            diagnostics.maximumComplementarityResidual,
            std::abs(solution.impulses[row] * dual)
        );
    }

    solution.generalizedVelocity = problem.freeVelocity;
    const ArticulatedJointLimitDiagnostics applyDiagnostics =
        applyArticulatedJointLimitImpulses(
            problem,
            solution.impulses,
            solution.generalizedVelocity
        );
    if (!applyDiagnostics.succeeded()) {
        diagnostics.status = applyDiagnostics.status;
        diagnostics.maximumFactorSolveResidual =
            applyDiagnostics.maximumFactorSolveResidual;
        solution.impulses.clear();
        solution.constraintVelocity.clear();
        solution.generalizedVelocity.clear();
        return solution;
    }
    diagnostics.minimumCholeskyPivot =
        applyDiagnostics.minimumCholeskyPivot;
    diagnostics.maximumCholeskyPivot =
        applyDiagnostics.maximumCholeskyPivot;
    diagnostics.maximumFactorSolveResidual =
        applyDiagnostics.maximumFactorSolveResidual;

    for (std::size_t row = 0u; row < rowCount; ++row) {
        const ArticulatedJointLimitRow& limit = problem.rows[row];
        const double directVelocity =
            limit.direction *
            solution.generalizedVelocity[limit.localVIndex];
        const double scale = 1.0 + std::max(
            std::abs(directVelocity),
            std::abs(solution.constraintVelocity[row])
        );
        if (!finite(directVelocity) ||
            std::abs(
                directVelocity -
                solution.constraintVelocity[row]
            ) > 2.0e-11 * scale) {
            diagnostics.status =
                ArticulatedJointLimitStatus::nonfiniteResult;
            solution.impulses.clear();
            solution.constraintVelocity.clear();
            solution.generalizedVelocity.clear();
            return solution;
        }
    }
    return solution;
}

const char* articulatedJointLimitStatusName(
    const ArticulatedJointLimitStatus status
) noexcept {
    switch (status) {
    case ArticulatedJointLimitStatus::success:
        return "success";
    case ArticulatedJointLimitStatus::invalidConfiguration:
        return "invalid_configuration";
    case ArticulatedJointLimitStatus::invalidArticulation:
        return "invalid_articulation";
    case ArticulatedJointLimitStatus::invalidDimensions:
        return "invalid_dimensions";
    case ArticulatedJointLimitStatus::nonfiniteInput:
        return "nonfinite_input";
    case ArticulatedJointLimitStatus::invalidDofMetadata:
        return "invalid_dof_metadata";
    case ArticulatedJointLimitStatus::capacityExceeded:
        return "capacity_exceeded";
    case ArticulatedJointLimitStatus::dynamicsFailure:
        return "dynamics_failure";
    case ArticulatedJointLimitStatus::factorizationFailure:
        return "factorization_failure";
    case ArticulatedJointLimitStatus::invalidProblem:
        return "invalid_problem";
    case ArticulatedJointLimitStatus::invalidWarmStart:
        return "invalid_warm_start";
    case ArticulatedJointLimitStatus::invalidImpulse:
        return "invalid_impulse";
    case ArticulatedJointLimitStatus::solverFailure:
        return "solver_failure";
    case ArticulatedJointLimitStatus::didNotConverge:
        return "did_not_converge";
    case ArticulatedJointLimitStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
