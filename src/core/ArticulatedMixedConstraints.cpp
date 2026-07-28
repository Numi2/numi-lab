#include "metalrobo/ArticulatedMixedConstraints.hpp"

#include <algorithm>
#include <array>
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

template <std::size_t Size>
bool finite(const std::array<double, Size>& values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

bool squareSize(
    const std::size_t dimension,
    const std::size_t size
) {
    return
        (dimension == 0u || dimension <=
            std::numeric_limits<std::size_t>::max() / dimension) &&
        size == dimension * dimension;
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

struct RetainedFactor {
    std::span<const double> lower;
    std::size_t dimension = 0u;
    bool valid = false;
};

RetainedFactor retainedFactor(
    const ArticulatedContactProblem& problem
) {
    RetainedFactor result;
    result.lower = problem.massCholeskyLower;
    result.dimension = problem.nv;
    if (result.dimension == 0u ||
        !squareSize(result.dimension, result.lower.size()) ||
        !finite(result.lower)) {
        return result;
    }
    for (std::size_t row = 0u; row < result.dimension; ++row) {
        const double pivot =
            result.lower[row * result.dimension + row];
        if (!(pivot > 0.0) || !finite(pivot)) {
            return result;
        }
        for (std::size_t column = row + 1u;
             column < result.dimension;
             ++column) {
            if (result.lower[row * result.dimension + column] !=
                0.0) {
                return result;
            }
        }
    }
    result.valid = true;
    return result;
}

bool solve(
    const RetainedFactor& factor,
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
    const RetainedFactor& factor,
    const std::span<const double> solution,
    const std::span<const double> right
) {
    if (!factor.valid ||
        solution.size() != factor.dimension ||
        right.size() != factor.dimension) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t dimension = factor.dimension;
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

bool validContactBlock(
    const DenseContactBlock& block,
    const std::size_t nv,
    const std::span<const double> packedJacobian,
    const std::size_t contactIndex
) {
    if (block.normalJacobian.size() != nv ||
        block.tangentUJacobian.size() != nv ||
        block.tangentVJacobian.size() != nv ||
        !finite(block.normalJacobian) ||
        !finite(block.tangentUJacobian) ||
        !finite(block.tangentVJacobian) ||
        !finite(block.targetVelocity) ||
        !finite(block.regularization) ||
        !finite(block.warmImpulse) ||
        !finite(block.friction) ||
        block.friction < 0.0 ||
        !std::ranges::all_of(
            block.regularization,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        )) {
        return false;
    }
    const std::array<std::span<const double>, 3> rows{
        block.normalJacobian,
        block.tangentUJacobian,
        block.tangentVJacobian,
    };
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const std::size_t packedRow = 3u * contactIndex + axis;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            if (rows[axis][dof] !=
                packedJacobian[packedRow * nv + dof]) {
                return false;
            }
        }
    }
    return true;
}

bool validContactProblem(
    const ArticulatedContactProblem& problem
) {
    const std::size_t nv = problem.nv;
    const std::size_t contactCount = problem.contactCount;
    if (contactCount >
            std::numeric_limits<std::size_t>::max() / 3u ||
        !retainedFactor(problem).valid ||
        problem.conic.nv != problem.nv ||
        problem.conic.freeVelocity.size() != nv ||
        !finite(problem.conic.freeVelocity) ||
        problem.conic.contacts.size() != contactCount ||
        problem.pointA.size() != contactCount ||
        problem.pointB.size() != contactCount) {
        return false;
    }
    const std::size_t contactRows = 3u * contactCount;
    if (contactRows != 0u &&
        nv > std::numeric_limits<std::size_t>::max() /
            contactRows) {
        return false;
    }
    if (problem.contactJacobian.size() != contactRows * nv ||
        !squareSize(contactRows, problem.delassus.size()) ||
        !finite(problem.contactJacobian) ||
        !finite(problem.delassus)) {
        return false;
    }
    for (std::size_t contact = 0u;
         contact < contactCount;
         ++contact) {
        if (!validContactBlock(
                problem.conic.contacts[contact],
                nv,
                problem.contactJacobian,
                contact
            )) {
            return false;
        }
    }
    for (std::size_t row = 0u; row < contactRows; ++row) {
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            const double left =
                problem.delassus[row * contactRows + column];
            const double right =
                problem.delassus[column * contactRows + row];
            const double scale =
                1.0 + std::max(std::abs(left), std::abs(right));
            if (std::abs(left - right) > 1.0e-12 * scale) {
                return false;
            }
        }
    }
    return true;
}

bool validLimitRows(
    const std::span<const ArticulatedJointLimitRow> rows,
    const std::span<const double> freeVelocity
) {
    std::uint64_t previousKey = 0u;
    std::uint32_t qOffset = 0u;
    std::uint32_t vOffset = 0u;
    bool hasPrevious = false;
    for (const ArticulatedJointLimitRow& row : rows) {
        const bool lower =
            row.side == ArticulatedJointLimitSide::lower;
        const bool upper =
            row.side == ArticulatedJointLimitSide::upper;
        const double expectedDirection = lower ? 1.0 : -1.0;
        const std::uint64_t expectedKey =
            2u * static_cast<std::uint64_t>(row.globalVIndex) +
            (upper ? 1u : 0u);
        if ((!lower && !upper) ||
            row.globalQIndex == MR_INVALID_INDEX ||
            row.globalVIndex == MR_INVALID_INDEX ||
            row.localQIndex == MR_INVALID_INDEX ||
            row.localVIndex >= freeVelocity.size() ||
            row.globalQIndex < row.localQIndex ||
            row.globalVIndex < row.localVIndex ||
            row.direction != expectedDirection ||
            row.stableKey != expectedKey ||
            (hasPrevious && row.stableKey <= previousKey) ||
            !finite(row.positionLimit) ||
            !finite(row.gap) ||
            !finite(row.freeNormalVelocity) ||
            !finite(row.targetVelocity) ||
            !(row.regularization > 0.0) ||
            !finite(row.regularization)) {
            return false;
        }
        const std::uint32_t rowQOffset =
            row.globalQIndex - row.localQIndex;
        const std::uint32_t rowVOffset =
            row.globalVIndex - row.localVIndex;
        if (hasPrevious &&
            (rowQOffset != qOffset || rowVOffset != vOffset)) {
            return false;
        }
        const double expectedVelocity =
            row.direction * freeVelocity[row.localVIndex];
        const double scale = 1.0 + std::max(
            std::abs(expectedVelocity),
            std::abs(row.freeNormalVelocity)
        );
        if (std::abs(
                expectedVelocity - row.freeNormalVelocity
            ) > 32.0 * std::numeric_limits<double>::epsilon() *
                scale) {
            return false;
        }
        qOffset = rowQOffset;
        vOffset = rowVOffset;
        previousKey = row.stableKey;
        hasPrevious = true;
    }
    return true;
}

ArticulatedMixedConstraintStatus statusForSolverCode(
    const MRStepStatusCode code
) {
    if (code == MR_STEP_DID_NOT_CONVERGE) {
        return ArticulatedMixedConstraintStatus::didNotConverge;
    }
    if (code == MR_STEP_FACTORIZATION_FAILED) {
        return
            ArticulatedMixedConstraintStatus::factorizationFailure;
    }
    return ArticulatedMixedConstraintStatus::solverFailure;
}

} // namespace

ArticulatedMixedConstraintSolution solveArticulatedMixedConstraints(
    const ArticulatedContactProblem& contactProblem,
    const std::span<const double> freeVelocity,
    const std::span<const ArticulatedJointLimitRow> limitRows,
    const QualityContactSolverConfig& config
) {
    ArticulatedMixedConstraintSolution result;
    ArticulatedMixedConstraintDiagnostics& diagnostics =
        result.diagnostics;
    diagnostics.articulationIndex =
        contactProblem.articulationIndex;
    diagnostics.nv = contactProblem.nv;
    diagnostics.contactCount = contactProblem.contactCount;
    diagnostics.limitRowCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(
            limitRows.size(),
            std::numeric_limits<std::uint32_t>::max()
        )
    );
    if (!validQualityConfig(config)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::
                invalidConfiguration;
        return result;
    }
    if (!validContactProblem(contactProblem)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::
                invalidContactProblem;
        return result;
    }
    if (freeVelocity.size() != contactProblem.nv ||
        limitRows.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::invalidDimensions;
        return result;
    }
    if (!finite(freeVelocity)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::nonfiniteInput;
        return result;
    }
    if (!validLimitRows(limitRows, freeVelocity)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::invalidLimitRow;
        return result;
    }

    const std::size_t nv = contactProblem.nv;
    const std::size_t contactCount =
        contactProblem.contactCount;
    const std::size_t contactRows = 3u * contactCount;
    if (limitRows.size() >
        std::numeric_limits<std::size_t>::max() -
            contactCount) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::invalidDimensions;
        return result;
    }
    const std::size_t blockCount =
        contactCount + limitRows.size();
    if (blockCount == 0u) {
        result.generalizedVelocity.assign(
            freeVelocity.begin(),
            freeVelocity.end()
        );
        return result;
    }
    if (blockCount >
        std::numeric_limits<std::size_t>::max() / 3u) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::invalidDimensions;
        return result;
    }
    const std::size_t fullDimension = 3u * blockCount;
    if (fullDimension >
        std::numeric_limits<std::size_t>::max() /
            fullDimension) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::invalidDimensions;
        return result;
    }

    ContactSpaceConicProblem mixed;
    mixed.delassus.assign(
        fullDimension * fullDimension,
        0.0
    );
    mixed.freeContactVelocity.assign(fullDimension, 0.0);
    mixed.contacts.resize(blockCount);

    for (std::size_t contact = 0u;
         contact < contactCount;
         ++contact) {
        const DenseContactBlock& source =
            contactProblem.conic.contacts[contact];
        mixed.contacts[contact] = {
            source.targetVelocity,
            source.regularization,
            source.warmImpulse,
            source.friction,
        };
    }
    for (std::size_t row = 0u; row < contactRows; ++row) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            mixed.freeContactVelocity[row] +=
                contactProblem.contactJacobian[row * nv + dof] *
                freeVelocity[dof];
        }
        for (std::size_t column = 0u;
             column < contactRows;
             ++column) {
            mixed.delassus[row * fullDimension + column] =
                contactProblem.delassus[
                    row * contactRows + column
                ];
        }
    }

    const RetainedFactor factor =
        retainedFactor(contactProblem);
    std::vector<std::vector<double>> limitResponses(
        limitRows.size()
    );
    std::vector<double> right(nv, 0.0);
    for (std::size_t limit = 0u;
         limit < limitRows.size();
         ++limit) {
        const ArticulatedJointLimitRow& row = limitRows[limit];
        const std::size_t mixedRow =
            3u * (contactCount + limit);
        mixed.freeContactVelocity[mixedRow] =
            row.direction * freeVelocity[row.localVIndex];
        ContactConicBlock& block =
            mixed.contacts[contactCount + limit];
        block.targetVelocity[0] = row.targetVelocity;
        block.regularization = {
            row.regularization,
            row.regularization,
            row.regularization,
        };
        block.friction = 0.0;

        std::ranges::fill(right, 0.0);
        right[row.localVIndex] = row.direction;
        if (!solve(factor, right, limitResponses[limit])) {
            diagnostics.status =
                ArticulatedMixedConstraintStatus::
                    factorizationFailure;
            return result;
        }
        for (std::size_t contactRow = 0u;
             contactRow < contactRows;
             ++contactRow) {
            double crossValue = 0.0;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                crossValue +=
                    contactProblem.contactJacobian[
                        contactRow * nv + dof
                    ] *
                    limitResponses[limit][dof];
            }
            if (!finite(crossValue)) {
                diagnostics.status =
                    ArticulatedMixedConstraintStatus::
                        nonfiniteResult;
                return result;
            }
            mixed.delassus[
                contactRow * fullDimension + mixedRow
            ] = crossValue;
            mixed.delassus[
                mixedRow * fullDimension + contactRow
            ] = crossValue;
            diagnostics.maximumCrossDelassusMagnitude =
                std::max(
                    diagnostics.maximumCrossDelassusMagnitude,
                    std::abs(crossValue)
                );
        }
    }
    for (std::size_t row = 0u;
         row < limitRows.size();
         ++row) {
        const std::size_t mixedRow =
            3u * (contactCount + row);
        const ArticulatedJointLimitRow& limitRow =
            limitRows[row];
        for (std::size_t column = 0u;
             column < limitRows.size();
             ++column) {
            const std::size_t mixedColumn =
                3u * (contactCount + column);
            mixed.delassus[
                mixedRow * fullDimension + mixedColumn
            ] =
                limitRow.direction *
                limitResponses[column][limitRow.localVIndex];
        }
    }
    for (std::size_t row = 0u;
         row < limitRows.size();
         ++row) {
        const std::size_t mixedRow =
            3u * (contactCount + row);
        for (std::size_t column = row + 1u;
             column < limitRows.size();
             ++column) {
            const std::size_t mixedColumn =
                3u * (contactCount + column);
            const double symmetric = 0.5 * (
                mixed.delassus[
                    mixedRow * fullDimension + mixedColumn
                ] +
                mixed.delassus[
                    mixedColumn * fullDimension + mixedRow
                ]
            );
            mixed.delassus[
                mixedRow * fullDimension + mixedColumn
            ] = symmetric;
            mixed.delassus[
                mixedColumn * fullDimension + mixedRow
            ] = symmetric;
        }
    }
    if (!finite(mixed.delassus) ||
        !finite(mixed.freeContactVelocity)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::nonfiniteResult;
        return result;
    }

    const QualityContactSolution quality =
        solveQualityContactSpaceProblem(mixed, config);
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
        return result;
    }
    if (quality.impulses.size() != fullDimension ||
        quality.velocity.size() != fullDimension ||
        !finite(quality.impulses) ||
        !finite(quality.velocity)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::nonfiniteResult;
        return result;
    }

    std::vector<double> generalizedImpulse(nv, 0.0);
    for (std::size_t row = 0u; row < contactRows; ++row) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            generalizedImpulse[dof] +=
                contactProblem.contactJacobian[row * nv + dof] *
                quality.impulses[row];
        }
    }
    for (std::size_t limit = 0u;
         limit < limitRows.size();
         ++limit) {
        const std::size_t mixedRow =
            3u * (contactCount + limit);
        generalizedImpulse[limitRows[limit].localVIndex] +=
            limitRows[limit].direction *
            quality.impulses[mixedRow];
        const double tangentMagnitude = std::max(
            std::abs(quality.impulses[mixedRow + 1u]),
            std::abs(quality.impulses[mixedRow + 2u])
        );
        if (tangentMagnitude >
            64.0 * std::numeric_limits<double>::epsilon()) {
            diagnostics.status =
                ArticulatedMixedConstraintStatus::
                    inconsistentOperator;
            return result;
        }
    }

    std::vector<double> deltaVelocity;
    if (!solve(factor, generalizedImpulse, deltaVelocity)) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::
                factorizationFailure;
        return result;
    }
    diagnostics.finalFactorApplications = 1u;
    diagnostics.maximumFactorSolveResidual =
        factorSolveResidual(
            factor,
            deltaVelocity,
            generalizedImpulse
        );
    if (!finite(diagnostics.maximumFactorSolveResidual) ||
        diagnostics.maximumFactorSolveResidual > 1.0e-10) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::
                inconsistentOperator;
        return result;
    }

    std::vector<double> generalizedVelocity(
        freeVelocity.begin(),
        freeVelocity.end()
    );
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        generalizedVelocity[dof] += deltaVelocity[dof];
    }
    std::vector<double> contactVelocity(contactRows, 0.0);
    for (std::size_t row = 0u; row < contactRows; ++row) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            contactVelocity[row] +=
                contactProblem.contactJacobian[row * nv + dof] *
                generalizedVelocity[dof];
        }
        diagnostics.maximumContactVelocityConsistencyError =
            std::max(
                diagnostics.maximumContactVelocityConsistencyError,
                std::abs(
                    contactVelocity[row] -
                    quality.velocity[row]
                )
            );
    }
    std::vector<double> limitVelocity(limitRows.size(), 0.0);
    for (std::size_t limit = 0u;
         limit < limitRows.size();
         ++limit) {
        const std::size_t mixedRow =
            3u * (contactCount + limit);
        limitVelocity[limit] =
            limitRows[limit].direction *
            generalizedVelocity[limitRows[limit].localVIndex];
        diagnostics.maximumLimitVelocityConsistencyError =
            std::max(
                diagnostics.maximumLimitVelocityConsistencyError,
                std::abs(
                    limitVelocity[limit] -
                    quality.velocity[mixedRow]
                )
            );
    }
    double velocityScale = 1.0;
    for (const double value : quality.velocity) {
        velocityScale = std::max(velocityScale, std::abs(value));
    }
    for (const double value : generalizedVelocity) {
        velocityScale = std::max(velocityScale, std::abs(value));
    }
    const double consistencyTolerance =
        8192.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(
            std::max({nv, fullDimension, std::size_t{1u}})
        ) *
        velocityScale;
    if (!finite(generalizedVelocity) ||
        !finite(contactVelocity) ||
        !finite(limitVelocity) ||
        diagnostics.maximumContactVelocityConsistencyError >
            consistencyTolerance ||
        diagnostics.maximumLimitVelocityConsistencyError >
            consistencyTolerance) {
        diagnostics.status =
            ArticulatedMixedConstraintStatus::
                inconsistentOperator;
        return result;
    }

    result.contactImpulses.assign(
        quality.impulses.begin(),
        quality.impulses.begin() +
            static_cast<std::ptrdiff_t>(contactRows)
    );
    result.limitImpulses.resize(limitRows.size(), 0.0);
    for (std::size_t limit = 0u;
         limit < limitRows.size();
         ++limit) {
        result.limitImpulses[limit] =
            quality.impulses[
                3u * (contactCount + limit)
            ];
    }
    result.contactVelocity = std::move(contactVelocity);
    result.limitVelocity = std::move(limitVelocity);
    result.generalizedVelocity =
        std::move(generalizedVelocity);
    return result;
}

const char* articulatedMixedConstraintStatusName(
    const ArticulatedMixedConstraintStatus status
) noexcept {
    switch (status) {
    case ArticulatedMixedConstraintStatus::success:
        return "success";
    case ArticulatedMixedConstraintStatus::invalidConfiguration:
        return "invalid_configuration";
    case ArticulatedMixedConstraintStatus::invalidDimensions:
        return "invalid_dimensions";
    case ArticulatedMixedConstraintStatus::invalidContactProblem:
        return "invalid_contact_problem";
    case ArticulatedMixedConstraintStatus::invalidLimitRow:
        return "invalid_limit_row";
    case ArticulatedMixedConstraintStatus::nonfiniteInput:
        return "nonfinite_input";
    case ArticulatedMixedConstraintStatus::factorizationFailure:
        return "factorization_failure";
    case ArticulatedMixedConstraintStatus::inconsistentOperator:
        return "inconsistent_operator";
    case ArticulatedMixedConstraintStatus::solverFailure:
        return "solver_failure";
    case ArticulatedMixedConstraintStatus::didNotConverge:
        return "did_not_converge";
    case ArticulatedMixedConstraintStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
