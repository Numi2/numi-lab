#include "metalrobo/QualityContactSolver.hpp"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct ConeBlock {
    std::size_t offset = 0u;
    std::size_t dimension = 0u;
    std::size_t contact = 0u;
};

struct CompactVariable {
    std::size_t fullIndex = 0u;
    double lambdaScale = 1.0;
};

struct PreparedProblem {
    std::size_t nv = 0u;
    std::size_t fullDimension = 0u;
    std::size_t dimension = 0u;
    std::vector<double> jacobian;
    std::vector<double> hessian;
    std::vector<double> linear;
    std::vector<double> compactHessian;
    std::vector<double> compactLinear;
    std::vector<double> compactWarm;
    std::vector<ConeBlock> blocks;
    std::vector<CompactVariable> variables;
    double lipschitz = 0.0;
};

double dot(
    const std::span<const double> left,
    const std::span<const double> right
) {
    double value = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        value += left[index] * right[index];
    }
    return value;
}

double norm(const std::span<const double> value) {
    return std::sqrt(dot(value, value));
}

bool finite(const std::span<const double> values) {
    return std::ranges::all_of(values, [](const double value) {
        return std::isfinite(value);
    });
}

std::vector<double> multiply(
    const std::span<const double> matrix,
    const std::size_t rows,
    const std::size_t columns,
    const std::span<const double> vector
) {
    std::vector<double> result(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row] +=
                matrix[row * columns + column] * vector[column];
        }
    }
    return result;
}

double objective(
    const std::span<const double> hessian,
    const std::span<const double> linear,
    const std::span<const double> value
) {
    const std::vector<double> product =
        multiply(hessian, value.size(), value.size(), value);
    return 0.5 * dot(value, product) + dot(linear, value);
}

bool positiveDefinite(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    if (dimension == 0u ||
        matrix.size() != dimension * dimension ||
        !finite(matrix)) {
        return false;
    }
    double matrixScale = 0.0;
    for (const double value : matrix) {
        matrixScale = std::max(matrixScale, std::abs(value));
    }
    if (!(matrixScale > 0.0) || !std::isfinite(matrixScale)) {
        return false;
    }
    const double pivotThreshold =
        64.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(dimension) * matrixScale;
    std::vector<double> factor(matrix.begin(), matrix.end());
    for (std::size_t column = 0; column < dimension; ++column) {
        for (std::size_t row = column; row < dimension; ++row) {
            double value = factor[row * dimension + column];
            for (std::size_t inner = 0; inner < column; ++inner) {
                value -= factor[row * dimension + inner] *
                    factor[column * dimension + inner];
            }
            if (row == column) {
                if (!(value > pivotThreshold) ||
                    !std::isfinite(value)) {
                    return false;
                }
                factor[row * dimension + column] = std::sqrt(value);
            } else {
                factor[row * dimension + column] =
                    value / factor[column * dimension + column];
            }
        }
    }
    return true;
}

bool positiveSemidefinite(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    if (dimension == 0u ||
        matrix.size() != dimension * dimension ||
        !finite(matrix)) {
        return false;
    }
    double matrixScale = 0.0;
    for (const double value : matrix) {
        matrixScale = std::max(matrixScale, std::abs(value));
    }
    if (matrixScale == 0.0) {
        return true;
    }
    if (!std::isfinite(matrixScale)) {
        return false;
    }

    // Symmetric diagonal-pivoted Cholesky/Schur validation. Pivoting avoids
    // treating a harmless zero leading diagonal as a factorization failure,
    // while the terminal Schur-block check rejects indefinite off-diagonal
    // structure that positive contact regularization could otherwise mask.
    const double tolerance =
        256.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(dimension) * matrixScale;
    const auto minimumEigenvalue = [&matrix, dimension](
        double& minimum
    ) {
        if (dimension >
            static_cast<std::size_t>(
                std::numeric_limits<__LAPACK_int>::max()
            )) {
            return false;
        }
        const __LAPACK_int order =
            static_cast<__LAPACK_int>(dimension);
        const __LAPACK_int leadingDimension = order;
        const char noVectors = 'N';
        const char upperTriangle = 'U';
        std::vector<double> working(matrix.begin(), matrix.end());
        std::vector<double> eigenvalues(dimension, 0.0);
        double workspaceQuery = 0.0;
        __LAPACK_int workspaceSize = -1;
        __LAPACK_int info = 0;
        dsyev_(
            &noVectors,
            &upperTriangle,
            &order,
            working.data(),
            &leadingDimension,
            eigenvalues.data(),
            &workspaceQuery,
            &workspaceSize,
            &info
        );
        if (info != 0 ||
            !std::isfinite(workspaceQuery) ||
            workspaceQuery < 1.0 ||
            workspaceQuery >
                static_cast<double>(
                    std::numeric_limits<__LAPACK_int>::max()
                )) {
            return false;
        }
        workspaceSize = static_cast<__LAPACK_int>(
            std::ceil(workspaceQuery)
        );
        std::vector<double> workspace(
            static_cast<std::size_t>(workspaceSize),
            0.0
        );
        dsyev_(
            &noVectors,
            &upperTriangle,
            &order,
            working.data(),
            &leadingDimension,
            eigenvalues.data(),
            workspace.data(),
            &workspaceSize,
            &info
        );
        if (info != 0 || !finite(eigenvalues)) {
            return false;
        }
        minimum = eigenvalues.front();
        return std::isfinite(minimum);
    };

    std::vector<double> schur(matrix.begin(), matrix.end());
    std::vector<double> column(dimension, 0.0);
    for (std::size_t pivotIndex = 0u;
         pivotIndex < dimension;
         ++pivotIndex) {
        std::size_t selected = pivotIndex;
        for (std::size_t candidate = pivotIndex + 1u;
             candidate < dimension;
             ++candidate) {
            if (schur[candidate * dimension + candidate] >
                schur[selected * dimension + selected]) {
                selected = candidate;
            }
        }
        if (selected != pivotIndex) {
            for (std::size_t columnIndex = 0u;
                 columnIndex < dimension;
                 ++columnIndex) {
                std::swap(
                    schur[pivotIndex * dimension + columnIndex],
                    schur[selected * dimension + columnIndex]
                );
            }
            for (std::size_t rowIndex = 0u;
                 rowIndex < dimension;
                 ++rowIndex) {
                std::swap(
                    schur[rowIndex * dimension + pivotIndex],
                    schur[rowIndex * dimension + selected]
                );
            }
        }

        const double pivot =
            schur[pivotIndex * dimension + pivotIndex];
        if (!std::isfinite(pivot) || pivot < -tolerance) {
            return false;
        }
        if (pivot <= tolerance) {
            // A tiny dense Schur tail cannot be certified elementwise:
            // O(n) individually small off-diagonals may still form a
            // materially negative eigenmode. Use Accelerate's symmetric
            // eigensolver only for this rank-deficient/ambiguous case.
            double minimum = 0.0;
            return minimumEigenvalue(minimum) &&
                minimum >= -tolerance;
        }

        for (std::size_t row = pivotIndex + 1u;
             row < dimension;
             ++row) {
            column[row] =
                schur[row * dimension + pivotIndex];
        }
        for (std::size_t row = pivotIndex + 1u;
             row < dimension;
             ++row) {
            for (std::size_t columnIndex = row;
                 columnIndex < dimension;
                 ++columnIndex) {
                const double value =
                    schur[row * dimension + columnIndex] -
                    column[row] * column[columnIndex] / pivot;
                if (!std::isfinite(value)) {
                    return false;
                }
                schur[row * dimension + columnIndex] = value;
                schur[columnIndex * dimension + row] = value;
            }
        }
    }
    return true;
}

bool validProblem(
    const DenseConicProblem& problem,
    const QualityContactSolverConfig& config,
    std::string& failure
) {
    if (problem.nv == 0u) {
        failure = "nv must be positive";
        return false;
    }
    if (problem.inverseMass.size() !=
            static_cast<std::size_t>(problem.nv) * problem.nv ||
        problem.freeVelocity.size() != problem.nv ||
        problem.contacts.empty()) {
        failure = "problem dimensions are inconsistent";
        return false;
    }
    if (config.maximumIterations == 0u ||
        config.maximumLineSearchIterations == 0u ||
        !(config.kktTolerance > 0.0) ||
        !(config.armijoCoefficient > 0.0 &&
          config.armijoCoefficient < 0.5) ||
        !(config.factorizationPivotTolerance > 0.0) ||
        !(config.gaussNewtonRegularization > 0.0) ||
        !std::isfinite(config.kktTolerance) ||
        !std::isfinite(config.armijoCoefficient) ||
        !std::isfinite(config.factorizationPivotTolerance) ||
        !std::isfinite(config.gaussNewtonRegularization)) {
        failure = "solver configuration is invalid";
        return false;
    }
    if (!finite(problem.inverseMass) ||
        !finite(problem.freeVelocity)) {
        failure = "problem contains non-finite mass or velocity data";
        return false;
    }
    for (std::size_t row = 0; row < problem.nv; ++row) {
        for (std::size_t column = 0; column < problem.nv; ++column) {
            const double left =
                problem.inverseMass[row * problem.nv + column];
            const double right =
                problem.inverseMass[column * problem.nv + row];
            const double scale =
                1.0 + std::max(std::abs(left), std::abs(right));
            if (std::abs(left - right) > 1.0e-12 * scale) {
                failure = "inverse mass matrix is not symmetric";
                return false;
            }
        }
    }
    if (!positiveDefinite(problem.inverseMass, problem.nv)) {
        failure = "inverse mass matrix is not positive definite";
        return false;
    }
    for (const DenseContactBlock& contact : problem.contacts) {
        if (contact.normalJacobian.size() != problem.nv ||
            contact.tangentUJacobian.size() != problem.nv ||
            contact.tangentVJacobian.size() != problem.nv ||
            !finite(contact.normalJacobian) ||
            !finite(contact.tangentUJacobian) ||
            !finite(contact.tangentVJacobian) ||
            !finite(contact.targetVelocity) ||
            !finite(contact.regularization) ||
            !finite(contact.warmImpulse) ||
            !(contact.friction >= 0.0) ||
            !std::isfinite(contact.friction) ||
            !std::ranges::all_of(
                contact.regularization,
                [](const double value) {
                    return value > 0.0 && std::isfinite(value);
                }
            )) {
            failure = "contact data is invalid or non-finite";
            return false;
        }
    }
    return true;
}

bool validProblem(
    const ContactSpaceConicProblem& problem,
    const QualityContactSolverConfig& config,
    std::string& failure
) {
    const std::size_t dimension = 3u * problem.contacts.size();
    if (problem.contacts.empty() ||
        problem.delassus.size() != dimension * dimension ||
        problem.freeContactVelocity.size() != dimension) {
        failure = "contact-space problem dimensions are inconsistent";
        return false;
    }
    if (config.maximumIterations == 0u ||
        config.maximumLineSearchIterations == 0u ||
        !(config.kktTolerance > 0.0) ||
        !(config.armijoCoefficient > 0.0 &&
          config.armijoCoefficient < 0.5) ||
        !(config.factorizationPivotTolerance > 0.0) ||
        !(config.gaussNewtonRegularization > 0.0) ||
        !std::isfinite(config.kktTolerance) ||
        !std::isfinite(config.armijoCoefficient) ||
        !std::isfinite(config.factorizationPivotTolerance) ||
        !std::isfinite(config.gaussNewtonRegularization) ||
        !finite(problem.delassus) ||
        !finite(problem.freeContactVelocity)) {
        failure = "contact-space solver input is invalid or non-finite";
        return false;
    }
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            const double left =
                problem.delassus[row * dimension + column];
            const double right =
                problem.delassus[column * dimension + row];
            const double scale =
                1.0 + std::max(std::abs(left), std::abs(right));
            if (std::abs(left - right) > 1.0e-12 * scale) {
                failure = "Delassus operator is not symmetric";
                return false;
            }
        }
    }
    if (!positiveSemidefinite(problem.delassus, dimension)) {
        failure = "Delassus operator is not positive semidefinite";
        return false;
    }
    for (const ContactConicBlock& contact : problem.contacts) {
        if (!finite(contact.targetVelocity) ||
            !finite(contact.regularization) ||
            !finite(contact.warmImpulse) ||
            !(contact.friction >= 0.0) ||
            !std::isfinite(contact.friction) ||
            !std::ranges::all_of(
                contact.regularization,
                [](const double value) {
                    return value > 0.0 && std::isfinite(value);
                }
            )) {
            failure = "contact-space material data is invalid";
            return false;
        }
    }
    return true;
}

template <typename ContactContainer>
void finishPrepared(
    PreparedProblem& prepared,
    const ContactContainer& contacts
) {
    const std::size_t full = prepared.fullDimension;
    for (std::size_t contact = 0;
         contact < contacts.size();
         ++contact) {
        const double friction = contacts[contact].friction;
        const std::size_t offset = prepared.variables.size();
        if (friction > 0.0) {
            prepared.blocks.push_back({offset, 3u, contact});
            prepared.variables.push_back({
                3u * contact,
                1.0 / friction,
            });
            prepared.variables.push_back({
                3u * contact + 1u,
                1.0,
            });
            prepared.variables.push_back({
                3u * contact + 2u,
                1.0,
            });
        } else {
            // K_0 = { (lambda_n, 0, 0) | lambda_n >= 0 }.
            prepared.blocks.push_back({offset, 1u, contact});
            prepared.variables.push_back({3u * contact, 1.0});
        }
    }

    prepared.dimension = prepared.variables.size();
    prepared.compactHessian.assign(
        prepared.dimension * prepared.dimension,
        0.0
    );
    prepared.compactLinear.assign(prepared.dimension, 0.0);
    prepared.compactWarm.assign(prepared.dimension, 0.0);
    for (std::size_t row = 0; row < prepared.dimension; ++row) {
        const CompactVariable& rowVariable = prepared.variables[row];
        prepared.compactLinear[row] =
            rowVariable.lambdaScale *
            prepared.linear[rowVariable.fullIndex];
        prepared.compactWarm[row] =
            contacts[rowVariable.fullIndex / 3u]
                .warmImpulse[rowVariable.fullIndex % 3u] /
            rowVariable.lambdaScale;
        for (std::size_t column = 0;
             column < prepared.dimension;
             ++column) {
            const CompactVariable& columnVariable =
                prepared.variables[column];
            prepared.compactHessian[
                row * prepared.dimension + column
            ] =
                rowVariable.lambdaScale *
                prepared.hessian[
                    rowVariable.fullIndex * full +
                    columnVariable.fullIndex
                ] *
                columnVariable.lambdaScale;
        }
    }

    prepared.lipschitz = 0.0;
    for (std::size_t row = 0; row < prepared.dimension; ++row) {
        double rowSum = 0.0;
        for (std::size_t column = 0;
             column < prepared.dimension;
             ++column) {
            rowSum += std::abs(
                prepared.compactHessian[
                    row * prepared.dimension + column
                ]
            );
        }
        prepared.lipschitz = std::max(prepared.lipschitz, rowSum);
    }
}

PreparedProblem prepare(const DenseConicProblem& problem) {
    PreparedProblem prepared;
    prepared.nv = problem.nv;
    prepared.fullDimension = 3u * problem.contacts.size();
    prepared.jacobian.assign(
        prepared.fullDimension * prepared.nv,
        0.0
    );
    prepared.linear.assign(prepared.fullDimension, 0.0);

    std::vector<double> regularization(prepared.fullDimension, 0.0);
    for (std::size_t contact = 0;
         contact < problem.contacts.size();
         ++contact) {
        const DenseContactBlock& block = problem.contacts[contact];
        const std::array<std::span<const double>, 3> rows{
            block.normalJacobian,
            block.tangentUJacobian,
            block.tangentVJacobian,
        };
        for (std::size_t axis = 0; axis < 3u; ++axis) {
            const std::size_t row = 3u * contact + axis;
            std::copy(
                rows[axis].begin(),
                rows[axis].end(),
                prepared.jacobian.begin() + row * prepared.nv
            );
            prepared.linear[row] =
                dot(rows[axis], problem.freeVelocity) -
                block.targetVelocity[axis];
            regularization[row] = block.regularization[axis];
        }
    }

    const std::size_t full = prepared.fullDimension;
    std::vector<double> jacobianInverseMass(full * prepared.nv, 0.0);
    for (std::size_t row = 0; row < full; ++row) {
        for (std::size_t column = 0;
             column < prepared.nv;
             ++column) {
            for (std::size_t inner = 0;
                 inner < prepared.nv;
                 ++inner) {
                jacobianInverseMass[row * prepared.nv + column] +=
                    prepared.jacobian[row * prepared.nv + inner] *
                    problem.inverseMass[inner * prepared.nv + column];
            }
        }
    }

    prepared.hessian.assign(full * full, 0.0);
    for (std::size_t row = 0; row < full; ++row) {
        for (std::size_t column = 0; column < full; ++column) {
            prepared.hessian[row * full + column] = dot(
                std::span(
                    jacobianInverseMass.data() + row * prepared.nv,
                    prepared.nv
                ),
                std::span(
                    prepared.jacobian.data() + column * prepared.nv,
                    prepared.nv
                )
            );
        }
        prepared.hessian[row * full + row] += regularization[row];
    }

    finishPrepared(prepared, problem.contacts);
    return prepared;
}

PreparedProblem prepare(const ContactSpaceConicProblem& problem) {
    PreparedProblem prepared;
    prepared.fullDimension = 3u * problem.contacts.size();
    prepared.linear.assign(prepared.fullDimension, 0.0);
    prepared.hessian = problem.delassus;
    for (std::size_t contact = 0u;
         contact < problem.contacts.size();
         ++contact) {
        const ContactConicBlock& block =
            problem.contacts[contact];
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const std::size_t row = 3u * contact + axis;
            prepared.linear[row] =
                problem.freeContactVelocity[row] -
                block.targetVelocity[axis];
            prepared.hessian[
                row * prepared.fullDimension + row
            ] += block.regularization[axis];
        }
    }
    finishPrepared(prepared, problem.contacts);
    return prepared;
}

void projectCones(
    const std::span<const double> input,
    const std::span<const ConeBlock> blocks,
    std::span<double> output,
    std::span<double> derivative
) {
    std::fill(output.begin(), output.end(), 0.0);
    if (!derivative.empty()) {
        std::fill(derivative.begin(), derivative.end(), 0.0);
    }
    const std::size_t dimension = input.size();
    for (const ConeBlock& block : blocks) {
        const std::size_t offset = block.offset;
        if (block.dimension == 1u) {
            const double value = input[offset];
            output[offset] = std::max(value, 0.0);
            if (!derivative.empty()) {
                derivative[offset * dimension + offset] =
                    value > 0.0 ? 1.0 : (value < 0.0 ? 0.0 : 0.5);
            }
            continue;
        }

        const double scalar = input[offset];
        const double first = input[offset + 1u];
        const double second = input[offset + 2u];
        const double radial = std::hypot(first, second);
        if (scalar > radial) {
            output[offset] = scalar;
            output[offset + 1u] = first;
            output[offset + 2u] = second;
            if (!derivative.empty()) {
                derivative[offset * dimension + offset] = 1.0;
                derivative[
                    (offset + 1u) * dimension + offset + 1u
                ] = 1.0;
                derivative[
                    (offset + 2u) * dimension + offset + 2u
                ] = 1.0;
            }
            continue;
        }
        if (scalar < -radial) {
            continue;
        }
        if (radial == 0.0) {
            // At the cone apex, 0.5 I is a bounded Clarke element.
            if (!derivative.empty()) {
                derivative[offset * dimension + offset] = 0.5;
                derivative[
                    (offset + 1u) * dimension + offset + 1u
                ] = 0.5;
                derivative[
                    (offset + 2u) * dimension + offset + 2u
                ] = 0.5;
            }
            continue;
        }

        const double unitFirst = first / radial;
        const double unitSecond = second / radial;
        const double projectedScalar = 0.5 * (scalar + radial);
        const double tangentScale = projectedScalar / radial;
        output[offset] = projectedScalar;
        output[offset + 1u] = tangentScale * first;
        output[offset + 2u] = tangentScale * second;

        if (!derivative.empty()) {
            const double units[2]{unitFirst, unitSecond};
            derivative[offset * dimension + offset] = 0.5;
            for (std::size_t axis = 0; axis < 2u; ++axis) {
                derivative[
                    offset * dimension + offset + 1u + axis
                ] = 0.5 * units[axis];
                derivative[
                    (offset + 1u + axis) * dimension + offset
                ] = 0.5 * units[axis];
            }
            const double beta =
                0.5 * (1.0 + scalar / radial);
            const double rankScale =
                -0.5 * scalar / radial;
            for (std::size_t row = 0; row < 2u; ++row) {
                for (std::size_t column = 0;
                     column < 2u;
                     ++column) {
                    derivative[
                        (offset + 1u + row) * dimension +
                        offset + 1u + column
                    ] =
                        (row == column ? beta : 0.0) +
                        rankScale * units[row] * units[column];
                }
            }
        }
    }
}

struct NaturalMap {
    std::vector<double> gradient;
    std::vector<double> projected;
    std::vector<double> residual;
    std::vector<double> derivative;
    double residualNorm = 0.0;
    double scaledResidual = 0.0;
};

NaturalMap evaluateNaturalMap(
    const PreparedProblem& problem,
    const std::span<const double> value,
    const double gamma,
    const bool needDerivative
) {
    NaturalMap result;
    result.gradient = multiply(
        problem.compactHessian,
        problem.dimension,
        problem.dimension,
        value
    );
    std::vector<double> argument(problem.dimension, 0.0);
    for (std::size_t index = 0;
         index < problem.dimension;
         ++index) {
        result.gradient[index] += problem.compactLinear[index];
        argument[index] =
            value[index] - gamma * result.gradient[index];
    }
    result.projected.assign(problem.dimension, 0.0);
    if (needDerivative) {
        result.derivative.assign(
            problem.dimension * problem.dimension,
            0.0
        );
    }
    projectCones(
        argument,
        problem.blocks,
        result.projected,
        result.derivative
    );
    result.residual.assign(problem.dimension, 0.0);
    for (std::size_t index = 0;
         index < problem.dimension;
         ++index) {
        result.residual[index] =
            value[index] - result.projected[index];
    }
    result.residualNorm = norm(result.residual);
    result.scaledResidual =
        result.residualNorm /
        (gamma * (1.0 + norm(problem.compactLinear)));
    return result;
}

std::vector<double> generalizedJacobian(
    const PreparedProblem& problem,
    const std::span<const double> projectionDerivative,
    const double gamma
) {
    const std::size_t dimension = problem.dimension;
    std::vector<double> result(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0;
             column < dimension;
             ++column) {
            double derivativeTimesHessian = 0.0;
            for (std::size_t inner = 0;
                 inner < dimension;
                 ++inner) {
                derivativeTimesHessian +=
                    projectionDerivative[row * dimension + inner] *
                    problem.compactHessian[
                        inner * dimension + column
                    ];
            }
            result[row * dimension + column] =
                (row == column ? 1.0 : 0.0) -
                projectionDerivative[row * dimension + column] +
                gamma * derivativeTimesHessian;
        }
    }
    return result;
}

bool solveLinearSystem(
    std::vector<double> matrix,
    std::vector<double> rightHandSide,
    const double relativePivotTolerance,
    std::vector<double>& solution
) {
    const std::size_t dimension = rightHandSide.size();
    double maximumEntry = 0.0;
    for (const double value : matrix) {
        maximumEntry = std::max(maximumEntry, std::abs(value));
    }
    if (!(maximumEntry > 0.0) || !std::isfinite(maximumEntry)) {
        return false;
    }
    const double pivotTolerance =
        relativePivotTolerance * maximumEntry;

    for (std::size_t column = 0;
         column < dimension;
         ++column) {
        std::size_t pivotRow = column;
        double pivotMagnitude =
            std::abs(matrix[column * dimension + column]);
        for (std::size_t row = column + 1u;
             row < dimension;
             ++row) {
            const double candidate =
                std::abs(matrix[row * dimension + column]);
            if (candidate > pivotMagnitude) {
                pivotMagnitude = candidate;
                pivotRow = row;
            }
        }
        if (!(pivotMagnitude > pivotTolerance) ||
            !std::isfinite(pivotMagnitude)) {
            return false;
        }
        if (pivotRow != column) {
            for (std::size_t entry = column;
                 entry < dimension;
                 ++entry) {
                std::swap(
                    matrix[column * dimension + entry],
                    matrix[pivotRow * dimension + entry]
                );
            }
            std::swap(rightHandSide[column], rightHandSide[pivotRow]);
        }

        const double pivot =
            matrix[column * dimension + column];
        for (std::size_t row = column + 1u;
             row < dimension;
             ++row) {
            const double scale =
                matrix[row * dimension + column] / pivot;
            matrix[row * dimension + column] = 0.0;
            for (std::size_t entry = column + 1u;
                 entry < dimension;
                 ++entry) {
                matrix[row * dimension + entry] -=
                    scale * matrix[column * dimension + entry];
            }
            rightHandSide[row] -=
                scale * rightHandSide[column];
        }
    }

    solution.assign(dimension, 0.0);
    for (std::size_t reverse = 0;
         reverse < dimension;
         ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = rightHandSide[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                matrix[row * dimension + column] * solution[column];
        }
        const double diagonal = matrix[row * dimension + row];
        if (!(std::abs(diagonal) > pivotTolerance) ||
            !std::isfinite(diagonal)) {
            return false;
        }
        solution[row] = value / diagonal;
    }
    return finite(solution);
}

bool gaussNewtonDirection(
    const std::span<const double> generalizedDerivative,
    const std::span<const double> residual,
    const double baseRegularization,
    const double pivotTolerance,
    std::vector<double>& direction
) {
    const std::size_t dimension = residual.size();
    std::vector<double> normal(dimension * dimension, 0.0);
    std::vector<double> rightHandSide(dimension, 0.0);
    double diagonalScale = 0.0;
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0;
             column < dimension;
             ++column) {
            rightHandSide[column] -=
                generalizedDerivative[row * dimension + column] *
                residual[row];
            for (std::size_t inner = 0;
                 inner < dimension;
                 ++inner) {
                normal[column * dimension + inner] +=
                    generalizedDerivative[
                        row * dimension + column
                    ] *
                    generalizedDerivative[
                        row * dimension + inner
                    ];
            }
        }
    }
    for (std::size_t index = 0; index < dimension; ++index) {
        diagonalScale = std::max(
            diagonalScale,
            normal[index * dimension + index]
        );
    }
    const double regularization =
        baseRegularization * std::max(diagonalScale, 1.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        normal[index * dimension + index] += regularization;
    }
    return solveLinearSystem(
        std::move(normal),
        std::move(rightHandSide),
        pivotTolerance,
        direction
    );
}

std::vector<double> toFullImpulses(
    const PreparedProblem& problem,
    const std::span<const double> compact
) {
    std::vector<double> full(problem.fullDimension, 0.0);
    for (std::size_t index = 0;
         index < problem.dimension;
         ++index) {
        full[problem.variables[index].fullIndex] =
            problem.variables[index].lambdaScale * compact[index];
    }
    return full;
}

template <typename ContactContainer>
void fillCertificate(
    const ContactContainer& contacts,
    const PreparedProblem& problem,
    const std::span<const double> compact,
    const double gamma,
    QualityContactSolution& solution
) {
    solution.impulses = toFullImpulses(problem, compact);
    const std::vector<double> dual = [&]() {
        std::vector<double> result = multiply(
            problem.hessian,
            problem.fullDimension,
            problem.fullDimension,
            solution.impulses
        );
        for (std::size_t index = 0;
             index < result.size();
             ++index) {
            result[index] += problem.linear[index];
        }
        return result;
    }();

    solution.maximumPrimalConeViolation = 0.0;
    solution.maximumDualConeViolation = 0.0;
    solution.maximumNormalImpulseViolation = 0.0;
    solution.maximumConicComplementarityResidual = 0.0;
    solution.maximumFrictionlessNormalComplementarityResidual = 0.0;
    solution.complementarityGap = 0.0;
    for (std::size_t contact = 0;
         contact < contacts.size();
         ++contact) {
        const std::size_t offset = 3u * contact;
        const double normal = solution.impulses[offset];
        const double tangent = std::hypot(
            solution.impulses[offset + 1u],
            solution.impulses[offset + 2u]
        );
        const double friction = contacts[contact].friction;
        solution.maximumNormalImpulseViolation = std::max(
            solution.maximumNormalImpulseViolation,
            std::max(-normal, 0.0)
        );
        solution.maximumPrimalConeViolation = std::max(
            solution.maximumPrimalConeViolation,
            std::max(
                std::max(-normal, 0.0),
                tangent - friction * normal
            )
        );

        const double dualNormal = dual[offset];
        const double dualTangent = std::hypot(
            dual[offset + 1u],
            dual[offset + 2u]
        );
        const double dualViolation =
            friction > 0.0
            ? friction * dualTangent - dualNormal
            : -dualNormal;
        solution.maximumDualConeViolation = std::max(
            solution.maximumDualConeViolation,
            std::max(dualViolation, 0.0)
        );

        const double contactComplementarity =
            normal * dualNormal +
            solution.impulses[offset + 1u] * dual[offset + 1u] +
            solution.impulses[offset + 2u] * dual[offset + 2u];
        solution.maximumConicComplementarityResidual = std::max(
            solution.maximumConicComplementarityResidual,
            std::abs(contactComplementarity)
        );
        if (friction == 0.0) {
            solution.maximumFrictionlessNormalComplementarityResidual =
                std::max(
                    solution
                        .maximumFrictionlessNormalComplementarityResidual,
                    std::abs(normal * dualNormal)
                );
        }
        solution.complementarityGap += contactComplementarity;
    }

    const NaturalMap natural = evaluateNaturalMap(
        problem,
        compact,
        gamma,
        false
    );
    solution.scaledNaturalResidual = natural.scaledResidual;
    const double impulseNorm = norm(solution.impulses);
    const double dualNorm = norm(dual);
    const double primalScaled =
        solution.maximumPrimalConeViolation /
        (1.0 + impulseNorm);
    const double dualScaled =
        solution.maximumDualConeViolation /
        (1.0 + dualNorm);
    const double complementarityScaled =
        std::abs(solution.complementarityGap) /
        (1.0 + impulseNorm * dualNorm);
    solution.scaledKktCertificate = std::max({
        solution.scaledNaturalResidual,
        primalScaled,
        dualScaled,
        complementarityScaled,
    });
    solution.objective = objective(
        problem.hessian,
        problem.linear,
        solution.impulses
    );
}

template <typename ContactContainer>
QualityContactSolution solvePreparedQualityProblem(
    const PreparedProblem& prepared,
    const ContactContainer& contacts,
    const QualityContactSolverConfig& config
) {
    QualityContactSolution solution;
    if (!finite(prepared.hessian) ||
        !finite(prepared.linear) ||
        !finite(prepared.compactHessian) ||
        !finite(prepared.compactLinear) ||
        !positiveDefinite(
            prepared.compactHessian,
            prepared.dimension
        ) ||
        !(prepared.lipschitz > 0.0) ||
        !std::isfinite(prepared.lipschitz)) {
        solution.code = MR_STEP_FACTORIZATION_FAILED;
        solution.failure =
            "contact Hessian is non-finite or not positive definite";
        return solution;
    }
    solution.lipschitzBound = prepared.lipschitz;
    const double gamma = 1.0 / prepared.lipschitz;

    std::vector<double> value(prepared.dimension, 0.0);
    if (config.enableWarmStart) {
        std::vector<double> projectedWarm(prepared.dimension, 0.0);
        projectCones(
            prepared.compactWarm,
            prepared.blocks,
            projectedWarm,
            {}
        );
        if (objective(
                prepared.compactHessian,
                prepared.compactLinear,
                projectedWarm
            ) < 0.0) {
            value = std::move(projectedWarm);
            solution.usedWarmStart = true;
        }
    }

    bool converged = false;
    for (std::uint32_t iteration = 0u;
         iteration < config.maximumIterations;
         ++iteration) {
        const NaturalMap natural = evaluateNaturalMap(
            prepared,
            value,
            gamma,
            true
        );
        if (!std::isfinite(natural.scaledResidual) ||
            !finite(natural.residual) ||
            !finite(natural.derivative)) {
            solution.code = MR_STEP_NONFINITE_RESULT;
            solution.failure =
                "natural residual became non-finite";
            fillCertificate(
                contacts,
                prepared,
                value,
                gamma,
                solution
            );
            return solution;
        }
        if (natural.scaledResidual <= config.kktTolerance) {
            // Return a point projected onto the exact cone, then re-check the
            // natural residual so feasibility is not just tolerance-based.
            std::vector<double> projected(prepared.dimension, 0.0);
            projectCones(value, prepared.blocks, projected, {});
            const NaturalMap projectedNatural = evaluateNaturalMap(
                prepared,
                projected,
                gamma,
                false
            );
            if (projectedNatural.scaledResidual <=
                1.05 * config.kktTolerance) {
                value = std::move(projected);
                converged = true;
                solution.iterations = iteration;
                break;
            }
            value = std::move(projected);
            continue;
        }

        const std::vector<double> derivative =
            generalizedJacobian(
                prepared,
                natural.derivative,
                gamma
            );
        std::vector<double> rightHandSide(natural.residual.size(), 0.0);
        std::ranges::transform(
            natural.residual,
            rightHandSide.begin(),
            [](const double entry) {
                return -entry;
            }
        );

        std::vector<double> direction;
        bool newtonDirection = solveLinearSystem(
            derivative,
            std::move(rightHandSide),
            config.factorizationPivotTolerance,
            direction
        );
        double slope = 0.0;
        if (newtonDirection) {
            const std::vector<double> derivativeDirection = multiply(
                derivative,
                prepared.dimension,
                prepared.dimension,
                direction
            );
            slope = dot(natural.residual, derivativeDirection);
            if (!std::isfinite(slope) ||
                slope >=
                    -1.0e-8 *
                    natural.residualNorm *
                    natural.residualNorm) {
                newtonDirection = false;
            }
        }

        bool usedGaussNewton = false;
        if (!newtonDirection) {
            usedGaussNewton = gaussNewtonDirection(
                derivative,
                natural.residual,
                config.gaussNewtonRegularization,
                config.factorizationPivotTolerance,
                direction
            );
            if (usedGaussNewton) {
                const std::vector<double> derivativeDirection =
                    multiply(
                        derivative,
                        prepared.dimension,
                        prepared.dimension,
                        direction
                    );
                slope = dot(
                    natural.residual,
                    derivativeDirection
                );
                if (!std::isfinite(slope) || !(slope < 0.0)) {
                    usedGaussNewton = false;
                }
            }
        }

        bool accepted = false;
        if (newtonDirection || usedGaussNewton) {
            const double merit =
                0.5 * natural.residualNorm * natural.residualNorm;
            double step = 1.0;
            for (std::uint32_t lineSearch = 0u;
                 lineSearch <
                    config.maximumLineSearchIterations;
                 ++lineSearch) {
                std::vector<double> candidate = value;
                for (std::size_t index = 0;
                     index < candidate.size();
                     ++index) {
                    candidate[index] += step * direction[index];
                }
                const NaturalMap candidateNatural =
                    evaluateNaturalMap(
                        prepared,
                        candidate,
                        gamma,
                        false
                    );
                const double candidateMerit =
                    0.5 *
                    candidateNatural.residualNorm *
                    candidateNatural.residualNorm;
                if (std::isfinite(candidateMerit) &&
                    candidateMerit <=
                        merit +
                        config.armijoCoefficient * step * slope) {
                    value = std::move(candidate);
                    accepted = true;
                    if (newtonDirection) {
                        ++solution.semismoothNewtonSteps;
                    } else {
                        ++solution.gaussNewtonFallbackSteps;
                    }
                    break;
                }
                step *= 0.5;
                ++solution.lineSearchBacktracks;
            }
        }

        if (!accepted) {
            // T(x) = Pi_K(x - gamma grad f(x)) is contractive for the
            // strongly-convex QP and gamma <= 1/L. This is the global safety
            // step; Newton remains the local fast path.
            value = natural.projected;
            ++solution.projectedGradientFallbackSteps;
        }
        solution.iterations = iteration + 1u;
    }

    fillCertificate(
        contacts,
        prepared,
        value,
        gamma,
        solution
    );
    if (!finite(solution.impulses) ||
        !std::isfinite(solution.objective) ||
        !std::isfinite(solution.scaledKktCertificate)) {
        solution.code = MR_STEP_NONFINITE_RESULT;
        solution.failure = "solution or KKT certificate is non-finite";
        return solution;
    }

    if (converged &&
        solution.scaledKktCertificate <=
            1.1 * config.kktTolerance) {
        solution.code = MR_STEP_SUCCESS;
        solution.failure.clear();
    } else {
        solution.code = MR_STEP_DID_NOT_CONVERGE;
        solution.failure =
            "globalized semismooth Newton iteration limit reached "
            "before the KKT certificate met tolerance";
    }
    return solution;
}

} // namespace

QualityContactSolution solveQualityContactProblem(
    const DenseConicProblem& problem,
    const QualityContactSolverConfig& config
) {
    QualityContactSolution solution;
    if (!validProblem(problem, config, solution.failure)) {
        solution.code = MR_STEP_NONFINITE_INPUT;
        return solution;
    }

    const PreparedProblem prepared = prepare(problem);
    solution = solvePreparedQualityProblem(
        prepared,
        problem.contacts,
        config
    );
    if (solution.impulses.size() != prepared.fullDimension) {
        return solution;
    }

    std::vector<double> generalizedImpulse(problem.nv, 0.0);
    for (std::size_t contact = 0u;
         contact < problem.contacts.size();
         ++contact) {
        const DenseContactBlock& block = problem.contacts[contact];
        const std::array<std::span<const double>, 3u> rows{
            block.normalJacobian,
            block.tangentUJacobian,
            block.tangentVJacobian,
        };
        for (std::size_t axis = 0u; axis < rows.size(); ++axis) {
            const double impulse = solution.impulses[
                3u * contact + axis
            ];
            for (std::size_t column = 0u;
                 column < problem.nv;
                 ++column) {
                generalizedImpulse[column] +=
                    rows[axis][column] * impulse;
            }
        }
    }
    solution.velocity = problem.freeVelocity;
    const std::vector<double> velocityDelta = multiply(
        problem.inverseMass,
        problem.nv,
        problem.nv,
        generalizedImpulse
    );
    for (std::size_t index = 0u; index < problem.nv; ++index) {
        solution.velocity[index] += velocityDelta[index];
    }
    if (!finite(solution.velocity)) {
        solution.code = MR_STEP_NONFINITE_RESULT;
        solution.failure =
            "generalized post-impulse velocity is non-finite";
    }
    return solution;
}

QualityContactSolution solveQualityContactSpaceProblem(
    const ContactSpaceConicProblem& problem,
    const QualityContactSolverConfig& config
) {
    QualityContactSolution solution;
    if (!validProblem(problem, config, solution.failure)) {
        solution.code = MR_STEP_NONFINITE_INPUT;
        return solution;
    }

    const PreparedProblem prepared = prepare(problem);
    solution = solvePreparedQualityProblem(
        prepared,
        problem.contacts,
        config
    );
    if (solution.impulses.size() != prepared.fullDimension) {
        return solution;
    }

    solution.velocity = problem.freeContactVelocity;
    const std::vector<double> contactVelocityDelta = multiply(
        problem.delassus,
        prepared.fullDimension,
        prepared.fullDimension,
        solution.impulses
    );
    for (std::size_t index = 0u;
         index < prepared.fullDimension;
         ++index) {
        solution.velocity[index] += contactVelocityDelta[index];
    }
    if (!finite(solution.velocity)) {
        solution.code = MR_STEP_NONFINITE_RESULT;
        solution.failure =
            "contact-space post-impulse velocity is non-finite";
    }
    return solution;
}

} // namespace metalrobo
