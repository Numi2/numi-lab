#include "metalrobo/ReferenceConicSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

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

std::array<double, 3> projectCone(
    const std::array<double, 3> value,
    const double friction
) {
    const double tangentNorm = std::hypot(value[1], value[2]);
    if (!(friction > 0.0)) {
        return {std::max(value[0], 0.0), 0.0, 0.0};
    }
    if (value[0] >= 0.0 && tangentNorm <= friction * value[0]) {
        return value;
    }
    if (value[0] + friction * tangentNorm <= 0.0) {
        return {};
    }

    const double normal =
        (value[0] + friction * tangentNorm) /
        (1.0 + friction * friction);
    const double projectedTangentNorm = friction * normal;
    const double scale =
        tangentNorm > 0.0 ? projectedTangentNorm / tangentNorm : 0.0;
    return {normal, value[1] * scale, value[2] * scale};
}

void projectAll(
    std::span<double> values,
    const std::span<const DenseContactBlock> contacts
) {
    for (std::size_t contact = 0; contact < contacts.size(); ++contact) {
        const std::size_t offset = 3u * contact;
        const auto projected = projectCone(
            {values[offset], values[offset + 1], values[offset + 2]},
            contacts[contact].friction
        );
        values[offset] = projected[0];
        values[offset + 1] = projected[1];
        values[offset + 2] = projected[2];
    }
}

std::vector<double> multiply(
    const std::span<const double> matrix,
    const std::size_t dimension,
    const std::span<const double> vector
) {
    std::vector<double> result(dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            result[row] +=
                matrix[row * dimension + column] * vector[column];
        }
    }
    return result;
}

double objective(
    const std::span<const double> hessian,
    const std::span<const double> linear,
    const std::span<const double> impulses
) {
    const std::vector<double> product =
        multiply(hessian, impulses.size(), impulses);
    return 0.5 * dot(impulses, product) + dot(linear, impulses);
}

bool positiveDefinite(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    std::vector<double> factor(matrix.begin(), matrix.end());
    for (std::size_t column = 0; column < dimension; ++column) {
        for (std::size_t row = column; row < dimension; ++row) {
            double value = factor[row * dimension + column];
            for (std::size_t inner = 0; inner < column; ++inner) {
                value -= factor[row * dimension + inner] *
                    factor[column * dimension + inner];
            }
            if (row == column) {
                if (!(value > 1.0e-15) || !std::isfinite(value)) {
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

bool validProblem(
    const DenseConicProblem& problem,
    const ReferenceConicSolverConfig& config
) {
    if (problem.nv == 0u ||
        problem.inverseMass.size() !=
            static_cast<std::size_t>(problem.nv) * problem.nv ||
        problem.freeVelocity.size() != problem.nv ||
        problem.contacts.empty() ||
        config.maximumIterations == 0u ||
        !(config.optimalityTolerance > 0.0) ||
        !std::isfinite(config.optimalityTolerance) ||
        !finite(problem.inverseMass) ||
        !finite(problem.freeVelocity)) {
        return false;
    }
    for (std::size_t row = 0; row < problem.nv; ++row) {
        for (std::size_t column = 0; column < problem.nv; ++column) {
            const double difference = std::abs(
                problem.inverseMass[row * problem.nv + column] -
                problem.inverseMass[column * problem.nv + row]
            );
            if (difference > 1.0e-10) {
                return false;
            }
        }
    }
    if (!positiveDefinite(problem.inverseMass, problem.nv)) {
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
            return false;
        }
    }
    return true;
}

} // namespace

ReferenceConicSolution solveReferenceConicProblem(
    const DenseConicProblem& problem,
    const ReferenceConicSolverConfig& config
) {
    ReferenceConicSolution solution;
    if (!validProblem(problem, config)) {
        solution.code = MR_STEP_NONFINITE_INPUT;
        return solution;
    }

    const std::size_t nv = problem.nv;
    const std::size_t contactCount = problem.contacts.size();
    const std::size_t dimension = 3u * contactCount;
    std::vector<double> jacobian(dimension * nv, 0.0);
    std::vector<double> target(dimension, 0.0);
    std::vector<double> regularization(dimension, 0.0);
    std::vector<double> warm(dimension, 0.0);

    for (std::size_t contact = 0; contact < contactCount; ++contact) {
        const DenseContactBlock& block = problem.contacts[contact];
        const std::array<std::span<const double>, 3> rows{
            block.normalJacobian,
            block.tangentUJacobian,
            block.tangentVJacobian,
        };
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::size_t row = 3u * contact + axis;
            std::copy(
                rows[axis].begin(),
                rows[axis].end(),
                jacobian.begin() + row * nv
            );
            target[row] = block.targetVelocity[axis];
            regularization[row] = block.regularization[axis];
            warm[row] = block.warmImpulse[axis];
        }
    }

    std::vector<double> jacobianInverseMass(dimension * nv, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < nv; ++column) {
            for (std::size_t inner = 0; inner < nv; ++inner) {
                jacobianInverseMass[row * nv + column] +=
                    jacobian[row * nv + inner] *
                    problem.inverseMass[inner * nv + column];
            }
        }
    }

    std::vector<double> hessian(dimension * dimension, 0.0);
    std::vector<double> linear(dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        linear[row] =
            dot(
                std::span(
                    jacobian.data() + row * nv,
                    nv
                ),
                problem.freeVelocity
            ) -
            target[row];
        for (std::size_t column = 0; column < dimension; ++column) {
            hessian[row * dimension + column] =
                dot(
                    std::span(
                        jacobianInverseMass.data() + row * nv,
                        nv
                    ),
                    std::span(
                        jacobian.data() + column * nv,
                        nv
                    )
                );
        }
        hessian[row * dimension + row] += regularization[row];
    }
    if (!finite(hessian) || !finite(linear) ||
        !positiveDefinite(hessian, dimension)) {
        solution.code = MR_STEP_FACTORIZATION_FAILED;
        return solution;
    }

    double lipschitz = 0.0;
    for (std::size_t row = 0; row < dimension; ++row) {
        double rowSum = 0.0;
        for (std::size_t column = 0; column < dimension; ++column) {
            rowSum += std::abs(hessian[row * dimension + column]);
        }
        lipschitz = std::max(lipschitz, rowSum);
    }
    if (!(lipschitz > 0.0) || !std::isfinite(lipschitz)) {
        solution.code = MR_STEP_FACTORIZATION_FAILED;
        return solution;
    }
    solution.lipschitzBound = lipschitz;
    const double inverseLipschitz = 1.0 / lipschitz;

    std::vector<double> zero(dimension, 0.0);
    projectAll(warm, problem.contacts);
    std::vector<double> current = zero;
    if (config.enableWarmStart &&
        objective(hessian, linear, warm) < 0.0) {
        current = warm;
        solution.usedWarmStart = true;
    }
    std::vector<double> extrapolated = current;
    double acceleration = 1.0;
    double currentObjective = objective(hessian, linear, current);
    const double residualScale = 1.0 + norm(linear);

    for (std::uint32_t iteration = 0;
         iteration < config.maximumIterations;
         ++iteration) {
        std::vector<double> gradient =
            multiply(hessian, dimension, extrapolated);
        for (std::size_t index = 0; index < dimension; ++index) {
            gradient[index] += linear[index];
        }
        std::vector<double> candidate(dimension, 0.0);
        for (std::size_t index = 0; index < dimension; ++index) {
            candidate[index] =
                extrapolated[index] -
                inverseLipschitz * gradient[index];
        }
        projectAll(candidate, problem.contacts);
        double candidateObjective =
            objective(hessian, linear, candidate);

        if (candidateObjective > currentObjective + 1.0e-14 &&
            config.enableAcceleration) {
            extrapolated = current;
            acceleration = 1.0;
            gradient = multiply(hessian, dimension, extrapolated);
            for (std::size_t index = 0; index < dimension; ++index) {
                gradient[index] += linear[index];
                candidate[index] =
                    extrapolated[index] -
                    inverseLipschitz * gradient[index];
            }
            projectAll(candidate, problem.contacts);
            candidateObjective =
                objective(hessian, linear, candidate);
        }

        std::vector<double> candidateGradient =
            multiply(hessian, dimension, candidate);
        std::vector<double> projectedStep(dimension, 0.0);
        for (std::size_t index = 0; index < dimension; ++index) {
            candidateGradient[index] += linear[index];
            projectedStep[index] =
                candidate[index] -
                inverseLipschitz * candidateGradient[index];
        }
        projectAll(projectedStep, problem.contacts);
        for (std::size_t index = 0; index < dimension; ++index) {
            projectedStep[index] =
                lipschitz * (candidate[index] - projectedStep[index]);
        }
        const double scaledResidual =
            norm(projectedStep) / residualScale;
        solution.iterations = iteration + 1u;
        solution.scaledOptimalityResidual = scaledResidual;

        if (!std::isfinite(candidateObjective) ||
            !std::isfinite(scaledResidual) ||
            !finite(candidate)) {
            solution.code = MR_STEP_NONFINITE_RESULT;
            return solution;
        }
        if (scaledResidual <= config.optimalityTolerance) {
            current = std::move(candidate);
            currentObjective = candidateObjective;
            solution.code = MR_STEP_SUCCESS;
            break;
        }

        if (config.enableAcceleration) {
            const double nextAcceleration =
                0.5 * (
                    1.0 +
                    std::sqrt(1.0 + 4.0 * acceleration * acceleration)
                );
            std::vector<double> nextExtrapolated(dimension, 0.0);
            const double scale =
                (acceleration - 1.0) / nextAcceleration;
            for (std::size_t index = 0; index < dimension; ++index) {
                nextExtrapolated[index] =
                    candidate[index] +
                    scale * (candidate[index] - current[index]);
            }
            extrapolated = std::move(nextExtrapolated);
            acceleration = nextAcceleration;
        } else {
            extrapolated = candidate;
        }
        current = std::move(candidate);
        currentObjective = candidateObjective;

        if (iteration + 1u == config.maximumIterations) {
            solution.code = MR_STEP_DID_NOT_CONVERGE;
        }
    }

    solution.impulses = current;
    solution.objective = currentObjective;
    solution.maximumConeViolation = 0.0;
    for (std::size_t contact = 0; contact < contactCount; ++contact) {
        const std::size_t offset = 3u * contact;
        const double tangent =
            std::hypot(current[offset + 1], current[offset + 2]);
        solution.maximumConeViolation = std::max(
            solution.maximumConeViolation,
            std::max(
                -current[offset],
                tangent -
                    problem.contacts[contact].friction * current[offset]
            )
        );
    }

    solution.velocity = problem.freeVelocity;
    std::vector<double> generalizedImpulse(nv, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < nv; ++column) {
            generalizedImpulse[column] +=
                jacobian[row * nv + column] * current[row];
        }
    }
    const std::vector<double> velocityDelta =
        multiply(problem.inverseMass, nv, generalizedImpulse);
    for (std::size_t index = 0; index < nv; ++index) {
        solution.velocity[index] += velocityDelta[index];
    }
    if (!finite(solution.velocity) || !finite(solution.impulses)) {
        solution.code = MR_STEP_NONFINITE_RESULT;
    }
    return solution;
}

} // namespace metalrobo
