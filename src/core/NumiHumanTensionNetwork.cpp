#include "metalrobo/NumiHumanTensionNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace metalrobo {
namespace {

using Vec3 = std::array<double, 3>;

bool finite(const double value) { return std::isfinite(value); }

bool finite(const Vec3& value) {
    return std::all_of(value.begin(), value.end(), [](const double x) {
        return finite(x);
    });
}

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 scale(const Vec3& value, const double factor) {
    return {factor * value[0], factor * value[1], factor * value[2]};
}

double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }

NumiHumanTensionNetworkDiagnostics fail(
    const NumiHumanTensionNetworkStatus status,
    const std::uint32_t index = 0u
) {
    return {.status = status, .failingIndex = index};
}

struct Evaluation {
    std::vector<Vec3> residual;
    std::vector<double> tension;
    std::vector<double> stiffness;
    double strainEnergy = 0.0;
    double potentialEnergy = 0.0;
    double maximumFreeResidual = 0.0;
    std::uint32_t activeElementCount = 0u;
};

bool evaluate(
    const std::span<const NumiHumanTensionNetworkNode> nodes,
    const std::span<const NumiHumanTensionNetworkElement> elements,
    const std::span<const NumiHumanTensionNetworkLoad> loads,
    const std::span<const Vec3> position,
    Evaluation& output
) {
    output = {};
    output.residual.assign(nodes.size(), {});
    output.tension.assign(elements.size(), 0.0);
    output.stiffness.assign(elements.size(), 0.0);
    for (std::size_t index = 0u; index < elements.size(); ++index) {
        const auto& element = elements[index];
        const Vec3 delta = subtract(
            position[element.nodeB], position[element.nodeA]);
        const double length = norm(delta);
        if (!(length > 0.0) || !finite(length)) return false;
        const double extension = length - element.restLength;
        if (!(extension > 0.0)) continue;
        const double axialStiffness =
            element.youngModulus * element.area / element.restLength;
        const double tension = axialStiffness * extension;
        const Vec3 direction = scale(delta, 1.0 / length);
        const Vec3 force = scale(direction, tension);
        output.residual[element.nodeA] =
            add(output.residual[element.nodeA], force);
        output.residual[element.nodeB] =
            subtract(output.residual[element.nodeB], force);
        output.tension[index] = tension;
        output.stiffness[index] = axialStiffness;
        output.strainEnergy +=
            0.5 * axialStiffness * extension * extension;
        ++output.activeElementCount;
    }
    output.potentialEnergy = output.strainEnergy;
    for (const auto& load : loads) {
        output.residual[load.nodeIndex] =
            add(output.residual[load.nodeIndex], load.force);
        output.potentialEnergy -= dot(load.force, position[load.nodeIndex]);
    }
    if (!finite(output.strainEnergy) || !finite(output.potentialEnergy)) {
        return false;
    }
    for (std::size_t node = 0u; node < nodes.size(); ++node) {
        if (!finite(output.residual[node])) return false;
        if (!nodes[node].fixed) {
            output.maximumFreeResidual = std::max(
                output.maximumFreeResidual, norm(output.residual[node]));
        }
    }
    return true;
}

bool choleskySolve(
    std::vector<double> matrix,
    std::vector<double> rhs,
    std::vector<double>& solution
) {
    const std::size_t n = rhs.size();
    for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * n + column];
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -= matrix[row * n + inner] *
                    matrix[column * n + inner];
            }
            if (row == column) {
                if (!(value > 0.0) || !finite(value)) return false;
                matrix[row * n + column] = std::sqrt(value);
            } else {
                matrix[row * n + column] =
                    value / matrix[column * n + column];
            }
        }
    }
    for (std::size_t row = 0u; row < n; ++row) {
        double value = rhs[row];
        for (std::size_t column = 0u; column < row; ++column) {
            value -= matrix[row * n + column] * rhs[column];
        }
        rhs[row] = value / matrix[row * n + row];
    }
    solution.assign(n, 0.0);
    for (std::size_t reverse = 0u; reverse < n; ++reverse) {
        const std::size_t row = n - 1u - reverse;
        double value = rhs[row];
        for (std::size_t column = row + 1u; column < n; ++column) {
            value -= matrix[column * n + row] * solution[column];
        }
        solution[row] = value / matrix[row * n + row];
    }
    return std::all_of(
        solution.begin(), solution.end(),
        [](const double value) { return finite(value); });
}

} // namespace

NumiHumanTensionNetworkDiagnostics solveNumiHumanTensionNetwork(
    const std::span<const NumiHumanTensionNetworkNode> nodes,
    const std::span<const NumiHumanTensionNetworkElement> elements,
    const std::span<const NumiHumanTensionNetworkLoad> loads,
    NumiHumanTensionNetworkResult& result,
    const NumiHumanTensionNetworkConfig& config
) {
    if (nodes.size() < 2u || elements.empty() ||
        config.maximumIterations == 0u ||
        config.maximumLineSearchSteps == 0u ||
        !(config.forceTolerance > 0.0) ||
        !(config.minimumLength > 0.0) ||
        !(config.diagonalRegularization > 0.0) ||
        !(config.armijoFraction > 0.0 && config.armijoFraction < 1.0)) {
        return fail(NumiHumanTensionNetworkStatus::invalidTopology);
    }
    std::vector<std::uint32_t> freeIndex(nodes.size(), UINT32_MAX);
    std::uint32_t freeCount = 0u;
    bool hasFixed = false;
    for (std::size_t index = 0u; index < nodes.size(); ++index) {
        if (!finite(nodes[index].position)) {
            return fail(NumiHumanTensionNetworkStatus::invalidTopology,
                        static_cast<std::uint32_t>(index));
        }
        if (nodes[index].fixed) hasFixed = true;
        else freeIndex[index] = freeCount++;
    }
    if (!hasFixed || freeCount == 0u) {
        return fail(NumiHumanTensionNetworkStatus::invalidTopology);
    }
    for (std::size_t index = 0u; index < elements.size(); ++index) {
        const auto& element = elements[index];
        if (element.nodeA >= nodes.size() || element.nodeB >= nodes.size() ||
            element.nodeA == element.nodeB ||
            !(element.restLength >= config.minimumLength) ||
            !(element.youngModulus > 0.0) || !(element.area > 0.0) ||
            !finite(element.restLength) || !finite(element.youngModulus) ||
            !finite(element.area)) {
            return fail(NumiHumanTensionNetworkStatus::invalidMaterial,
                        static_cast<std::uint32_t>(index));
        }
    }
    for (std::size_t index = 0u; index < loads.size(); ++index) {
        if (loads[index].nodeIndex >= nodes.size() ||
            !finite(loads[index].force)) {
            return fail(NumiHumanTensionNetworkStatus::invalidLoad,
                        static_cast<std::uint32_t>(index));
        }
    }

    std::vector<Vec3> position;
    position.reserve(nodes.size());
    for (const auto& node : nodes) position.push_back(node.position);
    Evaluation evaluation;
    if (!evaluate(nodes, elements, loads, position, evaluation)) {
        return fail(NumiHumanTensionNetworkStatus::nonfiniteResult);
    }

    const std::size_t dimension = 3u * freeCount;
    std::uint32_t completedIterations = 0u;
    for (; completedIterations < config.maximumIterations;
         ++completedIterations) {
        if (evaluation.maximumFreeResidual <= config.forceTolerance) break;
        std::vector<double> matrix(dimension * dimension, 0.0);
        std::vector<double> rhs(dimension, 0.0);
        for (std::size_t node = 0u; node < nodes.size(); ++node) {
            if (nodes[node].fixed) continue;
            const std::size_t base = 3u * freeIndex[node];
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                rhs[base + axis] = evaluation.residual[node][axis];
            }
        }
        for (std::size_t index = 0u; index < elements.size(); ++index) {
            if (!(evaluation.tension[index] > 0.0)) continue;
            const auto& element = elements[index];
            const Vec3 delta = subtract(
                position[element.nodeB], position[element.nodeA]);
            const double length = norm(delta);
            if (!(length >= config.minimumLength)) {
                return fail(NumiHumanTensionNetworkStatus::nonfiniteResult,
                            static_cast<std::uint32_t>(index));
            }
            const Vec3 direction = scale(delta, 1.0 / length);
            double block[3][3]{};
            for (std::size_t row = 0u; row < 3u; ++row) {
                for (std::size_t column = 0u; column < 3u; ++column) {
                    const double outer = direction[row] * direction[column];
                    block[row][column] =
                        evaluation.stiffness[index] * outer +
                        evaluation.tension[index] / length *
                            ((row == column ? 1.0 : 0.0) - outer);
                }
            }
            const auto addBlock = [&](const std::uint32_t nodeRow,
                                      const std::uint32_t nodeColumn,
                                      const double sign) {
                if (nodes[nodeRow].fixed || nodes[nodeColumn].fixed) return;
                const std::size_t rowBase = 3u * freeIndex[nodeRow];
                const std::size_t columnBase = 3u * freeIndex[nodeColumn];
                for (std::size_t row = 0u; row < 3u; ++row) {
                    for (std::size_t column = 0u; column < 3u; ++column) {
                        matrix[(rowBase + row) * dimension +
                               columnBase + column] += sign * block[row][column];
                    }
                }
            };
            addBlock(element.nodeA, element.nodeA, 1.0);
            addBlock(element.nodeB, element.nodeB, 1.0);
            addBlock(element.nodeA, element.nodeB, -1.0);
            addBlock(element.nodeB, element.nodeA, -1.0);
        }
        double matrixScale = 1.0;
        for (const double value : matrix) {
            matrixScale = std::max(matrixScale, std::abs(value));
        }
        for (std::size_t diagonal = 0u; diagonal < dimension; ++diagonal) {
            matrix[diagonal * dimension + diagonal] +=
                config.diagonalRegularization * matrixScale;
        }
        std::vector<double> step;
        if (!choleskySolve(std::move(matrix), rhs, step)) {
            return fail(NumiHumanTensionNetworkStatus::singularSystem);
        }
        double descent = 0.0;
        for (std::size_t index = 0u; index < dimension; ++index) {
            descent += rhs[index] * step[index];
        }
        if (!(descent > 0.0) || !finite(descent)) {
            return fail(NumiHumanTensionNetworkStatus::singularSystem);
        }
        bool accepted = false;
        double alpha = 1.0;
        std::vector<Vec3> candidate = position;
        Evaluation candidateEvaluation;
        for (std::uint32_t line = 0u;
             line < config.maximumLineSearchSteps; ++line) {
            candidate = position;
            for (std::size_t node = 0u; node < nodes.size(); ++node) {
                if (nodes[node].fixed) continue;
                const std::size_t base = 3u * freeIndex[node];
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    candidate[node][axis] += alpha * step[base + axis];
                }
            }
            if (evaluate(nodes, elements, loads, candidate,
                         candidateEvaluation) &&
                candidateEvaluation.potentialEnergy <=
                    evaluation.potentialEnergy -
                    config.armijoFraction * alpha * descent) {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            return fail(NumiHumanTensionNetworkStatus::didNotConverge);
        }
        position = std::move(candidate);
        evaluation = std::move(candidateEvaluation);
    }
    if (evaluation.maximumFreeResidual > config.forceTolerance) {
        return fail(NumiHumanTensionNetworkStatus::didNotConverge,
                    completedIterations);
    }

    NumiHumanTensionNetworkResult staged;
    staged.position = position;
    staged.elementTension = evaluation.tension;
    staged.nodeResidualForce = evaluation.residual;
    staged.fixedReactionForce.assign(nodes.size(), {});
    staged.strainEnergy = evaluation.strainEnergy;
    staged.potentialEnergy = evaluation.potentialEnergy;
    staged.maximumFreeNodeResidual = evaluation.maximumFreeResidual;
    staged.activeElementCount = evaluation.activeElementCount;
    staged.completedIterations = completedIterations;
    Vec3 externalForce{};
    Vec3 externalMoment{};
    for (const auto& load : loads) {
        externalForce = add(externalForce, load.force);
        externalMoment = add(
            externalMoment, cross(position[load.nodeIndex], load.force));
    }
    for (std::size_t node = 0u; node < nodes.size(); ++node) {
        if (!nodes[node].fixed) continue;
        staged.fixedReactionForce[node] = scale(evaluation.residual[node], -1.0);
        staged.forceClosureResidual = add(
            staged.forceClosureResidual, staged.fixedReactionForce[node]);
        staged.momentClosureResidual = add(
            staged.momentClosureResidual,
            cross(position[node], staged.fixedReactionForce[node]));
    }
    staged.forceClosureResidual = add(staged.forceClosureResidual, externalForce);
    staged.momentClosureResidual = add(staged.momentClosureResidual, externalMoment);
    if (!finite(staged.forceClosureResidual) ||
        !finite(staged.momentClosureResidual)) {
        return fail(NumiHumanTensionNetworkStatus::nonfiniteResult);
    }
    result = std::move(staged);
    return {};
}

const char* numiHumanTensionNetworkStatusName(
    const NumiHumanTensionNetworkStatus status
) noexcept {
    switch (status) {
    case NumiHumanTensionNetworkStatus::success: return "success";
    case NumiHumanTensionNetworkStatus::invalidTopology: return "invalidTopology";
    case NumiHumanTensionNetworkStatus::invalidMaterial: return "invalidMaterial";
    case NumiHumanTensionNetworkStatus::invalidLoad: return "invalidLoad";
    case NumiHumanTensionNetworkStatus::singularSystem: return "singularSystem";
    case NumiHumanTensionNetworkStatus::didNotConverge: return "didNotConverge";
    case NumiHumanTensionNetworkStatus::nonfiniteResult: return "nonfiniteResult";
    }
    return "unknown";
}

} // namespace metalrobo
