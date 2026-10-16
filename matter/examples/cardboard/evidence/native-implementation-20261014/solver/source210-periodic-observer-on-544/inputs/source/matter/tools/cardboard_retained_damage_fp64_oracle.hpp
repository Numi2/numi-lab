#pragma once

// Independent FP64 interpreter and local-projection reference for the
// retained-matrix native instrument. It consumes the compiled scalar programs
// but uses host double arithmetic and re-solves the complete local history.

#include "numi/matter/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::cardboard::retained_damage::fp64 {

using Matrix = std::array<double, 9>;
using State = std::vector<double>;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Reconstruct an F that was authored as two binary32 words. Each word is
// rounded to the exact float supplied to the native kernel, widened, then
// summed in FP64. This is the oracle's explicit replay of the production
// high-plus-low deformation representation.
inline Matrix combineFloatWords(const Matrix& high, const Matrix& low) {
    Matrix result{};
    for (std::size_t i = 0u; i < result.size(); ++i) {
        const float highWord = static_cast<float>(high[i]);
        const float lowWord = static_cast<float>(low[i]);
        require(std::isfinite(highWord) && std::isfinite(lowWord),
                "FP64 oracle received a nonfinite deformation word");
        result[i] = static_cast<double>(highWord) + static_cast<double>(lowWord);
    }
    return result;
}

struct CompensatedWords {
    float high = 0.0f;
    float low = 0.0f;
};

// Form base +/- epsilon * direction in the same ordered binary32 high/low
// representation supplied to the native material evaluator. The FMA
// residuals preserve the product terms before the final high/low renormalize.
inline CompensatedWords perturbFloatWords(const float baseHigh,
                                          const float baseLow,
                                          const float direction,
                                          const double epsilon,
                                          const double sign) {
    require(std::isfinite(baseHigh) && std::isfinite(baseLow) &&
            std::isfinite(direction) && std::isfinite(epsilon) &&
            std::isfinite(sign) && epsilon > 0.0 &&
            (sign == -1.0 || sign == 1.0),
            "invalid compensated deformation perturbation input");
    const float epsilonHigh = static_cast<float>(epsilon);
    const float epsilonLow = static_cast<float>(
        epsilon - static_cast<double>(epsilonHigh));
    require(std::isfinite(epsilonHigh) && std::isfinite(epsilonLow),
            "compensated perturbation epsilon is outside binary32 range");
    const float productHigh = epsilonHigh * direction;
    const float productHighError = std::fma(epsilonHigh, direction, -productHigh);
    const float productLow = epsilonLow * direction;
    const float productLowError = std::fma(epsilonLow, direction, -productLow);
    double exact = static_cast<double>(baseHigh);
    exact += static_cast<double>(baseLow);
    exact += sign * static_cast<double>(productHigh);
    exact += sign * static_cast<double>(productHighError);
    exact += sign * static_cast<double>(productLow);
    exact += sign * static_cast<double>(productLowError);
    require(std::isfinite(exact), "compensated perturbation result is nonfinite");
    const float high = static_cast<float>(exact);
    require(std::isfinite(high), "compensated perturbation exceeds binary32 range");
    const float low = static_cast<float>(exact - static_cast<double>(high));
    require(std::isfinite(low), "compensated perturbation low word is nonfinite");
    return {high, low};
}

inline double eval(const numi::matter::ScalarBytecode& code,
                   const numi::matter::MaterialProgram& material,
                   const Matrix& f, const Matrix& df, const State& accepted,
                   const State& candidate, double dt) {
    std::vector<double> stack;
    stack.reserve(code.maximumStack);
    const auto pop = [&]() {
        require(!stack.empty(), "FP64 bytecode stack underflow");
        const double value = stack.back();
        stack.pop_back();
        return value;
    };
    for (const auto& instruction : code.instructions) {
        double value = 0.0;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value = instruction.immediate.x; break;
        case NM_EXPR_PARAMETER:
            value = static_cast<float>(material.parameters.at(instruction.index).defaultValue);
            break;
        case NM_EXPR_F: value = f.at(instruction.index); break;
        case NM_EXPR_DF:
        case NM_EXPR_RATE: value = df.at(instruction.index); break;
        case NM_EXPR_STATE: value = accepted.at(instruction.index); break;
        case NM_EXPR_NEXT_STATE: value = candidate.at(instruction.index); break;
        case NM_EXPR_DT: value = dt; break;
        case NM_EXPR_TEMPERATURE: value = 293.15f; break;
        case NM_EXPR_ADD: { const double b = pop(); value = pop() + b; break; }
        case NM_EXPR_SUBTRACT: { const double b = pop(); value = pop() - b; break; }
        case NM_EXPR_MULTIPLY: { const double b = pop(); value = pop() * b; break; }
        case NM_EXPR_DIVIDE: {
            const double b = pop();
            const double a = pop();
            require(std::isfinite(b) && b != 0.0, "FP64 bytecode divide by zero");
            value = a / b;
            break;
        }
        case NM_EXPR_NEGATE: value = -pop(); break;
        case NM_EXPR_LOG: value = std::log(pop()); break;
        case NM_EXPR_EXP: value = std::exp(pop()); break;
        case NM_EXPR_SQRT: value = std::sqrt(pop()); break;
        case NM_EXPR_ABS: value = std::abs(pop()); break;
        case NM_EXPR_MIN: { const double b = pop(); value = std::min(pop(), b); break; }
        case NM_EXPR_MAX: { const double b = pop(); value = std::max(pop(), b); break; }
        case NM_EXPR_POW_INTEGER: {
            const double x = pop(); value = std::pow(x, instruction.integer); break;
        }
        case NM_EXPR_CLAMP: {
            const double high = pop();
            const double low = pop();
            value = std::clamp(pop(), low, high);
            break;
        }
        case NM_EXPR_SELECT_NONNEGATIVE: {
            const double negative = pop();
            const double positive = pop();
            value = pop() >= 0.0 ? positive : negative;
            break;
        }
        case NM_EXPR_EXPM1_MINUS_X: {
            const double x = pop(); value = std::expm1(x) - x; break;
        }
        default:
            throw std::runtime_error("unsupported opcode in FP64 retained-damage oracle");
        }
        require(std::isfinite(value), "nonfinite FP64 bytecode value");
        stack.push_back(value);
    }
    require(stack.size() == 1u, "FP64 bytecode did not produce one scalar");
    return stack.front();
}

inline bool solveDense(std::vector<double> a, State& b, unsigned n) {
    for (unsigned col = 0; col < n; ++col) {
        unsigned pivot = col;
        for (unsigned row = col + 1; row < n; ++row)
            if (std::abs(a[row * n + col]) > std::abs(a[pivot * n + col])) pivot = row;
        const double magnitude = std::abs(a[pivot * n + col]);
        if (!(std::isfinite(magnitude) && magnitude > 1.0e-28)) return false;
        if (pivot != col) {
            for (unsigned j = col; j < n; ++j)
                std::swap(a[col * n + j], a[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }
        for (unsigned row = col + 1; row < n; ++row) {
            const double factor = a[row * n + col] / a[col * n + col];
            for (unsigned j = col; j < n; ++j) a[row * n + j] -= factor * a[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
        double value = b[static_cast<unsigned>(row)];
        for (unsigned col = static_cast<unsigned>(row) + 1; col < n; ++col)
            value -= a[static_cast<unsigned>(row) * n + col] * b[col];
        b[static_cast<unsigned>(row)] = value /
            a[static_cast<unsigned>(row) * n + static_cast<unsigned>(row)];
        if (!std::isfinite(b[static_cast<unsigned>(row)])) return false;
    }
    return true;
}

struct Projection {
    State state;
    double residualInf = 0.0;
    unsigned iterations = 0;
};

inline Projection project(const numi::matter::ConstitutiveProgram& program,
                          const Matrix& f, const State& accepted, double dt,
                          double tolerance = 2.0e-11) {
    const unsigned n = static_cast<unsigned>(accepted.size());
    require(program.implicitResiduals.size() == n &&
            program.implicitJacobians.size() == n * n,
            "FP64 oracle got a malformed implicit system");
    const Matrix zero{};
    State candidate = accepted;
    for (unsigned i = 0; i < n; ++i)
        candidate[i] = eval(program.stateUpdates.at(i), program.material,
                            f, zero, accepted, candidate, dt);
    const auto residuals = [&](const State& state) {
        State result(n);
        for (unsigned row = 0; row < n; ++row)
            result[row] = eval(program.implicitResiduals[row], program.material,
                               f, zero, accepted, state, dt);
        return result;
    };
    const auto normInf = [](const State& values) {
        double norm = 0.0;
        for (double value : values) norm = std::max(norm, std::abs(value));
        return norm;
    };
    double norm = normInf(residuals(candidate));
    unsigned iteration = 0;
    for (; iteration < 96u && norm > tolerance; ++iteration) {
        const State r = residuals(candidate);
        std::vector<double> jacobian(n * n);
        for (unsigned row = 0; row < n; ++row)
            for (unsigned col = 0; col < n; ++col)
                jacobian[row * n + col] = eval(program.implicitJacobians[row * n + col],
                    program.material, f, zero, accepted, candidate, dt);
        State delta(n);
        for (unsigned i = 0; i < n; ++i) delta[i] = -r[i];
        require(solveDense(std::move(jacobian), delta, n),
                "FP64 oracle local Jacobian is singular");
        bool acceptedStep = false;
        for (double alpha = 1.0; alpha >= 1.0 / 65536.0; alpha *= 0.5) {
            State trial = candidate;
            for (unsigned i = 0; i < n; ++i) trial[i] += alpha * delta[i];
            const double trialNorm = normInf(residuals(trial));
            if (std::isfinite(trialNorm) && trialNorm < norm) {
                candidate = std::move(trial);
                norm = trialNorm;
                acceptedStep = true;
                break;
            }
        }
        require(acceptedStep, "FP64 oracle local projection line search failed");
    }
    require(norm <= tolerance, "FP64 oracle local projection did not converge");
    return {std::move(candidate), norm, iteration};
}

inline Matrix stress(const numi::matter::ConstitutiveProgram& program,
                     const Matrix& f, const State& accepted,
                     const State& candidate, double dt) {
    Matrix result{};
    const Matrix zero{};
    for (unsigned i = 0; i < 9; ++i)
        result[i] = eval(program.stress[i], program.material, f, zero,
                         accepted, candidate, dt);
    return result;
}

inline Matrix tangentAction(const numi::matter::ConstitutiveProgram& program,
                            const Matrix& f, const State& accepted,
                            const State& candidate, const Matrix& direction,
                            double dt) {
    const unsigned n = static_cast<unsigned>(accepted.size());
    const Matrix zero{};
    std::vector<double> jacobian(n * n);
    State stateDirection(n);
    for (unsigned row = 0; row < n; ++row) {
        for (unsigned col = 0; col < n; ++col)
            jacobian[row * n + col] = eval(program.implicitJacobians[row * n + col],
                program.material, f, zero, accepted, candidate, dt);
        stateDirection[row] = -eval(program.implicitDeformationDirections[row],
            program.material, f, direction, accepted, candidate, dt);
    }
    require(solveDense(std::move(jacobian), stateDirection, n),
            "FP64 oracle tangent Jacobian is singular");
    Matrix result{};
    for (unsigned component = 0; component < 9; ++component) {
        double value = eval(program.tangentVector[component], program.material,
                            f, direction, accepted, candidate, dt);
        for (unsigned state = 0; state < n; ++state)
            value += eval(program.stressStateDerivatives[component * n + state],
                program.material, f, zero, accepted, candidate, dt) * stateDirection[state];
        result[component] = value;
    }
    return result;
}

inline double norm2(const Matrix& a) {
    double sum = 0.0;
    for (double value : a) sum += value * value;
    return std::sqrt(sum);
}

inline double differenceNorm(const Matrix& a, const Matrix& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

} // namespace numi::cardboard::retained_damage::fp64
