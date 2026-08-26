#include "metalrobo/OpenSimFunction.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace metalrobo {
namespace {

constexpr double kSimmTiny = 1.0e-7;
constexpr double kSimmRoundoff = 2.0e-13;

bool finite(const double value) {
    return std::isfinite(value);
}

bool finiteValues(const std::vector<double>& values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const double value) { return finite(value); }
    );
}

bool validSpline(const OpenSimFunctionDefinition& definition) {
    if (definition.abscissae.size() < 2u ||
        definition.abscissae.size() != definition.ordinates.size() ||
        !finiteValues(definition.abscissae) ||
        !finiteValues(definition.ordinates)) {
        return false;
    }
    return std::adjacent_find(
        definition.abscissae.begin(),
        definition.abscissae.end(),
        [](const double left, const double right) { return right <= left; }
    ) == definition.abscissae.end();
}

bool simmCoefficients(
    const OpenSimFunctionDefinition& definition,
    std::vector<double>& slope,
    std::vector<double>& quadratic,
    std::vector<double>& cubic
) {
    if (!validSpline(definition)) {
        return false;
    }
    const std::size_t count = definition.abscissae.size();
    slope.assign(count, 0.0);
    quadratic.assign(count, 0.0);
    cubic.assign(count, 0.0);
    const auto& x = definition.abscissae;
    const auto& y = definition.ordinates;
    if (count == 2u) {
        const double value = (y[1] - y[0]) /
            std::max(kSimmTiny, x[1] - x[0]);
        slope[0] = value;
        slope[1] = value;
        return true;
    }

    const std::size_t final = count - 1u;
    const std::size_t penultimate = count - 2u;
    cubic[0] = std::max(kSimmTiny, x[1] - x[0]);
    quadratic[1] = (y[1] - y[0]) / cubic[0];
    for (std::size_t index = 1u; index < final; ++index) {
        cubic[index] = std::max(kSimmTiny, x[index + 1u] - x[index]);
        slope[index] = 2.0 * (cubic[index - 1u] + cubic[index]);
        quadratic[index + 1u] = (y[index + 1u] - y[index]) / cubic[index];
        quadratic[index] = quadratic[index + 1u] - quadratic[index];
    }
    slope[0] = -cubic[0];
    slope[final] = -cubic[penultimate];
    quadratic[0] = 0.0;
    quadratic[final] = 0.0;
    if (count > 3u) {
        const double d31 = std::max(kSimmTiny, x[3] - x[1]);
        const double d20 = std::max(kSimmTiny, x[2] - x[0]);
        const double d1 = std::max(kSimmTiny, x[final] - x[count - 3u]);
        const double d2 = std::max(kSimmTiny, x[penultimate] - x[count - 4u]);
        const double d30 = std::max(kSimmTiny, x[3] - x[0]);
        const double d3 = std::max(kSimmTiny, x[final] - x[count - 4u]);
        quadratic[0] = (quadratic[2] / d31 - quadratic[1] / d20) *
            cubic[0] * cubic[0] / d30;
        quadratic[final] = -(quadratic[penultimate] / d1 -
            quadratic[count - 3u] / d2) * cubic[penultimate] *
            cubic[penultimate] / d3;
    }
    for (std::size_t index = 1u; index < count; ++index) {
        const double scale = cubic[index - 1u] / slope[index - 1u];
        slope[index] -= scale * cubic[index - 1u];
        quadratic[index] -= scale * quadratic[index - 1u];
    }
    quadratic[final] /= slope[final];
    for (std::size_t offset = 0u; offset < final; ++offset) {
        const std::size_t index = penultimate - offset;
        quadratic[index] = (quadratic[index] -
            cubic[index] * quadratic[index + 1u]) / slope[index];
    }
    slope[final] = (y[final] - y[penultimate]) / cubic[penultimate] +
        cubic[penultimate] * (quadratic[penultimate] + 2.0 * quadratic[final]);
    for (std::size_t index = 0u; index < final; ++index) {
        slope[index] = (y[index + 1u] - y[index]) / cubic[index] -
            cubic[index] * (quadratic[index + 1u] + 2.0 * quadratic[index]);
        cubic[index] = (quadratic[index + 1u] - quadratic[index]) / cubic[index];
        quadratic[index] *= 3.0;
    }
    quadratic[final] *= 3.0;
    cubic[final] = cubic[penultimate];
    return finiteValues(slope) && finiteValues(quadratic) && finiteValues(cubic);
}

OpenSimFunctionEvaluation result(const double value, const double derivative) {
    if (!finite(value) || !finite(derivative)) {
        return {.status = OpenSimFunctionStatus::nonfiniteResult};
    }
    return {.value = value, .derivative = derivative};
}

} // namespace

OpenSimFunctionEvaluation evaluateOpenSimFunction(
    const OpenSimFunctionDefinition& definition,
    const double argument
) {
    if (!finite(argument)) {
        return {.status = OpenSimFunctionStatus::nonfiniteArgument};
    }
    switch (definition.kind) {
    case OpenSimFunctionKind::constant:
        if (definition.coefficients.size() != 1u ||
            !finiteValues(definition.coefficients)) {
            return {.status = OpenSimFunctionStatus::invalidDefinition};
        }
        return result(definition.coefficients[0], 0.0);
    case OpenSimFunctionKind::linear:
        if (definition.coefficients.size() != 2u ||
            !finiteValues(definition.coefficients)) {
            return {.status = OpenSimFunctionStatus::invalidDefinition};
        }
        return result(
            definition.coefficients[0] * argument + definition.coefficients[1],
            definition.coefficients[0]
        );
    case OpenSimFunctionKind::polynomial: {
        if (definition.coefficients.empty() ||
            !finiteValues(definition.coefficients)) {
            return {.status = OpenSimFunctionStatus::invalidDefinition};
        }
        double value = 0.0;
        double derivative = 0.0;
        for (const double coefficient : definition.coefficients) {
            derivative = derivative * argument + value;
            value = value * argument + coefficient;
        }
        return result(value, derivative);
    }
    case OpenSimFunctionKind::simmSpline: {
        std::vector<double> slope;
        std::vector<double> quadratic;
        std::vector<double> cubic;
        if (!simmCoefficients(definition, slope, quadratic, cubic)) {
            return {.status = OpenSimFunctionStatus::invalidDefinition};
        }
        const auto& x = definition.abscissae;
        const auto& y = definition.ordinates;
        const std::size_t final = x.size() - 1u;
        if (argument < x[0]) {
            return result(y[0] + (argument - x[0]) * slope[0], slope[0]);
        }
        if (argument > x[final]) {
            return result(
                y[final] + (argument - x[final]) * slope[final], slope[final]
            );
        }
        if (std::abs(argument - x[0]) <= kSimmRoundoff) {
            return result(y[0], slope[0]);
        }
        if (std::abs(argument - x[final]) <= kSimmRoundoff) {
            return result(y[final], slope[final]);
        }
        std::size_t low = 0u;
        std::size_t high = final;
        std::size_t index = 0u;
        while (true) {
            index = (low + high) / 2u;
            if (argument < x[index]) {
                high = index;
            } else if (argument > x[index + 1u]) {
                low = index;
            } else {
                break;
            }
        }
        const double delta = argument - x[index];
        return result(
            y[index] + delta * (slope[index] + delta *
                (quadratic[index] + delta * cubic[index])),
            slope[index] + delta * (2.0 * quadratic[index] +
                3.0 * delta * cubic[index])
        );
    }
    }
    return {.status = OpenSimFunctionStatus::invalidDefinition};
}

const char* openSimFunctionStatusName(const OpenSimFunctionStatus status) noexcept {
    switch (status) {
    case OpenSimFunctionStatus::success:
        return "success";
    case OpenSimFunctionStatus::invalidDefinition:
        return "invalid_definition";
    case OpenSimFunctionStatus::nonfiniteArgument:
        return "nonfinite_argument";
    case OpenSimFunctionStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
