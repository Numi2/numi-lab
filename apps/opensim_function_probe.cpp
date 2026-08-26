#include "metalrobo/OpenSimFunction.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(
    const double actual,
    const double expected,
    const std::string& label
) {
    require(
        std::abs(actual - expected) <= 1.0e-12,
        label + " mismatch"
    );
}

metalrobo::OpenSimFunctionEvaluation evaluate(
    const metalrobo::OpenSimFunctionDefinition& definition,
    const double argument
) {
    const auto result = metalrobo::evaluateOpenSimFunction(definition, argument);
    require(
        result.succeeded(),
        std::string("OpenSim function failed: ") +
            metalrobo::openSimFunctionStatusName(result.status)
    );
    return result;
}

} // namespace

int main() {
    try {
        using metalrobo::OpenSimFunctionDefinition;
        using metalrobo::OpenSimFunctionKind;

        const auto constant = evaluate(
            {.kind = OpenSimFunctionKind::constant, .coefficients = {4.0}},
            123.0
        );
        requireNear(constant.value, 4.0, "constant value");
        requireNear(constant.derivative, 0.0, "constant derivative");

        const auto linear = evaluate(
            {.kind = OpenSimFunctionKind::linear, .coefficients = {2.0, 1.0}},
            0.25
        );
        requireNear(linear.value, 1.5, "linear value");
        requireNear(linear.derivative, 2.0, "linear derivative");

        const auto polynomial = evaluate(
            {.kind = OpenSimFunctionKind::polynomial, .coefficients = {3.0, 2.0, 1.0}},
            0.25
        );
        requireNear(polynomial.value, 1.6875, "polynomial value");
        requireNear(polynomial.derivative, 3.5, "polynomial derivative");

        const OpenSimFunctionDefinition spline{
            .kind = OpenSimFunctionKind::simmSpline,
            .abscissae = {0.0, 1.0},
            .ordinates = {0.0, 2.0},
        };
        const auto splineInterior = evaluate(spline, 0.25);
        const auto splineExtrapolated = evaluate(spline, -0.5);
        requireNear(splineInterior.value, 0.5, "spline interior value");
        requireNear(splineInterior.derivative, 2.0, "spline interior derivative");
        requireNear(splineExtrapolated.value, -1.0, "spline extrapolated value");
        requireNear(splineExtrapolated.derivative, 2.0, "spline extrapolated derivative");
        const auto compiledSpline = metalrobo::compileOpenSimFunction(spline);
        require(compiledSpline.succeeded(), "spline compilation failed");
        const auto repeatedSpline = metalrobo::evaluateOpenSimFunction(
            compiledSpline.function,
            0.25
        );
        require(repeatedSpline.succeeded(), "compiled spline evaluation failed");
        requireNear(repeatedSpline.value, splineInterior.value, "compiled spline value");
        requireNear(
            repeatedSpline.derivative,
            splineInterior.derivative,
            "compiled spline derivative"
        );

        const auto invalid = metalrobo::evaluateOpenSimFunction(
            OpenSimFunctionDefinition{
                .kind = OpenSimFunctionKind::simmSpline,
                .abscissae = {0.0, 0.0},
                .ordinates = {0.0, 1.0},
            },
            0.0
        );
        require(
            invalid.status == metalrobo::OpenSimFunctionStatus::invalidDefinition,
            "invalid spline was accepted"
        );

        std::cout << "opensim_function=ok"
                  << " linear_value=" << linear.value
                  << " spline_value=" << splineInterior.value
                  << " spline_derivative=" << splineInterior.derivative
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "opensim_function=failed " << error.what() << '\n';
        return 1;
    }
}
