#pragma once

#include <cstdint>
#include <vector>

namespace metalrobo {

// OpenSim Function kinds used by the pinned Rajagopal CustomJoint
// SpatialTransforms. This is deliberately a small host-side value/derivative
// primitive; a future function-based articulated joint owns frame composition,
// coordinate selection, generalized-force projection, and device-side storage.
enum class OpenSimFunctionKind : std::uint32_t {
    constant = 0u,
    linear,
    polynomial,
    simmSpline,
};

enum class OpenSimFunctionStatus : std::uint32_t {
    success = 0u,
    invalidDefinition,
    nonfiniteArgument,
    nonfiniteResult,
};

struct OpenSimFunctionDefinition {
    OpenSimFunctionKind kind = OpenSimFunctionKind::constant;
    // constant: one value; linear: slope then intercept; polynomial:
    // descending-power coefficients, as serialized by OpenSim.
    std::vector<double> coefficients;
    // SimmSpline knots. Values outside the knot interval extrapolate with the
    // source spline's endpoint slope, matching OpenSim SimmSpline semantics.
    std::vector<double> abscissae;
    std::vector<double> ordinates;
};

struct OpenSimFunctionEvaluation {
    OpenSimFunctionStatus status = OpenSimFunctionStatus::success;
    double value = 0.0;
    double derivative = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == OpenSimFunctionStatus::success;
    }
};

[[nodiscard]] OpenSimFunctionEvaluation evaluateOpenSimFunction(
    const OpenSimFunctionDefinition& definition,
    double argument
);

[[nodiscard]] const char* openSimFunctionStatusName(
    OpenSimFunctionStatus status
) noexcept;

} // namespace metalrobo
