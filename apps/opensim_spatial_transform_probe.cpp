#include "metalrobo/OpenSimSpatialTransform.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using metalrobo::OpenSimFunctionDefinition;
using metalrobo::OpenSimFunctionKind;
using metalrobo::OpenSimSpatialAxisDefinition;
using metalrobo::OpenSimSpatialTransformDefinition;
using metalrobo::OpenSimSpatialTransformEvaluation;
using metalrobo::kOpenSimNoCoordinate;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string& label
) {
    require(std::abs(actual - expected) <= tolerance, label + " mismatch");
}

OpenSimSpatialAxisDefinition axis(
    const std::array<double, 3> direction,
    const std::uint32_t coordinate,
    const OpenSimFunctionDefinition& function
) {
    return {
        .axis = direction,
        .coordinateIndex = coordinate,
        .function = function,
    };
}

OpenSimSpatialTransformDefinition rajagopalWalkerKnee() {
    // Exact source coefficients for walker_knee_r from
    // RajagopalLaiUhlrich2023.osim (SHA-256 recorded by numilab-human).
    OpenSimSpatialTransformDefinition source{};
    source.coordinateCount = 1u;
    source.axes = {
        axis(
            {1.0, 0.0, 0.0},
            0u,
            {.kind = OpenSimFunctionKind::linear, .coefficients = {1.0, 0.0}}
        ),
        axis(
            {0.0, 0.0, 1.0},
            0u,
            {.kind = OpenSimFunctionKind::polynomial,
             .coefficients = {
                 0.010832094539863, -0.025218325501241,
                 -0.032847810398852, 0.079100011967027,
                 -1.473252350900463e-08,
             }}
        ),
        axis(
            {0.0, 1.0, 0.0},
            0u,
            {.kind = OpenSimFunctionKind::polynomial,
             .coefficients = {
                 0.025165762727423, -0.16948005139054,
                 0.369499348688249, -4.430358308836305e-08,
             }}
        ),
        axis(
            {1.0, 0.0, 0.0},
            0u,
            {.kind = OpenSimFunctionKind::polynomial,
             .coefficients = {
                 0.0001590447878850381, -0.001015149915669,
                 0.001817510974968, 2.64142664519923e-05,
                 -7.746563532471892e-07,
             }}
        ),
        axis(
            {0.0, 1.0, 0.0},
            0u,
            {.kind = OpenSimFunctionKind::polynomial,
             .coefficients = {
                 -0.0005796878052338684, 0.005079765745626,
                 -0.011442375726364, 0.003936908668844,
                 -2.516350383213525e-05,
             }}
        ),
        axis(
            {0.0, 0.0, 1.0},
            0u,
            {.kind = OpenSimFunctionKind::polynomial,
             .coefficients = {
                 0.001208086889206, -0.004453611224706,
                 0.000611649407298173, 0.006265429606387,
                 -1.461912533723326e-05,
             }}
        ),
    };
    return source;
}

void requireMotionNear(
    const OpenSimSpatialTransformEvaluation& actual,
    const OpenSimSpatialTransformEvaluation& expected,
    const double tolerance,
    const std::string& label
) {
    require(
        actual.motionSubspace.size() == expected.motionSubspace.size() &&
            actual.motionSubspaceDot.size() == expected.motionSubspaceDot.size(),
        label + " dimensions"
    );
    for (std::size_t coordinate = 0u;
         coordinate < actual.motionSubspace.size();
         ++coordinate) {
        for (std::size_t component = 0u; component < 3u; ++component) {
            requireNear(
                actual.motionSubspace[coordinate].angular[component],
                expected.motionSubspace[coordinate].angular[component],
                tolerance,
                label + " angular H"
            );
            requireNear(
                actual.motionSubspace[coordinate].linear[component],
                expected.motionSubspace[coordinate].linear[component],
                tolerance,
                label + " linear H"
            );
            requireNear(
                actual.motionSubspaceDot[coordinate].angular[component],
                expected.motionSubspaceDot[coordinate].angular[component],
                tolerance,
                label + " angular Hdot"
            );
            requireNear(
                actual.motionSubspaceDot[coordinate].linear[component],
                expected.motionSubspaceDot[coordinate].linear[component],
                tolerance,
                label + " linear Hdot"
            );
        }
    }
}

} // namespace

int main() {
    try {
        const auto compiled = metalrobo::compileOpenSimSpatialTransform(
            rajagopalWalkerKnee()
        );
        require(compiled.succeeded(), "walker knee compilation failed");
        MROpenSimSpatialTransformGPU packed{};
        require(
            metalrobo::packOpenSimSpatialTransformGPU(compiled.transform, packed) ==
                metalrobo::OpenSimSpatialTransformStatus::success,
            "walker knee GPU packing failed"
        );
        require(
            packed.abiVersion == MR_OPENSIM_SPATIAL_TRANSFORM_GPU_ABI_VERSION &&
                packed.coordinateCount == 1u &&
                packed.axes[3u].kind == MR_OPENSIM_FUNCTION_POLYNOMIAL &&
                packed.axes[3u].coefficientCount == 5u,
            "walker knee GPU payload mismatch"
        );

        constexpr double coordinate = 0.43;
        constexpr double velocity = -0.71;
        const auto evaluated = metalrobo::evaluateOpenSimSpatialTransform(
            compiled.transform,
            {coordinate},
            {velocity}
        );
        require(evaluated.succeeded(), "walker knee evaluation failed");
        requireNear(evaluated.translation[0], 0.0002713671579462591, 1.0e-15, "knee tx");
        requireNear(evaluated.translation[1], -0.00006392948537864571, 1.0e-15, "knee ty");
        requireNear(evaluated.translation[2], 0.0024798183998249526, 1.0e-15, "knee tz");

        // Hdot is checked independently by a centred directional derivative
        // of H along qdot. This catches both the source-order rotation terms
        // and nonlinear Function derivatives without making a dynamics claim.
        constexpr double step = 1.0e-6;
        const auto before = metalrobo::evaluateOpenSimSpatialTransform(
            compiled.transform,
            {coordinate - step * velocity},
            {velocity}
        );
        const auto after = metalrobo::evaluateOpenSimSpatialTransform(
            compiled.transform,
            {coordinate + step * velocity},
            {velocity}
        );
        require(before.succeeded() && after.succeeded(), "finite-difference evaluation failed");
        OpenSimSpatialTransformEvaluation finiteDifference{};
        finiteDifference.motionSubspace.resize(1u);
        finiteDifference.motionSubspaceDot.resize(1u);
        for (std::size_t component = 0u; component < 3u; ++component) {
            finiteDifference.motionSubspace[0u].angular[component] =
                evaluated.motionSubspace[0u].angular[component];
            finiteDifference.motionSubspace[0u].linear[component] =
                evaluated.motionSubspace[0u].linear[component];
            finiteDifference.motionSubspaceDot[0u].angular[component] =
                (after.motionSubspace[0u].angular[component] -
                 before.motionSubspace[0u].angular[component]) / (2.0 * step);
            finiteDifference.motionSubspaceDot[0u].linear[component] =
                (after.motionSubspace[0u].linear[component] -
                 before.motionSubspace[0u].linear[component]) / (2.0 * step);
        }
        requireMotionNear(evaluated, finiteDifference, 1.0e-8, "walker knee");

        const auto invalid = metalrobo::compileOpenSimSpatialTransform({
            .coordinateCount = 1u,
            .axes = {
                axis({1.0, 0.0, 0.0}, 0u, {.kind = OpenSimFunctionKind::linear, .coefficients = {1.0, 0.0}}),
                axis({1.0, 0.0, 0.0}, 0u, {.kind = OpenSimFunctionKind::linear, .coefficients = {1.0, 0.0}}),
                axis({0.0, 1.0, 0.0}, 0u, {.kind = OpenSimFunctionKind::linear, .coefficients = {1.0, 0.0}}),
                axis({1.0, 0.0, 0.0}, kOpenSimNoCoordinate, {.kind = OpenSimFunctionKind::constant, .coefficients = {0.0}}),
                axis({0.0, 1.0, 0.0}, kOpenSimNoCoordinate, {.kind = OpenSimFunctionKind::constant, .coefficients = {0.0}}),
                axis({0.0, 0.0, 1.0}, kOpenSimNoCoordinate, {.kind = OpenSimFunctionKind::constant, .coefficients = {0.0}}),
            },
        });
        require(!invalid.succeeded(), "colinear source axes were accepted");

        std::cout << "opensim_spatial_transform=ok"
                  << " tx=" << evaluated.translation[0]
                  << " h_angular_x=" << evaluated.motionSubspace[0u].angular[0u]
                  << " hdot_linear_x=" << evaluated.motionSubspaceDot[0u].linear[0u]
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "opensim_spatial_transform=failed " << error.what() << '\n';
        return 1;
    }
}
