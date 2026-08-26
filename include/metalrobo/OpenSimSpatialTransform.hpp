#pragma once

#include "metalrobo/OpenSimFunction.hpp"
#include "metalrobo/opensim_spatial_transform_gpu.h"

#include <array>
#include <cstdint>
#include <vector>

namespace metalrobo {

// This models the subset used by every CustomJoint in the pinned
// RajagopalLaiUhlrich2023 source: six SpatialTransform axes, with zero or one
// generalized coordinate selected by each axis function. It preserves the
// FunctionBased order (rotation X/Y/Z slots first, then translation X/Y/Z
// slots) without inventing a serial URDF chain or massless intermediate body.
inline constexpr std::uint32_t kOpenSimNoCoordinate = 0xffffffffu;
inline constexpr std::size_t kOpenSimSpatialAxisCount = 6u;

struct OpenSimSpatialAxisDefinition {
    // Axis expressed in the source CustomJoint frame. It is normalized at
    // compilation, matching Simbody's FunctionBased constructor.
    std::array<double, 3> axis{};
    // kOpenSimNoCoordinate is valid only for a Constant function.
    std::uint32_t coordinateIndex = kOpenSimNoCoordinate;
    OpenSimFunctionDefinition function{};
};

struct OpenSimSpatialTransformDefinition {
    std::uint32_t coordinateCount = 0u;
    std::array<OpenSimSpatialAxisDefinition, kOpenSimSpatialAxisCount> axes{};
};

enum class OpenSimSpatialTransformStatus : std::uint32_t {
    success = 0u,
    invalidDefinition,
    invalidCoordinates,
    nonfiniteInput,
    nonfiniteResult,
};

struct CompiledOpenSimSpatialAxis {
    std::array<double, 3> axis{};
    std::uint32_t coordinateIndex = kOpenSimNoCoordinate;
    CompiledOpenSimFunction function{};
};

struct CompiledOpenSimSpatialTransform {
    std::uint32_t coordinateCount = 0u;
    std::array<CompiledOpenSimSpatialAxis, kOpenSimSpatialAxisCount> axes{};
};

struct OpenSimSpatialTransformCompilation {
    OpenSimSpatialTransformStatus status = OpenSimSpatialTransformStatus::success;
    CompiledOpenSimSpatialTransform transform{};

    [[nodiscard]] bool succeeded() const noexcept {
        return status == OpenSimSpatialTransformStatus::success;
    }
};

// A spatial vector is expressed in the parent CustomJoint frame. The six
// entries in motionSubspace/Hdot are ordered by the source coordinate index.
struct OpenSimSpatialVector {
    std::array<double, 3> angular{};
    std::array<double, 3> linear{};
};

struct OpenSimSpatialTransformEvaluation {
    OpenSimSpatialTransformStatus status = OpenSimSpatialTransformStatus::success;
    // Row-major parent-frame rotation constructed as R0 * R1 * R2.
    std::array<double, 9> rotation{};
    std::array<double, 3> translation{};
    std::vector<OpenSimSpatialVector> motionSubspace;
    std::vector<OpenSimSpatialVector> motionSubspaceDot;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == OpenSimSpatialTransformStatus::success;
    }
};

[[nodiscard]] OpenSimSpatialTransformCompilation compileOpenSimSpatialTransform(
    const OpenSimSpatialTransformDefinition& definition
);

// Evaluates X_FM, H, and Hdot for source-order FunctionBased kinematics.
// Coordinate velocities are qdot, one entry per source coordinate.
[[nodiscard]] OpenSimSpatialTransformEvaluation evaluateOpenSimSpatialTransform(
    const CompiledOpenSimSpatialTransform& transform,
    const std::vector<double>& coordinates,
    const std::vector<double>& coordinateVelocities
);

// Converts immutable source-derived function tables into the fixed-capacity
// Metal program. This is a compiler boundary only; the program is evaluated
// by Metal but is not yet admitted as an ABA joint type.
[[nodiscard]] OpenSimSpatialTransformStatus packOpenSimSpatialTransformGPU(
    const CompiledOpenSimSpatialTransform& transform,
    MROpenSimSpatialTransformGPU& program
);

// Validates and decodes a canonical fixed-capacity Metal program sidecar.
// A successful decode round-trips byte-for-byte through the packer. This is
// a program-loader boundary only; it does not admit the transform as an ABA
// joint type.
[[nodiscard]] OpenSimSpatialTransformCompilation unpackOpenSimSpatialTransformGPU(
    const MROpenSimSpatialTransformGPU& program
);

[[nodiscard]] const char* openSimSpatialTransformStatusName(
    OpenSimSpatialTransformStatus status
) noexcept;

} // namespace metalrobo
