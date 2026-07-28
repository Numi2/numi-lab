#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace metalrobo {

enum class RobotDescriptionStatus : std::uint32_t {
    success = 0u,
    ioFailure,
    malformedXml,
    invalidRobot,
    invalidTopology,
    invalidInertial,
    unsupportedJoint,
    unsupportedGeometry,
    invalidSrdf,
    capacityOverflow,
    invalidEngineModel,
    internalFailure,
};

enum class RobotDescriptionRootMode : std::uint32_t {
    fixed = 0u,
    floating = 1u,
};

struct RobotDescriptionCookOptions {
    RobotDescriptionRootMode rootMode =
        RobotDescriptionRootMode::fixed;
    mr_float4 gravityAndTimestep{
        0.0F, 0.0F, -9.81F, 1.0F / 1000.0F,
    };
    // static friction, dynamic friction, rolling, torsional.
    mr_float4 friction{0.8F, 0.6F, 0.0F, 0.0F};
    // restitution, threshold, compliance, dissipation.
    mr_float4 response{0.0F, 0.5F, 0.0F, 0.0F};
    float contactOffset = 0.002F;
    float restOffset = 0.0F;
    float defaultArmature = 0.0F;
    bool actuateMovableJoints = true;
    std::uint32_t collisionGroup = 1u;
    std::uint32_t collisionMask = ~0u;
};

struct RobotDescriptionDiagnostics {
    RobotDescriptionStatus status =
        RobotDescriptionStatus::success;
    std::uint32_t linkCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t colliderCount = 0u;
    std::uint32_t exclusionCount = 0u;
    std::uint64_t sourceFingerprint = 0u;
    std::string sourceName;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == RobotDescriptionStatus::success;
    }
};

// Deterministic URDF/SRDF cooker. The initial executable surface supports
// fixed, revolute, continuous, and prismatic tree joints plus sphere, box,
// and cylinder collision geometry. Meshes fail explicitly until their URI is
// resolved through the versioned geometry cooker. Output is unchanged on
// every failure.
[[nodiscard]] RobotDescriptionDiagnostics cookRobotDescription(
    std::string_view urdf,
    std::string_view srdf,
    EngineModel& output,
    const RobotDescriptionCookOptions& options = {},
    std::string sourceName = {}
);

[[nodiscard]] RobotDescriptionDiagnostics cookRobotDescriptionFiles(
    const std::filesystem::path& urdfPath,
    const std::filesystem::path& srdfPath,
    EngineModel& output,
    const RobotDescriptionCookOptions& options = {}
);

[[nodiscard]] const char* robotDescriptionStatusName(
    RobotDescriptionStatus status
) noexcept;

} // namespace metalrobo
