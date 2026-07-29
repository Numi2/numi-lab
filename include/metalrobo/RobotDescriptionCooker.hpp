#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

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
    invalidMimic,
    invalidTransmission,
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
    // When transmissions exist, only their movable joints are actuated.
    // URDFs without transmissions retain the all-movable policy.
    bool respectTransmissions = true;
    std::uint32_t collisionGroup = 1u;
    std::uint32_t collisionMask = ~0u;
    // Relative mesh filenames resolve from this directory. File-based cooks
    // default it to the URDF's parent directory.
    std::filesystem::path meshAssetRoot;
    // package://name/path searches these roots in authored order, accepting
    // either <root>/<name>/path or <root>/path when root itself is the named
    // package. Resolution is local-only and deterministic.
    std::vector<std::filesystem::path> packageSearchRoots;
};

struct RobotDescriptionDiagnostics {
    RobotDescriptionStatus status =
        RobotDescriptionStatus::success;
    std::uint32_t linkCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t colliderCount = 0u;
    std::uint32_t exclusionCount = 0u;
    std::uint32_t mimicConstraintCount = 0u;
    std::uint32_t transmissionJointCount = 0u;
    std::uint32_t passiveJointCount = 0u;
    std::uint32_t meshAssetCount = 0u;
    std::uint32_t meshVertexCount = 0u;
    std::uint32_t meshTriangleCount = 0u;
    std::uint64_t sourceFingerprint = 0u;
    std::string sourceName;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == RobotDescriptionStatus::success;
    }
};

// Deterministic URDF/SRDF cooker. The initial executable surface supports
// fixed, revolute, continuous, and prismatic tree joints; URDF transmissions
// and mimic gears; SRDF passive joints and collision exclusions; sphere, box,
// cylinder, and capsule collision geometry; plus closed convex OBJ/STL mesh
// collision resolved through explicit local or package roots. Articulated
// concave meshes fail explicitly. Output is unchanged on every failure.
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
