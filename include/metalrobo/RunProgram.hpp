#pragma once

#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/InteractionPack.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/WorldCompiler.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {

enum class RobotSemanticKind : std::uint32_t {
    body = 0u,
    joint = 1u,
    dof = 2u,
};

struct RobotSemanticRole {
    std::string id;
    RobotSemanticKind kind = RobotSemanticKind::body;
    std::vector<std::string> members;
};

// A robot is mechanics plus stable semantic roles and default sensor mounts.
// Tasks and policies never own the mechanics. A pack may be copied and
// configured without changing the bundled default.
struct RobotPack {
    std::string id;
    std::uint64_t revision = 1u;
    std::string sourceRepository;
    std::string sourceRevision;
    std::string license;
    EngineModel mechanics;
    std::vector<MRBodyStateGPU> defaultSceneBodies;
    std::uint32_t primaryArticulationIndex = 0u;
    std::vector<std::string> capabilities;
    std::vector<RobotSemanticRole> roles;
};

struct SceneObject {
    std::string id;
    std::string semanticClass;
    MRWorldAssetRole role = MR_WORLD_ASSET_CLUTTER;
    MRWorldRenderRepresentation render = MR_WORLD_RENDER_NONE;
    MRWorldCollisionRepresentation collision =
        MR_WORLD_COLLISION_PRIMITIVES;
    MRWorldDynamicsRepresentation dynamics = MR_WORLD_DYNAMICS_RIGID;
    EngineModel mechanics;
    std::vector<MRBodyStateGPU> defaultBodyStates;
};

// Objects are independent physical components. The compiler deterministically
// composes their topology with the selected robot before Metal sees the run.
struct ScenePack {
    std::string id;
    std::vector<SceneObject> objects;
};

struct MountedSensor {
    SensorSpec sensor;
    // A robot body role. Exactly one member is required for a physical mount.
    std::string mountRole;
};

// Physical mounts, timing, calibration, noise and dropout have one owner.
// TaskPack may consume the resulting observation, but does not own the sensor.
struct SensorPack {
    std::string id;
    std::vector<MountedSensor> mounted;
    std::vector<SensorSpec> worldSensors;
};

// Reality variations use the existing device-resident WorldProgram targets:
// friction, restitution, damping, payload, actuator gains/latency, sensor
// pose/noise/dropout and appearance. Unsupported failure modes are rejected;
// they are never approximated silently.
struct RealityPack {
    std::string id;
    WorldProgram program;
};

enum class TeacherKind : std::uint32_t {
    none = 0u,
    motionImagination = 1u,
    foundationActionChunk = 2u,
    demonstration = 3u,
};

// Teacher proposals are learning evidence. They cannot change gravity,
// contact, collision or simulator state outside ordinary actions/resets.
struct TeacherPack {
    std::string id;
    TeacherKind kind = TeacherKind::none;
    std::string provider;
    std::string model;
    std::string artifact;
    std::uint64_t artifactFingerprint = 0u;
};

struct RunProfile {
    std::string id;
    EngineModelComposeConfig physics;
    MetalWorldCapacityProfile capacities;
    std::uint32_t environmentCount = 1u;
    std::uint32_t controlSteps = 1u;
    std::uint32_t physicsSubsteps = 1u;
    std::uint32_t velocityIterations = 4u;
    std::uint32_t finalVelocityIterations = 2u;
    float controlTimestepSeconds = 1.0f / 50.0f;
    std::uint64_t seed = 1u;
};

struct RunManifest {
    std::string id;
    RobotPack robot;
    ScenePack scene;
    SensorPack sensors;
    TaskPack task;
    RealityPack reality;
    TeacherPack teacher;
    std::optional<InteractionPack> interactions;
    std::string interactionClip;
    std::optional<PolicyPack> policy;
    RunProfile profile;
};

enum class RunCompileStatus : std::uint32_t {
    success = 0u,
    invalidManifest,
    invalidRobot,
    unresolvedRole,
    compositionFailure,
    worldFailure,
    taskFailure,
    policyFailure,
    internalFailure,
};

struct RunCompileDiagnostics {
    RunCompileStatus status = RunCompileStatus::success;
    std::string element;
    std::string message;
    [[nodiscard]] bool succeeded() const noexcept {
        return status == RunCompileStatus::success;
    }
};

class CompiledRun {
public:
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t robotFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t sensorFingerprint() const noexcept;
    [[nodiscard]] const WorldFamily& worldFamily() const noexcept;
    [[nodiscard]] const CompiledWorld& world() const noexcept;
    [[nodiscard]] const EngineModel& model() const noexcept;
    [[nodiscard]] std::span<const MRBodyStateGPU>
    defaultSceneBodies() const noexcept;
    [[nodiscard]] const CompiledTaskProgram& task() const noexcept;
    [[nodiscard]] const CompiledPolicyProgram& policy() const noexcept;
    [[nodiscard]] const PolicyPack* boundPolicy() const noexcept;
    [[nodiscard]] const RunProfile& profile() const noexcept;
    [[nodiscard]] const TeacherPack& teacher() const noexcept;

private:
    std::uint64_t fingerprint_ = 0u;
    std::uint64_t robotFingerprint_ = 0u;
    std::uint64_t sensorFingerprint_ = 0u;
    WorldFamily worldFamily_;
    EngineModel model_;
    std::vector<MRBodyStateGPU> defaultSceneBodies_;
    CompiledWorld world_;
    CompiledTaskProgram task_;
    CompiledPolicyProgram policy_;
    std::optional<PolicyPack> boundPolicy_;
    RunProfile profile_;
    TeacherPack teacher_;

    friend RunCompileDiagnostics compileRun(
        const RunManifest&,
        CompiledRun&
    );
};

// Transactional single entry point for robot + scene + sensors + task + brain.
[[nodiscard]] RunCompileDiagnostics compileRun(
    const RunManifest& manifest,
    CompiledRun& output
);

[[nodiscard]] std::vector<std::string> builtinRobotIds();
[[nodiscard]] std::optional<RobotPack> builtinRobotPack(
    std::string_view id
);
[[nodiscard]] ScenePack makeFrankaPickPlaceScenePack();
[[nodiscard]] const char* runCompileStatusName(
    RunCompileStatus status
) noexcept;

} // namespace metalrobo
