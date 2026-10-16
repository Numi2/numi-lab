#pragma once

#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/InteractionPack.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
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

struct MulticopterActuatorPack {
    MRMulticopterModelGPU model{};
    std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS> rotors{};
    MRMulticopterMixerGPU mixer{};
    std::string bodyRole;
    mr_float4 windVelocity{};
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
    std::vector<RobotActuatorSpec> actuators;
    std::optional<MulticopterActuatorPack> multicopter;
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
    // Exact asset ownership for a precompiled scene package such as
    // MRWorldPack. When present, these bindings describe the already-composed
    // mechanics directly; the run compiler validates them instead of
    // flattening the whole world into one synthetic robot asset.
    std::vector<WorldAsset> authoredAssets;
};

struct MountedSensor {
    SensorSpec sensor;
    // A robot body role. Exactly one member is required for a physical mount.
    std::string mountRole;
};

struct VisualSensorAssetProgram {
    std::string path;
    std::string assetId;
    std::string contentHash;
    std::uint32_t semanticId = 0u;
    std::uint32_t instanceId = 0u;
};

// Immutable resource and camera program for the device-resident visual
// sensor. File paths locate content-addressed artifacts; their hashes bind
// the compiled run and are rechecked when device resources are materialized.
struct VisualSensorProgram {
    std::vector<VisualSensorAssetProgram> assets;
    std::string environmentPath;
    std::string environmentContentHash;
    std::string rendererProfile = "sensor_fast";
    std::string cameraParentBody;
    mr_float4 cameraPosition{};
    mr_float4 cameraOrientation{0.0f, 0.0f, 0.0f, 1.0f};
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t minimumVisiblePixels = 0u;
    float verticalFieldOfViewDegrees = 0.0f;
    float nominalRateHz = 0.0f;
    std::uint64_t maximumRetainedBytes = 0u;
    std::uint32_t captureWidth = 0u;
    std::uint32_t captureHeight = 0u;
    bool capturePolicyCamera = false;
    std::uint64_t fingerprint = 0u;
};

// Content identity for the complete executable visual sensor program. Paths
// are deliberately excluded: artifacts are bound by their content hashes.
[[nodiscard]] std::uint64_t visualSensorProgramFingerprint(
    const VisualSensorProgram& program
);

// Physical mounts, timing, calibration, noise and dropout have one owner.
// TaskPack may consume the resulting observation, but does not own the sensor.
struct SensorPack {
    std::string id;
    std::vector<MountedSensor> mounted;
    std::vector<SensorSpec> worldSensors;
    std::optional<VisualSensorProgram> deviceVisual;
    // Executable observation program. These operators are resolved once by
    // the run compiler and executed by Metal after accepted physics, before
    // policy inference. TaskPack owns objectives, never sensing semantics.
    TaskObservationProgram observation;
};

// Reality variations use the existing device-resident WorldProgram targets:
// friction, restitution, damping, payload, actuator gains/latency, sensor
// pose/noise/dropout and appearance. Unsupported failure modes are rejected;
// they are never approximated silently.
struct RealityPack {
    std::string id;
    WorldProgram program;
    // Preserves the artifact identity of an imported compiled WorldProgram.
    // The semantic program still compiles and executes directly through this
    // pack; this value binds the resulting run to its persisted provenance.
    std::uint64_t sourceProgramFingerprint = 0u;
    // Executable reset program. WorldProgram is the source for scene,
    // physics, controller, camera and appearance variation. These additional
    // task-state operators cover reset semantics whose targets are not world
    // assets (joint state, action/observation delay and event scheduling).
    // Both streams execute inside the same atomic Metal reset transaction.
    TaskResetProgram reset;
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
    // Motion imagination is compiled directly into the native teacher
    // program. The runtime never reloads or interprets the artifact.
    std::optional<InteractionPack> interactions;
    std::string interactionClip;
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
    // These are execution semantics, not call-site preferences. Keeping them
    // in the compiled profile prevents a constructor from silently changing
    // the solver or reset distribution after the run was fingerprinted.
    bool streamedArticulatedContactResponses = true;
    std::uint32_t minimumDifficultyBand = 0u;
    std::uint32_t maximumDifficultyBand = MR_INVALID_INDEX;
};

struct RunManifest {
    std::string id;
    RobotPack robot;
    ScenePack scene;
    SensorPack sensors;
    TaskPack task;
    RealityPack reality;
    TeacherPack teacher;
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
    [[nodiscard]] std::uint64_t realityFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t teacherFingerprint() const noexcept;
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
    [[nodiscard]] const VisualSensorProgram*
    visualSensorProgram() const noexcept;
    [[nodiscard]] const MetalWorldMulticopterProgram*
    multicopterProgram() const noexcept;

private:
    std::uint64_t fingerprint_ = 0u;
    std::uint64_t robotFingerprint_ = 0u;
    std::uint64_t sensorFingerprint_ = 0u;
    std::uint64_t realityFingerprint_ = 0u;
    std::uint64_t teacherFingerprint_ = 0u;
    WorldFamily worldFamily_;
    EngineModel model_;
    std::vector<MRBodyStateGPU> defaultSceneBodies_;
    CompiledWorld world_;
    CompiledTaskProgram task_;
    CompiledPolicyProgram policy_;
    std::optional<PolicyPack> boundPolicy_;
    RunProfile profile_;
    TeacherPack teacher_;
    std::optional<VisualSensorProgram> visualSensorProgram_;
    std::optional<MetalWorldMulticopterProgram> multicopterProgram_;

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
[[nodiscard]] ScenePack makePX4X500HoverScenePack();
[[nodiscard]] TaskPack makePX4X500HoverTaskPack(
    TaskObservationProgram& observations,
    TaskResetProgram& reset
);
[[nodiscard]] const char* runCompileStatusName(
    RunCompileStatus status
) noexcept;

} // namespace metalrobo
