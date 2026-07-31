#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/Tactile.hpp"
#include "metalrobo/r2s2r_types.h"
#include "metalrobo/visual_platform_types.h"
#include "metalrobo/world_compiler_types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct WorldPose {
    mr_float4 position{0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
};

enum class EpisodeArtifactKind : std::uint32_t {
    capture = 0u,
    cameraCalibration = 1u,
    entityGraph = 2u,
    segmentation = 3u,
    geometry = 4u,
    appearance = 5u,
    poseTrack = 6u,
    physicalPrior = 7u,
    robotTrajectory = 8u,
    interactionTrace = 9u,
    replay = 10u,
};

enum class EpisodeArtifactProducer : std::uint32_t {
    measured = 0u,
    deterministicTool = 1u,
    agentDecision = 2u,
    authored = 3u,
};

struct EpisodeArtifact {
    std::string id;
    EpisodeArtifactKind kind = EpisodeArtifactKind::capture;
    EpisodeArtifactProducer producer = EpisodeArtifactProducer::measured;
    std::string assetId;
    std::string uri;
    std::string contentHash;
    double startTimeSeconds = 0.0;
    double endTimeSeconds = 0.0;
};

struct SemanticAnchor {
    std::string id;
    WorldPose localPose;
    float radius = 0.0f;
    std::uint32_t flags = 0u;
};

struct WorldAsset {
    std::string id;
    std::string semanticClass;
    MRWorldAssetRole role = MR_WORLD_ASSET_BACKGROUND;
    MRWorldRenderRepresentation render =
        MR_WORLD_RENDER_NONE;
    MRWorldCollisionRepresentation collision =
        MR_WORLD_COLLISION_NONE;
    MRWorldDynamicsRepresentation dynamics =
        MR_WORLD_DYNAMICS_STATIC;
    WorldPose initialPose;
    float uniformScale = 1.0f;
    float massScale = 1.0f;
    float frictionScale = 1.0f;
    float restitutionScale = 1.0f;
    float dampingScale = 1.0f;
    float controllerGainScale = 1.0f;
    float controllerDampingScale = 1.0f;
    float controllerLatencySeconds = 0.0f;
    float payloadScale = 1.0f;
    std::uint32_t renderAlternative = 0u;
    std::uint32_t collisionAlternative = 0u;
    std::uint32_t articulationIndex = MR_INVALID_INDEX;
    std::string topologyCohort = "default";
    std::vector<std::uint32_t> bodyIndices;
    std::vector<std::uint32_t> shapeIndices;
    std::vector<std::uint32_t> materialIndices;
    std::vector<SemanticAnchor> anchors;
};

struct SensorSpec {
    std::string id;
    std::string parentAssetId;
    MRWorldSensorParentKind parentKind =
        MR_WORLD_SENSOR_PARENT_ASSET;
    // Global EngineModel body index for rigid-body or articulated-link
    // parents. Asset/world parents require MR_INVALID_INDEX.
    std::uint32_t parentBodyIndex = MR_INVALID_INDEX;
    MRWorldSensorKind kind = MR_WORLD_SENSOR_STATE;
    // Asset/body-frame authored transform. For body parents the sensor
    // compiler converts this once to the COM-centred runtime origin used by
    // articulated and rigid-body state. Tactile geometry retains the cooked
    // tactile transform as its canonical spatial authority.
    WorldPose localPose;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    mr_float4 intrinsics{0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 distortion{0.0f, 0.0f, 0.0f, 0.0f};
    float focalScale = 1.0f;
    float colorNoiseSigma = 0.0f;
    float depthNoiseSigma = 0.0f;
    float depthDropout = 0.0f;
    float latencySeconds = 0.0f;
    float nominalRateHz = 15.0f;
    MRWorldSensorSchedulePhase schedulePhase =
        MR_WORLD_SENSOR_PHASE_POST_PHYSICS;
    std::uint32_t historyLength = 1u;
    std::uint32_t consumerFlags =
        MR_WORLD_SENSOR_CONSUMER_ACTOR |
        MR_WORLD_SENSOR_CONSUMER_CRITIC |
        MR_WORLD_SENSOR_CONSUMER_TRUTH |
        MR_WORLD_SENSOR_CONSUMER_RECORDER;
    // Modality-independent corruption. Image-specific color/depth
    // corruption remains below and is applied by the presentation pass.
    float valueNoiseSigma = 0.0f;
    float biasNoiseSigma = 0.0f;
    float dropoutProbability = 0.0f;
    float exposureSeconds = 1.0f / 120.0f;
    float shutterReadoutSeconds = 0.0f;
    MRVisualShutterModel shutterModel =
        MR_VISUAL_SHUTTER_GLOBAL;
    MRVisualShutterDirection shutterDirection =
        MR_VISUAL_SHUTTER_TOP_TO_BOTTOM;
    float frameJitterSeconds = 0.0f;
    float minimumDepthMeters = 0.05f;
    float maximumDepthMeters = 10.0f;
    float depthQuantumMeters = 0.001f;
    float motionBlurScale = 0.0f;
};

struct TaskSpec {
    std::string id;
    std::string robotAssetId;
    std::string manipulatedAssetId;
    std::string targetAssetId;
    std::string targetAnchorId;
    double controlPeriodSeconds = 0.05;
    double horizonSeconds = 15.0;
};

struct EpisodeTwin {
    std::string id;
    std::string coordinateConvention = "x-forward,y-left,z-up";
    std::vector<WorldAsset> assets;
    std::vector<SensorSpec> sensors;
    // Tactile atlas authoring is separate from camera-like SensorSpec data.
    // compileEpisodeTwin cooks it into one immutable, GPU-ready arena and
    // creates/verifies matching generic sensor metadata.
    std::vector<TactileSensorSpec> tactileSensors;
    std::vector<EpisodeArtifact> artifacts;
    TaskSpec task;
};

struct AppearanceSpec {
    std::string id = "default";
    float exposureStops = 0.0f;
    float whiteBalanceScale = 1.0f;
    float saturation = 1.0f;
    float lightIntensity = 1.0f;
    float hueRadians = 0.0f;
    float contrast = 1.0f;
    float roughnessScale = 1.0f;
    float metallicScale = 1.0f;
    std::uint32_t alternative = 0u;
    std::uint32_t environmentMap = 0u;
};

struct WorldTemplate {
    std::string id;
    std::uint64_t fingerprint = 0u;
    std::uint32_t capabilities = 0u;
    EngineModel engineModel;
    std::vector<WorldAsset> assets;
    std::vector<SensorSpec> sensors;
    CookedTactileSystem tactileSystem;
    std::vector<AppearanceSpec> appearances;
    std::vector<MRWorldAssetBindingGPU> assetBindings;
    std::vector<std::uint32_t> bindingIndices;
    std::vector<EpisodeArtifact> artifacts;
    TaskSpec task;
    std::vector<std::string> topologyCohorts;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
    [[nodiscard]] std::uint32_t assetIndex(const std::string& id) const;
    [[nodiscard]] std::uint32_t sensorIndex(const std::string& id) const;
};

struct VariationParameter {
    std::string id;
    MRWorldVariationAxis axis = MR_WORLD_VARIATION_APPEARANCE;
    MRWorldDistributionKind distribution =
        MR_WORLD_DISTRIBUTION_CONSTANT;
    MRWorldVariationTarget target = MR_WORLD_TARGET_APPEARANCE_EXPOSURE;
    std::string targetId;
    // constant: x=value
    // uniform/log-uniform: x=lower, y=upper
    // normal-clamped: x=mean, y=sigma, z=lower, w=upper
    mr_float4 parameters{0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<std::uint32_t> categoricalValues;
    std::uint32_t stream = 0u;
    std::uint64_t salt = 0u;
};

struct WorldProgram {
    std::string id;
    std::uint32_t instanceFlags =
        MR_WORLD_INSTANCE_FAMILY_RENDER |
        MR_WORLD_INSTANCE_PRIVILEGED_STATE;
    std::vector<VariationParameter> variations;
};

struct CompiledWorldProgram {
    std::string id;
    std::uint64_t fingerprint = 0u;
    std::uint32_t instanceFlags = 0u;
    // Authored semantic names stay parallel to the pointer-free descriptors.
    // They are persisted in MRWorldPack so hardware manifests and evaluation
    // records can address scenario dimensions without depending on ordinals.
    std::vector<std::string> variationIds;
    std::vector<std::string> variationTargetIds;
    std::vector<MRWorldVariationGPU> variations;
    std::vector<std::uint32_t> categoricalValues;
};

struct WorldInstanceBatch {
    std::uint64_t familyFingerprint = 0u;
    std::vector<MRWorldInstanceHeaderGPU> instances;
    std::vector<MRWorldScenarioHeaderGPU> scenarioHeaders;
    std::vector<MRWorldScenarioValueGPU> scenarioValues;
    std::vector<MRWorldAssetInstanceGPU> assets;
    std::vector<MRWorldSensorInstanceGPU> sensors;
    std::vector<MRWorldAppearanceInstanceGPU> appearances;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct WorldFamily {
    WorldTemplate worldTemplate;
    CompiledWorldProgram program;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] WorldInstanceBatch sample(
        std::uint32_t instanceCount,
        std::uint64_t seed
    ) const;
};

enum class WorldCompileStatus : std::uint32_t {
    success = 0u,
    invalidEpisodeTwin = 1u,
    invalidEngineModel = 2u,
    invalidWorldTemplate = 3u,
    invalidWorldProgram = 4u,
    capacityOverflow = 5u,
};

struct WorldCompileResult {
    WorldCompileStatus status = WorldCompileStatus::success;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == WorldCompileStatus::success;
    }
};

[[nodiscard]] WorldCompileResult compileEpisodeTwin(
    const EpisodeTwin& episode,
    const EngineModel& engineModel,
    WorldTemplate& output
);

[[nodiscard]] WorldCompileResult compileWorldFamily(
    const WorldTemplate& worldTemplate,
    const WorldProgram& program,
    WorldFamily& output
);

[[nodiscard]] std::uint64_t worldCompilerFingerprint(
    const EpisodeTwin& episode,
    const EngineModel& engineModel
);

} // namespace metalrobo
