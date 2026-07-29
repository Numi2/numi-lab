#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/visual_platform_types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kVisualSceneManifestVersion = 1u;
inline constexpr std::uint32_t kVisualFrameBatchVersion = 1u;
inline constexpr std::uint32_t kPerceptionProviderVersion = 1u;
inline constexpr std::uint32_t kVisualEpisodeStreamVersion = 1u;

struct VisualAssetManifestV1 {
    std::string id;
    std::string semanticClass;
    MRVisualRepresentation representation =
        MR_VISUAL_REPRESENTATION_NONE;
    MRVisualBindingKind binding = MR_VISUAL_BINDING_WORLD;
    std::string sourceUri;
    std::string contentHash;
    std::string license;
    std::string preprocessingProvenance;
    std::uint32_t semanticId = 0u;
    std::uint32_t instanceId = 0u;
    std::vector<std::uint32_t> bodyIndices;
    std::vector<std::uint32_t> shapeIndices;

    [[nodiscard]] bool valid(
        std::uint32_t bodyCount,
        std::string* reason = nullptr
    ) const;
};

struct VisualSensorProfileV1 {
    std::string id;
    double nominalRateHz = 15.0;
    double exposureSeconds = 1.0 / 120.0;
    double shutterReadoutSeconds = 0.0;
    double frameJitterSeconds = 0.0;
    double minimumDepthMeters = 0.05;
    double maximumDepthMeters = 10.0;
    double depthQuantumMeters = 0.001;
    double motionBlurScale = 0.0;
    double latencySeconds = 0.0;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

[[nodiscard]] std::uint64_t computeVisualSensorProfileFingerprint(
    const VisualSensorProfileV1& profile
);

struct VisualSceneManifestV1 {
    std::uint32_t schemaVersion = kVisualSceneManifestVersion;
    std::string id;
    std::string coordinateConvention = "x-forward,y-left,z-up";
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t renderSceneFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::uint32_t bodyCount = 0u;
    std::vector<VisualAssetManifestV1> assets;
    HybridGaussianScene renderScene;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

// Compiles visual geometry from the canonical engine shapes. Procedural
// primitive shapes are triangulated deterministically; cooked convex/mesh
// geometry reuses the immutable EngineModel arenas. Every triangle is bound
// to the same global body index that owns its collision shape.
[[nodiscard]] bool compileVisualSceneManifest(
    const WorldTemplate& world,
    VisualSceneManifestV1& output,
    std::string* reason = nullptr
);

// Adds a captured Gaussian appearance layer to an existing manifest while
// retaining physics-bound meshes for interaction, metric depth, and identity.
[[nodiscard]] bool attachGaussianField(
    VisualSceneManifestV1& scene,
    const std::string& assetId,
    std::span<const MRHybridGaussianGPU> gaussians,
    std::string sourceUri,
    std::string contentHash,
    std::string license,
    std::string preprocessingProvenance,
    std::string* reason = nullptr
);

// Reference integration for the canonical Franka world. It binds the fixed
// camera to the workspace and the wrist camera to the final articulated body.
[[nodiscard]] VisualSceneManifestV1
makeFrankaPickPlaceVisualSceneManifest();

[[nodiscard]] bool writeVisualSceneManifest(
    const VisualSceneManifestV1& scene,
    const std::filesystem::path& path,
    std::string* reason = nullptr
);

// Converts packed generalized and scene-body state into one environment-major
// global-body stream usable by the visual runtime. Articulated bodies are
// evaluated by the same FP64 kinematics used by contact validation.
[[nodiscard]] bool composeVisualBodyStates(
    const EngineModel& model,
    std::uint32_t environmentCount,
    std::span<const float> q,
    std::span<const float> v,
    std::span<const MRBodyStateGPU> sceneBodies,
    std::vector<MRBodyStateGPU>& output,
    std::string* reason = nullptr
);

struct VisualDeviceBufferViewV1 {
    std::uintptr_t handle = 0u;
    std::size_t offsetBytes = 0u;
    std::size_t sizeBytes = 0u;
    MRVisualStorageKind storage = MR_VISUAL_STORAGE_HOST;
    MRVisualPixelFormat format = MR_VISUAL_FORMAT_UNKNOWN;
    std::uint32_t modality = 0u;

    [[nodiscard]] bool valid() const noexcept;
};

struct VisualCameraFrameV1 {
    std::string sensorId;
    mr_float4 intrinsics{};
    mr_float4 distortion{};
    WorldPose baseFromCamera;
    double captureTimestampSeconds = 0.0;
    double frameAgeSeconds = 0.0;
    double exposureSeconds = 0.0;
    double shutterReadoutSeconds = 0.0;
    std::uint64_t frameIndex = 0u;
    std::uint32_t sensorSequence = 0u;
    bool valid = false;
};

struct VisualFrameBatchV1 {
    std::uint32_t schemaVersion = kVisualFrameBatchVersion;
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;
    std::uint32_t environmentCount = 0u;
    std::uint32_t viewCount = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t modalities = 0u;
    std::uint64_t episodeTwinFingerprint = 0u;
    std::uint64_t scenarioFingerprint = 0u;
    std::uint64_t rendererFingerprint = 0u;
    std::uint64_t sensorProfileFingerprint = 0u;
    std::uint64_t calibrationFingerprint = 0u;
    std::vector<VisualCameraFrameV1> cameras;
    std::vector<mr_float4> rgbLinear;
    std::vector<float> depthMeters;
    std::vector<std::uint8_t> depthValidity;
    std::vector<VisualDeviceBufferViewV1> deviceBuffers;

    [[nodiscard]] std::size_t pixelCount() const noexcept;
    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualTruthBatchV1 {
    std::uint32_t schemaVersion = kVisualFrameBatchVersion;
    std::uint32_t modalities = 0u;
    MRVisualCoordinateFrame coordinateFrame =
        MR_VISUAL_FRAME_ROBOT_BASE;
    std::uint32_t environmentCount = 0u;
    std::uint32_t viewCount = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;
    std::vector<mr_float4> normals;
    std::vector<mr_float4> motion;
    std::vector<std::uint32_t> semanticIds;
    std::vector<std::uint32_t> instanceIds;
    std::vector<std::uint32_t> linkIds;
    // Visible truth-surface coverage in [0, 1].
    std::vector<float> visibility;
    // Quantized inverse visibility: 0 visible, 255 absent/fully occluded.
    std::vector<std::uint8_t> occlusion;
    std::vector<MRVisualPoseGPU> objectPoses;
    std::vector<MRVisualPoseGPU> linkPoses;
    std::vector<MRVisualKeypointGPU> keypoints;
    std::vector<MRVisualContactAnnotationGPU> contacts;
    std::vector<VisualDeviceBufferViewV1> deviceBuffers;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualSensorCaptureV1 {
    std::uint64_t frameIndex = 0u;
    std::uint32_t sensorSequence = 0u;
    double nominalTimestampSeconds = 0.0;
    double exposureOpenSeconds = 0.0;
    double exposureCloseSeconds = 0.0;
    double publishTimestampSeconds = 0.0;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

// Stateless, deterministic frame scheduling. Identical scenario, sensor, and
// frame identities produce identical jitter and exposure windows.
[[nodiscard]] VisualSensorCaptureV1 makeVisualSensorCapture(
    const VisualSensorProfileV1& profile,
    std::uint64_t scenarioIdentity,
    std::uint64_t sensorIdentity,
    std::uint64_t frameIndex,
    double episodeStartSeconds = 0.0
);

struct VisualBatchProvenanceV1 {
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;
    std::uint64_t episodeTwinFingerprint = 0u;
    std::uint64_t scenarioFingerprint = 0u;
    std::uint64_t rendererFingerprint = 0u;
    std::uint64_t sensorProfileFingerprint = 0u;
    std::uint64_t calibrationFingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualBatchAssemblyV1 {
    VisualBatchProvenanceV1 provenance;
    std::span<const std::uint32_t> cameraIndices{};
    std::span<const HybridObservationBatch> observations{};
    std::span<const MRBodyStateGPU> currentBodyStates{};
};

// Converts policy-step-aligned renderer readbacks into the exact same
// deployable/supervisory contracts accepted from physical RGB-D capture.
// Views share a global frame index but retain independent capture timestamps.
[[nodiscard]] bool assembleVisualBatches(
    const WorldTemplate& world,
    const WorldInstanceBatch& sampledWorlds,
    const VisualBatchAssemblyV1& input,
    VisualFrameBatchV1& frames,
    VisualTruthBatchV1& truth,
    std::string* reason = nullptr
);

enum class PerceptionElementType : std::uint32_t {
    float32 = 0u,
    uint32 = 1u,
    uint8 = 2u,
};

struct PerceptionTensorV1 {
    std::string id;
    std::uint32_t modality = 0u;
    MRVisualCoordinateFrame coordinateFrame = MR_VISUAL_FRAME_PIXEL;
    PerceptionElementType elementType = PerceptionElementType::float32;
    std::vector<std::uint32_t> shape;
    std::vector<float> floatValues;
    std::vector<std::uint32_t> uintValues;
    std::vector<std::uint8_t> byteValues;
    std::vector<VisualDeviceBufferViewV1> deviceBuffers;
    double timestampSeconds = 0.0;
    float confidence = 1.0f;
    bool valid = true;

    [[nodiscard]] std::size_t elementCount() const noexcept;
    [[nodiscard]] bool validContract(std::string* reason = nullptr) const;
};

struct PerceptionProviderDescriptorV1 {
    std::uint32_t schemaVersion = kPerceptionProviderVersion;
    std::string id;
    std::string contentHash;
    std::uint32_t inputModalities = 0u;
    std::uint32_t capabilities = 0u;
    std::uint32_t temporalWindow = 1u;
    bool acceptsDeviceBuffers = false;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct PerceptionResultBatchV1 {
    std::string providerId;
    std::string providerContentHash;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;
    std::vector<PerceptionTensorV1> tensors;

    [[nodiscard]] const PerceptionTensorV1* tensor(
        std::uint32_t modality
    ) const noexcept;
    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

class PerceptionProviderV1 {
public:
    virtual ~PerceptionProviderV1() = default;

    [[nodiscard]] virtual PerceptionProviderDescriptorV1
    descriptor() const = 0;
    [[nodiscard]] virtual bool infer(
        const VisualFrameBatchV1& frames,
        PerceptionResultBatchV1& output,
        std::string* reason = nullptr
    ) = 0;
};

enum class ObservationProfileV1 : std::uint32_t {
    rawRGBD = 0u,
    rgbXYZ = 1u,
    objectCentric = 2u,
    denseFeatures = 3u,
    compactLatent = 4u,
};

struct PolicyObservationRequestV1 {
    ObservationProfileV1 profile = ObservationProfileV1::rgbXYZ;
    std::uint32_t environmentCount = 0u;
    std::span<const float> proprioception{};
    std::uint32_t proprioceptionWidth = 0u;
    std::span<const float> previousActions{};
    std::uint32_t previousActionWidth = 0u;
    std::span<const float> taskCommands{};
    std::uint32_t taskCommandWidth = 0u;
    std::span<const float> privilegedState{};
    std::uint32_t privilegedStateWidth = 0u;
};

struct PolicyObservationBatchV1 {
    ObservationProfileV1 profile = ObservationProfileV1::rgbXYZ;
    std::uint32_t environmentCount = 0u;
    std::uint32_t deployableWidth = 0u;
    std::uint32_t privilegedWidth = 0u;
    std::vector<float> deployable;
    std::vector<float> privileged;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

class PolicyObservationAssemblerV1 {
public:
    [[nodiscard]] bool assemble(
        const VisualFrameBatchV1& frames,
        const PerceptionResultBatchV1* perception,
        const PolicyObservationRequestV1& request,
        PolicyObservationBatchV1& output,
        std::string* reason = nullptr
    ) const;
};

struct VisualEpisodeStepV1 {
    std::uint64_t frameIndex = 0u;
    std::uint64_t scenarioKey = 0u;
    double timestampSeconds = 0.0;
    double reward = 0.0;
    std::vector<float> proprioception;
    std::vector<float> action;
    std::vector<float> taskCommand;
    std::vector<float> privilegedState;
    std::string frameContentHash;
    std::string truthContentHash;
    std::uint32_t eventFlags = 0u;
};

struct VisualEpisodeStreamV1 {
    std::uint32_t schemaVersion = kVisualEpisodeStreamVersion;
    std::string id;
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;
    std::uint64_t episodeTwinFingerprint = 0u;
    std::uint64_t worldFamilyFingerprint = 0u;
    std::uint64_t scenarioFingerprint = 0u;
    std::uint64_t rendererFingerprint = 0u;
    std::uint64_t visualSceneFingerprint = 0u;
    std::uint64_t sensorProfileFingerprint = 0u;
    std::uint64_t calibrationFingerprint = 0u;
    std::uint64_t physicsFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::vector<VisualEpisodeStepV1> steps;

    [[nodiscard]] bool append(
        VisualEpisodeStepV1 step,
        std::string* reason = nullptr
    );
    [[nodiscard]] bool finalize(std::string* reason = nullptr);
    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

[[nodiscard]] bool writeVisualEpisodeManifest(
    const VisualEpisodeStreamV1& stream,
    const std::filesystem::path& path,
    std::string* reason = nullptr
);

} // namespace metalrobo
