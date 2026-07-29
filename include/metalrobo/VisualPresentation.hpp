#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/visual_platform_types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kVisualAssetPackVersion = 1u;
inline constexpr std::uint32_t kVisualSceneManifestV2Version = 2u;
inline constexpr std::uint32_t kVisualMotionSampleBatchVersion = 1u;

struct VisualTextureImageV1 {
    std::string id;
    std::string contentHash;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t flags = 0u;
    // RGBA8 mip levels are tightly packed in largest-to-smallest order.
    std::vector<std::uint32_t> mipTexelOffsets;
    std::vector<std::uint8_t> rgba8;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualSymbolicBindingV1 {
    std::string node;
    std::string link;
    std::uint32_t instanceIndex = MR_INVALID_INDEX;
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    MRVisualBindingKind binding =
        MR_VISUAL_BINDING_ASSET;

    [[nodiscard]] bool valid(
        std::uint32_t instanceCount,
        std::string* reason = nullptr
    ) const;
};

struct VisualAssetPackV1 {
    std::uint32_t schemaVersion = kVisualAssetPackVersion;
    std::string id;
    std::string sourceUri;
    std::string sourceContentHash;
    std::string contentHash;
    std::string license;
    std::string preprocessingProvenance;
    std::string coordinateConvention = "x-forward,y-left,z-up";
    std::vector<MRVisualVertexGPUV2> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MRVisualPrimitiveGPUV2> primitives;
    std::vector<MRVisualInstanceGPUV2> instances;
    std::vector<MRVisualMaterialGPUV2> materials;
    std::vector<VisualTextureImageV1> textures;
    std::vector<VisualSymbolicBindingV1> symbolicBindings;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualEnvironmentV1 {
    std::string id = "neutral_studio";
    std::string contentHash = "builtin:neutral-studio-v1";
    std::uint32_t textureIndex = MR_INVALID_INDEX;
    float intensity = 1.0f;
    float rotationRadians = 0.0f;
    // Third-order real spherical harmonics in linear RGB.
    std::array<mr_float4, 9u> diffuseSH{{
        {0.282095f, 0.282095f, 0.282095f, 0.0f},
    }};
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(
        std::size_t textureCount,
        std::string* reason = nullptr
    ) const;
};

struct VisualLightRigV1 {
    std::string id = "studio_key";
    std::string contentHash = "builtin:studio-key-v1";
    std::vector<MRVisualLightGPUV1> lights;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualRenderSceneV2 {
    std::string id;
    std::uint32_t assetCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::vector<MRHybridGaussianGPU> gaussians;
    std::vector<MRVisualVertexGPUV2> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MRVisualPrimitiveGPUV2> primitives;
    std::vector<MRVisualInstanceGPUV2> instances;
    std::vector<MRVisualMaterialGPUV2> materials;
    std::vector<VisualTextureImageV1> textures;
    std::vector<MRVisualSensorBindingGPU> sensorBindings;
    VisualEnvironmentV1 environment;
    VisualLightRigV1 lightRig;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualSceneManifestV2 {
    std::uint32_t schemaVersion = kVisualSceneManifestV2Version;
    std::string id;
    std::string coordinateConvention = "x-forward,y-left,z-up";
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::vector<std::string> visualPackHashes;
    std::string environmentMapHash;
    std::string lightRigHash;
    std::string preprocessingProvenance;
    VisualRenderSceneV2 renderScene;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualRendererProfileV1 {
    std::string id = "sensor_fast";
    MRVisualRendererProfileKind kind =
        MR_VISUAL_RENDERER_SENSOR_FAST;
    std::uint32_t temporalSamples = 2u;
    std::uint32_t rollingShutterBands = 16u;
    std::uint32_t shadowMapResolution = 128u;
    std::uint32_t areaLightSamples = 4u;
    bool rayQueryVisibility = false;
    bool retainObservation = false;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;

    [[nodiscard]] static VisualRendererProfileV1 sensorFast();
    [[nodiscard]] static VisualRendererProfileV1 sensorReference();
};

struct VisualSensorProfileV2 {
    std::string id;
    double nominalRateHz = 15.0;
    double exposureSeconds = 1.0 / 120.0;
    double shutterReadoutSeconds = 0.0;
    double frameJitterSeconds = 0.0;
    double minimumDepthMeters = 0.05;
    double maximumDepthMeters = 10.0;
    double depthQuantumMeters = 0.001;
    double latencySeconds = 0.0;
    MRVisualShutterModel shutterModel =
        MR_VISUAL_SHUTTER_GLOBAL;
    MRVisualShutterDirection shutterDirection =
        MR_VISUAL_SHUTTER_TOP_TO_BOTTOM;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualMotionSampleBatchV1 {
    std::uint32_t schemaVersion =
        kVisualMotionSampleBatchVersion;
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t sampleCount = 0u;
    double exposureOpenSeconds = 0.0;
    double exposureCloseSeconds = 0.0;
    std::vector<double> timestampsSeconds;
    // sample-major, then environment-major, then global body index.
    std::vector<MRBodyStateGPU> bodyStates;
    std::uint64_t scenarioIdentity = 0u;
    std::uint64_t sensorIdentity = 0u;
    std::uint64_t frameIndex = 0u;
    std::uint32_t sensorSequence = 0u;
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
    [[nodiscard]] std::span<const MRBodyStateGPU>
    sample(std::uint32_t sampleIndex) const noexcept;
};

enum class VisualAssetCookStatus : std::uint32_t {
    success = 0u,
    ioFailure,
    unsupportedFormat,
    malformedAsset,
    unsupportedFeature,
    invalidGeometry,
    invalidMaterial,
    invalidTexture,
    invalidBinding,
    capacityOverflow,
    writeFailure,
    internalFailure,
};

struct VisualAssetCookOptions {
    std::string id;
    std::string license = "NOASSERTION";
    std::string preprocessingProvenance =
        "metalrobo_visual_cook/v1";
    bool generateNormals = true;
    bool generateTangents = true;
    bool generateMipmaps = true;
    bool preserveVertexColors = true;
    std::map<std::string, std::uint32_t> linkBodyIndices;
};

struct VisualAssetCookDiagnostics {
    VisualAssetCookStatus status = VisualAssetCookStatus::success;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t primitiveCount = 0u;
    std::uint32_t instanceCount = 0u;
    std::uint32_t materialCount = 0u;
    std::uint32_t textureCount = 0u;
    std::string sourceHash;
    std::string packHash;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == VisualAssetCookStatus::success;
    }
};

[[nodiscard]] VisualAssetCookDiagnostics cookVisualAsset(
    const std::filesystem::path& source,
    VisualAssetPackV1& output,
    const VisualAssetCookOptions& options = {}
);

[[nodiscard]] VisualAssetCookDiagnostics
cookUrdfVisualDescription(
    const std::filesystem::path& urdf,
    std::vector<VisualAssetPackV1>& output,
    const VisualAssetCookOptions& options = {}
);

[[nodiscard]] bool writeVisualAssetPack(
    const VisualAssetPackV1& pack,
    const std::filesystem::path& path,
    std::string* reason = nullptr
);

[[nodiscard]] bool readVisualAssetPack(
    const std::filesystem::path& path,
    VisualAssetPackV1& output,
    std::string* reason = nullptr
);

[[nodiscard]] std::string computeVisualAssetPackContentHash(
    const VisualAssetPackV1& pack
);

[[nodiscard]] bool appendVisualAssetPack(
    VisualAssetPackV1&& pack,
    std::uint32_t assetIndex,
    std::uint32_t semanticId,
    std::uint32_t instanceId,
    VisualRenderSceneV2& scene,
    std::string* reason = nullptr
);

[[nodiscard]] VisualEnvironmentV1
makeNeutralStudioEnvironmentV1();

[[nodiscard]] VisualLightRigV1 makeStudioKeyLightRigV1();

[[nodiscard]] VisualLightRigV1 makeIndoorAreaLightRigV1();

[[nodiscard]] std::uint64_t computeVisualRenderSceneV2Fingerprint(
    const VisualRenderSceneV2& scene
);

[[nodiscard]] std::uint64_t computeVisualSceneManifestV2Fingerprint(
    const VisualSceneManifestV2& manifest
);

[[nodiscard]] bool writeVisualSceneManifestV2(
    const VisualSceneManifestV2& manifest,
    const std::filesystem::path& path,
    std::string* reason = nullptr
);

[[nodiscard]] std::uint64_t computeVisualRendererProfileFingerprint(
    const VisualRendererProfileV1& profile
);

[[nodiscard]] const char* visualAssetCookStatusName(
    VisualAssetCookStatus status
) noexcept;

} // namespace metalrobo
