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

inline constexpr std::uint32_t kVisualAssetPackVersion = 2u;
inline constexpr std::uint32_t kVisualEnvironmentPackVersion = 2u;
inline constexpr std::uint32_t kVisualSceneManifestV3Version = 3u;
inline constexpr std::uint32_t kVisualMotionSampleBatchVersion = 1u;

enum class VisualTexturePixelFormatV2 : std::uint32_t {
    rgba8Unorm = 1u,
    rgba8UnormSrgb = 2u,
    rg11b10Float = 3u,
    rg16Float = 4u,
    rgba16Float = 5u,
};

enum class VisualTextureDimensionV2 : std::uint32_t {
    texture2D = 1u,
    cube = 2u,
};

enum class VisualAssetSectionKindV2 : std::uint32_t {
    metadata = 1u,
    vertices = 2u,
    indices = 3u,
    primitives = 4u,
    instances = 5u,
    materials = 6u,
    textureBindings = 7u,
    textureDescriptors = 8u,
    texturePayload = 9u,
    symbolicBindings = 10u,
};

struct VisualTextureSubresourceV2 {
    std::uint32_t mipLevel = 0u;
    std::uint32_t arraySlice = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint64_t dataOffset = 0u;
    std::uint64_t dataSize = 0u;
    std::uint32_t bytesPerRow = 0u;
    std::uint32_t bytesPerImage = 0u;
};

struct VisualTextureImageV2 {
    std::string id;
    std::string contentHash;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t mipCount = 0u;
    std::uint32_t arrayLength = 1u;
    VisualTexturePixelFormatV2 pixelFormat =
        VisualTexturePixelFormatV2::rgba8Unorm;
    VisualTextureDimensionV2 dimension =
        VisualTextureDimensionV2::texture2D;
    std::uint32_t flags = 0u;
    std::vector<VisualTextureSubresourceV2> subresources;
    // Cook-only storage. Each subresource points into this arena until the
    // sectioned pack writer streams it to its final file offset.
    std::vector<std::uint8_t> data;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualSymbolicBindingV2 {
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

struct VisualPackSectionV2 {
    VisualAssetSectionKindV2 kind =
        VisualAssetSectionKindV2::metadata;
    std::uint32_t index = 0u;
    std::uint64_t fileOffset = 0u;
    std::uint64_t byteCount = 0u;
    std::uint64_t elementCount = 0u;
    std::uint32_t elementStride = 0u;
    std::string contentHash;
};

struct VisualAssetPackV2 {
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
    std::vector<MRVisualTextureBindingGPUV2> textureBindings;
    std::vector<VisualTextureImageV2> textures;
    std::vector<VisualSymbolicBindingV2> symbolicBindings;
    std::vector<VisualPackSectionV2> sections;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualEnvironmentPackV2 {
    std::uint32_t schemaVersion =
        kVisualEnvironmentPackVersion;
    std::string id;
    std::string sourceUri;
    std::string sourceContentHash;
    std::string contentHash;
    std::string sourceColorSpace = "linear-rec709";
    std::string preprocessingProvenance;
    std::uint32_t specularFaceSize = 0u;
    std::uint32_t diffuseFaceSize = 64u;
    std::uint32_t brdfLutSize = 256u;
    VisualTextureImageV2 diffuseIrradiance;
    VisualTextureImageV2 prefilteredSpecular;
    VisualTextureImageV2 brdfLut;
    std::vector<VisualPackSectionV2> sections;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualEnvironmentReferenceV2 {
    std::string id = "neutral_studio";
    std::filesystem::path packPath;
    std::string contentHash = "builtin:neutral-studio-v2";
    float intensity = 1.0f;
    float rotationRadians = 0.0f;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualLightRigV1 {
    std::string id = "studio_key";
    std::string contentHash = "builtin:studio-key-v1";
    std::vector<MRVisualLightGPUV1> lights;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualAssetReferenceV3 {
    std::filesystem::path packPath;
    std::string contentHash;
    std::uint32_t assetIndex = 0u;
    std::uint32_t semanticId = 1u;
    std::uint32_t instanceId = 1u;
};

struct VisualRenderSceneV3 {
    std::string id;
    std::uint32_t assetCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::vector<MRHybridGaussianGPU> gaussians;
    std::vector<VisualAssetReferenceV3> visualPacks;
    std::vector<MRVisualSensorBindingGPU> sensorBindings;
    VisualEnvironmentReferenceV2 environment;
    VisualLightRigV1 lightRig;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct VisualSceneManifestV3 {
    std::uint32_t schemaVersion = kVisualSceneManifestV3Version;
    std::string id;
    std::string coordinateConvention = "x-forward,y-left,z-up";
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::vector<std::string> visualPackHashes;
    std::string environmentMapHash;
    std::string lightRigHash;
    std::string preprocessingProvenance;
    VisualRenderSceneV3 renderScene;

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
        "metalrobo_visual_cook/v3";
    bool generateNormals = true;
    bool generateTangents = true;
    bool generateMipmaps = true;
    bool preserveVertexColors = true;
    std::map<std::string, std::uint32_t> linkBodyIndices;
    std::map<std::string, std::uint32_t> rigidBodyIndices;
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
    VisualAssetPackV2& output,
    const VisualAssetCookOptions& options = {}
);

[[nodiscard]] VisualAssetCookDiagnostics
cookUrdfVisualDescription(
    const std::filesystem::path& urdf,
    std::vector<VisualAssetPackV2>& output,
    const VisualAssetCookOptions& options = {}
);

[[nodiscard]] bool writeVisualAssetPack(
    const VisualAssetPackV2& pack,
    const std::filesystem::path& path,
    std::string* reason = nullptr
);

[[nodiscard]] bool readVisualAssetPack(
    const std::filesystem::path& path,
    VisualAssetPackV2& output,
    std::string* reason = nullptr
);

// Reads pack metadata, section offsets, materials, instances, primitives, and
// texture bindings without materializing vertex, index, or texture payloads.
[[nodiscard]] bool readVisualAssetPackIndex(
    const std::filesystem::path& path,
    VisualAssetPackV2& output,
    std::string* reason = nullptr
);

[[nodiscard]] std::string computeVisualAssetPackContentHash(
    const VisualAssetPackV2& pack
);

[[nodiscard]] bool appendVisualAssetPackReference(
    const std::filesystem::path& packPath,
    std::uint32_t assetIndex,
    std::uint32_t semanticId,
    std::uint32_t instanceId,
    VisualRenderSceneV3& scene,
    std::string* reason = nullptr
);

[[nodiscard]] bool writeVisualEnvironmentPack(
    const VisualEnvironmentPackV2& pack,
    const std::filesystem::path& path,
    std::string* reason = nullptr
);

[[nodiscard]] bool readVisualEnvironmentPack(
    const std::filesystem::path& path,
    VisualEnvironmentPackV2& output,
    std::string* reason = nullptr
);

// Reads environment metadata and section offsets without texture payloads.
[[nodiscard]] bool readVisualEnvironmentPackIndex(
    const std::filesystem::path& path,
    VisualEnvironmentPackV2& output,
    std::string* reason = nullptr
);

[[nodiscard]] std::string computeVisualEnvironmentPackContentHash(
    const VisualEnvironmentPackV2& pack
);

struct VisualEnvironmentCookOptions {
    std::string id;
    std::string sourceColorSpace = "auto";
    std::uint32_t faceSize = 0u;
};

[[nodiscard]] VisualAssetCookDiagnostics cookVisualEnvironment(
    const std::filesystem::path& source,
    VisualEnvironmentPackV2& output,
    const VisualEnvironmentCookOptions& options = {}
);

[[nodiscard]] VisualEnvironmentReferenceV2
makeNeutralStudioEnvironmentV2();

[[nodiscard]] VisualLightRigV1 makeStudioKeyLightRigV1();

[[nodiscard]] VisualLightRigV1 makeIndoorAreaLightRigV1();

[[nodiscard]] std::uint64_t computeVisualRenderSceneV3Fingerprint(
    const VisualRenderSceneV3& scene
);

[[nodiscard]] std::uint64_t computeVisualSceneManifestV3Fingerprint(
    const VisualSceneManifestV3& manifest
);

[[nodiscard]] bool writeVisualSceneManifestV3(
    const VisualSceneManifestV3& manifest,
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
