#pragma once

#include "metalrobo/WorldPack.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

enum class CaptureAdapterKind : std::uint32_t {
    arkit = 0u,
    rgbdRobotTelemetry,
    rosBag,
    libfrankaLog,
    cadUrdf,
    videoFallback,
};

enum class CaptureStreamKind : std::uint32_t {
    rgb = 0u,
    depth,
    rgbd,
    video,
    cameraCalibration,
    cameraPoses,
    robotTelemetry,
    robotCommands,
    gripperState,
    forceTorque,
    cad,
    urdf,
};

enum class EpisodeTwinStage : std::uint32_t {
    ingest = 0u,
    selectFrames,
    discoverEntities,
    segment,
    reconstructGeometry,
    trackPoses,
    inferPhysics,
    assembleReplay,
    alignReplay,
    publish,
};

enum class CaptureProfile : std::uint32_t {
    // Compatibility profile for deterministic or manually assembled worlds.
    authoredSeed = 0u,
    // First fail-closed physical path: synchronized fixed RGB-D plus Franka
    // state/commands and deterministic object reconstruction products.
    frankaFixedRGBD = 1u,
};

enum class EpisodeTwinProductKind : std::uint32_t {
    artifactOnly = 0u,
    synchronizedSensorStream,
    sensorCalibration,
    robotStateTrace,
    robotCommandTrace,
    semanticGraph,
    segmentation,
    objectPoseTrack,
    renderGeometry,
    collisionGeometry,
    physicalPrior,
    taskEvents,
};

struct CaptureCalibration {
    bool hasResolution = false;
    bool hasIntrinsics = false;
    bool hasDistortion = false;
    bool hasPose = false;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    mr_float4 intrinsics{};
    mr_float4 distortion{};
    WorldPose worldFromSensor;

    [[nodiscard]] bool present() const noexcept {
        return hasResolution || hasIntrinsics || hasDistortion || hasPose;
    }
};

// Typed, deterministic assembly instructions. The artifact proves where the
// data came from; this payload is the bounded part allowed to alter the seed
// world. Geometry providers select a representation and target, never inject
// device pointers or unchecked collision records.
struct EpisodeTwinProductPayload {
    EpisodeTwinProductKind kind = EpisodeTwinProductKind::artifactOnly;
    std::string targetId;
    CaptureCalibration calibration;
    bool hasWorldPose = false;
    WorldPose worldPose;
    bool hasPhysicalPrior = false;
    float massScale = 1.0f;
    float frictionScale = 1.0f;
    float restitutionScale = 1.0f;
    float dampingScale = 1.0f;
    MRWorldRenderRepresentation renderRepresentation = MR_WORLD_RENDER_NONE;
    MRWorldCollisionRepresentation collisionRepresentation =
        MR_WORLD_COLLISION_NONE;
    bool hasCollisionBox = false;
    mr_float4 collisionBoxHalfExtents{};
};

struct CaptureStream {
    std::string id;
    CaptureStreamKind kind = CaptureStreamKind::video;
    std::string assetId;
    std::string sensorId;
    std::filesystem::path source;
    std::string expectedContentHash;
    std::string timestampDomain = "monotonic";
    double startTimeSeconds = 0.0;
    double endTimeSeconds = 0.0;
    double nominalRateHz = 0.0;
    CaptureCalibration calibration;
};

// A deterministic or measured product already supplied by an adapter, or
// produced by an EpisodeTwinStageProvider. `source` is imported into the
// content-addressed artifact store before becoming an EpisodeArtifact.
struct CaptureProduct {
    std::string id;
    EpisodeTwinStage stage = EpisodeTwinStage::ingest;
    EpisodeArtifactKind kind = EpisodeArtifactKind::capture;
    EpisodeArtifactProducer producer =
        EpisodeArtifactProducer::deterministicTool;
    std::string assetId;
    std::filesystem::path source;
    std::string expectedContentHash;
    double startTimeSeconds = 0.0;
    double endTimeSeconds = 0.0;
    EpisodeTwinProductPayload payload;
};

struct EpisodeTwinProduct {
    EpisodeArtifact artifact;
    EpisodeTwinProductPayload payload;
};

struct CaptureManifest {
    std::uint32_t schemaVersion = 1u;
    std::string id;
    CaptureAdapterKind adapter = CaptureAdapterKind::rgbdRobotTelemetry;
    CaptureProfile profile = CaptureProfile::authoredSeed;
    std::string coordinateConvention = "x-forward,y-left,z-up";
    std::string engineModelId;
    std::string worldProgramId;
    std::filesystem::path sourceRoot;
    std::vector<CaptureStream> streams;
    std::vector<CaptureProduct> products;
    // Semantic assembly supplied by a bounded agent, authoring tool, or
    // importer. Geometry and physics still come from deterministic products.
    EpisodeTwin seedEpisode;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct EpisodeArtifactImport {
    EpisodeArtifact artifact;
    EpisodeTwinProduct product;
    std::filesystem::path storedPath;
    std::uint64_t byteCount = 0u;
};

class EpisodeArtifactStore {
public:
    explicit EpisodeArtifactStore(std::filesystem::path root);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] bool prepare(std::string* reason = nullptr);
    [[nodiscard]] bool importProduct(const CaptureProduct& product,
                                     EpisodeArtifactImport& output,
                                     std::string* reason = nullptr);

private:
    std::filesystem::path root_;
};

struct EpisodeTwinStageRequest {
    EpisodeTwinStage stage = EpisodeTwinStage::ingest;
    std::string stageKey;
    const CaptureManifest* manifest = nullptr;
    std::span<const EpisodeArtifact> inputs;
};

class EpisodeTwinStageProvider {
public:
    virtual ~EpisodeTwinStageProvider() = default;
    [[nodiscard]] virtual std::string id() const = 0;
    [[nodiscard]] virtual std::string version() const = 0;
    [[nodiscard]] virtual bool
    supports(EpisodeTwinStage stage, const CaptureManifest& manifest) const = 0;
    // Providers make bounded semantic choices or invoke deterministic tools,
    // then return files for the compiler to import. They never inject device
    // pointers, collision geometry, or unchecked physics records.
    virtual void execute(const EpisodeTwinStageRequest& request,
                         std::vector<CaptureProduct>& outputs) = 0;
};

struct EpisodeTwinStageReceipt {
    EpisodeTwinStage stage = EpisodeTwinStage::ingest;
    std::string stageKey;
    bool cacheHit = false;
    std::vector<EpisodeArtifact> artifacts;
    std::vector<EpisodeTwinProduct> products;
};

enum class EpisodeTwinCompilerStatus : std::uint32_t {
    success = 0u,
    invalidManifest,
    invalidConfiguration,
    artifactFailure,
    providerFailure,
    assemblyFailure,
    episodeCompileFailure,
    familyCompileFailure,
    packCompileFailure,
    ioFailure,
    internalFailure,
};

struct EpisodeTwinCompilerResult {
    EpisodeTwinCompilerStatus status = EpisodeTwinCompilerStatus::success;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == EpisodeTwinCompilerStatus::success;
    }
};

struct EpisodeTwinCompilerConfig {
    std::filesystem::path artifactStore;
    bool resume = true;
    bool requireExpectedHashes = false;
};

struct CompiledEpisodeTwin {
    EpisodeTwin episode;
    std::vector<EpisodeTwinProduct> products;
    WorldTemplate worldTemplate;
    WorldFamily worldFamily;
    MRWorldPack worldPack;
    std::vector<EpisodeTwinStageReceipt> receipts;
};

class EpisodeTwinAssembler {
public:
    [[nodiscard]] static EpisodeTwinCompilerResult assemble(
        const CaptureManifest& manifest,
        std::span<const EpisodeTwinProduct> products,
        const EngineModel& seedEngineModel,
        EpisodeTwin& episodeOutput,
        EngineModel& engineOutput);
};

class EpisodeTwinCompiler {
public:
    explicit EpisodeTwinCompiler(EpisodeTwinCompilerConfig config);

    void addProvider(std::shared_ptr<EpisodeTwinStageProvider> provider);

    [[nodiscard]] EpisodeTwinCompilerResult
    compile(const CaptureManifest& manifest, const EngineModel& engineModel,
            const WorldProgram& worldProgram, CompiledEpisodeTwin& output);

private:
    EpisodeTwinCompilerConfig config_;
    std::vector<std::shared_ptr<EpisodeTwinStageProvider>> providers_;
};

// Apple-native JSON manifest loader. Relative stream/product paths resolve
// against source_root when present, otherwise the manifest directory.
[[nodiscard]] EpisodeTwinCompilerResult
loadCaptureManifestJSON(const std::filesystem::path& path,
                        CaptureManifest& output);

[[nodiscard]] const char* episodeTwinStageName(EpisodeTwinStage stage) noexcept;
[[nodiscard]] const char*
episodeTwinCompilerStatusName(EpisodeTwinCompilerStatus status) noexcept;

} // namespace metalrobo
