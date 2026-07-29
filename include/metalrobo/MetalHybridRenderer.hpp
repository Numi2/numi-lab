#pragma once

#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/hybrid_renderer_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {
struct MetalHybridRendererState;
} // namespace detail

struct HybridGaussianScene {
    std::string id;
    std::uint32_t assetCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::vector<MRHybridGaussianGPU> gaussians;
    std::vector<MRVisualMeshVertexGPU> meshVertices;
    std::vector<MRVisualMeshTriangleGPU> meshTriangles;
    std::vector<MRVisualMaterialGPU> materials;
    std::vector<MRVisualSensorBindingGPU> sensorBindings;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

struct MetalHybridRendererConfig {
    std::string metallibPath;
    std::uint32_t width = 160u;
    std::uint32_t height = 120u;
    std::uint32_t maximumGaussiansPerTile = MR_HYBRID_MAX_GAUSSIANS_PER_TILE;
    mr_float4 clearColorAndDepth{0.0f, 0.0f, 0.0f, 1.0e30f};
};

enum class MetalHybridRendererStatus : std::uint32_t {
    success = 0u,
    invalidScene,
    invalidConfiguration,
    capacityOverflow,
    missingLiveState,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    incompatibleWorldFamily,
    notCompiled,
    internalFailure,
};

struct MetalHybridRendererLayout {
    std::uint32_t capacity = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t tileCountX = 0u;
    std::uint32_t tileCountY = 0u;
    std::uint32_t gaussianCount = 0u;
    std::uint32_t meshVertexCount = 0u;
    std::uint32_t meshTriangleCount = 0u;
    std::uint32_t materialCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t sensorBindingCount = 0u;
    std::uint32_t maximumGaussiansPerTile = 0u;
    std::size_t retainedPrivateBytes = 0u;
};

struct MetalHybridRendererDiagnostics {
    MetalHybridRendererStatus status = MetalHybridRendererStatus::success;
    MetalHybridRendererLayout layout;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalHybridRendererStatus::success;
    }
};

struct HybridObservationBatch {
    std::uint32_t environmentCount = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::vector<mr_float4> rgb;
    std::vector<float> depth;
    std::vector<std::uint32_t> segmentation;
    std::vector<mr_uint4> identities;
    std::vector<mr_float4> normals;
    std::vector<mr_float4> motion;
    std::vector<std::uint32_t> validity;
    MRVisualFrameMetadataGPU metadata{};
};

struct HybridLiveStateBatch {
    // Environment-major global EngineModel body state. Articulated and scene
    // bodies share the same global body index used by MRShapeGPU.
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::span<const MRBodyStateGPU> currentBodies{};
    std::span<const MRBodyStateGPU> previousBodies{};
    std::uint64_t frameIndex = 0u;
    std::uint32_t sensorSequence = 0u;
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;
    double captureTimestampSeconds = 0.0;
    double frameAgeSeconds = 0.0;
};

struct HybridDeviceStateBatch {
    // Borrowed id<MTLBuffer> values. Both buffers contain environment-major
    // MRBodyStateGPU records. previousBodyStates may be null on the first
    // frame, in which case motion is zero.
    void* currentBodyStates = nullptr;
    void* previousBodyStates = nullptr;
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint64_t frameIndex = 0u;
    std::uint32_t sensorSequence = 0u;
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;
    double captureTimestampSeconds = 0.0;
    double frameAgeSeconds = 0.0;
};

enum class MetalHybridRendererBuffer : std::uint32_t {
    rgb = 0u,
    depth = 1u,
    segmentation = 2u,
    projectedGaussians = 3u,
    tileOverflowCounts = 4u,
    identities = 5u,
    normals = 6u,
    motion = 7u,
    validity = 8u,
    meshWinners = 9u,
};

// First native hybrid observation stage. Static, rigid-object, and other
// asset-local Gaussian fields consume the sampled WorldFamily buffers
// directly. The output buffers remain private and can be imported by an MLX
// active-encoder primitive without a host readback.
class MetalHybridRenderer {
public:
    explicit MetalHybridRenderer(MetalHybridRendererConfig config = {});
    ~MetalHybridRenderer();

    MetalHybridRenderer(MetalHybridRenderer&& other) noexcept;
    MetalHybridRenderer& operator=(MetalHybridRenderer&& other) noexcept;

    MetalHybridRenderer(const MetalHybridRenderer&) = delete;
    MetalHybridRenderer& operator=(const MetalHybridRenderer&) = delete;

    [[nodiscard]] MetalHybridRendererDiagnostics
    compile(const HybridGaussianScene& scene, std::uint32_t capacity);

    [[nodiscard]] MetalHybridRendererDiagnostics
    render(const MetalWorldFamilyContext& worlds,
           std::uint32_t environmentCount, std::uint32_t cameraIndex = 0u);

    [[nodiscard]] MetalHybridRendererDiagnostics renderLive(
        const MetalWorldFamilyContext& worlds,
        const HybridLiveStateBatch& liveState,
        std::uint32_t cameraIndex = 0u
    );

    // Encodes into a caller-owned id<MTLComputeCommandEncoder> without
    // committing, waiting, or reading back. This is the MLX/Core ML/native
    // graph composition boundary. Output buffers remain owned by the
    // renderer and are exposed through nativeBuffer().
    [[nodiscard]] MetalHybridRendererDiagnostics encode(
        const MetalWorldFamilyContext& worlds,
        const HybridDeviceStateBatch& liveState,
        std::uint32_t cameraIndex,
        void* metalComputeCommandEncoder
    );

    [[nodiscard]] MetalHybridRendererDiagnostics
    readback(HybridObservationBatch& output);

    [[nodiscard]] MetalHybridRendererLayout layout() const noexcept;

    [[nodiscard]] void*
    nativeBuffer(MetalHybridRendererBuffer buffer) const noexcept;

private:
    std::shared_ptr<detail::MetalHybridRendererState> state_;
};

[[nodiscard]] const char*
metalHybridRendererStatusName(MetalHybridRendererStatus status) noexcept;

} // namespace metalrobo
