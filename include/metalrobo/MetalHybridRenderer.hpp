#pragma once

#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/VisualPresentation.hpp"
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

struct MetalHybridRendererConfig {
    std::string metallibPath;
    std::uint32_t width = 160u;
    std::uint32_t height = 120u;
    std::uint32_t maximumGaussiansPerTile = MR_HYBRID_MAX_GAUSSIANS_PER_TILE;
    std::uint32_t maximumMeshTrianglesPerTile =
        MR_HYBRID_MAX_MESH_TRIANGLES_PER_TILE;
    // Shadow layers are transient and reused in environment batches. This
    // controls encoding granularity, not total world capacity.
    std::uint32_t shadowLayerBatchSize = 32u;
    // Reference frames recycle motion/TLAS workspaces. Bounding the pool
    // prevents asynchronous callers from multiplying large ray-build
    // allocations while still allowing a normal triple-buffered producer.
    std::uint32_t maximumReferenceFramesInFlight = 3u;
    mr_float4 clearColorAndDepth{0.0f, 0.0f, 0.0f, 1.0e30f};
    // MLX graph execution supplies all observation planes. Disabling retained
    // planes avoids keeping a second capacity-sized copy of every image in
    // unified memory. Standalone render/readback is unavailable in that mode.
    bool retainObservationBuffers = true;
    // Hard bounds on unified-memory retention. Compilation rejects an
    // oversized profile before asking Metal for any large allocation.
    std::size_t maximumRetainedBytes = 2ull * 1024ull * 1024ull * 1024ull;
    std::size_t maximumShadowAtlasBytes =
        384ull * 1024ull * 1024ull;
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
    std::uint32_t meshClusterCount = 0u;
    std::uint32_t meshPrimitiveCount = 0u;
    std::uint32_t meshInstanceCount = 0u;
    std::uint32_t meshIndexCount = 0u;
    std::uint32_t materialCount = 0u;
    std::uint32_t textureCount = 0u;
    std::uint32_t lightCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t sensorBindingCount = 0u;
    std::uint32_t maximumGaussiansPerTile = 0u;
    std::uint32_t maximumMeshTrianglesPerTile = 0u;
    std::uint32_t shadowLayerCapacity = 0u;
    std::uint32_t rayInstanceCount = 0u;
    std::size_t shadowWorkspaceBytes = 0u;
    std::size_t accelerationStructureBytes = 0u;
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
    std::size_t currentBodyOffset = 0u;
    std::size_t previousBodyOffset = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint64_t frameIndex = 0u;
    std::uint32_t sensorSequence = 0u;
    MRVisualFrameSource source = MR_VISUAL_SOURCE_SIMULATION;
    double captureTimestampSeconds = 0.0;
    double frameAgeSeconds = 0.0;
};

struct HybridDeviceObservationBuffers {
    // Borrowed id<MTLBuffer> values. RGB, depth, and validity are mandatory.
    // Optional truth buffers may be null when their bit is absent from
    // outputMask.
    void* rgb = nullptr;
    void* depth = nullptr;
    void* segmentation = nullptr;
    void* identities = nullptr;
    void* normals = nullptr;
    void* motion = nullptr;
    void* validity = nullptr;
    std::uint32_t outputMask = MR_HYBRID_OUTPUT_ALL_TRUTH;
};

struct MetalHybridComputeEncoderCallbacks {
    void* context = nullptr;
    void (*setLabel)(void* context, const char* label) = nullptr;
    void (*useHeap)(void* context, void* heap) = nullptr;
    void (*useResidencySet)(
        void* context,
        void* residencySet
    ) = nullptr;
    void (*setPipeline)(void* context, void* pipeline) = nullptr;
    void (*setBuffer)(
        void* context,
        void* buffer,
        std::size_t offset,
        std::uint32_t index
    ) = nullptr;
    void (*setBytes)(
        void* context,
        const void* bytes,
        std::size_t length,
        std::uint32_t index
    ) = nullptr;
    void (*dispatchThreads)(
        void* context,
        std::size_t threadCount,
        std::size_t threadsPerThreadgroup
    ) = nullptr;
    void (*dispatchThreadgroups)(
        void* context,
        std::size_t threadgroupCount,
        std::size_t threadsPerThreadgroup
    ) = nullptr;
    void (*dispatchThreadgroupsIndirect)(
        void* context,
        void* arguments,
        std::size_t offset,
        std::size_t threadsPerThreadgroup
    ) = nullptr;

    [[nodiscard]] bool valid() const noexcept {
        return context != nullptr &&
            (useHeap != nullptr || useResidencySet != nullptr) &&
            setPipeline != nullptr &&
            setBuffer != nullptr &&
            setBytes != nullptr &&
            dispatchThreads != nullptr &&
            dispatchThreadgroups != nullptr;
    }
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
    shadowAtlas = 10u,
    temporalAccumulation = 11u,
    meshTileOverflowCounts = 12u,
    cameraStates = 13u,
};

struct MetalHybridFrameCommandContext {
    // Borrowed id<MTLCommandBuffer>. The reference profile creates the
    // acceleration-structure, compute, and resolve encoders it needs without
    // committing or waiting. The fast profile may instead receive a borrowed
    // active id<MTLComputeCommandEncoder>.
    void* commandBuffer = nullptr;
    void* activeComputeCommandEncoder = nullptr;
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

    [[nodiscard]] MetalHybridRendererDiagnostics compile(
        VisualRenderSceneV3&& scene,
        const VisualRendererProfileV1& profile,
        std::uint32_t capacity
    );

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

    // MLX and other lazy graph runtimes use this callback surface to keep the
    // renderer on their active compute encoder. All observation buffers are
    // caller-owned and written in place; no copy, commit, wait, or readback is
    // introduced.
    [[nodiscard]] MetalHybridRendererDiagnostics encodeGraph(
        const MetalWorldFamilyContext& worlds,
        const HybridDeviceStateBatch& liveState,
        std::uint32_t cameraIndex,
        const MetalHybridComputeEncoderCallbacks& encoder,
        const HybridDeviceObservationBuffers& outputs
    );

    // Physical-exposure path. Motion samples cover exposure open through
    // exposure close and remain authoritative; the renderer never predicts
    // future body poses. Outputs are the same buffers exposed by encode().
    [[nodiscard]] MetalHybridRendererDiagnostics encodeFrame(
        const MetalWorldFamilyContext& worlds,
        const VisualMotionSampleBatchV1& motion,
        std::uint32_t cameraIndex,
        const MetalHybridFrameCommandContext& commandContext
    );

    [[nodiscard]] MetalHybridRendererDiagnostics
    readback(HybridObservationBatch& output);

    [[nodiscard]] MetalHybridRendererLayout layout() const noexcept;

    [[nodiscard]] void*
    nativeBuffer(MetalHybridRendererBuffer buffer) const noexcept;

private:
    friend class MetalHybridObjectTracker;
    std::shared_ptr<detail::MetalHybridRendererState> state_;
};

struct MetalHybridObjectTrackBinding {
    std::uint32_t instanceId = MR_INVALID_INDEX;
    std::uint32_t actorFrameOffset = 0u;
    float positionScale = 1.0f;
    float velocityScale = 1.0f;
    std::uint32_t minimumVisiblePixels = 4u;
};

struct MetalHybridObjectTrackerConfig {
    std::uint32_t capacity = 0u;
    std::uint32_t cameraIndex = 0u;
    std::uint32_t rootBodyIndex = MR_INVALID_INDEX;
    std::uint32_t maximumActorHistoryLength = 1u;
    float timestepSeconds = 1.0f / 50.0f;
    // Reject temporal centroid discontinuities without suppressing the
    // current position/confidence measurement.
    float maximumTrackSpeedMetersPerSecond = 1.0e30f;
    std::vector<MetalHybridObjectTrackBinding> bindings;
};

// Reduces rendered metric depth and instance identity directly into the
// compact object-track slots consumed by a TaskPack. The returned device
// observation program executes renderer, reduction, and policy inference in
// one MetalWorld command buffer with no host readback.
class MetalHybridObjectTracker {
public:
    MetalHybridObjectTracker();
    ~MetalHybridObjectTracker();

    MetalHybridObjectTracker(MetalHybridObjectTracker&&) noexcept;
    MetalHybridObjectTracker& operator=(
        MetalHybridObjectTracker&&
    ) noexcept;

    MetalHybridObjectTracker(const MetalHybridObjectTracker&) = delete;
    MetalHybridObjectTracker& operator=(
        const MetalHybridObjectTracker&
    ) = delete;

    [[nodiscard]] MetalHybridRendererDiagnostics compile(
        MetalHybridRenderer& renderer,
        const MetalWorldFamilyContext& worlds,
        MetalHybridObjectTrackerConfig config
    );

    [[nodiscard]] MetalWorldDeviceObservationProgram
    observationProgram() noexcept;

    // Clears device-resident temporal tracks at an explicit simulator reset
    // boundary. No observation plane is read back or reallocated.
    [[nodiscard]] MetalHybridRendererDiagnostics reset();

private:
    struct State;
    static bool encodeObservation(
        void* context,
        const MetalWorldDeviceObservationPass& pass
    );
    std::unique_ptr<State> state_;
};

[[nodiscard]] const char*
metalHybridRendererStatusName(MetalHybridRendererStatus status) noexcept;

} // namespace metalrobo
