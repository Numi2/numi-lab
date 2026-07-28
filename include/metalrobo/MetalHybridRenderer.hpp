#pragma once

#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/hybrid_renderer_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {
struct MetalHybridRendererState;
} // namespace detail

struct HybridGaussianScene {
    std::string id;
    std::uint32_t assetCount = 0u;
    std::vector<MRHybridGaussianGPU> gaussians;

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
};

enum class MetalHybridRendererBuffer : std::uint32_t {
    rgb = 0u,
    depth = 1u,
    segmentation = 2u,
    projectedGaussians = 3u,
    tileOverflowCounts = 4u,
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
