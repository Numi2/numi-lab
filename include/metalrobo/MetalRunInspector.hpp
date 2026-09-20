#pragma once

#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace metalrobo {

namespace detail {
struct MetalRunInspectorState;
} // namespace detail

// A bounded, presentation-only view of one environment in a resident Metal
// rollout. The inspector produces GPU-resident linear RGBA frames; callers
// acquire a completed slot and release it after their presentation command
// buffer completes. When all slots are busy, it drops a frame rather than
// blocking the rollout scheduler.
struct MetalRunInspectorConfig {
    std::string metallibPath;
    std::uint32_t width = 640u;
    std::uint32_t height = 360u;
    std::uint32_t environmentIndex = 0u;
    std::uint32_t maximumFramesInFlight = 3u;
    std::size_t maximumRetainedBytes = 256ull * 1024ull * 1024ull;
};

struct MetalRunInspectorFrame {
    // Borrowed id<MTLBuffer>. Valid until releaseFrame() for slotIndex.
    void* rgb = nullptr;
    std::uint32_t slotIndex = MR_INVALID_INDEX;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint64_t frameIndex = 0u;
    std::uint64_t submissionIndex = 0u;
    std::uint32_t environmentIndex = 0u;
    std::uint32_t droppedFrames = 0u;
};

class MetalRunInspector {
public:
    explicit MetalRunInspector(MetalRunInspectorConfig config = {});
    ~MetalRunInspector();

    MetalRunInspector(MetalRunInspector&& other) noexcept;
    MetalRunInspector& operator=(MetalRunInspector&& other) noexcept;

    MetalRunInspector(const MetalRunInspector&) = delete;
    MetalRunInspector& operator=(const MetalRunInspector&) = delete;

    [[nodiscard]] MetalHybridRendererDiagnostics compile(
        VisualRenderSceneV3&& scene,
        const VisualRendererProfileV1& profile,
        const MetalWorldFamilyContext& worlds
    );

    [[nodiscard]] MetalWorldInspectionProgram
    inspectionProgram() noexcept;

    // Gates presentation encoding without changing the rollout's physics or
    // accepted-state submission. Existing display-owned slots remain valid.
    void setEnabled(bool enabled) noexcept;

    // Returns false when no completed presentation frame is available. A
    // successful acquire transfers one ring slot to the caller.
    [[nodiscard]] bool acquireLatestFrame(
        MetalRunInspectorFrame& output
    ) noexcept;

    // Releases a frame slot after the consumer's GPU presentation completes.
    void releaseFrame(std::uint32_t slotIndex) noexcept;

private:
    static bool encodeInspection(
        void* context,
        const MetalWorldInspectionPass& pass
    );

    std::shared_ptr<detail::MetalRunInspectorState> state_;
};

} // namespace metalrobo
