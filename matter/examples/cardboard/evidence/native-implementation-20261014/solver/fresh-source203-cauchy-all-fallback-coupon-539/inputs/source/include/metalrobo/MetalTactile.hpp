#pragma once

#include "metalrobo/Tactile.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace metalrobo {

namespace detail {
struct MetalTactileState;
} // namespace detail

struct MetalTactileConfig {
    std::string metallibPath;
    std::uint32_t contactCapacityPerEnvironment = 128u;
    bool enableDebugHits = false;
    std::size_t maximumRetainedBytes =
        2ull * 1024ull * 1024ull * 1024ull;
};

enum class MetalTactileStatus : std::uint32_t {
    success = 0u,
    invalidSystem,
    invalidConfiguration,
    capacityOverflow,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    incompatibleState,
    notCompiled,
    internalFailure,
};

struct MetalTactileLayout {
    std::uint32_t environmentCapacity = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t shapeCount = 0u;
    std::uint32_t sensorCount = 0u;
    std::uint32_t sampleCount = 0u;
    std::uint32_t targetCount = 0u;
    std::uint32_t contactCapacityPerEnvironment = 0u;
    std::size_t retainedBytes = 0u;
    std::size_t bytesPerEnvironment = 0u;
    MRTactileQueryBackend queryBackend =
        MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4;
    bool hardwareRayQueriesAvailable = false;
};

struct MetalTactileDiagnostics {
    MetalTactileStatus status = MetalTactileStatus::success;
    MetalTactileLayout layout;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalTactileStatus::success;
    }
};

struct MetalTactileHostFrame {
    std::uint32_t environmentCount = 0u;
    std::span<const MRBodyStateGPU> bodies;
    std::span<const MRTactileContactGPU> contacts;
    std::span<const std::uint32_t> contactCounts;
    std::span<const std::uint32_t> resetMask;
    // Base simulation/control-step interval. Sensor decimation is applied
    // internally when computing metric depth velocity.
    float observationTimestepSeconds = 0.0f;
    float contactImpulseTimestepSeconds = 0.0f;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;
};

struct MetalTactileDeviceFrame {
    // Borrowed id<MTLBuffer> values. Arrays are environment-major. Contacts
    // and reset masks may be null, in which case zero-filled persistent
    // buffers are used.
    void* bodyStates = nullptr;
    void* contacts = nullptr;
    void* contactCounts = nullptr;
    void* resetMask = nullptr;
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t contactCapacityPerEnvironment = 0u;
    // Base simulation/control-step interval. Sensor decimation is applied
    // internally when computing metric depth velocity.
    float observationTimestepSeconds = 0.0f;
    float contactImpulseTimestepSeconds = 0.0f;
    std::uint64_t frameIndex = 0u;
    double timestampSeconds = 0.0;
};

enum class MetalTactileBuffer : std::uint32_t {
    penetrationDepth = 0u,
    depthVelocity = 1u,
    validity = 2u,
    objectShapeIds = 3u,
    debugHits = 4u,
    summaries = 5u,
    statuses = 6u,
    tangentialMotion = 7u,
};

// Persistent fixed-capacity Apple-GPU tactile runtime. encode() composes into
// an existing Metal command encoder and performs no allocation, commit,
// synchronization, or readback. The caller must keep borrowed buffers alive
// and serialize command-buffer use of one context. nativeBuffer() is the
// MLX/Core ML/native tensor publication boundary.
class MetalTactileContext {
public:
    explicit MetalTactileContext(MetalTactileConfig config = {});
    ~MetalTactileContext();

    MetalTactileContext(MetalTactileContext&& other) noexcept;
    MetalTactileContext& operator=(MetalTactileContext&& other) noexcept;

    MetalTactileContext(const MetalTactileContext&) = delete;
    MetalTactileContext& operator=(const MetalTactileContext&) = delete;

    [[nodiscard]] MetalTactileDiagnostics compile(
        const CookedTactileSystem& tactile,
        const EngineModel& model,
        std::uint32_t environmentCapacity
    );

    // Convenience/test path: copies host state into persistent unified-memory
    // staging, runs one command buffer, and waits. RL should call encode().
    [[nodiscard]] MetalTactileDiagnostics observe(
        const MetalTactileHostFrame& frame
    );

    [[nodiscard]] MetalTactileDiagnostics encode(
        const MetalTactileDeviceFrame& frame,
        void* metalComputeCommandEncoder
    );

    // Explicitly requested diagnostic path. Copies only the active prefix of
    // device outputs into reusable shared staging.
    [[nodiscard]] MetalTactileDiagnostics readback(
        std::uint32_t environmentCount,
        TactileObservationBatch& output
    );

    // Clears temporal history with a bounded GPU fill. No output definition
    // or static sensor data changes.
    [[nodiscard]] MetalTactileDiagnostics clearHistory();

    [[nodiscard]] MetalTactileLayout layout() const noexcept;

    [[nodiscard]] void*
    nativeBuffer(MetalTactileBuffer buffer) const noexcept;

private:
    std::shared_ptr<detail::MetalTactileState> state_;
};

[[nodiscard]] const char*
metalTactileStatusName(MetalTactileStatus status) noexcept;

} // namespace metalrobo
