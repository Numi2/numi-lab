#pragma once

#include "metalrobo/WorldCompiler.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace metalrobo {

namespace detail {
struct MetalWorldFamilyContextState;
} // namespace detail

struct MetalWorldFamilyConfig {
    // Empty discovers the co-installed MetalRobo metallib. A non-empty path
    // is an explicit trusted ABI-compatible override.
    std::string metallibPath;
};

enum class MetalWorldFamilyStatus : std::uint32_t {
    success = 0u,
    invalidFamily,
    invalidCapacity,
    capacityOverflow,
    arithmeticOverflow,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    notCompiled,
    internalFailure,
};

struct MetalWorldFamilyLayout {
    std::uint32_t capacity = 0u;
    std::uint32_t activeInstanceCount = 0u;
    std::uint32_t assetCountPerInstance = 0u;
    std::uint32_t sensorCountPerInstance = 0u;
    std::uint32_t appearanceCountPerInstance = 0u;
    std::uint32_t variationCount = 0u;
    std::uint32_t categoricalValueCount = 0u;
    std::size_t immutablePrivateBytes = 0u;
    std::size_t instancePrivateBytes = 0u;
    std::size_t assetPrivateBytes = 0u;
    std::size_t sensorPrivateBytes = 0u;
    std::size_t appearancePrivateBytes = 0u;

    [[nodiscard]] std::size_t totalPrivateBytes() const noexcept {
        return immutablePrivateBytes + instancePrivateBytes +
            assetPrivateBytes + sensorPrivateBytes +
            appearancePrivateBytes;
    }
};

struct MetalWorldFamilyDiagnostics {
    MetalWorldFamilyStatus status = MetalWorldFamilyStatus::success;
    MetalWorldFamilyLayout layout{};
    std::uint64_t familyFingerprint = 0u;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalWorldFamilyStatus::success;
    }
};

struct MetalWorldFamilyStats {
    std::uint64_t compileCount = 0u;
    std::uint64_t sampleCount = 0u;
    std::uint64_t readbackCount = 0u;
    std::size_t retainedPrivateBytes = 0u;
    std::uint32_t activeInstanceCount = 0u;
    std::uint64_t familyFingerprint = 0u;
};

enum class MetalWorldFamilyBuffer : std::uint32_t {
    instanceHeaders = 0u,
    assetInstances = 1u,
    sensorInstances = 2u,
    appearanceInstances = 3u,
};

// Persistent GPU-resident storage for one topology-compatible WorldFamily.
// compile() uploads immutable template and variation streams once and sizes
// private output arenas for capacity worlds. sample() then changes only one
// small shared uniform record and dispatches one thread per environment.
class MetalWorldFamilyContext {
public:
    explicit MetalWorldFamilyContext(
        MetalWorldFamilyConfig config = {}
    );
    ~MetalWorldFamilyContext();

    MetalWorldFamilyContext(MetalWorldFamilyContext&& other) noexcept;
    MetalWorldFamilyContext& operator=(
        MetalWorldFamilyContext&& other
    ) noexcept;

    MetalWorldFamilyContext(const MetalWorldFamilyContext&) = delete;
    MetalWorldFamilyContext& operator=(
        const MetalWorldFamilyContext&
    ) = delete;

    [[nodiscard]] MetalWorldFamilyDiagnostics compile(
        const WorldFamily& family,
        std::uint32_t capacity
    );

    // Samples directly into private Metal buffers and leaves them resident for
    // physics, rendering, sensor, and policy stages encoded afterward.
    [[nodiscard]] MetalWorldFamilyDiagnostics sample(
        std::uint32_t instanceCount,
        std::uint64_t seed
    );

    // Explicit diagnostic/export boundary. The normal simulation loop should
    // consume nativeBuffer() and never call readback().
    [[nodiscard]] MetalWorldFamilyDiagnostics readback(
        WorldInstanceBatch& output
    );

    [[nodiscard]] MetalWorldFamilyLayout layout() const noexcept;
    [[nodiscard]] MetalWorldFamilyStats stats() const noexcept;

    // Borrowed id<MTLBuffer> exposed as an opaque pointer for Objective-C++,
    // MLX custom primitives, and other native Metal graph stages. The pointer
    // remains valid until the next successful compile() or context destruction.
    [[nodiscard]] void* nativeBuffer(
        MetalWorldFamilyBuffer buffer
    ) const noexcept;

private:
    std::shared_ptr<detail::MetalWorldFamilyContextState> state_;
};

[[nodiscard]] const char* metalWorldFamilyStatusName(
    MetalWorldFamilyStatus status
) noexcept;

} // namespace metalrobo
