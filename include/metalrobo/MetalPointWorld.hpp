#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {
struct MetalPointWorldPreprocessorState;
}

enum class MetalPointWorldStatus : std::uint32_t {
    success = 0u,
    invalidInput,
    metalUnavailable,
    libraryFailure,
    pipelineFailure,
    bufferFailure,
    commandFailure,
};

struct MetalPointWorldDiagnostics {
    MetalPointWorldStatus status = MetalPointWorldStatus::success;
    std::string deviceName;
    std::string message;
    double elapsedMilliseconds = 0.0;
    std::size_t retainedBytes = 0u;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalPointWorldStatus::success;
    }
};

struct PointWorldPreprocessHostInput {
    std::span<const std::uint8_t> rgba;
    std::span<const float> metricDepth;
    std::span<const std::uint8_t> depthValidity;
    std::array<float, 4u> intrinsics{}; // fx, fy, cx, cy
    std::uint32_t width = 320u;
    std::uint32_t height = 180u;
};

struct PointWorldPreprocessResult {
    std::vector<std::array<float, 4u>> normalizedRGBA;
    std::vector<std::array<float, 4u>> cameraPoints;
};

struct PointWorldPreprocessDeviceFrame {
    // Borrowed id<MTLBuffer> values. encode() allocates, commits, waits, and
    // retains none of them; the owner serializes one context's use.
    void* rgba = nullptr;
    void* metricDepth = nullptr;
    void* depthValidity = nullptr;
    void* normalizedRGBA = nullptr;
    void* cameraPoints = nullptr;
    std::array<float, 4u> intrinsics{};
    std::uint32_t width = 320u;
    std::uint32_t height = 180u;
};

class MetalPointWorldPreprocessor {
public:
    explicit MetalPointWorldPreprocessor(std::string metallibPath = {});
    ~MetalPointWorldPreprocessor();

    MetalPointWorldPreprocessor(MetalPointWorldPreprocessor&&) noexcept;
    MetalPointWorldPreprocessor& operator=(MetalPointWorldPreprocessor&&) noexcept;
    MetalPointWorldPreprocessor(const MetalPointWorldPreprocessor&) = delete;
    MetalPointWorldPreprocessor& operator=(const MetalPointWorldPreprocessor&) = delete;

    [[nodiscard]] MetalPointWorldDiagnostics encode(
        const PointWorldPreprocessDeviceFrame& frame,
        void* metalComputeCommandEncoder
    );

    // Qualification path using persistent shared staging and one explicit wait.
    [[nodiscard]] MetalPointWorldDiagnostics run(
        const PointWorldPreprocessHostInput& input,
        PointWorldPreprocessResult& output
    );

private:
    std::shared_ptr<detail::MetalPointWorldPreprocessorState> state_;
};

[[nodiscard]] const char* metalPointWorldStatusName(MetalPointWorldStatus status) noexcept;

} // namespace metalrobo
