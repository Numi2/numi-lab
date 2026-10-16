#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {
struct MetalArticulatedABAContextState;
struct MetalArticulatedABASubmissionState;
} // namespace detail

// Compact environment-major state for one articulation. q, v, and effort
// contain environmentCount * nq/nv/nv floats respectively. An empty wrench
// span disables external wrenches; otherwise it contains environmentCount *
// bodyCount articulation-local records.
struct MetalArticulatedABAInput {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::span<const float> q{};
    std::span<const float> v{};
    std::span<const float> effort{};
    std::span<const MRABABodyWrenchGPU> bodyWrenches{};
    bool applyBodyDamping = true;
};

struct MetalArticulatedABAConfig {
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as a fallback.
    // A non-empty path is an explicit trusted ABI-compatible override.
    std::string metallibPath;
};

enum class MetalArticulatedABAHostStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    capacityOverflow,
    arithmeticOverflow,
    nonfiniteInput,
    invalidBodyWrench,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    gpuEnvironmentFailure,
    internalFailure,
    contextBusy,
};

// The checked host derives compact strides; raw caller-selected strides never
// cross this boundary. Counts and bytes make overflow and allocation decisions
// auditable without reconstructing them from the dispatch record.
struct MetalArticulatedABALayout {
    MRABADispatchGPU dispatch{};
    std::size_t qElements = 0u;
    std::size_t qBytes = 0u;
    std::size_t vElements = 0u;
    std::size_t vBytes = 0u;
    std::size_t effortElements = 0u;
    std::size_t effortBytes = 0u;
    std::size_t wrenchElements = 0u;
    std::size_t wrenchBytes = 0u;
    std::size_t accelerationElements = 0u;
    std::size_t accelerationBytes = 0u;
    std::size_t nextVElements = 0u;
    std::size_t nextVBytes = 0u;
    std::size_t nextQElements = 0u;
    std::size_t nextQBytes = 0u;
    std::size_t statusElements = 0u;
    std::size_t statusBytes = 0u;
    // Includes immutable model streams and typed one-element placeholders
    // required for logically empty Metal buffers.
    std::size_t totalRequiredBytes = 0u;
};

struct MetalArticulatedABAResult {
    MetalArticulatedABALayout layout{};
    std::vector<float> acceleration;
    std::vector<float> nextV;
    std::vector<float> nextQ;
    std::vector<MRABAStatusGPU> statuses;
};

struct MetalArticulatedABADiagnostics {
    MetalArticulatedABAHostStatus status =
        MetalArticulatedABAHostStatus::success;
    MetalArticulatedABALayout layout{};
    bool dispatched = false;
    bool published = false;
    std::uint32_t successfulEnvironmentCount = 0u;
    std::uint32_t failedEnvironmentCount = 0u;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode = MR_ABA_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalArticulatedABAHostStatus::success;
    }
};

struct MetalArticulatedABAContextStats {
    std::uint64_t pipelineCreationCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    bool hasInFlightSubmission = false;
};

class MetalArticulatedABAContext;

// A committed ABA batch. submit() has copied every caller-owned span before
// returning, so its storage can immediately be reused. wait() consumes the
// ticket and transactionally publishes one typed result. Destroying a live
// ticket waits and discards its result; the ticket retains its execution
// context, so context-before-ticket destruction is safe.
class MetalArticulatedABASubmission {
public:
    MetalArticulatedABASubmission() noexcept;
    ~MetalArticulatedABASubmission();

    MetalArticulatedABASubmission(
        MetalArticulatedABASubmission&& other
    ) noexcept;
    MetalArticulatedABASubmission& operator=(
        MetalArticulatedABASubmission&& other
    ) noexcept;

    MetalArticulatedABASubmission(
        const MetalArticulatedABASubmission&
    ) = delete;
    MetalArticulatedABASubmission& operator=(
        const MetalArticulatedABASubmission&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] MetalArticulatedABADiagnostics wait(
        MetalArticulatedABAResult& result
    );

private:
    friend class MetalArticulatedABAContext;
    std::unique_ptr<detail::MetalArticulatedABASubmissionState>
        state_;
};

// Reusable ABA execution context. Device discovery, command queue, metallib
// loading, and pipeline compilation happen once. Its fourteen shared buffers
// are owned by the context, grow geometrically, and never expose caller aliases
// to the GPU. Calls are thread-safe. One submission may be in flight because
// all batches intentionally share the same arena; independent contexts provide
// safe overlap when multiple queues are useful.
class MetalArticulatedABAContext {
public:
    explicit MetalArticulatedABAContext(
        MetalArticulatedABAConfig config = {}
    );
    ~MetalArticulatedABAContext();

    MetalArticulatedABAContext(
        MetalArticulatedABAContext&& other
    ) noexcept;
    MetalArticulatedABAContext& operator=(
        MetalArticulatedABAContext&& other
    ) noexcept;

    MetalArticulatedABAContext(
        const MetalArticulatedABAContext&
    ) = delete;
    MetalArticulatedABAContext& operator=(
        const MetalArticulatedABAContext&
    ) = delete;

    // Encodes and commits without waiting. Host validation, capacity growth,
    // and all input copies are complete before this returns. submission must
    // be empty and remains unchanged on rejection.
    [[nodiscard]] MetalArticulatedABADiagnostics submit(
        const EngineModel& model,
        const MetalArticulatedABAInput& input,
        MetalArticulatedABASubmission& submission
    );

    // Convenience synchronous path using submit() followed by wait(). Host-side
    // failure leaves result bit-for-bit unchanged. After completion, all
    // payloads and statuses are validated before one transactional publication.
    // A valid GPU environment failure publishes zeroed payload slots plus its
    // typed status and returns gpuEnvironmentFailure.
    [[nodiscard]] MetalArticulatedABADiagnostics run(
        const EngineModel& model,
        const MetalArticulatedABAInput& input,
        MetalArticulatedABAResult& result
    );

    [[nodiscard]] MetalArticulatedABAContextStats stats()
        const noexcept;

private:
    std::shared_ptr<detail::MetalArticulatedABAContextState> state_;
};

// One-shot convenience path. Repeated simulation should retain a context.
[[nodiscard]] MetalArticulatedABADiagnostics runMetalArticulatedABA(
    const EngineModel& model,
    const MetalArticulatedABAInput& input,
    MetalArticulatedABAResult& result,
    const MetalArticulatedABAConfig& config = {}
);

[[nodiscard]] const char* metalArticulatedABAHostStatusName(
    MetalArticulatedABAHostStatus status
) noexcept;

} // namespace metalrobo
