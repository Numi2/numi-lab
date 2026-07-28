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
struct MetalArticulatedOperatorContextState;
struct MetalArticulatedOperatorSubmissionState;
} // namespace detail

// Packed, environment-major input for the synchronous Metal articulated
// operator. q contains environmentCount * articulation.nq floats. points
// contains environmentCount * pointCount records. All query body indices are
// global EngineModel body indices.
struct MetalArticulatedOperatorInput {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::size_t pointCount = 0u;
    std::span<const float> q{};
    std::span<const MRArticulatedPointImpulseGPU> points{};
};

struct MetalArticulatedOperatorConfig {
    // The dense mass matrix is a correctness diagnostic. The factor-backed
    // impulse solve always runs; disabling this avoids its output bandwidth.
    bool writeDiagnosticMassMatrix = false;
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as a fallback.
    // A non-empty path is an explicit trusted ABI-compatible override.
    std::string metallibPath;
};

enum class MetalArticulatedOperatorHostStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    capacityOverflow,
    arithmeticOverflow,
    nonfiniteInput,
    invalidPointQuery,
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

// The host always derives compact strides; callers cannot smuggle unchecked
// buffer layouts into the raw kernel. Counts and byte sizes are retained so a
// batch can be audited without reconstructing overflow-sensitive arithmetic.
struct MetalArticulatedOperatorLayout {
    MRArticulatedOperatorDispatchGPU dispatch{};
    std::size_t qElements = 0u;
    std::size_t qBytes = 0u;
    std::size_t pointElements = 0u;
    std::size_t pointBytes = 0u;
    std::size_t bodyPoseElements = 0u;
    std::size_t bodyPoseBytes = 0u;
    std::size_t pointWorldElements = 0u;
    std::size_t pointWorldBytes = 0u;
    std::size_t massMatrixElements = 0u;
    std::size_t massMatrixBytes = 0u;
    std::size_t pointJacobianElements = 0u;
    std::size_t pointJacobianBytes = 0u;
    std::size_t generalizedElements = 0u;
    std::size_t generalizedBytes = 0u;
    std::size_t statusElements = 0u;
    std::size_t statusBytes = 0u;
    // Includes immutable model buffers and one-element placeholders required
    // to bind logically empty Metal buffers.
    std::size_t totalAllocatedBytes = 0u;
};

struct MetalArticulatedOperatorResult {
    MetalArticulatedOperatorLayout layout{};
    std::vector<MRArticulatedBodyPoseGPU> bodyPoses;
    std::vector<MRArticulatedPointWorldGPU> pointWorld;
    std::vector<float> diagnosticMassMatrix;
    std::vector<float> pointJacobians;
    std::vector<float> generalizedImpulse;
    std::vector<float> deltaVelocity;
    std::vector<MRArticulatedOperatorStatusGPU> statuses;
};

struct MetalArticulatedOperatorDiagnostics {
    MetalArticulatedOperatorHostStatus status =
        MetalArticulatedOperatorHostStatus::success;
    MetalArticulatedOperatorLayout layout{};
    bool dispatched = false;
    bool published = false;
    std::uint32_t successfulEnvironmentCount = 0u;
    std::uint32_t failedEnvironmentCount = 0u;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode =
        MR_ARTICULATED_OPERATOR_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalArticulatedOperatorHostStatus::success;
    }
};

// Lifetime counters for a reusable operator context. retainedBufferBytes is
// the physical capacity of the grow-only Metal buffer arena; an individual
// result layout continues to report only that batch's required bytes.
struct MetalArticulatedOperatorContextStats {
    std::uint64_t pipelineCreationCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    bool hasInFlightSubmission = false;
};

class MetalArticulatedOperatorContext;

// A committed Metal batch. submit() copies all caller-owned spans before
// returning, so their storage can immediately be reused. wait() may be called
// once and transactionally publishes the completed typed result. Destroying a
// live submission waits for the GPU and discards its result, making context
// and submission destruction safe in either order.
class MetalArticulatedOperatorSubmission {
public:
    MetalArticulatedOperatorSubmission() noexcept;
    ~MetalArticulatedOperatorSubmission();

    MetalArticulatedOperatorSubmission(
        MetalArticulatedOperatorSubmission&& other
    ) noexcept;
    MetalArticulatedOperatorSubmission& operator=(
        MetalArticulatedOperatorSubmission&& other
    ) noexcept;

    MetalArticulatedOperatorSubmission(
        const MetalArticulatedOperatorSubmission&
    ) = delete;
    MetalArticulatedOperatorSubmission& operator=(
        const MetalArticulatedOperatorSubmission&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] MetalArticulatedOperatorDiagnostics wait(
        MetalArticulatedOperatorResult& result
    );

private:
    friend class MetalArticulatedOperatorContext;
    std::unique_ptr<
        detail::MetalArticulatedOperatorSubmissionState
    > state_;
};

// Reusable execution context for steady-state simulation. Device discovery,
// metallib loading, command-queue creation, and pipeline compilation happen at
// most once. Its 15-buffer arena is reused and grows geometrically as batch
// sizes increase. Calls are thread-safe, but a context deliberately admits
// only one in-flight submission because every batch shares the same arena;
// independent contexts provide safe overlap when multiple queues are useful.
class MetalArticulatedOperatorContext {
public:
    explicit MetalArticulatedOperatorContext(
        MetalArticulatedOperatorConfig config = {}
    );
    ~MetalArticulatedOperatorContext();

    MetalArticulatedOperatorContext(
        MetalArticulatedOperatorContext&& other
    ) noexcept;
    MetalArticulatedOperatorContext& operator=(
        MetalArticulatedOperatorContext&& other
    ) noexcept;

    MetalArticulatedOperatorContext(
        const MetalArticulatedOperatorContext&
    ) = delete;
    MetalArticulatedOperatorContext& operator=(
        const MetalArticulatedOperatorContext&
    ) = delete;

    // Encodes and commits without waiting for GPU completion. Host validation,
    // capacity growth, and all input copies are complete before this returns.
    // submission must be empty and remains unchanged on rejection.
    [[nodiscard]] MetalArticulatedOperatorDiagnostics submit(
        const EngineModel& model,
        const MetalArticulatedOperatorInput& input,
        MetalArticulatedOperatorSubmission& submission
    );

    // Convenience synchronous path using submit() followed by wait().
    [[nodiscard]] MetalArticulatedOperatorDiagnostics run(
        const EngineModel& model,
        const MetalArticulatedOperatorInput& input,
        MetalArticulatedOperatorResult& result
    );

    [[nodiscard]] MetalArticulatedOperatorContextStats stats()
        const noexcept;

private:
    std::shared_ptr<
        detail::MetalArticulatedOperatorContextState
    > state_;
};

// Executes one synchronous command buffer. The EngineModel and every layout,
// span, query, required element count, byte count, Metal allocation, and
// pipeline capability are checked before encoding. The raw kernel binding
// table remains private to the Objective-C++ implementation.
//
// Host-side failure leaves result bit-for-bit unchanged. Once a command
// completes, the entire typed batch and its per-environment GPU statuses are
// published together. A GPU environment failure therefore returns
// gpuEnvironmentFailure with published=true so the status stream remains
// inspectable; failed environments have zeroed payload slots.
[[nodiscard]] MetalArticulatedOperatorDiagnostics
runMetalArticulatedOperator(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorResult& result,
    const MetalArticulatedOperatorConfig& config = {}
);

[[nodiscard]] const char* metalArticulatedOperatorHostStatusName(
    MetalArticulatedOperatorHostStatus status
) noexcept;

} // namespace metalrobo
