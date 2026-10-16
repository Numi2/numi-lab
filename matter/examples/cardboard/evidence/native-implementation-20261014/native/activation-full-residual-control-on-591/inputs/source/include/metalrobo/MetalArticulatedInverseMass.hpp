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
struct MetalArticulatedInverseMassContextState;
struct MetalArticulatedInverseMassSubmissionState;
} // namespace detail

// Compact environment-major input for one immutable articulation.
// rightHandSides is packed [environment][rhs][local v].
struct MetalArticulatedInverseMassInput {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::size_t rhsCount = 0u;
    std::span<const float> q{};
    std::span<const float> rightHandSides{};
};

struct MetalArticulatedInverseMassConfig {
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as a fallback.
    // A non-empty path is an explicit trusted ABI-compatible override.
    std::string metallibPath;
};

enum class MetalArticulatedInverseMassHostStatus :
    std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    capacityOverflow,
    arithmeticOverflow,
    nonfiniteInput,
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

struct MetalArticulatedInverseMassLayout {
    MRInverseMassDispatchGPU dispatch{};
    std::size_t qElements = 0u;
    std::size_t qBytes = 0u;
    std::size_t rhsElements = 0u;
    std::size_t rhsBytes = 0u;
    std::size_t outputElements = 0u;
    std::size_t outputBytes = 0u;
    std::size_t statusElements = 0u;
    std::size_t statusBytes = 0u;
    // Includes immutable model streams and the one-element joint placeholder
    // required for a floating free body with no joint records.
    std::size_t totalRequiredBytes = 0u;
};

struct MetalArticulatedInverseMassResult {
    MetalArticulatedInverseMassLayout layout{};
    // Packed [environment][rhs][local v], matching the compact input.
    std::vector<float> output;
    std::vector<MRInverseMassStatusGPU> statuses;
};

struct MetalArticulatedInverseMassDiagnostics {
    MetalArticulatedInverseMassHostStatus status =
        MetalArticulatedInverseMassHostStatus::success;
    MetalArticulatedInverseMassLayout layout{};
    bool dispatched = false;
    bool published = false;
    std::uint32_t successfulEnvironmentCount = 0u;
    std::uint32_t failedEnvironmentCount = 0u;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode =
        MR_INVERSE_MASS_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MetalArticulatedInverseMassHostStatus::success;
    }
};

struct MetalArticulatedInverseMassContextStats {
    std::uint64_t pipelineCreationCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    bool hasInFlightSubmission = false;
};

class MetalArticulatedInverseMassContext;

// A committed inverse-mass action. submit() snapshots every caller-owned span
// and immutable model stream before returning. wait() consumes the ticket and
// publishes atomically. Destruction waits and discards; the ticket retains the
// context so destroying the public context first is safe.
class MetalArticulatedInverseMassSubmission {
public:
    MetalArticulatedInverseMassSubmission() noexcept;
    ~MetalArticulatedInverseMassSubmission();

    MetalArticulatedInverseMassSubmission(
        MetalArticulatedInverseMassSubmission&& other
    ) noexcept;
    MetalArticulatedInverseMassSubmission& operator=(
        MetalArticulatedInverseMassSubmission&& other
    ) noexcept;

    MetalArticulatedInverseMassSubmission(
        const MetalArticulatedInverseMassSubmission&
    ) = delete;
    MetalArticulatedInverseMassSubmission& operator=(
        const MetalArticulatedInverseMassSubmission&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] MetalArticulatedInverseMassDiagnostics wait(
        MetalArticulatedInverseMassResult& result
    );

private:
    friend class MetalArticulatedInverseMassContext;
    std::unique_ptr<
        detail::MetalArticulatedInverseMassSubmissionState
    > state_;
};

// Persistent checked Metal executor. Pipeline construction occurs once and
// its ten-buffer shared arena grows geometrically but never shrinks. One batch
// may be in flight because every submission intentionally reuses that arena.
class MetalArticulatedInverseMassContext {
public:
    explicit MetalArticulatedInverseMassContext(
        MetalArticulatedInverseMassConfig config = {}
    );
    ~MetalArticulatedInverseMassContext();

    MetalArticulatedInverseMassContext(
        MetalArticulatedInverseMassContext&& other
    ) noexcept;
    MetalArticulatedInverseMassContext& operator=(
        MetalArticulatedInverseMassContext&& other
    ) noexcept;

    MetalArticulatedInverseMassContext(
        const MetalArticulatedInverseMassContext&
    ) = delete;
    MetalArticulatedInverseMassContext& operator=(
        const MetalArticulatedInverseMassContext&
    ) = delete;

    [[nodiscard]] MetalArticulatedInverseMassDiagnostics submit(
        const EngineModel& model,
        const MetalArticulatedInverseMassInput& input,
        MetalArticulatedInverseMassSubmission& submission
    );

    [[nodiscard]] MetalArticulatedInverseMassDiagnostics run(
        const EngineModel& model,
        const MetalArticulatedInverseMassInput& input,
        MetalArticulatedInverseMassResult& result
    );

    [[nodiscard]] MetalArticulatedInverseMassContextStats stats()
        const noexcept;

private:
    std::shared_ptr<
        detail::MetalArticulatedInverseMassContextState
    > state_;
};

[[nodiscard]] MetalArticulatedInverseMassDiagnostics
runMetalArticulatedInverseMass(
    const EngineModel& model,
    const MetalArticulatedInverseMassInput& input,
    MetalArticulatedInverseMassResult& result,
    const MetalArticulatedInverseMassConfig& config = {}
);

[[nodiscard]] const char*
metalArticulatedInverseMassHostStatusName(
    MetalArticulatedInverseMassHostStatus status
) noexcept;

} // namespace metalrobo
