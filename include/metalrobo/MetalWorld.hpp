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
struct MetalWorldContextState;
struct MetalWorldSubmissionState;
} // namespace detail

enum class MetalWorldSolverMode : std::uint32_t {
    // The first production graph composes generic articulated free motion,
    // resets, transactional substep commits, and state observations. Contact
    // modes are reserved now so callers cannot confuse this with a completed
    // collision/contact world.
    freeMotionABA = 0u,
    throughputPGS = 1u,
    throughputTGS = 2u,
};

enum class MetalWorldCapacityClass : std::uint32_t {
    uncompiled = 0u,
    compactABA12 = 1u,
    fullABA32 = 2u,
};

enum class MetalWorldHostStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    unsupportedSolverMode,
    invalidDimensions,
    capacityOverflow,
    arithmeticOverflow,
    nonfiniteInput,
    invalidReset,
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

struct MetalWorldCompileDiagnostics {
    MetalWorldHostStatus status = MetalWorldHostStatus::success;
    std::uint64_t fingerprint = 0u;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalWorldHostStatus::success;
    }
};

// Immutable validated snapshot consumed by MetalWorldContext. The current
// capability advances exactly one selected articulation; collision geometry
// remains compiled in the snapshot for the next graph tranche but is not yet
// executed. Accessors expose facts without permitting fingerprint forgery.
class CompiledWorld {
public:
    CompiledWorld() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const EngineModel& model() const noexcept;
    [[nodiscard]] std::uint32_t articulationIndex() const noexcept;
    [[nodiscard]] std::uint32_t nq() const noexcept;
    [[nodiscard]] std::uint32_t nv() const noexcept;
    [[nodiscard]] std::uint32_t bodyCount() const noexcept;
    [[nodiscard]] MetalWorldCapacityClass capacityClass()
        const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;

private:
    friend MetalWorldCompileDiagnostics compileMetalWorld(
        const EngineModel&,
        std::uint32_t,
        CompiledWorld&
    );
    friend class MetalWorldContext;

    EngineModel model_;
    std::uint32_t articulationIndex_ = MR_INVALID_INDEX;
    MetalWorldCapacityClass capacityClass_ =
        MetalWorldCapacityClass::uncompiled;
    std::uint64_t fingerprint_ = 0u;
};

// Validates and snapshots one canonical articulation transactionally. On
// failure, compiled remains bit-for-bit logically unchanged.
[[nodiscard]] MetalWorldCompileDiagnostics compileMetalWorld(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    CompiledWorld& compiled
);

// One submission encodes controlStepCount control steps. initialQ/initialV are
// packed [environment][local coordinate]. efforts are packed
// [control step][environment][local v]. Optional reset masks are packed
// [control step][environment]; when present, resetQ/resetV contain one reset
// state per environment and are applied immediately before that step.
struct MetalWorldBatch {
    std::size_t environmentCount = 0u;
    std::size_t controlStepCount = 0u;
    std::span<const float> initialQ{};
    std::span<const float> initialV{};
    std::span<const float> efforts{};
    std::span<const std::uint32_t> resetMasks{};
    std::span<const float> resetQ{};
    std::span<const float> resetV{};
};

struct MetalWorldStepConfig {
    // Control-period duration. The immutable model gravity is retained and
    // its authored integration timestep is replaced by
    // timestepSeconds / physicsSubsteps for this submission.
    float timestepSeconds = 1.0f / 60.0f;
    std::uint32_t physicsSubsteps = 1u;
    MetalWorldSolverMode solverMode =
        MetalWorldSolverMode::freeMotionABA;
    bool applyBodyDamping = true;
    bool deterministic = true;
};

struct MetalWorldConfig {
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as fallback.
    std::string metallibPath;
};

struct MetalWorldLayout {
    MRMetalWorldDispatchGPU dispatch{};
    MRABADispatchGPU abaDispatch{};
    std::size_t initialQElements = 0u;
    std::size_t initialVElements = 0u;
    std::size_t effortElements = 0u;
    std::size_t resetMaskElements = 0u;
    std::size_t resetQElements = 0u;
    std::size_t resetVElements = 0u;
    std::size_t observationElements = 0u;
    std::size_t accelerationElements = 0u;
    std::size_t statusElements = 0u;
    std::size_t totalRequiredBytes = 0u;
};

struct MetalWorldResult {
    MetalWorldLayout layout{};
    // Accepted state after the last encoded control step.
    std::vector<float> finalQ;
    std::vector<float> finalV;
    // Packed [control step][environment][q then v].
    std::vector<float> observations;
    // Packed [control step][environment][local v]. Failed steps publish zero
    // acceleration and preserve their pre-step accepted state.
    std::vector<float> accelerations;
    std::vector<MRMetalWorldStatusGPU> statuses;
};

struct MetalWorldDiagnostics {
    MetalWorldHostStatus status = MetalWorldHostStatus::success;
    MetalWorldLayout layout{};
    bool dispatched = false;
    bool published = false;
    std::uint32_t successfulStepCount = 0u;
    std::uint32_t failedStepCount = 0u;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstFailingControlStep = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode = MR_STEP_SUCCESS;
    // Metal command-buffer GPU timestamps exclude host validation, encoding,
    // queueing, waits, allocation, and publication. Submission time begins
    // immediately before commit and ends after completion.
    double gpuElapsedMilliseconds = 0.0;
    double submissionElapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string thermalState;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalWorldHostStatus::success;
    }
};

struct MetalWorldContextStats {
    std::uint64_t pipelineCreationCount = 0u;
    std::uint64_t modelUploadCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    bool hasInFlightSubmission = false;
};

class MetalWorldContext;

// A committed multi-step world graph. submit() snapshots all caller-owned
// spans before returning. wait() consumes the ticket and publishes one result.
// Destroying a live ticket waits and discards safely.
class MetalWorldSubmission {
public:
    MetalWorldSubmission() noexcept;
    ~MetalWorldSubmission();

    MetalWorldSubmission(MetalWorldSubmission&& other) noexcept;
    MetalWorldSubmission& operator=(
        MetalWorldSubmission&& other
    ) noexcept;

    MetalWorldSubmission(const MetalWorldSubmission&) = delete;
    MetalWorldSubmission& operator=(
        const MetalWorldSubmission&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] MetalWorldDiagnostics wait(
        MetalWorldResult& result
    );

private:
    friend class MetalWorldContext;
    std::unique_ptr<detail::MetalWorldSubmissionState> state_;
};

// Persistent checked executor. Pipeline creation and immutable-model upload
// are cached; the shared arena grows geometrically and never shrinks. One
// batch may be in flight because submissions reuse this arena.
class MetalWorldContext {
public:
    explicit MetalWorldContext(MetalWorldConfig config = {});
    ~MetalWorldContext();

    MetalWorldContext(MetalWorldContext&& other) noexcept;
    MetalWorldContext& operator=(MetalWorldContext&& other) noexcept;

    MetalWorldContext(const MetalWorldContext&) = delete;
    MetalWorldContext& operator=(const MetalWorldContext&) = delete;

    [[nodiscard]] MetalWorldDiagnostics submit(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldSubmission& submission
    );

    [[nodiscard]] MetalWorldDiagnostics run(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldResult& result
    );

    [[nodiscard]] MetalWorldContextStats stats() const noexcept;

private:
    std::shared_ptr<detail::MetalWorldContextState> state_;
};

[[nodiscard]] const char* metalWorldHostStatusName(
    MetalWorldHostStatus status
) noexcept;

} // namespace metalrobo
