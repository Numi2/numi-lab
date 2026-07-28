#pragma once

#include "metalrobo/EngineModel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
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

// Fixed per-environment capacities used by the device-resident contact graph.
// Zero fields use the compiled recommendation. Explicit stage capacities may
// be smaller to exercise exact transactional overflow; structural capacities
// that cannot address the compiled graph are rejected.
struct MetalWorldCapacityProfile {
    std::uint32_t candidatePairs = 0u;
    std::uint32_t rawContacts = 0u;
    std::uint32_t manifolds = 0u;
    std::uint32_t constraintBlocks = 0u;
    std::uint32_t constraintRows = 0u;
    std::uint32_t islands = 0u;
    std::uint32_t spillRows = 0u;
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
// capability advances exactly one selected articulation plus dynamic,
// kinematic, and static scene bodies. Accessors expose facts without
// permitting fingerprint forgery.
class CompiledWorld {
public:
    CompiledWorld() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const EngineModel& model() const noexcept;
    [[nodiscard]] std::uint32_t articulationIndex() const noexcept;
    [[nodiscard]] std::uint32_t nq() const noexcept;
    [[nodiscard]] std::uint32_t nv() const noexcept;
    [[nodiscard]] std::uint32_t bodyCount() const noexcept;
    [[nodiscard]] std::uint32_t sceneBodyCount() const noexcept;
    [[nodiscard]] std::uint32_t colliderCount() const noexcept;
    [[nodiscard]] std::uint32_t eligiblePairCount() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> sceneBodyIndices()
        const noexcept;
    [[nodiscard]] std::span<const MRCompiledCollisionPairGPU>
    eligiblePairs() const noexcept;
    [[nodiscard]] const MetalWorldCapacityProfile& capacities()
        const noexcept;
    [[nodiscard]] const MetalWorldCapacityProfile&
    minimumCapacities() const noexcept;
    [[nodiscard]] MetalWorldCapacityClass capacityClass()
        const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;

private:
    friend MetalWorldCompileDiagnostics compileMetalWorld(
        const EngineModel&,
        std::uint32_t,
        CompiledWorld&,
        const MetalWorldCapacityProfile&
    );
    friend class MetalWorldContext;

    EngineModel model_;
    std::uint32_t articulationIndex_ = MR_INVALID_INDEX;
    MetalWorldCapacityClass capacityClass_ =
        MetalWorldCapacityClass::uncompiled;
    MetalWorldCapacityProfile capacities_{};
    MetalWorldCapacityProfile minimumCapacities_{};
    std::vector<std::uint32_t> sceneBodyIndices_;
    std::vector<MRCompiledCollisionPairGPU> eligiblePairs_;
    std::uint64_t fingerprint_ = 0u;
};

// Validates and snapshots one canonical articulation transactionally. On
// failure, compiled remains bit-for-bit logically unchanged.
[[nodiscard]] MetalWorldCompileDiagnostics compileMetalWorld(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    CompiledWorld& compiled,
    const MetalWorldCapacityProfile& capacities = {}
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
    // Packed [environment][compiled scene body]. These records provide the
    // initial pose/velocity for every non-articulated dynamic, static, or
    // kinematic body. Articulation-owned body states are generated on-device.
    std::span<const MRBodyStateGPU> initialSceneBodies{};
    // Optional reset state with the same packing as initialSceneBodies.
    std::span<const MRBodyStateGPU> resetSceneBodies{};
    // Optional packed [control step][environment][compiled scene body].
    // Only kinematic records are consumed; empty means velocity-driven
    // kinematics retain their current state.
    std::span<const MRBodyStateGPU> kinematicTargets{};
};

struct MetalWorldStepConfig {
    // Control-period duration. The immutable model gravity is retained and
    // its authored integration timestep is replaced by
    // timestepSeconds / physicsSubsteps for this submission.
    float timestepSeconds = 1.0f / 60.0f;
    std::uint32_t physicsSubsteps = 1u;
    MetalWorldSolverMode solverMode =
        MetalWorldSolverMode::throughputTGS;
    std::uint32_t velocityIterations = 1u;
    std::uint32_t finalVelocityIterations = 1u;
    bool applyBodyDamping = true;
    bool deterministic = true;
    bool warmStart = true;
    bool captureContactEvidence = false;
    float manifoldBreakingSeparation = 0.02f;
    float manifoldBreakingTangential = 0.02f;
    float manifoldMergeDistance = 0.002f;
    float manifoldNormalCosine = 0.95f;
};

struct MetalWorldConfig {
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as fallback.
    std::string metallibPath;
};

struct MetalWorldLayout {
    MRMetalWorldDispatchGPU dispatch{};
    MRABADispatchGPU abaDispatch{};
    MRMetalWorldContactDispatchGPU contactDispatch{};
    std::size_t initialQElements = 0u;
    std::size_t initialVElements = 0u;
    std::size_t initialSceneBodyElements = 0u;
    std::size_t effortElements = 0u;
    std::size_t resetMaskElements = 0u;
    std::size_t resetQElements = 0u;
    std::size_t resetVElements = 0u;
    std::size_t resetSceneBodyElements = 0u;
    std::size_t kinematicTargetElements = 0u;
    std::size_t observationElements = 0u;
    std::size_t accelerationElements = 0u;
    std::size_t statusElements = 0u;
    std::size_t contactStatusElements = 0u;
    std::size_t manifoldHeaderElements = 0u;
    std::size_t manifoldPointElements = 0u;
    std::size_t contactConstraintElements = 0u;
    std::size_t constraintRowElements = 0u;
    std::size_t islandElements = 0u;
    std::size_t totalRequiredBytes = 0u;
};

struct MetalWorldContactEvidence {
    // Final accepted cache, packed [environment][capacity].
    std::vector<MRManifoldHeaderGPU> manifoldHeaders;
    std::vector<MRManifoldPointGPU> manifoldPoints;
    std::vector<std::uint32_t> manifoldCounts;
    std::vector<MRContactConstraintGPU> contacts;
    std::vector<MRContactPointMetaGPU> contactMetadata;
    std::vector<MRConstraintIRBlockGPU> blocks;
    std::vector<MRConstraintIREndpointGPU> endpoints;
    std::vector<MRConstraintIRRowGPU> rows;
    std::vector<MRConstraintIRConeGPU> cones;
    std::vector<MREvaluatedConstraintIRRowGPU> evaluatedRows;
    std::vector<MREvaluatedConstraintIRConeGPU> evaluatedCones;
    std::vector<MRContactIslandGPU> islands;
};

struct MetalWorldStageCounts {
    std::uint32_t candidatePairs = 0u;
    std::uint32_t rawContacts = 0u;
    std::uint32_t manifolds = 0u;
    std::uint32_t constraintBlocks = 0u;
    std::uint32_t constraintRows = 0u;
    std::uint32_t islands = 0u;
    std::uint32_t spillRows = 0u;
};

// Horizon aggregate for one environment. Required counts retain exact
// overflow requirements; highWater records the maximum resident usage.
// firstFailingStableKey is stable within the CompiledWorld fingerprint:
// pair indices occupy the low half and constraint indices set bit 63.
struct MetalWorldStatus {
    std::uint32_t environment = 0u;
    std::uint32_t code = MR_STEP_SUCCESS;
    std::uint32_t successfulControlSteps = 0u;
    std::uint32_t failedControlSteps = 0u;
    std::uint32_t firstFailingControlStep = MR_INVALID_INDEX;
    std::uint32_t firstFailingPair = MR_INVALID_INDEX;
    std::uint32_t firstFailingConstraint = MR_INVALID_INDEX;
    std::uint32_t maximumSolverIterations = 0u;
    std::uint64_t firstFailingStableKey =
        std::numeric_limits<std::uint64_t>::max();
    MetalWorldStageCounts required{};
    MetalWorldStageCounts highWater{};
    float manifoldRetention = 1.0f;
    std::array<float, 4> maximumResiduals{};
};

struct MetalWorldResult {
    MetalWorldLayout layout{};
    // Accepted state after the last encoded control step.
    std::vector<float> finalQ;
    std::vector<float> finalV;
    std::vector<MRBodyStateGPU> finalSceneBodies;
    // Packed [control step][environment][q then v].
    std::vector<float> observations;
    // Packed [control step][environment][local v]. Failed steps publish zero
    // acceleration and preserve their pre-step accepted state.
    std::vector<float> accelerations;
    std::vector<MRMetalWorldStatusGPU> statuses;
    std::vector<MRMetalWorldContactStatusGPU> contactStatuses;
    std::vector<MetalWorldStatus> environmentStatuses;
    MetalWorldContactEvidence contactEvidence;
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
