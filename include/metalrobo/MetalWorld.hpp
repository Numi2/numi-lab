#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorldCapacity.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/SensorProgram.hpp"
#include "metalrobo/TaskProgram.hpp"
#include "metalrobo/parallel_aba_shared.h"
#include "metalrobo/rod_gpu_shared.h"
#include "metalrobo/unified_quality_shared.h"

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
struct MetalWorldContextPool;
struct MetalWorldResidentStateData;
struct MetalWorldSubmissionState;
struct MetalRolloutRingData;
struct MetalRolloutAppendData;
} // namespace detail

enum class MetalWorldExecutionMode : std::uint32_t {
    freeMotionABA = 0u,
    numiSolver = 1u,
};

enum class NumiSolverIterationPolicy : std::uint32_t {
    // Deterministic rollout profile. Work budgets are fixed even when a
    // residual reaches tolerance early.
    fixedBudget = 0u,
    // Uses the same public NumiSolver contract with residual-based Newton and
    // Krylov budgets. The concrete dense/matrix-free backend remains private.
    residualConverged = 1u,
};

struct NumiSolverSettings {
    NumiSolverIterationPolicy iterationPolicy =
        NumiSolverIterationPolicy::fixedBudget;
    // Every authored physics substep is divided into this many integrated and
    // relinearized temporal microsteps in the fixed-budget profile.
    std::uint32_t temporalSubsteps = 4u;
    std::uint32_t rodContactOuterIterations = 2u;
    std::uint32_t maximumNewtonIterations = 20u;
    std::uint32_t maximumPCGIterations = 128u;
    std::uint32_t maximumLineSearchIterations = 16u;
    std::uint32_t directMaximumGeneralizedVelocities = 64u;
    std::uint32_t directMaximumRows = 256u;
    float optimalityTolerance = 1.0e-5f;
    float feasibilityTolerance = 2.0e-5f;
    float armijoConstant = 1.0e-4f;
    float lineSearchContraction = 0.5f;
    float complianceFloorMultiplier = 64.0f;
};

enum class MetalWorldActuationMode : std::uint32_t {
    effort = 0u,
    implicitPositionDrive = 1u,
};

enum class MetalWorldCCDMode : std::uint32_t {
    disabled = MR_WORLD_CCD_DISABLED,
    speculative = MR_WORLD_CCD_SPECULATIVE,
    hybrid = MR_WORLD_CCD_HYBRID,
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

// Immutable validated snapshot consumed by MetalWorldContext. Every
// articulation tree, free body, and connected rod component has one stable
// dynamic-node record. Accessors expose facts without permitting fingerprint
// forgery.
class CompiledWorld {
public:
    CompiledWorld() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const EngineModel& model() const noexcept;
    [[nodiscard]] std::uint32_t articulationIndex() const noexcept;
    [[nodiscard]] std::uint32_t articulationCount() const noexcept;
    [[nodiscard]] std::uint32_t nq() const noexcept;
    [[nodiscard]] std::uint32_t nv() const noexcept;
    [[nodiscard]] std::uint32_t bodyCount() const noexcept;
    [[nodiscard]] std::uint32_t sceneBodyCount() const noexcept;
    [[nodiscard]] std::uint32_t colliderCount() const noexcept;
    [[nodiscard]] std::uint32_t eligiblePairCount() const noexcept;
    [[nodiscard]] std::uint32_t rodCount() const noexcept;
    [[nodiscard]] std::uint32_t rodNodeCount() const noexcept;
    [[nodiscard]] std::uint32_t rodEdgeCount() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> sceneBodyIndices()
        const noexcept;
    [[nodiscard]] std::span<const MRCompiledCollisionPairGPU>
    eligiblePairs() const noexcept;
    [[nodiscard]] std::span<const MRGeometryHeaderGPU>
    geometryHeaders() const noexcept;
    [[nodiscard]] std::span<const mr_float4>
    geometryVertices() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    geometryIndices() const noexcept;
    [[nodiscard]] std::span<const MRConvexFaceGPU>
    convexFaces() const noexcept;
    [[nodiscard]] std::span<const MRConvexHalfEdgeGPU>
    convexHalfEdges() const noexcept;
    [[nodiscard]] std::span<const MRMeshBVHNodeGPU>
    meshBvhNodes() const noexcept;
    [[nodiscard]] std::span<const MRMeshTriangleGPU>
    meshTriangles() const noexcept;
    [[nodiscard]] std::span<const MRRodColliderGPU>
    rodColliders() const noexcept;
    [[nodiscard]] std::span<const MRShapeGPU>
    rodShapeSources() const noexcept;
    [[nodiscard]] std::span<const MRRodToolPairGPU>
    rodToolPairs() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    rodNodeOffsets() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    rodEdgeOffsets() const noexcept;
    [[nodiscard]] std::span<const MRRodNodeStateGPU>
    defaultRodNodes() const noexcept;
    [[nodiscard]] std::span<const MRRodEdgeStateGPU>
    defaultRodEdges() const noexcept;
    [[nodiscard]] std::span<const HeterogeneousRodProgram>
    rodPrograms() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    articulationQOffsets() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    articulationVOffsets() const noexcept;
    [[nodiscard]] std::span<const MRWorldDynamicNodeGPU>
    dynamicNodes() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    bodyDynamicNodes() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    sceneBodyDynamicNodes() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t>
    rodDynamicNodes() const noexcept;
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
    friend MetalWorldCompileDiagnostics compileMetalWorld(
        const HeterogeneousWorld&,
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
    std::vector<MRRodColliderGPU> rodColliders_;
    std::vector<MRShapeGPU> rodShapeSources_;
    std::vector<MRRodToolPairGPU> rodToolPairs_;
    std::vector<std::uint32_t> rodNodeOffsets_;
    std::vector<std::uint32_t> rodEdgeOffsets_;
    std::vector<MRRodNodeStateGPU> defaultRodNodes_;
    std::vector<MRRodEdgeStateGPU> defaultRodEdges_;
    std::vector<HeterogeneousRodProgram> rodPrograms_;
    std::vector<std::uint32_t> articulationQOffsets_;
    std::vector<std::uint32_t> articulationVOffsets_;
    std::vector<MRWorldDynamicNodeGPU> dynamicNodes_;
    std::vector<std::uint32_t> bodyDynamicNodes_;
    std::vector<std::uint32_t> sceneBodyDynamicNodes_;
    std::vector<std::uint32_t> rodDynamicNodes_;
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

[[nodiscard]] MetalWorldCompileDiagnostics compileMetalWorld(
    const HeterogeneousWorld& world,
    CompiledWorld& compiled,
    const MetalWorldCapacityProfile& capacities = {}
);

struct MetalRolloutRingConfig {
    std::uint32_t environmentCount = 0u;
    std::uint32_t controlStepCapacity = 0u;
    std::uint32_t actorObservationCount = 0u;
    std::uint32_t criticObservationCount = 0u;
    std::uint32_t actionCount = 0u;
    std::uint32_t slotCount = 3u;
};

struct MetalRolloutRingLayout : MetalRolloutRingConfig {
    std::size_t retainedBytes = 0u;
};

// Opaque append token produced only by a live rollout lease. It contains no
// public raw Metal bindings and can be copied into an asynchronous world
// batch; the command buffer retains the underlying resources until completion.
class MetalRolloutAppendTarget {
public:
    MetalRolloutAppendTarget() = default;
    [[nodiscard]] bool valid() const noexcept;

private:
    friend class MetalRolloutBufferView;
    friend class MetalWorldContext;
    std::shared_ptr<detail::MetalRolloutAppendData> state_;
};

class MetalRolloutBufferView {
public:
    MetalRolloutBufferView() noexcept;
    ~MetalRolloutBufferView();
    MetalRolloutBufferView(MetalRolloutBufferView&&) noexcept;
    MetalRolloutBufferView& operator=(
        MetalRolloutBufferView&&
    ) noexcept;
    MetalRolloutBufferView(const MetalRolloutBufferView&) = delete;
    MetalRolloutBufferView& operator=(
        const MetalRolloutBufferView&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const MetalRolloutRingLayout& layout() const noexcept;
    [[nodiscard]] std::uint64_t policyRevision() const noexcept;
    [[nodiscard]] std::uint32_t writtenControlSteps() const noexcept;
    [[nodiscard]] MetalRolloutAppendTarget beginAppend(
        std::uint32_t controlStepCount,
        bool includeBootstrapValues
    );
    void cancelPendingAppend() noexcept;
    void seal();
    [[nodiscard]] bool sealed() const noexcept;

    [[nodiscard]] float* actorObservations() const noexcept;
    [[nodiscard]] float* criticObservations() const noexcept;
    [[nodiscard]] float* latents() const noexcept;
    [[nodiscard]] float* logProbabilities() const noexcept;
    [[nodiscard]] float* values() const noexcept;
    [[nodiscard]] float* bootstrapValues() const noexcept;
    [[nodiscard]] MRTaskTransitionGPU* transitions() const noexcept;

private:
    friend class MetalRolloutRing;
    void release() noexcept;
    [[nodiscard]] void* streamData(std::uint32_t stream) const noexcept;
    std::shared_ptr<detail::MetalRolloutRingData> ring_;
    std::uint32_t slotIndex_ = MR_INVALID_INDEX;
    std::uint64_t generation_ = 0u;
};

// Native fixed-capacity shared-buffer ring. The buffers are allocated by the
// runtime on the same default Apple GPU used by MetalWorld; Swift receives
// only opaque leases and compact CPU-addressable views after sealing.
class MetalRolloutRing {
public:
    explicit MetalRolloutRing(MetalRolloutRingConfig config);
    ~MetalRolloutRing();
    MetalRolloutRing(MetalRolloutRing&&) noexcept;
    MetalRolloutRing& operator=(MetalRolloutRing&&) noexcept;
    MetalRolloutRing(const MetalRolloutRing&) = delete;
    MetalRolloutRing& operator=(const MetalRolloutRing&) = delete;

    [[nodiscard]] const MetalRolloutRingLayout& layout() const noexcept;
    [[nodiscard]] MetalRolloutBufferView acquire(
        std::uint64_t policyRevision
    );

private:
    std::shared_ptr<detail::MetalRolloutRingData> state_;
};

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
    // A compiled task program consumes packed
    // [control step][environment][task action] normalized actions and leaves
    // efforts empty; its native control operators produce nv-wide targets.
    std::span<const float> actions{};
    std::uint64_t policyRevision = 0u;
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
    // Packed [environment][compiled rod node/edge]. Empty spans use the
    // immutable heterogeneous-world defaults. Reset streams are optional and
    // transactionally replace both mechanics and all contact caches.
    std::span<const MRRodNodeStateGPU> initialRodNodes{};
    std::span<const MRRodEdgeStateGPU> initialRodEdges{};
    std::span<const MRRodNodeStateGPU> resetRodNodes{};
    std::span<const MRRodEdgeStateGPU> resetRodEdges{};
    // Optional compact rollout destination. Its dimensions, policy revision,
    // cursor, and bootstrap boundary are validated before any GPU work.
    MetalRolloutAppendTarget rolloutTarget{};
};

struct MetalWorldStepConfig {
    // Control-period duration. The immutable model gravity is retained and
    // its authored integration timestep is replaced by
    // timestepSeconds / physicsSubsteps for this submission.
    float timestepSeconds = 1.0f / 60.0f;
    std::uint32_t physicsSubsteps = 1u;
    MetalWorldExecutionMode executionMode =
        MetalWorldExecutionMode::numiSolver;
    NumiSolverSettings numiSolver{};
    // In effort mode, MetalWorldBatch::efforts is generalized effort. In
    // implicitPositionDrive mode it is the desired position per scalar
    // driven DoF; floating-root and unactuated entries are ignored.
    MetalWorldActuationMode actuationMode =
        MetalWorldActuationMode::effort;
    // Empty means a policy-independent physics submission. A valid compiled
    // program owns reset, control, observation, reward, and termination
    // semantics without selecting a robot-specific shader path.
    CompiledTaskProgram taskProgram{};
    // Optional canonical native sensor schedule, independent of whether a
    // TaskIR program is installed. Parent-frame pose, world twist, IMU,
    // solver-authoritative contact-wrench, and contact-state sensors execute
    // with native latency, history, and counter-based corruption on the
    // control timeline. Unsupported domains fail validation instead of
    // falling back to a parallel runtime.
    CompiledSensorProgram sensorProgram{};
    // Optional generic native inference program. With no policy program,
    // normalized actions remain an explicit learner/deployment input.
    CompiledPolicyProgram policyProgram{};
    // Publish V(s_T) from the accepted post-rollout state in the same command
    // buffer. This does not apply the sampled action or advance physics.
    bool evaluateFinalPolicy = false;
    std::uint64_t taskSeed = 0u;
    // Initial task-wide command curriculum restored at a training boundary.
    // It is consumed only when a new resident state is initialized.
    std::uint32_t taskCurriculumLevel = 0u;
    // Scalar position limits are compiled as paired unilateral ConstraintIR
    // candidates and solved in the same island as contact. These defaults
    // intentionally match the FP64 articulated joint-limit reference.
    float jointLimitActivationDistance = 2.0e-3f;
    float jointLimitPositionSlop = 1.0e-8f;
    float jointLimitRecoveryFraction = 0.2f;
    float jointLimitRegularization = 1.0e-10f;
    MetalWorldCCDMode ccdMode = MetalWorldCCDMode::speculative;
    std::uint32_t maxCCDEvents = MR_CCD_DEFAULT_MAX_EVENTS;
    std::uint32_t maxCCDAdvanceSolvePasses =
        MR_CCD_DEFAULT_ADVANCE_SOLVE_PASSES;
    std::uint32_t maxCCDZeroTimeReplays =
        MR_CCD_DEFAULT_ZERO_TIME_REPLAYS;
    std::uint32_t maxConservativeAdvancementIterations = 16u;
    bool applyBodyDamping = true;
    bool deterministic = true;
    bool warmStart = true;
    bool captureContactEvidence = false;
    // Full state/trajectory publication is an explicit inspection boundary.
    // Native rollout sessions disable both and retain simulator state on the
    // device while still publishing checked status records.
    bool publishFinalState = true;
    bool publishStateTrajectory = true;
    // Compact task/policy streams are ordinarily copied into
    // MetalWorldResult for inspection. A leased rollout destination may
    // disable this copy: a GPU reduction validates the complete payload and
    // MetalWorldResult contains status/diagnostic records only.
    bool publishLearningOutputs = true;
    // Explicit inspection boundary for compact latest sensor values and
    // metadata. Persistent histories remain device-owned.
    bool publishSensorOutputs = false;
    float manifoldBreakingSeparation = 0.02f;
    float manifoldBreakingTangential = 0.02f;
    float manifoldMergeDistance = 0.002f;
    float manifoldNormalCosine = 0.95f;
    float ccdMinimumAdvance = 1.0e-5f;
    float ccdTimeTolerance = 1.0e-5f;
    float ccdSimultaneousTolerance = 1.0e-5f;
    float speculativeMarginScale = 1.0f;
    float ccdSpeedEnvelope = 1.0e4f;
};

struct MetalWorldConfig {
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as fallback.
    std::string metallibPath;
    bool preferPrivateHeaps = true;
    std::uint32_t maximumInFlightSubmissions = 3u;
};

struct MetalWorldMemoryPlan {
    std::size_t immutablePrivateBytes = 0u;
    std::size_t persistentStatePrivateBytes = 0u;
    std::size_t transientPrivateBytes = 0u;
    std::size_t sharedBoundaryBytes = 0u;
    std::size_t peakAliasedBytes = 0u;
};

struct MetalWorldLayout {
    MRMetalWorldDispatchGPU dispatch{};
    MRABADispatchGPU abaDispatch{};
    std::vector<MRMultiABADispatchGPU> abaDispatches;
    std::vector<MRArticulatedOperatorDispatchGPU>
        kinematicsDispatches;
    std::vector<MRArticulatedOperatorDispatchGPU>
        factorDispatches;
    MRMetalWorldContactDispatchGPU contactDispatch{};
    MRUnifiedQualityDispatchGPU qualityDispatch{};
    std::size_t initialQElements = 0u;
    std::size_t initialVElements = 0u;
    std::size_t initialSceneBodyElements = 0u;
    std::size_t effortElements = 0u;
    std::size_t actionElements = 0u;
    std::size_t resetMaskElements = 0u;
    std::size_t resetQElements = 0u;
    std::size_t resetVElements = 0u;
    std::size_t resetSceneBodyElements = 0u;
    std::size_t kinematicTargetElements = 0u;
    std::size_t observationElements = 0u;
    std::size_t actorObservationElements = 0u;
    std::size_t criticObservationElements = 0u;
    std::size_t transitionElements = 0u;
    std::size_t sensorStateElements = 0u;
    std::size_t sensorHistoryElements = 0u;
    std::size_t sensorOutputElements = 0u;
    std::size_t sensorMetadataElements = 0u;
    std::size_t actuatorStateElements = 0u;
    std::size_t actuatorCommandHistoryElements = 0u;
    std::uint32_t actuatorCommandHistorySlots = 0u;
    std::size_t policyLatentElements = 0u;
    std::size_t policyLogProbabilityElements = 0u;
    std::size_t policyValueElements = 0u;
    std::size_t accelerationElements = 0u;
    std::size_t statusElements = 0u;
    std::size_t articulationStatusElements = 0u;
    std::size_t contactStatusElements = 0u;
    std::size_t manifoldHeaderElements = 0u;
    std::size_t manifoldPointElements = 0u;
    std::size_t contactConstraintElements = 0u;
    std::size_t constraintRowElements = 0u;
    std::size_t islandElements = 0u;
    std::size_t workQueueHeaderElements = 0u;
    std::size_t pairWorkElements = 0u;
    std::size_t pairRawStagingElements = 0u;
    std::size_t convexCacheElements = 0u;
    std::size_t ccdPairElements = 0u;
    std::size_t ccdEventStateElements = 0u;
    std::size_t ccdImpactClusterElements = 0u;
    std::size_t manifoldScatterElements = 0u;
    std::size_t endpointRuntimeElements = 0u;
    std::size_t rodNodeStateElements = 0u;
    std::size_t rodEdgeStateElements = 0u;
    std::size_t resetRodNodeStateElements = 0u;
    std::size_t resetRodEdgeStateElements = 0u;
    std::size_t dynamicNodeElements = 0u;
    std::size_t islandNodeReferenceElements = 0u;
    std::size_t islandConstraintReferenceElements = 0u;
    std::size_t rodFactorCacheElements = 0u;
    std::size_t operatorVelocityElements = 0u;
    std::size_t rodBendStateElements = 0u;
    std::size_t rodStatusElements = 0u;
    MetalWorldMemoryPlan memoryPlan{};
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
    std::vector<MRConstraintEndpointRuntimeGPU> endpointRuntime;
    std::vector<MRConstraintIRRowGPU> rows;
    std::vector<MRConstraintIRConeGPU> cones;
    std::vector<MREvaluatedConstraintIRRowGPU> evaluatedRows;
    std::vector<MREvaluatedConstraintIRConeGPU> evaluatedCones;
    std::vector<MRContactIslandGPU> islands;
    std::vector<MRIslandNodeRefGPU> islandNodes;
    std::vector<MRIslandConstraintRefGPU> islandConstraints;
};

struct MetalWorldStageCounts {
    std::uint32_t candidatePairs = 0u;
    std::uint32_t rawContacts = 0u;
    std::uint32_t manifolds = 0u;
    std::uint32_t constraintBlocks = 0u;
    std::uint32_t constraintRows = 0u;
    std::uint32_t islands = 0u;
    std::uint32_t hardConvexPairs = 0u;
    std::uint32_t meshTriangleCandidates = 0u;
    std::uint32_t ccdCandidates = 0u;
    std::uint32_t ccdEvents = 0u;
    std::uint32_t endpointRuntimeRecords = 0u;
    std::uint32_t articulationPointQueries = 0u;
    std::uint32_t rodCandidatePairs = 0u;
    std::uint32_t rodRawContacts = 0u;
    std::uint32_t rodManifolds = 0u;
    std::uint32_t rodCCDEvents = 0u;
    std::uint32_t numiGeneralizedVelocities = 0u;
    std::uint32_t numiRows = 0u;
    std::uint32_t numiKrylovVectors = 0u;
    std::uint32_t numiDirectTiles = 0u;
    std::uint32_t dynamicNodes = 0u;
    std::uint32_t islandNodeReferences = 0u;
    std::uint32_t islandConstraintReferences = 0u;
    std::uint32_t rodFactorBlocks = 0u;
    std::uint32_t operatorVelocityElements = 0u;
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
    std::uint32_t hardConvexFallbacks = 0u;
    std::uint32_t unresolvedCCDCount = 0u;
    std::uint32_t maximumCCDAdvanceCount = 0u;
    std::uint32_t maximumClusteredCCDImpacts = 0u;
    std::uint32_t maximumZeroTimeCCDReplays = 0u;
    std::uint32_t maximumWorkerPackets = 0u;
    std::uint32_t maximumNumiNewtonIterations = 0u;
    std::uint32_t maximumNumiPCGIterations = 0u;
    std::uint32_t maximumNumiLineSearchBacktracks = 0u;
    std::uint32_t maximumRodContacts = 0u;
    float maximumUnconsumedCCDTime = 0.0f;
    std::array<float, 4> maximumResiduals{};
    std::array<float, 4> maximumNumiCertificates{};
};

struct MetalWorldResult {
    MetalWorldLayout layout{};
    // Accepted state after the last encoded control step.
    std::vector<float> finalQ;
    std::vector<float> finalV;
    std::vector<MRBodyStateGPU> finalSceneBodies;
    std::vector<MRRodNodeStateGPU> finalRodNodes;
    std::vector<MRRodEdgeStateGPU> finalRodEdges;
    // Packed [control step][environment][q then v].
    std::vector<float> observations;
    // Compact learning boundary produced only by a native task graph.
    std::vector<float> actorObservations;
    std::vector<float> criticObservations;
    std::vector<MRTaskTransitionGPU> transitions;
    std::vector<float> sensorOutputs;
    std::vector<MRSensorSampleMetadataGPU> sensorMetadata;
    std::vector<float> policyLatents;
    std::vector<float> policyLogProbabilities;
    std::vector<float> policyValues;
    // Packed [control step][environment][local v]. Failed steps publish zero
    // acceleration and preserve their pre-step accepted state.
    std::vector<float> accelerations;
    std::vector<MRMetalWorldStatusGPU> statuses;
    std::vector<MRMetalWorldContactStatusGPU> contactStatuses;
    // One record per environment when the residual-converged NumiSolver
    // policy is selected.
    std::vector<MRUnifiedQualityStatusGPU> numiConvergenceStatuses;
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
    std::uint64_t policyBankUploadCount = 0u;
    std::uint64_t policyBankReuseCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    MetalWorldMemoryPlan memoryPlan{};
    std::uint32_t queriedThreadExecutionWidth = 0u;
    bool usingPrivateHeaps = false;
    bool hasInFlightSubmission = false;
};

class MetalWorldContext;

// Move-only ownership token for one device-resident world state. The token
// pins one context arena between submissions so q/v, scene, contact caches,
// rods, and warm starts can continue without a host round-trip.
class MetalWorldResidentState {
public:
    MetalWorldResidentState() noexcept;
    ~MetalWorldResidentState();

    MetalWorldResidentState(
        MetalWorldResidentState&& other
    ) noexcept;
    MetalWorldResidentState& operator=(
        MetalWorldResidentState&& other
    ) noexcept;

    MetalWorldResidentState(
        const MetalWorldResidentState&
    ) = delete;
    MetalWorldResidentState& operator=(
        const MetalWorldResidentState&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;

private:
    friend class MetalWorldContext;
    std::shared_ptr<detail::MetalWorldResidentStateData> state_;
};

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
// are cached; each private placement-heap slot grows geometrically and never
// shrinks. Slots are reserved without waiting, so up to the configured ring
// size can execute asynchronously.
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

    // Seeds a new resident state from the full host-authored batch. The batch
    // also establishes the reset-state contract retained by the session.
    [[nodiscard]] MetalWorldDiagnostics initializeResidentState(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldResidentState& state,
        MetalWorldSubmission& submission
    );

    // Continues an initialized resident state. initialQ/initialV, initial
    // scene/rod state, and reset state spans must be empty; only compact
    // control/reset-mask/kinematic streams cross the host boundary.
    [[nodiscard]] MetalWorldDiagnostics submitResident(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldResidentState& state,
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
    [[nodiscard]] MetalWorldDiagnostics submitImpl(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldResidentState* residentState,
        bool initializeResidentState,
        MetalWorldSubmission& submission
    );

    std::shared_ptr<detail::MetalWorldContextPool> pool_;
};

[[nodiscard]] const char* metalWorldHostStatusName(
    MetalWorldHostStatus status
) noexcept;

} // namespace metalrobo
