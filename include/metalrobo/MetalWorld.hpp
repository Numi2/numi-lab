#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorldCapacity.hpp"
#include "metalrobo/ParallelABASchedule.hpp"
#include "metalrobo/flapping_wing_types.h"
#include "metalrobo/multicopter_types.h"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/TaskProgram.hpp"
#include "metalrobo/parallel_aba_shared.h"
#include "metalrobo/rod_gpu_shared.h"
#include "metalrobo/millard_muscle_gpu.h"
#include "metalrobo/unified_quality_shared.h"

#include <array>
#include <cmath>
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
} // namespace detail

enum class MetalWorldSolverMode : std::uint32_t {
    // The first production graph composes generic articulated free motion,
    // resets, transactional substep commits, and state observations. Contact
    // modes are reserved now so callers cannot confuse this with a completed
    // collision/contact world.
    freeMotionABA = 0u,
    temporalCone = 2u,
    qualityNewton = 3u,
};

struct MetalWorldQualityConfig {
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
    [[nodiscard]] const ParallelABASchedule& parallelABASchedule()
        const noexcept;
    [[nodiscard]] const MetalWorldCapacityProfile& capacities()
        const noexcept;
    [[nodiscard]] const MetalWorldCapacityProfile&
    minimumCapacities() const noexcept;
    [[nodiscard]] MetalWorldCapacityClass capacityClass()
        const noexcept;
    [[nodiscard]] std::uint64_t modelFingerprint() const noexcept;
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
    ParallelABASchedule parallelABASchedule_;
    std::uint64_t modelFingerprint_ = 0u;
    std::uint64_t fingerprint_ = 0u;
};

// Canonical mechanics fingerprint independent of solver capacities, scene
// scheduling, or task configuration. Returns zero for an invalid model.
[[nodiscard]] std::uint64_t engineModelFingerprint(
    const EngineModel& model
) noexcept;

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
    // Optional source Millard controls, packed
    // [control step][environment][source muscle]. Entries are normalized
    // excitations in [0, 1], not generalized efforts. They require a valid
    // source Millard program and explicit activation time constants below.
    std::span<const float> millardExcitations{};
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
};

// Device-resident physics extension encoded inside every rigid-world
// microstep after the global body-wrench arena is cleared and articulated/
// scene body state is projected, but before ABA and scene prediction consume
// those wrenches.
// The callback borrows every object and may encode work only; it must not
// commit, wait, retain, or replace the command buffer or its buffers.
enum class MetalWorldDevicePhysicsPhase : std::uint32_t {
    preDynamics = 0u,
    postCommit = 1u,
};

enum MetalWorldDevicePhysicsFlags : std::uint32_t {
    // The pre-dynamics pass contributes forces/torques to MetalWorld's
    // global MRABABodyWrenchGPU arena. This capability is independent of the
    // rigid contact solver: a continuum-only contact world may drive ABA or
    // free scene bodies while MetalWorld remains in free-motion mode.
    MetalWorldDevicePhysicsWritesBodyWrenches = 1u << 0u,
    // The post-commit pass consumes accepted rigid contact constraints. This
    // is required for adaptive rigid->continuum promotion, but should not
    // force every device-physics program through the rigid contact pipeline.
    MetalWorldDevicePhysicsRequiresRigidContactEvidence = 1u << 1u,
    // The extension owns a primal rigid/articulated candidate inside its
    // nonlinear solve. MetalWorld still owns generalized coordinates, mass
    // operators, ABA, and integration; the extension may only use the
    // borrowed coupled-candidate service below.
    MetalWorldDevicePhysicsOwnsCoupledCandidate = 1u << 2u,
    // The extension reads the active MRRodNodeStateGPU arena and may add an
    // accepted contact impulse to node velocities before MetalWorld's DER
    // substep. The immutable inverse-mass arena is borrowed alongside it.
    // Publication and rollback remain owned by MetalWorld's rod transaction.
    MetalWorldDevicePhysicsCouplesRodNodes = 1u << 3u,
};

inline constexpr std::uint32_t kMetalWorldDevicePhysicsKnownFlags =
    MetalWorldDevicePhysicsWritesBodyWrenches |
    MetalWorldDevicePhysicsRequiresRigidContactEvidence |
    MetalWorldDevicePhysicsOwnsCoupledCandidate |
    MetalWorldDevicePhysicsCouplesRodNodes;

struct MetalWorldDevicePhysicsPass;

enum class MetalWorldCoupledCandidateOperation : std::uint32_t {
    // Integrate q(q0, v0 + dv) over the borrowed substep and materialize the
    // corresponding articulated body states without publishing them.
    candidateKinematics = 0u,
    // output = M(q0) input. The block-diagonal articulated mass action is
    // evaluated by MetalWorld at its current generalized coordinates.
    massAction = 1u,
    // output = M(q0)^-1 input using MetalWorld's same-timeline ABA operator.
    inverseMassPreconditioner = 2u,
    // Publish an accepted dv as M(q0) dv / dt into MetalWorld's generalized
    // effort stream. ABA and generalized-coordinate integration remain owned
    // by MetalWorld and source q/v are never modified by the extension.
    publishCandidate = 3u,
};

// Borrowed primal articulated operator. Buffers are environment-major and
// use the global q/v/body strides supplied by the enclosing pass. `input` and
// `output` hold float generalized vectors; candidateQ holds float generalized
// coordinates; candidateBodies holds MRBodyStateGPU. `statuses` is one
// MRInverseMassStatusGPU per [articulation][environment] for inverse mass and
// may be null for the other operations. Output buffers are overwritten.
struct MetalWorldCoupledCandidateQuery {
    void* input = nullptr;
    void* output = nullptr;
    void* candidateQ = nullptr;
    void* candidateBodies = nullptr;
    void* statuses = nullptr;
    void* pointQueries = nullptr;
    void* pointJacobians = nullptr;
    MetalWorldCoupledCandidateOperation operation =
        MetalWorldCoupledCandidateOperation::candidateKinematics;
    std::uint32_t generalizedVectorStride = 0u;
    std::uint32_t candidateQStride = 0u;
    std::uint32_t candidateBodyStride = 0u;
    std::uint32_t statusStride = 0u;
    std::uint32_t pointCount = 0u;
    std::uint32_t pointStride = 0u;
    std::uint32_t pointJacobianStride = 0u;
};

using MetalWorldEncodeCoupledCandidate = bool (*)(
    void* context,
    const MetalWorldDevicePhysicsPass& pass,
    const MetalWorldCoupledCandidateQuery& query
);

struct MetalWorldDevicePhysicsPass {
    void* commandBuffer = nullptr;
    void* q = nullptr;
    void* v = nullptr;
    void* sceneBodies = nullptr;
    void* currentBodies = nullptr;
    void* bodyWrenches = nullptr;
    void* resetMasks = nullptr;
    void* environmentStatuses = nullptr;
    // Final per-environment contact state for this rigid substep.  Device
    // physics may read this only during postCommit, after MetalWorld has
    // accepted the contact solve; it remains borrowed with the command
    // buffer and is never retained by the extension.
    void* contactConstraints = nullptr;
    void* contactStatuses = nullptr;
    // Active environment-major DER node state and immutable world-local node
    // inverse masses. These are non-null only when the compiled world owns a
    // rod. A coupling extension may update velocity.xyz during preDynamics;
    // position and all postCommit state are read-only.
    void* rodNodes = nullptr;
    void* rodInverseMasses = nullptr;
    void* coupledCandidateContext = nullptr;
    MetalWorldEncodeCoupledCandidate encodeCoupledCandidate = nullptr;
    std::uint64_t seed = 0u;
    MetalWorldDevicePhysicsPhase phase =
        MetalWorldDevicePhysicsPhase::preDynamics;
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t physicsSubsteps = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t articulationCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t sceneBodyCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t bodyStateStride = 0u;
    std::uint32_t sceneBodyStride = 0u;
    std::uint32_t bodyWrenchStride = 0u;
    std::uint32_t contactConstraintStride = 0u;
    std::uint32_t rodNodeCount = 0u;
    std::uint32_t rodNodeStride = 0u;
    std::uint32_t articulationRootBody = 0u;
    std::uint32_t qStride = 0u;
    std::uint32_t articulatedInverseMassFlags = 0u;
    std::uint32_t resetMaskStepStride = 0u;
    float timestepSeconds = 0.0f;
};

using MetalWorldDevicePhysicsEncode = bool (*)(
    void* context,
    const MetalWorldDevicePhysicsPass& pass
);

// Called only when MetalWorld abandons a command buffer after a device-physics
// pass has been encoded but before submission. It releases runtime-side
// transaction ownership; it must not commit, wait, or touch borrowed buffers.
using MetalWorldDevicePhysicsAbort = void (*)(
    void* context,
    void* commandBuffer
);

struct MetalWorldDevicePhysicsProgram {
    void* context = nullptr;
    MetalWorldDevicePhysicsEncode encode = nullptr;
    MetalWorldDevicePhysicsAbort abort = nullptr;
    std::uint64_t fingerprint = 0u;
    std::uint32_t flags = 0u;
    // Maximum point-query count used by the borrowed coupled-candidate
    // service. Declaring this capacity keeps MetalWorld's private operator
    // scratch exact instead of reserving the ABI-wide maximum per environment.
    std::uint32_t coupledCandidatePointCapacity = 0u;

    [[nodiscard]] bool valid() const noexcept {
        const bool ownsCoupledCandidate =
            (flags & MetalWorldDevicePhysicsOwnsCoupledCandidate) != 0u;
        return context != nullptr && encode != nullptr && abort != nullptr &&
            fingerprint != 0u &&
            (flags & ~kMetalWorldDevicePhysicsKnownFlags) == 0u &&
            coupledCandidatePointCapacity <=
                MR_ARTICULATED_OPERATOR_MAX_POINTS &&
            (ownsCoupledCandidate
                 ? coupledCandidatePointCapacity != 0u
                 : coupledCandidatePointCapacity == 0u);
    }

    [[nodiscard]] bool configured() const noexcept {
        return context != nullptr || encode != nullptr || abort != nullptr ||
            fingerprint != 0u || flags != 0u ||
            coupledCandidatePointCapacity != 0u;
    }
};

// Device-resident observation extension encoded after the generic SensorPack
// has built proprioception and immediately before policy inference. Buffers
// are borrowed id<MTLBuffer> values and commandBuffer is a borrowed
// id<MTLCommandBuffer>; the callback may encode work but must not commit,
// wait, or retain them. This is the renderer/perception composition boundary.
struct MetalWorldDeviceObservationPass {
    void* commandBuffer = nullptr;
    void* q = nullptr;
    void* v = nullptr;
    void* sceneBodies = nullptr;
    void* currentBodies = nullptr;
    void* resetMasks = nullptr;
    void* taskStates = nullptr;
    void* actorHistory = nullptr;
    void* actorObservations = nullptr;
    std::size_t actorObservationOffsetElements = 0u;
    std::uint64_t seed = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t sceneBodyCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t actorFrameSize = 0u;
    std::uint32_t actorHistoryLength = 0u;
    std::uint32_t actorObservationSize = 0u;
};

using MetalWorldDeviceObservationEncode = bool (*)(
    void* context,
    const MetalWorldDeviceObservationPass& pass
);

struct MetalWorldDeviceObservationProgram {
    void* context = nullptr;
    MetalWorldDeviceObservationEncode encode = nullptr;

    [[nodiscard]] bool valid() const noexcept {
        return context != nullptr && encode != nullptr;
    }

    [[nodiscard]] bool configured() const noexcept {
        return context != nullptr || encode != nullptr;
    }
};

// Presentation-only extension encoded after the final accepted state of a
// submitted rollout. Like device observations, it borrows every resource and
// must neither commit nor wait. Unlike device observations, it has no actor
// observation or policy dependency: it is deliberately outside the physics
// and policy contracts.
struct MetalWorldInspectionPass {
    void* commandBuffer = nullptr;
    void* currentBodies = nullptr;
    std::uint64_t seed = 0u;
    std::uint64_t submissionIndex = 0u;
    std::uint32_t controlStepCount = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t bodyCount = 0u;
};

using MetalWorldInspectionEncode = bool (*) (
    void* context,
    const MetalWorldInspectionPass& pass
);

struct MetalWorldInspectionProgram {
    void* context = nullptr;
    MetalWorldInspectionEncode encode = nullptr;

    [[nodiscard]] bool valid() const noexcept {
        return context != nullptr && encode != nullptr;
    }

    [[nodiscard]] bool configured() const noexcept {
        return context != nullptr || encode != nullptr;
    }
};

// Immutable robot-authored multicopter actuator program. MetalWorld executes
// it immediately before every ABA microstep and writes the resulting
// world-frame wrench into the same external-wrench arena used by articulated
// dynamics. Gravity, collision, contact, and reset therefore remain the
// ordinary universal physics path.
struct MetalWorldMulticopterProgram {
    MRMulticopterModelGPU model{};
    std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS> rotors{};
    MRMulticopterMixerGPU mixer{};
    std::uint32_t articulationIndex = MR_INVALID_INDEX;
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t firstAction = MR_INVALID_INDEX;
    mr_float4 windVelocity{};

    [[nodiscard]] bool valid() const noexcept {
        return model.rotorCount != 0u &&
            model.rotorCount <= MR_MULTICOPTER_MAX_ROTORS &&
            articulationIndex != MR_INVALID_INDEX &&
            bodyIndex != MR_INVALID_INDEX &&
            firstAction != MR_INVALID_INDEX;
    }
};

// Immutable source-materialized Millard program executed inside every
// MetalWorld FunctionBased microstep.  `states` is one source state per
// muscle; MetalWorld expands it across environments privately, evaluates the
// path/Jacobian force on device, and adds the reduced generalized effort to
// the resident effort arena before source dynamics.  Activation control is
// intentionally not inferred from generic effort actions: callers must
// provide an explicit muscle-state program when that control contract is
// authored.
struct MetalWorldMillardProgram {
    std::uint32_t articulationIndex = MR_INVALID_INDEX;
    std::span<const MRArticulatedPointImpulseGPU> pointQueries{};
    std::span<const MRMillardMuscleGPU> muscles{};
    std::span<const MRMillardMuscleStateGPU> states{};
    std::span<const MRMillardPathPointGPU> pathPoints{};
    std::span<const MRMillardSourceCurveGPU> curves{};
    std::span<const MRMillardCylinderWrapGPU> cylinderWraps{};

    [[nodiscard]] bool configured() const noexcept {
        return articulationIndex != MR_INVALID_INDEX ||
            !pointQueries.empty() || !muscles.empty() || !states.empty() ||
            !pathPoints.empty() || !curves.empty() ||
            !cylinderWraps.empty();
    }

    [[nodiscard]] bool valid() const noexcept {
        return articulationIndex != MR_INVALID_INDEX &&
            !pointQueries.empty() && !muscles.empty() &&
            states.size() == muscles.size() &&
            curves.size() == muscles.size() && !pathPoints.empty();
    }
};

// Caller-owned temporal contract for per-control source Millard excitation.
// The source model does not provide a universally valid activation default;
// importers must materialize/provenance both positive time constants before
// submitting controls. An unconfigured record preserves the static-source
// state behavior used by existing reference executions.
struct MetalWorldMillardActivationDynamics {
    float activationTimeConstantSeconds = 0.0f;
    float deactivationTimeConstantSeconds = 0.0f;

    [[nodiscard]] bool configured() const noexcept {
        return activationTimeConstantSeconds != 0.0f ||
            deactivationTimeConstantSeconds != 0.0f;
    }

    [[nodiscard]] bool valid() const noexcept {
        return std::isfinite(activationTimeConstantSeconds) &&
            std::isfinite(deactivationTimeConstantSeconds) &&
            activationTimeConstantSeconds > 0.0f &&
            deactivationTimeConstantSeconds > 0.0f;
    }
};

// Immutable bilateral articulated-wing load program. Policies actuate the
// authored hinge drives; this program contributes air-relative body loads
// from accepted articulated state during every physics microstep.
struct MetalWorldFlappingWingProgram {
    std::array<MRFlappingWingGPU, 2u> wings{};
    MRAeroTailGPU tail{};
    MRAeroFuselageGPU fuselage{};
    std::uint32_t articulationIndex = MR_INVALID_INDEX;
    std::uint32_t rootBodyIndex = MR_INVALID_INDEX;
    mr_float4 windVelocityAndDensity{};

    [[nodiscard]] bool valid() const noexcept {
        return articulationIndex != MR_INVALID_INDEX &&
            rootBodyIndex != MR_INVALID_INDEX &&
            windVelocityAndDensity.w > 0.0f &&
            wings[0].bodyIndex != MR_INVALID_INDEX &&
            wings[1].bodyIndex != MR_INVALID_INDEX &&
            tail.bodyIndex != MR_INVALID_INDEX &&
            tail.rootBodyIndex == rootBodyIndex &&
            tail.rootToCenterAndArea.w > 0.0f &&
            tail.chordAndCoefficients.x > 0.0f &&
            fuselage.bodyIndex == rootBodyIndex &&
            fuselage.rootBodyIndex == rootBodyIndex &&
            fuselage.referenceAreasAndDrag.x > 0.0f &&
            fuselage.referenceAreasAndDrag.y > 0.0f &&
            fuselage.referenceAreasAndDrag.z > 0.0f &&
            fuselage.referenceAreasAndDrag.w >= 0.0f;
    }
};

struct MetalWorldStepConfig {
    // Control-period duration. The immutable model gravity is retained and
    // its authored integration timestep is replaced by
    // timestepSeconds / physicsSubsteps for this submission.
    float timestepSeconds = 1.0f / 60.0f;
    std::uint32_t physicsSubsteps = 1u;
    MetalWorldSolverMode solverMode =
        MetalWorldSolverMode::temporalCone;
    // In effort mode, MetalWorldBatch::efforts is requested generalized
    // actuator effort. Every entry is clipped by its immutable effort and
    // torque-speed envelope; floating-root and unactuated entries resolve to
    // zero. External loads belong in the body-wrench path. In
    // implicitPositionDrive mode, efforts holds the desired position per
    // scalar driven DoF; floating-root and unactuated entries are ignored.
    MetalWorldActuationMode actuationMode =
        MetalWorldActuationMode::effort;
    // Empty means a policy-independent physics submission. A valid compiled
    // program owns reset, control, observation, reward, and termination
    // semantics without selecting a robot-specific shader path.
    CompiledTaskProgram taskProgram{};
    // Optional generic native inference program. With no policy program,
    // normalized actions remain an explicit learner/deployment input.
    CompiledPolicyProgram policyProgram{};
    // Optional immutable carrier actor used by staged Crow training. The
    // primary policy remains the stochastic behavior policy.
    CompiledPolicyProgram basePolicyProgram{};
    // Optional compiled robot actuator program. This is an execution program,
    // not a constructor hint; its complete contents participate in the run
    // fingerprint before reaching MetalWorld.
    MetalWorldMulticopterProgram multicopterProgram{};
    MetalWorldFlappingWingProgram flappingWingProgram{};
    // Optional source Millard muscle-tendon program. It is admitted only by
    // the bounded FunctionBased direct-effort path for a fixed or mobile
    // root: free motion
    // or the streamed temporal-cone contact response. A native task may join
    // only through its complete ordered `millardExcitation` action surface;
    // generic task body/controller parameterization and anatomical collider
    // admission remain separate gates.
    MetalWorldMillardProgram millardProgram{};
    MetalWorldMillardActivationDynamics millardActivationDynamics{};
    // Optional multiphysics pass. It executes before rigid dynamics and again
    // after transactional publication in every physics substep. The pre pass
    // contributes to the shared global body-wrench arena; the post pass may
    // synchronize accepted state and perform representation transfer.
    MetalWorldDevicePhysicsProgram devicePhysicsProgram{};
    // Optional renderer/perception pass. It receives only borrowed device
    // resources and executes inside the native rollout command buffer.
    MetalWorldDeviceObservationProgram deviceObservationProgram{};
    // Optional non-authoritative presentation pass. It runs once after a
    // completed rollout chunk has produced its final accepted body state.
    MetalWorldInspectionProgram inspectionProgram{};
    // Publish V(s_T) from the accepted post-rollout state in the same command
    // buffer. This does not apply the sampled action or advance physics.
    bool evaluateFinalPolicy = false;
    // Invocation-scoped assisted labels for Crow journey training. Neither
    // field participates in autonomous PolicyPack execution.
    bool birdFlowJourneyTeacher = false;
    float birdFlowJourneyStudentAuthority = 0.0f;
    // Invocation-scoped reset curriculum for the V10 navigation route. This
    // changes only the initial sequential waypoint stage; it grants neither
    // teacher nor actuator authority and is disabled during evaluation.
    // 0 disables, 1 selects the mixed reset distribution, and 2...6 focus
    // navigation stages 0...4 respectively.
    std::uint32_t birdFlowNavigationCurriculumMode = 0u;
    // Invocation-only paired regression fixture. It pins the authored Crow
    // course bodies while preserving root, sensor, controller, and physical
    // randomization. It is not a qualification mode.
    bool birdFlowNavigationDevelopmentReference = false;
    std::uint64_t taskSeed = 0u;
    // Invocation-scoped reset sampling. These select an overlapping region of
    // one compiled TaskPack; they never alter reward, success, or promotion.
    std::uint32_t minimumDifficultyBand = 0u;
    std::uint32_t maximumDifficultyBand = MR_INVALID_INDEX;
    float difficultySamplingExponentOverride = 0.0f;
    std::uint32_t velocityIterations = 1u;
    std::uint32_t finalVelocityIterations = 1u;
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
    // Uses inverse ABA instead of a dense articulated factor for eligible
    // single-articulation contact graphs.
    bool matrixFreeArticulatedContact = true;
    // Streams inverse-ABA response columns directly into the contact solve.
    bool streamedArticulatedContactResponses = true;
    bool captureContactEvidence = false;
    // Full state/trajectory publication is an explicit inspection boundary.
    // Native rollout sessions disable both and retain simulator state on the
    // device while still publishing checked status records.
    bool publishFinalState = true;
    bool publishStateTrajectory = true;
    float manifoldBreakingSeparation = 0.02f;
    float manifoldBreakingTangential = 0.02f;
    float manifoldMergeDistance = 0.002f;
    float manifoldNormalCosine = 0.95f;
    float ccdMinimumAdvance = 1.0e-5f;
    float ccdTimeTolerance = 1.0e-5f;
    float ccdSimultaneousTolerance = 1.0e-5f;
    float speculativeMarginScale = 1.0f;
    float ccdSpeedEnvelope = 1.0e4f;
    std::uint32_t rodContactOuterIterations = 2u;
    MetalWorldQualityConfig quality{};
};

struct MetalWorldConfig {
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as fallback.
    std::string metallibPath;
    bool preferPrivateHeaps = true;
    // Production uses schedule-driven SIMD32 ABA for branching frontiers and
    // keeps narrow serial chains on the lower-overhead ordered kernel. False
    // forces the serial oracle for paired numerical/performance replay.
    bool preferParallelABA = true;
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
    std::uint64_t devicePhysicsFingerprint = 0u;
    std::uint64_t millardProgramFingerprint = 0u;
    std::uint32_t millardMuscleCount = 0u;
    bool usesParallelABA = false;
    std::uint32_t parallelABAMaximumLevelWidth = 0u;
    MRABADispatchGPU abaDispatch{};
    std::vector<MRMultiABADispatchGPU> abaDispatches;
    std::vector<MRArticulatedOperatorDispatchGPU>
        kinematicsDispatches;
    std::vector<MRArticulatedOperatorDispatchGPU>
        factorDispatches;
    MRInverseMassDispatchGPU inverseMassDispatch{};
    MRMetalWorldContactDispatchGPU contactDispatch{};
    MRUnifiedQualityDispatchGPU qualityDispatch{};
    std::size_t initialQElements = 0u;
    std::size_t initialVElements = 0u;
    std::size_t initialSceneBodyElements = 0u;
    std::size_t effortElements = 0u;
    std::size_t actionElements = 0u;
    std::size_t millardExcitationElements = 0u;
    std::size_t resetMaskElements = 0u;
    std::size_t resetQElements = 0u;
    std::size_t resetVElements = 0u;
    std::size_t resetSceneBodyElements = 0u;
    std::size_t kinematicTargetElements = 0u;
    std::size_t observationElements = 0u;
    std::size_t actorObservationElements = 0u;
    std::size_t criticObservationElements = 0u;
    std::size_t transitionElements = 0u;
    std::size_t motionFeatureElements = 0u;
    std::size_t teacherActionElements = 0u;
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
    std::size_t islandWorkElements = 0u;
    std::size_t contactTileElements = 0u;
    std::size_t convexCacheElements = 0u;
    std::size_t ccdPairElements = 0u;
    std::size_t ccdEventStateElements = 0u;
    std::size_t ccdImpactClusterElements = 0u;
    std::size_t waveWorkPacketElements = 0u;
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
    std::vector<MRIslandWorkGPU> islandWork;
    std::vector<MRContactTileGPU> contactTiles;
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
    std::uint32_t solverTiles = 0u;
    std::uint32_t spillRows = 0u;
    std::uint32_t ccdCandidates = 0u;
    std::uint32_t ccdEvents = 0u;
    std::uint32_t endpointRuntimeRecords = 0u;
    std::uint32_t articulationPointQueries = 0u;
    std::uint32_t rodCandidatePairs = 0u;
    std::uint32_t rodRawContacts = 0u;
    std::uint32_t rodManifolds = 0u;
    std::uint32_t rodCCDEvents = 0u;
    std::uint32_t qualityGeneralizedVelocities = 0u;
    std::uint32_t qualityRows = 0u;
    std::uint32_t qualityKrylovVectors = 0u;
    std::uint32_t qualityDirectTiles = 0u;
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
    std::uint32_t maximumQualityNewtonIterations = 0u;
    std::uint32_t maximumQualityPCGIterations = 0u;
    std::uint32_t maximumQualityLineSearchBacktracks = 0u;
    std::uint32_t maximumRodContacts = 0u;
    float maximumUnconsumedCCDTime = 0.0f;
    std::array<float, 4> maximumResiduals{};
    std::array<float, 4> maximumQualityCertificates{};
};

struct MetalWorldResult {
    MetalWorldLayout layout{};
    // Accepted state after the last encoded control step.
    std::vector<float> finalQ;
    std::vector<float> finalV;
    std::vector<MRBodyStateGPU> finalSceneBodies;
    std::vector<MRRodNodeStateGPU> finalRodNodes;
    std::vector<MRRodEdgeStateGPU> finalRodEdges;
    // Final per-environment/per-rod convergence certificate. Unlike the
    // accepted node state, this remains available even when final-state
    // publication is disabled so a caller can qualify the implicit solve.
    std::vector<MRRodGPUStatus> rodStatuses;
    // Packed [control step][environment][q then v].
    std::vector<float> observations;
    // Compact learning boundary produced only by a native task graph.
    std::vector<float> actorObservations;
    std::vector<float> criticObservations;
    std::vector<MRTaskTransitionGPU> transitions;
    // Task-wide state after the final encoded evidence reduction. Reading this
    // fixed record does not publish per-environment simulator state or add
    // work inside the physics/task hot loops.
    MRTaskEvidenceStateGPU evidenceState{};
    // Training-only anchor-relative tracked-link poses, packed
    // [control step][environment][feature].
    std::vector<float> motionFeatures;
    // Canonical normalized actions relative to the deployment task's default
    // pose. Populated when generated imagination is physically executed.
    std::vector<float> teacherActions;
    // Canonical action selected by the device-resident deployment policy.
    // Kept separate from teacherActions so policy evaluation and training
    // evidence never silently substitute one authority for the other.
    std::vector<float> policyActions;
    std::vector<float> policyLatents;
    std::vector<float> policyLogProbabilities;
    std::vector<float> policyValues;
    // Packed [control step][environment][local v]. Failed steps publish zero
    // acceleration and preserve their pre-step accepted state.
    std::vector<float> accelerations;
    // Final source-muscle evaluation in this command-buffer submission.
    // Individual generalized forces remain unpacked [environment][muscle][v]
    // so an audit can establish that the active source elements, rather than
    // a host-restaged aggregate, drove the accepted state.
    std::vector<MRMillardMuscleResultGPU> millardResults;
    std::vector<float> millardGeneralizedForces;
    // Final private source-muscle state, packed [environment][muscle]. This
    // lets an activation-control audit verify the exact device-updated state
    // that produced the final force records without publishing a host update.
    std::vector<MRMillardMuscleStateGPU> millardStates;
    std::vector<MRMetalWorldStatusGPU> statuses;
    std::vector<MRMetalWorldContactStatusGPU> contactStatuses;
    // One record per environment when qualityNewton is selected.
    std::vector<MRUnifiedQualityStatusGPU> qualityStatuses;
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
