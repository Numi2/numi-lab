#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/millard_muscle_gpu.h"
#include "metalrobo/mujoco_muscle_gpu.h"

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

[[nodiscard]] constexpr std::size_t
articulatedOperatorThreadgroupBytes(
    const std::size_t bodyCount,
    const std::size_t dofCount,
    const bool includeDenseDynamics = true
) noexcept {
    const auto aligned16 = [](const std::size_t value) {
        return (value + 15u) & ~std::size_t{15u};
    };
    std::size_t bytes = 0u;
    const auto append = [&bytes, &aligned16](
        const std::size_t value
    ) {
        bytes = aligned16(bytes);
        bytes += value;
    };
    // float3 occupies a 16-byte slot in Metal threadgroup memory.
    append(16u * bodyCount); // body position
    append(16u * bodyCount); // body rotation
    append(16u * bodyCount); // joint position
    append(16u * bodyCount); // joint axis
    append(sizeof(std::uint32_t) * bodyCount); // inbound joint
    append(sizeof(std::uint32_t) * bodyCount); // parent body
    append(sizeof(std::uint8_t) * bodyCount); // topology-known flags
    if (includeDenseDynamics) {
        append(sizeof(float) * dofCount * dofCount); // dense factor
        append(sizeof(float) * dofCount); // right-hand side
        append(sizeof(float) * dofCount); // forward solve
        append(sizeof(float) * dofCount); // solution
    }
    return aligned16(bytes);
}
} // namespace detail

// Immutable source program and mutable state sidecar for the Millard
// reference pass. It is optional: all spans empty preserves the articulated
// operator's historical kinematics/mass behavior. When present, the muscle
// pass executes after the FunctionBased kinematics/Jacobian kernel in the
// same command buffer and receives no CPU-restaged pose or Jacobian data.
struct MetalMillardReferenceInput {
    std::span<const MRMillardMuscleGPU> muscles{};
    std::span<const MRMillardMuscleStateGPU> states{};
    std::span<const MRMillardPathPointGPU> pathPoints{};
    std::span<const MRMillardSourceCurveGPU> curves{};
    std::span<const MRMillardCylinderWrapGPU> cylinderWraps{};

    [[nodiscard]] bool enabled() const noexcept {
        return !muscles.empty();
    }
};

// Immutable MyoSim source program and environment-major activation state for
// the MuJoCo general-muscle device reference. Its path kernel reads the
// articulated pose stream produced immediately before it in the same command
// buffer; no CPU-restaged geometry is admitted.
struct MetalMujocoMuscleReferenceInput {
    std::span<const MRMujocoMuscleGPU> muscles{};
    std::span<const MRMujocoMuscleStateGPU> states{};
    std::span<const MRMujocoMuscleSiteGPU> sites{};
    std::span<const MRMujocoMuscleWrapGPU> wraps{};
    std::span<const MRMujocoMuscleRouteNodeGPU> routeNodes{};
    // Index in the enclosing point-query stream of a dense per-body probe
    // block: COM, local +X, local +Y, local +Z. The source route kernel uses
    // it to reconstruct a world-space spatial Jacobian at each authored site
    // and tangent point, then emits source generalized muscle force on Metal.
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;

    [[nodiscard]] bool enabled() const noexcept {
        return !muscles.empty();
    }
};

// Packed, environment-major input for the synchronous Metal articulated
// operator. q contains environmentCount * articulation.nq floats. points
// contains environmentCount * pointCount records. All query body indices are
// global EngineModel body indices. A Millard source program indexes `points`
// directly and therefore retains one authoritative device kinematics stream.
struct MetalArticulatedOperatorInput {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::size_t pointCount = 0u;
    std::span<const float> q{};
    std::span<const MRArticulatedPointImpulseGPU> points{};
    MetalMillardReferenceInput millard{};
    MetalMujocoMuscleReferenceInput mujoco{};
};

struct MetalArticulatedOperatorConfig {
    // The dense mass matrix is a correctness diagnostic. The factor-backed
    // impulse solve always runs; disabling this avoids its output bandwidth.
    bool writeDiagnosticMassMatrix = false;
    // Emits body/point kinematics and analytic point Jacobians without
    // assembling or factorizing the mass matrix. Generalized impulse and
    // delta-velocity outputs are deterministically zero. This mode is used by
    // the multi-articulation contact frontend before batched inverse ABA.
    bool pointJacobiansOnly = false;
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
    std::size_t millardMuscleElements = 0u;
    std::size_t millardMuscleBytes = 0u;
    std::size_t millardStateElements = 0u;
    std::size_t millardStateBytes = 0u;
    std::size_t millardPathPointElements = 0u;
    std::size_t millardPathPointBytes = 0u;
    std::size_t millardCurveElements = 0u;
    std::size_t millardCurveBytes = 0u;
    std::size_t millardWrapElements = 0u;
    std::size_t millardWrapBytes = 0u;
    std::size_t millardResultElements = 0u;
    std::size_t millardResultBytes = 0u;
    std::size_t millardGeneralizedForceElements = 0u;
    std::size_t millardGeneralizedForceBytes = 0u;
    std::size_t mujocoMuscleElements = 0u;
    std::size_t mujocoMuscleBytes = 0u;
    std::size_t mujocoStateElements = 0u;
    std::size_t mujocoStateBytes = 0u;
    std::size_t mujocoSiteElements = 0u;
    std::size_t mujocoSiteBytes = 0u;
    std::size_t mujocoWrapElements = 0u;
    std::size_t mujocoWrapBytes = 0u;
    std::size_t mujocoRouteNodeElements = 0u;
    std::size_t mujocoRouteNodeBytes = 0u;
    std::size_t mujocoResultElements = 0u;
    std::size_t mujocoResultBytes = 0u;
    std::size_t mujocoMuscleGeneralizedForceElements = 0u;
    std::size_t mujocoMuscleGeneralizedForceBytes = 0u;
    std::size_t mujocoGeneralizedForceElements = 0u;
    std::size_t mujocoGeneralizedForceBytes = 0u;
    // Private suballocation in the mutually-exclusive source-muscle force
    // workspace: per-muscle rows followed by their environment reduction.
    std::size_t mujocoForceWorkspaceElements = 0u;
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
    std::vector<MRMillardMuscleResultGPU> millardResults;
    std::vector<float> millardGeneralizedForces;
    std::vector<MRMujocoMuscleResultGPU> mujocoResults;
    // Device-produced [environment][muscle][dof] source contributions and
    // their deterministic [environment][dof] reduction.
    std::vector<float> mujocoMuscleGeneralizedForces;
    std::vector<float> mujocoGeneralizedForces;
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
// most once. Its fixed-binding buffer arena is reused and grows geometrically as batch
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
