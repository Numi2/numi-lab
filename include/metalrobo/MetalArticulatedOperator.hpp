#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

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
