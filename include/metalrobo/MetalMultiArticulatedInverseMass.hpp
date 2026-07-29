#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/parallel_aba_shared.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

// Environment-major global state and generalized right-hand sides. Each RHS
// spans the complete EngineModel velocity layout; every articulation owns and
// writes only its compiled v range.
struct MetalMultiArticulatedInverseMassInput {
    std::size_t environmentCount = 0u;
    std::size_t rhsCount = 0u;
    std::span<const float> q{};
    std::span<const float> rightHandSides{};
};

struct MetalMultiArticulatedInverseMassConfig {
    std::string metallibPath;
};

enum class MetalMultiArticulatedInverseMassStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    arithmeticOverflow,
    nonfiniteInput,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    gpuArticulationFailure,
    nonfiniteResult,
};

struct MetalMultiArticulatedInverseMassLayout {
    std::vector<MRMultiInverseMassDispatchGPU> dispatches;
    std::size_t qElements = 0u;
    std::size_t rhsElements = 0u;
    std::size_t outputElements = 0u;
    std::size_t statusElements = 0u;
    std::size_t totalAllocatedBytes = 0u;
};

struct MetalMultiArticulatedInverseMassResult {
    MetalMultiArticulatedInverseMassLayout layout;
    // Packed [environment][rhs][global v].
    std::vector<float> output;
    // Packed [articulation][environment].
    std::vector<MRInverseMassStatusGPU> statuses;
};

struct MetalMultiArticulatedInverseMassDiagnostics {
    MetalMultiArticulatedInverseMassStatus status =
        MetalMultiArticulatedInverseMassStatus::success;
    MetalMultiArticulatedInverseMassLayout layout;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingArticulation = MR_INVALID_INDEX;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode = MR_INVERSE_MASS_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MetalMultiArticulatedInverseMassStatus::success;
    }
};

// Applies the block-diagonal articulated inverse mass for every articulation
// and environment in one two-dimensional Metal grid. No global dense mass
// matrix is constructed. Publication is all-or-nothing.
[[nodiscard]] MetalMultiArticulatedInverseMassDiagnostics
runMetalMultiArticulatedInverseMass(
    const EngineModel& model,
    const MetalMultiArticulatedInverseMassInput& input,
    MetalMultiArticulatedInverseMassResult& output,
    const MetalMultiArticulatedInverseMassConfig& config = {}
);

[[nodiscard]] const char*
metalMultiArticulatedInverseMassStatusName(
    MetalMultiArticulatedInverseMassStatus status
) noexcept;

} // namespace metalrobo
