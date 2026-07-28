#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/parallel_aba_shared.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct MetalMultiArticulatedABAInput {
    std::size_t environmentCount = 0u;
    // Environment-major canonical global state/effort.
    std::span<const float> q{};
    std::span<const float> v{};
    std::span<const float> effort{};
    // Optional [environment][global body]. Articulation dispatches rebase
    // their own body span without copying or host-visible intermediate work.
    std::span<const MRABABodyWrenchGPU> bodyWrenches{};
    bool applyBodyDamping = true;
    bool implicitDrives = false;
};

struct MetalMultiArticulatedABAConfig {
    std::string metallibPath;
};

enum class MetalMultiArticulatedABAStatus : std::uint32_t {
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
    internalFailure,
};

struct MetalMultiArticulatedABALayout {
    std::vector<MRMultiABADispatchGPU> dispatches;
    std::size_t qElements = 0u;
    std::size_t vElements = 0u;
    std::size_t effortElements = 0u;
    std::size_t wrenchElements = 0u;
    std::size_t accelerationElements = 0u;
    std::size_t nextVElements = 0u;
    std::size_t nextQElements = 0u;
    std::size_t statusElements = 0u;
    std::size_t totalAllocatedBytes = 0u;
};

struct MetalMultiArticulatedABAResult {
    MetalMultiArticulatedABALayout layout;
    std::vector<float> acceleration;
    std::vector<float> nextV;
    std::vector<float> nextQ;
    // Articulation-major [articulation][environment].
    std::vector<MRABAStatusGPU> statuses;
};

struct MetalMultiArticulatedABADiagnostics {
    MetalMultiArticulatedABAStatus status =
        MetalMultiArticulatedABAStatus::success;
    MetalMultiArticulatedABALayout layout;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingArticulation = MR_INVALID_INDEX;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode = MR_ABA_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalMultiArticulatedABAStatus::success;
    }
};

// One command buffer and one 2-D grid execute every articulation/environment
// packet. Host-visible state is published only when every packet succeeds.
[[nodiscard]] MetalMultiArticulatedABADiagnostics
runMetalMultiArticulatedABA(
    const EngineModel& model,
    const MetalMultiArticulatedABAInput& input,
    MetalMultiArticulatedABAResult& output,
    const MetalMultiArticulatedABAConfig& config = {}
);

[[nodiscard]] const char* metalMultiArticulatedABAStatusName(
    MetalMultiArticulatedABAStatus status
) noexcept;

} // namespace metalrobo
