#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/unified_quality_shared.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct MetalUnifiedQualityConfig {
    std::uint32_t maximumNewtonIterations = 20u;
    std::uint32_t maximumPCGIterations = 128u;
    std::uint32_t maximumLineSearchIterations = 8u;
    std::uint32_t directMaximumGeneralizedVelocities = 64u;
    std::uint32_t directMaximumRows = 256u;
    float optimalityTolerance = 1.0e-5f;
    float feasibilityTolerance = 2.0e-5f;
    float armijoConstant = 1.0e-4f;
    float lineSearchContraction = 0.5f;
    float complianceFloorMultiplier = 64.0f;
    float minimumPivot = 1.0e-10f;
    float minimumPCGDenominator = 1.0e-20f;
    float regularizationRetryScale = 64.0f;
    std::string metallibPath;
};

// All topology is shared by the problem batch. Dynamics, Jacobian, bias,
// free velocity, and warm state are environment-major. The solver consumes A
// and J as compatibility operators; it never forms J A^-1 J^T. Production
// world adapters bind ABA/free-body/rod operator kernels to the same HVP
// contract.
struct MetalUnifiedQualityProblem {
    std::size_t problemCount = 0u;
    std::size_t generalizedVelocityCount = 0u;
    std::size_t rowCount = 0u;
    std::span<const MRUnifiedQualityBlockGPU> blocks{};
    std::span<const float> dynamics{};
    std::span<const float> jacobian{};
    std::span<const float> bias{};
    std::span<const float> freeVelocity{};
    std::span<const float> warmVelocity{};
    std::span<const float> warmImpulses{};
};

enum class MetalUnifiedQualityHostStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidProblem,
    arithmeticOverflow,
    nonfiniteInput,
    metallibUnavailable,
    metalUnavailable,
    deviceUnsupported,
    pipelineFailure,
    bufferFailure,
    commandFailure,
    gpuFailure,
    nonfiniteResult,
};

struct MetalUnifiedQualityResult {
    std::vector<float> velocity;
    std::vector<float> impulses;
    std::vector<MRUnifiedQualityStatusGPU> statuses;
};

struct MetalUnifiedQualityDiagnostics {
    MetalUnifiedQualityHostStatus status =
        MetalUnifiedQualityHostStatus::success;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingProblem = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode =
        MR_UNIFIED_QUALITY_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::size_t allocatedBytes = 0u;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalUnifiedQualityHostStatus::success;
    }
};

[[nodiscard]] MetalUnifiedQualityDiagnostics
solveMetalUnifiedQuality(
    const MetalUnifiedQualityProblem& problem,
    MetalUnifiedQualityResult& output,
    const MetalUnifiedQualityConfig& config = {}
);

[[nodiscard]] const char* metalUnifiedQualityHostStatusName(
    MetalUnifiedQualityHostStatus status
) noexcept;

} // namespace metalrobo
