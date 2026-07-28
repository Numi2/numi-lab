#pragma once

#include "metalrobo/ReferenceConicSolver.hpp"
#include "metalrobo/quality_solver_shared.h"

#include <cstdint>
#include <string>
#include <vector>

namespace metalrobo {

struct MetalQualityContactSolverConfig {
    std::uint32_t maximumNewtonIterations = 48u;
    std::uint32_t maximumCGIterations = 96u;
    std::uint32_t maximumLineSearchIterations = 16u;
    float convergenceTolerance = 2.0e-5f;
    float armijoCoefficient = 1.0e-4f;
    float normalEquationRegularization = 1.0e-6f;
    float minimumCGDenominator = 1.0e-20f;
    bool enableWarmStart = true;
    std::string metallibPath;
};

enum class MetalQualityContactHostStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidProblem,
    unsupportedProblem,
    capacityOverflow,
    nonfiniteInput,
    metallibUnavailable,
    metalUnavailable,
    pipelineFailure,
    bufferFailure,
    commandFailure,
    gpuFailure,
    nonfiniteResult,
};

struct MetalQualityContactSolution {
    std::vector<double> impulses;
    std::vector<double> velocity;
    MRMetalQualityStatusGPU gpuStatus{};
};

struct MetalQualityContactDiagnostics {
    MetalQualityContactHostStatus status =
        MetalQualityContactHostStatus::success;
    std::uint32_t contactCount = 0u;
    std::uint32_t dimension = 0u;
    bool dispatched = false;
    bool published = false;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalQualityContactHostStatus::success;
    }
};

// FP32 Metal quality solve for a precomputed physical contact-space problem.
// Every friction coefficient must be strictly positive in this first compact
// kernel. The host maps anisotropic physical cone coordinates into standard
// Lorentz blocks, while output impulses and velocities retain physical units.
// Publication is transactional.
[[nodiscard]] MetalQualityContactDiagnostics
solveMetalQualityContactSpace(
    const ContactSpaceConicProblem& problem,
    MetalQualityContactSolution& output,
    const MetalQualityContactSolverConfig& config = {}
);

[[nodiscard]] const char* metalQualityContactHostStatusName(
    MetalQualityContactHostStatus status
) noexcept;

} // namespace metalrobo
