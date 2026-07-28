#pragma once

#include "metalrobo/engine_types.h"

#include <cstdint>
#include <span>

namespace metalrobo {

enum class FreeBodyIntegrator : std::uint32_t {
    symplecticEuler = 0,
    implicitMidpoint = 1,
};

using BodyWrench = MRBodyWrenchGPU;

struct FreeBodyIntegratorConfig {
    double timestep = 1.0 / 1000.0;
    mr_float4 gravity{0.0f, -9.81f, 0.0f, 0.0f};
    FreeBodyIntegrator integrator = FreeBodyIntegrator::implicitMidpoint;
    std::uint32_t nonlinearIterations = 12;
    double nonlinearTolerance = 1.0e-12;
};

struct FreeBodyIntegratorDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    std::uint32_t bodiesIntegrated = 0;
    std::uint32_t maximumIterations = 0;
    double maximumResidual = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

// Advances independent free bodies. Articulation roots use the same SO(3)
// integration and gyroscopic equation after ABA supplies their root
// acceleration; this standalone form is also the FP64 oracle for free bodies.
[[nodiscard]] FreeBodyIntegratorDiagnostics integrateFreeBodies(
    std::span<const MRBodyPropertiesGPU> properties,
    std::span<MRBodyStateGPU> states,
    std::span<const BodyWrench> wrenches,
    const FreeBodyIntegratorConfig& config
);

// Predicts force-, gravity-, damping-, and gyroscopic-response velocities
// without advancing configuration. The returned angular velocity is expressed
// in the current world frame and inverseInertiaWorld remains evaluated at the
// current pose. This is the reusable free-motion phase for a constrained
// semi-implicit step.
//
// The operation is transactional: states are unchanged on failure.
// A caller that later uses integrateFreeBodyConfigurations must select
// symplecticEuler. Reconstructing an implicit-midpoint pose update from only
// the endpoint velocity is not equivalent to the full midpoint integrator.
[[nodiscard]] FreeBodyIntegratorDiagnostics predictFreeBodyVelocities(
    std::span<const MRBodyPropertiesGPU> properties,
    std::span<MRBodyStateGPU> states,
    std::span<const BodyWrench> wrenches,
    const FreeBodyIntegratorConfig& config
);

// Advances pose once from already-solved velocities and refreshes world-frame
// inverse inertia. No force, damping, or gyroscopic velocity update is applied.
// Static bodies remain fixed; dynamic and kinematic bodies use their authored
// world-frame velocities. The operation is transactional.
[[nodiscard]] FreeBodyIntegratorDiagnostics
integrateFreeBodyConfigurations(
    std::span<const MRBodyPropertiesGPU> properties,
    std::span<MRBodyStateGPU> states,
    double timestep
);

} // namespace metalrobo
