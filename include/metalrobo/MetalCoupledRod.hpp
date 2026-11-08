#pragma once

#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <cstdint>
#include <span>
#include <string>

namespace metalrobo {

// Research API for a single isolated straight-rest isotropic rod with two
// hard endpoint supports. It does not replace MetalWorld or the XPBD solver.
// Linear and angular tolerances constrain the GPU-computed discrete momentum
// residual divided by mass and rotational inertia (m/s and rad/s), plus the
// measured state-conversion error. They are not an independent FP64 residual
// certificate. Timestep and model coefficients are rounded to FP32 on device.
struct MetalCoupledRodConfig {
    double timestep = 0.001;
    std::uint32_t maximumIterations = 30u;
    double linearResidualTolerance = 0.00025;
    double angularResidualTolerance = 0.00025;
    std::string metallibPath;
};

struct MetalCoupledRodResult {
    DiscreteElasticRodState state;
    std::array<DiscreteRodAttachmentReaction, 2> reactions{};
    std::uint32_t iterations = 0u;
    std::uint32_t backtracks = 0u;
    double maximumLinearResidual = 0.0;
    double maximumAngularResidual = 0.0;
};

// No gravity, damping, self-contact, tool or body coupling is applied. Both
// targets are end-of-step positions. The target velocity must match their
// displacement over the step. Failure leaves output untouched and never falls
// back to a different solver.
[[nodiscard]] MetalDiscreteElasticRodDiagnostics runResearchMetalCoupledRod(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& initial,
    std::span<const DiscreteRodAttachment> attachments,
    MetalCoupledRodResult& output,
    const MetalCoupledRodConfig& config
);

} // namespace metalrobo
