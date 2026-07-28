#pragma once

#include "metalrobo/ReferenceConicSolver.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace metalrobo {

struct QualityContactSolverConfig {
    // This is a converged quality path, not a fixed-budget throughput path.
    std::uint32_t maximumIterations = 200u;
    std::uint32_t maximumLineSearchIterations = 32u;
    double kktTolerance = 1.0e-11;
    double armijoCoefficient = 1.0e-4;
    double factorizationPivotTolerance = 1.0e-14;
    double gaussNewtonRegularization = 1.0e-12;
    bool enableWarmStart = true;
};

struct QualityContactSolution {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    std::vector<double> velocity;
    // Packed as normal, tangent-u, tangent-v for every input contact.
    std::vector<double> impulses;

    std::uint32_t iterations = 0u;
    std::uint32_t semismoothNewtonSteps = 0u;
    std::uint32_t gaussNewtonFallbackSteps = 0u;
    std::uint32_t projectedGradientFallbackSteps = 0u;
    std::uint32_t lineSearchBacktracks = 0u;
    bool usedWarmStart = false;

    double objective = 0.0;
    double scaledNaturalResidual = 0.0;
    double scaledKktCertificate = 0.0;
    double maximumPrimalConeViolation = 0.0;
    double maximumDualConeViolation = 0.0;
    double maximumNormalImpulseViolation = 0.0;
    double maximumConicComplementarityResidual = 0.0;
    // Exact lambda_n * w_n residual over frictionless (mu == 0) blocks.
    // For frictional blocks, use maximumConicComplementarityResidual.
    double maximumFrictionlessNormalComplementarityResidual = 0.0;
    double complementarityGap = 0.0;
    double lipschitzBound = 0.0;
    std::string failure;

    [[nodiscard]] bool converged() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

// Solves the coupled, strongly-convex contact dual QP
//
//   min 0.5 lambda' (J M^-1 J' + R) lambda
//       + (J v_free - v_target)' lambda
//   s.t. lambda_n >= 0, ||lambda_t||_2 <= mu lambda_n
//
// with exact circular Coulomb cones. The algorithm is a globalized
// semismooth Newton solve of the natural residual
//
//   F(x) = x - Pi_K(x - gamma (Q x + c)),
//
// after mapping every friction cone to a standard Lorentz cone. It uses the
// analytic generalized derivative of the Lorentz projection, an Armijo
// globalization, a regularized semismooth Gauss-Newton descent fallback, and
// a globally convergent projected-gradient fallback.
//
// The reported dual-cone feasibility and per-contact conic inner product are
// the correct complementarity certificate for Coulomb contact. A separate
// lambda_n * w_n test is intentionally not used: it is generally nonzero for
// a valid sliding solution because the tangent term cancels it in
// <lambda, w> = 0.
[[nodiscard]] QualityContactSolution solveQualityContactProblem(
    const DenseConicProblem& problem,
    const QualityContactSolverConfig& config = {}
);

} // namespace metalrobo
