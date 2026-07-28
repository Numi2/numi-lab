#pragma once

#include "metalrobo/engine_types.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// One exact circular Coulomb contact in generalized-velocity coordinates.
// Jacobian rows map generalized velocity to relative contact velocity using
// the same A->B sign convention as MRContactConstraintGPU.
struct DenseContactBlock {
    std::vector<double> normalJacobian;
    std::vector<double> tangentUJacobian;
    std::vector<double> tangentVJacobian;
    std::array<double, 3> targetVelocity{};
    std::array<double, 3> regularization{};
    std::array<double, 3> warmImpulse{};
    double friction = 0.7;
};

struct DenseConicProblem {
    std::uint32_t nv = 0;
    // Row-major symmetric positive-definite inverse mass operator.
    std::vector<double> inverseMass;
    std::vector<double> freeVelocity;
    std::vector<DenseContactBlock> contacts;
};

struct ReferenceConicSolverConfig {
    std::uint32_t maximumIterations = 20000;
    double optimalityTolerance = 1.0e-10;
    bool enableWarmStart = true;
    bool enableAcceleration = true;
};

struct ReferenceConicSolution {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    std::vector<double> velocity;
    // Packed as normal, tangent-u, tangent-v for each contact.
    std::vector<double> impulses;
    std::uint32_t iterations = 0;
    double objective = 0.0;
    double scaledOptimalityResidual = 0.0;
    double maximumConeViolation = 0.0;
    double lipschitzBound = 0.0;
    bool usedWarmStart = false;

    [[nodiscard]] bool converged() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

// FP64, exact-cone, converged dual reference for the velocity contact problem:
//
//   min 0.5 lambda' (J M^-1 J' + R) lambda
//       + (J v_free - v_target)' lambda
//   s.t. lambda_n >= 0, ||lambda_t|| <= mu lambda_n.
//
// This is an independent projected-gradient oracle for throughput and
// QualityContactSolver validation; the latter is the globalized semismooth
// Newton quality path described in ENGINE_TARGET.
[[nodiscard]] ReferenceConicSolution solveReferenceConicProblem(
    const DenseConicProblem& problem,
    const ReferenceConicSolverConfig& config = {}
);

} // namespace metalrobo
