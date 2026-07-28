#pragma once

#include "metalrobo/ArticulatedContact.hpp"
#include "metalrobo/ArticulatedJointLimits.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class ArticulatedMixedConstraintStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidDimensions,
    invalidContactProblem,
    invalidLimitRow,
    nonfiniteInput,
    factorizationFailure,
    inconsistentOperator,
    solverFailure,
    didNotConverge,
    nonfiniteResult,
};

struct ArticulatedMixedConstraintDiagnostics {
    ArticulatedMixedConstraintStatus status =
        ArticulatedMixedConstraintStatus::success;
    MRStepStatusCode solverCode = MR_STEP_SUCCESS;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t limitRowCount = 0u;
    std::uint32_t iterations = 0u;
    std::uint32_t semismoothNewtonSteps = 0u;
    std::uint32_t fallbackSteps = 0u;
    // The final combined J' * lambda is factor-applied exactly once.
    std::uint32_t finalFactorApplications = 0u;
    double scaledKktCertificate = 0.0;
    double maximumCrossDelassusMagnitude = 0.0;
    double maximumFactorSolveResidual = 0.0;
    double maximumContactVelocityConsistencyError = 0.0;
    double maximumLimitVelocityConsistencyError = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedMixedConstraintStatus::success;
    }
};

struct ArticulatedMixedConstraintSolution {
    ArticulatedMixedConstraintDiagnostics diagnostics{};
    // Packed normal, tangent-u, tangent-v in contact problem order.
    std::vector<double> contactImpulses;
    // One non-negative scalar impulse in limitRows order.
    std::vector<double> limitImpulses;
    // Physical post-impulse J_contact * v, excluding regularization.
    std::vector<double> contactVelocity;
    // Physical post-impulse direction * v[localVIndex].
    std::vector<double> limitVelocity;
    // Articulation-local post-impulse generalized velocity.
    std::vector<double> generalizedVelocity;

    [[nodiscard]] bool converged() const noexcept {
        return diagnostics.succeeded();
    }
};

// Solves contact and active joint stops in one exact-cone contact-space
// problem:
//
//   W = [ Jc M^-1 Jc'  Jc M^-1 Jl' ]
//       [ Jl M^-1 Jc'  Jl M^-1 Jl' ].
//
// Contact blocks retain their circular Coulomb cones. Every scalar limit is
// embedded as a frictionless cone block, so only its normal coordinate is an
// optimization variable. The contact-limit cross terms are formed from the
// retained Cholesky factor, never from a materialized inverse mass.
//
// `contactProblem` supplies the authoritative factor and any contact rows.
// A limit-only call may pass a factor-only view with contactCount == 0 and
// empty contact/Jacobian/Delassus/point arrays while retaining nv,
// articulationIndex, conic.nv, conic.freeVelocity, and
// massCholeskyLower. `freeVelocity` is authoritative for this solve.
//
// Limit rows must be in strictly increasing stableKey order, retain one
// consistent global-to-local q/v offset, and have a cached
// freeNormalVelocity matching freeVelocity. They can be produced without a
// second mass factorization by compileArticulatedJointLimitRows(). Payload
// vectors remain empty on every failure. ArticulatedJointLimitRow does not
// carry an articulation index, so the caller must pass rows compiled for
// contactProblem.articulationIndex; cross-articulation ownership cannot be
// inferred from a bare row span.
[[nodiscard]] ArticulatedMixedConstraintSolution
solveArticulatedMixedConstraints(
    const ArticulatedContactProblem& contactProblem,
    std::span<const double> freeVelocity,
    std::span<const ArticulatedJointLimitRow> limitRows,
    const QualityContactSolverConfig& config = {}
);

[[nodiscard]] const char* articulatedMixedConstraintStatusName(
    ArticulatedMixedConstraintStatus status
) noexcept;

} // namespace metalrobo
