#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/ReferenceConicSolver.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// Use as bodyB for contact against an inertial world. bodyA must always be a
// body owned by the selected articulation.
inline constexpr std::uint32_t kArticulatedStaticWorld =
    MR_INVALID_INDEX;

enum class ArticulatedContactStatus : std::uint32_t {
    success = 0u,
    invalidDimensions,
    invalidContact,
    nonfiniteInput,
    dynamicsFailure,
    factorizationFailure,
    nonfiniteResult,
};

// Solver-facing exact circular Coulomb contact. Local points are relative to
// each body's COM and expressed in its body axes. When bodyB is static world,
// localPointB is instead the witness position in world coordinates. normal
// points A->B in world coordinates, matching DenseContactBlock and
// MRContactConstraintGPU.
// tangentU need only be nonzero and nonparallel to normal: construction
// orthonormalizes it and derives tangentV = normal x tangentU.
struct ArticulatedContact {
    std::uint32_t bodyA = 0u;
    std::uint32_t bodyB = kArticulatedStaticWorld;
    std::array<double, 3> localPointA{};
    std::array<double, 3> localPointB{};
    std::array<double, 3> normal{0.0, 0.0, -1.0};
    std::array<double, 3> tangentU{1.0, 0.0, 0.0};
    // Packed normal, tangent-u, tangent-v.
    std::array<double, 3> targetVelocity{};
    // Strictly positive diagonal impulse regularization/compliance.
    std::array<double, 3> regularization{
        1.0e-10,
        1.0e-10,
        1.0e-10,
    };
    std::array<double, 3> warmImpulse{};
    double friction = 0.7;
};

// Reusable exact generalized contact operator. contactJacobian is packed
// contact-major [normal, tangent-u, tangent-v] x nv. massCholeskyLower is the
// retained row-major lower factor used by operator applications; impulse
// response solves M delta_v = J' lambda and does not multiply a materialized
// inverse. delassus is the physical, unregularized J M^-1 J' matrix.
//
// DenseConicProblem currently requires a dense inverseMass field, so `conic`
// carries a compatibility adapter for ReferenceConicSolver and
// QualityContactSolver. It is not the intended production representation.
// Solver regularization remains in each DenseContactBlock.
struct ArticulatedContactProblem {
    std::uint32_t articulationIndex = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t contactCount = 0u;
    DenseConicProblem conic;
    std::vector<double> massCholeskyLower;
    std::vector<double> contactJacobian;
    std::vector<double> delassus;
    std::vector<ArticulatedPointKinematics> pointA;
    std::vector<ArticulatedPointKinematics> pointB;
};

struct ArticulatedContactDiagnostics {
    ArticulatedContactStatus status =
        ArticulatedContactStatus::success;
    ArticulatedDynamicsStatus dynamicsStatus =
        ArticulatedDynamicsStatus::success;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t nv = 0u;
    double minimumCholeskyPivot = 0.0;
    double maximumCholeskyPivot = 0.0;
    double maximumDenseInverseAdapterResidual = 0.0;
    double maximumDelassusAsymmetry = 0.0;
    double maximumActionResidual = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedContactStatus::success;
    }
};

// Builds analytic point Jacobians, a CRBA mass factor, contact Delassus, and
// the DenseConicProblem compatibility adapter transactionally. freeVelocity
// is the unconstrained generalized velocity at which the velocity-level
// contact problem is solved.
[[nodiscard]] ArticulatedContactDiagnostics
buildArticulatedContactProblem(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> freeVelocity,
    std::span<const ArticulatedContact> contacts,
    ArticulatedContactProblem& problem,
    const ArticulatedDynamicsConfig& config = {}
);

// Transactional matrix-free-style actions over the retained analytic
// Jacobian. These are useful to iterative production solvers without forcing
// them through DenseConicProblem's compatibility representation.
[[nodiscard]] ArticulatedContactDiagnostics
applyArticulatedContactJacobian(
    const ArticulatedContactProblem& problem,
    std::span<const double> generalizedVelocity,
    std::span<double> contactVelocity
);

[[nodiscard]] ArticulatedContactDiagnostics
applyArticulatedContactJacobianTranspose(
    const ArticulatedContactProblem& problem,
    std::span<const double> contactImpulse,
    std::span<double> generalizedImpulse
);

// Applies the exact inverse-mass action
//   delta_v = M^-1 J' impulse
//   delta_contact_velocity = J delta_v = W impulse
// without modifying either output unless the entire operation succeeds.
[[nodiscard]] ArticulatedContactDiagnostics
computeArticulatedContactImpulseResponse(
    const ArticulatedContactProblem& problem,
    std::span<const double> impulses,
    std::span<double> generalizedVelocityDelta,
    std::span<double> contactVelocityDelta
);

// Transactional in-place generalized-velocity update by M^-1 J' impulse.
[[nodiscard]] ArticulatedContactDiagnostics
applyArticulatedContactImpulses(
    const ArticulatedContactProblem& problem,
    std::span<const double> impulses,
    std::span<double> generalizedVelocity
);

} // namespace metalrobo
