#pragma once

#include "metalrobo/MultiArticulatedWorld.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class MultiArticulatedContactStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidDimensions,
    invalidContact,
    nonfiniteInput,
    kinematicsFailure,
    factorizationFailure,
    solverFailure,
    nonfiniteResult,
};

// Exact contact-space operator spanning every articulation in an EngineModel.
// Jacobian and responseColumns are row-major [3 * contactCount][nv].
// responseColumns[row] is M^-1 J_row' and is assembled through the retained
// block-local articulation factors; no dense global mass matrix or inverse is
// constructed.
struct MultiArticulatedContactProblem {
    std::uint32_t nv = 0u;
    std::uint32_t contactCount = 0u;
    MultiArticulationFactorCache factors;
    std::vector<double> freeVelocity;
    std::vector<double> contactJacobian;
    std::vector<double> responseColumns;
    ContactSpaceConicProblem conic;
    std::vector<ArticulatedPointKinematics> pointA;
    std::vector<ArticulatedPointKinematics> pointB;
};

struct MultiArticulatedContactSolution {
    std::vector<double> generalizedVelocity;
    std::vector<double> impulses;
    QualityContactSolution quality;
};

struct MultiArticulatedContactDiagnostics {
    MultiArticulatedContactStatus status =
        MultiArticulatedContactStatus::success;
    ArticulatedDynamicsStatus dynamicsStatus =
        ArticulatedDynamicsStatus::success;
    std::uint32_t articulationCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t rowCount = 0u;
    std::uint32_t firstFailingContact = MR_INVALID_INDEX;
    std::uint32_t firstFailingArticulation = MR_INVALID_INDEX;
    double maximumFactorResidual = 0.0;
    double maximumDelassusAsymmetry = 0.0;
    double maximumContactVelocityResidual = 0.0;
    double minimumDelassusDiagonal = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MultiArticulatedContactStatus::success;
    }
};

// Builds a transactional FP64 exact-cone contact problem for contacts whose
// dynamic endpoints are articulation-owned bodies. bodyB may be
// kArticulatedStaticWorld. Self-contact and contacts between distinct
// articulations share the same path. Non-articulated dynamic scene bodies are
// deliberately rejected until their 6D response columns join this operator.
[[nodiscard]] MultiArticulatedContactDiagnostics
buildMultiArticulatedContactProblem(
    const EngineModel& model,
    std::span<const double> q,
    std::span<const double> freeVelocity,
    std::span<const ArticulatedContact> contacts,
    MultiArticulatedContactProblem& output,
    const ArticulatedDynamicsConfig& config = {}
);

// Solves the precomputed contact-space problem and transactionally publishes
// generalized velocity and impulses. The post-contact generalized velocity is
// reconstructed from the retained response columns.
[[nodiscard]] MultiArticulatedContactDiagnostics
solveMultiArticulatedContactProblem(
    const MultiArticulatedContactProblem& problem,
    MultiArticulatedContactSolution& output,
    const QualityContactSolverConfig& config = {}
);

[[nodiscard]] const char* multiArticulatedContactStatusName(
    MultiArticulatedContactStatus status
) noexcept;

} // namespace metalrobo
