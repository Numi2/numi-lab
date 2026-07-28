#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class ArticulatedJointLimitSide : std::uint32_t {
    lower = 0u,
    upper = 1u,
};

// One deterministic scalar unilateral row:
//
//   gap(q) >= 0
//   normalVelocity = direction * v[localVIndex]
//   lambda >= 0
//
// direction is +1 for a lower stop and -1 for an upper stop. stableKey is
// globally unique within one EngineModel and orders lower before upper for
// the same generalized coordinate.
struct ArticulatedJointLimitRow {
    std::uint64_t stableKey = 0u;
    std::uint32_t globalQIndex = MR_INVALID_INDEX;
    std::uint32_t globalVIndex = MR_INVALID_INDEX;
    std::uint32_t localQIndex = MR_INVALID_INDEX;
    std::uint32_t localVIndex = MR_INVALID_INDEX;
    ArticulatedJointLimitSide side =
        ArticulatedJointLimitSide::lower;
    double direction = 1.0;
    double positionLimit = 0.0;
    double gap = 0.0;
    double freeNormalVelocity = 0.0;
    double targetVelocity = 0.0;
    double regularization = 0.0;
};

struct ArticulatedJointLimitConfig {
    double timestep = 1.0 / 1000.0;
    // A row is compiled when its current or one-step predicted gap enters
    // this distance. Inactive coordinates cost nothing in the solve.
    double activationDistance = 2.0e-3;
    // Non-penetrating motion may consume this much gap in one step. Existing
    // penetration receives a positive recovery target instead.
    double positionSlop = 1.0e-8;
    double recoveryFraction = 0.2;
    double maximumRecoverySpeed = 2.0;
    double regularization = 1.0e-10;
    std::uint32_t maximumRows = 128u;
    QualityContactSolverConfig quality{};
};

enum class ArticulatedJointLimitStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidArticulation,
    invalidDimensions,
    nonfiniteInput,
    invalidDofMetadata,
    capacityExceeded,
    dynamicsFailure,
    factorizationFailure,
    invalidProblem,
    invalidWarmStart,
    invalidImpulse,
    solverFailure,
    didNotConverge,
    nonfiniteResult,
};

struct ArticulatedJointLimitDiagnostics {
    ArticulatedJointLimitStatus status =
        ArticulatedJointLimitStatus::success;
    ArticulatedDynamicsStatus dynamicsStatus =
        ArticulatedDynamicsStatus::success;
    MRStepStatusCode solverCode = MR_STEP_SUCCESS;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t rowCount = 0u;
    std::uint32_t lowerRowCount = 0u;
    std::uint32_t upperRowCount = 0u;
    std::uint32_t penetratingRowCount = 0u;
    std::uint32_t iterations = 0u;
    std::uint32_t semismoothNewtonSteps = 0u;
    std::uint32_t fallbackSteps = 0u;
    double minimumGap = 0.0;
    double maximumPenetration = 0.0;
    double minimumCholeskyPivot = 0.0;
    double maximumCholeskyPivot = 0.0;
    double maximumDelassusAsymmetry = 0.0;
    double maximumImpulse = 0.0;
    double maximumPhysicalVelocityViolation = 0.0;
    double maximumDualViolation = 0.0;
    double maximumComplementarityResidual = 0.0;
    double scaledKktCertificate = 0.0;
    double maximumFactorSolveResidual = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedJointLimitStatus::success;
    }
};

// Reusable factor-backed limit operator. All indices in rows retain their
// global EngineModel identity while freeVelocity and the factor are
// articulation-local. delassus is the physical, unregularized
// J M^-1 J' operator in row order. When no limit row is active, factor and
// Delassus storage remain empty so the common unconstrained path does not
// assemble or factor a mass matrix.
struct ArticulatedJointLimitProblem {
    std::uint32_t articulationIndex = 0u;
    std::uint32_t nv = 0u;
    std::vector<ArticulatedJointLimitRow> rows;
    std::vector<double> freeVelocity;
    std::vector<double> massCholeskyLower;
    std::vector<double> delassus;
};

struct ArticulatedJointLimitSolution {
    ArticulatedJointLimitDiagnostics diagnostics{};
    // One non-negative impulse per problem row.
    std::vector<double> impulses;
    // Physical post-impulse J*v, excluding regularization.
    std::vector<double> constraintVelocity;
    // Articulation-local post-impulse generalized velocity.
    std::vector<double> generalizedVelocity;

    [[nodiscard]] bool converged() const noexcept {
        return diagnostics.succeeded();
    }
};

// Compiles active rows without assembling or factorizing the mass operator.
// This is the path used by a monolithic contact+limit solve, which already
// owns the authoritative articulated factor. `rows` is unchanged on failure.
[[nodiscard]] ArticulatedJointLimitDiagnostics
compileArticulatedJointLimitRows(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> freeVelocity,
    std::vector<ArticulatedJointLimitRow>& rows,
    const ArticulatedJointLimitConfig& config = {}
);

// Compiles only active scalar position limits from authoritative
// MRDofPropertiesGPU metadata, assembles the articulated mass operator, and
// constructs its physical Delassus action. q and freeVelocity are
// articulation-local. The destination is published only on complete success.
[[nodiscard]] ArticulatedJointLimitDiagnostics
buildArticulatedJointLimitProblem(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> freeVelocity,
    ArticulatedJointLimitProblem& problem,
    const ArticulatedJointLimitConfig& config = {},
    const ArticulatedDynamicsConfig& dynamicsConfig = {}
);

// Solves the strongly-convex unilateral limit QP in contact space, then
// factor-applies J' * lambda. warmImpulses is empty or exactly one finite,
// non-negative value per row. No q coordinate is clamped or modified.
[[nodiscard]] ArticulatedJointLimitSolution
solveArticulatedJointLimits(
    const ArticulatedJointLimitProblem& problem,
    const ArticulatedJointLimitConfig& config = {},
    std::span<const double> warmImpulses = {}
);

// Applies M^-1 J' lambda transactionally to an articulation-local velocity.
// Negative impulses are rejected because these rows are unilateral.
[[nodiscard]] ArticulatedJointLimitDiagnostics
applyArticulatedJointLimitImpulses(
    const ArticulatedJointLimitProblem& problem,
    std::span<const double> impulses,
    std::span<double> generalizedVelocity
);

[[nodiscard]] const char* articulatedJointLimitStatusName(
    ArticulatedJointLimitStatus status
) noexcept;

} // namespace metalrobo
