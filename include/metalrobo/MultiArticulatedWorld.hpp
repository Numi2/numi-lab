#pragma once

#include "metalrobo/ArticulatedContact.hpp"
#include "metalrobo/ConstraintIR.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class MultiArticulatedWorldStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidModel,
    invalidDimensions,
    unsupportedConstraint,
    nonfiniteInput,
    freeDynamicsFailure,
    factorizationFailure,
    constraintEvaluationFailure,
    solverFailure,
    didNotConverge,
    integrationFailure,
    nonfiniteResult,
};

struct MultiArticulationFactor {
    std::uint32_t articulationIndex = 0u;
    std::uint32_t qOffset = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t vOffset = 0u;
    std::uint32_t nv = 0u;
    std::vector<double> massCholeskyLower;
};

struct MultiArticulationFactorCache {
    std::uint64_t generation = 0u;
    std::vector<MultiArticulationFactor> factors;
};

struct MultiArticulatedWorldConfig {
    ArticulatedDynamicsConfig dynamics{};
    ConstraintIREvaluationConfig constraintEvaluation{};
    ConstraintIRResidualConfig constraintResidual{};
    std::uint32_t solverIterations = 64u;
    double solverTolerance = 1.0e-9;
};

struct MultiArticulatedWorldDiagnostics {
    MultiArticulatedWorldStatus status =
        MultiArticulatedWorldStatus::success;
    std::uint32_t articulationCount = 0u;
    std::uint32_t constraintBlockCount = 0u;
    std::uint32_t constraintRowCount = 0u;
    std::uint32_t solverIterations = 0u;
    std::uint32_t firstFailingArticulation =
        kConstraintIRInvalidIndex;
    double maximumFactorResidual = 0.0;
    double maximumImpulseDelta = 0.0;
    ConstraintIRDiagnostics constraintEvaluation{};
    ConstraintIRResidualReport residual{};
    std::vector<ArticulatedDynamicsDiagnostics> freeDynamics;
    std::vector<ArticulatedDynamicsDiagnostics> integration;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MultiArticulatedWorldStatus::success;
    }
};

// Builds one retained mass factor per articulation. q and velocity use the
// global EngineModel layout; factors remain articulation-local. Publication
// is transactional.
[[nodiscard]] MultiArticulatedWorldDiagnostics
buildMultiArticulationFactorCache(
    const EngineModel& model,
    std::span<const double> q,
    std::span<const double> velocity,
    MultiArticulationFactorCache& output,
    const ArticulatedDynamicsConfig& config = {}
);

// Applies the block-diagonal articulated inverse mass without materializing a
// global inverse. Input and output use global v indexing.
[[nodiscard]] MultiArticulatedWorldDiagnostics
applyMultiArticulationInverseMass(
    const EngineModel& model,
    const MultiArticulationFactorCache& factors,
    std::span<const double> generalizedImpulse,
    std::span<double> velocityDelta
);

// Transactional multi-tree correctness step for the model-owned generalized
// ConstraintIR program. It supports bilateral, limit, gear/mimic, tendon,
// drive and bounded dry-friction blocks expressed as sparse generalized
// endpoints. Spatial loop/contact rows enter in the following composition
// slice after point-Jacobian ownership is generalized.
[[nodiscard]] MultiArticulatedWorldDiagnostics
stepMultiArticulatedWorldCpu(
    const EngineModel& model,
    std::span<double> q,
    std::span<double> v,
    std::span<const double> generalizedForce,
    std::span<const ArticulatedBodyWrench> externalWrenches,
    MultiArticulationFactorCache& cache,
    const MultiArticulatedWorldConfig& config = {}
);

[[nodiscard]] const char* multiArticulatedWorldStatusName(
    MultiArticulatedWorldStatus status
) noexcept;

} // namespace metalrobo
