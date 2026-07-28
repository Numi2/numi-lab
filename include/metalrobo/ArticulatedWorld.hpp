#pragma once

#include "metalrobo/ArticulatedCollision.hpp"
#include "metalrobo/QualityContactSolver.hpp"
#include "metalrobo/RigidBodyWorld.hpp"

#include <cstdint>
#include <span>

namespace metalrobo {

enum class ArticulatedWorldFailure : std::uint32_t {
    none = 0u,
    invalidConfiguration,
    freeDynamics,
    collisionStateProjection,
    collision,
    contactAssembly,
    constraintCompilation,
    constraintKinematics,
    constraintEvaluation,
    contactAdaptation,
    contactOperator,
    contactSpaceAdapter,
    contactSolver,
    impulseApplication,
    solverVelocityMismatch,
    residualEvaluation,
    configurationIntegration,
};

// One-articulation correctness path. The velocity solve can use either the
// independent projected-gradient reference or the semismooth-Newton quality
// solver. Throughput solvers are deliberately not accepted here.
struct ArticulatedWorldConfig {
    ArticulatedDynamicsConfig dynamics{};
    CollisionConfig collision{
        .capacities = {
            .pairCapacity = MR_MAX_CONTACTS_PER_SOLVER_BATCH,
            .rawContactCapacity = MR_MAX_CONTACTS_PER_SOLVER_BATCH,
            .manifoldCapacity = MR_MAX_CONTACTS_PER_SOLVER_BATCH,
        },
    };
    ConstraintIRV1AdapterConfig constraintAdapter{};
    // Keep the semantic evaluation step synchronized with the default
    // articulated dynamics step. A positive floor is mandatory in this
    // composed path because a redundant contact patch must have one
    // solver-independent impulse even when authored compliance is zero.
    ConstraintIREvaluationConfig constraintEvaluation{
        .timestep = 1.0 / 1000.0,
        .minimumRegularization = 1.0e-6,
    };
    ConstraintIRResidualConfig constraintResidual{};
    QualityContactSolverConfig quality{};
    ReferenceConicSolverConfig reference{};
    MRSolverType solverType = MR_SOLVER_QUALITY_NEWTON;
    // Relative infinity-norm agreement required between the solver's
    // post-impulse velocity and the independently factor-applied impulse.
    // Quality is checked in contact coordinates; the reference oracle is
    // checked in generalized coordinates.
    double solverVelocityAgreementTolerance = 1.0e-9;
    std::uint32_t constraintCapacity =
        MR_MAX_CONTACTS_PER_SOLVER_BATCH;
    std::uint64_t impulseCacheMaximumAge = 8u;
};

struct ArticulatedWorldCache {
    PersistentManifoldCache manifolds;
    ContactImpulseCache impulses;
    std::uint64_t step = 0u;
};

struct ArticulatedWorldStepDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    ArticulatedWorldFailure failure = ArticulatedWorldFailure::none;
    ArticulatedDynamicsDiagnostics freeDynamics{};
    ArticulatedDynamicsDiagnostics collisionKinematics{};
    CollisionDiagnostics collision{};
    ContactAssemblyDiagnostics assembly{};
    ConstraintIRDiagnostics constraintCompilation{};
    ConstraintIRDiagnostics constraintKinematics{};
    ConstraintIRDiagnostics constraintEvaluation{};
    ArticulatedCollisionDiagnostics adaptation{};
    ArticulatedContactDiagnostics contactOperator{};
    ArticulatedContactDiagnostics contactSpaceAdapter{};
    QualityContactSolution qualitySolver{};
    ReferenceConicSolution referenceSolver{};
    ArticulatedContactDiagnostics impulseApplication{};
    ArticulatedDynamicsDiagnostics postSolveKinematics{};
    ConstraintIRResidualReport constraintResidual{};
    ArticulatedDynamicsDiagnostics integration{};
    std::uint32_t contactCount = 0u;
    std::uint64_t semanticFingerprint = 0u;
    double maximumPenetration = 0.0;
    double maximumNormalImpulse = 0.0;
    double maximumVelocityCorrection = 0.0;
    double solverVelocityRelativeDisagreement = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

// Transactional semi-implicit step for an EngineModel containing exactly one
// executable articulation:
//
//   a = M(q)^-1 (tau - bias + J_body' f_ext)
//   v_free = v + h a
//   collision(q, v_free) -> canonical ConstraintIR
//   evaluated = evaluateConstraintIR(IR, J*v_free, h)
//   lambda = exact-cone velocity solve
//   v_next = v_free + solve(M, J' lambda)
//   residual = commonConstraintResidual(evaluated, J*v_next, lambda)
//   q_next = integrateConfiguration(q, v_next, h)
//
// The impulse path uses the retained mass Cholesky factor, never the dense
// inverse compatibility adapter carried by DenseConicProblem.
//
// `environmentShapes` use body indices local to `environmentBodies`; they are
// appended after model.shapes and remapped internally. Their material indices
// reference model.materials. Environment bodies must be static or kinematic,
// have zero inverse mass, and have no articulation/link binding. Exclusions
// index the combined collider sequence [model shapes, environment shapes].
//
// Constraint evaluation is the only place that chooses restitution,
// static/dynamic friction, stabilization, or discrete regularization. The
// selected solver sees only the fingerprinted evaluated view, and the common
// residual is an acceptance gate. The composed path currently admits
// symplectic Euler only. On every failure, q, v, manifold cache, impulse cache,
// and cache step are unchanged.
[[nodiscard]] ArticulatedWorldStepDiagnostics
stepArticulatedWorldCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<double> q,
    std::span<double> v,
    std::span<const double> generalizedForce,
    std::span<const ArticulatedBodyWrench> externalWrenches,
    std::span<const MRBodyStateGPU> environmentBodies,
    std::span<const MRShapeGPU> environmentShapes,
    const ArticulatedWorldConfig& config,
    ArticulatedWorldCache& cache,
    std::span<const CollisionPairExclusion> exclusions = {}
);

} // namespace metalrobo
