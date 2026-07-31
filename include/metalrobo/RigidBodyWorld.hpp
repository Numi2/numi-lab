#pragma once

#include "metalrobo/Collision.hpp"
#include "metalrobo/ConstraintSolver.hpp"
#include "metalrobo/FreeBodyDynamics.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// Executable CPU reference pipeline for maximal-coordinate rigid bodies.
// It deliberately composes the same collision/contact records consumed by
// Metal instead of hiding a second physics convention in a test harness.
struct RigidBodyWorldConfig {
    // A constrained split step presently supports symplectic Euler only.
    // Implicit midpoint needs its midpoint configuration increment carried
    // across the impulse phase rather than reconstructed from endpoint
    // velocity.
    FreeBodyIntegratorConfig freeMotion{
        .timestep = 1.0 / 1000.0,
        .gravity = {0.0f, -9.81f, 0.0f, 0.0f},
        .integrator = FreeBodyIntegrator::symplecticEuler,
        .nonlinearIterations = 12u,
        .nonlinearTolerance = 1.0e-12,
    };
    CollisionConfig collision{};
    ContactSolverConfig contact{};
    QualityContactSolverConfig quality{};
    MRSolverType solverType = MR_SOLVER_LEGACY_PROJECTED;
    // Strictly-positive diagonal regularization used for tangent rows in the
    // strongly-convex quality QP.
    double qualityTangentialRegularization = 1.0e-9;
    std::uint32_t constraintCapacity = 128u;
};

// Current quality-world material boundary: one isotropic Coulomb coefficient
// (`static == dynamic`), no rolling/torsional friction, and no hard impulse
// cap. Unsupported material semantics return MR_STEP_UNSUPPORTED without
// publishing state or cache changes.

struct ContactAssemblyDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    std::uint32_t requiredConstraints = 0u;
    std::uint32_t newImpactCount = 0u;
    double maximumPenetration = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

struct ContactAssemblyResult {
    ContactAssemblyDiagnostics diagnostics{};
    std::vector<MRContactConstraintGPU> constraints;
};

// Converts geometric witnesses into the common solver ABI. Material mixing is
// symmetric: geometric mean for friction/effective patch lengths, maximum for
// restitution and restitution threshold, and sum for compliance.
[[nodiscard]] ContactAssemblyResult assembleContactConstraints(
    const CollisionFrame& collision,
    std::span<const MRShapeGPU> shapes,
    std::span<const MRMaterialGPU> materials,
    std::span<const MRBodyStateGPU> bodies,
    std::uint32_t constraintCapacity
);

struct RigidBodyWorldCache {
    PersistentManifoldCache manifolds;
    ContactImpulseCache impulses;
    std::uint64_t step = 0u;
};

struct RigidBodyStepDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    FreeBodyIntegratorDiagnostics freeMotion{};
    CollisionDiagnostics collision{};
    ContactAssemblyDiagnostics assembly{};
    ContactSolverDiagnostics solver{};
    QualityContactSolution qualitySolver{};
    std::uint32_t contactCount = 0u;
    double maximumPenetration = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS ||
            code == MR_STEP_FIXED_BUDGET_COMPLETE;
    }
};

// One transactional semi-implicit step:
//   unconstrained velocity prediction -> collision/manifold refresh ->
//   warm-started contact solve -> configuration integration.
// On failure, states and caches are unchanged.
[[nodiscard]] RigidBodyStepDiagnostics stepRigidBodyWorldCpu(
    std::span<const MRBodyPropertiesGPU> properties,
    std::span<MRBodyStateGPU> states,
    std::span<const MRShapeGPU> shapes,
    std::span<const MRMaterialGPU> materials,
    std::span<const BodyWrench> wrenches,
    const RigidBodyWorldConfig& config,
    RigidBodyWorldCache& cache
);

} // namespace metalrobo
