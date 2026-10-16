#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/ArticulatedJointLimits.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

enum class CoupledArticulatedRigidContactStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidDimensions,
    invalidRigidBody,
    invalidContact,
    invalidJointLimit,
    nonfiniteInput,
    dynamicsFailure,
    factorizationFailure,
    solverFailure,
    nonfiniteResult,
};

enum class CoupledContactEndpointKind : std::uint32_t {
    articulated = 0u,
    sceneBody = 1u,
};

// One endpoint of a mixed articulated/maximal-coordinate contact. `body`
// indexes EngineModel::bodies for articulated endpoints and the sceneBodies
// span for sceneBody endpoints. localPoint is COM-relative in body axes.
struct CoupledContactEndpoint {
    CoupledContactEndpointKind kind =
        CoupledContactEndpointKind::sceneBody;
    std::uint32_t body = 0u;
    std::array<double, 3> localPoint{};
};

// Canonical velocity-level contact for one mixed island. normal points A -> B
// in world coordinates and the physical relative velocity is
//
//   v_contact = frame' * (v_B(point_B) - v_A(point_A)).
//
// targetVelocity is the desired *physical* relative velocity. Static and
// kinematic scene endpoints have no generalized degree of freedom; the solver
// subtracts their prescribed point-velocity contribution from the target
// exactly once. Every contact must contain at least one articulated or dynamic
// scene endpoint. Two articulated endpoints are deliberately outside this
// first mixed-island contract.
struct CoupledArticulatedRigidIslandContact {
    CoupledContactEndpoint endpointA{};
    CoupledContactEndpoint endpointB{};
    std::array<double, 3> normal{0.0, 0.0, 1.0};
    std::array<double, 3> tangentU{1.0, 0.0, 0.0};
    std::array<double, 3> tangentV{0.0, 1.0, 0.0};
    std::array<double, 3> targetVelocity{};
    std::array<double, 3> regularization{
        1.0e-10,
        1.0e-10,
        1.0e-10,
    };
    std::array<double, 3> warmImpulse{};
    double friction = 0.7;
};

// One velocity-level contact between a body owned by the selected
// articulation and an independent maximal-coordinate rigid body.
//
// localPointArticulated and localPointRigid are COM-relative and expressed in
// their owning body axes. normal points articulated -> rigid in world
// coordinates, so the relative contact velocity is
//
//   v_contact = frame' * (v_rigid_point - v_articulated_point).
//
// The supplied frame must already be finite, near-orthonormal, and
// right-handed: cross(normal, tangentU) == tangentV. It is consumed verbatim;
// this operator never silently repairs contact coordinates.
struct CoupledArticulatedRigidContact {
    std::uint32_t articulatedBody = 0u;
    // Index into the rigidBodies span passed to the solve.
    std::uint32_t rigidBody = 0u;
    std::array<double, 3> localPointArticulated{};
    std::array<double, 3> localPointRigid{};
    std::array<double, 3> normal{0.0, 0.0, 1.0};
    std::array<double, 3> tangentU{1.0, 0.0, 0.0};
    std::array<double, 3> tangentV{0.0, 1.0, 0.0};
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

// FP64 post-impulse velocity for an independent rigid body. The linear
// component is evaluated at COM and both components are world-frame.
struct CoupledRigidBodyVelocity {
    std::array<double, 3> linear{};
    std::array<double, 3> angular{};
};

struct CoupledArticulatedRigidContactDiagnostics {
    CoupledArticulatedRigidContactStatus status =
        CoupledArticulatedRigidContactStatus::success;
    ArticulatedDynamicsStatus dynamicsStatus =
        ArticulatedDynamicsStatus::success;
    QualityContactSolution quality{};
    std::uint32_t articulationIndex = 0u;
    std::uint32_t articulationNv = 0u;
    std::uint32_t rigidBodyCount = 0u;
    std::uint32_t dynamicRigidBodyCount = 0u;
    std::uint32_t prescribedRigidBodyCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t jointLimitCount = 0u;
    double minimumArticulationCholeskyPivot = 0.0;
    double maximumArticulationCholeskyPivot = 0.0;
    double maximumArticulationInverseResidual = 0.0;
    double maximumInverseMassAsymmetry = 0.0;
    double maximumVelocityReconstructionError = 0.0;
    double maximumContactVelocityConsistencyError = 0.0;
    std::vector<double> freeContactVelocity;
    std::vector<double> postContactVelocity;
    std::vector<double> freeJointLimitVelocity;
    std::vector<double> postJointLimitVelocity;
    // Contact impulses retain the legacy packed [normal, u, v] layout.
    std::vector<double> impulses;
    std::vector<double> contactImpulses;
    // One normal impulse per scalar joint-limit row.
    std::vector<double> jointLimitImpulses;
    std::string failure;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            CoupledArticulatedRigidContactStatus::success;
    }
};

// Generic mixed-island solve. sceneBodies may contain independent dynamic,
// static, and kinematic bodies. Only dynamic scene bodies receive six
// generalized coordinates; static/kinematic point velocities are prescribed
// inputs and their output velocities are copied through unchanged.
//
// The output spans are transactional: neither is modified until every model,
// endpoint, mass, limit, solve, and reconstruction gate succeeds.
[[nodiscard]] CoupledArticulatedRigidContactDiagnostics
solveCoupledArticulatedRigidIslandCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> freeArticulationVelocity,
    std::span<const MRBodyStateGPU> sceneBodies,
    std::span<const CoupledArticulatedRigidIslandContact> contacts,
    std::span<double> postArticulationVelocity,
    std::span<CoupledRigidBodyVelocity> postSceneBodyVelocities,
    const ArticulatedDynamicsConfig& dynamicsConfig = {},
    const QualityContactSolverConfig& solverConfig = {},
    std::span<const ArticulatedJointLimitRow> jointLimitRows = {},
    std::span<const double> jointLimitWarmImpulses = {}
);

// Solves one coupled CPU FP64 contact/limit island transactionally:
//
//   M^-1 = block_diag(
//       M_articulation(q)^-1,
//       diag(m_0^-1 I, I_0,world^-1),
//       ...
//   )
//
// with analytic articulated point Jacobians and exact circular Coulomb cones.
// Optional articulation joint limits are embedded in the same solve as
// frictionless unilateral cone blocks, preserving all contact-limit cross
// terms through the combined inverse-mass operator.
// The independent rigid states provide pose, free velocity, inverse mass, and
// world inverse inertia. Every rigid state must be dynamic and unbound from an
// articulation.
//
// This compatibility entry point translates every legacy record into an
// articulated-A/dynamic-scene-B island contact and delegates to
// solveCoupledArticulatedRigidIslandCpu().
//
// postArticulationVelocity and postRigidVelocities are separate outputs and
// are not modified unless model/state/contact validation, CRBA factorization,
// the quality solve, and result reconstruction all succeed.
//
// This is a reusable velocity-level contact operator. It does not generate
// contacts, integrate configurations, update collision caches, model
// puncture/tissue/thread mechanics, or establish clinical validity.
[[nodiscard]] CoupledArticulatedRigidContactDiagnostics
solveCoupledArticulatedRigidContactsCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> freeArticulationVelocity,
    std::span<const MRBodyStateGPU> rigidBodies,
    std::span<const CoupledArticulatedRigidContact> contacts,
    std::span<double> postArticulationVelocity,
    std::span<CoupledRigidBodyVelocity> postRigidVelocities,
    const ArticulatedDynamicsConfig& dynamicsConfig = {},
    const QualityContactSolverConfig& solverConfig = {},
    std::span<const ArticulatedJointLimitRow> jointLimitRows = {},
    std::span<const double> jointLimitWarmImpulses = {}
);

} // namespace metalrobo
