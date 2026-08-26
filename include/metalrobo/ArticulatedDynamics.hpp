#pragma once

#include "metalrobo/EngineModel.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace metalrobo {

// Status is deliberately independent of MRStepStatusCode: this FP64 reference
// reports model/topology/state failures before a production dispatch exists.
enum class ArticulatedDynamicsStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    nonfiniteInput,
    invalidQuaternion,
    jointLimitViolation,
    bodySpeedLimitViolation,
    massMatrixNotPositiveDefinite,
    nonlinearSolveFailed,
    nonfiniteResult,
};

enum class ArticulatedIntegrator : std::uint32_t {
    symplecticEuler = 0u,
    implicitMidpoint = 1u,
};

// Optional limits are articulation-local and use the same nq layout as q.
// Supplying either array requires supplying both arrays at exactly nq entries.
// Infinity is permitted; NaN and lower > upper are rejected.
struct ArticulatedJointLimits {
    std::span<const double> lower{};
    std::span<const double> upper{};
    double tolerance = 1.0e-10;
};

struct ArticulatedDynamicsConfig {
    std::array<double, 3> gravity{0.0, 0.0, -9.81};
    double timestep = 1.0 / 1000.0;
    bool applyBodyDamping = true;
    bool enforceBodySpeedLimits = false;
    ArticulatedIntegrator integrator =
        ArticulatedIntegrator::symplecticEuler;
    std::uint32_t nonlinearIterations = 10u;
    double nonlinearTolerance = 1.0e-11;
    ArticulatedJointLimits limits{};
};

// World-frame force and torque about the body's center of mass. The span
// passed to dynamics routines is either empty or indexed by global body index.
struct ArticulatedBodyWrench {
    std::array<double, 3> force{};
    std::array<double, 3> torque{};
};

struct ArticulatedDynamicsDiagnostics {
    ArticulatedDynamicsStatus status =
        ArticulatedDynamicsStatus::success;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t nonlinearIterations = 0u;
    double minimumCholeskyPivot = 0.0;
    double maximumCholeskyPivot = 0.0;
    double estimatedMassMatrixCondition = 0.0;
    double nonlinearResidual = 0.0;
    double quaternionNormError = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedDynamicsStatus::success;
    }
};

struct ArticulatedInvariants {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    std::array<double, 3> linearMomentum{};
    // Angular momentum about the world origin.
    std::array<double, 3> angularMomentum{};
};

// Public FP64 kinematics records use the same COM-centred convention as the
// engine ABI. Orientation is body-to-world quaternion xyzw; twists are
// expressed in world coordinates and linear velocity is evaluated at COM.
struct ArticulatedBodyKinematics {
    std::uint32_t bodyIndex = 0u;
    std::array<double, 3> centerOfMassPosition{};
    std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};
    std::array<double, 3> linearVelocity{};
    std::array<double, 3> angularVelocity{};
};

// localPoint is relative to the body's COM and expressed in body axes.
struct ArticulatedPointQuery {
    std::uint32_t bodyIndex = 0u;
    std::array<double, 3> localPoint{};
};

struct ArticulatedPointKinematics {
    std::array<double, 3> position{};
    std::array<double, 3> linearVelocity{};
};

// Generalized-coordinate convention:
//   floating q = world root-COM xyz, body-to-world quaternion xyzw,
//                followed by joint coordinates;
//   floating v = world linear velocity of the root COM, world angular
//                velocity, followed by joint rates.
// Fixed roots have identity pose at the world origin. Joint anchors and
// joint-frame rotations come directly from MRJointDescriptorGPU; anchors are
// coordinates relative to each body's COM (not its URDF link-frame origin).
//
// Supported production-independent reference topology is a tree with a fixed
// or floating root and revolute, prismatic, continuous, fixed, or immutable
// OpenSim FunctionBased joints. The computation is FP64. The dense mass
// matrix is assembled from analytic tree Jacobians, while velocity/gravity
// bias is evaluated by recursive Newton-Euler kinematics. Generic O(n) Metal
// ABA does not admit FunctionBased joints; MetalWorld separately admits the
// bounded fixed-root source tree through its dense source-dynamics kernel.
//
// The following two queries expose that same analytic tree recursion to
// constraint layers. Results are transactional. Point Jacobians are packed
// query-major, then xyz row, then generalized-velocity column:
//   jacobian[(query * 3 + axis) * nv + dof].
// No configuration perturbation or finite differencing is used.
[[nodiscard]] ArticulatedDynamicsDiagnostics
computeArticulatedBodyKinematics(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<ArticulatedBodyKinematics> bodyKinematics,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] ArticulatedDynamicsDiagnostics
computeArticulatedPointJacobians(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const ArticulatedPointQuery> points,
    std::span<ArticulatedPointKinematics> pointKinematics,
    std::span<double> pointJacobiansRowMajor,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] ArticulatedDynamicsDiagnostics
computeArticulatedMassMatrix(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<double> massMatrixRowMajor,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] ArticulatedDynamicsDiagnostics
computeArticulatedInverseDynamics(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const double> generalizedAcceleration,
    std::span<const ArticulatedBodyWrench> externalWrenches,
    std::span<double> generalizedForce,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] ArticulatedDynamicsDiagnostics
computeArticulatedForwardDynamics(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const double> generalizedForce,
    std::span<const ArticulatedBodyWrench> externalWrenches,
    std::span<double> generalizedAcceleration,
    const ArticulatedDynamicsConfig& config = {}
);

// Advances q/v transactionally. Floating quaternions use the SO(3)
// exponential map and are normalized after composition. Implicit midpoint is
// a converged FP64 reference step, not the intended high-throughput path.
[[nodiscard]] ArticulatedDynamicsDiagnostics
integrateArticulatedState(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<double> q,
    std::span<double> v,
    std::span<const double> generalizedForce,
    std::span<const ArticulatedBodyWrench> externalWrenches,
    const ArticulatedDynamicsConfig& config = {}
);

// Advances only q from an already-computed generalized velocity. No force,
// bias, mass-matrix, or acceleration evaluation occurs and v is never
// modified. Floating roots use the same world-angular-velocity SO(3)
// exponential convention as integrateArticulatedState. The candidate
// configuration and resulting body kinematics are fully validated before q
// is published, so failure is transactional.
[[nodiscard]] ArticulatedDynamicsDiagnostics
integrateArticulatedConfiguration(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<double> q,
    std::span<const double> velocity,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] ArticulatedDynamicsDiagnostics
computeArticulatedInvariants(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    ArticulatedInvariants& invariants,
    const ArticulatedDynamicsConfig& config = {}
);

} // namespace metalrobo
