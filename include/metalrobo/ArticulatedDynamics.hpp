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
// or floating root and revolute, continuous, or fixed joints. The computation
// is FP64. The dense mass matrix is assembled by a world-coordinate composite
// rigid-body recursion, while velocity/gravity bias is evaluated by recursive
// Newton-Euler kinematics.
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
