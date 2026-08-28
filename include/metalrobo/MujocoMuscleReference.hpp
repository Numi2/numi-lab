#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace metalrobo {

// Native, FP64 reference for MyoSim's MuJoCo ``general`` muscle actuators.
// It is intentionally separate from the OpenSim Millard bridge: MyoSim uses
// MuJoCo's activation, FLV, passive-force, and spatial-tendon definitions.
enum class MujocoMuscleReferenceStatus : std::uint32_t {
    success = 0u,
    invalidDefinition,
    invalidState,
    invalidPath,
    kinematicsFailure,
    nonfiniteResult,
};

enum class MujocoRouteNodeType : std::uint32_t {
    site = 1u,
    sphere = 2u,
    cylinder = 3u,
};

struct MujocoMuscleSite {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    // COM-relative, Core body/inertia coordinates (m).
    std::array<double, 3> localPoint{};
};

struct MujocoWrapGeometry {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    MujocoRouteNodeType type = MujocoRouteNodeType::sphere;
    // COM-relative Core body/inertia coordinates (m).
    std::array<double, 3> localCenter{};
    // Row-major geometry-to-Core-body rotation.
    std::array<double, 9> rotationBody{};
    double radius = 0.0;
};

struct MujocoRouteNode {
    MujocoRouteNodeType type = MujocoRouteNodeType::site;
    // Site index for ``site`` and wrap-geometry index otherwise.
    std::uint32_t targetIndex = MR_INVALID_INDEX;
    // Optional site selecting the wrap side; MR_INVALID_INDEX means absent.
    std::uint32_t sideSiteIndex = MR_INVALID_INDEX;
};

struct MujocoMuscleDefinition {
    std::vector<MujocoRouteNode> route;
    std::array<double, 2> lengthRange{};
    double accelerationScale = 0.0;
    std::array<double, 2> controlRange{};
    std::array<double, 10> gainParameters{};
    std::array<double, 10> biasParameters{};
    std::array<double, 10> dynamicParameters{};
};

struct MujocoMuscleState {
    double excitation = 0.0;
    double activation = 0.0;
};

// NHMYO2 adds only architecture that MyoSim itself does not identify. Its
// active/passive/velocity curves, Fmax, activation law, and spatial route
// remain the source MuJoCo program above. Every muscle shares the same
// normalized tendon law; only the positive fitted lengths differ.
struct MujocoCompliantMuscleArchitecture {
    double optimalFiberLength = 0.0;
    double tendonSlackLength = 0.0;
    double tendonStrainAtOneNormalizedForce = 0.049;
    double tendonStiffnessAtOneNormalizedForce = 1.375 / 0.049;
    double tendonNormalizedForceAtToeEnd = 2.0 / 3.0;
    double tendonCurviness = 0.5;
    double normalizedFiberDamping = 0.1;
    double fitNormalizedRmse = 0.0;
};

struct MujocoCompliantMuscleState {
    double excitation = 0.0;
    double activation = 0.0;
    // Zero requests deterministic zero-velocity fibre/tendon equilibrium at
    // the current path and activation. It must not imply an arbitrary tendon
    // preload: that would inject a large, pose-dependent force on the first
    // accepted dynamics transaction.
    double fiberLength = 0.0;
    double fiberVelocity = 0.0;
};

struct MujocoCompliantMuscleResult {
    double activationDerivative = 0.0;
    double candidateFiberLength = 0.0;
    double candidateFiberVelocity = 0.0;
    double tendonTension = 0.0;
    double actuatorForce = 0.0;
    double normalizedEquilibriumResidual = 0.0;
};

struct MujocoMuscleReferenceDiagnostics;

// One backward-Euler damped fibre/tendon equilibrium update. The returned
// state is a candidate: a caller commits it only when the enclosing dynamics
// transaction is accepted.
[[nodiscard]] MujocoMuscleReferenceDiagnostics evaluateMujocoCompliantMuscle(
    double pathLength,
    double pathVelocity,
    double timestepSeconds,
    const MujocoMuscleDefinition& definition,
    const MujocoCompliantMuscleArchitecture& architecture,
    const MujocoCompliantMuscleState& acceptedState,
    MujocoCompliantMuscleResult& result
);

struct MujocoMusclePathSample {
    // World-space centreline point (m).  Wrapped portions are sampled along
    // the tangent-preserving sphere/cylinder arc rather than joined through a
    // wrap centre or by a chord.
    std::array<double, 3> world{};
    // Source-site body for a rendered attachment endpoint. Tangent contacts
    // and arc samples retain the sentinel: they are analytical path samples,
    // not authored anatomical attachment sites.
    std::uint32_t attachmentBodyIndex = std::numeric_limits<std::uint32_t>::max();
};

struct MujocoMusclePathResult {
    double length = 0.0;
    double velocity = 0.0;
    std::uint32_t appliedWrapCount = 0u;
    // d(length)/d(v), articulation-local velocity order.
    std::vector<double> lengthJacobian;
    // Ordered, source-route centreline for inspection and rendering. This is
    // diagnostic geometry only; force/Jacobian evaluation remains analytic.
    std::vector<MujocoMusclePathSample> centreline;
};

struct MujocoMuscleResult {
    MujocoMusclePathResult path{};
    double activationDerivative = 0.0;
    // MuJoCo actuator force: negative is tensile under the source convention.
    double actuatorForce = 0.0;
};

struct MujocoMuscleReferenceDiagnostics {
    MujocoMuscleReferenceStatus status = MujocoMuscleReferenceStatus::success;
    std::uint32_t failingIndex = MR_INVALID_INDEX;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MujocoMuscleReferenceStatus::success;
    }
};

// Evaluates only the source MuJoCo scalar force law at an already-resolved
// spatial-tendon length/rate. This lets offline compilers sample recruitment
// without redundantly rebuilding identical path kinematics. The returned
// force retains MuJoCo's convention: tensile force is negative.
[[nodiscard]] MujocoMuscleReferenceDiagnostics evaluateMujocoMuscleForceLaw(
    double pathLength,
    double pathVelocity,
    const MujocoMuscleDefinition& definition,
    const MujocoMuscleState& state,
    double& actuatorForce,
    double* activationDerivative = nullptr
);

// Evaluates MyoSim's source spatial route and MuJoCo general-muscle force at
// an arbitrary native Core state. Sphere/cylinder wrapping follows MuJoCo
// 3.12's mju_wrap algorithm (Apache-2.0 source) exactly in FP64 arithmetic.
[[nodiscard]] MujocoMuscleReferenceDiagnostics evaluateMujocoMuscle(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const MujocoMuscleSite> sites,
    std::span<const MujocoWrapGeometry> wraps,
    const MujocoMuscleDefinition& definition,
    const MujocoMuscleState& state,
    MujocoMuscleResult& result,
    const ArticulatedDynamicsConfig& config = {}
);

// Adds the source actuator generalized force to ``generalizedForce``. The
// force-sign and tangent/Jacobian scatter match MuJoCo's spatial tendon path:
// generalizedForce += actuatorForce * d(length)/d(v).
[[nodiscard]] MujocoMuscleReferenceDiagnostics projectMujocoMuscleForce(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const MujocoMuscleSite> sites,
    std::span<const MujocoWrapGeometry> wraps,
    const MujocoMuscleDefinition& definition,
    const MujocoMuscleState& state,
    std::span<double> generalizedForce,
    MujocoMuscleResult* result = nullptr,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] const char* mujocoMuscleReferenceStatusName(
    MujocoMuscleReferenceStatus status
) noexcept;

} // namespace metalrobo
