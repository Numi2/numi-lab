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
