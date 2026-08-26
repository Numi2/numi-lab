#pragma once

#include "metalrobo/ArticulatedDynamics.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// This is a bounded FP64 reference bridge for source-derived
// Millard2012EquilibriumMuscle records. It deliberately keeps source curve
// evaluation separate from path geometry: callers supply curve values at the
// candidate fiber/tendon state, while this layer owns the force-equilibrium
// equation, GeometryPath force directions, and analytic generalized-force
// scatter through the articulated tree.
enum class MillardMuscleReferenceStatus : std::uint32_t {
    success = 0u,
    invalidDefinition,
    invalidState,
    invalidPath,
    invalidWrap,
    kinematicsFailure,
    nonfiniteResult,
};

struct MillardMusclePathPoint {
    // Global EngineModel body index. localPoint is COM-relative body-frame m.
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::array<double, 3> localPoint{};
};

struct MillardCylinderWrap {
    // Global EngineModel body index. center and xyzBodyRotation are expressed
    // in the body frame, and cylinder local +z is its axial direction.
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::array<double, 3> center{};
    std::array<double, 3> xyzBodyRotation{};
    double radius = 0.0;
    double length = 0.0;
    // OpenSim PathWrap range endpoints are source 1-based point indices. -1
    // retains the OpenSim default of the first or last source path point.
    std::int32_t startPoint = -1;
    std::int32_t endPoint = -1;
    MRMillardPathWrapMethod method = MR_MILLARD_PATH_WRAP_HYBRID;
};

struct MillardMuscleDefinition {
    double maxIsometricForce = 0.0;
    double optimalFiberLength = 0.0;
    double tendonSlackLength = 0.0;
    double pennationAngleAtOptimal = 0.0;
    double fiberDamping = 0.0;
    double minimumActivation = 0.0;
    std::vector<MillardMusclePathPoint> pathPoints;
    // Source-order GeometryPath wraps. A finite-cylinder wrap is considered
    // only inside its authored PathWrap range; `method` is retained verbatim
    // even though OpenSim's WrapCylinder solver does not select among the
    // ellipsoid-only hybrid/midpoint/axial algorithms.
    std::vector<MillardCylinderWrap> cylinderWraps;
};

// Values of the source ActiveForceLengthCurve, ForceVelocityCurve,
// FiberForceLengthCurve, and TendonForceLengthCurve at one candidate state.
// Keeping these explicit makes the formulation auditable and lets the source
// curve evaluator evolve without changing geometry/force projection ABI.
struct MillardCurveValues {
    double activeForceLength = 0.0;
    double forceVelocity = 0.0;
    double passiveForceLength = 0.0;
    double tendonForceLength = 0.0;
};

// Source parameters for the four SmoothSegmentedFunction curves owned by an
// OpenSim Millard2012EquilibriumMuscle. Optional OpenSim properties are
// materialized by the importer using the source-class defaults, so this
// definition is self-contained and does not invent fitted curves.
struct MillardSourceCurveDefinition {
    double minNormActiveFiberLength = 0.0;
    double transitionNormFiberLength = 0.0;
    double maxNormActiveFiberLength = 0.0;
    double shallowAscendingSlope = 0.0;
    double activeMinimumValue = 0.0;
    double concentricSlopeAtVmax = 0.0;
    double concentricSlopeNearVmax = 0.0;
    double isometricSlope = 0.0;
    double eccentricSlopeAtVmax = 0.0;
    double eccentricSlopeNearVmax = 0.0;
    double maxEccentricVelocityForceMultiplier = 0.0;
    double concentricCurviness = 0.0;
    double eccentricCurviness = 0.0;
    double fiberStrainAtZeroForce = 0.0;
    double fiberStrainAtOneNormForce = 0.0;
    double fiberStiffnessAtLowForce = 0.0;
    double fiberStiffnessAtOneNormForce = 0.0;
    double fiberCurviness = 0.0;
    double tendonStrainAtOneNormForce = 0.0;
    double tendonStiffnessAtOneNormForce = 0.0;
    double tendonNormForceAtToeEnd = 0.0;
    double tendonCurviness = 0.0;
};

struct MillardMuscleState {
    double activation = 0.0;
    double normalizedFiberVelocity = 0.0;
    double fiberLength = 0.0;
    MillardCurveValues curves{};
};

struct MillardMuscleForce {
    double fiberForce = 0.0;
    double tendonForce = 0.0;
    double pennationAngle = 0.0;
    double equilibriumResidual = 0.0;
};

struct MillardMusclePathResult {
    double length = 0.0;
    std::uint32_t appliedCylinderWrapCount = 0u;
    std::vector<ArticulatedPointKinematics> points;
    // One unit world tangent per source path point; the negative of this
    // vector times tendon force is the physical force applied at that point.
    std::vector<std::array<double, 3>> lengthGradient;
};

struct MillardMuscleReferenceDiagnostics {
    MillardMuscleReferenceStatus status =
        MillardMuscleReferenceStatus::success;
    std::uint32_t failingIndex = MR_INVALID_INDEX;
    double equilibriumResidual = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MillardMuscleReferenceStatus::success;
    }
};

// Evaluates the exact Millard equilibrium-force identity
//   (Factive + Fpassive + Fdamping) cos(phi) - Ftendon = 0
// for supplied source curve values. It never silently treats a residual as a
// force-balanced state; callers choose their own equilibrium solve tolerance.
[[nodiscard]] MillardMuscleReferenceDiagnostics
evaluateMillardMuscleForce(
    const MillardMuscleDefinition& definition,
    const MillardMuscleState& state,
    MillardMuscleForce& result
);

// Evaluates the four source SmoothSegmentedFunction curves using the same
// quintic-Bezier construction and linear endpoint extension as OpenSim's
// curve factories. This is a source-code-level implementation; numerical
// equivalence against a pinned OpenSim binary is a separate qualification.
[[nodiscard]] MillardMuscleReferenceDiagnostics
evaluateMillardSourceCurves(
    const MillardSourceCurveDefinition& definition,
    double normalizedFiberLength,
    double normalizedFiberVelocity,
    double normalizedTendonLength,
    MillardCurveValues& result
);

// Evaluates source body-frame path points at the supplied articulated state,
// applying a finite-cylinder geodesic only for penetrating line segments.
// Path point output uses the same source order as definition.pathPoints.
[[nodiscard]] MillardMuscleReferenceDiagnostics
evaluateMillardMusclePath(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    const MillardMuscleDefinition& definition,
    MillardMusclePathResult& result,
    const ArticulatedDynamicsConfig& config = {}
);

// Scatters a tensile path force into articulation-local generalized effort.
// generalizedForce[dof] = -tendonForce * d(pathLength)/d(q[dof]).
[[nodiscard]] MillardMuscleReferenceDiagnostics
projectMillardMuscleTension(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    const MillardMuscleDefinition& definition,
    double tendonForce,
    std::span<double> generalizedForce,
    MillardMusclePathResult* pathResult = nullptr,
    const ArticulatedDynamicsConfig& config = {}
);

[[nodiscard]] const char* millardMuscleReferenceStatusName(
    MillardMuscleReferenceStatus status
) noexcept;

} // namespace metalrobo
