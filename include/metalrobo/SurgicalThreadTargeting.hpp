#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace metalrobo {

using SurgicalThreadTargetPoint = std::array<double, 3>;
using SurgicalThreadSurfaceTriangle = std::array<std::uint32_t, 3>;

struct SurgicalThreadObstacleCapsule {
    SurgicalThreadTargetPoint firstM{};
    SurgicalThreadTargetPoint secondM{};
    double radiusM = 0.0;
};

struct SurgicalThreadTargetingSpec {
    double threadRadiusM = 0.0;
    double jawEnvelopeRadiusM = 0.0;
    double jawContactLengthM = 0.0;
    double minimumArcLengthFromSwageM = 0.0;
    double minimumFreeTailLengthM = 0.0;
    double preferredArcLengthFromSwageM = 0.0;
    double maximumCenterlineDeviationM = 0.0;
    double maximumTurningAngleRad = 0.0;
    double minimumTissueClearanceM = 0.0;
    double minimumObstacleClearanceM = 0.0;
    SurgicalThreadTargetPoint preferredApproachDirection{};
};

enum class SurgicalThreadTargetStatus : std::uint32_t {
    success = 0u,
    invalidDimensions,
    invalidTopology,
    nonfiniteInput,
    invalidSpecification,
    noAccessibleSegment,
};

struct SurgicalThreadTarget {
    std::uint32_t centerEdge = 0u;
    std::uint32_t windowFirstNode = 0u;
    std::uint32_t windowLastNode = 0u;
    SurgicalThreadTargetPoint centerM{};
    SurgicalThreadTargetPoint railDirection{};
    SurgicalThreadTargetPoint separationDirection{};
    SurgicalThreadTargetPoint approachDirection{};
    double arcLengthFromSwageM = 0.0;
    double freeTailLengthM = 0.0;
    double centerlineWindowLengthM = 0.0;
    double maximumCenterlineDeviationM = 0.0;
    double maximumTurningAngleRad = 0.0;
    double minimumTissueClearanceM = 0.0;
    double minimumObstacleClearanceM = 0.0;
    double score = 0.0;
};

struct SurgicalThreadTargetDiagnostics {
    SurgicalThreadTargetStatus status =
        SurgicalThreadTargetStatus::success;
    std::uint32_t evaluatedCandidates = 0u;
    std::uint32_t geometricallyStraightCandidates = 0u;
    std::uint32_t tissueClearCandidates = 0u;
    std::uint32_t obstacleClearCandidates = 0u;
    SurgicalThreadTarget target{};

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalThreadTargetStatus::success;
    }
};

struct SurgicalThreadJawSurfaceClearance {
    SurgicalThreadTargetStatus status =
        SurgicalThreadTargetStatus::success;
    std::uint32_t closestTriangle = 0u;
    double minimumAxisDistanceM = 0.0;
    double minimumEnvelopeClearanceM = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalThreadTargetStatus::success;
    }
};

// Selects a finite segment of an already resolved DER strand for a temporary
// Large Needle Driver grasp. This is a geometric accessibility certificate;
// articulated IK, continuous collision, jaw contact, and frictional retention
// remain live execution authorities.
[[nodiscard]] SurgicalThreadTargetDiagnostics
selectSurgicalThreadGraspTarget(
    std::span<const SurgicalThreadTargetPoint> threadNodes,
    std::span<const SurgicalThreadTargetPoint> tissueNodes,
    std::span<const SurgicalThreadSurfaceTriangle> tissueTriangles,
    std::span<const SurgicalThreadObstacleCapsule> obstacleCapsules,
    const SurgicalThreadTargetingSpec& spec
) noexcept;

// Evaluates the same finite jaw-axis capsule used by target selection at one
// articulated pose. This supports sampled path qualification without
// duplicating the triangle-distance implementation in an application.
[[nodiscard]] SurgicalThreadJawSurfaceClearance
evaluateSurgicalThreadJawSurfaceClearance(
    const SurgicalThreadTargetPoint& jawCenterM,
    const SurgicalThreadTargetPoint& jawRailDirection,
    double jawContactLengthM,
    double jawEnvelopeRadiusM,
    std::span<const SurgicalThreadTargetPoint> surfaceNodes,
    std::span<const SurgicalThreadSurfaceTriangle> surfaceTriangles
) noexcept;

[[nodiscard]] const char* surgicalThreadTargetStatusName(
    SurgicalThreadTargetStatus status
) noexcept;

} // namespace metalrobo
