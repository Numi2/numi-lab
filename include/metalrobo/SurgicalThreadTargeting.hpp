#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

using SurgicalThreadTargetPoint = std::array<double, 3>;
using SurgicalThreadSurfaceTriangle = std::array<std::uint32_t, 3>;

struct SurgicalThreadObstacleCapsule {
    SurgicalThreadTargetPoint firstM{};
    SurgicalThreadTargetPoint secondM{};
    double radiusM = 0.0;
};

struct SurgicalThreadChannelCapsule {
    SurgicalThreadTargetPoint firstM{};
    SurgicalThreadTargetPoint secondM{};
    double radiusM = 0.0;
    std::uint32_t tract = 0u;
};

struct SurgicalThreadContactSelectionSpec {
    double threadRadiusM = 0.0;
    double maximumSurfaceSeparationM = 0.0;
    std::uint32_t tractCount = 0u;
    std::uint32_t proxyCount = 0u;
};

enum class SurgicalThreadContactSelectionStatus : std::uint32_t {
    success = 0u,
    invalidDimensions,
    invalidTopology,
    nonfiniteInput,
    invalidSpecification,
    noContactEdgeSet,
};

struct SurgicalThreadContactSelection {
    SurgicalThreadContactSelectionStatus status =
        SurgicalThreadContactSelectionStatus::success;
    std::uint32_t evaluatedEdges = 0u;
    std::vector<std::uint32_t> edges;
    std::vector<std::uint32_t> tractEdges;
    std::vector<double> tractSurfaceSeparationsM;
    double maximumSurfaceSeparationM = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalThreadContactSelectionStatus::success;
    }
};

struct SurgicalThreadProxyRebindPlan {
    SurgicalThreadContactSelectionStatus status =
        SurgicalThreadContactSelectionStatus::success;
    // Slot-ordered bindings after each single-slot maintenance command.
    std::vector<std::vector<std::uint32_t>> transitions;
    std::vector<std::uint32_t> finalEdges;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalThreadContactSelectionStatus::success;
    }
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

// Selects the smallest fixed-cost DER edge set that retains at least one
// physical material edge in each live puncture tract. One tract uses remaining
// slots as overlap during pull-through; two tracts reserve one edge for each.
// The result is geometric ownership only: Matter's live contact transaction
// remains the acceptance authority.
[[nodiscard]] SurgicalThreadContactSelection
selectSurgicalThreadContactEdges(
    std::span<const SurgicalThreadTargetPoint> threadNodes,
    std::span<const SurgicalThreadChannelCapsule> channels,
    const SurgicalThreadContactSelectionSpec& spec
);

// Converts an unordered desired edge set into deterministic slot bindings.
// Each transition changes exactly one retired slot so a caller can apply the
// Runtime maintenance contract repeatedly without simulating between updates.
[[nodiscard]] SurgicalThreadProxyRebindPlan
planSurgicalThreadProxyRebind(
    std::span<const std::uint32_t> currentSlotEdges,
    std::span<const std::uint32_t> desiredEdges,
    std::uint32_t rodEdgeCount
);

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

[[nodiscard]] const char* surgicalThreadContactSelectionStatusName(
    SurgicalThreadContactSelectionStatus status
) noexcept;

} // namespace metalrobo
