#include "metalrobo/SurgicalThreadTargeting.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

namespace metalrobo {
namespace {

using Point = SurgicalThreadTargetPoint;

constexpr double kGeometryEpsilon = 1.0e-12;

Point add(const Point& left, const Point& right) noexcept {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

Point subtract(const Point& left, const Point& right) noexcept {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

Point multiply(const Point& value, const double scale) noexcept {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double dot(const Point& left, const Point& right) noexcept {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

Point cross(const Point& left, const Point& right) noexcept {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

double squaredLength(const Point& value) noexcept {
    return dot(value, value);
}

double length(const Point& value) noexcept {
    return std::sqrt(squaredLength(value));
}

bool finite(const Point& value) noexcept {
    return std::isfinite(value[0]) &&
        std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

Point normalized(const Point& value) noexcept {
    const double magnitude = length(value);
    if (!(magnitude > kGeometryEpsilon) ||
        !std::isfinite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0 / magnitude);
}

double pointTriangleSquaredDistance(
    const Point& point,
    const Point& first,
    const Point& second,
    const Point& third
) noexcept {
    // Ericson's Voronoi-region closest-point construction. Degenerate
    // triangles are rejected before this helper is reached.
    const Point firstSecond = subtract(second, first);
    const Point firstThird = subtract(third, first);
    const Point firstPoint = subtract(point, first);
    const double d1 = dot(firstSecond, firstPoint);
    const double d2 = dot(firstThird, firstPoint);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return squaredLength(firstPoint);
    }

    const Point secondPoint = subtract(point, second);
    const double d3 = dot(firstSecond, secondPoint);
    const double d4 = dot(firstThird, secondPoint);
    if (d3 >= 0.0 && d4 <= d3) {
        return squaredLength(secondPoint);
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double fraction = d1 / (d1 - d3);
        return squaredLength(subtract(
            point,
            add(first, multiply(firstSecond, fraction))
        ));
    }

    const Point thirdPoint = subtract(point, third);
    const double d5 = dot(firstSecond, thirdPoint);
    const double d6 = dot(firstThird, thirdPoint);
    if (d6 >= 0.0 && d5 <= d6) {
        return squaredLength(thirdPoint);
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double fraction = d2 / (d2 - d6);
        return squaredLength(subtract(
            point,
            add(first, multiply(firstThird, fraction))
        ));
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 &&
        (d5 - d6) >= 0.0) {
        const Point secondThird = subtract(third, second);
        const double fraction =
            (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return squaredLength(subtract(
            point,
            add(second, multiply(secondThird, fraction))
        ));
    }

    const double denominator = 1.0 / (va + vb + vc);
    const double secondWeight = vb * denominator;
    const double thirdWeight = vc * denominator;
    const Point closest = add(
        first,
        add(
            multiply(firstSecond, secondWeight),
            multiply(firstThird, thirdWeight)
        )
    );
    return squaredLength(subtract(point, closest));
}

double segmentSegmentSquaredDistance(
    const Point& firstStart,
    const Point& firstEnd,
    const Point& secondStart,
    const Point& secondEnd
) noexcept {
    const Point firstDirection = subtract(firstEnd, firstStart);
    const Point secondDirection = subtract(secondEnd, secondStart);
    const Point offset = subtract(firstStart, secondStart);
    const double firstLengthSquared = squaredLength(firstDirection);
    const double secondLengthSquared = squaredLength(secondDirection);
    const double secondProjection = dot(secondDirection, offset);

    double firstFraction = 0.0;
    double secondFraction = 0.0;
    if (firstLengthSquared <= kGeometryEpsilon * kGeometryEpsilon &&
        secondLengthSquared <= kGeometryEpsilon * kGeometryEpsilon) {
        return squaredLength(offset);
    }
    if (firstLengthSquared <= kGeometryEpsilon * kGeometryEpsilon) {
        secondFraction = std::clamp(
            secondProjection / secondLengthSquared,
            0.0,
            1.0
        );
    } else {
        const double firstProjection = dot(firstDirection, offset);
        if (secondLengthSquared <=
            kGeometryEpsilon * kGeometryEpsilon) {
            firstFraction = std::clamp(
                -firstProjection / firstLengthSquared,
                0.0,
                1.0
            );
        } else {
            const double directionDot =
                dot(firstDirection, secondDirection);
            const double denominator =
                firstLengthSquared * secondLengthSquared -
                directionDot * directionDot;
            if (denominator > kGeometryEpsilon * kGeometryEpsilon) {
                firstFraction = std::clamp(
                    (directionDot * secondProjection -
                     firstProjection * secondLengthSquared) /
                        denominator,
                    0.0,
                    1.0
                );
            }
            secondFraction =
                (directionDot * firstFraction + secondProjection) /
                secondLengthSquared;
            if (secondFraction < 0.0) {
                secondFraction = 0.0;
                firstFraction = std::clamp(
                    -firstProjection / firstLengthSquared,
                    0.0,
                    1.0
                );
            } else if (secondFraction > 1.0) {
                secondFraction = 1.0;
                firstFraction = std::clamp(
                    (directionDot - firstProjection) /
                        firstLengthSquared,
                    0.0,
                    1.0
                );
            }
        }
    }

    const Point separation = subtract(
        add(offset, multiply(firstDirection, firstFraction)),
        multiply(secondDirection, secondFraction)
    );
    return squaredLength(separation);
}

bool segmentIntersectsTriangle(
    const Point& segmentStart,
    const Point& segmentEnd,
    const Point& first,
    const Point& second,
    const Point& third
) noexcept {
    const Point direction = subtract(segmentEnd, segmentStart);
    const Point firstSecond = subtract(second, first);
    const Point firstThird = subtract(third, first);
    const Point h = cross(direction, firstThird);
    const double determinant = dot(firstSecond, h);
    if (std::abs(determinant) <= kGeometryEpsilon) {
        return false;
    }
    const double inverseDeterminant = 1.0 / determinant;
    const Point startOffset = subtract(segmentStart, first);
    const double secondWeight =
        inverseDeterminant * dot(startOffset, h);
    if (secondWeight < 0.0 || secondWeight > 1.0) {
        return false;
    }
    const Point q = cross(startOffset, firstSecond);
    const double thirdWeight = inverseDeterminant * dot(direction, q);
    if (thirdWeight < 0.0 ||
        secondWeight + thirdWeight > 1.0) {
        return false;
    }
    const double segmentFraction =
        inverseDeterminant * dot(firstThird, q);
    return segmentFraction >= 0.0 && segmentFraction <= 1.0;
}

double segmentTriangleSquaredDistance(
    const Point& segmentStart,
    const Point& segmentEnd,
    const Point& first,
    const Point& second,
    const Point& third
) noexcept {
    if (segmentIntersectsTriangle(
            segmentStart,
            segmentEnd,
            first,
            second,
            third
        )) {
        return 0.0;
    }
    return std::min({
        pointTriangleSquaredDistance(
            segmentStart,
            first,
            second,
            third
        ),
        pointTriangleSquaredDistance(
            segmentEnd,
            first,
            second,
            third
        ),
        segmentSegmentSquaredDistance(
            segmentStart,
            segmentEnd,
            first,
            second
        ),
        segmentSegmentSquaredDistance(
            segmentStart,
            segmentEnd,
            second,
            third
        ),
        segmentSegmentSquaredDistance(
            segmentStart,
            segmentEnd,
            third,
            first
        ),
    });
}

double axisDeviation(
    const Point& point,
    const Point& axisOrigin,
    const Point& axisDirection
) noexcept {
    const Point offset = subtract(point, axisOrigin);
    const Point orthogonal = subtract(
        offset,
        multiply(axisDirection, dot(offset, axisDirection))
    );
    return length(orthogonal);
}

bool finiteSpecification(
    const SurgicalThreadTargetingSpec& spec
) noexcept {
    return std::isfinite(spec.threadRadiusM) &&
        std::isfinite(spec.jawEnvelopeRadiusM) &&
        std::isfinite(spec.jawContactLengthM) &&
        std::isfinite(spec.minimumArcLengthFromSwageM) &&
        std::isfinite(spec.minimumFreeTailLengthM) &&
        std::isfinite(spec.preferredArcLengthFromSwageM) &&
        std::isfinite(spec.maximumCenterlineDeviationM) &&
        std::isfinite(spec.maximumTurningAngleRad) &&
        std::isfinite(spec.minimumTissueClearanceM) &&
        std::isfinite(spec.minimumObstacleClearanceM) &&
        finite(spec.preferredApproachDirection);
}

bool validSpecification(
    const SurgicalThreadTargetingSpec& spec
) noexcept {
    return spec.threadRadiusM > 0.0 &&
        spec.jawEnvelopeRadiusM >= spec.threadRadiusM &&
        spec.jawContactLengthM > 0.0 &&
        spec.minimumArcLengthFromSwageM >= 0.0 &&
        spec.minimumFreeTailLengthM >= 0.0 &&
        spec.preferredArcLengthFromSwageM >= 0.0 &&
        spec.maximumCenterlineDeviationM >= 0.0 &&
        spec.maximumTurningAngleRad > 0.0 &&
        spec.maximumTurningAngleRad < std::numbers::pi &&
        spec.minimumTissueClearanceM >= 0.0 &&
        spec.minimumObstacleClearanceM >= 0.0 &&
        length(spec.preferredApproachDirection) > kGeometryEpsilon;
}

} // namespace

SurgicalThreadTargetDiagnostics selectSurgicalThreadGraspTarget(
    const std::span<const SurgicalThreadTargetPoint> threadNodes,
    const std::span<const SurgicalThreadTargetPoint> tissueNodes,
    const std::span<const SurgicalThreadSurfaceTriangle> tissueTriangles,
    const std::span<const SurgicalThreadObstacleCapsule> obstacleCapsules,
    const SurgicalThreadTargetingSpec& spec
) noexcept {
    SurgicalThreadTargetDiagnostics diagnostics;
    if (threadNodes.size() < 2u || tissueNodes.size() < 3u ||
        tissueTriangles.empty()) {
        diagnostics.status =
            SurgicalThreadTargetStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finiteSpecification(spec)) {
        diagnostics.status = SurgicalThreadTargetStatus::nonfiniteInput;
        return diagnostics;
    }
    if (!validSpecification(spec)) {
        diagnostics.status =
            SurgicalThreadTargetStatus::invalidSpecification;
        return diagnostics;
    }

    double totalArcLength = 0.0;
    for (std::size_t node = 0u; node < threadNodes.size(); ++node) {
        if (!finite(threadNodes[node])) {
            diagnostics.status =
                SurgicalThreadTargetStatus::nonfiniteInput;
            return diagnostics;
        }
        if (node > 0u) {
            const double edgeLength = length(subtract(
                threadNodes[node],
                threadNodes[node - 1u]
            ));
            if (!(edgeLength > kGeometryEpsilon) ||
                !std::isfinite(edgeLength)) {
                diagnostics.status =
                    SurgicalThreadTargetStatus::invalidTopology;
                return diagnostics;
            }
            totalArcLength += edgeLength;
        }
    }
    for (const Point& point : tissueNodes) {
        if (!finite(point)) {
            diagnostics.status =
                SurgicalThreadTargetStatus::nonfiniteInput;
            return diagnostics;
        }
    }
    for (const SurgicalThreadSurfaceTriangle& triangle : tissueTriangles) {
        if (triangle[0] >= tissueNodes.size() ||
            triangle[1] >= tissueNodes.size() ||
            triangle[2] >= tissueNodes.size() ||
            triangle[0] == triangle[1] ||
            triangle[1] == triangle[2] ||
            triangle[2] == triangle[0]) {
            diagnostics.status =
                SurgicalThreadTargetStatus::invalidTopology;
            return diagnostics;
        }
        const Point firstSecond = subtract(
            tissueNodes[triangle[1]],
            tissueNodes[triangle[0]]
        );
        const Point firstThird = subtract(
            tissueNodes[triangle[2]],
            tissueNodes[triangle[0]]
        );
        if (!(length(cross(firstSecond, firstThird)) >
              kGeometryEpsilon * kGeometryEpsilon)) {
            diagnostics.status =
                SurgicalThreadTargetStatus::invalidTopology;
            return diagnostics;
        }
    }
    for (const SurgicalThreadObstacleCapsule& capsule :
         obstacleCapsules) {
        if (!finite(capsule.firstM) || !finite(capsule.secondM) ||
            !std::isfinite(capsule.radiusM)) {
            diagnostics.status =
                SurgicalThreadTargetStatus::nonfiniteInput;
            return diagnostics;
        }
        if (capsule.radiusM < 0.0) {
            diagnostics.status =
                SurgicalThreadTargetStatus::invalidSpecification;
            return diagnostics;
        }
    }
    if (totalArcLength + kGeometryEpsilon <
        spec.minimumArcLengthFromSwageM +
            spec.minimumFreeTailLengthM +
            spec.jawContactLengthM) {
        diagnostics.status =
            SurgicalThreadTargetStatus::invalidDimensions;
        return diagnostics;
    }

    const Point preferredApproach = normalized(
        spec.preferredApproachDirection
    );
    double arcAtEdgeStart = 0.0;
    double bestScore = std::numeric_limits<double>::infinity();
    bool found = false;
    for (std::size_t edge = 0u;
         edge + 1u < threadNodes.size();
         ++edge) {
        ++diagnostics.evaluatedCandidates;
        const Point edgeVector = subtract(
            threadNodes[edge + 1u],
            threadNodes[edge]
        );
        const double edgeLength = length(edgeVector);
        const double centerArcLength =
            arcAtEdgeStart + 0.5 * edgeLength;
        const double freeTailLength =
            totalArcLength - centerArcLength;
        arcAtEdgeStart += edgeLength;
        if (centerArcLength < spec.minimumArcLengthFromSwageM ||
            freeTailLength < spec.minimumFreeTailLengthM) {
            continue;
        }

        std::size_t firstWindowNode = edge;
        std::size_t lastWindowNode = edge + 1u;
        double leftWindowLength = 0.5 * edgeLength;
        double rightWindowLength = 0.5 * edgeLength;
        while (leftWindowLength + rightWindowLength +
                   kGeometryEpsilon <
               spec.jawContactLengthM) {
            const bool canGrowLeft = firstWindowNode > 0u;
            const bool canGrowRight =
                lastWindowNode + 1u < threadNodes.size();
            if (!canGrowLeft && !canGrowRight) {
                break;
            }
            if (canGrowLeft &&
                (!canGrowRight ||
                 leftWindowLength <= rightWindowLength)) {
                leftWindowLength += length(subtract(
                    threadNodes[firstWindowNode],
                    threadNodes[firstWindowNode - 1u]
                ));
                --firstWindowNode;
            } else {
                rightWindowLength += length(subtract(
                    threadNodes[lastWindowNode + 1u],
                    threadNodes[lastWindowNode]
                ));
                ++lastWindowNode;
            }
        }
        const double windowLength =
            leftWindowLength + rightWindowLength;
        if (windowLength + kGeometryEpsilon <
            spec.jawContactLengthM) {
            continue;
        }

        const Point railDirection = normalized(subtract(
            threadNodes[lastWindowNode],
            threadNodes[firstWindowNode]
        ));
        if (squaredLength(railDirection) <=
            kGeometryEpsilon * kGeometryEpsilon) {
            continue;
        }
        const Point center = multiply(
            add(threadNodes[edge], threadNodes[edge + 1u]),
            0.5
        );
        double maximumDeviation = 0.0;
        for (std::size_t node = firstWindowNode;
             node <= lastWindowNode;
             ++node) {
            maximumDeviation = std::max(
                maximumDeviation,
                axisDeviation(
                    threadNodes[node],
                    center,
                    railDirection
                )
            );
        }
        double maximumTurningAngle = 0.0;
        for (std::size_t node = firstWindowNode + 1u;
             node < lastWindowNode;
             ++node) {
            const Point incoming = normalized(subtract(
                threadNodes[node],
                threadNodes[node - 1u]
            ));
            const Point outgoing = normalized(subtract(
                threadNodes[node + 1u],
                threadNodes[node]
            ));
            maximumTurningAngle = std::max(
                maximumTurningAngle,
                std::acos(std::clamp(dot(incoming, outgoing), -1.0, 1.0))
            );
        }
        if (maximumDeviation >
                spec.maximumCenterlineDeviationM ||
            maximumTurningAngle > spec.maximumTurningAngleRad) {
            continue;
        }

        Point approachDirection = subtract(
            preferredApproach,
            multiply(
                railDirection,
                dot(preferredApproach, railDirection)
            )
        );
        approachDirection = normalized(approachDirection);
        if (squaredLength(approachDirection) <=
            kGeometryEpsilon * kGeometryEpsilon) {
            continue;
        }
        Point separationDirection = normalized(cross(
            approachDirection,
            railDirection
        ));
        if (squaredLength(separationDirection) <=
            kGeometryEpsilon * kGeometryEpsilon) {
            continue;
        }
        ++diagnostics.geometricallyStraightCandidates;

        const Point contactHalfSpan = multiply(
            railDirection,
            0.5 * spec.jawContactLengthM
        );
        const Point contactStart = subtract(center, contactHalfSpan);
        const Point contactEnd = add(center, contactHalfSpan);
        double minimumTissueDistance =
            std::numeric_limits<double>::infinity();
        for (const SurgicalThreadSurfaceTriangle& triangle :
             tissueTriangles) {
            minimumTissueDistance = std::min(
                minimumTissueDistance,
                std::sqrt(segmentTriangleSquaredDistance(
                    contactStart,
                    contactEnd,
                    tissueNodes[triangle[0]],
                    tissueNodes[triangle[1]],
                    tissueNodes[triangle[2]]
                ))
            );
        }
        const double tissueClearance =
            minimumTissueDistance - spec.jawEnvelopeRadiusM;
        if (tissueClearance < spec.minimumTissueClearanceM) {
            continue;
        }
        ++diagnostics.tissueClearCandidates;

        double obstacleClearance =
            std::numeric_limits<double>::infinity();
        for (const SurgicalThreadObstacleCapsule& capsule :
             obstacleCapsules) {
            obstacleClearance = std::min(
                obstacleClearance,
                std::sqrt(segmentSegmentSquaredDistance(
                    contactStart,
                    contactEnd,
                    capsule.firstM,
                    capsule.secondM
                )) - spec.jawEnvelopeRadiusM - capsule.radiusM
            );
        }
        if (obstacleClearance < spec.minimumObstacleClearanceM) {
            continue;
        }
        ++diagnostics.obstacleClearCandidates;

        const double preferredError = std::abs(
            centerArcLength - spec.preferredArcLengthFromSwageM
        ) / totalArcLength;
        const double deviationFraction = maximumDeviation /
            std::max(
                spec.maximumCenterlineDeviationM,
                kGeometryEpsilon
            );
        const double turningFraction = maximumTurningAngle /
            spec.maximumTurningAngleRad;
        const double tissueMargin = std::clamp(
            (tissueClearance - spec.minimumTissueClearanceM) /
                spec.jawContactLengthM,
            0.0,
            1.0
        );
        const double obstacleMargin = std::isfinite(obstacleClearance)
            ? std::clamp(
                  (obstacleClearance -
                   spec.minimumObstacleClearanceM) /
                      spec.jawContactLengthM,
                  0.0,
                  1.0
              )
            : 1.0;
        const double score = preferredError +
            0.25 * deviationFraction +
            0.25 * turningFraction -
            0.01 * tissueMargin -
            0.01 * obstacleMargin;
        if (!found || score < bestScore) {
            found = true;
            bestScore = score;
            diagnostics.target = {
                .centerEdge = static_cast<std::uint32_t>(edge),
                .windowFirstNode = static_cast<std::uint32_t>(
                    firstWindowNode
                ),
                .windowLastNode = static_cast<std::uint32_t>(
                    lastWindowNode
                ),
                .centerM = center,
                .railDirection = railDirection,
                .separationDirection = separationDirection,
                .approachDirection = approachDirection,
                .arcLengthFromSwageM = centerArcLength,
                .freeTailLengthM = freeTailLength,
                .centerlineWindowLengthM = windowLength,
                .maximumCenterlineDeviationM = maximumDeviation,
                .maximumTurningAngleRad = maximumTurningAngle,
                .minimumTissueClearanceM = tissueClearance,
                .minimumObstacleClearanceM = obstacleClearance,
                .score = score,
            };
        }
    }

    diagnostics.status = found
        ? SurgicalThreadTargetStatus::success
        : SurgicalThreadTargetStatus::noAccessibleSegment;
    return diagnostics;
}

SurgicalThreadJawSurfaceClearance
evaluateSurgicalThreadJawSurfaceClearance(
    const SurgicalThreadTargetPoint& jawCenterM,
    const SurgicalThreadTargetPoint& jawRailDirection,
    const double jawContactLengthM,
    const double jawEnvelopeRadiusM,
    const std::span<const SurgicalThreadTargetPoint> surfaceNodes,
    const std::span<const SurgicalThreadSurfaceTriangle> surfaceTriangles
) noexcept {
    SurgicalThreadJawSurfaceClearance result;
    if (surfaceNodes.size() < 3u || surfaceTriangles.empty()) {
        result.status = SurgicalThreadTargetStatus::invalidDimensions;
        return result;
    }
    if (!finite(jawCenterM) || !finite(jawRailDirection) ||
        !std::isfinite(jawContactLengthM) ||
        !std::isfinite(jawEnvelopeRadiusM)) {
        result.status = SurgicalThreadTargetStatus::nonfiniteInput;
        return result;
    }
    const Point rail = normalized(jawRailDirection);
    if (!(jawContactLengthM > 0.0) ||
        !(jawEnvelopeRadiusM > 0.0) ||
        squaredLength(rail) <=
            kGeometryEpsilon * kGeometryEpsilon) {
        result.status =
            SurgicalThreadTargetStatus::invalidSpecification;
        return result;
    }
    for (const Point& point : surfaceNodes) {
        if (!finite(point)) {
            result.status = SurgicalThreadTargetStatus::nonfiniteInput;
            return result;
        }
    }
    const Point halfSpan = multiply(rail, 0.5 * jawContactLengthM);
    const Point first = subtract(jawCenterM, halfSpan);
    const Point second = add(jawCenterM, halfSpan);
    double minimumSquaredDistance =
        std::numeric_limits<double>::infinity();
    std::uint32_t closestTriangle = 0u;
    for (std::size_t triangleIndex = 0u;
         triangleIndex < surfaceTriangles.size();
         ++triangleIndex) {
        const SurgicalThreadSurfaceTriangle& triangle =
            surfaceTriangles[triangleIndex];
        if (triangle[0] >= surfaceNodes.size() ||
            triangle[1] >= surfaceNodes.size() ||
            triangle[2] >= surfaceNodes.size() ||
            triangle[0] == triangle[1] ||
            triangle[1] == triangle[2] ||
            triangle[2] == triangle[0]) {
            result.status = SurgicalThreadTargetStatus::invalidTopology;
            return result;
        }
        const Point firstSecond = subtract(
            surfaceNodes[triangle[1]],
            surfaceNodes[triangle[0]]
        );
        const Point firstThird = subtract(
            surfaceNodes[triangle[2]],
            surfaceNodes[triangle[0]]
        );
        if (!(length(cross(firstSecond, firstThird)) >
              kGeometryEpsilon * kGeometryEpsilon)) {
            result.status = SurgicalThreadTargetStatus::invalidTopology;
            return result;
        }
        const double squaredDistance =
            segmentTriangleSquaredDistance(
                first,
                second,
                surfaceNodes[triangle[0]],
                surfaceNodes[triangle[1]],
                surfaceNodes[triangle[2]]
            );
        if (squaredDistance < minimumSquaredDistance) {
            minimumSquaredDistance = squaredDistance;
            closestTriangle = static_cast<std::uint32_t>(triangleIndex);
        }
    }
    result.status = SurgicalThreadTargetStatus::success;
    result.closestTriangle = closestTriangle;
    result.minimumAxisDistanceM = std::sqrt(minimumSquaredDistance);
    result.minimumEnvelopeClearanceM =
        result.minimumAxisDistanceM - jawEnvelopeRadiusM;
    return result;
}

const char* surgicalThreadTargetStatusName(
    const SurgicalThreadTargetStatus status
) noexcept {
    switch (status) {
        case SurgicalThreadTargetStatus::success:
            return "success";
        case SurgicalThreadTargetStatus::invalidDimensions:
            return "invalid_dimensions";
        case SurgicalThreadTargetStatus::invalidTopology:
            return "invalid_topology";
        case SurgicalThreadTargetStatus::nonfiniteInput:
            return "nonfinite_input";
        case SurgicalThreadTargetStatus::invalidSpecification:
            return "invalid_specification";
        case SurgicalThreadTargetStatus::noAccessibleSegment:
            return "no_accessible_segment";
    }
    return "unknown";
}

} // namespace metalrobo
