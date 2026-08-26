#include "metalrobo/MillardMuscleReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

constexpr double kMinimumLength = 1.0e-10;
constexpr double kGeometryTolerance = 1.0e-10;

bool finite(const double value) { return std::isfinite(value); }

bool finite(const Vec3& value) {
    return std::all_of(value.begin(), value.end(), [](const double scalar) {
        return finite(scalar);
    });
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

Vec3 subtract(const Vec3& left, const Vec3& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

Vec3 scale(const Vec3& value, const double multiplier) {
    return {value[0] * multiplier, value[1] * multiplier, value[2] * multiplier};
}

double dot(const Vec3& left, const Vec3& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }

Vec3 normalized(const Vec3& value) {
    const double length = norm(value);
    return scale(value, 1.0 / length);
}

Mat3 transpose(const Mat3& matrix) {
    return {{
        {matrix[0][0], matrix[1][0], matrix[2][0]},
        {matrix[0][1], matrix[1][1], matrix[2][1]},
        {matrix[0][2], matrix[1][2], matrix[2][2]},
    }};
}

Vec3 applyMatrix(const Mat3& matrix, const Vec3& value) {
    return {
        dot(matrix[0], value), dot(matrix[1], value), dot(matrix[2], value),
    };
}

Mat3 bodyFixedXYZ(const Vec3& angles) {
    const double cx = std::cos(angles[0]);
    const double sx = std::sin(angles[0]);
    const double cy = std::cos(angles[1]);
    const double sy = std::sin(angles[1]);
    const double cz = std::cos(angles[2]);
    const double sz = std::sin(angles[2]);
    return {{
        {cy * cz, -cy * sz, sy},
        {cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx},
        {sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy},
    }};
}

Mat3 quaternionMatrix(const std::array<double, 4>& q) {
    const double x = q[0];
    const double y = q[1];
    const double z = q[2];
    const double w = q[3];
    return {{
        {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)},
        {2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)},
        {2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)},
    }};
}

MillardMuscleReferenceDiagnostics failure(
    const MillardMuscleReferenceStatus status,
    const std::uint32_t index = MR_INVALID_INDEX
) {
    MillardMuscleReferenceDiagnostics diagnostics;
    diagnostics.status = status;
    diagnostics.failingIndex = index;
    return diagnostics;
}

using BezierControls = std::array<double, 6>;

struct BezierSegment {
    BezierControls x{};
    BezierControls y{};
};

struct SmoothCurve {
    std::vector<BezierSegment> segments;
    double x0 = 0.0;
    double x1 = 0.0;
    double y0 = 0.0;
    double y1 = 0.0;
    double slope0 = 0.0;
    double slope1 = 0.0;
};

double bezierValue(const BezierControls& points, const double u) {
    const double oneMinusU = 1.0 - u;
    return points[0] * oneMinusU * oneMinusU * oneMinusU * oneMinusU * oneMinusU +
        5.0 * points[1] * u * oneMinusU * oneMinusU * oneMinusU * oneMinusU +
        10.0 * points[2] * u * u * oneMinusU * oneMinusU * oneMinusU +
        10.0 * points[3] * u * u * u * oneMinusU * oneMinusU +
        5.0 * points[4] * u * u * u * u * oneMinusU +
        points[5] * u * u * u * u * u;
}

bool cornerControls(
    const double x0,
    const double y0,
    const double slope0,
    const double x1,
    const double y1,
    const double slope1,
    const double curviness,
    BezierSegment& result
) {
    if (!finite(x0) || !finite(y0) || !finite(slope0) || !finite(x1) ||
        !finite(y1) || !finite(slope1) || !finite(curviness) || x1 <= x0 ||
        curviness < 0.0 || curviness > 1.0) {
        return false;
    }
    const double difference = slope0 - slope1;
    const double intersectionX = std::abs(difference) > 1.0e-8
        ? (y1 - y0 - x1 * slope1 + x0 * slope0) / difference
        : 0.5 * (x0 + x1);
    const double intersectionY = (intersectionX - x1) * slope1 + y1;
    const double dx0 = intersectionX - x0;
    const double dy0 = intersectionY - y0;
    const double dx1 = intersectionX - x1;
    const double dy1 = intersectionY - y1;
    const double endpointDx = x1 - x0;
    const double endpointDy = y1 - y0;
    const double endpointDistance2 = endpointDx * endpointDx + endpointDy * endpointDy;
    if (!(endpointDistance2 > dx0 * dx0 + dy0 * dy0) ||
        !(endpointDistance2 > dx1 * dx1 + dy1 * dy1)) {
        return false;
    }
    const double x0Mid = x0 + curviness * dx0;
    const double y0Mid = y0 + curviness * dy0;
    const double x1Mid = x1 + curviness * dx1;
    const double y1Mid = y1 + curviness * dy1;
    result.x = {x0, x0Mid, x0Mid, x1Mid, x1Mid, x1};
    result.y = {y0, y0Mid, y0Mid, y1Mid, y1Mid, y1};
    return std::all_of(
               result.x.begin(), result.x.end(),
               [](const double value) { return finite(value); }
           ) &&
        std::all_of(
            result.y.begin(), result.y.end(),
            [](const double value) { return finite(value); }
        );
}

bool appendCorner(
    SmoothCurve& curve,
    const double x0,
    const double y0,
    const double slope0,
    const double x1,
    const double y1,
    const double slope1,
    const double curviness
) {
    BezierSegment segment;
    if (!cornerControls(x0, y0, slope0, x1, y1, slope1, curviness, segment)) {
        return false;
    }
    curve.segments.push_back(segment);
    return true;
}

bool evaluateCurve(const SmoothCurve& curve, const double x, double& result) {
    if (!finite(x) || curve.segments.empty() || !finite(curve.x0) ||
        !finite(curve.x1) || curve.x1 <= curve.x0) {
        return false;
    }
    if (x < curve.x0) {
        result = curve.y0 + curve.slope0 * (x - curve.x0);
        return finite(result);
    }
    if (x > curve.x1) {
        result = curve.y1 + curve.slope1 * (x - curve.x1);
        return finite(result);
    }
    const BezierSegment* segment = &curve.segments.back();
    for (const BezierSegment& candidate : curve.segments) {
        if (x <= candidate.x[5]) {
            segment = &candidate;
            break;
        }
    }
    double lower = 0.0;
    double upper = 1.0;
    for (std::uint32_t iteration = 0u; iteration < 64u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (bezierValue(segment->x, middle) < x) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    result = bezierValue(segment->y, 0.5 * (lower + upper));
    return finite(result);
}

bool buildActiveCurve(
    const MillardSourceCurveDefinition& source,
    SmoothCurve& curve
) {
    const double x0 = source.minNormActiveFiberLength;
    const double x1 = source.transitionNormFiberLength;
    const double x2 = 1.0;
    const double x3 = source.maxNormActiveFiberLength;
    const double yLow = source.activeMinimumValue;
    const double slope = source.shallowAscendingSlope;
    if (!(x0 >= 0.0 && x1 > x0 && x2 > x1 && x3 > x2 && yLow >= 0.0 &&
          slope >= 0.0 && slope < (1.0 - yLow) / (x2 - x1))) {
        return false;
    }
    const double c = 0.9;
    const double xDelta = 0.05 * x2;
    const double shoulderX = x2 - xDelta;
    const double y0 = 0.0;
    const double y1 = 1.0 - slope * (shoulderX - x1);
    const double slope01 = 1.25 * (y1 - y0) / (x1 - x0);
    const double x01 = x0 + 0.5 * (x1 - x0);
    const double y01 = y0 + 0.5 * (y1 - y0);
    const double x1s = x1 + 0.5 * (shoulderX - x1);
    const double y1s = y1 + 0.5 * (1.0 - y1);
    const double x23 = (x2 + xDelta) + 0.5 * (x3 - (x2 + xDelta));
    const double y23 = 0.5;
    const double slope23 = -1.0 / ((x3 - xDelta) - (x2 + xDelta));
    curve = {{}, x0, x3, yLow, yLow, 0.0, 0.0};
    return appendCorner(curve, x0, yLow, 0.0, x01, y01, slope01, c) &&
        appendCorner(curve, x01, y01, slope01, x1s, y1s, slope, c) &&
        appendCorner(curve, x1s, y1s, slope, x2, 1.0, 0.0, c) &&
        appendCorner(curve, x2, 1.0, 0.0, x23, y23, slope23, c) &&
        appendCorner(curve, x23, y23, slope23, x3, yLow, 0.0, c);
}

bool buildForceVelocityCurve(
    const MillardSourceCurveDefinition& source,
    SmoothCurve& curve
) {
    const double fmax = source.maxEccentricVelocityForceMultiplier;
    const double concentricAtMax = source.concentricSlopeAtVmax;
    const double concentricNearMax = source.concentricSlopeNearVmax;
    const double isometric = source.isometricSlope;
    const double eccentricAtMax = source.eccentricSlopeAtVmax;
    const double eccentricNearMax = source.eccentricSlopeNearVmax;
    if (!(fmax > 1.0 && concentricAtMax >= 0.0 && concentricAtMax < 1.0 &&
          concentricNearMax > concentricAtMax && concentricNearMax <= 1.0 &&
          isometric > 1.0 && eccentricAtMax >= 0.0 &&
          eccentricAtMax < fmax - 1.0 && eccentricNearMax >= eccentricAtMax &&
          eccentricNearMax < fmax - 1.0 && source.concentricCurviness >= 0.0 &&
          source.concentricCurviness <= 1.0 && source.eccentricCurviness >= 0.0 &&
          source.eccentricCurviness <= 1.0)) {
        return false;
    }
    const double cConcentric = 0.1 + 0.8 * source.concentricCurviness;
    const double cEccentric = 0.1 + 0.8 * source.eccentricCurviness;
    const double xConcentric = -1.0;
    const double xNearConcentric = -0.9;
    const double yNearConcentric = 0.5 *
        (concentricNearMax + concentricAtMax) *
        (xNearConcentric - xConcentric);
    const double xNearEccentric = 0.9;
    const double yNearEccentric = fmax + 0.5 *
        (eccentricNearMax + eccentricAtMax) * (xNearEccentric - 1.0);
    curve = {{}, -1.0, 1.0, 0.0, fmax, concentricAtMax, eccentricAtMax};
    return appendCorner(
               curve, xConcentric, 0.0, concentricAtMax,
               xNearConcentric, yNearConcentric, concentricNearMax,
               cConcentric
           ) &&
        appendCorner(
            curve, xNearConcentric, yNearConcentric, concentricNearMax,
            0.0, 1.0, isometric, cConcentric
        ) &&
        appendCorner(
            curve, 0.0, 1.0, isometric, xNearEccentric, yNearEccentric,
            eccentricNearMax, cEccentric
        ) &&
        appendCorner(
            curve, xNearEccentric, yNearEccentric, eccentricNearMax,
            1.0, fmax, eccentricAtMax, cEccentric
        );
}

bool buildFiberForceLengthCurve(
    const MillardSourceCurveDefinition& source,
    SmoothCurve& curve
) {
    const double eZero = source.fiberStrainAtZeroForce;
    const double eIso = source.fiberStrainAtOneNormForce;
    const double lowStiffness = source.fiberStiffnessAtLowForce;
    const double isoStiffness = source.fiberStiffnessAtOneNormForce;
    if (!(eIso > eZero && isoStiffness > 1.0 / (eIso - eZero) &&
          lowStiffness > 0.0 && lowStiffness < 1.0 / (eIso - eZero) &&
          source.fiberCurviness >= 0.0 && source.fiberCurviness <= 1.0)) {
        return false;
    }
    const double xZero = 1.0 + eZero;
    const double xIso = 1.0 + eIso;
    const double deltaX = std::min(0.1 / isoStiffness, 0.1 * (xIso - xZero));
    const double xLow = xZero + deltaX;
    const double xFoot = xZero + 0.5 * (xLow - xZero);
    const double yLow = lowStiffness * (xLow - xFoot);
    const double c = 0.1 + 0.8 * source.fiberCurviness;
    curve = {{}, xZero, xIso, 0.0, 1.0, 0.0, isoStiffness};
    return appendCorner(curve, xZero, 0.0, 0.0, xLow, yLow, lowStiffness, c) &&
        appendCorner(curve, xLow, yLow, lowStiffness, xIso, 1.0, isoStiffness, c);
}

bool buildTendonForceLengthCurve(
    const MillardSourceCurveDefinition& source,
    SmoothCurve& curve
) {
    const double eIso = source.tendonStrainAtOneNormForce;
    const double isoStiffness = source.tendonStiffnessAtOneNormForce;
    const double toeForce = source.tendonNormForceAtToeEnd;
    if (!(eIso > 0.0 && toeForce > 0.0 && toeForce < 1.0 &&
          isoStiffness > 1.0 / eIso && source.tendonCurviness >= 0.0 &&
          source.tendonCurviness <= 1.0)) {
        return false;
    }
    const double xIso = 1.0 + eIso;
    const double xToe = (toeForce - 1.0) / isoStiffness + xIso;
    const double xFoot = 1.0 + (xToe - 1.0) / 10.0;
    const double yToeMid = 0.5 * toeForce;
    const double xToeMid = (yToeMid - 1.0) / isoStiffness + xIso;
    const double toeMidSlope = yToeMid / (xToeMid - xFoot);
    const double xToeControl = xFoot + 0.5 * (xToeMid - xFoot);
    const double yToeControl = toeMidSlope * (xToeControl - xFoot);
    const double c = 0.1 + 0.8 * source.tendonCurviness;
    curve = {{}, 1.0, xToe, 0.0, toeForce, 0.0, isoStiffness};
    return appendCorner(
               curve, 1.0, 0.0, 0.0, xToeControl, yToeControl,
               toeMidSlope, c
           ) &&
        appendCorner(
            curve, xToeControl, yToeControl, toeMidSlope,
            xToe, toeForce, isoStiffness, c
        );
}

bool validDefinition(const MillardMuscleDefinition& definition) {
    return finite(definition.maxIsometricForce) &&
        finite(definition.optimalFiberLength) &&
        finite(definition.tendonSlackLength) &&
        finite(definition.pennationAngleAtOptimal) &&
        finite(definition.fiberDamping) &&
        finite(definition.minimumActivation) &&
        definition.maxIsometricForce > 0.0 &&
        definition.optimalFiberLength > kMinimumLength &&
        definition.tendonSlackLength > kMinimumLength &&
        definition.pennationAngleAtOptimal >= 0.0 &&
        definition.pennationAngleAtOptimal < std::numbers::pi / 2.0 &&
        definition.fiberDamping >= 0.0 &&
        definition.minimumActivation >= 0.0 &&
        definition.minimumActivation <= 1.0 &&
        definition.pathPoints.size() >= 2u;
}

struct SegmentEvaluation {
    double length = 0.0;
    Vec3 gradientFirst{};
    Vec3 gradientSecond{};
    bool wrapped = false;
};

bool finiteCylinder(const MillardCylinderWrap& wrap) {
    return wrap.bodyIndex != MR_INVALID_INDEX && finite(wrap.center) &&
        finite(wrap.xyzBodyRotation) && finite(wrap.radius) &&
        finite(wrap.length) && wrap.radius > kMinimumLength &&
        wrap.length > kMinimumLength;
}

// Returns a surface path around an infinite cylinder, then accepts it only
// when its direct segment enters the finite axial extent. By the envelope
// theorem, endpoint derivatives are simply the two exterior tangent
// directions, so this remains an analytic path-to-Jacobian projection.
bool wrappedSegment(
    const Vec3& firstWorld,
    const Vec3& secondWorld,
    const ArticulatedBodyKinematics& body,
    const MillardCylinderWrap& wrap,
    SegmentEvaluation& result
) {
    const Mat3 bodyRotation = quaternionMatrix(body.orientation);
    const Mat3 cylinderRotation = bodyFixedXYZ(wrap.xyzBodyRotation);
    const Mat3 worldToCylinder = transpose(cylinderRotation);
    const Mat3 worldToBody = transpose(bodyRotation);
    const auto toCylinder = [&](const Vec3& world) {
        return applyMatrix(
            worldToCylinder,
            subtract(
            applyMatrix(worldToBody, subtract(world, body.centerOfMassPosition)),
                wrap.center
            )
        );
    };
    const Vec3 first = toCylinder(firstWorld);
    const Vec3 second = toCylinder(secondWorld);
    const Vec3 delta = subtract(second, first);
    const double xySquared = delta[0] * delta[0] + delta[1] * delta[1];
    const double t = xySquared > kGeometryTolerance
        ? std::clamp(-(first[0] * delta[0] + first[1] * delta[1]) / xySquared, 0.0, 1.0)
        : 0.0;
    const Vec3 closest = add(first, scale(delta, t));
    if (closest[0] * closest[0] + closest[1] * closest[1] >=
            wrap.radius * wrap.radius ||
        std::abs(closest[2]) > 0.5 * wrap.length) {
        return false;
    }
    const auto radialDistance = [](const Vec3& point) {
        return std::hypot(point[0], point[1]);
    };
    const double firstRadius = radialDistance(first);
    const double secondRadius = radialDistance(second);
    if (!(firstRadius > wrap.radius + kGeometryTolerance) ||
        !(secondRadius > wrap.radius + kGeometryTolerance)) {
        return false;
    }
    const auto tangencyAngles = [&](const Vec3& point, const double distance) {
        const double axisAngle = std::atan2(point[1], point[0]);
        const double tangentAngle = std::acos(wrap.radius / distance);
        return std::array<double, 2>{axisAngle + tangentAngle, axisAngle - tangentAngle};
    };
    const auto firstAngles = tangencyAngles(first, firstRadius);
    const auto secondAngles = tangencyAngles(second, secondRadius);
    double bestLength = std::numeric_limits<double>::infinity();
    Vec3 bestFirst{};
    Vec3 bestSecond{};
    for (const double firstAngle : firstAngles) {
        for (const double secondAngle : secondAngles) {
            double difference = std::remainder(
                secondAngle - firstAngle,
                2.0 * std::numbers::pi
            );
            const double arc = std::hypot(
                wrap.radius * difference,
                second[2] - first[2]
            );
            const double candidate =
                std::sqrt(firstRadius * firstRadius - wrap.radius * wrap.radius) +
                arc +
                std::sqrt(secondRadius * secondRadius - wrap.radius * wrap.radius);
            if (candidate < bestLength) {
                bestLength = candidate;
                bestFirst = {wrap.radius * std::cos(firstAngle), wrap.radius * std::sin(firstAngle), first[2]};
                bestSecond = {wrap.radius * std::cos(secondAngle), wrap.radius * std::sin(secondAngle), second[2]};
            }
        }
    }
    const auto toWorld = [&](const Vec3& cylinder) {
        return add(
            body.centerOfMassPosition,
            applyMatrix(
                bodyRotation,
                add(wrap.center, applyMatrix(cylinderRotation, cylinder))
            )
        );
    };
    const Vec3 tangentFirstWorld = toWorld(bestFirst);
    const Vec3 tangentSecondWorld = toWorld(bestSecond);
    result.length = bestLength;
    result.gradientFirst = normalized(subtract(firstWorld, tangentFirstWorld));
    result.gradientSecond = normalized(subtract(secondWorld, tangentSecondWorld));
    result.wrapped = true;
    return finite(result.length) && finite(result.gradientFirst) && finite(result.gradientSecond);
}

MillardMuscleReferenceDiagnostics buildPath(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const MillardMuscleDefinition& definition,
    MillardMusclePathResult& result,
    std::vector<ArticulatedPointQuery>* queries,
    std::vector<double>* jacobians,
    const ArticulatedDynamicsConfig& config
) {
    if (!validDefinition(definition)) {
        return failure(MillardMuscleReferenceStatus::invalidDefinition);
    }
    if (articulationIndex >= model.articulations.size()) {
        return failure(MillardMuscleReferenceStatus::invalidPath);
    }
    const MRArticulationGPU& articulation = model.articulations[articulationIndex];
    std::vector<ArticulatedPointQuery> localQueries(definition.pathPoints.size());
    for (std::size_t index = 0u; index < definition.pathPoints.size(); ++index) {
        const MillardMusclePathPoint& point = definition.pathPoints[index];
        if (point.bodyIndex < articulation.firstBody ||
            point.bodyIndex >= articulation.firstBody + articulation.bodyCount ||
            !finite(point.localPoint)) {
            return failure(MillardMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(index));
        }
        localQueries[index] = {point.bodyIndex, point.localPoint};
    }
    MillardMusclePathResult staged;
    staged.points.resize(localQueries.size());
    std::vector<double> localJacobians(localQueries.size() * 3u * articulation.nv);
    const ArticulatedDynamicsDiagnostics pathDiagnostics =
        computeArticulatedPointJacobians(
            model, articulationIndex, q, v, localQueries, staged.points,
            localJacobians, config
        );
    if (!pathDiagnostics.succeeded()) {
        return failure(MillardMuscleReferenceStatus::kinematicsFailure);
    }
    std::vector<ArticulatedBodyKinematics> bodies(articulation.bodyCount);
    const ArticulatedDynamicsDiagnostics bodyDiagnostics =
        computeArticulatedBodyKinematics(
            model, articulationIndex, q, v, bodies, config
        );
    if (!bodyDiagnostics.succeeded()) {
        return failure(MillardMuscleReferenceStatus::kinematicsFailure);
    }
    staged.lengthGradient.assign(localQueries.size(), {});
    std::vector<bool> usedWrap(definition.cylinderWraps.size(), false);
    for (std::size_t segment = 0u; segment + 1u < staged.points.size(); ++segment) {
        const Vec3 first = staged.points[segment].position;
        const Vec3 second = staged.points[segment + 1u].position;
        SegmentEvaluation evaluation;
        bool wrapped = false;
        for (std::size_t wrapIndex = 0u;
             wrapIndex < definition.cylinderWraps.size(); ++wrapIndex) {
            if (usedWrap[wrapIndex]) {
                continue;
            }
            const MillardCylinderWrap& wrap = definition.cylinderWraps[wrapIndex];
            if (!finiteCylinder(wrap) ||
                wrap.bodyIndex < articulation.firstBody ||
                wrap.bodyIndex >= articulation.firstBody + articulation.bodyCount) {
                return failure(MillardMuscleReferenceStatus::invalidWrap, static_cast<std::uint32_t>(wrapIndex));
            }
            if (wrappedSegment(
                    first,
                    second,
                    bodies[wrap.bodyIndex - articulation.firstBody],
                    wrap,
                    evaluation
                )) {
                usedWrap[wrapIndex] = true;
                wrapped = true;
                ++staged.appliedCylinderWrapCount;
                break;
            }
        }
        if (!wrapped) {
            const Vec3 direction = subtract(second, first);
            const double length = norm(direction);
            if (!(length > kMinimumLength) || !finite(length)) {
                return failure(MillardMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(segment));
            }
            evaluation.length = length;
            evaluation.gradientFirst = scale(direction, -1.0 / length);
            evaluation.gradientSecond = scale(direction, 1.0 / length);
        }
        staged.length += evaluation.length;
        staged.lengthGradient[segment] = add(
            staged.lengthGradient[segment], evaluation.gradientFirst
        );
        staged.lengthGradient[segment + 1u] = add(
            staged.lengthGradient[segment + 1u], evaluation.gradientSecond
        );
    }
    if (!(staged.length > kMinimumLength) || !finite(staged.length) ||
        !std::all_of(
            staged.lengthGradient.begin(), staged.lengthGradient.end(),
            [](const Vec3& value) { return finite(value); }
        )) {
        return failure(MillardMuscleReferenceStatus::nonfiniteResult);
    }
    result = std::move(staged);
    if (queries != nullptr) {
        *queries = std::move(localQueries);
    }
    if (jacobians != nullptr) {
        *jacobians = std::move(localJacobians);
    }
    return {};
}

} // namespace

MillardMuscleReferenceDiagnostics evaluateMillardMuscleForce(
    const MillardMuscleDefinition& definition,
    const MillardMuscleState& state,
    MillardMuscleForce& result
) {
    if (!validDefinition(definition)) {
        return failure(MillardMuscleReferenceStatus::invalidDefinition);
    }
    const MillardCurveValues& curves = state.curves;
    if (!finite(state.activation) || !finite(state.normalizedFiberVelocity) ||
        !finite(state.fiberLength) || !finite(curves.activeForceLength) ||
        !finite(curves.forceVelocity) || !finite(curves.passiveForceLength) ||
        !finite(curves.tendonForceLength) ||
        state.activation < definition.minimumActivation || state.activation > 1.0 ||
        state.fiberLength <= kMinimumLength || curves.activeForceLength < 0.0 ||
        curves.forceVelocity < 0.0 || curves.passiveForceLength < 0.0 ||
        curves.tendonForceLength < 0.0) {
        return failure(MillardMuscleReferenceStatus::invalidState);
    }
    const double sinePennation = std::sin(definition.pennationAngleAtOptimal) *
        definition.optimalFiberLength / state.fiberLength;
    if (!finite(sinePennation) || std::abs(sinePennation) >= 1.0) {
        return failure(MillardMuscleReferenceStatus::invalidState);
    }
    MillardMuscleForce staged;
    staged.pennationAngle = std::asin(sinePennation);
    staged.fiberForce = definition.maxIsometricForce * (
        state.activation * curves.activeForceLength * curves.forceVelocity +
        curves.passiveForceLength +
        definition.fiberDamping * state.normalizedFiberVelocity
    );
    staged.tendonForce = definition.maxIsometricForce * curves.tendonForceLength;
    staged.equilibriumResidual =
        staged.fiberForce * std::cos(staged.pennationAngle) - staged.tendonForce;
    if (!finite(staged.fiberForce) || !finite(staged.tendonForce) ||
        !finite(staged.equilibriumResidual) || staged.tendonForce < 0.0) {
        return failure(MillardMuscleReferenceStatus::nonfiniteResult);
    }
    result = staged;
    MillardMuscleReferenceDiagnostics diagnostics;
    diagnostics.equilibriumResidual = staged.equilibriumResidual;
    return diagnostics;
}

MillardMuscleReferenceDiagnostics evaluateMillardSourceCurves(
    const MillardSourceCurveDefinition& definition,
    const double normalizedFiberLength,
    const double normalizedFiberVelocity,
    const double normalizedTendonLength,
    MillardCurveValues& result
) {
    if (!finite(normalizedFiberLength) || !finite(normalizedFiberVelocity) ||
        !finite(normalizedTendonLength)) {
        return failure(MillardMuscleReferenceStatus::invalidState);
    }
    SmoothCurve active;
    SmoothCurve velocity;
    SmoothCurve passive;
    SmoothCurve tendon;
    if (!buildActiveCurve(definition, active) ||
        !buildForceVelocityCurve(definition, velocity) ||
        !buildFiberForceLengthCurve(definition, passive) ||
        !buildTendonForceLengthCurve(definition, tendon)) {
        return failure(MillardMuscleReferenceStatus::invalidDefinition);
    }
    MillardCurveValues staged;
    if (!evaluateCurve(active, normalizedFiberLength, staged.activeForceLength) ||
        !evaluateCurve(velocity, normalizedFiberVelocity, staged.forceVelocity) ||
        !evaluateCurve(passive, normalizedFiberLength, staged.passiveForceLength) ||
        !evaluateCurve(tendon, normalizedTendonLength, staged.tendonForceLength)) {
        return failure(MillardMuscleReferenceStatus::nonfiniteResult);
    }
    const auto clampRoundoff = [](double& value) {
        if (value < 0.0 && value > -1.0e-12) {
            value = 0.0;
        }
    };
    clampRoundoff(staged.activeForceLength);
    clampRoundoff(staged.forceVelocity);
    clampRoundoff(staged.passiveForceLength);
    clampRoundoff(staged.tendonForceLength);
    if (staged.activeForceLength < 0.0 || staged.forceVelocity < 0.0 ||
        staged.passiveForceLength < 0.0 || staged.tendonForceLength < 0.0) {
        return failure(MillardMuscleReferenceStatus::nonfiniteResult);
    }
    result = staged;
    return {};
}

MillardMuscleReferenceDiagnostics evaluateMillardMusclePath(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const MillardMuscleDefinition& definition,
    MillardMusclePathResult& result,
    const ArticulatedDynamicsConfig& config
) {
    return buildPath(
        model, articulationIndex, q, v, definition, result, nullptr, nullptr,
        config
    );
}

MillardMuscleReferenceDiagnostics projectMillardMuscleTension(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const MillardMuscleDefinition& definition,
    const double tendonForce,
    const std::span<double> generalizedForce,
    MillardMusclePathResult* pathResult,
    const ArticulatedDynamicsConfig& config
) {
    if (!finite(tendonForce) || tendonForce < 0.0 ||
        articulationIndex >= model.articulations.size()) {
        return failure(MillardMuscleReferenceStatus::invalidState);
    }
    const MRArticulationGPU& articulation = model.articulations[articulationIndex];
    if (generalizedForce.size() != articulation.nv) {
        return failure(MillardMuscleReferenceStatus::invalidPath);
    }
    MillardMusclePathResult path;
    std::vector<ArticulatedPointQuery> queries;
    std::vector<double> jacobians;
    const auto diagnostics = buildPath(
        model, articulationIndex, q, v, definition, path, &queries, &jacobians,
        config
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    std::vector<double> staged(articulation.nv, 0.0);
    for (std::size_t point = 0u; point < path.lengthGradient.size(); ++point) {
        for (std::size_t dof = 0u; dof < staged.size(); ++dof) {
            double derivative = 0.0;
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                derivative += path.lengthGradient[point][axis] * jacobians[
                    (point * 3u + axis) * staged.size() + dof
                ];
            }
            staged[dof] -= tendonForce * derivative;
        }
    }
    if (!std::all_of(staged.begin(), staged.end(), [](const double value) {
            return finite(value);
        })) {
        return failure(MillardMuscleReferenceStatus::nonfiniteResult);
    }
    std::copy(staged.begin(), staged.end(), generalizedForce.begin());
    if (pathResult != nullptr) {
        *pathResult = std::move(path);
    }
    return {};
}

const char* millardMuscleReferenceStatusName(
    const MillardMuscleReferenceStatus status
) noexcept {
    switch (status) {
    case MillardMuscleReferenceStatus::success: return "success";
    case MillardMuscleReferenceStatus::invalidDefinition: return "invalid_definition";
    case MillardMuscleReferenceStatus::invalidState: return "invalid_state";
    case MillardMuscleReferenceStatus::invalidPath: return "invalid_path";
    case MillardMuscleReferenceStatus::invalidWrap: return "invalid_wrap";
    case MillardMuscleReferenceStatus::kinematicsFailure: return "kinematics_failure";
    case MillardMuscleReferenceStatus::nonfiniteResult: return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
