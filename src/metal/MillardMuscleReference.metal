#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/millard_muscle_gpu.h"

using namespace metal;

namespace {

constant float kMinimumLength = 1.0e-8f;
constant float kGeometryTolerance = 1.0e-7f;
constant float kStaticFiberMargin = 0.01f;

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}

// OpenSim BodyFixedXYZ orientation, written in the same row-major convention
// as the FP64 source bridge. Its transpose is the inverse because every
// source rotation is rigid.
inline float3 bodyFixedXYZRotate(const float3 angles, const float3 value) {
    const float cx = cos(angles.x);
    const float sx = sin(angles.x);
    const float cy = cos(angles.y);
    const float sy = sin(angles.y);
    const float cz = cos(angles.z);
    const float sz = sin(angles.z);
    return float3(
        cy * cz * value.x - cy * sz * value.y + sy * value.z,
        (cx * sz + cz * sx * sy) * value.x +
            (cx * cz - sx * sy * sz) * value.y - cy * sx * value.z,
        (sx * sz - cx * cz * sy) * value.x +
            (cz * sx + cx * sy * sz) * value.y + cx * cy * value.z
    );
}

inline float3 bodyFixedXYZTransposeRotate(const float3 angles, const float3 value) {
    const float cx = cos(angles.x);
    const float sx = sin(angles.x);
    const float cy = cos(angles.y);
    const float sy = sin(angles.y);
    const float cz = cos(angles.z);
    const float sz = sin(angles.z);
    return float3(
        cy * cz * value.x + (cx * sz + cz * sx * sy) * value.y +
            (sx * sz - cx * cz * sy) * value.z,
        -cy * sz * value.x + (cx * cz - sx * sy * sz) * value.y +
            (cz * sx + cx * sy * sz) * value.z,
        sy * value.x - cy * sx * value.y + cx * cy * value.z
    );
}

inline float sourceValue(
    thread const MRMillardSourceCurveGPU& curve,
    const uint index
) {
    return curve.values[index >> 2u][index & 3u];
}

inline float bezier5(
    const float a, const float b, const float c, const float d,
    const float e, const float f, const float u
) {
    const float v = 1.0f - u;
    return a * v * v * v * v * v + 5.0f * b * u * v * v * v * v +
        10.0f * c * u * u * v * v * v + 10.0f * d * u * u * u * v * v +
        5.0f * e * u * u * u * u * v + f * u * u * u * u * u;
}

// Evaluate one quintic Bezier corner without allocating a dynamic curve
// object. This exactly mirrors SmoothSegmentedFunctionFactory's corner
// control construction used by the host source bridge.
inline bool evaluateCorner(
    const float x0, const float y0, const float slope0,
    const float x1, const float y1, const float slope1,
    const float curviness, const float target,
    thread float& value
) {
    if (!(isfinite(x0) && isfinite(y0) && isfinite(slope0) &&
          isfinite(x1) && isfinite(y1) && isfinite(slope1) &&
          isfinite(curviness) && x1 > x0 && curviness >= 0.0f &&
          curviness <= 1.0f && target >= x0 && target <= x1)) {
        return false;
    }
    const float difference = slope0 - slope1;
    const float intersectionX = abs(difference) > 1.0e-8f
        ? (y1 - y0 - x1 * slope1 + x0 * slope0) / difference
        : 0.5f * (x0 + x1);
    const float intersectionY = (intersectionX - x1) * slope1 + y1;
    const float dx0 = intersectionX - x0;
    const float dy0 = intersectionY - y0;
    const float dx1 = intersectionX - x1;
    const float dy1 = intersectionY - y1;
    const float endpointDx = x1 - x0;
    const float endpointDy = y1 - y0;
    const float endpointDistance2 = endpointDx * endpointDx + endpointDy * endpointDy;
    if (!(endpointDistance2 > dx0 * dx0 + dy0 * dy0) ||
        !(endpointDistance2 > dx1 * dx1 + dy1 * dy1)) {
        return false;
    }
    const float x0Mid = x0 + curviness * dx0;
    const float y0Mid = y0 + curviness * dy0;
    const float x1Mid = x1 + curviness * dx1;
    const float y1Mid = y1 + curviness * dy1;
    float lower = 0.0f;
    float upper = 1.0f;
    for (uint iteration = 0u; iteration < 40u; ++iteration) {
        const float middle = 0.5f * (lower + upper);
        const float x = bezier5(x0, x0Mid, x0Mid, x1Mid, x1Mid, x1, middle);
        if (x < target) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const float u = 0.5f * (lower + upper);
    value = bezier5(y0, y0Mid, y0Mid, y1Mid, y1Mid, y1, u);
    return isfinite(value);
}

inline bool evaluateActiveCurve(
    thread const MRMillardSourceCurveGPU& curve,
    const float x, thread float& result
) {
    const float x0 = sourceValue(curve, 0u);
    const float x1 = sourceValue(curve, 1u);
    const float x2 = 1.0f;
    const float x3 = sourceValue(curve, 2u);
    const float slope = sourceValue(curve, 3u);
    const float yLow = sourceValue(curve, 4u);
    if (!(x0 >= 0.0f && x1 > x0 && x2 > x1 && x3 > x2 && yLow >= 0.0f &&
          slope >= 0.0f && slope < (1.0f - yLow) / (x2 - x1))) {
        return false;
    }
    if (x < x0 || x > x3) {
        result = yLow;
        return true;
    }
    const float c = 0.9f;
    const float xDelta = 0.05f * x2;
    const float shoulderX = x2 - xDelta;
    const float y1 = 1.0f - slope * (shoulderX - x1);
    const float slope01 = 1.25f * y1 / (x1 - x0);
    const float x01 = x0 + 0.5f * (x1 - x0);
    const float y01 = 0.5f * y1;
    const float x1s = x1 + 0.5f * (shoulderX - x1);
    const float y1s = y1 + 0.5f * (1.0f - y1);
    const float x23 = (x2 + xDelta) + 0.5f * (x3 - (x2 + xDelta));
    const float y23 = 0.5f;
    const float slope23 = -1.0f / ((x3 - xDelta) - (x2 + xDelta));
    if (x <= x01) {
        return evaluateCorner(x0, yLow, 0.0f, x01, y01, slope01, c, x, result);
    }
    if (x <= x1s) {
        return evaluateCorner(x01, y01, slope01, x1s, y1s, slope, c, x, result);
    }
    if (x <= x2) {
        return evaluateCorner(x1s, y1s, slope, x2, 1.0f, 0.0f, c, x, result);
    }
    if (x <= x23) {
        return evaluateCorner(x2, 1.0f, 0.0f, x23, y23, slope23, c, x, result);
    }
    return evaluateCorner(x23, y23, slope23, x3, yLow, 0.0f, c, x, result);
}

inline bool evaluateForceVelocityCurve(
    thread const MRMillardSourceCurveGPU& curve,
    const float x, thread float& result
) {
    const float concentricAtMax = sourceValue(curve, 5u);
    const float concentricNearMax = sourceValue(curve, 6u);
    const float isometric = sourceValue(curve, 7u);
    const float eccentricAtMax = sourceValue(curve, 8u);
    const float eccentricNearMax = sourceValue(curve, 9u);
    const float fmax = sourceValue(curve, 10u);
    const float concentricCurviness = sourceValue(curve, 11u);
    const float eccentricCurviness = sourceValue(curve, 12u);
    if (!(fmax > 1.0f && concentricAtMax >= 0.0f && concentricAtMax < 1.0f &&
          concentricNearMax > concentricAtMax && concentricNearMax <= 1.0f &&
          isometric > 1.0f && eccentricAtMax >= 0.0f &&
          eccentricAtMax < fmax - 1.0f && eccentricNearMax >= eccentricAtMax &&
          eccentricNearMax < fmax - 1.0f && concentricCurviness >= 0.0f &&
          concentricCurviness <= 1.0f && eccentricCurviness >= 0.0f &&
          eccentricCurviness <= 1.0f)) {
        return false;
    }
    if (x < -1.0f) {
        result = concentricAtMax * (x + 1.0f);
        return true;
    }
    if (x > 1.0f) {
        result = fmax + eccentricAtMax * (x - 1.0f);
        return true;
    }
    const float cConcentric = 0.1f + 0.8f * concentricCurviness;
    const float cEccentric = 0.1f + 0.8f * eccentricCurviness;
    const float xNearConcentric = -0.9f;
    const float yNearConcentric = 0.5f * (concentricNearMax + concentricAtMax) *
        (xNearConcentric + 1.0f);
    const float xNearEccentric = 0.9f;
    const float yNearEccentric = fmax + 0.5f * (eccentricNearMax + eccentricAtMax) *
        (xNearEccentric - 1.0f);
    if (x <= xNearConcentric) {
        return evaluateCorner(-1.0f, 0.0f, concentricAtMax,
            xNearConcentric, yNearConcentric, concentricNearMax,
            cConcentric, x, result);
    }
    if (x <= 0.0f) {
        return evaluateCorner(xNearConcentric, yNearConcentric,
            concentricNearMax, 0.0f, 1.0f, isometric, cConcentric, x, result);
    }
    if (x <= xNearEccentric) {
        return evaluateCorner(0.0f, 1.0f, isometric, xNearEccentric,
            yNearEccentric, eccentricNearMax, cEccentric, x, result);
    }
    return evaluateCorner(xNearEccentric, yNearEccentric, eccentricNearMax,
        1.0f, fmax, eccentricAtMax, cEccentric, x, result);
}

inline bool evaluatePassiveCurve(
    thread const MRMillardSourceCurveGPU& curve,
    const float x, thread float& result
) {
    const float eZero = sourceValue(curve, 13u);
    const float eIso = sourceValue(curve, 14u);
    const float lowStiffness = sourceValue(curve, 15u);
    const float isoStiffness = sourceValue(curve, 16u);
    const float curviness = sourceValue(curve, 17u);
    if (!(eIso > eZero && isoStiffness > 1.0f / (eIso - eZero) &&
          lowStiffness > 0.0f && lowStiffness < 1.0f / (eIso - eZero) &&
          curviness >= 0.0f && curviness <= 1.0f)) {
        return false;
    }
    const float xZero = 1.0f + eZero;
    const float xIso = 1.0f + eIso;
    if (x < xZero) {
        result = 0.0f;
        return true;
    }
    if (x > xIso) {
        result = 1.0f + isoStiffness * (x - xIso);
        return true;
    }
    const float deltaX = min(0.1f / isoStiffness, 0.1f * (xIso - xZero));
    const float xLow = xZero + deltaX;
    const float xFoot = xZero + 0.5f * (xLow - xZero);
    const float yLow = lowStiffness * (xLow - xFoot);
    const float c = 0.1f + 0.8f * curviness;
    if (x <= xLow) {
        return evaluateCorner(xZero, 0.0f, 0.0f, xLow, yLow,
            lowStiffness, c, x, result);
    }
    return evaluateCorner(xLow, yLow, lowStiffness, xIso, 1.0f,
        isoStiffness, c, x, result);
}

inline bool evaluateTendonCurve(
    thread const MRMillardSourceCurveGPU& curve,
    const float x, thread float& result
) {
    const float eIso = sourceValue(curve, 18u);
    const float isoStiffness = sourceValue(curve, 19u);
    const float toeForce = sourceValue(curve, 20u);
    const float curviness = sourceValue(curve, 21u);
    if (!(eIso > 0.0f && toeForce > 0.0f && toeForce < 1.0f &&
          isoStiffness > 1.0f / eIso && curviness >= 0.0f && curviness <= 1.0f)) {
        return false;
    }
    const float xIso = 1.0f + eIso;
    const float xToe = (toeForce - 1.0f) / isoStiffness + xIso;
    if (x < 1.0f) {
        result = 0.0f;
        return true;
    }
    if (x > xToe) {
        result = toeForce + isoStiffness * (x - xToe);
        return true;
    }
    const float xFoot = 1.0f + (xToe - 1.0f) / 10.0f;
    const float yToeMid = 0.5f * toeForce;
    const float xToeMid = (yToeMid - 1.0f) / isoStiffness + xIso;
    const float toeMidSlope = yToeMid / (xToeMid - xFoot);
    const float xToeControl = xFoot + 0.5f * (xToeMid - xFoot);
    const float yToeControl = toeMidSlope * (xToeControl - xFoot);
    const float c = 0.1f + 0.8f * curviness;
    if (x <= xToeControl) {
        return evaluateCorner(1.0f, 0.0f, 0.0f, xToeControl,
            yToeControl, toeMidSlope, c, x, result);
    }
    return evaluateCorner(xToeControl, yToeControl, toeMidSlope,
        xToe, toeForce, isoStiffness, c, x, result);
}

inline bool evaluateSourceCurves(
    thread const MRMillardSourceCurveGPU& curve,
    const float normalizedFiberLength,
    const float normalizedFiberVelocity,
    const float normalizedTendonLength,
    thread float4& values
) {
    float active = 0.0f;
    float velocity = 0.0f;
    float passive = 0.0f;
    float tendon = 0.0f;
    if (!(isfinite(normalizedFiberLength) && isfinite(normalizedFiberVelocity) &&
          isfinite(normalizedTendonLength)) ||
        !evaluateActiveCurve(curve, normalizedFiberLength, active) ||
        !evaluateForceVelocityCurve(curve, normalizedFiberVelocity, velocity) ||
        !evaluatePassiveCurve(curve, normalizedFiberLength, passive) ||
        !evaluateTendonCurve(curve, normalizedTendonLength, tendon)) {
        return false;
    }
    values = max(float4(active, velocity, passive, tendon), 0.0f);
    return finite4(values);
}

struct SegmentEvaluation {
    float length;
    float3 gradientFirst;
    float3 gradientSecond;
    bool wrapped;
};

inline bool wrappedSegment(
    const float3 firstWorld,
    const float3 secondWorld,
    const MRArticulatedBodyPoseGPU body,
    const MRMillardCylinderWrapGPU wrap,
    thread SegmentEvaluation& result
) {
    if (!(wrap.rotationAndRadius.w > 0.0f && wrap.length.x > 0.0f &&
          finite3(firstWorld) && finite3(secondWorld) && finite4(body.position) &&
          finite4(body.orientation) && finite4(wrap.center) &&
          finite4(wrap.rotationAndRadius) && finite4(wrap.length))) {
        return false;
    }
    const float4 inverseBody = quaternionConjugate(body.orientation);
    const float3 first = bodyFixedXYZTransposeRotate(
        wrap.rotationAndRadius.xyz,
        quaternionRotate(inverseBody, firstWorld - body.position.xyz) - wrap.center.xyz
    );
    const float3 second = bodyFixedXYZTransposeRotate(
        wrap.rotationAndRadius.xyz,
        quaternionRotate(inverseBody, secondWorld - body.position.xyz) - wrap.center.xyz
    );
    const float3 delta = second - first;
    const float xySquared = dot(delta.xy, delta.xy);
    const float t = xySquared > kGeometryTolerance
        ? clamp(-dot(first.xy, delta.xy) / xySquared, 0.0f, 1.0f)
        : 0.0f;
    const float3 closest = first + t * delta;
    const float radius = wrap.rotationAndRadius.w;
    if (dot(closest.xy, closest.xy) >= radius * radius ||
        abs(closest.z) > 0.5f * wrap.length.x) {
        return false;
    }
    const float firstRadius = length(first.xy);
    const float secondRadius = length(second.xy);
    if (!(firstRadius > radius + kGeometryTolerance &&
          secondRadius > radius + kGeometryTolerance)) {
        return false;
    }
    const float firstAxisAngle = atan2(first.y, first.x);
    const float secondAxisAngle = atan2(second.y, second.x);
    const float firstTangent = acos(radius / firstRadius);
    const float secondTangent = acos(radius / secondRadius);
    const float2 firstAngles = float2(firstAxisAngle + firstTangent,
        firstAxisAngle - firstTangent);
    const float2 secondAngles = float2(secondAxisAngle + secondTangent,
        secondAxisAngle - secondTangent);
    float bestLength = INFINITY;
    float3 bestFirst = float3(0.0f);
    float3 bestSecond = float3(0.0f);
    for (uint i = 0u; i < 2u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            float difference = fmod(secondAngles[j] - firstAngles[i] + M_PI_F,
                2.0f * M_PI_F);
            if (difference < 0.0f) {
                difference += 2.0f * M_PI_F;
            }
            difference -= M_PI_F;
            const float candidate = sqrt(firstRadius * firstRadius - radius * radius) +
                sqrt(radius * radius * difference * difference +
                    (second.z - first.z) * (second.z - first.z)) +
                sqrt(secondRadius * secondRadius - radius * radius);
            if (candidate < bestLength) {
                bestLength = candidate;
                bestFirst = float3(radius * cos(firstAngles[i]),
                    radius * sin(firstAngles[i]), first.z);
                bestSecond = float3(radius * cos(secondAngles[j]),
                    radius * sin(secondAngles[j]), second.z);
            }
        }
    }
    const float3 tangentFirst = body.position.xyz + quaternionRotate(body.orientation,
        wrap.center.xyz + bodyFixedXYZRotate(wrap.rotationAndRadius.xyz, bestFirst));
    const float3 tangentSecond = body.position.xyz + quaternionRotate(body.orientation,
        wrap.center.xyz + bodyFixedXYZRotate(wrap.rotationAndRadius.xyz, bestSecond));
    const float firstLength = length(firstWorld - tangentFirst);
    const float secondLength = length(secondWorld - tangentSecond);
    if (!(bestLength > kMinimumLength && firstLength > kMinimumLength &&
          secondLength > kMinimumLength)) {
        return false;
    }
    result.length = bestLength;
    result.gradientFirst = (firstWorld - tangentFirst) / firstLength;
    result.gradientSecond = (secondWorld - tangentSecond) / secondLength;
    result.wrapped = finite3(result.gradientFirst) && finite3(result.gradientSecond);
    return result.wrapped;
}

inline bool directSegment(
    const float3 first, const float3 second, thread SegmentEvaluation& result
) {
    const float3 direction = second - first;
    const float pathLength = length(direction);
    if (!(pathLength > kMinimumLength) || !isfinite(pathLength)) {
        return false;
    }
    result.length = pathLength;
    result.gradientFirst = -direction / pathLength;
    result.gradientSecond = direction / pathLength;
    result.wrapped = false;
    return true;
}

inline bool evaluateStaticState(
    thread const MRMillardMuscleGPU& muscle,
    thread const MRMillardSourceCurveGPU& curve,
    const float activation,
    const float normalizedVelocity,
    const float pathLength,
    const float fiberLength,
    thread float& tendonForce,
    thread float& residual
) {
    const float maximumForce = muscle.forceAndLengths.x;
    const float optimalLength = muscle.forceAndLengths.y;
    const float slackLength = muscle.forceAndLengths.z;
    const float pennation = muscle.forceAndLengths.w;
    const float thickness = optimalLength * sin(pennation);
    if (!(maximumForce > 0.0f && optimalLength > 0.0f && slackLength > 0.0f &&
          fiberLength > thickness && activation >= muscle.dampingAndActivation.z &&
          activation <= 1.0f)) {
        return false;
    }
    const float sinePennation = thickness / fiberLength;
    if (!(abs(sinePennation) < 1.0f)) {
        return false;
    }
    const float cosinePennation = sqrt(1.0f - sinePennation * sinePennation);
    const float tendonLength = pathLength - fiberLength * cosinePennation;
    float4 curves;
    if (!evaluateSourceCurves(curve, fiberLength / optimalLength,
            normalizedVelocity, tendonLength / slackLength, curves)) {
        return false;
    }
    const float fiberForce = maximumForce * (activation * curves.x * curves.y +
        curves.z + muscle.dampingAndActivation.x * normalizedVelocity);
    tendonForce = maximumForce * curves.w;
    residual = fiberForce * cosinePennation - tendonForce;
    return isfinite(fiberForce) && isfinite(tendonForce) && isfinite(residual) &&
        tendonForce >= 0.0f;
}

inline bool solveStaticEquilibrium(
    thread const MRMillardMuscleGPU& muscle,
    thread const MRMillardSourceCurveGPU& curve,
    const float activation,
    const float normalizedVelocity,
    const float pathLength,
    thread float& fiberLength,
    thread float& tendonForce,
    thread float& residual
) {
    const float optimalLength = muscle.forceAndLengths.y;
    const float thickness = optimalLength * sin(muscle.forceAndLengths.w);
    const float minimumFiberLength = max(
        sourceValue(curve, 0u) * optimalLength,
        thickness / sqrt(1.0f - kStaticFiberMargin)
    );
    float lower = minimumFiberLength * (1.0f + 1.0e-5f);
    float upper = max(lower * 1.01f, pathLength + optimalLength);
    float lowerTendon = 0.0f;
    float lowerResidual = 0.0f;
    float upperTendon = 0.0f;
    float upperResidual = 0.0f;
    if (!evaluateStaticState(muscle, curve, activation, normalizedVelocity,
            pathLength, lower, lowerTendon, lowerResidual) ||
        !evaluateStaticState(muscle, curve, activation, normalizedVelocity,
            pathLength, upper, upperTendon, upperResidual) ||
        lowerResidual > 0.0f || upperResidual < 0.0f) {
        return false;
    }
    fiberLength = lower;
    tendonForce = lowerTendon;
    residual = lowerResidual;
    for (uint iteration = 0u; iteration < 48u; ++iteration) {
        const float middle = 0.5f * (lower + upper);
        float middleTendon = 0.0f;
        float middleResidual = 0.0f;
        if (!evaluateStaticState(muscle, curve, activation, normalizedVelocity,
                pathLength, middle, middleTendon, middleResidual)) {
            return false;
        }
        if (middleResidual < 0.0f) {
            lower = middle;
        } else {
            upper = middle;
        }
        fiberLength = middle;
        tendonForce = middleTendon;
        residual = middleResidual;
    }
    return isfinite(fiberLength) && isfinite(tendonForce) && isfinite(residual);
}

} // namespace

kernel void mr_millard_reference(
    device const MRArticulatedOperatorDispatchGPU& operatorDispatch [[buffer(5)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(8)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(9)]],
    device const float* pointJacobians [[buffer(11)]],
    device const MRMillardReferenceDispatchGPU& dispatch [[buffer(16)]],
    device const MRMillardMuscleGPU* muscles [[buffer(17)]],
    device const MRMillardMuscleStateGPU* states [[buffer(18)]],
    device const MRMillardPathPointGPU* pathPoints [[buffer(19)]],
    device const MRMillardSourceCurveGPU* curves [[buffer(20)]],
    device const MRMillardCylinderWrapGPU* wraps [[buffer(21)]],
    device MRMillardMuscleResultGPU* results [[buffer(22)]],
    device float* generalizedForces [[buffer(23)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.muscleCount == 0u ||
        globalIndex >= dispatch.environmentCount * dispatch.muscleCount) {
        return;
    }
    const uint environment = globalIndex / dispatch.muscleCount;
    const uint muscleIndex = globalIndex - environment * dispatch.muscleCount;
    const uint forceBase = globalIndex * dispatch.dofCount;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        generalizedForces[forceBase + dof] = 0.0f;
    }
    MRMillardMuscleResultGPU result{};
    result.status = MR_MILLARD_REFERENCE_INVALID_PROGRAM;
    result.environment = environment;
    result.muscleIndex = muscleIndex;
    if (dispatch.abiVersion != MR_MILLARD_REFERENCE_GPU_ABI_VERSION ||
        dispatch.environmentCount != operatorDispatch.environmentCount ||
        dispatch.dofCount != operatorDispatch.generalizedStride ||
        dispatch.pointWorldStride != operatorDispatch.pointWorldStride ||
        dispatch.pointJacobianStride != operatorDispatch.pointJacobianStride ||
        dispatch.bodyPoseStride != operatorDispatch.bodyPoseStride ||
        muscleIndex >= dispatch.muscleCount) {
        results[globalIndex] = result;
        return;
    }
    const MRMillardMuscleGPU muscle = muscles[muscleIndex];
    const uint pathOffset = muscle.pathAndWrap.x;
    const uint pathCount = muscle.pathAndWrap.y;
    const uint wrapOffset = muscle.pathAndWrap.z;
    const uint wrapCount = muscle.pathAndWrap.w;
    if (pathCount < 2u || pathOffset > dispatch.pathPointCount ||
        pathCount > dispatch.pathPointCount - pathOffset ||
        wrapCount > MR_MILLARD_REFERENCE_MAX_WRAPS_PER_MUSCLE ||
        wrapOffset > dispatch.wrapCount || wrapCount > dispatch.wrapCount - wrapOffset ||
        !finite4(muscle.forceAndLengths) || !finite4(muscle.dampingAndActivation) ||
        muscle.dampingAndActivation.w != 0.0f || muscle.flags.y != 0u ||
        muscle.flags.z != 0u || muscle.flags.w != 0u) {
        results[globalIndex] = result;
        return;
    }
    const MRMillardMuscleStateGPU state = states[globalIndex];
    const float activation = max(state.activationAndVelocity.x,
        muscle.dampingAndActivation.z);
    const float normalizedVelocity = state.activationAndVelocity.y;
    if (!(isfinite(activation) && isfinite(normalizedVelocity) && activation <= 1.0f &&
          state.activationAndVelocity.z == 0.0f && state.activationAndVelocity.w == 0.0f)) {
        result.status = MR_MILLARD_REFERENCE_INVALID_STATE;
        results[globalIndex] = result;
        return;
    }
    bool usedWrap[MR_MILLARD_REFERENCE_MAX_WRAPS_PER_MUSCLE]{};
    float totalLength = 0.0f;
    uint appliedWrapCount = 0u;
    bool validPath = true;
    bool projectedUsedWrap[MR_MILLARD_REFERENCE_MAX_WRAPS_PER_MUSCLE]{};
    for (uint segment = 0u; segment + 1u < pathCount; ++segment) {
        const MRMillardPathPointGPU firstPoint = pathPoints[pathOffset + segment];
        const MRMillardPathPointGPU secondPoint = pathPoints[pathOffset + segment + 1u];
        if (firstPoint.pointQueryIndex >= dispatch.pointWorldStride ||
            secondPoint.pointQueryIndex >= dispatch.pointWorldStride ||
            firstPoint.reserved0 != 0u || firstPoint.reserved1 != 0u ||
            secondPoint.reserved0 != 0u || secondPoint.reserved1 != 0u) {
            validPath = false;
            break;
        }
        const float3 firstWorld = pointWorld[
            environment * dispatch.pointWorldStride + firstPoint.pointQueryIndex
        ].position.xyz;
        const float3 secondWorld = pointWorld[
            environment * dispatch.pointWorldStride + secondPoint.pointQueryIndex
        ].position.xyz;
        SegmentEvaluation evaluation{};
        bool wrapped = false;
        for (uint wrapLocal = 0u; wrapLocal < wrapCount; ++wrapLocal) {
            if (usedWrap[wrapLocal]) {
                continue;
            }
            const MRMillardCylinderWrapGPU wrap = wraps[wrapOffset + wrapLocal];
            if (wrap.bodyIndex < dispatch.articulationFirstBody ||
                wrap.bodyIndex >= dispatch.articulationFirstBody + dispatch.bodyPoseStride ||
                wrap.reserved0 != 0u || wrap.reserved1 != 0u || wrap.reserved2 != 0u ||
                wrap.center.w != 0.0f || wrap.length.y != 0.0f ||
                wrap.length.z != 0.0f || wrap.length.w != 0.0f) {
                validPath = false;
                break;
            }
            const uint localBody = wrap.bodyIndex - dispatch.articulationFirstBody;
            if (wrappedSegment(firstWorld, secondWorld,
                    bodyPoses[environment * dispatch.bodyPoseStride + localBody], wrap,
                    evaluation)) {
                usedWrap[wrapLocal] = true;
                wrapped = true;
                ++appliedWrapCount;
                break;
            }
        }
        if (!validPath || (!wrapped && !directSegment(firstWorld, secondWorld, evaluation))) {
            validPath = false;
            break;
        }
        totalLength += evaluation.length;
    }
    if (!validPath || !(totalLength > kMinimumLength) || !isfinite(totalLength)) {
        result.status = MR_MILLARD_REFERENCE_INVALID_PATH;
        results[globalIndex] = result;
        return;
    }
    float fiberLength = 0.0f;
    float tendonForce = 0.0f;
    float residual = 0.0f;
    const MRMillardSourceCurveGPU curve = curves[muscleIndex];
    if (!solveStaticEquilibrium(muscle, curve, activation,
            normalizedVelocity, totalLength, fiberLength, tendonForce, residual)) {
        result.status = MR_MILLARD_REFERENCE_UNBRACKETED;
        result.pathFiberTendonResidual = float4(totalLength, fiberLength, tendonForce, residual);
        results[globalIndex] = result;
        return;
    }
    for (uint segment = 0u; segment + 1u < pathCount; ++segment) {
        const MRMillardPathPointGPU firstPoint = pathPoints[pathOffset + segment];
        const MRMillardPathPointGPU secondPoint = pathPoints[pathOffset + segment + 1u];
        const float3 firstWorld = pointWorld[
            environment * dispatch.pointWorldStride + firstPoint.pointQueryIndex
        ].position.xyz;
        const float3 secondWorld = pointWorld[
            environment * dispatch.pointWorldStride + secondPoint.pointQueryIndex
        ].position.xyz;
        SegmentEvaluation evaluation{};
        bool wrapped = false;
        for (uint wrapLocal = 0u; wrapLocal < wrapCount; ++wrapLocal) {
            if (projectedUsedWrap[wrapLocal]) {
                continue;
            }
            const MRMillardCylinderWrapGPU wrap = wraps[wrapOffset + wrapLocal];
            const uint localBody = wrap.bodyIndex - dispatch.articulationFirstBody;
            if (wrappedSegment(firstWorld, secondWorld,
                    bodyPoses[environment * dispatch.bodyPoseStride + localBody], wrap,
                    evaluation)) {
                projectedUsedWrap[wrapLocal] = true;
                wrapped = true;
                break;
            }
        }
        if (!wrapped) {
            if (!directSegment(firstWorld, secondWorld, evaluation)) {
                result.status = MR_MILLARD_REFERENCE_INVALID_PATH;
                results[globalIndex] = result;
                return;
            }
        }
        for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
            float derivative = 0.0f;
            for (uint axis = 0u; axis < 3u; ++axis) {
                derivative += evaluation.gradientFirst[axis] * pointJacobians[
                    environment * dispatch.pointJacobianStride +
                    (firstPoint.pointQueryIndex * 3u + axis) * dispatch.dofCount + dof
                ];
                derivative += evaluation.gradientSecond[axis] * pointJacobians[
                    environment * dispatch.pointJacobianStride +
                    (secondPoint.pointQueryIndex * 3u + axis) * dispatch.dofCount + dof
                ];
            }
            generalizedForces[forceBase + dof] -= tendonForce * derivative;
        }
    }
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        if (!isfinite(generalizedForces[forceBase + dof])) {
            result.status = MR_MILLARD_REFERENCE_NONFINITE_RESULT;
            results[globalIndex] = result;
            return;
        }
    }
    result.status = MR_MILLARD_REFERENCE_SUCCESS;
    result.appliedCylinderWrapCount = appliedWrapCount;
    result.pathFiberTendonResidual = float4(totalLength, fiberLength, tendonForce, residual);
    results[globalIndex] = result;
}
