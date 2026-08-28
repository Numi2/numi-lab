#include <metal_stdlib>

#include "metalrobo/mujoco_muscle_gpu.h"

using namespace metal;

namespace {

constant float kMinimum = 1.0e-7f;
constant float kPi = 3.14159265358979323846f;

inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline float3 quaternionConjugateRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 conjugate = -quaternion.xyz;
    const float3 doubledCross = 2.0f * cross(conjugate, value);
    return value + quaternion.w * doubledCross + cross(conjugate, doubledCross);
}

inline float3 quaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}

inline float3 matrixApply(
    const MRMujocoMuscleWrapGPU wrap,
    const float3 value
) {
    return float3(
        dot(wrap.rotationRow0.xyz, value),
        dot(wrap.rotationRow1.xyz, value),
        dot(wrap.rotationRow2.xyz, value)
    );
}

inline float3 matrixTransposeApply(
    const MRMujocoMuscleWrapGPU wrap,
    const float3 value
) {
    return wrap.rotationRow0.xyz * value.x +
        wrap.rotationRow1.xyz * value.y +
        wrap.rotationRow2.xyz * value.z;
}

inline bool segmentIntersects(
    const float2 firstA,
    const float2 secondA,
    const float2 firstB,
    const float2 secondB
) {
    const float determinant =
        (secondB.y - firstB.y) * (secondA.x - firstA.x) -
        (secondB.x - firstB.x) * (secondA.y - firstA.y);
    if (abs(determinant) < kMinimum) {
        return false;
    }
    const float a = ((secondB.x - firstB.x) * (firstA.y - firstB.y) -
                     (secondB.y - firstB.y) * (firstA.x - firstB.x)) /
        determinant;
    const float b = ((secondA.x - firstA.x) * (firstA.y - firstB.y) -
                     (secondA.y - firstA.y) * (firstA.x - firstB.x)) /
        determinant;
    return a >= 0.0f && a <= 1.0f && b >= 0.0f && b <= 1.0f;
}

inline float circleLength(
    const float2 first,
    const float2 second,
    const uint index,
    const float radius
) {
    const float2 unitFirst = normalize(first);
    const float2 unitSecond = normalize(second);
    float angle = acos(clamp(dot(unitFirst, unitSecond), -1.0f, 1.0f));
    const float determinant = first.y * second.x - first.x * second.y;
    if ((determinant > 0.0f && index != 0u) ||
        (determinant < 0.0f && index == 0u)) {
        angle = 2.0f * kPi - angle;
    }
    return radius * angle;
}

inline bool wrapCircle(
    const float2 first,
    const float2 second,
    const bool hasSide,
    const float2 side,
    const float radius,
    thread float4& contact,
    thread float& wrappingLength
) {
    const float squaredFirst = dot(first, first);
    const float squaredSecond = dot(second, second);
    const float squaredRadius = radius * radius;
    if (squaredFirst < squaredRadius || squaredSecond < squaredRadius ||
        radius < kMinimum) {
        return false;
    }
    const float2 difference = second - first;
    const float differenceSquared = dot(difference, difference);
    if (differenceSquared < kMinimum) {
        return false;
    }
    const float fraction = clamp(-dot(difference, first) / differenceSquared,
                                 0.0f, 1.0f);
    const float2 nearest = fraction * difference + first;
    if (dot(nearest, nearest) > squaredRadius &&
        (!hasSide || dot(side, nearest) >= 0.0f)) {
        return false;
    }
    const float rootFirst = sqrt(max(0.0f, squaredFirst - squaredRadius));
    const float rootSecond = sqrt(max(0.0f, squaredSecond - squaredRadius));
    float2 firstSolutions[2];
    float2 secondSolutions[2];
    float goodness[2];
    for (uint index = 0u; index < 2u; ++index) {
        const float sign = index == 0u ? 1.0f : -1.0f;
        firstSolutions[index] = float2(
            (first.x * squaredRadius + sign * radius * first.y * rootFirst) / squaredFirst,
            (first.y * squaredRadius - sign * radius * first.x * rootFirst) / squaredFirst
        );
        secondSolutions[index] = float2(
            (second.x * squaredRadius - sign * radius * second.y * rootSecond) / squaredSecond,
            (second.y * squaredRadius + sign * radius * second.x * rootSecond) / squaredSecond
        );
        if (hasSide) {
            goodness[index] = dot(
                normalize(firstSolutions[index] + secondSolutions[index]), side
            );
        } else {
            const float2 delta = firstSolutions[index] - secondSolutions[index];
            goodness[index] = -dot(delta, delta);
        }
        if (segmentIntersects(first, firstSolutions[index], second,
                              secondSolutions[index])) {
            goodness[index] = -10000.0f;
        }
    }
    const uint selected = goodness[0] > goodness[1] ? 0u : 1u;
    contact = float4(firstSolutions[selected], secondSolutions[selected]);
    if (segmentIntersects(first, contact.xy, second, contact.zw)) {
        return false;
    }
    wrappingLength = circleLength(contact.xy, contact.zw, selected, radius);
    return isfinite(wrappingLength);
}

inline bool wrapInside(
    const float2 first,
    const float2 second,
    const float radius,
    thread float4& contact
) {
    const float firstLength = length(first);
    const float secondLength = length(second);
    if (firstLength <= radius || secondLength <= radius ||
        radius < kMinimum || firstLength < kMinimum || secondLength < kMinimum) {
        return false;
    }
    const float2 difference = second - first;
    const float differenceSquared = dot(difference, difference);
    if (differenceSquared > kMinimum) {
        const float fraction = -dot(first, difference) / differenceSquared;
        if (fraction > 0.0f && fraction < 1.0f &&
            length(first + fraction * difference) <= radius) {
            return false;
        }
    }
    float2 midpoint = first + second;
    if (length(midpoint) < kMinimum) {
        return false;
    }
    midpoint = radius * normalize(midpoint);
    const float a = radius / firstLength;
    const float b = radius / secondLength;
    const float cosine = (firstLength * firstLength + secondLength * secondLength -
                          differenceSquared) / (2.0f * firstLength * secondLength);
    contact = float4(midpoint, midpoint);
    if (cosine < -1.0f + kMinimum) {
        return false;
    }
    if (cosine > 1.0f - kMinimum) return true;
    float z = 1.0f - 1.0e-7f;
    float residual = asin(a * z) + asin(b * z) - 2.0f * asin(z) + acos(cosine);
    if (residual > 0.0f) return true;
    uint iteration = 0u;
    for (; iteration < 20u && abs(residual) > 1.0e-6f; ++iteration) {
        const float derivative =
            a / max(kMinimum, sqrt(max(0.0f, 1.0f - z * z * a * a))) +
            b / max(kMinimum, sqrt(max(0.0f, 1.0f - z * z * b * b))) -
            2.0f / max(kMinimum, sqrt(max(0.0f, 1.0f - z * z)));
        if (derivative > -kMinimum) return true;
        const float next = z - residual / derivative;
        if (next > z) return true;
        z = next;
        residual = asin(a * z) + asin(b * z) - 2.0f * asin(z) + acos(cosine);
        if (residual > 1.0e-6f) return true;
    }
    if (iteration >= 20u) return true;
    const bool firstSide = first.x * second.y - first.y * second.x > 0.0f;
    const float2 vector = normalize(firstSide ? first : second);
    const float rotation = firstSide
        ? asin(z) - asin(a * z)
        : asin(z) - asin(b * z);
    const float2 point = radius * float2(
        cos(rotation) * vector.x - sin(rotation) * vector.y,
        sin(rotation) * vector.x + cos(rotation) * vector.y
    );
    contact = float4(point, point);
    return true;
}

inline bool siteWorld(
    const uint environment,
    const MRMujocoMuscleReferenceDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const MRMujocoMuscleSiteGPU* sites,
    const uint index,
    thread float3& world
) {
    if (index >= dispatch.siteCount) return false;
    const MRMujocoMuscleSiteGPU site = sites[index];
    if (site.bodyIndex < dispatch.articulationFirstBody ||
        site.bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride ||
        site.reserved0 != 0u || site.reserved1 != 0u || site.reserved2 != 0u ||
        !finite4(site.localPoint) || site.localPoint.w != 0.0f) return false;
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride +
        site.bodyIndex - dispatch.articulationFirstBody
    ];
    if (!finite4(pose.position) || !finite4(pose.orientation)) return false;
    world = pose.position.xyz + quaternionRotate(pose.orientation, site.localPoint.xyz);
    return all(isfinite(world));
}

// The enclosing articulated pass supplies four analytic point-Jacobian
// probes for every body: COM followed by body-local unit X/Y/Z.  Their
// difference reconstructs angular velocity Jacobians at an arbitrary current
// world point, so route tension can be projected without CPU-restaged poses,
// finite differencing, or a parallel kinematics implementation.
inline bool addPointLengthGradient(
    const uint environment,
    const MRMujocoMuscleReferenceDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float* pointJacobians,
    const uint bodyIndex,
    const float3 worldPoint,
    const float3 gradient,
    device float* lengthJacobian
) {
    if (bodyIndex < dispatch.articulationFirstBody ||
        bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride ||
        dispatch.dofCount == 0u || dispatch.pointJacobianStride == 0u ||
        dispatch.bodyJacobianPointStride != 4u ||
        dispatch.bodyJacobianPointOffset >
            dispatch.pointJacobianStride / (3u * dispatch.dofCount)) {
        return false;
    }
    const uint localBody = bodyIndex - dispatch.articulationFirstBody;
    const uint bodyPoint = dispatch.bodyJacobianPointOffset +
        localBody * dispatch.bodyJacobianPointStride;
    const uint pointCount = dispatch.pointJacobianStride /
        (3u * dispatch.dofCount);
    if (bodyPoint > pointCount ||
        dispatch.bodyJacobianPointStride > pointCount - bodyPoint) {
        return false;
    }
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody
    ];
    if (!finite4(pose.position) || !finite4(pose.orientation) ||
        !all(isfinite(worldPoint)) || !all(isfinite(gradient))) {
        return false;
    }
    const uint environmentBase = environment * dispatch.pointJacobianStride;
    const uint centerBase = environmentBase +
        bodyPoint * 3u * dispatch.dofCount;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        const float3 centerJacobian = float3(
            pointJacobians[centerBase + dof],
            pointJacobians[centerBase + dispatch.dofCount + dof],
            pointJacobians[centerBase + 2u * dispatch.dofCount + dof]
        );
        float3 angularJacobian = float3(0.0f);
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float3 localAxis = axis == 0u
                ? float3(1.0f, 0.0f, 0.0f)
                : (axis == 1u
                    ? float3(0.0f, 1.0f, 0.0f)
                    : float3(0.0f, 0.0f, 1.0f));
            const float3 worldAxis = quaternionRotate(
                pose.orientation,
                localAxis
            );
            const uint axisBase = centerBase +
                (axis + 1u) * 3u * dispatch.dofCount;
            const float3 axisJacobian = float3(
                pointJacobians[axisBase + dof],
                pointJacobians[axisBase + dispatch.dofCount + dof],
                pointJacobians[axisBase + 2u * dispatch.dofCount + dof]
            );
            angularJacobian += 0.5f * cross(
                worldAxis,
                axisJacobian - centerJacobian
            );
        }
        const float3 pointJacobian = centerJacobian + cross(
            angularJacobian,
            worldPoint - pose.position.xyz
        );
        if (!all(isfinite(pointJacobian))) return false;
        lengthJacobian[dof] += dot(gradient, pointJacobian);
    }
    return true;
}

inline bool addSegmentLengthJacobian(
    const uint environment,
    const MRMujocoMuscleReferenceDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float* pointJacobians,
    const uint firstBody,
    const float3 firstWorld,
    const uint secondBody,
    const float3 secondWorld,
    device float* lengthJacobian
) {
    const float3 difference = secondWorld - firstWorld;
    const float distance = length(difference);
    // Match the FP64 reference: coincident source points retain zero path
    // length but do not introduce a singular tangent/Jacobian contribution.
    if (!(distance > kMinimum)) return true;
    const float3 direction = difference / distance;
    return addPointLengthGradient(
               environment, dispatch, bodyPoses, pointJacobians,
               firstBody, firstWorld, -direction, lengthJacobian
           ) &&
        addPointLengthGradient(
            environment, dispatch, bodyPoses, pointJacobians,
            secondBody, secondWorld, direction, lengthJacobian
        );
}

inline float gainLength(const float value, const float lower, const float upper) {
    if (value < lower || value > upper) return 0.0f;
    const float lowerMid = 0.5f * (lower + 1.0f);
    const float upperMid = 0.5f * (1.0f + upper);
    if (value <= lowerMid) { const float x = (value - lower) / max(kMinimum, lowerMid - lower); return 0.5f * x * x; }
    if (value <= 1.0f) { const float x = (1.0f - value) / max(kMinimum, 1.0f - lowerMid); return 1.0f - 0.5f * x * x; }
    if (value <= upperMid) { const float x = (value - 1.0f) / max(kMinimum, upperMid - 1.0f); return 1.0f - 0.5f * x * x; }
    const float x = (upper - value) / max(kMinimum, upper - upperMid); return 0.5f * x * x;
}

inline float parameter(const mr_float4 blocks[3], const uint index) {
    return blocks[index >> 2u][index & 3u];
}

inline float forceScale(const mr_float4 parameters[3], const float accelerationScale) {
    const float direct = parameter(parameters, 2u);
    return direct < 0.0f ? parameter(parameters, 3u) / max(kMinimum, accelerationScale) : direct;
}

inline float muscleGain(
    const float pathLength,
    const float pathVelocity,
    const MRMujocoMuscleGPU muscle
) {
    const float optimum = (muscle.lengthRangeAndAcceleration.y - muscle.lengthRangeAndAcceleration.x) /
        max(kMinimum, parameter(muscle.gainParameters, 1u) - parameter(muscle.gainParameters, 0u));
    const float normalized = parameter(muscle.gainParameters, 0u) +
        (pathLength - muscle.lengthRangeAndAcceleration.x) / max(kMinimum, optimum);
    const float lengthGain = gainLength(normalized, parameter(muscle.gainParameters, 4u), parameter(muscle.gainParameters, 5u));
    const float normalizedVelocity = pathVelocity / max(
        kMinimum, optimum * parameter(muscle.gainParameters, 6u)
    );
    const float eccentricLimit = parameter(muscle.gainParameters, 8u);
    const float transition = eccentricLimit - 1.0f;
    float velocityGain = 0.0f;
    if (normalizedVelocity <= -1.0f) velocityGain = 0.0f;
    else if (normalizedVelocity <= 0.0f) velocityGain =
        (normalizedVelocity + 1.0f) * (normalizedVelocity + 1.0f);
    else if (normalizedVelocity <= transition) velocityGain =
        eccentricLimit - (transition - normalizedVelocity) *
            (transition - normalizedVelocity) / max(kMinimum, transition);
    else velocityGain = eccentricLimit;
    return -forceScale(muscle.gainParameters, muscle.lengthRangeAndAcceleration.z) *
        lengthGain * velocityGain;
}

inline float normalizedPassiveForce(
    const float normalizedLength,
    const MRMujocoMuscleGPU muscle
) {
    const float upperMid = 0.5f * (1.0f + parameter(muscle.biasParameters, 5u));
    if (normalizedLength <= 1.0f) return 0.0f;
    if (normalizedLength <= upperMid) {
        const float x = (normalizedLength - 1.0f) /
            max(kMinimum, upperMid - 1.0f);
        return parameter(muscle.biasParameters, 7u) * 0.5f * x * x;
    }
    const float x = (normalizedLength - upperMid) /
        max(kMinimum, upperMid - 1.0f);
    return parameter(muscle.biasParameters, 7u) * (0.5f + x);
}

inline float normalizedVelocityGain(
    const float fiberVelocity,
    const float optimalFiberLength,
    const MRMujocoMuscleGPU muscle
) {
    const float normalizedVelocity = fiberVelocity / max(
        kMinimum,
        optimalFiberLength * parameter(muscle.gainParameters, 6u)
    );
    const float eccentricLimit = parameter(muscle.gainParameters, 8u);
    const float transition = eccentricLimit - 1.0f;
    if (normalizedVelocity <= -1.0f) return 0.0f;
    if (normalizedVelocity <= 0.0f) return
        (normalizedVelocity + 1.0f) * (normalizedVelocity + 1.0f);
    if (normalizedVelocity <= transition) return
        eccentricLimit - (transition - normalizedVelocity) *
            (transition - normalizedVelocity) / max(kMinimum, transition);
    return eccentricLimit;
}

inline float normalizedTendonForce(
    const float normalizedLength,
    const MRMujocoMuscleGPU muscle
) {
    const float strain = normalizedLength - 1.0f;
    if (strain <= 0.0f) return 0.0f;
    const float strainAtOne = muscle.compliantArchitecture0.z;
    const float stiffness = muscle.compliantArchitecture0.w;
    const float forceAtToe = muscle.compliantArchitecture1.x;
    const float strainAtToe = strainAtOne - (1.0f - forceAtToe) /
        max(kMinimum, stiffness);
    if (strain >= strainAtToe) {
        return forceAtToe + stiffness * (strain - strainAtToe);
    }
    const float t = clamp(strain / max(kMinimum, strainAtToe), 0.0f, 1.0f);
    return (-2.0f * t * t * t + 3.0f * t * t) * forceAtToe +
        (t * t * t - t * t) * strainAtToe * stiffness;
}

inline bool compliantArchitecture(const MRMujocoMuscleGPU muscle) {
    return muscle.compliantArchitecture0.x > kMinimum &&
        muscle.compliantArchitecture0.y > kMinimum;
}

inline bool solveCompliantFiber(
    const float pathLength,
    const float pathVelocity,
    const float timestep,
    const float activation,
    const MRMujocoMuscleStateGPU state,
    const MRMujocoMuscleGPU muscle,
    thread float& fiberLength,
    thread float& fiberVelocity,
    thread float& tendonForce,
    thread float& equilibriumResidual
) {
    const float optimalFiberLength = muscle.compliantArchitecture0.x;
    const float tendonSlackLength = muscle.compliantArchitecture0.y;
    const float damping = muscle.compliantArchitecture1.z;
    const float predictedPath = max(
        kMinimum, pathLength + max(0.0f, timestep) * pathVelocity
    );
    float acceptedFiber = state.excitationAndActivation.z;
    if (!(acceptedFiber > kMinimum)) {
        const float initializationLower = min(
            0.2f * optimalFiberLength,
            0.5f * pathLength
        );
        acceptedFiber = clamp(
            pathLength - 1.02f * tendonSlackLength,
            initializationLower,
            pathLength
        );
    }
    const float effectiveTimestep = max(1.0e-5f, timestep);
    float lower = min(0.05f * optimalFiberLength, 0.5f * predictedPath);
    float upper = predictedPath;
    if (!(lower < upper)) return false;
    float residual = 0.0f;
    float normalizedTension = 0.0f;
    for (uint iteration = 0u; iteration < 48u; ++iteration) {
        const float candidate = 0.5f * (lower + upper);
        const float velocity = (candidate - acceptedFiber) / effectiveTimestep;
        const float normalizedFiber = candidate / optimalFiberLength;
        normalizedTension = normalizedTendonForce(
            (predictedPath - candidate) / tendonSlackLength, muscle
        );
        const float active = activation * gainLength(
            normalizedFiber,
            parameter(muscle.gainParameters, 4u),
            parameter(muscle.gainParameters, 5u)
        ) * normalizedVelocityGain(velocity, optimalFiberLength, muscle);
        const float passive = normalizedPassiveForce(normalizedFiber, muscle);
        residual = normalizedTension - active - passive -
            damping * velocity / optimalFiberLength;
        if (residual > 0.0f) lower = candidate;
        else upper = candidate;
    }
    fiberLength = 0.5f * (lower + upper);
    fiberVelocity = (fiberLength - acceptedFiber) / effectiveTimestep;
    tendonForce = max(0.0f, normalizedTendonForce(
        (predictedPath - fiberLength) / tendonSlackLength, muscle
    ));
    equilibriumResidual = residual;
    return isfinite(fiberLength) && isfinite(fiberVelocity) &&
        isfinite(tendonForce) && isfinite(equilibriumResidual);
}

inline float muscleBias(const float pathLength, const MRMujocoMuscleGPU muscle) {
    const float optimum = (muscle.lengthRangeAndAcceleration.y - muscle.lengthRangeAndAcceleration.x) /
        max(kMinimum, parameter(muscle.biasParameters, 1u) - parameter(muscle.biasParameters, 0u));
    const float normalized = parameter(muscle.biasParameters, 0u) +
        (pathLength - muscle.lengthRangeAndAcceleration.x) / max(kMinimum, optimum);
    const float upperMid = 0.5f * (1.0f + parameter(muscle.biasParameters, 5u));
    const float scale = forceScale(muscle.biasParameters, muscle.lengthRangeAndAcceleration.z);
    if (normalized <= 1.0f) return 0.0f;
    if (normalized <= upperMid) { const float x = (normalized - 1.0f) / max(kMinimum, upperMid - 1.0f); return -scale * parameter(muscle.biasParameters, 7u) * 0.5f * x * x; }
    const float x = (normalized - upperMid) / max(kMinimum, upperMid - 1.0f);
    return -scale * parameter(muscle.biasParameters, 7u) * (0.5f + x);
}

inline float activationDerivative(
    const MRMujocoMuscleGPU muscle,
    const float excitation,
    const float activation
) {
    const float control = clamp(excitation, 0.0f, 1.0f);
    const float act = clamp(activation, 0.0f, 1.0f);
    const float activationTime = parameter(muscle.dynamicParameters, 0u) * (0.5f + 1.5f * act);
    const float deactivationTime = parameter(muscle.dynamicParameters, 1u) / (0.5f + 1.5f * act);
    const float excess = control - activation;
    const float smoothing = parameter(muscle.dynamicParameters, 2u);
    const float tau = smoothing < kMinimum
        ? (excess > 0.0f ? activationTime : deactivationTime)
        : deactivationTime + (activationTime - deactivationTime) /
            (1.0f + exp(-(excess / smoothing + 0.5f)));
    return excess / max(kMinimum, tau);
}

} // namespace

kernel void mr_mujoco_muscle_reference(
    device const float* generalizedVelocities [[buffer(7)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(8)]],
    device const float* pointJacobians [[buffer(11)]],
    device const MRMujocoMuscleReferenceDispatchGPU& dispatch [[buffer(24)]],
    device const MRMujocoMuscleGPU* muscles [[buffer(25)]],
    device const MRMujocoMuscleStateGPU* states [[buffer(26)]],
    device const MRMujocoMuscleSiteGPU* sites [[buffer(27)]],
    device const MRMujocoMuscleWrapGPU* wraps [[buffer(28)]],
    device const MRMujocoMuscleRouteNodeGPU* routes [[buffer(29)]],
    device MRMujocoMuscleResultGPU* results [[buffer(30)]],
    device float* muscleGeneralizedForces [[buffer(23)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.muscleCount == 0u || globalIndex >= dispatch.environmentCount * dispatch.muscleCount) return;
    const uint environment = globalIndex / dispatch.muscleCount;
    const uint muscleIndex = globalIndex - environment * dispatch.muscleCount;
    MRMujocoMuscleResultGPU result{};
    result.status = MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PROGRAM;
    result.environment = environment;
    result.muscleIndex = muscleIndex;
    if (dispatch.abiVersion != MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION ||
        dispatch.bodyPoseStride == 0u || dispatch.dofCount == 0u ||
        dispatch.pointJacobianStride == 0u ||
        dispatch.bodyJacobianPointStride != 4u ||
        !finite4(dispatch.timestepSecondsAndReserved) ||
        any(dispatch.timestepSecondsAndReserved.yzw != float3(0.0f)) ||
        muscleIndex >= dispatch.muscleCount) {
        results[globalIndex] = result; return;
    }
    const uint forceBase = globalIndex * dispatch.dofCount;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        muscleGeneralizedForces[forceBase + dof] = 0.0f;
    }
    const MRMujocoMuscleGPU muscle = muscles[muscleIndex];
    const uint routeOffset = muscle.route.x;
    const uint routeCount = muscle.route.y;
    if (routeCount < 2u || routeOffset > dispatch.routeNodeCount ||
        routeCount > dispatch.routeNodeCount - routeOffset || muscle.route.z != 0u || muscle.route.w != 0u ||
        !finite4(muscle.lengthRangeAndAcceleration) || !finite4(muscle.controlRange) ||
        !finite4(muscle.compliantArchitecture0) ||
        !finite4(muscle.compliantArchitecture1) ||
        muscle.lengthRangeAndAcceleration.w != 0.0f || muscle.controlRange.z != 0.0f || muscle.controlRange.w != 0.0f) {
        results[globalIndex] = result; return;
    }
    for (uint block = 0u; block < 3u; ++block) {
        if (!finite4(muscle.gainParameters[block]) || !finite4(muscle.biasParameters[block]) || !finite4(muscle.dynamicParameters[block]) ||
            (block == 2u && (any(muscle.gainParameters[block].zw != float2(0.0f)) || any(muscle.biasParameters[block].zw != float2(0.0f)) || any(muscle.dynamicParameters[block].zw != float2(0.0f))))) {
            results[globalIndex] = result; return;
        }
    }
    const MRMujocoMuscleStateGPU state = states[globalIndex];
    if (!finite4(state.excitationAndActivation) || state.excitationAndActivation.z < 0.0f ||
        (!compliantArchitecture(muscle) &&
         (state.excitationAndActivation.z != 0.0f || state.excitationAndActivation.w != 0.0f))) {
        result.status = MR_MUJOCO_MUSCLE_REFERENCE_INVALID_STATE; results[globalIndex] = result; return;
    }
    float totalLength = 0.0f;
    uint appliedWraps = 0u;
    uint cursor = 0u;
    bool validPath = true;
    float3 originLengthGradient = float3(0.0f);
    float3 insertionLengthGradient = float3(0.0f);
    bool originGradientObserved = false;
    bool insertionGradientObserved = false;
    while (cursor + 1u < routeCount) {
        const MRMujocoMuscleRouteNodeGPU firstNode = routes[routeOffset + cursor];
        const MRMujocoMuscleRouteNodeGPU nextNode = routes[routeOffset + cursor + 1u];
        float3 firstWorld;
        if (firstNode.type != MR_MUJOCO_MUSCLE_ROUTE_SITE || firstNode.reserved0 != 0u ||
            !siteWorld(environment, dispatch, bodyPoses, sites, firstNode.targetIndex, firstWorld)) { validPath = false; break; }
        if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SITE) {
            float3 secondWorld;
            if (nextNode.reserved0 != 0u || !siteWorld(environment, dispatch, bodyPoses, sites, nextNode.targetIndex, secondWorld)) { validPath = false; break; }
            const MRMujocoMuscleSiteGPU secondSite = sites[nextNode.targetIndex];
            const float3 endpointDifference = secondWorld - firstWorld;
            const float endpointDistance = length(endpointDifference);
            const float3 endpointDirection = endpointDistance > kMinimum
                ? endpointDifference / endpointDistance : float3(0.0f);
            if (cursor == 0u) {
                originLengthGradient = -endpointDirection;
                originGradientObserved = true;
            }
            if (cursor + 1u == routeCount - 1u) {
                insertionLengthGradient = endpointDirection;
                insertionGradientObserved = true;
            }
            if (!addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, pointJacobians,
                    firstNode.targetIndex < dispatch.siteCount
                        ? sites[firstNode.targetIndex].bodyIndex
                        : MR_INVALID_INDEX,
                    firstWorld, secondSite.bodyIndex, secondWorld,
                    muscleGeneralizedForces + forceBase
                )) { validPath = false; break; }
            totalLength += length(secondWorld - firstWorld); cursor += 1u; continue;
        }
        if ((nextNode.type != MR_MUJOCO_MUSCLE_ROUTE_SPHERE && nextNode.type != MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) ||
            nextNode.reserved0 != 0u || nextNode.targetIndex >= dispatch.wrapCount || cursor + 2u >= routeCount) { validPath = false; break; }
        const MRMujocoMuscleRouteNodeGPU lastNode = routes[routeOffset + cursor + 2u];
        float3 lastWorld;
        if (lastNode.type != MR_MUJOCO_MUSCLE_ROUTE_SITE || lastNode.reserved0 != 0u ||
            !siteWorld(environment, dispatch, bodyPoses, sites, lastNode.targetIndex, lastWorld)) { validPath = false; break; }
        const MRMujocoMuscleWrapGPU wrap = wraps[nextNode.targetIndex];
        if (wrap.type != nextNode.type || wrap.bodyIndex < dispatch.articulationFirstBody ||
            wrap.bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride || wrap.reserved0 != 0u || wrap.reserved1 != 0u ||
            !finite4(wrap.localCenter) || !finite4(wrap.rotationRow0) || !finite4(wrap.rotationRow1) || !finite4(wrap.rotationRow2) || !finite4(wrap.radius) ||
            wrap.localCenter.w != 0.0f || wrap.rotationRow0.w != 0.0f || wrap.rotationRow1.w != 0.0f || wrap.rotationRow2.w != 0.0f ||
            wrap.radius.y != 0.0f || wrap.radius.z != 0.0f || wrap.radius.w != 0.0f || !(wrap.radius.x > kMinimum)) { validPath = false; break; }
        const MRArticulatedBodyPoseGPU pose = bodyPoses[environment * dispatch.bodyPoseStride + wrap.bodyIndex - dispatch.articulationFirstBody];
        const float3 center = pose.position.xyz + quaternionRotate(pose.orientation, wrap.localCenter.xyz);
        const float3 localFirst = matrixTransposeApply(wrap, quaternionConjugateRotate(pose.orientation, firstWorld - center));
        const float3 localLast = matrixTransposeApply(wrap, quaternionConjugateRotate(pose.orientation, lastWorld - center));
        const float3 basis0 = nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ? normalize(localFirst) : float3(1.0f, 0.0f, 0.0f);
        float3 normal = cross(localFirst, localLast);
        if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE) {
            if (length(normal) < kMinimum) {
                const uint selected = abs(basis0.y) > abs(basis0.x) && abs(basis0.y) > abs(basis0.z) ? 1u : (abs(basis0.z) > abs(basis0.x) && abs(basis0.z) > abs(basis0.y) ? 2u : 0u);
                const float3 alternate = selected == 0u ? float3(0.0f, 1.0f, 1.0f) : (selected == 1u ? float3(1.0f, 0.0f, 1.0f) : float3(1.0f, 1.0f, 0.0f));
                normal = cross(basis0, alternate);
            }
            normal = normalize(normal);
        }
        const float3 basis1 = nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ? normalize(cross(normal, basis0)) : float3(0.0f, 1.0f, 0.0f);
        const float2 projectedFirst = float2(dot(localFirst, basis0), dot(localFirst, basis1));
        const float2 projectedLast = float2(dot(localLast, basis0), dot(localLast, basis1));
        const bool hasSideSite = nextNode.sideSiteIndex != MR_INVALID_INDEX;
        bool hasProjectedSide = hasSideSite;
        bool sideInsideWrap = false;
        float2 side{};
        float3 localSide{};
        if (hasSideSite) {
            float3 sideWorld;
            if (!siteWorld(environment, dispatch, bodyPoses, sites, nextNode.sideSiteIndex, sideWorld)) { validPath = false; break; }
            localSide = matrixTransposeApply(wrap, quaternionConjugateRotate(pose.orientation, sideWorld - center));
            side = float2(dot(localSide, basis0), dot(localSide, basis1));
            sideInsideWrap = length(localSide) < wrap.radius.x;
            if (length(side) < kMinimum) hasProjectedSide = false; else side = wrap.radius.x * normalize(side);
        }
        float4 contact{}; float wrappingLength = 0.0f;
        bool wrapped = false;
        if (hasSideSite && sideInsideWrap) {
            wrapped = wrapInside(projectedFirst, projectedLast, wrap.radius.x, contact);
            wrappingLength = 0.0f;
        } else {
            wrapped = wrapCircle(projectedFirst, projectedLast, hasProjectedSide, side, wrap.radius.x, contact, wrappingLength);
        }
        if (!wrapped) {
            const float3 endpointDifference = lastWorld - firstWorld;
            const float endpointDistance = length(endpointDifference);
            const float3 endpointDirection = endpointDistance > kMinimum
                ? endpointDifference / endpointDistance : float3(0.0f);
            if (cursor == 0u) {
                originLengthGradient = -endpointDirection;
                originGradientObserved = true;
            }
            if (cursor + 2u == routeCount - 1u) {
                insertionLengthGradient = endpointDirection;
                insertionGradientObserved = true;
            }
            if (!addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, pointJacobians,
                    sites[firstNode.targetIndex].bodyIndex, firstWorld,
                    sites[lastNode.targetIndex].bodyIndex, lastWorld,
                    muscleGeneralizedForces + forceBase
                )) { validPath = false; break; }
            totalLength += length(lastWorld - firstWorld);
        } else {
            float3 tangentFirst = basis0 * contact.x + basis1 * contact.y;
            float3 tangentLast = basis0 * contact.z + basis1 * contact.w;
            if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) {
                const float firstLeg = length(projectedFirst - contact.xy);
                const float secondLeg = length(projectedLast - contact.zw);
                const float denominator = firstLeg + wrappingLength + secondLeg;
                if (!(denominator > kMinimum)) { validPath = false; break; }
                tangentFirst.z = localFirst.z + (localLast.z - localFirst.z) * firstLeg / denominator;
                tangentLast.z = localFirst.z + (localLast.z - localFirst.z) * (firstLeg + wrappingLength) / denominator;
                wrappingLength = sqrt(wrappingLength * wrappingLength + (tangentLast.z - tangentFirst.z) * (tangentLast.z - tangentFirst.z));
            }
            const float3 worldTangentFirst = center + quaternionRotate(pose.orientation, matrixApply(wrap, tangentFirst));
            const float3 worldTangentLast = center + quaternionRotate(pose.orientation, matrixApply(wrap, tangentLast));
            if (cursor == 0u) {
                const float distance = length(worldTangentFirst - firstWorld);
                originLengthGradient = distance > kMinimum
                    ? (firstWorld - worldTangentFirst) / distance : float3(0.0f);
                originGradientObserved = true;
            }
            if (cursor + 2u == routeCount - 1u) {
                const float distance = length(lastWorld - worldTangentLast);
                insertionLengthGradient = distance > kMinimum
                    ? (lastWorld - worldTangentLast) / distance : float3(0.0f);
                insertionGradientObserved = true;
            }
            if (!addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, pointJacobians,
                    sites[firstNode.targetIndex].bodyIndex, firstWorld,
                    wrap.bodyIndex, worldTangentFirst,
                    muscleGeneralizedForces + forceBase
                ) || !addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, pointJacobians,
                    wrap.bodyIndex, worldTangentFirst,
                    wrap.bodyIndex, worldTangentLast,
                    muscleGeneralizedForces + forceBase
                ) || !addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, pointJacobians,
                    wrap.bodyIndex, worldTangentLast,
                    sites[lastNode.targetIndex].bodyIndex, lastWorld,
                    muscleGeneralizedForces + forceBase
                )) { validPath = false; break; }
            totalLength += length(worldTangentFirst - firstWorld) + wrappingLength + length(lastWorld - worldTangentLast);
            ++appliedWraps;
        }
        cursor += 2u;
    }
    if (!validPath || cursor != routeCount - 1u || !(totalLength > kMinimum) ||
        !isfinite(totalLength) || !originGradientObserved || !insertionGradientObserved ||
        !all(isfinite(originLengthGradient)) || !all(isfinite(insertionLengthGradient))) {
        result.status = MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PATH;
        results[globalIndex] = result;
        return;
    }
    float pathVelocity = 0.0f;
    const uint velocityBase = environment * dispatch.dofCount;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        pathVelocity += muscleGeneralizedForces[forceBase + dof] *
            generalizedVelocities[velocityBase + dof];
    }
    const float derivative = activationDerivative(muscle, state.excitationAndActivation.x, state.excitationAndActivation.y);
    float force = 0.0f;
    float activeForce = 0.0f;
    float candidateFiberLength = 0.0f;
    float candidateFiberVelocity = 0.0f;
    float tendonTension = 0.0f;
    float equilibriumResidual = 0.0f;
    if (compliantArchitecture(muscle)) {
        if (!solveCompliantFiber(
                totalLength,
                pathVelocity,
                dispatch.timestepSecondsAndReserved.x,
                state.excitationAndActivation.y,
                state,
                muscle,
                candidateFiberLength,
                candidateFiberVelocity,
                tendonTension,
                equilibriumResidual
            )) {
            result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
            results[globalIndex] = result;
            return;
        }
        const float maximumForce = forceScale(
            muscle.gainParameters, muscle.lengthRangeAndAcceleration.z
        );
        force = -maximumForce * tendonTension;
        activeForce = force;
    } else {
        force = muscleGain(totalLength, pathVelocity, muscle) *
            state.excitationAndActivation.y + muscleBias(totalLength, muscle);
        activeForce = muscleGain(totalLength, pathVelocity, muscle) *
            state.excitationAndActivation.y;
    }
    if (!isfinite(derivative) || !isfinite(force) || !isfinite(pathVelocity)) { result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT; results[globalIndex] = result; return; }
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        muscleGeneralizedForces[forceBase + dof] *= force;
        if (!isfinite(muscleGeneralizedForces[forceBase + dof])) {
            result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
            results[globalIndex] = result;
            return;
        }
    }
    result.status = MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS;
    result.appliedWrapCount = appliedWraps;
    result.pathForceAndActivationDerivative = float4(totalLength, pathVelocity, force, derivative);
    result.endpointLengthGradients[0] = float4(originLengthGradient, 0.0f);
    result.endpointLengthGradients[1] = float4(insertionLengthGradient, 0.0f);
    result.activeForceAndReserved = float4(activeForce, 0.0f, 0.0f, 0.0f);
    result.fiberStateTendonForceResidual = float4(
        candidateFiberLength,
        candidateFiberVelocity,
        tendonTension,
        equilibriumResidual
    );
    results[globalIndex] = result;
}

// Converts total source force rows to their activation-dependent component
// after route evaluation and before deterministic reduction. The imported
// passive MuJoCo bias is retained in the typed result for inspection, but it
// is not a registered equilibrium preload for Numi Human v1 and therefore is
// not injected into the standing dynamics horizon.
kernel void mr_mujoco_muscle_active_force_rows(
    device const MRMujocoMuscleGPU* muscles [[buffer(0)]],
    device const MRMujocoMuscleStateGPU* states [[buffer(1)]],
    device MRMujocoMuscleResultGPU* results [[buffer(2)]],
    device float* muscleGeneralizedForces [[buffer(3)]],
    constant MRMujocoMuscleActiveForceDispatchGPU& dispatch [[buffer(4)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.abiVersion != MR_MUJOCO_MUSCLE_ACTIVE_FORCE_GPU_ABI_VERSION ||
        dispatch.muscleCount == 0u || dispatch.environmentCount == 0u ||
        dispatch.dofCount == 0u ||
        globalIndex >= dispatch.environmentCount * dispatch.muscleCount) {
        return;
    }
    device MRMujocoMuscleResultGPU& result = results[globalIndex];
    if (result.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS) return;
    const uint muscleIndex = globalIndex % dispatch.muscleCount;
    const float totalForce = result.pathForceAndActivationDerivative.z;
    const MRMujocoMuscleGPU muscle = muscles[muscleIndex];
    const float activeForce = compliantArchitecture(muscle)
        ? totalForce
        : muscleGain(
            result.pathForceAndActivationDerivative.x,
            result.pathForceAndActivationDerivative.y,
            muscle
        ) * states[globalIndex].excitationAndActivation.y;
    result.activeForceAndReserved = float4(activeForce, 0.0f, 0.0f, 0.0f);
    const uint forceBase = globalIndex * dispatch.dofCount;
    if (abs(totalForce) <= kMinimum) {
        if (abs(activeForce) > kMinimum) {
            result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
        } else {
            for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
                muscleGeneralizedForces[forceBase + dof] = 0.0f;
            }
        }
        return;
    }
    const float scale = activeForce / totalForce;
    if (!isfinite(scale)) {
        result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
        return;
    }
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        muscleGeneralizedForces[forceBase + dof] *= scale;
    }
}

kernel void mr_mujoco_muscle_reduce(
    device const MRMujocoMuscleReferenceDispatchGPU& dispatch [[buffer(24)]],
    device float* muscleAndGeneralizedForces [[buffer(23)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.abiVersion != MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION ||
        dispatch.muscleCount == 0u || dispatch.dofCount == 0u ||
        globalIndex >= dispatch.environmentCount * dispatch.dofCount) return;
    const uint environment = globalIndex / dispatch.dofCount;
    const uint dof = globalIndex - environment * dispatch.dofCount;
    float total = 0.0f;
    const uint muscleBase = environment * dispatch.muscleCount * dispatch.dofCount;
    for (uint muscle = 0u; muscle < dispatch.muscleCount; ++muscle) {
        total += muscleAndGeneralizedForces[
            muscleBase + muscle * dispatch.dofCount + dof
        ];
    }
    const uint outputBase = dispatch.environmentCount *
        dispatch.muscleCount * dispatch.dofCount;
    muscleAndGeneralizedForces[outputBase + globalIndex] = total;
}

// The reference pass evaluates force from the activation at the start of the
// transaction. Advance the sidecar only after that result is available so a
// step has an unambiguous, explicit-Euler time ordering. Invalid reference
// records deliberately retain their previous state rather than publishing a
// fabricated recovery value.
kernel void mr_mujoco_muscle_activation_step(
    device MRMujocoMuscleStateGPU* states [[buffer(0)]],
    device const MRMujocoMuscleResultGPU* results [[buffer(1)]],
    constant MRMujocoMuscleActivationDispatchGPU& dispatch [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.abiVersion != MR_MUJOCO_MUSCLE_ACTIVATION_GPU_ABI_VERSION ||
        dispatch.reserved0 != 0u || dispatch.reserved1 != 0u ||
        dispatch.timestepSecondsAndReserved.y != 0.0f ||
        dispatch.timestepSecondsAndReserved.z != 0.0f ||
        dispatch.timestepSecondsAndReserved.w != 0.0f ||
        !isfinite(dispatch.timestepSecondsAndReserved.x) ||
        !(dispatch.timestepSecondsAndReserved.x > 0.0f) ||
        globalIndex >= dispatch.stateCount) {
        return;
    }
    const MRMujocoMuscleStateGPU current = states[globalIndex];
    const MRMujocoMuscleResultGPU reference = results[globalIndex];
    if (reference.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
        !finite4(current.excitationAndActivation) ||
        current.excitationAndActivation.z != 0.0f ||
        current.excitationAndActivation.w != 0.0f ||
        !finite4(reference.pathForceAndActivationDerivative) ||
        !finite4(reference.fiberStateTendonForceResidual)) {
        return;
    }
    const float nextActivation = clamp(
        current.excitationAndActivation.y +
            dispatch.timestepSecondsAndReserved.x *
                reference.pathForceAndActivationDerivative.w,
        0.0f,
        1.0f
    );
    if (!isfinite(nextActivation)) return;
    MRMujocoMuscleStateGPU next = current;
    next.excitationAndActivation.y = nextActivation;
    if (reference.fiberStateTendonForceResidual.x > 0.0f) {
        next.excitationAndActivation.z =
            reference.fiberStateTendonForceResidual.x;
        next.excitationAndActivation.w =
            reference.fiberStateTendonForceResidual.y;
    }
    states[globalIndex] = next;
}
