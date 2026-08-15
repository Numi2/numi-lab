#include <metal_stdlib>

#include "metalrobo/rod_gpu_shared.h"

using namespace metal;

namespace {

constant float kGeometryFloor = 1.0e-12f;
constant float kFrameDenominatorFloor = 1.0e-6f;
constant float kFloatEpsilon = 1.1920928955078125e-7f;

inline float rodEventSegmentDuration(
    device const MRCCDEventStateGPU& state,
    const uint mode
) {
    return max(
        mode == MR_CCD_SEGMENT_SELECTED
        ? state.time.w
        : state.time.y,
        0.0f
    );
}

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool rodToolBodyDynamic(
    device const MRBodyStateGPU& body
) {
    return
        body.flagsAndIndices[0] == MR_MOTION_DYNAMIC &&
        body.linearVelocityAndInverseMass.w > 0.0f;
}

inline float3 rodToolInverseInertia(
    device const MRBodyStateGPU& body,
    const float3 value
) {
    return float3(
        dot(body.inverseInertiaWorldRow0.xyz, value),
        dot(body.inverseInertiaWorldRow1.xyz, value),
        dot(body.inverseInertiaWorldRow2.xyz, value)
    );
}

inline float rodToolDirectionalResponse(
    const float3 left,
    const float3 right,
    const float leftTwist,
    const float rightTwist,
    const float firstWeight,
    const float secondWeight,
    const float firstInverseMass,
    const float secondInverseMass,
    const float inverseTwistInertia,
    device const MRBodyStateGPU& body,
    const float3 bodyLever
) {
    float result =
        (
            firstWeight * firstWeight * firstInverseMass +
            secondWeight * secondWeight * secondInverseMass
        ) * dot(left, right) +
        inverseTwistInertia * leftTwist * rightTwist;
    if (rodToolBodyDynamic(body)) {
        result +=
            body.linearVelocityAndInverseMass.w *
                dot(left, right) +
            dot(
                cross(bodyLever, left),
                rodToolInverseInertia(
                    body,
                    cross(bodyLever, right)
                )
            );
    }
    return result;
}

inline bool rodToolInvertSymmetric3(
    thread const float matrix[3][3],
    thread float inverse[3][3]
) {
    const float c00 =
        matrix[1][1] * matrix[2][2] -
        matrix[1][2] * matrix[2][1];
    const float c01 =
        matrix[1][2] * matrix[2][0] -
        matrix[1][0] * matrix[2][2];
    const float c02 =
        matrix[1][0] * matrix[2][1] -
        matrix[1][1] * matrix[2][0];
    const float determinant =
        matrix[0][0] * c00 +
        matrix[0][1] * c01 +
        matrix[0][2] * c02;
    const float scale = max(
        max(
            abs(matrix[0][0] * matrix[1][1] *
                matrix[2][2]),
            1.0f
        ),
        abs(determinant)
    );
    if (!(determinant >
          64.0f * kFloatEpsilon * scale) ||
        !isfinite(determinant)) {
        return false;
    }
    const float reciprocal = 1.0f / determinant;
    inverse[0][0] = c00 * reciprocal;
    inverse[0][1] =
        (matrix[0][2] * matrix[2][1] -
         matrix[0][1] * matrix[2][2]) * reciprocal;
    inverse[0][2] =
        (matrix[0][1] * matrix[1][2] -
         matrix[0][2] * matrix[1][1]) * reciprocal;
    inverse[1][0] = c01 * reciprocal;
    inverse[1][1] =
        (matrix[0][0] * matrix[2][2] -
         matrix[0][2] * matrix[2][0]) * reciprocal;
    inverse[1][2] =
        (matrix[0][2] * matrix[1][0] -
         matrix[0][0] * matrix[1][2]) * reciprocal;
    inverse[2][0] = c02 * reciprocal;
    inverse[2][1] =
        (matrix[0][1] * matrix[2][0] -
         matrix[0][0] * matrix[2][1]) * reciprocal;
    inverse[2][2] =
        (matrix[0][0] * matrix[1][1] -
         matrix[0][1] * matrix[1][0]) * reciprocal;
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            if (!isfinite(inverse[row][column])) {
                return false;
            }
        }
    }
    return true;
}

inline float2 rodToolProjectFriction(
    const float2 candidate,
    const float normalImpulse,
    const float staticFriction,
    const float dynamicFriction
) {
    if (!(normalImpulse > 0.0f)) {
        return float2(0.0f);
    }
    const float magnitudeSquared = dot(candidate, candidate);
    if (!(magnitudeSquared > 0.0f)) {
        return float2(0.0f);
    }
    const float magnitude = sqrt(magnitudeSquared);
    if (magnitude <= staticFriction * normalImpulse) {
        return candidate;
    }
    return candidate *
        (dynamicFriction * normalImpulse / magnitude);
}

inline bool applyRodToolImpulse(
    const MRRodGPUDispatch dispatch,
    const MRRodColliderGPU rod,
    device const float* inverseMasses,
    device const float* inverseRotationalInertias,
    device float4* positions,
    device float4* velocities,
    device float* twists,
    device float* twistRates,
    device MRBodyStateGPU& body,
    const MRRodToolWitnessGPU witness,
    const float3 impulseOnTool,
    const bool advancePosition
) {
    const float weightB = witness.rodPointAndWeight.w;
    const float weightA = 1.0f - weightB;
    const float3 endpointA = positions[rod.nodeA].xyz;
    const float3 endpointB = positions[rod.nodeB].xyz;
    const float3 edge = endpointB - endpointA;
    const float edgeSquared = dot(edge, edge);
    if (!(edgeSquared > 1.0e-20f)) {
        return false;
    }
    const float3 edgeAxis = edge * rsqrt(edgeSquared);
    const float3 radial =
        witness.radialAndTwistJacobianV.xyz;
    const float3 surfaceJacobian = cross(
        edgeAxis,
        rod.radiusAndOffsets.x * radial
    );
    const float firstInverseMass = inverseMasses[rod.nodeA];
    const float secondInverseMass = inverseMasses[rod.nodeB];
    const float inverseTwistInertia =
        inverseRotationalInertias[rod.edgeIndex];
    const float3 firstDelta =
        -weightA * firstInverseMass * impulseOnTool;
    const float3 secondDelta =
        -weightB * secondInverseMass * impulseOnTool;
    const float twistDelta =
        -inverseTwistInertia *
        dot(surfaceJacobian, impulseOnTool);
    velocities[rod.nodeA].xyz += firstDelta;
    velocities[rod.nodeB].xyz += secondDelta;
    twistRates[rod.edgeIndex] += twistDelta;
    if (advancePosition) {
        const float timestep =
            dispatch.gravityAndTimestep.w;
        positions[rod.nodeA].xyz += timestep * firstDelta;
        positions[rod.nodeB].xyz += timestep * secondDelta;
        twists[rod.edgeIndex] += timestep * twistDelta;
    }
    if (rodToolBodyDynamic(body)) {
        const float3 lever =
            witness.toolPointAndSeparation.xyz -
            body.position.xyz;
        body.linearVelocityAndInverseMass.xyz +=
            body.linearVelocityAndInverseMass.w *
            impulseOnTool;
        body.angularVelocity.xyz += rodToolInverseInertia(
            body,
            cross(lever, impulseOnTool)
        );
    }
    return
        finite3(velocities[rod.nodeA].xyz) &&
        finite3(velocities[rod.nodeB].xyz) &&
        isfinite(twistRates[rod.edgeIndex]) &&
        finite3(body.linearVelocityAndInverseMass.xyz) &&
        finite3(body.angularVelocity.xyz);
}

inline bool normalizeChecked(
    const float3 value,
    thread float3& output
) {
    const float magnitude = length(value);
    if (!(magnitude > kGeometryFloor) ||
        !isfinite(magnitude)) {
        return false;
    }
    output = value / magnitude;
    return finite3(output);
}

inline float3 rotateAroundAxis(
    const float3 value,
    const float3 axis,
    const float angle
) {
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return
        value * cosine +
        cross(axis, value) * sine +
        axis * dot(axis, value) * (1.0f - cosine);
}

inline float3 rotateQuaternion(
    const float4 quaternion,
    const float3 value
) {
    const float3 imaginary = quaternion.xyz;
    return value +
        2.0f * cross(
            imaginary,
            cross(imaginary, value) +
                quaternion.w * value
        );
}

inline float3 applyInverseInertia(
    const MRBodyStateGPU body,
    const float3 value
) {
    return float3(
        dot(body.inverseInertiaWorldRow0.xyz, value),
        dot(body.inverseInertiaWorldRow1.xyz, value),
        dot(body.inverseInertiaWorldRow2.xyz, value)
    );
}

inline bool transport(
    const float3 director,
    const float3 from,
    const float3 to,
    thread float3& output
) {
    const float3 axis = cross(from, to);
    const float sine = length(axis);
    const float cosine = clamp(dot(from, to), -1.0f, 1.0f);
    if (sine <= kGeometryFloor) {
        if (cosine < 0.0f) {
            return false;
        }
        output = director;
        return true;
    }
    output = rotateAroundAxis(
        director,
        axis / sine,
        atan2(sine, cosine)
    );
    output -= to * dot(output, to);
    return normalizeChecked(output, output);
}

inline bool localCurvature(
    thread const float3* positions,
    thread const float* twists,
    const float3 referenceTangentLeft,
    const float3 referenceDirectorLeft,
    thread float2& curvature
) {
    float3 left;
    float3 right;
    if (!normalizeChecked(positions[1] - positions[0], left) ||
        !normalizeChecked(positions[2] - positions[1], right)) {
        return false;
    }
    float3 referenceLeft;
    float3 referenceRight;
    if (!transport(
            referenceDirectorLeft,
            referenceTangentLeft,
            left,
            referenceLeft
        ) ||
        !transport(
            referenceLeft,
            left,
            right,
            referenceRight
        )) {
        return false;
    }
    const float3 directorLeft = rotateAroundAxis(
        referenceLeft,
        left,
        twists[0]
    );
    const float3 directorRight = rotateAroundAxis(
        referenceRight,
        right,
        twists[1]
    );
    const float3 secondLeft = cross(left, directorLeft);
    const float3 secondRight = cross(right, directorRight);
    const float denominator = 1.0f + dot(left, right);
    if (!(denominator > kFrameDenominatorFloor) ||
        !isfinite(denominator)) {
        return false;
    }
    const float3 binormal =
        2.0f * cross(left, right) / denominator;
    curvature = float2(
        0.5f * dot(binormal, secondLeft + secondRight),
        -0.5f * dot(binormal, directorLeft + directorRight)
    );
    return all(isfinite(curvature));
}

inline void recordFailure(
    threadgroup atomic_uint& failure,
    const uint code
) {
    uint expected = MR_ROD_GPU_SUCCESS;
    while (
        expected == MR_ROD_GPU_SUCCESS &&
        !atomic_compare_exchange_weak_explicit(
            &failure,
            &expected,
            code,
            memory_order_relaxed,
            memory_order_relaxed
        )
    ) {
    }
}

inline void recordPositiveMaximum(
    threadgroup atomic_uint& maximumBits,
    const float value
) {
    if (isfinite(value) && value >= 0.0f) {
        atomic_fetch_max_explicit(
            &maximumBits,
            as_type<uint>(value),
            memory_order_relaxed
        );
    }
}

inline void recordCorrectionOwner(
    threadgroup atomic_uint& ownerKey,
    const float correction,
    const uint owner
) {
    if (!(correction >= 0.0f) || !isfinite(correction) || owner > 254u) {
        return;
    }
    // Positive IEEE-754 values are monotonically ordered as uints. Preserve
    // the upper 24 value bits and use the low byte for a deterministic owner;
    // this keeps ample diagnostic precision without requiring 64-bit
    // threadgroup atomics on older Apple GPU families.
    const uint key =
        (as_type<uint>(correction) & 0xffffff00u) |
        (255u - owner);
    atomic_fetch_max_explicit(
        &ownerKey,
        key,
        memory_order_relaxed
    );
}

inline void projectStretch(
    const uint edge,
    const float timestep,
    device const float* restLengths,
    device const float* inverseMasses,
    device const float* stretchStiffness,
    threadgroup float3* positions,
    threadgroup float* accumulatedMultipliers,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits
) {
    const float3 delta =
        positions[edge + 1u] - positions[edge];
    const float currentLength = length(delta);
    if (!(currentLength > kGeometryFloor) ||
        !isfinite(currentLength)) {
        recordFailure(
            failure,
            MR_ROD_GPU_DEGENERATE_GEOMETRY
        );
        return;
    }
    const float constraint =
        currentLength - restLengths[edge];
    const float inverseA = inverseMasses[edge];
    const float inverseB = inverseMasses[edge + 1u];
    const float alpha =
        restLengths[edge] /
        stretchStiffness[edge] /
        (timestep * timestep);
    const float lambda =
        (-constraint -
         alpha * accumulatedMultipliers[edge]) /
        (inverseA + inverseB + alpha);
    accumulatedMultipliers[edge] += lambda;
    const float3 direction = delta / currentLength;
    const float3 first = -inverseA * lambda * direction;
    const float3 second = inverseB * lambda * direction;
    positions[edge] += first;
    positions[edge + 1u] += second;
    recordPositiveMaximum(maximumErrorBits, abs(constraint));
    recordPositiveMaximum(
        maximumCorrectionBits,
        max(length(first), length(second))
    );
}

inline void projectTwist(
    const uint constraintIndex,
    const float timestep,
    device const float* restTwists,
    device const float* restLengths,
    device const float* inverseRotationalInertias,
    device const float* twistStiffness,
    threadgroup float* twists,
    threadgroup float* accumulatedMultipliers,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits
) {
    const float constraint =
        (twists[constraintIndex + 1u] - twists[constraintIndex]) -
        (
            restTwists[constraintIndex + 1u] -
            restTwists[constraintIndex]
        );
    const float inverseA =
        inverseRotationalInertias[constraintIndex];
    const float inverseB =
        inverseRotationalInertias[constraintIndex + 1u];
    const float voronoi =
        0.5f * (
            restLengths[constraintIndex] +
            restLengths[constraintIndex + 1u]
        );
    const float alpha =
        voronoi /
        twistStiffness[constraintIndex] /
        (timestep * timestep);
    const float lambda =
        (-constraint -
         alpha * accumulatedMultipliers[constraintIndex]) /
        (inverseA + inverseB + alpha);
    accumulatedMultipliers[constraintIndex] += lambda;
    const float first = -inverseA * lambda;
    const float second = inverseB * lambda;
    twists[constraintIndex] += first;
    twists[constraintIndex + 1u] += second;
    if (!isfinite(twists[constraintIndex]) ||
        !isfinite(twists[constraintIndex + 1u])) {
        recordFailure(
            failure,
            MR_ROD_GPU_NONFINITE_RESULT
        );
        return;
    }
    recordPositiveMaximum(maximumErrorBits, abs(constraint));
    recordPositiveMaximum(
        maximumCorrectionBits,
        max(abs(first), abs(second))
    );
}

inline void projectAttachment(
    const MRRodGPUAttachment attachment,
    const uint attachmentIndex,
    const float timestep,
    device const float* inverseMasses,
    threadgroup float3* positions,
    threadgroup float3* accumulatedMultipliers,
    threadgroup float3* targetImpulses,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits
) {
    const uint node = attachment.nodeIndex;
    const float3 delta =
        positions[node] - attachment.targetAndCompliance.xyz;
    const float inverseMass = inverseMasses[node];
    const float alpha =
        attachment.targetAndCompliance.w /
        (timestep * timestep);
    const float3 deltaMultiplier =
        (-delta -
         alpha * accumulatedMultipliers[attachmentIndex]) /
        (inverseMass + alpha);
    accumulatedMultipliers[attachmentIndex] += deltaMultiplier;
    const float3 correction =
        inverseMass * deltaMultiplier;
    positions[node] += correction;
    targetImpulses[attachmentIndex] -=
        correction / (inverseMass * timestep);
    if (!finite3(positions[node])) {
        recordFailure(
            failure,
            MR_ROD_GPU_NONFINITE_RESULT
        );
        return;
    }
    recordPositiveMaximum(maximumErrorBits, length(delta));
    recordPositiveMaximum(
        maximumCorrectionBits,
        length(correction)
    );
}

struct RodClosestSegments {
    float first;
    float second;
    float3 delta;
    float distance;
};

inline bool closestRodSegments(
    const float3 firstA,
    const float3 firstB,
    const float3 secondA,
    const float3 secondB,
    thread RodClosestSegments& result
) {
    const float3 firstDirection = firstB - firstA;
    const float3 secondDirection = secondB - secondA;
    const float3 offset = firstA - secondA;
    const float firstLengthSquared =
        dot(firstDirection, firstDirection);
    const float secondLengthSquared =
        dot(secondDirection, secondDirection);
    if (!(firstLengthSquared > 1.0e-20f) ||
        !(secondLengthSquared > 1.0e-20f)) {
        return false;
    }
    const float firstOffset =
        dot(firstDirection, offset);
    const float secondOffset =
        dot(secondDirection, offset);
    const float coupling =
        dot(firstDirection, secondDirection);
    const float denominator =
        firstLengthSquared * secondLengthSquared -
        coupling * coupling;
    result.first =
        denominator >
            1.0e-7f * firstLengthSquared *
                secondLengthSquared
        ? clamp(
              (
                  coupling * secondOffset -
                  firstOffset * secondLengthSquared
              ) / denominator,
              0.0f,
              1.0f
          )
        : 0.0f;
    const float secondNumerator =
        coupling * result.first + secondOffset;
    if (secondNumerator < 0.0f) {
        result.second = 0.0f;
        result.first = clamp(
            -firstOffset / firstLengthSquared,
            0.0f,
            1.0f
        );
    } else if (secondNumerator >
               secondLengthSquared) {
        result.second = 1.0f;
        result.first = clamp(
            (
                coupling - firstOffset
            ) / firstLengthSquared,
            0.0f,
            1.0f
        );
    } else {
        result.second =
            secondNumerator / secondLengthSquared;
    }
    const float3 firstPoint =
        firstA + result.first * firstDirection;
    const float3 secondPoint =
        secondA + result.second * secondDirection;
    result.delta = secondPoint - firstPoint;
    result.distance = length(result.delta);
    return isfinite(result.first) &&
        isfinite(result.second) &&
        finite3(result.delta) &&
        isfinite(result.distance);
}

inline float3 stableSelfContactNormal(
    const float3 firstDirection,
    const float3 secondDirection,
    const uint firstEdge,
    const uint secondEdge
) {
    float3 normal = cross(
        firstDirection,
        secondDirection
    );
    if (dot(normal, normal) <= 1.0e-20f) {
        const float3 tangent = normalize(firstDirection);
        const float3 absoluteTangent = abs(tangent);
        const float3 axis =
            absoluteTangent.x <= absoluteTangent.y &&
                absoluteTangent.x <= absoluteTangent.z
            ? float3(1.0f, 0.0f, 0.0f)
            : (
                absoluteTangent.y <= absoluteTangent.z
                ? float3(0.0f, 1.0f, 0.0f)
                : float3(0.0f, 0.0f, 1.0f)
            );
        normal = axis - tangent * dot(axis, tangent);
    }
    normal = normalize(normal);
    return ((firstEdge ^ secondEdge) & 1u) != 0u
        ? -normal
        : normal;
}

inline void projectSelfContact(
    const uint firstEdge,
    const uint secondEdge,
    const MRRodGPUDispatch dispatch,
    device const float* inverseMasses,
    threadgroup float3* positions,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits,
    threadgroup atomic_uint& maximumPenetrationBits,
    threadgroup atomic_uint& projectedContactCount
) {
    RodClosestSegments closest;
    if (!closestRodSegments(
            positions[firstEdge],
            positions[firstEdge + 1u],
            positions[secondEdge],
            positions[secondEdge + 1u],
            closest
        )) {
        recordFailure(
            failure,
            MR_ROD_GPU_DEGENERATE_GEOMETRY
        );
        return;
    }
    const float contactDistance =
        2.0f * dispatch.selfCollision.x +
        dispatch.selfCollision.y;
    const float penetration =
        contactDistance - closest.distance;
    if (!(penetration > 0.0f)) {
        return;
    }
    const float3 firstDirection =
        positions[firstEdge + 1u] -
        positions[firstEdge];
    const float3 secondDirection =
        positions[secondEdge + 1u] -
        positions[secondEdge];
    const float3 normal = closest.distance > 1.0e-10f
        ? closest.delta / closest.distance
        : stableSelfContactNormal(
              firstDirection,
              secondDirection,
              firstEdge,
              secondEdge
          );
    const float4 weights = float4(
        1.0f - closest.first,
        closest.first,
        1.0f - closest.second,
        closest.second
    );
    const uint4 nodes = uint4(
        firstEdge,
        firstEdge + 1u,
        secondEdge,
        secondEdge + 1u
    );
    float denominator = 0.0f;
    for (uint slot = 0u; slot < 4u; ++slot) {
        denominator +=
            weights[slot] * weights[slot] *
            inverseMasses[nodes[slot]];
    }
    const float timestep =
        dispatch.gravityAndTimestep.w;
    const float alpha =
        dispatch.selfCollision.z /
        (timestep * timestep);
    if (!(denominator + alpha > 0.0f) ||
        !isfinite(denominator) ||
        !finite3(normal)) {
        recordFailure(
            failure,
            MR_ROD_GPU_NONFINITE_RESULT
        );
        return;
    }
    const float lambda =
        penetration / (denominator + alpha);
    for (uint slot = 0u; slot < 4u; ++slot) {
        const float sign = slot < 2u ? -1.0f : 1.0f;
        const float3 correction =
            sign * normal * weights[slot] * lambda *
            inverseMasses[nodes[slot]];
        positions[nodes[slot]] += correction;
        recordPositiveMaximum(
            maximumCorrectionBits,
            length(correction)
        );
    }
    recordPositiveMaximum(maximumErrorBits, penetration);
    recordPositiveMaximum(
        maximumPenetrationBits,
        penetration
    );
    atomic_fetch_add_explicit(
        &projectedContactCount,
        1u,
        memory_order_relaxed
    );
}

// Four SIMD32 cohorts cooperatively evaluate the quadratic capsule
// broadphase while lane zero retains the canonical pair-order Gauss-Seidel
// projection. Candidate publication is a deterministic bitset: atomics
// combine only integer flags, never physical corrections. This removes the
// Apple-GPU lane-zero distance bottleneck without changing the accepted
// projection order.
inline void projectSelfContactsCooperative(
    const MRRodGPUDispatch dispatch,
    device const float* inverseMasses,
    threadgroup float3* positions,
    threadgroup atomic_uint* candidateWords,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits,
    threadgroup atomic_uint& maximumPenetrationBits,
    threadgroup atomic_uint& projectedContactCount,
    const uint lane,
    const uint laneCount
) {
    for (uint word = lane;
         word < MR_ROD_GPU_SELF_CONTACT_PAIR_WORDS;
         word += laneCount) {
        atomic_store_explicit(
            &candidateWords[word],
            0u,
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float contactDistance =
        2.0f * dispatch.selfCollision.x +
        dispatch.selfCollision.y;
    for (uint firstEdge = lane;
         firstEdge + 2u < dispatch.edgeCount;
         firstEdge += laneCount) {
        for (uint secondEdge = firstEdge + 2u;
             secondEdge < dispatch.edgeCount;
             ++secondEdge) {
            const uint pairOrdinal =
                firstEdge * (
                    2u * dispatch.edgeCount - firstEdge - 3u
                ) / 2u +
                secondEdge - firstEdge - 2u;
            RodClosestSegments closest;
            if (!closestRodSegments(
                    positions[firstEdge],
                    positions[firstEdge + 1u],
                    positions[secondEdge],
                    positions[secondEdge + 1u],
                    closest
                )) {
                recordFailure(
                    failure,
                    MR_ROD_GPU_DEGENERATE_GEOMETRY
                );
                continue;
            }
            if (closest.distance < contactDistance) {
                atomic_fetch_or_explicit(
                    &candidateWords[pairOrdinal >> 5u],
                    1u << (pairOrdinal & 31u),
                    memory_order_relaxed
                );
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        const uint activePairCount =
            dispatch.edgeCount >= 2u
            ? (
                (dispatch.edgeCount - 1u) *
                (dispatch.edgeCount - 2u)
            ) / 2u
            : 0u;
        uint firstEdge = 0u;
        uint rowStart = 0u;
        uint rowLength = dispatch.edgeCount >= 2u
            ? dispatch.edgeCount - 2u
            : 0u;
        for (uint word = 0u;
             word < MR_ROD_GPU_SELF_CONTACT_PAIR_WORDS;
             ++word) {
            uint candidates = atomic_load_explicit(
                &candidateWords[word],
                memory_order_relaxed
            );
            while (candidates != 0u) {
                const uint bit = ctz(candidates);
                const uint pairOrdinal = 32u * word + bit;
                if (pairOrdinal >= activePairCount) {
                    break;
                }
                while (pairOrdinal >= rowStart + rowLength) {
                    rowStart += rowLength;
                    ++firstEdge;
                    --rowLength;
                }
                const uint secondEdge =
                    firstEdge + 2u + pairOrdinal - rowStart;
                projectSelfContact(
                    firstEdge,
                    secondEdge,
                    dispatch,
                    inverseMasses,
                    positions,
                    failure,
                    maximumErrorBits,
                    maximumCorrectionBits,
                    maximumPenetrationBits,
                    projectedContactCount
                );
                candidates &= candidates - 1u;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline bool selfFrictionKinematics(
    const uint firstEdge,
    const uint secondEdge,
    device const float* inverseMasses,
    threadgroup const float3* positions,
    thread RodClosestSegments& closest,
    thread float3& normal,
    thread float4& weights,
    thread uint4& nodes,
    thread float& inverseEffectiveMass
) {
    if (!closestRodSegments(
            positions[firstEdge],
            positions[firstEdge + 1u],
            positions[secondEdge],
            positions[secondEdge + 1u],
            closest
        )) {
        return false;
    }
    const float3 firstDirection =
        positions[firstEdge + 1u] - positions[firstEdge];
    const float3 secondDirection =
        positions[secondEdge + 1u] - positions[secondEdge];
    normal = closest.distance > 1.0e-10f
        ? closest.delta / closest.distance
        : stableSelfContactNormal(
              firstDirection,
              secondDirection,
              firstEdge,
              secondEdge
          );
    weights = float4(
        1.0f - closest.first,
        closest.first,
        1.0f - closest.second,
        closest.second
    );
    nodes = uint4(
        firstEdge,
        firstEdge + 1u,
        secondEdge,
        secondEdge + 1u
    );
    inverseEffectiveMass = 0.0f;
    for (uint slot = 0u; slot < 4u; ++slot) {
        inverseEffectiveMass +=
            weights[slot] * weights[slot] *
            inverseMasses[nodes[slot]];
    }
    return inverseEffectiveMass > 0.0f &&
        isfinite(inverseEffectiveMass) && finite3(normal);
}

inline float3 selfFrictionRelativeVelocity(
    threadgroup const float3* velocities,
    const float4 weights,
    const uint4 nodes
) {
    float3 first = float3(0.0f);
    float3 second = float3(0.0f);
    for (uint slot = 0u; slot < 4u; ++slot) {
        if (slot < 2u) {
            first += weights[slot] * velocities[nodes[slot]];
        } else {
            second += weights[slot] * velocities[nodes[slot]];
        }
    }
    return second - first;
}

// Rebuild the final contact shell cooperatively, compact active contacts, and
// solve their velocity cones with a bounded projected Gauss-Seidel iteration.
// Every contact retains its accumulated tangent impulse, so alternating
// sweeps can remove network slip without exceeding the total Coulomb disk.
// Pair discovery remains SIMD32-wide; lane zero owns canonical physical
// updates for deterministic replay and equal-and-opposite node impulses.
inline void applySelfContactFrictionCooperative(
    const MRRodGPUDispatch dispatch,
    device const float* inverseMasses,
    threadgroup const float3* positions,
    threadgroup const float3* unconstrainedVelocities,
    threadgroup float3* constrainedVelocities,
    threadgroup atomic_uint* candidateWords,
    threadgroup uint* contactPairs,
    threadgroup float3* accumulatedImpulses,
    threadgroup float* normalImpulses,
    threadgroup uint& contactCount,
    threadgroup atomic_uint& failure,
    const uint lane,
    const uint laneCount
) {
    for (uint word = lane;
         word < MR_ROD_GPU_SELF_CONTACT_PAIR_WORDS;
         word += laneCount) {
        atomic_store_explicit(
            &candidateWords[word],
            0u,
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float contactThreshold =
        2.0f * dispatch.selfCollision.x +
        dispatch.selfCollision.y +
        dispatch.dampingDerivativeTolerance.w;
    for (uint firstEdge = lane;
         firstEdge + 2u < dispatch.edgeCount;
         firstEdge += laneCount) {
        for (uint secondEdge = firstEdge + 2u;
             secondEdge < dispatch.edgeCount;
             ++secondEdge) {
            const uint pairOrdinal =
                firstEdge * (
                    2u * dispatch.edgeCount - firstEdge - 3u
                ) / 2u +
                secondEdge - firstEdge - 2u;
            RodClosestSegments closest;
            if (!closestRodSegments(
                    positions[firstEdge],
                    positions[firstEdge + 1u],
                    positions[secondEdge],
                    positions[secondEdge + 1u],
                    closest
                )) {
                recordFailure(
                    failure,
                    MR_ROD_GPU_DEGENERATE_GEOMETRY
                );
                continue;
            }
            if (closest.distance <= contactThreshold) {
                atomic_fetch_or_explicit(
                    &candidateWords[pairOrdinal >> 5u],
                    1u << (pairOrdinal & 31u),
                    memory_order_relaxed
                );
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        contactCount = 0u;
        const uint activePairCount =
            dispatch.edgeCount >= 2u
            ? (
                (dispatch.edgeCount - 1u) *
                (dispatch.edgeCount - 2u)
            ) / 2u
            : 0u;
        uint firstEdge = 0u;
        uint rowStart = 0u;
        uint rowLength = dispatch.edgeCount >= 2u
            ? dispatch.edgeCount - 2u
            : 0u;
        for (uint word = 0u;
             word < MR_ROD_GPU_SELF_CONTACT_PAIR_WORDS;
             ++word) {
            uint candidates = atomic_load_explicit(
                &candidateWords[word],
                memory_order_relaxed
            );
            while (candidates != 0u) {
                const uint bit = ctz(candidates);
                const uint pairOrdinal = 32u * word + bit;
                if (pairOrdinal >= activePairCount) {
                    break;
                }
                while (pairOrdinal >= rowStart + rowLength) {
                    rowStart += rowLength;
                    ++firstEdge;
                    --rowLength;
                }
                const uint secondEdge =
                    firstEdge + 2u + pairOrdinal - rowStart;
                RodClosestSegments closest;
                float3 normal;
                float4 weights;
                uint4 nodes;
                float inverseEffectiveMass = 0.0f;
                if (!selfFrictionKinematics(
                        firstEdge,
                        secondEdge,
                        inverseMasses,
                        positions,
                        closest,
                        normal,
                        weights,
                        nodes,
                        inverseEffectiveMass
                    )) {
                    recordFailure(
                        failure,
                        MR_ROD_GPU_DEGENERATE_GEOMETRY
                    );
                    candidates &= candidates - 1u;
                    continue;
                }
                const float3 relativeVelocity =
                    selfFrictionRelativeVelocity(
                        constrainedVelocities,
                        weights,
                        nodes
                    );
                const float3 unconstrainedRelativeVelocity =
                    selfFrictionRelativeVelocity(
                        unconstrainedVelocities,
                        weights,
                        nodes
                    );
                const float normalImpulse = max(
                    dot(
                        relativeVelocity -
                            unconstrainedRelativeVelocity,
                        normal
                    ),
                    0.0f
                ) / inverseEffectiveMass;
                if (normalImpulse > 0.0f) {
                    if (contactCount >=
                            MR_ROD_GPU_MAX_SELF_FRICTION_CONTACTS) {
                        recordFailure(
                            failure,
                            MR_ROD_GPU_SELF_FRICTION_CAPACITY_OVERFLOW
                        );
                    } else {
                        contactPairs[contactCount] =
                            firstEdge | (secondEdge << 16u);
                        accumulatedImpulses[contactCount] =
                            float3(0.0f);
                        normalImpulses[contactCount] = normalImpulse;
                        ++contactCount;
                    }
                }
                candidates &= candidates - 1u;
            }
        }

        if (atomic_load_explicit(
                &failure,
                memory_order_relaxed
            ) == MR_ROD_GPU_SUCCESS) {
            for (uint iteration = 0u;
                 iteration < MR_ROD_GPU_SELF_FRICTION_ITERATIONS;
                 ++iteration) {
                for (uint order = 0u;
                     order < contactCount;
                     ++order) {
                    const uint contactIndex =
                        (iteration & 1u) == 0u
                        ? order
                        : contactCount - 1u - order;
                    const uint packedPair =
                        contactPairs[contactIndex];
                    const uint contactFirstEdge =
                        packedPair & 0xffffu;
                    const uint contactSecondEdge =
                        packedPair >> 16u;
                    RodClosestSegments closest;
                    float3 normal;
                    float4 weights;
                    uint4 nodes;
                    float inverseEffectiveMass = 0.0f;
                    if (!selfFrictionKinematics(
                            contactFirstEdge,
                            contactSecondEdge,
                            inverseMasses,
                            positions,
                            closest,
                            normal,
                            weights,
                            nodes,
                            inverseEffectiveMass
                        )) {
                        recordFailure(
                            failure,
                            MR_ROD_GPU_DEGENERATE_GEOMETRY
                        );
                        continue;
                    }
                    const float3 relativeVelocity =
                        selfFrictionRelativeVelocity(
                            constrainedVelocities,
                            weights,
                            nodes
                        );
                    const float3 tangentVelocity =
                        relativeVelocity -
                        normal * dot(relativeVelocity, normal);
                    float3 projectedImpulse =
                        accumulatedImpulses[contactIndex] -
                        tangentVelocity / inverseEffectiveMass;
                    const float frictionLimit =
                        dispatch.selfCollision.w *
                        normalImpulses[contactIndex];
                    const float projectedMagnitude =
                        length(projectedImpulse);
                    if (projectedMagnitude > frictionLimit &&
                        projectedMagnitude > 0.0f) {
                        projectedImpulse *=
                            frictionLimit / projectedMagnitude;
                    }
                    const float3 deltaImpulse =
                        projectedImpulse -
                        accumulatedImpulses[contactIndex];
                    accumulatedImpulses[contactIndex] =
                        projectedImpulse;
                    for (uint slot = 0u; slot < 4u; ++slot) {
                        const float sign =
                            slot < 2u ? -1.0f : 1.0f;
                        constrainedVelocities[nodes[slot]] +=
                            sign * weights[slot] * deltaImpulse *
                            inverseMasses[nodes[slot]];
                        if (!finite3(
                                constrainedVelocities[nodes[slot]]
                            )) {
                            recordFailure(
                                failure,
                                MR_ROD_GPU_NONFINITE_RESULT
                            );
                        }
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Attachments run after self-contact. Predict the exact positional correction
// of any contact they introduce so convergence cannot exit before the next
// deterministic projection sweep.
inline void recordPendingSelfContactCorrection(
    const MRRodGPUDispatch dispatch,
    device const float* inverseMasses,
    device const MRRodGPUAttachment* attachments,
    const uint attachmentBase,
    threadgroup float3* positions,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits,
    threadgroup atomic_uint& maximumPenetrationBits,
    const uint lane,
    const uint laneCount
) {
    const float contactDistance =
        2.0f * dispatch.selfCollision.x +
        dispatch.selfCollision.y;
    for (uint attachmentIndex = lane;
         attachmentIndex < dispatch.attachmentCount;
         attachmentIndex += laneCount) {
        const uint attachedNode =
            attachments[
                attachmentBase + attachmentIndex
            ].nodeIndex;
        if (attachedNode >= dispatch.nodeCount) {
            recordFailure(
                failure,
                MR_ROD_GPU_INVALID_DISPATCH
            );
            continue;
        }
        // Attachment projection changes exactly one node. Only its one or
        // two incident edges can have a new separation after the complete
        // self-contact sweep above; all other edge pairs are unchanged.
        for (uint incident = 0u; incident < 2u; ++incident) {
            if ((incident == 0u && attachedNode == 0u) ||
                (incident == 1u &&
                 attachedNode >= dispatch.edgeCount)) {
                continue;
            }
            const uint attachedEdge = incident == 0u
                ? attachedNode - 1u
                : attachedNode;
            for (uint otherEdge = 0u;
                 otherEdge < dispatch.edgeCount;
                 ++otherEdge) {
                const uint firstEdge = min(attachedEdge, otherEdge);
                const uint secondEdge = max(attachedEdge, otherEdge);
                if (secondEdge < firstEdge + 2u) {
                    continue;
                }
                RodClosestSegments closest;
                if (!closestRodSegments(
                        positions[firstEdge],
                        positions[firstEdge + 1u],
                        positions[secondEdge],
                        positions[secondEdge + 1u],
                        closest
                    )) {
                    recordFailure(
                        failure,
                        MR_ROD_GPU_DEGENERATE_GEOMETRY
                    );
                    continue;
                }
                const float penetration =
                    contactDistance - closest.distance;
                if (!(penetration > 0.0f)) {
                    continue;
                }
                const float4 weights = float4(
                    1.0f - closest.first,
                    closest.first,
                    1.0f - closest.second,
                    closest.second
                );
                const uint4 nodes = uint4(
                    firstEdge,
                    firstEdge + 1u,
                    secondEdge,
                    secondEdge + 1u
                );
                float denominator = 0.0f;
                for (uint slot = 0u; slot < 4u; ++slot) {
                    denominator +=
                        weights[slot] * weights[slot] *
                        inverseMasses[nodes[slot]];
                }
                const float alpha =
                    dispatch.selfCollision.z /
                    (
                        dispatch.gravityAndTimestep.w *
                        dispatch.gravityAndTimestep.w
                    );
                if (!(denominator + alpha > 0.0f) ||
                    !isfinite(denominator) || !isfinite(alpha)) {
                    recordFailure(
                        failure,
                        MR_ROD_GPU_NONFINITE_RESULT
                    );
                    continue;
                }
                const float lambda =
                    penetration / (denominator + alpha);
                float maximumCorrection = 0.0f;
                for (uint slot = 0u; slot < 4u; ++slot) {
                    maximumCorrection = max(
                        maximumCorrection,
                        abs(weights[slot]) * lambda *
                            inverseMasses[nodes[slot]]
                    );
                }
                recordPositiveMaximum(
                    maximumErrorBits,
                    penetration
                );
                recordPositiveMaximum(
                    maximumPenetrationBits,
                    penetration
                );
                recordPositiveMaximum(
                    maximumCorrectionBits,
                    maximumCorrection
                );
            }
        }
    }
}

inline void projectBend(
    const uint constraintIndex,
    const float timestep,
    const float derivativeScale,
    device const float4* restCurvature,
    device const float* restLengths,
    device const float4* referenceTangents,
    device const float4* referenceDirectors,
    device const float* inverseMasses,
    device const float* inverseRotationalInertias,
    device const float* bendStiffness,
    threadgroup float3* positions,
    threadgroup float* twists,
    threadgroup float2* accumulatedMultipliers,
    threadgroup atomic_uint& failure,
    threadgroup atomic_uint& maximumErrorBits,
    threadgroup atomic_uint& maximumCorrectionBits
) {
    float3 localPositions[3] = {
        positions[constraintIndex],
        positions[constraintIndex + 1u],
        positions[constraintIndex + 2u],
    };
    float localTwists[2] = {
        twists[constraintIndex],
        twists[constraintIndex + 1u],
    };
    float2 current;
    if (!localCurvature(
            localPositions,
            localTwists,
            referenceTangents[constraintIndex].xyz,
            referenceDirectors[constraintIndex].xyz,
            current
        )) {
        recordFailure(
            failure,
            MR_ROD_GPU_DEGENERATE_GEOMETRY
        );
        return;
    }
    const float2 constraint =
        current - restCurvature[constraintIndex].xy;
    const bool zeroIntrinsicCurvature =
        dot(
            restCurvature[constraintIndex].xy,
            restCurvature[constraintIndex].xy
        ) <= 1.0e-12f;
    float3 positionGradient0[3] = {
        float3(0.0f),
        float3(0.0f),
        float3(0.0f),
    };
    float3 positionGradient1[3] = {
        float3(0.0f),
        float3(0.0f),
        float3(0.0f),
    };
    float twistGradient0[2] = {0.0f, 0.0f};
    float twistGradient1[2] = {0.0f, 0.0f};
    for (uint node = 0u; node < 3u; ++node) {
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float scale = max(
                max(
                    restLengths[constraintIndex],
                    restLengths[constraintIndex + 1u]
                ),
                max(abs(localPositions[node][axis]), 1.0e-3f)
            );
            const float step = derivativeScale * scale;
            localPositions[node][axis] += step;
            float2 plus;
            const bool plusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents[constraintIndex].xyz,
                referenceDirectors[constraintIndex].xyz,
                plus
            );
            localPositions[node][axis] -= 2.0f * step;
            float2 minus;
            const bool minusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents[constraintIndex].xyz,
                referenceDirectors[constraintIndex].xyz,
                minus
            );
            localPositions[node][axis] += step;
            if (!plusOk || !minusOk) {
                recordFailure(
                    failure,
                    MR_ROD_GPU_DEGENERATE_GEOMETRY
                );
                return;
            }
            const float2 derivative =
                (plus - minus) / (2.0f * step);
            positionGradient0[node][axis] = derivative.x;
            positionGradient1[node][axis] = derivative.y;
        }
    }
    // Straight circular rods have isotropic EI*kappa^2 bending energy.
    // Their material spin belongs exclusively to the GJ torsion block;
    // retaining noisy finite-difference twist gradients here can pump the
    // thread's legitimately tiny polar inertia.
    if (!zeroIntrinsicCurvature) {
        for (uint edge = 0u; edge < 2u; ++edge) {
            const float step = derivativeScale;
            localTwists[edge] += step;
            float2 plus;
            const bool plusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents[constraintIndex].xyz,
                referenceDirectors[constraintIndex].xyz,
                plus
            );
            localTwists[edge] -= 2.0f * step;
            float2 minus;
            const bool minusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents[constraintIndex].xyz,
                referenceDirectors[constraintIndex].xyz,
                minus
            );
            localTwists[edge] += step;
            if (!plusOk || !minusOk) {
                recordFailure(
                    failure,
                    MR_ROD_GPU_DEGENERATE_GEOMETRY
                );
                return;
            }
            const float2 derivative =
                (plus - minus) / (2.0f * step);
            twistGradient0[edge] = derivative.x;
            twistGradient1[edge] = derivative.y;
        }
    }

    float effective00 = 0.0f;
    float effective01 = 0.0f;
    float effective11 = 0.0f;
    for (uint node = 0u; node < 3u; ++node) {
        const float inverseMass =
            inverseMasses[constraintIndex + node];
        effective00 +=
            dot(
                positionGradient0[node],
                positionGradient0[node]
            ) * inverseMass;
        effective01 +=
            dot(
                positionGradient0[node],
                positionGradient1[node]
            ) * inverseMass;
        effective11 +=
            dot(
                positionGradient1[node],
                positionGradient1[node]
            ) * inverseMass;
    }
    if (!zeroIntrinsicCurvature) {
        for (uint edge = 0u; edge < 2u; ++edge) {
            const float inverseInertia =
                inverseRotationalInertias[
                    constraintIndex + edge
                ];
            effective00 +=
                twistGradient0[edge] *
                twistGradient0[edge] *
                inverseInertia;
            effective01 +=
                twistGradient0[edge] *
                twistGradient1[edge] *
                inverseInertia;
            effective11 +=
                twistGradient1[edge] *
                twistGradient1[edge] *
                inverseInertia;
        }
    }
    const float voronoi =
        0.5f * (
            restLengths[constraintIndex] +
            restLengths[constraintIndex + 1u]
        );
    const float alpha =
        voronoi /
        bendStiffness[constraintIndex] /
        (timestep * timestep);
    effective00 += alpha;
    effective11 += alpha;
    const float determinant =
        effective00 * effective11 -
        effective01 * effective01;
    const float determinantScale = max(
        effective00 * effective11,
        1.0f
    );
    if (!(effective00 > 0.0f) ||
        !(effective11 > 0.0f) ||
        !(determinant >
            32.0f * kFloatEpsilon * determinantScale) ||
        !isfinite(effective00) ||
        !isfinite(effective01) ||
        !isfinite(effective11) ||
        !isfinite(determinant)) {
        recordFailure(
            failure,
            MR_ROD_GPU_DEGENERATE_GEOMETRY
        );
        return;
    }
    // The two material-curvature coordinates share every position and twist
    // degree of freedom. Solve their complete symmetric block instead of
    // discarding the cross response and repeating the derivative pass.
    const float2 rhs =
        constraint +
        alpha * accumulatedMultipliers[constraintIndex];
    const float2 lambda = float2(
        (
            -effective11 * rhs.x +
            effective01 * rhs.y
        ) / determinant,
        (
            effective01 * rhs.x -
            effective00 * rhs.y
        ) / determinant
    );
    accumulatedMultipliers[constraintIndex] += lambda;
    for (uint node = 0u; node < 3u; ++node) {
        const float3 correction =
            inverseMasses[constraintIndex + node] *
            (
                lambda.x * positionGradient0[node] +
                lambda.y * positionGradient1[node]
            );
        positions[constraintIndex + node] += correction;
        recordPositiveMaximum(
            maximumCorrectionBits,
            length(correction)
        );
    }
    if (!zeroIntrinsicCurvature) {
        for (uint edge = 0u; edge < 2u; ++edge) {
            twists[constraintIndex + edge] +=
                inverseRotationalInertias[
                    constraintIndex + edge
                ] * (
                    lambda.x * twistGradient0[edge] +
                    lambda.y * twistGradient1[edge]
                );
        }
    }
    recordPositiveMaximum(
        maximumErrorBits,
        max(abs(constraint.x), abs(constraint.y))
    );
}

} // namespace

// Resolve immutable homogeneous-world body bindings into per-environment
// attachment targets. The rod kernel consumes only this canonical record, so
// explicit and rigid targets retain identical constraint semantics.
kernel void mr_resolve_rod_rigid_attachments(
    device const MRRodGPUDispatch& dispatch [[buffer(0)]],
    device const MRRodGPUAttachment* inputAttachments [[buffer(1)]],
    device const MRRodGPURigidBinding* bindings [[buffer(2)]],
    device const MRBodyStateGPU* rigidBodies [[buffer(3)]],
    device MRRodGPUAttachment* resolvedAttachments [[buffer(4)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint count =
        dispatch.environmentCount * dispatch.attachmentCount;
    if (globalIndex >= count) {
        return;
    }
    const uint environment =
        globalIndex / dispatch.attachmentCount;
    const uint attachmentIndex =
        globalIndex - environment * dispatch.attachmentCount;
    MRRodGPUAttachment resolved =
        inputAttachments[globalIndex];
    const MRRodGPURigidBinding binding =
        bindings[attachmentIndex];
    if (binding.bodyIndex != MR_ROD_GPU_INVALID_BODY) {
        const MRBodyStateGPU body =
            rigidBodies[
                environment * dispatch.stateBodyStride +
                binding.bodyIndex
            ];
        const float3 worldOffset = rotateQuaternion(
            body.orientation,
            binding.localAnchor.xyz
        );
        resolved.targetAndCompliance.xyz =
            body.position.xyz + worldOffset;
        resolved.velocity.xyz =
            body.linearVelocityAndInverseMass.xyz +
            cross(body.angularVelocity.xyz, worldOffset);
    }
    resolvedAttachments[globalIndex] = resolved;
}

// One threadgroup owns one environment and retains the complete rod state in
// threadgroup memory. Constraint coloring makes every phase write-disjoint:
// two colors for edges/twist, three for bend triples. No floating atomics
// participate in the physical update; atomics only reduce diagnostics.
kernel void mr_discrete_elastic_rod_step(
    device const MRRodGPUDispatch& sourceDispatch [[buffer(0)]],
    device const float* restLengths [[buffer(1)]],
    device const float* restTwists [[buffer(2)]],
    device const float4* restCurvature [[buffer(3)]],
    device const float* inverseMasses [[buffer(4)]],
    device const float* inverseRotationalInertias [[buffer(5)]],
    device const float* stretchStiffness [[buffer(6)]],
    device const float* bendStiffness [[buffer(7)]],
    device const float* twistStiffness [[buffer(8)]],
    device const float4* inputPositions [[buffer(9)]],
    device const float4* inputVelocities [[buffer(10)]],
    device const float* inputTwists [[buffer(11)]],
    device const float* inputTwistRates [[buffer(12)]],
    device const MRRodGPUAttachment* attachments [[buffer(13)]],
    device float4* outputPositions [[buffer(14)]],
    device float4* outputVelocities [[buffer(15)]],
    device float* outputTwists [[buffer(16)]],
    device float* outputTwistRates [[buffer(17)]],
    device MRRodGPUStatus* statuses [[buffer(18)]],
    device MRRodGPUAttachmentReaction* reactions [[buffer(19)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(20)]],
    constant uint& eventSegmentMode [[buffer(21)]],
    device const float4* referenceTangents [[buffer(22)]],
    device const float4* referenceDirectors [[buffer(23)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint laneCount [[threads_per_threadgroup]]
) {
    MRRodGPUDispatch dispatch = sourceDispatch;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const bool eventSegment =
        eventSegmentMode != MR_CCD_SEGMENT_FULL_MICROSTEP;
    if (eventSegment) {
        const MRCCDEventStateGPU eventState =
            eventStates[environment];
        const float duration = rodEventSegmentDuration(
            eventStates[environment],
            eventSegmentMode
        );
        // Finished environments remain present in MLX's statically encoded
        // worker grid. Preserve their accepted input exactly instead of
        // attempting a zero-duration DER factorization.
        if ((eventState.flags & MR_CCD_EVENT_FINISHED) != 0u ||
            !(duration > 0.0f)) {
            const uint nodeBase =
                environment * dispatch.stateNodeStride;
            const uint edgeBase =
                environment * dispatch.stateEdgeStride;
            for (uint node = lane;
                 node < dispatch.nodeCount;
                 node += laneCount) {
                outputPositions[nodeBase + node] =
                    inputPositions[nodeBase + node];
                outputVelocities[nodeBase + node] =
                    inputVelocities[nodeBase + node];
            }
            for (uint edge = lane;
                 edge < dispatch.edgeCount;
                 edge += laneCount) {
                outputTwists[edgeBase + edge] =
                    inputTwists[edgeBase + edge];
                outputTwistRates[edgeBase + edge] =
                    inputTwistRates[edgeBase + edge];
            }
            if (lane == 0u) {
                MRRodGPUStatus status = {};
                status.environment = environment;
                status.code = MR_ROD_GPU_SUCCESS;
                statuses[environment] = status;
            }
            return;
        }
        dispatch.gravityAndTimestep.w = duration;
    }
    threadgroup float3 positions[MR_ROD_GPU_MAX_NODES];
    threadgroup float3 originalPositions[MR_ROD_GPU_MAX_NODES];
    threadgroup float3 iterationPositions[MR_ROD_GPU_MAX_NODES];
    threadgroup float3 velocities[MR_ROD_GPU_MAX_NODES];
    threadgroup float twists[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float originalTwists[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float iterationTwists[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float twistRates[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float stretchMultipliers[
        MR_ROD_GPU_MAX_NODES - 1u
    ];
    threadgroup float2 bendMultipliers[
        MR_ROD_GPU_MAX_NODES - 2u
    ];
    threadgroup float twistMultipliers[
        MR_ROD_GPU_MAX_NODES - 2u
    ];
    threadgroup float3 attachmentMultipliers[
        MR_ROD_GPU_MAX_ATTACHMENTS
    ];
    threadgroup atomic_uint selfContactCandidateWords[
        MR_ROD_GPU_SELF_CONTACT_PAIR_WORDS
    ];
    threadgroup uint selfFrictionPairs[
        MR_ROD_GPU_MAX_SELF_FRICTION_CONTACTS
    ];
    threadgroup float3 selfFrictionImpulses[
        MR_ROD_GPU_MAX_SELF_FRICTION_CONTACTS
    ];
    threadgroup float selfFrictionNormalImpulses[
        MR_ROD_GPU_MAX_SELF_FRICTION_CONTACTS
    ];
    threadgroup float3 targetImpulses[
        MR_ROD_GPU_MAX_ATTACHMENTS
    ];
    threadgroup atomic_uint failure;
    threadgroup atomic_uint maximumErrorBits;
    threadgroup atomic_uint maximumCorrectionBits;
    threadgroup atomic_uint maximumCorrectionOwnerKey;
    threadgroup atomic_uint maximumPenetrationBits;
    threadgroup atomic_uint projectedContactCount;
    threadgroup uint completedIterations;
    threadgroup uint converged;
    threadgroup uint selfFrictionContactCount;

    if (lane == 0u) {
        atomic_store_explicit(
            &failure,
            uint(MR_ROD_GPU_SUCCESS),
            memory_order_relaxed
        );
        atomic_store_explicit(
            &maximumErrorBits,
            0u,
            memory_order_relaxed
        );
        atomic_store_explicit(
            &maximumCorrectionBits,
            0u,
            memory_order_relaxed
        );
        atomic_store_explicit(
            &maximumCorrectionOwnerKey,
            0u,
            memory_order_relaxed
        );
        atomic_store_explicit(
            &maximumPenetrationBits,
            0u,
            memory_order_relaxed
        );
        atomic_store_explicit(
            &projectedContactCount,
            0u,
            memory_order_relaxed
        );
        completedIterations = 0u;
        converged = 0u;
        if (dispatch.abiVersion != MR_ROD_GPU_ABI_VERSION ||
            dispatch.environmentCount == 0u ||
            dispatch.nodeCount < 2u ||
            dispatch.nodeCount > MR_ROD_GPU_MAX_NODES ||
            dispatch.edgeCount + 1u != dispatch.nodeCount ||
            dispatch.attachmentCount >
                MR_ROD_GPU_MAX_ATTACHMENTS ||
            dispatch.solverIterations == 0u ||
            dispatch.stateNodeStride < dispatch.nodeCount ||
            dispatch.stateEdgeStride < dispatch.edgeCount ||
            dispatch.stateBodyStride <
                dispatch.rigidBodyCount ||
            (dispatch.flags &
                ~uint(
                    MR_ROD_GPU_FLAG_SELF_COLLISION |
                    MR_ROD_GPU_FLAG_TOOL_COLLISION |
                    MR_ROD_GPU_FLAG_ENABLE_CCD |
                    MR_ROD_GPU_FLAG_TOOL_WARM_START
                )) !=
                    0u ||
            !(dispatch.gravityAndTimestep.w > 0.0f) ||
            !all(isfinite(dispatch.gravityAndTimestep)) ||
            any(dispatch.dampingDerivativeTolerance < 0.0f) ||
            !(dispatch.dampingDerivativeTolerance.z > 0.0f) ||
            !(dispatch.dampingDerivativeTolerance.w > 0.0f) ||
            !all(isfinite(
                dispatch.dampingDerivativeTolerance
            )) ||
            !all(isfinite(dispatch.selfCollision)) ||
            any(dispatch.selfCollision < 0.0f) ||
            (
                (dispatch.flags &
                    MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u &&
                !(dispatch.selfCollision.x > 0.0f)
            ) ||
            (
                (dispatch.flags &
                    MR_ROD_GPU_FLAG_TOOL_COLLISION) != 0u &&
                (
                    dispatch.toolShapeCount == 0u ||
                    dispatch.toolPairCount == 0u ||
                    dispatch.toolContactIterations == 0u ||
                    dispatch.toolContactStride <
                        dispatch.toolPairCount *
                            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR ||
                    any(dispatch.toolContact < 0.0f) ||
                    any(dispatch.toolResponse < 0.0f) ||
                    dispatch.toolResponse.x > 1.0f ||
                    !(dispatch.toolResponse.z > 0.0f) ||
                    !(dispatch.toolResponse.w > 0.0f)
                )
            )) {
            atomic_store_explicit(
                &failure,
                uint(MR_ROD_GPU_INVALID_DISPATCH),
                memory_order_relaxed
            );
        }
    }
    for (uint attachmentIndex = lane;
         attachmentIndex < dispatch.attachmentCount;
         attachmentIndex += laneCount) {
        attachmentMultipliers[attachmentIndex] = float3(0.0f);
        targetImpulses[attachmentIndex] = float3(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nodeBase =
        environment * dispatch.stateNodeStride;
    const uint edgeBase =
        environment * dispatch.stateEdgeStride;
    for (uint node = lane;
         node < dispatch.nodeCount;
         node += laneCount) {
        const float4 position = inputPositions[nodeBase + node];
        const float4 velocity = inputVelocities[nodeBase + node];
        if (!all(isfinite(position)) ||
            !all(isfinite(velocity)) ||
            position.w != 1.0f ||
            velocity.w != 0.0f) {
            recordFailure(
                failure,
                MR_ROD_GPU_NONFINITE_RESULT
            );
        }
        originalPositions[node] = position.xyz;
        velocities[node] =
            velocity.xyz +
            dispatch.gravityAndTimestep.xyz *
                dispatch.gravityAndTimestep.w;
        positions[node] =
            position.xyz +
            velocities[node] * dispatch.gravityAndTimestep.w;
    }
    for (uint edge = lane;
         edge < dispatch.edgeCount;
         edge += laneCount) {
        const float twist = inputTwists[edgeBase + edge];
        const float rate = inputTwistRates[edgeBase + edge];
        if (!isfinite(twist) || !isfinite(rate)) {
            recordFailure(
                failure,
                MR_ROD_GPU_NONFINITE_RESULT
            );
        }
        originalTwists[edge] = twist;
        twistRates[edge] = rate;
        twists[edge] =
            twist + dispatch.gravityAndTimestep.w * rate;
        stretchMultipliers[edge] = 0.0f;
        if (edge + 1u < dispatch.edgeCount) {
            bendMultipliers[edge] = float2(0.0f);
            twistMultipliers[edge] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint iteration = 0u;
         iteration < dispatch.solverIterations;
         ++iteration) {
        if (lane == 0u) {
            atomic_store_explicit(
                &maximumErrorBits,
                0u,
                memory_order_relaxed
            );
            atomic_store_explicit(
                &maximumCorrectionBits,
                0u,
                memory_order_relaxed
            );
            atomic_store_explicit(
                &maximumCorrectionOwnerKey,
                0u,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(
                &failure,
                memory_order_relaxed
            ) != MR_ROD_GPU_SUCCESS) {
            break;
        }
        for (uint node = lane;
             node < dispatch.nodeCount;
             node += laneCount) {
            iterationPositions[node] = positions[node];
        }
        for (uint edge = lane;
             edge < dispatch.edgeCount;
             edge += laneCount) {
            iterationTwists[edge] = twists[edge];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint attachmentBase =
            environment * dispatch.attachmentCount;
        if (dispatch.nodeCount <= 16u) {
            // A short surgical thread has too little work per color to repay
            // Jacobi-like convergence loss. One deterministic cohort owner
            // applies the same ordered nonlinear GS sweep as the CPU oracle;
            // batching remains across environment threadgroups.
            if (lane == 0u) {
                for (uint edge = 0u;
                     edge < dispatch.edgeCount;
                     ++edge) {
                    projectStretch(
                        edge,
                        dispatch.gravityAndTimestep.w,
                        restLengths,
                        inverseMasses,
                        stretchStiffness,
                        positions,
                        stretchMultipliers,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                for (uint constraintIndex = 0u;
                     constraintIndex + 1u <
                         dispatch.edgeCount;
                     ++constraintIndex) {
                    projectBend(
                        constraintIndex,
                        dispatch.gravityAndTimestep.w,
                        dispatch.dampingDerivativeTolerance.z,
                        restCurvature,
                        restLengths,
                        referenceTangents,
                        referenceDirectors,
                        inverseMasses,
                        inverseRotationalInertias,
                        bendStiffness,
                        positions,
                        twists,
                        bendMultipliers,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                    projectTwist(
                        constraintIndex,
                        dispatch.gravityAndTimestep.w,
                        restTwists,
                        restLengths,
                        inverseRotationalInertias,
                        twistStiffness,
                        twists,
                        twistMultipliers,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                if ((dispatch.flags &
                     MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u) {
                    for (uint firstEdge = 0u;
                         firstEdge < dispatch.edgeCount;
                         ++firstEdge) {
                        for (
                            uint secondEdge = firstEdge + 2u;
                            secondEdge < dispatch.edgeCount;
                            ++secondEdge
                        ) {
                            projectSelfContact(
                                firstEdge,
                                secondEdge,
                                dispatch,
                                inverseMasses,
                                positions,
                                failure,
                                maximumErrorBits,
                                maximumCorrectionBits,
                                maximumPenetrationBits,
                                projectedContactCount
                            );
                        }
                    }
                }
                for (uint attachmentIndex = 0u;
                     attachmentIndex < dispatch.attachmentCount;
                     ++attachmentIndex) {
                    projectAttachment(
                        attachments[
                            attachmentBase + attachmentIndex
                        ],
                        attachmentIndex,
                        dispatch.gravityAndTimestep.w,
                        inverseMasses,
                        positions,
                        attachmentMultipliers,
                        targetImpulses,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (uint color = 0u; color < 2u; ++color) {
                for (uint edge = color + 2u * lane;
                     edge < dispatch.edgeCount;
                     edge += 2u * laneCount) {
                    projectStretch(
                        edge,
                        dispatch.gravityAndTimestep.w,
                        restLengths,
                        inverseMasses,
                        stretchStiffness,
                        positions,
                        stretchMultipliers,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint color = 0u; color < 3u; ++color) {
                for (
                    uint constraintIndex = color + 3u * lane;
                    constraintIndex + 1u <
                        dispatch.edgeCount;
                    constraintIndex += 3u * laneCount
                ) {
                    projectBend(
                        constraintIndex,
                        dispatch.gravityAndTimestep.w,
                        dispatch.dampingDerivativeTolerance.z,
                        restCurvature,
                        restLengths,
                        referenceTangents,
                        referenceDirectors,
                        inverseMasses,
                        inverseRotationalInertias,
                        bendStiffness,
                        positions,
                        twists,
                        bendMultipliers,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                threadgroup_barrier(
                    mem_flags::mem_threadgroup
                );
            }
            for (uint color = 0u; color < 2u; ++color) {
                for (
                    uint constraintIndex = color + 2u * lane;
                    constraintIndex + 1u < dispatch.edgeCount;
                    constraintIndex += 2u * laneCount
                ) {
                    projectTwist(
                        constraintIndex,
                        dispatch.gravityAndTimestep.w,
                        restTwists,
                        restLengths,
                        inverseRotationalInertias,
                        twistStiffness,
                        twists,
                        twistMultipliers,
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if ((dispatch.flags &
                 MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u) {
                projectSelfContactsCooperative(
                    dispatch,
                    inverseMasses,
                    positions,
                    selfContactCandidateWords,
                    failure,
                    maximumErrorBits,
                    maximumCorrectionBits,
                    maximumPenetrationBits,
                    projectedContactCount,
                    lane,
                    laneCount
                );
            }
            for (uint attachmentIndex = lane;
                 attachmentIndex < dispatch.attachmentCount;
                 attachmentIndex += laneCount) {
                projectAttachment(
                    attachments[
                        attachmentBase + attachmentIndex
                    ],
                    attachmentIndex,
                    dispatch.gravityAndTimestep.w,
                    inverseMasses,
                    positions,
                    attachmentMultipliers,
                    targetImpulses,
                    failure,
                    maximumErrorBits,
                    maximumCorrectionBits
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            atomic_store_explicit(
                &maximumCorrectionBits,
                0u,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((dispatch.flags &
             MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u) {
            recordPendingSelfContactCorrection(
                dispatch,
                inverseMasses,
                attachments,
                attachmentBase,
                positions,
                failure,
                maximumErrorBits,
                maximumCorrectionBits,
                maximumPenetrationBits,
                lane,
                laneCount
            );
        }
        for (uint node = lane;
             node < dispatch.nodeCount;
             node += laneCount) {
            const float correction = length(
                positions[node] - iterationPositions[node]
            );
            recordPositiveMaximum(
                maximumCorrectionBits,
                correction
            );
            recordCorrectionOwner(
                maximumCorrectionOwnerKey,
                correction,
                node
            );
        }
        for (uint edge = lane;
             edge < dispatch.edgeCount;
             edge += laneCount) {
            const float correction =
                abs(twists[edge] - iterationTwists[edge]);
            recordPositiveMaximum(
                maximumCorrectionBits,
                dispatch.selfCollision.x * correction
            );
            recordCorrectionOwner(
                maximumCorrectionOwnerKey,
                dispatch.selfCollision.x * correction,
                MR_ROD_GPU_TWIST_CORRECTION_INDEX_BASE + edge
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            completedIterations = iteration + 1u;
            const float correction = as_type<float>(
                atomic_load_explicit(
                    &maximumCorrectionBits,
                    memory_order_relaxed
                )
            );
            if (atomic_load_explicit(
                    &failure,
                    memory_order_relaxed
                ) == MR_ROD_GPU_SUCCESS &&
                correction <=
                    dispatch.dampingDerivativeTolerance.w) {
                converged = 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (converged != 0u) {
            break;
        }
    }

    if (lane == 0u &&
        atomic_load_explicit(
            &failure,
            memory_order_relaxed
        ) == MR_ROD_GPU_SUCCESS &&
        converged == 0u) {
        atomic_store_explicit(
            &failure,
            uint(MR_ROD_GPU_DID_NOT_CONVERGE),
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint projectionFailureCode = atomic_load_explicit(
        &failure,
        memory_order_relaxed
    );
    const float inverseTimestep =
        1.0f / dispatch.gravityAndTimestep.w;
    const float linearDecay = exp(
        -dispatch.dampingDerivativeTolerance.x *
        dispatch.gravityAndTimestep.w
    );
    const float twistDecay = exp(
        -dispatch.dampingDerivativeTolerance.y *
        dispatch.gravityAndTimestep.w
    );
    for (uint node = lane;
         node < dispatch.nodeCount;
         node += laneCount) {
        iterationPositions[node] =
            projectionFailureCode == MR_ROD_GPU_SUCCESS
            ? (
                positions[node] - originalPositions[node]
            ) * (linearDecay * inverseTimestep)
            : inputVelocities[nodeBase + node].xyz;
        velocities[node] =
            projectionFailureCode == MR_ROD_GPU_SUCCESS
            ? velocities[node] * linearDecay
            : inputVelocities[nodeBase + node].xyz;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (projectionFailureCode == MR_ROD_GPU_SUCCESS &&
        (dispatch.flags & MR_ROD_GPU_FLAG_SELF_COLLISION) != 0u &&
        dispatch.selfCollision.w > 0.0f &&
        atomic_load_explicit(
            &projectedContactCount,
            memory_order_relaxed
        ) > 0u) {
        applySelfContactFrictionCooperative(
            dispatch,
            inverseMasses,
            positions,
            velocities,
            iterationPositions,
            selfContactCandidateWords,
            selfFrictionPairs,
            selfFrictionImpulses,
            selfFrictionNormalImpulses,
            selfFrictionContactCount,
            failure,
            lane,
            laneCount
        );
    }
    const uint failureCode = atomic_load_explicit(
        &failure,
        memory_order_relaxed
    );
    for (uint node = lane;
         node < dispatch.nodeCount;
         node += laneCount) {
        float3 position = failureCode == MR_ROD_GPU_SUCCESS
            ? positions[node]
            : originalPositions[node];
        float3 velocity = failureCode == MR_ROD_GPU_SUCCESS
            ? iterationPositions[node]
            : inputVelocities[nodeBase + node].xyz;
        if (failureCode == MR_ROD_GPU_SUCCESS) {
            const uint attachmentBase =
                environment * dispatch.attachmentCount;
            for (uint attachmentIndex = 0u;
                 attachmentIndex < dispatch.attachmentCount;
                 ++attachmentIndex) {
                const MRRodGPUAttachment attachment =
                    attachments[
                        attachmentBase + attachmentIndex
                    ];
                if (attachment.nodeIndex == node &&
                    attachment.targetAndCompliance.w == 0.0f) {
                    targetImpulses[attachmentIndex] -=
                        (
                            attachment.velocity.xyz -
                            velocity
                        ) / inverseMasses[node];
                    position = attachment.targetAndCompliance.xyz;
                    velocity = attachment.velocity.xyz;
                }
            }
        }
        outputPositions[nodeBase + node] =
            float4(position, 1.0f);
        outputVelocities[nodeBase + node] =
            float4(velocity, 0.0f);
    }
    for (uint edge = lane;
         edge < dispatch.edgeCount;
         edge += laneCount) {
        const float twist = failureCode == MR_ROD_GPU_SUCCESS
            ? twists[edge]
            : originalTwists[edge];
        outputTwists[edgeBase + edge] = twist;
        outputTwistRates[edgeBase + edge] =
            failureCode == MR_ROD_GPU_SUCCESS
            ? (
                twist - originalTwists[edge]
            ) * (twistDecay * inverseTimestep)
            : inputTwistRates[edgeBase + edge];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint attachmentIndex = lane;
         attachmentIndex < dispatch.attachmentCount;
         attachmentIndex += laneCount) {
        const uint globalAttachment =
            environment * dispatch.attachmentCount +
            attachmentIndex;
        const MRRodGPUAttachment attachment =
            attachments[globalAttachment];
        const float3 impulse =
            failureCode == MR_ROD_GPU_SUCCESS
            ? targetImpulses[attachmentIndex]
            : float3(0.0f);
        MRRodGPUAttachmentReaction reaction{};
        reaction.impulseAndError = float4(
            impulse,
            failureCode == MR_ROD_GPU_SUCCESS
            ? (
                attachment.targetAndCompliance.w == 0.0f
                ? 0.0f
                : length(
                    positions[attachment.nodeIndex] -
                    attachment.targetAndCompliance.xyz
                )
            )
            : 0.0f
        );
        reaction.averageForce = float4(
            impulse * inverseTimestep,
            0.0f
        );
        reaction.nodeIndex = attachment.nodeIndex;
        reaction.attachmentIndex = attachmentIndex;
        reaction.bodyIndex = MR_ROD_GPU_INVALID_BODY;
        reactions[globalAttachment] = reaction;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        MRRodGPUStatus status{};
        status.code = failureCode;
        status.environment = environment;
        status.iterations = completedIterations;
        const uint correctionOwnerKey = atomic_load_explicit(
            &maximumCorrectionOwnerKey,
            memory_order_relaxed
        );
        status.failingIndex =
            failureCode == MR_ROD_GPU_DID_NOT_CONVERGE &&
                correctionOwnerKey != 0u
            ? 255u - (correctionOwnerKey & 0xffu)
            : 0xffffffffu;
        status.diagnostics = float4(
            as_type<float>(atomic_load_explicit(
                &maximumErrorBits,
                memory_order_relaxed
            )),
            as_type<float>(atomic_load_explicit(
                &maximumCorrectionBits,
                memory_order_relaxed
            )),
            as_type<float>(atomic_load_explicit(
                &maximumPenetrationBits,
                memory_order_relaxed
            )),
            float(atomic_load_explicit(
                &projectedContactCount,
                memory_order_relaxed
            ))
        );
        statuses[environment] = status;
    }
}

// Apply the equal-and-opposite rod reaction at each rigid local anchor.
// Bindings are required to name disjoint bodies, so lane zero can update the
// small surgical coupling set in stable attachment order without atomics.
// All body candidates are copied before any reaction is applied; on a
// non-finite update the environment's rigid candidates and rod status roll
// back together.
kernel void mr_apply_rod_rigid_reactions(
    device const MRRodGPUDispatch& dispatch [[buffer(0)]],
    device const MRRodGPURigidBinding* bindings [[buffer(1)]],
    device const MRBodyStateGPU* inputBodies [[buffer(2)]],
    device MRRodGPUAttachmentReaction* reactions [[buffer(3)]],
    device MRRodGPUStatus* statuses [[buffer(4)]],
    device MRBodyStateGPU* outputBodies [[buffer(5)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint laneCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint bodyBase =
        environment * dispatch.stateBodyStride;
    for (uint bodyIndex = lane;
         bodyIndex < dispatch.rigidBodyCount;
         bodyIndex += laneCount) {
        outputBodies[bodyBase + bodyIndex] =
            inputBodies[bodyBase + bodyIndex];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane != 0u ||
        statuses[environment].code != MR_ROD_GPU_SUCCESS) {
        return;
    }

    bool valid = true;
    const uint attachmentBase =
        environment * dispatch.attachmentCount;
    for (uint attachmentIndex = 0u;
         attachmentIndex < dispatch.attachmentCount;
         ++attachmentIndex) {
        const MRRodGPURigidBinding binding =
            bindings[attachmentIndex];
        if (binding.bodyIndex == MR_ROD_GPU_INVALID_BODY) {
            continue;
        }
        device MRRodGPUAttachmentReaction& reaction =
            reactions[attachmentBase + attachmentIndex];
        device MRBodyStateGPU& body =
            outputBodies[bodyBase + binding.bodyIndex];
        const float3 impulse = reaction.impulseAndError.xyz;
        const float3 worldOffset = rotateQuaternion(
            body.orientation,
            binding.localAnchor.xyz
        );
        body.linearVelocityAndInverseMass.xyz +=
            body.linearVelocityAndInverseMass.w * impulse;
        body.angularVelocity.xyz += applyInverseInertia(
            body,
            cross(worldOffset, impulse)
        );
        reaction.bodyIndex = binding.bodyIndex;
        valid =
            valid &&
            finite3(body.linearVelocityAndInverseMass.xyz) &&
            finite3(body.angularVelocity.xyz);
    }
    if (!valid) {
        for (uint bodyIndex = 0u;
             bodyIndex < dispatch.rigidBodyCount;
             ++bodyIndex) {
            outputBodies[bodyBase + bodyIndex] =
                inputBodies[bodyBase + bodyIndex];
        }
        statuses[environment].code =
            MR_ROD_GPU_NONFINITE_RESULT;
    }
}

// Common three-row cone response for procedural rod capsules against rigid
// tools. One threadgroup owns an environment; lane zero performs stable
// pair/feature ordered block GS while the remaining lanes provide the
// transactional state copy. Physical writes are therefore atomic-free.
kernel void mr_solve_rod_tool_contacts(
    device const MRRodGPUDispatch& dispatch [[buffer(0)]],
    device const float* inverseMasses [[buffer(1)]],
    device const float* inverseRotationalInertias [[buffer(2)]],
    device const MRRodColliderGPU* rodColliders [[buffer(3)]],
    device const MRRodToolPairGPU* toolPairs [[buffer(4)]],
    device const MRShapeGPU* toolShapes [[buffer(5)]],
    device const MRMaterialGPU* materials [[buffer(6)]],
    device const uint* pairContactCounts [[buffer(7)]],
    device const MRRodToolWitnessGPU* inputWitnesses [[buffer(8)]],
    device const float4* inputPositions [[buffer(9)]],
    device const float4* inputVelocities [[buffer(10)]],
    device const float* inputTwists [[buffer(11)]],
    device const float* inputTwistRates [[buffer(12)]],
    device const MRBodyStateGPU* inputBodies [[buffer(13)]],
    device MRRodGPUStatus* statuses [[buffer(14)]],
    device MRRodToolWitnessGPU* outputWitnesses [[buffer(15)]],
    device float4* outputPositions [[buffer(16)]],
    device float4* outputVelocities [[buffer(17)]],
    device float* outputTwists [[buffer(18)]],
    device float* outputTwistRates [[buffer(19)]],
    device MRBodyStateGPU* outputBodies [[buffer(20)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint laneCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint nodeBase =
        environment * dispatch.stateNodeStride;
    const uint edgeBase =
        environment * dispatch.stateEdgeStride;
    const uint bodyBase =
        environment * dispatch.stateBodyStride;
    const uint pairBase =
        environment * dispatch.toolPairCount;
    const uint witnessBase =
        pairBase * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
    for (uint node = lane;
         node < dispatch.nodeCount;
         node += laneCount) {
        outputPositions[nodeBase + node] =
            inputPositions[nodeBase + node];
        outputVelocities[nodeBase + node] =
            inputVelocities[nodeBase + node];
    }
    for (uint edge = lane;
         edge < dispatch.edgeCount;
         edge += laneCount) {
        outputTwists[edgeBase + edge] =
            inputTwists[edgeBase + edge];
        outputTwistRates[edgeBase + edge] =
            inputTwistRates[edgeBase + edge];
    }
    for (uint body = lane;
         body < dispatch.rigidBodyCount;
         body += laneCount) {
        outputBodies[bodyBase + body] =
            inputBodies[bodyBase + body];
    }
    for (uint localWitness = lane;
         localWitness <
             dispatch.toolPairCount *
                 MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
         localWitness += laneCount) {
        outputWitnesses[witnessBase + localWitness] =
            inputWitnesses[witnessBase + localWitness];
    }
    threadgroup_barrier(
        mem_flags::mem_device |
        mem_flags::mem_threadgroup
    );
    if (lane != 0u ||
        statuses[environment].code != MR_ROD_GPU_SUCCESS) {
        return;
    }
    if ((dispatch.flags &
         MR_ROD_GPU_FLAG_TOOL_COLLISION) == 0u ||
        dispatch.toolPairCount == 0u) {
        return;
    }
    if (dispatch.toolContactIterations == 0u ||
        dispatch.toolContactStride <
            dispatch.toolPairCount *
                MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR ||
        !(dispatch.gravityAndTimestep.w > 0.0f) ||
        any(dispatch.toolContact < 0.0f) ||
        any(dispatch.toolResponse < 0.0f) ||
        dispatch.toolResponse.x > 1.0f ||
        !(dispatch.toolResponse.z > 0.0f) ||
        !(dispatch.toolResponse.w > 0.0f)) {
        statuses[environment].code =
            MR_ROD_GPU_INVALID_DISPATCH;
        return;
    }
    for (uint pairIndex = 0u;
         pairIndex < dispatch.toolPairCount;
         ++pairIndex) {
        const uint count =
            pairContactCounts[pairBase + pairIndex];
        if ((count & 0x80000000u) != 0u) {
            statuses[environment].code =
                count & 0x7fffffffu;
            statuses[environment].failingIndex =
                pairIndex;
            return;
        }
    }

    const float timestep = dispatch.gravityAndTimestep.w;
    const float normalRegularization =
        dispatch.toolContact.z / (timestep * timestep) +
        dispatch.toolContact.w / timestep;
    uint contactCount = 0u;
    float maximumPenetration = 0.0f;
    bool valid = true;

    // Persistent impulses are physical velocity warm starts. Apply them once
    // before the first nonlinear sweep, then solve only accepted deltas.
    if ((dispatch.flags &
         MR_ROD_GPU_FLAG_TOOL_WARM_START) != 0u) {
        for (uint pairIndex = 0u;
             pairIndex < dispatch.toolPairCount;
             ++pairIndex) {
            const MRRodToolPairGPU pair =
                toolPairs[pairIndex];
            const MRRodColliderGPU rod =
                rodColliders[pair.rodCollider];
            const uint count = min(
                pairContactCounts[pairBase + pairIndex],
                uint(MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR)
            );
            for (uint slot = 0u; slot < count; ++slot) {
                const uint witnessIndex =
                    witnessBase +
                    pairIndex *
                        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR +
                    slot;
                const MRRodToolWitnessGPU witness =
                    outputWitnesses[witnessIndex];
                if ((witness.featuresAndFlags.w &
                     MR_ROD_TOOL_WITNESS_VALID) == 0u) {
                    continue;
                }
                const float3 normal =
                    witness.normalAndPreSolveVelocity.xyz;
                const float3 tangentU =
                    witness.tangentUAndTwistJacobian.xyz;
                const float3 tangentV =
                    cross(normal, tangentU);
                const float3 impulse =
                    normal * witness.impulses.x +
                    tangentU * witness.impulses.y +
                    tangentV * witness.impulses.z;
                device MRBodyStateGPU& body =
                    outputBodies[
                        bodyBase +
                        witness.featuresAndFlags.z
                    ];
                valid = valid && applyRodToolImpulse(
                    dispatch,
                    rod,
                    inverseMasses,
                    inverseRotationalInertias,
                    outputPositions + nodeBase,
                    outputVelocities + nodeBase,
                    outputTwists + edgeBase,
                    outputTwistRates + edgeBase,
                    body,
                    witness,
                    impulse,
                    false
                );
            }
        }
    }

    for (uint iteration = 0u;
         valid && iteration < dispatch.toolContactIterations;
         ++iteration) {
        for (uint pairIndex = 0u;
             pairIndex < dispatch.toolPairCount;
             ++pairIndex) {
            const MRRodToolPairGPU pair =
                toolPairs[pairIndex];
            const MRRodColliderGPU rod =
                rodColliders[pair.rodCollider];
            const uint count = min(
                pairContactCounts[pairBase + pairIndex],
                uint(MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR)
            );
            contactCount += iteration == 0u ? count : 0u;
            for (uint slot = 0u; slot < count; ++slot) {
                const uint witnessIndex =
                    witnessBase +
                    pairIndex *
                        MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR +
                    slot;
                device MRRodToolWitnessGPU& witness =
                    outputWitnesses[witnessIndex];
                if ((witness.featuresAndFlags.w &
                     MR_ROD_TOOL_WITNESS_VALID) == 0u ||
                    witness.featuresAndFlags.z >=
                        dispatch.rigidBodyCount) {
                    valid = false;
                    break;
                }
                const float3 normal =
                    witness.normalAndPreSolveVelocity.xyz;
                const float3 tangentU =
                    witness.tangentUAndTwistJacobian.xyz;
                const float3 tangentV =
                    cross(normal, tangentU);
                const float3 directions[3] = {
                    normal,
                    tangentU,
                    tangentV,
                };
                const float weightB =
                    witness.rodPointAndWeight.w;
                const float weightA = 1.0f - weightB;
                device MRBodyStateGPU& body =
                    outputBodies[
                        bodyBase +
                        witness.featuresAndFlags.z
                    ];
                const float3 bodyLever =
                    witness.toolPointAndSeparation.xyz -
                    body.position.xyz;
                const float3 edgeVector =
                    outputPositions[
                        nodeBase + rod.nodeB
                    ].xyz -
                    outputPositions[
                        nodeBase + rod.nodeA
                    ].xyz;
                const float edgeSquared =
                    dot(edgeVector, edgeVector);
                if (!(edgeSquared > 1.0e-20f)) {
                    valid = false;
                    break;
                }
                const float3 edgeAxis =
                    edgeVector * rsqrt(edgeSquared);
                const float3 surfaceJacobian = cross(
                    edgeAxis,
                    rod.radiusAndOffsets.x *
                        witness
                            .radialAndTwistJacobianV.xyz
                );
                const float twistJacobians[3] = {
                    dot(normal, surfaceJacobian),
                    dot(tangentU, surfaceJacobian),
                    dot(tangentV, surfaceJacobian),
                };
                float response[3][3];
                for (uint row = 0u; row < 3u; ++row) {
                    for (uint column = 0u;
                         column < 3u;
                         ++column) {
                        response[row][column] =
                            rodToolDirectionalResponse(
                                directions[row],
                                directions[column],
                                twistJacobians[row],
                                twistJacobians[column],
                                weightA,
                                weightB,
                                inverseMasses[rod.nodeA],
                                inverseMasses[rod.nodeB],
                                inverseRotationalInertias[
                                    rod.edgeIndex
                                ],
                                body,
                                bodyLever
                            );
                    }
                }
                response[0][0] += normalRegularization;
                float inverseResponse[3][3];
                if (!rodToolInvertSymmetric3(
                        response,
                        inverseResponse
                    )) {
                    valid = false;
                    break;
                }

                const float3 centerVelocity =
                    weightA *
                        outputVelocities[
                            nodeBase + rod.nodeA
                        ].xyz +
                    weightB *
                        outputVelocities[
                            nodeBase + rod.nodeB
                        ].xyz;
                const float3 rodVelocity =
                    centerVelocity +
                    outputTwistRates[
                        edgeBase + rod.edgeIndex
                    ] * surfaceJacobian;
                const float3 toolVelocity =
                    body.flagsAndIndices[0] ==
                            MR_MOTION_STATIC
                    ? float3(0.0f)
                    : body.linearVelocityAndInverseMass.xyz +
                        cross(
                            body.angularVelocity.xyz,
                            bodyLever
                        );
                const float3 relative =
                    toolVelocity - rodVelocity;
                const float separation =
                    witness.toolPointAndSeparation.w;
                maximumPenetration = max(
                    maximumPenetration,
                    max(-separation, 0.0f)
                );
                const float positionalTarget = min(
                    max(
                        -0.2f * separation / timestep,
                        0.0f
                    ),
                    dispatch.toolResponse.w
                );
                float restitutionTarget = 0.0f;
                if ((witness.featuresAndFlags.w &
                     MR_ROD_TOOL_WITNESS_NEW_IMPACT) != 0u &&
                    witness.normalAndPreSolveVelocity.w <
                        -dispatch.toolResponse.y) {
                    restitutionTarget =
                        -dispatch.toolResponse.x *
                        witness.normalAndPreSolveVelocity.w;
                }
                const float3 previous =
                    witness.impulses.xyz;
                const float3 rightHandSide = float3(
                    max(positionalTarget, restitutionTarget) -
                        dot(normal, relative) -
                        normalRegularization * previous.x,
                    -dot(tangentU, relative),
                    -dot(tangentV, relative)
                );
                float3 delta = float3(
                    dot(
                        float3(
                            inverseResponse[0][0],
                            inverseResponse[0][1],
                            inverseResponse[0][2]
                        ),
                        rightHandSide
                    ),
                    dot(
                        float3(
                            inverseResponse[1][0],
                            inverseResponse[1][1],
                            inverseResponse[1][2]
                        ),
                        rightHandSide
                    ),
                    dot(
                        float3(
                            inverseResponse[2][0],
                            inverseResponse[2][1],
                            inverseResponse[2][2]
                        ),
                        rightHandSide
                    )
                );
                float3 candidate = previous + delta;
                candidate.x = max(candidate.x, 0.0f);
                const uint toolMaterialIndex =
                    witness.materialAndGeneration.y;
                const MRMaterialGPU rodMaterial =
                    materials[dispatch.rodMaterialIndex];
                const MRMaterialGPU toolMaterial =
                    materials[toolMaterialIndex];
                const float staticFriction =
                    dispatch.toolResponse.z *
                    sqrt(max(
                        rodMaterial.friction.x *
                            toolMaterial.friction.x,
                        0.0f
                    ));
                const float dynamicFriction =
                    min(
                        staticFriction,
                        dispatch.toolResponse.z *
                            sqrt(max(
                                rodMaterial.friction.y *
                                    toolMaterial.friction.y,
                                0.0f
                            ))
                    );
                candidate.yz = rodToolProjectFriction(
                    candidate.yz,
                    candidate.x,
                    staticFriction,
                    dynamicFriction
                );
                delta = candidate - previous;
                if (!finite3(delta) ||
                    !applyRodToolImpulse(
                        dispatch,
                        rod,
                        inverseMasses,
                        inverseRotationalInertias,
                        outputPositions + nodeBase,
                        outputVelocities + nodeBase,
                        outputTwists + edgeBase,
                        outputTwistRates + edgeBase,
                        body,
                        witness,
                        normal * delta.x +
                            tangentU * delta.y +
                            tangentV * delta.z,
                        true
                    )) {
                    valid = false;
                    break;
                }
                witness.impulses.xyz = candidate;
                witness.featuresAndFlags.w |=
                    MR_ROD_TOOL_WITNESS_WARM_STARTED;
                witness.featuresAndFlags.w &=
                    ~uint(MR_ROD_TOOL_WITNESS_NEW_IMPACT);
            }
            if (!valid) {
                break;
            }
        }
    }

    if (!valid) {
        statuses[environment].code =
            MR_ROD_GPU_NONFINITE_RESULT;
        statuses[environment].failingIndex = 0u;
        for (uint node = 0u;
             node < dispatch.nodeCount;
             ++node) {
            outputPositions[nodeBase + node] =
                inputPositions[nodeBase + node];
            outputVelocities[nodeBase + node] =
                inputVelocities[nodeBase + node];
        }
        for (uint edge = 0u;
             edge < dispatch.edgeCount;
             ++edge) {
            outputTwists[edgeBase + edge] =
                inputTwists[edgeBase + edge];
            outputTwistRates[edgeBase + edge] =
                inputTwistRates[edgeBase + edge];
        }
        for (uint bodyIndex = 0u;
             bodyIndex < dispatch.rigidBodyCount;
             ++bodyIndex) {
            outputBodies[bodyBase + bodyIndex] =
                inputBodies[bodyBase + bodyIndex];
        }
        for (uint localWitness = 0u;
             localWitness <
                 dispatch.toolPairCount *
                     MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
             ++localWitness) {
            outputWitnesses[witnessBase + localWitness] =
                inputWitnesses[witnessBase + localWitness];
        }
        return;
    }
    statuses[environment].diagnostics.z = max(
        statuses[environment].diagnostics.z,
        maximumPenetration
    );
    statuses[environment].diagnostics.w +=
        float(contactCount);
}
