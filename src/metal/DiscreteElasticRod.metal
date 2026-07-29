#include <metal_stdlib>

#include "metalrobo/rod_gpu_shared.h"

using namespace metal;

namespace {

constant float kGeometryFloor = 1.0e-12f;
constant float kFrameDenominatorFloor = 1.0e-6f;
constant float kFloatEpsilon = 1.1920928955078125e-7f;

inline bool finite3(const float3 value) {
    return all(isfinite(value));
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

inline float3 leastAlignedDirector(const float3 tangent) {
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
    return normalize(
        axis - tangent * dot(axis, tangent)
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
    thread float2& curvature
) {
    float3 left;
    float3 right;
    if (!normalizeChecked(positions[1] - positions[0], left) ||
        !normalizeChecked(positions[2] - positions[1], right)) {
        return false;
    }
    float3 referenceLeft = leastAlignedDirector(left);
    float3 referenceRight;
    if (!transport(
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

inline void projectStretch(
    const uint edge,
    const float timestep,
    device const float* restLengths,
    device const float* inverseMasses,
    device const float* stretchStiffness,
    threadgroup float3* positions,
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
        -constraint / (inverseA + inverseB + alpha);
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
        -constraint / (inverseA + inverseB + alpha);
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
    const float timestep,
    device const float* inverseMasses,
    threadgroup float3* positions,
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
    const float3 correction =
        -delta * inverseMass / (inverseMass + alpha);
    positions[node] += correction;
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

inline void projectBend(
    const uint constraintIndex,
    const float timestep,
    const float derivativeScale,
    device const float4* restCurvature,
    device const float* restLengths,
    device const float* inverseMasses,
    device const float* inverseRotationalInertias,
    device const float* bendStiffness,
    threadgroup float3* positions,
    threadgroup float* twists,
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
                plus
            );
            localPositions[node][axis] -= 2.0f * step;
            float2 minus;
            const bool minusOk = localCurvature(
                localPositions,
                localTwists,
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
    for (uint edge = 0u; edge < 2u; ++edge) {
        const float step = derivativeScale;
        localTwists[edge] += step;
        float2 plus;
        const bool plusOk = localCurvature(
            localPositions,
            localTwists,
            plus
        );
        localTwists[edge] -= 2.0f * step;
        float2 minus;
        const bool minusOk = localCurvature(
            localPositions,
            localTwists,
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
    const float2 lambda = float2(
        (
            -effective11 * constraint.x +
            effective01 * constraint.y
        ) / determinant,
        (
            effective01 * constraint.x -
            effective00 * constraint.y
        ) / determinant
    );
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
    for (uint edge = 0u; edge < 2u; ++edge) {
        twists[constraintIndex + edge] +=
            inverseRotationalInertias[
                constraintIndex + edge
            ] * (
                lambda.x * twistGradient0[edge] +
                lambda.y * twistGradient1[edge]
            );
    }
    recordPositiveMaximum(
        maximumErrorBits,
        max(abs(constraint.x), abs(constraint.y))
    );
}

} // namespace

// One threadgroup owns one environment and retains the complete rod state in
// threadgroup memory. Constraint coloring makes every phase write-disjoint:
// two colors for edges/twist, three for bend triples. No floating atomics
// participate in the physical update; atomics only reduce diagnostics.
kernel void mr_discrete_elastic_rod_step(
    device const MRRodGPUDispatch& dispatch [[buffer(0)]],
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
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint laneCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    threadgroup float3 positions[MR_ROD_GPU_MAX_NODES];
    threadgroup float3 originalPositions[MR_ROD_GPU_MAX_NODES];
    threadgroup float3 iterationPositions[MR_ROD_GPU_MAX_NODES];
    threadgroup float3 velocities[MR_ROD_GPU_MAX_NODES];
    threadgroup float twists[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float originalTwists[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float iterationTwists[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup float twistRates[MR_ROD_GPU_MAX_NODES - 1u];
    threadgroup atomic_uint failure;
    threadgroup atomic_uint maximumErrorBits;
    threadgroup atomic_uint maximumCorrectionBits;
    threadgroup uint completedIterations;
    threadgroup uint converged;

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
            !(dispatch.gravityAndTimestep.w > 0.0f) ||
            !all(isfinite(dispatch.gravityAndTimestep)) ||
            any(dispatch.dampingDerivativeTolerance < 0.0f) ||
            !(dispatch.dampingDerivativeTolerance.z > 0.0f) ||
            !(dispatch.dampingDerivativeTolerance.w > 0.0f) ||
            !all(isfinite(
                dispatch.dampingDerivativeTolerance
            ))) {
            atomic_store_explicit(
                &failure,
                uint(MR_ROD_GPU_INVALID_DISPATCH),
                memory_order_relaxed
            );
        }
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
                        inverseMasses,
                        inverseRotationalInertias,
                        bendStiffness,
                        positions,
                        twists,
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
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                for (uint attachmentIndex = 0u;
                     attachmentIndex < dispatch.attachmentCount;
                     ++attachmentIndex) {
                    projectAttachment(
                        attachments[
                            attachmentBase + attachmentIndex
                        ],
                        dispatch.gravityAndTimestep.w,
                        inverseMasses,
                        positions,
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
                        inverseMasses,
                        inverseRotationalInertias,
                        bendStiffness,
                        positions,
                        twists,
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
                        failure,
                        maximumErrorBits,
                        maximumCorrectionBits
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint attachmentIndex = lane;
                 attachmentIndex < dispatch.attachmentCount;
                 attachmentIndex += laneCount) {
                projectAttachment(
                    attachments[
                        attachmentBase + attachmentIndex
                    ],
                    dispatch.gravityAndTimestep.w,
                    inverseMasses,
                    positions,
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
        for (uint node = lane;
             node < dispatch.nodeCount;
             node += laneCount) {
            recordPositiveMaximum(
                maximumCorrectionBits,
                length(
                    positions[node] -
                    iterationPositions[node]
                )
            );
        }
        for (uint edge = lane;
             edge < dispatch.edgeCount;
             edge += laneCount) {
            recordPositiveMaximum(
                maximumCorrectionBits,
                abs(twists[edge] - iterationTwists[edge])
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

    const uint failureCode = atomic_load_explicit(
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
        float3 position = failureCode == MR_ROD_GPU_SUCCESS
            ? positions[node]
            : originalPositions[node];
        float3 velocity = failureCode == MR_ROD_GPU_SUCCESS
            ? (
                position - originalPositions[node]
            ) * (linearDecay * inverseTimestep)
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
    if (lane == 0u) {
        MRRodGPUStatus status{};
        status.code = failureCode;
        status.environment = environment;
        status.iterations = completedIterations;
        status.failingIndex = 0xffffffffu;
        status.diagnostics = float4(
            as_type<float>(atomic_load_explicit(
                &maximumErrorBits,
                memory_order_relaxed
            )),
            as_type<float>(atomic_load_explicit(
                &maximumCorrectionBits,
                memory_order_relaxed
            )),
            0.0f,
            0.0f
        );
        statuses[environment] = status;
    }
}
