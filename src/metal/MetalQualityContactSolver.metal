#include <metal_stdlib>

#include "metalrobo/quality_solver_shared.h"

using namespace metal;

namespace {

constant uint kWidth = 32u;
constant uint kMaximumDimension =
    MR_METAL_QUALITY_MAX_DIMENSION;
constant uint kMaximumContacts =
    MR_METAL_QUALITY_MAX_CONTACTS;

inline void matrixVector(
    device const float* matrix,
    threadgroup const float* input,
    threadgroup float* output,
    const uint dimension,
    const uint lane
) {
    for (uint row = lane;
         row < dimension;
         row += kWidth) {
        float value = 0.0f;
        for (uint column = 0u;
             column < dimension;
             ++column) {
            value +=
                matrix[row * dimension + column] *
                input[column];
        }
        output[row] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float groupDot(
    threadgroup const float* left,
    threadgroup const float* right,
    const uint dimension,
    const uint lane
) {
    float partial = 0.0f;
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        partial += left[index] * right[index];
    }
    return simd_sum(partial);
}

inline void projectLorentz(
    threadgroup const float* input,
    threadgroup float* output,
    threadgroup float* derivative,
    const uint contactCount,
    const uint lane
) {
    if (lane < contactCount) {
        const uint offset = 3u * lane;
        const float scalar = input[offset];
        const float first = input[offset + 1u];
        const float second = input[offset + 2u];
        const float radial = length(float2(first, second));
        const uint derivativeBase = 9u * lane;
        for (uint entry = 0u; entry < 9u; ++entry) {
            derivative[derivativeBase + entry] = 0.0f;
        }
        if (scalar > radial) {
            output[offset] = scalar;
            output[offset + 1u] = first;
            output[offset + 2u] = second;
            derivative[derivativeBase + 0u] = 1.0f;
            derivative[derivativeBase + 4u] = 1.0f;
            derivative[derivativeBase + 8u] = 1.0f;
        } else if (scalar < -radial) {
            output[offset] = 0.0f;
            output[offset + 1u] = 0.0f;
            output[offset + 2u] = 0.0f;
        } else if (radial <= 1.0e-20f) {
            output[offset] = 0.5f * max(scalar, 0.0f);
            output[offset + 1u] = 0.0f;
            output[offset + 2u] = 0.0f;
            derivative[derivativeBase + 0u] = 0.5f;
            derivative[derivativeBase + 4u] = 0.5f;
            derivative[derivativeBase + 8u] = 0.5f;
        } else {
            const float2 tangent = float2(first, second);
            const float2 unit = tangent / radial;
            const float projectedScalar =
                0.5f * (scalar + radial);
            const float tangentScale =
                projectedScalar / radial;
            const float2 projectedTangent =
                tangentScale * tangent;
            output[offset] = projectedScalar;
            output[offset + 1u] = projectedTangent.x;
            output[offset + 2u] = projectedTangent.y;

            derivative[derivativeBase + 0u] = 0.5f;
            derivative[derivativeBase + 1u] =
                0.5f * unit.x;
            derivative[derivativeBase + 2u] =
                0.5f * unit.y;
            derivative[derivativeBase + 3u] =
                0.5f * unit.x;
            derivative[derivativeBase + 6u] =
                0.5f * unit.y;
            const float ratio = scalar / radial;
            const float diagonal = 0.5f * (1.0f + ratio);
            derivative[derivativeBase + 4u] =
                diagonal - 0.5f * ratio * unit.x * unit.x;
            derivative[derivativeBase + 5u] =
                -0.5f * ratio * unit.x * unit.y;
            derivative[derivativeBase + 7u] =
                derivative[derivativeBase + 5u];
            derivative[derivativeBase + 8u] =
                diagonal - 0.5f * ratio * unit.y * unit.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void applyDerivative(
    threadgroup const float* derivative,
    threadgroup const float* input,
    threadgroup float* output,
    const uint contactCount,
    const uint lane
) {
    if (lane < contactCount) {
        const uint offset = 3u * lane;
        const uint base = 9u * lane;
        for (uint row = 0u; row < 3u; ++row) {
            output[offset + row] =
                derivative[base + 3u * row + 0u] *
                    input[offset + 0u] +
                derivative[base + 3u * row + 1u] *
                    input[offset + 1u] +
                derivative[base + 3u * row + 2u] *
                    input[offset + 2u];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void evaluateNaturalResidual(
    device const float* matrix,
    device const float* linear,
    threadgroup const float* value,
    threadgroup float* matrixAction,
    threadgroup float* projectionInput,
    threadgroup float* projection,
    threadgroup float* derivative,
    threadgroup float* residual,
    threadgroup float* scalars,
    const uint contactCount,
    const uint dimension,
    const float gamma,
    const uint lane
) {
    matrixVector(
        matrix,
        value,
        matrixAction,
        dimension,
        lane
    );
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        matrixAction[index] += linear[index];
        projectionInput[index] =
            value[index] - gamma * matrixAction[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    projectLorentz(
        projectionInput,
        projection,
        derivative,
        contactCount,
        lane
    );
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        residual[index] = value[index] - projection[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float residualSquared =
        groupDot(residual, residual, dimension, lane);
    const float valueSquared =
        groupDot(value, value, dimension, lane);
    const float valueMatrixValue =
        groupDot(value, matrixAction, dimension, lane);
    if (lane == 0u) {
        // matrixAction is Q*x+c, hence x'(Q*x+c) - 0.5*x'Q*x
        // is recovered below after a second Q*x dot in the caller when
        // needed. Merit and residual are the globalization invariants.
        scalars[0] = 0.5f * residualSquared;
        scalars[1] =
            sqrt(max(residualSquared, 0.0f)) /
            (1.0f + sqrt(max(valueSquared, 0.0f)));
        scalars[2] = valueMatrixValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void applyJacobian(
    device const float* matrix,
    threadgroup const float* derivative,
    threadgroup const float* input,
    threadgroup float* output,
    threadgroup float* matrixAction,
    threadgroup float* transformed,
    threadgroup float* derivativeAction,
    const uint contactCount,
    const uint dimension,
    const float gamma,
    const uint lane
) {
    matrixVector(
        matrix,
        input,
        matrixAction,
        dimension,
        lane
    );
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        transformed[index] =
            input[index] - gamma * matrixAction[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    applyDerivative(
        derivative,
        transformed,
        derivativeAction,
        contactCount,
        lane
    );
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        output[index] =
            input[index] - derivativeAction[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void applyJacobianTranspose(
    device const float* matrix,
    threadgroup const float* derivative,
    threadgroup const float* input,
    threadgroup float* output,
    threadgroup float* derivativeAction,
    threadgroup float* matrixAction,
    const uint contactCount,
    const uint dimension,
    const float gamma,
    const uint lane
) {
    applyDerivative(
        derivative,
        input,
        derivativeAction,
        contactCount,
        lane
    );
    matrixVector(
        matrix,
        derivativeAction,
        matrixAction,
        dimension,
        lane
    );
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        output[index] =
            input[index] -
            derivativeAction[index] +
            gamma * matrixAction[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float maximumConeViolation(
    threadgroup const float* value,
    const uint contactCount,
    const uint lane
) {
    float violation = 0.0f;
    if (lane < contactCount) {
        const uint offset = 3u * lane;
        violation = max(
            max(
                length(float2(
                    value[offset + 1u],
                    value[offset + 2u]
                )) - value[offset],
                -value[offset]
            ),
            0.0f
        );
    }
    return simd_max(violation);
}

} // namespace

kernel void mr_quality_contact_solve(
    device const MRMetalQualityDispatchGPU& dispatch [[buffer(0)]],
    device const float* matrices [[buffer(1)]],
    device const float* linearVectors [[buffer(2)]],
    device const float* warmStarts [[buffer(3)]],
    device float* outputImpulses [[buffer(4)]],
    device MRMetalQualityStatusGPU* statuses [[buffer(5)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]]
) {
    if (problem >= dispatch.problemCount) {
        return;
    }

    threadgroup float value[kMaximumDimension];
    threadgroup float oldValue[kMaximumDimension];
    threadgroup float candidate[kMaximumDimension];
    threadgroup float residual[kMaximumDimension];
    threadgroup float oldResidual[kMaximumDimension];
    threadgroup float derivative[kMaximumContacts * 9u];
    threadgroup float matrixAction[kMaximumDimension];
    threadgroup float projectionInput[kMaximumDimension];
    threadgroup float projection[kMaximumDimension];
    threadgroup float gradient[kMaximumDimension];
    threadgroup float direction[kMaximumDimension];
    threadgroup float rightHandSide[kMaximumDimension];
    threadgroup float cgResidual[kMaximumDimension];
    threadgroup float cgDirection[kMaximumDimension];
    threadgroup float cgAction[kMaximumDimension];
    threadgroup float workA[kMaximumDimension];
    threadgroup float workB[kMaximumDimension];
    threadgroup float workC[kMaximumDimension];
    threadgroup float scalars[8u];
    threadgroup atomic_uint invalidInput;
    threadgroup uint linearSolveFailed;
    threadgroup uint converged;
    threadgroup uint totalCGIterations;
    threadgroup uint totalBacktracks;
    threadgroup uint projectedFallbacks;

    MRMetalQualityStatusGPU status{};
    status.code = MR_METAL_QUALITY_SUCCESS;
    status.problemIndex = problem;
    if (lane == 0u) {
        atomic_store_explicit(
            &invalidInput,
            0u,
            memory_order_relaxed
        );
        linearSolveFailed = 0u;
        converged = 0u;
        totalCGIterations = 0u;
        totalBacktracks = 0u;
        projectedFallbacks = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint dimension = dispatch.dimension;
    const uint contactCount = dispatch.contactCount;
    const bool validDispatch =
        dispatch.abiVersion ==
            MR_METAL_QUALITY_SOLVER_ABI_VERSION &&
        dispatch.problemCount > 0u &&
        contactCount > 0u &&
        contactCount <= kMaximumContacts &&
        dimension == 3u * contactCount &&
        dispatch.matrixStride >= dimension * dimension &&
        dispatch.vectorStride >= dimension &&
        dispatch.maximumNewtonIterations > 0u &&
        dispatch.maximumCGIterations > 0u &&
        dispatch.maximumLineSearchIterations > 0u &&
        dispatch.flags == 0u &&
        dispatch.reserved0 == 0u &&
        dispatch.reserved1 == 0u &&
        all(isfinite(dispatch.tolerances)) &&
        dispatch.tolerances.x > 0.0f &&
        dispatch.tolerances.y > 0.0f &&
        dispatch.tolerances.y < 0.5f &&
        dispatch.tolerances.z > 0.0f &&
        dispatch.tolerances.w > 0.0f;
    if (!validDispatch) {
        if (lane == 0u) {
            status.code = MR_METAL_QUALITY_INVALID_DISPATCH;
            statuses[problem] = status;
        }
        return;
    }

    device const float* matrix =
        matrices + problem * dispatch.matrixStride;
    device const float* linear =
        linearVectors + problem * dispatch.vectorStride;
    device const float* warm =
        warmStarts + problem * dispatch.vectorStride;
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        if (!isfinite(linear[index]) ||
            !isfinite(warm[index])) {
            atomic_store_explicit(
                &invalidInput,
                1u,
                memory_order_relaxed
            );
        }
        value[index] = warm[index];
        for (uint column = 0u;
             column < dimension;
             ++column) {
            if (!isfinite(
                    matrix[index * dimension + column]
                )) {
                atomic_store_explicit(
                    &invalidInput,
                    1u,
                    memory_order_relaxed
                );
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(
            &invalidInput,
            memory_order_relaxed
        ) != 0u) {
        if (lane == 0u) {
            status.code = MR_METAL_QUALITY_NONFINITE_INPUT;
            statuses[problem] = status;
        }
        return;
    }

    // Project the warm start once so every iterate remains cone feasible.
    projectLorentz(
        value,
        candidate,
        derivative,
        contactCount,
        lane
    );
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        value[index] = candidate[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximumRowSum = 0.0f;
    for (uint row = lane;
         row < dimension;
         row += kWidth) {
        float rowSum = 0.0f;
        for (uint column = 0u;
             column < dimension;
             ++column) {
            rowSum += abs(
                matrix[row * dimension + column]
            );
        }
        maximumRowSum = max(maximumRowSum, rowSum);
    }
    maximumRowSum = simd_max(maximumRowSum);
    if (!(maximumRowSum > 0.0f) ||
        !isfinite(maximumRowSum)) {
        if (lane == 0u) {
            status.code = MR_METAL_QUALITY_NONFINITE_INPUT;
            statuses[problem] = status;
        }
        return;
    }
    const float gamma = 1.0f / maximumRowSum;

    for (uint newton = 0u;
         newton < dispatch.maximumNewtonIterations;
         ++newton) {
        evaluateNaturalResidual(
            matrix,
            linear,
            value,
            matrixAction,
            projectionInput,
            projection,
            derivative,
            residual,
            scalars,
            contactCount,
            dimension,
            gamma,
            lane
        );
        if (scalars[1] <= dispatch.tolerances.x) {
            if (lane == 0u) {
                converged = 1u;
                status.newtonIterations = newton;
            }
            break;
        }

        applyJacobianTranspose(
            matrix,
            derivative,
            residual,
            gradient,
            workA,
            workB,
            contactCount,
            dimension,
            gamma,
            lane
        );
        for (uint index = lane;
             index < dimension;
             index += kWidth) {
            rightHandSide[index] = -gradient[index];
            direction[index] = 0.0f;
            cgResidual[index] = rightHandSide[index];
            cgDirection[index] = cgResidual[index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float residualSquared = groupDot(
            cgResidual,
            cgResidual,
            dimension,
            lane
        );
        if (!isfinite(residualSquared)) {
            if (lane == 0u) {
                linearSolveFailed = 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint cg = 0u;
             cg < dispatch.maximumCGIterations &&
             linearSolveFailed == 0u;
             ++cg) {
            applyJacobian(
                matrix,
                derivative,
                cgDirection,
                workA,
                workB,
                workC,
                projection,
                contactCount,
                dimension,
                gamma,
                lane
            );
            applyJacobianTranspose(
                matrix,
                derivative,
                workA,
                cgAction,
                workB,
                workC,
                contactCount,
                dimension,
                gamma,
                lane
            );
            for (uint index = lane;
                 index < dimension;
                 index += kWidth) {
                cgAction[index] +=
                    dispatch.tolerances.z *
                    cgDirection[index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float denominator = groupDot(
                cgDirection,
                cgAction,
                dimension,
                lane
            );
            if (!(denominator >
                    dispatch.tolerances.w) ||
                !isfinite(denominator)) {
                if (lane == 0u) {
                    linearSolveFailed = 1u;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                break;
            }
            const float alpha =
                residualSquared / denominator;
            for (uint index = lane;
                 index < dimension;
                 index += kWidth) {
                direction[index] +=
                    alpha * cgDirection[index];
                cgResidual[index] -=
                    alpha * cgAction[index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float nextResidualSquared = groupDot(
                cgResidual,
                cgResidual,
                dimension,
                lane
            );
            if (lane == 0u) {
                ++totalCGIterations;
            }
            if (!isfinite(nextResidualSquared)) {
                if (lane == 0u) {
                    linearSolveFailed = 1u;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                break;
            }
            if (sqrt(max(nextResidualSquared, 0.0f)) <=
                0.25f * dispatch.tolerances.x) {
                residualSquared = nextResidualSquared;
                break;
            }
            const float beta =
                nextResidualSquared /
                max(residualSquared, 1.0e-30f);
            for (uint index = lane;
                 index < dimension;
                 index += kWidth) {
                cgDirection[index] =
                    cgResidual[index] +
                    beta * cgDirection[index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            residualSquared = nextResidualSquared;
        }

        const float descent = groupDot(
            gradient,
            direction,
            dimension,
            lane
        );
        if (linearSolveFailed != 0u ||
            !(descent < 0.0f) ||
            !isfinite(descent)) {
            for (uint index = lane;
                 index < dimension;
                 index += kWidth) {
                direction[index] = -gradient[index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                linearSolveFailed = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = lane;
             index < dimension;
             index += kWidth) {
            oldValue[index] = value[index];
            oldResidual[index] = residual[index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float oldMerit = scalars[0];
        const float slope = groupDot(
            gradient,
            direction,
            dimension,
            lane
        );
        float step = 1.0f;
        bool accepted = false;
        for (uint lineSearch = 0u;
             lineSearch <
                 dispatch.maximumLineSearchIterations;
             ++lineSearch) {
            for (uint index = lane;
                 index < dimension;
                 index += kWidth) {
                candidate[index] =
                    oldValue[index] +
                    step * direction[index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            evaluateNaturalResidual(
                matrix,
                linear,
                candidate,
                matrixAction,
                projectionInput,
                projection,
                derivative,
                residual,
                scalars,
                contactCount,
                dimension,
                gamma,
                lane
            );
            accepted =
                isfinite(scalars[0]) &&
                scalars[0] <=
                    oldMerit +
                    dispatch.tolerances.y *
                    step * slope;
            if (accepted) {
                for (uint index = lane;
                     index < dimension;
                     index += kWidth) {
                    value[index] = candidate[index];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                break;
            }
            step *= 0.5f;
            if (lane == 0u) {
                ++totalBacktracks;
            }
        }
        if (!accepted) {
            // Globally convergent projected-gradient fallback:
            // P(x-gamma grad) = x-F(x).
            for (uint index = lane;
                 index < dimension;
                 index += kWidth) {
                value[index] =
                    oldValue[index] - oldResidual[index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                ++projectedFallbacks;
            }
        }
        if (lane == 0u) {
            status.newtonIterations = newton + 1u;
        }
    }

    evaluateNaturalResidual(
        matrix,
        linear,
        value,
        matrixAction,
        projectionInput,
        projection,
        derivative,
        residual,
        scalars,
        contactCount,
        dimension,
        gamma,
        lane
    );
    const float coneViolation =
        maximumConeViolation(value, contactCount, lane);
    if (scalars[1] <= dispatch.tolerances.x) {
        if (lane == 0u) {
            converged = 1u;
        }
    }
    for (uint index = lane;
         index < dimension;
         index += kWidth) {
        if (!isfinite(value[index])) {
            atomic_store_explicit(
                &invalidInput,
                1u,
                memory_order_relaxed
            );
        }
        outputImpulses[
            problem * dispatch.vectorStride + index
        ] = value[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        status.cgIterations = totalCGIterations;
        status.lineSearchBacktracks = totalBacktracks;
        status.projectedGradientFallbacks =
            projectedFallbacks;
        status.code =
            atomic_load_explicit(
                &invalidInput,
                memory_order_relaxed
            ) != 0u
            ? MR_METAL_QUALITY_NONFINITE_RESULT
            : (converged != 0u
                ? MR_METAL_QUALITY_SUCCESS
                : MR_METAL_QUALITY_DID_NOT_CONVERGE);
        status.diagnostics = float4(
            scalars[1],
            scalars[0],
            coneViolation,
            0.0f
        );
        statuses[problem] = status;
    }
}
