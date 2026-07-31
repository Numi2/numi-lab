#include <metal_stdlib>

#include "metalrobo/unified_quality_shared.h"

using namespace metal;

namespace {

constant uint kWidth = 32u;
constant uint kMaxV =
    MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES;
constant uint kMaxRows = MR_UNIFIED_QUALITY_MAX_ROWS;
constant uint kMaxBlocks = MR_UNIFIED_QUALITY_MAX_BLOCKS;
constant uint kMaxBlockDimension =
    MR_UNIFIED_QUALITY_MAX_BLOCK_DIMENSION;
constant float kFloatEpsilon =
    1.1920928955078125e-7f;

inline float blockFloat(
    const float4 first,
    const float4 second,
    const uint index
) {
    return index < 4u
        ? first[index]
        : second[index - 4u];
}

inline bool finiteBlock(
    const MRUnifiedQualityBlockGPU block
) {
    return
        all(isfinite(block.scale0)) &&
        all(isfinite(block.scale1)) &&
        all(isfinite(block.regularization0)) &&
        all(isfinite(block.regularization1)) &&
        all(isfinite(block.boundsAndShift));
}

inline float groupDot(
    threadgroup const float* left,
    threadgroup const float* right,
    const uint count,
    const uint lane
) {
    float value = 0.0f;
    for (uint index = lane; index < count; index += kWidth) {
        value = fma(left[index], right[index], value);
    }
    return simd_sum(value);
}

inline float groupMaximumAbsolute(
    threadgroup const float* values,
    const uint count,
    const uint lane
) {
    float value = 0.0f;
    for (uint index = lane; index < count; index += kWidth) {
        value = max(value, abs(values[index]));
    }
    return simd_max(value);
}

inline bool invertSmallMatrix(
    thread float matrix[7][7],
    thread float inverse[7][7],
    const uint dimension,
    const float pivotFloor
) {
    for (uint row = 0u; row < 7u; ++row) {
        for (uint column = 0u; column < 7u; ++column) {
            inverse[row][column] =
                row == column ? 1.0f : 0.0f;
        }
    }
    for (uint column = 0u;
         column < dimension;
         ++column) {
        uint pivot = column;
        float maximum = abs(matrix[column][column]);
        for (uint row = column + 1u;
             row < dimension;
             ++row) {
            const float candidate =
                abs(matrix[row][column]);
            if (candidate > maximum) {
                maximum = candidate;
                pivot = row;
            }
        }
        if (!(maximum > pivotFloor) ||
            !isfinite(maximum)) {
            return false;
        }
        if (pivot != column) {
            for (uint entry = 0u;
                 entry < dimension;
                 ++entry) {
                const float matrixSwap =
                    matrix[column][entry];
                matrix[column][entry] =
                    matrix[pivot][entry];
                matrix[pivot][entry] = matrixSwap;
                const float inverseSwap =
                    inverse[column][entry];
                inverse[column][entry] =
                    inverse[pivot][entry];
                inverse[pivot][entry] = inverseSwap;
            }
        }
        const float reciprocal =
            1.0f / matrix[column][column];
        for (uint entry = 0u;
             entry < dimension;
             ++entry) {
            matrix[column][entry] *= reciprocal;
            inverse[column][entry] *= reciprocal;
        }
        for (uint row = 0u;
             row < dimension;
             ++row) {
            if (row == column) {
                continue;
            }
            const float scale = matrix[row][column];
            for (uint entry = 0u;
                 entry < dimension;
                 ++entry) {
                matrix[row][entry] = fma(
                    -scale,
                    matrix[column][entry],
                    matrix[row][entry]
                );
                inverse[row][entry] = fma(
                    -scale,
                    inverse[column][entry],
                    inverse[row][entry]
                );
            }
        }
    }
    return true;
}

inline float coneBoundaryFunction(
    thread const float* diagonal,
    thread const float* transformed,
    const uint dimension,
    const float eta,
    thread float* projected,
    thread float& derivative
) {
    const float normal =
        transformed[0] + eta / diagonal[0];
    if (!(normal > 0.0f) || !isfinite(normal)) {
        derivative = -INFINITY;
        return INFINITY;
    }
    const float alpha = eta / normal;
    float tangentSquared = 0.0f;
    float tangentDerivativeNumerator = 0.0f;
    const float alphaDerivative =
        transformed[0] / (normal * normal);
    projected[0] = normal;
    for (uint index = 1u;
         index < dimension;
         ++index) {
        projected[index] =
            diagonal[index] * transformed[index] /
            (diagonal[index] + alpha);
        tangentSquared = fma(
            projected[index],
            projected[index],
            tangentSquared
        );
        const float projectedDerivative =
            -projected[index] /
            (diagonal[index] + alpha) *
            alphaDerivative;
        tangentDerivativeNumerator = fma(
            projected[index],
            projectedDerivative,
            tangentDerivativeNumerator
        );
    }
    const float tangent =
        sqrt(max(tangentSquared, 0.0f));
    derivative =
        (
            tangent > 1.0e-20f
            ? tangentDerivativeNumerator / tangent
            : 0.0f
        ) - 1.0f / diagonal[0];
    return tangent - normal;
}

inline float ballBoundaryFunction(
    thread const float* diagonal,
    thread const float* transformed,
    const uint dimension,
    const float radius,
    const float alpha,
    thread float* projected,
    thread float& derivative
) {
    float tangentSquared = 0.0f;
    float tangentDerivativeNumerator = 0.0f;
    projected[0] = radius;
    for (uint index = 1u;
         index < dimension;
         ++index) {
        projected[index] =
            diagonal[index] * transformed[index] /
            (diagonal[index] + alpha);
        tangentSquared = fma(
            projected[index],
            projected[index],
            tangentSquared
        );
        const float projectedDerivative =
            -projected[index] /
            (diagonal[index] + alpha);
        tangentDerivativeNumerator = fma(
            projected[index],
            projectedDerivative,
            tangentDerivativeNumerator
        );
    }
    const float tangent =
        sqrt(max(tangentSquared, 0.0f));
    derivative =
        tangent > 1.0e-20f
        ? tangentDerivativeNumerator / tangent
        : -1.0f;
    return tangent - radius;
}

// Exact diagonal-metric projection onto an elliptic cone. The returned
// derivative is W = -d lambda / d w, the PSD block used by
// A + J' W J. Boundary derivatives come from the deterministic KKT inverse,
// rather than finite differences or a pyramidal approximation.
inline bool projectConeBlock(
    const MRUnifiedQualityBlockGPU block,
    thread const float* relativeVelocity,
    thread float* impulse,
    thread float derivative[36],
    thread float& objective,
    thread float& feasibility,
    thread float& complementarity,
    const float pivotFloor
) {
    const uint dimension = block.layout.y;
    thread float scale[6];
    thread float regularization[6];
    thread float diagonal[6];
    thread float transformed[6];
    thread float projected[6];
    for (uint index = 0u;
         index < kMaxBlockDimension;
         ++index) {
        impulse[index] = 0.0f;
        scale[index] = 1.0f;
        regularization[index] = 1.0f;
        diagonal[index] = 1.0f;
        transformed[index] = 0.0f;
        projected[index] = 0.0f;
    }
    for (uint entry = 0u; entry < 36u; ++entry) {
        derivative[entry] = 0.0f;
    }
    const float adhesion = block.boundsAndShift.x;
    const float maximumNormal = block.boundsAndShift.y;
    if (dimension < 3u ||
        dimension > kMaxBlockDimension ||
        !(adhesion >= 0.0f) ||
        !(maximumNormal >= 0.0f)) {
        return false;
    }
    for (uint index = 0u; index < dimension; ++index) {
        scale[index] = blockFloat(
            block.scale0,
            block.scale1,
            index
        );
        regularization[index] = blockFloat(
            block.regularization0,
            block.regularization1,
            index
        );
        if (!(scale[index] > 0.0f) ||
            !(regularization[index] > 0.0f) ||
            !isfinite(relativeVelocity[index])) {
            return false;
        }
        diagonal[index] =
            regularization[index] *
            scale[index] * scale[index];
        transformed[index] =
            -relativeVelocity[index] /
            (regularization[index] * scale[index]);
    }
    if (abs(scale[0] - 1.0f) >
        8.0f * kFloatEpsilon) {
        return false;
    }
    transformed[0] += adhesion;

    float tangentNormSquared = 0.0f;
    float weightedDualSquared = 0.0f;
    for (uint index = 1u; index < dimension; ++index) {
        tangentNormSquared = fma(
            transformed[index],
            transformed[index],
            tangentNormSquared
        );
        const float dual =
            diagonal[index] * transformed[index];
        weightedDualSquared =
            fma(dual, dual, weightedDualSquared);
    }
    const float tangentNorm =
        sqrt(max(tangentNormSquared, 0.0f));
    const float weightedDual =
        sqrt(max(weightedDualSquared, 0.0f));
    enum ProjectionRegion : uint {
        zeroRegion = 0u,
        interiorRegion = 1u,
        boundaryRegion = 2u,
        cappedInteriorRegion = 3u,
        cappedBoundaryRegion = 4u,
    };
    ProjectionRegion region = boundaryRegion;
    float eta = 0.0f;
    if (transformed[0] >= tangentNorm) {
        region = interiorRegion;
        for (uint index = 0u;
             index < dimension;
             ++index) {
            projected[index] = transformed[index];
        }
    } else if (
        diagonal[0] * transformed[0] <=
            -weightedDual
    ) {
        region = zeroRegion;
        for (uint index = 0u;
             index < dimension;
             ++index) {
            projected[index] = 0.0f;
        }
    } else {
        float lower = max(
            0.0f,
            -diagonal[0] * transformed[0]
        );
        lower += max(
            pivotFloor,
            8.0f * kFloatEpsilon *
                max(diagonal[0], 1.0f)
        );
        float upper = max(
            lower + max(diagonal[0], 1.0f),
            1.0f
        );
        thread float work[6];
        float boundaryDerivative = 0.0f;
        float upperValue = coneBoundaryFunction(
            diagonal,
            transformed,
            dimension,
            upper,
            work,
            boundaryDerivative
        );
        for (uint expansion = 0u;
             expansion < 24u &&
                 (!isfinite(upperValue) ||
                  upperValue > 0.0f);
             ++expansion) {
            upper =
                2.0f * upper + max(diagonal[0], 1.0f);
            upperValue = coneBoundaryFunction(
                diagonal,
                transformed,
                dimension,
                upper,
                work,
                boundaryDerivative
            );
        }
        if (!isfinite(upperValue) ||
            upperValue > 0.0f) {
            return false;
        }
        float trial = 0.5f * (lower + upper);
        float bestTrial = trial;
        float bestAbsoluteValue = INFINITY;
        for (uint iteration = 0u;
             iteration < 12u;
             ++iteration) {
            const float value = coneBoundaryFunction(
                diagonal,
                transformed,
                dimension,
                trial,
                work,
                boundaryDerivative
            );
            if (isfinite(value) &&
                abs(value) < bestAbsoluteValue) {
                bestAbsoluteValue = abs(value);
                bestTrial = trial;
            }
            if (!isfinite(value)) {
                lower = trial;
            } else if (value > 0.0f) {
                lower = trial;
            } else {
                upper = trial;
            }
            const float newton =
                trial -
                value /
                    (
                        isfinite(boundaryDerivative) &&
                            abs(boundaryDerivative) >
                                pivotFloor
                        ? boundaryDerivative
                        : -pivotFloor
                    );
            trial =
                isfinite(newton) &&
                    newton > lower &&
                    newton < upper
                ? newton
                : 0.5f * (lower + upper);
        }
        eta = bestTrial;
        if (!isfinite(coneBoundaryFunction(
                diagonal,
                transformed,
                dimension,
                eta,
                projected,
                boundaryDerivative
            ))) {
            return false;
        }
    }

    const float cap =
        maximumNormal > 0.0f
        ? maximumNormal + adhesion
        : 0.0f;
    float capAlpha = 0.0f;
    if (maximumNormal > 0.0f &&
        projected[0] > cap) {
        float projectedTangentSquared = 0.0f;
        for (uint index = 1u;
             index < dimension;
             ++index) {
            projectedTangentSquared = fma(
                transformed[index],
                transformed[index],
                projectedTangentSquared
            );
        }
        projected[0] = cap;
        if (sqrt(max(projectedTangentSquared, 0.0f)) <=
            cap) {
            region = cappedInteriorRegion;
            for (uint index = 1u;
                 index < dimension;
                 ++index) {
                projected[index] = transformed[index];
            }
        } else {
            region = cappedBoundaryRegion;
            float lower = 0.0f;
            float upper = 1.0f;
            thread float work[6];
            float boundaryDerivative = 0.0f;
            float upperValue = ballBoundaryFunction(
                diagonal,
                transformed,
                dimension,
                cap,
                upper,
                work,
                boundaryDerivative
            );
            for (uint expansion = 0u;
                 expansion < 24u &&
                     upperValue > 0.0f;
                 ++expansion) {
                upper =
                    2.0f * upper +
                    max(diagonal[0], 1.0f);
                upperValue = ballBoundaryFunction(
                    diagonal,
                    transformed,
                    dimension,
                    cap,
                    upper,
                    work,
                    boundaryDerivative
                );
            }
            if (!isfinite(upperValue) ||
                upperValue > 0.0f) {
                return false;
            }
            float trial = 0.5f * (lower + upper);
            float bestTrial = trial;
            float bestAbsoluteValue = INFINITY;
            for (uint iteration = 0u;
                 iteration < 12u;
                 ++iteration) {
                const float value = ballBoundaryFunction(
                    diagonal,
                    transformed,
                    dimension,
                    cap,
                    trial,
                    work,
                    boundaryDerivative
                );
                if (isfinite(value) &&
                    abs(value) < bestAbsoluteValue) {
                    bestAbsoluteValue = abs(value);
                    bestTrial = trial;
                }
                if (value > 0.0f) {
                    lower = trial;
                } else {
                    upper = trial;
                }
                const float newton =
                    trial -
                    value /
                        (
                            isfinite(boundaryDerivative) &&
                                abs(boundaryDerivative) >
                                    pivotFloor
                            ? boundaryDerivative
                            : -pivotFloor
                        );
                trial =
                    isfinite(newton) &&
                        newton > lower &&
                        newton < upper
                    ? newton
                    : 0.5f * (lower + upper);
            }
            capAlpha = bestTrial;
            static_cast<void>(ballBoundaryFunction(
                diagonal,
                transformed,
                dimension,
                cap,
                capAlpha,
                projected,
                boundaryDerivative
            ));
        }
    }

    if (region == interiorRegion) {
        for (uint index = 0u;
             index < dimension;
             ++index) {
            derivative[index * dimension + index] =
                1.0f / regularization[index];
        }
    } else if (region == boundaryRegion) {
        const float radius = max(projected[0], pivotFloor);
        thread float unit[6];
        unit[0] = -1.0f;
        for (uint index = 1u;
             index < dimension;
             ++index) {
            unit[index] = projected[index] / radius;
        }
        thread float kkt[7][7];
        thread float inverse[7][7];
        for (uint row = 0u; row < 7u; ++row) {
            for (uint column = 0u; column < 7u; ++column) {
                kkt[row][column] = 0.0f;
            }
        }
        kkt[0][0] = diagonal[0];
        for (uint row = 1u; row < dimension; ++row) {
            for (uint column = 1u;
                 column < dimension;
                 ++column) {
                const float curvature =
                    eta / radius *
                    (
                        (row == column ? 1.0f : 0.0f) -
                        unit[row] * unit[column]
                    );
                kkt[row][column] =
                    (row == column ? diagonal[row] : 0.0f) +
                    curvature;
            }
        }
        for (uint index = 0u; index < dimension; ++index) {
            kkt[index][dimension] = unit[index];
            kkt[dimension][index] = unit[index];
        }
        if (!invertSmallMatrix(
                kkt,
                inverse,
                dimension + 1u,
                pivotFloor
            )) {
            return false;
        }
        for (uint row = 0u; row < dimension; ++row) {
            for (uint column = 0u;
                 column < dimension;
                 ++column) {
                derivative[row * dimension + column] =
                    scale[row] *
                    inverse[row][column] *
                    scale[column];
            }
        }
    } else if (region == cappedInteriorRegion) {
        for (uint index = 1u;
             index < dimension;
             ++index) {
            derivative[index * dimension + index] =
                1.0f / regularization[index];
        }
    } else if (region == cappedBoundaryRegion) {
        const uint tangentDimension = dimension - 1u;
        const float radius = max(cap, pivotFloor);
        thread float unit[6];
        for (uint index = 0u;
             index < tangentDimension;
             ++index) {
            unit[index] =
                projected[index + 1u] / radius;
        }
        thread float kkt[7][7];
        thread float inverse[7][7];
        for (uint row = 0u; row < 7u; ++row) {
            for (uint column = 0u; column < 7u; ++column) {
                kkt[row][column] = 0.0f;
            }
        }
        for (uint row = 0u;
             row < tangentDimension;
             ++row) {
            for (uint column = 0u;
                 column < tangentDimension;
                 ++column) {
                const float curvature =
                    capAlpha / radius *
                    (
                        (row == column ? 1.0f : 0.0f) -
                        unit[row] * unit[column]
                    );
                kkt[row][column] =
                    (
                        row == column
                        ? diagonal[row + 1u]
                        : 0.0f
                    ) + curvature;
            }
            kkt[row][tangentDimension] = unit[row];
            kkt[tangentDimension][row] = unit[row];
        }
        if (!invertSmallMatrix(
                kkt,
                inverse,
                tangentDimension + 1u,
                pivotFloor
            )) {
            return false;
        }
        for (uint row = 0u;
             row < tangentDimension;
             ++row) {
            for (uint column = 0u;
                 column < tangentDimension;
                 ++column) {
                derivative[
                    (row + 1u) * dimension +
                    column + 1u
                ] =
                    scale[row + 1u] *
                    inverse[row][column] *
                    scale[column + 1u];
            }
        }
    }

    float scaledTangentSquared = 0.0f;
    float dualInner = 0.0f;
    objective = 0.0f;
    for (uint index = 0u; index < dimension; ++index) {
        impulse[index] =
            scale[index] * projected[index] -
            (index == 0u ? adhesion : 0.0f);
        objective -=
            0.5f * regularization[index] *
                impulse[index] * impulse[index] +
            relativeVelocity[index] * impulse[index];
        if (index > 0u) {
            const float scaled =
                impulse[index] / scale[index];
            scaledTangentSquared =
                fma(scaled, scaled, scaledTangentSquared);
        }
        dualInner = fma(
            impulse[index],
            relativeVelocity[index] +
                regularization[index] * impulse[index],
            dualInner
        );
    }
    const float shiftedNormal = impulse[0] + adhesion;
    feasibility = max(
        max(
            -shiftedNormal,
            sqrt(max(scaledTangentSquared, 0.0f)) -
                shiftedNormal
        ),
        maximumNormal > 0.0f
        ? impulse[0] - maximumNormal
        : 0.0f
    );
    feasibility = max(feasibility, 0.0f);
    complementarity = abs(dualInner);
    return
        isfinite(objective) &&
        isfinite(feasibility) &&
        isfinite(complementarity);
}

inline bool projectScalarBlock(
    const MRUnifiedQualityBlockGPU block,
    const float relativeVelocity,
    thread float& impulse,
    thread float& derivative,
    thread float& objective,
    thread float& feasibility,
    thread float& complementarity
) {
    const float regularization = block.regularization0.x;
    const float lower = block.boundsAndShift.x;
    const float upper = block.boundsAndShift.y;
    if (!(regularization > 0.0f) ||
        !isfinite(regularization) ||
        !isfinite(relativeVelocity) ||
        !isfinite(lower) ||
        !isfinite(upper) ||
        lower > upper) {
        return false;
    }
    const float unconstrained =
        -relativeVelocity / regularization;
    impulse = clamp(unconstrained, lower, upper);
    derivative =
        unconstrained > lower && unconstrained < upper
        ? 1.0f / regularization
        : 0.0f;
    objective =
        -0.5f * regularization * impulse * impulse -
        relativeVelocity * impulse;
    feasibility = max(
        max(lower - impulse, impulse - upper),
        0.0f
    );
    complementarity = abs(
        impulse *
        (relativeVelocity + regularization * impulse)
    );
    return
        isfinite(impulse) &&
        isfinite(derivative) &&
        isfinite(objective);
}

inline bool validBlockTopology(
    const MRUnifiedQualityBlockGPU block,
    const uint rowCount
) {
    const uint rowOffset = block.layout.x;
    const uint dimension = block.layout.y;
    const uint kind = block.layout.z;
    return
        finiteBlock(block) &&
        block.layout.w ==
            (
                block.layout.w &
                (
                    MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY |
                    MR_UNIFIED_QUALITY_BLOCK_REPORT_FLOOR
                )
            ) &&
        dimension > 0u &&
        dimension <= kMaxBlockDimension &&
        rowOffset <= rowCount &&
        dimension <= rowCount - rowOffset &&
        (
            (kind == MR_UNIFIED_QUALITY_SCALAR_INTERVAL &&
             dimension == 1u) ||
            (kind == MR_UNIFIED_QUALITY_ELLIPTIC_CONE &&
             dimension >= 3u)
        );
}

// Evaluates the common primal objective, gradient, product-cone prox, and
// generalized derivative. Exactly one SIMD32 group owns the problem.
inline bool evaluateState(
    device const MRUnifiedQualityDispatchGPU& dispatch,
    device const MRUnifiedQualityBlockGPU* blocks,
    device const float* dynamics,
    device const float* jacobian,
    device const float* bias,
    device const float* freeVelocity,
    threadgroup const float* velocity,
    threadgroup float* rowVelocity,
    threadgroup float* impulses,
    threadgroup float* gradient,
    threadgroup float* dynamicsAction,
    threadgroup float* constraintAction,
    device float* derivatives,
    threadgroup float* scalars,
    threadgroup uint* control,
    const bool writeDerivatives,
    const uint lane
) {
    const uint nv = dispatch.generalizedVelocityCount;
    const uint rowCount = dispatch.rowCount;
    for (uint row = lane; row < rowCount; row += kWidth) {
        float value = bias[row];
        for (uint dof = 0u; dof < nv; ++dof) {
            value = fma(
                jacobian[row * nv + dof],
                velocity[dof],
                value
            );
        }
        rowVelocity[row] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localObjective = 0.0f;
    float localFeasibility = 0.0f;
    float localEquality = 0.0f;
    float localComplementarity = 0.0f;
    uint localFailingBlock =
        MR_UNIFIED_QUALITY_INVALID_INDEX;
    for (uint blockIndex = lane;
         blockIndex < dispatch.blockCount;
         blockIndex += kWidth) {
        const MRUnifiedQualityBlockGPU block =
            blocks[blockIndex];
        if (!validBlockTopology(block, rowCount)) {
            localFailingBlock =
                min(localFailingBlock, blockIndex);
            continue;
        }
        const uint rowOffset = block.layout.x;
        const uint dimension = block.layout.y;
        thread float localVelocity[6];
        thread float localImpulse[6];
        thread float localDerivative[36];
        for (uint index = 0u;
             index < kMaxBlockDimension;
             ++index) {
            localVelocity[index] =
                index < dimension
                ? rowVelocity[rowOffset + index]
                : 0.0f;
            localImpulse[index] = 0.0f;
        }
        for (uint entry = 0u; entry < 36u; ++entry) {
            localDerivative[entry] = 0.0f;
        }
        float blockObjective = 0.0f;
        float blockFeasibility = 0.0f;
        float blockComplementarity = 0.0f;
        bool valid = false;
        if (block.layout.z ==
            MR_UNIFIED_QUALITY_SCALAR_INTERVAL) {
            valid = projectScalarBlock(
                block,
                localVelocity[0],
                localImpulse[0],
                localDerivative[0],
                blockObjective,
                blockFeasibility,
                blockComplementarity
            );
        } else {
            valid = projectConeBlock(
                block,
                localVelocity,
                localImpulse,
                localDerivative,
                blockObjective,
                blockFeasibility,
                blockComplementarity,
                dispatch.numerics.y
            );
        }
        if (!valid) {
            localFailingBlock =
                min(localFailingBlock, blockIndex);
            continue;
        }
        for (uint index = 0u;
             index < dimension;
             ++index) {
            impulses[rowOffset + index] =
                localImpulse[index];
        }
        if (writeDerivatives) {
            const uint derivativeBase = blockIndex * 36u;
            for (uint entry = 0u; entry < 36u; ++entry) {
                derivatives[derivativeBase + entry] =
                    localDerivative[entry];
            }
        }
        localObjective += blockObjective;
        localFeasibility =
            max(localFeasibility, blockFeasibility);
        localComplementarity = max(
            localComplementarity,
            blockComplementarity
        );
        if ((block.layout.w &
             MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY) != 0u) {
            for (uint index = 0u;
                 index < dimension;
                 ++index) {
                localEquality = max(
                    localEquality,
                    abs(localVelocity[index])
                );
            }
        }
    }
    const uint failingBlock =
        simd_min(localFailingBlock);
    if (lane == 0u) {
        control[0] = failingBlock;
    }
    // The derivative arena is invocation-local. An unconditional device
    // barrier keeps the direct and PCG consumers correct while preserving
    // one identical evaluation function for objective-only line searches.
    threadgroup_barrier(
        mem_flags::mem_threadgroup |
        mem_flags::mem_device
    );
    if (control[0] !=
        MR_UNIFIED_QUALITY_INVALID_INDEX) {
        return false;
    }

    float localDynamicsObjective = 0.0f;
    for (uint row = lane; row < nv; row += kWidth) {
        float action = 0.0f;
        for (uint column = 0u; column < nv; ++column) {
            action = fma(
                dynamics[row * nv + column],
                velocity[column] - freeVelocity[column],
                action
            );
        }
        dynamicsAction[row] = action;
        localDynamicsObjective +=
            0.5f *
            (velocity[row] - freeVelocity[row]) *
            action;
        float constraint = 0.0f;
        for (uint constraintRow = 0u;
             constraintRow < rowCount;
             ++constraintRow) {
            constraint = fma(
                jacobian[constraintRow * nv + row],
                impulses[constraintRow],
                constraint
            );
        }
        constraintAction[row] = constraint;
        gradient[row] = action - constraint;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float dynamicsObjective =
        simd_sum(localDynamicsObjective);
    const float proxObjective = simd_sum(localObjective);
    const float feasibility =
        simd_max(localFeasibility);
    const float equality = simd_max(localEquality);
    const float complementarity =
        simd_max(localComplementarity);
    const float gradientMaximum =
        groupMaximumAbsolute(gradient, nv, lane);
    const float dynamicsMaximum =
        groupMaximumAbsolute(dynamicsAction, nv, lane);
    const float constraintMaximum =
        groupMaximumAbsolute(constraintAction, nv, lane);
    if (lane == 0u) {
        scalars[0] = dynamicsObjective + proxObjective;
        scalars[1] =
            gradientMaximum /
            (1.0f + dynamicsMaximum + constraintMaximum);
        scalars[2] = feasibility;
        scalars[3] = equality;
        scalars[4] = complementarity;
        scalars[5] =
            gradientMaximum /
            (1.0f + dynamicsMaximum + constraintMaximum);
        scalars[6] = dynamicsMaximum;
        scalars[7] = constraintMaximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return
        all(isfinite(float4(
            scalars[0],
            scalars[1],
            scalars[2],
            scalars[3]
        ))) &&
        isfinite(scalars[4]);
}

inline void applyHessian(
    device const MRUnifiedQualityDispatchGPU& dispatch,
    device const MRUnifiedQualityBlockGPU* blocks,
    device const float* dynamics,
    device const float* jacobian,
    device const float* derivatives,
    threadgroup const float* input,
    threadgroup float* rowWork,
    threadgroup float* rowAction,
    threadgroup float* output,
    const float diagonalRegularization,
    const uint lane
) {
    const uint nv = dispatch.generalizedVelocityCount;
    for (uint row = lane;
         row < dispatch.rowCount;
         row += kWidth) {
        float value = 0.0f;
        for (uint dof = 0u; dof < nv; ++dof) {
            value = fma(
                jacobian[row * nv + dof],
                input[dof],
                value
            );
        }
        rowWork[row] = value;
        rowAction[row] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blockIndex = lane;
         blockIndex < dispatch.blockCount;
         blockIndex += kWidth) {
        const MRUnifiedQualityBlockGPU block =
            blocks[blockIndex];
        const uint offset = block.layout.x;
        const uint dimension = block.layout.y;
        const uint derivativeBase = blockIndex * 36u;
        for (uint row = 0u; row < dimension; ++row) {
            float value = 0.0f;
            for (uint column = 0u;
                 column < dimension;
                 ++column) {
                value = fma(
                    derivatives[
                        derivativeBase +
                        row * dimension +
                        column
                    ],
                    rowWork[offset + column],
                    value
                );
            }
            rowAction[offset + row] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dof = lane; dof < nv; dof += kWidth) {
        float value =
            diagonalRegularization * input[dof];
        for (uint column = 0u; column < nv; ++column) {
            value = fma(
                dynamics[dof * nv + column],
                input[column],
                value
            );
        }
        for (uint row = 0u;
             row < dispatch.rowCount;
             ++row) {
            value = fma(
                jacobian[row * nv + dof],
                rowAction[row],
                value
            );
        }
        output[dof] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float directHessianEntry(
    device const MRUnifiedQualityDispatchGPU& dispatch,
    device const MRUnifiedQualityBlockGPU* blocks,
    device const float* dynamics,
    device const float* jacobian,
    device const float* derivatives,
    const uint row,
    const uint column,
    const float diagonalRegularization
) {
    const uint nv = dispatch.generalizedVelocityCount;
    float value =
        dynamics[row * nv + column] +
        (
            row == column
            ? diagonalRegularization
            : 0.0f
        );
    for (uint blockIndex = 0u;
         blockIndex < dispatch.blockCount;
         ++blockIndex) {
        const MRUnifiedQualityBlockGPU block =
            blocks[blockIndex];
        const uint offset = block.layout.x;
        const uint dimension = block.layout.y;
        const uint derivativeBase = blockIndex * 36u;
        for (uint localRow = 0u;
             localRow < dimension;
             ++localRow) {
            const float left =
                jacobian[(offset + localRow) * nv + row];
            for (uint localColumn = 0u;
                 localColumn < dimension;
                 ++localColumn) {
                value = fma(
                    left *
                        derivatives[
                            derivativeBase +
                            localRow * dimension +
                            localColumn
                        ],
                    jacobian[
                        (offset + localColumn) * nv + column
                    ],
                    value
                );
            }
        }
    }
    return value;
}

inline bool factorAndSolveDirect(
    device float* hessian,
    threadgroup const float* gradient,
    threadgroup float* direction,
    threadgroup float* scalars,
    const uint nv,
    const float pivotFloor,
    const uint lane
) {
    if (lane == 0u) {
        bool valid = true;
        float minimumPivot = INFINITY;
        float maximumPivot = 0.0f;
        for (uint row = 0u; row < nv && valid; ++row) {
            for (uint column = 0u;
                 column <= row;
                 ++column) {
                float value = hessian[row * nv + column];
                for (uint inner = 0u;
                     inner < column;
                     ++inner) {
                    value = fma(
                        -hessian[row * nv + inner],
                        hessian[column * nv + inner],
                        value
                    );
                }
                if (row == column) {
                    if (!(value > pivotFloor) ||
                        !isfinite(value)) {
                        valid = false;
                        break;
                    }
                    const float pivot = sqrt(value);
                    hessian[row * nv + row] = pivot;
                    minimumPivot =
                        min(minimumPivot, pivot);
                    maximumPivot =
                        max(maximumPivot, pivot);
                } else {
                    hessian[row * nv + column] =
                        value /
                        hessian[column * nv + column];
                }
            }
        }
        if (valid) {
            for (uint row = 0u; row < nv; ++row) {
                float value = -gradient[row];
                for (uint column = 0u;
                     column < row;
                     ++column) {
                    value = fma(
                        -hessian[row * nv + column],
                        direction[column],
                        value
                    );
                }
                direction[row] =
                    value / hessian[row * nv + row];
            }
            for (uint reverse = 0u;
                 reverse < nv;
                 ++reverse) {
                const uint row = nv - 1u - reverse;
                float value = direction[row];
                for (uint column = row + 1u;
                     column < nv;
                     ++column) {
                    value = fma(
                        -hessian[column * nv + row],
                        direction[column],
                        value
                    );
                }
                direction[row] =
                    value / hessian[row * nv + row];
            }
        }
        scalars[8] = valid ? 1.0f : 0.0f;
        scalars[9] = minimumPivot;
        scalars[10] = maximumPivot;
    }
    threadgroup_barrier(
        mem_flags::mem_threadgroup |
        mem_flags::mem_device
    );
    return scalars[8] == 1.0f;
}

// Node dynamics are block diagonal across articulations, maximal bodies, and
// connected rods. A deterministic symmetric Gauss-Seidel application keeps
// all of each node's retained off-diagonal dynamics in the PCG
// preconditioner. Unlike scalar Jacobi this captures the rod band and the
// articulated tree factor without adding contact-space storage. The
// (D+L) D^-1 (D+U) construction is SPD whenever A is SPD.
inline bool applyDynamicsSSORPreconditioner(
    device const float* dynamics,
    threadgroup const float* residual,
    threadgroup float* output,
    const uint nv,
    const float pivotFloor,
    threadgroup float* scalars,
    const uint lane
) {
    if (lane == 0u) {
        bool valid = true;
        for (uint row = 0u; row < nv && valid; ++row) {
            const float diagonal =
                dynamics[row * nv + row];
            float value = residual[row];
            for (uint column = 0u;
                 column < row;
                 ++column) {
                value = fma(
                    -dynamics[row * nv + column],
                    output[column],
                    value
                );
            }
            valid =
                diagonal > pivotFloor &&
                isfinite(diagonal) &&
                isfinite(value);
            output[row] =
                valid ? value / diagonal : 0.0f;
        }
        for (uint reverse = 0u;
             reverse < nv && valid;
             ++reverse) {
            const uint row = nv - 1u - reverse;
            const float diagonal =
                dynamics[row * nv + row];
            float value = diagonal * output[row];
            for (uint column = row + 1u;
                 column < nv;
                 ++column) {
                value = fma(
                    -dynamics[row * nv + column],
                    output[column],
                    value
                );
            }
            valid = isfinite(value);
            output[row] =
                valid ? value / diagonal : 0.0f;
        }
        scalars[11] = valid ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scalars[11] == 1.0f;
}

} // namespace

inline void mrUnifiedQualitySolveProblem(
    device const MRUnifiedQualityDispatchGPU& dispatch,
    device const MRUnifiedQualityBlockGPU* blocks,
    device const float* dynamicsMatrices,
    device const float* jacobianMatrices,
    device const float* biasVectors,
    device const float* freeVelocities,
    device const float* warmVelocities,
    device const float* warmImpulses,
    device float* outputVelocities,
    device float* outputImpulses,
    device float* derivativeScratch,
    device float* hessianScratch,
    device MRUnifiedQualityStatusGPU* statuses,
    const uint problem,
    const uint lane,
    threadgroup float* velocity,
    threadgroup float* candidateVelocity,
    threadgroup float* direction,
    threadgroup float* gradient,
    threadgroup float* candidateGradient,
    threadgroup float* dynamicsAction,
    threadgroup float* candidateDynamicsAction,
    threadgroup float* constraintAction,
    threadgroup float* candidateConstraintAction,
    threadgroup float* cgResidual,
    threadgroup float* cgPreconditioned,
    threadgroup float* cgDirection,
    threadgroup float* cgAction,
    threadgroup float* preconditioner,
    threadgroup float* rowVelocity,
    threadgroup float* candidateRowVelocity,
    threadgroup float* impulses,
    threadgroup float* candidateImpulses,
    threadgroup float* rowWork,
    threadgroup float* rowAction,
    threadgroup float* scalars,
    threadgroup float* candidateScalars,
    threadgroup uint* control
) {
    if (problem >= dispatch.problemCount) {
        return;
    }
    MRUnifiedQualityStatusGPU status = {};
    status.code = MR_UNIFIED_QUALITY_SUCCESS;
    status.problemIndex = problem;
    status.failingBlock =
        MR_UNIFIED_QUALITY_INVALID_INDEX;
    status.firstFailingStableKey =
        uint4(MR_UNIFIED_QUALITY_INVALID_INDEX);

    const uint nv = dispatch.generalizedVelocityCount;
    const uint rowCount = dispatch.rowCount;
    const bool directPath =
        nv <= dispatch.directMaximumGeneralizedVelocities &&
        rowCount <= dispatch.directMaximumRows;
    const bool validDispatch =
        dispatch.abiVersion ==
            MR_UNIFIED_QUALITY_ABI_VERSION &&
        dispatch.problemCount > 0u &&
        nv > 0u &&
        nv <= kMaxV &&
        rowCount > 0u &&
        rowCount <= kMaxRows &&
        dispatch.blockCount > 0u &&
        dispatch.blockCount <= kMaxBlocks &&
        dispatch.dynamicsStride >= nv * nv &&
        dispatch.jacobianStride >= rowCount * nv &&
        dispatch.vectorStride >= max(nv, rowCount) &&
        dispatch.derivativeStride >=
            dispatch.blockCount * 36u &&
        dispatch.hessianStride >=
            (directPath ? nv * nv : 1u) &&
        (
            dispatch.blockStride == 0u ||
            dispatch.blockStride >= dispatch.blockCount
        ) &&
        dispatch.maximumNewtonIterations > 0u &&
        dispatch.maximumPCGIterations > 0u &&
        dispatch.maximumLineSearchIterations > 0u &&
        dispatch.directMaximumGeneralizedVelocities > 0u &&
        dispatch.directMaximumRows > 0u &&
        all(isfinite(dispatch.tolerances)) &&
        all(isfinite(dispatch.numerics)) &&
        dispatch.tolerances.x > 0.0f &&
        dispatch.tolerances.y > 0.0f &&
        dispatch.tolerances.z > 0.0f &&
        dispatch.tolerances.z < 0.5f &&
        dispatch.tolerances.w > 0.0f &&
        dispatch.tolerances.w < 1.0f &&
        dispatch.numerics.x > 0.0f &&
        dispatch.numerics.y > 0.0f &&
        dispatch.numerics.z > 0.0f &&
        dispatch.numerics.w > 1.0f;
    if (!validDispatch) {
        if (lane == 0u) {
            status.code =
                MR_UNIFIED_QUALITY_INVALID_DISPATCH;
            statuses[problem] = status;
        }
        return;
    }

    device const float* dynamics =
        dynamicsMatrices +
        problem * dispatch.dynamicsStride;
    device const float* jacobian =
        jacobianMatrices +
        problem * dispatch.jacobianStride;
    device const float* bias =
        biasVectors + problem * dispatch.vectorStride;
    device const float* freeVelocity =
        freeVelocities + problem * dispatch.vectorStride;
    device const float* warmVelocity =
        warmVelocities + problem * dispatch.vectorStride;
    device const float* warmImpulse =
        warmImpulses + problem * dispatch.vectorStride;
    device float* derivatives =
        derivativeScratch +
        problem * dispatch.derivativeStride;
    device float* hessian =
        hessianScratch + problem * dispatch.hessianStride;
    device const MRUnifiedQualityBlockGPU* problemBlocks =
        blocks +
        (
            dispatch.blockStride == 0u
            ? 0u
            : problem * dispatch.blockStride
        );

    uint localInputFailure =
        MR_UNIFIED_QUALITY_INVALID_INDEX;
    for (uint dof = lane; dof < nv; dof += kWidth) {
        if (!isfinite(freeVelocity[dof]) ||
            !isfinite(warmVelocity[dof])) {
            localInputFailure = 0u;
        }
        for (uint column = 0u; column < nv; ++column) {
            if (!isfinite(dynamics[dof * nv + column])) {
                localInputFailure = 0u;
            }
        }
        velocity[dof] = freeVelocity[dof];
        candidateVelocity[dof] = warmVelocity[dof];
    }
    for (uint row = lane; row < rowCount; row += kWidth) {
        if (!isfinite(bias[row]) ||
            !isfinite(warmImpulse[row])) {
            localInputFailure = 0u;
        }
        for (uint dof = 0u; dof < nv; ++dof) {
            if (!isfinite(jacobian[row * nv + dof])) {
                localInputFailure = 0u;
            }
        }
    }
    const uint inputFailure = simd_min(localInputFailure);
    if (inputFailure !=
        MR_UNIFIED_QUALITY_INVALID_INDEX) {
        if (lane == 0u) {
            status.code =
                MR_UNIFIED_QUALITY_NONFINITE_INPUT;
            statuses[problem] = status;
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool freeValid = evaluateState(
        dispatch,
        problemBlocks,
        dynamics,
        jacobian,
        bias,
        freeVelocity,
        velocity,
        rowVelocity,
        impulses,
        gradient,
        dynamicsAction,
        constraintAction,
        derivatives,
        scalars,
        control,
        false,
        lane
    );
    const bool warmValid = evaluateState(
        dispatch,
        problemBlocks,
        dynamics,
        jacobian,
        bias,
        freeVelocity,
        candidateVelocity,
        candidateRowVelocity,
        candidateImpulses,
        candidateGradient,
        candidateDynamicsAction,
        candidateConstraintAction,
        derivatives,
        candidateScalars,
        control,
        false,
        lane
    );
    if (!freeValid || !warmValid) {
        if (lane == 0u) {
            status.code =
                control[0] ==
                    MR_UNIFIED_QUALITY_INVALID_INDEX
                ? MR_UNIFIED_QUALITY_NONFINITE_INPUT
                : MR_UNIFIED_QUALITY_INVALID_BLOCK;
            status.failingBlock = control[0];
            if (control[0] < dispatch.blockCount) {
                status.firstFailingStableKey =
                    problemBlocks[control[0]].stableKey;
            }
            statuses[problem] = status;
        }
        return;
    }
    float localMinimumBlockRegularization = INFINITY;
    for (uint blockIndex = lane;
         blockIndex < dispatch.blockCount;
         blockIndex += kWidth) {
        const MRUnifiedQualityBlockGPU block =
            problemBlocks[blockIndex];
        for (uint component = 0u;
             component < block.layout.y;
             ++component) {
            localMinimumBlockRegularization = min(
                localMinimumBlockRegularization,
                blockFloat(
                    block.regularization0,
                    block.regularization1,
                    component
                )
            );
        }
    }
    const float minimumBlockRegularization =
        simd_min(localMinimumBlockRegularization);
    if (candidateScalars[0] < scalars[0]) {
        for (uint dof = lane; dof < nv; dof += kWidth) {
            velocity[dof] = candidateVelocity[dof];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    status.solvePath =
        directPath
        ? MR_UNIFIED_QUALITY_PATH_DIRECT
        : MR_UNIFIED_QUALITY_PATH_PCG;
    bool converged = false;
    uint totalPCGIterations = 0u;
    uint totalBacktracks = 0u;
    uint regularizationRetries = 0u;
    float previousGradientNorm = INFINITY;
    float objectiveChange = INFINITY;
    float newtonDecrement = INFINITY;
    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    uint completedNewtonIterations = 0u;
    uint failureCode = MR_UNIFIED_QUALITY_SUCCESS;

    for (uint newton = 0u;
         newton < dispatch.maximumNewtonIterations;
         ++newton) {
        if (!evaluateState(
                dispatch,
                problemBlocks,
                dynamics,
                jacobian,
                bias,
                freeVelocity,
                velocity,
                rowVelocity,
                impulses,
                gradient,
                dynamicsAction,
                constraintAction,
                derivatives,
                scalars,
                control,
                true,
                lane
            )) {
            failureCode =
                control[0] ==
                    MR_UNIFIED_QUALITY_INVALID_INDEX
                ? MR_UNIFIED_QUALITY_NONFINITE_RESULT
                : MR_UNIFIED_QUALITY_INVALID_BLOCK;
            break;
        }
        const float gradientNormSquared =
            groupDot(gradient, gradient, nv, lane);
        const float gradientNorm =
            sqrt(max(gradientNormSquared, 0.0f));
        if (scalars[1] <= dispatch.tolerances.x &&
            scalars[2] <= dispatch.tolerances.y) {
            converged = true;
            completedNewtonIterations = newton;
            break;
        }

        bool directionValid = true;
        float appliedRetry = 0.0f;
        if (directPath) {
            for (uint entry = lane;
                 entry < nv * nv;
                 entry += kWidth) {
                const uint row = entry / nv;
                const uint column = entry - row * nv;
                hessian[entry] = directHessianEntry(
                    dispatch,
                    problemBlocks,
                    dynamics,
                    jacobian,
                    derivatives,
                    row,
                    column,
                    0.0f
                );
            }
            threadgroup_barrier(mem_flags::mem_device);
            directionValid = factorAndSolveDirect(
                hessian,
                gradient,
                direction,
                scalars,
                nv,
                dispatch.numerics.y,
                lane
            );
            if (!directionValid) {
                appliedRetry =
                    dispatch.numerics.w *
                    dispatch.numerics.y;
                for (uint entry = lane;
                     entry < nv * nv;
                     entry += kWidth) {
                    const uint row = entry / nv;
                    const uint column = entry - row * nv;
                    hessian[entry] = directHessianEntry(
                        dispatch,
                        problemBlocks,
                        dynamics,
                        jacobian,
                        derivatives,
                        row,
                        column,
                        appliedRetry
                    );
                }
                threadgroup_barrier(mem_flags::mem_device);
                directionValid = factorAndSolveDirect(
                    hessian,
                    gradient,
                    direction,
                    scalars,
                    nv,
                    dispatch.numerics.y,
                    lane
                );
                regularizationRetries += 1u;
            }
            minimumPivot = min(minimumPivot, scalars[9]);
            maximumPivot = max(maximumPivot, scalars[10]);
        } else {
            for (uint dof = lane; dof < nv; dof += kWidth) {
                float diagonal =
                    dynamics[dof * nv + dof];
                for (uint blockIndex = 0u;
                     blockIndex < dispatch.blockCount;
                     ++blockIndex) {
                    const MRUnifiedQualityBlockGPU block =
                        problemBlocks[blockIndex];
                    const uint offset = block.layout.x;
                    const uint dimension = block.layout.y;
                    const uint derivativeBase =
                        blockIndex * 36u;
                    for (uint row = 0u;
                         row < dimension;
                         ++row) {
                        const float left =
                            jacobian[
                                (offset + row) * nv + dof
                            ];
                        for (uint column = 0u;
                             column < dimension;
                             ++column) {
                            diagonal = fma(
                                left *
                                    derivatives[
                                        derivativeBase +
                                        row * dimension +
                                        column
                                    ],
                                jacobian[
                                    (offset + column) *
                                        nv +
                                    dof
                                ],
                                diagonal
                            );
                        }
                    }
                }
                preconditioner[dof] = diagonal;
                direction[dof] = 0.0f;
                cgResidual[dof] = -gradient[dof];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            directionValid =
                applyDynamicsSSORPreconditioner(
                    dynamics,
                    cgResidual,
                    cgPreconditioned,
                    nv,
                    dispatch.numerics.y,
                    scalars,
                    lane
                );
            for (uint dof = lane;
                 dof < nv;
                 dof += kWidth) {
                cgDirection[dof] =
                    cgPreconditioned[dof];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float rz = groupDot(
                cgResidual,
                cgPreconditioned,
                nv,
                lane
            );
            const float forcing = clamp(
                isfinite(previousGradientNorm)
                ? sqrt(
                      gradientNorm /
                      max(previousGradientNorm, 1.0e-20f)
                  )
                : 0.1f,
                1.0e-4f,
                0.1f
            );
            const float targetSquared =
                forcing * forcing *
                max(gradientNormSquared, 1.0e-24f);
            directionValid = directionValid &&
                isfinite(rz) && rz >= 0.0f;
            uint pcgIterations = 0u;
            for (; directionValid &&
                   pcgIterations <
                       dispatch.maximumPCGIterations;
                 ++pcgIterations) {
                const float residualSquared =
                    groupDot(
                        cgResidual,
                        cgResidual,
                        nv,
                        lane
                    );
                if (residualSquared <= targetSquared) {
                    break;
                }
                applyHessian(
                    dispatch,
                    problemBlocks,
                    dynamics,
                    jacobian,
                    derivatives,
                    cgDirection,
                    rowWork,
                    rowAction,
                    cgAction,
                    0.0f,
                    lane
                );
                const float denominator = groupDot(
                    cgDirection,
                    cgAction,
                    nv,
                    lane
                );
                if (!(denominator >
                      dispatch.numerics.z) ||
                    !isfinite(denominator)) {
                    directionValid = false;
                    break;
                }
                const float alpha = rz / denominator;
                for (uint dof = lane;
                     dof < nv;
                     dof += kWidth) {
                    direction[dof] = fma(
                        alpha,
                        cgDirection[dof],
                        direction[dof]
                    );
                    cgResidual[dof] = fma(
                        -alpha,
                        cgAction[dof],
                        cgResidual[dof]
                    );
                }
                threadgroup_barrier(
                    mem_flags::mem_threadgroup
                );
                directionValid =
                    applyDynamicsSSORPreconditioner(
                        dynamics,
                        cgResidual,
                        cgPreconditioned,
                        nv,
                        dispatch.numerics.y,
                        scalars,
                        lane
                    );
                if (!directionValid) {
                    break;
                }
                const float nextRz = groupDot(
                    cgResidual,
                    cgPreconditioned,
                    nv,
                    lane
                );
                if (!isfinite(nextRz) || nextRz < 0.0f) {
                    directionValid = false;
                    break;
                }
                const float beta =
                    nextRz /
                    max(rz, dispatch.numerics.z);
                for (uint dof = lane;
                     dof < nv;
                     dof += kWidth) {
                    cgDirection[dof] = fma(
                        beta,
                        cgDirection[dof],
                        cgPreconditioned[dof]
                    );
                }
                threadgroup_barrier(
                    mem_flags::mem_threadgroup
                );
                rz = nextRz;
            }
            totalPCGIterations += pcgIterations;
            if (directionValid) {
                const float finalResidualSquared =
                    groupDot(
                        cgResidual,
                        cgResidual,
                        nv,
                        lane
                    );
                directionValid =
                    isfinite(finalResidualSquared) &&
                    finalResidualSquared <=
                        max(
                            targetSquared,
                            dispatch.tolerances.x *
                                dispatch.tolerances.x
                        );
            }
        }
        previousGradientNorm = gradientNorm;
        if (!directionValid) {
            failureCode =
                directPath
                ? MR_UNIFIED_QUALITY_FACTORIZATION_FAILED
                : MR_UNIFIED_QUALITY_PCG_FAILED;
            break;
        }

        float directionalDerivative =
            groupDot(gradient, direction, nv, lane);
        newtonDecrement = max(
            -directionalDerivative,
            0.0f
        );
        if (!isfinite(directionalDerivative) ||
            !(directionalDerivative < 0.0f)) {
            failureCode =
                directPath
                ? MR_UNIFIED_QUALITY_FACTORIZATION_FAILED
                : MR_UNIFIED_QUALITY_PCG_FAILED;
            break;
        }

        bool accepted = false;
        float stepLength = 1.0f;
        for (uint lineSearch = 0u;
             lineSearch <
                 dispatch.maximumLineSearchIterations;
             ++lineSearch) {
            for (uint dof = lane;
                 dof < nv;
                 dof += kWidth) {
                candidateVelocity[dof] = fma(
                    stepLength,
                    direction[dof],
                    velocity[dof]
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const bool candidateValid = evaluateState(
                dispatch,
                problemBlocks,
                dynamics,
                jacobian,
                bias,
                freeVelocity,
                candidateVelocity,
                candidateRowVelocity,
                candidateImpulses,
                candidateGradient,
                candidateDynamicsAction,
                candidateConstraintAction,
                derivatives,
                candidateScalars,
                control,
                false,
                lane
            );
            const float objectiveRoundoff =
                32.0f * kFloatEpsilon *
                max(abs(scalars[0]), 1.0f);
            const bool residualFilterAccepted =
                candidateValid &&
                candidateScalars[0] <=
                    scalars[0] + objectiveRoundoff &&
                candidateScalars[1] <
                    (1.0f - 1.0e-3f) * scalars[1] &&
                candidateScalars[2] <=
                    max(
                        scalars[2],
                        dispatch.tolerances.y
                    ) + objectiveRoundoff;
            if (candidateValid &&
                (
                    candidateScalars[0] <=
                        scalars[0] +
                        dispatch.tolerances.z *
                            stepLength *
                            directionalDerivative ||
                    residualFilterAccepted
                )) {
                accepted = true;
                objectiveChange =
                    candidateScalars[0] - scalars[0];
                break;
            }
            ++totalBacktracks;
            stepLength *= dispatch.tolerances.w;
        }
        if (!accepted) {
            // One visible safeguarded retry remains within the same primal
            // objective. It is a diagonally preconditioned gradient step,
            // not a Wave-Jacobi/contact-space fallback.
            for (uint dof = lane;
                 dof < nv;
                 dof += kWidth) {
                const float diagonal = directHessianEntry(
                    dispatch,
                    problemBlocks,
                    dynamics,
                    jacobian,
                    derivatives,
                    dof,
                    dof,
                    dispatch.numerics.w *
                        dispatch.numerics.y
                );
                direction[dof] =
                    -gradient[dof] /
                    max(diagonal, dispatch.numerics.y);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            directionalDerivative =
                groupDot(gradient, direction, nv, lane);
            ++regularizationRetries;
            stepLength = 1.0f;
            if (isfinite(directionalDerivative) &&
                directionalDerivative < 0.0f) {
                for (uint lineSearch = 0u;
                     lineSearch <
                         dispatch.maximumLineSearchIterations;
                     ++lineSearch) {
                    for (uint dof = lane;
                         dof < nv;
                         dof += kWidth) {
                        candidateVelocity[dof] = fma(
                            stepLength,
                            direction[dof],
                            velocity[dof]
                        );
                    }
                    threadgroup_barrier(
                        mem_flags::mem_threadgroup
                    );
                    const bool candidateValid =
                        evaluateState(
                            dispatch,
                            problemBlocks,
                            dynamics,
                            jacobian,
                            bias,
                            freeVelocity,
                            candidateVelocity,
                            candidateRowVelocity,
                            candidateImpulses,
                            candidateGradient,
                            candidateDynamicsAction,
                            candidateConstraintAction,
                            derivatives,
                            candidateScalars,
                            control,
                            false,
                            lane
                        );
                    const float objectiveRoundoff =
                        32.0f * kFloatEpsilon *
                        max(abs(scalars[0]), 1.0f);
                    const bool residualFilterAccepted =
                        candidateValid &&
                        candidateScalars[0] <=
                            scalars[0] +
                                objectiveRoundoff &&
                        candidateScalars[1] <
                            (1.0f - 1.0e-3f) *
                                scalars[1] &&
                        candidateScalars[2] <=
                            max(
                                scalars[2],
                                dispatch.tolerances.y
                            ) + objectiveRoundoff;
                    if (candidateValid &&
                        (
                            candidateScalars[0] <=
                                scalars[0] +
                                dispatch.tolerances.z *
                                    stepLength *
                                    directionalDerivative ||
                            residualFilterAccepted
                        )) {
                        accepted = true;
                        objectiveChange =
                            candidateScalars[0] -
                            scalars[0];
                        break;
                    }
                    ++totalBacktracks;
                    stepLength *=
                        dispatch.tolerances.w;
                }
            }
            if (!accepted) {
                failureCode =
                    MR_UNIFIED_QUALITY_LINE_SEARCH_FAILED;
                break;
            }
        }
        for (uint dof = lane; dof < nv; dof += kWidth) {
            velocity[dof] = candidateVelocity[dof];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        completedNewtonIterations = newton + 1u;
        static_cast<void>(appliedRetry);
    }

    if (failureCode == MR_UNIFIED_QUALITY_SUCCESS &&
        !converged) {
        if (evaluateState(
                dispatch,
                problemBlocks,
                dynamics,
                jacobian,
                bias,
                freeVelocity,
                velocity,
                rowVelocity,
                impulses,
                gradient,
                dynamicsAction,
                constraintAction,
                derivatives,
                scalars,
                control,
                true,
                lane
            ) &&
            scalars[1] <= dispatch.tolerances.x &&
            scalars[2] <= dispatch.tolerances.y) {
            converged = true;
        } else {
            failureCode =
                MR_UNIFIED_QUALITY_DID_NOT_CONVERGE;
        }
    }
    if (failureCode != MR_UNIFIED_QUALITY_SUCCESS) {
        if (lane == 0u) {
            status.code = failureCode;
            status.failingBlock = control[0];
            if (control[0] < dispatch.blockCount) {
                status.firstFailingStableKey =
                    problemBlocks[control[0]].stableKey;
            }
            status.newtonIterations =
                completedNewtonIterations;
            status.pcgIterations = totalPCGIterations;
            status.lineSearchBacktracks = totalBacktracks;
            status.regularizationRetries =
                regularizationRetries;
            status.certificates0 = float4(
                scalars[1],
                scalars[2],
                scalars[3],
                scalars[4]
            );
            status.certificates1 = float4(
                scalars[5],
                isfinite(newtonDecrement)
                    ? newtonDecrement
                    : 0.0f,
                isfinite(objectiveChange)
                    ? objectiveChange
                    : 0.0f,
                scalars[0]
            );
            status.numerics = float4(
                isfinite(minimumBlockRegularization)
                    ? minimumBlockRegularization
                    : 0.0f,
                regularizationRetries == 0u
                ? 0.0f
                : dispatch.numerics.w *
                    dispatch.numerics.y,
                isfinite(minimumPivot)
                    ? minimumPivot
                    : 0.0f,
                isfinite(maximumPivot)
                    ? maximumPivot
                    : 0.0f
            );
            statuses[problem] = status;
        }
        return;
    }

    bool nonfiniteOutput = false;
    for (uint dof = lane; dof < nv; dof += kWidth) {
        nonfiniteOutput =
            nonfiniteOutput || !isfinite(velocity[dof]);
        outputVelocities[
            problem * dispatch.vectorStride + dof
        ] = velocity[dof];
    }
    for (uint row = lane; row < rowCount; row += kWidth) {
        nonfiniteOutput =
            nonfiniteOutput || !isfinite(impulses[row]);
        outputImpulses[
            problem * dispatch.vectorStride + row
        ] = impulses[row];
    }
    if (simd_any(nonfiniteOutput)) {
        if (lane == 0u) {
            status.code =
                MR_UNIFIED_QUALITY_NONFINITE_RESULT;
            statuses[problem] = status;
        }
        return;
    }
    if (lane == 0u) {
        status.code = MR_UNIFIED_QUALITY_SUCCESS;
        status.newtonIterations =
            completedNewtonIterations;
        status.pcgIterations = totalPCGIterations;
        status.lineSearchBacktracks = totalBacktracks;
        status.regularizationRetries =
            regularizationRetries;
        status.certificates0 = float4(
            scalars[1],
            scalars[2],
            scalars[3],
            scalars[4]
        );
        status.certificates1 = float4(
            scalars[5],
            isfinite(newtonDecrement)
                ? newtonDecrement
                : 0.0f,
            isfinite(objectiveChange)
                ? objectiveChange
                : 0.0f,
            scalars[0]
        );
        status.numerics = float4(
            isfinite(minimumBlockRegularization)
                ? minimumBlockRegularization
                : 0.0f,
            regularizationRetries == 0u
            ? 0.0f
            : dispatch.numerics.w *
                dispatch.numerics.y,
            isfinite(minimumPivot)
                ? minimumPivot
                : 0.0f,
            isfinite(maximumPivot)
                ? maximumPivot
                : 0.0f
        );
        statuses[problem] = status;
    }
}

struct MRUnifiedQualityThreadgroupWorkspace {
    float velocity[kMaxV];
    float candidateVelocity[kMaxV];
    float direction[kMaxV];
    float gradient[kMaxV];
    // State evaluation, PCG, and line search never consume these phases
    // concurrently. Reusing the same Wave32-owned storage keeps the 384-DoF
    // articulated/body/rod bucket below Apple GPU threadgroup-memory limits.
    float dynamicsAction[kMaxV];
    float constraintAction[kMaxV];
    float cgDirection[kMaxV];
    float cgAction[kMaxV];
    float preconditioner[kMaxV];
    float rowVelocity[kMaxRows];
    float impulses[kMaxRows];
    float rowWork[kMaxRows];
    float rowAction[kMaxRows];
    float scalars[16];
    float candidateScalars[16];
    uint control[16];
};

inline void mrUnifiedQualitySolveWithWorkspace(
    device const MRUnifiedQualityDispatchGPU& dispatch,
    device const MRUnifiedQualityBlockGPU* blocks,
    device const float* dynamicsMatrices,
    device const float* jacobianMatrices,
    device const float* biasVectors,
    device const float* freeVelocities,
    device const float* warmVelocities,
    device const float* warmImpulses,
    device float* outputVelocities,
    device float* outputImpulses,
    device float* derivativeScratch,
    device float* hessianScratch,
    device MRUnifiedQualityStatusGPU* statuses,
    const uint problem,
    const uint lane,
    threadgroup MRUnifiedQualityThreadgroupWorkspace& workspace
) {
    mrUnifiedQualitySolveProblem(
        dispatch,
        blocks,
        dynamicsMatrices,
        jacobianMatrices,
        biasVectors,
        freeVelocities,
        warmVelocities,
        warmImpulses,
        outputVelocities,
        outputImpulses,
        derivativeScratch,
        hessianScratch,
        statuses,
        problem,
        lane,
        workspace.velocity,
        workspace.candidateVelocity,
        workspace.direction,
        workspace.gradient,
        workspace.cgDirection,
        workspace.dynamicsAction,
        workspace.cgAction,
        workspace.constraintAction,
        workspace.preconditioner,
        workspace.dynamicsAction,
        workspace.constraintAction,
        workspace.cgDirection,
        workspace.cgAction,
        workspace.preconditioner,
        workspace.rowVelocity,
        workspace.rowWork,
        workspace.impulses,
        workspace.rowAction,
        workspace.rowWork,
        workspace.rowAction,
        workspace.scalars,
        workspace.candidateScalars,
        workspace.control
    );
}

kernel void mr_unified_quality_solve(
    device const MRUnifiedQualityDispatchGPU& dispatch
        [[buffer(0)]],
    device const MRUnifiedQualityBlockGPU* blocks
        [[buffer(1)]],
    device const float* dynamicsMatrices [[buffer(2)]],
    device const float* jacobianMatrices [[buffer(3)]],
    device const float* biasVectors [[buffer(4)]],
    device const float* freeVelocities [[buffer(5)]],
    device const float* warmVelocities [[buffer(6)]],
    device const float* warmImpulses [[buffer(7)]],
    device float* outputVelocities [[buffer(8)]],
    device float* outputImpulses [[buffer(9)]],
    device float* derivativeScratch [[buffer(10)]],
    device float* hessianScratch [[buffer(11)]],
    device MRUnifiedQualityStatusGPU* statuses [[buffer(12)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup MRUnifiedQualityThreadgroupWorkspace workspace;
    mrUnifiedQualitySolveWithWorkspace(
        dispatch,
        blocks,
        dynamicsMatrices,
        jacobianMatrices,
        biasVectors,
        freeVelocities,
        warmVelocities,
        warmImpulses,
        outputVelocities,
        outputImpulses,
        derivativeScratch,
        hessianScratch,
        statuses,
        problem,
        lane,
        workspace
    );
}

// Standalone Metal consumes the same stable packet stream through
// device-generated indirect dispatch. The dispatch slot maps through the
// immutable packet instead of assuming a dense environment index.
kernel void mr_unified_quality_solve_queued(
    device const MRUnifiedQualityDispatchGPU& dispatch
        [[buffer(0)]],
    device const MRUnifiedQualityBlockGPU* blocks
        [[buffer(1)]],
    device const float* dynamicsMatrices [[buffer(2)]],
    device const float* jacobianMatrices [[buffer(3)]],
    device const float* biasVectors [[buffer(4)]],
    device const float* freeVelocities [[buffer(5)]],
    device const float* warmVelocities [[buffer(6)]],
    device const float* warmImpulses [[buffer(7)]],
    device float* outputVelocities [[buffer(8)]],
    device float* outputImpulses [[buffer(9)]],
    device float* derivativeScratch [[buffer(10)]],
    device float* hessianScratch [[buffer(11)]],
    device MRUnifiedQualityStatusGPU* statuses [[buffer(12)]],
    device const MRUnifiedQualityWorkPacketGPU* packets
        [[buffer(13)]],
    device MRUnifiedQualityWorkQueueGPU& queue [[buffer(14)]],
    const uint packetSlot [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]]
) {
    if (packetSlot >= queue.count) {
        return;
    }
    threadgroup MRUnifiedQualityThreadgroupWorkspace workspace;
    const uint problem = packets[packetSlot].identity.x;
    mrUnifiedQualitySolveWithWorkspace(
        dispatch,
        blocks,
        dynamicsMatrices,
        jacobianMatrices,
        biasVectors,
        freeVelocities,
        warmVelocities,
        warmImpulses,
        outputVelocities,
        outputImpulses,
        derivativeScratch,
        hessianScratch,
        statuses,
        problem,
        lane,
        workspace
    );
    if (lane == 0u) {
        device atomic_uint* processed =
            reinterpret_cast<device atomic_uint*>(
                &queue.packetsProcessed
            );
        atomic_fetch_add_explicit(
            processed,
            1u,
            memory_order_relaxed
        );
    }
}

// MLX's active encoder cannot issue device-generated indirect dispatch. A
// fixed occupancy-sized SIMD32 grid therefore pulls stable packets from the
// invocation-local queue. Atomic claim order changes only which worker owns a
// problem; every problem retains disjoint state and canonical row reductions.
kernel void mr_unified_quality_solve_persistent(
    device const MRUnifiedQualityDispatchGPU& dispatch
        [[buffer(0)]],
    device const MRUnifiedQualityBlockGPU* blocks
        [[buffer(1)]],
    device const float* dynamicsMatrices [[buffer(2)]],
    device const float* jacobianMatrices [[buffer(3)]],
    device const float* biasVectors [[buffer(4)]],
    device const float* freeVelocities [[buffer(5)]],
    device const float* warmVelocities [[buffer(6)]],
    device const float* warmImpulses [[buffer(7)]],
    device float* outputVelocities [[buffer(8)]],
    device float* outputImpulses [[buffer(9)]],
    device float* derivativeScratch [[buffer(10)]],
    device float* hessianScratch [[buffer(11)]],
    device MRUnifiedQualityStatusGPU* statuses [[buffer(12)]],
    device MRUnifiedQualityWorkQueueGPU& queue [[buffer(13)]],
    device const MRUnifiedQualityWorkPacketGPU* packets
        [[buffer(14)]],
    const uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup MRUnifiedQualityThreadgroupWorkspace workspace;
    threadgroup uint claimedPacket;
    device atomic_uint* cursor =
        reinterpret_cast<device atomic_uint*>(&queue.cursor);
    device atomic_uint* processed =
        reinterpret_cast<device atomic_uint*>(
            &queue.packetsProcessed
        );
    device atomic_uint* emptyPulls =
        reinterpret_cast<device atomic_uint*>(&queue.emptyPulls);
    while (true) {
        if (lane == 0u) {
            claimedPacket = atomic_fetch_add_explicit(
                cursor,
                1u,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (claimedPacket >= queue.count) {
            if (lane == 0u) {
                atomic_fetch_add_explicit(
                    emptyPulls,
                    1u,
                    memory_order_relaxed
                );
            }
            break;
        }
        const uint problem =
            packets[claimedPacket].identity.x;
        mrUnifiedQualitySolveWithWorkspace(
            dispatch,
            blocks,
            dynamicsMatrices,
            jacobianMatrices,
            biasVectors,
            freeVelocities,
            warmVelocities,
            warmImpulses,
            outputVelocities,
            outputImpulses,
            derivativeScratch,
            hessianScratch,
            statuses,
            problem,
            lane,
            workspace
        );
        threadgroup_barrier(
            mem_flags::mem_threadgroup |
            mem_flags::mem_device
        );
        if (lane == 0u) {
            atomic_fetch_add_explicit(
                processed,
                1u,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
