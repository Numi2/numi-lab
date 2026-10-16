#pragma once

// Shared FP32 scalar initializer for the source-authored 13-state associative
// ideal-Hill paper law. It only supplies a local Newton candidate; callers
// must run the normal constitutive residual, convergence, and transaction
// checks afterward. Parameter order is the 20-float Haj-Ali material row.

#if defined(__METAL_VERSION__)
#define NM_HILL_INIT_INLINE inline
#define NM_HILL_INIT_THREAD thread
#define NM_HILL_INIT_PARAMETER device const float*
#define NM_HILL_INIT_ABS fabs
#define NM_HILL_INIT_MAX fmax
#define NM_HILL_INIT_SQRT sqrt
#else
#include <math.h>
#define NM_HILL_INIT_INLINE inline
#define NM_HILL_INIT_THREAD
#define NM_HILL_INIT_PARAMETER const float*
#define NM_HILL_INIT_ABS fabsf
#define NM_HILL_INIT_MAX fmaxf
#define NM_HILL_INIT_SQRT sqrtf
#endif

enum NMHillScalarInitializerStatus {
    NM_HILL_SCALAR_INITIALIZER_SUCCESS = 0,
    NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT = 1,
    NM_HILL_SCALAR_INITIALIZER_SINGULAR_STRESS_SOLVE = 2,
    NM_HILL_SCALAR_INITIALIZER_NO_BRACKET = 3,
    NM_HILL_SCALAR_INITIALIZER_NONFINITE = 4
};

struct NMHillScalarInitializerEvaluation {
    bool valid;
    int status;
    float stress[6];
    float q;
    float residual;
};

NM_HILL_INIT_INLINE bool nmHillScalarInitializerFinite(float value) {
#if defined(__METAL_VERSION__)
    return isfinite(value);
#else
    return isfinite(value) != 0;
#endif
}

NM_HILL_INIT_INLINE void nmHillScalarInitializerFlowMatrix(
    NM_HILL_INIT_PARAMETER parameters, NM_HILL_INIT_THREAD float* matrix) {
    const float F = parameters[11];
    const float G = parameters[12];
    const float H = parameters[13];
    for (unsigned i = 0; i < 36; ++i) matrix[i] = 0.0f;
    matrix[0] = G + H; matrix[1] = -H; matrix[2] = -G;
    matrix[6] = -H; matrix[7] = F + H; matrix[8] = -F;
    matrix[12] = -G; matrix[13] = -F; matrix[14] = F + G;
    matrix[21] = parameters[14];
    matrix[28] = parameters[15];
    matrix[35] = parameters[16];
}

NM_HILL_INIT_INLINE void nmHillScalarInitializerElasticMatrix(
    NM_HILL_INIT_PARAMETER parameters, NM_HILL_INIT_THREAD float* matrix) {
    const float inverseReference = 1.0f / parameters[10];
    for (unsigned i = 0; i < 36; ++i) matrix[i] = 0.0f;
    matrix[0] = parameters[1] * inverseReference;
    matrix[7] = parameters[2] * inverseReference;
    matrix[14] = parameters[3] * inverseReference;
    matrix[1] = matrix[6] = parameters[4] * inverseReference;
    matrix[2] = matrix[12] = parameters[5] * inverseReference;
    matrix[8] = matrix[13] = parameters[6] * inverseReference;
    matrix[21] = 2.0f * parameters[9] * inverseReference;
    matrix[28] = 2.0f * parameters[8] * inverseReference;
    matrix[35] = 2.0f * parameters[7] * inverseReference;
}

NM_HILL_INIT_INLINE void nmHillScalarInitializerMultiply6(
    NM_HILL_INIT_THREAD const float* a, NM_HILL_INIT_THREAD const float* b,
    NM_HILL_INIT_THREAD float* result) {
    for (unsigned i = 0; i < 36; ++i) result[i] = 0.0f;
    for (unsigned i = 0; i < 6; ++i)
        for (unsigned j = 0; j < 6; ++j)
            for (unsigned k = 0; k < 6; ++k)
                result[6 * i + j] += a[6 * i + k] * b[6 * k + j];
}

NM_HILL_INIT_INLINE float nmHillScalarInitializerEquivalent(
    NM_HILL_INIT_PARAMETER parameters, NM_HILL_INIT_THREAD const float* stress) {
    const float F = parameters[11];
    const float G = parameters[12];
    const float H = parameters[13];
    const float qSquared =
        F * (stress[1] - stress[2]) * (stress[1] - stress[2]) +
        G * (stress[2] - stress[0]) * (stress[2] - stress[0]) +
        H * (stress[0] - stress[1]) * (stress[0] - stress[1]) +
        2.0f * parameters[14] * stress[3] * stress[3] +
        2.0f * parameters[15] * stress[4] * stress[4] +
        2.0f * parameters[16] * stress[5] * stress[5] +
        parameters[17] * parameters[17];
    if (!nmHillScalarInitializerFinite(qSquared))
        return -2.0f;
    if (!(qSquared >= 0.0f))
        return -1.0f;
    return NM_HILL_INIT_SQRT(qSquared);
}

NM_HILL_INIT_INLINE bool nmHillScalarInitializerSolve6(
    NM_HILL_INIT_THREAD const float* sourceMatrix,
    NM_HILL_INIT_THREAD const float* sourceRightHandSide,
    NM_HILL_INIT_THREAD float* solution) {
    float matrix[36];
    float rhs[6];
    for (unsigned i = 0; i < 36; ++i) matrix[i] = sourceMatrix[i];
    for (unsigned i = 0; i < 6; ++i) rhs[i] = sourceRightHandSide[i];
    for (unsigned k = 0; k < 6; ++k) {
        unsigned pivot = k;
        float largest = NM_HILL_INIT_ABS(matrix[6 * k + k]);
        for (unsigned i = k + 1; i < 6; ++i) {
            const float candidate = NM_HILL_INIT_ABS(matrix[6 * i + k]);
            if (candidate > largest) { largest = candidate; pivot = i; }
        }
        if (!(largest > 0.0f) || !nmHillScalarInitializerFinite(largest)) return false;
        if (pivot != k) {
            for (unsigned j = k; j < 6; ++j) {
                const float temporary = matrix[6 * k + j];
                matrix[6 * k + j] = matrix[6 * pivot + j];
                matrix[6 * pivot + j] = temporary;
            }
            const float temporary = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = temporary;
        }
        const float diagonal = matrix[6 * k + k];
        for (unsigned i = k + 1; i < 6; ++i) {
            const float factor = matrix[6 * i + k] / diagonal;
            matrix[6 * i + k] = 0.0f;
            for (unsigned j = k + 1; j < 6; ++j)
                matrix[6 * i + j] -= factor * matrix[6 * k + j];
            rhs[i] -= factor * rhs[k];
        }
    }
    for (int i = 5; i >= 0; --i) {
        float value = rhs[unsigned(i)];
        for (unsigned j = unsigned(i) + 1; j < 6; ++j)
            value -= matrix[6 * unsigned(i) + j] * solution[j];
        const float diagonal = matrix[6 * unsigned(i) + unsigned(i)];
        if (!(NM_HILL_INIT_ABS(diagonal) > 0.0f) || !nmHillScalarInitializerFinite(diagonal)) return false;
        solution[unsigned(i)] = value / diagonal;
        if (!nmHillScalarInitializerFinite(solution[unsigned(i)])) return false;
    }
    return true;
}

NM_HILL_INIT_INLINE NMHillScalarInitializerEvaluation nmHillScalarInitializerEvaluate(
    NM_HILL_INIT_PARAMETER parameters, NM_HILL_INIT_THREAD const float* trialStress,
    NM_HILL_INIT_THREAD const float* elasticTimesFlow, float t) {
    NMHillScalarInitializerEvaluation result;
    result.valid = false;
    result.status = NM_HILL_SCALAR_INITIALIZER_SINGULAR_STRESS_SOLVE;
    for (unsigned i = 0; i < 6; ++i) result.stress[i] = 0.0f;
    result.q = 0.0f;
    result.residual = 0.0f;
    float system[36];
    for (unsigned i = 0; i < 6; ++i)
        for (unsigned j = 0; j < 6; ++j)
            system[6 * i + j] = (i == j ? 1.0f : 0.0f) +
                t * elasticTimesFlow[6 * i + j];
    for (unsigned i = 0; i < 36; ++i)
        if (!nmHillScalarInitializerFinite(system[i])) {
            result.status = NM_HILL_SCALAR_INITIALIZER_NONFINITE;
            return result;
        }
    if (!nmHillScalarInitializerSolve6(system, trialStress, result.stress)) return result;
    result.q = nmHillScalarInitializerEquivalent(parameters, result.stress);
    if (result.q == -2.0f) {
        result.status = NM_HILL_SCALAR_INITIALIZER_NONFINITE;
        return result;
    }
    if (!(result.q >= 0.0f)) return result;
    const float lambda = t * result.q;
    const float gap = 1.0f - result.q;
    result.residual = lambda * gap -
        (parameters[18] * parameters[18] / 2.0f);
    if (!nmHillScalarInitializerFinite(result.residual)) {
        result.status = NM_HILL_SCALAR_INITIALIZER_NONFINITE;
        return result;
    }
    result.valid = true;
    result.status = NM_HILL_SCALAR_INITIALIZER_SUCCESS;
    return result;
}

// Parameters use the captured 20-float order: density, six normal C values,
// G12/G13/G23, sigmaReference, Hill F/G/H/L/M/N, q_floor, FB epsilon,
// residual_scale. The last entry is validated but does not affect this map.
// accepted and trialStress are thread-local in Metal and ordinary arrays in C++.
// candidate is written only on success; failed calls preserve its prior bytes.
NM_HILL_INIT_INLINE int nmHillScalarInitializeCandidate(
    NM_HILL_INIT_PARAMETER parameters,
    NM_HILL_INIT_THREAD const float* accepted,
    NM_HILL_INIT_THREAD const float* trialStress,
    NM_HILL_INIT_THREAD float* candidate) {
    for (unsigned i = 0; i < 20; ++i)
        if (!nmHillScalarInitializerFinite(parameters[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i = 0; i < 13; ++i)
        if (!nmHillScalarInitializerFinite(accepted[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    for (unsigned i = 0; i < 6; ++i)
        if (!nmHillScalarInitializerFinite(trialStress[i]))
            return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    if (!(parameters[10] > 0.0f) || !(parameters[17] > 0.0f) ||
        !(parameters[18] > 0.0f))
        return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;

    float elastic[36];
    float flow[36];
    float elasticTimesFlow[36];
    nmHillScalarInitializerElasticMatrix(parameters, elastic);
    nmHillScalarInitializerFlowMatrix(parameters, flow);
    nmHillScalarInitializerMultiply6(elastic, flow, elasticTimesFlow);
    float scale = 0.0f;
    for (unsigned i = 0; i < 36; ++i) {
        if (!nmHillScalarInitializerFinite(elasticTimesFlow[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
        scale = NM_HILL_INIT_MAX(scale, NM_HILL_INIT_ABS(elasticTimesFlow[i]));
    }
    if (!(scale > 0.0f) || !nmHillScalarInitializerFinite(scale))
        return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;

    const float q0 = nmHillScalarInitializerEquivalent(parameters, trialStress);
    if (q0 == -2.0f) return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    if (!(q0 >= 0.0f)) return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;
    const float halfEpsilonSquared = parameters[18] * parameters[18] / 2.0f;
    if (!(halfEpsilonSquared > 0.0f) ||
        !nmHillScalarInitializerFinite(halfEpsilonSquared))
        return NM_HILL_SCALAR_INITIALIZER_INVALID_INPUT;

    float high = 1.0f / scale;
    if (q0 < 1.0f) {
        const float denominator = q0 * (1.0f - q0);
        if (denominator > 0.0f)
            high = NM_HILL_INIT_MAX(high, 2.0f * halfEpsilonSquared / denominator);
    }
    if (!(high > 0.0f) || !nmHillScalarInitializerFinite(high)) high = 1.0f / scale;
    if (!nmHillScalarInitializerFinite(high)) return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    NMHillScalarInitializerEvaluation upper =
        nmHillScalarInitializerEvaluate(parameters, trialStress, elasticTimesFlow, high);
    unsigned bracketExpansions = 0;
    while (upper.valid && upper.residual < 0.0f && bracketExpansions < 100) {
        high *= 2.0f;
        if (!nmHillScalarInitializerFinite(high)) break;
        upper = nmHillScalarInitializerEvaluate(parameters, trialStress, elasticTimesFlow, high);
        ++bracketExpansions;
    }
    if (!upper.valid) return upper.status;
    if (!(upper.residual >= 0.0f)) return NM_HILL_SCALAR_INITIALIZER_NO_BRACKET;

    float low = 0.0f;
    float bestT = high;
    NMHillScalarInitializerEvaluation best = upper;
    for (unsigned iteration = 0; iteration < 80; ++iteration) {
        const float middle = low + (high - low) / 2.0f;
        if (!(middle > low) || !(middle < high)) break;
        const NMHillScalarInitializerEvaluation current =
            nmHillScalarInitializerEvaluate(parameters, trialStress, elasticTimesFlow, middle);
        if (!current.valid) return current.status;
        if (NM_HILL_INIT_ABS(current.residual) < NM_HILL_INIT_ABS(best.residual)) {
            best = current;
            bestT = middle;
        }
        if (current.residual < 0.0f) low = middle;
        else high = middle;
    }
    const NMHillScalarInitializerEvaluation atLow =
        nmHillScalarInitializerEvaluate(parameters, trialStress, elasticTimesFlow, low);
    const NMHillScalarInitializerEvaluation atHigh =
        nmHillScalarInitializerEvaluate(parameters, trialStress, elasticTimesFlow, high);
    if (!atLow.valid) return atLow.status;
    if (!atHigh.valid) return atHigh.status;
    if (NM_HILL_INIT_ABS(atLow.residual) < NM_HILL_INIT_ABS(best.residual)) {
        best = atLow; bestT = low;
    }
    if (NM_HILL_INIT_ABS(atHigh.residual) < NM_HILL_INIT_ABS(best.residual)) {
        best = atHigh; bestT = high;
    }

    float result[13];
    for (unsigned i = 0; i < 13; ++i) result[i] = accepted[i];
    const float inverseQ = 1.0f / best.q;
    for (unsigned i = 0; i < 6; ++i) {
        float flowNumerator = 0.0f;
        for (unsigned j = 0; j < 6; ++j)
            flowNumerator += flow[6 * i + j] * best.stress[j];
        const float deltaPlastic = bestT * flowNumerator;
        result[i] = accepted[i] + deltaPlastic;
    }
    for (unsigned i = 0; i < 6; ++i) result[6 + i] = best.stress[i];
    const float deltaLambda = bestT * best.q;
    result[12] = accepted[12] + deltaLambda;
    if (!nmHillScalarInitializerFinite(inverseQ))
        return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    for (unsigned i = 0; i < 13; ++i)
        if (!nmHillScalarInitializerFinite(result[i]))
            return NM_HILL_SCALAR_INITIALIZER_NONFINITE;
    for (unsigned i = 0; i < 13; ++i) candidate[i] = result[i];
    return NM_HILL_SCALAR_INITIALIZER_SUCCESS;
}

#undef NM_HILL_INIT_INLINE
#undef NM_HILL_INIT_THREAD
#undef NM_HILL_INIT_PARAMETER
#undef NM_HILL_INIT_ABS
#undef NM_HILL_INIT_MAX
#undef NM_HILL_INIT_SQRT
