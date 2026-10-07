#pragma once

#include "metalrobo/numi_human_constraint_projection.h"

// Conditional maximum-dissipation solve in the existing contact sweep:
// min 0.5*p^T W*p - rhs^T*p, ||p|| <= mu*normalImpulse, W symmetric SPD.
// A radial clip of W^-1 rhs is only correct for an isotropic tangent response.
// This does not solve or change the unilateral normal contact law.
struct MRNumiHumanFrictionImpulse {
    float x;
    float y;
    bool valid;
};

struct MRNumiHumanPreparedFrictionMetric {
    float a;
    float b;
    float d;
    float scale;
    float determinant;
    bool valid;
};

inline bool mrNumiHumanFrictionFinite(float value) {
#if defined(__METAL_VERSION__)
    return metal::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

inline float mrNumiHumanFrictionNorm(float x, float y) {
    const float ax = x < 0.0f ? -x : x;
    const float ay = y < 0.0f ? -y : y;
    const float scale = ax > ay ? ax : ay;
    if (scale == 0.0f) return 0.0f;
    const float sx = x / scale, sy = y / scale;
#if defined(__METAL_VERSION__)
    return scale * metal::sqrt(sx * sx + sy * sy);
#else
    return scale * std::sqrt(sx * sx + sy * sy);
#endif
}

// The contact response metric stays fixed during all coupled sweeps. Prepare
// its normalization and determinant once; fall back to the general solver
// whenever the unconstrained friction impulse reaches the disk boundary.
inline MRNumiHumanPreparedFrictionMetric mrNumiHumanPrepareFrictionMetric(
    float a, float b, float d
) {
    if (!mrNumiHumanFrictionFinite(a) || !mrNumiHumanFrictionFinite(b) ||
        !mrNumiHumanFrictionFinite(d) || !(a > 0.0f) || !(d > 0.0f))
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false};
    const float scale = a > d ? a : d;
    a /= scale; b /= scale; d /= scale;
    const float determinant = mrNumiHumanFusedMultiplyAdd(a, d, -b * b);
    return {a, b, d, scale, determinant, determinant > 0.0f};
}

inline MRNumiHumanFrictionImpulse mrNumiHumanTryInteriorFrictionDisk(
    const MRNumiHumanPreparedFrictionMetric metric,
    float rhsX, float rhsY, float radius
) {
    const MRNumiHumanFrictionImpulse unavailable{0.0f, 0.0f, false};
    if (!metric.valid || !mrNumiHumanFrictionFinite(rhsX) ||
        !mrNumiHumanFrictionFinite(rhsY) ||
        !mrNumiHumanFrictionFinite(radius) || !(radius > 0.0f))
        return unavailable;
    const float gx = rhsX / metric.scale, gy = rhsY / metric.scale;
    if (!mrNumiHumanFrictionFinite(gx) || !mrNumiHumanFrictionFinite(gy))
        return unavailable;
    const float x = mrNumiHumanFusedMultiplyAdd(
        metric.d, gx, -metric.b * gy) / metric.determinant;
    const float y = mrNumiHumanFusedMultiplyAdd(
        metric.a, gy, -metric.b * gx) / metric.determinant;
    if (!mrNumiHumanFrictionFinite(x) || !mrNumiHumanFrictionFinite(y) ||
        mrNumiHumanFrictionNorm(x, y) > radius)
        return unavailable;
    return {x, y, true};
}

inline MRNumiHumanFrictionImpulse mrNumiHumanFrictionShiftedSolve(
    float a, float b, float d, float gx, float gy, float shift
) {
    // Scaling avoids squaring a large dual shift in the determinant.
    const float scale = shift > 1.0f ? shift : 1.0f;
    a = a / scale + shift / scale;
    b /= scale;
    d = d / scale + shift / scale;
    gx /= scale;
    gy /= scale;
    const float determinant = mrNumiHumanFusedMultiplyAdd(a, d, -b * b);
    if (!(determinant > 0.0f)) return {0.0f, 0.0f, false};
    const float x = mrNumiHumanFusedMultiplyAdd(d, gx, -b * gy) / determinant;
    const float y = mrNumiHumanFusedMultiplyAdd(a, gy, -b * gx) / determinant;
    return {x, y, mrNumiHumanFrictionFinite(x) && mrNumiHumanFrictionFinite(y)};
}

template<bool CollectIterations>
inline MRNumiHumanFrictionImpulse mrNumiHumanSolveFrictionDiskCore(
    float a, float b, float d, float rhsX, float rhsY, float radius,
#if defined(__METAL_VERSION__)
    thread unsigned& boundaryIterations
#else
    unsigned& boundaryIterations
#endif
) {
    if constexpr (CollectIterations) boundaryIterations = 0u;
    const MRNumiHumanFrictionImpulse invalid{0.0f, 0.0f, false};
    if (!mrNumiHumanFrictionFinite(a) || !mrNumiHumanFrictionFinite(b) ||
        !mrNumiHumanFrictionFinite(d) || !mrNumiHumanFrictionFinite(rhsX) ||
        !mrNumiHumanFrictionFinite(rhsY) || !mrNumiHumanFrictionFinite(radius) ||
        !(a > 0.0f) || !(d > 0.0f) || radius < 0.0f) return invalid;
    const float scale = a > d ? a : d;
    a /= scale; b /= scale; d /= scale;
    const float determinant = mrNumiHumanFusedMultiplyAdd(a, d, -b * b);
    if (!(determinant > 0.0f)) return invalid;
    if (radius == 0.0f) return {0.0f, 0.0f, true};
    const float gx = rhsX / scale, gy = rhsY / scale;
    if (!mrNumiHumanFrictionFinite(gx) || !mrNumiHumanFrictionFinite(gy)) return invalid;
    const auto free = mrNumiHumanFrictionShiftedSolve(a, b, d, gx, gy, 0.0f);
    if (free.valid && mrNumiHumanFrictionNorm(free.x, free.y) <= radius) return free;
    // The unique boundary multiplier satisfies ||(W+sI)^-1 rhs||=radius.
    // SPD implies s=||rhs||/radius is an upper bound. No compliance is added:
    // s is the multiplier of the friction-disk constraint, not a material term.
    float upper = mrNumiHumanFrictionNorm(gx, gy) / radius;
    if (!(upper > 0.0f) || !mrNumiHumanFrictionFinite(upper)) return invalid;
    float lower = 0.0f;
    auto accepted = mrNumiHumanFrictionShiftedSolve(a, b, d, gx, gy, upper);
    if (!accepted.valid) return invalid;
    // Safeguarded Newton iteration on the reciprocal impulse norm. For
    // p=(W+sI)^-1*g and u=p/||p||, the Newton increment is
    // (||p||/radius-1)/(u^T*(W+sI)^-1*u). The SPD metric makes the
    // denominator positive. Keep the feasible upper endpoint and retain
    // bisection whenever the proposal leaves the bracket or is nonfinite.
    // This is the same Coulomb-disk optimum; no compliance is introduced.
    float trialShift = upper;
    bool normConverged = false;
    for (unsigned iteration = 0u; iteration < 8u; ++iteration) {
        if constexpr (CollectIterations) ++boundaryIterations;
        // The first Newton trial is exactly the feasible upper solve above.
        // Reuse it without changing the bracket, iteration count, or solver path.
        const auto trial = iteration == 0u ? accepted :
            mrNumiHumanFrictionShiftedSolve(a, b, d, gx, gy, trialShift);
        if (!trial.valid) break;
        const float norm = mrNumiHumanFrictionNorm(trial.x, trial.y);
        if (!(norm > 0.0f) || !mrNumiHumanFrictionFinite(norm)) break;
        if (norm <= radius) {
            upper = trialShift;
            accepted = trial;
            if (norm >= radius * (1.0f - 8.0f * 1.1920928955078125e-7f)) {
                normConverged = true;
                break;
            }
        } else {
            lower = trialShift;
        }
        const float ux = trial.x / norm, uy = trial.y / norm;
        const auto derivative = mrNumiHumanFrictionShiftedSolve(
            a, b, d, ux, uy, trialShift);
        const float slope = mrNumiHumanFusedMultiplyAdd(
            ux, derivative.x, uy * derivative.y);
        const float proposal = trialShift + (norm / radius - 1.0f) / slope;
        trialShift = derivative.valid && slope > 0.0f &&
            mrNumiHumanFrictionFinite(proposal) &&
            proposal > lower && proposal < upper
            ? proposal : 0.5f * lower + 0.5f * upper;
        if (trialShift == lower || trialShift == upper) break;
    }
    for (unsigned iteration = 0u; !normConverged && iteration < 40u; ++iteration) {
        const float middle = 0.5f * lower + 0.5f * upper;
        if (middle == lower || middle == upper) break;
        if constexpr (CollectIterations) ++boundaryIterations;
        const auto candidate = mrNumiHumanFrictionShiftedSolve(a, b, d, gx, gy, middle);
        if (!candidate.valid) {
            lower = middle;
        } else {
            const float candidateNorm = mrNumiHumanFrictionNorm(candidate.x, candidate.y);
            if (candidateNorm > radius) {
                lower = middle;
            } else {
                upper = middle;
                accepted = candidate;
                // Use the same feasible-norm convergence criterion as the
                // safeguarded-Newton phase; further bisection cannot improve the
                // already accepted impulse beyond its existing relative tolerance.
                if (candidateNorm >= radius *
                        (1.0f - 8.0f * 1.1920928955078125e-7f)) break;
            }
        }
    }
    // Correct only final FP32 norm roundoff at the feasible endpoint.
    const float norm = mrNumiHumanFrictionNorm(accepted.x, accepted.y);
    if (!mrNumiHumanFrictionFinite(norm)) return invalid;
    if (norm > radius) {
        accepted.x *= radius / norm;
        accepted.y *= radius / norm;
    }
    return accepted;
}

inline MRNumiHumanFrictionImpulse mrNumiHumanSolveFrictionDisk(
    float a, float b, float d, float rhsX, float rhsY, float radius
) {
    unsigned unusedIterations = 0u;
    return mrNumiHumanSolveFrictionDiskCore<false>(
        a, b, d, rhsX, rhsY, radius, unusedIterations);
}

inline MRNumiHumanFrictionImpulse mrNumiHumanSolveFrictionDiskCounted(
    float a, float b, float d, float rhsX, float rhsY, float radius,
#if defined(__METAL_VERSION__)
    thread unsigned& boundaryIterations
#else
    unsigned& boundaryIterations
#endif
) {
    return mrNumiHumanSolveFrictionDiskCore<true>(
        a, b, d, rhsX, rhsY, radius, boundaryIterations);
}
