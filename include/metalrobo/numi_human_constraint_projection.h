#pragma once

// Scalar policy shared by the Metal standing owner and portable regressions.
// Callers validate finite inputs, positive dt/response, ordered position bounds,
// and stabilization in [0,1]. No mass, force, or timestep is hidden here.

inline float mrNumiHumanContactVelocityTarget(
    const float gap, const float timestep, const float stabilization
) {
    // A separated speculative witness may approach the plane. Requiring zero
    // normal velocity here would support the body before physical contact.
    return gap >= 0.0f ? -gap / timestep : -stabilization * gap / timestep;
}

inline float mrNumiHumanSupportSeedImpulse(
    const float force, const float timestep, const float gap, const float slop
) {
    // A seed is only a warm start. The constraint solver must be able to undo
    // all of it, and friction must use the solved TOTAL normal impulse.
    return gap <= slop ? force * timestep : 0.0f;
}

inline float mrNumiHumanPositionLimitSlop(
    const float position, const float bound
) {
    // Source coordinates are authored in FP64 and executed in FP32. Equality
    // projection and q integration can therefore place a coordinate a few ULPs
    // beyond an otherwise coincident hard stop. Treat that representational
    // band as zero position error instead of converting it into a timestep-
    // amplified restitution-like velocity. The band is scale-aware, symmetric,
    // and materially smaller than any admitted anatomical calibration error.
    const float absolutePosition = position < 0.0f ? -position : position;
    const float absoluteBound = bound < 0.0f ? -bound : bound;
    float scale = absolutePosition > absoluteBound
        ? absolutePosition
        : absoluteBound;
    if (scale < 1.0f) scale = 1.0f;
    return 16.0f * 1.1920928955078125e-7f * scale;
}

inline float mrNumiHumanLowerLimitVelocityTarget(
    const float position, const float lower, const float timestep
) {
    const float gap = position - lower;
    if (gap >= 0.0f) return -gap / timestep;
    const float slop = mrNumiHumanPositionLimitSlop(position, lower);
    if (gap >= -slop) return 0.0f;
    const float correction = -0.2f * (gap + slop) / timestep;
    return correction < 4.0f ? correction : 4.0f;
}

inline float mrNumiHumanUpperLimitVelocityTarget(
    const float position, const float upper, const float timestep
) {
    return -mrNumiHumanLowerLimitVelocityTarget(-position, -upper, timestep);
}

inline float mrNumiHumanProjectIntervalImpulse(
    const float accumulated, const float velocity,
    const float lowerVelocity, const float upperVelocity,
    const float response
) {
    // Exact scalar dual-coordinate minimization for a velocity interval.
    // Project the accumulated impulse, NOT the incremental correction. Later
    // contact/equality updates may release a previously loaded source stop.
    const float lowerCandidate = accumulated +
        (lowerVelocity - velocity) / response;
    const float upperCandidate = accumulated +
        (upperVelocity - velocity) / response;
    if (lowerCandidate > 0.0f) return lowerCandidate;
    if (upperCandidate < 0.0f) return upperCandidate;
    return 0.0f;
}

#if defined(__METAL_VERSION__)
inline float mrNumiHumanFusedMultiplyAdd(float a, float b, float c) {
    return metal::fma(a, b, c);
}
#else
#include <cmath>
inline float mrNumiHumanFusedMultiplyAdd(float a, float b, float c) {
    return std::fma(a, b, c);
}
#endif

struct MRNumiHumanEqualityLimitBlock {
    float equalityDelta;
    float limitImpulse;
    float projectedResponse;
    bool valid;
};

// A local Schur block inside the existing coupled sweep, not a second global
// solver. The four entries are contractions of the SAME factored mass responses:
// a=E M^-1 E^T, b=E M^-1 L^T, c=L M^-1 E^T, d=L M^-1 L^T.
// Keeping b and c separate respects the actual FP32 response vectors.
inline MRNumiHumanEqualityLimitBlock mrNumiHumanProjectEqualityLimitBlock(
    float accumulatedLimit, float equalityVelocity, float limitVelocity,
    float equalityTarget, float lowerVelocity, float upperVelocity,
    float a, float b, float c, float d
) {
    MRNumiHumanEqualityLimitBlock result = {0.0f, accumulatedLimit, 0.0f, false};
    if (!(a > 0.0f) || !(d > 0.0f)) return result;
    const float ca = c / a;
    const float projected = mrNumiHumanFusedMultiplyAdd(-ca, b, d);
    result.projectedResponse = projected;
    // A rank-deficient pair has no independent limit direction. Do not
    // manufacture one with diagonal compliance; the caller retains its
    // ordinary row solve and the existing residual gates.
    if (!(projected > 1.0e-7f * d)) return result;
    const float equalityOnlyDelta = (equalityTarget - equalityVelocity) / a;
    const float projectedVelocity = mrNumiHumanFusedMultiplyAdd(
        c, equalityOnlyDelta, limitVelocity);
    result.limitImpulse = mrNumiHumanProjectIntervalImpulse(
        accumulatedLimit, projectedVelocity, lowerVelocity, upperVelocity, projected);
    const float limitDelta = result.limitImpulse - accumulatedLimit;
    result.equalityDelta = mrNumiHumanFusedMultiplyAdd(-b / a, limitDelta, equalityOnlyDelta);
    result.valid = true;
    return result;
}
