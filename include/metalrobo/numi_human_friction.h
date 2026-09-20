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

inline MRNumiHumanFrictionImpulse mrNumiHumanSolveFrictionDisk(
    float a, float b, float d, float rhsX, float rhsY, float radius
) {
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
    for (unsigned iteration = 0u; iteration < 40u; ++iteration) {
        const float middle = 0.5f * lower + 0.5f * upper;
        if (middle == lower || middle == upper) break;
        const auto candidate = mrNumiHumanFrictionShiftedSolve(a, b, d, gx, gy, middle);
        if (!candidate.valid || mrNumiHumanFrictionNorm(candidate.x, candidate.y) > radius) {
            lower = middle;
        } else {
            upper = middle;
            accepted = candidate;
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
