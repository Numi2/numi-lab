#pragma once
#include "numi/matter/shared.h"

// Scale rounding guards for a fixed plane by the arithmetic which forms its
// signed distance, not by unrelated tangential world coordinates. The plane
// distance is dot(unitNormal, point) - offset, so this sum bounds the
// magnitudes of the actual dot products and offset subtraction.
inline float nmFixedPlaneContactArithmeticScale(
    const float normalX, const float normalY, const float normalZ,
    const float pointX, const float pointY, const float pointZ,
    const float offset, const float minimumScale) {
    const float ax = normalX < 0.0f ? -normalX : normalX;
    const float ay = normalY < 0.0f ? -normalY : normalY;
    const float az = normalZ < 0.0f ? -normalZ : normalZ;
    const float px = pointX < 0.0f ? -pointX : pointX;
    const float py = pointY < 0.0f ? -pointY : pointY;
    const float pz = pointZ < 0.0f ? -pointZ : pointZ;
    const float d = offset < 0.0f ? -offset : offset;
    const float projectedMagnitude = ax * px + ay * py + az * pz + d;
    return projectedMagnitude > minimumScale ? projectedMagnitude : minimumScale;
}

// For fixed convex geometry, positive-distance supporting planes certify
// outward affine continuum motion. Dynamic/rotating/nonconvex proxies retain
// the general Lipschitz travel bound. Unresolved normal/dot rounding also
// retains that bound. The final trial CCD and production certificate still run.
inline bool nmFixedConvexContactOutward(
    const nm_u32 flags, const nm_u32 shape, const float separation,
    const float feasibleSeparation, const float travel,
    const float directionalSeparation, const float geometryRoundoff) {
    if ((flags & (NM_RIGID_DYNAMIC | NM_RIGID_ARTICULATED |
                  NM_RIGID_SUTURE_STRAND | NM_RIGID_PUNCTURE_TIP |
                  NM_RIGID_PUNCTURE_DILATOR)) != 0u ||
        !(shape == NM_RIGID_PLANE || shape == NM_RIGID_SPHERE ||
          shape == NM_RIGID_CAPSULE || shape == NM_RIGID_BOX) ||
        !(separation > feasibleSeparation) || !(travel > 0.0f) ||
        !(geometryRoundoff >= 0.0f)) return false;
    const float normalUncertainty = 32.0f * 0x1p-23f +
        8.0f * geometryRoundoff / separation;
    if (!(normalUncertainty < 0.25f)) return false;
    const float outwardMargin = normalUncertainty * travel +
        2.0f * geometryRoundoff;
    return directionalSeparation > outwardMargin;
}
