#pragma once
#include "numi/matter/shared.h"

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
