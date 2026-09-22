#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cmath>
#endif

namespace numi_matter_contact {

// The existing final-certificate tolerance, shared with Newton stopping.
// Negative iterates remain legal during the solve, never at publication
// beyond this tolerance. This predicate does not project or alter impulses.
inline bool normalImpulseFeasible(const float normalImpulse) {
#ifdef __METAL_VERSION__
    return metal::isfinite(normalImpulse) && normalImpulse >= -1.0e-7f;
#else
    return std::isfinite(normalImpulse) && normalImpulse >= -1.0e-7f;
#endif
}

} // namespace numi_matter_contact
