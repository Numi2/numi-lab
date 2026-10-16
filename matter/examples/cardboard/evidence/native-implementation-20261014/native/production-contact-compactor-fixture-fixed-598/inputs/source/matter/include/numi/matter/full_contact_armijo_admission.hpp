#pragma once

#include "numi/matter/shared.h"

#include <span>

namespace numi::matter::detail {

// Full-contact residual descent may be compiled with mixed cohesive faces
// that are authored but still dormant. Eligibility is structural and must
// therefore inspect dormant faces as well as already activated ones. The
// transaction's accepted/candidate face state remains the authority for
// whether a face contributes force in a particular solve.
[[nodiscard]] inline bool fullContactArmijoSupportsCohesiveFaces(
    const std::span<const NMCohesiveFaceGPU> faces
) noexcept {
    if (faces.empty()) return false;
    constexpr std::uint32_t requiredFlags =
        NM_TOPOLOGY_ACTIVE |
        NM_TOPOLOGY_PAIRED_COHESIVE |
        NM_TOPOLOGY_COHESIVE_MIXED_MODE;
    for (const NMCohesiveFaceGPU& face : faces) {
        if ((face.adjacency.w & requiredFlags) != requiredFlags ||
            face.lawIdentity.x != NM_COHESIVE_LAW_MIXED_MODE_BILINEAR ||
            face.lawIdentity.y == NM_INVALID_INDEX)
            return false;
    }
    return true;
}

} // namespace numi::matter::detail
