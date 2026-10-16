#pragma once

#include "numi/matter/matter.hpp"

#include <cmath>
#include <cstdint>
#include <span>

namespace numi::matter::detail {

// The compiler always reserves one field-state row per FEM node, including
// purely mechanical worlds. Restart augmentation admits that required arena
// only when every row is finite and explicitly marks multiphysics inactive.
// This is a host-side eligibility predicate; it does not edit or normalize the
// supplied state.
[[nodiscard]] inline bool regionalRestartFieldsAreMechanical(
    const std::span<const NMContinuumObjectGPU> objects,
    const std::span<const NMFEMFieldStateGPU> fields,
    const std::uint32_t femNodeCount
) noexcept {
    if (femNodeCount == 0u || fields.size() != femNodeCount) return false;

    for (const NMContinuumObjectGPU& object : objects) {
        if ((object.flags & NM_OBJECT_MULTIPHYSICS) != 0u) return false;
    }

    for (const NMFEMFieldStateGPU& field : fields) {
        if (!std::isfinite(field.primary.x) ||
            !std::isfinite(field.primary.y) ||
            !std::isfinite(field.primary.z) ||
            !std::isfinite(field.primary.w) ||
            !std::isfinite(field.secondary.x) ||
            !std::isfinite(field.secondary.y) ||
            !std::isfinite(field.secondary.z) ||
            !std::isfinite(field.secondary.w) ||
            field.secondary.w != 0.0f) {
            return false;
        }
    }
    return true;
}

} // namespace numi::matter::detail
