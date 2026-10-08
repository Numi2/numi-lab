#pragma once
#include <cstdint>
#include <set>

// Receipt-validated retired source identities remain in the visual pack for
// provenance, but must never acquire a selectable inspection-layer bit.
inline std::uint32_t numiHumanRestingInspectionLayerMask(
    std::uint32_t semantic,
    std::uint32_t stableId,
    std::uint32_t sourceLayerMask,
    std::uint32_t retiredSemantic,
    const std::set<unsigned>& retiredInspectionStableIDs) noexcept {
    const bool retiredIdentity = semantic == retiredSemantic &&
        retiredInspectionStableIDs.count(stableId) != 0u;
    return retiredIdentity ? 0u : sourceLayerMask;
}
