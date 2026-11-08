#pragma once

#include <cstdint>

// Bound channel growth while preserving the historical contact-only phases.
// The caller separately checks contact evidence, channel identity/geometry,
// puncture evidence, tissue state, solver certificates and grasp.
constexpr bool sutureChannelCountAdmissible(
    const std::uint32_t active,
    const std::uint32_t channels,
    const std::uint32_t steps,
    const std::uint32_t substeps,
    const bool contactOnlyPhase
) {
    return active == 0u
        ? channels == 1u || (contactOnlyPhase && channels == 0u)
        : channels >= active && channels <= active + steps * substeps;
}
