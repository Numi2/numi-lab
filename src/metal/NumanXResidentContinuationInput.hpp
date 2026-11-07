#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"

namespace metalrobo {

// A configured continuation must name the predecessor, never reset its
// resident state from the original host image. Do not validate or repair the
// identity here: the articulated owner remains responsible for fail-closed
// identity, family, and arena validation, including partial identities.
inline void selectNumanXResidentContinuationInput(
    MetalArticulatedOperatorInput& input
) noexcept {
    if (!input.residentContinuation.configured()) return;
    input.q = {};
    input.v = {};
    input.rootTranslations = {};
    input.stand.v = {};
    input.mujoco.states = {};
}

} // namespace metalrobo
