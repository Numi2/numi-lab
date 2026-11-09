#pragma once

#include "metalrobo/compensated_geometry_gpu.h"

// Internal arithmetic for the restricted one-axis midpoint owner. This is not
// an admission or a transaction: callers must use both words throughout the
// solve and restore both on rejection. Compile every consumer without fast
// math or implicit contraction, as required by compensated_translation_gpu.h.
struct MRSliderMidpointState {
    MRCompensatedScalar position;
    MRCompensatedScalar velocity;
};

inline MRSliderMidpointState mrSliderMidpointCandidate(
    MRSliderMidpointState source, MRCompensatedScalar velocityIncrement,
    float fullTimestep, bool endpoint
) {
    const auto midpointVelocity = mrCompensatedAdd(source.velocity, velocityIncrement);
    const float duration = endpoint ? fullTimestep : 0.5f * fullTimestep;
    const auto position = mrCompensatedAdd(source.position,
        mrCompensatedMultiply({duration, 0.0f}, midpointVelocity));
    const auto velocity = endpoint
        ? mrCompensatedAdd(midpointVelocity, velocityIncrement)
        : midpointVelocity;
    return {position, velocity};
}

inline bool mrSliderMidpointStateCanonical(MRSliderMidpointState state) {
    return mrCompensatedCanonicalScalar(state.position.high, state.position.low) &&
        mrCompensatedCanonicalScalar(state.velocity.high, state.velocity.low);
}
