#pragma once

// Scalar policy shared by the Metal standing owner and portable regressions.
// Callers validate finite inputs, positive dt/response, ordered position bounds,
// and stabilization in [0,1]. No mass, force, or timestep is hidden here.

inline float mrNumiHumanContactVelocityTarget(
    const float gap, const float timestep, const float stabilization
) {
    // A separated speculative witness may approach the plane. Requiring zero
    // normal velocity here would support the body before physical contact.
    return gap >= 0.0f ? -gap / timestep : -stabilization * gap / timestep;
}

inline float mrNumiHumanSupportSeedImpulse(
    const float force, const float timestep, const float gap, const float slop
) {
    // A seed is only a warm start. The constraint solver must be able to undo
    // all of it, and friction must use the solved TOTAL normal impulse.
    return gap <= slop ? force * timestep : 0.0f;
}

inline float mrNumiHumanLowerLimitVelocityTarget(
    const float position, const float lower, const float timestep
) {
    const float gap = position - lower;
    if (gap >= 0.0f) return -gap / timestep;
    const float correction = -0.2f * gap / timestep;
    return correction < 4.0f ? correction : 4.0f;
}

inline float mrNumiHumanUpperLimitVelocityTarget(
    const float position, const float upper, const float timestep
) {
    return -mrNumiHumanLowerLimitVelocityTarget(-position, -upper, timestep);
}

inline float mrNumiHumanProjectIntervalImpulse(
    const float accumulated, const float velocity,
    const float lowerVelocity, const float upperVelocity,
    const float response
) {
    // Exact scalar dual-coordinate minimization for a velocity interval.
    // Project the accumulated impulse, NOT the incremental correction. Later
    // contact/equality updates may release a previously loaded source stop.
    const float lowerCandidate = accumulated +
        (lowerVelocity - velocity) / response;
    const float upperCandidate = accumulated +
        (upperVelocity - velocity) / response;
    if (lowerCandidate > 0.0f) return lowerCandidate;
    if (upperCandidate < 0.0f) return upperCandidate;
    return 0.0f;
}
