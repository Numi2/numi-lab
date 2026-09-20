#pragma once

// Shared backward-Euler terms for an authored linear passive joint law.
// K is symmetric positive semidefinite and has zero floating-root rows/columns.
// The persistent owner solves for acceleration, then advances v and q:
// (M + h D + h^2 K) a = tau - bias - K(q-r) - h K v.
// Contact/equality/limit responses MUST use that same effective factor.
inline float mrNumiHumanPassiveImplicitBias(
    const float stiffness, const float displacement,
    const float velocity, const float timestep
) {
    return stiffness * (displacement + timestep * velocity);
}

inline float mrNumiHumanPassiveEffectiveInertia(
    const float stiffness, const float timestep
) {
    return (timestep * timestep) * stiffness;
}
