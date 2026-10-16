#pragma once

#include "metalrobo/numi_human_stand_gpu.h"

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

// C2 unilateral toe law shared by the CPU reference and Metal owner.
// The positive-part branch is exactly zero, with zero first and second
// derivatives, on the slack side.
inline float mrNumiHumanHipCapsuleGap(const MRNumiHumanHipCapsuleTermGPU term,
                                      const float q0, const float q1) {
    return term.coordinate0 * q0 + term.coordinate1 * q1 - term.threshold;
}

inline float mrNumiHumanHipCapsulePotential(
    const MRNumiHumanHipCapsuleTermGPU term, const float gap
) {
    if (!(gap > 0.0f)) return 0.0f;
    const float square = gap * gap;
    return (term.toeQuadratic * square * gap) * (1.0f / 3.0f) +
        (term.toeCubic * square * square) * 0.25f;
}

inline float mrNumiHumanHipCapsuleRestoringMagnitude(
    const MRNumiHumanHipCapsuleTermGPU term, const float gap
) {
    if (!(gap > 0.0f)) return 0.0f;
    const float square = gap * gap;
    return term.toeQuadratic * square + term.toeCubic * square * gap;
}

inline float mrNumiHumanHipCapsuleTangent(
    const MRNumiHumanHipCapsuleTermGPU term, const float gap
) {
    if (!(gap > 0.0f)) return 0.0f;
    const float square = gap * gap;
    return 2.0f * term.toeQuadratic * gap +
        3.0f * term.toeCubic * square;
}

inline float mrNumiHumanHipCapsuleCoefficient(
    const MRNumiHumanHipCapsuleTermGPU term, const unsigned dof
) {
    return dof == term.dofIndex0 ? term.coordinate0 :
        dof == term.dofIndex1 ? term.coordinate1 : 0.0f;
}

inline float mrNumiHumanHipCapsuleImplicitBias(
    const MRNumiHumanHipCapsuleTermGPU term, const float gap,
    const float generalizedVelocity, const float timestep,
    const unsigned rowDof
) {
    return mrNumiHumanHipCapsuleCoefficient(term, rowDof) *
        (mrNumiHumanHipCapsuleRestoringMagnitude(term, gap) +
         timestep * mrNumiHumanHipCapsuleTangent(term, gap) *
             generalizedVelocity);
}

inline float mrNumiHumanHipCapsuleEffectiveInertia(
    const MRNumiHumanHipCapsuleTermGPU term, const float gap,
    const float timestep, const unsigned rowDof, const unsigned columnDof
) {
    return (timestep * timestep) *
        mrNumiHumanHipCapsuleTangent(term, gap) *
        mrNumiHumanHipCapsuleCoefficient(term, rowDof) *
        mrNumiHumanHipCapsuleCoefficient(term, columnDof);
}
