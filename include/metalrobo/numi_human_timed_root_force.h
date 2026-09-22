#pragma once

#include "metalrobo/gpu_types.h"

#if !defined(__METAL_VERSION__)
#include <cmath>
#endif

// One episode-bound, open-loop world-space force at the floating-root origin.
// It changes no muscle excitation, assistance gain or constraint reaction.
// Its fixed half-open step window is evaluated against the authoritative
// accepted horizon; it has no separate clock that advances on rejected work.
typedef struct MR_ALIGN16 MRNumiHumanTimedRootForceGPU {
    // x = first step, y = duration in steps, z/w reserved zero.
    mr_uint4 stepWindow;
    // xyz = force in newtons, w reserved zero. No additional root torque.
    mr_float4 forceNewtons;
} MRNumiHumanTimedRootForceGPU;

inline bool mrNumiHumanTimedRootForceConfigured(
    const MRNumiHumanTimedRootForceGPU program
) {
    return program.stepWindow.x != 0u || program.stepWindow.y != 0u ||
        program.stepWindow.z != 0u || program.stepWindow.w != 0u ||
        program.forceNewtons.x != 0.0f || program.forceNewtons.y != 0.0f ||
        program.forceNewtons.z != 0.0f || program.forceNewtons.w != 0.0f;
}

inline bool mrNumiHumanTimedRootForceValid(
    const MRNumiHumanTimedRootForceGPU program,
    const mr_u32 authoritativeStepCount
) {
#if defined(__METAL_VERSION__)
    const bool finiteForce = metal::all(metal::isfinite(program.forceNewtons));
#else
    const bool finiteForce = std::isfinite(program.forceNewtons.x) &&
        std::isfinite(program.forceNewtons.y) &&
        std::isfinite(program.forceNewtons.z) &&
        std::isfinite(program.forceNewtons.w);
#endif
    if (!finiteForce || program.stepWindow.z != 0u ||
        program.stepWindow.w != 0u || program.forceNewtons.w != 0.0f) {
        return false;
    }
    if (program.stepWindow.y == 0u) {
        return !mrNumiHumanTimedRootForceConfigured(program);
    }
    // Subtraction after the range check avoids unsigned window-end overflow.
    return program.stepWindow.x < authoritativeStepCount &&
        program.stepWindow.y <= authoritativeStepCount - program.stepWindow.x;
}

inline bool mrNumiHumanTimedRootForceActive(
    const MRNumiHumanTimedRootForceGPU program,
    const mr_u32 authoritativeStepIndex
) {
    return authoritativeStepIndex >= program.stepWindow.x &&
        authoritativeStepIndex - program.stepWindow.x < program.stepWindow.y;
}

#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRNumiHumanTimedRootForceGPU) == 32);
#endif
