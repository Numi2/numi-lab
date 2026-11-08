#pragma once
#include "metalrobo/rod_gpu_shared.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

struct SutureRodWitnessAudit {
    bool capacity = false;
    bool identity = true;
    bool finiteImpulses = true;
    std::size_t valid = 0u;
    std::size_t warmStarted = 0u;
    std::size_t nonzeroImpulse = 0u;
};

inline SutureRodWitnessAudit auditSutureRodWitnesses(
    std::span<const MRRodToolWitnessGPU> witnesses,
    std::span<const MRRodToolPairGPU> pairs,
    std::span<const MRRodColliderGPU> colliders
) {
    SutureRodWitnessAudit a;
    if (pairs.size() > std::numeric_limits<std::size_t>::max() /
            MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR ||
        witnesses.size() != pairs.size() * MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR) {
        return a;
    }
    a.capacity = true;
    for (std::size_t slot = 0u; slot < witnesses.size(); ++slot) {
        const auto& w = witnesses[slot];
        if ((w.featuresAndFlags.w & MR_ROD_TOOL_WITNESS_VALID) == 0u) continue;
        ++a.valid;
        a.warmStarted += (w.featuresAndFlags.w & MR_ROD_TOOL_WITNESS_WARM_STARTED) != 0u;
        const std::size_t index = slot / MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
        const auto& pair = pairs[index];
        a.identity = a.identity && pair.rodCollider < colliders.size() &&
            w.identity.x == 0u && w.identity.y == index &&
            w.identity.z == pair.rodCollider && w.identity.w == pair.rigidCollider;
        if (pair.rodCollider < colliders.size()) {
            a.identity = a.identity &&
                w.materialAndGeneration.z == colliders[pair.rodCollider].topologyGeneration;
        }
        const auto impulse = w.impulses;
        a.finiteImpulses = a.finiteImpulses &&
            std::isfinite(impulse.x) && std::isfinite(impulse.y) &&
            std::isfinite(impulse.z) && std::isfinite(impulse.w);
        a.nonzeroImpulse += impulse.x != 0.0f || impulse.y != 0.0f ||
            impulse.z != 0.0f || impulse.w != 0.0f;
    }
    return a;
}
