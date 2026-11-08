#pragma once
#include "metalrobo/engine_types.h"
#include <span>

inline bool sutureRigidCacheTopologyMatches(
    const std::span<const MRManifoldHeaderGPU> headers,
    const std::span<const MRShapeGPU> shapes
) {
    for (const MRManifoldHeaderGPU& header : headers) {
        const auto a = header.pairAndCount[1];
        const auto b = header.pairAndCount[2];
        if (header.pairAndCount[0] != 0u || a >= shapes.size() ||
            b >= shapes.size() || header.pairAndCount[3] == 0u ||
            header.pairAndCount[3] > MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY ||
            header.generationsAndFlags[0] != shapes[a].slotGeneration ||
            header.generationsAndFlags[1] != shapes[b].slotGeneration) {
            return false;
        }
    }
    return true;
}
