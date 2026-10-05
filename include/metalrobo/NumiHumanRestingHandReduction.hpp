#pragma once

#include "metalrobo/NumiHumanJointEquality.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace metalrobo {

// Explicit mechanical resolution choice for the source reference adult. The
// forty digit coordinates become internal bilateral constraints at qpos0.
// Wrist/root/arm coordinates, all body inertias and MyoSim/tendon state remain
// owned by the existing articulated solver. This is a rigid-hand approximation,
// not a passive-tissue calibration or a world-space pose servo.
inline bool compileNumiHumanRestingHandReduction(
    const NumiHumanJointEqualityPayload& source,
    const std::span<const float> referenceQ,
    const std::span<const MRDofPropertiesGPU> dofs,
    std::vector<MRNumiHumanJointEqualityGPU>& output,
    std::string& error
) {
    const auto fail = [&error](const char* message) {
        error = message;
        return false;
    };
    constexpr std::array<std::uint8_t, 32> referenceArchive{
        0x28,0x0d,0x29,0x7a,0xa4,0x96,0xac,0xcc,
        0xf3,0xf1,0xc5,0x37,0x3a,0x13,0x04,0xd2,
        0x3f,0x95,0x69,0x36,0x2c,0x2d,0x69,0x60,
        0x91,0x01,0x28,0xbf,0xba,0x14,0x49,0x75};
    if (source.sourceSha256 != referenceArchive || source.nq != 129u ||
        source.nv != 128u || referenceQ.size() != 129u || dofs.size() != 128u ||
        source.records.size() != 51u)
        return fail("rigid hands require the unchanged reference MyoSim core and 51 source equalities");
    const auto selected = [](std::uint32_t q) {
        return (q >= 43u && q <= 62u) || (q >= 81u && q <= 100u);
    };
    for (const auto& row : source.records) {
        if (selected(row.indices.x) || selected(row.indices.z))
            return fail("rigid-hand coordinates overlap a source equality dependency");
    }
    auto candidate = source.records;
    for (const std::uint32_t begin : {43u, 81u}) {
        for (std::uint32_t q = begin; q < begin + 20u; ++q) {
            const std::uint32_t v = q - 1u;
            const auto& dof = dofs[v];
            const float target = referenceQ[q];
            if (dof.qIndex != q || dof.vIndex != v ||
                dof.articulationIndex != 0u || !std::isfinite(target) ||
                (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
                !std::isfinite(dof.limits.x) || !std::isfinite(dof.limits.y) ||
                target < dof.limits.x || target > dof.limits.y)
                return fail("rigid-hand reference coordinate is invalid or outside source limits");
            MRNumiHumanJointEqualityGPU row{};
            row.indices = {q, v, MR_INVALID_INDEX, MR_INVALID_INDEX};
            row.referencesAndCoefficients0.x = target;
            // No measured source compliance is claimed for this ideal reduction.
            candidate.push_back(row);
        }
    }
    output = std::move(candidate);
    error.clear();
    return true;
}

} // namespace metalrobo
