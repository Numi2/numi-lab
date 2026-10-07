#pragma once

#include "metalrobo/NumiHumanJointEquality.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

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


// Receipt actions are stable values so the derived model identity is
// deterministic across runs and independent of host bool/padding layout.
enum class NumiHumanRestingFixedBoundAction : std::uint32_t {
    eliminated = 1u,
    retainedOutsideSourceInterval = 2u,
};
struct NumiHumanRestingFixedBoundReceiptRecord {
    std::uint32_t vIndex = MR_INVALID_INDEX;
    std::uint32_t qIndex = MR_INVALID_INDEX;
    float target = 0.0f;
    float sourceLower = 0.0f;
    float sourceUpper = 0.0f;
    std::uint32_t originalFlags = 0u;
    std::uint32_t derivedFlags = 0u;
    std::uint32_t action = 0u;
};
static_assert(sizeof(NumiHumanRestingFixedBoundReceiptRecord) == 32u);
static_assert(offsetof(NumiHumanRestingFixedBoundReceiptRecord, target) == 8u);
static_assert(offsetof(NumiHumanRestingFixedBoundReceiptRecord, action) == 28u);
// Clear only source POSITION_LIMIT flags whose scalar coordinate is exactly
// fixed by a fixed-master row in the verified rigid-hand equality program.
// Source intervals and source payload stay untouched. Coupled rows do not
// qualify; finite-sweep trajectories may change as redundant rows are removed.
inline bool compileNumiHumanRestingFixedBoundElimination(
    const NumiHumanJointEqualityPayload& source,
    const std::span<const float> referenceQ,
    const std::span<const MRDofPropertiesGPU> sourceDofs,
    const std::span<const MRNumiHumanJointEqualityGPU> compiledEqualities,
    std::vector<MRDofPropertiesGPU>& derivedDofs,
    std::vector<NumiHumanRestingFixedBoundReceiptRecord>& receipt,
    std::string& error
) {
    const auto fail = [&error](const char* message) { error = message; return false; };
    std::vector<MRNumiHumanJointEqualityGPU> expectedEqualities;
    std::string reductionError;
    if (!compileNumiHumanRestingHandReduction(
            source, referenceQ, sourceDofs, expectedEqualities, reductionError))
        return fail("fixed-bound elimination requires the admitted rigid-hand reduction");
    if (compiledEqualities.size() != expectedEqualities.size() ||
        std::memcmp(compiledEqualities.data(), expectedEqualities.data(),
                    expectedEqualities.size() * sizeof(MRNumiHumanJointEqualityGPU)) != 0)
        return fail("fixed-bound elimination equality program differs from the admitted rigid-hand reduction");

    std::vector<MRDofPropertiesGPU> candidateDofs(sourceDofs.begin(), sourceDofs.end());
    std::vector<NumiHumanRestingFixedBoundReceiptRecord> candidateReceipt;
    std::vector<std::uint8_t> seenFixedV(sourceDofs.size(), 0u);
    for (const MRNumiHumanJointEqualityGPU& equality : compiledEqualities) {
        const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
            equality.indices.w == MR_INVALID_INDEX;
        const bool coupled = equality.indices.z < source.nq &&
            equality.indices.w < source.nv &&
            equality.indices.x != equality.indices.z &&
            equality.indices.y != equality.indices.w;
        if (equality.indices.x >= source.nq ||
            equality.indices.y >= source.nv || (!fixed && !coupled))
            return fail("fixed-bound elimination encountered malformed equality indices");
        if (!fixed) continue;
        const std::uint32_t q = equality.indices.x;
        const std::uint32_t v = equality.indices.y;
        const MRDofPropertiesGPU& dof = sourceDofs[v];
        if (dof.qIndex != q || dof.vIndex != v || dof.articulationIndex != 0u)
            return fail("fixed equality q/v indices do not match the source scalar DoF");
        if (seenFixedV[v] != 0u)
            return fail("fixed-bound elimination encountered duplicate fixed coordinates");
        seenFixedV[v] = 1u;
        // Mirrors evaluateJointEquality's fixed-master FP32 target: delta is
        // zero, so polynomial is referencesAndCoefficients0.z.
        const float target = equality.referencesAndCoefficients0.x +
            equality.referencesAndCoefficients0.z;
        if (!std::isfinite(target))
            return fail("fixed equality target is nonfinite");
        if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u) continue;
        if (!std::isfinite(dof.limits.x) || !std::isfinite(dof.limits.y) ||
            dof.limits.x >= dof.limits.y)
            return fail("fixed equality source position interval is invalid");
        const bool inside = target >= dof.limits.x && target <= dof.limits.y;
        const std::uint32_t derivedFlags = inside
            ? (dof.flags & ~MR_DOF_FLAG_POSITION_LIMIT) : dof.flags;
        candidateDofs[v].flags = derivedFlags;
        if (inside) {
            // EngineModel requires inactive position bounds to be zero.
            // Preserve the authored interval in the receipt and clear only
            // its derived-model representation alongside the flag.
            candidateDofs[v].limits.x = 0.0f;
            candidateDofs[v].limits.y = 0.0f;
        }
        candidateReceipt.push_back({
            .vIndex = v, .qIndex = q, .target = target,
            .sourceLower = dof.limits.x, .sourceUpper = dof.limits.y,
            .originalFlags = dof.flags, .derivedFlags = derivedFlags,
            .action = static_cast<std::uint32_t>(inside
                ? NumiHumanRestingFixedBoundAction::eliminated
                : NumiHumanRestingFixedBoundAction::retainedOutsideSourceInterval),
        });
    }
    std::sort(candidateReceipt.begin(), candidateReceipt.end(),
        [](const auto& a, const auto& b) { return a.vIndex < b.vIndex; });
    derivedDofs = std::move(candidateDofs);
    receipt = std::move(candidateReceipt);
    error.clear();
    return true;
}

} // namespace metalrobo
