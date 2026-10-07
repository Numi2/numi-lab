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

inline constexpr std::array<std::uint8_t, 32>
    kNumiHumanRestingReferenceArchiveSHA256{
        0x28,0x0d,0x29,0x7a,0xa4,0x96,0xac,0xcc,
        0xf3,0xf1,0xc5,0x37,0x3a,0x13,0x04,0xd2,
        0x3f,0x95,0x69,0x36,0x2c,0x2d,0x69,0x60,
        0x91,0x01,0x28,0xbf,0xba,0x14,0x49,0x75};
inline constexpr std::array<std::uint8_t, 32>
    kNumiHumanRestingReferenceNHEQPayloadSHA256{
        0x0e,0x3c,0x59,0x37,0x92,0xa8,0xc4,0xcc,
        0xd6,0x19,0xf1,0x99,0x9f,0x66,0x85,0xb6,
        0x8e,0x6f,0xf0,0xfd,0x5a,0xe5,0xf6,0xf0,
        0xbc,0x66,0xb6,0xb5,0x94,0xa0,0xe9,0xb8};

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
    if (source.sourceSha256 != kNumiHumanRestingReferenceArchiveSHA256 ||
        source.nq != 129u ||
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

// This opt-in source reduction compiles only the 35 known linear, bounded
// dependent coordinates in the authenticated reference NHEQ1/NHRIGID model.
// The source rows and source DoFs remain immutable; only a derived DoF table
// receives the intersected master intervals and inactive dependent bounds.
enum class NumiHumanRestingLinearBoundAction : std::uint32_t {
    consolidated = 1u,
};

struct NumiHumanRestingLinearBoundReceiptRecord {
    std::uint32_t equalityIndex = MR_INVALID_INDEX;
    std::uint32_t dependentVIndex = MR_INVALID_INDEX;
    std::uint32_t dependentQIndex = MR_INVALID_INDEX;
    std::uint32_t masterVIndex = MR_INVALID_INDEX;
    std::uint32_t masterQIndex = MR_INVALID_INDEX;
    float sourceLower = 0.0f;
    float sourceUpper = 0.0f;
    float sourceMasterLower = 0.0f;
    float sourceMasterUpper = 0.0f;
    float inverseMasterLower = 0.0f;
    float inverseMasterUpper = 0.0f;
    float derivedMasterLower = 0.0f;
    float derivedMasterUpper = 0.0f;
    std::uint32_t dependentOriginalFlags = 0u;
    std::uint32_t dependentDerivedFlags = 0u;
    std::uint32_t masterFlags = 0u;
    std::uint32_t action = static_cast<std::uint32_t>(
        NumiHumanRestingLinearBoundAction::consolidated);
};
static_assert(sizeof(NumiHumanRestingLinearBoundReceiptRecord) == 68u);
static_assert(offsetof(NumiHumanRestingLinearBoundReceiptRecord,
                       sourceLower) == 20u);
static_assert(offsetof(NumiHumanRestingLinearBoundReceiptRecord,
                       action) == 64u);

namespace detail {

struct NumiHumanRestingLinearBoundSourceRow {
    std::uint32_t equalityIndex;
    std::uint32_t dependentQIndex;
    std::uint32_t masterQIndex;
};

inline constexpr std::array<NumiHumanRestingLinearBoundSourceRow, 35u>
    kNumiHumanRestingLinearBoundSourceRows{{
        {0u,12u,7u}, {1u,13u,7u}, {2u,15u,9u}, {3u,14u,8u},
        {4u,16u,7u}, {5u,18u,9u}, {6u,17u,8u}, {7u,19u,7u},
        {8u,21u,9u}, {9u,20u,8u}, {10u,22u,7u}, {11u,24u,9u},
        {12u,23u,8u}, {15u,25u,36u}, {16u,26u,36u}, {17u,28u,36u},
        {18u,27u,36u}, {19u,31u,36u}, {20u,29u,36u}, {21u,30u,36u},
        {22u,32u,36u}, {23u,34u,36u}, {24u,33u,36u}, {25u,37u,35u},
        {26u,63u,74u}, {27u,64u,74u}, {28u,66u,74u}, {29u,65u,74u},
        {30u,69u,74u}, {31u,67u,74u}, {32u,68u,74u}, {33u,70u,74u},
        {34u,72u,74u}, {35u,71u,74u}, {36u,75u,73u},
    }};

inline std::uint32_t orderedFloatKey(const float value) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

inline float floatFromOrderedKey(const std::uint32_t key) noexcept {
    const std::uint32_t bits = (key & 0x80000000u) != 0u
        ? (key ^ 0x80000000u) : ~key;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline bool evaluateLinearSourceTargetF32(
    const MRNumiHumanJointEqualityGPU& equality,
    const float masterPosition,
    float& target
) noexcept {
    const float delta = masterPosition -
        equality.referencesAndCoefficients0.y;
    const float polynomial = std::fma(
        delta, equality.referencesAndCoefficients0.w,
        equality.referencesAndCoefficients0.z);
    target = equality.referencesAndCoefficients0.x + polynomial;
    return std::isfinite(delta) && std::isfinite(polynomial) &&
        std::isfinite(target);
}

template <typename Predicate>
inline bool firstOrderedFloatSatisfying(
    const std::uint32_t minimumKey,
    const std::uint32_t maximumKey,
    const MRNumiHumanJointEqualityGPU& equality,
    Predicate&& predicate,
    std::uint64_t& resultKey
) noexcept {
    std::uint64_t first = minimumKey;
    std::uint64_t pastLast = static_cast<std::uint64_t>(maximumKey) + 1u;
    while (first < pastLast) {
        const std::uint64_t middle = first + (pastLast - first) / 2u;
        const float position = floatFromOrderedKey(
            static_cast<std::uint32_t>(middle));
        float target = 0.0f;
        if (!evaluateLinearSourceTargetF32(equality, position, target))
            return false;
        if (predicate(target)) pastLast = middle;
        else first = middle + 1u;
    }
    resultKey = first;
    return true;
}

} // namespace detail

// Convert the dependent source interval to the exact set of representable
// master F32 coordinates using the same affine Horner/FMA target operation as
// the native Stand projection. Endpoints are found by ordered-F32 binary
// search; no tolerance or interval widening is introduced.
inline bool compileNumiHumanRestingLinearBoundConsolidation(
    const NumiHumanJointEqualityPayload& source,
    const std::span<const std::uint8_t> sourcePayloadSHA256,
    const std::span<const float> referenceQ,
    const std::span<const MRDofPropertiesGPU> sourceDofs,
    const std::span<const MRDofPropertiesGPU> inputDerivedDofs,
    const std::span<const MRNumiHumanJointEqualityGPU> compiledEqualities,
    std::vector<MRDofPropertiesGPU>& derivedDofs,
    std::vector<NumiHumanRestingLinearBoundReceiptRecord>& receipt,
    std::string& error
) {
    const auto fail = [&error](const char* message) {
        error = message;
        return false;
    };
    if (sourcePayloadSHA256.size() !=
            kNumiHumanRestingReferenceNHEQPayloadSHA256.size() ||
        !std::equal(sourcePayloadSHA256.begin(), sourcePayloadSHA256.end(),
                    kNumiHumanRestingReferenceNHEQPayloadSHA256.begin()))
        return fail("linear-bound consolidation requires the exact reference NHEQ1 payload");
    std::vector<MRNumiHumanJointEqualityGPU> expectedEqualities;
    std::string reductionError;
    if (!compileNumiHumanRestingHandReduction(
            source, referenceQ, sourceDofs, expectedEqualities,
            reductionError))
        return fail("linear-bound consolidation requires the admitted rigid-hand reduction");
    if (compiledEqualities.size() != expectedEqualities.size() ||
        std::memcmp(compiledEqualities.data(), expectedEqualities.data(),
                    expectedEqualities.size() *
                        sizeof(MRNumiHumanJointEqualityGPU)) != 0)
        return fail("linear-bound consolidation equality program differs from the admitted rigid-hand reduction");
    if (inputDerivedDofs.size() != sourceDofs.size())
        return fail("linear-bound consolidation DoF dimensions differ");

    std::vector<std::uint8_t> sourceDependents(source.nq, 0u);
    for (const auto& equality : source.records) {
        const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
            equality.indices.w == MR_INVALID_INDEX;
        const bool coupled = equality.indices.z < source.nq &&
            equality.indices.w < source.nv;
        if (equality.indices.x >= source.nq ||
            equality.indices.y >= source.nv || (!fixed && !coupled) ||
            (coupled && (equality.indices.x == equality.indices.z ||
                         equality.indices.y == equality.indices.w)) ||
            !std::isfinite(equality.referencesAndCoefficients0.x) ||
            !std::isfinite(equality.referencesAndCoefficients0.y) ||
            !std::isfinite(equality.referencesAndCoefficients0.z) ||
            !std::isfinite(equality.referencesAndCoefficients0.w) ||
            !std::isfinite(equality.coefficients1.x) ||
            !std::isfinite(equality.coefficients1.y) ||
            !std::isfinite(equality.coefficients1.z))
            return fail("linear-bound consolidation encountered a malformed source equality");
        if (sourceDependents[equality.indices.x] != 0u)
            return fail("linear-bound consolidation encountered duplicate dependents");
        sourceDependents[equality.indices.x] = 1u;
    }

    std::vector<float> masterLower(sourceDofs.size(), 0.0f);
    std::vector<float> masterUpper(sourceDofs.size(), 0.0f);
    std::vector<std::uint8_t> masterSeen(sourceDofs.size(), 0u);
    std::vector<std::uint8_t> rowSeen(source.records.size(), 0u);
    for (const auto& expected : detail::kNumiHumanRestingLinearBoundSourceRows) {
        if (expected.equalityIndex >= source.records.size() ||
            expected.dependentQIndex >= source.nq ||
            expected.masterQIndex >= source.nq)
            return fail("linear-bound source map is outside the admitted NHEQ dimensions");
        const auto& equality = source.records[expected.equalityIndex];
        const std::uint32_t dependentV = expected.dependentQIndex - 1u;
        const std::uint32_t masterV = expected.masterQIndex - 1u;
        if (equality.indices.x != expected.dependentQIndex ||
            equality.indices.y != dependentV ||
            equality.indices.z != expected.masterQIndex ||
            equality.indices.w != masterV ||
            equality.coefficients1.x != 0.0f ||
            equality.coefficients1.y != 0.0f ||
            equality.coefficients1.z != 0.0f ||
            !std::isfinite(equality.referencesAndCoefficients0.w) ||
            equality.referencesAndCoefficients0.w == 0.0f)
            return fail("linear-bound source row is not the admitted nonzero affine dependency");
        if (sourceDependents[expected.masterQIndex] != 0u)
            return fail("linear-bound source dependency is chained or lacks an independent master");
        if (dependentV >= sourceDofs.size() || masterV >= sourceDofs.size())
            return fail("linear-bound source DoF is missing");
        const MRDofPropertiesGPU& dependent = sourceDofs[dependentV];
        const MRDofPropertiesGPU& master = sourceDofs[masterV];
        const MRDofPropertiesGPU& currentDependent = inputDerivedDofs[dependentV];
        const MRDofPropertiesGPU& currentMaster = inputDerivedDofs[masterV];
        const auto validScalar = [](const MRDofPropertiesGPU& dof,
                                    const std::uint32_t q,
                                    const std::uint32_t v) {
            return dof.articulationIndex == 0u && dof.qIndex == q &&
                dof.vIndex == v &&
                (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                std::isfinite(dof.limits.x) && std::isfinite(dof.limits.y) &&
                dof.limits.x < dof.limits.y;
        };
        if (!validScalar(dependent, expected.dependentQIndex, dependentV) ||
            !validScalar(master, expected.masterQIndex, masterV))
            return fail("linear-bound dependent or master source interval is invalid");
        if (std::memcmp(&dependent, &currentDependent, sizeof(dependent)) != 0 ||
            std::memcmp(&master, &currentMaster, sizeof(master)) != 0)
            return fail("linear-bound source coordinates were already modified in the derived model");
        rowSeen[expected.equalityIndex] = 1u;
        if (masterSeen[masterV] == 0u) {
            masterLower[masterV] = master.limits.x;
            masterUpper[masterV] = master.limits.y;
            masterSeen[masterV] = 1u;
        } else if (masterLower[masterV] != master.limits.x ||
                   masterUpper[masterV] != master.limits.y) {
            return fail("shared linear-bound master has inconsistent source intervals");
        }
    }

    std::size_t affineBoundCount = 0u;
    for (std::size_t index = 0u; index < source.records.size(); ++index) {
        const auto& equality = source.records[index];
        if (equality.indices.z == MR_INVALID_INDEX ||
            equality.indices.w == MR_INVALID_INDEX ||
            equality.indices.y >= sourceDofs.size() ||
            (sourceDofs[equality.indices.y].flags &
                MR_DOF_FLAG_POSITION_LIMIT) == 0u)
            continue;
        if (equality.coefficients1.x == 0.0f &&
            equality.coefficients1.y == 0.0f &&
            equality.coefficients1.z == 0.0f) {
            ++affineBoundCount;
            if (equality.referencesAndCoefficients0.w == 0.0f ||
                rowSeen[index] == 0u)
                return fail("linear-bound source contains an unexpected or degenerate bounded affine row");
        }
    }
    if (affineBoundCount != detail::kNumiHumanRestingLinearBoundSourceRows.size())
        return fail("linear-bound source count differs from the admitted 35 rows");

    std::vector<NumiHumanRestingLinearBoundReceiptRecord> candidateReceipt;
    candidateReceipt.reserve(detail::kNumiHumanRestingLinearBoundSourceRows.size());
    for (const auto& expected : detail::kNumiHumanRestingLinearBoundSourceRows) {
        const auto& equality = source.records[expected.equalityIndex];
        const std::uint32_t dependentV = expected.dependentQIndex - 1u;
        const std::uint32_t masterV = expected.masterQIndex - 1u;
        const MRDofPropertiesGPU& dependent = sourceDofs[dependentV];
        const MRDofPropertiesGPU& master = sourceDofs[masterV];
        const std::uint32_t minimumKey = detail::orderedFloatKey(master.limits.x);
        const std::uint32_t maximumKey = detail::orderedFloatKey(master.limits.y);
        const float slope = equality.referencesAndCoefficients0.w;
        const float dependentLower = dependent.limits.x;
        const float dependentUpper = dependent.limits.y;
        const auto predicate = [&](const float target, const bool lowerEdge) {
            if (slope > 0.0f)
                return lowerEdge ? target >= dependentLower
                                 : target > dependentUpper;
            return lowerEdge ? target <= dependentUpper
                             : target < dependentLower;
        };
        std::uint64_t firstKey = 0u;
        std::uint64_t pastLastKey = 0u;
        if (!detail::firstOrderedFloatSatisfying(
                minimumKey, maximumKey, equality,
                [&](const float target) { return predicate(target, true); },
                firstKey) ||
            !detail::firstOrderedFloatSatisfying(
                minimumKey, maximumKey, equality,
                [&](const float target) { return predicate(target, false); },
                pastLastKey))
            return fail("linear-bound endpoint evaluation became nonfinite");
        const std::uint64_t pastDomain =
            static_cast<std::uint64_t>(maximumKey) + 1u;
        if (firstKey > maximumKey || pastLastKey <= minimumKey ||
            firstKey >= pastLastKey || pastLastKey > pastDomain)
            return fail("linear-bound source interval has no representable master coordinates");
        const std::uint64_t lastKey = pastLastKey == pastDomain
            ? maximumKey : pastLastKey - 1u;
        const float inverseLower = detail::floatFromOrderedKey(
            static_cast<std::uint32_t>(firstKey));
        const float inverseUpper = detail::floatFromOrderedKey(
            static_cast<std::uint32_t>(lastKey));
        float lowerTarget = 0.0f;
        float upperTarget = 0.0f;
        if (!(inverseLower < inverseUpper) ||
            !detail::evaluateLinearSourceTargetF32(
                equality, inverseLower, lowerTarget) ||
            !detail::evaluateLinearSourceTargetF32(
                equality, inverseUpper, upperTarget) ||
            lowerTarget < dependentLower || lowerTarget > dependentUpper ||
            upperTarget < dependentLower || upperTarget > dependentUpper)
            return fail("linear-bound inverse interval is degenerate or violates its source endpoints");
        if (firstKey > minimumKey) {
            float previousTarget = 0.0f;
            if (!detail::evaluateLinearSourceTargetF32(
                    equality, detail::floatFromOrderedKey(
                        static_cast<std::uint32_t>(firstKey - 1u)),
                    previousTarget) ||
                (previousTarget >= dependentLower &&
                 previousTarget <= dependentUpper))
                return fail("linear-bound lower inverse endpoint is not exact");
        }
        if (pastLastKey < pastDomain) {
            float nextTarget = 0.0f;
            if (!detail::evaluateLinearSourceTargetF32(
                    equality, detail::floatFromOrderedKey(
                        static_cast<std::uint32_t>(pastLastKey)),
                    nextTarget) ||
                (nextTarget >= dependentLower &&
                 nextTarget <= dependentUpper))
                return fail("linear-bound upper inverse endpoint is not exact");
        }
        const float mergedLower = std::max(
            masterLower[masterV], inverseLower);
        const float mergedUpper = std::min(
            masterUpper[masterV], inverseUpper);
        if (!(mergedLower < mergedUpper))
            return fail("linear-bound master intersection is empty or degenerate");
        masterLower[masterV] = mergedLower;
        masterUpper[masterV] = mergedUpper;
        candidateReceipt.push_back({
            .equalityIndex = expected.equalityIndex,
            .dependentVIndex = dependentV,
            .dependentQIndex = expected.dependentQIndex,
            .masterVIndex = masterV,
            .masterQIndex = expected.masterQIndex,
            .sourceLower = dependentLower,
            .sourceUpper = dependentUpper,
            .sourceMasterLower = master.limits.x,
            .sourceMasterUpper = master.limits.y,
            .inverseMasterLower = inverseLower,
            .inverseMasterUpper = inverseUpper,
            .dependentOriginalFlags = dependent.flags,
            .dependentDerivedFlags = dependent.flags &
                ~MR_DOF_FLAG_POSITION_LIMIT,
            .masterFlags = master.flags,
            .action = static_cast<std::uint32_t>(
                NumiHumanRestingLinearBoundAction::consolidated),
        });
    }

    for (const auto& expected : detail::kNumiHumanRestingLinearBoundSourceRows) {
        const std::uint32_t masterV = expected.masterQIndex - 1u;
        const float referenceMaster = referenceQ[expected.masterQIndex];
        if (!std::isfinite(referenceMaster) ||
            referenceMaster < masterLower[masterV] ||
            referenceMaster > masterUpper[masterV])
            return fail("linear-bound intersection excludes the reference master coordinate");
    }

    std::vector<MRDofPropertiesGPU> candidateDofs(
        inputDerivedDofs.begin(), inputDerivedDofs.end());
    for (const auto& row : candidateReceipt) {
        MRDofPropertiesGPU& dependent = candidateDofs[row.dependentVIndex];
        dependent.flags &= ~MR_DOF_FLAG_POSITION_LIMIT;
        dependent.limits.x = 0.0f;
        dependent.limits.y = 0.0f;
        candidateDofs[row.masterVIndex].limits.x =
            masterLower[row.masterVIndex];
        candidateDofs[row.masterVIndex].limits.y =
            masterUpper[row.masterVIndex];
    }
    for (auto& row : candidateReceipt) {
        row.derivedMasterLower = masterLower[row.masterVIndex];
        row.derivedMasterUpper = masterUpper[row.masterVIndex];
    }
    derivedDofs = std::move(candidateDofs);
    receipt = std::move(candidateReceipt);
    error.clear();
    return true;
}

} // namespace metalrobo
