#include "../src/metal/fem_backtracking_failure_witness.metalinc"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>

static float uintBitsAsFloat(const std::uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
static std::uint32_t floatBitsAsUint(const float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

struct Partial {
    float x;
    float trialJ;
    float currentJ;
    float reason;
};
struct Witness {
    float trialJ;
    float currentJ;
    std::uint32_t globalTet;
    std::uint32_t reason;
};

static Witness reducePartials(const Partial* partials, const std::size_t count,
                              float& acceptedMinimumJ) {
    std::uint32_t firstBad = 0xffffffffu;
    Witness selected{0.0f, 0.0f, firstBad, 0u};
    acceptedMinimumJ = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        if (partials[i].reason != 0.0f) {
            const auto tet = floatBitsAsUint(partials[i].x);
            if (tet < firstBad) {
                firstBad = tet;
                selected = {partials[i].trialJ, partials[i].currentJ, tet,
                            static_cast<std::uint32_t>(partials[i].reason)};
            }
        } else {
            acceptedMinimumJ = std::fmin(acceptedMinimumJ, partials[i].x);
        }
    }
    return selected;
}

int main() {
    constexpr float lower = 0.2f;
    constexpr float upper = 2.5f;
    // A zero-candidate line search skips every trial and retains the
    // deterministic lane-zero no-trial witness tuple.
    const auto emptyWitness = nmFEMBacktrackingEmptyFailureWitness();
    assert(emptyWitness.trialJ == 0.0f);
    assert(emptyWitness.currentJ == 0.0f);
    assert(std::isinf(emptyWitness.firstBad) && emptyWitness.firstBad > 0.0f);
    assert(emptyWitness.reason == 0u);
    assert(nmFEMBacktrackingFailureReason(0.199f, lower, upper, true) == 2u);
    assert(nmFEMBacktrackingFailureReason(2.501f, lower, upper, true) == 4u);
    assert(nmFEMBacktrackingFailureReason(lower, lower, upper, true) == 0u);
    assert(nmFEMBacktrackingFailureReason(upper, lower, upper, true) == 0u);
    assert(nmFEMBacktrackingFailureReason(1.0f, lower, upper, false) == 8u);
    assert(nmFEMBacktrackingFailureReason(
        std::numeric_limits<float>::quiet_NaN(), lower, upper, true) == 1u);
    assert(nmFEMBacktrackingFailureReason(
        std::numeric_limits<float>::infinity(), lower, upper, true) == 1u);
    assert(nmFEMBacktrackingFailureReason(
        -std::numeric_limits<float>::infinity(), lower, upper, true) == 1u);

    // The legacy determinant predicate was (J >= lower && J <= upper),
    // which rejects NaN bounds. Compare helper results against that predicate
    // across finite, NaN, and +/-infinity bound combinations.
    const float nanBound = std::numeric_limits<float>::quiet_NaN();
    const float posInf = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    struct BoundCase { float lower; float upper; unsigned int expected; };
    const std::array<BoundCase, 8> boundCases{{
        {lower, upper, 0u},
        {nanBound, upper, 2u},
        {lower, nanBound, 4u},
        {nanBound, nanBound, 6u},
        {negInf, posInf, 0u},
        {posInf, posInf, 2u},
        {negInf, negInf, 4u},
        {posInf, negInf, 6u},
    }};
    for (const auto& bounds : boundCases) {
        const float value = 1.0f;
        const bool oldPredicate = std::isfinite(value) &&
            value >= bounds.lower && value <= bounds.upper;
        const unsigned int reason = nmFEMBacktrackingFailureReason(
            value, bounds.lower, bounds.upper, true);
        assert((reason == 0u) == oldPredicate);
        assert(reason == bounds.expected);
    }
    assert(nmFEMBacktrackingFailureReason(1.0f, nanBound, upper, false) == 2u);
    assert(nmFEMBacktrackingFailureReason(1.0f, lower, nanBound, false) == 4u);
    assert(nmFEMBacktrackingFailureReason(1.0f, negInf, posInf, false) == 8u);

    // Invalid partial x is a bit-cast uint, not numeric float conversion.
    // This verifies adjacent indices beyond float's exact integer range and
    // the top of the uint domain without large allocations.
    const std::array<std::uint32_t, 4> ids{{
        16777216u, 16777217u, 0xfffffffeu, 0xffffffffu}};
    for (const auto id : ids) {
        assert(floatBitsAsUint(uintBitsAsFloat(id)) == id);
    }
    assert(floatBitsAsUint(uintBitsAsFloat(16777216u)) !=
           floatBitsAsUint(uintBitsAsFloat(16777217u)));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<Partial, 4> mixed{{
        {uintBitsAsFloat(29419u), 2.5087922f, 2.4985766f, 4.0f},
        {uintBitsAsFloat(29418u), 0.19f, 0.21f, 2.0f},
        {uintBitsAsFloat(16777217u), nan, 1.0f, 1.0f},
        {0.73f, 0.0f, 0.0f, 0.0f},
    }};
    float acceptedMinimumJ = 0.0f;
    const Witness selected = reducePartials(
        mixed.data(), mixed.size(), acceptedMinimumJ);
    assert(selected.globalTet == 29418u && selected.reason == 2u);
    assert(selected.trialJ == 0.19f && selected.currentJ == 0.21f);
    assert(acceptedMinimumJ == 0.73f);

    // Nonfinite trial J is retained when its global tet is the selected one.
    const std::array<Partial, 2> nonfinite{{
        {uintBitsAsFloat(16777217u), nan, 1.0f, 1.0f},
        {uintBitsAsFloat(0xfffffffeu), 2.6f, 2.4f, 4.0f},
    }};
    const Witness nonfiniteSelected = reducePartials(
        nonfinite.data(), nonfinite.size(), acceptedMinimumJ);
    assert(nonfiniteSelected.globalTet == 16777217u);
    assert(nonfiniteSelected.reason == 1u);
    assert(std::isnan(nonfiniteSelected.trialJ));
    assert(nonfiniteSelected.currentJ == 1.0f);

    // The diagnostic material/bounds join remains globalTet -> tet material
    // index -> existing material table; the witness tuple needs no new field.
    struct TetMaterial { unsigned int materialIndex; };
    struct MaterialBounds { float lower; float upper; };
    const std::array<TetMaterial, 2> tets{{{1u}, {0u}}};
    const std::array<MaterialBounds, 2> materials{{{0.2f, 2.5f}, {0.4f, 3.0f}}};
    constexpr unsigned int failingGlobalTet = 0u;
    const unsigned int materialIndex = tets[failingGlobalTet].materialIndex;
    assert(materialIndex == 1u);
    assert(materials[materialIndex].lower == 0.4f);
    assert(materials[materialIndex].upper == 3.0f);
    assert(nmFEMBacktrackingFailureReason(
        3.01f, materials[materialIndex].lower,
        materials[materialIndex].upper, true) == 4u);

    // The external legacy diagnostic slot is a float, so its display value
    // cannot represent all uint tet IDs even though internal selection does.
    assert(static_cast<std::uint32_t>(static_cast<float>(16777217u)) ==
           16777216u);
    std::puts("PASS: zero-trial tuple, legacy bound predicate parity, exact uint selection, nonfinite witness, minimum-J and material join");
}
