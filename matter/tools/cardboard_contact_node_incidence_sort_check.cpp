#include "cardboard_contact_node_incidence_sort_reference.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cardboard::contact_incidence_sort;

namespace {
std::uint64_t checks = 0u;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

Contact contact(std::array<std::uint32_t, 6> nodes, bool valid = true) {
    return {nodes, valid ? kValid : 0u};
}

void compare(const std::uint32_t nodes, const std::uint32_t capacity,
             const std::vector<Environment>& environments,
             const std::string& label) {
    const Result legacy = buildLegacyReference(nodes, capacity, environments);
    const Result sorted = buildSortedReference(nodes, capacity, environments);
    const Result radix = buildRadixReference(nodes, capacity, environments);
    require(equivalentUsedIncidence(legacy, sorted),
            label + ": sorted CSR differs from the legacy activeSlot/local order");
    require(equivalentUsedIncidence(legacy, radix),
            label + ": radix CSR differs from the legacy activeSlot/local order");
}

void explicitControls() {
    const std::uint32_t invalid = kInvalid;
    std::vector<Environment> environments(2u);
    environments[0].contacts = {
        contact({1u, 1u, 2u, 0u, 5u, invalid}),
        contact({0u, 0u, 0u, 0u, 0u, 0u}, false),
        contact({2u, 0u, 3u, 1u, 2u, 4u}),
        contact({4u, 4u, 4u, 4u, 4u, 4u}),
    };
    // Deliberate duplicate slot plus an invalid slot verifies that the sort
    // key preserves each active occurrence and drops only what legacy drops.
    environments[0].activeIndices = {2u, 0u, 2u, 3u};
    environments[0].activeCount = 4u;
    environments[1].contacts = {
        contact({0u, 1u, 2u, 3u, 4u, 5u}, false),
        contact({0u, 2u, 2u, 3u, 4u, 4u}),
        contact({5u, 5u, 5u, 5u, 5u, 5u}),
        contact({1u, 3u, 1u, 3u, invalid, 0u}),
    };
    environments[1].activeIndices = {1u, 0u, 3u, 2u};
    environments[1].activeCount = 2u;
    compare(6u, 4u, environments, "multi-environment duplicates/invalids");

    for (Environment& environment : environments) environment.activeCount = 0u;
    compare(6u, 4u, environments, "zero active count");

    environments[0].activeCount = 99u;
    environments[0].activeIndices = {invalid, invalid, invalid, invalid};
    environments[1].activeCount = 99u;
    environments[1].activeIndices = {3u, 0u, 1u, 2u};
    compare(6u, 4u, environments, "clamped counts and invalid contact slots");
}

void exhaustiveSingleSlot() {
    constexpr std::uint32_t nodeCount = 3u;
    constexpr std::uint32_t capacity = 1u;
    constexpr std::uint32_t choices = nodeCount + 1u; // last value maps to invalid
    std::uint64_t supportAssignments = 1u;
    for (std::uint32_t local = 0u; local < 6u; ++local)
        supportAssignments *= choices;
    for (std::uint64_t encoded = 0u; encoded < supportAssignments; ++encoded) {
        std::array<std::uint32_t, 6> nodes{};
        std::uint64_t value = encoded;
        for (std::uint32_t local = 0u; local < 6u; ++local) {
            const std::uint32_t choice =
                static_cast<std::uint32_t>(value % choices);
            value /= choices;
            nodes[local] = choice == nodeCount ? kInvalid : choice;
        }
        for (const bool valid : {false, true}) {
            for (const std::uint32_t activeSlot : {0u, kInvalid}) {
                for (const std::uint32_t activeCount : {0u, 1u, 2u}) {
                    std::vector<Environment> environments(1u);
                    environments[0].contacts = {contact(nodes, valid)};
                    environments[0].activeIndices = {activeSlot};
                    environments[0].activeCount = activeCount;
                    compare(nodeCount, capacity, environments,
                        "exhaustive one-slot support assignment");
                }
            }
        }
    }
    require(checks > 49000u,
            "exhaustive one-slot incidence sweep did not execute its full state space");
}


void hierarchicalPrefixModel(
    const std::uint32_t activeTiles,
    const std::uint32_t maxTileCount,
    const std::uint32_t environmentCount
) {
    constexpr std::uint32_t bucketCount = 16u;
    constexpr std::uint32_t chunkTiles = 256u;
    require(activeTiles <= maxTileCount && environmentCount != 0u,
            "hierarchical prefix model dimensions are invalid");
    const std::uint32_t maxChunks =
        (maxTileCount + chunkTiles - 1u) / chunkTiles;
    const std::uint32_t activeChunks =
        (activeTiles + chunkTiles - 1u) / chunkTiles;
    const std::uint64_t histogramWords =
        std::uint64_t(maxTileCount) * bucketCount;
    const std::uint64_t requiredWords =
        histogramWords + std::uint64_t(bucketCount) * maxChunks +
        bucketCount;
    const std::uint64_t strideWords =
        std::max<std::uint64_t>(maxTileCount * 32ull,
                                (requiredWords + 1u) / 2u) * 2u;
    require(strideWords >= requiredWords,
            "hierarchical prefix scratch stride is too small");
    require(strideWords * environmentCount >= requiredWords * environmentCount,
            "per-environment hierarchical scratch slices overlap");

    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        std::vector<std::uint32_t> histogram(
            std::uint64_t(maxTileCount) * bucketCount, 0u);
        const std::uint32_t totalRecords = activeTiles == 0u
            ? 0u : activeTiles * 32u - (activeTiles % 7u);
        for (std::uint32_t tile = 0u; tile < activeTiles; ++tile) {
            const std::uint32_t records = std::min(
                32u, totalRecords - tile * 32u);
            for (std::uint32_t lane = 0u; lane < records; ++lane) {
                const std::uint32_t bucket =
                    (lane * 7u + tile * 3u + environment * 5u) % bucketCount;
                ++histogram[std::size_t(tile) * bucketCount + bucket];
            }
        }
        std::vector<std::uint32_t> expected(histogram.size(), 0u);
        // Independently construct the exact global (bucket-major, then
        // tile-major) exclusive destination offsets.
        std::uint32_t bucketBase = 0u;
        for (std::uint32_t bucket = 0u; bucket < bucketCount; ++bucket) {
            std::uint32_t tilePrefix = 0u;
            for (std::uint32_t tile = 0u; tile < activeTiles; ++tile) {
                const std::size_t index =
                    std::size_t(tile) * bucketCount + bucket;
                expected[index] = bucketBase + tilePrefix;
                tilePrefix += histogram[index];
            }
            bucketBase += tilePrefix;
        }

        std::vector<std::uint32_t> actual(histogram.size(), 0u);
        if (activeTiles <= chunkTiles) {
            std::uint32_t base = 0u;
            for (std::uint32_t bucket = 0u; bucket < bucketCount; ++bucket) {
                std::uint32_t prefix = 0u;
                for (std::uint32_t tile = 0u; tile < activeTiles; ++tile) {
                    const std::size_t index =
                        std::size_t(tile) * bucketCount + bucket;
                    actual[index] = base + prefix;
                    prefix += histogram[index];
                }
                base += prefix;
            }
        } else {
            std::vector<std::uint32_t> localPrefixes(histogram.size(), 0u);
            std::vector<std::uint32_t> chunkTotals(
                std::size_t(bucketCount) * maxChunks, 0u);
            for (std::uint32_t bucket = 0u; bucket < bucketCount; ++bucket) {
                for (std::uint32_t chunk = 0u; chunk < activeChunks; ++chunk) {
                    std::uint32_t local = 0u;
                    const std::uint32_t first = chunk * chunkTiles;
                    const std::uint32_t end = std::min(
                        activeTiles, first + chunkTiles);
                    for (std::uint32_t tile = first; tile < end; ++tile) {
                        const std::size_t index =
                            std::size_t(tile) * bucketCount + bucket;
                        localPrefixes[index] = local;
                        local += histogram[index];
                    }
                    chunkTotals[std::size_t(bucket) * maxChunks + chunk] = local;
                }
            }
            std::array<std::uint32_t, bucketCount> bucketBases{};
            std::uint32_t base = 0u;
            for (std::uint32_t bucket = 0u; bucket < bucketCount; ++bucket) {
                bucketBases[bucket] = base;
                std::uint32_t chunkPrefix = 0u;
                for (std::uint32_t chunk = 0u; chunk < activeChunks; ++chunk) {
                    const std::size_t chunkIndex =
                        std::size_t(bucket) * maxChunks + chunk;
                    const std::uint32_t chunkTotal = chunkTotals[chunkIndex];
                    chunkTotals[chunkIndex] = chunkPrefix;
                    chunkPrefix += chunkTotal;
                }
                base += chunkPrefix;
            }
            for (std::uint32_t bucket = 0u; bucket < bucketCount; ++bucket) {
                for (std::uint32_t tile = 0u; tile < activeTiles; ++tile) {
                    const std::uint32_t chunk = tile / chunkTiles;
                    const std::size_t index =
                        std::size_t(tile) * bucketCount + bucket;
                    actual[index] = bucketBases[bucket] +
                        chunkTotals[std::size_t(bucket) * maxChunks + chunk] +
                        localPrefixes[index];
                }
            }
        }
        require(actual == expected,
                "hierarchical radix prefix differs from the exact stable bucket/tile offsets");
    }
}

void checkedBounds() {
    const WorkspaceLayout maximum = checkedWorkspaceLayout(131072u, 4u, 528444u);
    require(maximum.valid && maximum.paddedPerEnvironment == 1048576u &&
            maximum.totalRecords == 4194304u &&
            maximum.keyBytes == 33554432u &&
            maximum.payloadBytes == 16777216u,
            "131072-slot/four-environment workspace sizing mismatch");
    require(!checkedWorkspaceLayout(0u, 1u, 1u).valid,
            "zero capacity must reject");
    require(!checkedWorkspaceLayout(1u, 0u, 1u).valid,
            "zero environment count must reject");
    require(!checkedWorkspaceLayout(1u, 1u, 0u).valid,
            "zero node count must reject");
    require(!checkedWorkspaceLayout(UINT32_MAX, 1u, 1u).valid,
            "payload/index overflow must reject");
    require(!checkedWorkspaceLayout(1u, UINT32_MAX, 2u).valid,
            "multi-environment node indexing overflow must reject");
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 16> radixBoundaries{{
        {0u, 1u}, {1u, 1u}, {15u, 1u}, {16u, 2u}, {17u, 2u},
        {255u, 2u}, {256u, 3u}, {257u, 3u}, {65535u, 4u},
        {65536u, 5u}, {65537u, 5u}, {(1u << 24u) - 1u, 6u},
        {1u << 24u, 7u}, {55296u, 4u}, {(1u << 28u) - 1u, 7u},
        {1u << 28u, 8u},
    }};
    for (const auto [nodes, expectedPasses] : radixBoundaries) {
        const std::uint32_t passes = stableRadixPassCount(nodes);
        require(passes == expectedPasses,
                "radix pass count mismatch at a nibble/domain boundary");
        const std::uint64_t radixDomain =
            std::uint64_t{1} << (4u * passes);
        const std::uint64_t maskedInvalid = radixDomain - 1u;
        require(nodes == 0u || maskedInvalid > std::uint64_t(nodes - 1u),
                "masked invalid sentinel collides with a valid node ID");
    }
    hierarchicalPrefixModel(0u, 512u, 2u);
    hierarchicalPrefixModel(256u, 512u, 1u);
    hierarchicalPrefixModel(257u, 512u, 1u);
    hierarchicalPrefixModel(256u, 512u, 2u);
    hierarchicalPrefixModel(257u, 512u, 2u);
    require(stableRadixPassCount(UINT32_MAX) == 8u,
            "eight passes must cover the full uint32 node domain");
    const std::uint64_t fullRadixDomain = std::uint64_t{1} << 32u;
    require(fullRadixDomain - 1u > std::uint64_t(UINT32_MAX - 1u),
            "full-width masked invalid sentinel must sort after every valid ID");
}
}

int main() {
    try {
        explicitControls();
        exhaustiveSingleSlot();
        checkedBounds();
        std::cout << "PASS deterministic sorted contact incidence CPU reference; checks="
                  << checks << "; exhaustive_single_slot_supports=4096; environments=2"
                  << "; physical_steps=0\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL deterministic sorted contact incidence CPU reference: "
                  << exception.what() << '\n';
        return 1;
    }
}
