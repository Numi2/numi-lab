#include "cardboard_contact_node_incidence_sort_reference.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
    require(stableRadixPassCount((1u << 28u) - 1u) == 7u,
            "seven passes must cover node IDs below the 28-bit sentinel boundary");
    require(stableRadixPassCount(1u << 28u) == 8u,
            "eight passes must preserve the invalid sentinel at the 28-bit boundary");
    require(stableRadixPassCount(UINT32_MAX) == 8u,
            "eight passes must cover the full uint32 node domain");
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
