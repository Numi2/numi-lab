#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace cardboard::contact_incidence_sort {

inline constexpr std::uint32_t kValid = 1u;
inline constexpr std::uint32_t kInvalid = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kCorners = 6u;

inline std::uint32_t stableRadixPassCount(const std::uint32_t nodeCount) {
    // Valid IDs are [0, nodeCount), while UINT32_MAX is the invalid key.
    // The radix domain must therefore be strictly larger than nodeCount:
    // at exact powers of 16 the next nibble is needed to keep the invalid
    // sentinel distinct from the largest valid node ID.
    const std::uint32_t keyBits =
        static_cast<std::uint32_t>(std::bit_width(nodeCount));
    return std::max(1u, (keyBits + 3u) / 4u);
}

struct Contact {
    std::array<std::uint32_t, kCorners> nodes{};
    std::uint32_t flags = 0u;
};

struct Environment {
    std::vector<Contact> contacts;
    std::vector<std::uint32_t> activeIndices;
    std::uint32_t activeCount = 0u;
};

struct Range {
    std::uint32_t first = 0u;
    std::uint32_t count = 0u;
    friend bool operator==(const Range&, const Range&) = default;
};

struct Result {
    std::vector<Range> ranges;
    std::vector<std::uint32_t> incidence;
};

struct WorkspaceLayout {
    bool valid = false;
    std::uint32_t paddedPerEnvironment = 0u;
    std::uint64_t totalRecords = 0u;
    std::uint64_t keyBytes = 0u;
    std::uint64_t payloadBytes = 0u;
    std::string error;
};

inline WorkspaceLayout checkedWorkspaceLayout(
    const std::uint32_t capacity,
    const std::uint32_t environments,
    const std::uint32_t nodeCount
) {
    WorkspaceLayout result;
    if (capacity == 0u || environments == 0u || nodeCount == 0u) {
        result.error = "contact, environment, and node capacities must be nonzero";
        return result;
    }
    if (std::uint64_t(nodeCount) * environments >
        std::numeric_limits<std::uint32_t>::max()) {
        result.error = "node dispatch index exceeds uint32";
        return result;
    }
    if (capacity > (std::numeric_limits<std::uint32_t>::max() - 5u) / 8u) {
        result.error = "slot/local payload exceeds uint32";
        return result;
    }
    const std::uint64_t logical = std::uint64_t(kCorners) * capacity;
    if (logical > std::numeric_limits<std::uint32_t>::max() ||
        logical > (std::uint64_t{1} << 31u)) {
        result.error = "logical corner count exceeds bitonic sort domain";
        return result;
    }
    const std::uint64_t padded = std::bit_ceil(logical);
    if (padded == 0u || padded > std::numeric_limits<std::uint32_t>::max() ||
        padded > std::numeric_limits<std::uint32_t>::max() / environments) {
        result.error = "padded sort dispatch exceeds uint32";
        return result;
    }
    result.totalRecords = padded * environments;
    if (result.totalRecords > std::numeric_limits<std::uint64_t>::max() /
            (sizeof(std::uint64_t) + sizeof(std::uint32_t))) {
        result.error = "sort byte count exceeds uint64";
        return result;
    }
    result.paddedPerEnvironment = static_cast<std::uint32_t>(padded);
    result.keyBytes = result.totalRecords * sizeof(std::uint64_t);
    result.payloadBytes = result.totalRecords * sizeof(std::uint32_t);
    result.valid = true;
    return result;
}

inline Result buildLegacyReference(
    const std::uint32_t nodeCount,
    const std::uint32_t capacity,
    const std::vector<Environment>& environments
) {
    Result result;
    const std::size_t envCount = environments.size();
    result.ranges.resize(envCount * nodeCount);
    result.incidence.resize(envCount * std::size_t(kCorners) * capacity,
                            kInvalid);
    for (std::size_t environment = 0u; environment < envCount; ++environment) {
        const Environment& input = environments[environment];
        std::vector<std::vector<std::uint32_t>> byNode(nodeCount);
        const std::uint32_t activeCount = std::min<std::uint32_t>(
            input.activeCount, capacity);
        for (std::uint32_t activeSlot = 0u; activeSlot < activeCount;
             ++activeSlot) {
            if (activeSlot >= input.activeIndices.size()) continue;
            const std::uint32_t slot = input.activeIndices[activeSlot];
            if (slot >= capacity || slot >= input.contacts.size()) continue;
            const Contact& contact = input.contacts[slot];
            if ((contact.flags & kValid) == 0u) continue;
            for (std::uint32_t local = 0u; local < kCorners; ++local) {
                const std::uint32_t node = contact.nodes[local];
                if (node < nodeCount)
                    byNode[node].push_back(8u * slot + local);
            }
        }
        std::uint32_t cursor = static_cast<std::uint32_t>(
            environment * std::size_t(kCorners) * capacity);
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            Range& range = result.ranges[environment * nodeCount + node];
            range = {cursor, static_cast<std::uint32_t>(byNode[node].size())};
            for (const std::uint32_t payload : byNode[node])
                result.incidence[cursor++] = payload;
        }
    }
    return result;
}

inline Result buildSortedReference(
    const std::uint32_t nodeCount,
    const std::uint32_t capacity,
    const std::vector<Environment>& environments
) {
    struct Record {
        std::uint32_t node;
        std::uint32_t order;
        std::uint32_t payload;
    };
    Result result;
    const std::size_t envCount = environments.size();
    result.ranges.resize(envCount * nodeCount);
    result.incidence.resize(envCount * std::size_t(kCorners) * capacity,
                            kInvalid);
    for (std::size_t environment = 0u; environment < envCount; ++environment) {
        const Environment& input = environments[environment];
        const std::uint32_t logicalCount = kCorners * capacity;
        const std::uint32_t padded = static_cast<std::uint32_t>(
            std::bit_ceil(std::uint64_t(logicalCount)));
        constexpr Record invalidRecord{kInvalid, kInvalid, 0u};
        std::vector<Record> records(padded, invalidRecord);
        const std::uint32_t activeCount = std::min<std::uint32_t>(
            input.activeCount, capacity);
        for (std::uint32_t activeSlot = 0u; activeSlot < activeCount;
             ++activeSlot) {
            if (activeSlot >= input.activeIndices.size()) continue;
            const std::uint32_t slot = input.activeIndices[activeSlot];
            if (slot >= capacity || slot >= input.contacts.size()) continue;
            const Contact& contact = input.contacts[slot];
            if ((contact.flags & kValid) == 0u) continue;
            for (std::uint32_t local = 0u; local < kCorners; ++local) {
                const std::uint32_t node = contact.nodes[local];
                if (node < nodeCount) {
                    records[activeSlot * kCorners + local] = {
                        node, activeSlot * 8u + local, slot * 8u + local};
                }
            }
        }
        const auto less = [](const Record& a, const Record& b) {
            return std::tie(a.node, a.order) < std::tie(b.node, b.order);
        };
        // Mirror the GPU's fixed ascending bitonic network exactly, including
        // the padded invalid records and swap direction for each stage.
        for (std::uint32_t width = 2u;; width *= 2u) {
            for (std::uint32_t distance = width / 2u; distance != 0u;
                 distance /= 2u) {
                for (std::uint32_t index = 0u; index < padded; ++index) {
                    const std::uint32_t partner = index ^ distance;
                    if (partner <= index) continue;
                    const bool ascending = (index & width) == 0u;
                    const bool swap = ascending
                        ? less(records[partner], records[index])
                        : less(records[index], records[partner]);
                    if (swap) std::swap(records[index], records[partner]);
                }
            }
            if (width == padded) break;
        }
        const std::uint32_t base = static_cast<std::uint32_t>(
            environment * std::size_t(kCorners) * capacity);
        std::size_t recordIndex = 0u;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            const std::uint32_t first = static_cast<std::uint32_t>(recordIndex);
            while (recordIndex < records.size() &&
                   records[recordIndex].node == node) {
                result.incidence[base + recordIndex] =
                    records[recordIndex].payload;
                ++recordIndex;
            }
            result.ranges[environment * nodeCount + node] = {
                base + first,
                static_cast<std::uint32_t>(recordIndex - first)};
        }
        for (; recordIndex < records.size(); ++recordIndex)
            if (records[recordIndex].node != kInvalid)
                throw std::logic_error(
                    "sorted reference retained an out-of-range node");
    }
    return result;
}


inline Result buildRadixReference(
    const std::uint32_t nodeCount,
    const std::uint32_t capacity,
    const std::vector<Environment>& environments
) {
    struct Record {
        std::uint32_t node;
        std::uint32_t payload;
    };
    Result result;
    const std::size_t envCount = environments.size();
    result.ranges.resize(envCount * nodeCount);
    result.incidence.resize(envCount * std::size_t(kCorners) * capacity,
                            kInvalid);
    for (std::size_t environment = 0u; environment < envCount; ++environment) {
        const Environment& input = environments[environment];
        const std::uint32_t activeCount = std::min<std::uint32_t>(
            input.activeCount, capacity);
        std::vector<Record> records(
            std::size_t(activeCount) * kCorners, {kInvalid, kInvalid});
        for (std::uint32_t activeSlot = 0u; activeSlot < activeCount;
             ++activeSlot) {
            if (activeSlot >= input.activeIndices.size()) continue;
            const std::uint32_t slot = input.activeIndices[activeSlot];
            if (slot >= capacity || slot >= input.contacts.size()) continue;
            const Contact& contact = input.contacts[slot];
            if ((contact.flags & kValid) == 0u) continue;
            for (std::uint32_t local = 0u; local < kCorners; ++local) {
                const std::uint32_t node = contact.nodes[local];
                if (node < nodeCount)
                    records[std::size_t(activeSlot) * kCorners + local] = {
                        node, slot * 8u + local};
            }
        }

        std::vector<Record> scratch(records.size());
        for (std::uint32_t pass = 0u;
             pass < stableRadixPassCount(nodeCount); ++pass) {
            std::array<std::uint32_t, 16u> offsets{};
            for (const Record& record : records)
                ++offsets[(record.node >> (4u * pass)) & 0x0fu];
            std::uint32_t cursor = 0u;
            for (std::uint32_t bucket = 0u; bucket < offsets.size(); ++bucket) {
                const std::uint32_t count = offsets[bucket];
                offsets[bucket] = cursor;
                cursor += count;
            }
            for (const Record& record : records) {
                const std::uint32_t bucket =
                    (record.node >> (4u * pass)) & 0x0fu;
                scratch[offsets[bucket]++] = record;
            }
            records.swap(scratch);
        }

        const std::uint32_t base = static_cast<std::uint32_t>(
            environment * std::size_t(kCorners) * capacity);
        std::size_t recordIndex = 0u;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            const std::uint32_t first = static_cast<std::uint32_t>(recordIndex);
            while (recordIndex < records.size() &&
                   records[recordIndex].node == node) {
                result.incidence[base + recordIndex] =
                    records[recordIndex].payload;
                ++recordIndex;
            }
            result.ranges[environment * nodeCount + node] = {
                base + first,
                static_cast<std::uint32_t>(recordIndex - first)};
        }
        for (; recordIndex < records.size(); ++recordIndex)
            if (records[recordIndex].node != kInvalid)
                throw std::logic_error(
                    "radix reference retained an out-of-range node");
    }
    return result;
}

inline bool equivalentUsedIncidence(const Result& a, const Result& b) {
    if (a.ranges != b.ranges || a.incidence.size() != b.incidence.size())
        return false;
    for (const Range& range : a.ranges) {
        const std::uint64_t end = std::uint64_t(range.first) + range.count;
        if (end > a.incidence.size()) return false;
        for (std::uint32_t index = 0u; index < range.count; ++index)
            if (a.incidence[range.first + index] !=
                b.incidence[range.first + index]) return false;
    }
    return true;
}

} // namespace cardboard::contact_incidence_sort
