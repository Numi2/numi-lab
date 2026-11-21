#pragma once

#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace numi::matter::fem_body_contact_grouping {

inline constexpr std::uint32_t kTriangleContactFlags =
    NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
    NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;

enum class OwnershipFailure : std::uint8_t {
    none,
    invalidSlave,
    anchoredSlave,
    duplicateNonTriangle,
    duplicateSameBody,
};

struct OwnershipReport {
    OwnershipFailure failure = OwnershipFailure::none;
    bool hasDistinctBodyTriangleDuplicates = false;
    [[nodiscard]] bool accepted() const noexcept {
        return failure == OwnershipFailure::none;
    }
};

// Keep the existing unique-slave rule for every contact mode. Its only
// exception is multiple prepared triangle queries for one slave when each
// query belongs to a distinct articulated body. Mode-specific payload checks
// remain in the owning adapter.
inline OwnershipReport validateDistinctBodyTriangleSlaves(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanTendonFEMNodeAnchorGPU> anchors
) {
    std::vector<std::vector<std::uint32_t>> bodiesBySlave(anchors.size());
    std::vector<std::uint32_t> firstFlags(anchors.size(), 0u);
    OwnershipReport report{};
    for (const auto& sample : samples) {
        if (sample.slaveNode >= anchors.size()) {
            report.failure = OwnershipFailure::invalidSlave;
            return report;
        }
        if ((anchors[sample.slaveNode].flags &
             NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) != 0u) {
            report.failure = OwnershipFailure::anchoredSlave;
            return report;
        }
        auto& bodies = bodiesBySlave[sample.slaveNode];
        if (bodies.empty()) {
            bodies.push_back(sample.bodyIndex);
            firstFlags[sample.slaveNode] = sample.flags;
            continue;
        }
        if (sample.flags != kTriangleContactFlags ||
            firstFlags[sample.slaveNode] != kTriangleContactFlags) {
            report.failure = OwnershipFailure::duplicateNonTriangle;
            return report;
        }
        if (std::find(bodies.begin(), bodies.end(), sample.bodyIndex) !=
            bodies.end()) {
            report.failure = OwnershipFailure::duplicateSameBody;
            return report;
        }
        bodies.push_back(sample.bodyIndex);
        report.hasDistinctBodyTriangleDuplicates = true;
    }
    return report;
}

// Reuse an already grouped layout without copying the immutable contact rows.
// Output one range per slave, in first-appearance order.
inline bool contiguousSlaveGroupRanges(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::vector<NMIncidenceRangeGPU>& ranges
) {
    if (!ranges.empty() || samples.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::unordered_set<std::uint32_t> seen;
    for (std::size_t first = 0; first < samples.size();) {
        const auto slave = samples[first].slaveNode;
        if (!seen.insert(slave).second) {
            ranges.clear();
            return false;
        }
        std::size_t end = first + 1;
        while (end < samples.size() && samples[end].slaveNode == slave) ++end;
        ranges.push_back({static_cast<std::uint32_t>(first),
                          static_cast<std::uint32_t>(end - first), 0u, 0u});
        first = end;
    }
    return true;
}

// Deterministic sample-domain layout for the grouped force kernel. Group order
// follows first appearance of each slave; rows within a group retain original
// order. Ranges move with samples, while firstTriangle remains an index into
// the shared unchanged triangle buffer. Each range indexes one complete group.
inline bool stableGroupRowsBySlave(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    std::vector<NMNumiHumanFEMBodyContactSampleGPU>& groupedSamples,
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>& groupedRanges,
    std::vector<NMIncidenceRangeGPU>& groupRanges
) {
    if ((!ranges.empty() && ranges.size() != samples.size()) ||
        !groupedSamples.empty() || !groupedRanges.empty() ||
        !groupRanges.empty() ||
        samples.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::unordered_map<std::uint32_t, std::size_t> groupForSlave;
    std::vector<std::vector<std::size_t>> groups;
    for (std::size_t row = 0u; row < samples.size(); ++row) {
        const auto slave = samples[row].slaveNode;
        auto found = groupForSlave.find(slave);
        if (found == groupForSlave.end()) {
            const std::size_t group = groups.size();
            groupForSlave.emplace(slave, group);
            groups.emplace_back();
            found = groupForSlave.find(slave);
        }
        groups[found->second].push_back(row);
    }
    groupedSamples.reserve(samples.size());
    if (!ranges.empty()) groupedRanges.reserve(ranges.size());
    groupRanges.reserve(groups.size());
    for (const auto& group : groups) {
        if (group.empty() || group.size() >
                std::numeric_limits<std::uint32_t>::max()) return false;
        groupRanges.push_back({static_cast<std::uint32_t>(groupedSamples.size()),
                               static_cast<std::uint32_t>(group.size()), 0u, 0u});
        for (std::size_t local = 0u; local < group.size(); ++local) {
            const auto sourceRow = group[local];
            groupedSamples.push_back(samples[sourceRow]);
            if (!ranges.empty()) groupedRanges.push_back(ranges[sourceRow]);
        }
    }
    return groupedSamples.size() == samples.size() &&
        (ranges.empty() || groupedRanges.size() == ranges.size());
}

} // namespace numi::matter::fem_body_contact_grouping
