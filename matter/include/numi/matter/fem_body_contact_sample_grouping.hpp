#pragma once

#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
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

// Deterministic sample-domain layout for the grouped force kernel. Group order
// follows first appearance of each slave; rows within a group retain original
// order. Ranges move with samples, while firstTriangle remains an index into
// the shared unchanged triangle buffer. Only each group head has a nonzero
// span count; non-head rows are zero.
inline bool stableGroupRowsBySlave(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    std::vector<NMNumiHumanFEMBodyContactSampleGPU>& groupedSamples,
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>& groupedRanges,
    std::vector<std::uint32_t>& groupCounts
) {
    if ((!ranges.empty() && ranges.size() != samples.size()) ||
        !groupedSamples.empty() || !groupedRanges.empty() ||
        !groupCounts.empty() ||
        samples.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::unordered_map<std::uint32_t, std::size_t> groupForSlave;
    std::vector<std::vector<std::size_t>> groups;
    groups.reserve(samples.size());
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
    groupCounts.reserve(samples.size());
    for (const auto& group : groups) {
        if (group.empty() || group.size() >
                std::numeric_limits<std::uint32_t>::max()) return false;
        const auto span = static_cast<std::uint32_t>(group.size());
        for (std::size_t local = 0u; local < group.size(); ++local) {
            const auto sourceRow = group[local];
            groupedSamples.push_back(samples[sourceRow]);
            if (!ranges.empty()) groupedRanges.push_back(ranges[sourceRow]);
            groupCounts.push_back(local == 0u ? span : 0u);
        }
    }
    return groupedSamples.size() == samples.size() &&
        groupCounts.size() == samples.size() &&
        (ranges.empty() || groupedRanges.size() == ranges.size());
}

} // namespace numi::matter::fem_body_contact_grouping
