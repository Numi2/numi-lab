#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace numi::matter::detail {

// A graph factor may be retained only for a later Newton iteration in the
// same solver transaction, and only after the caller has excluded forced
// rebuilds and the exact selected diagnostic capture iteration. The Metal
// kernels additionally require that the environment's nonlinear-completion
// latch is set.
[[nodiscard]] inline bool regionalGraphCapturePendingForIteration(
    const bool selectorValidAndEnabled,
    const bool captureAlreadyCompleted,
    const std::uint32_t selectorRoot,
    const std::uint32_t currentRoot,
    const std::uint32_t selectorIteration,
    const std::uint32_t currentIteration
) noexcept {
    return selectorValidAndEnabled && !captureAlreadyCompleted &&
        selectorRoot == currentRoot && selectorIteration == currentIteration;
}

[[nodiscard]] inline bool regionalGraphCompletedTailReuseAllowed(
    const std::uint32_t solverIteration,
    const bool forceRebuild,
    const bool selectedCapturePending) noexcept {
    return solverIteration > 0u && !forceRebuild &&
        !selectedCapturePending;
}

[[nodiscard]] inline std::uint32_t regionalHumanGraphFactorReuseMode(
    const bool factorLagEnabled,
    const std::uint32_t solverIteration,
    const bool forceRebuild,
    const bool selectedCapturePending
) noexcept {
    if (solverIteration == 0u || forceRebuild || selectedCapturePending)
        return 0u;
    return factorLagEnabled ? 2u : 1u;
}

[[nodiscard]] inline bool regionalGraphMaySkipPatchFactorization(
    const bool completedTailReuseAllowed,
    const bool nonlinearCompletionLatch,
    const bool priorFactorValid) noexcept {
    return completedTailReuseAllowed && nonlinearCompletionLatch &&
        priorFactorValid;
}

// A bounded host-built graph layout for the selector-5 additive Schwarz
// preconditioner. The graph is derived from exact active tetrahedron
// connectivity and the source Dirichlet partition. Source-free patches retain
// the original free-only construction as an exact prefix. Supplementary
// patches contain only initially fixed nodes, so dynamic release gains local
// coverage without changing original free-node patch membership. Patches are ordered
// deterministically by ascending free-node anchors, breadth-first sorted
// neighbors, then sorted node IDs. This states an ordering rule only; it does
// not claim membership parity with any offline mesh.
struct RegionalGraphSchwarzPatch {
    std::uint32_t nodeOffset = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t matrixFloatOffset = 0u;
    std::uint32_t pivotOffset = 0u;
};

struct RegionalGraphSchwarzLayout {
    std::vector<RegionalGraphSchwarzPatch> patches;
    std::vector<std::uint32_t> patchNodes;
    // One row descriptor per local scalar DOF: patch, local node, axis,
    // global node.  Matrix assembly uses this to give each output row exactly
    // one writer.
    std::vector<std::array<std::uint32_t, 4>> rows;
    // Node-to-patch-value CSR. Values index patchNodes and therefore the
    // patch-local correction buffer, without a dense node-by-patch map.
    std::vector<std::uint32_t> nodePatchOffsets;
    std::vector<std::uint32_t> nodePatchValues;
    std::vector<std::uint32_t> nodeCoverage;
    std::uint32_t nodeCount = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t totalPatchDofs = 0u;
    std::uint32_t matrixFloatCount = 0u;
    // Three n-word segments per patch: row permutation and row/column scale.
    std::uint32_t pivotCount = 0u;
    std::uint64_t metadataBytes = 0u;
    std::uint64_t deviceBytes = 0u;
    std::string error;
};

[[nodiscard]] inline bool regionalGraphSchwarzCheckedAdd(
    const std::uint64_t a, const std::uint64_t b,
    std::uint64_t& result) noexcept {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) return false;
    result = a + b;
    return true;
}

[[nodiscard]] inline bool regionalGraphSchwarzCheckedMul(
    const std::uint64_t a, const std::uint64_t b,
    std::uint64_t& result) noexcept {
    if (a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a)
        return false;
    result = a * b;
    return true;
}

[[nodiscard]] inline RegionalGraphSchwarzLayout buildRegionalGraphSchwarzLayout(
    const std::uint32_t nodeCount,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    // Exact authored source mask: 1 means initially free; 0 means initially
    // fixed. Runtime separately checks each candidate-free node against the
    // union of the source-free and supplementary initial-fixed patches.
    const std::span<const std::uint8_t> freeMask,
    const std::uint32_t environmentCount,
    const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull,
    const std::uint32_t maximumNodeCount = 4096u,
    const std::uint32_t maximumTetrahedronCount = 32768u,
    const std::uint32_t maximumPatchCount = 128u,
    const std::uint32_t maximumPatchNodes = 256u,
    const std::uint32_t anchorStride = 128u,
    const bool includeInitiallyFixedPatches = true,
    const bool groupUncoveredNodes = false
) {
    RegionalGraphSchwarzLayout out;
    out.nodeCount = nodeCount;
    out.environmentCount = environmentCount;
    const auto fail = [&](const char* message) {
        out.error = message;
        return out;
    };
    if (nodeCount == 0u || nodeCount > maximumNodeCount ||
        tetrahedra.empty() || tetrahedra.size() > maximumTetrahedronCount ||
        freeMask.size() != nodeCount || environmentCount == 0u ||
        environmentCount > 4u || maximumPatchNodes == 0u ||
        maximumPatchNodes > 256u || anchorStride == 0u ||
        anchorStride > maximumPatchNodes || maximumPatchCount == 0u) {
        return fail("graph Schwarz input exceeds its explicit FEM admission bounds");
    }

    std::vector<std::set<std::uint32_t>> adjacency(nodeCount);
    std::vector<std::uint32_t> sourceFreeNodes;
    std::vector<std::uint32_t> sourceFixedNodes;
    sourceFreeNodes.reserve(nodeCount);
    sourceFixedNodes.reserve(nodeCount);
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        if (freeMask[node] > 1u)
            return fail("graph Schwarz free mask is not binary");
        if (freeMask[node] != 0u) sourceFreeNodes.push_back(node);
        else sourceFixedNodes.push_back(node);
    }
    if (sourceFreeNodes.empty())
        return fail("graph Schwarz requires at least one unconstrained FEM node");

    for (const auto& tet : tetrahedra) {
        for (std::uint32_t i = 0u; i < 4u; ++i) {
            if (tet[i] >= nodeCount)
                return fail("graph Schwarz tetrahedron references an invalid FEM node");
            for (std::uint32_t j = 0u; j < i; ++j)
                if (tet[i] == tet[j])
                    return fail("graph Schwarz tetrahedron repeats a node");
        }
        for (std::uint32_t i = 0u; i < 4u; ++i) {
            for (std::uint32_t j = 0u; j < 4u; ++j) {
                if (i != j && freeMask[tet[i]] == freeMask[tet[j]])
                    adjacency[tet[i]].insert(tet[j]);
            }
        }
    }

    std::vector<std::vector<std::uint32_t>> nodePatches;
    nodePatches.reserve(sourceFreeNodes.size() + sourceFixedNodes.size());
    std::vector<std::uint8_t> covered(nodeCount, 0u);
    const auto appendPartitionPatches = [&](
        const std::vector<std::uint32_t>& partitionNodes) {
        if (partitionNodes.empty()) return;
        std::vector<std::uint32_t> anchors;
        for (std::size_t index = 0u; index < partitionNodes.size();
             index += anchorStride)
            anchors.push_back(partitionNodes[index]);
        if (anchors.back() != partitionNodes.back())
            anchors.push_back(partitionNodes.back());
        for (const std::uint32_t anchor : anchors) {
            std::vector<std::uint32_t> selected{anchor};
            std::vector<std::uint8_t> visited(nodeCount, 0u);
            visited[anchor] = 1u;
            std::deque<std::uint32_t> frontier{anchor};
            while (!frontier.empty() && selected.size() < maximumPatchNodes) {
                const std::uint32_t current = frontier.front();
                frontier.pop_front();
                for (const std::uint32_t neighbor : adjacency[current]) {
                    if (visited[neighbor] != 0u) continue;
                    visited[neighbor] = 1u;
                    selected.push_back(neighbor);
                    frontier.push_back(neighbor);
                    if (selected.size() == maximumPatchNodes) break;
                }
            }
            std::sort(selected.begin(), selected.end());
            for (const std::uint32_t node : selected) covered[node] = 1u;
            nodePatches.push_back(std::move(selected));
        }
        if (!groupUncoveredNodes) {
            for (const std::uint32_t node : partitionNodes)
                if (covered[node] == 0u) nodePatches.push_back({node});
        } else {
            // Disconnected or missed nodes may share a bounded dense patch.
            // Absent graph edges produce exact zero off-diagonal blocks; this
            // only batches small local solves, with the same production A.
            std::vector<std::uint32_t> remainder;
            for (const std::uint32_t node : partitionNodes) {
                if (covered[node] != 0u) continue;
                remainder.push_back(node);
                if (remainder.size() == maximumPatchNodes) {
                    nodePatches.push_back(std::move(remainder));
                    remainder.clear();
                }
            }
            if (!remainder.empty()) nodePatches.push_back(std::move(remainder));
        }
    };
    // Preserve the old source-free patches in exactly their historical order.
    appendPartitionPatches(sourceFreeNodes);
    if (includeInitiallyFixedPatches)
        appendPartitionPatches(sourceFixedNodes);
    if (nodePatches.empty() || nodePatches.size() > maximumPatchCount)
        return fail("graph Schwarz patch count exceeds its bounded layout");

    std::uint64_t matrixFloatCount = 0u;
    std::uint64_t pivotCount = 0u;
    std::uint64_t totalPatchDofs = 0u;
    std::uint64_t totalPatchNodes = 0u;
    out.patches.reserve(nodePatches.size());
    for (const auto& patch : nodePatches) {
        if (patch.empty() || patch.size() > maximumPatchNodes)
            return fail("graph Schwarz contains an empty or oversized patch");
        const std::uint64_t dofs = 3u * static_cast<std::uint64_t>(patch.size());
        std::uint64_t matrixEntries = 0u;
        std::uint64_t pivotSlots = 0u;
        if (!regionalGraphSchwarzCheckedMul(dofs, 3u, pivotSlots) ||
            !regionalGraphSchwarzCheckedMul(dofs, dofs, matrixEntries) ||
            matrixFloatCount > std::numeric_limits<std::uint32_t>::max() - matrixEntries ||
            pivotCount > std::numeric_limits<std::uint32_t>::max() - pivotSlots ||
            totalPatchDofs > std::numeric_limits<std::uint32_t>::max() - dofs ||
            totalPatchNodes > std::numeric_limits<std::uint32_t>::max() - patch.size()) {
            return fail("graph Schwarz packed offsets overflow their 32-bit device indices");
        }
        out.patches.push_back({
            static_cast<std::uint32_t>(totalPatchNodes),
            static_cast<std::uint32_t>(patch.size()),
            static_cast<std::uint32_t>(matrixFloatCount),
            static_cast<std::uint32_t>(pivotCount)});
        matrixFloatCount += matrixEntries;
        pivotCount += pivotSlots;
        totalPatchDofs += dofs;
        totalPatchNodes += patch.size();
        out.patchNodes.insert(out.patchNodes.end(), patch.begin(), patch.end());
    }
    out.matrixFloatCount = static_cast<std::uint32_t>(matrixFloatCount);
    out.pivotCount = static_cast<std::uint32_t>(pivotCount);
    out.totalPatchDofs = static_cast<std::uint32_t>(totalPatchDofs);

    out.rows.reserve(out.totalPatchDofs);
    for (std::uint32_t patchIndex = 0u; patchIndex < out.patches.size(); ++patchIndex) {
        const RegionalGraphSchwarzPatch patch = out.patches[patchIndex];
        for (std::uint32_t localNode = 0u; localNode < patch.nodeCount; ++localNode) {
            const std::uint32_t node = out.patchNodes[patch.nodeOffset + localNode];
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                out.rows.push_back({patchIndex, localNode, axis, node});
        }
    }
    if (out.rows.size() != out.totalPatchDofs)
        return fail("graph Schwarz row map is incomplete");

    out.nodeCoverage.assign(nodeCount, 0u);
    for (const std::uint32_t node : out.patchNodes) {
        if (node >= nodeCount ||
            (!includeInitiallyFixedPatches && freeMask[node] == 0u))
            return fail("graph Schwarz patches include an excluded or invalid node");
        ++out.nodeCoverage[node];
    }
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if ((freeMask[node] != 0u || includeInitiallyFixedPatches) &&
            out.nodeCoverage[node] == 0u)
            return fail("graph Schwarz patches do not cover every free node");
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if (!includeInitiallyFixedPatches && freeMask[node] == 0u &&
            out.nodeCoverage[node] != 0u)
            return fail("free-only graph Schwarz patches cover a constrained node");

    out.nodePatchOffsets.assign(static_cast<std::size_t>(nodeCount) + 1u, 0u);
    for (const std::uint32_t node : out.patchNodes)
        ++out.nodePatchOffsets[static_cast<std::size_t>(node) + 1u];
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        out.nodePatchOffsets[node + 1u] += out.nodePatchOffsets[node];
    out.nodePatchValues.resize(out.patchNodes.size());
    std::vector<std::uint32_t> cursor = out.nodePatchOffsets;
    for (std::uint32_t valueIndex = 0u; valueIndex < out.patchNodes.size(); ++valueIndex) {
        const std::uint32_t node = out.patchNodes[valueIndex];
        out.nodePatchValues[cursor[node]++] = valueIndex;
    }

    std::uint64_t metadataBytes = 0u;
    std::uint64_t term = 0u;
    const auto addArray = [&](const std::uint64_t count, const std::uint64_t width) {
        return regionalGraphSchwarzCheckedMul(count, width, term) &&
            regionalGraphSchwarzCheckedAdd(metadataBytes, term, metadataBytes);
    };
    if (!addArray(out.patches.size(), sizeof(RegionalGraphSchwarzPatch)) ||
        !addArray(out.patchNodes.size(), sizeof(std::uint32_t)) ||
        !addArray(out.rows.size(), sizeof(std::array<std::uint32_t, 4>)) ||
        !addArray(out.nodePatchOffsets.size(), sizeof(std::uint32_t)) ||
        !addArray(out.nodePatchValues.size(), sizeof(std::uint32_t)) ||
        !addArray(out.nodeCoverage.size(), sizeof(std::uint32_t))) {
        return fail("graph Schwarz metadata byte count overflow");
    }
    out.metadataBytes = metadataBytes;

    std::uint64_t perEnvironment = 0u;
    std::uint64_t factorBytes = 0u;
    std::uint64_t originalMatrixBytes = 0u;
    std::uint64_t pivotBytes = 0u;
    std::uint64_t correctionBytes = 0u;
    std::uint64_t validBytes = 0u;
    if (!regionalGraphSchwarzCheckedMul(matrixFloatCount, sizeof(float), factorBytes) ||
        !regionalGraphSchwarzCheckedMul(matrixFloatCount, sizeof(float), originalMatrixBytes) ||
        !regionalGraphSchwarzCheckedMul(pivotCount, sizeof(std::uint32_t), pivotBytes) ||
        !regionalGraphSchwarzCheckedMul(totalPatchNodes, 4u * sizeof(float), correctionBytes) ||
        !regionalGraphSchwarzCheckedMul(nodePatches.size(), sizeof(std::uint32_t), validBytes) ||
        !regionalGraphSchwarzCheckedAdd(factorBytes, pivotBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedAdd(perEnvironment, originalMatrixBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedAdd(perEnvironment, correctionBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedAdd(perEnvironment, validBytes, perEnvironment)) {
        return fail("graph Schwarz per-environment device byte count overflow");
    }
    std::uint64_t allEnvironments = 0u;
    if (!regionalGraphSchwarzCheckedMul(perEnvironment, environmentCount, allEnvironments) ||
        !regionalGraphSchwarzCheckedAdd(allEnvironments, metadataBytes, out.deviceBytes) ||
        out.deviceBytes > maximumDeviceBytes) {
        return fail("graph Schwarz packed device arenas, including source matrices for one refinement pass, exceed the configured device-memory bound");
    }
    return out;
}


// Return the exact optic region indices from a complete skin+optic basis.
// Non-optic skin/fat regions remain in the source basis and are deliberately
// not removed by this graph-neighborhood selector.
[[nodiscard]] inline bool regionalGraphSchwarzSelectOpticRegionIndices(
    const std::span<const std::uint32_t> regionLabels,
    std::vector<std::uint32_t>& opticRegionIndices,
    std::uint32_t& binsPerSide,
    std::string& error
) {
    opticRegionIndices.clear();
    binsPerSide = 0u;
    constexpr std::uint32_t leftBase = 0xF1772001u;
    constexpr std::uint32_t rightBase = 0xF1819001u;
    constexpr std::uint32_t leftPrefix = leftBase & 0xFFFFFF00u;
    constexpr std::uint32_t rightPrefix = rightBase & 0xFFFFFF00u;
    std::array<std::uint8_t, 10u> leftSeen{};
    std::array<std::uint8_t, 10u> rightSeen{};
    std::uint32_t leftCount = 0u;
    std::uint32_t rightCount = 0u;
    for (std::uint32_t index = 0u; index < regionLabels.size(); ++index) {
        const std::uint32_t label = regionLabels[index];
        if (index != 0u && regionLabels[index - 1u] >= label) {
            error = "full optic graph source labels must be strictly ordered";
            opticRegionIndices.clear();
            binsPerSide = 0u;
            return false;
        }
        std::array<std::uint8_t, 10u>* seen = nullptr;
        std::uint32_t base = 0u;
        std::uint32_t* count = nullptr;
        if ((label & 0xFFFFFF00u) == leftPrefix) {
            seen = &leftSeen;
            base = leftBase;
            count = &leftCount;
        } else if ((label & 0xFFFFFF00u) == rightPrefix) {
            seen = &rightSeen;
            base = rightBase;
            count = &rightCount;
        } else {
            continue;
        }
        const std::uint32_t bin = label - base;
        if (bin >= seen->size() || (*seen)[bin] != 0u) {
            error = "full optic graph source has an unsupported or duplicate optic bin";
            opticRegionIndices.clear();
            binsPerSide = 0u;
            return false;
        }
        (*seen)[bin] = 1u;
        ++*count;
        opticRegionIndices.push_back(index);
    }
    if (leftCount != rightCount ||
        (leftCount != 5u && leftCount != 10u)) {
        error = "full optic graph source must include five or ten bins for each optic side";
        opticRegionIndices.clear();
        binsPerSide = 0u;
        return false;
    }
    binsPerSide = leftCount;
    for (std::uint32_t bin = 0u; bin < 10u; ++bin) {
        const std::uint8_t expected = bin < binsPerSide ? 1u : 0u;
        if (leftSeen[bin] != expected || rightSeen[bin] != expected) {
            error = "full optic graph source is missing a bilateral bin or has an incomplete ten-bin tail";
            opticRegionIndices.clear();
            binsPerSide = 0u;
            return false;
        }
    }
    error.clear();
    return true;
}

struct RegionalGraphSchwarzMappedPatchSummary {
    std::uint32_t patchCount = 0u;
    std::uint32_t patchNodeSlots = 0u;
    std::uint32_t maximumPatchNodes = 0u;
};

// Append local anatomy-domain patches to a full-world graph layout. The
// caller constructs `localToGlobal` from an exact typed source owner and must
// retain the original full operator/certificate. Existing patch arrays remain
// an unchanged prefix; coverage/CSR are rebuilt so additive weights normalize
// over both the original skin layout and the new local blocks.
[[nodiscard]] inline bool appendRegionalGraphSchwarzMappedPatches(
    RegionalGraphSchwarzLayout& layout,
    const RegionalGraphSchwarzLayout& localLayout,
    const std::span<const std::uint32_t> localToGlobal,
    const std::span<const std::uint32_t> globalNodeOwners,
    const std::span<const std::uint32_t> allowedOwners,
    const std::uint32_t maximumPatchCount,
    const std::uint64_t maximumDeviceBytes,
    RegionalGraphSchwarzMappedPatchSummary& summary,
    std::string& error
) {
    summary = {};
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (!layout.error.empty() || !localLayout.error.empty() ||
        layout.nodeCount == 0u || localLayout.nodeCount == 0u ||
        layout.environmentCount != localLayout.environmentCount ||
        layout.environmentCount == 0u || layout.environmentCount > 4u ||
        globalNodeOwners.size() != layout.nodeCount ||
        localToGlobal.size() != localLayout.nodeCount ||
        layout.nodeCoverage.size() != layout.nodeCount ||
        allowedOwners.size() != 2u ||
        !std::is_sorted(allowedOwners.begin(), allowedOwners.end()) ||
        allowedOwners[0] == allowedOwners[1] || maximumPatchCount == 0u ||
        maximumDeviceBytes == 0u)
        return fail("mapped graph patches have invalid base/local layouts or bounds");
    if (localLayout.patches.empty() || localLayout.patchNodes.empty() ||
        localLayout.patches.size() == 0u ||
        localLayout.patches.size() > maximumPatchCount ||
        localLayout.nodeCoverage.size() != localLayout.nodeCount)
        return fail("mapped optic graph has no bounded local patches");
    std::vector<std::uint8_t> mappedGlobal(layout.nodeCount, 0u);
    for (std::uint32_t local = 0u; local < localLayout.nodeCount; ++local) {
        const std::uint32_t global = localToGlobal[local];
        if (global >= layout.nodeCount || mappedGlobal[global] != 0u ||
            layout.nodeCoverage[global] == 0u ||
            !std::binary_search(allowedOwners.begin(), allowedOwners.end(),
                                globalNodeOwners[global]))
            return fail("mapped optic node is absent, duplicated, or outside the exact owner set");
        mappedGlobal[global] = 1u;
    }
    std::uint64_t matrixFloats = layout.matrixFloatCount;
    std::uint64_t pivots = layout.pivotCount;
    std::uint64_t totalDofs = layout.totalPatchDofs;
    std::uint64_t totalPatchNodes = layout.patchNodes.size();
    std::vector<RegionalGraphSchwarzPatch> patches;
    std::vector<std::uint32_t> patchNodes;
    std::vector<std::array<std::uint32_t, 4>> rows;
    patches.reserve(localLayout.patches.size());
    patchNodes.reserve(localLayout.patchNodes.size());
    rows.reserve(localLayout.rows.size());
    std::uint32_t observedMaximumNodes = 0u;
    for (std::size_t index = 0u; index < localLayout.patches.size(); ++index) {
        const RegionalGraphSchwarzPatch& localPatch =
            localLayout.patches[index];
        const std::uint64_t end =
            static_cast<std::uint64_t>(localPatch.nodeOffset) +
            localPatch.nodeCount;
        if (localPatch.nodeCount == 0u ||
            end > localLayout.patchNodes.size() ||
            patches.size() + layout.patches.size() >= maximumPatchCount)
            return fail("mapped optic patch range or combined patch count is invalid");
        const std::uint64_t dofs = 3u * localPatch.nodeCount;
        std::uint64_t matrixEntries = 0u;
        std::uint64_t pivotSlots = 0u;
        if (!regionalGraphSchwarzCheckedMul(dofs, dofs, matrixEntries) ||
            !regionalGraphSchwarzCheckedMul(dofs, 3u, pivotSlots) ||
            matrixFloats > std::numeric_limits<std::uint32_t>::max() - matrixEntries ||
            pivots > std::numeric_limits<std::uint32_t>::max() - pivotSlots ||
            totalDofs > std::numeric_limits<std::uint32_t>::max() - dofs ||
            totalPatchNodes > std::numeric_limits<std::uint32_t>::max() -
                localPatch.nodeCount ||
            patchNodes.size() > std::numeric_limits<std::uint32_t>::max() -
                localPatch.nodeCount)
            return fail("mapped optic patch packed offsets overflow 32-bit device indices");
        const std::uint32_t first = static_cast<std::uint32_t>(patchNodes.size());
        std::uint32_t owner = std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t local = 0u; local < localPatch.nodeCount; ++local) {
            const std::uint32_t localNode =
                localLayout.patchNodes[localPatch.nodeOffset + local];
            if (localNode >= localToGlobal.size())
                return fail("mapped optic patch references a local node outside its source map");
            const std::uint32_t globalNode = localToGlobal[localNode];
            const std::uint32_t nodeOwner = globalNodeOwners[globalNode];
            if (owner != std::numeric_limits<std::uint32_t>::max() && owner != nodeOwner)
                return fail("one optic graph patch crosses bilateral source-object ownership");
            owner = nodeOwner;
            patchNodes.push_back(globalNode);
        }
        if (!std::binary_search(allowedOwners.begin(), allowedOwners.end(), owner))
            return fail("mapped optic patch has an unapproved source owner");
        const std::uint32_t patchIndex = static_cast<std::uint32_t>(
            layout.patches.size() + patches.size());
        patches.push_back({first, localPatch.nodeCount,
            static_cast<std::uint32_t>(matrixFloats),
            static_cast<std::uint32_t>(pivots)});
        for (std::uint32_t local = 0u; local < localPatch.nodeCount; ++local)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                rows.push_back({patchIndex, local, axis,
                    patchNodes[first + local]});
        matrixFloats += matrixEntries;
        pivots += pivotSlots;
        totalDofs += dofs;
        totalPatchNodes += localPatch.nodeCount;
        observedMaximumNodes = std::max(observedMaximumNodes,
                                         localPatch.nodeCount);
    }
    if (patches.empty() || patches.size() + layout.patches.size() >
            maximumPatchCount ||
        patches.size() != localLayout.patches.size() ||
        patchNodes.size() != localLayout.patchNodes.size() ||
        rows.size() != localLayout.rows.size() ||
        matrixFloats * layout.environmentCount >
            std::numeric_limits<std::uint32_t>::max() ||
        pivots * layout.environmentCount >
            std::numeric_limits<std::uint32_t>::max() ||
        totalPatchNodes != layout.patchNodes.size() + patchNodes.size())
        return fail("mapped optic graph packed extents exceed bounded device indices");

    const std::uint64_t finalPatchCount = layout.patches.size() + patches.size();
    const std::uint64_t finalPatchNodeCount =
        layout.patchNodes.size() + patchNodes.size();
    const std::uint64_t finalRowCount = layout.rows.size() + rows.size();
    std::uint64_t metadataBytes = 0u;
    std::uint64_t term = 0u;
    const auto addMetadata = [&](const std::uint64_t count,
                                 const std::uint64_t width) {
        return regionalGraphSchwarzCheckedMul(count, width, term) &&
            regionalGraphSchwarzCheckedAdd(metadataBytes, term, metadataBytes);
    };
    if (!addMetadata(finalPatchCount, sizeof(RegionalGraphSchwarzPatch)) ||
        !addMetadata(finalPatchNodeCount, sizeof(std::uint32_t)) ||
        !addMetadata(finalRowCount, sizeof(std::array<std::uint32_t, 4>)) ||
        !addMetadata(static_cast<std::uint64_t>(layout.nodeCount) + 1u,
                     sizeof(std::uint32_t)) ||
        !addMetadata(finalPatchNodeCount, sizeof(std::uint32_t)) ||
        !addMetadata(layout.nodeCount, sizeof(std::uint32_t)))
        return fail("mapped optic graph metadata byte count overflows");
    std::uint64_t matrixBytes = 0u;
    std::uint64_t pivotBytes = 0u;
    std::uint64_t correctionBytes = 0u;
    std::uint64_t validityBytes = 0u;
    std::uint64_t perEnvironment = 0u;
    std::uint64_t allEnvironments = 0u;
    std::uint64_t deviceBytes = 0u;
    if (!regionalGraphSchwarzCheckedMul(matrixFloats, sizeof(float), matrixBytes) ||
        !regionalGraphSchwarzCheckedMul(pivots, sizeof(std::uint32_t), pivotBytes) ||
        !regionalGraphSchwarzCheckedMul(finalPatchNodeCount, 4u * sizeof(float), correctionBytes) ||
        !regionalGraphSchwarzCheckedMul(finalPatchCount, sizeof(std::uint32_t), validityBytes) ||
        !regionalGraphSchwarzCheckedAdd(matrixBytes, matrixBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedAdd(perEnvironment, pivotBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedAdd(perEnvironment, correctionBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedAdd(perEnvironment, validityBytes, perEnvironment) ||
        !regionalGraphSchwarzCheckedMul(perEnvironment, layout.environmentCount,
                                        allEnvironments) ||
        !regionalGraphSchwarzCheckedAdd(allEnvironments, metadataBytes,
                                        deviceBytes) ||
        deviceBytes > maximumDeviceBytes)
        return fail("mapped optic graph exceeds the configured device-memory bound");

    std::vector<std::uint32_t> combinedPatchNodes = layout.patchNodes;
    combinedPatchNodes.insert(combinedPatchNodes.end(),
                              patchNodes.begin(), patchNodes.end());
    std::vector<std::uint32_t> coverage = layout.nodeCoverage;
    for (const std::uint32_t node : patchNodes) {
        if (coverage[node] == std::numeric_limits<std::uint32_t>::max())
            return fail("mapped optic graph normalization coverage overflows");
        ++coverage[node];
    }
    std::vector<std::uint32_t> offsets(
        static_cast<std::size_t>(layout.nodeCount) + 1u, 0u);
    for (const std::uint32_t node : combinedPatchNodes)
        ++offsets[static_cast<std::size_t>(node) + 1u];
    for (std::uint32_t node = 0u; node < layout.nodeCount; ++node)
        offsets[node + 1u] += offsets[node];
    std::vector<std::uint32_t> values(combinedPatchNodes.size());
    std::vector<std::uint32_t> cursor = offsets;
    for (std::uint32_t valueIndex = 0u;
         valueIndex < combinedPatchNodes.size(); ++valueIndex) {
        const std::uint32_t node = combinedPatchNodes[valueIndex];
        values[cursor[node]++] = valueIndex;
    }
    layout.patches.insert(layout.patches.end(), patches.begin(), patches.end());
    layout.patchNodes = std::move(combinedPatchNodes);
    layout.rows.insert(layout.rows.end(), rows.begin(), rows.end());
    layout.nodeCoverage = std::move(coverage);
    layout.nodePatchOffsets = std::move(offsets);
    layout.nodePatchValues = std::move(values);
    layout.matrixFloatCount = static_cast<std::uint32_t>(matrixFloats);
    layout.pivotCount = static_cast<std::uint32_t>(pivots);
    layout.totalPatchDofs = static_cast<std::uint32_t>(totalDofs);
    layout.metadataBytes = metadataBytes;
    layout.deviceBytes = deviceBytes;
    summary.patchCount = static_cast<std::uint32_t>(patches.size());
    summary.patchNodeSlots = static_cast<std::uint32_t>(patchNodes.size());
    summary.maximumPatchNodes = observedMaximumNodes;
    error.clear();
    return true;
}

// Preserve the historical layout byte for byte whenever it fits. Only a
// resource rejection permits a conditioning-only fallback with smaller
// patches or batched disconnected nodes. All input, index, patch-count and
// 96 MiB arena limits remain enforced by the same checked builder. This does
// not modify the mechanical operator, residual, tolerances or certificate.
[[nodiscard]] inline RegionalGraphSchwarzLayout buildBudgetedRegionalGraphSchwarzLayout(
    const std::uint32_t nodeCount,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const std::uint8_t> freeMask,
    const std::uint32_t environmentCount,
    const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull,
    const bool extendedPureFEM = false,
    const bool includeInitiallyFixedPatches = true
) {
    auto original = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, environmentCount, maximumDeviceBytes,
        4096u, 32768u, 128u, 256u, 128u, includeInitiallyFixedPatches);
    if (original.error.empty()) return original;
    const bool resourceRejection =
        original.error == "graph Schwarz patch count exceeds its bounded layout" ||
        original.error == "graph Schwarz packed device arenas, including source matrices for one refinement pass, exceed the configured device-memory bound";
    if (!resourceRejection && !extendedPureFEM) return original;
    if (resourceRejection) for (const std::uint32_t patchNodes : {256u, 128u, 64u, 32u}) {
        auto candidate = buildRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, freeMask, environmentCount,
            maximumDeviceBytes, 4096u, 32768u, 128u,
            patchNodes, patchNodes / 2u, includeInitiallyFixedPatches, true);
        if (candidate.error.empty()) return candidate;
    }
    // The caller must prove the static contact-free pure FEM scope before
    // enabling this path. The same checked byte budget is retained. Smaller
    // graph solves affect conditioning only; every residual and operator
    // certificate still uses the original full FEM equations.
    if (extendedPureFEM) {
        for (const std::uint32_t patchNodes : {256u, 128u, 64u, 32u, 16u, 8u, 4u}) {
            auto candidate = buildRegionalGraphSchwarzLayout(
                nodeCount, tetrahedra, freeMask, environmentCount,
                maximumDeviceBytes, 32768u, 131072u, 16384u,
                patchNodes, patchNodes / 2u, includeInitiallyFixedPatches, true);
            if (candidate.error.empty()) return candidate;
            if (candidate.error != "graph Schwarz patch count exceeds its bounded layout" &&
                candidate.error != "graph Schwarz packed device arenas, including source matrices for one refinement pass, exceed the configured device-memory bound")
                return candidate;
        }
    }
    return original;
}



[[nodiscard]] inline const char* regionalGraphDueRowsIsolationIneligibility(
    const RegionalGraphSchwarzLayout& layout,
    const std::span<const std::uint32_t> nodeOwners,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedronNodes,
    const std::span<const std::uint32_t> tetrahedronOwners,
    const std::span<const std::uint8_t> activeTetrahedra,
    const std::uint32_t objectCount
) noexcept {
    if (objectCount == 0u || layout.nodeCount == 0u ||
        nodeOwners.size() != layout.nodeCount ||
        tetrahedronNodes.size() != tetrahedronOwners.size() ||
        tetrahedronNodes.size() != activeTetrahedra.size())
        return "compiled_owner_extent_mismatch";

    // Mixed-owner patches are safe when active FEM element stencils never
    // cross owners: the graph builder filters each row by its node owner, so
    // every cross-owner matrix block is then exactly zero. This rejects the
    // actual coupling hazard (an active tet spanning owners), not mere patch
    // cohabitation.
    for (const RegionalGraphSchwarzPatch& patch : layout.patches) {
        const std::uint64_t end =
            static_cast<std::uint64_t>(patch.nodeOffset) + patch.nodeCount;
        if (patch.nodeCount == 0u || end > layout.patchNodes.size())
            return "graph_patch_node_range_invalid";
        for (std::uint32_t local = 0u; local < patch.nodeCount; ++local) {
            const std::uint32_t node = layout.patchNodes[patch.nodeOffset + local];
            if (node >= nodeOwners.size()) return "graph_patch_node_invalid";
            if (nodeOwners[node] >= objectCount)
                return "graph_patch_owner_invalid";
        }
    }

    for (std::size_t tetIndex = 0u; tetIndex < tetrahedronNodes.size();
         ++tetIndex) {
        if (activeTetrahedra[tetIndex] == 0u) continue;
        const std::uint32_t owner = tetrahedronOwners[tetIndex];
        if (owner >= objectCount) return "active_tet_owner_invalid";
        for (const std::uint32_t node : tetrahedronNodes[tetIndex]) {
            if (node >= nodeOwners.size()) return "active_tet_node_invalid";
            if (nodeOwners[node] != owner)
                return "active_tet_crosses_object_owners";
        }
    }
    return nullptr;
}

} // namespace numi::matter::detail
