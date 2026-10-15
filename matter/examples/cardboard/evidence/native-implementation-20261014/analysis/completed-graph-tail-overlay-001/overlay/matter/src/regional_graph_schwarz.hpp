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
// rebuilds and selected diagnostic captures. The Metal kernels additionally
// require that the environment's nonlinear-completion latch is set.
[[nodiscard]] inline bool regionalGraphCompletedTailReuseAllowed(
    const std::uint32_t solverIteration,
    const bool forceRebuild,
    const bool selectedCapturePending) noexcept {
    return solverIteration > 0u && !forceRebuild &&
        !selectedCapturePending;
}

[[nodiscard]] inline bool regionalGraphMaySkipPatchFactorization(
    const bool completedTailReuseAllowed,
    const bool nonlinearCompletionLatch,
    const bool priorFactorValid) noexcept {
    return completedTailReuseAllowed && nonlinearCompletionLatch &&
        priorFactorValid;
}

// A bounded host-built graph layout for the selector-5 additive Schwarz
// preconditioner. The graph is derived from the exact active tetrahedron
// connectivity and a caller-supplied topology-coverage mask. Selector 5
// supplies an all-potential-node mask so later FEM target transactions may
// change which covered nodes are constrained. Patches are ordered
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
    // 1 means a node is included in the immutable graph topology; 0 excludes
    // it. Selector 5 may pass an all-node potential-free mask so later FEM
    // target transactions can change which covered nodes are constrained.
    // Every currently free candidate node must be included in this mask.
    const std::span<const std::uint8_t> freeMask,
    const std::uint32_t environmentCount,
    const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull,
    const std::uint32_t maximumNodeCount = 4096u,
    const std::uint32_t maximumTetrahedronCount = 32768u,
    const std::uint32_t maximumPatchCount = 128u,
    const std::uint32_t maximumPatchNodes = 256u,
    const std::uint32_t anchorStride = 128u
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
    std::vector<std::uint32_t> freeNodes;
    freeNodes.reserve(nodeCount);
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        if (freeMask[node] > 1u)
            return fail("graph Schwarz free mask is not binary");
        if (freeMask[node] != 0u) freeNodes.push_back(node);
    }
    if (freeNodes.empty())
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
            if (freeMask[tet[i]] == 0u) continue;
            for (std::uint32_t j = 0u; j < 4u; ++j) {
                if (i != j && freeMask[tet[j]] != 0u)
                    adjacency[tet[i]].insert(tet[j]);
            }
        }
    }

    std::vector<std::uint32_t> anchors;
    for (std::size_t index = 0u; index < freeNodes.size(); index += anchorStride)
        anchors.push_back(freeNodes[index]);
    if (anchors.empty() || anchors.back() != freeNodes.back())
        anchors.push_back(freeNodes.back());

    std::vector<std::vector<std::uint32_t>> nodePatches;
    nodePatches.reserve(anchors.size());
    std::vector<std::uint8_t> covered(nodeCount, 0u);
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
    for (const std::uint32_t node : freeNodes) {
        if (covered[node] == 0u) nodePatches.push_back({node});
    }
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
        if (!regionalGraphSchwarzCheckedMul(dofs, dofs, matrixEntries) ||
            matrixFloatCount > std::numeric_limits<std::uint32_t>::max() - matrixEntries ||
            pivotCount > std::numeric_limits<std::uint32_t>::max() - dofs ||
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
        pivotCount += dofs;
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
        if (node >= nodeCount || freeMask[node] == 0u)
            return fail("graph Schwarz patches include a constrained or invalid node");
        ++out.nodeCoverage[node];
    }
    for (const std::uint32_t node : freeNodes)
        if (out.nodeCoverage[node] == 0u)
            return fail("graph Schwarz patches do not cover every free node");
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if (freeMask[node] == 0u && out.nodeCoverage[node] != 0u)
            return fail("graph Schwarz patches cover a constrained node");

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
        return fail("graph Schwarz packed device arenas, including source matrices for one refinement pass, exceed the explicit 96 MiB bound");
    }
    return out;
}

} // namespace numi::matter::detail
