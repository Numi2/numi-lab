#pragma once

#include "regional_graph_schwarz.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace numi::matter::detail {

// Structural partition for exact block Gaussian elimination. A node is
// interior only when every incident tetrahedron has the same source region
// label and the caller proves it is absent from every non-FEM/cross-region
// coupling. The external anchor list must therefore contain every endpoint
// touched by attachment, contact, cavity, vascular, rigid, boundary, or other
// non-element matrix entries. Missing that proof is an admission error, not a
// reason to silently omit a coupling from the Schur complement.
struct RegionalSchurPatchRange {
    std::uint32_t interiorOffset = 0u;
    std::uint32_t interiorCount = 0u;
    std::uint32_t interfaceOffset = 0u;
    std::uint32_t interfaceCount = 0u;
    bool eligible = false;
};

struct RegionalSchurGraphLayout {
    std::vector<RegionalSchurPatchRange> patches;
    // Values index graph.patchNodes. This keeps the Schur permutation adjacent
    // to the existing graph patch storage without duplicating global IDs.
    std::vector<std::uint32_t> interiorPatchValues;
    std::vector<std::uint32_t> interfacePatchValues;
    std::string error;
    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

[[nodiscard]] inline RegionalSchurGraphLayout buildRegionalSchurGraphLayout(
    const RegionalGraphSchwarzLayout& graph,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const std::array<std::uint32_t, 2>> tetrahedronSources,
    const std::span<const std::uint32_t> externalCouplingAnchors
);

// An element-domain partition for exact local elimination. Unlike the
// overlapping BFS Schwarz graph, each tetrahedron is assigned to exactly one
// deterministic source-homogeneous core. A patch contains the union of its
// core tetrahedra; nodes shared by cores remain interface values. A node can
// be interior only when every incident tetrahedron belongs to the same core
// and the node is absent from the caller's complete non-FEM coupling anchor
// set. The resulting A_ii is therefore assembled from the exact current FEM
// tangent, while every cross-core or external coupling remains global.
struct RegionalSchurElementCoreLayout {
    RegionalGraphSchwarzLayout graph;
    RegionalSchurGraphLayout partition;
    // Extraction descriptors: {graph patch, interior-value offset,
    // interior node count, destination dense-matrix float offset}.
    std::vector<std::array<std::uint32_t, 4>> extractionSources;
    // Global FEM node IDs for the disjoint interior solve vectors.
    std::vector<std::uint32_t> interiorGlobalNodes;
    std::uint32_t interiorMatrixFloatCount = 0u;
    std::string error;
    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

[[nodiscard]] inline RegionalSchurElementCoreLayout
buildRegionalSchurElementCoreLayoutAttempt(
    const std::uint32_t nodeCount,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const std::array<std::uint32_t, 2>> tetrahedronSources,
    const std::span<const std::uint8_t> sourceFreeMask,
    const std::span<const std::uint32_t> externalCouplingAnchors,
    const std::uint32_t environmentCount,
    const std::uint32_t maximumPatchNodes = 64u,
    const std::uint32_t maximumPatchCount = 16384u,
    const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull
) {
    RegionalSchurElementCoreLayout out;
    const auto fail = [&](const char* message) {
        out.error = message;
        return out;
    };
    constexpr std::uint32_t invalid =
        std::numeric_limits<std::uint32_t>::max();
    if (nodeCount == 0u || tetrahedra.empty() ||
        tetrahedra.size() != tetrahedronSources.size() ||
        sourceFreeMask.size() != nodeCount || environmentCount == 0u ||
        environmentCount > 4u || maximumPatchNodes < 4u ||
        maximumPatchNodes > 256u || maximumPatchCount == 0u)
        return fail("Schur element cores exceed their explicit input bounds");

    std::vector<std::vector<std::uint32_t>> incident(nodeCount);
    std::map<std::array<std::uint32_t, 2>, std::vector<std::uint32_t>> bySource;
    for (std::uint32_t tetIndex = 0u; tetIndex < tetrahedra.size(); ++tetIndex) {
        const auto& tet = tetrahedra[tetIndex];
        const auto source = tetrahedronSources[tetIndex];
        if (source[0] == invalid || source[1] == invalid)
            return fail("Schur element core has an invalid source label");
        for (std::uint32_t i = 0u; i < 4u; ++i) {
            if (tet[i] >= nodeCount)
                return fail("Schur element core tet references an invalid node");
            for (std::uint32_t j = 0u; j < i; ++j)
                if (tet[i] == tet[j])
                    return fail("Schur element core tet repeats a node");
            incident[tet[i]].push_back(tetIndex);
        }
        bySource[source].push_back(tetIndex);
    }
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        if (sourceFreeMask[node] > 1u)
            return fail("Schur element core free mask is not binary");
        if (incident[node].empty())
            return fail("Schur element core cannot classify an unreferenced node");
    }

    // Constrained coordinates are not part of the reduced solve. Keep them
    // on the global interface so the production mask and full operator remain
    // the only owners of fixed-coordinate semantics.
    std::vector<std::uint8_t> anchored(nodeCount, 0u);
    for (const std::uint32_t node : externalCouplingAnchors) {
        if (node >= nodeCount)
            return fail("Schur element core external anchor is out of range");
        anchored[node] = 1u;
    }
    std::vector<std::uint32_t> allAnchors(
        externalCouplingAnchors.begin(), externalCouplingAnchors.end());
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if (sourceFreeMask[node] == 0u) anchored[node] = 1u;
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if (sourceFreeMask[node] == 0u) allAnchors.push_back(node);

    std::vector<std::uint8_t> assigned(tetrahedra.size(), 0u);
    std::vector<std::vector<std::uint32_t>> coreNodes;
    for (const auto& [source, sourceTets] : bySource) {
        (void)source;
        for (const std::uint32_t seed : sourceTets) {
            if (assigned[seed] != 0u) continue;
            std::vector<std::uint32_t> selectedTets;
            std::vector<std::uint32_t> selectedNodes;
            std::vector<std::uint8_t> selectedNode(nodeCount, 0u);
            std::vector<std::uint8_t> queued(tetrahedra.size(), 0u);
            std::vector<std::uint32_t> frontier{seed};
            queued[seed] = 1u;
            for (std::size_t cursor = 0u; cursor < frontier.size(); ++cursor) {
                const std::uint32_t tetIndex = frontier[cursor];
                if (assigned[tetIndex] != 0u) continue;
                std::uint32_t newNodes = 0u;
                for (const std::uint32_t node : tetrahedra[tetIndex])
                    if (selectedNode[node] == 0u) ++newNodes;
                if (selectedNodes.size() + newNodes > maximumPatchNodes) continue;
                selectedTets.push_back(tetIndex);
                assigned[tetIndex] = 1u;
                for (const std::uint32_t node : tetrahedra[tetIndex]) {
                    if (selectedNode[node] != 0u) continue;
                    selectedNode[node] = 1u;
                    selectedNodes.push_back(node);
                }
                for (const std::uint32_t node : tetrahedra[tetIndex])
                    for (const std::uint32_t neighborTet : incident[node])
                        if (assigned[neighborTet] == 0u &&
                            queued[neighborTet] == 0u &&
                            tetrahedronSources[neighborTet] ==
                                tetrahedronSources[tetIndex]) {
                            queued[neighborTet] = 1u;
                            frontier.push_back(neighborTet);
                        }
            }
            if (selectedTets.empty())
                return fail("Schur element core could not admit its seed tet");
            std::sort(selectedNodes.begin(), selectedNodes.end());
            coreNodes.push_back(std::move(selectedNodes));
            if (coreNodes.size() > maximumPatchCount)
                return fail("Schur element core patch count exceeds its bound");
        }
    }
    if (std::any_of(assigned.begin(), assigned.end(),
                    [](const std::uint8_t value) { return value == 0u; }))
        return fail("Schur element cores do not assign every tetrahedron");

    RegionalGraphSchwarzLayout& graph = out.graph;
    graph.nodeCount = nodeCount;
    graph.environmentCount = environmentCount;
    std::uint64_t matrixFloatCount = 0u;
    std::uint64_t pivotCount = 0u;
    std::uint64_t totalPatchDofs = 0u;
    std::uint64_t totalPatchNodes = 0u;
    graph.patches.reserve(coreNodes.size());
    for (const std::vector<std::uint32_t>& nodes : coreNodes) {
        const std::uint64_t dofs = 3u * nodes.size();
        const std::uint64_t entries = dofs * dofs;
        if (nodes.empty() || nodes.size() > maximumPatchNodes ||
            matrixFloatCount + entries > std::numeric_limits<std::uint32_t>::max() ||
            pivotCount + dofs > std::numeric_limits<std::uint32_t>::max() ||
            totalPatchDofs + dofs > std::numeric_limits<std::uint32_t>::max() ||
            totalPatchNodes + nodes.size() > std::numeric_limits<std::uint32_t>::max())
            return fail("Schur element core packed offsets overflow");
        graph.patches.push_back({
            static_cast<std::uint32_t>(totalPatchNodes),
            static_cast<std::uint32_t>(nodes.size()),
            static_cast<std::uint32_t>(matrixFloatCount),
            static_cast<std::uint32_t>(pivotCount)});
        graph.patchNodes.insert(graph.patchNodes.end(), nodes.begin(), nodes.end());
        matrixFloatCount += entries;
        pivotCount += dofs;
        totalPatchDofs += dofs;
        totalPatchNodes += nodes.size();
    }
    graph.matrixFloatCount = static_cast<std::uint32_t>(matrixFloatCount);
    graph.pivotCount = static_cast<std::uint32_t>(pivotCount);
    graph.totalPatchDofs = static_cast<std::uint32_t>(totalPatchDofs);
    graph.nodeCoverage.assign(nodeCount, 0u);
    for (const std::uint32_t node : graph.patchNodes)
        ++graph.nodeCoverage[node];
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if (graph.nodeCoverage[node] == 0u)
            return fail("Schur element core graph omits an FEM node");

    graph.rows.reserve(graph.totalPatchDofs);
    for (std::uint32_t patchIndex = 0u;
         patchIndex < graph.patches.size(); ++patchIndex) {
        const RegionalGraphSchwarzPatch patch = graph.patches[patchIndex];
        for (std::uint32_t localNode = 0u; localNode < patch.nodeCount; ++localNode)
            for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                graph.rows.push_back({patchIndex, localNode, axis,
                    graph.patchNodes[patch.nodeOffset + localNode]});
    }
    graph.nodePatchOffsets.assign(static_cast<std::size_t>(nodeCount) + 1u, 0u);
    for (const std::uint32_t node : graph.patchNodes)
        ++graph.nodePatchOffsets[static_cast<std::size_t>(node) + 1u];
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        graph.nodePatchOffsets[node + 1u] += graph.nodePatchOffsets[node];
    graph.nodePatchValues.resize(graph.patchNodes.size());
    std::vector<std::uint32_t> cursor = graph.nodePatchOffsets;
    for (std::uint32_t value = 0u; value < graph.patchNodes.size(); ++value)
        graph.nodePatchValues[cursor[graph.patchNodes[value]]++] = value;

    // The bounded arena includes both per-environment work buffers and the
    // shared uploaded graph descriptors. Omitting rows/CSR/coverage made this
    // admission estimate smaller than the buffers the runtime actually owns.
    const std::uint64_t metadataBytes =
        graph.patches.size() * sizeof(RegionalGraphSchwarzPatch) +
        graph.patchNodes.size() * sizeof(std::uint32_t) +
        graph.rows.size() * sizeof(std::array<std::uint32_t, 4>) +
        graph.nodePatchOffsets.size() * sizeof(std::uint32_t) +
        graph.nodePatchValues.size() * sizeof(std::uint32_t) +
        graph.nodeCoverage.size() * sizeof(std::uint32_t);
    graph.metadataBytes = metadataBytes;
    const std::uint64_t bytes = matrixFloatCount * sizeof(float) * 2u +
        pivotCount * sizeof(std::uint32_t) +
        totalPatchNodes * sizeof(float) * 4u +
        coreNodes.size() * sizeof(std::uint32_t);
    const std::uint64_t perEnvironmentBytes = bytes * environmentCount;
    if (perEnvironmentBytes > maximumDeviceBytes ||
        metadataBytes > maximumDeviceBytes - perEnvironmentBytes) {
        graph.deviceBytes = perEnvironmentBytes + metadataBytes;
        return fail("Schur element core device arenas exceed their byte bound");
    }
    graph.deviceBytes = perEnvironmentBytes + metadataBytes;

    out.partition = buildRegionalSchurGraphLayout(
        graph, tetrahedra, tetrahedronSources, allAnchors);
    if (!out.partition.valid()) return fail("Schur element-core closure failed");
    std::uint64_t interiorMatrixFloatCount = 0u;
    for (std::uint32_t patch = 0u; patch < out.partition.patches.size(); ++patch) {
        const RegionalSchurPatchRange range = out.partition.patches[patch];
        if (range.interiorCount == 0u) continue;
        const std::uint64_t dofs = 3u * range.interiorCount;
        const std::uint64_t entries = dofs * dofs;
        if (interiorMatrixFloatCount + entries >
            std::numeric_limits<std::uint32_t>::max())
            return fail("Schur interior matrix offsets overflow");
        out.extractionSources.push_back({patch, range.interiorOffset,
            range.interiorCount,
            static_cast<std::uint32_t>(interiorMatrixFloatCount)});
        interiorMatrixFloatCount += entries;
    }
    out.interiorGlobalNodes.reserve(
        out.partition.interiorPatchValues.size());
    for (const std::uint32_t patchValue :
            out.partition.interiorPatchValues)
        out.interiorGlobalNodes.push_back(graph.patchNodes[patchValue]);
    if (out.extractionSources.empty())
        return fail("Schur element cores contain no eliminable interior nodes");
    out.interiorMatrixFloatCount =
        static_cast<std::uint32_t>(interiorMatrixFloatCount);
    const std::uint64_t interiorVectorBytes =
        static_cast<std::uint64_t>(out.interiorGlobalNodes.size()) *
        (sizeof(std::uint32_t) + sizeof(float) * 4u);
    const std::uint64_t schurMatrixBytes = interiorMatrixFloatCount *
        (2u * sizeof(float)) +
        static_cast<std::uint64_t>(out.interiorGlobalNodes.size()) *
            3u * sizeof(std::uint32_t);
    const std::uint64_t totalDeviceBytes = graph.deviceBytes +
        (interiorVectorBytes + schurMatrixBytes) * environmentCount;
    graph.deviceBytes = totalDeviceBytes;
    if (totalDeviceBytes > maximumDeviceBytes)
        return fail("Schur element core plus A_ii arenas exceed their byte bound");
    return out;
}

// Preserve the requested core size when it fits. If dense core and A_ii arenas
// exceed the same explicit byte bound, retry a deterministic descending set
// of smaller node caps. Every retry repartitions all tetrahedra from scratch,
// then recomputes exact element/interface closure; it never truncates a core,
// drops an element, or relaxes the memory budget. Smaller cores increase the
// global interface and may make an interior ineligible, in which case the
// selector fails closed after bounded retries.
[[nodiscard]] inline RegionalSchurElementCoreLayout
buildRegionalSchurElementCoreLayout(
    const std::uint32_t nodeCount,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const std::array<std::uint32_t, 2>> tetrahedronSources,
    const std::span<const std::uint8_t> sourceFreeMask,
    const std::span<const std::uint32_t> externalCouplingAnchors,
    const std::uint32_t environmentCount,
    const std::uint32_t maximumPatchNodes = 64u,
    const std::uint32_t maximumPatchCount = 16384u,
    const std::uint64_t maximumDeviceBytes = 96ull * 1024ull * 1024ull
) {
    std::uint32_t patchNodeLimit = maximumPatchNodes;
    RegionalSchurElementCoreLayout lastFailure;
    for (std::uint32_t attempt = 0u; attempt < 16u; ++attempt) {
        RegionalSchurElementCoreLayout candidate =
            buildRegionalSchurElementCoreLayoutAttempt(
                nodeCount, tetrahedra, tetrahedronSources, sourceFreeMask,
                externalCouplingAnchors, environmentCount, patchNodeLimit,
                maximumPatchCount, maximumDeviceBytes);
        if (candidate.valid()) return candidate;

        const std::string error = candidate.error;
        const std::uint64_t estimatedBytes = candidate.graph.deviceBytes;
        candidate = {}; // release the failed attempt's dense CPU descriptors
        lastFailure.error = error;
        lastFailure.graph.deviceBytes = estimatedBytes;

        const bool retryable =
            error == "Schur element core device arenas exceed their byte bound" ||
            error == "Schur element core plus A_ii arenas exceed their byte bound" ||
            error == "Schur element cores contain no eliminable interior nodes";
        if (!retryable || patchNodeLimit <= 4u) return lastFailure;

        std::uint32_t nextLimit = 4u;
        if (patchNodeLimit > 64u) {
            nextLimit = (patchNodeLimit * 3u / 4u) & ~3u;
            nextLimit = std::max(64u, nextLimit);
        } else if (patchNodeLimit > 16u) {
            nextLimit = patchNodeLimit - 8u;
        } else {
            nextLimit = (patchNodeLimit / 2u) & ~3u;
            nextLimit = std::max(4u, nextLimit);
        }
        if (nextLimit >= patchNodeLimit) nextLimit = patchNodeLimit - 4u;
        patchNodeLimit = std::max(4u, nextLimit);
    }
    return lastFailure;
}

// Partition each existing graph patch into a true element-closed interior
// and an interface. A patch node remains on the interface if it is touched by
// any non-FEM term, crosses a compiled region label, or has a tetrahedral
// neighbor outside that patch. Therefore every element contribution in an
// interior row is present in the patch principal matrix already assembled by
// nm_fgmres_graph_build_patch_matrices. This is only exact if the caller's
// externalCouplingAnchors is the complete endpoint set for all non-tet A
// entries; otherwise this function must not be admitted.
// The static surface primitive emitter reads only adjacency.x and sides.x,
// then emits the three vertices opposite sides.x. A contact feature (including
// a clipped/mortar feature) is mapped back onto those primitive vertices, so
// these are the complete potential FEM contact endpoints. Keep both adjacency
// sides in the descriptor to mirror NMFEMSurfaceFaceGPU, but deliberately
// consume only the side the emitter actually uses. Disabled faces are skipped
// before validating their unused geometry, matching the emitter's early exit.
struct RegionalSchurSurfaceFace {
    std::array<std::uint32_t, 2> adjacency{};
    std::array<std::uint32_t, 2> oppositeCorners{};
    std::uint32_t flags = 0u;
};

[[nodiscard]] inline bool regionalSchurContactTopologyIsImmutable(
    const bool mutationCommandsEmpty,
    const bool punctureChannelsEmpty
) noexcept {
    return mutationCommandsEmpty && punctureChannelsEmpty;
}

[[nodiscard]] inline bool appendRegionalSchurSurfaceVertexAnchors(
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const RegionalSchurSurfaceFace> surfaceFaces,
    const std::uint32_t disabledContactFlag,
    std::vector<std::uint32_t>& anchors
) {
    constexpr std::uint32_t invalid =
        std::numeric_limits<std::uint32_t>::max();
    if (disabledContactFlag == 0u) return false;
    std::vector<std::uint32_t> emitted;
    if (surfaceFaces.size() > std::numeric_limits<std::size_t>::max() / 3u)
        return false;
    emitted.reserve(surfaceFaces.size() * 3u);
    for (const RegionalSchurSurfaceFace& face : surfaceFaces) {
        if ((face.flags & disabledContactFlag) != 0u) continue;
        const std::uint32_t tetIndex = face.adjacency[0];
        const std::uint32_t opposite = face.oppositeCorners[0];
        if (tetIndex == invalid || tetIndex >= tetrahedra.size() ||
            opposite >= 4u) return false;
        for (std::uint32_t corner = 0u; corner < 4u; ++corner)
            if (corner != opposite)
                emitted.push_back(tetrahedra[tetIndex][corner]);
    }
    anchors.insert(anchors.end(), emitted.begin(), emitted.end());
    return true;
}

// Conservative fallback for non-compact/dynamic surface maps. That path can
// enumerate tet faces from rebuilt incidence, so anchor the full adjacent
// tetrahedra unless the immutable emitted-triangle map is available.
[[nodiscard]] inline bool appendRegionalSchurSurfaceTetAnchors(
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const std::array<std::uint32_t, 2>> surfaceAdjacencies,
    std::vector<std::uint32_t>& anchors
) {
    constexpr std::uint32_t invalid =
        std::numeric_limits<std::uint32_t>::max();
    for (const auto& adjacency : surfaceAdjacencies) {
        for (const std::uint32_t tetIndex : adjacency) {
            if (tetIndex == invalid) continue;
            if (tetIndex >= tetrahedra.size()) return false;
            for (const std::uint32_t node : tetrahedra[tetIndex])
                anchors.push_back(node);
        }
    }
    return true;
}

[[nodiscard]] inline RegionalSchurGraphLayout buildRegionalSchurGraphLayout(
    const RegionalGraphSchwarzLayout& graph,
    const std::span<const std::array<std::uint32_t, 4>> tetrahedra,
    const std::span<const std::array<std::uint32_t, 2>> tetrahedronSources,
    const std::span<const std::uint32_t> externalCouplingAnchors
) {
    RegionalSchurGraphLayout out;
    const auto fail = [&](const char* message) {
        out.error = message;
        return out;
    };
    constexpr std::uint32_t invalid =
        std::numeric_limits<std::uint32_t>::max();
    if (graph.nodeCount == 0u || graph.patches.empty() ||
        graph.patchNodes.empty() || tetrahedra.empty() ||
        graph.patchNodes.size() > std::numeric_limits<std::uint32_t>::max() ||
        tetrahedra.size() > std::numeric_limits<std::uint32_t>::max() ||
        tetrahedra.size() != tetrahedronSources.size())
        return fail("Schur graph requires exact tetrahedron labels and nonempty patches");

    // Source identity includes the owning FEM object and compiled region. A
    // region tag alone is insufficient when two object-local operators happen
    // to reuse the same tag value.
    std::vector<std::array<std::uint32_t, 2>> nodeSource(
        graph.nodeCount, std::array<std::uint32_t, 2>{invalid, invalid});
    std::vector<std::uint8_t> ambiguous(graph.nodeCount, 0u);
    std::vector<std::uint8_t> incident(graph.nodeCount, 0u);
    std::vector<std::uint8_t> external(graph.nodeCount, 0u);
    std::vector<std::vector<std::uint32_t>> incidentTetrahedra(graph.nodeCount);
    for (std::uint32_t tetIndex = 0u; tetIndex < tetrahedra.size(); ++tetIndex) {
        const auto& tet = tetrahedra[tetIndex];
        const auto source = tetrahedronSources[tetIndex];
        if (source[0] == invalid || source[1] == invalid)
            return fail("Schur graph contains an invalid object or region source");
        for (std::uint32_t i = 0u; i < 4u; ++i) {
            if (tet[i] >= graph.nodeCount)
                return fail("Schur graph tetrahedron references an invalid node");
            for (std::uint32_t j = 0u; j < i; ++j)
                if (tet[i] == tet[j])
                    return fail("Schur graph tetrahedron repeats a node");
            incident[tet[i]] = 1u;
            incidentTetrahedra[tet[i]].push_back(tetIndex);
            if (nodeSource[tet[i]][0] == invalid) nodeSource[tet[i]] = source;
            else if (nodeSource[tet[i]] != source) ambiguous[tet[i]] = 1u;
        }
    }
    for (std::uint32_t node = 0u; node < graph.nodeCount; ++node)
        if (incident[node] == 0u)
            return fail("Schur graph cannot classify an unreferenced FEM node");
    for (const std::uint32_t node : externalCouplingAnchors) {
        if (node >= graph.nodeCount || incident[node] == 0u)
            return fail("Schur graph external anchor is not an active FEM node");
        external[node] = 1u;
    }

    // Graph patches overlap for additive Schwarz. Assign each FEM node to
    // its first deterministic patch owner; an element crossing owners makes
    // every participating node an interface value. Interior DOFs are thus
    // disjoint between patches and every incident tet is contained in one.
    std::vector<std::uint32_t> owner(graph.nodeCount, invalid);
    for (std::uint32_t patchIndex = 0u;
         patchIndex < graph.patches.size(); ++patchIndex) {
        const RegionalGraphSchwarzPatch& patch = graph.patches[patchIndex];
        if (patch.nodeCount == 0u || patch.nodeOffset > graph.patchNodes.size() ||
            patch.nodeCount > graph.patchNodes.size() - patch.nodeOffset)
            return fail("Schur graph patch range is invalid");
        std::vector<std::uint8_t> unique(graph.nodeCount, 0u);
        for (std::uint32_t local = 0u; local < patch.nodeCount; ++local) {
            const std::uint32_t node = graph.patchNodes[patch.nodeOffset + local];
            if (node >= graph.nodeCount || unique[node] != 0u)
                return fail("Schur graph patch contains an invalid or repeated node");
            unique[node] = 1u;
            if (owner[node] == invalid) owner[node] = patchIndex;
        }
    }
    if (std::any_of(owner.begin(), owner.end(),
                    [](const std::uint32_t value) {
                        return value == invalid;
                    }))
        return fail("Schur graph patches do not cover every FEM node");

    out.patches.reserve(graph.patches.size());
    for (std::uint32_t patchIndex = 0u;
         patchIndex < graph.patches.size(); ++patchIndex) {
        const RegionalGraphSchwarzPatch& patch = graph.patches[patchIndex];
        if (patch.nodeCount == 0u || patch.nodeOffset > graph.patchNodes.size() ||
            patch.nodeCount > graph.patchNodes.size() - patch.nodeOffset)
            return fail("Schur graph patch range is invalid");
        std::vector<std::uint8_t> member(graph.nodeCount, 0u);
        for (std::uint32_t local = 0u; local < patch.nodeCount; ++local) {
            const std::uint32_t node = graph.patchNodes[patch.nodeOffset + local];
            if (node >= graph.nodeCount || member[node] != 0u)
                return fail("Schur graph patch contains an invalid or repeated node");
            member[node] = 1u;
        }
        RegionalSchurPatchRange range;
        range.interiorOffset = static_cast<std::uint32_t>(out.interiorPatchValues.size());
        range.interfaceOffset = static_cast<std::uint32_t>(out.interfacePatchValues.size());
        for (std::uint32_t local = 0u; local < patch.nodeCount; ++local) {
            const std::uint32_t node = graph.patchNodes[patch.nodeOffset + local];
            bool isInterface = external[node] != 0u || ambiguous[node] != 0u ||
                owner[node] != patchIndex;
            for (const std::uint32_t tetIndex : incidentTetrahedra[node]) {
                const auto& tet = tetrahedra[tetIndex];
                if (tetrahedronSources[tetIndex] != nodeSource[node])
                    isInterface = true;
                for (const std::uint32_t neighbor : tet)
                    if (member[neighbor] == 0u) isInterface = true;
            }
            const std::uint32_t patchValue = patch.nodeOffset + local;
            if (isInterface) out.interfacePatchValues.push_back(patchValue);
            else out.interiorPatchValues.push_back(patchValue);
        }
        range.interiorCount = static_cast<std::uint32_t>(
            out.interiorPatchValues.size()) - range.interiorOffset;
        range.interfaceCount = static_cast<std::uint32_t>(
            out.interfacePatchValues.size()) - range.interfaceOffset;
        range.eligible = range.interiorCount != 0u && range.interfaceCount != 0u;
        if (range.interiorCount + range.interfaceCount != patch.nodeCount)
            return fail("Schur graph partition does not cover a patch exactly once");
        out.patches.push_back(range);
    }
    const bool anyEligible = std::any_of(out.patches.begin(), out.patches.end(),
        [](const RegionalSchurPatchRange& range) { return range.eligible; });
    if (!anyEligible)
        return fail("Schur graph has no patch with both interior and interface unknowns");
    return out;
}

} // namespace numi::matter::detail
