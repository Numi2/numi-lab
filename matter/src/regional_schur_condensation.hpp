#pragma once

#include "regional_graph_schwarz.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
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

// Partition each existing graph patch into a true element-closed interior
// and an interface. A patch node remains on the interface if it is touched by
// any non-FEM term, crosses a compiled region label, or has a tetrahedral
// neighbor outside that patch. Therefore every element contribution in an
// interior row is present in the patch principal matrix already assembled by
// nm_fgmres_graph_build_patch_matrices. This is only exact if the caller's
// externalCouplingAnchors is the complete endpoint set for all non-tet A
// entries; otherwise this function must not be admitted.
// Expand a candidate dynamic FEM surface face to all nodes of every
// adjacent tetrahedron. Contact can activate at any point on a compiled
// surface, so anchoring only the initial contact endpoints or face vertices
// is not enough to keep the exact contact Jacobian out of A_ii.
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
                    if (member[neighbor] == 0u ||
                        owner[neighbor] != patchIndex) isInterface = true;
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
