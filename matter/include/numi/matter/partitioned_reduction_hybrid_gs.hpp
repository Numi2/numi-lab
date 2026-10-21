#pragma once

#include "numi/matter/partitioned_reduction_fem_tangent.h"

typedef struct NM_ALIGN16 NMPartitionedReductionHybridGSTermGPU {
    nm_u32 supportIndex;
    nm_u32 region;
    nm_u32 localCorner;
    nm_u32 reserved;
} NMPartitionedReductionHybridGSTermGPU;

#ifndef __METAL_VERSION__
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace numi::matter {

// One sparse link from a fine-complement global FEM node to one regional
// element-mode action. The action buffer is indexed by supportIndex and mode.
static_assert(sizeof(NMPartitionedReductionHybridGSTermGPU) == 16u);

struct PartitionedReductionHybridGSSupport {
    std::vector<NMPartitionedReductionSparseRangeGPU> nodeTermRanges;
    std::vector<NMPartitionedReductionHybridGSTermGPU> nodeTerms;
    std::uint32_t fineComplementNodeCount = 0u;
    std::uint32_t maximumNodeTermCount = 0u;
    std::string error;
    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

// Build a deterministic node-to-element reverse map for the hybrid
// triangular preconditioner. Only nodes with no active xyz basis coefficient
// are included, so q-to-fine corrections cannot alter selected coordinates.
// The physical operator and its certification continue to use the full mesh.
[[nodiscard]] inline PartitionedReductionHybridGSSupport
buildPartitionedReductionHybridGSSupport(
    std::span<const nm_float4> basisValues,
    std::span<const std::uint32_t> globalFEMNodeIndices,
    std::uint32_t globalNodeCount,
    std::uint32_t basisNodeCount,
    std::uint32_t modeCount,
    const PartitionedReductionFEMTangentSupport& tangentSupport,
    std::uint32_t regionCount,
    std::span<const NMTetrahedronGPU> tetrahedra) {
    PartitionedReductionHybridGSSupport out;
    auto fail = [&](const char* message) {
        out.error = message;
        out.nodeTermRanges.clear();
        out.nodeTerms.clear();
        out.fineComplementNodeCount = 0u;
        out.maximumNodeTermCount = 0u;
        return out;
    };
    if (globalNodeCount == 0u || basisNodeCount == 0u || modeCount == 0u ||
        regionCount == 0u || globalFEMNodeIndices.size() != basisNodeCount ||
        basisValues.size() != std::size_t(basisNodeCount) * modeCount ||
        tangentSupport.regionTetrahedronRanges.size() != regionCount ||
        tangentSupport.tetrahedronBasisNodeRows.size() !=
            tetrahedra.size() * 4u ||
        !tangentSupport.valid())
        return fail("hybrid GS support dimensions mismatch");
    for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
        if (globalFEMNodeIndices[row] >= globalNodeCount ||
            (row != 0u && globalFEMNodeIndices[row - 1u] >=
                globalFEMNodeIndices[row]))
            return fail("hybrid GS global node map is invalid");
    }
    std::vector<std::uint8_t> selected(basisNodeCount, 0u);
    for (std::uint32_t mode = 0u; mode < modeCount; ++mode) {
        const std::size_t base = std::size_t(mode) * basisNodeCount;
        for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
            const auto value = basisValues[base + row];
            if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                !std::isfinite(value.z))
                return fail("hybrid GS basis contains non-finite xyz");
            selected[row] |= static_cast<std::uint8_t>(
                value.x != 0.0f || value.y != 0.0f || value.z != 0.0f);
        }
    }
    std::vector<std::vector<NMPartitionedReductionHybridGSTermGPU>> byNode(
        globalNodeCount);
    const auto& ranges = tangentSupport.regionTetrahedronRanges;
    const auto& supportTets = tangentSupport.regionTetrahedronIndices;
    for (std::uint32_t region = 0u; region < regionCount; ++region) {
        const auto range = ranges[region];
        if (range.first > supportTets.size() ||
            range.count > supportTets.size() - range.first)
            return fail("hybrid GS regional tet range is out of bounds");
        for (std::uint32_t local = 0u; local < range.count; ++local) {
            const std::uint32_t supportIndex = range.first + local;
            const std::uint32_t tetIndex = supportTets[supportIndex];
            if (tetIndex >= tetrahedra.size())
                return fail("hybrid GS support tet index is out of bounds");
            const auto& tet = tetrahedra[tetIndex];
            if ((tet.identity.w & NM_OBJECT_ACTIVE) == 0u)
                return fail("hybrid GS support contains an inactive tet");
            const std::uint32_t nodes[4]{
                tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
            for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
                const std::uint32_t row =
                    tangentSupport.tetrahedronBasisNodeRows[
                        std::size_t(tetIndex) * 4u + corner];
                if (row >= basisNodeCount ||
                    globalFEMNodeIndices[row] != nodes[corner])
                    return fail("hybrid GS tet-to-basis row map is invalid");
                if (selected[row]) continue;
                NMPartitionedReductionHybridGSTermGPU term{
                    supportIndex, region, corner, 0u};
                byNode[nodes[corner]].push_back(term);
            }
        }
    }
    out.nodeTermRanges.resize(globalNodeCount);
    for (std::uint32_t node = 0u; node < globalNodeCount; ++node) {
        auto& range = out.nodeTermRanges[node];
        if (out.nodeTerms.size() > std::numeric_limits<std::uint32_t>::max())
            return fail("hybrid GS node-term offset overflows uint32");
        range.first = static_cast<std::uint32_t>(out.nodeTerms.size());
        const auto& terms = byNode[node];
        if (terms.size() > std::numeric_limits<std::uint32_t>::max())
            return fail("hybrid GS node has too many element terms");
        range.count = static_cast<std::uint32_t>(terms.size());
        if (range.count != 0u) ++out.fineComplementNodeCount;
        out.maximumNodeTermCount = std::max(
            out.maximumNodeTermCount, range.count);
        out.nodeTerms.insert(out.nodeTerms.end(), terms.begin(), terms.end());
    }
    if (out.nodeTerms.size() > std::numeric_limits<std::uint32_t>::max())
        return fail("hybrid GS node-term list exceeds uint32");
    return out;
}

} // namespace numi::matter
#endif
