#pragma once

#include "numi/matter/shared.h"
#include "numi/matter/partitioned_reduction_projection.h"

#define NM_PARTITIONED_REDUCTION_FEM_TANGENT_TILE_WIDTH 256u

typedef struct NM_ALIGN16 NMPartitionedReductionFEMTangentDispatchGPU {
    nm_u32 globalNodeCount, basisNodeCount, modeCount, tetrahedronCount;
    nm_u32 environmentCount, regionCount, blockModeCount, tileCount;
    nm_u32 objectIndex, supportTetrahedronCount, reserved0, reserved1;
} NMPartitionedReductionFEMTangentDispatchGPU;

// Per-(environment, regional-support-tet, local mode) element action. The
// diagonal pieces let the regional matrix add only FEM off-diagonal coupling;
// the already assembled nodal blocks retain FEM diagonal, inertia, and contact.
typedef struct NM_ALIGN16 NMPartitionedReductionFEMTangentActionGPU {
    NMFEMElementVectorGPU full;
    nm_float4 selfDiagonal[4];
} NMPartitionedReductionFEMTangentActionGPU;

#ifndef __METAL_VERSION__
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace numi::matter {
static_assert(sizeof(NMPartitionedReductionFEMTangentDispatchGPU) == 48u);
static_assert(sizeof(NMPartitionedReductionFEMTangentActionGPU) == 128u);

struct PartitionedReductionFEMTangentSupport {
    std::vector<NMPartitionedReductionSparseRangeGPU> regionTetrahedronRanges;
    std::vector<std::uint32_t> regionTetrahedronIndices;
    std::vector<std::uint32_t> tetrahedronBasisNodeRows;
    std::uint32_t maximumRegionTetrahedronCount = 0u;
    std::uint32_t tileCount = 0u;
    std::string error;
    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

// Exact union of tetrahedra touched by each regional twelve-mode block.
// Selected source-object tets must map all four corners into the exact global
// row map; dormant and unrelated-object capacity slots are omitted.
[[nodiscard]] inline PartitionedReductionFEMTangentSupport
buildPartitionedReductionFEMTangentSupport(
    std::span<const nm_float4> basisValues,
    std::span<const std::uint32_t> globalFEMNodeIndices,
    std::uint32_t basisNodeCount, std::uint32_t modeCount,
    std::span<const std::uint32_t> blockModeIndices,
    std::uint32_t regionCount, std::uint32_t blockModeCount,
    std::span<const NMTetrahedronGPU> tetrahedra,
    std::uint32_t objectIndex) {
    PartitionedReductionFEMTangentSupport out;
    auto fail = [&](const char* message) {
        out.error = message;
        out.regionTetrahedronRanges.clear();
        out.regionTetrahedronIndices.clear();
        out.tetrahedronBasisNodeRows.clear();
        out.maximumRegionTetrahedronCount = out.tileCount = 0u;
        return out;
    };
    if (!basisNodeCount || !modeCount || !regionCount ||
        !blockModeCount || blockModeCount > 32u ||
        globalFEMNodeIndices.size() != basisNodeCount ||
        basisValues.size() != std::size_t(basisNodeCount) * modeCount ||
        blockModeIndices.size() != std::size_t(regionCount) * blockModeCount ||
        tetrahedra.empty() ||
        tetrahedra.size() > std::numeric_limits<std::uint32_t>::max())
        return fail("FEM tangent support dimensions mismatch");
    for (std::uint32_t row = 1u; row < basisNodeCount; ++row)
        if (globalFEMNodeIndices[row - 1u] >= globalFEMNodeIndices[row])
            return fail("FEM tangent support row map is not strictly ascending");
    std::vector<std::uint8_t> seen(modeCount, 0u);
    for (const auto mode : blockModeIndices)
        if (mode >= modeCount || seen[mode]++)
            return fail("FEM tangent support mode map is invalid or duplicated");
    std::vector<std::vector<std::uint8_t>> supported(
        regionCount, std::vector<std::uint8_t>(basisNodeCount, 0u));
    for (std::uint32_t region = 0u; region < regionCount; ++region)
        for (std::uint32_t row = 0u; row < basisNodeCount; ++row)
            for (std::uint32_t local = 0u; local < blockModeCount; ++local) {
                const auto mode = blockModeIndices[
                    std::size_t(region) * blockModeCount + local];
                const auto value = basisValues[
                    std::size_t(mode) * basisNodeCount + row];
                if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                    !std::isfinite(value.z))
                    return fail("FEM tangent support basis has non-finite xyz");
                supported[region][row] |= value.x != 0.0f ||
                    value.y != 0.0f || value.z != 0.0f;
            }
    if (tetrahedra.size() > std::numeric_limits<std::size_t>::max() / 4u)
        return fail("FEM tangent support row map overflows size_t");
    out.tetrahedronBasisNodeRows.assign(tetrahedra.size() * 4u,
                                         NM_INVALID_INDEX);
    out.regionTetrahedronRanges.resize(regionCount);
    for (std::uint32_t region = 0u; region < regionCount; ++region) {
        auto& range = out.regionTetrahedronRanges[region];
        if (out.regionTetrahedronIndices.size() >
            std::numeric_limits<std::uint32_t>::max())
            return fail("FEM tangent support offset exceeds uint32");
        range.first = static_cast<std::uint32_t>(
            out.regionTetrahedronIndices.size());
        for (std::uint32_t tetIndex = 0u;
             tetIndex < tetrahedra.size(); ++tetIndex) {
            const auto& tet = tetrahedra[tetIndex];
            if (tet.identity.y != objectIndex ||
                (tet.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
            const std::uint32_t nodes[4]{
                tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
            bool touches = false;
            for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
                const auto found = std::lower_bound(
                    globalFEMNodeIndices.begin(), globalFEMNodeIndices.end(),
                    nodes[slot]);
                if (found == globalFEMNodeIndices.end() ||
                    *found != nodes[slot])
                    return fail("active source-object tet escapes the exact basis row map");
                const auto row = static_cast<std::uint32_t>(
                    found - globalFEMNodeIndices.begin());
                out.tetrahedronBasisNodeRows[
                    std::size_t(tetIndex) * 4u + slot] = row;
                touches |= supported[region][row] != 0u;
            }
            if (touches) out.regionTetrahedronIndices.push_back(tetIndex);
        }
        const std::size_t count = out.regionTetrahedronIndices.size() -
            range.first;
        if (!count || count > std::numeric_limits<std::uint32_t>::max())
            return fail("FEM tangent support has empty/oversized regional tet range");
        range.count = static_cast<std::uint32_t>(count);
        out.maximumRegionTetrahedronCount = std::max(
            out.maximumRegionTetrahedronCount, range.count);
    }
    if (out.regionTetrahedronIndices.size() >
        std::numeric_limits<std::uint32_t>::max())
        return fail("FEM tangent support flat list exceeds uint32");
    const std::uint64_t tiles =
        (std::uint64_t(out.maximumRegionTetrahedronCount) +
         NM_PARTITIONED_REDUCTION_FEM_TANGENT_TILE_WIDTH - 1u) /
        NM_PARTITIONED_REDUCTION_FEM_TANGENT_TILE_WIDTH;
    if (!tiles || tiles > std::numeric_limits<std::uint32_t>::max())
        return fail("FEM tangent support tile count invalid");
    out.tileCount = static_cast<std::uint32_t>(tiles);
    return out;
}
} // namespace numi::matter
#endif

