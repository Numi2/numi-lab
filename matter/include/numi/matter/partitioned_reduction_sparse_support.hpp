#pragma once

#include "numi/matter/partitioned_reduction_basis.hpp"
#include "numi/matter/partitioned_reduction_projection.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace numi::matter {

// Immutable upload metadata for the exact sparse support of the exported FP32
// basis. Ranges are {first,count} pairs into the accompanying flat arrays.
struct PartitionedReductionSparseSupport {
    std::vector<NMPartitionedReductionSparseRangeGPU> regionSupportRanges;
    std::vector<std::uint32_t> regionSupportRows;
    std::vector<NMPartitionedReductionSparseRangeGPU> nodeModeRanges;
    // Packed as region * blockModeCount + localMode.
    std::vector<std::uint32_t> nodeModeSlots;
    std::uint32_t sparseTileCount = 0u;
    std::uint32_t maximumNodeModeCount = 0u;
    std::string error;

    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

// Hybrid Bq requires bijective coordinates and a race-free source row map.
// The compiled Human owner promises strictly ascending global node IDs; retain
// that contract here so GPU lift/scatter never aliases two basis rows.
[[nodiscard]] inline std::string validatePartitionedHybridIndexMaps(
    const std::span<const std::uint32_t> globalFEMNodeIndices,
    const std::uint32_t basisNodeCount,
    const std::uint32_t globalNodeCount,
    const std::span<const std::uint32_t> blockModeIndices,
    const std::uint32_t modeCount,
    const std::uint32_t regionCount,
    const std::uint32_t blockModeCount
) {
    if (basisNodeCount == 0u || globalNodeCount == 0u || modeCount == 0u ||
        regionCount == 0u || blockModeCount == 0u || blockModeCount > 32u ||
        globalFEMNodeIndices.size() != basisNodeCount ||
        blockModeIndices.size() != modeCount ||
        static_cast<std::uint64_t>(regionCount) * blockModeCount != modeCount)
        return "hybrid index map dimensions do not match the full basis";

    for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
        const std::uint32_t node = globalFEMNodeIndices[row];
        if (node >= globalNodeCount)
            return "hybrid source node is outside the compiled FEM range";
        if (row != 0u && globalFEMNodeIndices[row - 1u] >= node)
            return "hybrid source nodes must be unique and strictly ascending";
    }

    std::vector<std::uint8_t> modeSeen(modeCount, 0u);
    for (const std::uint32_t mode : blockModeIndices) {
        if (mode >= modeCount)
            return "hybrid regional mode is outside the full coordinate range";
        if (modeSeen[mode] != 0u)
            return "hybrid regional mode map contains a duplicate coordinate";
        modeSeen[mode] = 1u;
    }
    return {};
}

// Builds deterministic compact support from the exact uploaded float B. The
// caller supplies the existing region-major [region][localMode] global-mode
// map; no thresholds are used, so a coefficient is present iff any xyz
// component is exactly nonzero in the GPU basis representation.
[[nodiscard]] inline PartitionedReductionSparseSupport
buildPartitionedReductionSparseSupport(
    const std::span<const nm_float4> basisValues,
    const std::uint32_t basisNodeCount,
    const std::uint32_t modeCount,
    const std::span<const std::uint32_t> blockModeIndices,
    const std::uint32_t regionCount,
    const std::uint32_t blockModeCount
) {
    PartitionedReductionSparseSupport result;
    const auto fail = [&](const char* message) {
        result.error = message;
        result.regionSupportRanges.clear();
        result.regionSupportRows.clear();
        result.nodeModeRanges.clear();
        result.nodeModeSlots.clear();
        result.sparseTileCount = 0u;
        result.maximumNodeModeCount = 0u;
        return result;
    };
    if (basisNodeCount == 0u || modeCount == 0u ||
        basisValues.size() != static_cast<std::size_t>(basisNodeCount) *
            modeCount || regionCount == 0u || blockModeCount == 0u ||
        blockModeCount > 32u ||
        blockModeIndices.size() != static_cast<std::size_t>(regionCount) *
            blockModeCount) {
        return fail("sparse support inputs do not match the dense basis and regional block map");
    }
    std::vector<std::uint8_t> modeSeen(modeCount, 0u);
    for (const std::uint32_t mode : blockModeIndices) {
        if (mode >= modeCount || modeSeen[mode] != 0u)
            return fail("sparse support regional mode map is out of range or duplicated");
        modeSeen[mode] = 1u;
    }

    result.regionSupportRanges.resize(regionCount);
    result.nodeModeRanges.resize(basisNodeCount);
    std::uint32_t maximumRegionRows = 0u;
    for (std::uint32_t region = 0u; region < regionCount; ++region) {
        if (result.regionSupportRows.size() >
            std::numeric_limits<std::uint32_t>::max())
            return fail("sparse support row offsets exceed uint32 range");
        auto& range = result.regionSupportRanges[region];
        range.first = static_cast<std::uint32_t>(result.regionSupportRows.size());
        for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
            bool supported = false;
            for (std::uint32_t localMode = 0u;
                 localMode < blockModeCount; ++localMode) {
                const std::uint32_t mode = blockModeIndices[
                    static_cast<std::size_t>(region) * blockModeCount + localMode];
                const auto& value = basisValues[
                    static_cast<std::size_t>(mode) * basisNodeCount + row];
                if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                    !std::isfinite(value.z))
                    return fail("sparse support basis contains a non-finite xyz coefficient");
                supported = supported || value.x != 0.0f || value.y != 0.0f ||
                    value.z != 0.0f;
            }
            if (supported) result.regionSupportRows.push_back(row);
        }
        const std::size_t count = result.regionSupportRows.size() - range.first;
        if (count > std::numeric_limits<std::uint32_t>::max())
            return fail("sparse support region row count exceeds uint32 range");
        range.count = static_cast<std::uint32_t>(count);
        maximumRegionRows = std::max(maximumRegionRows, range.count);
    }
    const std::uint64_t tileCount =
        (static_cast<std::uint64_t>(maximumRegionRows) +
         NM_PARTITIONED_REDUCTION_TILE_WIDTH - 1u) /
        NM_PARTITIONED_REDUCTION_TILE_WIDTH;
    if (tileCount == 0u)
        return fail("sparse support has no nonzero regional basis rows");
    if (tileCount > std::numeric_limits<std::uint32_t>::max())
        return fail("sparse support tile count exceeds uint32 range");
    result.sparseTileCount = static_cast<std::uint32_t>(tileCount);

    for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
        if (result.nodeModeSlots.size() >
            std::numeric_limits<std::uint32_t>::max())
            return fail("sparse support node mode offsets exceed uint32 range");
        auto& range = result.nodeModeRanges[row];
        range.first = static_cast<std::uint32_t>(result.nodeModeSlots.size());
        for (std::uint32_t region = 0u; region < regionCount; ++region) {
            for (std::uint32_t localMode = 0u;
                 localMode < blockModeCount; ++localMode) {
                const std::uint32_t mode = blockModeIndices[
                    static_cast<std::size_t>(region) * blockModeCount + localMode];
                const auto& value = basisValues[
                    static_cast<std::size_t>(mode) * basisNodeCount + row];
                if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                    !std::isfinite(value.z))
                    return fail("sparse support basis contains a non-finite xyz coefficient");
                if (value.x == 0.0f && value.y == 0.0f && value.z == 0.0f)
                    continue;
                const std::uint64_t packed =
                    static_cast<std::uint64_t>(region) * blockModeCount + localMode;
                if (packed > std::numeric_limits<std::uint32_t>::max())
                    return fail("sparse support packed node mode exceeds uint32 range");
                result.nodeModeSlots.push_back(static_cast<std::uint32_t>(packed));
            }
        }
        const std::size_t count = result.nodeModeSlots.size() - range.first;
        if (count > std::numeric_limits<std::uint32_t>::max())
            return fail("sparse support node mode count exceeds uint32 range");
        range.count = static_cast<std::uint32_t>(count);
        result.maximumNodeModeCount =
            std::max(result.maximumNodeModeCount, range.count);
    }
    if (result.regionSupportRows.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        result.nodeModeSlots.size() >
            std::numeric_limits<std::uint32_t>::max())
        return fail("sparse support flattened arrays exceed uint32 address range");
    return result;
}


[[nodiscard]] inline PartitionedReductionSparseSupport
buildPartitionedReductionSparseSupport(
    const PartitionedReductionBasis& basis,
    const std::span<const std::uint32_t> blockModeIndices,
    const std::uint32_t regionCount,
    const std::uint32_t blockModeCount
) {
    if (!basis.valid()) {
        PartitionedReductionSparseSupport result;
        result.error = "sparse support basis is invalid: " + basis.error;
        return result;
    }
    return buildPartitionedReductionSparseSupport(
        std::span<const nm_float4>(basis.values), basis.nodeCount,
        basis.modeCount, blockModeIndices, regionCount, blockModeCount);
}

} // namespace numi::matter
