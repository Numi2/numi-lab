#pragma once

#include "numi/matter/shared.h"

#define NM_PARTITIONED_REDUCTION_TILE_WIDTH 256u

typedef struct NM_ALIGN16 NMPartitionedReductionProjectionDispatchGPU {
    nm_u32 globalNodeCount;
    nm_u32 basisNodeCount;
    nm_u32 modeCount;
    nm_u32 tileCount;
    nm_u32 environmentCount;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u32 reserved2;
} NMPartitionedReductionProjectionDispatchGPU;

typedef struct NM_ALIGN16 NMPartitionedReductionBlockDispatchGPU {
    nm_u32 globalNodeCount;
    nm_u32 basisNodeCount;
    nm_u32 modeCount;
    nm_u32 tileCount;
    nm_u32 environmentCount;
    nm_u32 regionCount;
    nm_u32 blockModeCount;
    nm_u32 reserved;
} NMPartitionedReductionBlockDispatchGPU;

// Compact immutable GPU range for sparse regional basis support. The host
// stores one range per region or basis row; first/count index flat uint arrays.
typedef struct NMPartitionedReductionSparseRangeGPU {
    nm_u32 first;
    nm_u32 count;
} NMPartitionedReductionSparseRangeGPU;

#ifndef __METAL_VERSION__
#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::matter {

static_assert(sizeof(NMPartitionedReductionSparseRangeGPU) == 8u);

[[nodiscard]] inline bool validPartitionedReductionProjectionDispatch(
    const NMPartitionedReductionProjectionDispatchGPU& dispatch
) noexcept {
    if (dispatch.globalNodeCount == 0u || dispatch.basisNodeCount == 0u ||
        dispatch.modeCount == 0u || dispatch.environmentCount == 0u) return false;
    const std::uint64_t expectedTiles =
        (static_cast<std::uint64_t>(dispatch.basisNodeCount) +
         NM_PARTITIONED_REDUCTION_TILE_WIDTH - 1u) /
        NM_PARTITIONED_REDUCTION_TILE_WIDTH;
    if (expectedTiles != dispatch.tileCount) return false;
    const auto fits = [](const std::uint64_t a, const std::uint64_t b) {
        return a == 0u || b <= std::numeric_limits<std::size_t>::max() / a;
    };
    if (!fits(dispatch.globalNodeCount, dispatch.environmentCount) ||
        !fits(dispatch.basisNodeCount, dispatch.modeCount)) return false;
    const std::uint64_t partialPerEnvironment =
        static_cast<std::uint64_t>(dispatch.modeCount) * dispatch.tileCount;
    return fits(partialPerEnvironment, dispatch.environmentCount);
}

[[nodiscard]] inline bool validPartitionedReductionBlockDispatch(
    const NMPartitionedReductionBlockDispatchGPU& dispatch
) noexcept {
    if (dispatch.globalNodeCount == 0u || dispatch.basisNodeCount == 0u ||
        dispatch.modeCount == 0u || dispatch.environmentCount == 0u ||
        dispatch.regionCount == 0u || dispatch.blockModeCount == 0u ||
        dispatch.blockModeCount > 32u) return false;
    const std::uint64_t expectedTiles =
        (static_cast<std::uint64_t>(dispatch.basisNodeCount) +
         NM_PARTITIONED_REDUCTION_TILE_WIDTH - 1u) /
        NM_PARTITIONED_REDUCTION_TILE_WIDTH;
    if (expectedTiles != dispatch.tileCount) return false;
    const auto checkedProduct = [](const std::uint64_t a, const std::uint64_t b,
                                   std::uint64_t& result) {
        if (a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a)
            return false;
        result = a * b;
        return result <= std::numeric_limits<std::size_t>::max();
    };
    std::uint64_t pairs = 0u, perEnvironment = 0u, total = 0u;
    return checkedProduct(dispatch.blockModeCount, dispatch.blockModeCount, pairs) &&
        checkedProduct(pairs, dispatch.tileCount, perEnvironment) &&
        checkedProduct(perEnvironment, dispatch.regionCount, total) &&
        checkedProduct(total, dispatch.environmentCount, total);
}

} // namespace numi::matter
#endif

