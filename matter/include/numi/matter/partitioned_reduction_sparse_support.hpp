#pragma once

#include "numi/matter/partitioned_reduction_basis.hpp"
#include "numi/matter/matter.hpp"
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

// Validates that one complete compiled FEM object remains on the identity-fine
// path of a hybrid Bq vector. A row with no B support is copied through by
// hybrid_lift and is never selected/zeroed by hybrid_restrict; bind that rule
// to the exact object-local range and reject any nonzero coefficient there.
[[nodiscard]] inline std::string validatePartitionedHybridFineOwnerRows(
    const std::span<const std::uint32_t> globalFEMNodeIndices,
    const std::span<const std::uint32_t> rowObjectIndices,
    const std::span<const nm_float4> basisValues,
    const std::uint32_t basisNodeCount,
    const std::uint32_t modeCount,
    const std::uint32_t fineObjectIndex,
    const std::uint32_t fineStateOffset,
    const std::uint32_t fineStateCount
) {
    if (basisNodeCount == 0u || modeCount == 0u ||
        globalFEMNodeIndices.size() != basisNodeCount ||
        rowObjectIndices.size() != basisNodeCount || fineStateCount == 0u ||
        static_cast<std::uint64_t>(basisNodeCount) * modeCount !=
            basisValues.size() ||
        static_cast<std::uint64_t>(fineStateOffset) + fineStateCount >
            std::numeric_limits<std::uint32_t>::max())
        return "hybrid identity-fine owner dimensions are invalid";

    std::vector<std::uint8_t> seen(fineStateCount, 0u);
    std::uint32_t ownerRows = 0u;
    const std::uint64_t fineEnd =
        static_cast<std::uint64_t>(fineStateOffset) + fineStateCount;
    for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
        const std::uint32_t global = globalFEMNodeIndices[row];
        const bool inFineRange = global >= fineStateOffset && global < fineEnd;
        const bool ownedByFineObject = rowObjectIndices[row] == fineObjectIndex;
        if (inFineRange != ownedByFineObject)
            return "hybrid identity-fine source range and compiled row ownership disagree";
        if (!ownedByFineObject) continue;
        const std::uint32_t local = global - fineStateOffset;
        if (seen[local] != 0u)
            return "hybrid identity-fine source range contains a duplicate row";
        seen[local] = 1u;
        ++ownerRows;
        for (std::uint32_t mode = 0u; mode < modeCount; ++mode) {
            const nm_float4 value = basisValues[
                static_cast<std::size_t>(mode) * basisNodeCount + row];
            if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                !std::isfinite(value.z) || !std::isfinite(value.w))
                return "hybrid identity-fine source coefficient is non-finite";
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f ||
                value.w != 0.0f)
                return "hybrid identity-fine source row has a nonzero B coefficient";
        }
    }
    if (ownerRows != fineStateCount ||
        std::find(seen.begin(), seen.end(), std::uint8_t{0u}) != seen.end())
        return "hybrid identity-fine source object is not represented by its complete row range";
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
    if (basis.sparseCoefficientEncoding) {
        PartitionedReductionSparseSupport out;
        const auto reject=[&](const char* message) {
            PartitionedReductionSparseSupport failed;failed.error=message;return failed;
        };
        if (!basis.coefficientStorageValid() || regionCount==0u || blockModeCount==0u ||
            blockModeCount>32u || blockModeIndices.size()!=std::size_t(regionCount)*blockModeCount ||
            blockModeIndices.size()>basis.modeCount)
            return reject("streamed sparse support dimensions are invalid");
        std::vector<std::uint8_t> seen(basis.modeCount,0u);
        out.regionSupportRanges.resize(regionCount);
        out.nodeModeRanges.resize(basis.nodeCount);
        std::vector<std::uint32_t> counts(basis.nodeCount,0u);
        for (std::uint32_t region=0;region<regionCount;++region) {
            std::vector<std::uint32_t> rows;
            for (std::uint32_t local=0;local<blockModeCount;++local) {
                const auto mode=blockModeIndices[region*blockModeCount+local];
                if (mode>=basis.modeCount || seen[mode]++)
                    return reject("streamed sparse regional mode map is invalid");
                for (auto i=basis.modeValueOffsets[mode];i<basis.modeValueOffsets[mode+1u];++i) {
                    const auto v=basis.sparseValues[i];
                    if (!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z)||v.w!=0.f)
                        return reject("streamed sparse coefficient is invalid");
                    if (v.x==0.f&&v.y==0.f&&v.z==0.f) continue;
                    rows.push_back(basis.coefficientNodeIndices[i]);
                    ++counts[basis.coefficientNodeIndices[i]];
                }
            }
            std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
            if (out.regionSupportRows.size()+rows.size()>std::numeric_limits<std::uint32_t>::max())
                return reject("streamed sparse support exceeds uint32 offsets");
            out.regionSupportRanges[region]={std::uint32_t(out.regionSupportRows.size()),std::uint32_t(rows.size())};
            out.regionSupportRows.insert(out.regionSupportRows.end(),rows.begin(),rows.end());
            out.sparseTileCount=std::max(out.sparseTileCount,std::uint32_t((rows.size()+255u)/256u));
        }
        std::uint64_t total=0u;
        for (std::uint32_t row=0;row<basis.nodeCount;++row) {
            if (total+counts[row]>std::numeric_limits<std::uint32_t>::max())
                return reject("streamed sparse node modes exceed uint32 offsets");
            out.nodeModeRanges[row]={std::uint32_t(total),counts[row]};
            total+=counts[row];out.maximumNodeModeCount=std::max(out.maximumNodeModeCount,counts[row]);
        }
        out.nodeModeSlots.resize(total);std::fill(counts.begin(),counts.end(),0u);
        for (std::uint32_t packed=0;packed<blockModeIndices.size();++packed) {
            const auto mode=blockModeIndices[packed];
            for (auto i=basis.modeValueOffsets[mode];i<basis.modeValueOffsets[mode+1u];++i) {
                const auto v=basis.sparseValues[i];
                if (v.x==0.f&&v.y==0.f&&v.z==0.f) continue;
                const auto row=basis.coefficientNodeIndices[i];
                out.nodeModeSlots[out.nodeModeRanges[row].first+counts[row]++]=packed;
            }
        }
        if (!out.sparseTileCount) return reject("streamed sparse support has no active rows");
        return out;
    }
    return buildPartitionedReductionSparseSupport(
        std::span<const nm_float4>(basis.values), basis.nodeCount,
        basis.modeCount, blockModeIndices, regionCount, blockModeCount);
}


// Compact coefficient storage for the exact regional support map. Each mode
// stores one value for each row in its owning region's support list; rows not
// in that list are exact zero. The mode offsets form a CSR start array.
struct PartitionedReductionSparseCoefficients {
    std::vector<std::uint32_t> modeValueOffsets;
    std::vector<nm_float4> values;
    std::vector<std::uint32_t> nodeModeValueIndices;
    std::uint64_t denseCoefficientCount = 0u;
    std::string error;
    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

[[nodiscard]] inline PartitionedReductionSparseCoefficients
buildPartitionedReductionSparseCoefficients(
    const std::span<const nm_float4> denseBasis,
    const std::uint32_t basisNodeCount,
    const std::uint32_t modeCount,
    const std::span<const std::uint32_t> blockModeIndices,
    const std::uint32_t regionCount,
    const std::uint32_t blockModeCount,
    const PartitionedReductionSparseSupport& support
) {
    PartitionedReductionSparseCoefficients out;
    const auto fail = [&](const char* message) {
        out.error = message;
        out.modeValueOffsets.clear();
        out.values.clear();
        out.nodeModeValueIndices.clear();
        return out;
    };
    if (!support.valid() || basisNodeCount == 0u || modeCount == 0u ||
        regionCount == 0u || blockModeCount == 0u ||
        denseBasis.size() != static_cast<std::size_t>(basisNodeCount) * modeCount ||
        blockModeIndices.size() != static_cast<std::size_t>(regionCount) * blockModeCount ||
        support.regionSupportRanges.size() != regionCount ||
        support.nodeModeRanges.size() != basisNodeCount ||
        support.nodeModeSlots.size() > std::numeric_limits<std::uint32_t>::max())
        return fail("sparse coefficient inputs do not match exact dense basis support");

    out.denseCoefficientCount = static_cast<std::uint64_t>(basisNodeCount) * modeCount;
    out.modeValueOffsets.assign(static_cast<std::size_t>(modeCount) + 1u,
                                std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> modeRegion(modeCount, NM_INVALID_INDEX);
    std::vector<std::uint8_t> modeSeen(modeCount, 0u);
    std::uint64_t sparseCount = 0u;
    for (std::uint32_t region = 0u; region < regionCount; ++region) {
        const auto range = support.regionSupportRanges[region];
        if (static_cast<std::uint64_t>(range.first) + range.count >
            support.regionSupportRows.size())
            return fail("sparse region support range escapes its row list");
        for (std::uint32_t local = 0u; local < blockModeCount; ++local) {
            const std::uint32_t mode = blockModeIndices[
                static_cast<std::size_t>(region) * blockModeCount + local];
            if (mode >= modeCount || modeSeen[mode] != 0u)
                return fail("sparse coefficient mode map is out of range or duplicated");
            modeSeen[mode] = 1u;
            modeRegion[mode] = region;
            sparseCount += range.count;
            if (sparseCount > std::numeric_limits<std::uint32_t>::max())
                return fail("sparse coefficient storage exceeds uint32 addressing");
        }
    }
    if (std::find(modeSeen.begin(), modeSeen.end(), std::uint8_t{0u}) != modeSeen.end())
        return fail("sparse coefficient mode map does not cover every mode");
    out.values.reserve(static_cast<std::size_t>(sparseCount));
    for (std::uint32_t mode = 0u; mode < modeCount; ++mode) {
        const auto region = modeRegion[mode];
        const auto range = support.regionSupportRanges[region];
        out.modeValueOffsets[mode] = static_cast<std::uint32_t>(out.values.size());
        for (std::uint32_t index = 0u; index < range.count; ++index) {
            const auto row = support.regionSupportRows[range.first + index];
            if (row >= basisNodeCount)
                return fail("sparse coefficient support row is outside the basis row domain");
            const auto value = denseBasis[
                static_cast<std::size_t>(mode) * basisNodeCount + row];
            if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                !std::isfinite(value.z) || !std::isfinite(value.w))
                return fail("sparse coefficient source contains a non-finite value");
            out.values.push_back(value);
        }
    }
    out.modeValueOffsets[modeCount] = static_cast<std::uint32_t>(out.values.size());

    out.nodeModeValueIndices.reserve(support.nodeModeSlots.size());
    for (std::uint32_t row = 0u; row < basisNodeCount; ++row) {
        const auto nodeRange = support.nodeModeRanges[row];
        if (static_cast<std::uint64_t>(nodeRange.first) + nodeRange.count >
            support.nodeModeSlots.size())
            return fail("sparse node-mode range escapes its support list");
        for (std::uint32_t slot = 0u; slot < nodeRange.count; ++slot) {
            const auto packed = support.nodeModeSlots[nodeRange.first + slot];
            const auto region = packed / blockModeCount;
            const auto local = packed - region * blockModeCount;
            if (region >= regionCount || local >= blockModeCount)
                return fail("sparse node-mode slot is outside the regional map");
            const auto mode = blockModeIndices[
                static_cast<std::size_t>(region) * blockModeCount + local];
            const auto range = support.regionSupportRanges[region];
            const auto begin = support.regionSupportRows.begin() + range.first;
            const auto end = begin + range.count;
            const auto found = std::lower_bound(begin, end, row);
            if (found == end || *found != row)
                return fail("sparse node-mode row is missing from its region support");
            const std::uint64_t valueIndex =
                static_cast<std::uint64_t>(out.modeValueOffsets[mode]) +
                static_cast<std::uint64_t>(found - begin);
            if (valueIndex >= out.values.size())
                return fail("sparse node-mode value index escapes coefficient storage");
            const auto value = out.values[valueIndex];
            if (value.x == 0.0f && value.y == 0.0f && value.z == 0.0f)
                return fail("sparse node-mode map references a zero coefficient");
            out.nodeModeValueIndices.push_back(static_cast<std::uint32_t>(valueIndex));
        }
    }
    if (out.nodeModeValueIndices.size() != support.nodeModeSlots.size())
        return fail("sparse node-mode coefficient map has the wrong extent");
    return out;
}

[[nodiscard]] inline nm_float4 humanPartitionedCoefficient(
    const detail::HumanPartitionedPreconditionerSource& source,
    const std::uint32_t mode, const std::uint32_t row
) noexcept {
    if (!source.sparseCoefficientEncoding)
        return source.basisValues[std::size_t(mode)*source.basisNodeCount+row];
    const auto region=mode/source.blockModeCount;
    const auto first=source.sparseRegionSupportOffsets[region];
    const auto end=source.sparseRegionSupportOffsets[region+1u];
    const auto begin=source.sparseRegionSupportRows.begin()+first;
    const auto finish=source.sparseRegionSupportRows.begin()+end;
    const auto found=std::lower_bound(begin,finish,row);
    return found!=finish&&*found==row
        ?source.sparseBasisValues[source.sparseModeValueOffsets[mode]+std::size_t(found-begin)]
        :nm_float4{};
}

// Validate the compact source and rebuild exact worklists directly from it.
// In particular, never allocate or scan a mode*node dense mirror.
[[nodiscard]] inline PartitionedReductionSparseSupport
buildPartitionedReductionSparseSupport(
    const detail::HumanPartitionedPreconditionerSource& source
) {
    if (!source.sparseCoefficientEncoding)
        return buildPartitionedReductionSparseSupport(source.basisValues,source.basisNodeCount,
            source.modeCount,source.blockModeIndices,source.regionCount,source.blockModeCount);
    PartitionedReductionSparseSupport out;
    const auto reject=[&](const char* message) {
        PartitionedReductionSparseSupport failed;failed.error=message;return failed;
    };
    if (!source.basisValues.empty()||!source.regionCount||!source.basisNodeCount||
        !source.blockModeCount||source.modeCount!=std::uint64_t(source.regionCount)*source.blockModeCount||
        source.blockModeIndices.size()!=source.modeCount||
        source.sparseRegionSupportOffsets.size()!=std::size_t(source.regionCount)+1u||
        source.sparseModeValueOffsets.size()!=std::size_t(source.modeCount)+1u||
        source.sparseRegionSupportOffsets.front()!=0u||source.sparseModeValueOffsets.front()!=0u||
        source.sparseRegionSupportOffsets.back()!=source.sparseRegionSupportRows.size()||
        source.sparseModeValueOffsets.back()!=source.sparseBasisValues.size()||
        source.sparseBasisValues.size()>kHumanPartitionedBasisCoefficientByteBudget/sizeof(nm_float4))
        return reject("compact Human source dimensions or byte budget are invalid");
    PartitionedReductionBasis compact;
    compact.sparseCoefficientEncoding=true;compact.nodeCount=source.basisNodeCount;
    compact.modeCount=source.modeCount;compact.modeValueOffsets.push_back(0u);
    out.regionSupportRanges.resize(source.regionCount);
    for (std::uint32_t region=0;region<source.regionCount;++region) {
        const auto first=source.sparseRegionSupportOffsets[region],end=source.sparseRegionSupportOffsets[region+1u];
        if (first>=end||end>source.sparseRegionSupportRows.size())
            return reject("compact Human region range is empty or invalid");
        for (auto i=first;i<end;++i)
            if (source.sparseRegionSupportRows[i]>=source.basisNodeCount||
                (i>first&&source.sparseRegionSupportRows[i-1u]>=source.sparseRegionSupportRows[i]))
                return reject("compact Human region rows are not ordered unique source rows");
        for (std::uint32_t local=0;local<source.blockModeCount;++local) {
            const auto mode=region*source.blockModeCount+local;
            const auto a=source.sparseModeValueOffsets[mode],b=source.sparseModeValueOffsets[mode+1u];
            if (source.blockModeIndices[mode]!=mode||a>b||b>source.sparseBasisValues.size()||b-a!=end-first)
                return reject("compact Human coefficient modes do not match exact regional support");
            for (auto i=a;i<b;++i) {
                const auto v=source.sparseBasisValues[i];
                if (!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z)||v.w!=0.f)
                    return reject("compact Human coefficient is nonfinite or packed");
                if (v.x==0.f&&v.y==0.f&&v.z==0.f) continue;
                compact.sparseValues.push_back(v);
                compact.coefficientNodeIndices.push_back(source.sparseRegionSupportRows[first+i-a]);
            }
            compact.modeValueOffsets.push_back(std::uint32_t(compact.sparseValues.size()));
        }
    }
    out=buildPartitionedReductionSparseSupport(compact,source.blockModeIndices,source.regionCount,source.blockModeCount);
    if (!out.valid()) return out;
    if (out.regionSupportRows!=source.sparseRegionSupportRows)
        return reject("compact Human payload contains rows without any exact nonzero coefficient");
    if (source.sparseNodeModeValueIndices.size()!=out.nodeModeSlots.size())
        return reject("compact Human node coefficient index extent differs");
    for (std::uint32_t row=0;row<source.basisNodeCount;++row) {
        const auto range=out.nodeModeRanges[row];
        for (auto i=range.first;i<range.first+range.count;++i) {
            const auto mode=out.nodeModeSlots[i],region=mode/source.blockModeCount;
            const auto first=source.sparseRegionSupportOffsets[region],end=source.sparseRegionSupportOffsets[region+1u];
            const auto begin=source.sparseRegionSupportRows.begin()+first;
            const auto found=std::lower_bound(begin,source.sparseRegionSupportRows.begin()+end,row);
            const auto expected=source.sparseModeValueOffsets[mode]+std::uint32_t(found-begin);
            if (source.sparseNodeModeValueIndices[i]!=expected)
                return reject("compact Human node coefficient index identity differs");
        }
    }
    return out;
}

// Pack directly into the existing Human sparse payload, retaining source
// mode selection and exact Float32 coefficients without a dense compatibility copy.
[[nodiscard]] inline bool packSparseHumanPartitionedBasis(
    const PartitionedReductionBasis& basis,
    const std::span<const std::uint32_t> selectedModes,
    detail::HumanPartitionedPreconditionerSource& payload,
    std::string& error
) {
    const auto support=buildPartitionedReductionSparseSupport(
        basis,selectedModes,payload.regionCount,payload.blockModeCount);
    if (!support.valid()) {error=support.error;return false;}
    payload.basisValues.clear();payload.sparseCoefficientEncoding=true;
    payload.sparseRegionSupportOffsets.assign(payload.regionCount+1u,0u);
    payload.sparseRegionSupportRows=support.regionSupportRows;
    payload.sparseModeValueOffsets.assign(payload.modeCount+1u,0u);
    payload.sparseBasisValues.clear();payload.sparseNodeModeValueIndices.clear();
    for (std::uint32_t region=0;region<payload.regionCount;++region) {
        const auto range=support.regionSupportRanges[region];
        payload.sparseRegionSupportOffsets[region+1u]=range.first+range.count;
        for (std::uint32_t local=0;local<payload.blockModeCount;++local) {
            const auto mode=region*payload.blockModeCount+local;
            for (std::uint32_t i=0;i<range.count;++i)
                payload.sparseBasisValues.push_back(
                    basis.coefficient(selectedModes[mode],support.regionSupportRows[range.first+i]));
            if (payload.sparseBasisValues.size()>kHumanPartitionedBasisCoefficientByteBudget/sizeof(nm_float4)) {
                error="streamed Human coefficient payload exceeds checked byte budget";return false;
            }
            payload.sparseModeValueOffsets[mode+1u]=std::uint32_t(payload.sparseBasisValues.size());
        }
    }
    for (std::uint32_t row=0;row<payload.basisNodeCount;++row) {
        const auto range=support.nodeModeRanges[row];
        for (auto i=range.first;i<range.first+range.count;++i) {
            const auto mode=support.nodeModeSlots[i],region=mode/payload.blockModeCount;
            const auto first=payload.sparseRegionSupportOffsets[region],end=payload.sparseRegionSupportOffsets[region+1u];
            const auto begin=payload.sparseRegionSupportRows.begin()+first;
            const auto found=std::lower_bound(begin,payload.sparseRegionSupportRows.begin()+end,row);
            if (found==payload.sparseRegionSupportRows.begin()+end||*found!=row) {
                error="streamed Human node mode lacks exact regional support";return false;
            }
            payload.sparseNodeModeValueIndices.push_back(
                payload.sparseModeValueOffsets[mode]+std::uint32_t(found-begin));
        }
    }
    const auto check=buildPartitionedReductionSparseSupport(payload);
    if (!check.valid()) {error=check.error;return false;}
    error.clear();return true;
}

} // namespace numi::matter
