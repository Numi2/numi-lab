#pragma once

// Compiler-side construction of a sparse, anatomically partitioned nodal
// basis. The returned mode-major storage matches the per-node layout used by
// RegionalGalerkinBasis, but this helper does not define runtime coordinates:
// x = Bq is not enabled until the runtime owns state, force, tangent, and
// contact projection through this basis.

#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace numi::matter {

// Construction limit is deliberately larger than the legacy default. The
// Human consumer applies a stricter mode and coefficient-byte budget.
inline constexpr std::uint32_t kPartitionedReductionMaximumConstructionModeCount = 4096u;
inline constexpr std::uint32_t kHumanPartitionedBasisMaximumRegionCount = 170u;
inline constexpr std::uint32_t kHumanPartitionedBasisMaximumModeCount =
    kHumanPartitionedBasisMaximumRegionCount * 12u;
inline constexpr std::uint64_t kHumanPartitionedBasisCoefficientByteBudget =
    384ull * 1024ull * 1024ull;

[[nodiscard]] inline bool humanPartitionedBasisCoefficientExtentWithinBudget(
    const std::uint32_t modeCount,
    const std::uint32_t nodeCount
) noexcept {
    if (modeCount == 0u || nodeCount == 0u) return false;
    const std::uint64_t maximumValues =
        kHumanPartitionedBasisCoefficientByteBudget / sizeof(nm_float4);
    return static_cast<std::uint64_t>(modeCount) <=
        maximumValues / nodeCount;
}

enum class PartitionedReductionBasisRole : std::uint8_t {
    preconditionerOnly,
};

enum class PartitionedReductionModeKind : std::uint8_t {
    regionalTranslation,
    regionalRotation,
    // component = outputAxis * 3 + inputAxis; value is
    // partitionWeight * (x[inputAxis] - regionCenter[inputAxis]) / scale.
    regionalAffineGradient,
    anchorTranslation,
};

struct PartitionedReductionMode {
    PartitionedReductionModeKind kind =
        PartitionedReductionModeKind::regionalTranslation;
    // Original authored region label; NM_INVALID_INDEX for an anchor mode.
    std::uint32_t regionLabel = NM_INVALID_INDEX;
    // Translation/rotation axis, affine-gradient row/column pair, or
    // anchor-local Cartesian axis.
    std::uint32_t component = 0u;
    // Original anchor node for anchorTranslation; NM_INVALID_INDEX otherwise.
    std::uint32_t anchorNode = NM_INVALID_INDEX;
    // Rotation and affine-gradient modes use this characteristic radius as
    // their coefficient scale.
    double scale = 1.0;
};

struct PartitionedReductionRegion {
    std::uint32_t sourceLabel = NM_INVALID_INDEX;
    std::array<double, 3> center{};
    // RMS radius for scaling regional rotation and affine-gradient modes.
    double rotationRadius = 0.0;
    double volume = 0.0;
};

struct PartitionedReductionBasisInput {
    std::span<const std::array<double, 3>> restPositions;
    std::span<const double> lumpedMass;
    // One byte per source node: 0 means free, 1 means fixed.
    std::span<const std::uint8_t> fixedMask;
    std::span<const std::array<std::uint32_t, 4>> tetrahedra;
    // Original anatomical/compiler labels, one per tetrahedron.
    std::span<const std::uint32_t> tetrahedronRegionLabels;
    // Source node IDs for sensitive anchors. Their source identity is retained.
    std::span<const std::uint32_t> anchorNodes;
};

struct PartitionedReductionBasisOptions {
    // Require at least this many source coordinates per retained mode.
    // Four gives a minimum 4:1 coordinate reduction when accepted.
    std::uint32_t minimumCoordinateReductionFactor = 4u;
    std::uint32_t maximumModeCount = 32u;
    // Optional caller-owned ceiling for the dense nm_float4 coefficient rows.
    // Zero keeps the generic builder's existing unbounded behavior.
    std::uint64_t maximumDenseCoefficientBytes = 0u;
    bool sparseCoefficientEncoding = false;
};

struct PartitionedReductionBasis {
    static constexpr PartitionedReductionBasisRole role =
        PartitionedReductionBasisRole::preconditionerOnly;
    static constexpr bool runtimeCoordinateMapAvailable = false;

    // Same mode-major [mode][source node] organization as
    // RegionalGalerkinBasis::values. Values are sparse by region support, but
    // intentionally stored in that existing dense-per-mode compatibility
    // layout. The source mesh and source node numbering are never modified.
    std::vector<nm_float4> values;
    // Internal compiler storage; the exported solver payload retains its
    // existing sparse regional encoding. Omitted entries are exactly zero.
    bool sparseCoefficientEncoding = false;
    std::vector<std::uint32_t> modeValueOffsets;
    std::vector<std::uint32_t> coefficientNodeIndices;
    std::vector<nm_float4> sparseValues;

    [[nodiscard]] nm_float4 coefficient(std::uint32_t mode, std::uint32_t node) const noexcept {
        if (!sparseCoefficientEncoding)
            return values[static_cast<std::size_t>(mode) * nodeCount + node];
        const auto begin = coefficientNodeIndices.begin() + modeValueOffsets[mode];
        const auto end = coefficientNodeIndices.begin() + modeValueOffsets[mode + 1u];
        const auto found = std::lower_bound(begin, end, node);
        return found != end && *found == node
            ? sparseValues[static_cast<std::size_t>(found - coefficientNodeIndices.begin())]
            : nm_float4{};
    }
    [[nodiscard]] bool coefficientStorageValid() const noexcept {
        if (!sparseCoefficientEncoding)
            return values.size() == static_cast<std::size_t>(modeCount) * nodeCount;
        if (!values.empty() || modeValueOffsets.size() != static_cast<std::size_t>(modeCount) + 1u ||
            modeValueOffsets.front() != 0u || modeValueOffsets.back() != sparseValues.size() ||
            coefficientNodeIndices.size() != sparseValues.size()) return false;
        for (std::uint32_t mode=0; mode<modeCount; ++mode) {
            const auto first=modeValueOffsets[mode], end=modeValueOffsets[mode+1u];
            if (first>end || end>sparseValues.size()) return false;
            for (auto i=first; i<end; ++i)
                if (coefficientNodeIndices[i]>=nodeCount ||
                    (i>first && coefficientNodeIndices[i-1u]>=coefficientNodeIndices[i]))
                    return false;
        }
        return true;
    }
    void clearCoefficients() {
        values.clear(); modeValueOffsets.clear();
        coefficientNodeIndices.clear(); sparseValues.clear();
    }
    void appendCoefficientMode(const std::vector<std::array<double,3>>& field) {
        if (sparseCoefficientEncoding) {
            if (modeValueOffsets.empty()) modeValueOffsets.push_back(0u);
            for (std::uint32_t node=0; node<field.size(); ++node) {
                const auto& v=field[node];
                const nm_float4 value{float(v[0]),float(v[1]),float(v[2]),0.0f};
                if (value.x!=0.0f || value.y!=0.0f || value.z!=0.0f) {
                    coefficientNodeIndices.push_back(node); sparseValues.push_back(value);
                }
            }
            modeValueOffsets.push_back(static_cast<std::uint32_t>(sparseValues.size()));
        } else {
            for (const auto& v:field) values.push_back({float(v[0]),float(v[1]),float(v[2]),0.0f});
        }
    }
    std::vector<PartitionedReductionMode> modes;
    std::vector<PartitionedReductionRegion> regions;
    std::vector<std::uint32_t> anchorNodes;

    // Node-major [node][region + anchor] nonnegative partition weights.
    // Every free node sums to one; fixed nodes sum to zero. Anchor nodes have
    // a dedicated one-hot anchor partition and no regional support.
    std::vector<double> partitionWeights;
    std::uint32_t partitionCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t freeCoordinateCount = 0u;
    std::uint32_t modeCount = 0u;
    std::uint32_t minimumCoordinateReductionFactor = 4u;

    // Dense row-major B^T M B, formed from the exact float values exported in
    // `values` and the supplied lumped nodal masses.
    std::vector<double> projectedMass;
    std::string error;

    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

// Form the exact Float32-exported Gram matrix in ascending source-node
// order. Sparse construction visits actual entries, never mode*node capacity.
[[nodiscard]] inline std::vector<double> partitionedProjectedMass(
    const PartitionedReductionBasis& basis, const std::span<const double> mass
) {
    std::vector<double> gram(static_cast<std::size_t>(basis.modeCount)*basis.modeCount,0.0);
    std::vector<std::uint32_t> offsets(basis.nodeCount+1u,0u), modes, entries;
    if (basis.sparseCoefficientEncoding) {
        for (auto node:basis.coefficientNodeIndices) ++offsets[node+1u];
        for (std::uint32_t node=0;node<basis.nodeCount;++node) offsets[node+1u]+=offsets[node];
        modes.resize(basis.sparseValues.size()); entries.resize(modes.size());
        auto cursor=offsets;
        for (std::uint32_t mode=0;mode<basis.modeCount;++mode)
            for (auto i=basis.modeValueOffsets[mode];i<basis.modeValueOffsets[mode+1u];++i) {
                auto dst=cursor[basis.coefficientNodeIndices[i]]++;
                modes[dst]=mode;entries[dst]=i;
            }
    }
    std::vector<std::uint32_t> active;
    for (std::uint32_t node=0;node<basis.nodeCount;++node) {
        active.clear();
        if (!basis.sparseCoefficientEncoding)
            for (std::uint32_t mode=0;mode<basis.modeCount;++mode) {
                const auto v=basis.coefficient(mode,node);
                if (v.x!=0.0f||v.y!=0.0f||v.z!=0.0f) active.push_back(mode);
            }
        const auto count=basis.sparseCoefficientEncoding?offsets[node+1u]-offsets[node]:active.size();
        for (std::size_t i=0;i<count;++i) {
            const auto row=basis.sparseCoefficientEncoding?modes[offsets[node]+i]:active[i];
            const auto a=basis.sparseCoefficientEncoding?basis.sparseValues[entries[offsets[node]+i]]:basis.coefficient(row,node);
            for (std::size_t j=0;j<count;++j) {
                const auto column=basis.sparseCoefficientEncoding?modes[offsets[node]+j]:active[j];
                const auto b=basis.sparseCoefficientEncoding?basis.sparseValues[entries[offsets[node]+j]]:basis.coefficient(column,node);
                gram[static_cast<std::size_t>(row)*basis.modeCount+column]+=mass[node]*
                    (double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z);
            }
        }
    }
    return gram;
}

namespace partitioned_reduction_detail {

[[nodiscard]] inline std::array<double, 3> subtract(
    const std::array<double, 3>& a, const std::array<double, 3>& b
) noexcept {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] inline std::array<double, 3> cross(
    const std::array<double, 3>& a, const std::array<double, 3>& b
) noexcept {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] inline double dot(
    const std::array<double, 3>& a, const std::array<double, 3>& b
) noexcept {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] inline bool finite(const std::array<double, 3>& value) noexcept {
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

[[nodiscard]] inline double tetrahedronVolume(
    const std::span<const std::array<double, 3>> positions,
    const std::array<std::uint32_t, 4>& tet
) noexcept {
    const auto a = subtract(positions[tet[1]], positions[tet[0]]);
    const auto b = subtract(positions[tet[2]], positions[tet[0]]);
    const auto c = subtract(positions[tet[3]], positions[tet[0]]);
    return std::abs(dot(a, cross(b, c))) / 6.0;
}

[[nodiscard]] inline bool choleskyPositiveDefinite(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    if (dimension == 0u || matrix.size() != dimension * dimension) return false;
    std::vector<double> lower(dimension * dimension, 0.0);
    double scale = 0.0;
    for (std::size_t i = 0u; i < dimension; ++i)
        scale = std::max(scale, std::abs(matrix[i * dimension + i]));
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    const double threshold = scale * 1.0e-12;
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * dimension + column];
            for (std::size_t k = 0u; k < column; ++k)
                value -= lower[row * dimension + k] *
                    lower[column * dimension + k];
            if (row == column) {
                if (!(value > threshold) || !std::isfinite(value)) return false;
                lower[row * dimension + column] = std::sqrt(value);
            } else {
                lower[row * dimension + column] = value /
                    lower[column * dimension + column];
            }
        }
    }
    return true;
}

} // namespace partitioned_reduction_detail

// Builds regional rigid modes blended by volume-derived partition-of-unity
// weights, plus three identity-local modes for every free anchor. Positive
// tetrahedral volume, exact source arrays, and a minimum coordinate reduction
// factor are admission requirements. This creates preconditioner input only;
// it does not project nonlinear state, force, tangent, pressure, or contact.
[[nodiscard]] inline PartitionedReductionBasis buildPartitionedReductionBasis(
    const PartitionedReductionBasisInput& input,
    const PartitionedReductionBasisOptions& options = {}
) {
    using namespace partitioned_reduction_detail;
    PartitionedReductionBasis result;
    const std::size_t nodeCount = input.restPositions.size();
    const std::size_t tetCount = input.tetrahedra.size();
    const auto fail = [&](const char* message) {
        result.error = message;
        result.clearCoefficients();
        result.modes.clear();
        result.regions.clear();
        result.projectedMass.clear();
        return result;
    };
    if (nodeCount == 0u || input.lumpedMass.size() != nodeCount ||
        input.fixedMask.size() != nodeCount || tetCount == 0u ||
        input.tetrahedronRegionLabels.size() != tetCount) {
        return fail("partitioned basis requires exact nonempty source node, mass, fixed-mask, tetrahedron, and region-label arrays");
    }
    if (options.minimumCoordinateReductionFactor < 2u ||
        options.maximumModeCount == 0u || options.maximumModeCount > kPartitionedReductionMaximumConstructionModeCount) {
        return fail("partitioned basis reduction factor or mode limit is invalid");
    }
    if (nodeCount > std::numeric_limits<std::uint32_t>::max())
        return fail("partitioned basis source node count exceeds its index range");

    result.sparseCoefficientEncoding = options.sparseCoefficientEncoding;
    result.nodeCount = static_cast<std::uint32_t>(nodeCount);
    result.minimumCoordinateReductionFactor =
        options.minimumCoordinateReductionFactor;
    result.anchorNodes.assign(input.anchorNodes.begin(), input.anchorNodes.end());
    std::vector<std::uint8_t> isAnchor(nodeCount, 0u);
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        if (!finite(input.restPositions[node]) ||
            !std::isfinite(input.lumpedMass[node]) ||
            !(input.lumpedMass[node] > 0.0) || input.fixedMask[node] > 1u) {
            return fail("partitioned basis source position, positive mass, or binary fixed mask is invalid");
        }
        if (input.fixedMask[node] == 0u) {
            if (result.freeCoordinateCount >
                std::numeric_limits<std::uint32_t>::max() - 3u) {
                return fail("partitioned basis free coordinate count exceeds its index range");
            }
            result.freeCoordinateCount += 3u;
        }
    }
    for (const std::uint32_t node : input.anchorNodes) {
        if (node >= nodeCount || isAnchor[node] != 0u)
            return fail("partitioned basis anchor node is out of range or duplicated");
        isAnchor[node] = 1u;
    }

    std::map<std::uint32_t, std::size_t> regionIndex;
    for (const std::uint32_t label : input.tetrahedronRegionLabels)
        regionIndex.emplace(label, regionIndex.size());
    // Rebuild from sorted source labels so ordering is deterministic and
    // independent of tetrahedron visitation order.
    std::vector<std::uint32_t> sourceRegionLabels;
    sourceRegionLabels.reserve(regionIndex.size());
    for (const auto& entry : regionIndex) sourceRegionLabels.push_back(entry.first);
    regionIndex.clear();
    for (std::size_t index = 0u; index < sourceRegionLabels.size(); ++index)
        regionIndex.emplace(sourceRegionLabels[index], index);
    if (sourceRegionLabels.empty() || sourceRegionLabels.size() >
            options.maximumModeCount / 6u) {
        return fail("partitioned basis regional rigid modes exceed the authored mode limit");
    }

    const std::size_t regionCount = sourceRegionLabels.size();
    std::uint64_t freeAnchorCount = 0u;
    for (const std::uint32_t node : input.anchorNodes)
        if (input.fixedMask[node] == 0u) ++freeAnchorCount;
    const std::uint64_t baseModeCount =
        static_cast<std::uint64_t>(regionCount) * 6u + freeAnchorCount * 3u;
    if (baseModeCount == 0u || baseModeCount > options.maximumModeCount)
        return fail("partitioned basis exceeds the authored mode limit after anchor preservation");
    if (!options.sparseCoefficientEncoding && options.maximumDenseCoefficientBytes != 0u &&
        baseModeCount > options.maximumDenseCoefficientBytes /
             sizeof(nm_float4) / nodeCount)
        return fail("partitioned basis dense coefficient rows exceed the caller's checked byte budget");
    std::vector<double> nodeRegionVolume(nodeCount * regionCount, 0.0);
    std::vector<double> regionVolume(regionCount, 0.0);
    std::vector<std::array<double, 3>> regionCentroidNumerator(regionCount);
    for (std::size_t tetIndex = 0u; tetIndex < tetCount; ++tetIndex) {
        const auto& tet = input.tetrahedra[tetIndex];
        for (std::size_t corner = 0u; corner < 4u; ++corner) {
            if (tet[corner] >= nodeCount)
                return fail("partitioned basis tetrahedron references a missing source node");
            for (std::size_t other = 0u; other < corner; ++other) {
                if (tet[corner] == tet[other])
                    return fail("partitioned basis tetrahedron repeats a source node");
            }
        }
        const double volume = tetrahedronVolume(input.restPositions, tet);
        if (!(volume > 0.0) || !std::isfinite(volume))
            return fail("partitioned basis requires finite positive source tetrahedron volumes");
        const std::size_t region = regionIndex.at(
            input.tetrahedronRegionLabels[tetIndex]);
        regionVolume[region] += volume;
        std::array<double, 3> tetCenter{};
        for (const std::uint32_t node : tet) {
            const auto& position = input.restPositions[node];
            nodeRegionVolume[static_cast<std::size_t>(node) * regionCount +
                             region] += 0.25 * volume;
            for (std::size_t axis = 0u; axis < 3u; ++axis)
                tetCenter[axis] += 0.25 * position[axis];
        }
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            regionCentroidNumerator[region][axis] += volume * tetCenter[axis];
    }

    result.regions.resize(regionCount);
    for (std::size_t region = 0u; region < regionCount; ++region) {
        if (!(regionVolume[region] > 0.0) || !std::isfinite(regionVolume[region]))
            return fail("partitioned basis contains an empty or invalid anatomical region");
        auto& metadata = result.regions[region];
        metadata.sourceLabel = sourceRegionLabels[region];
        metadata.volume = regionVolume[region];
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            metadata.center[axis] = regionCentroidNumerator[region][axis] /
                regionVolume[region];
        double secondMoment = 0.0;
        for (std::size_t node = 0u; node < nodeCount; ++node) {
            const double weight = nodeRegionVolume[node * regionCount + region];
            const auto offset = subtract(input.restPositions[node], metadata.center);
            secondMoment += weight * dot(offset, offset);
        }
        metadata.rotationRadius = std::sqrt(secondMoment / regionVolume[region]);
        if (!(metadata.rotationRadius > 0.0) ||
            !std::isfinite(metadata.rotationRadius))
            return fail("partitioned basis anatomical region has no finite rotational extent");
    }

    if (regionCount + input.anchorNodes.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        return fail("partitioned basis partition count exceeds its index range");
    }
    result.partitionCount = static_cast<std::uint32_t>(
        regionCount + input.anchorNodes.size());
    result.partitionWeights.assign(nodeCount * result.partitionCount, 0.0);
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        if (input.fixedMask[node] != 0u) continue;
        if (isAnchor[node] != 0u) {
            const auto found = std::find(input.anchorNodes.begin(),
                                         input.anchorNodes.end(),
                                         static_cast<std::uint32_t>(node));
            const std::size_t anchor = static_cast<std::size_t>(
                found - input.anchorNodes.begin());
            result.partitionWeights[node * result.partitionCount +
                                    regionCount + anchor] = 1.0;
            continue;
        }
        double total = 0.0;
        for (std::size_t region = 0u; region < regionCount; ++region)
            total += nodeRegionVolume[node * regionCount + region];
        if (!(total > 0.0) || !std::isfinite(total))
            return fail("partitioned basis has a free source node without tetrahedral or anchor support");
        for (std::size_t region = 0u; region < regionCount; ++region)
            result.partitionWeights[node * result.partitionCount + region] =
                nodeRegionVolume[node * regionCount + region] / total;
    }

    const auto appendMode = [&](const PartitionedReductionMode& descriptor,
                                const std::vector<std::array<double, 3>>& field) {
        result.modes.push_back(descriptor);
        result.appendCoefficientMode(field);
    };
    for (std::size_t region = 0u; region < regionCount; ++region) {
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            std::vector<std::array<double, 3>> field(nodeCount);
            for (std::size_t node = 0u; node < nodeCount; ++node) {
                const double weight = input.fixedMask[node] == 0u
                    ? result.partitionWeights[node * result.partitionCount + region]
                    : 0.0;
                field[node][component] = weight;
            }
            appendMode({PartitionedReductionModeKind::regionalTranslation,
                        result.regions[region].sourceLabel, component,
                        NM_INVALID_INDEX, 1.0}, field);
        }
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            std::array<double, 3> axis{};
            axis[component] = 1.0;
            std::vector<std::array<double, 3>> field(nodeCount);
            for (std::size_t node = 0u; node < nodeCount; ++node) {
                const double weight = input.fixedMask[node] == 0u
                    ? result.partitionWeights[node * result.partitionCount + region]
                    : 0.0;
                const auto rotation = cross(
                    axis, subtract(input.restPositions[node],
                                   result.regions[region].center));
                for (std::size_t coordinate = 0u; coordinate < 3u; ++coordinate)
                    field[node][coordinate] = weight * rotation[coordinate] /
                        result.regions[region].rotationRadius;
            }
            appendMode({PartitionedReductionModeKind::regionalRotation,
                        result.regions[region].sourceLabel, component,
                        NM_INVALID_INDEX,
                        result.regions[region].rotationRadius}, field);
        }
    }
    for (const std::uint32_t anchorNode : input.anchorNodes) {
        if (input.fixedMask[anchorNode] != 0u) continue;
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            std::vector<std::array<double, 3>> field(nodeCount);
            field[anchorNode][component] = 1.0;
            appendMode({PartitionedReductionModeKind::anchorTranslation,
                        NM_INVALID_INDEX, component, anchorNode, 1.0}, field);
        }
    }
    result.modeCount = static_cast<std::uint32_t>(result.modes.size());
    if (result.modeCount == 0u || result.modeCount > options.maximumModeCount)
        return fail("partitioned basis exceeds the authored mode limit after anchor preservation");
    if (static_cast<std::uint64_t>(result.modeCount) *
            options.minimumCoordinateReductionFactor > result.freeCoordinateCount) {
        return fail("partitioned basis does not meet the requested minimum coordinate reduction factor");
    }

    if (result.sparseCoefficientEncoding && options.maximumDenseCoefficientBytes != 0u &&
        result.sparseValues.size() > options.maximumDenseCoefficientBytes /
            (sizeof(nm_float4)+sizeof(std::uint32_t)))
        return fail("partitioned sparse coefficient storage exceeds the checked byte budget");
    result.projectedMass = partitionedProjectedMass(result,input.lumpedMass);
    if (!choleskyPositiveDefinite(result.projectedMass, result.modeCount))
        return fail("partitioned basis modes are linearly dependent under the supplied nodal mass");
    return result;
}

// Computes B^T f using the exact mode-major float basis consumed by the
// existing RegionalGalerkinBasis preconditioner path.
[[nodiscard]] inline bool projectPartitionedNodalForce(
    const PartitionedReductionBasis& basis,
    const std::span<const std::array<double, 3>> nodalForce,
    const std::span<double> reducedForce
) noexcept {
    if (!basis.valid() || nodalForce.size() != basis.nodeCount ||
        reducedForce.size() != basis.modeCount ||
        !basis.coefficientStorageValid()) return false;
    std::fill(reducedForce.begin(), reducedForce.end(), 0.0);
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
            const auto b = basis.coefficient(mode,node);
            reducedForce[mode] += static_cast<double>(b.x) * nodalForce[node][0] +
                static_cast<double>(b.y) * nodalForce[node][1] +
                static_cast<double>(b.z) * nodalForce[node][2];
        }
    }
    return true;
}

// Forms a linear combination for basis algebra checks and preconditioner
// assembly. This is not a runtime state-coordinate projection API.
[[nodiscard]] inline bool combinePartitionedPreconditionerModes(
    const PartitionedReductionBasis& basis,
    const std::span<const double> reducedCoordinates,
    const std::span<std::array<double, 3>> nodalDisplacement
) noexcept {
    if (!basis.valid() || reducedCoordinates.size() != basis.modeCount ||
        nodalDisplacement.size() != basis.nodeCount ||
        !basis.coefficientStorageValid()) return false;
    std::fill(nodalDisplacement.begin(), nodalDisplacement.end(),
              std::array<double, 3>{});
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
            const auto b = basis.coefficient(mode,node);
            for (std::size_t axis = 0u; axis < 3u; ++axis)
                nodalDisplacement[node][axis] += reducedCoordinates[mode] *
                    (axis == 0u ? b.x : axis == 1u ? b.y : b.z);
        }
    }
    return true;
}

} // namespace numi::matter
