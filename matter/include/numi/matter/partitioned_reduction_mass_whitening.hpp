#pragma once

// Optional compiler-side rebasing of existing regional modes. This preserves
// each region's span and source rows; it does not remove coordinates.
#include "numi/matter/partitioned_reduction_basis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace numi::matter {

struct PartitionedRegionalMassWhiteningBlock {
    std::uint32_t sourceRegionLabel = NM_INVALID_INDEX;
    std::vector<std::uint32_t> modeIndices;
    // Row-major lower triangle of L where B_region^T M B_region = L L^T.
    // Retained so callers/tests can map between the original and rebased q.
    std::vector<double> choleskyLower;
};

struct PartitionedReductionMassWhiteningReport {
    std::vector<PartitionedRegionalMassWhiteningBlock> regions;
    double maximumRegionalIdentityResidual = 0.0;
    std::string error;

    [[nodiscard]] bool valid() const noexcept { return error.empty(); }
};

// Rebase each anatomical region independently so its exported FP32 modes have
// unit lumped-mass Gram matrix. For each source node and xyz component, this
// solves L y = b using forward substitution; it never forms L^-1. Anchor
// identity modes and all zero-valued sensitive-anchor rows remain untouched.
// Mode order, region labels, node identities, masses, and the regional span are
// preserved. The caller must rebuild sparse support from the resulting values.
[[nodiscard]] inline PartitionedReductionMassWhiteningReport
massWhitenPartitionedReductionBasis(
    PartitionedReductionBasis& basis,
    const std::span<const double> lumpedMass,
    const double maximumIdentityResidual = 5.0e-4
) {
    PartitionedReductionMassWhiteningReport report;
    const auto fail = [&](const char* message) {
        report.error = message;
        report.regions.clear();
        report.maximumRegionalIdentityResidual = 0.0;
        return report;
    };
    if (!basis.valid() || basis.nodeCount == 0u || basis.modeCount == 0u ||
        basis.modes.size() != basis.modeCount ||
        basis.values.size() != static_cast<std::size_t>(basis.modeCount) *
            basis.nodeCount || lumpedMass.size() != basis.nodeCount ||
        basis.regions.empty() || !std::isfinite(maximumIdentityResidual) ||
        !(maximumIdentityResidual > 0.0)) {
        return fail("mass whitening basis dimensions, metadata, or tolerance are invalid");
    }

    for (const double mass : lumpedMass) {
        if (!std::isfinite(mass) || !(mass > 0.0))
            return fail("mass whitening requires finite positive source nodal masses");
    }

    std::map<std::uint32_t, std::vector<std::uint32_t>> regionalModes;
    std::map<std::uint32_t, std::uint32_t> regionSlot;
    for (std::size_t region = 0u; region < basis.regions.size(); ++region) {
        const std::uint32_t label = basis.regions[region].sourceLabel;
        if (label == NM_INVALID_INDEX ||
            !regionSlot.emplace(label, static_cast<std::uint32_t>(region)).second)
            return fail("mass whitening anatomical region labels are invalid or duplicated");
        regionalModes.emplace(label, std::vector<std::uint32_t>{});
    }

    std::vector<std::uint8_t> anchorNode(basis.nodeCount, 0u);
    for (const std::uint32_t node : basis.anchorNodes) {
        if (node >= basis.nodeCount || anchorNode[node] != 0u)
            return fail("mass whitening sensitive-anchor identities are invalid or duplicated");
        anchorNode[node] = 1u;
    }

    std::vector<std::uint32_t> anchorModeIndices;
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        const PartitionedReductionMode& descriptor = basis.modes[mode];
        if (descriptor.kind == PartitionedReductionModeKind::anchorTranslation) {
            if (descriptor.regionLabel != NM_INVALID_INDEX ||
                descriptor.anchorNode >= basis.nodeCount ||
                anchorNode[descriptor.anchorNode] == 0u ||
                descriptor.component >= 3u) {
                return fail("mass whitening anchor mode lost its original source identity");
            }
            anchorModeIndices.push_back(mode);
            for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
                const nm_float4& value = basis.values[
                    static_cast<std::size_t>(mode) * basis.nodeCount + node];
                const float expectedX = node == descriptor.anchorNode &&
                    descriptor.component == 0u ? 1.0f : 0.0f;
                const float expectedY = node == descriptor.anchorNode &&
                    descriptor.component == 1u ? 1.0f : 0.0f;
                const float expectedZ = node == descriptor.anchorNode &&
                    descriptor.component == 2u ? 1.0f : 0.0f;
                if (value.x != expectedX || value.y != expectedY ||
                    value.z != expectedZ)
                    return fail("mass whitening anchor modes must remain exact one-hot identities");
            }
            continue;
        }
        const auto found = regionalModes.find(descriptor.regionLabel);
        if (found == regionalModes.end() ||
            (descriptor.kind != PartitionedReductionModeKind::regionalTranslation &&
             descriptor.kind != PartitionedReductionModeKind::regionalRotation &&
             descriptor.kind != PartitionedReductionModeKind::regionalAffineGradient)) {
            return fail("mass whitening found a regional mode with no source region identity");
        }
        found->second.push_back(mode);
        const std::size_t offset = static_cast<std::size_t>(mode) * basis.nodeCount;
        for (const std::uint32_t node : basis.anchorNodes) {
            const nm_float4& value = basis.values[offset + node];
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f)
                return fail("mass whitening requires exact zero regional coefficients at sensitive anchors");
        }
    }
    if (anchorModeIndices.size() > basis.anchorNodes.size() * 3u)
        return fail("mass whitening anchor mode count exceeds the preserved identity map");

    std::vector<nm_float4> transformed = basis.values;
    report.regions.reserve(basis.regions.size());
    for (const PartitionedReductionRegion& region : basis.regions) {
        const auto modesIt = regionalModes.find(region.sourceLabel);
        if (modesIt == regionalModes.end() || modesIt->second.empty())
            return fail("mass whitening region has no retained affine modes");
        const std::vector<std::uint32_t>& modes = modesIt->second;
        const std::size_t count = modes.size();
        if (count > 32u || count > std::numeric_limits<std::size_t>::max() / count)
            return fail("mass whitening regional block exceeds the supported bounded factor size");

        std::vector<double> gram(count * count, 0.0);
        for (std::size_t row = 0u; row < count; ++row) {
            const std::size_t rowOffset =
                static_cast<std::size_t>(modes[row]) * basis.nodeCount;
            for (std::size_t column = 0u; column <= row; ++column) {
                const std::size_t columnOffset =
                    static_cast<std::size_t>(modes[column]) * basis.nodeCount;
                double value = 0.0;
                for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
                    const nm_float4& a = basis.values[rowOffset + node];
                    const nm_float4& b = basis.values[columnOffset + node];
                    value += lumpedMass[node] *
                        (static_cast<double>(a.x) * b.x +
                         static_cast<double>(a.y) * b.y +
                         static_cast<double>(a.z) * b.z);
                }
                if (!std::isfinite(value))
                    return fail("mass whitening regional Gram matrix is non-finite");
                gram[row * count + column] = value;
                gram[column * count + row] = value;
            }
        }

        double scale = 0.0;
        for (std::size_t index = 0u; index < count; ++index)
            scale = std::max(scale, std::abs(gram[index * count + index]));
        if (!(scale > 0.0) || !std::isfinite(scale))
            return fail("mass whitening regional Gram matrix has no positive scale");
        const double pivotThreshold = scale * 1.0e-12;
        std::vector<double> lower(count * count, 0.0);
        for (std::size_t row = 0u; row < count; ++row) {
            for (std::size_t column = 0u; column <= row; ++column) {
                double value = gram[row * count + column];
                for (std::size_t k = 0u; k < column; ++k)
                    value -= lower[row * count + k] *
                        lower[column * count + k];
                if (row == column) {
                    if (!std::isfinite(value) || !(value > pivotThreshold))
                        return fail("mass whitening regional Gram matrix is rank deficient");
                    lower[row * count + column] = std::sqrt(value);
                } else {
                    value /= lower[column * count + column];
                    if (!std::isfinite(value))
                        return fail("mass whitening Cholesky factor is non-finite");
                    lower[row * count + column] = value;
                }
            }
        }

        std::vector<double> solved(count, 0.0);
        for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                for (std::size_t row = 0u; row < count; ++row) {
                    const nm_float4& original = basis.values[
                        static_cast<std::size_t>(modes[row]) * basis.nodeCount + node];
                    double value = axis == 0u ? original.x :
                        axis == 1u ? original.y : original.z;
                    for (std::size_t k = 0u; k < row; ++k)
                        value -= lower[row * count + k] * solved[k];
                    value /= lower[row * count + row];
                    if (!std::isfinite(value) ||
                        std::abs(value) > std::numeric_limits<float>::max())
                        return fail("mass whitening triangular solve produced an invalid coefficient");
                    solved[row] = value;
                }
                for (std::size_t row = 0u; row < count; ++row) {
                    nm_float4& output = transformed[
                        static_cast<std::size_t>(modes[row]) * basis.nodeCount + node];
                    const float value = static_cast<float>(solved[row]);
                    if (!std::isfinite(value))
                        return fail("mass whitening coefficient overflowed Float32");
                    if (axis == 0u) output.x = value;
                    else if (axis == 1u) output.y = value;
                    else output.z = value;
                }
            }
        }

        PartitionedRegionalMassWhiteningBlock block;
        block.sourceRegionLabel = region.sourceLabel;
        block.modeIndices = modes;
        block.choleskyLower = std::move(lower);
        report.regions.push_back(std::move(block));
    }

    std::vector<double> projectedMass(
        static_cast<std::size_t>(basis.modeCount) * basis.modeCount, 0.0);
    // Regional whitening preserves each mode's node support. Accumulate the
    // exact full Gram matrix, including overlapping-region cross terms, only
    // for modes supported at each node. Ascending node order retains the
    // original summation order of every matrix entry.
    std::vector<std::uint32_t> activeModes;
    for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
        activeModes.clear();
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const nm_float4& value = transformed[
                static_cast<std::size_t>(mode) * basis.nodeCount + node];
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f)
                activeModes.push_back(mode);
        }
        for (const std::uint32_t row : activeModes) {
            const nm_float4& a = transformed[
                static_cast<std::size_t>(row) * basis.nodeCount + node];
            for (const std::uint32_t column : activeModes) {
                const nm_float4& b = transformed[
                    static_cast<std::size_t>(column) * basis.nodeCount + node];
                projectedMass[static_cast<std::size_t>(row) *
                              basis.modeCount + column] += lumpedMass[node] *
                    (static_cast<double>(a.x) * b.x +
                     static_cast<double>(a.y) * b.y +
                     static_cast<double>(a.z) * b.z);
            }
        }
    }
    if (!std::all_of(projectedMass.begin(), projectedMass.end(),
                     [](double value) { return std::isfinite(value); }))
        return fail("rebased projected mass contains a non-finite entry");
    if (!partitioned_reduction_detail::choleskyPositiveDefinite(
            projectedMass, basis.modeCount))
        return fail("rebased full projected mass lost positive definiteness");

    for (const PartitionedRegionalMassWhiteningBlock& block : report.regions) {
        const std::size_t count = block.modeIndices.size();
        for (std::size_t row = 0u; row < count; ++row) {
            for (std::size_t column = 0u; column < count; ++column) {
                const std::uint32_t globalRow = block.modeIndices[row];
                const std::uint32_t globalColumn = block.modeIndices[column];
                const double actual = projectedMass[
                    static_cast<std::size_t>(globalRow) * basis.modeCount +
                    globalColumn];
                const double expected = row == column ? 1.0 : 0.0;
                report.maximumRegionalIdentityResidual = std::max(
                    report.maximumRegionalIdentityResidual,
                    std::abs(actual - expected));
            }
        }
    }
    if (!std::isfinite(report.maximumRegionalIdentityResidual) ||
        report.maximumRegionalIdentityResidual > maximumIdentityResidual)
        return fail("rebased regional projected mass missed the Float32 roundoff identity bound");

    basis.values.swap(transformed);
    basis.projectedMass.swap(projectedMass);
    return report;
}

} // namespace numi::matter
