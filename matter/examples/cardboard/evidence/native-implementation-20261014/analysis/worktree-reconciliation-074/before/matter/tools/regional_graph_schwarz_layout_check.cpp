#include "../src/regional_graph_schwarz.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

using numi::matter::detail::buildRegionalGraphSchwarzLayout;
using numi::matter::detail::regionalGraphSchwarzCheckedAdd;
using numi::matter::detail::regionalGraphSchwarzCheckedMul;

int main() {
    // Small CSR and fail-closed cases exercise fixed/free, disconnected,
    // deterministic, and overflow behavior.
    constexpr std::uint32_t nodeCount = 302u;
    std::vector<std::array<std::uint32_t, 4>> tetrahedra;
    for (std::uint32_t node = 0u; node + 3u < 299u; ++node)
        tetrahedra.push_back({node, node + 1u, node + 2u, node + 3u});
    std::vector<std::uint8_t> freeMask(nodeCount, 1u);
    freeMask[0] = 0u;
    freeMask[299] = 0u;
    // Nodes 300 and 301 are disconnected from tetrahedra. The layout must
    // preserve both free rows; any uncovered free node receives a singleton.
    const auto first = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, 2u);
    const auto second = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, 2u);
    if (!first.error.empty() || !second.error.empty() ||
        first.patches.size() != second.patches.size() ||
        first.patchNodes != second.patchNodes ||
        first.nodePatchOffsets != second.nodePatchOffsets ||
        first.nodePatchValues != second.nodePatchValues ||
        first.rows != second.rows || first.nodeCoverage != second.nodeCoverage) {
        std::cerr << "deterministic graph layout check failed\n";
        return 1;
    }
    if (first.patches.empty() || first.patchNodes.size() < 300u ||
        first.totalPatchDofs != first.patchNodes.size() * 3u ||
        first.rows.size() != first.totalPatchDofs ||
        first.nodeCoverage[0] != 0u || first.nodeCoverage[299] != 0u ||
        first.nodeCoverage[300] == 0u || first.nodeCoverage[301] == 0u) {
        std::cerr << "free/fixed coverage check failed\n";
        return 2;
    }
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        if ((freeMask[node] != 0u) != (first.nodeCoverage[node] != 0u)) {
            std::cerr << "coverage includes fixed nodes or misses free nodes\n";
            return 3;
        }
        const std::uint32_t begin = first.nodePatchOffsets[node];
        const std::uint32_t end = first.nodePatchOffsets[node + 1u];
        if (end - begin != first.nodeCoverage[node]) {
            std::cerr << "node-to-patch CSR disagrees with coverage counts\n";
            return 4;
        }
        for (std::uint32_t item = begin; item < end; ++item) {
            const std::uint32_t flatPatchNode = first.nodePatchValues[item];
            if (flatPatchNode >= first.patchNodes.size() ||
                first.patchNodes[flatPatchNode] != node) {
                std::cerr << "node-to-patch CSR points at the wrong local node\n";
                return 5;
            }
        }
    }
    for (std::size_t p = 0u; p < first.patches.size(); ++p) {
        const auto patch = first.patches[p];
        const std::uint64_t dofs = 3ull * patch.nodeCount;
        const std::uint64_t expectedMatrixOffset =
            p == 0u ? 0ull : first.patches[p - 1u].matrixFloatOffset +
                9ull * first.patches[p - 1u].nodeCount *
                    first.patches[p - 1u].nodeCount;
        const std::uint64_t expectedPivotOffset =
            p == 0u ? 0ull : first.patches[p - 1u].pivotOffset +
                3ull * first.patches[p - 1u].nodeCount;
        if (patch.matrixFloatOffset != expectedMatrixOffset ||
            patch.pivotOffset != expectedPivotOffset ||
            patch.nodeCount == 0u || patch.nodeCount > 256u ||
            dofs != 3ull * patch.nodeCount) {
            std::cerr << "packed patch offsets or local dimensions are invalid\n";
            return 6;
        }
    }

    std::uint64_t output = 0u;
    if (regionalGraphSchwarzCheckedAdd(5u, 7u, output) != true || output != 12u ||
        regionalGraphSchwarzCheckedAdd(
            std::numeric_limits<std::uint64_t>::max(), 1u, output) ||
        regionalGraphSchwarzCheckedMul(7u, 9u, output) != true || output != 63u ||
        regionalGraphSchwarzCheckedMul(
            std::numeric_limits<std::uint64_t>::max(), 2u, output)) {
        std::cerr << "64-bit checked sizing arithmetic failed\n";
        return 7;
    }

    const auto tooLarge = buildRegionalGraphSchwarzLayout(
        4097u, tetrahedra, std::span<const std::uint8_t>(freeMask), 1u);
    const auto tinyBudget = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, 2u, 1u);
    auto badTet = tetrahedra;
    badTet[0][3] = nodeCount;
    const auto invalidTet = buildRegionalGraphSchwarzLayout(
        nodeCount, badTet, freeMask, 2u);
    if (tooLarge.error.empty() || tinyBudget.error.empty() ||
        invalidTet.error.empty()) {
        std::cerr << "fail-closed bounds checks did not reject malformed input\n";
        return 8;
    }

    // Synthetic allocation-shape regression only: reuse the recorded node
    // and patch-size counts (846 nodes, eight 256-node patches, five singleton
    // patches) with a constructed chain and five excluded node IDs. This is
    // not the studied mesh topology or exact graph membership; it only checks
    // deterministic allocation offsets at those dimensions.
    constexpr std::uint32_t syntheticShapeNodeCount = 846u;
    std::vector<std::uint32_t> connected;
    for (std::uint32_t node = 0u; node < syntheticShapeNodeCount; ++node) {
        if (node != 10u && node != 30u && node != 50u &&
            node != 70u && node != 90u)
            connected.push_back(node);
    }
    std::vector<std::array<std::uint32_t, 4>> syntheticShapeTets;
    for (std::size_t index = 0u; index + 3u < connected.size(); ++index)
        syntheticShapeTets.push_back({connected[index], connected[index + 1u],
                                 connected[index + 2u],
                                 connected[index + 3u]});
    std::vector<std::uint8_t> syntheticShapeFreeMask(syntheticShapeNodeCount, 1u);
    const auto syntheticAllocationShape = buildRegionalGraphSchwarzLayout(
        syntheticShapeNodeCount, syntheticShapeTets, syntheticShapeFreeMask, 2u);
    const std::array<std::uint32_t, 8> expectedPatchSizes{
        256u, 256u, 256u, 256u, 256u, 256u, 256u, 256u};
    if (!syntheticAllocationShape.error.empty() || syntheticAllocationShape.patches.size() != 13u ||
        syntheticAllocationShape.patchNodes.size() != 2053u ||
        syntheticAllocationShape.totalPatchDofs != 6159u ||
        syntheticAllocationShape.matrixFloatCount != 4'718'637u ||
        syntheticAllocationShape.pivotCount != 6159u ||
        syntheticAllocationShape.deviceBytes >= 96ull * 1024ull * 1024ull) {
        std::cerr << "synthetic allocation-shape workspace regression failed\n";
        return 9;
    }
    for (std::size_t index = 0u; index < expectedPatchSizes.size(); ++index)
        if (syntheticAllocationShape.patches[index].nodeCount != expectedPatchSizes[index]) {
            std::cerr << "synthetic allocation-shape large-patch size regression failed\n";
            return 10;
        }
    for (std::size_t index = expectedPatchSizes.size();
         index < syntheticAllocationShape.patches.size(); ++index)
        if (syntheticAllocationShape.patches[index].nodeCount != 1u) {
            std::cerr << "synthetic allocation-shape singleton size regression failed\n";
            return 11;
        }

    std::cout << "graph_schwarz_layout_checks=11"
              << " patches=" << first.patches.size()
              << " patch_nodes=" << first.patchNodes.size()
              << " matrix_floats=" << first.matrixFloatCount
              << " pivots=" << first.pivotCount
              << " device_bytes=" << first.deviceBytes
              << " synthetic_allocation_shape_patches=" << syntheticAllocationShape.patches.size()
              << " synthetic_allocation_shape_patch_nodes=" << syntheticAllocationShape.patchNodes.size()
              << " synthetic_allocation_shape_matrix_floats=" << syntheticAllocationShape.matrixFloatCount
              << " synthetic_allocation_shape_pivots=" << syntheticAllocationShape.pivotCount
              << " synthetic_allocation_shape_device_bytes=" << syntheticAllocationShape.deviceBytes << '\n';
    return 0;
}
