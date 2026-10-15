#include "../src/regional_graph_schwarz.hpp"

#include <array>
#include <cmath>
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

    // Dynamic-grip contract: the immutable graph is built from a superset of
    // all potentially free FEM nodes. Candidate masks may vary by environment;
    // every currently free node must remain in the prebuilt coverage map.
    // This CPU fixture validates layout/mask invariants, while the companion
    // native selector-5 fixture exercises the production Metal kernels.
    std::vector<std::uint8_t> potentialMask(nodeCount, 1u);
    const auto potentialLayout = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, potentialMask, 2u);
    if (!potentialLayout.error.empty() ||
        potentialLayout.nodeCoverage.size() != nodeCount ||
        potentialLayout.patchNodes.size() < nodeCount) {
        std::cerr << "potential-node graph layout failed\n";
        return 12;
    }
    for (std::uint32_t node = 0u; node < nodeCount; ++node)
        if (potentialLayout.nodeCoverage[node] == 0u) {
            std::cerr << "potential-node graph misses a dynamically free node\n";
            return 13;
        }
    std::vector<std::uint8_t> environment0(nodeCount, 0u);
    std::vector<std::uint8_t> environment1(nodeCount, 0u);
    std::vector<float> inverseMass0(nodeCount, 1.0f);
    std::vector<float> inverseMass1(nodeCount, 1.0f);
    std::vector<float> mass0(nodeCount, 1.0f);
    std::vector<float> mass1(nodeCount, 1.0f);
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        environment0[node] = freeMask[node] != 0u ? 0u : 1u;
        environment1[node] = freeMask[node] != 0u ? 0u : 1u;
        inverseMass0[node] = freeMask[node] != 0u ? 1.0f : 0.0f;
        inverseMass1[node] = inverseMass0[node];
    }
    environment0[0u] = 0u; // release an initially fixed node
    inverseMass0[0u] = 1.0f;
    environment0[8u] = 3u; // acquire a temporary grip on a free node
    inverseMass0[8u] = 0.0f;
    environment1[299u] = 0u; // different per-environment release
    inverseMass1[299u] = 1.0f;
    environment1[9u] = 3u;   // different per-environment acquisition
    inverseMass1[9u] = 0.0f;
    const auto candidateCovered = [&](const std::vector<std::uint8_t>& tags,
                                      const std::vector<float>& inverseMass,
                                      const std::vector<float>& mass) {
        if (tags.size() != nodeCount || inverseMass.size() != nodeCount ||
            mass.size() != nodeCount)
            return false;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            const std::uint8_t tag = tags[node];
            const float inverse = inverseMass[node];
            if ((tag != 0u && tag != 1u && tag != 3u) ||
                !std::isfinite(inverse) || !std::isfinite(mass[node]) ||
                !(mass[node] > 0.0f)) return false;
            if ((tag == 0u && (!(inverse > 0.0f) ||
                               potentialLayout.nodeCoverage[node] == 0u)) ||
                (tag != 0u && inverse != 0.0f))
                return false;
        }
        return true;
    };
    if (!candidateCovered(environment0, inverseMass0, mass0) ||
        !candidateCovered(environment1, inverseMass1, mass1) ||
        environment0 == environment1) {
        std::cerr << "per-environment dynamic fixed-mask coverage check failed\n";
        return 14;
    }
    auto unsupportedTags = environment0;
    unsupportedTags[4u] = 2u; // Human tag is outside selector-5 admission.
    if (candidateCovered(unsupportedTags, inverseMass0, mass0)) {
        std::cerr << "unsupported candidate constraint tag was admitted\n";
        return 15;
    }
    auto inconsistentInverseMass = inverseMass0;
    inconsistentInverseMass[8u] = 1.0f; // held grip must have zero inverse mass
    if (candidateCovered(environment0, inconsistentInverseMass, mass0)) {
        std::cerr << "constraint tag/inverse-mass mismatch was admitted\n";
        return 18;
    }
    auto zeroMass = mass1;
    zeroMass[9u] = 0.0f;
    if (candidateCovered(environment1, inverseMass1, zeroMass)) {
        std::cerr << "nonpositive candidate mass was admitted\n";
        return 19;
    }
    auto nonfiniteInverseMass = inverseMass1;
    nonfiniteInverseMass[299u] = std::numeric_limits<float>::quiet_NaN();
    if (candidateCovered(environment1, nonfiniteInverseMass, mass1)) {
        std::cerr << "nonfinite candidate inverse mass was admitted\n";
        return 20;
    }

    // Exact local algebra contract for a conditioning-only restricted patch:
    // retain A_ff, use identity on constrained rows/columns, and make the
    // preconditioned correction zero at constrained coordinates.
    constexpr std::array<double, 16> physicalA{
        7.0, 2.0, -1.0, 0.5,
        2.0, 6.0, 1.5, -0.25,
        -1.0, 1.5, 5.0, 3.0,
        0.5, -0.25, 3.0, 4.0};
    const std::array<std::uint8_t, 4> fixedTags{0u, 3u, 1u, 0u};
    std::array<double, 16> restricted{};
    for (std::size_t row = 0u; row < 4u; ++row) {
        if (fixedTags[row] != 0u) {
            restricted[row * 4u + row] = 1.0;
            continue;
        }
        for (std::size_t column = 0u; column < 4u; ++column)
            if (fixedTags[column] == 0u)
                restricted[row * 4u + column] =
                    physicalA[row * 4u + column];
    }
    if (restricted[0u] != 7.0 || restricted[3u] != 0.5 ||
        restricted[12u] != 0.5 || restricted[15u] != 4.0 ||
        restricted[5u] != 1.0 || restricted[10u] != 1.0 ||
        restricted[1u] != 0.0 || restricted[4u] != 0.0 ||
        restricted[2u] != 0.0 || restricted[8u] != 0.0) {
        std::cerr << "restricted free-free plus fixed-identity block contract failed\n";
        return 16;
    }
    const std::array<double, 4> rhs{1.0, 9.0, -3.0, 2.0};
    std::array<double, 4> correction{rhs[0], 0.0, 0.0, rhs[3]};
    if (correction[1] != 0.0 || correction[2] != 0.0) {
        std::cerr << "fixed rows received a nonzero preconditioned correction\n";
        return 17;
    }

    std::cout << "graph_schwarz_layout_checks=20"
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
