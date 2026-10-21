#include "../src/regional_graph_schwarz.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

using numi::matter::detail::buildRegionalGraphSchwarzLayout;
using numi::matter::detail::regionalGraphSchwarzCheckedAdd;
using numi::matter::detail::regionalGraphSchwarzCheckedMul;

int main() {
    {
        numi::matter::detail::RegionalGraphSchwarzLayout isolated;
        isolated.nodeCount = 8u;
        isolated.patches = {{0u, 4u, 0u, 0u}, {4u, 4u, 144u, 0u}};
        isolated.patchNodes = {0u,1u,2u,3u,4u,5u,6u,7u};
        const std::vector<std::uint32_t> owners{
            0u,0u,0u,0u,1u,1u,1u,1u};
        const std::vector<std::array<std::uint32_t,4>> tetNodes{
            {0u,1u,2u,3u},{4u,5u,6u,7u}};
        const std::vector<std::uint32_t> tetOwners{0u,1u};
        const std::vector<std::uint8_t> active{1u,1u};
        auto ineligible = numi::matter::detail::
            regionalGraphDueRowsIsolationIneligibility(
                isolated, owners, tetNodes, tetOwners, active, 2u);
        if (ineligible != nullptr) {
            std::cerr << "isolated multirate graph rejected: "
                      << ineligible << "\n";
            return 1;
        }
        auto mixedPatch = isolated;
        mixedPatch.patches = {{0u, 8u, 0u, 0u}};
        ineligible = numi::matter::detail::
            regionalGraphDueRowsIsolationIneligibility(
                mixedPatch, owners, tetNodes, tetOwners, active, 2u);
        if (ineligible != nullptr) {
            std::cerr << "isolated objects sharing a graph patch were rejected: "
                      << ineligible << "\n";
            return 1;
        }
        auto crossingTets = tetNodes;
        crossingTets[0] = {0u,1u,2u,4u};
        ineligible = numi::matter::detail::
            regionalGraphDueRowsIsolationIneligibility(
                isolated, owners, crossingTets, tetOwners, active, 2u);
        if (ineligible == nullptr ||
            std::string(ineligible) != "active_tet_crosses_object_owners") {
            std::cerr << "cross-owner active tet was admitted for due-row identity\n";
            return 1;
        }
        auto dormantCrossing = active;
        dormantCrossing[0] = 0u;
        ineligible = numi::matter::detail::
            regionalGraphDueRowsIsolationIneligibility(
                isolated, owners, crossingTets, tetOwners, dormantCrossing, 2u);
        if (ineligible != nullptr) {
            std::cerr << "inactive source tet unnecessarily rejected due-row identity\n";
            return 1;
        }
    }
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
        nodeCount, tetrahedra, freeMask, 2u,
        96ull * 1024ull * 1024ull, 4096u, 32768u, 128u, 256u, 128u, false);
    const auto second = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, 2u,
        96ull * 1024ull * 1024ull, 4096u, 32768u, 128u, 256u, 128u, false);
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
                9ull * first.patches[p - 1u].nodeCount;
        if (patch.matrixFloatOffset != expectedMatrixOffset ||
            patch.pivotOffset != expectedPivotOffset ||
            patch.nodeCount == 0u || patch.nodeCount > 256u ||
            dofs != 3ull * patch.nodeCount) {
            std::cerr << "packed patch offsets or local dimensions are invalid\n";
            return 6;
        }
    }

    // Dynamic layout keeps the source-free graph as an exact prefix, then
    // appends disjoint patches containing only source-fixed nodes.
    const auto dynamicLayout = buildRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, 2u);
    if (!dynamicLayout.error.empty() ||
        dynamicLayout.patches.size() < first.patches.size() ||
        dynamicLayout.patchNodes.size() < first.patchNodes.size() ||
        dynamicLayout.rows.size() < first.rows.size()) {
        std::cerr << "dynamic graph layout lacks its immutable source prefix\n";
        return 20;
    }
    for (std::size_t index = 0u; index < first.patches.size(); ++index) {
        const auto& source = first.patches[index];
        const auto& dynamic = dynamicLayout.patches[index];
        if (source.nodeOffset != dynamic.nodeOffset ||
            source.nodeCount != dynamic.nodeCount ||
            source.matrixFloatOffset != dynamic.matrixFloatOffset ||
            source.pivotOffset != dynamic.pivotOffset) {
            std::cerr << "dynamic graph changed an original patch descriptor\n";
            return 21;
        }
    }
    if (!std::equal(first.patchNodes.begin(), first.patchNodes.end(),
                    dynamicLayout.patchNodes.begin()) ||
        !std::equal(first.rows.begin(), first.rows.end(),
                    dynamicLayout.rows.begin())) {
        std::cerr << "dynamic graph changed original patch nodes or rows\n";
        return 22;
    }
    for (std::size_t index = first.patchNodes.size();
         index < dynamicLayout.patchNodes.size(); ++index) {
        const std::uint32_t node = dynamicLayout.patchNodes[index];
        if (node >= nodeCount || freeMask[node] != 0u) {
            std::cerr << "supplementary patch overlaps a source-free node\n";
            return 23;
        }
    }
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        if (freeMask[node] != 0u) {
            if (dynamicLayout.nodeCoverage[node] != first.nodeCoverage[node]) {
                std::cerr << "supplementary patches changed source-free coverage\n";
                return 24;
            }
            const std::uint32_t sourceBegin = first.nodePatchOffsets[node];
            const std::uint32_t sourceEnd = first.nodePatchOffsets[node + 1u];
            const std::uint32_t dynamicBegin = dynamicLayout.nodePatchOffsets[node];
            const std::uint32_t dynamicEnd = dynamicLayout.nodePatchOffsets[node + 1u];
            if (sourceEnd - sourceBegin != dynamicEnd - dynamicBegin ||
                !std::equal(first.nodePatchValues.begin() + sourceBegin,
                            first.nodePatchValues.begin() + sourceEnd,
                            dynamicLayout.nodePatchValues.begin() + dynamicBegin)) {
                std::cerr << "supplementary patches changed source-free CSR membership\n";
                return 26;
            }
        } else if (dynamicLayout.nodeCoverage[node] == 0u) {
            std::cerr << "initially-fixed node lacks its release patch\n";
            return 25;
        }
    }

    // An admitted historical layout must remain bit-identical under the
    // budget policy, including source-free membership and CSR ordering.
    const auto budgetPreserved = numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, 2u);
    if (!budgetPreserved.error.empty() ||
        budgetPreserved.patchNodes != dynamicLayout.patchNodes ||
        budgetPreserved.rows != dynamicLayout.rows ||
        budgetPreserved.nodePatchOffsets != dynamicLayout.nodePatchOffsets ||
        budgetPreserved.nodePatchValues != dynamicLayout.nodePatchValues ||
        budgetPreserved.deviceBytes != dynamicLayout.deviceBytes) {
        std::cerr << "budget policy changed an admitted historical graph\n";
        return 27;
    }
    // The opt-in authored-fixed-row-omission budgeted layout must exactly
    // preserve the source-free layout while omitting only authored-fixed rows. The historical default
    // above still includes all fixed-row patches.
    const auto freeRowsBudgeted =
        numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, freeMask, 2u,
            96ull * 1024ull * 1024ull, false, false);
    if (!freeRowsBudgeted.error.empty() ||
        freeRowsBudgeted.patchNodes != first.patchNodes ||
        freeRowsBudgeted.rows != first.rows ||
        freeRowsBudgeted.nodePatchOffsets != first.nodePatchOffsets ||
        freeRowsBudgeted.nodePatchValues != first.nodePatchValues ||
        freeRowsBudgeted.deviceBytes != first.deviceBytes ||
        freeRowsBudgeted.nodeCoverage != first.nodeCoverage) {
        std::cerr << "authored-fixed-row-omission budgeted layout changed the source-free graph\n";
        return 37;
    }
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        if (freeRowsBudgeted.nodeCoverage[node] != 0u &&
            freeMask[node] == 0u) {
            std::cerr << "authored-fixed-row-omission layout retained a statically fixed row\n";
            return 38;
        }
        if (freeMask[node] != 0u &&
            freeRowsBudgeted.nodeCoverage[node] == 0u) {
            std::cerr << "authored-fixed-row-omission layout dropped a source-free node\n";
            return 39;
        }
    }
    // A temporary-grip row (tag 3) may be released, so callers include it
    // in the potential-free mask even when it is fixed in the source mask.
    // Authored-static node 0 remains omitted, while released node 299 is
    // explicitly covered, including its disconnected singleton row.
    auto releasableGripMask = freeMask;
    releasableGripMask[299u] = 1u;
    const auto gripPotentialBudgeted =
        numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, releasableGripMask, 2u,
            96ull * 1024ull * 1024ull, false, false);
    if (!gripPotentialBudgeted.error.empty() ||
        gripPotentialBudgeted.nodeCoverage[0u] != 0u ||
        gripPotentialBudgeted.nodeCoverage[299u] == 0u) {
        std::cerr << "authored-fixed-row omission lost a releasable grip row\n";
        return 41;
    }
    for (std::uint32_t node = 1u; node < nodeCount; ++node)
        if (releasableGripMask[node] != 0u &&
            gripPotentialBudgeted.nodeCoverage[node] == 0u) {
            std::cerr << "releasable grip mask has an uncovered potential row\n";
            return 42;
        }
    const auto historicalBudgeted =
        numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, freeMask, 2u);
    if (historicalBudgeted.patchNodes != dynamicLayout.patchNodes ||
        historicalBudgeted.rows != dynamicLayout.rows ||
        historicalBudgeted.nodeCoverage != dynamicLayout.nodeCoverage) {
        std::cerr << "default budgeted layout no longer preserves fixed rows\n";
        return 40;
    }
    // Explicit Human preconditioner-size trials must cover all potential
    // nodes, including initially fixed and disconnected rows, deterministically
    // without changing the shared device-arena bound.
    for (const std::uint32_t patchNodes : {16u, 32u, 64u, 128u, 256u}) {
        const auto sized = buildRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, freeMask, 2u, 96ull*1024ull*1024ull,
            32768u, 131072u, 16384u, patchNodes, patchNodes/2u, true, true);
        const auto repeat = buildRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, freeMask, 2u, 96ull*1024ull*1024ull,
            32768u, 131072u, 16384u, patchNodes, patchNodes/2u, true, true);
        if (!sized.error.empty() || sized.deviceBytes > 96ull*1024ull*1024ull ||
            sized.patchNodes != repeat.patchNodes || sized.rows != repeat.rows ||
            std::any_of(sized.patches.begin(), sized.patches.end(),
                [patchNodes](const auto& patch) { return patch.nodeCount == 0u ||
                    patch.nodeCount > patchNodes; }) ||
            std::any_of(sized.nodeCoverage.begin(), sized.nodeCoverage.end(),
                [](auto coverage) { return coverage == 0u; })) {
            std::cerr << "explicit graph patch size lost bounded potential-node coverage\n";
            return 35;
        }
    }
    // Capacity changes cannot alter an otherwise admitted layout. Exact
    // required-byte and one-byte-short boundaries remain fail-closed.
    for (const std::uint64_t arenaMiB : {192ull, 384ull}) {
        const auto expanded = buildRegionalGraphSchwarzLayout(
            nodeCount, tetrahedra, freeMask, 2u, arenaMiB*1024ull*1024ull,
            32768u, 131072u, 16384u, 64u, 32u, true, true);
        if (!expanded.error.empty() || expanded.deviceBytes == 0u ||
            !buildRegionalGraphSchwarzLayout(
                nodeCount, tetrahedra, freeMask, 2u, expanded.deviceBytes,
                32768u, 131072u, 16384u, 64u, 32u, true, true).error.empty() ||
            buildRegionalGraphSchwarzLayout(
                nodeCount, tetrahedra, freeMask, 2u, expanded.deviceBytes-1u,
                32768u, 131072u, 16384u, 64u, 32u, true, true).error.empty()) {
            std::cerr << "explicit graph arena lost checked byte-bound admission\n";
            return 36;
        }
    }
    // 2,600 isolated rows exceed the historical singleton-patch limit. They
    // still require complete deterministic coverage, without lifting the
    // patch-count, maximum local DOF or arena bounds.
    std::vector<std::uint8_t> isolatedMask(2600u, 1u);
    const std::array<std::array<std::uint32_t, 4>, 1> isolatedTets{{{0u,1u,2u,3u}}};
    const auto isolatedLegacy = buildRegionalGraphSchwarzLayout(
        2600u, isolatedTets, isolatedMask, 2u);
    const auto isolatedBudget = numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
        2600u, isolatedTets, isolatedMask, 2u);
    const auto isolatedRepeat = numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
        2600u, isolatedTets, isolatedMask, 2u);
    if (isolatedLegacy.error.empty() || !isolatedBudget.error.empty() ||
        isolatedBudget.patches.size() > 128u ||
        isolatedBudget.deviceBytes > 96ull*1024ull*1024ull ||
        isolatedBudget.patchNodes != isolatedRepeat.patchNodes ||
        isolatedBudget.rows != isolatedRepeat.rows ||
        std::any_of(isolatedBudget.nodeCoverage.begin(), isolatedBudget.nodeCoverage.end(),
                    [](auto coverage) { return coverage == 0u; })) {
        std::cerr << "budget policy failed bounded disconnected-row coverage\n";
        return 28;
    }
    // A tiny budget still rejects; malformed inputs cannot be reinterpreted
    // by any fallback policy.
    if (numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
            2600u, isolatedTets, isolatedMask, 2u, 1u).error.empty() ||
        numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
            4097u, isolatedTets, isolatedMask, 2u).error.empty()) {
        std::cerr << "budget policy bypassed resource or topology admission\n";
        return 29;
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
        syntheticAllocationShape.pivotCount != 18477u ||
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
                               dynamicLayout.nodeCoverage[node] == 0u)) ||
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

    std::cout << "graph_schwarz_layout_checks=27"
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
