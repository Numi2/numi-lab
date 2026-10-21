#include "../src/regional_graph_schwarz.hpp"
#include "../include/numi/matter/cavity_pressure_mode.hpp"

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

bool checkCavityPressureMode() {
    using namespace numi::matter;
    using detail::buildCavityPressureLoadMode;
    std::vector<NMFEMNodeStateGPU> nodes(4u);
    const std::array<std::array<float, 3>, 4> position{{
        {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}}, {{0.0f, 0.0f, 1.0f}}}};
    std::vector<NMFEMTopologyNodeGPU> topology(4u);
    for (std::uint32_t node = 0u; node < 4u; ++node) {
        nodes[node].restAndFixed = {position[node][0], position[node][1],
                                    position[node][2], 0.0f};
        nodes[node].positionAndMass.w = 1.0f;
        topology[node].identity = {node, 7u, 1u, 0u};
    }
    NMVascularCavityGPU cavity{};
    cavity.identity = {11u, 0u, 7u, 1u};
    cavity.faces = {0u, 4u, 0u, 0u};
    std::vector<NMVascularCavityFaceGPU> faces(4u);
    const std::array<std::array<std::uint32_t, 3>, 4> triangles{{
        {{1u, 2u, 3u}}, {{0u, 3u, 2u}},
        {{0u, 1u, 3u}}, {{0u, 2u, 1u}}}};
    for (std::uint32_t i = 0u; i < 4u; ++i) {
        faces[i].nodes = {triangles[i][0], triangles[i][1], triangles[i][2], 0u};
        faces[i].identity = {0u, i, 1u, 0u};
    }
    std::vector<nm_float4> mode;
    std::string error;
    if (!buildCavityPressureLoadMode(nodes, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, mode, error) || mode.size() != nodes.size()) {
        std::cerr << "valid cavity rejected: " << error << "\n";
        return false;
    }
    const auto rawMode = mode;
    auto nonuniformMassNodes = nodes;
    const std::array<float, 4> masses{1.0f, 2.0f, 4.0f, 8.0f};
    for (std::size_t node = 0u; node < masses.size(); ++node)
        nonuniformMassNodes[node].positionAndMass.w = masses[node];
    std::vector<nm_float4> weightedMode;
    if (!buildCavityPressureLoadMode(nonuniformMassNodes, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, weightedMode, error, true) || weightedMode.size() != nodes.size()) {
        std::cerr << "valid mass-weighted cavity rejected: " << error << "\n";
        return false;
    }
    bool weightingChangedDirection = false;
    for (std::size_t node = 0u; node < masses.size(); ++node) {
        const std::array<float, 3> raw{
            rawMode[node].x, rawMode[node].y, rawMode[node].z};
        const std::array<float, 3> weighted{
            weightedMode[node].x, weightedMode[node].y, weightedMode[node].z};
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            const double recovered = double(weighted[axis]) * masses[node];
            if (std::abs(recovered - raw[axis]) >
                2.0e-6 * std::max(1.0, std::abs(double(raw[axis])))) {
                std::cerr << "inverse-mass cavity direction does not equal M^-1 g\n";
                return false;
            }
            weightingChangedDirection = weightingChangedDirection ||
                (node != 0u && std::abs(double(weighted[axis]) - raw[axis]) > 1.0e-7);
        }
    }
    if (!weightingChangedDirection) {
        std::cerr << "nonuniform mass fixture did not change cavity direction\n";
        return false;
    }
    std::array<double, 3> sum{};
    double dilation = 0.0;
    for (std::size_t i = 0; i < mode.size(); ++i) {
        const std::array<double, 3> v{mode[i].x, mode[i].y, mode[i].z};
        for (unsigned a = 0; a < 3u; ++a) sum[a] += v[a];
        dilation += v[0] * (position[i][0] - 0.25) +
                    v[1] * (position[i][1] - 0.25) +
                    v[2] * (position[i][2] - 0.25);
    }
    if (std::abs(sum[0]) > 1e-6 || std::abs(sum[1]) > 1e-6 ||
        std::abs(sum[2]) > 1e-6 || !(dilation > 0.0)) {
        std::cerr << "pressure mode translation/dilation invariant failed\n";
        return false;
    }
    auto fixed = nodes;
    fixed[0].restAndFixed.w = 1.0f;
    if (!buildCavityPressureLoadMode(fixed, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, mode, error) || mode[0].x != 0.0f ||
        mode[0].y != 0.0f || mode[0].z != 0.0f) {
        std::cerr << "fixed-node support was not removed: " << error << "\n";
        return false;
    }
    auto hydraulicClosure = faces;
    hydraulicClosure[0].identity.z = 3u;
    if (!buildCavityPressureLoadMode(nodes, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u),
            hydraulicClosure, 7u, mode, error)) {
        std::cerr << "valid hydraulic closure role rejected: " << error << "\n";
        return false;
    }
    auto unsupportedRole = faces;
    unsupportedRole[0].identity.z = 2u;
    if (buildCavityPressureLoadMode(nodes, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u),
            unsupportedRole, 7u, mode, error)) {
        std::cerr << "unsupported cavity face role admitted\n";
        return false;
    }
    auto masslessNodes = nodes;
    masslessNodes[0].positionAndMass.w = 0.0f;
    if (!buildCavityPressureLoadMode(masslessNodes, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, mode, error) ||
        mode[0].x != 0.0f || mode[0].y != 0.0f || mode[0].z != 0.0f ||
        (mode[1].x == 0.0f && mode[1].y == 0.0f && mode[1].z == 0.0f)) {
        std::cerr << "zero-mass cavity row mask is incorrect: " << error << "\n";
        return false;
    }
    auto dynamicMarker = nodes;
    dynamicMarker[0].restAndFixed.w = 2.0f;
    if (!buildCavityPressureLoadMode(dynamicMarker, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, mode, error) ||
        (mode[0].x == 0.0f && mode[0].y == 0.0f && mode[0].z == 0.0f)) {
        std::cerr << "dynamic marker incorrectly masked\n";
        return false;
    }
    auto open = faces;
    open.pop_back();
    if (buildCavityPressureLoadMode(nodes, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), open,
            7u, mode, error)) {
        std::cerr << "open boundary admitted\n";
        return false;
    }
    auto foreign = topology;
    foreign[2].identity.y = 8u;
    if (buildCavityPressureLoadMode(nodes, foreign,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, mode, error)) {
        std::cerr << "foreign object node admitted\n";
        return false;
    }
    auto flat = nodes;
    flat[3].restAndFixed.z = 0.0f;
    if (buildCavityPressureLoadMode(flat, topology,
            std::span<const NMVascularCavityGPU>(&cavity, 1u), faces,
            7u, mode, error)) {
        std::cerr << "zero-volume cavity admitted\n";
        return false;
    }
    return true;
}

int main() {
    if (!checkCavityPressureMode()) return 1;
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

    // Select optic graph neighborhoods from a complete skin-plus-optic
    // payload without removing the skin labels or modes.
    {
        using numi::matter::detail::regionalGraphSchwarzSelectOpticRegionIndices;
        auto verifyFullPayload = [&](const std::uint32_t binCount) {
            std::vector<std::uint32_t> labels{100u, 200u, 300u};
            std::vector<std::uint32_t> expectedOpticLabels;
            for (std::uint32_t bin = 0u; bin < binCount; ++bin) {
                expectedOpticLabels.push_back(0xF1772001u + bin);
                expectedOpticLabels.push_back(0xF1819001u + bin);
            }
            labels.insert(labels.end(), expectedOpticLabels.begin(),
                          expectedOpticLabels.end());
            std::sort(labels.begin(), labels.end());
            std::vector<std::uint32_t> selected;
            std::uint32_t selectedBins = 0u;
            std::string selectError;
            if (!regionalGraphSchwarzSelectOpticRegionIndices(
                    labels, selected, selectedBins, selectError) ||
                selectedBins != binCount ||
                selected.size() != expectedOpticLabels.size() ||
                !selectError.empty() ||
                !std::binary_search(labels.begin(), labels.end(), 100u) ||
                !std::binary_search(labels.begin(), labels.end(), 200u) ||
                !std::binary_search(labels.begin(), labels.end(), 300u))
                return false;
            std::vector<std::uint32_t> selectedLabels;
            for (const std::uint32_t index : selected) {
                if (index >= labels.size()) return false;
                selectedLabels.push_back(labels[index]);
            }
            std::sort(expectedOpticLabels.begin(), expectedOpticLabels.end());
            return selectedLabels == expectedOpticLabels;
        };
        if (!verifyFullPayload(5u) || !verifyFullPayload(10u)) {
            std::cerr << "complete skin plus five/ten-bin optic selection failed\n";
            return 28;
        }

        std::vector<std::uint32_t> labels{100u, 200u, 300u,
            0xF1772001u, 0xF1772002u, 0xF1772003u, 0xF1772004u,
            0xF1819001u, 0xF1819002u, 0xF1819003u, 0xF1819004u};
        std::sort(labels.begin(), labels.end());
        std::vector<std::uint32_t> selected;
        std::uint32_t selectedBins = 0u;
        std::string selectError;
        if (regionalGraphSchwarzSelectOpticRegionIndices(
                labels, selected, selectedBins, selectError) ||
            !selected.empty() || selectedBins != 0u || selectError.empty()) {
            std::cerr << "incomplete optic family was admitted from full payload\n";
            return 29;
        }

        labels.push_back(0xF1772001u);
        std::sort(labels.begin(), labels.end());
        if (regionalGraphSchwarzSelectOpticRegionIndices(
                labels, selected, selectedBins, selectError) ||
            !selected.empty() || selectedBins != 0u || selectError.empty()) {
            std::cerr << "duplicate optic region label was admitted\n";
            return 30;
        }

        labels = {100u, 200u, 300u, 0xF177200Bu,
            0xF1819001u, 0xF1819002u, 0xF1819003u,
            0xF1819004u, 0xF1819005u};
        std::sort(labels.begin(), labels.end());
        if (regionalGraphSchwarzSelectOpticRegionIndices(
                labels, selected, selectedBins, selectError) ||
            !selected.empty() || selectedBins != 0u || selectError.empty()) {
            std::cerr << "unsupported optic bin label was admitted\n";
            return 31;
        }
    }

    // Mapped anatomy patches append to the existing skin/world graph without
    // changing its prefix. A synthetic pair of disconnected typed owners
    // exercises exact owner closure, source-row coverage, and fail-closed byte
    // bounds independently of the production optic mesh.
    {
        using numi::matter::detail::appendRegionalGraphSchwarzMappedPatches;
        using numi::matter::detail::RegionalGraphSchwarzMappedPatchSummary;
        constexpr std::uint32_t mappedNodes = 12u;
        const std::vector<std::array<std::uint32_t, 4>> fullTets{
            {0u,1u,2u,3u}, {4u,5u,6u,7u}, {8u,9u,10u,11u}};
        const std::vector<std::uint8_t> mappedFree(mappedNodes, 1u);
        const auto baseLayout = buildRegionalGraphSchwarzLayout(
            mappedNodes, fullTets, mappedFree, 1u,
            96ull * 1024ull * 1024ull, 32u, 32u, 128u, 4u, 2u, true, true);
        const std::vector<std::array<std::uint32_t, 4>> opticTets{
            {0u,1u,2u,3u}, {4u,5u,6u,7u}};
        const std::vector<std::uint8_t> opticFree(8u, 1u);
        const auto localOptic = buildRegionalGraphSchwarzLayout(
            8u, opticTets, opticFree, 1u,
            96ull * 1024ull * 1024ull, 16u, 16u, 64u, 4u, 2u, true, true);
        if (!baseLayout.error.empty() || !localOptic.error.empty()) {
            std::cerr << "mapped optic graph fixture layouts did not build\n";
            return 21;
        }
        auto augmented = baseLayout;
        const std::vector<std::uint32_t> localToGlobal{
            4u,5u,6u,7u,8u,9u,10u,11u};
        const std::vector<std::uint32_t> nodeOwners{
            0u,0u,0u,0u,1u,1u,1u,1u,2u,2u,2u,2u};
        const std::array<std::uint32_t, 2> allowedOwners{1u,2u};
        RegionalGraphSchwarzMappedPatchSummary mappedSummary{};
        std::string mappedError;
        if (!appendRegionalGraphSchwarzMappedPatches(
                augmented, localOptic, localToGlobal, nodeOwners,
                allowedOwners, 128u, 96ull * 1024ull * 1024ull,
                mappedSummary, mappedError)) {
            std::cerr << "mapped optic graph append failed: " << mappedError << '\n';
            return 22;
        }
        const auto samePatch = [](const auto& a, const auto& b) {
            return a.nodeOffset == b.nodeOffset && a.nodeCount == b.nodeCount &&
                a.matrixFloatOffset == b.matrixFloatOffset &&
                a.pivotOffset == b.pivotOffset;
        };
        const auto patchPrefix = [&](const auto& before, const auto& after) {
            return after.size() >= before.size() &&
                std::equal(before.begin(), before.end(), after.begin(), samePatch);
        };
        const auto prefix = [](const auto& before, const auto& after) {
            return after.size() >= before.size() &&
                std::equal(before.begin(), before.end(), after.begin());
        };
        const auto samePatchVector = [&](const auto& a, const auto& b) {
            return a.size() == b.size() &&
                std::equal(a.begin(), a.end(), b.begin(), samePatch);
        };
        bool homogeneous = true;
        for (std::size_t patchIndex = baseLayout.patches.size();
             patchIndex < augmented.patches.size(); ++patchIndex) {
            const auto& patch = augmented.patches[patchIndex];
            const auto begin = patch.nodeOffset;
            const auto end = begin + patch.nodeCount;
            if (patch.nodeCount == 0u || end > augmented.patchNodes.size()) {
                homogeneous = false;
                break;
            }
            const auto firstNode = augmented.patchNodes[begin];
            if (firstNode >= nodeOwners.size()) {
                homogeneous = false;
                break;
            }
            const auto owner = nodeOwners[firstNode];
            for (std::uint32_t item = begin; item < end; ++item) {
                const auto node = augmented.patchNodes[item];
                homogeneous &= node < nodeOwners.size() && nodeOwners[node] == owner;
            }
        }
        bool coverageCorrect = true;
        for (std::uint32_t node = 0u; node < mappedNodes; ++node) {
            std::uint32_t expected = baseLayout.nodeCoverage[node];
            if (node >= 4u && node < 12u) {
                const std::uint32_t local = node - 4u;
                expected += localOptic.nodeCoverage[local];
            }
            coverageCorrect &= augmented.nodeCoverage[node] == expected &&
                augmented.nodePatchOffsets[node + 1u] -
                    augmented.nodePatchOffsets[node] == expected;
        }
        if (!patchPrefix(baseLayout.patches, augmented.patches) ||
            !prefix(baseLayout.patchNodes, augmented.patchNodes) ||
            !prefix(baseLayout.rows, augmented.rows) ||
            mappedSummary.patchCount != localOptic.patches.size() ||
            mappedSummary.patchNodeSlots != localOptic.patchNodes.size() ||
            mappedSummary.maximumPatchNodes > 4u || !homogeneous ||
            !coverageCorrect || augmented.deviceBytes > 96ull * 1024ull * 1024ull) {
            std::cerr << "mapped optic append changed skin prefix or violated coverage/owner bounds\n";
            return 23;
        }
        auto deterministic = baseLayout;
        RegionalGraphSchwarzMappedPatchSummary repeatSummary{};
        std::string repeatError;
        if (!appendRegionalGraphSchwarzMappedPatches(
                deterministic, localOptic, localToGlobal, nodeOwners,
                allowedOwners, 128u, 96ull * 1024ull * 1024ull,
                repeatSummary, repeatError) ||
            !samePatchVector(deterministic.patches, augmented.patches) ||
            deterministic.patchNodes != augmented.patchNodes ||
            deterministic.rows != augmented.rows ||
            deterministic.nodePatchOffsets != augmented.nodePatchOffsets ||
            deterministic.nodePatchValues != augmented.nodePatchValues ||
            deterministic.nodeCoverage != augmented.nodeCoverage) {
            std::cerr << "mapped optic append is not deterministic\n";
            return 24;
        }
        auto boundedFailure = baseLayout;
        RegionalGraphSchwarzMappedPatchSummary failedSummary{};
        std::string failedError;
        if (appendRegionalGraphSchwarzMappedPatches(
                boundedFailure, localOptic, localToGlobal, nodeOwners,
                allowedOwners, 128u, 1u, failedSummary, failedError) ||
            !samePatchVector(boundedFailure.patches, baseLayout.patches) ||
            boundedFailure.patchNodes != baseLayout.patchNodes ||
            boundedFailure.rows != baseLayout.rows ||
            boundedFailure.nodeCoverage != baseLayout.nodeCoverage) {
            std::cerr << "mapped optic arena rejection was not atomic and fail-closed\n";
            return 25;
        }
        auto duplicateMap = localToGlobal;
        duplicateMap[1u] = duplicateMap[0u];
        auto duplicateFailure = baseLayout;
        if (appendRegionalGraphSchwarzMappedPatches(
                duplicateFailure, localOptic, duplicateMap, nodeOwners,
                allowedOwners, 128u, 96ull * 1024ull * 1024ull,
                failedSummary, failedError) ||
            !samePatchVector(duplicateFailure.patches, baseLayout.patches)) {
            std::cerr << "mapped optic duplicate global node was admitted\n";
            return 26;
        }
        auto crossingPatch = localOptic;
        crossingPatch.patchNodes[crossingPatch.patches.front().nodeOffset] = 4u;
        auto crossingFailure = baseLayout;
        if (appendRegionalGraphSchwarzMappedPatches(
                crossingFailure, crossingPatch, localToGlobal, nodeOwners,
                allowedOwners, 128u, 96ull * 1024ull * 1024ull,
                failedSummary, failedError) ||
            !samePatchVector(crossingFailure.patches, baseLayout.patches)) {
            std::cerr << "mapped optic cross-owner patch was admitted\n";
            return 27;
        }
    }

    std::cout << "graph_schwarz_layout_checks=38"
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
