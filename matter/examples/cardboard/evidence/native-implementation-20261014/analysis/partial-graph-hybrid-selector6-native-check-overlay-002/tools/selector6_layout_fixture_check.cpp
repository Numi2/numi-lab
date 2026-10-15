#include "../source/matter/src/regional_graph_schwarz.hpp"

#include <array>
#include <cstdint>
#include <iostream>

using namespace numi::matter::detail;

int main() {
    constexpr std::array<std::array<std::uint32_t, 4>, 1> tetrahedra{{
        {0u, 1u, 2u, 3u}}};
    constexpr std::array<std::uint8_t, 4> authoredFree{
        1u, 1u, 0u, 0u};
    const auto layout = buildHybridRegionalGraphSchwarzLayout(
        4u, tetrahedra, authoredFree, 1u);
    if (!layout.error.empty() || layout.patches.size() != 2u ||
        layout.patchNodes != std::vector<std::uint32_t>{0u, 1u, 0u, 1u} ||
        layout.nodeCoverage != std::vector<std::uint32_t>{2u, 2u, 0u, 0u} ||
        layout.matrixFloatCount != 72u || layout.pivotCount != 12u ||
        layout.deviceBytes != 988u) {
        std::cerr << "exact selector6 fixture layout failed: " << layout.error << '\n';
        return 1;
    }
    if (!regionalHybridGraphNodeEligible(0.0f, 1.0f, 1.0f, 0u, true) ||
        !regionalHybridGraphNodeEligible(1.0f, 1.0f, 0.0f, 0u, false) ||
        regionalHybridGraphNodeEligible(0.0f, 1.0f, 1.0f, 0u, false)) {
        std::cerr << "released-uncovered / remaining-fixed admission failed\n";
        return 2;
    }
    std::uint64_t blockBytes = 0u;
    if (!regionalHybridBlockInverseBytes(4u, 1u, blockBytes) ||
        blockBytes != 192u) {
        std::cerr << "selector6 fixture block sizing failed\n";
        return 3;
    }
    std::cout << "selector6_exact_fixture_layout=pass nodes=4 tets=1 patches=2"
              << " flat_patch_nodes=4 coverage=2,2,0,0 graph_bytes=988"
              << " block_bytes=192 exact_shipping_topology=false peak_measured=false\n";
    return 0;
}
