#include "/Users/home/cardboard-implementation-20261014/snapshots/source-runtime-diagnostic-return-repair-079/matter/src/regional_graph_schwarz.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: layout_from_source input.bin output.json\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    std::uint32_t nodeCount = 0u, tetCount = 0u, environments = 0u;
    input.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    input.read(reinterpret_cast<char*>(&tetCount), sizeof(tetCount));
    input.read(reinterpret_cast<char*>(&environments), sizeof(environments));
    if (!input || nodeCount == 0u || tetCount == 0u || environments == 0u ||
        nodeCount > 4096u || tetCount > 32768u || environments > 4u) {
        std::cerr << "invalid bounded graph-layout input header\n";
        return 2;
    }
    std::vector<std::array<std::uint32_t, 4>> tetrahedra(tetCount);
    input.read(reinterpret_cast<char*>(tetrahedra.data()),
               static_cast<std::streamsize>(tetrahedra.size() * sizeof(tetrahedra[0])));
    std::vector<std::uint8_t> freeMask(nodeCount);
    input.read(reinterpret_cast<char*>(freeMask.data()),
               static_cast<std::streamsize>(freeMask.size()));
    if (!input) {
        std::cerr << "truncated bounded graph-layout input\n";
        return 2;
    }
    char extra = 0;
    if (input.read(&extra, 1)) {
        std::cerr << "unexpected trailing graph-layout input bytes\n";
        return 2;
    }
    const auto layout = numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
        nodeCount, tetrahedra, freeMask, environments);
    if (!layout.error.empty()) {
        std::cerr << layout.error << "\n";
        return 3;
    }
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) return 2;
    output << "{\"node_count\":" << nodeCount
           << ",\"tetrahedron_count\":" << tetCount
           << ",\"environment_count\":" << environments
           << ",\"patch_count\":" << layout.patches.size()
           << ",\"patch_node_count\":" << layout.patchNodes.size()
           << ",\"total_patch_dofs\":" << layout.totalPatchDofs
           << ",\"matrix_float_count\":" << layout.matrixFloatCount
           << ",\"device_bytes\":" << layout.deviceBytes
           << ",\"patches\":[";
    for (std::size_t p = 0; p < layout.patches.size(); ++p) {
        if (p) output << ',';
        const auto patch = layout.patches[p];
        output << '[';
        for (std::uint32_t i = 0u; i < patch.nodeCount; ++i) {
            if (i) output << ',';
            output << layout.patchNodes[patch.nodeOffset + i];
        }
        output << ']';
    }
    output << "],\"node_coverage\":[";
    for (std::size_t i = 0; i < layout.nodeCoverage.size(); ++i) {
        if (i) output << ',';
        output << layout.nodeCoverage[i];
    }
    output << "]}\n";
    return output ? 0 : 2;
}
