#include "regional_graph_schwarz.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) return 3;
    std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(in)), {});
    if (raw.size() < 12) return 4;
    auto u32 = [&](std::size_t off) -> std::uint32_t {
        if (off + 4 > raw.size()) throw 1;
        return std::uint32_t(raw[off]) | (std::uint32_t(raw[off+1]) << 8) |
            (std::uint32_t(raw[off+2]) << 16) | (std::uint32_t(raw[off+3]) << 24);
    };
    const std::uint32_t env = u32(0), nodes = u32(4), tetsN = u32(8);
    const std::uint64_t need = 12ull + std::uint64_t(tetsN)*16ull + nodes;
    if (env != 2 || nodes != 882 || tetsN != 1920 || need != raw.size()) return 5;
    std::vector<std::array<std::uint32_t,4>> tets(tetsN);
    std::size_t cursor = 12;
    for (auto& tet : tets) for (auto& n : tet) { n = u32(cursor); cursor += 4; }
    std::vector<std::uint8_t> freeMask(raw.begin()+cursor, raw.end());
    auto layout = numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(
        nodes, tets, freeMask, env);
    if (!layout.error.empty()) { std::cerr << layout.error << '\n'; return 6; }
    std::uint32_t checksum = 2166136261u;
    auto mix = [&](std::uint32_t w) { checksum = (checksum ^ w) * 16777619u; };
    for (const auto& p : layout.patches) {
        mix(p.nodeOffset); mix(p.nodeCount); mix(p.matrixFloatOffset); mix(p.pivotOffset);
    }
    for (const auto& r : layout.rows) for (const auto w : r) mix(w);
    for (const auto w : layout.patchNodes) mix(w);
    for (const auto w : layout.nodePatchOffsets) mix(w);
    for (const auto w : layout.nodePatchValues) mix(w);
    std::uint64_t freeCount = 0;
    for (auto b : freeMask) freeCount += b;
    std::cout << "{\"schema\":\"numi.cardboard.graph-layout-preflight.v1\","
              << "\"environment_count\":" << env << ",\"node_count\":" << nodes
              << ",\"tetrahedron_count\":" << tetsN << ",\"free_nodes\":" << freeCount
              << ",\"fixed_nodes\":" << (nodes-freeCount)
              << ",\"patch_count\":" << layout.patches.size()
              << ",\"patch_node_count\":" << layout.patchNodes.size()
              << ",\"dof_count\":" << layout.totalPatchDofs
              << ",\"matrix_float_count\":" << layout.matrixFloatCount
              << ",\"pivot_count\":" << layout.pivotCount
              << ",\"metadata_bytes\":" << layout.metadataBytes
              << ",\"device_bytes\":" << layout.deviceBytes
              << ",\"layout_checksum_fnv1a32\":" << checksum << "}\n";
}
