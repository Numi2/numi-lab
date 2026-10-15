#include "regional_graph_schwarz.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cstring>
#include <vector>

template<class T> std::vector<T> read(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("input open failed");
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    if (raw.size() % sizeof(T)) throw std::runtime_error("input element alignment failed");
    std::vector<T> out(raw.size() / sizeof(T));
    if (!raw.empty()) std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}
template<class T> void write(const std::filesystem::path& p, const std::vector<T>& values) {
    std::ofstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("output open failed");
    f.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()*sizeof(T)));
    if (!f) throw std::runtime_error("output write failed");
}
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("topology.u32 mask.u8 output-dir");
        auto words = read<std::uint32_t>(argv[1]);
        auto mask = read<std::uint8_t>(argv[2]);
        if (words.size() < 2) throw std::runtime_error("topology header absent");
        const auto nodes=words[0], tets=words[1];
        if (words.size()!=2ull+4ull*tets || mask.size()!=nodes) throw std::runtime_error("topology/mask dimensions mismatch");
        std::vector<std::array<std::uint32_t,4>> conn(tets);
        for (std::uint32_t i=0;i<tets;i++) for (unsigned j=0;j<4;j++) conn[i][j]=words[2ull+4ull*i+j];
        auto result=numi::matter::detail::buildRegionalGraphSchwarzLayout(nodes,conn,mask,2u);
        if (!result.error.empty()) throw std::runtime_error(result.error);
        std::filesystem::create_directories(argv[3]); const std::filesystem::path out=argv[3];
        write(out/"patches.bin",result.patches); write(out/"patch_nodes.bin",result.patchNodes);
        write(out/"rows.bin",result.rows); write(out/"node_patch_offsets.bin",result.nodePatchOffsets);
        write(out/"node_patch_values.bin",result.nodePatchValues); write(out/"node_coverage.bin",result.nodeCoverage);
        std::uint32_t checksum=2166136261u;
        auto mix=[&](std::uint32_t w){checksum=(checksum^w)*16777619u;};
        for(auto p:result.patches){mix(p.nodeOffset);mix(p.nodeCount);mix(p.matrixFloatOffset);mix(p.pivotOffset);}
        for(auto r:result.rows){mix(r[0]);mix(r[1]);mix(r[2]);mix(r[3]);}
        for(auto w:result.patchNodes)mix(w); for(auto w:result.nodePatchOffsets)mix(w); for(auto w:result.nodePatchValues)mix(w);
        std::cout<<"{\"node_count\":"<<nodes<<",\"tet_count\":"<<tets<<",\"patch_count\":"<<result.patches.size()<<",\"patch_node_count\":"<<result.patchNodes.size()<<",\"dof_count\":"<<result.totalPatchDofs<<",\"matrix_float_count\":"<<result.matrixFloatCount<<",\"pivot_count\":"<<result.pivotCount<<",\"checksum\":"<<checksum<<",\"device_bytes\":"<<result.deviceBytes<<"}\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;} return 0;
}
