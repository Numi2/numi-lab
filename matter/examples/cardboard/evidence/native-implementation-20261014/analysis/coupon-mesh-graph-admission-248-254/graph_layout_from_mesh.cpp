#include "regional_graph_schwarz.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1]);
    std::uint32_t n=0,t=0,e=0;
    in >> n >> t >> e;
    std::vector<std::uint8_t> mask(n);
    for(auto& x:mask) { unsigned value=0; in >> value; if(value>1) return 3; x=static_cast<std::uint8_t>(value); }
    std::vector<std::array<std::uint32_t,4>> tets(t);
    for(auto& tet:tets) for(auto& x:tet) in>>x;
    if(!in) return 4;
    const auto l=numi::matter::detail::buildRegionalGraphSchwarzLayout(n,tets,mask,e);
    std::cout << "{\"nodes\":" << n << ",\"tetrahedra\":" << t << ",\"environments\":" << e
              << ",\"patches\":" << l.patches.size() << ",\"matrix_floats\":" << l.matrixFloatCount
              << ",\"device_bytes\":" << l.deviceBytes << ",\"error\":\"" << l.error << "\"}\n";
    return 0;
}
