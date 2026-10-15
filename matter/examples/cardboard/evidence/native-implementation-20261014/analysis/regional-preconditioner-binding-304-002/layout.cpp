#include "regional_graph_schwarz.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char**argv){
 if(argc!=6)return 2; unsigned n=std::stoul(argv[1]),t=std::stoul(argv[2]),e=std::stoul(argv[3]);
 std::vector<std::array<std::uint32_t,4>> ts(t); std::vector<std::uint8_t> m(n);
 {std::ifstream f(argv[4],std::ios::binary);f.read((char*)ts.data(),ts.size()*sizeof(ts[0]));if(!f)return 3;}
 {std::ifstream f(argv[5],std::ios::binary);f.read((char*)m.data(),m.size());if(!f)return 4;}
 auto x=numi::matter::detail::buildBudgetedRegionalGraphSchwarzLayout(n,ts,m,e);
 if(!x.error.empty()){std::cerr<<x.error<<"\n";return 5;}
 std::cout<<"patches "<<x.patches.size()<<" patchNodes "<<x.patchNodes.size()<<" totalDofs "<<x.totalPatchDofs<<" matrixFloatCount "<<x.matrixFloatCount<<" pivotCount "<<x.pivotCount<<" metadataBytes "<<x.metadataBytes<<" deviceBytes "<<x.deviceBytes<<"\n";
 for(const auto&p:x.patches)std::cout<<p.nodeOffset<<" "<<p.nodeCount<<" "<<p.matrixFloatOffset<<" "<<p.pivotOffset<<"\n";
 std::cout<<"nodes\n";for(auto q:x.patchNodes)std::cout<<q<<" ";std::cout<<"\ncoverage\n";for(auto q:x.nodeCoverage)std::cout<<q<<" ";std::cout<<"\n";return 0;
}
