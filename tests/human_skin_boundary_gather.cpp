#include "NumiHumanSkinBoundaryNodeMap.hpp"
#include <cassert>
#include <array>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
int main(){
 using namespace numi::human;
 SkinBoundaryNodeMap map{12,{2,4,6,8},{1,3,5,7,9},{0,10,11}};
 std::vector<std::array<float,4>> snapshot(14);
 for(unsigned i=0;i<snapshot.size();++i)snapshot[i]={float(i),float(i+1),float(i+2),100.f};
 std::vector<std::array<float,4>> out(8,{99,99,99,99});
 std::string error;
 auto read=[&](std::size_t i){return snapshot.at(i);};
 assert(gatherAcceptedSkinInnerFatPositions(map,2,snapshot.size(),read,out,error));
 unsigned j=0;
 for(const auto* rows:{&map.skinInner,&map.fatDeep})
  for(auto row:*rows){assert(out[j][0]==float(row+2));assert(out[j][3]==1.f);++j;}
 auto accepted=out;
 snapshot[2+map.fatDeep.back()][0]=NAN;
 assert(!gatherAcceptedSkinInnerFatPositions(map,2,snapshot.size(),read,out,error));
 assert(out==accepted);
 snapshot[2+map.fatDeep.back()][0]=13.f;
 assert(!gatherAcceptedSkinInnerFatPositions(map,3,snapshot.size(),read,out,error));
 assert(out==accepted);
 out.pop_back();auto shortOutput=out;
 assert(!gatherAcceptedSkinInnerFatPositions(map,2,snapshot.size(),read,out,error));
 assert(out==shortOutput);
 out=accepted;auto alias=map;alias.skinInner[0]=map.outer[0];
 assert(!gatherAcceptedSkinInnerFatPositions(alias,2,snapshot.size(),read,out,error));
 assert(out==accepted);
 SkinBoundaryNodeMap legacy;
 assert(makePairedSkinBoundaryNodeMap(4,legacy,error));
 std::vector<std::array<float,4>> pairedOut(8);
 assert(gatherAcceptedSkinInnerFatPositions(legacy,0,snapshot.size(),read,pairedOut,error));
 puts("PASS independent accepted boundary gather counts=4,5,3 output=8 finite/range/alias/count failures atomic; paired compatibility");
}

