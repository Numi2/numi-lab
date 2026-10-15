#include "cardboard_shipping_snapshot.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace numi::matter;
using namespace numi::cardboard::shipping_snapshot_detail;
static constexpr std::uint64_t kMax = 16ull*1024*1024*1024;
static void readAll(std::istream& in, RuntimeStateSnapshot& s) {
#define R(f) s.f = readVector<decltype(s.f)::value_type>(in, kMax)
 R(sutureProxyEdges); R(particles); R(femNodes); R(femFields); R(vascularState); R(vascularClock);
 R(femTopologyNodes); R(femTopologyTetrahedra); R(cohesiveFaces); R(punctureChannels); R(topologyStates);
 R(statuses); R(solverCertificates); R(mpmActiveNodeIndices); R(mpmNodeToActive); R(mpmActiveNodeCounts);
 R(rigidGeneralizedCandidate); R(learnedWeights); R(adaptive); R(schedulers); R(reactions); R(rigidStates);
 R(contactSamples); R(deformableContactFailures); R(contactHistories); R(humanSupportHistories); R(humanSupportConsequences);
 R(deformableContactHistories); R(particleMaterialState); R(femMaterialState); R(identification); R(environmentParameters);
#undef R
}
static RuntimeStateSnapshot load(const char* path) {
 std::ifstream in(path, std::ios::binary); if(!in) throw std::runtime_error("open");
 RuntimeStateSnapshot s; s.message=readString(in,kMax); readScalarFields(in,s); readAll(in,s); return s;
}
static void print(const char* label,const RuntimeStateSnapshot& s) {
 std::cout<<label<<" control="<<s.controlStep<<" status="<<s.statuses.at(0).code<<"\n";
 const auto& f=s.cohesiveFaces.at(0);
 auto p4=[](const nm_float4& v){std::cout<<" ["<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"]";};
 auto u4=[](const nm_uint4& v){std::cout<<" ["<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"]";};
 std::cout<<"nodesAndFirst";u4(f.nodesAndFirst);std::cout<<" adjacency";u4(f.adjacency);std::cout<<" state";p4(f.state);std::cout<<" geometry";p4(f.geometry);std::cout<<" separated";u4(f.separatedNodes);std::cout<<" law";u4(f.lawIdentity);std::cout<<" activationControl";p4(f.activationControl);std::cout<<"\n";
 for(int i=0;i<3;i++){std::cout<<" local"<<i;p4(f.activationLocalGap[i]);std::cout<<" sideA";p4(f.activationSideAPosition[i]);std::cout<<" mA";p4(f.mortarSideAVertexWeights[i]);std::cout<<" mB";p4(f.mortarSideBVertexWeights[i]);std::cout<<" hist";p4(f.mixedPointHistory[i]);std::cout<<"\n";}
 std::cout<<" tangent";p4(f.activationTangent1);std::cout<<" work";p4(f.mixedPointWork);std::cout<<"\n";
 std::cout<<"topology count"<<s.topologyStates.size(); if(!s.topologyStates.empty()){auto t=s.topologyStates.at(0);std::cout<<" counts";u4(t.counts);std::cout<<" roles";u4(t.roles);std::cout<<" accounting";p4(t.accounting);}std::cout<<"\n";
}
int main(int argc,char**argv){auto a=load(argv[1]);auto b=load(argv[2]);print("before",a);print("after",b);}
