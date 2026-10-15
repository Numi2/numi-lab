#include "cardboard_shipping_snapshot.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <bit>
using namespace numi::matter;
using namespace numi::cardboard::shipping_snapshot_detail;
static constexpr std::uint64_t kMax=16ull*1024*1024*1024;
static void readAll(std::istream& in, RuntimeStateSnapshot& s){
#define R(f) s.f=readVector<decltype(s.f)::value_type>(in,kMax)
R(sutureProxyEdges);R(particles);R(femNodes);R(femFields);R(vascularState);R(vascularClock);R(femTopologyNodes);R(femTopologyTetrahedra);R(cohesiveFaces);R(punctureChannels);R(topologyStates);R(statuses);R(solverCertificates);R(mpmActiveNodeIndices);R(mpmNodeToActive);R(mpmActiveNodeCounts);R(rigidGeneralizedCandidate);R(learnedWeights);R(adaptive);R(schedulers);R(reactions);R(rigidStates);R(contactSamples);R(deformableContactFailures);R(contactHistories);R(humanSupportHistories);R(humanSupportConsequences);R(deformableContactHistories);R(particleMaterialState);R(femMaterialState);R(identification);R(environmentParameters);
#undef R
}
static RuntimeStateSnapshot load(const char*p){std::ifstream in(p,std::ios::binary);RuntimeStateSnapshot s;s.message=readString(in,kMax);readScalarFields(in,s);readAll(in,s);return s;}
#define CMP(a) do{bool d=std::memcmp(&x.a,&y.a,sizeof(x.a))!=0;std::cout<<#a<<" "<<(d?"DIFF":"same")<<" bytes="<<sizeof(x.a)<<"\n";if constexpr(sizeof(x.a)==16){auto xx=reinterpret_cast<const uint32_t*>(&x.a);auto yy=reinterpret_cast<const uint32_t*>(&y.a);std::cout<<"  ";for(int j=0;j<4;j++)std::cout<<std::hex<<xx[j]<<"/"<<yy[j]<<" ";std::cout<<std::dec<<"\n";}}while(0)
int main(int c,char**v){auto a=load(v[1]),b=load(v[2]);auto&x=a.cohesiveFaces.at(0);auto&y=b.cohesiveFaces.at(0);CMP(nodesAndFirst);CMP(adjacency);CMP(state);CMP(geometry);CMP(separatedNodes);CMP(lawIdentity);CMP(activationControl);CMP(activationLocalGap);CMP(activationSideAPosition);CMP(activationTangent1);CMP(mortarSideAVertexWeights);CMP(mortarSideBVertexWeights);CMP(mixedPointHistory);CMP(mixedPointWork);}
