#include "cardboard_shipping_snapshot.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){
 using namespace numi::cardboard::shipping_snapshot_detail;
 if(argc!=2)return 2;
 std::ifstream in(argv[1],std::ios::binary);
 numi::matter::RuntimeStateSnapshot s{};
 s.message=readString(in,1u<<20);readScalarFields(in,s);
 try{readSnapshotVectors(in,s,1ull<<30);}catch(const std::exception&e){std::cerr<<"reader validation: "<<e.what()<<"\n";}
 std::cout<<"statuscount "<<s.statuses.size()<<" failcount "<<s.deformableContactFailures.size()<<"\n";
 for(unsigned i=0;i<s.statuses.size();++i){auto x=s.statuses[i];std::cout<<"status "<<i<<" code "<<x.code<<" failing "<<x.failingIndex<<" diagnostics "<<x.diagnostics.x<<","<<x.diagnostics.y<<","<<x.diagnostics.z<<","<<x.diagnostics.w<<"\n";}
 for(unsigned i=0;i<s.deformableContactFailures.size();++i){auto f=s.deformableContactFailures[i];std::cout<<"failure "<<i<<" attempt "<<f.attempt.x<<","<<f.attempt.y<<","<<f.attempt.z<<","<<f.attempt.w<<" pair "<<f.primitivePair.x<<","<<f.primitivePair.y<<","<<f.primitivePair.z<<","<<f.primitivePair.w<<" diag "<<f.diagnostics.x<<","<<f.diagnostics.y<<","<<f.diagnostics.z<<","<<f.diagnostics.w<<" params "<<f.parameters.x<<","<<f.parameters.y<<","<<f.parameters.z<<","<<f.parameters.w<<"\n";}
}
