#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
using Owner=numi::human::AcceptedFEMSkinTransfer;
static void check(bool ok,const char* msg){if(!ok){std::cerr<<"FAIL: "<<msg<<"\\n";std::exit(1);}}
static double norm(Owner::Point a,Owner::Point b){double s=0;for(int k=0;k<3;k++){double d=a[k]-b[k];s+=d*d;}return std::sqrt(s);}
int main(){
    Owner transfer; Owner::BuildReport report; std::string err;
    std::vector<Owner::Point> q{{0,0,0},{1,0,0},{0,1,0}};
    std::vector<std::uint32_t> detailSource{0,1,2},detailDonor{0,0,0},coarseSource{0,1,2},coarseDonor{0,0,0};
    std::vector<std::uint8_t> used{1,1,1};
    std::vector<Owner::Face> faces{{0,1,2}};
    check(transfer.build(q,detailSource,detailDonor,used,q,coarseSource,coarseDonor,faces,&report,&err),err.c_str());
    Owner::Point offset{0.003,-0.002,0.004};
    std::array<double,9> F{1.2,0.1,-0.03, 0.02,0.9,0.08, 0.0,-0.04,1.1};
    Owner::Point t{0.2,-0.3,0.1};
    std::vector<Owner::Point> detail(3);
    std::vector<Owner::Point> offsets(3,offset);
    std::vector<std::array<double,9>> gradients(3,F);
    std::vector<Owner::Point> acceptedCoarse(3);
    std::vector<std::array<float,4>> delta(3);
    for(int i=0;i<3;i++){
        for(int a=0;a<3;a++){
            detail[i][a]=q[i][a]+offset[a];
            acceptedCoarse[i][a]=t[a];for(int b=0;b<3;b++)acceptedCoarse[i][a]+=F[3*a+b]*q[i][b];
            delta[i][a]=float(acceptedCoarse[i][a]-q[i][a]);
        }
    }
    std::vector<Owner::Point> oldPath,newPath;
    check(transfer.applyReference(detail,delta,oldPath,&err),err.c_str());
    check(transfer.applyReferenceWithP1OffsetTransport(detail,delta,offsets,gradients,newPath,&err),err.c_str());
    for(int i=0;i<3;i++){
        Owner::Point expected{};
        for(int a=0;a<3;a++){expected[a]=t[a];for(int b=0;b<3;b++)expected[a]+=F[3*a+b]*detail[i][b];}
        check(norm(newPath[i],expected)<2e-8,"affine P1 map with offset is reproduced");
        check(norm(oldPath[i],expected)>1e-5,"legacy additive path omits offset transport");
    }
    std::vector<std::array<double,9>> identity(3);for(auto& m:identity)m={1,0,0,0,1,0,0,0,1};
    check(transfer.applyReferenceWithP1OffsetTransport(detail,delta,offsets,identity,newPath,&err),err.c_str());
    for(int i=0;i<3;i++)check(norm(newPath[i],oldPath[i])<1e-12,"identity gradient preserves legacy transfer");
    auto badOffsets=offsets;badOffsets.pop_back();
    check(!transfer.applyReferenceWithP1OffsetTransport(detail,delta,badOffsets,gradients,newPath,&err),"mismatched offset rows fail closed");
    auto badGradients=gradients;badGradients[0][0]=NAN;
    check(!transfer.applyReferenceWithP1OffsetTransport(detail,delta,offsets,badGradients,newPath,&err),"nonfinite gradient fails closed");
    std::cout<<"PASS affine P1 offset transport; identity parity; invalid-input rejection\\n";
}
