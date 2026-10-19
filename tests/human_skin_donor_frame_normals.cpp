#include "NumiHumanCoarseTissuePreparation.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main(){
 RegisteredSurface s;s.name="mixed donor planar skin";s.provider="test";s.sourceMember="reference";s.sourcePayloadSHA256=std::string(64,'1');s.sourceMeshSHA256=std::string(64,'2');s.aggregateIdentity=true;
 for(unsigned y=0;y<3;y++)for(unsigned x=0;x<3;x++){
  V3 p{.02*x,.02*y,0};s.worldPoints.push_back(p);s.outwardWorldNormals.push_back({0,0,1});s.donorBodyIndices.push_back(x%2);
  std::array<double,9> m=x%2?std::array<double,9>{1,0,0,0,0,-1,0,1,0}:std::array<double,9>{1,0,0,0,1,0,0,0,1};
  s.worldToDonorLocalLinear.push_back(m);s.donorLocalPoints.push_back({p[0]+.1*x,m[3]*p[0]+m[4]*p[1]+m[5]*p[2]+.1*y,m[6]*p[0]+m[7]*p[1]+m[8]*p[2]});
 }
 for(unsigned y=0;y<2;y++)for(unsigned x=0;x<2;x++){unsigned a=y*3+x;s.faces.push_back({a,a+1,a+4});s.faces.push_back({a,a+4,a+3});}
 SkinFatReference ref;PairedBoundaryLayer a,b;std::string error;
 if(!deriveRegisteredSkinFatLayers(s,ref,.002,a,b,error)){std::cerr<<error;return 1;}
 double maxError=0;
 for(unsigned i=0;i<9;i++){
  auto dw=sub(b.innerWorld[i],b.outerWorld[i]);auto dl=sub(b.innerLocal[i],b.outerLocal[i]);auto m=s.worldToDonorLocalLinear[i];
  V3 expected{m[0]*dw[0]+m[1]*dw[1]+m[2]*dw[2],m[3]*dw[0]+m[4]*dw[1]+m[5]*dw[2],m[6]*dw[0]+m[7]*dw[1]+m[8]*dw[2]};
  maxError=std::max(maxError,norm(sub(dl,expected)));
 }
 std::cout<<"mixed_donor_frame_max_error_m="<<maxError<<"\n";
 return maxError<1e-12?0:1;
}