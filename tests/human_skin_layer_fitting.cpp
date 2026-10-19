#include "NumiHumanCoarseTissuePreparation.hpp"
#include <iostream>
#include <numbers>
using namespace metalrobo::human::coarse_tissue;
int main(){
 RegisteredSurface s;s.name="torus thickness constraint fixture";s.provider="analytic fixture";s.sourceMember="torus";s.sourcePayloadSHA256=std::string(64,'1');s.sourceMeshSHA256=std::string(64,'2');s.stableIdentifier=1;s.sourceBodyIndex=20;
 constexpr unsigned nu=32,nv=16;
 for(unsigned u=0;u<nu;u++)for(unsigned v=0;v<nv;v++){
  const double a=2*std::numbers::pi*u/nu,b=2*std::numbers::pi*v/nv;
  s.worldPoints.push_back({(.04+.008*std::cos(b))*std::cos(a),(.04+.008*std::cos(b))*std::sin(a),.008*std::sin(b)});
 }
 s.donorLocalPoints=s.worldPoints;
 for(unsigned u=0;u<nu;u++)for(unsigned v=0;v<nv;v++){
  unsigned a=u*nv+v,b=((u+1)%nu)*nv+v,c=((u+1)%nu)*nv+(v+1)%nv,e=u*nv+(v+1)%nv;
  s.faces.push_back({a,b,c});s.faces.push_back({a,c,e});
 }
 const auto original=s.worldPoints;SkinFatReference ref;std::vector<double> st(original.size(),.002),ft(original.size(),.008);
 SkinFatLayerFitReport r;PairedBoundaryLayer skin,fat;std::string error;
 bool ok=fitRegisteredSkinFatLayers(s,ref,st,ft,"inferred test skin","inferred test fat",skin,fat,r,error);
 if(!ok){std::cerr<<error<<"\n";return 1;}
 if(s.worldPoints!=original||r.fatAdjustedVertices==0||fat.volumeM3<=0||r.minFatM<ref.minFatThicknessM)return 2;
 std::cout<<"fitting=PASS iterations="<<r.iterations<<" adjusted_fat="<<r.fatAdjustedVertices<<" min_fat_m="<<r.minFatM<<"\n";
 auto bad=s;std::vector<double> badSkin(st.size(),.004);const auto saved=bad.worldPoints;
 if(fitRegisteredSkinFatLayers(bad,ref,badSkin,ft,"test","test",skin,fat,r,error)||bad.worldPoints!=saved)return 3;
 std::cout<<"invalid_field_rejected=PASS exterior_unchanged=PASS\n";
}