#include "NumiHumanSkinRigidPlacement.hpp"
#include <cstdint>
#include <iostream>
#include <limits>
namespace metalrobo::human::coarse_tissue {using Face=std::array<std::uint32_t,3>;}
#include "NumiHumanWholeBodySkinReduction.hpp"
using namespace metalrobo::human::coarse_tissue;
using P=RegisteredSkinRigidPlacement::Point;
using M=RegisteredSkinRigidPlacement::Matrix;
template<class T,std::size_t N> std::array<double,N> d(const std::array<T,N>& a){
 std::array<double,N> o{};std::copy(a.begin(),a.end(),o.begin());return o;
}
P sub(P a,P b){for(unsigned i=0;i<3;++i)a[i]-=b[i];return a;}
P product(M m,P p,bool transpose=false){
 P out{};for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)
  out[r]+=m[transpose?3*c+r:3*r+c]*p[c];return out;
}
double vol(P a,P b,P c,P e){b=sub(b,a);c=sub(c,a);e=sub(e,a);return
 (b[0]*(c[1]*e[2]-c[2]*e[1])+b[1]*(c[2]*e[0]-c[0]*e[2])+b[2]*(c[0]*e[1]-c[1]*e[0]))/6.;}
int main(){
 // Same supported-root rotation as the retained bed fixture, independent of donor orientation.
 const double angle=-1.5533430342749532,c=std::cos(angle),s=std::sin(angle);
 RegisteredSkinRigidPlacement truth;truth.rotation={1,0,0,0,c,-s,0,s,c};truth.translation={.13,-.24,.31};
 const auto& f=kWholeBodySkinDonorFrames[0];std::string err;RegisteredSkinRigidPlacement got;
 if(!deriveRegisteredSkinRigidPlacement(d(f.registeredWorldPoint),d(f.worldToDonorLocalLinear),
   truth.point(d(f.registeredWorldPoint)),truth.donorFrame(d(f.worldToDonorLocalLinear)),got,err))return 1;
 for(const auto& r:kWholeBodySkinDonorFrames){
  // Recover the donor origin from the reference and independently compute
  // current local coordinates from the placed donor pose.
  const auto world=d(r.registeredWorldPoint),local=d(r.donorLocalPoint);
  const auto frame=d(r.worldToDonorLocalLinear);
  const auto donorOrigin=sub(world,product(frame,local,true));
  const auto currentWorld=truth.point(world);
  const auto currentFrame=truth.donorFrame(frame);
  const auto currentLocal=product(currentFrame,sub(currentWorld,truth.point(donorOrigin)));
  if(!validateRegisteredSkinPlacementVertex(d(r.registeredWorldPoint),d(r.donorLocalPoint),
    d(r.registeredWorldNormal),d(r.worldToDonorLocalLinear),truth.point(d(r.registeredWorldPoint)),
    currentLocal,truth.vector(d(r.registeredWorldNormal)),currentFrame,got,err))return 2;
 }
 auto badLocal=d(f.donorLocalPoint);badLocal[0]+=.001;
 if(validateRegisteredSkinPlacementVertex(d(f.registeredWorldPoint),d(f.donorLocalPoint),
  d(f.registeredWorldNormal),d(f.worldToDonorLocalLinear),truth.point(d(f.registeredWorldPoint)),
  badLocal,truth.vector(d(f.registeredWorldNormal)),truth.donorFrame(d(f.worldToDonorLocalLinear)),got,err))return 3;
 M reflection{-1,0,0,0,1,0,0,0,1},identity{1,0,0,0,1,0,0,0,1};
 if(deriveRegisteredSkinRigidPlacement({},identity,{},reflection,got,err))return 4;
 M shear=identity;shear[1]=.01;
 if(deriveRegisteredSkinRigidPlacement({},identity,{},shear,got,err))return 5;
 P invalid{0,0,std::numeric_limits<double>::infinity()};
 if(deriveRegisteredSkinRigidPlacement({},identity,invalid,identity,got,err))return 6;
 double total[2]{},moved[2]{},maxDifference=0;std::size_t tets=0;
 for(auto face:kWholeBodySkinFaces){
  std::sort(face.begin(),face.end());
  for(unsigned layer=0;layer<2;++layer){
   P outer[3],inner[3];
   for(unsigned j=0;j<3;++j){auto i=face[j];
    outer[j]=layer?kWholeBodySkinInnerWorld[i]:d(kWholeBodySkinDonorFrames[i].registeredWorldPoint);
    if(!layer)for(unsigned a=0;a<3;++a)outer[j][a]+=kWholeBodySkinMechanicalWorldDeltas[i][a];
    inner[j]=layer?kWholeBodyFatDeepWorld[i]:kWholeBodySkinInnerWorld[i];
   }
   const std::array<std::array<P,4>,3> cells{{
    {outer[0],outer[1],outer[2],inner[2]},
    {outer[0],outer[1],inner[1],inner[2]},
    {outer[0],inner[0],inner[1],inner[2]}}};
   for(auto cell:cells){
    const double v=vol(cell[0],cell[1],cell[2],cell[3]);
    for(auto& p:cell)p=truth.point(p);
    const double w=vol(cell[0],cell[1],cell[2],cell[3]);
    maxDifference=std::max(maxDifference,std::abs(v-w));
    if(v*w<=0||std::abs(v-w)>1e-17)return 7;
    total[layer]+=std::abs(v);moved[layer]+=std::abs(w);++tets;
   }
  }
 }
 if(std::abs(total[0]-kWholeBodySkinLayerVolumeM3)>total[0]*1e-6||
    std::abs(total[1]-kWholeBodyFatLayerVolumeM3)>total[1]*1e-6)return 8;
 std::cout<<"PASS vertices="<<kWholeBodySkinDonorFrames.size()<<" tetrahedra="<<tets
  <<" volume_difference_m3="<<maxDifference<<" skin_mass_kg="<<moved[0]*1000
  <<" fat_mass_kg="<<moved[1]*900<<" nonrigid_reflection_shear_nonfinite_rejected=1\n";
}
