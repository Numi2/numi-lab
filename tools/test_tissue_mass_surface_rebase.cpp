#include "metalrobo/NumiHumanSurfaceFrameRebase.hpp"
#include <iostream>
struct Vec {float x=0,y=0,z=0,w=0;};
struct Body {unsigned x=0,y=0,z=0,w=0;};
struct Term {Vec positionAndWeight; Body body;};
struct Binding {unsigned bodyIndex=0; float translationX=0,translationY=0,translationZ=0;};
struct Bones {std::vector<Binding> records;};
struct Skin {std::vector<Binding> bindings;};
struct B {float translation[3];};
struct R {unsigned bodyIndex[1];unsigned bindingCount=1;B binding[1];};
struct Tissue {
 std::vector<R> records;
 std::map<unsigned,std::vector<std::vector<Term>>> parentInfluences,anatomyParentInfluences;
};
struct V {float positionX,positionY,positionZ;};
struct AR {unsigned firstVertex,vertexCount,bodyIndex;};
struct Anatomy {std::vector<AR> records;std::vector<V> vertices;};
int main(){
 double maxError=0;
 for(unsigned k=0;k<64;++k){
  const float x=.013f*k,y=-.021f*k,z=.017f*k;
  const double cx=.00023*(k+1),cy=-.00017*(k+1),cz=.00011*(k+1);
  metalrobo::NumiHumanTissueMassPartition p{};p.donorBody=3;p.remainingCOMOffsetM={cx,cy,cz};
  Bones b{{{3,x,y,z}}};Skin s{{{3,x,y,z}}};
  Tissue t{{{{3},1,{{{x,y,z}}}}},{{17,{{{{x,y,z,.73f},{3,0,0,0}}}}}},{{18,{{{{x,y,z,.27f},{3,0,0,0}}}}}}};
  Anatomy a{{{0,1,3},{0,1,3}},{{x,y,z}}};
  rebaseNumiHumanSurfaceFrames(&b,&s,&t,&a,std::span(&p,1));
  const std::array<std::array<float,3>,6> q{{
   {b.records[0].translationX,b.records[0].translationY,b.records[0].translationZ},
   {s.bindings[0].translationX,s.bindings[0].translationY,s.bindings[0].translationZ},
   {t.records[0].binding[0].translation[0],t.records[0].binding[0].translation[1],t.records[0].binding[0].translation[2]},
   {t.parentInfluences.at(17)[0][0].positionAndWeight.x,t.parentInfluences.at(17)[0][0].positionAndWeight.y,t.parentInfluences.at(17)[0][0].positionAndWeight.z},
   {t.anatomyParentInfluences.at(18)[0][0].positionAndWeight.x,t.anatomyParentInfluences.at(18)[0][0].positionAndWeight.y,t.anatomyParentInfluences.at(18)[0][0].positionAndWeight.z},
   {a.vertices[0].positionX,a.vertices[0].positionY,a.vertices[0].positionZ}
  }};
  const double sn=std::sin(.17*k),cs=std::cos(.17*k);
  const auto world=[&](double px,double py,double pz,double dx,double dy,double dz){
   return std::array<double,3>{cs*px-sn*py+dx,sn*px+cs*py+dy,pz+dz};
  };
  const auto old=world(x,y,z,.4,-.2,.7);
  for(const auto& v:q){
   const auto now=world(v[0],v[1],v[2],.4+cs*cx-sn*cy,-.2+sn*cx+cs*cy,.7+cz);
   for(unsigned j=0;j<3;++j)maxError=std::max(maxError,std::abs(old[j]-now[j]));
  }
  if(t.parentInfluences.at(17)[0][0].positionAndWeight.w!=.73f ||
     t.anatomyParentInfluences.at(18)[0][0].positionAndWeight.w!=.27f ||
     a.records.size()!=2 || b.records[0].bodyIndex!=3)return 2;
 }
 if(maxError>2e-7)return 3;
 std::cout<<"{\"status\":\"PASS\",\"rotated_frame_cases\":64,\"binding_channels\":6,\"maximum_world_position_error_m\":"<<maxError<<"}\n";
}
