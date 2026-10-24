#include "numi/matter/fem_body_triangle_contact_sample.hpp"
#include "NumiHumanD37InitialSkinBoneWitness.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
namespace d37w=numi::human::d37_initial_skin_bone_witness;
namespace w=numi::human::initial_skin_bone_contact_witness;
namespace bvh=numi::matter::fem_body_contact_bvh;
namespace {
void require(bool ok,std::string_view what) {
 if(!ok){std::fprintf(stderr,"FAIL: %.*s\n",int(what.size()),what.data());std::exit(1);}
}
using Tri=NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU;
Tri makeTri(std::uint32_t stable,const std::array<std::array<float,3>,8>& p,
            std::array<unsigned,3> f) {
 Tri t{};
 t.bodyLocalVertex0={p[f[0]][0],p[f[0]][1],p[f[0]][2],0.f};
 t.bodyLocalVertex1={p[f[1]][0],p[f[1]][1],p[f[1]][2],0.f};
 t.bodyLocalVertex2={p[f[2]][0],p[f[2]][1],p[f[2]][2],0.f};
 const auto a=w::read3(t.bodyLocalVertex0),b=w::read3(t.bodyLocalVertex1),c=w::read3(t.bodyLocalVertex2);
 const auto raw=w::cross(w::sub(b,a),w::sub(c,a));const double n=w::norm(raw);
 t.authoredOutwardNormal={float(raw.x/n),float(raw.y/n),float(raw.z/n),0.f};
 t.bodyIndex=7u;t.boneStableId=77u;t.triangleStableId=stable;return t;
}
void compare(w::Vec3 query,const std::array<Tri,12>& input,std::uint32_t expectedStable=UINT32_MAX) {
 const auto built=bvh::buildStaticCompleteBodyBVH(input,input.size(),true);
 require(built.succeeded(),"closed oriented synthetic bank builds");
 require(built.bodies.size()==1u&&built.bodies[0].bodyIndex==7u,"one body range");
 const auto& body=built.bodies[0];
 const auto found=d37w::nearestPointOnBodyTree(query,body.bodyIndex,built.triangles,
     built.nodes,body.firstNode,body.nodeCount);
 require(found.valid,"stackless query returns nearest triangle");
 double bestD2=INFINITY;std::pair<std::uint32_t,std::uint32_t> bestID{UINT32_MAX,UINT32_MAX};
 w::Closest bestPoint{};
 for(const auto& t:built.triangles) {
  const auto a=w::read3(t.bodyLocalVertex0),b=w::read3(t.bodyLocalVertex1),c=w::read3(t.bodyLocalVertex2);
  const auto cp=w::closestPointOnTriangle(query,a,b,c);const double d2=w::norm2(w::sub(query,cp.point));
  const auto id=std::pair{t.boneStableId,t.triangleStableId};
  if(d2<bestD2||(d2==bestD2&&id<bestID)){bestD2=d2;bestID=id;bestPoint=cp;}
 }
 require(found.distanceSquared==bestD2,"squared distance exactly matches brute force");
 require(found.bone==bestID.first&&found.triangle==bestID.second,"stable identity tie matches brute force");
 if(expectedStable!=UINT32_MAX)require(found.triangle==expectedStable,"expected stable triangle wins exact tie");
 require(found.closest.point.x==bestPoint.point.x&&found.closest.point.y==bestPoint.point.y&&
         found.closest.point.z==bestPoint.point.z,"closest point matches brute force");
 require(std::abs(w::norm(found.faceNormal)-1.0)<1.e-12,"face normal unit length");
}
}
int main(){
 const std::array<std::array<float,3>,8> p{{
  {{-1,-1,-1}},{{1,-1,-1}},{{1,1,-1}},{{-1,1,-1}},
  {{-1,-1,1}},{{1,-1,1}},{{1,1,1}},{{-1,1,1}}}};
 const std::array<std::array<unsigned,3>,12> f{{
  {{0,2,1}},{{0,3,2}},{{4,5,6}},{{4,6,7}},{{0,4,7}},{{0,7,3}},
  {{1,2,6}},{{1,6,5}},{{0,1,5}},{{0,5,4}},{{3,7,6}},{{3,6,2}}}};
 std::array<Tri,12> input{};for(std::uint32_t i=0;i<input.size();++i)input[i]=makeTri(100u+i,p,f[i]);
 compare({0.0,0.0,2.0},input,102u); // exact top-diagonal tie
 compare({0.2,0.3,1.8},input);
 const auto built=bvh::buildStaticCompleteBodyBVH(input,input.size(),true);
 const auto& body=built.bodies[0];
 const auto invalid=d37w::nearestPointOnBodyTree({0.2,0.3,1.8},7u,built.triangles,built.nodes,
     body.firstNode,body.nodeCount-1u);
 require(!invalid.valid,"truncated BVH range fails closed");
 std::puts("PASS: D37 stackless nearest query matches brute-force distance/point, stable tie and truncated-range failure");
}
