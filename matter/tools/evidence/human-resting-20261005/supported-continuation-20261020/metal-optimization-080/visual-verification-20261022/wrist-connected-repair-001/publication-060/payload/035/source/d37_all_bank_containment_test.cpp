#include "numi/matter/fem_body_triangle_contact_sample.hpp"
#include "NumiHumanD37InitialSkinBoneWitness.hpp"
#include <array>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include <string_view>
namespace d37w=numi::human::d37_initial_skin_bone_witness;
namespace w=numi::human::initial_skin_bone_contact_witness;
namespace bvh=numi::matter::fem_body_contact_bvh;
namespace {
void require(bool ok,std::string_view what){if(!ok){std::fprintf(stderr,"FAIL: %.*s\n",int(what.size()),what.data());std::exit(1);}}
using Tri=NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU;
Tri tri(const std::array<std::array<float,3>,8>& p,std::array<unsigned,3> f,std::uint32_t id,
        std::uint32_t body=7u,std::uint32_t bone=77u){
 Tri t{}; t.bodyLocalVertex0={p[f[0]][0],p[f[0]][1],p[f[0]][2],0.f};
 t.bodyLocalVertex1={p[f[1]][0],p[f[1]][1],p[f[1]][2],0.f};
 t.bodyLocalVertex2={p[f[2]][0],p[f[2]][1],p[f[2]][2],0.f};
 const auto a=w::read3(t.bodyLocalVertex0),b=w::read3(t.bodyLocalVertex1),c=w::read3(t.bodyLocalVertex2);
 const auto n=w::cross(w::sub(b,a),w::sub(c,a));const double length=w::norm(n);
 t.authoredOutwardNormal={float(n.x/length),float(n.y/length),float(n.z/length),0.f};
 t.bodyIndex=body;t.boneStableId=bone;t.triangleStableId=id;return t;
}
struct FakeNode { nm_float4 positionAndMass{}; };
struct FakeQuat { float x=0,y=0,z=0,w=1; };
struct FakeBody { nm_float4 position{}; FakeQuat orientation{}; };
struct FakeTree { std::uint32_t body=0,firstNode=0,nodeCount=0,firstTriangle=0,triangleCount=0; };
std::array<Tri,4> tetra(std::array<float,3> shift={0,0,0}){
 std::array<std::array<float,3>,8> p{};
 p[0]={shift[0],shift[1],shift[2]};p[1]={shift[0]+1,shift[1],shift[2]};
 p[2]={shift[0],shift[1]+1,shift[2]};p[3]={shift[0],shift[1],shift[2]+1};
 const std::array<std::array<unsigned,3>,4> f{{{{0,2,1}},{{0,1,3}},{{0,3,2}},{{1,2,3}}}};
 std::array<Tri,4> t{};for(unsigned i=0;i<4;++i)t[i]=tri(p,f[i],100+i);return t;
}
std::array<Tri,4> scaledTetra(float scale,std::uint32_t body,std::uint32_t bone,
                              std::uint32_t firstStable){
 std::array<std::array<float,3>,8> p{};
 p[0]={0,0,0};p[1]={scale,0,0};p[2]={0,scale,0};p[3]={0,0,scale};
 const std::array<std::array<unsigned,3>,4> f{{{{0,2,1}},{{0,1,3}},{{0,3,2}},{{1,2,3}}}};
 std::array<Tri,4> t{};for(unsigned i=0;i<4;++i)t[i]=tri(p,f[i],firstStable+i,body,bone);return t;
}
}
int main(){
 const auto bank=tetra();
 const auto inside=d37w::classifyClosedBank({0.1,0.2,0.15},bank);
 require(inside.state==d37w::BankContainmentState::parityInside,"tetra interior classified inside");
 require(inside.parity.insideRays==3&&inside.parity.outsideRays==0,"all parity rays agree inside");
 require(d37w::bankAabbContains(inside.bounds,{0.1,0.2,0.15}),"interior lies inside exact bank AABB");
 // Any point classified inside a closed bank must lie within the vertex AABB.
 for(double x:{0.07,0.19,0.31})for(double y:{0.11,0.23})for(double z:{0.13,0.29}){
  const auto c=d37w::classifyClosedBank({x,y,z},bank);
  if(c.state==d37w::BankContainmentState::parityInside)
   require(d37w::bankAabbContains(c.bounds,{x,y,z}),"AABB cull cannot remove an inside point");
 }
 const auto parityOutside=d37w::classifyClosedBank({0.8,0.8,0.8},bank);
 require(parityOutside.state==d37w::BankContainmentState::parityOutside,
         "point in AABB but outside tetra is parity-classified, not culled");
 const auto shifted=tetra({3,0,0});
 const auto culled=d37w::classifyClosedBank({0.1,0.2,0.15},shifted);
 require(culled.state==d37w::BankContainmentState::aabbProvenOutside,
         "point outside bank AABB is reported as proven outside");
 const std::array<std::array<Tri,4>,3> banks{{bank,bank,shifted}};
 std::size_t in=0,out=0,culledCount=0;
 for(const auto& b:banks){
  const auto c=d37w::classifyClosedBank({0.1,0.2,0.15},b);
  in+=c.state==d37w::BankContainmentState::parityInside;
  out+=c.state==d37w::BankContainmentState::parityOutside;
  culledCount+=c.state==d37w::BankContainmentState::aabbProvenOutside;
 }
 require(in==2&&out==0&&culledCount==1,"every candidate bank receives a class or complete AABB proof");
 const std::span<const Tri> empty{};
 require(d37w::classifyClosedBank({0,0,0},empty).state==d37w::BankContainmentState::invalid,
         "empty bank fails closed as invalid");
 auto input=bank;for(auto& t:input)t.bodyIndex=0u;
 const auto built=bvh::buildStaticCompleteBodyBVH(input,input.size(),true);
 require(built.succeeded()&&built.bodies.size()==1u,"closed bank builds exact BVH");
 const auto& b=built.bodies[0];
 const std::array<FakeTree,1> tree{{{0u,b.firstNode,b.nodeCount,b.firstTriangle,b.triangleCount}}};
 const std::array<FakeBody,1> bodies{{FakeBody{}}};
 const std::array<std::uint8_t,1> role{{2u}};
 const std::array<std::uint32_t,1> roleIndex{{0u}};
 const auto zeroOffset=[](std::uint32_t){return std::array<double,3>{0.0,0.0,0.0};};
 std::array<FakeNode,1> node{{FakeNode{{0.1f,0.2f,0.15f,0.f}}}};
 std::ostringstream insideCsv,insideNodeSummary;
 const auto auditInside=d37w::writeAllBankContainment(insideCsv,node,bodies,tree,
     built.triangles,built.nodes,role,roleIndex,0u,1u,zeroOffset,&insideNodeSummary);
 require(auditInside.valid&&auditInside.insidePairs==1u&&auditInside.aabbProvenOutsidePairs==0u,
         "all-bank writer reports the containing bank");
 require(insideNodeSummary.str().find("0,0,1,0,2,1,0,0,1,0,0,1")!=std::string::npos,
         "node summary accounts for each bone bank and records the inside bank");
 node[0].positionAndMass={0.8f,0.8f,0.8f,0.f};
 std::ostringstream parityOutsideCsv;
 const auto auditParityOutside=d37w::writeAllBankContainment(parityOutsideCsv,node,bodies,tree,
     built.triangles,built.nodes,role,roleIndex,0u,1u,zeroOffset);
 require(auditParityOutside.valid&&auditParityOutside.parityOutsidePairs==1u,
         "all-bank writer parity-tests points within bounds");
 node[0].positionAndMass={2.f,2.f,2.f,0.f};
 std::ostringstream aabbOutsideCsv,aabbOutsideSummary;
 const auto auditAabbOutside=d37w::writeAllBankContainment(aabbOutsideCsv,node,bodies,tree,
     built.triangles,built.nodes,role,roleIndex,0u,1u,zeroOffset,&aabbOutsideSummary);
 require(auditAabbOutside.valid&&auditAabbOutside.aabbProvenOutsidePairs==1u&&
         aabbOutsideCsv.str().find("global_node\n")==std::string::npos&&
         aabbOutsideCsv.str().find("local_node,global_node") == 0u,
         "all-bank writer records AABB proof in counters and does not mislabel unknown");
 require(aabbOutsideSummary.str().find("0,0,1,0,2,1,1,0,0,0,0,0")!=std::string::npos,
         "node summary proves every bone bank was excluded by its AABB");
 // Two nested bone solids under one body must be classified independently.
 // Their union has even ray parity at the shared interior point, so body-level
 // parity would incorrectly report outside.
 auto outer=scaledTetra(1.0f,0u,101u,200u);
 auto inner=scaledTetra(0.5f,0u,202u,300u);
 std::vector<Tri> nested; nested.insert(nested.end(),outer.begin(),outer.end());
 nested.insert(nested.end(),inner.begin(),inner.end());
 const w::Vec3 nestedPoint{0.13,0.17,0.11};
 const auto unionParity=w::classifyClosedSolid(nestedPoint,nested);
 require(unionParity.outsideRays==3&&unionParity.insideRays==0,
         "nested union demonstrates why body-wide parity is invalid");
 const auto nestedBuilt=bvh::buildStaticCompleteBodyBVH(nested,nested.size(),true);
 require(nestedBuilt.succeeded()&&nestedBuilt.bodies.size()==1u,
         "two closed bone banks under one body pass exact per-bone closure validation");
 const auto& nestedRange=nestedBuilt.bodies[0];
 const std::array<FakeTree,1> nestedTree{{{0u,nestedRange.firstNode,nestedRange.nodeCount,
     nestedRange.firstTriangle,nestedRange.triangleCount}}};
 std::array<FakeNode,1> nestedNode{{FakeNode{{0.13f,0.17f,0.11f,0.f}}}};
 std::ostringstream nestedCsv,nestedNodeSummary;
 const auto nestedAudit=d37w::writeAllBankContainment(nestedCsv,nestedNode,bodies,nestedTree,
     nestedBuilt.triangles,nestedBuilt.nodes,role,roleIndex,0u,1u,zeroOffset,&nestedNodeSummary);
 require(nestedAudit.valid&&nestedAudit.banks==2u&&nestedAudit.insidePairs==2u,
         "all-bank audit reports each containing bone independently");
 require(nestedCsv.str().find(",0,101,inside,")!=std::string::npos&&
         nestedCsv.str().find(",0,202,inside,")!=std::string::npos,
         "CSV preserves body and stable bone identities separately");
 require(nestedNodeSummary.str().find("0,0,1,0,2,2,0,0,2,0,0,2")!=std::string::npos,
         "node summary accounts for both nested bone banks independently");
 std::puts("PASS: exact per-bone AABB cull, closed-solid parity, and all-bank writer fixtures");
}
