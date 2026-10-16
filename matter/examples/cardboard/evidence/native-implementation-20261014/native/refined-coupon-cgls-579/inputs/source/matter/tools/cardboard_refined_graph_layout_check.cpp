#include "cardboard_glue_mesh.hpp"
#include "regional_graph_schwarz.hpp"
#include "numi/matter/detail/pure_fem_armijo_cgls_arena_plan.hpp"
#include "numi/matter/detail/true_residual_arena_plan.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <array>
using namespace numi::matter::detail;
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
bool same(const RegionalGraphSchwarzLayout&a,const RegionalGraphSchwarzLayout&b){
 return a.patchNodes==b.patchNodes&&a.rows==b.rows&&a.nodePatchOffsets==b.nodePatchOffsets&&a.nodePatchValues==b.nodePatchValues&&a.nodeCoverage==b.nodeCoverage&&a.matrixFloatCount==b.matrixFloatCount&&a.pivotCount==b.pivotCount&&a.deviceBytes==b.deviceBytes;
}
int main(){try{
 unsigned cases=0,extended=0;
 for(unsigned x:{8u,16u,32u})for(unsigned y:{2u,4u,8u})for(unsigned z:{1u,2u,4u}){
  numi::cardboard::GlueMeshConfig c{};
  c.length=.0316;c.pitch=.0079;c.caliper=.004210;c.linerThickness=.000277;c.mediumThickness=.000191;
  c.glueMinimumGap=.0001;c.bondWidth=.0008;c.upperGlueMinimumGap=.00003;c.upperBondWidth=.0006;c.nxPerPitch=x;c.thicknessSlices=z;
  auto m=numi::cardboard::buildConformingGlueCrossSection(c);
  unsigned points=m.points.size(),nodes=points*(y+1);
  std::vector<std::array<unsigned,4>>tets;
  for(unsigned j=0;j<y;++j)for(auto triangle:m.triangles){
   auto p=triangle.points;std::sort(p.begin(),p.end());unsigned a=j*points+p[0],b=j*points+p[1],c0=j*points+p[2];
   // Same three sorted-node prism memberships as the production addPrism.
   // Its positive-volume orientation swaps do not change graph adjacency.
   tets.push_back({a,b,c0,c0+points});tets.push_back({a,b,b+points,c0+points});tets.push_back({a,a+points,b+points,c0+points});
  }
  std::vector<std::uint8_t>mask(nodes,1);
  for(unsigned j=0;j<=y;++j)for(unsigned i=0;i<points;++i)
   if(std::abs(m.points[i].x)<=1e-10||std::abs(m.points[i].x-c.length)<=1e-10)mask[j*points+i]=0;
  auto legacy=buildBudgetedRegionalGraphSchwarzLayout(nodes,tets,mask,2);
  auto graph=buildBudgetedRegionalGraphSchwarzLayout(nodes,tets,mask,2,96ull*1024*1024,true);
  if(!graph.error.empty()){std::cerr<<x<<" "<<y<<" "<<z<<" "<<graph.error<<"\n";return 2;}
  require(graph.deviceBytes<=96ull*1024*1024,"graph exceeds byte budget");
  require(std::all_of(graph.nodeCoverage.begin(),graph.nodeCoverage.end(),[](auto n){return n>0;}),"uncovered node");
  auto repeated=buildBudgetedRegionalGraphSchwarzLayout(nodes,tets,mask,2,96ull*1024*1024,true);
  require(repeated.error.empty()&&same(graph,repeated),"nondeterministic graph layout");
  if(legacy.error.empty())require(same(legacy,graph),"changed an admitted legacy layout");else ++extended;
  FEMPureArmijoArenaShape sh{};sh.environmentCount=2;sh.femNodeCount=nodes;sh.tetrahedronCount=tets.size();sh.unknownCount=4ull*nodes;sh.maxBufferLength=1ull<<30;sh.oneStaticExplicitFEM=true;sh.includeCauchy=true;sh.includeCGLS=true;
  auto pure=planFEMPureArmijoCGLSArena(sh);
  auto residual=planFEMTrueResidualArena({2,nodes,tets.size(),0,0,1ull<<30,true});
  require(pure.admitted(),"pure arena rejected");require(residual.admitted(),"true residual arena rejected");
  unsigned maxPatch=0;for(auto p:graph.patches)maxPatch=std::max(maxPatch,p.nodeCount);
  std::cout<<"{\"nx\":"<<x<<",\"ny\":"<<y<<",\"thickness\":"<<z<<",\"nodes\":"<<nodes<<",\"tets\":"<<tets.size()<<",\"patches\":"<<graph.patches.size()<<",\"max_patch_nodes\":"<<maxPatch<<",\"graph_bytes\":"<<graph.deviceBytes<<",\"legacy_admitted\":"<<(legacy.error.empty()?"true":"false")<<",\"pure_scratch_bytes\":"<<pure.scratchBytes<<",\"true_residual_bytes\":"<<residual.aggregateBytes<<"}\n";
  ++cases;
 }
 std::array<std::array<unsigned,4>,1> t{{{0,1,2,3}}};std::vector<std::uint8_t> mask(32769,1);
 require(!buildBudgetedRegionalGraphSchwarzLayout(32769,t,mask,2,96ull*1024*1024,true).error.empty(),"node cap bypass");
 mask.assign(8,1);
 require(!buildBudgetedRegionalGraphSchwarzLayout(8,t,mask,5,96ull*1024*1024,true).error.empty(),"environment cap bypass");
 require(!buildBudgetedRegionalGraphSchwarzLayout(8,t,mask,2,1,true).error.empty(),"byte cap bypass");
 t[0][3]=8;
 require(!buildBudgetedRegionalGraphSchwarzLayout(8,t,mask,2,96ull*1024*1024,true).error.empty(),"invalid tet admitted");
 require(cases==27,"missing refinement cases");
 std::cout<<"{\"all_pass\":true,\"cases\":"<<cases<<",\"extended_cases\":"<<extended<<",\"physical_steps\":0,\"mesh_convergence_qualified\":false}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
