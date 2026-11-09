#include "../../matter/src/fem_plane_quadrature.hpp"
#include <cmath>
#include <iostream>
using namespace numi::matter;
static_assert(sizeof(NMPlaneQuadratureRowGPU)==48);
static_assert(sizeof(NMPlaneQuadratureIncidenceGPU)==8);
int main() {
 // Independent simplex moments, normalized to unit triangle area.
 for (unsigned a=0;a<=5;++a) for(unsigned b=0;b<=5-a;++b)
  for(unsigned c=0;c<=5-a-b;++c) {
   double got=0;
   for(const auto& q:detail::femPlaneQuadrature) {
    if (!(q.weight>0) || !(q.barycentric[0]>0) || !(q.barycentric[1]>0) || !(q.barycentric[2]>0)) return 5;
    got+=q.weight*std::pow(q.barycentric[0],a)*std::pow(q.barycentric[1],b)*std::pow(q.barycentric[2],c);
   }
   const double expected=2*std::tgamma(a+1)*std::tgamma(b+1)*std::tgamma(c+1)/std::tgamma(a+b+c+3);
   if(std::abs(got-expected)>2e-14){std::cerr<<a<<","<<b<<","<<c;return 6;}
  }
 for(unsigned divisions:{1u,2u,4u,8u}) {
  CompiledWorld w;NMRigidProxyGPU p{};p.localCenterAndRadius={0,0,1,0};w.contact.rigidProxies.push_back(p);
  const auto addNode=[&](float x,float y,float z) {
   NMFEMNodeStateGPU v{};v.restAndFixed={x,y,z,0};w.fem.nodes.push_back(v);return unsigned(w.fem.nodes.size()-1);
  };
  const float h=.02f/divisions;
  for(unsigned y=0;y<divisions;++y)for(unsigned x=0;x<divisions;++x) {
   unsigned a=addNode(h*x,h*y,0),b=addNode(h*(x+1),h*y,0),
      c=addNode(h*(x+1),h*(y+1),0),d=addNode(h*x,h*(y+1),0),apex=addNode(h*x,h*y,.02);
   for(auto ids:{std::array<unsigned,3>{a,b,c},std::array<unsigned,3>{a,c,d}}) {
    NMTetrahedronGPU t{};t.nodes={ids[0],ids[1],ids[2],apex};unsigned j=unsigned(w.fem.tetrahedra.size());w.fem.tetrahedra.push_back(t);
    NMFEMSurfaceFaceGPU f{};f.adjacency={j,NM_INVALID_INDEX,0,0};f.sides={3,0,0,0};w.fem.surfaceFaces.push_back(f);
   }
  }
  std::vector<float> areas;std::vector<detail::FEMPlaneReferenceTriangle> triangles;std::string error;
  if(!detail::femFixedPlaneReferenceAreas(w,areas,error,&triangles)){std::cerr<<error;return 1;}
  double sum=0;for(float a:areas)sum+=a;
  double triangleSum=0;for(const auto& t:triangles){if(!(t.referenceArea>0))return 7;triangleSum+=t.referenceArea;}
  if(triangles.size()!=2*divisions*divisions || std::abs(triangleSum-sum)>1e-10)return 8;
  detail::FEMPlaneQuadratureLayout layout;
  if(!detail::femPlaneQuadratureLayout(w.fem.nodes.size(),triangles,layout,error))return 9;
  if(layout.rows.size()!=triangles.size()*7 || layout.incidence.size()!=layout.rows.size()*3 ||
     layout.offsets.size()!=w.fem.nodes.size()+1 || layout.offsets.back()!=layout.incidence.size())return 10;
  double rowArea=0;for(const auto& row:layout.rows){
   if(row.nodes.w!=0 || !(row.shapeAndWeightedArea.w>0) ||
      std::abs(row.shapeAndWeightedArea.x+row.shapeAndWeightedArea.y+row.shapeAndWeightedArea.z-1.f)>1e-6f)return 11;
   rowArea+=double(row.shapeAndWeightedArea.w)+row.shapeAndWeightedAreaLow.w;
  }
  if(std::abs(rowArea-triangleSum)>1e-16)return 12;
  for(size_t node=0;node<w.fem.nodes.size();++node){
   std::uint32_t previous=0;bool first=true;
   for(auto i=layout.offsets[node];i<layout.offsets[node+1];++i){
    const auto incidence=layout.incidence[i];if(incidence.row>=layout.rows.size()||incidence.corner>2)return 13;
    const auto n=layout.rows[incidence.row].nodes;const unsigned ids[3]{n.x,n.y,n.z};
    if(ids[incidence.corner]!=node || (!first && incidence.row<=previous))return 14;
    previous=incidence.row;first=false;
   }
  }
  auto bad=triangles;bad.front().nodes[1]=bad.front().nodes[0];
  const auto priorSize=layout.rows.size();
  if(detail::femPlaneQuadratureLayout(w.fem.nodes.size(),bad,layout,error) || layout.rows.size()!=priorSize)return 15;
  bad=triangles;bad.front().referenceArea=std::numeric_limits<double>::infinity();
  if(detail::femPlaneQuadratureLayout(w.fem.nodes.size(),bad,layout,error))return 16;
  bad=triangles;bad.front().referenceArea=1e-300;
  if(detail::femPlaneQuadratureLayout(w.fem.nodes.size(),bad,layout,error))return 17;

  if(std::abs(sum-.0004)>1e-9){std::cerr<<divisions<<" "<<sum;return 2;}
  if(areas.back()!=0)return 3;
  w.fem.nodes[0].restAndFixed.z=.001f;
  if(detail::femFixedPlaneReferenceAreas(w,areas,error))return 4;
 }
 std::cout<<"positive triangle rule, sorted node incidence and reference area pass; invalid patches fail closed\\n";
}
