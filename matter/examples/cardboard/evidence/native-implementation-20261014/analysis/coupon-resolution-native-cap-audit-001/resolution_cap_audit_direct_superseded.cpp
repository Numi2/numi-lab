#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include "cardboard_glue_mesh.hpp"
#include "regional_graph_schwarz.hpp"
int main(){
 constexpr double length=.0316,pitch=.0079,caliper=.00421,liner=.000277,medium=.000191,gap=.0001,upperGap=.00003,bond=.0008,upperBond=.0006;
 std::cout<<"nx,ny,t,points2d,triangles2d,nodes,tets,authored_fixed_nodes,contact_capacity_if_contact_enabled,graph_patches,graph_matrix_floats,graph_bytes,graph_status,selector5_node_cap_pass,selector5_tet_cap_pass,unrelated_fullcontact_capacity_cap_pass\n";
 for(unsigned nx: {8u,16u,32u}) for(unsigned ny: {2u,4u,8u}) for(unsigned t: {1u,2u,4u}) {
  numi::cardboard::GlueMeshConfig c{.length=length,.pitch=pitch,.caliper=caliper,.linerThickness=liner,.mediumThickness=medium,.glueMinimumGap=gap,.bondWidth=bond,.upperGlueMinimumGap=upperGap,.upperBondWidth=upperBond,.nxPerPitch=nx,.thicknessSlices=t,.allowPartialTerminalPitch=false};
  auto m=numi::cardboard::buildConformingGlueCrossSection(c);
  const unsigned p=static_cast<unsigned>(m.points.size());
  std::vector<std::array<std::uint32_t,4>> tet;
  tet.reserve(m.triangles.size()*ny*3ull);
  for(unsigned y=0;y<ny;++y) for(const auto& tri:m.triangles){
   auto ids=tri.points;std::sort(ids.begin(),ids.end());
   auto b0=y*p+ids[0],b1=y*p+ids[1],b2=y*p+ids[2],u0=b0+p,u1=b1+p,u2=b2+p;
   tet.push_back({b0,b1,b2,u2});tet.push_back({b0,b1,u1,u2});tet.push_back({b0,u0,u1,u2});
  }
  const unsigned nodes=p*(ny+1);
  std::vector<std::uint8_t> freeMask(nodes,1);
  for(unsigned y=0;y<=ny;++y)for(unsigned i=0;i<p;++i){double x=m.points[i].x;if(std::abs(x)<1e-10||std::abs(x-length)<1e-10)freeMask[y*p+i]=0;}
  const auto g=numi::matter::detail::buildRegionalGraphSchwarzLayout(nodes,tet,freeMask,2u);
  std::string status=g.error.empty()?"admitted":"rejected";
  std::string error=g.error; std::replace(error.begin(),error.end(),',',';');
  std::cout<<nx<<','<<ny<<','<<t<<','<<p<<','<<m.triangles.size()<<','<<nodes<<','<<tet.size()<<','<<std::count(freeMask.begin(),freeMask.end(),std::uint8_t{0})<<','<<(8ull*tet.size())<<','<<g.patches.size()<<','<<g.matrixFloatCount<<','<<g.deviceBytes<<','<<status<<':'<<error<<','<<(nodes<=4096)<<','<<(tet.size()<=32768)<<','<<(8ull*tet.size()<=131072)<<'\n';
 }
}
