#include "NumiHumanSkinFatFixedCavityReconditioning.hpp"
#include <fstream>
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main(int argc,char** argv) {
 if(argc!=4)return 2;
 using namespace current_fractional_support_npz_detail;
 std::string error;std::map<std::string,NpyArray> arrays;
 if(!readRequestedArrays(argv[1],{"points_reference_world_m_f64","volume_tetrahedra"},arrays,error)) {
  std::cerr<<error;return 1;
 }
 std::vector<double> p;std::vector<std::uint32_t> t;
 if(!readF64(arrays.at("points_reference_world_m_f64"),p,error)||
    !readU32(arrays.at("volume_tetrahedra"),t,error)||p.size()%3u||t.size()%4u)return 1;
 std::vector<V3> points(p.size()/3u);
 for(std::size_t i=0;i<points.size();++i)for(unsigned a=0;a<3;++a)points[i][a]=p[3*i+a];
 std::vector<skin_fat_cavity_detail::Tet> tets(t.size()/4u);
 for(std::size_t i=0;i<tets.size();++i)std::copy_n(t.begin()+4*i,4,tets[i].begin());
 arrays.clear();p.clear();t.clear();
 if(!readRequestedArrays(argv[2],{"replacement_tetrahedra","replacement_parent_cell_rows"},arrays,error))return 1;
 std::vector<std::uint32_t> repl,parents;
 if(!readU32(arrays.at("replacement_tetrahedra"),repl,error)||
    !readU32(arrays.at("replacement_parent_cell_rows"),parents,error)||
    repl.size()%8u||parents.size()!=repl.size()/4u*3u)return 1;
 std::vector<std::size_t> passing;std::map<std::string,std::vector<std::size_t>> failed;
 for(std::size_t i=0;i<repl.size()/8u;++i) {
  std::array<skin_fat_cavity_detail::Tet,3> old;
  std::array<skin_fat_cavity_detail::Tet,2> next;
  for(unsigned k=0;k<3;++k) {
   const auto row=parents[6*i+k];if(row>=tets.size())return 1;old[k]=tets[row];
  }
  for(unsigned k=0;k<2;++k)std::copy_n(repl.begin()+8*i+4*k,4,next[k].begin());
  skin_fat_cavity_detail::Edge edge;skin_fat_cavity_detail::Triangle ring;
  double before,after,fraction;
  if(skin_fat_cavity_detail::certifyGeometry(points,old,next,edge,ring,before,after,fraction,error))
   passing.push_back(i);
  else failed[error].push_back(i);
 }
 std::ofstream out(argv[3]);
 out<<"{\"scope\":\"existing_native_certifyGeometry_on_emitted_Float32_reference_coordinates_not_mass_qualification\",\"eligible_indices\":[";
 for(std::size_t i=0;i<passing.size();++i){if(i)out<<",";out<<passing[i];}out<<"],\"rejected\":{";
 bool first=true;for(const auto& [reason,indices]:failed) {
  if(!first)out<<",";first=false;out<<"\""<<reason<<"\":[";
  for(std::size_t i=0;i<indices.size();++i){if(i)out<<",";out<<indices[i];}out<<"]";
 }out<<"}}\n";out.flush();
 std::cout<<"native_geometry_screen passing="<<passing.size()<<" total="<<repl.size()/8u
  <<" rejected_reason_count="<<failed.size()<<"\n";
 return out.good()?0:1;
}
