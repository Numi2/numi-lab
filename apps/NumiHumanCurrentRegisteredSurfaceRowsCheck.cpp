#include "NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp"
#include <chrono>
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main(){
 namespace detail=physical_skin_registered_surface_csv_detail;
 const auto root=std::filesystem::temp_directory_path()/
  ("numi-current-ID-rows-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 std::filesystem::create_directory(root);
 PhysicalSkinRegisteredLayerCSV files;files.nodes=root/"nodes.csv";files.faces=root/"faces.csv";
 files.name="typed fixture";files.sourceMember="fixture";
 const std::string header="vertex,donor,wx,wy,wz,nx,ny,nz,lx,ly,lz,m00,m01,m02,m10,m11,m12,m20,m21,m22,source_domain,d37_parent_node_row,d37_parent_surface_face_row,beta0,beta1,beta2,source_residual_x_m,source_residual_y_m,source_residual_z_m,frame_owner_method,source_vertex_row\n";
 const std::string rows="0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,typed-ID,0,-1,0,0,0,0,0,0,fixture,70000\n"
 "1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,typed-ID,1,-1,0,0,0,0,0,0,fixture,70001\n"
 "2,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,typed-ID,2,-1,0,0,0,0,0,0,fixture,70002\n";
 const std::string faceHeader="face,v0,v1,v2,d37_parent_surface_face_row,d37_parent_tet_row,material_index,source_role\n";
 std::vector<PhysicalSkinDonorFrameRow> frames(1);frames[0].available=true;
 frames[0].currentR=frames[0].restR={1,0,0,0,1,0,0,0,1};
 RegisteredSkinCurrentSourceDomain domain{"typed-ID",std::string(64,'a'),80000,120000};
 unsigned checks=0;
 for(unsigned role=1;role<=2;++role){
  const std::string face="0,0,1,2,110000,2,"+std::to_string(role==1?15:16)+",fixture\n";
  auto read=[&](std::string data,bool typed){
   std::ofstream(files.nodes)<<header<<data;std::ofstream(files.faces)<<faceHeader<<face;
   std::string ns,fs,error;
   if(!physical_skin_child_mass_detail::sha256File(files.nodes,files.nodesSHA256,error)||
      !physical_skin_child_mass_detail::sha256File(files.faces,files.facesSHA256,error))return false;
   RegisteredSurface surface;std::vector<PhysicalSkinNodeSourceRow> sources;
   std::vector<PhysicalSkinFaceSourceRow> faces;
   const bool ok=detail::readID(files,role,1,frames,std::string(64,'b'),surface,sources,faces,ns,fs,error,typed?&domain:nullptr);
   if(!ok&&data==rows&&typed)std::cerr<<"positive fixture: "<<error<<"\n";
   if(ok&&(surface.sourceVertexIndices!=std::vector<std::uint32_t>{70000,70001,70002}||
          sources.back().vertex!=70002||sources.back().domain!="typed-ID"||faces[0].sourceFace!=110000))return false;
   return ok;
  };
  if(!read(rows,true))return 1;++checks;
  if(read(rows,false))return 2;++checks;
  auto changed=rows;changed.replace(changed.find("70000"),5,"80000");if(read(changed,true))return 3;++checks;
  changed=rows;changed.replace(changed.find("typed-ID"),8,"wrong-ID");if(read(changed,true))return 4;++checks;
  changed=rows;changed.replace(changed.find("typed-ID,0,-1"),13,"typed-ID,9,-1");if(read(changed,true))return 5;++checks;
  changed=rows;changed.replace(changed.find("1,0,1,0,0"),9,"1,1,1,0,0");if(read(changed,true))return 6;++checks;
 }
 std::filesystem::remove_all(root);
 std::cout<<"current_registered_ID_source_rows=PASS checks="<<checks<<" physical_qualification=not_claimed\n";
}
