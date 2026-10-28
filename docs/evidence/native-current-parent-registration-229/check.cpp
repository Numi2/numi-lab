
#include "NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
static bool expect(bool x,const char* why){if(!x)std::cerr<<why<<"\n";return x;}
static PhysicalSkinRegisteredSurfaceCSVFiles files(const std::filesystem::path& d){
 PhysicalSkinRegisteredSurfaceCSVFiles f;f.donorCount=157;f.d37InnerFaceCount=4777;f.d37DeepFaceCount=4777;
 std::string e;
 for(unsigned k=0;k<3;++k){auto role=std::array<std::string,3>{"O","I","D"}[k];auto& l=f.layers[k];
  l.nodes=d/("registered-"+role+".csv");l.faces=d/("registered-"+role+"-faces.csv");
  l.name=role;l.sourceMember=l.nodes.string();
  if(!physical_skin_child_mass_detail::sha256File(l.nodes,l.nodesSHA256,e)||!physical_skin_child_mass_detail::sha256File(l.faces,l.facesSHA256,e))throw std::runtime_error(e);
 }
 f.nodeRegistration=d/"node-parent-registration.csv";f.donorFrames=d/"donor-frames.csv";f.outerFaceAncestry=d/"O-face-ancestry.csv";
 if(!physical_skin_child_mass_detail::sha256File(f.nodeRegistration,f.nodeRegistrationSHA256,e)||
    !physical_skin_child_mass_detail::sha256File(f.donorFrames,f.donorFramesSHA256,e)||
    !physical_skin_child_mass_detail::sha256File(f.outerFaceAncestry,f.outerFaceAncestrySHA256,e))throw std::runtime_error(e);
 return f;
}
int main(int argc,char** argv){
 if(argc!=3&&argc!=4)return 2;
 PhysicalSkinParentMassPacket p;p.currentWorldPoints={V3{0,0,0},V3{1,0,0},V3{0,1,0},V3{0,0,1},V3{2,0,0},V3{3,0,0},V3{2,1,0},V3{2,0,1}};
 p.tetrahedra={{0,1,2,3},{4,5,6,7},{0,1,2,3}};p.materialIndices={15,15,16};
 PhysicalSkinRegisteredSurfaceCSVSet s;
 for(unsigned l=0;l<3;++l){s.surfaces[l].worldPoints={{.25,.25,.25}};
  PhysicalSkinParentRegistrationRow r;r.available=true;r.tet=l==2?2:0;r.bary={.25,.25,.25,.25};s.nodeParent[l]={r};}
 PhysicalSkinSurfaceParentRegistrationReport r;std::string e;
 if(!expect(validatePhysicalSkinSurfaceParentRegistration(s,p,r,e),"synthetic exact stencil failed"))return 1;
 auto bad=s;bad.nodeParent[0][0].tet=1;PhysicalSkinSurfaceParentRegistrationReport kept;kept.checkedNodes={123,456,789};
 if(!expect(!validatePhysicalSkinSurfaceParentRegistration(bad,p,kept,e)&&e.find("does not reconstruct")!=std::string::npos&&kept.checkedNodes==std::array<std::size_t,3>{123,456,789},"wrong-cell corruption not rejected atomically"))return 1;
 bad=s;bad.nodeParent[0][0].available=false;
 if(!expect(!validatePhysicalSkinSurfaceParentRegistration(bad,p,kept,e),"unavailable O unexpectedly accepted"))return 1;
 bad=s;bad.nodeParent[2][0].tet=0;
 if(!expect(!validatePhysicalSkinSurfaceParentRegistration(bad,p,kept,e)&&e.find("material")!=std::string::npos,"wrong material unexpectedly accepted"))return 1;
 PhysicalSkinParentMassPacket actual;
 if(!expect(loadCompiledSkinFatParentMassPacketCSV(argv[2],actual,e),e.c_str()))return 1;
 if(!expect(loadCompiledSkinFatParentLabelsCSV(std::string(argv[2])+".labels.csv",actual,e),e.c_str()))return 1;
 auto f=files(argv[1]);PhysicalSkinRegisteredSurfaceCSVSet old;PhysicalSkinRegisteredSurfaceCSVReport parsed;
 if(!expect(loadPhysicalSkinRegisteredSurfaceCSVSet(f,old,parsed,e),e.c_str()))return 1;
 PhysicalSkinRegisteredSurfaceCSVSet atomic=s;PhysicalSkinRegisteredSurfaceCSVReport oldParsed;oldParsed.nodes={11,22,33};
 const bool accepted=loadPhysicalSkinRegisteredSurfaceCSVSetForParent(f,actual,atomic,oldParsed,kept,e);
 const bool expectPass=argc==4&&std::string(argv[3])=="--expect-pass";
 if(expectPass){
  if(!expect(accepted,e.c_str()))return 1;
  std::cout<<"actual_parent_registration=PASS checked="<<kept.checkedNodes[0]<<","<<kept.checkedNodes[1]<<","<<kept.checkedNodes[2]<<" max_error_m="<<kept.maximumReconstructionErrorM[0]<<","<<kept.maximumReconstructionErrorM[1]<<","<<kept.maximumReconstructionErrorM[2]<<"\n";
 }else{
  if(!expect(!accepted&&e.find("no valid parent-cell registration")!=std::string::npos&&atomic.surfaces[0].worldPoints==s.surfaces[0].worldPoints&&oldParsed.nodes==std::array<std::size_t,3>{11,22,33}&&kept.checkedNodes==std::array<std::size_t,3>{123,456,789},"baseline incomplete registration not rejected atomically"))return 1;
  std::cout<<"baseline_parent_registration=EXPECTED_REJECT reason="<<e<<"\n";
  // Test the real I/D reconstruction separately. Copying I into diagnostic O
  // here exercises the import arithmetic only, never a child geometry candidate.
  auto diagnostic=old;diagnostic.surfaces[0]=old.surfaces[1];diagnostic.nodeParent[0]=old.nodeParent[1];
  if(!validatePhysicalSkinSurfaceParentRegistration(diagnostic,actual,r,e)){
    std::cout<<"actual_ID_parent_reconstruction=REJECT reason="<<e<<"\n";
  }else std::cout<<"actual_ID_parent_reconstruction=PASS I="<<r.checkedNodes[1]<<" D="<<r.checkedNodes[2]<<" error_m="<<r.maximumReconstructionErrorM[1]<<","<<r.maximumReconstructionErrorM[2]<<"\n";
 }
 std::cout<<"parent_reconstruction_guard_regressions=PASS native_physical_steps=0\n";
}
