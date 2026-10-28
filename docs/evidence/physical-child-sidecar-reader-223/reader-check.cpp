#include "NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;

static PhysicalSkinRegisteredSurfaceCSVFiles filesFor(const std::filesystem::path& d,bool normalized){
 PhysicalSkinRegisteredSurfaceCSVFiles f;
 f.layers[0]={d/"registered-O.csv",d/"registered-O-faces.csv",
 "ec18bb15e679b643d8f878a6051786c26244865e33ddaa19e6135eefd8decc05",
 "83031346f2b2eb90ebc07f15c6192d37c8f9a854d5bd867f497320912ac60346",
 "O registered candidate",d.filename().string()+"/registered-O.csv",0};
 f.layers[1]={d/"registered-I.csv",d/"registered-I-faces.csv",
 normalized?"746f3db1679b1db94112ee183e148fa4351b3582eeb33acac6142f30287bf680":
 "0f48361523e5aca266cfd599a62a61d5da1e3ec8bc09b11f0e078b155209561e",
 "0a08d73b4dc4d6cd5f7260a8795364e4da49991c9ba50e0cf14d8bfa1d909cc1",
 "I registered layer",d.filename().string()+"/registered-I.csv",0};
 f.layers[2]={d/"registered-D.csv",d/"registered-D-faces.csv",
 normalized?"be864434b74a46fe1c0b0c9316238fec6dff527c7c6523f450ccbaa0760ec8dc":
 "311c9a2cdd2d86155af7500f8a5b097848087094a25018b25bd015d460360aa1",
 "8f380ce85a2625be901caf3eaa054519b4c359f197acdfa10de3cd0c2df1fea6",
 "D registered layer",d.filename().string()+"/registered-D.csv",0};
 f.nodeRegistration=d/"node-parent-registration.csv";
 f.nodeRegistrationSHA256=normalized?
 "f22cd8a808f101b00e7daaab882fc16ac8748beea8319281d03d977fb3ef4de9":
 "0f5de74856bf3b7a0b0b79aa4f76f9e3a3b81d70ef002ac853f3eb952b0c838d";
 f.donorFrames=d/"donor-frames.csv";
 f.donorFramesSHA256="5ab076c2861a4e965d807c5bfebce9de3a54a348118b664a573028b62f19b74d";
 f.outerFaceAncestry=d/"O-face-ancestry.csv";
 f.outerFaceAncestrySHA256="33d9e894ed35c4807dce94aad6ad8a7da6ed08603ea6476afd635497757f0feb";
 f.donorCount=157;f.d37InnerFaceCount=4777;f.d37DeepFaceCount=4777;
 return f;
}

int main(int argc,char**argv){
 if(argc!=2)return 2;
 const std::filesystem::path root(argv[1]);
 auto f6=filesFor(root/"attempt-006",false);
 PhysicalSkinRegisteredSurfaceCSVSet old;PhysicalSkinRegisteredSurfaceCSVReport oldReport;std::string error;
 if(loadPhysicalSkinRegisteredSurfaceCSVSet(f6,old,oldReport,error)||
    error.find("I/D node row 2411 source barycentric fails native tolerance")==std::string::npos||
    !old.surfaces[0].worldPoints.empty()){
  std::cerr<<"attempt-006 strict rejection was not stable and atomic: "<<error<<"\n";return 1;
 }

 auto f7=filesFor(root/"attempt-007-normalized-source-bary",true);
 PhysicalSkinRegisteredSurfaceCSVSet current;PhysicalSkinRegisteredSurfaceCSVReport r;error.clear();
 if(!loadPhysicalSkinRegisteredSurfaceCSVSet(f7,current,r,error)){
  std::cerr<<"attempt-007 sidecar import failed: "<<error<<"\n";return 1;
 }
 if(r.nodes!=std::array<std::size_t,3>{32218,15022,15022}||
    r.faces!=std::array<std::size_t,3>{63959,29999,29999}||
    r.parentUnmappedNodes[0]!=32218||r.parentMappedNodes[1]!=15022||
    r.parentMappedNodes[2]!=15022||r.donorFrames!=86||
    r.directSourceNodes[1]!=2411||r.faceEmbeddedNodes[1]!=12611||
    r.directSourceNodes[2]!=2411||r.faceEmbeddedNodes[2]!=12611){
  std::cerr<<"attempt-007 validated counts differ from pinned receipt\n";return 1;
 }

 auto bad=f7;bad.layers[2].facesSHA256=std::string(64,'0');
 PhysicalSkinRegisteredSurfaceCSVSet rejected;PhysicalSkinRegisteredSurfaceCSVReport rr;error.clear();
 if(loadPhysicalSkinRegisteredSurfaceCSVSet(bad,rejected,rr,error)||
    error.find("hash mismatch")==std::string::npos||!rejected.surfaces[0].worldPoints.empty()){
  std::cerr<<"bad-hash input was not rejected atomically: "<<error<<"\n";return 1;
 }
 std::cout<<"registered-sidecar-reader=PASS attempt006=expected_reject_I2411 attempt007=PASS"
          <<" O="<<r.nodes[0]<<"/"<<r.faces[0]<<" I="<<r.nodes[1]<<"/"<<r.faces[1]
          <<" D="<<r.nodes[2]<<"/"<<r.faces[2]<<" donor_frames="<<r.donorFrames
          <<" I_D_parent_registration="<<r.parentMappedNodes[1]<<"/"<<r.parentMappedNodes[2]
          <<" O_parent_registration_unavailable="<<r.parentUnmappedNodes[0]
          <<" I_source_vertices="<<r.directSourceNodes[1]<<" I_source_faces="<<r.faceEmbeddedNodes[1]
          <<" D_source_vertices="<<r.directSourceNodes[2]<<" D_source_faces="<<r.faceEmbeddedNodes[2]
          <<" bad_hash=reject_atomic physical_admission=false child_volume_maps=unavailable\n";
}
