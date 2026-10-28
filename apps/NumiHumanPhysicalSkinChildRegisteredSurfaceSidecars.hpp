#pragma once
#include "NumiHumanRegisteredSkinCandidateCSV.hpp"
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {

// Reads the current registered O/I/D CSV contract and keeps its source
// ancestry distinct from parent-reference registration. It does not create
// child volume maps and cannot by itself admit physical geometry.
struct PhysicalSkinRegisteredLayerCSV {
    std::filesystem::path nodes,faces;
    std::string nodesSHA256,facesSHA256,name,sourceMember;
    std::uint32_t stableIdentifier=0;
};
enum class PhysicalSkinPoseArithmetic { strictDouble, nativeFloat32 };
struct PhysicalSkinDonorFrameRow {
    bool available=false;
    PhysicalSkinPoseArithmetic arithmetic=PhysicalSkinPoseArithmetic::strictDouble;
    std::uint32_t binding=NM_INVALID_INDEX;
    V3 currentOrigin{},restOrigin{};
    std::array<double,9> currentR{},restR{};
};
struct PhysicalSkinRegisteredSurfaceCSVFiles {
    std::array<PhysicalSkinRegisteredLayerCSV,3> layers; // O,I,D
    std::filesystem::path nodeRegistration,donorFrames,outerFaceAncestry;
    std::string nodeRegistrationSHA256,donorFramesSHA256,outerFaceAncestrySHA256;
    std::size_t donorCount=0,d37InnerFaceCount=0,d37DeepFaceCount=0;
    double maxParentRegistrationDistanceM=0.02;
    // Populated only by the live native owner, never by a CSV field. Every
    // admitted F32 frame must exactly match this complete donor-indexed table.
    std::span<const PhysicalSkinDonorFrameRow> nativeFloat32FramesByDonor{};
};
struct PhysicalSkinNodeSourceRow {
    std::int64_t vertex=-1,face=-1;
    std::array<double,3> bary{};
    V3 residual{};
    std::string domain,frameMethod;
};
struct PhysicalSkinParentRegistrationRow {
    bool available=false;
    std::uint32_t tet=NM_INVALID_INDEX,tieCount=0;
    std::array<double,4> bary{};
    V3 residual{};
    double distance=0;
    bool inferredCurrentParent=false;
};
struct PhysicalSkinFaceSourceRow {
    std::uint32_t sourceFace=NM_INVALID_INDEX,parentTet=NM_INVALID_INDEX,material=NM_INVALID_INDEX;
    std::string role;
};

struct PhysicalSkinRegisteredSurfaceCSVSet {
    std::array<RegisteredSurface,3> surfaces;
    std::array<std::vector<PhysicalSkinNodeSourceRow>,3> nodeSource;
    std::array<std::vector<PhysicalSkinParentRegistrationRow>,3> nodeParent;
    std::array<std::vector<PhysicalSkinFaceSourceRow>,3> faceSource;
    std::vector<PhysicalSkinDonorFrameRow> framesByDonor;
    std::vector<std::uint64_t> oFaceOffsets;
    std::vector<std::int64_t> oFaceParentRows,oFaceOriginalIDs,oFaceSourceRows;
    std::vector<std::string> oFaceRoles;
    std::string payloadSHA256;
};
struct PhysicalSkinRegisteredSurfaceCSVReport {
    std::array<std::size_t,3> nodes{},faces{},parentMappedNodes{},parentUnmappedNodes{};
    std::array<std::size_t,3> directSourceNodes{},faceEmbeddedNodes{};
    std::size_t donorFrames=0,oFaceAncestryRows=0;
    std::string payloadSHA256;
};

namespace physical_skin_registered_surface_csv_detail {
using namespace physical_skin_child_mass_detail;
inline bool header(std::ifstream& s,const std::filesystem::path& p,
                   const std::vector<std::string>& expected,std::string& e) {
    std::string line;if(!std::getline(s,line)||splitCSV(line)!=expected){
        e="unexpected registered surface CSV header: "+p.string();return false;}return true;
}
inline bool number(const std::string& s,double& x){return parseNumber(s,x)&&std::isfinite(x);}
inline bool optInt(const std::string& s,std::int64_t& x){if(s.empty()){x=-1;return true;}return parseNumber(s,x);}
inline std::array<double,9> tr(const std::array<double,9>& a){return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};}
inline V3 mul(const std::array<double,9>& a,const V3& x){return {
 a[0]*x[0]+a[1]*x[1]+a[2]*x[2],a[3]*x[0]+a[4]*x[1]+a[5]*x[2],a[6]*x[0]+a[7]*x[1]+a[8]*x[2]};}
inline double det(const std::array<double,9>& m){return m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);}
inline bool rotation(const std::array<double,9>& m){
 for(double x:m)if(!std::isfinite(x))return false;
 for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j){double v=0;for(unsigned k=0;k<3;++k)v+=m[3*k+i]*m[3*k+j];
  if(std::abs(v-(i==j?1.:0.))>2e-7)return false;}
 return std::abs(det(m)-1.)<=2e-7;
}
// rotatePoint constructs the native matrix in Float32. Casting its entries to
// double does not restore exact orthogonality. Bound that arithmetic error only
// for trusted native frames; the default external rotation contract stays strict.
inline bool nativeFloat32Rotation(const std::array<double,9>& m){
 // Stay inside the downstream physical owner's existing 1e-6 bound.
 constexpr double bound=8.0*std::numeric_limits<float>::epsilon();
 for(double x:m)if(!std::isfinite(x)||double(float(x))!=x)return false;
 for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j){double v=0;
  for(unsigned k=0;k<3;++k)v+=m[3*k+i]*m[3*k+j];
  if(std::abs(v-(i==j?1.:0.))>bound)return false;}
 return std::abs(det(m)-1.)<=bound;
}
inline bool exactNativeFrame(const PhysicalSkinDonorFrameRow& r,
                             const PhysicalSkinDonorFrameRow& native){
 return native.available&&r.binding==native.binding&&
  r.currentOrigin==native.currentOrigin&&r.restOrigin==native.restOrigin&&
  r.currentR==native.currentR&&r.restR==native.restR&&
  nativeFloat32Rotation(r.currentR)&&nativeFloat32Rotation(r.restR);
}
// The trusted F32 matrix is near SO3, not exactly orthogonal. Transpose is
// therefore not its inverse. Keep local-coordinate reconstruction in double.
inline std::array<double,9> worldToLocal(const std::array<double,9>& m,
                                       PhysicalSkinPoseArithmetic arithmetic){
 if(arithmetic!=PhysicalSkinPoseArithmetic::nativeFloat32)return tr(m);
 const double d=det(m);
 return {(m[4]*m[8]-m[5]*m[7])/d,(m[2]*m[7]-m[1]*m[8])/d,(m[1]*m[5]-m[2]*m[4])/d,
         (m[5]*m[6]-m[3]*m[8])/d,(m[0]*m[8]-m[2]*m[6])/d,(m[2]*m[3]-m[0]*m[5])/d,
         (m[3]*m[7]-m[4]*m[6])/d,(m[1]*m[6]-m[0]*m[7])/d,(m[0]*m[4]-m[1]*m[3])/d};
}
inline bool sameWorldToLocal(const std::array<double,9>& m,const std::array<double,9>& expected,
                             PhysicalSkinPoseArithmetic arithmetic){
 if(arithmetic!=PhysicalSkinPoseArithmetic::nativeFloat32)return m==expected;
 // Only last-bit differences between F64 inverse implementations are allowed.
 constexpr double tolerance=8.0*std::numeric_limits<double>::epsilon();
 for(unsigned i=0;i<9;++i)if(!std::isfinite(m[i])||
    std::abs(m[i]-expected[i])>tolerance*std::max(1.0,std::abs(expected[i])))return false;
 return true;
}
inline bool hash(const std::filesystem::path& p,std::string_view expected,std::string& got,std::string& e){
 if(!validSHA256(expected)){e="missing or invalid pinned SHA-256: "+p.string();return false;}
 if(!sha256File(p,got,e))return false;if(got!=expected){e="registered sidecar hash mismatch: "+p.string()+" got="+got+" expected="+std::string(expected);return false;}return true;
}
inline RegisteredSurface identity(const PhysicalSkinRegisteredLayerCSV& f,unsigned l,
                                  std::string_view nodeSHA,std::string_view payload){
 RegisteredSurface s;s.name=f.name;s.provider="NumiHumanRegisteredSurfaceSidecarCSV";
 s.sourceMember=f.sourceMember;s.sourceMeshSHA256=nodeSHA;s.sourcePayloadSHA256=payload;
 s.stableIdentifier=f.stableIdentifier;s.sourceBodyIndex=NM_INVALID_INDEX;
 s.sourceLayer=l;s.aggregateIdentity=true;return s;
}
inline bool readFrames(const PhysicalSkinRegisteredSurfaceCSVFiles& f,PhysicalSkinRegisteredSurfaceCSVSet& out,
                       PhysicalSkinRegisteredSurfaceCSVReport& report,std::string& e){
 std::ifstream in(f.donorFrames);if(!in){e="cannot open donor frame CSV";return false;}
 std::vector<std::string> h{"binding","donor","current_x","current_y","current_z"};
 for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)h.push_back("current_R"+std::to_string(i)+std::to_string(j));
 h.insert(h.end(),{"rest_x","rest_y","rest_z"});
 for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)h.push_back("rest_R"+std::to_string(i)+std::to_string(j));
 if(!header(in,f.donorFrames,h,e))return false;
 if(!f.nativeFloat32FramesByDonor.empty()&&f.nativeFloat32FramesByDonor.size()!=f.donorCount){
  e="trusted native frame table has wrong donor domain";return false;}
 out.framesByDonor.assign(f.donorCount,{});
 std::vector<bool> bindings(4096,false),donors(f.donorCount,false);
 std::string line;std::size_t n=0;
 while(std::getline(in,line)){
  if(line.size()>4096||n>=4096){e="donor frame CSV exceeds bounded rows";return false;}
  auto v=splitCSV(line);std::uint32_t bind=0,donor=0;
  if(v.size()!=h.size()||!parseNumber(v[0],bind)||!parseNumber(v[1],donor)||bind>=bindings.size()||
     donor>=f.donorCount||bindings[bind]||donors[donor]){e="invalid donor-frame identity or row width";return false;}
  PhysicalSkinDonorFrameRow r;r.available=true;r.binding=bind;
  for(unsigned k=0;k<3;++k)if(!number(v[2+k],r.currentOrigin[k])||!number(v[14+k],r.restOrigin[k])){
   e="nonfinite donor origin";return false;}
  for(unsigned k=0;k<9;++k)if(!number(v[5+k],r.currentR[k])||!number(v[17+k],r.restR[k])){
   e="nonfinite donor rotation";return false;}
  if(f.nativeFloat32FramesByDonor.empty()){
   if(!rotation(r.currentR)||!rotation(r.restR)){e="donor frame is not a proper rotation";return false;}
  }else if(!exactNativeFrame(r,f.nativeFloat32FramesByDonor[donor])){
   const auto& pin=f.nativeFloat32FramesByDonor[donor];
   e="donor frame differs from trusted native Float32 frame or exceeds its arithmetic bound: donor="+
     std::to_string(donor)+" binding="+std::to_string(bind)+
     " current_origin_equal="+std::to_string(r.currentOrigin==pin.currentOrigin)+
     " rest_origin_equal="+std::to_string(r.restOrigin==pin.restOrigin)+
     " current_matrix_equal="+std::to_string(r.currentR==pin.currentR)+
     " rest_matrix_equal="+std::to_string(r.restR==pin.restR)+
     " bounded="+std::to_string(nativeFloat32Rotation(r.currentR)&&nativeFloat32Rotation(r.restR));
   return false;
  }
  if(!f.nativeFloat32FramesByDonor.empty())r.arithmetic=PhysicalSkinPoseArithmetic::nativeFloat32;
  bindings[bind]=donors[donor]=true;out.framesByDonor[donor]=r;++n;
 }
 if(!in.eof()||n==0){e="donor-frame CSV is incomplete";return false;}report.donorFrames=n;return true;
}
inline bool readID(const PhysicalSkinRegisteredLayerCSV& file,unsigned layer,const std::size_t donors,
                   const std::vector<PhysicalSkinDonorFrameRow>& frames,std::string_view payload,
                   RegisteredSurface& s,std::vector<PhysicalSkinNodeSourceRow>& nodeSource,
                   std::vector<PhysicalSkinFaceSourceRow>& faceSource,
                   std::string& ns,std::string& fs,std::string& e){
 const auto fail=[&](std::string x){e=std::move(x);return false;};
 if(layer!=1&&layer!=2)return fail("I/D reader called with non-I/D role");
 if(!hash(file.nodes,file.nodesSHA256,ns,e)||!hash(file.faces,file.facesSHA256,fs,e))return false;
 std::ifstream ni(file.nodes),fi(file.faces);if(!ni||!fi)return fail("cannot open I/D CSV tables");
 std::vector<std::string> nh{"vertex","donor","wx","wy","wz","nx","ny","nz","lx","ly","lz",
  "m00","m01","m02","m10","m11","m12","m20","m21","m22","source_domain",
  "d37_parent_node_row","d37_parent_surface_face_row","beta0","beta1","beta2",
  "source_residual_x_m","source_residual_y_m","source_residual_z_m","frame_owner_method"};
 std::vector<std::string> fh{"face","v0","v1","v2","d37_parent_surface_face_row","d37_parent_tet_row","material_index","source_role"};
 if(!header(ni,file.nodes,nh,e)||!header(fi,file.faces,fh,e))return false;
 const unsigned expectedMaterial=layer==1?15:16;
 s=identity(file,layer,ns,payload);std::string line;std::size_t n=0;
 while(std::getline(ni,line)){
  if(line.size()>8192||n>=300000)return fail("I/D node CSV exceeds row bound");
  auto v=splitCSV(line);std::uint32_t id=0,donor=0;std::int64_t sv=-1,sf=-1;
  if(v.size()!=nh.size()||!parseNumber(v[0],id)||id!=n||!parseNumber(v[1],donor)||donor>=donors||
     !optInt(v[21],sv)||!optInt(v[22],sf)||(sv>=0)==(sf>=0)||v[20]!="D37_coarse_parent"||v[29].empty())
   return fail("I/D node has invalid row, frame owner, or source identity");
  V3 p{},normal{},local{},sr{};std::array<double,9> w2l{};
  for(unsigned a=0;a<3;++a)if(!number(v[2+a],p[a])||double(float(p[a]))!=p[a]||
      !number(v[5+a],normal[a])||!number(v[8+a],local[a])||!number(v[26+a],sr[a]))
   return fail("I/D node has invalid F32 world point, normal, local point, or residual");
  for(unsigned a=0;a<9;++a)if(!number(v[11+a],w2l[a]))return fail("I/D node transform is nonfinite");
  if(std::abs(norm(normal)-1.)>1e-5||donor>=frames.size()||!frames[donor].available)
   return fail("I/D node has invalid normal or missing donor frame");
  auto expected=worldToLocal(frames[donor].currentR,frames[donor].arithmetic);
  for(unsigned a=0;a<9;++a)if(std::abs(w2l[a]-expected[a])>2e-10)
   return fail("I/D node matrix differs from exact donor-frame table");
  if(norm(sub(mul(w2l,sub(p,frames[donor].currentOrigin)),local))>2e-8)
   return fail("I/D donor-local point does not reconstruct from donor frame");
  PhysicalSkinNodeSourceRow a;a.vertex=sv;a.face=sf;a.domain=v[20];a.frameMethod=v[29];a.residual=sr;
  for(unsigned k=0;k<3;++k)if(!number(v[23+k],a.bary[k]))return fail("I/D source barycentric is nonfinite");
  if(sf>=0){
   if(!validateBarycentricCoordinates(a.bary,e)){
    e="I/D node row "+std::to_string(n)+" source barycentric fails native tolerance: "+e;return false;
   }
  }else if(std::any_of(a.bary.begin(),a.bary.end(),[](double x){return !std::isfinite(x);}))
   return fail("direct I/D node has invalid barycentric placeholders");
  s.worldPoints.push_back(p);s.outwardWorldNormals.push_back(normal);s.donorLocalPoints.push_back(local);
  s.donorBodyIndices.push_back(donor);s.worldToDonorLocalLinear.push_back(w2l);
  s.sourceVertexIndices.push_back(sv>=0?static_cast<std::uint32_t>(sv):NM_INVALID_INDEX);
  s.sourceEmbeddingFaceIndices.push_back(sf>=0?static_cast<std::uint32_t>(sf):NM_INVALID_INDEX);
  s.sourceEmbeddingBarycentric.push_back(a.bary);s.sourceEmbeddingResidualWorldM.push_back(sr);
  nodeSource.push_back(std::move(a));++n;
 }
 if(!ni.eof()||n==0)return fail("I/D node CSV is incomplete");
 std::size_t m=0;
 while(std::getline(fi,line)){
  if(line.size()>4096||m>=600000)return fail("I/D face CSV exceeds row bound");
  auto v=splitCSV(line);std::uint32_t id=0,sf=0,pt=0,mat=0;Face tri{};
  if(v.size()!=fh.size()||!parseNumber(v[0],id)||id!=m||!parseNumber(v[4],sf)||
     !parseNumber(v[5],pt)||!parseNumber(v[6],mat)||mat!=expectedMaterial||v[7].empty())
   return fail("I/D face has invalid row, ancestry, or material");
  for(unsigned a=0;a<3;++a)if(!parseNumber(v[1+a],tri[a])||tri[a]>=n)return fail("I/D face corner out of domain");
  if(tri[0]==tri[1]||tri[1]==tri[2]||tri[2]==tri[0]||
     norm(cross(sub(s.worldPoints[tri[1]],s.worldPoints[tri[0]]),
                sub(s.worldPoints[tri[2]],s.worldPoints[tri[0]])))<=0)
   return fail("I/D face is degenerate");
  s.faces.push_back(tri);s.sourceFaceIndices.push_back(sf);faceSource.push_back({sf,pt,mat,v[7]});++m;
 }
 if(!fi.eof()||m==0)return fail("I/D face CSV is incomplete");
 return true;
}
inline bool readOuterSourceRows(const PhysicalSkinRegisteredLayerCSV& file,
                                std::vector<PhysicalSkinNodeSourceRow>& rows,
                                PhysicalSkinRegisteredSurfaceCSVReport& rep,std::string& e){
 std::ifstream in(file.nodes);if(!in){e="cannot reopen O registered node CSV";return false;}
 std::string line;if(!std::getline(in,line)){e="O registered node CSV has no header";return false;}
 auto h=splitCSV(line);std::map<std::string,std::size_t> col;
 for(std::size_t i=0;i<h.size();++i)if(!col.emplace(h[i],i).second){e="O registered node CSV repeats a field";return false;}
 for(const auto* key:{"vertex","source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2",
                      "source_residual_x_m","source_residual_y_m","source_residual_z_m","source_domain","frame_owner_method"})
  if(!col.contains(key)){e="O registered node CSV lacks source ancestry fields";return false;}
 std::size_t row=0;
 while(std::getline(in,line)){
  if(line.size()>16384||row>=300000){e="O registered node CSV exceeds bounded rows";return false;}
  auto v=splitCSV(line);std::uint32_t id=0;std::int64_t sv=-1,sf=-1;
  if(v.size()!=h.size()||!parseNumber(v[col["vertex"]],id)||id!=row||
     !optInt(v[col["source_vertex_row"]],sv)||!optInt(v[col["source_face_row"]],sf)||
     (sv>=0)==(sf>=0)){e="O registered node source ancestry is invalid";return false;}
  PhysicalSkinNodeSourceRow a;a.vertex=sv;a.face=sf;
  a.domain=v[col["source_domain"]];a.frameMethod=v[col["frame_owner_method"]];
  if(a.domain.empty()||a.frameMethod.empty()){e="O source/frame provenance is empty";return false;}
  for(unsigned k=0;k<3;++k)
   if(!number(v[col["source_bary"+std::to_string(k)]],a.bary[k])||
      !number(v[col["source_residual_"+std::string(1,"xyz"[k])+"_m"]],a.residual[k])){
    e="O source barycentric/residual is nonfinite";return false;}
  if(sf>=0){if(!validateBarycentricCoordinates(a.bary,e))return false;++rep.faceEmbeddedNodes[0];}
  else ++rep.directSourceNodes[0];
  rows.push_back(std::move(a));++row;
 }
 if(!in.eof()||row==0){e="O source ancestry rows are incomplete";return false;}
 return true;
}
inline bool readRegistrations(const PhysicalSkinRegisteredSurfaceCSVFiles& f,
                              PhysicalSkinRegisteredSurfaceCSVSet& out,
                              PhysicalSkinRegisteredSurfaceCSVReport& rep,std::string& e){
 std::ifstream in(f.nodeRegistration);if(!in){e="cannot open node-parent-registration CSV";return false;}
 const std::vector<std::string> h{"layer","surface_vertex_row","parent_registration_kind","parent_tet_row",
 "parent_bary0","parent_bary1","parent_bary2","parent_bary3","parent_residual_x_m","parent_residual_y_m",
 "parent_residual_z_m","parent_registration_distance_m","parent_registration_tie_count","frame_owner_method",
 "source_domain","source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2",
 "source_residual_x_m","source_residual_y_m","source_residual_z_m"};
 if(!header(in,f.nodeRegistration,h,e))return false;
 std::array<std::vector<bool>,3> seen;
 for(unsigned l=0;l<3;++l){auto n=out.surfaces[l].worldPoints.size();seen[l].assign(n,false);out.nodeParent[l].resize(n);}
 std::string line;std::size_t total=0;
 while(std::getline(in,line)){
  if(line.size()>8192||total>=900000){e="node registration CSV exceeds row bound";return false;}
  auto v=splitCSV(line);unsigned l=3;for(unsigned k=0;k<3;++k)if(v.size()&&v[0]==std::array<std::string_view,3>{"O","I","D"}[k])l=k;
  std::uint32_t row=0;std::int64_t pt=-1,tie=-1,sv=-1,sf=-1;
  if(v.size()!=h.size()||l>=3||!parseNumber(v[1],row)||row>=seen[l].size()||seen[l][row]||
     !optInt(v[3],pt)||!optInt(v[12],tie)||!optInt(v[15],sv)||!optInt(v[16],sf)){
   e="node registration row identity is invalid";return false;}
  seen[l][row]=true;const auto& src=out.nodeSource[l][row];
  if(src.vertex!=sv||src.face!=sf||src.domain!=v[14]||src.frameMethod!=v[13]){
   e="node registration source identity differs from registered surface row";return false;}
  for(unsigned a=0;a<3;++a){double b=0,r=0;if(!number(v[17+a],b)||!number(v[20+a],r)||
      b!=src.bary[a]||r!=src.residual[a]){e="node registration source embedding differs from surface sidecar";return false;}}
  auto& r=out.nodeParent[l][row];r.available=pt>=0;
  if(r.available){
   r.inferredCurrentParent=v[2]=="inferred_current_parent_tet_same_material_donor_region";
   if(!parseNumber(v[3],r.tet)||tie<0||tie>std::numeric_limits<std::uint32_t>::max()||
      (v[2]!="source_parent_tet_reference_registration"&&!r.inferredCurrentParent)){
    e="node parent registration has invalid cell or method";return false;}
   double sum=0;for(unsigned a=0;a<4;++a){if(!number(v[4+a],r.bary[a])||r.bary[a]<-1e-8||r.bary[a]>1+1e-8){
      e="node parent barycentric is out of range";return false;}sum+=r.bary[a];}
   for(unsigned a=0;a<3;++a)if(!number(v[8+a],r.residual[a])){e="node parent residual is nonfinite";return false;}
   if(!number(v[11],r.distance)||r.distance<0||r.distance>f.maxParentRegistrationDistanceM||
      std::abs(sum-1)>1e-8||std::abs(norm(r.residual)-r.distance)>2e-10){
    e="node parent registration violates residual, distance, or barycentric bounds";return false;}
   r.tieCount=static_cast<std::uint32_t>(tie);++rep.parentMappedNodes[l];
  }else{
   if(l!=0||v[2]!="unavailable_ambiguous_outer_reference_registration"||pt!=-1||tie!=-1||
      std::any_of(v.begin()+4,v.begin()+12,[](const auto& x){return !x.empty();})){
    e="only explicitly unavailable O parent registration may have blank cell fields";return false;}
   ++rep.parentUnmappedNodes[l];
  }
  ++total;
 }
 if(!in.eof()){e="node registration CSV is unreadable";return false;}
 for(unsigned l=0;l<3;++l)if(std::find(seen[l].begin(),seen[l].end(),false)!=seen[l].end()){
  e="node registration CSV does not cover every O/I/D row";return false;}
 return true;
}
inline bool readOuterAncestry(const PhysicalSkinRegisteredSurfaceCSVFiles& f,
                              PhysicalSkinRegisteredSurfaceCSVSet& out,
                              PhysicalSkinRegisteredSurfaceCSVReport& rep,std::string& e){
 std::ifstream in(f.outerFaceAncestry);if(!in){e="cannot open O face ancestry";return false;}
 const std::vector<std::string> h{"output_face_row","parent_face_row","source_original_id","source_face_row","ancestry_role"};
 if(!header(in,f.outerFaceAncestry,h,e))return false;
 auto n=out.surfaces[0].faces.size();out.oFaceOffsets.assign(n+1,0);
 std::string line;std::size_t rows=0;std::uint32_t last=0;
 while(std::getline(in,line)){
  if(line.size()>4096||rows>=2000000){e="O face ancestry exceeds row bound";return false;}
  auto v=splitCSV(line);std::uint32_t id=0;std::int64_t p=-1,orig=-1,src=-1;
  if(v.size()!=5||!parseNumber(v[0],id)||id>=n||!optInt(v[1],p)||!optInt(v[2],orig)||
     !optInt(v[3],src)||v[4].empty()||(rows&&id<last)){e="O face ancestry row is malformed/unordered";return false;}
  last=id;out.oFaceParentRows.push_back(p);out.oFaceOriginalIDs.push_back(orig);
  out.oFaceSourceRows.push_back(src);out.oFaceRoles.push_back(v[4]);++out.oFaceOffsets[id+1];++rows;
 }
 if(!in.eof()){e="O face ancestry unreadable";return false;}
 for(std::size_t i=1;i<out.oFaceOffsets.size();++i)out.oFaceOffsets[i]+=out.oFaceOffsets[i-1];
 rep.oFaceAncestryRows=rows;return true;
}
inline bool validateFrames(const RegisteredSurface& s,const std::vector<PhysicalSkinDonorFrameRow>& frames,std::string& e){
 if(s.worldPoints.size()!=s.donorBodyIndices.size()||s.worldPoints.size()!=s.donorLocalPoints.size()||
    s.worldPoints.size()!=s.worldToDonorLocalLinear.size()){e="RegisteredSurface donor rows incomplete";return false;}
 for(std::size_t i=0;i<s.worldPoints.size();++i){
  auto d=s.donorBodyIndices[i];if(d>=frames.size()||!frames[d].available){e="surface donor frame missing";return false;}
  auto m=worldToLocal(frames[d].currentR,frames[d].arithmetic);for(unsigned k=0;k<9;++k)if(std::abs(m[k]-s.worldToDonorLocalLinear[i][k])>2e-10){
   e="surface donor matrix differs from frame table";return false;}
  if(norm(sub(mul(m,sub(s.worldPoints[i],frames[d].currentOrigin)),s.donorLocalPoints[i]))>2e-8){
   e="surface donor-local point does not reconstruct";return false;}
 }return true;
}
} // namespace physical_skin_registered_surface_csv_detail

[[nodiscard]] inline bool loadPhysicalSkinRegisteredSurfaceCSVSet(
 const PhysicalSkinRegisteredSurfaceCSVFiles& f,PhysicalSkinRegisteredSurfaceCSVSet& output,
 PhysicalSkinRegisteredSurfaceCSVReport& report,std::string& error){
 using namespace physical_skin_registered_surface_csv_detail;
 const auto fail=[&](std::string x){error=std::move(x);return false;};
 if(!f.donorCount||f.donorCount>4096||!std::isfinite(f.maxParentRegistrationDistanceM)||f.maxParentRegistrationDistanceM<=0)
  return fail("registered sidecar donor domain or parent-registration bound is invalid");
 for(const auto& l:f.layers)if(l.nodes.empty()||l.faces.empty()||l.name.empty()||l.sourceMember.empty()||
    !validSHA256(l.nodesSHA256)||!validSHA256(l.facesSHA256))return fail("registered O/I/D files lack paths, identity, or SHA pins");
 if(f.nodeRegistration.empty()||f.donorFrames.empty()||f.outerFaceAncestry.empty()||
    !validSHA256(f.nodeRegistrationSHA256)||!validSHA256(f.donorFramesSHA256)||!validSHA256(f.outerFaceAncestrySHA256))
  return fail("registered frame, node registration, and O ancestry must be explicitly hash-pinned");
 PhysicalSkinRegisteredSurfaceCSVSet next;PhysicalSkinRegisteredSurfaceCSVReport measured;
 std::array<std::string,9> hashes{};
 for(unsigned l=0;l<3;++l)if(!hash(f.layers[l].nodes,f.layers[l].nodesSHA256,hashes[2*l],error)||
    !hash(f.layers[l].faces,f.layers[l].facesSHA256,hashes[2*l+1],error))return false;
 if(!hash(f.nodeRegistration,f.nodeRegistrationSHA256,hashes[6],error)||
    !hash(f.donorFrames,f.donorFramesSHA256,hashes[7],error)||
    !hash(f.outerFaceAncestry,f.outerFaceAncestrySHA256,hashes[8],error))return false;
 spatial_mass_transport_detail::SHA256Builder payload;payload.string("registered-OID-sidecar-set.v1");
 for(const auto& h:hashes)payload.string(h);next.payloadSHA256=payload.finish();
 if(!readFrames(f,next,measured,error))return false;
 for(unsigned l=0;l<3;++l){
  const auto& in=f.layers[l];
  if(l==0){
   auto id=identity(in,l,hashes[0],next.payloadSHA256);RegisteredSkinCandidateCSVReport parsed;
   const auto hashPath=[](const std::filesystem::path& path){
    std::string digest,hashError;
    return sha256File(path,digest,hashError)?digest:std::string{};
   };
   if(!loadRegisteredSkinCandidateCSV(in.nodes,in.faces,f.donorCount,id,in.facesSHA256,hashPath,
      next.surfaces[0],parsed,error))return false;
   measured.nodes[0]=parsed.nodeRows;measured.faces[0]=parsed.faceRows;
   if(!readOuterSourceRows(in,next.nodeSource[0],measured,error))return false;
   if(next.nodeSource[0].size()!=parsed.nodeRows)return fail("O source ancestry row count differs from registered surface");
  }else{
   std::string ns,fs;if(!readID(in,l,f.donorCount,next.framesByDonor,next.payloadSHA256,
      next.surfaces[l],next.nodeSource[l],next.faceSource[l],ns,fs,error))return false;
   measured.nodes[l]=next.surfaces[l].worldPoints.size();measured.faces[l]=next.surfaces[l].faces.size();
   const auto sfcount=l==1?f.d37InnerFaceCount:f.d37DeepFaceCount;
   if(!validateRegisteredSourceEmbedding(next.surfaces[l],sfcount,error))return false;
   for(const auto& a:next.nodeSource[l])if(a.vertex>=0)++measured.directSourceNodes[l];else ++measured.faceEmbeddedNodes[l];
   for(const auto& a:next.faceSource[l])if(a.material!=(l==1?15u:16u)||a.parentTet==NM_INVALID_INDEX||
       (sfcount&&a.sourceFace>=sfcount))return fail("I/D face ancestry is outside declared parent/material domain");
  }
 }
 if(!validateFrames(next.surfaces[0],next.framesByDonor,error)||
    !validateFrames(next.surfaces[1],next.framesByDonor,error)||
    !validateFrames(next.surfaces[2],next.framesByDonor,error))return false;
 if(!readRegistrations(f,next,measured,error)||!readOuterAncestry(f,next,measured,error))return false;
 if(next.oFaceOffsets.size()!=next.surfaces[0].faces.size()+1)return fail("O face ancestry does not cover O faces");
 // The O node row source ancestry is preserved in nodeParent rows only after
 // the strict registration cross-check. Its parent-cell map may be unavailable.
 measured.nodes[0]=next.surfaces[0].worldPoints.size();measured.faces[0]=next.surfaces[0].faces.size();
 for(unsigned l=0;l<3;++l)for(std::size_t r=0;r<next.nodeParent[l].size();++r){
  if(next.nodeParent[l][r].available)++measured.parentMappedNodes[l];else ++measured.parentUnmappedNodes[l];
 }
 // readRegistrations increments the same counters; avoid double-counting.
 for(unsigned l=0;l<3;++l){
  measured.parentMappedNodes[l]=0;measured.parentUnmappedNodes[l]=0;
  for(const auto& row:next.nodeParent[l])(row.available?measured.parentMappedNodes[l]:measured.parentUnmappedNodes[l])++;
 }
 measured.payloadSHA256=next.payloadSHA256;output=std::move(next);report=std::move(measured);error.clear();return true;
}


// Import-time physical check, separate from parsing. A complete, numerically
// bounded stencil must reconstruct the actual registered point in the pinned
// current parent table; matching row counts and hashes alone cannot prove that.
struct PhysicalSkinSurfaceParentRegistrationReport {
    std::array<std::size_t,3> checkedNodes{};
    std::array<double,3> maximumReconstructionErrorM{};
    std::array<double,3> maximumRegistrationDistanceM{};
};

[[nodiscard]] inline bool validatePhysicalSkinSurfaceParentRegistration(
        const PhysicalSkinRegisteredSurfaceCSVSet& surfaces,
        const PhysicalSkinParentMassPacket& parent,
        PhysicalSkinSurfaceParentRegistrationReport& report,
        std::string& error) {
    using namespace physical_skin_child_mass_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(parent.currentWorldPoints.empty()||parent.tetrahedra.empty()||
       parent.materialIndices.size()!=parent.tetrahedra.size())
        return fail("physical skin registration requires a complete current parent table");
    PhysicalSkinSurfaceParentRegistrationReport next;
    for(unsigned layer=0;layer<3;++layer){
        const auto& s=surfaces.surfaces[layer];const auto& rows=surfaces.nodeParent[layer];
        if(s.worldPoints.empty()||rows.size()!=s.worldPoints.size())
            return fail("physical skin registration does not cover every surface node");
        for(std::size_t node=0;node<rows.size();++node){
            const auto& row=rows[node];
            const auto label=[&](std::string_view why){
                return "physical skin layer "+std::to_string(layer)+" node "+
                    std::to_string(node)+" "+std::string(why);};
            if(!row.available||row.tet>=parent.tetrahedra.size())
                return fail(label("has no valid parent-cell registration"));
            const auto material=parent.materialIndices[row.tet];
            if((layer==0&&material!=15u)||(layer==1&&material!=15u&&material!=16u)||
               (layer==2&&material!=16u))
                return fail(label("parent material does not match its skin/fat role"));
            // This narrow inference is valid only when the surface donor is
            // itself the anatomical region. Other geometric/anatomical label
            // combinations require an explicit separately typed region map.
            if(row.inferredCurrentParent&&
               (s.donorBodyIndices.size()!=s.worldPoints.size()||
                parent.donorBodyIndices.size()!=parent.tetrahedra.size()||
                parent.anatomicalRegionIndices.size()!=parent.tetrahedra.size()||
                parent.donorBodyIndices[row.tet]!=s.donorBodyIndices[node]||
                parent.anatomicalRegionIndices[row.tet]!=s.donorBodyIndices[node]))
                return fail(label("inferred current parent violates its explicit same-donor-region restriction"));
            V3 reconstructed=row.residual;double sum=0.;
            for(unsigned c=0;c<4;++c){
                const auto v=parent.tetrahedra[row.tet][c];const auto beta=row.bary[c];
                if(v>=parent.currentWorldPoints.size()||!std::isfinite(beta)||beta<-1e-8||beta>1.+1e-8)
                    return fail(label("has an invalid parent corner or barycentric coefficient"));
                sum+=beta;
                for(unsigned a=0;a<3;++a){
                    const auto value=parent.currentWorldPoints[v][a];
                    if(!std::isfinite(value)||!std::isfinite(s.worldPoints[node][a])||!std::isfinite(row.residual[a]))
                        return fail(label("has a nonfinite current point or residual"));
                    reconstructed[a]+=beta*value;
                }
            }
            const auto distance=norm(row.residual);
            if(std::abs(sum-1.)>1e-8||!std::isfinite(row.distance)||row.distance<0.||
               row.distance>0.02||std::abs(distance-row.distance)>2e-10)
                return fail(label("violates the existing physical parent-distance or barycentric bounds"));
            const auto delta=norm(sub(reconstructed,s.worldPoints[node]));
            if(!std::isfinite(delta)||delta>2e-8){
                std::ostringstream message;message<<label("does not reconstruct from its current parent: error_m=")<<delta;
                return fail(message.str());
            }
            ++next.checkedNodes[layer];
            next.maximumReconstructionErrorM[layer]=std::max(next.maximumReconstructionErrorM[layer],delta);
            next.maximumRegistrationDistanceM[layer]=std::max(next.maximumRegistrationDistanceM[layer],distance);
        }
    }
    report=std::move(next);error.clear();return true;
}


[[nodiscard]] inline bool loadPhysicalSkinRegisteredSurfaceCSVSetForParent(
        const PhysicalSkinRegisteredSurfaceCSVFiles& files,
        const PhysicalSkinParentMassPacket& parent,
        PhysicalSkinRegisteredSurfaceCSVSet& output,
        PhysicalSkinRegisteredSurfaceCSVReport& report,
        PhysicalSkinSurfaceParentRegistrationReport& registrationReport,
        std::string& error) {
    PhysicalSkinRegisteredSurfaceCSVSet next;
    PhysicalSkinRegisteredSurfaceCSVReport parsed;
    PhysicalSkinSurfaceParentRegistrationReport checked;
    if(!loadPhysicalSkinRegisteredSurfaceCSVSet(files,next,parsed,error)||
       !validatePhysicalSkinSurfaceParentRegistration(next,parent,checked,error))
        return false;
    output=std::move(next);report=std::move(parsed);registrationReport=std::move(checked);
    error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
