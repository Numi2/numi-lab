#pragma once
#include "metalrobo/NumiHumanTissueMass.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human_shared.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Inferred registered-skeleton ACL reference object. Mutually exclusive with
// Open Knee ACL segmentation; emits no passive ligament row or muscle route.
namespace metalrobo::inferred_acl {
using V3=std::array<double,3>;
struct BoneSurface {
 std::uint32_t body=MR_INVALID_INDEX,stableId=0,sourceRecord=MR_INVALID_INDEX;
 std::array<std::uint8_t,32> payloadSha256{};
 std::span<const V3> worldVertices;
 std::span<const V3> worldNormals;
 std::span<const std::uint32_t> triangleIndices;
};
struct Frame {
 std::uint32_t side=0; // 0 left, 1 right
 std::uint32_t femurBody=MR_INVALID_INDEX,tibiaBody=MR_INVALID_INDEX;
 std::uint32_t patellaBody=MR_INVALID_INDEX,patellaBoneStableId=0;
 std::span<const MRBodyStateGPU> neutralBodyPoses;
 V3 jointCenterWorld{},anteriorWorld{},superiorWorld{};
};
struct Provenance {
 std::string identifier="inferred_registered_skeleton_acl_v1";
 std::string className="inferred_reference_geometry";
 std::string materialSha256="c5166dfcdbcb8402cd107d087ac5071e548f8154b41f088bee3924e40ac9871e";
 std::uint32_t side=0,femurBody=MR_INVALID_INDEX,tibiaBody=MR_INVALID_INDEX;
 std::uint32_t patellaBody=MR_INVALID_INDEX,patellaBoneStableId=0;
 std::uint32_t femurBoneStableId=0,tibiaBoneStableId=0;
 std::uint32_t femurSourceRecord=MR_INVALID_INDEX,tibiaSourceRecord=MR_INVALID_INDEX;
 std::array<std::uint8_t,32> femurPayloadSha256{},tibiaPayloadSha256{};
 std::array<std::uint32_t,2> selectedSurfaceTriangles{};
 std::array<std::uint32_t,4> femurAnchorTriangles{},tibiaAnchorTriangles{};
 double maximumAttachmentProjectionResidualMeters=0;
 double maximumAttachmentFootprintRadiusMeters=0;
 double minimumInterboneFacingNormalDot=0;
 double minimumPatchNormalAlignmentDot=1;
 double inferredFootprintRadiusLimitMeters=.012;
 double inferredPatchNormalAngleLimitRadians=1.0471975511965976; // 60 degrees
 double maximumAttachmentProjectionResidualLimitMeters=.020;
 double minimumInterboneFacingNormalDotLimit=.10;
 double minimumPatchNormalAlignmentDotLimit=.50;
 double minimumAssumedLengthMeters=.012,maximumAssumedLengthMeters=.060;
 double assumedLengthMeters=0,assumedWidthMeters=.008,assumedThicknessMeters=.005;
 // Seed offsets are posterior, proximal, and lateral in metres. The captured
 // left candidate passed source-face footprint/span checks; right is a mirrored
 // lateral reference only and has not been independently checked.
 std::array<double,3> femurSeedOffsetsPosteriorProximalLateralMeters{}; // full combined offsets
 std::array<double,3> tibiaSeedOffsetsPosteriorProximalLateralMeters{}; // full combined offsets
 std::array<double,3> femurSeedBasePosteriorProximalLateralMeters{};
 std::array<double,3> femurSeedAdjustmentPosteriorProximalLateralMeters{};
 std::array<double,3> tibiaSeedBasePosteriorProximalLateralMeters{};
 std::array<double,3> tibiaSeedAdjustmentPosteriorProximalLateralMeters{};
 V3 seedPosteriorAxisWorld{},seedProximalAxisWorld{},seedLateralAxisWorld{};
 bool seedGeometryInferred=true,sourceCandidateValidatedForThisSide=false;
 double densityKgM3=1000,initialStretch=1;
 bool inferred=true,replacesExistingACLPassiveOwner=true,requiresOpenKneeACLDisabled=true;
 bool emitsPassiveLigamentRow=false,emitsMyoSimRoute=false,emitsMatterHumanAttachments=false,attachmentReactionsEqualAndOpposite=true;
};
struct BuiltObject {
 numi::matter::ObjectSource object;
 numi::matter::MaterialProgram material;
 std::vector<std::uint32_t> donorBodyByLocalNode;
 std::vector<V3> donorLocalPositionByLocalNode;
 std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors;
 std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads;
 Provenance provenance;
};
namespace detail {
inline V3 add(V3 a,V3 b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline V3 sub(V3 a,V3 b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline V3 mul(V3 a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
inline double dot(V3 a,V3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline V3 cross(V3 a,V3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double norm(V3 a){return std::sqrt(dot(a,a));}
inline V3 unit(V3 a,const char* m){double n=norm(a);if(!std::isfinite(n)||n<1e-9)throw std::runtime_error(m);return mul(a,1/n);}
inline V3 xyz(mr_float4 p){return {p.x,p.y,p.z};}
inline V3 rotate(mr_float4 q,V3 v){V3 u{q.x,q.y,q.z};V3 uv=cross(u,v),uuv=cross(u,uv);return add(v,add(mul(uv,2*q.w),mul(uuv,2)));}
inline V3 invRotate(mr_float4 q,V3 v){return rotate({-q.x,-q.y,-q.z,q.w},v);}
struct Projection{V3 p{},normal{};std::uint32_t tri=MR_INVALID_INDEX;double d2=std::numeric_limits<double>::infinity();};
inline std::array<double,3> barycentric(V3 q,V3 a,V3 b,V3 c){
 V3 v0=sub(b,a),v1=sub(c,a),v2=sub(q,a);
 const double d00=dot(v0,v0),d01=dot(v0,v1),d11=dot(v1,v1);
 const double d20=dot(v2,v0),d21=dot(v2,v1);
 const double den=d00*d11-d01*d01;
 if(!std::isfinite(den)||std::abs(den)<1e-24)
  throw std::runtime_error("ACL closest-point barycentric triangle is degenerate");
 const double v=(d11*d20-d01*d21)/den,w=(d00*d21-d01*d20)/den;
 return {1.0-v-w,v,w};
}
inline V3 validateTrianglePoint(V3 q,V3 a,V3 b,V3 c){
 const auto weights=barycentric(q,a,b,c);
 constexpr double tolerance=1e-10;
 if(!std::all_of(weights.begin(),weights.end(),[=](double x){
       return std::isfinite(x)&&x>=-tolerance&&x<=1.0+tolerance;
     }))
  throw std::runtime_error("ACL closest-point result is outside triangle barycentric domain");
 const V3 reconstructed=add(a,add(mul(sub(b,a),weights[1]),
                                  mul(sub(c,a),weights[2])));
 const double scale=std::max({1.0,norm(q),norm(a),norm(b),norm(c)});
 if(!std::isfinite(norm(sub(reconstructed,q)))||
    norm(sub(reconstructed,q))>1e-10*scale)
  throw std::runtime_error("ACL closest-point barycentric reconstruction failed");
 return q;
}
inline V3 closest(V3 p,V3 a,V3 b,V3 c){
 V3 ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);
 const double d1=dot(ab,ap),d2=dot(ac,ap);
 if(d1<=0.0&&d2<=0.0)return validateTrianglePoint(a,a,b,c);
 V3 bp=sub(p,b);
 const double d3=dot(ab,bp),d4=dot(ac,bp);
 if(d3>=0.0&&d4<=d3)return validateTrianglePoint(b,a,b,c);
 const double vc=d1*d4-d3*d2;
 if(vc<=0.0&&d1>=0.0&&d3<=0.0)
  return validateTrianglePoint(add(a,mul(ab,d1/(d1-d3))),a,b,c);
 V3 cp=sub(p,c);
 const double d5=dot(ab,cp),d6=dot(ac,cp);
 if(d6>=0.0&&d5<=d6)return validateTrianglePoint(c,a,b,c);
 const double vb=d5*d2-d1*d6;
 if(vb<=0.0&&d2>=0.0&&d6<=0.0)
  return validateTrianglePoint(add(a,mul(ac,d2/(d2-d6))),a,b,c);
 const double va=d3*d6-d5*d4;
 // Ericson's BC-edge region requires both d4-d3 and d5-d6 >= 0.
 if(va<=0.0&&d4>=d3&&d5>=d6)
  return validateTrianglePoint(
      add(b,mul(sub(c,b),(d4-d3)/((d4-d3)+(d5-d6)))),a,b,c);
 const double den=va+vb+vc;
 if(!std::isfinite(den)||den<=1e-20)
  throw std::runtime_error("ACL closest-point face-region denominator invalid");
 const double w1=vb/den,w2=vc/den,w0=1.0-w1-w2;
 constexpr double tolerance=1e-10;
 if(!std::isfinite(w0)||!std::isfinite(w1)||!std::isfinite(w2)||
    w0<-tolerance||w1<-tolerance||w2<-tolerance||
    w0>1.0+tolerance||w1>1.0+tolerance||w2>1.0+tolerance)
  throw std::runtime_error("ACL closest-point face-region barycentric invalid");
 return validateTrianglePoint(add(a,add(mul(ab,w1),mul(ac,w2))),a,b,c);
}
inline Projection project(V3 p,const BoneSurface& s){
 if(s.triangleIndices.empty()||s.triangleIndices.size()%3)throw std::runtime_error("ACL surface has no triangles");
 if(s.worldNormals.size()!=s.worldVertices.size())throw std::runtime_error("ACL source surface normals are absent or misaligned");
 Projection best;for(std::size_t i=0;i<s.triangleIndices.size();i+=3){auto ia=s.triangleIndices[i],ib=s.triangleIndices[i+1],ic=s.triangleIndices[i+2];
  if(ia>=s.worldVertices.size()||ib>=s.worldVertices.size()||ic>=s.worldVertices.size())throw std::runtime_error("ACL source triangle index out of range");
  V3 a=s.worldVertices[ia],b=s.worldVertices[ib],c=s.worldVertices[ic];if(norm(cross(sub(b,a),sub(c,a)))<1e-14)continue;
  V3 q=closest(p,a,b,c);double d=dot(sub(q,p),sub(q,p));if(d<best.d2){
   V3 v0=sub(b,a),v1=sub(c,a),v2=sub(q,a);double d00=dot(v0,v0),d01=dot(v0,v1),d11=dot(v1,v1),d20=dot(v2,v0),d21=dot(v2,v1);
   double den=d00*d11-d01*d01;if(!std::isfinite(den)||std::abs(den)<1e-24)continue;
   double bv=(d11*d20-d01*d21)/den,cw=(d00*d21-d01*d20)/den,aw=1-bv-cw;
   V3 n=unit(add(mul(s.worldNormals[ia],aw),add(mul(s.worldNormals[ib],bv),mul(s.worldNormals[ic],cw))),"ACL interpolated source normal invalid");
   best.p=q;best.normal=n;best.tri=static_cast<std::uint32_t>(i/3);best.d2=d;
  }}
 if(best.tri==MR_INVALID_INDEX)throw std::runtime_error("ACL surface has no nondegenerate triangle");return best;
}
inline void writeGeometryDiagnostic(const Frame& frame,const BoneSurface& fem,
                                    const BoneSurface& tib,V3 femSeed,V3 tibSeed,
                                    const Projection& fp,const Projection& tp,
                                    V3 axis,double femFacing,double tibFacing,
                                    double facingLimit) {
 const char* path=std::getenv("NUMI_HUMAN_INFERRED_ACL_DIAGNOSTIC_PATH");
 if(!path||!*path)return;
 std::ofstream out(path,std::ios::out|std::ios::trunc);
 if(!out)throw std::runtime_error("cannot open inferred ACL geometry diagnostic output");
 out<<std::setprecision(17);
 const auto vec=[&](V3 v){out<<'['<<v[0]<<','<<v[1]<<','<<v[2]<<']';};
 const auto hash=[&](const auto& bytes){
  for(const auto b:bytes)out<<std::hex<<std::setw(2)<<std::setfill('0')
      <<static_cast<unsigned>(b);
  out<<std::dec<<std::setfill(' ');
 };
 const auto projection=[&](const char* name,V3 seed,const Projection& p,
                           const BoneSurface& surface){
  out<<'\"'<<name<<"\":{\"seed_world_m\":";vec(seed);
  out<<",\"point_world_m\":";vec(p.p);
  out<<",\"interpolated_normal_world\":";vec(p.normal);
  out<<",\"triangle_record_local\":"<<p.tri
     <<",\"projection_residual_m\":"<<std::sqrt(p.d2)
     <<",\"triangle_world_vertices_m\":[";
  const std::size_t base=std::size_t(p.tri)*3u;
  for(unsigned k=0;k<3u;++k){if(k)out<<',';vec(surface.worldVertices[surface.triangleIndices[base+k]]);}
  out<<"],\"triangle_vertex_normals_world\":[";
  for(unsigned k=0;k<3u;++k){if(k)out<<',';vec(surface.worldNormals[surface.triangleIndices[base+k]]);}
  out<<"]}";
 };
 const auto surface=[&](const BoneSurface& b){
  out<<"{\"body_index\":"<<b.body<<",\"stable_id\":"<<b.stableId
     <<",\"source_record\":"<<b.sourceRecord<<",\"payload_sha256\":\"";
  hash(b.payloadSha256);
  out<<"\",\"vertices_world_m\":[";
  for(std::size_t i=0;i<b.worldVertices.size();++i){if(i)out<<',';vec(b.worldVertices[i]);}
  out<<"],\"vertex_normals_world\":[";
  for(std::size_t i=0;i<b.worldNormals.size();++i){if(i)out<<',';vec(b.worldNormals[i]);}
  out<<"],\"triangle_indices_record_local\":[";
  for(std::size_t i=0;i<b.triangleIndices.size();++i){if(i)out<<',';out<<b.triangleIndices[i];}
  out<<"]}";
 };
 out<<"{\"schema\":\"numi.human.inferred-acl-geometry-diagnostic.v1\","
    <<"\"side\":"<<frame.side<<",\"femur_body\":"<<frame.femurBody
    <<",\"tibia_body\":"<<frame.tibiaBody
    <<",\"patella_body\":"<<frame.patellaBody<<",\"frame_world_m\":{"
    <<"\"joint_center\":";vec(frame.jointCenterWorld);
 out<<",\"anterior\":";vec(frame.anteriorWorld);
 out<<",\"proximal\":";vec(frame.superiorWorld);
 out<<"},\"facing\":{\"axis_femur_to_tibia\":";vec(axis);
 out<<",\"femur_dot_axis\":"<<femFacing
    <<",\"tibia_dot_negative_axis\":"<<tibFacing
    <<",\"minimum_dot\":"<<std::min(femFacing,tibFacing)
    <<",\"required_minimum_dot\":"<<facingLimit
    <<"},\"selected_projections\":{";
 projection("femur",femSeed,fp,fem);out<<',';
 projection("tibia",tibSeed,tp,tib);
 out<<"},\"registered_surfaces\":[";surface(fem);out<<',';surface(tib);out<<"]}\n";
 out.flush();
 if(!out)throw std::runtime_error("failed writing inferred ACL geometry diagnostic output");
}
inline void param(numi::matter::MaterialProgram& m,const char* n,double v){
 auto i=std::find_if(m.parameters.begin(),m.parameters.end(),[&](const auto& p){return p.name==n;});
 if(i==m.parameters.end()||!std::isfinite(v)||v<i->lower||v>i->upper)throw std::runtime_error(std::string("ACL material parameter invalid: ")+n);
 i->defaultValue=v;i->identifiable=false;
}
}

// Copies one source bone record into neutral-pose world vertices and
// source-order local triangles. The caller must select record by the admitted
// stable ID/body registration; this helper does not guess anatomy from a mesh.
struct BoneSurfaceStorage {
 std::uint32_t body=MR_INVALID_INDEX,stableId=0,sourceRecord=MR_INVALID_INDEX;
 std::array<std::uint8_t,32> payloadSha256{};
 std::vector<V3> vertices,normals;
 std::vector<std::uint32_t> triangles;
 BoneSurface view() const { return {body,stableId,sourceRecord,payloadSha256,vertices,normals,triangles}; }
};
template<class BoneRecord,class BoneVertex>
inline BoneSurfaceStorage bakeRegisteredBoneSurface(
 const BoneRecord& record,std::uint32_t sourceRecord,
 std::span<const BoneVertex> sourceVertices,
 std::span<const std::uint32_t> sourceIndices,
 const MRBodyStateGPU& bodyPose,
 std::array<std::uint8_t,32> payloadSha256) {
 using namespace detail;
 if(record.firstVertex>sourceVertices.size()||record.vertexCount>sourceVertices.size()-record.firstVertex||
    record.firstIndex>sourceIndices.size()||record.indexCount>sourceIndices.size()-record.firstIndex||record.indexCount%3)
  throw std::runtime_error("registered ACL bone record range invalid");
 BoneSurfaceStorage out;out.body=record.bodyIndex;out.stableId=record.stableId;out.sourceRecord=sourceRecord;out.payloadSha256=payloadSha256;
 out.vertices.reserve(record.vertexCount);out.normals.reserve(record.vertexCount);
 mr_float4 boneQ{record.quaternionX,record.quaternionY,record.quaternionZ,record.quaternionW};
 for(std::uint32_t i=0;i<record.vertexCount;i++){
  const auto& v=sourceVertices[record.firstVertex+i];
  V3 local=add(V3{record.translationX,record.translationY,record.translationZ},
    rotate(boneQ,V3{v.positionX*record.uniformScale,v.positionY*record.uniformScale,v.positionZ*record.uniformScale}));
  out.vertices.push_back(add(xyz(bodyPose.position),rotate(bodyPose.orientation,local)));
  V3 sourceNormal=unit(V3{v.normalX,v.normalY,v.normalZ},"ACL source bone vertex normal invalid");
  out.normals.push_back(unit(rotate(bodyPose.orientation,rotate(boneQ,sourceNormal)),"ACL transformed source normal invalid"));
 }
 out.triangles.reserve(record.indexCount);
 for(std::uint32_t i=0;i<record.indexCount;i++){
  const std::uint32_t source=sourceIndices[record.firstIndex+i];
  if(source<record.firstVertex||source>=record.firstVertex+record.vertexCount)
   throw std::runtime_error("registered ACL bone triangle crosses source record range");
  out.triangles.push_back(source-record.firstVertex);
 }
 return out;
}

// Derive the ligament frame from the admitted femur-tibia joint-chain
// anchors, proximal hip joint, and registered patella body centre before mass
// rebasing. The segmented knee chain and proximal femur joint must be unique.
inline Frame deriveRegisteredKneeFrame(
 const EngineModel& model,std::span<const MRBodyStateGPU> poses,
 std::uint32_t side,std::uint32_t femurBody,std::uint32_t tibiaBody,
 std::uint32_t patellaBody,std::uint32_t patellaBoneStableId) {
 using namespace detail;
 if(femurBody>=poses.size()||tibiaBody>=poses.size()||patellaBody>=poses.size()||
    femurBody==tibiaBody) throw std::runtime_error("registered ACL frame body index invalid");
 const auto jointAnchor=[&](const MRJointDescriptorGPU& j) {
  if(j.parentBody>=poses.size()||j.childBody>=poses.size())
   throw std::runtime_error("ACL joint chain body index invalid");
  V3 p=add(xyz(poses[j.parentBody].position),rotate(poses[j.parentBody].orientation,xyz(j.parentAnchor)));
  V3 c=add(xyz(poses[j.childBody].position),rotate(poses[j.childBody].orientation,xyz(j.childAnchor)));
  const double residual=norm(sub(p,c));
  if(!std::isfinite(residual)||residual>2e-4)
   throw std::runtime_error("ACL source joint anchor residual exceeds 0.2 mm");
  return mul(add(p,c),.5);
 };
 // The admitted MyoSim chain can put a small carrier between femur and tibia.
 // Walk the unique incoming-joint path instead of assuming direct adjacency.
 std::vector<const MRJointDescriptorGPU*> tibiaToFemur;
 std::uint32_t cursor=tibiaBody;
 for(std::size_t depth=0;depth<model.bodies.size()&&cursor!=femurBody;++depth) {
  const MRJointDescriptorGPU* incoming=nullptr;
  for(const auto& j:model.joints) if(j.childBody==cursor) {
   if(incoming) throw std::runtime_error("ambiguous ACL knee joint ancestor");
   incoming=&j;
  }
  if(!incoming) throw std::runtime_error("missing ACL femur-tibia ancestor chain");
  tibiaToFemur.push_back(incoming);
  cursor=incoming->parentBody;
 }
 if(cursor!=femurBody||tibiaToFemur.empty())
  throw std::runtime_error("ACL femur-tibia ancestor chain does not terminate at femur");
 const auto upperJoint=tibiaToFemur.back();
 const auto lowerJoint=tibiaToFemur.front();
 if(upperJoint->parentBody!=femurBody||lowerJoint->childBody!=tibiaBody)
  throw std::runtime_error("ACL knee chain endpoint ownership changed");
 const V3 upperKnee=jointAnchor(*upperJoint);
 const V3 lowerKnee=jointAnchor(*lowerJoint);
 const double chainResidual=norm(sub(upperKnee,lowerKnee));
 if(!std::isfinite(chainResidual)||chainResidual>.030)
  throw std::runtime_error("ACL segmented knee anchors disagree by more than 30 mm");
 const V3 knee=mul(add(upperKnee,lowerKnee),.5);
 const MRJointDescriptorGPU* hipJoint=nullptr;
 for(const auto& j:model.joints) if(j.childBody==femurBody) {
  if(hipJoint) throw std::runtime_error("ambiguous ACL proximal femur joint");
  hipJoint=&j;
 }
 if(!hipJoint) throw std::runtime_error("missing ACL proximal femur joint");
 const V3 hip=jointAnchor(*hipJoint);
 const V3 proximal=unit(sub(hip,knee),"ACL hip-to-knee anatomical axis invalid");
 V3 ant=sub(xyz(poses[patellaBody].position),knee);
 ant=unit(sub(ant,mul(proximal,dot(ant,proximal))),"ACL patella anterior reference degenerate");
 Frame f;f.side=side;f.femurBody=femurBody;f.tibiaBody=tibiaBody;
 f.patellaBody=patellaBody;f.patellaBoneStableId=patellaBoneStableId;
 f.neutralBodyPoses=poses;f.jointCenterWorld=knee;f.anteriorWorld=ant;f.superiorWorld=proximal;
 return f;
}

struct ACLSeedSelection {
 V3 posteriorAxisWorld{},proximalAxisWorld{},lateralAxisWorld{};
 std::array<double,3> femurBasePosteriorProximalLateralMeters{};
 std::array<double,3> femurAdjustmentPosteriorProximalLateralMeters{};
 std::array<double,3> femurTotalPosteriorProximalLateralMeters{};
 std::array<double,3> tibiaBasePosteriorProximalLateralMeters{};
 std::array<double,3> tibiaAdjustmentPosteriorProximalLateralMeters{};
 std::array<double,3> tibiaTotalPosteriorProximalLateralMeters{};
 V3 femurSeedWorld{},tibiaSeedWorld{};
};

inline ACLSeedSelection deriveRegisteredACLSeedSelection(const Frame& f) {
 using namespace detail;
 if(f.side>1)throw std::runtime_error("ACL seed side index invalid");
 const V3 ant=unit(f.anteriorWorld,"ACL patella-derived anterior invalid");
 const V3 proximal=unit(sub(f.superiorWorld,mul(ant,dot(f.superiorWorld,ant))),"ACL superior invalid");
 ACLSeedSelection out;
 out.posteriorAxisWorld=mul(ant,-1);
 out.proximalAxisWorld=proximal;
 out.lateralAxisWorld=unit(cross(proximal,ant),"ACL inferred lateral seed axis invalid");
 out.femurBasePosteriorProximalLateralMeters={.015,.004,0};
 out.femurAdjustmentPosteriorProximalLateralMeters={.002,-.003,-.004*(f.side==0?1.0:-1.0)};
 out.tibiaBasePosteriorProximalLateralMeters={-.012,-.004,0};
 out.tibiaAdjustmentPosteriorProximalLateralMeters={.005,-.003,.005*(f.side==0?1.0:-1.0)};
 const auto sum=[](const std::array<double,3>& a,const std::array<double,3>& b) {
  return std::array<double,3>{a[0]+b[0],a[1]+b[1],a[2]+b[2]};
 };
 out.femurTotalPosteriorProximalLateralMeters=sum(
     out.femurBasePosteriorProximalLateralMeters,
     out.femurAdjustmentPosteriorProximalLateralMeters);
 out.tibiaTotalPosteriorProximalLateralMeters=sum(
     out.tibiaBasePosteriorProximalLateralMeters,
     out.tibiaAdjustmentPosteriorProximalLateralMeters);
 const auto& femur=out.femurTotalPosteriorProximalLateralMeters;
 const auto& tibia=out.tibiaTotalPosteriorProximalLateralMeters;
 out.femurSeedWorld=add(f.jointCenterWorld,add(mul(out.posteriorAxisWorld,femur[0]),
     add(mul(out.proximalAxisWorld,femur[1]),mul(out.lateralAxisWorld,femur[2]))));
 out.tibiaSeedWorld=add(f.jointCenterWorld,add(mul(out.posteriorAxisWorld,tibia[0]),
     add(mul(out.proximalAxisWorld,tibia[1]),mul(out.lateralAxisWorld,tibia[2]))));
 return out;
}

inline BuiltObject buildRegisteredSkeletonACL(const Frame& f,const std::array<BoneSurface,2>& s,numi::matter::MaterialProgram mat){
 using namespace detail;const auto& fem=s[0];const auto& tib=s[1];
 if(f.side>1||f.femurBody==MR_INVALID_INDEX||f.tibiaBody==MR_INVALID_INDEX||f.femurBody==f.tibiaBody||fem.body!=f.femurBody||tib.body!=f.tibiaBody||!fem.stableId||!tib.stableId||
 f.femurBody>=f.neutralBodyPoses.size()||f.tibiaBody>=f.neutralBodyPoses.size()||f.patellaBody>=f.neutralBodyPoses.size())throw std::runtime_error("ACL registered anatomy/body frame mismatch");
 const Provenance limits{};
 const ACLSeedSelection seedSelection=deriveRegisteredACLSeedSelection(f);
 const V3 femSeed=seedSelection.femurSeedWorld,tibSeed=seedSelection.tibiaSeedWorld;
 auto fc=project(femSeed,fem),tc=project(tibSeed,tib);
 V3 line=sub(tc.p,fc.p);double length=norm(line);if(!std::isfinite(length)||length<limits.minimumAssumedLengthMeters||length>limits.maximumAssumedLengthMeters)throw std::runtime_error("ACL projected attachment length outside 12-60 mm");
 V3 axis=mul(line,1/length);
 const double femFacing=dot(fc.normal,axis),tibFacing=dot(tc.normal,mul(axis,-1));
 double facing=std::min(femFacing,tibFacing);
 if(!std::isfinite(facing)||facing<limits.minimumInterboneFacingNormalDotLimit) {
  writeGeometryDiagnostic(f,fem,tib,femSeed,tibSeed,fc,tc,axis,femFacing,tibFacing,
                          limits.minimumInterboneFacingNormalDotLimit);
  std::ostringstream message;message<<std::setprecision(17)
   <<"ACL attachment source normals do not face across the interbone span"
   <<" side="<<f.side<<" femur_body="<<f.femurBody<<" tibia_body="<<f.tibiaBody
   <<" femur_triangle="<<fc.tri<<" tibia_triangle="<<tc.tri
   <<" femur_dot_axis="<<femFacing<<" tibia_dot_negative_axis="<<tibFacing
   <<" minimum_dot="<<facing
   <<" required_minimum_dot="<<limits.minimumInterboneFacingNormalDotLimit;
  throw std::runtime_error(message.str());
 }
 V3 width=unit(cross(axis,seedSelection.proximalAxisWorld),"ACL width invalid"),depth=unit(cross(width,axis),"ACL depth invalid");
 constexpr double hw=.004,hd=.0025;constexpr std::array<std::array<double,2>,4> c{{{{-1,-1}},{{1,-1}},{{1,1}},{{-1,1}}}};
 BuiltObject o;o.object.name=f.side?"inferred_registered_right_ACL_reference_v1":"inferred_registered_left_ACL_reference_v1";
 o.object.representation=numi::matter::Representation::fem;o.object.mixedFEM=false;o.object.deformableSelfContact=false;o.object.characteristicLength=.004;o.object.identifiable=false;
 o.material=std::move(mat);o.material.name=o.object.name+"_febio_reference";o.material.fingerprint=0;
 param(o.material,"density",1000);param(o.material,"c1",1.95e6);param(o.material,"bulk",146.41e6);param(o.material,"initial_stretch",1);param(o.material,"fiber_scale",1);
 param(o.material,"fiber_x",axis[0]);param(o.material,"fiber_y",axis[1]);param(o.material,"fiber_z",axis[2]);
 std::array<V3,4> end[2];std::array<std::uint32_t,4> ftri{},ttri{};double maxProjection=std::max(std::sqrt(fc.d2),std::sqrt(tc.d2)),maxFootprint=0,minPatchAlignment=1;
 for(unsigned k=0;k<4;k++){
  V3 off=add(mul(width,hw*c[k][0]),mul(depth,hd*c[k][1]));auto fp=project(add(fc.p,off),fem),tp=project(add(tc.p,off),tib);
  const double fr=norm(sub(fp.p,fc.p)),tr=norm(sub(tp.p,tc.p));
  const double fa=dot(fp.normal,fc.normal),ta=dot(tp.normal,tc.normal);
  if(!std::isfinite(fr)||!std::isfinite(tr)||fr>limits.inferredFootprintRadiusLimitMeters||tr>limits.inferredFootprintRadiusLimitMeters)throw std::runtime_error("ACL attachment corner escaped 12 mm registered source footprint");
  if(!std::isfinite(fa)||!std::isfinite(ta)||fa<limits.minimumPatchNormalAlignmentDotLimit||ta<limits.minimumPatchNormalAlignmentDotLimit)throw std::runtime_error("ACL attachment corner crosses registered source surface normal patch");
  maxFootprint=std::max({maxFootprint,fr,tr});minPatchAlignment=std::min({minPatchAlignment,fa,ta});
  end[0][k]=fp.p;end[1][k]=tp.p;ftri[k]=fp.tri;ttri[k]=tp.tri;maxProjection=std::max({maxProjection,std::sqrt(fp.d2),std::sqrt(tp.d2)});
 }
 if(maxProjection>limits.maximumAttachmentProjectionResidualLimitMeters)throw std::runtime_error("ACL attachment projection exceeds 20 mm");
 o.nodeAnchors.resize(16);o.nodeLoads.resize(16);o.donorBodyByLocalNode.resize(16);o.donorLocalPositionByLocalNode.resize(16);
 for(unsigned st=0;st<4;st++)for(unsigned k=0;k<4;k++){double t=st/3.0;V3 p=st==0?end[0][k]:st==3?end[1][k]:add(mul(end[0][k],1-t),mul(end[1][k],t));unsigned n=st*4+k;o.object.femNodes.push_back(p);
  unsigned donor=st<2?f.femurBody:f.tibiaBody;o.donorBodyByLocalNode[n]=donor;auto pose=f.neutralBodyPoses[donor];o.donorLocalPositionByLocalNode[n]=invRotate(pose.orientation,sub(p,xyz(pose.position)));
  std::fill_n(o.nodeLoads[n].endpointIndex,4,NM_INVALID_INDEX);o.nodeAnchors[n].bodyIndex=MR_INVALID_INDEX;
  if(st==0||st==3){unsigned body=st==0?f.femurBody:f.tibiaBody;auto bp=f.neutralBodyPoses[body];V3 local=invRotate(bp.orientation,sub(p,xyz(bp.position)));auto& a=o.nodeAnchors[n];a.bodyIndex=body;a.flags=NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;a.localPoint={float(local[0]),float(local[1]),float(local[2]),0};
   o.object.femFixedNodes.push_back(n); // Passive standalone FEM owner; nodeAnchors supplies exact body-local reactions.
  }
 }
 constexpr std::array<std::array<std::uint32_t,4>,6> pat{{{{0,1,2,6}},{{0,2,3,6}},{{0,3,7,6}},{{0,7,4,6}},{{0,4,5,6}},{{0,5,1,6}}}};
 o.object.tetrahedra.reserve(18);
 for(unsigned seg=0;seg<3;seg++)for(auto p:pat){std::array<std::uint32_t,4> t{};for(unsigned j=0;j<4;j++)t[j]=4*seg+p[j]%4+(p[j]>=4?4:0);
  V3 a=o.object.femNodes[t[0]],b=o.object.femNodes[t[1]],cc=o.object.femNodes[t[2]],d=o.object.femNodes[t[3]];double vol=dot(sub(b,a),cross(sub(cc,a),sub(d,a)))/6;
  if(!std::isfinite(vol)||std::abs(vol)<1e-12)throw std::runtime_error("ACL tetrahedron degenerate");if(vol<0)std::swap(t[0],t[1]);o.object.tetrahedra.push_back({t});}
 std::map<std::array<std::uint32_t,3>,std::uint32_t> faceUse;
 for(const auto& tet:o.object.tetrahedra){
  constexpr unsigned faces[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
  for(const auto& fidx:faces){std::array<std::uint32_t,3> fkey{tet.nodes[fidx[0]],tet.nodes[fidx[1]],tet.nodes[fidx[2]]};std::sort(fkey.begin(),fkey.end());if(++faceUse[fkey]>2)throw std::runtime_error("ACL tetra mesh has nonmanifold face");}
 }
 std::map<std::array<std::uint32_t,2>,std::uint32_t> boundaryEdgeUse;std::uint32_t boundaryFaceCount=0;
 for(const auto& [face,count]:faceUse)if(count==1){++boundaryFaceCount;for(unsigned e=0;e<3;e++){std::array<std::uint32_t,2> edge{face[e],face[(e+1)%3]};std::sort(edge.begin(),edge.end());++boundaryEdgeUse[edge];}}
 if(boundaryFaceCount!=28||std::any_of(boundaryEdgeUse.begin(),boundaryEdgeUse.end(),[](const auto& e){return e.second!=2;}))throw std::runtime_error("ACL tetra boundary is not a closed two-manifold");
 auto& p=o.provenance;p.side=f.side;p.femurBody=f.femurBody;p.tibiaBody=f.tibiaBody;p.patellaBody=f.patellaBody;p.patellaBoneStableId=f.patellaBoneStableId;
 p.femurBoneStableId=fem.stableId;p.tibiaBoneStableId=tib.stableId;p.femurSourceRecord=fem.sourceRecord;p.tibiaSourceRecord=tib.sourceRecord;p.femurPayloadSha256=fem.payloadSha256;p.tibiaPayloadSha256=tib.payloadSha256;p.selectedSurfaceTriangles={fc.tri,tc.tri};p.femurAnchorTriangles=ftri;p.tibiaAnchorTriangles=ttri;p.maximumAttachmentProjectionResidualMeters=maxProjection;p.maximumAttachmentFootprintRadiusMeters=maxFootprint;p.minimumInterboneFacingNormalDot=facing;p.minimumPatchNormalAlignmentDot=minPatchAlignment;p.assumedLengthMeters=length;p.femurSeedOffsetsPosteriorProximalLateralMeters=seedSelection.femurTotalPosteriorProximalLateralMeters;p.tibiaSeedOffsetsPosteriorProximalLateralMeters=seedSelection.tibiaTotalPosteriorProximalLateralMeters;p.femurSeedBasePosteriorProximalLateralMeters=seedSelection.femurBasePosteriorProximalLateralMeters;p.femurSeedAdjustmentPosteriorProximalLateralMeters=seedSelection.femurAdjustmentPosteriorProximalLateralMeters;p.tibiaSeedBasePosteriorProximalLateralMeters=seedSelection.tibiaBasePosteriorProximalLateralMeters;p.tibiaSeedAdjustmentPosteriorProximalLateralMeters=seedSelection.tibiaAdjustmentPosteriorProximalLateralMeters;p.seedPosteriorAxisWorld=seedSelection.posteriorAxisWorld;p.seedProximalAxisWorld=seedSelection.proximalAxisWorld;p.seedLateralAxisWorld=seedSelection.lateralAxisWorld;p.seedGeometryInferred=true;p.sourceCandidateValidatedForThisSide=false;return o;
}
inline std::vector<NumiHumanTissueMassNode> makeCookedMassNodes(const BuiltObject& o,std::uint32_t first,std::span<const float> masses){
 if(masses.size()!=o.object.femNodes.size()||masses.size()!=o.donorBodyByLocalNode.size()||masses.size()!=o.donorLocalPositionByLocalNode.size())throw std::runtime_error("ACL cooked mass map size mismatch");
 std::vector<NumiHumanTissueMassNode> r;r.reserve(masses.size());for(std::uint32_t i=0;i<masses.size();i++){if(!std::isfinite(masses[i])||masses[i]<=0)throw std::runtime_error("ACL cooked nodal mass invalid");r.push_back({.nodeIndex=first+i,.donorBody=o.donorBodyByLocalNode[i],.massKg=masses[i],.localPosition=o.donorLocalPositionByLocalNode[i]});}return r;
}
inline void applyCOMRebase(BuiltObject& o,std::span<const NumiHumanTissueMassPartition> p){
 for(const auto& part:p){for(auto& a:o.nodeAnchors)if(a.bodyIndex==part.donorBody){a.localPoint.x-=float(part.remainingCOMOffsetM[0]);a.localPoint.y-=float(part.remainingCOMOffsetM[1]);a.localPoint.z-=float(part.remainingCOMOffsetM[2]);}}
}
} // namespace metalrobo::inferred_acl
