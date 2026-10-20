#pragma once
#include "metalrobo/EngineModel.hpp"
#include "NumiHumanInferredACL.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace metalrobo::registered_ligaments {
using V3=inferred_acl::V3;
enum class Group : std::uint32_t { hipCapsule, shoulderCapsuloligament, ankle, elbow, knee };
inline constexpr std::string_view kReferenceCatalogSchema="numi-human-registered-ligament-reference-v4-elbow-rcl-bones-43a31ef4771e";
inline constexpr double kC4=1e-3,kLambdaMax=1.05,kMaxProjectionM=.020,kMaxJointResidualM=.0002;
inline constexpr std::array<std::uint8_t,32> kReferenceBonesPayloadSha256{{
 0x43,0xa3,0x1e,0xf4,0x77,0x1e,0x2a,0x38,0x3a,0xe2,0xc7,0xca,0xab,0x1b,0x99,0x4f,
 0x0c,0x28,0x07,0x6c,0x2a,0x43,0x50,0x18,0x60,0xcd,0x1b,0xa3,0xfe,0x77,0x08,0xc0}};
struct Spec {
 std::string_view name;Group group{};std::uint32_t side=0;
 std::uint32_t body0=MR_INVALID_INDEX,stable0=0,body1=MR_INVALID_INDEX,stable1=0;
 std::uint32_t frameParent=MR_INVALID_INDEX,frameChild=MR_INVALID_INDEX;
 V3 seed0{},seed1{};double stiffnessNPerM=0,areaM2=0;
 // The admitted catalog stores side-specific registered-surface witnesses.
 // Generic caller-created specs may request an explicit local-Z mirror.
 bool seedLateralityMirrored=true;
 bool requiresPinnedSourceIdentity=false;std::uint32_t expectedFrameJoint=MR_INVALID_INDEX;
 // Literature cohort area input; never a measurement on the registered person.
 bool areaMeasured=false,areaIsAssumedSplit=false;
 std::string_view stiffnessSource,geometrySource,limitations;
};
struct OwnershipPolicy {bool openKneeOwner=false,inferredACLOwner=false,hipCapsuleAggregateOwner=false,overlappingSourcePassiveCoordinateOwner=false;};
struct Provenance {
 std::string name,className="inferred_registered_reference_fiber";
 std::uint32_t side=0,group=0,body0=MR_INVALID_INDEX,body1=MR_INVALID_INDEX,stable0=0,stable1=0,record0=MR_INVALID_INDEX,record1=MR_INVALID_INDEX;
 std::array<std::uint8_t,32> payloadSha0{},payloadSha1{};
 std::uint32_t frameJoint=MR_INVALID_INDEX,frameParent=MR_INVALID_INDEX,frameChild=MR_INVALID_INDEX,triangle0=MR_INVALID_INDEX,triangle1=MR_INVALID_INDEX;
 double residual0M=0,residual1M=0,stiffnessNPerM=0,areaM2=0,c3Pa=0,c4=0,c5Pa=0,lambdaMax=0,referenceLengthM=0,inSituStretch=1;
 std::array<double,3> seedOffset0FrameM{},seedOffset1FrameM{};
 std::array<double,7> referenceBody0Pose{},referenceBody1Pose{},referenceFramePose{};
 bool areaMeasured=false,areaIsAssumedSplit=false,seedLateralityMirrored=true;
 bool inferred=true,subjectCalibrated=false,sourceEnthesesMeasured=false;
 std::string stiffnessSource,geometrySource,limitations;
};
struct Built {NMNumiHumanPassiveLigamentGPU row{};Provenance provenance;};
struct Frame {V3 p{};mr_float4 q{0,0,0,1};std::uint32_t index=MR_INVALID_INDEX,parent=MR_INVALID_INDEX,child=MR_INVALID_INDEX;};
namespace detail {
inline std::uint32_t mapBody(std::uint32_t b,std::span<const std::uint32_t> m){if(m.empty())return b;if(b>=m.size())throw std::runtime_error("ligament source body outside runtime map");return m[b];}
inline mr_float4 qmul(mr_float4 a,mr_float4 b){return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};}
inline V3 xyz(mr_float4 a){return {a.x,a.y,a.z};}
inline V3 world(const MRBodyStateGPU& b,mr_float4 p){return inferred_acl::detail::add(xyz(b.position),inferred_acl::detail::rotate(b.orientation,xyz(p)));}
inline Frame frame(const Spec& s,const EngineModel& m,std::span<const MRBodyStateGPU> poses,std::span<const std::uint32_t> map){
 auto pb=mapBody(s.frameParent,map),cb=mapBody(s.frameChild,map);const MRJointDescriptorGPU* f=nullptr;std::uint32_t ji=MR_INVALID_INDEX;
 for(std::uint32_t i=0;i<m.joints.size();++i)if(m.joints[i].parentBody==pb&&m.joints[i].childBody==cb){if(f)throw std::runtime_error("ligament frame descriptor ambiguous");f=&m.joints[i];ji=i;}
 if(!f||pb>=poses.size()||cb>=poses.size())throw std::runtime_error("ligament source joint frame absent");
 V3 a=world(poses[pb],f->parentAnchor),b=world(poses[cb],f->childAnchor);double r=inferred_acl::detail::norm(inferred_acl::detail::sub(a,b));
 if(!std::isfinite(r)||r>kMaxJointResidualM)throw std::runtime_error("ligament joint anchor residual exceeds 0.2 mm");
 return {inferred_acl::detail::mul(inferred_acl::detail::add(a,b),.5),qmul(poses[pb].orientation,f->parentRotation),ji,pb,cb};
}
inline V3 seed(const Frame& f,V3 x,std::uint32_t side,bool mirrorLaterality=true){
 if(side>1u)throw std::runtime_error("ligament side must be left=0 or right=1");
 // Source joint frames use local Z as the mediolateral axis. This fallback is
 // only for synthetic/caller-created specs; the reference catalog stores
 // exact side-specific surface witnesses from the admitted source payload.
 if(mirrorLaterality&&side==0u)x[2]=-x[2];
 return inferred_acl::detail::add(f.p,inferred_acl::detail::rotate(f.q,x));
}
inline const inferred_acl::BoneSurface& surface(std::span<const inferred_acl::BoneSurface> ss,std::uint32_t b,std::uint32_t id,V3 p){
 const inferred_acl::BoneSurface* best=nullptr;double d=std::numeric_limits<double>::infinity();
 for(const auto& s:ss)if(s.body==b&&(id==0||s.stableId==id)){if(!s.stableId||s.sourceRecord==MR_INVALID_INDEX)throw std::runtime_error("ligament surface lacks source provenance");auto x=inferred_acl::detail::project(p,s);if(x.d2<d){d=x.d2;best=&s;}}
 if(!best)throw std::runtime_error("ligament registered source surface absent");return *best;
}
inline V3 local(const MRBodyStateGPU& p,V3 x){return inferred_acl::detail::invRotate(p.orientation,inferred_acl::detail::sub(x,xyz(p.position)));}
inline std::string vectorString(V3 v){return "["+std::to_string(v[0])+","+std::to_string(v[1])+","+std::to_string(v[2])+"]";}
inline bool segmentHitsSurface(V3 a,V3 b,const inferred_acl::BoneSurface& s){
 const auto sub=inferred_acl::detail::sub;const auto dot=inferred_acl::detail::dot;const auto cross=inferred_acl::detail::cross;
 const V3 d=sub(b,a);const double d2=dot(d,d);if(!(d2>0)||s.worldVertices.empty())return false;
 const auto& v=s.worldVertices;const auto& ix=s.triangleIndices;
 V3 segmentLo{},segmentHi{},surfaceLo=v.front(),surfaceHi=v.front();
 for(std::size_t axis=0;axis<3;++axis){
  segmentLo[axis]=std::min(a[axis],b[axis])-1e-12;
  segmentHi[axis]=std::max(a[axis],b[axis])+1e-12;
 }
 for(const auto& point:v)for(std::size_t axis=0;axis<3;++axis){
  surfaceLo[axis]=std::min(surfaceLo[axis],point[axis]);
  surfaceHi[axis]=std::max(surfaceHi[axis],point[axis]);
 }
 for(std::size_t axis=0;axis<3;++axis)
  if(surfaceHi[axis]<segmentLo[axis]||surfaceLo[axis]>segmentHi[axis])return false;
 for(std::size_t i=0;i+2<ix.size();i+=3){
  const V3 v0=v[ix[i]],e1=sub(v[ix[i+1]],v0),e2=sub(v[ix[i+2]],v0);
  const V3 h=cross(d,e2);const double det=dot(e1,h);
  const double scale=std::sqrt(d2*dot(e1,e1)*dot(e2,e2));
  if(!std::isfinite(scale)||scale==0||std::abs(det)<=32.0*std::numeric_limits<double>::epsilon()*scale)continue;
  const double inv=1.0/det;const V3 ss=sub(a,v0);const double u=inv*dot(ss,h);
  if(u < -1e-8||u > 1.0+1e-8)continue;
  const V3 q=cross(ss,e1);const double w=inv*dot(d,q);
  if(w < -1e-8||u+w > 1.0+1e-8)continue;
  const double t=inv*dot(e2,q);
  if(t>1e-5&&t<1.0-1e-5)return true;
 }
 return false;
}
inline void ownership(const Spec& s,const OwnershipPolicy& p){
 if(p.overlappingSourcePassiveCoordinateOwner)throw std::runtime_error("ligament conflicts with overlapping source passive-coordinate owner");
 if(s.group==Group::hipCapsule&&p.hipCapsuleAggregateOwner)throw std::runtime_error("hip rows conflict with --resting-hip-capsule-reference");
 if(s.group==Group::knee) {
  if(p.openKneeOwner||p.inferredACLOwner)
   throw std::runtime_error("ligament duplicates ACL/OpenKnee owner");
  throw std::runtime_error("catalog may not create knee row");
 }
}
}
inline std::vector<Spec> referenceCatalog(){
 constexpr std::string_view hs="Karunaseelan et al., Bone Joint Res 2021;10(9):594-601, PMID 34555959, doi:10.1302/2046-3758.109.BJR-2020-0536.R1; cadaver-derived capsular restraint model priors, not subject calibration.";
 constexpr std::string_view hg="Pieroh et al., PLoS One 2016, doi:10.1371/journal.pone.0163306; cohort tested-strip CSA means. The ILF sample was the inferior portion; no full-band or subject-specific CSA is claimed.";
 constexpr std::string_view hl="Inferred regional hip-capsule proxy fibers with anchors projected in the exact supported reference pose; no subject-specific entheses. Each straight path is checked against all 185 retained registered source-bone meshes at construction; the open source meshes do not prove closed-solid clearance. Zero-prestrain is assigned at construction, not measured from tissue; stiffness is a literature prior, not subject calibration.";
 constexpr std::string_view ss="Boardman et al., J Shoulder Elbow Surg 1996, PMID 8872921; fresh-frozen bone-ligament-bone tests, n=10 each; CHL 36.7 N/mm, SGHL 17.4 N/mm stiffness priors. CHL cohort CSA 53.7 mm^2 is a literature normalization, not this subject.";
 constexpr std::string_view sg="Boardman et al., J Shoulder Elbow Surg 1996, PMID 8872921; Dekker et al. 2020, PMID 32441993, attachment mapping. CHL geometry is an inferred coracoid-region to proximal-humeral regional fiber, not a measured attachment.";
 constexpr std::string_view sl="Inferred capsuloligamentous fibers, not measured subject entheses. Geometry uses NHMYO2 SHA-256 e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b: CORB records 224/287 route-first sites 672/1093 on scapula bodies 34/84 as coracoid-region landmarks, not CHL entheses; the humeral sites are bounded by SUBSC records 215/278 route-first sites 707/1128, a lesser-tubercle-region proxy, not a measured CHL insertion. CHL fibers vary and may be capsular reflections, not a discrete ligament (PMID 22971672). Anchors project in the exact supported reference pose and each straight path is checked against all 185 retained registered source-bone meshes; open meshes do not prove closed-solid clearance. Zero-prestrain is assigned at construction.";
 constexpr std::string_view as="Ji et al., Front Bioeng Biotechnol 2025, doi:10.3389/fbioe.2025.1697096, Table 2; stiffness priors cited to Hoefnagels et al. 2007.";
 constexpr std::string_view ag="Inferred named ankle-region fibers use exact-invocation source-mesh surface witnesses projected in the supported mortise pose. Fibula shares the tibia rigid owner and navicular is unavailable. Each straight path is checked against all 185 retained registered source-bone meshes; open meshes do not prove closed-solid clearance or enthesis identity.";
 constexpr std::string_view al="FE spring priors, not subject measurements. Superior and inferior ATFL regional fibers use separate cited stiffness priors; anchors are inferred and zero-prestrain is assigned at construction. Effective area is stiffness-preserving normalization, not measured.";
 constexpr std::string_view es="Rahman et al., Adv Biomech Appl 2014;1(3):169-185, doi:10.12989/aba.2014.1.3.169, Table 3: RCL linear-region stiffness coefficient 15.5 N/mm per bundle, three modeled bundles. The catalog aggregates those priors to 46.5 N/mm (46500 N/m) in one effective row; it is not a subject-specific measurement or an exact reproduction of Rahman zero-load lengths.";
 constexpr std::string_view eg="Inferred radial-collateral-to-radial-head regional proxy. NHMYO2 SHA-256 e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b ECRB route nodes 763/1215, sites 682/1103, mark the lateral epicondylar neighborhood; source NHRIGID ulna-radius joints 42/92 locate the radial-head neighborhood. Endpoint anchors are projections to stable source surfaces 12/13 and 16/17 in the exact supported pose.";
 constexpr std::string_view el="The radial collateral ligament normally blends with the annular ligament; no independent annular-ligament body is present. This is an inferred straight radial-head proxy, not a measured RCL enthesis or annular wrap. MCL anterior/posterior and LUCL rows are intentionally not instantiated: bounded NHMYO landmark-neighborhood searches at the supported pose did not yield all-185-mesh-clear, non-inward straight paths for this owner; their absence here is a representation limit, not evidence that the ligaments are absent. Three published bundle tangent priors are combined into one line; the effective 10 mm^2 area only normalizes the GPU stress law and is not measured. Exact-pose all-185 source-mesh segment checks run at construction; open surface meshes do not prove closed-solid clearance. Zero-prestrain is assigned at construction; stiffness is a literature prior, not subject calibration.";
 std::vector<Spec> v;auto add=[&](std::string_view n,Group g,unsigned side,unsigned b0,unsigned s0,unsigned b1,unsigned s1,unsigned fp,unsigned fc,V3 x,V3 y,double k,double a,bool am,std::string_view ks,std::string_view gs,std::string_view l){
  const bool splitIlf=n=="iliofemoral_lateral"||n=="iliofemoral_medial";
  Spec spec;spec.name=n;spec.group=g;spec.side=side;spec.body0=b0;spec.stable0=s0;spec.body1=b1;spec.stable1=s1;
  spec.frameParent=fp;spec.frameChild=fc;spec.seed0=x;spec.seed1=y;spec.stiffnessNPerM=k;spec.areaM2=a;
  spec.areaMeasured=am&&!splitIlf;spec.areaIsAssumedSplit=splitIlf;spec.stiffnessSource=ks;spec.geometrySource=gs;spec.limitations=l;
  spec.requiresPinnedSourceIdentity=true;
  spec.expectedFrameJoint=g==Group::hipCapsule?(side==0?144u:130u):g==Group::shoulderCapsuloligament?(side==0?90u:40u):g==Group::elbow?(side==0?91u:41u):(side==0?150u:136u);
  auto witness=[&](V3 p0,V3 p1){spec.seed0=p0;spec.seed1=p1;spec.seedLateralityMirrored=false;};
  // Exact-invocation source-mesh witnesses, in each source joint frame (m).
  // These are inferred regional proxies, not measured subject entheses.
  if(n=="iliofemoral_lateral") witness(side==0?V3{.014512955211102962,.03726789727807045,-.016731154173612595}:V3{.014972856268286705,.036702800542116165,.014917828142642975},side==0?V3{.006924922578036785,-.01169358566403389,-.025930127128958702}:V3{.007530724164098501,-.006709011271595955,.02496389113366604});
  else if(n=="iliofemoral_medial") witness(side==0?V3{.020890163257718086,.023971397429704666,.004934210795909166}:V3{.022738156840205193,.02266797050833702,-.01300419308245182},side==0?V3{.020040323957800865,-.0005213750991970301,.005589937325567007}:V3{.02070983313024044,-.0015570733230561018,-.006645498797297478});
  else if(n=="pubofemoral") witness(side==0?V3{.02560751512646675,.00952922273427248,.028393015265464632}:V3{.026240482926368713,.009608231484889984,-.030112123116850853},side==0?V3{.010582463815553188,-.014423437416553497,.012788429856300354}:V3{.012648583389818668,-.015249849297106266,-.009725012816488743});
  else if(n=="ischiofemoral") witness(side==0?V3{-.024657,.039298,-.009126}:V3{-.024690,.039468,.009012},side==0?V3{-.018644,-.023469,-.027584}:V3{-.018456,-.023710,.027856});
  else if(n=="superior_glenohumeral_capsuloligament") witness(side==0?V3{.007663650700963652,-.001221965494529854,.005317389265646196}:V3{.008435920554484111,-.001735097926967254,-.0082541712677735},side==0?V3{.024076873387642927,-.0056162956963085,.001631718179675269}:V3{.02220140408607163,-.004828616297396103,-.003245167447768914});
  else if(n=="coracohumeral_capsuloligament") witness(side==0?V3{.0219944774023,-.0057980709895,.0368455091244}:V3{.0220194916719,-.0058712032429,-.0368796293892},side==0?V3{.0167655609548,.0007808510563,.0135440574959}:V3{.0162193756551,.0005840471131,-.0131858745590});
  else if(n=="anterior_talofibular_superior") witness(side==0?V3{.003912,.003813,-.022334}:V3{.004979,.003812,.023618},side==0?V3{.009882,-.000752,-.015468}:V3{.014512,.001264,.014345});
  else if(n=="anterior_talofibular_inferior") witness(side==0?V3{.0017184370663,.0012525668135,-.0233985316008}:V3{.0040541850030,.0028941668570,.0245388466865},side==0?V3{.0074792266823,-.0056220921688,-.0168760288507}:V3{.0134015027434,-.0014928911114,.0152084706351});
  else if(n=="calcaneofibular") witness(side==0?V3{-.008130,-.006293,-.023612}:V3{-.008661,-.007684,.024949},side==0?V3{-.015050,-.022054,-.017260}:V3{-.015271,-.023844,.018228});
  else if(n=="posterior_talofibular") witness(side==0?V3{-.011375,-.007257,-.021888}:V3{-.010201,-.003836,.020461},side==0?V3{-.010239,-.005716,-.011190}:V3{-.010693,-.004244,.011173});
  else if(n=="anterior_tibiotalar") witness(side==0?V3{.012921,.014373,.021524}:V3{.014198,.013982,-.022502},side==0?V3{.014191,.001188,.016064}:V3{.010697,-.000925,-.017029});
  else if(n=="tibiocalcaneal") witness(side==0?V3{.0027626773808151484,.01667203940451145,.023803692311048508}:V3{.001981951529160142,.015796242281794548,-.02046188712120056},side==0?V3{.00022980802168603987,-.0142558254301548,.019495569169521332}:V3{.0000623340965830721,-.014136694371700287,-.02014334686100483});
  else if(n=="posterior_tibiotalar") witness(side==0?V3{-.002293,.020198,.018242}:V3{-.003675,.020437,-.014791},side==0?V3{-.010004,.002604,.013728}:V3{-.011908,.003734,-.009455});
  else if(n=="elbow_rcl") witness(side==0?V3{.004511761,-.003844726,-.028748974}:V3{.005582781,-.000521388,.028226875},side==0?V3{-.001601858,-.008539562,-.027125230}:V3{.003317729,-.007840753,.027638573});
  v.push_back(spec);
 };
 for(unsigned side=0;side<2;++side){
  bool r=side==1;unsigned fem=r?131:145,fid=r?2:3,pid=r?18:19,hp=r?130:144;
  add("iliofemoral_lateral",Group::hipCapsule,side,128,pid,fem,fid,hp,fem,{.024,.042,.018},{.012,-.016,.028},43000,.5*53.5e-6,true,hs,hg,hl);
  add("iliofemoral_medial",Group::hipCapsule,side,128,pid,fem,fid,hp,fem,{.020,.042,-.008},{.010,-.016,-.008},39000,.5*53.5e-6,true,hs,hg,hl);
  add("pubofemoral",Group::hipCapsule,side,128,pid,fem,fid,hp,fem,{.032,.008,-.012},{.013,-.030,-.012},25000,15.2e-6,true,hs,hg,hl);
  add("ischiofemoral",Group::hipCapsule,side,128,pid,fem,fid,hp,fem,{-.028,.044,.008},{-.018,-.024,.028},35000,19.2e-6,true,hs,hg,hl);
  unsigned scap=r?34:84,hum=r?41:91,scid=r?10:11,hid=r?12:13,shp=r?40:90;
  add("superior_glenohumeral_capsuloligament",Group::shoulderCapsuloligament,side,scap,scid,hum,hid,shp,hum,{.008,0,-.008},{.018,-.010,.004},17400,11.3e-6,true,ss,sg,sl);
  add("coracohumeral_capsuloligament",Group::shoulderCapsuloligament,side,scap,scid,hum,hid,shp,hum,{.022,-.006,.037},{.017,.001,.014},36700,53.7e-6,true,ss,sg,sl);
  unsigned tib=r?136:150,tid=r?4:5,fidb=r?20:21,tal=r?137:151,taid=r?22:23,cal=r?138:152,caid=r?6:7;
  add("anterior_talofibular_superior",Group::ankle,side,tib,fidb,tal,taid,tib,tal,{.010,-.003,.025},{.016,-.003,.012},108600,10e-6,false,as,ag,al);
  add("anterior_talofibular_inferior",Group::ankle,side,tib,fidb,tal,taid,tib,tal,{.007,-.005,.025},{.012,-.008,.012},33200,10e-6,false,as,ag,al);
  add("calcaneofibular",Group::ankle,side,tib,fidb,cal,caid,tib,tal,{-.008,-.004,.025},{-.015,-.025,.015},80000,10e-6,false,as,ag,al);
  add("posterior_talofibular",Group::ankle,side,tib,fidb,tal,taid,tib,tal,{-.012,-.004,.025},{-.018,-.004,.012},82000,10e-6,false,as,ag,al);
  add("anterior_tibiotalar",Group::ankle,side,tib,tid,tal,taid,tib,tal,{.012,.012,-.020},{.012,0,-.012},122600,10e-6,false,as,ag,al);
  add("tibiocalcaneal",Group::ankle,side,tib,tid,cal,caid,tib,tal,{0,.012,-.020},{0,-.020,-.015},80000,10e-6,false,as,ag,al);
  add("posterior_tibiotalar",Group::ankle,side,tib,tid,tal,taid,tib,tal,{-.012,.012,-.020},{-.018,0,-.012},60000,10e-6,false,as,ag,al);
  unsigned humElbow=r?41:91,humid=r?12:13,ulna=r?42:92,rad=r?43:93,radid=r?16:17;
  add("elbow_rcl",Group::elbow,side,humElbow,humid,rad,radid,humElbow,ulna,{.006,-.002,.028},{.003,-.008,.028},46500,10e-6,false,es,eg,el);
 }
 return v;
}
inline Built cookReference(const Spec& s,const EngineModel& m,std::span<const MRBodyStateGPU> poses,std::span<const inferred_acl::BoneSurface> surfaces,std::span<const std::uint32_t> map={},const OwnershipPolicy& policy={}){
 detail::ownership(s,policy);auto b0=detail::mapBody(s.body0,map),b1=detail::mapBody(s.body1,map);if(b0==b1)throw std::runtime_error("ligament endpoints share rigid owner");
 auto f=detail::frame(s,m,poses,map);
 if(s.expectedFrameJoint!=MR_INVALID_INDEX&&f.index!=s.expectedFrameJoint)
  throw std::runtime_error("registered ligament source joint identity changed name="+std::string(s.name)+" expected="+std::to_string(s.expectedFrameJoint)+" actual="+std::to_string(f.index));
 auto x=detail::seed(f,s.seed0,s.side,s.seedLateralityMirrored),y=detail::seed(f,s.seed1,s.side,s.seedLateralityMirrored);auto& u=detail::surface(surfaces,b0,s.stable0,x);auto& v=detail::surface(surfaces,b1,s.stable1,y);
 if(s.requiresPinnedSourceIdentity&&(u.payloadSha256!=kReferenceBonesPayloadSha256||v.payloadSha256!=kReferenceBonesPayloadSha256))
  throw std::runtime_error("registered ligament source bones payload does not match admitted reference catalog name="+std::string(s.name)+" side="+std::to_string(s.side));
 auto p0=inferred_acl::detail::project(x,u),p1=inferred_acl::detail::project(y,v);double d0=std::sqrt(p0.d2),d1=std::sqrt(p1.d2);
 if(!std::isfinite(d0)||!std::isfinite(d1)||d0>kMaxProjectionM||d1>kMaxProjectionM)
  throw std::runtime_error("registered ligament projection rejected name="+
      std::string(s.name)+" side="+std::to_string(s.side)+
      " stable_ids="+std::to_string(s.stable0)+","+std::to_string(s.stable1)+
      " residuals_mm="+std::to_string(1000.0*d0)+","+
      std::to_string(1000.0*d1)+" bound_mm=20");
 if(b0>=poses.size()||b1>=poses.size())throw std::runtime_error("ligament endpoint body pose absent");double L=inferred_acl::detail::norm(inferred_acl::detail::sub(p1.p,p0.p));
 if(!std::isfinite(L)||L<1e-4||s.stiffnessNPerM<=0||s.areaM2<=0)throw std::runtime_error("ligament reference geometry/profile invalid");
 const V3 axis=inferred_acl::detail::mul(inferred_acl::detail::sub(p1.p,p0.p),1.0/L);
 const double facing0=inferred_acl::detail::dot(p0.normal,axis),facing1=inferred_acl::detail::dot(p1.normal,inferred_acl::detail::mul(axis,-1));
 if(!std::isfinite(facing0)||!std::isfinite(facing1)||std::min(facing0,facing1)<-1e-4){
  std::string message="registered ligament endpoint normals do not face across inferred fiber span";
  message+=" name="+std::string(s.name)+" side="+std::to_string(s.side);
  message+=" facing="+std::to_string(facing0)+","+std::to_string(facing1);
  message+=" frame_joint="+std::to_string(f.index)+" frame_pair="+
      std::to_string(f.parent)+","+std::to_string(f.child);
  message+=" frame_origin_world_m="+detail::vectorString(f.p);
  message+=" frame_q_xyzw="+std::to_string(f.q.x)+","+std::to_string(f.q.y)+","+
      std::to_string(f.q.z)+","+std::to_string(f.q.w);
  message+=" triangles="+std::to_string(p0.tri)+","+std::to_string(p1.tri);
  message+=" stable_ids="+std::to_string(u.stableId)+","+std::to_string(v.stableId);
  message+=" p0_world_m="+detail::vectorString(p0.p)+" p1_world_m="+detail::vectorString(p1.p);
  message+=" normal0_world="+detail::vectorString(p0.normal)+
      " normal1_world="+detail::vectorString(p1.normal);
  message+=" residuals_mm="+std::to_string(1000.0*d0)+","+std::to_string(1000.0*d1);
  throw std::runtime_error(message);
 }
 for(const auto& candidate:surfaces)if(detail::segmentHitsSurface(p0.p,p1.p,candidate))
  throw std::runtime_error("registered ligament straight reference path intersects registered source mesh name="+
      std::string(s.name)+" side="+std::to_string(s.side)+
      " intersected_body="+std::to_string(candidate.body)+
      " intersected_stable_id="+std::to_string(candidate.stableId));
 double c3=s.stiffnessNPerM*L/(s.areaM2*kC4),c5=c3*kC4*std::exp(kC4*(kLambdaMax-1));if(!std::isfinite(c3)||!std::isfinite(c5))throw std::runtime_error("ligament fitted constitutive coefficients nonfinite");
 Built out;auto& r=out.row;r.firstBodyIndex=b0;r.secondBodyIndex=b1;r.flags=NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE;r.reserved0=0;
 auto l0=detail::local(poses[b0],p0.p),l1=detail::local(poses[b1],p1.p);r.firstLocalPoint={float(l0[0]),float(l0[1]),float(l0[2]),0};r.secondLocalPoint={float(l1[0]),float(l1[1]),float(l1[2]),0};
 r.material={float(c3),float(kC4),float(c5),float(kLambdaMax)};r.reference={float(L),float(s.areaM2),1,0};
 auto& q=out.provenance;q.name=std::string(s.name);q.side=s.side;q.group=unsigned(s.group);q.body0=b0;q.body1=b1;q.stable0=u.stableId;q.stable1=v.stableId;q.record0=u.sourceRecord;q.record1=v.sourceRecord;q.payloadSha0=u.payloadSha256;q.payloadSha1=v.payloadSha256;q.frameJoint=f.index;q.frameParent=f.parent;q.frameChild=f.child;q.triangle0=p0.tri;q.triangle1=p1.tri;q.residual0M=d0;q.residual1M=d1;q.stiffnessNPerM=s.stiffnessNPerM;q.areaM2=s.areaM2;q.areaMeasured=s.areaMeasured;q.areaIsAssumedSplit=s.areaIsAssumedSplit;q.seedLateralityMirrored=s.seedLateralityMirrored;
 q.seedOffset0FrameM=s.seed0;q.seedOffset1FrameM=s.seed1;
 if(s.seedLateralityMirrored&&s.side==0u){q.seedOffset0FrameM[2]=-q.seedOffset0FrameM[2];q.seedOffset1FrameM[2]=-q.seedOffset1FrameM[2];}
 q.referenceBody0Pose={poses[b0].position.x,poses[b0].position.y,poses[b0].position.z,poses[b0].orientation.x,poses[b0].orientation.y,poses[b0].orientation.z,poses[b0].orientation.w};
 q.referenceBody1Pose={poses[b1].position.x,poses[b1].position.y,poses[b1].position.z,poses[b1].orientation.x,poses[b1].orientation.y,poses[b1].orientation.z,poses[b1].orientation.w};
 q.referenceFramePose={f.p[0],f.p[1],f.p[2],f.q.x,f.q.y,f.q.z,f.q.w};
 q.c3Pa=c3;q.c4=kC4;q.c5Pa=c5;q.lambdaMax=kLambdaMax;q.referenceLengthM=L;q.stiffnessSource=std::string(s.stiffnessSource);q.geometrySource=std::string(s.geometrySource);q.limitations=std::string(s.limitations);return out;
}
}
