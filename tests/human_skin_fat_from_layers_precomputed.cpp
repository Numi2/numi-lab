#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanMaterialColumnSelfContact.hpp"
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace nm = numi::matter;
using namespace metalrobo::human::coarse_tissue;
static bool near(double a,double b,double e=1e-11){return std::abs(a-b)<=e*std::max({1.0,std::abs(a),std::abs(b)});}
static RegisteredSurface octa(double scale=1.0){
 RegisteredSurface s;s.name="registered octahedral skin fixture";s.provider="fixture";s.sourceMember="registered-surface";s.sourceMeshSHA256=std::string(64,'a');s.sourcePayloadSHA256=std::string(64,'b');s.aggregateIdentity=true;
 s.worldPoints={{scale,0,0},{-scale,0,0},{0,scale,0},{0,-scale,0},{0,0,scale},{0,0,-scale}};
 s.sourceVertexIndices={101,103,107,109,113,127};s.sourceFaceIndices={211,223,227,229,233,239,241,251};
 for(int sx:{1,-1})for(int sy:{1,-1})for(int sz:{1,-1}){
  const std::uint32_t x=sx>0?0:1,y=sy>0?2:3,z=sz>0?4:5;Face f{x,y,z};
  const V3 centroid=mul(add(add(s.worldPoints[f[0]],s.worldPoints[f[1]]),s.worldPoints[f[2]]),1.0/3.0);
  if(dot(cross(sub(s.worldPoints[f[1]],s.worldPoints[f[0]]),sub(s.worldPoints[f[2]],s.worldPoints[f[0]])),centroid)<0)std::swap(f[1],f[2]);
  s.faces.push_back(f);
 }
 const std::array<double,9> id{1,0,0,0,1,0,0,0,1},rot{0,1,0,-1,0,0,0,0,1};
 const std::array<V3,6> offsets{V3{.1,.2,.3},V3{-.2,.1,.4},V3{.3,-.1,.2},V3{.2,.3,-.1},V3{-.1,.2,.1},V3{.3,.1,-.2}};
 for(std::size_t i=0;i<s.worldPoints.size();i++){
  const auto&m=i%2?rot:id;s.worldToDonorLocalLinear.push_back(m);s.donorBodyIndices.push_back(i%2?20:19);const auto p=s.worldPoints[i];
  s.donorLocalPoints.push_back({m[0]*p[0]+m[1]*p[1]+m[2]*p[2]+offsets[i][0],m[3]*p[0]+m[4]*p[1]+m[5]*p[2]+offsets[i][1],m[6]*p[0]+m[7]*p[1]+m[8]*p[2]+offsets[i][2]});
  s.outwardWorldNormals.push_back(mul(p,1.0/norm(p)));
 }
 return s;
}
static numi::matter::MaterialProgram material(std::string name,double lo,double hi){numi::matter::MaterialProgram m;m.name=std::move(name);m.supportedRepresentations.push_back(numi::matter::Representation::fem);numi::matter::Parameter p;p.name="density";p.lower=lo;p.upper=hi;p.defaultValue=(lo+hi)/2;m.parameters.push_back(p);return m;}
int main(){
 std::string e;auto s=octa();SkinFatReference ref;std::vector<double> st{.0018,.0021,.0024,.0028,.0031,.0034},ft{.003,.006,.009,.013,.016,.019};std::string sp="caller supplied test field, no measured-tissue claim",fp=sp;
 PairedBoundaryLayer sl,fl;if(!deriveRegisteredSkinFatLayers(s,ref,st,ft,sp,fp,sl,fl,e)){std::cerr<<e<<"\n";return 1;}
 if(sl.thicknessByVertexM!=st||fl.thicknessByVertexM!=ft||sl.thicknessM!=0||fl.thicknessM!=0||sl.minThicknessM!=.0018||sl.maxThicknessM!=.0034||fl.minThicknessM!=.003||fl.maxThicknessM!=.019||sl.thicknessProvenance!=sp||fl.thicknessProvenance!=fp)return 2;
 double mapErr=0;for(std::size_t i=0;i<s.worldPoints.size();i++){const auto&m=s.worldToDonorLocalLinear[i];auto map=[&](V3 v){return V3{m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]};};mapErr=std::max(mapErr,norm(sub(sub(sl.innerLocal[i],sl.outerLocal[i]),map(sub(sl.innerWorld[i],sl.outerWorld[i])))));mapErr=std::max(mapErr,norm(sub(sub(fl.innerLocal[i],fl.outerLocal[i]),map(sub(fl.innerWorld[i],fl.outerWorld[i])))));}
 if(mapErr>1e-12)return 3;
 numi::matter::WorldSource w;w.compactImmutableFEMSurfaces=true;w.frameTimestep=0.001;w.environmentCount=1;w.materials={material("dermis",900,1300),material("adipose",700,1100)};std::array<std::uint64_t,4> id{1,2,3,4};PreparedOrganicFEM ps,pf;double sd=sl.volumeM3*1120,fd=fl.volumeM3*880;
 // A positive-volume shell can still be numerically unusable at Float32:
 // 1 um layer on a 1 m octahedron. Its two boundaries are individually valid.
 {
  PairedBoundaryLayer thin=sl;thin.innerWorld.clear();thin.innerLocal.clear();
  double volume=0.0;
  for(std::size_t i=0;i<s.worldPoints.size();++i){
   const auto d=mul(s.outwardWorldNormals[i],-1e-6);
   thin.innerWorld.push_back(add(thin.outerWorld[i],d));
   const auto& m=s.worldToDonorLocalLinear[i];
   thin.innerLocal.push_back(add(thin.outerLocal[i],V3{m[0]*d[0]+m[1]*d[1]+m[2]*d[2],m[3]*d[0]+m[4]*d[1]+m[5]*d[2],m[6]*d[0]+m[7]*d[1]+m[8]*d[2]}));
  }
  for(auto f:s.faces){std::sort(f.begin(),f.end());auto a=f[0],b=f[1],c=f[2];
   volume+=std::abs(tetVolume(thin.outerWorld[a],thin.outerWorld[b],thin.outerWorld[c],thin.innerWorld[c]))+
     std::abs(tetVolume(thin.outerWorld[a],thin.outerWorld[b],thin.innerWorld[b],thin.innerWorld[c]))+
     std::abs(tetVolume(thin.outerWorld[a],thin.innerWorld[a],thin.innerWorld[b],thin.innerWorld[c]));
  }
  thin.volumeM3=volume;PreparedShellFEM rejected;std::string witness;
  if(preparePairedBoundaryFEM(w,s,thin,0,"dermis","sliver regression",volume*1120,"thin",0,false,rejected,witness)||
     witness.find("ill-conditioned")==std::string::npos||!w.objects.empty())return 24;
  std::cout<<"positive_volume_float32_sliver=REJECTED source_witness=PASS\n";
 }
 if(!appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,st,ft,sp,fp,ref,sl,fl,"skin","fat",id,ps,pf,e)){std::cerr<<e<<"\n";return 4;}
 auto badVolume=sl;badVolume.volumeM3*=2.0;PreparedOrganicFEM rejectedSkin,rejectedFat;std::string rejectedError;
 if(appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,st,ft,sp,fp,ref,badVolume,fl,"bad skin","bad fat",id,rejectedSkin,rejectedFat,rejectedError)||w.objects.size()!=1u||rejectedError.find("volume")==std::string::npos)return 18;
 auto badRef=ref;badRef.maxFatThicknessM=std::numeric_limits<double>::quiet_NaN();rejectedError.clear();
 if(appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,st,ft,sp,fp,badRef,sl,fl,"bad skin","bad fat",id,rejectedSkin,rejectedFat,rejectedError)||w.objects.size()!=1u)return 19;
 auto badSummary=sl;badSummary.minThicknessM=std::numeric_limits<double>::quiet_NaN();rejectedError.clear();
 if(appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,st,ft,sp,fp,ref,badSummary,fl,"bad skin","bad fat",id,rejectedSkin,rejectedFat,rejectedError)||w.objects.size()!=1u)return 22;
 auto badFrame=s;badFrame.worldToDonorLocalLinear[0][0]=std::numeric_limits<double>::quiet_NaN();rejectedError.clear();
 if(appendRegisteredSkinFatFEMFromLayers(w,badFrame,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,st,ft,sp,fp,ref,sl,fl,"bad skin","bad fat",id,rejectedSkin,rejectedFat,rejectedError)||w.objects.size()!=1u||rejectedError.find("frame")==std::string::npos)return 20;
 auto reversedSkin=sl,reversedFat=fl;
 for(std::size_t i=0;i<s.worldPoints.size();++i){
  reversedSkin.innerWorld[i]=add(sl.outerWorld[i],sub(sl.outerWorld[i],sl.innerWorld[i]));
  reversedSkin.innerLocal[i]=add(sl.outerLocal[i],sub(sl.outerLocal[i],sl.innerLocal[i]));
  reversedFat.outerWorld[i]=reversedSkin.innerWorld[i];
  reversedFat.outerLocal[i]=reversedSkin.innerLocal[i];
  reversedFat.innerWorld[i]=add(reversedFat.outerWorld[i],sub(fl.outerWorld[i],fl.innerWorld[i]));
  reversedFat.innerLocal[i]=add(reversedFat.outerLocal[i],sub(fl.outerLocal[i],fl.innerLocal[i]));
 }
 rejectedError.clear();
 if(appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,st,ft,sp,fp,ref,reversedSkin,reversedFat,"reversed skin","reversed fat",id,rejectedSkin,rejectedFat,rejectedError)||w.objects.size()!=1u||rejectedError.find("orientation")==std::string::npos)return 21;
 auto foldedSkin=sl,foldedFat=fl;auto foldedSkinField=st;
 const std::size_t foldedVertex=5;foldedSkinField[foldedVertex]=0.0018;
 foldedSkin.innerWorld[foldedVertex]=add(s.worldPoints[foldedVertex],mul(s.outwardWorldNormals[foldedVertex],foldedSkinField[foldedVertex]));
 const auto& fm=s.worldToDonorLocalLinear[foldedVertex];const auto foldedDelta=sub(foldedSkin.innerWorld[foldedVertex],s.worldPoints[foldedVertex]);
 foldedSkin.innerLocal[foldedVertex]=add(s.donorLocalPoints[foldedVertex],V3{fm[0]*foldedDelta[0]+fm[1]*foldedDelta[1]+fm[2]*foldedDelta[2],fm[3]*foldedDelta[0]+fm[4]*foldedDelta[1]+fm[5]*foldedDelta[2],fm[6]*foldedDelta[0]+fm[7]*foldedDelta[1]+fm[8]*foldedDelta[2]});
 foldedSkin.thicknessByVertexM=foldedSkinField;foldedSkin.minThicknessM=*std::min_element(foldedSkinField.begin(),foldedSkinField.end());foldedSkin.maxThicknessM=*std::max_element(foldedSkinField.begin(),foldedSkinField.end());
 for(std::size_t i=0;i<s.worldPoints.size();++i){
  foldedFat.outerWorld[i]=foldedSkin.innerWorld[i];foldedFat.outerLocal[i]=foldedSkin.innerLocal[i];
  const auto normal=s.outwardWorldNormals[i];foldedFat.innerWorld[i]=sub(foldedFat.outerWorld[i],mul(normal,ft[i]));
  const auto& m=s.worldToDonorLocalLinear[i];const auto d=sub(foldedFat.innerWorld[i],foldedFat.outerWorld[i]);
  foldedFat.innerLocal[i]=add(foldedFat.outerLocal[i],V3{m[0]*d[0]+m[1]*d[1]+m[2]*d[2],m[3]*d[0]+m[4]*d[1]+m[5]*d[2],m[6]*d[0]+m[7]*d[1]+m[8]*d[2]});
 }
 auto recomputeVolume=[&](const std::vector<V3>& outer,const std::vector<V3>& inner){double v=0;for(auto f:s.faces){std::sort(f.begin(),f.end());auto a=f[0],b=f[1],c=f[2];v+=std::abs(tetVolume(outer[a],outer[b],outer[c],inner[c]))+std::abs(tetVolume(outer[a],outer[b],inner[b],inner[c]))+std::abs(tetVolume(outer[a],inner[a],inner[b],inner[c]));}return v;};
 foldedSkin.volumeM3=recomputeVolume(foldedSkin.outerWorld,foldedSkin.innerWorld);foldedFat.volumeM3=recomputeVolume(foldedFat.outerWorld,foldedFat.innerWorld);
 rejectedError.clear();
 if(appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","fixture law",sd,1,"adipose","fixture law",fd,foldedSkinField,ft,sp,fp,ref,foldedSkin,foldedFat,"folded skin","folded fat",id,rejectedSkin,rejectedFat,rejectedError)||w.objects.size()!=1u||rejectedError.find("orientation")==std::string::npos)return 23;
 std::cout<<"precomputed_layers=PASS exactfit_reused=PASS nan_summary_rejected=PASS reversed_winding_rejected=PASS local_prism_orientation_rejected=PASS transactionally\n";
 const auto&o=w.objects[0];if(o.femNodes.size()!=18||o.tetrahedra.size()!=48||o.femMaterialIndices.size()!=48||!std::all_of(o.femMaterialIndices.begin(),o.femMaterialIndices.begin()+24,[](auto x){return x==0;})||!std::all_of(o.femMaterialIndices.begin()+24,o.femMaterialIndices.end(),[](auto x){return x==1;}))return 5;
 if(ps.layerThicknessBySurfaceVertexM!=st||pf.layerThicknessBySurfaceVertexM!=ft||ps.layerThicknessProvenance!=sp||pf.layerThicknessProvenance!=fp||ps.layerThicknessMinM!=.0018||ps.layerThicknessMaxM!=.0034||pf.layerThicknessMinM!=.003||pf.layerThicknessMaxM!=.019||!near(ps.volumeM3,sl.volumeM3)||!near(pf.volumeM3,fl.volumeM3)||!near(ps.massDebitKg,sd)||!near(pf.massDebitKg,fd)||!near(ps.densityKgM3,1120)||!near(pf.densityKgM3,880))return 6;
 double mass=0;for(auto& row:ps.donorMassNodes)mass+=row.massKg;if(!pf.donorMassNodes.empty()||!near(mass,sd+fd)||std::abs(ps.massResidualKg)>1e-12*sd||std::abs(pf.massResidualKg)>1e-12*fd)return 7;
 std::vector<double> sc(6,ref.skinThicknessM),fc(6,.006);PairedBoundaryLayer a,b,c,d;
 if(!deriveRegisteredSkinFatLayers(s,ref,.006,a,b,e)||!deriveRegisteredSkinFatLayers(s,ref,sc,fc,ref.skinEvidence,ref.fatEvidence,c,d,e))return 8;
 if(a.outerWorld!=c.outerWorld||a.innerWorld!=c.innerWorld||a.innerLocal!=c.innerLocal||a.volumeM3!=c.volumeM3||b.innerWorld!=d.innerWorld||b.innerLocal!=d.innerLocal||b.volumeM3!=d.volumeM3||a.thicknessM!=c.thicknessM||b.thicknessM!=d.thicknessM)return 9;
 numi::matter::WorldSource w0,w1;w0.materials={material("dermis",900,1300),material("adipose",700,1100)};w1.materials=w0.materials;PreparedOrganicFEM a0,b0,a1,b1;double sd0=a.volumeM3*1120,fd0=b.volumeM3*880;
 if(!appendRegisteredSkinFatFEM(w0,s,0,"dermis","law",sd0,1,"adipose","law",fd0,.006,ref,"skin","fat",id,a0,b0,e)||!appendRegisteredSkinFatFEM(w1,s,0,"dermis","law",sd0,1,"adipose","law",fd0,sc,fc,ref.skinEvidence,ref.fatEvidence,ref,"skin","fat",id,a1,b1,e))return 10;
 if(w0.objects[0].femNodes!=w1.objects[0].femNodes||w0.objects[0].femMaterialIndices!=w1.objects[0].femMaterialIndices||w0.objects[0].femContactNodes!=w1.objects[0].femContactNodes||w0.objects[0].tetrahedra.size()!=w1.objects[0].tetrahedra.size()||a0.volumeM3!=a1.volumeM3||b0.volumeM3!=b1.volumeM3||a0.massDebitKg!=a1.massDebitKg||b0.massDebitKg!=b1.massDebitKg||a0.layerThicknessBySurfaceVertexM!=std::vector<double>(6,ref.skinThicknessM)||b0.layerThicknessBySurfaceVertexM!=std::vector<double>(6,.006)||a0.layerThicknessMinM!=ref.skinThicknessM||a0.layerThicknessMaxM!=ref.skinThicknessM||b0.layerThicknessMinM!=.006||b0.layerThicknessMaxM!=.006||a0.nodeSourceVertex!=a1.nodeSourceVertex||b0.nodeSourceVertex!=b1.nodeSourceVertex||a0.tetSourceFace!=a1.tetSourceFace||b0.tetSourceFace!=b1.tetSourceFace)return 11;
 for(std::size_t i=0;i<w0.objects[0].tetrahedra.size();i++)if(w0.objects[0].tetrahedra[i].nodes!=w1.objects[0].tetrahedra[i].nodes)return 15;
 if(a0.donorMassNodes.size()!=a1.donorMassNodes.size())return 16;
 for(std::size_t i=0;i<a0.donorMassNodes.size();i++){const auto&x=a0.donorMassNodes[i];const auto&y=a1.donorMassNodes[i];if(x.nodeIndex!=y.nodeIndex||x.donorBody!=y.donorBody||x.massKg!=y.massKg||x.localPosition!=y.localPosition)return 17;}
 auto rejects=[&](auto x,auto y){PairedBoundaryLayer q,r;std::string why;return !deriveRegisteredSkinFatLayers(s,ref,x,y,sp,fp,q,r,why);};
 if(!rejects(std::vector<double>{.002,.002,.002},std::vector<double>(6,.01))||!rejects(std::vector<double>{.002,std::numeric_limits<double>::quiet_NaN(),.002,.002,.002,.002},std::vector<double>(6,.01))||!rejects(std::vector<double>{.002,.002,.002,.004,.002,.002},std::vector<double>(6,.01)))return 12;
 auto tiny=octa(.006);std::vector<double> maxSkin(6,ref.maxSkinThicknessM),maxFat(6,ref.maxFatThicknessM);PairedBoundaryLayer q,r;std::string why;
 if(deriveRegisteredSkinFatLayers(tiny,ref,maxSkin,maxFat,sp,fp,q,r,why))return 13;
 auto bad=s;bad.worldToDonorLocalLinear[0]={1,0,0,0,1,0,0,0,0};if(deriveRegisteredSkinFatLayers(bad,ref,st,ft,sp,fp,q,r,why))return 14;
 const auto materialPath=std::filesystem::path(__FILE__).parent_path()/"fixtures/material_column_silicone.nmatter"; auto parsed=nm::parseMatterFile(materialPath.string()); if(!parsed.succeeded())throw std::runtime_error("fixture material parse failed"); auto skinMaterial=parsed.material; auto fatMaterial=parsed.material; skinMaterial.name="synthetic test dermis"; fatMaterial.name="synthetic test adipose"; w.materials={skinMaterial,fatMaterial}; for(auto& m:w.materials)for(auto& parameter:m.parameters)parameter.identifiable=false; auto compiled=nm::compileWorld(w,nm::CompileOptions{});
 std::string compileError;for(const auto& d:compiled.diagnostics)compileError+=d.message+"; ";
 if(!compiled.succeeded())throw std::runtime_error("actual compileWorld failed: "+compileError);
 auto& cooked=compiled.world;std::string layoutError;
 if((cooked.dispatch.flags&NM_MATTER_STATIC_FEM_SURFACE_MAP)==0u)throw std::runtime_error("compileWorld did not create compact immutable FEM surface map");
 if(!nm::validateCompiledWorldLayout(cooked,&layoutError))throw std::runtime_error("prebind layout invalid: "+layoutError);
 if(cooked.objects.size()!=1u||w.objects.size()!=1u||w.objects[0].femNodes.size()!=18u||w.objects[0].tetrahedra.size()!=48u)throw std::runtime_error("expected one cooked 18-node, 48-tet three-layer octa shell");
 if(w.objects[0].deformableSelfContact||!w.objects[0].deformableContact)throw std::runtime_error("test shell must be authored contact-enabled with self-contact disabled");
 const auto oldNodes=cooked.fem.nodes;const auto oldTopology=cooked.fem.topologyNodes;
 const auto oldTets=cooked.fem.tetrahedra;const auto oldFaces=cooked.fem.surfaceFaces;
 const auto oldObject=cooked.objects[0];const auto oldPhysics=cooked.physicsFingerprint;const auto oldFingerprint=cooked.fingerprint;
 double massBefore=0.0;for(std::uint32_t i=0;i<oldObject.stateCount;i++)massBefore+=cooked.fem.nodes[oldObject.stateOffset+i].positionAndMass.w;
 const auto audit=numi::matter::human::bindHumanThreeLayerMaterialColumnSelfContact(cooked,w.objects[0],0u,6u);
 if(audit.nodeCount!=18u||audit.columnCount!=6u||audit.priorPhysicsFingerprint!=oldPhysics||audit.physicsFingerprint==oldPhysics||audit.compiledFingerprint==oldFingerprint)throw std::runtime_error("binder audit/fingerprint update mismatch");
 if(!nm::validateCompiledWorldLayout(cooked,&layoutError))throw std::runtime_error("postbind layout invalid: "+layoutError);
 auto bytesEqual=[](const auto& a,const auto& b){using T=typename std::decay_t<decltype(a)>::value_type;return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);};
 if(!bytesEqual(oldNodes,cooked.fem.nodes)||!bytesEqual(oldTets,cooked.fem.tetrahedra)||!bytesEqual(oldFaces,cooked.fem.surfaceFaces))throw std::runtime_error("binding changed cooked node values/masses, tetrahedra, or compact faces");
 if(cooked.objects[0].stateOffset!=oldObject.stateOffset||cooked.objects[0].stateCount!=oldObject.stateCount||cooked.objects[0].elementOffset!=oldObject.elementOffset||cooked.objects[0].elementCount!=oldObject.elementCount)throw std::runtime_error("binding changed compiled object ranges");
 if((cooked.objects[0].flags&NM_OBJECT_DISABLE_SELF_CONTACT)!=0u||(oldObject.flags&NM_OBJECT_DISABLE_SELF_CONTACT)==0u||(cooked.objects[0].flags|NM_OBJECT_DISABLE_SELF_CONTACT)!=oldObject.flags)throw std::runtime_error("binding changed unexpected object flags");
 std::uint32_t tagged=0u;for(std::size_t i=0;i<oldTopology.size();++i){auto expected=oldTopology[i];if(i>=oldObject.stateOffset&&i<oldObject.stateOffset+oldObject.stateCount){const auto local=static_cast<std::uint32_t>(i-oldObject.stateOffset);expected.identity.w=(expected.identity.w&numi::matter::human::kTopologyLowFlagMask)|((local%6u+1u)<<numi::matter::human::kMaterialColumnFlagShift);if(numi::matter::human::materialColumnTag(cooked.fem.topologyNodes[i].identity.w)!=local%6u+1u)throw std::runtime_error("wrong material column tag");++tagged;}if(std::memcmp(&expected,&cooked.fem.topologyNodes[i],sizeof(expected))!=0)throw std::runtime_error("binding altered an unexpected topology field");}
 if(tagged!=18u)throw std::runtime_error("expected exactly 18 tagged topology nodes");
 double massAfter=0.0;for(std::uint32_t i=0;i<cooked.objects[0].stateCount;i++)massAfter+=cooked.fem.nodes[cooked.objects[0].stateOffset+i].positionAndMass.w;
 if(massAfter!=massBefore)throw std::runtime_error("binding changed shell total mass");
 auto unsupported=w.objects[0];unsupported.deformableSelfContact=true;auto rejectedWorld=cooked;bool rejected=false;
 try{(void)numi::matter::human::bindHumanThreeLayerMaterialColumnSelfContact(rejectedWorld,unsupported,0u,6u);}catch(const std::invalid_argument&){rejected=true;}
 if(!rejected||rejectedWorld.fingerprint!=cooked.fingerprint||!bytesEqual(rejectedWorld.fem.topologyNodes,cooked.fem.topologyNodes))throw std::runtime_error("invalid source was not rejected transactionally");
 std::cout<<"compile_bind_world=PASS nodes="<<cooked.dispatch.femNodeCount<<" tets="<<cooked.dispatch.tetrahedronCount<<" compact_faces="<<cooked.dispatch.surfaceFaceCount<<" tags="<<tagged<<" mass_kg="<<massAfter<<" physics_fp="<<cooked.physicsFingerprint<<" compiled_fp="<<cooked.fingerprint<<" invalid_source=REJECTED\n";
 std::cout<<"variable_fields=PASS skin_minmax_m=.0018,.0034 fat_minmax_m=.003,.019 skin_volume_m3="<<ps.volumeM3<<" fat_volume_m3="<<pf.volumeM3<<" unique_mass_residual_kg="<<ps.massResidualKg+pf.massResidualKg<<" shared_nodes=18 tets=48 donor_rotation_map_error_m="<<mapErr<<" scalar_constant_field=bitwise_equal invalid_count_nan_range_tiny_inversion_singular_map=REJECTED\n";return 0;
}
