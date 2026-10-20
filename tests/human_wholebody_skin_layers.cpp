#include "NumiHumanSkinFEMIntegration.wholebody.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace ct=metalrobo::human::coarse_tissue;
using namespace ct;
static V3 v3(const std::array<float,3>& a){return {a[0],a[1],a[2]};}
static numi::matter::MaterialProgram material(std::string name,double lo,double hi,double rho){
 numi::matter::MaterialProgram m;m.name=std::move(name);m.supportedRepresentations.push_back(numi::matter::Representation::fem);
 numi::matter::Parameter p;p.name="density";p.lower=lo;p.upper=hi;p.defaultValue=rho;p.identifiable=false;m.parameters.push_back(p);return m;
}
int main(){
 constexpr std::size_t n=kWholeBodySkinSourceVertices.size(),nf=kWholeBodySkinFaces.size();static_assert(n==2332&&nf==4619,"expected exact 2,332 vertex / 4,619 face reduction");
 if(std::string(kWholeBodySkinLayerCoordinatesSHA256).size()!=64u||std::string(kWholeBodySkinReductionSHA256)!="909d364267a09340ecd2038ed901efb6732b5d3b5a525f65221dbccc3cd40526"||std::string(kWholeBodySkinSourceCSV_SHA256)!="4dd63097a035f5f385712307f2336ef6fd46c964d1b93a2e85d84193ac12d327")return 5;
 if(kWholeBodySkinDonorFrames.size()!=n||kWholeBodySkinMechanicalWorldDeltas.size()!=n||kWholeBodySkinMechanicalDonorLocalDeltas.size()!=n||kWholeBodySkinOutwardDirections.size()!=n||kWholeBodySkinInnerWorld.size()!=n||kWholeBodySkinInnerLocal.size()!=n||kWholeBodyFatDeepWorld.size()!=n||kWholeBodyFatDeepLocal.size()!=n||kWholeBodySkinThicknessM.size()!=n||kWholeBodyFatThicknessM.size()!=n||kWholeBodySkinSourceFaces.size()!=nf)return 1;
 RegisteredSurface s;s.name="BodyParts3D admitted reduced whole-body skin";s.provider="BodyParts3D";s.sourceMember="registered whole-body skin";s.sourceMeshSHA256=std::string(kWholeBodySkinReductionSHA256);s.sourcePayloadSHA256=std::string(kWholeBodySkinSourcePayloadSHA256);s.aggregateIdentity=true;
 s.worldPoints.reserve(n);s.donorLocalPoints.reserve(n);s.donorBodyIndices.reserve(n);s.worldToDonorLocalLinear.reserve(n);s.outwardWorldNormals.reserve(n);
 for(std::size_t i=0;i<n;i++){const auto& f=kWholeBodySkinDonorFrames[i];s.worldPoints.push_back(v3(f.registeredWorldPoint));s.donorLocalPoints.push_back(v3(f.donorLocalPoint));s.sourceVertexIndices.push_back(kWholeBodySkinSourceVertices[i]);s.donorBodyIndices.push_back(f.donorBodyIndex);std::array<double,9> m{};for(std::size_t j=0;j<9;j++)m[j]=f.worldToDonorLocalLinear[j];s.worldToDonorLocalLinear.push_back(m);s.outwardWorldNormals.push_back(kWholeBodySkinOutwardDirections[i]);}
 s.faces.assign(kWholeBodySkinFaces.begin(),kWholeBodySkinFaces.end());s.sourceFaceIndices.assign(kWholeBodySkinSourceFaces.begin(),kWholeBodySkinSourceFaces.end());
 RegisteredSkinMechanicalDeltaAudit mechanicalAudit;std::string error;
 if(!applyRegisteredSkinMechanicalDeltas(s,kWholeBodySkinSourceVertices,kWholeBodySkinMechanicalWorldDeltas,kWholeBodySkinMechanicalDonorLocalDeltas,kWholeBodySkinMechanicalCapClass,kWholeBodySkinHeadHandDisplacementCapM,kWholeBodySkinBodyDisplacementCapM,mechanicalAudit,error)){std::cerr<<"mechanical repair rejected: "<<error<<"\n";return 2;}
 PairedBoundaryLayer skin,fat;skin.outerWorld=s.worldPoints;skin.outerLocal=s.donorLocalPoints;skin.innerWorld.assign(kWholeBodySkinInnerWorld.begin(),kWholeBodySkinInnerWorld.end());skin.innerLocal.assign(kWholeBodySkinInnerLocal.begin(),kWholeBodySkinInnerLocal.end());skin.volumeM3=kWholeBodySkinLayerVolumeM3;skin.thicknessByVertexM.assign(kWholeBodySkinThicknessM.begin(),kWholeBodySkinThicknessM.end());{auto mm=std::minmax_element(skin.thicknessByVertexM.begin(),skin.thicknessByVertexM.end());skin.minThicknessM=*mm.first;skin.maxThicknessM=*mm.second;}skin.thicknessProvenance=std::string(kWholeBodySkinThicknessProvenance);
 fat.outerWorld=skin.innerWorld;fat.outerLocal=skin.innerLocal;fat.innerWorld.assign(kWholeBodyFatDeepWorld.begin(),kWholeBodyFatDeepWorld.end());fat.innerLocal.assign(kWholeBodyFatDeepLocal.begin(),kWholeBodyFatDeepLocal.end());fat.volumeM3=kWholeBodyFatLayerVolumeM3;fat.thicknessByVertexM.assign(kWholeBodyFatThicknessM.begin(),kWholeBodyFatThicknessM.end());{auto mm=std::minmax_element(fat.thicknessByVertexM.begin(),fat.thicknessByVertexM.end());fat.minThicknessM=*mm.first;fat.maxThicknessM=*mm.second;}fat.thicknessProvenance=std::string(kWholeBodyFatThicknessProvenance);
 SkinFatReference ref;ref.minSkinThicknessM=kWholeBodySkinMinThicknessM;ref.maxSkinThicknessM=kWholeBodySkinMaxThicknessM;ref.minFatThicknessM=kWholeBodyFatMinThicknessM;ref.maxFatThicknessM=kWholeBodyFatMaxThicknessM;
 numi::matter::WorldSource w;w.compactImmutableFEMSurfaces=true;w.frameTimestep=.001;w.environmentCount=1;w.materials={material("synthetic whole-body dermis",500,1500,1000),material("synthetic whole-body fat",500,1400,900)};
 std::array<std::uint64_t,4> identity{1,2,3,4};PreparedOrganicFEM ps,pf;std::vector<double> st(skin.thicknessByVertexM),ft(fat.thicknessByVertexM);const double sd=kWholeBodySkinMassDebitKg,fd=kWholeBodyFatMassDebitKg;
 if(!appendRegisteredSkinFatFEMFromLayers(w,s,0,"synthetic whole-body dermis","CPU gate material reference",sd,1,"synthetic whole-body fat","CPU gate material reference",fd,st,ft,skin.thicknessProvenance,fat.thicknessProvenance,ref,skin,fat,"whole-body skin","whole-body fat",identity,ps,pf,error)){std::cerr<<"FromLayers rejected: "<<error<<"\n";return 2;}
 if(w.objects.size()!=1||w.objects[0].femNodes.size()!=3*n||w.objects[0].tetrahedra.size()!=6*nf||w.objects[0].femMaterialIndices.size()!=6*nf||ps.tetSourceFace.size()!=3*nf||pf.tetSourceFace.size()!=3*nf)return 3;
 if(ps.volumeM3!=skin.volumeM3||pf.volumeM3!=fat.volumeM3||ps.massDebitKg!=sd||pf.massDebitKg!=fd||std::abs(ps.massResidualKg)>1e-10*sd||std::abs(pf.massResidualKg)>1e-10*fd)return 4;
 // The frozen sites bind source identity, not a geometry-dependent ranking.
 std::set<std::uint32_t> attachmentSources;std::map<std::uint32_t,std::vector<V3>> attachmentByBody;
 const auto& object=w.objects[0];
 for(const auto& site:kWholeBodySkinDeepAttachmentSites){
  if(site.reducedVertex>=n||kWholeBodySkinSourceVertices[site.reducedVertex]!=site.sourceVertex||kWholeBodySkinDonorFrames[site.reducedVertex].donorBodyIndex!=site.donorBody||!attachmentSources.insert(site.sourceVertex).second)return 6;
  auto& previous=attachmentByBody[site.donorBody];if(previous.size()>=3u)return 6;
  for(const auto& point:previous)if(norm(sub(point,s.worldPoints[site.reducedVertex]))<.025)return 6;
  previous.push_back(s.worldPoints[site.reducedVertex]);
  const auto anchor=static_cast<std::uint32_t>(2*n+site.reducedVertex);double minimumAltitude=1.0;
  for(const auto& tet:object.tetrahedra)if(std::find(tet.nodes.begin(),tet.nodes.end(),anchor)!=tet.nodes.end()){
   const auto& ids=tet.nodes;const double sixV=6*std::abs(tetVolume(object.femNodes[ids[0]],object.femNodes[ids[1]],object.femNodes[ids[2]],object.femNodes[ids[3]]));
   for(unsigned opposite=0;opposite<4;++opposite){std::array<std::uint32_t,3> face{};unsigned slot=0;for(unsigned corner=0;corner<4;++corner)if(corner!=opposite)face[slot++]=ids[corner];
    const auto area2=norm(cross(sub(object.femNodes[face[1]],object.femNodes[face[0]]),sub(object.femNodes[face[2]],object.femNodes[face[0]])));minimumAltitude=std::min(minimumAltitude,sixV/area2);}
  }
  if(!std::isfinite(minimumAltitude)||minimumAltitude<5e-5)return 7;
 }
 if(attachmentSources.size()!=20u||attachmentByBody.size()!=7u)return 6;
 std::cout<<"frozen_source_donor_attachments=PASS sites="<<attachmentSources.size()<<" donors="<<attachmentByBody.size()<<"\n";
 std::cout<<"wholebody_FromLayers=PASS admitted_flag="<<unsigned(kWholeBodySkinLayersAdmitted)<<" vertices="<<n<<" faces="<<nf<<" nodes="<<w.objects[0].femNodes.size()<<" tets="<<w.objects[0].tetrahedra.size()<<" skin_volume_m3="<<ps.volumeM3<<" fat_volume_m3="<<pf.volumeM3<<" skin_debit_kg="<<ps.massDebitKg<<" fat_debit_kg="<<pf.massDebitKg<<" skin_field_m="<<ps.layerThicknessMinM<<","<<ps.layerThicknessMaxM<<" fat_field_m="<<pf.layerThicknessMinM<<","<<pf.layerThicknessMaxM<<" reduction_sha256="<<kWholeBodySkinReductionSHA256<<" coordinates_sha256="<<kWholeBodySkinLayerCoordinatesSHA256<<"\n";
}
