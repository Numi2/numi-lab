#include "NumiHumanWholeBodySkinReduction.hpp"
#include "NumiHumanWholeBodySkinTransferAncestry.hpp"
#include "NumiHumanMaterialColumnSelfContact.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
static numi::matter::MaterialProgram material(std::string name,double lo,double hi){
 numi::matter::MaterialProgram m;m.name=std::move(name);m.supportedRepresentations.push_back(numi::matter::Representation::fem);
 numi::matter::Parameter p;p.name="density";p.lower=lo;p.upper=hi;p.defaultValue=(lo+hi)/2;m.parameters.push_back(p);return m;
}
int main(int argc,char** argv){
 if(argc!=3)return 64;
 RegisteredSurface s;s.name="candidate011 exact static construction diagnostic";s.provider="registered reference assembly";s.sourceMember="candidate-layers.npz";s.sourceMeshSHA256=kWholeBodySkinMechanicalLayerNPZ_SHA256;s.sourcePayloadSHA256=kWholeBodySkinSourcePayloadSHA256;s.aggregateIdentity=true;
 s.faces.assign(kWholeBodySkinFaces.begin(),kWholeBodySkinFaces.end());s.sourceFaceIndices.assign(kWholeBodySkinSourceFaces.begin(),kWholeBodySkinSourceFaces.end());s.sourceVertexIndices.assign(kWholeBodySkinSourceVertices.begin(),kWholeBodySkinSourceVertices.end());
 s.sourceEmbeddingFaceIndices.assign(kWholeBodySkinSourceEmbeddingFaces.begin(),kWholeBodySkinSourceEmbeddingFaces.end());s.sourceEmbeddingBarycentric.assign(kWholeBodySkinSourceEmbeddingBarycentric.begin(),kWholeBodySkinSourceEmbeddingBarycentric.end());s.sourceEmbeddingResidualWorldM.assign(kWholeBodySkinSourceEmbeddingResidualWorldM.begin(),kWholeBodySkinSourceEmbeddingResidualWorldM.end());
 for(std::size_t i=0;i<kWholeBodySkinSourceVertexCount;++i){
  const auto& f=kWholeBodySkinDonorFrames[i];V3 p{},l{},normal{};std::array<double,9> m{};
  for(unsigned c=0;c<3;++c){p[c]=double(f.registeredWorldPoint[c])+kWholeBodySkinMechanicalWorldDeltas[i][c];l[c]=double(f.donorLocalPoint[c])+kWholeBodySkinMechanicalDonorLocalDeltas[i][c];normal[c]=kWholeBodySkinOutwardDirections[i][c];}
  std::copy(f.worldToDonorLocalLinear.begin(),f.worldToDonorLocalLinear.end(),m.begin());
  s.worldPoints.push_back(p);s.donorLocalPoints.push_back(l);s.donorBodyIndices.push_back(f.donorBodyIndex);s.worldToDonorLocalLinear.push_back(m);s.outwardWorldNormals.push_back(normal);
 }
 PairedBoundaryLayer sl,fl;
 auto fill=[&](PairedBoundaryLayer& l,const auto& outer,const auto& inner,const auto& ol,const auto& il,const auto& ts,std::string_view prov,double volume){
  l.outerWorld.assign(outer.begin(),outer.end());l.innerWorld.assign(inner.begin(),inner.end());l.outerLocal.assign(ol.begin(),ol.end());l.innerLocal.assign(il.begin(),il.end());l.thicknessByVertexM.assign(ts.begin(),ts.end());l.thicknessProvenance=std::string(prov);l.minThicknessM=*std::min_element(ts.begin(),ts.end());l.maxThicknessM=*std::max_element(ts.begin(),ts.end());l.volumeM3=volume;
 };
 fill(sl,s.worldPoints,kWholeBodySkinInnerWorld,s.donorLocalPoints,kWholeBodySkinInnerLocal,kWholeBodySkinThicknessM,kWholeBodySkinThicknessProvenance,kWholeBodySkinLayerVolumeM3);
 fill(fl,kWholeBodySkinInnerWorld,kWholeBodyFatDeepWorld,kWholeBodySkinInnerLocal,kWholeBodyFatDeepLocal,kWholeBodyFatThicknessM,kWholeBodyFatThicknessProvenance,kWholeBodyFatLayerVolumeM3);
 SkinFatReference ref;ref.minSkinThicknessM=*std::min_element(kWholeBodySkinThicknessLowerBoundM.begin(),kWholeBodySkinThicknessLowerBoundM.end());ref.maxSkinThicknessM=*std::max_element(kWholeBodySkinThicknessUpperBoundM.begin(),kWholeBodySkinThicknessUpperBoundM.end());ref.minFatThicknessM=*std::min_element(kWholeBodyFatThicknessLowerBoundM.begin(),kWholeBodyFatThicknessLowerBoundM.end());ref.maxFatThicknessM=*std::max_element(kWholeBodyFatThicknessUpperBoundM.begin(),kWholeBodyFatThicknessUpperBoundM.end());
 numi::matter::WorldSource w;w.compactImmutableFEMSurfaces=true;w.frameTimestep=.001;w.environmentCount=1;w.materials={material("dermis",900,1100),material("adipose",850,950)};
 PreparedOrganicFEM skin,fat;std::string error;
 if(!appendRegisteredSkinFatFEMFromLayers(w,s,0,"dermis","existing exact candidate mass debit",kWholeBodySkinMassDebitKg,1,"adipose","existing exact candidate mass debit",kWholeBodyFatMassDebitKg,kWholeBodySkinThicknessM,kWholeBodyFatThicknessM,kWholeBodySkinThicknessProvenance,kWholeBodyFatThicknessProvenance,ref,sl,fl,"candidate skin","candidate fat",{1,2,3,4},skin,fat,error)){std::cerr<<"actual_candidate_from_layers=FAIL reason="<<error<<'\n';return 1;}
 double mass=0;for(const auto& n:skin.donorMassNodes)mass+=n.massKg;
 if(w.objects.size()!=1||std::abs(mass-kWholeBodySkinMassDebitKg-kWholeBodyFatMassDebitKg)>1e-7)return 2;
 std::cout<<std::setprecision(17)<<"actual_candidate_from_layers=PASS nodes="<<w.objects.front().femNodes.size()<<" tets="<<w.objects.front().tetrahedra.size()<<" mass_kg="<<mass<<" skin_mass_kg="<<skin.massDebitKg<<" fat_mass_kg="<<fat.massDebitKg<<" admission_flag="<<kWholeBodySkinLayersAdmitted<<" mechanics_run=false ancestry_qualification=false attachments_qualification=false\n";

 for(unsigned i=0;i<2;++i){auto m=numi::matter::parseMatterFile(argv[i+1]);if(!m.succeeded()){std::cerr<<"material parse failed\n";return 3;}w.materials[i]=std::move(m.material);for(auto& parameter:w.materials[i].parameters)parameter.identifiable=false;}
 auto compiled=numi::matter::compileWorld(w,numi::matter::CompileOptions{});
 if(!compiled.succeeded()){for(const auto& d:compiled.diagnostics)std::cerr<<d.message<<'\n';return 4;}
 auto& c=compiled.world;std::string layoutError;
 if(!numi::matter::validateCompiledWorldLayout(c,&layoutError)){std::cerr<<layoutError;return 5;}
 const auto binding=numi::matter::human::bindHumanThreeLayerMaterialColumnSelfContact(c,w.objects[0],0,kWholeBodySkinSourceVertexCount);
 if(!numi::matter::validateCompiledWorldLayout(c,&layoutError)){std::cerr<<layoutError;return 6;}
 double compiledMass=0;for(const auto& n:c.fem.nodes)compiledMass+=n.positionAndMass.w;
 if(std::abs(compiledMass-mass)>1e-6*mass)return 7;
 for(std::size_t i=0;i<c.fem.topologyNodes.size();++i)if(numi::matter::human::materialColumnTag(c.fem.topologyNodes[i].identity.w)!=i%kWholeBodySkinSourceVertexCount+1)return 8;
 std::cout<<"actual_material_compile=PASS nodes="<<c.fem.nodes.size()<<" tets="<<c.fem.tetrahedra.size()<<" mass_kg="<<compiledMass<<" material_column_self_contact=bound columns="<<binding.columnCount<<" physical_run=false\n";
 return 0;
}
