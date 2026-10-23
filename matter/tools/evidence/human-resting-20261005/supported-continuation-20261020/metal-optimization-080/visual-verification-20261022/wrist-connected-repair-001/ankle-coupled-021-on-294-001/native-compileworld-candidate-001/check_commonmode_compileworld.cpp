#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanWholeBodySkinReduction.hpp"
#include "NumiHumanMaterialColumnSelfContact.hpp"
#include "candidate_data.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;
namespace cc=commonmode_candidate;
static constexpr double kSkinMassDebitKg=3.4525707761420956;
static constexpr double kFatMassDebitKg=10.487108964465744;
static double layerVolume(const std::vector<V3>& outer,const std::vector<V3>& inner,const std::vector<Face>& faces){
 double v=0.0;for(const auto& face:faces){auto f=face;std::sort(f.begin(),f.end());auto a=f[0],b=f[1],c=f[2];
  v+=std::abs(tetVolume(outer[a],outer[b],outer[c],inner[c]));
  v+=std::abs(tetVolume(outer[a],outer[b],inner[b],inner[c]));
  v+=std::abs(tetVolume(outer[a],inner[a],inner[b],inner[c]));
 }return v;
}
static double fieldMin(const std::vector<double>& x){return *std::min_element(x.begin(),x.end());}
static double fieldMax(const std::vector<double>& x){return *std::max_element(x.begin(),x.end());}
static numi::matter::MaterialProgram parseMaterial(const char* path){
 auto r=numi::matter::parseMatterFile(path);if(!r.succeeded())throw std::runtime_error("material parse failed");
 for(auto& p:r.material.parameters)p.identifiable=false;return std::move(r.material);
}
static double signedVolumeFromRest(const numi::matter::CompiledWorld& c,const NMTetrahedronGPU& t){
 const auto id=std::array<std::uint32_t,4>{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w};
 auto p=[&](std::uint32_t i){const auto& v=c.fem.nodes[i].restAndFixed;return V3{double(v.x),double(v.y),double(v.z)};};
 return tetVolume(p(id[0]),p(id[1]),p(id[2]),p(id[3]));
}
int main(int argc,char** argv){
 if(argc!=3){std::cerr<<"expected skin and fat material paths\n";return 64;}
 constexpr std::size_t n=kWholeBodySkinSourceVertexCount;
 if(n!=2332||kWholeBodySkinFaceCount!=4619)return 2;
 RegisteredSurface s;s.name="294 ankle common-mode layer candidate";s.provider="frozen 294 BodyParts3D source";s.sourceMember="main294-commonmode-layer-candidate-001/candidate-layers.npz";s.sourceMeshSHA256=cc::layerNPZSHA256;s.sourcePayloadSHA256=cc::sourcePayloadSHA256;s.aggregateIdentity=true;
 s.faces.assign(kWholeBodySkinFaces.begin(),kWholeBodySkinFaces.end());s.sourceFaceIndices.assign(kWholeBodySkinSourceFaces.begin(),kWholeBodySkinSourceFaces.end());s.sourceVertexIndices.assign(kWholeBodySkinSourceVertices.begin(),kWholeBodySkinSourceVertices.end());
 s.worldPoints.assign(cc::outerWorld.begin(),cc::outerWorld.end());s.donorLocalPoints.assign(cc::outerLocal.begin(),cc::outerLocal.end());
 s.donorBodyIndices.assign(cc::donorBody.begin(),cc::donorBody.end());s.worldToDonorLocalLinear.reserve(n);
 double frameResidual=0.0;
 for(std::size_t i=0;i<n;i++){const auto& f=kWholeBodySkinDonorFrames[i];std::array<double,9> m{};std::copy(f.worldToDonorLocalLinear.begin(),f.worldToDonorLocalLinear.end(),m.begin());s.worldToDonorLocalLinear.push_back(m);if(f.donorBodyIndex!=cc::donorBody[i])return 15;for(unsigned a=0;a<3;a++){double pred=f.donorLocalPoint[a];for(unsigned b=0;b<3;b++)pred+=m[3*a+b]*(cc::outerWorld[i][b]-double(f.registeredWorldPoint[b]));frameResidual=std::max(frameResidual,std::abs(pred-cc::outerLocal[i][a]));}}
 if(s.worldPoints.size()!=n||s.donorLocalPoints.size()!=n||s.faces.size()!=kWholeBodySkinFaceCount||frameResidual>2e-6)return 3;
 PairedBoundaryLayer skin, fat;
 auto assign=[&](PairedBoundaryLayer& dst,const std::vector<V3>& ow,const std::vector<V3>& iw,const std::vector<V3>& ol,const std::vector<V3>& il,const auto& thickness,const char* provenance){
  dst.outerWorld=ow;dst.innerWorld=iw;dst.outerLocal=ol;dst.innerLocal=il;dst.thicknessByVertexM.assign(thickness.begin(),thickness.end());dst.thicknessProvenance=provenance;dst.minThicknessM=fieldMin(dst.thicknessByVertexM);dst.maxThicknessM=fieldMax(dst.thicknessByVertexM);dst.volumeM3=layerVolume(dst.outerWorld,dst.innerWorld,s.faces);
 };
 assign(skin,s.worldPoints,std::vector<V3>(cc::skinInnerWorld.begin(),cc::skinInnerWorld.end()),s.donorLocalPoints,std::vector<V3>(cc::skinInnerLocal.begin(),cc::skinInnerLocal.end()),cc::skinThickness,"candidate common-mode 294 skin field; exact carried O/I columns");
 assign(fat,std::vector<V3>(cc::skinInnerWorld.begin(),cc::skinInnerWorld.end()),std::vector<V3>(cc::fatDeepWorld.begin(),cc::fatDeepWorld.end()),std::vector<V3>(cc::skinInnerLocal.begin(),cc::skinInnerLocal.end()),std::vector<V3>(cc::fatDeepLocal.begin(),cc::fatDeepLocal.end()),cc::fatThickness,"candidate common-mode 294 fat field; exact carried I/D columns");
 std::vector<double> st(cc::skinThickness.begin(),cc::skinThickness.end()),ft(cc::fatThickness.begin(),cc::fatThickness.end());
 SkinFatReference ref;ref.minSkinThicknessM=kWholeBodySkinMinThicknessM;ref.maxSkinThicknessM=kWholeBodySkinMaxThicknessM;ref.minFatThicknessM=kWholeBodyFatMinThicknessM;ref.maxFatThicknessM=kWholeBodyFatMaxThicknessM;
 numi::matter::WorldSource w;w.compactImmutableFEMSurfaces=true;w.frameTimestep=.001;w.environmentCount=1;w.materials={parseMaterial(argv[1]),parseMaterial(argv[2])};
 PreparedOrganicFEM skinReport,fatReport;std::string error;
 const std::array<std::uint64_t,4> identity{0x414e4b4c45303231ull,0x434f4d4d4f4e4d4full,0x444541544543414eull,0x4449444154453031ull};
 if(!appendRegisteredSkinFatFEMFromLayers(w,s,0,w.materials[0].name,"existing source debit, exact candidate volume",kSkinMassDebitKg,1,w.materials[1].name,"existing source debit, exact candidate volume",kFatMassDebitKg,st,ft,skin.thicknessProvenance,fat.thicknessProvenance,ref,skin,fat,"294 ankle candidate skin","294 ankle candidate fat",identity,skinReport,fatReport,error)){
  std::cerr<<"from_layers=FAIL reason="<<error<<"\n";return 1;
 }
 if(w.objects.size()!=1)return 4;
 const auto& obj=w.objects.front();
 if(obj.femNodes.size()!=3*n||obj.tetrahedra.size()!=6*s.faces.size()||obj.femMaterialIndices.size()!=obj.tetrahedra.size())return 5;
 if(skinReport.donorMassNodes.size()!=3*n)return 6;
 std::map<std::uint32_t,std::pair<std::size_t,double>> owner;
 double partitionMass=0;for(const auto& row:skinReport.donorMassNodes){if(row.nodeIndex>=3*n||row.donorBody!=cc::donorBody[row.nodeIndex%n]||!std::isfinite(row.massKg)||row.massKg<=0)return 7;auto& x=owner[row.donorBody];++x.first;x.second+=row.massKg;partitionMass+=row.massKg;}
 if(std::abs(partitionMass-kSkinMassDebitKg-kFatMassDebitKg)>1e-8)return 8;
 auto density=[&](std::size_t i){for(const auto& p:w.materials[i].parameters)if(p.name=="density")return p.defaultValue;return -1.0;};
 const double rs=kSkinMassDebitKg/skin.volumeM3,rf=kFatMassDebitKg/fat.volumeM3;
 if(std::abs(density(0)-rs)>1e-12||std::abs(density(1)-rf)>1e-12)return 9;
 std::cout<<std::setprecision(17)<<"from_layers=PASS candidate_only=true nodes="<<obj.femNodes.size()<<" tets="<<obj.tetrahedra.size()<<" skin_tets="<<3*s.faces.size()<<" fat_tets="<<3*s.faces.size()<<" source_rows="<<n<<" faces="<<s.faces.size()<<"\n";
 std::cout<<"skin_volume_m3="<<skin.volumeM3<<" fat_volume_m3="<<fat.volumeM3<<" skin_mass_debit_kg="<<kSkinMassDebitKg<<" fat_mass_debit_kg="<<kFatMassDebitKg<<" skin_density_kg_m3="<<density(0)<<" fat_density_kg_m3="<<density(1)<<" source_frame_displacement_max_residual_m="<<cc::frameDisplacementMaxResidualM<<"\n";
 std::cout<<"donor_owner_rows="<<skinReport.donorMassNodes.size()<<" donor_count="<<owner.size()<<" nodal_mass_sum_kg="<<partitionMass<<" layer_columns=O/I/D\n";
 for(const auto& [body,m]:owner)std::cout<<"donor="<<body<<" owned_nodes="<<m.first<<" tissue_mass_kg="<<m.second<<"\n";
 auto compiled=numi::matter::compileWorld(w,numi::matter::CompileOptions{});if(!compiled.succeeded()){for(const auto& d:compiled.diagnostics)std::cerr<<d.message<<"\n";return 10;}
 auto& c=compiled.world;std::string layoutError;if(!numi::matter::validateCompiledWorldLayout(c,&layoutError)){std::cerr<<"layout="<<layoutError<<"\n";return 11;}
 if(c.fem.nodes.size()!=3*n||c.fem.tetrahedra.size()!=6*s.faces.size())return 12;
 double compiledMass=0,minJ=std::numeric_limits<double>::infinity(),minNormDet=std::numeric_limits<double>::infinity(),minVol=std::numeric_limits<double>::infinity(),maxVol=0;std::size_t neg=0;
 for(const auto& nd:c.fem.nodes)compiledMass+=double(nd.positionAndMass.w);
 for(const auto& t:c.fem.tetrahedra){double v=signedVolumeFromRest(c,t);if(!(std::isfinite(v)&&v>0)){++neg;continue;}const double rv=double(t.inverseRestRow0.w);if(!(std::isfinite(rv)&&rv>0)){++neg;continue;}minJ=std::min(minJ,v/rv);minVol=std::min(minVol,v);maxVol=std::max(maxVol,v);
  const std::array<std::uint32_t,4> id{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w};double le=0;for(unsigned i=0;i<4;i++)for(unsigned j=i+1;j<4;j++){const auto& a=c.fem.nodes[id[i]].restAndFixed;const auto& b=c.fem.nodes[id[j]].restAndFixed;le=std::max(le,norm(V3{double(a.x-b.x),double(a.y-b.y),double(a.z-b.z)}));}minNormDet=std::min(minNormDet,6*v/(le*le*le));
 }
 std::cerr<<std::setprecision(17)<<"compiled_diag nodes="<<c.fem.nodes.size()<<" tets="<<c.fem.tetrahedra.size()<<" neg_or_nonpositive="<<neg<<" minJ="<<minJ<<" minNormDet="<<minNormDet<<" minVol="<<minVol<<" maxVol="<<maxVol<<" compiledMass="<<compiledMass<<" expectedMass="<<partitionMass<<" massDelta="<<(compiledMass-partitionMass)<<"\n";
 if(neg||!std::isfinite(minJ)||!std::isfinite(minNormDet)||std::abs(compiledMass-partitionMass)>1e-6*partitionMass)return 13;
 const auto binding=numi::matter::human::bindHumanThreeLayerMaterialColumnSelfContact(c,obj,0u,static_cast<std::uint32_t>(n));
 if(!numi::matter::validateCompiledWorldLayout(c,&layoutError)){std::cerr<<"post_bind_layout="<<layoutError<<"\n";return 14;}
 std::cout<<"compile_world=PASS compiled_nodes="<<c.fem.nodes.size()<<" compiled_tets="<<c.fem.tetrahedra.size()<<" positive_tets="<<c.fem.tetrahedra.size()<<" min_J_over_reference="<<minJ<<" min_normalized_det="<<minNormDet<<" min_tet_volume_m3="<<minVol<<" max_tet_volume_m3="<<maxVol<<" compiled_mass_kg="<<compiledMass<<" binding_columns="<<binding.columnCount<<" gpu_execution=false simulation_steps=0 layers_admitted=false\n";
 return 0;
}
