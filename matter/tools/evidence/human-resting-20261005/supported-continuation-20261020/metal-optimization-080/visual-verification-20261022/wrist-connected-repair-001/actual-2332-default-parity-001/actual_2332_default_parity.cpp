#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanWholeBodySkinReduction.hpp"
#include "NumiHumanSkinFEMIntegration.wholebody.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace ct = metalrobo::human::coarse_tissue;
using ct::Face;
using ct::V3;
using Tet = std::array<std::uint32_t,4>;

static numi::matter::MaterialProgram material(std::string name,double rho,double lo,double hi) {
    numi::matter::MaterialProgram m;
    m.name=std::move(name);
    m.supportedRepresentations.push_back(numi::matter::Representation::fem);
    numi::matter::Parameter p;
    p.name="density";p.lower=lo;p.upper=hi;p.defaultValue=rho;
    m.parameters.push_back(p);
    return m;
}
static numi::matter::WorldSource world() {
    numi::matter::WorldSource w;
    w.compactImmutableFEMSurfaces=true;w.frameTimestep=0.001;w.environmentCount=1;
    w.materials={material("human_skin_dermis_frequency_specific_effective",1000.0,900.0,1100.0),
                 material("human_subcutaneous_adipose_low_strain_effective",900.0,850.0,950.0)};
    return w;
}
static ct::RegisteredSurface surface() {
    ct::RegisteredSurface s;
    s.name="BodyParts3D coarse regional skin-fat bonded shell";
    s.provider="clean 56db9520 static whole-body reduction header";
    s.sourceMember="BodyParts3D registered whole-body skin";
    s.sourceMeshSHA256=ct::kWholeBodySkinSourcePayloadSHA256; // the frozen static reduction retains payload identity, not a separate raw mesh digest
    s.sourcePayloadSHA256=ct::kWholeBodySkinSourcePayloadSHA256;
    s.aggregateIdentity=true;
    s.worldPoints.reserve(ct::kWholeBodySkinSourceVertexCount);
    s.donorLocalPoints.reserve(ct::kWholeBodySkinSourceVertexCount);
    s.sourceVertexIndices.assign(ct::kWholeBodySkinSourceVertices.begin(),ct::kWholeBodySkinSourceVertices.end());
    s.sourceFaceIndices.assign(ct::kWholeBodySkinSourceFaces.begin(),ct::kWholeBodySkinSourceFaces.end());
    s.faces.assign(ct::kWholeBodySkinFaces.begin(),ct::kWholeBodySkinFaces.end());
    s.donorBodyIndices.reserve(ct::kWholeBodySkinSourceVertexCount);
    s.worldToDonorLocalLinear.reserve(ct::kWholeBodySkinSourceVertexCount);
    s.outwardWorldNormals.reserve(ct::kWholeBodySkinSourceVertexCount);
    for(const auto& f:ct::kWholeBodySkinDonorFrames){
        s.worldPoints.push_back({double(f.registeredWorldPoint[0]),double(f.registeredWorldPoint[1]),double(f.registeredWorldPoint[2])});
        s.donorLocalPoints.push_back({double(f.donorLocalPoint[0]),double(f.donorLocalPoint[1]),double(f.donorLocalPoint[2])});
        s.donorBodyIndices.push_back(f.donorBodyIndex);
        std::array<double,9> frame{};for(unsigned i=0;i<9;i++)frame[i]=double(f.worldToDonorLocalLinear[i]);
        s.worldToDonorLocalLinear.push_back(frame);
        s.outwardWorldNormals.push_back({double(f.registeredWorldNormal[0]),double(f.registeredWorldNormal[1]),double(f.registeredWorldNormal[2])});
    }
    return s;
}
static void layers(const ct::RegisteredSurface& s,const ct::SkinFatReference& ref,
                   ct::PairedBoundaryLayer& skin,ct::PairedBoundaryLayer& fat,
                   std::vector<double>& skinThickness,std::vector<double>& fatThickness) {
    skinThickness.assign(ct::kWholeBodySkinThicknessM.begin(),ct::kWholeBodySkinThicknessM.end());
    fatThickness.assign(ct::kWholeBodyFatThicknessM.begin(),ct::kWholeBodyFatThicknessM.end());
    skin.outerWorld=s.worldPoints;skin.outerLocal=s.donorLocalPoints;
    skin.innerWorld.assign(ct::kWholeBodySkinInnerWorld.begin(),ct::kWholeBodySkinInnerWorld.end());
    skin.innerLocal.assign(ct::kWholeBodySkinInnerLocal.begin(),ct::kWholeBodySkinInnerLocal.end());
    skin.volumeM3=ct::kWholeBodySkinLayerVolumeM3;skin.thicknessM=0.0;
    skin.thicknessByVertexM=skinThickness;skin.thicknessProvenance=std::string(ct::kWholeBodySkinThicknessProvenance);
    const auto sk=std::minmax_element(skinThickness.begin(),skinThickness.end());skin.minThicknessM=*sk.first;skin.maxThicknessM=*sk.second;
    fat.outerWorld=skin.innerWorld;fat.outerLocal=skin.innerLocal;
    fat.innerWorld.assign(ct::kWholeBodyFatDeepWorld.begin(),ct::kWholeBodyFatDeepWorld.end());
    fat.innerLocal.assign(ct::kWholeBodyFatDeepLocal.begin(),ct::kWholeBodyFatDeepLocal.end());
    fat.volumeM3=ct::kWholeBodyFatLayerVolumeM3;fat.thicknessM=0.0;
    fat.thicknessByVertexM=fatThickness;fat.thicknessProvenance=std::string(ct::kWholeBodyFatThicknessProvenance);
    const auto fk=std::minmax_element(fatThickness.begin(),fatThickness.end());fat.minThicknessM=*fk.first;fat.maxThicknessM=*fk.second;
    (void)ref;
}
static bool sameObject(const numi::matter::ObjectSource& a,const numi::matter::ObjectSource& b) {
    if(a.name!=b.name||a.representation!=b.representation||a.materialIndex!=b.materialIndex||
       a.femNodes!=b.femNodes||a.femMaterialIndices!=b.femMaterialIndices||
       a.femContactNodes!=b.femContactNodes||a.tetrahedra.size()!=b.tetrahedra.size())return false;
    for(std::size_t i=0;i<a.tetrahedra.size();i++)if(a.tetrahedra[i].nodes!=b.tetrahedra[i].nodes)return false;
    return true;
}
static bool sameMass(const std::vector<metalrobo::NumiHumanTissueMassNode>& a,
                     const std::vector<metalrobo::NumiHumanTissueMassNode>& b) {
    if(a.size()!=b.size())return false;
    for(std::size_t i=0;i<a.size();i++){
        if(a[i].nodeIndex!=b[i].nodeIndex||a[i].donorBody!=b[i].donorBody||std::abs(a[i].massKg-b[i].massKg)>1e-12*std::max({1.0,std::abs(a[i].massKg),std::abs(b[i].massKg)})||
           ct::norm(ct::sub(a[i].localPosition,b[i].localPosition))>1e-12)return false;
    }
    return true;
}
int main() {
    auto s=surface();
    ct::RegisteredSkinMechanicalDeltaAudit mechanicalAudit;std::string mechanicalError;
    if(!ct::applyRegisteredSkinMechanicalDeltas(s,ct::kWholeBodySkinSourceVertices,
        ct::kWholeBodySkinMechanicalWorldDeltas,ct::kWholeBodySkinMechanicalDonorLocalDeltas,
        ct::kWholeBodySkinMechanicalCapClass,ct::kWholeBodySkinHeadHandDisplacementCapM,
        ct::kWholeBodySkinBodyDisplacementCapM,mechanicalAudit,mechanicalError)){
        std::cerr<<"frozen mechanical repair rejected: "<<mechanicalError<<"\n";return 12;
    }
    if(s.worldPoints.size()!=2332||s.faces.size()!=4619||s.sourceVertexIndices.size()!=2332||s.sourceFaceIndices.size()!=4619)return 1;
    ct::SkinFatReference ref;
    ref.minSkinThicknessM=ct::kWholeBodySkinMinThicknessM;ref.maxSkinThicknessM=ct::kWholeBodySkinMaxThicknessM;
    ref.minFatThicknessM=ct::kWholeBodyFatMinThicknessM;ref.maxFatThicknessM=ct::kWholeBodyFatMaxThicknessM;
    ct::PairedBoundaryLayer skinLayer,fatLayer;std::vector<double> skinThickness,fatThickness;
    layers(s,ref,skinLayer,fatLayer,skinThickness,fatThickness);
    const std::string skinProv(skinLayer.thicknessProvenance),fatProv(fatLayer.thicknessProvenance);
    constexpr std::uint32_t skinMaterial=0,fatMaterial=1;
    constexpr double skinDebit=ct::kWholeBodySkinMassDebitKg,fatDebit=ct::kWholeBodyFatMassDebitKg;
    const std::array<std::uint64_t,4> identity{0x56db9520u,0x2332u,0x4619u,0x004u};
    auto defaultWorld=world();ct::PreparedOrganicFEM defaultSkin,defaultFat;std::string error;
    if(!ct::appendRegisteredSkinFatFEMFromLayers(defaultWorld,s,skinMaterial,defaultWorld.materials[skinMaterial].name,
        "pinned 56db9520 skin material density/reference",skinDebit,
        fatMaterial,defaultWorld.materials[fatMaterial].name,"pinned 56db9520 fat material density/reference",fatDebit,
        skinThickness,fatThickness,skinProv,fatProv,ref,skinLayer,fatLayer,"registered skin","registered subcutaneous fat",
        identity,defaultSkin,defaultFat,error)){
        std::cerr<<"default baseline owner rejected: "<<error<<"\n";return 2;
    }
    if(defaultWorld.objects.size()!=1){std::cerr<<"default owner did not append exactly one combined object\n";return 3;}
    const auto& object=defaultWorld.objects.back();
    std::vector<Tet> childTets,parentTets;childTets.reserve(object.tetrahedra.size());parentTets.reserve(object.tetrahedra.size());
    std::vector<std::uint32_t> childMaterials=object.femMaterialIndices,parentMaterials=object.femMaterialIndices;
    std::vector<std::uint32_t> childParents(object.tetrahedra.size()),childFaces;childFaces.reserve(object.tetrahedra.size());
    std::vector<std::uint32_t> stableFaces;stableFaces.reserve(object.tetrahedra.size());
    std::size_t skinRow=0,fatRow=0;
    for(std::size_t i=0;i<object.tetrahedra.size();i++){
        const auto& tet=object.tetrahedra[i].nodes;childTets.push_back(tet);parentTets.push_back(tet);childParents[i]=std::uint32_t(i);
        const std::uint32_t stable=object.femMaterialIndices[i]==skinMaterial?defaultSkin.tetSourceFace.at(skinRow++):defaultFat.tetSourceFace.at(fatRow++);
        stableFaces.push_back(stable);
        const auto it=std::find(s.sourceFaceIndices.begin(),s.sourceFaceIndices.end(),stable);
        if(it==s.sourceFaceIndices.end()){std::cerr<<"owner emitted an unresolvable stable face identity="<<stable<<"\n";return 4;}
        childFaces.push_back(static_cast<std::uint32_t>(it-s.sourceFaceIndices.begin()));
    }
    if(skinRow!=defaultSkin.tetSourceFace.size()||fatRow!=defaultFat.tetSourceFace.size()||object.femNodes.empty())return 5;
    ct::ExplicitPairedLayerTopology topology{childTets,childMaterials,childParents,parentTets,object.femNodes,parentMaterials,childFaces,stableFaces};
    auto explicitWorld=world();ct::PreparedOrganicFEM explicitSkin,explicitFat;
    if(!ct::appendRegisteredSkinFatFEMFromLayers(explicitWorld,s,skinMaterial,explicitWorld.materials[skinMaterial].name,
        "pinned 56db9520 skin material density/reference",skinDebit,
        fatMaterial,explicitWorld.materials[fatMaterial].name,"pinned 56db9520 fat material density/reference",fatDebit,
        skinThickness,fatThickness,skinProv,fatProv,ref,skinLayer,fatLayer,"registered skin","registered subcutaneous fat",
        identity,explicitSkin,explicitFat,error,&topology)){
        std::cerr<<"explicit baseline identity topology rejected: "<<error<<"\n";return 6;
    }
    if(explicitWorld.objects.size()!=1)return 7;
    const auto& explicitObject=explicitWorld.objects.back();
    const auto exactReports=[&](const ct::PreparedOrganicFEM& a,const ct::PreparedOrganicFEM& b){
        return a.firstGlobalFEMNode==b.firstGlobalFEMNode&&a.materialIndex==b.materialIndex&&
            std::abs(a.volumeM3-b.volumeM3)<=1e-14*std::max({1.0,std::abs(a.volumeM3),std::abs(b.volumeM3)})&&
            a.massDebitKg==b.massDebitKg&&std::abs(a.densityKgM3-b.densityKgM3)<=1e-12*std::max({1.0,std::abs(a.densityKgM3),std::abs(b.densityKgM3)})&&
            std::abs(a.massResidualKg-b.massResidualKg)<=1e-12*std::max({1.0,std::abs(a.massResidualKg),std::abs(b.massResidualKg)})&&
            a.layerThicknessBySurfaceVertexM==b.layerThicknessBySurfaceVertexM&&
            a.layerThicknessProvenance==b.layerThicknessProvenance&&a.tetSourceFace==b.tetSourceFace&&
            sameMass(a.donorMassNodes,b.donorMassNodes);
    };
    if(!sameObject(object,explicitObject)||!exactReports(defaultSkin,explicitSkin)||!exactReports(defaultFat,explicitFat)){
        std::cerr<<"default/explicit mismatch object="<<sameObject(object,explicitObject)
                 <<" nodes_equal="<<(object.femNodes==explicitObject.femNodes)
                 <<" tet_count="<<object.tetrahedra.size()<<"/"<<explicitObject.tetrahedra.size()
                 <<" material_equal="<<(object.femMaterialIndices==explicitObject.femMaterialIndices)
                 <<" contacts_equal="<<(object.femContactNodes==explicitObject.femContactNodes)
                 <<" skin_report="<<exactReports(defaultSkin,explicitSkin)
                 <<" fat_report="<<exactReports(defaultFat,explicitFat)
                 <<" skin_mass="<<sameMass(defaultSkin.donorMassNodes,explicitSkin.donorMassNodes)
                 <<" fat_mass="<<sameMass(defaultFat.donorMassNodes,explicitFat.donorMassNodes)
                 <<" skinVol="<<std::setprecision(17)<<defaultSkin.volumeM3<<"/"<<explicitSkin.volumeM3
                 <<" fatVol="<<defaultFat.volumeM3<<"/"<<explicitFat.volumeM3
                 <<" skinDensity="<<defaultSkin.densityKgM3<<"/"<<explicitSkin.densityKgM3
                 <<" fatDensity="<<defaultFat.densityKgM3<<"/"<<explicitFat.densityKgM3<<"\n";return 8;
    }
    if(explicitSkin.tetChildIndex.size()!=defaultSkin.tetSourceFace.size()||explicitFat.tetChildIndex.size()!=defaultFat.tetSourceFace.size())return 9;
    for(std::size_t i=0;i<explicitSkin.tetChildIndex.size();i++)if(explicitSkin.tetChildIndex[i]!=i||explicitSkin.tetParentTetrahedron[i]!=i||explicitSkin.tetParentSurfaceFace[i]!=childFaces[i]||explicitSkin.tetParentMaterialIndex[i]!=skinMaterial||explicitSkin.tetSourceFaceIdentity[i]!=stableFaces[i])return 10;
    for(std::size_t i=0;i<explicitFat.tetChildIndex.size();i++){
        const auto combined=defaultSkin.tetSourceFace.size()+i;
        if(explicitFat.tetChildIndex[i]!=combined||explicitFat.tetParentTetrahedron[i]!=combined||explicitFat.tetParentSurfaceFace[i]!=childFaces[combined]||explicitFat.tetParentMaterialIndex[i]!=fatMaterial||explicitFat.tetSourceFaceIdentity[i]!=stableFaces[combined])return 11;
    }
    std::cout<<"actual_2332_default_explicit_owner_parity=PASS"
             <<" source_payload="<<ct::kWholeBodySkinSourcePayloadSHA256
             <<" surface_vertices="<<s.worldPoints.size()<<" surface_faces="<<s.faces.size()
             <<" object_nodes="<<object.femNodes.size()<<" object_tets="<<object.tetrahedra.size()
             <<" skin_tets="<<defaultSkin.tetSourceFace.size()<<" fat_tets="<<defaultFat.tetSourceFace.size()
             <<" skin_volume_m3="<<defaultSkin.volumeM3<<" fat_volume_m3="<<defaultFat.volumeM3
             <<" skin_mass_kg="<<defaultSkin.massDebitKg<<" fat_mass_kg="<<defaultFat.massDebitKg
             <<" skin_mass_rows="<<defaultSkin.donorMassNodes.size()<<" fat_mass_rows="<<defaultFat.donorMassNodes.size()
             <<" object_nodes_exact=1 connectivity_exact=1 material_rows_exact=1 contact_nodes_exact=1"
             <<" donor_mass_rows_exact=1 stable_face_ancestry_exact=1\n";
    return 0;
}
