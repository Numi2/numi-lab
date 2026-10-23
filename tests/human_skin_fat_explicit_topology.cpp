#include "NumiHumanCoarseTissuePreparation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ct = metalrobo::human::coarse_tissue;
using namespace ct;
using V3=ct::V3;
using Tet=std::array<std::uint32_t,4>;
using Matrix=std::array<std::array<double,15>,15>;

static numi::matter::MaterialProgram makeMaterial(std::string name,double lo,double hi) {
    numi::matter::MaterialProgram material;
    material.name=std::move(name);
    material.supportedRepresentations.push_back(numi::matter::Representation::fem);
    numi::matter::Parameter density;
    density.name="density";density.lower=lo;density.upper=hi;density.defaultValue=(lo+hi)*0.5;
    material.parameters.push_back(density);
    return material;
}
static ct::RegisteredSurface octa() {
    ct::RegisteredSurface s;
    s.name="registered octahedron";s.provider="topology seam fixture";s.sourceMember="octa";
    s.sourceMeshSHA256=std::string(64,'a');s.sourcePayloadSHA256=std::string(64,'b');s.aggregateIdentity=true;
    s.worldPoints={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    s.sourceVertexIndices={101,103,107,109,113,127};
    s.sourceFaceIndices={211,223,227,229,233,239,241,251};
    for(int sx:{1,-1})for(int sy:{1,-1})for(int sz:{1,-1}) {
        const std::uint32_t x=sx>0?0:1,y=sy>0?2:3,z=sz>0?4:5;
        ct::Face f{x,y,z};
        const auto centroid=ct::mul(ct::add(ct::add(s.worldPoints[f[0]],s.worldPoints[f[1]]),s.worldPoints[f[2]]),1.0/3.0);
        if(ct::dot(ct::cross(ct::sub(s.worldPoints[f[1]],s.worldPoints[f[0]]),
                             ct::sub(s.worldPoints[f[2]],s.worldPoints[f[0]])),centroid)<0)std::swap(f[1],f[2]);
        s.faces.push_back(f);
    }
    for(std::size_t i=0;i<s.worldPoints.size();++i) {
        s.donorLocalPoints.push_back(s.worldPoints[i]);s.donorBodyIndices.push_back(19u);
        s.worldToDonorLocalLinear.push_back({1,0,0,0,1,0,0,0,1});
        const auto p=s.worldPoints[i];s.outwardWorldNormals.push_back(ct::mul(p,1.0/ct::norm(p)));
    }
    return s;
}
static numi::matter::WorldSource world() {
    numi::matter::WorldSource w;
    w.compactImmutableFEMSurfaces=true;w.frameTimestep=0.001;w.environmentCount=1;
    w.materials={makeMaterial("dermis",900,1300),makeMaterial("adipose",700,1100)};
    return w;
}
static bool sameMassRows(const std::vector<metalrobo::NumiHumanTissueMassNode>& a,
                         const std::vector<metalrobo::NumiHumanTissueMassNode>& b) {
    if(a.size()!=b.size())return false;
    for(std::size_t i=0;i<a.size();++i){
        if(a[i].nodeIndex!=b[i].nodeIndex||a[i].donorBody!=b[i].donorBody||
           std::abs(a[i].massKg-b[i].massKg)>1e-12*std::max({1.0,std::abs(a[i].massKg),std::abs(b[i].massKg)}))return false;
        for(unsigned axis=0;axis<3;axis++)if(std::abs(a[i].localPosition[axis]-b[i].localPosition[axis])>1e-12)return false;
    }
    return true;
}
static bool sameObject(const numi::matter::ObjectSource& a,const numi::matter::ObjectSource& b) {
    if(a.femNodes!=b.femNodes||a.femMaterialIndices!=b.femMaterialIndices||
       a.femContactNodes!=b.femContactNodes||a.tetrahedra.size()!=b.tetrahedra.size())return false;
    for(std::size_t i=0;i<a.tetrahedra.size();++i)if(a.tetrahedra[i].nodes!=b.tetrahedra[i].nodes)return false;
    return true;
}
static double maxAbsDifference(const Matrix& a,const Matrix& b,std::size_t n) {
    double error=0;
    for(std::size_t i=0;i<3*n;i++)for(std::size_t j=0;j<3*n;j++)error=std::max(error,std::abs(a[i][j]-b[i][j]));
    return error;
}
static Matrix tetStiffness(const std::vector<V3>& points,const std::vector<Tet>& tets,double young,double poisson) {
    Matrix K{};
    const double mu=young/(2.0*(1.0+poisson));
    const double lambda=young*poisson/((1.0+poisson)*(1.0-2.0*poisson));
    std::array<std::array<double,6>,6> D{};
    for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)D[i][j]=(i==j?2.0*mu:0.0)+lambda;
    for(unsigned i=3;i<6;i++)D[i][i]=mu;
    for(const auto& tet:tets) {
        const auto p0=points[tet[0]],p1=points[tet[1]],p2=points[tet[2]],p3=points[tet[3]];
        const auto a=ct::sub(p1,p0),b=ct::sub(p2,p0),c=ct::sub(p3,p0);
        const double determinant=ct::dot(a,ct::cross(b,c));
        if(!(std::isfinite(determinant)&&determinant>0.0))return {};
        const double volume=determinant/6.0;
        const std::array<V3,3> grad123{
            ct::mul(ct::cross(b,c),1.0/determinant),
            ct::mul(ct::cross(c,a),1.0/determinant),
            ct::mul(ct::cross(a,b),1.0/determinant)};
        const std::array<V3,4> grad{
            ct::mul(ct::add(ct::add(grad123[0],grad123[1]),grad123[2]),-1.0),
            grad123[0],grad123[1],grad123[2]};
        std::array<std::array<double,12>,6> B{};
        for(unsigned i=0;i<4;i++) {
            const auto g=grad[i];const auto x=3u*i,y=x+1u,z=x+2u;
            B[0][x]=g[0];B[1][y]=g[1];B[2][z]=g[2];
            B[3][x]=g[1];B[3][y]=g[0];
            B[4][y]=g[2];B[4][z]=g[1];
            B[5][x]=g[2];B[5][z]=g[0];
        }
        std::array<std::array<double,12>,12> local{};
        for(unsigned i=0;i<12;i++)for(unsigned j=0;j<12;j++)
            for(unsigned r=0;r<6;r++)for(unsigned c2=0;c2<6;c2++)local[i][j]+=volume*B[r][i]*D[r][c2]*B[c2][j];
        for(unsigned i=0;i<4;i++)for(unsigned ci=0;ci<3;ci++)
            for(unsigned j=0;j<4;j++)for(unsigned cj=0;cj<3;cj++)
                K[3u*tet[i]+ci][3u*tet[j]+cj]+=local[3u*i+ci][3u*j+cj];
    }
    return K;
}
static bool edgeProlongationConservesP1Mechanics() {
    const std::vector<V3> parentPoints{{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const std::vector<Tet> parentTets{{0,1,2,3}};
    std::vector<V3> childPoints=parentPoints;childPoints.push_back(ct::mul(ct::add(parentPoints[0],parentPoints[1]),0.5));
    const std::vector<Tet> childTets{{0,4,2,3},{4,1,2,3}};
    const auto Kp=tetStiffness(parentPoints,parentTets,2.1e6,0.31);
    const auto Kc=tetStiffness(childPoints,childTets,2.1e6,0.31);
    std::array<std::array<double,12>,15> P{};
    for(unsigned i=0;i<12;i++)P[i][i]=1.0;
    for(unsigned axis=0;axis<3;axis++){P[12u+axis][axis]=0.5;P[12u+axis][3u+axis]=0.5;}
    Matrix projected{};
    for(unsigned i=0;i<12;i++)for(unsigned j=0;j<12;j++)
        for(unsigned r=0;r<15;r++)for(unsigned c=0;c<15;c++)projected[i][j]+=P[r][i]*Kc[r][c]*P[c][j];
    if(maxAbsDifference(Kp,projected,4)>2e-8)return false;
    std::array<double,12> q{},direction{};
    for(unsigned i=0;i<12;i++){q[i]=std::sin(0.37*double(i+1));direction[i]=std::cos(0.23*double(i+2));}
    std::array<double,15> uq{},uv{},fcq{},fcv{};
    for(unsigned i=0;i<15;i++)for(unsigned j=0;j<12;j++){uq[i]+=P[i][j]*q[j];uv[i]+=P[i][j]*direction[j];}
    for(unsigned i=0;i<15;i++)for(unsigned j=0;j<15;j++){fcq[i]+=Kc[i][j]*uq[j];fcv[i]+=Kc[i][j]*uv[j];}
    double energyParent=0,energyChild=0,maxForceError=0,maxTangentError=0;
    std::array<double,12> forceParent{},tangentParent{},forceProjected{},tangentProjected{};
    for(unsigned i=0;i<12;i++)for(unsigned j=0;j<12;j++){
        forceParent[i]+=Kp[i][j]*q[j];tangentParent[i]+=Kp[i][j]*direction[j];
        energyParent+=0.5*q[i]*Kp[i][j]*q[j];
    }
    for(unsigned i=0;i<15;i++)for(unsigned j=0;j<15;j++)energyChild+=0.5*uq[i]*Kc[i][j]*uq[j];
    for(unsigned i=0;i<12;i++)for(unsigned j=0;j<15;j++){forceProjected[i]+=P[j][i]*fcq[j];tangentProjected[i]+=P[j][i]*fcv[j];}
    for(unsigned i=0;i<12;i++){maxForceError=std::max(maxForceError,std::abs(forceParent[i]-forceProjected[i]));maxTangentError=std::max(maxTangentError,std::abs(tangentParent[i]-tangentProjected[i]));}
    const auto relative=[](double a,double b){return std::abs(a-b)/std::max({1.0,std::abs(a),std::abs(b)});};
    return relative(energyParent,energyChild)<1e-11&&maxForceError<2e-8&&maxTangentError<2e-8;
}
int main() {
    auto surface=octa();SkinFatReference reference;
    const std::vector<double> skinThickness(surface.worldPoints.size(),reference.skinThicknessM);
    const std::vector<double> fatThickness(surface.worldPoints.size(),0.006);
    const std::string skinProv="synthetic explicit-topology fixture skin",fatProv="synthetic explicit-topology fixture fat";
    PairedBoundaryLayer skinLayer,fatLayer;std::string error;
    if(!deriveRegisteredSkinFatLayers(surface,reference,skinThickness,fatThickness,skinProv,fatProv,skinLayer,fatLayer,error)){
        std::cerr<<"layer preflight: "<<error<<"\n";return 1;
    }
    const std::array<std::uint64_t,4> identity{1,2,3,4};
    auto legacyWorld=world();PreparedOrganicFEM legacySkin,legacyFat;
    if(!appendRegisteredSkinFatFEM(legacyWorld,surface,0,"dermis","fixture donor law",skinLayer.volumeM3*1120.0,
           1,"adipose","fixture donor law",fatLayer.volumeM3*880.0,skinThickness,fatThickness,skinProv,fatProv,
           reference,"skin","fat",identity,legacySkin,legacyFat,error)){std::cerr<<error<<"\n";return 2;}
    auto directWorld=world();PreparedOrganicFEM directSkin,directFat;
    if(!appendRegisteredSkinFatFEMFromLayers(directWorld,surface,0,"dermis","fixture donor law",skinLayer.volumeM3*1120.0,
           1,"adipose","fixture donor law",fatLayer.volumeM3*880.0,skinThickness,fatThickness,skinProv,fatProv,
           reference,skinLayer,fatLayer,"skin","fat",identity,directSkin,directFat,error)){std::cerr<<error<<"\n";return 3;}
    if(legacyWorld.objects.size()!=1||directWorld.objects.size()!=1||!sameObject(legacyWorld.objects[0],directWorld.objects[0])||
       legacySkin.volumeM3!=directSkin.volumeM3||legacyFat.volumeM3!=directFat.volumeM3||
       legacySkin.massDebitKg!=directSkin.massDebitKg||legacyFat.massDebitKg!=directFat.massDebitKg||
       legacySkin.donorMassNodes.size()!=directSkin.donorMassNodes.size()||
       !sameMassRows(legacySkin.donorMassNodes,directSkin.donorMassNodes))return 4;

    const auto& oldObject=legacyWorld.objects[0];
    std::vector<std::array<std::uint32_t,4>> childTets,parentTets;
    std::vector<V3> parentWorldPoints=oldObject.femNodes;
    std::vector<std::uint32_t> childMaterials=oldObject.femMaterialIndices,childParents,childFaces,childStableFaces,parentMaterials=oldObject.femMaterialIndices;
    childTets.reserve(oldObject.tetrahedra.size());parentTets.reserve(oldObject.tetrahedra.size());
    childParents.resize(childMaterials.size());childFaces.reserve(childMaterials.size());childStableFaces.reserve(childMaterials.size());
    for(std::size_t i=0;i<oldObject.tetrahedra.size();i++){
        childTets.push_back(oldObject.tetrahedra[i].nodes);parentTets.push_back(oldObject.tetrahedra[i].nodes);
        childParents[i]=static_cast<std::uint32_t>(i);
        const auto stable=i<24u?legacySkin.tetSourceFace[i]:legacyFat.tetSourceFace[i-24u];
        childStableFaces.push_back(stable);
        const auto found=std::find(surface.sourceFaceIndices.begin(),surface.sourceFaceIndices.end(),stable);
        if(found==surface.sourceFaceIndices.end())return 5;
        childFaces.push_back(static_cast<std::uint32_t>(found-surface.sourceFaceIndices.begin()));
    }
    ExplicitPairedLayerTopology topology{childTets,childMaterials,childParents,parentTets,parentWorldPoints,parentMaterials,childFaces,childStableFaces};
    auto explicitWorld=world();PreparedOrganicFEM explicitSkin,explicitFat;
    if(!appendRegisteredSkinFatFEMFromLayers(explicitWorld,surface,0,"dermis","fixture donor law",skinLayer.volumeM3*1120.0,
           1,"adipose","fixture donor law",fatLayer.volumeM3*880.0,skinThickness,fatThickness,skinProv,fatProv,
           reference,skinLayer,fatLayer,"skin","fat",identity,explicitSkin,explicitFat,error,&topology)){
        std::cerr<<"explicit topology: "<<error<<"\n";return 6;
    }
    const bool explicitParity=explicitWorld.objects.size()==1&&sameObject(legacyWorld.objects[0],explicitWorld.objects[0])&&
       legacySkin.volumeM3==explicitSkin.volumeM3&&legacyFat.volumeM3==explicitFat.volumeM3&&
       legacySkin.massDebitKg==explicitSkin.massDebitKg&&legacyFat.massDebitKg==explicitFat.massDebitKg&&
       explicitSkin.tetChildIndex.size()==24u&&explicitFat.tetChildIndex.size()==24u&&
       explicitSkin.tetParentTetrahedron.size()==24u&&explicitFat.tetParentTetrahedron.size()==24u&&
       explicitSkin.tetParentMaterialIndex.size()==24u&&explicitFat.tetParentMaterialIndex.size()==24u;
    if(!explicitParity||!sameMassRows(legacySkin.donorMassNodes,explicitSkin.donorMassNodes))return 7;
    for(std::size_t i=0;i<explicitSkin.tetParentTetrahedron.size();i++)
        if(explicitSkin.tetChildIndex[i]>=24u||explicitSkin.tetParentMaterialIndex[i]!=0u)return 8;
    for(std::size_t i=0;i<explicitFat.tetParentTetrahedron.size();i++)
        if(explicitFat.tetChildIndex[i]<24u||explicitFat.tetParentMaterialIndex[i]!=1u)return 9;

    // Test the boundary contract independently of the separate whole-surface
    // signed-volume admission gate. Reversing provenance triangle winding must
    // not reject an unchanged, positive-volume tetrahedral boundary.
    auto checkWindingContract=[&](bool fat){
        const auto n=static_cast<std::uint32_t>(surface.worldPoints.size());
        std::map<Face,unsigned> use;std::map<Face,Face> oriented;
        std::map<Face,std::uint32_t> parent;
        const unsigned facets[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
        const std::size_t begin=fat?24u:0u,end=fat?48u:24u;
        for(std::size_t ti=begin;ti<end;++ti){
            auto tet=oldObject.tetrahedra[ti].nodes;
            if(fat)for(auto& node:tet)node-=n;
            for(const auto& indices:facets){
                Face face{tet[indices[0]],tet[indices[1]],tet[indices[2]]},key=face;
                std::sort(key.begin(),key.end());++use[key];
                oriented.try_emplace(key,face);parent.try_emplace(key,childFaces[ti]);
            }
        }
        std::vector<Face> boundary;for(const auto& [key,count]:use)if(count==1u)boundary.push_back(key);
        auto reversedSource=surface.faces;for(auto& face:reversedSource)std::swap(face[1],face[2]);
        const auto& layer=fat?fatLayer:skinLayer;
        if(!validateExplicitBoundaryAreaCoverage(layer.outerWorld,layer.innerWorld,reversedSource,
                parent,oriented,boundary,n,error)){std::cerr<<error<<"\n";return false;}
        // Changing actual boundary orientation is still rejected.
        auto reversedBoundary=oriented;
        auto& first=reversedBoundary[boundary.front()];std::swap(first[1],first[2]);
        return !validateExplicitBoundaryAreaCoverage(layer.outerWorld,layer.innerWorld,reversedSource,
                parent,reversedBoundary,boundary,n,error)&&error.find("invalid area or orientation")!=std::string::npos;
    };
    if(!checkWindingContract(false)||!checkWindingContract(true))return 20;

    auto rejects=[&](ExplicitPairedLayerTopology bad,std::string_view expected){
        auto w=world();PreparedOrganicFEM a,b;std::string why;
        const auto density0=w.materials[0].parameters[0].defaultValue;
        const bool ok=appendRegisteredSkinFatFEMFromLayers(w,surface,0,"dermis","fixture donor law",skinLayer.volumeM3*1120.0,
            1,"adipose","fixture donor law",fatLayer.volumeM3*880.0,skinThickness,fatThickness,skinProv,fatProv,
            reference,skinLayer,fatLayer,"skin","fat",identity,a,b,why,&bad);
        return !ok&&w.objects.empty()&&w.materials[0].parameters[0].defaultValue==density0&&why.find(expected)!=std::string::npos;
    };
    auto badMaterial=parentMaterials;badMaterial[0]=1u;
    if(!rejects({childTets,childMaterials,childParents,parentTets,parentWorldPoints,badMaterial,childFaces,childStableFaces},"parent tetrahedron material"))return 10;
    auto badFaces=childFaces;badFaces[0]=static_cast<std::uint32_t>(surface.faces.size());
    if(!rejects({childTets,childMaterials,childParents,parentTets,parentWorldPoints,parentMaterials,badFaces,childStableFaces},"parent-surface row"))return 11;
    auto badStable=childStableFaces;badStable[0]++;
    if(!rejects({childTets,childMaterials,childParents,parentTets,parentWorldPoints,parentMaterials,childFaces,badStable},"stable source-face identity"))return 12;
    auto badTet=childTets;badTet[0][0]=static_cast<std::uint32_t>(3u*surface.worldPoints.size());
    if(!rejects({badTet,childMaterials,childParents,parentTets,parentWorldPoints,parentMaterials,childFaces,childStableFaces},"outside contiguous O/I/D"))return 13;
    auto reversed=childTets;std::swap(reversed[0][0],reversed[0][1]);
    if(!rejects({reversed,childMaterials,childParents,parentTets,parentWorldPoints,parentMaterials,childFaces,childStableFaces},"negative orientation"))return 14;
    auto shortMaterials=childMaterials;shortMaterials.pop_back();
    if(!rejects({childTets,shortMaterials,childParents,parentTets,parentWorldPoints,parentMaterials,childFaces,childStableFaces},"incomplete child/parent arrays"))return 15;
    if(!edgeProlongationConservesP1Mechanics())return 16;
    std::vector<V3> splitParentPoints{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0.5,0,0}};
    const std::array<std::array<std::uint32_t,4>,1> splitParentTets{{{{0,1,2,3}}}};
    const std::array<std::array<std::uint32_t,4>,2> splitChildren{{{{0,4,2,3}},{{4,1,2,3}}}};
    const std::array<std::uint32_t,1> splitParentMaterials{{7}};
    const std::array<std::uint32_t,2> splitChildMaterials{{7,7}},splitChildParents{{0,0}};
    if(!validateExplicitParentTetPartition(splitParentPoints,splitChildren,splitChildMaterials,splitChildParents,
            splitParentPoints,splitParentTets,splitParentMaterials,error))return 17;
    const std::array<std::array<std::uint32_t,4>,1> missingHalf{{{{0,4,2,3}}}};
    if(validateExplicitParentTetPartition(splitParentPoints,missingHalf,std::span<const std::uint32_t>(splitChildMaterials.data(),1),
            std::span<const std::uint32_t>(splitChildParents.data(),1),splitParentPoints,splitParentTets,splitParentMaterials,error)||
       error.find("volume does not close")==std::string::npos)return 18;
    auto outsidePoints=splitParentPoints;outsidePoints[4]={1.2,0,0};
    if(validateExplicitParentTetPartition(outsidePoints,splitChildren,splitChildMaterials,splitChildParents,
            splitParentPoints,splitParentTets,splitParentMaterials,error)||
       error.find("outside its mapped parent")==std::string::npos)return 19;
    std::cout<<"explicit_parent_child_topology=PASS children="<<childTets.size()
             <<" skin_mass_kg="<<explicitSkin.massDebitKg<<" fat_mass_kg="<<explicitFat.massDebitKg
             <<" default_path_parity=PASS parent/material/face/node/orientation_guards=PASS"
             <<" parent_geometry_partition=PASS boundary_face_area_coverage=PASS source_winding_independence=PASS true_bisection=PASS"
             <<" edge_BTB_energy_force_tangent=PASS\n";
    return 0;
}

