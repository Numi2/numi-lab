#include "NumiHumanCoarseTissuePreparation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
namespace ct=metalrobo::human::coarse_tissue;
using ct::Face;using ct::V3;
static numi::matter::MaterialProgram material(std::string name,double lo,double hi){
    numi::matter::MaterialProgram m;m.name=std::move(name);m.supportedRepresentations.push_back(numi::matter::Representation::fem);
    numi::matter::Parameter p;p.name="density";p.lower=lo;p.upper=hi;p.defaultValue=(lo+hi)*0.5;m.parameters.push_back(p);return m;
}
int main(){
    std::string error;
    const V3 shift{-0.05,0.20,2.0},mechanicalDelta{0,0,0.0001};const double radius=0.03;
    std::vector<V3> sourceWorld{{radius,0,0},{-radius,0,0},{0,radius,0},{0,-radius,0},{0,0,radius},{0,0,-radius}};
    for(auto& p:sourceWorld)p=ct::add(p,shift);
    std::vector<Face> sourceFaces;
    for(int sx:{1,-1})for(int sy:{1,-1})for(int sz:{1,-1}){
        Face f{sx>0?0u:1u,sy>0?2u:3u,sz>0?4u:5u};
        const V3 centroid=ct::mul(ct::add(ct::add(sourceWorld[f[0]],sourceWorld[f[1]]),sourceWorld[f[2]]),1.0/3.0);
        if(ct::dot(ct::cross(ct::sub(sourceWorld[f[1]],sourceWorld[f[0]]),ct::sub(sourceWorld[f[2]],sourceWorld[f[0]])),ct::sub(centroid,shift))<0)std::swap(f[1],f[2]);
        sourceFaces.push_back(f);
    }
    const Face split=sourceFaces.front();std::vector<V3> outer=sourceWorld;
    V3 center=ct::mul(ct::add(ct::add(sourceWorld[split[0]],sourceWorld[split[1]]),sourceWorld[split[2]]),1.0/3.0);
    for(auto& p:outer)p=ct::add(p,mechanicalDelta);outer.push_back(ct::add(center,mechanicalDelta));
    std::vector<Face> faces{Face{split[0],split[1],6u},Face{split[1],split[2],6u},Face{split[2],split[0],6u}};
    std::vector<std::uint32_t> faceParents{0u,0u,0u};
    for(std::uint32_t i=1;i<sourceFaces.size();i++){faces.push_back(sourceFaces[i]);faceParents.push_back(i);}
    std::vector<V3> normals;if(!ct::outwardVertexNormals(outer,faces,normals,error)){std::cerr<<error<<"\n";return 1;}
    std::vector<double> skinT(7,0.0005),fatT{0.00024,0.00026,0.00028,0.00030,0.00032,0.00034,0.00036};
    std::vector<double> skinLo(7,0.00045),skinHi(7,0.00055),fatLo,fatHi;
    for(double t:fatT){fatLo.push_back(t-0.00001);fatHi.push_back(t+0.00001);}
    std::vector<V3> inner(7),deep(7);
    for(std::size_t i=0;i<7;i++){inner[i]=ct::sub(outer[i],ct::mul(normals[i],skinT[i]));deep[i]=ct::sub(inner[i],ct::mul(normals[i],fatT[i]));}
    std::vector<std::int64_t> sourceIds{0,1,2,3,4,5,-1};
    std::uint32_t oneHotFace=NM_INVALID_INDEX;std::array<double,3> oneHotBeta{};
    for(std::uint32_t f=0;f<sourceFaces.size();f++)for(unsigned k=0;k<3;k++)if(sourceFaces[f][k]==1u){
        oneHotFace=f;oneHotBeta[k]=1.0;break;
    }
    if(oneHotFace==NM_INVALID_INDEX)return 2;
    std::vector<std::uint32_t> sourceFacesByNode{NM_INVALID_INDEX,oneHotFace,NM_INVALID_INDEX,NM_INVALID_INDEX,NM_INVALID_INDEX,NM_INVALID_INDEX,0u};
    std::vector<std::array<double,3>> beta(7);beta[1]=oneHotBeta;beta[6]={1.0/3.0,1.0/3.0,1.0/3.0};
    std::vector<std::uint8_t> kinds{0,2,0,0,0,0,3};
    std::vector<V3> residuals(7);residuals[1]=mechanicalDelta;residuals[6]=mechanicalDelta;
    std::vector<float> sourceWeights(sourceWorld.size()*86u,0.0f);for(std::size_t i=0;i<sourceWorld.size();i++)sourceWeights[i*86u+i]=1.0f;
    const std::string skinTProvenance="CPU fixture regional dermis field";
    const std::string fatTProvenance="CPU fixture local finger-fat field";
    const std::string skinBounds="fixture nodewise skin interval 0.45-0.55 mm";
    const std::string fatBounds="fixture nodewise fat interval 0.23-0.37 mm";
    const std::string meshHash(64,'a'),payloadHash(64,'b');
    ct::RegisteredSkinFatCandidateView c;c.name="candidate hand patch";c.provider="CPU fixture";
    c.sourceMember="registered-skin.csv";c.sourceMeshSHA256=meshHash;c.sourcePayloadSHA256=payloadHash;
    c.stableIdentifier=1;c.sourceBodyIndex=NM_INVALID_INDEX;c.aggregateIdentity=true;
    c.outerWorld=outer;c.skinInnerWorld=inner;c.fatInnerWorld=deep;c.faces=faces;c.sourceFaceParents=faceParents;c.sourceFaces=sourceFaces;
    c.sourceVertexIds=sourceIds;c.sourceFaceIds=sourceFacesByNode;c.sourceBarycentric=beta;c.vertexKinds=kinds;
    c.sourceResidualWorldM=residuals;c.sourceWeights86=sourceWeights;c.sourceVertexCount=sourceWorld.size();
    c.skinThicknessM=skinT;c.fatThicknessM=fatT;c.skinMinimumByVertexM=skinLo;c.skinMaximumByVertexM=skinHi;
    c.fatMinimumByVertexM=fatLo;c.fatMaximumByVertexM=fatHi;c.skinThicknessProvenance=skinTProvenance;
    c.fatThicknessProvenance=fatTProvenance;c.skinBoundsProvenance=skinBounds;c.fatBoundsProvenance=fatBounds;
    c.sourcePointToleranceM=1.0e-9;
    auto pointAt=[&](std::uint32_t sourceID,std::uint32_t sourceFace,const std::array<double,3>& b,std::uint8_t,V3& point,std::string& why){
        if(sourceID!=NM_INVALID_INDEX){if(sourceID>=sourceWorld.size()){why="bad direct source slot";return false;}point=sourceWorld[sourceID];return true;}
        if(sourceFace>=sourceFaces.size()){why="bad source face";return false;}
        point={};for(unsigned k=0;k<3;k++)point=ct::add(point,ct::mul(sourceWorld[sourceFaces[sourceFace][k]],b[k]));return true;
    };
    const std::array<double,9> identity{1,0,0,0,1,0,0,0,1};
    auto frameAt=[&](const std::array<double,86>& weights,const V3&,ct::RegisteredSkinFatDonorFrame& frame,std::string& why){
        double sum=0.0;std::size_t owner=0;for(std::size_t i=0;i<weights.size();i++){
            if(!std::isfinite(weights[i])||weights[i]<-1e-8){why="bad source weight";return false;}
            sum+=weights[i];if(weights[i]>weights[owner])owner=i;
        }
        if(std::abs(sum-1.0)>1e-4){why="weight row does not partition";return false;}
        frame.donorBody=static_cast<std::uint32_t>(2000+owner);frame.originWorld={0,0,0};frame.worldToDonorLocalLinear=identity;return true;
    };
    ct::RegisteredSurface surface;ct::PairedBoundaryLayer skinLayer,fatLayer;ct::SkinFatReference ref;
    ct::RegisteredSkinFatCandidateBuildAudit audit;
    if(!ct::buildRegisteredSkinFatCandidateLayers(c,pointAt,frameAt,surface,skinLayer,fatLayer,ref,audit,error)){
        std::cerr<<"builder rejected fixture: "<<error<<"\n";return 3;}
    if(surface.sourceVertexIndices[0]!=0||surface.sourceVertexIndices[1]!=1||
       surface.sourceVertexIndices[6]!=NM_INVALID_INDEX||surface.sourceEmbeddingFaceIndices[0]!=NM_INVALID_INDEX||
       surface.sourceEmbeddingFaceIndices[1]!=oneHotFace||surface.sourceEmbeddingFaceIndices[6]!=0||
       audit.kindZeroRows!=5||audit.nonzeroKindRows!=2||audit.oneHotRows!=1||
       audit.maximumBarycentricReconstructionErrorM>1e-12||audit.maximumRetainedMechanicalDeltaM<0.00009)return 4;
    if(!(ref.minFatThicknessM<0.0005&&ref.maxFatThicknessM>0.00035&&
         skinLayer.thicknessProvenance.find(skinBounds)!=std::string::npos&&fatLayer.thicknessProvenance.find(fatBounds)!=std::string::npos))return 5;
    if(!ct::validateRegisteredSourceEmbedding(surface,sourceFaces.size(),error)){std::cerr<<error<<"\n";return 6;}
    numi::matter::WorldSource world;world.compactImmutableFEMSurfaces=true;world.frameTimestep=0.001;world.environmentCount=1;
    world.materials={material("dermis",900,1300),material("adipose",700,1100)};
    ct::PreparedOrganicFEM preparedSkin,preparedFat;const double skinDebit=skinLayer.volumeM3*1120.0,fatDebit=fatLayer.volumeM3*880.0;
    const std::array<std::uint64_t,4> materialIdentity{1,2,3,4};
    const std::string skinProv=skinTProvenance+"; node bounds: "+skinBounds,fatProv=fatTProvenance+"; node bounds: "+fatBounds;
    if(!ct::appendRegisteredSkinFatFEMFromLayers(world,surface,0,"dermis","fixture donor partition",skinDebit,
       1,"adipose","fixture donor partition",fatDebit,skinT,fatT,skinProv,fatProv,ref,skinLayer,fatLayer,
       "fixture skin","fixture fat",materialIdentity,preparedSkin,preparedFat,error)){
        std::cerr<<"existing FEM layer owner rejected fixture: "<<error<<"\n";return 7;}
    if(world.objects.size()!=1||preparedSkin.volumeM3<=0||preparedFat.volumeM3<=0||
       std::abs(preparedSkin.densityKgM3*preparedSkin.volumeM3-skinDebit)>1e-10*skinDebit||
       std::abs(preparedFat.densityKgM3*preparedFat.volumeM3-fatDebit)>1e-10*fatDebit||
       std::abs(preparedSkin.massResidualKg)>1e-10*skinDebit||std::abs(preparedFat.massResidualKg)>1e-10*fatDebit||
       preparedSkin.nodeSourceVertex[6]!=NM_INVALID_INDEX)return 8;
    std::cout<<"candidate_surface_builder=PASS vertices="<<audit.vertexCount<<" faces="<<audit.faceCount
             <<" exact_ids="<<audit.retainedExactSourceVertices<<" projected_rows="<<audit.barycentricRows
             <<" max_reconstruction_m="<<audit.maximumBarycentricReconstructionErrorM
             <<" fat_node_bounds_m=["<<audit.fatBoundEnvelopeMinimumM<<","<<audit.fatBoundEnvelopeMaximumM<<"]"
             <<" existing_from_layers=PASS skin_rhoV="<<preparedSkin.massDebitKg
             <<" fat_rhoV="<<preparedFat.massDebitKg<<" donor_mass_debit=unchanged\n";
    auto rejects=[&](ct::RegisteredSkinFatCandidateView bad){
        ct::RegisteredSurface s;ct::PairedBoundaryLayer sl,fl;ct::SkinFatReference r;ct::RegisteredSkinFatCandidateBuildAudit a;std::string why;
        return !ct::buildRegisteredSkinFatCandidateLayers(bad,pointAt,frameAt,s,sl,fl,r,a,why);
    };
    auto badResidual=residuals;badResidual[6][0]+=1e-4;auto bad=c;bad.sourceResidualWorldM=badResidual;if(!rejects(bad))return 9;
    auto badID=sourceIds;badID[1]=2;bad=c;bad.sourceVertexIds=badID;if(!rejects(bad))return 10;
    auto badBounds=fatLo;badBounds[6]=0.00037;bad=c;bad.fatMinimumByVertexM=badBounds;if(!rejects(bad))return 11;
    auto badWeights=sourceWeights;badWeights[2u*86u]=0.5f;bad=c;bad.sourceWeights86=badWeights;if(!rejects(bad))return 12;
    std::cout<<"candidate_builder_negative_gates=PASS residual identity per_node_bounds full86_partition\n";
    return 0;
}
