#include "NumiHumanSkinFatReductionLabels.hpp"
#include "NumiHumanParentReferenceMassTransfer.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <limits>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using numi::matter::Representation;
namespace {
constexpr std::uint32_t kInvalid=NM_INVALID_INDEX;
constexpr std::array<double,9> kIdentity{1,0,0,0,1,0,0,0,1};

struct Fixture {
    RegisteredSurface surface;
    numi::human::SkinBoundaryNodeMap boundary;
    ExplicitUnpairedVolumeTopology topology;
    std::vector<V3> current,reference,parentPoints,localPoints,origins;
    std::vector<std::array<double,9>> nodeRotations,bodyRotations;
    std::vector<std::uint32_t> nodeDonor,childMaterial,dominant,parentSurface,childStable,childDonor;
    std::vector<std::array<std::uint32_t,4>> children,parentTets;
    std::vector<std::uint32_t> parentMaterial,parentStable,parentDonor,parentRegion,childRegion;
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> sourceMassContributions;
    std::vector<std::uint32_t> parentOffsets,parentRows;
    std::vector<double> parentFractions;
    std::vector<std::uint32_t> facetRows,facetStable,sourceVertex,sourceFace,nodeParent;
    std::vector<std::array<double,3>> sourceBary;
    std::vector<std::uint8_t> sidewallFlags;
    std::vector<std::uint32_t> sidewallStable;
    std::vector<std::array<double,4>> nodeParentBary;
    std::vector<V3> nodeParentResidual,bodyOrigins;
    std::string receipt=std::string(64,static_cast<char>(0x61));

    static int layer(std::uint32_t id) { return id<18u?static_cast<int>(id/6u):-1; }
    static std::uint32_t surfaceRow(std::uint32_t id) { return id%6u; }

    void refreshChildMetadata() {
        parentRegion.resize(parentMaterial.size());
        for(std::size_t p=0;p<parentMaterial.size();++p)
            parentRegion[p]=parentMaterial[p]==0u?101u:202u;
        childRegion.resize(childMaterial.size());
        for(std::size_t c=0;c<childMaterial.size();++c)
            childRegion[c]=childMaterial[c]==0u?101u:202u;
        parentOffsets.assign(children.size()+1u,0u);parentRows.clear();parentFractions.clear();
        for(std::size_t c=0;c<children.size();++c) {
            parentRows.push_back(dominant[c]);parentFractions.push_back(1.0);
            parentOffsets[c+1u]=static_cast<std::uint32_t>(parentRows.size());
        }
        facetRows.assign(4u*children.size(),kInvalid);facetStable.assign(4u*children.size(),kInvalid);
        sidewallFlags.assign(4u*children.size(),0u);sidewallStable.assign(4u*children.size(),kInvalid);
        std::map<std::array<std::uint32_t,3>,std::uint32_t> faceRows;
        for(std::uint32_t f=0;f<surface.faces.size();++f) {
            auto key=surface.faces[f];std::sort(key.begin(),key.end());faceRows.emplace(key,f);
        }
        for(std::size_t c=0;c<children.size();++c)for(unsigned omitted=0;omitted<4u;++omitted) {
            std::array<std::uint32_t,3> nodes{};unsigned at=0;
            for(unsigned k=0;k<4u;++k)if(k!=omitted)nodes[at++]=children[c][k];
            const int role=layer(nodes[0]);bool same=role>=0;
            for(auto node:nodes)if(layer(node)!=role)same=false;
            if(!same)continue;
            std::array<std::uint32_t,3> rows{};
            for(unsigned k=0;k<3u;++k)rows[k]=surfaceRow(nodes[k]);
            std::sort(rows.begin(),rows.end());
            auto found=faceRows.find(rows);
            if(found!=faceRows.end()) {
                facetRows[4u*c+omitted]=found->second;
                facetStable[4u*c+omitted]=surface.sourceFaceIndices[found->second];
            }
        }
        topology.childTetrahedra=children;topology.childMaterialIndices=childMaterial;
        topology.childDominantParentTetrahedra=dominant;topology.childParentSurfaceFaces=parentSurface;
        topology.childStableSourceFaceIdentities=childStable;topology.childDonorBodyIndices=childDonor;
        topology.childAnatomicalRegionIndices=childRegion;
        topology.childParentAncestryOffsets=parentOffsets;topology.childParentAncestryTetrahedra=parentRows;
        topology.childParentAncestryVolumeFractions=parentFractions;
        topology.childRegisteredLayerFacetSurfaceFaces=facetRows;
        topology.childRegisteredLayerFacetSourceFaceIdentities=facetStable;
        topology.childAuthoredHeadSidewallFacetFlags=sidewallFlags;
        topology.childAuthoredHeadSidewallStableFaceIdentities=sidewallStable;
    }

    Fixture() {
        surface.name="closed triangular-prism source";surface.provider="synthetic test";
        surface.sourceMember="test/source";surface.stableIdentifier=1u;surface.sourceBodyIndex=0u;
        surface.sourceMeshSHA256=std::string(64,static_cast<char>(0x31));
        surface.sourcePayloadSHA256=std::string(64,static_cast<char>(0x32));
        const double q=std::sqrt(3.0)*0.5;
        const std::array<V3,6> unit{{{{1,0,-1}},{{-.5,q,-1}},{{-.5,-q,-1}},{{1,0,1}},{{-.5,q,1}},{{-.5,-q,1}}}};
        surface.worldPoints.assign(unit.begin(),unit.end());
        surface.faces={{{0,2,1}},{{3,4,5}},{{0,1,4}},{{0,4,3}},{{1,2,5}},{{1,5,4}},{{2,0,3}},{{2,3,5}}};
        surface.sourceVertexIndices={0,1,2,3,4,5};surface.sourceFaceIndices.resize(surface.faces.size());
        for(std::uint32_t f=0;f<surface.faces.size();++f)surface.sourceFaceIndices[f]=100u+f;
        surface.donorBodyIndices.assign(6u,0u);surface.donorLocalPoints=surface.worldPoints;
        boundary.volumeNodeCount=19u;boundary.outer={0,1,2,3,4,5};
        boundary.skinInner={6,7,8,9,10,11};boundary.fatDeep={12,13,14,15,16,17};

        const std::array<double,3> scales{1.0,0.8,0.6};
        for(double scale:scales)for(const auto& point:unit) {
            V3 v=point;for(auto& x:v)x*=scale;
            current.push_back(v);reference.push_back(v);localPoints.push_back(v);origins.push_back({0,0,0});
            nodeRotations.push_back(kIdentity);sourceVertex.push_back(kInvalid);
            sourceFace.push_back(kInvalid);sourceBary.push_back({});nodeParent.push_back(kInvalid);
            nodeParentBary.push_back({});nodeParentResidual.push_back({0,0,0});
        }
        for(std::uint32_t row=0;row<6u;++row)sourceVertex[row]=row;
        bodyRotations.push_back(kIdentity);bodyOrigins.push_back({0,0,0});

        const auto orient=[&](std::array<std::uint32_t,4>& tet,const std::vector<V3>& points) {
            if(tetVolume(points[tet[0]],points[tet[1]],points[tet[2]],points[tet[3]])<0.0)
                std::swap(tet[0],tet[1]);
        };
        auto appendInterval=[&](unsigned lowerBlock,unsigned upperBlock,std::uint32_t material) {
            for(const auto& face:surface.faces) {
                auto sorted=face;std::sort(sorted.begin(),sorted.end());
                const auto a=sorted[0],b=sorted[1],c=sorted[2];
                std::array<std::array<std::uint32_t,4>,3> local{{
                    {{lowerBlock+a,lowerBlock+b,lowerBlock+c,upperBlock+a}},
                    {{lowerBlock+b,upperBlock+b,lowerBlock+c,upperBlock+a}},
                    {{lowerBlock+c,upperBlock+b,upperBlock+c,upperBlock+a}}
                }};
                for(auto tet:local) { orient(tet,reference);parentTets.push_back(tet);parentMaterial.push_back(material); }
            }
        };
        appendInterval(6u,0u,0u);   // skin shell: O between I
        appendInterval(12u,6u,1u);  // fat shell: I between D
        parentStable.resize(parentTets.size());parentDonor.assign(parentTets.size(),0u);
        for(std::uint32_t i=0;i<parentTets.size();++i)parentStable[i]=10000u+i;

        const auto split=parentTets.front();V3 center{};
        for(auto node:split)for(unsigned axis=0;axis<3u;++axis)center[axis]+=reference[node][axis]*0.25;
        current.push_back(center);reference.push_back(center);localPoints.push_back(center);origins.push_back({0,0,0});
        nodeRotations.push_back(kIdentity);sourceVertex.push_back(kInvalid);sourceFace.push_back(kInvalid);sourceBary.push_back({});
        nodeParent.push_back(0u);nodeParentBary.push_back({.25,.25,.25,.25});nodeParentResidual.push_back({0,0,0});
        for(std::uint32_t parent=0;parent<parentTets.size();++parent) {
            if(parent!=0u) {
                children.push_back(parentTets[parent]);childMaterial.push_back(parentMaterial[parent]);
                dominant.push_back(parent);parentSurface.push_back(kInvalid);
                childStable.push_back(parentStable[parent]);childDonor.push_back(0u);continue;
            }
            for(unsigned omitted=0;omitted<4u;++omitted) {
                auto tet=std::array<std::uint32_t,4>{};unsigned at=0;
                for(unsigned corner=0;corner<4u;++corner)if(corner!=omitted)tet[at++]=split[corner];
                tet[3]=18u;orient(tet,reference);
                children.push_back(tet);childMaterial.push_back(parentMaterial[parent]);dominant.push_back(parent);
                parentSurface.push_back(kInvalid);childStable.push_back(parentStable[parent]);childDonor.push_back(0u);
            }
        }
        parentPoints.assign(reference.begin(),reference.begin()+18);
        nodeParent.assign(19u,kInvalid);nodeParent[18]=0u;
        nodeParentBary.assign(19u,{});nodeParentBary[18]={.25,.25,.25,.25};
        nodeParentResidual.assign(19u,{0,0,0});

        refreshChildMetadata();
        nodeDonor.assign(19u,0u);
        topology.currentWorldPoints=current;topology.referenceWorldPoints=reference;
        topology.parentReferenceWorldPoints=parentPoints;topology.parentTetrahedra=parentTets;
        topology.parentMaterialIndices=parentMaterial;topology.parentStableSourceFaceIdentities=parentStable;
        topology.parentDonorBodyIndices=parentDonor;topology.parentAnatomicalRegionIndices=parentRegion;
        topology.volumeNodeDonorBodyIndices=nodeDonor;topology.volumeNodeDonorLocalPoints=localPoints;
        topology.volumeNodeDonorLocalToWorldLinear=nodeRotations;topology.volumeNodeDonorOriginWorld=origins;
        topology.registeredBodyLocalToWorldLinear=bodyRotations;topology.registeredBodyOriginWorld=bodyOrigins;
        topology.detailedSourceVertexCount=6u;topology.detailedSourceFaceCount=surface.faces.size();
        topology.volumeNodeSourceVertexIndices=sourceVertex;topology.volumeNodeSourceFaceIndices=sourceFace;
        topology.volumeNodeSourceBarycentric=sourceBary;topology.retainedOriginalNodeCount=18u;
        topology.volumeNodeParentTetrahedra=nodeParent;topology.volumeNodeParentBarycentric=nodeParentBary;
        topology.volumeNodeParentResidualWorldM=nodeParentResidual;
        topology.ancestryVerificationKind=VolumeAncestryVerificationKind::conservativeReferencePullback;
        topology.ancestryVerificationReceiptSHA256=receipt; // Syntax fixture only; not a physical ancestry receipt.
        topology.referenceSourceIdentity={1,2,3,4};topology.materialSourceIdentity={5,6,7,8};
    }
};


// Refine just one bonded I-layer face, without changing any O/D row or face.
// Both incident material cells split conformingly, retaining exact parent volumes.
void independentInnerFixture(Fixture& f,RegisteredSurface& inner,RegisteredSurface& deep) {
    inner=f.surface;deep=f.surface;
    inner.name="independent refined I";deep.name="independent D";
    inner.worldPoints.clear();deep.worldPoints.clear();
    for(auto node:f.boundary.skinInner)inner.worldPoints.push_back(f.current[node]);
    for(auto node:f.boundary.fatDeep)deep.worldPoints.push_back(f.current[node]);
    inner.donorLocalPoints=inner.worldPoints;deep.donorLocalPoints=deep.worldPoints;
    for(std::uint32_t row=0;row<6u;++row){
        f.sourceVertex[f.boundary.skinInner[row]]=row;
        f.sourceVertex[f.boundary.fatDeep[row]]=row;
    }
    const auto iface=f.surface.faces[0];
    std::array<std::uint32_t,3> oldFace{};
    V3 point{};
    for(unsigned k=0;k<3u;++k){
        oldFace[k]=f.boundary.skinInner[iface[k]];
        for(unsigned a=0;a<3u;++a)point[a]+=f.reference[oldFace[k]][a]/3.0;
    }
    const auto added=static_cast<std::uint32_t>(f.current.size());
    std::uint32_t parent=kInvalid;
    for(std::uint32_t at=0;at<f.parentTets.size();++at){
        bool found=true;
        for(auto node:oldFace)found&=std::find(f.parentTets[at].begin(),f.parentTets[at].end(),node)!=f.parentTets[at].end();
        if(found){parent=at;break;}
    }
    assert(parent!=kInvalid);
    std::array<double,4> bary{};
    for(unsigned k=0;k<4u;++k)
        if(std::find(oldFace.begin(),oldFace.end(),f.parentTets[parent][k])!=oldFace.end())bary[k]=1.0/3.0;
    f.current.push_back(point);f.reference.push_back(point);f.localPoints.push_back(point);
    f.origins.push_back({0,0,0});f.nodeRotations.push_back(kIdentity);f.nodeDonor.push_back(0u);
    f.sourceVertex.push_back(kInvalid);f.sourceFace.push_back(0u);f.sourceBary.push_back({1.0/3,1.0/3,1.0/3});
    f.nodeParent.push_back(parent);f.nodeParentBary.push_back(bary);f.nodeParentResidual.push_back({0,0,0});
    f.boundary.skinInner.push_back(added);f.boundary.volumeNodeCount++;
    const auto innerRow=static_cast<std::uint32_t>(inner.worldPoints.size());
    inner.worldPoints.push_back(point);inner.donorLocalPoints.push_back(point);inner.donorBodyIndices.push_back(0u);
    inner.sourceVertexIndices.push_back(kInvalid);
    inner.sourceEmbeddingFaceIndices.assign(inner.worldPoints.size(),kInvalid);
    inner.sourceEmbeddingBarycentric.assign(inner.worldPoints.size(),{});
    inner.sourceEmbeddingFaceIndices.back()=0u;inner.sourceEmbeddingBarycentric.back()={1.0/3,1.0/3,1.0/3};
    inner.faces.erase(inner.faces.begin());inner.sourceFaceIndices.erase(inner.sourceFaceIndices.begin());
    for(unsigned k=0;k<3u;++k){
        inner.faces.push_back({iface[k],iface[(k+1u)%3u],innerRow});
        inner.sourceFaceIndices.push_back(1000u+k);
    }
    const auto oldChildren=f.children;const auto oldMaterial=f.childMaterial,oldDominant=f.dominant;
    const auto oldParentSurface=f.parentSurface,oldStable=f.childStable,oldDonor=f.childDonor;
    f.children.clear();f.childMaterial.clear();f.dominant.clear();f.parentSurface.clear();f.childStable.clear();f.childDonor.clear();
    unsigned splits=0u;
    for(std::size_t c=0;c<oldChildren.size();++c){
        bool hasFace=true;
        for(auto node:oldFace)hasFace&=std::find(oldChildren[c].begin(),oldChildren[c].end(),node)!=oldChildren[c].end();
        const auto append=[&](std::array<std::uint32_t,4> tet){
            if(tetVolume(f.reference[tet[0]],f.reference[tet[1]],f.reference[tet[2]],f.reference[tet[3]])<0.0)std::swap(tet[0],tet[1]);
            f.children.push_back(tet);f.childMaterial.push_back(oldMaterial[c]);f.dominant.push_back(oldDominant[c]);
            f.parentSurface.push_back(oldParentSurface[c]);f.childStable.push_back(oldStable[c]);f.childDonor.push_back(oldDonor[c]);
        };
        if(!hasFace){append(oldChildren[c]);continue;}
        ++splits;std::uint32_t other=kInvalid;
        for(auto node:oldChildren[c])if(std::find(oldFace.begin(),oldFace.end(),node)==oldFace.end())other=node;
        assert(other!=kInvalid);
        for(unsigned k=0;k<3u;++k)append({other,oldFace[k],oldFace[(k+1u)%3u],added});
    }
    assert(splits==2u);
    f.refreshChildMetadata();
    std::array<const RegisteredSurface*,3> surfaces{&f.surface,&inner,&deep};
    std::array<const std::vector<std::uint32_t>*,3> maps{&f.boundary.outer,&f.boundary.skinInner,&f.boundary.fatDeep};
    std::vector<std::uint32_t> layer(f.current.size(),kInvalid),row(f.current.size(),kInvalid);
    std::array<std::map<Face,std::uint32_t>,3> faceRows;
    for(unsigned role=0;role<3u;++role){
        for(std::uint32_t r=0;r<maps[role]->size();++r){layer[(*maps[role])[r]]=role;row[(*maps[role])[r]]=r;}
        for(std::uint32_t r=0;r<surfaces[role]->faces.size();++r){auto key=surfaces[role]->faces[r];std::sort(key.begin(),key.end());faceRows[role].emplace(key,r);}
    }
    std::fill(f.facetRows.begin(),f.facetRows.end(),kInvalid);std::fill(f.facetStable.begin(),f.facetStable.end(),kInvalid);
    for(std::size_t c=0;c<f.children.size();++c)for(unsigned omit=0;omit<4u;++omit){
        std::array<std::uint32_t,3> nodes{};unsigned at=0;
        for(unsigned k=0;k<4u;++k)if(k!=omit)nodes[at++]=f.children[c][k];
        const auto role=layer[nodes[0]];bool same=role!=kInvalid;
        for(auto node:nodes)if(layer[node]!=role)same=false;
        if(!same)continue;
        Face key{};for(unsigned k=0;k<3u;++k)key[k]=row[nodes[k]];std::sort(key.begin(),key.end());
        const auto found=faceRows[role].find(key);if(found==faceRows[role].end())continue;
        f.facetRows[4*c+omit]=found->second;
        f.facetStable[4*c+omit]=surfaces[role]->sourceFaceIndices[found->second];
    }
    f.topology.independentSkinInnerSurface=&inner;f.topology.independentFatDeepSurface=&deep;
    f.topology.currentWorldPoints=f.current;f.topology.referenceWorldPoints=f.reference;
    f.topology.volumeNodeDonorBodyIndices=f.nodeDonor;f.topology.volumeNodeDonorLocalPoints=f.localPoints;
    f.topology.volumeNodeDonorLocalToWorldLinear=f.nodeRotations;f.topology.volumeNodeDonorOriginWorld=f.origins;
    f.topology.volumeNodeSourceVertexIndices=f.sourceVertex;f.topology.volumeNodeSourceFaceIndices=f.sourceFace;
    f.topology.volumeNodeSourceBarycentric=f.sourceBary;
    f.topology.volumeNodeParentTetrahedra=f.nodeParent;f.topology.volumeNodeParentBarycentric=f.nodeParentBary;
    f.topology.volumeNodeParentResidualWorldM=f.nodeParentResidual;
}

numi::matter::WorldSource worldFor() {
    numi::matter::WorldSource world;world.materials.resize(2u);
    for(std::uint32_t i=0;i<2u;++i) {
        world.materials[i].name=i==0u?"skin":"fat";
        world.materials[i].supportedRepresentations={Representation::fem};
        numi::matter::Parameter density;density.name="density";density.defaultValue=i==0u?1000.0:900.0;
        density.lower=i==0u?900.0:850.0;density.upper=i==0u?1100.0:950.0;world.materials[i].parameters.push_back(density);
    }
    return world;
}

bool emit(Fixture& f,numi::matter::WorldSource& world,ExplicitUnpairedVolumeReport& report,std::string& error,
          PreparedOrganicFEM* skinOutput=nullptr,PreparedOrganicFEM* fatOutput=nullptr) {
    double skinVolume=0.0,fatVolume=0.0;
    for(std::size_t i=0;i<f.children.size();++i) {
        const auto tet=f.children[i];
        const double v=tetVolume(f.reference[tet[0]],f.reference[tet[1]],f.reference[tet[2]],f.reference[tet[3]]);
        (f.childMaterial[i]==0u?skinVolume:fatVolume)+=v;
    }
    PreparedOrganicFEM skin,fat;
    if(!appendRegisteredSkinFatUnpairedVolume(world,f.surface,f.boundary,f.topology,
        0u,"skin","test skin material",skinVolume*1000.0,1u,"fat","test fat material",
        fatVolume*900.0,1u,"unpaired test shell",skin,fat,report,error))return false;
    if(skinOutput)*skinOutput=skin;
    if(fatOutput)*fatOutput=fat;
    return true;
}

bool emitMassLedger(Fixture& f,const std::vector<double>& parentMass,
                    const std::vector<double>& childMeasure,numi::matter::WorldSource& world,
                    PreparedOrganicFEM& skin,PreparedOrganicFEM& fat,
                    ExplicitUnpairedVolumeReport& report,std::string& error,
                    double skinDebitScale=1.0,double fatDebitScale=1.0,
                    std::span<const ExplicitUnpairedVolumeSourceMassContribution> sourceContributions={}) {
    f.topology.parentTetrahedronMassKg=parentMass;
    f.topology.childAncestryMeasureVolumeM3=childMeasure;
    f.topology.parentNodeSourceMassContributions=sourceContributions;
    double skinDebit=0.0,fatDebit=0.0;
    for(std::size_t p=0;p<parentMass.size();++p)
        (f.parentMaterial[p]==0u?skinDebit:fatDebit)+=parentMass[p];
    return appendRegisteredSkinFatUnpairedVolume(world,f.surface,f.boundary,f.topology,
        0u,"skin","test skin material",skinDebit*skinDebitScale,1u,"fat","test fat material",
        fatDebit*fatDebitScale,f.bodyRotations.size(),"per-child mass-ledger test",skin,fat,report,error);
}

double sumDonorMass(const PreparedOrganicFEM& prepared) {
    double total=0.0;
    for(const auto& row:prepared.donorMassNodes)total+=row.massKg;
    return total;
}

std::vector<ExplicitUnpairedVolumeSourceMassContribution> sourceContributionFixture(
        const Fixture& fixture,const std::vector<double>& parentMass,double rawCookScale) {
    using Key=std::pair<std::uint32_t,std::uint32_t>;
    std::map<Key,double> cookedByNodeMaterial;
    for(std::size_t parent=0;parent<fixture.parentTets.size();++parent)
        for(const auto node:fixture.parentTets[parent])
            cookedByNodeMaterial[{node,fixture.parentMaterial[parent]}]+=parentMass[parent]*0.25;
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> rows;
    const auto splitKey=Key{fixture.parentTets.front()[0],fixture.parentMaterial.front()};
    for(const auto& [key,cookedMass]:cookedByNodeMaterial) {
        const auto [node,material]=key;
        const double rawMass=cookedMass*rawCookScale;
        if(key==splitKey) {
            V3 localForSecondDonor=fixture.parentPoints[node];
            localForSecondDonor[0]-=fixture.bodyOrigins[1][0];
            rows.push_back({node,material,0u,rawMass*0.65,fixture.parentPoints[node]});
            rows.push_back({node,material,1u,rawMass*0.35,localForSecondDonor});
        } else {
            rows.push_back({node,material,0u,rawMass,fixture.parentPoints[node]});
        }
    }
    return rows;
}

bool buildSpatialTransportFixture(const Fixture& fixture,
        const std::vector<double>& parentMass,
        const std::vector<ExplicitUnpairedVolumeSourceMassContribution>& sourceRows,
        const double maximumDistance,
        ConservativeSpatialMassTransportResult& result,
        std::string& error) {
    const auto parentCount=fixture.parentTets.size();
    const auto childCount=fixture.children.size();
    if(parentMass.size()!=parentCount||sourceRows.empty()||
       fixture.parentRegion.size()!=parentCount||fixture.childRegion.size()!=childCount) {
        error="test spatial transfer inputs do not cover the fixture cells";
        return false;
    }
    std::vector<SpatialMassTransportPoint> parentCentroids(parentCount),childCentroids(childCount);
    std::vector<double> parentVolumes(parentCount),childVolumes(childCount);
    for(std::size_t p=0;p<parentCount;++p){
        const auto tet=fixture.parentTets[p];
        parentVolumes[p]=parentReferenceTetVolumeM3(tet,fixture.parentPoints);
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
            parentCentroids[p][axis]+=fixture.parentPoints[node][axis]*0.25;
    }
    for(std::size_t c=0;c<childCount;++c){
        const auto tet=fixture.children[c];
        childVolumes[c]=parentReferenceTetVolumeM3(tet,fixture.reference);
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
            childCentroids[c][axis]+=fixture.reference[node][axis]*0.25;
    }
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceMassRows;
    sourceMassRows.reserve(sourceRows.size());
    for(const auto& row:sourceRows)
        sourceMassRows.push_back({.parentNodeIndex=row.parentNodeIndex,
            .materialIndex=row.materialIndex,.sourceDonorBodyIndex=row.donorBodyIndex,
            .massKg=row.massKg});
    std::vector<std::uint32_t> sourceOffsets,sourceDonors;
    std::vector<double> sourceMasses;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            parentMass,fixture.parentMaterial,fixture.parentTets,sourceMassRows,
            1.0e-10,5.0e-4,sourceOffsets,sourceDonors,sourceMasses,error))
        return false;
    const std::array<double,2> densityLower{900.0,850.0};
    const std::array<double,2> densityUpper{1100.0,950.0};
    static const std::string parentHash(64u,'1'),sourceHash(64u,'2'),childHash(64u,'3'),
        regionHash(64u,'4'),frameHash(64u,'5');
    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentCentroids;
    input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMass;
    input.parentMaterialIndices=fixture.parentMaterial;
    input.parentDonorBodyIndices=fixture.parentDonor;
    input.parentAnatomicalRegionIndices=fixture.parentRegion;
    input.parentSourceDonorOffsets=sourceOffsets;
    input.parentSourceDonorBodyIndices=sourceDonors;
    input.parentSourceDonorMassKg=sourceMasses;
    input.childCentroidsWorld=childCentroids;
    input.childReferenceVolumeM3=childVolumes;
    input.childMaterialIndices=fixture.childMaterial;
    input.childDonorBodyIndices=fixture.childDonor;
    input.childAnatomicalRegionIndices=fixture.childRegion;
    input.materialDensityLowerKgM3=densityLower;
    input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=parentHash;
    input.parentSourceContributionSHA256=sourceHash;
    input.childMeshSHA256=childHash;
    input.regionLabelSHA256=regionHash;
    input.coordinateFrameSHA256=frameHash;
    input.maximumTransportDistanceM=maximumDistance;
    return assembleConservativeSpatialMassTransport(input,result,error);
}

struct SourceMoments {
    double mass=0.0;
    std::array<double,3> first{};
    std::array<double,9> second{};
};

void accumulateSourceMoments(SourceMoments& moments,double mass,const V3& position) {
    moments.mass+=mass;
    for(unsigned row=0;row<3u;++row) {
        moments.first[row]+=mass*position[row];
        for(unsigned column=0;column<3u;++column)
            moments.second[3u*row+column]+=mass*position[row]*position[column];
    }
}
}

int main() {
    Fixture f;auto world=worldFor();ExplicitUnpairedVolumeReport report;std::string error;
    PreparedOrganicFEM legacySkin,legacyFat;
    if(!emit(f,world,report,error,&legacySkin,&legacyFat)){std::cerr<<error;return 1;}
    assert(world.objects.size()==1u&&report.volumeNodeCount==19u);
    assert(report.inputVolumeRowToObjectLocalNode.size()==19u&&report.inputVolumeRowToObjectLocalNode[18]==18u);
    assert(!world.objects.front().deformableSelfContact); // legacy paired rows retain the old contact owner.
    assert(world.objects.front().femTetrahedronMassesKg.empty());
    assert(std::all_of(world.objects.front().femTetrahedronMassSourceIdentity.begin(),
        world.objects.front().femTetrahedronMassSourceIdentity.end(),
        [](auto word){return word==0u;}));
    assert(report.ancestryVerificationKind==VolumeAncestryVerificationKind::conservativeReferencePullback);
    assert(legacySkin.donorMassNodes.size()==report.volumeNodeCount);
    assert(legacyFat.donorMassNodes.empty());
    assert(legacySkin.donorMassContributions.empty()&&legacyFat.donorMassContributions.empty());
    assert(std::abs(sumDonorMass(legacySkin)-legacySkin.massDebitKg-legacyFat.massDebitKg)<1.0e-10);
    {
        Fixture splitSource;
        splitSource.bodyRotations.push_back(kIdentity);
        splitSource.bodyOrigins.push_back({0.25,0.0,0.0});
        splitSource.topology.registeredBodyLocalToWorldLinear=splitSource.bodyRotations;
        splitSource.topology.registeredBodyOriginWorld=splitSource.bodyOrigins;
        std::vector<double> parentMass(splitSource.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p) {
            const auto tet=splitSource.parentTets[p];
            const double volume=tetVolume(splitSource.parentPoints[tet[0]],splitSource.parentPoints[tet[1]],
                                          splitSource.parentPoints[tet[2]],splitSource.parentPoints[tet[3]]);
            parentMass[p]=volume*(splitSource.parentMaterial[p]==0u?1000.0:900.0);
        }
        std::vector<double> childMeasure(splitSource.children.size(),0.0);
        for(std::size_t c=0;c<childMeasure.size();++c) {
            const auto tet=splitSource.children[c];
            childMeasure[c]=tetVolume(splitSource.reference[tet[0]],splitSource.reference[tet[1]],
                                      splitSource.reference[tet[2]],splitSource.reference[tet[3]]);
        }
        const auto rawContributions=sourceContributionFixture(splitSource,parentMass,1.0+1.0e-5);
        splitSource.sourceMassContributions=rawContributions;
        auto splitWorld=worldFor();PreparedOrganicFEM splitSkin,splitFat;
        assert(emitMassLedger(splitSource,parentMass,childMeasure,splitWorld,splitSkin,splitFat,
                              report,error,1.0,1.0,splitSource.sourceMassContributions));
        assert(report.usesPerChildMassLedger&&report.usesParentNodeSourceMassContributions);
        assert(report.maximumSourceNodeCookRoundingErrorKg>0.0);
        assert(report.maximumSourceNodeCookRoundingRelativeError<5.0e-4);
        assert(report.maximumOutputSourceNodeMassClosureErrorKg<1.0e-12);
        assert(splitSkin.donorMassNodes.size()==splitSource.current.size());
        assert(splitFat.donorMassNodes.empty());
        assert(splitSkin.donorMassContributions.size()>splitSkin.donorMassNodes.size());
        std::map<std::uint32_t,double> frameMass,sourceMassByNode;
        std::map<std::uint32_t,std::uint32_t> frameBodyByNode;
        using MomentKey=std::tuple<std::uint32_t,std::uint32_t,V3>;
        std::map<MomentKey,SourceMoments> expectedMoments,actualMoments;
        for(const auto& node:splitSkin.donorMassNodes) {
            assert(frameBodyByNode.emplace(node.nodeIndex,node.donorBody).second);
            assert(node.donorBody==0u);
            frameMass[node.nodeIndex]+=node.massKg;
        }
        for(const auto& contribution:splitSkin.donorMassContributions) {
            sourceMassByNode[contribution.nodeIndex]+=contribution.massKg;
            accumulateSourceMoments(actualMoments[{contribution.nodeIndex,contribution.donorBody,
                                    contribution.localPosition}],contribution.massKg,contribution.localPosition);
        }
        for(std::size_t child=0;child<splitSource.children.size();++child) {
            const auto parent=splitSource.dominant[child];
            const auto material=splitSource.childMaterial[child];
            const double childMass=splitWorld.objects.front().femTetrahedronMassesKg[child];
            for(const auto childNode:splitSource.children[child])
                for(const auto parentNode:splitSource.parentTets[parent]) {
                    double rawNodeMaterial=0.0;
                    for(const auto& row:rawContributions)
                        if(row.parentNodeIndex==parentNode&&row.materialIndex==material)
                            rawNodeMaterial+=row.massKg;
                    for(const auto& row:rawContributions) {
                        if(row.parentNodeIndex!=parentNode||row.materialIndex!=material)continue;
                        const double amount=childMass/16.0*(row.massKg/rawNodeMaterial);
                        accumulateSourceMoments(expectedMoments[{childNode,row.donorBodyIndex,
                                                row.donorLocalPosition}],amount,row.donorLocalPosition);
                    }
                }
        }
        assert(frameMass.size()==splitSource.current.size());
        assert(sourceMassByNode.size()==splitSource.current.size());
        for(const auto& [node,mass]:frameMass)
            assert(std::abs(mass-sourceMassByNode[node])<=
                1.0e-9*std::max({std::abs(mass),std::abs(sourceMassByNode[node]),1.0e-12}));
        assert(expectedMoments.size()==actualMoments.size());
        for(const auto& [key,expected]:expectedMoments) {
            const auto& actual=actualMoments.at(key);
            assert(std::abs(actual.mass-expected.mass)<=1.0e-10*std::max(expected.mass,1.0e-12));
            for(unsigned axis=0;axis<3u;++axis)
                assert(std::abs(actual.first[axis]-expected.first[axis])<=1.0e-10*std::max(std::abs(expected.first[axis]),1.0e-12));
            for(unsigned component=0;component<9u;++component)
                assert(std::abs(actual.second[component]-expected.second[component])<=
                    1.0e-10*std::max(std::abs(expected.second[component]),1.0e-12));
        }
        bool secondDonorSeen=false,rawCookDifferenceSeen=false;
        for(const auto& [key,moments]:actualMoments) {
            if(std::get<1>(key)==1u)secondDonorSeen=true;
            (void)moments;
        }
        for(const auto& closure:report.sourceMassClosure) {
            if(closure.materialIndex==0u&&closure.donorBodyIndex==1u) {
                assert(std::abs(closure.childSourceMassKg-closure.cookedParentMassKg)<1.0e-10);
                assert(std::abs(closure.inputSourceMassKg-closure.cookedParentMassKg)>0.0);
                rawCookDifferenceSeen=true;
            }
        }
        assert(secondDonorSeen&&rawCookDifferenceSeen);
    }
    {
        Fixture badRawCook;
        badRawCook.bodyRotations.push_back(kIdentity);
        badRawCook.bodyOrigins.push_back({0.25,0.0,0.0});
        badRawCook.topology.registeredBodyLocalToWorldLinear=badRawCook.bodyRotations;
        badRawCook.topology.registeredBodyOriginWorld=badRawCook.bodyOrigins;
        std::vector<double> parentMass(badRawCook.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p) {
            const auto tet=badRawCook.parentTets[p];
            parentMass[p]=tetVolume(badRawCook.parentPoints[tet[0]],badRawCook.parentPoints[tet[1]],
                badRawCook.parentPoints[tet[2]],badRawCook.parentPoints[tet[3]])*
                (badRawCook.parentMaterial[p]==0u?1000.0:900.0);
        }
        std::vector<double> childMeasure(badRawCook.children.size(),0.0);
        for(std::size_t c=0;c<childMeasure.size();++c) {
            const auto tet=badRawCook.children[c];
            childMeasure[c]=tetVolume(badRawCook.reference[tet[0]],badRawCook.reference[tet[1]],
                badRawCook.reference[tet[2]],badRawCook.reference[tet[3]]);
        }
        badRawCook.sourceMassContributions=sourceContributionFixture(badRawCook,parentMass,1.001);
        auto badRawWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(badRawCook,parentMass,childMeasure,badRawWorld,a,b,report,error,
            1.0,1.0,badRawCook.sourceMassContributions));
        assert(error.find("cook-rounding bound")!=std::string::npos);
        assert(badRawWorld.objects.empty());
    }
    {
        Fixture ledger;
        std::vector<double> parentMass(ledger.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=ledger.parentTets[p];
            const double v=tetVolume(ledger.parentPoints[tet[0]],ledger.parentPoints[tet[1]],
                                     ledger.parentPoints[tet[2]],ledger.parentPoints[tet[3]]);
            parentMass[p]=v*(ledger.parentMaterial[p]==0u?1000.0:900.0);
        }
        parentMass[0]*=1.05; // Heterogeneous authoritative mass within the configured material bounds.
        std::vector<double> pulledMeasure(ledger.children.size(),0.0);
        for(std::size_t c=0;c<ledger.children.size();++c){
            const auto tet=ledger.children[c];
            pulledMeasure[c]=tetVolume(ledger.reference[tet[0]],ledger.reference[tet[1]],
                                       ledger.reference[tet[2]],ledger.reference[tet[3]]);
        }
        // Move only the appended interior reference point. The child geometric
        // volumes change, while the supplied reference-pullback measures stay
        // tied to the original parent material and still close its CSR columns.
        // Derive owner ancestry measures from this exact single-parent
        // barycentric refinement. The helper computes a conservative
        // parent-reference pullback, not a spatial overlap certificate.
        std::vector<double> parentReferenceVolumes(ledger.parentTets.size());
        for(std::size_t p=0;p<parentReferenceVolumes.size();++p)
            parentReferenceVolumes[p]=parentReferenceTetVolumeM3(ledger.parentTets[p],ledger.parentPoints);
        std::vector<double> childGeometricVolumes(ledger.children.size());
        for(std::size_t c=0;c<childGeometricVolumes.size();++c){
            const auto tet=ledger.children[c];
            childGeometricVolumes[c]=tetVolume(ledger.reference[tet[0]],ledger.reference[tet[1]],
                                               ledger.reference[tet[2]],ledger.reference[tet[3]]);
        }
        std::vector<std::uint32_t> transferOffsets,transferParents;
        std::vector<double> transferMeasures;
        assert(deriveSingleParentRefinementMeasures(ledger.parentPoints,ledger.parentTets,
            ledger.parentMaterial,ledger.children,ledger.childMaterial,ledger.dominant,
            ledger.nodeParent,ledger.nodeParentBary,transferOffsets,transferParents,transferMeasures,error));
        const std::array<double,2> densityLower{900.0,850.0},densityUpper{1100.0,950.0};
        ParentReferenceMassTransferInput transferInput{
            parentReferenceVolumes,parentMass,ledger.parentMaterial,ledger.parentDonor,
            childGeometricVolumes,ledger.childMaterial,transferOffsets,transferParents,transferMeasures,
            densityLower,densityUpper};
        ParentReferenceMassTransferResult transfer;
        assert(assembleParentReferenceMassTransfer(transferInput,transfer,error));
        assert(transfer.maximumParentMeasureClosureErrorM3<1.0e-14);
        assert(transfer.maximumParentMassClosureErrorKg<1.0e-14);
        ledger.topology.childParentAncestryOffsets=transfer.childParentOffsets;
        ledger.topology.childParentAncestryTetrahedra=transfer.childParentRows;
        ledger.topology.childParentAncestryVolumeFractions=transfer.childParentVolumeFractions;
        pulledMeasure=transfer.childAncestryMeasureVolumeM3;
        double transferSkinMass=0.0,transferFatMass=0.0;
        for(std::size_t c=0;c<transfer.childMassKg.size();++c)
            (ledger.childMaterial[c]==0u?transferSkinMass:transferFatMass)+=transfer.childMassKg[c];
        double expectedSkinMass=0.0,expectedFatMass=0.0;
        for(std::size_t p=0;p<parentMass.size();++p)
            (ledger.parentMaterial[p]==0u?expectedSkinMass:expectedFatMass)+=parentMass[p];
        assert(std::abs(transferSkinMass-expectedSkinMass)<1.0e-12);
        assert(std::abs(transferFatMass-expectedFatMass)<1.0e-12);
        ledger.reference[18][0]+=0.02;
        ledger.nodeParentResidual[18][0]=0.02;
        ledger.topology.referenceWorldPoints=ledger.reference;
        ledger.topology.volumeNodeParentResidualWorldM=ledger.nodeParentResidual;
        auto ledgerWorld=worldFor();
        ledgerWorld.materials[0].parameters[0].defaultValue=1050.0;
        ledgerWorld.materials[1].parameters[0].defaultValue=925.0;
        PreparedOrganicFEM ledgerSkin,ledgerFat;
        assert(emitMassLedger(ledger,parentMass,pulledMeasure,ledgerWorld,ledgerSkin,ledgerFat,report,error));
        assert(report.usesPerChildMassLedger);
        assert(ledgerWorld.objects.size()==1u);
        assert(ledgerWorld.objects.front().femTetrahedronMassesKg.size()==ledger.children.size());
        assert(ledgerWorld.objects.front().femTetrahedronMassSourceIdentity==
            (std::array<std::uint64_t,4>{0xaaaaaaaaaaaaaaaaULL,0xaaaaaaaaaaaaaaaaULL,
                                         0xaaaaaaaaaaaaaaaaULL,0xaaaaaaaaaaaaaaaaULL}));
        for(std::size_t child=0;child<ledger.children.size();++child)
            assert(ledgerWorld.objects.front().femTetrahedronMassesKg[child]>0.0);
        assert(report.parentAncestryMatchesNodeFrameMassOwnership);
        assert(report.nodeOwnerMassByDonor.size()==2u);
        assert(std::abs(report.skinParentMassKg-report.skinChildMassKg)<1.0e-10);
        assert(std::abs(report.fatParentMassKg-report.fatChildMassKg)<1.0e-10);
        assert(ledgerSkin.donorMassNodes.size()==report.volumeNodeCount);
        assert(ledgerFat.donorMassNodes.empty());
        assert(std::abs(sumDonorMass(ledgerSkin)-report.skinChildMassKg-report.fatChildMassKg)<1.0e-10);
        assert(ledgerSkin.donorMassContributions.empty()&&ledgerFat.donorMassContributions.empty());
        assert(report.donorMassClosure.size()==2u);
        for(const auto& closure:report.donorMassClosure)
            assert(std::abs(closure.parentMassKg-closure.childMassKg)<1.0e-10);
        assert(report.skinMinimumChildDensityKgM3>=1.0&&report.skinMaximumChildDensityKgM3<=5000.0);
        assert(report.fatMinimumChildDensityKgM3>=1.0&&report.fatMaximumChildDensityKgM3<=5000.0);
        assert(ledgerWorld.materials[0].parameters[0].defaultValue==1050.0);
        assert(ledgerWorld.materials[1].parameters[0].defaultValue==925.0);
        assert(std::abs(report.skinMeanImpliedDensityKgM3-report.skinChildMassKg/report.skinVolumeM3)<1.0e-10);

        // Exercise the production owner-to-compiler seam: the ancestry ledger's
        // child masses and source receipt must become compiled nodal inertia.
        auto compilerLedgerWorld=ledgerWorld;
#ifndef NUMI_TEST_MATERIAL_PATH
#error "NUMI_TEST_MATERIAL_PATH must identify the compiler integration test material"
#endif
        const auto compiledMaterial=numi::matter::parseMatterFile(
            std::filesystem::path(NUMI_TEST_MATERIAL_PATH));
        assert(compiledMaterial.succeeded());
        for(std::size_t material=0;material<compilerLedgerWorld.materials.size();++material) {
            compilerLedgerWorld.materials[material]=compiledMaterial.material;
            compilerLedgerWorld.materials[material].name=material==0u?"skin":"fat";
            auto density=std::find_if(compilerLedgerWorld.materials[material].parameters.begin(),
                compilerLedgerWorld.materials[material].parameters.end(),
                [](const auto& parameter){return parameter.name=="density";});
            assert(density!=compilerLedgerWorld.materials[material].parameters.end());
            density->defaultValue=ledgerWorld.materials[material].parameters[0].defaultValue;
        }
        auto densityBaselineWorld=compilerLedgerWorld;
        densityBaselineWorld.objects.front().femTetrahedronMassesKg.clear();
        densityBaselineWorld.objects.front().femTetrahedronMassSourceIdentity={};
        const auto densityBaseline=numi::matter::compileWorld(densityBaselineWorld);
        if(!densityBaseline.succeeded()) {
            for(const auto& diagnostic:densityBaseline.diagnostics)
                std::cerr<<diagnostic.message<<"; ";
        }
        assert(densityBaseline.succeeded());
        const auto compiledLedger=numi::matter::compileWorld(compilerLedgerWorld);
        if(!compiledLedger.succeeded()) {
            for(const auto& diagnostic:compiledLedger.diagnostics)
                std::cerr<<diagnostic.message<<"; ";
        }
        assert(compiledLedger.succeeded());
        const auto& sourceObject=ledgerWorld.objects.front();
        const auto& cookedLedger=compiledLedger.world;
        assert(cookedLedger.materials.size()==densityBaseline.world.materials.size());
        assert(std::memcmp(cookedLedger.materials.data(),densityBaseline.world.materials.data(),
            cookedLedger.materials.size()*sizeof(cookedLedger.materials[0]))==0);
        assert(cookedLedger.femTetrahedronMassSourceIdentities.size()==1u);
        assert(cookedLedger.femTetrahedronMassSourceIdentities.front()==
            sourceObject.femTetrahedronMassSourceIdentity);
        assert(cookedLedger.femTetrahedronMassesKg==sourceObject.femTetrahedronMassesKg);
        assert(cookedLedger.fem.nodes.size()==sourceObject.femNodes.size());
        std::vector<double> expectedCompiledNodeMass(sourceObject.femNodes.size(),0.0);
        for(std::size_t cell=0;cell<sourceObject.tetrahedra.size();++cell) {
            const auto tet=sourceObject.tetrahedra[cell];
            const double quarter=sourceObject.femTetrahedronMassesKg[cell]*0.25;
            for(const auto node:tet.nodes) expectedCompiledNodeMass[node]+=quarter;
        }
        for(std::size_t node=0;node<expectedCompiledNodeMass.size();++node) {
            const auto& state=cookedLedger.fem.nodes[node];
            const double actual=state.positionAndMass.w;
            assert(std::abs(actual-expectedCompiledNodeMass[node])<=
                std::max(1.0e-12,expectedCompiledNodeMass[node]*2.0e-6));
            const bool fixed=std::find(sourceObject.femFixedNodes.begin(),
                sourceObject.femFixedNodes.end(),static_cast<std::uint32_t>(node))!=
                sourceObject.femFixedNodes.end();
            if(fixed) assert(state.velocityAndInverseMass.w==0.0f);
            else assert(std::abs(double(state.velocityAndInverseMass.w)-1.0/actual)<=
                std::max(1.0e-5,2.0e-6/actual));
        }
        assert(std::abs(ledgerWorld.materials[0].parameters[0].defaultValue-1050.0)<1.0e-12);
        assert(std::abs(ledgerWorld.materials[1].parameters[0].defaultValue-925.0)<1.0e-12);
        const auto ledgerPackagePath=std::filesystem::temp_directory_path()/
            "numi-owner-emitted-explicit-mass-ledger.nmpkg";
        std::string ledgerPackageError;
        assert(numi::matter::writePackage(compiledLedger,ledgerPackagePath,&ledgerPackageError));
        numi::matter::CompiledWorld decodedLedger;
        assert(numi::matter::readPackage(ledgerPackagePath,decodedLedger,nullptr,&ledgerPackageError));
        assert(decodedLedger.femTetrahedronMassSourceIdentities==
            cookedLedger.femTetrahedronMassSourceIdentities);
        assert(decodedLedger.femTetrahedronMassesKg==sourceObject.femTetrahedronMassesKg);
        for(std::size_t node=0;node<expectedCompiledNodeMass.size();++node)
            assert(decodedLedger.fem.nodes[node].positionAndMass.w==
                cookedLedger.fem.nodes[node].positionAndMass.w);
        std::filesystem::remove(ledgerPackagePath);
    }
    {
        Fixture spatial;
        // The child flow retains source-donor contributions from both bodies;
        // all compiled volume nodes still use body 0 as their rigid frame owner.
        spatial.bodyRotations.push_back(kIdentity);
        spatial.bodyOrigins.push_back({0.25,0.0,0.0});
        spatial.topology.registeredBodyLocalToWorldLinear=spatial.bodyRotations;
        spatial.topology.registeredBodyOriginWorld=spatial.bodyOrigins;
        std::vector<double> parentMass(spatial.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=spatial.parentTets[p];
            parentMass[p]=parentReferenceTetVolumeM3(tet,spatial.parentPoints)*
                (spatial.parentMaterial[p]==0u?1000.0:900.0);
        }
        const auto sourceRows=sourceContributionFixture(spatial,parentMass,1.0+1.0e-5);
        ConservativeSpatialMassTransportResult spatialTransfer;
        if(!buildSpatialTransportFixture(spatial,parentMass,sourceRows,5.0,
                                         spatialTransfer,error)){
            std::cerr<<"spatial transport fixture assembly failed: "<<error<<"\n";
            return 26;
        }
        spatial.topology.ancestryVerificationKind=VolumeAncestryVerificationKind::unverified;
        spatial.topology.ancestryVerificationReceiptSHA256={};
        spatial.topology.massTransferKind=
            UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
        spatial.topology.spatialMassTransportResult=&spatialTransfer;
        spatial.topology.maximumSpatialMassTransportDistanceM=5.0;
        spatial.topology.parentTetrahedronMassKg=parentMass;
        spatial.topology.childAncestryMeasureVolumeM3={};
        spatial.topology.parentNodeSourceMassContributions=sourceRows;
        spatial.topology.childParentAncestryOffsets={};
        spatial.topology.childParentAncestryTetrahedra={};
        spatial.topology.childParentAncestryVolumeFractions={};
        spatial.topology.childDominantParentTetrahedra=spatialTransfer.childDominantParentRows;
        double skinDebit=0.0,fatDebit=0.0;
        for(std::size_t p=0;p<parentMass.size();++p)
            (spatial.parentMaterial[p]==0u?skinDebit:fatDebit)+=parentMass[p];
        auto spatialWorld=worldFor();PreparedOrganicFEM spatialSkin,spatialFat;
        if(!appendRegisteredSkinFatUnpairedVolume(spatialWorld,spatial.surface,spatial.boundary,
            spatial.topology,0u,"skin","test skin material",skinDebit,
            1u,"fat","test fat material",fatDebit,2u,"spatial mass-transfer test",
            spatialSkin,spatialFat,report,error)){
            std::cerr<<"spatial append failed: "<<error<<"\n";
            return 27;
        }
        assert(report.usesPerChildMassLedger&&report.usesParentNodeSourceMassContributions);
        assert(report.massTransferKind==
            UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport);
        assert(!report.massCouplingIsGeometricAncestry);
        assert(report.ancestryVerificationKind==VolumeAncestryVerificationKind::unverified);
        assert(report.ancestryVerificationReceiptSHA256.empty());
        assert(report.massTransferReceiptSHA256==spatialTransfer.receiptSHA256);
        assert(report.massTransportEdgeCount==spatialTransfer.transportEdgeCount);
        assert(spatialSkin.tetParentAncestryOffsets.empty());
        assert(spatialFat.tetParentAncestryOffsets.empty());
        assert(spatialWorld.objects.size()==1u);
        assert(spatialWorld.objects.front().femTetrahedronMassesKg==spatialTransfer.childMassKg);
        assert(spatialSkin.donorMassContributions.size()>spatialSkin.donorMassNodes.size());
        assert(report.sourceMassClosure.size()==3u);
        for(const auto& closure:report.sourceMassClosure)
            assert(std::abs(closure.cookedParentMassKg-closure.childSourceMassKg)<1.0e-10);

        RegisteredSkinFatReductionSourceLabels spatialLabels;
        const auto sourceBinding=[](std::uint32_t,std::uint32_t donor){return donor==0u;};
        const auto sourceParent=[](std::uint32_t){return NM_INVALID_INDEX;};
        const std::function<bool(std::uint32_t,std::uint32_t)> embeddedBinding=
            [](std::uint32_t,std::uint32_t donor){return donor==0u;};
        assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(spatial.surface,
            spatialWorld.objects.front(),spatial.boundary,spatial.topology,2u,
            sourceBinding,sourceParent,spatialLabels,error,36u,embeddedBinding));
        assert(error.find("owning materials")!=std::string::npos);
        if(!buildRegisteredSkinFatUnpairedReductionSourceLabels(spatial.surface,
            spatialWorld.objects.front(),spatial.boundary,spatial.topology,2u,
            sourceBinding,sourceParent,spatialLabels,error,36u,embeddedBinding,&spatialWorld)) {
            std::cerr<<"spatial source labels rejected: "<<error<<"\n";return 28;
        }
        assert(spatialLabels.massTransferKind==
            UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport);
        assert(spatialLabels.massTransferReceiptSHA256==spatialTransfer.receiptSHA256);
        assert(spatialLabels.ancestryVerificationReceiptSHA256.empty());
        assert(spatialLabels.tetrahedronParentAncestryOffsets.empty());
        assert(validateRegisteredSkinFatRemeshNodeAncestry(
            spatialWorld.objects.front(),spatialLabels,error));
        auto badSpatialLabels=spatialLabels;
        badSpatialLabels.massTransferReceiptSHA256[0]=
            badSpatialLabels.massTransferReceiptSHA256[0]=='a'?'b':'a';
        assert(!validateRegisteredSkinFatMassTransferLabels(
            spatialWorld.objects.front(),badSpatialLabels,error));
        badSpatialLabels=spatialLabels;
        badSpatialLabels.ancestryVerificationReceiptSHA256=spatialTransfer.receiptSHA256;
        assert(!validateRegisteredSkinFatMassTransferLabels(
            spatialWorld.objects.front(),badSpatialLabels,error));
        auto mismatchedMassShell=spatialWorld.objects.front();
        mismatchedMassShell.femTetrahedronMassesKg[0]*=1.000001;
        assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(spatial.surface,
            mismatchedMassShell,spatial.boundary,spatial.topology,2u,
            sourceBinding,sourceParent,badSpatialLabels,error,36u,embeddedBinding,&spatialWorld));
        assert(error.find("admitted physical shell")!=std::string::npos);
        assert(badSpatialLabels.sourceTetrahedronCount==0u);

        auto compilerSpatialWorld=spatialWorld;
#ifndef NUMI_TEST_MATERIAL_PATH
#error "NUMI_TEST_MATERIAL_PATH must identify the compiler integration test material"
#endif
        const auto spatialMaterial=numi::matter::parseMatterFile(
            std::filesystem::path(NUMI_TEST_MATERIAL_PATH));
        assert(spatialMaterial.succeeded());
        for(std::size_t material=0;material<compilerSpatialWorld.materials.size();++material){
            compilerSpatialWorld.materials[material]=spatialMaterial.material;
            compilerSpatialWorld.materials[material].name=material==0u?"skin":"fat";
            auto density=std::find_if(compilerSpatialWorld.materials[material].parameters.begin(),
                compilerSpatialWorld.materials[material].parameters.end(),
                [](const auto& parameter){return parameter.name=="density";});
            assert(density!=compilerSpatialWorld.materials[material].parameters.end());
            density->defaultValue=spatialWorld.materials[material].parameters[0].defaultValue;
        }
        const auto compiledSpatial=numi::matter::compileWorld(compilerSpatialWorld);
        if(!compiledSpatial.succeeded()){
            for(const auto& diagnostic:compiledSpatial.diagnostics)
                std::cerr<<diagnostic.message<<"; ";
        }
        assert(compiledSpatial.succeeded());
        assert(compiledSpatial.world.femTetrahedronMassesKg==
               spatialWorld.objects.front().femTetrahedronMassesKg);
        assert(compiledSpatial.world.fem.nodes.size()==spatialWorld.objects.front().femNodes.size());
        assert(verifyRegisteredSkinFatCompiledShell(compiledSpatial.world,0u,
            compilerSpatialWorld.objects.front(),spatialLabels,error));
        auto tamperedCook=compiledSpatial.world;
        tamperedCook.femTetrahedronMassesKg[0]*=1.000001;
        assert(!verifyRegisteredSkinFatCompiledShell(tamperedCook,0u,
            compilerSpatialWorld.objects.front(),spatialLabels,error));
        assert(error.find("cell mass differs")!=std::string::npos);
        badSpatialLabels=spatialLabels;
        badSpatialLabels.massTransferKind=UnpairedVolumeMassTransferKind::none;
        assert(!verifyRegisteredSkinFatCompiledShell(compiledSpatial.world,0u,
            compilerSpatialWorld.objects.front(),badSpatialLabels,error));
        for(std::size_t child=0;child<spatialTransfer.childMassKg.size();++child)
            assert(spatialWorld.objects.front().femTetrahedronMassesKg[child]==
                   spatialTransfer.childMassKg[child]);

        auto badFlow=spatialTransfer;
        badFlow.childParentTransportDistanceM[0]+=1.0e-4;
        auto badFlowTopology=spatial.topology;
        badFlowTopology.spatialMassTransportResult=&badFlow;
        auto badFlowWorld=worldFor();PreparedOrganicFEM badFlowSkin,badFlowFat;
        assert(!appendRegisteredSkinFatUnpairedVolume(badFlowWorld,spatial.surface,
            spatial.boundary,badFlowTopology,0u,"skin","test skin material",skinDebit,
            1u,"fat","test fat material",fatDebit,2u,"bad recorded flow",
            badFlowSkin,badFlowFat,report,error));
        assert(error.find("edge distance differs from current cells")!=std::string::npos);
        assert(badFlowWorld.objects.empty());

        auto badRegion=spatial.topology;
        auto changedRegions=spatial.childRegion;
        changedRegions[0]=999u;
        badRegion.childAnatomicalRegionIndices=changedRegions;
        auto badRegionWorld=worldFor();PreparedOrganicFEM badRegionSkin,badRegionFat;
        assert(!appendRegisteredSkinFatUnpairedVolume(badRegionWorld,spatial.surface,
            spatial.boundary,badRegion,0u,"skin","test skin material",skinDebit,
            1u,"fat","test fat material",fatDebit,2u,"bad region map",
            badRegionSkin,badRegionFat,report,error));
        assert(error.find("current-tet/material/region/source admission")!=std::string::npos);
        assert(badRegionWorld.objects.empty());
    }
    {
        Fixture crossed;
        // Source ancestry belongs to body 1 for one parent cell, while every
        // volume node remains transformed/attached by body 0. The owner must
        // reject rather than debit body 0 merely because it owns the node frame.
        crossed.parentDonor[0]=1u;
        for(std::size_t child=0;child<crossed.children.size();++child)
            crossed.childDonor[child]=crossed.parentDonor[crossed.dominant[child]];
        crossed.bodyRotations.push_back(kIdentity);
        crossed.bodyOrigins.push_back({1.0,0.0,0.0});
        crossed.refreshChildMetadata();
        crossed.topology.parentDonorBodyIndices=crossed.parentDonor;
        crossed.topology.registeredBodyLocalToWorldLinear=crossed.bodyRotations;
        crossed.topology.registeredBodyOriginWorld=crossed.bodyOrigins;
        std::vector<double> parentMass(crossed.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=crossed.parentTets[p];
            const double v=tetVolume(crossed.parentPoints[tet[0]],crossed.parentPoints[tet[1]],
                                     crossed.parentPoints[tet[2]],crossed.parentPoints[tet[3]]);
            parentMass[p]=v*(crossed.parentMaterial[p]==0u?1000.0:900.0);
        }
        std::vector<double> measure(crossed.children.size(),0.0);
        for(std::size_t c=0;c<crossed.children.size();++c){
            const auto tet=crossed.children[c];
            measure[c]=tetVolume(crossed.reference[tet[0]],crossed.reference[tet[1]],
                                 crossed.reference[tet[2]],crossed.reference[tet[3]]);
        }
        auto crossedWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(crossed,parentMass,measure,crossedWorld,a,b,report,error));
        assert(error.find("source-contribution mass-moment consumer is required")!=std::string::npos);
        assert(report.usesPerChildMassLedger&&!report.parentAncestryMatchesNodeFrameMassOwnership);
        assert(!report.nodeMassAttributionMismatches.empty());
        const auto sourceBodyOne=std::find_if(report.donorMassClosure.begin(),report.donorMassClosure.end(),
            [](const auto& x){return x.donorBodyIndex==1u;});
        assert(sourceBodyOne!=report.donorMassClosure.end()&&sourceBodyOne->parentMassKg>0.0);
        const auto ownerBodyOne=std::find_if(report.nodeOwnerMassByDonor.begin(),report.nodeOwnerMassByDonor.end(),
            [](const auto& x){return x.donorBodyIndex==1u;});
        assert(ownerBodyOne!=report.nodeOwnerMassByDonor.end());
        assert(ownerBodyOne->parentAncestryMassKg>ownerBodyOne->nodeFrameOwnerMassKg);
        assert(crossedWorld.objects.empty());
    }
    {
        Fixture badSource;
        badSource.bodyRotations.push_back(kIdentity);
        badSource.bodyOrigins.push_back({0.25,0.0,0.0});
        badSource.topology.registeredBodyLocalToWorldLinear=badSource.bodyRotations;
        badSource.topology.registeredBodyOriginWorld=badSource.bodyOrigins;
        std::vector<double> parentMass(badSource.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p) {
            const auto tet=badSource.parentTets[p];
            parentMass[p]=tetVolume(badSource.parentPoints[tet[0]],badSource.parentPoints[tet[1]],
                badSource.parentPoints[tet[2]],badSource.parentPoints[tet[3]])*
                (badSource.parentMaterial[p]==0u?1000.0:900.0);
        }
        std::vector<double> measure(badSource.children.size(),0.0);
        for(std::size_t c=0;c<measure.size();++c) {
            const auto tet=badSource.children[c];
            measure[c]=tetVolume(badSource.reference[tet[0]],badSource.reference[tet[1]],
                                 badSource.reference[tet[2]],badSource.reference[tet[3]]);
        }
        badSource.sourceMassContributions=sourceContributionFixture(badSource,parentMass,1.0);
        badSource.sourceMassContributions.front().donorBodyIndex=2u;
        auto badSourceWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(badSource,parentMass,measure,badSourceWorld,a,b,report,error,1.0,1.0,
                               badSource.sourceMassContributions));
        assert(error.find("invalid node, material, donor")!=std::string::npos);
        assert(badSourceWorld.objects.empty());
    }
    {
        Fixture badMass;
        std::vector<double> parentMass(badMass.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=badMass.parentTets[p];
            const double v=tetVolume(badMass.parentPoints[tet[0]],badMass.parentPoints[tet[1]],
                                     badMass.parentPoints[tet[2]],badMass.parentPoints[tet[3]]);
            parentMass[p]=v*(badMass.parentMaterial[p]==0u?1000.0:900.0);
        }
        parentMass[0]=-1.0e-6;
        std::vector<double> measure(badMass.children.size(),0.0);
        for(std::size_t c=0;c<badMass.children.size();++c){
            const auto tet=badMass.children[c];
            measure[c]=tetVolume(badMass.reference[tet[0]],badMass.reference[tet[1]],
                                 badMass.reference[tet[2]],badMass.reference[tet[3]]);
        }
        auto badMassWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(badMass,parentMass,measure,badMassWorld,a,b,report,error));
        assert(error.find("negative or nonfinite mass")!=std::string::npos&&badMassWorld.objects.empty());
    }
    {
        Fixture badDebit;
        std::vector<double> parentMass(badDebit.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=badDebit.parentTets[p];
            const double v=tetVolume(badDebit.parentPoints[tet[0]],badDebit.parentPoints[tet[1]],
                                     badDebit.parentPoints[tet[2]],badDebit.parentPoints[tet[3]]);
            parentMass[p]=v*(badDebit.parentMaterial[p]==0u?1000.0:900.0);
        }
        std::vector<double> measure(badDebit.children.size(),0.0);
        for(std::size_t c=0;c<badDebit.children.size();++c){
            const auto tet=badDebit.children[c];
            measure[c]=tetVolume(badDebit.reference[tet[0]],badDebit.reference[tet[1]],
                                 badDebit.reference[tet[2]],badDebit.reference[tet[3]]);
        }
        auto badDebitWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(badDebit,parentMass,measure,badDebitWorld,a,b,report,error,1.01));
        assert(error.find("does not close its per-material donor debit")!=std::string::npos);
        assert(badDebitWorld.objects.empty());
    }
    {
        Fixture exactGeometry;
        std::vector<double> parentMass(exactGeometry.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=exactGeometry.parentTets[p];
            const double v=tetVolume(exactGeometry.parentPoints[tet[0]],exactGeometry.parentPoints[tet[1]],
                                     exactGeometry.parentPoints[tet[2]],exactGeometry.parentPoints[tet[3]]);
            parentMass[p]=v*(exactGeometry.parentMaterial[p]==0u?1000.0:900.0);
        }
        std::vector<double> measure(exactGeometry.children.size(),0.0);
        for(std::size_t c=0;c<exactGeometry.children.size();++c){
            const auto tet=exactGeometry.children[c];
            measure[c]=tetVolume(exactGeometry.reference[tet[0]],exactGeometry.reference[tet[1]],
                                 exactGeometry.reference[tet[2]],exactGeometry.reference[tet[3]]);
        }
        exactGeometry.reference[18][0]+=0.02;
        exactGeometry.nodeParentResidual[18][0]=0.02;
        exactGeometry.topology.referenceWorldPoints=exactGeometry.reference;
        exactGeometry.topology.volumeNodeParentResidualWorldM=exactGeometry.nodeParentResidual;
        exactGeometry.topology.ancestryVerificationKind=VolumeAncestryVerificationKind::exactGeometricOverlap;
        auto exactWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(exactGeometry,parentMass,measure,exactWorld,a,b,report,error));
        assert(error.find("exact geometric-overlap mass measure differs")!=std::string::npos);
        assert(exactWorld.objects.empty());
    }
    {
        Fixture badDensity;
        std::vector<double> parentMass(badDensity.parentTets.size(),0.0);
        for(std::size_t p=0;p<parentMass.size();++p){
            const auto tet=badDensity.parentTets[p];
            const double v=tetVolume(badDensity.parentPoints[tet[0]],badDensity.parentPoints[tet[1]],
                                     badDensity.parentPoints[tet[2]],badDensity.parentPoints[tet[3]]);
            parentMass[p]=v*6000.0;
        }
        std::vector<double> measure(badDensity.children.size(),0.0);
        for(std::size_t c=0;c<badDensity.children.size();++c){
            const auto tet=badDensity.children[c];
            measure[c]=tetVolume(badDensity.reference[tet[0]],badDensity.reference[tet[1]],
                                 badDensity.reference[tet[2]],badDensity.reference[tet[3]]);
        }
        auto badDensityWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emitMassLedger(badDensity,parentMass,measure,badDensityWorld,a,b,report,error));
        assert(error.find("density outside material bounds")!=std::string::npos&&badDensityWorld.objects.empty());
    }
    {
        Fixture partialLedger;
        std::vector<double> onlyParentMass(partialLedger.parentTets.size(),1.0);
        partialLedger.topology.parentTetrahedronMassKg=onlyParentMass;
        auto partialWorld=worldFor();PreparedOrganicFEM a,b;
        assert(!emit(partialLedger,partialWorld,report,error));
        assert(error.find("requires one parent mass and one child ancestry measure")!=std::string::npos);
        assert(partialWorld.objects.empty());
    }
    {
        Fixture legacyDensity;
        auto legacyWorld=worldFor();
        legacyWorld.materials[0].parameters[0].defaultValue=1050.0;
        legacyWorld.materials[1].parameters[0].defaultValue=925.0;
        PreparedOrganicFEM legacyDensitySkin,legacyDensityFat;
        assert(emit(legacyDensity,legacyWorld,report,error,&legacyDensitySkin,&legacyDensityFat));
        assert(!report.usesPerChildMassLedger);
        assert(legacyDensitySkin.donorMassNodes.size()==report.volumeNodeCount);
        assert(legacyDensityFat.donorMassNodes.empty());
        assert(std::abs(legacyWorld.materials[0].parameters[0].defaultValue-1000.0)<1.0e-9);
        assert(std::abs(legacyWorld.materials[1].parameters[0].defaultValue-900.0)<1.0e-9);
    }
    RegisteredSkinFatReductionSourceLabels labels;
    assert(buildRegisteredSkinFatUnpairedReductionSourceLabels(f.surface,world.objects.front(),f.boundary,f.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},labels,error));
    assert(labels.sourceNodeLayerIndices.size()==19u&&labels.sourceNodeLayerIndices[18]==NM_INVALID_INDEX);
    assert(validateRegisteredSkinFatRemeshNodeAncestry(world.objects.front(),labels,error));
    {
        // One anatomical/source face can cross a mechanical donor partition.
        // The preconditioner must separate donors without changing that face ID.
        Fixture splitFace;
        splitFace.bodyRotations.push_back(kIdentity);
        splitFace.bodyOrigins.push_back({0,0,0});
        splitFace.topology.registeredBodyLocalToWorldLinear=splitFace.bodyRotations;
        splitFace.topology.registeredBodyOriginWorld=splitFace.bodyOrigins;
        splitFace.parentStable[1]=splitFace.parentStable[0];
        splitFace.parentDonor[1]=1u;
        for(std::size_t c=0;c<splitFace.children.size();++c) {
            splitFace.childStable[c]=splitFace.parentStable[splitFace.dominant[c]];
            splitFace.childDonor[c]=splitFace.parentDonor[splitFace.dominant[c]];
        }
        splitFace.refreshChildMetadata();
        RegisteredSkinFatReductionSourceLabels splitLabels;
        const auto bound=[](std::uint32_t,std::uint32_t donor){return donor<2u;};
        const auto parent=[](std::uint32_t){return NM_INVALID_INDEX;};
        if(!buildRegisteredSkinFatUnpairedReductionSourceLabels(splitFace.surface,
                world.objects.front(),splitFace.boundary,splitFace.topology,2u,
                bound,parent,splitLabels,error)) {
            std::cerr<<"shared source-face donor partition rejected: "<<error<<'\n';
            return 3;
        }
        assert(splitLabels.tetrahedronOriginalSourceFaceIndices==splitFace.childStable);
        assert(splitLabels.tetrahedronSourceDonorBodyIndices==splitFace.childDonor);
        std::map<std::uint32_t,std::uint32_t> basisByDonor;
        for(std::size_t c=0;c<splitFace.children.size();++c) {
            if(splitFace.childStable[c]!=splitFace.parentStable[0])continue;
            const auto donor=splitFace.childDonor[c];
            auto [entry,inserted]=basisByDonor.emplace(donor,splitLabels.tetrahedronBasisRegionLabels[c]);
            assert(inserted||entry->second==splitLabels.tetrahedronBasisRegionLabels[c]);
        }
        assert(basisByDonor.size()==2u&&basisByDonor.at(0u)!=basisByDonor.at(1u));
        // Multiple donor ownership still requires each cell's matching parent.
        splitFace.childDonor[0]=1u;
        assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(splitFace.surface,
            world.objects.front(),splitFace.boundary,splitFace.topology,2u,
            bound,parent,splitLabels,error));
        assert(error.find("invalid dominant donor or parent provenance")!=std::string::npos);
    }
    auto malformedShell=world.objects.front();
    malformedShell.tetrahedra[0].nodes[0]=19u;
    assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(f.surface,malformedShell,f.boundary,f.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},labels,error));
    assert(error.find("out-of-range node")!=std::string::npos);

    Fixture badSourceFaceShape;badSourceFaceShape.surface.sourceFaceIndices.pop_back();
    auto sourceFaceWorld=worldFor();assert(!emit(badSourceFaceShape,sourceFaceWorld,report,error));
    assert(sourceFaceWorld.objects.empty());

    Fixture badParent;auto badParentTets=badParent.parentTets;badParentTets[0][0]=999999u;
    badParent.topology.parentTetrahedra=badParentTets;auto badWorld=worldFor();
    assert(!emit(badParent,badWorld,report,error));
    assert(error.find("absent reference point")!=std::string::npos&&badWorld.objects.empty());

    Fixture badBoundary;badBoundary.children.erase(badBoundary.children.begin()+4);
    badBoundary.childMaterial.erase(badBoundary.childMaterial.begin()+4);
    badBoundary.dominant.erase(badBoundary.dominant.begin()+4);
    badBoundary.parentSurface.erase(badBoundary.parentSurface.begin()+4);
    badBoundary.childStable.erase(badBoundary.childStable.begin()+4);
    badBoundary.childDonor.erase(badBoundary.childDonor.begin()+4);
    badBoundary.refreshChildMetadata();
    badBoundary.topology.currentWorldPoints=badBoundary.current;
    auto boundaryWorld=worldFor();assert(!emit(badBoundary,boundaryWorld,report,error));
    assert(error.find("external boundary")!=std::string::npos&&boundaryWorld.objects.empty());

    Fixture collapsed;const auto p=collapsed.parentTets[0];
    V3 near=collapsed.current[p[0]];
    for(unsigned axis=0;axis<3u;++axis)near[axis]+=1.0e-10*(collapsed.current[18][axis]-near[axis]);
    collapsed.current[18]=near;collapsed.localPoints[18]=near;
    collapsed.topology.currentWorldPoints=collapsed.current;
    collapsed.topology.volumeNodeDonorLocalPoints=collapsed.localPoints;
    auto floatWorld=worldFor();assert(!emit(collapsed,floatWorld,report,error));
    assert(error.find("Float32 current orientation")!=std::string::npos&&floatWorld.objects.empty());

    Fixture independent;RegisteredSurface inner,deep;
    independentInnerFixture(independent,inner,deep);
    auto independentWorld=worldFor();
    if(!emit(independent,independentWorld,report,error)){std::cerr<<"independent boundary: "<<error;return 2;}
    assert(report.volumeNodeCount==20u&&independent.boundary.outer.size()==6u&&independent.boundary.skinInner.size()==7u&&independent.boundary.fatDeep.size()==6u);
    assert(inner.faces.size()==10u&&independent.surface.faces.size()==8u&&deep.faces.size()==8u);
    assert(buildRegisteredSkinFatUnpairedReductionSourceLabels(independent.surface,independentWorld.objects.front(),
        independent.boundary,independent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},labels,error,36u,
        [](std::uint32_t,std::uint32_t){return true;}));
    assert(labels.sourceNodeLayerIndices[19]==1u);
    assert((labels.registeredLayerVertexCounts==std::array<std::size_t,3>{6u,7u,6u}));
    assert((labels.registeredLayerFaceCounts==std::array<std::size_t,3>{8u,10u,8u}));
    assert(labels.registeredLayerVolumeNodeRows[1]==independent.boundary.skinInner);
    assert(labels.registeredLayerFaces[1]==inner.faces);
    assert(labels.registeredLayerStableFaceIdentities[1]==inner.sourceFaceIndices);
    assert(validateRegisteredSkinFatRemeshNodeAncestry(independentWorld.objects.front(),labels,error));
    auto wrongRoleMap=labels;
    wrongRoleMap.registeredLayerVolumeNodeRows[1][0]=wrongRoleMap.registeredLayerVolumeNodeRows[0][0];
    assert(!validateRegisteredSkinFatRemeshNodeAncestry(independentWorld.objects.front(),wrongRoleMap,error));
    auto wrongRoleFace=labels;
    wrongRoleFace.registeredLayerStableFaceIdentities[1][0]=kInvalid;
    assert(!validateRegisteredSkinFatRemeshNodeAncestry(independentWorld.objects.front(),wrongRoleFace,error));
    inner.sourceEmbeddingFaceIndices.back()=1u;
    assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(independent.surface,independentWorld.objects.front(),
        independent.boundary,independent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},labels,error,36u,
        [](std::uint32_t,std::uint32_t){return true;}));
    assert(labels.sourceNodeLayerIndices.empty());
    inner.sourceEmbeddingFaceIndices.back()=0u;
    const auto independentObject=independentWorld.objects.front();
    assert(independentObject.femNodes==independent.current);
    assert(independentObject.femReferenceNodes==independent.reference);
    assert(independentObject.deformableSelfContact); // Canonical node-identity FEM self-contact, independent path only.
    std::map<std::array<std::uint32_t,3>,unsigned> tetFacetUse;
    for(const auto& tet:independent.children)for(unsigned omit=0;omit<4u;++omit){
        std::array<std::uint32_t,3> face{};unsigned at=0;
        for(unsigned corner=0;corner<4u;++corner)if(corner!=omit)face[at++]=tet[corner];
        std::sort(face.begin(),face.end());++tetFacetUse[face];
    }
    std::vector<std::uint32_t> actualExteriorNodes;
    for(const auto& [face,count]:tetFacetUse)if(count==1u)
        actualExteriorNodes.insert(actualExteriorNodes.end(),face.begin(),face.end());
    std::sort(actualExteriorNodes.begin(),actualExteriorNodes.end());
    actualExteriorNodes.erase(std::unique(actualExteriorNodes.begin(),actualExteriorNodes.end()),actualExteriorNodes.end());
    assert(independentObject.femContactNodes==actualExteriorNodes);

    auto savedDeepFaceIds=deep.sourceFaceIndices;
    deep.sourceFaceIndices.clear();
    auto missingFaceIdsWorld=worldFor();
    assert(!emit(independent,missingFaceIdsWorld,report,error));
    assert(error.find("source-face ancestry identities")!=std::string::npos&&missingFaceIdsWorld.objects.empty());
    assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(independent.surface,independentObject,
        independent.boundary,independent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},
        labels,error,36u,[](std::uint32_t,std::uint32_t){return true;}));
    assert(labels.sourceNodeLayerIndices.empty());
    deep.sourceFaceIndices=std::move(savedDeepFaceIds);

    Fixture typedIndependent;RegisteredSurface typedInner,typedDeep;
    independentInnerFixture(typedIndependent,typedInner,typedDeep);
    typedIndependent.topology.independentSkinInnerSurface=&typedInner;
    typedIndependent.topology.independentFatDeepSurface=&typedDeep;
    const std::vector<PhysicalSkinSourceDomainRegistration> typedDomains{
        {"fixture-detail",PhysicalSkinSourceDomainRole::outerSurface,std::string(64,'a'),6u,8u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::innerSurface,std::string(64,'b'),6u,8u},
        {"fixture-detail",PhysicalSkinSourceDomainRole::deepSurface,std::string(64,'c'),6u,8u}};
    typedIndependent.surface.sourceDomainIndices.assign(typedIndependent.surface.worldPoints.size(),0u);
    typedInner.sourceDomainIndices.assign(typedInner.worldPoints.size(),1u);
    typedDeep.sourceDomainIndices.assign(typedDeep.worldPoints.size(),2u);
    std::vector<std::uint32_t> typedNodeDomains(typedIndependent.current.size(),kInvalid);
    for(const auto node:typedIndependent.boundary.outer)typedNodeDomains[node]=0u;
    for(const auto node:typedIndependent.boundary.skinInner)typedNodeDomains[node]=1u;
    for(const auto node:typedIndependent.boundary.fatDeep)typedNodeDomains[node]=2u;
    typedIndependent.topology.sourceDomains=typedDomains;
    typedIndependent.topology.volumeNodeSourceDomainIndices=typedNodeDomains;
    typedIndependent.topology.volumeNodeSourceVertexIndices=typedIndependent.sourceVertex;
    typedIndependent.topology.volumeNodeSourceFaceIndices=typedIndependent.sourceFace;
    typedIndependent.topology.volumeNodeSourceBarycentric=typedIndependent.sourceBary;
    typedIndependent.topology.detailedSourceVertexCount=0u;
    typedIndependent.topology.detailedSourceFaceCount=0u;
    auto typedWorld=worldFor();
    if(!emit(typedIndependent,typedWorld,report,error)){std::cerr<<"typed independent source domains: "<<error;return 3;}
    auto typedLabels=RegisteredSkinFatReductionSourceLabels{};
    const auto verifyTypedBinding=[](std::uint32_t domain,std::uint32_t vertex,std::uint32_t face,
            const std::array<double,3>& bary,std::uint32_t donor){
        return domain<3u&&donor==0u&&((vertex!=kInvalid&&face==kInvalid)||
            (vertex==kInvalid&&face==0u&&std::abs(bary[0]+bary[1]+bary[2]-1.0)<1.0e-12));
    };
    assert(buildRegisteredSkinFatUnpairedReductionSourceLabels(typedIndependent.surface,
        typedWorld.objects.front(),typedIndependent.boundary,typedIndependent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return kInvalid;},
        typedLabels,error,36u,[](std::uint32_t,std::uint32_t){return true;},nullptr,verifyTypedBinding));
    assert(typedLabels.sourceDomains==typedDomains&&typedLabels.sourceDomainFingerprint!=0u);
    assert(typedLabels.sourceNodeSourceDomainIndices.size()==typedIndependent.current.size());
    assert(typedLabels.sourceNodeSourceDomainIndices[18]==kInvalid);
    const auto rejectTypedBinding=[](std::uint32_t,std::uint32_t,std::uint32_t,
            const std::array<double,3>&,std::uint32_t){return false;};
    assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(typedIndependent.surface,
        typedWorld.objects.front(),typedIndependent.boundary,typedIndependent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return kInvalid;},
        typedLabels,error,36u,[](std::uint32_t,std::uint32_t){return true;},nullptr,rejectTypedBinding));
    assert(typedLabels.sourceNodeLayerIndices.empty());

    // Runtime compilation is covered by the separate owner integration receipt.\n    // Mismatched independent contracts must fail before source/density mutation.
    inner.sourceEmbeddingBarycentric.back()={.1,.1,.1};
    auto badIndependentBaryWorld=worldFor();
    assert(!emit(independent,badIndependentBaryWorld,report,error)&&badIndependentBaryWorld.objects.empty());
    inner.sourceEmbeddingBarycentric.back()={1.0/3.0,1.0/3.0,1.0/3.0};
    inner.sourceEmbeddingFaceIndices.back()=1u;
    auto mismatchedIndependentWorld=worldFor();
    assert(!emit(independent,mismatchedIndependentWorld,report,error)&&mismatchedIndependentWorld.objects.empty());
    inner.sourceEmbeddingFaceIndices.back()=0u;
    independent.topology.independentFatDeepSurface=nullptr;
    auto incompleteWorld=worldFor();
    assert(!emit(independent,incompleteWorld,report,error)&&incompleteWorld.objects.empty());
    independent.topology.independentFatDeepSurface=&deep;
    inner.worldPoints.back()[0]+=.01;
    auto staleLayerWorld=worldFor();
    assert(!emit(independent,staleLayerWorld,report,error)&&staleLayerWorld.objects.empty());
    inner.worldPoints.back()[0]-=.01;
    independent.boundary.skinInner.back()=independent.boundary.outer[0];
    auto aliasedWorld=worldFor();
    assert(!emit(independent,aliasedWorld,report,error)&&aliasedWorld.objects.empty());

    std::cout<<"unpaired skin/fat volume owner tests passed\n";
}
