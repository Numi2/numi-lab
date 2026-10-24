#include "NumiHumanSkinFatReductionLabels.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
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
    std::vector<std::uint32_t> parentMaterial,parentStable,parentDonor;
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
        topology.parentDonorBodyIndices=parentDonor;
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

numi::matter::WorldSource worldFor() {
    numi::matter::WorldSource world;world.materials.resize(2u);
    for(std::uint32_t i=0;i<2u;++i) {
        world.materials[i].name=i==0u?"skin":"fat";
        world.materials[i].supportedRepresentations={Representation::fem};
        numi::matter::Parameter density;density.name="density";density.defaultValue=1000.0;
        density.lower=1.0;density.upper=5000.0;world.materials[i].parameters.push_back(density);
    }
    return world;
}

bool emit(Fixture& f,numi::matter::WorldSource& world,ExplicitUnpairedVolumeReport& report,std::string& error) {
    double skinVolume=0.0,fatVolume=0.0;
    for(std::size_t i=0;i<f.children.size();++i) {
        const auto tet=f.children[i];
        const double v=tetVolume(f.reference[tet[0]],f.reference[tet[1]],f.reference[tet[2]],f.reference[tet[3]]);
        (f.childMaterial[i]==0u?skinVolume:fatVolume)+=v;
    }
    PreparedOrganicFEM skin,fat;
    return appendRegisteredSkinFatUnpairedVolume(world,f.surface,f.boundary,f.topology,
        0u,"skin","test skin material",skinVolume*1000.0,1u,"fat","test fat material",
        fatVolume*1000.0,1u,"unpaired test shell",skin,fat,report,error);
}
}

int main() {
    Fixture f;auto world=worldFor();ExplicitUnpairedVolumeReport report;std::string error;
    if(!emit(f,world,report,error)){std::cerr<<error;return 1;}
    assert(world.objects.size()==1u&&report.volumeNodeCount==19u);
    assert(report.inputVolumeRowToObjectLocalNode.size()==19u&&report.inputVolumeRowToObjectLocalNode[18]==18u);
    assert(report.ancestryVerificationKind==VolumeAncestryVerificationKind::conservativeReferencePullback);
    RegisteredSkinFatReductionSourceLabels labels;
    assert(buildRegisteredSkinFatUnpairedReductionSourceLabels(f.surface,world.objects.front(),f.boundary,f.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},labels,error));
    assert(labels.sourceNodeLayerIndices.size()==19u&&labels.sourceNodeLayerIndices[18]==NM_INVALID_INDEX);
    assert(validateRegisteredSkinFatRemeshNodeAncestry(world.objects.front(),labels,error));
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
    std::cout<<"unpaired skin/fat volume owner tests passed\n";
}
