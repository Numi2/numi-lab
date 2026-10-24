#include "NumiHumanSkinFatReductionLabels.hpp"
#include "NumiHumanUnpairedSkinFatContactDispatch.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using numi::matter::Representation;
namespace {
void installCompilerMaterials(numi::matter::WorldSource& world) {
    const char* root=std::getenv("NUMI_MATTER_SOURCE_ROOT");
    if(root==nullptr||*root=='\0')throw std::runtime_error("NUMI_MATTER_SOURCE_ROOT is required for compile integration");
    const auto parsed=numi::matter::parseMatterFile(std::filesystem::path(root)/"materials"/"damageable_silicone.nmatter");
    if(!parsed.succeeded())throw std::runtime_error("compiler integration material did not parse");
    auto skin=parsed.material;auto fat=parsed.material;
    skin.name="skin";fat.name="fat";
    for(auto& parameter:skin.parameters)parameter.identifiable=false;
    for(auto& parameter:fat.parameters)parameter.identifiable=false;
    world.materials={std::move(skin),std::move(fat)};
    world.gravity={0.0,0.0,0.0};
}

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
    assert(!world.objects.front().deformableSelfContact); // legacy paired rows retain the old contact owner.
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

    Fixture independent;RegisteredSurface inner,deep;
    independentInnerFixture(independent,inner,deep);
    auto independentWorld=worldFor();
    installCompilerMaterials(independentWorld);
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
    installCompilerMaterials(missingFaceIdsWorld);
    assert(!emit(independent,missingFaceIdsWorld,report,error));
    assert(error.find("source-face ancestry identities")!=std::string::npos&&missingFaceIdsWorld.objects.empty());
    assert(!buildRegisteredSkinFatUnpairedReductionSourceLabels(independent.surface,independentObject,
        independent.boundary,independent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},
        labels,error,36u,[](std::uint32_t,std::uint32_t){return true;}));
    assert(labels.sourceNodeLayerIndices.empty());
    deep.sourceFaceIndices=std::move(savedDeepFaceIds);
    assert(buildRegisteredSkinFatUnpairedReductionSourceLabels(independent.surface,independentObject,
        independent.boundary,independent.topology,1u,
        [](std::uint32_t,std::uint32_t){return true;},[](std::uint32_t){return NM_INVALID_INDEX;},
        labels,error,36u,[](std::uint32_t,std::uint32_t){return true;}));

    // Prove the owner opts into Matter's canonical FEM same-object contact
    // compilation only for independent-role topology. The cooked exterior
    // facets stay enabled and are derived from actual count-one tet facets.
    numi::matter::CompileOptions compileOptions;
    compileOptions.emitSpecializedMetal=false;
    const auto cooked=numi::matter::compileWorld(independentWorld,compileOptions);
    if(!cooked.succeeded()){
        for(const auto& diagnostic:cooked.diagnostics)std::cerr<<diagnostic.message<<"; ";
        return 3;
    }
    assert(cooked.world.objects.size()==1u);
    assert((cooked.world.objects.front().flags&NM_OBJECT_DISABLE_SELF_CONTACT)==0u);
    assert(numi::human::usesIndependentSkinFatCanonicalContact(&labels));
    assert(!numi::human::usesIndependentSkinFatCanonicalContact<RegisteredSkinFatReductionSourceLabels>(nullptr));
    numi::human::IndependentSkinFatCanonicalContactAudit canonicalAudit;
    if(!numi::human::validateIndependentSkinFatCanonicalContact(
        cooked.world,independentObject,labels,0u,canonicalAudit,error)){
        std::cerr<<"canonical dispatch audit: "<<error<<"\n";return 4;
    }
    assert(canonicalAudit.expectedExteriorFaceCount>0u);
    assert(canonicalAudit.compiledExteriorFaceCount==canonicalAudit.expectedExteriorFaceCount);
    assert(canonicalAudit.expectedContactFaceCount==canonicalAudit.compiledContactFaceCount);
    assert(canonicalAudit.compiledContactFaceCount>0u);
    auto noMetadata=labels;noMetadata.explicitUnpairedVolumeTopology=false;
    assert(!numi::human::usesIndependentSkinFatCanonicalContact(&noMetadata));
    assert(!numi::human::validateIndependentSkinFatCanonicalContact(
        cooked.world,independentObject,noMetadata,0u,canonicalAudit,error));
    auto disabledCanonical=cooked.world;
    disabledCanonical.objects[0].flags|=NM_OBJECT_DISABLE_SELF_CONTACT;
    assert(!numi::human::validateIndependentSkinFatCanonicalContact(
        disabledCanonical,independentObject,labels,0u,canonicalAudit,error));
    auto missingCanonicalFace=cooked.world;
    missingCanonicalFace.fem.surfaceFaces.pop_back();
    assert(!numi::human::validateIndependentSkinFatCanonicalContact(
        missingCanonicalFace,independentObject,labels,0u,canonicalAudit,error));
    std::size_t expectedExterior=0u;
    for(const auto& [face,count]:tetFacetUse){(void)face;if(count==1u)++expectedExterior;}
    std::size_t cookedExterior=0u;
    for(const auto& face:cooked.world.fem.surfaceFaces){
        if(face.adjacency.y==NM_INVALID_INDEX){
            ++cookedExterior;
            assert((face.sides.w&NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT)==0u);
        }
    }
    assert(cookedExterior==expectedExterior);
    auto selfContactDisabledSource=independentWorld;
    selfContactDisabledSource.objects.front().deformableSelfContact=false;
    const auto selfContactDisabledCook=numi::matter::compileWorld(selfContactDisabledSource,compileOptions);
    assert(selfContactDisabledCook.succeeded());
    assert((selfContactDisabledCook.world.objects.front().flags&NM_OBJECT_DISABLE_SELF_CONTACT)!=0u);
    assert(selfContactDisabledCook.world.fem.surfaceFaces.size()==cooked.world.fem.surfaceFaces.size());
    for(std::size_t face=0;face<cooked.world.fem.surfaceFaces.size();++face){
        const auto& a=cooked.world.fem.surfaceFaces[face];const auto& b=selfContactDisabledCook.world.fem.surfaceFaces[face];
        assert(a.adjacency.x==b.adjacency.x&&a.adjacency.y==b.adjacency.y&&a.adjacency.z==b.adjacency.z&&a.adjacency.w==b.adjacency.w);
        assert(a.sides.x==b.sides.x&&a.sides.y==b.sides.y&&a.sides.z==b.sides.z&&a.sides.w==b.sides.w);
    }
    // Mismatched independent contracts must fail before source/density mutation.
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
    // The dispatch predicate keeps the legacy paired/custom-column path on the
    // existing binder and verifies its postcompile mutation remains available.
    Fixture legacyFixture;auto legacySource=worldFor();installCompilerMaterials(legacySource);
    legacySource.compactImmutableFEMSurfaces=true;
    ExplicitUnpairedVolumeReport legacyReport;
    assert(emit(legacyFixture,legacySource,legacyReport,error));
    RegisteredSkinFatReductionSourceLabels legacyLabels;
    assert(!numi::human::usesIndependentSkinFatCanonicalContact(&legacyLabels));
    const auto legacyCook=numi::matter::compileWorld(legacySource,compileOptions);
    assert(legacyCook.succeeded());
    std::vector<std::uint32_t> legacyColumns(legacySource.objects.front().femNodes.size(),0u);
    for(std::uint32_t row=0u;row<legacyFixture.boundary.outer.size();++row)
        for(const auto node:{legacyFixture.boundary.outer[row],legacyFixture.boundary.skinInner[row],legacyFixture.boundary.fatDeep[row]})
            legacyColumns[node]=row+1u;
    auto legacyCompiled=legacyCook.world;
    const auto legacyAudit=numi::matter::human::bindHumanThreeLayerMaterialColumnSelfContact(
        legacyCompiled,legacySource.objects.front(),0u,
        static_cast<std::uint32_t>(legacyFixture.boundary.outer.size()),legacyColumns);
    assert(legacyAudit.objectIndex==0u&&legacyAudit.nodeCount==legacySource.objects.front().femNodes.size());
    assert((legacyCompiled.objects[0].flags&NM_OBJECT_DISABLE_SELF_CONTACT)==0u);

    inner.worldPoints.back()[0]+=.01;
    auto staleLayerWorld=worldFor();
    assert(!emit(independent,staleLayerWorld,report,error)&&staleLayerWorld.objects.empty());
    inner.worldPoints.back()[0]-=.01;
    independent.boundary.skinInner.back()=independent.boundary.outer[0];
    auto aliasedWorld=worldFor();
    assert(!emit(independent,aliasedWorld,report,error)&&aliasedWorld.objects.empty());

    std::cout<<"unpaired skin/fat volume owner tests passed\n";
}
