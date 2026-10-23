#include "NumiHumanSkinFatLocalRemeshLabels.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using namespace metalrobo::human::coarse_tissue;
using numi::matter::ObjectSource;
struct Report {
    std::vector<std::uint32_t> nodeParentTetRows,nodeSourceVertexIndices,
        nodeLayerOrInterior,nodeDonorBodyIndices,tetParentTetRows,
        tetMaterialIndices,tetSourceFaceIdentities,tetDonorBodyLabels,tetBasisRegionLabels;
    std::vector<std::array<double,4>> nodeParentBarycentric;
};
int main() {
    ObjectSource before;
    before.femNodes={{{0,0,0}},{{1,0,0}},{{0,1,0}},{{1,1,0}},{{0,0,1}},{{1,0,1}}};
    before.femReferenceNodes=before.femNodes;
    before.tetrahedra={{{{0,1,2,4}}},{{{1,3,4,5}}}};
    before.femMaterialIndices={0,1};
    RegisteredSkinFatReductionSourceLabels labels;
    labels.registeredVertexCount=2; labels.registeredFaceCount=2; labels.sourceTetrahedronCount=2;
    labels.sourceNodeVertexIndices={10,11,10,11,10,11};
    labels.sourceNodeLayerIndices={0,0,1,1,2,2};
    labels.sourceNodeDonorBodyIndices={1,1,1,1,1,1};
    labels.tetrahedronSourceFaceIndices={100,101};
    labels.tetrahedronSourceDonorBodyIndices={1,1};
    labels.tetrahedronSourceParentBodyIndices={0,0};
    labels.tetrahedronRegionLabels={1,1};
    labels.tetrahedronBasisRegionLabels={21,21};
    labels.donorBodyAncestry[1]={1,0};
    ObjectSource after=before;
    after.femNodes.push_back({.25,.25,.25});
    after.femReferenceNodes=after.femNodes;
    after.tetrahedra={{{{6,1,2,4}}},{{{0,6,2,4}}},{{{0,1,6,4}}},{{{0,1,2,6}}},before.tetrahedra[1]};
    after.femMaterialIndices={0,0,0,0,1};
    Report report;
    report.nodeParentTetRows.assign(7,NM_INVALID_INDEX);report.nodeParentTetRows[6]=0;
    report.nodeParentBarycentric.resize(7); report.nodeParentBarycentric[6]={.25,.25,.25,.25};
    report.nodeSourceVertexIndices=labels.sourceNodeVertexIndices;report.nodeSourceVertexIndices.push_back(NM_INVALID_INDEX);
    report.nodeLayerOrInterior=labels.sourceNodeLayerIndices;report.nodeLayerOrInterior.push_back(NM_INVALID_INDEX);
    report.nodeDonorBodyIndices.assign(7,1);
    report.tetParentTetRows={0,0,0,0,1};report.tetMaterialIndices={0,0,0,0,1};
    report.tetSourceFaceIdentities={1000,1001,1002,1003,101};
    report.tetDonorBodyLabels.assign(5,1);report.tetBasisRegionLabels.assign(5,21);
    RegisteredSkinFatReductionSourceLabels output;std::string error;
    assert(remapRegisteredSkinFatReductionSourceLabels(before,after,labels,report,output,error));
    assert(output.remeshedOriginalNodeCount==6&&output.registeredVertexCount==2);
    assert(output.sourceNodeParentTetrahedronIndices[6]==0);
    assert((output.tetrahedronOriginalSourceFaceIndices==std::vector<std::uint32_t>{100,100,100,100,101}));
    const auto good=output;
    numi::matter::CompiledWorld world;
    world.objects.resize(1);
    auto& object=world.objects[0];
    object.representation=NM_REPRESENTATION_FEM;object.stateCount=after.femNodes.size();
    object.elementCount=after.tetrahedra.size();object.topologyGeneration=1;
    for(std::uint32_t i=0;i<after.femNodes.size();++i) {
        const auto& p=after.femNodes[i];NMFEMNodeStateGPU node{};
        node.positionAndMass={float(p[0]),float(p[1]),float(p[2]),1.f};
        world.fem.nodes.push_back(node);NMFEMTopologyNodeGPU topology{};
        topology.identity={i,0,1,NM_TOPOLOGY_ACTIVE};world.fem.topologyNodes.push_back(topology);
    }
    for(std::uint32_t i=0;i<after.tetrahedra.size();++i) {
        const auto& v=after.tetrahedra[i].nodes;NMTetrahedronGPU tet{};
        tet.nodes={v[0],v[1],v[2],v[3]};tet.identity={i,0,1,NM_OBJECT_ACTIVE};
        world.fem.tetrahedra.push_back(tet);
    }
    assert(verifyRegisteredSkinFatCompiledShell(world,0,after,good,error));
    auto badWorld=world;badWorld.fem.nodes[6].positionAndMass.x+=.001f;
    assert(!verifyRegisteredSkinFatCompiledShell(badWorld,0,after,good,error));
    badWorld=world;badWorld.fem.tetrahedra[0].nodes.x=0;
    assert(!verifyRegisteredSkinFatCompiledShell(badWorld,0,after,good,error));
    int rejected=0;
    auto reject=[&](const ObjectSource& b,const ObjectSource& a,const RegisteredSkinFatReductionSourceLabels& l,const Report& r) {
        output=good;
        assert(!remapRegisteredSkinFatReductionSourceLabels(b,a,l,r,output,error));
        assert(!error.empty());assert(output.sourceNodeParentBarycentric==good.sourceNodeParentBarycentric);
        assert(output.tetrahedronOriginalSourceFaceIndices==good.tetrahedronOriginalSourceFaceIndices);
        ++rejected;
    };
    auto r=report;r.nodeParentBarycentric[6][0]=.5;reject(before,after,labels,r);
    r=report;r.nodeParentBarycentric[6][0]=std::numeric_limits<double>::quiet_NaN();reject(before,after,labels,r);
    r=report;r.nodeParentTetRows[6]=2;reject(before,after,labels,r);
    r=report;r.nodeDonorBodyIndices[6]=2;reject(before,after,labels,r);
    r=report;r.nodeLayerOrInterior[6]=3;reject(before,after,labels,r);
    r=report;r.tetMaterialIndices[0]=1;reject(before,after,labels,r);
    r=report;r.tetParentTetRows[0]=1;reject(before,after,labels,r);
    r=report;r.tetDonorBodyLabels[0]=2;reject(before,after,labels,r);
    r=report;r.tetSourceFaceIdentities.pop_back();reject(before,after,labels,r);
    auto a=after;a.femNodes[0][0]=.01;reject(before,a,labels,report);
    a=after;a.femReferenceNodes[6][0]+=.01;reject(before,a,labels,report);
    a=after;a.femReferenceNodes[6][0]=std::numeric_limits<double>::quiet_NaN();reject(before,a,labels,report);
    auto b=before;b.femMaterialIndices.clear();reject(b,after,labels,report);
    auto l=labels;l.tetrahedronOriginalSourceFaceIndices={100};reject(before,after,l,report);
    l=labels;l.sourceNodeLayerIndices[0]=NM_INVALID_INDEX;reject(before,after,l,report);
    // A second splice preserves already certified interior ancestry.
    Report identity=report;
    identity.tetParentTetRows={0,1,2,3,4};
    assert(remapRegisteredSkinFatReductionSourceLabels(after,after,good,identity,output,error));
    assert(output.sourceNodeParentBarycentric==good.sourceNodeParentBarycentric);
    // Two parent donors must never share one basis label.
    b=before;l=labels;r=report;
    l.tetrahedronSourceDonorBodyIndices[1]=2;l.donorBodyAncestry[2]={2,0};
    r.tetDonorBodyLabels[4]=2;reject(b,after,l,r);
    std::cout<<"local remesh label checks passed; malformed cases rejected="<<rejected<<"\n";
}
