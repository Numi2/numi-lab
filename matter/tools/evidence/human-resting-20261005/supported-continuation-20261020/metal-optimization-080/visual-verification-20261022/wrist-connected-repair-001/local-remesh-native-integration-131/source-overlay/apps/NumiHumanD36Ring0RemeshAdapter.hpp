#pragma once
#include "NumiHumanSkinFatLocalTetSplice.hpp"
#include "NumiHumanD36Ring0RemeshInput.hpp"
namespace metalrobo::human::coarse_tissue {
// Private integrated qualification input: same reference surface, new cavity
// tessellation. Representative parent IDs are centroid lineage, not overlap
// weights. This neither changes material laws nor advances physical state.
template<class HashBytes>
[[nodiscard]] inline bool applyD36Ring0Remesh(
        numi::matter::ObjectSource& object, PreparedOrganicFEM& skin,
        PreparedOrganicFEM& fat, RegisteredSkinFatReductionSourceLabels& labels,
        std::span<const std::uint32_t> oldParentSurfaceRows,
        LocalSkinFatTetSpliceReport& report, std::string& error, HashBytes hash) {
    using namespace d36_ring0_remesh;
    const auto fail=[&](const char* why){error=why;return false;};
    if(object.femNodes.size()!=7233u||object.tetrahedra.size()!=28632u||
       object.femMaterialIndices.size()!=object.tetrahedra.size()||
       labels.sourceNodeVertexIndices.size()!=object.femNodes.size()||
       labels.sourceNodeLayerIndices.size()!=object.femNodes.size()||
       labels.sourceNodeDonorBodyIndices.size()!=object.femNodes.size()||
       oldParentSurfaceRows.size()!=object.tetrahedra.size())
        return fail("ring0 diagnostic is not bound to the D36 shell dimensions");
    for(std::size_t i=0;i<kReplacedTetRows.size();++i) {
        const auto t=kReplacedTetRows[i];
        const auto material=kExpectedMaterialKinds[i]==1u?skin.materialIndex:fat.materialIndex;
        if(object.tetrahedra[t].nodes!=kExpectedParentTets[i]||
           object.femMaterialIndices[t]!=material)
            return fail("ring0 diagnostic source connectivity/material differs from D36");
    }
    std::vector<V3> donorLocal(object.femNodes.size());
    std::vector<bool> seen(object.femNodes.size(),false);
    for(const auto& row:skin.donorMassNodes) {
        if(row.nodeIndex<skin.firstGlobalFEMNode)return fail("ring0 donor mass precedes shell");
        const auto n=row.nodeIndex-skin.firstGlobalFEMNode;
        if(n>=seen.size()||seen[n]||row.donorBody!=labels.sourceNodeDonorBodyIndices[n])
            return fail("ring0 donor-local mass ownership is incomplete or repeated");
        seen[n]=true;donorLocal[n]=row.localPosition;
    }
    if(std::find(seen.begin(),seen.end(),false)!=seen.end())
        return fail("ring0 donor-local reference does not cover every existing node");
    std::vector<LocalSkinFatTetNode> nodes;
    nodes.reserve(kExistingNodeIDs.size());
    for(const auto n:kExistingNodeIDs) {
        LocalSkinFatTetNode row;row.existingObjectNode=n;
        row.world=object.femNodes[n];row.referenceWorld=object.femReferenceNodes.empty()?
            row.world:object.femReferenceNodes[n];row.donorLocal=donorLocal[n];
        row.donorBody=labels.sourceNodeDonorBodyIndices[n];
        row.sourceVertex=labels.sourceNodeVertexIndices[n];row.layerOrInterior=labels.sourceNodeLayerIndices[n];
        nodes.push_back(row);
    }
    std::vector<std::uint32_t> material,sourceFace,donor,basis,parentSurface;
    for(const auto parent:kChildParentTetRows) {
        material.push_back(object.femMaterialIndices[parent]);
        sourceFace.push_back(labels.tetrahedronOriginalSourceFaceIndices.empty()?
            labels.tetrahedronSourceFaceIndices[parent]:labels.tetrahedronOriginalSourceFaceIndices[parent]);
        donor.push_back(labels.tetrahedronSourceDonorBodyIndices[parent]);
        basis.push_back(labels.tetrahedronBasisRegionLabels[parent]);
        parentSurface.push_back(oldParentSurfaceRows[parent]);
    }
    const auto identity=[&](const char* domain,const std::array<std::uint64_t,4>& prior) {
        std::string text(domain);text+='|';text+=kInputSHA256;
        text.append(reinterpret_cast<const char*>(prior.data()),sizeof(prior));
        const auto digest=hash(text.data(),text.size());
        std::array<std::uint64_t,4> result{};
        for(unsigned i=0;i<4u;++i)for(unsigned j=0;j<8u;++j)
            result[i]=(result[i]<<8u)|digest[8u*i+j];
        return result;
    };
    LocalSkinFatTetSpliceInput in;
    in.replacedTetRows=kReplacedTetRows;in.oldTetParentSurfaceFaceRows=oldParentSurfaceRows;
    in.oldNodeDonorLocalPoints=donorLocal;in.nodes=nodes;
    in.childTetrahedra=kChildLocalTets;in.childMaterialIndices=material;
    in.childParentTetRows=kChildParentTetRows;in.childParentSurfaceFaceRows=parentSurface;
    in.childSourceFaceIdentities=sourceFace;in.childDonorBodyLabels=donor;
    in.childBasisRegionLabels=basis;
    in.materialSourceIdentity=identity("numi.human.d36.ring0.material-lineage.v1",object.femMaterialSourceIdentity);
    if(!object.femMaterialFrameRotations.empty())
        in.materialFrameSourceIdentity=identity("numi.human.d36.ring0.frame-lineage.v1",object.femMaterialFrameSourceIdentity);
    auto nextLabels=labels;
    if(!spliceRegisteredSkinFatLocalTets(object,skin,fat,labels,in,report,nextLabels,error))return false;
    labels=std::move(nextLabels);return true;
}
}
