#pragma once
#include "NumiHumanSkinFatReductionLabels.hpp"
namespace metalrobo::human::coarse_tissue {
// Applies the existing local-splice report to source labels. No forces,
// material state, geometry, or runtime coordinates are evaluated here.
template<class SpliceReport>
[[nodiscard]] inline bool remapRegisteredSkinFatReductionSourceLabels(
        const numi::matter::ObjectSource& before,
        const numi::matter::ObjectSource& after,
        const RegisteredSkinFatReductionSourceLabels& original,
        const SpliceReport& splice,
        RegisteredSkinFatReductionSourceLabels& output,
        std::string& error) {
    const auto fail=[&](const char* why){error=why;return false;};
    const auto old=before.femNodes.size(),n=after.femNodes.size(),nt=after.tetrahedra.size();
    if(old==0u||n<old||before.tetrahedra.size()!=original.sourceTetrahedronCount||
       original.sourceNodeVertexIndices.size()!=old||
       original.sourceNodeLayerIndices.size()!=old||
       original.sourceNodeDonorBodyIndices.size()!=old||
       before.femMaterialIndices.size()!=before.tetrahedra.size()||
       original.tetrahedronSourceFaceIndices.size()!=before.tetrahedra.size()||
       original.tetrahedronSourceDonorBodyIndices.size()!=before.tetrahedra.size()||
       original.tetrahedronSourceParentBodyIndices.size()!=before.tetrahedra.size()||
       (!original.tetrahedronOriginalSourceFaceIndices.empty()&&
        original.tetrahedronOriginalSourceFaceIndices.size()!=before.tetrahedra.size())||
       splice.nodeParentTetRows.size()!=n||splice.nodeParentBarycentric.size()!=n||
       splice.nodeSourceVertexIndices.size()!=n||splice.nodeLayerOrInterior.size()!=n||
       splice.nodeDonorBodyIndices.size()!=n||splice.tetParentTetRows.size()!=nt||
       splice.tetMaterialIndices.size()!=nt||splice.tetSourceFaceIdentities.size()!=nt||
       splice.tetDonorBodyLabels.size()!=nt||splice.tetBasisRegionLabels.size()!=nt||
       after.femMaterialIndices.size()!=nt||
       (!before.femReferenceNodes.empty()&&before.femReferenceNodes.size()!=old)||
       (!after.femReferenceNodes.empty()&&after.femReferenceNodes.size()!=n))
        return fail("local skin/fat splice labels have incomplete node/tet coverage");
    if(!validateRegisteredSkinFatRemeshNodeAncestry(before,original,error))return false;
    auto next=original;
    next.remeshedOriginalNodeCount=original.remeshedOriginalNodeCount?
        original.remeshedOriginalNodeCount:old;
    next.sourceNodeVertexIndices=splice.nodeSourceVertexIndices;
    next.sourceNodeLayerIndices=splice.nodeLayerOrInterior;
    next.sourceNodeDonorBodyIndices=splice.nodeDonorBodyIndices;
    next.sourceNodeParentTetrahedronIndices.assign(n,NM_INVALID_INDEX);
    next.sourceNodeParentBarycentric.assign(n,{});
    next.sourceNodeParentReferenceVertices.resize(n);
    if(original.remeshedOriginalNodeCount) {
        if(original.sourceNodeParentTetrahedronIndices.size()!=old||
           original.sourceNodeParentBarycentric.size()!=old||
           original.sourceNodeParentReferenceVertices.size()!=old)
            return fail("prior skin/fat remesh lost its retained parent ancestry");
        std::copy(original.sourceNodeParentTetrahedronIndices.begin(),
                  original.sourceNodeParentTetrahedronIndices.end(),
                  next.sourceNodeParentTetrahedronIndices.begin());
        std::copy(original.sourceNodeParentBarycentric.begin(),
                  original.sourceNodeParentBarycentric.end(),
                  next.sourceNodeParentBarycentric.begin());
        std::copy(original.sourceNodeParentReferenceVertices.begin(),
                  original.sourceNodeParentReferenceVertices.end(),
                  next.sourceNodeParentReferenceVertices.begin());
    }
    if(!next.sourceNodeEmbeddingFaceIndices.empty()) {
        if(next.sourceNodeEmbeddingFaceIndices.size()!=old||
           next.sourceNodeEmbeddingBarycentric.size()!=old)
            return fail("prior skin/fat projected source ancestry is incomplete");
        next.sourceNodeEmbeddingFaceIndices.resize(n,NM_INVALID_INDEX);
        next.sourceNodeEmbeddingBarycentric.resize(n,{});
    }
    for(std::size_t i=0;i<old;++i) {
        if(before.femNodes[i]!=after.femNodes[i]||
           (!before.femReferenceNodes.empty()&&
             (after.femReferenceNodes.empty()||
              before.femReferenceNodes[i]!=after.femReferenceNodes[i]))||
           original.sourceNodeVertexIndices[i]!=next.sourceNodeVertexIndices[i]||
           original.sourceNodeLayerIndices[i]!=next.sourceNodeLayerIndices[i]||
           original.sourceNodeDonorBodyIndices[i]!=next.sourceNodeDonorBodyIndices[i])
            return fail("local skin/fat splice changed a retained node or its anatomy");
    }
    for(std::size_t i=old;i<n;++i) {
        const auto parent=splice.nodeParentTetRows[i];
        if(parent>=before.tetrahedra.size())
            return fail("new skin/fat node has no valid original parent tet");
        next.sourceNodeParentTetrahedronIndices[i]=parent;
        next.sourceNodeParentBarycentric[i]=splice.nodeParentBarycentric[i];
        for(std::size_t k=0;k<4u;++k) {
            const auto v=before.tetrahedra[parent].nodes[k];
            if(v>=old)return fail("skin/fat node reference parent escapes original object");
            next.sourceNodeParentReferenceVertices[i][k]=before.femReferenceNodes.empty()?
                before.femNodes[v]:before.femReferenceNodes[v];
        }
    }
    next.registeredVertexCount=static_cast<std::size_t>(std::count(
        next.sourceNodeLayerIndices.begin(),next.sourceNodeLayerIndices.end(),0u));
    next.sourceTetrahedronCount=nt;
    next.tetrahedronRegionLabels=splice.tetDonorBodyLabels;
    next.tetrahedronBasisRegionLabels=splice.tetBasisRegionLabels;
    next.tetrahedronSourceFaceIndices=splice.tetSourceFaceIdentities;
    next.tetrahedronSourceDonorBodyIndices=splice.tetDonorBodyLabels;
    next.tetrahedronParentTetrahedronIndices=splice.tetParentTetRows;
    next.tetrahedronOriginalSourceFaceIndices.resize(nt);
    next.tetrahedronSourceParentBodyIndices.resize(nt);
    std::map<std::uint32_t,std::uint32_t> basisDonors;
    for(std::size_t i=0;i<nt;++i) {
        const auto parent=splice.tetParentTetRows[i],donor=splice.tetDonorBodyLabels[i];
        const auto ancestry=next.donorBodyAncestry.find(donor);
        if(parent>=before.tetrahedra.size()||
           parent>=original.tetrahedronSourceFaceIndices.size()||
           parent>=original.tetrahedronSourceDonorBodyIndices.size()||
           parent>=original.tetrahedronSourceParentBodyIndices.size()||
           donor!=original.tetrahedronSourceDonorBodyIndices[parent]||
           splice.tetMaterialIndices[i]!=after.femMaterialIndices[i]||
           before.femMaterialIndices[parent]!=after.femMaterialIndices[i]||
           ancestry==next.donorBodyAncestry.end()||ancestry->second.empty()||
           ancestry->second.front()!=donor)
            return fail("remeshed skin/fat tet lost its parent material/donor ancestry");
        next.tetrahedronOriginalSourceFaceIndices[i]=
            original.tetrahedronOriginalSourceFaceIndices.empty()?
                original.tetrahedronSourceFaceIndices[parent]:
                original.tetrahedronOriginalSourceFaceIndices[parent];
        next.tetrahedronSourceParentBodyIndices[i]=original.tetrahedronSourceParentBodyIndices[parent];
        const auto [it,inserted]=basisDonors.emplace(splice.tetBasisRegionLabels[i],donor);
        if(!inserted&&it->second!=donor)
            return fail("one remeshed skin/fat basis region aliases distinct donors");
    }
    if(!validateRegisteredSkinFatRemeshNodeAncestry(after,next,error))return false;
    output=std::move(next);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
