#pragma once
#include "NumiHumanParentTetRefinement.hpp"

namespace metalrobo::human::coarse_tissue {

// App-side, opt-in adapter for the existing appendRegisteredSkinFatFEMFromLayers
// owner. It refines that exact ObjectSource and updates the matching surface,
// layer, thickness, donor, and source-C ancestry views transactionally.
inline bool applyParentTetResolutionRefinement(
        numi::matter::WorldSource& world,
        std::size_t objectIndex,
        RegisteredSurface& surface,
        PreparedOrganicFEM& skinReport,
        PreparedOrganicFEM& fatReport,
        std::vector<double>& skinThicknessM,
        std::vector<double>& fatThicknessM,
        PairedBoundaryLayer& skinLayer,
        PairedBoundaryLayer& fatLayer,
        std::span<const std::array<std::uint32_t,2>> selectedEdges,
        std::span<const Face> detailedSourceFaces,
        std::span<const std::uint32_t> detailedSourceFaceIdentities,
        std::span<const V3> detailedSourceWorldPoints,
        std::span<const V3> detailedSourceWorldNormals,
        std::span<const std::uint32_t> detailedSourceDonorBodyByVertex,
        ParentTetRefinementResult& result,
        std::string& error) {
    error.clear();
    const auto fail=[&](const char* why){error=why;return false;};
    const std::size_t n=surface.worldPoints.size();
    if(objectIndex>=world.objects.size()||n==0u||
       skinThicknessM.size()!=n||fatThicknessM.size()!=n||
       skinLayer.outerWorld.size()!=n||skinLayer.innerWorld.size()!=n||
       fatLayer.outerWorld.size()!=n||fatLayer.innerWorld.size()!=n)
        return fail("app parent-tet adapter requires one existing appended paired-layer object and aligned boundary arrays");
    if(surface.donorLocalPoints.size()!=n||surface.donorBodyIndices.size()!=n||
       surface.worldToDonorLocalLinear.size()!=n||surface.outwardWorldNormals.size()!=n||
       surface.sourceVertexIndices.size()!=n||surface.sourceFaceIndices.size()!=surface.faces.size())
        return fail("app parent-tet adapter requires complete surface donor/source ancestry");
    const auto& originalObject=world.objects[objectIndex];
    if(originalObject.femNodes.size()!=3u*n||skinReport.donorMassNodes.empty()||
       skinReport.firstGlobalFEMNode!=fatReport.firstGlobalFEMNode||
       skinReport.firstGlobalFEMNode!=skinReport.donorMassNodes.front().nodeIndex)
        return fail("app parent-tet adapter received a noncanonical contiguous O/I/D object");
    const auto oldSourceVertices=surface.sourceVertexIndices;
    const auto oldSourceFaces=surface.sourceFaceIndices;
    const auto oldSourceEmbeddingFaces=surface.sourceEmbeddingFaceIndices;
    const auto oldSourceEmbeddingBary=surface.sourceEmbeddingBarycentric;
    const auto oldSourceEmbeddingResidual=surface.sourceEmbeddingResidualWorldM;
    const auto oldDonors=surface.donorBodyIndices;
    const auto oldThicknessSkin=skinThicknessM,oldThicknessFat=fatThicknessM;
    numi::matter::ObjectSource refinedObject;
    PreparedOrganicFEM refinedSkin,refinedFat;
    ParentTetRefinementResult nextResult;
    if(!refinePreparedPairedSkinFatParentTetEdges(
            originalObject,surface,skinReport,fatReport,selectedEdges,
            detailedSourceFaces,detailedSourceFaceIdentities,detailedSourceWorldPoints,
            detailedSourceWorldNormals,detailedSourceDonorBodyByVertex,{}, {},
            refinedObject,refinedSkin,refinedFat,nextResult,error))
        return false;
    const std::size_t R=nextResult.refinedSurfaceVertexCount;
    if(R!=n+selectedEdges.size()||refinedObject.femNodes.size()!=3u*R||
       nextResult.refinedSourceVertexIds.size()!=R||
       nextResult.refinedSourceEmbeddingFaceIds.size()!=R||
       nextResult.refinedSourceEmbeddingFaceIdentities.size()!=R||
       nextResult.refinedSourceEmbeddingBarycentric.size()!=R||
       nextResult.refinedSourceEmbeddingResidualWorldM.size()!=R||
       nextResult.oldGlobalFEMNodeToNewGlobalFEMNode.size()!=3u*n||
       nextResult.midpointGlobalFEMNodeIndices.size()!=3u*selectedEdges.size())
        return fail("app parent-tet adapter result has incomplete node/C-map remaps");

    RegisteredSurface refinedSurface=surface;
    refinedSurface.worldPoints.resize(R);refinedSurface.donorLocalPoints.resize(R);
    refinedSurface.outwardWorldNormals.resize(R);refinedSurface.donorBodyIndices.resize(R);
    refinedSurface.worldToDonorLocalLinear.resize(R);refinedSurface.sourceEmbeddingFaceIndices.resize(R);
    refinedSurface.sourceEmbeddingBarycentric.resize(R);refinedSurface.sourceEmbeddingResidualWorldM.resize(R);
    refinedSurface.sourceVertexIndices=nextResult.refinedSourceVertexIds;
    refinedSurface.sourceFaceIndices=nextResult.refinedOuterFaceSourceIdentity;
    refinedSurface.faces=nextResult.refinedOuterSurfaceFaces;
    std::vector<V3> localByGlobal(3u*R);std::vector<std::uint8_t> localPresent(3u*R,0u);
    for(const auto& row:refinedSkin.donorMassNodes){
        if(row.nodeIndex<refinedSkin.firstGlobalFEMNode)return fail("app parent-tet adapter donor row precedes the FEM object");
        const auto local=std::size_t(row.nodeIndex-refinedSkin.firstGlobalFEMNode);
        if(local>=localByGlobal.size()||localPresent[local])return fail("app parent-tet adapter donor row is duplicate or out of range");
        localByGlobal[local]=row.localPosition;localPresent[local]=1u;
    }
    if(std::find(localPresent.begin(),localPresent.end(),std::uint8_t(0))!=localPresent.end())
        return fail("app parent-tet adapter donor rows do not cover all O/I/D nodes");
    auto normalized=[](V3 p,V3& out){const double l=norm(p);if(!std::isfinite(l)||l<=1e-12)return false;out=mul(p,1.0/l);return true;};
    for(std::size_t row=0;row<R;++row){
        refinedSurface.worldPoints[row]=refinedObject.femNodes[row];
        refinedSurface.donorLocalPoints[row]=localByGlobal[row];
        refinedSurface.outwardWorldNormals[row]=surface.outwardWorldNormals[row<n?row:0u];
        refinedSurface.donorBodyIndices[row]=surface.donorBodyIndices[row<n?row:0u];
        refinedSurface.sourceEmbeddingFaceIndices[row]=nextResult.refinedSourceEmbeddingFaceIds[row];
        refinedSurface.sourceEmbeddingBarycentric[row]=nextResult.refinedSourceEmbeddingBarycentric[row];
        refinedSurface.sourceEmbeddingResidualWorldM[row]=nextResult.refinedSourceEmbeddingResidualWorldM[row];
        if(row<n){
            if(refinedSurface.sourceVertexIndices[row]!=oldSourceVertices[row]||
               refinedSurface.donorBodyIndices[row]!=oldDonors[row]||
               refinedSurface.sourceEmbeddingFaceIndices[row]!=(oldSourceEmbeddingFaces.empty()?NM_INVALID_INDEX:oldSourceEmbeddingFaces[row])||
               (!oldSourceEmbeddingBary.empty()&&refinedSurface.sourceEmbeddingBarycentric[row]!=oldSourceEmbeddingBary[row])||
               (!oldSourceEmbeddingResidual.empty()&&refinedSurface.sourceEmbeddingResidualWorldM[row]!=oldSourceEmbeddingResidual[row]))
                return fail("app parent-tet adapter changed an existing source identity or C-map row");
            continue;
        }
        const std::size_t ei=row-n;
        auto edge=selectedEdges[ei];if(edge[0]>edge[1])std::swap(edge[0],edge[1]);
        if(surface.donorBodyIndices[edge[0]]!=surface.donorBodyIndices[edge[1]])
            return fail("app parent-tet adapter selected edge crosses donor owners");
        const auto donor=surface.donorBodyIndices[edge[0]];
        if(surface.worldToDonorLocalLinear[edge[0]]!=surface.worldToDonorLocalLinear[edge[1]])
            return fail("same-donor midpoint endpoints have inconsistent registered frames");
        refinedSurface.donorBodyIndices[row]=donor;
        refinedSurface.worldToDonorLocalLinear[row]=surface.worldToDonorLocalLinear[edge[0]];
        const auto face=nextResult.refinedSourceEmbeddingFaceIds[row];
        if(face>=detailedSourceFaces.size()||
           detailedSourceFaceIdentities[face]!=nextResult.refinedSourceEmbeddingFaceIdentities[row])
            return fail("app parent-tet adapter C row cannot resolve its stable source-face identity");
        const auto tri=detailedSourceFaces[face];const auto beta=nextResult.refinedSourceEmbeddingBarycentric[row];
        V3 normal{};for(unsigned c=0;c<3u;++c)normal=add(normal,mul(detailedSourceWorldNormals[tri[c]],beta[c]));
        if(!normalized(normal,refinedSurface.outwardWorldNormals[row]))
            return fail("app parent-tet adapter C source normal is degenerate");
    }
    if(refinedSurface.sourceFaceIndices.size()!=refinedSurface.faces.size())
        return fail("app parent-tet adapter child surface lost parent source-face identities");

    auto refinedSkinThickness=refinedSkin.layerThicknessBySurfaceVertexM;
    auto refinedFatThickness=refinedFat.layerThicknessBySurfaceVertexM;
    if(refinedSkinThickness.size()!=R||refinedFatThickness.size()!=R)
        return fail("app parent-tet adapter thickness ancestry does not cover refined surface");
    PairedBoundaryLayer nextSkinLayer=skinLayer,nextFatLayer=fatLayer;
    nextSkinLayer.outerWorld.assign(refinedObject.femNodes.begin(),refinedObject.femNodes.begin()+R);
    nextSkinLayer.innerWorld.assign(refinedObject.femNodes.begin()+R,refinedObject.femNodes.begin()+2u*R);
    nextSkinLayer.outerLocal.assign(localByGlobal.begin(),localByGlobal.begin()+R);
    nextSkinLayer.innerLocal.assign(localByGlobal.begin()+R,localByGlobal.begin()+2u*R);
    nextSkinLayer.thicknessByVertexM=refinedSkinThickness;
    nextSkinLayer.minThicknessM=*std::min_element(refinedSkinThickness.begin(),refinedSkinThickness.end());
    nextSkinLayer.maxThicknessM=*std::max_element(refinedSkinThickness.begin(),refinedSkinThickness.end());
    nextFatLayer.outerWorld=nextSkinLayer.innerWorld;nextFatLayer.outerLocal=nextSkinLayer.innerLocal;
    nextFatLayer.innerWorld.assign(refinedObject.femNodes.begin()+2u*R,refinedObject.femNodes.end());
    nextFatLayer.innerLocal.assign(localByGlobal.begin()+2u*R,localByGlobal.end());
    nextFatLayer.thicknessByVertexM=refinedFatThickness;
    nextFatLayer.minThicknessM=*std::min_element(refinedFatThickness.begin(),refinedFatThickness.end());
    nextFatLayer.maxThicknessM=*std::max_element(refinedFatThickness.begin(),refinedFatThickness.end());

    // Preserve all old C/direct rows bit-for-bit and validate the explicit old
    // global-to-new map before replacing the app-owned objects.
    for(std::uint32_t old=0;old<3u*n;++old){
        const auto mapped=nextResult.oldNodeToNew[old];
        if(nextResult.oldGlobalFEMNodeToNewGlobalFEMNode[old]!=
           refinedSkin.firstGlobalFEMNode+mapped||
           refinedObject.femNodes[mapped]!=originalObject.femNodes[old])
            return fail("app parent-tet adapter old global row map or old coordinate changed");
    }
    world.objects[objectIndex]=std::move(refinedObject);
    surface=std::move(refinedSurface);
    skinReport=std::move(refinedSkin);fatReport=std::move(refinedFat);
    skinThicknessM=std::move(refinedSkinThickness);fatThicknessM=std::move(refinedFatThickness);
    skinLayer=std::move(nextSkinLayer);fatLayer=std::move(nextFatLayer);
    result=std::move(nextResult);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue

