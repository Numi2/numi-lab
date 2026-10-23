#pragma once
#include "NumiHumanCoarseTissuePreparation.hpp"

namespace metalrobo::human::coarse_tissue {

struct ParentTetBoundaryFace {
    Face nodes{};
    std::uint32_t parentSurfaceFace = NM_INVALID_INDEX;
    std::uint32_t sourceFaceIdentity = NM_INVALID_INDEX;
};

// Closest point on one finite triangle. Returns barycentrics in that triangle;
// unlike a plane projection, edge and vertex regions remain bounded.
inline bool parentTetClosestPointOnTriangleBarycentric(
        const V3& p, const V3& a, const V3& b, const V3& c,
        std::array<double,3>& bary, V3& closest) {
    const V3 ab=sub(b,a), ac=sub(c,a), ap=sub(p,a);
    if(norm(cross(ab,ac))<=1.0e-20)return false;
    const double d1=dot(ab,ap), d2=dot(ac,ap);
    if(d1<=0.0&&d2<=0.0){bary={1.0,0.0,0.0};closest=a;return true;}
    const V3 bp=sub(p,b);const double d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0.0&&d4<=d3){bary={0.0,1.0,0.0};closest=b;return true;}
    const double vc=d1*d4-d3*d2;
    if(vc<=0.0&&d1>=0.0&&d3<=0.0){const double v=d1/(d1-d3);bary={1.0-v,v,0.0};closest=add(a,mul(ab,v));return true;}
    const V3 cp=sub(p,c);const double d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0.0&&d5<=d6){bary={0.0,0.0,1.0};closest=c;return true;}
    const double vb=d5*d2-d1*d6;
    if(vb<=0.0&&d2>=0.0&&d6<=0.0){const double w=d2/(d2-d6);bary={1.0-w,0.0,w};closest=add(a,mul(ac,w));return true;}
    const double va=d3*d6-d5*d4;
    if(va<=0.0&&(d4-d3)>=0.0&&(d5-d6)>=0.0){const double w=(d4-d3)/((d4-d3)+(d5-d6));bary={0.0,1.0-w,w};closest=add(b,mul(sub(c,b),w));return true;}
    const double denom=va+vb+vc;
    if(!std::isfinite(denom)||std::abs(denom)<=1.0e-30)return false;
    const double inv=1.0/denom,v=vb*inv,w=vc*inv;
    bary={1.0-v-w,v,w};closest=add(a,add(mul(ab,v),mul(ac,w)));
    return std::all_of(bary.begin(),bary.end(),[](double x){return std::isfinite(x)&&x>=-1.0e-12&&x<=1.0+1.0e-12;});
}

struct ParentTetRefinementResult {
    std::uint32_t originalSurfaceVertexCount = 0;
    std::uint32_t refinedSurfaceVertexCount = 0;
    // Original combined-object row -> refined O/I/D row. The returned object
    // keeps the required contiguous O, I, D blocks.
    std::vector<std::uint32_t> oldNodeToNew;
    // Explicit global FEM-node remaps for callers whose ObjectSource begins at
    // a nonzero world node index. Midpoint rows are layer-major O/I/D.
    std::vector<std::uint32_t> oldGlobalFEMNodeToNewGlobalFEMNode;
    std::vector<std::uint32_t> midpointGlobalFEMNodeIndices;
    // New midpoint rows are ordered exactly as the input edge list. Each
    // midpoint has a parent triangle in the original RegisteredSurface frame;
    // this is conservative coarse-surface ancestry, not a detailed-skin C map.
    std::vector<std::uint32_t> midpointParentSurfaceFace;
    std::vector<std::uint32_t> midpointParentSourceFace;
    std::vector<std::array<double,3>> midpointParentBarycentric;
    // Optional detailed C-map rows for the new vertices. When detailed source
    // topology is not supplied, these remain explicitly invalid and callers
    // must rebuild through the existing source-transfer owner before use.
    bool detailedSourceEmbeddingComplete = false;
    std::vector<std::uint32_t> refinedSourceVertexIds;
    // Local row index and stable source-face identity are deliberately separate.
    std::vector<std::uint32_t> refinedSourceEmbeddingFaceIds;
    std::vector<std::uint32_t> refinedSourceEmbeddingFaceIdentities;
    std::vector<std::uint32_t> refinedSourceEmbeddingNeighborhoodRing;
    std::vector<float> refinedSourceEmbeddingNormalDot;
    std::vector<float> refinedSourceEmbeddingProjectionDistanceM;
    std::vector<std::array<double,3>> refinedSourceEmbeddingBarycentric;
    std::vector<V3> refinedSourceEmbeddingResidualWorldM;
    std::vector<Face> refinedOuterSurfaceFaces;
    std::vector<std::uint32_t> refinedOuterFaceParentSurfaceFace;
    std::vector<std::uint32_t> refinedOuterFaceSourceIdentity;
    std::vector<std::uint32_t> childParentTet;
    std::vector<std::uint32_t> childParentSurfaceFace;
    std::vector<std::uint32_t> childSourceFaceIdentity;
    std::vector<std::uint32_t> childMaterialIndex;
    std::vector<ParentTetBoundaryFace> skinBoundaryFaces;
    std::vector<ParentTetBoundaryFace> fatBoundaryFaces;
    std::vector<ParentTetBoundaryFace> outerSkinFaces;
    std::vector<ParentTetBoundaryFace> skinFatInterfaceSkinFaces;
    std::vector<ParentTetBoundaryFace> skinFatInterfaceFatFaces;
    std::vector<ParentTetBoundaryFace> deepFatFaces;
    std::vector<ParentTetBoundaryFace> skinSidewallFaces;
    std::vector<ParentTetBoundaryFace> fatSidewallFaces;
    double originalSkinVolumeM3 = 0.0, refinedSkinVolumeM3 = 0.0;
    double originalFatVolumeM3 = 0.0, refinedFatVolumeM3 = 0.0;
    double minimumEmittedFloat32SignedVolumeM3 = std::numeric_limits<double>::infinity();
    double minimumEmittedFloat32NormalizedDeterminant = std::numeric_limits<double>::infinity();
};

// Conforming edge bisection of already accepted skin/fat parent tetrahedra.
// Each edge is split in all three O/I/D layers, and every incident parent
// tetrahedron is replaced by two children with the same orientation. The helper
// changes resolution only: no node is projected and no rest geometry is moved.
// It supports the current stateless paired-shell ObjectSource shape and fails
// closed if extra topology-bearing runtime metadata would need a new remap.
inline bool refinePreparedPairedSkinFatParentTetEdges(
        const numi::matter::ObjectSource& input,
        const RegisteredSurface& parentSurface,
        const PreparedOrganicFEM& preparedSkinInput,
        const PreparedOrganicFEM& preparedFatInput,
        std::span<const std::array<std::uint32_t,2>> selectedEdges,
        std::span<const Face> detailedSourceFaces,
        std::span<const std::uint32_t> detailedSourceFaceIdentities,
        std::span<const V3> detailedSourceWorldPoints,
        std::span<const V3> detailedSourceWorldNormals,
        std::span<const std::uint32_t> detailedSourceDonorBodyByVertex,
        std::span<const std::uint32_t> midpointDonorBodyByLayer,
        std::span<const V3> midpointDonorLocalPointByLayer,
        numi::matter::ObjectSource& output,
        PreparedOrganicFEM& preparedSkinOutput,
        PreparedOrganicFEM& preparedFatOutput,
        ParentTetRefinementResult& result,
        std::string& error) {
    using Tet = std::array<std::uint32_t,4>;
    auto fail=[&](const char* message){error=message;return false;};
    result={}; error.clear();
    const std::size_t n=parentSurface.worldPoints.size();
    if(n<4u||parentSurface.faces.size()<4u||selectedEdges.empty()||n>std::numeric_limits<std::uint32_t>::max()/3u)
        return fail("parent-tet refinement requires a nonempty edge patch on a valid registered surface");
    if(input.representation!=numi::matter::Representation::fem||input.mixedFEM||
       input.femNodes.size()!=3u*n||input.tetrahedra.empty()||
       input.femMaterialIndices.size()!=input.tetrahedra.size())
        return fail("parent-tet refinement only accepts the current explicit paired O/I/D FEM object");
    if(parentSurface.sourceVertexIndices.size()!=n||parentSurface.sourceFaceIndices.size()!=parentSurface.faces.size())
        return fail("parent-tet refinement requires complete existing surface vertex and face identities");
    if(detailedSourceFaces.empty()!=detailedSourceWorldPoints.empty()||
       detailedSourceFaces.empty()!=detailedSourceFaceIdentities.empty()||
       detailedSourceFaces.empty()!=detailedSourceWorldNormals.empty()||
       detailedSourceFaces.empty()!=detailedSourceDonorBodyByVertex.empty())
        return fail("detailed C-map inputs require aligned faces, stable identities, registered points, normals, and donor owners");
    if(!detailedSourceFaces.empty()&&
       (parentSurface.sourceEmbeddingFaceIndices.size()!=n||parentSurface.sourceEmbeddingBarycentric.size()!=n||
        parentSurface.sourceEmbeddingResidualWorldM.size()!=n||
        parentSurface.donorBodyIndices.size()!=n||
        detailedSourceWorldNormals.size()!=detailedSourceWorldPoints.size()||
        detailedSourceDonorBodyByVertex.size()!=detailedSourceWorldPoints.size()))
        return fail("detailed C-map rebuilding requires complete input ancestry and source registration rows");
    std::map<std::uint32_t,std::uint32_t> detailedFaceRowByIdentity;
    std::vector<std::vector<std::uint32_t>> detailedFaceAdjacency;
    if(!detailedSourceFaces.empty()){
        detailedFaceAdjacency.resize(detailedSourceFaces.size());
        std::map<std::array<std::uint32_t,2>,std::vector<std::uint32_t>> detailEdgeFaces;
        for(std::uint32_t fi=0;fi<detailedSourceFaces.size();++fi){
            if(!detailedFaceRowByIdentity.emplace(detailedSourceFaceIdentities[fi],fi).second)
                return fail("detailed source face identities are not unique");
            const auto f=detailedSourceFaces[fi];
            if(f[0]>=detailedSourceWorldPoints.size()||f[1]>=detailedSourceWorldPoints.size()||f[2]>=detailedSourceWorldPoints.size()||
               f[0]==f[1]||f[1]==f[2]||f[2]==f[0])
                return fail("detailed source face has invalid or duplicate vertex indices");
            for(auto v:f){
                const auto& p=detailedSourceWorldPoints[v];const auto& nn=detailedSourceWorldNormals[v];
                if(!std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);})||
                   !std::all_of(nn.begin(),nn.end(),[](double x){return std::isfinite(x);})||
                   norm(nn)<=1.0e-12||detailedSourceDonorBodyByVertex[v]==NM_INVALID_INDEX)
                    return fail("detailed source registration has a nonfinite point/normal or invalid donor owner");
            }
            for(unsigned k=0;k<3u;++k){auto a=f[k],b=f[(k+1u)%3u];if(a>b)std::swap(a,b);detailEdgeFaces[{a,b}].push_back(fi);}
        }
        for(const auto& [edge,faces]:detailEdgeFaces){(void)edge;for(auto a:faces)for(auto b:faces)if(a!=b)detailedFaceAdjacency[a].push_back(b);}
        for(auto& row:detailedFaceAdjacency){std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());}
    }
    if(preparedSkinInput.materialIndex==NM_INVALID_INDEX||preparedFatInput.materialIndex==NM_INVALID_INDEX||
       preparedSkinInput.materialIndex==preparedFatInput.materialIndex||
       !std::isfinite(preparedSkinInput.densityKgM3)||preparedSkinInput.densityKgM3<=0.0||
       !std::isfinite(preparedFatInput.densityKgM3)||preparedFatInput.densityKgM3<=0.0||
       !std::isfinite(preparedSkinInput.massDebitKg)||preparedSkinInput.massDebitKg<=0.0||
       !std::isfinite(preparedFatInput.massDebitKg)||preparedFatInput.massDebitKg<=0.0||
       !std::isfinite(preparedSkinInput.volumeM3)||preparedSkinInput.volumeM3<=0.0||
       !std::isfinite(preparedFatInput.volumeM3)||preparedFatInput.volumeM3<=0.0)
        return fail("parent-tet refinement requires exact existing skin and fat material debit reports");
    // Current owner inputs are stateless paired layers. Do not preserve a
    // topology-indexed optional field by accident if this helper does not remap it.
    if(!input.femHumanAttachments.empty()||!input.femCohesiveInterfaces.empty()||
       !input.femCohesiveMortarPatches.empty()||!input.fieldBoundaries.empty()||
       input.multiphysics.enabled||input.mutationPolicy.enabled||!input.mutationCommands.empty()||
       !input.femDisabledContactSurfaceFaces.empty())
        return fail("paired parent-tet refinement does not support attached, cohesive, field, mutation, or disabled-face state");
    if(!input.femReferenceNodes.empty()&&input.femReferenceNodes.size()!=input.femNodes.size())
        return fail("paired parent-tet refinement reference-node rows are incomplete");
    if(!input.femPreconditionerRegionIds.empty()&&input.femPreconditionerRegionIds.size()!=input.tetrahedra.size())
        return fail("paired parent-tet refinement preconditioner rows are incomplete");
    if(!input.femMaterialFrameRotations.empty()&&input.femMaterialFrameRotations.size()!=input.tetrahedra.size())
        return fail("paired parent-tet refinement material-frame rows are incomplete");
    if(preparedSkinInput.nodeSourceVertex.size()!=2u*n||preparedFatInput.nodeSourceVertex.size()!=2u*n||
       preparedSkinInput.tetSourceFace.size()+preparedFatInput.tetSourceFace.size()!=input.tetrahedra.size())
        return fail("paired parent-tet refinement source ancestry does not cover the existing O/I/D object");
    if((!midpointDonorBodyByLayer.empty()&&midpointDonorBodyByLayer.size()!=3u*selectedEdges.size())||
       (!midpointDonorLocalPointByLayer.empty()&&midpointDonorLocalPointByLayer.size()!=3u*selectedEdges.size())||
       (midpointDonorBodyByLayer.empty()!=midpointDonorLocalPointByLayer.empty()))
        return fail("midpoint donor owner/local-point rows must be supplied as one complete O/I/D table");
    if(!preparedSkinInput.layerThicknessBySurfaceVertexM.empty()&&preparedSkinInput.layerThicknessBySurfaceVertexM.size()!=n)
        return fail("skin thickness ancestry has the wrong parent vertex count");
    if(!preparedFatInput.layerThicknessBySurfaceVertexM.empty()&&preparedFatInput.layerThicknessBySurfaceVertexM.size()!=n)
        return fail("fat thickness ancestry has the wrong parent vertex count");

    const auto ns=preparedSkinInput.tetSourceFace.size();
    const auto nf=preparedFatInput.tetSourceFace.size();
    if(ns%3u!=0u||nf%3u!=0u||ns/3u!=parentSurface.faces.size()||nf/3u!=parentSurface.faces.size())
        return fail("current parent tetrahedra are not the three-tets-per-face paired shell recipe");
    if(input.femContactNodes.empty())return fail("paired parent-tet refinement requires the existing explicit contact-node list");

    std::map<std::array<std::uint32_t,2>,std::uint32_t> edgeParentFace;
    std::map<std::array<std::uint32_t,2>,std::vector<std::uint32_t>> edgeIncidentFaces;
    std::map<std::array<std::uint32_t,2>,std::uint32_t> edgeFaceUse;
    for(std::uint32_t fi=0;fi<parentSurface.faces.size();++fi){
        const auto f=parentSurface.faces[fi];
        if(f[0]>=n||f[1]>=n||f[2]>=n||f[0]==f[1]||f[1]==f[2]||f[2]==f[0])
            return fail("registered parent surface contains an invalid face");
        for(unsigned k=0;k<3u;++k){auto a=f[k],b=f[(k+1u)%3u];if(a>b)std::swap(a,b);edgeParentFace.try_emplace({a,b},fi);edgeIncidentFaces[{a,b}].push_back(fi);++edgeFaceUse[{a,b}];}
    }
    std::set<std::array<std::uint32_t,2>> uniqueEdges;
    for(const auto& raw:selectedEdges){
        auto e=raw;if(e[0]>e[1])std::swap(e[0],e[1]);
        if(e[0]>=n||e[1]>=n||e[0]==e[1]||!uniqueEdges.insert(e).second||!edgeParentFace.contains(e))
            return fail("selected midpoint edge is duplicate, degenerate, out of range, or absent from the parent surface");
    }
    const std::size_t m=selectedEdges.size(),R=n+m;
    if(R>std::numeric_limits<std::uint32_t>::max()/3u)return fail("refined O/I/D node rows exceed the existing index format");
    result.originalSurfaceVertexCount=static_cast<std::uint32_t>(n);
    result.refinedSurfaceVertexCount=static_cast<std::uint32_t>(R);

    // Determine all original row identities before expanding into three new
    // contiguous layer blocks. Input rows must already be exact O, I, D blocks.
    std::vector<std::uint32_t> oldToNew(3u*n);
    for(std::uint32_t layer=0;layer<3u;++layer)for(std::uint32_t v=0;v<n;++v)
        oldToNew[layer*n+v]=layer*static_cast<std::uint32_t>(R)+v;
    auto newNodes=std::vector<V3>(3u*R);
    auto newReference=input.femReferenceNodes.empty()?std::vector<V3>{}:std::vector<V3>(3u*R);
    for(std::uint32_t layer=0;layer<3u;++layer){
        for(std::uint32_t v=0;v<n;++v){newNodes[layer*R+v]=input.femNodes[layer*n+v];if(!newReference.empty())newReference[layer*R+v]=input.femReferenceNodes[layer*n+v];}
        for(std::uint32_t ei=0;ei<m;++ei){
            auto e=selectedEdges[ei];if(e[0]>e[1])std::swap(e[0],e[1]);
            const auto a=input.femNodes[layer*n+e[0]],b=input.femNodes[layer*n+e[1]];
            newNodes[layer*R+n+ei]=mul(add(a,b),0.5);
            if(!newReference.empty())newReference[layer*R+n+ei]=mul(add(input.femReferenceNodes[layer*n+e[0]],input.femReferenceNodes[layer*n+e[1]]),0.5);
        }
    }

    struct TetRecord { Tet nodes{}; std::uint32_t material=0, sourceFace=NM_INVALID_INDEX, parentSurfaceFace=NM_INVALID_INDEX, parentTet=NM_INVALID_INDEX, region=0; std::array<double,4> frame{}; };
    std::vector<TetRecord> tets; tets.reserve(input.tetrahedra.size()+selectedEdges.size()*24u);
    std::size_t skinSource=0,fatSource=0;double baseSkinTetVolume=0.0,baseFatTetVolume=0.0;
    for(std::uint32_t ti=0;ti<input.tetrahedra.size();++ti){
        const auto material=input.femMaterialIndices[ti];
        const bool isSkin=material==preparedSkinInput.materialIndex,isFat=material==preparedFatInput.materialIndex;
        if(isSkin==isFat)return fail("paired parent-tet refinement found a tet with an unowned or ambiguous material");
        const auto sourceFace=isSkin?preparedSkinInput.tetSourceFace[skinSource++]:preparedFatInput.tetSourceFace[fatSource++];
        const auto parentFace=static_cast<std::uint32_t>((isSkin?skinSource:fatSource)-1u)/3u;
        if(parentFace>=parentSurface.faces.size()||sourceFace!=parentSurface.sourceFaceIndices[parentFace])
            return fail("parent tet source-face identity is inconsistent with its exact paired surface parent");
        TetRecord record;record.material=material;record.sourceFace=sourceFace;record.parentSurfaceFace=parentFace;record.parentTet=ti;
        if(!input.femPreconditionerRegionIds.empty())record.region=input.femPreconditionerRegionIds[ti];
        if(!input.femMaterialFrameRotations.empty())record.frame=input.femMaterialFrameRotations[ti];
        for(unsigned k=0;k<4u;++k){
            const auto old=input.tetrahedra[ti].nodes[k];if(old>=3u*n)return fail("parent tetrahedron references an O/I/D row outside the paired object");
            if((isSkin&&old>=2u*n)||(isFat&&old<n))return fail("parent tetrahedron crosses the wrong material-layer block");
            record.nodes[k]=oldToNew[old];
        }
        const double baseVolume=tetVolume(newNodes[record.nodes[0]],newNodes[record.nodes[1]],newNodes[record.nodes[2]],newNodes[record.nodes[3]]);
        if(!std::isfinite(baseVolume)||baseVolume<=0.0)return fail("existing parent tetrahedron is nonpositive before refinement");
        if(isSkin)baseSkinTetVolume+=baseVolume;else baseFatTetVolume+=baseVolume;
        tets.push_back(record);
    }
    if(skinSource!=ns||fatSource!=nf)return fail("parent tetrahedron/material ownership counts differ from prepared reports");
    if(std::abs(baseSkinTetVolume-preparedSkinInput.volumeM3)>1.0e-6*preparedSkinInput.volumeM3||
       std::abs(baseFatTetVolume-preparedFatInput.volumeM3)>1.0e-6*preparedFatInput.volumeM3)
        return fail("existing parent tetrahedron volume does not match the prepared material report");

    // Sequence bisections on the already-existing tet complex. The same global
    // edge midpoint is shared by every incident tetrahedron and by all three
    // corresponding boundary layers.
    for(std::uint32_t ei=0;ei<m;++ei){
        auto e=selectedEdges[ei];if(e[0]>e[1])std::swap(e[0],e[1]);
        for(std::uint32_t layer=0;layer<3u;++layer){
            const auto a=layer*static_cast<std::uint32_t>(R)+e[0];
            const auto b=layer*static_cast<std::uint32_t>(R)+e[1];
            const auto mid=layer*static_cast<std::uint32_t>(R)+static_cast<std::uint32_t>(n)+ei;
            std::vector<TetRecord> next;next.reserve(tets.size()+8u);std::uint32_t affected=0;
            for(const auto& old:tets){
                int ia=-1,ib=-1;for(int k=0;k<4;++k){if(old.nodes[k]==a)ia=k;if(old.nodes[k]==b)ib=k;}
                if(ia<0||ib<0){next.push_back(old);continue;}
                ++affected;TetRecord left=old,right=old;left.nodes[ib]=mid;right.nodes[ia]=mid;
                next.push_back(left);next.push_back(right);
            }
            if(affected==0u)return fail("selected surface edge is absent from one of the O/I/D parent tet layers");
            tets.swap(next);
        }
    }

    // Build the output object in a local copy, so every failure leaves caller
    // state untouched. Existing direct rows retain their exact coordinates.
    numi::matter::ObjectSource candidate=input;
    candidate.femNodes=std::move(newNodes);
    if(!newReference.empty())candidate.femReferenceNodes=std::move(newReference);
    candidate.tetrahedra.clear();candidate.femMaterialIndices.clear();
    candidate.tetrahedra.reserve(tets.size());candidate.femMaterialIndices.reserve(tets.size());
    candidate.femPreconditionerRegionIds.clear();candidate.femMaterialFrameRotations.clear();
    if(!input.femPreconditionerRegionIds.empty())candidate.femPreconditionerRegionIds.reserve(tets.size());
    if(!input.femMaterialFrameRotations.empty())candidate.femMaterialFrameRotations.reserve(tets.size());
    for(const auto& t:tets){candidate.tetrahedra.push_back({t.nodes});candidate.femMaterialIndices.push_back(t.material);if(!input.femPreconditionerRegionIds.empty())candidate.femPreconditionerRegionIds.push_back(t.region);if(!input.femMaterialFrameRotations.empty())candidate.femMaterialFrameRotations.push_back(t.frame);}
    auto remapNodeList=[&](const std::vector<std::uint32_t>& src,std::vector<std::uint32_t>& dst)->bool{dst.clear();dst.reserve(src.size());for(auto id:src){if(id>=oldToNew.size())return false;dst.push_back(oldToNew[id]);}return true;};
    if(!remapNodeList(input.femFixedNodes,candidate.femFixedNodes))return fail("fixed-node list contains an invalid old row");
    std::set<std::uint32_t> oldContact(input.femContactNodes.begin(),input.femContactNodes.end());
    if(oldContact.size()!=input.femContactNodes.size()||*oldContact.rbegin()>=oldToNew.size())return fail("contact-node list is duplicate or out of range");
    if(!remapNodeList(input.femContactNodes,candidate.femContactNodes))return fail("contact-node list contains an invalid old row");
    for(std::uint32_t layer=0;layer<3u;++layer)for(std::uint32_t ei=0;ei<m;++ei){
        auto e=selectedEdges[ei];if(e[0]>e[1])std::swap(e[0],e[1]);
        if(oldContact.contains(layer*n+e[0])&&oldContact.contains(layer*n+e[1]))candidate.femContactNodes.push_back(layer*static_cast<std::uint32_t>(R)+static_cast<std::uint32_t>(n)+ei);
    }
    std::sort(candidate.femContactNodes.begin(),candidate.femContactNodes.end());candidate.femContactNodes.erase(std::unique(candidate.femContactNodes.begin(),candidate.femContactNodes.end()),candidate.femContactNodes.end());
    if(candidate.femCapacity.nodes!=0u)candidate.femCapacity.nodes=std::max(candidate.femCapacity.nodes,static_cast<std::uint32_t>(candidate.femNodes.size()));
    if(candidate.femCapacity.tetrahedra!=0u)candidate.femCapacity.tetrahedra=std::max(candidate.femCapacity.tetrahedra,static_cast<std::uint32_t>(candidate.tetrahedra.size()));

    // Report exact parent ancestry and material-preserving tet splits.
    result.oldNodeToNew=oldToNew;
    if(preparedSkinInput.firstGlobalFEMNode>std::numeric_limits<std::uint32_t>::max()-3u*static_cast<std::uint32_t>(R))
        return fail("refined global FEM-node remap overflows the existing index format");
    result.oldGlobalFEMNodeToNewGlobalFEMNode.resize(oldToNew.size());
    for(std::uint32_t old=0;old<oldToNew.size();++old)
        result.oldGlobalFEMNodeToNewGlobalFEMNode[old]=preparedSkinInput.firstGlobalFEMNode+oldToNew[old];
    result.midpointGlobalFEMNodeIndices.reserve(3u*m);
    for(std::uint32_t layer=0;layer<3u;++layer)for(std::uint32_t ei=0;ei<m;++ei)
        result.midpointGlobalFEMNodeIndices.push_back(preparedSkinInput.firstGlobalFEMNode+layer*static_cast<std::uint32_t>(R)+static_cast<std::uint32_t>(n)+ei);
    result.childParentTet.reserve(tets.size());result.childParentSurfaceFace.reserve(tets.size());result.childSourceFaceIdentity.reserve(tets.size());result.childMaterialIndex.reserve(tets.size());
    std::vector<std::uint32_t> skinChildFaces,fatChildFaces;
    for(const auto& t:tets){result.childParentTet.push_back(t.parentTet);result.childParentSurfaceFace.push_back(t.parentSurfaceFace);result.childSourceFaceIdentity.push_back(t.sourceFace);result.childMaterialIndex.push_back(t.material);if(t.material==preparedSkinInput.materialIndex)skinChildFaces.push_back(t.sourceFace);else fatChildFaces.push_back(t.sourceFace);}
    result.midpointParentSurfaceFace.reserve(m);result.midpointParentSourceFace.reserve(m);result.midpointParentBarycentric.reserve(m);
    for(const auto& raw:selectedEdges){auto e=raw;if(e[0]>e[1])std::swap(e[0],e[1]);const auto fi=edgeParentFace.at(e);const auto f=parentSurface.faces[fi];std::array<double,3> beta{};for(unsigned c=0;c<3u;++c)if(f[c]==e[0]||f[c]==e[1])beta[c]=0.5;else beta[c]=0.0;result.midpointParentSurfaceFace.push_back(fi);result.midpointParentSourceFace.push_back(parentSurface.sourceFaceIndices[fi]);result.midpointParentBarycentric.push_back(beta);}
    result.refinedSourceVertexIds.assign(R,NM_INVALID_INDEX);
    result.refinedSourceEmbeddingFaceIds.assign(R,NM_INVALID_INDEX);
    result.refinedSourceEmbeddingFaceIdentities.assign(R,NM_INVALID_INDEX);
    result.refinedSourceEmbeddingNeighborhoodRing.assign(R,NM_INVALID_INDEX);
    result.refinedSourceEmbeddingNormalDot.assign(R,std::numeric_limits<float>::quiet_NaN());
    result.refinedSourceEmbeddingProjectionDistanceM.assign(R,std::numeric_limits<float>::quiet_NaN());
    result.refinedSourceEmbeddingBarycentric.assign(R,{});
    result.refinedSourceEmbeddingResidualWorldM.assign(R,{});
    for(std::size_t v=0;v<n;++v){result.refinedSourceVertexIds[v]=parentSurface.sourceVertexIndices[v];
        if(!parentSurface.sourceEmbeddingFaceIndices.empty())result.refinedSourceEmbeddingFaceIds[v]=parentSurface.sourceEmbeddingFaceIndices[v];
        if(!parentSurface.sourceEmbeddingBarycentric.empty())result.refinedSourceEmbeddingBarycentric[v]=parentSurface.sourceEmbeddingBarycentric[v];
        if(!parentSurface.sourceEmbeddingResidualWorldM.empty())result.refinedSourceEmbeddingResidualWorldM[v]=parentSurface.sourceEmbeddingResidualWorldM[v];}

    // Signed volume, Float32 emission, and per-material volume are checked from
    // the exact output connectivity, not inferred from the edge recipe.
    std::vector<double> nodalMass(candidate.femNodes.size(),0.0);double skinVolume=0.0,fatVolume=0.0;
    for(std::size_t ti=0;ti<tets.size();++ti){const auto& tr=tets[ti];const auto& x=candidate.femNodes;const double vol=tetVolume(x[tr.nodes[0]],x[tr.nodes[1]],x[tr.nodes[2]],x[tr.nodes[3]]);if(!std::isfinite(vol)||vol<=0.0)return fail("parent-tet bisection produced a nonpositive or nonfinite double-precision child");
        std::array<V3,4> f32{};for(unsigned k=0;k<4u;++k)for(unsigned a=0;a<3u;++a)f32[k][a]=static_cast<double>(static_cast<float>(x[tr.nodes[k]][a]));const double emitted=tetVolume(f32[0],f32[1],f32[2],f32[3]);if(!std::isfinite(emitted)||emitted<=0.0)return fail("parent-tet bisection child is nonpositive after Float32 coordinate emission");result.minimumEmittedFloat32SignedVolumeM3=std::min(result.minimumEmittedFloat32SignedVolumeM3,emitted);
        double longest=0.0;for(unsigned a=0;a<4u;++a)for(unsigned b=a+1u;b<4u;++b)longest=std::max(longest,norm(sub(f32[a],f32[b])));
        const double normalized=6.0*emitted/(longest*longest*longest);
        if(!std::isfinite(normalized)||normalized<1.0e-5)return fail("parent-tet bisection child falls below the existing Float32 normalized-determinant admission floor");
        result.minimumEmittedFloat32NormalizedDeterminant=std::min(result.minimumEmittedFloat32NormalizedDeterminant,normalized);
        const bool isSkin=tr.material==preparedSkinInput.materialIndex;const double rho=isSkin?preparedSkinInput.densityKgM3:preparedFatInput.densityKgM3;if(isSkin)skinVolume+=vol;else fatVolume+=vol;for(auto node:tr.nodes)nodalMass[node]+=rho*vol/4.0;
    }
    result.originalSkinVolumeM3=baseSkinTetVolume;result.refinedSkinVolumeM3=skinVolume;result.originalFatVolumeM3=baseFatTetVolume;result.refinedFatVolumeM3=fatVolume;
    if(std::abs(skinVolume-baseSkinTetVolume)>2e-12*baseSkinTetVolume||std::abs(fatVolume-baseFatTetVolume)>2e-12*baseFatTetVolume)
        return fail("parent-tet bisection changed an exact input material volume beyond accumulated floating-point tolerance");
    if(std::abs(skinVolume*preparedSkinInput.densityKgM3-preparedSkinInput.massDebitKg)>1e-7*preparedSkinInput.massDebitKg||std::abs(fatVolume*preparedFatInput.densityKgM3-preparedFatInput.massDebitKg)>1e-7*preparedFatInput.massDebitKg)
        return fail("parent-tet bisection no longer closes the existing per-material mass debits");

    // Preserve the exact old donor ownership/local point for every old node;
    // an inserted midpoint is valid only when both endpoints retain one owner.
    struct Donor {std::uint32_t body=NM_INVALID_INDEX;V3 local{};};std::vector<Donor> oldDonor(3u*n);std::vector<bool> haveDonor(3u*n,false);
    for(const auto& dm:preparedSkinInput.donorMassNodes){if(dm.nodeIndex<preparedSkinInput.firstGlobalFEMNode)continue;const auto local=dm.nodeIndex-preparedSkinInput.firstGlobalFEMNode;if(local>=3u*n||haveDonor[local])return fail("existing paired-shell mass ancestry has duplicate or out-of-range rows");oldDonor[local]={dm.donorBody,dm.localPosition};haveDonor[local]=true;}
    if(std::find(haveDonor.begin(),haveDonor.end(),false)!=haveDonor.end())return fail("existing paired-shell donor-mass ancestry does not cover every O/I/D node");
    std::vector<Donor> newDonor(3u*R);
    for(std::uint32_t old=0;old<3u*n;++old)newDonor[oldToNew[old]]=oldDonor[old];
    for(std::uint32_t layer=0;layer<3u;++layer)for(std::uint32_t ei=0;ei<m;++ei){
        auto e=selectedEdges[ei];if(e[0]>e[1])std::swap(e[0],e[1]);const auto a=layer*n+e[0],b=layer*n+e[1];
        const std::size_t supplied=std::size_t(layer)*m+ei;
        if(!midpointDonorBodyByLayer.empty()){
            const auto body=midpointDonorBodyByLayer[supplied];const auto local=midpointDonorLocalPointByLayer[supplied];
            if(body==NM_INVALID_INDEX||!std::all_of(local.begin(),local.end(),[](double x){return std::isfinite(x);}))return fail("caller-supplied midpoint donor frame is invalid");
            newDonor[layer*R+n+ei]={body,local};
        }else{
            if(oldDonor[a].body==NM_INVALID_INDEX||oldDonor[a].body!=oldDonor[b].body)return fail("a new midpoint spans distinct donor bodies; provide the existing owner-frame resolver output");
            newDonor[layer*R+n+ei]={oldDonor[a].body,mul(add(oldDonor[a].local,oldDonor[b].local),0.5)};
        }
    }
    PreparedOrganicFEM skinReport=preparedSkinInput,fatReport=preparedFatInput;
    auto extendSurfaceReport=[&](PreparedOrganicFEM& report)->bool{
        if(report.nodeSourceVertex.size()!=2u*n)return false;std::vector<std::uint32_t> src(2u*R,NM_INVALID_INDEX);for(std::uint32_t v=0;v<n;++v){src[v]=report.nodeSourceVertex[v];src[R+v]=report.nodeSourceVertex[n+v];}report.nodeSourceVertex=std::move(src);
        if(!report.layerThicknessBySurfaceVertexM.empty()){auto old=report.layerThicknessBySurfaceVertexM;report.layerThicknessBySurfaceVertexM.resize(R);for(std::uint32_t ei=0;ei<m;++ei){auto e=selectedEdges[ei];report.layerThicknessBySurfaceVertexM[n+ei]=0.5*(old[e[0]]+old[e[1]]);}report.layerThicknessM=0.0;report.layerThicknessMinM=*std::min_element(report.layerThicknessBySurfaceVertexM.begin(),report.layerThicknessBySurfaceVertexM.end());report.layerThicknessMaxM=*std::max_element(report.layerThicknessBySurfaceVertexM.begin(),report.layerThicknessBySurfaceVertexM.end());}
        return true;};
    if(!extendSurfaceReport(skinReport)||!extendSurfaceReport(fatReport))return fail("paired-shell node/source-thickness report cannot be extended without dropping ancestry");
    skinReport.tetSourceFace.clear();fatReport.tetSourceFace.clear();skinReport.tetSourceFace.reserve(skinChildFaces.size());fatReport.tetSourceFace.reserve(fatChildFaces.size());
    for(const auto& t:tets){if(t.material==preparedSkinInput.materialIndex)skinReport.tetSourceFace.push_back(t.sourceFace);else fatReport.tetSourceFace.push_back(t.sourceFace);}
    skinReport.firstGlobalFEMNode=preparedSkinInput.firstGlobalFEMNode;fatReport.firstGlobalFEMNode=preparedSkinInput.firstGlobalFEMNode+static_cast<std::uint32_t>(R);
    skinReport.massResidualKg=skinVolume*skinReport.densityKgM3-skinReport.massDebitKg;fatReport.massResidualKg=fatVolume*fatReport.densityKgM3-fatReport.massDebitKg;
    skinReport.donorMassNodes.clear();skinReport.donorMassNodes.reserve(3u*R);
    for(std::uint32_t node=0;node<3u*R;++node){if(!std::isfinite(nodalMass[node])||nodalMass[node]<=0.0||newDonor[node].body==NM_INVALID_INDEX)return fail("refined node has no positive mass or explicit donor owner");skinReport.donorMassNodes.push_back({.nodeIndex=skinReport.firstGlobalFEMNode+node,.donorBody=newDonor[node].body,.massKg=nodalMass[node],.localPosition=newDonor[node].local});}
    fatReport.donorMassNodes.clear();

    // Derive material exterior facets from the actual refined tet adjacency.
    // Interface triangulations must match exactly from the skin and fat sides.
    struct FaceRecord {Face oriented{};unsigned count=0;std::uint32_t parentSurface=NM_INVALID_INDEX,sourceFace=NM_INVALID_INDEX;};
    auto materialBoundary=[&](std::uint32_t material,std::vector<ParentTetBoundaryFace>& boundary)->bool{
        std::map<std::array<std::uint32_t,3>,FaceRecord> uses;
        static constexpr unsigned fidx[4][3]={{1,3,2},{0,2,3},{0,3,1},{0,1,2}};
        for(const auto& t:tets)if(t.material==material)for(unsigned opposite=0;opposite<4u;++opposite){Face f{t.nodes[fidx[opposite][0]],t.nodes[fidx[opposite][1]],t.nodes[fidx[opposite][2]]};auto key=f;std::sort(key.begin(),key.end());auto& rec=uses[key];++rec.count;if(rec.count==1u){rec.oriented=f;rec.parentSurface=t.parentSurfaceFace;rec.sourceFace=t.sourceFace;}else if(rec.count>2u)return false;}
        boundary.clear();for(const auto& [key,rec]:uses){(void)key;if(rec.count==1u)boundary.push_back({rec.oriented,rec.parentSurface,rec.sourceFace});}return true;};
    if(!materialBoundary(preparedSkinInput.materialIndex,result.skinBoundaryFaces)||!materialBoundary(preparedFatInput.materialIndex,result.fatBoundaryFaces))return fail("refined parent tetrahedra contain a nonmanifold material boundary");
    auto classify=[&](const std::vector<ParentTetBoundaryFace>& faces,std::uint32_t kind){std::vector<ParentTetBoundaryFace> outFaces;for(const auto& face:faces){bool all=true;for(auto v:face.nodes){if(kind==0u)all=all&&(v<R);else if(kind==1u)all=all&&(v>=R&&v<2u*R);else if(kind==2u)all=all&&(v>=2u*R&&v<3u*R);}if(all)outFaces.push_back(face);}return outFaces;};
    result.outerSkinFaces=classify(result.skinBoundaryFaces,0u);result.skinFatInterfaceSkinFaces=classify(result.skinBoundaryFaces,1u);result.skinFatInterfaceFatFaces=classify(result.fatBoundaryFaces,1u);result.deepFatFaces=classify(result.fatBoundaryFaces,2u);
    result.refinedOuterSurfaceFaces.reserve(result.outerSkinFaces.size());result.refinedOuterFaceParentSurfaceFace.reserve(result.outerSkinFaces.size());result.refinedOuterFaceSourceIdentity.reserve(result.outerSkinFaces.size());
    for(const auto& face:result.outerSkinFaces){result.refinedOuterSurfaceFaces.push_back(face.nodes);result.refinedOuterFaceParentSurfaceFace.push_back(face.parentSurfaceFace);result.refinedOuterFaceSourceIdentity.push_back(face.sourceFaceIdentity);}
    if(!detailedSourceFaces.empty()){
        // Seed from the actual emitted child boundary triangles and their
        // stable parent identities, then search only a bounded exact edge-ring
        // neighborhood on the registered detailed surface.
        constexpr int maximumAdjacencyRings=16;
        for(std::uint32_t ei=0;ei<m;++ei){
            auto edge=selectedEdges[ei];if(edge[0]>edge[1])std::swap(edge[0],edge[1]);
            const auto ownerA=parentSurface.donorBodyIndices[edge[0]],ownerB=parentSurface.donorBodyIndices[edge[1]];
            if(ownerA==NM_INVALID_INDEX||ownerA!=ownerB)return fail("refined midpoint endpoints do not have one exact source donor owner");
            V3 expectedNormal=add(parentSurface.outwardWorldNormals[edge[0]],parentSurface.outwardWorldNormals[edge[1]]);
            const double expectedLength=norm(expectedNormal);if(!std::isfinite(expectedLength)||expectedLength<=1.0e-12)return fail("refined midpoint lacks a stable registered endpoint normal");
            expectedNormal=mul(expectedNormal,1.0/expectedLength);
            const std::uint32_t midpoint=n+ei;
            std::vector<std::uint32_t> seeds;
            for(const auto& child:result.outerSkinFaces){
                if(child.nodes[0]!=midpoint&&child.nodes[1]!=midpoint&&child.nodes[2]!=midpoint)continue;
                if(child.parentSurfaceFace>=parentSurface.sourceFaceIndices.size()||
                   parentSurface.sourceFaceIndices[child.parentSurfaceFace]!=child.sourceFaceIdentity)
                    return fail("emitted child face has inconsistent parent/source identity");
                auto it=detailedFaceRowByIdentity.find(child.sourceFaceIdentity);
                if(it==detailedFaceRowByIdentity.end())return fail("emitted child parent identity is absent from detailed stable-face table");
                seeds.push_back(it->second);
            }
            std::sort(seeds.begin(),seeds.end());seeds.erase(std::unique(seeds.begin(),seeds.end()),seeds.end());
            if(seeds.empty())return fail("new midpoint is absent from the actual outer child boundary");
            std::vector<int> ringDistance(detailedSourceFaces.size(),-1);std::vector<std::uint32_t> queue;
            queue.reserve(seeds.size()*32u);
            for(auto seed:seeds){ringDistance[seed]=0;queue.push_back(seed);}
            for(std::size_t q=0;q<queue.size();++q){
                const auto f=queue[q];if(ringDistance[f]>=maximumAdjacencyRings)continue;
                for(auto neighbor:detailedFaceAdjacency[f])if(ringDistance[neighbor]<0){ringDistance[neighbor]=ringDistance[f]+1;queue.push_back(neighbor);}
            }
            const V3 outerMid=candidate.femNodes[midpoint];double best=std::numeric_limits<double>::infinity();
            std::uint32_t bestRow=NM_INVALID_INDEX,bestIdentity=NM_INVALID_INDEX,bestRing=0;
            std::array<double,3> bestBeta{};V3 bestPoint{};double bestNormalDot=-1.0;
            for(std::uint32_t fi=0;fi<detailedSourceFaces.size();++fi){
                const int ring=ringDistance[fi];if(ring<0||ring>maximumAdjacencyRings)continue;
                const auto tri=detailedSourceFaces[fi];
                if(detailedSourceDonorBodyByVertex[tri[0]]!=ownerA||detailedSourceDonorBodyByVertex[tri[1]]!=ownerA||detailedSourceDonorBodyByVertex[tri[2]]!=ownerA)continue;
                std::array<double,3> beta{};V3 point{};
                if(!parentTetClosestPointOnTriangleBarycentric(outerMid,detailedSourceWorldPoints[tri[0]],detailedSourceWorldPoints[tri[1]],detailedSourceWorldPoints[tri[2]],beta,point))continue;
                V3 sourceNormal{};for(unsigned c=0;c<3u;++c)sourceNormal=add(sourceNormal,mul(detailedSourceWorldNormals[tri[c]],beta[c]));
                const double sourceNormalLength=norm(sourceNormal);if(!std::isfinite(sourceNormalLength)||sourceNormalLength<=1.0e-12)continue;
                const double normalDot=dot(expectedNormal,mul(sourceNormal,1.0/sourceNormalLength));if(normalDot<0.2)continue;
                const double distance=norm(sub(outerMid,point));const auto identity=detailedSourceFaceIdentities[fi];
                if(distance<best-1.0e-15||(std::abs(distance-best)<=1.0e-15&&identity<bestIdentity)){
                    best=distance;bestRow=fi;bestIdentity=identity;bestRing=static_cast<std::uint32_t>(ring);
                    bestBeta=beta;bestPoint=point;bestNormalDot=normalDot;
                }
            }
            if(bestRow==NM_INVALID_INDEX)return fail("no same-donor, outward-normal-compatible detailed face exists in the 16-ring child-parent neighborhood");
            if(!validateBarycentricCoordinates(bestBeta,error))return false;
            const auto residual=sub(outerMid,bestPoint);
            if(norm(sub(add(bestPoint,residual),outerMid))>2e-7)return fail("new detailed C map plus residual does not reconstruct its emitted outer midpoint");
            result.refinedSourceVertexIds[midpoint]=NM_INVALID_INDEX;
            result.refinedSourceEmbeddingFaceIds[midpoint]=bestRow;
            result.refinedSourceEmbeddingFaceIdentities[midpoint]=bestIdentity;
            result.refinedSourceEmbeddingNeighborhoodRing[midpoint]=bestRing;
            result.refinedSourceEmbeddingNormalDot[midpoint]=static_cast<float>(bestNormalDot);
            result.refinedSourceEmbeddingProjectionDistanceM[midpoint]=static_cast<float>(best);
            result.refinedSourceEmbeddingBarycentric[midpoint]=bestBeta;
            result.refinedSourceEmbeddingResidualWorldM[midpoint]=residual;
        }
        result.detailedSourceEmbeddingComplete=true;
    }
    auto sidewall=[&](const std::vector<ParentTetBoundaryFace>& faces,std::uint32_t layer0,std::uint32_t layer1){std::vector<ParentTetBoundaryFace> side;for(const auto& face:faces){bool inOnly0=true,inOnly1=true;for(auto v:face.nodes){const bool a=v>=layer0*R&&v<(layer0+1u)*R,b=v>=layer1*R&&v<(layer1+1u)*R;inOnly0=inOnly0&&a;inOnly1=inOnly1&&b;}if(!inOnly0&&!inOnly1)side.push_back(face);}return side;};
    result.skinSidewallFaces=sidewall(result.skinBoundaryFaces,0u,1u);result.fatSidewallFaces=sidewall(result.fatBoundaryFaces,1u,2u);
    auto keys=[](const std::vector<ParentTetBoundaryFace>& faces){std::set<std::array<std::uint32_t,3>> out;for(auto f:faces){std::sort(f.nodes.begin(),f.nodes.end());out.insert(f.nodes);}return out;};
    if(keys(result.skinFatInterfaceSkinFaces)!=keys(result.skinFatInterfaceFatFaces))return fail("refined skin and fat tetrahedra do not share the exact same interface triangulation");
    std::size_t expectedFaceCount=parentSurface.faces.size();
    for(const auto& raw:selectedEdges){auto e=raw;if(e[0]>e[1])std::swap(e[0],e[1]);expectedFaceCount+=edgeFaceUse.at(e);}
    if(result.outerSkinFaces.size()!=expectedFaceCount||result.deepFatFaces.size()!=expectedFaceCount)
        return fail("derived refined exterior face count disagrees with exact conforming edge bisection");

    // Check every midpoint's declared coarse-face barycentric point reconstructs
    // the exact midpoint on all three layers, and every exterior child remains
    // within its declared parent triangle with area conserved.
    for(std::uint32_t ei=0;ei<m;++ei){const auto fi=result.midpointParentSurfaceFace[ei];const auto f=parentSurface.faces[fi];const auto beta=result.midpointParentBarycentric[ei];V3 expected{};for(unsigned c=0;c<3u;++c)expected=add(expected,mul(parentSurface.worldPoints[f[c]],beta[c]));const auto actual=candidate.femNodes[n+ei];if(norm(sub(expected,actual))>2e-7)return fail("midpoint coarse-face barycentric ancestry does not reconstruct its outer point");}
    std::map<std::uint32_t,double> childAreaByParent;
    for(const auto& face:result.outerSkinFaces){const auto parent=face.parentSurfaceFace;if(parent>=parentSurface.faces.size())return fail("outer child triangle lacks an original parent face");const auto oldFace=parentSurface.faces[parent];const auto p0=parentSurface.worldPoints[oldFace[0]],p1=parentSurface.worldPoints[oldFace[1]],p2=parentSurface.worldPoints[oldFace[2]];const auto& x=candidate.femNodes;const auto a=x[face.nodes[0]],b=x[face.nodes[1]],c=x[face.nodes[2]];const double area=0.5*norm(cross(sub(b,a),sub(c,a)));childAreaByParent[parent]+=area;
        const auto normal=cross(sub(p1,p0),sub(p2,p0));const double nn=norm(normal);if(nn<=0.0)return fail("parent surface contains a degenerate triangle");for(auto v:face.nodes){const double dist=std::abs(dot(sub(x[v],p0),normal))/nn;if(dist>2e-7)return fail("refined outer boundary triangle leaves its original parent plane");}
    }
    for(std::uint32_t fi=0;fi<parentSurface.faces.size();++fi){const auto f=parentSurface.faces[fi];const double area0=0.5*norm(cross(sub(parentSurface.worldPoints[f[1]],parentSurface.worldPoints[f[0]]),sub(parentSurface.worldPoints[f[2]],parentSurface.worldPoints[f[0]])));if(std::abs(childAreaByParent[fi]-area0)>1e-9*std::max(area0,1e-20))return fail("refined outer child triangle areas do not conserve their parent face");}

    output=std::move(candidate);preparedSkinOutput=std::move(skinReport);preparedFatOutput=std::move(fatReport);return true;
}

} // namespace metalrobo::human::coarse_tissue
