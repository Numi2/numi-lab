#pragma once

#include "NumiHumanSkinFatLocalRemeshLabels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct LocalSkinFatTetNode {
    std::uint32_t existingObjectNode = NM_INVALID_INDEX;
    V3 world{};
    V3 referenceWorld{};
    V3 donorLocal{};
    std::uint32_t donorBody = NM_INVALID_INDEX;
    std::uint32_t sourceVertex = NM_INVALID_INDEX;
    std::uint32_t layerOrInterior = NM_INVALID_INDEX;
    std::uint32_t parentTetRow = NM_INVALID_INDEX;
    std::array<double, 4> parentBarycentric{};
};

struct LocalSkinFatTetSpliceInput {
    std::span<const std::uint32_t> replacedTetRows;
    std::span<const std::uint32_t> oldTetParentSurfaceFaceRows;
    std::span<const V3> oldNodeDonorLocalPoints;
    std::span<const LocalSkinFatTetNode> nodes;
    std::span<const std::array<std::uint32_t, 4>> childTetrahedra;
    std::span<const std::uint32_t> childMaterialIndices;
    std::span<const std::uint32_t> childParentTetRows;
    std::span<const std::uint32_t> childParentSurfaceFaceRows;
    std::span<const std::uint32_t> childSourceFaceIdentities;
    std::span<const std::uint32_t> childDonorBodyLabels;
    std::span<const std::uint32_t> childBasisRegionLabels;
    std::span<const std::uint32_t> addedContactLocalNodeIndices;
    std::array<std::uint64_t, 4> materialSourceIdentity{};
    std::array<std::uint64_t, 4> materialFrameSourceIdentity{};
    std::array<std::uint64_t, 4> referenceSourceIdentity{};
};

struct LocalSkinFatTetSpliceReport {
    std::size_t originalNodeCount = 0u;
    std::size_t originalTetCount = 0u;
    std::uint32_t firstGlobalFEMNode = NM_INVALID_INDEX;
    std::vector<std::uint32_t> localToObjectNode;
    // Exact arrays consumed by remapRegisteredSkinFatReductionSourceLabels.
    std::vector<std::uint32_t> nodeParentTetRows;
    std::vector<std::array<double, 4>> nodeParentBarycentric;
    std::vector<std::uint32_t> nodeSourceVertexIndices;
    std::vector<std::uint32_t> nodeLayerOrInterior;
    std::vector<std::uint32_t> nodeDonorBodyIndices;
    std::vector<std::uint32_t> tetParentTetRows;
    std::vector<std::uint32_t> tetMaterialIndices;
    std::vector<std::uint32_t> tetSourceFaceIdentities;
    std::vector<std::uint32_t> tetDonorBodyLabels;
    std::vector<std::uint32_t> tetBasisRegionLabels;
    std::vector<std::uint32_t> tetParentSurfaceFaceRows;
    std::vector<std::uint32_t> oldTetToNewTet;
    std::vector<NumiHumanTissueMassNode> skinMaterialMassNodes;
    std::vector<NumiHumanTissueMassNode> fatMaterialMassNodes;
    std::vector<NumiHumanTissueMassNode> aggregateDonorMassNodes;
    double skinVolumeBeforeM3 = 0.0, skinVolumeAfterM3 = 0.0;
    double fatVolumeBeforeM3 = 0.0, fatVolumeAfterM3 = 0.0;
    double skinMassBeforeKg = 0.0, skinMassAfterKg = 0.0;
    double fatMassBeforeKg = 0.0, fatMassAfterKg = 0.0;
    double maxAbsoluteTetVolumeChangeM3 = 0.0;
    std::size_t replacedParentsWithoutChildCentroid = 0u;
    std::map<std::uint32_t, double> donorMassDeltaKg;
    std::map<std::uint32_t, V3> donorFirstMomentDeltaKgM;
    std::map<std::uint32_t, std::array<double, 9>> donorSecondMomentDeltaKgM2;
};

namespace local_tet_splice_detail {
using FaceKey = std::array<std::uint32_t, 3>;
struct FaceUse { unsigned count=0u; int orientationBalance=0; std::uint32_t materialA=NM_INVALID_INDEX, materialB=NM_INVALID_INDEX; };
inline int orientationSign(const std::array<std::uint32_t,3>& f) {
    unsigned inv=0u; for(unsigned i=0;i<3u;++i) for(unsigned j=i+1u;j<3u;++j) inv += f[i]>f[j];
    return (inv&1u)?-1:1;
}
inline FaceKey key(const std::array<std::uint32_t,3>& f) { auto k=f; std::sort(k.begin(),k.end()); return k; }
inline double volume6(const V3& a,const V3& b,const V3& c,const V3& d) {
    const V3 u=sub(b,a),v=sub(c,a),w=sub(d,a);
    return u[0]*(v[1]*w[2]-v[2]*w[1]) - u[1]*(v[0]*w[2]-v[2]*w[0]) + u[2]*(v[0]*w[1]-v[1]*w[0]);
}
inline bool finitePoint(const V3& p) { return std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);}); }
inline bool zeroIdentity(const std::array<std::uint64_t,4>& a) { return std::all_of(a.begin(),a.end(),[](auto v){return v==0u;}); }
inline void addTetFaces(std::map<FaceKey,FaceUse>& faces,const std::array<std::uint32_t,4>& t,std::uint32_t material) {
    static constexpr unsigned f[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
    for(const auto& ix:f) {
        std::array<std::uint32_t,3> oriented{t[ix[0]],t[ix[1]],t[ix[2]]};
        auto& u=faces[key(oriented)];
        if(u.count==0u)u.materialA=material;else u.materialB=material;
        ++u.count;u.orientationBalance+=orientationSign(oriented);
    }
}
inline bool faceBoundary(const std::map<FaceKey,FaceUse>& m,std::map<FaceKey,int>& out) {
    out.clear(); for(const auto& [k,u]:m) {
        if(u.count>2u||(u.count==2u&&u.orientationBalance!=0))return false;
        if(u.count==1u)out.emplace(k,u.orientationBalance);
    } return true;
}
inline std::map<FaceKey,int> materialInterfaces(const std::map<FaceKey,FaceUse>& m) {
    std::map<FaceKey,int> out; for(const auto& [k,u]:m) if(u.count==2u&&u.materialA!=u.materialB) out.emplace(k,u.orientationBalance); return out;
}
inline bool closedBoundaryEdges(const std::map<FaceKey,int>& boundary) {
    std::map<std::array<std::uint32_t,2>,unsigned> edges;
    for(const auto& [f,s]:boundary) { (void)s; for(unsigned i=0;i<3u;++i) { auto e=std::array<std::uint32_t,2>{f[i],f[(i+1u)%3u]};std::sort(e.begin(),e.end());++edges[e]; } }
    return std::all_of(edges.begin(),edges.end(),[](const auto& e){return e.second==2u;});
}
inline bool cyclicVertexLinks(const std::map<FaceKey,int>& boundary) {
    std::map<std::uint32_t,std::map<std::uint32_t,std::set<std::uint32_t>>> links;
    for(const auto& [f,s]:boundary) {
        (void)s;
        for(unsigned i=0;i<3u;++i) {
            const auto v=f[i],a=f[(i+1u)%3u],b=f[(i+2u)%3u];
            links[v][a].insert(b);links[v][b].insert(a);
        }
    }
    for(const auto& [v,graph]:links) {
        (void)v;
        if(graph.empty()||std::any_of(graph.begin(),graph.end(),[](const auto& row){return row.second.size()!=2u;}))return false;
        std::set<std::uint32_t> visited;std::vector<std::uint32_t> stack{graph.begin()->first};
        while(!stack.empty()) {const auto n=stack.back();stack.pop_back();if(!visited.insert(n).second)continue;for(auto q:graph.at(n))stack.push_back(q);}
        if(visited.size()!=graph.size())return false;
    }
    return !links.empty();
}
inline bool barycentric(const V3& p,const std::array<V3,4>& t,std::array<double,4>& b) {
    const double det=volume6(t[0],t[1],t[2],t[3]);
    if(!std::isfinite(det)||std::abs(det)<1.0e-30)return false;
    b={volume6(p,t[1],t[2],t[3])/det,volume6(t[0],p,t[2],t[3])/det,
       volume6(t[0],t[1],p,t[3])/det,volume6(t[0],t[1],t[2],p)/det};
    return std::all_of(b.begin(),b.end(),[](double x){return std::isfinite(x);});
}
inline double tetVolume(const numi::matter::ObjectSource& o,const std::array<std::uint32_t,4>& t) {
    const auto& p=o.femReferenceNodes.empty()?o.femNodes:o.femReferenceNodes;
    return volume6(p[t[0]],p[t[1]],p[t[2]],p[t[3]])/6.0;
}
} // namespace local_tet_splice_detail

// Transactional parent-contained local cavity replacement. The candidate
// must preserve the exact oriented cavity boundary and material interface.
// New rows require exact old-tet barycentric ancestry; this does not accept
// tissue placed outside the original reference domain (such as a new axilla wall).
[[nodiscard]] inline bool spliceRegisteredSkinFatLocalTets(
        numi::matter::ObjectSource& object,
        PreparedOrganicFEM& preparedSkin,
        PreparedOrganicFEM& preparedFat,
        const RegisteredSkinFatReductionSourceLabels& originalLabels,
        const LocalSkinFatTetSpliceInput& in,
        LocalSkinFatTetSpliceReport& outputReport,
        RegisteredSkinFatReductionSourceLabels& outputLabels,
        std::string& error) {
    using namespace local_tet_splice_detail;
    const auto fail=[&](const char* why){error=why;return false;};
    constexpr double kBaryTolerance=1.0e-10;
    constexpr double kRefinementReconstructionToleranceM=1.0e-8;
    constexpr double kVolumeClosureTolerance=1.0e-6;
    constexpr double kFloat32NormalizedDetFloor=1.0e-5;
    const auto oldNodeCount=object.femNodes.size(),oldTetCount=object.tetrahedra.size();
    if(object.representation!=numi::matter::Representation::fem||oldNodeCount<4u||oldTetCount==0u||
       object.femMaterialIndices.size()!=oldTetCount||
       (!object.femReferenceNodes.empty()&&object.femReferenceNodes.size()!=oldNodeCount)||
       originalLabels.sourceTetrahedronCount!=oldTetCount||
       originalLabels.sourceNodeVertexIndices.size()!=oldNodeCount||
       originalLabels.sourceNodeLayerIndices.size()!=oldNodeCount||
       originalLabels.sourceNodeDonorBodyIndices.size()!=oldNodeCount||
       originalLabels.tetrahedronSourceDonorBodyIndices.size()!=oldTetCount||
       originalLabels.tetrahedronSourceFaceIndices.size()!=oldTetCount||
       originalLabels.tetrahedronBasisRegionLabels.size()!=oldTetCount||
       originalLabels.tetrahedronSourceParentBodyIndices.size()!=oldTetCount||
       in.oldTetParentSurfaceFaceRows.size()!=oldTetCount||
       in.oldNodeDonorLocalPoints.size()!=oldNodeCount||
       in.replacedTetRows.empty()||in.nodes.empty()||in.childTetrahedra.empty()||
       in.childMaterialIndices.size()!=in.childTetrahedra.size()||
       in.childParentTetRows.size()!=in.childTetrahedra.size()||
       in.childParentSurfaceFaceRows.size()!=in.childTetrahedra.size()||
       in.childSourceFaceIdentities.size()!=in.childTetrahedra.size()||
       in.childDonorBodyLabels.size()!=in.childTetrahedra.size()||
       in.childBasisRegionLabels.size()!=in.childTetrahedra.size()||
       preparedSkin.materialIndex==NM_INVALID_INDEX||preparedFat.materialIndex==NM_INVALID_INDEX||
       preparedSkin.materialIndex==preparedFat.materialIndex||
       !std::isfinite(preparedSkin.densityKgM3)||preparedSkin.densityKgM3<=0.0||
       !std::isfinite(preparedFat.densityKgM3)||preparedFat.densityKgM3<=0.0||
       !std::isfinite(preparedSkin.massDebitKg)||preparedSkin.massDebitKg<=0.0||
       !std::isfinite(preparedFat.massDebitKg)||preparedFat.massDebitKg<=0.0||
       preparedSkin.firstGlobalFEMNode==NM_INVALID_INDEX||
       preparedSkin.donorMassNodes.empty())
        return fail("local skin/fat splice source, labels, or prepared material reports are incomplete");
    if(!object.femCohesiveInterfaces.empty()||!object.femCohesiveMortarPatches.empty()||
       !object.fieldBoundaries.empty()||!object.mutationCommands.empty())
        return fail("local skin/fat splice does not remap cohesive, field-boundary, or mutation references");
    if(oldNodeCount>std::numeric_limits<std::uint32_t>::max()||
       in.nodes.size()>std::numeric_limits<std::uint32_t>::max()-oldNodeCount||
       oldTetCount>std::numeric_limits<std::uint32_t>::max()||
       in.childTetrahedra.size()>std::numeric_limits<std::uint32_t>::max()-oldTetCount)
        return fail("local skin/fat splice index capacity would overflow");
    if(!zeroIdentity(object.femMaterialSourceIdentity)&&zeroIdentity(in.materialSourceIdentity))
        return fail("local skin/fat splice requires a regenerated material-source identity");
    if(!object.femMaterialFrameRotations.empty()&&
       (object.femMaterialFrameRotations.size()!=oldTetCount||
        (!zeroIdentity(object.femMaterialFrameSourceIdentity)&&zeroIdentity(in.materialFrameSourceIdentity))))
        return fail("local skin/fat splice lacks complete regenerated material-frame provenance");
    if(!object.femReferenceNodes.empty()&&!zeroIdentity(object.femReferenceSourceIdentity)&&
       (!in.nodes.empty()&&std::any_of(in.nodes.begin(),in.nodes.end(),[](const auto& n){return n.existingObjectNode==NM_INVALID_INDEX;})&&
        zeroIdentity(in.referenceSourceIdentity)))
        return fail("appended local nodes require regenerated reference-source identity");

    const auto stableSource=[&](std::size_t t){return originalLabels.tetrahedronOriginalSourceFaceIndices.empty()?originalLabels.tetrahedronSourceFaceIndices[t]:originalLabels.tetrahedronOriginalSourceFaceIndices[t];};
    if(!originalLabels.registeredMechanicalFaceIndices.empty()&&
       std::any_of(in.childParentSurfaceFaceRows.begin(),in.childParentSurfaceFaceRows.end(),[&](auto f){return f>=originalLabels.registeredMechanicalFaceIndices.size();}))
        return fail("local child parent-surface face row is outside the registered source surface");
    std::set<std::uint32_t> replaced;
    for(const auto t:in.replacedTetRows)if(t>=oldTetCount||!replaced.insert(t).second)
        return fail("local skin/fat splice repeats or exceeds a parent tetrahedron row");
    std::vector<std::vector<std::size_t>> childByParent(oldTetCount);
    for(std::size_t c=0;c<in.childTetrahedra.size();++c) {
        const auto p=in.childParentTetRows[c];
        if(p>=oldTetCount||!replaced.contains(p))return fail("local child tetrahedron does not name a selected old parent");
        childByParent[p].push_back(c);
    }
    const std::size_t finalTetCount=oldTetCount-in.replacedTetRows.size()+in.childTetrahedra.size();
    if(finalTetCount>std::numeric_limits<std::uint32_t>::max())
        return fail("local tet splice index capacity would overflow");

    LocalSkinFatTetSpliceReport next;
    next.originalNodeCount=oldNodeCount;next.originalTetCount=oldTetCount;
    next.firstGlobalFEMNode=preparedSkin.firstGlobalFEMNode;
    next.oldTetToNewTet.assign(oldTetCount,NM_INVALID_INDEX);
    next.localToObjectNode.resize(in.nodes.size(),NM_INVALID_INDEX);
    auto candidate=object;
    const auto firstGlobal=preparedSkin.firstGlobalFEMNode;
    const auto oldWorld=object.femNodes;
    const auto oldReference=object.femReferenceNodes.empty()?object.femNodes:object.femReferenceNodes;
    std::set<std::uint32_t> localOldNodes;std::vector<std::uint32_t> appendedLocal;
    for(std::size_t i=0;i<in.nodes.size();++i) {
        const auto& node=in.nodes[i];
        if(!finitePoint(node.world)||!finitePoint(node.referenceWorld)||!finitePoint(node.donorLocal)||
           node.donorBody==NM_INVALID_INDEX||(node.layerOrInterior>2u&&node.layerOrInterior!=NM_INVALID_INDEX))
            return fail("local splice node has nonfinite coordinates or invalid owner/layer labels");
        if(node.existingObjectNode!=NM_INVALID_INDEX) {
            const auto g=node.existingObjectNode;
            if(g>=oldNodeCount||!localOldNodes.insert(g).second||
               node.world!=oldWorld[g]||node.referenceWorld!=oldReference[g]||
               node.donorLocal!=in.oldNodeDonorLocalPoints[g]||
               node.sourceVertex!=originalLabels.sourceNodeVertexIndices[g]||
               node.layerOrInterior!=originalLabels.sourceNodeLayerIndices[g]||
               node.donorBody!=originalLabels.sourceNodeDonorBodyIndices[g])
                return fail("local splice changed or duplicated a retained node, coordinate, or source owner");
            next.localToObjectNode[i]=g;
        } else {
            const auto p=node.parentTetRow;
            if(p>=oldTetCount)return fail("appended local splice node lacks an original parent tetrahedron");
            double sum=0.0;V3 reconstructed{};
            const auto& parent=object.tetrahedra[p].nodes;
            for(const auto v:parent)if(v>=oldNodeCount)return fail("appended node parent tetrahedron references an absent original node");
            for(unsigned k=0;k<4u;++k) {
                const double b=node.parentBarycentric[k];
                if(!std::isfinite(b)||b<-kBaryTolerance||b>1.0+kBaryTolerance)
                    return fail("appended local splice node has invalid parent barycentrics");
                sum+=b;const auto v=oldReference[parent[k]];
                for(unsigned a=0;a<3u;++a)reconstructed[a]+=b*v[a];
            }
            if(std::abs(sum-1.0)>kBaryTolerance||norm(sub(reconstructed,node.referenceWorld))>kRefinementReconstructionToleranceM)
                return fail("appended local splice node is outside its declared reference parent");
            if(!object.femReferenceNodes.empty()&&zeroIdentity(in.referenceSourceIdentity)&&
               !zeroIdentity(object.femReferenceSourceIdentity))
                return fail("appended local node changed a pinned reference without a new identity");
            next.localToObjectNode[i]=static_cast<std::uint32_t>(candidate.femNodes.size());
            appendedLocal.push_back(static_cast<std::uint32_t>(i));
            candidate.femNodes.push_back(node.world);
            if(!candidate.femReferenceNodes.empty())candidate.femReferenceNodes.push_back(node.referenceWorld);
            else if(node.referenceWorld!=node.world)return fail("local node cannot introduce a separate reference into a legacy object");
        }
    }
    for(std::size_t c=0;c<in.childTetrahedra.size();++c) {
        const auto& t=in.childTetrahedra[c];std::array<std::uint32_t,4> mapped{};
        for(unsigned k=0;k<4u;++k) {
            if(t[k]>=next.localToObjectNode.size())return fail("local child tetrahedron references an absent splice node");
            mapped[k]=next.localToObjectNode[t[k]];
        }
        auto sorted=mapped;std::sort(sorted.begin(),sorted.end());
        if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())return fail("local child tetrahedron repeats a node");
        if(in.childMaterialIndices[c]!=object.femMaterialIndices[in.childParentTetRows[c]])
            return fail("local child changes its original parent material");
        const auto parentTet=in.childParentTetRows[c];
        if(in.childSourceFaceIdentities[c]!=stableSource(parentTet)||
           in.childParentSurfaceFaceRows[c]!=in.oldTetParentSurfaceFaceRows[parentTet])
            return fail("local child source-face or registered parent-surface identity differs from its selected parent");
        std::array<V3,4> parentPoints{};
        for(unsigned k=0;k<4u;++k) {
            const auto oldNode=object.tetrahedra[parentTet].nodes[k];
            if(oldNode>=oldNodeCount)return fail("child parent tetrahedron references an absent original node");
            parentPoints[k]=oldReference[oldNode];
        }
        V3 centroid{};
        for(unsigned k=0;k<4u;++k)for(unsigned a=0;a<3u;++a)
            centroid[a]+=0.25*oldReference[mapped[k]][a];
        std::array<double,4> centroidBary{};
        if(!barycentric(centroid,parentPoints,centroidBary)||
           std::any_of(centroidBary.begin(),centroidBary.end(),[](double x){return x < -2.0e-7 || x > 1.0+2.0e-7;}))
            return fail("local child centroid is not contained by its declared original parent tet");
        if(in.childDonorBodyLabels[c]!=originalLabels.tetrahedronSourceDonorBodyIndices[in.childParentTetRows[c]]||
           in.childBasisRegionLabels[c]!=originalLabels.tetrahedronBasisRegionLabels[in.childParentTetRows[c]])
            return fail("local child donor or basis label differs from its exact parent");
        const double vol=volume6(candidate.femReferenceNodes.empty()?candidate.femNodes[mapped[0]]:candidate.femReferenceNodes[mapped[0]],
            candidate.femReferenceNodes.empty()?candidate.femNodes[mapped[1]]:candidate.femReferenceNodes[mapped[1]],
            candidate.femReferenceNodes.empty()?candidate.femNodes[mapped[2]]:candidate.femReferenceNodes[mapped[2]],
            candidate.femReferenceNodes.empty()?candidate.femNodes[mapped[3]]:candidate.femReferenceNodes[mapped[3]])/6.0;
        if(!std::isfinite(vol)||vol<=0.0)return fail("local child has a nonpositive reference tetrahedron");
        std::array<V3,4> f32{};double longest=0.0;
        for(unsigned k=0;k<4u;++k) {
            const auto& p=candidate.femReferenceNodes.empty()?candidate.femNodes[mapped[k]]:candidate.femReferenceNodes[mapped[k]];
            for(unsigned a=0;a<3u;++a)f32[k][a]=static_cast<double>(static_cast<float>(p[a]));
            for(unsigned q=0;q<k;++q)longest=std::max(longest,norm(sub(f32[k],f32[q])));
        }
        const double det=volume6(f32[0],f32[1],f32[2],f32[3])/(longest*longest*longest);
        if(!std::isfinite(det)||det<kFloat32NormalizedDetFloor)return fail("local child fails the existing Float32 normalized determinant floor");
        const auto& cp=candidate.femNodes;
        std::array<V3,4> currentF32{};double currentLongest=0.0;
        for(unsigned k=0;k<4u;++k) {
            for(unsigned a=0;a<3u;++a)currentF32[k][a]=static_cast<double>(static_cast<float>(cp[mapped[k]][a]));
            for(unsigned q=0;q<k;++q)currentLongest=std::max(currentLongest,norm(sub(currentF32[k],currentF32[q])));
        }
        const double currentDet=volume6(currentF32[0],currentF32[1],currentF32[2],currentF32[3])/
            (currentLongest*currentLongest*currentLongest);
        if(!std::isfinite(currentDet)||currentDet<kFloat32NormalizedDetFloor)
            return fail("local child fails the Float32 current-space normalized determinant floor");
    }

    // Compare the cavity's exact oriented boundary and internal skin/fat interface.
    std::map<FaceKey,FaceUse> oldFaces,newFaces;
    for(auto p:replaced) {
        const auto t=object.tetrahedra[p].nodes;
        if(tetVolume(object,t)<=0.0)return fail("selected old cavity contains a nonpositive reference tet");
        addTetFaces(oldFaces,t,object.femMaterialIndices[p]);
    }
    std::array<double,2> oldVol{},newVol{};
    for(auto p:replaced) {
        const auto mat=object.femMaterialIndices[p];const double v=tetVolume(object,object.tetrahedra[p].nodes);
        if(mat==preparedSkin.materialIndex)oldVol[0]+=v;else if(mat==preparedFat.materialIndex)oldVol[1]+=v;else return fail("selected cavity contains a material outside the paired skin/fat reports");
    }
    std::vector<std::array<std::uint32_t,4>> mappedChildren(in.childTetrahedra.size());
    for(std::size_t c=0;c<in.childTetrahedra.size();++c) {
        auto t=in.childTetrahedra[c];for(auto& x:t)x=next.localToObjectNode[x];mappedChildren[c]=t;
        const double v=tetVolume(candidate,t);if(v<=0.0)return fail("local child orientation changed after object-node mapping");
        const std::size_t mi=in.childMaterialIndices[c]==preparedSkin.materialIndex?0u:1u;
        newVol[mi]+=v;
        addTetFaces(newFaces,t,in.childMaterialIndices[c]);
    }
    std::map<FaceKey,int> oldBoundary,newBoundary;
    if(!faceBoundary(oldFaces,oldBoundary)||!faceBoundary(newFaces,newBoundary)||
       !closedBoundaryEdges(oldBoundary)||!closedBoundaryEdges(newBoundary)||
       !cyclicVertexLinks(oldBoundary)||!cyclicVertexLinks(newBoundary)||oldBoundary!=newBoundary||
       materialInterfaces(oldFaces)!=materialInterfaces(newFaces))
        return fail("local child cavity changes the exact oriented stitch boundary or skin/fat interface");
    for(unsigned m=0;m<2u;++m)if(oldVol[m]<=0.0||newVol[m]<=0.0||
        std::abs(newVol[m]-oldVol[m])>kVolumeClosureTolerance*oldVol[m])
        return fail("local child cavity does not conserve a material volume within owner tolerance");
    for(auto p:replaced)if(childByParent[p].empty())++next.replacedParentsWithoutChildCentroid;
    next.maxAbsoluteTetVolumeChangeM3=std::max(std::abs(newVol[0]-oldVol[0]),std::abs(newVol[1]-oldVol[1]));

    // Build final connectivity in old-parent order, retaining every outside tet row's order.
    candidate.tetrahedra.clear();candidate.femMaterialIndices.clear();
    std::vector<std::uint32_t> finalParent,finalStable,finalDonor,finalBasis,finalParentSurface;
    candidate.tetrahedra.reserve(oldTetCount-in.replacedTetRows.size()+in.childTetrahedra.size());
    candidate.femMaterialIndices.reserve(candidate.tetrahedra.capacity());
    if(!originalLabels.registeredMechanicalFaceIndices.empty()&&
       std::any_of(in.childParentSurfaceFaceRows.begin(),in.childParentSurfaceFaceRows.end(),[&](auto f){return f>=originalLabels.registeredMechanicalFaceIndices.size();}))
        return fail("local child parent-surface face row is outside the registered source surface");
    for(std::uint32_t old=0;old<oldTetCount;++old) {
        if(!replaced.contains(old)) {
            next.oldTetToNewTet[old]=static_cast<std::uint32_t>(candidate.tetrahedra.size());
            candidate.tetrahedra.push_back(object.tetrahedra[old]);candidate.femMaterialIndices.push_back(object.femMaterialIndices[old]);
            finalParent.push_back(old);finalStable.push_back(stableSource(old));
            finalDonor.push_back(originalLabels.tetrahedronSourceDonorBodyIndices[old]);finalBasis.push_back(originalLabels.tetrahedronBasisRegionLabels[old]);
            finalParentSurface.push_back(in.oldTetParentSurfaceFaceRows[old]);
        } else {
            for(auto c:childByParent[old]) {
                next.oldTetToNewTet[old]=static_cast<std::uint32_t>(candidate.tetrahedra.size());
                candidate.tetrahedra.push_back({mappedChildren[c]});candidate.femMaterialIndices.push_back(in.childMaterialIndices[c]);
                finalParent.push_back(old);finalStable.push_back(in.childSourceFaceIdentities[c]);
                finalDonor.push_back(in.childDonorBodyLabels[c]);finalBasis.push_back(in.childBasisRegionLabels[c]);
                finalParentSurface.push_back(in.childParentSurfaceFaceRows[c]);
            }
        }
    }
    if(!candidate.femPreconditionerRegionIds.empty())candidate.femPreconditionerRegionIds=finalBasis;
    if(!candidate.femMaterialFrameRotations.empty()) {
        std::vector<std::array<double,4>> frames;frames.reserve(candidate.tetrahedra.size());
        for(std::uint32_t old=0;old<oldTetCount;++old)if(!replaced.contains(old))frames.push_back(object.femMaterialFrameRotations[old]);else for(std::size_t c=0;c<childByParent[old].size();++c)frames.push_back(object.femMaterialFrameRotations[old]);
        candidate.femMaterialFrameRotations=std::move(frames);
        if(!zeroIdentity(in.materialFrameSourceIdentity))candidate.femMaterialFrameSourceIdentity=in.materialFrameSourceIdentity;
    }
    if(!zeroIdentity(in.materialSourceIdentity))candidate.femMaterialSourceIdentity=in.materialSourceIdentity;
    if(!zeroIdentity(in.referenceSourceIdentity))candidate.femReferenceSourceIdentity=in.referenceSourceIdentity;
    if(candidate.femCapacity.nodes&&candidate.femCapacity.nodes<candidate.femNodes.size())return fail("local node splice exceeds authored node capacity");
    if(candidate.femCapacity.tetrahedra&&candidate.femCapacity.tetrahedra<candidate.tetrahedra.size())return fail("local tet splice exceeds authored tet capacity");

    for(const auto l:in.addedContactLocalNodeIndices) {
        if(l>=next.localToObjectNode.size())return fail("new contact node map references an absent local node");
        candidate.femContactNodes.push_back(next.localToObjectNode[l]);
    }
    std::sort(candidate.femContactNodes.begin(),candidate.femContactNodes.end());
    candidate.femContactNodes.erase(std::unique(candidate.femContactNodes.begin(),candidate.femContactNodes.end()),candidate.femContactNodes.end());
    if(std::any_of(candidate.femContactNodes.begin(),candidate.femContactNodes.end(),[&](auto n){return n>=candidate.femNodes.size();}))return fail("local splice contact node map is out of range");
    std::vector<std::uint32_t> disabled;
    for(auto encoded:object.femDisabledContactSurfaceFaces) {
        const auto old=encoded/4u,corner=encoded%4u;
        if(old>=oldTetCount||replaced.contains(old))return fail("local splice needs an explicit remap for a disabled face on a replaced tet");
        disabled.push_back(4u*next.oldTetToNewTet[old]+corner);
    }
    candidate.femDisabledContactSurfaceFaces=std::move(disabled);

    // Rebuild full source label arrays and verify them through the same consumer used by the app.
    next.nodeParentTetRows.assign(candidate.femNodes.size(),NM_INVALID_INDEX);
    next.nodeParentBarycentric.assign(candidate.femNodes.size(),{});
    next.nodeSourceVertexIndices=originalLabels.sourceNodeVertexIndices;
    next.nodeLayerOrInterior=originalLabels.sourceNodeLayerIndices;
    next.nodeDonorBodyIndices=originalLabels.sourceNodeDonorBodyIndices;
    if(originalLabels.remeshedOriginalNodeCount) {
        if(originalLabels.sourceNodeParentTetrahedronIndices.size()!=oldNodeCount||originalLabels.sourceNodeParentBarycentric.size()!=oldNodeCount)return fail("existing node remesh ancestry is incomplete");
        std::copy(originalLabels.sourceNodeParentTetrahedronIndices.begin(),originalLabels.sourceNodeParentTetrahedronIndices.end(),next.nodeParentTetRows.begin());
        std::copy(originalLabels.sourceNodeParentBarycentric.begin(),originalLabels.sourceNodeParentBarycentric.end(),next.nodeParentBarycentric.begin());
    }
    for(const auto local:appendedLocal) {
        const auto& row=in.nodes[local];const auto g=next.localToObjectNode[local];
        next.nodeParentTetRows[g]=row.parentTetRow;next.nodeParentBarycentric[g]=row.parentBarycentric;
        next.nodeSourceVertexIndices.push_back(row.sourceVertex);next.nodeLayerOrInterior.push_back(row.layerOrInterior);next.nodeDonorBodyIndices.push_back(row.donorBody);
    }
    next.tetParentTetRows=std::move(finalParent);next.tetMaterialIndices=candidate.femMaterialIndices;
    next.tetSourceFaceIdentities=std::move(finalStable);next.tetDonorBodyLabels=std::move(finalDonor);
    next.tetBasisRegionLabels=std::move(finalBasis);next.tetParentSurfaceFaceRows=std::move(finalParentSurface);
    RegisteredSkinFatReductionSourceLabels labelsNext;
    if(!remapRegisteredSkinFatReductionSourceLabels(object,candidate,originalLabels,next,labelsNext,error))return false;

    // Rebuild material-specific nodal mass and the legacy unique aggregate exactly once.
    const auto nodeCount=candidate.femNodes.size();
    if(next.nodeDonorBodyIndices.size()!=nodeCount||in.oldNodeDonorLocalPoints.size()!=oldNodeCount)
        return fail("local splice mass owner lacks one donor-local frame per retained node");
    std::vector<std::uint32_t> nodeBody=originalLabels.sourceNodeDonorBodyIndices;
    std::vector<V3> nodeLocal(in.oldNodeDonorLocalPoints.begin(),in.oldNodeDonorLocalPoints.end());
    for(auto local:appendedLocal){nodeBody.push_back(in.nodes[local].donorBody);nodeLocal.push_back(in.nodes[local].donorLocal);}
    if(nodeBody.size()!=nodeCount||nodeLocal.size()!=nodeCount)return fail("local splice donor frame arrays do not cover the output object");
    std::map<std::uint32_t,NumiHumanTissueMassNode> skinMass,fatMass,aggregate;
    auto addMass=[&](std::map<std::uint32_t,NumiHumanTissueMassNode>& dst,std::uint32_t node,double dm)->bool {
        const auto body=nodeBody[node];if(body==NM_INVALID_INDEX||!finitePoint(nodeLocal[node])||!std::isfinite(dm)||dm<=0.0)return false;
        const auto global=firstGlobal+node;auto [it,inserted]=dst.emplace(global,NumiHumanTissueMassNode{.nodeIndex=global,.donorBody=body,.massKg=dm,.localPosition=nodeLocal[node]});
        if(!inserted){if(it->second.donorBody!=body)return false;for(unsigned a=0;a<3u;++a)if(std::abs(it->second.localPosition[a]-nodeLocal[node][a])>1.0e-7)return false;it->second.massKg+=dm;}return true;
    };
    double skinVol=0.0,fatVol=0.0,skinMassKg=0.0,fatMassKg=0.0;
    for(std::size_t ti=0;ti<candidate.tetrahedra.size();++ti) {
        const auto mat=candidate.femMaterialIndices[ti];const bool isSkin=mat==preparedSkin.materialIndex,isFat=mat==preparedFat.materialIndex;
        if(!isSkin&&!isFat)return fail("local splice output contains a material outside the paired reports");
        const double rho=isSkin?preparedSkin.densityKgM3:preparedFat.densityKgM3;
        const double vol=tetVolume(candidate,candidate.tetrahedra[ti].nodes);if(!std::isfinite(vol)||vol<=0.0)return fail("local splice final ObjectSource contains a nonpositive reference tet");
        const double dm=rho*vol/4.0;if(isSkin){skinVol+=vol;skinMassKg+=rho*vol;}else{fatVol+=vol;fatMassKg+=rho*vol;}
        for(auto node:candidate.tetrahedra[ti].nodes)if(!addMass(isSkin?skinMass:fatMass,node,dm))return fail("local splice material mass node has ambiguous donor/frame ownership");
    }
    if(std::abs(skinMassKg-preparedSkin.massDebitKg)>1.0e-8*preparedSkin.massDebitKg||
       std::abs(fatMassKg-preparedFat.massDebitKg)>1.0e-8*preparedFat.massDebitKg)
        return fail("local splice material-lumped mass no longer closes to the existing donor debit");
    for(auto& [node,row]:skinMass)aggregate.emplace(node,row);
    for(auto& [node,row]:fatMass) {
        auto [it,inserted]=aggregate.emplace(node,row);
        if(!inserted){if(it->second.donorBody!=row.donorBody)return fail("shared skin/fat node donor ownership changed across the material interface");for(unsigned a=0;a<3u;++a)if(std::abs(it->second.localPosition[a]-row.localPosition[a])>1.0e-7)return fail("shared skin/fat node donor frame changed across the material interface");it->second.massKg+=row.massKg;}
    }
    next.skinMaterialMassNodes.reserve(skinMass.size());for(auto& [id,row]:skinMass){(void)id;next.skinMaterialMassNodes.push_back(row);}
    next.fatMaterialMassNodes.reserve(fatMass.size());for(auto& [id,row]:fatMass){(void)id;next.fatMaterialMassNodes.push_back(row);}
    next.aggregateDonorMassNodes.reserve(aggregate.size());for(auto& [id,row]:aggregate){(void)id;next.aggregateDonorMassNodes.push_back(row);}

    const auto oldSkinVol=preparedSkin.volumeM3,oldFatVol=preparedFat.volumeM3;
    next.skinVolumeBeforeM3=oldSkinVol;next.skinVolumeAfterM3=skinVol;next.fatVolumeBeforeM3=oldFatVol;next.fatVolumeAfterM3=fatVol;
    next.skinMassBeforeKg=preparedSkin.massDebitKg;next.skinMassAfterKg=skinMassKg;next.fatMassBeforeKg=preparedFat.massDebitKg;next.fatMassAfterKg=fatMassKg;
    auto updateReport=[&](PreparedOrganicFEM& r,std::uint32_t mat,double volume,const std::map<std::uint32_t,NumiHumanTissueMassNode>& massRows) {
        r.volumeM3=volume;r.massResidualKg=r.densityKgM3*volume-r.massDebitKg;r.donorMassNodes.clear();
        r.tetSourceFace.clear();r.tetChildIndex.clear();r.tetParentTetrahedron.clear();r.tetParentSurfaceFace.clear();r.tetParentMaterialIndex.clear();r.tetSourceFaceIdentity.clear();
        r.nodeSourceVertex=next.nodeSourceVertexIndices;
        for(std::size_t ti=0;ti<candidate.tetrahedra.size();++ti)if(candidate.femMaterialIndices[ti]==mat){
            r.tetSourceFace.push_back(next.tetSourceFaceIdentities[ti]);r.tetChildIndex.push_back(static_cast<std::uint32_t>(ti));r.tetParentTetrahedron.push_back(next.tetParentTetRows[ti]);
            r.tetParentSurfaceFace.push_back(next.tetParentSurfaceFaceRows[ti]);r.tetParentMaterialIndex.push_back(mat);r.tetSourceFaceIdentity.push_back(next.tetSourceFaceIdentities[ti]);
        }
        for(const auto& [id,row]:massRows){(void)id;r.donorMassNodes.push_back(row);}
    };
    PreparedOrganicFEM skinNext=preparedSkin,fatNext=preparedFat;
    updateReport(skinNext,preparedSkin.materialIndex,skinVol,skinMass);updateReport(fatNext,preparedFat.materialIndex,fatVol,fatMass);
    skinNext.donorMassNodes=next.aggregateDonorMassNodes;fatNext.donorMassNodes.clear();

    // Per-body debit and moment deltas, measured from the old published mass rows.
    std::map<std::uint32_t,NumiHumanTissueMassNode> oldMass;
    for(const auto& m:preparedSkin.donorMassNodes)oldMass.emplace(m.nodeIndex,m);
    for(const auto& m:next.aggregateDonorMassNodes) {
        const auto found=oldMass.find(m.nodeIndex);
        if(found!=oldMass.end()&&found->second.donorBody!=m.donorBody)
            return fail("local splice changed an existing donor mass-node body owner");
        const double prior=found==oldMass.end()?0.0:found->second.massKg;
        const auto body=m.donorBody;const double dm=m.massKg-prior;next.donorMassDeltaKg[body]+=dm;
        const auto& x=candidate.femReferenceNodes.empty()?candidate.femNodes[m.nodeIndex-firstGlobal]:candidate.femReferenceNodes[m.nodeIndex-firstGlobal];
        for(unsigned a=0;a<3u;++a)next.donorFirstMomentDeltaKgM[body][a]+=dm*x[a];
        const double xx=x[0]*x[0],yy=x[1]*x[1],zz=x[2]*x[2];
        std::array<double,9> q{xx, x[0]*x[1],x[0]*x[2],x[1]*x[0],yy,x[1]*x[2],x[2]*x[0],x[2]*x[1],zz};
        for(unsigned a=0;a<3u;++a)for(unsigned b=0;b<3u;++b)next.donorSecondMomentDeltaKgM2[body][3u*a+b]+=dm*q[3u*a+b];
        oldMass.erase(m.nodeIndex);
    }
    for(const auto& [id,m]:oldMass){
        next.donorMassDeltaKg[m.donorBody]-=m.massKg;
        if(id<firstGlobal||id-firstGlobal>=oldNodeCount)return fail("old donor mass node lies outside the object node range");
        const auto& x=oldReference[id-firstGlobal];
        for(unsigned a=0;a<3u;++a)next.donorFirstMomentDeltaKgM[m.donorBody][a]-=m.massKg*x[a];
        for(unsigned a=0;a<3u;++a)for(unsigned b=0;b<3u;++b)next.donorSecondMomentDeltaKgM2[m.donorBody][3u*a+b]-=m.massKg*x[a]*x[b];
    }

    // Full final topology audit: no nonmanifold faces or orientation mismatch; closed volume boundary.
    std::map<FaceKey,FaceUse> allFaces;
    for(std::size_t ti=0;ti<candidate.tetrahedra.size();++ti) {
        if(tetVolume(candidate,candidate.tetrahedra[ti].nodes)<=0.0)return fail("local splice final object has a nonpositive tet");
        addTetFaces(allFaces,candidate.tetrahedra[ti].nodes,candidate.femMaterialIndices[ti]);
    }
    std::map<FaceKey,int> allBoundary;
    if(!faceBoundary(allFaces,allBoundary)||!closedBoundaryEdges(allBoundary)||!cyclicVertexLinks(allBoundary))return fail("local splice final object has nonmanifold or open volume topology");
    next.tetMaterialIndices=candidate.femMaterialIndices;
    outputReport=std::move(next);outputLabels=std::move(labelsNext);
    object=std::move(candidate);preparedSkin=std::move(skinNext);preparedFat=std::move(fatNext);
    error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
