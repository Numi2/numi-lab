#pragma once
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"

namespace metalrobo::human::coarse_tissue {

// Source identity follows an explicit node/tet bijection through a bounded
// geometry edit. This proves correspondence only; it creates no contact,
// force, lock, mass, or physical admission.
struct PhysicalSkinSupportParentMap {
    std::vector<std::uint32_t> childNodeToParent;
    double maximumNodeDisplacementM=0.0;
};

// A retained constraint may outlive changes elsewhere in the mesh. Require its
// own physical node, rest position and explicit source-node ancestry to remain
// exact; this never admits a newly chosen attachment or a moved anchor.
inline bool provePhysicalSkinInheritedAttachmentNode(
        const PhysicalSkinChildMassMappingBundle& b,
        std::span<const std::array<float,3>> actualParentPoints,
        std::uint32_t childNode,std::uint32_t expectedParentNode) {
    if(childNode>=b.currentWorldPoints.size()||childNode>=b.referenceWorldPoints.size()||
       childNode>=b.volumeNodeParentTetrahedra.size()||
       childNode>=b.volumeNodeParentBarycentric.size()||
       expectedParentNode>=actualParentPoints.size()||
       expectedParentNode>=b.parent.currentWorldPoints.size()||
       expectedParentNode>=b.parent.referenceWorldPoints.size())return false;
    const auto tet=b.volumeNodeParentTetrahedra[childNode];
    if(tet>=b.parent.tetrahedra.size())return false;
    unsigned selected=4,units=0;
    for(unsigned corner=0;corner<4;++corner) {
        const auto weight=b.volumeNodeParentBarycentric[childNode][corner];
        if(weight==1.0){selected=corner;++units;}
        else if(weight!=0.0)return false;
    }
    if(units!=1||b.parent.tetrahedra[tet][selected]!=expectedParentNode)return false;
    for(unsigned axis=0;axis<3;++axis) {
        const auto parent=actualParentPoints[expectedParentNode][axis];
        const auto current=b.currentWorldPoints[childNode][axis];
        const auto reference=b.referenceWorldPoints[childNode][axis];
        const auto parentReference=b.parent.referenceWorldPoints[expectedParentNode][axis];
        if(!std::isfinite(parent)||!std::isfinite(current)||!std::isfinite(reference)||
           !std::isfinite(parentReference)||
           double(parent)!=b.parent.currentWorldPoints[expectedParentNode][axis]||
           current!=double(parent)||float(reference)!=float(parentReference))return false;
    }
    return true;
}

inline bool provePhysicalSkinSupportParentMap(
        const PhysicalSkinChildMassMappingBundle& b,
        std::span<const std::array<float,3>> actualParentPoints,
        std::span<const std::array<std::uint32_t,4>> actualParentTets,
        double maximumNodeDisplacementM,
        PhysicalSkinSupportParentMap& output,std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    const auto n=b.parent.currentWorldPoints.size(),m=b.parent.tetrahedra.size();
    if(!n||!m||actualParentPoints.size()!=n||actualParentTets.size()!=m||
       b.parent.referenceWorldPoints.size()!=n||b.currentWorldPoints.size()!=n||
       b.referenceWorldPoints.size()!=n||b.childTetrahedra.size()!=m||
       b.volumeNodeParentTetrahedra.size()!=n||b.volumeNodeParentBarycentric.size()!=n||
       b.volumeNodeParentResidualWorldM.size()!=n||
       !std::isfinite(maximumNodeDisplacementM)||maximumNodeDisplacementM<=0)
        return fail("support parent map needs complete bounded parent topology and node ancestry");
    const auto arraysOK=[&](const auto& p,const auto& c){return p.size()==m&&c.size()==m;};
    if(!arraysOK(b.parent.materialIndices,b.childMaterialIndices)||
       !arraysOK(b.parent.donorBodyIndices,b.childDonorBodyIndices)||
       !arraysOK(b.parent.anatomicalRegionIndices,b.childAnatomicalRegionIndices)||
       !arraysOK(b.parent.stableSourceFaceIdentities,b.childStableSourceFaceIdentities))
        return fail("support parent map has incomplete semantic cell labels");
    for(std::size_t row=0;row<m;++row)
        if(actualParentTets[row]!=b.parent.tetrahedra[row])
            return fail("support parent map packet differs from actual parent cell topology");
    for(std::size_t node=0;node<n;++node)for(unsigned a=0;a<3;++a)
        if(!std::isfinite(actualParentPoints[node][a])||
           double(actualParentPoints[node][a])!=b.parent.currentWorldPoints[node][a])
            return fail("support parent map packet differs from actual compiled parent coordinates");
    PhysicalSkinSupportParentMap next;
    next.childNodeToParent.resize(n,NM_INVALID_INDEX);
    std::vector<std::uint8_t> seen(n,0);
    for(std::size_t node=0;node<n;++node) {
        const auto parentTet=b.volumeNodeParentTetrahedra[node];
        if(parentTet>=m)return fail("support parent node has invalid source tet");
        const auto& beta=b.volumeNodeParentBarycentric[node];
        unsigned selected=4,units=0;
        for(unsigned a=0;a<4;++a) {
            if(beta[a]==1.0){selected=a;++units;}
            else if(beta[a]!=0.0)return fail("support parent map requires exact node ancestry");
        }
        if(units!=1)return fail("support parent map requires one exact source-node corner");
        const auto parent=b.parent.tetrahedra[parentTet][selected];
        if(parent>=n||seen[parent]++)return fail("support parent node map is not bijective");
        const auto& x=b.currentWorldPoints[node];
        const auto& r=b.referenceWorldPoints[node];
        const auto& residual=b.volumeNodeParentResidualWorldM[node];
        double d2=0;
        for(unsigned a=0;a<3;++a) {
            if(!std::isfinite(x[a])||!std::isfinite(r[a])||!std::isfinite(residual[a])||
               !std::isfinite(b.parent.referenceWorldPoints[parent][a])||
               std::abs((b.parent.referenceWorldPoints[parent][a]+residual[a])-r[a])>2.e-10)
                return fail("support parent node ancestry does not reconstruct reference geometry");
            const double d=x[a]-b.parent.currentWorldPoints[parent][a];
            d2+=d*d;
        }
        const double distance=std::sqrt(d2);
        if(!std::isfinite(distance)||distance>maximumNodeDisplacementM)
            return fail("support parent edit exceeds its pinned node-displacement bound");
        next.maximumNodeDisplacementM=std::max(next.maximumNodeDisplacementM,distance);
        next.childNodeToParent[node]=parent;
    }
    std::map<std::array<std::uint32_t,4>,std::uint32_t> parentByNodes;
    for(std::size_t row=0;row<m;++row) {
        auto key=b.parent.tetrahedra[row];std::sort(key.begin(),key.end());
        if(key.back()>=n||std::adjacent_find(key.begin(),key.end())!=key.end()||
           !parentByNodes.emplace(key,std::uint32_t(row)).second)
            return fail("support parent cell topology is degenerate or duplicated");
    }
    const auto& flow=b.spatialMassTransportResult;
    if(flow.childParentOffsets.size()!=m+1||flow.childParentOffsets.front()!=0||
       flow.childParentOffsets.back()!=flow.childParentRows.size())
        return fail("support parent map requires the prepared conservative mass ancestry");
    std::vector<std::uint8_t> used(m,0);
    for(std::size_t row=0;row<m;++row) {
        auto mapped=b.childTetrahedra[row];
        for(auto& node:mapped) {
            if(node>=n)return fail("support child cell has an invalid node");
            node=next.childNodeToParent[node];
        }
        auto key=mapped;std::sort(key.begin(),key.end());
        const auto found=parentByNodes.find(key);
        if(found==parentByNodes.end()||used[found->second]++||
           !physical_skin_child_mass_mapping_detail::evenTetNodePermutation(
               b.parent.tetrahedra[found->second],mapped))
            return fail("support child topology is not an oriented parent cell bijection");
        const auto p=found->second;
        const auto& cell=b.childTetrahedra[row];
        const auto positive=[&](const auto& points) {
            const double v=physical_skin_child_mass_detail::signedTetVolume(
                points[cell[0]],points[cell[1]],points[cell[2]],points[cell[3]]);
            return std::isfinite(v)&&v>0;
        };
        if(!positive(b.currentWorldPoints)||!positive(b.referenceWorldPoints))
            return fail("support child geometry has a nonpositive current or reference cell");
        if(b.parent.materialIndices[p]!=b.childMaterialIndices[row]||
           b.parent.donorBodyIndices[p]!=b.childDonorBodyIndices[row]||
           b.parent.anatomicalRegionIndices[p]!=b.childAnatomicalRegionIndices[row]||
           b.parent.stableSourceFaceIdentities[p]!=b.childStableSourceFaceIdentities[row])
            return fail("support parent cell material/donor/region/source ancestry changed");
        const auto begin=flow.childParentOffsets[row],end=flow.childParentOffsets[row+1];
        if(begin>=end||end>flow.childParentRows.size())
            return fail("support parent mass ancestry row is invalid");
        for(auto e=begin;e<end;++e)
            if(flow.childParentRows[e]!=p)
                return fail("support parent mass ancestry crosses its source cell");
    }
    output=std::move(next);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
