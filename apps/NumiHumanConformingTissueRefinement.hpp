#pragma once
// Offline source-construction proof for the existing physical skin/fat owner.
// Runtime mechanics, physical admission and exact surface checks remain separate.
#include "NumiHumanParentReferenceMassTransfer.hpp"
#include <set>
#include <utility>

namespace metalrobo::human::coarse_tissue {

struct ConformingTissueRefinementInput {
    std::span<const std::array<double,3>> sourceReferencePoints;
    std::span<const std::array<std::uint32_t,4>> sourceTetrahedra;
    // Ordered, normalized edges used by the 530 complete-star constructor.
    std::span<const std::array<std::uint32_t,2>> orderedSplitEdges;
    std::span<const std::array<double,3>> refinedReferencePoints;
    std::span<const std::array<std::uint32_t,4>> refinedTetrahedra;
    std::span<const std::uint32_t> childSourceTetRows;
    std::span<const double> childSourceMeasureFractions;
    std::span<const std::uint32_t> nodeSourceOffsets;
    std::span<const std::uint32_t> nodeSourceRows;
    std::span<const double> nodeSourceWeights;
    std::span<const std::uint32_t> sourceMaterialIds,refinedMaterialIds;
    std::span<const std::uint32_t> sourceGeometricDonorIds,refinedGeometricDonorIds;
    std::span<const std::uint32_t> sourceRegionIds,refinedRegionIds;
    std::span<const std::uint32_t> sourceStableFaceIds,refinedStableFaceIds;
    // Frozen 330 row domain, authenticated against source 002 by the file owner.
    std::span<const double> immutableParentReferenceVolumeM3;
    std::span<const double> immutableParentMassKg;
    std::span<const double> materialDensityLowerKgM3,materialDensityUpperKgM3;
    std::string_view sourceVolumeSHA256,refinedVolumeSHA256,lineagePacketSHA256;
    std::string_view parentMassPacketSHA256,parentSourceContributionSHA256;
    std::uint32_t sourceBodyCount=0u;
};

struct ConformingTissueRefinementResult {
    std::vector<std::uint32_t> childParentOffsets,childParentRows;
    std::vector<double> childParentImmutableReferenceMeasureM3;
    std::vector<double> childSourceReferenceMeasureM3;
    std::vector<double> childSourceMeasureFractions;
    std::vector<double> childMassKg,childReferenceVolumeM3,childImpliedDensityKgM3;
    double maximumCanonicalRoundingDistanceM=0.0;
    double maximumParentMassClosureErrorKg=0.0;
    double maximumParentMeasureClosureErrorM3=0.0;
    std::string verificationSHA256;
};

namespace conforming_tissue_refinement_detail {
using Point=std::array<double,3>;
using Tet=std::array<std::uint32_t,4>;
inline double signedVolume(const Tet& tet,std::span<const Point> p) {
    if(std::any_of(tet.begin(),tet.end(),[&](auto node){return node>=p.size();}))
        return std::numeric_limits<double>::quiet_NaN();
    Point a{},b{},c{};
    for(unsigned i=0;i<3u;++i){a[i]=p[tet[1]][i]-p[tet[0]][i];b[i]=p[tet[2]][i]-p[tet[0]][i];c[i]=p[tet[3]][i]-p[tet[0]][i];}
    return (a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6.0;
}
struct Cell {Tet nodes;std::uint32_t parent;double fraction;};
}

// Replays the actual complete-star construction, instead of accepting an
// arbitrary volume-closing child collection as a proof of subdivision.
// Atomic: output is assigned only after construction, labels and mass pass.
[[nodiscard]] inline bool verifyConformingTissueRefinement(
        const ConformingTissueRefinementInput& in,
        ConformingTissueRefinementResult& out,std::string& error) {
    using namespace conforming_tissue_refinement_detail;
    const auto fail=[&](const char* text){error=text;return false;};
    const auto ns=in.sourceReferencePoints.size(),nt=in.sourceTetrahedra.size();
    const auto nr=in.refinedReferencePoints.size(),nc=in.refinedTetrahedra.size();
    if(!ns||!nt||!nr||!nc||!in.sourceBodyCount||
       nr!=ns+in.orderedSplitEdges.size()||
       nr>std::numeric_limits<std::uint32_t>::max()||nc>std::numeric_limits<std::uint32_t>::max()||
       in.childSourceTetRows.size()!=nc||in.childSourceMeasureFractions.size()!=nc||
       in.nodeSourceOffsets.size()!=nr+1u||in.nodeSourceOffsets.front()!=0u||
       in.nodeSourceOffsets.back()!=in.nodeSourceRows.size()||
       in.nodeSourceRows.size()!=in.nodeSourceWeights.size()||
       in.immutableParentReferenceVolumeM3.size()!=nt||in.immutableParentMassKg.size()!=nt||
       in.sourceMaterialIds.size()!=nt||in.sourceGeometricDonorIds.size()!=nt||
       in.sourceRegionIds.size()!=nt||in.sourceStableFaceIds.size()!=nt||
       in.refinedMaterialIds.size()!=nc||in.refinedGeometricDonorIds.size()!=nc||
       in.refinedRegionIds.size()!=nc||in.refinedStableFaceIds.size()!=nc)
        return fail("conforming refinement input domains are incomplete");
    for(const auto hash:{in.sourceVolumeSHA256,in.refinedVolumeSHA256,in.lineagePacketSHA256,
                         in.parentMassPacketSHA256,in.parentSourceContributionSHA256})
        if(!spatial_mass_transport_detail::validSHA256(hash))
            return fail("conforming refinement requires pinned source, child, lineage and immutable mass identities");
    std::vector<Point> points(in.sourceReferencePoints.begin(),in.sourceReferencePoints.end());
    std::vector<std::vector<std::pair<std::uint32_t,double>>> ancestry(ns);
    for(std::size_t n=0;n<ns;++n){
        for(const auto value:points[n])
            if(!std::isfinite(value)||static_cast<double>(static_cast<float>(value))!=value)
                return fail("conforming refinement source is not canonical finite Float32 initialization geometry");
        ancestry[n]={{static_cast<std::uint32_t>(n),1.0}};
    }
    std::vector<Cell> cells;cells.reserve(nc);
    std::set<Tet> uniqueSource;
    std::vector<double> sourceVolume(nt);
    for(std::size_t c=0;c<nt;++c){
        auto sorted=in.sourceTetrahedra[c];std::sort(sorted.begin(),sorted.end());
        sourceVolume[c]=signedVolume(in.sourceTetrahedra[c],points);
        if(sorted[0]==sorted[1]||sorted[1]==sorted[2]||sorted[2]==sorted[3]||
           !uniqueSource.insert(sorted).second||!std::isfinite(sourceVolume[c])||sourceVolume[c]<=0.0||
           in.sourceGeometricDonorIds[c]>=in.sourceBodyCount||
           in.sourceRegionIds[c]==std::numeric_limits<std::uint32_t>::max()||
           in.sourceStableFaceIds[c]==std::numeric_limits<std::uint32_t>::max())
            return fail("conforming refinement source cell is invalid or lacks typed anatomical identity");
        cells.push_back({in.sourceTetrahedra[c],static_cast<std::uint32_t>(c),1.0});
    }
    for(const auto& edge:in.orderedSplitEdges){
        const auto a=edge[0],b=edge[1];
        if(a>=b||b>=points.size())return fail("conforming refinement edge is not normalized or references an unavailable node");
        const auto node=static_cast<std::uint32_t>(points.size());
        Point midpoint{};for(unsigned axis=0;axis<3u;++axis)midpoint[axis]=points[a][axis]*0.5+points[b][axis]*0.5;
        std::map<std::uint32_t,double> combined;
        for(const auto endpoint:{a,b})for(const auto& [row,weight]:ancestry[endpoint])combined[row]+=weight*0.5;
        std::vector<Cell> untouched,left,right;
        for(const auto& cell:cells){
            const bool hasA=std::find(cell.nodes.begin(),cell.nodes.end(),a)!=cell.nodes.end();
            const bool hasB=std::find(cell.nodes.begin(),cell.nodes.end(),b)!=cell.nodes.end();
            if(hasA&&hasB){
                auto l=cell,r=cell;l.fraction*=0.5;r.fraction*=0.5;
                for(auto& v:l.nodes)if(v==b)v=node;
                for(auto& v:r.nodes)if(v==a)v=node;
                left.push_back(l);right.push_back(r);
            }else untouched.push_back(cell);
        }
        if(left.empty()||untouched.size()+left.size()+right.size()>nc)
            return fail("conforming refinement edge has no complete incident star or exceeds declared child domain");
        points.push_back(midpoint);ancestry.emplace_back(combined.begin(),combined.end());
        untouched.insert(untouched.end(),left.begin(),left.end());
        untouched.insert(untouched.end(),right.begin(),right.end());
        cells=std::move(untouched);
    }
    if(cells.size()!=nc)return fail("conforming refinement replay cell count differs from declared child");
    ConformingTissueRefinementResult next;
    for(std::size_t n=0;n<nr;++n){
        const auto begin=in.nodeSourceOffsets[n],end=in.nodeSourceOffsets[n+1u];
        if(begin>end||end>in.nodeSourceRows.size()||end-begin!=ancestry[n].size())
            return fail("conforming refinement node ancestry does not match exact construction");
        for(std::size_t j=0;j<ancestry[n].size();++j)
            if(in.nodeSourceRows[begin+j]!=ancestry[n][j].first||
               in.nodeSourceWeights[begin+j]!=ancestry[n][j].second)
                return fail("conforming refinement source-node ancestry was changed");
        double rounding2=0.0;
        for(unsigned axis=0;axis<3u;++axis){
            const double canonical=static_cast<double>(static_cast<float>(points[n][axis]));
            if(!std::isfinite(canonical)||in.refinedReferencePoints[n][axis]!=canonical)
                return fail("conforming refinement output coordinates differ from canonical replay");
            const double d=canonical-points[n][axis];rounding2+=d*d;
        }
        next.maximumCanonicalRoundingDistanceM=std::max(next.maximumCanonicalRoundingDistanceM,std::sqrt(rounding2));
    }
    std::vector<double> measure(nc),geometric(nc);
    next.childParentOffsets.resize(nc+1u);next.childParentRows.resize(nc);
    next.childSourceReferenceMeasureM3.resize(nc);next.childSourceMeasureFractions.resize(nc);
    for(std::size_t c=0;c<nc;++c){
        const auto& cell=cells[c];const auto p=cell.parent;
        if(cell.nodes!=in.refinedTetrahedra[c]||p!=in.childSourceTetRows[c]||
           cell.fraction!=in.childSourceMeasureFractions[c]||
           in.refinedMaterialIds[c]!=in.sourceMaterialIds[p]||
           in.refinedGeometricDonorIds[c]!=in.sourceGeometricDonorIds[p]||
           in.refinedRegionIds[c]!=in.sourceRegionIds[p]||
           in.refinedStableFaceIds[c]!=in.sourceStableFaceIds[p])
            return fail("conforming refinement connectivity, source fraction or anatomical label differs from replay");
        const auto unrounded=signedVolume(cell.nodes,points);
        geometric[c]=signedVolume(cell.nodes,in.refinedReferencePoints);
        if(!std::isfinite(unrounded)||unrounded<=0.0||!std::isfinite(geometric[c])||geometric[c]<=0.0)
            return fail("conforming refinement creates a nonpositive source or emitted Float32 cell");
        const double expected=sourceVolume[p]*cell.fraction;
        if(std::abs(unrounded-expected)>std::max(1e-25,2e-11*expected))
            return fail("conforming refinement source geometry disagrees with its subdivision measure");
        next.childParentOffsets[c]=static_cast<std::uint32_t>(c);next.childParentRows[c]=p;
        next.childSourceMeasureFractions[c]=cell.fraction;
        next.childSourceReferenceMeasureM3[c]=expected;
        measure[c]=in.immutableParentReferenceVolumeM3[p]*cell.fraction;
    }
    next.childParentOffsets[nc]=static_cast<std::uint32_t>(nc);
    ParentReferenceMassTransferInput massInput{
        in.immutableParentReferenceVolumeM3,in.immutableParentMassKg,in.sourceMaterialIds,
        in.sourceGeometricDonorIds,geometric,in.refinedMaterialIds,next.childParentOffsets,
        next.childParentRows,measure,in.materialDensityLowerKgM3,in.materialDensityUpperKgM3};
    ParentReferenceMassTransferResult massResult;
    if(!assembleParentReferenceMassTransfer(massInput,massResult,error))return false;
    next.childParentImmutableReferenceMeasureM3=std::move(measure);
    next.childMassKg=std::move(massResult.childMassKg);
    next.childImpliedDensityKgM3=std::move(massResult.childImpliedDensityKgM3);
    next.childReferenceVolumeM3=std::move(geometric);
    next.maximumParentMassClosureErrorKg=massResult.maximumParentMassClosureErrorKg;
    next.maximumParentMeasureClosureErrorM3=massResult.maximumParentMeasureClosureErrorM3;
    spatial_mass_transport_detail::SHA256Builder hash;hash.string("NumiHumanConformingTissueRefinement/v1");
    for(const auto value:{in.sourceVolumeSHA256,in.refinedVolumeSHA256,in.lineagePacketSHA256,in.parentMassPacketSHA256,in.parentSourceContributionSHA256})hash.string(value);
    hash.u32(in.sourceBodyCount);hash.u64(ns);hash.u64(nt);hash.u64(nr);hash.u64(nc);
    for(const auto& p:in.sourceReferencePoints)for(auto v:p)hash.real(v);
    for(const auto& t:in.sourceTetrahedra)for(auto v:t)hash.u32(v);
    for(const auto& e:in.orderedSplitEdges)for(auto v:e)hash.u32(v);
    for(auto v:in.nodeSourceOffsets)hash.u32(v);
    for(auto v:in.nodeSourceRows)hash.u32(v);
    for(auto v:in.nodeSourceWeights)hash.real(v);
    for(std::size_t c=0;c<nc;++c){
        for(auto v:in.refinedTetrahedra[c])hash.u32(v);
        hash.u32(in.childSourceTetRows[c]);hash.real(in.childSourceMeasureFractions[c]);
        hash.u32(in.refinedMaterialIds[c]);hash.u32(in.refinedGeometricDonorIds[c]);
        hash.u32(in.refinedRegionIds[c]);hash.u32(in.refinedStableFaceIds[c]);
        hash.real(next.childParentImmutableReferenceMeasureM3[c]);hash.real(next.childMassKg[c]);
        hash.real(next.childReferenceVolumeM3[c]);
    }
    for(const auto& p:in.refinedReferencePoints)for(auto v:p)hash.real(v);
    for(std::size_t c=0;c<nt;++c){hash.real(in.immutableParentReferenceVolumeM3[c]);hash.real(in.immutableParentMassKg[c]);}
    for(auto v:in.materialDensityLowerKgM3)hash.real(v);
    for(auto v:in.materialDensityUpperKgM3)hash.real(v);
    next.verificationSHA256=hash.finish();out=std::move(next);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
