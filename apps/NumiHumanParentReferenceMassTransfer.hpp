#pragma once

// Conservative transfer from an authoritative parent-tetrahedron mass ledger.
// This validates algebraic coverage/density bounds; it does not prove that a
// caller-supplied overlap measure is geometrically correct. The independent,
// pinned overlap or reference-pullback receipt remains a separate requirement.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct ParentReferenceMassTransferInput {
    std::span<const double> parentReferenceVolumeM3;
    std::span<const double> parentMassKg;
    std::span<const std::uint32_t> parentMaterialIds;
    std::span<const std::uint32_t> parentOriginDonorIds;
    std::span<const double> childGeometricReferenceVolumeM3;
    std::span<const std::uint32_t> childMaterialIds;
    // CSR entries are reference-pullback overlap measures m_cp in m^3, not
    // fractions. Each row must be sorted by parent ID and each parent column
    // must close to its authoritative reference volume.
    std::span<const std::uint32_t> childParentOffsets;
    std::span<const std::uint32_t> childParentRows;
    std::span<const double> childParentReferenceMeasureM3;
    // Bounds are indexed by material ID.
    std::span<const double> materialDensityLowerKgM3;
    std::span<const double> materialDensityUpperKgM3;
    double relativeClosureTolerance=2.0e-11;
    double absoluteMeasureToleranceM3=1.0e-15;
    double absoluteMassToleranceKg=1.0e-14;
};

struct ParentReferenceMassTransferResult {
    std::vector<std::uint32_t> childParentOffsets;
    std::vector<std::uint32_t> childParentRows;
    // Owner contract: row fractions sum to one; measure*fraction is the
    // parent-reference measure debited from that parent.
    std::vector<double> childParentVolumeFractions;
    std::vector<double> childAncestryMeasureVolumeM3;
    std::vector<double> childMassKg;
    std::vector<double> childImpliedDensityKgM3;
    std::vector<double> parentMeasureCoveredM3;
    std::vector<double> parentMassTransferredKg;
    double maximumParentMeasureClosureErrorM3=0.0;
    double maximumParentMassClosureErrorKg=0.0;
    double maximumChildFractionSumError=0.0;
    std::map<std::pair<std::uint32_t,std::uint32_t>,double> parentMassByMaterialOriginDonorKg;
    std::map<std::pair<std::uint32_t,std::uint32_t>,double> childMassByMaterialOriginDonorKg;
};

[[nodiscard]] inline bool assembleParentReferenceMassTransfer(
        const ParentReferenceMassTransferInput& in,
        ParentReferenceMassTransferResult& out,
        std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    const std::size_t np=in.parentReferenceVolumeM3.size();
    const std::size_t nc=in.childGeometricReferenceVolumeM3.size();
    if(np==0u||nc==0u||in.parentMassKg.size()!=np||in.parentMaterialIds.size()!=np||
       in.parentOriginDonorIds.size()!=np||in.childMaterialIds.size()!=nc||
       in.childParentOffsets.size()!=nc+1u||in.childParentOffsets.front()!=0u||
       in.childParentOffsets.back()!=in.childParentRows.size()||
       in.childParentRows.size()!=in.childParentReferenceMeasureM3.size()||
       in.materialDensityLowerKgM3.empty()||
       in.materialDensityUpperKgM3.size()!=in.materialDensityLowerKgM3.size()||
       !std::isfinite(in.relativeClosureTolerance)||in.relativeClosureTolerance<0.0||
       !std::isfinite(in.absoluteMeasureToleranceM3)||in.absoluteMeasureToleranceM3<0.0||
       !std::isfinite(in.absoluteMassToleranceKg)||in.absoluteMassToleranceKg<0.0)
        return fail("parent-mass transfer arrays or tolerances are invalid");
    ParentReferenceMassTransferResult next;
    next.childParentOffsets.assign(in.childParentOffsets.begin(),in.childParentOffsets.end());
    next.childParentRows.assign(in.childParentRows.begin(),in.childParentRows.end());
    next.childParentVolumeFractions.resize(in.childParentRows.size());
    next.childAncestryMeasureVolumeM3.assign(nc,0.0);
    next.childMassKg.assign(nc,0.0);
    next.childImpliedDensityKgM3.assign(nc,0.0);
    next.parentMeasureCoveredM3.assign(np,0.0);
    next.parentMassTransferredKg.assign(np,0.0);
    for(std::size_t p=0;p<np;++p){
        if(!std::isfinite(in.parentReferenceVolumeM3[p])||in.parentReferenceVolumeM3[p]<=0.0||
           !std::isfinite(in.parentMassKg[p])||in.parentMassKg[p]<=0.0||
           in.parentMaterialIds[p]>=in.materialDensityLowerKgM3.size()||
           in.parentOriginDonorIds[p]==std::numeric_limits<std::uint32_t>::max())
            return fail("parent mass row has invalid volume, mass, material, or donor identity");
        const auto m=in.parentMaterialIds[p];
        const double rho=in.parentMassKg[p]/in.parentReferenceVolumeM3[p];
        if(!std::isfinite(in.materialDensityLowerKgM3[m])||
           !std::isfinite(in.materialDensityUpperKgM3[m])||
           in.materialDensityLowerKgM3[m]>in.materialDensityUpperKgM3[m]||
           !std::isfinite(rho)||rho<in.materialDensityLowerKgM3[m]||rho>in.materialDensityUpperKgM3[m])
            return fail("authoritative parent mass implies density outside material bounds");
        next.parentMassByMaterialOriginDonorKg[{m,in.parentOriginDonorIds[p]}]+=in.parentMassKg[p];
    }
    for(std::size_t c=0;c<nc;++c){
        const double geom=in.childGeometricReferenceVolumeM3[c];
        if(!std::isfinite(geom)||geom<=0.0||in.childMaterialIds[c]>=in.materialDensityLowerKgM3.size()||
           in.childParentOffsets[c]>=in.childParentOffsets[c+1u]||
           in.childParentOffsets[c+1u]>in.childParentRows.size())
            return fail("child mass row has invalid geometric volume, material, or empty ancestry");
        std::uint32_t previous=0u;
        for(std::uint32_t e=in.childParentOffsets[c];e<in.childParentOffsets[c+1u];++e){
            const auto p=in.childParentRows[e];
            const double measure=in.childParentReferenceMeasureM3[e];
            if(p>=np||(e>in.childParentOffsets[c]&&p<=previous)||!std::isfinite(measure)||measure<=0.0||
               in.parentMaterialIds[p]!=in.childMaterialIds[c])
                return fail("child-parent measure has invalid, unsorted, duplicate, or cross-material ancestry");
            previous=p;
            next.childAncestryMeasureVolumeM3[c]+=measure;
            next.parentMeasureCoveredM3[p]+=measure;
            const double mass=in.parentMassKg[p]*(measure/in.parentReferenceVolumeM3[p]);
            if(!std::isfinite(mass)||mass<=0.0)return fail("transferred child mass is nonpositive or nonfinite");
            next.childMassKg[c]+=mass;
            next.parentMassTransferredKg[p]+=mass;
            next.childMassByMaterialOriginDonorKg[{in.childMaterialIds[c],in.parentOriginDonorIds[p]}]+=mass;
        }
        const double measure=next.childAncestryMeasureVolumeM3[c];
        if(!std::isfinite(measure)||measure<=0.0||!std::isfinite(next.childMassKg[c])||next.childMassKg[c]<=0.0)
            return fail("child pullback measure or transferred mass is nonpositive");
        double fractions=0.0;
        for(std::uint32_t e=in.childParentOffsets[c];e<in.childParentOffsets[c+1u];++e){
            const double f=in.childParentReferenceMeasureM3[e]/measure;
            next.childParentVolumeFractions[e]=f;fractions+=f;
        }
        next.maximumChildFractionSumError=std::max(next.maximumChildFractionSumError,std::abs(fractions-1.0));
        const double rho=next.childMassKg[c]/geom;
        const auto m=in.childMaterialIds[c];
        if(!std::isfinite(rho)||rho<in.materialDensityLowerKgM3[m]||rho>in.materialDensityUpperKgM3[m])
            return fail("transferred child mass implies density outside material bounds");
        next.childImpliedDensityKgM3[c]=rho;
    }
    for(std::size_t p=0;p<np;++p){
        const double measureError=next.parentMeasureCoveredM3[p]-in.parentReferenceVolumeM3[p];
        const double massError=next.parentMassTransferredKg[p]-in.parentMassKg[p];
        next.maximumParentMeasureClosureErrorM3=std::max(next.maximumParentMeasureClosureErrorM3,std::abs(measureError));
        next.maximumParentMassClosureErrorKg=std::max(next.maximumParentMassClosureErrorKg,std::abs(massError));
        const double measureTol=std::max(in.absoluteMeasureToleranceM3,in.relativeClosureTolerance*in.parentReferenceVolumeM3[p]);
        const double massTol=std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*in.parentMassKg[p]);
        if(std::abs(measureError)>measureTol||std::abs(massError)>massTol)
            return fail("reference measure or mass does not close an authoritative parent column");
    }
    if(next.maximumChildFractionSumError>5.0e-14)
        return fail("computed child ancestry fractions do not sum to one");
    for(const auto& [key,mass]:next.parentMassByMaterialOriginDonorKg){
        const auto it=next.childMassByMaterialOriginDonorKg.find(key);
        const double child=it==next.childMassByMaterialOriginDonorKg.end()?0.0:it->second;
        const double tol=std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*mass);
        if(std::abs(child-mass)>tol)return fail("per-material origin-donor mass does not close");
    }
    out=std::move(next);error.clear();return true;
}

[[nodiscard]] inline double parentReferenceTetVolumeM3(
        const std::array<std::uint32_t,4>& tet,
        std::span<const std::array<double,3>> points) {
    if(std::any_of(tet.begin(),tet.end(),[&](auto node){return node>=points.size();}))return std::numeric_limits<double>::quiet_NaN();
    const auto sub=[](const auto& a,const auto& b){return std::array<double,3>{a[0]-b[0],a[1]-b[1],a[2]-b[2]};};
    const auto a=sub(points[tet[1]],points[tet[0]]),b=sub(points[tet[2]],points[tet[0]]),c=sub(points[tet[3]],points[tet[0]]);
    const double det=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
    return std::abs(det)/6.0;
}

// Derive exact reference-coordinate measures for unchanged topology. This is
// appropriate when child row c is precisely parent row c; moved current points
// are irrelevant, but callers must use the actual source reference geometry.
[[nodiscard]] inline bool deriveIdentityParentMeasures(
        std::span<const std::array<double,3>> parentReferencePoints,
        std::span<const std::array<std::uint32_t,4>> parentTets,
        std::span<const std::uint32_t> parentMaterials,
        std::span<const std::array<std::uint32_t,4>> childTets,
        std::span<const std::uint32_t> childMaterials,
        std::vector<std::uint32_t>& offsets,
        std::vector<std::uint32_t>& parents,
        std::vector<double>& measures,
        std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    if(parentTets.empty()||childTets.size()!=parentTets.size()||parentMaterials.size()!=parentTets.size()||childMaterials.size()!=childTets.size())
        return fail("identity parent map requires equal nonempty parent and child cell tables");
    std::vector<std::uint32_t> nextOffsets(childTets.size()+1u,0u),nextParents(childTets.size());
    std::vector<double> nextMeasures(childTets.size());
    for(std::size_t c=0;c<childTets.size();++c){
        auto a=parentTets[c],b=childTets[c];std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
        if(a!=b||parentMaterials[c]!=childMaterials[c])return fail("identity parent map topology/material differs at a child row");
        const double v=parentReferenceTetVolumeM3(parentTets[c],parentReferencePoints);
        if(!std::isfinite(v)||v<=0.0)return fail("identity parent map has nonpositive reference parent volume");
        nextParents[c]=static_cast<std::uint32_t>(c);nextMeasures[c]=v;nextOffsets[c+1u]=static_cast<std::uint32_t>(c+1u);
    }
    offsets=std::move(nextOffsets);parents=std::move(nextParents);measures=std::move(nextMeasures);error.clear();return true;
}

// Derive pullback measures for a conforming refinement whose every child lies
// in exactly one original parent tet. Original shared vertices are identified
// by node ID; inserted vertices use pinned parent-tet+barycentric ancestry,
// then are remapped across a shared parent face/edge by source coordinates.
// Cross-parent/retriangulated children must use independently computed overlap
// measures with assembleParentReferenceMassTransfer instead.
[[nodiscard]] inline bool deriveSingleParentRefinementMeasures(
        std::span<const std::array<double,3>> parentReferencePoints,
        std::span<const std::array<std::uint32_t,4>> parentTets,
        std::span<const std::uint32_t> parentMaterials,
        std::span<const std::array<std::uint32_t,4>> childTets,
        std::span<const std::uint32_t> childMaterials,
        std::span<const std::uint32_t> childParentRows,
        std::span<const std::uint32_t> nodeParentTetRows,
        std::span<const std::array<double,4>> nodeParentBarycentric,
        std::vector<std::uint32_t>& offsets,
        std::vector<std::uint32_t>& parents,
        std::vector<double>& measures,
        std::string& error,
        double barycentricTolerance=2.0e-12) {
    const auto fail=[&](const char* message){error=message;return false;};
    if(parentTets.empty()||childTets.empty()||parentMaterials.size()!=parentTets.size()||
       childMaterials.size()!=childTets.size()||childParentRows.size()!=childTets.size()||
       nodeParentTetRows.size()!=nodeParentBarycentric.size()||
       !std::isfinite(barycentricTolerance)||barycentricTolerance<0.0)
        return fail("single-parent refinement ancestry arrays are incomplete");
    std::vector<std::uint32_t> nextOffsets(childTets.size()+1u,0u),nextParents(childTets.size());
    std::vector<double> nextMeasures(childTets.size());
    for(std::size_t c=0;c<childTets.size();++c){
        const auto p=childParentRows[c];if(p>=parentTets.size()||childMaterials[c]!=parentMaterials[p])return fail("single-parent child has invalid or cross-material parent");
        std::array<std::array<double,4>,4> bary{};
        for(unsigned k=0;k<4u;++k){
            const auto node=childTets[c][k];if(node>=nodeParentTetRows.size())return fail("child node is outside the ancestry table");
            const auto at=std::find(parentTets[p].begin(),parentTets[p].end(),node);
            if(at!=parentTets[p].end())bary[k][static_cast<std::size_t>(at-parentTets[p].begin())]=1.0;
            else {
                const auto sourceParent=nodeParentTetRows[node];
                if(sourceParent>=parentTets.size())return fail("child crosses parent tets or lacks source-parent node ancestry");
                const auto sourceBary=nodeParentBarycentric[node];
                double sourceSum=0.0;
                std::array<double,3> pullbackPoint{};
                for(unsigned q=0;q<4u;++q){
                    if(!std::isfinite(sourceBary[q])||sourceBary[q]< -barycentricTolerance||
                       sourceBary[q]>1.0+barycentricTolerance)
                        return fail("node source-parent barycentric coordinate is outside its parent tet");
                    sourceSum+=sourceBary[q];
                    const auto sourceNode=parentTets[sourceParent][q];
                    if(sourceNode>=parentReferencePoints.size())return fail("source-parent node ancestry references an absent point");
                    for(unsigned axis=0;axis<3u;++axis)
                        pullbackPoint[axis]+=sourceBary[q]*parentReferencePoints[sourceNode][axis];
                }
                if(std::abs(sourceSum-1.0)>barycentricTolerance)
                    return fail("node source-parent barycentric coordinates do not sum to one");
                const auto target=parentTets[p];
                for(auto sourceNode:target)if(sourceNode>=parentReferencePoints.size())
                    return fail("target parent tetrahedron references an absent reference point");
                const auto& x0=parentReferencePoints[target[0]];
                const auto& x1=parentReferencePoints[target[1]];
                const auto& x2=parentReferencePoints[target[2]];
                const auto& x3=parentReferencePoints[target[3]];
                const auto sub=[](const auto& a,const auto& b){return std::array<double,3>{a[0]-b[0],a[1]-b[1],a[2]-b[2]};};
                const auto a=sub(x1,x0),b=sub(x2,x0),d=sub(x3,x0);
                const std::array<double,3> r{pullbackPoint[0]-x0[0],pullbackPoint[1]-x0[1],pullbackPoint[2]-x0[2]};
                const auto det=[](const auto& u,const auto& v,const auto& w){
                    return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]);};
                const double determinant=det(a,b,d);
                if(!std::isfinite(determinant)||determinant==0.0)
                    return fail("target parent tetrahedron is singular while mapping a shared split node");
                const double l1=det(r,b,d)/determinant,l2=det(a,r,d)/determinant,l3=det(a,b,r)/determinant;
                bary[k]={1.0-l1-l2-l3,l1,l2,l3};
            }
            double sum=0.0;
            for(double x:bary[k]){if(!std::isfinite(x)||x< -barycentricTolerance||x>1.0+barycentricTolerance)return fail("child pullback vertex lies outside its single parent tetrahedron");sum+=x;}
            if(std::abs(sum-1.0)>barycentricTolerance)return fail("child pullback barycentric coordinates do not sum to one");
        }
        const auto d=[&](unsigned r,unsigned col){return bary[r+1u][col+1u]-bary[0][col+1u];};
        const double det=d(0,0)*(d(1,1)*d(2,2)-d(1,2)*d(2,1))-
                         d(0,1)*(d(1,0)*d(2,2)-d(1,2)*d(2,0))+
                         d(0,2)*(d(1,0)*d(2,1)-d(1,1)*d(2,0));
        const double parentVolume=parentReferenceTetVolumeM3(parentTets[p],parentReferencePoints);
        const double measure=std::abs(det)*parentVolume;
        if(!std::isfinite(parentVolume)||parentVolume<=0.0||!std::isfinite(measure)||measure<=0.0)
            return fail("single-parent child has nonpositive pullback measure");
        nextParents[c]=p;nextMeasures[c]=measure;nextOffsets[c+1u]=static_cast<std::uint32_t>(c+1u);
    }
    offsets=std::move(nextOffsets);parents=std::move(nextParents);measures=std::move(nextMeasures);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
