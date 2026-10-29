#pragma once

#include "NumiHumanConservativeSpatialMassTransport.hpp"

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
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

// Conservative transfer from an authoritative parent-tetrahedron mass ledger.
// This validates algebraic coverage/density bounds; it does not prove that a
// caller-supplied overlap measure is geometrically correct. The independent,
// pinned overlap or reference-pullback receipt remains a separate requirement.

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



// A source-preserving operational reference update is a different path from
// centroid-neighborhood transport. This proof accepts only a bijection of the
// original node and tet domains with exact material/donor/region/source-face
// labels. The supplied external surface-audit hashes and pair counts must come
// from receipts already read and matched to source/candidate asset hashes by
// the owning loader; this function binds those pins but does not parse receipts.
struct SameTopologyReferencePullbackInput {
    std::span<const std::array<double,3>> sourceReferencePoints;
    std::span<const std::array<double,3>> candidateReferencePoints;
    // Candidate row -> original source row; must be a bijection.
    std::span<const std::uint32_t> candidateToSourceNodeRows;
    std::span<const std::array<std::uint32_t,4>> sourceTetrahedra;
    std::span<const std::array<std::uint32_t,4>> candidateTetrahedra;
    // Candidate cell row -> original source cell row; must be a bijection.
    std::span<const std::uint32_t> candidateToSourceTetRows;
    std::span<const std::uint32_t> sourceMaterialIds;
    std::span<const std::uint32_t> candidateMaterialIds;
    std::span<const std::uint32_t> sourceGeometricDonorIds;
    std::span<const std::uint32_t> candidateGeometricDonorIds;
    std::span<const std::uint32_t> sourceRegionIds;
    std::span<const std::uint32_t> candidateRegionIds;
    std::span<const std::uint32_t> sourceStableFaceIds;
    std::span<const std::uint32_t> candidateStableFaceIds;
    std::span<const double> sourceCellMassKg;
    std::span<const double> materialDensityLowerKgM3;
    std::span<const double> materialDensityUpperKgM3;
    std::string_view sourceVolumeSHA256;
    std::string_view candidateVolumeSHA256;
    std::string_view parentMassPacketSHA256;
    std::string_view parentSourceContributionSHA256;
    std::string_view sourceBoundarySelfAuditSHA256;
    std::string_view candidateBoundarySelfAuditSHA256;
    std::size_t sourceBoundarySelfIntersectionPairs=0u;
    std::size_t candidateBoundarySelfIntersectionPairs=0u;
};

struct SameTopologyReferencePullbackResult {
    // Candidate-to-source geometry correspondence. Each child has one source
    // cell and childAncestryMeasureVolumeM3 is that source geometry's cell
    // volume. This result is not directly interchangeable with a separate
    // immutable mass parent's reference measures; the owning loader must bind
    // the source-to-mass-parent cell map and use the authoritative parent
    // measure when populating ExplicitUnpairedVolumeTopology.
    std::vector<std::uint32_t> childParentAncestryOffsets;
    std::vector<std::uint32_t> childParentAncestryTetrahedra;
    std::vector<double> childParentAncestryVolumeFractions;
    std::vector<double> childAncestryMeasureVolumeM3;
    std::vector<double> sourceReferenceVolumeM3;
    std::vector<double> candidateReferenceVolumeM3;
    std::vector<double> childJacobianDeterminants;
    std::vector<double> childMassKg;
    std::vector<double> childImpliedDensityKgM3;
    double maximumNodeDisplacementM=0.0;
    double maximumTetCentroidDisplacementM=0.0;
    double minimumJacobianDeterminant=0.0;
    double maximumJacobianDeterminant=0.0;
    std::string verificationSHA256;
};

[[nodiscard]] inline double signedReferenceTetVolumeM3(
        const std::array<std::uint32_t,4>& tet,
        const std::span<const std::array<double,3>> points) {
    if(std::any_of(tet.begin(),tet.end(),[&](const auto node){return node>=points.size();}))
        return std::numeric_limits<double>::quiet_NaN();
    const auto sub=[](const auto& a,const auto& b){
        return std::array<double,3>{a[0]-b[0],a[1]-b[1],a[2]-b[2]};
    };
    const auto a=sub(points[tet[1]],points[tet[0]]);
    const auto b=sub(points[tet[2]],points[tet[0]]);
    const auto c=sub(points[tet[3]],points[tet[0]]);
    const double determinant=
        a[0]*(b[1]*c[2]-b[2]*c[1])-
        a[1]*(b[0]*c[2]-b[2]*c[0])+
        a[2]*(b[0]*c[1]-b[1]*c[0]);
    return determinant/6.0;
}

// Verify an exact, oriented, piecewise-affine reference map and prepare its
// single-parent pullback CSR. Every original tet is assigned exactly once;
// mass stays with that original cell and candidate density is checked against
// the actual material bounds. No centroid-distance threshold is used here.
[[nodiscard]] inline bool verifySameTopologyReferencePullback(
        const SameTopologyReferencePullbackInput& in,
        SameTopologyReferencePullbackResult& out,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const std::size_t sourceNodeCount=in.sourceReferencePoints.size();
    const std::size_t candidateNodeCount=in.candidateReferencePoints.size();
    const std::size_t sourceTetCount=in.sourceTetrahedra.size();
    const std::size_t candidateTetCount=in.candidateTetrahedra.size();
    if(sourceNodeCount<4u||candidateNodeCount!=sourceNodeCount||
       in.candidateToSourceNodeRows.size()!=candidateNodeCount||
       sourceTetCount==0u||candidateTetCount!=sourceTetCount||
       in.candidateToSourceTetRows.size()!=candidateTetCount||
       in.sourceMaterialIds.size()!=sourceTetCount||
       in.candidateMaterialIds.size()!=candidateTetCount||
       in.sourceGeometricDonorIds.size()!=sourceTetCount||
       in.candidateGeometricDonorIds.size()!=candidateTetCount||
       in.sourceRegionIds.size()!=sourceTetCount||
       in.candidateRegionIds.size()!=candidateTetCount||
       in.sourceStableFaceIds.size()!=sourceTetCount||
       in.candidateStableFaceIds.size()!=candidateTetCount||
       in.sourceCellMassKg.size()!=sourceTetCount||
       in.materialDensityLowerKgM3.empty()||
       in.materialDensityUpperKgM3.size()!=in.materialDensityLowerKgM3.size())
        return fail("reference pullback is missing a complete same-topology node, cell, label, mass, or density table");
    if(!spatial_mass_transport_detail::validSHA256(in.sourceVolumeSHA256)||!spatial_mass_transport_detail::validSHA256(in.candidateVolumeSHA256)||
       !spatial_mass_transport_detail::validSHA256(in.parentMassPacketSHA256)||
       !spatial_mass_transport_detail::validSHA256(in.parentSourceContributionSHA256)||
       !spatial_mass_transport_detail::validSHA256(in.sourceBoundarySelfAuditSHA256)||
       !spatial_mass_transport_detail::validSHA256(in.candidateBoundarySelfAuditSHA256))
        return fail("reference pullback requires pinned source, candidate, mass, contribution, and boundary-audit hashes");
    if(in.sourceBoundarySelfIntersectionPairs!=0u||
       in.candidateBoundarySelfIntersectionPairs!=0u)
        return fail("reference pullback requires zero source and candidate boundary self-intersection pairs");

    std::vector<std::uint8_t> sourceNodeSeen(sourceNodeCount,0u);
    std::vector<std::uint8_t> sourceNodeUsed(sourceNodeCount,0u);
    double maximumNodeDisplacement=0.0;
    for(std::size_t node=0;node<candidateNodeCount;++node){
        const auto parent=in.candidateToSourceNodeRows[node];
        if(parent>=sourceNodeCount||sourceNodeSeen[parent]!=0u)
            return fail("reference pullback candidate node correspondence is out of range or not one-to-one");
        sourceNodeSeen[parent]=1u;
        double distanceSquared=0.0;
        for(unsigned axis=0;axis<3u;++axis){
            const double source=in.sourceReferencePoints[parent][axis];
            const double candidate=in.candidateReferencePoints[node][axis];
            if(!std::isfinite(source)||!std::isfinite(candidate))
                return fail("reference pullback contains a nonfinite source or candidate node coordinate");
            const double delta=candidate-source;
            distanceSquared+=delta*delta;
        }
        const double distance=std::sqrt(distanceSquared);
        if(!std::isfinite(distance))
            return fail("reference pullback node displacement is nonfinite");
        maximumNodeDisplacement=std::max(maximumNodeDisplacement,distance);
    }
    if(std::any_of(sourceNodeSeen.begin(),sourceNodeSeen.end(),[](const auto seen){return seen!=1u;}))
        return fail("reference pullback candidate does not cover every original source node exactly once");

    std::vector<std::uint8_t> sourceTetSeen(sourceTetCount,0u);
    std::set<std::array<std::uint32_t,4>> sourceTetKeys,candidateTetKeys;
    SameTopologyReferencePullbackResult next;
    next.childParentAncestryOffsets.resize(candidateTetCount+1u,0u);
    next.childParentAncestryTetrahedra.resize(candidateTetCount);
    next.childParentAncestryVolumeFractions.assign(candidateTetCount,1.0);
    next.childAncestryMeasureVolumeM3.resize(candidateTetCount);
    next.sourceReferenceVolumeM3.resize(sourceTetCount);
    next.candidateReferenceVolumeM3.resize(candidateTetCount);
    next.childJacobianDeterminants.resize(candidateTetCount);
    next.childMassKg.resize(candidateTetCount);
    next.childImpliedDensityKgM3.resize(candidateTetCount);
    next.minimumJacobianDeterminant=std::numeric_limits<double>::infinity();
    next.maximumJacobianDeterminant=0.0;
    using GroupKey=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    std::map<GroupKey,long double> sourceMassByGroup,candidateMassByGroup;

    const auto evenPermutation=[](const std::array<unsigned,4>& permutation){
        unsigned inversions=0u;
        for(unsigned i=0;i<4u;++i)for(unsigned j=i+1u;j<4u;++j)
            inversions+=permutation[i]>permutation[j]?1u:0u;
        return (inversions&1u)==0u;
    };
    for(std::size_t source=0;source<sourceTetCount;++source){
        const auto tet=in.sourceTetrahedra[source];
        auto key=tet;std::sort(key.begin(),key.end());
        if(key[0]==key[1]||key[1]==key[2]||key[2]==key[3]||
           key[3]>=sourceNodeCount||!sourceTetKeys.insert(key).second)
            return fail("reference pullback source tet table contains a duplicate or invalid cell");
        const double volume=signedReferenceTetVolumeM3(tet,in.sourceReferencePoints);
        if(!std::isfinite(volume)||volume<=0.0)
            return fail("reference pullback source has a nonpositive oriented reference cell at row "+std::to_string(source));
        next.sourceReferenceVolumeM3[source]=volume;
    }

    for(std::size_t child=0;child<candidateTetCount;++child){
        const auto source=in.candidateToSourceTetRows[child];
        if(source>=sourceTetCount||sourceTetSeen[source]!=0u)
            return fail("reference pullback candidate cell correspondence is out of range or not one-to-one");
        sourceTetSeen[source]=1u;
        const auto childTet=in.candidateTetrahedra[child];
        auto childKey=childTet;std::sort(childKey.begin(),childKey.end());
        if(childKey[0]==childKey[1]||childKey[1]==childKey[2]||childKey[2]==childKey[3]||
           childKey[3]>=candidateNodeCount||!candidateTetKeys.insert(childKey).second)
            return fail("reference pullback candidate tet table contains a duplicate or invalid cell");

        const auto sourceTet=in.sourceTetrahedra[source];
        std::array<unsigned,4> permutation{};
        std::array<std::uint32_t,4> mappedNodes{};
        for(unsigned corner=0;corner<4u;++corner){
            const auto candidateNode=childTet[corner];
            const auto mapped=in.candidateToSourceNodeRows[candidateNode];
            mappedNodes[corner]=mapped;
            sourceNodeUsed[mapped]=1u;
            const auto position=std::find(sourceTet.begin(),sourceTet.end(),mapped);
            if(position==sourceTet.end())
                return fail("reference pullback candidate cell is not the mapped source cell at row "+std::to_string(child));
            permutation[corner]=static_cast<unsigned>(position-sourceTet.begin());
        }
        auto mappedKey=mappedNodes;std::sort(mappedKey.begin(),mappedKey.end());
        auto sourceCellKey=sourceTet;std::sort(sourceCellKey.begin(),sourceCellKey.end());
        if(mappedKey!=sourceCellKey||!evenPermutation(permutation))
            return fail("reference pullback candidate tet is not the same oriented source simplex at row "+std::to_string(child));

        const double sourceVolume=next.sourceReferenceVolumeM3[source];
        const double candidateVolume=signedReferenceTetVolumeM3(childTet,in.candidateReferencePoints);
        if(!std::isfinite(candidateVolume)||candidateVolume<=0.0)
            return fail("reference pullback candidate has a nonpositive oriented reference cell at row "+std::to_string(child));
        const double jacobian=candidateVolume/sourceVolume;
        if(!std::isfinite(jacobian)||jacobian<=0.0)
            return fail("reference pullback cell Jacobian is nonpositive or nonfinite");
        const auto material=in.sourceMaterialIds[source];
        const auto donor=in.sourceGeometricDonorIds[source];
        const auto region=in.sourceRegionIds[source];
        if(material>=in.materialDensityLowerKgM3.size()||
           donor==invalid||region==invalid||in.sourceStableFaceIds[source]==invalid||
           in.candidateMaterialIds[child]!=material||
           in.candidateGeometricDonorIds[child]!=donor||
           in.candidateRegionIds[child]!=region||
           in.candidateStableFaceIds[child]!=in.sourceStableFaceIds[source])
            return fail("reference pullback changes material, geometric donor, region, or stable source-face identity at child row "+std::to_string(child));
        const double lower=in.materialDensityLowerKgM3[material];
        const double upper=in.materialDensityUpperKgM3[material];
        const double mass=in.sourceCellMassKg[source];
        if(!std::isfinite(lower)||!std::isfinite(upper)||lower<=0.0||lower>upper||
           !std::isfinite(mass)||mass<=0.0)
            return fail("reference pullback source mass or actual material density bounds are invalid");
        const double sourceDensity=mass/sourceVolume;
        const double candidateDensity=mass/candidateVolume;
        if(!std::isfinite(sourceDensity)||sourceDensity<lower||sourceDensity>upper)
            return fail("reference pullback source cell mass is outside its actual material density bounds at row "+std::to_string(source));
        if(!std::isfinite(candidateDensity)||candidateDensity<lower||candidateDensity>upper)
            return fail("reference pullback preserved cell mass implies density outside actual bounds at child row "+std::to_string(child));

        double centroidDisplacementSquared=0.0;
        for(unsigned axis=0;axis<3u;++axis){
            double sourceCentroid=0.0,candidateCentroid=0.0;
            for(unsigned corner=0;corner<4u;++corner){
                sourceCentroid+=in.sourceReferencePoints[sourceTet[corner]][axis]*0.25;
                candidateCentroid+=in.candidateReferencePoints[childTet[corner]][axis]*0.25;
            }
            const double delta=candidateCentroid-sourceCentroid;
            centroidDisplacementSquared+=delta*delta;
        }
        next.maximumTetCentroidDisplacementM=std::max(next.maximumTetCentroidDisplacementM,
            std::sqrt(centroidDisplacementSquared));
        next.childParentAncestryOffsets[child]=static_cast<std::uint32_t>(child);
        next.childParentAncestryTetrahedra[child]=static_cast<std::uint32_t>(source);
        next.childAncestryMeasureVolumeM3[child]=sourceVolume;
        next.sourceReferenceVolumeM3[source]=sourceVolume;
        next.candidateReferenceVolumeM3[child]=candidateVolume;
        next.childJacobianDeterminants[child]=jacobian;
        next.childMassKg[child]=mass;
        next.childImpliedDensityKgM3[child]=candidateDensity;
        next.minimumJacobianDeterminant=std::min(next.minimumJacobianDeterminant,jacobian);
        next.maximumJacobianDeterminant=std::max(next.maximumJacobianDeterminant,jacobian);
        sourceMassByGroup[{material,donor,region}]+=mass;
        candidateMassByGroup[{material,donor,region}]+=mass;
    }
    next.childParentAncestryOffsets[candidateTetCount]=
        static_cast<std::uint32_t>(candidateTetCount);
    if(std::any_of(sourceTetSeen.begin(),sourceTetSeen.end(),[](const auto seen){return seen!=1u;}))
        return fail("reference pullback candidate does not cover every original source cell exactly once");
    if(std::any_of(sourceNodeUsed.begin(),sourceNodeUsed.end(),[](const auto used){return used!=1u;}))
        return fail("reference pullback source or candidate volume has an unused node");
    if(sourceMassByGroup.size()!=candidateMassByGroup.size())
        return fail("reference pullback does not preserve per-material/donor/region cell mass groups");
    for(const auto& [key,sourceMass]:sourceMassByGroup){
        const auto it=candidateMassByGroup.find(key);
        if(it==candidateMassByGroup.end()||
           std::abs(sourceMass-it->second)>
               64.0L*std::numeric_limits<double>::epsilon()*
                   std::max({std::abs(sourceMass),std::abs(it->second),1.0e-12L}))
            return fail("reference pullback does not preserve per-material/donor/region cell mass totals");
    }

    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("NumiHumanSameTopologyReferencePullback/v1");
    hash.string(in.sourceVolumeSHA256);hash.string(in.candidateVolumeSHA256);
    hash.string(in.parentMassPacketSHA256);hash.string(in.parentSourceContributionSHA256);
    hash.string(in.sourceBoundarySelfAuditSHA256);hash.string(in.candidateBoundarySelfAuditSHA256);
    hash.u64(in.sourceBoundarySelfIntersectionPairs);
    hash.u64(in.candidateBoundarySelfIntersectionPairs);
    hash.u64(sourceNodeCount);hash.u64(sourceTetCount);
    for(const auto& point:in.sourceReferencePoints)for(const auto value:point)hash.real(value);
    for(const auto& point:in.candidateReferencePoints)for(const auto value:point)hash.real(value);
    for(const auto row:in.candidateToSourceNodeRows)hash.u32(row);
    for(const auto& tet:in.sourceTetrahedra)for(const auto node:tet)hash.u32(node);
    for(const auto& tet:in.candidateTetrahedra)for(const auto node:tet)hash.u32(node);
    for(const auto row:in.candidateToSourceTetRows)hash.u32(row);
    for(std::size_t row=0;row<sourceTetCount;++row){
        hash.u32(in.sourceMaterialIds[row]);hash.u32(in.sourceGeometricDonorIds[row]);
        hash.u32(in.sourceRegionIds[row]);hash.u32(in.sourceStableFaceIds[row]);
        hash.real(in.sourceCellMassKg[row]);
    }
    for(std::size_t row=0;row<candidateTetCount;++row){
        hash.u32(in.candidateMaterialIds[row]);hash.u32(in.candidateGeometricDonorIds[row]);
        hash.u32(in.candidateRegionIds[row]);hash.u32(in.candidateStableFaceIds[row]);
        hash.real(next.sourceReferenceVolumeM3[next.childParentAncestryTetrahedra[row]]);
        hash.real(next.candidateReferenceVolumeM3[row]);
        hash.real(next.childJacobianDeterminants[row]);
        hash.real(next.childMassKg[row]);hash.real(next.childImpliedDensityKgM3[row]);
    }
    for(const auto value:in.materialDensityLowerKgM3)hash.real(value);
    for(const auto value:in.materialDensityUpperKgM3)hash.real(value);
    next.maximumNodeDisplacementM=maximumNodeDisplacement;
    next.verificationSHA256=hash.finish();
    out=std::move(next);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
