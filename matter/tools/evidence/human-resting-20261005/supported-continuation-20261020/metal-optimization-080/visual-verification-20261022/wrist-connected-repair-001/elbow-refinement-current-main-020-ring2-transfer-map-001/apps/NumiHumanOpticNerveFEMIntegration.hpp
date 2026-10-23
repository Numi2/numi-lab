#pragma once
// App-facing passive optic-nerve FEM admission over the existing registered
// TetGen adapter. The asset is geometry-only; this adapter creates no neural
// conduction, blood state, or standalone Runtime.
#include "NumiHumanOpticNerveTetGenAssets.hpp"
#include "NumiHumanTetGenOrganicFEM.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace numi::human::optic_nerve {

struct Built {
    metalrobo::human::coarse_tissue::PreparedOrganicFEM fem;
    metalrobo::human::coarse_tissue::PreparedTetrahedralAncestry ancestry;
    std::uint32_t objectIndex=NM_INVALID_INDEX;
    std::vector<std::uint32_t> sparseReferenceAnchorNodes;
};

inline constexpr bool sameAssetStorage(std::span<const tetgen_assets::Point> a,
                                       std::span<const tetgen_assets::Point> b) noexcept {
    return a.data()==b.data()&&a.size()==b.size();
}
inline constexpr bool sameAssetStorage(std::span<const std::uint32_t> a,
                                       std::span<const std::uint32_t> b) noexcept {
    return a.data()==b.data()&&a.size()==b.size();
}
inline constexpr bool sameAssetStorage(std::span<const tetgen_assets::Tet> a,
                                       std::span<const tetgen_assets::Tet> b) noexcept {
    return a.data()==b.data()&&a.size()==b.size();
}
inline constexpr bool sameAssetStorage(std::span<const tetgen_assets::Face> a,
                                       std::span<const tetgen_assets::Face> b) noexcept {
    return a.data()==b.data()&&a.size()==b.size();
}
inline constexpr bool exactPinnedAssetView(const tetgen_assets::AssetView& candidate,
                                           const tetgen_assets::AssetView& pinned) noexcept {
    return candidate.member==pinned.member&&
           candidate.anatomicalConcept==pinned.anatomicalConcept&&
           candidate.meshSHA256==pinned.meshSHA256&&
           candidate.payloadSHA256==pinned.payloadSHA256&&
           candidate.sourceVertexCount==pinned.sourceVertexCount&&
           candidate.sourceFaceCount==pinned.sourceFaceCount&&
           candidate.sourceVolumeM3==pinned.sourceVolumeM3&&
           sameAssetStorage(candidate.donorLocalPoints,pinned.donorLocalPoints)&&
           sameAssetStorage(candidate.sourceVertex,pinned.sourceVertex)&&
           sameAssetStorage(candidate.tetrahedra,pinned.tetrahedra)&&
           sameAssetStorage(candidate.boundaryFaces,pinned.boundaryFaces)&&
           sameAssetStorage(candidate.boundarySourceFace,pinned.boundarySourceFace);
}
inline constexpr bool exactPinnedOpticNerveAsset(const tetgen_assets::AssetView& candidate) noexcept {
    return exactPinnedAssetView(candidate,tetgen_assets::kLeftAsset)||
           exactPinnedAssetView(candidate,tetgen_assets::kRightAsset);
}
constexpr auto kSwappedLeftOpticNerveLateralityForRegression=[] {
    auto swapped=tetgen_assets::kLeftAsset;
    swapped.anatomicalConcept=tetgen_assets::kRightAsset.anatomicalConcept;
    return swapped;
}();
static_assert(!exactPinnedOpticNerveAsset(kSwappedLeftOpticNerveLateralityForRegression),
              "optic-nerve pinned asset guard must reject swapped laterality");

inline std::array<double,3> rotateUnitQuaternion(
        const std::array<double,4>& q,const std::array<double,3>& p) {
    const std::array<double,3> qv{q[0],q[1],q[2]};
    const auto cross=[](const auto& a,const auto& b) {
        return std::array<double,3>{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    };
    const auto add=[](const auto& a,const auto& b) { return std::array<double,3>{a[0]+b[0],a[1]+b[1],a[2]+b[2]}; };
    const auto scale=[](const auto& a,double s) { return std::array<double,3>{a[0]*s,a[1]*s,a[2]*s}; };
    const auto qCrossP=cross(qv,p);
    return add(p,add(scale(qCrossP,2.0*q[3]),scale(cross(qv,qCrossP),2.0)));
}

inline std::vector<std::uint32_t> chooseSparseReferenceAnchors(
        const tetgen_assets::AssetView& asset,std::size_t count=3u) {
    std::vector<std::uint32_t> eligible;
    for(std::uint32_t i=0;i<asset.donorLocalPoints.size();++i)
        if(asset.sourceVertex[i]!=NM_INVALID_INDEX)eligible.push_back(i);
    if(eligible.size()<count)return {};
    std::sort(eligible.begin(),eligible.end(),[&](auto a,auto b){
        return asset.sourceVertex[a]==asset.sourceVertex[b]?a<b:asset.sourceVertex[a]<asset.sourceVertex[b];});
    std::vector<std::uint32_t> chosen{eligible.front()};
    const auto distanceSquared=[&](std::uint32_t a,std::uint32_t b) {
        const auto& x=asset.donorLocalPoints[a];const auto& y=asset.donorLocalPoints[b];
        const double dx=x[0]-y[0],dy=x[1]-y[1],dz=x[2]-y[2];return dx*dx+dy*dy+dz*dz;
    };
    while(chosen.size()<count) {
        double best=-1.0;std::uint32_t selected=NM_INVALID_INDEX;
        for(const auto candidate:eligible) {
            if(std::find(chosen.begin(),chosen.end(),candidate)!=chosen.end())continue;
            double separation=std::numeric_limits<double>::infinity();
            for(const auto prior:chosen)separation=std::min(separation,distanceSquared(candidate,prior));
            if(separation>best||(separation==best&&selected!=NM_INVALID_INDEX&&asset.sourceVertex[candidate]<asset.sourceVertex[selected])){best=separation;selected=candidate;}
        }
        if(selected==NM_INVALID_INDEX||!std::isfinite(best)||best<=1.0e-16)return {};
        chosen.push_back(selected);
    }
    return chosen;
}

inline bool appendRegisteredOpticNerve(
        numi::matter::WorldSource& world,const tetgen_assets::AssetView& asset,
        const std::array<double,3>& bodyPosition,const std::array<double,4>& bodyOrientation,
        std::uint32_t runtimeDonorBody,std::uint32_t sourceLayer,
        std::uint32_t materialIndex,std::string materialName,std::string materialProvenance,
        double densityKgM3,double availableRigidDonorMassKg,
        const std::array<std::uint64_t,4>& materialSourceIdentity,
        const std::array<std::uint64_t,4>& referenceSourceIdentity,
        std::string objectName,Built& out,std::string& error) {
    using namespace metalrobo::human::coarse_tissue;
    auto fail=[&](const char* why){error=why;return false;};
    if(!exactPinnedOpticNerveAsset(asset)||
       asset.payloadSHA256!=tetgen_assets::kActiveNHAPayloadSHA256||
       asset.sourceVertexCount==0u||asset.sourceFaceCount==0u||
       asset.donorLocalPoints.size()<4u||asset.tetrahedra.empty()||asset.boundaryFaces.size()<4u||
       asset.sourceVertex.size()!=asset.donorLocalPoints.size()||
       asset.boundarySourceFace.size()!=asset.boundaryFaces.size())
        return fail("optic nerve member, laterality, mesh hash, source counts, or generated arrays differ from the exact pinned bilateral assets");
    if(runtimeDonorBody==NM_INVALID_INDEX||sourceLayer==NM_INVALID_INDEX||
       materialIndex>=world.materials.size()||materialName.empty()||materialProvenance.empty()||objectName.empty())
        return fail("optic nerve material, runtime donor, source layer, or object identity is incomplete");
    if(!std::isfinite(densityKgM3)||densityKgM3<=0.0||
       !std::isfinite(availableRigidDonorMassKg)||availableRigidDonorMassKg<=0.0)
        return fail("optic nerve proxy density and available rigid donor mass must be finite and positive");
    double qn=0.0;for(double x:bodyOrientation){if(!std::isfinite(x))return fail("optic nerve donor orientation is non-finite");qn+=x*x;}
    qn=std::sqrt(qn);if(!std::isfinite(qn)||qn<=1.0e-12)return fail("optic nerve donor orientation is degenerate");
    std::array<double,4> q{};for(unsigned k=0;k<4;k++)q[k]=bodyOrientation[k]/qn;
    for(double x:bodyPosition)if(!std::isfinite(x))return fail("optic nerve donor position is non-finite");
    for(const auto& old:world.objects)if(old.name==objectName)return fail("optic nerve FEM object name is already present in this world");
    const auto anchors=chooseSparseReferenceAnchors(asset);
    if(anchors.size()!=3u)return fail("optic nerve source mesh lacks three distinct source-backed sparse anchors");

    RegisteredTetrahedralVolume source;
    source.name=asset.member=="FJ1772"?"left optic nerve":"right optic nerve";
    source.provider="BodyParts3D";source.hierarchy="is_a";source.sourceMember=std::string(asset.member);
    source.anatomicalConceptIdentifier=std::string(asset.anatomicalConcept);
    source.sourceMeshSHA256=std::string(asset.meshSHA256);source.sourcePayloadSHA256=std::string(asset.payloadSHA256);
    source.geometryProvenance="BodyParts3D OBJ exact-coordinate seam weld; TetGen 1.6.0 PLC -pYq1.5a1e-9; source marker-preserving boundary; generated arrays bound by NumiHumanOpticNerveTetGenAssets.hpp";
    source.intersectionAuditProvenance="TetGen 1.6.0 -pd closed-PLC intersection check; exact log SHA pinned in NumiHumanOpticNerveTetGenAssets.hpp";
    source.referenceConfigurationProvenance="registered current scene pose used as passive mechanical reference; no unloaded-state claim";
    source.frameTransformProvenance="BodyParts3D global_source_mm_to_myosim_world_m registration followed by inverse NHRIGID2 source body 23 rest pose; registration, NHA payload, and NHRIGID2 hashes pinned in NumiHumanOpticNerveTetGenAssets.hpp; body 23 remapped through the active Human source-body map";
    source.stableIdentifier=0u;source.sourceBodyIndex=runtimeDonorBody;source.sourceLayer=sourceLayer;
    source.sourceVertexCount=asset.sourceVertexCount;source.sourceFaceCount=asset.sourceFaceCount;
    source.materialSourceIdentity=materialSourceIdentity;source.referenceSourceIdentity=referenceSourceIdentity;
    source.worldPoints.reserve(asset.donorLocalPoints.size());source.referenceWorldPoints.reserve(asset.donorLocalPoints.size());
    source.donorLocalPoints.assign(asset.donorLocalPoints.begin(),asset.donorLocalPoints.end());
    source.nodeSourceVertex.assign(asset.sourceVertex.begin(),asset.sourceVertex.end());
    source.donorBodyIndices.assign(asset.donorLocalPoints.size(),runtimeDonorBody);
    for(const auto& local:asset.donorLocalPoints) {
        const auto rotated=rotateUnitQuaternion(q,local);
        const V3 current{bodyPosition[0]+rotated[0],bodyPosition[1]+rotated[1],bodyPosition[2]+rotated[2]};
        if(!std::all_of(current.begin(),current.end(),[](double x){return std::isfinite(x);}))return fail("optic nerve current world point is non-finite");
        source.worldPoints.push_back(current);source.referenceWorldPoints.push_back(current);
    }
    source.tetrahedra.assign(asset.tetrahedra.begin(),asset.tetrahedra.end());
    source.boundaryFaces.assign(asset.boundaryFaces.begin(),asset.boundaryFaces.end());
    source.boundarySourceFaceIndices.assign(asset.boundarySourceFace.begin(),asset.boundarySourceFace.end());
    source.materials.push_back({materialIndex,std::move(materialName),std::move(materialProvenance)});
    double referenceVolume=0.0;
    for(const auto& t:source.tetrahedra) {
        for(auto node:t)if(node>=source.referenceWorldPoints.size())return fail("optic nerve tet references an absent source node");
        referenceVolume+=tetVolume(source.referenceWorldPoints[t[0]],source.referenceWorldPoints[t[1]],source.referenceWorldPoints[t[2]],source.referenceWorldPoints[t[3]]);
    }
    if(!std::isfinite(referenceVolume)||referenceVolume<=0.0||
       std::abs(referenceVolume-asset.sourceVolumeM3)>1.0e-6*asset.sourceVolumeM3)
        return fail("optic nerve registered TetGen volume differs from its pinned source receipt");
    const double donorDebitKg=densityKgM3*referenceVolume;
    if(!std::isfinite(donorDebitKg)||donorDebitKg<=0.0||donorDebitKg>=availableRigidDonorMassKg)
        return fail("inferred optic nerve proxy mass does not fit within the mapped rigid donor mass");

    PreparedOrganicFEM prepared;PreparedTetrahedralAncestry ancestry;std::string appendError;
    const std::string objectNameCopy=objectName;
    if(!appendRegisteredTetrahedralFEM(world,source,source.name,source.sourceMember,
            source.anatomicalConceptIdentifier,0u,sourceLayer,std::move(objectName),donorDebitKg,
            prepared,ancestry,appendError)) {
        error="registered optic nerve TetGen FEM rejected: "+appendError;return false;
    }
    if(world.objects.empty()||world.objects.back().name!=objectNameCopy) return fail("optic nerve append did not return its requested Matter object");
    auto& object=world.objects.back();
    if(object.femNodes.size()!=source.worldPoints.size()||object.femReferenceNodes.size()!=object.femNodes.size())
        return fail("optic nerve Matter object lost its registered current/reference node pairing");
    object.femReferenceDisplacementGradient=true;
    out.fem=std::move(prepared);out.ancestry=std::move(ancestry);
    out.objectIndex=static_cast<std::uint32_t>(world.objects.size()-1u);out.sparseReferenceAnchorNodes=anchors;
    error.clear();return true;
}

} // namespace numi::human::optic_nerve
