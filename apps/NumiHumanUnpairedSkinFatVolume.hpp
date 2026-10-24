#pragma once

// Opt-in source preparation for a conforming skin/fat volume mesh whose
// tetrahedral node set is independent of its registered O/I/D boundary rows.
// The legacy paired-layer builder remains unchanged.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanSkinBoundaryNodeMap.hpp"

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
#include <vector>

namespace metalrobo::human::coarse_tissue {

enum class VolumeAncestryVerificationKind : std::uint8_t {
    unverified=0, exactGeometricOverlap=1, conservativeReferencePullback=2
};

// All node/tet indices use the source-volume domain. This API preserves that
// order when constructing ObjectSource; surface rows remain separate in the
// SkinBoundaryNodeMap. CSR fractions are caller-supplied ancestry bookkeeping;
// they establish neither geometric overlap nor conservative pullback by themselves.
struct ExplicitUnpairedVolumeTopology {
    // Optional independent I/D registered boundaries. Both must be present
    // together; null keeps the legacy matched O/I/D face topology. The outer
    // boundary remains the RegisteredSurface supplied to the compiler.
    const RegisteredSurface* independentSkinInnerSurface=nullptr;
    const RegisteredSurface* independentFatDeepSurface=nullptr;
    std::span<const V3> currentWorldPoints;
    std::span<const V3> referenceWorldPoints;
    std::span<const std::array<std::uint32_t,4>> childTetrahedra;
    std::span<const std::uint32_t> childMaterialIndices;
    std::span<const std::uint32_t> childDominantParentTetrahedra;
    std::span<const std::uint32_t> childParentSurfaceFaces;
    std::span<const std::uint32_t> childStableSourceFaceIdentities;
    std::span<const std::uint32_t> childDonorBodyIndices;

    std::span<const V3> parentReferenceWorldPoints;
    std::span<const std::array<std::uint32_t,4>> parentTetrahedra;
    std::span<const std::uint32_t> parentMaterialIndices;
    std::span<const std::uint32_t> parentStableSourceFaceIdentities;
    std::span<const std::uint32_t> parentDonorBodyIndices;
    // CSR entry fractions sum to one per child and weighted child volume closes
    // each parent. The declared method requires its own independent verification receipt.
    std::span<const std::uint32_t> childParentAncestryOffsets;
    std::span<const std::uint32_t> childParentAncestryTetrahedra;
    std::span<const double> childParentAncestryVolumeFractions;
    // Exact current RegisteredSurface row/stable ID for each tet facet, flattened
    // child*4+omitted-corner. Unmatched interior facets use INVALID in both.
    std::span<const std::uint32_t> childRegisteredLayerFacetSurfaceFaces;
    std::span<const std::uint32_t> childRegisteredLayerFacetSourceFaceIdentities;
    // A non-registered exterior facet is allowed only when construction explicitly
    // identifies it as an authored head sidewall and supplies its stable wall-face ID.
    std::span<const std::uint8_t> childAuthoredHeadSidewallFacetFlags;
    std::span<const std::uint32_t> childAuthoredHeadSidewallStableFaceIdentities;

    std::span<const std::uint32_t> volumeNodeDonorBodyIndices;
    std::span<const V3> volumeNodeDonorLocalPoints;
    std::span<const std::array<double,9>> volumeNodeDonorLocalToWorldLinear;
    std::span<const V3> volumeNodeDonorOriginWorld;
    // Exact current registered donor-frame table, indexed by source body ID.
    std::span<const std::array<double,9>> registeredBodyLocalToWorldLinear;
    std::span<const V3> registeredBodyOriginWorld;
    std::size_t detailedSourceVertexCount=0;
    std::size_t detailedSourceFaceCount=0;
    std::span<const std::uint32_t> volumeNodeSourceVertexIndices;
    // Detailed-skin ancestry is distinct from parent-volume ancestry. Direct
    // source vertices use INVALID face IDs; projected rows provide face+bary.
    std::span<const std::uint32_t> volumeNodeSourceFaceIndices;
    std::span<const std::array<double,3>> volumeNodeSourceBarycentric;
    std::size_t retainedOriginalNodeCount=0;
    // Appended nodes (including new boundary rows) point into the original
    // reference tets; an explicit residual preserves a measured offset from the
    // barycentric source point. Old retained rows use INVALID parent IDs.
    std::span<const std::uint32_t> volumeNodeParentTetrahedra;
    std::span<const std::array<double,4>> volumeNodeParentBarycentric;
    std::span<const V3> volumeNodeParentResidualWorldM;
    VolumeAncestryVerificationKind ancestryVerificationKind=VolumeAncestryVerificationKind::unverified;
    // SHA-256 of an external exact-overlap or conservative reference-pullback receipt.
    std::string_view ancestryVerificationReceiptSHA256;
    std::array<std::uint64_t,4> referenceSourceIdentity{};
    std::array<std::uint64_t,4> materialSourceIdentity{};
};

struct ExplicitUnpairedVolumeReport {
    std::vector<std::uint32_t> inputVolumeRowToObjectLocalNode;
    std::uint32_t volumeNodeCount=0;
    std::uint32_t tetrahedronCount=0;
    double skinVolumeM3=0.0, fatVolumeM3=0.0;
    double skinDensityKgM3=0.0, fatDensityKgM3=0.0;
    VolumeAncestryVerificationKind ancestryVerificationKind=VolumeAncestryVerificationKind::unverified;
    std::string ancestryVerificationReceiptSHA256;
};

[[nodiscard]] inline bool appendRegisteredSkinFatUnpairedVolume(
        numi::matter::WorldSource& world,
        const RegisteredSurface& surface,
        const numi::human::SkinBoundaryNodeMap& boundaryMap,
        const ExplicitUnpairedVolumeTopology& topology,
        const std::uint32_t skinMaterialIndex,
        const std::string_view skinMaterialName,
        const std::string_view skinMaterialProvenance,
        const double skinDonorDebitKg,
        const std::uint32_t fatMaterialIndex,
        const std::string_view fatMaterialName,
        const std::string_view fatMaterialProvenance,
        const double fatDonorDebitKg,
        const std::size_t sourceBodyCount,
        const std::string_view objectName,
        PreparedOrganicFEM& preparedSkin,
        PreparedOrganicFEM& preparedFat,
        ExplicitUnpairedVolumeReport& report,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    constexpr std::size_t kMaxUnpairedTets=200000u;
    const auto invalid=NM_INVALID_INDEX;
    const std::size_t n=topology.currentWorldPoints.size();
    const std::size_t nt=topology.childTetrahedra.size();
    if(n<4u||n>std::numeric_limits<std::uint32_t>::max()||nt==0u||nt>kMaxUnpairedTets||
       topology.volumeNodeDonorBodyIndices.size()!=n||topology.volumeNodeDonorLocalPoints.size()!=n||
       topology.volumeNodeDonorLocalToWorldLinear.size()!=n||topology.volumeNodeDonorOriginWorld.size()!=n||
       topology.registeredBodyLocalToWorldLinear.size()!=sourceBodyCount||topology.registeredBodyOriginWorld.size()!=sourceBodyCount||
       (!topology.volumeNodeSourceVertexIndices.empty()&&topology.volumeNodeSourceVertexIndices.size()!=n)||
       (!topology.volumeNodeSourceFaceIndices.empty()&&topology.volumeNodeSourceFaceIndices.size()!=n)||
       (!topology.volumeNodeSourceBarycentric.empty()&&topology.volumeNodeSourceBarycentric.size()!=n)||
       (topology.volumeNodeSourceFaceIndices.empty()!=topology.volumeNodeSourceBarycentric.empty())||
       topology.detailedSourceVertexCount==0u||topology.detailedSourceFaceCount==0u||
       topology.childMaterialIndices.size()!=nt||topology.childDominantParentTetrahedra.size()!=nt||
       topology.childParentSurfaceFaces.size()!=nt||topology.childStableSourceFaceIdentities.size()!=nt||
       topology.childDonorBodyIndices.size()!=nt||surface.worldPoints.empty()||surface.faces.empty()||
       surface.donorBodyIndices.size()!=surface.worldPoints.size()||
       surface.sourceVertexIndices.size()!=surface.worldPoints.size()||
       (!surface.sourceFaceIndices.empty()&&surface.sourceFaceIndices.size()!=surface.faces.size())||
       surface.donorLocalPoints.size()!=surface.worldPoints.size()||
       skinMaterialIndex==fatMaterialIndex||skinMaterialIndex>=world.materials.size()||
       fatMaterialIndex>=world.materials.size()||sourceBodyCount==0u||objectName.empty()||
       skinMaterialName.empty()||fatMaterialName.empty()||skinMaterialProvenance.empty()||
       fatMaterialProvenance.empty()||!std::isfinite(skinDonorDebitKg)||skinDonorDebitKg<=0.0||
       !std::isfinite(fatDonorDebitKg)||fatDonorDebitKg<=0.0)
        return fail("unpaired skin/fat volume arrays, materials, donor data, or budgets are incomplete");
    const bool independent=topology.independentSkinInnerSurface!=nullptr;
    if(independent!=(topology.independentFatDeepSurface!=nullptr))
        return fail("unpaired skin/fat independent I/D surfaces must be supplied together");
    const std::array<const RegisteredSurface*,3> roleSurfaces{
        &surface, independent?topology.independentSkinInnerSurface:&surface,
        independent?topology.independentFatDeepSurface:&surface};
    const std::array<std::size_t,3> roleSizes{
        roleSurfaces[0]->worldPoints.size(),roleSurfaces[1]->worldPoints.size(),
        roleSurfaces[2]->worldPoints.size()};
    if(!numi::human::validateSkinBoundaryNodeMap(boundaryMap,roleSizes,error))return false;
    for(const auto* layer:roleSurfaces)
        if(layer->faces.empty()||layer->donorBodyIndices.size()!=layer->worldPoints.size()||
           layer->donorLocalPoints.size()!=layer->worldPoints.size()||
           (!layer->sourceFaceIndices.empty()&&layer->sourceFaceIndices.size()!=layer->faces.size()))
            return fail("unpaired skin/fat registered layer geometry or identity arrays are incomplete");
    if(independent)for(const auto* layer:roleSurfaces){
        if(layer->sourceFaceIndices.size()!=layer->faces.size()||
           std::any_of(layer->sourceFaceIndices.begin(),layer->sourceFaceIndices.end(),
               [&](auto id){return id==invalid;}))
            return fail("independent registered layer lacks exact source-face ancestry identities");
    }
    if(independent)for(unsigned role=1u;role<3u;++role)
        if(!validateRegisteredSourceEmbedding(*roleSurfaces[role],topology.detailedSourceFaceCount,error))
            return false;
    if(boundaryMap.volumeNodeCount!=n)
        return fail("unpaired skin/fat boundary map volume count differs from the exact source node table");
    const auto receipt=topology.ancestryVerificationReceiptSHA256;
    if((topology.ancestryVerificationKind!=VolumeAncestryVerificationKind::exactGeometricOverlap&&
        topology.ancestryVerificationKind!=VolumeAncestryVerificationKind::conservativeReferencePullback)||
       receipt.size()!=64u||!std::all_of(receipt.begin(),receipt.end(),[](char c){
            return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
        return fail("unpaired skin/fat ancestry lacks a pinned exact-overlap or conservative-pullback verification receipt");
    if(topology.referenceWorldPoints.size()!=n||topology.parentReferenceWorldPoints.empty()||
       topology.parentTetrahedra.empty()||topology.parentMaterialIndices.size()!=topology.parentTetrahedra.size()||
       topology.parentStableSourceFaceIdentities.size()!=topology.parentTetrahedra.size()||
       topology.parentDonorBodyIndices.size()!=topology.parentTetrahedra.size())
        return fail("unpaired skin/fat reference or parent source topology is incomplete");
    if(topology.retainedOriginalNodeCount>n||
       topology.volumeNodeParentTetrahedra.size()!=n||
       topology.volumeNodeParentBarycentric.size()!=n||
       topology.volumeNodeParentResidualWorldM.size()!=n)
        return fail("unpaired skin/fat source-node parent ancestry does not cover the exact volume row domain");
    if(topology.childParentAncestryOffsets.size()!=nt+1u||
       topology.childParentAncestryOffsets.front()!=0u||
       topology.childParentAncestryOffsets.back()!=topology.childParentAncestryTetrahedra.size()||
       topology.childParentAncestryTetrahedra.size()!=topology.childParentAncestryVolumeFractions.size())
        return fail("unpaired skin/fat volume-parent CSR does not cover every child tetrahedron");
    if(topology.childRegisteredLayerFacetSurfaceFaces.size()!=4u*nt||
       topology.childRegisteredLayerFacetSourceFaceIdentities.size()!=4u*nt||
       topology.childAuthoredHeadSidewallFacetFlags.size()!=4u*nt||
       topology.childAuthoredHeadSidewallStableFaceIdentities.size()!=4u*nt)
        return fail("unpaired skin/fat registered boundary-facet CSR does not cover all child faces");
    if(world.objects.size()>std::numeric_limits<std::uint32_t>::max())
        return fail("unpaired skin/fat source object index exceeds the existing source format");
    std::uint64_t firstGlobal=0u;
    for(const auto& object:world.objects){
        if(object.femNodes.size()>std::numeric_limits<std::uint32_t>::max()-firstGlobal)
            return fail("existing FEM source node index overflow before unpaired skin/fat append");
        firstGlobal+=object.femNodes.size();
    }
    if(firstGlobal+n>std::numeric_limits<std::uint32_t>::max())
        return fail("unpaired skin/fat FEM node range exceeds the existing source format");
    const auto finitePoint=[](const V3& p){return std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);});};
    const auto& current=topology.currentWorldPoints;
    const auto& reference=topology.referenceWorldPoints;
    // Validate every parent index before any source-parent reconstruction below.
    for(std::size_t parent=0;parent<topology.parentTetrahedra.size();++parent){
        for(auto node:topology.parentTetrahedra[parent])
            if(node>=topology.parentReferenceWorldPoints.size())
                return fail("unpaired source parent tetrahedron references an absent reference point");
        const auto tet=topology.parentTetrahedra[parent];
        const auto volume=tetVolume(topology.parentReferenceWorldPoints[tet[0]],topology.parentReferenceWorldPoints[tet[1]],topology.parentReferenceWorldPoints[tet[2]],topology.parentReferenceWorldPoints[tet[3]]);
        if(!std::isfinite(volume)||volume<=0.0)
            return fail("unpaired source parent tetrahedron is nonpositive or nonfinite");
    }
    for(std::size_t i=0;i<n;++i){
        if(!finitePoint(current[i])||!finitePoint(reference[i])||
           !finitePoint(topology.volumeNodeDonorLocalPoints[i])||!finitePoint(topology.volumeNodeDonorOriginWorld[i])||
           topology.volumeNodeDonorBodyIndices[i]==invalid||topology.volumeNodeDonorBodyIndices[i]>=sourceBodyCount)
            return fail("unpaired skin/fat volume node has nonfinite geometry/frame or invalid donor identity at row "+std::to_string(i));
        const auto donor=topology.volumeNodeDonorBodyIndices[i];
        const auto& rotation=topology.volumeNodeDonorLocalToWorldLinear[i];
        const auto& registeredRotation=topology.registeredBodyLocalToWorldLinear[donor];
        const auto& registeredOrigin=topology.registeredBodyOriginWorld[donor];
        if(!finitePoint(registeredOrigin)||topology.volumeNodeDonorOriginWorld[i]!=registeredOrigin||
           rotation!=registeredRotation)
            return fail("unpaired volume node transform differs from the exact registered donor-frame table at row "+std::to_string(i));
        if(!std::all_of(rotation.begin(),rotation.end(),[](double x){return std::isfinite(x);}))
            return fail("unpaired volume node donor transform is nonfinite at row "+std::to_string(i));
        double determinant=rotation[0]*(rotation[4]*rotation[8]-rotation[5]*rotation[7])-
            rotation[1]*(rotation[3]*rotation[8]-rotation[5]*rotation[6])+
            rotation[2]*(rotation[3]*rotation[7]-rotation[4]*rotation[6]);
        if(!std::isfinite(determinant)||std::abs(determinant-1.0)>1.0e-6)
            return fail("unpaired volume node donor transform is not a proper rigid rotation at row "+std::to_string(i));
        for(unsigned a=0;a<3u;++a)for(unsigned b=0;b<3u;++b){
            double dot=0.0;for(unsigned k=0;k<3u;++k)dot+=rotation[3u*k+a]*rotation[3u*k+b];
            if(std::abs(dot-(a==b?1.0:0.0))>1.0e-6)
                return fail("unpaired volume node donor transform is not orthonormal at row "+std::to_string(i));
        }
        V3 transformed=topology.volumeNodeDonorOriginWorld[i];
        for(unsigned a=0;a<3u;++a)for(unsigned b=0;b<3u;++b)
            transformed[a]+=rotation[3u*a+b]*topology.volumeNodeDonorLocalPoints[i][b];
        if(norm(sub(transformed,current[i]))>1.0e-7)
            return fail("unpaired volume node donor-local point does not reproduce its current world coordinate at row "+std::to_string(i));
        if(!topology.volumeNodeSourceVertexIndices.empty()&&topology.volumeNodeSourceVertexIndices[i]!=invalid&&
           topology.volumeNodeSourceVertexIndices[i]>=topology.detailedSourceVertexCount)
            return fail("unpaired volume node direct detailed-source identity is out of range at row "+std::to_string(i));
        if(!topology.volumeNodeSourceFaceIndices.empty()&&topology.volumeNodeSourceFaceIndices[i]!=invalid){
            if(topology.volumeNodeSourceFaceIndices[i]>=topology.detailedSourceFaceCount)
                return fail("unpaired volume node detailed-source face identity is out of range at row "+std::to_string(i));
            std::string baryError;if(!validateBarycentricCoordinates(topology.volumeNodeSourceBarycentric[i],baryError))
                return fail("unpaired volume node detailed-source embedding is invalid at row "+std::to_string(i));
        }
        if(i<topology.retainedOriginalNodeCount){
            if(topology.volumeNodeParentTetrahedra[i]!=invalid)
                return fail("retained unpaired volume row unexpectedly claims appended-node ancestry");
        }else{
            const auto parent=topology.volumeNodeParentTetrahedra[i];
            if(parent>=topology.parentTetrahedra.size())
                return fail("appended unpaired volume row lacks an exact reference-parent tetrahedron");
            const auto& bary=topology.volumeNodeParentBarycentric[i];
            const auto ids=topology.parentTetrahedra[parent];
            V3 reconstructed{};double sum=0.0;
            for(unsigned k=0;k<4u;++k){
                if(!std::isfinite(bary[k])||bary[k]<-1.0e-9||bary[k]>1.0+1.0e-9)
                    return fail("appended unpaired volume row has invalid parent barycentric coordinates");
                sum+=bary[k];
                for(unsigned a=0;a<3u;++a)reconstructed[a]+=bary[k]*topology.parentReferenceWorldPoints[ids[k]][a];
            }
            const auto residual=topology.volumeNodeParentResidualWorldM[i];
            if(!finitePoint(residual)||!std::isfinite(sum)||std::abs(sum-1.0)>1.0e-8)
                return fail("appended unpaired volume row has invalid parent residual or barycentric sum");
            reconstructed=add(reconstructed,residual);
            if(norm(sub(reconstructed,reference[i]))>1.0e-8)
                return fail("appended unpaired volume row does not reconstruct its registered reference coordinate");
        }
    }
    std::vector<std::uint32_t> volumeNodeLayer(n,invalid),volumeNodeSurfaceRow(n,invalid);
    const std::array<const std::vector<std::uint32_t>*,3> mappedRoles{
        &boundaryMap.outer,&boundaryMap.skinInner,&boundaryMap.fatDeep};
    std::array<std::vector<std::uint32_t>,3> roleNodeByFace;
    for(unsigned role=0;role<3u;++role)roleNodeByFace[role].assign(roleSizes[role],0u);
    for(std::uint32_t role=0u;role<3u;++role){
        const auto& mapping=*mappedRoles[role];
        const auto& layer=*roleSurfaces[role];
        for(std::size_t row=0u;row<mapping.size();++row){
            const auto node=mapping[row];
            if(node>=n||row>=layer.worldPoints.size())
                return fail("unpaired skin/fat boundary map contains an out-of-range volume or surface row");
            if(volumeNodeLayer[node]!=invalid)
                return fail("unpaired skin/fat boundary map aliases a volume node across surface roles");
            volumeNodeLayer[node]=role;volumeNodeSurfaceRow[node]=static_cast<std::uint32_t>(row);
            if(topology.volumeNodeDonorBodyIndices[node]!=layer.donorBodyIndices[row])
                return fail("unpaired skin/fat mapped boundary donor differs from RegisteredSurface at row "+std::to_string(row));
            if(independent&&role!=0u&&
               (norm(sub(current[node],layer.worldPoints[row]))>1.0e-8||
                norm(sub(topology.volumeNodeDonorLocalPoints[node],layer.donorLocalPoints[row]))>1.0e-7))
                return fail("unpaired skin/fat independent layer row differs from its exact current geometry or donor-local point");
            if(independent&&role!=0u){
                if(!layer.sourceVertexIndices.empty()&&
                   layer.sourceVertexIndices[row]!=invalid&&
                   (topology.volumeNodeSourceVertexIndices.size()!=n||
                    topology.volumeNodeSourceVertexIndices[node]!=layer.sourceVertexIndices[row]))
                    return fail("unpaired independent layer direct source identity differs from its volume node");
                if(!layer.sourceEmbeddingFaceIndices.empty()&&
                   layer.sourceEmbeddingFaceIndices[row]!=invalid&&
                   (topology.volumeNodeSourceFaceIndices.size()!=n||
                    topology.volumeNodeSourceFaceIndices[node]!=layer.sourceEmbeddingFaceIndices[row]||
                    topology.volumeNodeSourceBarycentric[node]!=layer.sourceEmbeddingBarycentric[row]))
                    return fail("unpaired independent layer source embedding differs from its volume node");
            }
            if(role==0u){
                if(topology.volumeNodeSourceVertexIndices.size()==n&&
                   topology.volumeNodeSourceVertexIndices[node]!=surface.sourceVertexIndices[row])
                    return fail("unpaired skin/fat outer direct source identity differs from RegisteredSurface at row "+std::to_string(row));
                if(norm(sub(current[node],surface.worldPoints[row]))>1.0e-8)
                    return fail("unpaired skin/fat outer boundary differs from registered current geometry at surface row "+std::to_string(row));
                if(norm(sub(topology.volumeNodeDonorLocalPoints[node],surface.donorLocalPoints[row]))>1.0e-7)
                    return fail("unpaired skin/fat outer donor-local point differs from RegisteredSurface at row "+std::to_string(row));
            }
            ++roleNodeByFace[role][row];
        }
    }
    for(const auto& counts:roleNodeByFace)
        if(std::any_of(counts.begin(),counts.end(),[](auto count){return count!=1u;}))
            return fail("unpaired skin/fat surface-row map is not one-to-one within a boundary layer");

    std::array<std::map<Face,std::uint32_t>,3> registeredFaceByKey;
    for(unsigned role=0;role<3u;++role){
        const auto& layer=*roleSurfaces[role];
        for(std::uint32_t row=0;row<layer.faces.size();++row){
            auto key=layer.faces[row];std::sort(key.begin(),key.end());
            if(key[0]>=layer.worldPoints.size()||key[1]>=layer.worldPoints.size()||key[2]>=layer.worldPoints.size()||
               key[0]==key[1]||key[1]==key[2]||!registeredFaceByKey[role].emplace(key,row).second)
                return fail("unpaired skin/fat RegisteredSurface layer contains an invalid or duplicate face");
        }
    }
    struct FacetUse { std::uint32_t child=0, omitted=0, material=0; int orientation=0; };
    std::map<Face,std::vector<FacetUse>> facetUses;
    std::array<std::vector<std::uint32_t>,3> layerFaceCounts;
    for(unsigned role=0;role<3u;++role)layerFaceCounts[role].assign(roleSurfaces[role]->faces.size(),0u);
    std::vector<std::array<std::uint32_t,2>> interfaceMaterialCounts(roleSurfaces[1]->faces.size());
    for(std::size_t child=0;child<nt;++child){
        const auto material=topology.childMaterialIndices[child];
        const auto parent=topology.childDominantParentTetrahedra[child];
        const auto stable=topology.childStableSourceFaceIdentities[child];
        const auto donor=topology.childDonorBodyIndices[child];
        if((material!=skinMaterialIndex&&material!=fatMaterialIndex)||
           parent>=topology.parentTetrahedra.size()||donor==invalid||donor>=sourceBodyCount||stable==invalid||
           topology.parentMaterialIndices[parent]!=material||
           topology.parentDonorBodyIndices[parent]!=donor||
           topology.parentStableSourceFaceIdentities[parent]==invalid||
           (topology.childParentSurfaceFaces[child]!=invalid&&topology.childParentSurfaceFaces[child]>=surface.faces.size()))
            return fail("unpaired skin/fat child has inconsistent material, dominant donor, parent, or source-face identity at tet "+std::to_string(child));
        auto ids=topology.childTetrahedra[child];auto sorted=ids;std::sort(sorted.begin(),sorted.end());
        if(sorted[0]==sorted[1]||sorted[1]==sorted[2]||sorted[2]==sorted[3]||sorted[3]>=n)
            return fail("unpaired skin/fat child has repeated or out-of-range node IDs at tet "+std::to_string(child));
        for(unsigned omitted=0;omitted<4u;++omitted){
            constexpr std::array<std::array<unsigned,3>,4> outwardCorners{{{{1,2,3}},{{0,3,2}},{{0,1,3}},{{0,2,1}}}};
            std::array<std::uint32_t,3> oriented{};
            for(unsigned k=0;k<3u;++k)oriented[k]=ids[outwardCorners[omitted][k]];
            auto key=oriented;std::sort(key.begin(),key.end());
            int orientation=1;for(unsigned a=0;a<3u;++a)for(unsigned b=a+1u;b<3u;++b)if(oriented[a]>oriented[b])orientation=-orientation;
            facetUses[key].push_back(FacetUse{static_cast<std::uint32_t>(child),omitted,material,orientation});
            std::array<std::uint32_t,3> nodes{};unsigned at=0;
            for(unsigned corner=0;corner<4u;++corner)if(corner!=omitted)nodes[at++]=ids[corner];
            const auto declared=topology.childRegisteredLayerFacetSurfaceFaces[4u*child+omitted];
            const auto declaredStable=topology.childRegisteredLayerFacetSourceFaceIdentities[4u*child+omitted];
            const auto wallFlag=topology.childAuthoredHeadSidewallFacetFlags[4u*child+omitted];
            if(wallFlag>1u)return fail("unpaired skin/fat authored-sidewall facet flag is not boolean");
            const auto role=volumeNodeLayer[nodes[0]];
            bool sameRole=role!=invalid;
            for(auto node:nodes)if(volumeNodeLayer[node]!=role)sameRole=false;
            std::uint32_t exact=invalid;
            if(sameRole){
                Face key{};for(unsigned k=0;k<3u;++k)key[k]=volumeNodeSurfaceRow[nodes[k]];
                std::sort(key.begin(),key.end());const auto found=registeredFaceByKey[role].find(key);
                if(found!=registeredFaceByKey[role].end())exact=found->second;
            }
            if(exact==invalid){
                if(declared!=invalid||declaredStable!=invalid)
                    return fail("unpaired skin/fat child facet claims a registered boundary face that its mapped nodes do not form");
                continue;
            }
            const auto expectedStable=roleSurfaces[role]->sourceFaceIndices.empty()?exact:roleSurfaces[role]->sourceFaceIndices[exact];
            if(declared!=exact||declaredStable!=expectedStable)
                return fail("unpaired skin/fat child facet differs from the exact registered layer face row/stable identity");
            ++layerFaceCounts[role][exact];
            if(role==1u)++interfaceMaterialCounts[exact][material==skinMaterialIndex?0u:1u];
            else if((role==0u&&material!=skinMaterialIndex)||(role==2u&&material!=fatMaterialIndex))
                return fail("unpaired skin/fat boundary facet is assigned to the wrong tissue material");
        }
    }
    // Every volume face must have valid manifold incidence and an explicit owner
    // if it lies on the external boundary. This closes gaps that O/I/D-only counts miss.
    for(const auto& [key,uses]:facetUses){
        if(uses.empty()||uses.size()>2u)
            return fail("unpaired skin/fat tetrahedra have a nonmanifold facet incidence");
        const auto nodes=key;
        const auto role=volumeNodeLayer[nodes[0]];
        bool sameRole=role!=invalid;for(auto node:nodes)if(volumeNodeLayer[node]!=role)sameRole=false;
        std::uint32_t exact=invalid;
        if(sameRole){Face rowKey{};for(unsigned k=0;k<3u;++k)rowKey[k]=volumeNodeSurfaceRow[nodes[k]];
            std::sort(rowKey.begin(),rowKey.end());const auto found=registeredFaceByKey[role].find(rowKey);
            if(found!=registeredFaceByKey[role].end())exact=found->second;}
        if(uses.size()==2u){
            if(uses[0].orientation+uses[1].orientation!=0)
                return fail("unpaired skin/fat interior facet orientations do not oppose");
            if(exact!=invalid&&role!=1u)
                return fail("unpaired skin/fat registered O/D face is internal to the volume mesh");
            if(uses[0].material!=uses[1].material&&!(exact!=invalid&&role==1u))
                return fail("unpaired skin/fat material interface is not an exact registered I-layer face");
            for(const auto& use:uses){
                const auto slot=4u*use.child+use.omitted;
                if(topology.childAuthoredHeadSidewallFacetFlags[slot]!=0u||
                   topology.childAuthoredHeadSidewallStableFaceIdentities[slot]!=invalid)
                    return fail("unpaired skin/fat internal facet is incorrectly tagged as an authored head sidewall");
            }
        }else{
            const auto& use=uses.front();const auto slot=4u*use.child+use.omitted;
            const auto wall=topology.childAuthoredHeadSidewallFacetFlags[slot];
            const auto wallID=topology.childAuthoredHeadSidewallStableFaceIdentities[slot];
            const auto declared=topology.childRegisteredLayerFacetSurfaceFaces[slot];
            const auto declaredStable=topology.childRegisteredLayerFacetSourceFaceIdentities[slot];
            if(exact!=invalid&&role!=1u){
                const auto expected=roleSurfaces[role]->sourceFaceIndices.empty()?exact:roleSurfaces[role]->sourceFaceIndices[exact];
                if(declared!=exact||declaredStable!=expected||wall!=0u||wallID!=invalid)
                    return fail("unpaired skin/fat external O/D facet lacks exact registered identity or conflicts with sidewall metadata");
            }else{
                if(exact!=invalid&&role==1u)
                    return fail("unpaired skin/fat registered I-layer face is exposed as an external boundary");
                if(declared!=invalid||declaredStable!=invalid||wall!=1u||wallID==invalid)
                    return fail("unpaired skin/fat external boundary is neither O/D registered surface nor explicitly authored head sidewall");
            }
        }
    }
    for(unsigned role=0;role<3u;++role)for(std::size_t face=0;face<layerFaceCounts[role].size();++face){
        if(layerFaceCounts[role][face]!=(role==1u?2u:1u)||
           (role==1u&&(interfaceMaterialCounts[face][0]!=1u||interfaceMaterialCounts[face][1]!=1u)))
            return fail("unpaired skin/fat child topology does not close every independent registered layer face once per material side");
    }

    // Independently audit the three mapped closed boundaries at both current
    // and stress-free reference coordinates; the volume mesh is not assumed paired.
    std::array<RegisteredSurface,3> mappedSurfaces;
    for(std::uint32_t role=0;role<3u;++role){
        auto& mapped=mappedSurfaces[role];mapped=*roleSurfaces[role];mapped.name=roleSurfaces[role]->name+" unpaired layer boundary";
        mapped.worldPoints.clear();mapped.worldPoints.reserve(roleSizes[role]);
        for(auto node:*mappedRoles[role])mapped.worldPoints.push_back(current[node]);
        GeometryAudit audit;
        if(!auditClosedSurface(mapped,audit,error,true))return false;
        mapped.worldPoints.clear();mapped.worldPoints.reserve(roleSizes[role]);
        for(auto node:*mappedRoles[role])mapped.worldPoints.push_back(reference[node]);
        if(!auditClosedSurface(mapped,audit,error,true))return false;
    }

    const bool hasAncestry=!topology.childParentAncestryOffsets.empty()||
        !topology.childParentAncestryTetrahedra.empty()||!topology.childParentAncestryVolumeFractions.empty();
    if(!hasAncestry)return fail("unpaired skin/fat volume mesh requires complete parent-volume ancestry CSR");
    for(std::size_t i=1;i<topology.childParentAncestryOffsets.size();++i)
        if(topology.childParentAncestryOffsets[i]<topology.childParentAncestryOffsets[i-1u]||
           topology.childParentAncestryOffsets[i]>topology.childParentAncestryTetrahedra.size())
            return fail("unpaired skin/fat parent-volume CSR offsets are not monotone");
    std::set<std::array<std::uint32_t,4>> uniqueChildren;
    std::vector<double> parentCovered(topology.parentTetrahedra.size(),0.0);
    std::vector<double> parentVolume(topology.parentTetrahedra.size(),0.0);
    for(std::size_t p=0;p<topology.parentTetrahedra.size();++p){
        const auto tet=topology.parentTetrahedra[p];
        for(auto node:tet)if(node>=topology.parentReferenceWorldPoints.size())
            return fail("unpaired skin/fat parent tetrahedron references an absent reference node");
        parentVolume[p]=tetVolume(topology.parentReferenceWorldPoints[tet[0]],topology.parentReferenceWorldPoints[tet[1]],topology.parentReferenceWorldPoints[tet[2]],topology.parentReferenceWorldPoints[tet[3]]);
        if(!std::isfinite(parentVolume[p])||parentVolume[p]<=0.0||
           (topology.parentMaterialIndices[p]!=skinMaterialIndex&&topology.parentMaterialIndices[p]!=fatMaterialIndex)||
           topology.parentDonorBodyIndices[p]==invalid||topology.parentDonorBodyIndices[p]>=sourceBodyCount)
            return fail("unpaired skin/fat parent tetrahedron has invalid material, volume, or donor identity");
    }
    double skinVolume=0.0,fatVolume=0.0;
    std::vector<double> nodalMass(n,0.0),skinNodeMass(n,0.0),fatNodeMass(n,0.0);
    std::vector<std::uint32_t> usedNodes(n,0u);
    std::vector<double> childVolumes(nt,0.0);
    for(std::size_t child=0;child<nt;++child){
        const auto ids=topology.childTetrahedra[child];auto canonical=ids;std::sort(canonical.begin(),canonical.end());
        if(!uniqueChildren.insert(canonical).second)return fail("unpaired skin/fat mesh repeats a tetrahedron");
        const double v=tetVolume(reference[ids[0]],reference[ids[1]],reference[ids[2]],reference[ids[3]]);
        const double currentV=tetVolume(current[ids[0]],current[ids[1]],current[ids[2]],current[ids[3]]);
        if(!std::isfinite(v)||v<=0.0||!std::isfinite(currentV)||currentV<=0.0)
            return fail("unpaired skin/fat child tetrahedron has nonpositive current or reference volume");
        auto emitted=std::array<V3,4>{};for(unsigned k=0;k<4u;++k)for(unsigned a=0;a<3u;++a)
            emitted[k][a]=static_cast<double>(static_cast<float>(reference[ids[k]][a]));
        double longest=0.0;for(unsigned i=0;i<4u;++i)for(unsigned j=i+1u;j<4u;++j)longest=std::max(longest,norm(sub(emitted[i],emitted[j])));
        const double normalizedDet=6.0*tetVolume(emitted[0],emitted[1],emitted[2],emitted[3])/(longest*longest*longest);
        if(!std::isfinite(normalizedDet)||normalizedDet<1.0e-5)
            return fail("unpaired skin/fat child loses positive Float32 reference quality at tet "+std::to_string(child));
        auto emittedCurrent=std::array<V3,4>{};for(unsigned k=0;k<4u;++k)for(unsigned a=0;a<3u;++a)
            emittedCurrent[k][a]=static_cast<double>(static_cast<float>(current[ids[k]][a]));
        const double emittedCurrentVolume=tetVolume(emittedCurrent[0],emittedCurrent[1],emittedCurrent[2],emittedCurrent[3]);
        if(!std::isfinite(emittedCurrentVolume)||emittedCurrentVolume<=0.0)
            return fail("unpaired skin/fat child loses positive emitted-Float32 current orientation at tet "+std::to_string(child));
        const double contribution=topology.childMaterialIndices[child]==skinMaterialIndex?skinVolume+=v:fatVolume+=v;
        (void)contribution;childVolumes[child]=v;
        for(auto node:ids)++usedNodes[node];
        if(topology.childParentAncestryOffsets[child]>=topology.childParentAncestryOffsets[child+1u])
            return fail("unpaired skin/fat child has an empty parent-volume CSR row");
        double sum=0.0;std::uint32_t prior=0u;
        for(std::uint32_t e=topology.childParentAncestryOffsets[child];e<topology.childParentAncestryOffsets[child+1u];++e){
            const auto parent=topology.childParentAncestryTetrahedra[e];const double f=topology.childParentAncestryVolumeFractions[e];
            if(parent>=parentVolume.size()||(e>topology.childParentAncestryOffsets[child]&&parent<=prior)||
               !std::isfinite(f)||f<=0.0||topology.parentMaterialIndices[parent]!=topology.childMaterialIndices[child])
                return fail("unpaired skin/fat child parent CSR has invalid, duplicate, or cross-material ancestry");
            prior=parent;sum+=f;parentCovered[parent]+=v*f;
        }
        if(!std::isfinite(sum)||std::abs(sum-1.0)>1.0e-6)
            return fail("unpaired skin/fat child parent-volume fractions do not sum to one");
    }
    if(std::any_of(usedNodes.begin(),usedNodes.end(),[](auto x){return x==0u;}))
        return fail("unpaired skin/fat volume node table contains an unused source row");
    for(std::size_t p=0;p<parentVolume.size();++p)
        if(std::abs(parentCovered[p]-parentVolume[p])>1.0e-6*parentVolume[p])
            return fail("unpaired skin/fat CSR volume closure differs from original parent tet "+std::to_string(p));

    auto densityFor=[&](std::uint32_t index,double debit,double volume,double& density)->bool{
        auto& material=world.materials[index];
        if(material.name.empty()||std::find(material.supportedRepresentations.begin(),material.supportedRepresentations.end(),numi::matter::Representation::fem)==material.supportedRepresentations.end())return false;
        auto it=std::find_if(material.parameters.begin(),material.parameters.end(),[](const auto& p){return p.name=="density";});
        if(it==material.parameters.end()||it->identifiable||!std::isfinite(it->lower)||!std::isfinite(it->upper)||it->lower>it->upper)return false;
        density=debit/volume;return std::isfinite(density)&&density>=it->lower&&density<=it->upper;
    };
    double rhoSkin=0.0,rhoFat=0.0;
    if(world.materials[skinMaterialIndex].name!=skinMaterialName||world.materials[fatMaterialIndex].name!=fatMaterialName||
       !(skinVolume>0.0&&fatVolume>0.0)||!densityFor(skinMaterialIndex,skinDonorDebitKg,skinVolume,rhoSkin)||
       !densityFor(fatMaterialIndex,fatDonorDebitKg,fatVolume,rhoFat))
        return fail("unpaired skin/fat geometric volume implies an invalid or out-of-bounds existing material density");
    for(std::size_t child=0;child<nt;++child){
        const auto rho=topology.childMaterialIndices[child]==skinMaterialIndex?rhoSkin:rhoFat;
        const auto share=rho*childVolumes[child]*0.25;
        for(auto node:topology.childTetrahedra[child]){
            nodalMass[node]+=share;
            if(topology.childMaterialIndices[child]==skinMaterialIndex)skinNodeMass[node]+=share;else fatNodeMass[node]+=share;
        }
    }
    numi::matter::ObjectSource object;object.name=std::string(objectName);object.materialIndex=skinMaterialIndex;
    object.representation=numi::matter::Representation::fem;object.twoWayCoupling=true;object.adaptive=false;
    // Only the separately validated independent-role mesh uses the canonical FEM self-contact path.
    // The legacy shared-row O/I/D topology retains its paired material-column owner.
    object.mixedFEM=false;object.deformableContact=true;object.deformableSelfContact=independent;
    object.characteristicLength=std::cbrt(skinVolume+fatVolume);object.femNodes.assign(current.begin(),current.end());
    object.femReferenceNodes.assign(reference.begin(),reference.end());
    object.femReferenceSourceIdentity=topology.referenceSourceIdentity;
    object.femMaterialSourceIdentity=topology.materialSourceIdentity;
    const auto identityPresent=[](const auto& id){return std::any_of(id.begin(),id.end(),[](auto x){return x!=0u;});};
    if(!identityPresent(object.femReferenceSourceIdentity)||!identityPresent(object.femMaterialSourceIdentity))
        return fail("unpaired skin/fat source-reference or material identity is not pinned");
    // No compaction or reordering: supplied volume row i remains object-local node i.
    object.tetrahedra.reserve(nt);object.femMaterialIndices.assign(topology.childMaterialIndices.begin(),topology.childMaterialIndices.end());
    for(const auto& tet:topology.childTetrahedra)object.tetrahedra.push_back({tet});
    if(independent){
        // The conforming independent-role path uses actual count-one volume
        // facets, including explicitly validated head sidewalls. It creates no
        // synthetic 3N material columns and never treats the I interface as exterior.
        for(const auto& [facet,uses]:facetUses)if(uses.size()==1u)
            object.femContactNodes.insert(object.femContactNodes.end(),facet.begin(),facet.end());
    }else{
        // Preserve the paired legacy contact-row selection exactly.
        for(auto node:boundaryMap.outer)object.femContactNodes.push_back(node);
        for(auto node:boundaryMap.fatDeep)object.femContactNodes.push_back(node);
        std::map<std::array<std::uint32_t,2>,unsigned> edgeUse;
        for(const auto& face:surface.faces)for(unsigned e=0;e<3u;++e){auto a=face[e],b=face[(e+1u)%3u];if(a>b)std::swap(a,b);++edgeUse[{a,b}];}
        for(const auto& [edge,count]:edgeUse){if(count>2u)return fail("unpaired RegisteredSurface has a nonmanifold edge while selecting contact rows");if(count==1u){object.femContactNodes.push_back(boundaryMap.skinInner[edge[0]]);object.femContactNodes.push_back(boundaryMap.skinInner[edge[1]]);}}
    }
    std::sort(object.femContactNodes.begin(),object.femContactNodes.end());object.femContactNodes.erase(std::unique(object.femContactNodes.begin(),object.femContactNodes.end()),object.femContactNodes.end());
    if(std::any_of(object.femContactNodes.begin(),object.femContactNodes.end(),[&](auto node){return node>=n;}))return fail("unpaired skin/fat contact node map escapes the object-local volume domain");

    PreparedOrganicFEM nextSkin,nextFat;
    nextSkin.sourceName=surface.name;nextSkin.sourceMember=surface.sourceMember;nextSkin.meshSHA256=surface.sourceMeshSHA256;nextSkin.payloadSHA256=surface.sourcePayloadSHA256;
    nextSkin.materialName=std::string(skinMaterialName);nextSkin.materialProvenance=std::string(skinMaterialProvenance);nextSkin.materialIndex=skinMaterialIndex;nextSkin.firstGlobalFEMNode=static_cast<std::uint32_t>(firstGlobal);nextSkin.volumeM3=skinVolume;nextSkin.massDebitKg=skinDonorDebitKg;nextSkin.densityKgM3=rhoSkin;
    nextFat.sourceName=surface.name;nextFat.sourceMember=surface.sourceMember;nextFat.meshSHA256=surface.sourceMeshSHA256;nextFat.payloadSHA256=surface.sourcePayloadSHA256;
    nextFat.materialName=std::string(fatMaterialName);nextFat.materialProvenance=std::string(fatMaterialProvenance);nextFat.materialIndex=fatMaterialIndex;nextFat.firstGlobalFEMNode=static_cast<std::uint32_t>(firstGlobal);nextFat.volumeM3=fatVolume;nextFat.massDebitKg=fatDonorDebitKg;nextFat.densityKgM3=rhoFat;
    nextSkin.tetSourceFace.reserve(nt);nextFat.tetSourceFace.reserve(nt);
    nextSkin.tetParentAncestryOffsets.push_back(0u);nextFat.tetParentAncestryOffsets.push_back(0u);
    nextSkin.donorMassNodes.reserve(n);nextFat.donorMassNodes.reserve(n);
    for(std::size_t node=0;node<n;++node){
        if(!std::isfinite(nodalMass[node])||nodalMass[node]<=0.0)return fail("unpaired skin/fat node has nonpositive combined density*volume mass");
        const auto body=topology.volumeNodeDonorBodyIndices[node];
        if(skinNodeMass[node]>0.0)nextSkin.donorMassNodes.push_back({.nodeIndex=static_cast<std::uint32_t>(firstGlobal+node),.donorBody=body,.massKg=skinNodeMass[node],.localPosition=topology.volumeNodeDonorLocalPoints[node]});
        if(fatNodeMass[node]>0.0)nextFat.donorMassNodes.push_back({.nodeIndex=static_cast<std::uint32_t>(firstGlobal+node),.donorBody=body,.massKg=fatNodeMass[node],.localPosition=topology.volumeNodeDonorLocalPoints[node]});
    }
    // Store per-material tet ancestry using the same exact child row order.
    for(std::size_t child=0;child<nt;++child){
        auto& prepared=topology.childMaterialIndices[child]==skinMaterialIndex?nextSkin:nextFat;
        prepared.tetChildIndex.push_back(static_cast<std::uint32_t>(child));
        prepared.tetParentTetrahedron.push_back(topology.childDominantParentTetrahedra[child]);
        prepared.tetParentSurfaceFace.push_back(topology.childParentSurfaceFaces[child]);
        prepared.tetParentMaterialIndex.push_back(topology.childMaterialIndices[child]);
        prepared.tetSourceFaceIdentity.push_back(topology.childStableSourceFaceIdentities[child]);
        prepared.tetSourceFace.push_back(topology.childStableSourceFaceIdentities[child]);
        for(std::uint32_t entry=topology.childParentAncestryOffsets[child];
            entry<topology.childParentAncestryOffsets[child+1u];++entry){
            prepared.tetParentAncestryTetrahedra.push_back(topology.childParentAncestryTetrahedra[entry]);
            prepared.tetParentAncestryVolumeFractions.push_back(topology.childParentAncestryVolumeFractions[entry]);
        }
        if(prepared.tetParentAncestryTetrahedra.size()>std::numeric_limits<std::uint32_t>::max())
            return fail("unpaired skin/fat material ancestry entry count exceeds source format");
        prepared.tetParentAncestryOffsets.push_back(static_cast<std::uint32_t>(prepared.tetParentAncestryTetrahedra.size()));
    }
    const auto setDensity=[&](std::uint32_t index,double rho){
        auto& parameters=world.materials[index].parameters;
        auto it=std::find_if(parameters.begin(),parameters.end(),[](const auto& p){return p.name=="density";});
        it->defaultValue=rho;
    };
    setDensity(skinMaterialIndex,rhoSkin);setDensity(fatMaterialIndex,rhoFat);
    world.objects.push_back(std::move(object));
    report={};report.volumeNodeCount=static_cast<std::uint32_t>(n);report.tetrahedronCount=static_cast<std::uint32_t>(nt);
    report.skinVolumeM3=skinVolume;report.fatVolumeM3=fatVolume;report.skinDensityKgM3=rhoSkin;report.fatDensityKgM3=rhoFat;
    report.ancestryVerificationKind=topology.ancestryVerificationKind;
    report.ancestryVerificationReceiptSHA256=std::string(receipt);
    report.inputVolumeRowToObjectLocalNode.resize(n);for(std::uint32_t i=0;i<n;++i)report.inputVolumeRowToObjectLocalNode[i]=i;
    preparedSkin=std::move(nextSkin);preparedFat=std::move(nextFat);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
