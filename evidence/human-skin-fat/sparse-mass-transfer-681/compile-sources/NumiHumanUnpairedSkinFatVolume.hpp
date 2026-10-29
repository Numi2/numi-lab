#pragma once

// Opt-in source preparation for a conforming skin/fat volume mesh whose
// tetrahedral node set is independent of its registered O/I/D boundary rows.
// The legacy paired-layer builder remains unchanged.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "NumiHumanParentReferenceMassTransfer.hpp"
#include "NumiHumanSkinBoundaryNodeMap.hpp"

#include <array>
#include <charconv>
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
#include <vector>

namespace metalrobo::human::coarse_tissue {

enum class VolumeAncestryVerificationKind : std::uint8_t {
    unverified=0, exactGeometricOverlap=1, conservativeReferencePullback=2
};

// All node/tet indices use the source-volume domain. This API preserves that
// order when constructing ObjectSource; surface rows remain separate in the
// SkinBoundaryNodeMap. CSR fractions are caller-supplied ancestry bookkeeping;
// they establish neither geometric overlap nor conservative pullback by themselves.
struct ExplicitUnpairedVolumeSourceMassContribution {
    // Source-domain node/material row. This is independent of a child node's
    // single registered frame owner and is used only for rigid mass debit.
    std::uint32_t parentNodeIndex=NM_INVALID_INDEX;
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::uint32_t donorBodyIndex=NM_INVALID_INDEX;
    double massKg=0.0;
    V3 donorLocalPosition{};
};

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
    // Required only for conservative spatial mass transport. Region identity is
    // checked as an exact barrier alongside material, donor and distance.
    std::span<const std::uint32_t> childAnatomicalRegionIndices;

    std::span<const V3> parentReferenceWorldPoints;
    std::span<const std::array<std::uint32_t,4>> parentTetrahedra;
    std::span<const std::uint32_t> parentMaterialIndices;
    std::span<const std::uint32_t> parentStableSourceFaceIdentities;
    std::span<const std::uint32_t> parentDonorBodyIndices;
    std::span<const std::uint32_t> parentAnatomicalRegionIndices;
    // CSR entry fractions sum to one per child and weighted child volume closes
    // each parent. The declared method requires its own independent verification receipt.
    std::span<const std::uint32_t> childParentAncestryOffsets;
    std::span<const std::uint32_t> childParentAncestryTetrahedra;
    std::span<const double> childParentAncestryVolumeFractions;
    // Optional authoritative parent-mass ledger. Both arrays must be supplied
    // together. childAncestryMeasureVolumeM3 is the measure in the same domain
    // as the CSR fractions, not necessarily the emitted child reference volume:
    // exactGeometricOverlap requires equality, while conservativeReferencePullback
    // may carry a distinct pulled-back measure with its own pinned receipt.
    std::span<const double> parentTetrahedronMassKg;
    std::span<const double> childAncestryMeasureVolumeM3;
    // Optional cooked source ledger in the original parent-node domain. Multiple
    // source donors may contribute to one node/material row; these records never
    // replace the unique per-child-node frame owner.
    std::span<const ExplicitUnpairedVolumeSourceMassContribution> parentNodeSourceMassContributions;
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
    std::span<const PhysicalSkinSourceDomainRegistration> sourceDomains;
    std::span<const std::uint32_t> volumeNodeSourceDomainIndices;
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
    // Spatial mass coupling is an initialization-time transfer, not geometric
    // ancestry. Its separately verified result never sets ancestryVerified.
    UnpairedVolumeMassTransferKind massTransferKind=UnpairedVolumeMassTransferKind::none;
    const ConservativeSpatialMassTransportResult* spatialMassTransportResult=nullptr;
    double maximumSpatialMassTransportDistanceM=0.0;
    // SHA-256 of an external exact-overlap or conservative reference-pullback receipt.
    std::string_view ancestryVerificationReceiptSHA256;
    std::array<std::uint64_t,4> referenceSourceIdentity{};
    std::array<std::uint64_t,4> materialSourceIdentity{};
};

// Shared namespace validator used by both volume admission and reduction-label
// construction. An empty registry is the legacy canonical path; mixed domains
// require complete pinned namespace metadata and never use canonical counts.
[[nodiscard]] inline bool validatePhysicalSkinSourceDomains(
        const RegisteredSurface& outer,const ExplicitUnpairedVolumeTopology& topology,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t nodeCount=topology.currentWorldPoints.size();
    const std::array<const RegisteredSurface*,3> surfaces{
        &outer,topology.independentSkinInnerSurface?topology.independentSkinInnerSurface:&outer,
        topology.independentFatDeepSurface?topology.independentFatDeepSurface:&outer};
    const bool anySurfaceDomains=std::any_of(surfaces.begin(),surfaces.end(),[](const auto* layer){
        return !layer->sourceDomainIndices.empty();
    });
    const bool anyTyped=!topology.sourceDomains.empty()||
        !topology.volumeNodeSourceDomainIndices.empty()||anySurfaceDomains;
    if(!anyTyped){error.clear();return true;}
    if(topology.sourceDomains.empty()||topology.sourceDomains.size()>std::numeric_limits<std::uint32_t>::max()||
       topology.volumeNodeSourceDomainIndices.size()!=nodeCount||
       topology.volumeNodeSourceVertexIndices.size()!=nodeCount||
       topology.volumeNodeSourceFaceIndices.size()!=nodeCount||
       topology.volumeNodeSourceBarycentric.size()!=nodeCount||
       topology.detailedSourceVertexCount!=0u||topology.detailedSourceFaceCount!=0u)
        return fail("typed source namespaces require complete per-node rows and zero canonical detail counts");
    std::set<std::pair<std::string,std::uint8_t>> keys;
    for(const auto& domain:topology.sourceDomains){
        const auto role=static_cast<std::uint8_t>(domain.role);
        if(domain.domain.empty()||role>3u||!validPhysicalSkinSourceDomainSHA256(domain.sourceSHA256)||
           domain.vertexCount==0u||domain.faceCount==0u||
           !keys.emplace(domain.domain,role).second)
            return fail("typed source registry has an invalid, unpinned, empty, or duplicate role namespace");
    }
    const auto expectedRole=[](unsigned role){
        return static_cast<PhysicalSkinSourceDomainRole>(role+1u);
    };
    for(unsigned role=0;role<3u;++role){
        const auto& layer=*surfaces[role];
        const std::size_t count=layer.worldPoints.size();
        if(count==0u||layer.sourceDomainIndices.size()!=count||
           layer.sourceVertexIndices.size()!=count||
           (!layer.sourceEmbeddingFaceIndices.empty()&&layer.sourceEmbeddingFaceIndices.size()!=count)||
           (!layer.sourceEmbeddingBarycentric.empty()&&layer.sourceEmbeddingBarycentric.size()!=count))
            return fail("typed registered surface lacks one source namespace and source identity per node");
        if(layer.sourceEmbeddingFaceIndices.empty()!=layer.sourceEmbeddingBarycentric.empty())
            return fail("typed registered surface face and barycentric rows are incomplete");
        if(!layer.sourceEmbeddingResidualWorldM.empty()&&layer.sourceEmbeddingResidualWorldM.size()!=count)
            return fail("typed registered surface source residual rows are incomplete");
        for(std::size_t row=0;row<count;++row){
            const auto domainIndex=layer.sourceDomainIndices[row];
            if(domainIndex>=topology.sourceDomains.size())
                return fail("typed registered surface references an absent source namespace");
            const auto& domain=topology.sourceDomains[domainIndex];
            if(domain.role!=expectedRole(role)&&domain.role!=PhysicalSkinSourceDomainRole::detailedPayload)
                return fail("typed registered surface source namespace has the wrong anatomical role");
            const auto direct=layer.sourceVertexIndices[row];
            const auto face=layer.sourceEmbeddingFaceIndices.empty()?NM_INVALID_INDEX:
                layer.sourceEmbeddingFaceIndices[row];
            if((direct==NM_INVALID_INDEX)==(face==NM_INVALID_INDEX))
                return fail("typed registered surface node needs exactly one direct or face source identity");
            if(direct!=NM_INVALID_INDEX){
                if(direct>=domain.vertexCount)return fail("typed registered surface vertex ID exceeds its pinned namespace");
                if(!layer.sourceEmbeddingBarycentric.empty()&&
                   layer.sourceEmbeddingBarycentric[row]!=std::array<double,3>{})
                    return fail("direct typed registered source vertex has nonzero barycentric placeholders");
            }else{
                if(face>=domain.faceCount)return fail("typed registered surface face ID exceeds its pinned namespace");
                if(!validateBarycentricCoordinates(layer.sourceEmbeddingBarycentric[row],error))return false;
            }
            if(!layer.sourceEmbeddingResidualWorldM.empty()&&
               !std::all_of(layer.sourceEmbeddingResidualWorldM[row].begin(),
                            layer.sourceEmbeddingResidualWorldM[row].end(),[](double x){return std::isfinite(x);}))
                return fail("typed registered surface source residual is nonfinite");
        }
    }
    for(std::size_t node=0;node<nodeCount;++node){
        const auto domainIndex=topology.volumeNodeSourceDomainIndices[node];
        const auto direct=topology.volumeNodeSourceVertexIndices[node];
        const auto face=topology.volumeNodeSourceFaceIndices[node];
        const auto& bary=topology.volumeNodeSourceBarycentric[node];
        if(domainIndex==NM_INVALID_INDEX){
            if(direct!=NM_INVALID_INDEX||face!=NM_INVALID_INDEX||bary!=std::array<double,3>{})
                return fail("source-free typed volume node carries direct, face, or barycentric ancestry");
            continue;
        }
        if(domainIndex>=topology.sourceDomains.size())
            return fail("typed volume node references an absent source namespace");
        const auto& domain=topology.sourceDomains[domainIndex];
        if((direct==NM_INVALID_INDEX)==(face==NM_INVALID_INDEX))
            return fail("typed volume node needs exactly one direct or face source identity");
        if(direct!=NM_INVALID_INDEX){
            if(direct>=domain.vertexCount||bary!=std::array<double,3>{})
                return fail("typed volume direct source ID or barycentric placeholder is invalid");
        }else if(face>=domain.faceCount||!validateBarycentricCoordinates(bary,error)){
            return fail("typed volume source face ID or barycentric embedding is invalid");
        }
    }
    error.clear();return true;
}

struct ExplicitUnpairedVolumeDonorMassClosure {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::uint32_t donorBodyIndex=NM_INVALID_INDEX;
    double parentMassKg=0.0;
    double childMassKg=0.0;
};

struct ExplicitUnpairedVolumeNodeMassAttribution {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::uint32_t volumeNodeIndex=NM_INVALID_INDEX;
    std::uint32_t donorBodyIndex=NM_INVALID_INDEX;
    double parentAncestryMassKg=0.0;
    double nodeFrameOwnerMassKg=0.0;
};

struct ExplicitUnpairedVolumeNodeOwnerMass {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::uint32_t donorBodyIndex=NM_INVALID_INDEX;
    double parentAncestryMassKg=0.0;
    double nodeFrameOwnerMassKg=0.0;
};

struct ExplicitUnpairedVolumeSourceMassClosure {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::uint32_t donorBodyIndex=NM_INVALID_INDEX;
    double inputSourceMassKg=0.0;
    double cookedParentMassKg=0.0;
    double childSourceMassKg=0.0;
};

struct ExplicitUnpairedVolumeReport {
    std::vector<std::uint32_t> inputVolumeRowToObjectLocalNode;
    std::uint32_t volumeNodeCount=0;
    std::uint32_t tetrahedronCount=0;
    double skinVolumeM3=0.0, fatVolumeM3=0.0;
    double skinDensityKgM3=0.0, fatDensityKgM3=0.0;
    // In the opt-in mass-ledger path, densityKgM3 above remains the unchanged
    // material default. These fields report mass divided by emitted geometric
    // volume and its per-child extrema; they do not replace the density field.
    bool usesPerChildMassLedger=false;
    bool usesParentNodeSourceMassContributions=false;
    double maximumSourceNodeCookRoundingErrorKg=0.0;
    double maximumSourceNodeCookRoundingRelativeError=0.0;
    double maximumOutputSourceNodeMassClosureErrorKg=0.0;
    double skinParentMassKg=0.0, fatParentMassKg=0.0;
    double skinChildMassKg=0.0, fatChildMassKg=0.0;
    double skinMeanImpliedDensityKgM3=0.0, fatMeanImpliedDensityKgM3=0.0;
    double skinMinimumChildDensityKgM3=0.0, skinMaximumChildDensityKgM3=0.0;
    double fatMinimumChildDensityKgM3=0.0, fatMaximumChildDensityKgM3=0.0;
    double maximumDonorMassClosureErrorKg=0.0;
    std::vector<ExplicitUnpairedVolumeDonorMassClosure> donorMassClosure;
    std::vector<ExplicitUnpairedVolumeSourceMassClosure> sourceMassClosure;
    // Legacy diagnostic comparing geometric dominant-parent attribution with the
    // single node/frame owner. With an explicit source ledger, split mass is emitted
    // separately and this diagnostic mismatch is retained without rejecting it.
    bool parentAncestryMatchesNodeFrameMassOwnership=false;
    std::vector<ExplicitUnpairedVolumeNodeOwnerMass> nodeOwnerMassByDonor;
    std::vector<ExplicitUnpairedVolumeNodeMassAttribution> nodeMassAttributionMismatches;
    VolumeAncestryVerificationKind ancestryVerificationKind=VolumeAncestryVerificationKind::unverified;
    std::string ancestryVerificationReceiptSHA256;
    UnpairedVolumeMassTransferKind massTransferKind=UnpairedVolumeMassTransferKind::none;
    bool massCouplingIsGeometricAncestry=false;
    std::string massTransferReceiptSHA256;
    std::size_t massTransportEdgeCount=0u;
    double maximumMassTransportDistanceUsedM=0.0;
    double massWeightedMeanTransportDistanceM=0.0;
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
    const bool typedSourceDomains=!topology.sourceDomains.empty();
    if(!validatePhysicalSkinSourceDomains(surface,topology,error))
        return fail("unpaired source-domain validation failed: "+error);
    if(n<4u||n>std::numeric_limits<std::uint32_t>::max()||nt==0u||nt>kMaxUnpairedTets||
       topology.volumeNodeDonorBodyIndices.size()!=n||topology.volumeNodeDonorLocalPoints.size()!=n||
       topology.volumeNodeDonorLocalToWorldLinear.size()!=n||topology.volumeNodeDonorOriginWorld.size()!=n||
       topology.registeredBodyLocalToWorldLinear.size()!=sourceBodyCount||topology.registeredBodyOriginWorld.size()!=sourceBodyCount||
       (!topology.volumeNodeSourceVertexIndices.empty()&&topology.volumeNodeSourceVertexIndices.size()!=n)||
       (!topology.volumeNodeSourceFaceIndices.empty()&&topology.volumeNodeSourceFaceIndices.size()!=n)||
       (!topology.volumeNodeSourceBarycentric.empty()&&topology.volumeNodeSourceBarycentric.size()!=n)||
       (topology.volumeNodeSourceFaceIndices.empty()!=topology.volumeNodeSourceBarycentric.empty())||
       (!typedSourceDomains&&(topology.detailedSourceVertexCount==0u||topology.detailedSourceFaceCount==0u))||
       (typedSourceDomains&&(topology.detailedSourceVertexCount!=0u||topology.detailedSourceFaceCount!=0u))||
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
    if(independent&&!typedSourceDomains)for(unsigned role=1u;role<3u;++role)
        if(!validateRegisteredSourceEmbedding(*roleSurfaces[role],topology.detailedSourceFaceCount,error))
            return false;
    if(boundaryMap.volumeNodeCount!=n)
        return fail("unpaired skin/fat boundary map volume count differs from the exact source node table");
    const bool hasSpatialMassTransfer=topology.massTransferKind==
        UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    const auto receipt=topology.ancestryVerificationReceiptSHA256;
    if(topology.massTransferKind!=UnpairedVolumeMassTransferKind::none&&!hasSpatialMassTransfer)
        return fail("unpaired skin/fat mass transfer kind is unsupported");
    if(hasSpatialMassTransfer){
        if(topology.spatialMassTransportResult==nullptr||
           !std::isfinite(topology.maximumSpatialMassTransportDistanceM)||
           topology.maximumSpatialMassTransportDistanceM<=0.0||
           topology.ancestryVerificationKind!=VolumeAncestryVerificationKind::unverified||
           !topology.ancestryVerificationReceiptSHA256.empty())
            return fail("spatial mass coupling requires its bounded recorded flow and keeps geometric ancestry unverified");
    }else{
        const auto receipt=topology.ancestryVerificationReceiptSHA256;
        if((topology.ancestryVerificationKind!=VolumeAncestryVerificationKind::exactGeometricOverlap&&
            topology.ancestryVerificationKind!=VolumeAncestryVerificationKind::conservativeReferencePullback)||
           receipt.size()!=64u||!std::all_of(receipt.begin(),receipt.end(),[](char c){
                return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
            return fail("unpaired skin/fat ancestry lacks a pinned exact-overlap or conservative-pullback verification receipt");
    }
    if(topology.referenceWorldPoints.size()!=n||topology.parentReferenceWorldPoints.empty()||
       topology.parentTetrahedra.empty()||topology.parentMaterialIndices.size()!=topology.parentTetrahedra.size()||
       topology.parentStableSourceFaceIdentities.size()!=topology.parentTetrahedra.size()||
       topology.parentDonorBodyIndices.size()!=topology.parentTetrahedra.size())
        return fail("unpaired skin/fat reference or parent source topology is incomplete");
    if(hasSpatialMassTransfer&&
       (topology.parentAnatomicalRegionIndices.size()!=topology.parentTetrahedra.size()||
        topology.childAnatomicalRegionIndices.size()!=nt))
        return fail("spatial mass coupling requires explicit parent and child anatomical region labels");
    const bool hasPerChildMassLedger=!topology.parentTetrahedronMassKg.empty()||
        !topology.childAncestryMeasureVolumeM3.empty()||hasSpatialMassTransfer;
    if(hasSpatialMassTransfer){
        if(topology.parentTetrahedronMassKg.size()!=topology.parentTetrahedra.size()||
           !topology.childAncestryMeasureVolumeM3.empty())
            return fail("unpaired spatial mass coupling arrays do not match the selected transfer method");
    }else if(hasPerChildMassLedger&&
       (topology.parentTetrahedronMassKg.size()!=topology.parentTetrahedra.size()||
        topology.childAncestryMeasureVolumeM3.size()!=nt))
        return fail("unpaired skin/fat per-child mass ledger requires one parent mass and one child ancestry measure per row");
    const bool hasParentNodeSourceMassLedger=!topology.parentNodeSourceMassContributions.empty();
    if(hasParentNodeSourceMassLedger&&!hasPerChildMassLedger)
        return fail("unpaired source-mass contributions require the authoritative per-child parent-mass ledger");
    if(hasSpatialMassTransfer&&!hasParentNodeSourceMassLedger)
        return fail("spatial mass coupling requires the authoritative parent node/material/source-donor ledger");
    if(topology.retainedOriginalNodeCount>n||
       topology.volumeNodeParentTetrahedra.size()!=n||
       topology.volumeNodeParentBarycentric.size()!=n||
       topology.volumeNodeParentResidualWorldM.size()!=n)
        return fail("unpaired skin/fat source-node parent ancestry does not cover the exact volume row domain");
    if(hasSpatialMassTransfer){
        if(!topology.childParentAncestryOffsets.empty()||
           !topology.childParentAncestryTetrahedra.empty()||
           !topology.childParentAncestryVolumeFractions.empty())
            return fail("spatial mass coupling cannot be relabeled or duplicated as geometric parent ancestry");
    }else if(topology.childParentAncestryOffsets.size()!=nt+1u||
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
        if(!typedSourceDomains){
            if(!topology.volumeNodeSourceVertexIndices.empty()&&topology.volumeNodeSourceVertexIndices[i]!=invalid&&
               topology.volumeNodeSourceVertexIndices[i]>=topology.detailedSourceVertexCount)
                return fail("unpaired volume node direct detailed-source identity is out of range at row "+std::to_string(i));
            if(!topology.volumeNodeSourceFaceIndices.empty()&&topology.volumeNodeSourceFaceIndices[i]!=invalid){
                if(topology.volumeNodeSourceFaceIndices[i]>=topology.detailedSourceFaceCount)
                    return fail("unpaired volume node detailed-source face identity is out of range at row "+std::to_string(i));
                std::string baryError;if(!validateBarycentricCoordinates(topology.volumeNodeSourceBarycentric[i],baryError))
                    return fail("unpaired volume node detailed-source embedding is invalid at row "+std::to_string(i));
            }
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
            if(typedSourceDomains){
                if(topology.volumeNodeSourceDomainIndices[node]!=layer.sourceDomainIndices[row]||
                   topology.volumeNodeSourceVertexIndices[node]!=layer.sourceVertexIndices[row])
                    return fail("unpaired boundary source namespace or direct identity differs from its registered surface row");
                const auto surfaceFace=layer.sourceEmbeddingFaceIndices.empty()?invalid:
                    layer.sourceEmbeddingFaceIndices[row];
                const auto volumeFace=topology.volumeNodeSourceFaceIndices[node];
                const auto surfaceBary=layer.sourceEmbeddingBarycentric.empty()?std::array<double,3>{}:
                    layer.sourceEmbeddingBarycentric[row];
                if(volumeFace!=surfaceFace||topology.volumeNodeSourceBarycentric[node]!=surfaceBary)
                    return fail("unpaired boundary source face/barycentric identity differs from its registered surface row");
            }
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
    if(typedSourceDomains)for(std::size_t node=0;node<n;++node)
        if(volumeNodeLayer[node]==invalid&&topology.volumeNodeSourceDomainIndices[node]!=invalid)
            return fail("unpaired interior volume node claims registered surface ancestry");

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
    if(!hasAncestry&&!hasSpatialMassTransfer)
        return fail("unpaired skin/fat volume mesh requires complete parent-volume ancestry CSR");
    for(std::size_t i=1;i<topology.childParentAncestryOffsets.size();++i)
        if(topology.childParentAncestryOffsets[i]<topology.childParentAncestryOffsets[i-1u]||
           topology.childParentAncestryOffsets[i]>topology.childParentAncestryTetrahedra.size())
            return fail("unpaired skin/fat parent-volume CSR offsets are not monotone");
    std::set<std::array<std::uint32_t,4>> uniqueChildren;
    std::vector<double> parentCovered(topology.parentTetrahedra.size(),0.0);
    std::vector<double> parentVolume(topology.parentTetrahedra.size(),0.0);
    std::vector<double> childMassKg(nt,0.0);
    std::map<std::uint32_t,double> parentMassByMaterialKg,childMassByMaterialKg;
    std::map<std::pair<std::uint32_t,std::uint32_t>,double> parentMassByDonorKg,childMassByDonorKg;
    using NodeDonorKey=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    using NodeMaterialKey=std::pair<std::uint32_t,std::uint32_t>;
    using MaterialDonorKey=std::pair<std::uint32_t,std::uint32_t>;
    std::map<NodeDonorKey,double> ancestryMassByNodeDonorKg,nodeOwnerMassByNodeDonorKg;
    std::map<MaterialDonorKey,double> nodeOwnerMassByDonorKg;
    std::map<NodeMaterialKey,std::vector<std::size_t>> sourceRowsByNodeMaterial;
    std::map<NodeMaterialKey,double> rawSourceMassByNodeMaterial,cookedParentMassByNodeMaterial;
    std::map<MaterialDonorKey,double> rawSourceMassByMaterialDonor,cookedParentSourceMassByMaterialDonor,
        childSourceMassByMaterialDonor;
    bool parentAncestryMatchesNodeFrameMassOwnership=true;
    std::map<std::pair<std::uint32_t,std::size_t>,double> childSourceMassByOutputNodeRow;
    if(hasParentNodeSourceMassLedger){
        for(std::size_t row=0;row<topology.parentNodeSourceMassContributions.size();++row){
            const auto& source=topology.parentNodeSourceMassContributions[row];
            if(source.parentNodeIndex>=topology.parentReferenceWorldPoints.size()||
               (source.materialIndex!=skinMaterialIndex&&source.materialIndex!=fatMaterialIndex)||
               source.donorBodyIndex==invalid||source.donorBodyIndex>=sourceBodyCount||
               !std::isfinite(source.massKg)||source.massKg<=0.0||
               !finitePoint(source.donorLocalPosition))
                return fail("unpaired parent-node source-mass ledger has an invalid node, material, donor, mass, or local point at row "+std::to_string(row));
            const NodeMaterialKey key{source.parentNodeIndex,source.materialIndex};
            sourceRowsByNodeMaterial[key].push_back(row);
            rawSourceMassByNodeMaterial[key]+=source.massKg;
            rawSourceMassByMaterialDonor[{source.materialIndex,source.donorBodyIndex}]+=source.massKg;
        }
    }
    for(std::size_t p=0;p<topology.parentTetrahedra.size();++p){
        const auto tet=topology.parentTetrahedra[p];
        for(auto node:tet)if(node>=topology.parentReferenceWorldPoints.size())
            return fail("unpaired skin/fat parent tetrahedron references an absent reference node");
        parentVolume[p]=tetVolume(topology.parentReferenceWorldPoints[tet[0]],topology.parentReferenceWorldPoints[tet[1]],topology.parentReferenceWorldPoints[tet[2]],topology.parentReferenceWorldPoints[tet[3]]);
        if(!std::isfinite(parentVolume[p])||parentVolume[p]<=0.0||
           (topology.parentMaterialIndices[p]!=skinMaterialIndex&&topology.parentMaterialIndices[p]!=fatMaterialIndex)||
           topology.parentDonorBodyIndices[p]==invalid||topology.parentDonorBodyIndices[p]>=sourceBodyCount)
            return fail("unpaired skin/fat parent tetrahedron has invalid material, volume, or donor identity");
        if(hasPerChildMassLedger){
            const double mass=topology.parentTetrahedronMassKg[p];
            if(!std::isfinite(mass)||mass<0.0)
                return fail("unpaired skin/fat parent mass ledger contains a negative or nonfinite mass");
            const auto material=topology.parentMaterialIndices[p];
            parentMassByMaterialKg[material]+=mass;
            parentMassByDonorKg[{material,topology.parentDonorBodyIndices[p]}]+=mass;
            if(hasParentNodeSourceMassLedger){
                for(auto node:tet){
                    const NodeMaterialKey key{node,material};
                    const auto found=sourceRowsByNodeMaterial.find(key);
                    if(found==sourceRowsByNodeMaterial.end()||found->second.empty()||
                       !std::isfinite(rawSourceMassByNodeMaterial[key])||rawSourceMassByNodeMaterial[key]<=0.0)
                        return fail("unpaired parent-node source-mass ledger omits a material corner contribution");
                    cookedParentMassByNodeMaterial[key]+=mass*0.25;
                    for(auto sourceRow:found->second){
                        const auto& source=topology.parentNodeSourceMassContributions[sourceRow];
                        cookedParentSourceMassByMaterialDonor[{material,source.donorBodyIndex}]+=
                            mass*0.25*(source.massKg/rawSourceMassByNodeMaterial[key]);
                    }
                }
            }
        }
    }
    double maximumSourceCookErrorKg=0.0,maximumSourceCookRelativeError=0.0;
    if(hasParentNodeSourceMassLedger){
        std::set<NodeMaterialKey> sourceKeys;
        for(const auto& [key,value]:rawSourceMassByNodeMaterial){(void)value;sourceKeys.insert(key);}
        for(const auto& [key,value]:cookedParentMassByNodeMaterial){(void)value;sourceKeys.insert(key);}
        for(const auto& key:sourceKeys){
            const double raw=rawSourceMassByNodeMaterial[key],cooked=cookedParentMassByNodeMaterial[key];
            if(!(raw>0.0&&cooked>0.0&&std::isfinite(raw)&&std::isfinite(cooked)))
                return fail("unpaired parent-node source ledger does not exactly cover the active parent material-node domain");
            const double errorKg=std::abs(raw-cooked);
            const double relative=errorKg/std::max({raw,cooked,1.0e-12});
            maximumSourceCookErrorKg=std::max(maximumSourceCookErrorKg,errorKg);
            maximumSourceCookRelativeError=std::max(maximumSourceCookRelativeError,relative);
            // The source contribution CSV and the compiled F32 cell ledger use
            // different accumulation paths. Permit only the measured sub-5e-4
            // per-node/material rounding range; retain the exact error in report.
            if(errorKg>std::max(1.0e-10,5.0e-4*std::max(raw,cooked)))
                return fail("unpaired raw source-node mass differs from cooked parent-corner mass beyond the declared Float32 cook-rounding bound");
        }
    }
    if(hasSpatialMassTransfer){
        const auto& recorded=*topology.spatialMassTransportResult;
        std::vector<SpatialMassTransportPoint> parentCentroids(topology.parentTetrahedra.size());
        std::vector<double> parentTransportVolumes(topology.parentTetrahedra.size(),0.0);
        for(std::size_t p=0;p<topology.parentTetrahedra.size();++p){
            const auto tet=topology.parentTetrahedra[p];
            parentTransportVolumes[p]=parentReferenceTetVolumeM3(
                tet,topology.parentReferenceWorldPoints);
            for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
                parentCentroids[p][axis]+=topology.parentReferenceWorldPoints[node][axis]*0.25;
        }
        std::vector<SpatialMassTransportPoint> childCentroids(nt);
        std::vector<double> childReferenceVolumes(nt,0.0);
        for(std::size_t c=0;c<nt;++c){
            const auto tet=topology.childTetrahedra[c];
            for(const auto node:tet)if(node>=reference.size())
                return fail("spatial mass coupling child cell references an absent current source node");
            childReferenceVolumes[c]=parentReferenceTetVolumeM3(tet,reference);
            if(!std::isfinite(childReferenceVolumes[c])||childReferenceVolumes[c]<=0.0)
                return fail("spatial mass coupling child cell has invalid current reference volume");
            for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
                childCentroids[c][axis]+=reference[node][axis]*0.25;
        }
        std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceRows;
        sourceRows.reserve(topology.parentNodeSourceMassContributions.size());
        for(const auto& source:topology.parentNodeSourceMassContributions)
            sourceRows.push_back({.parentNodeIndex=source.parentNodeIndex,
                .materialIndex=source.materialIndex,.sourceDonorBodyIndex=source.donorBodyIndex,
                .massKg=source.massKg});
        std::vector<std::uint32_t> sourceDonorOffsets,sourceDonorBodyIndices;
        std::vector<double> sourceDonorMassKg;
        std::string splitError;
        if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
                topology.parentTetrahedronMassKg,topology.parentMaterialIndices,
                topology.parentTetrahedra,sourceRows,1.0e-10,5.0e-4,
                sourceDonorOffsets,sourceDonorBodyIndices,sourceDonorMassKg,splitError))
            return fail("spatial mass coupling cannot reconstruct cooked parent source-donor split: "+splitError);
        std::vector<double> densityLower(world.materials.size(),
            std::numeric_limits<double>::quiet_NaN());
        std::vector<double> densityUpper(world.materials.size(),
            std::numeric_limits<double>::quiet_NaN());
        for(const auto materialIndex:std::array<std::uint32_t,2>{skinMaterialIndex,fatMaterialIndex}){
            const auto& material=world.materials[materialIndex];
            const auto density=std::find_if(material.parameters.begin(),material.parameters.end(),
                [](const auto& parameter){return parameter.name=="density";});
            if(density==material.parameters.end()||density->identifiable||
               !std::isfinite(density->lower)||!std::isfinite(density->upper)||
               density->lower<=0.0||density->lower>density->upper)
                return fail("spatial mass coupling requires actual finite material density bounds");
            densityLower[materialIndex]=density->lower;
            densityUpper[materialIndex]=density->upper;
        }
        ConservativeSpatialMassTransportInput transportInput;
        transportInput.parentCentroidsWorld=parentCentroids;
        transportInput.parentReferenceVolumeM3=parentTransportVolumes;
        transportInput.parentMassKg=topology.parentTetrahedronMassKg;
        transportInput.parentMaterialIndices=topology.parentMaterialIndices;
        transportInput.parentDonorBodyIndices=topology.parentDonorBodyIndices;
        transportInput.parentAnatomicalRegionIndices=topology.parentAnatomicalRegionIndices;
        transportInput.parentSourceDonorOffsets=sourceDonorOffsets;
        transportInput.parentSourceDonorBodyIndices=sourceDonorBodyIndices;
        transportInput.parentSourceDonorMassKg=sourceDonorMassKg;
        transportInput.childCentroidsWorld=childCentroids;
        transportInput.childReferenceVolumeM3=childReferenceVolumes;
        transportInput.childMaterialIndices=topology.childMaterialIndices;
        transportInput.childDonorBodyIndices=topology.childDonorBodyIndices;
        transportInput.childAnatomicalRegionIndices=topology.childAnatomicalRegionIndices;
        transportInput.materialDensityLowerKgM3=densityLower;
        transportInput.materialDensityUpperKgM3=densityUpper;
        transportInput.parentMassPacketSHA256=recorded.parentMassPacketSHA256;
        transportInput.parentSourceContributionSHA256=recorded.parentSourceContributionSHA256;
        transportInput.childMeshSHA256=recorded.childMeshSHA256;
        transportInput.regionLabelSHA256=recorded.regionLabelSHA256;
        transportInput.coordinateFrameSHA256=recorded.coordinateFrameSHA256;
        transportInput.maximumTransportDistanceM=topology.maximumSpatialMassTransportDistanceM;
        // Admission uses fixed conservative tolerances and work budgets. The
        // candidate cannot widen these by editing the recorded receipt.
        transportInput.relativeClosureTolerance=1.0e-10;
        transportInput.absoluteMassToleranceKg=1.0e-13;
        transportInput.absoluteMeasureToleranceM3=1.0e-15;
        transportInput.maximumAllowedEdges=3'000'000u;
        transportInput.maximumAugmentations=300'000u;
        if(recorded.childDominantParentRows.size()!=nt||
           !std::equal(recorded.childDominantParentRows.begin(),recorded.childDominantParentRows.end(),
                       topology.childDominantParentTetrahedra.begin()))
            return fail("spatial mass coupling dominant parent labels differ from the recorded child flow");
        if(!verifyRecordedConservativeSpatialMassTransport(
                transportInput,recorded,error))
            return fail("spatial mass coupling recorded flow failed current-tet/material/region/source admission: "+error);
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
        if(topology.childMaterialIndices[child]==skinMaterialIndex)skinVolume+=v;else fatVolume+=v;
        childVolumes[child]=v;
        const double ancestryMeasure=hasPerChildMassLedger&&!hasSpatialMassTransfer?
            topology.childAncestryMeasureVolumeM3[child]:v;
        if(!std::isfinite(ancestryMeasure)||ancestryMeasure<=0.0)
            return fail("unpaired skin/fat child ancestry measure is nonpositive or nonfinite at tet "+std::to_string(child));
        if(hasPerChildMassLedger&&!hasSpatialMassTransfer&&
           topology.ancestryVerificationKind==VolumeAncestryVerificationKind::exactGeometricOverlap&&
           std::abs(ancestryMeasure-v)>1.0e-6*std::max(ancestryMeasure,v))
            return fail("exact geometric-overlap mass measure differs from emitted child reference volume at tet "+std::to_string(child));
        for(auto node:ids)++usedNodes[node];
        if(hasAncestry&&topology.childParentAncestryOffsets[child]>=topology.childParentAncestryOffsets[child+1u])
            return fail("unpaired skin/fat child has an empty parent-volume CSR row");
        double sum=0.0;std::uint32_t prior=0u;
        if(hasAncestry)for(std::uint32_t e=topology.childParentAncestryOffsets[child];e<topology.childParentAncestryOffsets[child+1u];++e){
            const auto parent=topology.childParentAncestryTetrahedra[e];const double f=topology.childParentAncestryVolumeFractions[e];
            if(parent>=parentVolume.size()||(e>topology.childParentAncestryOffsets[child]&&parent<=prior)||
               !std::isfinite(f)||f<=0.0||topology.parentMaterialIndices[parent]!=topology.childMaterialIndices[child])
                return fail("unpaired skin/fat child parent CSR has invalid, duplicate, or cross-material ancestry");
            prior=parent;sum+=f;parentCovered[parent]+=ancestryMeasure*f;
            if(hasPerChildMassLedger&&!hasSpatialMassTransfer){
                const double transferred=topology.parentTetrahedronMassKg[parent]*
                    (ancestryMeasure*f/parentVolume[parent]);
                if(!std::isfinite(transferred)||transferred<0.0)
                    return fail("unpaired skin/fat parent-to-child mass transfer is negative or nonfinite");
                childMassKg[child]+=transferred;
                childMassByMaterialKg[topology.childMaterialIndices[child]]+=transferred;
                const auto material=topology.childMaterialIndices[child];
                const auto sourceDonor=topology.parentDonorBodyIndices[parent];
                childMassByDonorKg[{material,sourceDonor}]+=transferred;
                const double nodeShare=0.25*transferred;
                for(auto node:ids){
                    ancestryMassByNodeDonorKg[{material,node,sourceDonor}]+=nodeShare;
                    nodeOwnerMassByNodeDonorKg[{material,node,topology.volumeNodeDonorBodyIndices[node]}]+=nodeShare;
                }
                if(hasParentNodeSourceMassLedger){
                    // Lump child mass to each child corner, then distribute that
                    // corner mass evenly over the four source-parent corners. This
                    // preserves parent source-mass moments without changing the
                    // child node's independent frame owner.
                    const auto parentTet=topology.parentTetrahedra[parent];
                    const double childCornerMass=transferred*0.25;
                    for(auto childNode:ids)for(auto parentNode:parentTet){
                        const NodeMaterialKey sourceKey{parentNode,material};
                        const auto& sourceRows=sourceRowsByNodeMaterial[sourceKey];
                        for(auto sourceRow:sourceRows){
                            const auto& source=topology.parentNodeSourceMassContributions[sourceRow];
                            const double amount=childCornerMass*0.25*
                                (source.massKg/rawSourceMassByNodeMaterial[sourceKey]);
                            childSourceMassByOutputNodeRow[{childNode,sourceRow}]+=amount;
                            childSourceMassByMaterialDonor[{material,source.donorBodyIndex}]+=amount;
                        }
                    }
                }
            }
        }
        if(hasAncestry&&(!std::isfinite(sum)||std::abs(sum-1.0)>1.0e-6))
            return fail("unpaired skin/fat child parent-volume fractions do not sum to one");
    }
    if(hasSpatialMassTransfer){
        const auto& transfer=*topology.spatialMassTransportResult;
        for(std::size_t child=0;child<nt;++child){
            const auto material=topology.childMaterialIndices[child];
            const double transferred=transfer.childMassKg[child];
            childMassKg[child]=transferred;
            childMassByMaterialKg[material]+=transferred;
            childMassByDonorKg[{material,topology.childDonorBodyIndices[child]}]+=transferred;
            const double nodeShare=0.25*transferred;
            for(const auto node:topology.childTetrahedra[child]){
                ancestryMassByNodeDonorKg[{material,node,topology.childDonorBodyIndices[child]}]+=nodeShare;
                nodeOwnerMassByNodeDonorKg[{material,node,topology.volumeNodeDonorBodyIndices[node]}]+=nodeShare;
            }
            for(std::uint32_t edge=transfer.childParentOffsets[child];
                edge<transfer.childParentOffsets[child+1u];++edge){
                const auto parent=transfer.childParentRows[edge];
                const double edgeMass=transfer.childParentTransportMassKg[edge];
                const auto parentTet=topology.parentTetrahedra[parent];
                const double childCornerMass=edgeMass*0.25;
                for(const auto childNode:topology.childTetrahedra[child])
                    for(const auto parentNode:parentTet){
                        const NodeMaterialKey sourceKey{parentNode,material};
                        const auto& sourceRows=sourceRowsByNodeMaterial[sourceKey];
                        for(const auto sourceRow:sourceRows){
                            const auto& source=topology.parentNodeSourceMassContributions[sourceRow];
                            const double amount=childCornerMass*0.25*
                                (source.massKg/rawSourceMassByNodeMaterial[sourceKey]);
                            childSourceMassByOutputNodeRow[{childNode,sourceRow}]+=amount;
                            childSourceMassByMaterialDonor[{material,source.donorBodyIndex}]+=amount;
                        }
                    }
            }
        }
    }
    if(std::any_of(usedNodes.begin(),usedNodes.end(),[](auto x){return x==0u;}))
        return fail("unpaired skin/fat volume node table contains an unused source row");
    if(hasAncestry)for(std::size_t p=0;p<parentVolume.size();++p)
        if(std::abs(parentCovered[p]-parentVolume[p])>1.0e-6*parentVolume[p])
            return fail("unpaired skin/fat CSR volume closure differs from original parent tet "+std::to_string(p));

    double maximumDonorMassClosureErrorKg=0.0;
    if(hasPerChildMassLedger){
        const auto closeMass=[](double a,double b){
            return std::isfinite(a)&&std::isfinite(b)&&
                std::abs(a-b)<=1.0e-6*std::max({std::abs(a),std::abs(b),1.0e-12});
        };
        for(const auto [material,debit]:std::array<std::pair<std::uint32_t,double>,2>{{
                {skinMaterialIndex,skinDonorDebitKg},{fatMaterialIndex,fatDonorDebitKg}}}){
            const double parentMass=parentMassByMaterialKg[material];
            const double childMass=childMassByMaterialKg[material];
            if(!closeMass(parentMass,debit)||!closeMass(childMass,debit))
                return fail("unpaired skin/fat parent/child mass ledger does not close its per-material donor debit");
        }
        for(const auto& [key,parentMass]:parentMassByDonorKg){
            const double childMass=childMassByDonorKg[key];
            const double errorKg=std::abs(parentMass-childMass);
            maximumDonorMassClosureErrorKg=std::max(maximumDonorMassClosureErrorKg,errorKg);
            if(!closeMass(parentMass,childMass))
                return fail("unpaired skin/fat parent-to-child mass ledger does not close a material/donor region");
        }
        for(const auto& [key,childMass]:childMassByDonorKg)
            if(!parentMassByDonorKg.contains(key)||!closeMass(parentMassByDonorKg[key],childMass))
                return fail("unpaired skin/fat child mass ledger introduces a material/donor region without parent mass");
        std::set<NodeDonorKey> nodeAttributionKeys;
        for(const auto& [key,mass]:ancestryMassByNodeDonorKg){(void)mass;nodeAttributionKeys.insert(key);}
        for(const auto& [key,mass]:nodeOwnerMassByNodeDonorKg){(void)mass;nodeAttributionKeys.insert(key);}
        for(const auto& [key,mass]:nodeOwnerMassByNodeDonorKg)
            nodeOwnerMassByDonorKg[{std::get<0>(key),std::get<2>(key)}]+=mass;
        std::vector<ExplicitUnpairedVolumeNodeMassAttribution> attributionMismatches;
        for(const auto& key:nodeAttributionKeys){
            const double sourceMass=ancestryMassByNodeDonorKg[key];
            const double ownerMass=nodeOwnerMassByNodeDonorKg[key];
            if(!closeMass(sourceMass,ownerMass)){
                attributionMismatches.push_back({.materialIndex=std::get<0>(key),
                    .volumeNodeIndex=std::get<1>(key),.donorBodyIndex=std::get<2>(key),
                    .parentAncestryMassKg=sourceMass,.nodeFrameOwnerMassKg=ownerMass});
            }
        }
        report={};report.usesPerChildMassLedger=true;
        report.usesParentNodeSourceMassContributions=hasParentNodeSourceMassLedger;
        report.maximumSourceNodeCookRoundingErrorKg=maximumSourceCookErrorKg;
        report.maximumSourceNodeCookRoundingRelativeError=maximumSourceCookRelativeError;
        parentAncestryMatchesNodeFrameMassOwnership=attributionMismatches.empty();
        report.parentAncestryMatchesNodeFrameMassOwnership=parentAncestryMatchesNodeFrameMassOwnership;
        report.nodeMassAttributionMismatches=attributionMismatches;
        report.skinParentMassKg=parentMassByMaterialKg[skinMaterialIndex];
        report.fatParentMassKg=parentMassByMaterialKg[fatMaterialIndex];
        report.skinChildMassKg=childMassByMaterialKg[skinMaterialIndex];
        report.fatChildMassKg=childMassByMaterialKg[fatMaterialIndex];
        report.maximumDonorMassClosureErrorKg=maximumDonorMassClosureErrorKg;
        for(const auto& [key,parentMass]:parentMassByDonorKg)
            report.donorMassClosure.push_back({.materialIndex=key.first,.donorBodyIndex=key.second,
                .parentMassKg=parentMass,.childMassKg=childMassByDonorKg[key]});
        std::set<std::pair<std::uint32_t,std::uint32_t>> ownerKeys;
        for(const auto& [key,mass]:parentMassByDonorKg){(void)mass;ownerKeys.insert(key);}
        for(const auto& [key,mass]:nodeOwnerMassByDonorKg){(void)mass;ownerKeys.insert(key);}
        for(const auto& key:ownerKeys)
            report.nodeOwnerMassByDonor.push_back({.materialIndex=key.first,.donorBodyIndex=key.second,
                .parentAncestryMassKg=childMassByDonorKg[key],
                .nodeFrameOwnerMassKg=nodeOwnerMassByDonorKg[key]});
        if(!attributionMismatches.empty()&&!hasParentNodeSourceMassLedger)
            return fail("unpaired skin/fat parent-ancestry mass is assigned to a different rigid-body debit donor than node frame ownership; source-contribution mass-moment consumer is required");
        if(hasParentNodeSourceMassLedger){
            std::set<MaterialDonorKey> sourceKeys;
            for(const auto& [key,value]:rawSourceMassByMaterialDonor){(void)value;sourceKeys.insert(key);}
            for(const auto& [key,value]:cookedParentSourceMassByMaterialDonor){(void)value;sourceKeys.insert(key);}
            for(const auto& [key,value]:childSourceMassByMaterialDonor){(void)value;sourceKeys.insert(key);}
            for(const auto& key:sourceKeys){
                const double raw=rawSourceMassByMaterialDonor[key];
                const double cooked=cookedParentSourceMassByMaterialDonor[key];
                const double child=childSourceMassByMaterialDonor[key];
                if(!std::isfinite(raw)||!std::isfinite(cooked)||!std::isfinite(child)||
                   std::abs(cooked-child)>1.0e-6*std::max({std::abs(cooked),std::abs(child),1.0e-12}))
                    return fail("unpaired source-donor mass does not close through the selected child transfer");
                report.sourceMassClosure.push_back({.materialIndex=key.first,.donorBodyIndex=key.second,
                    .inputSourceMassKg=raw,.cookedParentMassKg=cooked,.childSourceMassKg=child});
            }
        }
    }

    auto densityFor=[&](std::uint32_t index,double debit,double volume,double& density)->bool{
        auto& material=world.materials[index];
        if(material.name.empty()||std::find(material.supportedRepresentations.begin(),material.supportedRepresentations.end(),numi::matter::Representation::fem)==material.supportedRepresentations.end())return false;
        auto it=std::find_if(material.parameters.begin(),material.parameters.end(),[](const auto& p){return p.name=="density";});
        if(it==material.parameters.end()||it->identifiable||!std::isfinite(it->lower)||!std::isfinite(it->upper)||it->lower>it->upper)return false;
        density=debit/volume;return std::isfinite(density)&&density>=it->lower&&density<=it->upper;
    };
    double rhoSkin=0.0,rhoFat=0.0;
    double skinDensityLower=0.0,skinDensityUpper=0.0,fatDensityLower=0.0,fatDensityUpper=0.0;
    const auto densityBoundsFor=[&](std::uint32_t index,double& lower,double& upper,double& defaultValue)->bool{
        auto& material=world.materials[index];
        if(material.name.empty()||std::find(material.supportedRepresentations.begin(),material.supportedRepresentations.end(),numi::matter::Representation::fem)==material.supportedRepresentations.end())return false;
        auto it=std::find_if(material.parameters.begin(),material.parameters.end(),[](const auto& p){return p.name=="density";});
        if(it==material.parameters.end()||it->identifiable||!std::isfinite(it->lower)||!std::isfinite(it->upper)||
           !std::isfinite(it->defaultValue)||it->lower>it->upper||it->defaultValue<it->lower||it->defaultValue>it->upper)return false;
        lower=it->lower;upper=it->upper;defaultValue=it->defaultValue;return true;
    };
    double skinDefaultDensity=0.0,fatDefaultDensity=0.0;
    if(world.materials[skinMaterialIndex].name!=skinMaterialName||world.materials[fatMaterialIndex].name!=fatMaterialName||
       !(skinVolume>0.0&&fatVolume>0.0))
        return fail("unpaired skin/fat material identities or geometric volumes are invalid");
    if(hasPerChildMassLedger){
        if(!densityBoundsFor(skinMaterialIndex,skinDensityLower,skinDensityUpper,skinDefaultDensity)||
           !densityBoundsFor(fatMaterialIndex,fatDensityLower,fatDensityUpper,fatDefaultDensity))
            return fail("unpaired skin/fat per-child mass ledger lacks finite material density bounds/defaults");
        // Keep material defaults unchanged; local mass remains in the explicit
        // donorMassNodes ledger rather than being hidden in a fitted uniform rho.
        rhoSkin=skinDefaultDensity;rhoFat=fatDefaultDensity;
    }else if(!densityFor(skinMaterialIndex,skinDonorDebitKg,skinVolume,rhoSkin)||
             !densityFor(fatMaterialIndex,fatDonorDebitKg,fatVolume,rhoFat)){
        return fail("unpaired skin/fat geometric volume implies an invalid or out-of-bounds existing material density");
    }
    double skinMinChildDensity=std::numeric_limits<double>::infinity();
    double skinMaxChildDensity=0.0,fatMinChildDensity=std::numeric_limits<double>::infinity(),fatMaxChildDensity=0.0;
    for(std::size_t child=0;child<nt;++child){
        double childMass=0.0;
        if(hasPerChildMassLedger){
            childMass=childMassKg[child];
            const double impliedDensity=childMass/childVolumes[child];
            const bool isSkin=topology.childMaterialIndices[child]==skinMaterialIndex;
            const double lower=isSkin?skinDensityLower:fatDensityLower;
            const double upper=isSkin?skinDensityUpper:fatDensityUpper;
            const double densityTolerance=hasSpatialMassTransfer?
                spatial_mass_transport_detail::densityBoundRoundoff(lower,upper,impliedDensity):0.0;
            if(!std::isfinite(impliedDensity)||impliedDensity<lower-densityTolerance||
               impliedDensity>upper+densityTolerance)
                return fail("unpaired skin/fat per-child mass implies density outside material bounds at tet "+std::to_string(child));
            if(isSkin){skinMinChildDensity=std::min(skinMinChildDensity,impliedDensity);skinMaxChildDensity=std::max(skinMaxChildDensity,impliedDensity);}
            else{fatMinChildDensity=std::min(fatMinChildDensity,impliedDensity);fatMaxChildDensity=std::max(fatMaxChildDensity,impliedDensity);}
        }else{
            const auto rho=topology.childMaterialIndices[child]==skinMaterialIndex?rhoSkin:rhoFat;
            childMass=rho*childVolumes[child];
        }
        const auto share=childMass*0.25;
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
    if(hasPerChildMassLedger){
        object.femTetrahedronMassesKg=childMassKg;
        const auto massReceipt=hasSpatialMassTransfer?
            std::string_view(topology.spatialMassTransportResult->receiptSHA256):receipt;
        for(std::size_t word=0;word<object.femTetrahedronMassSourceIdentity.size();++word){
            const auto begin=massReceipt.data()+word*16u;
            const auto parsed=std::from_chars(begin,begin+16u,
                object.femTetrahedronMassSourceIdentity[word],16);
            if(parsed.ec!=std::errc{}||parsed.ptr!=begin+16u)
                return fail("unpaired skin/fat mass source identity could not decode the selected transfer receipt");
        }
    }
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
    if(hasAncestry){
        nextSkin.tetParentAncestryOffsets.push_back(0u);
        nextFat.tetParentAncestryOffsets.push_back(0u);
    }
    nextSkin.donorMassNodes.reserve(n);
    for(std::size_t node=0;node<n;++node){
        if(!std::isfinite(nodalMass[node])||nodalMass[node]<=0.0)return fail("unpaired skin/fat node has nonpositive combined density*volume mass");
        const auto body=topology.volumeNodeDonorBodyIndices[node];
        // Exactly one frame/attachment row per compiled node, independent of
        // which source bodies contributed its skin and fat mass.
        nextSkin.donorMassNodes.push_back({.nodeIndex=static_cast<std::uint32_t>(firstGlobal+node),
            .donorBody=body,.massKg=nodalMass[node],.localPosition=topology.volumeNodeDonorLocalPoints[node]});
    }
    if(hasParentNodeSourceMassLedger){
        std::vector<double> explicitMassByNode(n,0.0);
        nextSkin.donorMassContributions.reserve(childSourceMassByOutputNodeRow.size());
        for(const auto& [key,mass]:childSourceMassByOutputNodeRow){
            const auto [node,sourceRow]=key;
            const auto& source=topology.parentNodeSourceMassContributions[sourceRow];
            explicitMassByNode[node]+=mass;
            nextSkin.donorMassContributions.push_back({static_cast<std::uint32_t>(firstGlobal+node),
                source.donorBodyIndex,mass,source.donorLocalPosition});
        }
        double maximumClosure=0.0;
        for(std::size_t node=0;node<n;++node){
            const double errorKg=std::abs(explicitMassByNode[node]-nodalMass[node]);
            maximumClosure=std::max(maximumClosure,errorKg);
            if(errorKg>1.0e-9*std::max({explicitMassByNode[node],nodalMass[node],1.0e-12}))
                return fail("unpaired source contributions do not close the unique compiled node/frame mass");
        }
        report.maximumOutputSourceNodeMassClosureErrorKg=maximumClosure;
    }
    // Store per-material tet ancestry using the same exact child row order.
    for(std::size_t child=0;child<nt;++child){
        auto& prepared=topology.childMaterialIndices[child]==skinMaterialIndex?nextSkin:nextFat;
        prepared.tetChildIndex.push_back(static_cast<std::uint32_t>(child));
        prepared.tetParentTetrahedron.push_back(hasAncestry?
            topology.childDominantParentTetrahedra[child]:invalid);
        prepared.tetParentSurfaceFace.push_back(topology.childParentSurfaceFaces[child]);
        prepared.tetParentMaterialIndex.push_back(topology.childMaterialIndices[child]);
        prepared.tetSourceFaceIdentity.push_back(topology.childStableSourceFaceIdentities[child]);
        prepared.tetSourceFace.push_back(topology.childStableSourceFaceIdentities[child]);
        if(hasAncestry){
            for(std::uint32_t entry=topology.childParentAncestryOffsets[child];
                entry<topology.childParentAncestryOffsets[child+1u];++entry){
                prepared.tetParentAncestryTetrahedra.push_back(topology.childParentAncestryTetrahedra[entry]);
                prepared.tetParentAncestryVolumeFractions.push_back(topology.childParentAncestryVolumeFractions[entry]);
            }
            if(prepared.tetParentAncestryTetrahedra.size()>std::numeric_limits<std::uint32_t>::max())
                return fail("unpaired skin/fat material ancestry entry count exceeds source format");
            prepared.tetParentAncestryOffsets.push_back(static_cast<std::uint32_t>(prepared.tetParentAncestryTetrahedra.size()));
        }
    }
    if(!hasPerChildMassLedger){
        const auto setDensity=[&](std::uint32_t index,double rho){
            auto& parameters=world.materials[index].parameters;
            auto it=std::find_if(parameters.begin(),parameters.end(),[](const auto& p){return p.name=="density";});
            it->defaultValue=rho;
        };
        setDensity(skinMaterialIndex,rhoSkin);setDensity(fatMaterialIndex,rhoFat);
    }
    world.objects.push_back(std::move(object));
    // Preserve ledger details assembled during validation; the final common
    // report initialization must not erase the source/cooked reconciliation.
    const bool usesSourceMassContributions=report.usesParentNodeSourceMassContributions;
    const double preservedSourceCookErrorKg=report.maximumSourceNodeCookRoundingErrorKg;
    const double preservedSourceCookRelativeError=report.maximumSourceNodeCookRoundingRelativeError;
    const double maximumOutputSourceClosureErrorKg=report.maximumOutputSourceNodeMassClosureErrorKg;
    auto sourceMassClosure=std::move(report.sourceMassClosure);
    auto nodeMassAttributionMismatches=std::move(report.nodeMassAttributionMismatches);
    report={};report.volumeNodeCount=static_cast<std::uint32_t>(n);report.tetrahedronCount=static_cast<std::uint32_t>(nt);
    report.usesParentNodeSourceMassContributions=usesSourceMassContributions;
    report.maximumSourceNodeCookRoundingErrorKg=preservedSourceCookErrorKg;
    report.maximumSourceNodeCookRoundingRelativeError=preservedSourceCookRelativeError;
    report.maximumOutputSourceNodeMassClosureErrorKg=maximumOutputSourceClosureErrorKg;
    report.sourceMassClosure=std::move(sourceMassClosure);
    report.nodeMassAttributionMismatches=std::move(nodeMassAttributionMismatches);
    report.skinVolumeM3=skinVolume;report.fatVolumeM3=fatVolume;report.skinDensityKgM3=rhoSkin;report.fatDensityKgM3=rhoFat;
    report.usesPerChildMassLedger=hasPerChildMassLedger;
    if(hasPerChildMassLedger){
        report.skinParentMassKg=parentMassByMaterialKg[skinMaterialIndex];
        report.fatParentMassKg=parentMassByMaterialKg[fatMaterialIndex];
        report.skinChildMassKg=childMassByMaterialKg[skinMaterialIndex];
        report.fatChildMassKg=childMassByMaterialKg[fatMaterialIndex];
        report.skinMeanImpliedDensityKgM3=report.skinChildMassKg/skinVolume;
        report.fatMeanImpliedDensityKgM3=report.fatChildMassKg/fatVolume;
        report.skinMinimumChildDensityKgM3=skinMinChildDensity;
        report.skinMaximumChildDensityKgM3=skinMaxChildDensity;
        report.fatMinimumChildDensityKgM3=fatMinChildDensity;
        report.fatMaximumChildDensityKgM3=fatMaxChildDensity;
        report.maximumDonorMassClosureErrorKg=maximumDonorMassClosureErrorKg;
        report.parentAncestryMatchesNodeFrameMassOwnership=parentAncestryMatchesNodeFrameMassOwnership;
        std::set<std::pair<std::uint32_t,std::uint32_t>> ownerKeys;
        for(const auto& [key,parentMass]:parentMassByDonorKg){
            ownerKeys.insert(key);
            report.donorMassClosure.push_back({.materialIndex=key.first,.donorBodyIndex=key.second,
                .parentMassKg=parentMass,.childMassKg=childMassByDonorKg[key]});
        }
        for(const auto& [key,nodeMass]:nodeOwnerMassByDonorKg){(void)nodeMass;ownerKeys.insert(key);}
        for(const auto& key:ownerKeys)
            report.nodeOwnerMassByDonor.push_back({.materialIndex=key.first,.donorBodyIndex=key.second,
                .parentAncestryMassKg=childMassByDonorKg[key],
                .nodeFrameOwnerMassKg=nodeOwnerMassByDonorKg[key]});
    }else{
        report.skinMeanImpliedDensityKgM3=rhoSkin;report.fatMeanImpliedDensityKgM3=rhoFat;
    }
    report.ancestryVerificationKind=topology.ancestryVerificationKind;
    report.ancestryVerificationReceiptSHA256=std::string(receipt);
    report.massTransferKind=topology.massTransferKind;
    report.massCouplingIsGeometricAncestry=false;
    if(hasSpatialMassTransfer){
        report.massTransferReceiptSHA256=topology.spatialMassTransportResult->receiptSHA256;
        report.massTransportEdgeCount=topology.spatialMassTransportResult->transportEdgeCount;
        report.maximumMassTransportDistanceUsedM=
            topology.spatialMassTransportResult->maximumTransportDistanceUsedM;
        report.massWeightedMeanTransportDistanceM=
            topology.spatialMassTransportResult->massWeightedMeanTransportDistanceM;
    }
    report.inputVolumeRowToObjectLocalNode.resize(n);for(std::uint32_t i=0;i<n;++i)report.inputVolumeRowToObjectLocalNode[i]=i;
    preparedSkin=std::move(nextSkin);preparedFat=std::move(nextFat);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
