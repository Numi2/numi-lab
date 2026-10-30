#pragma once

// Transactional file-level import for the existing combined TetGen,
// registered node-map, and facet-ancestry CSV assets. The caller supplies
// already pinned parent and registered-surface state; this helper does not
// admit a physical child volume or start a simulation.
#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "NumiHumanConformingTissueRefinementAssets.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredVolumeCSV.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredFacetMaps.hpp"
#include "NumiHumanParentReferenceMassTransfer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinOperationalReferencePullbackAssets {
    std::filesystem::path sourceParentPacketCSV;
    std::filesystem::path sourceTetgenNode,sourceTetgenElements,sourceLabels;
    std::array<std::string,3> sourceTetGenSHA256{};
    std::filesystem::path sourceVolumeAsset,sourceProducerReceipt,sourceSelfAuditReceipt;
    std::string sourceVolumeSHA256,sourceProducerReceiptSHA256,sourceSelfAuditReceiptSHA256;
    std::filesystem::path candidateVolumeAsset,candidateProducerReceipt,candidateSelfAuditReceipt;
    std::string candidateVolumeSHA256,candidateProducerReceiptSHA256,candidateSelfAuditReceiptSHA256;
    // Optional verified source 002 -> exact 530 refinement -> fitted candidate chain.
    std::optional<PhysicalSkinConformingTissueRefinementAssets> conformingTissueRefinement;
    // Authenticate geometry/source receipts, then solve and certify the existing
    // conservative spatial mass problem for arbitrary repaired cell topology.
    // This is not a same-topology pullback or a conforming-refinement claim.
    bool conservativeSpatialReference=false;
};

struct PhysicalSkinRegisteredVolumeAssetFiles {
    std::filesystem::path tetgenNode;
    std::filesystem::path tetgenElements;
    std::filesystem::path childLabels;
    std::filesystem::path combinedNodeMapCSV; // existing plain CSV after one-time decompression
    std::filesystem::path facetAncestryCSV;   // existing quoted-ancestry CSV
    // Order: node, elements, labels, combined node map, facet ancestry.
    std::array<std::string,5> expectedSHA256{};
    PhysicalSkinRegisteredVolumeRegistrationDomain registrationDomain=
        PhysicalSkinRegisteredVolumeRegistrationDomain::sourceMassParent;
    std::optional<PhysicalSkinOperationalReferencePullbackAssets> operationalReferencePullback;
};

struct PhysicalSkinRegisteredVolumeFileImportReport {
    PhysicalSkinChildTetGenImportReport tetgen;
    PhysicalSkinCombinedNodeMapCSVReport nodeMap;
    PhysicalSkinRegisteredFacetCSVReport facetAncestry;
    PhysicalSkinRegisteredVolumeImportReport registeredVolume;
    PhysicalSkinRegisteredFacetMapReport facetMaps;
    std::array<std::string,5> inputSHA256{};
    PhysicalSkinRegisteredVolumeRegistrationDomain registrationDomain=
        PhysicalSkinRegisteredVolumeRegistrationDomain::sourceMassParent;
    std::string operationalReferenceDomainSHA256;
    bool operationalReferencePullbackVerified=false;
    std::string operationalReferencePullbackSHA256;
    bool conformingTissueRefinementVerified=false;
    std::string conformingTissueRefinementVerificationSHA256;
    bool operationalReferenceSpatialTransportVerified=false;
    std::string operationalReferenceSpatialTransportVerificationSHA256;
    double maximumSpatialMassTransportDistanceM=0.0;
    double maximumSourceParentRegistrationDistanceM=0.0;
    double maximumSourceToCandidateReferenceDistanceM=0.0;
    bool physicalAdmission=false;
};

namespace physical_skin_registered_volume_integration_detail {
inline std::string operationalReferenceDomainSHA256(
        const PhysicalSkinChildTetGenImportReport& tetgen,const std::string_view coordinateFrameSHA256) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("physical-skin-operational-reference-domain.v1");
    hash.string(tetgen.nodeFileSHA256);hash.string(tetgen.elementFileSHA256);
    hash.string(tetgen.labelFileSHA256);hash.string(coordinateFrameSHA256);
    hash.string("positive-f32-tetgen-reference-with-explicit-cell-labels");
    return hash.finish();
}
inline bool sameImmutableParentPacket(const PhysicalSkinParentMassPacket& a,
                                      const PhysicalSkinParentMassPacket& b) {
    return a.currentWorldPoints==b.currentWorldPoints&&a.referenceWorldPoints==b.referenceWorldPoints&&
        a.compiledLumpedNodeMassKg==b.compiledLumpedNodeMassKg&&
        a.compiledGlobalNodeIndices==b.compiledGlobalNodeIndices&&
        a.nodeFrameOwnerBodyIndices==b.nodeFrameOwnerBodyIndices&&
        a.nodeFrameOwnerLocalPoints==b.nodeFrameOwnerLocalPoints&&a.tetrahedra==b.tetrahedra&&
        a.referenceTetVolumeM3==b.referenceTetVolumeM3&&a.tetrahedronMassKg==b.tetrahedronMassKg&&
        a.materialIndices==b.materialIndices&&a.donorBodyIndices==b.donorBodyIndices&&
        a.anatomicalRegionIndices==b.anatomicalRegionIndices&&
        a.stableSourceFaceIdentities==b.stableSourceFaceIdentities&&
        a.sourceMassContributions.size()==b.sourceMassContributions.size()&&
        std::equal(a.sourceMassContributions.begin(),a.sourceMassContributions.end(),
            b.sourceMassContributions.begin(),[](const auto& x,const auto& y){
                return x.parentNodeIndex==y.parentNodeIndex&&x.materialIndex==y.materialIndex&&
                    x.donorBodyIndex==y.donorBodyIndex&&x.massKg==y.massKg&&
                    x.donorLocalPosition==y.donorLocalPosition;
            })&&
        a.parentMassPacketSHA256==b.parentMassPacketSHA256&&
        a.parentSourceContributionSHA256==b.parentSourceContributionSHA256&&
        a.parentTopologySHA256==b.parentTopologySHA256&&a.parentLabelSHA256==b.parentLabelSHA256&&
        a.totalMassKg==b.totalMassKg&&
        a.maximumNodalMassClosureErrorKg==b.maximumNodalMassClosureErrorKg&&
        a.maximumNodalMassClosureRelativeError==b.maximumNodalMassClosureRelativeError;
}
inline bool pinnedFile(const std::filesystem::path& path,const std::string& expected,
                       std::string& actual,std::string& error) {
    if(path.empty()||!physical_skin_child_mass_detail::validSHA256(expected)){
        error="registered volume import requires an explicit path and lowercase SHA-256 pin";
        return false;
    }
    if(!physical_skin_child_mass_detail::sha256File(path,actual,error))return false;
    if(actual!=expected){
        error="registered volume import file hash differs from its declared pin: "+path.string();
        return false;
    }
    return true;
}
inline bool readPinnedText(const std::filesystem::path& path,const std::string& expected,
                           std::string& text,std::string& error) {
    std::string actual;
    if(!pinnedFile(path,expected,actual,error))return false;
    std::error_code sizeError;
    const auto bytes=std::filesystem::file_size(path,sizeError);
    if(sizeError||bytes>4u*1024u*1024u){error="operational-reference receipt is unreadable or oversized";return false;}
    std::ifstream input(path,std::ios::binary);
    if(!input){error="cannot open pinned operational-reference receipt: "+path.string();return false;}
    text.assign(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
    if(input.bad()){error="failed while reading pinned operational-reference receipt";return false;}
    std::string after;
    if(!pinnedFile(path,expected,after,error)||after!=actual){
        if(error.empty())error="pinned operational-reference receipt changed while it was being read";
        return false;
    }
    return true;
}
inline bool jsonValueDelimiter(const std::string_view json,const std::size_t pos) {
    return pos<json.size()&&
        (std::isspace(static_cast<unsigned char>(json[pos]))||json[pos]==','||
         json[pos]=='}'||json[pos]==']');
}
inline bool jsonStringFieldMatches(const std::string_view json,const std::string_view key,
                                   const std::string_view expected,const std::size_t expectedCount,
                                   std::string& error) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos){
        ++count;pos+=token.size();
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!=':'){error="malformed JSON receipt field "+std::string(key);return false;}
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!='\"'){error="JSON receipt field is not a string: "+std::string(key);return false;}
        std::string value;
        bool closed=false;
        while(pos<json.size()){
            const char c=json[pos++];
            if(c=='\\'){
                if(pos>=json.size()){error="truncated JSON escape in "+std::string(key);return false;}
                value.push_back(json[pos++]);
            }else if(c=='\"'){closed=true;break;}
            else value.push_back(c);
        }
        if(!closed||!jsonValueDelimiter(json,pos)||value!=expected){
            error="JSON receipt field differs from its pin or is not a complete token: "+std::string(key);return false;
        }
    }
    if(count!=expectedCount){error="JSON receipt field count differs from its contract: "+std::string(key);return false;}
    return true;
}
inline bool jsonBooleanFieldMatches(const std::string_view json,const std::string_view key,
                                    const bool expected,std::string& error,
                                    const std::size_t expectedCount=1u) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos){
        ++count;auto cursor=pos+token.size();
        while(cursor<json.size()&&std::isspace(static_cast<unsigned char>(json[cursor])))++cursor;
        if(cursor>=json.size()||json[cursor++]!=':'){error="malformed JSON receipt boolean field";return false;}
        while(cursor<json.size()&&std::isspace(static_cast<unsigned char>(json[cursor])))++cursor;
        const auto literal=expected?std::string_view("true"):std::string_view("false");
        if(json.substr(cursor,literal.size())!=literal){error="JSON receipt boolean field differs from its contract: "+std::string(key);return false;}
        cursor+=literal.size();
        if(!jsonValueDelimiter(json,cursor)){
            error="JSON receipt boolean is not a complete token: "+std::string(key);return false;
        }
        pos=cursor;
    }
    if(count!=expectedCount){error="JSON receipt boolean field count differs from its contract: "+std::string(key)+
        " actual="+std::to_string(count)+" expected="+std::to_string(expectedCount);return false;}
    return true;
}
inline bool jsonExactlyZeroFields(const std::string_view json,const std::string_view key,
                                  const std::size_t expectedCount,std::string& error) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos){
        ++count;pos+=token.size();
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!=':'){error="malformed JSON receipt numeric field "+std::string(key);return false;}
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!='0'||!jsonValueDelimiter(json,pos)){
            error="self-intersection receipt contains a nonzero or malformed integer pair count";return false;
        }
    }
    if(count!=expectedCount){error="self-intersection receipt does not contain the expected exact layer counts";return false;}
    return true;
}
inline bool authenticateSelfReceipt(const std::filesystem::path& path,const std::string& receiptSHA,
                                    const std::string& volumeSHA,std::string& error,
                                    const bool allowRefinedSelfAudit=false) {
    std::string text;
    if(!readPinnedText(path,receiptSHA,text,error))return false;
    if(!jsonStringFieldMatches(text,"status","exact_coarse_material_boundary_self_audit",1u,error)){
        error.clear();
        if(!allowRefinedSelfAudit||
           !jsonStringFieldMatches(text,"status","exact_refined_material_boundary_self_audit",1u,error))
            return false;
    }
    return jsonStringFieldMatches(text,"candidate_sha256",volumeSHA,1u,error)&&
        jsonBooleanFieldMatches(text,"physical_admission",false,error)&&
        jsonExactlyZeroFields(text,"self_intersection_pairs",2u,error);
}
inline bool authenticateOperationalReferenceReceipts(
        const PhysicalSkinOperationalReferencePullbackAssets& proof,
        const std::array<std::string,3>& sourceFileSHA256,
        const std::array<std::string,3>& candidateFileSHA256,
        const std::string& candidateNodeMapSHA256,
        const std::string& candidateFacetMapSHA256,
        const std::string_view coordinateFrameSHA256,
        const PhysicalSkinParentMassPacket& parent,std::string& error) {
    std::string actual;
    if(!pinnedFile(proof.sourceParentPacketCSV,parent.parentMassPacketSHA256,actual,error))return false;
    if(!pinnedFile(proof.sourceVolumeAsset,proof.sourceVolumeSHA256,actual,error))return false;
    if(!pinnedFile(proof.candidateVolumeAsset,proof.candidateVolumeSHA256,actual,error))return false;
    std::string sourceReceipt,candidateReceipt;
    if(!readPinnedText(proof.sourceProducerReceipt,proof.sourceProducerReceiptSHA256,sourceReceipt,error)||
       !readPinnedText(proof.candidateProducerReceipt,proof.candidateProducerReceiptSHA256,candidateReceipt,error))return false;
    const std::array<std::string_view,3> sourceOutputKeys{
        "parent-reduced-boundary.node","parent-reduced-boundary.ele","parent-reduced-boundary.labels.csv"};
    if(sourceReceipt.find(R"json("schema": "NumiHuman.reduced-parent-boundary-candidate.v1")json")==std::string::npos||
       !jsonStringFieldMatches(sourceReceipt,"status","bounded_reduced_boundary_candidate_not_physical_admission",1u,error)||
       !jsonBooleanFieldMatches(sourceReceipt,"physical_admission",false,error)||
       !jsonBooleanFieldMatches(sourceReceipt,"all_child_rows_match_parent_row",true,error)||
       !jsonStringFieldMatches(sourceReceipt,"candidate.npz",proof.sourceVolumeSHA256,1u,error))return false;
    for(unsigned i=0u;i<sourceOutputKeys.size();++i)
        if(!jsonStringFieldMatches(sourceReceipt,sourceOutputKeys[i],sourceFileSHA256[i],1u,error))return false;
    if(candidateReceipt.find(R"json("schema": "numi-human.registered-volume-parent-identity-source-control.v1")json")==std::string::npos||
       !jsonBooleanFieldMatches(candidateReceipt,"physical_admission",false,error,2u)||
       !jsonStringFieldMatches(candidateReceipt,"registration_domain","candidateOperationalReference",1u,error)||
       !jsonStringFieldMatches(candidateReceipt,"candidate_npz_sha256",proof.candidateVolumeSHA256,2u,error)||
       !jsonStringFieldMatches(candidateReceipt,"candidate_node_sha256",candidateFileSHA256[0],1u,error)||
       !jsonStringFieldMatches(candidateReceipt,"candidate_element_sha256",candidateFileSHA256[1],1u,error)||
       !jsonStringFieldMatches(candidateReceipt,"candidate_labels_sha256",candidateFileSHA256[2],1u,error)||
       !jsonStringFieldMatches(candidateReceipt,"node_map_sha256",candidateNodeMapSHA256,1u,error)||
       !jsonStringFieldMatches(candidateReceipt,(proof.candidateProducerReceipt.parent_path()/"volume/facet-ancestry.csv").string(),
                               candidateFacetMapSHA256,1u,error)||
       !jsonStringFieldMatches(candidateReceipt,"coordinate_frame_sha256",coordinateFrameSHA256,2u,error)||
       !jsonStringFieldMatches(candidateReceipt,"immutable_source_parent_packet_sha256",
                               parent.parentMassPacketSHA256,1u,error))return false;
    return authenticateSelfReceipt(proof.sourceSelfAuditReceipt,proof.sourceSelfAuditReceiptSHA256,
                                   proof.sourceVolumeSHA256,error)&&
        authenticateSelfReceipt(proof.candidateSelfAuditReceipt,proof.candidateSelfAuditReceiptSHA256,
                                proof.candidateVolumeSHA256,error,proof.conformingTissueRefinement.has_value());
}
inline bool sameOperationalReferenceCellDomain(
        const PhysicalSkinChildMassMappingBundle& a,
        const PhysicalSkinChildMassMappingBundle& b) {
    return a.referenceWorldPoints.size()==b.referenceWorldPoints.size()&&
        a.childTetrahedra==b.childTetrahedra&&
        a.childMaterialIndices==b.childMaterialIndices&&
        a.childDonorBodyIndices==b.childDonorBodyIndices&&
        a.childAnatomicalRegionIndices==b.childAnatomicalRegionIndices&&
        a.childStableSourceFaceIdentities==b.childStableSourceFaceIdentities;
}
inline bool sameOperationalReferenceCellDomain(
        const PhysicalSkinChildMassMappingBundle& bundle,
        const PhysicalSkinVerifiedConformingReference& refined) {
    return bundle.referenceWorldPoints.size()==refined.referenceWorldPoints.size()&&
        bundle.childTetrahedra==refined.tetrahedra&&
        bundle.childMaterialIndices==refined.materialIndices&&
        bundle.childDonorBodyIndices==refined.geometricDonorBodyIndices&&
        bundle.childAnatomicalRegionIndices==refined.anatomicalRegionIndices&&
        bundle.childStableSourceFaceIdentities==refined.stableSourceFaceIds;
}

inline bool verifyOperationalReferencePullback(
        const PhysicalSkinOperationalReferencePullbackAssets& proof,
        const std::string_view coordinateFrameSHA256,const std::size_t sourceBodyCount,
        const double maximumMassTransportDistanceM,const PhysicalSkinChildMassMappingBundle& candidate,
        const std::span<const PhysicalSkinRegisteredVolumeNodeRow> candidateNodeRows,
        const std::array<std::string,3>& candidateFileSHA256,
        const std::string& candidateNodeMapSHA256,const std::string& candidateFacetMapSHA256,
        SameTopologyReferencePullbackResult& result,double& maximumSourceParentRegistrationDistanceM,
        std::string& conformingTissueVerificationSHA256,
        PhysicalSkinVerifiedConformingMassRoute& verifiedMassRoute,
        ConservativeSpatialMassTransportResult& verifiedSpatialTransport,
        std::string& spatialReferenceVerificationSHA256,std::string& error) {
    PhysicalSkinChildMassMappingBundle source;source.parent=candidate.parent;
    PhysicalSkinChildTetGenImportReport sourceReport;
    if(!loadPhysicalSkinChildTetGenReference(proof.sourceTetgenNode,proof.sourceTetgenElements,
            proof.sourceLabels,coordinateFrameSHA256,sourceBodyCount,maximumMassTransportDistanceM,
            source,sourceReport,error))return false;
    if(sourceReport.nodeFileSHA256!=proof.sourceTetGenSHA256[0]||
       sourceReport.elementFileSHA256!=proof.sourceTetGenSHA256[1]||
       sourceReport.labelFileSHA256!=proof.sourceTetGenSHA256[2])
        {error="operational-reference source TetGen file pins differ from the producer receipt";return false;}
    const auto& parent=candidate.parent;
    if(!authenticateOperationalReferenceReceipts(proof,
            {sourceReport.nodeFileSHA256,sourceReport.elementFileSHA256,sourceReport.labelFileSHA256},
            candidateFileSHA256,candidateNodeMapSHA256,candidateFacetMapSHA256,
            coordinateFrameSHA256,parent,error))return false;
    if(source.referenceWorldPoints.size()!=parent.referenceWorldPoints.size()||
       source.childTetrahedra!=parent.tetrahedra||source.childMaterialIndices!=parent.materialIndices||
       source.childDonorBodyIndices!=parent.donorBodyIndices||
       source.childAnatomicalRegionIndices!=parent.anatomicalRegionIndices||
       source.childStableSourceFaceIdentities!=parent.stableSourceFaceIdentities)
        {error="operational-reference source volume requires exact immutable-parent cell and label correspondence";return false;}
    maximumSourceParentRegistrationDistanceM=0.0;
    for(std::size_t node=0u;node<source.referenceWorldPoints.size();++node){
        V3 f32Parent{};
        for(unsigned axis=0u;axis<3u;++axis)
            f32Parent[axis]=static_cast<double>(static_cast<float>(parent.referenceWorldPoints[node][axis]));
        maximumSourceParentRegistrationDistanceM=std::max(maximumSourceParentRegistrationDistanceM,
            norm(sub(source.referenceWorldPoints[node],f32Parent)));
    }
    if(maximumSourceParentRegistrationDistanceM>0.0004){
        error="operational-reference source registration exceeds the unchanged 0.4 mm source gate";return false;
    }
    if(proof.conservativeSpatialReference&&proof.conformingTissueRefinement){
        error="operational-reference routes are mutually exclusive";return false;
    }
    if(proof.conformingTissueRefinement){
        PhysicalSkinVerifiedConformingReference refined;
        if(!verifyPhysicalSkinConformingTissueRefinementAssets(
                *proof.conformingTissueRefinement,source,proof.sourceVolumeSHA256,sourceBodyCount,refined,error))
            return false;
        if(!sameOperationalReferenceCellDomain(candidate,refined)){
            error="candidate operational reference is not the exact node/cell/label row domain produced by the authenticated 530 refinement";
            return false;
        }
        std::vector<std::uint32_t> identityNodes(candidate.referenceWorldPoints.size());
        std::vector<std::uint32_t> identityTets(candidate.childTetrahedra.size());
        std::iota(identityNodes.begin(),identityNodes.end(),0u);
        std::iota(identityTets.begin(),identityTets.end(),0u);
        std::vector<double> densityLower(17u,0.0),densityUpper(17u,0.0);
        densityLower[15u]=900.0;densityUpper[15u]=1100.0;
        densityLower[16u]=850.0;densityUpper[16u]=950.0;
        for(const auto material:refined.materialIndices)
            if(material!=15u&&material!=16u){error="530 refined pullback has no authoritative material density bounds";return false;}
        SameTopologyReferencePullbackInput refinedInput{
            .sourceReferencePoints=refined.referenceWorldPoints,
            .candidateReferencePoints=candidate.referenceWorldPoints,
            .candidateToSourceNodeRows=identityNodes,
            .sourceTetrahedra=refined.tetrahedra,
            .candidateTetrahedra=candidate.childTetrahedra,
            .candidateToSourceTetRows=identityTets,
            .sourceMaterialIds=refined.materialIndices,
            .candidateMaterialIds=candidate.childMaterialIndices,
            .sourceGeometricDonorIds=refined.geometricDonorBodyIndices,
            .candidateGeometricDonorIds=candidate.childDonorBodyIndices,
            .sourceRegionIds=refined.anatomicalRegionIndices,
            .candidateRegionIds=candidate.childAnatomicalRegionIndices,
            .sourceStableFaceIds=refined.stableSourceFaceIds,
            .candidateStableFaceIds=candidate.childStableSourceFaceIdentities,
            .sourceCellMassKg=refined.refinement.childMassKg,
            .materialDensityLowerKgM3=densityLower,.materialDensityUpperKgM3=densityUpper,
            .sourceVolumeSHA256=refined.refinedVolumeSHA256,
            .candidateVolumeSHA256=proof.candidateVolumeSHA256,
            .parentMassPacketSHA256=parent.parentMassPacketSHA256,
            .parentSourceContributionSHA256=parent.parentSourceContributionSHA256,
            .sourceBoundarySelfAuditSHA256=proof.conformingTissueRefinement->selfAuditReceiptSHA256,
            .candidateBoundarySelfAuditSHA256=proof.candidateSelfAuditReceiptSHA256,
            .sourceBoundarySelfIntersectionPairs=0u,.candidateBoundarySelfIntersectionPairs=0u};
        if(!verifySameTopologyReferencePullback(refinedInput,result,error))return false;
        spatial_mass_transport_detail::SHA256Builder chainHash;
        chainHash.string("physical-skin-operational-reference-conforming-chain.v1");
        chainHash.string(refined.refinement.verificationSHA256);
        chainHash.string(refined.sourceDonorLedgerVerificationSHA256);
        chainHash.string(result.verificationSHA256);
        chainHash.string(proof.sourceVolumeSHA256);chainHash.string(proof.candidateVolumeSHA256);
        conformingTissueVerificationSHA256=chainHash.finish();
        PhysicalSkinVerifiedConformingMassRoute nextRoute;
        nextRoute.authenticated=true;
        nextRoute.refinedSourceCellCount=refined.tetrahedra.size();
        nextRoute.childRefinedSourceRows.resize(candidate.childTetrahedra.size());
        nextRoute.childImmutableParentRows.resize(candidate.childTetrahedra.size());
        nextRoute.childImmutableParentFractions.resize(candidate.childTetrahedra.size());
        nextRoute.childImmutableParentReferenceMeasureM3.resize(candidate.childTetrahedra.size());
        nextRoute.childMassKg.resize(candidate.childTetrahedra.size());
        nextRoute.childSourceDonorOffsets.reserve(candidate.childTetrahedra.size()+1u);
        nextRoute.childSourceDonorOffsets.push_back(0u);
        for(std::size_t child=0u;child<candidate.childTetrahedra.size();++child){
            const auto begin=result.childParentAncestryOffsets[child];
            const auto end=result.childParentAncestryOffsets[child+1u];
            if(end!=begin+1u||end>result.childParentAncestryTetrahedra.size()||
               end>result.childParentAncestryVolumeFractions.size()||
               result.childParentAncestryVolumeFractions[begin]!=1.0){
                error="verified 530-to-candidate reference pullback is not a single exact row ancestry";return false;
            }
            const auto refinedRow=result.childParentAncestryTetrahedra[begin];
            if(refinedRow>=refined.refinement.childParentRows.size()||
               refinedRow>=refined.refinement.childSourceMeasureFractions.size()||
               refinedRow>=refined.refinement.childParentImmutableReferenceMeasureM3.size()||
               refinedRow>=refined.refinement.childMassKg.size()||
               refinedRow+1u>=refined.sourceDonorOffsets.size()||
               result.childMassKg[child]!=refined.refinement.childMassKg[refinedRow]){
                error="verified 530-to-candidate child mass differs from the immutable-330 refinement ledger";return false;
            }
            nextRoute.childRefinedSourceRows[child]=refinedRow;
            nextRoute.childImmutableParentRows[child]=refined.refinement.childParentRows[refinedRow];
            nextRoute.childImmutableParentFractions[child]=refined.refinement.childSourceMeasureFractions[refinedRow];
            nextRoute.childImmutableParentReferenceMeasureM3[child]=
                refined.refinement.childParentImmutableReferenceMeasureM3[refinedRow];
            nextRoute.childMassKg[child]=result.childMassKg[child];
            const auto donorBegin=refined.sourceDonorOffsets[refinedRow];
            const auto donorEnd=refined.sourceDonorOffsets[refinedRow+1u];
            if(donorBegin>=donorEnd||donorEnd>refined.sourceDonorBodyIndices.size()||
               donorEnd>refined.sourceDonorMassKg.size()){
                error="verified 534 source-donor ledger has an empty or invalid refined child row";return false;
            }
            nextRoute.childSourceDonorBodyIndices.insert(nextRoute.childSourceDonorBodyIndices.end(),
                refined.sourceDonorBodyIndices.begin()+donorBegin,
                refined.sourceDonorBodyIndices.begin()+donorEnd);
            nextRoute.childSourceDonorMassKg.insert(nextRoute.childSourceDonorMassKg.end(),
                refined.sourceDonorMassKg.begin()+donorBegin,
                refined.sourceDonorMassKg.begin()+donorEnd);
            nextRoute.childSourceDonorOffsets.push_back(
                static_cast<std::uint32_t>(nextRoute.childSourceDonorBodyIndices.size()));
        }
        nextRoute.parentMassPacketSHA256=parent.parentMassPacketSHA256;
        nextRoute.parentSourceContributionSHA256=parent.parentSourceContributionSHA256;
        nextRoute.sourceVolumeSHA256=proof.sourceVolumeSHA256;
        nextRoute.refinedVolumeSHA256=refined.refinedVolumeSHA256;
        nextRoute.candidateVolumeSHA256=proof.candidateVolumeSHA256;
        nextRoute.refinementVerificationSHA256=refined.refinement.verificationSHA256;
        nextRoute.sourceDonorLedgerVerificationSHA256=refined.sourceDonorLedgerVerificationSHA256;
        nextRoute.pullbackVerificationSHA256=result.verificationSHA256;
        nextRoute.chainVerificationSHA256=conformingTissueVerificationSHA256;
        nextRoute.verificationSHA256=physical_skin_child_mass_mapping_detail::
            verifiedConformingMassRouteSHA256(nextRoute);
        verifiedMassRoute=std::move(nextRoute);
        return true;
    }
    if(proof.conservativeSpatialReference){
        auto mapped=candidate;
        mapped.operationalSpatialMassRouteRequired=false;
        mapped.operationalSpatialReferenceVerificationSHA256.clear();
        mapped.operationalSpatialTransportReceiptSHA256.clear();
        mapped.conformingLineageMassRouteRequired=false;
        mapped.conformingLineageMassTransportUsed=false;
        mapped.conformingLineageMassRoute={};
        std::vector<double> lower(17u,0.0),upper(17u,0.0);
        lower[15u]=900.0;upper[15u]=1100.0;
        lower[16u]=850.0;upper[16u]=950.0;
        if(!preparePhysicalSkinChildMassMapping(mapped,lower,upper,
                maximumMassTransportDistanceM,sourceBodyCount,error)){
            error="operational-reference conservative transport rejected: "+error;return false;
        }
        spatial_mass_transport_detail::SHA256Builder chain;
        chain.string("physical-skin-operational-spatial-reference.v1");
        chain.string(proof.sourceVolumeSHA256);chain.string(proof.candidateVolumeSHA256);
        chain.string(proof.sourceProducerReceiptSHA256);chain.string(proof.candidateProducerReceiptSHA256);
        chain.string(proof.sourceSelfAuditReceiptSHA256);chain.string(proof.candidateSelfAuditReceiptSHA256);
        chain.string(candidateNodeMapSHA256);chain.string(candidateFacetMapSHA256);
        chain.string(coordinateFrameSHA256);
        chain.string(mapped.spatialMassTransportResult.receiptSHA256);
        spatialReferenceVerificationSHA256=chain.finish();
        verifiedSpatialTransport=std::move(mapped.spatialMassTransportResult);
        result={}; // No same-topology correspondence is asserted for this route.
        error.clear();return true;
    }
    // The legacy route still requires the source and candidate to share one
    // exact node/cell/label domain. The explicit refinement route above instead
    // binds the candidate to the separately authenticated 530 domain.
    if(!sameOperationalReferenceCellDomain(candidate,source))
        {error="legacy operational-reference pullback requires exact source/candidate cell and label correspondence";return false;}
    std::vector<std::uint32_t> candidateToSourceNodeRows(candidate.referenceWorldPoints.size(),NM_INVALID_INDEX);
    for(const auto& row:candidateNodeRows){
        if(row.volumeNodeRow>=candidateToSourceNodeRows.size()||
           row.sourceParentNodeRow>=source.referenceWorldPoints.size()||
           candidateToSourceNodeRows[row.volumeNodeRow]!=NM_INVALID_INDEX){
            error="operational-reference node map lacks a unique retained source-node correspondence";return false;
        }
        candidateToSourceNodeRows[row.volumeNodeRow]=row.sourceParentNodeRow;
    }
    if(std::any_of(candidateToSourceNodeRows.begin(),candidateToSourceNodeRows.end(),
                   [](auto row){return row==NM_INVALID_INDEX;})){
        error="operational-reference node map does not cover every source node";return false;
    }
    std::vector<std::uint32_t> candidateToSourceTetRows(candidate.childTetrahedra.size());
    std::iota(candidateToSourceTetRows.begin(),candidateToSourceTetRows.end(),0u);
    std::vector<double> densityLower(17u,0.0),densityUpper(17u,0.0);
    densityLower[15u]=900.0;densityUpper[15u]=1100.0;
    densityLower[16u]=850.0;densityUpper[16u]=950.0;
    for(const auto material:parent.materialIndices)
        if(material!=15u&&material!=16u){error="operational-reference pullback has no authoritative material density bounds";return false;}
    SameTopologyReferencePullbackInput input{
        .sourceReferencePoints=source.referenceWorldPoints,
        .candidateReferencePoints=candidate.referenceWorldPoints,
        .candidateToSourceNodeRows=candidateToSourceNodeRows,
        .sourceTetrahedra=source.childTetrahedra,
        .candidateTetrahedra=candidate.childTetrahedra,
        .candidateToSourceTetRows=candidateToSourceTetRows,
        .sourceMaterialIds=source.childMaterialIndices,
        .candidateMaterialIds=candidate.childMaterialIndices,
        .sourceGeometricDonorIds=source.childDonorBodyIndices,
        .candidateGeometricDonorIds=candidate.childDonorBodyIndices,
        .sourceRegionIds=source.childAnatomicalRegionIndices,
        .candidateRegionIds=candidate.childAnatomicalRegionIndices,
        .sourceStableFaceIds=source.childStableSourceFaceIdentities,
        .candidateStableFaceIds=candidate.childStableSourceFaceIdentities,
        .sourceCellMassKg=parent.tetrahedronMassKg,
        .materialDensityLowerKgM3=densityLower,.materialDensityUpperKgM3=densityUpper,
        .sourceVolumeSHA256=proof.sourceVolumeSHA256,
        .candidateVolumeSHA256=proof.candidateVolumeSHA256,
        .parentMassPacketSHA256=parent.parentMassPacketSHA256,
        .parentSourceContributionSHA256=parent.parentSourceContributionSHA256,
        .sourceBoundarySelfAuditSHA256=proof.sourceSelfAuditReceiptSHA256,
        .candidateBoundarySelfAuditSHA256=proof.candidateSelfAuditReceiptSHA256,
        .sourceBoundarySelfIntersectionPairs=0u,.candidateBoundarySelfIntersectionPairs=0u};
    return verifySameTopologyReferencePullback(input,result,error);
}
}

[[nodiscard]] inline bool importPhysicalSkinRegisteredVolumeFiles(
        const PhysicalSkinRegisteredVolumeAssetFiles& files,
        const std::string_view coordinateFrameSHA256,
        const std::size_t sourceBodyCount,
        const double maximumMassTransportDistanceM,
        const double maximumParentRegistrationDistanceM,
        const std::string_view canonicalDetailedSourceDomain,
        const std::size_t detailedSourceVertexCount,
        const std::size_t detailedSourceFaceCount,
        const PhysicalSkinRegisteredSurfaceCSVSet& registered,
        const std::span<const PhysicalSkinStableDeepAttachmentRequest> deepAttachments,
        const std::span<const PhysicalSkinRegisteredBodyPose> allBodyCurrentPoses,
        PhysicalSkinChildMassMappingBundle& outputBundle,
        PhysicalSkinRegisteredVolumeImportSidecars& outputSidecars,
        PhysicalSkinRegisteredVolumeFileImportReport& outputReport,
        std::string& error,
        const std::span<const PhysicalSkinSourceDomainRegistration> sourceDomains={}) {
    using namespace physical_skin_registered_volume_integration_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(files.combinedNodeMapCSV.extension()==".gz"||files.facetAncestryCSV.extension()==".gz")
        return fail("registered volume import accepts plain existing CSV files after one-time decompression");
    if(sourceBodyCount==0u||!std::isfinite(maximumParentRegistrationDistanceM)||
       maximumParentRegistrationDistanceM<=0.0)
        return fail("registered volume import has an invalid body domain or parent registration bound");

    // All parsing and compilation happens in local state. No caller output is
    // changed unless TetGen, node maps, O/I/D import, and facet maps all pass.
    PhysicalSkinChildMassMappingBundle nextBundle=outputBundle;
    PhysicalSkinChildTetGenImportReport tetgenReport;
    if(!loadPhysicalSkinChildTetGenReference(files.tetgenNode,files.tetgenElements,files.childLabels,
            coordinateFrameSHA256,sourceBodyCount,maximumMassTransportDistanceM,
            nextBundle,tetgenReport,error))
        return false;
    const std::array<std::string,3> tetgenHashes{
        tetgenReport.nodeFileSHA256,tetgenReport.elementFileSHA256,tetgenReport.labelFileSHA256};
    const bool separatedOperationalReference=files.registrationDomain==
        PhysicalSkinRegisteredVolumeRegistrationDomain::candidateOperationalReference;
    if(separatedOperationalReference&&!files.operationalReferencePullback)
        return fail("candidate operational-reference import requires authenticated source and candidate receipts plus a same-topology pullback proof");
    if(!separatedOperationalReference&&files.operationalReferencePullback)
        return fail("source-parent import cannot carry an operational-reference pullback packet");
    auto operationalDomainHash=separatedOperationalReference?
        operationalReferenceDomainSHA256(tetgenReport,coordinateFrameSHA256):std::string{};
    if(separatedOperationalReference&&
       !physical_skin_child_mass_detail::validSHA256(operationalDomainHash))
        return fail("candidate operational-reference hash could not be derived from its pinned geometry/label inputs");
    if(separatedOperationalReference!=registered.parentRegistrationIsHistoricalProvenanceOnly)
        return fail("surface source-parent registration mode differs from the declared operational-reference split");
    for(unsigned i=0;i<3u;++i)
        if(!physical_skin_child_mass_detail::validSHA256(tetgenHashes[i])||tetgenHashes[i]!=files.expectedSHA256[i])
            return fail("registered volume import TetGen file hash differs from its declared pin: "+
                std::array<std::filesystem::path,3>{files.tetgenNode,files.tetgenElements,files.childLabels}[i].string());

    nextBundle.operationalSpatialMassRouteRequired=false;
    nextBundle.operationalSpatialReferenceVerificationSHA256.clear();
    nextBundle.operationalSpatialTransportReceiptSHA256.clear();
    nextBundle.usesSeparatedOperationalReference=separatedOperationalReference;
    nextBundle.conformingLineageMassRouteRequired=separatedOperationalReference&&
        files.operationalReferencePullback->conformingTissueRefinement.has_value();
    std::string mapHashBefore,mapHashAfter;
    if(!pinnedFile(files.combinedNodeMapCSV,files.expectedSHA256[3],mapHashBefore,error))return false;
    std::ifstream nodeMapInput(files.combinedNodeMapCSV);
    if(!nodeMapInput)return fail("cannot open plain combined TetGen node-map CSV");
    PhysicalSkinCombinedNodeMapCSVReport nodeMapReport;
    std::vector<PhysicalSkinRegisteredVolumeNodeRow> nodeRows;
    if(!readPhysicalSkinCombinedNodeMapCSV(nodeMapInput,nextBundle.referenceWorldPoints.size(),
            sourceBodyCount,nodeRows,nodeMapReport,error,separatedOperationalReference))
        return false;
    if(!pinnedFile(files.combinedNodeMapCSV,files.expectedSHA256[3],mapHashAfter,error)||
       mapHashAfter!=mapHashBefore)
        return fail("combined TetGen node-map CSV changed while it was being imported");
    SameTopologyReferencePullbackResult pullbackProof;
    double maximumSourceParentRegistrationDistance=0.0;
    std::string conformingTissueVerificationSHA256;
    PhysicalSkinVerifiedConformingMassRoute verifiedMassRoute;
    ConservativeSpatialMassTransportResult verifiedSpatialTransport;
    std::string spatialReferenceVerificationSHA256;
    if(separatedOperationalReference&&
       !verifyOperationalReferencePullback(*files.operationalReferencePullback,coordinateFrameSHA256,
           sourceBodyCount,maximumMassTransportDistanceM,nextBundle,nodeRows,tetgenHashes,
           mapHashAfter,files.expectedSHA256[4],pullbackProof,maximumSourceParentRegistrationDistance,
           conformingTissueVerificationSHA256,verifiedMassRoute,
           verifiedSpatialTransport,spatialReferenceVerificationSHA256,error))
        return fail("authenticated operational-reference pullback rejected: "+error);
    if(separatedOperationalReference){
        if(!conformingTissueVerificationSHA256.empty()){
            spatial_mass_transport_detail::SHA256Builder domainHash;
            domainHash.string("physical-skin-operational-reference-domain-with-conforming-proof.v1");
            domainHash.string(operationalDomainHash);domainHash.string(conformingTissueVerificationSHA256);
            domainHash.string(pullbackProof.verificationSHA256);
            operationalDomainHash=domainHash.finish();
        }
        if(!spatialReferenceVerificationSHA256.empty()){
            spatial_mass_transport_detail::SHA256Builder hash;
            hash.string("physical-skin-operational-reference-domain-with-spatial-proof.v1");
            hash.string(operationalDomainHash);hash.string(spatialReferenceVerificationSHA256);
            operationalDomainHash=hash.finish();
            nextBundle.operationalSpatialMassRouteRequired=true;
            nextBundle.operationalSpatialReferenceVerificationSHA256=spatialReferenceVerificationSHA256;
            nextBundle.operationalSpatialTransportReceiptSHA256=verifiedSpatialTransport.receiptSHA256;
            nextBundle.spatialMassTransportResult=std::move(verifiedSpatialTransport);
        }else{
            nextBundle.operationalSpatialMassRouteRequired=false;
            nextBundle.operationalSpatialReferenceVerificationSHA256.clear();
            nextBundle.operationalSpatialTransportReceiptSHA256.clear();
        }
        nextBundle.operationalReferenceDomainSHA256=operationalDomainHash;
        if(nextBundle.conformingLineageMassRouteRequired){
            if(!verifiedMassRoute.authenticated){
                error="conforming operational-reference route did not produce an authenticated immutable-mass lineage";return false;
            }
            verifiedMassRoute.candidateOperationalReferenceDomainSHA256=operationalDomainHash;
            verifiedMassRoute.candidateCompiledDataSHA256=physical_skin_child_mass_mapping_detail::
                operationalChildCompiledDataSHA256(nextBundle);
            verifiedMassRoute.verificationSHA256=physical_skin_child_mass_mapping_detail::
                verifiedConformingMassRouteSHA256(verifiedMassRoute);
            nextBundle.conformingLineageMassRoute=std::move(verifiedMassRoute);
        }
    }

    std::string facetHashBefore,facetHashAfter;
    if(!pinnedFile(files.facetAncestryCSV,files.expectedSHA256[4],facetHashBefore,error))return false;
    std::ifstream facetInput(files.facetAncestryCSV);
    if(!facetInput)return fail("cannot open existing registered facet-ancestry CSV");
    PhysicalSkinRegisteredFacetCSVReport facetCSVReport;
    std::vector<PhysicalSkinAuthoredHeadWallRow> authoredWalls;
    if(!readPhysicalSkinAuthoredHeadWallsCSV(facetInput,nextBundle.childTetrahedra.size(),
            nextBundle.referenceWorldPoints.size(),authoredWalls,facetCSVReport,error))
        return false;
    if(!pinnedFile(files.facetAncestryCSV,files.expectedSHA256[4],facetHashAfter,error)||
       facetHashAfter!=facetHashBefore)
        return fail("facet-ancestry CSV changed while it was being imported");
    const auto& surfaces=registered.surfaces;
    const auto expectedRegisteredFacetRows=surfaces[0].faces.size()+
        2u*surfaces[1].faces.size()+surfaces[2].faces.size();
    if(facetCSVReport.registeredRows!=expectedRegisteredFacetRows)
        return fail("facet-ancestry CSV registered O/I/D row count differs from exact registered surfaces");

    PhysicalSkinRegisteredVolumeImportSidecars nextSidecars;
    PhysicalSkinRegisteredVolumeImportReport registeredReport;
    if(!compilePhysicalSkinRegisteredVolumeImport(nextBundle,registered,nodeRows,deepAttachments,
            allBodyCurrentPoses,sourceBodyCount,maximumParentRegistrationDistanceM,
            canonicalDetailedSourceDomain,detailedSourceVertexCount,detailedSourceFaceCount,
            nextSidecars,registeredReport,error,sourceDomains,files.registrationDomain,operationalDomainHash))
        return false;
    PhysicalSkinRegisteredFacetMapReport facetMapReport;
    if(!preparePhysicalSkinRegisteredFacetMaps(nextBundle,authoredWalls,15u,16u,facetMapReport,error))
        return false;
    if(!sameImmutableParentPacket(outputBundle.parent,nextBundle.parent))
        return fail("registered operational-reference import mutated the immutable source-mass parent packet");
    PhysicalSkinRegisteredVolumeFileImportReport measured;
    measured.tetgen=std::move(tetgenReport);measured.nodeMap=nodeMapReport;
    measured.facetAncestry=facetCSVReport;measured.registeredVolume=registeredReport;
    measured.facetMaps=facetMapReport;
    measured.inputSHA256={measured.tetgen.nodeFileSHA256,measured.tetgen.elementFileSHA256,
        measured.tetgen.labelFileSHA256,mapHashAfter,facetHashAfter};
    measured.registrationDomain=files.registrationDomain;
    measured.operationalReferenceDomainSHA256=operationalDomainHash;
    measured.operationalReferencePullbackVerified=separatedOperationalReference&&
        spatialReferenceVerificationSHA256.empty();
    measured.operationalReferenceSpatialTransportVerified=!spatialReferenceVerificationSHA256.empty();
    measured.operationalReferenceSpatialTransportVerificationSHA256=spatialReferenceVerificationSHA256;
    if(measured.operationalReferenceSpatialTransportVerified)
        measured.maximumSpatialMassTransportDistanceM=nextBundle.spatialMassTransportResult.maximumTransportDistanceUsedM;
    if(separatedOperationalReference){
        measured.operationalReferencePullbackSHA256=pullbackProof.verificationSHA256;
        measured.conformingTissueRefinementVerified=!conformingTissueVerificationSHA256.empty();
        measured.conformingTissueRefinementVerificationSHA256=conformingTissueVerificationSHA256;
        measured.maximumSourceParentRegistrationDistanceM=maximumSourceParentRegistrationDistance;
        measured.maximumSourceToCandidateReferenceDistanceM=pullbackProof.maximumNodeDisplacementM;
    }
    measured.physicalAdmission=false;
    outputBundle=std::move(nextBundle);outputSidecars=std::move(nextSidecars);
    outputReport=std::move(measured);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
