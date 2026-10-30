#pragma once
#include "NumiHumanPhysicalSkinChildCapacity.hpp"

// Initialization bridge for Boolean/derived O/I/D role surfaces. Authoring
// vertex IDs stay in their typed role namespace; they never enter the legacy
// NHSKIN direct-source transfer arrays.
#include "NumiHumanPhysicalSkinChildRegisteredVolumeImport.hpp"
#include "NumiHumanAcceptedFEMSkinTransfer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {

inline constexpr std::size_t kPhysicalSkinAuthoringFullWeightBindingCount=86u;

struct PhysicalSkinAuthoringFullWeightsCSV {
    std::filesystem::path path;
    std::string expectedSHA256;
};

struct PhysicalSkinAuthoringRoleSurfaceRows {
    PhysicalSkinSourceDomainRegistration sourceDomain;
    // These IDs are meaningful only inside sourceDomain; there is deliberately
    // no directSourceIDs/sourceFaces field compatible with NHSKIN transfer.
    std::vector<std::uint32_t> sourceDomainIndices,authoringVertexRows,volumeNodeRows;
    std::vector<std::array<double,kPhysicalSkinAuthoringFullWeightBindingCount>> fullBodyWeights86;
    std::vector<std::uint32_t> dominantBindingIndices,frameOwnerBodyIndices;
    std::vector<std::uint32_t> parentGeometricDonorBodyIndices;
    std::vector<V3> currentWorldPoints,donorLocalPoints;
    std::vector<Face> roleLocalFaces;
    std::string fullWeightsSHA256;
};

struct PhysicalSkinAuthoringRoleConsumerState {
    // Order is O, I, D and each node/face index remains role-local.
    std::array<PhysicalSkinAuthoringRoleSurfaceRows,3> roles;
    std::vector<std::uint32_t> bindingBodyByLane;
    bool physicalAdmission=false;
};

struct PhysicalSkinAuthoringRoleConsumerReport {
    std::array<std::size_t,3> rows{},faces{},frameOwnerParentDonorDifferences{};
    std::array<double,3> maximumWeightNormalizationError{};
    std::array<double,3> maximumWorldLocalReconstructionErrorM{};
    std::array<std::string,3> weightFileSHA256{};
    std::size_t bindingCount=0u;
    bool physicalAdmission=false;
};

// The O role is also the physical outer shell. This presentation domain keeps
// the typed authoring namespace and exact volume-node map alongside the
// one-hot transfer rows, so callers never substitute legacy NHSKIN row IDs.
struct PhysicalSkinAuthoringOuterRoleRenderDomain {
    using Transfer=numi::human::AcceptedFEMSkinTransfer;
    PhysicalSkinSourceDomainRegistration sourceDomain;
    std::vector<Transfer::Point> worldPoints;
    std::vector<Transfer::Face> faces;
    std::vector<Transfer::GPUTransferRow> transferRows;
    std::vector<std::uint32_t> authoringVertexRows,volumeNodeRows,frameOwnerBodyIndices;
    bool physicalAdmission=false;
};


// Render-only embedding from a detailed, typed surface into the accepted
// physical volume. `faceTransferRows` retain the independently checked
// source-connected boundary correspondence. `tetraRows` are the rows consumed
// by the accepted-state GPU pass; their four weights include each detail
// point's reference offset from its selected physical tet, so rigid and affine
// motion carry that offset with the same accepted tet.
struct PhysicalSkinAuthoringDetailBridgeInput {
    PhysicalSkinSourceDomainRegistration detailDomain,physicalBoundaryDomain;
    std::vector<V3> detailReferenceWorldPoints;
    std::vector<V3> detailReferenceWorldNormals;
    std::vector<std::array<double,kPhysicalSkinAuthoringFullWeightBindingCount>>
        detailFullBodyWeights86;
    std::vector<std::uint32_t> detailDominantBindingIndices;
    std::vector<std::uint32_t> detailSourceVertexRows,detailGeometricDonorBodyIndices;
    // The registered detail-row donor is retained separately from the
    // independently registered parent-tet geometric donor.
    std::vector<std::uint32_t> detailRegisteredSourceDonorBodyIndices;
    std::vector<std::uint32_t> detailMaterialIndices,detailAnatomicalRegionIndices;
    std::vector<std::uint32_t> detailFrameOwnerBodyIndices;
    std::vector<std::uint32_t> detailBoundaryProjectionDonorBodyIndices;
    std::vector<std::uint8_t> detailRegionScopedDonorFallback;
    // A source-tet ancestry walk can be required even when the geometric
    // donor label is unchanged. Empty preserves original direct callers,
    // where only explicit donor remaps needed a walk.
    std::vector<std::uint8_t> detailRequiresParentTetPath;
    std::vector<std::uint32_t> detailParentTetRows,detailStableSourceFaceIDs;
    // Enabled only by explicit source-only export/preflight modes.
    bool allowSourceOnlyStrictChildFaceSelection=false;
    std::vector<Face> detailFaces;
    std::size_t sourceParentNodeCount=0u;
    std::vector<std::array<std::uint32_t,4>> sourceParentTetNodeRows;
    std::vector<std::uint32_t> sourceParentTetMaterialIndices,sourceParentTetAnatomicalRegionIndices;
    std::vector<std::uint32_t> sourceParentTetStableSourceFaceIDs;
    std::vector<std::uint32_t> physicalBoundaryFaceParentTetRows;
    std::vector<std::uint32_t> physicalBoundaryFaceChildTetRows;
    std::uint32_t maximumSourceParentTetHops=8u;

    std::vector<V3> physicalBoundaryReferenceWorldPoints;
    std::vector<std::uint32_t> physicalBoundarySourceVertexRows;
    std::vector<std::uint32_t> physicalBoundaryGeometricDonorBodyIndices;
    std::vector<std::uint32_t> physicalBoundaryFrameOwnerBodyIndices;
    std::vector<std::uint32_t> physicalBoundaryVolumeNodeRows;
    std::vector<Face> physicalBoundaryFaces;
    std::vector<std::uint32_t> physicalBoundaryStableFaceIDs;
    std::vector<std::uint32_t> physicalBoundaryFaceMaterialIndices;
    std::vector<std::uint32_t> physicalBoundaryFaceGeometricDonorBodyIndices;
    std::vector<std::uint32_t> physicalBoundaryFaceAnatomicalRegionIndices;
    std::vector<std::uint32_t> physicalBoundaryFaceFrameOwnerBodyIndices;

    // Producer-owned source-connected nearest-face witness for each detail
    // row. The native transfer builder independently recomputes the same
    // selected face and barycentrics before this witness is accepted.
    std::vector<std::uint32_t> detailRepresentativeBoundaryVertexRows;
    std::vector<std::uint32_t> detailExpectedBoundaryFaceRows;
    std::vector<std::array<double,3>> detailExpectedBoundaryFaceBarycentrics;

    // The selected tet row is in this physical volume's cell namespace. Its
    // four node IDs are in mechanicalReferenceWorldPoints and, later, the
    // same accepted FEM position array.
    std::vector<V3> mechanicalReferenceWorldPoints;
    std::vector<std::array<std::uint32_t,4>> mechanicalTetNodeRows;
    std::vector<std::uint32_t> mechanicalTetMaterialIndices;
    std::vector<std::uint32_t> mechanicalTetGeometricDonorBodyIndices;
    std::vector<std::uint32_t> mechanicalTetAnatomicalRegionIndices;
    std::vector<std::uint32_t> detailMechanicalTetRows;

    double maximumBoundaryProjectionDistanceM=0.0;
    double maximumTetExtrapolationDistanceM=0.0;
    double maximumAbsoluteEffectiveWeight=1.5;
    double maximumTetConditionEstimate=1.0e8;
    double maximumReferenceReconstructionErrorM=2.0e-7;
    double faceBarycentricComparisonTolerance=2.0e-6;
};

struct PhysicalSkinAuthoringDetailBridgeState {
    using TetRow=numi::human::AcceptedFEMSkinTransfer::GPUTetrahedralTransferRow;
    PhysicalSkinSourceDomainRegistration detailDomain,physicalBoundaryDomain;
    std::vector<TetRow> tetraRows;
    // Audit-only surface correspondence; these are physical boundary face
    // rows, not detailed row IDs or contact anchors.
    std::vector<numi::human::AcceptedFEMSkinTransfer::GPUTransferRow> faceTransferRows;
    std::vector<std::uint32_t> selectedPhysicalBoundaryFaceRows;
    std::vector<std::uint32_t> detailGeometricDonorBodyIndices,detailFrameOwnerBodyIndices;
    std::vector<std::uint32_t> selectedPhysicalBoundaryFaceDonorBodyIndices;
    std::vector<std::uint32_t> selectedPhysicalBoundaryFaceFrameOwnerBodyIndices;
    std::vector<std::uint8_t> regionScopedDonorFallbackRows;
    std::vector<std::uint32_t> sourceParentTetHopCounts;
    std::vector<std::vector<std::uint32_t>> sourceParentTetPathRows;
    bool provenDirectIdentity=false;
    bool physicalAdmission=false;
};

struct PhysicalSkinAuthoringDetailBridgeReport {
    std::size_t detailRows=0,boundaryRows=0,physicalNodeRows=0;
    std::size_t surfaceProjectedRows=0,surfaceExactAncestryRows=0;
    std::size_t tetraEmbeddedRows=0,directIdentityRows=0;
    std::size_t regionScopedDonorFallbackRows=0,sourceParentTetWalkRows=0;
    std::uint32_t maximumSourceParentTetHops=0u;
    double maximumBoundaryProjectionDistanceM=0.0;
    double maximumTetExtrapolationDistanceM=0.0;
    double maximumAbsoluteEffectiveWeight=0.0;
    double maximumTetConditionEstimate=0.0;
    double maximumReferenceReconstructionErrorM=0.0;
    bool sourceNamespaceIdentitySuppressed=false;
    bool physicalAdmission=false;
};

namespace physical_skin_authoring_role_consumer_detail {

inline std::vector<std::string> weightHeader() {
    std::vector<std::string> header{"vertex"};
    for(std::size_t lane=0u;lane<kPhysicalSkinAuthoringFullWeightBindingCount;++lane)
        header.push_back("w"+std::to_string(lane));
    return header;
}

inline bool readPinnedWeights(const PhysicalSkinAuthoringFullWeightsCSV& asset,
        const std::size_t expectedRows,
        std::vector<std::array<double,kPhysicalSkinAuthoringFullWeightBindingCount>>& output,
        std::string& actualSHA256,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_child_mass_detail::sha256File;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(asset.path.empty()||!physical_skin_child_mass_detail::validSHA256(asset.expectedSHA256)||
       expectedRows==0u||expectedRows>physical_skin_child_capacity_detail::maxRoleNodes)
        return fail("authoring full86 weights require an explicit pinned file and bounded role node count");
    std::string hashBefore;
    if(!sha256File(asset.path,hashBefore,error))return false;
    if(hashBefore!=asset.expectedSHA256)
        return fail("authoring full86 weight CSV differs from its declared SHA-256: "+asset.path.string());
    std::ifstream input(asset.path);
    if(!input)return fail("cannot open authoring full86 weight CSV: "+asset.path.string());
    std::string line;
    if(!std::getline(input,line)||physical_skin_child_mass_detail::splitCSV(line)!=weightHeader())
        return fail("authoring full86 weight CSV must use the existing vertex,w0..w85 header");
    std::vector<std::array<double,kPhysicalSkinAuthoringFullWeightBindingCount>> next(expectedRows);
    std::vector<std::uint8_t> seen(expectedRows,0u);
    std::size_t fileRows=0u;
    while(std::getline(input,line)) {
        if(line.size()>16384u||fileRows>=expectedRows)
            return fail("authoring full86 weight CSV exceeds its bounded row domain");
        const auto fields=physical_skin_child_mass_detail::splitCSV(line);
        if(fields.size()!=kPhysicalSkinAuthoringFullWeightBindingCount+1u)
            return fail("authoring full86 weight CSV row has an invalid width");
        std::uint32_t vertex=NM_INVALID_INDEX;
        if(!parseNumber(fields[0],vertex)||vertex>=expectedRows||seen[vertex])
            return fail("authoring full86 weight CSV has a duplicate or out-of-range surface vertex row");
        double sum=0.0;
        for(std::size_t lane=0u;lane<kPhysicalSkinAuthoringFullWeightBindingCount;++lane) {
            double value=0.0;
            if(!parseNumber(fields[lane+1u],value)||!std::isfinite(value)||value<0.0)
                return fail("authoring full86 weight CSV contains a nonfinite or negative weight");
            next[vertex][lane]=value;sum+=value;
        }
        if(!std::isfinite(sum)||std::abs(sum-1.0)>1.0e-4)
            return fail("authoring full86 weight CSV row does not normalize");
        seen[vertex]=1u;++fileRows;
    }
    if(!input.eof()||fileRows!=expectedRows||
       std::find(seen.begin(),seen.end(),std::uint8_t(0u))!=seen.end())
        return fail("authoring full86 weight CSV does not bijectively cover every role-local surface row");
    std::string hashAfter;
    if(!sha256File(asset.path,hashAfter,error)||hashAfter!=hashBefore)
        return fail("authoring full86 weight CSV changed during import");
    output=std::move(next);actualSHA256=std::move(hashAfter);error.clear();return true;
}

inline std::array<double,9> expectedWorldToDonorLocal(const PhysicalSkinRegisteredBodyPose& pose) {
    return physical_skin_registered_surface_csv_detail::worldToLocal(
        pose.currentLocalToWorld,pose.arithmetic);
}
inline V3 expectedDonorLocal(const PhysicalSkinRegisteredBodyPose& pose,const V3& world) {
    return physical_skin_registered_surface_csv_detail::mul(
        expectedWorldToDonorLocal(pose),sub(world,pose.currentOrigin));
}

inline bool closeMatrix(const std::array<double,9>& a,const std::array<double,9>& b,
                        const double tolerance) {
    for(std::size_t i=0u;i<a.size();++i)
        if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::abs(a[i]-b[i])>tolerance)return false;
    return true;
}

} // namespace physical_skin_authoring_role_consumer_detail

// Consumes the already pinned import state and one existing-format full86 CSV
// per role. The output is separate from WholeBodySkinSourceConsumerRows because
// this source identity is candidate-authoring identity, never an NHSKIN ID.
[[nodiscard]] inline bool preparePhysicalSkinAuthoringRoleConsumerState(
        const PhysicalSkinChildMassMappingBundle& bundle,
        const PhysicalSkinRegisteredSurfaceCSVSet& registered,
        const PhysicalSkinRegisteredVolumeImportSidecars& importSidecars,
        const std::span<const PhysicalSkinRegisteredBodyPose> allBodyCurrentPoses,
        const std::array<PhysicalSkinAuthoringFullWeightsCSV,3>& weightsFiles,
        PhysicalSkinAuthoringRoleConsumerState& output,
        PhysicalSkinAuthoringRoleConsumerReport& report,
        std::string& error,
        const double worldLocalToleranceM=2.0e-8,
        const double frameMatrixTolerance=2.0e-7) {
    using namespace physical_skin_authoring_role_consumer_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    constexpr std::array<PhysicalSkinSourceDomainRole,3> expectedRoles{
        PhysicalSkinSourceDomainRole::outerSurface,
        PhysicalSkinSourceDomainRole::innerSurface,
        PhysicalSkinSourceDomainRole::deepSurface};
    const std::array<const RegisteredSurface*,3> surfaces{
        &bundle.outerSkin,&bundle.independentSkinInner,&bundle.independentFatDeep};
    const std::array<const std::vector<std::uint32_t>*,3> boundaryMaps{
        &bundle.boundaryMap.outer,&bundle.boundaryMap.skinInner,&bundle.boundaryMap.fatDeep};
    const std::size_t volumeNodes=bundle.referenceWorldPoints.size();
    if(volumeNodes<4u||bundle.currentWorldPoints.size()!=volumeNodes||
       bundle.boundaryMap.volumeNodeCount!=volumeNodes||
       importSidecars.currentParentRows.size()!=volumeNodes||
       bundle.sourceDomains.empty()||bundle.volumeNodeSourceDomainIndices.size()!=volumeNodes||
       bundle.volumeNodeSourceVertexIndices.size()!=volumeNodes||
       bundle.volumeNodeSourceFaceIndices.size()!=volumeNodes||
       bundle.volumeNodeParentTetrahedra.size()!=volumeNodes||
       registered.framesByDonor.size()!=allBodyCurrentPoses.size()||
       allBodyCurrentPoses.empty()||!std::isfinite(worldLocalToleranceM)||worldLocalToleranceM<=0.0||
       !std::isfinite(frameMatrixTolerance)||frameMatrixTolerance<=0.0)
        return fail("authoring role consumer requires a complete typed import, parent registration and native pose table");
    std::string mapError;
    if(!numi::human::validateSkinBoundaryNodeMap(bundle.boundaryMap,
            {bundle.outerSkin.worldPoints.size(),bundle.independentSkinInner.worldPoints.size(),
             bundle.independentFatDeep.worldPoints.size()},mapError))
        return fail("authoring role consumer received invalid exact O/I/D boundary maps: "+mapError);
    if(bundle.parent.donorBodyIndices.size()!=bundle.parent.tetrahedra.size()||
       bundle.parent.materialIndices.size()!=bundle.parent.tetrahedra.size()||
       bundle.parent.anatomicalRegionIndices.size()!=bundle.parent.tetrahedra.size())
        return fail("authoring role consumer lacks the separate labeled parent-cell domain");

    std::vector<std::uint32_t> bindingBodyByLane(kPhysicalSkinAuthoringFullWeightBindingCount,NM_INVALID_INDEX);
    std::size_t registeredFrameCount=0u;
    for(std::size_t body=0u;body<registered.framesByDonor.size();++body) {
        const auto& frame=registered.framesByDonor[body];
        if(!frame.available)continue;
        if(body>=allBodyCurrentPoses.size()||frame.binding>=bindingBodyByLane.size())
            return fail("registered authoring frame points outside the actual body or 86-lane binding domain");
        const auto& pose=allBodyCurrentPoses[body];
        if(!pose.available||pose.bodyIndex!=body||pose.binding!=frame.binding||
           pose.currentOrigin!=frame.currentOrigin||pose.currentLocalToWorld!=frame.currentR)
            return fail("registered authoring frame does not match the pinned actual current body pose");
        if(bindingBodyByLane[frame.binding]!=NM_INVALID_INDEX)
            return fail("registered authoring pose lanes are not a one-to-one 86-lane map");
        bindingBodyByLane[frame.binding]=static_cast<std::uint32_t>(body);++registeredFrameCount;
    }
    if(registeredFrameCount!=kPhysicalSkinAuthoringFullWeightBindingCount||
       std::find(bindingBodyByLane.begin(),bindingBodyByLane.end(),NM_INVALID_INDEX)!=bindingBodyByLane.end())
        return fail("registered authoring pose table does not cover every 86 full-weight lane");

    PhysicalSkinAuthoringRoleConsumerState next;
    next.bindingBodyByLane=bindingBodyByLane;
    PhysicalSkinAuthoringRoleConsumerReport measured;
    measured.bindingCount=kPhysicalSkinAuthoringFullWeightBindingCount;
    for(unsigned layer=0u;layer<3u;++layer) {
        const auto& surface=*surfaces[layer];
        const auto& boundary=*boundaryMaps[layer];
        const auto& nodeSource=registered.nodeSource[layer];
        const std::size_t count=surface.worldPoints.size();
        if(count<3u||surface.faces.empty()||surface.sourceLayer!=layer||
           surface.donorLocalPoints.size()!=count||surface.donorBodyIndices.size()!=count||
           surface.worldToDonorLocalLinear.size()!=count||surface.sourceVertexIndices.size()!=count||
           surface.sourceDomainIndices.size()!=count||nodeSource.size()!=count||boundary.size()!=count)
            return fail("authoring O/I/D role surface lacks complete exact point, frame, source, or boundary rows");
        if(!surface.sourceEmbeddingFaceIndices.empty()&&surface.sourceEmbeddingFaceIndices.size()!=count)
            return fail("authoring surface source embedding rows have an invalid extent");
        if(!surface.sourceEmbeddingBarycentric.empty()&&surface.sourceEmbeddingBarycentric.size()!=count)
            return fail("authoring surface source barycentric rows have an invalid extent");

        std::vector<std::array<double,kPhysicalSkinAuthoringFullWeightBindingCount>> weights;
        std::string weightsSHA;
        if(!readPinnedWeights(weightsFiles[layer],count,weights,weightsSHA,error))return false;
        PhysicalSkinAuthoringRoleSurfaceRows roleRows;
        roleRows.fullWeightsSHA256=weightsSHA;roleRows.fullBodyWeights86=std::move(weights);
        roleRows.sourceDomainIndices=surface.sourceDomainIndices;
        roleRows.currentWorldPoints=surface.worldPoints;roleRows.donorLocalPoints=surface.donorLocalPoints;
        roleRows.roleLocalFaces=surface.faces;roleRows.authoringVertexRows.resize(count,NM_INVALID_INDEX);
        roleRows.volumeNodeRows.resize(count,NM_INVALID_INDEX);
        roleRows.dominantBindingIndices.resize(count,NM_INVALID_INDEX);
        roleRows.frameOwnerBodyIndices.resize(count,NM_INVALID_INDEX);
        roleRows.parentGeometricDonorBodyIndices.resize(count,NM_INVALID_INDEX);
        double maxNormError=0.0,maxLocalError=0.0;
        for(std::size_t row=0u;row<count;++row) {
            const auto domainIndex=surface.sourceDomainIndices[row];
            if(domainIndex>=bundle.sourceDomains.size())
                return fail("authoring surface row has an out-of-range pinned source namespace index");
            const auto& domain=bundle.sourceDomains[domainIndex];
            const auto& source=nodeSource[row];
            if(domain.role!=expectedRoles[layer]||domain.domain.empty()||
               !validPhysicalSkinSourceDomainSHA256(domain.sourceSHA256)||
               domain.vertexCount!=count||domain.faceCount!=surface.faces.size()||
               source.domain!=domain.domain||source.vertex<0||source.face!=-1||
               static_cast<std::uint64_t>(source.vertex)!=row||
               surface.sourceVertexIndices[row]!=static_cast<std::uint32_t>(row))
                return fail("authoring vertex ID is not the exact role-local row in its pinned namespace");
            if(source.bary!=std::array<double,3>{}||source.residual!=V3{})
                return fail("direct authoring identity carries nonzero source embedding barycentric or residual data");
            if(!surface.sourceEmbeddingFaceIndices.empty()&&
               surface.sourceEmbeddingFaceIndices[row]!=NM_INVALID_INDEX)
                return fail("derived authoring vertex row was misinterpreted as a detailed NHSKIN face embedding");
            if(!surface.sourceEmbeddingBarycentric.empty()&&
               surface.sourceEmbeddingBarycentric[row]!=std::array<double,3>{})
                return fail("direct authoring vertex identity unexpectedly carries face barycentrics");
            if(!surface.sourceEmbeddingResidualWorldM.empty()&&
               surface.sourceEmbeddingResidualWorldM[row]!=V3{})
                return fail("direct authoring vertex identity unexpectedly carries source residual data");

            const auto volumeNode=boundary[row];
            if(volumeNode>=volumeNodes||volumeNode>=bundle.volumeNodeDonorBodyIndices.size()||
               bundle.volumeNodeSourceDomainIndices[volumeNode]!=domainIndex||
               bundle.volumeNodeSourceVertexIndices[volumeNode]!=static_cast<std::uint32_t>(source.vertex)||
               bundle.volumeNodeSourceFaceIndices[volumeNode]!=NM_INVALID_INDEX||
               !physical_skin_registered_volume_import_detail::closePoint(bundle.currentWorldPoints[volumeNode],surface.worldPoints[row],worldLocalToleranceM))
                return fail("typed authoring surface row differs from its exact imported FEM boundary node");
            const auto& parentRow=importSidecars.currentParentRows[volumeNode];
            if(!parentRow.available||parentRow.tet>=bundle.parent.tetrahedra.size()||
               bundle.volumeNodeParentTetrahedra[volumeNode]!=parentRow.tet)
                return fail("authoring surface boundary node lost its exact parent-tet registration");
            const auto parentGeometricDonor=bundle.parent.donorBodyIndices[parentRow.tet];

            const auto& fullWeights=roleRows.fullBodyWeights86[row];
            std::size_t dominant=0u;double sum=0.0;
            for(std::size_t lane=0u;lane<fullWeights.size();++lane) {
                const double weight=fullWeights[lane];
                if(!std::isfinite(weight)||weight<0.0)
                    return fail("authoring full86 row contains a nonfinite or negative binding weight");
                sum+=weight;
                if(lane>0u&&weight>fullWeights[dominant])dominant=lane;
            }
            const double sumError=std::abs(sum-1.0);maxNormError=std::max(maxNormError,sumError);
            if(!std::isfinite(sum)||sumError>1.0e-4)
                return fail("authoring full86 row does not normalize");
            const auto frameOwner=bindingBodyByLane[dominant];
            if(frameOwner>=allBodyCurrentPoses.size()||surface.donorBodyIndices[row]!=frameOwner||
               bundle.volumeNodeDonorBodyIndices[volumeNode]!=frameOwner)
                return fail("registered authoring frame owner differs from the dominant interpolated full86 binding");
            const auto& pose=allBodyCurrentPoses[frameOwner];
            const auto& parsedFrame=registered.framesByDonor[frameOwner];
            const auto expectedW2L=expectedWorldToDonorLocal(pose);
            if(!parsedFrame.available||parsedFrame.binding!=dominant||
               !closeMatrix(surface.worldToDonorLocalLinear[row],expectedW2L,frameMatrixTolerance))
                return fail("authoring full86 binding lane has no matching registered current donor frame");
            const auto expectedLocal=expectedDonorLocal(pose,surface.worldPoints[row]);
            const double localError=norm(sub(expectedLocal,surface.donorLocalPoints[row]));
            maxLocalError=std::max(maxLocalError,localError);
            if(!std::isfinite(localError)||localError>worldLocalToleranceM)
                return fail("authoring current world/local coordinates do not reconstruct through the dominant binding pose");

            roleRows.authoringVertexRows[row]=static_cast<std::uint32_t>(source.vertex);
            roleRows.volumeNodeRows[row]=volumeNode;
            roleRows.dominantBindingIndices[row]=static_cast<std::uint32_t>(dominant);
            roleRows.frameOwnerBodyIndices[row]=frameOwner;
            roleRows.parentGeometricDonorBodyIndices[row]=parentGeometricDonor;
            if(frameOwner!=parentGeometricDonor)++measured.frameOwnerParentDonorDifferences[layer];
        }
        roleRows.sourceDomain=bundle.sourceDomains[surface.sourceDomainIndices.front()];
        for(const auto domainIndex:surface.sourceDomainIndices)
            if(bundle.sourceDomains[domainIndex]!=roleRows.sourceDomain)
                return fail("one authoring role surface mixes source namespace registrations");
        measured.rows[layer]=count;measured.faces[layer]=surface.faces.size();
        measured.maximumWeightNormalizationError[layer]=maxNormError;
        measured.maximumWorldLocalReconstructionErrorM[layer]=maxLocalError;
        measured.weightFileSHA256[layer]=weightsSHA;
        next.roles[layer]=std::move(roleRows);
    }
    next.physicalAdmission=false;measured.physicalAdmission=false;
    output=std::move(next);report=std::move(measured);error.clear();return true;
}

// Build a direct O render/motion-capture map after the import and full86
// consumer have passed. It contains no projection, offset, or NHSKIN identity
// conversion: authoring row i selects exactly the mapped imported volume node.
[[nodiscard]] inline bool preparePhysicalSkinAuthoringOuterRoleRenderDomain(
        const PhysicalSkinChildMassMappingBundle& bundle,
        const PhysicalSkinAuthoringRoleConsumerState& roleState,
        PhysicalSkinAuthoringOuterRoleRenderDomain& output,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(roleState.physicalAdmission||roleState.bindingBodyByLane.size()!=
           kPhysicalSkinAuthoringFullWeightBindingCount)
        return fail("direct O render mapping requires the non-admitting complete full86 role state");
    const auto& role=roleState.roles[0];
    const std::size_t count=role.authoringVertexRows.size();
    if(role.sourceDomain.role!=PhysicalSkinSourceDomainRole::outerSurface||
       role.sourceDomain.domain.empty()||
       !validPhysicalSkinSourceDomainSHA256(role.sourceDomain.sourceSHA256)||
       count==0u||role.sourceDomainIndices.size()!=count||
       role.volumeNodeRows.size()!=count||role.frameOwnerBodyIndices.size()!=count||
       role.currentWorldPoints.size()!=count||role.donorLocalPoints.size()!=count||
       role.fullBodyWeights86.size()!=count||role.dominantBindingIndices.size()!=count||
       role.parentGeometricDonorBodyIndices.size()!=count||
       role.roleLocalFaces.empty()||role.roleLocalFaces!=bundle.outerSkin.faces||
       bundle.outerSkin.worldPoints.size()!=count||
       bundle.outerSkin.sourceVertexIndices.size()!=count||
       bundle.outerSkin.sourceDomainIndices.size()!=count||
       bundle.boundaryMap.outer.size()!=count||
       bundle.volumeNodeSourceDomainIndices.size()!=bundle.currentWorldPoints.size()||
       bundle.volumeNodeSourceVertexIndices.size()!=bundle.currentWorldPoints.size()||
       bundle.volumeNodeSourceFaceIndices.size()!=bundle.currentWorldPoints.size()||
       bundle.volumeNodeDonorBodyIndices.size()!=bundle.currentWorldPoints.size())
        return fail("direct O render source, face, or volume-node rows are incomplete");
    std::uint32_t domainIndex=NM_INVALID_INDEX;
    for(std::size_t i=0u;i<bundle.sourceDomains.size();++i)
        if(bundle.sourceDomains[i]==role.sourceDomain) {
            if(domainIndex!=NM_INVALID_INDEX)
                return fail("direct O render source registration is duplicated");
            domainIndex=static_cast<std::uint32_t>(i);
        }
    if(domainIndex==NM_INVALID_INDEX)
        return fail("direct O render source registry is absent from the imported volume bundle");
    std::set<std::uint32_t> authoringRows,volumeRows;
    PhysicalSkinAuthoringOuterRoleRenderDomain next;
    next.sourceDomain=role.sourceDomain;
    next.authoringVertexRows=role.authoringVertexRows;
    next.volumeNodeRows=role.volumeNodeRows;
    next.frameOwnerBodyIndices=role.frameOwnerBodyIndices;
    next.worldPoints.reserve(count);next.transferRows.resize(count);
    for(std::size_t row=0u;row<count;++row) {
        const auto authoring=role.authoringVertexRows[row];
        const auto volume=role.volumeNodeRows[row];
        const auto owner=role.frameOwnerBodyIndices[row];
        const auto dominant=role.dominantBindingIndices[row];
        if(authoring>=role.sourceDomain.vertexCount||!authoringRows.insert(authoring).second||
           volume>=bundle.currentWorldPoints.size()||!volumeRows.insert(volume).second||
           volume!=bundle.boundaryMap.outer[row]||
           role.sourceDomainIndices[row]!=domainIndex||
           bundle.outerSkin.sourceDomainIndices[row]!=domainIndex)
            return fail("direct O authoring row does not select its unique exact role boundary row");
        if(bundle.outerSkin.sourceVertexIndices[row]!=authoring||
           bundle.volumeNodeSourceDomainIndices[volume]!=domainIndex||
           bundle.volumeNodeSourceVertexIndices[volume]!=authoring||
           bundle.volumeNodeSourceFaceIndices[volume]!=NM_INVALID_INDEX)
            return fail("direct O source registry differs from the imported volume-node identity");
        if(bundle.volumeNodeDonorBodyIndices[volume]!=owner||
           dominant>=roleState.bindingBodyByLane.size()||
           roleState.bindingBodyByLane[dominant]!=owner)
            return fail("direct O row uses a different source frame owner than its volume node");
        if(role.currentWorldPoints[row]!=bundle.outerSkin.worldPoints[row]||
           role.currentWorldPoints[row]!=bundle.currentWorldPoints[volume])
            return fail("direct O role geometry differs from its exact imported volume-node coordinate");
        for(const auto value:role.currentWorldPoints[row])
            if(!std::isfinite(value))
                return fail("direct O render source contains a nonfinite world point");
        for(const auto value:role.fullBodyWeights86[row])
            if(!std::isfinite(value)||value<0.0)
                return fail("direct O render source contains an invalid full86 row");
        next.worldPoints.push_back(role.currentWorldPoints[row]);
        auto& transfer=next.transferRows[row];
        transfer.coarseVertex[0]=transfer.coarseVertex[1]=transfer.coarseVertex[2]=
            static_cast<std::uint32_t>(row);
        transfer.donorBodyIndex=owner;
        transfer.weight[0]=1.0f;transfer.weight[1]=transfer.weight[2]=0.0f;
        transfer.restDistance=0.0f;
    }
    next.faces.reserve(role.roleLocalFaces.size());
    for(const auto& face:role.roleLocalFaces) {
        for(const auto vertex:face)
            if(vertex>=count)return fail("direct O render face exceeds its typed role-local vertex domain");
        next.faces.push_back(face);
    }
    next.physicalAdmission=false;
    output=std::move(next);error.clear();return true;
}

namespace physical_skin_authoring_detail_bridge_detail {
using TetRow=numi::human::AcceptedFEMSkinTransfer::GPUTetrahedralTransferRow;
using Transfer=numi::human::AcceptedFEMSkinTransfer;
inline bool finitePoint(const V3& p) {
    return std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);});
}
inline double distance(const V3& a,const V3& b) { return norm(sub(a,b)); }
inline bool validDomain(const PhysicalSkinSourceDomainRegistration& d) {
    return !d.domain.empty()&&validPhysicalSkinSourceDomainSHA256(d.sourceSHA256)&&
        d.vertexCount>0u&&d.faceCount>0u;
}
inline bool tetBarycentric(const std::vector<V3>& points,
        const std::array<std::uint32_t,4>& nodes,const V3& point,
        std::array<double,4>& bary,double& condition,std::string& error) {
    for(auto n:nodes)if(n>=points.size()||!finitePoint(points[n])) {
        error="selected physical tet has an invalid node row";return false;
    }
    if(nodes[0]==nodes[1]||nodes[0]==nodes[2]||nodes[0]==nodes[3]||
       nodes[1]==nodes[2]||nodes[1]==nodes[3]||nodes[2]==nodes[3]) {
        error="selected physical tet repeats a node row";return false;
    }
    const V3 e1=sub(points[nodes[1]],points[nodes[0]]);
    const V3 e2=sub(points[nodes[2]],points[nodes[0]]);
    const V3 e3=sub(points[nodes[3]],points[nodes[0]]);
    const double det=dot(e1,cross(e2,e3));
    if(!std::isfinite(det)||det<=1.0e-24) {
        error="selected physical reference tet is inverted or singular";return false;
    }
    const V3 r=sub(point,points[nodes[0]]);
    const V3 inv0=mul(cross(e2,e3),1.0/det);
    const V3 inv1=mul(cross(e3,e1),1.0/det);
    const V3 inv2=mul(cross(e1,e2),1.0/det);
    const double c0=dot(r,inv0),c1=dot(r,inv1),c2=dot(r,inv2);
    bary={1.0-c0-c1-c2,c0,c1,c2};
    const double matrixF2=dot(e1,e1)+dot(e2,e2)+dot(e3,e3);
    const double inverseF2=dot(inv0,inv0)+dot(inv1,inv1)+dot(inv2,inv2);
    condition=std::sqrt(matrixF2*inverseF2);
    if(!std::isfinite(condition)||!std::all_of(bary.begin(),bary.end(),
          [](double x){return std::isfinite(x);})) {
        error="selected physical tet has nonfinite barycentrics or conditioning";return false;
    }
    return true;
}
inline V3 weightedTetPoint(const std::vector<V3>& points,
        const std::array<std::uint32_t,4>& nodes,const std::array<double,4>& weights) {
    V3 p{};
    for(unsigned k=0;k<4;++k)for(unsigned a=0;a<3;++a)
        p[a]+=weights[k]*points[nodes[k]][a];
    return p;
}
inline double conservativeTetExtrapolationDistance(const V3& point,
        const std::vector<V3>& points,const std::array<std::uint32_t,4>& nodes,
        const std::array<double,4>& bary) {
    std::array<double,4> clipped{};double sum=0.0;
    for(unsigned k=0;k<4;++k){clipped[k]=std::clamp(bary[k],0.0,1.0);sum+=clipped[k];}
    if(!(sum>0.0)||!std::isfinite(sum))return std::numeric_limits<double>::infinity();
    for(double& w:clipped)w/=sum;
    return distance(point,weightedTetPoint(points,nodes,clipped));
}

inline bool strictChildTetEmbeddingForBoundaryFace(
        const PhysicalSkinAuthoringDetailBridgeInput& in,
        const std::size_t row,const std::size_t face) {
    const auto n=in.detailReferenceWorldPoints.size();
    if(row>=n||face>=in.physicalBoundaryFaceChildTetRows.size()||
       face>=in.physicalBoundaryFaceGeometricDonorBodyIndices.size()||
       face>=in.physicalBoundaryStableFaceIDs.size()||
       face>=in.physicalBoundaryFaceParentTetRows.size()||
       row>=in.detailMaterialIndices.size()||row>=in.detailBoundaryProjectionDonorBodyIndices.size()||
       row>=in.detailAnatomicalRegionIndices.size())return false;
    const auto child=in.physicalBoundaryFaceChildTetRows[face];
    if(child>=in.mechanicalTetNodeRows.size()||
       child>=in.mechanicalTetMaterialIndices.size()||
       child>=in.mechanicalTetGeometricDonorBodyIndices.size()||
       child>=in.mechanicalTetAnatomicalRegionIndices.size())return false;
    const auto expectedMaterial=in.detailMaterialIndices[row];
    const auto expectedDonor=in.detailBoundaryProjectionDonorBodyIndices[row];
    const auto expectedRegion=in.detailAnatomicalRegionIndices[row];
    if(in.mechanicalTetMaterialIndices[child]!=expectedMaterial||
       in.mechanicalTetGeometricDonorBodyIndices[child]!=expectedDonor||
       in.mechanicalTetAnatomicalRegionIndices[child]!=expectedRegion||
       in.physicalBoundaryFaceGeometricDonorBodyIndices[face]!=expectedDonor)return false;
    const auto parent=in.physicalBoundaryFaceParentTetRows[face];
    if(parent>=in.sourceParentTetStableSourceFaceIDs.size()||
       in.sourceParentTetStableSourceFaceIDs[parent]!=in.physicalBoundaryStableFaceIDs[face])return false;
    const auto& tet=in.mechanicalTetNodeRows[child];
    std::array<double,4> weights{};
    double condition=std::numeric_limits<double>::infinity();
    std::string baryError;
    if(!tetBarycentric(in.mechanicalReferenceWorldPoints,tet,
            in.detailReferenceWorldPoints[row],weights,condition,baryError)||
       condition>in.maximumTetConditionEstimate)return false;
    const double extrapolation=conservativeTetExtrapolationDistance(
        in.detailReferenceWorldPoints[row],in.mechanicalReferenceWorldPoints,tet,weights);
    if(!std::isfinite(extrapolation)||extrapolation>in.maximumTetExtrapolationDistanceM)return false;
    std::array<float,4> floatWeights{};
    double maxAbsWeight=0.0,floatSum=0.0;
    for(unsigned k=0;k<4u;++k) {
        maxAbsWeight=std::max(maxAbsWeight,std::abs(weights[k]));
        floatWeights[k]=static_cast<float>(weights[k]);
        floatSum+=double(floatWeights[k]);
    }
    if(maxAbsWeight>in.maximumAbsoluteEffectiveWeight||std::abs(floatSum-1.0)>2.0e-6)return false;
    const V3 reconstructed=weightedTetPoint(in.mechanicalReferenceWorldPoints,tet,
        {double(floatWeights[0]),double(floatWeights[1]),double(floatWeights[2]),double(floatWeights[3])});
    const double reconstruction=distance(reconstructed,in.detailReferenceWorldPoints[row]);
    return std::isfinite(reconstruction)&&
        reconstruction<=in.maximumReferenceReconstructionErrorM;
}
inline bool buildParentTetFaceAdjacency(
        const std::vector<std::array<std::uint32_t,4>>& tets,
        std::vector<std::vector<std::uint32_t>>& adjacency,std::string& error) {
    using FaceKey=std::array<std::uint32_t,3>;
    std::map<FaceKey,std::vector<std::uint32_t>> faceCells;
    adjacency.assign(tets.size(),{});
    for(std::size_t cell=0;cell<tets.size();++cell){
        const auto& tet=tets[cell];
        auto sorted=tet;std::sort(sorted.begin(),sorted.end());
        if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()){
            error="source parent tet topology repeats a node";return false;
        }
        for(unsigned omitted=0;omitted<4;++omitted){
            FaceKey face{};unsigned out=0;
            for(unsigned k=0;k<4;++k)if(k!=omitted)face[out++]=tet[k];
            std::sort(face.begin(),face.end());
            faceCells[face].push_back(static_cast<std::uint32_t>(cell));
        }
    }
    for(const auto& [face,cells]:faceCells){
        (void)face;
        if(cells.size()>2u){error="source parent topology has a nonmanifold tetra face";return false;}
        if(cells.size()==2u){
            adjacency[cells[0]].push_back(cells[1]);
            adjacency[cells[1]].push_back(cells[0]);
        }
    }
    for(auto& neighbors:adjacency){
        std::sort(neighbors.begin(),neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(),neighbors.end()),neighbors.end());
    }
    return true;
}
inline bool boundedSameRegionParentTetPath(
        const std::vector<std::vector<std::uint32_t>>& adjacency,
        const std::vector<std::uint32_t>& material,
        const std::vector<std::uint32_t>& region,
        const std::uint32_t start,const std::uint32_t target,
        const std::uint32_t expectedMaterial,const std::uint32_t expectedRegion,
        const std::uint32_t maximumHops,std::uint32_t& hops,
        std::vector<std::uint32_t>& path) {
    if(start>=adjacency.size()||target>=adjacency.size()||material.size()!=adjacency.size()||
       region.size()!=adjacency.size()||material[start]!=expectedMaterial||
       material[target]!=expectedMaterial||region[start]!=expectedRegion||
       region[target]!=expectedRegion)return false;
    std::vector<std::uint32_t> depth(adjacency.size(),NM_INVALID_INDEX);
    std::vector<std::uint32_t> parent(adjacency.size(),NM_INVALID_INDEX);
    std::vector<std::uint32_t> queue;queue.reserve(std::min<std::size_t>(adjacency.size(),512u));
    depth[start]=0u;queue.push_back(start);
    for(std::size_t cursor=0;cursor<queue.size();++cursor){
        const auto cell=queue[cursor];
        if(cell==target){
            hops=depth[cell];path.clear();
            for(std::uint32_t row=target;row!=NM_INVALID_INDEX;row=parent[row])path.push_back(row);
            std::reverse(path.begin(),path.end());return true;
        }
        if(depth[cell]>=maximumHops)continue;
        for(const auto next:adjacency[cell]){
            if(depth[next]!=NM_INVALID_INDEX||material[next]!=expectedMaterial||
               region[next]!=expectedRegion)continue;
            depth[next]=depth[cell]+1u;parent[next]=cell;queue.push_back(next);
        }
    }
    return false;
}

} // namespace physical_skin_authoring_detail_bridge_detail

// The canonical face producer and strict runtime consumer share this exact
// eligibility predicate. Frame ownership is validated independently from the
// geometric donor selected from the physical face.
struct PhysicalSkinAuthoringDetailFaceEligibilityContext {
    const PhysicalSkinAuthoringDetailBridgeInput* input=nullptr;
    const std::vector<std::vector<std::uint32_t>>* parentAdjacency=nullptr;
};
inline bool physicalSkinAuthoringDetailFaceEligible(
        const PhysicalSkinAuthoringDetailFaceEligibilityContext& context,
        const std::size_t row,const std::size_t face) {
    if(!context.input||!context.parentAdjacency)return false;
    const auto& in=*context.input;
    if(row>=in.detailReferenceWorldPoints.size()||
       face>=in.physicalBoundaryFaceParentTetRows.size()||
       face>=in.physicalBoundaryFaceMaterialIndices.size()||
       face>=in.physicalBoundaryFaceAnatomicalRegionIndices.size()||
       face>=in.physicalBoundaryFaceFrameOwnerBodyIndices.size()||
       in.physicalBoundaryFaceFrameOwnerBodyIndices[face]==NM_INVALID_INDEX||
       row>=in.detailMaterialIndices.size()||row>=in.detailAnatomicalRegionIndices.size()||
       in.detailMaterialIndices[row]!=in.physicalBoundaryFaceMaterialIndices[face]||
       in.detailAnatomicalRegionIndices[row]!=in.physicalBoundaryFaceAnatomicalRegionIndices[face])
        return false;
    const auto sourceTet=in.detailParentTetRows[row];
    const auto targetTet=in.physicalBoundaryFaceParentTetRows[face];
    if(sourceTet>=context.parentAdjacency->size()||
       targetTet>=in.sourceParentTetMaterialIndices.size()||
       in.sourceParentTetMaterialIndices.size()!=context.parentAdjacency->size()||
       in.sourceParentTetAnatomicalRegionIndices.size()!=context.parentAdjacency->size())
        return false;
    std::uint32_t hops=0u;std::vector<std::uint32_t> path;
    return physical_skin_authoring_detail_bridge_detail::boundedSameRegionParentTetPath(
        *context.parentAdjacency,in.sourceParentTetMaterialIndices,
        in.sourceParentTetAnatomicalRegionIndices,sourceTet,targetTet,
        in.detailMaterialIndices[row],in.detailAnatomicalRegionIndices[row],
        in.maximumSourceParentTetHops,hops,path);
}

struct PhysicalSkinAuthoringDetailStrictRescueEligibilityContext {
    const PhysicalSkinAuthoringDetailBridgeInput* input=nullptr;
    const std::vector<std::vector<std::uint32_t>>* parentAdjacency=nullptr;
    const std::vector<std::uint32_t>* sourceRows=nullptr;
};
inline bool physicalSkinAuthoringDetailStrictSelectionFaceEligible(
        const PhysicalSkinAuthoringDetailBridgeInput& input,
        const std::vector<std::vector<std::uint32_t>>& parentAdjacency,
        const std::size_t row,const std::size_t face) {
    const PhysicalSkinAuthoringDetailFaceEligibilityContext base{&input,&parentAdjacency};
    return physicalSkinAuthoringDetailFaceEligible(base,row,face)&&
        physical_skin_authoring_detail_bridge_detail::strictChildTetEmbeddingForBoundaryFace(
            input,row,face);
}

inline bool physicalSkinAuthoringDetailStrictRescueFaceEligible(
        const void* opaque,const std::size_t subsetRow,const std::size_t face) {
    const auto& context=*static_cast<const PhysicalSkinAuthoringDetailStrictRescueEligibilityContext*>(opaque);
    if(!context.input||!context.parentAdjacency||!context.sourceRows||
       subsetRow>=context.sourceRows->size())return false;
    return physicalSkinAuthoringDetailStrictSelectionFaceEligible(
        *context.input,*context.parentAdjacency,(*context.sourceRows)[subsetRow],face);
}

// The same strict source-only transfer is used by canonical export and by
// prepare-time recomputation. The row map keeps subset Transfer IDs separate
// from original detail rows while preserving identical eligibility and search
// semantics.
inline bool buildPhysicalSkinAuthoringStrictChildFaceTransfer(
        const PhysicalSkinAuthoringDetailBridgeInput& input,
        const std::vector<std::vector<std::uint32_t>>& parentAdjacency,
        const std::vector<std::uint32_t>& sourceRows,
        const std::vector<V3>& detailPoints,
        const std::vector<std::uint32_t>& detailSourceKeys,
        const std::vector<std::uint32_t>& detailDonors,
        const std::vector<std::uint8_t>& detailUsed,
        const std::vector<std::uint32_t>& boundarySourceKeys,
        std::vector<std::uint32_t>& representatives,
        numi::human::AcceptedFEMSkinTransfer& finalTransfer,
        std::size_t& representativeAdjustments,std::string& error) {
    using Transfer=numi::human::AcceptedFEMSkinTransfer;
    const auto fail=[&](const char* message){error=message;return false;};
    const std::size_t count=sourceRows.size();
    if(count==0u||detailPoints.size()!=count||detailSourceKeys.size()!=count||
       detailDonors.size()!=count||detailUsed.size()!=count||representatives.size()!=count||
       boundarySourceKeys.size()!=input.physicalBoundaryReferenceWorldPoints.size())
        return fail("strict source-only transfer subset arrays have inconsistent domains");
    for(const auto row:sourceRows)
        if(row>=input.detailReferenceWorldPoints.size())
            return fail("strict source-only transfer subset references an invalid detail row");
    PhysicalSkinAuthoringDetailStrictRescueEligibilityContext context{
        &input,&parentAdjacency,&sourceRows};
    representativeAdjustments=0u;
    constexpr unsigned maximumPasses=8u;
    for(unsigned pass=0u;pass<maximumPasses;++pass) {
        Transfer transfer;Transfer::BuildReport buildReport;std::string buildError;
        if(!transfer.build(detailPoints,detailSourceKeys,detailDonors,detailUsed,
                input.physicalBoundaryReferenceWorldPoints,boundarySourceKeys,
                input.physicalBoundaryGeometricDonorBodyIndices,input.physicalBoundaryFaces,
                &buildReport,&buildError,input.maximumBoundaryProjectionDistanceM,
                &representatives,nullptr,nullptr,nullptr,3u,
                &input.physicalBoundaryStableFaceIDs,nullptr,
                physicalSkinAuthoringDetailStrictRescueFaceEligible,&context)) {
            error="strict source-only child-face transfer rejected: "+buildError;return false;
        }
        std::vector<std::uint32_t> nextRepresentatives=representatives;
        for(std::size_t local=0u;local<count;++local) {
            const auto faceRow=transfer.selectedFaceIndex(local);
            if(faceRow>=input.physicalBoundaryFaces.size())
                return fail("strict source-only transfer returned an invalid face row");
            const auto& face=input.physicalBoundaryFaces[faceRow];
            if(std::find(face.begin(),face.end(),representatives[local])!=face.end())continue;
            const auto& transferRow=transfer.rows()[local];
            unsigned best=0u;
            for(unsigned corner=1u;corner<3u;++corner)
                if(transferRow.weight[corner]>transferRow.weight[best])best=corner;
            nextRepresentatives[local]=face[best];
        }
        std::size_t changed=0u;
        for(std::size_t local=0u;local<count;++local)
            changed+=nextRepresentatives[local]!=representatives[local];
        if(changed==0u) {
            finalTransfer=std::move(transfer);error.clear();return true;
        }
        representativeAdjustments+=changed;
        representatives.swap(nextRepresentatives);
    }
    return fail("strict source-only child-face representatives did not reach a fixed point in eight builds");
}

// Validate the producer's source-connected boundary correspondence against
// the existing native transfer builder, then compile a four-node affine
// embedding into the same physical tet rows consumed by accepted FEM state.
// This is a render/presentation bridge only; it never admits or changes physics.
[[nodiscard]] inline bool preparePhysicalSkinAuthoringDetailBridge(
        const PhysicalSkinAuthoringDetailBridgeInput& in,
        PhysicalSkinAuthoringDetailBridgeState& output,
        PhysicalSkinAuthoringDetailBridgeReport& report,
        std::string& error) {
    using namespace physical_skin_authoring_detail_bridge_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    std::string transferError;
    const std::size_t n=in.detailReferenceWorldPoints.size();
    const std::size_t b=in.physicalBoundaryReferenceWorldPoints.size();
    const std::size_t m=in.mechanicalReferenceWorldPoints.size();
    const std::size_t cells=in.mechanicalTetNodeRows.size();
    if(!validDomain(in.detailDomain)||!validDomain(in.physicalBoundaryDomain)||
       in.detailDomain.role!=in.physicalBoundaryDomain.role||n==0u||b<3u||m<4u||cells==0u)
        return fail("detail bridge needs pinned compatible source domains and nonempty bounded geometry");
    if(n>physical_skin_child_capacity_detail::maxRoleNodes||
       b>physical_skin_child_capacity_detail::maxRoleNodes||
       m>physical_skin_child_capacity_detail::maxMechanicalNodes||
       cells>physical_skin_child_capacity_detail::maxCells)
        return fail("detail bridge exceeds the existing bounded registered-volume row domain");
    if(in.detailDomain.vertexCount!=n||in.detailDomain.faceCount!=in.detailFaces.size()||
       in.physicalBoundaryDomain.vertexCount!=b||in.physicalBoundaryDomain.faceCount!=in.physicalBoundaryFaces.size())
        return fail("detail bridge source registration counts differ from the supplied geometry");
    if((!in.detailReferenceWorldNormals.empty()&&in.detailReferenceWorldNormals.size()!=n)||
       in.detailSourceVertexRows.size()!=n||in.detailGeometricDonorBodyIndices.size()!=n||
       (!in.detailRegisteredSourceDonorBodyIndices.empty()&&
        in.detailRegisteredSourceDonorBodyIndices.size()!=n)||
       in.detailBoundaryProjectionDonorBodyIndices.size()!=n||in.detailRegionScopedDonorFallback.size()!=n||
       in.detailFrameOwnerBodyIndices.size()!=n||in.detailParentTetRows.size()!=n||
       in.detailStableSourceFaceIDs.size()!=n||
       (!in.detailRequiresParentTetPath.empty()&&in.detailRequiresParentTetPath.size()!=n)||
       in.detailMaterialIndices.size()!=n||in.detailAnatomicalRegionIndices.size()!=n||
       in.physicalBoundarySourceVertexRows.size()!=b||
       in.physicalBoundaryGeometricDonorBodyIndices.size()!=b||
       in.physicalBoundaryFrameOwnerBodyIndices.size()!=b||
       in.physicalBoundaryVolumeNodeRows.size()!=b||
       in.physicalBoundaryFaces.empty()||
       in.physicalBoundaryStableFaceIDs.size()!=in.physicalBoundaryFaces.size()||
       in.physicalBoundaryFaceMaterialIndices.size()!=in.physicalBoundaryFaces.size()||
       in.physicalBoundaryFaceGeometricDonorBodyIndices.size()!=in.physicalBoundaryFaces.size()||
       in.physicalBoundaryFaceAnatomicalRegionIndices.size()!=in.physicalBoundaryFaces.size()||
       in.physicalBoundaryFaceFrameOwnerBodyIndices.size()!=in.physicalBoundaryFaces.size()||
       in.physicalBoundaryFaceParentTetRows.size()!=in.physicalBoundaryFaces.size()||
       in.detailRepresentativeBoundaryVertexRows.size()!=n||
       in.detailExpectedBoundaryFaceRows.size()!=n||
       in.detailExpectedBoundaryFaceBarycentrics.size()!=n||
       in.detailMechanicalTetRows.size()!=n||
       in.mechanicalTetMaterialIndices.size()!=cells||
       in.mechanicalTetGeometricDonorBodyIndices.size()!=cells||
       in.mechanicalTetAnatomicalRegionIndices.size()!=cells)
        return fail("detail bridge source, witness, cell-label, or volume-row arrays are incomplete");
    const std::array<double,6> limits{in.maximumBoundaryProjectionDistanceM,
        in.maximumTetExtrapolationDistanceM,in.maximumAbsoluteEffectiveWeight,
        in.maximumTetConditionEstimate,in.maximumReferenceReconstructionErrorM,
        in.faceBarycentricComparisonTolerance};
    if(!std::all_of(limits.begin(),limits.end(),[](double x){return std::isfinite(x)&&x>=0.0;})||
       in.maximumAbsoluteEffectiveWeight<1.0||in.maximumAbsoluteEffectiveWeight>16.0||
       in.maximumTetConditionEstimate<1.0||in.maximumBoundaryProjectionDistanceM>0.25||
       in.maximumTetExtrapolationDistanceM>0.05||in.maximumBoundaryProjectionDistanceM>0.05||
       in.faceBarycentricComparisonTolerance>1.0e-3)
        return fail("detail bridge projection, extrapolation, weight, or conditioning bound is invalid");
    if(in.detailFaces.empty()||in.physicalBoundaryFaces.empty())
        return fail("detail bridge requires detailed and physical boundary topology");
    for(const auto& p:in.detailReferenceWorldPoints)if(!finitePoint(p))
        return fail("detail bridge contains a nonfinite detailed world point");
    for(const auto& normal:in.detailReferenceWorldNormals)
        if(!finitePoint(normal)||std::abs(norm(normal)-1.0)>2.0e-5)
            return fail("detail bridge contains an invalid or unnormalized detailed normal");
    for(const auto& p:in.physicalBoundaryReferenceWorldPoints)if(!finitePoint(p))
        return fail("detail bridge contains a nonfinite physical boundary point");
    for(const auto& p:in.mechanicalReferenceWorldPoints)if(!finitePoint(p))
        return fail("detail bridge contains a nonfinite mechanical reference point");

    std::vector<std::uint8_t> used(n,0u);
    for(const auto& face:in.detailFaces)for(auto v:face){
        if(v>=n)return fail("detail bridge surface face exceeds its typed detail namespace");
        used[v]=1u;
    }
    if(std::find(used.begin(),used.end(),std::uint8_t(0u))!=used.end())
        return fail("detail bridge has a detail row outside its source surface topology");
    std::set<std::uint32_t> boundaryVolumeRows;
    for(std::size_t row=0;row<b;++row){
        if(in.physicalBoundarySourceVertexRows[row]>=in.physicalBoundaryDomain.vertexCount||
           in.physicalBoundaryGeometricDonorBodyIndices[row]==NM_INVALID_INDEX||
           in.physicalBoundaryFrameOwnerBodyIndices[row]==NM_INVALID_INDEX||
           in.physicalBoundaryVolumeNodeRows[row]>=m||
           !boundaryVolumeRows.insert(in.physicalBoundaryVolumeNodeRows[row]).second)
            return fail("physical boundary source or volume-node rows are invalid or duplicated");
    }
    for(const auto& face:in.physicalBoundaryFaces)for(auto v:face)
        if(v>=b)return fail("physical boundary face exceeds its role-local vertex namespace");
    bool needsParentAncestryPath=false;
    for(std::size_t i=0;i<n;++i){
        const auto geometricDonor=in.detailGeometricDonorBodyIndices[i];
        const auto mappedDonor=in.detailBoundaryProjectionDonorBodyIndices[i];
        const auto fallback=in.detailRegionScopedDonorFallback[i];
        const bool requiresParentPath=in.detailRequiresParentTetPath.empty()?
            fallback!=0u:in.detailRequiresParentTetPath[i]!=0u;
        if(geometricDonor==NM_INVALID_INDEX||mappedDonor==NM_INVALID_INDEX||
           (!in.detailRegisteredSourceDonorBodyIndices.empty()&&
            in.detailRegisteredSourceDonorBodyIndices[i]==NM_INVALID_INDEX)||
           in.detailFrameOwnerBodyIndices[i]==NM_INVALID_INDEX||fallback>1u||
           (!in.detailRequiresParentTetPath.empty()&&in.detailRequiresParentTetPath[i]>1u)||
           ((geometricDonor!=mappedDonor)!=(fallback!=0u))||
           (fallback!=0u&&!requiresParentPath))
            return fail("detail donor crosswalk or frame-owner identity is invalid or implicit");
        needsParentAncestryPath|=requiresParentPath;
    }
    std::vector<std::vector<std::uint32_t>> sourceParentTetAdjacency;
    std::vector<std::vector<std::uint32_t>> sourceParentTetPaths(n);
    std::vector<std::uint32_t> sourceParentTetHops(n,0u);
    const bool hasPinnedParentAncestryDomain=!in.sourceParentTetNodeRows.empty();
    if(needsParentAncestryPath||hasPinnedParentAncestryDomain){
        const std::size_t parentCells=in.sourceParentTetNodeRows.size();
        if(in.sourceParentNodeCount<4u||parentCells==0u||
           in.sourceParentTetMaterialIndices.size()!=parentCells||
           in.sourceParentTetAnatomicalRegionIndices.size()!=parentCells||
           in.sourceParentTetStableSourceFaceIDs.size()!=parentCells||
           in.maximumSourceParentTetHops==0u||in.maximumSourceParentTetHops>32u)
            return fail("source-connected detail face selection lacks a bounded pinned parent-tet ancestry domain");
        for(const auto& tet:in.sourceParentTetNodeRows)
            for(auto node:tet)if(node>=in.sourceParentNodeCount)
                return fail("source parent ancestry tet references a node outside its pinned parent domain");
        if(!buildParentTetFaceAdjacency(in.sourceParentTetNodeRows,
                sourceParentTetAdjacency,transferError))
            return fail("source parent ancestry topology is invalid: "+transferError);
    }
    for(std::size_t i=0;i<n;++i){
        if(in.detailSourceVertexRows[i]>=in.detailDomain.vertexCount||
           in.detailGeometricDonorBodyIndices[i]==NM_INVALID_INDEX||
           in.detailMaterialIndices[i]==NM_INVALID_INDEX||
           in.detailAnatomicalRegionIndices[i]==NM_INVALID_INDEX||
           in.detailRepresentativeBoundaryVertexRows[i]>=b||
           in.detailExpectedBoundaryFaceRows[i]>=in.physicalBoundaryFaces.size()||
           in.detailMechanicalTetRows[i]>=cells)
            return fail("detail bridge row has an invalid source, semantic, representative, face, or tet index");
        const auto faceRow=in.detailExpectedBoundaryFaceRows[i];
        if(in.physicalBoundaryFaceMaterialIndices[faceRow]!=in.detailMaterialIndices[i]||
           in.physicalBoundaryFaceGeometricDonorBodyIndices[faceRow]!=
               in.detailBoundaryProjectionDonorBodyIndices[i]||
           in.physicalBoundaryFaceAnatomicalRegionIndices[faceRow]!=in.detailAnatomicalRegionIndices[i]||
           in.physicalBoundaryFaceFrameOwnerBodyIndices[faceRow]==NM_INVALID_INDEX)
            return fail("source-connected boundary face has a different mapped donor, material, region, or frame owner");
        const bool requiresParentPath=in.detailRequiresParentTetPath.empty()?
            in.detailRegionScopedDonorFallback[i]!=0u:in.detailRequiresParentTetPath[i]!=0u;
        if(requiresParentPath){
            const auto sourceTet=in.detailParentTetRows[i];
            const auto targetTet=in.physicalBoundaryFaceParentTetRows[faceRow];
            const auto stableFace=in.detailStableSourceFaceIDs[i];
            if(sourceTet>=sourceParentTetAdjacency.size()||targetTet>=in.sourceParentTetNodeRows.size()||
               stableFace==NM_INVALID_INDEX||
               stableFace!=in.physicalBoundaryStableFaceIDs[faceRow]||
               in.sourceParentTetStableSourceFaceIDs[targetTet]!=stableFace)
                return fail("region-scoped donor crosswalk lacks matching source-face and parent-tet endpoint ancestry");
            std::uint32_t hops=0u;
            if(!boundedSameRegionParentTetPath(sourceParentTetAdjacency,
                    in.sourceParentTetMaterialIndices,in.sourceParentTetAnatomicalRegionIndices,
                    sourceTet,targetTet,in.detailMaterialIndices[i],
                    in.detailAnatomicalRegionIndices[i],in.maximumSourceParentTetHops,hops,
                    sourceParentTetPaths[i]))
                return fail("region-scoped donor crosswalk is outside its same-material/region source-tet patch");
            sourceParentTetHops[i]=hops;
        }
        const auto representative=in.detailRepresentativeBoundaryVertexRows[i];
        const auto& expectedFace=in.physicalBoundaryFaces[faceRow];
        if(std::find(expectedFace.begin(),expectedFace.end(),representative)==expectedFace.end())
            return fail("source-connected representative is outside its witnessed physical face");
        const auto& bary=in.detailExpectedBoundaryFaceBarycentrics[i];
        double sum=0.0;
        for(double w:bary) {
            if(!std::isfinite(w)||w< -1.0e-8||w>1.0+1.0e-8)
                return fail("detail bridge face witness has an out-of-face barycentric");
            sum+=w;
        }
        if(std::abs(sum-1.0)>1.0e-8)
            return fail("detail bridge face witness barycentrics do not sum to one");
        const V3 q=add(add(mul(in.physicalBoundaryReferenceWorldPoints[expectedFace[0]],bary[0]),
                           mul(in.physicalBoundaryReferenceWorldPoints[expectedFace[1]],bary[1])),
                       mul(in.physicalBoundaryReferenceWorldPoints[expectedFace[2]],bary[2]));
        if(distance(in.detailReferenceWorldPoints[i],q)>in.maximumBoundaryProjectionDistanceM)
            return fail("source-connected boundary witness exceeds its declared projection distance");
    }

    // A direct identity requires the complete pinned descriptor, indexed
    // topology, source rows, donor labels, and coordinates to agree. Numeric
    // row equality alone is explicitly insufficient.
    bool directIdentity=in.detailDomain==in.physicalBoundaryDomain&&n==b&&
        in.detailFaces==in.physicalBoundaryFaces&&
        in.detailSourceVertexRows==in.physicalBoundarySourceVertexRows&&
        in.detailGeometricDonorBodyIndices==in.physicalBoundaryGeometricDonorBodyIndices&&
        in.detailBoundaryProjectionDonorBodyIndices==in.detailGeometricDonorBodyIndices&&
        std::all_of(in.detailRegionScopedDonorFallback.begin(),in.detailRegionScopedDonorFallback.end(),
                    [](auto fallback){return fallback==0u;})&&
        (in.detailRequiresParentTetPath.empty()||
         std::all_of(in.detailRequiresParentTetPath.begin(),in.detailRequiresParentTetPath.end(),
                    [](auto path){return path==0u;}));
    if(directIdentity){
        std::set<std::uint32_t> sourceRows;
        for(std::size_t i=0;i<n;++i){
            directIdentity&=sourceRows.insert(in.detailSourceVertexRows[i]).second&&
                in.detailReferenceWorldPoints[i]==in.physicalBoundaryReferenceWorldPoints[i]&&
                in.physicalBoundaryVolumeNodeRows[i]<m&&
                in.mechanicalReferenceWorldPoints[in.physicalBoundaryVolumeNodeRows[i]]==
                    in.detailReferenceWorldPoints[i];
        }
    }

    std::vector<std::uint32_t> detailTransferSource(n),boundaryTransferSource(b,NM_INVALID_INDEX);
    if(directIdentity){
        detailTransferSource=in.detailSourceVertexRows;
        boundaryTransferSource=in.physicalBoundarySourceVertexRows;
    }else{
        // Use a disjoint synthetic key for detail rows and no physical source
        // keys. This prevents AcceptedFEMSkinTransfer's direct-source fast path
        // from treating equal integers in different domains as identity.
        for(std::size_t i=0;i<n;++i)
            detailTransferSource[i]=0x80000000u+static_cast<std::uint32_t>(i);
    }
    Transfer faceTransfer;
    Transfer::BuildReport faceReport;
    PhysicalSkinAuthoringDetailFaceEligibilityContext faceEligibilityContext{
        &in,&sourceParentTetAdjacency};
    Transfer::DetailFaceEligibilityPredicate faceEligibility=nullptr;
    if(hasPinnedParentAncestryDomain)
        faceEligibility=+[](const void* opaque,std::size_t row,std::size_t face)->bool {
            return physicalSkinAuthoringDetailFaceEligible(
                *static_cast<const PhysicalSkinAuthoringDetailFaceEligibilityContext*>(opaque),
                row,face);
        };
    if(!faceTransfer.build(in.detailReferenceWorldPoints,detailTransferSource,
            in.detailBoundaryProjectionDonorBodyIndices,used,
            in.physicalBoundaryReferenceWorldPoints,boundaryTransferSource,
            in.physicalBoundaryGeometricDonorBodyIndices,in.physicalBoundaryFaces,
            &faceReport,&transferError,in.maximumBoundaryProjectionDistanceM,
            &in.detailRepresentativeBoundaryVertexRows,nullptr,nullptr,nullptr,1u,
            &in.physicalBoundaryStableFaceIDs,nullptr,faceEligibility,
            hasPinnedParentAncestryDomain?&faceEligibilityContext:nullptr))
        return fail("native source-connected boundary transfer rejected: "+transferError);
    if((directIdentity&&faceReport.exactAncestryRows!=n)||
       (!directIdentity&&faceReport.exactAncestryRows!=0u))
        return fail("native transfer ancestry mode differs from the proven source-domain identity");
    std::vector<std::uint32_t> selectedFaceRows(n);
    std::vector<Transfer::GPUTransferRow> selectedFaceTransferRows=faceTransfer.rows();
    for(std::size_t i=0u;i<n;++i)selectedFaceRows[i]=faceTransfer.selectedFaceIndex(i);
    if(in.allowSourceOnlyStrictChildFaceSelection) {
        if(!hasPinnedParentAncestryDomain||
           in.detailRegisteredSourceDonorBodyIndices.size()!=n)
            return fail("source-only canonical recomputation lacks its authenticated strict-child selection domain");
        std::vector<std::uint32_t> strictRows;
        for(std::size_t i=0u;i<n;++i) {
            if(selectedFaceRows[i]==in.detailExpectedBoundaryFaceRows[i])continue;
            const auto expectedFace=in.detailExpectedBoundaryFaceRows[i];
            if(expectedFace>=in.physicalBoundaryFaces.size()||
               !physicalSkinAuthoringDetailStrictSelectionFaceEligible(
                    in,sourceParentTetAdjacency,i,expectedFace)) {
                std::ostringstream witness;
                witness<<"source-only expected-face witness is not strict-eligible at row="<<i
                    <<" expected_face="<<expectedFace<<" base_face="<<selectedFaceRows[i]
                    <<" source_parent="<<in.detailParentTetRows[i]
                    <<" source_stable="<<in.detailStableSourceFaceIDs[i]
                    <<" expected_parent="<<(expectedFace<in.physicalBoundaryFaceParentTetRows.size()?
                        in.physicalBoundaryFaceParentTetRows[expectedFace]:NM_INVALID_INDEX)
                    <<" expected_stable="<<(expectedFace<in.physicalBoundaryStableFaceIDs.size()?
                        in.physicalBoundaryStableFaceIDs[expectedFace]:NM_INVALID_INDEX)
                    <<" material="<<in.detailMaterialIndices[i]
                    <<" region="<<in.detailAnatomicalRegionIndices[i]
                    <<" donor="<<in.detailBoundaryProjectionDonorBodyIndices[i];
                return fail(witness.str());
            }
            strictRows.push_back(static_cast<std::uint32_t>(i));
        }
        if(!strictRows.empty()) {
            std::vector<V3> subsetPoints;std::vector<std::uint32_t> subsetSources,subsetDonors,
                subsetRepresentatives;std::vector<std::uint8_t> subsetUsed;
            subsetPoints.reserve(strictRows.size());subsetSources.reserve(strictRows.size());
            subsetDonors.reserve(strictRows.size());subsetRepresentatives.reserve(strictRows.size());
            subsetUsed.reserve(strictRows.size());
            for(const auto row:strictRows) {
                subsetPoints.push_back(in.detailReferenceWorldPoints[row]);
                subsetSources.push_back(detailTransferSource[row]);
                subsetDonors.push_back(in.detailRegisteredSourceDonorBodyIndices[row]);
                subsetRepresentatives.push_back(in.detailRepresentativeBoundaryVertexRows[row]);
                subsetUsed.push_back(used[row]);
            }
            Transfer strictTransfer;std::size_t strictRepresentativeAdjustments=0u;
            if(!buildPhysicalSkinAuthoringStrictChildFaceTransfer(
                    in,sourceParentTetAdjacency,strictRows,subsetPoints,subsetSources,subsetDonors,
                    subsetUsed,boundaryTransferSource,subsetRepresentatives,strictTransfer,
                    strictRepresentativeAdjustments,transferError))
                return fail("native source-only strict face selection could not reproduce its pinned witness: "+transferError);
            for(std::size_t local=0u;local<strictRows.size();++local) {
                const auto row=strictRows[local];
                if(subsetRepresentatives[local]!=in.detailRepresentativeBoundaryVertexRows[row])
                    return fail("native source-only strict selection changed the pinned representative at row "+
                        std::to_string(row));
                selectedFaceRows[row]=strictTransfer.selectedFaceIndex(local);
                selectedFaceTransferRows[row]=strictTransfer.rows()[local];
            }
        }
    }
    for(std::size_t i=0;i<n;++i){
        const auto faceRow=selectedFaceRows[i];
        if(faceRow!=in.detailExpectedBoundaryFaceRows[i]) {
            std::ostringstream witness;
            witness<<"native source-connected face differs from its pinned producer witness at row="<<i
                <<" selected_face="<<faceRow<<" expected_face="<<in.detailExpectedBoundaryFaceRows[i]
                <<" source_parent="<<in.detailParentTetRows[i]
                <<" source_stable="<<in.detailStableSourceFaceIDs[i]
                <<" expected_parent="<<in.physicalBoundaryFaceParentTetRows[in.detailExpectedBoundaryFaceRows[i]]
                <<" expected_stable="<<in.physicalBoundaryStableFaceIDs[in.detailExpectedBoundaryFaceRows[i]]
                <<" material="<<in.detailMaterialIndices[i]
                <<" region="<<in.detailAnatomicalRegionIndices[i]
                <<" donor="<<in.detailBoundaryProjectionDonorBodyIndices[i];
            return fail(witness.str());
        }
        const auto& row=selectedFaceTransferRows[i];
        const auto& expectedFace=in.physicalBoundaryFaces[faceRow];
        const auto& expectedBary=in.detailExpectedBoundaryFaceBarycentrics[i];
        for(unsigned k=0;k<3;++k)
            if(row.coarseVertex[k]!=expectedFace[k]||
               std::abs(double(row.weight[k])-expectedBary[k])>in.faceBarycentricComparisonTolerance)
                return fail("native source-connected face or barycentrics differ from the pinned producer witness at row "+
                    std::to_string(i));
        if(!std::isfinite(row.restDistance)||row.restDistance>in.maximumBoundaryProjectionDistanceM)
            return fail("native source-connected face projection exceeds its declared bound at row "+
                std::to_string(i));
    }

    PhysicalSkinAuthoringDetailBridgeState next;
    PhysicalSkinAuthoringDetailBridgeReport measured;
    next.detailDomain=in.detailDomain;next.physicalBoundaryDomain=in.physicalBoundaryDomain;
    next.provenDirectIdentity=directIdentity;
    next.faceTransferRows=std::move(selectedFaceTransferRows);
    next.selectedPhysicalBoundaryFaceRows.resize(n);
    next.detailGeometricDonorBodyIndices=in.detailGeometricDonorBodyIndices;
    next.detailFrameOwnerBodyIndices=in.detailFrameOwnerBodyIndices;
    next.selectedPhysicalBoundaryFaceDonorBodyIndices.resize(n);
    next.selectedPhysicalBoundaryFaceFrameOwnerBodyIndices.resize(n);
    next.regionScopedDonorFallbackRows=in.detailRegionScopedDonorFallback;
    next.sourceParentTetHopCounts=sourceParentTetHops;
    next.sourceParentTetPathRows=sourceParentTetPaths;
    next.tetraRows.resize(n);
    measured.detailRows=n;measured.boundaryRows=b;measured.physicalNodeRows=m;
    measured.surfaceProjectedRows=faceReport.projectedRows;
    measured.surfaceExactAncestryRows=faceReport.exactAncestryRows;
    measured.maximumBoundaryProjectionDistanceM=0.0;
    for(const auto& transferRow:next.faceTransferRows)
        measured.maximumBoundaryProjectionDistanceM=std::max(
            measured.maximumBoundaryProjectionDistanceM,double(transferRow.restDistance));
    measured.sourceNamespaceIdentitySuppressed=!directIdentity;
    for(std::size_t i=0;i<n;++i){
        const auto faceRow=selectedFaceRows[i];
        next.selectedPhysicalBoundaryFaceRows[i]=faceRow;
        next.selectedPhysicalBoundaryFaceDonorBodyIndices[i]=
            in.physicalBoundaryFaceGeometricDonorBodyIndices[faceRow];
        next.selectedPhysicalBoundaryFaceFrameOwnerBodyIndices[i]=
            in.physicalBoundaryFaceFrameOwnerBodyIndices[faceRow];
        const bool requiresParentPath=in.detailRequiresParentTetPath.empty()?
            in.detailRegionScopedDonorFallback[i]!=0u:in.detailRequiresParentTetPath[i]!=0u;
        if(requiresParentPath){
            measured.maximumSourceParentTetHops=std::max(
                measured.maximumSourceParentTetHops,next.sourceParentTetHopCounts[i]);
            ++measured.sourceParentTetWalkRows;
        }
        if(in.detailRegionScopedDonorFallback[i])++measured.regionScopedDonorFallbackRows;
        const auto cell=in.detailMechanicalTetRows[i];
        const auto& tet=in.mechanicalTetNodeRows[cell];
        if(in.mechanicalTetMaterialIndices[cell]!=in.detailMaterialIndices[i]||
           in.mechanicalTetGeometricDonorBodyIndices[cell]!=in.detailBoundaryProjectionDonorBodyIndices[i]||
           in.mechanicalTetAnatomicalRegionIndices[cell]!=in.detailAnatomicalRegionIndices[i])
            return fail("detail tet embedding crosses its declared material, geometric donor, or anatomical region");
        auto& row=next.tetraRows[i];
        if(directIdentity){
            const auto node=in.physicalBoundaryVolumeNodeRows[i];
            row.node[0]=row.node[1]=row.node[2]=row.node[3]=node;
            row.weight[0]=1.0f;row.weight[1]=row.weight[2]=row.weight[3]=0.0f;
            ++measured.directIdentityRows;
        }else{
            std::array<double,4> weights{};double condition=0.0;
            if(!tetBarycentric(in.mechanicalReferenceWorldPoints,tet,
                    in.detailReferenceWorldPoints[i],weights,condition,transferError))
                return fail("detail-to-volume affine tet embedding rejected: "+transferError);
            if(condition>in.maximumTetConditionEstimate)
                return fail("detail-to-volume tet exceeds its registered conditioning bound");
            measured.maximumTetConditionEstimate=std::max(measured.maximumTetConditionEstimate,condition);
            const double extrapolation=conservativeTetExtrapolationDistance(
                in.detailReferenceWorldPoints[i],in.mechanicalReferenceWorldPoints,tet,weights);
            if(!std::isfinite(extrapolation)||extrapolation>in.maximumTetExtrapolationDistanceM)
                return fail("detail-to-volume tet extrapolation exceeds its declared distance bound");
            measured.maximumTetExtrapolationDistanceM=
                std::max(measured.maximumTetExtrapolationDistanceM,extrapolation);
            double sum=0.0;
            for(unsigned k=0;k<4;++k){
                if(std::abs(weights[k])>in.maximumAbsoluteEffectiveWeight)
                    return fail("detail-to-volume effective tet weight exceeds its declared bound");
                row.node[k]=tet[k];row.weight[k]=static_cast<float>(weights[k]);
                measured.maximumAbsoluteEffectiveWeight=std::max(
                    measured.maximumAbsoluteEffectiveWeight,std::abs(double(row.weight[k])));
                sum+=double(row.weight[k]);
            }
            if(std::abs(sum-1.0)>2.0e-6)
                return fail("float tetrahedral transfer weights do not preserve affine translations");
            const V3 reconstructed=weightedTetPoint(in.mechanicalReferenceWorldPoints,tet,
                {double(row.weight[0]),double(row.weight[1]),double(row.weight[2]),double(row.weight[3])});
            const double reconstructionError=distance(reconstructed,in.detailReferenceWorldPoints[i]);
            measured.maximumReferenceReconstructionErrorM=std::max(
                measured.maximumReferenceReconstructionErrorM,reconstructionError);
            if(!std::isfinite(reconstructionError)||
               reconstructionError>in.maximumReferenceReconstructionErrorM)
                return fail("float tetrahedral transfer row does not reproduce its pinned detail reference point");
            ++measured.tetraEmbeddedRows;
        }
    }
    measured.physicalAdmission=false;next.physicalAdmission=false;
    output=std::move(next);report=std::move(measured);error.clear();return true;
}

// CPU reference used by the focused fixture to check the prepared four-node
// map. Production updates are performed by the accepted-state Metal owner.
[[nodiscard]] inline bool evaluatePhysicalSkinAuthoringDetailBridgeForCheck(
        const PhysicalSkinAuthoringDetailBridgeState& state,
        const std::vector<V3>& acceptedMechanicalWorldPoints,
        std::vector<V3>& output,std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(state.physicalAdmission||state.tetraRows.empty())
        return fail("detail bridge reference evaluator requires a prepared non-admitting state");
    std::vector<V3> next(state.tetraRows.size());
    for(std::size_t i=0;i<state.tetraRows.size();++i){
        V3 p{};double sum=0.0;
        for(unsigned k=0;k<4;++k){
            const auto node=state.tetraRows[i].node[k];
            const double weight=state.tetraRows[i].weight[k];
            if(node>=acceptedMechanicalWorldPoints.size()||
               !physical_skin_authoring_detail_bridge_detail::finitePoint(acceptedMechanicalWorldPoints[node])||
               !std::isfinite(weight))
                return fail("detail bridge reference evaluator received an invalid accepted node row");
            sum+=weight;
            for(unsigned a=0;a<3;++a)p[a]+=weight*acceptedMechanicalWorldPoints[node][a];
        }
        if(std::abs(sum-1.0)>2.0e-6||
           !physical_skin_authoring_detail_bridge_detail::finitePoint(p))
            return fail("detail bridge reference evaluator found a non-affine or nonfinite row");
        next[i]=p;
    }
    output=std::move(next);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
