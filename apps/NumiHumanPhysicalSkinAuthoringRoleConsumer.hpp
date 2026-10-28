#pragma once

// Initialization bridge for Boolean/derived O/I/D role surfaces. Authoring
// vertex IDs stay in their typed role namespace; they never enter the legacy
// NHSKIN direct-source transfer arrays.
#include "NumiHumanPhysicalSkinChildRegisteredVolumeImport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
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
       expectedRows==0u||expectedRows>200000u)
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

inline V3 expectedDonorLocal(const PhysicalSkinRegisteredBodyPose& pose,const V3& world) {
    const V3 delta=sub(world,pose.currentOrigin);
    const auto& m=pose.currentLocalToWorld;
    return {m[0]*delta[0]+m[3]*delta[1]+m[6]*delta[2],
            m[1]*delta[0]+m[4]*delta[1]+m[7]*delta[2],
            m[2]*delta[0]+m[5]*delta[1]+m[8]*delta[2]};
}

inline std::array<double,9> expectedWorldToDonorLocal(const PhysicalSkinRegisteredBodyPose& pose) {
    const auto& r=pose.currentLocalToWorld;
    return {r[0],r[3],r[6],r[1],r[4],r[7],r[2],r[5],r[8]};
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

} // namespace metalrobo::human::coarse_tissue
