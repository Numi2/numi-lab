#include "NumiHumanPhysicalSkinChildIntegration.hpp"

#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {

std::filesystem::path makePacket(const std::filesystem::path& directory, double referenceOffset=0.005) {
    const auto mass=directory/"compiled-parent-cell-mass.csv";
    const auto nodes=std::filesystem::path(mass.string()+".nodes.csv");
    const auto sources=std::filesystem::path(mass.string()+".source.csv");
    const auto tets=std::filesystem::path(mass.string()+".tets.csv");
    const std::uint32_t bits=std::bit_cast<std::uint32_t>(25.0f);
    {
        std::ofstream out(mass);
        out<<"source_tet_row,material_index,mass_kg\n0,0,100\n";
    }
    {
        std::ofstream out(nodes);
        out<<"parent_node_index,compiled_global_node,current_x_m,current_y_m,current_z_m,"
              "rest_x_m,rest_y_m,rest_z_m,compiled_lumped_mass_kg,node_frame_owner_body,"
              "donor_local_x_m,donor_local_y_m,donor_local_z_m,mass_bits\n";
        const std::array<V3,4> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
        for(std::uint32_t node=0;node<4u;++node){
            const auto& x=p[node];
            out<<node<<','<<(10u+node)<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','
               <<x[0]<<','<<x[1]<<','<<x[2]-referenceOffset<<",25,7,"
               <<x[0]<<','<<x[1]<<','<<x[2]<<','<<bits<<'\n';
        }
    }
    {
        std::ofstream out(sources);
        out<<"object_local_node,material_index,donor_body,mass_kg,local_x_m,local_y_m,local_z_m\n";
        const std::array<V3,4> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
        for(std::uint32_t node=0;node<4u;++node)
            out<<node<<",0,7,25,"<<p[node][0]<<','<<p[node][1]<<','<<p[node][2]<<'\n';
    }
    {
        std::ofstream out(tets);
        out<<"parent_tet_row,material_index,donor_body,parent_node0,parent_node1,"
              "parent_node2,parent_node3,parent_reference_volume_m3,parent_mass_kg\n"
           <<"0,0,7,0,1,2,3,0.16666666666666666,100\n";
    }
    return mass;
}

void fillOneChild(PhysicalSkinChildMassMappingBundle& bundle) {
    const std::array<V3,4> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
    bundle.currentWorldPoints.assign(p.begin(),p.end());
    bundle.referenceWorldPoints.assign(p.begin(),p.end());
    bundle.childTetrahedra.push_back({0u,1u,2u,3u});
    bundle.childMaterialIndices.push_back(0u);
    bundle.childDonorBodyIndices.push_back(7u);
    bundle.childAnatomicalRegionIndices.push_back(3u);
    bundle.childStableSourceFaceIdentities.push_back(901u);
    bundle.childMeshSHA256=std::string(64u,'a');
    bundle.regionLabelSHA256=std::string(64u,'b');
    bundle.coordinateFrameSHA256=std::string(64u,'c');
    bundle.maximumSpatialMassTransportDistanceM=1.0;
}

void fillTwoConformingChildren(PhysicalSkinChildMassMappingBundle& bundle) {
    const std::array<V3,5> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}},{{0.5,0,0}}}};
    bundle.currentWorldPoints.assign(p.begin(),p.end());
    bundle.referenceWorldPoints.assign(p.begin(),p.end());
    bundle.childTetrahedra={{0u,4u,2u,3u},{4u,1u,2u,3u}};
    bundle.childMaterialIndices={0u,0u};
    bundle.childDonorBodyIndices={7u,7u};
    bundle.childAnatomicalRegionIndices={3u,3u};
    bundle.childStableSourceFaceIdentities={901u,901u};
    bundle.childMeshSHA256=std::string(64u,'a');
    bundle.regionLabelSHA256=std::string(64u,'b');
    bundle.coordinateFrameSHA256=std::string(64u,'c');
    bundle.maximumSpatialMassTransportDistanceM=1.0;
}

} // namespace

int main() {
    {
        SourceCellCarrierTransportTrace skinTrace,fatTrace,pairedTrace;
        skinTrace.childParentCarrierOffsets={0u,1u};skinTrace.childParentCarrierRows={0u};
        skinTrace.childParentCarrierMassKg={0.6};skinTrace.childParentCarrierMeasureM3={0.0006};
        skinTrace.childParentCarrierDistanceM={0.001};
        fatTrace.childParentCarrierOffsets={0u,1u};fatTrace.childParentCarrierRows={0u};
        fatTrace.childParentCarrierMassKg={0.4};fatTrace.childParentCarrierMeasureM3={0.0004};
        fatTrace.childParentCarrierDistanceM={0.002};
        pairedTrace.childParentCarrierOffsets={0u};
        const std::vector<std::uint32_t> skinVirtualToCurrent{0u},fatVirtualToCurrent{1u};
        std::string mergeError;
        if(!physical_skin_child_mass_mapping_detail::appendRecordedCarrierFlowEdge(
               skinTrace,0u,skinVirtualToCurrent,pairedTrace,mergeError)||
           !physical_skin_child_mass_mapping_detail::appendRecordedCarrierFlowEdge(
               fatTrace,0u,fatVirtualToCurrent,pairedTrace,mergeError)||
           pairedTrace.childParentCarrierOffsets!=std::vector<std::uint32_t>{0u,1u,2u}||
           pairedTrace.childParentCarrierRows!=std::vector<std::uint32_t>{0u,1u}||
           pairedTrace.childParentCarrierMassKg!=std::vector<double>{0.6,0.4}){
            std::cerr<<"two-role recorded carrier CSR merge failed: "<<mergeError<<"\n";return 80;
        }
        const double tinyMass=1.0e-22;
        ConservativeSpatialMassTransportResult roleLedger,canonicalLedger;
        roleLedger.childSourceDonorOffsets={0u,1u};roleLedger.childSourceDonorBodyIndices={145u};
        roleLedger.childSourceDonorMassKg={tinyMass};
        canonicalLedger=roleLedger;
        auto repinnedTamper=roleLedger;repinnedTamper.childSourceDonorMassKg[0]=1.01e-22;
        auto zeroedTiny=roleLedger;zeroedTiny.childSourceDonorMassKg[0]=0.0;
        if(!physical_skin_child_mass_mapping_detail::sourceCarrierRelativeMassClose(tinyMass,tinyMass)||
           physical_skin_child_mass_mapping_detail::sourceCarrierRelativeMassClose(0.0,tinyMass)||
           physical_skin_child_mass_mapping_detail::sourceCarrierRelativeMassClose(1.01e-22,tinyMass)||
           !physical_skin_child_mass_mapping_detail::recordedRoleChildDonorLedgerMatches(
               roleLedger,0u,canonicalLedger,0u)||
           physical_skin_child_mass_mapping_detail::recordedRoleChildDonorLedgerMatches(
               repinnedTamper,0u,canonicalLedger,0u)||
           physical_skin_child_mass_mapping_detail::recordedRoleChildDonorLedgerMatches(
               zeroedTiny,0u,canonicalLedger,0u)){
            std::cerr<<"relative-only recorded carrier mass closure mishandled a 1e-22 kg child\\n";return 82;
        }
        auto invalidTrace=fatTrace;invalidTrace.childParentCarrierOffsets={0u,0u};
        const auto prefix=pairedTrace.childParentCarrierRows;
        if(physical_skin_child_mass_mapping_detail::appendRecordedCarrierFlowEdge(
               invalidTrace,0u,fatVirtualToCurrent,pairedTrace,mergeError)||
           pairedTrace.childParentCarrierRows!=prefix){
            std::cerr<<"paired carrier CSR merge accepted an empty/malformed role edge\n";return 81;
        }
    }
    // Reference and current are distinct even for an unchanged topology.
    // This catches accidentally using a rest .node file as the current shape.
    std::string identityFrameError;
    std::vector<V3> identityRest{{0.25,0.5,0.75},{1.0,0.0,0.0}};
    std::vector<V3> identityCurrent{{0.375,0.5,0.75},{1.125,0.0,0.0}};
    ExplicitUnpairedVolumeTopology identityFrame;
    identityFrame.referenceWorldPoints=identityRest;
    identityFrame.currentWorldPoints=identityCurrent;
    PhysicalSkinChildMassMappingBundle identityCoordinates;
    identityCoordinates.referenceWorldPoints=identityRest;
    identityCoordinates.currentWorldPoints=identityCurrent;
    if(!physical_skin_child_integration_detail::identityImportedNodeRowsMatch(
            identityCoordinates,identityFrame,identityFrameError)) {
        std::cerr<<"separate identity reference/current coordinate fixture rejected: "
                 <<identityFrameError<<'\n';return 35;
    }
    identityCoordinates.currentWorldPoints=identityRest;
    if(physical_skin_child_integration_detail::identityImportedNodeRowsMatch(
            identityCoordinates,identityFrame,identityFrameError)) {
        std::cerr<<"identity bridge accepted reference rows as current state\n";return 36;
    }
    identityCoordinates.currentWorldPoints=identityCurrent;
    identityCoordinates.referenceWorldPoints=identityCurrent;
    if(physical_skin_child_integration_detail::identityImportedNodeRowsMatch(
            identityCoordinates,identityFrame,identityFrameError)) {
        std::cerr<<"identity bridge accepted current rows as reference state\n";return 37;
    }

    const auto directory=std::filesystem::temp_directory_path()/
        ("numi-physical-child-mass-check-"+
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    const auto massCSV=makePacket(directory);

    PhysicalSkinParentMassPacket packet;
    std::string error;
    if(!loadCompiledSkinFatParentMassPacketCSV(massCSV,packet,error)){
        std::cerr<<"parent CSV fixture rejected: "<<error<<'\n';return 2;
    }
    if(packet.tetrahedra.size()!=1u||packet.referenceWorldPoints.size()!=4u||
       packet.sourceMassContributions.size()!=4u||packet.totalMassKg!=100.0||
       packet.maximumNodalMassClosureErrorKg!=0.0||
       packet.parentMassPacketSHA256.size()!=64u||
       packet.parentSourceContributionSHA256.size()!=64u||
       packet.parentTopologySHA256.size()!=64u){
        std::cerr<<"parent CSV fixture did not preserve the complete native packet\n";return 3;
    }
    const std::array<std::uint32_t,1> region{3u},stableFace{901u};
    if(!attachExplicitParentMassMappingLabels(packet,region,stableFace,error)){
        std::cerr<<"explicit parent labels rejected: "<<error<<'\n';return 4;
    }

    const auto labels=std::filesystem::path(massCSV.string()+".labels.csv");
    const std::string labelHeader="parent_tet_row,material_index,geometric_donor_body,"
        "anatomical_region_id,stable_source_face_id\n";
    std::ofstream(labels)<<labelHeader<<"0,0,7,3,901\n";
    if(!loadCompiledSkinFatParentLabelsCSV(labels,packet,error)||
       packet.anatomicalRegionIndices!=std::vector<std::uint32_t>{3u}||
       packet.stableSourceFaceIdentities!=std::vector<std::uint32_t>{901u}||
       packet.parentLabelSHA256.size()!=64u) {
        std::cerr<<"native parent label roundtrip rejected: "<<error<<'\n';return 10;
    }
    const auto validLabelHash=packet.parentLabelSHA256;
    std::ofstream(labels)<<labelHeader<<"0,0,8,4,902\n";
    if(loadCompiledSkinFatParentLabelsCSV(labels,packet,error)||
       packet.parentLabelSHA256!=validLabelHash||
       packet.anatomicalRegionIndices!=std::vector<std::uint32_t>{3u}||
       packet.stableSourceFaceIdentities!=std::vector<std::uint32_t>{901u}) {
        std::cerr<<"foreign donor labels accepted or mutated source identity\n";return 11;
    }
    std::ofstream(labels)<<labelHeader;
    if(loadCompiledSkinFatParentLabelsCSV(labels,packet,error)||
       packet.parentLabelSHA256!=validLabelHash) {
        std::cerr<<"truncated labels accepted or mutated parent packet\n";return 12;
    }

    PhysicalSkinChildMassMappingBundle bundle;
    bundle.parent=std::move(packet);
    fillOneChild(bundle);
    const std::array<double,1> densityLower{599.0},densityUpper{601.0};
    if(!preparePhysicalSkinChildMassMapping(bundle,densityLower,densityUpper,1.0,8u,error)){
        std::cerr<<"positive mass mapping fixture rejected: "<<error<<'\n';return 5;
    }
    if(bundle.spatialMassTransportResult.childMassKg.size()!=1u||
       std::abs(bundle.spatialMassTransportResult.childMassKg[0]-100.0)>1.0e-10||
       bundle.spatialMassTransportResult.childDominantParentRows!=std::vector<std::uint32_t>{0u}||
       std::abs(bundle.spatialMassTransportResult.maximumTransportDistanceUsedM-0.005)>1.0e-12||
       bundle.spatialMassTransportResult.transportEdgeCount!=1u||
       bundle.spatialMassTransportResult.augmentationCount!=1u||
       !bundle.exactTopologyIdentityTransportUsed||
       bundle.spatialMassTransportResult.receiptSHA256.size()!=64u||
       bundle.massMappingPathSHA256.size()!=64u||
       bundle.massMappingPathSHA256!=physical_skin_child_mass_mapping_detail::massMappingPathSHA256(
           bundle,true)||
       bundle.childDominantParentTetrahedra!=std::vector<std::uint32_t>{0u}){
        std::cerr<<"positive exact identity mapping did not preserve measured rest/current offset and receipt\n";return 6;
    }
    const auto exactPathHash=bundle.massMappingPathSHA256;
    if(exactPathHash!="426dd9b26bc71207a9ba74f18d346c5a82567cafa04d5631a37892e5ff03a02b"){
        std::cerr<<"legacy exact-identity v2 path hash changed\n";return 65;
    }

    // The authenticated refinement path maps two candidate cells back to the
    // same immutable 330 parent. It must accumulate that parent row, preserve
    // the verified donor split, and use the ordinary recorded-flow validator.
    auto conformingBundle=bundle;
    fillTwoConformingChildren(conformingBundle);
    conformingBundle.exactTopologyIdentityTransportUsed=false;
    conformingBundle.conformingLineageMassTransportUsed=false;
    conformingBundle.conformingLineageMassRouteRequired=true;
    conformingBundle.usesSeparatedOperationalReference=true;
    conformingBundle.operationalReferenceDomainSHA256=std::string(64u,'d');
    auto& lineage=conformingBundle.conformingLineageMassRoute;
    lineage.authenticated=true;lineage.refinedSourceCellCount=2u;
    lineage.childRefinedSourceRows={0u,1u};lineage.childImmutableParentRows={0u,0u};
    lineage.childImmutableParentFractions={0.5,0.5};
    lineage.childImmutableParentReferenceMeasureM3={
        conformingBundle.parent.referenceTetVolumeM3[0]*0.5,
        conformingBundle.parent.referenceTetVolumeM3[0]*0.5};
    lineage.childMassKg={50.0,50.0};lineage.childSourceDonorOffsets={0u,1u,2u};
    lineage.childSourceDonorBodyIndices={7u,7u};lineage.childSourceDonorMassKg={50.0,50.0};
    lineage.parentMassPacketSHA256=conformingBundle.parent.parentMassPacketSHA256;
    lineage.parentSourceContributionSHA256=conformingBundle.parent.parentSourceContributionSHA256;
    lineage.sourceVolumeSHA256=std::string(64u,'1');lineage.refinedVolumeSHA256=std::string(64u,'2');
    lineage.candidateVolumeSHA256=std::string(64u,'3');
    lineage.refinementVerificationSHA256=std::string(64u,'4');
    lineage.sourceDonorLedgerVerificationSHA256=std::string(64u,'5');
    lineage.pullbackVerificationSHA256=std::string(64u,'6');
    lineage.chainVerificationSHA256=std::string(64u,'7');
    lineage.candidateOperationalReferenceDomainSHA256=conformingBundle.operationalReferenceDomainSHA256;
    lineage.candidateCompiledDataSHA256=
        physical_skin_child_mass_mapping_detail::operationalChildCompiledDataSHA256(conformingBundle);
    lineage.verificationSHA256=
        physical_skin_child_mass_mapping_detail::verifiedConformingMassRouteSHA256(lineage);
    if(!preparePhysicalSkinChildMassMapping(conformingBundle,densityLower,densityUpper,1.0,8u,error)||
       !conformingBundle.conformingLineageMassTransportUsed||
       conformingBundle.exactTopologyIdentityTransportUsed||
       conformingBundle.spatialMassTransportResult.childMassKg!=std::vector<double>{50.0,50.0}||
       conformingBundle.spatialMassTransportResult.childDominantParentRows!=
           std::vector<std::uint32_t>{0u,0u}||
       conformingBundle.spatialMassTransportResult.parentMassTransferredKg!=std::vector<double>{100.0}||
       conformingBundle.spatialMassTransportResult.transportEdgeCount!=2u||
       conformingBundle.spatialMassTransportResult.augmentationCount!=2u||
       conformingBundle.massMappingPathSHA256!=
           physical_skin_child_mass_mapping_detail::massMappingPathSHA256(conformingBundle,false,true)) {
        std::cerr<<"verified conforming lineage transport rejected or misreported repeated immutable parent rows: "
                 <<error<<'\n';return 60;
    }

    std::vector<SpatialMassTransportPoint> lineageParentCentroids(1u),lineageChildCentroids(2u);
    for(const auto node:conformingBundle.parent.tetrahedra[0])
        for(unsigned axis=0u;axis<3u;++axis)
            lineageParentCentroids[0][axis]+=conformingBundle.parent.referenceWorldPoints[node][axis]*0.25;
    std::vector<double> lineageParentVolumes=conformingBundle.parent.referenceTetVolumeM3;
    std::vector<double> lineageChildVolumes(2u);
    for(std::size_t cell=0u;cell<2u;++cell){
        const auto tet=conformingBundle.childTetrahedra[cell];
        lineageChildVolumes[cell]=parentReferenceTetVolumeM3(tet,conformingBundle.referenceWorldPoints);
        for(const auto node:tet)for(unsigned axis=0u;axis<3u;++axis)
            lineageChildCentroids[cell][axis]+=conformingBundle.referenceWorldPoints[node][axis]*0.25;
    }
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> lineageSourceRows;
    for(const auto& row:conformingBundle.parent.sourceMassContributions)
        lineageSourceRows.push_back({.parentNodeIndex=row.parentNodeIndex,.materialIndex=row.materialIndex,
            .sourceDonorBodyIndex=row.donorBodyIndex,.massKg=row.massKg});
    std::vector<std::uint32_t> lineageSourceOffsets,lineageSourceDonors;
    std::vector<double> lineageSourceMasses;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
           conformingBundle.parent.tetrahedronMassKg,conformingBundle.parent.materialIndices,
           conformingBundle.parent.tetrahedra,lineageSourceRows,1.0e-10,5.0e-4,
           lineageSourceOffsets,lineageSourceDonors,lineageSourceMasses,error)){
        std::cerr<<"lineage mass-check fixture could not reconstruct 330 donor splits: "<<error<<'\n';return 61;
    }
    ConservativeSpatialMassTransportInput lineageInput;
    lineageInput.parentCentroidsWorld=lineageParentCentroids;
    lineageInput.parentReferenceVolumeM3=lineageParentVolumes;
    lineageInput.parentMassKg=conformingBundle.parent.tetrahedronMassKg;
    lineageInput.parentMaterialIndices=conformingBundle.parent.materialIndices;
    lineageInput.parentDonorBodyIndices=conformingBundle.parent.donorBodyIndices;
    lineageInput.parentAnatomicalRegionIndices=conformingBundle.parent.anatomicalRegionIndices;
    lineageInput.parentSourceDonorOffsets=lineageSourceOffsets;
    lineageInput.parentSourceDonorBodyIndices=lineageSourceDonors;
    lineageInput.parentSourceDonorMassKg=lineageSourceMasses;
    lineageInput.childCentroidsWorld=lineageChildCentroids;
    lineageInput.childReferenceVolumeM3=lineageChildVolumes;
    lineageInput.childMaterialIndices=conformingBundle.childMaterialIndices;
    lineageInput.childDonorBodyIndices=conformingBundle.childDonorBodyIndices;
    lineageInput.childAnatomicalRegionIndices=conformingBundle.childAnatomicalRegionIndices;
    lineageInput.materialDensityLowerKgM3=densityLower;
    lineageInput.materialDensityUpperKgM3=densityUpper;
    lineageInput.parentMassPacketSHA256=conformingBundle.parent.parentMassPacketSHA256;
    lineageInput.parentSourceContributionSHA256=conformingBundle.parent.parentSourceContributionSHA256;
    lineageInput.childMeshSHA256=conformingBundle.childMeshSHA256;
    lineageInput.regionLabelSHA256=conformingBundle.regionLabelSHA256;
    lineageInput.coordinateFrameSHA256=conformingBundle.coordinateFrameSHA256;
    lineageInput.maximumTransportDistanceM=1.0;
    lineageInput.relativeClosureTolerance=1.0e-10;
    lineageInput.absoluteMassToleranceKg=1.0e-13;
    lineageInput.absoluteMeasureToleranceM3=1.0e-15;
    lineageInput.maximumAllowedEdges=3'000'000u;lineageInput.maximumAugmentations=300'000u;
    const auto validLineageTransfer=conformingBundle.spatialMassTransportResult;
    const auto rejectsTransferMutation=[&](auto mutate,const char* label){
        auto bad=validLineageTransfer;mutate(bad);std::string rejectError;
        if(verifyRecordedConservativeSpatialMassTransport(lineageInput,bad,rejectError)){
            std::cerr<<"recorded conforming transport accepted altered "<<label<<'\n';return false;
        }
        return true;
    };
    if(!rejectsTransferMutation([](auto& bad){bad.childParentVolumeFractions[0]=0.5;},"fraction")||
       !rejectsTransferMutation([](auto& bad){bad.childParentTransportMassKg[0]+=1.0;},"mass")||
       !rejectsTransferMutation([](auto& bad){bad.childSourceDonorMassKg[0]-=1.0;},"donor share")||
       !rejectsTransferMutation([](auto& bad){bad.childImpliedDensityKgM3[0]+=1.0;},"density")||
       !rejectsTransferMutation([](auto& bad){bad.childParentTransportDistanceM[0]+=0.001;},"distance"))
        return 62;
    auto partialRouteBundle=conformingBundle;
    partialRouteBundle.conformingLineageMassRoute.childImmutableParentFractions[0]=0.25;
    partialRouteBundle.conformingLineageMassRoute.childImmutableParentReferenceMeasureM3[0]=
        lineageParentVolumes[0]*0.25;
    partialRouteBundle.conformingLineageMassRoute.childMassKg[0]=25.0;
    partialRouteBundle.conformingLineageMassRoute.childSourceDonorMassKg[0]=25.0;
    partialRouteBundle.conformingLineageMassRoute.verificationSHA256=
        physical_skin_child_mass_mapping_detail::verifiedConformingMassRouteSHA256(
            partialRouteBundle.conformingLineageMassRoute);
    const auto preservedReceipt=partialRouteBundle.spatialMassTransportResult.receiptSHA256;
    const auto preservedPath=partialRouteBundle.massMappingPathSHA256;
    const auto preservedParents=partialRouteBundle.childDominantParentTetrahedra;
    if(preparePhysicalSkinChildMassMapping(partialRouteBundle,densityLower,densityUpper,1.0,8u,error)||
       partialRouteBundle.spatialMassTransportResult.receiptSHA256!=preservedReceipt||
       partialRouteBundle.massMappingPathSHA256!=preservedPath||
       partialRouteBundle.childDominantParentTetrahedra!=preservedParents){
        std::cerr<<"incomplete lineage mass transfer was accepted or committed non-atomically\n";return 63;
    }
    auto staleCompiledDomain=conformingBundle;
    staleCompiledDomain.referenceWorldPoints[4][0]+=0.001;
    if(preparePhysicalSkinChildMassMapping(staleCompiledDomain,densityLower,densityUpper,1.0,8u,error)){
        std::cerr<<"conforming mass route accepted child arrays changed after authenticated import\n";return 64;
    }

    // Preserve a nontrivial parent current/reference translation while mapping
    // an exact child whose current rows match the parent current rows and whose
    // reference rows match the parent reference rows.
    const auto referenceIdentityDirectory=directory/"reference-preserving-identity";
    std::filesystem::create_directory(referenceIdentityDirectory);
    const auto referenceIdentityMassCSV=makePacket(referenceIdentityDirectory,0.125001);
    PhysicalSkinParentMassPacket referenceIdentityParent;
    if(!loadCompiledSkinFatParentMassPacketCSV(referenceIdentityMassCSV,referenceIdentityParent,error)||
       !attachExplicitParentMassMappingLabels(referenceIdentityParent,region,stableFace,error)){
        std::cerr<<"reference-preserving parent fixture rejected: "<<error<<'\n';return 31;
    }
    const auto referenceIdentityLabels=std::filesystem::path(
        referenceIdentityMassCSV.string()+".labels.csv");
    std::ofstream(referenceIdentityLabels)<<labelHeader<<"0,0,7,3,901\n";
    if(!loadCompiledSkinFatParentLabelsCSV(referenceIdentityLabels,referenceIdentityParent,error)){
        std::cerr<<"reference-preserving parent labels rejected: "<<error<<'\n';return 32;
    }
    PhysicalSkinChildMassMappingBundle referenceIdentityBundle;
    referenceIdentityBundle.parent=std::move(referenceIdentityParent);
    fillOneChild(referenceIdentityBundle);
    referenceIdentityBundle.referenceWorldPoints=
        referenceIdentityBundle.parent.referenceWorldPoints;
    referenceIdentityBundle.childMeshSHA256=std::string(64u,'d');
    referenceIdentityBundle.regionLabelSHA256=std::string(64u,'e');
    referenceIdentityBundle.coordinateFrameSHA256=std::string(64u,'f');
    if(!preparePhysicalSkinChildMassMapping(referenceIdentityBundle,
            densityLower,densityUpper,1.0,8u,error)||
       !referenceIdentityBundle.exactTopologyIdentityTransportUsed||
       referenceIdentityBundle.currentWorldPoints!=
           referenceIdentityBundle.parent.currentWorldPoints||
       referenceIdentityBundle.referenceWorldPoints!=
           referenceIdentityBundle.parent.referenceWorldPoints||
       referenceIdentityBundle.currentWorldPoints==
           referenceIdentityBundle.referenceWorldPoints||
       std::abs(referenceIdentityBundle.spatialMassTransportResult.maximumTransportDistanceUsedM)>1.0e-12||
       std::abs(referenceIdentityBundle.spatialMassTransportResult.childMassKg[0]-100.0)>1.0e-10){
        std::cerr<<"reference-preserving identity route lost the parent current/reference distinction: "
            <<error<<'\n';return 33;
    }

    // Use a skew, non-binary reference tetrahedron so algebraically equivalent
    // determinant formulas cannot silently produce a different receipt domain.
    auto skewIdentity=referenceIdentityBundle;
    skewIdentity.parent.referenceWorldPoints={
        {0.0314159265359,-0.2718281828459,0.1618033988749},
        {1.1732050807568,-0.1341640786499,0.3035533905933},
        {0.2526457513111,0.7316624790355,0.2196152422707},
        {0.1483239697419,-0.0123456789012,1.1180339887499}};
    skewIdentity.parent.currentWorldPoints=skewIdentity.parent.referenceWorldPoints;
    for(auto& point:skewIdentity.parent.currentWorldPoints)for(auto& value:point)
        value=static_cast<double>(static_cast<float>(value));
    skewIdentity.currentWorldPoints=skewIdentity.parent.currentWorldPoints;
    skewIdentity.referenceWorldPoints=skewIdentity.parent.referenceWorldPoints;
    skewIdentity.parent.referenceTetVolumeM3[0]=parentReferenceTetVolumeM3(
        skewIdentity.parent.tetrahedra[0],skewIdentity.parent.referenceWorldPoints);
    const std::array<double,1> skewLower{1.0},skewUpper{1000.0};
    if(!preparePhysicalSkinChildMassMapping(skewIdentity,skewLower,skewUpper,1.0,8u,error)) {
        std::cerr<<"skew reference identity rejected: "<<error<<'\n';return 42;
    }
    std::vector<SpatialMassTransportPoint> replayParentCentroids(1u),replayChildCentroids(1u);
    std::array<double,1> replayParentVolumes{
        parentReferenceTetVolumeM3(skewIdentity.parent.tetrahedra[0],skewIdentity.parent.referenceWorldPoints)};
    std::array<double,1> replayChildVolumes{
        parentReferenceTetVolumeM3(skewIdentity.childTetrahedra[0],skewIdentity.referenceWorldPoints)};
    for(auto node:skewIdentity.parent.tetrahedra[0])for(unsigned axis=0;axis<3u;++axis)
        replayParentCentroids[0][axis]+=skewIdentity.parent.referenceWorldPoints[node][axis]*0.25;
    for(auto node:skewIdentity.childTetrahedra[0])for(unsigned axis=0;axis<3u;++axis)
        replayChildCentroids[0][axis]+=skewIdentity.referenceWorldPoints[node][axis]*0.25;
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> replaySources;
    for(const auto& row:skewIdentity.parent.sourceMassContributions)
        replaySources.push_back({.parentNodeIndex=row.parentNodeIndex,.materialIndex=row.materialIndex,
            .sourceDonorBodyIndex=row.donorBodyIndex,.massKg=row.massKg});
    std::vector<std::uint32_t> replaySourceOffsets,replaySourceDonors;
    std::vector<double> replaySourceMasses;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            skewIdentity.parent.tetrahedronMassKg,skewIdentity.parent.materialIndices,
            skewIdentity.parent.tetrahedra,replaySources,1.0e-10,5.0e-4,
            replaySourceOffsets,replaySourceDonors,replaySourceMasses,error))return 43;
    ConservativeSpatialMassTransportInput replayInput;
    replayInput.parentCentroidsWorld=replayParentCentroids;
    replayInput.parentReferenceVolumeM3=replayParentVolumes;
    replayInput.parentMassKg=skewIdentity.parent.tetrahedronMassKg;
    replayInput.parentMaterialIndices=skewIdentity.parent.materialIndices;
    replayInput.parentDonorBodyIndices=skewIdentity.parent.donorBodyIndices;
    replayInput.parentAnatomicalRegionIndices=skewIdentity.parent.anatomicalRegionIndices;
    replayInput.parentSourceDonorOffsets=replaySourceOffsets;
    replayInput.parentSourceDonorBodyIndices=replaySourceDonors;
    replayInput.parentSourceDonorMassKg=replaySourceMasses;
    replayInput.childCentroidsWorld=replayChildCentroids;
    replayInput.childReferenceVolumeM3=replayChildVolumes;
    replayInput.childMaterialIndices=skewIdentity.childMaterialIndices;
    replayInput.childDonorBodyIndices=skewIdentity.childDonorBodyIndices;
    replayInput.childAnatomicalRegionIndices=skewIdentity.childAnatomicalRegionIndices;
    replayInput.materialDensityLowerKgM3=skewLower;
    replayInput.materialDensityUpperKgM3=skewUpper;
    std::string replayParentHash,replayChildHash,replayBoundsHash;
    spatial_mass_transport_detail::hashInputArrays(replayInput,
        replayParentHash,replayChildHash,replayBoundsHash);
    if(replayParentHash!=skewIdentity.spatialMassTransportResult.parentDataSHA256||
       replayChildHash!=skewIdentity.spatialMassTransportResult.childDataSHA256||
       replayBoundsHash!=skewIdentity.spatialMassTransportResult.materialBoundsSHA256) {
        std::cerr<<"mass-map receipt geometry differs from unpaired replay-owner arithmetic\n";return 44;
    }

    auto compilerEquivalentCurrent=referenceIdentityBundle;
    compilerEquivalentCurrent.currentWorldPoints[1][0]+=1.0e-9;
    if(!preparePhysicalSkinChildMassMapping(compilerEquivalentCurrent,
            densityLower,densityUpper,1.0,8u,error)||
       !compilerEquivalentCurrent.exactTopologyIdentityTransportUsed||
       compilerEquivalentCurrent.currentWorldPoints[1][0]==
           compilerEquivalentCurrent.parent.currentWorldPoints[1][0]||
       compilerEquivalentCurrent.massMappingPathSHA256==referenceIdentityBundle.massMappingPathSHA256) {
        std::cerr<<"compiler-equivalent source current lost exact identity or digest binding: "<<error<<'\n';return 40;
    }
    auto changedCompilerBits=referenceIdentityBundle;
    changedCompilerBits.currentWorldPoints[1][0]=static_cast<double>(std::nextafterf(
        1.0f,std::numeric_limits<float>::infinity()));
    std::vector<SpatialMassTransportPoint> bitCentroids(1u);
    std::vector<std::uint32_t> bitParentRows;
    if(physical_skin_child_mass_mapping_detail::exactIdentityParentRows(
            changedCompilerBits,bitCentroids,bitCentroids,1.0,bitParentRows)) {
        std::cerr<<"changed GPU current bits passed exact identity\n";return 41;
    }

    auto changedNativeReference=referenceIdentityBundle;
    changedNativeReference.referenceWorldPoints.front()[0]+=1.0e-12;
    if(preparePhysicalSkinChildMassMapping(changedNativeReference,
            densityLower,densityUpper,1.0,8u,error)) {
        std::cerr<<"changed non-Float32 reference bypassed generic owner-coordinate gate\n";return 38;
    }
    auto changedNativeCurrent=referenceIdentityBundle;
    changedNativeCurrent.currentWorldPoints.front()[0]+=1.0e-12;
    if(preparePhysicalSkinChildMassMapping(changedNativeCurrent,
            densityLower,densityUpper,1.0,8u,error)) {
        std::cerr<<"non-Float32 current bypassed GPU owner-coordinate gate\n";return 39;
    }

    // A reference mesh that mixes the two allowed contracts node by node is
    // not an identity case, even when every row matches one parent coordinate.
    auto mixedReferenceIdentity=referenceIdentityBundle;
    mixedReferenceIdentity.referenceWorldPoints.front()=
        mixedReferenceIdentity.parent.currentWorldPoints.front();
    std::vector<SpatialMassTransportPoint> mixedReferenceCentroids(1u);
    std::vector<std::uint32_t> mixedReferenceParentRows;
    if(physical_skin_child_mass_mapping_detail::exactIdentityParentRows(
           mixedReferenceIdentity,mixedReferenceCentroids,mixedReferenceCentroids,
           1.0,mixedReferenceParentRows)) {
        std::cerr<<"mixed per-node reference contracts passed exact identity admission\n";
        return 34;
    }

    auto stalePath=bundle;
    stalePath.currentWorldPoints[0][0]=static_cast<double>(std::nextafterf(
        static_cast<float>(stalePath.currentWorldPoints[0][0]),
        std::numeric_limits<float>::infinity()));
    if(stalePath.massMappingPathSHA256==physical_skin_child_mass_mapping_detail::massMappingPathSHA256(
           stalePath,stalePath.exactTopologyIdentityTransportUsed)){
        std::cerr<<"mass mapping path digest did not bind current child geometry\n";return 23;
    }
    auto oddTetOrder=bundle;
    std::swap(oddTetOrder.parent.tetrahedra[0][0],oddTetOrder.parent.tetrahedra[0][1]);
    std::vector<SpatialMassTransportPoint> identityCentroids(1u);
    std::vector<std::uint32_t> oddParentRows;
    if(physical_skin_child_mass_mapping_detail::exactIdentityParentRows(
           oddTetOrder,identityCentroids,identityCentroids,1.0,oddParentRows)){
        std::cerr<<"odd child/parent node-order permutation passed exact identity orientation proof\n";return 28;
    }

    auto appendRejectsUnboundPathAtomically=[&](PhysicalSkinChildMassMappingBundle& candidate) {
        numi::matter::WorldSource world;
        numi::matter::ObjectSource sentinel;sentinel.name="sentinel";
        world.objects.push_back(sentinel);
        const auto before=world.objects.size();
        PreparedOrganicFEM preparedSkin{},preparedFat{};
        ExplicitUnpairedVolumeReport report;report.tetrahedronCount=777u;
        std::string appendError;
        const bool accepted=appendPreparedPhysicalSkinChildCandidate(world,candidate,
            15u,"skin","test",0.0,16u,"fat","test",0.0,8u,
            "stale-mapping-test",preparedSkin,preparedFat,report,appendError);
        return !accepted&&appendError.find("path receipt")!=std::string::npos&&
            world.objects.size()==before&&world.objects.front().name=="sentinel"&&
            report.tetrahedronCount==777u;
    };
    auto truncatedReference=bundle;
    truncatedReference.parent.referenceWorldPoints.pop_back();
    auto truncatedLocalOwner=bundle;
    truncatedLocalOwner.parent.nodeFrameOwnerLocalPoints.pop_back();
    auto missingOwnerMetadata=bundle;
    missingOwnerMetadata.parent.nodeFrameOwnerBodyIndices.clear();
    if(!appendRejectsUnboundPathAtomically(truncatedReference)||
       !appendRejectsUnboundPathAtomically(truncatedLocalOwner)||
       !appendRejectsUnboundPathAtomically(missingOwnerMetadata)){
        std::cerr<<"truncated parent reference or missing owner metadata was not rejected atomically before append\n";return 29;
    }
    auto missingOwnerPrepare=bundle;
    missingOwnerPrepare.parent.nodeFrameOwnerLocalPoints.clear();
    if(preparePhysicalSkinChildMassMapping(missingOwnerPrepare,densityLower,densityUpper,1.0,8u,error)||
       error.find("missing")==std::string::npos){
        std::cerr<<"missing owner-local metadata reached mass mapping: "<<error<<'\n';return 30;
    }

    auto wrongRegionFresh=bundle;
    wrongRegionFresh.spatialMassTransportResult={};
    wrongRegionFresh.childDominantParentTetrahedra.clear();
    wrongRegionFresh.exactTopologyIdentityTransportUsed=false;
    wrongRegionFresh.massMappingPathSHA256.clear();
    wrongRegionFresh.childAnatomicalRegionIndices[0]=4u;
    if(preparePhysicalSkinChildMassMapping(wrongRegionFresh,densityLower,densityUpper,1.0,8u,error)||
       error.find("not feasible")==std::string::npos||
       wrongRegionFresh.exactTopologyIdentityTransportUsed||
       !wrongRegionFresh.massMappingPathSHA256.empty()){
        std::cerr<<"label mismatch did not fall through to the bounded spatial transport gate: "<<error<<'\n';return 24;
    }

    auto wrongRegion=bundle;
    wrongRegion.childAnatomicalRegionIndices[0]=4u;
    const auto priorReceipt=bundle.spatialMassTransportResult.receiptSHA256;
    if(preparePhysicalSkinChildMassMapping(wrongRegion,densityLower,densityUpper,1.0,8u,error)||
       error.find("not feasible")==std::string::npos||
       bundle.spatialMassTransportResult.receiptSHA256!=priorReceipt||
       bundle.massMappingPathSHA256!=exactPathHash){
        std::cerr<<"region barrier negative case was not rejected transactionally: "<<error<<'\n';return 7;
    }

    auto oneULP=bundle;
    const auto nextX=static_cast<double>(std::nextafterf(
        static_cast<float>(oneULP.referenceWorldPoints[1][0]),
        std::numeric_limits<float>::infinity()));
    oneULP.currentWorldPoints[1][0]=nextX;
    oneULP.referenceWorldPoints[1][0]=nextX;
    if(!preparePhysicalSkinChildMassMapping(oneULP,densityLower,densityUpper,1.0,8u,error)||
       oneULP.exactTopologyIdentityTransportUsed||
       oneULP.massMappingPathSHA256==exactPathHash||
       oneULP.spatialMassTransportResult.transportEdgeCount!=1u||
       oneULP.spatialMassTransportResult.maximumTransportDistanceUsedM<=0.0||
       oneULP.spatialMassTransportResult.maximumTransportDistanceUsedM>1.0){
        std::cerr<<"one-ULP node change did not fall through to bounded spatial transport: "<<error<<'\n';return 25;
    }

    auto negativeTet=bundle;
    negativeTet.childTetrahedra[0]={0u,2u,1u,3u};
    if(preparePhysicalSkinChildMassMapping(negativeTet,densityLower,densityUpper,1.0,8u,error)||
       error.find("nonpositive")==std::string::npos){
        std::cerr<<"negative child orientation was not rejected: "<<error<<'\n';return 8;
    }

    auto duplicateTet=bundle;
    duplicateTet.spatialMassTransportResult={};
    duplicateTet.childDominantParentTetrahedra.clear();
    duplicateTet.exactTopologyIdentityTransportUsed=false;
    duplicateTet.massMappingPathSHA256.clear();
    duplicateTet.childTetrahedra.push_back(duplicateTet.childTetrahedra[0]);
    duplicateTet.childMaterialIndices.push_back(duplicateTet.childMaterialIndices[0]);
    duplicateTet.childDonorBodyIndices.push_back(duplicateTet.childDonorBodyIndices[0]);
    duplicateTet.childAnatomicalRegionIndices.push_back(duplicateTet.childAnatomicalRegionIndices[0]);
    duplicateTet.childStableSourceFaceIdentities.push_back(duplicateTet.childStableSourceFaceIdentities[0]);
    if(preparePhysicalSkinChildMassMapping(duplicateTet,densityLower,densityUpper,1.0,8u,error)||
       duplicateTet.exactTopologyIdentityTransportUsed||
       !duplicateTet.massMappingPathSHA256.empty()){
        std::cerr<<"duplicate child tetrahedron was accepted or bypassed the bounded transport fallback: "<<error<<'\n';return 26;
    }

    auto missingTet=bundle;
    missingTet.spatialMassTransportResult={};
    missingTet.childDominantParentTetrahedra.clear();
    missingTet.exactTopologyIdentityTransportUsed=false;
    missingTet.massMappingPathSHA256.clear();
    missingTet.childTetrahedra.clear();
    missingTet.childMaterialIndices.clear();
    missingTet.childDonorBodyIndices.clear();
    missingTet.childAnatomicalRegionIndices.clear();
    missingTet.childStableSourceFaceIdentities.clear();
    if(preparePhysicalSkinChildMassMapping(missingTet,densityLower,densityUpper,1.0,8u,error)||
       error.find("missing")==std::string::npos||
       missingTet.exactTopologyIdentityTransportUsed||
       !missingTet.massMappingPathSHA256.empty()){
        std::cerr<<"missing child tetrahedron domain was accepted: "<<error<<'\n';return 27;
    }

    PhysicalSkinParentMassPacket unchanged=bundle.parent;
    std::ofstream(massCSV,std::ios::trunc)
        <<"source_tet_row,material_index,mass_kg\n0,0,-1\n";
    if(loadCompiledSkinFatParentMassPacketCSV(massCSV,unchanged,error)||
       unchanged.totalMassKg!=bundle.parent.totalMassKg){
        std::cerr<<"invalid parent packet was accepted or mutated output\n";return 9;
    }


    const auto childNode=directory/"child.node",childEle=directory/"child.ele",childLabels=directory/"child.labels.csv";
    std::ofstream(childNode)<<"4 3 0 1\n11 0 0 0 1\n22 1 0 0 1\n33 0 1 0 1\n44 0 0 1 1\n";
    std::ofstream(childEle)<<"1 4 1\n50 11 22 33 44 0\n";
    std::ofstream(childLabels)<<"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n0,0,7,3,901\n";
    PhysicalSkinChildMassMappingBundle imported;imported.parent=bundle.parent;
    PhysicalSkinChildTetGenImportReport importedReport;
    if(!loadPhysicalSkinChildTetGenReference(childNode,childEle,childLabels,std::string(64u,'a'),
            8u,1.0,imported,importedReport,error)||imported.childTetrahedra!=bundle.childTetrahedra||
       imported.childAnatomicalRegionIndices!=bundle.childAnatomicalRegionIndices||
       imported.currentWorldPoints!=imported.referenceWorldPoints||
       importedReport.nodeCount!=4u||importedReport.cellCount!=1u||
       !preparePhysicalSkinChildMassMapping(imported,densityLower,densityUpper,1.0,8u,error)) {
        std::cerr<<"TetGen reference/explicit label import or downstream transfer rejected: "<<error<<'\n';return 13;
    }
    const auto checkReject=[&](const char* expected) {
        PhysicalSkinChildMassMappingBundle empty;empty.parent=bundle.parent;
        PhysicalSkinChildTetGenImportReport prior;prior.nodeCount=999u;
        const bool accepted=loadPhysicalSkinChildTetGenReference(childNode,childEle,childLabels,
            std::string(64u,'a'),8u,1.0,empty,prior,error);
        return !accepted&&error.find(expected)!=std::string::npos&&empty.childTetrahedra.empty()&&
            empty.referenceWorldPoints.empty()&&prior.nodeCount==999u&&empty.parent.totalMassKg==bundle.parent.totalMassKg;
    };
    std::ofstream(childEle)<<"2 4 1\n50 11 22 33 44 0\n51 11 22 33 44 0\n";
    if(!checkReject("duplicate tetrahedron")){std::cerr<<"TetGen duplicate cells accepted\n";return 22;}
    std::ofstream(childEle)<<"1 4 1\n50 11 33 22 44 0\n";
    if(!checkReject("nonpositive")){std::cerr<<"TetGen inverted cell accepted or mutated output\n";return 14;}
    std::ofstream(childEle)<<"1 4 1\n50 11 22 33 45 0\n";
    if(!checkReject("missing node")){std::cerr<<"TetGen missing node accepted\n";return 15;}
    std::ofstream(childEle)<<"1 4 1\n50 11 22 33 44 0\n";
    std::ofstream(childLabels)<<"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n0,1,7,3,901\n";
    if(!checkReject("domains")){std::cerr<<"TetGen mismatched material labels accepted\n";return 16;}
    std::ofstream(childLabels)<<"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n0,0,8,3,901\n";
    if(!checkReject("domains")){std::cerr<<"TetGen out-of-domain donor accepted\n";return 17;}
    std::ofstream(childLabels)<<"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n";
    if(!checkReject("truncated")){std::cerr<<"TetGen truncated labels accepted\n";return 18;}
    std::ofstream(childLabels)<<"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n0,0,7,3,901\n";
    // The original double tet is positive, but loses its height at Float32 emission.
    std::ofstream(childNode)<<"4 3 0 0\n11 0 0 1\n22 1 0 1\n33 0 1 1\n44 0 0 1.000000001\n";
    if(!checkReject("nonpositive")){std::cerr<<"Float32-collapsed TetGen tet accepted\n";return 19;}
    std::ofstream(childNode)<<"4 3 0 0\n11 0 0 0\n11 1 0 0\n33 0 1 0\n44 0 0 1\n";
    if(!checkReject("duplicate")){std::cerr<<"TetGen duplicate node ID accepted\n";return 20;}
    std::ofstream(childNode)<<"4 3 0 0\n11 0 0 0\n22 1 0 0\n33 0 1 0\n44 0 0 1\n55 1 1 1\n";
    if(!checkReject("extra")){std::cerr<<"TetGen extra node accepted\n";return 21;}

    std::filesystem::remove_all(directory);
    std::cout<<"physical_child_mass_mapping_fixture=PASS "
             <<"parent_csv_roundtrip=PASS exact_identity_receipt_replay=PASS "
             <<"measured_parent_rest_current_offset=PASS one_ulp_fallback=PASS "
             <<"strict_region_barrier=PASS duplicate_missing_inverted_tet_gates=PASS "
             <<"odd_orientation_parity=PASS truncated_path_receipt_rejections=PASS "
             <<"tetgen_reference_import=PASS transactional_import_rejections=PASS transfer=fixture_only\n";
    return 0;
}

