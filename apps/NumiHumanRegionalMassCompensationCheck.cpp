#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
using namespace metalrobo::human::coarse_tissue;
bool runRegionalCompensationControl() {
    const std::array<SpatialMassTransportPoint,3> parentPoints{{
        SpatialMassTransportPoint{0.15,0.0,0.0},
        SpatialMassTransportPoint{0.10,0.0,0.0},
        SpatialMassTransportPoint{0.0,0.0,0.0}}};
    const std::array<double,3> parentVolumes{1.0,1.0,1.0};
    const std::array<double,3> parentMasses{1.0,1.0,1.0};
    const std::array<std::uint32_t,3> parentMaterials{15,15,15};
    const std::array<std::uint32_t,3> parentDonors{1,1,1};
    const std::array<std::uint32_t,3> parentRegions{10,15,20};
    const std::array<std::uint32_t,4> donorOffsets{0,1,2,3};
    const std::array<std::uint32_t,3> donorRows{1,1,1};
    const std::array<double,3> donorMasses{1.0,1.0,1.0};
    const std::array<SpatialMassTransportPoint,3> childPoints{{
        SpatialMassTransportPoint{0.05,0.0,0.0},
        SpatialMassTransportPoint{0.10,0.0,0.0},
        SpatialMassTransportPoint{0.30,0.0,0.0}}};
    const std::array<double,3> childVolumes{1.0,1.0,1.0};
    const std::array<std::uint32_t,3> childMaterials{15,15,15};
    const std::array<std::uint32_t,3> childDonors{1,1,1};
    const std::array<std::uint32_t,3> childRegions{10,15,10};
    std::array<double,16> densityLower{};densityLower.fill(1.0);
    std::array<double,16> densityUpper{};densityUpper.fill(2.0);
    const std::array<std::uint32_t,4> childOffsets{0,1,2,3};
    const std::array<std::uint32_t,3> slotGroups{0,1,0};
    const std::array<double,3> slotFractions{1,1,1};
    const std::array<std::uint32_t,3> groupMaterial{15,15,15};
    const std::array<std::uint32_t,3> groupDonor{1,1,1};
    const std::array<std::uint32_t,3> groupRegion{10,15,20};
    const std::array<std::uint64_t,3> groupIdentity{0x8000000000000010ull,0x8000000000000011ull,0x8000000000000012ull};
    const std::array<std::uint32_t,3> parentStableFaces{501,502,503};
    const std::array<std::uint32_t,4> groupParentOffsets{0,1,2,3};
    const std::array<std::uint32_t,3> groupParentRows{0,1,2};
    const std::array<std::uint32_t,4> groupFaceOffsets{0,1,2,3};
    const std::array<std::uint32_t,3> groupFaceRows{501,502,503};
    const std::string pin(64,'d');
    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentPoints;input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMasses;input.parentMaterialIndices=parentMaterials;
    input.parentDonorBodyIndices=parentDonors;input.parentAnatomicalRegionIndices=parentRegions;
    input.parentSourceDonorOffsets=donorOffsets;input.parentSourceDonorBodyIndices=donorRows;
    input.parentSourceDonorMassKg=donorMasses;input.childCentroidsWorld=childPoints;
    input.childReferenceVolumeM3=childVolumes;input.childMaterialIndices=childMaterials;
    input.childDonorBodyIndices=childDonors;input.childAnatomicalRegionIndices=childRegions;
    input.materialDensityLowerKgM3=densityLower;input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=pin;input.parentSourceContributionSHA256=pin;
    input.childMeshSHA256=pin;input.regionLabelSHA256=pin;input.coordinateFrameSHA256=pin;
    input.maximumTransportDistanceM=0.2;input.maximumAllowedEdges=16;input.maximumAugmentations=100;
    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;support.slotGroupIndices=slotGroups;
    support.volumeFractions=slotFractions;support.groupMaterialIndices=groupMaterial;
    support.donorBodyIndices=groupDonor;support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=parentStableFaces;
    support.sourceParentTetOffsets=groupParentOffsets;support.sourceParentTetRows=groupParentRows;
    support.sourceStableFaceOffsets=groupFaceOffsets;support.sourceStableFaceRows=groupFaceRows;
    support.typedSupportSHA256=pin;
    const std::array<std::uint32_t,1> compParent{2},compChild{2};
    support.compensationParentRows=compParent;support.compensationChildRows=compChild;
    support.compensationEvidenceSHA256=pin;support.compensationMaximumMassKg=1.0;
    support.compensationMaximumDistanceM=.4;
    input.maximumTransportDistanceM=.4;
    ConservativeSpatialMassTransportResult result;std::string error;
    std::ostringstream diagnostics;auto* savedErr=std::cerr.rdbuf(diagnostics.rdbuf());
    const bool ok=assembleConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error);
    std::cerr.rdbuf(savedErr);
    if(!ok||!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)){
        std::cerr<<"bounded regional fixture failed: "<<error<<"\n"<<diagnostics.str();return false;
    }
    if(result.childMassKg!=std::vector<double>({1.0,1.0,1.0}))return false;
    auto wrong=support;wrong.compensationMaximumMassKg=.99;
    ConservativeSpatialMassTransportResult rejected;
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(input,wrong,rejected,error)||error.find("budget")==std::string::npos)return false;
    wrong=support;const std::array<std::uint32_t,2> duplicate{2,2};wrong.compensationParentRows=duplicate;
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(input,wrong,rejected,error))return false;
    wrong=support;wrong.compensationEvidenceSHA256=std::string_view{};
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(input,wrong,rejected,error))return false;
    wrong=support;wrong.compensationMaximumMassKg=1.1;
    if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input,wrong,result,error))return false;

    ConservativeSpatialChildSemanticSupportData native;
    native.childOffsets.assign(childOffsets.begin(),childOffsets.end());
    native.slotGroupIndices.assign(slotGroups.begin(),slotGroups.end());
    native.volumeFractions.assign(slotFractions.begin(),slotFractions.end());
    native.groupMaterialIndices.assign(groupMaterial.begin(),groupMaterial.end());
    native.donorBodyIndices.assign(groupDonor.begin(),groupDonor.end());
    native.anatomicalRegionIndices.assign(groupRegion.begin(),groupRegion.end());
    native.derivedGroupIdentityIDs.assign(groupIdentity.begin(),groupIdentity.end());
    native.parentStableSourceFaceIdentities.assign(parentStableFaces.begin(),parentStableFaces.end());
    native.sourceParentTetOffsets.assign(groupParentOffsets.begin(),groupParentOffsets.end());
    native.sourceParentTetRows.assign(groupParentRows.begin(),groupParentRows.end());
    native.sourceStableFaceOffsets.assign(groupFaceOffsets.begin(),groupFaceOffsets.end());
    native.sourceStableFaceRows.assign(groupFaceRows.begin(),groupFaceRows.end());native.typedSupportSHA256=pin;
    PhysicalSkinParentMassPacket topology;
    topology.tetrahedra={{0,1,2,3},{1,2,3,4},{2,3,4,5}};
    topology.materialIndices.assign(parentMaterials.begin(),parentMaterials.end());
    topology.anatomicalRegionIndices.assign(parentRegions.begin(),parentRegions.end());
    CurrentSourceCarrierQuadratureData policy;
    policy.compensationParentRows={2};policy.compensationChildRows={2};
    policy.compensationEvidenceSHA256=pin;policy.compensationMaximumMassKg=1.0;
    policy.compensationMaximumDistanceM=.4;
    if(!bindCurrentRegionalMassCompensation(native,input,topology,policy,error)){
        std::cerr<<"two-hop source-face path rejected: "<<error<<'\n';return false;
    }
    auto disconnected=topology;disconnected.tetrahedra[2]={6,7,8,9};
    auto copy=native;
    if(bindCurrentRegionalMassCompensation(copy,input,disconnected,policy,error)||
       error.find("two region hops")==std::string::npos)return false;
    auto crossMaterial=topology;crossMaterial.materialIndices[1]=16;
    copy=native;
    if(bindCurrentRegionalMassCompensation(copy,input,crossMaterial,policy,error)||
       error.find("two region hops")==std::string::npos)return false;

    // Exercise the same carrier expansion and solver-free replay used by native import.
    std::array<std::array<std::array<double,3>,4>,3> tetPoints{};
    const std::array<std::array<double,3>,4> offset{{{-.01,-.01,-.01},{.01,-.01,.01},{-.01,.01,.01},{.01,.01,-.01}}};
    for(unsigned p=0;p<3;++p)for(unsigned n=0;n<4;++n)for(unsigned k=0;k<3;++k)
        tetPoints[p][n][k]=parentPoints[p][k]+offset[n][k];
    const std::array<std::array<std::uint32_t,4>,3> tetRows{{{0,1,2,3},{4,5,6,7},{8,9,10,11}}};
    const std::array<std::uint32_t,4> carrierOffsets{0,0,0,2};
    const std::array<std::uint32_t,3> globalRows{0,1,2};
    const std::array<double,2> weights{.5,.5};
    const std::array<std::array<double,4>,2> bary{{{.4,.1,.25,.25},{.1,.4,.25,.25}}};
    const std::array<std::uint32_t,2> witnesses{2,2};std::array<double,2> distances{};
    for(unsigned r=0;r<2;++r)distances[r]=source_carrier_transport_detail::pointDistance(
        source_carrier_transport_detail::barycentricPoint(tetPoints[2],bary[r]),childPoints[2]);
    SourceCellCarrierQuadratureInput carriers;
    carriers.parentOffsets=carrierOffsets;carriers.globalSourceTetRows=globalRows;
    carriers.sourceTetNodeRows=tetRows;carriers.parentReferenceTetWorldVertices=tetPoints;
    carriers.carrierWeights=weights;carriers.carrierBarycentric=bary;
    carriers.witnessChildRows=witnesses;carriers.witnessChildCentroidDistancesM=distances;
    carriers.sourceCarrierPacketSHA256=pin;carriers.sourceTetTableSHA256=pin;carriers.sourceNodeTableSHA256=pin;
    ConservativeSpatialMassTransportResult carried;SourceCellCarrierTransportTrace trace;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
            input,native.view(),carriers,carried,trace,error)||
       !canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            input,native.view(),carriers,carried,trace,error)){
        std::cerr<<"carrier compensation replay rejected: "<<error<<'\n';return false;
    }
    native.childAdmissibleGroupOffsets={0,1,2,3};
    native.childAdmissibleGroupIndices={0,1,0};native.admissibleGroupEvidenceSHA256=pin;
    ConservativeSpatialMassTransportResult indexed;SourceCellCarrierTransportTrace indexedTrace;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
            input,native.view(),carriers,indexed,indexedTrace,error)||
       indexed.childParentRows!=carried.childParentRows||indexed.childMassKg!=carried.childMassKg||
       indexed.childParentTransportMassKg!=carried.childParentTransportMassKg||
       indexed.childParentTransportDistanceM!=carried.childParentTransportDistanceM||
       indexedTrace.originalCentroidNearestAdmissibleDistanceM!=trace.originalCentroidNearestAdmissibleDistanceM){
        std::cerr<<"explicit-allowlist index differs from full predicate scan: "<<error<<'\n';return false;
    }
    if(!canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            input,native.view(),carriers,indexed,indexedTrace,error))return false;
    carried=std::move(indexed);trace=std::move(indexedTrace);
    std::cout<<"explicit allowlist index/full-scan mass, flow, distance and nearest-distance parity PASS\n";
    const auto rejectReplay=[&](ConservativeSpatialChildSemanticSupportData bad){
        auto value=carried;auto path=trace;
        return !canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            input,bad.view(),carriers,value,path,error);
    };
    copy=native;copy.compensationParentRows={0};if(!rejectReplay(copy))return false;
    copy=native;copy.compensationChildRows={0};if(!rejectReplay(copy))return false;
    copy=native;copy.compensationMaximumDistanceM=.29;if(!rejectReplay(copy))return false;
    std::cout<<"source-face two-hop path, disconnected/material-boundary rejection, carrier expansion and bounded native replay PASS\n";
    std::cout<<"bounded regional compensation conserves mass and density; budget, duplicate, missing evidence and changed-policy replay rejection PASS\n";
    return true;
}

int main(){return runRegionalCompensationControl()?0:1;}
