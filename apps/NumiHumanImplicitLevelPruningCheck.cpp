#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
using namespace metalrobo::human::coarse_tissue;
bool runSelectiveRadiusGroupBucketControl() {
    const std::array<SpatialMassTransportPoint,2> parentPoints{{
        SpatialMassTransportPoint{0.15,0.0,0.0},
        SpatialMassTransportPoint{0.0,0.0,0.0}}};
    const std::array<double,2> parentVolumes{1.0,1.0};
    const std::array<double,2> parentMasses{1.0,1.0};
    const std::array<std::uint32_t,2> parentMaterials{15,15};
    const std::array<std::uint32_t,2> parentDonors{1,1};
    const std::array<std::uint32_t,2> parentRegions{10,10};
    const std::array<std::uint32_t,3> donorOffsets{0,1,2};
    const std::array<std::uint32_t,2> donorRows{1,1};
    const std::array<double,2> donorMasses{1.0,1.0};
    const std::array<SpatialMassTransportPoint,2> childPoints{{
        SpatialMassTransportPoint{0.05,0.0,0.0},
        SpatialMassTransportPoint{0.30,0.0,0.0}}};
    const std::array<double,2> childVolumes{1.0,1.0};
    const std::array<std::uint32_t,2> childMaterials{15,15};
    const std::array<std::uint32_t,2> childDonors{1,1};
    const std::array<std::uint32_t,2> childRegions{10,10};
    std::array<double,16> densityLower{};densityLower.fill(1.0);
    std::array<double,16> densityUpper{};densityUpper.fill(2.0);
    const std::array<std::uint32_t,3> childOffsets{0,1,2};
    const std::array<std::uint32_t,2> slotGroups{0,0};
    const std::array<double,2> slotFractions{1,1};
    const std::array<std::uint32_t,1> groupMaterial{15};
    const std::array<std::uint32_t,1> groupDonor{1};
    const std::array<std::uint32_t,1> groupRegion{10};
    const std::array<std::uint64_t,1> groupIdentity{0x8000000000000010ull};
    const std::array<std::uint32_t,2> parentStableFaces{501,502};
    const std::array<std::uint32_t,2> groupParentOffsets{0,2};
    const std::array<std::uint32_t,2> groupParentRows{0,1};
    const std::array<std::uint32_t,2> groupFaceOffsets{0,2};
    const std::array<std::uint32_t,2> groupFaceRows{501,502};
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
    ConservativeSpatialMassTransportResult result;std::string error;
    std::ostringstream diagnostics;auto* savedErr=std::cerr.rdbuf(diagnostics.rdbuf());
    const bool ok=assembleConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error);
    std::cerr.rdbuf(savedErr);
    if(!ok||!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)){
        std::cerr<<"selective-radius same-group fixture failed: "<<error<<"\n"<<diagnostics.str();return false;
    }
    const double total=std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0);
    if(result.semanticCandidateEdgeCount<2u||result.semanticCandidateEdgeCount>3u||
       result.childMassKg.size()!=2u||
       std::abs(result.childMassKg[0]-1.0)>1e-12||std::abs(result.childMassKg[1]-1.0)>1e-12||
       std::abs(total-2.0)>1e-12||
       diagnostics.str().find("sink_depth=5")==std::string::npos){
        std::cerr<<"selective-radius/reverse-residual fixture lost a route or mass: edges="
                 <<result.semanticCandidateEdgeCount<<" total="<<total
                 <<" saw_reverse_length5="<<(diagnostics.str().find("sink_depth=5")!=std::string::npos)<<"\n";return false;
    }
    const auto pruneToken=std::string("bfs_pruned_child_group_entries=");
    const auto pruneAt=diagnostics.str().find(pruneToken);
    if(pruneAt==std::string::npos||
       std::stoull(diagnostics.str().substr(pruneAt+pruneToken.size()))==0u){
        std::cerr<<"selective-radius fixture did not exercise BFS bucket retirement\n";return false;
    }
    std::cout<<"same-group repeated children, distinct source-point radius neighborhoods, and density/mass replay PASS\n";
    return true;
}

int main(){return runSelectiveRadiusGroupBucketControl()?0:1;}
