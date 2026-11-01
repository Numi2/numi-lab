#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

bool runMaximumPhaseImplicitForestControl() {
    const std::array<SpatialMassTransportPoint,2> parentPoints{{
        SpatialMassTransportPoint{0.0,0.0,0.0},
        SpatialMassTransportPoint{0.875,0.0,0.0}}};
    const std::array<double,2> parentVolumes{1.0,1.5};
    const std::array<double,2> parentMasses{1.0,1.5};
    const std::array<std::uint32_t,2> parentMaterials{15,15};
    const std::array<std::uint32_t,2> parentDonors{1,1};
    const std::array<std::uint32_t,2> parentRegions{10,10};
    const std::array<std::uint32_t,3> donorOffsets{0,1,2};
    const std::array<std::uint32_t,2> donorRows{1,1};
    const std::array<double,2> donorMasses{1.0,1.5};
    const std::array<SpatialMassTransportPoint,5> childPoints{{
        SpatialMassTransportPoint{0.125,0.0,0.0},
        SpatialMassTransportPoint{0.25,0.0,0.0},
        SpatialMassTransportPoint{0.375,0.0,0.0},
        SpatialMassTransportPoint{0.5,0.0,0.0},
        SpatialMassTransportPoint{0.875,0.0,0.0}}};
    const std::array<double,5> childVolumes{0.125,0.125,0.125,0.125,1.5};
    const std::array<std::uint32_t,5> childMaterials{15,15,15,15,15};
    const std::array<std::uint32_t,5> childDonors{1,1,1,1,1};
    const std::array<std::uint32_t,5> childRegions{10,10,10,10,10};
    std::array<double,16> densityLower{};densityLower.fill(1.0);
    std::array<double,16> densityUpper{};densityUpper.fill(1.5);
    const std::array<std::uint32_t,6> childOffsets{0,1,2,3,4,5};
    const std::array<std::uint32_t,5> slotGroups{0,0,0,0,0};
    const std::array<double,5> slotFractions{1,1,1,1,1};
    const std::array<std::uint32_t,1> groupMaterial{15};
    const std::array<std::uint32_t,1> groupDonor{1};
    const std::array<std::uint32_t,1> groupRegion{10};
    const std::array<std::uint64_t,1> groupIdentity{0x8000000000000010ull};
    const std::array<std::uint32_t,2> parentStableFaces{501,502};
    const std::array<std::uint32_t,2> groupParentOffsets{0,2};
    const std::array<std::uint32_t,2> groupParentRows{0,1};
    const std::array<std::uint32_t,2> groupFaceOffsets{0,2};
    const std::array<std::uint32_t,2> groupFaceRows{501,502};
    const std::string pin(64,'b');

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
    input.maximumTransportDistanceM=1.0;input.maximumAllowedEdges=100;input.maximumAugmentations=100;

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
    if(!assembleConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)){
        std::cerr<<"maximum-phase implicit forest failed: "<<error<<"\n";return false;
    }
    if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)){
        std::cerr<<"maximum-phase implicit forest replay failed: "<<error<<"\n";return false;
    }
    const double total=std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0);
    bool p0SuppliesLargeChild=false;
    for(std::uint32_t row=result.childParentOffsets[4];row<result.childParentOffsets[5];++row)
        p0SuppliesLargeChild|=result.childParentRows[row]==0u;
    if(std::abs(total-2.5)>1.0e-12||result.semanticCandidateEdgeCount!=9u||!p0SuppliesLargeChild){
        std::cerr<<"maximum-phase implicit forest did not add the omitted broad-capacity edge"
                 <<" total="<<total<<" edges="<<result.semanticCandidateEdgeCount
                 <<" p0_large_child="<<(p0SuppliesLargeChild?1:0)<<"\n";return false;
    }
    std::cout<<"maximum-phase ordinary-sink implicit residual forest PASS\n";
    return true;
}

std::vector<std::string> parseCSVFields(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted=false;
    for(std::size_t i=0;i<line.size();++i){
        const char ch=line[i];
        if(ch=='"'){
            if(quoted&&i+1u<line.size()&&line[i+1u]=='"'){field.push_back('"');++i;}
            else quoted=!quoted;
        }else if(ch==','&&!quoted){
            fields.push_back(field);field.clear();
        }else field.push_back(ch);
    }
    fields.push_back(field);
    return fields;
}

std::vector<std::vector<std::string>> parseWitnessSection(const std::string& witness,
                                                          const std::string& begin,
                                                          const std::string& end) {
    const auto start=witness.find(begin);
    if(start==std::string::npos)return {};
    const auto contentStart=witness.find('\n',start);
    const auto finish=witness.find(end,contentStart);
    if(contentStart==std::string::npos||finish==std::string::npos)return {};
    std::istringstream rows(witness.substr(contentStart+1u,finish-contentStart-1u));
    std::vector<std::vector<std::string>> parsed;
    std::string line;
    while(std::getline(rows,line))
        if(!line.empty())parsed.push_back(parseCSVFields(line));
    return parsed;
}

bool runFullImplicitCutWitnessControl() {
    const std::array<SpatialMassTransportPoint,2> parentPoints{{
        SpatialMassTransportPoint{0.0,0.0,0.0},
        SpatialMassTransportPoint{10.0,0.0,0.0}}};
    const std::array<double,2> parentVolumes{0.0011,0.0001};
    const std::array<double,2> parentMasses{1.1,0.1};
    const std::array<std::uint32_t,2> parentMaterials{15,15};
    const std::array<std::uint32_t,2> parentDonors{1,1};
    const std::array<std::uint32_t,2> parentRegions{10,10};
    const std::array<std::uint32_t,3> donorOffsets{0,1,2};
    const std::array<std::uint32_t,2> donorRows{1,1};
    const std::array<double,2> donorMasses{1.1,0.1};
    const std::array<SpatialMassTransportPoint,2> childPoints{{
        SpatialMassTransportPoint{0.0,0.0,0.0},
        SpatialMassTransportPoint{10.0,0.0,0.0}}};
    const std::array<double,2> childVolumes{0.0005,0.0006};
    const std::array<std::uint32_t,2> childMaterials{15,15};
    const std::array<std::uint32_t,2> childDonors{1,1};
    const std::array<std::uint32_t,2> childRegions{10,10};
    std::array<double,16> densityLower{};densityLower.fill(999.0);
    std::array<double,16> densityUpper{};densityUpper.fill(2000.0);
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
    const std::string pin(64,'c');

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
    input.maximumTransportDistanceM=1.0;input.maximumAllowedEdges=100;input.maximumAugmentations=100;

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
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)||
       error.find("full_implicit_residual_closure=exhausted") == std::string::npos||
       error.find("no_augmenting_path=1") == std::string::npos||
       error.find("residual_child_witnesses=") == std::string::npos){
        std::cerr<<"full implicit cut witness control failed: "<<error<<"\n";return false;
    }
    const auto marker=error.find("shared_capacity_full_cut_begin");
    if(marker==std::string::npos){std::cerr<<"full cut witness marker missing\n";return false;}
    const std::string witness=error.substr(marker);
    const auto firstLineEnd=witness.find('\n');
    if(firstLineEnd==std::string::npos||
       witness.find("phase=minimum_flow")==std::string::npos||
       witness.find("cut_capacity_minus_flow_kg=")==std::string::npos||
       witness.find("residual_child_rows=1")==std::string::npos||
       witness.find("unreachable_residual_child_rows=1")==std::string::npos){
        std::cerr<<"full cut witness metadata/newline roundtrip failed\n";return false;
    }
    const auto groupRows=parseWitnessSection(witness,"group_summary_csv_begin","group_summary_csv_end");
    const auto parentRows=parseWitnessSection(witness,"parent_group_rows_csv_begin","parent_group_rows_csv_end");
    const auto childRows=parseWitnessSection(witness,"residual_child_rows_csv_begin","residual_child_rows_csv_end");
    if(groupRows.size()!=2u||groupRows[0].size()!=8u||
       parentRows.size()!=3u||parentRows[0].size()!=13u||parentRows[1].size()!=13u||
       parentRows[1][0]!="0"||parentRows[1][5]!="0"||parentRows[1][9]!="1.1000000000000001"||
       parentRows[1][11]!="0;1"||parentRows[1][12]!="501;502"||
       childRows.size()!=2u||childRows[0].size()!=10u||
       childRows[1][0]!="1"||childRows[1][1]!="15"||
       childRows[1][9].find("0:15:1:10:")==std::string::npos){
        std::cerr<<"full cut witness CSV section roundtrip failed groups="<<groupRows.size()
                 <<" parents="<<parentRows.size()<<" children="<<childRows.size()<<"\n";
        return false;
    }
    std::cout<<"full implicit residual cut certificate and CSV roundtrip control PASS\n";
    return true;
}

int main() {
    static const std::array<SpatialMassTransportPoint, 2> parentPoints{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.85, 0.0, 0.0}}};
    static const std::array<double, 2> parentVolumes{0.002, 0.0004};
    static const std::array<double, 2> parentMasses{2.0, 0.4};
    static const std::array<std::uint32_t, 2> parentMaterials{15, 15};
    static const std::array<std::uint32_t, 2> parentDonors{1, 1};
    static const std::array<std::uint32_t, 2> parentRegions{10, 10};
    static const std::array<std::uint32_t, 3> donorOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> donorRows{1, 1};
    static const std::array<double, 2> donorMasses{2.0, 0.4};
    static const std::array<SpatialMassTransportPoint, 6> childPoints{{
        SpatialMassTransportPoint{0.10, 0.0, 0.0},
        SpatialMassTransportPoint{0.20, 0.0, 0.0},
        SpatialMassTransportPoint{0.25, 0.0, 0.0},
        SpatialMassTransportPoint{0.30, 0.0, 0.0},
        SpatialMassTransportPoint{0.80, 0.0, 0.0},
        SpatialMassTransportPoint{0.90, 0.0, 0.0}}};
    static const std::array<double, 6> childVolumes{
        0.0003, 0.0003, 0.0003, 0.0003, 0.0006, 0.0006};
    static const std::array<std::uint32_t, 6> childMaterials{15, 15, 15, 15, 15, 15};
    static const std::array<std::uint32_t, 6> childDonors{1, 1, 1, 1, 1, 1};
    static const std::array<std::uint32_t, 6> childRegions{10, 10, 10, 10, 10, 10};
    static const std::array<double, 16> densityLower=[] {
        std::array<double, 16> out{}; out.fill(999.0); return out;
    }();
    static const std::array<double, 16> densityUpper=[] {
        std::array<double, 16> out{}; out.fill(1001.0); return out;
    }();
    static const std::array<std::uint32_t, 7> childOffsets{0, 1, 2, 3, 4, 5, 6};
    static const std::array<std::uint32_t, 6> slotGroups{0, 0, 0, 0, 0, 0};
    static const std::array<double, 6> slotFractions{1, 1, 1, 1, 1, 1};
    static const std::array<std::uint32_t, 1> groupMaterial{15};
    static const std::array<std::uint32_t, 1> groupDonor{1};
    static const std::array<std::uint32_t, 1> groupRegion{10};
    static const std::array<std::uint64_t, 1> groupIdentity{0x8000000000000010ull};
    static const std::array<std::uint32_t, 2> parentStableFaces{501, 502};
    static const std::array<std::uint32_t, 2> groupParentOffsets{0, 2};
    static const std::array<std::uint32_t, 2> groupParentRows{0, 1};
    static const std::array<std::uint32_t, 2> groupFaceOffsets{0, 2};
    static const std::array<std::uint32_t, 2> groupFaceRows{501, 502};
    const std::string pin(64, 'a');

    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentPoints;
    input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMasses;
    input.parentMaterialIndices=parentMaterials;
    input.parentDonorBodyIndices=parentDonors;
    input.parentAnatomicalRegionIndices=parentRegions;
    input.parentSourceDonorOffsets=donorOffsets;
    input.parentSourceDonorBodyIndices=donorRows;
    input.parentSourceDonorMassKg=donorMasses;
    input.childCentroidsWorld=childPoints;
    input.childReferenceVolumeM3=childVolumes;
    input.childMaterialIndices=childMaterials;
    input.childDonorBodyIndices=childDonors;
    input.childAnatomicalRegionIndices=childRegions;
    input.materialDensityLowerKgM3=densityLower;
    input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=pin;
    input.parentSourceContributionSHA256=pin;
    input.childMeshSHA256=pin;
    input.regionLabelSHA256=pin;
    input.coordinateFrameSHA256=pin;
    input.maximumTransportDistanceM=1.0;
    input.maximumAllowedEdges=100;
    input.maximumAugmentations=100;

    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;
    support.slotGroupIndices=slotGroups;
    support.volumeFractions=slotFractions;
    support.groupMaterialIndices=groupMaterial;
    support.donorBodyIndices=groupDonor;
    support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=parentStableFaces;
    support.sourceParentTetOffsets=groupParentOffsets;
    support.sourceParentTetRows=groupParentRows;
    support.sourceStableFaceOffsets=groupFaceOffsets;
    support.sourceStableFaceRows=groupFaceRows;
    support.typedSupportSHA256=pin;

    ConservativeSpatialMassTransportResult result;
    std::string error;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(input, support, result, error)) {
        std::cerr << "optimization trial fixture failed: " << error << "\n";
        return 1;
    }
    const double childMass=std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0);
    if (result.semanticCandidateEdgeCount != 10u || result.childMassKg.size()!=6u ||
        std::abs(childMass-2.4)>1e-12 ||
        !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input, support, result, error)) {
        std::cerr << "optimization trial replay mismatch edges=" << result.semanticCandidateEdgeCount
                  << " error=" << error << "\n";
        return 2;
    }
    if(!runMaximumPhaseImplicitForestControl()||!runFullImplicitCutWitnessControl())return 3;
    std::cout << "optimized parent-K4/reverse-child-K1/full-implicit-residual-forest exact-capacity fixture PASS\n";
    return 0;
}
