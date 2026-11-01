#include "NumiHumanCurrentFractionalRecordedFlowArtifacts.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {
void appendParent(PhysicalSkinChildMassMappingBundle& bundle,
                  const std::array<V3,4>& points,
                  const std::array<std::uint32_t,4>& tet,
                  const std::uint32_t material,const std::uint32_t donor,
                  const std::uint32_t region,const std::uint32_t face,
                  const std::uint32_t globalTet) {
    const auto parentRow=static_cast<std::uint32_t>(bundle.parent.tetrahedra.size());
    bundle.parent.tetrahedra.push_back(tet);
    bundle.parent.referenceTetVolumeM3.push_back(1.0/6.0);
    bundle.parent.tetrahedronMassKg.push_back(1.0);
    bundle.parent.materialIndices.push_back(material);
    bundle.parent.donorBodyIndices.push_back(donor);
    bundle.parent.anatomicalRegionIndices.push_back(region);
    bundle.parent.stableSourceFaceIdentities.push_back(face);
    bundle.childTetrahedra.push_back(tet);
    bundle.childMaterialIndices.push_back(material);
    bundle.childDonorBodyIndices.push_back(donor);
    bundle.childAnatomicalRegionIndices.push_back(region);
    bundle.childStableSourceFaceIdentities.push_back(face);
    bundle.parent.totalMassKg+=1.0;
    for(std::uint32_t corner=0;corner<4u;++corner){
        const auto node=tet[corner];
        bundle.parent.sourceMassContributions.push_back({
            .parentNodeIndex=node,.materialIndex=material,.donorBodyIndex=donor,
            .massKg=0.25,.donorLocalPosition=points[corner]});
    }
    (void)parentRow;
    (void)globalTet;
}
std::string hash(char c){return std::string(64u,c);}
bool check(bool ok,const std::string& what){
    if(!ok)std::cerr<<"FAIL "<<what<<'\n';
    return ok;
}
}

int main() {
    const auto root=std::filesystem::temp_directory_path()/"numi-current-recorded-flow-artifact-check";
    std::error_code ec;std::filesystem::remove_all(root,ec);
    std::filesystem::create_directories(root,ec);
    if(ec){std::cerr<<"cannot create fixture directory\n";return 2;}

    PhysicalSkinChildMassMappingBundle bundle;
    const std::array<V3,8> points{{
        V3{0,0,0},V3{1,0,0},V3{0,1,0},V3{0,0,1},
        V3{3,0,0},V3{4,0,0},V3{3,1,0},V3{3,0,1}}};
    bundle.parent.referenceWorldPoints.assign(points.begin(),points.end());
    bundle.referenceWorldPoints.assign(points.begin(),points.end());
    bundle.currentWorldPoints.assign(points.begin(),points.end());
    appendParent(bundle,{points[0],points[1],points[2],points[3]},{0,1,2,3},15,7,4,100,1000);
    appendParent(bundle,{points[4],points[5],points[6],points[7]},{4,5,6,7},16,8,5,200,2000);
    bundle.parent.parentMassPacketSHA256=hash('a');
    bundle.parent.parentSourceContributionSHA256=hash('b');
    bundle.childMeshSHA256=hash('c');bundle.regionLabelSHA256=hash('d');
    bundle.coordinateFrameSHA256=hash('e');

    ConservativeSpatialMassTransportResult result;
    result.childParentOffsets={0,1,2};result.childParentRows={0,1};
    result.childParentTransportMassKg={1,1};
    result.childParentTransportMeasureM3={1.0/6.0,1.0/6.0};
    result.childParentTransportDistanceM={0.001,0.002};
    result.childParentVolumeFractions={1,1};
    result.childAncestryMeasureVolumeM3={1.0/6.0,1.0/6.0};
    result.childMassKg={1,1};result.childImpliedDensityKgM3={6,6};
    result.childDominantParentRows={0,1};
    result.childParentOffsets={0,1,2};
    result.childSourceDonorOffsets={0,1,2};
    result.childSourceDonorBodyIndices={7,8};
    result.childSourceDonorMassKg={1,1};
    result.transportEdgeCount=2;result.augmentationCount=3;
    result.semanticCandidateEdgeCount=8;result.receiptSHA256=hash('f');

    SourceCellCarrierQuadratureInput carrier;
    const std::vector<std::uint32_t> parentOffsets{0,1,2},globalTets{1000,2000};
    const std::vector<std::array<std::uint32_t,4>> nodeRows{{0,1,2,3},{4,5,6,7}};
    const std::vector<std::array<std::array<double,3>,4>> tetPoints{
        {{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}},
        {{{3,0,0},{4,0,0},{3,1,0},{3,0,1}}}};
    const std::vector<double> weights{1,1};
    const std::vector<std::array<double,4>> bary{{0.25,0.25,0.25,0.25},{0.25,0.25,0.25,0.25}};
    const std::vector<std::uint32_t> witnesses{0,1};
    const std::vector<double> distances{0.001,0.002};
    carrier.parentOffsets=parentOffsets;carrier.globalSourceTetRows=globalTets;
    carrier.sourceTetNodeRows=nodeRows;carrier.parentReferenceTetWorldVertices=tetPoints;
    carrier.carrierWeights=weights;carrier.carrierBarycentric=bary;
    carrier.witnessChildRows=witnesses;carrier.witnessChildCentroidDistancesM=distances;
    const auto packetHash=hash('1'),tetHash=hash('2'),nodeHash=hash('3');
    carrier.sourceCarrierPacketSHA256=packetHash;carrier.sourceTetTableSHA256=tetHash;
    carrier.sourceNodeTableSHA256=nodeHash;

    SourceCellCarrierTransportTrace trace;
    trace.expandedParentToSourceParentRows={0,1};trace.expandedParentToCarrierRows={0,1};
    trace.childParentCarrierOffsets={0,1,2};trace.childParentCarrierRows={0,1};
    trace.childParentCarrierMassKg={1,1};
    trace.childParentCarrierMeasureM3={1.0/6.0,1.0/6.0};
    trace.childParentCarrierDistanceM={0.001,0.002};

    const std::array<std::string,2> supportSHA{hash('4'),hash('5')};
    const std::array<std::string,2> packetSHA{hash('1'),hash('6')};
    const std::array<std::array<std::string,10>,2> inputSHA{{
        {hash('a'),hash('b'),hash('c'),hash('d'),hash('e'),hash('7'),hash('8'),hash('9'),hash('4'),hash('1')},
        {hash('a'),hash('b'),hash('c'),hash('d'),hash('e'),hash('7'),hash('8'),hash('9'),hash('5'),hash('6')}}};
    CurrentRecordedRoleFlowArtifacts artifacts;std::string error;
    if(!writeCurrentFractionalRecordedRoleFlowArtifacts(root,bundle,result,trace,carrier,
            supportSHA,packetSHA,inputSHA,artifacts,error)){
        std::cerr<<"serializer rejected tiny paired fixture: "<<error<<'\n';return 3;
    }
    bool okay=true;
    std::array<PhysicalSkinSourceCarrierRoleRecordedAllocationData,2> importedRoles{};
    std::array<PhysicalSkinRecordedCarrierAllocationCSVAssets,2> importedAssets{};
    for(unsigned role=0;role<2u;++role){
        auto& assets=importedAssets[role];
        assets.materialIndex=role==0u?15u:16u;
        assets.csvPaths=artifacts.csvPaths[role];assets.csvSHA256=artifacts.csvSHA256[role];
        assets.receiptPath=artifacts.receiptPaths[role];assets.receiptSHA256=artifacts.receiptSHA256[role];
        assets.expectedInputSHA256.assign(inputSHA[role].begin(),inputSHA[role].end());
        auto& roleData=importedRoles[role];
        PhysicalSkinRecordedCarrierAllocationCSVImportReport report;
        if(!recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(
            assets,assets.materialIndex,1,1,roleData,report,error)){
            std::cerr<<"importer rejected serialized tiny role "<<role<<": "<<error<<'\n';return 4;
        }
        okay&=check(roleData.pairedSolverMetrics&&roleData.pairedSolverRunSHA256==result.receiptSHA256,
                    "paired receipt metadata did not roundtrip");
        okay&=check(report.aggregateFlowRows==1&&report.carrierFlowRows==1&&report.sourceDonorRows==1,
                    "six role CSV row domains did not roundtrip");
        okay&=check(roleData.sourceGlobalTetRows==std::vector<std::uint32_t>{globalTets[role]},
                    "source parent global row identity changed");
    }
    std::uint64_t pairedAugmentations=0,pairedEdges=0;
    okay&=check(combineRecordedFlowSolverMetrics(importedRoles,pairedAugmentations,pairedEdges,error)&&
                pairedAugmentations==3&&pairedEdges==8,
                "paired counters from imported receipts should count once");
    auto badPin=importedAssets[1];badPin.csvSHA256[0]=hash('0');
    PhysicalSkinSourceCarrierRoleRecordedAllocationData rejectedRole;
    PhysicalSkinRecordedCarrierAllocationCSVImportReport rejectedReport;
    okay&=check(!recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(
                badPin,16,1,1,rejectedRole,rejectedReport,error),
                "tampered role CSV pin should reject");
    auto roles=importedRoles;
    okay&=check(combineRecordedFlowSolverMetrics(roles,pairedAugmentations,pairedEdges,error)&&
                pairedAugmentations==3&&pairedEdges==8,
                "paired solver counters should count once");
    roles[1].pairedSolverRunSHA256=hash('0');
    okay&=check(!combineRecordedFlowSolverMetrics(roles,pairedAugmentations,pairedEdges,error),
                "mismatched paired solver run should reject");
    roles[1].pairedSolverRunSHA256=result.receiptSHA256;roles[1].augmentationCount++;
    okay&=check(!combineRecordedFlowSolverMetrics(roles,pairedAugmentations,pairedEdges,error),
                "mismatched paired counters should reject");
    roles[1].augmentationCount=roles[0].augmentationCount;roles[1].pairedSolverMetrics=false;
    okay&=check(!combineRecordedFlowSolverMetrics(roles,pairedAugmentations,pairedEdges,error),
                "one missing paired role marker should reject");
    std::array<PhysicalSkinSourceCarrierRoleRecordedAllocationData,2> legacy{};
    legacy[0].augmentationCount=2;legacy[0].semanticCandidateEdgeCount=11;
    legacy[1].augmentationCount=3;legacy[1].semanticCandidateEdgeCount=13;
    okay&=check(combineRecordedFlowSolverMetrics(legacy,pairedAugmentations,pairedEdges,error)&&
                pairedAugmentations==5&&pairedEdges==24,
                "legacy per-role solver counters should continue to sum");
    if(!okay)return 5;
    std::filesystem::remove_all(root,ec);
    std::cout<<"PASS_CURRENT_FRACTIONAL_RECORDED_FLOW_ARTIFACTS\n";
    return 0;
}
