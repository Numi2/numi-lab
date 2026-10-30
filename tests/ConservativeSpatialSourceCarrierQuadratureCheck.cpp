#include "ConservativeSpatialSourceCarrierQuadrature.hpp"
#include "NumiHumanRecordedCarrierAllocationImport.hpp"
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using P = SpatialMassTransportPoint;
using B = std::array<double,4>;
using Tet = std::array<P,4>;

static P point(const Tet& t,const B& b) {
    P p{}; for (std::size_t i=0;i<4;++i) for(std::size_t a=0;a<3;++a) p[a]+=b[i]*t[i][a]; return p;
}
static double dist(const P&a,const P&b) {
    double q=0; for(int i=0;i<3;++i){double d=a[i]-b[i];q+=d*d;} return std::sqrt(q);
}
struct Fixture {
    double s=std::cbrt(0.006);
    Tet tet{{P{0,0,0},P{s,0,0},P{0,s,0},P{0,0,s}}};
    std::vector<P> pc{P{s/4,s/4,s/4}}, cc;
    std::vector<double> pv{1.0e-3}, pm{1.0}, cv{2.5e-4,2.5e-4,2.5e-4,2.5e-4};
    std::vector<std::uint32_t> pmat{15}, pdonor{145}, pregion{145}, poff{0,1}, pdids{145};
    std::vector<double> pdmass{1.0};
    std::vector<std::uint32_t> cmat{15,15,15,15}, cdonor{145,145,145,145}, cregion{145,145,145,145};
    std::vector<double> lower=std::vector<double>(16,900.0), upper=std::vector<double>(16,1100.0);
    std::vector<std::uint32_t> childOff{0,1,2,3,4}, slotGroup{0,0,0,0}, gm{15}, gd{145}, gr{145};
    std::vector<double> preference{1,1,1,1};
    std::vector<std::uint64_t> gids{0x8000000000000100ull};
    std::vector<std::uint32_t> stable{77608}, gpoff{0,1}, gprows{0}, gfoff{0,1}, gfrows{77608};
    std::vector<std::uint32_t> carrierOff{0,4}, tetRows{9908}, tetNodes{0,1,2,3}, witness{0,1,2,3};
    std::vector<double> weights{.25,.25,.25,.25}, witnessDists{0,0,0,0};
    std::vector<B> bary;
    std::vector<Tet> parentTets{tet};
    std::string parentHash=std::string(64,'a'), sourceHash=std::string(64,'b'), childHash=std::string(64,'c'), regionHash=std::string(64,'d'), frameHash=std::string(64,'e'), supportHash=std::string(64,'f'), carrierHash=std::string(64,'1'), tetHash=std::string(64,'2'), nodeHash=std::string(64,'3');
    ConservativeSpatialMassTransportInput in;
    ConservativeSpatialChildSemanticSupport support;
    SourceCellCarrierQuadratureInput carriers;

    Fixture() {
        bary.resize(4);
        for(std::size_t r=0;r<4;++r) {
            for(std::size_t i=0;i<4;++i) bary[r][i]=0.25+0.3*(i==r?0.75:-0.25);
            cc.push_back(point(tet,bary[r]));
        }
        std::array<std::uint32_t,4> tr{0,1,2,3};
        sourceTetNodeRows.push_back(tr);
        witnessDists.resize(4);
        for(std::size_t i=0;i<4;++i) witnessDists[i]=dist(cc[i],point(tet,bary[i]));
        in.parentCentroidsWorld=pc; in.parentReferenceVolumeM3=pv; in.parentMassKg=pm;
        in.parentMaterialIndices=pmat; in.parentDonorBodyIndices=pdonor; in.parentAnatomicalRegionIndices=pregion;
        in.parentSourceDonorOffsets=poff; in.parentSourceDonorBodyIndices=pdids; in.parentSourceDonorMassKg=pdmass;
        in.childCentroidsWorld=cc; in.childReferenceVolumeM3=cv; in.childMaterialIndices=cmat;
        in.childDonorBodyIndices=cdonor; in.childAnatomicalRegionIndices=cregion;
        in.materialDensityLowerKgM3=lower; in.materialDensityUpperKgM3=upper;
        in.parentMassPacketSHA256=parentHash; in.parentSourceContributionSHA256=sourceHash;
        in.childMeshSHA256=childHash; in.regionLabelSHA256=regionHash;
        in.coordinateFrameSHA256=frameHash; in.maximumTransportDistanceM=0.02;
        in.maximumAllowedEdges=100; in.maximumAugmentations=100;
        support.childOffsets=childOff; support.slotGroupIndices=slotGroup; support.volumeFractions=preference;
        support.groupMaterialIndices=gm; support.donorBodyIndices=gd; support.anatomicalRegionIndices=gr;
        support.derivedGroupIdentityIDs=gids; support.parentStableSourceFaceIdentities=stable;
        support.sourceParentTetOffsets=gpoff; support.sourceParentTetRows=gprows;
        support.sourceStableFaceOffsets=gfoff; support.sourceStableFaceRows=gfrows;
        support.typedSupportSHA256=supportHash;
        carriers.parentOffsets=carrierOff; carriers.globalSourceTetRows=tetRows;
        carriers.sourceTetNodeRows=sourceTetNodeRows; carriers.parentReferenceTetWorldVertices=parentTets;
        carriers.carrierWeights=weights; carriers.carrierBarycentric=bary;
        carriers.witnessChildRows=witness; carriers.witnessChildCentroidDistancesM=witnessDists;
        carriers.sourceCarrierPacketSHA256=carrierHash;
        carriers.sourceTetTableSHA256=tetHash; carriers.sourceNodeTableSHA256=nodeHash;
    }
    std::vector<std::array<std::uint32_t,4>> sourceTetNodeRows;
};



static bool recordedRoleCSVImportFixture(
        const Fixture& f,const ConservativeSpatialMassTransportResult& original,
        const SourceCellCarrierTransportTrace& originalTrace) {
    namespace fs=std::filesystem;
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root=fs::temp_directory_path()/("numi-recorded-carrier-723-"+std::to_string(stamp));
    fs::create_directories(root);
    const std::array<std::string,6> names{
        "transport-child-mass.csv","transport-flow.csv","source-carrier-flow.csv",
        "transport-child-source-donor.csv","transport-source-parent.csv","transport-source-parent-donor.csv"};
    std::array<fs::path,6> paths;std::string error;
    for(std::size_t i=0;i<names.size();++i)paths[i]=root/names[i];
    {
        std::ofstream o(paths[0]);o<<std::setprecision(17)<<"child_row,reference_volume_m3,mass_kg,density_kg_m3\n";
        for(std::size_t c=0;c<f.cv.size();++c)o<<c<<","<<f.cv[c]<<","<<original.childMassKg[c]<<","<<original.childImpliedDensityKgM3[c]<<"\n";
    }
    {
        std::ofstream o(paths[1]);o<<std::setprecision(17)<<"child_row,parent_row,mass_kg,source_measure_m3,distance_m\n";
        for(std::size_t c=0;c<f.cv.size();++c)
            for(std::uint32_t e=original.childParentOffsets[c];e<original.childParentOffsets[c+1u];++e)
                o<<c<<","<<original.childParentRows[e]<<","<<original.childParentTransportMassKg[e]<<","
                 <<original.childParentTransportMeasureM3[e]<<","<<original.childParentTransportDistanceM[e]<<"\n";
    }
    {
        std::ofstream o(paths[2]);o<<std::setprecision(17)<<"child_row,parent_row,expanded_parent_row,carrier_row,mass_kg,source_measure_m3,distance_m\n";
        for(std::size_t c=0;c<f.cv.size();++c)
            for(std::uint32_t e=original.childParentOffsets[c];e<original.childParentOffsets[c+1u];++e)
                for(std::uint32_t k=originalTrace.childParentCarrierOffsets[e];k<originalTrace.childParentCarrierOffsets[e+1u];++k){
                    const auto v=originalTrace.childParentCarrierRows[k];
                    o<<c<<","<<original.childParentRows[e]<<","<<v<<","<<originalTrace.expandedParentToCarrierRows[v]<<","
                     <<originalTrace.childParentCarrierMassKg[k]<<","<<originalTrace.childParentCarrierMeasureM3[k]<<","
                     <<originalTrace.childParentCarrierDistanceM[k]<<"\n";
                }
    }
    {
        std::ofstream o(paths[3]);o<<std::setprecision(17)<<"child_row,source_donor_body,mass_kg\n";
        for(std::size_t c=0;c<f.cv.size();++c)
            for(std::uint32_t d=original.childSourceDonorOffsets[c];d<original.childSourceDonorOffsets[c+1u];++d)
                o<<c<<","<<original.childSourceDonorBodyIndices[d]<<","<<original.childSourceDonorMassKg[d]<<"\n";
    }
    {
        std::ofstream o(paths[4]);o<<std::setprecision(17)
            <<"parent_row,global_source330_row,material,geometric_donor,region,stable_source_face,mass_kg,reference_volume_m3,source_centroid_x_m,source_centroid_y_m,source_centroid_z_m\n";
        o<<"0,"<<f.tetRows[0]<<","<<f.in.parentMaterialIndices[0]<<","<<f.in.parentDonorBodyIndices[0]<<","
         <<f.in.parentAnatomicalRegionIndices[0]<<","<<f.support.parentStableSourceFaceIdentities[0]<<","
         <<f.in.parentMassKg[0]<<","<<f.in.parentReferenceVolumeM3[0]<<","<<f.in.parentCentroidsWorld[0][0]<<","
         <<f.in.parentCentroidsWorld[0][1]<<","<<f.in.parentCentroidsWorld[0][2]<<"\n";
    }
    {
        std::ofstream o(paths[5]);o<<std::setprecision(17)<<"parent_row,source_donor_body,mass_kg\n";
        for(std::uint32_t d=f.in.parentSourceDonorOffsets[0];d<f.in.parentSourceDonorOffsets[1];++d)
            o<<"0,"<<f.in.parentSourceDonorBodyIndices[d]<<","<<f.in.parentSourceDonorMassKg[d]<<"\n";
    }
    std::array<std::string,6> hashes{};std::array<std::uintmax_t,6> sizes{};
    for(std::size_t i=0;i<paths.size();++i){
        if(!physical_skin_child_mass_detail::sha256File(paths[i],hashes[i],error)){
            std::cerr<<"fixture CSV hash failed: "<<error<<"\n";return false;
        }
        sizes[i]=fs::file_size(paths[i]);
    }
    const auto receiptPath=root/"allocation-receipt.json";
    {
        std::ofstream o(receiptPath);o<<"{\n"
            <<" \"status\": \"PASS_SOURCE_ONLY_SKIN_SOURCE_CARRIER_SHARED_CAPACITY_ALLOCATION\",\n"
            <<" \"physical_admission\": false, \"native_steps\": 0,\n"
            <<" \"recorded_flow_deterministic_replay\": \"PASS\",\n"
            <<" \"parent_skin_tets\": 1, \"child_skin_tets\": 4,\n"
            <<" \"transport_edges\": "<<original.transportEdgeCount<<", \"candidate_edges\": "<<original.semanticCandidateEdgeCount
            <<", \"augmentations\": "<<original.augmentationCount<<",\n"
            <<" \"source_carrier_packet_sha256\": \""<<f.carrierHash<<"\",\n"
            <<" \"source_carrier_array_sha256\": \""<<originalTrace.sourceCarrierArraySHA256<<"\",\n"
            <<" \"source_carrier_flow_trace_sha256\": \""<<originalTrace.receiptSHA256<<"\",\n"
            <<" \"transport_receipt_sha256\": \""<<original.receiptSHA256<<"\",\n"
            <<" \"input_parent_mass_sha256\": \""<<f.parentHash<<"\",\n"
            <<" \"input_parent_source_sha256\": \""<<f.sourceHash<<"\",\n"
            <<" \"child_mesh_sha256\": \""<<f.childHash<<"\",\n"
            <<" \"region_label_sha256\": \""<<f.regionHash<<"\",\n"
            <<" \"coordinate_frame_sha256\": \""<<f.frameHash<<"\",\n"
            <<" \"flow_csv\": {";
        for(std::size_t i=0;i<names.size();++i){
            if(i)o<<",";
            o<<" \""<<names[i]<<"\": {\"sha256\": \""<<hashes[i]<<"\", \"size_bytes\": "<<sizes[i]<<"}";
        }
        o<<" }\n}\n";
    }
    PhysicalSkinRecordedCarrierAllocationCSVAssets assets;
    assets.materialIndex=15u;assets.csvPaths=paths;assets.csvSHA256=hashes;
    assets.receiptPath=receiptPath;
    if(!physical_skin_child_mass_detail::sha256File(receiptPath,assets.receiptSHA256,error)){
        std::cerr<<"fixture receipt hash failed: "<<error<<"\n";return false;
    }
    assets.expectedInputSHA256={f.parentHash,f.sourceHash,f.childHash,f.regionHash,f.frameHash};
    PhysicalSkinSourceCarrierRoleRecordedAllocationData loaded;
    PhysicalSkinRecordedCarrierAllocationCSVImportReport report;
    if(!recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(assets,15u,1u,4u,loaded,report,error)){
        std::cerr<<"six-CSV role import rejected: "<<error<<"\n";return false;
    }
    loaded.result.parentMassPacketSHA256=f.in.parentMassPacketSHA256;
    loaded.result.parentSourceContributionSHA256=f.in.parentSourceContributionSHA256;
    loaded.result.childMeshSHA256=f.in.childMeshSHA256;loaded.result.regionLabelSHA256=f.in.regionLabelSHA256;
    loaded.result.coordinateFrameSHA256=f.in.coordinateFrameSHA256;
    loaded.result.maximumTransportDistanceM=f.in.maximumTransportDistanceM;
    loaded.result.relativeClosureTolerance=f.in.relativeClosureTolerance;
    loaded.result.absoluteMassToleranceKg=f.in.absoluteMassToleranceKg;
    loaded.result.absoluteMeasureToleranceM3=f.in.absoluteMeasureToleranceM3;
    loaded.result.maximumAllowedEdges=f.in.maximumAllowedEdges;
    loaded.result.maximumAugmentations=f.in.maximumAugmentations;
    loaded.trace.expandedParentToSourceParentRows=originalTrace.expandedParentToSourceParentRows;
    loaded.trace.expandedParentToCarrierRows=originalTrace.expandedParentToCarrierRows;
    loaded.trace.carrierSourceParentRows=originalTrace.carrierSourceParentRows;
    loaded.trace.originalCentroidNearestAdmissibleDistanceM=originalTrace.originalCentroidNearestAdmissibleDistanceM;
    loaded.trace.sourceCarrierPacketSHA256=originalTrace.sourceCarrierPacketSHA256;
    loaded.trace.sourceCarrierArraySHA256=originalTrace.sourceCarrierArraySHA256;
    loaded.trace.expandedSolverReceiptSHA256=originalTrace.expandedSolverReceiptSHA256;
    if(!canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            f.in,f.support,f.carriers,loaded.result,loaded.trace,error)){
        std::cerr<<"six-CSV role solver-free feasibility rejected: "<<error<<"\n";return false;
    }
    if(!report.verified||report.physicalAdmission||report.nativeSteps!=0u||
       loaded.result.childParentRows!=original.childParentRows||
       loaded.result.childParentTransportMassKg!=original.childParentTransportMassKg||
       loaded.result.childMassKg!=original.childMassKg||
       loaded.trace.childParentCarrierRows!=originalTrace.childParentCarrierRows||
       loaded.trace.childParentCarrierMassKg!=originalTrace.childParentCarrierMassKg){
        std::cerr<<"six-CSV role import changed current feasible rows or overstated admission\n";return false;
    }
    // A feasibility-only LP receipt is distinct from deterministic optimizer
    // replay. The same six CSVs must parse under the exact solver-free status.
    {
        std::ifstream receiptInput(receiptPath);
        std::string receiptText((std::istreambuf_iterator<char>(receiptInput)),{});
        const std::string oldStatus="PASS_SOURCE_ONLY_SKIN_SOURCE_CARRIER_SHARED_CAPACITY_ALLOCATION";
        const std::string newStatus="PASS_SOURCE_ONLY_SKIN_SOURCE_CARRIER_FEASIBLE_ALLOCATION";
        const auto statusAt=receiptText.find(oldStatus);
        const std::string replayLine=" \"recorded_flow_deterministic_replay\": \"PASS\",\n";
        if(statusAt==std::string::npos||receiptText.find(replayLine)==std::string::npos){
            std::cerr<<"solver-free receipt fixture source fields missing\n";return false;
        }
        receiptText.replace(statusAt,oldStatus.size(),newStatus);
        const auto replayAt=receiptText.find(replayLine);
        receiptText.replace(replayAt,replayLine.size(),
            " \"recorded_flow_feasibility\": \"PASS_SOLVER_FREE\",\n"
            " \"deterministic_optimizer_replay\": \"NOT_RUN\",\n"
            " \"complete_allocation\": true,\n");
        const auto solverFreeReceipt=root/"solver-free-allocation-receipt.json";
        {std::ofstream output(solverFreeReceipt,std::ios::binary);output<<receiptText;}
        auto solverFreeAssets=assets;solverFreeAssets.receiptPath=solverFreeReceipt;
        if(!physical_skin_child_mass_detail::sha256File(solverFreeReceipt,
                solverFreeAssets.receiptSHA256,error)){
            std::cerr<<"solver-free receipt hash failed: "<<error<<"\n";return false;
        }
        PhysicalSkinSourceCarrierRoleRecordedAllocationData solverFreeImported;
        if(!recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(
               solverFreeAssets,15u,1u,4u,solverFreeImported,report,error)||
           !report.verified||!report.solverFreeFeasibility||report.deterministicOptimizerReplay){
            std::cerr<<"solver-free complete allocation receipt was rejected or mislabeled: "<<error<<"\n";return false;
        }
        auto incompleteText=receiptText;
        const std::string completeTrue="\"complete_allocation\": true";
        const auto completeAt=incompleteText.find(completeTrue);
        if(completeAt==std::string::npos){std::cerr<<"solver-free complete-allocation fixture field missing\n";return false;}
        incompleteText.replace(completeAt,completeTrue.size(),"\"complete_allocation\": false");
        const auto incompleteReceipt=root/"solver-free-incomplete-receipt.json";
        {std::ofstream output(incompleteReceipt,std::ios::binary);output<<incompleteText;}
        auto incompleteAssets=solverFreeAssets;incompleteAssets.receiptPath=incompleteReceipt;
        if(!physical_skin_child_mass_detail::sha256File(incompleteReceipt,
                incompleteAssets.receiptSHA256,error)){
            std::cerr<<"incomplete receipt hash failed: "<<error<<"\n";return false;
        }
        if(recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(
               incompleteAssets,15u,1u,4u,solverFreeImported,report,error)){
            std::cerr<<"solver-free importer accepted an incomplete allocation receipt\n";return false;
        }
    }
    // Re-pin a syntactically valid but inconsistent donor ledger. Import must
    // read it, while the post-canonicalization comparison rejects the changed mass.
    {
        std::ifstream input(paths[3]);std::string donorCSV((std::istreambuf_iterator<char>(input)),{});
        const std::string oldRow="0,145,0.25\n",newRow="0,145,1.0\n";
        const auto rowAt=donorCSV.find(oldRow);
        if(rowAt==std::string::npos){std::cerr<<"repinned donor tamper fixture row missing\n";return false;}
        donorCSV.replace(rowAt,oldRow.size(),newRow);
        {std::ofstream output(paths[3],std::ios::binary|std::ios::trunc);output<<donorCSV;}
        std::string changedHash;
        if(!physical_skin_child_mass_detail::sha256File(paths[3],changedHash,error)){
            std::cerr<<"repinned donor tamper hash failed: "<<error<<"\n";return false;
        }
        std::ifstream receiptInput(receiptPath);std::string receiptText((std::istreambuf_iterator<char>(receiptInput)),{});
        const std::string oldEntry="\""+names[3]+"\": {\"sha256\": \""+hashes[3]+"\", \"size_bytes\": "+std::to_string(sizes[3])+"}";
        const std::string newEntry="\""+names[3]+"\": {\"sha256\": \""+changedHash+"\", \"size_bytes\": "+std::to_string(fs::file_size(paths[3]))+"}";
        const auto entryAt=receiptText.find(oldEntry);
        if(entryAt==std::string::npos){std::cerr<<"repinned donor tamper receipt entry missing\n";return false;}
        receiptText.replace(entryAt,oldEntry.size(),newEntry);
        {std::ofstream output(receiptPath,std::ios::binary|std::ios::trunc);output<<receiptText;}
        auto repinned=assets;repinned.csvSHA256[3]=changedHash;
        if(!physical_skin_child_mass_detail::sha256File(receiptPath,repinned.receiptSHA256,error)){
            std::cerr<<"repinned receipt hash failed: "<<error<<"\n";return false;
        }
        PhysicalSkinSourceCarrierRoleRecordedAllocationData changedLedger;
        if(!recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(
               repinned,15u,1u,4u,changedLedger,report,error)||
           !report.verified||changedLedger.result.childSourceDonorMassKg.size()!=4u||
           changedLedger.result.childSourceDonorMassKg[0]!=1.0||
           physical_skin_child_mass_mapping_detail::recordedRoleChildDonorLedgerMatches(
               changedLedger.result,0u,original,0u)){
            std::cerr<<"repinned changed donor ledger passed current flow comparison: "<<error<<"\n";return false;
        }
    }
    auto tampered=assets;tampered.expectedInputSHA256.push_back(std::string(64,'9'));
    PhysicalSkinSourceCarrierRoleRecordedAllocationData rejected;
    if(recorded_carrier_allocation_import_detail::loadPhysicalSkinRecordedCarrierAllocationCSVs(tampered,15u,1u,4u,rejected,report,error)){
        std::cerr<<"six-CSV role import accepted an unpinned current input\n";return false;
    }
    fs::remove_all(root);
    return true;
}

static bool rejects(Fixture f,const char* label) {
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    std::string error;
    if(assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"unexpected acceptance: "<<label<<"\n"; return false;
    }
    return true;
}

static bool oneHopGroupAbsentFromP1PreferenceFixture() {
    Fixture f;
    f.pc={f.pc[0],f.pc[0]};
    f.pv={5.0e-4,5.0e-4};f.pm={0.5,0.5};
    f.pmat={15,15};f.pdonor={145,145};f.pregion={145,146};
    f.poff={0,1,2};f.pdids={145,145};f.pdmass={0.5,0.5};
    f.slotGroup={0,0,0,0};f.gm={15,15};f.gd={145,145};f.gr={145,146};
    f.gids={0x8000000000000100ull,0x8000000000000101ull};
    f.stable={77608,77609};f.gpoff={0,1,2};f.gprows={0,1};
    f.gfoff={0,1,2};f.gfrows={77608,77609};
    f.carrierOff={0,4,8};f.tetRows={9908,9909};
    f.tetNodes={0,1,2,3,4,5,6,7};f.witness={0,1,2,3,0,1,2,3};
    f.weights.assign(8u,0.25);f.witnessDists.assign(8u,0.0);
    const auto originalBary=f.bary;
    f.bary.insert(f.bary.end(),originalBary.begin(),originalBary.end());
    f.parentTets={f.tet,f.tet};
    f.sourceTetNodeRows={std::array<std::uint32_t,4>{0,1,2,3},
                         std::array<std::uint32_t,4>{4,5,6,7}};
    const std::array<std::uint32_t,5> admissibleOffsets{0,2,4,6,8};
    const std::array<std::uint32_t,8> admissibleRegions{145,146,145,146,145,146,145,146};
    f.support.cellAdmissibleRegionOffsets=admissibleOffsets;
    f.support.cellAdmissibleRegionIds=admissibleRegions;
    const std::string adjacencyHash(64,'6');
    f.support.admissibilitySourceSupportSHA256=adjacencyHash;
    f.support.admissibilityMaximumSourceRegionHops=1u;

    f.in.parentCentroidsWorld=f.pc;f.in.parentReferenceVolumeM3=f.pv;f.in.parentMassKg=f.pm;
    f.in.parentMaterialIndices=f.pmat;f.in.parentDonorBodyIndices=f.pdonor;
    f.in.parentAnatomicalRegionIndices=f.pregion;f.in.parentSourceDonorOffsets=f.poff;
    f.in.parentSourceDonorBodyIndices=f.pdids;f.in.parentSourceDonorMassKg=f.pdmass;
    f.support.childOffsets=f.childOff;f.support.slotGroupIndices=f.slotGroup;
    f.support.volumeFractions=f.preference;f.support.groupMaterialIndices=f.gm;
    f.support.donorBodyIndices=f.gd;f.support.anatomicalRegionIndices=f.gr;
    f.support.derivedGroupIdentityIDs=f.gids;f.support.parentStableSourceFaceIdentities=f.stable;
    f.support.sourceParentTetOffsets=f.gpoff;f.support.sourceParentTetRows=f.gprows;
    f.support.sourceStableFaceOffsets=f.gfoff;f.support.sourceStableFaceRows=f.gfrows;
    f.carriers.parentOffsets=f.carrierOff;f.carriers.globalSourceTetRows=f.tetRows;
    f.carriers.sourceTetNodeRows=f.sourceTetNodeRows;f.carriers.parentReferenceTetWorldVertices=f.parentTets;
    f.carriers.carrierWeights=f.weights;f.carriers.carrierBarycentric=f.bary;
    f.carriers.witnessChildRows=f.witness;f.carriers.witnessChildCentroidDistancesM=f.witnessDists;

    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    std::string error;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
           f.in,f.support,f.carriers,result,trace,error)){
        std::cerr<<"one-hop source group absent from P1 preference was rejected: "<<error<<"\n";return false;
    }
    if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
           f.in,f.support,f.carriers,result,trace,error)){
        std::cerr<<"solver-free verification rejected one-hop source group absent from P1 preference: "<<error<<"\n";return false;
    }
    bool sawNonP1Parent=false;
    for(std::size_t child=0;child<f.cc.size();++child){
        for(std::uint32_t edge=result.childParentOffsets[child];edge<result.childParentOffsets[child+1u];++edge)
            if(result.childParentRows[edge]==1u)sawNonP1Parent=true;
    }
    if(!sawNonP1Parent){std::cerr<<"fixture did not exercise the one-hop-only source group\n";return false;}

    const std::array<std::uint32_t,4> p1OnlyOffsets{0,1,2,3};
    const std::array<std::uint32_t,4> p1OnlyRegions{145,145,145,145};
    f.support.cellAdmissibleRegionOffsets=p1OnlyOffsets;
    f.support.cellAdmissibleRegionIds=p1OnlyRegions;
    if(assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
           f.in,f.support,f.carriers,result,trace,error)){
        std::cerr<<"carrier transport accepted a source group outside the authenticated one-hop region set\n";return false;
    }
    return true;
}

int main() {
    Fixture f;
    double nearest=1.0;
    for(const auto& c:f.cc) nearest=std::min(nearest,dist(f.pc[0],c));
    if(!(nearest>f.in.maximumTransportDistanceM)) {std::cerr<<"fixture does not exceed strict centroid bound\n";return 1;}
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    std::string error;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"carrier transport failed: "<<error<<"\n";return 2;
    }
    if(result.childMassKg.size()!=4||result.parentMassTransferredKg!=std::vector<double>{1.0}||
       std::abs(result.parentMeasureTransferredM3[0]-1.0e-3)>1e-14||
       result.childParentRows.size()!=4||trace.carrierSourceParentRows!=std::vector<std::uint32_t>{0}||
       trace.expandedParentToSourceParentRows!=std::vector<std::uint32_t>({0,0,0,0})||
       trace.expandedParentToCarrierRows!=std::vector<std::uint32_t>({0,1,2,3})||
       trace.childParentCarrierRows.size()!=4||trace.childParentCarrierOffsets.size()!=5||
       trace.sourceCarrierPacketSHA256!=std::string(64,'1')||trace.sourceCarrierArraySHA256.empty()||
       trace.originalCentroidNearestAdmissibleDistanceM.size()!=1||
       !(trace.originalCentroidNearestAdmissibleDistanceM[0]>0.02)||
       result.groupClosure.size()!=1||std::abs(result.groupClosure[0].childMassKg-1.0)>1e-12) {
        std::cerr<<"carrier flow/aggregate mapping invariant failed\n";return 3;
    }
    for(std::size_t c=0;c<4;++c)
        if(std::abs(result.childMassKg[c]-0.25)>1e-10||
           std::abs(result.childImpliedDensityKgM3[c]-1000.0)>1e-8||
           trace.childParentCarrierOffsets[c+1]-trace.childParentCarrierOffsets[c]!=1u) {
            std::cerr<<"real child density or carrier edge failed at "<<c<<"\n";return 4;
        }
    if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"carrier deterministic replay failed: "<<error<<"\n";return 5;
    }
    if(!recordedRoleCSVImportFixture(f,result,trace))return 20;
    if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"solver-free carrier feasibility verification failed: "<<error<<"\n";return 16;
    }
    {
        auto imported=result;auto importedTrace=trace;
        imported.groupClosure.clear();imported.sourceDonorClosure.clear();
        imported.childSemanticGroupOffsets.clear();imported.childSemanticGroupIndices.clear();
        imported.childSemanticVolumeFractions.clear();
        if(!canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            f.in,f.support,f.carriers,imported,importedTrace,error)||
           imported.childParentOffsets!=result.childParentOffsets||
           imported.childParentRows!=result.childParentRows||
           imported.childParentTransportMassKg!=result.childParentTransportMassKg||
           imported.childParentTransportMeasureM3!=result.childParentTransportMeasureM3||
           importedTrace.childParentCarrierRows!=trace.childParentCarrierRows||
           importedTrace.childParentCarrierMassKg!=trace.childParentCarrierMassKg||
           importedTrace.receiptSHA256!=trace.receiptSHA256||
           !spatial_mass_transport_detail::validSHA256(imported.receiptSHA256)) {
            std::cerr<<"solver-free six-CSV metadata reconstruction failed: "<<error<<"\n";return 18;
        }
    }
    {
        auto badResult=result;badResult.groupClosure[0].childMaximumCapacityKg=0.1;
        if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            f.in,f.support,f.carriers,badResult,trace,error)) {
            std::cerr<<"solver-free verifier accepted a false semantic capacity\n";return 19;
        }
    }
    auto badTrace=trace; badTrace.childParentCarrierMassKg[0]+=0.01;
    if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        f.in,f.support,f.carriers,result,badTrace,error)) {
        std::cerr<<"solver-free feasibility accepted tampered carrier flow\n";return 17;
    }
    if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,badTrace,error)) {std::cerr<<"tampered carrier flow accepted\n";return 6;}
    {
        ConservativeSpatialMassTransportResult legacy;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(f.in,f.support,legacy,error)) {
            std::cerr<<"legacy centroid route unexpectedly accepted the out-of-bound parent\n"; return 7;
        }
    }
    {
        Fixture x; x.weights[0]=0.30;
        if(!rejects(x,"weight sum"))return 8;
    }
    {
        Fixture x; x.bary[0][0]=1.1;
        if(!rejects(x,"outside tetrahedron"))return 9;
    }
    {
        Fixture x; x.bary[0][0]+=0.01;x.bary[0][1]-=0.01;
        if(!rejects(x,"first moment"))return 10;
    }
    {
        Fixture x; x.witness[0]=99;
        if(!rejects(x,"witness child range"))return 11;
    }
    {
        Fixture x; x.witnessDists[0]=0.001;
        if(!rejects(x,"false distance"))return 12;
    }
    {
        Fixture x; x.in.maximumTransportDistanceM=0.03;
        ConservativeSpatialMassTransportResult expanded;
        SourceCellCarrierTransportTrace expandedTrace;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
            x.in,x.support,x.carriers,expanded,expandedTrace,error)||
           expandedTrace.originalCentroidNearestAdmissibleDistanceM.size()!=1u||
           !(expandedTrace.originalCentroidNearestAdmissibleDistanceM[0]<=0.03)){
            std::cerr<<"explicit source quadrature was rejected solely because the old centroid route was reachable: "<<error<<"\n";return 13;
        }
    }
    {
        Fixture x; x.parentTets[0][0][0]+=0.01;
        if(!rejects(x,"tet centroid mismatch"))return 14;
    }
    {
        Fixture x; x.carrierOff[1]=0;
        if(!rejects(x,"no fallback rows"))return 15;
    }
    if(!oneHopGroupAbsentFromP1PreferenceFixture())return 21;
    std::cout<<"PASS source carrier quadrature: base fixture plus two-parent one-hop source group absent from P1 preferences; solver-free feasibility accepts it and rejects an out-of-one-hop route\n";
    return 0;
}
