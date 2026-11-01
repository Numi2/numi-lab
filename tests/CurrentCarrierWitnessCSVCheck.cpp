#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace fs=std::filesystem;

static const std::string kHeader=
    "record_kind,role,carrier_construction,packet_carrier_row,local_parent_row,"
    "global_source330_tet_row,child_local_row,child_global_row,material,donor,region,"
    "provenance_mode,p1_group_id,projection_csv_row,bary0,bary1,bary2,bary3,weight,"
    "source_x_m,source_y_m,source_z_m,child_centroid_x_m,child_centroid_y_m,"
    "child_centroid_z_m,distance_m";

static fs::path writeRoleCSV(const std::string& role,const std::uint32_t material,
                             const std::uint32_t donor,const std::uint32_t region,
                             const std::uint32_t globalParent,const std::uint32_t child,
                             bool withFallback=false){
    const auto path=fs::temp_directory_path()/(role+"-h706-carrier-witness.csv");
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    out<<kHeader<<"\n";
    out<<"carrier,"<<role<<",inherited_090_quadrature,0,0,"<<globalParent<<",0,"
       <<child<<","<<material<<","<<donor<<","<<region
       <<",p1_exact_tuple_slot,0,-1,0.25,0.25,0.25,0.25,0.05,"
       <<"0.25,0.25,0.25,0.25,0.25,0.25,0\n";
    if(withFallback)
        out<<"centroid_route_fallback,"<<role<<",unchanged_source_centroid,-1,0,"
           <<globalParent<<",0,"<<child<<","<<material<<","<<donor<<","<<region
           <<",p1_exact_tuple_slot,0,-1,0.25,0.25,0.25,0.25,1.0,"
           <<"0.25,0.25,0.25,0.25,0.25,0.25,0\n";
    return path;
}

int main(){
    std::string error,skinSHA,fatSHA;
    auto skinPath=writeRoleCSV("Skin",15u,43u,43u,0u,0u,true);
    auto fatPath=writeRoleCSV("Fat",16u,42u,42u,1u,1u,false);
    if(!physical_skin_child_mass_detail::sha256File(skinPath,skinSHA,error)||
       !physical_skin_child_mass_detail::sha256File(fatPath,fatSHA,error)){
        std::cerr<<"fixture hash failed: "<<error<<"\n";return 1;
    }
    std::vector<CurrentCarrierWitnessEvidenceRow> skinRows,fatRows;
    CurrentCarrierWitnessCSVSummary skinSummary,fatSummary;
    if(!loadCurrentCarrierWitnessCSV(skinPath,skinSHA,15u,skinRows,skinSummary,error)||
       !loadCurrentCarrierWitnessCSV(fatPath,fatSHA,16u,fatRows,fatSummary,error)||
       skinRows.size()!=1u||fatRows.size()!=1u||
       skinSummary.carrierRecordCount!=1u||skinSummary.centroidFallbackRecordCount!=1u){
        std::cerr<<"valid sidecars failed parsing: "<<error<<"\n";return 2;
    }
    std::vector<CurrentCarrierWitnessEvidenceRow> allRows=skinRows;
    allRows.insert(allRows.end(),fatRows.begin(),fatRows.end());

    std::array<SpatialMassTransportPoint,2> parentCentroids{{{0.25,0.25,0.25},{0.25,0.25,0.25}}};
    std::array<double,2> parentVolumes{1.0/6.0,1.0/6.0},parentMass{0.5,0.5};
    std::array<std::uint32_t,2> parentMaterial{15u,16u},parentDonor{43u,42u},parentRegion{43u,42u};
    std::array<SpatialMassTransportPoint,2> childCentroids{{{0.25,0.25,0.25},{0.25,0.25,0.25}}};
    std::array<double,2> childVolumes{1.0e-3,1.0e-3};
    std::array<std::uint32_t,2> childMaterial{15u,16u},childDonor{43u,42u},childRegion{43u,42u};
    ConservativeSpatialMassTransportInput input{};
    input.parentCentroidsWorld=parentCentroids;input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMass;input.parentMaterialIndices=parentMaterial;
    input.parentDonorBodyIndices=parentDonor;input.parentAnatomicalRegionIndices=parentRegion;
    input.childCentroidsWorld=childCentroids;input.childReferenceVolumeM3=childVolumes;
    input.childMaterialIndices=childMaterial;input.childDonorBodyIndices=childDonor;
    input.childAnatomicalRegionIndices=childRegion;input.maximumTransportDistanceM=0.019353;

    ConservativeSpatialChildSemanticSupportData support;
    support.childOffsets={0u,1u,2u};support.slotGroupIndices={0u,1u};support.volumeFractions={1.0,1.0};
    support.childAdmissibleGroupOffsets=support.childOffsets;
    support.childAdmissibleGroupIndices=support.slotGroupIndices;
    support.groupMaterialIndices={15u,16u};support.donorBodyIndices={43u,42u};
    support.anatomicalRegionIndices={43u,42u};support.derivedGroupIdentityIDs={1u,2u};
    support.sourceParentTetOffsets={0u,1u,2u};support.sourceParentTetRows={0u,1u};
    support.sourceStableFaceOffsets={0u,1u,2u};support.sourceStableFaceRows={100u,101u};
    support.parentStableSourceFaceIdentities={100u,101u};
    support.cellAdmissibleRegionOffsets={0u,1u,2u};support.cellAdmissibleRegionIds={43u,42u};
    support.admissibilitySourceSupportSHA256=std::string(64u,'b');
    support.admissibilityMaximumSourceRegionHops=1u;support.typedSupportSHA256=std::string(64u,'c');

    CurrentSourceCarrierQuadratureData carrier;
    carrier.parentOffsets={0u,1u,2u};carrier.globalSourceTetRows={0u,1u};
    carrier.sourceTetNodeRows={{{0u,1u,2u,3u}},{{4u,1u,2u,3u}}};
    const std::array<std::array<double,3>,4> tet{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
    carrier.parentReferenceTetWorldVertices={tet,tet};
    carrier.carrierWeights={0.05,0.05};carrier.carrierBarycentric={{0.25,0.25,0.25,0.25},{0.25,0.25,0.25,0.25}};
    carrier.witnessChildRows={0u,1u};carrier.witnessChildCentroidDistancesM={0.0,0.0};
    carrier.carrierRoleLocalRows={0u,0u};carrier.carrierRoleMaterialIndices={15u,16u};
    carrier.carrierRoleLocalChildRows={0u,0u};carrier.carrierRoleLocalParentRows={0u,0u};
    carrier.sourceCarrierPacketSHA256=std::string(64u,'d');

    if(!bindCurrentCarrierWitnessAllowlist(support,input,carrier,allRows,{},
          std::string(64u,'e'),std::string(64u,'f'),error)){
        std::cerr<<"valid exact-role sidecars did not bind: "<<error<<"\n";return 3;
    }
    if(support.childAdmissibleGroupOffsets!=std::vector<std::uint32_t>{0u,1u,2u}||
       support.childAdmissibleGroupIndices!=std::vector<std::uint32_t>{0u,1u}||
       support.volumeFractions!=std::vector<double>{1.0,1.0}){
        std::cerr<<"sidecar binding modified primary P1 preferences or exact role slots\n";return 4;
    }

    std::vector<CurrentCarrierWitnessEvidenceRow> wrongSHA;
    if(loadCurrentCarrierWitnessCSV(skinPath,std::string(64u,'0'),15u,wrongSHA,skinSummary,error)){
        std::cerr<<"carrier sidecar with wrong SHA was accepted\n";return 5;
    }
    auto inactiveRole=skinRows;
    if(loadCurrentCarrierWitnessCSV(skinPath,skinSHA,16u,inactiveRole,skinSummary,error)){
        std::cerr<<"Skin sidecar accepted under Fat role\n";return 6;
    }
    auto duplicatePath=fs::temp_directory_path()/"duplicate-h706-carrier-witness.csv";
    {
        std::ifstream in(skinPath,std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
        auto first=all.find('\n');auto rowStart=first+1u;auto rowEnd=all.find('\n',rowStart);
        std::string row=all.substr(rowStart,rowEnd-rowStart);
        std::ofstream out(duplicatePath,std::ios::binary|std::ios::trunc);
        out<<all<<row<<"\n";
    }
    std::string duplicateSHA;
    if(!physical_skin_child_mass_detail::sha256File(duplicatePath,duplicateSHA,error)||
       loadCurrentCarrierWitnessCSV(duplicatePath,duplicateSHA,15u,wrongSHA,skinSummary,error)){
        std::cerr<<"duplicate role-local carrier row was accepted\n";return 7;
    }
    fs::remove(skinPath);fs::remove(fatPath);fs::remove(duplicatePath);
    std::cout<<"PASS current 1046c carrier sidecars: pinned full schema, role-local rows, barycentric geometry, radius, exact tuple and P1 preservation\n";
    return 0;
}
