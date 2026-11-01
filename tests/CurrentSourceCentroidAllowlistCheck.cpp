#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

static constexpr std::string_view kHashA="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static constexpr std::string_view kHashB="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

struct Fixture {
    std::array<SpatialMassTransportPoint,2> parentCentroids{{{0.25,0.25,0.25},{0.25,0.25,0.25}}};
    std::array<std::uint32_t,2> parentMaterial{15,15},parentDonor{145,146},parentRegion{10,10};
    std::array<double,2> parentMass{0.5,0.5},parentVolume{5.0e-4,5.0e-4};
    std::array<SpatialMassTransportPoint,1> childCentroids{{{0.25,0.25,0.25}}};
    std::array<std::uint32_t,1> childMaterial{15},childDonor{145},childRegion{10};
    std::array<double,1> childVolume{1.0e-3};
    ConservativeSpatialMassTransportInput input{};
    PhysicalSkinParentMassPacket parent{};
    ConservativeSpatialChildSemanticSupportData support{};

    Fixture(){
        input.parentCentroidsWorld=parentCentroids;
        input.parentMassKg=parentMass;
        input.parentReferenceVolumeM3=parentVolume;
        input.parentMaterialIndices=parentMaterial;
        input.parentDonorBodyIndices=parentDonor;
        input.parentAnatomicalRegionIndices=parentRegion;
        input.childCentroidsWorld=childCentroids;
        input.childReferenceVolumeM3=childVolume;
        input.childMaterialIndices=childMaterial;
        input.childDonorBodyIndices=childDonor;
        input.childAnatomicalRegionIndices=childRegion;
        input.maximumTransportDistanceM=0.019353;

        parent.referenceWorldPoints={{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}};
        parent.tetrahedra={{{0,1,2,3}},{{0,1,2,3}}};
        parent.parentMassPacketSHA256=std::string(64,'f');

        support.childOffsets={0,1};
        support.slotGroupIndices={0};
        support.volumeFractions={1.0};
        support.childAdmissibleGroupOffsets={0,1};
        support.childAdmissibleGroupIndices={0};
        support.admissibleGroupEvidenceSHA256=std::string(kHashA);
        support.groupMaterialIndices={15,15};
        support.donorBodyIndices={145,146};
        support.anatomicalRegionIndices={10,10};
        support.derivedGroupIdentityIDs={1,2};
        support.sourceParentTetOffsets={0,1,2};
        support.sourceParentTetRows={0,1};
        support.sourceStableFaceOffsets={0,1,2};
        support.sourceStableFaceRows={100,101};
        support.parentStableSourceFaceIdentities={100,101};
        support.cellAdmissibleRegionOffsets={0,1};
        support.cellAdmissibleRegionIds={10};
        support.admissibilitySourceSupportSHA256=std::string(kHashB);
        support.admissibilityMaximumSourceRegionHops=1;
        support.typedSupportSHA256=std::string(64,'c');
    }
};


static std::filesystem::path writeWitnessCSV(){
    const auto path=std::filesystem::temp_directory_path()/"h706-centroid-slot-fixture.csv";
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    out<<"source_parent_row,receiver_child_tet_row,source_material,source_donor,source_region,"
          "source_barycentric0,source_barycentric1,source_barycentric2,source_barycentric3,"
          "source_centroid_x_m,source_centroid_y_m,source_centroid_z_m,"
          "receiver_child_centroid_x_m,receiver_child_centroid_y_m,receiver_child_centroid_z_m,"
          "source_to_current_child_centroid_distance_m,one_hop_source_region,"
          "one_hop_max_source_region_hops,receiver_current_material,receiver_current_donor,"
          "receiver_current_region,capacity_reason_bits\n"
          "1,0,15,146,10,.25,.25,.25,.25,.25,.25,.25,.25,.25,.25,0,10,1,15,145,10,1\n";
    out.close();return path;
}
static bool rejects(Fixture fixture,CurrentSourceCentroidSlotWitnessRow row,
                    const std::string& witnessSHA,const std::string& what){
    std::string error;
    if(bindCurrentSourceCentroidSlotWitnesses(fixture.support,fixture.input,fixture.parent,
                                               std::span(&row,1),witnessSHA,error)){
        std::cerr<<"unexpectedly accepted "<<what<<"\n";return false;
    }
    if(error.empty()){std::cerr<<"missing rejection reason for "<<what<<"\n";return false;}
    return true;
}

int main(){
    auto witnessPath=writeWitnessCSV();
    std::string witnessSHA,error;
    if(!physical_skin_child_mass_detail::sha256File(witnessPath,witnessSHA,error)){
        std::cerr<<"cannot hash centroid witness control CSV: "<<error<<"\\n";return 1;
    }
    std::vector<CurrentSourceCentroidSlotWitnessRow> loadedWitnesses;
    if(!loadCurrentSourceCentroidSlotWitnessCSV(witnessPath,witnessSHA,loadedWitnesses,error)||
       loadedWitnesses.size()!=1u||loadedWitnesses.front().sourceParentRow!=1u||
       loadedWitnesses.front().receiverChildRow!=0u){
        std::cerr<<"valid centroid witness CSV failed its reader: "<<error<<"\\n";return 1;
    }
    std::vector<CurrentSourceCentroidSlotWitnessRow> badPinRows;
    if(loadCurrentSourceCentroidSlotWitnessCSV(witnessPath,kHashA,badPinRows,error)){
        std::cerr<<"centroid witness loader accepted a stale CSV SHA\\n";return 1;
    }
    const auto row=loadedWitnesses.front();
    Fixture valid;
    const auto oldChildOffsets=valid.support.childOffsets;
    const auto oldP1Groups=valid.support.slotGroupIndices;
    const auto oldP1Fractions=valid.support.volumeFractions;
    if(!bindCurrentSourceCentroidSlotWitnesses(valid.support,valid.input,valid.parent,
                                                std::span(&row,1),witnessSHA,error)){
        std::cerr<<"valid exact source-centroid slot rejected: "<<error<<"\n";return 1;
    }
    if(valid.support.childOffsets!=oldChildOffsets||
       valid.support.slotGroupIndices!=oldP1Groups||
       valid.support.volumeFractions!=oldP1Fractions||
       valid.support.childAdmissibleGroupOffsets!=std::vector<std::uint32_t>{0,2}||
       valid.support.childAdmissibleGroupIndices!=std::vector<std::uint32_t>{0,1}){
        std::cerr<<"proof-backed slot changed P1 scores or failed to add only the exact tuple allowlist\n";return 2;
    }
    const auto evidenceAfterValid=valid.support.admissibleGroupEvidenceSHA256;

    Fixture wrongMaterial;
    wrongMaterial.childMaterial[0]=16;
    if(!rejects(wrongMaterial,row,witnessSHA,"cross-material centroid slot"))return 3;
    Fixture wrongRegion;
    wrongRegion.parentRegion[1]=20;
    if(!rejects(wrongRegion,row,witnessSHA,"source region outside authenticated one-hop CSR"))return 4;
    Fixture tooFar;
    tooFar.childCentroids[0]={0.5,0.5,0.5};
    auto tooFarRow=row;
    tooFarRow.claimedChildCentroidWorld={0.5,0.5,0.5};
    tooFarRow.claimedDistanceM=std::sqrt(0.1875);
    if(!rejects(tooFar,tooFarRow,witnessSHA,"source centroid outside the unchanged distance bound"))return 5;
    Fixture wrongParentCSR;
    wrongParentCSR.support.sourceParentTetRows={0,0};
    if(!rejects(wrongParentCSR,row,witnessSHA,"parent not present in exact tuple source CSR"))return 6;
    Fixture noOneHop;
    noOneHop.support.admissibilityMaximumSourceRegionHops=2;
    if(!rejects(noOneHop,row,witnessSHA,"unapproved region-hop budget"))return 7;
    Fixture duplicate;
    const std::array<CurrentSourceCentroidSlotWitnessRow,2> duplicateRows{row,row};
    if(bindCurrentSourceCentroidSlotWitnesses(duplicate.support,duplicate.input,duplicate.parent,
                                               duplicateRows,witnessSHA,error)){
        std::cerr<<"duplicate source/receiver witness pair was accepted\n";return 8;
    }
    Fixture badParentCenter;
    badParentCenter.parentCentroids[1][0]+=2.0e-11;
    if(!rejects(badParentCenter,row,witnessSHA,"source centroid differs from transport parent centroid"))return 9;
    if(evidenceAfterValid==std::string(kHashA)){
        std::cerr<<"proof binding did not update the exact-group evidence digest\n";return 10;
    }
    std::filesystem::remove(witnessPath);
    std::cout<<"PASS current source-centroid proof slots: exact tuple/material/region/radius/CSR gates; P1 preferences unchanged\n";
    return 0;
}
