#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <iostream>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;

static ConservativeSpatialMassTransportInput makeInput(){
    static const std::array<SpatialMassTransportPoint,6> parents{{
        {1.00,0.0,0.0},{0.90,0.0,0.0},{0.90,0.0,0.0},
        {0.90,0.0,0.0},{0.90,0.0,0.0},{0.90,0.0,0.0}}};
    static const std::array<double,6> volumes{0.001,0.001,0.001,0.001,0.001,0.001};
    static const std::array<double,6> masses{1,1,1,1,1,1};
    static const std::array<std::uint32_t,6> materials{15,15,15,15,15,15};
    static const std::array<std::uint32_t,6> donors{1,2,2,2,2,2};
    static const std::array<std::uint32_t,6> regions{10,20,20,20,20,20};
    static const std::array<std::uint32_t,7> sourceOffsets{0,1,2,3,4,5,6};
    static const std::array<std::uint32_t,6> sourceDonors{1,2,2,2,2,2};
    static const std::array<double,6> sourceMass{1,1,1,1,1,1};
    static const std::array<SpatialMassTransportPoint,6> children{{
        {1.00,0.0,0.0},{0.91,0.0,0.0},{0.92,0.0,0.0},
        {0.93,0.0,0.0},{0.94,0.0,0.0},{0.95,0.0,0.0}}};
    static const std::array<double,6> childVolumes{0.001,0.001,0.001,0.001,0.001,0.001};
    static const std::array<std::uint32_t,6> childMaterials{15,15,15,15,15,15};
    static const std::array<std::uint32_t,6> childDonors{1,2,2,2,2,2};
    static const std::array<std::uint32_t,6> childRegions{10,20,20,20,20,20};
    static const std::array<double,16> lower=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
    static const std::array<double,16> upper=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
    static const std::string a(64,'a'),b(64,'b'),c(64,'c'),d(64,'d'),e(64,'e');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=parents;in.parentReferenceVolumeM3=volumes;in.parentMassKg=masses;
    in.parentMaterialIndices=materials;in.parentDonorBodyIndices=donors;in.parentAnatomicalRegionIndices=regions;
    in.parentSourceDonorOffsets=sourceOffsets;in.parentSourceDonorBodyIndices=sourceDonors;in.parentSourceDonorMassKg=sourceMass;
    in.childCentroidsWorld=children;in.childReferenceVolumeM3=childVolumes;
    in.childMaterialIndices=childMaterials;in.childDonorBodyIndices=childDonors;in.childAnatomicalRegionIndices=childRegions;
    in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
    in.parentMassPacketSHA256=a;in.parentSourceContributionSHA256=b;in.childMeshSHA256=c;
    in.regionLabelSHA256=d;in.coordinateFrameSHA256=e;
    in.maximumTransportDistanceM=0.11;in.maximumAllowedEdges=1000;in.maximumAugmentations=1000;
    return in;
}
static ConservativeSpatialChildSemanticSupport makeSupport(){
    static const std::array<std::uint32_t,7> p1Offsets{0,1,2,3,4,5,6};
    static const std::array<std::uint32_t,6> p1Groups{0,1,1,1,1,1};
    static const std::array<double,6> fractions{1,1,1,1,1,1};
    static const std::array<std::uint32_t,2> materials{15,15},donors{1,2},regions{10,20};
    static const std::array<std::uint64_t,2> ids{0x8000000000000010ull,0x8000000000000020ull};
    static const std::array<std::uint32_t,6> stableParents{501,601,601,601,601,601};
    static const std::array<std::uint32_t,3> pOffsets{0,1,6};
    static const std::array<std::uint32_t,6> pRows{0,1,2,3,4,5};
    static const std::array<std::uint32_t,3> sOffsets{0,1,2};
    static const std::array<std::uint32_t,2> sRows{501,601};
    static const std::array<std::uint32_t,7> allowOffsets{0,2,3,4,5,6,7};
    static const std::array<std::uint32_t,7> allowGroups{0,1,1,1,1,1,1};
    static const std::string typed(64,'f'),evidence(64,'1');
    ConservativeSpatialChildSemanticSupport s;
    s.childOffsets=p1Offsets;s.slotGroupIndices=p1Groups;s.volumeFractions=fractions;
    s.childAdmissibleGroupOffsets=allowOffsets;s.childAdmissibleGroupIndices=allowGroups;
    s.admissibleGroupEvidenceSHA256=evidence;s.groupMaterialIndices=materials;
    s.donorBodyIndices=donors;s.anatomicalRegionIndices=regions;s.derivedGroupIdentityIDs=ids;
    s.parentStableSourceFaceIdentities=stableParents;s.sourceParentTetOffsets=pOffsets;
    s.sourceParentTetRows=pRows;s.sourceStableFaceOffsets=sOffsets;s.sourceStableFaceRows=sRows;
    s.typedSupportSHA256=typed;return s;
}
static bool same(const ConservativeSpatialMassTransportResult&a,const ConservativeSpatialMassTransportResult&b){
    return a.receiptSHA256==b.receiptSHA256&&a.childParentOffsets==b.childParentOffsets&&
      a.childParentRows==b.childParentRows&&a.childParentTransportMassKg==b.childParentTransportMassKg&&
      a.childMassKg==b.childMassKg&&a.childImpliedDensityKgM3==b.childImpliedDensityKgM3&&
      a.parentMassTransferredKg==b.parentMassTransferredKg&&
      a.childSemanticGroupOffsets==b.childSemanticGroupOffsets&&
      a.childSemanticGroupIndices==b.childSemanticGroupIndices&&
      a.childSemanticVolumeFractions==b.childSemanticVolumeFractions;
}
int main(){
    const auto in=makeInput();const auto s=makeSupport();
    ConservativeSpatialMassTransportResult out,repeat;std::string error;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupport(in,s,out,error)){
      std::cerr<<"bounded/all-group seed solve failed: "<<error<<"\n";return 1;}
    if(!assembleConservativeSpatialMassTransportWithSemanticSupport(in,s,repeat,error)||!same(out,repeat)){
      std::cerr<<"bounded seed repeat is not deterministic: "<<error<<"\n";return 2;}
    if(out.childMassKg!=std::vector<double>(6,1.0)||out.childImpliedDensityKgM3!=std::vector<double>(6,1000.0)||
       out.parentMassTransferredKg!=std::vector<double>(6,1.0)||out.childParentOffsets.size()!=7u||
       out.transportEdgeCount!=6u){std::cerr<<"seed reduction changed feasible mass/density accounting\n";return 3;}
    for(std::uint32_t c=0;c<6u;++c){
      const auto want=c==0u?0u:c;
      if(out.childParentOffsets[c+1u]-out.childParentOffsets[c]!=1u||out.childParentRows[out.childParentOffsets[c]]!=want){
        std::cerr<<"unexpected deterministic parent assignment child="<<c<<"\n";return 4;}
    }
    std::cout<<"PASS deterministic bounded-seed repeat; exact 6 kg, 6 x 1000 kg/m3, parent totals and flow receipt="<<out.receiptSHA256<<"\n";
    return 0;
}
