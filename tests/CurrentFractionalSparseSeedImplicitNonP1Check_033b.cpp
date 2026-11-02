#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <iostream>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;

static ConservativeSpatialMassTransportInput makeInput(){
    static const std::array<SpatialMassTransportPoint,3> parents{{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.09,0.0,0.0}}};
    static const std::array<double,3> parentVolumes{0.004,0.001,0.001};
    static const std::array<double,3> parentMasses{4.0,1.0,1.0};
    static const std::array<std::uint32_t,3> parentMaterials{15,15,15};
    static const std::array<std::uint32_t,3> parentDonors{1,2,3};
    static const std::array<std::uint32_t,3> parentRegions{10,20,30};
    static const std::array<std::uint32_t,4> sourceOffsets{0,1,2,3};
    static const std::array<std::uint32_t,3> sourceDonors{1,2,3};
    static const std::array<double,3> sourceMass{4.0,1.0,1.0};
    static const std::array<SpatialMassTransportPoint,6> children{{
        {0.00,0.0,0.0},{0.01,0.0,0.0},{0.02,0.0,0.0},
        {0.03,0.0,0.0},{0.09,0.0,0.0},{0.10,0.0,0.0}}};
    static const std::array<double,6> volumes{.001,.001,.001,.001,.001,.001};
    static const std::array<std::uint32_t,6> childMaterials{15,15,15,15,15,15};
    static const std::array<std::uint32_t,6> childDonors{1,1,1,1,3,3};
    static const std::array<std::uint32_t,6> childRegions{10,10,10,10,30,30};
    static const std::array<double,16> lower=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
    static const std::array<double,16> upper=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
    static const std::string a(64,'a'),b(64,'b'),c(64,'c'),d(64,'d'),e(64,'e');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=parents;in.parentReferenceVolumeM3=parentVolumes;in.parentMassKg=parentMasses;
    in.parentMaterialIndices=parentMaterials;in.parentDonorBodyIndices=parentDonors;in.parentAnatomicalRegionIndices=parentRegions;
    in.parentSourceDonorOffsets=sourceOffsets;in.parentSourceDonorBodyIndices=sourceDonors;in.parentSourceDonorMassKg=sourceMass;
    in.childCentroidsWorld=children;in.childReferenceVolumeM3=volumes;in.childMaterialIndices=childMaterials;
    in.childDonorBodyIndices=childDonors;in.childAnatomicalRegionIndices=childRegions;
    in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
    in.parentMassPacketSHA256=a;in.parentSourceContributionSHA256=b;in.childMeshSHA256=c;
    in.regionLabelSHA256=d;in.coordinateFrameSHA256=e;in.maximumTransportDistanceM=.11;
    in.maximumAllowedEdges=100;in.maximumAugmentations=1000;return in;
}
static ConservativeSpatialChildSemanticSupport makeSupport(){
    static const std::array<std::uint32_t,7> p1Offsets{0,1,2,3,4,5,6};
    static const std::array<std::uint32_t,6> p1Groups{0,0,0,0,2,2};
    static const std::array<double,6> p1Fractions{1,1,1,1,1,1};
    static const std::array<std::uint32_t,3> groupMaterials{15,15,15};
    static const std::array<std::uint32_t,3> groupDonors{1,2,3};
    static const std::array<std::uint32_t,3> groupRegions{10,20,30};
    static const std::array<std::uint64_t,3> groupIDs{0x8000000000000010ull,0x8000000000000020ull,0x8000000000000030ull};
    static const std::array<std::uint32_t,3> parentFaces{501,601,701};
    static const std::array<std::uint32_t,4> parentTetOffsets{0,1,2,3};
    static const std::array<std::uint32_t,3> parentTets{0,1,2};
    static const std::array<std::uint32_t,4> stableFaceOffsets{0,1,2,3};
    static const std::array<std::uint32_t,3> stableFaces{501,601,701};
    static const std::array<std::uint32_t,7> allowOffsets{0,2,4,6,8,10,12};
    static const std::array<std::uint32_t,12> allowGroups{0,1,0,1,0,1,0,1,1,2,1,2};
    static const std::string typed(64,'f'),evidence(64,'1');
    ConservativeSpatialChildSemanticSupport s;
    s.childOffsets=p1Offsets;s.slotGroupIndices=p1Groups;s.volumeFractions=p1Fractions;
    s.childAdmissibleGroupOffsets=allowOffsets;s.childAdmissibleGroupIndices=allowGroups;
    s.admissibleGroupEvidenceSHA256=evidence;s.groupMaterialIndices=groupMaterials;
    s.donorBodyIndices=groupDonors;s.anatomicalRegionIndices=groupRegions;s.derivedGroupIdentityIDs=groupIDs;
    s.parentStableSourceFaceIdentities=parentFaces;s.sourceParentTetOffsets=parentTetOffsets;
    s.sourceParentTetRows=parentTets;s.sourceStableFaceOffsets=stableFaceOffsets;
    s.sourceStableFaceRows=stableFaces;s.typedSupportSHA256=typed;return s;
}
static bool run(ConservativeSpatialMassTransportResult& out,std::string& error){
    return assembleConservativeSpatialMassTransportWithSemanticSupport(makeInput(),makeSupport(),out,error);
}
static bool sameResult(const ConservativeSpatialMassTransportResult&a,const ConservativeSpatialMassTransportResult&b){
    return a.receiptSHA256==b.receiptSHA256&&a.childParentOffsets==b.childParentOffsets&&
      a.childParentRows==b.childParentRows&&a.childParentTransportMassKg==b.childParentTransportMassKg&&
      a.childMassKg==b.childMassKg&&a.childImpliedDensityKgM3==b.childImpliedDensityKgM3&&
      a.parentMassTransferredKg==b.parentMassTransferredKg;
}
int main(){
    ConservativeSpatialMassTransportResult first,repeat;std::string error;
    if(!run(first,error)){std::cerr<<"six-child bounded sparse solve failed: "<<error<<"\n";return 1;}
    if(!run(repeat,error)||!sameResult(first,repeat)){std::cerr<<"bounded implicit-route repeat differs: "<<error<<"\n";return 2;}
    if(first.childMassKg!=std::vector<double>(6,1.0)||
       first.childImpliedDensityKgM3!=std::vector<double>(6,1000.0)||
       first.parentMassTransferredKg!=std::vector<double>{4.0,1.0,1.0}){
        std::cerr<<"bounded implicit route changed exact mass/density/parent accounting\n";return 3;}
    bool bToFar=false;
    for(std::uint32_t c=4;c<6;++c)
      for(std::uint32_t row=first.childParentOffsets[c];row<first.childParentOffsets[c+1u];++row)
        if(first.childParentRows[row]==1u&&first.childParentTransportMassKg[row]>0.0)bToFar=true;
    if(!bToFar){std::cerr<<"required non-P1 parent B to far-child flow is absent\n";return 4;}
#ifdef SPARSE_SEED_BASELINE_CONTROL
    std::cout<<"PASS dense baseline supports required non-P1 B-to-far flow; 6 kg, 6 x 1000 kg/m3, parent totals {4,1,1}; deterministic receipt="<<first.receiptSHA256<<"\n";
#else
    std::cout<<"PASS bounded seed requires/materializes non-P1 implicit B-to-far flow; 6 kg, 6 x 1000 kg/m3, parent totals {4,1,1}; deterministic receipt="<<first.receiptSHA256<<"\n";
#endif
    return 0;
}
