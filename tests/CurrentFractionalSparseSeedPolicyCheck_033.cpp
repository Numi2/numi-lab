#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

static ConservativeSpatialMassTransportInput makeInput(std::size_t edgeBudget=100u){
    static const std::array<SpatialMassTransportPoint,2> parents{{{0.0,0.0,0.0},{1.0,0.0,0.0}}};
    static const std::array<double,2> volumes{1.0e-3,1.0e-3};
    static const std::array<double,2> masses{1.0,1.0};
    static const std::array<std::uint32_t,2> materials{15u,15u};
    static const std::array<std::uint32_t,2> donors{1u,2u};
    static const std::array<std::uint32_t,2> regions{10u,20u};
    static const std::array<std::uint32_t,3> sourceOffsets{0u,1u,2u};
    static const std::array<std::uint32_t,2> sourceDonors{1u,2u};
    static const std::array<double,2> sourceMass{1.0,1.0};
    static const std::array<SpatialMassTransportPoint,2> children{{{1.0,0.0,0.0},{0.0,0.0,0.0}}};
    static const std::array<double,2> childVolumes{1.0e-3,1.0e-3};
    static const std::array<std::uint32_t,2> childMaterials{15u,15u};
    static const std::array<std::uint32_t,2> childDonors{1u,1u};
    static const std::array<std::uint32_t,2> childRegions{10u,10u};
    static const std::array<double,16> lower=[] {std::array<double,16>x{};x.fill(900.0);return x;}();
    static const std::array<double,16> upper=[] {std::array<double,16>x{};x.fill(1100.0);return x;}();
    static const std::string parentHash(64u,'a'),sourceHash(64u,'b'),childHash(64u,'c'),regionHash(64u,'d'),frameHash(64u,'e');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=parents;in.parentReferenceVolumeM3=volumes;in.parentMassKg=masses;
    in.parentMaterialIndices=materials;in.parentDonorBodyIndices=donors;in.parentAnatomicalRegionIndices=regions;
    in.parentSourceDonorOffsets=sourceOffsets;in.parentSourceDonorBodyIndices=sourceDonors;in.parentSourceDonorMassKg=sourceMass;
    in.childCentroidsWorld=children;in.childReferenceVolumeM3=childVolumes;
    in.childMaterialIndices=childMaterials;in.childDonorBodyIndices=childDonors;in.childAnatomicalRegionIndices=childRegions;
    in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
    in.parentMassPacketSHA256=parentHash;in.parentSourceContributionSHA256=sourceHash;
    in.childMeshSHA256=childHash;in.regionLabelSHA256=regionHash;in.coordinateFrameSHA256=frameHash;
    in.maximumTransportDistanceM=0.1;in.maximumAllowedEdges=edgeBudget;in.maximumAugmentations=100u;
    return in;
}

static ConservativeSpatialChildSemanticSupport makeSupport(){
    static const std::array<std::uint32_t,3> p1Offsets{0u,1u,2u};
    static const std::array<std::uint32_t,2> p1Groups{0u,0u};
    static const std::array<double,2> fractions{1.0,1.0};
    static const std::array<std::uint32_t,2> groupMaterials{15u,15u};
    static const std::array<std::uint32_t,2> groupDonors{1u,2u};
    static const std::array<std::uint32_t,2> groupRegions{10u,20u};
    static const std::array<std::uint64_t,2> identities{0x8000000000000010ull,0x8000000000000020ull};
    static const std::array<std::uint32_t,2> parentFaces{501u,601u};
    static const std::array<std::uint32_t,3> parentTetOffsets{0u,1u,2u};
    static const std::array<std::uint32_t,2> parentTets{0u,1u};
    static const std::array<std::uint32_t,3> stableFaceOffsets{0u,1u,2u};
    static const std::array<std::uint32_t,2> stableFaces{501u,601u};
    // Child 0 has P1 group 0, but its only within-radius route is non-P1 group 1.
    // Child 1 has an ordinary P1 route through group 0.
    static const std::array<std::uint32_t,3> allowOffsets{0u,2u,3u};
    static const std::array<std::uint32_t,3> allowGroups{0u,1u,0u};
    static const std::string typedHash(64u,'f'),evidenceHash(64u,'1');
    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=p1Offsets;support.slotGroupIndices=p1Groups;support.volumeFractions=fractions;
    support.childAdmissibleGroupOffsets=allowOffsets;support.childAdmissibleGroupIndices=allowGroups;
    support.admissibleGroupEvidenceSHA256=evidenceHash;
    support.groupMaterialIndices=groupMaterials;support.donorBodyIndices=groupDonors;
    support.anatomicalRegionIndices=groupRegions;support.derivedGroupIdentityIDs=identities;
    support.parentStableSourceFaceIdentities=parentFaces;support.sourceParentTetOffsets=parentTetOffsets;
    support.sourceParentTetRows=parentTets;support.sourceStableFaceOffsets=stableFaceOffsets;
    support.sourceStableFaceRows=stableFaces;support.typedSupportSHA256=typedHash;
    return support;
}

static bool solve(ConservativeSpatialMassTransportInput in,
                  ConservativeSpatialMassTransportResult& out,std::string& error){
    return assembleConservativeSpatialMassTransportWithSemanticSupport(in,makeSupport(),out,error);
}

int main(){
    auto in=makeInput();ConservativeSpatialMassTransportResult first,repeat;std::string error;
    if(!solve(in,first,error)){std::cerr<<"feasible P1/non-P1 control failed: "<<error<<"\n";return 1;}
    if(!solve(in,repeat,error)){std::cerr<<"deterministic repeat failed: "<<error<<"\n";return 2;}
    if(first.receiptSHA256.empty()||first.receiptSHA256!=repeat.receiptSHA256||
       first.childParentOffsets!=repeat.childParentOffsets||first.childParentRows!=repeat.childParentRows||
       first.childParentTransportMassKg!=repeat.childParentTransportMassKg||
       first.parentMassTransferredKg!=repeat.parentMassTransferredKg||
       first.childMassKg!=repeat.childMassKg||first.childImpliedDensityKgM3!=repeat.childImpliedDensityKgM3)
        {std::cerr<<"repeat run differs in flow/accounting/receipt\n";return 3;}
    if(first.childMassKg!=std::vector<double>{1.0,1.0}||
       first.childImpliedDensityKgM3!=std::vector<double>{1000.0,1000.0}||
       first.parentMassTransferredKg!=std::vector<double>{1.0,1.0}||
       first.childParentOffsets!=std::vector<std::uint32_t>{0u,1u,2u}||
       first.childParentRows!=std::vector<std::uint32_t>{1u,0u})
        {std::cerr<<"non-P1 route did not preserve exact feasible mass/density accounting\n";return 4;}
    bool group1AtChild0=false;
    for(std::uint32_t row=first.childSemanticGroupOffsets[0];row<first.childSemanticGroupOffsets[1];++row)
        group1AtChild0|=first.childSemanticGroupIndices[row]==1u;
    if(!group1AtChild0){std::cerr<<"child 0 did not use its authenticated non-P1 exact group\n";return 5;}
    ConservativeSpatialMassTransportResult capped;error.clear();
    if(solve(makeInput(1u),capped,error)||
       error.find("INCONCLUSIVE shared-capacity sparse seed graph exceeds pinned edge budget")==std::string::npos){
        std::cerr<<"actual seed insertion budget exhaustion did not fail closed: "<<error<<"\n";return 6;
    }
    std::cout<<"PASS all-allowed-group baseline-compatible feasibility/accounting; non-P1 fallback; repeat_receipt="
             <<first.receiptSHA256<<" edges="<<first.transportEdgeCount<<" cap_reject=PASS\n";
    return 0;
}
