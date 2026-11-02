#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
using namespace metalrobo::human::coarse_tissue;
int main(){
    const std::array<SpatialMassTransportPoint,1> pp{{{0,0,0}}};
    const std::array<SpatialMassTransportPoint,3> cp{{{0,0,0},{.01,0,0},{.02,0,0}}};
    const std::array<double,1> pv{1},pm{1},dm{1};
    const std::array<double,3> cv{.1,.1,.1},vf{1,1,1};
    const std::array<std::uint32_t,1> mat{15},don{1},reg{10},pr{0},face{501};
    const std::array<std::uint32_t,2> po{0,1};
    const std::array<std::uint32_t,3> cm{15,15,15},cd{1,1,1},cr{10,10,10},groups{0,0,0};
    const std::array<std::uint32_t,4> co{0,1,2,3};
    const std::array<std::uint64_t,1> identity{0x8000000000000010ull};
    std::array<double,16> lo{},hi{};lo.fill(1);hi.fill(4);
    const std::string pin(64,'e');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=pp;in.parentReferenceVolumeM3=pv;in.parentMassKg=pm;
    in.parentMaterialIndices=mat;in.parentDonorBodyIndices=don;in.parentAnatomicalRegionIndices=reg;
    in.parentSourceDonorOffsets=po;in.parentSourceDonorBodyIndices=don;in.parentSourceDonorMassKg=dm;
    in.childCentroidsWorld=cp;in.childReferenceVolumeM3=cv;in.childMaterialIndices=cm;
    in.childDonorBodyIndices=cd;in.childAnatomicalRegionIndices=cr;
    in.materialDensityLowerKgM3=lo;in.materialDensityUpperKgM3=hi;
    in.parentMassPacketSHA256=pin;in.parentSourceContributionSHA256=pin;in.childMeshSHA256=pin;
    in.regionLabelSHA256=pin;in.coordinateFrameSHA256=pin;
    in.maximumTransportDistanceM=.4;in.maximumAllowedEdges=20;in.maximumAugmentations=100;
    ConservativeSpatialChildSemanticSupport s;
    s.childOffsets=co;s.slotGroupIndices=groups;s.volumeFractions=vf;
    s.groupMaterialIndices=mat;s.donorBodyIndices=don;s.anatomicalRegionIndices=reg;
    s.derivedGroupIdentityIDs=identity;s.parentStableSourceFaceIdentities=face;
    s.sourceParentTetOffsets=po;s.sourceParentTetRows=pr;s.sourceStableFaceOffsets=po;
    s.sourceStableFaceRows=face;s.typedSupportSHA256=pin;
    ConservativeSpatialMassTransportResult result;std::string error;
    std::ostringstream diag;auto* saved=std::cerr.rdbuf(diag.rdbuf());
    const bool okay=assembleConservativeSpatialMassTransportWithSemanticSupport(in,s,result,error);
    std::cerr.rdbuf(saved);
    if(!okay){std::cerr<<error.substr(0,1000)<<'\n';return 1;}
    if(std::abs(std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0)-1)>1e-15||
       result.maximumParentMassClosureErrorKg>1e-15)return 2;
    for(auto d:result.childImpliedDensityKgM3)if(d<1||d>4)return 3;
    // The exact same source must still fail a genuinely insufficient receiver envelope.
    hi.fill(2);
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(in,s,result,error))return 4;
    std::cout<<"nonrepresentable residual capacity: full source mass and every density bound PASS; insufficient capacity rejected\n";
}
