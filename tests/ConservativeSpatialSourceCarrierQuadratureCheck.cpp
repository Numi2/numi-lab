#include "ConservativeSpatialSourceCarrierQuadrature.hpp"
#include <array>
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
    auto badTrace=trace; badTrace.childParentCarrierMassKg[0]+=0.01;
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

    // Group IDs are registry indices, not region IDs. Two different donors can
    // share one authenticated region; same-region material alone is insufficient
    // for a carrier witness, while an exact allowlist admits the exact group.
    {
        Fixture x;
        x.pm={1.0,1.0};x.pv={1.0e-3,1.0e-3};x.pmat={15,15};
        x.pdonor={146,145};x.pregion={145,145};
        x.poff={0,1,2};x.pdids={146,145};x.pdmass={1.0,1.0};
        x.stable={77608,77609};
        x.gm={15,15};x.gd={146,145};x.gr={145,145};
        x.gids={0x8000000000000100ull,0x8000000000000101ull};
        x.gpoff={0,1,2};x.gprows={0,1};x.gfoff={0,1,2};x.gfrows={77608,77609};
        x.childOff={0,1,2,3,4};x.slotGroup={1,1,1,1};x.preference={1,1,1,1};
        x.carrierOff={0,4,8};x.tetRows={9908,9909};
        x.sourceTetNodeRows={{0,1,2,3},{0,1,2,3}};x.parentTets={x.tet,x.tet};
        x.weights={.25,.25,.25,.25,.25,.25,.25,.25};
        x.bary={x.bary[0],x.bary[1],x.bary[2],x.bary[3],
                x.bary[0],x.bary[1],x.bary[2],x.bary[3]};
        x.witness={0,1,2,3,0,1,2,3};
        x.witnessDists.resize(8);
        for(std::size_t i=0;i<8;++i)
            x.witnessDists[i]=dist(x.cc[i%4],point(x.tet,x.bary[i]));
        std::vector<P> parentCentroids{x.pc[0],x.pc[0]};
        std::vector<double> childVolumes{.0005,.0005,.0005,.0005};
        std::vector<std::uint32_t> childDonors{145,145,145,145},childRegions{145,145,145,145};
        std::string parentMassHash(64,'a'),sourceContributionHash(64,'b');
        std::string sourceCarrierHash(64,'1'),sourceTetHash(64,'2'),sourceNodeHash(64,'3');
        x.in.parentCentroidsWorld=parentCentroids;
        x.in.parentReferenceVolumeM3=x.pv;x.in.parentMassKg=x.pm;
        x.in.parentMaterialIndices=x.pmat;x.in.parentDonorBodyIndices=x.pdonor;
        x.in.parentAnatomicalRegionIndices=x.pregion;x.in.parentSourceDonorOffsets=x.poff;
        x.in.parentSourceDonorBodyIndices=x.pdids;x.in.parentSourceDonorMassKg=x.pdmass;
        x.in.parentMassPacketSHA256=parentMassHash;
        x.in.parentSourceContributionSHA256=sourceContributionHash;
        x.in.childReferenceVolumeM3=childVolumes;
        x.in.childDonorBodyIndices=childDonors;
        x.in.childAnatomicalRegionIndices=childRegions;
        x.support.childOffsets=x.childOff;x.support.slotGroupIndices=x.slotGroup;
        x.support.volumeFractions=x.preference;x.support.groupMaterialIndices=x.gm;
        x.support.donorBodyIndices=x.gd;x.support.anatomicalRegionIndices=x.gr;
        x.support.derivedGroupIdentityIDs=x.gids;x.support.parentStableSourceFaceIdentities=x.stable;
        x.support.sourceParentTetOffsets=x.gpoff;x.support.sourceParentTetRows=x.gprows;
        x.support.sourceStableFaceOffsets=x.gfoff;x.support.sourceStableFaceRows=x.gfrows;
        x.support.typedSupportSHA256=x.supportHash;
        static const std::array<std::uint32_t,5> allowOffsets{0,1,2,3,4};
        static const std::array<std::uint32_t,4> onlyPrimary{1,1,1,1};
        static const std::array<std::uint32_t,5> regionOffsets{0,1,2,3,4};
        static const std::array<std::uint32_t,4> sameRegion{145,145,145,145};
        static const std::string projectionHash(64,'8'), adjacencyHash(64,'9');
        x.support.childAdmissibleGroupOffsets=allowOffsets;
        x.support.childAdmissibleGroupIndices=onlyPrimary;
        x.support.admissibleGroupEvidenceSHA256=projectionHash;
        x.support.cellAdmissibleRegionOffsets=regionOffsets;
        x.support.cellAdmissibleRegionIds=sameRegion;
        x.support.admissibilitySourceSupportSHA256=adjacencyHash;
        x.support.admissibilityMaximumSourceRegionHops=1;
        x.carriers.parentOffsets=x.carrierOff;x.carriers.globalSourceTetRows=x.tetRows;
        x.carriers.sourceTetNodeRows=x.sourceTetNodeRows;
        x.carriers.parentReferenceTetWorldVertices=x.parentTets;
        x.carriers.carrierWeights=x.weights;x.carriers.carrierBarycentric=x.bary;
        x.carriers.witnessChildRows=x.witness;
        x.carriers.witnessChildCentroidDistancesM=x.witnessDists;
        x.carriers.sourceCarrierPacketSHA256=sourceCarrierHash;
        x.carriers.sourceTetTableSHA256=sourceTetHash;
        x.carriers.sourceNodeTableSHA256=sourceNodeHash;
        std::vector<std::uint32_t> parentGroup;
        std::vector<std::map<std::uint32_t,double>> preference;
        std::vector<std::vector<std::uint32_t>> allowed;
        std::vector<P> carrierPoints;
        std::vector<std::uint32_t> carrierRows;
        std::vector<double> nearestRows;
        if(source_carrier_transport_detail::validateCarrierInput(
                x.in,x.support,x.carriers,parentGroup,preference,allowed,
                carrierPoints,carrierRows,nearestRows,error)||
           error.find("exact source tuple")==std::string::npos){
            std::cerr<<"same material+region but unallowed donor carrier group was accepted: "<<error<<"\\n";return 18;
        }
        static const std::array<std::uint32_t,5> bothOffsets{0,2,4,6,8};
        static const std::array<std::uint32_t,8> bothGroups{0,1,0,1,0,1,0,1};
        x.support.childAdmissibleGroupOffsets=bothOffsets;
        x.support.childAdmissibleGroupIndices=bothGroups;
        if(!source_carrier_transport_detail::validateCarrierInput(
                x.in,x.support,x.carriers,parentGroup,preference,allowed,
                carrierPoints,carrierRows,nearestRows,error)||
           parentGroup!=std::vector<std::uint32_t>({0,1})||
           allowed.size()!=4u||allowed[0]!=std::vector<std::uint32_t>({0,1})){
            std::cerr<<"exact same-region donor allowlist failed carrier validation: "<<error<<"\n";return 19;
        }
        ConservativeSpatialMassTransportResult exactTupleResult;
        SourceCellCarrierTransportTrace exactTupleTrace;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
                x.in,x.support,x.carriers,exactTupleResult,exactTupleTrace,error)||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
                x.in,x.support,x.carriers,exactTupleResult,exactTupleTrace,error)){
            std::cerr<<"exact same-region donor carrier allocation failed replay: "<<error<<"\n";return 20;
        }
    }

    std::cout<<"PASS source carrier quadrature: 1 parent, 4 carriers, 4 real cells; aggregate mass/density/measure and replay verified; legacy route unchanged; 7 tamper/eligibility controls rejected; explicit quadrature may also split a centroid-reachable source cell\n";
    return 0;
}
