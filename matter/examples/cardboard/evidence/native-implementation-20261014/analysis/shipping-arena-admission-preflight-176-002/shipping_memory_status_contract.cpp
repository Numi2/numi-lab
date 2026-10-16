#include "cardboard_shipping_memory.hpp"
#include "numi/matter/detail/full_contact_armijo_scratch_budget.hpp"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace numi::cardboard;
using namespace numi::matter::detail;
static void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
static ShippingMemoryBudgetInput input(bool aliasSurface) {
    ShippingMemoryBudgetInput x;
    x.nodeCount=528444u; x.tetrahedronCount=1477728u;
    x.cohesiveFaceCount=260835u; x.contactPairCount=6335568u;
    x.activeContactCapacity=x.contactPairCount;
    x.deformableContactCapacity=131072u; x.surfaceFaceCount=1054876u;
    x.compactImmutableSurfaceMap=true; x.rigidBodyCount=9u;
    x.rigidProxyCount=12u; x.materialStateStride=15u;
    x.compactPreconditionerRegionCount=3u; x.environmentCount=1u;
    x.fgmresRestart=16u; x.fullContactArmijoRequired=true;
    x.deformableSelfContactEnabled=true;
    x.fullContactArmijoAliasesSurfaceScratch=aliasSurface;
    x.fullContactArmijoTiledCohesive=true;
    x.cohesiveTrialPageCapacity=65536u;
    x.sparsePrivateContactRequested=true;
    x.sparsePrivateContactCapacity=65536u;
    x.hostSourceMeshBytes=shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(x.nodeCount,256u),
        shippingMemoryCheckedMultiply(x.tetrahedronCount,512u));
    x.hostCompiledWorldBytes=shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(x.nodeCount,128u),
        shippingMemoryCheckedMultiply(x.tetrahedronCount,256u));
    x.hostGeneratedMetalBytes=64ull*1024ull*1024ull;
    x.uploadStagingBytes=x.hostCompiledWorldBytes;
    x.retainedCompleteCheckpointCount=2u;
    x.estimatedSnapshotPayloadBytes=estimateShippingSnapshotPayloadBytes(x);
    return x;
}
int main() {
    try {
        static_assert(sizeof(NMMatterStatusGPU)==48u);
        constexpr std::uint64_t E=1u,N=528444u,C=131072u,P=1054876u;
        const auto L=std::bit_ceil(P);
        FullContactArmijoScratchBytes alias{}, noAlias{};
        require(estimateFullContactArmijoScratchBytes(E,P,L,C,N,true,alias),
                "alias helper estimate");
        require(estimateFullContactArmijoScratchBytes(E,P,L,C,N,false,noAlias),
                "unaliased helper estimate");
        const auto budgetAlias=estimateShippingMemoryBudget(input(true));
        const auto budgetNoAlias=estimateShippingMemoryBudget(input(false));
        require(budgetAlias.requiredFullContactArmijoBytes==alias.total,
                "shipping estimator must use shared exact helper with surface alias");
        require(budgetNoAlias.requiredFullContactArmijoBytes==noAlias.total,
                "shipping estimator must use shared exact helper without alias");
        require(alias.total==77661184u && alias.total<=kFullContactArmijoScratchBudgetBytes,
                "E1 aliased scratch boundary");
        require(noAlias.total==296269008u && noAlias.total>kFullContactArmijoScratchBudgetBytes,
                "E1 unaliased scratch boundary");
        const std::uint64_t oldEstimate=
            C*(sizeof(NMDeformableContactCandidateGPU)+sizeof(NMDeformableContactGPU)+
               2u*sizeof(std::uint32_t)+6u*sizeof(std::uint32_t))+
            N*sizeof(NMIncidenceRangeGPU)+4u*sizeof(std::uint32_t)+sizeof(NMIndirectDispatchGPU)+
            P*(sizeof(NMContinuumSurfacePrimitiveGPU)+4u*sizeof(std::uint32_t))+
            4u*L*sizeof(nm_float4);
        require(noAlias.total-oldEstimate==48u*C+48u,
                "exact correction is candidate status plus typed sentinel deltas");
        FullContactArmijoScratchBytes e2fit{},e2over{},overflow{};
        require(estimateFullContactArmijoScratchBytes(2u,0u,0u,127100u,0u,true,e2fit) &&
                e2fit.total<=kFullContactArmijoScratchBudgetBytes,"E2 fitting boundary");
        require(estimateFullContactArmijoScratchBytes(2u,0u,0u,127101u,0u,true,e2over) &&
                e2over.total>kFullContactArmijoScratchBudgetBytes,"E2 first over boundary");
        require(!estimateFullContactArmijoScratchBytes(
                    2u,std::numeric_limits<std::uint64_t>::max(),1u,1u,1u,false,overflow),
                "environment product overflow fails closed");
        std::cout<<"shipping_memory_status_contract_checks=9"
                 <<" E1_alias="<<alias.total
                 <<" E1_unalias="<<noAlias.total
                 <<" corrected_delta="<<noAlias.total-oldEstimate
                 <<" E2_fit="<<e2fit.total
                 <<" E2_first_over="<<e2over.total
                 <<" no_runtime_or_physical_qualification\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"shipping memory contract failed: "<<e.what()<<"\n";
        return 1;
    }
}
