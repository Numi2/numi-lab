#include "cardboard_shipping_memory.hpp"
#include <bit>
#include <cstdint>
#include <iostream>
using namespace numi::cardboard;
int main() {
 constexpr std::uint64_t N=528444, T=1477728, F=1054876, CF=260835;
 constexpr std::uint64_t E=1, proxies=12, cap=131072, pair=N*proxies, rigidCap=65536;
 numi::matter::detail::FullContactArmijoScratchBytes scratchAlias{}, scratchCopy{};
 const auto primitive=F;
 const auto leaves=static_cast<std::uint64_t>(std::bit_ceil(primitive));
 const auto aliasOK=numi::matter::detail::estimateFullContactArmijoScratchBytes(
     E,primitive,leaves,cap,N,true,scratchAlias);
 const auto copyOK=numi::matter::detail::estimateFullContactArmijoScratchBytes(
     E,primitive,leaves,cap,N,false,scratchCopy);
 const auto sparse=numi::matter::detail::admitSparsePrivateContact(
     E,static_cast<std::uint32_t>(pair),rigidCap,false,true,true);
 numi::matter::cardboard::cohesive_tiles::Layout tiles{};
 const auto tileOK=numi::matter::cardboard::cohesive_tiles::makeLayout(
     E,CF,0u,8ull*1024ull*1024ull,tiles);
 const auto persistent=shippingPersistentContactBytes(
     N,T,0u,cap,E,nullptr,true,F);
 ShippingMemoryBudgetInput in;
 in.nodeCount=N; in.tetrahedronCount=T; in.cohesiveFaceCount=CF;
 in.contactPairCount=pair; in.activeContactCapacity=pair;
 in.deformableContactCapacity=cap; in.environmentCount=E;
 in.rigidProxyCount=proxies; in.rigidBodyCount=proxies;
 in.rigidGeneralizedCapacity=6; in.materialStateStride=32;
 in.compactPreconditionerRegionCount=3; in.fgmresRestart=16;
 in.compactImmutableSurfaceMap=true; in.surfaceFaceCount=F;
 in.fullContactArmijoRequired=true; in.deformableSelfContactEnabled=true;
 in.fullContactArmijoAliasesSurfaceScratch=true;
 in.fullContactArmijoTiledCohesive=true;
 in.sparsePrivateContactRequested=true; in.sparsePrivateContactCapacity=rigidCap;
 in.sparsePrivateContactBoundedFreeBoxScope=true;
 auto budget=estimateShippingMemoryBudget(in);
 const std::uint64_t logical=6u*cap;
 const std::uint64_t padded=std::bit_ceil(logical);
 const std::uint64_t log2p=std::countr_zero(padded);
 const std::uint64_t comparatorDispatches=log2p*(log2p+1u)/2u;
 const std::uint64_t sortBytes=padded*(sizeof(std::uint64_t)+sizeof(std::uint32_t));
 std::cout<<"{"
 <<"\"N\":"<<N<<",\"T\":"<<T<<",\"F\":"<<F<<",\"cohesive_faces\":"<<CF
 <<",\"pair_domain\":"<<pair<<",\"contact_capacity\":"<<cap<<",\"sparse_capacity\":"<<rigidCap
 <<",\"surface_primitive_count\":"<<primitive<<",\"bvh_leaf_count\":"<<leaves
 <<",\"full_scratch_alias_ok\":"<<(aliasOK?"true":"false")
 <<",\"full_scratch_alias_bytes\":"<<scratchAlias.total
 <<",\"full_scratch_duplicate_ok\":"<<(copyOK?"true":"false")
 <<",\"full_scratch_duplicate_bytes\":"<<scratchCopy.total
 <<",\"production_persistent_contact_bytes\":"<<persistent
 <<",\"sort_logical_corners\":"<<logical<<",\"sort_padded_corners\":"<<padded
 <<",\"sort_workspace_bytes\":"<<sortBytes
 <<",\"sort_comparator_dispatches\":"<<comparatorDispatches
 <<",\"sort_element_visits_upper\":"<<(padded*comparatorDispatches)
 <<",\"sparse_admitted\":"<<(sparse.admitted()?"true":"false")
 <<",\"sparse_bytes\":"<<sparse.totalBytes
 <<",\"sparse_pair_slot_bytes\":"<<sparse.pairSlotBytes
 <<",\"sparse_sample_bytes\":"<<sparse.sampleBytes
 <<",\"sparse_error\":"<<(sparse.error==numi::matter::detail::SparsePrivateContactAdmissionError::none?0:(unsigned)sparse.error)
 <<",\"cohesive_tile_admitted\":"<<(tileOK?"true":"false")
 <<",\"cohesive_page_capacity\":"<<tiles.pageCapacity
 <<",\"cohesive_page_count\":"<<tiles.pages.size()
 <<",\"cohesive_page_bytes\":"<<tiles.pageBytes
 <<",\"runtime_required_descent_admitted\":"<<(budget.requiredResidualDescentArenaAdmitted?"true":"false")
 <<",\"runtime_failure\":\""<<budget.requiredResidualDescentArenaFailure<<"\""
 <<",\"within_memory_estimate\":"<<(budget.withinLimit?"true":"false")
 <<",\"estimated_peak_bytes\":"<<budget.estimatedPeakDomainBytes
 <<",\"estimated_metal_peak_bytes\":"<<budget.estimatedMetalPeakBytes
 <<"}\n";
}