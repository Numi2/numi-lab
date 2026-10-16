#include "cardboard_shipping_memory.hpp"
#include <bit>
#include <cstdint>
#include <iostream>
using namespace numi::cardboard;
int main() {
  constexpr std::uint64_t E=1,N=528444,T=1477728,F=1054876,C=131072;
  constexpr std::uint64_t grid=0;
  auto leaf=std::uint64_t(std::bit_ceil(F));
  numi::matter::detail::FullContactArmijoScratchBytes a{},d{};
  numi::matter::detail::estimateFullContactArmijoScratchBytes(E,F,leaf,C,N,true,a);
  numi::matter::detail::estimateFullContactArmijoScratchBytes(E,F,leaf,C,N,false,d);
  ShippingMemoryBudget b{};
  auto p=shippingPersistentContactBytes(N,T,grid,C,E,&b,true,F);
  std::cout << "{\"alias_components\":{"
  << "\"candidates\":"<<a.deformableCandidates<<",\"candidate_counts\":"<<a.candidateCounts
  <<",\"candidate_statuses\":"<<a.candidateStatuses<<",\"contacts\":"<<a.deformableContacts
  <<",\"active_indices\":"<<a.deformableActiveIndices<<",\"active_counts\":"<<a.deformableActiveCounts
  <<",\"node_incidence\":"<<a.nodeIncidence<<",\"node_ranges\":"<<a.nodeRanges
  <<",\"active_offsets\":"<<a.activeOffsets<<",\"global_active_indices\":"<<a.globalActiveIndices
  <<",\"dispatch\":"<<a.activeDispatch<<",\"total\":"<<a.total<<"},"
  <<"\"unaliased_total\":"<<d.total<<",\"persistent_components\":{"
  <<"\"surface\":"<<b.persistentContactSurfaceBytes<<",\"slots\":"<<b.persistentContactSlotBytes
  <<",\"auxiliary\":"<<b.persistentContactAuxiliaryBytes<<",\"total\":"<<p<<"}}\n";
}