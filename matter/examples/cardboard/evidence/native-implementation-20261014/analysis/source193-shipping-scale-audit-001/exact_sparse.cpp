#include "numi/matter/sparse_private_contact_admission.hpp"
#include <cstdint>
#include <iostream>
int main() {
 constexpr std::uint32_t environments=1u, contactNodes=527964u, proxies=12u, capacity=65536u;
 constexpr std::uint64_t pair64=std::uint64_t(contactNodes)*proxies;
 const auto a=numi::matter::detail::admitSparsePrivateContact(
     environments, static_cast<std::uint32_t>(pair64), capacity, false, true, true);
 std::cout<<"{\"contact_nodes\":"<<contactNodes<<",\"proxy_count\":"<<proxies
 <<",\"exact_pair_count\":"<<pair64<<",\"candidate_capacity\":"<<capacity
 <<",\"admitted\":"<<(a.admitted()?"true":"false")<<",\"bytes\":"<<a.totalBytes
 <<",\"pair_slot_bytes\":"<<a.pairSlotBytes<<",\"candidate_id_bytes\":"<<a.candidateIdBytes
 <<",\"sample_bytes\":"<<a.sampleBytes<<",\"error_code\":"<<unsigned(a.error)
 <<",\"budget_bytes\":"<<numi::matter::detail::kSparsePrivateContactScratchBudgetBytes<<"}\n";
}