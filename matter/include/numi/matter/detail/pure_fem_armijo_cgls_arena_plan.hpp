#pragma once
#include "numi/matter/shared.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::matter::detail {
inline constexpr std::uint64_t kFEMPureArmijoMaximumScratchBytes=64ull*1024ull*1024ull;
inline constexpr std::uint64_t kFEMPureArmijoAccountedResidentCeiling=16ull*1024ull*1024ull*1024ull;
inline constexpr std::size_t kFEMPureArmijoMaximumArenaBuffers=25u;
struct FEMPureArmijoArenaShape {
 std::uint64_t environmentCount=0,femNodeCount=0,tetrahedronCount=0,unknownCount=0;
 std::uint64_t contactPairCount=0,cohesiveFaceCount=0,rigidGeneralizedCapacity=0,rigidQCapacity=0;
 std::uint64_t fieldBoundaryCount=0,punctureChannelCount=0,mpmActiveNodeCapacity=0;
 std::uint64_t particleCount=0,mpmGridCount=0,mpmBlockCount=0;
 std::uint64_t maxBufferLength=0,accountedResidentBytes=0;
 bool oneStaticExplicitFEM=false,unsupportedMatterFlags=false,includeCauchy=false,includeCGLS=false,includeRightScale=false;
};
struct FEMPureArmijoArenaBuffer {const char* name=nullptr;std::uint64_t logicalCount=0,elementBytes=0,allocatedBytes=0;};
enum class FEMPureArmijoArenaFailure : std::uint32_t {
 none=0,environment_count,fem_node_count,tetrahedron_count,unsupported_world,unknown_count,cgls_layout,
 right_scale_layout,arithmetic_overflow,dispatch_index_overflow,per_buffer_limit,scratch_limit,accounted_resident_limit
};
struct FEMPureArmijoArenaPlan {
 FEMPureArmijoArenaFailure failure=FEMPureArmijoArenaFailure::none;
 std::array<FEMPureArmijoArenaBuffer,kFEMPureArmijoMaximumArenaBuffers> buffers{};
 std::size_t bufferCount=0;
 std::uint64_t unknownCount=0,scratchBytes=0,maximumScratchBytes=kFEMPureArmijoMaximumScratchBytes;
 std::uint64_t accountedResidentPlusScratchBytes=0,accountedResidentCeiling=kFEMPureArmijoAccountedResidentCeiling;
 bool includeCGLS=false;
 [[nodiscard]] bool admitted() const noexcept{return failure==FEMPureArmijoArenaFailure::none;}
};
[[nodiscard]] inline bool femPureArmijoCheckedMultiply(std::uint64_t a,std::uint64_t b,std::uint64_t& r) noexcept {
 if(a&&b>std::numeric_limits<std::uint64_t>::max()/a)return false;r=a*b;return true;
}
[[nodiscard]] inline bool femPureArmijoCheckedAdd(std::uint64_t a,std::uint64_t b,std::uint64_t& r) noexcept {
 if(b>std::numeric_limits<std::uint64_t>::max()-a)return false;r=a+b;return true;
}
[[nodiscard]] inline bool femPureArmijoAppendBuffer(FEMPureArmijoArenaPlan& p,const char* name,
 std::uint64_t n,std::uint64_t elem,std::uint64_t maxBuffer) noexcept {
 if(!name||!elem||p.bufferCount>=p.buffers.size())return false;
 std::uint64_t bytes=0;if(!femPureArmijoCheckedMultiply(n,elem,bytes))return false;
 const std::uint64_t allocated=bytes<16u?16u:bytes;
 if(allocated>maxBuffer){p.failure=FEMPureArmijoArenaFailure::per_buffer_limit;return false;}
 std::uint64_t next=0;if(!femPureArmijoCheckedAdd(p.scratchBytes,allocated,next))return false;
 p.scratchBytes=next;p.buffers[p.bufferCount++]={name,n,elem,allocated};return true;
}
[[nodiscard]] inline FEMPureArmijoArenaPlan planFEMPureArmijoCGLSArena(const FEMPureArmijoArenaShape& s) noexcept {
 FEMPureArmijoArenaPlan p{};
 if(!s.environmentCount||s.environmentCount>4u){p.failure=FEMPureArmijoArenaFailure::environment_count;return p;}
 if(!s.femNodeCount||s.femNodeCount>32768u){p.failure=FEMPureArmijoArenaFailure::fem_node_count;return p;}
 if(!s.tetrahedronCount||s.tetrahedronCount>131072u){p.failure=FEMPureArmijoArenaFailure::tetrahedron_count;return p;}
 if(!s.oneStaticExplicitFEM||s.unsupportedMatterFlags||s.contactPairCount||s.cohesiveFaceCount||
    s.rigidGeneralizedCapacity||s.rigidQCapacity||s.fieldBoundaryCount||s.punctureChannelCount||s.mpmActiveNodeCapacity||s.particleCount||s.mpmGridCount||s.mpmBlockCount){
  p.failure=FEMPureArmijoArenaFailure::unsupported_world;return p;}
 p.includeCGLS=s.includeCGLS;
 if(s.includeCGLS&&(!s.includeCauchy||s.environmentCount>2u)){p.failure=FEMPureArmijoArenaFailure::cgls_layout;return p;}
 if(s.includeRightScale&&!s.includeCGLS){p.failure=FEMPureArmijoArenaFailure::right_scale_layout;return p;}
 std::uint64_t en=0,et=0,un=0,n3=0;
 if(!femPureArmijoCheckedMultiply(s.environmentCount,s.femNodeCount,en)||
    !femPureArmijoCheckedMultiply(s.environmentCount,s.tetrahedronCount,et)||
    !femPureArmijoCheckedMultiply(en,2u,un)||
    !femPureArmijoCheckedMultiply(en,3u,n3)){p.failure=FEMPureArmijoArenaFailure::arithmetic_overflow;return p;}
 p.unknownCount=un;
 if(s.unknownCount!=un||un>UINT32_MAX){p.failure=FEMPureArmijoArenaFailure::unknown_count;return p;}
 if(en>UINT32_MAX||et>UINT32_MAX||(s.includeRightScale&&n3>UINT32_MAX)){
  p.failure=FEMPureArmijoArenaFailure::dispatch_index_overflow;return p;}
 if(s.maxBufferLength<16u){p.failure=FEMPureArmijoArenaFailure::per_buffer_limit;return p;}
 const auto add=[&](const char* name,std::uint64_t n,std::uint64_t b) noexcept {
  if(p.failure!=FEMPureArmijoArenaFailure::none)return false;
  bool ok=femPureArmijoAppendBuffer(p,name,n,b,s.maxBufferLength);
  if(!ok&&p.failure==FEMPureArmijoArenaFailure::none)p.failure=FEMPureArmijoArenaFailure::arithmetic_overflow;
  return ok;
 };
 const auto e=s.environmentCount;
 if(!add("armijo_candidate_nodes",en,sizeof(NMFEMNodeStateGPU))||
    !add("armijo_element_forces",et,sizeof(NMFEMElementVectorGPU))||
    !add("armijo_residual",un,sizeof(nm_float4))||
    !add("armijo_preconditioned",en,sizeof(nm_float4))||
    !add("armijo_direction",en,sizeof(nm_float4))||
    !add("armijo_operator",en,sizeof(nm_float4))||
    !add("armijo_reactions",en,sizeof(nm_float4))||
    !add("armijo_metrics",e,sizeof(nm_float4))||
    !add("armijo_statuses",e,sizeof(NMMatterStatusGPU)))return p;
 if(s.includeCauchy&&(!add("cauchy_rhs",un,sizeof(nm_float4))||!add("cauchy_trace",e,sizeof(nm_float4))))return p;
 if(s.includeCGLS){
  const char* names[]={"cgls_b","cgls_x","cgls_r","cgls_s","cgls_p","cgls_q","cgls_mask"};
  for(const char* name:names)if(!add(name,un,sizeof(nm_float4)))return p;
  if(!add("cgls_environment_map",un,sizeof(std::uint32_t))||
     !add("cgls_selected",e,sizeof(std::uint32_t))||
     !add("cgls_scalars",e,sizeof(nm_float4))||
     !add("cgls_control",e,sizeof(nm_uint4))||
     !add("cgls_errors",e,sizeof(std::uint32_t)))return p;
 }
 if(s.includeRightScale&&(!add("cgls_right_scale_rows",n3,sizeof(nm_float4))||
                           !add("cgls_right_scale_physical",un,sizeof(nm_float4))))return p;
 if(p.scratchBytes>p.maximumScratchBytes){p.failure=FEMPureArmijoArenaFailure::scratch_limit;return p;}
 if(!femPureArmijoCheckedAdd(s.accountedResidentBytes,p.scratchBytes,p.accountedResidentPlusScratchBytes)){
  p.failure=FEMPureArmijoArenaFailure::arithmetic_overflow;return p;}
 if(p.accountedResidentPlusScratchBytes>p.accountedResidentCeiling){
  p.failure=FEMPureArmijoArenaFailure::accounted_resident_limit;return p;}
 return p;
}
[[nodiscard]] inline const char* femPureArmijoCGLSArenaFailureReason(FEMPureArmijoArenaFailure f) noexcept {
 switch(f){
 case FEMPureArmijoArenaFailure::none:return nullptr;
 case FEMPureArmijoArenaFailure::environment_count:return "pure_fem_armijo_environment_cap_4";
 case FEMPureArmijoArenaFailure::fem_node_count:return "pure_fem_armijo_node_cap_32768";
 case FEMPureArmijoArenaFailure::tetrahedron_count:return "pure_fem_armijo_tetrahedron_cap_131072";
 case FEMPureArmijoArenaFailure::unsupported_world:return "pure_fem_armijo_requires_contact_free_static_explicit_fem";
 case FEMPureArmijoArenaFailure::unknown_count:return "pure_fem_armijo_unknown_count_mismatch_or_uint32_overflow";
 case FEMPureArmijoArenaFailure::cgls_layout:return "pure_fem_armijo_cgls_requires_cauchy_and_at_most_two_environments";
 case FEMPureArmijoArenaFailure::right_scale_layout:return "pure_fem_armijo_right_scale_requires_cgls";
 case FEMPureArmijoArenaFailure::arithmetic_overflow:return "pure_fem_armijo_arena_checked_arithmetic_overflow";
 case FEMPureArmijoArenaFailure::dispatch_index_overflow:return "pure_fem_armijo_arena_exceeds_uint32_dispatch_range";
 case FEMPureArmijoArenaFailure::per_buffer_limit:return "pure_fem_armijo_arena_exceeds_device_max_buffer_length";
 case FEMPureArmijoArenaFailure::scratch_limit:return "pure_fem_armijo_cgls_scratch_exceeds_64_mib_cap";
 case FEMPureArmijoArenaFailure::accounted_resident_limit:return "pure_fem_armijo_accounted_runtime_resident_exceeds_16_gib";
 }
 return "pure_fem_armijo_unknown_arena_failure";
}
} // namespace numi::matter::detail
