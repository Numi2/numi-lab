#pragma once
#include "numi/matter/shared.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
namespace numi::matter::detail {
struct FEMTrueResidualArenaShape { std::uint64_t environmentCount=0u,femNodeCount=0u,tetrahedronCount=0u,cohesiveFaceCount=0u,rigidGeneralizedCapacity=0u,maxBufferLength=0u; bool extendedPureFEM=false; };
struct FEMTrueResidualArenaBuffer { std::uint64_t logicalCount=0u,elementBytes=0u,allocatedBytes=0u; };
enum class FEMTrueResidualArenaFailure : std::uint32_t { none=0u,extended_fem_node_count,extended_tetrahedron_count,extended_pure_layout,environment_count,fem_node_count,tetrahedron_count,cohesive_face_count,rigid_layout,arithmetic_overflow,per_buffer_limit,aggregate_limit };
struct FEMTrueResidualArenaPlan {
 FEMTrueResidualArenaFailure failure=FEMTrueResidualArenaFailure::none;
 FEMTrueResidualArenaBuffer rightHandSide,metrics,statuses,elementOperator,cohesiveOperator,operatorValue;
 std::uint64_t unknownCount=0u,aggregateBytes=0u,maximumAggregateBytes=0u;
 [[nodiscard]] bool admitted() const noexcept {return failure==FEMTrueResidualArenaFailure::none;}
};
[[nodiscard]] inline bool femTrueResidualCheckedMultiply(std::uint64_t a,std::uint64_t b,std::uint64_t& r) noexcept {if(a&&b>std::numeric_limits<std::uint64_t>::max()/a)return false;r=a*b;return true;}
[[nodiscard]] inline bool femTrueResidualCheckedAdd(std::uint64_t a,std::uint64_t b,std::uint64_t& r) noexcept {if(b>std::numeric_limits<std::uint64_t>::max()-a)return false;r=a+b;return true;}
[[nodiscard]] inline bool femTrueResidualPlanBuffer(std::uint64_t n,std::uint64_t element,FEMTrueResidualArenaBuffer& out) noexcept {
 std::uint64_t bytes=0u;if(!element||!femTrueResidualCheckedMultiply(n,element,bytes))return false;
 out.logicalCount=n;out.elementBytes=element;out.allocatedBytes=bytes>element?(bytes>16u?bytes:16u):(element>16u?element:16u);return true;
}
[[nodiscard]] inline bool femTrueResidualSumArenaBytes(FEMTrueResidualArenaPlan& p) noexcept {
 std::uint64_t total=0u;const FEMTrueResidualArenaBuffer* b[]={&p.rightHandSide,&p.metrics,&p.statuses,&p.elementOperator,&p.cohesiveOperator,&p.operatorValue};
 for(const auto* x:b)if(!femTrueResidualCheckedAdd(total,x->allocatedBytes,total))return false;p.aggregateBytes=total;return true;
}
[[nodiscard]] inline bool femTrueResidualPlanSupportedCounts(std::uint64_t e,std::uint64_t n,std::uint64_t t,std::uint64_t c,std::uint64_t r,FEMTrueResidualArenaPlan& p,bool extendedPureFEM=false) noexcept {
 if(extendedPureFEM&&(c||r)){p.failure=FEMTrueResidualArenaFailure::extended_pure_layout;return false;}
 if(!e||e>4u){p.failure=FEMTrueResidualArenaFailure::environment_count;return false;}
 if(!n||n>(extendedPureFEM?32768u:8192u)){p.failure=extendedPureFEM?FEMTrueResidualArenaFailure::extended_fem_node_count:FEMTrueResidualArenaFailure::fem_node_count;return false;}
 if(!t||t>(extendedPureFEM?131072u:32768u)){p.failure=extendedPureFEM?FEMTrueResidualArenaFailure::extended_tetrahedron_count:FEMTrueResidualArenaFailure::tetrahedron_count;return false;}
 if(c>4096u){p.failure=FEMTrueResidualArenaFailure::cohesive_face_count;return false;}
 if((r!=0u&&r!=6u)||(r==6u&&e!=1u)){p.failure=FEMTrueResidualArenaFailure::rigid_layout;return false;}
 std::uint64_t en=0u,mf=0u,rigid=0u,nt=0u,nc=0u;
 if(!femTrueResidualCheckedMultiply(e,n,en)||!femTrueResidualCheckedMultiply(en,2u,mf)||!femTrueResidualCheckedMultiply(e,r,rigid)||!femTrueResidualCheckedAdd(mf,rigid,p.unknownCount)||p.unknownCount>UINT32_MAX||!femTrueResidualCheckedMultiply(e,t,nt)||!femTrueResidualCheckedMultiply(e,c,nc)||
 !femTrueResidualPlanBuffer(p.unknownCount,sizeof(nm_float4),p.rightHandSide)||!femTrueResidualPlanBuffer(e,sizeof(nm_float4),p.metrics)||!femTrueResidualPlanBuffer(e,sizeof(NMMatterStatusGPU),p.statuses)||!femTrueResidualPlanBuffer(nt,sizeof(NMFEMElementVectorGPU),p.elementOperator)||!femTrueResidualPlanBuffer(nc,sizeof(NMCohesiveFaceVectorGPU),p.cohesiveOperator)||!femTrueResidualPlanBuffer(p.unknownCount,sizeof(nm_float4),p.operatorValue)||!femTrueResidualSumArenaBytes(p)){p.failure=FEMTrueResidualArenaFailure::arithmetic_overflow;return false;}
 return true;
}
[[nodiscard]] inline std::uint64_t femTrueResidualMaximumSupportedArenaBytes() noexcept {
 FEMTrueResidualArenaPlan a{},b{},c{};if(!femTrueResidualPlanSupportedCounts(4u,8192u,32768u,4096u,0u,a)||!femTrueResidualPlanSupportedCounts(1u,8192u,32768u,4096u,6u,b))return 0u;if(!femTrueResidualPlanSupportedCounts(4u,32768u,131072u,0u,0u,c,true))return 0u;return std::max(c.aggregateBytes,std::max(a.aggregateBytes,b.aggregateBytes));
}
[[nodiscard]] inline FEMTrueResidualArenaPlan planFEMTrueResidualArena(const FEMTrueResidualArenaShape& s) noexcept {
 FEMTrueResidualArenaPlan p{};if(!femTrueResidualPlanSupportedCounts(s.environmentCount,s.femNodeCount,s.tetrahedronCount,s.cohesiveFaceCount,s.rigidGeneralizedCapacity,p,s.extendedPureFEM))return p;
 const FEMTrueResidualArenaBuffer* b[]={&p.rightHandSide,&p.metrics,&p.statuses,&p.elementOperator,&p.cohesiveOperator,&p.operatorValue};for(const auto* x:b)if(x->allocatedBytes>s.maxBufferLength){p.failure=FEMTrueResidualArenaFailure::per_buffer_limit;return p;}
 p.maximumAggregateBytes=femTrueResidualMaximumSupportedArenaBytes();if(!p.maximumAggregateBytes||p.aggregateBytes>p.maximumAggregateBytes){p.failure=FEMTrueResidualArenaFailure::aggregate_limit;return p;}return p;
}
[[nodiscard]] inline const char* femTrueResidualArenaFailureReason(FEMTrueResidualArenaFailure f) noexcept {
 switch(f){case FEMTrueResidualArenaFailure::extended_fem_node_count:return "pure_fem_node_cap_32768";case FEMTrueResidualArenaFailure::extended_tetrahedron_count:return "pure_tetrahedron_cap_131072";case FEMTrueResidualArenaFailure::extended_pure_layout:return "extended_true_residual_requires_no_cohesive_or_rigid_unknowns";case FEMTrueResidualArenaFailure::none:return nullptr;case FEMTrueResidualArenaFailure::environment_count:return "environment_cap_4";case FEMTrueResidualArenaFailure::fem_node_count:return "fem_node_cap_8192";case FEMTrueResidualArenaFailure::tetrahedron_count:return "tetrahedron_cap_32768";case FEMTrueResidualArenaFailure::cohesive_face_count:return "cohesive_face_cap_4096";case FEMTrueResidualArenaFailure::rigid_layout:return "unsupported_true_residual_rigid_layout";case FEMTrueResidualArenaFailure::arithmetic_overflow:return "true_residual_arena_checked_arithmetic_overflow";case FEMTrueResidualArenaFailure::per_buffer_limit:return "true_residual_arena_exceeds_device_max_buffer_length";case FEMTrueResidualArenaFailure::aggregate_limit:return "true_residual_arena_exceeds_supported_aggregate_bound";}return "true_residual_arena_unknown_failure";
}
} // namespace numi::matter::detail
