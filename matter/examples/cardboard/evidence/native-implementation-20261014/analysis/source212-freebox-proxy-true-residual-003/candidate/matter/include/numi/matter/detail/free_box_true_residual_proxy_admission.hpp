#pragma once
#include "numi/matter/shared.h"
#include <cstddef>
#include <cstdint>
namespace numi::matter::detail {
enum class FreeBoxCapsuleProxyScopeFailure : std::uint32_t {
 none=0u, proxy_count, contact_state, current_body_state, scene_body_state,
 unsupported_proxy, free_box_count, contact_pair_map, contact_pair_coverage
};
struct FreeBoxCapsuleProxyScopeInput {
 const NMRigidProxyGPU* proxies=nullptr; std::size_t proxyCount=0u;
 const NMContactPairGPU* pairs=nullptr; std::size_t pairCount=0u;
 std::uint32_t femNodeCount=0u, currentBodyCount=0u, currentBodyStride=0u;
 std::uint32_t sceneBodyCount=0u, sceneBodyStride=0u;
 bool requiresCurrentBodies=false, hasPrescribedTranslation=false;
 bool hasPrescribedPrevious=false, hasPrescribedEndpoints=false;
 bool requiresSceneBodies=false, hasPrimalContactArguments=false;
 bool hasCoupledGeneralizedCandidate=false, hasCoupledPointJacobians=false;
};
inline FreeBoxCapsuleProxyScopeFailure validateFreeBoxCapsuleProxyScope(
 const FreeBoxCapsuleProxyScopeInput& in) noexcept {
 if(in.proxyCount<2u || in.proxyCount>16u || !in.proxies ||
    in.pairCount==0u || !in.pairs || in.femNodeCount==0u)
  return FreeBoxCapsuleProxyScopeFailure::proxy_count;
 if(!in.hasPrimalContactArguments || !in.hasCoupledGeneralizedCandidate ||
    !in.hasCoupledPointJacobians)
  return FreeBoxCapsuleProxyScopeFailure::contact_state;
 if(!in.requiresCurrentBodies || !in.hasPrescribedTranslation ||
    !in.hasPrescribedPrevious || !in.hasPrescribedEndpoints ||
    in.currentBodyCount==0u || in.currentBodyStride<in.currentBodyCount)
  return FreeBoxCapsuleProxyScopeFailure::current_body_state;
 std::size_t capsules=0u, boxes=0u, freeBoxes=0u, freeProxy=in.proxyCount;
 for(std::size_t i=0u;i<in.proxyCount;++i) {
  const NMRigidProxyGPU& p=in.proxies[i];
  if(p.adaptiveObjectIndex!=NM_INVALID_INDEX)
   return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
  if(p.shapeKind==NM_RIGID_CAPSULE) {
   if(p.flags!=NM_RIGID_PRESCRIBED_TRANSLATION ||
      p.generalizedFreeBodyIndex!=NM_INVALID_INDEX ||
      p.bodyIndex==NM_INVALID_INDEX || p.bodyIndex>=in.currentBodyCount)
    return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
   ++capsules; continue;
  }
  if(p.shapeKind!=NM_RIGID_BOX || p.flags!=NM_RIGID_DYNAMIC ||
     p.generalizedFreeBodyIndex!=0u || p.reserved2!=0u ||
     p.bodyIndex==NM_INVALID_INDEX || p.bodyIndex>=in.currentBodyCount ||
     p.sceneBodyIndex==NM_INVALID_INDEX || p.sceneBodyIndex>=in.sceneBodyCount)
   return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
  ++boxes; ++freeBoxes; freeProxy=i;
 }
 if(capsules==0u || boxes!=1u || freeBoxes!=1u)
  return FreeBoxCapsuleProxyScopeFailure::free_box_count;
 if(!in.requiresSceneBodies || in.sceneBodyCount==0u ||
    in.sceneBodyStride<in.sceneBodyCount)
  return FreeBoxCapsuleProxyScopeFailure::scene_body_state;
 bool capsulePair=false, freeBoxPair=false;
 for(std::size_t i=0u;i<in.pairCount;++i) {
  const NMContactPairGPU& p=in.pairs[i];
  if(p.objectIndex!=0u || p.continuumNode>=in.femNodeCount ||
     p.rigidProxy>=in.proxyCount)
   return FreeBoxCapsuleProxyScopeFailure::contact_pair_map;
  const NMRigidProxyGPU& proxy=in.proxies[p.rigidProxy];
  capsulePair=capsulePair || proxy.shapeKind==NM_RIGID_CAPSULE;
  freeBoxPair=freeBoxPair ||
      (std::size_t(p.rigidProxy)==freeProxy && proxy.shapeKind==NM_RIGID_BOX);
 }
 if(!capsulePair || !freeBoxPair)
  return FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage;
 return FreeBoxCapsuleProxyScopeFailure::none;
}
inline const char* freeBoxCapsuleProxyScopeFailureReason(
 FreeBoxCapsuleProxyScopeFailure f) noexcept {
 switch(f) {
  case FreeBoxCapsuleProxyScopeFailure::none:return nullptr;
  case FreeBoxCapsuleProxyScopeFailure::proxy_count:return "requires_two_to_sixteen_capsule_box_proxies_and_authored_pairs";
  case FreeBoxCapsuleProxyScopeFailure::contact_state:return "free_box_capsule_true_residual_contact_operator_unavailable";
  case FreeBoxCapsuleProxyScopeFailure::current_body_state:return "free_box_capsule_true_residual_prescribed_body_state_unavailable";
  case FreeBoxCapsuleProxyScopeFailure::scene_body_state:return "free_box_capsule_true_residual_scene_body_state_unavailable";
  case FreeBoxCapsuleProxyScopeFailure::unsupported_proxy:return "requires_one_nonarticulated_free_box_and_prescribed_capsules_only";
  case FreeBoxCapsuleProxyScopeFailure::free_box_count:return "requires_exactly_one_free_box_and_at_least_one_prescribed_capsule";
  case FreeBoxCapsuleProxyScopeFailure::contact_pair_map:return "free_box_capsule_true_residual_pair_outside_fem_proxy_map";
  case FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage:return "requires_authored_contact_pairs_for_capsule_and_free_box";
 }
 return "free_box_capsule_true_residual_unknown_proxy_scope_failure";
}
} // namespace numi::matter::detail
