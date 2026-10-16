#pragma once

#include "numi/matter/shared.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace numi::matter::detail {

// Shipping uses eight distinct prescribed tool/support bodies. Multiple BOX
// rails intentionally share one prescribed body identity.
inline constexpr std::size_t kFreeBoxTrueResidualMaxPrescribedBodies = 8u;

enum class FreeBoxCapsuleProxyScopeFailure : std::uint32_t {
    none = 0u,
    proxy_count,
    contact_state,
    current_body_state,
    scene_body_state,
    unsupported_proxy,
    free_box_count,
    prescribed_body_count,
    contact_pair_map,
    contact_pair_coverage,
};

struct FreeBoxCapsuleProxyScopeInput {
    const NMRigidProxyGPU* proxies = nullptr;
    std::size_t proxyCount = 0u;
    const NMContactPairGPU* pairs = nullptr;
    std::size_t pairCount = 0u;
    std::uint32_t femNodeCount = 0u;
    std::uint32_t currentBodyCount = 0u;
    std::uint32_t currentBodyStride = 0u;
    std::uint32_t sceneBodyCount = 0u;
    std::uint32_t sceneBodyStride = 0u;
    bool requiresCurrentBodies = false;
    bool hasPrescribedTranslation = false;
    bool hasPrescribedPrevious = false;
    bool hasPrescribedEndpoints = false;
    bool requiresSceneBodies = false;
    bool hasPrimalContactArguments = false;
    bool hasCoupledGeneralizedCandidate = false;
    bool hasCoupledPointJacobians = false;
};

inline FreeBoxCapsuleProxyScopeFailure validateFreeBoxCapsuleProxyScope(
    const FreeBoxCapsuleProxyScopeInput& in) noexcept {
    if (in.proxyCount == 0u || in.proxyCount > 16u || !in.proxies ||
        in.pairCount == 0u || !in.pairs || in.femNodeCount == 0u)
        return FreeBoxCapsuleProxyScopeFailure::proxy_count;
    if (!in.hasPrimalContactArguments ||
        !in.hasCoupledGeneralizedCandidate ||
        !in.hasCoupledPointJacobians)
        return FreeBoxCapsuleProxyScopeFailure::contact_state;
    if (!in.requiresCurrentBodies || in.currentBodyCount == 0u ||
        in.currentBodyStride < in.currentBodyCount)
        return FreeBoxCapsuleProxyScopeFailure::current_body_state;

    std::array<std::uint32_t, 16u> prescribedBodyIndices{};
    std::size_t prescribedBodyCount = 0u;
    std::size_t capsules = 0u;
    std::size_t prescribedBoxes = 0u;
    std::size_t freeBoxes = 0u;
    std::size_t freeProxy = in.proxyCount;

    for (std::size_t i = 0u; i < in.proxyCount; ++i) {
        const NMRigidProxyGPU& p = in.proxies[i];
        if (p.adaptiveObjectIndex != NM_INVALID_INDEX)
            return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;

        const bool prescribed =
            p.flags == NM_RIGID_PRESCRIBED_TRANSLATION;
        if (p.shapeKind == NM_RIGID_CAPSULE) {
            if (!prescribed ||
                p.generalizedFreeBodyIndex != NM_INVALID_INDEX ||
                p.bodyIndex == NM_INVALID_INDEX ||
                p.bodyIndex >= in.currentBodyCount ||
                (p.sceneBodyIndex != NM_INVALID_INDEX &&
                 p.sceneBodyIndex >= in.sceneBodyCount))
                return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
            ++capsules;
        } else if (p.shapeKind == NM_RIGID_BOX && prescribed) {
            if (p.generalizedFreeBodyIndex != NM_INVALID_INDEX ||
                p.bodyIndex == NM_INVALID_INDEX ||
                p.bodyIndex >= in.currentBodyCount ||
                (p.sceneBodyIndex != NM_INVALID_INDEX &&
                 p.sceneBodyIndex >= in.sceneBodyCount))
                return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
            ++prescribedBoxes;
        } else if (p.shapeKind == NM_RIGID_BOX &&
                   p.flags == NM_RIGID_DYNAMIC) {
            if (p.generalizedFreeBodyIndex != 0u || p.reserved2 != 0u ||
                p.bodyIndex == NM_INVALID_INDEX ||
                p.bodyIndex >= in.currentBodyCount ||
                p.sceneBodyIndex == NM_INVALID_INDEX ||
                p.sceneBodyIndex >= in.sceneBodyCount)
                return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
            ++freeBoxes;
            freeProxy = i;
        } else {
            return FreeBoxCapsuleProxyScopeFailure::unsupported_proxy;
        }

        if (prescribed) {
            bool seen = false;
            for (std::size_t body = 0u;
                 body < prescribedBodyCount; ++body) {
                seen = seen ||
                    prescribedBodyIndices[body] == p.bodyIndex;
            }
            if (!seen) {
                if (prescribedBodyCount >=
                    kFreeBoxTrueResidualMaxPrescribedBodies)
                    return FreeBoxCapsuleProxyScopeFailure::prescribed_body_count;
                prescribedBodyIndices[prescribedBodyCount++] = p.bodyIndex;
            }
        }
    }

    if (freeBoxes != 1u)
        return FreeBoxCapsuleProxyScopeFailure::free_box_count;
    if ((capsules != 0u || prescribedBoxes != 0u) &&
        (!in.hasPrescribedTranslation || !in.hasPrescribedPrevious ||
         !in.hasPrescribedEndpoints))
        return FreeBoxCapsuleProxyScopeFailure::current_body_state;
    if (!in.requiresSceneBodies || in.sceneBodyCount == 0u ||
        in.sceneBodyStride < in.sceneBodyCount)
        return FreeBoxCapsuleProxyScopeFailure::scene_body_state;

    bool capsulePair = false;
    bool prescribedBoxPair = false;
    bool freeBoxPair = false;
    for (std::size_t i = 0u; i < in.pairCount; ++i) {
        const NMContactPairGPU& p = in.pairs[i];
        if (p.objectIndex != 0u || p.continuumNode >= in.femNodeCount ||
            p.rigidProxy >= in.proxyCount)
            return FreeBoxCapsuleProxyScopeFailure::contact_pair_map;
        const NMRigidProxyGPU& proxy = in.proxies[p.rigidProxy];
        capsulePair = capsulePair ||
            proxy.shapeKind == NM_RIGID_CAPSULE;
        prescribedBoxPair = prescribedBoxPair ||
            (proxy.shapeKind == NM_RIGID_BOX &&
             proxy.flags == NM_RIGID_PRESCRIBED_TRANSLATION);
        freeBoxPair = freeBoxPair ||
            (static_cast<std::size_t>(p.rigidProxy) == freeProxy &&
             proxy.shapeKind == NM_RIGID_BOX &&
             proxy.flags == NM_RIGID_DYNAMIC);
    }
    if (!freeBoxPair || (capsules != 0u && !capsulePair) ||
        (prescribedBoxes != 0u && !prescribedBoxPair))
        return FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage;
    return FreeBoxCapsuleProxyScopeFailure::none;
}

inline const char* freeBoxCapsuleProxyScopeFailureReason(
    const FreeBoxCapsuleProxyScopeFailure failure) noexcept {
    switch (failure) {
        case FreeBoxCapsuleProxyScopeFailure::none:
            return nullptr;
        case FreeBoxCapsuleProxyScopeFailure::proxy_count:
            return "requires_one_to_sixteen_supported_proxies_and_authored_pairs";
        case FreeBoxCapsuleProxyScopeFailure::contact_state:
            return "free_box_capsule_true_residual_contact_operator_unavailable";
        case FreeBoxCapsuleProxyScopeFailure::current_body_state:
            return "free_box_capsule_true_residual_prescribed_body_state_unavailable";
        case FreeBoxCapsuleProxyScopeFailure::scene_body_state:
            return "free_box_capsule_true_residual_scene_body_state_unavailable";
        case FreeBoxCapsuleProxyScopeFailure::unsupported_proxy:
            return "requires_one_nonarticulated_free_box_and_prescribed_capsule_or_box_tools";
        case FreeBoxCapsuleProxyScopeFailure::free_box_count:
            return "requires_exactly_one_free_box";
        case FreeBoxCapsuleProxyScopeFailure::prescribed_body_count:
            return "prescribed_tool_body_count_exceeds_eight";
        case FreeBoxCapsuleProxyScopeFailure::contact_pair_map:
            return "free_box_capsule_true_residual_pair_outside_fem_proxy_map";
        case FreeBoxCapsuleProxyScopeFailure::contact_pair_coverage:
            return "requires_authored_contact_pairs_for_each_proxy_shape_class";
    }
    return "free_box_capsule_true_residual_unknown_proxy_scope_failure";
}

} // namespace numi::matter::detail
