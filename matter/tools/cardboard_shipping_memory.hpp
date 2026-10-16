#pragma once

// Memory admission helpers for shipping-size native FEM runs. The static
// estimate intentionally counts only known FEM and Krylov allocations; it is
// a lower bound, never a substitute for RuntimeDiagnostics::residentBytes.

#include "numi/matter/shared.h"
#include "numi/matter/sparse_private_contact_admission.hpp"
#include "cardboard_inside_clearance.hpp"
#include "cardboard_cohesive_armijo_tiles.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace numi::cardboard {

static_assert(sizeof(NMContactSampleGPU) == 160u,
              "rigid contact sample arena estimate must track the native ABI");

inline constexpr std::uint64_t kShippingPeakMemoryLimitBytes =
    16ull * 1024ull * 1024ull * 1024ull;
inline constexpr std::uint32_t kShippingFgmresRestartCeiling =
    NM_MIXED_FGMRES_RESTART;
// The full-contact residual Armijo transaction currently delegates to the
// capsule/box scope, whose native FEM node limit is 8,192.
inline constexpr std::uint64_t kShippingFullContactArmijoNodeLimit = 8192u;
// Three shipping blank bands, each with 16 configured subdivisions along Y.
inline constexpr std::uint32_t kShippingReferenceYIntervals = 48u;
inline constexpr std::uint32_t kShippingYIntervalCeiling = 192u;
inline constexpr std::uint64_t kShippingReferenceNodes = 453378u;
inline constexpr std::uint64_t kShippingReferenceTetrahedra = 1477728u;
// Exact prepared FEM node count for the reference shipping topology after the
// authored starch-side node split. This is an upper bound on contact-eligible
// nodes because all prepared nodes are counted, including interior nodes.
inline constexpr std::uint64_t kShippingReferencePreparedNodes = 528444u;
// Native shipping-memory-preflight-032 measurement at the same reference
// mesh and FGMRES restart 16.
inline constexpr std::uint64_t kShippingCase032RuntimeResidentBytes =
    10'467'551'760ull;

struct ShippingStaticMemoryLowerBound {
    std::uint64_t femNodeStatesBytes = 0u;
    std::uint64_t tetrahedraBytes = 0u;
    std::uint64_t fgmresBasisBytes = 0u;
    std::uint64_t totalBytes = 0u;
    std::uint32_t environmentCount = 1u;
    std::uint32_t restart = kShippingFgmresRestartCeiling;
};

struct ShippingPrebuildMemoryScreen {
    std::uint64_t nominalNodeCount = 0u;
    std::uint64_t nominalTetrahedronCount = 0u;
    std::uint64_t estimatedNodeCount = 0u;
    std::uint64_t estimatedTetrahedronCount = 0u;
    std::uint64_t referencePitchIntervals = 0u;
    std::uint64_t requestedPitchIntervals = 0u;
    std::uint32_t yIntervals = kShippingReferenceYIntervals;
    std::uint32_t thicknessSlices = 1u;
    double cardinalityMargin = 1.25;
    ShippingStaticMemoryLowerBound staticArenaScreen{};
};

// A pre-allocation envelope for the shipping protocol. Runtime does not
// expose per-arena counters, so the FEM/tet/Krylov-derived base is deliberately
// padded to exceed the measured case032 initialization ratio. Known optional
// arenas (RAP, cohesive state, private Armijo contact), host mesh/compile
// allocations, and retained host checkpoints are added as a conservative
// simultaneous-allocation model. Upload staging and one snapshot readback
// reserve appear in both side subtotals but are counted once in the combined
// estimate. RuntimeDiagnostics/currentAllocatedSize remain required
// observations; this envelope is not an exact allocator report.
struct ShippingMemoryBudgetInput {
    std::uint64_t nodeCount = 0u;
    std::uint64_t tetrahedronCount = 0u;
    std::uint64_t cohesiveFaceCount = 0u;
    std::uint64_t contactPairCount = 0u;
    std::uint64_t activeContactCapacity = 0u;
    std::uint64_t deformableContactCapacity = 0u;
    std::uint64_t gridNodeCount = 0u;
    // When the immutable FEM surface map is enabled, `surfaceFaceCount` is
    // the compiled exterior-face count. Zero is valid only for an empty FEM
    // surface; the flag remains explicit so zero cannot silently select the
    // legacy 4*tetrahedra upper bound.
    std::uint64_t surfaceFaceCount = 0u;
    bool compactImmutableSurfaceMap = false;
    std::uint64_t rigidGeneralizedCapacity = 0u;
    std::uint64_t rigidQCapacity = 0u;
    std::uint64_t rigidBodyCount = 0u;
    std::uint64_t rigidProxyCount = 0u;
    std::uint64_t materialStateStride = 0u;
    std::uint64_t compactPreconditionerRegionCount = 0u;
    std::uint64_t environmentCount = 1u;
    std::uint64_t fgmresRestart = kShippingFgmresRestartCeiling;
    std::uint64_t hostSourceMeshBytes = 0u;
    std::uint64_t hostCompiledWorldBytes = 0u;
    std::uint64_t hostGeneratedMetalBytes = 0u;
    std::uint64_t retainedCompleteCheckpointCount = 0u;
    std::uint64_t estimatedSnapshotPayloadBytes = 0u;
    std::uint64_t uploadStagingBytes = 0u;
    // Host-side accepted-state inside-clearance map. It runs after release
    // while the compiled world, source mesh, and checkpoints remain alive.
    bool acceptedInsideClearanceWorkspaceRequired = false;
    bool acceptedInsideClearanceWorkspaceAdmitted = true;
    std::uint64_t acceptedInsideClearanceWorkspaceBytes = 0u;
    std::string acceptedInsideClearanceWorkspaceFailure;
    bool regionalGalerkinDiagnosticsCapture = false;
    bool regionalGalerkinEnabled = false;
    bool regionalGalerkinBlockFineEnabled = false;
    bool rigidContactArmijoEnabled = false;
    bool fullContactArmijoEnabled = false;
    bool cohesiveArmijoEnabled = false;
    // Full shipping assembly requires the private residual-descent arenas,
    // even when this process has not selected the runtime diagnostic env var.
    // Required bytes are estimated independently from currently allocated
    // bytes so preflight can fail closed without pretending to allocate them.
    bool fullContactArmijoRequired = false;
    bool deformableSelfContactEnabled = false;
    bool fullContactArmijoAliasesSurfaceScratch = false;
    bool fullContactArmijoAliasesCohesiveScratch = false;
    bool fullContactArmijoTiledCohesive = false;
    std::uint32_t cohesiveTrialPageCapacity = 0u;
    bool sparsePrivateContactRequested = false;
    bool sparsePrivateContactParity = false;
    std::uint64_t sparsePrivateContactCapacity = 0u;
    bool sparsePrivateContactBoundedFreeBoxScope = false;
    // Probe-side validation of optional runtime selectors. A nonempty value
    // makes shipping admission fail closed before any native initialization.
    std::string runtimeConfigurationFailure;
};

struct ShippingMemoryBudget {
    std::uint64_t staticFEMSolverFloorBytes = 0u;
    std::uint64_t measuredNoncontactRuntimeEnvelopeBytes = 0u;
    std::uint64_t constraintContactForceReadbackBytes = 0u;
    std::uint64_t persistentContactSurfaceBytes = 0u;
    std::uint64_t persistentContactSlotBytes = 0u;
    std::uint64_t persistentContactAuxiliaryBytes = 0u;
    std::uint64_t persistentContactBytes = 0u;
    std::uint64_t rigidContactRuntimeBytes = 0u;
    std::uint64_t measuredBaseAndContactEnvelopeBytes = 0u;
    std::uint64_t cohesiveRuntimeBytes = 0u;
    std::uint64_t regionalGalerkinBytes = 0u;
    std::uint64_t regionalGalerkinBlockFineBytes = 0u;
    std::uint64_t regionalGalerkinDiagnosticsBytes = 0u;
    std::uint64_t rigidContactArmijoBytes = 0u;
    std::uint64_t fullContactArmijoBytes = 0u;
    std::uint64_t cohesiveArmijoBytes = 0u;
    std::uint64_t requiredRigidContactArmijoBytes = 0u;
    std::uint64_t requiredFullContactArmijoBytes = 0u;
    std::uint64_t requiredCohesiveArmijoBytes = 0u;
    std::uint64_t sparsePrivateContactBytes = 0u;
    bool sparsePrivateContactRequested = false;
    bool sparsePrivateContactParity = false;
    std::uint64_t sparsePrivateContactCapacity = 0u;
    bool sparsePrivateContactBoundedFreeBoxScope = false;
    std::uint64_t rigidContactPairDomainUpperBound = 0u;
    std::string runtimeConfigurationFailure;
    bool privateRigidContactAdmitted = true;
    std::string privateRigidContactFailure;
    bool requiredResidualDescentArenaAdmitted = true;
    std::string requiredResidualDescentArenaFailure;
    std::uint64_t uploadStagingBytes = 0u;
    std::uint64_t snapshotReadbackBytes = 0u;
    // Modeled Metal allocation envelope. Upload staging and one snapshot
    // readback reserve are included here and also appear in the host-side
    // subtotal below because both sides may need those buffers during the
    // transfer; the combined estimate counts each shared reserve once.
    std::uint64_t estimatedMetalPeakBytes = 0u;
    std::uint64_t hostMeshAndCompileBytes = 0u;
    std::uint64_t hostCheckpointBytes = 0u;
    bool acceptedInsideClearanceWorkspaceRequired = false;
    std::uint64_t acceptedInsideClearanceWorkspaceBytes = 0u;
    bool acceptedInsideClearanceWorkspaceAdmitted = true;
    std::string acceptedInsideClearanceWorkspaceFailure;
    // Host-only subtotal, including upload/readback reserves shared with the
    // Metal envelope. This is not an observed process-footprint measurement.
    std::uint64_t estimatedHostPeakBytes = 0u;
    // Conservative modeled simultaneous allocation total:
    // Metal envelope + host mesh/compile objects + retained checkpoints +
    // accepted-state inside-clearance host scratch.
    // Shared upload/readback reserves are already in the Metal envelope and
    // are therefore not added again from the host subtotal.
    std::uint64_t estimatedPeakDomainBytes = 0u;
    std::uint64_t limitBytes = kShippingPeakMemoryLimitBytes;
    bool withinLimit = false;
};

inline std::uint64_t shippingMemoryCheckedMultiply(
    const std::uint64_t left, const std::uint64_t right) {
    if (right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right)
        throw std::overflow_error("shipping memory estimate overflow");
    return left * right;
}

inline std::uint64_t shippingMemoryCheckedAdd(
    const std::uint64_t left, const std::uint64_t right) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right)
        throw std::overflow_error("shipping memory estimate overflow");
    return left + right;
}

inline bool shippingAcceptedInsideClearanceWorkspaceBytes(
    const std::uint64_t tetrahedronCount,
    const std::uint64_t nodeCount,
    const std::uint64_t surfaceFaceCount,
    std::uint64_t& workspaceBytes) {
    std::uint64_t hashCapacity = 0u;
    std::uint64_t hashBytes = 0u;
    return inside_clearance_detail::ExteriorFaceIndex::capacityFor(
               tetrahedronCount, hashCapacity, hashBytes) &&
        inside_clearance_detail::estimateWorkspaceBytes(
               hashBytes, nodeCount, surfaceFaceCount, workspaceBytes);
}

// Before mesh preparation, the exact reference shipping blank has a known
// prepared-node count (including the starch-side duplicate nodes). For other
// requested geometries, use the existing 25%-padded cardinality estimate.
// The encoder still replaces this planning bound with exact compiled pair
// counts and verifies the rigid-proxy layout before any trial dispatch.
inline std::uint64_t shippingPrebuildRigidNodeUpperBound(
    const ShippingPrebuildMemoryScreen& screen) {
    if (screen.nominalNodeCount == kShippingReferenceNodes &&
        screen.nominalTetrahedronCount == kShippingReferenceTetrahedra)
        return kShippingReferencePreparedNodes;
    return std::max(screen.nominalNodeCount, screen.estimatedNodeCount);
}

inline std::uint64_t shippingPrebuildTetrahedronUpperBound(
    const ShippingPrebuildMemoryScreen& screen) {
    if (screen.nominalNodeCount == kShippingReferenceNodes &&
        screen.nominalTetrahedronCount == kShippingReferenceTetrahedra)
        return kShippingReferenceTetrahedra;
    return std::max(screen.nominalTetrahedronCount,
                    screen.estimatedTetrahedronCount);
}

inline std::uint64_t shippingCombinedAllocationPeakBytes(
    const ShippingMemoryBudget& budget) {
    return shippingMemoryCheckedAdd(shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(budget.estimatedMetalPeakBytes,
                                 budget.hostMeshAndCompileBytes),
        budget.hostCheckpointBytes),
        budget.acceptedInsideClearanceWorkspaceBytes);
}

inline std::uint64_t shippingMemoryCheckedMultiply3(
    const std::uint64_t first, const std::uint64_t second,
    const std::uint64_t third) {
    return shippingMemoryCheckedMultiply(
        shippingMemoryCheckedMultiply(first, second), third);
}

struct ShippingSparsePrivateContactOptions {
    bool requested = false;
    bool parity = false;
    std::uint32_t capacity = 0u;
    const char* error = nullptr;
};

struct ShippingFullContactSelectorOptions {
    bool requested = false;
    bool selectsAllSteps = false;
    std::uint32_t controlStep = 0u;
    const char* error = nullptr;
};

// Runtime accepts `all` or an unsigned decimal uint32 control-step index.
// Mirror that syntax in the shipping preflight so a malformed selector cannot
// be reported as a viable sparse allocation before Runtime rejects it.
inline ShippingFullContactSelectorOptions parseShippingFullContactSelector(
    const char* selectorText) noexcept {
    ShippingFullContactSelectorOptions result;
    result.requested = selectorText != nullptr;
    if (!result.requested) return result;
    if (std::strcmp(selectorText, "all") == 0) {
        result.selectsAllSteps = true;
        return result;
    }
    if (selectorText[0] == '\0') {
        result.error = "full-contact Armijo selector is empty";
        return result;
    }
    std::uint64_t value = 0u;
    for (const char* cursor = selectorText; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            result.error =
                "full-contact Armijo selector must be decimal or all";
            return result;
        }
        const std::uint32_t digit = static_cast<std::uint32_t>(*cursor - '0');
        if (value >
            (std::numeric_limits<std::uint32_t>::max() - digit) / 10u) {
            result.error = "full-contact Armijo selector exceeds uint32 indexing";
            return result;
        }
        value = value * 10u + digit;
    }
    result.controlStep = static_cast<std::uint32_t>(value);
    return result;
}

[[nodiscard]] inline bool shippingRigidPrivatePairDomainAdmitted(
    const std::uint64_t pairCount,
    const bool sparseRequested) noexcept {
    return pairCount != 0u &&
        (sparseRequested ||
         pairCount <= numi::matter::detail::kDensePrivateContactPairLimit);
}

// Keep the shipping preflight parser byte-for-byte compatible with runtime's
// strict decimal environment contract. This function accepts the captured
// text so CPU checks can exercise malformed values without mutating process
// state.
inline ShippingSparsePrivateContactOptions
parseShippingSparsePrivateContactOptions(
    const char* capacityText,
    const char* parityText) noexcept {
    ShippingSparsePrivateContactOptions result;
    result.requested = capacityText != nullptr;
    if (parityText != nullptr) {
        if (std::strcmp(parityText, "1") != 0) {
            result.error =
                "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY must be exactly 1";
            return result;
        }
        result.parity = true;
    }
    if (!result.requested) {
        if (result.parity)
            result.error =
                "sparse private parity requires an explicit candidate capacity";
        return result;
    }
    if (capacityText[0] == '\0') {
        result.error =
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY must be a positive decimal integer";
        return result;
    }
    std::uint64_t value = 0u;
    for (const char* cursor = capacityText; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            result.error =
                "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY must contain only decimal digits";
            return result;
        }
        const std::uint32_t digit = static_cast<std::uint32_t>(*cursor - '0');
        if (value >
            (std::numeric_limits<std::uint32_t>::max() - digit) / 10u) {
            result.error =
                "sparse private candidate capacity exceeds uint32 indexing";
            return result;
        }
        value = value * 10u + digit;
    }
    if (value == 0u) {
        result.error =
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY must be positive";
        return result;
    }
    result.capacity = static_cast<std::uint32_t>(value);
    return result;
}

// Keep optional trial-arena savings conditional on the same strict text and
// exclusive-selector contract used by runtime creation. Invalid requests
// retain the legacy conservative accounting and a configuration failure.
template<class Options>
inline void configureShippingTiledCohesiveTrial(
    Options& options, const char* tiledText, const char* pageText,
    const bool exclusiveFullContactSelected,
    const bool cohesiveScratchAliasRequested) {
    options.fullContactArmijoTiledCohesive = false;
    options.cohesiveTrialPageCapacity = 0u;
    const auto reject = [&](const char* error) {
        if (options.runtimeConfigurationFailure.empty())
            options.runtimeConfigurationFailure = error;
    };
    if (tiledText == nullptr) {
        if (pageText != nullptr)
            reject("cohesive trial page capacity requires tiled evaluation");
        return;
    }
    if (std::strcmp(tiledText, "1") != 0) {
        reject("NM_FEM_FULL_CONTACT_TILED_COHESIVE must be exactly 1");
        return;
    }
    if (!exclusiveFullContactSelected || cohesiveScratchAliasRequested) {
        reject("tiled cohesive evaluation requires exclusive full-contact Armijo without cohesive scratch aliasing");
        return;
    }
    std::uint32_t capacity = 0u;
    if (pageText != nullptr) {
        if (*pageText == '\0') {
            reject("cohesive trial page capacity must be a positive decimal integer");
            return;
        }
        for (const char* p = pageText; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') {
                reject("cohesive trial page capacity must contain decimal digits only");
                return;
            }
            const auto digit = static_cast<std::uint32_t>(*p - '0');
            if (capacity >
                (std::numeric_limits<std::uint32_t>::max() - digit) / 10u) {
                reject("cohesive trial page capacity exceeds uint32 indexing");
                return;
            }
            capacity = capacity * 10u + digit;
        }
        if (capacity == 0u) {
            reject("cohesive trial page capacity must be positive");
            return;
        }
    }
    if (options.runtimeConfigurationFailure.empty()) {
        options.fullContactArmijoTiledCohesive = true;
        options.cohesiveTrialPageCapacity = capacity;
    }
}

// Allocation-time predicate mirrored from the runtime's bounded sparse
// one-body BOX scope. The transaction encoder still checks the uploaded body
// bridge, exact pair coverage, and force rows before dispatch.
inline bool shippingSparsePrivateBoundedFreeBoxLayout(
    const NMMatterDispatchGPU& dispatch,
    const std::span<const NMRigidProxyGPU> proxies) noexcept {
    if (dispatch.rigidGeneralizedCapacity != 6u ||
        dispatch.rigidQCapacity != 0u ||
        proxies.size() != dispatch.rigidProxyCount || proxies.empty())
        return false;
    bool hasFreeBody = false;
    bool hasCapsule = false;
    bool hasBox = false;
    std::uint32_t freeBodyIndex = NM_INVALID_INDEX;
    std::uint32_t freeSceneBodyIndex = NM_INVALID_INDEX;
    for (const NMRigidProxyGPU& proxy : proxies) {
        if (proxy.adaptiveObjectIndex != NM_INVALID_INDEX) return false;
        if (proxy.shapeKind == NM_RIGID_CAPSULE) {
            hasCapsule = true;
            if (proxy.flags != NM_RIGID_PRESCRIBED_TRANSLATION ||
                proxy.generalizedFreeBodyIndex != NM_INVALID_INDEX ||
                proxy.bodyIndex == NM_INVALID_INDEX)
                return false;
            continue;
        }
        if (proxy.shapeKind != NM_RIGID_BOX) return false;
        hasBox = true;
        if (proxy.generalizedFreeBodyIndex != NM_INVALID_INDEX) {
            if (proxy.flags != NM_RIGID_DYNAMIC ||
                proxy.generalizedFreeBodyIndex != 0u ||
                proxy.bodyIndex == NM_INVALID_INDEX ||
                proxy.sceneBodyIndex == NM_INVALID_INDEX ||
                proxy.reserved2 != 0u)
                return false;
            if (hasFreeBody &&
                (freeBodyIndex != proxy.bodyIndex ||
                 freeSceneBodyIndex != proxy.sceneBodyIndex))
                return false;
            hasFreeBody = true;
            freeBodyIndex = proxy.bodyIndex;
            freeSceneBodyIndex = proxy.sceneBodyIndex;
            continue;
        }
        const bool staticBox = proxy.flags == 0u &&
            proxy.sceneBodyIndex == NM_INVALID_INDEX;
        const bool prescribedBox =
            proxy.flags == NM_RIGID_PRESCRIBED_TRANSLATION &&
            proxy.bodyIndex != NM_INVALID_INDEX &&
            proxy.sceneBodyIndex == NM_INVALID_INDEX &&
            proxy.reserved2 == 0u;
        if ((!staticBox && !prescribedBox) ||
            (proxy.flags & NM_RIGID_ARTICULATED) != 0u)
            return false;
    }
    return hasFreeBody && hasCapsule && hasBox;
}

inline std::uint64_t shippingPersistentContactBytes(
    const std::uint64_t nodeCount,
    const std::uint64_t tetrahedronCount,
    const std::uint64_t gridNodeCount,
    const std::uint64_t deformableContactCapacity,
    const std::uint64_t environmentCount,
    ShippingMemoryBudget* breakdown = nullptr,
    const bool compactImmutableSurfaceMap = false,
    const std::uint64_t surfaceFaceCount = 0u) {
    if (compactImmutableSurfaceMap &&
        tetrahedronCount != 0u && surfaceFaceCount == 0u)
        throw std::invalid_argument(
            "compact shipping contact map requires its compiled exterior-face count");
    const std::uint64_t femPrimitiveCount = compactImmutableSurfaceMap
        ? surfaceFaceCount
        : shippingMemoryCheckedMultiply(tetrahedronCount, 4u);
    const std::uint64_t primitiveCount = shippingMemoryCheckedAdd(
        femPrimitiveCount, gridNodeCount);
    if (primitiveCount > (std::uint64_t{1u} << 63u))
        throw std::overflow_error(
            "shipping contact BVH primitive count exceeds bit-ceil range");
    const std::uint64_t leafCount =
        primitiveCount == 0u || deformableContactCapacity == 0u
            ? 0u : static_cast<std::uint64_t>(std::bit_ceil(primitiveCount));
    const std::uint64_t surfaceAndSort = shippingMemoryCheckedMultiply3(
        environmentCount, primitiveCount,
        sizeof(NMContinuumSurfacePrimitiveGPU) +
            4u * sizeof(std::uint32_t));
    const std::uint64_t bvh = shippingMemoryCheckedMultiply3(
        environmentCount, leafCount, 4u * sizeof(nm_float4));
    const std::uint64_t slots = shippingMemoryCheckedMultiply3(
        environmentCount, deformableContactCapacity,
        sizeof(NMDeformableContactCandidateGPU) +
            sizeof(NMDeformableContactGPU) +
            3u * sizeof(NMDeformableContactHistoryGPU) +
            2u * sizeof(std::uint32_t) + 6u * sizeof(std::uint32_t));
    const std::uint64_t ranges = shippingMemoryCheckedMultiply3(
        environmentCount, shippingMemoryCheckedAdd(nodeCount, gridNodeCount),
        sizeof(NMIncidenceRangeGPU));
    constexpr std::uint64_t perEnvironmentScalars =
        4u * sizeof(std::uint32_t) + sizeof(NMIndirectDispatchGPU);
    const std::uint64_t smallCounts = shippingMemoryCheckedMultiply(
        environmentCount, perEnvironmentScalars);
    const std::uint64_t surface = shippingMemoryCheckedAdd(surfaceAndSort, bvh);
    const std::uint64_t auxiliary = shippingMemoryCheckedAdd(ranges, smallCounts);
    const std::uint64_t total = shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(surface, slots), auxiliary);
    if (breakdown) {
        breakdown->persistentContactSurfaceBytes = surface;
        breakdown->persistentContactSlotBytes = slots;
        breakdown->persistentContactAuxiliaryBytes = auxiliary;
        breakdown->persistentContactBytes = total;
    }
    return total;
}

inline std::uint64_t shippingRigidContactRuntimeBytes(
    const ShippingMemoryBudgetInput& input) {
    const std::uint64_t environments = input.environmentCount;
    const std::uint64_t pairCount = input.contactPairCount;
    const std::uint64_t proxyCount = input.rigidProxyCount;
    const std::uint64_t bodyCount = input.rigidBodyCount;
    // Immutable authored tables/incidence maps use UploadPlan::one and are
    // not duplicated per environment.
    constexpr std::uint64_t perSharedPairBytes =
        sizeof(NMContactPairGPU) + 2u * 2u * sizeof(std::uint32_t);
    std::uint64_t shared = shippingMemoryCheckedMultiply(
        pairCount, perSharedPairBytes);
    shared = shippingMemoryCheckedAdd(shared, shippingMemoryCheckedMultiply(
        shippingMemoryCheckedAdd(input.nodeCount, input.gridNodeCount),
        sizeof(NMIncidenceRangeGPU)));
    shared = shippingMemoryCheckedAdd(shared, shippingMemoryCheckedMultiply(
        proxyCount, sizeof(NMIncidenceRangeGPU) + sizeof(NMRigidProxyGPU) +
            sizeof(std::uint32_t)));
    shared = shippingMemoryCheckedAdd(shared, shippingMemoryCheckedMultiply(
        bodyCount, sizeof(NMIncidenceRangeGPU)));

    // Per-environment contact samples, accepted/candidate/checkpoint friction
    // history, active maps, rigid state/reaction workspaces, and body-proxy
    // reductions remain mutable and are not shared.
    constexpr std::uint64_t perEnvironmentPairBytes =
        sizeof(NMContactSampleGPU) + 3u * sizeof(nm_float4) +
        sizeof(std::uint32_t);
    std::uint64_t repeated = shippingMemoryCheckedMultiply3(
        environments, pairCount, perEnvironmentPairBytes);
    repeated = shippingMemoryCheckedAdd(repeated,
        shippingMemoryCheckedMultiply3(
            environments, input.activeContactCapacity, sizeof(std::uint32_t)));
    repeated = shippingMemoryCheckedAdd(repeated,
        shippingMemoryCheckedMultiply3(
            environments, proxyCount,
            4u * sizeof(NMRigidStateGPU) +
                3u * sizeof(NMRigidReactionGPU)));
    return shippingMemoryCheckedAdd(shared, repeated);
}

inline ShippingStaticMemoryLowerBound estimateShippingStaticMemoryLowerBound(
    const std::uint64_t nodeCount,
    const std::uint64_t tetrahedronCount,
    const std::uint32_t environmentCount = 1u,
    const std::uint32_t restart = kShippingFgmresRestartCeiling) {
    if (nodeCount == 0u || tetrahedronCount == 0u || environmentCount == 0u ||
        restart == 0u || restart > kShippingFgmresRestartCeiling)
        throw std::invalid_argument("shipping memory estimate dimensions are invalid");

    ShippingStaticMemoryLowerBound result;
    result.environmentCount = environmentCount;
    result.restart = restart;
    const std::uint64_t nodesAcrossEnvironments = shippingMemoryCheckedMultiply(
        nodeCount, environmentCount);
    const std::uint64_t tetrahedraAcrossEnvironments = shippingMemoryCheckedMultiply(
        tetrahedronCount, environmentCount);
    result.femNodeStatesBytes = shippingMemoryCheckedMultiply(
        nodesAcrossEnvironments, sizeof(NMFEMNodeStateGPU));
    result.tetrahedraBytes = shippingMemoryCheckedMultiply(
        tetrahedraAcrossEnvironments, sizeof(NMTetrahedronGPU));

    // The runtime has two FEM mixed unknowns per node before any rigid,
    // contact, field, MPM, or vascular unknowns. Its two Krylov vector arenas
    // contain (restart+1)+restart nm_float4 entries per mixed unknown.
    const std::uint64_t mixedUnknownLowerBound = shippingMemoryCheckedMultiply(
        nodesAcrossEnvironments, 2u);
    const std::uint64_t krylovVectors =
        static_cast<std::uint64_t>(restart + 1u) + restart;
    result.fgmresBasisBytes = shippingMemoryCheckedMultiply(
        shippingMemoryCheckedMultiply(mixedUnknownLowerBound, krylovVectors),
        sizeof(nm_float4));
    result.totalBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(result.femNodeStatesBytes,
                                 result.tetrahedraBytes),
        result.fgmresBasisBytes);
    return result;
}

inline ShippingMemoryBudget estimateShippingMemoryBudget(
    const ShippingMemoryBudgetInput& input,
    const std::uint64_t limitBytes = kShippingPeakMemoryLimitBytes) {
    if (input.nodeCount == 0u || input.tetrahedronCount == 0u ||
        input.environmentCount == 0u || input.fgmresRestart == 0u ||
        input.fgmresRestart > kShippingFgmresRestartCeiling || limitBytes == 0u ||
        input.materialStateStride > 4096u ||
        input.compactPreconditionerRegionCount > 32u ||
        input.contactPairCount > std::numeric_limits<std::uint32_t>::max() ||
        input.activeContactCapacity > input.contactPairCount ||
        input.rigidGeneralizedCapacity > std::numeric_limits<std::uint32_t>::max() ||
        input.rigidQCapacity > std::numeric_limits<std::uint32_t>::max() ||
        input.deformableContactCapacity > std::numeric_limits<std::uint32_t>::max() ||
        input.rigidProxyCount > std::numeric_limits<std::uint32_t>::max() ||
        input.rigidBodyCount > std::numeric_limits<std::uint32_t>::max() ||
        input.surfaceFaceCount > std::numeric_limits<std::uint32_t>::max() ||
        input.sparsePrivateContactCapacity >
            std::numeric_limits<std::uint32_t>::max() ||
        (input.sparsePrivateContactParity &&
         !input.sparsePrivateContactRequested) ||
        (input.regionalGalerkinEnabled &&
         input.compactPreconditionerRegionCount == 0u) ||
        (input.regionalGalerkinBlockFineEnabled &&
         !input.regionalGalerkinEnabled))
        throw std::invalid_argument("shipping arena budget dimensions are invalid");
    if (input.compactImmutableSurfaceMap &&
        input.surfaceFaceCount > shippingMemoryCheckedMultiply(
            input.tetrahedronCount, 4u))
        throw std::invalid_argument(
            "compact shipping surface count exceeds all tetrahedron faces");

    const auto asU32 = [](const std::uint64_t value, const char* field) {
        if (value > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(std::string("shipping arena count overflow: ") + field);
        return static_cast<std::uint32_t>(value);
    };
    const auto staticFloor = estimateShippingStaticMemoryLowerBound(
        input.nodeCount, input.tetrahedronCount,
        asU32(input.environmentCount, "environment count"),
        asU32(input.fgmresRestart, "FGMRES restart"));

    ShippingMemoryBudget result;
    result.limitBytes = limitBytes;
    result.sparsePrivateContactRequested =
        input.sparsePrivateContactRequested;
    result.sparsePrivateContactParity = input.sparsePrivateContactParity;
    result.sparsePrivateContactCapacity = input.sparsePrivateContactCapacity;
    result.sparsePrivateContactBoundedFreeBoxScope =
        input.sparsePrivateContactBoundedFreeBoxScope;
    result.rigidContactPairDomainUpperBound = input.contactPairCount;
    result.runtimeConfigurationFailure = input.runtimeConfigurationFailure;
    result.staticFEMSolverFloorBytes = staticFloor.totalBytes;
    // Decompose the measured reference runtime into exact contact allocations
    // and a conservative remainder. Scale the remainder with the FEM/tet/
    // Krylov floor, then add the actual contact capacities below. This avoids
    // double-counting the large contact arenas already present in case032.
    const std::uint64_t referenceFloor = estimateShippingStaticMemoryLowerBound(
        kShippingReferenceNodes, kShippingReferenceTetrahedra, 1u,
        kShippingFgmresRestartCeiling).totalBytes;
    const std::uint64_t referenceContact = shippingPersistentContactBytes(
        kShippingReferenceNodes, kShippingReferenceTetrahedra, 0u,
        shippingMemoryCheckedMultiply(kShippingReferenceTetrahedra, 8u),
        1u);
    ShippingMemoryBudgetInput referenceRigidInput;
    referenceRigidInput.nodeCount = kShippingReferenceNodes;
    referenceRigidInput.tetrahedronCount = kShippingReferenceTetrahedra;
    const std::uint64_t referenceRigidContact =
        shippingRigidContactRuntimeBytes(referenceRigidInput);
    if (referenceFloor == 0u ||
        kShippingCase032RuntimeResidentBytes <=
            shippingMemoryCheckedAdd(referenceContact, referenceRigidContact))
        throw std::logic_error(
            "shipping case032 memory anchor is inconsistent with contact allocation inventory");
    const std::uint64_t referenceRemainder =
        kShippingCase032RuntimeResidentBytes -
        shippingMemoryCheckedAdd(referenceContact, referenceRigidContact);
    result.measuredNoncontactRuntimeEnvelopeBytes =
        static_cast<std::uint64_t>(std::ceil(
            static_cast<long double>(referenceRemainder) *
            static_cast<long double>(staticFloor.totalBytes) /
            static_cast<long double>(referenceFloor)));

    const std::uint64_t environments = input.environmentCount;
    const std::uint64_t persistentContact = shippingPersistentContactBytes(
        input.nodeCount, input.tetrahedronCount, input.gridNodeCount,
        input.deformableContactCapacity, environments, &result,
        input.compactImmutableSurfaceMap, input.surfaceFaceCount);
    result.rigidContactRuntimeBytes = shippingRigidContactRuntimeBytes(input);
    // Case032 predates the separate final-candidate constrained-node contact
    // force channel. Account for its exact new private allocation separately
    // rather than rewriting that historical measurement.
    result.constraintContactForceReadbackBytes = shippingMemoryCheckedMultiply3(
        environments, input.nodeCount, sizeof(nm_float4));
    result.measuredBaseAndContactEnvelopeBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(result.measuredNoncontactRuntimeEnvelopeBytes,
                                 persistentContact),
        shippingMemoryCheckedAdd(result.rigidContactRuntimeBytes,
                                 result.constraintContactForceReadbackBytes));
    const std::uint64_t faces = input.cohesiveFaceCount;
    // Accepted/default/candidate/checkpoint face records, force/operator
    // vectors, and an extra per-face allowance for activation/coupling tables.
    constexpr std::uint64_t cohesivePerFaceBytes =
        4u * sizeof(NMCohesiveFaceGPU) +
        2u * sizeof(NMCohesiveFaceVectorGPU) + 128u;
    result.cohesiveRuntimeBytes = shippingMemoryCheckedMultiply3(
        environments, faces, cohesivePerFaceBytes);

    if (input.regionalGalerkinEnabled) {
        // Selectors 3 and 4 may append global plate-bending trial fields and the
        // runtime conservatively admits 32 coarse modes before rank reduction.
        const std::uint64_t modes = input.regionalGalerkinBlockFineEnabled
            ? 32u
            : shippingMemoryCheckedMultiply(
                input.compactPreconditionerRegionCount, 6u);
        const std::uint64_t rapScalarCount = shippingMemoryCheckedAdd(
            shippingMemoryCheckedAdd(
                shippingMemoryCheckedMultiply(input.nodeCount, modes),
                shippingMemoryCheckedMultiply3(2u, environments,
                    shippingMemoryCheckedMultiply(input.nodeCount, modes))),
            shippingMemoryCheckedAdd(
                shippingMemoryCheckedMultiply3(environments, modes, modes),
                shippingMemoryCheckedAdd(
                    shippingMemoryCheckedMultiply3(4u, environments, modes),
                    environments)));
        const std::uint64_t regionalRAPBytes = shippingMemoryCheckedMultiply(
            rapScalarCount, sizeof(nm_float4));
        result.regionalGalerkinBlockFineBytes =
            input.regionalGalerkinBlockFineEnabled
                ? shippingMemoryCheckedMultiply3(
                    3u, environments,
                    shippingMemoryCheckedMultiply(input.nodeCount,
                                                   sizeof(nm_float4)))
                : 0u;
        result.regionalGalerkinBytes = shippingMemoryCheckedAdd(
            regionalRAPBytes, result.regionalGalerkinBlockFineBytes);
        if (result.regionalGalerkinBytes > 256ull * 1024ull * 1024ull)
            throw std::length_error(
                "shipping regional Galerkin arena exceeds the runtime 256 MiB admission cap");
        if (input.regionalGalerkinDiagnosticsCapture) {
            const std::uint64_t basisReadback = shippingMemoryCheckedMultiply(
                shippingMemoryCheckedMultiply(input.nodeCount, modes),
                sizeof(nm_float4));
            const std::uint64_t effectiveAndOperatorReadback =
                shippingMemoryCheckedMultiply3(
                    2u, environments,
                    shippingMemoryCheckedMultiply(
                        shippingMemoryCheckedMultiply(input.nodeCount, modes),
                        sizeof(nm_float4)));
            const std::uint64_t nodeReadback = shippingMemoryCheckedMultiply3(
                environments, input.nodeCount, sizeof(NMFEMNodeStateGPU));
            const std::uint64_t coarseReadback = shippingMemoryCheckedMultiply3(
                environments, shippingMemoryCheckedMultiply(modes, modes),
                sizeof(float));
            const std::uint64_t pivotAndFlagReadback = shippingMemoryCheckedAdd(
                shippingMemoryCheckedMultiply3(
                    environments, modes, sizeof(std::uint32_t)),
                shippingMemoryCheckedMultiply(environments,
                                               sizeof(std::uint32_t)));
            result.regionalGalerkinDiagnosticsBytes = shippingMemoryCheckedAdd(
                shippingMemoryCheckedAdd(basisReadback,
                    effectiveAndOperatorReadback),
                shippingMemoryCheckedAdd(nodeReadback,
                    shippingMemoryCheckedAdd(coarseReadback,
                                             pivotAndFlagReadback)));
            result.regionalGalerkinDiagnosticsBytes = shippingMemoryCheckedAdd(
                result.regionalGalerkinDiagnosticsBytes,
                result.regionalGalerkinBlockFineBytes);
            if (result.regionalGalerkinDiagnosticsBytes >
                    64ull * 1024ull * 1024ull)
                throw std::length_error(
                    "shipping regional Galerkin diagnostics exceed the runtime 64 MiB snapshot cap");
        }
    }

    const std::uint64_t femPrimitiveCount = input.compactImmutableSurfaceMap
        ? input.surfaceFaceCount
        : shippingMemoryCheckedMultiply(input.tetrahedronCount, 4u);
    const std::uint64_t primitiveCount = shippingMemoryCheckedAdd(
        femPrimitiveCount, input.gridNodeCount);
    if (primitiveCount > (std::uint64_t{1u} << 63u))
        throw std::overflow_error(
            "shipping contact BVH primitive count exceeds bit-ceil range");
    if (input.rigidContactArmijoEnabled ||
        input.fullContactArmijoRequired) {
        constexpr std::uint64_t boxArmijoMaximumBytes =
            numi::matter::detail::kSparsePrivateContactScratchBudgetBytes;
        std::uint64_t bytes = 0u;
        bool admitted = false;
        if (input.sparsePrivateContactRequested) {
            const auto sparse = numi::matter::detail::admitSparsePrivateContact(
                asU32(environments, "environment count"),
                asU32(input.contactPairCount, "contact pair count"),
                asU32(input.sparsePrivateContactCapacity,
                      "sparse contact candidate capacity"),
                input.sparsePrivateContactParity,
                input.rigidGeneralizedCapacity != 0u ||
                    input.rigidQCapacity != 0u,
                input.sparsePrivateContactBoundedFreeBoxScope);
            bytes = sparse.totalBytes;
            admitted = sparse.admitted();
            if (!admitted)
                result.privateRigidContactFailure =
                    numi::matter::detail::sparsePrivateContactAdmissionErrorText(sparse.error);
        } else {
            const std::uint64_t samples = shippingMemoryCheckedMultiply3(
                environments, input.contactPairCount,
                sizeof(NMContactSampleGPU));
            const bool hasRigidRows = input.rigidGeneralizedCapacity != 0u;
            const std::uint64_t activePairIndexBytes = hasRigidRows
                ? shippingMemoryCheckedAdd(
                    shippingMemoryCheckedMultiply3(
                        environments, input.activeContactCapacity,
                        sizeof(std::uint32_t)),
                    shippingMemoryCheckedMultiply3(
                        environments, input.contactPairCount,
                        sizeof(std::uint32_t)))
                : 0u;
            const std::uint64_t activeCounts = hasRigidRows
                ? shippingMemoryCheckedMultiply(
                    environments, sizeof(std::uint32_t)) : 0u;
            bytes = shippingMemoryCheckedAdd(
                shippingMemoryCheckedAdd(samples, activePairIndexBytes),
                activeCounts);
            admitted = environments <= 4u &&
                numi::matter::detail::densePrivateContactPairDomainSupported(
                    asU32(input.contactPairCount, "contact pair count")) &&
                input.rigidProxyCount >= 2u && input.rigidProxyCount <= 16u &&
                samples <= boxArmijoMaximumBytes;
            if (!admitted)
                result.privateRigidContactFailure =
                    "dense private pair domain or sample arena is outside runtime admission";
        }
        admitted = admitted && input.rigidProxyCount >= 2u &&
            input.rigidProxyCount <= 16u;
        if (!admitted && result.privateRigidContactFailure.empty())
            result.privateRigidContactFailure =
                "rigid proxy count is outside runtime admission";
        if (admitted) result.privateRigidContactFailure.clear();
        result.privateRigidContactAdmitted = admitted;
        result.sparsePrivateContactBytes = input.sparsePrivateContactRequested
            ? bytes : 0u;
        if (input.rigidContactArmijoEnabled)
            result.rigidContactArmijoBytes = bytes;
        if (input.fullContactArmijoRequired)
            result.requiredRigidContactArmijoBytes = bytes;
        if (!admitted && input.fullContactArmijoRequired) {
            result.requiredResidualDescentArenaAdmitted = false;
            result.requiredResidualDescentArenaFailure =
                result.privateRigidContactFailure.empty()
                    ? "rigid-contact private arena exceeds runtime admission"
                    : result.privateRigidContactFailure;
        }
        if (!admitted && input.rigidContactArmijoEnabled)
            throw std::length_error(
                "shipping rigid-contact Armijo arena is outside runtime admission caps");
    }
    std::uint64_t deformableArmijoBytes = 0u;
    if (input.fullContactArmijoEnabled ||
        input.fullContactArmijoRequired) {
        const std::uint64_t leafCount = primitiveCount == 0u ? 0u :
            static_cast<std::uint64_t>(std::bit_ceil(primitiveCount));
        const std::uint64_t privateSurfaceBytes =
            input.fullContactArmijoAliasesSurfaceScratch ? 0u :
                shippingMemoryCheckedAdd(
                    shippingMemoryCheckedMultiply3(
                        environments, primitiveCount,
                        sizeof(NMContinuumSurfacePrimitiveGPU) +
                            4u * sizeof(std::uint32_t)),
                    shippingMemoryCheckedMultiply3(
                        environments, leafCount,
                        4u * sizeof(nm_float4)));
        constexpr std::uint64_t perDeformableCandidateBytes =
            sizeof(NMDeformableContactCandidateGPU) +
            sizeof(NMDeformableContactGPU) +
            2u * sizeof(std::uint32_t) + 6u * sizeof(std::uint32_t);
        const std::uint64_t candidateBytes = shippingMemoryCheckedMultiply3(
            environments, input.deformableContactCapacity,
            perDeformableCandidateBytes);
        const std::uint64_t nodeRangeBytes = shippingMemoryCheckedMultiply3(
            environments,
            shippingMemoryCheckedAdd(input.nodeCount, input.gridNodeCount),
            sizeof(NMIncidenceRangeGPU));
        const std::uint64_t perEnvironmentBytes = shippingMemoryCheckedAdd(
            shippingMemoryCheckedAdd(candidateBytes, nodeRangeBytes),
            shippingMemoryCheckedMultiply(
                environments, 4u * sizeof(std::uint32_t) +
                    sizeof(NMIndirectDispatchGPU)));
        deformableArmijoBytes = shippingMemoryCheckedAdd(
            privateSurfaceBytes, perEnvironmentBytes);
        const bool countsSupported = environments <= 4u &&
            input.tetrahedronCount <= 32768u &&
            input.nodeCount <= kShippingFullContactArmijoNodeLimit &&
            input.deformableContactCapacity <= 131072u &&
            primitiveCount <= std::numeric_limits<std::uint32_t>::max() &&
            leafCount <= std::numeric_limits<std::uint32_t>::max() / 4u;
        const bool admitted = countsSupported &&
            deformableArmijoBytes <= 128ull * 1024ull * 1024ull &&
            input.environmentCount <= 4u && input.tetrahedronCount <= 32768u &&
            input.nodeCount <= kShippingFullContactArmijoNodeLimit &&
            input.deformableContactCapacity <= 131072u;
        if (input.fullContactArmijoEnabled)
            result.fullContactArmijoBytes = deformableArmijoBytes;
        if (input.fullContactArmijoRequired)
            result.requiredFullContactArmijoBytes = deformableArmijoBytes;
        if (!admitted && input.fullContactArmijoRequired) {
            result.requiredResidualDescentArenaAdmitted = false;
            if (result.requiredResidualDescentArenaFailure.empty())
                result.requiredResidualDescentArenaFailure =
                    "deformable private arena exceeds runtime node/tet/contact/128 MiB caps";
        }
        if (!admitted && input.fullContactArmijoEnabled)
            throw std::length_error(
                "shipping full-contact Armijo arena is outside runtime admission caps");
        if (input.fullContactArmijoRequired &&
            !input.deformableSelfContactEnabled) {
            result.requiredResidualDescentArenaAdmitted = false;
            if (result.requiredResidualDescentArenaFailure.empty())
                result.requiredResidualDescentArenaFailure =
                    "full-contact Armijo requires deformable self-contact enabled on the source FEM object";
        }
    }
    if (input.cohesiveArmijoEnabled || input.fullContactArmijoRequired) {
        const std::uint64_t arenaBytes = shippingMemoryCheckedMultiply3(
            environments, faces,
            sizeof(NMCohesiveFaceGPU) + sizeof(NMCohesiveFaceVectorGPU));
        std::uint64_t bytes =
            input.fullContactArmijoAliasesCohesiveScratch ? 0u : arenaBytes;
        constexpr std::uint64_t maximumCohesiveArmijoBytes =
            8ull * 1024ull * 1024ull;
        bool admitted = faces > 0u && faces <= 4096u &&
            environments <= 4u && arenaBytes <= maximumCohesiveArmijoBytes;
        if (input.fullContactArmijoTiledCohesive) {
            if (input.fullContactArmijoAliasesCohesiveScratch)
                throw std::invalid_argument(
                    "tiled cohesive trial pages cannot alias accepted cohesive scratch");
            numi::matter::cardboard::cohesive_tiles::Layout layout;
            admitted = faces <= std::numeric_limits<std::uint32_t>::max() &&
                environments <= std::numeric_limits<std::uint32_t>::max() &&
                numi::matter::cardboard::cohesive_tiles::makeLayout(
                    static_cast<std::uint32_t>(environments),
                    static_cast<std::uint32_t>(faces),
                    input.cohesiveTrialPageCapacity,
                    maximumCohesiveArmijoBytes, layout);
            // This is the modeled private force page only. The persistent
            // cohesive faces/forces are included in cohesiveRuntimeBytes.
            // Runtime still checks the actual Metal device and allocation.
            if (admitted) bytes = layout.pageBytes;
        } else if (input.cohesiveTrialPageCapacity != 0u) {
            throw std::invalid_argument(
                "cohesive trial page capacity requires tiled evaluation");
        }
        if (input.cohesiveArmijoEnabled)
            result.cohesiveArmijoBytes = bytes;
        if (input.fullContactArmijoRequired)
            result.requiredCohesiveArmijoBytes = bytes;
        if (!admitted && input.fullContactArmijoRequired) {
            result.requiredResidualDescentArenaAdmitted = false;
            if (result.requiredResidualDescentArenaFailure.empty())
                result.requiredResidualDescentArenaFailure =
                    input.fullContactArmijoTiledCohesive
                        ? "tiled cohesive private page exceeds checked face/dispatch/8 MiB limits"
                        : "cohesive private arena requires 1..4096 faces and at most 8 MiB";
        }
        if (!admitted && input.cohesiveArmijoEnabled)
            throw std::length_error(
                "shipping cohesive Armijo arena is outside runtime admission caps");
    }

    result.uploadStagingBytes = input.uploadStagingBytes;
    result.snapshotReadbackBytes = input.estimatedSnapshotPayloadBytes;
    result.estimatedMetalPeakBytes = result.measuredBaseAndContactEnvelopeBytes;
    for (const std::uint64_t category : {
             result.cohesiveRuntimeBytes, result.regionalGalerkinBytes,
             result.rigidContactArmijoBytes,
             std::max(result.fullContactArmijoBytes,
                      result.requiredFullContactArmijoBytes),
             std::max(result.rigidContactArmijoBytes,
                      result.requiredRigidContactArmijoBytes),
             std::max(result.cohesiveArmijoBytes,
                      result.requiredCohesiveArmijoBytes),
             result.regionalGalerkinDiagnosticsBytes,
             result.uploadStagingBytes, result.snapshotReadbackBytes})
        result.estimatedMetalPeakBytes = shippingMemoryCheckedAdd(
            result.estimatedMetalPeakBytes, category);

    result.hostMeshAndCompileBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(input.hostSourceMeshBytes,
                                 input.hostCompiledWorldBytes),
        input.hostGeneratedMetalBytes);
    result.hostCheckpointBytes = shippingMemoryCheckedMultiply(
        input.estimatedSnapshotPayloadBytes,
        input.retainedCompleteCheckpointCount);
    result.acceptedInsideClearanceWorkspaceRequired =
        input.acceptedInsideClearanceWorkspaceRequired;
    result.acceptedInsideClearanceWorkspaceAdmitted =
        input.acceptedInsideClearanceWorkspaceAdmitted;
    result.acceptedInsideClearanceWorkspaceFailure =
        input.acceptedInsideClearanceWorkspaceFailure;
    if (input.acceptedInsideClearanceWorkspaceRequired &&
        input.acceptedInsideClearanceWorkspaceAdmitted) {
        if (input.acceptedInsideClearanceWorkspaceBytes == 0u ||
            input.acceptedInsideClearanceWorkspaceBytes >
                inside_clearance_detail::maximumWorkspaceBytes) {
            result.acceptedInsideClearanceWorkspaceAdmitted = false;
            result.acceptedInsideClearanceWorkspaceFailure =
                "accepted inside-clearance workspace is outside its checked 1 GiB cap";
        } else {
            result.acceptedInsideClearanceWorkspaceBytes =
                input.acceptedInsideClearanceWorkspaceBytes;
        }
    }
    result.estimatedHostPeakBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(
            shippingMemoryCheckedAdd(result.hostMeshAndCompileBytes,
                                     result.hostCheckpointBytes),
            result.acceptedInsideClearanceWorkspaceBytes),
        shippingMemoryCheckedAdd(result.uploadStagingBytes,
                                 result.snapshotReadbackBytes));
    // This is a modeled allocation inventory, not a comparison of observed
    // unified-memory counters. The Metal envelope and the host mesh/compile
    // objects/checkpoints have distinct allocation lifetimes and are summed.
    // Upload staging and one snapshot readback reserve are present in both
    // subreports but counted once through the Metal envelope.
    result.estimatedPeakDomainBytes =
        shippingCombinedAllocationPeakBytes(result);
    result.withinLimit = result.estimatedPeakDomainBytes <= result.limitBytes &&
        (!input.fullContactArmijoRequired ||
         result.requiredResidualDescentArenaAdmitted) &&
        (!input.acceptedInsideClearanceWorkspaceRequired ||
         result.acceptedInsideClearanceWorkspaceAdmitted);
    return result;
}

inline std::uint64_t estimateShippingSnapshotPayloadBytes(
    const ShippingMemoryBudgetInput& input) {
    const std::uint64_t environments = input.environmentCount;
    if (environments == 0u || input.materialStateStride > 4096u)
        throw std::invalid_argument("shipping snapshot estimate needs an environment");
    std::uint64_t perEnvironment = 0u;
    const auto addProduct = [&](const std::uint64_t count,
                                const std::uint64_t stride) {
        perEnvironment = shippingMemoryCheckedAdd(
            perEnvironment, shippingMemoryCheckedMultiply(count, stride));
    };
    addProduct(input.nodeCount,
        sizeof(NMFEMNodeStateGPU) + sizeof(NMFEMFieldStateGPU) +
        sizeof(NMFEMTopologyNodeGPU) + 4u * sizeof(nm_float4));
    addProduct(input.tetrahedronCount,
        sizeof(NMTetrahedronGPU) + sizeof(NMTetrahedronGPU) +
        shippingMemoryCheckedMultiply(input.materialStateStride, sizeof(float)) +
        sizeof(NMFEMElementVectorGPU));
    addProduct(input.cohesiveFaceCount,
        sizeof(NMCohesiveFaceGPU) + sizeof(NMCohesiveFaceVectorGPU));
    addProduct(input.deformableContactCapacity,
        sizeof(NMDeformableContactHistoryGPU) + sizeof(std::uint32_t));
    addProduct(input.contactPairCount,
        2u * sizeof(nm_float4) + sizeof(NMContactSampleGPU));
    addProduct(input.rigidProxyCount,
        sizeof(NMRigidStateGPU) + sizeof(NMRigidReactionGPU));
    addProduct(input.rigidGeneralizedCapacity,
        2u * sizeof(float));
    addProduct(1u,
        sizeof(NMMatterStatusGPU) + sizeof(NMSolverCertificateGPU));
    return shippingMemoryCheckedMultiply(perEnvironment, environments);
}

inline void enforceShippingMemoryBudget(const ShippingMemoryBudget& budget) {
    if (budget.limitBytes == 0u || budget.estimatedPeakDomainBytes == 0u)
        throw std::invalid_argument("shipping memory budget is incomplete");
    if (!budget.withinLimit || budget.estimatedPeakDomainBytes > budget.limitBytes)
        throw std::runtime_error(
            "shipping estimated peak memory exceeds the configured pre-allocation limit");
}

// Screen grossly oversized shipping requests before constructing the 3D
// dieline. The reference cardinalities are from the exact CPU compile-only
// shipping blank: 300x200x200 mm inside, 25 mm lap, 1 mm kerf, 7.9 mm pitch,
// 130 clipped pitch intervals, 48 y intervals (16 in each of three bands),
// and one thickness slice (453,378 nodes and 1,477,728 tetrahedra). Counts
// scale with clipped flute intervals, y intervals and thickness slices,
// then receive a 25% planning margin. This remains an estimate, not exact
// geometry or native memory admission; post-build counts and initialized
// runtime observations remain mandatory.
inline ShippingPrebuildMemoryScreen screenShippingPrebuildMemory(
    const double perimeterM,
    const double pitchM,
    const std::uint32_t environmentCount = 1u,
    const std::uint32_t restart = kShippingFgmresRestartCeiling,
    const std::uint32_t yIntervals = kShippingReferenceYIntervals,
    const std::uint32_t thicknessSlices = 1u) {
    if (!(std::isfinite(perimeterM) && perimeterM > 0.0 &&
          std::isfinite(pitchM) && pitchM > 0.0) ||
        yIntervals == 0u || yIntervals > kShippingYIntervalCeiling ||
        thicknessSlices == 0u || thicknessSlices > 4u)
        throw std::invalid_argument(
            "shipping prebuild geometry and discretization must be in range");
    constexpr double referencePerimeterM = 1.025;
    constexpr double referencePitchM = 0.0079;
    ShippingPrebuildMemoryScreen result;
    result.referencePitchIntervals = static_cast<std::uint64_t>(
        std::ceil(referencePerimeterM / referencePitchM));
    result.requestedPitchIntervals = static_cast<std::uint64_t>(
        std::ceil(perimeterM / pitchM));
    result.yIntervals = yIntervals;
    result.thicknessSlices = thicknessSlices;
    if (result.requestedPitchIntervals == 0u ||
        result.referencePitchIntervals == 0u)
        throw std::invalid_argument("shipping prebuild pitch interval count is invalid");
    const long double longitudinalScale =
        static_cast<long double>(result.requestedPitchIntervals) /
        static_cast<long double>(result.referencePitchIntervals);
    const long double nodeScale = longitudinalScale *
        static_cast<long double>(yIntervals + 1u) /
            static_cast<long double>(kShippingReferenceYIntervals + 1u) *
        static_cast<long double>(thicknessSlices);
    const long double tetScale = longitudinalScale *
        static_cast<long double>(yIntervals) /
            static_cast<long double>(kShippingReferenceYIntervals) *
        static_cast<long double>(thicknessSlices);
    const auto scaledCeiling = [&](const std::uint64_t reference,
                                   const long double scale,
                                   const long double margin) {
        const long double estimate = static_cast<long double>(reference) * scale * margin;
        if (!std::isfinite(static_cast<double>(estimate)) ||
            estimate > static_cast<long double>(
                std::numeric_limits<std::uint64_t>::max()))
            throw std::overflow_error("shipping prebuild mesh estimate overflow");
        return static_cast<std::uint64_t>(std::ceil(estimate));
    };
    result.nominalNodeCount = scaledCeiling(kShippingReferenceNodes, nodeScale, 1.0L);
    result.nominalTetrahedronCount = scaledCeiling(
        kShippingReferenceTetrahedra, tetScale, 1.0L);
    result.estimatedNodeCount = scaledCeiling(
        kShippingReferenceNodes, nodeScale,
        static_cast<long double>(result.cardinalityMargin));
    result.estimatedTetrahedronCount = scaledCeiling(
        kShippingReferenceTetrahedra, tetScale,
        static_cast<long double>(result.cardinalityMargin));
    result.staticArenaScreen = estimateShippingStaticMemoryLowerBound(
        result.estimatedNodeCount, result.estimatedTetrahedronCount,
        environmentCount, restart);
    return result;
}

struct ShippingPrebuildArenaOptions {
    // Zero selects the native default of eight deformable-contact slots per
    // tetrahedron. A positive value is an explicit allocation cap only; the
    // runtime retains overflow rejection/rollback semantics.
    std::uint64_t deformableContactCapacity = 0u;
    std::uint64_t cohesiveFaceUpperBound = 0u;
    std::uint64_t rigidProxyCount = 0u;
    std::uint64_t rigidBodyCount = 0u;
    std::uint64_t rigidGeneralizedCapacity = 6u;
    std::uint64_t fgmresRestart = kShippingFgmresRestartCeiling;
    bool compactImmutableSurfaceMap = false;
    bool includeAcceptedInsideClearanceWorkspace = false;
    bool requireFullContactResidualDescent = false;
    bool deformableSelfContactEnabled = false;
    bool fullContactArmijoTiledCohesive = false;
    std::uint32_t cohesiveTrialPageCapacity = 0u;
    bool sparsePrivateContactRequested = false;
    bool sparsePrivateContactParity = false;
    std::uint64_t sparsePrivateContactCapacity = 0u;
    // This describes only the known shipping source's intended proxy recipe
    // at the prebuild stage. Exact compiled proxies must still pass
    // shippingSparsePrivateBoundedFreeBoxLayout before runtime dispatch.
    bool sparsePrivateContactBoundedFreeBoxScope = false;
    std::string runtimeConfigurationFailure;
};

[[nodiscard]] inline bool shippingPrebuildBoundedFreeBoxScopeCandidate(
    const bool shippingBox,
    const bool nativeSceneExecution,
    const std::uint64_t rigidProxyCount,
    const std::uint64_t rigidBodyCount,
    const std::uint64_t rigidGeneralizedCapacity,
    const std::uint64_t expectedRigidProxyCount,
    const std::uint64_t expectedRigidBodyCount) noexcept {
    // This recognizes only the fixed source recipe that emits the shipping
    // scene. It is deliberately weaker than the compiled-world predicate:
    // the actual proxy records and body bridge are checked again after compile.
    return shippingBox && nativeSceneExecution &&
        rigidProxyCount == expectedRigidProxyCount &&
        rigidBodyCount == expectedRigidBodyCount &&
        rigidGeneralizedCapacity == 6u;
}

inline void configureShippingPrebuildSparsePrivateContact(
    ShippingPrebuildArenaOptions& options,
    const char* capacityText,
    const char* parityText,
    const bool fullContactWillBeSelected,
    const bool boundedFreeBoxScopeCandidate) {
    const auto sparse = parseShippingSparsePrivateContactOptions(
        capacityText, parityText);
    if (sparse.error != nullptr) {
        options.sparsePrivateContactRequested = false;
        options.sparsePrivateContactParity = false;
        options.sparsePrivateContactCapacity = 0u;
        options.sparsePrivateContactBoundedFreeBoxScope = false;
        if (options.runtimeConfigurationFailure.empty())
            options.runtimeConfigurationFailure = sparse.error;
        return;
    }
    options.sparsePrivateContactRequested = sparse.requested;
    options.sparsePrivateContactParity = sparse.parity;
    options.sparsePrivateContactCapacity = sparse.capacity;
    options.sparsePrivateContactBoundedFreeBoxScope =
        sparse.requested && boundedFreeBoxScopeCandidate;
    if (sparse.requested && !fullContactWillBeSelected &&
        options.runtimeConfigurationFailure.empty())
        options.runtimeConfigurationFailure =
            "sparse private contact capacity requires the full-contact Armijo selector";
}

inline ShippingMemoryBudget estimateShippingPrebuildMemoryBudget(
    const ShippingPrebuildMemoryScreen& screen,
    const ShippingPrebuildArenaOptions& options = {}) {
    if (screen.nominalNodeCount == 0u ||
        screen.nominalTetrahedronCount == 0u ||
        options.fgmresRestart == 0u ||
        options.fgmresRestart > kShippingFgmresRestartCeiling ||
        options.rigidProxyCount > std::numeric_limits<std::uint32_t>::max() ||
        options.rigidBodyCount > std::numeric_limits<std::uint32_t>::max() ||
        options.rigidGeneralizedCapacity > std::numeric_limits<std::uint32_t>::max() ||
        options.sparsePrivateContactCapacity >
            std::numeric_limits<std::uint32_t>::max() ||
        (options.sparsePrivateContactParity &&
         !options.sparsePrivateContactRequested))
        throw std::invalid_argument("shipping prebuild arena options are invalid");
    ShippingMemoryBudgetInput input;
    input.nodeCount = shippingPrebuildRigidNodeUpperBound(screen);
    input.tetrahedronCount = shippingPrebuildTetrahedronUpperBound(screen);
    input.cohesiveFaceCount = options.cohesiveFaceUpperBound;
    // Every authored proxy could pair with every contact-eligible FEM node.
    // Contact nodes are a subset, so this is a safe precompile upper bound.
    input.rigidProxyCount = options.rigidProxyCount;
    input.rigidBodyCount = options.rigidBodyCount;
    input.contactPairCount = shippingMemoryCheckedMultiply(
        input.nodeCount, input.rigidProxyCount);
    input.activeContactCapacity = input.contactPairCount;
    input.deformableContactCapacity = options.deformableContactCapacity == 0u
        ? shippingMemoryCheckedMultiply(input.tetrahedronCount, 8u)
        : options.deformableContactCapacity;
    input.rigidGeneralizedCapacity = options.rigidGeneralizedCapacity;
    input.materialStateStride = 32u;
    input.compactPreconditionerRegionCount = 3u;
    input.environmentCount = 1u;
    input.fgmresRestart = options.fgmresRestart;
    input.compactImmutableSurfaceMap = options.compactImmutableSurfaceMap;
    input.acceptedInsideClearanceWorkspaceRequired =
        options.includeAcceptedInsideClearanceWorkspace;
    if (input.acceptedInsideClearanceWorkspaceRequired) {
        // Before compilation the surface count is unknown. Reserve the
        // diagnostic's full checked workspace ceiling; after compile, exact
        // tet/surface counts replace this bound. The map rejects before any
        // allocations if its actual workspace estimate exceeds this ceiling.
        input.acceptedInsideClearanceWorkspaceBytes =
            inside_clearance_detail::maximumWorkspaceBytes;
        if (input.tetrahedronCount >
            inside_clearance_detail::maximumIndexedTetrahedra) {
            input.acceptedInsideClearanceWorkspaceAdmitted = false;
            input.acceptedInsideClearanceWorkspaceFailure =
                "prebuild tetrahedron upper bound exceeds the accepted inside-clearance topology cap";
        }
    }
    input.fullContactArmijoRequired =
        options.requireFullContactResidualDescent;
    input.deformableSelfContactEnabled =
        options.deformableSelfContactEnabled;
    input.fullContactArmijoTiledCohesive =
        options.fullContactArmijoTiledCohesive;
    input.cohesiveTrialPageCapacity = options.cohesiveTrialPageCapacity;
    input.sparsePrivateContactRequested =
        options.sparsePrivateContactRequested;
    input.sparsePrivateContactParity = options.sparsePrivateContactParity;
    input.sparsePrivateContactCapacity = options.sparsePrivateContactCapacity;
    input.sparsePrivateContactBoundedFreeBoxScope =
        options.sparsePrivateContactBoundedFreeBoxScope;
    // A compact map cannot exceed the legacy all-four-faces-per-tet domain.
    // Use that bound before compile; exact compact counts replace it after.
    input.surfaceFaceCount = options.compactImmutableSurfaceMap
        ? shippingMemoryCheckedMultiply(input.tetrahedronCount, 4u) : 0u;
    const std::uint64_t nodeHost = shippingMemoryCheckedMultiply(
        input.nodeCount, 256u);
    const std::uint64_t tetHost = shippingMemoryCheckedMultiply(
        input.tetrahedronCount, 512u);
    input.hostSourceMeshBytes = shippingMemoryCheckedAdd(nodeHost, tetHost);
    input.hostCompiledWorldBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(input.nodeCount, 128u),
        shippingMemoryCheckedMultiply(input.tetrahedronCount, 256u));
    input.hostGeneratedMetalBytes = 64ull * 1024ull * 1024ull;
    input.uploadStagingBytes = input.hostCompiledWorldBytes;
    input.retainedCompleteCheckpointCount = 2u;
    input.estimatedSnapshotPayloadBytes =
        estimateShippingSnapshotPayloadBytes(input);
    ShippingMemoryBudget budget = estimateShippingMemoryBudget(input);
    if (!options.runtimeConfigurationFailure.empty()) {
        budget.runtimeConfigurationFailure =
            options.runtimeConfigurationFailure;
        budget.requiredResidualDescentArenaAdmitted = false;
        budget.requiredResidualDescentArenaFailure =
            options.runtimeConfigurationFailure;
        budget.withinLimit = false;
    }
    return budget;
}

struct ShippingRuntimeMemoryObservation {
    // RuntimeDiagnostics::residentBytes is the exact sum of Matter's Metal
    // buffers allocated during Runtime::initialize for this compiled world.
    std::uint64_t runtimeResidentBytes = 0u;
    std::uint64_t metalCurrentAllocatedBytes = 0u;
    std::uint64_t metalRecommendedWorkingSetBytes = 0u;
    // Optional host process resident/footprint observation. Zero means the
    // platform sampler was unavailable, not that host memory use was zero.
    std::uint64_t hostResidentBytes = 0u;
    bool hostResidentSampleAvailable = false;
};

inline std::uint64_t shippingObservedPeakBytes(
    const ShippingRuntimeMemoryObservation& observation) {
    return std::max({observation.runtimeResidentBytes,
                     observation.metalCurrentAllocatedBytes,
                     observation.hostResidentSampleAvailable
                         ? observation.hostResidentBytes : 0u});
}

inline void enforceShippingPeakMemoryLimit(
    const ShippingRuntimeMemoryObservation& observation,
    const std::uint64_t limitBytes = kShippingPeakMemoryLimitBytes) {
    if (limitBytes == 0u)
        throw std::invalid_argument("shipping peak-memory limit must be positive");
    if (observation.runtimeResidentBytes == 0u ||
        observation.metalCurrentAllocatedBytes == 0u)
        throw std::invalid_argument(
            "shipping native memory gate requires initialized runtime and Metal allocation observations");
    if (observation.hostResidentSampleAvailable &&
        observation.hostResidentBytes == 0u)
        throw std::invalid_argument(
            "available host memory sampler returned a zero footprint");
    if (shippingObservedPeakBytes(observation) > limitBytes)
        throw std::runtime_error("shipping native memory observation exceeds the peak-memory limit");
}

} // namespace numi::cardboard
