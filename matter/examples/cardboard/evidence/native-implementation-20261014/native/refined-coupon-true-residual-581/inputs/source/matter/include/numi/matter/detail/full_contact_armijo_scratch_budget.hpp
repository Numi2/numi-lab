#pragma once

#include "numi/matter/shared.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::matter::detail {

// Byte inventory for the additional private full-contact Armijo arena. Every
// field is one actual privateScratch allocation (or zero when the surface
// arrays alias production storage). Empty typed allocations retain the same
// complete-pointee sentinel as runtime::checkedBytes().
struct FullContactArmijoScratchBytes {
    std::uint64_t surfacePrimitives = 0u;
    std::uint64_t surfaceBVHBounds = 0u;
    std::uint64_t surfaceSortArrays = 0u;
    std::uint64_t surfaceActiveCounts = 0u;
    std::uint64_t deformableCandidates = 0u;
    std::uint64_t candidateCounts = 0u;
    std::uint64_t candidateStatuses = 0u;
    std::uint64_t deformableContacts = 0u;
    std::uint64_t deformableActiveIndices = 0u;
    std::uint64_t deformableActiveCounts = 0u;
    std::uint64_t nodeIncidence = 0u;
    std::uint64_t nodeRanges = 0u;
    std::uint64_t activeOffsets = 0u;
    std::uint64_t globalActiveIndices = 0u;
    std::uint64_t activeDispatch = 0u;
    std::uint64_t total = 0u;
};

namespace full_contact_armijo_scratch_detail {

inline bool checkedMultiply(const std::uint64_t a, const std::uint64_t b,
                            std::uint64_t& out) noexcept {
    if (a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a)
        return false;
    out = a * b;
    return true;
}

inline bool checkedAdd(const std::uint64_t a, const std::uint64_t b,
                       std::uint64_t& out) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

inline bool typedAllocationBytes(const std::uint64_t count,
                                 const std::size_t elementSize,
                                 std::uint64_t& out) noexcept {
    if (elementSize == 0u || count >
        std::numeric_limits<std::size_t>::max() / elementSize)
        return false;
    const std::size_t logical = static_cast<std::size_t>(count) * elementSize;
    const std::size_t bytes = std::max({logical, elementSize, std::size_t{16u}});
    if (bytes > std::numeric_limits<std::uint64_t>::max()) return false;
    out = static_cast<std::uint64_t>(bytes);
    return true;
}

inline bool addTyped(std::uint64_t count, std::size_t elementSize,
                     std::uint64_t& component,
                     std::uint64_t& total) noexcept {
    if (!typedAllocationBytes(count, elementSize, component)) return false;
    std::uint64_t next = 0u;
    if (!checkedAdd(total, component, next)) return false;
    total = next;
    return true;
}

inline bool addRepeatedTyped(std::uint64_t count, std::size_t elementSize,
                             std::uint64_t repetitions,
                             std::uint64_t& component,
                             std::uint64_t& total) noexcept {
    std::uint64_t one = 0u;
    if (!typedAllocationBytes(count, elementSize, one) ||
        !checkedMultiply(one, repetitions, component))
        return false;
    std::uint64_t next = 0u;
    if (!checkedAdd(total, component, next)) return false;
    total = next;
    return true;
}

} // namespace full_contact_armijo_scratch_detail

// Reconcile the estimator against the exact privateScratch call inventory in
// runtime.mm. Counts are total elements passed to each single allocation, so
// the environment multiplier is included before applying checkedBytes' typed
// sentinel rule. The one indirect-dispatch record is shared across envs.
inline bool estimateFullContactArmijoScratchBytes(
    const std::uint64_t environments,
    const std::uint64_t surfacePrimitiveCount,
    const std::uint64_t surfaceBVHLeafCount,
    const std::uint64_t deformableCapacity,
    const std::uint64_t nodeCount,
    const bool aliasesProductionSurfaceScratch,
    FullContactArmijoScratchBytes& out
) noexcept {
    using namespace full_contact_armijo_scratch_detail;
    out = {};
    if (environments == 0u) return false;

    std::uint64_t envPrimitives = 0u;
    std::uint64_t envCapacity = 0u;
    std::uint64_t envNodes = 0u;
    std::uint64_t envLeaves = 0u;
    std::uint64_t envSixCapacity = 0u;
    if (!checkedMultiply(environments, surfacePrimitiveCount, envPrimitives) ||
        !checkedMultiply(environments, deformableCapacity, envCapacity) ||
        !checkedMultiply(environments, nodeCount, envNodes) ||
        !checkedMultiply(environments, surfaceBVHLeafCount, envLeaves) ||
        !checkedMultiply(envCapacity, 6u, envSixCapacity))
        return false;

    if (!aliasesProductionSurfaceScratch) {
        std::uint64_t bvhCount = 0u;
        if (!checkedMultiply(envLeaves, 4u, bvhCount) ||
            !addTyped(envPrimitives, sizeof(NMContinuumSurfacePrimitiveGPU),
                      out.surfacePrimitives, out.total) ||
            !addTyped(bvhCount, sizeof(nm_float4), out.surfaceBVHBounds,
                      out.total) ||
            !addRepeatedTyped(envPrimitives, sizeof(std::uint32_t), 4u,
                              out.surfaceSortArrays, out.total) ||
            !addTyped(environments, sizeof(std::uint32_t),
                      out.surfaceActiveCounts, out.total))
            return false;
    }

    if (!addTyped(envCapacity, sizeof(NMDeformableContactCandidateGPU),
                  out.deformableCandidates, out.total) ||
        !addTyped(environments, sizeof(std::uint32_t), out.candidateCounts,
                  out.total) ||
        !addTyped(envCapacity, sizeof(NMMatterStatusGPU),
                  out.candidateStatuses, out.total) ||
        !addTyped(envCapacity, sizeof(NMDeformableContactGPU),
                  out.deformableContacts, out.total) ||
        !addTyped(envCapacity, sizeof(std::uint32_t),
                  out.deformableActiveIndices, out.total) ||
        !addTyped(environments, sizeof(std::uint32_t),
                  out.deformableActiveCounts, out.total) ||
        !addTyped(envSixCapacity, sizeof(std::uint32_t), out.nodeIncidence,
                  out.total) ||
        !addTyped(envNodes, sizeof(NMIncidenceRangeGPU), out.nodeRanges,
                  out.total) ||
        !addTyped(environments, sizeof(std::uint32_t), out.activeOffsets,
                  out.total) ||
        !addTyped(envCapacity, sizeof(std::uint32_t), out.globalActiveIndices,
                  out.total) ||
        !addTyped(1u, sizeof(NMIndirectDispatchGPU), out.activeDispatch,
                  out.total))
        return false;
    return true;
}

inline constexpr std::uint64_t kFullContactArmijoScratchBudgetBytes =
    128ull * 1024ull * 1024ull;

} // namespace numi::matter::detail
