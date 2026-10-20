#include "numi/matter/detail/full_contact_armijo_scratch_budget.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
unsigned checks = 0u;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t sum(const numi::matter::detail::FullContactArmijoScratchBytes& b) {
    return b.surfacePrimitives + b.surfaceBVHBounds + b.surfaceSortArrays +
        b.surfaceActiveCounts + b.surfaceCacheFlags + b.deformableCandidates + b.candidateCounts +
        b.candidateStatuses + b.deformableContacts + b.deformableActiveIndices +
        b.deformableActiveCounts + b.nodeIncidence + b.nodeRanges +
        b.activeOffsets + b.globalActiveIndices + b.activeDispatch;
}
}

int main() {
    using namespace numi::matter::detail;
    try {
        static_assert(sizeof(NMDeformableContactCandidateGPU) == 16u);
        static_assert(sizeof(NMDeformableContactGPU) == 432u);
        static_assert(sizeof(NMMatterStatusGPU) == 48u);
        static_assert(sizeof(NMIncidenceRangeGPU) == 16u);
        static_assert(sizeof(NMIndirectDispatchGPU) == 16u);

        FullContactArmijoScratchBytes fixture;
        require(estimateFullContactArmijoScratchBytes(
                    1u, 17u, 32u, 7u, 9u, false, fixture),
                "finite fixture estimate must succeed");
        require(fixture.surfacePrimitives == 17u * 64u &&
                fixture.surfaceBVHBounds == 4u * 32u * 16u &&
                fixture.surfaceSortArrays == 4u * 17u * 4u &&
                fixture.surfaceActiveCounts == 16u &&
                fixture.surfaceCacheFlags == 16u,
                "surface allocations must match exact runtime counts");
        require(fixture.deformableCandidates == 7u * 16u &&
                fixture.candidateCounts == 16u &&
                fixture.candidateStatuses == 7u * 48u &&
                fixture.deformableContacts == 7u * 432u &&
                fixture.deformableActiveIndices == 7u * 4u &&
                fixture.deformableActiveCounts == 16u &&
                fixture.nodeIncidence == 7u * 6u * 4u &&
                fixture.nodeRanges == 9u * 16u &&
                fixture.activeOffsets == 16u &&
                fixture.globalActiveIndices == 7u * 4u &&
                fixture.activeDispatch == 16u,
                "deformable allocation inventory must include the 48-byte status slot");
        require(fixture.total == sum(fixture),
                "reported total must equal the named actual allocation inventory");

        FullContactArmijoScratchBytes radixFixture;
        require(estimateFullContactArmijoScratchBytes(
                    1u, 1u, 1u, 1u, 1u, false, radixFixture, true),
                "tiny radix surface estimate must succeed");
        require(radixFixture.surfaceSortArrays == 3u * 16u + 16u * 4u &&
                    radixFixture.total == sum(radixFixture),
                "radix scratch stride must cover sixteen histogram words at P=1");

        FullContactArmijoScratchBytes empty;
        require(estimateFullContactArmijoScratchBytes(
                    1u, 0u, 0u, 0u, 0u, false, empty),
                "zero logical counts still have typed sentinel allocations");
        require(empty.surfacePrimitives == sizeof(NMContinuumSurfacePrimitiveGPU) &&
                empty.surfaceBVHBounds == sizeof(nm_float4) &&
                empty.surfaceSortArrays == 4u * 16u &&
                empty.surfaceActiveCounts == 16u &&
                empty.surfaceCacheFlags == 16u &&
                empty.candidateCounts == 16u &&
                empty.candidateStatuses == sizeof(NMMatterStatusGPU) &&
                empty.deformableContacts == sizeof(NMDeformableContactGPU) &&
                empty.deformableActiveIndices == 16u &&
                empty.deformableActiveCounts == 16u &&
                empty.nodeIncidence == 16u && empty.nodeRanges == 16u &&
                empty.activeOffsets == 16u &&
                empty.globalActiveIndices == 16u &&
                empty.activeDispatch == 16u && empty.total == sum(empty),
                "zero-count buffers must reconcile to checkedBytes sentinels");

        FullContactArmijoScratchBytes aliasOne;
        require(estimateFullContactArmijoScratchBytes(
                    1u, 0u, 0u, 131072u, 0u, true, aliasOne),
                "one-environment max-capacity plan must be representable");
        require(aliasOne.candidateStatuses == 131072ull * 48u &&
                aliasOne.surfacePrimitives == 0u &&
                aliasOne.surfaceBVHBounds == 0u &&
                aliasOne.surfaceSortArrays == 0u &&
                aliasOne.surfaceActiveCounts == 0u &&
                aliasOne.total == 131072ull * 528u + 3u * 16u + 16u + 16u + 16u &&
                aliasOne.total <= kFullContactArmijoScratchBudgetBytes,
                "E=1 boundary must include exact statuses and fit unchanged cap");

        FullContactArmijoScratchBytes aliasTwoFit;
        FullContactArmijoScratchBytes aliasTwoOver;
        require(estimateFullContactArmijoScratchBytes(
                    2u, 0u, 0u, 127100u, 0u, true, aliasTwoFit) &&
                aliasTwoFit.total == 2ull * 127100u * 528u +
                    3u * 16u + 16u + 16u + 16u &&
                aliasTwoFit.total <= kFullContactArmijoScratchBudgetBytes,
                "E=2 exact final fitting boundary must include one shared dispatch");
        require(estimateFullContactArmijoScratchBytes(
                    2u, 0u, 0u, 127101u, 0u, true, aliasTwoOver) &&
                aliasTwoOver.total == 134218752u &&
                aliasTwoOver.total > kFullContactArmijoScratchBudgetBytes,
                "E=2 next capacity must exceed unchanged 128 MiB cap");
        const std::uint64_t aliasTwoFirstOverBytes = aliasTwoOver.total;
        FullContactArmijoScratchBytes aliasTwoHardMax;
        require(estimateFullContactArmijoScratchBytes(
                    2u, 0u, 0u, 131072u, 0u, true, aliasTwoOver) &&
                aliasTwoOver.candidateStatuses == 2ull * 131072u * 48u &&
                aliasTwoOver.total > kFullContactArmijoScratchBudgetBytes,
                "E=2 admitted hard capacity must fail total arena cap without wrap");
        require(estimateFullContactArmijoScratchBytes(
                    2u, 0u, 0u, 131072u, 0u, true, aliasTwoHardMax),
                "E=2 hard-cap plan must be computable for exact report");

        FullContactArmijoScratchBytes overflow;
        require(!estimateFullContactArmijoScratchBytes(
                    2u, std::numeric_limits<std::uint64_t>::max(), 1u,
                    1u, 1u, false, overflow),
                "environment multiplication overflow must fail closed");
        require(!estimateFullContactArmijoScratchBytes(
                    0u, 1u, 1u, 1u, 1u, false, overflow),
                "zero environment domain must fail closed");

        FullContactArmijoScratchBytes shippingOneAliased;
        FullContactArmijoScratchBytes shippingOneUnaliased;
        FullContactArmijoScratchBytes shippingTwoAliased;
        constexpr std::uint64_t shippingPrimitives = 1054876u;
        constexpr std::uint64_t shippingLeaves = 2097152u;
        constexpr std::uint64_t shippingNodes = 528444u;
        require(estimateFullContactArmijoScratchBytes(
                    1u, shippingPrimitives, shippingLeaves, 131072u,
                    shippingNodes, true, shippingOneAliased) &&
                shippingOneAliased.total == 77661200u &&
                shippingOneAliased.total <= kFullContactArmijoScratchBudgetBytes,
                "shipping-count E1 aliased-surface scratch must reconcile within cap");
        require(estimateFullContactArmijoScratchBytes(
                    1u, shippingPrimitives, shippingLeaves, 131072u,
                    shippingNodes, false, shippingOneUnaliased) &&
                shippingOneUnaliased.total == 296269024u &&
                shippingOneUnaliased.total > kFullContactArmijoScratchBudgetBytes,
                "shipping-count E1 unaliased BVH must fail the existing scratch cap");
        require(estimateFullContactArmijoScratchBytes(
                    2u, shippingPrimitives, shippingLeaves, 131072u,
                    shippingNodes, true, shippingTwoAliased) &&
                shippingTwoAliased.total == 155322320u &&
                shippingTwoAliased.total > kFullContactArmijoScratchBudgetBytes,
                "shipping-count E2 aliased scratch must fail the unchanged cap");

        std::cout << "full_contact_armijo_scratch_checks=" << checks
                  << " status_bytes=" << sizeof(NMMatterStatusGPU)
                  << " E1_max_capacity_bytes=" << aliasOne.total
                  << " E2_boundary_fit_bytes=" << aliasTwoFit.total
                  << " E2_first_over_bytes=" << aliasTwoFirstOverBytes
                  << " E2_hard_capacity_bytes=" << aliasTwoHardMax.total
                  << " shipping_E1_alias_bytes=" << shippingOneAliased.total
                  << " shipping_E1_unalias_bytes=" << shippingOneUnaliased.total
                  << " shipping_E2_alias_bytes=" << shippingTwoAliased.total
                  << " cap_bytes=" << kFullContactArmijoScratchBudgetBytes
                  << " physical_transactions=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "full-contact scratch accounting check failed: "
                  << error.what() << '\n';
        return 1;
    }
}
