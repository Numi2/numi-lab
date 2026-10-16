#include "numi/matter/detail/shipping_contact_page_plan.hpp"

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

bool completeOrdinalEnumeration(std::uint64_t total, std::uint64_t cap) {
    std::uint64_t expected = 0u;
    std::uint64_t pageCount = total == 0u || cap == 0u ? 0u :
        1u + (total - 1u) / cap;
    for (std::uint64_t page = 0; page < pageCount; ++page) {
        numi::matter::detail::ShippingContactPageRange range{};
        if (!numi::matter::detail::shippingContactPageRange(
                total, cap, page, range))
            return false;
        for (std::uint32_t local = 0; local < range.rowCount; ++local) {
            const std::uint64_t id =
                static_cast<std::uint64_t>(range.firstRow) + local;
            if (id != expected) return false;
            ++expected;
        }
    }
    numi::matter::detail::ShippingContactPageRange out{};
    return expected == total &&
        !numi::matter::detail::shippingContactPageRange(
            total, cap, pageCount, out);
}
}

int main() {
    using namespace numi::matter::detail;
    try {
        static_assert(sizeof(NMContactPairGPU) == 16u);
        static_assert(sizeof(NMContactSampleGPU) == 160u);
        static_assert(sizeof(NMCohesiveFaceGPU) == 384u);
        static_assert(sizeof(NMCohesiveFaceVectorGPU) == 96u);

        ShippingContactPagePlanInput shipping;
        shipping.environmentCount = 1u;
        shipping.contactPairCount = 6335568u;
        shipping.cohesiveFaceCount = 260835u;
        shipping.retainedGlobalUnknownBufferCount = 2u;
        shipping.retainedGlobalUnknownBytesPerBuffer = 287473536u;
        shipping.maxBufferLength = 1ull * 1024ull * 1024ull * 1024ull;
        const auto plan = makeShippingContactPagePlan(shipping);
        require(plan.admitted(), "authored shipping counts should fit bounded page scratch");
        require(plan.contactPageRows == 65536u &&
                plan.contactPageCount == 97u &&
                plan.contactLastPageRows == 44112u,
                "contact pages must cover the exact immutable pair-row domain");
        require(plan.contactSamplePageBytes == 10485760u &&
                plan.contactCandidateIdPageBytes == 262144u &&
                plan.contactSlotPageBytes == 262144u &&
                plan.contactPageScratchBytes == 11010064u,
                "contact page extents must match the declared GPU record layouts");
        require(plan.cohesivePageFaces == 87381u &&
                plan.cohesivePageCount == 3u &&
                plan.cohesiveLastPageFaces == 86073u &&
                plan.cohesiveForcePageBytes == 8388576u,
                "cohesive pages must cover all faces under the unchanged 8 MiB cap");
        require(plan.contactPairTableBytes == 101369088u &&
                plan.contactAcceptedHistoryBytes == 101369088u &&
                plan.cohesiveAcceptedFaceBytes == 100160640u &&
                plan.retainedGlobalUnknownBytes == 574947072u,
                "full immutable tables, histories and global Krylov vectors remain resident");
        require(plan.persistentBytes > plan.contactPairTableBytes &&
                plan.serializedPagePeakBytes >= plan.persistentBytes &&
                plan.conservativeOverlapPeakBytes ==
                    plan.persistentBytes + plan.contactPageScratchBytes +
                    plan.cohesivePageScratchBytes,
                "lifetime and conservative overlap totals must reconcile");
        require(completeOrdinalEnumeration(shipping.contactPairCount,
                                           plan.contactPageRows),
                "every canonical pair ID must be enumerated exactly once, with no tail drop");
        require(completeOrdinalEnumeration(shipping.cohesiveFaceCount,
                                           plan.cohesivePageFaces),
                "every canonical cohesive face ID must be enumerated exactly once");

        ShippingContactPagePlanInput fourEnv = shipping;
        fourEnv.environmentCount = 4u;
        fourEnv.maxBufferLength = 1ull * 1024ull * 1024ull * 1024ull;
        const auto plan4 = makeShippingContactPagePlan(fourEnv);
        require(plan4.admitted() && plan4.contactPageRows < 65536u &&
                plan4.contactPageScratchBytes <=
                    kShippingContactPageScratchBudgetBytes &&
                plan4.cohesiveForcePageBytes <=
                    kShippingCohesiveForcePageBufferBytes,
                "multi-environment page sizing must reduce capacities to fit unchanged byte budgets");
        require(completeOrdinalEnumeration(fourEnv.contactPairCount,
                                           plan4.contactPageRows) &&
                completeOrdinalEnumeration(fourEnv.cohesiveFaceCount,
                                           plan4.cohesivePageFaces),
                "multi-environment page plan must retain complete ID coverage");

        const std::uint64_t bytesPerRow4 =
            4u * (sizeof(NMContactSampleGPU) + 2u * sizeof(std::uint32_t));
        require(plan4.contactPageScratchBytes <=
                    kShippingContactPageScratchBudgetBytes &&
                plan4.contactPageScratchBytes + bytesPerRow4 >
                    kShippingContactPageScratchBudgetBytes,
                "four-environment contact page must use the maximal fitting row count");

        ShippingContactPagePlanInput noCohesive = shipping;
        noCohesive.cohesiveFaceCount = 0u;
        const auto noCohesivePlan = makeShippingContactPagePlan(noCohesive);
        require(noCohesivePlan.admitted() &&
                noCohesivePlan.cohesivePageCount == 0u &&
                noCohesivePlan.cohesivePageScratchBytes == 0u,
                "an empty cohesive domain must not allocate or invent a page");

        ShippingContactPagePlanInput aggregateBoundary = shipping;
        aggregateBoundary.otherResidentBytes =
            kShippingModeledAggregateBudgetBytes -
            plan.conservativeOverlapPeakBytes;
        const auto aggregateFits =
            makeShippingContactPagePlan(aggregateBoundary);
        require(aggregateFits.admitted() &&
                aggregateFits.conservativeOverlapPeakBytes ==
                    kShippingModeledAggregateBudgetBytes,
                "the exact modeled aggregate boundary must fit");
        ++aggregateBoundary.otherResidentBytes;
        require(makeShippingContactPagePlan(aggregateBoundary).error ==
                    ShippingContactPagePlanError::modeled_aggregate_exceeded,
                "the first byte over the modeled aggregate cap must reject");

        ShippingContactPagePlanInput exactBuffer = shipping;
        exactBuffer.maxBufferLength = plan.contactPairTableBytes;
        const auto exactLimit = makeShippingContactPagePlan(exactBuffer);
        require(exactLimit.error ==
                    ShippingContactPagePlanError::buffer_exceeds_metal_limit,
                "maxBufferLength must be checked against every individual retained/page buffer");
        exactBuffer.maxBufferLength = exactBuffer.retainedGlobalUnknownBytesPerBuffer;
        const auto fitsLargest = makeShippingContactPagePlan(exactBuffer);
        require(fitsLargest.admitted(),
                "a maxBufferLength equal to the largest named buffer must pass");

        ShippingContactPagePlanInput tooManyRows = shipping;
        tooManyRows.contactPageRows = kShippingContactSamplePageRows + 1u;
        require(makeShippingContactPagePlan(tooManyRows).error ==
                    ShippingContactPagePlanError::contact_page_capacity_exceeds_authored_limit,
                "contact page capacity cannot silently exceed its authored limit");
        ShippingContactPagePlanInput tooManyFaces = shipping;
        tooManyFaces.cohesivePageFaces = 87382u;
        require(makeShippingContactPagePlan(tooManyFaces).error ==
                    ShippingContactPagePlanError::cohesive_page_capacity_exceeds_byte_limit,
                "cohesive page capacity cannot silently exceed its per-buffer byte cap");
        ShippingContactPagePlanInput tooManyEnv = shipping;
        tooManyEnv.environmentCount = 5u;
        require(makeShippingContactPagePlan(tooManyEnv).error ==
                    ShippingContactPagePlanError::unsupported_environment_count,
                "environment cap remains explicit");
        ShippingContactPagePlanInput pairOverflow = shipping;
        pairOverflow.contactPairCount =
            std::numeric_limits<std::uint32_t>::max();
        pairOverflow.environmentCount = 2u;
        require(makeShippingContactPagePlan(pairOverflow).error ==
                    ShippingContactPagePlanError::contact_domain_exceeds_u32,
                "flattened pair index overflow must reject before enumeration");
        ShippingContactPagePlanInput faceOverflow = shipping;
        faceOverflow.cohesiveFaceCount =
            std::numeric_limits<std::uint32_t>::max() / 6u + 1u;
        require(makeShippingContactPagePlan(faceOverflow).error ==
                    ShippingContactPagePlanError::cohesive_domain_exceeds_u32,
                "six-slot cohesive indexing overflow must reject");
        ShippingContactPagePlanInput multiplyOverflow = shipping;
        multiplyOverflow.contactPairCount =
            std::numeric_limits<std::uint64_t>::max();
        require(makeShippingContactPagePlan(multiplyOverflow).error ==
                    ShippingContactPagePlanError::contact_domain_exceeds_u32,
                "oversized pair domain must fail before byte multiplication");
        ShippingContactPagePlanInput aggregateOverflow = shipping;
        aggregateOverflow.otherResidentBytes =
            std::numeric_limits<std::uint64_t>::max();
        require(makeShippingContactPagePlan(aggregateOverflow).error ==
                    ShippingContactPagePlanError::arithmetic_overflow,
                "aggregate byte addition must fail closed");
        ShippingContactPageRange invalid{};
        require(!shippingContactPageRange(shipping.contactPairCount,
                                           plan.contactPageRows,
                                           plan.contactPageCount, invalid),
                "out-of-range page indices must reject");

        std::cout << "shipping_contact_page_plan_checks=" << checks
                  << " pairs=" << shipping.contactPairCount
                  << " pair_pages=" << plan.contactPageCount
                  << " pair_last=" << plan.contactLastPageRows
                  << " cohesive_faces=" << shipping.cohesiveFaceCount
                  << " cohesive_pages=" << plan.cohesivePageCount
                  << " cohesive_last=" << plan.cohesiveLastPageFaces
                  << " contact_page_bytes=" << plan.contactPageScratchBytes
                  << " cohesive_pages_bytes=" << plan.cohesivePageScratchBytes
                  << " persistent_subset_bytes=" << plan.persistentBytes
                  << " serialized_peak_bytes=" << plan.serializedPagePeakBytes
                  << " conservative_overlap_peak_bytes="
                  << plan.conservativeOverlapPeakBytes
                  << " full_pair_history_bytes="
                  << plan.contactPairTableBytes +
                     plan.contactAcceptedHistoryBytes +
                     plan.contactCandidateHistoryBytes +
                     plan.contactCheckpointHistoryBytes
                  << " whole_process_admission=false"
                  << " production_paging_enabled=false"
                  << " physical_transactions=0\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "shipping contact page plan check failed: "
                  << e.what() << '\n';
        return 1;
    }
}
