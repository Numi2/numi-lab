#pragma once

#include "numi/matter/shared.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::matter::detail {

// Authored workspace bounds for a prospective streaming implementation.
// These do not change runtime admission: source225 keeps the production
// contact/N/T gates unchanged until every consumer is page-aware and measured.
inline constexpr std::uint32_t kShippingContactSamplePageRows = 65536u;
inline constexpr std::uint64_t kShippingContactPageScratchBudgetBytes =
    36ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kShippingCohesiveForcePageBufferBytes =
    8ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kShippingModeledAggregateBudgetBytes =
    16ull * 1024ull * 1024ull * 1024ull;

enum class ShippingContactPagePlanError : std::uint8_t {
    none,
    zero_environment_count,
    unsupported_environment_count,
    zero_contact_domain,
    zero_cohesive_accumulator_node_domain,
    page_capacity_zero,
    contact_page_capacity_exceeds_authored_limit,
    cohesive_page_capacity_exceeds_byte_limit,
    contact_domain_exceeds_u32,
    cohesive_domain_exceeds_u32,
    arithmetic_overflow,
    buffer_exceeds_metal_limit,
    contact_page_scratch_exceeded,
    cohesive_page_scratch_exceeded,
    modeled_aggregate_exceeded,
};

struct ShippingContactPagePlanInput {
    std::uint64_t environmentCount = 1u;
    std::uint64_t contactPairCount = 0u;
    std::uint64_t cohesiveFaceCount = 0u;
    std::uint64_t femNodeCount = 0u;
    std::uint64_t contactPageRows = kShippingContactSamplePageRows;
    // Zero selects the maximum face count bounded by the existing 8 MiB
    // single-buffer rule. Nonzero is an explicit smaller authored limit.
    std::uint64_t cohesivePageFaces = 0u;
    std::uint64_t retainedGlobalUnknownBufferCount = 0u;
    std::uint64_t retainedGlobalUnknownBytesPerBuffer = 0u;
    std::uint64_t otherResidentBytes = 0u;
    std::uint64_t maxBufferLength = std::numeric_limits<std::uint64_t>::max();
};

struct ShippingContactPagePlan {
    ShippingContactPagePlanError error = ShippingContactPagePlanError::none;
    std::uint32_t environmentCount = 0u;
    std::uint32_t contactPageRows = 0u;
    std::uint32_t contactPageCount = 0u;
    std::uint32_t contactLastPageRows = 0u;
    std::uint32_t cohesivePageFaces = 0u;
    std::uint32_t cohesivePageCount = 0u;
    std::uint32_t cohesiveLastPageFaces = 0u;

    std::uint64_t contactPairTableBytes = 0u;
    std::uint64_t contactAcceptedHistoryBytes = 0u;
    std::uint64_t contactCandidateHistoryBytes = 0u;
    std::uint64_t contactCheckpointHistoryBytes = 0u;
    std::uint64_t cohesiveAcceptedFaceBytes = 0u;
    std::uint64_t cohesiveDefaultFaceBytes = 0u;
    std::uint64_t cohesiveCandidateFaceBytes = 0u;
    std::uint64_t cohesiveCheckpointFaceBytes = 0u;
    std::uint64_t cohesiveAccumulatorBytes = 0u;
    std::uint64_t retainedGlobalUnknownBytes = 0u;
    std::uint64_t otherResidentBytes = 0u;
    std::uint64_t persistentBytes = 0u;

    std::uint64_t contactSamplePageBytes = 0u;
    std::uint64_t contactCandidateIdPageBytes = 0u;
    std::uint64_t contactSlotPageBytes = 0u;
    std::uint64_t contactPageCountsBytes = 0u;
    std::uint64_t contactPageScratchBytes = 0u;

    std::uint64_t cohesiveForcePageBytes = 0u;
    std::uint64_t cohesiveOperatorPageBytes = 0u;
    std::uint64_t cohesivePageScratchBytes = 0u;

    std::uint64_t contactPhasePeakBytes = 0u;
    std::uint64_t cohesivePhasePeakBytes = 0u;
    std::uint64_t serializedPagePeakBytes = 0u;
    std::uint64_t conservativeOverlapPeakBytes = 0u;

    [[nodiscard]] constexpr bool admitted() const noexcept {
        return error == ShippingContactPagePlanError::none;
    }
};

namespace shipping_contact_page_detail {
inline constexpr std::uint64_t kContactHistoryCopies = 3u;
inline constexpr std::uint64_t kCohesiveFaceRecordCopies = 4u;
inline constexpr std::uint64_t kCohesiveForcePageCopies = 2u;

[[nodiscard]] constexpr bool multiply(std::uint64_t a, std::uint64_t b,
                                      std::uint64_t& out) noexcept {
    if (b != 0u && a > std::numeric_limits<std::uint64_t>::max() / b)
        return false;
    out = a * b;
    return true;
}
[[nodiscard]] constexpr bool add(std::uint64_t a, std::uint64_t b,
                                 std::uint64_t& out) noexcept {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) return false;
    out = a + b;
    return true;
}
[[nodiscard]] constexpr bool addTo(std::uint64_t value,
                                   std::uint64_t& total) noexcept {
    std::uint64_t next = 0u;
    if (!add(total, value, next)) return false;
    total = next;
    return true;
}
[[nodiscard]] constexpr bool ceilPages(std::uint64_t count,
                                       std::uint64_t capacity,
                                       std::uint64_t& out) noexcept {
    if (capacity == 0u) return false;
    out = count == 0u ? 0u : 1u + (count - 1u) / capacity;
    return true;
}
[[nodiscard]] constexpr bool bytes(std::uint64_t environments,
                                   std::uint64_t count,
                                   std::uint64_t stride,
                                   std::uint64_t& out) noexcept {
    std::uint64_t elements = 0u;
    return multiply(environments, count, elements) &&
        multiply(elements, stride, out) &&
        out <= std::numeric_limits<std::size_t>::max();
}
} // namespace shipping_contact_page_detail

// A checked range is a half-open interval in the canonical, immutable row
// domain. It carries the original global row indices; pages never renumber
// rows and never alter accepted history or the global unknown vectors.

// Checked sizing for the production cohesive-face force/operator tile. The
// face domain remains canonical and global; only the transient six-node
// vector output is paged. Nodal accumulation remains a separate global arena.
enum class ShippingCohesivePageError : std::uint8_t {
    none, zero_environment_count, zero_face_domain, zero_node_domain,
    face_domain_exceeds_u32, node_domain_exceeds_u32, page_capacity_zero,
    arithmetic_overflow, page_exceeds_metal_limit,
    accumulator_exceeds_metal_limit
};
struct ShippingCohesivePagePlanInput {
    std::uint64_t environmentCount = 1u;
    std::uint64_t cohesiveFaceCount = 0u;
    std::uint64_t femNodeCount = 0u;
    std::uint64_t pageBufferBudgetBytes = kShippingCohesiveForcePageBufferBytes;
    std::uint64_t maxBufferLength = std::numeric_limits<std::uint64_t>::max();
};
struct ShippingCohesivePagePlan {
    ShippingCohesivePageError error = ShippingCohesivePageError::none;
    std::uint32_t facesPerPage = 0u;
    std::uint32_t pageCount = 0u;
    std::uint32_t lastPageFaces = 0u;
    bool paged = false;
    std::uint64_t fullFaceBytes = 0u;
    std::uint64_t facePageBytes = 0u;
    std::uint64_t accumulatorBytes = 0u;
    std::uint64_t totalScratchBytes = 0u;
    [[nodiscard]] constexpr bool admitted() const noexcept {
        return error == ShippingCohesivePageError::none;
    }
};
[[nodiscard]] constexpr ShippingCohesivePagePlan makeShippingCohesivePagePlan(
    const ShippingCohesivePagePlanInput& in
) noexcept {
    using namespace shipping_contact_page_detail;
    ShippingCohesivePagePlan out;
    if (in.environmentCount == 0u) {
        out.error = ShippingCohesivePageError::zero_environment_count;
        return out;
    }
    if (in.cohesiveFaceCount == 0u) {
        out.error = ShippingCohesivePageError::zero_face_domain;
        return out;
    }
    if (in.femNodeCount == 0u) {
        out.error = ShippingCohesivePageError::zero_node_domain;
        return out;
    }
    if (in.environmentCount >
            std::numeric_limits<std::uint32_t>::max() ||
        in.cohesiveFaceCount >
            std::numeric_limits<std::uint32_t>::max() / in.environmentCount) {
        out.error = ShippingCohesivePageError::face_domain_exceeds_u32;
        return out;
    }
    if (in.femNodeCount >
            std::numeric_limits<std::uint32_t>::max() / in.environmentCount) {
        out.error = ShippingCohesivePageError::node_domain_exceeds_u32;
        return out;
    }
    std::uint64_t perFace = 0u;
    if (!multiply(in.environmentCount,
                  sizeof(NMCohesiveFaceVectorGPU), perFace) ||
        perFace == 0u) {
        out.error = ShippingCohesivePageError::arithmetic_overflow;
        return out;
    }
    const std::uint64_t pageBudget =
        std::min(in.pageBufferBudgetBytes, in.maxBufferLength);
    const std::uint64_t maxFaces = pageBudget / perFace;
    if (maxFaces == 0u) {
        out.error = ShippingCohesivePageError::page_capacity_zero;
        return out;
    }
    const std::uint64_t capacity =
        std::min(in.cohesiveFaceCount, maxFaces);
    out.paged = in.cohesiveFaceCount > capacity;
    std::uint64_t pages = 0u;
    if (!ceilPages(in.cohesiveFaceCount, capacity, pages) ||
        pages > std::numeric_limits<std::uint32_t>::max() ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
        out.error = ShippingCohesivePageError::arithmetic_overflow;
        return out;
    }
    std::uint64_t fullElements = 0u;
    std::uint64_t pageElements = 0u;
    std::uint64_t nodeElements = 0u;
    if (!multiply(in.environmentCount, in.cohesiveFaceCount, fullElements) ||
        !multiply(fullElements, sizeof(NMCohesiveFaceVectorGPU),
                  out.fullFaceBytes) ||
        !multiply(in.environmentCount, capacity, pageElements) ||
        !multiply(pageElements, sizeof(NMCohesiveFaceVectorGPU),
                  out.facePageBytes) ||
        !multiply(in.environmentCount, in.femNodeCount, nodeElements) ||
        !multiply(nodeElements, sizeof(nm_float4), out.accumulatorBytes)) {
        out.error = ShippingCohesivePageError::arithmetic_overflow;
        return out;
    }
    if (out.facePageBytes > in.maxBufferLength ||
        out.facePageBytes > in.pageBufferBudgetBytes) {
        out.error = ShippingCohesivePageError::page_exceeds_metal_limit;
        return out;
    }
    std::uint64_t twoPages = 0u;
    if (!multiply(out.facePageBytes, 2u, twoPages)) {
        out.error = ShippingCohesivePageError::arithmetic_overflow;
        return out;
    }
    if (out.paged) {
        if (out.accumulatorBytes > in.maxBufferLength) {
            out.error = ShippingCohesivePageError::accumulator_exceeds_metal_limit;
            return out;
        }
        if (!add(twoPages, out.accumulatorBytes, out.totalScratchBytes)) {
            out.error = ShippingCohesivePageError::arithmetic_overflow;
            return out;
        }
    } else {
        if (out.fullFaceBytes > in.maxBufferLength) {
            out.error = ShippingCohesivePageError::page_exceeds_metal_limit;
            return out;
        }
        out.totalScratchBytes = twoPages;
    }
    if (out.totalScratchBytes > std::numeric_limits<std::size_t>::max()) {
        out.error = ShippingCohesivePageError::arithmetic_overflow;
        return out;
    }
    out.facesPerPage = static_cast<std::uint32_t>(capacity);
    out.pageCount = static_cast<std::uint32_t>(pages);
    out.lastPageFaces = static_cast<std::uint32_t>(
        in.cohesiveFaceCount - (pages - 1u) * capacity);
    return out;
}

struct ShippingContactPageRange {
    std::uint32_t firstRow = 0u;
    std::uint32_t rowCount = 0u;
};
[[nodiscard]] constexpr bool shippingContactPageRange(
    const std::uint64_t totalRows,
    const std::uint64_t pageRows,
    const std::uint64_t pageIndex,
    ShippingContactPageRange& out
) noexcept {
    out = {};
    if (totalRows == 0u || pageRows == 0u ||
        totalRows > std::numeric_limits<std::uint32_t>::max() ||
        pageRows > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::uint64_t pageCount = 0u;
    if (!shipping_contact_page_detail::ceilPages(
            totalRows, pageRows, pageCount) || pageIndex >= pageCount)
        return false;
    std::uint64_t first = 0u;
    if (!shipping_contact_page_detail::multiply(pageIndex, pageRows, first) ||
        first >= totalRows || first > std::numeric_limits<std::uint32_t>::max())
        return false;
    const std::uint64_t count = std::min(pageRows, totalRows - first);
    if (count == 0u || count > std::numeric_limits<std::uint32_t>::max() ||
        first + count > totalRows)
        return false;
    out.firstRow = static_cast<std::uint32_t>(first);
    out.rowCount = static_cast<std::uint32_t>(count);
    return true;
}

[[nodiscard]] constexpr ShippingContactPagePlan makeShippingContactPagePlan(
    const ShippingContactPagePlanInput& in
) noexcept {
    using namespace shipping_contact_page_detail;
    ShippingContactPagePlan out;
    if (in.environmentCount == 0u) {
        out.error = ShippingContactPagePlanError::zero_environment_count;
        return out;
    }
    if (in.environmentCount > 4u) {
        out.error = ShippingContactPagePlanError::unsupported_environment_count;
        return out;
    }
    if (in.contactPairCount == 0u) {
        out.error = ShippingContactPagePlanError::zero_contact_domain;
        return out;
    }
    if (in.contactPageRows == 0u) {
        out.error = ShippingContactPagePlanError::page_capacity_zero;
        return out;
    }
    if (in.contactPageRows > kShippingContactSamplePageRows) {
        out.error = ShippingContactPagePlanError::contact_page_capacity_exceeds_authored_limit;
        return out;
    }
    if (in.contactPairCount >
        std::numeric_limits<std::uint32_t>::max() / in.environmentCount) {
        out.error = ShippingContactPagePlanError::contact_domain_exceeds_u32;
        return out;
    }
    if (in.cohesiveFaceCount != 0u && in.femNodeCount == 0u) {
        out.error = ShippingContactPagePlanError::zero_cohesive_accumulator_node_domain;
        return out;
    }
    if (in.femNodeCount >
            std::numeric_limits<std::uint32_t>::max() / in.environmentCount) {
        out.error = ShippingContactPagePlanError::cohesive_domain_exceeds_u32;
        return out;
    }
    if (in.cohesiveFaceCount >
            std::numeric_limits<std::uint32_t>::max() / 6u ||
        in.cohesiveFaceCount >
            std::numeric_limits<std::uint32_t>::max() / in.environmentCount) {
        out.error = ShippingContactPagePlanError::cohesive_domain_exceeds_u32;
        return out;
    }

    std::uint64_t cohesiveMaxFaces =
        kShippingCohesiveForcePageBufferBytes /
        (in.environmentCount * sizeof(NMCohesiveFaceVectorGPU));
    const std::uint64_t cohesivePageFaces =
        in.cohesiveFaceCount == 0u ? 0u :
        (in.cohesivePageFaces == 0u
            ? std::min(in.cohesiveFaceCount, cohesiveMaxFaces)
            : in.cohesivePageFaces);
    if (in.cohesiveFaceCount != 0u && cohesivePageFaces == 0u) {
        out.error = ShippingContactPagePlanError::page_capacity_zero;
        return out;
    }
    if (cohesivePageFaces > cohesiveMaxFaces) {
        out.error = ShippingContactPagePlanError::cohesive_page_capacity_exceeds_byte_limit;
        return out;
    }

    const std::uint64_t contactCounterBytes =
        std::max<std::uint64_t>(16u, in.environmentCount * sizeof(std::uint32_t));
    const std::uint64_t contactBytesPerRow =
        in.environmentCount * (sizeof(NMContactSampleGPU) +
            2u * sizeof(std::uint32_t));
    if (contactBytesPerRow == 0u ||
        contactCounterBytes >= kShippingContactPageScratchBudgetBytes) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    const std::uint64_t maxRowsByBudget =
        (kShippingContactPageScratchBudgetBytes - contactCounterBytes) /
            contactBytesPerRow;
    const std::uint64_t contactPageRows =
        std::min({in.contactPairCount, in.contactPageRows,
                  maxRowsByBudget});
    if (contactPageRows == 0u) {
        out.error = ShippingContactPagePlanError::page_capacity_zero;
        return out;
    }
    out.environmentCount = static_cast<std::uint32_t>(in.environmentCount);
    out.otherResidentBytes = in.otherResidentBytes;
    out.contactPageRows = static_cast<std::uint32_t>(contactPageRows);
    out.cohesivePageFaces = static_cast<std::uint32_t>(cohesivePageFaces);
    std::uint64_t contactPages = 0u, cohesivePages = 0u;
    const bool cohesivePageCountValid =
        in.cohesiveFaceCount == 0u
            ? (cohesivePages = 0u, true)
            : ceilPages(in.cohesiveFaceCount, cohesivePageFaces, cohesivePages);
    if (!ceilPages(in.contactPairCount, contactPageRows, contactPages) ||
        !cohesivePageCountValid ||
        contactPages > std::numeric_limits<std::uint32_t>::max() ||
        cohesivePages > std::numeric_limits<std::uint32_t>::max()) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    out.contactPageCount = static_cast<std::uint32_t>(contactPages);
    out.cohesivePageCount = static_cast<std::uint32_t>(cohesivePages);
    ShippingContactPageRange last{};
    if (!shippingContactPageRange(in.contactPairCount, contactPageRows,
                                  contactPages - 1u, last)) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    out.contactLastPageRows = last.rowCount;
    if (cohesivePages != 0u) {
        if (!shippingContactPageRange(in.cohesiveFaceCount, cohesivePageFaces,
                                      cohesivePages - 1u, last)) {
            out.error = ShippingContactPagePlanError::arithmetic_overflow;
            return out;
        }
        out.cohesiveLastPageFaces = last.rowCount;
    }

    const std::uint64_t e = in.environmentCount;
    const std::uint64_t contactCapacity = contactPageRows;
    const std::uint64_t cohesiveCapacity = cohesivePageFaces;
    if (!bytes(e, in.contactPairCount, sizeof(NMContactPairGPU),
               out.contactPairTableBytes) ||
        !bytes(e, in.contactPairCount, sizeof(nm_float4),
               out.contactAcceptedHistoryBytes) ||
        !bytes(e, in.contactPairCount, sizeof(nm_float4),
               out.contactCandidateHistoryBytes) ||
        !bytes(e, in.contactPairCount, sizeof(nm_float4),
               out.contactCheckpointHistoryBytes) ||
        !bytes(e, in.cohesiveFaceCount, sizeof(NMCohesiveFaceGPU),
               out.cohesiveAcceptedFaceBytes) ||
        !bytes(e, in.cohesiveFaceCount, sizeof(NMCohesiveFaceGPU),
               out.cohesiveDefaultFaceBytes) ||
        !bytes(e, in.cohesiveFaceCount, sizeof(NMCohesiveFaceGPU),
               out.cohesiveCandidateFaceBytes) ||
        !bytes(e, in.cohesiveFaceCount, sizeof(NMCohesiveFaceGPU),
               out.cohesiveCheckpointFaceBytes) ||
        !bytes(e, in.cohesiveFaceCount == 0u ? 0u : in.femNodeCount,
               sizeof(nm_float4), out.cohesiveAccumulatorBytes) ||
        !multiply(e, in.retainedGlobalUnknownBufferCount,
                  out.retainedGlobalUnknownBytes) ||
        !multiply(out.retainedGlobalUnknownBytes,
                  in.retainedGlobalUnknownBytesPerBuffer,
                  out.retainedGlobalUnknownBytes) ||
        !bytes(e, contactCapacity, sizeof(NMContactSampleGPU),
               out.contactSamplePageBytes) ||
        !bytes(e, contactCapacity, sizeof(std::uint32_t),
               out.contactCandidateIdPageBytes) ||
        !bytes(e, contactCapacity, sizeof(std::uint32_t),
               out.contactSlotPageBytes) ||
        !bytes(e, 1u, sizeof(std::uint32_t),
               out.contactPageCountsBytes) ||
        !bytes(e, cohesiveCapacity, sizeof(NMCohesiveFaceVectorGPU),
               out.cohesiveForcePageBytes) ||
        !bytes(e, cohesiveCapacity, sizeof(NMCohesiveFaceVectorGPU),
               out.cohesiveOperatorPageBytes)) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    if (out.contactPageCountsBytes < 16u) out.contactPageCountsBytes = 16u;

    if (!addTo(out.contactPairTableBytes, out.persistentBytes) ||
        !addTo(out.contactAcceptedHistoryBytes, out.persistentBytes) ||
        !addTo(out.contactCandidateHistoryBytes, out.persistentBytes) ||
        !addTo(out.contactCheckpointHistoryBytes, out.persistentBytes) ||
        !addTo(out.cohesiveAcceptedFaceBytes, out.persistentBytes) ||
        !addTo(out.cohesiveDefaultFaceBytes, out.persistentBytes) ||
        !addTo(out.cohesiveCandidateFaceBytes, out.persistentBytes) ||
        !addTo(out.cohesiveCheckpointFaceBytes, out.persistentBytes) ||
        !addTo(out.retainedGlobalUnknownBytes, out.persistentBytes) ||
        !addTo(in.otherResidentBytes, out.persistentBytes) ||
        !addTo(out.contactSamplePageBytes, out.contactPageScratchBytes) ||
        !addTo(out.contactCandidateIdPageBytes, out.contactPageScratchBytes) ||
        !addTo(out.contactSlotPageBytes, out.contactPageScratchBytes) ||
        !addTo(out.contactPageCountsBytes, out.contactPageScratchBytes) ||
        !addTo(out.cohesiveForcePageBytes, out.cohesivePageScratchBytes) ||
        !addTo(out.cohesiveOperatorPageBytes, out.cohesivePageScratchBytes) ||
        !addTo(out.cohesiveAccumulatorBytes, out.cohesivePageScratchBytes)) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    if (out.contactPairTableBytes > in.maxBufferLength ||
        out.contactAcceptedHistoryBytes > in.maxBufferLength ||
        out.contactCandidateHistoryBytes > in.maxBufferLength ||
        out.contactCheckpointHistoryBytes > in.maxBufferLength ||
        out.cohesiveAcceptedFaceBytes > in.maxBufferLength ||
        out.cohesiveDefaultFaceBytes > in.maxBufferLength ||
        out.cohesiveCandidateFaceBytes > in.maxBufferLength ||
        out.cohesiveCheckpointFaceBytes > in.maxBufferLength ||
        in.retainedGlobalUnknownBytesPerBuffer > in.maxBufferLength ||
        out.contactSamplePageBytes > in.maxBufferLength ||
        out.contactCandidateIdPageBytes > in.maxBufferLength ||
        out.contactSlotPageBytes > in.maxBufferLength ||
        out.contactPageCountsBytes > in.maxBufferLength ||
        out.cohesiveForcePageBytes > in.maxBufferLength ||
        out.cohesiveOperatorPageBytes > in.maxBufferLength ||
        out.cohesiveAccumulatorBytes > in.maxBufferLength) {
        out.error = ShippingContactPagePlanError::buffer_exceeds_metal_limit;
        return out;
    }
    if (!add(out.contactSamplePageBytes,
             out.contactCandidateIdPageBytes, out.contactPageScratchBytes) ||
        !addTo(out.contactSlotPageBytes, out.contactPageScratchBytes) ||
        !addTo(out.contactPageCountsBytes, out.contactPageScratchBytes)) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    if (out.contactPageScratchBytes > kShippingContactPageScratchBudgetBytes) {
        out.error = ShippingContactPagePlanError::contact_page_scratch_exceeded;
        return out;
    }
    if (out.cohesiveForcePageBytes >
            kShippingCohesiveForcePageBufferBytes ||
        out.cohesiveOperatorPageBytes >
            kShippingCohesiveForcePageBufferBytes) {
        out.error = ShippingContactPagePlanError::cohesive_page_scratch_exceeded;
        return out;
    }
    if (!add(out.persistentBytes, out.contactPageScratchBytes,
             out.contactPhasePeakBytes) ||
        !add(out.persistentBytes, out.cohesivePageScratchBytes,
             out.cohesivePhasePeakBytes) ||
        !add(out.persistentBytes,
             std::max(out.contactPageScratchBytes,
                      out.cohesivePageScratchBytes),
             out.serializedPagePeakBytes) ||
        !add(out.persistentBytes,
             out.contactPageScratchBytes, out.conservativeOverlapPeakBytes) ||
        !addTo(out.cohesivePageScratchBytes,
               out.conservativeOverlapPeakBytes)) {
        out.error = ShippingContactPagePlanError::arithmetic_overflow;
        return out;
    }
    if (out.conservativeOverlapPeakBytes >
        kShippingModeledAggregateBudgetBytes) {
        out.error = ShippingContactPagePlanError::modeled_aggregate_exceeded;
    }
    return out;
}

[[nodiscard]] constexpr const char* shippingContactPagePlanErrorText(
    const ShippingContactPagePlanError error) noexcept {
    switch (error) {
        case ShippingContactPagePlanError::none: return "";
        case ShippingContactPagePlanError::zero_environment_count: return "environment count is zero";
        case ShippingContactPagePlanError::unsupported_environment_count: return "environment count exceeds the authored limit";
        case ShippingContactPagePlanError::zero_contact_domain: return "contact pair domain is zero";
        case ShippingContactPagePlanError::zero_cohesive_accumulator_node_domain: return "cohesive accumulator node domain is zero";
        case ShippingContactPagePlanError::page_capacity_zero: return "page capacity is zero";
        case ShippingContactPagePlanError::contact_page_capacity_exceeds_authored_limit: return "contact sample page exceeds the authored row cap";
        case ShippingContactPagePlanError::cohesive_page_capacity_exceeds_byte_limit: return "cohesive force page exceeds the existing per-buffer byte cap";
        case ShippingContactPagePlanError::contact_domain_exceeds_u32: return "flattened contact row domain exceeds uint32";
        case ShippingContactPagePlanError::cohesive_domain_exceeds_u32: return "cohesive face or six-slot domain exceeds uint32";
        case ShippingContactPagePlanError::arithmetic_overflow: return "page plan byte or index arithmetic overflow";
        case ShippingContactPagePlanError::buffer_exceeds_metal_limit: return "a named full or page buffer exceeds Metal maxBufferLength";
        case ShippingContactPagePlanError::contact_page_scratch_exceeded: return "contact sample/id/slot page exceeds the existing 36 MiB scratch budget";
        case ShippingContactPagePlanError::cohesive_page_scratch_exceeded: return "cohesive force/operator page exceeds the existing 8 MiB per-buffer cap";
        case ShippingContactPagePlanError::modeled_aggregate_exceeded: return "named page-plan aggregate exceeds the modeled 16 GiB limit";
    }
    return "unknown contact page plan error";
}

} // namespace numi::matter::detail
