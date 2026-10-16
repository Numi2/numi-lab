#pragma once

#include "numi/matter/shared.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::matter::detail {

inline constexpr std::uint64_t kSparsePrivateContactScratchBudgetBytes =
    36ull * 1024ull * 1024ull;
inline constexpr std::uint32_t kDensePrivateContactPairLimit = 65536u;

[[nodiscard]] constexpr bool densePrivateContactPairDomainSupported(
    const std::uint32_t pairCount
) noexcept {
    return pairCount != 0u && pairCount <= kDensePrivateContactPairLimit;
}

enum class SparsePrivateContactAdmissionError : std::uint8_t {
    none,
    invalid_environment_count,
    zero_pair_count,
    zero_candidate_capacity,
    candidate_capacity_exceeds_pair_domain,
    flattened_pair_domain_exceeds_uint32,
    free_rigid_unknowns_unsupported,
    size_arithmetic_overflow,
    scratch_budget_exceeded,
};

struct SparsePrivateContactAdmission {
    SparsePrivateContactAdmissionError error =
        SparsePrivateContactAdmissionError::none;
    std::uint64_t sampleBytes = 0u;
    std::uint64_t pairSlotBytes = 0u;
    std::uint64_t candidateIdBytes = 0u;
    std::uint64_t parityReferenceBytes = 0u;
    std::uint64_t candidateCountBytes = 0u;
    std::uint64_t totalBytes = 0u;

    [[nodiscard]] constexpr bool admitted() const noexcept {
        return error == SparsePrivateContactAdmissionError::none;
    }
};

[[nodiscard]] constexpr bool sparsePrivateCheckedMultiply(
    const std::uint64_t lhs,
    const std::uint64_t rhs,
    std::uint64_t& result
) noexcept {
    if (rhs != 0u && lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
        return false;
    result = lhs * rhs;
    return true;
}

[[nodiscard]] constexpr bool sparsePrivateCheckedAdd(
    const std::uint64_t lhs,
    const std::uint64_t rhs,
    std::uint64_t& result
) noexcept {
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) return false;
    result = lhs + rhs;
    return true;
}

// The sparse Armijo path still addresses canonical pair IDs and the flattened
// environment/pair domain with Metal uint indices. It compacts only trial
// samples; accepted contact history and the authored pair table stay dense.
// This helper is shared by runtime admission and CPU checks so the checked
// address and byte arithmetic cannot drift apart.
[[nodiscard]] constexpr SparsePrivateContactAdmission
admitSparsePrivateContact(
    const std::uint32_t environmentCount,
    const std::uint32_t pairCount,
    const std::uint32_t candidateCapacity,
    const bool parity,
    const bool hasFreeRigidUnknowns,
    // Allocation-time exception for the bounded one-body, translation-only
    // BOX Armijo scope. The encoder still verifies the complete proxy/body
    // layout before dispatching any trial kernels.
    const bool boundedFreeBoxScope = false
) noexcept {
    SparsePrivateContactAdmission result;
    if (environmentCount == 0u || environmentCount > 4u) {
        result.error = SparsePrivateContactAdmissionError::invalid_environment_count;
        return result;
    }
    if (pairCount == 0u) {
        result.error = SparsePrivateContactAdmissionError::zero_pair_count;
        return result;
    }
    if (candidateCapacity == 0u) {
        result.error = SparsePrivateContactAdmissionError::zero_candidate_capacity;
        return result;
    }
    if (candidateCapacity > pairCount) {
        result.error =
            SparsePrivateContactAdmissionError::candidate_capacity_exceeds_pair_domain;
        return result;
    }
    if (hasFreeRigidUnknowns && !boundedFreeBoxScope) {
        result.error =
            SparsePrivateContactAdmissionError::free_rigid_unknowns_unsupported;
        return result;
    }

    std::uint64_t flattenedPairs = 0u;
    if (!sparsePrivateCheckedMultiply(environmentCount, pairCount,
                                      flattenedPairs) ||
        flattenedPairs > std::numeric_limits<std::uint32_t>::max()) {
        result.error =
            SparsePrivateContactAdmissionError::flattened_pair_domain_exceeds_uint32;
        return result;
    }
    std::uint64_t flattenedCandidates = 0u;
    if (!sparsePrivateCheckedMultiply(environmentCount, candidateCapacity,
                                      flattenedCandidates) ||
        flattenedCandidates > std::numeric_limits<std::uint32_t>::max()) {
        result.error =
            SparsePrivateContactAdmissionError::flattened_pair_domain_exceeds_uint32;
        return result;
    }

    std::uint64_t perEnvironmentSamples = 0u;
    std::uint64_t perEnvironmentPairSlots = 0u;
    std::uint64_t perEnvironmentIds = 0u;
    std::uint64_t perEnvironmentReference = 0u;
    if (!sparsePrivateCheckedMultiply(
            candidateCapacity, sizeof(NMContactSampleGPU),
            perEnvironmentSamples) ||
        !sparsePrivateCheckedMultiply(pairCount, sizeof(std::uint32_t),
                                      perEnvironmentPairSlots) ||
        !sparsePrivateCheckedMultiply(candidateCapacity, sizeof(std::uint32_t),
                                      perEnvironmentIds) ||
        (parity && !sparsePrivateCheckedMultiply(
            pairCount, sizeof(NMContactSampleGPU), perEnvironmentReference))) {
        result.error = SparsePrivateContactAdmissionError::size_arithmetic_overflow;
        return result;
    }

    const std::uint64_t environments = environmentCount;
    if (!sparsePrivateCheckedMultiply(environments, perEnvironmentSamples,
                                      result.sampleBytes) ||
        !sparsePrivateCheckedMultiply(environments, perEnvironmentPairSlots,
                                      result.pairSlotBytes) ||
        !sparsePrivateCheckedMultiply(environments, perEnvironmentIds,
                                      result.candidateIdBytes) ||
        !sparsePrivateCheckedMultiply(environments, perEnvironmentReference,
                                      result.parityReferenceBytes) ||
        !sparsePrivateCheckedMultiply(environments, sizeof(std::uint32_t),
                                      result.candidateCountBytes)) {
        result.error = SparsePrivateContactAdmissionError::size_arithmetic_overflow;
        return result;
    }
    result.candidateCountBytes =
        result.candidateCountBytes < 16u ? 16u : result.candidateCountBytes;

    std::uint64_t total = 0u;
    if (!sparsePrivateCheckedAdd(total, result.sampleBytes, total) ||
        !sparsePrivateCheckedAdd(total, result.pairSlotBytes, total) ||
        !sparsePrivateCheckedAdd(total, result.candidateIdBytes, total) ||
        !sparsePrivateCheckedAdd(total, result.parityReferenceBytes, total) ||
        !sparsePrivateCheckedAdd(total, result.candidateCountBytes, total) ||
        total > std::numeric_limits<std::size_t>::max()) {
        result.error = SparsePrivateContactAdmissionError::size_arithmetic_overflow;
        return result;
    }
    result.totalBytes = total;
    if (total > kSparsePrivateContactScratchBudgetBytes) {
        result.error = SparsePrivateContactAdmissionError::scratch_budget_exceeded;
    }
    return result;
}

[[nodiscard]] constexpr const char* sparsePrivateContactAdmissionErrorText(
    const SparsePrivateContactAdmissionError error
) noexcept {
    switch (error) {
        case SparsePrivateContactAdmissionError::none:
            return "";
        case SparsePrivateContactAdmissionError::invalid_environment_count:
            return "sparse private contact requires one through four environments";
        case SparsePrivateContactAdmissionError::zero_pair_count:
            return "sparse private contact pair domain must be nonzero";
        case SparsePrivateContactAdmissionError::zero_candidate_capacity:
            return "sparse private candidate capacity must be positive";
        case SparsePrivateContactAdmissionError::candidate_capacity_exceeds_pair_domain:
            return "sparse private candidate capacity exceeds the canonical pair domain";
        case SparsePrivateContactAdmissionError::flattened_pair_domain_exceeds_uint32:
            return "sparse private flattened pair index exceeds uint32";
        case SparsePrivateContactAdmissionError::free_rigid_unknowns_unsupported:
            return "sparse private free-rigid residual is limited to the admitted one-body six-DOF BOX scope";
        case SparsePrivateContactAdmissionError::size_arithmetic_overflow:
            return "sparse private contact scratch size arithmetic overflow";
        case SparsePrivateContactAdmissionError::scratch_budget_exceeded:
            return "requested sparse private contact arenas exceed the checked 36 MiB transaction budget";
    }
    return "sparse private contact admission failed";
}

} // namespace numi::matter::detail
