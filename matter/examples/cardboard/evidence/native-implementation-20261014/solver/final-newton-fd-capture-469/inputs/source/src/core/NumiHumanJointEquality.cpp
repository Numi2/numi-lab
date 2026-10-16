#include "metalrobo/NumiHumanJointEquality.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace metalrobo {
namespace {

constexpr std::array<char, 8u> kMagic{
    'N', 'H', 'E', 'Q', '1', '\0', '\0', '\0'
};
constexpr std::uint32_t kPayloadAbi = 1u;

#pragma pack(push, 1)
struct Header {
    std::array<char, 8u> magic{};
    std::uint32_t abi = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t recordCount = 0u;
    std::uint32_t recordBytes = 0u;
    std::uint32_t sourceRecordCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint32_t reserved2 = 0u;
    std::uint32_t reserved3 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 80u);

NumiHumanJointEqualityDiagnostics failure(
    const NumiHumanJointEqualityStatus status,
    const std::uint32_t index = MR_INVALID_INDEX
) {
    return {.status = status, .failingIndex = index};
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

} // namespace

NumiHumanJointEqualityDiagnostics decodeNumiHumanJointEqualityPayload(
    const std::span<const std::byte> bytes,
    const std::span<const std::uint8_t> expectedSourceSha256,
    NumiHumanJointEqualityPayload& payload
) {
    if (bytes.size() < sizeof(Header)) {
        return failure(NumiHumanJointEqualityStatus::truncatedPayload);
    }
    Header header;
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.magic != kMagic) {
        return failure(NumiHumanJointEqualityStatus::invalidMagic);
    }
    if (header.abi != kPayloadAbi ||
        header.recordBytes != sizeof(MRNumiHumanJointEqualityGPU)) {
        return failure(NumiHumanJointEqualityStatus::unsupportedAbi);
    }
    if (header.nq == 0u || header.nv == 0u ||
        header.recordCount == 0u ||
        header.sourceRecordCount != header.recordCount ||
        header.reserved0 != 0u || header.reserved1 != 0u ||
        header.reserved2 != 0u || header.reserved3 != 0u ||
        bytes.size() != sizeof(Header) +
            static_cast<std::size_t>(header.recordCount) *
                sizeof(MRNumiHumanJointEqualityGPU)) {
        return failure(NumiHumanJointEqualityStatus::invalidDimensions);
    }
    if (!expectedSourceSha256.empty() &&
        (expectedSourceSha256.size() != header.sourceSha256.size() ||
         !std::equal(
             expectedSourceSha256.begin(), expectedSourceSha256.end(),
             header.sourceSha256.begin()
         ))) {
        return failure(NumiHumanJointEqualityStatus::sourceMismatch);
    }
    NumiHumanJointEqualityPayload candidate;
    candidate.nq = header.nq;
    candidate.nv = header.nv;
    candidate.sourceSha256 = header.sourceSha256;
    candidate.records.resize(header.recordCount);
    std::memcpy(
        candidate.records.data(), bytes.data() + sizeof(Header),
        candidate.records.size() * sizeof(MRNumiHumanJointEqualityGPU)
    );
    std::vector<std::uint8_t> dependent(header.nv, 0u);
    for (std::size_t index = 0u; index < candidate.records.size(); ++index) {
        const auto& record = candidate.records[index];
        const bool fixed = record.indices.z == MR_INVALID_INDEX &&
            record.indices.w == MR_INVALID_INDEX;
        const bool coupled = record.indices.z < header.nq &&
            record.indices.w < header.nv;
        if (record.indices.x >= header.nq || record.indices.y >= header.nv ||
            (!fixed && !coupled) ||
            (coupled && (record.indices.x == record.indices.z ||
                         record.indices.y == record.indices.w)) ||
            !finite(record.referencesAndCoefficients0) ||
            !finite(record.coefficients1) || !finite(record.solref) ||
            !finite(record.solimp0) || !finite(record.solimp1) ||
            record.coefficients1.w != 0.0f || record.solref.z != 0.0f ||
            record.solref.w != 0.0f || record.solimp1.y != 0.0f ||
            record.solimp1.z != 0.0f || record.solimp1.w != 0.0f) {
            return failure(
                NumiHumanJointEqualityStatus::invalidRecord,
                static_cast<std::uint32_t>(index)
            );
        }
        if (dependent[record.indices.y] != 0u) {
            return failure(
                NumiHumanJointEqualityStatus::duplicateDependent,
                static_cast<std::uint32_t>(index)
            );
        }
        dependent[record.indices.y] = 1u;
    }
    for (std::size_t index = 0u; index < candidate.records.size(); ++index) {
        const auto& record = candidate.records[index];
        if (record.indices.w != MR_INVALID_INDEX &&
            dependent[record.indices.w] != 0u) {
            return failure(
                NumiHumanJointEqualityStatus::chainedDependency,
                static_cast<std::uint32_t>(index)
            );
        }
    }
    payload = std::move(candidate);
    return {};
}

NumiHumanJointEqualityDiagnostics evaluateNumiHumanJointEquality(
    const MRNumiHumanJointEqualityGPU& equality,
    const std::span<const double> q,
    NumiHumanJointEqualityEvaluation& evaluation
) {
    if (equality.indices.x >= q.size() ||
        (equality.indices.z != MR_INVALID_INDEX &&
         equality.indices.z >= q.size())) {
        return failure(NumiHumanJointEqualityStatus::invalidRecord);
    }
    if (!std::all_of(q.begin(), q.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        return failure(NumiHumanJointEqualityStatus::nonfiniteInput);
    }
    const double delta = equality.indices.z == MR_INVALID_INDEX
        ? 0.0
        : q[equality.indices.z] -
            equality.referencesAndCoefficients0.y;
    const double a0 = equality.referencesAndCoefficients0.z;
    const double a1 = equality.referencesAndCoefficients0.w;
    const double a2 = equality.coefficients1.x;
    const double a3 = equality.coefficients1.y;
    const double a4 = equality.coefficients1.z;
    const double polynomial = a0 + delta * (
        a1 + delta * (a2 + delta * (a3 + delta * a4))
    );
    const double derivative = equality.indices.z == MR_INVALID_INDEX
        ? 0.0
        : a1 + delta * (2.0 * a2 + delta * (3.0 * a3 + 4.0 * delta * a4));
    const double target = equality.referencesAndCoefficients0.x + polynomial;
    const double error = q[equality.indices.x] - target;
    if (!std::isfinite(target) || !std::isfinite(error) ||
        !std::isfinite(derivative)) {
        return failure(NumiHumanJointEqualityStatus::nonfiniteResult);
    }
    evaluation = {
        .positionError = error,
        .derivative = derivative,
        .dependentTarget = target,
    };
    return {};
}

NumiHumanJointEqualityDiagnostics projectNumiHumanJointEqualities(
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    const std::span<double> q,
    double* const maximumCorrection
) {
    std::vector<double> candidate(q.begin(), q.end());
    double maximum = 0.0;
    for (std::size_t index = 0u; index < equalities.size(); ++index) {
        NumiHumanJointEqualityEvaluation evaluation;
        const auto diagnostics = evaluateNumiHumanJointEquality(
            equalities[index], candidate, evaluation
        );
        if (!diagnostics.succeeded()) {
            auto failed = diagnostics;
            failed.failingIndex = static_cast<std::uint32_t>(index);
            return failed;
        }
        maximum = std::max(maximum, std::abs(evaluation.positionError));
        candidate[equalities[index].indices.x] = evaluation.dependentTarget;
    }
    if (!std::all_of(candidate.begin(), candidate.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        return failure(NumiHumanJointEqualityStatus::nonfiniteResult);
    }
    std::copy(candidate.begin(), candidate.end(), q.begin());
    if (maximumCorrection != nullptr) *maximumCorrection = maximum;
    return {};
}

const char* numiHumanJointEqualityStatusName(
    const NumiHumanJointEqualityStatus status
) noexcept {
    switch (status) {
    case NumiHumanJointEqualityStatus::success: return "success";
    case NumiHumanJointEqualityStatus::truncatedPayload:
        return "truncatedPayload";
    case NumiHumanJointEqualityStatus::invalidMagic: return "invalidMagic";
    case NumiHumanJointEqualityStatus::unsupportedAbi:
        return "unsupportedAbi";
    case NumiHumanJointEqualityStatus::sourceMismatch:
        return "sourceMismatch";
    case NumiHumanJointEqualityStatus::invalidDimensions:
        return "invalidDimensions";
    case NumiHumanJointEqualityStatus::invalidRecord: return "invalidRecord";
    case NumiHumanJointEqualityStatus::duplicateDependent:
        return "duplicateDependent";
    case NumiHumanJointEqualityStatus::chainedDependency:
        return "chainedDependency";
    case NumiHumanJointEqualityStatus::nonfiniteInput:
        return "nonfiniteInput";
    case NumiHumanJointEqualityStatus::nonfiniteResult:
        return "nonfiniteResult";
    }
    return "unknown";
}

} // namespace metalrobo
