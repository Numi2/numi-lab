#pragma once

#include "metalrobo/numi_human_joint_equality_gpu.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class NumiHumanJointEqualityStatus : std::uint32_t {
    success = 0u,
    truncatedPayload,
    invalidMagic,
    unsupportedAbi,
    sourceMismatch,
    invalidDimensions,
    invalidRecord,
    duplicateDependent,
    chainedDependency,
    nonfiniteInput,
    nonfiniteResult,
};

struct NumiHumanJointEqualityDiagnostics {
    NumiHumanJointEqualityStatus status =
        NumiHumanJointEqualityStatus::success;
    std::uint32_t failingIndex = MR_INVALID_INDEX;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanJointEqualityStatus::success;
    }
};

struct NumiHumanJointEqualityPayload {
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
    std::vector<MRNumiHumanJointEqualityGPU> records;
};

struct NumiHumanJointEqualityEvaluation {
    double positionError = 0.0;
    double derivative = 0.0;
    double dependentTarget = 0.0;
};

[[nodiscard]] NumiHumanJointEqualityDiagnostics
decodeNumiHumanJointEqualityPayload(
    std::span<const std::byte> bytes,
    std::span<const std::uint8_t> expectedSourceSha256,
    NumiHumanJointEqualityPayload& payload
);

[[nodiscard]] NumiHumanJointEqualityDiagnostics
evaluateNumiHumanJointEquality(
    const MRNumiHumanJointEqualityGPU& equality,
    std::span<const double> q,
    NumiHumanJointEqualityEvaluation& evaluation
);

// Projects only dependent coordinates onto the exact source polynomial.
// The program is required to have unique, non-chained dependents, so source
// order is deterministic and no nonlinear iteration is hidden here.
[[nodiscard]] NumiHumanJointEqualityDiagnostics
projectNumiHumanJointEqualities(
    std::span<const MRNumiHumanJointEqualityGPU> equalities,
    std::span<double> q,
    double* maximumCorrection = nullptr
);

[[nodiscard]] const char* numiHumanJointEqualityStatusName(
    NumiHumanJointEqualityStatus status
) noexcept;

} // namespace metalrobo
