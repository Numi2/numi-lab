#pragma once

#include "metalrobo/NumiHumanLoadedKneeBinding.hpp"
#include "numi/matter/matter.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kNumiHumanSourceConstraintProgramVersionV1 =
    1u;

// The expected SHA-256 values must come from an independently authenticated
// source-law contract. The loader never accepts a payload merely because its
// self-reported header is internally consistent.
struct NumiHumanSourceConstraintExpectationV1 {
    NumiHumanLoadedKneeDigest equalityFileSHA256{};
    NumiHumanLoadedKneeDigest limitFileSHA256{};
    NumiHumanLoadedKneeDigest sourceArchiveSHA256{};
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t equalityRowCount = 0u;
    std::uint32_t limitRowCount = 0u;
    std::uint32_t policy = 0u;
    std::uint32_t flags = 0u;
};

// Owns the immutable bytes as well as decoded Matter rows. RuntimeConfiguration
// borrows the row spans, so this object must outlive Runtime::initialize().
// Matter remains the sole owner of runtime constraint force and accepted state.
struct NumiHumanSourceConstraintProgramV1 {
    std::uint32_t version = kNumiHumanSourceConstraintProgramVersionV1;
    std::vector<std::byte> equalityPayloadBytes;
    std::vector<std::byte> limitPayloadBytes;
    std::vector<NMHumanJointEqualityGPU> jointEqualities;
    std::vector<NMHumanJointLimitGPU> jointLimits;
    NMHumanEqualityDispatchGPU equalityDispatch{};
    NMHumanLimitDispatchGPU limitDispatch{};
    std::uint64_t equalityFingerprint = 0u;
    std::uint64_t limitFingerprint = 0u;
    NumiHumanLoadedKneeDigest equalityFileSHA256{};
    NumiHumanLoadedKneeDigest limitFileSHA256{};
    NumiHumanLoadedKneeDigest sourceArchiveSHA256{};
};

// Opens both paths without following symlinks, checks regular-file identity,
// exact SHA-256, the complete NHEQ2/NHLIM1 envelopes, cross-program identity,
// and every scalar row before publishing output. On failure output is reset.
[[nodiscard]] bool loadNumiHumanSourceConstraintProgramV1(
    const std::filesystem::path& jointEqualityPath,
    const std::filesystem::path& jointLimitPath,
    const NumiHumanSourceConstraintExpectationV1& expectation,
    NumiHumanSourceConstraintProgramV1& output,
    std::string& error);

[[nodiscard]] bool loadNumiHumanLoadedKneeSourceConstraintProgramV1(
    const std::filesystem::path& jointEqualityPath,
    const std::filesystem::path& jointLimitPath,
    const NumiHumanLoadedKneeBindingAdmissionV1& authenticatedBase,
    const NumiHumanLoadedKneeSourceComplianceAdmissionV1&
        authenticatedSourceCompliance,
    NumiHumanSourceConstraintProgramV1& output,
    std::string& error);

// Binds the admitted immutable rows to the one Matter runtime configuration.
// Existing nonempty constraint fields are rejected instead of overwritten.
// This function does not initialize or step Matter and makes no qualification
// claim. The program must remain alive until Runtime::initialize() returns.
[[nodiscard]] bool configureNumiHumanSourceConstraintsV1(
    const NumiHumanSourceConstraintProgramV1& program,
    numi::matter::RuntimeConfiguration& configuration,
    std::string& error);

} // namespace metalrobo
