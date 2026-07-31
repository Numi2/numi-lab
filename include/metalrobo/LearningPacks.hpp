#pragma once

#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/TaskProgram.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace metalrobo {

inline constexpr std::uint32_t kTaskPackFormatVersion = 1u;
inline constexpr std::uint32_t kPolicyPackFormatVersion = 1u;

enum class LearningPackStatus : std::uint32_t {
    success = 0u,
    invalidPack,
    ioFailure,
    unsupportedVersion,
    corruptPayload,
    capacityOverflow,
    internalFailure,
};

struct LearningPackResult {
    LearningPackStatus status = LearningPackStatus::success;
    std::uint64_t contentHash = 0u;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == LearningPackStatus::success;
    }
};

// Both formats are deterministic, little-endian native artifacts with an
// explicit wire version, payload length, and content fingerprint. Reads and
// writes publish transactionally.
[[nodiscard]] LearningPackResult writeTaskPack(
    const TaskPack& pack,
    const std::filesystem::path& path
);

[[nodiscard]] LearningPackResult readTaskPack(
    const std::filesystem::path& path,
    TaskPack& output
);

[[nodiscard]] LearningPackResult writePolicyPack(
    const PolicyPack& pack,
    const std::filesystem::path& path
);

[[nodiscard]] LearningPackResult readPolicyPack(
    const std::filesystem::path& path,
    PolicyPack& output
);

[[nodiscard]] const char* learningPackStatusName(
    LearningPackStatus status
) noexcept;

} // namespace metalrobo
