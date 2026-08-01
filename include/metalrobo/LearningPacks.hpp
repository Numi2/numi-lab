#pragma once

#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/TaskProgram.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kTaskPackFormatVersion = 10u;
inline constexpr std::uint32_t kPolicyPackFormatVersion = 3u;
inline constexpr std::uint32_t
    kPolicyRolloutPackFormatVersion = 3u;

struct PolicyRolloutPack {
    std::string id;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t policyFingerprint = 0u;
    std::uint64_t policyRevision = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t controlStepCount = 0u;
    std::uint32_t actorObservationCount = 0u;
    std::uint32_t criticObservationCount = 0u;
    std::uint32_t actionCount = 0u;
    std::vector<float> actorObservations;
    std::vector<float> criticObservations;
    std::vector<float> latents;
    std::vector<float> logProbabilities;
    std::vector<float> values;
    std::vector<float> bootstrapValues;
    std::vector<MRTaskTransitionGPU> transitions;
};

// Synchronous zero-copy publication view for large native rollout batches.
// The writer never retains these spans beyond writePolicyRolloutPack().
struct PolicyRolloutPackView {
    std::string_view id;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t policyFingerprint = 0u;
    std::uint64_t policyRevision = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t controlStepCount = 0u;
    std::uint32_t actorObservationCount = 0u;
    std::uint32_t criticObservationCount = 0u;
    std::uint32_t actionCount = 0u;
    std::span<const float> actorObservations;
    std::span<const float> criticObservations;
    std::span<const float> latents;
    std::span<const float> logProbabilities;
    std::span<const float> values;
    std::span<const float> bootstrapValues;
    std::span<const MRTaskTransitionGPU> transitions;
};

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

// Canonical content fingerprint used by every learning-pack wire format.
// Exposed so zero-copy readers can validate an already mapped payload without
// allocating a second payload or reimplementing the hash in another language.
[[nodiscard]] std::uint64_t learningPackContentHash(
    std::span<const std::byte> payload
) noexcept;

// Learning packs are deterministic, little-endian native artifacts with an
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

[[nodiscard]] LearningPackResult writePolicyRolloutPack(
    const PolicyRolloutPack& pack,
    const std::filesystem::path& path
);

[[nodiscard]] LearningPackResult writePolicyRolloutPack(
    const PolicyRolloutPackView& pack,
    const std::filesystem::path& path
);

[[nodiscard]] LearningPackResult readPolicyRolloutPack(
    const std::filesystem::path& path,
    PolicyRolloutPack& output
);

[[nodiscard]] const char* learningPackStatusName(
    LearningPackStatus status
) noexcept;

} // namespace metalrobo
