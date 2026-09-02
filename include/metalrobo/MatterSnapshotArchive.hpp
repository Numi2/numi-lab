#pragma once

#include "numi/matter/matter.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace metalrobo {

constexpr std::uint32_t kMatterSnapshotArchiveVersion = 4u;

enum class MatterSnapshotArchiveStatus : std::uint32_t {
    success = 0u,
    invalidSnapshot,
    unsupportedVersion,
    corruptPayload,
    capacityOverflow,
    ioFailure,
};

struct MatterSnapshotArchiveResult {
    MatterSnapshotArchiveStatus status =
        MatterSnapshotArchiveStatus::success;
    std::string message;
    std::uint64_t contentHash = 0u;
    std::uint64_t payloadBytes = 0u;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MatterSnapshotArchiveStatus::success;
    }
};

// Publishes a completion-boundary Matter snapshot through a same-directory
// temporary file and atomic rename. The archive is intentionally ABI-bound:
// raw GPU state records retain their exact bytes, and read/restore requires
// the matching Matter ABI and device-program fingerprint.
[[nodiscard]] MatterSnapshotArchiveResult writeMatterSnapshotArchive(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    const std::filesystem::path& path
);

[[nodiscard]] MatterSnapshotArchiveResult readMatterSnapshotArchive(
    const std::filesystem::path& path,
    numi::matter::RuntimeStateSnapshot& output
);

// Compares only the state that owns deterministic continuation. Diagnostic
// availability text is deliberately excluded.
[[nodiscard]] bool sameMatterSnapshotAuthority(
    const numi::matter::RuntimeStateSnapshot& left,
    const numi::matter::RuntimeStateSnapshot& right
) noexcept;

[[nodiscard]] const char* matterSnapshotArchiveStatusName(
    MatterSnapshotArchiveStatus status
) noexcept;

} // namespace metalrobo
