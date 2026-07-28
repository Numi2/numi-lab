#pragma once

#include "metalrobo/WorldCompiler.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace metalrobo {

inline constexpr std::uint32_t kWorldPackFormatVersion = 2u;

enum class WorldPackStatus : std::uint32_t {
    success = 0u,
    invalidFamily,
    invalidPack,
    ioFailure,
    unsupportedVersion,
    corruptPayload,
    capacityOverflow,
    internalFailure,
};

struct WorldPackResult {
    WorldPackStatus status = WorldPackStatus::success;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == WorldPackStatus::success;
    }
};

// Persistent compiler artifact. The first format stores the complete rich
// WorldFamily rather than only reset tensors so later render/capture stages
// can resolve semantic identities and artifact provenance after loading.
struct MRWorldPack {
    std::uint32_t formatVersion = kWorldPackFormatVersion;
    std::uint64_t contentHash = 0u;
    WorldFamily family;

    [[nodiscard]] bool valid(std::string* reason = nullptr) const;
};

[[nodiscard]] WorldPackResult compileWorldPack(
    const WorldFamily& family,
    MRWorldPack& output
);

// Writes through a sibling temporary file and publishes with one rename.
[[nodiscard]] WorldPackResult writeWorldPack(
    const MRWorldPack& pack,
    const std::filesystem::path& path
);

// Transactional: output is unchanged on any parsing or validation failure.
[[nodiscard]] WorldPackResult readWorldPack(
    const std::filesystem::path& path,
    MRWorldPack& output
);

[[nodiscard]] const char* worldPackStatusName(
    WorldPackStatus status
) noexcept;

} // namespace metalrobo
