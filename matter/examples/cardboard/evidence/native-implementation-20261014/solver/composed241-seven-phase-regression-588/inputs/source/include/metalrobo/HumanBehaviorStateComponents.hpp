#pragma once

#include "metalrobo/mrnx_bridge_v1.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace metalrobo {

using HumanBehaviorStateDigest = std::array<std::uint8_t, 32u>;

// Host-visible copy of one exact candidate channel. The descriptor supplies
// canonical, pointer-free metadata; values and validity are the bytes copied
// from the device-private ranges on the producing command-buffer timeline.
struct HumanBehaviorSensorStateChannelV1 {
    const mrnx_candidate_channel_v2* descriptor = nullptr;
    std::span<const std::byte> values{};
    std::span<const std::byte> validity{};
};

[[nodiscard]] HumanBehaviorStateDigest
humanBehaviorUnpublishedSensorStateSHA256() noexcept;

[[nodiscard]] bool humanBehaviorExactSensorStateSHA256(
    std::uint32_t sourcePacketABI,
    const mrnx_candidate_timing_v2& timing,
    std::span<const HumanBehaviorSensorStateChannelV1> channels,
    HumanBehaviorStateDigest& output,
    std::string& error
) noexcept;

[[nodiscard]] HumanBehaviorStateDigest
humanBehaviorUnpublishedPublicationStateSHA256() noexcept;

[[nodiscard]] bool humanBehaviorExactPublicationStateSHA256(
    std::uint64_t publicationEpoch,
    const mrnx_publication_v2& publication,
    HumanBehaviorStateDigest& output,
    std::string& error
) noexcept;

[[nodiscard]] bool humanBehaviorStateDigestPresent(
    const HumanBehaviorStateDigest& digest
) noexcept;

} // namespace metalrobo
