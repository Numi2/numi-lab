#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace metalrobo {

// Canonical FNV-1a identity used by the NumanX Human runtime for immutable
// payload bytes. The offset basis is the identity of an empty payload.
[[nodiscard]] std::uint64_t numiHumanRuntimePayloadFingerprint(
    std::span<const std::byte> payload) noexcept;

// Base Human source identity. Payload order and byte lengths are part of the
// contract: NHRIGID, NHMYO, then NHCNT.
[[nodiscard]] std::uint64_t numiHumanRuntimeBaseSourceFingerprint(
    std::span<const std::byte> rigid,
    std::span<const std::byte> muscle,
    std::span<const std::byte> support) noexcept;

// Add a source-owning payload such as NHEQ2 or NHLIM1. The owner domain is
// independently FNV-1a hashed before its immutable payload fingerprint is
// mixed. This intentionally preserves the existing runtime ABI.
[[nodiscard]] std::uint64_t numiHumanRuntimeAppendPayloadOwner(
    std::uint64_t sourceFingerprint,
    std::string_view ownerDomain,
    std::uint64_t payloadFingerprint) noexcept;

// Bind an admitted NHINIT payload to the composed Human identity. All NHINIT
// ABI revisions use the historical raw marker "NHINIT1" followed by the
// little-endian payload fingerprint; the marker is not the on-disk magic.
// This historical final append uniquely preserves a raw zero result rather
// than remapping it to the FNV offset basis.
[[nodiscard]] std::uint64_t numiHumanRuntimeAppendInitialState(
    std::uint64_t sourceFingerprint,
    std::uint64_t initialStateFingerprint) noexcept;

} // namespace metalrobo
