#include "metalrobo/NumiHumanRuntimeIdentity.hpp"

#include <array>

namespace metalrobo {
namespace {

constexpr std::uint64_t fnvOffset = 14695981039346656037ull;
constexpr std::uint64_t fnvPrime = 1099511628211ull;

void appendBytes(
    std::uint64_t& hash,
    const std::span<const std::byte> bytes) noexcept {
    for (const std::byte value : bytes) {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= fnvPrime;
    }
}

void appendU64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (std::uint32_t index = 0u; index < 8u; ++index) {
        const std::byte byte{static_cast<std::uint8_t>(
            (value >> (index * 8u)) & 0xffu)};
        appendBytes(hash, std::span<const std::byte>(&byte, 1u));
    }
}

[[nodiscard]] std::uint64_t normalized(const std::uint64_t hash) noexcept {
    return hash == 0u ? fnvOffset : hash;
}

} // namespace

std::uint64_t numiHumanRuntimePayloadFingerprint(
    const std::span<const std::byte> payload) noexcept {
    std::uint64_t hash = fnvOffset;
    appendBytes(hash, payload);
    return normalized(hash);
}

std::uint64_t numiHumanRuntimeBaseSourceFingerprint(
    const std::span<const std::byte> rigid,
    const std::span<const std::byte> muscle,
    const std::span<const std::byte> support) noexcept {
    constexpr std::string_view domain = "mrnx.fullbody.source.v1";
    std::uint64_t hash = fnvOffset;
    appendBytes(hash, std::as_bytes(
        std::span<const char>(domain.data(), domain.size())));
    for (const auto payload : std::array{rigid, muscle, support}) {
        appendU64(hash, static_cast<std::uint64_t>(payload.size()));
        appendBytes(hash, payload);
    }
    return normalized(hash);
}

std::uint64_t numiHumanRuntimeAppendPayloadOwner(
    std::uint64_t sourceFingerprint,
    const std::string_view ownerDomain,
    const std::uint64_t payloadFingerprint) noexcept {
    sourceFingerprint ^= numiHumanRuntimePayloadFingerprint(
        std::as_bytes(std::span<const char>(
            ownerDomain.data(), ownerDomain.size())));
    sourceFingerprint *= fnvPrime;
    sourceFingerprint ^= payloadFingerprint;
    sourceFingerprint *= fnvPrime;
    return normalized(sourceFingerprint);
}

std::uint64_t numiHumanRuntimeAppendInitialState(
    std::uint64_t sourceFingerprint,
    const std::uint64_t initialStateFingerprint) noexcept {
    constexpr std::string_view marker = "NHINIT1";
    appendBytes(sourceFingerprint, std::as_bytes(
        std::span<const char>(marker.data(), marker.size())));
    appendU64(sourceFingerprint, initialStateFingerprint);
    // Preserve the established runtime identity byte-for-byte. Unlike base
    // and source-owner composition, this historical final append did not
    // remap a zero result to the offset basis.
    return sourceFingerprint;
}

} // namespace metalrobo
