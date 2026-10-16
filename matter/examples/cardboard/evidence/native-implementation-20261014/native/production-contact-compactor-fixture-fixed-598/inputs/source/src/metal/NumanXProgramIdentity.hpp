#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>

namespace metalrobo::detail {

// Deterministic executable-image identity for trusted local program binding.
// This is replay/integrity evidence, not a cryptographic authenticator.
struct NumanXExecutableImageIdentity {
    std::uint64_t byteFingerprint = 0u;
    std::uint64_t byteCount = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return byteFingerprint != 0u;
    }

    friend bool operator==(
        const NumanXExecutableImageIdentity&,
        const NumanXExecutableImageIdentity&
    ) = default;
};

[[nodiscard]] inline bool numanXExecutableImageIdentity(
    const void* bytes,
    const std::size_t byteCount,
    NumanXExecutableImageIdentity& output
) noexcept {
    constexpr std::uint64_t offset = 14695981039346656037ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    if (bytes == nullptr || byteCount == 0u ||
        byteCount > std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    const auto* image = static_cast<const unsigned char*>(bytes);
    std::uint64_t hash = offset;
    for (std::size_t index = 0u; index < byteCount; ++index) {
        hash ^= image[index];
        hash *= prime;
    }
    output = {
        .byteFingerprint = hash == 0u ? offset : hash,
        .byteCount = static_cast<std::uint64_t>(byteCount),
    };
    return true;
}

[[nodiscard]] inline bool numanXExecutableImageIdentity(
    const std::filesystem::path& path,
    NumanXExecutableImageIdentity& output
) noexcept {
    try {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) return false;

        std::uint64_t fileByteCount = 0u;
        std::uint64_t hash = 14695981039346656037ull;
        std::array<char, 64u * 1024u> buffer{};
        while (stream) {
            stream.read(
                buffer.data(),
                static_cast<std::streamsize>(buffer.size())
            );
            const std::streamsize count = stream.gcount();
            if (count < 0 ||
                static_cast<std::uint64_t>(count) >
                    std::numeric_limits<std::uint64_t>::max() - fileByteCount) {
                return false;
            }
            for (std::streamsize index = 0; index < count; ++index) {
                hash ^= static_cast<unsigned char>(buffer[
                    static_cast<std::size_t>(index)
                ]);
                hash *= 1099511628211ull;
            }
            fileByteCount += static_cast<std::uint64_t>(count);
        }
        if (!stream.eof() || fileByteCount == 0u) return false;
        output = {
            .byteFingerprint = hash == 0u
                ? 14695981039346656037ull : hash,
            .byteCount = fileByteCount,
        };
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace metalrobo::detail
