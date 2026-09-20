#include "metalrobo/NumiHumanRuntimeIdentity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <std::size_t Count>
std::span<const std::byte> bytes(
    const std::array<std::uint8_t, Count>& value) noexcept {
    return std::as_bytes(std::span(value));
}

} // namespace

int main() try {
    constexpr std::array<std::uint8_t, 3u> rigid{0x01u, 0x02u, 0x03u};
    constexpr std::array<std::uint8_t, 2u> muscle{0x10u, 0x20u};
    constexpr std::array<std::uint8_t, 4u> support{
        0xaau, 0xbbu, 0xccu, 0xddu};
    constexpr std::array<std::uint8_t, 3u> equality{0x31u, 0x32u, 0x33u};
    constexpr std::array<std::uint8_t, 2u> limits{0xf0u, 0x0du};
    constexpr std::array<std::uint8_t, 4u> initial{
        0xdeu, 0xadu, 0xbeu, 0xefu};

    const auto base = metalrobo::numiHumanRuntimeBaseSourceFingerprint(
        bytes(rigid), bytes(muscle), bytes(support));
    const auto equalityFingerprint =
        metalrobo::numiHumanRuntimePayloadFingerprint(bytes(equality));
    const auto constrained = metalrobo::numiHumanRuntimeAppendPayloadOwner(
        base, "NHEQ2", equalityFingerprint);
    const auto limitFingerprint =
        metalrobo::numiHumanRuntimePayloadFingerprint(bytes(limits));
    const auto limited = metalrobo::numiHumanRuntimeAppendPayloadOwner(
        constrained, "NHLIM1", limitFingerprint);
    const auto initialFingerprint =
        metalrobo::numiHumanRuntimePayloadFingerprint(bytes(initial));
    const auto prepared = metalrobo::numiHumanRuntimeAppendInitialState(
        limited, initialFingerprint);

    require(base == 0x41eaddc008505780ull,
        "base source fingerprint changed");
    require(equalityFingerprint == 0x456fc2181822c4dbull,
        "payload fingerprint changed");
    require(constrained == 0x771f468d8e2b6060ull,
        "NHEQ2 source composition changed");
    require(limited == 0xc69e403855264314ull,
        "NHLIM1 source composition changed");
    require(initialFingerprint == 0x277045760cdd0993ull,
        "NHINIT payload fingerprint changed");
    require(prepared == 0x3fcab5efdc5b00f2ull,
        "NHINIT source composition changed");

    constexpr std::array<std::byte, 0u> empty{};
    require(metalrobo::numiHumanRuntimePayloadFingerprint(empty) ==
            14695981039346656037ull,
        "empty payload identity changed");
    require(metalrobo::numiHumanRuntimeAppendPayloadOwner(
            base, "NHEQ2", equalityFingerprint) == constrained,
        "source composition is not deterministic");
    require(metalrobo::numiHumanRuntimeAppendPayloadOwner(
            base, "NHLIM1", equalityFingerprint) != constrained,
        "owner domains are not identity-bearing");
    require(metalrobo::numiHumanRuntimeAppendInitialState(
            0x4c4ca642dbdadb4eull, 1u) == 0u,
        "historical NHINIT zero identity was remapped");

    std::cout << "Numi Human runtime identity contract passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
