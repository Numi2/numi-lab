#pragma once

#include "metalrobo/HumanBehaviorTrial.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace metalrobo {

inline constexpr std::uint32_t
    kHumanBehaviorAcceptedStateRootABIVersionV1 = 1u;
inline constexpr std::uint32_t
    kHumanBehaviorAcceptedStateRootExactNanosecondClockDomain = 2u;
inline constexpr std::uint32_t
    kHumanBehaviorAcceptedStateRootClockQuantumNanoseconds = 1u;
inline constexpr std::size_t
    kHumanBehaviorAcceptedStateRootComponentCountV1 = 7u;

// Stable owner roles in the accepted-state-root.v1 schema. Every source digest
// is produced and domain-separated by its named owner before it reaches this
// composer. In particular, physicalSHA256 is the direct-byte Human + Matter
// SHA-256 root; a 64-bit transaction/proof fingerprint is not a substitute.
enum class HumanBehaviorAcceptedStateComponentRoleV1 : std::uint32_t {
    physical = 1u,
    brain = 2u,
    sensor = 3u,
    controller = 4u,
    task = 5u,
    random = 6u,
    checkpoint = 7u,
};

enum class HumanBehaviorAcceptedStateDigestAlgorithmV1 : std::uint32_t {
    sha256 = 1u,
};

struct HumanBehaviorAcceptedStateComponentV1 {
    std::uint32_t abiVersion =
        kHumanBehaviorAcceptedStateRootABIVersionV1;
    std::uint32_t structSize =
        sizeof(HumanBehaviorAcceptedStateComponentV1);
    HumanBehaviorAcceptedStateComponentRoleV1 role =
        HumanBehaviorAcceptedStateComponentRoleV1::physical;
    HumanBehaviorAcceptedStateDigestAlgorithmV1 digestAlgorithm =
        HumanBehaviorAcceptedStateDigestAlgorithmV1::sha256;
    // Accepted publications carry positive, exact owner generations. Zero is
    // the fail-closed absent/uninitialized sentinel.
    std::uint64_t generation = 0u;
    HumanBehaviorDigest stateSHA256{};
};

struct HumanBehaviorAcceptedStateRootDescriptorV1 {
    std::uint32_t abiVersion =
        kHumanBehaviorAcceptedStateRootABIVersionV1;
    std::uint32_t structSize =
        sizeof(HumanBehaviorAcceptedStateRootDescriptorV1);
    std::uint32_t clockDomain =
        kHumanBehaviorAcceptedStateRootExactNanosecondClockDomain;
    std::uint32_t clockQuantumNanoseconds =
        kHumanBehaviorAcceptedStateRootClockQuantumNanoseconds;
    std::uint64_t acceptedTimestampNanoseconds = 0u;
    std::uint64_t reserved0 = 0u;
};

static_assert(sizeof(HumanBehaviorAcceptedStateComponentV1) == 56u);
static_assert(sizeof(HumanBehaviorAcceptedStateRootDescriptorV1) == 32u);

// Computes the canonical accepted-state-root.v1 SHA-256. Input order does not
// affect the result: roles are validated for exact one-of-each coverage and
// hashed in stable role order. Missing, zero, duplicate, copied-across-owner,
// malformed, non-SHA-256, or non-exact-clock input fails without changing
// output. The function consumes only cryptographic roots and positive exact
// generations; it has no overload accepting the runtime's 64-bit FNV replay
// fingerprints.
[[nodiscard]] bool composeHumanBehaviorAcceptedStateRootV1(
    const HumanBehaviorAcceptedStateRootDescriptorV1& descriptor,
    std::span<const HumanBehaviorAcceptedStateComponentV1> components,
    HumanBehaviorDigest& output,
    std::string& error
);

} // namespace metalrobo
