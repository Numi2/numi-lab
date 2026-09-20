#include "metalrobo/HumanBehaviorAcceptedStateRoot.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace metalrobo;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

HumanBehaviorDigest digest(const std::uint8_t seed) {
    HumanBehaviorDigest result{};
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(
            seed + static_cast<std::uint8_t>(17u * index) +
            static_cast<std::uint8_t>(index * index));
    }
    return result;
}

std::array<HumanBehaviorAcceptedStateComponentV1,
           kHumanBehaviorAcceptedStateRootComponentCountV1>
components() {
    std::array<HumanBehaviorAcceptedStateComponentV1,
               kHumanBehaviorAcceptedStateRootComponentCountV1>
        result{};
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index].role = static_cast<
            HumanBehaviorAcceptedStateComponentRoleV1>(index + 1u);
        result[index].generation = 100u + index * 11u;
        result[index].stateSHA256 = digest(
            static_cast<std::uint8_t>(0x21u + index * 19u));
    }
    return result;
}

HumanBehaviorAcceptedStateRootDescriptorV1 descriptor() {
    HumanBehaviorAcceptedStateRootDescriptorV1 result{};
    result.acceptedTimestampNanoseconds = 9'876'543'210u;
    return result;
}

HumanBehaviorDigest compose(
    const HumanBehaviorAcceptedStateRootDescriptorV1& rootDescriptor,
    const std::span<const HumanBehaviorAcceptedStateComponentV1> source
) {
    HumanBehaviorDigest result{};
    std::string error;
    require(composeHumanBehaviorAcceptedStateRootV1(
                rootDescriptor, source, result, error) && error.empty(),
            "valid accepted-state root composition failed");
    return result;
}

template <typename Components>
void requireDenied(
    const HumanBehaviorAcceptedStateRootDescriptorV1& rootDescriptor,
    const Components& source,
    std::uint32_t& negativeCases,
    const char* message
) {
    const auto sentinel = digest(0xe3u);
    auto output = sentinel;
    std::string error;
    require(!composeHumanBehaviorAcceptedStateRootV1(
                rootDescriptor,
                std::span<const HumanBehaviorAcceptedStateComponentV1>(
                    source.data(), source.size()),
                output, error) &&
                output == sentinel && !error.empty(),
            message);
    ++negativeCases;
}

} // namespace

int main() {
    try {
        const auto rootDescriptor = descriptor();
        const auto canonical = components();
        const auto baseline = compose(rootDescriptor, canonical);
        require(encodeHumanBehaviorDigest(baseline) ==
                    "93833642f7915f84d4dd8f0f93f03347"
                    "c76f247a82417e7bd09f0eb1025f849f",
                "accepted-state root canonical bytes changed");
        require(std::any_of(baseline.begin(), baseline.end(),
                    [](const auto byte) { return byte != 0u; }),
                "accepted-state root is zero");
        require(compose(rootDescriptor, canonical) == baseline,
                "accepted-state root is not deterministic");

        auto reordered = canonical;
        std::reverse(reordered.begin(), reordered.end());
        require(compose(rootDescriptor, reordered) == baseline,
                "accepted-state root depends on caller component order");

        auto changedTimestamp = rootDescriptor;
        ++changedTimestamp.acceptedTimestampNanoseconds;
        require(compose(changedTimestamp, canonical) != baseline,
                "accepted-state root omitted the exact timestamp");

        for (std::size_t index = 0u; index < canonical.size(); ++index) {
            auto changedGeneration = canonical;
            ++changedGeneration[index].generation;
            require(compose(rootDescriptor, changedGeneration) != baseline,
                    "accepted-state root omitted an owner generation");

            auto changedDigest = canonical;
            changedDigest[index].stateSHA256[index] ^= 0x80u;
            require(compose(rootDescriptor, changedDigest) != baseline,
                    "accepted-state root omitted an owner SHA-256");
        }

        std::uint32_t negativeCases = 0u;
        {
            std::vector<HumanBehaviorAcceptedStateComponentV1> missing(
                canonical.begin(), canonical.end() - 1);
            requireDenied(rootDescriptor, missing, negativeCases,
                "missing component was accepted or mutated output");
        }
        {
            auto duplicateRole = canonical;
            duplicateRole.back().role = duplicateRole.front().role;
            requireDenied(rootDescriptor, duplicateRole, negativeCases,
                "duplicate component role was accepted or mutated output");
        }
        {
            auto invalidRole = canonical;
            invalidRole.back().role = static_cast<
                HumanBehaviorAcceptedStateComponentRoleV1>(99u);
            requireDenied(rootDescriptor, invalidRole, negativeCases,
                "invalid component role was accepted or mutated output");
        }
        {
            auto zeroDigest = canonical;
            zeroDigest.front().stateSHA256.fill(0u);
            requireDenied(rootDescriptor, zeroDigest, negativeCases,
                "zero component digest was accepted or mutated output");
        }
        {
            auto zeroGeneration = canonical;
            zeroGeneration.front().generation = 0u;
            requireDenied(rootDescriptor, zeroGeneration, negativeCases,
                "zero component generation was accepted or mutated output");
        }
        {
            auto copiedDigest = canonical;
            copiedDigest[1].stateSHA256 = copiedDigest[0].stateSHA256;
            requireDenied(rootDescriptor, copiedDigest, negativeCases,
                "cross-owner copied digest was accepted or mutated output");
        }
        {
            auto invalidAlgorithm = canonical;
            invalidAlgorithm.front().digestAlgorithm = static_cast<
                HumanBehaviorAcceptedStateDigestAlgorithmV1>(2u);
            requireDenied(rootDescriptor, invalidAlgorithm, negativeCases,
                "non-SHA-256 component was accepted or mutated output");
        }
        {
            auto invalidVersion = canonical;
            ++invalidVersion.front().abiVersion;
            requireDenied(rootDescriptor, invalidVersion, negativeCases,
                "wrong component ABI was accepted or mutated output");
        }
        {
            auto invalidSize = canonical;
            --invalidSize.front().structSize;
            requireDenied(rootDescriptor, invalidSize, negativeCases,
                "wrong component size was accepted or mutated output");
        }
        {
            auto invalid = rootDescriptor;
            ++invalid.abiVersion;
            requireDenied(invalid, canonical, negativeCases,
                "wrong root ABI was accepted or mutated output");
        }
        {
            auto invalid = rootDescriptor;
            --invalid.structSize;
            requireDenied(invalid, canonical, negativeCases,
                "wrong root size was accepted or mutated output");
        }
        {
            auto invalid = rootDescriptor;
            invalid.clockDomain = 1u;
            requireDenied(invalid, canonical, negativeCases,
                "non-exact clock domain was accepted or mutated output");
        }
        {
            auto invalid = rootDescriptor;
            invalid.clockQuantumNanoseconds = 1000u;
            requireDenied(invalid, canonical, negativeCases,
                "non-exact clock quantum was accepted or mutated output");
        }
        {
            auto invalid = rootDescriptor;
            invalid.acceptedTimestampNanoseconds = 0u;
            requireDenied(invalid, canonical, negativeCases,
                "zero accepted timestamp was accepted or mutated output");
        }
        {
            auto invalid = rootDescriptor;
            invalid.reserved0 = 1u;
            requireDenied(invalid, canonical, negativeCases,
                "nonzero root reserved field was accepted or mutated output");
        }

        std::cout
            << "human_behavior_accepted_state_root=passed components="
            << canonical.size()
            << " owner_generation_mutations=" << canonical.size()
            << " owner_digest_mutations=" << canonical.size()
            << " negative_cases=" << negativeCases
            << " root_sha256=" << encodeHumanBehaviorDigest(baseline)
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
