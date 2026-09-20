#include "metalrobo/HumanBehaviorAcceptedStateRoot.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <string_view>

namespace metalrobo {
namespace {

constexpr std::string_view kComponentDomain =
    "numi.human.accepted-state-component.v1";
constexpr std::string_view kRootDomain =
    "numi.human.accepted-state-root.v1";

[[nodiscard]] bool fail(std::string& error, const char* message) {
    error = message;
    return false;
}

class SHA256Writer final {
public:
    SHA256Writer() noexcept : valid_(CC_SHA256_Init(&context_) == 1) {}

    [[nodiscard]] bool append(const void* bytes, std::size_t count) noexcept {
        if (!valid_ || (bytes == nullptr && count != 0u)) return false;
        const auto* cursor = static_cast<const std::uint8_t*>(bytes);
        while (count != 0u) {
            const auto chunk = static_cast<CC_LONG>(std::min<std::size_t>(
                count, std::numeric_limits<CC_LONG>::max()));
            if (CC_SHA256_Update(&context_, cursor, chunk) != 1) {
                valid_ = false;
                return false;
            }
            cursor += chunk;
            count -= chunk;
        }
        return true;
    }

    [[nodiscard]] bool appendU32(const std::uint32_t value) noexcept {
        std::array<std::uint8_t, 4u> bytes{};
        for (std::size_t index = 0u; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(
                (value >> (index * 8u)) & 0xffu);
        }
        return append(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool appendU64(const std::uint64_t value) noexcept {
        std::array<std::uint8_t, 8u> bytes{};
        for (std::size_t index = 0u; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(
                (value >> (index * 8u)) & 0xffu);
        }
        return append(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool appendDomain(const std::string_view domain) noexcept {
        return domain.size() <= std::numeric_limits<std::uint32_t>::max() &&
            appendU32(static_cast<std::uint32_t>(domain.size())) &&
            append(domain.data(), domain.size());
    }

    [[nodiscard]] bool finish(HumanBehaviorDigest& output) noexcept {
        if (!valid_ || CC_SHA256_Final(output.data(), &context_) != 1) {
            output.fill(0u);
            valid_ = false;
            return false;
        }
        valid_ = false;
        return true;
    }

private:
    CC_SHA256_CTX context_{};
    bool valid_ = false;
};

[[nodiscard]] bool digestPresent(
    const HumanBehaviorDigest& digest
) noexcept {
    return std::any_of(digest.begin(), digest.end(), [](const auto byte) {
        return byte != 0u;
    });
}

[[nodiscard]] std::size_t roleIndex(
    const HumanBehaviorAcceptedStateComponentRoleV1 role
) noexcept {
    const auto value = static_cast<std::uint32_t>(role);
    if (value < static_cast<std::uint32_t>(
                    HumanBehaviorAcceptedStateComponentRoleV1::physical) ||
        value > static_cast<std::uint32_t>(
                    HumanBehaviorAcceptedStateComponentRoleV1::checkpoint)) {
        return kHumanBehaviorAcceptedStateRootComponentCountV1;
    }
    return static_cast<std::size_t>(value - 1u);
}

[[nodiscard]] bool composeLeaf(
    const HumanBehaviorAcceptedStateComponentV1& component,
    HumanBehaviorDigest& output
) noexcept {
    SHA256Writer writer;
    return writer.appendDomain(kComponentDomain) &&
        writer.appendU32(kHumanBehaviorAcceptedStateRootABIVersionV1) &&
        writer.appendU32(static_cast<std::uint32_t>(component.role)) &&
        writer.appendU32(static_cast<std::uint32_t>(
            component.digestAlgorithm)) &&
        writer.appendU64(component.generation) &&
        writer.append(component.stateSHA256.data(),
                      component.stateSHA256.size()) &&
        writer.finish(output);
}

} // namespace

bool composeHumanBehaviorAcceptedStateRootV1(
    const HumanBehaviorAcceptedStateRootDescriptorV1& descriptor,
    const std::span<const HumanBehaviorAcceptedStateComponentV1> components,
    HumanBehaviorDigest& output,
    std::string& error
) {
    try {
        if (descriptor.abiVersion !=
                kHumanBehaviorAcceptedStateRootABIVersionV1 ||
            descriptor.structSize != sizeof(descriptor) ||
            descriptor.clockDomain !=
                kHumanBehaviorAcceptedStateRootExactNanosecondClockDomain ||
            descriptor.clockQuantumNanoseconds !=
                kHumanBehaviorAcceptedStateRootClockQuantumNanoseconds ||
            descriptor.acceptedTimestampNanoseconds == 0u ||
            descriptor.reserved0 != 0u) {
            return fail(error,
                "accepted-state root descriptor is invalid or not exact-nanosecond");
        }
        if (components.size() !=
            kHumanBehaviorAcceptedStateRootComponentCountV1) {
            return fail(error,
                "accepted-state root requires exactly seven owner components");
        }

        std::array<const HumanBehaviorAcceptedStateComponentV1*,
                   kHumanBehaviorAcceptedStateRootComponentCountV1>
            canonical{};
        for (const auto& component : components) {
            if (component.abiVersion !=
                    kHumanBehaviorAcceptedStateRootABIVersionV1 ||
                component.structSize != sizeof(component) ||
                component.digestAlgorithm !=
                    HumanBehaviorAcceptedStateDigestAlgorithmV1::sha256) {
                return fail(error,
                    "accepted-state component version or digest algorithm is invalid");
            }
            const std::size_t index = roleIndex(component.role);
            if (index >= canonical.size()) {
                return fail(error, "accepted-state component role is invalid");
            }
            if (canonical[index] != nullptr) {
                return fail(error, "accepted-state component role is duplicated");
            }
            if (!digestPresent(component.stateSHA256)) {
                return fail(error, "accepted-state component SHA-256 is zero");
            }
            if (component.generation == 0u) {
                return fail(error,
                    "accepted-state component generation is zero");
            }
            canonical[index] = &component;
        }
        if (std::any_of(canonical.begin(), canonical.end(),
                [](const auto* component) { return component == nullptr; })) {
            return fail(error, "accepted-state component role is missing");
        }
        for (std::size_t left = 0u; left < canonical.size(); ++left) {
            for (std::size_t right = left + 1u;
                 right < canonical.size(); ++right) {
                if (canonical[left]->stateSHA256 ==
                    canonical[right]->stateSHA256) {
                    return fail(error,
                        "accepted-state owner SHA-256 is duplicated across roles");
                }
            }
        }

        std::array<HumanBehaviorDigest,
                   kHumanBehaviorAcceptedStateRootComponentCountV1>
            leaves{};
        for (std::size_t index = 0u; index < canonical.size(); ++index) {
            if (!composeLeaf(*canonical[index], leaves[index]) ||
                !digestPresent(leaves[index])) {
                return fail(error,
                    "accepted-state component SHA-256 composition failed");
            }
        }

        HumanBehaviorDigest composed{};
        SHA256Writer writer;
        if (!writer.appendDomain(kRootDomain) ||
            !writer.appendU32(
                kHumanBehaviorAcceptedStateRootABIVersionV1) ||
            !writer.appendU32(descriptor.clockDomain) ||
            !writer.appendU32(descriptor.clockQuantumNanoseconds) ||
            !writer.appendU32(static_cast<std::uint32_t>(canonical.size())) ||
            !writer.appendU64(descriptor.acceptedTimestampNanoseconds)) {
            return fail(error, "accepted-state root SHA-256 initialization failed");
        }
        for (std::size_t index = 0u; index < canonical.size(); ++index) {
            const auto& component = *canonical[index];
            if (!writer.appendU32(static_cast<std::uint32_t>(component.role)) ||
                !writer.appendU32(static_cast<std::uint32_t>(
                    component.digestAlgorithm)) ||
                !writer.appendU64(component.generation) ||
                !writer.append(leaves[index].data(), leaves[index].size())) {
                return fail(error, "accepted-state root SHA-256 update failed");
            }
        }
        if (!writer.finish(composed) || !digestPresent(composed)) {
            return fail(error, "accepted-state root SHA-256 finalization failed");
        }

        output = composed;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "accepted-state root composition failed";
        return false;
    }
}

} // namespace metalrobo
