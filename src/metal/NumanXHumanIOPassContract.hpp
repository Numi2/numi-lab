#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"

namespace metalrobo::detail {

// HumanIO consumes the complete ABI-6 prefix and no later extension.
[[nodiscard]] constexpr bool validHumanIOTransactionPassHeader(
    std::uint32_t abiVersion, std::uint32_t structSize,
    std::uint32_t reserved0
) noexcept {
    return abiVersion == kMetalNumanXTransactionABIVersion &&
        structSize == metalNumanXTransactionPassStructSizeForABI(
            kMetalNumanXTransactionABIVersion) &&
        reserved0 == 0u;
}
} // namespace metalrobo::detail
