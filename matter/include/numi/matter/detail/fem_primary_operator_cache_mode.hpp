#pragma once
#include <cstdint>
namespace numi::matter::detail {
// One shared policy for primary Arnoldi and its direct-fallback variant.
// Cache mode 1 requires every participating element; mode 2 reuses valid
// entries while evaluating unsupported elements through the direct operator.
constexpr std::uint32_t femPrimaryOperatorCacheMode(
    const std::uint32_t executionBudgetFlag) noexcept {
    return executionBudgetFlag == 1u || executionBudgetFlag == 2u
        ? executionBudgetFlag : 0u;
}
}
