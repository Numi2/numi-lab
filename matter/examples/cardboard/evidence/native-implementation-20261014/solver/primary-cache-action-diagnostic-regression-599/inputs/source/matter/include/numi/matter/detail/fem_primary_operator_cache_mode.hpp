#pragma once
#include <cstdint>
namespace numi::matter::detail {
// One shared policy for primary Arnoldi, true residual, center action and FD.
// RAP mode2 and CGLS transpose/forward modes remain separate algorithms.
constexpr std::uint32_t femPrimaryOperatorCacheMode(
    const std::uint32_t executionBudgetFlag) noexcept {
    return executionBudgetFlag == 1u ? 1u : 0u;
}
}
