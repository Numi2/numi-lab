#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace numi::matter::detail {

[[nodiscard]] inline bool scaleGravityFromBase(
    const std::array<float, 3>& base,
    const float scale,
    std::array<float, 3>& result
) noexcept {
    if (!std::isfinite(scale) || scale < 0.0f || scale > 1.0f) return false;
    std::array<float, 3> candidate{};
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        if (!std::isfinite(base[i])) return false;
        candidate[i] = base[i] * scale;
        if (!std::isfinite(candidate[i])) return false;
    }
    result = candidate;
    return true;
}

} // namespace numi::matter::detail
