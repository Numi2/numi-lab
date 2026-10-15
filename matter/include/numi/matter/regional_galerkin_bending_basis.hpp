#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace numi::matter {

// Trial fields for an opt-in coarse preconditioner only. They do not replace
// volume elements, impose kinematics, or change the production operator.
struct RegionalBendingCoordinates {
    std::array<double, 3> center{};
    std::array<double, 3> halfExtent{};
    std::array<unsigned, 2> planeAxes{};
    unsigned normalAxis = 0u;
    bool available = false;
};

inline RegionalBendingCoordinates regionalBendingCoordinates(
    const std::array<double, 3>& minimum,
    const std::array<double, 3>& maximum) {
    RegionalBendingCoordinates result;
    std::array<unsigned, 3> order{0u, 1u, 2u};
    for (unsigned axis = 0u; axis < 3u; ++axis) {
        if (!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis]) ||
            maximum[axis] < minimum[axis]) return result;
        result.center[axis] = 0.5 * (minimum[axis] + maximum[axis]);
        result.halfExtent[axis] = 0.5 * (maximum[axis] - minimum[axis]);
    }
    std::stable_sort(order.begin(), order.end(), [&](unsigned left, unsigned right) {
        return result.halfExtent[left] > result.halfExtent[right];
    });
    result.planeAxes = {order[0], order[1]};
    result.normalAxis = order[2];
    result.available = result.halfExtent[order[1]] > 1.0e-15;
    return result;
}

inline std::array<double, 3> regionalBendingTrialField(
    const RegionalBendingCoordinates& coordinates,
    const std::array<double, 3>& position,
    unsigned mode) {
    if (!coordinates.available || mode >= 9u)
        throw std::invalid_argument("regional bending trial requires valid plane extents and mode0..8");
    constexpr std::array<std::array<unsigned, 2>, 9> powers{{
        {2u, 0u}, {3u, 0u}, {0u, 2u}, {1u, 1u},
        {2u, 1u}, {1u, 2u}, {0u, 3u}, {4u, 0u},
        {0u, 4u}}};
    const unsigned xAxis = coordinates.planeAxes[0];
    const unsigned yAxis = coordinates.planeAxes[1];
    const double x = (position[xAxis] - coordinates.center[xAxis]) /
        coordinates.halfExtent[xAxis];
    const double y = (position[yAxis] - coordinates.center[yAxis]) /
        coordinates.halfExtent[yAxis];
    const double z = position[coordinates.normalAxis] -
        coordinates.center[coordinates.normalAxis];
    const auto integerPower = [](double base, unsigned exponent) {
        double value = 1.0;
        for (unsigned index = 0u; index < exponent; ++index) value *= base;
        return value;
    };
    const unsigned px = powers[mode][0];
    const unsigned py = powers[mode][1];
    const double w = integerPower(x, px) * integerPower(y, py);
    const double dx = px == 0u ? 0.0 :
        double(px) * integerPower(x, px - 1u) * integerPower(y, py) /
        coordinates.halfExtent[xAxis];
    const double dy = py == 0u ? 0.0 :
        double(py) * integerPower(x, px) * integerPower(y, py - 1u) /
        coordinates.halfExtent[yAxis];
    std::array<double, 3> value{};
    value[coordinates.normalAxis] = w;
    value[xAxis] = -z * dx;
    value[yAxis] = -z * dy;
    return value;
}

} // namespace numi::matter
