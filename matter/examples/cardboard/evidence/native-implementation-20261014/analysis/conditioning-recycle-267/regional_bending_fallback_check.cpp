#include "numi/matter/regional_galerkin_bending_basis.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace numi::matter;
    const auto coordinates = regionalBendingCoordinates(
        {-1.0, -2.0, -0.1}, {1.0, 2.0, 0.1});
    if (!coordinates.available) return 1;
    const std::array<double, 3> point{0.25, 0.5, 0.05};
    const auto fallback = regionalBendingTrialField(coordinates, point, 8u);
    if (!std::isfinite(fallback[0]) || !std::isfinite(fallback[1]) ||
        !std::isfinite(fallback[2])) return 2;
    if (std::abs(fallback[coordinates.normalAxis]) < 1.0e-6) return 3;
    const unsigned xAxis = coordinates.planeAxes[0];
    const unsigned yAxis = coordinates.planeAxes[1];
    const double y = (point[yAxis] - coordinates.center[yAxis]) /
        coordinates.halfExtent[yAxis];
    const double z = point[coordinates.normalAxis] -
        coordinates.center[coordinates.normalAxis];
    const double expectedNormal = y * y * y * y;
    const double expectedTangential = -z * 4.0 * y * y * y /
        coordinates.halfExtent[yAxis];
    if (std::abs(fallback[coordinates.normalAxis] - expectedNormal) > 1e-14 ||
        std::abs(fallback[yAxis] - expectedTangential) > 1e-14 ||
        std::abs(fallback[xAxis]) > 1e-14) return 5;
    try {
        (void)regionalBendingTrialField(coordinates, point, 9u);
        return 4;
    } catch (const std::invalid_argument&) {
    }
    std::cout << "static y^4 fallback field check passed\n";
    return 0;
}
