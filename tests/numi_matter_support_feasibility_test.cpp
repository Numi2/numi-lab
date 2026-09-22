#include "numi/matter/support_feasibility.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using numi_matter_contact::normalImpulseFeasible;
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<float, 11> impulses{
        0.0f, -0.0f, 1.0f, std::numeric_limits<float>::max(),
        -1.0e-7f, std::nextafter(-1.0e-7f, 0.0f),
        std::nextafter(-1.0e-7f, -infinity),
        -0.000127762556f, infinity, -infinity,
        std::numeric_limits<float>::quiet_NaN()};
    const std::array<bool, 11> expected{
        true, true, true, true, true, true,
        false, false, false, false, false};
    for (std::size_t i = 0; i < impulses.size(); ++i) {
        const bool legacy = std::isfinite(impulses[i]) && impulses[i] >= -1.0e-7f;
        if (normalImpulseFeasible(impulses[i]) != expected[i] || legacy != expected[i])
            throw std::runtime_error("Final support admissibility predicate changed");
    }
    // This is a stopping-policy regression, not a replay of the native root.
    // A residual can meet its own tolerance while an impulse still fails
    // its separate physical predicate. The solver must continue correcting.
    const float residual = 0.0002f, tolerance = 0.001f;
    const bool oldStop = residual <= tolerance;
    const bool newStop = oldStop && normalImpulseFeasible(-0.000127762556f);
    if (!oldStop || newStop)
        throw std::runtime_error("Infeasible residual-small iterate stopped");
    if (!(oldStop && normalImpulseFeasible(0.0f)))
        throw std::runtime_error("Feasible residual-small iterate did not stop");
    std::cout << "PASS: 11 boundary/finite cases and two stopping-policy cases\n";
}
