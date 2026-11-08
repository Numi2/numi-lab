#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

// Physical impulse residuals, in N s. Keep the largest local residual and
// signed aggregate separately: cancellation cannot certify local equilibrium.
struct MidpointBalance {
    double maxLocal = 0;
    std::array<double, 3> signedSum{};
    void add(const std::array<double, 3>& momentumChange,
             const std::array<double, 3>& appliedImpulse) {
        std::array<double, 3> residual{};
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(momentumChange[axis]) ||
                !std::isfinite(appliedImpulse[axis]))
                throw std::runtime_error("nonfinite midpoint momentum accounting");
            residual[axis] = momentumChange[axis] - appliedImpulse[axis];
            signedSum[axis] += residual[axis];
            if (!std::isfinite(residual[axis]) || !std::isfinite(signedSum[axis]))
                throw std::runtime_error("overflow in midpoint momentum accounting");
        }
        const double local = std::hypot(residual[0], residual[1], residual[2]);
        if (!std::isfinite(local)) throw std::runtime_error("nonfinite local momentum residual");
        maxLocal = std::max(maxLocal, local);
    }
};
