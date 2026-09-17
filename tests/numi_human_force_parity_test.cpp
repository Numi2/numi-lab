#include "metalrobo/NumiHumanForceParity.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

bool close(const double left, const double right) {
    return std::abs(left - right) <= 1.0e-12 *
        (1.0 + std::max(std::abs(left), std::abs(right)));
}

} // namespace

int main() {
    {
        const std::vector<double> reference{0.0, 1.0, -2.0};
        const std::vector<double> candidate{0.0, 1.0, -2.0};
        const auto result = metalrobo::compareNumiHumanUnconstrainedVelocityParity(
            reference, candidate, 1.0e-4
        );
        require(result.valid, "identical parity input was rejected");
        require(result.maximumVelocityDeltaDof == 0u,
                "identical parity owner is not deterministic");
        require(result.maximumVelocityDelta == 0.0,
                "identical parity has velocity error");
        require(result.maximumAccelerationDelta == 0.0,
                "identical parity has acceleration error");
    }
    {
        const std::vector<double> reference{0.0, 0.25, -0.5, 0.0};
        const std::vector<double> candidate{1.0e-6, 0.250004, -0.500002, 0.0};
        const auto result = metalrobo::compareNumiHumanUnconstrainedVelocityParity(
            reference, candidate, 2.0e-5
        );
        require(result.valid, "finite parity input was rejected");
        require(result.maximumVelocityDeltaDof == 1u,
                "wrong maximum velocity-parity owner");
        require(close(result.maximumVelocityDelta, 4.0e-6),
                "wrong maximum velocity-parity delta");
        require(close(result.maximumAccelerationDelta, 0.2),
                "wrong acceleration-space parity delta");
    }
    {
        const std::vector<double> first{0.0};
        const std::vector<double> second{0.0, 1.0};
        require(!metalrobo::compareNumiHumanUnconstrainedVelocityParity(
                    first, second, 1.0e-4).valid,
                "dimension mismatch was admitted");
        require(!metalrobo::compareNumiHumanUnconstrainedVelocityParity(
                    first, first, 0.0).valid,
                "zero timestep was admitted");
        require(!metalrobo::compareNumiHumanUnconstrainedVelocityParity(
                    first, first, std::numeric_limits<double>::infinity()).valid,
                "non-finite timestep was admitted");
    }
    {
        const std::vector<double> reference{0.0, 1.0};
        const std::vector<double> candidate{
            0.0, std::numeric_limits<double>::quiet_NaN()
        };
        require(!metalrobo::compareNumiHumanUnconstrainedVelocityParity(
                    reference, candidate, 1.0e-4).valid,
                "non-finite velocity was admitted");
    }

    std::cout << "Human force parity: 13 checks passed\n";
    return 0;
}
