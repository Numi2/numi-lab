#include "metalrobo/numi_human_stand_gpu.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    std::size_t checks = 0u;
    const auto require = [&](bool value, const char* message) {
        ++checks;
        if (!value) throw std::runtime_error(message);
    };
    MRNumiHumanTimedRootForceGPU pulse{};
    require(!mrNumiHumanTimedRootForceConfigured(pulse), "default pulse enabled");
    require(mrNumiHumanTimedRootForceValid(pulse, 100u), "default pulse invalid");
    require(!mrNumiHumanTimedRootForceActive(pulse, 0u), "default pulse applies force");
    pulse.stepWindow = {11u, 7u, 0u, 0u};
    pulse.forceNewtons = {12.0f, -3.0f, 1.5f, 0.0f};
    require(mrNumiHumanTimedRootForceValid(pulse, 64u), "fixed pulse invalid");
    require(!mrNumiHumanTimedRootForceActive(pulse, 10u), "pulse begins early");
    require(mrNumiHumanTimedRootForceActive(pulse, 11u), "pulse misses inclusive start");
    require(mrNumiHumanTimedRootForceActive(pulse, 17u), "pulse misses final active step");
    require(!mrNumiHumanTimedRootForceActive(pulse, 18u), "pulse includes exclusive end");
    require(mrNumiHumanTimedRootForceValid(pulse, 18u), "window ending at horizon rejected");
    require(!mrNumiHumanTimedRootForceValid(pulse, 17u), "window exceeds horizon");

    auto malformed = pulse;
    malformed.stepWindow.x = std::numeric_limits<std::uint32_t>::max() - 2u;
    require(!mrNumiHumanTimedRootForceValid(malformed,
        std::numeric_limits<std::uint32_t>::max()), "overflowing window admitted");
    malformed = pulse; malformed.stepWindow.y = 0u;
    require(!mrNumiHumanTimedRootForceValid(malformed, 64u), "noncanonical disabled pulse admitted");
    malformed = pulse; malformed.stepWindow.z = 1u;
    require(!mrNumiHumanTimedRootForceValid(malformed, 64u), "reserved step word admitted");
    malformed = pulse; malformed.stepWindow.w = 1u;
    require(!mrNumiHumanTimedRootForceValid(malformed, 64u), "reserved second step word admitted");
    malformed = pulse; malformed.forceNewtons.w = 1.0f;
    require(!mrNumiHumanTimedRootForceValid(malformed, 64u), "reserved force word admitted");
    for (unsigned axis = 0u; axis < 3u; ++axis) {
        malformed = pulse;
        if (axis == 0u) malformed.forceNewtons.x = std::numeric_limits<float>::quiet_NaN();
        if (axis == 1u) malformed.forceNewtons.y = std::numeric_limits<float>::infinity();
        if (axis == 2u) malformed.forceNewtons.z = -std::numeric_limits<float>::infinity();
        require(!mrNumiHumanTimedRootForceValid(malformed, 64u), "nonfinite force admitted");
    }

    // Scheduling contract only: exact force impulse must be unchanged when
    // an episode is split across submission boundaries.
    // This is not a simulated physical trajectory or recovery qualification.
    constexpr double dt = 0.001;
    const std::array<unsigned, 7u> partitions{3u, 8u, 1u, 4u, 2u, 19u, 27u};
    unsigned acceptedStep = 0u, activeSteps = 0u;
    for (const unsigned count : partitions) {
        for (unsigned localStep = 0u; localStep < count; ++localStep) {
            const bool candidateActive = mrNumiHumanTimedRootForceActive(pulse, acceptedStep);
            activeSteps += candidateActive ? 1u : 0u;
            ++acceptedStep;
        }
    }
    require(acceptedStep == 64u && activeSteps == 7u, "split schedule changed pulse duration");
    require(std::abs(activeSteps * dt * pulse.forceNewtons.x - 0.084) < 1.0e-15,
        "prescribed x impulse differs from source N*s");
    require(std::abs(activeSteps * dt * pulse.forceNewtons.y + 0.021) < 1.0e-15,
        "prescribed y impulse differs from source N*s");
    require(std::abs(activeSteps * dt * pulse.forceNewtons.z - 0.0105) < 1.0e-15,
        "prescribed z impulse differs from source N*s");
    std::cout << "timed root force CPU contract checks=" << checks << '\n';
}
