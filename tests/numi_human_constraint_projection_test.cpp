#include "metalrobo/numi_human_constraint_projection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
bool close(float a, float b, float tolerance = 2.0e-6f) {
    return std::abs(a-b) <= tolerance;
}

// A two-row SPD Delassus operator. The same accumulated-impulse policy used
// by the native limit owner is exercised, not a second implementation of it.
std::array<float, 4> solve(float coupling, std::array<float, 2> freeVelocity) {
    auto velocity = freeVelocity;
    std::array<float, 2> lambda{};
    for (int sweep = 0; sweep < 256; ++sweep) {
        for (int row = 0; row < 2; ++row) {
            const float next = mrNumiHumanProjectIntervalImpulse(
                lambda[row], velocity[row], 0.0f, 100.0f, 1.0f);
            const float delta = next - lambda[row];
            lambda[row] = next;
            velocity[row] += delta;
            velocity[1-row] += coupling * delta;
        }
    }
    for (int row = 0; row < 2; ++row) {
        require(velocity[row] >= -1.0e-5f, "primal constraint residual");
        require(lambda[row] >= 0.0f, "negative unilateral impulse");
        require(std::abs(velocity[row]*lambda[row]) < 1.0e-4f,
                "unilateral complementarity residual");
        require(close(velocity[row], freeVelocity[row]+lambda[row]+
                      coupling*lambda[1-row], 1.0e-5f),
                "impulse/velocity ownership mismatch");
    }
    return {velocity[0], velocity[1], lambda[0], lambda[1]};
}
}

int main() {
    try {
        require(close(mrNumiHumanProjectIntervalImpulse(1, 2, 0, 100, 1), 0),
                "lower stop cannot release an earlier impulse");
        require(close(mrNumiHumanProjectIntervalImpulse(-1, -2, -100, 0, 1), 0),
                "upper stop cannot release an earlier impulse");
        require(close(mrNumiHumanProjectIntervalImpulse(0, -2, 0, 100, 2), 1),
                "lower stop response sign");
        require(close(mrNumiHumanProjectIntervalImpulse(0, 2, -100, 0, 2), -1),
                "upper stop response sign");
        require(close(mrNumiHumanProjectIntervalImpulse(1, 0.5f, 0, 100, 1), 0.5f),
                "lower stop cannot decrease a positive impulse");
        require(close(mrNumiHumanProjectIntervalImpulse(-1, -0.5f, -100, 0, 1), -0.5f),
                "upper stop cannot decrease a negative impulse magnitude");
        const auto released = solve(0.9f, {-1, -2});
        require(close(released[2], 0) && close(released[3], 2),
                "coupled inactive stop kept an artificial reaction");
        const auto activated = solve(-0.9f, {0.5f, -2});
        require(activated[2] > 0 && activated[3] > 0,
                "row activated by another constraint was omitted");
        require(close(activated[2], 1.3f/0.19f, 3.0e-5f),
                "coupled solution disagrees with analytic solve");
        require(close(mrNumiHumanLowerLimitVelocityTarget(0.1f, 0, 0.01f), -10),
                "interior lower bound wrongly freezes the coordinate");
        require(close(mrNumiHumanUpperLimitVelocityTarget(0.9f, 1, 0.01f), (1.0f-0.9f)/0.01f),
                "interior upper bound wrongly freezes the coordinate");
        const float zeroScaleSlop = mrNumiHumanPositionLimitSlop(0.0f, 0.0f);
        require(zeroScaleSlop > 1.9e-6f && zeroScaleSlop < 2.0e-6f,
                "position-limit slop scale changed");
        for (const float dt : {1.0e-4f, 5.0e-5f, 2.5e-5f, 1.25e-5f}) {
            require(mrNumiHumanLowerLimitVelocityTarget(
                        -0.5f * zeroScaleSlop, 0.0f, dt) == 0.0f,
                    "sub-ULP lower-limit drift became a timestep-amplified correction");
            require(mrNumiHumanUpperLimitVelocityTarget(
                        1.0f + 0.5f * mrNumiHumanPositionLimitSlop(1.0f, 1.0f),
                        1.0f, dt) == 0.0f,
                    "sub-ULP upper-limit drift became a timestep-amplified correction");
        }
        const float expectedLowerCorrection =
            0.2f * (0.001f - mrNumiHumanPositionLimitSlop(-0.001f, 0.0f)) /
            0.01f;
        const float expectedUpperCorrection =
            -0.2f * (0.001f - mrNumiHumanPositionLimitSlop(1.001f, 1.0f)) /
            0.01f;
        require(close(mrNumiHumanLowerLimitVelocityTarget(-0.001f, 0, 0.01f),
                      expectedLowerCorrection),
                "penetration stabilization outside the representational band changed");
        require(close(mrNumiHumanUpperLimitVelocityTarget(1.001f, 1, 0.01f),
                      expectedUpperCorrection),
                "upper penetration stabilization outside the representational band changed");
        for (const float dt : {1.0e-4f, 5.0e-5f, 2.5e-5f, 1.25e-5f}) {
            const float force = 981.0f;
            const float response = 0.01f;
            const float seed = mrNumiHumanSupportSeedImpulse(force, dt, 0, 0.002f);
            const float free = -9.81f*dt;
            const float warm = free + response*seed;
            const float next = mrNumiHumanProjectIntervalImpulse(
                seed, warm, 0, 100, response);
            const float final = warm + response*(next-seed);
            require(std::abs(final) < 1.0e-8f, "body-weight contact not equilibrated");
            require(close(next/dt, force, 1.0e-3f), "impulse/force timestep conversion");
            const float separatedTarget = mrNumiHumanContactVelocityTarget(0.001f, dt, 0.2f);
            const float separatedNext = mrNumiHumanProjectIntervalImpulse(
                seed, warm, separatedTarget, 1000, response);
            require(separatedNext == 0, "warm start supports a separated body");
            require(close(warm+response*(separatedNext-seed), free),
                    "retracted preload did not restore free motion");
            require(mrNumiHumanSupportSeedImpulse(force, dt, 0.003f, 0.002f) == 0,
                    "inactive contact retained a seed");
            // Reinitialize every step; a previous solve's multiplier is never
            // mistaken for an impulse already applied to a new free velocity.
            require(seed == mrNumiHumanSupportSeedImpulse(force, dt, 0, 0.002f),
                    "seed has hidden history");
        }
        std::cout << "Human constraint projection: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
