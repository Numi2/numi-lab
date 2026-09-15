#include "metalrobo/numi_human_constraint_projection.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    int checks = 0;
    const auto require = [&checks](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(message);
    };
    try {
        // Correlations include the published problematic Human trace pair.
        for (float b : {-0.9f, -0.99997094619206084f, 0.99999290632891391f}) {
            const float ev = 1.0e-4f, lv = 2.0e-4f;
            const auto block = mrNumiHumanProjectEqualityLimitBlock(
                0, ev, lv, 0, -100, 0, 1, b, b, 1);
            require(block.valid, "positive definite equality/limit pair rejected");
            const long double bb = b;
            const long double expectedLimit = -(static_cast<long double>(lv)-bb*ev)/(1-bb*bb);
            const long double expectedEquality = -ev-bb*expectedLimit;
            require(std::abs(block.limitImpulse-expectedLimit) <=
                        2.0e-5L*std::abs(expectedLimit), "block limit disagrees with FP64-or-better oracle");
            require(std::abs(block.equalityDelta-expectedEquality) <=
                        2.0e-5L*std::abs(expectedEquality), "block equality disagrees with oracle");
            const float finalE = std::fma(b, block.limitImpulse, ev+block.equalityDelta);
            const float finalL = std::fma(b, block.equalityDelta, lv+block.limitImpulse);
            require(std::abs(finalE) < 2.0e-6f && std::abs(finalL) < 2.0e-6f,
                    "local coupled constraint residual");
            require(block.limitImpulse <= 0, "upper-limit impulse sign");
        }
        const auto release = mrNumiHumanProjectEqualityLimitBlock(
            -0.1f, 0, -2, 0, -100, 0, 1, -0.9f, -0.9f, 1);
        require(release.valid && release.limitImpulse == 0, "paired limit cannot unload");
        require(std::abs(release.equalityDelta-0.09f) < 1e-6f,
                "equality not retained while limit unloads");
        const auto lower = mrNumiHumanProjectEqualityLimitBlock(
            0, 0, -1, 0, 0, 100, 2, 0.5f, 0.5f, 3);
        require(lower.valid && lower.limitImpulse > 0, "lower-limit pair sign");
        const auto dependent = mrNumiHumanProjectEqualityLimitBlock(
            0, 0, 0, 0, -1, 1, 1, 1, 1, 1);
        require(!dependent.valid, "rank-deficient pair silently regularized");
        const auto indefinite = mrNumiHumanProjectEqualityLimitBlock(
            0, 0, 0, 0, -1, 1, 1, 2, 2, 1);
        require(!indefinite.valid, "indefinite pair accepted");
        std::cout << "Human equality-limit block: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
