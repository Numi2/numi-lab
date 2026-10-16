#include "cardboard_mixed_cohesive_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using cardboard::mixed_cohesive_reference::History;
using cardboard::mixed_cohesive_reference::Law;
using cardboard::mixed_cohesive_reference::Response;

[[noreturn]] void fail(const std::string_view message) {
    std::cerr << "cardboard mixed cohesive check failed: " << message << '\n';
    std::exit(1);
}

void require(const bool condition, const std::string_view message) {
    if (!condition) fail(message);
}

bool near(const double a, const double b, const double absolute,
          const double relative = 0.0) {
    return std::abs(a - b) <= absolute + relative *
        std::max(std::abs(a), std::abs(b));
}

Response evaluate(const Law& law, const std::array<double, 3>& acceptedGap,
                  const std::array<double, 3>& trialGap,
                  const History& history = {}) {
    Response result;
    require(cardboard::mixed_cohesive_reference::evaluate(
        law, acceptedGap, trialGap, history, result),
        "oracle rejected a valid synthetic point");
    require(result.valid, "valid response marker was not set");
    return result;
}

void checkTangent(const Law& law, const std::array<double, 3>& acceptedGap,
                  const std::array<double, 3>& trialGap,
                  const History& history, const double step,
                  const double tolerance, const std::string_view label) {
    const Response center = evaluate(law, acceptedGap, trialGap, history);
    double worstRelative = 0.0;
    for (std::size_t column = 0; column < 3; ++column) {
        auto plusGap = trialGap;
        auto minusGap = trialGap;
        plusGap[column] += step;
        minusGap[column] -= step;
        const Response plus = evaluate(law, acceptedGap, plusGap, history);
        const Response minus = evaluate(law, acceptedGap, minusGap, history);
        for (std::size_t row = 0; row < 3; ++row) {
            const double fd = (plus.tractionPa[row] - minus.tractionPa[row]) /
                (2.0 * step);
            const double analytic = center.tangentPaPerM[row][column];
            const double relative = std::abs(fd - analytic) /
                std::max({1.0, std::abs(fd), std::abs(analytic)});
            worstRelative = std::max(worstRelative, relative);
        }
    }
    if (!(worstRelative <= tolerance)) {
        std::cerr << label << " tangent relative error=" << worstRelative << '\n';
        fail("directional algorithmic tangent disagrees with centered finite differences");
    }
}

} // namespace

int main() {
    // Synthetic values selected for conditioning of the oracle checks. They
    // are not measured starch parameters and do not constitute calibration.
    const Law law{
        .normalPenaltyPaPerM = 1000.0,
        .shearPenaltyPaPerM = 400.0,
        .compressionPenaltyPaPerM = 1200.0,
        .normalStrengthPa = 10.0,
        .shearStrengthPa = 20.0,
        .modeIFractureEnergyJPerM2 = 1.0,
        .modeIIFractureEnergyJPerM2 = 2.0,
        .bkExponent = 1.5,
    };

    const auto elastic = evaluate(law, {0.0, 0.0, 0.0},
        {0.004, 0.002, -0.001});
    require(!elastic.initiated && elastic.nextHistory.damage == 0.0,
        "subcritical separation initiated damage");
    require(near(elastic.tractionPa[0], 4.0, 1.0e-12) &&
        near(elastic.tractionPa[1], 0.8, 1.0e-12) &&
        near(elastic.tractionPa[2], -0.4, 1.0e-12),
        "subcritical elastic traction is incorrect");
    require(near(elastic.tangentPaPerM[0][0], 1000.0, 1.0e-10) &&
        near(elastic.tangentPaPerM[1][1], 400.0, 1.0e-10),
        "subcritical elastic tangent is incorrect");

    const Response virginCrossing = evaluate(law, {0.009, 0.0, 0.0},
        {0.015, 0.0, 0.0});
    const double expectedOnset = std::sqrt(1000.0) * 0.01;
    require(virginCrossing.initiated &&
        near(virginCrossing.nextHistory.onsetEnergySeparation,
            expectedOnset, 1.0e-12),
        "first normal q=1 crossing did not freeze onset at the threshold");
    require(virginCrossing.nextHistory.maximumEnergySeparation >=
            virginCrossing.nextHistory.onsetEnergySeparation &&
        virginCrossing.nextHistory.damage > 0.0 &&
        virginCrossing.nextHistory.damage < 1.0 &&
        virginCrossing.dissipatedWorkJPerM2 > 0.0,
        "post-initiation softening did not produce bounded irreversible damage/work");

    const Response subdividedFirst = evaluate(law, {0.0, 0.0, 0.0},
        {0.0095, 0.0, 0.0});
    require(!subdividedFirst.initiated,
        "subthreshold first substep unexpectedly initiated");
    const Response subdividedSecond = evaluate(law, {0.0095, 0.0, 0.0},
        {0.015, 0.0, 0.0}, subdividedFirst.nextHistory);
    require(near(subdividedSecond.nextHistory.onsetEnergySeparation,
            virginCrossing.nextHistory.onsetEnergySeparation, 1.0e-12) &&
        near(subdividedSecond.nextHistory.frozenFractureEnergyJPerM2,
            virginCrossing.nextHistory.frozenFractureEnergyJPerM2, 1.0e-12),
        "onset mix or threshold changed when the same path was subdivided");

    const Response moreLoaded = evaluate(law, {0.015, 0.0, 0.0},
        {0.020, 0.0, 0.0}, virginCrossing.nextHistory);
    const Response unloaded = evaluate(law, {0.020, 0.0, 0.0},
        {0.008, 0.0, 0.0}, moreLoaded.nextHistory);
    require(unloaded.nextHistory.damage == moreLoaded.nextHistory.damage &&
        unloaded.dissipatedWorkJPerM2 == moreLoaded.dissipatedWorkJPerM2,
        "unloading reduced accepted damage or cumulative dissipation");
    require(moreLoaded.nextHistory.damage >= virginCrossing.nextHistory.damage &&
        moreLoaded.dissipatedWorkJPerM2 >=
            virginCrossing.dissipatedWorkJPerM2,
        "loading reduced damage or cumulative dissipation");

    const Response closed = evaluate(law, {0.0, 0.0, 0.0},
        {-0.002, 0.0002, -0.0001});
    require(!closed.initiated && closed.nextHistory.damage == 0.0 &&
        near(closed.tractionPa[0], -2.4, 1.0e-12) &&
        near(closed.tractionPa[1], 0.08, 1.0e-12),
        "compressive closure was not separate from tensile damage");

    // A segment can cross the initiation surface and approach a lower-rho
    // direction before its endpoint. The history maximum must include onset.
    const Law skewLaw{
        .normalPenaltyPaPerM = 1000.0,
        .shearPenaltyPaPerM = 400.0,
        .compressionPenaltyPaPerM = 1000.0,
        .normalStrengthPa = 100.0,
        .shearStrengthPa = 2.0,
        .modeIFractureEnergyJPerM2 = 10.0,
        .modeIIFractureEnergyJPerM2 = 1.0,
        .bkExponent = 1.5,
    };
    const Response pathMaximum = evaluate(skewLaw, {0.09, 0.0, 0.0},
        {0.005, 0.0055, 0.0});
    require(pathMaximum.initiated &&
        near(pathMaximum.nextHistory.maximumEnergySeparation,
            pathMaximum.nextHistory.onsetEnergySeparation, 1.0e-12) &&
        pathMaximum.nextHistory.damage == 0.0 &&
        pathMaximum.dissipatedWorkJPerM2 == 0.0,
        "new onset was omitted from a lower-rho endpoint path maximum");

    const std::array<double, 3> mixedAccepted{0.008, 0.002, -0.001};
    const std::array<double, 3> mixedTrial{0.014, 0.010, -0.004};
    const Response mixed = evaluate(law, mixedAccepted, mixedTrial);
    require(mixed.initiated && mixed.nextHistory.damage > 0.0 &&
        mixed.dissipatedWorkJPerM2 >= 0.0,
        "mixed-mode BK fixture did not initiate with nonnegative dissipation");

    checkTangent(law, {0.009, 0.0, 0.0}, {0.015, 0.0, 0.0},
        {}, 1.0e-8, 2.0e-5, "normal onset plus softening");
    checkTangent(law, mixedAccepted, mixedTrial, {},
        1.0e-8, 3.0e-5, "mixed onset plus softening");
    checkTangent(law, {0.020, 0.0, 0.0}, {0.012, 0.0, 0.0},
        moreLoaded.nextHistory, 1.0e-8, 2.0e-7, "unloading");

    std::cout << "PASS mixed cohesive reference: elastic, q=1 onset path, "
        << "path maximum, BK mix, compression closure, irreversible damage/work, "
        << "and directional tangents; synthetic values only\n";
}
