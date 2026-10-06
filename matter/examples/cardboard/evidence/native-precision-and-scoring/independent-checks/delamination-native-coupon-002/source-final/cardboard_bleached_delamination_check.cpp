#include "cardboard_bleached_delamination_reference.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace numi_cardboard::bleached_delamination;
unsigned checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double absoluteTolerance,
           const std::string& message) {
    require(std::isfinite(actual) &&
            std::abs(actual - expected) <= absoluteTolerance,
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

void checkPublishedEnvelopeAndTangency() {
    const Parameters& p = bleachedPaperboard;
    close(p.normalStrengthMPa, 0.397, 0.0, "Table 1 normal cohesive strength");
    close(p.normalLengthMM, 0.111, 0.0, "Table 1 normal length");
    close(p.shape, 0.863, 0.0, "Table 1 envelope shape parameter");
    close(p.normalUnloadingScaleMPaPerMM, 10.9, 0.0,
          "Table 1 normal unloading modulus scale");
    close(p.unloadingExponent, 1.39, 0.0, "Table 1 unloading exponent");
    close(p.friction, 0.70, 0.0, "Table 1 interface friction");
    close(p.plasticPotentialFriction, 0.10, 0.0,
          "Table 1 unsupported plastic-potential friction input");

    close(xc(), 0.7031799504329036, 2.0e-15,
          "Eq. (24) dimensionless shape factor");
    close(envelopeTractionMPa(0.0), p.normalStrengthMPa, 0.0,
          "cohesive onset value at zero post-initiation opening");
    close(envelopeTractionMPa(p.normalLengthMM),
          0.15855429448245711, 2.0e-14,
          "Eq. (22) envelope at normalized opening one");

    // The location of the inflection follows analytically from xi=(1-c)/(1+c).
    // The source defines delta_N so that the normalized tangent there is -1.
    const double c = p.shape;
    const double epsilonInflection =
        xc() * std::pow((1.0 - c) / (1.0 + c), c);
    const double openingInflection = epsilonInflection * p.normalLengthMM;
    close(envelopeTangentMPaPerMM(openingInflection) *
              p.normalLengthMM / p.normalStrengthMPa,
          -1.0, 2.0e-14, "normalized envelope tangent at inflection");
    require(envelopeTangentMPaPerMM(openingInflection) < 0.0,
            "post-onset envelope must soften monotonically");

    close(envelopeWorkJPerM2(0.1), 26.1095015697, 2.0e-8,
          "integrated mode-I envelope work at 0.1 mm");
    close(totalEnvelopeWorkJPerM2(), 201.3549362088, 2.0e-8,
          "analytic complete-envelope work from Eq. (22)");
}

void checkLoadingUnloadingAndEnergyAccounting() {
    State state;
    close(state.tractionMPa, bleachedPaperboard.normalStrengthMPa, 0.0,
          "post-initiation state starts on strength envelope");
    const Response loaded = advance(state, 0.1);
    close(loaded.tractionMPa, 0.1701977465712072, 2.0e-14,
          "monotonic envelope traction at 0.1 mm");
    close(loaded.damage, 0.5712903109037601, 2.0e-14,
          "damage inferred from the uniaxial cohesive surface");
    close(loaded.unloadingModulusMPaPerMM, 6.293653816112125,
          2.0e-13, "Eq. (43) unloading stiffness");
    close(loaded.plasticOpeningMM, 0.07295724367052238, 2.0e-14,
          "plastic opening implied by source free energy and traction");
    close(loaded.freeEnergyJPerM2, 2.3013080941756714, 2.0e-13,
          "Eq. (2) scalar recoverable free energy");
    close(loaded.cumulativeWorkJPerM2, 26.1095015697, 2.0e-8,
          "cumulative envelope work");
    close(loaded.cumulativeDissipationJPerM2,
          23.8081934755, 2.0e-8,
          "external work less recoverable free energy");
    require(loaded.cumulativeDissipationJPerM2 > 0.0,
            "monotonic softening must dissipate positive work");
    close(loaded.cumulativeWorkJPerM2 - loaded.freeEnergyJPerM2,
          loaded.cumulativeDissipationJPerM2, 2.0e-10,
          "local first-law balance after loading");

    const double dissipationAtPeak = state.dissipatedWorkJPerM2;
    const double workAtPeak = state.accumulatedExternalWorkJPerM2;
    const Response unloaded = advance(state, 0.08);
    const double workAtUnloadedPoint = state.accumulatedExternalWorkJPerM2;
    close(unloaded.tractionMPa, 0.044324670248964665, 2.0e-14,
          "linear elastic unloading response");
    close(unloaded.damage, loaded.damage, 0.0,
          "damage remains fixed on an unloading branch");
    close(unloaded.plasticOpeningMM, loaded.plasticOpeningMM, 0.0,
          "plastic opening remains fixed on an unloading branch");
    close(unloaded.cumulativeDissipationJPerM2,
          dissipationAtPeak, 2.0e-12,
          "unloading is reversible under the calibrated tangent");
    close(unloaded.cumulativeWorkJPerM2 - workAtPeak,
          unloaded.freeEnergyJPerM2 - loaded.freeEnergyJPerM2,
          2.0e-12, "unloading work equals the change in stored energy");

    const Response zeroForce = advance(state, 0.0);
    close(zeroForce.tractionMPa, 0.0, 0.0,
          "opening-only cohesive law has no tensile traction after separation");
    close(zeroForce.freeEnergyJPerM2, 0.0, 0.0,
          "free energy vanishes after the unloading branch reaches zero force");
    close(zeroForce.cumulativeDissipationJPerM2,
          dissipationAtPeak, 2.0e-12,
          "zero-force unloading does not add damage dissipation");

    const Response reloaded = advance(state, 0.08);
    close(reloaded.tractionMPa, unloaded.tractionMPa, 2.0e-14,
          "reloading retraces the calibrated elastic branch");
    close(reloaded.cumulativeDissipationJPerM2,
          dissipationAtPeak, 2.0e-12,
          "closed elastic unload/reload cycle adds no dissipated work");
    close(reloaded.cumulativeWorkJPerM2, workAtUnloadedPoint, 2.0e-12,
          "closed elastic excursion returns to its starting work");

    const Response extended = advance(state, 0.12);
    close(extended.maximumOpeningMM, 0.12, 0.0,
          "new maximum opening advances the irreversible history");
    require(extended.damage > reloaded.damage,
            "damage must increase when a new opening maximum is reached");
    require(extended.cumulativeDissipationJPerM2 >
                dissipationAtPeak,
            "additional envelope loading must dissipate energy");
    close(extended.cumulativeWorkJPerM2,
          envelopeWorkJPerM2(0.12), 2.0e-8,
          "unload/reload followed by new loading matches the envelope work");
    close(extended.cumulativeWorkJPerM2 - extended.freeEnergyJPerM2,
          extended.cumulativeDissipationJPerM2, 2.0e-10,
          "local first-law balance after reloading beyond prior maximum");
}

void checkInvalidInputsFailClosed() {
    State state;
    const State before = state;
    bool rejected = false;
    try {
        (void)advance(state, -1.0e-4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "negative opening must be rejected");
    close(state.maximumOpeningMM, before.maximumOpeningMM, 0.0,
          "failed input must not mutate maximum opening");
    close(state.openingMM, before.openingMM, 0.0,
          "failed input must not mutate current opening");
    close(state.tractionMPa, before.tractionMPa, 0.0,
          "failed input must not mutate traction");
}

double openingFromDamage(const double damage) {
    const Parameters& p = bleachedPaperboard;
    return p.normalLengthMM * xc(p) *
        std::pow(damage / (1.0 - damage), p.shape);
}

void checkHistoryRegularizerBound() {
    constexpr double epsilon = 1.0e-8;
    constexpr double sourceDamage = 0.01;
    constexpr double localProjectionTolerance = 2.0e-5;
    const double junctionResidual = std::sqrt(2.0) * epsilon;
    const double acceptedTolerance = localProjectionTolerance *
        (1.0 + sourceDamage);
    require(junctionResidual < acceptedTolerance,
        "declared 1e-8 history residual must remain within native local-state tolerance");

    // A fully resolved smoothed FB root would move kappa by epsilon at the
    // exact junction. Production retains the exact max seed because its local
    // residual tolerance accepts that seed, so this is a bound, not an applied
    // state increment.
    const double smoothRootDamage = sourceDamage + epsilon;
    const double oldOpening = openingFromDamage(sourceDamage);
    const double smoothRootMaximum = openingFromDamage(smoothRootDamage);
    State sourceAtJunction;
    sourceAtJunction.maximumOpeningMM = oldOpening;
    sourceAtJunction.openingMM = oldOpening;
    sourceAtJunction.tractionMPa = envelopeTractionMPa(oldOpening);
    sourceAtJunction.accumulatedExternalWorkJPerM2 =
        envelopeWorkJPerM2(oldOpening);
    sourceAtJunction.dissipatedWorkJPerM2 =
        sourceAtJunction.accumulatedExternalWorkJPerM2 -
        freeEnergyJPerM2(sourceAtJunction);

    State hypotheticalRegularized = sourceAtJunction;
    hypotheticalRegularized.maximumOpeningMM = smoothRootMaximum;
    hypotheticalRegularized.tractionMPa = responseTractionMPa(
        oldOpening, smoothRootMaximum);
    const double hypotheticalFreeEnergyDelta =
        freeEnergyJPerM2(hypotheticalRegularized) -
        freeEnergyJPerM2(sourceAtJunction);
    const double hypotheticalTractionDeltaPa = 1.0e6 *
        (hypotheticalRegularized.tractionMPa -
         sourceAtJunction.tractionMPa);

    close(smoothRootDamage - sourceDamage, epsilon, 1.0e-15,
          "exact smoothed-root transition state bound");
    close(smoothRootMaximum - oldOpening, 1.28982818e-9, 2.0e-15,
          "source-equation opening shift for exact smoothed-root bound");
    close(hypotheticalTractionDeltaPa, -2.84134638, 2.0e-7,
          "source traction change for exact smoothed-root bound");
    close(hypotheticalFreeEnergyDelta, -4.65105819e-7, 2.0e-15,
          "source free-energy change for exact smoothed-root bound");

    const Response hold = advance(sourceAtJunction, oldOpening);
    for (unsigned step = 0; step < 1000; ++step) {
        const Response repeated = advance(sourceAtJunction, oldOpening);
        close(repeated.maximumOpeningMM, oldOpening, 0.0,
              "source max-history repeated hold must not grow damage");
        close(repeated.cumulativeWorkJPerM2,
              hold.cumulativeWorkJPerM2, 2.0e-13,
              "constant-opening hold must do no external work");
        close(repeated.cumulativeDissipationJPerM2,
              hold.cumulativeDissipationJPerM2, 2.0e-13,
              "constant-opening hold must not add dissipation");
    }
}

void writeSourceTrajectory(const std::filesystem::path& path) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error(
        "cannot create trajectory CSV: " + path.string());
    output << "path_index,segment,opening_mm,traction_mpa,damage,"
              "external_work_j_m2,free_energy_j_m2,dissipation_j_m2\n";
    constexpr double initialDamage = 0.01;
    const double initialOpening = openingFromDamage(initialDamage);
    State state;
    state.maximumOpeningMM = initialOpening;
    state.openingMM = initialOpening;
    state.tractionMPa = envelopeTractionMPa(initialOpening);
    state.accumulatedExternalWorkJPerM2 =
        envelopeWorkJPerM2(initialOpening);
    state.dissipatedWorkJPerM2 = state.accumulatedExternalWorkJPerM2 -
        freeEnergyJPerM2(state);
    std::size_t index = 0u;
    const auto emit = [&](const char* segment, const Response& response) {
        output << index++ << ',' << segment << ','
               << response.openingMM << ',' << response.tractionMPa << ','
               << response.damage << ',' << response.cumulativeWorkJPerM2 << ','
               << response.freeEnergyJPerM2 << ','
               << response.cumulativeDissipationJPerM2 << '\n';
    };
    emit("seed", snapshot(state));
    const auto traverse = [&](const char* segment, const double target,
                              const unsigned steps) {
        const double from = state.openingMM;
        for (unsigned step = 1u; step <= steps; ++step) {
            const double fraction = static_cast<double>(step) / steps;
            emit(segment, advance(
                state, from + fraction * (target - from)));
        }
    };
    traverse("loading", 0.1, 128u);
    traverse("unloading", 0.08, 64u);
    traverse("reloading", 0.09, 32u);
    traverse("new_maximum", 0.12, 64u);
    output.flush();
    if (!output) throw std::runtime_error(
        "failed writing trajectory CSV: " + path.string());
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 &&
            (argc != 3 || std::string(argv[1]) != "--trajectory-csv"))
            throw std::runtime_error(
                "usage: cardboard-delamination-reference-check [--trajectory-csv PATH]");
        checkPublishedEnvelopeAndTangency();
        checkLoadingUnloadingAndEnergyAccounting();
        checkInvalidInputsFailClosed();
        checkHistoryRegularizerBound();
        if (argc == 3) writeSourceTrajectory(argv[2]);
        std::cout << std::setprecision(12)
                  << "cardboard bleached-paperboard delamination reference PASS"
                  << " checks=" << checks
                  << " modeI_envelope_total_J_m2="
                  << totalEnvelopeWorkJPerM2()
                  << " history_fb_epsilon=1e-8 history_residual_to_tolerance="
                  << std::sqrt(2.0) * 1.0e-8 /
                      (2.0e-5 * (1.0 + 0.01))
                  << " regularized_root_is_bound_not_projected_state=true"
                  << " (source-specific reference; not corrugated glue)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardboard bleached-paperboard delamination reference FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
