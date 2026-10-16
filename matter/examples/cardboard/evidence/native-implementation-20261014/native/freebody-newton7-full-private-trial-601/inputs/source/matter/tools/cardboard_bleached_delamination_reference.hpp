#pragma once

// Source-bound, scalar pure-mode-I reference for a bleached paperboard
// delamination interface. This is deliberately NOT a starch-adhesive law and
// is not consumed by Matter's native FEM solver.
//
// Calibration source:
//   Tryding et al. (2023), Int. J. Solids Struct. 279, 112365,
//   DOI 10.1016/j.ijsolstr.2023.112365, Table 1 and Eqs. (22)-(24),(39)-(43).
// The paperboard is single-ply, bleached, clay-coated, 319 g/m^2, 0.410 mm.
// Its cohesive envelope values are taken from Biel et al. (2022),
// DOI 10.1016/j.ijsolstr.2022.111755, a different production batch of the
// same paperboard quality. No parameter here is identified with the
// corrugated cardboard liner/medium or its starch bond.
//
// The implementation is only the uniaxial normal specialization: published
// normalized traction-separation envelope plus the published normal elastic
// unloading/reloading modulus. It omits the full mixed-mode plastic flow,
// compression/shear coupling, and the calibrated full finite-element model.
// The source authors explicitly note that this unloading-stiffness fit does
// not reproduce measured hysteresis. Compression/contact is also outside this
// opening-only interface law.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace numi_cardboard::bleached_delamination {

struct Parameters {
    double normalStrengthMPa;
    double normalLengthMM;
    double shearStrengthMPa;
    double shearLengthMM;
    double shape;
    double normalUnloadingScaleMPaPerMM;
    double unloadingExponent;
    double friction;
    double plasticPotentialFriction;
};

// Table 1 of Tryding et al. (2023). The final parameter muBar=0.10 is marked
// by the authors as lacking experimental support. It is included for
// provenance but unused by the mode-I specialization.
inline constexpr Parameters bleachedPaperboard{
    0.397, 0.111, 1.47, 0.0397, 0.863, 10.9, 1.39, 0.70, 0.10
};

struct State {
    // State is initialized at cohesive-strength onset: delta=0, traction=Tn.
    // This is a post-initiation cohesive surface, not an intact bonded face.
    double maximumOpeningMM = 0.0;
    double openingMM = 0.0;
    double tractionMPa = bleachedPaperboard.normalStrengthMPa;
    double accumulatedExternalWorkJPerM2 = 0.0;
    double dissipatedWorkJPerM2 = 0.0;
};

struct Response {
    double openingMM = 0.0;
    double maximumOpeningMM = 0.0;
    double tractionMPa = 0.0;
    double damage = 0.0;
    double plasticOpeningMM = 0.0;
    double unloadingModulusMPaPerMM = 0.0;
    double freeEnergyJPerM2 = 0.0;
    double cumulativeWorkJPerM2 = 0.0;
    double cumulativeDissipationJPerM2 = 0.0;
};

inline double xc(const Parameters& p = bleachedPaperboard) {
    if (!(p.shape > 0.0 && p.shape < 1.0))
        throw std::invalid_argument("cohesive shape parameter must be in (0,1)");
    const double c = p.shape;
    return c / 4.0 * std::pow(1.0 / c + 1.0, 1.0 + c) *
           std::pow(1.0 / c - 1.0, 1.0 - c);
}

inline double envelopeTractionMPa(double openingMM,
                                  const Parameters& p = bleachedPaperboard) {
    if (!std::isfinite(openingMM) || openingMM < 0.0)
        throw std::invalid_argument("opening must be finite and nonnegative");
    if (openingMM == 0.0) return p.normalStrengthMPa;
    const double normalized = openingMM / p.normalLengthMM;
    const double xi = std::pow(normalized / xc(p), 1.0 / p.shape);
    return p.normalStrengthMPa / (1.0 + xi);
}

inline double envelopeTangentMPaPerMM(
    double openingMM, const Parameters& p = bleachedPaperboard) {
    if (!std::isfinite(openingMM) || openingMM < 0.0)
        throw std::invalid_argument("opening must be finite and nonnegative");
    if (openingMM == 0.0) return 0.0;
    const double c = p.shape;
    const double epsilon = openingMM / p.normalLengthMM;
    const double xi = std::pow(epsilon / xc(p), 1.0 / c);
    return -p.normalStrengthMPa * xi /
        (c * p.normalLengthMM * epsilon * (1.0 + xi) * (1.0 + xi));
}

// Eq. (43): experimentally calibrated normal elastic tangent used during
// unloading/reloading. It is undefined at the zero-opening initiation point
// and diverges there, as the paper's kappa=0 intact limit requires.
inline double unloadingModulusMPaPerMM(
    double maximumOpeningMM, const Parameters& p = bleachedPaperboard) {
    if (!std::isfinite(maximumOpeningMM) || maximumOpeningMM <= 0.0)
        throw std::invalid_argument(
            "finite unloading modulus requires positive maximum opening");
    const double dimensionless =
        xc(p) * p.normalLengthMM * p.shape / maximumOpeningMM;
    return p.normalUnloadingScaleMPaPerMM *
           std::pow(dimensionless, p.unloadingExponent);
}

inline double normalizedEnvelopeWorkMM(
    double openingMM, const Parameters& p = bleachedPaperboard) {
    if (!std::isfinite(openingMM) || openingMM < 0.0)
        throw std::invalid_argument("opening must be finite and nonnegative");
    if (openingMM == 0.0) return 0.0;

    // Integrate the dimensionless master curve using adaptive Simpson. Its
    // traction is continuous with zero initial tangent at epsilon=0.
    const double c = p.shape;
    const double upper = openingMM / p.normalLengthMM;
    const double xcrit = xc(p);
    const auto integrand = [c, xcrit](double epsilon) {
        if (epsilon == 0.0) return 1.0;
        const double xi = std::pow(epsilon / xcrit, 1.0 / c);
        return 1.0 / (1.0 + xi);
    };
    constexpr unsigned intervals = 4096u;
    static_assert(intervals % 2u == 0u);
    const double step = upper / static_cast<double>(intervals);
    double sum = integrand(0.0) + integrand(upper);
    for (unsigned index = 1u; index < intervals; ++index)
        sum += (index % 2u == 0u ? 2.0 : 4.0) *
               integrand(step * static_cast<double>(index));
    const double integral = sum * step / 3.0;
    return integral;
}

inline double envelopeWorkJPerM2(
    double openingMM, const Parameters& p = bleachedPaperboard) {
    // MPa*mm = 1000 J/m^2.
    return 1000.0 * p.normalStrengthMPa * p.normalLengthMM *
           normalizedEnvelopeWorkMM(openingMM, p);
}

inline double totalEnvelopeWorkJPerM2(
    const Parameters& p = bleachedPaperboard) {
    const double c = p.shape;
    const double normalizedArea = xc(p) * c *
        std::numbers::pi / std::sin(std::numbers::pi * c);
    return 1000.0 * p.normalStrengthMPa * p.normalLengthMM * normalizedArea;
}

inline double responseTractionMPa(double openingMM, double maximumOpeningMM,
                                   const Parameters& p = bleachedPaperboard) {
    if (!std::isfinite(openingMM) || openingMM < 0.0 ||
        !std::isfinite(maximumOpeningMM) || maximumOpeningMM < 0.0 ||
        openingMM > maximumOpeningMM + 1.0e-14)
        throw std::invalid_argument("invalid opening/history for unloading branch");
    if (maximumOpeningMM == 0.0) return p.normalStrengthMPa;
    const double peakTraction = envelopeTractionMPa(maximumOpeningMM, p);
    const double modulus = unloadingModulusMPaPerMM(maximumOpeningMM, p);
    return std::max(0.0, peakTraction + modulus *
                    (openingMM - maximumOpeningMM));
}

inline double freeEnergyJPerM2(const State& state,
                               const Parameters& p = bleachedPaperboard) {
    if (state.maximumOpeningMM == 0.0) return 0.0;
    const double modulus = unloadingModulusMPaPerMM(
        state.maximumOpeningMM, p);
    return 500.0 * state.tractionMPa * state.tractionMPa / modulus;
}

inline double linearBranchWorkJPerM2(double fromMM, double toMM,
                                      double maximumMM,
                                      const Parameters& p = bleachedPaperboard) {
    if (maximumMM == 0.0 || fromMM == toMM) return 0.0;
    const double peak = envelopeTractionMPa(maximumMM, p);
    const double slope = unloadingModulusMPaPerMM(maximumMM, p);
    const double zero = maximumMM - peak / slope;
    const double a = std::max(fromMM, zero);
    const double b = std::max(toMM, zero);
    if (a == b) return 0.0;
    const auto primitive = [=](double x) {
        const double offset = x - maximumMM;
        return peak * offset + 0.5 * slope * offset * offset;
    };
    return 1000.0 * (primitive(b) - primitive(a));
}

inline Response snapshot(const State& state,
                         const Parameters& p = bleachedPaperboard) {
    Response result{};
    result.openingMM = state.openingMM;
    result.maximumOpeningMM = state.maximumOpeningMM;
    result.tractionMPa = state.tractionMPa;
    if (state.maximumOpeningMM > 0.0) {
        const double peakTraction = envelopeTractionMPa(
            state.maximumOpeningMM, p);
        const double peakModulus = unloadingModulusMPaPerMM(
            state.maximumOpeningMM, p);
        result.damage = 1.0 - peakTraction / p.normalStrengthMPa;
        const double peakPlastic = state.maximumOpeningMM -
                                   peakTraction / peakModulus;
        result.plasticOpeningMM = peakPlastic;
        result.unloadingModulusMPaPerMM = peakModulus;
    }
    result.freeEnergyJPerM2 = freeEnergyJPerM2(state, p);
    result.cumulativeWorkJPerM2 = state.accumulatedExternalWorkJPerM2;
    result.cumulativeDissipationJPerM2 = state.dissipatedWorkJPerM2;
    return result;
}

// Advance the one-dimensional reference state. A fresh interface is at
// cohesive initiation (delta=0, traction=Tn). The loading envelope is
// irreversible; unloading/reloading to the previous maximum uses the
// calibrated Eq. (43) linear elastic slope, with zero tensile traction after
// the branch reaches zero. The returned cumulative dissipation is external
// work minus the Eq. (2) elastic free energy for this scalar specialization.
inline Response advance(State& state, double targetOpeningMM,
                        const Parameters& p = bleachedPaperboard) {
    if (!std::isfinite(targetOpeningMM) || targetOpeningMM < 0.0)
        throw std::invalid_argument("target opening must be finite and nonnegative");
    if (!std::isfinite(state.maximumOpeningMM) ||
        !std::isfinite(state.openingMM) ||
        !std::isfinite(state.tractionMPa) ||
        !std::isfinite(state.accumulatedExternalWorkJPerM2) ||
        !std::isfinite(state.dissipatedWorkJPerM2) ||
        state.maximumOpeningMM < state.openingMM || state.openingMM < 0.0)
        throw std::invalid_argument("invalid cohesive state");

    const double oldPsi = freeEnergyJPerM2(state, p);
    double workIncrement = 0.0;
    const double peak = state.maximumOpeningMM;
    if (peak > 0.0 && targetOpeningMM <= peak) {
        workIncrement = linearBranchWorkJPerM2(
            state.openingMM, targetOpeningMM, peak, p);
        state.tractionMPa = responseTractionMPa(
            targetOpeningMM, peak, p);
    } else if (peak > 0.0 && state.openingMM < peak &&
               targetOpeningMM > peak) {
        workIncrement += linearBranchWorkJPerM2(
            state.openingMM, peak, peak, p);
        workIncrement += envelopeWorkJPerM2(targetOpeningMM, p) -
                         envelopeWorkJPerM2(peak, p);
        state.maximumOpeningMM = targetOpeningMM;
        state.tractionMPa = envelopeTractionMPa(targetOpeningMM, p);
    } else {
        workIncrement = envelopeWorkJPerM2(targetOpeningMM, p) -
                        envelopeWorkJPerM2(state.openingMM, p);
        state.maximumOpeningMM = std::max(peak, targetOpeningMM);
        state.tractionMPa = envelopeTractionMPa(targetOpeningMM, p);
    }
    state.openingMM = targetOpeningMM;
    state.accumulatedExternalWorkJPerM2 += workIncrement;
    const double newPsi = freeEnergyJPerM2(state, p);
    state.dissipatedWorkJPerM2 += workIncrement - (newPsi - oldPsi);
    return snapshot(state, p);
}

} // namespace numi_cardboard::bleached_delamination
