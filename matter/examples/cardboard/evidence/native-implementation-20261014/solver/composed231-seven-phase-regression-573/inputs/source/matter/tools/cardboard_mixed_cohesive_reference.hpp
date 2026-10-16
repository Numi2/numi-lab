#pragma once

// Independent CPU oracle for the opt-in mixed-mode bilinear cohesive law.
// These equations are an engineering model with synthetic verification
// fixtures; the parameters are not calibrated to starch or corrugated board.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace cardboard::mixed_cohesive_reference {

struct Law {
    double normalPenaltyPaPerM = 0.0;
    double shearPenaltyPaPerM = 0.0;
    double compressionPenaltyPaPerM = 0.0;
    double normalStrengthPa = 0.0;
    double shearStrengthPa = 0.0;
    double modeIFractureEnergyJPerM2 = 0.0;
    double modeIIFractureEnergyJPerM2 = 0.0;
    double bkExponent = 0.0;
};

struct History {
    double maximumEnergySeparation = 0.0;
    double onsetEnergySeparation = 0.0;
    double frozenFractureEnergyJPerM2 = 0.0;
    double damage = 0.0;
};

struct Response {
    std::array<double, 3> tractionPa{};
    std::array<std::array<double, 3>, 3> tangentPaPerM{};
    History nextHistory{};
    double dissipatedWorkJPerM2 = 0.0;
    bool initiated = false;
    bool valid = false;
};

namespace detail {

struct Dual {
    double value = 0.0;
    double derivative = 0.0;
};

inline Dual add(Dual a, Dual b) {
    return {a.value + b.value, a.derivative + b.derivative};
}
inline Dual sub(Dual a, Dual b) {
    return {a.value - b.value, a.derivative - b.derivative};
}
inline Dual mul(Dual a, Dual b) {
    return {a.value * b.value,
        a.derivative * b.value + a.value * b.derivative};
}
inline Dual div(Dual a, Dual b) {
    const double inverse = 1.0 / b.value;
    return {a.value * inverse,
        (a.derivative - a.value * inverse * b.derivative) * inverse};
}
inline Dual sqrt(Dual a) {
    const double value = std::sqrt(std::max(a.value, 0.0));
    return {value, value > 0.0 ? 0.5 * a.derivative / value : 0.0};
}
inline Dual pow(Dual a, double exponent) {
    const double value = std::pow(std::max(a.value, 0.0), exponent);
    const double slope = a.value > 0.0
        ? exponent * std::pow(a.value, exponent - 1.0)
        : (exponent == 1.0 ? 1.0 : 0.0);
    return {value, slope * a.derivative};
}

inline double initiationQ(const Law& law, const std::array<double, 3>& gap) {
    const double normal = law.normalPenaltyPaPerM * std::max(gap[0], 0.0) /
        law.normalStrengthPa;
    const double shearScale = law.shearPenaltyPaPerM / law.shearStrengthPa;
    return std::sqrt(normal * normal + shearScale * shearScale *
        (gap[1] * gap[1] + gap[2] * gap[2]));
}

inline double energySeparation(const Law& law,
                               const std::array<double, 3>& gap) {
    const double normal = std::max(gap[0], 0.0);
    return std::sqrt(law.normalPenaltyPaPerM * normal * normal +
        law.shearPenaltyPaPerM * (gap[1] * gap[1] + gap[2] * gap[2]));
}

inline bool validLaw(const Law& law) {
    const std::array<double, 8> values{
        law.normalPenaltyPaPerM, law.shearPenaltyPaPerM,
        law.compressionPenaltyPaPerM, law.normalStrengthPa,
        law.shearStrengthPa, law.modeIFractureEnergyJPerM2,
        law.modeIIFractureEnergyJPerM2, law.bkExponent};
    return std::all_of(values.begin(), values.end(), [](double x) {
        return std::isfinite(x) && x > 0.0;
    }) && law.bkExponent >= 1.0;
}

struct DirectionalResponse {
    std::array<double, 3> traction{};
    std::array<double, 3> tractionDirection{};
    History next{};
    double work = 0.0;
    bool initiated = false;
    bool valid = false;
};

inline DirectionalResponse evaluateDirection(
    const Law& law,
    const std::array<double, 3>& acceptedGap,
    const std::array<double, 3>& trialGap,
    const std::array<double, 3>& direction,
    const History& accepted
) {
    DirectionalResponse result{};
    const auto finiteVector = [](const std::array<double, 3>& value) {
        return std::all_of(value.begin(), value.end(), [](double x) {
            return std::isfinite(x);
        });
    };
    if (!validLaw(law) || !std::isfinite(accepted.maximumEnergySeparation) ||
        !std::isfinite(accepted.onsetEnergySeparation) ||
        !std::isfinite(accepted.frozenFractureEnergyJPerM2) ||
        !std::isfinite(accepted.damage) || accepted.maximumEnergySeparation < 0.0 ||
        accepted.onsetEnergySeparation < 0.0 ||
        accepted.frozenFractureEnergyJPerM2 < 0.0 || accepted.damage < 0.0 ||
        accepted.damage > 1.0 || !finiteVector(acceptedGap) ||
        !finiteVector(trialGap) || !finiteVector(direction)) return result;

    const double q0 = initiationQ(law, acceptedGap);
    const double q1 = initiationQ(law, trialGap);
    const bool wasInitiated = accepted.onsetEnergySeparation > 0.0;
    bool newOnset = false;
    Dual onsetRho{accepted.onsetEnergySeparation, 0.0};
    Dual onsetGc{accepted.frozenFractureEnergyJPerM2, 0.0};
    if (!wasInitiated && q0 >= 1.0) return result;
    if (!wasInitiated && q1 >= 1.0) {
        double lower = 0.0;
        double upper = 1.0;
        for (unsigned iteration = 0; iteration < 64; ++iteration) {
            const double alpha = 0.5 * (lower + upper);
            std::array<double, 3> gap{};
            for (std::size_t i = 0; i < 3; ++i)
                gap[i] = acceptedGap[i] + alpha * (trialGap[i] - acceptedGap[i]);
            if (initiationQ(law, gap) >= 1.0) upper = alpha;
            else lower = alpha;
        }
        const double alpha = upper;
        std::array<double, 3> segment{};
        std::array<double, 3> onsetGap{};
        for (std::size_t i = 0; i < 3; ++i) {
            segment[i] = trialGap[i] - acceptedGap[i];
            onsetGap[i] = acceptedGap[i] + alpha * segment[i];
        }
        const double dn = std::max(onsetGap[0], 0.0);
        const double kn = law.normalPenaltyPaPerM;
        const double ks = law.shearPenaltyPaPerM;
        const double modeIEnergy = kn * dn * dn;
        const double modeIIEnergy = ks *
            (onsetGap[1] * onsetGap[1] + onsetGap[2] * onsetGap[2]);
        const double totalEnergy = modeIEnergy + modeIIEnergy;
        if (!(totalEnergy > 0.0) || !std::isfinite(totalEnergy)) return result;

        // At q=1, these are the exact gradient components of the initiation
        // surface in local separation coordinates.
        std::array<double, 3> qGradient{
            onsetGap[0] > 0.0 ? kn * kn * dn /
                (law.normalStrengthPa * law.normalStrengthPa) : 0.0,
            ks * ks * onsetGap[1] /
                (law.shearStrengthPa * law.shearStrengthPa),
            ks * ks * onsetGap[2] /
                (law.shearStrengthPa * law.shearStrengthPa)};
        double qSlope = 0.0;
        double qDirection = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            qSlope += qGradient[i] * segment[i];
            qDirection += qGradient[i] * direction[i];
        }
        if (!(std::abs(qSlope) > 1.0e-20) || !std::isfinite(qSlope))
            return result;
        const double alphaDirection = -alpha * qDirection / qSlope;
        std::array<Dual, 3> onsetDual{};
        for (std::size_t i = 0; i < 3; ++i)
            onsetDual[i] = {onsetGap[i],
                alpha * direction[i] + alphaDirection * segment[i]};
        if (onsetGap[0] <= 0.0) onsetDual[0].derivative = 0.0;
        const Dual normalDual{dn, onsetDual[0].derivative};
        const Dual energyDual = add(
            mul({kn, 0.0}, mul(normalDual, normalDual)),
            mul({ks, 0.0}, add(
                mul(onsetDual[1], onsetDual[1]),
                mul(onsetDual[2], onsetDual[2]))));
        onsetRho = sqrt(energyDual);
        const Dual modeIIDual = mul({ks, 0.0}, add(
            mul(onsetDual[1], onsetDual[1]),
            mul(onsetDual[2], onsetDual[2])));
        const Dual modeFraction = div(modeIIDual,
            add(mul({kn, 0.0}, mul(normalDual, normalDual)), modeIIDual));
        onsetGc = add({law.modeIFractureEnergyJPerM2, 0.0},
            mul({law.modeIIFractureEnergyJPerM2 -
                    law.modeIFractureEnergyJPerM2, 0.0},
                pow(modeFraction, law.bkExponent)));
        newOnset = true;
    }

    std::array<Dual, 3> gapDual{
        Dual{trialGap[0], direction[0]},
        Dual{trialGap[1], direction[1]},
        Dual{trialGap[2], direction[2]}};
    const Dual positiveNormal{std::max(trialGap[0], 0.0),
        trialGap[0] > 0.0 ? direction[0] : 0.0};
    const Dual energy = add(
        mul({law.normalPenaltyPaPerM, 0.0},
            mul(positiveNormal, positiveNormal)),
        mul({law.shearPenaltyPaPerM, 0.0}, add(
            mul(gapDual[1], gapDual[1]), mul(gapDual[2], gapDual[2]))));
    const Dual rho = sqrt(energy);
    const double trialRho = rho.value;
    const bool initiated = wasInitiated || newOnset;

    // For a virgin crossing, preserve the maximum on the post-onset portion
    // of the accepted-to-trial segment. Convexity puts it at onset or trial.
    double rhoMaximum = initiated
        ? std::max(accepted.maximumEnergySeparation, trialRho) : 0.0;
    if (newOnset) rhoMaximum = std::max(rhoMaximum, onsetRho.value);
    Dual rhoMaximumDual{initiated ? accepted.maximumEnergySeparation : 0.0, 0.0};
    if (initiated && trialRho > rhoMaximumDual.value) rhoMaximumDual = rho;
    if (newOnset && onsetRho.value > rhoMaximumDual.value)
        rhoMaximumDual = onsetRho;

    Dual damage{initiated ? accepted.damage : 0.0, 0.0};
    const Dual rhoFinal = div(mul({2.0, 0.0}, onsetGc), onsetRho);
    if (initiated && !(onsetRho.value > 0.0 && onsetGc.value > 0.0 &&
                       rhoFinal.value > onsetRho.value &&
                       std::isfinite(rhoFinal.value))) return result;
    if (initiated && rhoMaximum > onsetRho.value) {
        if (rhoMaximum >= rhoFinal.value) {
            damage = {1.0, 0.0};
        } else {
            damage = div(
                mul(rhoFinal, sub(rhoMaximumDual, onsetRho)),
                mul(rhoMaximumDual, sub(rhoFinal, onsetRho)));
            if (damage.value <= 0.0) damage = {0.0, 0.0};
            else if (damage.value >= 1.0) damage = {1.0, 0.0};
        }
    }
    if (!initiated) damage = {0.0, 0.0};

    if (trialGap[0] > 0.0) {
        const Dual intact = sub({1.0, 0.0}, damage);
        const std::array<Dual, 3> traction{
            mul(mul(intact, {law.normalPenaltyPaPerM, 0.0}), gapDual[0]),
            mul(mul(intact, {law.shearPenaltyPaPerM, 0.0}), gapDual[1]),
            mul(mul(intact, {law.shearPenaltyPaPerM, 0.0}), gapDual[2])};
        for (std::size_t i = 0; i < 3; ++i) {
            result.traction[i] = traction[i].value;
            result.tractionDirection[i] = traction[i].derivative;
        }
        result.next.damage = damage.value;
        result.initiated = initiated;
        result.next.maximumEnergySeparation = initiated ? rhoMaximum : 0.0;
        result.next.onsetEnergySeparation = initiated ? onsetRho.value : 0.0;
        result.next.frozenFractureEnergyJPerM2 = initiated ? onsetGc.value : 0.0;
        const double range = rhoFinal.value - onsetRho.value;
        if (initiated && range > 0.0)
            result.work = onsetGc.value * std::clamp(
                (rhoMaximum - onsetRho.value) / range, 0.0, 1.0);
        result.valid = true;
        return result;
    }

    result.traction = {
        law.compressionPenaltyPaPerM * trialGap[0],
        (1.0 - damage.value) * law.shearPenaltyPaPerM * trialGap[1],
        (1.0 - damage.value) * law.shearPenaltyPaPerM * trialGap[2]};
    result.tractionDirection = {
        law.compressionPenaltyPaPerM * direction[0],
        (1.0 - damage.value) * law.shearPenaltyPaPerM * direction[1] -
            damage.derivative * law.shearPenaltyPaPerM * trialGap[1],
        (1.0 - damage.value) * law.shearPenaltyPaPerM * direction[2] -
            damage.derivative * law.shearPenaltyPaPerM * trialGap[2]};
    result.next = {initiated ? rhoMaximum : 0.0,
        initiated ? onsetRho.value : 0.0,
        initiated ? onsetGc.value : 0.0, damage.value};
    const double range = rhoFinal.value - onsetRho.value;
    if (initiated && range > 0.0)
        result.work = onsetGc.value * std::clamp(
            (rhoMaximum - onsetRho.value) / range, 0.0, 1.0);
    result.initiated = initiated;
    result.valid = true;
    return result;
}

} // namespace detail

inline bool evaluate(const Law& law,
                     const std::array<double, 3>& acceptedGap,
                     const std::array<double, 3>& trialGap,
                     const History& accepted,
                     Response& response) {
    if (!detail::validLaw(law)) return false;
    response = {};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        std::array<double, 3> direction{};
        direction[axis] = 1.0;
        const auto column = detail::evaluateDirection(
            law, acceptedGap, trialGap, direction, accepted);
        if (!column.valid) return false;
        for (std::size_t row = 0; row < 3; ++row)
            response.tangentPaPerM[row][axis] =
                column.tractionDirection[row];
    }
    const auto final = detail::evaluateDirection(law, acceptedGap,
        trialGap, {}, accepted);
    if (!final.valid) return false;
    response.tractionPa = final.traction;
    response.nextHistory = final.next;
    response.dissipatedWorkJPerM2 = final.work;
    response.initiated = final.initiated;
    response.valid = true;
    return true;
}

inline bool directional(const Law& law,
                        const std::array<double, 3>& acceptedGap,
                        const std::array<double, 3>& trialGap,
                        const std::array<double, 3>& direction,
                        const History& accepted,
                        std::array<double, 3>& traction,
                        std::array<double, 3>& tractionDirection,
                        History& next,
                        double& work) {
    const auto value = detail::evaluateDirection(
        law, acceptedGap, trialGap, direction, accepted);
    if (!value.valid) return false;
    traction = value.traction;
    tractionDirection = value.tractionDirection;
    next = value.next;
    work = value.work;
    return true;
}

} // namespace cardboard::mixed_cohesive_reference
