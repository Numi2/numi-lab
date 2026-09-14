#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoffTypes.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {

inline bool numiHumanHandoffFinite(
    const std::span<const double> values
) noexcept {
    return std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
    });
}

inline bool numiHumanHandoffThresholdsValid(
    const NumiHumanHandoffThresholds& value
) noexcept {
    return std::isfinite(value.activationAbsolute) &&
        value.activationAbsolute >= 0.0 &&
        std::isfinite(value.fiberAbsoluteMeters) &&
        value.fiberAbsoluteMeters >= 0.0 &&
        std::isfinite(value.fiberRelative) && value.fiberRelative >= 0.0 &&
        std::isfinite(value.forceAbsoluteNewtons) &&
        value.forceAbsoluteNewtons >= 0.0 &&
        std::isfinite(value.forceRelative) && value.forceRelative >= 0.0 &&
        std::isfinite(value.decompositionAbsoluteNewtons) &&
        value.decompositionAbsoluteNewtons >= 0.0 &&
        std::isfinite(value.residualAbsolute) &&
        value.residualAbsolute >= 0.0 &&
        std::isfinite(value.assemblyAbsolute) &&
        value.assemblyAbsolute >= 0.0 &&
        std::isfinite(value.maximumDampedEquilibriumResidual) &&
        value.maximumDampedEquilibriumResidual >= 0.0;
}

inline NumiHumanHandoffComparison numiHumanHandoffCompare(
    const std::span<const double> reference,
    const std::span<const double> candidate,
    const double absoluteTolerance,
    const double relativeTolerance
) noexcept {
    NumiHumanHandoffComparison result;
    if (reference.empty() || reference.size() != candidate.size() ||
        !numiHumanHandoffFinite(reference) ||
        !numiHumanHandoffFinite(candidate) ||
        !std::isfinite(absoluteTolerance) || absoluteTolerance < 0.0 ||
        !std::isfinite(relativeTolerance) || relativeTolerance < 0.0) {
        return result;
    }
    result.count = reference.size();
    long double squared = 0.0L;
    double worstDelta = -1.0;
    for (std::size_t index = 0u; index < reference.size(); ++index) {
        const double expected = reference[index];
        const double actual = candidate[index];
        const double delta = std::abs(actual - expected);
        const double tolerance = absoluteTolerance + relativeTolerance *
            std::max(std::abs(expected), std::abs(actual));
        const double normalized = tolerance > 0.0
            ? delta / tolerance
            : (delta == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        squared += static_cast<long double>(delta) * delta;
        result.maximumAbsoluteDelta = std::max(
            result.maximumAbsoluteDelta, delta);
        if (normalized > result.maximumNormalizedError ||
            (normalized == result.maximumNormalizedError &&
             delta > worstDelta)) {
            result.maximumNormalizedError = normalized;
            result.worstIndex = index;
            result.worstReference = expected;
            result.worstCandidate = actual;
            result.worstAbsoluteDelta = delta;
            worstDelta = delta;
        }
    }
    result.rmsAbsoluteDelta = std::sqrt(
        static_cast<double>(squared / reference.size()));
    result.passed = result.maximumNormalizedError <= 1.0;
    return result;
}

inline std::vector<double> numiHumanReconstructForceResidual(
    const std::span<const double> muscle,
    const std::span<const double> equality,
    const std::span<const double> limit,
    const std::span<const double> support,
    const std::span<const double> passive,
    const std::span<const double> gravityTarget
) {
    std::vector<double> result(kNumiHumanHandoffDofCount, 0.0);
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = muscle[index] + equality[index] + limit[index] +
            support[index] + passive[index] - gravityTarget[index];
    }
    return result;
}

inline void numiHumanWriteJsonString(
    std::ostream& output, const std::string_view value
) {
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20u) {
                const auto flags = output.flags();
                const auto fill = output.fill();
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<unsigned>(character);
                output.flags(flags);
                output.fill(fill);
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
}

inline void numiHumanWriteJsonVector(
    std::ostream& output, const std::span<const double> values
) {
    output << '[';
    for (std::size_t index = 0u; index < values.size(); ++index) {
        if (index != 0u) output << ',';
        output << values[index];
    }
    output << ']';
}

inline bool numiHumanDynamicSnapshotValid(
    const NumiHumanDynamicHandoffSnapshot& value,
    std::string& error
) {
    const auto muscleVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffMuscleCount &&
            numiHumanHandoffFinite(values);
    };
    const auto dofVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffDofCount &&
            numiHumanHandoffFinite(values);
    };
    if (!muscleVector(value.activation) ||
        !muscleVector(value.fiberLengthMeters) ||
        !muscleVector(value.sourceTotalActuatorForceNewtons) ||
        !muscleVector(value.excludedPassiveBiasForceNewtons) ||
        !muscleVector(value.drivenActuatorForceNewtons) ||
        !muscleVector(value.dampedEquilibriumResidual)) {
        error = "dynamic Human handoff must publish six finite 416-muscle vectors";
        return false;
    }
    if (!dofVector(value.generalizedMuscleForce) ||
        !dofVector(value.generalizedJointEqualityForce) ||
        !dofVector(value.generalizedPositionLimitForce) ||
        !dofVector(value.generalizedSupportForce) ||
        !dofVector(value.generalizedPassiveForce) ||
        !dofVector(value.gravityTarget) ||
        !dofVector(value.generalizedForceResidual)) {
        error = "dynamic Human handoff must publish seven finite 128-DoF force-owner vectors";
        return false;
    }
    if (value.fiberStateSource != kNumiHumanAcceptedFiberStateSource) {
        error = "dynamic Human handoff did not transport the accepted static fibre state";
        return false;
    }
    if (value.stateOwner.empty() || value.forceOwner.empty()) {
        error = "dynamic Human handoff owners are missing";
        return false;
    }
    error.clear();
    return true;
}

} // namespace detail

} // namespace metalrobo
