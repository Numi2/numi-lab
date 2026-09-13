#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

namespace metalrobo {

inline constexpr std::size_t kNumiHumanHandoffMuscleCount = 416u;
inline constexpr std::size_t kNumiHumanHandoffDofCount = 128u;

struct NumiHumanHandoffThresholds {
    double activationAbsolute = 1.0e-7;
    double fiberAbsoluteMeters = 5.0e-7;
    double fiberRelative = 5.0e-6;
    double forceAbsoluteNewtons = 5.0e-2;
    double forceRelative = 5.0e-5;
    double residualAbsolute = 1.0e-3;
    double maximumDampedEquilibriumResidual = 1.0e-5;
};

struct NumiHumanHandoffComparison {
    std::size_t count = 0u;
    std::size_t worstIndex = 0u;
    double maximumAbsoluteDelta = 0.0;
    double rmsAbsoluteDelta = 0.0;
    double maximumNormalizedError = 0.0;
    double worstReference = 0.0;
    double worstCandidate = 0.0;
    bool passed = false;
};

struct NumiHumanDynamicHandoffSnapshot {
    std::span<const double> activation;
    std::span<const double> fiberLengthMeters;
    std::span<const double> actuatorForceNewtons;
    std::span<const double> passiveActuatorForceNewtons;
    std::span<const double> dampedEquilibriumResidual;
    std::span<const double> generalizedMuscleForce;
    std::span<const double> generalizedPassiveForce;
    std::span<const double> generalizedForceResidual;
    std::string_view stateOwner;
    std::string_view forceOwner;
};

struct NumiHumanStaticDynamicHandoffInput {
    std::span<const double> staticActivation;
    std::span<const double> staticFiberLengthMeters;
    std::span<const double> staticActuatorForceNewtons;
    std::span<const double> staticPassiveActuatorForceNewtons;
    std::span<const double> staticGeneralizedMuscleForce;
    std::span<const double> staticGeneralizedPassiveForce;
    std::span<const double> staticGeneralizedForceResidual;
    NumiHumanDynamicHandoffSnapshot dynamic;
};

struct NumiHumanStaticDynamicHandoffEvidence {
    bool inputValid = false;
    NumiHumanHandoffComparison activation;
    NumiHumanHandoffComparison fiberLength;
    NumiHumanHandoffComparison actuatorForce;
    NumiHumanHandoffComparison passiveActuatorForce;
    NumiHumanHandoffComparison generalizedMuscleForce;
    NumiHumanHandoffComparison generalizedPassiveForce;
    NumiHumanHandoffComparison generalizedForceResidual;
    std::size_t worstDampedEquilibriumMuscle = 0u;
    double maximumDampedEquilibriumResidual = 0.0;
    bool activationAndFiberStateParity = false;
    bool perMuscleForceParity = false;
    bool generalizedForceParity = false;
    bool fiberTendonEquilibriumClosed = false;
    bool complete = false;
    std::string error;
};

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
        std::isfinite(value.residualAbsolute) &&
        value.residualAbsolute >= 0.0 &&
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
            worstDelta = delta;
        }
    }
    result.rmsAbsoluteDelta = std::sqrt(
        static_cast<double>(squared / reference.size()));
    result.passed = result.maximumNormalizedError <= 1.0;
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
        !muscleVector(value.actuatorForceNewtons) ||
        !muscleVector(value.passiveActuatorForceNewtons) ||
        !muscleVector(value.dampedEquilibriumResidual)) {
        error = "dynamic Human handoff must publish five finite 416-muscle vectors";
        return false;
    }
    if (!dofVector(value.generalizedMuscleForce) ||
        !dofVector(value.generalizedPassiveForce) ||
        !dofVector(value.generalizedForceResidual)) {
        error = "dynamic Human handoff must publish three finite 128-DoF vectors";
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

[[nodiscard]] inline NumiHumanStaticDynamicHandoffEvidence
 auditNumiHumanStaticDynamicHandoff(
    const NumiHumanStaticDynamicHandoffInput& input,
    const NumiHumanHandoffThresholds& thresholds = {}
) {
    NumiHumanStaticDynamicHandoffEvidence result;
    if (!detail::numiHumanHandoffThresholdsValid(thresholds)) {
        result.error = "static-dynamic Human handoff thresholds are invalid";
        return result;
    }
    if (!detail::numiHumanDynamicSnapshotValid(input.dynamic, result.error)) {
        return result;
    }
    const auto staticMuscleVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffMuscleCount &&
            detail::numiHumanHandoffFinite(values);
    };
    const auto staticDofVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffDofCount &&
            detail::numiHumanHandoffFinite(values);
    };
    if (!staticMuscleVector(input.staticActivation) ||
        !staticMuscleVector(input.staticFiberLengthMeters) ||
        !staticMuscleVector(input.staticActuatorForceNewtons) ||
        !staticMuscleVector(input.staticPassiveActuatorForceNewtons)) {
        result.error = "static Human handoff must publish four finite 416-muscle vectors";
        return result;
    }
    if (!staticDofVector(input.staticGeneralizedMuscleForce) ||
        !staticDofVector(input.staticGeneralizedPassiveForce) ||
        !staticDofVector(input.staticGeneralizedForceResidual)) {
        result.error = "static Human handoff must publish three finite 128-DoF vectors";
        return result;
    }
    result.inputValid = true;
    result.activation = detail::numiHumanHandoffCompare(
        input.staticActivation, input.dynamic.activation,
        thresholds.activationAbsolute, 0.0);
    result.fiberLength = detail::numiHumanHandoffCompare(
        input.staticFiberLengthMeters, input.dynamic.fiberLengthMeters,
        thresholds.fiberAbsoluteMeters, thresholds.fiberRelative);
    result.actuatorForce = detail::numiHumanHandoffCompare(
        input.staticActuatorForceNewtons, input.dynamic.actuatorForceNewtons,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.passiveActuatorForce = detail::numiHumanHandoffCompare(
        input.staticPassiveActuatorForceNewtons,
        input.dynamic.passiveActuatorForceNewtons,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedMuscleForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedMuscleForce,
        input.dynamic.generalizedMuscleForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedPassiveForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedPassiveForce,
        input.dynamic.generalizedPassiveForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedForceResidual = detail::numiHumanHandoffCompare(
        input.staticGeneralizedForceResidual,
        input.dynamic.generalizedForceResidual,
        thresholds.residualAbsolute, thresholds.forceRelative);
    for (std::size_t index = 0u;
         index < input.dynamic.dampedEquilibriumResidual.size(); ++index) {
        const double value = std::abs(
            input.dynamic.dampedEquilibriumResidual[index]);
        if (value > result.maximumDampedEquilibriumResidual) {
            result.maximumDampedEquilibriumResidual = value;
            result.worstDampedEquilibriumMuscle = index;
        }
    }
    result.activationAndFiberStateParity =
        result.activation.passed && result.fiberLength.passed;
    result.perMuscleForceParity =
        result.actuatorForce.passed && result.passiveActuatorForce.passed;
    result.generalizedForceParity =
        result.generalizedMuscleForce.passed &&
        result.generalizedPassiveForce.passed &&
        result.generalizedForceResidual.passed;
    result.fiberTendonEquilibriumClosed =
        result.maximumDampedEquilibriumResidual <=
            thresholds.maximumDampedEquilibriumResidual;
    result.complete = result.activationAndFiberStateParity &&
        result.perMuscleForceParity && result.generalizedForceParity &&
        result.fiberTendonEquilibriumClosed;
    result.error.clear();
    return result;
}

[[nodiscard]] inline bool writeNumiHumanDynamicHandoffJson(
    std::ostream& output,
    const NumiHumanDynamicHandoffSnapshot& snapshot,
    std::string& error
) {
    if (!detail::numiHumanDynamicSnapshotValid(snapshot, error)) {
        return false;
    }
    const auto flags = output.flags();
    const auto precision = output.precision();
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\"schema\":\"numi.human.dynamic-handoff.v1\","
           << "\"stage\":\"pre_step\",\"completed_steps\":0,"
           << "\"state_owner\":";
    detail::numiHumanWriteJsonString(output, snapshot.stateOwner);
    output << ",\"force_owner\":";
    detail::numiHumanWriteJsonString(output, snapshot.forceOwner);
    output << ",\"activation_fp32\":";
    detail::numiHumanWriteJsonVector(output, snapshot.activation);
    output << ",\"fiber_length_m\":";
    detail::numiHumanWriteJsonVector(output, snapshot.fiberLengthMeters);
    output << ",\"actuator_force_n\":";
    detail::numiHumanWriteJsonVector(output, snapshot.actuatorForceNewtons);
    output << ",\"passive_actuator_force_n\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.passiveActuatorForceNewtons);
    output << ",\"damped_equilibrium_residual\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.dampedEquilibriumResidual);
    output << ",\"generalized_muscle_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedMuscleForce);
    output << ",\"generalized_passive_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedPassiveForce);
    output << ",\"force_residual\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedForceResidual);
    output << '}';
    output.flags(flags);
    output.precision(precision);
    if (!output.good()) {
        error = "could not write dynamic Human handoff JSON";
        return false;
    }
    error.clear();
    return true;
}

} // namespace metalrobo
