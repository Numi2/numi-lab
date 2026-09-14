#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoffDetail.hpp"
#include "metalrobo/mujoco_muscle_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct NumiHumanPreStepMuscleParityThresholds {
    double activationAbsolute = 1.0e-7;
    double fiberAbsoluteMeters = 5.0e-7;
    double fiberRelative = 5.0e-6;
    double forceAbsoluteNewtons = 5.0e-2;
    double forceRelative = 5.0e-5;
    double generalizedForceAbsolute = 5.0e-2;
    double maximumSeededFiberVelocityMetersPerSecond = 0.0;
    double maximumCandidateFiberVelocityMetersPerSecond = 1.0e-3;
    double maximumDampedEquilibriumResidual = 1.0e-5;
};

struct NumiHumanPreStepMuscleParityInput {
    std::span<const double> staticActivation;
    std::span<const double> staticFiberLengthMeters;
    std::span<const double> staticSourceTotalActuatorForceNewtons;
    std::span<const double> staticDrivenActuatorForceNewtons;
    std::span<const double> staticGeneralizedMuscleForce;
    std::span<const MRMujocoMuscleStateGPU> initialStates;
    std::span<const MRMujocoMuscleResultGPU> evaluatedMuscles;
    std::span<const float> dynamicGeneralizedMuscleForce;
};

struct NumiHumanPreStepMuscleParityEvidence {
    bool inputValid = false;
    NumiHumanHandoffComparison excitation;
    NumiHumanHandoffComparison activation;
    NumiHumanHandoffComparison seededFiberLength;
    NumiHumanHandoffComparison candidateFiberLength;
    NumiHumanHandoffComparison sourceTotalActuatorForce;
    NumiHumanHandoffComparison drivenActuatorForce;
    NumiHumanHandoffComparison excludedPassiveBiasForce;
    NumiHumanHandoffComparison generalizedMuscleForce;
    std::size_t worstSeededFiberVelocityMuscle = 0u;
    double maximumSeededFiberVelocityMetersPerSecond = 0.0;
    std::size_t worstCandidateFiberVelocityMuscle = 0u;
    double maximumCandidateFiberVelocityMetersPerSecond = 0.0;
    std::size_t worstDampedEquilibriumMuscle = 0u;
    double maximumDampedEquilibriumResidual = 0.0;
    bool acceptedStateTransport = false;
    bool perMuscleForceParity = false;
    bool excludedPassiveBiasParity = false;
    bool generalizedMuscleForceParity = false;
    bool fiberTendonEquilibriumClosed = false;
    bool complete = false;
    std::string error;
};

namespace detail {

[[nodiscard]] inline bool numiHumanPreStepThresholdsValid(
    const NumiHumanPreStepMuscleParityThresholds& value
) noexcept {
    const double values[] = {
        value.activationAbsolute,
        value.fiberAbsoluteMeters,
        value.fiberRelative,
        value.forceAbsoluteNewtons,
        value.forceRelative,
        value.generalizedForceAbsolute,
        value.maximumSeededFiberVelocityMetersPerSecond,
        value.maximumCandidateFiberVelocityMetersPerSecond,
        value.maximumDampedEquilibriumResidual,
    };
    return std::all_of(std::begin(values), std::end(values),
        [](const double item) {
            return std::isfinite(item) && item >= 0.0;
        });
}

[[nodiscard]] inline bool numiHumanPreStepResultFinite(
    const MRMujocoMuscleResultGPU& value
) noexcept {
    const double values[] = {
        value.pathForceAndActivationDerivative.z,
        value.activeForceAndReserved.x,
        value.fiberStateTendonForceResidual.x,
        value.fiberStateTendonForceResidual.y,
        value.fiberStateTendonForceResidual.w,
    };
    return std::all_of(std::begin(values), std::end(values),
        [](const double item) { return std::isfinite(item); });
}

} // namespace detail

[[nodiscard]] inline NumiHumanPreStepMuscleParityEvidence
auditNumiHumanPreStepMuscleParity(
    const NumiHumanPreStepMuscleParityInput& input,
    const NumiHumanPreStepMuscleParityThresholds& thresholds = {}
) {
    NumiHumanPreStepMuscleParityEvidence evidence;
    if (!detail::numiHumanPreStepThresholdsValid(thresholds)) {
        evidence.error = "pre-step Human muscle parity thresholds are invalid";
        return evidence;
    }
    const auto muscleVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffMuscleCount &&
            detail::numiHumanHandoffFinite(values);
    };
    if (!muscleVector(input.staticActivation) ||
        !muscleVector(input.staticFiberLengthMeters) ||
        !muscleVector(input.staticSourceTotalActuatorForceNewtons) ||
        !muscleVector(input.staticDrivenActuatorForceNewtons) ||
        input.staticGeneralizedMuscleForce.size() !=
            kNumiHumanHandoffDofCount ||
        !detail::numiHumanHandoffFinite(
            input.staticGeneralizedMuscleForce) ||
        input.initialStates.size() != kNumiHumanHandoffMuscleCount ||
        input.evaluatedMuscles.size() != kNumiHumanHandoffMuscleCount ||
        input.dynamicGeneralizedMuscleForce.size() !=
            kNumiHumanHandoffDofCount) {
        evidence.error =
            "pre-step Human muscle parity requires 416 muscle states/results and 128 generalized forces";
        return evidence;
    }

    std::vector<double> excitation(kNumiHumanHandoffMuscleCount);
    std::vector<double> activation(kNumiHumanHandoffMuscleCount);
    std::vector<double> seededFiber(kNumiHumanHandoffMuscleCount);
    std::vector<double> candidateFiber(kNumiHumanHandoffMuscleCount);
    std::vector<double> sourceForce(kNumiHumanHandoffMuscleCount);
    std::vector<double> drivenForce(kNumiHumanHandoffMuscleCount);
    std::vector<double> staticExcludedBias(kNumiHumanHandoffMuscleCount);
    std::vector<double> dynamicExcludedBias(kNumiHumanHandoffMuscleCount);
    std::vector<double> dynamicGeneralized(kNumiHumanHandoffDofCount);

    for (std::size_t muscle = 0u;
         muscle < kNumiHumanHandoffMuscleCount; ++muscle) {
        const auto& state = input.initialStates[muscle];
        const auto& result = input.evaluatedMuscles[muscle];
        const double stateValues[] = {
            state.excitationAndActivation.x,
            state.excitationAndActivation.y,
            state.excitationAndActivation.z,
            state.excitationAndActivation.w,
        };
        if (result.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
            result.environment != 0u || result.muscleIndex != muscle ||
            !std::all_of(std::begin(stateValues), std::end(stateValues),
                [](const double item) { return std::isfinite(item); }) ||
            !detail::numiHumanPreStepResultFinite(result)) {
            evidence.error =
                "pre-step Human muscle result/state is malformed at muscle " +
                std::to_string(muscle);
            return evidence;
        }
        excitation[muscle] = state.excitationAndActivation.x;
        activation[muscle] = state.excitationAndActivation.y;
        seededFiber[muscle] = state.excitationAndActivation.z;
        candidateFiber[muscle] =
            result.fiberStateTendonForceResidual.x;
        sourceForce[muscle] =
            result.pathForceAndActivationDerivative.z;
        drivenForce[muscle] = result.activeForceAndReserved.x;
        staticExcludedBias[muscle] =
            input.staticSourceTotalActuatorForceNewtons[muscle] -
            input.staticDrivenActuatorForceNewtons[muscle];
        dynamicExcludedBias[muscle] =
            sourceForce[muscle] - drivenForce[muscle];

        const double seededVelocity = std::abs(
            static_cast<double>(state.excitationAndActivation.w));
        if (seededVelocity >
            evidence.maximumSeededFiberVelocityMetersPerSecond) {
            evidence.maximumSeededFiberVelocityMetersPerSecond =
                seededVelocity;
            evidence.worstSeededFiberVelocityMuscle = muscle;
        }
        const double candidateVelocity = std::abs(
            static_cast<double>(
                result.fiberStateTendonForceResidual.y));
        if (candidateVelocity >
            evidence.maximumCandidateFiberVelocityMetersPerSecond) {
            evidence.maximumCandidateFiberVelocityMetersPerSecond =
                candidateVelocity;
            evidence.worstCandidateFiberVelocityMuscle = muscle;
        }
        const double residual = std::abs(static_cast<double>(
            result.fiberStateTendonForceResidual.w));
        if (residual > evidence.maximumDampedEquilibriumResidual) {
            evidence.maximumDampedEquilibriumResidual = residual;
            evidence.worstDampedEquilibriumMuscle = muscle;
        }
    }
    for (std::size_t dof = 0u;
         dof < kNumiHumanHandoffDofCount; ++dof) {
        const double value = input.dynamicGeneralizedMuscleForce[dof];
        if (!std::isfinite(value)) {
            evidence.error =
                "pre-step Human generalized muscle force is non-finite at coordinate " +
                std::to_string(dof);
            return evidence;
        }
        dynamicGeneralized[dof] = value;
    }

    evidence.inputValid = true;
    evidence.excitation = detail::numiHumanHandoffCompare(
        input.staticActivation, excitation,
        thresholds.activationAbsolute, 0.0);
    evidence.activation = detail::numiHumanHandoffCompare(
        input.staticActivation, activation,
        thresholds.activationAbsolute, 0.0);
    evidence.seededFiberLength = detail::numiHumanHandoffCompare(
        input.staticFiberLengthMeters, seededFiber,
        thresholds.fiberAbsoluteMeters, thresholds.fiberRelative);
    evidence.candidateFiberLength = detail::numiHumanHandoffCompare(
        input.staticFiberLengthMeters, candidateFiber,
        thresholds.fiberAbsoluteMeters, thresholds.fiberRelative);
    evidence.sourceTotalActuatorForce = detail::numiHumanHandoffCompare(
        input.staticSourceTotalActuatorForceNewtons, sourceForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    evidence.drivenActuatorForce = detail::numiHumanHandoffCompare(
        input.staticDrivenActuatorForceNewtons, drivenForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    evidence.excludedPassiveBiasForce = detail::numiHumanHandoffCompare(
        staticExcludedBias, dynamicExcludedBias,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    evidence.generalizedMuscleForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedMuscleForce, dynamicGeneralized,
        thresholds.generalizedForceAbsolute, thresholds.forceRelative);

    evidence.acceptedStateTransport =
        evidence.excitation.passed && evidence.activation.passed &&
        evidence.seededFiberLength.passed &&
        evidence.candidateFiberLength.passed &&
        evidence.maximumSeededFiberVelocityMetersPerSecond <=
            thresholds.maximumSeededFiberVelocityMetersPerSecond &&
        evidence.maximumCandidateFiberVelocityMetersPerSecond <=
            thresholds.maximumCandidateFiberVelocityMetersPerSecond;
    evidence.perMuscleForceParity =
        evidence.sourceTotalActuatorForce.passed &&
        evidence.drivenActuatorForce.passed;
    evidence.excludedPassiveBiasParity =
        evidence.excludedPassiveBiasForce.passed;
    evidence.generalizedMuscleForceParity =
        evidence.generalizedMuscleForce.passed;
    evidence.fiberTendonEquilibriumClosed =
        evidence.maximumDampedEquilibriumResidual <=
            thresholds.maximumDampedEquilibriumResidual;
    evidence.complete = evidence.acceptedStateTransport &&
        evidence.perMuscleForceParity &&
        evidence.excludedPassiveBiasParity &&
        evidence.generalizedMuscleForceParity &&
        evidence.fiberTendonEquilibriumClosed;
    evidence.error.clear();
    return evidence;
}

} // namespace metalrobo
