#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoffDetail.hpp"

#include <cmath>
#include <vector>

namespace metalrobo {

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
    const auto muscleVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffMuscleCount &&
            detail::numiHumanHandoffFinite(values);
    };
    const auto dofVector = [](const std::span<const double> values) {
        return values.size() == kNumiHumanHandoffDofCount &&
            detail::numiHumanHandoffFinite(values);
    };
    if (!muscleVector(input.staticActivation) ||
        !muscleVector(input.staticFiberLengthMeters) ||
        !muscleVector(input.staticSourceTotalActuatorForceNewtons) ||
        !muscleVector(input.staticExcludedPassiveBiasForceNewtons) ||
        !muscleVector(input.staticDrivenActuatorForceNewtons)) {
        result.error = "static Human handoff must publish five finite 416-muscle vectors";
        return result;
    }
    if (!dofVector(input.staticGeneralizedMuscleForce) ||
        !dofVector(input.staticGeneralizedJointEqualityForce) ||
        !dofVector(input.staticGeneralizedPositionLimitForce) ||
        !dofVector(input.staticGeneralizedSupportForce) ||
        !dofVector(input.staticGeneralizedPassiveForce) ||
        !dofVector(input.staticGravityTarget) ||
        !dofVector(input.staticGeneralizedForceResidual)) {
        result.error = "static Human handoff must publish seven finite 128-DoF force-owner vectors";
        return result;
    }
    result.inputValid = true;
    result.activation = detail::numiHumanHandoffCompare(
        input.staticActivation, input.dynamic.activation,
        thresholds.activationAbsolute, 0.0);
    result.fiberLength = detail::numiHumanHandoffCompare(
        input.staticFiberLengthMeters, input.dynamic.fiberLengthMeters,
        thresholds.fiberAbsoluteMeters, thresholds.fiberRelative);
    result.sourceTotalActuatorForce = detail::numiHumanHandoffCompare(
        input.staticSourceTotalActuatorForceNewtons,
        input.dynamic.sourceTotalActuatorForceNewtons,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.excludedPassiveBiasForce = detail::numiHumanHandoffCompare(
        input.staticExcludedPassiveBiasForceNewtons,
        input.dynamic.excludedPassiveBiasForceNewtons,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.drivenActuatorForce = detail::numiHumanHandoffCompare(
        input.staticDrivenActuatorForceNewtons,
        input.dynamic.drivenActuatorForceNewtons,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    std::vector<double> reconstructedMuscleForce(
        kNumiHumanHandoffMuscleCount, 0.0);
    for (std::size_t muscle = 0u;
         muscle < reconstructedMuscleForce.size(); ++muscle) {
        reconstructedMuscleForce[muscle] =
            input.dynamic.drivenActuatorForceNewtons[muscle] +
            input.dynamic.excludedPassiveBiasForceNewtons[muscle];
    }
    result.sourceForceDecomposition = detail::numiHumanHandoffCompare(
        input.dynamic.sourceTotalActuatorForceNewtons,
        reconstructedMuscleForce,
        thresholds.decompositionAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedMuscleForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedMuscleForce,
        input.dynamic.generalizedMuscleForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedJointEqualityForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedJointEqualityForce,
        input.dynamic.generalizedJointEqualityForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedPositionLimitForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedPositionLimitForce,
        input.dynamic.generalizedPositionLimitForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedSupportForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedSupportForce,
        input.dynamic.generalizedSupportForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedPassiveForce = detail::numiHumanHandoffCompare(
        input.staticGeneralizedPassiveForce,
        input.dynamic.generalizedPassiveForce,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.gravityTarget = detail::numiHumanHandoffCompare(
        input.staticGravityTarget, input.dynamic.gravityTarget,
        thresholds.forceAbsoluteNewtons, thresholds.forceRelative);
    result.generalizedForceResidual = detail::numiHumanHandoffCompare(
        input.staticGeneralizedForceResidual,
        input.dynamic.generalizedForceResidual,
        thresholds.residualAbsolute, thresholds.forceRelative);
    const std::vector<double> staticResidual =
        detail::numiHumanReconstructForceResidual(
            input.staticGeneralizedMuscleForce,
            input.staticGeneralizedJointEqualityForce,
            input.staticGeneralizedPositionLimitForce,
            input.staticGeneralizedSupportForce,
            input.staticGeneralizedPassiveForce,
            input.staticGravityTarget);
    const std::vector<double> dynamicResidual =
        detail::numiHumanReconstructForceResidual(
            input.dynamic.generalizedMuscleForce,
            input.dynamic.generalizedJointEqualityForce,
            input.dynamic.generalizedPositionLimitForce,
            input.dynamic.generalizedSupportForce,
            input.dynamic.generalizedPassiveForce,
            input.dynamic.gravityTarget);
    result.staticForceAssembly = detail::numiHumanHandoffCompare(
        input.staticGeneralizedForceResidual, staticResidual,
        thresholds.assemblyAbsolute, 0.0);
    result.dynamicForceAssembly = detail::numiHumanHandoffCompare(
        input.dynamic.generalizedForceResidual, dynamicResidual,
        thresholds.assemblyAbsolute, 0.0);
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
        result.sourceTotalActuatorForce.passed &&
        result.excludedPassiveBiasForce.passed &&
        result.drivenActuatorForce.passed;
    result.sourceForceDecompositionClosed =
        result.sourceForceDecomposition.passed;
    result.fullForceOwnerParity =
        result.generalizedMuscleForce.passed &&
        result.generalizedJointEqualityForce.passed &&
        result.generalizedPositionLimitForce.passed &&
        result.generalizedSupportForce.passed &&
        result.generalizedPassiveForce.passed &&
        result.gravityTarget.passed &&
        result.generalizedForceResidual.passed;
    result.staticForceAssemblyClosed = result.staticForceAssembly.passed;
    result.dynamicForceAssemblyClosed = result.dynamicForceAssembly.passed;
    result.generalizedForceParity = result.fullForceOwnerParity &&
        result.staticForceAssemblyClosed && result.dynamicForceAssemblyClosed;
    result.fiberTendonEquilibriumClosed =
        result.maximumDampedEquilibriumResidual <=
            thresholds.maximumDampedEquilibriumResidual;
    result.complete = result.activationAndFiberStateParity &&
        result.perMuscleForceParity &&
        result.sourceForceDecompositionClosed &&
        result.generalizedForceParity &&
        result.fiberTendonEquilibriumClosed;
    result.error.clear();
    return result;
}

} // namespace metalrobo
