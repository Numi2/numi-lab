#pragma once

#include "metalrobo/NumiHumanPreStepMuscleParity.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace metalrobo::test {

inline std::size_t runNumiHumanPreStepMuscleParityCases() {
    std::size_t checks = 0u;
    const auto require = [&checks](const bool condition, const char* label) {
        ++checks;
        if (!condition) throw std::runtime_error(label);
    };

    std::vector<double> activation(kNumiHumanHandoffMuscleCount, 0.2);
    std::vector<double> fiber(kNumiHumanHandoffMuscleCount, 0.1);
    std::vector<double> sourceForce(kNumiHumanHandoffMuscleCount, 100.0);
    std::vector<double> drivenForce(kNumiHumanHandoffMuscleCount, 98.0);
    std::vector<double> generalized(kNumiHumanHandoffDofCount, 3.0);
    std::vector<float> dynamicGeneralized(
        kNumiHumanHandoffDofCount, 3.0f);
    std::vector<MRMujocoMuscleStateGPU> states(
        kNumiHumanHandoffMuscleCount);
    std::vector<MRMujocoMuscleResultGPU> results(
        kNumiHumanHandoffMuscleCount);
    for (std::size_t muscle = 0u;
         muscle < kNumiHumanHandoffMuscleCount; ++muscle) {
        states[muscle].excitationAndActivation = {
            0.2f, 0.2f, 0.1f, 0.0f};
        results[muscle].status =
            MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS;
        results[muscle].environment = 0u;
        results[muscle].muscleIndex =
            static_cast<std::uint32_t>(muscle);
        results[muscle].pathForceAndActivationDerivative = {
            0.3f, 0.0f, 100.0f, 0.0f};
        results[muscle].activeForceAndReserved = {
            98.0f, 0.0f, 0.0f, 0.0f};
        results[muscle].fiberStateTendonForceResidual = {
            0.1f, 0.0f, 100.0f, 0.0f};
    }
    const auto input = [&]() {
        return NumiHumanPreStepMuscleParityInput{
            .staticActivation = activation,
            .staticFiberLengthMeters = fiber,
            .staticSourceTotalActuatorForceNewtons = sourceForce,
            .staticDrivenActuatorForceNewtons = drivenForce,
            .staticGeneralizedMuscleForce = generalized,
            .initialStates = states,
            .evaluatedMuscles = results,
            .dynamicGeneralizedMuscleForce = dynamicGeneralized,
        };
    };

    auto evidence = auditNumiHumanPreStepMuscleParity(input());
    require(evidence.inputValid && evidence.complete,
            "exact pre-step muscle parity did not close");
    require(evidence.acceptedStateTransport &&
                evidence.perMuscleForceParity &&
                evidence.excludedPassiveBiasParity &&
                evidence.generalizedMuscleForceParity &&
                evidence.fiberTendonEquilibriumClosed,
            "pre-step muscle qualification is incomplete");

    states[17u].excitationAndActivation.z += 0.001f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.complete &&
                evidence.seededFiberLength.worstIndex == 17u,
            "seeded fibre drift was not localized");
    states[17u].excitationAndActivation.z = 0.1f;

    results[37u].fiberStateTendonForceResidual.x += 0.001f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.complete &&
                evidence.candidateFiberLength.worstIndex == 37u,
            "candidate fibre drift was not localized");
    results[37u].fiberStateTendonForceResidual.x = 0.1f;

    results[42u].pathForceAndActivationDerivative.z += 5.0f;
    results[42u].activeForceAndReserved.x += 5.0f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.perMuscleForceParity &&
                evidence.sourceTotalActuatorForce.worstIndex == 42u &&
                evidence.excludedPassiveBiasParity,
            "matched source/driven force drift was not localized");
    results[42u].pathForceAndActivationDerivative.z = 100.0f;
    results[42u].activeForceAndReserved.x = 98.0f;

    results[13u].activeForceAndReserved.x += 1.0f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.excludedPassiveBiasParity &&
                evidence.excludedPassiveBiasForce.worstIndex == 13u,
            "passive-bias policy drift was not localized");
    results[13u].activeForceAndReserved.x = 98.0f;

    dynamicGeneralized[91u] += 5.0f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.generalizedMuscleForceParity &&
                evidence.generalizedMuscleForce.worstIndex == 91u,
            "generalized muscle-force drift was not localized");
    dynamicGeneralized[91u] = 3.0f;

    results[12u].fiberStateTendonForceResidual.w = 0.01f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.fiberTendonEquilibriumClosed &&
                evidence.worstDampedEquilibriumMuscle == 12u,
            "damped fibre/tendon residual was accepted");
    results[12u].fiberStateTendonForceResidual.w = 0.0f;

    results[8u].fiberStateTendonForceResidual.y = 0.01f;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.acceptedStateTransport &&
                evidence.worstCandidateFiberVelocityMuscle == 8u,
            "nonstationary candidate fibre was accepted");
    results[8u].fiberStateTendonForceResidual.y = 0.0f;

    results[5u].muscleIndex = 6u;
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.inputValid && !evidence.complete,
            "misindexed GPU muscle result was accepted");
    results[5u].muscleIndex = 5u;

    results[7u].activeForceAndReserved.x =
        std::numeric_limits<float>::quiet_NaN();
    evidence = auditNumiHumanPreStepMuscleParity(input());
    require(!evidence.inputValid,
            "non-finite pre-step muscle result was accepted");

    return checks;
}

} // namespace metalrobo::test
