#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoff.hpp"

#include <cstddef>
#include <vector>

namespace metalrobo::test::detail {

struct NumiHumanHandoffFixture {
    std::vector<double> activation = std::vector<double>(
        kNumiHumanHandoffMuscleCount, 0.2);
    std::vector<double> fiber = std::vector<double>(
        kNumiHumanHandoffMuscleCount, 0.1);
    std::vector<double> sourceTotal = std::vector<double>(
        kNumiHumanHandoffMuscleCount, 100.0);
    std::vector<double> excludedBias = std::vector<double>(
        kNumiHumanHandoffMuscleCount, 2.0);
    std::vector<double> driven = std::vector<double>(
        kNumiHumanHandoffMuscleCount, 98.0);
    std::vector<double> equilibriumResidual = std::vector<double>(
        kNumiHumanHandoffMuscleCount, 0.0);
    std::vector<double> muscle = std::vector<double>(
        kNumiHumanHandoffDofCount, 3.0);
    std::vector<double> equality = std::vector<double>(
        kNumiHumanHandoffDofCount, 2.0);
    std::vector<double> limit = std::vector<double>(
        kNumiHumanHandoffDofCount, 1.0);
    std::vector<double> support = std::vector<double>(
        kNumiHumanHandoffDofCount, 4.0);
    std::vector<double> passive = std::vector<double>(
        kNumiHumanHandoffDofCount, 0.5);
    std::vector<double> gravity = std::vector<double>(
        kNumiHumanHandoffDofCount, 9.0);
    std::vector<double> residual = std::vector<double>(
        kNumiHumanHandoffDofCount, 1.5);

    std::vector<double> dynamicActivation = activation;
    std::vector<double> dynamicFiber = fiber;
    std::vector<double> dynamicSourceTotal = sourceTotal;
    std::vector<double> dynamicExcludedBias = excludedBias;
    std::vector<double> dynamicDriven = driven;
    std::vector<double> dynamicMuscle = muscle;
    std::vector<double> dynamicEquality = equality;
    std::vector<double> dynamicLimit = limit;
    std::vector<double> dynamicSupport = support;
    std::vector<double> dynamicPassive = passive;
    std::vector<double> dynamicGravity = gravity;
    std::vector<double> dynamicResidual = residual;

    [[nodiscard]] NumiHumanStaticDynamicHandoffInput input() const {
        return {
            .staticActivation = activation,
            .staticFiberLengthMeters = fiber,
            .staticSourceTotalActuatorForceNewtons = sourceTotal,
            .staticExcludedPassiveBiasForceNewtons = excludedBias,
            .staticDrivenActuatorForceNewtons = driven,
            .staticGeneralizedMuscleForce = muscle,
            .staticGeneralizedJointEqualityForce = equality,
            .staticGeneralizedPositionLimitForce = limit,
            .staticGeneralizedSupportForce = support,
            .staticGeneralizedPassiveForce = passive,
            .staticGravityTarget = gravity,
            .staticGeneralizedForceResidual = residual,
            .dynamic = {
                .activation = dynamicActivation,
                .fiberLengthMeters = dynamicFiber,
                .sourceTotalActuatorForceNewtons = dynamicSourceTotal,
                .excludedPassiveBiasForceNewtons = dynamicExcludedBias,
                .drivenActuatorForceNewtons = dynamicDriven,
                .dampedEquilibriumResidual = equilibriumResidual,
                .generalizedMuscleForce = dynamicMuscle,
                .generalizedJointEqualityForce = dynamicEquality,
                .generalizedPositionLimitForce = dynamicLimit,
                .generalizedSupportForce = dynamicSupport,
                .generalizedPassiveForce = dynamicPassive,
                .gravityTarget = dynamicGravity,
                .generalizedForceResidual = dynamicResidual,
                .fiberStateSource = kNumiHumanAcceptedFiberStateSource,
                .stateOwner = "PersistentMetalHumanState.initial",
                .forceOwner = "PersistentMetalHumanState.pre_step_force",
            },
        };
    }
};

} // namespace metalrobo::test::detail
