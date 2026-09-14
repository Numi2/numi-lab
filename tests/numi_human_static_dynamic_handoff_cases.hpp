#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoff.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace metalrobo::test {

inline std::size_t runNumiHumanStaticDynamicHandoffCases() {
    std::size_t checks = 0u;
    const auto require = [&checks](const bool condition, const char* label) {
        ++checks;
        if (!condition) throw std::runtime_error(label);
    };

    std::vector<double> activation(kNumiHumanHandoffMuscleCount, 0.2);
    std::vector<double> fiber(kNumiHumanHandoffMuscleCount, 0.1);
    std::vector<double> sourceTotal(kNumiHumanHandoffMuscleCount, 100.0);
    std::vector<double> excludedBias(kNumiHumanHandoffMuscleCount, 2.0);
    std::vector<double> driven(kNumiHumanHandoffMuscleCount, 98.0);
    std::vector<double> equilibriumResidual(
        kNumiHumanHandoffMuscleCount, 0.0);
    std::vector<double> generalizedMuscle(
        kNumiHumanHandoffDofCount, 3.0);
    std::vector<double> generalizedPassive(
        kNumiHumanHandoffDofCount, 0.5);
    std::vector<double> generalizedResidual(
        kNumiHumanHandoffDofCount, 0.001);

    std::vector<double> dynamicActivation = activation;
    std::vector<double> dynamicFiber = fiber;
    std::vector<double> dynamicSourceTotal = sourceTotal;
    std::vector<double> dynamicExcludedBias = excludedBias;
    std::vector<double> dynamicDriven = driven;
    std::vector<double> dynamicGeneralizedMuscle = generalizedMuscle;
    std::vector<double> dynamicGeneralizedPassive = generalizedPassive;
    std::vector<double> dynamicGeneralizedResidual = generalizedResidual;

    const auto input = [&]() {
        return NumiHumanStaticDynamicHandoffInput{
            .staticActivation = activation,
            .staticFiberLengthMeters = fiber,
            .staticSourceTotalActuatorForceNewtons = sourceTotal,
            .staticExcludedPassiveBiasForceNewtons = excludedBias,
            .staticDrivenActuatorForceNewtons = driven,
            .staticGeneralizedMuscleForce = generalizedMuscle,
            .staticGeneralizedPassiveForce = generalizedPassive,
            .staticGeneralizedForceResidual = generalizedResidual,
            .dynamic = {
                .activation = dynamicActivation,
                .fiberLengthMeters = dynamicFiber,
                .sourceTotalActuatorForceNewtons = dynamicSourceTotal,
                .excludedPassiveBiasForceNewtons = dynamicExcludedBias,
                .drivenActuatorForceNewtons = dynamicDriven,
                .dampedEquilibriumResidual = equilibriumResidual,
                .generalizedMuscleForce = dynamicGeneralizedMuscle,
                .generalizedPassiveForce = dynamicGeneralizedPassive,
                .generalizedForceResidual = dynamicGeneralizedResidual,
                .stateOwner = "PersistentMetalHumanState.initial",
                .forceOwner = "PersistentMetalHumanState.pre_step_force",
            },
        };
    };

    auto evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(evidence.inputValid && evidence.complete,
            "exact v2 handoff did not close parity");
    require(evidence.activationAndFiberStateParity &&
                evidence.perMuscleForceParity &&
                evidence.sourceForceDecompositionClosed &&
                evidence.generalizedForceParity &&
                evidence.fiberTendonEquilibriumClosed,
            "exact v2 handoff qualification is incomplete");

    dynamicFiber[37u] += 0.001;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.complete && evidence.fiberLength.worstIndex == 37u,
            "fibre-state drift was not localized");
    dynamicFiber = fiber;

    dynamicDriven[42u] += 5.0;
    dynamicSourceTotal[42u] += 5.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.perMuscleForceParity &&
                evidence.drivenActuatorForce.worstIndex == 42u &&
                evidence.sourceForceDecompositionClosed,
            "driven-force drift was not localized");
    dynamicDriven = driven;
    dynamicSourceTotal = sourceTotal;

    dynamicExcludedBias[13u] += 1.0;
    dynamicSourceTotal[13u] += 1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.perMuscleForceParity &&
                evidence.excludedPassiveBiasForce.worstIndex == 13u &&
                evidence.sourceForceDecompositionClosed,
            "excluded-bias drift was not localized");
    dynamicExcludedBias = excludedBias;
    dynamicSourceTotal = sourceTotal;

    dynamicSourceTotal[9u] += 1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.sourceForceDecompositionClosed &&
                evidence.sourceForceDecomposition.worstIndex == 9u,
            "source-force decomposition failure was accepted");
    dynamicSourceTotal = sourceTotal;

    dynamicGeneralizedMuscle[91u] += 5.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.generalizedForceParity &&
                evidence.generalizedMuscleForce.worstIndex == 91u,
            "generalized-force drift was not localized");
    dynamicGeneralizedMuscle = generalizedMuscle;

    equilibriumResidual[12u] = 0.01;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.fiberTendonEquilibriumClosed &&
                evidence.worstDampedEquilibriumMuscle == 12u,
            "damped fibre/tendon residual was accepted");
    equilibriumResidual.assign(kNumiHumanHandoffMuscleCount, 0.0);

    sourceTotal[3u] = 0.0;
    dynamicSourceTotal[3u] = 0.051;
    sourceTotal[4u] = 1.0e6;
    dynamicSourceTotal[4u] = 1.0e6 + 10.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(evidence.sourceTotalActuatorForce.worstIndex == 3u,
            "normalized-worst source force is incorrect");
    require(evidence.sourceTotalActuatorForce.maximumAbsoluteDelta == 10.0,
            "absolute source-force drift was overwritten by ranking");
    sourceTotal.assign(kNumiHumanHandoffMuscleCount, 100.0);
    dynamicSourceTotal = sourceTotal;

    std::vector<double> shortActivation(
        kNumiHumanHandoffMuscleCount - 1u, 0.2);
    auto malformed = input();
    malformed.dynamic.activation = shortActivation;
    evidence = auditNumiHumanStaticDynamicHandoff(malformed);
    require(!evidence.inputValid && !evidence.complete &&
                !evidence.error.empty(),
            "malformed dynamic handoff was accepted");

    dynamicGeneralizedResidual[7u] =
        std::numeric_limits<double>::quiet_NaN();
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.inputValid && !evidence.complete,
            "non-finite dynamic handoff was accepted");
    dynamicGeneralizedResidual = generalizedResidual;

    NumiHumanHandoffThresholds badThresholds;
    badThresholds.decompositionAbsoluteNewtons = -1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input(), badThresholds);
    require(!evidence.inputValid && !evidence.complete,
            "invalid decomposition threshold was accepted");

    auto snapshot = input().dynamic;
    snapshot.stateOwner = "state\"owner";
    snapshot.forceOwner = "force\nowner";
    std::ostringstream json;
    std::string error;
    require(writeNumiHumanDynamicHandoffJson(json, snapshot, error),
            "dynamic handoff v2 JSON writer failed");
    const std::string encoded = json.str();
    require(encoded.find(
                "\"schema\":\"numi.human.dynamic-handoff.v2\"") !=
                std::string::npos,
            "dynamic handoff v2 schema was not published");
    require(encoded.find(std::string(kNumiHumanPassiveBiasPolicy)) !=
                std::string::npos,
            "passive-bias policy was not published");
    require(encoded.find("\"source_total_actuator_force_n\":[") !=
                std::string::npos &&
                encoded.find("\"excluded_passive_bias_force_n\":[") !=
                std::string::npos &&
                encoded.find("\"driven_actuator_force_n\":[") !=
                std::string::npos,
            "v2 force decomposition vectors were not published");
    require(encoded.find("state\\\"owner") != std::string::npos &&
                encoded.find("force\\nowner") != std::string::npos,
            "dynamic handoff owners were not JSON escaped");

    snapshot.activation = shortActivation;
    std::ostringstream rejected;
    error.clear();
    require(!writeNumiHumanDynamicHandoffJson(rejected, snapshot, error) &&
                rejected.str().empty() && !error.empty(),
            "malformed v2 snapshot produced partial JSON");

    return checks;
}

} // namespace metalrobo::test
