#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoff.hpp"

#include <cmath>
#include <cstddef>
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

    std::vector<double> staticActivation(
        kNumiHumanHandoffMuscleCount, 0.2);
    std::vector<double> staticFiberLength(
        kNumiHumanHandoffMuscleCount, 0.1);
    std::vector<double> staticActuatorForce(
        kNumiHumanHandoffMuscleCount, 100.0);
    std::vector<double> staticPassiveActuatorForce(
        kNumiHumanHandoffMuscleCount, 2.0);
    std::vector<double> staticGeneralizedMuscleForce(
        kNumiHumanHandoffDofCount, 3.0);
    std::vector<double> staticGeneralizedPassiveForce(
        kNumiHumanHandoffDofCount, 0.5);
    std::vector<double> staticGeneralizedResidual(
        kNumiHumanHandoffDofCount, 0.001);

    std::vector<double> dynamicActivation = staticActivation;
    std::vector<double> dynamicFiberLength = staticFiberLength;
    std::vector<double> dynamicActuatorForce = staticActuatorForce;
    std::vector<double> dynamicPassiveActuatorForce =
        staticPassiveActuatorForce;
    std::vector<double> dynamicEquilibriumResidual(
        kNumiHumanHandoffMuscleCount, 0.0);
    std::vector<double> dynamicGeneralizedMuscleForce =
        staticGeneralizedMuscleForce;
    std::vector<double> dynamicGeneralizedPassiveForce =
        staticGeneralizedPassiveForce;
    std::vector<double> dynamicGeneralizedResidual =
        staticGeneralizedResidual;

    const auto input = [&]() {
        return NumiHumanStaticDynamicHandoffInput{
            .staticActivation = staticActivation,
            .staticFiberLengthMeters = staticFiberLength,
            .staticActuatorForceNewtons = staticActuatorForce,
            .staticPassiveActuatorForceNewtons =
                staticPassiveActuatorForce,
            .staticGeneralizedMuscleForce =
                staticGeneralizedMuscleForce,
            .staticGeneralizedPassiveForce =
                staticGeneralizedPassiveForce,
            .staticGeneralizedForceResidual = staticGeneralizedResidual,
            .dynamic = {
                .activation = dynamicActivation,
                .fiberLengthMeters = dynamicFiberLength,
                .actuatorForceNewtons = dynamicActuatorForce,
                .passiveActuatorForceNewtons =
                    dynamicPassiveActuatorForce,
                .dampedEquilibriumResidual =
                    dynamicEquilibriumResidual,
                .generalizedMuscleForce =
                    dynamicGeneralizedMuscleForce,
                .generalizedPassiveForce =
                    dynamicGeneralizedPassiveForce,
                .generalizedForceResidual =
                    dynamicGeneralizedResidual,
                .stateOwner = "PersistentMetalHumanState.initial",
                .forceOwner =
                    "PersistentMetalHumanState.pre_step_force",
            },
        };
    };

    auto evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(evidence.inputValid, "exact handoff input rejected");
    require(evidence.complete, "exact handoff did not close parity");
    require(evidence.activationAndFiberStateParity,
            "exact activation/fibre state did not close");
    require(evidence.perMuscleForceParity,
            "exact per-muscle force did not close");
    require(evidence.generalizedForceParity,
            "exact generalized force did not close");
    require(evidence.fiberTendonEquilibriumClosed,
            "exact fibre/tendon equilibrium did not close");
    require(evidence.activation.count == kNumiHumanHandoffMuscleCount &&
                evidence.generalizedForceResidual.count ==
                    kNumiHumanHandoffDofCount,
            "handoff evidence dimensions drifted");

    dynamicFiberLength[37u] += 0.001;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.complete &&
                !evidence.activationAndFiberStateParity,
            "fibre-state drift was accepted");
    require(evidence.fiberLength.worstIndex == 37u,
            "wrong fibre-state drift owner reported");
    require(evidence.fiberLength.maximumAbsoluteDelta >= 0.000999,
            "fibre-state drift magnitude was lost");
    dynamicFiberLength = staticFiberLength;

    dynamicGeneralizedMuscleForce[91u] += 5.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.complete && !evidence.generalizedForceParity,
            "generalized-force drift was accepted");
    require(evidence.generalizedMuscleForce.worstIndex == 91u,
            "wrong generalized-force drift owner reported");
    dynamicGeneralizedMuscleForce = staticGeneralizedMuscleForce;

    dynamicEquilibriumResidual[12u] = 0.01;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(!evidence.complete &&
                !evidence.fiberTendonEquilibriumClosed,
            "damped fibre/tendon residual was accepted");
    require(evidence.worstDampedEquilibriumMuscle == 12u &&
                evidence.maximumDampedEquilibriumResidual == 0.01,
            "wrong damped-equilibrium owner reported");
    dynamicEquilibriumResidual.assign(
        kNumiHumanHandoffMuscleCount, 0.0);

    staticActuatorForce[3u] = 0.0;
    dynamicActuatorForce[3u] = 0.051;
    staticActuatorForce[4u] = 1.0e6;
    dynamicActuatorForce[4u] = 1.0e6 + 10.0;
    evidence = auditNumiHumanStaticDynamicHandoff(input());
    require(evidence.actuatorForce.worstIndex == 3u,
            "normalized-worst muscle selection is incorrect");
    require(evidence.actuatorForce.maximumAbsoluteDelta == 10.0,
            "maximum absolute force drift was overwritten by normalized rank");
    staticActuatorForce.assign(kNumiHumanHandoffMuscleCount, 100.0);
    dynamicActuatorForce = staticActuatorForce;

    std::vector<double> shortActivation(
        kNumiHumanHandoffMuscleCount - 1u, 0.2);
    auto malformed = input();
    malformed.dynamic.activation = shortActivation;
    evidence = auditNumiHumanStaticDynamicHandoff(malformed);
    require(!evidence.inputValid && !evidence.complete &&
                !evidence.error.empty(),
            "malformed dynamic handoff dimensions were accepted");

    auto nonfinite = input();
    dynamicGeneralizedResidual[7u] =
        std::numeric_limits<double>::quiet_NaN();
    evidence = auditNumiHumanStaticDynamicHandoff(nonfinite);
    require(!evidence.inputValid && !evidence.complete,
            "non-finite dynamic handoff value was accepted");
    dynamicGeneralizedResidual = staticGeneralizedResidual;

    NumiHumanHandoffThresholds badThresholds;
    badThresholds.forceRelative = -1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(
        input(), badThresholds);
    require(!evidence.inputValid && !evidence.complete &&
                !evidence.error.empty(),
            "invalid handoff thresholds were accepted");

    auto snapshot = input().dynamic;
    snapshot.stateOwner = "state\"owner";
    snapshot.forceOwner = "force\nowner";
    std::ostringstream json;
    std::string error;
    require(writeNumiHumanDynamicHandoffJson(json, snapshot, error),
            "dynamic handoff JSON writer failed");
    const std::string encoded = json.str();
    require(encoded.find(
                "\"schema\":\"numi.human.dynamic-handoff.v1\"") !=
                std::string::npos,
            "dynamic handoff schema was not published");
    require(encoded.find("\"stage\":\"pre_step\"") !=
                std::string::npos &&
                encoded.find("\"completed_steps\":0") !=
                std::string::npos,
            "dynamic handoff stage identity was not published");
    require(encoded.find("state\\\"owner") != std::string::npos &&
                encoded.find("force\\nowner") != std::string::npos,
            "dynamic handoff owners were not JSON escaped");
    require(encoded.find("\"activation_fp32\":[") !=
                std::string::npos &&
                encoded.find("\"force_residual\":[") !=
                std::string::npos,
            "dynamic handoff vectors were not published");

    snapshot.activation = shortActivation;
    std::ostringstream rejectedJson;
    error.clear();
    require(!writeNumiHumanDynamicHandoffJson(
                rejectedJson, snapshot, error) &&
                rejectedJson.str().empty() && !error.empty(),
            "malformed dynamic snapshot produced partial JSON");

    return checks;
}

} // namespace metalrobo::test
