#pragma once

#include "metalrobo/NumiHumanDynamicStateSeed.hpp"
#include "numi_human_static_dynamic_handoff_fixture.hpp"

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
    detail::NumiHumanHandoffFixture fixture;

    auto evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(evidence.inputValid && evidence.complete,
            "exact v3 handoff did not close parity");
    require(evidence.activationAndFiberStateParity &&
                evidence.perMuscleForceParity &&
                evidence.sourceForceDecompositionClosed &&
                evidence.fullForceOwnerParity &&
                evidence.staticForceAssemblyClosed &&
                evidence.dynamicForceAssemblyClosed &&
                evidence.generalizedForceParity &&
                evidence.fiberTendonEquilibriumClosed,
            "exact v3 handoff qualification is incomplete");

    std::vector<MRMujocoMuscleStateGPU> seeded(1u);
    seeded.front().excitationAndActivation = {9.0f, 9.0f, 9.0f, 9.0f};
    auto seed = seedNumiHumanDynamicMuscleState(
        fixture.activation, fixture.fiber, seeded);
    require(seed.succeeded() && seeded.size() == 416u &&
                seeded[0u].excitationAndActivation.x == 0.2f &&
                seeded[0u].excitationAndActivation.z == 0.1f &&
                seeded[0u].excitationAndActivation.w == 0.0f,
            "accepted static fibre state was not seeded into dynamics");
    const auto acceptedSeed = seeded;
    fixture.fiber[12u] = 0.0;
    seed = seedNumiHumanDynamicMuscleState(
        fixture.activation, fixture.fiber, seeded);
    require(!seed.succeeded() && seed.failingIndex == 12u &&
                seeded[0u].excitationAndActivation.z ==
                    acceptedSeed[0u].excitationAndActivation.z,
            "invalid accepted fibre state mutated the sidecar");
    fixture.fiber.assign(416u, 0.1);

    fixture.dynamicFiber[37u] += 0.001;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.complete && evidence.fiberLength.worstIndex == 37u,
            "fibre-state drift was not localized");
    fixture.dynamicFiber = fixture.fiber;

    fixture.dynamicDriven[42u] += 5.0;
    fixture.dynamicSourceTotal[42u] += 5.0;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.perMuscleForceParity &&
                evidence.drivenActuatorForce.worstIndex == 42u &&
                evidence.sourceForceDecompositionClosed,
            "driven-force drift was not localized");
    fixture.dynamicDriven = fixture.driven;
    fixture.dynamicSourceTotal = fixture.sourceTotal;

    fixture.dynamicExcludedBias[13u] += 1.0;
    fixture.dynamicSourceTotal[13u] += 1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.perMuscleForceParity &&
                evidence.excludedPassiveBiasForce.worstIndex == 13u &&
                evidence.sourceForceDecompositionClosed,
            "excluded-bias drift was not localized");
    fixture.dynamicExcludedBias = fixture.excludedBias;
    fixture.dynamicSourceTotal = fixture.sourceTotal;

    fixture.dynamicSourceTotal[9u] += 1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.sourceForceDecompositionClosed &&
                evidence.sourceForceDecomposition.worstIndex == 9u,
            "source-force decomposition failure was accepted");
    fixture.dynamicSourceTotal = fixture.sourceTotal;

    const auto ownerDrift = [&](
        std::vector<double>& owner, const std::size_t index,
        const double ownerDelta, const double residualDelta,
        const NumiHumanHandoffComparison
            NumiHumanStaticDynamicHandoffEvidence::* member,
        const char* label
    ) {
        owner[index] += ownerDelta;
        fixture.dynamicResidual[index] += residualDelta;
        const auto changed = auditNumiHumanStaticDynamicHandoff(
            fixture.input());
        require(!changed.fullForceOwnerParity &&
                    !(changed.*member).passed &&
                    (changed.*member).worstIndex == index &&
                    changed.dynamicForceAssemblyClosed,
                label);
        owner[index] -= ownerDelta;
        fixture.dynamicResidual[index] -= residualDelta;
    };
    ownerDrift(fixture.dynamicMuscle, 91u, 5.0, 5.0,
        &NumiHumanStaticDynamicHandoffEvidence::generalizedMuscleForce,
        "muscle-force owner drift was not localized");
    ownerDrift(fixture.dynamicEquality, 47u, 5.0, 5.0,
        &NumiHumanStaticDynamicHandoffEvidence::generalizedJointEqualityForce,
        "equality-force owner drift was not localized");
    ownerDrift(fixture.dynamicLimit, 103u, 5.0, 5.0,
        &NumiHumanStaticDynamicHandoffEvidence::generalizedPositionLimitForce,
        "limit-force owner drift was not localized");
    ownerDrift(fixture.dynamicSupport, 2u, 5.0, 5.0,
        &NumiHumanStaticDynamicHandoffEvidence::generalizedSupportForce,
        "support-force owner drift was not localized");
    ownerDrift(fixture.dynamicPassive, 67u, 5.0, 5.0,
        &NumiHumanStaticDynamicHandoffEvidence::generalizedPassiveForce,
        "passive-force owner drift was not localized");
    ownerDrift(fixture.dynamicGravity, 8u, 5.0, -5.0,
        &NumiHumanStaticDynamicHandoffEvidence::gravityTarget,
        "gravity owner drift was not localized");

    fixture.dynamicResidual[7u] += 1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.dynamicForceAssemblyClosed &&
                evidence.dynamicForceAssembly.worstIndex == 7u,
            "dynamic force assembly corruption was accepted");
    fixture.dynamicResidual = fixture.residual;
    fixture.residual[11u] += 1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.staticForceAssemblyClosed &&
                evidence.staticForceAssembly.worstIndex == 11u,
            "static force assembly corruption was accepted");
    fixture.residual[11u] -= 1.0;

    fixture.equilibriumResidual[12u] = 0.01;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.fiberTendonEquilibriumClosed &&
                evidence.worstDampedEquilibriumMuscle == 12u,
            "damped fibre/tendon residual was accepted");
    fixture.equilibriumResidual.assign(416u, 0.0);

    fixture.sourceTotal[3u] = 0.0;
    fixture.dynamicSourceTotal[3u] = 0.051;
    fixture.sourceTotal[4u] = 1.0e6;
    fixture.dynamicSourceTotal[4u] = 1.0e6 + 10.0;
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(evidence.sourceTotalActuatorForce.worstIndex == 3u &&
                evidence.sourceTotalActuatorForce.maximumAbsoluteDelta == 10.0,
            "normalized source-force ranking is incorrect");
    fixture.sourceTotal.assign(416u, 100.0);
    fixture.dynamicSourceTotal = fixture.sourceTotal;

    std::vector<double> shortActivation(415u, 0.2);
    auto malformed = fixture.input();
    malformed.dynamic.activation = shortActivation;
    evidence = auditNumiHumanStaticDynamicHandoff(malformed);
    require(!evidence.inputValid && !evidence.complete,
            "malformed dynamic handoff was accepted");
    auto wrongSource = fixture.input();
    wrongSource.dynamic.fiberStateSource = "zero_length_sentinel";
    evidence = auditNumiHumanStaticDynamicHandoff(wrongSource);
    require(!evidence.inputValid && !evidence.complete,
            "untransported fibre state was accepted");

    fixture.dynamicResidual[7u] =
        std::numeric_limits<double>::quiet_NaN();
    evidence = auditNumiHumanStaticDynamicHandoff(fixture.input());
    require(!evidence.inputValid, "non-finite handoff was accepted");
    fixture.dynamicResidual = fixture.residual;
    NumiHumanHandoffThresholds badThresholds;
    badThresholds.assemblyAbsolute = -1.0;
    evidence = auditNumiHumanStaticDynamicHandoff(
        fixture.input(), badThresholds);
    require(!evidence.inputValid, "invalid threshold was accepted");

    auto snapshot = fixture.input().dynamic;
    snapshot.stateOwner = "state\"owner";
    snapshot.forceOwner = "force\nowner";
    std::ostringstream json;
    std::string error;
    require(writeNumiHumanDynamicHandoffJson(json, snapshot, error),
            "dynamic handoff v3 JSON writer failed");
    const std::string encoded = json.str();
    require(encoded.find("dynamic-handoff.v3") != std::string::npos &&
                encoded.find(std::string(kNumiHumanGravityConvention)) !=
                    std::string::npos &&
                encoded.find(std::string(kNumiHumanAcceptedFiberStateSource)) !=
                    std::string::npos,
            "v3 ownership conventions were not published");
    require(encoded.find("generalized_joint_equality_force") !=
                std::string::npos &&
                encoded.find("generalized_position_limit_force") !=
                std::string::npos &&
                encoded.find("generalized_support_force") !=
                std::string::npos &&
                encoded.find("gravity_target") != std::string::npos,
            "v3 force-owner vectors were not published");
    require(encoded.find("state\\\"owner") != std::string::npos &&
                encoded.find("force\\nowner") != std::string::npos,
            "dynamic handoff owners were not escaped");
    snapshot.activation = shortActivation;
    std::ostringstream rejected;
    error.clear();
    require(!writeNumiHumanDynamicHandoffJson(rejected, snapshot, error) &&
                rejected.str().empty(),
            "malformed v3 snapshot produced partial JSON");

    return checks;
}

} // namespace metalrobo::test
