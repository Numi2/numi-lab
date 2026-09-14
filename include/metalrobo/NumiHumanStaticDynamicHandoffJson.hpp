#pragma once

#include "metalrobo/NumiHumanStaticDynamicHandoffDetail.hpp"

#include <iomanip>
#include <limits>
#include <ostream>
#include <string>

namespace metalrobo {

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
           << "{\"schema\":\"numi.human.dynamic-handoff.v3\","
           << "\"stage\":\"pre_step\",\"completed_steps\":0,"
           << "\"passive_bias_policy\":";
    detail::numiHumanWriteJsonString(output, kNumiHumanPassiveBiasPolicy);
    output << ",\"gravity_convention\":";
    detail::numiHumanWriteJsonString(output, kNumiHumanGravityConvention);
    output << ",\"fiber_state_source\":";
    detail::numiHumanWriteJsonString(output, snapshot.fiberStateSource);
    output << ",\"state_owner\":";
    detail::numiHumanWriteJsonString(output, snapshot.stateOwner);
    output << ",\"force_owner\":";
    detail::numiHumanWriteJsonString(output, snapshot.forceOwner);
    output << ",\"activation_fp32\":";
    detail::numiHumanWriteJsonVector(output, snapshot.activation);
    output << ",\"fiber_length_m\":";
    detail::numiHumanWriteJsonVector(output, snapshot.fiberLengthMeters);
    output << ",\"source_total_actuator_force_n\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.sourceTotalActuatorForceNewtons);
    output << ",\"excluded_passive_bias_force_n\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.excludedPassiveBiasForceNewtons);
    output << ",\"driven_actuator_force_n\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.drivenActuatorForceNewtons);
    output << ",\"damped_equilibrium_residual\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.dampedEquilibriumResidual);
    output << ",\"generalized_muscle_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedMuscleForce);
    output << ",\"generalized_joint_equality_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedJointEqualityForce);
    output << ",\"generalized_position_limit_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedPositionLimitForce);
    output << ",\"generalized_support_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedSupportForce);
    output << ",\"generalized_passive_force\":";
    detail::numiHumanWriteJsonVector(
        output, snapshot.generalizedPassiveForce);
    output << ",\"gravity_target\":";
    detail::numiHumanWriteJsonVector(output, snapshot.gravityTarget);
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
