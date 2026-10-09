#include "metalrobo/numi_human_motion_observer.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

bool near(double first, double second, double tolerance = 1.0e-12) {
    return std::abs(first - second) <= tolerance;
}
}

int main() {
    using namespace metalrobo::human::observer;

    require(!enabledFromEnvironment(nullptr), "unset observer must stay off");
    require(!enabledFromEnvironment(""), "empty observer value must stay off");
    require(!enabledFromEnvironment("0"), "zero observer value must stay off");
    require(enabledFromEnvironment("1"), "one observer value must enable");
    bool malformedRejected = false;
    try {
        (void)enabledFromEnvironment("true");
    } catch (const std::invalid_argument&) {
        malformedRejected = true;
    }
    require(malformedRejected, "malformed observer value must reject");
    const auto csvColumnCount = [](const std::string_view header) {
        std::size_t count = 1u;
        for (const char value : header) {
            if (value == ',') ++count;
        }
        return count;
    };
    require(kBodyMotionPartitionCsvHeader.starts_with("accepted_step,time_s,body_index,body_name,") &&
                csvColumnCount(kBodyMotionPartitionCsvHeader) == 27u,
            "body observer CSV schema lost accepted-state identity or motion columns");
    require(kSupportPointMotionCsvHeader.find("accepted_time_s,pre_step_index,pre_step_time_s") !=
                std::string_view::npos &&
            kSupportPointMotionCsvHeader.find("pre_step_velocity_fingerprint_fnv64") !=
                std::string_view::npos &&
            kSupportPointMotionCsvHeader.find("velocity_basis") != std::string_view::npos &&
            csvColumnCount(kSupportPointMotionCsvHeader) == 19u,
            "support observer CSV schema must identify times, velocity basis, and all components");
    require(kSupportSelectorCsvHeader.find(
                "accepted_time_s,pre_dynamics_step_index,pre_dynamics_time_s") !=
                std::string_view::npos &&
            kSupportSelectorCsvHeader.find("selected_skin_vertex_local") !=
                std::string_view::npos &&
            csvColumnCount(kSupportSelectorCsvHeader) == 16u,
            "support selector CSV schema must identify time and selected vertex");

    std::vector<metalrobo::ArticulatedBodyKinematics> kinematics(2u);
    kinematics[0].centerOfMassPosition = {11.0, 22.0, 33.0};
    kinematics[0].orientation = {0.0, 0.0, 0.0, 1.0};
    kinematics[0].linearVelocity = {1.0, 2.0, 3.0};
    kinematics[0].angularVelocity = {0.1, 0.2, 0.3};
    kinematics[1].centerOfMassPosition = {12.0, 19.0, 35.0};
    kinematics[1].orientation = {0.0, 0.0, 1.0, 0.0};
    kinematics[1].linearVelocity = {-1.0, 0.0, 2.0};
    kinematics[1].angularVelocity = {-0.1, 0.0, 0.2};
    const auto originalKinematics = kinematics;
    const std::array<double, 2u> masses{2.0, 3.0};
    const std::array<double, 3u> root{10.0, 20.0, 30.0};
    const auto rows = makeBodyMotionPartition(kinematics, masses, 7u, root);
    require(rows.size() == 2u && rows[0].bodyIndex == 7u &&
                rows[1].bodyIndex == 8u,
            "body partition IDs/count are wrong");
    require(near(rows[0].centerOfMassRelativeToRootMeters[0], 1.0) &&
                near(rows[0].centerOfMassRelativeToRootMeters[1], 2.0) &&
                near(rows[0].centerOfMassRelativeToRootMeters[2], 3.0) &&
                near(rows[0].massWeightedRelativeComKgMeters[2], 6.0) &&
                near(rows[1].massWeightedRelativeComKgMeters[1], -3.0) &&
                near(rows[1].linearMomentumKgMetersPerSecond[2], 6.0),
            "mass-weighted root-relative partition values are wrong");
    require(kinematics[0].centerOfMassPosition == originalKinematics[0].centerOfMassPosition &&
                kinematics[0].linearVelocity == originalKinematics[0].linearVelocity &&
                kinematics[1].orientation == originalKinematics[1].orientation,
            "body motion observer modified the supplied accepted state");

    const std::vector<float> jacobians{
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> preStepVelocity{0.5f, -1.0f};
    const auto originalJacobians = jacobians;
    const auto originalVelocity = preStepVelocity;
    const auto support = makeSupportPointVelocity(
        jacobians, 0u, 2u, preStepVelocity, {0.0, 0.0, 1.0});
    require(near(support.worldMetersPerSecond[0], -1.5) &&
                near(support.worldMetersPerSecond[1], -2.5) &&
                near(support.worldMetersPerSecond[2], -3.5) &&
                near(support.tangent0MetersPerSecond, -1.5) &&
                near(support.tangent1MetersPerSecond, -2.5) &&
                near(support.tangentialSpeedMetersPerSecond, std::sqrt(8.5)),
            "pre-step Jv/tangent projection is wrong");
    require(jacobians == originalJacobians &&
                preStepVelocity == originalVelocity,
            "support-point observer modified its Jacobian or velocity input");

    bool invalidExtentRejected = false;
    try {
        (void)makeSupportPointVelocity(jacobians, 1u, 2u, preStepVelocity,
                                       {0.0, 0.0, 1.0});
    } catch (const std::invalid_argument&) {
        invalidExtentRejected = true;
    }
    require(invalidExtentRejected,
            "out-of-range point query must not produce a slip sample");

    std::cout << "motion observer pure-helper tests passed\n";
    return 0;
}
