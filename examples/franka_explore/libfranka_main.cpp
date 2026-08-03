#include "trajectory.hpp"

#include <franka/exception.h>
#include <franka/robot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string robotIp;
    std::string output = "franka-hardware-exploration.csv";
    bool armed = false;
    bool freeSpaceConfirmed = false;
    bool userStopConfirmed = false;
};

Options parseOptions(const int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        const auto value = [&]() -> const char* {
            if (++index >= argc) {
                throw std::runtime_error(
                    std::string{"missing value after "} +
                    std::string{argument}
                );
            }
            return argv[index];
        };
        if (argument == "--robot-ip") {
            result.robotIp = value();
        } else if (argument == "--output") {
            result.output = value();
        } else if (argument == "--arm-hardware") {
            result.armed = true;
        } else if (argument == "--confirm-free-space") {
            result.freeSpaceConfirmed = true;
        } else if (argument == "--confirm-user-stop") {
            result.userStopConfirmed = true;
        } else if (argument == "--help") {
            std::cout
                << "usage: numi_franka_libfranka_explore --robot-ip IP "
                   "--arm-hardware --confirm-free-space "
                   "--confirm-user-stop [--output PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error(
                std::string{"unknown argument: "} + std::string{argument}
            );
        }
    }
    if (result.robotIp.empty() || !result.armed ||
        !result.freeSpaceConfirmed || !result.userStopConfirmed) {
        throw std::runtime_error(
            "hardware remains disarmed; robot IP, explicit arming, free-space "
            "confirmation, and user-stop confirmation are all required"
        );
    }
    return result;
}

void setOfficialDefaultBehavior(franka::Robot& robot) {
    // Exact collision thresholds from libfranka examples/examples_common.cpp
    // at 85912fe02258d8cb811d3eff1f11e52ce89e3217.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}
    );
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
}

struct Sample {
    double time = 0.0;
    std::size_t waypoint = 0u;
    std::array<double, 7u> q{};
    std::array<double, 7u> commanded{};
    double commandSuccessRate = 0.0;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        franka::Robot robot{options.robotIp};
        setOfficialDefaultBehavior(robot);
        std::vector<Sample> samples;
        samples.reserve(20000u);
        double totalTime = 0.0;

        for (std::size_t waypoint = 0u;
             waypoint < metalrobo::franka_explore::kOfficialWaypoints.size();
             ++waypoint) {
            const auto start = robot.readOnce().q;
            const auto target =
                metalrobo::franka_explore::kOfficialWaypoints[waypoint].q;
            double elapsed = 0.0;
            robot.control(
                [&](const franka::RobotState& state, const franka::Duration period) {
                    elapsed += period.toSec();
                    const double progress = elapsed /
                        metalrobo::franka_explore::kTransitionSeconds;
                    const auto commanded =
                        metalrobo::franka_explore::interpolate(
                            start,
                            target,
                            progress
                        );
                    if (samples.size() < samples.capacity()) {
                        samples.push_back({
                            totalTime + elapsed,
                            waypoint,
                            state.q,
                            commanded,
                            state.control_command_success_rate,
                        });
                    }
                    franka::JointPositions output{commanded};
                    if (elapsed >=
                        metalrobo::franka_explore::kTransitionSeconds +
                        metalrobo::franka_explore::kDwellSeconds) {
                        return franka::MotionFinished(output);
                    }
                    return output;
                },
                franka::ControllerMode::kJointImpedance,
                true,
                100.0
            );
            totalTime += metalrobo::franka_explore::kTransitionSeconds +
                metalrobo::franka_explore::kDwellSeconds;
        }

        std::ofstream csv{options.output};
        if (!csv) {
            throw std::runtime_error("could not open hardware evidence artifact");
        }
        csv << std::setprecision(12)
            << "time_s,waypoint,command_success_rate";
        for (std::size_t joint = 0u; joint < 7u; ++joint) {
            csv << ",q" << joint + 1u << ",commanded_q" << joint + 1u;
        }
        csv << '\n';
        double minimumSuccessRate = 1.0;
        double maximumJointError = 0.0;
        for (const Sample& sample : samples) {
            minimumSuccessRate = std::min(
                minimumSuccessRate,
                sample.commandSuccessRate
            );
            csv << sample.time << ','
                << metalrobo::franka_explore::kOfficialWaypoints[
                       sample.waypoint
                   ].label
                << ',' << sample.commandSuccessRate;
            for (std::size_t joint = 0u; joint < 7u; ++joint) {
                maximumJointError = std::max(
                    maximumJointError,
                    std::abs(sample.q[joint] - sample.commanded[joint])
                );
                csv << ',' << sample.q[joint] << ',' << sample.commanded[joint];
            }
            csv << '\n';
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "robot=franka"
                  << " backend=libfranka"
                  << " waypoints="
                  << metalrobo::franka_explore::kOfficialWaypoints.size()
                  << " samples=" << samples.size()
                  << " min_command_success_rate=" << minimumSuccessRate
                  << " max_joint_tracking_error_rad=" << maximumJointError
                  << " artifact=\"" << options.output << "\"\n";
        return 0;
    } catch (const franka::Exception& error) {
        std::cerr << "numi_franka_libfranka_explore: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "numi_franka_libfranka_explore: " << error.what() << '\n';
        return 1;
    }
}
