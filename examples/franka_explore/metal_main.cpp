#include "trajectory.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using metalrobo::franka_explore::kArmDofCount;
using metalrobo::franka_explore::kOfficialWaypoints;

constexpr std::size_t kHandBodyIndex = 8u;

struct Options {
    double controlTimestep = 0.02;
    std::uint32_t physicsSubsteps = 4u;
    std::filesystem::path output = "franka-exploration.csv";
    std::string metallib;
};

double parsePositiveDouble(const char* value, const char* option) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (value == end || (end != nullptr && *end != '\0') ||
        !(parsed > 0.0) || !std::isfinite(parsed)) {
        throw std::runtime_error(std::string{"invalid value for "} + option);
    }
    return parsed;
}

std::uint32_t parsePositiveUnsigned(const char* value, const char* option) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value == end || (end != nullptr && *end != '\0') || parsed == 0u ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string{"invalid value for "} + option);
    }
    return static_cast<std::uint32_t>(parsed);
}

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
        if (argument == "--dt") {
            result.controlTimestep = parsePositiveDouble(value(), "--dt");
        } else if (argument == "--physics-substeps") {
            result.physicsSubsteps = parsePositiveUnsigned(
                value(),
                "--physics-substeps"
            );
        } else if (argument == "--output") {
            result.output = value();
        } else if (argument == "--metallib") {
            result.metallib = value();
        } else if (argument == "--help") {
            std::cout
                << "usage: metalrobo_franka_explore [--dt SECONDS] "
                   "[--physics-substeps N] [--output PATH] "
                   "[--metallib PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error(
                std::string{"unknown argument: "} + std::string{argument}
            );
        }
    }
    if (result.controlTimestep > 0.05) {
        throw std::runtime_error("--dt must be no greater than 0.05 seconds");
    }
    return result;
}

struct AuthoredTrajectory {
    std::vector<float> targets;
    std::vector<std::size_t> waypointEndSteps;
    std::size_t transitionSteps = 0u;
    std::size_t dwellSteps = 0u;
};

AuthoredTrajectory makeTrajectory(
    const metalrobo::EngineModel& model,
    const double timestep
) {
    if (model.world.nq != model.world.nv ||
        model.defaultQ.size() != model.world.nq ||
        model.world.nq < kArmDofCount) {
        throw std::runtime_error("Franka exploration model layout is invalid");
    }
    AuthoredTrajectory result;
    result.transitionSteps = static_cast<std::size_t>(std::llround(
        metalrobo::franka_explore::kTransitionSeconds / timestep
    ));
    result.dwellSteps = static_cast<std::size_t>(std::llround(
        metalrobo::franka_explore::kDwellSeconds / timestep
    ));
    const std::size_t stepsPerWaypoint =
        result.transitionSteps + result.dwellSteps;
    result.targets.reserve(
        kOfficialWaypoints.size() * stepsPerWaypoint * model.world.nv
    );
    result.waypointEndSteps.reserve(kOfficialWaypoints.size());

    std::array<double, kArmDofCount> start{};
    std::copy_n(model.defaultQ.begin(), kArmDofCount, start.begin());
    for (const auto& waypoint : kOfficialWaypoints) {
        for (std::size_t step = 0u;
             step < result.transitionSteps;
             ++step) {
            const auto q = metalrobo::franka_explore::interpolate(
                start,
                waypoint.q,
                static_cast<double>(step + 1u) /
                    static_cast<double>(result.transitionSteps)
            );
            for (std::size_t coordinate = 0u;
                 coordinate < model.world.nv;
                 ++coordinate) {
                result.targets.push_back(
                    coordinate < kArmDofCount
                    ? static_cast<float>(q[coordinate])
                    : model.defaultQ[coordinate]
                );
            }
        }
        for (std::size_t step = 0u; step < result.dwellSteps; ++step) {
            for (std::size_t coordinate = 0u;
                 coordinate < model.world.nv;
                 ++coordinate) {
                result.targets.push_back(
                    coordinate < kArmDofCount
                    ? static_cast<float>(waypoint.q[coordinate])
                    : model.defaultQ[coordinate]
                );
            }
        }
        result.waypointEndSteps.push_back(
            result.targets.size() / model.world.nv - 1u
        );
        start = waypoint.q;
    }
    return result;
}

bool identicalStatuses(
    const std::vector<MRMetalWorldStatusGPU>& left,
    const std::vector<MRMetalWorldStatusGPU>& right
) {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(
            left.data(),
            right.data(),
            left.size() * sizeof(MRMetalWorldStatusGPU)
        ) == 0);
}

std::array<double, 3u> handPosition(
    const metalrobo::EngineModel& model,
    const std::span<const float> q,
    const std::span<const float> v
) {
    std::vector<double> q64(q.begin(), q.end());
    std::vector<double> v64(v.begin(), v.end());
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(
        model.bodies.size()
    );
    const auto diagnostics = metalrobo::computeArticulatedBodyKinematics(
        model,
        0u,
        q64,
        v64,
        bodies
    );
    if (!diagnostics.succeeded() || bodies.size() <= kHandBodyIndex) {
        throw std::runtime_error("FP64 Franka hand kinematics failed");
    }
    return bodies[kHandBodyIndex].centerOfMassPosition;
}

double distance(
    const std::array<double, 3u>& left,
    const std::array<double, 3u>& right
) {
    return std::hypot(
        std::hypot(left[0] - right[0], left[1] - right[1]),
        left[2] - right[2]
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const metalrobo::EngineModel model =
            metalrobo::makeFrankaPandaHandEngineModel();
        metalrobo::CompiledWorld compiled;
        const auto compilation = metalrobo::compileMetalWorld(
            model,
            0u,
            compiled
        );
        if (!compilation.succeeded()) {
            throw std::runtime_error(
                "could not compile Franka world: " + compilation.message
            );
        }
        const AuthoredTrajectory trajectory = makeTrajectory(
            model,
            options.controlTimestep
        );
        const std::size_t controlSteps =
            trajectory.targets.size() / compiled.nv();
        const metalrobo::MetalWorldBatch batch{
            .environmentCount = 1u,
            .controlStepCount = controlSteps,
            .initialQ = model.defaultQ,
            .initialV = model.defaultV,
            .efforts = trajectory.targets,
        };
        metalrobo::MetalWorldStepConfig config;
        config.timestepSeconds = static_cast<float>(options.controlTimestep);
        config.physicsSubsteps = options.physicsSubsteps;
        config.solverMode = metalrobo::MetalWorldSolverMode::temporalCone;
        config.actuationMode =
            metalrobo::MetalWorldActuationMode::implicitPositionDrive;
        config.velocityIterations = 16u;
        config.finalVelocityIterations = 8u;
        config.deterministic = true;
        config.publishFinalState = true;
        config.publishStateTrajectory = true;

        metalrobo::MetalWorldConfig worldConfig;
        worldConfig.metallibPath = options.metallib;
        metalrobo::MetalWorldContext context{worldConfig};
        metalrobo::MetalWorldResult first;
        const auto firstRun = context.run(compiled, batch, config, first);
        if (!firstRun.succeeded() || firstRun.failedStepCount != 0u) {
            throw std::runtime_error(
                "Franka Metal exploration failed: " + firstRun.message
            );
        }
        metalrobo::MetalWorldResult replay;
        const auto replayRun = context.run(compiled, batch, config, replay);
        if (!replayRun.succeeded() || replayRun.failedStepCount != 0u ||
            first.observations != replay.observations ||
            first.finalQ != replay.finalQ ||
            first.finalV != replay.finalV ||
            !identicalStatuses(first.statuses, replay.statuses)) {
            throw std::runtime_error("Franka deterministic replay diverged");
        }

        const std::size_t frameWidth = compiled.nq() + compiled.nv();
        if (first.observations.size() != controlSteps * frameWidth) {
            throw std::runtime_error("Franka state trajectory is incomplete");
        }
        std::ofstream csv{options.output};
        if (!csv) {
            throw std::runtime_error("could not open trajectory artifact");
        }
        csv << std::setprecision(9)
            << "step,time_s,waypoint,target_q1,actual_q1,hand_x_m,"
               "hand_y_m,hand_z_m\n";
        std::vector<std::array<double, 3u>> waypointPositions;
        waypointPositions.reserve(kOfficialWaypoints.size());
        double maximumJointError = 0.0;
        double minimumHeight = std::numeric_limits<double>::infinity();
        double maximumHeight = -std::numeric_limits<double>::infinity();
        std::size_t waypoint = 0u;
        for (std::size_t step = 0u; step < controlSteps; ++step) {
            while (waypoint + 1u < trajectory.waypointEndSteps.size() &&
                   step > trajectory.waypointEndSteps[waypoint]) {
                ++waypoint;
            }
            const std::span<const float> state{
                first.observations.data() + step * frameWidth,
                frameWidth
            };
            const std::span<const float> actualQ =
                state.first(compiled.nq());
            const std::span<const float> actualV =
                state.subspan(compiled.nq(), compiled.nv());
            const auto position = handPosition(model, actualQ, actualV);
            minimumHeight = std::min(minimumHeight, position[2]);
            maximumHeight = std::max(maximumHeight, position[2]);
            const std::size_t targetOffset = step * compiled.nv();
            csv << step << ',' << step * options.controlTimestep << ','
                << kOfficialWaypoints[waypoint].label << ','
                << trajectory.targets[targetOffset] << ',' << actualQ[0u]
                << ',' << position[0] << ',' << position[1] << ','
                << position[2] << '\n';
            if (step == trajectory.waypointEndSteps[waypoint]) {
                for (std::size_t joint = 0u;
                     joint < kArmDofCount;
                     ++joint) {
                    maximumJointError = std::max(
                        maximumJointError,
                        std::abs(
                            static_cast<double>(actualQ[joint]) -
                            kOfficialWaypoints[waypoint].q[joint]
                        )
                    );
                }
                waypointPositions.push_back(position);
            }
        }
        csv.close();
        if (waypointPositions.size() != kOfficialWaypoints.size()) {
            throw std::runtime_error("not every Franka waypoint was measured");
        }
        const double forwardRight = distance(
            waypointPositions[1u],
            waypointPositions[2u]
        );
        const double forwardLeft = distance(
            waypointPositions[1u],
            waypointPositions[3u]
        );
        const double rightLeft = distance(
            waypointPositions[2u],
            waypointPositions[3u]
        );
        const double returnError = distance(
            waypointPositions[0u],
            waypointPositions[4u]
        );
        if (std::min({forwardRight, forwardLeft, rightLeft}) < 0.05 ||
            maximumJointError > 0.08 || returnError > 0.02) {
            throw std::runtime_error(
                "Franka did not physically realize the exploration targets"
            );
        }

        const auto stats = context.stats();
        std::cout << std::fixed << std::setprecision(6)
                  << "robot=franka_fer_hand"
                  << " backend=metal"
                  << " device=\"" << firstRun.deviceName << "\""
                  << " source=libfranka@85912fe"
                  << " waypoints=" << waypointPositions.size()
                  << " control_steps=" << controlSteps
                  << " physics_steps="
                  << controlSteps * options.physicsSubsteps
                  << " failed_steps=" << firstRun.failedStepCount
                  << " max_waypoint_joint_error_rad=" << maximumJointError
                  << " forward_right_distance_m=" << forwardRight
                  << " forward_left_distance_m=" << forwardLeft
                  << " right_left_distance_m=" << rightLeft
                  << " return_error_m=" << returnError
                  << " height_range_m=" << minimumHeight << ':'
                  << maximumHeight
                  << " replay=bit_identical"
                  << " gpu_ms=" << firstRun.gpuElapsedMilliseconds
                  << " retained_bytes=" << stats.retainedBufferBytes
                  << " artifact=\"" << options.output.string() << "\"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_franka_explore: " << error.what() << '\n';
        return 1;
    }
}
