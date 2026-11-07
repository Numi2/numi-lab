#include "metalrobo/c_api.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::uint32_t environments = 1024;
    std::uint32_t steps = 1000;
    std::uint64_t seed = 7;
    std::string metallib;
};

std::uint64_t parseUnsigned(const char* value, const char* option) {
    const std::string_view digits(value);
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(),
            [](const char c) { return c >= '0' && c <= '9'; })) {
        throw std::runtime_error(std::string("Invalid value for ") + option);
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (value == end || errno == ERANGE || (end != nullptr && *end != '\0')) {
        throw std::runtime_error(std::string("Invalid value for ") + option);
    }
    return parsed;
}

Options parseOptions(const int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto requireValue = [&]() -> const char* {
            if (++index >= argc) {
                throw std::runtime_error(
                    std::string("Missing value after ") + std::string(argument)
                );
            }
            return argv[index];
        };

        if (argument == "--envs") {
            const auto value = parseUnsigned(requireValue(), "--envs");
            if (value > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("--envs exceeds uint32 capacity");
            result.environments = static_cast<std::uint32_t>(value);
        } else if (argument == "--steps") {
            const auto value = parseUnsigned(requireValue(), "--steps");
            if (value > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("--steps exceeds uint32 capacity");
            result.steps = static_cast<std::uint32_t>(value);
        } else if (argument == "--seed") {
            result.seed = parseUnsigned(requireValue(), "--seed");
        } else if (argument == "--metallib") {
            result.metallib = requireValue();
        } else if (argument == "--help") {
            std::cout
                << "usage: metalrobo_bench [--envs N] [--steps N] "
                   "[--seed N] [--metallib PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error(
                std::string("Unknown argument: ") + std::string(argument)
            );
        }
    }
    if (result.environments == 0 || result.steps == 0) {
        throw std::runtime_error("--envs and --steps must be positive.");
    }
    const auto requested = std::uint64_t(result.environments) * result.steps;
    if (requested > std::numeric_limits<std::uint32_t>::max() ||
        std::uint64_t(result.environments) * 20u > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("environment-step count exceeds native accounting capacity");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        MRTaskRolloutConfigC profile{};
        profile.environment_count = options.environments;
        profile.physics_substeps = 4u;
        profile.velocity_iterations = 4u;
        profile.final_velocity_iterations = 2u;
        profile.control_timestep_seconds = 1.0f / 60.0f;
        profile.seed = options.seed;
        const MRRunManifestC manifest{
            .profile = profile,
            .source = MR_RUN_SOURCE_FRANKA_PICK_PLACE,
            .metallib_path = options.metallib.empty()
                ? nullptr
                : options.metallib.c_str(),
        };
        const auto initializationStarted = std::chrono::steady_clock::now();
        MRTaskRolloutHandle* runtime = mr_create_task_rollout(&manifest);
        const auto initializationFinished = std::chrono::steady_clock::now();
        if (runtime == nullptr) {
            throw std::runtime_error(mr_last_error());
        }

        const MRTaskRolloutLayoutC layout = mr_task_rollout_layout(runtime);
        const std::uint32_t actionCount = layout.action_count;
        std::vector<float> actions(
            static_cast<std::size_t>(options.steps) * options.environments *
                actionCount,
            0.0f
        );

        auto fillActions = [&](const std::uint32_t outputStep,
                               const std::uint32_t phaseStep) {
            for (std::uint32_t environment = 0;
                 environment < options.environments;
                 ++environment) {
                for (std::uint32_t action = 0; action < actionCount; ++action) {
                    const float phase =
                        static_cast<float>(phaseStep) * 0.013f +
                        static_cast<float>(environment) * 0.00031f +
                        static_cast<float>(action) * 0.41f;
                    actions[
                        (static_cast<std::size_t>(outputStep) *
                            options.environments + environment) *
                            actionCount +
                        action
                    ] = 0.35f * std::sin(phase);
                }
            }
        };

        constexpr std::uint32_t warmupSteps = 20;
        std::vector<float> warmup(
            static_cast<std::size_t>(warmupSteps) * options.environments *
                actionCount
        );
        std::swap(actions, warmup);
        for (std::uint32_t step = 0; step < warmupSteps; ++step) {
            fillActions(step, step);
        }
        MRTaskRolloutAdvanceC warmupAdvance{};
        const auto warmupStarted = std::chrono::steady_clock::now();
        const int warmupResult = mr_task_rollout_advance(
            runtime, actions.data(), actions.size(), nullptr, 0u,
            warmupSteps, 0u, 0u, &warmupAdvance
        );
        const std::string warmupError = warmupResult != 0 ? mr_last_error() : "";
        const auto warmupFinished = std::chrono::steady_clock::now();
        if (warmupResult != 0 || warmupAdvance.failed_environment_steps != 0u ||
            warmupAdvance.successful_environment_steps !=
                std::uint64_t(warmupSteps) * options.environments) {
            std::cerr << "warmup accepted=" << warmupAdvance.successful_environment_steps
                      << " failed=" << warmupAdvance.failed_environment_steps
                      << " first_gpu_status=" << warmupAdvance.first_gpu_status_code << '\n';
            mr_task_rollout_destroy(runtime);
            throw std::runtime_error(warmupResult != 0 ? warmupError :
                                     "warmup did not complete every environment step");
        }
        std::swap(actions, warmup);
        for (std::uint32_t step = 0; step < options.steps; ++step) {
            fillActions(step, step + warmupSteps);
        }

        const auto started = std::chrono::steady_clock::now();
        MRTaskRolloutAdvanceC advance{};
        const int result = mr_task_rollout_advance(
            runtime, actions.data(), actions.size(), nullptr, 0u,
            options.steps, 1u, 0u, &advance
        );
        const auto finished = std::chrono::steady_clock::now();
        const std::string advanceError = result != 0 ? mr_last_error() : "";
        const double seconds =
            std::chrono::duration<double>(finished - started).count();
        const std::uint64_t requested =
            std::uint64_t(options.environments) * options.steps;
        const std::uint64_t accepted = advance.successful_environment_steps;
        const bool complete = result == 0 &&
            advance.failed_environment_steps == 0u && accepted == requested;
        std::uint64_t terminalCount = 0, transitionPhysicsErrors = 0, nonfiniteRewards = 0;
        double rewardAccumulator = 0.0;
        if (complete) {
            const MRTaskTransitionC* transitions =
                mr_task_rollout_transitions(runtime);
            if (transitions == nullptr) {
                mr_task_rollout_destroy(runtime);
                throw std::runtime_error("completed rollout has no transitions");
            }
            for (std::size_t index = 0u; index < requested; ++index) {
                if (std::isfinite(transitions[index].reward)) rewardAccumulator += transitions[index].reward;
                else ++nonfiniteRewards;
                terminalCount += transitions[index].done != 0u;
                transitionPhysicsErrors += transitions[index].physics_error != 0u;
            }
        }
        const auto finalLayout = mr_task_rollout_layout(runtime);
        const char* deviceName = mr_task_rollout_device_name(runtime);
        std::cout << std::fixed << std::setprecision(6)
                  << "device=\"" << (deviceName != nullptr ? deviceName : "unknown") << "\" "
                  << "envs=" << options.environments << ' '
                  << "steps=" << options.steps << ' '
                  << "seed=" << options.seed << ' '
                  << "run_fingerprint=" << layout.run_fingerprint << ' '
                  << "robot_fingerprint=" << layout.robot_fingerprint << ' '
                  << "initialization_s=" << std::chrono::duration<double>(initializationFinished-initializationStarted).count() << ' '
                  << "warmup_s=" << std::chrono::duration<double>(warmupFinished-warmupStarted).count() << ' '
                  << "wall_s=" << seconds << ' '
                  << "retained_buffer_bytes=" << finalLayout.retained_buffer_bytes << ' '
                  << "peak_aliased_bytes=" << finalLayout.peak_aliased_bytes << ' '
                  << "policy=host_sinusoidal "
                  << "requested_env_steps=" << requested << ' '
                  << "accepted_env_steps=" << accepted << ' '
                  << "failed_env_steps=" << advance.failed_environment_steps << ' '
                  << "first_failing_environment=" << advance.first_failing_environment << ' '
                  << "first_failing_step=" << advance.first_failing_control_step << ' '
                  << "first_gpu_status=" << advance.first_gpu_status_code << ' '
                  << "env_steps_per_s=" << (seconds > 0.0 ? accepted / seconds : 0.0) << ' '
                  << "accepted_simulated_s_per_wall_s="
                  << (seconds > 0.0 ? accepted * double(profile.control_timestep_seconds) / seconds : 0.0) << ' '
                  << "gpu_ms=" << advance.gpu_milliseconds << ' '
                  << "submission_ms=" << advance.submission_milliseconds << ' '
                  << "accepted_physics_steps=" << accepted * profile.physics_substeps << ' '
                  << "completed=" << (complete && transitionPhysicsErrors == 0 && nonfiniteRewards == 0 ? 1 : 0);
        if (complete) {
            std::cout << " terminal_events=" << terminalCount
                      << " transition_physics_errors=" << transitionPhysicsErrors
                      << " nonfinite_rewards=" << nonfiniteRewards;
            if (transitionPhysicsErrors == 0u && nonfiniteRewards == 0u)
                std::cout << " mean_reward=" << rewardAccumulator / double(requested);
        }
        std::cout << '\n';
        if (!complete || transitionPhysicsErrors != 0u || nonfiniteRewards != 0u) {
            if (result != 0) std::cerr << "metalrobo_bench: " << advanceError << '\n';
            else std::cerr << "metalrobo_bench: incomplete or invalid physical rollout\n";
            mr_task_rollout_destroy(runtime);
            return 1;
        }

        mr_task_rollout_destroy(runtime);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_bench: " << error.what() << '\n';
        return 1;
    }
}
