#include "metalrobo/c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (value == end || (end != nullptr && *end != '\0')) {
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
            result.environments = static_cast<std::uint32_t>(
                parseUnsigned(requireValue(), "--envs")
            );
        } else if (argument == "--steps") {
            result.steps = static_cast<std::uint32_t>(
                parseUnsigned(requireValue(), "--steps")
            );
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
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        MRRuntimeHandle* runtime = mr_create_franka(
            options.environments,
            options.seed,
            options.metallib.empty() ? nullptr : options.metallib.c_str()
        );
        if (runtime == nullptr) {
            throw std::runtime_error(mr_last_error());
        }

        const std::uint32_t actionCount = mr_action_count(runtime);
        std::vector<float> actions(
            static_cast<std::size_t>(options.environments) * actionCount,
            0.0f
        );

        auto fillActions = [&](const std::uint32_t step) {
            for (std::uint32_t environment = 0;
                 environment < options.environments;
                 ++environment) {
                for (std::uint32_t action = 0; action < actionCount; ++action) {
                    const float phase =
                        static_cast<float>(step) * 0.013f +
                        static_cast<float>(environment) * 0.00031f +
                        static_cast<float>(action) * 0.41f;
                    actions[
                        static_cast<std::size_t>(environment) * actionCount +
                        action
                    ] = 0.35f * std::sin(phase);
                }
            }
        };

        constexpr std::uint32_t warmupSteps = 20;
        for (std::uint32_t step = 0; step < warmupSteps; ++step) {
            fillActions(step);
            if (mr_step(runtime, actions.data(), actions.size()) != 0) {
                const std::string error = mr_last_error();
                mr_destroy(runtime);
                throw std::runtime_error(error);
            }
        }

        const auto started = std::chrono::steady_clock::now();
        std::uint64_t terminalCount = 0;
        double rewardAccumulator = 0.0;
        for (std::uint32_t step = 0; step < options.steps; ++step) {
            fillActions(step + warmupSteps);
            if (mr_step(runtime, actions.data(), actions.size()) != 0) {
                const std::string error = mr_last_error();
                mr_destroy(runtime);
                throw std::runtime_error(error);
            }
            const float* rewards = mr_rewards(runtime);
            const std::uint8_t* terminated = mr_terminated(runtime);
            for (std::uint32_t environment = 0;
                 environment < options.environments;
                 ++environment) {
                rewardAccumulator += rewards[environment];
                terminalCount += terminated[environment] != 0;
            }
        }
        const auto finished = std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(finished - started).count();
        const double environmentSteps =
            static_cast<double>(options.environments) * options.steps;
        const MRRuntimeStatsC stats = mr_stats(runtime);

        std::cout << std::fixed << std::setprecision(3)
                  << "device=\"" << mr_device_name(runtime) << "\" "
                  << "envs=" << options.environments << ' '
                  << "steps=" << options.steps << ' '
                  << "wall_s=" << seconds << ' '
                  << "env_steps_per_s=" << environmentSteps / seconds << ' '
                  << "mean_reward=" << rewardAccumulator / environmentSteps
                  << ' '
                  << "terminal_events=" << terminalCount << ' '
                  << "last_gpu_ms=" << stats.last_gpu_milliseconds << ' '
                  << "physics_steps=" << stats.physics_steps << '\n';

        mr_destroy(runtime);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_bench: " << error.what() << '\n';
        return 1;
    }
}
