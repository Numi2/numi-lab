#include "metalrobo/c_api.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
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
std::uint64_t integer(const char* text) {
    const std::string_view digits(text);
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(),
        [](char c) { return c >= '0' && c <= '9'; })) {
        throw std::runtime_error("expected an unsigned decimal integer");
    }
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0')
        throw std::runtime_error("integer exceeds uint64 capacity");
    return value;
}
struct Options {
    std::uint32_t steps = 1000;
    std::uint64_t seed = 7;
    std::string metallib;
};
Options options(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout << "usage: metalrobo_latency [--steps N] [--seed N] [--metallib PATH]\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("missing option value");
        if (flag == "--steps") {
            const auto value = integer(argv[i]);
            if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("--steps must fit a positive uint32");
            out.steps = static_cast<std::uint32_t>(value);
        } else if (flag == "--seed") out.seed = integer(argv[i]);
        else if (flag == "--metallib") out.metallib = argv[i];
        else throw std::runtime_error("unknown option");
    }
    return out;
}
double milliseconds(std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double index = p * static_cast<double>(sorted.size() - 1u);
    const auto lower = static_cast<std::size_t>(index);
    const auto upper = std::min(lower + 1u, sorted.size() - 1u);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}
}

int main(int argc, char** argv) {
    MRTaskRolloutHandle* runtime = nullptr;
    try {
        const auto config = options(argc, argv);
        MRTaskRolloutConfigC profile{};
        profile.environment_count = 1;
        profile.physics_substeps = 4;
        profile.velocity_iterations = 4;
        profile.final_velocity_iterations = 2;
        profile.control_timestep_seconds = 1.0f / 60.0f;
        profile.seed = config.seed;
        const MRRunManifestC manifest{
            .profile = profile, .source = MR_RUN_SOURCE_FRANKA_PICK_PLACE,
            .metallib_path = config.metallib.empty() ? nullptr : config.metallib.c_str()
        };
        const auto initStart = std::chrono::steady_clock::now();
        runtime = mr_create_task_rollout(&manifest);
        const auto initEnd = std::chrono::steady_clock::now();
        if (!runtime) throw std::runtime_error(mr_last_error());
        const auto layout = mr_task_rollout_layout(runtime);
        std::vector<float> actions(layout.action_count);
        auto fill = [&](std::uint32_t step) {
            for (std::uint32_t a = 0; a < layout.action_count; ++a)
                actions[a] = 0.35f * std::sin(static_cast<float>(step) * 0.013f +
                                               static_cast<float>(a) * 0.41f);
        };
        constexpr std::uint32_t warmupSteps = 20;
        const auto warmupStart = std::chrono::steady_clock::now();
        for (std::uint32_t step = 0; step < warmupSteps; ++step) {
            fill(step);
            MRTaskRolloutAdvanceC advance{};
            const int rc = mr_task_rollout_advance(runtime, actions.data(), actions.size(),
                                                    nullptr, 0u, 1u, 0u, 0u, &advance);
            if (rc != 0 || advance.successful_environment_steps != 1u ||
                advance.failed_environment_steps != 0u) {
                const std::string error = rc != 0 ? mr_last_error() : "warmup failed";
                throw std::runtime_error(error);
            }
        }
        const auto warmupEnd = std::chrono::steady_clock::now();
        std::vector<double> latencies;
        latencies.reserve(config.steps);
        std::uint64_t accepted = 0;
        double gpuMs = 0.0, submissionMs = 0.0, measuredMs = 0.0;
        std::uint32_t status = 0;
        std::string failure;
        const auto seriesStart = std::chrono::steady_clock::now();
        for (std::uint32_t step = 0; step < config.steps; ++step) {
            fill(step + warmupSteps);
            MRTaskRolloutAdvanceC advance{};
            const auto begin = std::chrono::steady_clock::now();
            const int rc = mr_task_rollout_advance(runtime, actions.data(), actions.size(),
                                                    nullptr, 0u, 1u, 1u, 0u, &advance);
            const auto end = std::chrono::steady_clock::now();
            const std::string error = rc != 0 ? mr_last_error() : "";
            const double elapsed = milliseconds(end - begin);
            latencies.push_back(elapsed);
            measuredMs += elapsed;
            accepted += advance.successful_environment_steps;
            gpuMs += advance.gpu_milliseconds;
            submissionMs += advance.submission_milliseconds;
            status = advance.first_gpu_status_code;
            if (rc != 0 || advance.successful_environment_steps != 1u ||
                advance.failed_environment_steps != 0u) {
                failure = rc != 0 ? error : "incomplete or rejected step";
                break;
            }
            const MRTaskTransitionC* transition = mr_task_rollout_transitions(runtime);
            if (!transition || transition->physics_error != 0u || !std::isfinite(transition->reward)) {
                failure = "invalid published transition";
                break;
            }
        }
        const auto seriesEnd = std::chrono::steady_clock::now();
        const double wallMs = milliseconds(seriesEnd - seriesStart);
        const bool complete = failure.empty() && accepted == config.steps;
        std::sort(latencies.begin(), latencies.end());
        const auto finalLayout = mr_task_rollout_layout(runtime);
        const char* deviceName = mr_task_rollout_device_name(runtime);
        std::cout << std::fixed << std::setprecision(6)
                  << "device=\"" << (deviceName ? deviceName : "unknown") << "\" "
                  << "seed=" << config.seed << " run_fingerprint=" << layout.run_fingerprint
                  << " robot_fingerprint=" << layout.robot_fingerprint
                  << " policy=host_sinusoidal requested_steps=" << config.steps
                  << " attempted_steps=" << latencies.size() << " accepted_env_steps=" << accepted
                  << " first_gpu_status=" << status << " completed=" << (complete ? 1 : 0)
                  << " initialization_ms=" << milliseconds(initEnd - initStart)
                  << " warmup_ms=" << milliseconds(warmupEnd - warmupStart)
                  << " wall_ms=" << wallMs << " call_sum_ms=" << measuredMs << " gpu_ms=" << gpuMs
                  << " submission_ms=" << submissionMs
                  << " latency_p50_ms=" << percentile(latencies, 0.50)
                  << " latency_p95_ms=" << percentile(latencies, 0.95)
                  << " latency_max_ms=" << (latencies.empty() ? 0.0 : latencies.back())
                  << " accepted_simulated_s_per_wall_s="
                  << (wallMs > 0 ? accepted * double(profile.control_timestep_seconds) * 1000.0 / wallMs : 0.0)
                  << " retained_buffer_bytes=" << finalLayout.retained_buffer_bytes
                  << " peak_aliased_bytes=" << finalLayout.peak_aliased_bytes << '\n';
        mr_task_rollout_destroy(runtime);
        runtime = nullptr;
        if (!complete) {
            std::cerr << "metalrobo_latency: " << failure << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        if (runtime) mr_task_rollout_destroy(runtime);
        std::cerr << "metalrobo_latency: " << error.what() << '\n';
        return 1;
    }
}
