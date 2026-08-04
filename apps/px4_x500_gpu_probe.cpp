#include "metalrobo/c_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t environments = argc > 2
            ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10))
            : 256u;
        const std::uint32_t steps = argc > 3
            ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
            : 104u;
        require(environments > 0u && steps > 0u,
            "environment and step counts must be positive");
        const char* metallib = argc > 1
            ? argv[1]
            : METALROBO_DEFAULT_METALLIB;
        const MRTaskRolloutConfigC profile{
            .environment_count = environments,
            .physics_substeps = 4u,
            .velocity_iterations = 4u,
            .final_velocity_iterations = 2u,
            .control_timestep_seconds = 1.0f / 60.0f,
            .seed = 0x50583458353030ull,
        };
        const MRRunManifestC manifest{
            .profile = profile,
            .source = MR_RUN_SOURCE_PX4_X500,
            .metallib_path = metallib,
        };
        std::unique_ptr<MRTaskRolloutHandle, decltype(&mr_task_rollout_destroy)>
            run{mr_create_task_rollout(&manifest), &mr_task_rollout_destroy};
        require(run != nullptr,
            "PX4 CompiledRun creation failed: " +
                std::string{mr_last_error()});
        const MRTaskRolloutLayoutC layout =
            mr_task_rollout_layout(run.get());
        require(
            layout.nq == 7u && layout.nv == 6u &&
                layout.action_count == 4u &&
                layout.scene_body_count == 1u &&
                layout.run_fingerprint != 0u &&
                layout.robot_fingerprint != 0u,
            "PX4 CompiledRun layout is incomplete");
        require(mr_task_rollout_set_state_readback(run.get(), 1u) == 0,
            "PX4 state readback could not be enabled");

        std::vector<float> actions(
            static_cast<std::size_t>(environments) * steps *
                layout.action_count,
            0.0f);
        MRTaskRolloutAdvanceC advance{};
        require(
            mr_task_rollout_advance(
                run.get(), actions.data(), actions.size(), nullptr, 0u,
                steps, 1u, 0u, &advance) == 0,
            "PX4 CompiledRun advance failed: " +
                std::string{mr_last_error()});
        require(
            advance.successful_environment_steps ==
                    static_cast<std::uint64_t>(environments) * steps &&
                advance.failed_environment_steps == 0u,
            "PX4 CompiledRun rejected a physical step");

        const float* q = mr_task_rollout_final_q(run.get());
        require(q != nullptr, "PX4 CompiledRun did not publish q");
        float minimumHeight = INFINITY;
        float maximumHeight = -INFINITY;
        float maximumTilt = 0.0f;
        float maximumLinearSpeed = 0.0f;
        float maximumAngularSpeed = 0.0f;
        const float* observations =
            mr_task_rollout_actor_observations(run.get());
        require(observations != nullptr,
            "PX4 actor observations were not published");
        for (std::uint32_t environment = 0u;
             environment < environments; ++environment) {
            const float* state = q +
                static_cast<std::size_t>(environment) * layout.nq;
            for (std::uint32_t index = 0u; index < layout.nq; ++index) {
                require(std::isfinite(state[index]),
                    "PX4 state became non-finite");
            }
            minimumHeight = std::min(minimumHeight, state[2]);
            maximumHeight = std::max(maximumHeight, state[2]);
            const float qx = state[3];
            const float qy = state[4];
            const float upZ = std::clamp(
                1.0f - 2.0f * (qx * qx + qy * qy), -1.0f, 1.0f);
            maximumTilt = std::max(maximumTilt, std::acos(upZ));
            const std::size_t observationBase =
                (static_cast<std::size_t>(steps - 1u) * environments +
                 environment) * layout.actor_observation_count;
            const float vx = 2.0f * observations[observationBase + 0u];
            const float wx = 4.0f * observations[observationBase + 1u];
            const float vy = 2.0f * observations[observationBase + 3u];
            const float wy = 4.0f * observations[observationBase + 4u];
            const float vz = 2.0f * observations[observationBase + 6u];
            const float wz = 4.0f * observations[observationBase + 7u];
            maximumLinearSpeed = std::max(maximumLinearSpeed,
                std::sqrt(vx * vx + vy * vy + vz * vz));
            maximumAngularSpeed = std::max(maximumAngularSpeed,
                std::sqrt(wx * wx + wy * wy + wz * wz));
        }
        require(
            minimumHeight > 1.5f && maximumHeight < 2.5f &&
                maximumTilt < 0.1f,
            "neutral PX4 hover drifted outside its physical envelope");
        require(mr_task_rollout_outcome_count(run.get()) == 8u,
            "PX4 typed outcomes were not published");

        std::cout
            << "px4_x500_compiled_run=ok"
            << " device=\"" << mr_task_rollout_device_name(run.get()) << '"'
            << " run=" << layout.run_fingerprint
            << " robot=" << layout.robot_fingerprint
            << " steps=" << advance.successful_environment_steps
            << " failed=" << advance.failed_environment_steps
            << " min_z=" << minimumHeight
            << " max_z=" << maximumHeight
            << " max_tilt=" << maximumTilt
            << " max_linear_speed=" << maximumLinearSpeed
            << " max_angular_speed=" << maximumAngularSpeed
            << " contacts=" << advance.maximum_active_contacts
            << " executor=compiled_run\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "px4_x500_compiled_run=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
