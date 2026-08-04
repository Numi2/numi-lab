#include "metalrobo/c_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        const char* metallib = argc > 1 ? argv[1] : nullptr;
        const MRTaskRolloutConfigC config{
            .environment_count = 1u,
            .physics_substeps = 4u,
            .velocity_iterations = 4u,
            .final_velocity_iterations = 2u,
            .control_timestep_seconds = 1.0f / 60.0f,
            .seed = 0x4652414e4b41ull,
        };
        const MRRunManifestC manifest{
            .profile = config,
            .source = MR_RUN_SOURCE_FRANKA_PICK_PLACE,
            .metallib_path = metallib,
        };
        std::unique_ptr<MRTaskRolloutHandle, decltype(&mr_task_rollout_destroy)>
            compiled{
                mr_create_task_rollout(&manifest),
                &mr_task_rollout_destroy,
            };
        require(compiled != nullptr,
            "Franka CompiledRun creation failed: " +
                std::string{mr_last_error()});
        const MRTaskRolloutLayoutC layout =
            mr_task_rollout_layout(compiled.get());
        require(
            layout.nq == 9u && layout.nv == 9u &&
                layout.action_count == 9u && layout.scene_body_count == 4u &&
                layout.run_fingerprint != 0u &&
                layout.robot_fingerprint != 0u,
            "Franka CompiledRun layout is incomplete"
        );
        require(mr_task_rollout_set_state_readback(compiled.get(), 1u) == 0,
            "Franka state readback could not be enabled");
        constexpr std::uint32_t steps = 64u;
        std::vector<float> actions(steps * layout.action_count, 0.0f);
        MRTaskRolloutAdvanceC advance{};
        require(
            mr_task_rollout_advance(
                compiled.get(), actions.data(), actions.size(), nullptr, 0u,
                steps, 1u, 0u, &advance
            ) == 0,
            "Franka CompiledRun advance failed: " +
                std::string{mr_last_error()}
        );
        require(
            advance.successful_environment_steps == steps &&
                advance.failed_environment_steps == 0u,
            "Franka CompiledRun did not complete every physical step"
        );
        const float* q = mr_task_rollout_final_q(compiled.get());
        require(q != nullptr, "Franka CompiledRun did not publish q");
        float maximumJointDrift = 0.0f;
        for (std::uint32_t index = 0u; index < layout.nq; ++index) {
            require(std::isfinite(q[index]), "Franka q became non-finite");
        }
        const std::vector<float> home{
            0.0f, -0.785398163f, 0.0f, -2.35619449f,
            0.0f, 1.570796327f, 0.785398163f, 0.035f, 0.035f,
        };
        for (std::uint32_t index = 0u; index < layout.nq; ++index) {
            maximumJointDrift = std::max(
                maximumJointDrift, std::abs(q[index] - home[index])
            );
        }
        std::vector<float> scene(layout.scene_body_count * 13u);
        require(
            mr_task_rollout_copy_final_scene_states(
                compiled.get(), scene.data(), scene.size()
            ) == 0,
            "Franka CompiledRun did not publish scene states"
        );
        const float objectHeight = scene[2u];
        const float clutterHeight = scene[3u * 13u + 2u];
        require(
            std::isfinite(objectHeight) && std::isfinite(clutterHeight) &&
                objectHeight >= 0.023f && clutterHeight >= 0.028f,
            "Franka scene penetrated its physical support surface"
        );
        require(
            mr_task_rollout_outcome_count(compiled.get()) == 9u,
            "Franka task did not publish its four typed manipulation outcomes"
        );
        constexpr std::array<const char*, 4u> manipulationOutcomes{
            "grasp_reward", "lift_reward", "object_position_reward",
            "placement_reward",
        };
        const float* outcomeValues =
            mr_task_rollout_outcome_values(compiled.get());
        require(outcomeValues != nullptr,
            "Franka typed outcome values were not published");
        for (std::uint32_t index = 0u;
             index < manipulationOutcomes.size(); ++index) {
            const char* id = mr_task_rollout_outcome_id(
                compiled.get(), 5u + index
            );
            require(
                id != nullptr && id == std::string{manipulationOutcomes[index]} &&
                    std::isfinite(outcomeValues[(steps - 1u) * 9u + 5u + index]),
                "Franka manipulation outcome channel is missing or non-finite"
            );
        }
        std::cout
            << "franka_compiled_run=ok"
            << " run=" << layout.run_fingerprint
            << " robot=" << layout.robot_fingerprint
            << " task=" << layout.task_fingerprint
            << " steps=" << advance.successful_environment_steps
            << " failed=" << advance.failed_environment_steps
            << " contacts=" << advance.maximum_active_contacts
            << " max_joint_drift=" << maximumJointDrift
            << " executor=compiled_run"
            << " object_z=" << objectHeight
            << " clutter_z=" << clutterHeight
            << " outcomes=" << mr_task_rollout_outcome_count(compiled.get())
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "franka_compiled_run=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
