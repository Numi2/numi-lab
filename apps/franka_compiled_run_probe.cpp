#include "metalrobo/c_api.h"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/Franka.hpp"

#include <algorithm>
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
        std::unique_ptr<MRTaskRolloutHandle, decltype(&mr_task_rollout_destroy)>
            compiled{
                mr_create_franka_pick_place_task_rollout(&config, metallib),
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
            mr_task_rollout_outcome_count(compiled.get()) == 5u,
            "Franka task leaked locomotion-shaped public outcomes"
        );
        std::unique_ptr<MRRuntimeHandle, decltype(&mr_destroy)> legacy{
            mr_create_franka(1u, config.seed, metallib), &mr_destroy
        };
        require(legacy != nullptr,
            "legacy Franka creation failed: " + std::string{mr_last_error()});
        std::vector<float> legacyActions(7u, 0.0f);
        for (std::uint32_t step = 0u; step < steps; ++step) {
            require(mr_step(legacy.get(), legacyActions.data(), 7u) == 0,
                "legacy Franka advance failed");
        }
        const float* legacyPositions = mr_body_positions(legacy.get());
        const float* legacyObservations = mr_observations(legacy.get());
        require(legacyPositions != nullptr,
            "legacy Franka did not publish body poses");
        require(legacyObservations != nullptr,
            "legacy Franka did not publish observations");
        const metalrobo::EngineModel robot =
            metalrobo::makeFrankaPandaHandEngineModel();
        std::vector<double> q64(q, q + layout.nq);
        std::vector<double> v64(layout.nv, 0.0);
        std::vector<metalrobo::ArticulatedBodyKinematics> kinematics(
            robot.bodies.size()
        );
        const auto kinematicStatus =
            metalrobo::computeArticulatedBodyKinematics(
                robot, 0u, q64, v64, kinematics
            );
        require(kinematicStatus.succeeded(),
            "Franka final-state kinematics failed");
        double maximumLegacyPositionError = 0.0;
        double maximumLegacyJointError = 0.0;
        for (std::uint32_t joint = 0u; joint < 7u; ++joint) {
            maximumLegacyJointError = std::max(
                maximumLegacyJointError,
                std::abs(q[joint] - legacyObservations[joint])
            );
        }
        for (std::uint32_t body = 0u; body < 8u; ++body) {
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                maximumLegacyPositionError = std::max(
                    maximumLegacyPositionError,
                    std::abs(
                        kinematics[body].centerOfMassPosition[axis] -
                        legacyPositions[body * 4u + axis]
                    )
                );
            }
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
            << " legacy_joint_error=" << maximumLegacyJointError
            << " legacy_position_error=" << maximumLegacyPositionError
            << " legacy_equivalent=no"
            << " legacy_executor_retained=yes"
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
