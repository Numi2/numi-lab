#include "metalrobo/RunProgram.hpp"
#include "metalrobo/FrankaWorld.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    try {
        auto robot = metalrobo::builtinRobotPack("unitree_g1");
        require(robot.has_value(), "bundled G1 RobotPack is missing");
        metalrobo::LocomotionWorld authored =
            metalrobo::makeUnitreeG1LocomotionWorld(
                metalrobo::LocomotionSurface::ground
            );
        metalrobo::RunManifest manifest;
        manifest.id = "run_program_check";
        manifest.robot = *robot;
        manifest.scene.id = "flat_ground_scene";
        const metalrobo::LocomotionSceneComponent surface =
            metalrobo::makeLocomotionSurfaceComponent(
                manifest.robot.mechanics,
                metalrobo::LocomotionSurface::ground
            );
        manifest.scene.objects.push_back({
            .id = "locomotion_ground",
            .semanticClass = "support_surface",
            .role = MR_WORLD_ASSET_FIXTURE,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_STATIC,
            .mechanics = surface.mechanics,
            .defaultBodyStates = surface.defaultBodyStates,
        });
        manifest.sensors.id = "g1_default_sensors";
        metalrobo::SensorSpec imu;
        imu.id = "pelvis_state";
        imu.kind = MR_WORLD_SENSOR_STATE;
        imu.nominalRateHz = 50.0f;
        manifest.sensors.mounted.push_back({imu, "pelvis"});
        manifest.task = authored.task;
        manifest.reality.id = "nominal_reality";
        manifest.profile.id = "check_profile";
        manifest.profile.environmentCount = 32u;
        manifest.profile.controlSteps = 104u;
        manifest.profile.physicsSubsteps = 4u;
        manifest.profile.controlTimestepSeconds = 0.02f;

        metalrobo::CompiledRun compiled;
        const auto status = metalrobo::compileRun(manifest, compiled);
        require(
            status.succeeded(),
            "CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(status.status)) +
                "] " + status.element + ": " + status.message
        );
        require(
            compiled.valid() && compiled.robotFingerprint() != 0u &&
                compiled.sensorFingerprint() != 0u &&
                compiled.world().sceneBodyCount() == 1u &&
                compiled.defaultSceneBodies().size() == 1u &&
                compiled.worldFamily().worldTemplate.sensors.size() == 1u &&
                compiled.task().fingerprint() != 0u,
            "CompiledRun did not retain modular package identities"
        );

        metalrobo::RunManifest invalid = manifest;
        invalid.sensors.mounted.front().mountRole = "missing_mount";
        const std::uint64_t preserved = compiled.fingerprint();
        const auto rejected = metalrobo::compileRun(invalid, compiled);
        require(
            rejected.status == metalrobo::RunCompileStatus::unresolvedRole &&
                compiled.fingerprint() == preserved,
            "failed package compilation was not transactionally rejected"
        );

        const auto ids = metalrobo::builtinRobotIds();
        require(
            ids.size() == 3u &&
                metalrobo::builtinRobotPack("franka_panda").has_value() &&
                metalrobo::builtinRobotPack("dvrk_psm").has_value(),
            "robot catalog is incomplete"
        );

        metalrobo::RunManifest franka;
        franka.id = "franka_pick_place_compiled_run_check";
        franka.robot = *metalrobo::builtinRobotPack("franka_panda");
        franka.scene = metalrobo::makeFrankaPickPlaceScenePack();
        franka.sensors.id = "franka_default_sensors";
        franka.task = metalrobo::makeFrankaPickPlaceTaskPack();
        franka.reality.id = "nominal_reality";
        franka.profile.id = "franka_check_profile";
        franka.profile.environmentCount = 8u;
        franka.profile.controlSteps = 32u;
        franka.profile.physicsSubsteps = 4u;
        franka.profile.controlTimestepSeconds = 1.0f / 60.0f;
        metalrobo::CompiledRun compiledFranka;
        const auto frankaStatus =
            metalrobo::compileRun(franka, compiledFranka);
        require(
            frankaStatus.succeeded(),
            "Franka CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(
                    frankaStatus.status)) + "] " +
                frankaStatus.element + ": " + frankaStatus.message
        );
        require(
            compiledFranka.valid() &&
                compiledFranka.model().articulations.size() == 1u &&
                compiledFranka.model().bodies.size() == 15u &&
                compiledFranka.defaultSceneBodies().size() == 4u &&
                compiledFranka.task().actionBindings().size() == 9u,
            "Franka CompiledRun lost robot, scene, reset, or action topology"
        );
        std::cout
            << "run_program_check=ok"
            << " run=" << compiled.fingerprint()
            << " robot=" << compiled.robotFingerprint()
            << " sensors=" << compiled.sensorFingerprint()
            << " world=" << compiled.world().fingerprint()
            << " task=" << compiled.task().fingerprint()
            << " robots=" << ids.size()
            << " franka_run=" << compiledFranka.fingerprint()
            << " franka_task=" << compiledFranka.task().fingerprint()
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "run_program_check=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
