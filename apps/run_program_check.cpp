#include "metalrobo/RunProgram.hpp"

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
        manifest.sensors.id = "g1_default_sensors";
        metalrobo::SensorSpec imu;
        imu.id = "pelvis_state";
        imu.kind = MR_WORLD_SENSOR_STATE;
        imu.nominalRateHz = 50.0f;
        manifest.sensors.mounted.push_back({imu, "pelvis"});
        manifest.task = authored.task;
        manifest.task.terrain = {};
        const auto removeTerrainObservation = [](auto& operators) {
            std::erase_if(operators, [](const auto& spec) {
                return spec.source ==
                    metalrobo::TaskObservationSource::terrainHeight;
            });
        };
        removeTerrainObservation(manifest.task.actorFrame);
        removeTerrainObservation(manifest.task.actorCurrent);
        removeTerrainObservation(manifest.task.critic);
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
        std::cout
            << "run_program_check=ok"
            << " run=" << compiled.fingerprint()
            << " robot=" << compiled.robotFingerprint()
            << " sensors=" << compiled.sensorFingerprint()
            << " world=" << compiled.world().fingerprint()
            << " task=" << compiled.task().fingerprint()
            << " robots=" << ids.size()
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "run_program_check=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
