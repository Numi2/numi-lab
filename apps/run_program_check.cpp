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

void assignPrograms(metalrobo::RunManifest& manifest) {
    manifest.sensors.actorFrame =
        std::move(manifest.task.actorFrame);
    manifest.sensors.actorHistoryLength =
        manifest.task.actorHistoryLength;
    manifest.sensors.actorCurrent =
        std::move(manifest.task.actorCurrent);
    manifest.sensors.critic = std::move(manifest.task.critic);
    manifest.sensors.criticHistoryLength =
        manifest.task.criticHistoryLength;
    manifest.sensors.criticIncludesCleanHistory =
        manifest.task.criticIncludesCleanHistory;
    manifest.sensors.visual = std::move(manifest.task.visual);
    manifest.task.actorHistoryLength = 1u;
    manifest.task.criticHistoryLength = 1u;
    manifest.task.criticIncludesCleanHistory = true;
    manifest.reality.taskState =
        std::move(manifest.task.randomization);
    manifest.reality.maximumActionDelaySteps =
        manifest.task.maximumActionDelaySteps;
    manifest.reality.maximumObservationDelaySteps =
        manifest.task.maximumObservationDelaySteps;
    manifest.task.maximumActionDelaySteps = 0u;
    manifest.task.maximumObservationDelaySteps = 0u;
    manifest.teacher.id = "no_teacher";
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
        manifest.reality.program.id = "runtime_reality";
        manifest.reality.program.variations.push_back({
            .id = "robot_gain",
            .axis = MR_WORLD_VARIATION_ROBOT_STATE,
            .distribution = MR_WORLD_DISTRIBUTION_UNIFORM,
            .target = MR_WORLD_TARGET_ROBOT_GAIN_SCALE,
            .targetId = manifest.robot.id,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        });
        manifest.profile.id = "check_profile";
        manifest.profile.environmentCount = 32u;
        manifest.profile.controlSteps = 104u;
        manifest.profile.physicsSubsteps = 4u;
        manifest.profile.controlTimestepSeconds = 0.02f;
        assignPrograms(manifest);

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
                compiled.task().fingerprint() != 0u &&
                compiled.task().outcomes().size() == 4u &&
                compiled.task().randomizationOperators().size() ==
                    manifest.reality.taskState.size() + 1u &&
                compiled.realityFingerprint() != 0u &&
                compiled.teacherFingerprint() != 0u,
            "CompiledRun did not retain modular package identities"
        );

        metalrobo::RunManifest alternateSensor = manifest;
        metalrobo::VisualSensorProgram visual;
        visual.assets.push_back({
            "fixture.visualpack",
            "robot",
            "fixture-content-hash",
            1u,
            1u,
        });
        visual.cameraParentBody = "pelvis";
        visual.width = 16u;
        visual.height = 16u;
        visual.minimumVisiblePixels = 1u;
        visual.nominalRateHz = 50.0f;
        visual.fingerprint =
            metalrobo::visualSensorProgramFingerprint(visual);
        const std::uint64_t visualFingerprint = visual.fingerprint;
        alternateSensor.sensors.deviceVisual = std::move(visual);
        metalrobo::CompiledRun compiledAlternateSensor;
        const auto alternateSensorStatus = metalrobo::compileRun(
            alternateSensor,
            compiledAlternateSensor
        );
        require(
            alternateSensorStatus.succeeded() &&
                compiledAlternateSensor.visualSensorProgram() != nullptr &&
                compiledAlternateSensor.visualSensorProgram()->fingerprint ==
                    visualFingerprint &&
                compiledAlternateSensor.sensorFingerprint() !=
                    compiled.sensorFingerprint() &&
                compiledAlternateSensor.fingerprint() !=
                    compiled.fingerprint(),
            "executable SensorPack program is missing from run identity"
        );

        metalrobo::RunManifest tamperedSensor = alternateSensor;
        tamperedSensor.sensors.deviceVisual->width += 1u;
        metalrobo::CompiledRun tamperedSensorOutput;
        require(
            metalrobo::compileRun(
                tamperedSensor,
                tamperedSensorOutput
            ).status == metalrobo::RunCompileStatus::invalidManifest &&
                !tamperedSensorOutput.valid(),
            "tampered visual SensorProgram fingerprint was accepted"
        );

        metalrobo::RunManifest duplicatedOwnership = manifest;
        duplicatedOwnership.task.actorFrame =
            duplicatedOwnership.sensors.actorFrame;
        metalrobo::CompiledRun duplicateOutput;
        const auto duplicateStatus = metalrobo::compileRun(
            duplicatedOwnership,
            duplicateOutput
        );
        require(
            duplicateStatus.status ==
                metalrobo::RunCompileStatus::invalidManifest,
            "duplicate TaskPack/SensorPack execution ownership was accepted"
        );

        metalrobo::RunManifest unsupportedTeacher = manifest;
        unsupportedTeacher.teacher = {
            .id = "passive_foundation_teacher",
            .kind = metalrobo::TeacherKind::foundationActionChunk,
        };
        const auto teacherStatus = metalrobo::compileRun(
            unsupportedTeacher,
            duplicateOutput
        );
        require(
            teacherStatus.status ==
                metalrobo::RunCompileStatus::invalidManifest,
            "TeacherPack without native execution was accepted"
        );

        metalrobo::RunManifest alternateProfile = manifest;
        alternateProfile.profile.velocityIterations += 1u;
        metalrobo::CompiledRun compiledAlternateProfile;
        const auto alternateProfileStatus = metalrobo::compileRun(
            alternateProfile,
            compiledAlternateProfile
        );
        require(
            alternateProfileStatus.succeeded() &&
                compiledAlternateProfile.fingerprint() !=
                    compiled.fingerprint(),
            "solver-profile semantics are missing from the run fingerprint"
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
        assignPrograms(franka);
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
                compiledFranka.task().actionBindings().size() == 9u &&
                compiledFranka.task().outcomes().size() == 4u,
            "Franka CompiledRun lost robot, scene, reset, or action topology"
        );
        std::cout
            << "run_program_check=ok"
            << " run=" << compiled.fingerprint()
            << " robot=" << compiled.robotFingerprint()
            << " sensors=" << compiled.sensorFingerprint()
            << " reality=" << compiled.realityFingerprint()
            << " teacher=" << compiled.teacherFingerprint()
            << " reality_ops="
            << compiled.task().randomizationOperators().size()
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
