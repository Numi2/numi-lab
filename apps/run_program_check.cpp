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

bool contains(const std::vector<std::string>& values, const std::string_view value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool containsRole(
    const std::vector<metalrobo::RobotSemanticRole>& roles,
    const std::string_view id
) {
    return std::any_of(
        roles.begin(),
        roles.end(),
        [id](const metalrobo::RobotSemanticRole& role) {
            return role.id == id;
        }
    );
}

}

int main() {
    try {
        auto robot = metalrobo::builtinRobotPack("unitree_g1");
        require(robot.has_value(), "bundled G1 RobotPack is missing");
        require(
            !contains(robot->capabilities, "manipulation") &&
                contains(robot->capabilities, "upper_body_motion") &&
                containsRole(robot->roles, "left_wrist") &&
                containsRole(robot->roles, "right_wrist") &&
                !containsRole(robot->roles, "left_hand") &&
                !containsRole(robot->roles, "right_hand"),
            "bundled 29-DoF G1 claims hand mechanics or manipulation it does not own"
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
        manifest.task = metalrobo::makeUnitreeG1LocomotionTaskPack(
            metalrobo::LocomotionSurface::ground,
            manifest.sensors.observation,
            manifest.reality.reset);
        metalrobo::SensorSpec imu;
        imu.id = "pelvis_state";
        imu.kind = MR_WORLD_SENSOR_STATE;
        imu.nominalRateHz = 50.0f;
        manifest.sensors.mounted.push_back({imu, "pelvis"});
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
        manifest.teacher.id = "no_teacher";

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
                    manifest.reality.reset.operators.size() + 1u &&
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
        duplicatedOwnership.sensors.observation.actorFrame.clear();
        metalrobo::CompiledRun duplicateOutput;
        const auto duplicateStatus = metalrobo::compileRun(
            duplicatedOwnership,
            duplicateOutput
        );
        require(
            duplicateStatus.status ==
                metalrobo::RunCompileStatus::invalidManifest,
            "missing SensorPack execution ownership was accepted"
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
        const auto px4Robot = metalrobo::builtinRobotPack("px4_x500");
        const auto psmRobot = metalrobo::builtinRobotPack("dvrk_psm");
        require(
            ids.size() == 4u &&
                metalrobo::builtinRobotPack("franka_panda").has_value() &&
                psmRobot.has_value() &&
                px4Robot.has_value() &&
                !contains(px4Robot->capabilities, "aerial_manipulation"),
            "robot catalog is incomplete"
        );
        require(
            psmRobot->roles.size() == 3u &&
                psmRobot->roles[0u].id == "whole_body" &&
                psmRobot->roles[0u].members ==
                    psmRobot->mechanics.bodyNames &&
                psmRobot->roles[1u].id == "all_joints" &&
                psmRobot->roles[1u].members ==
                    psmRobot->mechanics.jointNames &&
                psmRobot->roles[2u].id == "all_dofs" &&
                psmRobot->roles[2u].members ==
                    psmRobot->mechanics.dofNames &&
                psmRobot->actuators.size() ==
                    psmRobot->mechanics.jointNames.size(),
            "dVRK PSM built-in pack has unresolved semantic identities"
        );

        metalrobo::RunManifest franka;
        franka.id = "franka_pick_place_compiled_run_check";
        franka.robot = *metalrobo::builtinRobotPack("franka_panda");
        franka.scene = metalrobo::makeFrankaPickPlaceScenePack();
        franka.sensors.id = "franka_default_sensors";
        franka.task = metalrobo::makeFrankaPickPlaceTaskPack(
            franka.sensors.observation,
            franka.reality.reset
        );
        franka.reality.id = "nominal_reality";
        franka.profile.id = "franka_check_profile";
        franka.profile.environmentCount = 8u;
        franka.profile.controlSteps = 32u;
        franka.profile.physicsSubsteps = 4u;
        franka.profile.controlTimestepSeconds = 1.0f / 60.0f;
        franka.teacher.id = "no_teacher";
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

        metalrobo::RunManifest px4;
        px4.id = "px4_x500_compiled_run_check";
        px4.robot = *metalrobo::builtinRobotPack("px4_x500");
        px4.scene = metalrobo::makePX4X500HoverScenePack();
        px4.sensors.id = "px4_x500_state_sensors";
        px4.task = metalrobo::makePX4X500HoverTaskPack(
            px4.sensors.observation, px4.reality.reset);
        px4.reality.id = "px4_x500_nominal_reality";
        px4.teacher.id = "no_teacher";
        px4.profile.id = "px4_x500_check_profile";
        px4.profile.environmentCount = 8u;
        px4.profile.controlSteps = 64u;
        px4.profile.physicsSubsteps = 4u;
        px4.profile.controlTimestepSeconds = 1.0f / 60.0f;
        metalrobo::CompiledRun compiledPX4;
        const auto px4Status = metalrobo::compileRun(px4, compiledPX4);
        require(
            px4Status.succeeded(),
            "PX4 CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(
                    px4Status.status)) + "] " +
                px4Status.element + ": " + px4Status.message
        );
        require(
            compiledPX4.valid() &&
                compiledPX4.multicopterProgram() != nullptr &&
                compiledPX4.model().articulations.size() == 1u &&
                compiledPX4.defaultSceneBodies().size() == 1u &&
                compiledPX4.task().actionBindings().size() == 4u &&
                std::all_of(
                    compiledPX4.task().actionBindings().begin(),
                    compiledPX4.task().actionBindings().end(),
                    [](const MRTaskActionBindingGPU& binding) {
                        return binding.actuator.x ==
                            MR_TASK_ACTUATOR_ROTOR_MIXER;
                    }),
            "PX4 CompiledRun lost its rotor, scene, or action program"
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
            << " px4_run=" << compiledPX4.fingerprint()
            << " px4_task=" << compiledPX4.task().fingerprint()
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "run_program_check=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
