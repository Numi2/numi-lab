#include "metalrobo/RunProgram.hpp"
#include "metalrobo/FrankaWorld.hpp"

#include <algorithm>
#include <cmath>
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
        const auto doveRobot = metalrobo::builtinRobotPack(
            "birdflow_deetjen_dove_hybrid"
        );
        const auto crowRobot = metalrobo::builtinRobotPack(
            "birdflow_american_crow_estimated_hybrid"
        );
        require(
            ids.size() == 6u &&
                metalrobo::builtinRobotPack("franka_panda").has_value() &&
                psmRobot.has_value() &&
                px4Robot.has_value() &&
                doveRobot.has_value() &&
                crowRobot.has_value() &&
                contains(doveRobot->capabilities, "articulated_flight") &&
                contains(doveRobot->capabilities, "load_responsive_aero") &&
                contains(crowRobot->capabilities, "standing_to_flight") &&
                contains(crowRobot->capabilities, "estimated_crow_model") &&
                contains(crowRobot->capabilities, "articulated_wing_sweep") &&
                contains(crowRobot->capabilities, "articulated_wing_pronation") &&
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

        metalrobo::RunManifest dove;
        dove.id = "birdflow_deetjen_dove_compiled_run_check";
        dove.robot = *doveRobot;
        dove.scene = metalrobo::makeBirdFlowDoveFlightScenePack();
        dove.sensors.id = "birdflow_deetjen_dove_state_sensors";
        dove.task = metalrobo::makeBirdFlowDoveFlightTaskPack(
            dove.sensors.observation, dove.reality.reset);
        dove.reality.id = "birdflow_deetjen_dove_nominal_reality";
        dove.teacher.id = "no_teacher";
        dove.profile.id = "birdflow_deetjen_dove_check_profile";
        dove.profile.environmentCount = 8u;
        dove.profile.controlSteps = 64u;
        dove.profile.physicsSubsteps = 4u;
        dove.profile.controlTimestepSeconds = 1.0f / 60.0f;
        metalrobo::CompiledRun compiledDove;
        const auto doveStatus = metalrobo::compileRun(dove, compiledDove);
        require(
            doveStatus.succeeded(),
            "BirdFlow dove CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(
                    doveStatus.status)) + "] " + doveStatus.element + ": " +
                doveStatus.message
        );
        require(
            compiledDove.valid() &&
                compiledDove.flappingWingProgram() != nullptr &&
                compiledDove.multicopterProgram() == nullptr &&
                compiledDove.model().bodies.size() == 11u &&
                compiledDove.model().joints.size() == 9u &&
                compiledDove.task().actionBindings().size() == 10u &&
                compiledDove.task().actionBindings()[0u].actuator.x ==
                    MR_TASK_ACTUATOR_FLAPPING_POSITION &&
                compiledDove.task().actionBindings()[1u].actuator.x ==
                    MR_TASK_ACTUATOR_FLAPPING_POSITION &&
                compiledDove.task().actionBindings()[2u].actuator.x ==
                    MR_TASK_ACTUATOR_JOINT_POSITION &&
                compiledDove.flappingWingProgram()->tail.bodyIndex !=
                    MR_INVALID_INDEX &&
                compiledDove.flappingWingProgram()->fuselage.bodyIndex ==
                    compiledDove.flappingWingProgram()->rootBodyIndex &&
                compiledDove.flappingWingProgram()->fuselage
                    .referenceAreasAndDrag.w > 0.0f &&
                compiledDove.flappingWingProgram()->tail.qIndex !=
                    MR_INVALID_INDEX,
            "BirdFlow dove CompiledRun lost its whole-body action or aerodynamic program"
        );

        metalrobo::RunManifest crow;
        crow.id = "birdflow_american_crow_compiled_run_check";
        crow.robot = *crowRobot;
        crow.scene = metalrobo::makeBirdFlowAmericanCrowFlightScenePack();
        crow.sensors.id = "birdflow_american_crow_state_sensors";
        crow.task = metalrobo::makeBirdFlowAmericanCrowFlightTaskPack(
            crow.sensors.observation, crow.reality.reset
        );
        const auto crowTrackingReward = std::find_if(
            crow.task.rewards.begin(),
            crow.task.rewards.end(),
            [](const metalrobo::TaskRewardOperatorSpec& reward) {
                return reward.operation ==
                    metalrobo::TaskRewardOperator::linearVelocityTracking;
            }
        );
        require(
            crowTrackingReward != crow.task.rewards.end() &&
                crowTrackingReward->parameters.x == 0.25f,
            "BirdFlow American-crow training width must match its held-out tracking metric"
        );
        require(
            std::abs(crow.task.gaitPeriodSeconds - 1.0f / 4.6f) < 1.0e-6f,
            "BirdFlow American-crow task must retain its qualified 4.6 Hz clock"
        );
        crow.reality.id = "birdflow_american_crow_nominal_reality";
        crow.teacher.id = "no_teacher";
        crow.profile.id = "birdflow_american_crow_check_profile";
        crow.profile.environmentCount = 8u;
        crow.profile.controlSteps = 64u;
        crow.profile.physicsSubsteps = 4u;
        crow.profile.controlTimestepSeconds = 1.0f / 60.0f;
        metalrobo::CompiledRun compiledCrow;
        const auto crowStatus = metalrobo::compileRun(crow, compiledCrow);
        require(
            crowStatus.succeeded(),
            "BirdFlow American-crow CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(crowStatus.status)) +
                "] " + crowStatus.element + ": " + crowStatus.message
        );
        require(
            compiledCrow.valid() &&
                compiledCrow.flappingWingProgram() != nullptr &&
                compiledCrow.multicopterProgram() == nullptr &&
                compiledCrow.model().bodies.size() == 15u &&
                compiledCrow.model().joints.size() == 13u &&
                compiledCrow.task().actionBindings().size() == 14u &&
                compiledCrow.task().actionBindings()[0u].actuator.x ==
                    MR_TASK_ACTUATOR_FLAPPING_POSITION &&
                compiledCrow.task().actionBindings()[1u].actuator.x ==
                    MR_TASK_ACTUATOR_FLAPPING_POSITION &&
                compiledCrow.task().actionBindings()[2u].actuator.x ==
                    MR_TASK_ACTUATOR_JOINT_POSITION &&
                compiledCrow.task().actionBindings()[3u].actuator.x ==
                    MR_TASK_ACTUATOR_JOINT_POSITION &&
                compiledCrow.task().actionBindings()[4u].actuator.x ==
                    MR_TASK_ACTUATOR_JOINT_POSITION &&
                compiledCrow.task().actionBindings()[5u].actuator.x ==
                    MR_TASK_ACTUATOR_JOINT_POSITION &&
                (compiledCrow.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_GROUND_LEG_RESIDUAL) != 0u &&
                (compiledCrow.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_GROUND_TILT_ENVELOPE) != 0u &&
                (compiledCrow.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_GROUND_CARRIER_PHASE_OBSERVATION) !=
                    0u &&
                compiledCrow.task().layout().actorObservationSize == 83u &&
                compiledCrow.task().layout().criticObservationSize == 83u &&
                std::count_if(
                    compiledCrow.task().actorOperators().begin(),
                    compiledCrow.task().actorOperators().end(),
                    [](const MRTaskObservationOperatorGPU& operation) {
                        return operation.source.x ==
                            MR_TASK_OBSERVE_CROW_GROUND_CARRIER_PHASE;
                    }
                ) == 2 &&
                compiledCrow.flappingWingProgram()->wings[0u]
                    .pronationQIndex != MR_INVALID_INDEX &&
                compiledCrow.flappingWingProgram()->wings[1u]
                    .pronationQIndex != MR_INVALID_INDEX &&
                compiledCrow.flappingWingProgram()->wings[0u]
                    .sweepQIndex != MR_INVALID_INDEX &&
                compiledCrow.flappingWingProgram()->wings[1u]
                    .sweepQIndex != MR_INVALID_INDEX &&
                compiledCrow.flappingWingProgram()->wings[0u]
                    .rootJointParentAnchor.y != 0.0f &&
                compiledCrow.flappingWingProgram()->wings[0u]
                    .rootJointChildAnchor.y != 0.0f &&
                compiledCrow.flappingWingProgram()->wings[1u]
                    .rootJointParentAnchor.y != 0.0f &&
                compiledCrow.flappingWingProgram()->wings[1u]
                    .rootJointChildAnchor.y != 0.0f &&
                compiledCrow.flappingWingProgram()->wings[0u]
                    .rootToCenterAndArea.w == 0.075f &&
                compiledCrow.flappingWingProgram()->wings[1u]
                    .rootToCenterAndArea.w == 0.075f,
            "BirdFlow American-crow CompiledRun lost its standing-to-flight or aerodynamic program"
        );

        metalrobo::RunManifest journey;
        journey.id = "birdflow_american_crow_journey_compiled_run_check";
        journey.robot = *crowRobot;
        journey.scene = metalrobo::makeBirdFlowAmericanCrowFlightScenePack();
        journey.sensors.id = "birdflow_american_crow_journey_state_sensors";
        journey.task = metalrobo::makeBirdFlowAmericanCrowJourneyTaskPack(
            journey.sensors.observation, journey.reality.reset
        );
        journey.reality.id = "birdflow_american_crow_journey_reality";
        journey.teacher.id = "no_teacher";
        journey.profile.id = "birdflow_american_crow_journey_check_profile";
        journey.profile.environmentCount = 8u;
        journey.profile.controlSteps = 64u;
        journey.profile.physicsSubsteps = 4u;
        journey.profile.controlTimestepSeconds = 1.0f / 50.0f;
        metalrobo::CompiledRun compiledJourney;
        const auto journeyStatus = metalrobo::compileRun(
            journey, compiledJourney
        );
        require(
            journeyStatus.succeeded(),
            "BirdFlow American-crow journey CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(
                    journeyStatus.status)) + "] " + journeyStatus.element +
                ": " + journeyStatus.message
        );
        require(
            compiledJourney.valid() &&
                compiledJourney.task().actionBindings().size() == 15u &&
                compiledJourney.task().layout().actorObservationSize == 84u &&
                compiledJourney.task().layout().criticObservationSize == 84u &&
                (compiledJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
                (compiledJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_GROUND_GAIT_CARRIER) == 0u &&
                (compiledJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_LIFTOFF_TRIM_CARRIER) == 0u &&
                compiledJourney.task().header().schedule.z == 11u &&
                std::count_if(
                    compiledJourney.task().actorOperators().begin(),
                    compiledJourney.task().actorOperators().end(),
                    [](const MRTaskObservationOperatorGPU& operation) {
                        return operation.source.x ==
                            MR_TASK_OBSERVE_AVIAN_JOURNEY_PHASE;
                    }
                ) == 1 &&
                std::count_if(
                    compiledJourney.task().actorOperators().begin(),
                    compiledJourney.task().actorOperators().end(),
                    [](const MRTaskObservationOperatorGPU& operation) {
                        return operation.source.x ==
                            MR_TASK_OBSERVE_AVIAN_JOURNEY_STAGE;
                    }
                ) == 1,
            "BirdFlow American-crow journey lost its universal-policy contract"
        );

        metalrobo::RunManifest neuralJourney;
        neuralJourney.id =
            "birdflow_american_crow_neural_journey_compiled_run_check";
        neuralJourney.robot = *crowRobot;
        neuralJourney.scene =
            metalrobo::makeBirdFlowAmericanCrowFlightScenePack();
        neuralJourney.sensors.id =
            "birdflow_american_crow_neural_journey_state_sensors";
        neuralJourney.task =
            metalrobo::makeBirdFlowAmericanCrowNeuralJourneyTaskPack(
                neuralJourney.sensors.observation,
                neuralJourney.reality.reset
            );
        neuralJourney.reality.id =
            "birdflow_american_crow_neural_journey_reality";
        neuralJourney.teacher.id = "no_teacher";
        neuralJourney.profile.id =
            "birdflow_american_crow_neural_journey_check_profile";
        neuralJourney.profile.environmentCount = 8u;
        neuralJourney.profile.controlSteps = 64u;
        neuralJourney.profile.physicsSubsteps = 4u;
        neuralJourney.profile.controlTimestepSeconds = 1.0f / 50.0f;
        metalrobo::CompiledRun compiledNeuralJourney;
        const auto neuralJourneyStatus = metalrobo::compileRun(
            neuralJourney, compiledNeuralJourney
        );
        require(
            neuralJourneyStatus.succeeded(),
            "BirdFlow neural-only Crow journey CompiledRun failed [" +
                std::string(metalrobo::runCompileStatusName(
                    neuralJourneyStatus.status)) + "] " +
                neuralJourneyStatus.element + ": " +
                neuralJourneyStatus.message
        );
        const auto neuralOutcomes = compiledNeuralJourney.task().outcomes();
        require(
            compiledNeuralJourney.valid() &&
                compiledNeuralJourney.task().actionBindings().size() == 15u &&
                compiledNeuralJourney.task().layout().actorObservationSize ==
                    84u &&
                (compiledNeuralJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
                (compiledNeuralJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_APPROACH_ENVELOPE) == 0u &&
                compiledNeuralJourney.task().fingerprint() !=
                    compiledJourney.task().fingerprint() &&
                neuralOutcomes.size() == 10u &&
                neuralOutcomes[8u].id ==
                    "approach_pitch_warning_fraction" &&
                neuralOutcomes[8u].source == static_cast<std::uint32_t>(
                    metalrobo::TaskOutcomeSource::
                        avianJourneyApproachWarning
                ) &&
                neuralOutcomes[9u].id ==
                    "approach_pitch_full_envelope_fraction" &&
                neuralOutcomes[9u].source == static_cast<std::uint32_t>(
                    metalrobo::TaskOutcomeSource::
                        avianJourneyApproachFull
                ),
            "BirdFlow neural-only Crow journey retained hidden supervisor authority or lost shadow diagnostics"
        );
        metalrobo::RunManifest visualJourney = neuralJourney;
        visualJourney.id =
            "birdflow_american_crow_visual_neural_journey_compiled_run_check";
        visualJourney.sensors.observation = {};
        visualJourney.reality.reset = {};
        visualJourney.task =
            metalrobo::makeBirdFlowAmericanCrowVisualJourneyTaskPack(
                visualJourney.sensors.observation,
                visualJourney.reality.reset
            );
        metalrobo::CompiledRun compiledVisualJourney;
        const auto visualJourneyStatus = metalrobo::compileRun(
            visualJourney, compiledVisualJourney
        );
        require(
            visualJourneyStatus.succeeded() &&
                compiledVisualJourney.valid() &&
                compiledVisualJourney.task().layout().actorObservationSize ==
                    84u + 16u * 9u * 4u +
                        MR_TASK_MASKED_DEPTH_FEATURE_COUNT &&
                compiledVisualJourney.task().layout().criticObservationSize ==
                    84u &&
                compiledVisualJourney.task().header().visualLayout.x == 16u &&
                compiledVisualJourney.task().header().visualLayout.y == 9u &&
                compiledVisualJourney.task().header().visualLayout.z == 4u &&
                (compiledVisualJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_MASKED_DEPTH_FEATURES) != 0u &&
                (compiledVisualJourney.task().header().schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_APPROACH_ENVELOPE) == 0u &&
                compiledVisualJourney.task().observationFingerprint() !=
                    compiledNeuralJourney.task().observationFingerprint(),
            "BirdFlow v9 visual journey lost its distinct sensor-fast actor ABI"
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
            << " crow_journey_v7_task="
            << compiledJourney.task().fingerprint()
            << " crow_journey_v8_task="
            << compiledNeuralJourney.task().fingerprint()
            << " crow_journey_v9_task="
            << compiledVisualJourney.task().fingerprint()
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "run_program_check=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
