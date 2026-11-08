#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/MotionCompiler.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/TaskProgram.hpp"
#include "metalrobo/WorldPack.hpp"
#include "metalrobo/c_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>


namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }
struct TemporaryPackFiles {
 std::filesystem::path world="fixture/world.mrworld", task="fixture/task.taskpack", actuators="fixture/actuators.actuatorpack", sensors="fixture/sensors.sensorpack", reality="fixture/reality.realitypack";
};
void createFixture() {
    metalrobo::WorldTemplate worldTemplate;
    const metalrobo::WorldCompileResult templateStatus =
        metalrobo::compileEpisodeTwin(
            metalrobo::makeFrankaPickPlaceEpisodeTwin(),
            metalrobo::makeFrankaPickPlaceEngineModel(),
            worldTemplate
        );
    metalrobo::WorldFamily family;
    metalrobo::WorldProgram reality;
    reality.id = "world_pack_runtime_reality";
    reality.variations.push_back({
        .id = "pick_object_mass",
        .axis = MR_WORLD_VARIATION_PHYSICS,
        .distribution = MR_WORLD_DISTRIBUTION_UNIFORM,
        .target = MR_WORLD_TARGET_ASSET_MASS_SCALE,
        .targetId = "pick_object",
        .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
    });
    const metalrobo::WorldCompileResult familyStatus =
        metalrobo::compileWorldFamily(
            worldTemplate,
            reality,
            family
        );
    metalrobo::MRWorldPack pack;
    const metalrobo::WorldPackResult packStatus =
        metalrobo::compileWorldPack(family, pack);
    if (!templateStatus.succeeded() ||
        !familyStatus.succeeded() ||
        !packStatus.succeeded()) {
        fail("WorldPack locomotion fixture did not compile");
    }
    const metalrobo::LocomotionWorld imported =
        metalrobo::makeWorldPackLocomotionWorld(
            pack
        );
    const auto expectedScene =
        metalrobo::makeFrankaPickPlaceSceneState();
    const bool hasDynamicMass = std::any_of(
        imported.sceneBodies.begin(),
        imported.sceneBodies.end(),
        [](const MRBodyStateGPU& state) {
            return state.flagsAndIndices[0] ==
                    MR_MOTION_DYNAMIC &&
                state.linearVelocityAndInverseMass.w > 0.0f &&
                state.inverseInertiaWorldRow0.x > 0.0f;
        }
    );
    if (imported.articulationIndex != 0u ||
        imported.model.bodies.size() !=
            family.worldTemplate.engineModel.bodies.size() ||
        imported.sceneBodies.size() != expectedScene.size() ||
        !hasDynamicMass) {
        fail(
            "WorldPack did not materialize executable mechanics and scene state"
        );
    }

    TemporaryPackFiles files;
    metalrobo::TaskObservationProgram frankaObservations;
    metalrobo::TaskResetProgram frankaReset;
    metalrobo::TaskPack frankaTask;
    if (std::getenv("NUMI_HORIZON_AUTHORED_RESET")) {
        frankaReset.operators.push_back({
            .operation = metalrobo::TaskRandomizationOperator::sceneBodyPosition,
            .target = "pick_object", .component = 0u,
            .parameters = {0.9f, 0.9f, 0.0f, 0.0f},
        });
    }
    frankaTask.id = "franka_typed_actuator_pipeline_v1";
    metalrobo::RobotActuatorPack frankaActuators;
    frankaActuators.id = "franka_typed_actuators";
    const auto appendActuator = [&frankaTask, &frankaObservations,
                                 &frankaActuators](
        metalrobo::RobotActuatorSpec actuator
    ) {
        const std::string id = actuator.id;
        frankaTask.actions.push_back({.actuator = id});
        frankaObservations.actorFrame.push_back({
            .source = metalrobo::TaskObservationSource::previousAction,
            .target = id,
        });
        frankaObservations.critic.push_back({
            .source = metalrobo::TaskObservationSource::previousAction,
            .target = id,
        });
        frankaActuators.actuators.push_back(std::move(actuator));
    };
    appendActuator({
        .id = "joint1_velocity",
        .kind = metalrobo::RobotActuatorKind::jointVelocity,
        .target = "panda_joint1",
        .scale = 0.40f,
        .parameters = {20.0f, 0.0f, 0.0f, 0.0f},
    });
    appendActuator({
        .id = "joint2_effort",
        .kind = metalrobo::RobotActuatorKind::jointEffort,
        .target = "panda_joint2",
        .scale = 5.0f,
    });
    appendActuator({
        .id = "elbow_tendon",
        .kind = metalrobo::RobotActuatorKind::tendonPosition,
        .target = "elbow_synergy",
        .scale = 0.08f,
        .parameters = {45.0f, 4.0f, 18.0f, 0.0f},
        .terms = {
            {.joint = "panda_joint3", .coefficient = 1.0f},
            {.joint = "panda_joint4", .coefficient = -0.5f},
        },
    });
    constexpr std::array<std::string_view, 6u> wrenchIds{{
        "hand_force_x", "hand_force_y", "hand_force_z",
        "hand_torque_x", "hand_torque_y", "hand_torque_z",
    }};
    for (std::uint32_t component = 0u; component < wrenchIds.size();
         ++component) {
        appendActuator({
            .id = std::string{wrenchIds[component]},
            .kind = metalrobo::RobotActuatorKind::bodyWrench,
            .target = "panda_link7",
            .scale = component < 3u ? 8.0f : 2.0f,
            .component = component,
        });
    }
    frankaTask.rewards.push_back({
        .operation = metalrobo::TaskRewardOperator::constant,
        .weight = 0.0f,
    });
    if (std::getenv("NUMI_HORIZON_FORCE_FAILURE")) {
        // Artificially trigger an authored non-timeout termination. This is
        // a metric control, not evidence of an actual contact failure.
        frankaTask.contactGroups.push_back({
            .id = "test_failure", .bodies = {"panda_link7"}});
        frankaTask.terminations.push_back({
            .operation = metalrobo::TaskTerminationOperator::contactGroup,
            .sourceGroup = "test_failure",
            .reason = MR_TASK_TERMINATION_CONTACT,
            .priority = 100u,
            .threshold = -1.0f,
        });
    }
    frankaTask.maximumEpisodeSteps = 256u;
    const metalrobo::WorldPackResult worldWrite =
        metalrobo::writeWorldPack(pack, files.world);
    const metalrobo::LearningPackResult taskWrite =
        metalrobo::writeTaskPack(frankaTask, files.task);
    const metalrobo::LearningPackResult actuatorWrite =
        metalrobo::writeRobotActuatorPack(
            frankaActuators,
            files.actuators);
    const metalrobo::LearningPackResult sensorWrite =
        metalrobo::writeSensorProgramPack(
            {.id = "franka_sensors", .observation = frankaObservations},
            files.sensors);
    const metalrobo::LearningPackResult realityWrite =
        metalrobo::writeRealityProgramPack(
            {
                .id = "franka_reality",
                .program = reality,
                .sourceProgramFingerprint = family.program.fingerprint,
                .reset = frankaReset,
            },
            files.reality);
    if (!worldWrite.succeeded() || !taskWrite.succeeded() ||
        !actuatorWrite.succeeded() || !sensorWrite.succeeded() ||
        !realityWrite.succeeded()) {
        fail("WorldPack runtime fixtures could not be persisted");
    }

}
}
int main() { try { createFixture(); return 0; } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; } }
