#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/Simulation.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/TaskProgram.hpp"
#include "metalrobo/WorldPack.hpp"
#include "metalrobo/counter_rng.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

class TemporaryPackFiles {
public:
    TemporaryPackFiles() {
        const std::filesystem::path directory =
            std::filesystem::temp_directory_path();
        const std::string suffix =
            std::to_string(
                static_cast<unsigned long long>(::getpid())
            );
        task = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".taskpack");
        policy = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".policypack");
        rollout = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".rolloutpack");
        borrowedRollout = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".borrowed.rolloutpack");
        world = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".worldpack");
    }

    ~TemporaryPackFiles() {
        std::error_code ignored;
        std::filesystem::remove(task, ignored);
        std::filesystem::remove(policy, ignored);
        std::filesystem::remove(rollout, ignored);
        std::filesystem::remove(borrowedRollout, ignored);
        std::filesystem::remove(world, ignored);
    }

    std::filesystem::path task;
    std::filesystem::path policy;
    std::filesystem::path rollout;
    std::filesystem::path borrowedRollout;
    std::filesystem::path world;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

float sensorGaussianReference(
    const std::uint64_t seed,
    const std::uint32_t environment,
    const std::uint32_t episode,
    const std::uint64_t sensorIdentity,
    const std::uint64_t sample,
    const std::uint32_t channel,
    const std::uint32_t purpose
) {
    const float first = std::max(
        mr_sensor_counter_uniform(
            seed,
            environment,
            episode,
            sensorIdentity,
            sample,
            channel,
            purpose
        ),
        1.0f / 16777216.0f
    );
    const float second = mr_sensor_counter_uniform(
        seed,
        environment,
        episode,
        sensorIdentity,
        sample,
        channel,
        purpose + 1u
    );
    return std::sqrt(-2.0f * std::log(first)) *
        std::cos(6.28318530717958647692f * second);
}

std::uint64_t compileFloatingBaseTaskFixture() {
    constexpr std::string_view urdf = R"(
<robot name="generic_locomotion_fixture">
  <link name="base">
    <inertial>
      <origin xyz="0 0 -0.05"/>
      <mass value="5"/>
      <inertia ixx="0.2" ixy="0" ixz="0"
               iyy="0.25" iyz="0" izz="0.15"/>
    </inertial>
    <collision>
      <geometry><box size="0.3 0.2 0.4"/></geometry>
    </collision>
  </link>
  <link name="foot">
    <inertial>
      <origin xyz="0.01 0 -0.02"/>
      <mass value="1"/>
      <inertia ixx="0.02" ixy="0" ixz="0"
               iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <geometry><box size="0.2 0.15 0.08"/></geometry>
    </collision>
  </link>
  <joint name="hip" type="revolute">
    <parent link="base"/>
    <child link="foot"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="100" velocity="5"/>
  </joint>
</robot>
)";
    metalrobo::SimulationDescription authored;
    metalrobo::RobotDescriptionCookOptions options;
    options.rootMode =
        metalrobo::RobotDescriptionRootMode::floating;
    const metalrobo::RobotDescriptionDiagnostics cooked =
        metalrobo::cookRobotDescription(
            urdf,
            {},
            authored.model,
            options,
            "generic_locomotion_fixture.urdf"
        );
    if (!cooked.succeeded()) {
        fail(
            "generic URDF cook failed: " +
            cooked.element + ": " + cooked.message
        );
    }
    metalrobo::appendBuiltinSurface(
        authored.model,
        authored.sceneBodies,
        metalrobo::BuiltinSurface::ground
    );

    authored.task.id = "generic_locomotion_fixture";
    authored.task.actions = {{
        .joint = "hip",
        .scale = 0.2f,
    }};
    authored.task.actorHistoryLength = 2u;
    authored.task.actorFrame = {
        {
            .source =
                metalrobo::TaskObservationSource::
                    rootAngularVelocityLocal,
            .component = 1u,
        },
        {
            .source =
                metalrobo::TaskObservationSource::
                    jointPositionError,
            .target = "hip",
        },
        {
            .source =
                metalrobo::TaskObservationSource::
                    contactWrenchLocal,
            .target = "foot_contact",
            .component = 2u,
            .scale = 0.01f,
        },
    };
    authored.task.critic = {{
        .source =
            metalrobo::TaskObservationSource::rootHeight,
    }};
    authored.task.contactGroups = {{
        .id = "foot_contact",
        .bodies = {"foot"},
        .support = true,
        .referenceBody = "foot",
    }};
    authored.task.rewards = {{
        .operation =
            metalrobo::TaskRewardOperator::constant,
        .weight = 1.0f,
    }};
    authored.task.terminations = {{
        .operation =
            metalrobo::TaskTerminationOperator::
                minimumRootHeight,
        .reason = MR_TASK_TERMINATION_HEIGHT,
        .threshold = 0.1f,
    }};
    authored.task.terrain.body = "locomotion_ground";
    authored.task.terrain.sampleOffsets = {{
        0.0f, 0.0f, 0.0f, 0.0f,
    }};
    authored.task.terrain.resetTranslations = {{
        0.0f, 0.0f, 0.0f, 0.0f,
    }};

    metalrobo::CompiledSimulation compiled;
    const metalrobo::SimulationCompileDiagnostics status =
        metalrobo::compileSimulation(
            authored,
            0u,
            compiled
        );
    if (!status.world.succeeded() ||
        !status.task.succeeded()) {
        fail(
            "generic imported floating-base task compile failed: " +
            status.world.message + " " +
            status.task.element + ": " +
            status.task.message
        );
    }
    const metalrobo::TaskProgramLayout& layout =
        compiled.task.layout();
    if (layout.actionCount != 1u ||
        layout.actorFrameSize != 3u ||
        layout.actorObservationSize != 6u ||
        layout.criticFrameSize != 1u ||
        layout.criticHistoryLength != 1u ||
        layout.criticObservationSize != 7u ||
        layout.contactMetricCount != 12u ||
        compiled.task.actionBindings().front().indices.x != 0u ||
        std::abs(
            compiled.task.header().rootReference.z - 0.05f
        ) > 1.0e-6f ||
        compiled.task.contactGroups().size() != 1u ||
        compiled.task.contactGroups()
                .front()
                .reference.y != 6u ||
        std::abs(
            compiled.task.contactGroups()
                    .front()
                    .kinematicReference.x +
                0.01f
        ) > 1.0e-6f ||
        std::abs(
            compiled.task.contactGroups()
                    .front()
                    .kinematicReference.z -
                0.02f
        ) > 1.0e-6f ||
        compiled.task.terrainResetTranslations().size() != 1u) {
        fail(
            "generic imported robot did not compile the authored task contract"
        );
    }
    return compiled.task.fingerprint();
}

struct FixedBaseTaskEvidence {
    std::uint64_t fingerprint = 0u;
    std::uint64_t pipelineCount = 0u;
    float forceNormNewtons = 0.0f;
    float torqueNormNewtonMetres = 0.0f;
};

FixedBaseTaskEvidence compileFixedBaseTaskFixture() {
    constexpr std::string_view urdf = R"(
<robot name="fixed_base_task_fixture">
  <link name="mount">
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.2" ixy="0" ixz="0"
               iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="tool">
    <inertial>
      <origin xyz="0 0 0.1"/>
      <mass value="1"/>
      <inertia ixx="0.02" ixy="0" ixz="0"
               iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><capsule radius="0.04" length="0.2"/></geometry></collision>
  </link>
  <joint name="axis" type="revolute">
    <parent link="mount"/>
    <child link="tool"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="20" velocity="3"/>
  </joint>
</robot>
)";
    metalrobo::SimulationDescription authored;
    metalrobo::RobotDescriptionCookOptions options;
    options.rootMode =
        metalrobo::RobotDescriptionRootMode::fixed;
    const auto cooked = metalrobo::cookRobotDescription(
        urdf,
        {},
        authored.model,
        options,
        "fixed_base_task_fixture.urdf"
    );
    if (!cooked.succeeded()) {
        fail("fixed-base URDF cook failed: " + cooked.message);
    }
    metalrobo::appendBuiltinSurface(
        authored.model,
        authored.sceneBodies,
        metalrobo::BuiltinSurface::ground
    );
    authored.sceneBodies.back().position.z = -1.0f;
    authored.task.id = "fixed_base_joint_control";
    authored.task.actions = {{
        .joint = "axis",
        .scale = 0.15f,
    }};
    authored.task.actorFrame = {
        {
            .source =
                metalrobo::TaskObservationSource::jointPositionError,
            .target = "axis",
        },
        {
            .source = metalrobo::TaskObservationSource::jointVelocity,
            .target = "axis",
        },
        {
            .source = metalrobo::TaskObservationSource::previousAction,
            .target = "axis",
        },
        {
            .source =
                metalrobo::TaskObservationSource::framePositionWorld,
            .target = "tool_tip",
            .component = 0u,
        },
        {
            .source =
                metalrobo::TaskObservationSource::framePositionWorld,
            .target = "tool_tip",
            .component = 1u,
        },
        {
            .source =
                metalrobo::TaskObservationSource::framePositionWorld,
            .target = "tool_tip",
            .component = 2u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameGoalPositionError,
            .target = "tool_tip",
            .goal = "home",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameGoalPositionError,
            .target = "tool_tip",
            .goal = "home",
            .component = 1u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameGoalPositionError,
            .target = "tool_tip",
            .goal = "home",
            .component = 2u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "tool_pose_delayed",
            .component = 2u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValidity,
            .target = "tool_pose_delayed",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValidity,
            .target = "tool_pose_30hz",
            .component = 1u,
        },
    };
    authored.task.criticIncludesCleanHistory = false;
    authored.task.critic = {
        {
            .source =
                metalrobo::TaskObservationSource::jointPositionError,
            .target = "axis",
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameGoalOrientationError,
            .target = "tool_tip",
            .goal = "home",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameGoalOrientationError,
            .target = "tool_tip",
            .goal = "home",
            .component = 1u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameGoalOrientationError,
            .target = "tool_tip",
            .goal = "home",
            .component = 2u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "tool_pose_30hz",
            .component = 2u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValidity,
            .target = "tool_pose_30hz",
            .component = 3u,
        },
        {
            .source =
                metalrobo::TaskObservationSource::framePositionWorld,
            .target = "world_anchor",
            .component = 2u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameRelativePosition,
            .target = "tool_tip",
            .reference = "world_anchor",
            .component = 0u,
        },
    };
    authored.task.frames = {
        {
            .id = "tool_tip",
            .site = "tool_control_site",
            .localPosition = {0.0f, 0.0f, 0.05f, 0.0f},
            .localOrientation = {
                0.0f,
                0.0f,
                -0.7071067811865476f,
                0.7071067811865476f,
            },
        },
        {
            .id = "world_anchor",
            .body = "locomotion_ground",
            .localPosition = {0.125f, -0.25f, 1.4f, 0.0f},
            .localOrientation = {
                0.0f,
                0.0f,
                0.7071067811865476f,
                0.7071067811865476f,
            },
        },
    };
    authored.task.goals = {{
        .id = "home",
        .position = {0.0f, 0.0f, 0.3f, 1.0f},
    }};
    authored.task.rewards = {
        {
            .operation = metalrobo::TaskRewardOperator::constant,
            .weight = 1.0f,
        },
        {
            .operation =
                metalrobo::TaskRewardOperator::jointVelocitySquared,
            .weight = -0.01f,
        },
        {
            .operation = metalrobo::TaskRewardOperator::
                framePositionTracking,
            .sourceGroup = "tool_tip",
            .goal = "home",
            .weight = 0.5f,
            .parameters = {0.01f, 0.0f, 0.0f, 0.0f},
        },
    };
    authored.task.terminations = {{
        .operation = metalrobo::TaskTerminationOperator::
            maximumFramePositionError,
        .sourceGroup = "tool_tip",
        .goal = "home",
        .reason = MR_TASK_TERMINATION_GOAL_ERROR,
        .priority = 1u,
        .threshold = 1.0f,
        .failurePenalty = -1.0f,
    }};
    authored.task.randomization = {
        {
            .operation =
                metalrobo::TaskRandomizationOperator::actionPosition,
            .parameters = {0.0f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation =
                metalrobo::TaskRandomizationOperator::actionVelocity,
            .parameters = {0.0f, 0.0f, 0.0f, 0.0f},
        },
    };
    authored.task.maximumEpisodeSteps = 64u;

    const auto toolBody = std::find(
        authored.model.bodyNames.begin(),
        authored.model.bodyNames.end(),
        "tool"
    );
    if (toolBody == authored.model.bodyNames.end()) {
        fail("fixed-base sensor parent body did not resolve");
    }
    const std::uint32_t toolBodyIndex =
        static_cast<std::uint32_t>(
            toolBody - authored.model.bodyNames.begin()
        );
    authored.model.sites.push_back({
        .id = "tool_control_site",
        .bodyIndex = toolBodyIndex,
        .localPosition = {0.0f, 0.0f, 0.15f, 0.0f},
        .localOrientation = {
            0.0f,
            0.0f,
            0.7071067811865476f,
            0.7071067811865476f,
        },
    });
    authored.sensors = {
        {
            .id = "tool_pose_delayed",
            .parentSite = "tool_control_site",
            .parentKind = MR_WORLD_SENSOR_PARENT_ASSET,
            .kind = MR_WORLD_SENSOR_STATE,
            .localPose = {
                .position = {0.0f, 0.0f, 0.05f, 0.0f},
                .orientation = {
                    0.0f,
                    0.0f,
                    -0.7071067811865476f,
                    0.7071067811865476f,
                },
            },
            .latencySeconds = 0.02f,
            .nominalRateHz = 50.0f,
            .schedulePhase =
                MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 2u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_ACTOR |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
        {
            .id = "tool_pose_30hz",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_STATE,
            .localPose = {
                .position = {0.0f, 0.0f, 0.2f, 0.0f},
            },
            .latencySeconds = 0.0f,
            .nominalRateHz = 30.0f,
            .schedulePhase =
                MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 2u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_ACTOR |
                MR_WORLD_SENSOR_CONSUMER_CRITIC |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
    };

    metalrobo::PolicyPack policy;
    policy.id = "fixed_base_sensor_policy";
    policy.layers = {{
        .inputCount = 12u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = [] {
            std::vector<float> values(12u);
            for (std::size_t index = 0u;
                 index < values.size();
                 ++index) {
                values[index] =
                    0.01f * static_cast<float>(index + 1u);
            }
            return values;
        }(),
        .bias = std::vector<float>(1u, 0.05f),
    }};
    policy.criticLayers = {{
        .inputCount = 8u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(8u, 0.02f),
        .bias = std::vector<float>(1u, 0.1f),
    }};
    authored.policy = std::move(policy);

    metalrobo::CompiledSimulation compiled;
    const auto status = metalrobo::compileSimulation(
        authored,
        0u,
        compiled
    );
    if (!status.succeeded() ||
        !compiled.valid() ||
        !compiled.sensors.valid() ||
        compiled.sensors.layout().sensorCount != 2u ||
        compiled.sensors.layout().outputElementCount != 14u ||
        compiled.sensors.layout().historyElementCount != 28u ||
        compiled.task.layout().actionCount != 1u ||
        compiled.task.layout().actorObservationSize != 12u ||
        compiled.task.layout().criticObservationSize != 8u ||
        compiled.task.sensorFingerprint() !=
            compiled.sensors.fingerprint() ||
        compiled.task.header().typedCounts.x != 2u ||
        compiled.task.header().typedCounts.y != 1u ||
        compiled.task.frames().size() != 2u ||
        compiled.task.goals().size() != 1u ||
        std::abs(
            compiled.task.frames().front().localPosition.z -
            0.1f
        ) > 1.0e-6f ||
        std::abs(
            compiled.task.frames().front().localOrientation.x
        ) > 1.0e-6f ||
        std::abs(
            compiled.task.frames().front().localOrientation.y
        ) > 1.0e-6f ||
        std::abs(
            compiled.task.frames().front().localOrientation.z
        ) > 1.0e-6f ||
        std::abs(
            compiled.task.frames().front().localOrientation.w -
            1.0f
        ) > 1.0e-6f ||
        compiled.task.frames()[1].indices.y !=
            MR_TASK_FRAME_SOURCE_SCENE_BODY ||
        compiled.task.frames()[1].indices.z != 0u ||
        compiled.task.frames()[1].indices.w != MR_INVALID_INDEX ||
        (compiled.task.header().schedule.w &
         MR_TASK_PROGRAM_FLOATING_ROOT) != 0u) {
        fail(
            "fixed-base task did not compile through the generic task route: " +
            status.task.message
        );
    }
    metalrobo::TaskPack unresolvedSite = authored.task;
    unresolvedSite.frames.front().site = "missing_site";
    metalrobo::CompiledTaskProgram preservedTask = compiled.task;
    const std::uint64_t preservedTaskFingerprint =
        preservedTask.fingerprint();
    const auto unresolvedSiteStatus =
        metalrobo::compileTaskProgram(
            unresolvedSite,
            compiled.world,
            compiled.sensors,
            preservedTask
        );
    if (unresolvedSiteStatus.status !=
            metalrobo::TaskCompileStatus::unresolvedSemantic ||
        preservedTask.fingerprint() !=
            preservedTaskFingerprint) {
        fail(
            "unresolved task site did not fail transactionally"
        );
    }
    std::vector<metalrobo::SensorSpec> unresolvedSiteSensors =
        authored.sensors;
    unresolvedSiteSensors.front().parentSite = "missing_site";
    metalrobo::CompiledSensorProgram preservedSensors =
        compiled.sensors;
    const std::uint64_t preservedSensorFingerprint =
        preservedSensors.fingerprint();
    const auto unresolvedSensorSiteStatus =
        metalrobo::compileSensorProgram(
            unresolvedSiteSensors,
            authored.tactileSystem,
            compiled.world,
            preservedSensors
        );
    if (unresolvedSensorSiteStatus.status !=
            metalrobo::SensorCompileStatus::unresolvedSemantic ||
        preservedSensors.fingerprint() !=
            preservedSensorFingerprint) {
        fail(
            "unresolved sensor site did not fail transactionally"
        );
    }

    constexpr std::size_t environments = 2u;
    constexpr std::size_t controlSteps = 4u;
    std::vector<float> initialQ(
        environments * compiled.world.nq()
    );
    std::vector<float> initialV(
        environments * compiled.world.nv(),
        0.0f
    );
    std::vector<MRBodyStateGPU> initialSceneBodies(
        environments * authored.sceneBodies.size()
    );
    for (std::size_t environment = 0u;
         environment < environments;
         ++environment) {
        std::copy(
            authored.model.defaultQ.begin(),
            authored.model.defaultQ.end(),
            initialQ.begin() +
                static_cast<std::ptrdiff_t>(
                    environment * compiled.world.nq()
                )
        );
        std::copy(
            authored.sceneBodies.begin(),
            authored.sceneBodies.end(),
            initialSceneBodies.begin() +
                static_cast<std::ptrdiff_t>(
                    environment * authored.sceneBodies.size()
                )
        );
    }
    std::vector<std::uint32_t> resetMasks(
        environments * controlSteps,
        0u
    );
    resetMasks[2u * environments] = 1u;
    metalrobo::MetalWorldStepConfig step;
    step.timestepSeconds = 0.02f;
    step.physicsSubsteps = 2u;
    step.executionMode =
        metalrobo::MetalWorldExecutionMode::numiSolver;
    step.actuationMode =
        metalrobo::MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram = compiled.task;
    step.sensorProgram = compiled.sensors;
    step.policyProgram = compiled.policy;
    step.evaluateFinalPolicy = true;
    step.publishSensorOutputs = true;
    step.ccdMode = metalrobo::MetalWorldCCDMode::disabled;
    metalrobo::MetalWorldConfig contextConfiguration;
    contextConfiguration.maximumInFlightSubmissions = 1u;
    metalrobo::MetalWorldContext context(contextConfiguration);
    metalrobo::MetalWorldResult result;
    const auto executed = context.run(
        compiled.world,
        {
            .environmentCount = environments,
            .controlStepCount = controlSteps,
            .initialQ = initialQ,
            .initialV = initialV,
            .resetMasks = resetMasks,
            .initialSceneBodies = initialSceneBodies,
        },
        step,
        result
    );
    if (!executed.succeeded() ||
        result.transitions.size() !=
            environments * controlSteps ||
        result.actorObservations.size() !=
            environments * (controlSteps + 1u) * 12u ||
        result.policyLatents.size() !=
            environments * (controlSteps + 1u) ||
        result.sensorOutputs.size() != environments * 14u ||
        result.sensorMetadata.size() != environments * 2u ||
        std::any_of(
            result.environmentStatuses.begin(),
            result.environmentStatuses.end(),
            [](const metalrobo::MetalWorldStatus& status) {
                return status.code != MR_STEP_SUCCESS;
            }
        )) {
        fail(
            "fixed-base task did not execute through the generic Metal task graph: " +
            executed.message
        );
    }
    constexpr std::uint64_t corruptionSeed =
        0x5319a7c2d84e6b01ull;
    std::vector<metalrobo::SensorSpec> cleanCorruptionSensors{
        {
            .id = "tool_twist_corruption",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_FRAME_TWIST_WORLD,
            .localPose = {
                .position = {0.0f, 0.0f, 0.2f, 0.0f},
            },
            .nominalRateHz = 50.0f,
            .schedulePhase =
                MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
        {
            .id = "tool_twist_dropout",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_FRAME_TWIST_WORLD,
            .localPose = {
                .position = {0.0f, 0.0f, 0.2f, 0.0f},
            },
            .nominalRateHz = 50.0f,
            .schedulePhase =
                MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
        {
            .id = "tool_pose_corruption",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_STATE,
            .localPose = {
                .position = {0.0f, 0.0f, 0.2f, 0.0f},
            },
            .nominalRateHz = 50.0f,
            .schedulePhase =
                MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
    };
    std::vector<metalrobo::SensorSpec> corruptedSensors =
        cleanCorruptionSensors;
    corruptedSensors[0u].valueNoiseSigma = 0.07f;
    corruptedSensors[0u].biasNoiseSigma = 0.03f;
    corruptedSensors[1u].dropoutProbability = 1.0f;
    corruptedSensors[2u].valueNoiseSigma = 0.04f;
    corruptedSensors[2u].biasNoiseSigma = 0.02f;
    const metalrobo::CookedTactileSystem noTactile;
    metalrobo::CompiledSensorProgram cleanCorruptionProgram;
    metalrobo::CompiledSensorProgram corruptedProgram;
    const auto cleanCorruptionCompile =
        metalrobo::compileSensorProgram(
            cleanCorruptionSensors,
            noTactile,
            compiled.world,
            cleanCorruptionProgram
        );
    const auto corruptedCompile =
        metalrobo::compileSensorProgram(
            corruptedSensors,
            noTactile,
            compiled.world,
            corruptedProgram
        );
    const MRSensorDescriptorGPU cleanCorruptionDescriptor =
        cleanCorruptionProgram.valid()
        ? cleanCorruptionProgram.descriptors()[0u]
        : MRSensorDescriptorGPU{};
    const MRSensorDescriptorGPU corruptedDescriptor =
        corruptedProgram.valid()
        ? corruptedProgram.descriptors()[0u]
        : MRSensorDescriptorGPU{};
    if (!cleanCorruptionCompile.succeeded() ||
        !corruptedCompile.succeeded() ||
        cleanCorruptionProgram.layout().outputElementCount != 19u ||
        corruptedProgram.layout().outputElementCount != 19u ||
        cleanCorruptionDescriptor.randomIdentity.x !=
            corruptedDescriptor.randomIdentity.x ||
        cleanCorruptionDescriptor.randomIdentity.y !=
            corruptedDescriptor.randomIdentity.y ||
        (corruptedDescriptor.randomIdentity.x == 0u &&
         corruptedDescriptor.randomIdentity.y == 0u)) {
        fail("native SensorIR corruption contract did not compile");
    }
    constexpr std::size_t corruptionSteps = 2u;
    const auto executeCorruption = [&] (
        const metalrobo::CompiledSensorProgram& sensorProgram,
        const std::uint64_t seed,
        const bool resetFirstStep,
        metalrobo::MetalWorldResult& sensorResult
    ) {
        std::vector<float> zeroEfforts(
            environments * corruptionSteps *
                compiled.world.nv(),
            0.0f
        );
        std::vector<std::uint32_t> corruptionResetMasks(
            environments * corruptionSteps,
            0u
        );
        if (resetFirstStep) {
            std::fill_n(
                corruptionResetMasks.begin(),
                environments,
                1u
            );
        }
        metalrobo::MetalWorldBatch sensorBatch{
            .environmentCount = environments,
            .controlStepCount = corruptionSteps,
            .initialQ = initialQ,
            .initialV = initialV,
            .efforts = zeroEfforts,
            .resetMasks = resetFirstStep
                ? std::span<const std::uint32_t>{
                      corruptionResetMasks
                  }
                : std::span<const std::uint32_t>{},
            .resetQ = resetFirstStep
                ? std::span<const float>{initialQ}
                : std::span<const float>{},
            .resetV = resetFirstStep
                ? std::span<const float>{initialV}
                : std::span<const float>{},
            .initialSceneBodies = initialSceneBodies,
            .resetSceneBodies = resetFirstStep
                ? std::span<const MRBodyStateGPU>{
                      initialSceneBodies
                  }
                : std::span<const MRBodyStateGPU>{},
        };
        metalrobo::MetalWorldStepConfig sensorStep;
        sensorStep.timestepSeconds = 0.02f;
        sensorStep.physicsSubsteps = 2u;
        sensorStep.executionMode =
            metalrobo::MetalWorldExecutionMode::numiSolver;
        sensorStep.actuationMode =
            metalrobo::MetalWorldActuationMode::effort;
        sensorStep.sensorProgram = sensorProgram;
        sensorStep.taskSeed = seed;
        sensorStep.publishSensorOutputs = true;
        sensorStep.ccdMode =
            metalrobo::MetalWorldCCDMode::disabled;
        metalrobo::MetalWorldContext sensorContext(
            contextConfiguration
        );
        const auto sensorExecution = sensorContext.run(
            compiled.world,
            sensorBatch,
            sensorStep,
            sensorResult
        );
        if (!sensorExecution.succeeded() ||
            sensorResult.sensorOutputs.size() !=
                environments * 19u ||
            sensorResult.sensorMetadata.size() !=
                environments * 3u) {
            fail(
                "native SensorIR corruption execution failed: " +
                sensorExecution.message
            );
        }
    };
    metalrobo::MetalWorldResult cleanCorruptionResult;
    metalrobo::MetalWorldResult corruptedResult;
    metalrobo::MetalWorldResult replayedCorruptionResult;
    metalrobo::MetalWorldResult changedSeedResult;
    metalrobo::MetalWorldResult cleanResetCorruptionResult;
    metalrobo::MetalWorldResult resetCorruptionResult;
    executeCorruption(
        cleanCorruptionProgram,
        corruptionSeed,
        false,
        cleanCorruptionResult
    );
    executeCorruption(
        corruptedProgram,
        corruptionSeed,
        false,
        corruptedResult
    );
    executeCorruption(
        corruptedProgram,
        corruptionSeed,
        false,
        replayedCorruptionResult
    );
    executeCorruption(
        corruptedProgram,
        corruptionSeed + 1u,
        false,
        changedSeedResult
    );
    executeCorruption(
        cleanCorruptionProgram,
        corruptionSeed,
        true,
        cleanResetCorruptionResult
    );
    executeCorruption(
        corruptedProgram,
        corruptionSeed,
        true,
        resetCorruptionResult
    );
    const std::uint64_t sensorIdentity =
        static_cast<std::uint64_t>(
            corruptedDescriptor.randomIdentity.x
        ) |
        (static_cast<std::uint64_t>(
             corruptedDescriptor.randomIdentity.y
         ) << 32u);
    const MRSensorDescriptorGPU poseCorruptionDescriptor =
        corruptedProgram.descriptors()[2u];
    const std::uint64_t poseSensorIdentity =
        static_cast<std::uint64_t>(
            poseCorruptionDescriptor.randomIdentity.x
        ) |
        (static_cast<std::uint64_t>(
             poseCorruptionDescriptor.randomIdentity.y
         ) << 32u);
    const auto verifyCorruption = [&] (
        const metalrobo::MetalWorldResult& clean,
        const metalrobo::MetalWorldResult& noisy,
        const std::uint32_t episode,
        const std::uint64_t publishedSample
    ) {
        for (std::uint32_t environment = 0u;
             environment < environments;
             ++environment) {
            const std::size_t outputBase = environment * 19u;
            for (std::uint32_t channel = 0u;
                 channel < 6u;
                 ++channel) {
                const float expectedDelta =
                    corruptedSensors[0u].biasNoiseSigma *
                        sensorGaussianReference(
                            corruptionSeed,
                            environment,
                            episode,
                            sensorIdentity,
                            0u,
                            channel,
                            0u
                        ) +
                    corruptedSensors[0u].valueNoiseSigma *
                        sensorGaussianReference(
                            corruptionSeed,
                            environment,
                            episode,
                            sensorIdentity,
                            publishedSample,
                            channel,
                            2u
                        );
                const float actualDelta =
                    noisy.sensorOutputs[outputBase + channel] -
                    clean.sensorOutputs[outputBase + channel];
                if (std::abs(actualDelta - expectedDelta) >
                    8.0e-5f *
                        (1.0f + std::abs(expectedDelta))) {
                    fail(
                        "native SensorIR Gaussian corruption disagrees with the counter-RNG host mirror"
                    );
                }
            }
            for (std::size_t channel = 0u;
                 channel < 6u;
                 ++channel) {
                if (noisy.sensorOutputs[
                        outputBase + 6u + channel
                    ] != 0.0f) {
                    fail(
                        "native SensorIR dropout published a partial sample"
                    );
                }
            }
            const std::uint32_t noisyValidity =
                noisy.sensorMetadata[environment * 3u]
                    .ageValidityAndLayout.y;
            const std::uint32_t dropoutValidity =
                noisy.sensorMetadata[environment * 3u + 1u]
                    .ageValidityAndLayout.y;
            if ((noisyValidity &
                 (MR_SENSOR_SAMPLE_VALID |
                  MR_SENSOR_SAMPLE_FRESH)) !=
                    (MR_SENSOR_SAMPLE_VALID |
                     MR_SENSOR_SAMPLE_FRESH) ||
                (dropoutValidity &
                 (MR_SENSOR_SAMPLE_FRESH |
                  MR_SENSOR_SAMPLE_DROPPED)) !=
                    (MR_SENSOR_SAMPLE_FRESH |
                     MR_SENSOR_SAMPLE_DROPPED) ||
                (dropoutValidity & MR_SENSOR_SAMPLE_VALID) != 0u) {
                fail(
                    "native SensorIR dropout validity contract is inconsistent"
                );
            }
            const std::size_t poseBase = outputBase + 12u;
            for (std::uint32_t channel = 0u;
                 channel < 3u;
                 ++channel) {
                const float expectedDelta =
                    corruptedSensors[2u].biasNoiseSigma *
                        sensorGaussianReference(
                            corruptionSeed,
                            environment,
                            episode,
                            poseSensorIdentity,
                            0u,
                            channel,
                            0u
                        ) +
                    corruptedSensors[2u].valueNoiseSigma *
                        sensorGaussianReference(
                            corruptionSeed,
                            environment,
                            episode,
                            poseSensorIdentity,
                            publishedSample,
                            channel,
                            2u
                        );
                const float actualDelta =
                    noisy.sensorOutputs[poseBase + channel] -
                    clean.sensorOutputs[poseBase + channel];
                if (std::abs(actualDelta - expectedDelta) >
                    8.0e-5f *
                        (1.0f + std::abs(expectedDelta))) {
                    fail(
                        "native SensorIR pose translation corruption disagrees with the host mirror"
                    );
                }
            }
            std::array<float, 3u> tangent{};
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                const std::uint32_t channel = axis + 3u;
                tangent[axis] =
                    corruptedSensors[2u].biasNoiseSigma *
                        sensorGaussianReference(
                            corruptionSeed,
                            environment,
                            episode,
                            poseSensorIdentity,
                            0u,
                            channel,
                            0u
                        ) +
                    corruptedSensors[2u].valueNoiseSigma *
                        sensorGaussianReference(
                            corruptionSeed,
                            environment,
                            episode,
                            poseSensorIdentity,
                            publishedSample,
                            channel,
                            2u
                        );
            }
            const float angle = std::sqrt(
                tangent[0u] * tangent[0u] +
                tangent[1u] * tangent[1u] +
                tangent[2u] * tangent[2u]
            );
            const float halfAngle = 0.5f * angle;
            const float scale = angle > 1.0e-7f
                ? std::sin(halfAngle) / angle
                : 0.5f;
            const std::array<float, 4u> delta{
                tangent[0u] * scale,
                tangent[1u] * scale,
                tangent[2u] * scale,
                std::cos(halfAngle),
            };
            const std::array<float, 4u> cleanOrientation{
                clean.sensorOutputs[poseBase + 3u],
                clean.sensorOutputs[poseBase + 4u],
                clean.sensorOutputs[poseBase + 5u],
                clean.sensorOutputs[poseBase + 6u],
            };
            std::array<float, 4u> expectedOrientation{
                cleanOrientation[3u] * delta[0u] +
                    cleanOrientation[0u] * delta[3u] +
                    cleanOrientation[1u] * delta[2u] -
                    cleanOrientation[2u] * delta[1u],
                cleanOrientation[3u] * delta[1u] -
                    cleanOrientation[0u] * delta[2u] +
                    cleanOrientation[1u] * delta[3u] +
                    cleanOrientation[2u] * delta[0u],
                cleanOrientation[3u] * delta[2u] +
                    cleanOrientation[0u] * delta[1u] -
                    cleanOrientation[1u] * delta[0u] +
                    cleanOrientation[2u] * delta[3u],
                cleanOrientation[3u] * delta[3u] -
                    cleanOrientation[0u] * delta[0u] -
                    cleanOrientation[1u] * delta[1u] -
                    cleanOrientation[2u] * delta[2u],
            };
            const float orientationNorm = std::sqrt(
                expectedOrientation[0u] * expectedOrientation[0u] +
                expectedOrientation[1u] * expectedOrientation[1u] +
                expectedOrientation[2u] * expectedOrientation[2u] +
                expectedOrientation[3u] * expectedOrientation[3u]
            );
            for (std::uint32_t component = 0u;
                 component < 4u;
                 ++component) {
                expectedOrientation[component] /= orientationNorm;
                if (std::abs(
                        noisy.sensorOutputs[
                            poseBase + 3u + component
                        ] - expectedOrientation[component]
                    ) > 1.2e-4f) {
                    fail(
                        "native SensorIR pose orientation noise left tangent-space semantics"
                    );
                }
            }
            const std::uint32_t poseValidity =
                noisy.sensorMetadata[environment * 3u + 2u]
                    .ageValidityAndLayout.y;
            if ((poseValidity &
                 (MR_SENSOR_SAMPLE_VALID |
                  MR_SENSOR_SAMPLE_FRESH)) !=
                    (MR_SENSOR_SAMPLE_VALID |
                     MR_SENSOR_SAMPLE_FRESH)) {
                fail(
                    "native SensorIR pose corruption lost sample validity"
                );
            }
        }
    };
    verifyCorruption(
        cleanCorruptionResult,
        corruptedResult,
        0u,
        1u
    );
    verifyCorruption(
        cleanResetCorruptionResult,
        resetCorruptionResult,
        1u,
        2u
    );
    if (corruptedResult.sensorOutputs !=
            replayedCorruptionResult.sensorOutputs ||
        std::memcmp(
            corruptedResult.sensorMetadata.data(),
            replayedCorruptionResult.sensorMetadata.data(),
            corruptedResult.sensorMetadata.size() *
                sizeof(MRSensorSampleMetadataGPU)
        ) != 0 ||
        std::equal(
            corruptedResult.sensorOutputs.begin(),
            corruptedResult.sensorOutputs.begin() + 6,
            changedSeedResult.sensorOutputs.begin()
        )) {
        fail(
            "native SensorIR corruption did not preserve deterministic replay and seed separation"
        );
    }

    std::vector<metalrobo::SensorSpec> jointStateSensors{{
        .id = "axis_joint_state",
        .parentAssetId = "fixed_base_task_fixture",
        .parentKind = MR_WORLD_SENSOR_PARENT_ASSET,
        .kind = MR_WORLD_SENSOR_JOINT_STATE,
        .target = "axis",
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 1u,
        .consumerFlags =
            MR_WORLD_SENSOR_CONSUMER_ACTOR |
            MR_WORLD_SENSOR_CONSUMER_CRITIC |
            MR_WORLD_SENSOR_CONSUMER_RECORDER,
    }};
    metalrobo::CompiledSensorProgram jointStateProgram;
    const auto jointStateCompile =
        metalrobo::compileSensorProgram(
            jointStateSensors,
            noTactile,
            compiled.world,
            jointStateProgram
        );
    if (!jointStateCompile.succeeded() ||
        !jointStateProgram.valid() ||
        jointStateProgram.layout().outputElementCount != 2u ||
        jointStateProgram.layout().historyElementCount != 2u ||
        jointStateProgram.descriptors().size() != 1u ||
        jointStateProgram.descriptors()[0u].identity.x !=
            MR_WORLD_SENSOR_JOINT_STATE ||
        jointStateProgram.descriptors()[0u].source.x !=
            authored.model.joints[0u].qOffset ||
        jointStateProgram.descriptors()[0u].source.y !=
            authored.model.joints[0u].vOffset ||
        jointStateProgram.descriptors()[0u].source.z != 0u) {
        fail(
            "scalar joint-state SensorIR did not compile to stable q/v indices"
        );
    }
    const std::uint64_t preservedJointFingerprint =
        jointStateProgram.fingerprint();
    std::vector<metalrobo::SensorSpec> unresolvedJointSensors =
        jointStateSensors;
    unresolvedJointSensors[0u].target = "missing_axis";
    const auto unresolvedJointCompile =
        metalrobo::compileSensorProgram(
            unresolvedJointSensors,
            noTactile,
            compiled.world,
            jointStateProgram
        );
    if (unresolvedJointCompile.status !=
            metalrobo::SensorCompileStatus::unresolvedSemantic ||
        jointStateProgram.fingerprint() !=
            preservedJointFingerprint) {
        fail(
            "unresolved joint-state SensorIR did not fail transactionally"
        );
    }

    metalrobo::TaskPack jointSensorTask;
    jointSensorTask.id = "joint_sensor_task";
    jointSensorTask.actions = {{
        .joint = "axis",
        .scale = 0.1f,
    }};
    jointSensorTask.actorFrame = {
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "axis_joint_state",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "axis_joint_state",
            .component = 1u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValidity,
            .target = "axis_joint_state",
            .component = 0u,
        },
    };
    jointSensorTask.criticIncludesCleanHistory = false;
    jointSensorTask.critic = jointSensorTask.actorFrame;
    jointSensorTask.rewards = {{
        .operation = metalrobo::TaskRewardOperator::constant,
        .weight = 1.0f,
    }};
    jointSensorTask.maximumEpisodeSteps = 32u;
    metalrobo::CompiledTaskProgram compiledJointSensorTask;
    const auto jointSensorTaskCompile =
        metalrobo::compileTaskProgram(
            jointSensorTask,
            compiled.world,
            jointStateProgram,
            compiledJointSensorTask
        );
    if (!jointSensorTaskCompile.succeeded() ||
        compiledJointSensorTask.sensorFingerprint() !=
            jointStateProgram.fingerprint() ||
        compiledJointSensorTask.layout().actorObservationSize != 3u ||
        compiledJointSensorTask.layout().criticObservationSize != 3u) {
        fail(
            "TaskIR did not bind the compiled joint-state sensor contract: " +
            std::string{metalrobo::taskCompileStatusName(
                jointSensorTaskCompile.status
            )} + " " + jointSensorTaskCompile.element + " " +
            jointSensorTaskCompile.message + " actor=" +
            std::to_string(
                compiledJointSensorTask.layout()
                    .actorObservationSize
            ) + " critic=" +
            std::to_string(
                compiledJointSensorTask.layout()
                    .criticObservationSize
            )
        );
    }

    constexpr std::size_t jointSensorSteps = 3u;
    std::vector<float> jointInitialQ = initialQ;
    std::vector<float> jointInitialV = initialV;
    for (std::size_t environment = 0u;
         environment < environments;
         ++environment) {
        jointInitialQ[
            environment * compiled.world.nq() +
            authored.model.joints[0u].qOffset
        ] = 0.2f + 0.1f * static_cast<float>(environment);
        jointInitialV[
            environment * compiled.world.nv() +
            authored.model.joints[0u].vOffset
        ] = 0.45f - 0.15f * static_cast<float>(environment);
    }
    std::vector<float> jointEfforts(
        environments * jointSensorSteps * compiled.world.nv(),
        0.0f
    );
    metalrobo::MetalWorldStepConfig jointSensorStep;
    jointSensorStep.timestepSeconds = 0.02f;
    jointSensorStep.physicsSubsteps = 1u;
    jointSensorStep.executionMode =
        metalrobo::MetalWorldExecutionMode::freeMotionABA;
    jointSensorStep.actuationMode =
        metalrobo::MetalWorldActuationMode::effort;
    jointSensorStep.sensorProgram = jointStateProgram;
    jointSensorStep.publishSensorOutputs = true;
    metalrobo::MetalWorldContext jointSensorContext(
        contextConfiguration
    );
    metalrobo::MetalWorldResult jointSensorResult;
    const auto jointSensorExecution = jointSensorContext.run(
        compiled.world,
        {
            .environmentCount = environments,
            .controlStepCount = jointSensorSteps,
            .initialQ = jointInitialQ,
            .initialV = jointInitialV,
            .efforts = jointEfforts,
        },
        jointSensorStep,
        jointSensorResult
    );
    if (!jointSensorExecution.succeeded() ||
        jointSensorResult.sensorOutputs.size() !=
            environments * 2u ||
        jointSensorResult.sensorMetadata.size() != environments ||
        jointSensorContext.stats().pipelineCreationCount != 9u) {
        fail(
            "joint-state SensorIR did not execute on the q/v-only native graph: " +
            jointSensorExecution.message + " q=" +
            std::to_string(jointInitialQ.size()) + "/" +
            std::to_string(
                jointSensorExecution.layout.initialQElements
            ) + " v=" +
            std::to_string(jointInitialV.size()) + "/" +
            std::to_string(
                jointSensorExecution.layout.initialVElements
            ) + " effort=" +
            std::to_string(jointEfforts.size()) + "/" +
            std::to_string(
                jointSensorExecution.layout.effortElements
            ) + " scene=" +
            std::to_string(0u) + "/" +
            std::to_string(
                jointSensorExecution.layout.initialSceneBodyElements
            ) + " kinematic=" +
            std::to_string(
                jointSensorExecution.layout.kinematicTargetElements
            )
        );
    }
    for (std::size_t environment = 0u;
         environment < environments;
         ++environment) {
        const std::size_t outputBase = environment * 2u;
        const float acceptedQ = jointSensorResult.finalQ[
            environment * compiled.world.nq() +
            authored.model.joints[0u].qOffset
        ];
        const float acceptedV = jointSensorResult.finalV[
            environment * compiled.world.nv() +
            authored.model.joints[0u].vOffset
        ];
        const std::uint32_t validity =
            jointSensorResult.sensorMetadata[environment]
                .ageValidityAndLayout.y;
        if (jointSensorResult.sensorOutputs[outputBase] != acceptedQ ||
            jointSensorResult.sensorOutputs[outputBase + 1u] !=
                acceptedV ||
            (validity &
             (MR_SENSOR_SAMPLE_VALID |
              MR_SENSOR_SAMPLE_FRESH)) !=
                (MR_SENSOR_SAMPLE_VALID |
                 MR_SENSOR_SAMPLE_FRESH)) {
            fail(
                "joint-state SensorIR did not publish the accepted generalized state"
            );
        }
    }
    const std::uint64_t taskPipelineCount =
        context.stats().pipelineCreationCount;
    if (taskPipelineCount == 0u ||
        taskPipelineCount >= MR_RUNTIME_PIPELINE_COUNT) {
        fail(
            "compiled task execution did not prune unreachable Metal pipelines"
        );
    }
    for (std::size_t stepIndex = 0u;
         stepIndex < controlSteps;
         ++stepIndex) {
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            const std::size_t observationBase =
                stepIndex * environments * 12u +
                environment * 12u;
            float expected = 0.05f;
            for (std::size_t feature = 0u;
                 feature < 12u;
                 ++feature) {
                expected = std::fma(
                    0.01f * static_cast<float>(feature + 1u),
                    result.actorObservations[
                        observationBase + feature
                    ],
                    expected
                );
            }
            const float actual = result.policyLatents[
                stepIndex * environments + environment
            ];
            if (std::abs(actual - expected) >
                2.0e-5f * (1.0f + std::abs(expected))) {
                fail(
                    "SIMDgroup PolicyIR dense output disagrees with the host dot-product reference"
                );
            }
        }
    }
    {
        constexpr std::uint32_t chunkSteps = 2u;
        metalrobo::MetalRolloutRing rolloutRing({
            .environmentCount =
                static_cast<std::uint32_t>(environments),
            .controlStepCapacity =
                static_cast<std::uint32_t>(controlSteps),
            .actorObservationCount = 12u,
            .criticObservationCount = 8u,
            .actionCount = 1u,
            .slotCount = 1u,
        });
        if (rolloutRing.layout().retainedBytes == 0u) {
            fail("native rollout ring did not report retained storage");
        }
        metalrobo::MetalWorldContext rolloutContext(
            contextConfiguration
        );
        std::vector<float> expectedActor;
        std::vector<float> expectedCritic;
        std::vector<float> expectedLatents;
        std::vector<float> expectedLogProbabilities;
        std::vector<float> expectedValues;
        std::vector<MRTaskTransitionGPU> expectedTransitions;
        std::vector<float> expectedBootstrap;
        const std::vector<std::uint32_t> rolloutResetMasks(
            chunkSteps * environments,
            0u
        );
        {
            auto rollout = rolloutRing.acquire(
                compiled.policy.revision()
            );
            bool rejectedSecondLease = false;
            try {
                auto unavailable = rolloutRing.acquire(
                    compiled.policy.revision()
                );
                (void)unavailable;
            } catch (const std::runtime_error&) {
                rejectedSecondLease = true;
            }
            if (!rejectedSecondLease ||
                rollout.actorObservations() != nullptr) {
                fail(
                    "native rollout ring exposed or reused an unsealed lease"
                );
            }

            for (std::uint32_t chunk = 0u; chunk < 2u; ++chunk) {
                const bool finalChunk = chunk == 1u;
                step.evaluateFinalPolicy = finalChunk;
                const metalrobo::MetalRolloutAppendTarget target =
                    rollout.beginAppend(chunkSteps, finalChunk);
                metalrobo::MetalWorldResult chunkResult;
                const auto chunkExecution = rolloutContext.run(
                    compiled.world,
                    {
                        .environmentCount = environments,
                        .controlStepCount = chunkSteps,
                        .initialQ = initialQ,
                        .initialV = initialV,
                        .resetMasks = rolloutResetMasks,
                        .initialSceneBodies = initialSceneBodies,
                        .rolloutTarget = target,
                    },
                    step,
                    chunkResult
                );
                const std::size_t samples =
                    chunkSteps * environments;
                const auto appendPrefix = [samples](
                    std::vector<float>& destination,
                    const std::vector<float>& source,
                    const std::size_t width
                ) {
                    const std::size_t count = samples * width;
                    if (source.size() < count) {
                        fail(
                            "native rollout source stream is shorter than its compiled layout"
                        );
                    }
                    destination.insert(
                        destination.end(),
                        source.begin(),
                        source.begin() +
                            static_cast<std::ptrdiff_t>(count)
                    );
                };
                if (!chunkExecution.succeeded() ||
                    rollout.writtenControlSteps() !=
                        (chunk + 1u) * chunkSteps) {
                    fail(
                        "direct native rollout append did not commit with world state: " +
                        chunkExecution.message
                    );
                }
                appendPrefix(
                    expectedActor,
                    chunkResult.actorObservations,
                    12u
                );
                appendPrefix(
                    expectedCritic,
                    chunkResult.criticObservations,
                    8u
                );
                appendPrefix(
                    expectedLatents,
                    chunkResult.policyLatents,
                    1u
                );
                appendPrefix(
                    expectedLogProbabilities,
                    chunkResult.policyLogProbabilities,
                    1u
                );
                appendPrefix(
                    expectedValues,
                    chunkResult.policyValues,
                    1u
                );
                expectedTransitions.insert(
                    expectedTransitions.end(),
                    chunkResult.transitions.begin(),
                    chunkResult.transitions.end()
                );
                if (finalChunk) {
                    expectedBootstrap.assign(
                        chunkResult.policyValues.begin() +
                            static_cast<std::ptrdiff_t>(samples),
                        chunkResult.policyValues.end()
                    );
                }
            }
            rollout.seal();
            const auto equalFloats = [](
                const std::vector<float>& expected,
                const float* actual
            ) {
                return actual != nullptr &&
                    std::equal(
                        expected.begin(),
                        expected.end(),
                        actual
                    );
            };
            const MRTaskTransitionGPU* transitions =
                rollout.transitions();
            if (!rollout.sealed() ||
                !equalFloats(
                    expectedActor,
                    rollout.actorObservations()
                ) ||
                !equalFloats(
                    expectedCritic,
                    rollout.criticObservations()
                ) ||
                !equalFloats(
                    expectedLatents,
                    rollout.latents()
                ) ||
                !equalFloats(
                    expectedLogProbabilities,
                    rollout.logProbabilities()
                ) ||
                !equalFloats(
                    expectedValues,
                    rollout.values()
                ) ||
                !equalFloats(
                    expectedBootstrap,
                    rollout.bootstrapValues()
                ) ||
                transitions == nullptr ||
                std::memcmp(
                    expectedTransitions.data(),
                    transitions,
                    expectedTransitions.size() *
                        sizeof(MRTaskTransitionGPU)
                ) != 0) {
                fail(
                    "direct Metal rollout publication changed a compact learning stream"
                );
            }
        }
        {
            auto recycled = rolloutRing.acquire(3u);
            if (!recycled.valid() ||
                recycled.policyRevision() != 3u ||
                recycled.writtenControlSteps() != 0u) {
                fail(
                    "native rollout slot was not reusable after lease release"
                );
            }
        }

        metalrobo::MetalRolloutRing leasedOnlyRing({
            .environmentCount =
                static_cast<std::uint32_t>(environments),
            .controlStepCapacity =
                static_cast<std::uint32_t>(controlSteps),
            .actorObservationCount = 12u,
            .criticObservationCount = 8u,
            .actionCount = 1u,
            .slotCount = 1u,
        });
        auto leasedOnly = leasedOnlyRing.acquire(
            compiled.policy.revision()
        );
        const metalrobo::MetalRolloutAppendTarget leasedTarget =
            leasedOnly.beginAppend(
                static_cast<std::uint32_t>(controlSteps),
                true
            );
        step.evaluateFinalPolicy = true;
        step.publishLearningOutputs = false;
        metalrobo::MetalWorldContext leasedOnlyContext(
            contextConfiguration
        );
        metalrobo::MetalWorldResult leasedOnlyResult;
        const auto leasedOnlyExecution = leasedOnlyContext.run(
            compiled.world,
            {
                .environmentCount = environments,
                .controlStepCount = controlSteps,
                .initialQ = initialQ,
                .initialV = initialV,
                .resetMasks = resetMasks,
                .initialSceneBodies = initialSceneBodies,
                .rolloutTarget = leasedTarget,
            },
            step,
            leasedOnlyResult
        );
        if (!leasedOnlyExecution.succeeded() ||
            leasedOnly.writtenControlSteps() != controlSteps ||
            !leasedOnlyResult.actorObservations.empty() ||
            !leasedOnlyResult.criticObservations.empty() ||
            !leasedOnlyResult.policyLatents.empty() ||
            !leasedOnlyResult.policyLogProbabilities.empty() ||
            !leasedOnlyResult.policyValues.empty() ||
            !leasedOnlyResult.transitions.empty() ||
            leasedOnlyResult.statuses.size() !=
                environments * controlSteps) {
            fail(
                "leased-only rollout publication duplicated compact host vectors: " +
                leasedOnlyExecution.message
            );
        }
        leasedOnly.seal();
        const std::size_t leasedSamples =
            environments * controlSteps;
        const auto equalPrefix = [](
            const std::vector<float>& expected,
            const float* actual,
            const std::size_t count
        ) {
            return actual != nullptr && expected.size() >= count &&
                std::equal(
                    expected.begin(),
                    expected.begin() +
                        static_cast<std::ptrdiff_t>(count),
                    actual
                );
        };
        if (!leasedOnly.sealed() ||
            !equalPrefix(
                result.actorObservations,
                leasedOnly.actorObservations(),
                leasedSamples * 12u
            ) ||
            !equalPrefix(
                result.criticObservations,
                leasedOnly.criticObservations(),
                leasedSamples * 8u
            ) ||
            !equalPrefix(
                result.policyLatents,
                leasedOnly.latents(),
                leasedSamples
            ) ||
            !equalPrefix(
                result.policyLogProbabilities,
                leasedOnly.logProbabilities(),
                leasedSamples
            ) ||
            !equalPrefix(
                result.policyValues,
                leasedOnly.values(),
                leasedSamples
            ) ||
            result.policyValues.size() <
                leasedSamples + environments ||
            leasedOnly.bootstrapValues() == nullptr ||
            !std::equal(
                result.policyValues.begin() +
                    static_cast<std::ptrdiff_t>(leasedSamples),
                result.policyValues.begin() +
                    static_cast<std::ptrdiff_t>(
                        leasedSamples + environments
                    ),
                leasedOnly.bootstrapValues()
            ) ||
            leasedOnly.transitions() == nullptr ||
            std::memcmp(
                result.transitions.data(),
                leasedOnly.transitions(),
                leasedSamples * sizeof(MRTaskTransitionGPU)
            ) != 0) {
            fail(
                "leased-only rollout publication changed the compact learning payload"
            );
        }
        step.publishLearningOutputs = true;
        step.evaluateFinalPolicy = true;
    }
    metalrobo::PolicyPack revisedPolicy = *authored.policy;
    revisedPolicy.revision = 2u;
    std::fill(
        revisedPolicy.layers.front().weights.begin(),
        revisedPolicy.layers.front().weights.end(),
        0.0f
    );
    revisedPolicy.layers.front().bias.front() = 0.25f;
    metalrobo::CompiledPolicyProgram revisedProgram;
    const auto revisedStatus = metalrobo::compilePolicyProgram(
        revisedPolicy,
        compiled.task,
        revisedProgram
    );
    if (!revisedStatus.succeeded() ||
        revisedProgram.topologyFingerprint() !=
            compiled.policy.topologyFingerprint() ||
        revisedProgram.fingerprint() ==
            compiled.policy.fingerprint()) {
        fail(
            "weight-only policy revision changed its native topology contract"
        );
    }
    step.policyProgram = revisedProgram;
    metalrobo::MetalWorldResult revisedResult;
    const auto revisedExecution = context.run(
        compiled.world,
        {
            .environmentCount = environments,
            .controlStepCount = controlSteps,
            .initialQ = initialQ,
            .initialV = initialV,
            .resetMasks = resetMasks,
            .initialSceneBodies = initialSceneBodies,
        },
        step,
        revisedResult
    );
    if (!revisedExecution.succeeded() ||
        revisedResult.policyLatents.size() !=
            environments * (controlSteps + 1u) ||
        std::any_of(
            revisedResult.policyLatents.begin(),
            revisedResult.policyLatents.end(),
            [](const float value) {
                return std::abs(value - 0.25f) > 2.0e-6f;
            }
        )) {
        fail(
            "inactive native policy bank did not publish the revised weights"
        );
    }
    step.policyProgram = compiled.policy;
    metalrobo::MetalWorldResult reusedResult;
    const auto reusedExecution = context.run(
        compiled.world,
        {
            .environmentCount = environments,
            .controlStepCount = controlSteps,
            .initialQ = initialQ,
            .initialV = initialV,
            .resetMasks = resetMasks,
            .initialSceneBodies = initialSceneBodies,
        },
        step,
        reusedResult
    );
    const metalrobo::MetalWorldContextStats policyStats =
        context.stats();
    if (!reusedExecution.succeeded() ||
        policyStats.policyBankUploadCount != 2u ||
        policyStats.policyBankReuseCount != 1u ||
        policyStats.pipelineCreationCount != taskPipelineCount) {
        fail(
            "native policy or pipeline cache did not reuse retained resources"
        );
    }
    for (std::size_t stepIndex = 0u;
         stepIndex < controlSteps;
         ++stepIndex) {
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            const std::size_t observationBase =
                stepIndex * environments * 12u +
                environment * 12u;
            float expected = 0.05f;
            for (std::size_t feature = 0u;
                 feature < 12u;
                 ++feature) {
                expected = std::fma(
                    0.01f * static_cast<float>(feature + 1u),
                    reusedResult.actorObservations[
                        observationBase + feature
                    ],
                    expected
                );
            }
            const float actual = reusedResult.policyLatents[
                stepIndex * environments + environment
            ];
            if (std::abs(actual - expected) >
                2.0e-5f * (1.0f + std::abs(expected))) {
                fail(
                    "reactivated native policy bank lost its retained weights"
                );
            }
        }
    }
    for (std::size_t environment = 0u;
         environment < environments;
         ++environment) {
        const std::size_t outputBase = environment * 14u;
        const std::size_t metadataBase = environment * 2u;
        const auto validPose = [&](const std::size_t base) {
            return
                std::abs(result.sensorOutputs[base + 0u]) <=
                    2.0e-4f &&
                std::abs(result.sensorOutputs[base + 1u]) <=
                    2.0e-4f &&
                std::abs(
                    result.sensorOutputs[base + 2u] - 0.3f
                ) <= 2.0e-4f &&
                std::abs(result.sensorOutputs[base + 3u]) <=
                    2.0e-4f &&
                std::abs(result.sensorOutputs[base + 4u]) <=
                    2.0e-4f &&
                std::abs(result.sensorOutputs[base + 5u]) <=
                    2.0e-4f &&
                std::abs(
                    result.sensorOutputs[base + 6u] - 1.0f
                ) <= 2.0e-4f;
        };
        const MRSensorSampleMetadataGPU& delayed =
            result.sensorMetadata[metadataBase];
        const std::uint64_t delayedSequence =
            static_cast<std::uint64_t>(
                delayed.sequenceAndTimestamp.x
            ) |
            (static_cast<std::uint64_t>(
                 delayed.sequenceAndTimestamp.y
             ) << 32u);
        const std::uint64_t expectedDelayedSequence =
            environment == 0u ? 3u : controlSteps + 1u;
        const MRSensorSampleMetadataGPU& nonDivisor =
            result.sensorMetadata[metadataBase + 1u];
        const std::uint64_t nonDivisorSequence =
            static_cast<std::uint64_t>(
                nonDivisor.sequenceAndTimestamp.x
            ) |
            (static_cast<std::uint64_t>(
                 nonDivisor.sequenceAndTimestamp.y
             ) << 32u);
        const std::uint64_t expectedNonDivisorSequence =
            environment == 0u ? 2u : 3u;
        if (!validPose(outputBase) ||
            delayedSequence != expectedDelayedSequence ||
            delayed.ageValidityAndLayout.x != 0u ||
            (delayed.ageValidityAndLayout.y &
             (MR_SENSOR_SAMPLE_VALID |
              MR_SENSOR_SAMPLE_FRESH)) !=
                (MR_SENSOR_SAMPLE_VALID |
                 MR_SENSOR_SAMPLE_FRESH) ||
            delayed.ageValidityAndLayout.z != 0u ||
            delayed.ageValidityAndLayout.w != 7u ||
            !validPose(outputBase + 7u) ||
            nonDivisorSequence != expectedNonDivisorSequence ||
            nonDivisor.ageValidityAndLayout.x != 0u ||
            (nonDivisor.ageValidityAndLayout.y &
             (MR_SENSOR_SAMPLE_VALID |
              MR_SENSOR_SAMPLE_FRESH)) !=
                (MR_SENSOR_SAMPLE_VALID |
                 MR_SENSOR_SAMPLE_FRESH) ||
            nonDivisor.ageValidityAndLayout.z != 7u ||
            nonDivisor.ageValidityAndLayout.w != 7u) {
            fail(
                "native SensorIR did not preserve pose, latency, or reset scheduling"
            );
        }
    }
    for (const std::size_t sample : {
             std::size_t{0u},
             std::size_t{2u * environments},
         }) {
        const std::size_t base = sample * 12u;
        if (std::abs(result.actorObservations[base + 5u] - 0.3f) >
                2.0e-4f ||
            std::abs(result.actorObservations[base + 6u]) > 2.0e-4f ||
            std::abs(result.actorObservations[base + 7u]) > 2.0e-4f ||
            std::abs(result.actorObservations[base + 8u]) > 2.0e-4f) {
            fail(
                "reset frame observations were not refreshed from reset kinematics"
            );
        }
    }
    for (std::size_t stepIndex = 0u;
         stepIndex < controlSteps;
         ++stepIndex) {
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            const std::size_t sample =
                stepIndex * environments + environment;
            const std::size_t actorBase = sample * 12u;
            const bool reset =
                stepIndex == 0u ||
                (stepIndex == 2u && environment == 0u);
            const float delayedValue = reset ? 0.0f : 0.3f;
            const float delayedValid = reset ? 0.0f : 1.0f;
            const bool thirtyHertzFresh =
                stepIndex == 0u || stepIndex == 2u;
            if (std::abs(
                    result.actorObservations[actorBase + 9u] -
                    delayedValue
                ) > 2.0e-4f ||
                std::abs(
                    result.actorObservations[actorBase + 10u] -
                    delayedValid
                ) > 2.0e-4f ||
                std::abs(
                    result.actorObservations[actorBase + 11u] -
                    (thirtyHertzFresh ? 1.0f : 0.0f)
                ) > 2.0e-4f) {
                fail(
                    "TaskIR actor observations did not consume the scheduled SensorIR boundary"
                );
            }
            const std::size_t criticBase = sample * 8u;
            if (std::abs(
                    result.criticObservations[criticBase + 4u] -
                    0.3f
                ) > 2.0e-4f ||
                std::abs(
                    result.criticObservations[criticBase + 5u] -
                    (thirtyHertzFresh ? 0.0f : 1.0f)
                ) > 2.0e-4f ||
                std::abs(
                    result.criticObservations[criticBase + 6u] -
                    0.4f
                ) > 2.0e-4f ||
                std::abs(
                    result.criticObservations[criticBase + 7u] -
                    0.25f
                ) > 2.0e-4f) {
                fail(
                    "TaskIR critic observations did not consume SensorIR or scene-frame state"
                );
            }
        }
    }
    for (std::size_t environment = 0u;
         environment < environments;
         ++environment) {
        const std::size_t finalActorBase =
            (controlSteps * environments + environment) * 12u;
        if (std::abs(
                result.actorObservations[finalActorBase + 11u] - 1.0f
            ) > 2.0e-4f) {
            fail(
                "terminal actor observation was published before the accepted-state sensor sample"
            );
        }
    }
    if (std::any_of(
            result.transitions.begin(),
            result.transitions.end(),
            [](const MRTaskTransitionGPU& transition) {
                return !std::isfinite(
                           transition.rewardAndState.x
                       ) ||
                    transition.termination.x != 0u;
            }
        )) {
        fail("frame reward or termination execution is invalid");
    }

    TemporaryPackFiles packFiles;
    const auto taskWrite = metalrobo::writeTaskPack(
        authored.task,
        packFiles.task
    );
    metalrobo::TaskPack restored;
    const auto taskRead = metalrobo::readTaskPack(
        packFiles.task,
        restored
    );
    metalrobo::CompiledTaskProgram roundTrip;
    const auto roundTripStatus = metalrobo::compileTaskProgram(
        restored,
        compiled.world,
        compiled.sensors,
        roundTrip
    );
    if (!taskWrite.succeeded() ||
        !taskRead.succeeded() ||
        !roundTripStatus.succeeded() ||
        roundTrip.fingerprint() != compiled.task.fingerprint()) {
        fail("typed TaskPack round trip changed frame or goal semantics");
    }

    metalrobo::TaskPack missingReference = authored.task;
    missingReference.critic.back().reference =
        "missing_scene_reference";
    metalrobo::CompiledTaskProgram preservedReferenceTask =
        compiled.task;
    const std::uint64_t referenceFingerprint =
        preservedReferenceTask.fingerprint();
    const auto missingReferenceStatus =
        metalrobo::compileTaskProgram(
            missingReference,
            compiled.world,
            compiled.sensors,
            preservedReferenceTask
        );
    if (missingReferenceStatus.status !=
            metalrobo::TaskCompileStatus::unresolvedSemantic ||
        preservedReferenceTask.fingerprint() !=
            referenceFingerprint) {
        fail(
            "unresolved frame-to-frame reference was not transactionally rejected"
        );
    }

    metalrobo::TaskPack unauthorized = authored.task;
    unauthorized.critic.push_back({
        .source = metalrobo::TaskObservationSource::sensorValue,
        .target = "tool_pose_delayed",
        .component = 2u,
    });
    metalrobo::CompiledTaskProgram preservedSensorTask = compiled.task;
    const std::uint64_t sensorTaskFingerprint =
        preservedSensorTask.fingerprint();
    const auto unauthorizedStatus = metalrobo::compileTaskProgram(
        unauthorized,
        compiled.world,
        compiled.sensors,
        preservedSensorTask
    );
    if (unauthorizedStatus.status !=
            metalrobo::TaskCompileStatus::invalidPack ||
        preservedSensorTask.fingerprint() != sensorTaskFingerprint) {
        fail(
            "SensorIR consumer permission was not transactionally enforced"
        );
    }

    metalrobo::TaskPack invalid = authored.task;
    invalid.actorFrame.push_back({
        .source = metalrobo::TaskObservationSource::rootHeight,
    });
    metalrobo::CompiledTaskProgram preserved = compiled.task;
    const std::uint64_t fingerprint = preserved.fingerprint();
    const auto rejected = metalrobo::compileTaskProgram(
        invalid,
        compiled.world,
        compiled.sensors,
        preserved
    );
    if (rejected.status !=
            metalrobo::TaskCompileStatus::unsupportedOperator ||
        preserved.fingerprint() != fingerprint) {
        fail(
            "fixed-base floating-root operator was not transactionally rejected"
        );
    }

    metalrobo::SimulationDescription twistAuthored = authored;
    twistAuthored.task.id = "fixed_base_frame_twist";
    const auto twistReferenceBody = std::find(
        twistAuthored.model.bodyNames.begin(),
        twistAuthored.model.bodyNames.end(),
        "locomotion_ground"
    );
    if (twistReferenceBody ==
            twistAuthored.model.bodyNames.end() ||
        twistAuthored.sceneBodies.size() != 1u) {
        fail("frame-twist moving reference body did not resolve");
    }
    const auto twistReferenceBodyIndex =
        static_cast<std::size_t>(
            twistReferenceBody -
            twistAuthored.model.bodyNames.begin()
        );
    twistAuthored.model.bodies[twistReferenceBodyIndex]
        .motionType = MR_MOTION_KINEMATIC;
    twistAuthored.sceneBodies[0u].flagsAndIndices[0] =
        MR_MOTION_KINEMATIC;
    twistAuthored.sceneBodies[0u]
        .linearVelocityAndInverseMass.x = 0.02f;
    twistAuthored.task.actorFrame = {{
        .source = metalrobo::TaskObservationSource::jointVelocity,
        .target = "axis",
    }};
    twistAuthored.task.critic = {
        {
            .source = metalrobo::TaskObservationSource::
                frameLinearVelocityWorld,
            .target = "tool_tip",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameAngularVelocityWorld,
            .target = "tool_tip",
            .component = 1u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameRelativeLinearVelocity,
            .target = "tool_tip",
            .reference = "world_anchor",
            .component = 1u,
        },
        {
            .source = metalrobo::TaskObservationSource::
                frameRelativeAngularVelocity,
            .target = "tool_tip",
            .reference = "world_anchor",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "tool_twist",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "tool_twist",
            .component = 4u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValidity,
            .target = "tool_twist",
            .component = 0u,
        },
        {
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "reference_twist",
            .component = 0u,
        },
    };
    twistAuthored.task.criticIncludesCleanHistory = false;
    twistAuthored.task.rewards = {{
        .operation = metalrobo::TaskRewardOperator::constant,
        .weight = 1.0f,
    }};
    twistAuthored.task.terminations.clear();
    twistAuthored.task.randomization = {{
        .operation =
            metalrobo::TaskRandomizationOperator::actionVelocity,
        .parameters = {0.5f, 0.5f, 0.0f, 0.0f},
    }};
    twistAuthored.sensors = {
        {
            .id = "tool_twist",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_FRAME_TWIST_WORLD,
            .localPose = {
                .position = {0.0f, 0.0f, 0.2f, 0.0f},
            },
            .nominalRateHz = 50.0f,
            .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_CRITIC |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
        {
            .id = "reference_twist",
            .parentKind = MR_WORLD_SENSOR_PARENT_RIGID_BODY,
            .parentBodyIndex = static_cast<std::uint32_t>(
                twistReferenceBodyIndex
            ),
            .kind = MR_WORLD_SENSOR_FRAME_TWIST_WORLD,
            .nominalRateHz = 50.0f,
            .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_CRITIC |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
    };
    metalrobo::PolicyPack twistPolicy;
    twistPolicy.id = "fixed_base_frame_twist_policy";
    twistPolicy.layers = {{
        .inputCount = 1u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(1u, 0.0f),
        .bias = std::vector<float>(1u, 0.0f),
    }};
    twistPolicy.criticLayers = {{
        .inputCount = 8u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(8u, 0.0f),
        .bias = std::vector<float>(1u, 0.0f),
    }};
    twistAuthored.policy = std::move(twistPolicy);

    metalrobo::CompiledSimulation twistCompiled;
    const auto twistCompile = metalrobo::compileSimulation(
        twistAuthored,
        0u,
        twistCompiled
    );
    if (!twistCompile.succeeded()) {
        fail(
            "frame-twist task compile failed: " +
            twistCompile.task.message
        );
    }
    constexpr std::size_t twistSteps = 2u;
    const std::vector<std::uint32_t> twistResetMasks(
        twistSteps,
        0u
    );
    const std::vector<MRBodyStateGPU> twistKinematicTargets(
        twistSteps,
        twistAuthored.sceneBodies[0u]
    );
    metalrobo::MetalWorldStepConfig twistStep = step;
    twistStep.taskProgram = twistCompiled.task;
    twistStep.sensorProgram = twistCompiled.sensors;
    twistStep.policyProgram = twistCompiled.policy;
    twistStep.publishSensorOutputs = true;
    twistStep.evaluateFinalPolicy = true;
    metalrobo::MetalWorldContext twistContext(
        contextConfiguration
    );
    metalrobo::MetalWorldResult twistResult;
    const auto twistExecution = twistContext.run(
        twistCompiled.world,
        {
            .environmentCount = 1u,
            .controlStepCount = twistSteps,
            .initialQ = std::span{
                authored.model.defaultQ.data(),
                authored.model.defaultQ.size(),
            },
            .initialV = std::span{
                initialV.data(),
                compiled.world.nv(),
            },
            .resetMasks = twistResetMasks,
            .initialSceneBodies = twistAuthored.sceneBodies,
            .kinematicTargets = twistKinematicTargets,
        },
        twistStep,
        twistResult
    );
    if (!twistExecution.succeeded() ||
        twistResult.actorObservations.size() != twistSteps + 1u ||
        twistResult.criticObservations.size() !=
            (twistSteps + 1u) * 8u ||
        twistResult.sensorOutputs.size() != 12u ||
        std::abs(twistResult.actorObservations[0u] - 0.5f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[0u] - 0.1f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[1u] - 0.5f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[2u] + 0.1f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[3u] - 0.5f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[4u] - 0.1f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[5u] - 0.5f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[6u] - 1.0f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[7u]) >
            2.0e-4f) {
        fail(
            "frame twist did not materialize from the randomized reset state: " +
            twistExecution.message +
            " actor=" +
            (twistResult.actorObservations.empty()
                 ? std::string{"missing"}
                 : std::to_string(twistResult.actorObservations[0u])) +
            " critic0=" +
            (twistResult.criticObservations.empty()
                 ? std::string{"missing"}
                 : std::to_string(twistResult.criticObservations[0u])) +
            " critic1=" +
            (twistResult.criticObservations.size() < 2u
                 ? std::string{"missing"}
                 : std::to_string(twistResult.criticObservations[1u])) +
            " critic2=" +
            (twistResult.criticObservations.size() < 3u
                 ? std::string{"missing"}
                 : std::to_string(twistResult.criticObservations[2u])) +
            " critic3=" +
            (twistResult.criticObservations.size() < 4u
                 ? std::string{"missing"}
                 : std::to_string(twistResult.criticObservations[3u]))
        );
    }
    for (std::size_t sample = 0u;
         sample < twistSteps + 1u;
         ++sample) {
        const float jointVelocity =
            twistResult.actorObservations[sample];
        const float frameLinear =
            twistResult.criticObservations[8u * sample];
        const float frameAngular =
            twistResult.criticObservations[8u * sample + 1u];
        const float relativeLinear =
            twistResult.criticObservations[8u * sample + 2u];
        const float relativeAngular =
            twistResult.criticObservations[8u * sample + 3u];
        const float sensorLinear =
            twistResult.criticObservations[8u * sample + 4u];
        const float sensorAngular =
            twistResult.criticObservations[8u * sample + 5u];
        const float sensorValid =
            twistResult.criticObservations[8u * sample + 6u];
        const float referenceSensorLinear =
            twistResult.criticObservations[8u * sample + 7u];
        const float referenceLinear =
            sample == 0u ? 0.0f : 0.02f;
        if (!std::isfinite(frameLinear) ||
            std::abs(frameAngular - jointVelocity) > 2.0e-4f ||
            std::abs(
                relativeLinear + frameLinear - referenceLinear
            ) >
                2.0e-4f ||
            std::abs(relativeAngular - frameAngular) > 2.0e-4f ||
            std::abs(sensorLinear - frameLinear) > 2.0e-4f ||
            std::abs(sensorAngular - frameAngular) > 2.0e-4f ||
            sensorValid != 1.0f ||
            std::abs(referenceSensorLinear - referenceLinear) >
                2.0e-4f) {
            fail(
                "accepted world/relative frame twist disagrees with generalized velocity"
            );
        }
    }
    const std::size_t finalTwistBase = 8u * twistSteps;
    if (std::abs(
            twistResult.sensorOutputs[0u] -
            twistResult.criticObservations[finalTwistBase + 4u]
        ) > 2.0e-4f ||
        std::abs(
            twistResult.sensorOutputs[4u] -
            twistResult.criticObservations[finalTwistBase + 5u]
        ) > 2.0e-4f ||
        std::abs(
            twistResult.sensorOutputs[6u] -
            twistResult.criticObservations[finalTwistBase + 7u]
        ) > 2.0e-4f) {
        fail(
            "published SensorIR twist disagrees with the final TaskIR view"
        );
    }

    // IMU is an accepted-state instrument, not a locomotion-only tensor
    // reduction. The native schedule differentiates point velocity over the
    // actual sample interval, subtracts gravity, and rotates specific force
    // and angular velocity into the authored sensor frame. A moving
    // kinematic rigid body is an independent exact oracle for that contract.
    metalrobo::SimulationDescription imuAuthored = twistAuthored;
    imuAuthored.task.id = "fixed_base_native_imu";
    imuAuthored.sceneBodies[0u]
        .linearVelocityAndInverseMass.x = 0.0f;
    imuAuthored.sceneBodies[0u].angularVelocity = {
        0.0f, 0.0f, 0.25f, 0.0f,
    };
    imuAuthored.task.critic.clear();
    for (std::uint32_t component = 0u;
         component < 6u;
         ++component) {
        imuAuthored.task.critic.push_back({
            .source =
                metalrobo::TaskObservationSource::sensorValue,
            .target = "reference_imu",
            .component = component,
        });
    }
    imuAuthored.task.critic.push_back({
        .source = metalrobo::TaskObservationSource::sensorValidity,
        .target = "reference_imu",
        .component = 0u,
    });
    imuAuthored.sensors = {{
        .id = "reference_imu",
        .parentKind = MR_WORLD_SENSOR_PARENT_RIGID_BODY,
        .parentBodyIndex = static_cast<std::uint32_t>(
            twistReferenceBodyIndex
        ),
        .kind = MR_WORLD_SENSOR_IMU,
        .localPose = {
            .orientation = {
                0.0f,
                0.0f,
                0.7071067811865476f,
                0.7071067811865476f,
            },
        },
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 1u,
        .consumerFlags =
            MR_WORLD_SENSOR_CONSUMER_CRITIC |
            MR_WORLD_SENSOR_CONSUMER_RECORDER,
    }};
    metalrobo::PolicyPack imuPolicy;
    imuPolicy.id = "fixed_base_native_imu_policy";
    imuPolicy.layers = {{
        .inputCount = 1u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(1u, 0.0f),
        .bias = std::vector<float>(1u, 0.0f),
    }};
    imuPolicy.criticLayers = {{
        .inputCount = 7u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(7u, 0.0f),
        .bias = std::vector<float>(1u, 0.0f),
    }};
    imuAuthored.policy = std::move(imuPolicy);

    metalrobo::CompiledSimulation imuCompiled;
    const auto imuCompile = metalrobo::compileSimulation(
        imuAuthored,
        0u,
        imuCompiled
    );
    if (!imuCompile.succeeded() ||
        imuCompiled.sensors.layout().sensorCount != 1u ||
        imuCompiled.sensors.layout().outputElementCount != 6u ||
        imuCompiled.task.layout().criticObservationSize != 7u) {
        fail(
            "native IMU task compile failed: " +
            imuCompile.task.message
        );
    }
    constexpr std::size_t imuSteps = 2u;
    const std::vector<std::uint32_t> imuResetMasks(imuSteps, 0u);
    std::vector<MRBodyStateGPU> imuTargets(
        imuSteps,
        imuAuthored.sceneBodies[0u]
    );
    imuTargets[1u].linearVelocityAndInverseMass.x = 0.2f;
    metalrobo::MetalWorldStepConfig imuStep = step;
    imuStep.taskProgram = imuCompiled.task;
    imuStep.sensorProgram = imuCompiled.sensors;
    imuStep.policyProgram = imuCompiled.policy;
    imuStep.publishSensorOutputs = true;
    imuStep.evaluateFinalPolicy = true;
    metalrobo::MetalWorldContext imuContext(contextConfiguration);
    metalrobo::MetalWorldResult imuResult;
    const auto imuExecution = imuContext.run(
        imuCompiled.world,
        {
            .environmentCount = 1u,
            .controlStepCount = imuSteps,
            .initialQ = std::span{
                imuAuthored.model.defaultQ.data(),
                imuAuthored.model.defaultQ.size(),
            },
            .initialV = std::span{
                initialV.data(),
                imuCompiled.world.nv(),
            },
            .resetMasks = imuResetMasks,
            .initialSceneBodies = imuAuthored.sceneBodies,
            .kinematicTargets = imuTargets,
        },
        imuStep,
        imuResult
    );
    const float expectedGravity =
        -imuAuthored.model.world.gravityAndTimestep.z;
    const float expectedAcceleration =
        0.2f / imuStep.timestepSeconds;
    const float expectedSensorAngle =
        0.25f * imuStep.timestepSeconds;
    const auto validImuSample = [&] (
        const std::size_t base,
        const float expectedLocalX,
        const float expectedLocalY,
        const float expectedAngularZ
    ) {
        return
            std::abs(
                imuResult.criticObservations[base + 0u] -
                expectedLocalX
            ) < 2.0e-3f &&
            std::abs(
                imuResult.criticObservations[base + 1u] -
                expectedLocalY
            ) < 2.0e-3f &&
            std::abs(
                imuResult.criticObservations[base + 2u] -
                expectedGravity
            ) < 2.0e-3f &&
            std::abs(imuResult.criticObservations[base + 3u]) <
                2.0e-4f &&
            std::abs(imuResult.criticObservations[base + 4u]) <
                2.0e-4f &&
            std::abs(
                imuResult.criticObservations[base + 5u] -
                expectedAngularZ
            ) < 2.0e-4f &&
            imuResult.criticObservations[base + 6u] == 1.0f;
    };
    if (!imuExecution.succeeded() ||
        imuResult.criticObservations.size() !=
            (imuSteps + 1u) * 7u ||
        imuResult.sensorOutputs.size() != 6u ||
        imuResult.sensorMetadata.size() != 1u ||
        !validImuSample(0u, 0.0f, 0.0f, 0.0f) ||
        !validImuSample(7u, 0.0f, 0.0f, 0.25f) ||
        !validImuSample(
            14u,
            -expectedAcceleration * std::sin(expectedSensorAngle),
            -expectedAcceleration * std::cos(expectedSensorAngle),
            0.25f
        )) {
        fail(
            "native IMU did not preserve specific-force, frame, or scheduling semantics: " +
            imuExecution.message + " initial=" +
            (imuResult.criticObservations.size() < 7u
                 ? std::string{"missing"}
                 : std::to_string(
                       imuResult.criticObservations[0u]
                   ) + "," +
                       std::to_string(
                           imuResult.criticObservations[1u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[2u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[5u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[6u]
                       )) +
            " middle=" +
            (imuResult.criticObservations.size() < 14u
                 ? std::string{"missing"}
                 : std::to_string(
                       imuResult.criticObservations[7u]
                   ) + "," +
                       std::to_string(
                           imuResult.criticObservations[8u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[9u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[12u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[13u]
                       )) +
            " final=" +
            (imuResult.criticObservations.size() < 21u
                 ? std::string{"missing"}
                 : std::to_string(
                       imuResult.criticObservations[14u]
                   ) + "," +
                       std::to_string(
                           imuResult.criticObservations[15u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[16u]
                       ) + "," +
                       std::to_string(
                           imuResult.criticObservations[19u]
                       ))
        );
    }
    for (std::size_t channel = 0u;
         channel < 6u;
         ++channel) {
        if (std::abs(
                imuResult.sensorOutputs[channel] -
                imuResult.criticObservations[14u + channel]
            ) > 2.0e-4f) {
            fail(
                "published IMU disagrees with the final TaskIR view"
            );
        }
    }

    // A six-axis SensorIR force/torque sample must consume the exact solved
    // contact impulses, not infer a wrench from acceleration or a second
    // collision query. The independent TaskIR contact-group reduction is the
    // parity oracle for the same body, reference point, and local axes.
    metalrobo::SimulationDescription wrenchAuthored = authored;
    wrenchAuthored.model.defaultQ[0u] = 0.3f;
    wrenchAuthored.model.shapes[0u].collisionMask = 0u;
    const std::uint32_t obstacleBodyIndex =
        static_cast<std::uint32_t>(
            wrenchAuthored.model.bodies.size()
        );
    MRBodyPropertiesGPU obstacleBody{};
    obstacleBody.articulationIndex = MR_INVALID_INDEX;
    obstacleBody.parentBody = MR_INVALID_INDEX;
    obstacleBody.inboundJoint = MR_INVALID_INDEX;
    obstacleBody.motionType = MR_MOTION_STATIC;
    obstacleBody.dampingAndSpeedLimits = {
        0.0f, 0.0f, 1.0e6f, 1.0e6f,
    };
    wrenchAuthored.model.bodies.push_back(obstacleBody);
    wrenchAuthored.model.bodyNames.emplace_back("wrench_obstacle");
    MRMaterialGPU obstacleMaterial =
        wrenchAuthored.model.materials.front();
    obstacleMaterial.response.z = 1.0e-4f;
    const std::uint32_t obstacleMaterialIndex =
        static_cast<std::uint32_t>(
            wrenchAuthored.model.materials.size()
        );
    wrenchAuthored.model.materials.push_back(obstacleMaterial);
    MRShapeGPU obstacleShape{};
    obstacleShape.bodyIndex = obstacleBodyIndex;
    obstacleShape.shapeType = MR_SHAPE_SPHERE;
    obstacleShape.materialIndex = obstacleMaterialIndex;
    obstacleShape.collisionGroup = 1u;
    obstacleShape.collisionMask = ~0u;
    obstacleShape.slotGeneration = 1u;
    obstacleShape.localRotation.w = 1.0f;
    obstacleShape.dimensions.x = 0.06f;
    obstacleShape.contactRestAndBoundingRadius = {
        0.002f, 0.0f, 0.06f, 0.0f,
    };
    wrenchAuthored.model.shapes.push_back(obstacleShape);
    wrenchAuthored.model.shapeNames.emplace_back("wrench_obstacle");
    wrenchAuthored.model.world.bodyCount =
        static_cast<std::uint32_t>(wrenchAuthored.model.bodies.size());
    wrenchAuthored.model.world.shapeCount =
        static_cast<std::uint32_t>(wrenchAuthored.model.shapes.size());
    wrenchAuthored.model.world.materialCount =
        static_cast<std::uint32_t>(
            wrenchAuthored.model.materials.size()
        );
    MRBodyStateGPU obstacleState{};
    obstacleState.position = {0.06f, 0.0f, 0.026f, 1.0f};
    obstacleState.orientation.w = 1.0f;
    obstacleState.flagsAndIndices[0] = MR_MOTION_STATIC;
    obstacleState.flagsAndIndices[1] = MR_INVALID_INDEX;
    obstacleState.flagsAndIndices[2] = obstacleBodyIndex;
    wrenchAuthored.sceneBodies.push_back(obstacleState);
    wrenchAuthored.task.id = "fixed_base_force_torque";
    wrenchAuthored.task.actorFrame = {{
        .source = metalrobo::TaskObservationSource::jointVelocity,
        .target = "axis",
    }};
    wrenchAuthored.task.actorHistoryLength = 1u;
    wrenchAuthored.task.critic.clear();
    for (std::uint32_t component = 0u; component < 6u; ++component) {
        wrenchAuthored.task.critic.push_back({
            .source = metalrobo::TaskObservationSource::contactWrenchLocal,
            .target = "tool_contact",
            .component = component,
        });
    }
    for (std::uint32_t component = 0u; component < 6u; ++component) {
        wrenchAuthored.task.critic.push_back({
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "tool_force_torque",
            .component = component,
        });
    }
    for (std::uint32_t component = 0u; component < 5u; ++component) {
        wrenchAuthored.task.critic.push_back({
            .source = metalrobo::TaskObservationSource::sensorValue,
            .target = "tool_contact_state",
            .component = component,
        });
    }
    wrenchAuthored.task.critic.push_back({
        .source = metalrobo::TaskObservationSource::sensorValidity,
        .target = "tool_contact_state",
        .component = 0u,
    });
    wrenchAuthored.task.critic.push_back({
        .source = metalrobo::TaskObservationSource::sensorValidity,
        .target = "tool_ground_contact_state",
        .component = 5u,
    });
    wrenchAuthored.task.criticIncludesCleanHistory = false;
    wrenchAuthored.task.contactGroups = {{
        .id = "tool_contact",
        .bodies = {"tool"},
        .referenceBody = "tool",
    }};
    wrenchAuthored.task.rewards = {{
        .operation = metalrobo::TaskRewardOperator::constant,
        .weight = 1.0f,
    }};
    wrenchAuthored.task.terminations.clear();
    wrenchAuthored.sensors = {
        {
            .id = "tool_force_torque",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_FORCE_TORQUE,
            // The URDF link-frame value 0.1 m is this body's COM. TaskIR's
            // contact-group reference therefore remains zero in COM coordinates.
            .localPose = {
                .position = {0.0f, 0.0f, 0.1f, 0.0f},
            },
            .nominalRateHz = 50.0f,
            .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_CRITIC |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
        {
            .id = "tool_contact_state",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_CONTACT_STATE,
            .filterBodies = {"wrench_obstacle"},
            .nominalRateHz = 50.0f,
            .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_CRITIC |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
        },
        {
            .id = "tool_ground_contact_state",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_CONTACT_STATE,
            .filterBodies = {"locomotion_ground"},
            .nominalRateHz = 50.0f,
            .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
            .historyLength = 1u,
            .consumerFlags =
                MR_WORLD_SENSOR_CONSUMER_CRITIC |
                MR_WORLD_SENSOR_CONSUMER_RECORDER,
            .dropoutProbability = 1.0f,
        },
    };
    metalrobo::PolicyPack wrenchPolicy;
    wrenchPolicy.id = "fixed_base_force_torque_policy";
    wrenchPolicy.layers = {{
        .inputCount = 1u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(1u, 0.0f),
        .bias = std::vector<float>(1u, 0.0f),
    }};
    wrenchPolicy.criticLayers = {{
        .inputCount = 19u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(19u, 0.0f),
        .bias = std::vector<float>(1u, 0.0f),
    }};
    wrenchAuthored.policy = std::move(wrenchPolicy);

    metalrobo::CompiledSimulation wrenchCompiled;
    const auto wrenchCompile = metalrobo::compileSimulation(
        wrenchAuthored,
        0u,
        wrenchCompiled
    );
    if (!wrenchCompile.succeeded() ||
        wrenchCompiled.sensors.layout().sensorCount != 3u ||
        wrenchCompiled.sensors.layout().outputElementCount != 16u ||
        wrenchCompiled.sensors.layout().filterBodyCount != 2u ||
        wrenchCompiled.sensors.filterBodies().size() != 2u ||
        wrenchCompiled.sensors.filterBodies()[0u] !=
            obstacleBodyIndex ||
        wrenchCompiled.sensors.filterBodies()[1u] !=
            twistReferenceBodyIndex ||
        wrenchCompiled.sensors.descriptors()[1u].filter.x != 0u ||
        wrenchCompiled.sensors.descriptors()[1u].filter.y != 1u ||
        wrenchCompiled.sensors.descriptors()[2u].filter.x != 1u ||
        wrenchCompiled.sensors.descriptors()[2u].filter.y != 1u) {
        fail(
            "force-torque SensorIR compile failed: " +
            wrenchCompile.world.message + " " +
            wrenchCompile.sensors.message + " " +
            wrenchCompile.task.message + " " +
            wrenchCompile.policy.message
        );
    }
    std::vector<metalrobo::SensorSpec> unresolvedFilterSensors =
        wrenchAuthored.sensors;
    unresolvedFilterSensors[1u].filterBodies = {
        "missing_contact_filter_body",
    };
    metalrobo::CompiledSensorProgram preservedContactSensors =
        wrenchCompiled.sensors;
    const std::uint64_t preservedContactFingerprint =
        preservedContactSensors.fingerprint();
    const auto unresolvedFilter = metalrobo::compileSensorProgram(
        unresolvedFilterSensors,
        wrenchAuthored.tactileSystem,
        wrenchCompiled.world,
        preservedContactSensors
    );
    if (unresolvedFilter.status !=
            metalrobo::SensorCompileStatus::unresolvedSemantic ||
        preservedContactSensors.fingerprint() !=
            preservedContactFingerprint) {
        fail(
            "unresolved contact-state body filter was not rejected transactionally"
        );
    }
    constexpr std::size_t wrenchSteps = 4u;
    metalrobo::MetalWorldStepConfig wrenchStep = step;
    wrenchStep.taskProgram = wrenchCompiled.task;
    wrenchStep.sensorProgram = wrenchCompiled.sensors;
    wrenchStep.policyProgram = wrenchCompiled.policy;
    wrenchStep.publishSensorOutputs = true;
    wrenchStep.evaluateFinalPolicy = true;
    wrenchStep.captureContactEvidence = true;
    std::vector<std::uint32_t> wrenchResetMasks(
        wrenchSteps,
        0u
    );
    wrenchResetMasks[2u] = 1u;
    metalrobo::MetalWorldContext wrenchContext(contextConfiguration);
    metalrobo::MetalWorldResult wrenchResult;
    const auto wrenchExecution = wrenchContext.run(
        wrenchCompiled.world,
        {
            .environmentCount = 1u,
            .controlStepCount = wrenchSteps,
            .initialQ = std::span{
                wrenchAuthored.model.defaultQ.data(),
                wrenchAuthored.model.defaultQ.size(),
            },
            .initialV = std::span{
                initialV.data(),
                wrenchCompiled.world.nv(),
            },
            .resetMasks = wrenchResetMasks,
            .initialSceneBodies = wrenchAuthored.sceneBodies,
        },
        wrenchStep,
        wrenchResult
    );
    const std::size_t finalWrenchBase = 19u * wrenchSteps;
    double forceNormSquared = 0.0;
    double torqueNormSquared = 0.0;
    bool wrenchParity =
        wrenchExecution.succeeded() &&
        wrenchResult.sensorOutputs.size() == 16u &&
        wrenchResult.criticObservations.size() ==
            19u * (wrenchSteps + 1u);
    if (wrenchParity) {
        const std::size_t resetObservationBase = 19u * 2u;
        for (std::size_t component = 0u;
             component < 6u;
             ++component) {
            wrenchParity = wrenchParity &&
                std::abs(
                    wrenchResult.criticObservations[
                        resetObservationBase + 6u + component
                    ]
                ) <= 1.0e-6f;
        }
        for (std::size_t component = 0u;
             component < 5u;
             ++component) {
            wrenchParity = wrenchParity &&
                std::abs(
                    wrenchResult.criticObservations[
                        resetObservationBase + 12u + component
                    ]
                ) <= 1.0e-6f;
        }
        wrenchParity = wrenchParity &&
            wrenchResult.criticObservations[
                resetObservationBase + 17u
            ] == 1.0f &&
            wrenchResult.criticObservations[
                resetObservationBase + 18u
            ] == 1.0f;
    }
    if (wrenchParity) {
        for (std::size_t component = 0u;
             component < 6u;
             ++component) {
            const float taskWrench =
                wrenchResult.criticObservations[
                    finalWrenchBase + component
                ];
            const float sensorWrench =
                wrenchResult.sensorOutputs[component];
            double& normSquared = component < 3u
                ? forceNormSquared
                : torqueNormSquared;
            normSquared = std::fma(
                static_cast<double>(sensorWrench),
                static_cast<double>(sensorWrench),
                normSquared
            );
            wrenchParity = wrenchParity &&
                std::isfinite(sensorWrench) &&
                std::abs(sensorWrench - taskWrench) <=
                    3.0e-4f *
                    (1.0f + std::abs(taskWrench));
        }
    }
    const float forceNormNewtons =
        static_cast<float>(std::sqrt(forceNormSquared));
    const float torqueNormNewtonMetres =
        static_cast<float>(std::sqrt(torqueNormSquared));
    std::uint32_t expectedContactCount = 0u;
    double expectedNormalImpulse = 0.0;
    std::array<double, 3u> expectedTangentImpulse{};
    double expectedMaximumPenetration = 0.0;
    if (wrenchExecution.succeeded() &&
        !wrenchResult.contactStatuses.empty()) {
        const std::size_t published = std::min<std::size_t>(
            wrenchResult.contactStatuses.back().requiredConstraints,
            wrenchResult.layout.contactDispatch.constraintCapacity
        );
        const std::size_t begin = std::min<std::size_t>(
            wrenchResult.layout.contactDispatch.authoredConstraintCount,
            published
        );
        for (std::size_t index = begin; index < published; ++index) {
            const MRContactConstraintGPU& constraint =
                wrenchResult.contactEvidence.contacts[index];
            if ((constraint.flags &
                 (MR_CONSTRAINT_FLAG_DISABLED |
                  MR_CONSTRAINT_FLAG_GENERALIZED)) != 0u) {
                continue;
            }
            const bool parentA = constraint.bodyA == toolBodyIndex;
            const bool parentB = constraint.bodyB == toolBodyIndex;
            if (parentA == parentB ||
                (parentA ? constraint.bodyB : constraint.bodyA) !=
                    obstacleBodyIndex) {
                continue;
            }
            auto normalized = [](
                const std::array<double, 3u>& value,
                const std::array<double, 3u>& fallback
            ) {
                const double squared =
                    value[0u] * value[0u] +
                    value[1u] * value[1u] +
                    value[2u] * value[2u];
                if (!(squared > 1.0e-12) ||
                    !std::isfinite(squared)) {
                    return fallback;
                }
                const double inverse = 1.0 / std::sqrt(squared);
                return std::array<double, 3u>{
                    value[0u] * inverse,
                    value[1u] * inverse,
                    value[2u] * inverse,
                };
            };
            const std::array<double, 3u> normal = normalized(
                {
                    constraint.normal.x,
                    constraint.normal.y,
                    constraint.normal.z,
                },
                {0.0, 0.0, 1.0}
            );
            const double tangentAlongNormal =
                constraint.tangent.x * normal[0u] +
                constraint.tangent.y * normal[1u] +
                constraint.tangent.z * normal[2u];
            std::array<double, 3u> tangent = normalized(
                {
                    constraint.tangent.x -
                        tangentAlongNormal * normal[0u],
                    constraint.tangent.y -
                        tangentAlongNormal * normal[1u],
                    constraint.tangent.z -
                        tangentAlongNormal * normal[2u],
                },
                {1.0, 0.0, 0.0}
            );
            const std::array<double, 3u> bitangent{
                normal[1u] * tangent[2u] -
                    normal[2u] * tangent[1u],
                normal[2u] * tangent[0u] -
                    normal[0u] * tangent[2u],
                normal[0u] * tangent[1u] -
                    normal[1u] * tangent[0u],
            };
            const double parentSign = parentA ? -1.0 : 1.0;
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                expectedTangentImpulse[axis] += parentSign * (
                    tangent[axis] * constraint.impulses.y +
                    bitangent[axis] * constraint.impulses.z
                );
            }
            expectedNormalImpulse += std::max(
                static_cast<double>(constraint.impulses.x),
                0.0
            );
            expectedMaximumPenetration = std::max(
                expectedMaximumPenetration,
                std::max(
                    -static_cast<double>(
                        constraint.pointAndSeparation.w
                    ),
                    0.0
                )
            );
            ++expectedContactCount;
        }
    }
    const double inverseContactTimestep =
        1.0 /
        wrenchResult.layout.contactDispatch.timestepAndBias.x;
    const float expectedNormalForce = static_cast<float>(
        expectedNormalImpulse * inverseContactTimestep
    );
    const float expectedTangentialForce = static_cast<float>(
        std::sqrt(
            expectedTangentImpulse[0u] *
                expectedTangentImpulse[0u] +
            expectedTangentImpulse[1u] *
                expectedTangentImpulse[1u] +
            expectedTangentImpulse[2u] *
                expectedTangentImpulse[2u]
        ) * inverseContactTimestep
    );
    bool contactStateParity = expectedContactCount != 0u &&
        wrenchResult.sensorOutputs.size() == 16u &&
        wrenchResult.criticObservations.size() ==
            19u * (wrenchSteps + 1u);
    if (contactStateParity) {
        contactStateParity =
        wrenchResult.sensorOutputs[6u] == 1.0f &&
        std::abs(
            wrenchResult.sensorOutputs[7u] -
            static_cast<float>(expectedContactCount)
        ) <= 1.0e-6f &&
        std::abs(
            wrenchResult.sensorOutputs[8u] -
            expectedNormalForce
        ) <= 3.0e-4f * (1.0f + expectedNormalForce) &&
        std::abs(
            wrenchResult.sensorOutputs[9u] -
            expectedTangentialForce
        ) <= 3.0e-4f * (1.0f + expectedTangentialForce) &&
        std::abs(
            wrenchResult.sensorOutputs[10u] -
            static_cast<float>(expectedMaximumPenetration)
        ) <= 2.0e-5f;
        for (std::size_t channel = 0u;
             channel < 5u;
             ++channel) {
            contactStateParity = contactStateParity &&
                std::abs(
                    wrenchResult.sensorOutputs[6u + channel] -
                    wrenchResult.criticObservations[
                        finalWrenchBase + 12u + channel
                    ]
                ) <= 2.0e-5f &&
                std::abs(
                    wrenchResult.sensorOutputs[11u + channel]
                ) <= 1.0e-6f;
        }
        contactStateParity = contactStateParity &&
            wrenchResult.criticObservations[
                finalWrenchBase + 17u
            ] == 1.0f &&
            wrenchResult.criticObservations[
                finalWrenchBase + 18u
            ] == 1.0f &&
            wrenchResult.sensorMetadata.size() == 3u &&
            (wrenchResult.sensorMetadata[2u]
                 .ageValidityAndLayout.y &
             (MR_SENSOR_SAMPLE_FRESH |
              MR_SENSOR_SAMPLE_DROPPED)) ==
                (MR_SENSOR_SAMPLE_FRESH |
                 MR_SENSOR_SAMPLE_DROPPED) &&
            (wrenchResult.sensorMetadata[2u]
                 .ageValidityAndLayout.y &
             MR_SENSOR_SAMPLE_VALID) == 0u;
    }
    if (!wrenchParity || !contactStateParity ||
        !(forceNormNewtons > 0.1f) ||
        !(torqueNormNewtonMetres > 1.0e-4f)) {
        fail(
            "native force-torque SensorIR did not preserve the solved contact wrench: " +
            wrenchExecution.message +
            " force_n=" + std::to_string(forceNormNewtons) +
            " torque_nm=" +
            std::to_string(torqueNormNewtonMetres) +
            " contact_state=" +
            std::to_string(contactStateParity) +
            " count=" +
            std::to_string(expectedContactCount) + "/" +
            (wrenchResult.sensorOutputs.size() <= 8u
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.sensorOutputs[7u]
                   )) +
            " normal_n=" +
            std::to_string(expectedNormalForce) + "/" +
            (wrenchResult.sensorOutputs.size() <= 9u
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.sensorOutputs[8u]
                   )) +
            " host=" + std::to_string(
                static_cast<std::uint32_t>(wrenchExecution.status)
            ) +
            " gpu=" + std::to_string(
                wrenchExecution.firstGPUStatusCode
            ) +
            " contact=" +
            (wrenchResult.contactStatuses.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.contactStatuses.back().code
                   )) +
            " required=" +
            (wrenchResult.contactStatuses.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.contactStatuses.back()
                           .requiredConstraints
                   )) +
            " active=" +
            (wrenchResult.contactStatuses.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.contactStatuses.back()
                           .activeContacts
                   )) +
            " q=" +
            (wrenchResult.finalQ.empty()
                 ? std::string{"missing"}
                 : std::to_string(wrenchResult.finalQ.back())) +
            " contacts=" + std::to_string(
                wrenchResult.contactEvidence.contacts.size()
            ) +
            " impulse=" +
            (wrenchResult.contactEvidence.contacts.size() <=
                     wrenchResult.layout.contactDispatch
                         .authoredConstraintCount
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.contactEvidence.contacts[
                           wrenchResult.layout.contactDispatch
                               .authoredConstraintCount
                       ]
                           .impulses.x
                   )) +
            " bodies=" +
            (wrenchResult.contactEvidence.contacts.size() <=
                     wrenchResult.layout.contactDispatch
                         .authoredConstraintCount
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.contactEvidence.contacts[
                           wrenchResult.layout.contactDispatch
                               .authoredConstraintCount
                       ]
                           .bodyA
                   ) + "," +
                       std::to_string(
                           wrenchResult.contactEvidence.contacts[
                               wrenchResult.layout.contactDispatch
                                   .authoredConstraintCount
                           ]
                               .bodyB
                       )) +
            " task_f=" +
            (wrenchResult.criticObservations.size() <=
                     finalWrenchBase
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.criticObservations[
                           finalWrenchBase
                       ]
                   )) +
            " sensor_ft=" +
            (wrenchResult.sensorOutputs.size() < 6u
                 ? std::string{"missing"}
                 : std::to_string(wrenchResult.sensorOutputs[0u]) +
                       "," +
                       std::to_string(wrenchResult.sensorOutputs[1u]) +
                       "," +
                       std::to_string(wrenchResult.sensorOutputs[2u]) +
                       "," +
                       std::to_string(wrenchResult.sensorOutputs[3u]) +
                       "," +
                       std::to_string(wrenchResult.sensorOutputs[4u]) +
                       "," +
                       std::to_string(wrenchResult.sensorOutputs[5u])) +
            " validity=" +
            (wrenchResult.sensorMetadata.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       wrenchResult.sensorMetadata[0u]
                           .ageValidityAndLayout.y
                   ))
        );
    }

    // A reset is visible before physics executes. If that transaction then
    // overflows, q/v, scene state, manifolds, and every SensorIR schedule,
    // history, output, and metadata record must return to the last committed
    // boundary. The preceding resident step commits one obstacle contact;
    // reset brings a second obstacle into contact and exceeds a one-pair
    // operational capacity.
    metalrobo::SimulationDescription transactionAuthored =
        wrenchAuthored;
    transactionAuthored.task.id =
        "sensor_reset_failure_transaction";
    transactionAuthored.task.capacities.candidatePairs = 1u;
    transactionAuthored.model.bodies[obstacleBodyIndex]
        .motionType = MR_MOTION_KINEMATIC;
    transactionAuthored.sceneBodies.back().flagsAndIndices[0] =
        MR_MOTION_KINEMATIC;
    const std::uint32_t secondObstacleBodyIndex =
        static_cast<std::uint32_t>(
            transactionAuthored.model.bodies.size()
        );
    transactionAuthored.model.bodies.push_back(
        transactionAuthored.model.bodies[obstacleBodyIndex]
    );
    transactionAuthored.model.bodyNames.emplace_back(
        "wrench_obstacle_second"
    );
    MRShapeGPU secondObstacleShape = obstacleShape;
    secondObstacleShape.bodyIndex = secondObstacleBodyIndex;
    transactionAuthored.model.shapes.push_back(secondObstacleShape);
    transactionAuthored.model.shapeNames.emplace_back(
        "wrench_obstacle_second"
    );
    transactionAuthored.model.world.bodyCount =
        static_cast<std::uint32_t>(
            transactionAuthored.model.bodies.size()
        );
    transactionAuthored.model.world.shapeCount =
        static_cast<std::uint32_t>(
            transactionAuthored.model.shapes.size()
        );
    MRBodyStateGPU secondObstacleState =
        transactionAuthored.sceneBodies.back();
    secondObstacleState.flagsAndIndices[2] =
        secondObstacleBodyIndex;
    transactionAuthored.sceneBodies.push_back(secondObstacleState);
    const std::vector<MRBodyStateGPU> transactionResetScene =
        transactionAuthored.sceneBodies;
    transactionAuthored.sceneBodies.back().position.z += 2.0f;
    transactionAuthored.sensors.push_back({
        .id = "tool_pose_transaction_canary",
        .parentKind = MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
        .parentBodyIndex = toolBodyIndex,
        .kind = MR_WORLD_SENSOR_STATE,
        .localPose = {
            .position = {0.0f, 0.0f, 0.2f, 0.0f},
        },
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 2u,
        .consumerFlags = MR_WORLD_SENSOR_CONSUMER_RECORDER,
        .valueNoiseSigma = 0.02f,
        .biasNoiseSigma = 0.01f,
    });
    transactionAuthored.sensors.push_back({
        .id = "tool_imu_transaction_canary",
        .parentKind = MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
        .parentBodyIndex = toolBodyIndex,
        .kind = MR_WORLD_SENSOR_IMU,
        .localPose = {
            .position = {0.0f, 0.0f, 0.2f, 0.0f},
        },
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 2u,
        .consumerFlags = MR_WORLD_SENSOR_CONSUMER_RECORDER,
    });

    metalrobo::CompiledSimulation transactionCompiled;
    const auto transactionCompile = metalrobo::compileSimulation(
        transactionAuthored,
        0u,
        transactionCompiled
    );
    if (!transactionCompile.succeeded() ||
        transactionCompiled.world.capacities().candidatePairs != 1u ||
        transactionCompiled.world.eligiblePairCount() < 2u) {
        fail(
            "sensor rollback fixture did not compile its one-pair operational capacity: " +
            transactionCompile.world.message
        );
    }

    std::vector<float> transactionInitialQ =
        transactionAuthored.model.defaultQ;
    transactionInitialQ[0u] = -0.8f;
    const std::vector<float> transactionInitialV(
        transactionCompiled.world.nv(),
        0.0f
    );
    const std::vector<float> transactionEffort(
        transactionCompiled.world.nv(),
        0.0f
    );
    metalrobo::MetalWorldStepConfig transactionStep = wrenchStep;
    transactionStep.taskProgram = {};
    transactionStep.sensorProgram = transactionCompiled.sensors;
    transactionStep.policyProgram = {};
    transactionStep.evaluateFinalPolicy = false;
    transactionStep.actuationMode =
        metalrobo::MetalWorldActuationMode::effort;
    transactionStep.captureContactEvidence = true;
    metalrobo::MetalWorldContext transactionContext(
        contextConfiguration
    );
    metalrobo::MetalWorldResidentState transactionState;
    metalrobo::MetalWorldSubmission initialSubmission;
    const std::array<std::uint32_t, 1u> noReset{0u};
    const auto initialSubmit =
        transactionContext.initializeResidentState(
            transactionCompiled.world,
            {
                .environmentCount = 1u,
                .controlStepCount = 1u,
                .initialQ = transactionInitialQ,
                .initialV = transactionInitialV,
                .efforts = transactionEffort,
                .resetMasks = noReset,
                .resetQ = transactionAuthored.model.defaultQ,
                .resetV = transactionInitialV,
                .initialSceneBodies =
                    transactionAuthored.sceneBodies,
                .resetSceneBodies = transactionResetScene,
                .kinematicTargets =
                    transactionAuthored.sceneBodies,
            },
            transactionStep,
            transactionState,
            initialSubmission
        );
    metalrobo::MetalWorldResult initialTransactionResult;
    const auto initialTransaction = initialSubmit.succeeded()
        ? initialSubmission.wait(initialTransactionResult)
        : initialSubmit;
    const std::vector<float> committedSensorOutputs =
        initialTransactionResult.sensorOutputs;
    const std::vector<MRSensorSampleMetadataGPU>
        committedSensorMetadata =
            initialTransactionResult.sensorMetadata;
    const std::vector<float> committedQ =
        initialTransactionResult.finalQ;
    const std::vector<float> committedV =
        initialTransactionResult.finalV;
    const std::vector<MRBodyStateGPU> committedScene =
        initialTransactionResult.finalSceneBodies;
    const std::vector<std::uint32_t> committedManifoldCounts =
        initialTransactionResult.contactEvidence.manifoldCounts;
    const std::vector<MRManifoldHeaderGPU> committedManifoldHeaders =
        initialTransactionResult.contactEvidence.manifoldHeaders;
    const std::vector<MRManifoldPointGPU> committedManifoldPoints =
        initialTransactionResult.contactEvidence.manifoldPoints;
    if (!initialTransaction.succeeded() ||
        !transactionState.valid() ||
        committedSensorOutputs.size() != 29u ||
        committedSensorMetadata.size() != 5u ||
        committedManifoldCounts.size() != 1u ||
        committedManifoldCounts[0u] == 0u ||
        std::abs(committedSensorOutputs[18u]) < 1.0e-3f) {
        fail(
            "sensor rollback fixture did not establish a committed resident canary: " +
            initialTransaction.message
        );
    }

    metalrobo::MetalWorldSubmission rejectedSubmission;
    const std::array<std::uint32_t, 1u> requestReset{1u};
    const auto rejectedSubmit = transactionContext.submitResident(
        transactionCompiled.world,
        {
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .efforts = transactionEffort,
            .resetMasks = requestReset,
            .kinematicTargets = transactionResetScene,
        },
        transactionStep,
        transactionState,
        rejectedSubmission
    );
    metalrobo::MetalWorldResult rejectedTransactionResult;
    const auto rejectedTransaction = rejectedSubmit.succeeded()
        ? rejectedSubmission.wait(rejectedTransactionResult)
        : rejectedSubmit;
    const bool sensorOutputsRestored =
        rejectedTransactionResult.sensorOutputs ==
        committedSensorOutputs;
    const bool sensorMetadataRestored =
        rejectedTransactionResult.sensorMetadata.size() ==
            committedSensorMetadata.size() &&
        std::memcmp(
            rejectedTransactionResult.sensorMetadata.data(),
            committedSensorMetadata.data(),
            committedSensorMetadata.size() *
                sizeof(MRSensorSampleMetadataGPU)
        ) == 0;
    const bool sensorTransactionRestored =
        !rejectedTransaction.succeeded() &&
        rejectedTransaction.published &&
        rejectedTransaction.failedStepCount == 1u &&
        transactionState.valid() &&
        rejectedTransactionResult.contactStatuses.size() == 1u &&
        rejectedTransactionResult.contactStatuses[0u].code ==
            MR_STEP_PAIR_CAPACITY_OVERFLOW &&
        rejectedTransactionResult.contactStatuses[0u]
                .requiredPairs > 1u &&
        sensorOutputsRestored && sensorMetadataRestored;
    const bool physicalTransactionRestored =
        rejectedTransactionResult.finalQ == committedQ &&
        rejectedTransactionResult.finalV == committedV &&
        rejectedTransactionResult.finalSceneBodies.size() ==
            committedScene.size() &&
        std::memcmp(
            rejectedTransactionResult.finalSceneBodies.data(),
            committedScene.data(),
            committedScene.size() * sizeof(MRBodyStateGPU)
        ) == 0;
    const bool contactTransactionRestored =
        rejectedTransactionResult.contactEvidence.manifoldCounts ==
            committedManifoldCounts &&
        rejectedTransactionResult.contactEvidence.manifoldHeaders.size() ==
            committedManifoldHeaders.size() &&
        rejectedTransactionResult.contactEvidence.manifoldPoints.size() ==
            committedManifoldPoints.size() &&
        std::memcmp(
            rejectedTransactionResult.contactEvidence.manifoldHeaders.data(),
            committedManifoldHeaders.data(),
            committedManifoldHeaders.size() *
                sizeof(MRManifoldHeaderGPU)
        ) == 0 &&
        std::memcmp(
            rejectedTransactionResult.contactEvidence.manifoldPoints.data(),
            committedManifoldPoints.data(),
            committedManifoldPoints.size() *
                sizeof(MRManifoldPointGPU)
        ) == 0;
    if (!sensorTransactionRestored ||
        !physicalTransactionRestored ||
        !contactTransactionRestored) {
        fail(
            "reset-followed physics failure did not restore the committed SensorIR history: " +
            rejectedTransaction.message +
            " contact=" +
            (rejectedTransactionResult.contactStatuses.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       rejectedTransactionResult.contactStatuses[0u]
                           .code
                   )) +
            " required_pairs=" +
            (rejectedTransactionResult.contactStatuses.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       rejectedTransactionResult.contactStatuses[0u]
                           .requiredPairs
                   )) +
            " q=" +
            (rejectedTransactionResult.finalQ.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       rejectedTransactionResult.finalQ[0u]
                   )) +
            " published=" +
            std::to_string(rejectedTransaction.published) +
            " failed_steps=" +
            std::to_string(rejectedTransaction.failedStepCount) +
            " resident=" +
            std::to_string(transactionState.valid()) +
            " outputs=" + std::to_string(sensorOutputsRestored) +
            " metadata=" + std::to_string(sensorMetadataRestored) +
            " physical=" +
            std::to_string(physicalTransactionRestored) +
            " contact_state=" +
            std::to_string(contactTransactionRestored) +
            " committed_pose=" +
            std::to_string(committedSensorOutputs[16u]) + "," +
            std::to_string(committedSensorOutputs[17u]) + "," +
            std::to_string(committedSensorOutputs[18u]) +
            " rejected_pose=" +
            (rejectedTransactionResult.sensorOutputs.size() < 19u
                 ? std::string{"missing"}
                 : std::to_string(
                       rejectedTransactionResult.sensorOutputs[16u]
                   ) + "," +
                       std::to_string(
                           rejectedTransactionResult.sensorOutputs[17u]
                       ) + "," +
                       std::to_string(
                           rejectedTransactionResult.sensorOutputs[18u]
                       ))
        );
    }

    metalrobo::MetalWorldSubmission recoveredSubmission;
    const auto recoveredSubmit = transactionContext.submitResident(
        transactionCompiled.world,
        {
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .efforts = transactionEffort,
            .resetMasks = noReset,
            .kinematicTargets = transactionAuthored.sceneBodies,
        },
        transactionStep,
        transactionState,
        recoveredSubmission
    );
    metalrobo::MetalWorldResult recoveredTransactionResult;
    const auto recoveredTransaction = recoveredSubmit.succeeded()
        ? recoveredSubmission.wait(recoveredTransactionResult)
        : recoveredSubmit;
    if (!recoveredTransaction.succeeded() ||
        !transactionState.valid()) {
        fail(
            "resident execution did not recover after a transactionally rejected reset: " +
            recoveredTransaction.message
        );
    }

    // Prove the same rollback for native task state without publishing that
    // internal state through the API. Two deterministic resident sessions
    // start identically. Only the candidate branch attempts the rejected
    // reset; its following accepted transition must be byte-identical to the
    // reference branch that never observed the failure.
    metalrobo::MetalWorldStepConfig taskTransactionStep =
        transactionStep;
    taskTransactionStep.taskProgram = transactionCompiled.task;
    taskTransactionStep.policyProgram = transactionCompiled.policy;
    taskTransactionStep.evaluateFinalPolicy = true;
    taskTransactionStep.actuationMode =
        metalrobo::MetalWorldActuationMode::implicitPositionDrive;
    auto initializeTaskBranch = [&] (
        metalrobo::MetalWorldContext& branchContext,
        metalrobo::MetalWorldResidentState& branchState,
        metalrobo::MetalWorldResult& branchResult
    ) {
        metalrobo::MetalWorldSubmission submission;
        const auto submitted = branchContext.initializeResidentState(
            transactionCompiled.world,
            {
                .environmentCount = 1u,
                .controlStepCount = 1u,
                .initialQ = transactionInitialQ,
                .initialV = transactionInitialV,
                .resetMasks = noReset,
                .initialSceneBodies =
                    transactionAuthored.sceneBodies,
                .kinematicTargets =
                    transactionAuthored.sceneBodies,
            },
            taskTransactionStep,
            branchState,
            submission
        );
        return submitted.succeeded()
            ? submission.wait(branchResult)
            : submitted;
    };
    auto continueTaskBranch = [&] (
        metalrobo::MetalWorldContext& branchContext,
        metalrobo::MetalWorldResidentState& branchState,
        const std::span<const std::uint32_t> reset,
        const std::span<const MRBodyStateGPU> targets,
        metalrobo::MetalWorldResult& branchResult
    ) {
        metalrobo::MetalWorldSubmission submission;
        const auto submitted = branchContext.submitResident(
            transactionCompiled.world,
            {
                .environmentCount = 1u,
                .controlStepCount = 1u,
                .resetMasks = reset,
                .kinematicTargets = targets,
            },
            taskTransactionStep,
            branchState,
            submission
        );
        return submitted.succeeded()
            ? submission.wait(branchResult)
            : submitted;
    };
    const auto samePodVector = []<typename Value>(
        const std::vector<Value>& left,
        const std::vector<Value>& right
    ) {
        return left.size() == right.size() &&
            std::memcmp(
                left.data(),
                right.data(),
                left.size() * sizeof(Value)
            ) == 0;
    };

    metalrobo::MetalWorldContext taskReferenceContext(
        contextConfiguration
    );
    metalrobo::MetalWorldContext taskCandidateContext(
        contextConfiguration
    );
    metalrobo::MetalWorldResidentState taskReferenceState;
    metalrobo::MetalWorldResidentState taskCandidateState;
    metalrobo::MetalWorldResult taskReferenceInitial;
    metalrobo::MetalWorldResult taskCandidateInitial;
    const auto taskReferenceInitialization = initializeTaskBranch(
        taskReferenceContext,
        taskReferenceState,
        taskReferenceInitial
    );
    const auto taskCandidateInitialization = initializeTaskBranch(
        taskCandidateContext,
        taskCandidateState,
        taskCandidateInitial
    );
    if (!taskReferenceInitialization.succeeded() ||
        !taskCandidateInitialization.succeeded() ||
        !taskReferenceState.valid() ||
        !taskCandidateState.valid()) {
        fail(
            "native TaskIR rollback branches did not initialize: " +
            taskReferenceInitialization.message + " / " +
            taskCandidateInitialization.message
        );
    }

    metalrobo::MetalWorldResult taskRejectedResult;
    const auto taskRejected = continueTaskBranch(
        taskCandidateContext,
        taskCandidateState,
        requestReset,
        transactionResetScene,
        taskRejectedResult
    );
    const bool taskFailurePhysicsRestored =
        taskRejectedResult.finalQ == taskCandidateInitial.finalQ &&
        taskRejectedResult.finalV == taskCandidateInitial.finalV &&
        samePodVector(
            taskRejectedResult.finalSceneBodies,
            taskCandidateInitial.finalSceneBodies
        );
    if (taskRejected.succeeded() ||
        !taskRejected.published ||
        !taskCandidateState.valid() ||
        taskRejectedResult.contactStatuses.size() != 1u ||
        taskRejectedResult.contactStatuses[0u].code !=
            MR_STEP_PAIR_CAPACITY_OVERFLOW ||
        taskRejectedResult.transitions.size() != 1u ||
        taskRejectedResult.transitions[0u].termination.z == 0u ||
        taskRejectedResult.transitions[0u].termination.w !=
            MR_TASK_TERMINATION_PHYSICS_ERROR) {
        fail(
            "native TaskIR branch did not produce the expected rejected reset: " +
            taskRejected.message
        );
    }

    metalrobo::MetalWorldResult taskReferenceNext;
    metalrobo::MetalWorldResult taskCandidateNext;
    const auto taskReferenceContinuation = continueTaskBranch(
        taskReferenceContext,
        taskReferenceState,
        noReset,
        transactionAuthored.sceneBodies,
        taskReferenceNext
    );
    const auto taskCandidateContinuation = continueTaskBranch(
        taskCandidateContext,
        taskCandidateState,
        noReset,
        transactionAuthored.sceneBodies,
        taskCandidateNext
    );
    const bool samePhysics =
        taskReferenceNext.finalQ == taskCandidateNext.finalQ &&
        taskReferenceNext.finalV == taskCandidateNext.finalV &&
        samePodVector(
            taskReferenceNext.finalSceneBodies,
            taskCandidateNext.finalSceneBodies
        );
    const bool sameTask =
        taskReferenceNext.actorObservations ==
            taskCandidateNext.actorObservations &&
        taskReferenceNext.criticObservations ==
            taskCandidateNext.criticObservations &&
        samePodVector(
            taskReferenceNext.transitions,
            taskCandidateNext.transitions
        );
    const bool sameSensors =
        taskReferenceNext.sensorOutputs ==
            taskCandidateNext.sensorOutputs &&
        samePodVector(
            taskReferenceNext.sensorMetadata,
            taskCandidateNext.sensorMetadata
        );
    const bool samePolicy =
        taskReferenceNext.policyLatents ==
            taskCandidateNext.policyLatents &&
        taskReferenceNext.policyLogProbabilities ==
            taskCandidateNext.policyLogProbabilities &&
        taskReferenceNext.policyValues ==
            taskCandidateNext.policyValues;
    const bool sameContacts =
        samePodVector(
            taskReferenceNext.contactEvidence.manifoldHeaders,
            taskCandidateNext.contactEvidence.manifoldHeaders
        ) &&
        samePodVector(
            taskReferenceNext.contactEvidence.manifoldPoints,
            taskCandidateNext.contactEvidence.manifoldPoints
        ) &&
        taskReferenceNext.contactEvidence.manifoldCounts ==
            taskCandidateNext.contactEvidence.manifoldCounts;
    const bool taskBranchesMatch =
        taskReferenceContinuation.succeeded() &&
        taskCandidateContinuation.succeeded() &&
        samePhysics && sameTask && sameSensors && samePolicy &&
        sameContacts;
    if (!taskBranchesMatch) {
        fail(
            "native TaskIR state diverged after a rejected reset: " +
            taskReferenceContinuation.message + " / " +
            taskCandidateContinuation.message +
            " physics=" + std::to_string(samePhysics) +
            " task=" + std::to_string(sameTask) +
            " sensors=" + std::to_string(sameSensors) +
            " policy=" + std::to_string(samePolicy) +
            " contacts=" + std::to_string(sameContacts) +
            " failure_physics=" +
            std::to_string(taskFailurePhysicsRestored) +
            " q_ref=" +
            (taskReferenceNext.finalQ.empty()
                 ? std::string{"missing"}
                 : std::to_string(taskReferenceNext.finalQ[0u])) +
            " q_candidate=" +
            (taskCandidateNext.finalQ.empty()
                 ? std::string{"missing"}
                 : std::to_string(taskCandidateNext.finalQ[0u])) +
            " actor_ref=" +
            (taskReferenceNext.actorObservations.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       taskReferenceNext.actorObservations[0u]
                   )) +
            " actor_candidate=" +
            (taskCandidateNext.actorObservations.empty()
                 ? std::string{"missing"}
                 : std::to_string(
                       taskCandidateNext.actorObservations[0u]
                   ))
        );
    }
    return {
        .fingerprint = compiled.task.fingerprint(),
        .pipelineCount = taskPipelineCount,
        .forceNormNewtons = forceNormNewtons,
        .torqueNormNewtonMetres = torqueNormNewtonMetres,
    };
}

std::uint64_t checkWorldPackSimulationImport(
    const std::filesystem::path& path
) {
    metalrobo::EpisodeTwin episode =
        metalrobo::makeFrankaPickPlaceEpisodeTwin();
    metalrobo::EngineModel frankaModel =
        metalrobo::makeFrankaPickPlaceEngineModel();
    const auto wristSensor = std::find_if(
        episode.sensors.begin(),
        episode.sensors.end(),
        [](const metalrobo::SensorSpec& sensor) {
            return sensor.id == "wrist_rgbd";
        }
    );
    if (wristSensor == episode.sensors.end() ||
        wristSensor->parentBodyIndex == MR_INVALID_INDEX) {
        fail("WorldPack IMU fixture has no resolved robot link");
    }
    const std::uint32_t wristBodyIndex =
        wristSensor->parentBodyIndex;
    frankaModel.sites.push_back({
        .id = "worldpack_wrist_site",
        .bodyIndex = wristBodyIndex,
        .localPosition = {0.01f, -0.02f, 0.03f, 0.0f},
        .localOrientation = {
            0.0f,
            0.0f,
            0.7071067811865476f,
            0.7071067811865476f,
        },
    });
    episode.sensors.push_back({
        .id = "franka_worldpack_imu",
        .parentAssetId = episode.task.robotAssetId,
        .parentSite = "worldpack_wrist_site",
        .parentKind = MR_WORLD_SENSOR_PARENT_ASSET,
        .kind = MR_WORLD_SENSOR_IMU,
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 2u,
        .consumerFlags = MR_WORLD_SENSOR_CONSUMER_RECORDER,
        .valueNoiseSigma = 0.02f,
        .biasNoiseSigma = 0.01f,
        .dropoutProbability = 0.05f,
    });
    const auto objectAsset = std::find_if(
        episode.assets.begin(),
        episode.assets.end(),
        [](const metalrobo::WorldAsset& asset) {
            return asset.role == MR_WORLD_ASSET_MANIPULATED;
        }
    );
    if (objectAsset == episode.assets.end() ||
        objectAsset->bodyIndices.size() != 1u ||
        objectAsset->bodyIndices.front() >=
            frankaModel.bodyNames.size()) {
        fail("WorldPack contact-filter fixture has no object body");
    }
    const std::uint32_t objectBody =
        objectAsset->bodyIndices.front();
    episode.sensors.push_back({
        .id = "franka_worldpack_contact",
        .parentAssetId = episode.task.robotAssetId,
        .parentKind =
            MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
        .parentBodyIndex = wristBodyIndex,
        .kind = MR_WORLD_SENSOR_CONTACT_STATE,
        .filterBodies = {frankaModel.bodyNames[objectBody]},
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 2u,
        .consumerFlags = MR_WORLD_SENSOR_CONSUMER_RECORDER,
    });
    if (frankaModel.jointNames.empty()) {
        fail("WorldPack joint-state fixture has no semantic joint");
    }
    episode.sensors.push_back({
        .id = "franka_worldpack_joint",
        .parentAssetId = episode.task.robotAssetId,
        .parentKind = MR_WORLD_SENSOR_PARENT_ASSET,
        .kind = MR_WORLD_SENSOR_JOINT_STATE,
        .target = frankaModel.jointNames.front(),
        .nominalRateHz = 50.0f,
        .schedulePhase = MR_WORLD_SENSOR_PHASE_PRE_CONTROL,
        .historyLength = 2u,
        .consumerFlags = MR_WORLD_SENSOR_CONSUMER_RECORDER,
    });
    metalrobo::WorldTemplate worldTemplate;
    const metalrobo::WorldCompileResult templateStatus =
        metalrobo::compileEpisodeTwin(
            episode,
            frankaModel,
            worldTemplate
        );
    metalrobo::WorldFamily family;
    const metalrobo::WorldCompileResult familyStatus =
        metalrobo::compileWorldFamily(
            worldTemplate,
            metalrobo::makeFrankaPickPlaceWorldProgram(),
            family
        );
    metalrobo::MRWorldPack pack;
    const metalrobo::WorldPackResult packStatus =
        metalrobo::compileWorldPack(family, pack);
    const metalrobo::WorldPackResult writeStatus =
        packStatus.succeeded()
        ? metalrobo::writeWorldPack(pack, path)
        : packStatus;
    metalrobo::MRWorldPack loaded;
    const metalrobo::WorldPackResult readStatus =
        writeStatus.succeeded()
        ? metalrobo::readWorldPack(path, loaded)
        : writeStatus;
    if (!templateStatus.succeeded() ||
        !familyStatus.succeeded() ||
        !packStatus.succeeded() ||
        !writeStatus.succeeded() ||
        !readStatus.succeeded()) {
        fail("WorldPack simulation fixture did not compile");
    }
    metalrobo::TaskPack task;
    task.id = "world_pack_import_fixture";
    const metalrobo::SimulationDescription imported =
        metalrobo::makeWorldPackSimulation(
            loaded,
            std::move(task)
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
    const auto importedImu = std::find_if(
        imported.sensors.begin(),
        imported.sensors.end(),
        [](const metalrobo::SensorSpec& sensor) {
            return sensor.id == "franka_worldpack_imu";
        }
    );
    const auto importedContact = std::find_if(
        imported.sensors.begin(),
        imported.sensors.end(),
        [](const metalrobo::SensorSpec& sensor) {
            return sensor.id == "franka_worldpack_contact";
        }
    );
    const auto importedJoint = std::find_if(
        imported.sensors.begin(),
        imported.sensors.end(),
        [](const metalrobo::SensorSpec& sensor) {
            return sensor.id == "franka_worldpack_joint";
        }
    );
    metalrobo::CompiledWorld compiledWorld;
    const auto worldStatus = metalrobo::compileMetalWorld(
        imported.model,
        imported.articulationIndex,
        compiledWorld
    );
    metalrobo::CompiledSensorProgram compiledSensors;
    const auto sensorStatus = worldStatus.succeeded()
        ? metalrobo::compileSensorProgram(
              imported.sensors,
              imported.tactileSystem,
              compiledWorld,
              compiledSensors
          )
        : metalrobo::SensorCompileDiagnostics{
              .status =
                  metalrobo::SensorCompileStatus::invalidWorld,
          };
    const std::uint32_t compiledImu =
        compiledSensors.sensorIndex("franka_worldpack_imu");
    const std::uint32_t compiledContact =
        compiledSensors.sensorIndex("franka_worldpack_contact");
    const std::uint32_t compiledJoint =
        compiledSensors.sensorIndex("franka_worldpack_joint");
    const auto& loadedSites =
        loaded.family.worldTemplate.engineModel.sites;
    if (imported.articulationIndex != 0u ||
        imported.model.bodies.size() !=
            family.worldTemplate.engineModel.bodies.size() ||
        imported.sceneBodies.size() != expectedScene.size() ||
        !hasDynamicMass ||
        importedImu == imported.sensors.end() ||
        importedImu->parentKind !=
            MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK ||
        importedImu->parentBodyIndex != wristBodyIndex ||
        !importedImu->parentSite.empty() ||
        importedImu->localPose.position.x != 0.01f ||
        importedImu->localPose.position.y != -0.02f ||
        importedImu->localPose.position.z != 0.03f ||
        importedImu->localPose.orientation.z !=
            0.7071067811865476f ||
        importedImu->localPose.orientation.w !=
            0.7071067811865476f ||
        importedImu->valueNoiseSigma != 0.02f ||
        importedImu->biasNoiseSigma != 0.01f ||
        importedImu->dropoutProbability != 0.05f ||
        importedContact == imported.sensors.end() ||
        importedContact->filterBodies.size() != 1u ||
        importedContact->filterBodies.front() !=
            frankaModel.bodyNames[objectBody] ||
        importedJoint == imported.sensors.end() ||
        importedJoint->parentKind !=
            MR_WORLD_SENSOR_PARENT_ASSET ||
        importedJoint->parentBodyIndex != MR_INVALID_INDEX ||
        importedJoint->target !=
            frankaModel.jointNames.front() ||
        !worldStatus.succeeded() ||
        !sensorStatus.succeeded() ||
        compiledImu == MR_INVALID_INDEX ||
        compiledSensors.descriptors()[compiledImu].identity.x !=
            MR_WORLD_SENSOR_IMU ||
        compiledSensors.descriptors()[compiledImu].noise.x != 0.02f ||
        compiledSensors.descriptors()[compiledImu].noise.y != 0.01f ||
        compiledSensors.descriptors()[compiledImu].noise.z != 0.05f ||
        compiledContact == MR_INVALID_INDEX ||
        compiledSensors.descriptors()[compiledContact].identity.x !=
            MR_WORLD_SENSOR_CONTACT_STATE ||
        compiledSensors.descriptors()[compiledContact].filter.y != 1u ||
        compiledSensors.filterBodies()[
            compiledSensors.descriptors()[compiledContact].filter.x
        ] != objectBody ||
        compiledJoint == MR_INVALID_INDEX ||
        compiledSensors.descriptors()[compiledJoint].identity.x !=
            MR_WORLD_SENSOR_JOINT_STATE ||
        compiledSensors.descriptors()[compiledJoint].source.z != 0u ||
        compiledSensors.descriptors()[compiledJoint].source.x !=
            frankaModel.joints[0u].qOffset ||
        compiledSensors.descriptors()[compiledJoint].source.y !=
            frankaModel.joints[0u].vOffset ||
        loadedSites.size() != 1u ||
        loadedSites.front().id != "worldpack_wrist_site" ||
        loadedSites.front().bodyIndex !=
            wristBodyIndex ||
        loadedSites.front().localPosition.x != 0.01f ||
        loadedSites.front().localPosition.y != -0.02f ||
        loadedSites.front().localPosition.z != 0.03f ||
        loadedSites.front().localOrientation.z !=
            0.7071067811865476f ||
        loadedSites.front().localOrientation.w !=
            0.7071067811865476f ||
        loaded.contentHash != pack.contentHash) {
        fail(
            "WorldPack did not round-trip executable mechanics, scene state, and IMU SensorIR"
        );
    }
    return loaded.contentHash;
}

} // namespace

int main() {
    try {
        metalrobo::SimulationDescription authored =
            metalrobo::makeUnitreeG1Simulation(
                metalrobo::BuiltinSurface::terrain
            );
        metalrobo::CompiledSimulation compiledWorld;
        const metalrobo::SimulationCompileDiagnostics
            compileStatus = metalrobo::compileSimulation(
                authored,
                0u,
                compiledWorld
            );
        if (!compileStatus.world.succeeded()) {
            fail(
                "world compilation failed: " +
                compileStatus.world.message
            );
        }
        if (!compileStatus.task.succeeded()) {
            fail(
                "task compilation failed [" +
                std::string{
                    metalrobo::taskCompileStatusName(
                        compileStatus.task.status
                    )
                } +
                "]: " + compileStatus.task.element + ": " +
                compileStatus.task.message
            );
        }
        const metalrobo::CompiledWorld& world =
            compiledWorld.world;
        const metalrobo::CompiledTaskProgram& program =
            compiledWorld.task;
        for (const metalrobo::G1ActuatorPresetId preset : {
                 metalrobo::G1ActuatorPresetId::unitreeUrdfRev10,
                 metalrobo::G1ActuatorPresetId::unitreeMjcfRev10,
             }) {
            metalrobo::CompiledSimulation variant;
            const auto variantStatus = metalrobo::compileSimulation(
                metalrobo::makeUnitreeG1Simulation(
                    metalrobo::BuiltinSurface::terrain,
                    preset
                ),
                0u,
                variant
            );
            if (!variantStatus.succeeded() ||
                variant.world.fingerprint() == world.fingerprint() ||
                variant.task.fingerprint() == program.fingerprint()) {
                fail(
                    "G1 actuator preset is not bound into world/task fingerprints"
                );
            }
        }
        const metalrobo::TaskProgramLayout& layout =
            program.layout();
        const std::uint32_t expectedConstraintBlocks =
            static_cast<std::uint32_t>(
                world.model().constraintProgram.blocks.size()
            ) +
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY *
                world.capacities().manifolds;
        if (layout.actionCount != 29u ||
            layout.actorFrameSize != 96u ||
            layout.actorHistoryLength != 5u ||
            layout.actorObservationSize != 480u ||
            layout.criticFrameSize != 99u ||
            layout.criticHistoryLength != 5u ||
            layout.criticObservationSize != 495u ||
            layout.contactMetricCount != 37u ||
            layout.delayStateCount != 2u) {
            fail("compiled G1 task layout changed");
        }
        if (program.worldFingerprint() != world.fingerprint() ||
            world.capacities().candidatePairs != 128u ||
            world.capacities().rawContacts != 128u ||
            world.capacities().manifolds != 32u ||
            world.capacities().constraintBlocks !=
                expectedConstraintBlocks ||
            world.capacities().constraintRows !=
                3u * expectedConstraintBlocks ||
            world.capacities().endpointRuntimeRecords !=
                2u * expectedConstraintBlocks ||
            world.capacities().articulationPointQueries !=
                2u * expectedConstraintBlocks ||
            program.header().root.y != 0u ||
            std::abs(
                program.header().rootReference.z -
                0.076030060304f
            ) > 1.0e-6f ||
            program.header().counts0.w != 4u ||
            program.header().counts1.w != 19u ||
            program.header().counts2.x != 2u ||
            program.header().counts2.y != 5u ||
            program.header().articulation.w != 5u ||
            program.header().commandLower.x != 0.1f ||
            program.header().commandLower.y != 0.0f ||
            program.header().commandLower.z != 0.0f ||
            program.header().commandUpper.x != 0.1f ||
            program.header().commandUpper.y != 0.0f ||
            program.header().commandUpper.z != 0.0f ||
            std::abs(
                program.header().commandUpper.w - 0.8f
            ) > 1.0e-6f ||
            program.terminationOperators()[0].parameters.y !=
                -2.0f ||
            program.terminationOperators()[1].parameters.y !=
                -2.0f ||
            program.randomizationOperators()[0].target.w != 0u ||
            program.randomizationOperators()[1].target.w != 0u ||
            program.randomizationOperators()[2].target.w != 2u ||
            program.randomizationOperators()[3].target.w != 2u ||
            program.randomizationOperators()[4].target.w != 2u ||
            program.terrainSampleOffsets().size() != 187u ||
            program.terrainResetTranslations().size() != 11u) {
            fail("compiled G1 task tables are incomplete");
        }
        metalrobo::SimulationDescription invalidEndpointCapacity =
            authored;
        invalidEndpointCapacity.task.capacities
            .endpointRuntimeRecords =
                world.capacities().endpointRuntimeRecords - 1u;
        metalrobo::CompiledSimulation invalidCompiledWorld;
        const auto invalidEndpointStatus =
            metalrobo::compileSimulation(
                invalidEndpointCapacity,
                0u,
                invalidCompiledWorld
            );
        if (invalidEndpointStatus.world.status !=
            metalrobo::MetalWorldHostStatus::capacityOverflow) {
            fail(
                "noncanonical endpoint-runtime capacity was not rejected"
            );
        }
        metalrobo::SimulationDescription invalidQueryCapacity =
            authored;
        invalidQueryCapacity.task.capacities
            .articulationPointQueries =
                world.capacities().articulationPointQueries + 1u;
        const auto invalidQueryStatus =
            metalrobo::compileSimulation(
                invalidQueryCapacity,
                0u,
                invalidCompiledWorld
            );
        if (invalidQueryStatus.world.status !=
            metalrobo::MetalWorldHostStatus::capacityOverflow) {
            fail(
                "noncanonical articulation-query capacity was not rejected"
            );
        }
        for (std::uint32_t action = 0u;
             action < program.actionBindings().size();
             ++action) {
            const MRTaskActionBindingGPU& binding =
                program.actionBindings()[action];
            if (binding.indices.x != action ||
                binding.indices.y != 6u + action ||
                binding.indices.z != 7u + action ||
                binding.indices.w != 6u + action) {
                fail("semantic action binding did not resolve");
            }
        }

        metalrobo::CompiledTaskProgram repeated;
        const metalrobo::TaskCompileDiagnostics repeatedStatus =
            metalrobo::compileTaskProgram(
                authored.task,
                world,
                compiledWorld.sensors,
                repeated
            );
        if (!repeatedStatus.succeeded() ||
            repeated.fingerprint() != program.fingerprint()) {
            fail("task compilation is not content deterministic");
        }

        metalrobo::TaskPack broken = authored.task;
        broken.actions.front().joint =
            "missing_joint_from_import";
        const std::uint64_t preserved =
            repeated.fingerprint();
        const metalrobo::TaskCompileDiagnostics rejected =
            metalrobo::compileTaskProgram(
                broken,
                world,
                compiledWorld.sensors,
                repeated
            );
        if (rejected.status !=
                metalrobo::TaskCompileStatus::
                    unresolvedSemantic ||
            repeated.fingerprint() != preserved) {
            fail(
                "unresolved semantic binding was not transactionally rejected"
            );
        }
        metalrobo::TaskPack mismatched = authored.task;
        ++mismatched.capacities.candidatePairs;
        const metalrobo::TaskCompileDiagnostics
            capacityRejected = metalrobo::compileTaskProgram(
                mismatched,
                world,
                compiledWorld.sensors,
                repeated
            );
        if (capacityRejected.status !=
                metalrobo::TaskCompileStatus::invalidWorld ||
            repeated.fingerprint() != preserved) {
            fail(
                "mismatched task capacity contract was not transactionally rejected"
            );
        }

        metalrobo::PolicyPack policy;
        policy.id = "task_program_check_policy";
        policy.revision = 7u;
        policy.layers = {
            {
                .inputCount = layout.actorObservationSize,
                .outputCount = 32u,
                .activation =
                    metalrobo::PolicyActivation::tanh,
                .weights = std::vector<float>(
                    layout.actorObservationSize * 32u,
                    0.0f
                ),
                .bias = std::vector<float>(32u, 0.0f),
            },
            {
                .inputCount = 32u,
                .outputCount = layout.actionCount,
                .activation =
                    metalrobo::PolicyActivation::identity,
                .weights = std::vector<float>(
                    32u * layout.actionCount,
                    0.0f
                ),
                .bias = std::vector<float>(
                    layout.actionCount,
                    0.0f
                ),
            },
        };
        policy.criticLayers = {
            {
                .inputCount =
                    layout.criticObservationSize,
                .outputCount = 16u,
                .activation =
                    metalrobo::PolicyActivation::elu,
                .weights = std::vector<float>(
                    layout.criticObservationSize * 16u,
                    0.0f
                ),
                .bias = std::vector<float>(16u, 0.0f),
            },
            {
                .inputCount = 16u,
                .outputCount = 1u,
                .activation =
                    metalrobo::PolicyActivation::identity,
                .weights = std::vector<float>(16u, 0.0f),
                .bias = std::vector<float>(1u, 0.0f),
            },
        };
        policy.actionLogStandardDeviation.assign(
            layout.actionCount,
            -0.5f
        );
        metalrobo::CompiledPolicyProgram compiledPolicy;
        const metalrobo::PolicyCompileDiagnostics
            policyStatus = metalrobo::compilePolicyProgram(
                policy,
                program,
                compiledPolicy
            );
        if (!policyStatus.succeeded() ||
            !compiledPolicy.valid() ||
            compiledPolicy.topologyFingerprint() == 0u ||
            compiledPolicy.taskFingerprint() !=
                program.fingerprint() ||
            compiledPolicy.revision() != 7u ||
            compiledPolicy.layout().actorObservationCount !=
                layout.actorObservationSize ||
            compiledPolicy.layout().criticObservationCount !=
                layout.criticObservationSize ||
            compiledPolicy.layout().actionCount !=
                layout.actionCount ||
            compiledPolicy.layout().maximumHiddenCount !=
                32u ||
            !compiledPolicy.layout().stochastic ||
            compiledPolicy.actorLayers().size() != 2u ||
            compiledPolicy.criticLayers().size() != 2u) {
            fail("generic PolicyPack compilation failed");
        }
        metalrobo::PolicyPack weightRevision = policy;
        weightRevision.revision = 8u;
        weightRevision.layers.front().bias.front() = 0.125f;
        metalrobo::CompiledPolicyProgram revisedPolicyProgram;
        const auto weightRevisionStatus =
            metalrobo::compilePolicyProgram(
                weightRevision,
                program,
                revisedPolicyProgram
            );
        if (!weightRevisionStatus.succeeded() ||
            revisedPolicyProgram.fingerprint() ==
                compiledPolicy.fingerprint() ||
            revisedPolicyProgram.topologyFingerprint() !=
                compiledPolicy.topologyFingerprint()) {
            fail(
                "PolicyIR topology fingerprint changed across a weight-only revision"
            );
        }
        metalrobo::PolicyPack topologyRevision = policy;
        topologyRevision.revision = 9u;
        topologyRevision.layers.front().activation =
            metalrobo::PolicyActivation::relu;
        metalrobo::CompiledPolicyProgram changedTopologyProgram;
        const auto topologyRevisionStatus =
            metalrobo::compilePolicyProgram(
                topologyRevision,
                program,
                changedTopologyProgram
            );
        if (!topologyRevisionStatus.succeeded() ||
            changedTopologyProgram.topologyFingerprint() ==
                compiledPolicy.topologyFingerprint()) {
            fail(
                "PolicyIR topology fingerprint ignored an operator change"
            );
        }
        metalrobo::PolicyPack badPolicy = policy;
        --badPolicy.layers.front().inputCount;
        const std::uint64_t preservedPolicy =
            compiledPolicy.fingerprint();
        const metalrobo::PolicyCompileDiagnostics
            policyRejected = metalrobo::compilePolicyProgram(
                badPolicy,
                program,
                compiledPolicy
            );
        if (policyRejected.status !=
                metalrobo::PolicyCompileStatus::
                    incompatibleContract ||
            compiledPolicy.fingerprint() != preservedPolicy) {
            fail(
                "incompatible PolicyPack was not transactionally rejected"
            );
        }

        TemporaryPackFiles packFiles;
        const metalrobo::LearningPackResult taskWrite =
            metalrobo::writeTaskPack(
                authored.task,
                packFiles.task
            );
        const metalrobo::LearningPackResult policyWrite =
            metalrobo::writePolicyPack(
                policy,
                packFiles.policy
            );
        if (!taskWrite.succeeded() ||
            !policyWrite.succeeded()) {
            fail(
                "learning-pack write failed: " +
                taskWrite.message + " " +
                policyWrite.message
            );
        }
        metalrobo::TaskPack loadedTask;
        metalrobo::PolicyPack loadedPolicy;
        const metalrobo::LearningPackResult taskRead =
            metalrobo::readTaskPack(
                packFiles.task,
                loadedTask
            );
        const metalrobo::LearningPackResult policyRead =
            metalrobo::readPolicyPack(
                packFiles.policy,
                loadedPolicy
            );
        metalrobo::CompiledTaskProgram roundTripTask;
        const metalrobo::TaskCompileDiagnostics
            roundTripTaskStatus =
                metalrobo::compileTaskProgram(
                    loadedTask,
                    world,
                    compiledWorld.sensors,
                    roundTripTask
                );
        metalrobo::CompiledPolicyProgram roundTripPolicy;
        const metalrobo::PolicyCompileDiagnostics
            roundTripPolicyStatus =
                metalrobo::compilePolicyProgram(
                    loadedPolicy,
                    roundTripTask,
                    roundTripPolicy
                );
        if (!taskRead.succeeded() ||
            !policyRead.succeeded() ||
            !roundTripTaskStatus.succeeded() ||
            !roundTripPolicyStatus.succeeded() ||
            roundTripTask.fingerprint() !=
                program.fingerprint() ||
            roundTripPolicy.fingerprint() !=
                preservedPolicy) {
            fail(
                "TaskPack or PolicyPack round trip changed its compiled program"
            );
        }
        metalrobo::PolicyRolloutPack rollout;
        rollout.id = "task_program_check_rollout";
        rollout.taskFingerprint = program.fingerprint();
        rollout.policyFingerprint = preservedPolicy;
        rollout.policyRevision = policy.revision;
        rollout.environmentCount = 2u;
        rollout.controlStepCount = 3u;
        rollout.actorObservationCount =
            layout.actorObservationSize;
        rollout.criticObservationCount =
            layout.criticObservationSize;
        rollout.actionCount = layout.actionCount;
        constexpr std::size_t sampleCount = 6u;
        rollout.actorObservations.assign(
            sampleCount * layout.actorObservationSize,
            0.25f
        );
        rollout.criticObservations.assign(
            sampleCount * layout.criticObservationSize,
            -0.5f
        );
        rollout.latents.assign(
            sampleCount * layout.actionCount,
            0.125f
        );
        rollout.logProbabilities.assign(
            sampleCount,
            -3.0f
        );
        rollout.values.assign(sampleCount, 0.75f);
        rollout.bootstrapValues.assign(2u, 0.5f);
        rollout.transitions.resize(sampleCount);
        for (std::size_t index = 0u;
             index < sampleCount;
             ++index) {
            rollout.transitions[index].rewardAndState.x =
                static_cast<float>(index) * 0.1f;
            rollout.transitions[index].termination.x =
                index + 1u == sampleCount ? 1u : 0u;
            rollout.transitions[index].policyRevision =
                policy.revision;
        }
        const metalrobo::LearningPackResult rolloutWrite =
            metalrobo::writePolicyRolloutPack(
                rollout,
                packFiles.rollout
            );
        const metalrobo::LearningPackResult borrowedWrite =
            metalrobo::writePolicyRolloutPack(
                metalrobo::PolicyRolloutPackView{
                    .id = rollout.id,
                    .taskFingerprint =
                        rollout.taskFingerprint,
                    .policyFingerprint =
                        rollout.policyFingerprint,
                    .policyRevision =
                        rollout.policyRevision,
                    .environmentCount =
                        rollout.environmentCount,
                    .controlStepCount =
                        rollout.controlStepCount,
                    .actorObservationCount =
                        rollout.actorObservationCount,
                    .criticObservationCount =
                        rollout.criticObservationCount,
                    .actionCount = rollout.actionCount,
                    .actorObservations =
                        rollout.actorObservations,
                    .criticObservations =
                        rollout.criticObservations,
                    .latents = rollout.latents,
                    .logProbabilities =
                        rollout.logProbabilities,
                    .values = rollout.values,
                    .bootstrapValues =
                        rollout.bootstrapValues,
                    .transitions = rollout.transitions,
                },
                packFiles.borrowedRollout
            );
        metalrobo::PolicyRolloutPack loadedRollout;
        const metalrobo::LearningPackResult rolloutRead =
            metalrobo::readPolicyRolloutPack(
                packFiles.rollout,
                loadedRollout
            );
        if (!rolloutWrite.succeeded() ||
            !borrowedWrite.succeeded() ||
            borrowedWrite.contentHash !=
                rolloutWrite.contentHash ||
            !rolloutRead.succeeded() ||
            loadedRollout.id != rollout.id ||
            loadedRollout.taskFingerprint !=
                rollout.taskFingerprint ||
            loadedRollout.policyFingerprint !=
                rollout.policyFingerprint ||
            loadedRollout.actorObservations !=
                rollout.actorObservations ||
            loadedRollout.transitions.size() !=
                rollout.transitions.size() ||
            loadedRollout.transitions.back()
                    .termination.x != 1u) {
            fail(
                "PolicyRolloutPack round trip changed its learning records"
            );
        }
        const std::uint64_t importedFingerprint =
            compileFloatingBaseTaskFixture();
        const FixedBaseTaskEvidence fixedBase =
            compileFixedBaseTaskFixture();
        const std::uint64_t worldPackFingerprint =
            checkWorldPackSimulationImport(packFiles.world);

        std::cout
            << "task_program_check passed"
            << " fingerprint=" << program.fingerprint()
            << " actions=" << layout.actionCount
            << " actor=" << layout.actorObservationSize
            << " critic=" << layout.criticObservationSize
            << " contact_metrics=" << layout.contactMetricCount
            << " policy=" << preservedPolicy
            << " task_artifact=" << taskWrite.contentHash
            << " policy_artifact=" << policyWrite.contentHash
            << " rollout_artifact="
            << rolloutWrite.contentHash
            << " imported=" << importedFingerprint
            << " fixed_base=" << fixedBase.fingerprint
            << " plan_pipelines=" << fixedBase.pipelineCount
            << "/" << MR_RUNTIME_PIPELINE_COUNT
            << " force_n=" << fixedBase.forceNormNewtons
            << " torque_nm=" << fixedBase.torqueNormNewtonMetres
            << " sensor_transaction=pass"
            << " physical_transaction=pass"
            << " task_transaction=pass"
            << " imu=pass"
            << " joint_state=pass"
            << " contact_state=pass"
            << " sensor_corruption=pass"
            << " world_pack=" << worldPackFingerprint
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "task_program_check failed: "
            << error.what() << '\n';
        return 1;
    }
}
