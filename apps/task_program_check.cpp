#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/Simulation.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/TaskProgram.hpp"
#include "metalrobo/WorldPack.hpp"

#include <algorithm>
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
    }

    ~TemporaryPackFiles() {
        std::error_code ignored;
        std::filesystem::remove(task, ignored);
        std::filesystem::remove(policy, ignored);
        std::filesystem::remove(rollout, ignored);
        std::filesystem::remove(borrowedRollout, ignored);
    }

    std::filesystem::path task;
    std::filesystem::path policy;
    std::filesystem::path rollout;
    std::filesystem::path borrowedRollout;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
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

std::uint64_t compileFixedBaseTaskFixture() {
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
            .body = "tool",
            .localPosition = {0.0f, 0.0f, 0.2f, 0.0f},
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
    authored.sensors = {
        {
            .id = "tool_pose_delayed",
            .parentKind =
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK,
            .parentBodyIndex = toolBodyIndex,
            .kind = MR_WORLD_SENSOR_STATE,
            .localPose = {
                .position = {0.0f, 0.0f, 0.2f, 0.0f},
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
        policyStats.policyBankReuseCount != 1u) {
        fail(
            "native policy bank swap did not reuse the retained prior revision"
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
    twistAuthored.sensors.clear();
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
        .inputCount = 2u,
        .outputCount = 1u,
        .activation = metalrobo::PolicyActivation::identity,
        .weights = std::vector<float>(2u, 0.0f),
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
    metalrobo::MetalWorldStepConfig twistStep = step;
    twistStep.taskProgram = twistCompiled.task;
    twistStep.sensorProgram = {};
    twistStep.policyProgram = twistCompiled.policy;
    twistStep.publishSensorOutputs = false;
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
            .initialSceneBodies = authored.sceneBodies,
        },
        twistStep,
        twistResult
    );
    if (!twistExecution.succeeded() ||
        twistResult.actorObservations.size() != twistSteps + 1u ||
        twistResult.criticObservations.size() !=
            (twistSteps + 1u) * 2u ||
        std::abs(twistResult.actorObservations[0u] - 0.5f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[0u] - 0.1f) >
            2.0e-4f ||
        std::abs(twistResult.criticObservations[1u] - 0.5f) >
            2.0e-4f) {
        fail(
            "frame twist did not materialize from the randomized reset state: " +
            twistExecution.message
        );
    }
    for (std::size_t sample = 0u;
         sample < twistSteps + 1u;
         ++sample) {
        const float jointVelocity =
            twistResult.actorObservations[sample];
        const float frameLinear =
            twistResult.criticObservations[2u * sample];
        const float frameAngular =
            twistResult.criticObservations[2u * sample + 1u];
        if (!std::isfinite(frameLinear) ||
            std::abs(frameAngular - jointVelocity) > 2.0e-4f) {
            fail(
                "accepted frame twist disagrees with generalized velocity"
            );
        }
    }
    return compiled.task.fingerprint();
}

std::uint64_t checkWorldPackSimulationImport() {
    metalrobo::WorldTemplate worldTemplate;
    const metalrobo::WorldCompileResult templateStatus =
        metalrobo::compileEpisodeTwin(
            metalrobo::makeFrankaPickPlaceEpisodeTwin(),
            metalrobo::makeFrankaPickPlaceEngineModel(),
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
    if (!templateStatus.succeeded() ||
        !familyStatus.succeeded() ||
        !packStatus.succeeded()) {
        fail("WorldPack simulation fixture did not compile");
    }
    metalrobo::TaskPack task;
    task.id = "world_pack_import_fixture";
    const metalrobo::SimulationDescription imported =
        metalrobo::makeWorldPackSimulation(
            pack,
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
    if (imported.articulationIndex != 0u ||
        imported.model.bodies.size() !=
            family.worldTemplate.engineModel.bodies.size() ||
        imported.sceneBodies.size() != expectedScene.size() ||
        !hasDynamicMass) {
        fail(
            "WorldPack did not materialize executable mechanics and scene state"
        );
    }
    return pack.contentHash;
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
        const std::uint64_t fixedBaseFingerprint =
            compileFixedBaseTaskFixture();
        const std::uint64_t worldPackFingerprint =
            checkWorldPackSimulationImport();

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
            << " fixed_base=" << fixedBaseFingerprint
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
