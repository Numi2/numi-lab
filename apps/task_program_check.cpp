#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
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
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
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
        actuators = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".actuatorpack");
        sensors = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".sensorpack");
        reality = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".realitypack");
        policy = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".policypack");
        rollout = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".rolloutpack");
        borrowedRollout = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".borrowed.rolloutpack");
        motion = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".motionpack");
        interaction = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".interactionpack");
        world = directory /
            ("metalrobo_task_program_check_" + suffix +
             ".mrworld");
    }

    ~TemporaryPackFiles() {
        std::error_code ignored;
        std::filesystem::remove(task, ignored);
        std::filesystem::remove(actuators, ignored);
        std::filesystem::remove(sensors, ignored);
        std::filesystem::remove(reality, ignored);
        std::filesystem::remove(policy, ignored);
        std::filesystem::remove(rollout, ignored);
        std::filesystem::remove(borrowedRollout, ignored);
        std::filesystem::remove(motion, ignored);
        std::filesystem::remove(interaction, ignored);
        std::filesystem::remove(world, ignored);
    }

    std::filesystem::path task;
    std::filesystem::path actuators;
    std::filesystem::path sensors;
    std::filesystem::path reality;
    std::filesystem::path policy;
    std::filesystem::path rollout;
    std::filesystem::path borrowedRollout;
    std::filesystem::path motion;
    std::filesystem::path interaction;
    std::filesystem::path world;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

struct TaskWorldFixture : metalrobo::LocomotionWorld {
    metalrobo::TaskPack task;
    std::vector<metalrobo::RobotActuatorSpec> actuators;
    metalrobo::TaskObservationProgram observations;
    metalrobo::TaskResetProgram reset;
};

void configureG1RecoveryFixture(TaskWorldFixture& fixture) {
    if (!fixture.model.materials.empty()) {
        fixture.model.materials.front().response.z = 1.25e-7f;
    }
    for (auto& actuator : fixture.actuators) {
        const auto joint = std::ranges::find(
            fixture.model.jointNames, actuator.target);
        if (joint == fixture.model.jointNames.end()) continue;
        const auto jointIndex = static_cast<std::uint32_t>(
            joint - fixture.model.jointNames.begin());
        const auto dof = std::ranges::find_if(
            fixture.model.dofs,
            [jointIndex](const MRDofPropertiesGPU& candidate) {
                return candidate.jointIndex == jointIndex;
            });
        if (dof == fixture.model.dofs.end() ||
            dof->qIndex >= fixture.model.defaultQ.size()) continue;
        const float rest = fixture.model.defaultQ[dof->qIndex];
        actuator.scale = std::max(
            std::abs(dof->limits.x - rest),
            std::abs(dof->limits.y - rest));
        actuator.responseTimeSeconds = 0.0f;
    }
}

TaskWorldFixture makeG1TaskWorld(
    const metalrobo::LocomotionSurface surface,
    const metalrobo::UnitreeG1Task taskKind =
        metalrobo::UnitreeG1Task::velocity
) {
    TaskWorldFixture result;
    static_cast<metalrobo::LocomotionWorld&>(result) =
        metalrobo::makeUnitreeG1LocomotionWorld(surface);
    const auto robot = metalrobo::builtinRobotPack("unitree_g1");
    if (!robot) fail("bundled G1 controller contract is unavailable");
    result.actuators = robot->actuators;
    switch (taskKind) {
    case metalrobo::UnitreeG1Task::velocity:
        result.task = metalrobo::makeUnitreeG1LocomotionTaskPack(
            surface, result.observations, result.reset);
        break;
    case metalrobo::UnitreeG1Task::disturbanceRecovery:
        result.task = metalrobo::makeUnitreeG1DisturbanceRecoveryTaskPack(
            surface, result.observations, result.reset);
        break;
    case metalrobo::UnitreeG1Task::supineGetUpDiscovery:
        result.task = metalrobo::makeUnitreeG1SupineGetUpDiscoveryTaskPack(
            surface, result.observations, result.reset);
        configureG1RecoveryFixture(result);
        break;
    case metalrobo::UnitreeG1Task::ballDisturbanceRecovery:
        result.task =
            metalrobo::makeUnitreeG1BallDisturbanceRecoveryTaskPack(
                surface, result.observations, result.reset);
        break;
    case metalrobo::UnitreeG1Task::ballDodge:
        result.task = metalrobo::makeUnitreeG1BallDodgeTaskPack(
            surface, result.observations, result.reset);
        break;
    case metalrobo::UnitreeG1Task::developmentalRecovery:
        result.task = metalrobo::
            makeUnitreeG1DevelopmentalRecoveryTaskPack(
                surface, result.observations, result.reset);
        configureG1RecoveryFixture(result);
        break;
    case metalrobo::UnitreeG1Task::adultLocomotion:
        result.task = metalrobo::
            makeUnitreeG1AdultLocomotionTaskPack(
                surface, result.observations, result.reset);
        break;
    }
    return result;
}

metalrobo::LocomotionWorldCompileDiagnostics compileTaskWorld(
    const TaskWorldFixture& authored,
    metalrobo::CompiledLocomotionWorld& compiled
) {
    return metalrobo::compileLocomotionWorld(
        authored, authored.articulationIndex, authored.task,
        authored.actuators, authored.observations, authored.reset, compiled);
}

struct InteractionRuntimeEvidence {
    std::string device;
    double gpuMilliseconds = 0.0;
    float firstReward = 0.0f;
};

InteractionRuntimeEvidence runInteractionRuntimeProbe(
    const metalrobo::LocomotionWorld& authored,
    const metalrobo::CompiledWorld& world,
    const metalrobo::CompiledTaskProgram& program,
    const float framesPerSecond,
    const bool expectPhysicalTargets
) {
    constexpr std::size_t controlStepCount = 6u;
    const std::size_t nq = world.nq();
    const std::size_t nv = world.nv();
    const MRArticulationGPU& articulation =
        authored.model.articulations[world.articulationIndex()];
    std::vector<float> initialQ(nq);
    std::vector<float> initialV(nv);
    std::copy_n(
        authored.model.defaultQ.begin() + articulation.qOffset,
        nq,
        initialQ.begin()
    );
    std::copy_n(
        authored.model.defaultV.begin() + articulation.vOffset,
        nv,
        initialV.begin()
    );
    std::vector<float> actions(
        controlStepCount * program.layout().actionCount,
        0.0f
    );
    std::vector<std::uint32_t> resetMasks(controlStepCount, 0u);
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = controlStepCount,
        .initialQ = initialQ,
        .initialV = initialV,
        .actions = actions,
        .policyRevision = 19u,
        .resetMasks = resetMasks,
        .initialSceneBodies = authored.sceneBodies,
    };
    const metalrobo::MetalWorldStepConfig config{
        .timestepSeconds = 1.0f / framesPerSecond,
        .physicsSubsteps = 4u,
        .solverMode = metalrobo::MetalWorldSolverMode::temporalCone,
        .actuationMode =
            metalrobo::MetalWorldActuationMode::implicitPositionDrive,
        .taskProgram = program,
        .taskSeed = 0x41524459u,
        .velocityIterations = 2u,
        .finalVelocityIterations = 1u,
        .ccdMode = metalrobo::MetalWorldCCDMode::disabled,
        .deterministic = true,
        .warmStart = true,
    };
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResult first;
    metalrobo::MetalWorldResult replay;
    const metalrobo::MetalWorldDiagnostics firstStatus =
        context.run(world, batch, config, first);
    const metalrobo::MetalWorldDiagnostics replayStatus =
        context.run(world, batch, config, replay);
    const std::size_t actorSize =
        program.layout().actorObservationSize;
    const bool expectContactIntent =
        program.layout().interactionContactCount != 0u;
    if (!firstStatus.succeeded() || !replayStatus.succeeded() ||
        firstStatus.successfulStepCount != controlStepCount ||
        replayStatus.successfulStepCount != controlStepCount ||
        firstStatus.failedStepCount != 0u ||
        replayStatus.failedStepCount != 0u ||
        first.actorObservations.size() !=
            controlStepCount * actorSize ||
        first.transitions.size() != controlStepCount) {
        fail(
            "native interaction rollout failed: " +
            firstStatus.message + " " + replayStatus.message
        );
    }
    if (first.actorObservations != replay.actorObservations ||
        first.finalQ != replay.finalQ ||
        first.finalV != replay.finalV ||
        first.transitions.size() != replay.transitions.size() ||
        std::memcmp(
            first.transitions.data(),
            replay.transitions.data(),
            first.transitions.size() * sizeof(MRTaskTransitionGPU)
        ) != 0) {
        fail("native interaction rollout was not bit-identical on replay");
    }
    if (!std::ranges::all_of(
            first.actorObservations,
            [](const float value) { return std::isfinite(value); }
        ) ||
        !std::ranges::all_of(
            first.transitions,
            [](const MRTaskTransitionGPU& transition) {
                return std::isfinite(transition.rewardAndState.x);
            }
        )) {
        fail("native interaction rollout published non-finite learning data");
    }

    std::size_t progressIndex = actorSize;
    std::size_t confidenceIndex = actorSize;
    std::vector<std::size_t> physicalTargetIndices;
    std::vector<std::size_t> physicalValidityIndices;
    const auto operators = program.actorOperators();
    for (std::size_t index = 0u; index < operators.size(); ++index) {
        const MRTaskObservationOperatorGPU& operation = operators[index];
        if (operation.source.x == MR_TASK_OBSERVE_INTERACTION_PHASE &&
            operation.source.z == 2u) {
            progressIndex = index;
        } else if (
            operation.source.x == MR_TASK_OBSERVE_INTERACTION_CONTACT_MODE &&
            operation.source.z == 1u
        ) {
            confidenceIndex = index;
        } else if (
            operation.source.x == MR_TASK_OBSERVE_INTERACTION_CONTACT_TARGET
        ) {
            physicalTargetIndices.push_back(index);
        } else if (
            operation.source.x ==
                MR_TASK_OBSERVE_INTERACTION_CONTACT_VALIDITY
        ) {
            physicalValidityIndices.push_back(index);
        }
    }
    if (progressIndex == actorSize ||
        (expectContactIntent &&
         (confidenceIndex == actorSize ||
          physicalTargetIndices.empty() ||
          physicalValidityIndices.empty()))) {
        fail("native interaction probe could not resolve appended observations");
    }
    const float firstProgress = first.actorObservations[progressIndex];
    const float lastProgress = first.actorObservations[
        (controlStepCount - 1u) * actorSize + progressIndex
    ];
    bool observedPredictionConfidence = false;
    bool observedPhysicalTarget = false;
    bool observedPhysicalValidity = false;
    for (std::size_t step = 0u; step < controlStepCount; ++step) {
        const std::size_t base = step * actorSize;
        if (expectContactIntent) {
            const float confidence =
                first.actorObservations[base + confidenceIndex];
            observedPredictionConfidence |= confidence > 0.0f;
        }
        for (const std::size_t targetIndex : physicalTargetIndices) {
            const float target =
                first.actorObservations[base + targetIndex];
            observedPhysicalTarget |= target != 0.0f;
            if (!expectPhysicalTargets && target != 0.0f) {
                fail(
                    "predicted ARDY contact published fabricated physical targets"
                );
            }
        }
        for (const std::size_t validityIndex : physicalValidityIndices) {
            const float validity =
                first.actorObservations[base + validityIndex];
            observedPhysicalValidity |= validity == 1.0f;
            if (!expectPhysicalTargets && validity != 0.0f) {
                fail(
                    "predicted ARDY contact published fabricated physical validity"
                );
            }
        }
    }
    if (!(lastProgress > firstProgress) ||
        (expectContactIntent && !observedPredictionConfidence)) {
        fail("native interaction reference did not advance through frames");
    }
    if (expectPhysicalTargets &&
        (!observedPhysicalTarget || !observedPhysicalValidity)) {
        fail("native interaction reference did not publish physical targets");
    }
    return {
        .device = firstStatus.deviceName,
        .gpuMilliseconds = firstStatus.gpuElapsedMilliseconds,
        .firstReward = first.transitions.front().rewardAndState.x,
    };
}

metalrobo::InteractionPack makeG1InteractionFixture(
    const TaskWorldFixture& authored
) {
    constexpr std::uint32_t frameCount = 3u;
    constexpr std::uint32_t contactCount = 2u;
    metalrobo::InteractionPack pack{
        .id = "numilab_g1_weight_shift",
        .sourceRepository = "MetalRobo",
        .sourceRevision = "f95c488-fixture",
        .license = "Apache-2.0",
        .coordinateFrame = metalrobo::kInteractionCoordinateFrame,
        .contactTracks = {
            {
                .id = "left_foot",
                .taskContactGroup = "left_foot_contact",
                .counterpart = authored.task.terrain.body,
            },
            {
                .id = "right_foot",
                .taskContactGroup = "right_foot_contact",
                .counterpart = authored.task.terrain.body,
            },
        },
    };
    pack.jointNames.reserve(authored.task.actions.size());
    for (const metalrobo::TaskActionBinding& action :
         authored.task.actions) {
        pack.jointNames.push_back(action.actuator);
    }

    metalrobo::InteractionClip clip{
        .id = "weight_shift_left_lift_right",
        .desiredOutcome =
            "Shift weight onto the left foot and lift the right foot while balanced.",
        .framesPerSecond = 50.0f,
        .frameCount = frameCount,
        .loop = false,
    };
    clip.rootTargets.reserve(
        frameCount * metalrobo::kInteractionRootTargetCount
    );
    clip.jointTargets.reserve(
        frameCount * authored.task.actions.size()
    );
    for (std::uint32_t frame = 0u; frame < frameCount; ++frame) {
        const MRBodyPropertiesGPU& rootBody =
            authored.model.bodies[
                authored.model.articulations.front().rootBody
            ];
        const std::array<float, 7u> rootLinkPose{
            authored.model.defaultQ[0] - rootBody.centerOfMass.x,
            authored.model.defaultQ[1] - rootBody.centerOfMass.y,
            authored.model.defaultQ[2] - rootBody.centerOfMass.z,
            authored.model.defaultQ[3],
            authored.model.defaultQ[4],
            authored.model.defaultQ[5],
            authored.model.defaultQ[6],
        };
        clip.rootTargets.insert(
            clip.rootTargets.end(),
            rootLinkPose.begin(),
            rootLinkPose.end()
        );
        clip.jointTargets.insert(
            clip.jointTargets.end(),
            authored.model.defaultQ.begin() + 7,
            authored.model.defaultQ.begin() + 36
        );
    }
    // A small feasible guide accompanies the contact transition. Contact is
    // still the primary target and is measured from the solver.
    clip.jointTargets[1u * 29u + 1u] += 0.04f;
    clip.jointTargets[1u * 29u + 7u] += 0.04f;
    clip.jointTargets[2u * 29u + 1u] += 0.08f;
    clip.jointTargets[2u * 29u + 7u] += 0.08f;
    clip.jointTargets[2u * 29u + 9u] += 0.12f;

    const std::size_t sampleCount = frameCount * contactCount;
    clip.contactModes = {
        static_cast<std::uint32_t>(
            metalrobo::InteractionContactMode::stick
        ),
        static_cast<std::uint32_t>(
            metalrobo::InteractionContactMode::stick
        ),
        static_cast<std::uint32_t>(
            metalrobo::InteractionContactMode::stick
        ),
        static_cast<std::uint32_t>(
            metalrobo::InteractionContactMode::stick
        ),
        static_cast<std::uint32_t>(
            metalrobo::InteractionContactMode::stick
        ),
        static_cast<std::uint32_t>(
            metalrobo::InteractionContactMode::free
        ),
    };
    clip.contactFeatureMasks.assign(sampleCount, 0u);
    clip.contactSampleFlags.assign(
        sampleCount,
        metalrobo::interactionSamplePhysicsCertified
    );
    clip.contactConfidence.assign(sampleCount, 1.0f);
    clip.contactTargets.assign(
        sampleCount * metalrobo::kInteractionContactFeatureCount,
        0.0f
    );
    clip.contactTolerances.assign(
        sampleCount * metalrobo::kInteractionContactFeatureCount,
        0.0f
    );
    const auto setSupportTarget = [&clip](
        const std::uint32_t frame,
        const std::uint32_t track,
        const float load
    ) {
        const std::size_t sample = frame * 2u + track;
        constexpr std::uint32_t mask =
            (1u << 2u) | (1u << 6u) | (1u << 7u) |
            (1u << 8u) | (1u << 9u) | (1u << 10u) |
            (1u << 11u) | (1u << 12u);
        clip.contactFeatureMasks[sample] = mask;
        const std::size_t base =
            sample * metalrobo::kInteractionContactFeatureCount;
        clip.contactTargets[base + 2u] = load;
        clip.contactTargets[base + 8u] = 0.020f;
        for (std::uint32_t cell = 0u; cell < 4u; ++cell) {
            clip.contactTargets[base + 9u + cell] =
                load / 0.020f;
        }
        clip.contactTolerances[base + 2u] = 75.0f;
        clip.contactTolerances[base + 6u] = 0.02f;
        clip.contactTolerances[base + 7u] = 0.02f;
        clip.contactTolerances[base + 8u] = 0.01f;
        for (std::uint32_t cell = 0u; cell < 4u; ++cell) {
            clip.contactTolerances[base + 9u + cell] = 5000.0f;
        }
    };
    setSupportTarget(0u, 0u, 350.0f);
    setSupportTarget(0u, 1u, 350.0f);
    setSupportTarget(1u, 0u, 550.0f);
    setSupportTarget(1u, 1u, 150.0f);
    setSupportTarget(2u, 0u, 650.0f);
    pack.clips.push_back(std::move(clip));
    return pack;
}

std::uint64_t compileImportedRobotFixture() {
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
    TaskWorldFixture authored;
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
    metalrobo::appendLocomotionSurface(
        authored.model,
        authored.sceneBodies,
        metalrobo::LocomotionSurface::ground
    );

    authored.task.id = "generic_locomotion_fixture";
    authored.task.actions = {{
        .actuator = "hip",
    }};
    authored.actuators = {{
        .id = "hip",
        .kind = metalrobo::RobotActuatorKind::jointPosition,
        .target = "hip",
        .scale = 0.2f,
    }};
    authored.observations.actorHistoryLength = 2u;
    authored.observations.actorFrame = {
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
    authored.observations.actorCurrent = {{
        .source = metalrobo::TaskObservationSource::rootHeight,
    }};
    authored.observations.critic = {{
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

    metalrobo::CompiledLocomotionWorld compiled;
    const metalrobo::LocomotionWorldCompileDiagnostics status =
        compileTaskWorld(authored, compiled);
    if (!status.world.succeeded() ||
        !status.task.succeeded()) {
        fail(
            "generic imported locomotion compile failed: " +
            status.world.message + " " +
            status.task.element + ": " +
            status.task.message
        );
    }
    const metalrobo::TaskProgramLayout& layout =
        compiled.task.layout();
    if (layout.actionCount != 1u ||
        layout.actorFrameSize != 3u ||
        layout.actorObservationSize != 7u ||
        layout.criticFrameSize != 1u ||
        layout.criticHistoryLength != 1u ||
        layout.criticObservationSize != 7u ||
        layout.contactMetricCount != 12u ||
        compiled.task.header().counts3.w != 1u ||
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

std::uint64_t checkWorldPackLocomotionImport() {
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
    metalrobo::RealityProgramPack loadedReality;
    const metalrobo::LearningPackResult realityRead =
        metalrobo::readRealityProgramPack(files.reality, loadedReality);
    if (!realityRead.succeeded() ||
        loadedReality.id != "franka_reality" ||
        loadedReality.program.id != reality.id ||
        loadedReality.program.variations.size() != 1u ||
        loadedReality.program.variations.front().id !=
            "pick_object_mass" ||
        loadedReality.program.variations.front().targetId !=
            "pick_object" ||
        loadedReality.sourceProgramFingerprint !=
            family.program.fingerprint ||
        loadedReality.reset.operators.size() !=
            frankaReset.operators.size()) {
        fail("RealityProgramPack did not preserve executable ownership");
    }
    const MRTaskRolloutConfigC profile{
        .environment_count = 1u,
        .physics_substeps = 4u,
        .velocity_iterations = 4u,
        .final_velocity_iterations = 2u,
        .control_timestep_seconds = 1.0f / 60.0f,
        .seed = 0x574f524c44ull,
    };
    const std::string worldPath = files.world.string();
    const std::string taskPath = files.task.string();
    const std::string actuatorPath = files.actuators.string();
    const std::string sensorPath = files.sensors.string();
    const std::string realityPath = files.reality.string();
    const MRRunManifestC manifest{
        .profile = profile,
        .source = MR_RUN_SOURCE_WORLD_PACK,
        .world_pack_path = worldPath.c_str(),
        .task_pack_path = taskPath.c_str(),
        .robot_actuator_pack_path = actuatorPath.c_str(),
        .sensor_program_pack_path = sensorPath.c_str(),
        .reality_program_pack_path = realityPath.c_str(),
    };
    std::unique_ptr<MRTaskRolloutHandle, decltype(&mr_task_rollout_destroy)>
        rollout{
            mr_create_task_rollout(&manifest),
            &mr_task_rollout_destroy,
        };
    if (!rollout) {
        fail(
            "WorldPack runtime creation failed: " +
            std::string{mr_last_error()}
        );
    }
    const MRTaskRolloutLayoutC layout =
        mr_task_rollout_layout(rollout.get());
    if (layout.reality_fingerprint == 0u ||
        layout.action_count != frankaActuators.actuators.size() ||
        mr_task_rollout_set_state_readback(rollout.get(), 1u) != 0) {
        fail(
            "typed WorldPack rollout could not enable inspection: " +
            std::string{mr_last_error()}
        );
    }
    constexpr std::uint32_t kProbeSteps = 64u;
    const auto runProbe = [&](const std::uint32_t actionIndex,
                              const float command,
                              const std::uint64_t revision) {
        if (mr_task_rollout_reset(rollout.get(), profile.seed) != 0) {
            fail("typed actuator rollout reset failed: " +
                 std::string{mr_last_error()});
        }
        std::vector<float> actions(
            static_cast<std::size_t>(kProbeSteps) * layout.action_count,
            0.0f
        );
        if (actionIndex != MR_INVALID_INDEX) {
            for (std::uint32_t step = 0u; step < kProbeSteps; ++step) {
                actions[static_cast<std::size_t>(step) *
                    layout.action_count + actionIndex] = command;
            }
        }
        MRTaskRolloutAdvanceC advance{};
        if (mr_task_rollout_advance(
                rollout.get(), actions.data(), actions.size(), nullptr, 0u,
                kProbeSteps, revision, 0u, &advance
            ) != 0 ||
            advance.failed_environment_steps != 0u ||
            advance.successful_environment_steps != kProbeSteps) {
            fail("typed actuator Metal execution failed: " +
                 std::string{mr_last_error()});
        }
        const float* finalQ = mr_task_rollout_final_q(rollout.get());
        if (finalQ == nullptr) {
            fail("typed actuator rollout did not publish final state");
        }
        std::vector<float> result(finalQ, finalQ + layout.nq);
        if (!std::ranges::all_of(result, [](const float value) {
                return std::isfinite(value);
            })) {
            fail("typed actuator rollout produced a non-finite state");
        }
        return result;
    };
    const std::vector<float> baseline =
        runProbe(MR_INVALID_INDEX, 0.0f, 1u);
    const auto maximumDifference = [&baseline](
        const std::vector<float>& candidate
    ) {
        float difference = 0.0f;
        for (std::size_t index = 0u; index < baseline.size(); ++index) {
            difference = std::max(
                difference,
                std::abs(candidate[index] - baseline[index])
            );
        }
        return difference;
    };
    const std::array<std::pair<std::uint32_t, float>, 9u> probes{{
        {0u, 0.45f},
        {1u, 0.35f},
        {2u, 0.40f},
        {3u, 0.75f},
        {4u, 0.75f},
        {5u, 0.75f},
        {6u, 0.75f},
        {7u, 0.75f},
        {8u, 0.75f},
    }};
    std::array<float, probes.size()> physicalDifferences{};
    for (std::uint32_t probe = 0u; probe < probes.size(); ++probe) {
        const std::vector<float> actuated = runProbe(
            probes[probe].first,
            probes[probe].second,
            2u + probe
        );
        physicalDifferences[probe] = maximumDifference(actuated);
        if (!(physicalDifferences[probe] > 1.0e-5f)) {
            fail("typed actuator did not alter the physical state: " +
                 frankaActuators.actuators[probes[probe].first].id);
        }
    }
    const float minimumBodyWrenchDifference = *std::min_element(
        physicalDifferences.begin() + 3u,
        physicalDifferences.end()
    );
    std::cout
        << "typed_actuator_pipeline=ok"
        << " velocity_delta=" << physicalDifferences[0]
        << " effort_delta=" << physicalDifferences[1]
        << " tendon_delta=" << physicalDifferences[2]
        << " minimum_body_wrench_delta=" << minimumBodyWrenchDifference
        << " body_wrench_components=6"
        << " steps_per_probe=" << kProbeSteps
        << " failed=0\n";
    return pack.contentHash;
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        if (argc > 2) {
            fail("usage: metalrobo_task_program_check [interactionpack]");
        }
        std::uint64_t externalInteractionArtifact = 0u;
        std::uint64_t externalInteractionTask = 0u;
        InteractionRuntimeEvidence fixtureRuntime;
        InteractionRuntimeEvidence externalRuntime;
        bool externalNativeExecuted = false;
        TaskWorldFixture authored =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::terrain
            );
        metalrobo::CompiledLocomotionWorld compiledWorld;
        const metalrobo::LocomotionWorldCompileDiagnostics
            compileStatus = compileTaskWorld(authored, compiledWorld);
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
        const metalrobo::TaskProgramLayout& layout =
            program.layout();
        if (layout.actionCount != 29u ||
            layout.actorFrameSize != 98u ||
            layout.actorHistoryLength != 1u ||
            layout.actorObservationSize != 98u ||
            layout.criticFrameSize != 98u ||
            layout.criticHistoryLength != 1u ||
            layout.criticObservationSize != 98u ||
            layout.contactMetricCount != 45u ||
            layout.delayStateCount != 3u) {
            fail("compiled G1 task layout changed");
        }
        if (program.worldFingerprint() != world.fingerprint() ||
            world.capacities().candidatePairs != 192u ||
            world.capacities().rawContacts != 128u ||
            world.capacities().manifolds != 32u ||
            world.capacities().constraintBlocks != 64u ||
            world.capacities().constraintRows != 192u ||
            world.capacities().endpointRuntimeRecords != 128u ||
            world.capacities().articulationPointQueries != 128u ||
            program.header().root.y != 0u ||
            std::abs(
                program.header().rootReference.z -
                0.076030060304f
            ) > 1.0e-6f ||
            program.header().counts0.w != 4u ||
            program.header().counts1.w != 19u ||
            program.header().counts2.x != 2u ||
            program.header().counts2.y != 0u ||
            program.header().articulation.w != 1u ||
            program.contactGroups().size() < 2u ||
            program.contactGroups()[0].supportPatch.x != 2u ||
            program.contactGroups()[0].supportPatch.y != 2u ||
            program.contactGroups()[0].supportPatch.z != 4u ||
            program.contactGroups()[1].supportPatch.x != 2u ||
            program.contactGroups()[1].supportPatch.y != 2u ||
            program.contactGroups()[1].supportPatch.z != 4u ||
            program.header().commandLower.x != 0.0f ||
            program.header().commandLower.y != 0.0f ||
            program.header().commandLower.z != 0.0f ||
            program.header().commandUpper.x != 0.0f ||
            program.header().commandUpper.y != 0.0f ||
            program.header().commandUpper.z != 0.0f ||
            std::abs(
                program.header().commandUpper.w - 2.0f
            ) > 1.0e-6f ||
            program.terminationOperators()[0].parameters.y !=
                -2.0f ||
            program.terminationOperators()[1].parameters.y !=
                -2.0f ||
            !program.randomizationOperators().empty() ||
            program.terrainSampleOffsets().size() != 187u ||
            program.terrainResetTranslations().size() != 11u) {
            fail("compiled G1 task tables are incomplete");
        }

        TemporaryPackFiles interactionFiles;
        const metalrobo::InteractionPack authoredInteraction =
            makeG1InteractionFixture(authored);
        const metalrobo::LearningPackResult interactionWrite =
            metalrobo::writeInteractionPack(
                authoredInteraction,
                interactionFiles.interaction
            );
        metalrobo::InteractionPack loadedInteraction;
        const metalrobo::LearningPackResult interactionRead =
            metalrobo::readInteractionPack(
                interactionFiles.interaction,
                loadedInteraction
            );
        if (!interactionWrite.succeeded() ||
            !interactionRead.succeeded() ||
            loadedInteraction.id != authoredInteraction.id ||
            loadedInteraction.clips.front().desiredOutcome !=
                authoredInteraction.clips.front().desiredOutcome ||
            loadedInteraction.clips.front().jointTargets !=
                authoredInteraction.clips.front().jointTargets ||
            loadedInteraction.clips.front().contactTargets !=
                authoredInteraction.clips.front().contactTargets) {
            fail("InteractionPack round trip changed contact-first intent");
        }

        metalrobo::TaskPack interactionTask = authored.task;
        metalrobo::TaskObservationProgram interactionObservations =
            authored.observations;
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            interactionObservations.actorCurrent.push_back({
                .source =
                    metalrobo::TaskObservationSource::interactionPhase,
                .component = component,
            });
        }
        interactionObservations.actorCurrent.push_back({
            .source = metalrobo::TaskObservationSource::
                interactionJointPositionError,
            .target = "left_hip_roll_joint",
        });
        for (const std::string_view track :
             {std::string_view{"left_foot"},
              std::string_view{"right_foot"}}) {
            interactionObservations.actorCurrent.push_back({
                .source = metalrobo::TaskObservationSource::
                    interactionContactMode,
                .target = std::string{track},
                .component = 0u,
            });
            interactionObservations.actorCurrent.push_back({
                .source = metalrobo::TaskObservationSource::
                    interactionContactMode,
                .target = std::string{track},
                .component = 1u,
            });
            constexpr std::array<std::uint32_t, 8u> features{
                2u, 6u, 7u, 8u, 9u, 10u, 11u, 12u,
            };
            for (const std::uint32_t component : features) {
                interactionObservations.actorCurrent.push_back({
                    .source = metalrobo::TaskObservationSource::
                        interactionContactTarget,
                    .target = std::string{track},
                    .component = component,
                });
                interactionObservations.actorCurrent.push_back({
                    .source = metalrobo::TaskObservationSource::
                        interactionContactValidity,
                    .target = std::string{track},
                    .component = component,
                });
            }
        }
        interactionTask.rewards.push_back({
            .operation = metalrobo::TaskRewardOperator::
                interactionJointTracking,
            .weight = 0.5f,
            .parameters = {0.25f, 0.0f, 0.0f, 0.0f},
        });
        interactionTask.rewards.push_back({
            .operation = metalrobo::TaskRewardOperator::
                interactionContactTracking,
            .sourceGroup = "left_foot_contact",
            .target = "left_foot",
            .weight = 1.0f,
            .parameters = {0.6f, 1.0f, 0.0f, 0.0f},
        });
        interactionTask.rewards.push_back({
            .operation = metalrobo::TaskRewardOperator::
                interactionContactTracking,
            .sourceGroup = "right_foot_contact",
            .target = "right_foot",
            .weight = 1.0f,
            .parameters = {0.6f, 1.0f, 0.0f, 0.0f},
        });
        interactionTask.interactionResetPhaseFraction = 0.75f;
        metalrobo::CompiledTaskProgram interactionProgram;
        const metalrobo::TaskCompileDiagnostics interactionStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                loadedInteraction,
                "weight_shift_left_lift_right",
                world,
                interactionProgram
            );
        const auto& interactionLayout = interactionProgram.layout();
        if (!interactionStatus.succeeded() ||
            interactionLayout.actorFrameSize !=
                layout.actorFrameSize ||
            interactionLayout.actorObservationSize !=
                layout.actorObservationSize + 40u ||
            interactionLayout.interactionFrameCount != 3u ||
            interactionLayout.interactionContactCount != 2u ||
            interactionProgram.header().interaction.x != 3u ||
            interactionProgram.header().interaction.y != 29u ||
            interactionProgram.header().interaction.z != 2u ||
            std::abs(
                interactionProgram.header().interactionTiming.z - 0.1f
            ) > 1.0e-6f ||
            std::abs(
                interactionProgram.header().interactionTiming.w - 0.75f
            ) > 1.0e-6f ||
            std::abs(
                interactionProgram.header().interactionCurriculum.x - 0.75f
            ) > 1.0e-6f ||
            std::abs(
                interactionProgram.header().interactionCurriculum.y - 0.75f
            ) > 1.0e-6f ||
            interactionProgram.header().counts3.z != 2u ||
            interactionProgram.header().counts3.w != 40u ||
            (interactionProgram.header().schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) == 0u ||
            (interactionProgram.header().schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) == 0u ||
            interactionProgram.interactionRootTargets().size() != 21u ||
            interactionProgram.interactionJointTargets().size() != 87u ||
            interactionProgram.interactionContacts().size() != 2u ||
            interactionProgram.interactionSamples().size() != 6u ||
            interactionProgram.interactionContactTargets().size() != 78u ||
            interactionProgram.interactionContactTolerances().size() != 78u ||
            interactionProgram.interactionContacts()[0].binding.x != 0u ||
            interactionProgram.interactionContacts()[1].binding.x != 1u ||
            interactionProgram.interactionSamples()[5].metadata.x !=
                MR_TASK_INTERACTION_CONTACT_FREE ||
            interactionProgram.interactionSamples()[0].metadata.z !=
                metalrobo::interactionSamplePhysicsCertified ||
            interactionProgram.interactionContactTargets()[2u] != 350.0f ||
            interactionProgram.interactionContactTargets()[
                4u * metalrobo::kInteractionContactFeatureCount + 2u
            ] != 650.0f) {
            fail(
                "G1 InteractionPack did not compile into exact native reference tables"
            );
        }
        metalrobo::TaskPack splitCurriculumTask = interactionTask;
        splitCurriculumTask.interactionResetPhaseProbability = 0.2f;
        splitCurriculumTask.interactionResetMaximumPhase = 0.85f;
        metalrobo::CompiledTaskProgram splitCurriculumProgram;
        const auto splitCurriculumStatus = metalrobo::compileTaskProgram(
            splitCurriculumTask,
            authored.actuators,
            interactionObservations,
            authored.reset,
            loadedInteraction,
            "weight_shift_left_lift_right",
            world,
            splitCurriculumProgram
        );
        if (!splitCurriculumStatus.succeeded() ||
            std::abs(
                splitCurriculumProgram.header().interactionCurriculum.x - 0.2f
            ) > 1.0e-6f ||
            std::abs(
                splitCurriculumProgram.header().interactionCurriculum.y - 0.85f
            ) > 1.0e-6f) {
            fail(
                "Interaction reset probability and depth did not compile "
                "independently: " + splitCurriculumStatus.message
            );
        }
        std::uint32_t interactionPhase = 0u;
        std::uint32_t interactionJoint = 0u;
        std::uint32_t interactionMode = 0u;
        std::uint32_t interactionTarget = 0u;
        std::uint32_t interactionValidity = 0u;
        for (const MRTaskObservationOperatorGPU& operation :
             interactionProgram.actorOperators()) {
            interactionPhase += operation.source.x ==
                MR_TASK_OBSERVE_INTERACTION_PHASE ? 1u : 0u;
            interactionJoint += operation.source.x ==
                MR_TASK_OBSERVE_INTERACTION_JOINT_POSITION_ERROR ? 1u : 0u;
            interactionMode += operation.source.x ==
                MR_TASK_OBSERVE_INTERACTION_CONTACT_MODE ? 1u : 0u;
            interactionTarget += operation.source.x ==
                MR_TASK_OBSERVE_INTERACTION_CONTACT_TARGET ? 1u : 0u;
            interactionValidity += operation.source.x ==
                MR_TASK_OBSERVE_INTERACTION_CONTACT_VALIDITY ? 1u : 0u;
        }
        std::uint32_t interactionJointReward = 0u;
        std::uint32_t interactionContactReward = 0u;
        for (const MRTaskRewardOperatorGPU& operation :
             interactionProgram.rewardOperators()) {
            interactionJointReward += operation.source.x ==
                MR_TASK_REWARD_INTERACTION_JOINT_TRACKING ? 1u : 0u;
            interactionContactReward += operation.source.x ==
                MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING ? 1u : 0u;
        }
        if (interactionPhase != 3u || interactionJoint != 1u ||
            interactionMode != 4u || interactionTarget != 16u ||
            interactionValidity != 16u ||
            interactionJointReward != 1u ||
            interactionContactReward != 2u) {
            fail("contact-first actor or reward operators are incomplete");
        }
        metalrobo::CompiledTaskProgram repeatedInteraction;
        const auto repeatedInteractionStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                loadedInteraction,
                "weight_shift_left_lift_right",
                world,
                repeatedInteraction
            );
        if (!repeatedInteractionStatus.succeeded() ||
            repeatedInteraction.fingerprint() !=
                interactionProgram.fingerprint()) {
            fail("interaction task compilation is not deterministic");
        }
        const std::uint64_t preservedInteraction =
            repeatedInteraction.fingerprint();
        metalrobo::InteractionPack brokenInteraction =
            loadedInteraction;
        brokenInteraction.contactTracks[0].taskContactGroup =
            "missing_contact_group";
        const auto brokenInteractionStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                brokenInteraction,
                "weight_shift_left_lift_right",
                world,
                repeatedInteraction
            );
        if (brokenInteractionStatus.status !=
                metalrobo::TaskCompileStatus::unresolvedSemantic ||
            repeatedInteraction.fingerprint() != preservedInteraction) {
            fail(
                "unresolved interaction semantics were not transactionally rejected"
            );
        }
        metalrobo::InteractionPack unsafeInteraction =
            loadedInteraction;
        unsafeInteraction.clips.front().jointTargets.front() = 100.0f;
        const auto unsafeInteractionStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                unsafeInteraction,
                "weight_shift_left_lift_right",
                world,
                repeatedInteraction
            );
        if (unsafeInteractionStatus.status !=
                metalrobo::TaskCompileStatus::invalidPack ||
            repeatedInteraction.fingerprint() != preservedInteraction) {
            fail(
                "out-of-envelope interaction target was not transactionally rejected"
            );
        }
        metalrobo::InteractionPack unsafeVelocity =
            loadedInteraction;
        unsafeVelocity.clips.front().jointTargets[
            authored.task.actions.size()
        ] = 2.0f;
        const auto unsafeVelocityStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                unsafeVelocity,
                "weight_shift_left_lift_right",
                world,
                repeatedInteraction
            );
        if (unsafeVelocityStatus.status !=
                metalrobo::TaskCompileStatus::invalidPack ||
            repeatedInteraction.fingerprint() != preservedInteraction) {
            fail(
                "out-of-envelope interaction velocity was not transactionally rejected"
            );
        }
        metalrobo::InteractionPack wrongCounterpart =
            loadedInteraction;
        wrongCounterpart.contactTracks.front().counterpart =
            "missing_support_surface";
        const auto wrongCounterpartStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                wrongCounterpart,
                "weight_shift_left_lift_right",
                world,
                repeatedInteraction
            );
        if (wrongCounterpartStatus.status !=
                metalrobo::TaskCompileStatus::unresolvedSemantic ||
            repeatedInteraction.fingerprint() != preservedInteraction) {
            fail(
                "interaction counterpart mismatch was not transactionally rejected"
            );
        }
        const auto missingInteractionStatus =
            metalrobo::compileTaskProgram(
                interactionTask,
                authored.actuators,
                interactionObservations,
                authored.reset,
                world,
                repeatedInteraction
            );
        if (missingInteractionStatus.status !=
                metalrobo::TaskCompileStatus::invalidPack ||
            repeatedInteraction.fingerprint() != preservedInteraction) {
            fail(
                "interaction operators compiled without a selected reference"
            );
        }
        if (argc == 2) {
            fixtureRuntime = runInteractionRuntimeProbe(
                authored,
                world,
                interactionProgram,
                authoredInteraction.clips.front().framesPerSecond,
                true
            );
            metalrobo::InteractionPack externalInteraction;
            const auto externalRead = metalrobo::readInteractionPack(
                argv[1],
                externalInteraction
            );
            if (!externalRead.succeeded() ||
                externalInteraction.jointNames.size() != 29u ||
                externalInteraction.clips.size() != 1u) {
                fail(
                    "external InteractionPack did not satisfy the G1 contract: " +
                    externalRead.message +
                    " joints=" + std::to_string(
                        externalInteraction.jointNames.size()
                    ) +
                    " contacts=" + std::to_string(
                        externalInteraction.contactTracks.size()
                    ) +
                    " clips=" + std::to_string(
                        externalInteraction.clips.size()
                    )
                );
            }
            const metalrobo::InteractionClip& externalClip =
                externalInteraction.clips.front();
            const bool predictedOnly = std::ranges::all_of(
                externalClip.contactSampleFlags,
                [](const std::uint32_t flags) {
                    return flags ==
                        metalrobo::interactionSamplePredicted;
                }
            );
            const bool noFabricatedPhysicalTargets =
                std::ranges::all_of(
                    externalClip.contactFeatureMasks,
                    [](const std::uint32_t mask) {
                        return mask == 0u;
                    }
                );
            if (externalClip.frameCount < 2u ||
                !predictedOnly || !noFabricatedPhysicalTargets) {
                fail(
                    "external generated contact was mislabeled as physical evidence"
                );
            }
            const bool noAuthoredContact =
                externalInteraction.contactTracks.empty();
            const bool externalUsesGround = noAuthoredContact ||
                std::ranges::all_of(
                externalInteraction.contactTracks,
                [](const metalrobo::InteractionContactTrack& track) {
                    return track.counterpart == "locomotion_ground";
                }
            );
            const bool externalUsesTerrain = std::ranges::all_of(
                externalInteraction.contactTracks,
                [](const metalrobo::InteractionContactTrack& track) {
                    return track.counterpart == "locomotion_terrain";
                }
            );
            if (!externalUsesGround && !externalUsesTerrain) {
                fail(
                    "external InteractionPack mixes or does not name a bundled locomotion surface"
                );
            }
            const metalrobo::LocomotionSurface externalSurface =
                externalUsesGround
                ? metalrobo::LocomotionSurface::ground
                : metalrobo::LocomotionSurface::terrain;
            TaskWorldFixture externalAuthored =
                makeG1TaskWorld(externalSurface);
            metalrobo::CompiledLocomotionWorld externalWorld;
            const auto externalWorldStatus =
                compileTaskWorld(externalAuthored, externalWorld);
            if (!externalWorldStatus.world.succeeded() ||
                !externalWorldStatus.task.succeeded()) {
                fail("external InteractionPack surface world did not compile");
            }
            metalrobo::TaskPack externalTask = interactionTask;
            metalrobo::TaskObservationProgram externalObservations =
                interactionObservations;
            externalTask.terrain = externalAuthored.task.terrain;
            if (noAuthoredContact) {
                const auto isInteractionContactObservation =
                    [](const metalrobo::TaskObservationOperatorSpec& operation) {
                        return operation.source ==
                                metalrobo::TaskObservationSource::
                                    interactionContactMode ||
                            operation.source ==
                                metalrobo::TaskObservationSource::
                                    interactionContactTarget ||
                            operation.source ==
                                metalrobo::TaskObservationSource::
                                    interactionContactValidity;
                    };
                std::erase_if(
                    externalObservations.actorFrame,
                    isInteractionContactObservation
                );
                std::erase_if(
                    externalObservations.actorCurrent,
                    isInteractionContactObservation
                );
                std::erase_if(
                    externalObservations.critic,
                    isInteractionContactObservation
                );
                std::erase_if(
                    externalTask.rewards,
                    [](const metalrobo::TaskRewardOperatorSpec& reward) {
                        return reward.operation ==
                            metalrobo::TaskRewardOperator::
                                interactionContactTracking;
                    }
                );
            }
            metalrobo::CompiledTaskProgram externalProgram;
            const auto externalCompile =
                metalrobo::compileTaskProgram(
                    externalTask,
                    externalAuthored.actuators,
                    externalObservations,
                    externalAuthored.reset,
                    externalInteraction,
                    externalClip.id,
                    externalWorld.world,
                    externalProgram
                );
            if (!externalCompile.succeeded() ||
                externalProgram.layout().interactionFrameCount !=
                    externalClip.frameCount ||
                externalProgram.layout().interactionContactCount !=
                    externalInteraction.contactTracks.size() ||
                externalProgram.interactionJointTargets().size() !=
                    static_cast<std::size_t>(externalClip.frameCount) *
                        29u) {
                fail(
                    "external ARDY interaction did not compile into the G1 task: " +
                    externalCompile.element + ": " +
                    externalCompile.message
                );
            }
            externalInteractionArtifact = externalRead.contentHash;
            externalInteractionTask = externalProgram.fingerprint();
            externalRuntime = runInteractionRuntimeProbe(
                externalAuthored,
                externalWorld.world,
                externalProgram,
                externalClip.framesPerSecond,
                false
            );
            const MRTaskRolloutConfigC rolloutConfig{
                .environment_count = 1u,
                .physics_substeps = 4u,
                .velocity_iterations = 2u,
                .final_velocity_iterations = 1u,
                .control_timestep_seconds =
                    1.0f / externalClip.framesPerSecond,
                .seed = 0x41524459u,
            };
            const MRRunManifestC runManifest{
                .profile = rolloutConfig,
                .source = MR_RUN_SOURCE_UNITREE_G1,
                .surface = externalUsesGround
                    ? MR_LOCOMOTION_SURFACE_GROUND
                    : MR_LOCOMOTION_SURFACE_TERRAIN,
                .task = MR_UNITREE_G1_TASK_VELOCITY,
                .teacher_pack_path = argv[1],
                .teacher_clip_id = externalClip.id.c_str(),
            };
            std::unique_ptr<
                MRTaskRolloutHandle,
                decltype(&mr_task_rollout_destroy)
            > rollout{
                mr_create_task_rollout(&runManifest),
                &mr_task_rollout_destroy,
            };
            if (!rollout) {
                fail(
                    "C/Swift interaction rollout boundary failed: " +
                    std::string{mr_last_error()}
                );
            }
            const MRTaskRolloutLayoutC rolloutLayout =
                mr_task_rollout_layout(rollout.get());
            constexpr std::uint32_t rolloutSteps = 6u;
            const std::uint32_t expectedInteractionObservationCount =
                142u + 28u * static_cast<std::uint32_t>(
                    externalInteraction.contactTracks.size()
                );
            std::vector<float> rolloutActions(
                rolloutSteps * rolloutLayout.action_count,
                0.0f
            );
            MRTaskRolloutAdvanceC advance{};
            if (rolloutLayout.action_count != 29u ||
                rolloutLayout.actor_observation_count !=
                    expectedInteractionObservationCount ||
                rolloutLayout.critic_observation_count !=
                    expectedInteractionObservationCount ||
                mr_task_rollout_advance(
                    rollout.get(),
                    rolloutActions.data(),
                    rolloutActions.size(),
                    nullptr,
                    0u,
                    rolloutSteps,
                    23u,
                    0u,
                    &advance
                ) != 0 ||
                advance.successful_environment_steps != rolloutSteps ||
                advance.failed_environment_steps != 0u ||
                mr_task_rollout_actor_observations(rollout.get()) == nullptr ||
                mr_task_rollout_transitions(rollout.get()) == nullptr) {
                fail(
                    "C/Swift interaction rollout did not publish a complete native batch: " +
                    std::string{mr_last_error()}
                );
            }
            externalNativeExecuted = true;
        }
        TaskWorldFixture recovery =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::terrain,
                metalrobo::UnitreeG1Task::disturbanceRecovery
            );
        metalrobo::CompiledLocomotionWorld compiledRecovery;
        const auto recoveryStatus =
            compileTaskWorld(recovery, compiledRecovery);
        if (!recoveryStatus.succeeded() ||
            compiledRecovery.task.layout().actorObservationSize != 98u ||
            compiledRecovery.task.layout().criticObservationSize != 98u ||
            compiledRecovery.task.header().counts2.y != 7u ||
            compiledRecovery.task.header().layout.w != 5u ||
            std::abs(
                compiledRecovery.task.header().dynamics.x - 2.5f
            ) > 1.0e-6f ||
            compiledRecovery.task.header().commandLower.x != 0.0f ||
            compiledRecovery.task.header().commandUpper.x != 0.0f) {
            fail(
                "compiled G1 disturbance-recovery task is incomplete"
            );
        }
        TaskWorldFixture getUp =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::supineGetUpDiscovery
            );
        metalrobo::CompiledLocomotionWorld compiledGetUp;
        const auto getUpStatus = compileTaskWorld(getUp, compiledGetUp);
        if (!getUpStatus.succeeded()) {
            fail(
                "G1 supine get-up task failed to compile: " +
                getUpStatus.task.element + ": " +
                getUpStatus.task.message
            );
        }
        if (compiledGetUp.task.layout().actorFrameSize != 122u ||
            compiledGetUp.task.layout().actorHistoryLength != 5u ||
            compiledGetUp.task.layout().actorObservationSize != 610u ||
            compiledGetUp.task.layout().criticFrameSize != 99u ||
            compiledGetUp.task.layout().criticObservationSize != 990u ||
            compiledGetUp.task.layout().contactMetricCount != 81u ||
            compiledGetUp.task.actionBindings().size() != 29u ||
            compiledGetUp.task.actionBindings()[3].parameters.x < 2.5f ||
            compiledGetUp.task.actionBindings()[15].parameters.x < 3.4f ||
            compiledGetUp.task.header().counts1.w != 11u ||
            compiledGetUp.task.header().counts2.x != 2u ||
            compiledGetUp.task.header().counts2.y != 124u ||
            compiledGetUp.task.header().schedule.z != 8u ||
            compiledGetUp.task.terminationOperators()[0].schedule.x != 4u ||
            compiledGetUp.task.terminationOperators()[0].schedule.y !=
                MR_INVALID_INDEX ||
            compiledGetUp.task.terminationOperators()[1].schedule.x != 4u ||
            compiledGetUp.task.terminationOperators()[1].schedule.y !=
                MR_INVALID_INDEX) {
            fail(
                "compiled G1 supine get-up task is incomplete: actor_frame=" +
                std::to_string(
                    compiledGetUp.task.layout().actorFrameSize
                ) + " actor_history=" +
                std::to_string(
                    compiledGetUp.task.layout().actorHistoryLength
                ) + " actor=" +
                std::to_string(
                    compiledGetUp.task.layout().actorObservationSize
                ) + " critic_frame=" +
                std::to_string(
                    compiledGetUp.task.layout().criticFrameSize
                ) + " critic=" +
                std::to_string(
                    compiledGetUp.task.layout().criticObservationSize
                )
            );
        }
        const auto getUpBandOneOperators = std::ranges::count_if(
            compiledGetUp.task.randomizationOperators(),
            [](const MRTaskRandomizationOperatorGPU& operation) {
                return operation.target.w == 1u;
            }
        );
        if (getUpBandOneOperators != 0u) {
            fail("supine get-up was silently replaced by a developmental reset");
        }
        const auto getUpOutcomes = compiledGetUp.task.outcomes();
        if (getUpOutcomes.size() != 3u ||
            std::ranges::none_of(
                getUpOutcomes,
                [](const metalrobo::CompiledTaskOutcomeSpec& outcome) {
                    return outcome.id == "contact_reward";
                }
            )) {
            fail("supine get-up policy contract changed unexpectedly");
        }

        TaskWorldFixture developmental =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::developmentalRecovery
            );
        metalrobo::CompiledLocomotionWorld compiledDevelopmental;
        const auto developmentalStatus = compileTaskWorld(
            developmental, compiledDevelopmental);
        if (!developmentalStatus.succeeded()) {
            fail(
                "G1 developmental-recovery task failed to compile: " +
                developmentalStatus.task.element + ": " +
                developmentalStatus.task.message
            );
        }
        const auto developmentalBandOneOperators = std::ranges::count_if(
            compiledDevelopmental.task.randomizationOperators(),
            [](const MRTaskRandomizationOperatorGPU& operation) {
                return operation.target.w == 1u;
            }
        );
        const auto developmentalOutcomes =
            compiledDevelopmental.task.outcomes();
        const auto hasOutcome = [&developmentalOutcomes](
            const std::string_view id
        ) {
            return std::ranges::any_of(
                developmentalOutcomes,
                [id](const metalrobo::CompiledTaskOutcomeSpec& outcome) {
                    return outcome.id == id;
                }
            );
        };
        if (developmental.task.id !=
                "unitree_g1_developmental_recovery" ||
            developmentalBandOneOperators != 31u ||
            compiledDevelopmental.task.randomizationOperators().size() !=
                compiledGetUp.task.randomizationOperators().size() + 31u ||
            compiledDevelopmental.task.fingerprint() ==
                compiledGetUp.task.fingerprint() ||
            developmentalOutcomes.size() != 8u ||
            !hasOutcome("height_progress") ||
            !hasOutcome("tilt_progress") ||
            !hasOutcome("whole_body_recovery") ||
            !hasOutcome("restoration") ||
            hasOutcome("contact_reward")) {
            fail(
                "developmental recovery lost its distinct reset or typed physical outcomes"
            );
        }

        TaskWorldFixture adult =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::adultLocomotion
            );
        metalrobo::CompiledLocomotionWorld compiledAdult;
        const auto adultStatus = compileTaskWorld(
            adult, compiledAdult);
        if (!adultStatus.succeeded()) {
            fail(
                "G1 adult-locomotion task failed to compile: " +
                adultStatus.task.element + ": " +
                adultStatus.task.message
            );
        }
        const auto adultCommandSlots = std::ranges::count_if(
            adult.observations.actorFrame,
            [](const metalrobo::TaskObservationOperatorSpec& observation) {
                return observation.source ==
                    metalrobo::TaskObservationSource::command;
            }
        );
        if (adult.task.id != "unitree_g1_adult_locomotion" ||
            adultCommandSlots != 3u ||
            compiledAdult.task.layout().actorObservationSize !=
                compiledDevelopmental.task.layout().actorObservationSize ||
            compiledAdult.task.layout().actorHistoryLength != 5u ||
            compiledAdult.world.capacities().candidatePairs != 672u ||
            compiledAdult.world.capacities().manifolds != 672u ||
            compiledAdult.task.randomizationOperators().size() != 38u ||
            compiledAdult.task.fingerprint() ==
                compiledDevelopmental.task.fingerprint() ||
            compiledAdult.task.outcomes().size() != 6u) {
            fail(
                "adult locomotion lost its transferable actor ABI, standing reset, or stress curriculum"
            );
        }
        metalrobo::InteractionPack getUpInteraction = loadedInteraction;
        for (auto& track : getUpInteraction.contactTracks) {
            track.counterpart = "locomotion_ground";
        }
        metalrobo::CompiledTaskProgram getUpInteractionProgram;
        const auto getUpInteractionStatus =
            metalrobo::compileTaskProgram(
                getUp.task,
                getUp.actuators,
                getUp.observations,
                getUp.reset,
                getUpInteraction,
                "weight_shift_left_lift_right",
                compiledGetUp.world,
                getUpInteractionProgram
            );
        if (!getUpInteractionStatus.succeeded() ||
            getUpInteractionProgram.header().interactionTiming.z != 0.0f ||
            (getUpInteractionProgram.header().schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) == 0u ||
            (getUpInteractionProgram.header().schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) == 0u) {
            fail(
                "G1 get-up student did not compile in shadow mode: " +
                getUpInteractionStatus.element + ": " +
                getUpInteractionStatus.message
            );
        }
        metalrobo::TaskPack autonomousGetUp = getUp.task;
        autonomousGetUp.interactionControlReference = false;
        metalrobo::CompiledTaskProgram autonomousGetUpProgram;
        const auto autonomousGetUpStatus =
            metalrobo::compileTaskProgram(
                autonomousGetUp,
                getUp.actuators,
                getUp.observations,
                getUp.reset,
                getUpInteraction,
                "weight_shift_left_lift_right",
                compiledGetUp.world,
                autonomousGetUpProgram
            );
        if (!autonomousGetUpStatus.succeeded() ||
            (autonomousGetUpProgram.header().schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) == 0u ||
            (autonomousGetUpProgram.header().schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u) {
            fail(
                "G1 get-up autonomous reset did not separate initialization from reference control"
            );
        }
        const auto getUpRewards = compiledGetUp.task.rewardOperators();
        const auto standingReward = std::find_if(
            getUpRewards.begin(),
            getUpRewards.end(),
            [](const MRTaskRewardOperatorGPU& reward) {
                return reward.source.x ==
                    MR_TASK_REWARD_STANDING_COMPLETION;
            }
        );
        const auto restorationReward = std::find_if(
            getUpRewards.begin(),
            getUpRewards.end(),
            [](const MRTaskRewardOperatorGPU& reward) {
                return reward.source.x == MR_TASK_REWARD_RESTORATION;
            }
        );
        if (standingReward == getUpRewards.end() ||
            restorationReward == getUpRewards.end() ||
            std::abs(standingReward->parameters.x - 40.0f) > 1.0e-6f ||
            std::abs(standingReward->parameters.y - 0.65f) > 1.0e-6f ||
            std::abs(standingReward->parameters.z - 0.8f) > 1.0e-6f ||
            std::abs(restorationReward->parameters.x - 40.0f) > 1.0e-6f ||
            std::abs(restorationReward->parameters.y - 0.22f) > 1.0e-6f ||
            std::abs(restorationReward->parameters.z - 0.40f) > 1.0e-6f ||
            std::abs(restorationReward->parameters.w - 0.94f) > 1.0e-6f ||
            std::abs(restorationReward->auxiliary.x - 0.35f) > 1.0e-6f) {
            fail("get-up completion thresholds changed GPU ABI lanes");
        }
        TaskWorldFixture ballRecovery =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::ballDisturbanceRecovery
            );
        const std::array recoverySpheres{
            metalrobo::LocomotionDynamicSphere{
                .position = {-2.0f, 0.0f, 1.0f, 1.0f},
                .linearVelocity = {3.0f, 0.0f, 1.0f, 0.0f},
                .radius = 0.10f, .mass = 0.10f, .launchStep = 100u,
            },
            metalrobo::LocomotionDynamicSphere{
                .position = {2.0f, 0.0f, 1.0f, 1.0f},
                .linearVelocity = {-3.0f, 0.0f, 1.0f, 0.0f},
                .radius = 0.12f, .mass = 0.25f, .launchStep = 200u,
            },
            metalrobo::LocomotionDynamicSphere{
                .position = {0.0f, -2.0f, 1.0f, 1.0f},
                .linearVelocity = {0.0f, 3.0f, 1.0f, 0.0f},
                .radius = 0.14f, .mass = 0.50f, .launchStep = 300u,
            },
            metalrobo::LocomotionDynamicSphere{
                .position = {0.0f, 2.0f, 1.0f, 1.0f},
                .linearVelocity = {0.0f, -3.0f, 1.0f, 0.0f},
                .radius = 0.16f, .mass = 1.00f, .launchStep = 400u,
            },
            metalrobo::LocomotionDynamicSphere{
                .position = {-2.0f, 0.25f, 1.0f, 1.0f},
                .linearVelocity = {3.0f, 0.0f, 1.0f, 0.0f},
                .radius = 0.18f, .mass = 1.25f, .launchStep = 500u,
            },
            metalrobo::LocomotionDynamicSphere{
                .position = {2.0f, -0.25f, 1.0f, 1.0f},
                .linearVelocity = {-3.0f, 0.0f, 1.0f, 0.0f},
                .radius = 0.20f, .mass = 1.50f, .launchStep = 600u,
            },
        };
        metalrobo::appendLocomotionDynamicSpheres(
            ballRecovery,
            recoverySpheres
        );
        metalrobo::CompiledLocomotionWorld compiledBallRecovery;
        const auto ballRecoveryStatus =
            compileTaskWorld(ballRecovery, compiledBallRecovery);
        if (!ballRecoveryStatus.succeeded()) {
            fail(
                "G1 ball-recovery task failed to compile: " +
                ballRecoveryStatus.world.message + " " +
                ballRecoveryStatus.task.element + ": " +
                ballRecoveryStatus.task.message
            );
        }
        if (compiledBallRecovery.task.header().counts2.y != 67u ||
            compiledBallRecovery.task.layout().actorObservationSize != 140u ||
            compiledBallRecovery.task.layout().criticObservationSize != 148u ||
            compiledBallRecovery.task.layout().contactMetricCount != 51u ||
            compiledBallRecovery.task.header().counts0.w != 5u ||
            compiledBallRecovery.task.header().counts1.w != 20u ||
            std::abs(
                compiledBallRecovery.task.header().locomotion.w - 0.70f
            ) > 1.0e-6f ||
            compiledBallRecovery.task.header().dynamics.x != 0.0f ||
            compiledBallRecovery.world.sceneBodyCount() != 7u ||
            compiledBallRecovery.world.capacities().candidatePairs != 192u ||
            compiledBallRecovery.world.capacities().constraintRows != 192u) {
            fail("compiled G1 physical-ball task is incomplete");
        }
        std::uint32_t objectTrackOperators = 0u;
        for (const MRTaskObservationOperatorGPU& operation :
             compiledBallRecovery.task.actorOperators()) {
            if (operation.source.x == MR_TASK_OBSERVE_OBJECT_TRACK) {
                ++objectTrackOperators;
            }
        }
        if (objectTrackOperators != 42u) {
            fail("G1 physical-ball perception contract is incomplete");
        }
        std::uint32_t stagedImpactVelocities = 0u;
        std::uint32_t baseLaunchSchedules = 0u;
        for (const MRTaskRandomizationOperatorGPU& operation :
             compiledBallRecovery.task.randomizationOperators()) {
            if (operation.target.x ==
                    MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY &&
                operation.target.w >= 1u &&
                operation.target.w <= 3u) {
                ++stagedImpactVelocities;
            }
            if (operation.target.x ==
                    MR_TASK_RANDOMIZE_SCENE_BODY_LAUNCH_STEP) {
                if (operation.target.w == 0u) {
                    ++baseLaunchSchedules;
                }
            }
        }
        if (stagedImpactVelocities != 18u ||
            baseLaunchSchedules != 6u) {
            fail("G1 physical-ball difficulty distribution is incomplete");
        }
        const auto impactEvents =
            compiledBallRecovery.task.impactEvents();
        if (impactEvents.size() != 6u ||
            compiledBallRecovery.task.header().counts3.x != 6u) {
            fail("G1 event-driven impact sequence is incomplete");
        }
        for (std::uint32_t impact = 0u;
             impact < impactEvents.size();
             ++impact) {
            const MRTaskImpactEventGPU& event = impactEvents[impact];
            if (event.binding.y != impact ||
                event.binding.z != 1u ||
                event.gate.x != 0.05f ||
                event.gate.y != 0.50f ||
                event.gate.z != 2.0f ||
                event.gate.w != 0.70f) {
                fail("G1 event-driven impact gates changed");
            }
        }
        TaskWorldFixture dodge =
            makeG1TaskWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::ballDodge
            );
        metalrobo::appendLocomotionDynamicSpheres(
            dodge,
            recoverySpheres
        );
        metalrobo::CompiledLocomotionWorld compiledDodge;
        const auto dodgeStatus = compileTaskWorld(dodge, compiledDodge);
        if (!dodgeStatus.succeeded() ||
            compiledDodge.task.header().counts1.w != 34u ||
            compiledDodge.task.header().counts2.x != 3u ||
            compiledDodge.task.layout().actorFrameSize != 122u ||
            compiledDodge.task.layout().actorHistoryLength != 5u ||
            compiledDodge.task.layout().actorObservationSize != 1210u ||
            compiledDodge.task.layout().criticObservationSize != 174u ||
            compiledDodge.task.layout().contactMetricCount != 51u ||
            compiledDodge.task.header().dynamics.z != 0.20f ||
            compiledDodge.task.header().visualLayout.x != 16u ||
            compiledDodge.task.header().visualLayout.y != 9u ||
            compiledDodge.task.header().visualLayout.z != 4u ||
            compiledDodge.task.header().visualLayout.w != 18u ||
            compiledDodge.task.header().visualHistory.x != 0u ||
            compiledDodge.task.header().visualHistory.y != 3u ||
            compiledDodge.task.header().visualHistory.z != 8u ||
            compiledDodge.task.header().visualHistory.w != 18u ||
            compiledDodge.task.header().visualRange.x != 0.1f ||
            compiledDodge.task.header().visualRange.y != 5.0f ||
            compiledDodge.task.header().visualRange.z != 1.0f ||
            compiledDodge.task.header().visualRange.w != 1.0f ||
            compiledDodge.task.header().visualCorruption.x != 0.005f ||
            compiledDodge.task.header().visualCorruption.y != 0.03f ||
            compiledDodge.task.header().visualCorruption.z != 0.05f ||
            compiledDodge.task.header().visualCorruption.w != 0.01f ||
            compiledDodge.task.header().schedule.z != 4u ||
            std::abs(
                compiledDodge.task.header().locomotion.w - 0.70f
            ) > 1.0e-6f ||
            std::abs(
                compiledDodge.task.header().commandUpper.w - 1.5f
            ) > 1.0e-6f ||
            (compiledDodge.task.header().schedule.w &
             MR_TASK_PROGRAM_MASKED_DEPTH_FEATURES) == 0u ||
            (compiledDodge.task.header().schedule.w &
             MR_TASK_PROGRAM_THREAT_TEACHER) == 0u ||
            compiledDodge.task.header().threat.x == MR_INVALID_INDEX ||
            compiledDodge.task.header().threat.y != 1u ||
            compiledDodge.task.header().threatTiming.x != 0.5f ||
            compiledDodge.task.header().threatTiming.y != 2.0f ||
            compiledDodge.task.header().threatTiming.z != 0.05f ||
            compiledDodge.task.header().threatTiming.w != 2.0f ||
            compiledDodge.task.header().motion.y != 13u ||
            compiledDodge.task.header().motion.z != 117u ||
            compiledDodge.task.layout().motionFeatureCount != 117u ||
            compiledDodge.task.motionBodies().size() != 13u ||
            compiledDodge.task.header().projectile.x != 1.0f ||
            compiledDodge.task.header().projectile.y != 6.0f ||
            compiledDodge.task.header().projectile.z != 0.45f ||
            compiledDodge.task.header().projectile.w != 1.35f ||
            compiledDodge.task.header().projectileGravity.w != 0.40f ||
            compiledDodge.task.header().projectileGravity.z != -9.81f ||
            compiledDodge.task.header().counts3.y !=
                compiledDodge.task.contactMembers().size() ||
            compiledDodge.task.contactMemberRadii().size() !=
                compiledDodge.task.contactMembers().size()) {
            fail("G1 native projectile-dodge task is incomplete");
        }
        std::uint32_t collidableMembers = 0u;
        for (const float radius :
             compiledDodge.task.contactMemberRadii()) {
            if (radius < 0.0f || !std::isfinite(radius)) {
                fail("compiled contact-member envelope is invalid");
            }
            collidableMembers += radius > 0.0f ? 1u : 0u;
        }
        if (collidableMembers == 0u) {
            fail("G1 dodge group has no collidable members");
        }
        TaskWorldFixture manipulation = dodge;
        manipulation.task.id = "generic_rigid_object_manipulation_probe";
        manipulation.task.outcomes = {
            {"grasp", "reward",
                metalrobo::TaskOutcomeSource::rewardContribution,
                metalrobo::TaskOutcomeDirection::higherIsBetter,
                metalrobo::TaskRewardOperator::objectGrasp},
            {"lift", "reward",
                metalrobo::TaskOutcomeSource::rewardContribution,
                metalrobo::TaskOutcomeDirection::higherIsBetter,
                metalrobo::TaskRewardOperator::objectLift},
            {"position", "reward",
                metalrobo::TaskOutcomeSource::rewardContribution,
                metalrobo::TaskOutcomeDirection::higherIsBetter,
                metalrobo::TaskRewardOperator::objectPosition},
            {"placement", "reward",
                metalrobo::TaskOutcomeSource::rewardContribution,
                metalrobo::TaskOutcomeDirection::higherIsBetter,
                metalrobo::TaskRewardOperator::objectPlacement},
        };
        manipulation.task.rewards = {
            {
                .operation = metalrobo::TaskRewardOperator::objectGrasp,
                .sourceGroup = "impact_contact",
                .target = "locomotion_dynamic_sphere_0",
                .weight = 1.0f,
                .parameters = {2.0f, 20.0f, 0.0f, 0.0f},
            },
            {
                .operation = metalrobo::TaskRewardOperator::objectLift,
                .target = "locomotion_dynamic_sphere_0",
                .weight = 1.0f,
                .parameters = {0.1f, 0.5f, 0.0f, 0.0f},
            },
            {
                .operation = metalrobo::TaskRewardOperator::objectPosition,
                .target = "locomotion_dynamic_sphere_0",
                .weight = 1.0f,
                .parameters = {0.5f, 0.0f, 0.8f, 0.01f},
            },
            {
                .operation = metalrobo::TaskRewardOperator::objectPlacement,
                .sourceGroup = "locomotion_ground",
                .target = "locomotion_dynamic_sphere_0",
                .weight = 1.0f,
                .parameters = {0.01f, 0.01f, 0.04f, 0.0f},
            },
        };
        metalrobo::CompiledLocomotionWorld compiledManipulation;
        const auto manipulationStatus =
            compileTaskWorld(manipulation, compiledManipulation);
        if (!manipulationStatus.succeeded() ||
            compiledManipulation.task.rewardOperators().size() != 4u ||
            compiledManipulation.task.outcomes().size() != 4u ||
            compiledManipulation.task.header().interactionOffsets1.w != 4u) {
            fail("generic rigid-object manipulation rewards did not compile");
        }
        for (const MRTaskRewardOperatorGPU& operation :
             compiledManipulation.task.rewardOperators()) {
            if (operation.source.z != 1u) {
                fail("manipulation reward did not bind the selected scene object");
            }
            if (operation.source.x == MR_TASK_REWARD_OBJECT_GRASP &&
                operation.source.y == MR_INVALID_INDEX) {
                fail("object grasp did not bind its semantic contact group");
            }
        }
        std::uint32_t stagedDodgeVelocities = 0u;
        std::uint32_t acquisitionDodgeVelocities = 0u;
        std::uint32_t forwardDodgePositions = 0u;
        for (const MRTaskRandomizationOperatorGPU& operation :
             compiledDodge.task.randomizationOperators()) {
            if (operation.target.x ==
                    MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY &&
                operation.target.w == 0u) {
                const float lower = std::min(
                    std::abs(operation.parameters.x),
                    std::abs(operation.parameters.y)
                );
                const float upper = std::max(
                    std::abs(operation.parameters.x),
                    std::abs(operation.parameters.y)
                );
                acquisitionDodgeVelocities +=
                    lower == 1.0f && upper == 2.0f ? 1u : 0u;
            }
            if (operation.target.x ==
                    MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY &&
                operation.target.w >= 1u &&
                operation.target.w <= 3u) {
                constexpr std::array<std::array<float, 2>, 4>
                    expectedSpeedBands{{
                        {{1.0f, 2.0f}},
                        {{2.0f, 3.0f}},
                        {{3.5f, 4.5f}},
                        {{5.0f, 6.0f}},
                    }};
                const auto& expected =
                    expectedSpeedBands[operation.target.w];
                const float lower = std::min(
                    std::abs(operation.parameters.x),
                    std::abs(operation.parameters.y)
                );
                const float upper = std::max(
                    std::abs(operation.parameters.x),
                    std::abs(operation.parameters.y)
                );
                if (lower != expected[0] || upper != expected[1]) {
                    fail("G1 dodge projectile-speed band changed");
                }
                ++stagedDodgeVelocities;
            }
            if (operation.target.x !=
                    MR_TASK_RANDOMIZE_SCENE_BODY_POSITION) {
                continue;
            }
            ++forwardDodgePositions;
            const bool validRange = operation.target.z == 0u
                ? operation.parameters.x == 1.30f &&
                    operation.parameters.y == 1.70f
                : operation.target.z == 1u
                    ? operation.parameters.x >= -0.65f &&
                        operation.parameters.y <= 0.65f &&
                        operation.parameters.x < operation.parameters.y
                    : operation.target.z == 2u &&
                        operation.parameters.x == 1.00f &&
                        operation.parameters.y == 1.35f;
            if (!validRange) {
                fail("G1 dodge launch origin left the camera frustum");
            }
        }
        if (acquisitionDodgeVelocities != 6u ||
            stagedDodgeVelocities != 18u ||
            forwardDodgePositions != 18u) {
            fail("G1 dodge projectile-speed ladder is incomplete");
        }
        std::uint32_t barriers = 0u;
        std::uint32_t evasions = 0u;
        std::uint32_t misses = 0u;
        std::uint32_t safeStillness = 0u;
        std::uint32_t safeActionRate = 0u;
        std::uint32_t jointCbfCorrection = 0u;
        std::uint32_t jointCbfBuffer = 0u;
        std::uint32_t predictedClearance = 0u;
        std::uint32_t ungatedVelocityTracking = 0u;
        std::uint32_t maskedDepth = 0u;
        std::uint32_t scaledAngularVelocity = 0u;
        std::uint32_t scaledJointVelocity = 0u;
        std::uint32_t normalizedGravity = 0u;
        std::uint32_t supportSense = 0u;
        std::uint32_t supportPatch = 0u;
        std::uint32_t commandObservations = 0u;
        for (const MRTaskObservationOperatorGPU& operation :
             compiledDodge.task.actorOperators()) {
            maskedDepth += operation.source.x ==
                MR_TASK_OBSERVE_MASKED_DEPTH ? 1u : 0u;
            scaledAngularVelocity +=
                operation.source.x ==
                        MR_TASK_OBSERVE_ROOT_ANGULAR_VELOCITY_LOCAL &&
                    operation.transform.x == 0.2f
                ? 1u
                : 0u;
            scaledJointVelocity +=
                operation.source.x == MR_TASK_OBSERVE_JOINT_VELOCITY &&
                    operation.transform.x == 0.05f
                ? 1u
                : 0u;
            normalizedGravity +=
                operation.source.x == MR_TASK_OBSERVE_PROJECTED_GRAVITY &&
                    (operation.source.w &
                     MR_TASK_OBSERVATION_NORMALIZE_VECTOR3) != 0u
                ? 1u
                : 0u;
            supportSense += operation.source.x ==
                MR_TASK_OBSERVE_SUPPORT_SENSE ? 1u : 0u;
            supportPatch += operation.source.x ==
                MR_TASK_OBSERVE_SUPPORT_PATCH ? 1u : 0u;
            commandObservations += operation.source.x ==
                MR_TASK_OBSERVE_COMMAND ? 1u : 0u;
            if (operation.source.x == MR_TASK_OBSERVE_OBJECT_TRACK) {
                fail("G1 dodge actor contains privileged object tracks");
            }
        }
        if (maskedDepth !=
                4u * 16u * 9u +
                    MR_TASK_MASKED_DEPTH_FEATURE_COUNT ||
            scaledAngularVelocity != 3u ||
            scaledJointVelocity != 29u ||
            normalizedGravity != 3u ||
            supportSense != 3u ||
            supportPatch != 26u ||
            commandObservations != 0u) {
            fail("G1 dodge deployment observation contract changed");
        }
        for (const MRTaskRewardOperatorGPU& operation :
             compiledDodge.task.rewardOperators()) {
            if (operation.source.x ==
                    MR_TASK_REWARD_LINK_CLEARANCE_BARRIER) {
                const float expectedEnvelope =
                    recoverySpheres[barriers].radius + 0.05f;
                if (operation.source.y == MR_INVALID_INDEX ||
                    operation.source.z == MR_INVALID_INDEX ||
                    operation.parameters.y != 1.0f ||
                    std::abs(
                        operation.parameters.z - expectedEnvelope
                    ) >
                        1.0e-6f ||
                    operation.parameters.w != 2.0f) {
                    fail("compiled projectile link barrier changed");
                }
                ++barriers;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_PROJECTILE_EVASION) {
                if (operation.source.z == MR_INVALID_INDEX ||
                    operation.parameters.x != 1.0f ||
                    operation.parameters.y != 0.3f ||
                    operation.parameters.z != 1.0f ||
                    operation.parameters.w != 0.9f) {
                    fail("compiled projectile evasion changed");
                }
                ++evasions;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS) {
                if (operation.parameters.x != 0.5f ||
                    operation.parameters.y != 2.0f) {
                    fail("compiled safe stillness changed");
                }
                ++safeStillness;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE) {
                if (operation.parameters.x != -0.05f) {
                    fail("compiled safe action-rate shaping changed");
                }
                ++safeActionRate;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_JOINT_CBF_CORRECTION) {
                if (operation.parameters.x != -0.08f) {
                    fail("compiled Joint-CBF correction reward changed");
                }
                ++jointCbfCorrection;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_JOINT_CBF_BUFFER) {
                if (operation.parameters.x != -0.20f) {
                    fail("compiled Joint-CBF buffer reward changed");
                }
                ++jointCbfBuffer;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE) {
                if (operation.parameters.x != 2.0f ||
                    operation.parameters.y != 0.25f) {
                    fail("compiled predicted-clearance reward changed");
                }
                ++predictedClearance;
            } else if (operation.source.x ==
                           MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING ||
                       operation.source.x ==
                           MR_TASK_REWARD_YAW_VELOCITY_TRACKING) {
                ++ungatedVelocityTracking;
            } else if (operation.source.x ==
                       MR_TASK_REWARD_PROJECTILE_MISS) {
                ++misses;
            }
        }
        if (barriers != 6u || evasions != 6u || misses != 1u ||
            safeStillness != 1u || safeActionRate != 1u ||
            jointCbfCorrection != 1u || jointCbfBuffer != 1u ||
            predictedClearance != 1u ||
            ungatedVelocityTracking != 0u) {
            fail("G1 dodge shaping operators are incomplete");
        }
        const auto dodgeTerminations =
            compiledDodge.task.terminationOperators();
        if (dodgeTerminations.back().source.x !=
                MR_TASK_TERMINATE_PROJECTILE_CONTACT ||
            dodgeTerminations.back().source.z !=
                MR_TASK_TERMINATION_PROJECTILE_CONTACT ||
            dodgeTerminations.back().parameters.x != 5.0f) {
            fail("G1 projectile-contact termination changed");
        }
        for (const MRTaskImpactEventGPU& event :
             compiledDodge.task.impactEvents()) {
            if (event.binding.w == MR_INVALID_INDEX ||
                event.binding.z != 0u ||
                event.gate.x != 3.14159265f ||
                event.gate.y != 0.02f ||
                event.gate.z != 2.0f ||
                event.gate.w != 0.10f ||
                !(event.projectile.x > 0.0f)) {
                fail("G1 dodge event schedule is recovery-gated");
            }
        }
        TaskWorldFixture disturbed = authored;
        const std::array disturbanceSpheres{
            metalrobo::LocomotionDynamicSphere{
                .position = {-1.0f, 0.0f, 1.0f, 1.0f},
                .linearVelocity = {2.0f, 0.0f, 0.5f, 0.0f},
                .radius = 0.1f,
                .mass = 0.08f,
                .launchStep = 17u,
            },
        };
        metalrobo::appendLocomotionDynamicSpheres(
            disturbed,
            disturbanceSpheres
        );
        metalrobo::CompiledLocomotionWorld disturbedCompiled;
        const auto disturbedStatus =
            compileTaskWorld(disturbed, disturbedCompiled);
        const MRBodyStateGPU& disturbanceState =
            disturbed.sceneBodies.back();
        if (!disturbedStatus.succeeded() ||
            disturbed.model.bodies.size() !=
                authored.model.bodies.size() + 1u ||
            disturbed.model.shapes.size() !=
                authored.model.shapes.size() + 1u ||
            disturbed.sceneBodies.size() !=
                authored.sceneBodies.size() + 1u ||
            disturbanceState.linearVelocityAndInverseMass.x != 2.0f ||
            disturbanceState.linearVelocityAndInverseMass.w != 12.5f ||
            (disturbanceState.flagsAndIndices[3] &
             MR_BODY_STATE_PRESERVE_RESET_VELOCITY) == 0u ||
            ((disturbanceState.flagsAndIndices[3] &
              MR_BODY_STATE_LAUNCH_STEP_MASK) >>
             MR_BODY_STATE_LAUNCH_STEP_SHIFT) != 17u ||
            disturbedCompiled.world.fingerprint() ==
                world.fingerprint()) {
            fail(
                "generic dynamic locomotion sphere did not enter the compiled world"
            );
        }
        TaskWorldFixture invalidEndpointCapacity =
            authored;
        invalidEndpointCapacity.task.capacities
            .endpointRuntimeRecords -= 1u;
        metalrobo::CompiledLocomotionWorld invalidCompiledWorld;
        const auto invalidEndpointStatus =
            compileTaskWorld(invalidEndpointCapacity, invalidCompiledWorld);
        if (invalidEndpointStatus.world.status !=
            metalrobo::MetalWorldHostStatus::capacityOverflow) {
            fail(
                "noncanonical endpoint-runtime capacity was not rejected"
            );
        }
        TaskWorldFixture invalidQueryCapacity =
            authored;
        invalidQueryCapacity.task.capacities
            .articulationPointQueries += 1u;
        const auto invalidQueryStatus =
            compileTaskWorld(invalidQueryCapacity, invalidCompiledWorld);
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
                binding.indices.w != 6u + action ||
                binding.drive.x !=
                    authored.model.dofs[binding.indices.w].drive.x ||
                binding.drive.y !=
                    authored.model.dofs[binding.indices.w].drive.y) {
                fail("semantic action binding did not resolve");
            }
        }

        metalrobo::CompiledTaskProgram repeated;
        const metalrobo::TaskCompileDiagnostics repeatedStatus =
            metalrobo::compileTaskProgram(
                authored.task,
                authored.actuators,
                authored.observations,
                authored.reset,
                world,
                repeated
            );
        if (!repeatedStatus.succeeded() ||
            repeated.fingerprint() != program.fingerprint()) {
            fail("task compilation is not content deterministic");
        }

        metalrobo::TaskPack broken = authored.task;
        broken.actions.front().actuator =
            "missing_joint_from_import";
        const std::uint64_t preserved =
            repeated.fingerprint();
        const metalrobo::TaskCompileDiagnostics rejected =
            metalrobo::compileTaskProgram(
                broken,
                authored.actuators,
                authored.observations,
                authored.reset,
                world,
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
        std::vector<metalrobo::RobotActuatorSpec> velocityActuators =
            authored.actuators;
        velocityActuators.front().kind =
            metalrobo::RobotActuatorKind::jointVelocity;
        metalrobo::CompiledTaskProgram velocityProgram;
        const metalrobo::TaskCompileDiagnostics velocityActuator =
            metalrobo::compileTaskProgram(
                authored.task,
                velocityActuators,
                authored.observations,
                authored.reset,
                world,
                velocityProgram
            );
        if (!velocityActuator.succeeded() ||
            velocityProgram.actionBindings().front().actuator.x !=
                MR_TASK_ACTUATOR_JOINT_VELOCITY ||
            velocityProgram.fingerprint() == preserved) {
            fail(
                "typed joint-velocity actuator did not compile into its native program"
            );
        }
        metalrobo::TaskPack mismatched = authored.task;
        ++mismatched.capacities.candidatePairs;
        const metalrobo::TaskCompileDiagnostics
            capacityRejected = metalrobo::compileTaskProgram(
                mismatched,
                authored.actuators,
                authored.observations,
                authored.reset,
                world,
                repeated
            );
        if (capacityRejected.status !=
                metalrobo::TaskCompileStatus::invalidWorld ||
            repeated.fingerprint() != preserved) {
            fail(
                "mismatched task capacity contract was not transactionally rejected"
            );
        }
        metalrobo::TaskPack invalidPatch = authored.task;
        invalidPatch.contactGroups.front().supportPatchWidth = 0u;
        const metalrobo::TaskCompileDiagnostics patchRejected =
            metalrobo::compileTaskProgram(
                invalidPatch,
                authored.actuators,
                authored.observations,
                authored.reset,
                world,
                repeated
            );
        if (patchRejected.status !=
                metalrobo::TaskCompileStatus::invalidPack ||
            repeated.fingerprint() != preserved) {
            fail(
                "invalid support patch was not transactionally rejected"
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
        metalrobo::CompiledPolicyProgram unboundPolicyOutput;
        const auto unboundPolicyStatus =
            metalrobo::compilePolicyProgram(
                policy,
                program,
                unboundPolicyOutput
            );
        if (unboundPolicyStatus.status !=
                metalrobo::PolicyCompileStatus::invalidPack ||
            unboundPolicyOutput.valid()) {
            fail("unbound PolicyPack was accepted");
        }
        metalrobo::bindPolicyPack(policy, program);
        metalrobo::CompiledPolicyProgram compiledPolicy;
        const metalrobo::PolicyCompileDiagnostics
            policyStatus = metalrobo::compilePolicyProgram(
                policy,
                program,
                compiledPolicy
            );
        if (!policyStatus.succeeded() ||
            !compiledPolicy.valid() ||
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
        metalrobo::PolicyPack wrongContract = policy;
        wrongContract.contract.actionFingerprint ^= 1u;
        const auto contractRejected =
            metalrobo::compilePolicyProgram(
                wrongContract,
                program,
                compiledPolicy
            );
        if (contractRejected.status !=
                metalrobo::PolicyCompileStatus::incompatibleContract ||
            compiledPolicy.fingerprint() == 0u) {
            fail("semantic PolicyPack mismatch was not rejected");
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
                    authored.actuators,
                    authored.observations,
                    authored.reset,
                    world,
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
        rollout.motionFeatureCount = layout.motionFeatureCount;
        rollout.motionFeatures.assign(
            sampleCount * layout.motionFeatureCount,
            0.375f
        );
        rollout.teacherActions.assign(
            sampleCount * layout.actionCount,
            -0.25f
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
        rollout.outcomes = {
            {"tracking", "ratio", 1u},
            {"projectile_miss_reward", "reward", 1u},
        };
        rollout.outcomeValues.resize(sampleCount * 2u);
        rollout.transitions.resize(sampleCount);
        for (std::size_t index = 0u;
             index < sampleCount;
             ++index) {
            rollout.transitions[index].rewardAndBootstrap.x =
                static_cast<float>(index) * 0.1f;
            rollout.transitions[index].termination.x =
                index + 1u == sampleCount ? 1u : 0u;
            rollout.outcomeValues[index * 2u] = 0.03f;
            rollout.outcomeValues[index * 2u + 1u] = 0.05f;
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
                    .motionFeatureCount =
                        rollout.motionFeatureCount,
                    .actorObservations =
                        rollout.actorObservations,
                    .criticObservations =
                        rollout.criticObservations,
                    .motionFeatures =
                        rollout.motionFeatures,
                    .teacherActions =
                        rollout.teacherActions,
                    .latents = rollout.latents,
                    .logProbabilities =
                        rollout.logProbabilities,
                    .values = rollout.values,
                    .bootstrapValues =
                        rollout.bootstrapValues,
                    .outcomes = rollout.outcomes,
                    .outcomeValues = rollout.outcomeValues,
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
            loadedRollout.teacherActions !=
                rollout.teacherActions ||
            loadedRollout.transitions.size() !=
                rollout.transitions.size() ||
            loadedRollout.transitions.back()
                    .termination.x != 1u ||
            loadedRollout.outcomes.size() != 2u ||
            loadedRollout.outcomes.back().id !=
                "projectile_miss_reward" ||
            loadedRollout.outcomeValues.back() != 0.05f) {
            fail(
                "PolicyRolloutPack round trip changed its learning records: " +
                rolloutWrite.message + " / " + borrowedWrite.message +
                " / " + rolloutRead.message +
                " hashes=" + std::to_string(rolloutWrite.contentHash) +
                "," + std::to_string(borrowedWrite.contentHash) +
                " outcomes=" + std::to_string(loadedRollout.outcomes.size()) +
                " values=" + std::to_string(loadedRollout.outcomeValues.size()) +
                " last=" + (loadedRollout.outcomeValues.empty()
                    ? std::string{"empty"}
                    : std::to_string(loadedRollout.outcomeValues.back()))
            );
        }
        metalrobo::MotionPack motion{
            .id = "pacman_dodge_motion",
            .sourceRepository =
                "https://github.com/lzyang2000/perceptive_cbf_rl",
            .sourceRevision = "2d426697",
            .license = "MIT",
            .anchorBody = "base",
            .trackedBodies = {"base", "foot"},
            .featureCount = 18u,
            .clips = {{
                .id = "fixture",
                .framesPerSecond = 50.0f,
                .features = std::vector<float>(
                    36u,
                    0.125f
                ),
            }},
        };
        const auto motionWrite = metalrobo::writeMotionPack(
            motion,
            packFiles.motion
        );
        metalrobo::MotionPack loadedMotion;
        const auto motionRead = metalrobo::readMotionPack(
            packFiles.motion,
            loadedMotion
        );
        if (!motionWrite.succeeded() ||
            !motionRead.succeeded() ||
            loadedMotion.featureCount != motion.featureCount ||
            loadedMotion.clips.front().features !=
                motion.clips.front().features) {
            fail("MotionPack round trip changed expert features");
        }
        const std::uint64_t importedFingerprint =
            compileImportedRobotFixture();
        const std::uint64_t worldPackFingerprint =
            checkWorldPackLocomotionImport();

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
            << " interaction_artifact="
            << interactionWrite.contentHash
            << " interaction_task="
            << interactionProgram.fingerprint()
            << " external_interaction_artifact="
            << externalInteractionArtifact
            << " external_interaction_task="
            << externalInteractionTask
            << " external_native="
            << (externalNativeExecuted ? 1u : 0u)
            << " external_native_device=\""
            << externalRuntime.device << '"'
            << " external_native_gpu_ms="
            << externalRuntime.gpuMilliseconds
            << " external_native_reward="
            << externalRuntime.firstReward
            << " fixture_native_reward="
            << fixtureRuntime.firstReward
            << " imported=" << importedFingerprint
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
