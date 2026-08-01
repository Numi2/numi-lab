#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/TaskProgram.hpp"
#include "metalrobo/WorldPack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
    metalrobo::LocomotionWorld authored;
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

    metalrobo::CompiledLocomotionWorld compiled;
    const metalrobo::LocomotionWorldCompileDiagnostics status =
        metalrobo::compileLocomotionWorld(
            authored,
            0u,
            compiled
        );
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

std::uint64_t checkWorldPackLocomotionImport() {
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
        fail("WorldPack locomotion fixture did not compile");
    }
    metalrobo::TaskPack task;
    task.id = "world_pack_import_fixture";
    const metalrobo::LocomotionWorld imported =
        metalrobo::makeWorldPackLocomotionWorld(
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
        metalrobo::LocomotionWorld authored =
            metalrobo::makeUnitreeG1LocomotionWorld(
                metalrobo::LocomotionSurface::terrain
            );
        metalrobo::CompiledLocomotionWorld compiledWorld;
        const metalrobo::LocomotionWorldCompileDiagnostics
            compileStatus = metalrobo::compileLocomotionWorld(
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
        const metalrobo::TaskProgramLayout& layout =
            program.layout();
        if (layout.actionCount != 29u ||
            layout.actorFrameSize != 98u ||
            layout.actorHistoryLength != 1u ||
            layout.actorObservationSize != 98u ||
            layout.criticFrameSize != 98u ||
            layout.criticHistoryLength != 1u ||
            layout.criticObservationSize != 98u ||
            layout.contactMetricCount != 37u ||
            layout.delayStateCount != 3u) {
            fail("compiled G1 task layout changed");
        }
        if (program.worldFingerprint() != world.fingerprint() ||
            world.capacities().candidatePairs != 128u ||
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
            program.header().commandLower.x != 0.0f ||
            program.header().commandLower.y != 0.0f ||
            program.header().commandLower.z != 0.0f ||
            program.header().commandUpper.x != 0.0f ||
            program.header().commandUpper.y != 0.0f ||
            program.header().commandUpper.z != 0.0f ||
            std::abs(
                program.header().commandUpper.w - 0.8f
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
        metalrobo::LocomotionWorld recovery =
            metalrobo::makeUnitreeG1LocomotionWorld(
                metalrobo::LocomotionSurface::terrain,
                metalrobo::UnitreeG1Task::disturbanceRecovery
            );
        metalrobo::CompiledLocomotionWorld compiledRecovery;
        const auto recoveryStatus =
            metalrobo::compileLocomotionWorld(
                recovery,
                0u,
                compiledRecovery
            );
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
        metalrobo::LocomotionWorld getUp =
            metalrobo::makeUnitreeG1LocomotionWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::supineGetUpDiscovery
            );
        metalrobo::CompiledLocomotionWorld compiledGetUp;
        const auto getUpStatus = metalrobo::compileLocomotionWorld(
            getUp,
            0u,
            compiledGetUp
        );
        if (!getUpStatus.succeeded()) {
            fail(
                "G1 supine get-up task failed to compile: " +
                getUpStatus.task.element + ": " +
                getUpStatus.task.message
            );
        }
        if (compiledGetUp.task.layout().actorFrameSize != 92u ||
            compiledGetUp.task.layout().actorObservationSize != 920u ||
            compiledGetUp.task.layout().criticFrameSize != 98u ||
            compiledGetUp.task.layout().criticObservationSize != 980u ||
            compiledGetUp.task.layout().contactMetricCount != 43u ||
            compiledGetUp.task.header().counts1.w != 12u ||
            compiledGetUp.task.header().counts2.x != 0u ||
            compiledGetUp.task.header().counts2.y != 31u) {
            fail("compiled G1 supine get-up task is incomplete");
        }
        metalrobo::LocomotionWorld ballRecovery =
            metalrobo::makeUnitreeG1LocomotionWorld(
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
            metalrobo::compileLocomotionWorld(
                ballRecovery,
                0u,
                compiledBallRecovery
            );
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
            compiledBallRecovery.task.layout().contactMetricCount != 43u ||
            compiledBallRecovery.task.header().counts0.w != 5u ||
            compiledBallRecovery.task.header().counts1.w != 20u ||
            (compiledBallRecovery.task.header().schedule.w &
             MR_TASK_PROGRAM_RECOVERY_CURRICULUM) != 0u ||
            std::abs(
                compiledBallRecovery.task.header().locomotion.w - 0.70f
            ) > 1.0e-6f ||
            compiledBallRecovery.task.header().dynamics.x != 0.0f ||
            compiledBallRecovery.world.sceneBodyCount() != 7u ||
            compiledBallRecovery.world.capacities().candidatePairs != 256u ||
            compiledBallRecovery.world.capacities().constraintRows != 384u) {
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
            fail("G1 physical-ball curriculum is incomplete");
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
        metalrobo::LocomotionWorld dodge =
            metalrobo::makeUnitreeG1LocomotionWorld(
                metalrobo::LocomotionSurface::ground,
                metalrobo::UnitreeG1Task::ballDodge
            );
        metalrobo::appendLocomotionDynamicSpheres(
            dodge,
            recoverySpheres
        );
        metalrobo::CompiledLocomotionWorld compiledDodge;
        const auto dodgeStatus = metalrobo::compileLocomotionWorld(
            dodge,
            0u,
            compiledDodge
        );
        if (!dodgeStatus.succeeded() ||
            compiledDodge.task.header().counts1.w != 31u ||
            compiledDodge.task.header().counts2.x != 3u ||
            compiledDodge.task.layout().actorFrameSize != 96u ||
            compiledDodge.task.layout().actorHistoryLength != 5u ||
            compiledDodge.task.layout().actorObservationSize != 1056u ||
            compiledDodge.task.layout().criticObservationSize != 148u ||
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
            compiledDodge.task.header().visualRange.z != 0.15f ||
            compiledDodge.task.header().visualCorruption.x != 0.02f ||
            compiledDodge.task.header().visualCorruption.y != 0.10f ||
            compiledDodge.task.header().visualCorruption.z != 0.15f ||
            compiledDodge.task.header().visualCorruption.w != 0.03f ||
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
        std::uint32_t barriers = 0u;
        std::uint32_t evasions = 0u;
        std::uint32_t misses = 0u;
        std::uint32_t safeStillness = 0u;
        std::uint32_t safeActionRate = 0u;
        std::uint32_t ungatedVelocityTracking = 0u;
        std::uint32_t maskedDepth = 0u;
        std::uint32_t scaledAngularVelocity = 0u;
        std::uint32_t scaledJointVelocity = 0u;
        std::uint32_t normalizedGravity = 0u;
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
            if (operation.source.x == MR_TASK_OBSERVE_OBJECT_TRACK) {
                fail("G1 dodge actor contains privileged object tracks");
            }
        }
        if (maskedDepth != 4u * 16u * 9u ||
            scaledAngularVelocity != 3u ||
            scaledJointVelocity != 29u ||
            normalizedGravity != 3u) {
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
                event.gate.w != 0.10f) {
                fail("G1 dodge event schedule is recovery-gated");
            }
        }
        metalrobo::LocomotionWorld disturbed = authored;
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
            metalrobo::compileLocomotionWorld(
                disturbed,
                0u,
                disturbedCompiled
            );
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
        metalrobo::LocomotionWorld invalidEndpointCapacity =
            authored;
        invalidEndpointCapacity.task.capacities
            .endpointRuntimeRecords -= 1u;
        metalrobo::CompiledLocomotionWorld invalidCompiledWorld;
        const auto invalidEndpointStatus =
            metalrobo::compileLocomotionWorld(
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
        metalrobo::LocomotionWorld invalidQueryCapacity =
            authored;
        invalidQueryCapacity.task.capacities
            .articulationPointQueries += 1u;
        const auto invalidQueryStatus =
            metalrobo::compileLocomotionWorld(
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
