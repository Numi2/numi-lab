#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/TaskProgram.hpp"

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
    }

    ~TemporaryPackFiles() {
        std::error_code ignored;
        std::filesystem::remove(task, ignored);
        std::filesystem::remove(policy, ignored);
    }

    std::filesystem::path task;
    std::filesystem::path policy;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::uint64_t compileImportedRobotFixture() {
    constexpr std::string_view urdf = R"(
<robot name="generic_locomotion_fixture">
  <link name="base">
    <inertial>
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
        layout.actorFrameSize != 2u ||
        layout.actorObservationSize != 4u ||
        layout.criticObservationSize != 5u ||
        layout.contactMetricCount != 6u ||
        compiled.task.actionBindings().front().indices.x != 0u ||
        compiled.task.terrainResetTranslations().size() != 1u) {
        fail(
            "generic imported robot did not compile the authored task contract"
        );
    }
    return compiled.task.fingerprint();
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
            layout.actorFrameSize != 96u ||
            layout.actorHistoryLength != 5u ||
            layout.actorObservationSize != 480u ||
            layout.criticObservationSize != 696u ||
            layout.contactMetricCount != 16u ||
            layout.delayStateCount != 3u) {
            fail("compiled G1 task layout changed");
        }
        if (program.worldFingerprint() != world.fingerprint() ||
            world.capacities().candidatePairs != 256u ||
            world.capacities().rawContacts != 2048u ||
            world.capacities().manifolds != 256u ||
            world.capacities().constraintBlocks != 1024u ||
            world.capacities().constraintRows != 3072u ||
            world.capacities().endpointRuntimeRecords != 2048u ||
            world.capacities().articulationPointQueries != 2048u ||
            program.header().root.y != 0u ||
            program.header().counts0.w != 7u ||
            program.header().counts1.w != 19u ||
            program.header().counts2.x != 4u ||
            program.terrainSampleOffsets().size() != 187u ||
            program.terrainResetTranslations().size() != 11u) {
            fail("compiled G1 task tables are incomplete");
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
                    metalrobo::PolicyActivation::tanh,
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
            compiledPolicy.layout().observationCount !=
                layout.actorObservationSize ||
            compiledPolicy.layout().actionCount !=
                layout.actionCount ||
            compiledPolicy.layout().maximumHiddenCount !=
                32u) {
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
        const std::uint64_t importedFingerprint =
            compileImportedRobotFixture();

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
            << " imported=" << importedFingerprint
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "task_program_check failed: "
            << error.what() << '\n';
        return 1;
    }
}
