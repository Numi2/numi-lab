#include "metalrobo/c_api.h"

#include "metalrobo/EpisodeTwinCompiler.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalRunInspector.hpp"
#include "metalrobo/MetalMeasuredSurfaceMechanics.hpp"
#include "metalrobo/MetalTactile.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/RuntimeAbi.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include "metalrobo/WorldPack.hpp"

#include <cstring>
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct MRWorldFamilyHandle {
    std::atomic_uint32_t references{1u};
    metalrobo::MetalWorldFamilyContext context;
    metalrobo::WorldFamily family;
    metalrobo::ScenarioSchema scenarioSchema;
    std::uint64_t authoredPackHash = 0u;
    metalrobo::WorldInstanceBatch readback;
    std::string deviceName;
    double lastSampleMilliseconds = 0.0;
};

struct MRHybridRendererHandle {
    std::atomic_uint32_t references{1u};
    metalrobo::MetalHybridRenderer renderer;
    metalrobo::HybridObservationBatch readback;
    std::string deviceName;
    std::uint32_t activeEnvironmentCount = 0u;
    double lastRenderMilliseconds = 0.0;
};

struct MRTactileHandle {
    metalrobo::MetalTactileContext context;
    metalrobo::TactileObservationBatch readback;
    std::string deviceName;
    std::string observationMetadataJSON;
    std::uint32_t activeEnvironmentCount = 0u;
    double lastObserveMilliseconds = 0.0;
};

struct MRTaskVisualRuntime {
    metalrobo::WorldTemplate worldTemplate;
    metalrobo::WorldFamily family;
    metalrobo::MetalWorldFamilyContext worlds;
    metalrobo::MetalHybridRenderer renderer;
    metalrobo::MetalHybridRenderer captureRenderer;
    metalrobo::MetalHybridObjectTracker tracker;
    std::unique_ptr<metalrobo::MetalRunInspector> inspector;
    std::vector<MRBodyStateGPU> previousCaptureBodies;
    std::uint64_t captureFrameIndex = 0u;
    bool captureEnabled = false;
    bool capturePolicyCamera = false;
    bool captureUsesLiveDeformation = false;
    bool deviceObservationEnabled = false;
    std::uint64_t sceneFingerprint = 0u;
};

struct MRTaskOutcomeDescriptor {
    std::string id;
    std::string unit;
    std::uint32_t source = MR_TASK_OUTCOME_REWARD;
    std::uint32_t direction = MR_TASK_OUTCOME_NEUTRAL;
};

struct MRTaskRolloutHandle {
    explicit MRTaskRolloutHandle(
        metalrobo::MetalWorldConfig config,
        metalrobo::CompiledRun compiled
    ) : run(std::move(compiled)),
        model(run.model()),
        world(run.world()),
        taskProgram(run.task()),
        context(std::move(config)) {}

    metalrobo::CompiledRun run;
    const metalrobo::EngineModel& model;
    const metalrobo::CompiledWorld& world;
    const metalrobo::CompiledTaskProgram& taskProgram;
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResidentState residentState;
    metalrobo::MetalWorldStepConfig stepConfig;
    std::vector<MRBodyStateGPU> defaultSceneBodies;
    std::vector<float> resetQ;
    std::vector<float> resetV;
    std::vector<MRBodyStateGPU> resetSceneBodies;
    std::vector<std::uint32_t> resetMasks;
    metalrobo::MetalWorldResult result;
    std::vector<std::uint32_t> statusCodes;
    std::vector<std::uint32_t> activeContacts;
    std::vector<MRTaskOutcomeDescriptor> outcomes;
    std::vector<float> outcomeValues;
    std::unique_ptr<MRTaskVisualRuntime> visualRuntime;
    std::unique_ptr<MRTaskVisualRuntime> inspectionVisualRuntime;
    std::unique_ptr<metalrobo::MetalMeasuredSurfaceMechanics>
        measuredSurfaceRuntime;
    std::string deviceName;
    std::string metallibPath;
    std::string taskId;
    std::uint32_t environmentCount = 0u;
    std::uint64_t submittedControlSteps = 0u;
    std::uint64_t completedEnvironmentSteps = 0u;
    std::uint64_t submissionCount = 0u;
    double totalGPUMilliseconds = 0.0;
    double totalSubmissionMilliseconds = 0.0;
};

namespace {
void installTaskVisualRuntime(
    MRTaskRolloutHandle& handle,
    const metalrobo::VisualSensorProgram& program
);
void installTaskInspectionRuntime(
    MRTaskRolloutHandle& handle,
    const metalrobo::VisualSensorProgram& program
);
}

static_assert(sizeof(MRHybridGaussianC) == 80u);
static_assert(sizeof(MRHybridGaussianGPU) == 80u);
static_assert(
    sizeof(MRVisualFrameMetadataC) ==
    sizeof(MRVisualFrameMetadataGPU)
);
static_assert(
    sizeof(MRTactileSummaryC) ==
    sizeof(MRTactileSummaryGPU)
);
static_assert(
    sizeof(MRTaskTransitionC) ==
    sizeof(MRTaskTransitionGPU)
);
static_assert(
    offsetof(MRTaskTransitionC, policy_revision) ==
    offsetof(MRTaskTransitionGPU, policyRevision)
);
static_assert(
    offsetof(MRTaskTransitionC, timeout_bootstrap_value) ==
    offsetof(MRTaskTransitionGPU, timeoutBootstrapValue)
);
static_assert(
    offsetof(MRTaskTransitionC, episode_tracking_score) ==
    offsetof(MRTaskTransitionGPU, episodeTrackingScore)
);
static_assert(
    offsetof(MRTaskTransitionC, difficulty_band) ==
    offsetof(MRTaskTransitionGPU, taskProgress)
);
static_assert(sizeof(MRTaskEvidenceTelemetryC) == 48u);

namespace {

thread_local std::string gLastError;

std::string unitreeG1DeploymentContractJSON() {
    const metalrobo::G1ModelMetadata& metadata =
        metalrobo::unitreeG1Metadata();
    const metalrobo::EngineModel model =
        metalrobo::makeUnitreeG1EngineModel();
    const MRArticulationGPU& articulation =
        model.articulations.front();
    std::ostringstream output;
    output << std::setprecision(9);
    output
        << "{\"format\":\"metalrobo.unitree-g1-deployment\","
        << "\"schema\":1,"
        << "\"model_name\":" << std::quoted(
            std::string{metadata.modelName}
        ) << ","
        << "\"source_repository\":" << std::quoted(
            std::string{metadata.sourceRepository}
        ) << ","
        << "\"source_commit\":" << std::quoted(
            std::string{metadata.sourceCommit}
        ) << ","
        << "\"source_model_path\":" << std::quoted(
            std::string{metadata.sourceModelPath}
        ) << ","
        << "\"simulator_repository\":" << std::quoted(
            std::string{metadata.simulatorRepository}
        ) << ","
        << "\"simulator_commit\":" << std::quoted(
            std::string{metadata.simulatorCommit}
        ) << ","
        << "\"simulator_model_path\":" << std::quoted(
            std::string{metadata.simulatorModelPath}
        ) << ","
        << "\"rl_preset_repository\":" << std::quoted(
            std::string{metadata.rlPresetRepository}
        ) << ","
        << "\"rl_preset_commit\":" << std::quoted(
            std::string{metadata.rlPresetCommit}
        ) << ","
        << "\"physics_timestep_seconds\":0.005,"
        << "\"policy_timestep_seconds\":0.02,"
        << "\"drive_prediction_seconds\":0.005,"
        << "\"solver_root_frame\":\"center_of_mass\","
        << "\"interaction_root_frame\":\"root_link_origin\","
        << "\"root_center_of_mass_local_xyz\":["
        << model.bodies[articulation.rootBody].centerOfMass.x << ','
        << model.bodies[articulation.rootBody].centerOfMass.y << ','
        << model.bodies[articulation.rootBody].centerOfMass.z << "],"
        << "\"action_scale_radians\":0.25,"
        << "\"actor_frame_size\":96,"
        << "\"actor_history_length\":5,";

    const auto writeScalarArray =
        [&output](const char* key, const auto& values) {
            output << std::quoted(key) << ":[";
            for (std::size_t index = 0u;
                 index < values.size();
                 ++index) {
                if (index != 0u) {
                    output << ',';
                }
                output << values[index];
            }
            output << ']';
        };
    output << "\"joint_order\":[";
    for (std::size_t index = 0u;
         index < metadata.jointLimits.size();
         ++index) {
        if (index != 0u) {
            output << ',';
        }
        output << std::quoted(
            std::string{metadata.jointLimits[index].name}
        );
    }
    output << "],";

    std::array<float, metalrobo::kUnitreeG1JointCount>
        defaultPose{};
    std::array<float, metalrobo::kUnitreeG1JointCount>
        stiffness{};
    std::array<float, metalrobo::kUnitreeG1JointCount>
        damping{};
    std::array<float, metalrobo::kUnitreeG1JointCount>
        velocityLimits{};
    std::array<float, metalrobo::kUnitreeG1JointCount>
        effortLimits{};
    const std::span<const float> actionScales =
        metalrobo::unitreeG1LocomotionActionScales();
    for (std::size_t index = 0u;
         index < metadata.jointLimits.size();
         ++index) {
        defaultPose[index] =
            model.defaultQ[
                articulation.qOffset + 7u + index
            ];
        stiffness[index] =
            metadata.rlLabDrives[index].stiffness;
        damping[index] =
            metadata.rlLabDrives[index].damping;
        velocityLimits[index] =
            metadata.jointLimits[index].maximumVelocity;
        effortLimits[index] =
            metadata.jointLimits[index].maximumEffort;
    }
    writeScalarArray("default_pose", defaultPose);
    output << ',';
    writeScalarArray("task_action_scale", actionScales);
    output << ',';
    writeScalarArray("stiffness", stiffness);
    output << ',';
    writeScalarArray("damping", damping);
    output << ',';
    writeScalarArray("velocity_limits", velocityLimits);
    output << ',';
    writeScalarArray("effort_limits", effortLimits);
    output << ",\"position_limits\":[";
    for (std::size_t index = 0u;
         index < metadata.jointLimits.size();
         ++index) {
        if (index != 0u) {
            output << ',';
        }
        output
            << '['
            << metadata.jointLimits[index].lowerPosition
            << ','
            << metadata.jointLimits[index].upperPosition
            << ']';
    }
    output << "]}";
    return output.str();
}

template <typename Function>
int translateErrors(Function&& function) noexcept {
    try {
        function();
        gLastError.clear();
        return 0;
    } catch (const std::exception& error) {
        gLastError = error.what();
    } catch (...) {
        gLastError = "MetalRobo failed with an unknown native exception.";
    }
    return -1;
}

bool requireWorldFamilyHandle(const MRWorldFamilyHandle* handle) {
    if (handle != nullptr) {
        return true;
    }
    gLastError = "MetalRobo world-family handle is null.";
    return false;
}

bool requireHybridRendererHandle(
    const MRHybridRendererHandle* handle
) {
    if (handle != nullptr) {
        return true;
    }
    gLastError = "MetalRobo hybrid renderer handle is null.";
    return false;
}

bool requireTactileHandle(const MRTactileHandle* handle) {
    if (handle != nullptr) {
        return true;
    }
    gLastError = "MetalRobo tactile handle is null.";
    return false;
}

bool requireTaskRolloutHandle(
    const MRTaskRolloutHandle* handle
) {
    if (handle != nullptr) {
        return true;
    }
    gLastError = "MetalRobo task-rollout handle is null.";
    return false;
}

void accumulateTaskStageHighWater(
    MRTaskRolloutStageHighWaterC& target,
    const metalrobo::MetalWorldStageCounts& source
) {
    const auto maximum = [](uint32_t& value, const uint32_t sample) {
        value = std::max(value, sample);
    };
    maximum(target.candidate_pairs, source.candidatePairs);
    maximum(target.raw_contacts, source.rawContacts);
    maximum(target.manifolds, source.manifolds);
    maximum(target.constraint_blocks, source.constraintBlocks);
    maximum(target.constraint_rows, source.constraintRows);
    maximum(target.islands, source.islands);
    maximum(target.hard_convex_pairs, source.hardConvexPairs);
    maximum(
        target.mesh_triangle_candidates,
        source.meshTriangleCandidates
    );
    maximum(target.solver_tiles, source.solverTiles);
    maximum(target.spill_rows, source.spillRows);
    maximum(target.ccd_candidates, source.ccdCandidates);
    maximum(target.ccd_events, source.ccdEvents);
    maximum(
        target.endpoint_runtime_records,
        source.endpointRuntimeRecords
    );
    maximum(
        target.articulation_point_queries,
        source.articulationPointQueries
    );
    maximum(target.rod_candidate_pairs, source.rodCandidatePairs);
    maximum(target.rod_raw_contacts, source.rodRawContacts);
    maximum(target.rod_manifolds, source.rodManifolds);
    maximum(target.rod_ccd_events, source.rodCCDEvents);
    maximum(
        target.quality_generalized_velocities,
        source.qualityGeneralizedVelocities
    );
    maximum(target.quality_rows, source.qualityRows);
    maximum(
        target.quality_krylov_vectors,
        source.qualityKrylovVectors
    );
    maximum(
        target.quality_direct_tiles,
        source.qualityDirectTiles
    );
    maximum(target.dynamic_nodes, source.dynamicNodes);
    maximum(
        target.island_node_references,
        source.islandNodeReferences
    );
    maximum(
        target.island_constraint_references,
        source.islandConstraintReferences
    );
    maximum(target.rod_factor_blocks, source.rodFactorBlocks);
    maximum(
        target.operator_velocity_elements,
        source.operatorVelocityElements
    );
}

std::size_t checkedProduct(
    const std::initializer_list<std::size_t> values,
    const char* label
) {
    std::size_t product = 1u;
    for (const std::size_t value : values) {
        if (value != 0u &&
            product >
                std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error(
                std::string{label} + " size overflows size_t"
            );
        }
        product *= value;
    }
    return product;
}

void resetTaskRolloutState(
    MRTaskRolloutHandle& handle,
    const std::uint64_t seed
) {
    const std::size_t environmentCount =
        handle.environmentCount;
    const std::size_t nq = handle.world.nq();
    const std::size_t nv = handle.world.nv();
    const MRArticulationGPU& articulation =
        handle.model.articulations[
            handle.world.articulationIndex()
        ];
    handle.resetQ.resize(environmentCount * nq);
    handle.resetV.resize(environmentCount * nv);
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        const std::size_t qBase = environment * nq;
        const std::size_t vBase = environment * nv;
        std::copy_n(
            handle.model.defaultQ.begin() +
                articulation.qOffset,
            nq,
            handle.resetQ.begin() +
                static_cast<std::ptrdiff_t>(qBase)
        );
        std::copy_n(
            handle.model.defaultV.begin() +
                articulation.vOffset,
            nv,
            handle.resetV.begin() +
                static_cast<std::ptrdiff_t>(vBase)
        );
    }

    const std::size_t sceneBodyCount =
        handle.defaultSceneBodies.size();
    handle.resetSceneBodies.resize(
        environmentCount * sceneBodyCount
    );
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        std::copy(
            handle.defaultSceneBodies.begin(),
            handle.defaultSceneBodies.end(),
            handle.resetSceneBodies.begin() +
                static_cast<std::ptrdiff_t>(
                    environment * sceneBodyCount
                )
        );
    }

    handle.stepConfig.taskSeed = seed;
    handle.residentState =
        metalrobo::MetalWorldResidentState{};
    if (handle.visualRuntime != nullptr) {
        const auto sampled = handle.visualRuntime->worlds.sample(
            handle.environmentCount,
            seed
        );
        if (!sampled.succeeded()) {
            throw std::runtime_error(
                std::string{"visual WorldFamily reset failed ["} +
                metalrobo::metalWorldFamilyStatusName(sampled.status) +
                "]: " + sampled.message
            );
        }
        if (handle.visualRuntime->deviceObservationEnabled) {
            const auto tracker = handle.visualRuntime->tracker.reset();
            if (!tracker.succeeded()) {
                throw std::runtime_error(
                    std::string{"visual tracker reset failed ["} +
                    metalrobo::metalHybridRendererStatusName(tracker.status) +
                    "]: " + tracker.message
                );
            }
        }
    }
    handle.resetMasks.clear();
    handle.result = {};
    handle.statusCodes.clear();
    handle.activeContacts.clear();
}

void validateTaskRolloutConfiguration(
    const MRTaskRolloutConfigC& config
) {
    if (config.environment_count == 0u ||
        config.physics_substeps == 0u ||
        config.velocity_iterations == 0u ||
        !std::isfinite(config.control_timestep_seconds) ||
        !(config.control_timestep_seconds > 0.0f)) {
        throw std::invalid_argument(
            "task-rollout counts and timing must be finite and positive"
        );
    }
    if (config.dynamic_sphere_count > 0u &&
        config.dynamic_spheres == nullptr) {
        throw std::invalid_argument(
            "task-rollout dynamic sphere storage is null"
        );
    }
    if (config.dynamic_sphere_count > 64u) {
        throw std::invalid_argument(
            "task-rollout dynamic sphere count exceeds 64"
        );
    }
    if (config.interaction_reference_mode >
        MR_INTERACTION_REFERENCE_RESET_ONLY) {
        throw std::invalid_argument(
            "task-rollout interaction reference mode is invalid"
        );
    }
    if (config.override_interaction_student_authority > 1u ||
        (config.override_interaction_student_authority != 0u &&
         (!std::isfinite(config.interaction_student_authority) ||
          config.interaction_student_authority < 0.0f ||
          config.interaction_student_authority > 1.0f))) {
        throw std::invalid_argument(
            "task-rollout interaction student authority must be in [0, 1]"
        );
    }
    if (config.materialize_articulated_contact_responses > 1u) {
        throw std::invalid_argument(
            "materialized articulated-contact response flag is invalid"
        );
    }
    if (config.override_difficulty_band_range > 1u ||
        (config.override_difficulty_band_range != 0u &&
         config.minimum_difficulty_band >
             config.maximum_difficulty_band)) {
        throw std::invalid_argument(
            "task-rollout difficulty-band range is invalid"
        );
    }
    if (config.override_interaction_push > 1u ||
        (config.override_interaction_push != 0u &&
         (!std::isfinite(config.interaction_push_maximum_velocity) ||
          config.interaction_push_maximum_velocity < 0.0f ||
          !std::isfinite(config.interaction_push_interval_seconds) ||
          !(config.interaction_push_interval_seconds > 0.0f)))) {
        throw std::invalid_argument(
            "task-rollout interaction push requires a nonnegative finite velocity and positive finite interval"
        );
    }
}

metalrobo::LocomotionSurface locomotionSurface(
    const std::uint32_t value
) {
    switch (value) {
    case MR_LOCOMOTION_SURFACE_GROUND:
        return metalrobo::LocomotionSurface::ground;
    case MR_LOCOMOTION_SURFACE_TERRAIN:
        return metalrobo::LocomotionSurface::terrain;
    default:
        throw std::invalid_argument(
            "locomotion surface is invalid"
        );
    }
}

metalrobo::UnitreeG1Task unitreeG1Task(const std::uint32_t value) {
    switch (value) {
    case MR_UNITREE_G1_TASK_VELOCITY:
        return metalrobo::UnitreeG1Task::velocity;
    case MR_UNITREE_G1_TASK_DISTURBANCE_RECOVERY:
        return metalrobo::UnitreeG1Task::disturbanceRecovery;
    case MR_UNITREE_G1_TASK_SUPINE_GET_UP_DISCOVERY:
        return metalrobo::UnitreeG1Task::supineGetUpDiscovery;
    case MR_UNITREE_G1_TASK_BALL_DISTURBANCE_RECOVERY:
        return metalrobo::UnitreeG1Task::ballDisturbanceRecovery;
    case MR_UNITREE_G1_TASK_BALL_DODGE:
        return metalrobo::UnitreeG1Task::ballDodge;
    case MR_UNITREE_G1_TASK_DEVELOPMENTAL_RECOVERY:
        return metalrobo::UnitreeG1Task::developmentalRecovery;
    case MR_UNITREE_G1_TASK_ADULT_LOCOMOTION:
        return metalrobo::UnitreeG1Task::adultLocomotion;
    case MR_UNITREE_G1_TASK_G1_LEGS_LOCOMOTION:
        return metalrobo::UnitreeG1Task::g1LegsLocomotion;
    default:
        throw std::invalid_argument("Unitree G1 task is invalid");
    }
}

const metalrobo::InteractionClip& selectedInteractionClip(
    const metalrobo::InteractionPack& pack,
    const std::string_view clipId
) {
    const auto found = std::find_if(
        pack.clips.begin(),
        pack.clips.end(),
        [clipId](const metalrobo::InteractionClip& clip) {
            return clip.id == clipId;
        }
    );
    if (found == pack.clips.end()) {
        throw std::invalid_argument(
            "InteractionPack clip does not exist: " +
            std::string{clipId}
        );
    }
    return *found;
}

void authorG1InteractionTrackingTask(
    metalrobo::TaskPack& task,
    metalrobo::TaskObservationProgram& observations,
    metalrobo::TaskResetProgram& reset,
    const metalrobo::InteractionPack& interactions,
    const metalrobo::InteractionClip& clip,
    const MRTaskRolloutConfigC& config
) {
    task.id = "unitree_g1_interaction_tracking/" +
        interactions.id + "/" + clip.id;
    task.outcomes = {
        {"tracking", "ratio",
            metalrobo::TaskOutcomeSource::trackingScore,
            metalrobo::TaskOutcomeDirection::higherIsBetter},
        {"contact_reward", "reward",
            metalrobo::TaskOutcomeSource::contactReward,
            metalrobo::TaskOutcomeDirection::higherIsBetter},
    };
    // Publish solver-resident world-frame wrist poses through the existing
    // compact motion-feature stream. The static authored terrain is the
    // anchor, so the position triplet for each tracked wrist is a world-space
    // COM position produced from the accepted Metal body state. This is an
    // inspection/qualification boundary, not an actor observation or reward.
    task.motion = {
        .anchorBody = task.terrain.body,
        .trackedBodies = {
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        },
    };
    task.difficultyBandCount = 1u;
    task.commands = {};
    task.commands.standingProbability = 1.0f;
    task.pushes.maximumVelocity = 0.0f;
    task.pushes.minimumIntervalSeconds = 5.0f;
    task.pushes.maximumIntervalSeconds = 5.0f;
    if (config.override_interaction_push != 0u) {
        task.pushes.maximumVelocity =
            config.interaction_push_maximum_velocity;
        task.pushes.minimumIntervalSeconds =
            config.interaction_push_interval_seconds;
        task.pushes.maximumIntervalSeconds =
            config.interaction_push_interval_seconds;
    }
    reset.operators.clear();
    // The standing profile's 32-manifold envelope is too small for generated
    // whole-body motions. The first physically coherent ARDY residual batch
    // measured 33 simultaneous manifolds (37 raw contacts) at control step
    // zero. Reuse the qualified two-Wave32 get-up envelope without changing
    // contact generation, solver acceptance, or transactional rollback.
    task.capacities.manifolds = std::max(
        task.capacities.manifolds,
        64u
    );
    // A generic generated motion may deliberately leave the standing height
    // and orientation envelope (acrobatics, get-up, crawling, manipulation).
    // The InteractionPack root target is the task-relative posture authority;
    // inherited locomotion tilt/height terminations would reject the desired
    // motion before physics can evaluate it. Numerical failures remain owned
    // by the transactional solver and non-looping clips retain their horizon.
    const bool physicsGated =
        config.interaction_reference_mode !=
        MR_INTERACTION_REFERENCE_RESET_ONLY;
    task.interactionPhysicsGated = physicsGated;
    if (physicsGated) {
        task.id += "/physics-gated-v4";
        // Band zero is the exact nominal replay. Band one is a bounded,
        // seed-replayable robustness condition for generated upper-body
        // skills: the same reference must tolerate modest stiffness and
        // damping calibration error without changing its kinematic intent.
        task.difficultyBandCount = 2u;
        reset.operators = {
            {
                .operation = metalrobo::TaskRandomizationOperator::
                    controllerParameter,
                .component = 0u,
                .minimumDifficultyBand = 1u,
                .parameters = {0.95f, 1.05f, 0.0f, 0.0f},
            },
            {
                .operation = metalrobo::TaskRandomizationOperator::
                    controllerParameter,
                .component = 1u,
                .minimumDifficultyBand = 1u,
                .parameters = {0.95f, 1.05f, 0.0f, 0.0f},
            },
        };
    }
    task.terminations.clear();
    if (physicsGated && config.disable_task_terminations == 0u) {
        // ARDY_PHYSICS_GATED_REFERENCE_V4: stop before ground contact
        // can become a second, uncontrolled phase of the motion.
        task.terminations = {
            {
                .operation = metalrobo::TaskTerminationOperator::
                    minimumRootHeight,
                .reason = MR_TASK_TERMINATION_HEIGHT,
                .priority = 10u,
                .threshold = 0.55f,
                .failurePenalty = -5.0f,
            },
            {
                .operation = metalrobo::TaskTerminationOperator::
                    maximumTilt,
                .reason = MR_TASK_TERMINATION_TILT,
                .priority = 11u,
                .threshold = 0.50f,
                .failurePenalty = -5.0f,
            },
            {
                .operation = metalrobo::TaskTerminationOperator::
                    contactGroup,
                .sourceGroup = "undesired_contact",
                .reason = MR_TASK_TERMINATION_CONTACT,
                .priority = 12u,
                .threshold = 0.5f,
                .failurePenalty = -5.0f,
            },
        };
    }
    reset.maximumActionDelaySteps = 0u;
    reset.maximumObservationDelaySteps = 0u;
    if (!clip.loop) {
        const double durationSeconds =
            static_cast<double>(clip.frameCount - 1u) /
            static_cast<double>(clip.framesPerSecond);
        const double nominalSteps = std::ceil(
            durationSeconds /
            static_cast<double>(config.control_timestep_seconds)
        ) + 1.0;
        // A physics-gated clock may legitimately hold a frame while a
        // foot releases or lands. Give it four nominal clip durations.
        const double steps = physicsGated
            ? 4.0 * nominalSteps
            : nominalSteps;
        if (!std::isfinite(steps) || steps < 2.0 ||
            steps > static_cast<double>(
                std::numeric_limits<std::uint32_t>::max()
            )) {
            throw std::invalid_argument(
                "InteractionPack duration does not fit the task horizon"
            );
        }
        task.maximumEpisodeSteps =
            static_cast<std::uint32_t>(steps);
    }

    // Generated-motion tracking has no locomotion command, and the selected
    // interaction publishes its own phase below. Reuse those five otherwise
    // dead/redundant actor slots for solver-resolved support state without
    // changing the policy shape: total load, left/right load balance, maximum
    // support slip, and each foot's actual support load. ARDY remains intent;
    // these values come only from accepted NumiSolver contacts.
    const auto installSupportState = [](
        std::vector<metalrobo::TaskObservationOperatorSpec>& observations
    ) {
        for (metalrobo::TaskObservationOperatorSpec& observation :
             observations) {
            if (observation.source ==
                    metalrobo::TaskObservationSource::command) {
                observation.source =
                    metalrobo::TaskObservationSource::supportSense;
                observation.target.clear();
                observation.scale = observation.component == 0u
                    ? 0.002f
                    : 1.0f;
                observation.offset = 0.0f;
                observation.noiseAmplitude = 0.0f;
                observation.biasLower = 0.0f;
                observation.biasUpper = 0.0f;
                observation.normalizeVector3 = false;
                continue;
            }
            if (observation.source !=
                    metalrobo::TaskObservationSource::gaitPhase) {
                continue;
            }
            observation.source =
                metalrobo::TaskObservationSource::contactMetric;
            observation.target = observation.component == 0u
                ? "left_foot_contact"
                : "right_foot_contact";
            observation.component = 0u;
            observation.scale = 0.002f;
            observation.offset = 0.0f;
            observation.noiseAmplitude = 0.0f;
            observation.biasLower = 0.0f;
            observation.biasUpper = 0.0f;
            observation.normalizeVector3 = false;
        }
    };
    installSupportState(observations.actorFrame);
    installSupportState(observations.critic);

    const auto appendObservation =
        [&observations](const metalrobo::TaskObservationOperatorSpec& operation) {
            observations.actorCurrent.push_back(operation);
            observations.critic.push_back(operation);
        };
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        appendObservation({
            .source = metalrobo::TaskObservationSource::interactionPhase,
            .component = component,
        });
    }
    for (std::uint32_t component = 0u; component < 12u; ++component) {
        appendObservation({
            .source = metalrobo::TaskObservationSource::
                interactionRootTrackingError,
            .component = component,
        });
    }
    for (const metalrobo::TaskActionBinding& action : task.actions) {
        appendObservation({
            .source = metalrobo::TaskObservationSource::
                interactionJointPositionError,
            .target = action.actuator,
        });
    }
    for (const metalrobo::InteractionContactTrack& track :
         interactions.contactTracks) {
        for (std::uint32_t component = 0u; component < 2u; ++component) {
            appendObservation({
                .source = metalrobo::TaskObservationSource::
                    interactionContactMode,
                .target = track.id,
                .component = component,
            });
        }
        for (std::uint32_t component = 0u;
             component < metalrobo::kInteractionContactFeatureCount;
             ++component) {
            appendObservation({
                .source = metalrobo::TaskObservationSource::
                    interactionContactTarget,
                .target = track.id,
                .component = component,
            });
            appendObservation({
                .source = metalrobo::TaskObservationSource::
                    interactionContactValidity,
                .target = track.id,
                .component = component,
            });
        }
    }

    task.rewards = {
        {
            .operation = metalrobo::TaskRewardOperator::
                interactionJointTracking,
            .weight = 0.5f,
            .parameters = {0.25f, 1.0f, 0.0f, 0.0f},
        },
        {
            .operation = metalrobo::TaskRewardOperator::
                interactionRootTracking,
            .weight = 3.0f,
            // Broad widths retain a useful gradient after a dynamic tracking
            // error instead of saturating the exponential while the robot is
            // still recoverable.
            .parameters = {1.0f, 0.20f, 1.0f, 1.0f},
        },
        {
            .operation = metalrobo::TaskRewardOperator::
                interactionRootLinearVelocityError,
            .weight = -0.75f,
            .parameters = {1.0f, 0.0f, 0.0f, 0.0f},
        },
        // Do not add standing-relative height, uprightness, tilt, or root
        // velocity costs here.  Generated motion may intentionally crouch,
        // get up, turn, jump, or rotate.  interactionRootTracking already
        // scores position, orientation, linear velocity, and angular velocity
        // against the selected trajectory, so the same compiled objective is
        // meaningful for every motion without classifying it on the host.
        {
            .operation = metalrobo::TaskRewardOperator::
                jointVelocitySquared,
            .weight = -0.001f,
        },
        {
            .operation = metalrobo::TaskRewardOperator::
                jointAccelerationSquared,
            .weight = -2.5e-7f,
        },
        {
            .operation = metalrobo::TaskRewardOperator::
                actionRateSquared,
            .weight = -0.02f,
        },
        {
            .operation = metalrobo::TaskRewardOperator::
                jointLimitViolationAbsolute,
            .weight = -5.0f,
            .parameters = {0.9f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation = metalrobo::TaskRewardOperator::mechanicalPower,
            .weight = -2.0e-5f,
        },
        {
            .operation = metalrobo::TaskRewardOperator::supportSlip,
            .weight = -0.25f,
        },
        {
            .operation = metalrobo::TaskRewardOperator::forbiddenContact,
            .sourceGroup = "undesired_contact",
            .weight = -3.0f,
        },
    };
    for (const metalrobo::InteractionContactTrack& track :
         interactions.contactTracks) {
        task.rewards.push_back({
            .operation = metalrobo::TaskRewardOperator::
                interactionContactTracking,
            .sourceGroup = track.taskContactGroup,
            .target = track.id,
            .weight = 1.0f,
            .parameters = {0.65f, 1.0f, 0.0f, 0.0f},
        });
    }
}

void authorG1ImaginedTask(
    metalrobo::TaskPack& task,
    metalrobo::TaskObservationProgram& observations,
    const metalrobo::InteractionPack& interactions,
    const metalrobo::InteractionClip& clip,
    const MRTaskRolloutConfigC& config,
    const bool includeInteractionContacts
) {
    task.id += "/imagined_interaction";
    if (std::ranges::none_of(
            task.outcomes,
            [](const metalrobo::TaskOutcomeSpec& outcome) {
                return outcome.source ==
                    metalrobo::TaskOutcomeSource::trackingScore;
            }
        )) {
        task.outcomes.push_back({
            "tracking", "ratio",
            metalrobo::TaskOutcomeSource::trackingScore,
            metalrobo::TaskOutcomeDirection::higherIsBetter,
        });
    }
    if (!clip.loop) {
        const double durationSeconds =
            static_cast<double>(clip.frameCount - 1u) /
            static_cast<double>(clip.framesPerSecond);
        const double steps = std::ceil(
            durationSeconds /
            static_cast<double>(config.control_timestep_seconds)
        ) + 1.0;
        if (!std::isfinite(steps) || steps < 2.0 ||
            steps > static_cast<double>(
                std::numeric_limits<std::uint32_t>::max()
            )) {
            throw std::invalid_argument(
                "InteractionPack duration does not fit the imagined-task horizon"
            );
        }
        task.maximumEpisodeSteps = static_cast<std::uint32_t>(steps);
    }
    const auto appendObservation =
        [&observations](const metalrobo::TaskObservationOperatorSpec& operation) {
            observations.critic.push_back(operation);
        };
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        const metalrobo::TaskObservationOperatorSpec phase{
            .source = metalrobo::TaskObservationSource::interactionPhase,
            .component = component,
        };
        // Elapsed recovery phase is a deployable mode-clock signal. It lets
        // the actor distinguish rising from settling without exposing the
        // imagined joint/contact targets, which remain critic-only intent.
        observations.actorCurrent.push_back(phase);
        observations.critic.push_back(phase);
    }
    for (const metalrobo::TaskActionBinding& action : task.actions) {
        appendObservation({
            .source = metalrobo::TaskObservationSource::
                interactionJointPositionError,
            .target = action.actuator,
        });
    }
    if (includeInteractionContacts) for (
        const metalrobo::InteractionContactTrack& track :
        interactions.contactTracks
    ) {
        for (std::uint32_t component = 0u; component < 2u; ++component) {
            appendObservation({
                .source = metalrobo::TaskObservationSource::
                    interactionContactMode,
                .target = track.id,
                .component = component,
            });
        }
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            appendObservation({
                .source = metalrobo::TaskObservationSource::
                    contactWrenchLocal,
                .target = track.taskContactGroup,
                .component = component,
                .scale = 0.01f,
            });
        }
    }
    task.rewards.push_back({
        .operation = metalrobo::TaskRewardOperator::
            interactionJointTracking,
        .weight = 0.5f,
        .parameters = {0.25f, 1.0f, 0.0f, 0.0f},
    });
    task.rewards.push_back({
        .operation = metalrobo::TaskRewardOperator::
            interactionRootTracking,
        .weight = 2.0f,
        .parameters = {0.04f, 0.08f, 0.25f, 0.5f},
    });
    if (includeInteractionContacts) for (
        const metalrobo::InteractionContactTrack& track :
        interactions.contactTracks
    ) {
        task.rewards.push_back({
            .operation = metalrobo::TaskRewardOperator::
                interactionContactTracking,
            .sourceGroup = track.taskContactGroup,
            .target = track.id,
            .weight = 1.0f,
            .parameters = {1.0f, 1.0f, 1.0f, 0.0f},
        });
    }
}

std::unique_ptr<MRTaskRolloutHandle>
createTaskRolloutHandle(
    metalrobo::CompiledRun run,
    const char* metallibPath,
    std::string taskId,
    const std::string_view source
) {
    metalrobo::MetalWorldConfig worldConfig;
    if (metallibPath != nullptr && metallibPath[0] != '\0') {
        worldConfig.metallibPath = metallibPath;
    }
    worldConfig.maximumInFlightSubmissions = 1u;
    auto handle = std::make_unique<MRTaskRolloutHandle>(
        std::move(worldConfig),
        std::move(run)
    );
    if (metallibPath != nullptr && metallibPath[0] != '\0') {
        handle->metallibPath = metallibPath;
    }
    const metalrobo::CompiledTaskProgram& taskProgram =
        handle->run.task();
    const auto appendOutcome = [&handle](
        std::string id,
        std::string unit,
        const std::uint32_t source,
        const std::uint32_t direction
    ) {
        handle->outcomes.push_back({
            std::move(id), std::move(unit), source, direction
        });
    };
    appendOutcome("reward", "reward", MR_TASK_OUTCOME_REWARD,
        MR_TASK_OUTCOME_HIGHER_IS_BETTER);
    appendOutcome("task_reward", "reward", MR_TASK_OUTCOME_TASK_REWARD,
        MR_TASK_OUTCOME_HIGHER_IS_BETTER);
    appendOutcome("done", "bool", MR_TASK_OUTCOME_DONE,
        MR_TASK_OUTCOME_LOWER_IS_BETTER);
    appendOutcome("timeout", "bool", MR_TASK_OUTCOME_TIMEOUT,
        MR_TASK_OUTCOME_NEUTRAL);
    appendOutcome("physics_error", "bool", MR_TASK_OUTCOME_PHYSICS_ERROR,
        MR_TASK_OUTCOME_LOWER_IS_BETTER);
    for (const metalrobo::CompiledTaskOutcomeSpec& outcome :
         taskProgram.outcomes()) {
        appendOutcome(
            outcome.id,
            outcome.unit,
            outcome.source,
            static_cast<std::uint32_t>(outcome.direction)
        );
    }
    handle->defaultSceneBodies.assign(
        handle->run.defaultSceneBodies().begin(),
        handle->run.defaultSceneBodies().end()
    );
    handle->taskId = std::move(taskId);
    if (handle->run.world().sceneBodyCount() !=
            handle->defaultSceneBodies.size()) {
        throw std::runtime_error(
            std::string{source} +
            " scene-state count does not match compiled topology"
        );
    }
    const metalrobo::RunProfile& profile = handle->run.profile();
    handle->environmentCount = profile.environmentCount;
    handle->stepConfig.timestepSeconds = profile.controlTimestepSeconds;
    handle->stepConfig.physicsSubsteps = profile.physicsSubsteps;
    handle->stepConfig.solverMode =
        metalrobo::MetalWorldSolverMode::temporalCone;
    handle->stepConfig.actuationMode =
        metalrobo::MetalWorldActuationMode::implicitPositionDrive;
    handle->stepConfig.velocityIterations = profile.velocityIterations;
    handle->stepConfig.finalVelocityIterations =
        profile.finalVelocityIterations;
    handle->stepConfig.ccdMode = std::any_of(
        handle->model.shapes.begin(),
        handle->model.shapes.end(),
        [](const MRShapeGPU& shape) {
            return (shape.flags & MR_SHAPE_FLAG_ENABLE_CCD) != 0u;
        }
    )
        ? metalrobo::MetalWorldCCDMode::hybrid
        : metalrobo::MetalWorldCCDMode::disabled;
    handle->stepConfig.applyBodyDamping = true;
    handle->stepConfig.deterministic = true;
    handle->stepConfig.warmStart = true;
    handle->stepConfig.streamedArticulatedContactResponses =
        profile.streamedArticulatedContactResponses;
    handle->stepConfig.minimumDifficultyBand =
        profile.minimumDifficultyBand;
    handle->stepConfig.maximumDifficultyBand =
        profile.maximumDifficultyBand;
    handle->stepConfig.captureContactEvidence = false;
    handle->stepConfig.publishFinalState = false;
    handle->stepConfig.publishStateTrajectory = false;
    handle->stepConfig.taskProgram = taskProgram;
    if (const metalrobo::MetalWorldMulticopterProgram* multicopter =
            handle->run.multicopterProgram()) {
        handle->stepConfig.multicopterProgram = *multicopter;
    }
    if (const metalrobo::CompiledMeasuredSurfaceBinding* surface =
            handle->run.measuredSurfaceBinding()) {
        handle->measuredSurfaceRuntime =
            std::make_unique<metalrobo::MetalMeasuredSurfaceMechanics>(
                *surface);
        handle->stepConfig.deviceMechanicsProgram =
            handle->measuredSurfaceRuntime->program();
    }
    if (const metalrobo::VisualSensorProgram* visual =
            handle->run.visualSensorProgram()) {
        installTaskVisualRuntime(*handle, *visual);
    }
    resetTaskRolloutState(*handle, profile.seed);
    return handle;
}

float taskOutcomeValue(
    const MRTaskTransitionC& transition,
    const std::uint32_t source
) {
    switch (source) {
    case MR_TASK_OUTCOME_REWARD: return transition.reward;
    case MR_TASK_OUTCOME_TASK_REWARD: return transition.task_reward;
    case MR_TASK_OUTCOME_TRACKING_SCORE: return transition.tracking_score;
    case MR_TASK_OUTCOME_ROOT_HEIGHT: return transition.root_height;
    case MR_TASK_OUTCOME_TILT: return transition.tilt;
    case MR_TASK_OUTCOME_DONE: return static_cast<float>(transition.done);
    case MR_TASK_OUTCOME_TIMEOUT: return static_cast<float>(transition.timeout);
    case MR_TASK_OUTCOME_PHYSICS_ERROR:
        return static_cast<float>(transition.physics_error);
    case MR_TASK_OUTCOME_CONTACT_REWARD: return transition.contact_reward;
    case MR_TASK_OUTCOME_CHANNEL_0:
        return transition.task_outcome_channel_0;
    case MR_TASK_OUTCOME_CHANNEL_1:
        return transition.task_outcome_channel_1;
    case MR_TASK_OUTCOME_CHANNEL_2:
        return transition.task_outcome_channel_2;
    case MR_TASK_OUTCOME_CHANNEL_3:
        return transition.task_outcome_channel_3;
    case MR_TASK_OUTCOME_CHANNEL_4:
        return transition.task_outcome_channel_4;
    case MR_TASK_OUTCOME_CHANNEL_5:
        return transition.task_outcome_channel_5;
    case MR_TASK_OUTCOME_CHANNEL_6:
        return transition.task_outcome_channel_6;
    case MR_TASK_OUTCOME_CHANNEL_7:
        return transition.task_outcome_channel_7;
    default: return 0.0f;
    }
}

std::vector<metalrobo::LocomotionDynamicSphere>
locomotionDynamicSpheres(const MRTaskRolloutConfigC& config) {
    std::vector<metalrobo::LocomotionDynamicSphere> spheres;
    spheres.reserve(config.dynamic_sphere_count);
    for (std::uint32_t index = 0u;
         index < config.dynamic_sphere_count;
         ++index) {
        const MRTaskRolloutDynamicSphereC& source =
            config.dynamic_spheres[index];
        spheres.push_back({
            .position = {
                source.position[0], source.position[1],
                source.position[2], 1.0f,
            },
            .linearVelocity = {
                source.linear_velocity[0],
                source.linear_velocity[1],
                source.linear_velocity[2], 0.0f,
            },
            .radius = source.radius,
            .mass = source.mass,
            .launchStep = source.launch_step,
        });
    }
    return spheres;
}

void applyRunProfile(
    metalrobo::RunManifest& manifest,
    const MRTaskRolloutConfigC& config
) {
    manifest.profile.id = "native_task_rollout";
    manifest.profile.environmentCount = config.environment_count;
    manifest.profile.controlSteps = 1u;
    manifest.profile.physicsSubsteps = config.physics_substeps;
    manifest.profile.velocityIterations = config.velocity_iterations;
    manifest.profile.finalVelocityIterations =
        config.final_velocity_iterations;
    manifest.profile.controlTimestepSeconds =
        config.control_timestep_seconds;
    manifest.profile.seed = config.seed;
    manifest.profile.streamedArticulatedContactResponses =
        config.materialize_articulated_contact_responses == 0u;
    if (config.override_difficulty_band_range != 0u) {
        manifest.profile.minimumDifficultyBand =
            config.minimum_difficulty_band;
        manifest.profile.maximumDifficultyBand =
            config.maximum_difficulty_band;
    }
}

metalrobo::RunManifest makeUnitreeG1RunManifest(
    const metalrobo::LocomotionSurface surface,
    const metalrobo::UnitreeG1Task taskKind,
    const MRTaskRolloutConfigC& config
) {
    auto robot = metalrobo::builtinRobotPack("unitree_g1");
    if (!robot) {
        throw std::logic_error("bundled G1 RobotPack is unavailable");
    }
    metalrobo::RunManifest manifest;
    manifest.id = "unitree_g1_run";
    manifest.robot = std::move(*robot);
    manifest.scene.id = surface == metalrobo::LocomotionSurface::ground
        ? "ground_scene"
        : "terrain_scene";
    manifest.sensors.id = "unitree_g1_default_sensors";
    manifest.reality.id = "nominal_reality";
    applyRunProfile(manifest, config);

    switch (taskKind) {
    case metalrobo::UnitreeG1Task::velocity:
        manifest.task = metalrobo::makeUnitreeG1LocomotionTaskPack(
            surface,
            manifest.sensors.observation,
            manifest.reality.reset
        );
        break;
    case metalrobo::UnitreeG1Task::disturbanceRecovery:
        manifest.task =
            metalrobo::makeUnitreeG1DisturbanceRecoveryTaskPack(
                surface,
                manifest.sensors.observation,
                manifest.reality.reset
            );
        break;
    case metalrobo::UnitreeG1Task::supineGetUpDiscovery:
        manifest.task =
            metalrobo::makeUnitreeG1SupineGetUpDiscoveryTaskPack(
                surface,
                manifest.sensors.observation,
                manifest.reality.reset
            );
        break;
    case metalrobo::UnitreeG1Task::ballDisturbanceRecovery:
        manifest.task = metalrobo::
            makeUnitreeG1BallDisturbanceRecoveryTaskPack(
                surface,
                manifest.sensors.observation,
                manifest.reality.reset
            );
        break;
    case metalrobo::UnitreeG1Task::ballDodge:
        manifest.task = metalrobo::makeUnitreeG1BallDodgeTaskPack(
            surface,
            manifest.sensors.observation,
            manifest.reality.reset
        );
        break;
    case metalrobo::UnitreeG1Task::developmentalRecovery:
        manifest.task = metalrobo::
            makeUnitreeG1DevelopmentalRecoveryTaskPack(
                surface,
                manifest.sensors.observation,
                manifest.reality.reset
            );
        break;
    case metalrobo::UnitreeG1Task::adultLocomotion:
        manifest.task = metalrobo::
            makeUnitreeG1AdultLocomotionTaskPack(
                surface,
                manifest.sensors.observation,
                manifest.reality.reset
        );
        break;
    case metalrobo::UnitreeG1Task::g1LegsLocomotion:
        manifest.task = metalrobo::makeUnitreeG1LegsLocomotionTaskPack(
            surface,
            manifest.sensors.observation,
            manifest.reality.reset
        );
        break;
    }
    if (taskKind == metalrobo::UnitreeG1Task::g1LegsLocomotion) {
        // The imported actor owns only the twelve leg motors.  Preserve its
        // position-drive/reset contract for the rest of the G1 rather than
        // allowing the generic whole-body action profile to move the arms.
        constexpr std::array<float, 29u> targets{{
            -0.125f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
            -0.125f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.2f, 0.0f, 0.9f, 0.0f, 0.0f, 0.0f,
            0.0f, -0.2f, 0.0f, 0.9f, 0.0f, 0.0f, 0.0f,
        }};
        constexpr std::array<float, 29u> stiffness{{
            110.0f, 110.0f, 110.0f, 100.0f, 40.0f, 40.0f,
            110.0f, 110.0f, 110.0f, 100.0f, 40.0f, 40.0f,
            250.0f, 250.0f, 250.0f,
            90.0f, 60.0f, 20.0f, 60.0f, 4.0f, 4.0f, 4.0f,
            90.0f, 60.0f, 20.0f, 60.0f, 4.0f, 4.0f, 4.0f,
        }};
        constexpr std::array<float, 29u> damping{{
            1.5f, 1.5f, 1.5f, 4.0f, 2.5f, 2.5f,
            1.5f, 1.5f, 1.5f, 4.0f, 2.5f, 2.5f,
            5.0f, 5.0f, 5.0f,
            2.0f, 1.0f, 0.4f, 1.0f, 0.2f, 0.2f, 0.2f,
            2.0f, 1.0f, 0.4f, 1.0f, 0.2f, 0.2f, 0.2f,
        }};
        constexpr std::uint32_t rootQCount = 7u;
        // The imported policy was trained around a 0.83 m pelvis-link height. G1's
        // generic task reset is 0.80 m, so preserve the source standing
        // geometry before its PD controller takes the first policy action.
        if (manifest.robot.mechanics.defaultQ.size() > 2u) {
            manifest.robot.mechanics.defaultQ[2u] += 0.03f;
        }
        if (!manifest.robot.mechanics.materials.empty()) {
            // The source validation scene uses MuJoCo's unit-friction
            // flat floor.  Keep the imported policy on that authored contact
            // contract instead of the bundled RL-Lab 0.6 preset.
            manifest.robot.mechanics.materials.front().friction =
                {1.0f, 1.0f, 0.0f, 0.0f};
        }
        for (std::size_t motor = 0u; motor < targets.size(); ++motor) {
            const std::uint32_t qIndex = rootQCount +
                static_cast<std::uint32_t>(motor);
            if (qIndex < manifest.robot.mechanics.defaultQ.size()) {
                manifest.robot.mechanics.defaultQ[qIndex] = targets[motor];
            }
            for (MRDofPropertiesGPU& dof : manifest.robot.mechanics.dofs) {
                if (dof.jointIndex == motor) {
                    dof.drive.x = stiffness[motor];
                    dof.drive.y = damping[motor];
                    if (motor < 12u) {
                        // The source AGILE leg spec uses a 0.02 kg m^2
                        // reflected actuator armature.  The bundled G1
                        // locomotion preset has a lower hardware-specific
                        // value, which makes this imported gait overreact.
                        dof.drive.z = 0.02f;
                    }
                    break;
                }
            }
            if (motor < 12u && motor < manifest.robot.actuators.size()) {
                manifest.robot.actuators[motor].scale = 0.5f;
            }
        }
    }
    if (taskKind == metalrobo::UnitreeG1Task::supineGetUpDiscovery ||
        taskKind == metalrobo::UnitreeG1Task::developmentalRecovery) {
        if (!manifest.robot.mechanics.materials.empty()) {
            manifest.robot.mechanics.materials.front().response.z =
                1.25e-7f;
        }
        for (metalrobo::RobotActuatorSpec& actuator :
             manifest.robot.actuators) {
            const auto joint = std::ranges::find(
                manifest.robot.mechanics.jointNames,
                actuator.target
            );
            if (joint == manifest.robot.mechanics.jointNames.end()) {
                continue;
            }
            const std::uint32_t jointIndex =
                static_cast<std::uint32_t>(
                    joint - manifest.robot.mechanics.jointNames.begin()
                );
            const auto dof = std::ranges::find_if(
                manifest.robot.mechanics.dofs,
                [jointIndex](const MRDofPropertiesGPU& candidate) {
                    return candidate.jointIndex == jointIndex;
                }
            );
            if (dof == manifest.robot.mechanics.dofs.end() ||
                dof->qIndex >= manifest.robot.mechanics.defaultQ.size()) {
                continue;
            }
            // Keep G1 recovery actuators at unit scale too.  Joint travel
            // remains represented by the authored limits, not an implicit
            // per-joint action multiplier.
            actuator.scale = 1.0f;
            actuator.responseTimeSeconds = 0.0f;
        }
    }
    if (config.disable_task_terminations != 0u) {
        manifest.task.terminations.clear();
    }
    manifest.profile.capacities = manifest.task.capacities;
    const metalrobo::LocomotionSceneComponent surfaceComponent =
        metalrobo::makeLocomotionSurfaceComponent(
            manifest.robot.mechanics,
            surface
        );
    manifest.scene.objects.push_back({
        .id = surface == metalrobo::LocomotionSurface::ground
            ? "locomotion_ground"
            : "locomotion_terrain",
        .semanticClass = "support_surface",
        .role = MR_WORLD_ASSET_FIXTURE,
        .render = MR_WORLD_RENDER_NONE,
        .collision = surface == metalrobo::LocomotionSurface::ground
            ? MR_WORLD_COLLISION_PRIMITIVES
            : MR_WORLD_COLLISION_TRIANGLE_MESH,
        .dynamics = MR_WORLD_DYNAMICS_STATIC,
        .mechanics = surfaceComponent.mechanics,
        .defaultBodyStates = surfaceComponent.defaultBodyStates,
    });
    const auto spheres = locomotionDynamicSpheres(config);
    if (!spheres.empty()) {
        metalrobo::LocomotionSceneComponent sphereComponent =
            metalrobo::makeLocomotionDynamicSphereComponent(
                manifest.robot.mechanics,
                spheres
            );
        manifest.scene.objects.push_back({
            .id = "locomotion_dynamic_spheres",
            .semanticClass = "dynamic_projectile",
            .role = MR_WORLD_ASSET_MANIPULATED,
            .render = MR_WORLD_RENDER_NONE,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_RIGID,
            .mechanics = std::move(sphereComponent.mechanics),
            .defaultBodyStates =
                std::move(sphereComponent.defaultBodyStates),
        });
    }
    return manifest;
}

metalrobo::RunManifest makeFrankaPickPlaceRunManifest(
    const MRTaskRolloutConfigC& config
) {
    auto robot = metalrobo::builtinRobotPack("franka_panda");
    if (!robot) {
        throw std::logic_error("bundled Franka RobotPack is unavailable");
    }
    metalrobo::RunManifest manifest;
    manifest.id = "franka_pick_place_run";
    manifest.robot = std::move(*robot);
    manifest.scene = metalrobo::makeFrankaPickPlaceScenePack();
    manifest.sensors.id = "franka_default_sensors";
    manifest.task = metalrobo::makeFrankaPickPlaceTaskPack(
        manifest.sensors.observation,
        manifest.reality.reset
    );
    if (config.disable_task_terminations != 0u) {
        manifest.task.terminations.clear();
    }
    manifest.reality.id = "franka_nominal_reality";
    manifest.reality.program =
        metalrobo::makeFrankaPickPlaceWorldProgram();
    // This run has no visual sensor executor yet. Retain only variations with
    // a direct native physics/controller/reset effect; camera, appearance and
    // topology alternatives must never survive as passive metadata.
    std::erase_if(
        manifest.reality.program.variations,
        [](const metalrobo::VariationParameter& variation) {
            return variation.target >=
                    MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE ||
                variation.target == MR_WORLD_TARGET_ASSET_SCALE ||
                variation.target ==
                    MR_WORLD_TARGET_ASSET_ORIENTATION_ROLL ||
                variation.target ==
                    MR_WORLD_TARGET_ASSET_ORIENTATION_PITCH ||
                variation.target ==
                    MR_WORLD_TARGET_ASSET_ORIENTATION_YAW;
        }
    );
    for (metalrobo::VariationParameter& variation :
         manifest.reality.program.variations) {
        if (variation.targetId == "franka") {
            variation.targetId = manifest.robot.id;
        }
    }
    applyRunProfile(manifest, config);
    manifest.profile.capacities = manifest.task.capacities;
    return manifest;
}

metalrobo::RunManifest makeImportedRunManifest(
    metalrobo::EngineModel mechanics,
    std::vector<MRBodyStateGPU> defaultSceneBodies,
    const std::uint32_t articulationIndex,
    metalrobo::TaskPack task,
    metalrobo::RobotActuatorPack actuatorPack,
    metalrobo::SensorProgramPack sensorPack,
    metalrobo::RealityProgramPack realityPack,
    const MRTaskRolloutConfigC& config,
    std::string id
) {
    // A RobotActuatorPack is executable controller truth, not descriptive
    // metadata. Position-controller overrides must reach the cooked model's
    // implicit drives before WorldPack compilation; otherwise the task can
    // publish valid targets that exert no physical authority.
    for (const metalrobo::RobotActuatorSpec& actuator :
         actuatorPack.actuators) {
        if (actuator.kind !=
                metalrobo::RobotActuatorKind::jointPosition ||
            (actuator.parameters.x == 0.0f &&
             actuator.parameters.y == 0.0f)) {
            continue;
        }
        const auto first = std::find(
            mechanics.dofNames.begin(), mechanics.dofNames.end(),
            actuator.target
        );
        if (first == mechanics.dofNames.end() ||
            std::find(
                std::next(first), mechanics.dofNames.end(),
                actuator.target
            ) != mechanics.dofNames.end()) {
            throw std::invalid_argument(
                "joint-position actuator controller target is unresolved or ambiguous: " +
                actuator.target
            );
        }
        const std::size_t dofIndex = static_cast<std::size_t>(
            first - mechanics.dofNames.begin()
        );
        if (dofIndex >= mechanics.dofs.size() ||
            !(actuator.parameters.x > 0.0f) ||
            actuator.parameters.y < 0.0f) {
            throw std::invalid_argument(
                "joint-position actuator controller parameters are invalid: " +
                actuator.id
            );
        }
        MRDofPropertiesGPU& dof = mechanics.dofs[dofIndex];
        dof.flags |= MR_DOF_FLAG_DRIVE;
        dof.drive.x = actuator.parameters.x;
        dof.drive.y = actuator.parameters.y;
    }
    metalrobo::RunManifest manifest;
    manifest.id = std::move(id);
    manifest.robot.id = manifest.id + ".robot";
    manifest.robot.mechanics = std::move(mechanics);
    manifest.robot.defaultSceneBodies =
        std::move(defaultSceneBodies);
    manifest.robot.primaryArticulationIndex =
        articulationIndex;
    manifest.robot.actuators = std::move(actuatorPack.actuators);
    manifest.robot.capabilities = {"authored_task_execution"};
    manifest.robot.roles.push_back({
        .id = "whole_body",
        .kind = metalrobo::RobotSemanticKind::body,
        .members = manifest.robot.mechanics.bodyNames,
    });
    if (!manifest.robot.mechanics.jointNames.empty()) {
        manifest.robot.roles.push_back({
            .id = "all_joints",
            .kind = metalrobo::RobotSemanticKind::joint,
            .members = manifest.robot.mechanics.jointNames,
        });
    }
    if (!manifest.robot.mechanics.dofNames.empty()) {
        manifest.robot.roles.push_back({
            .id = "all_dofs",
            .kind = metalrobo::RobotSemanticKind::dof,
            .members = manifest.robot.mechanics.dofNames,
        });
    }
    manifest.scene.id = manifest.id + ".scene";
    manifest.sensors.id = std::move(sensorPack.id);
    manifest.sensors.observation = std::move(sensorPack.observation);
    manifest.task = std::move(task);
    manifest.reality.id = std::move(realityPack.id);
    manifest.reality.program = std::move(realityPack.program);
    manifest.reality.sourceProgramFingerprint =
        realityPack.sourceProgramFingerprint;
    manifest.reality.reset = std::move(realityPack.reset);
    manifest.teacher.id = "no_teacher";
    applyRunProfile(manifest, config);
    manifest.profile.capacities = manifest.task.capacities;
    return manifest;
}

metalrobo::VisualSensorProgram visualSensorProgram(
    const MRTaskVisualObservationConfigC& config
) {
    if (config.capture_policy_camera > 1u) {
        throw std::invalid_argument(
            "visual capture policy-camera flag is invalid"
        );
    }
    if (config.pack_count != 0u && config.packs == nullptr) {
        throw std::invalid_argument(
            "visual SensorPack asset bindings are missing"
        );
    }
    metalrobo::VisualSensorProgram program;
    program.assets.reserve(config.pack_count);
    for (std::uint32_t index = 0u; index < config.pack_count; ++index) {
        const MRTaskVisualPackC& reference = config.packs[index];
        if (reference.path == nullptr || reference.path[0] == '\0' ||
            reference.asset_id == nullptr || reference.asset_id[0] == '\0') {
            throw std::invalid_argument(
                "visual SensorPack contains an incomplete asset binding"
            );
        }
        metalrobo::VisualAssetPackV2 pack;
        std::string reason;
        if (!metalrobo::readVisualAssetPackIndex(
                reference.path,
                pack,
                &reason
            )) {
            throw std::invalid_argument(
                "visual SensorPack asset load failed: " + reason
            );
        }
        program.assets.push_back({
            reference.path,
            reference.asset_id,
            pack.contentHash,
            reference.semantic_id,
            reference.instance_id,
            reference.deformation_source,
        });
    }
    if (config.environment_pack_path != nullptr &&
        config.environment_pack_path[0] != '\0') {
        metalrobo::VisualEnvironmentPackV2 environment;
        std::string reason;
        if (!metalrobo::readVisualEnvironmentPackIndex(
                config.environment_pack_path,
                environment,
                &reason
            )) {
            throw std::invalid_argument(
                "visual SensorPack environment load failed: " + reason
            );
        }
        program.environmentPath = config.environment_pack_path;
        program.environmentContentHash = environment.contentHash;
    }
    program.rendererProfile =
        config.renderer_profile == nullptr
        ? std::string{}
        : config.renderer_profile;
    program.cameraParentBody =
        config.camera_parent_body == nullptr
        ? std::string{}
        : config.camera_parent_body;
    program.cameraPosition = {
        config.camera_position[0],
        config.camera_position[1],
        config.camera_position[2],
        0.0f,
    };
    program.cameraOrientation = {
        config.camera_orientation[0],
        config.camera_orientation[1],
        config.camera_orientation[2],
        config.camera_orientation[3],
    };
    program.width = config.width;
    program.height = config.height;
    program.minimumVisiblePixels = config.minimum_visible_pixels;
    program.verticalFieldOfViewDegrees =
        config.vertical_field_of_view_degrees;
    program.nominalRateHz = config.nominal_rate_hz;
    program.maximumRetainedBytes = config.maximum_retained_bytes;
    program.captureWidth = config.capture_width;
    program.captureHeight = config.capture_height;
    program.capturePolicyCamera = config.capture_policy_camera != 0u;
    program.fingerprint =
        metalrobo::visualSensorProgramFingerprint(program);
    return program;
}

std::unique_ptr<MRTaskRolloutHandle> createCompiledRunTaskRollout(
    metalrobo::RunManifest manifest,
    const char* metallibPath,
    const std::string_view source,
    const MRTaskVisualObservationConfigC* visualSensor
) {
    if (manifest.teacher.id.empty()) {
        manifest.teacher.id = "no_teacher";
    }
    if (visualSensor != nullptr) {
        manifest.sensors.deviceVisual =
            visualSensorProgram(*visualSensor);
    }
    // Task composition (notably an InteractionPack) may refine the measured
    // contact profile after the base manifest is authored. The executable run
    // must compile the final task's exact capacity contract, never a stale
    // snapshot copied before that composition.
    manifest.profile.capacities = manifest.task.capacities;
    metalrobo::CompiledRun compiled;
    const metalrobo::RunCompileDiagnostics status =
        metalrobo::compileRun(manifest, compiled);
    if (!status.succeeded()) {
        throw std::runtime_error(
            std::string{source} + " CompiledRun failed [" +
            metalrobo::runCompileStatusName(status.status) + "]: " +
            status.element + ": " + status.message
        );
    }
    auto handle = createTaskRolloutHandle(
        std::move(compiled),
        metallibPath,
        manifest.task.id,
        source
    );
    return handle;
}

std::vector<float> copyPolicyFloats(
    const float* values,
    const std::size_t count,
    const char* label
) {
    if (count != 0u && values == nullptr) {
        throw std::invalid_argument(
            std::string{label} + " pointer is null"
        );
    }
    return count == 0u
        ? std::vector<float>{}
        : std::vector<float>{values, values + count};
}

metalrobo::PolicyPack policyPackFromC(
    const MRPolicyPackC& policy
) {
    if (policy.id == nullptr ||
        policy.id[0] == '\0' ||
        policy.layers == nullptr ||
        policy.layer_count == 0u ||
        policy.layer_count >
            std::numeric_limits<std::uint32_t>::max() ||
        policy.critic_layer_count >
            std::numeric_limits<std::uint32_t>::max() ||
        (policy.critic_layer_count != 0u &&
         policy.critic_layers == nullptr)) {
        throw std::invalid_argument(
            "policy identity and dense layers are required"
        );
    }

    metalrobo::PolicyPack authored;
    authored.id = policy.id;
    authored.revision = policy.revision;
    authored.contract = {
        .version = policy.contract_version,
        .worldFingerprint = policy.world_fingerprint,
        .taskFingerprint = policy.task_fingerprint,
        .observationFingerprint = policy.observation_fingerprint,
        .actionFingerprint = policy.action_fingerprint,
    };
    if (policy.compatible_task_count >
        metalrobo::kMaximumPolicyTaskBindings) {
        throw std::invalid_argument(
            "policy compatible task count exceeds the artifact boundary"
        );
    }
    if (policy.compatible_task_count != 0u &&
        (policy.compatible_world_fingerprints == nullptr ||
         policy.compatible_task_fingerprints == nullptr)) {
        throw std::invalid_argument(
            "policy compatible task fingerprint pointer is null"
        );
    }
    authored.contract.compatibleTasks.reserve(
        policy.compatible_task_count
    );
    for (std::size_t index = 0u;
         index < policy.compatible_task_count;
         ++index) {
        authored.contract.compatibleTasks.push_back({
            .worldFingerprint =
                policy.compatible_world_fingerprints[index],
            .taskFingerprint =
                policy.compatible_task_fingerprints[index],
        });
    }
    authored.observationMean = copyPolicyFloats(
        policy.observation_mean,
        policy.observation_mean_count,
        "policy observation mean"
    );
    authored.observationInverseStandardDeviation =
        copyPolicyFloats(
            policy.observation_inverse_standard_deviation,
            policy.observation_inverse_standard_deviation_count,
            "policy observation inverse standard deviation"
        );
    authored.criticObservationMean = copyPolicyFloats(
        policy.critic_observation_mean,
        policy.critic_observation_mean_count,
        "policy critic observation mean"
    );
    authored.criticObservationInverseStandardDeviation =
        copyPolicyFloats(
            policy.critic_observation_inverse_standard_deviation,
            policy.critic_observation_inverse_standard_deviation_count,
            "policy critic observation inverse standard deviation"
        );
    authored.actionLogStandardDeviation =
        copyPolicyFloats(
            policy.action_log_standard_deviation,
            policy.action_log_standard_deviation_count,
            "policy action log standard deviation"
        );
    authored.actionBias = copyPolicyFloats(
        policy.action_bias,
        policy.action_bias_count,
        "policy action bias"
    );
    authored.actionScale = copyPolicyFloats(
        policy.action_scale,
        policy.action_scale_count,
        "policy action scale"
    );
    authored.observationClip = policy.observation_clip;
    authored.actionClip = policy.action_clip;
    const auto appendLayers = [](
        const MRPolicyDenseLayerC* sources,
        const std::size_t count,
        std::vector<metalrobo::PolicyDenseLayer>& destination,
        const char* label
    ) {
        destination.reserve(count);
        for (std::size_t index = 0u;
             index < count;
             ++index) {
            const MRPolicyDenseLayerC& source =
                sources[index];
            const std::uint64_t expectedWeights =
                static_cast<std::uint64_t>(
                    source.input_count
                ) * source.output_count;
            if (source.activation >
                    MR_POLICY_ACTIVATION_C_SILU ||
                expectedWeights != source.weight_count ||
                source.bias_count != source.output_count) {
                throw std::invalid_argument(
                    std::string{label} +
                    " dense layer shape or activation is invalid"
                );
            }
            destination.push_back({
                .inputCount = source.input_count,
                .outputCount = source.output_count,
                .activation =
                    static_cast<
                        metalrobo::PolicyActivation
                    >(source.activation),
                .weights = copyPolicyFloats(
                    source.weights,
                    source.weight_count,
                    "policy dense weights"
                ),
                .bias = copyPolicyFloats(
                    source.bias,
                    source.bias_count,
                    "policy dense bias"
                ),
            });
        }
    };
    appendLayers(
        policy.layers,
        policy.layer_count,
        authored.layers,
        "actor"
    );
    appendLayers(
        policy.critic_layers,
        policy.critic_layer_count,
        authored.criticLayers,
        "critic"
    );
    return authored;
}

void installPolicyPack(
    MRTaskRolloutHandle& handle,
    const metalrobo::PolicyPack& authored
) {
    metalrobo::CompiledPolicyProgram compiled;
    const metalrobo::PolicyCompileDiagnostics status =
        metalrobo::compilePolicyProgram(
            authored,
            handle.taskProgram,
            compiled
        );
    if (!status.succeeded()) {
        throw std::invalid_argument(
            std::string{"PolicyPack compile failed ["} +
            metalrobo::policyCompileStatusName(
                status.status
            ) + "]: " + status.element + ": " +
            status.message
        );
    }
    handle.stepConfig.policyProgram = std::move(compiled);
}

std::runtime_error worldFamilyError(
    const char* operation,
    const metalrobo::MetalWorldFamilyDiagnostics& diagnostics
);

metalrobo::WorldAsset makeRolloutVisualAsset(
    const MRTaskRolloutHandle& handle,
    const std::string& id,
    const std::vector<std::uint32_t>& bodies,
    const std::uint32_t role,
    const std::uint32_t render,
    const std::uint32_t dynamics,
    const std::uint32_t articulationIndex = MR_INVALID_INDEX
) {
    metalrobo::WorldAsset asset;
    asset.id = id;
    asset.semanticClass = id;
    asset.role = static_cast<MRWorldAssetRole>(role);
    asset.render = static_cast<MRWorldRenderRepresentation>(render);
    asset.collision = MR_WORLD_COLLISION_PRIMITIVES;
    asset.dynamics =
        static_cast<MRWorldDynamicsRepresentation>(dynamics);
    asset.articulationIndex = articulationIndex;
    asset.bodyIndices = bodies;
    std::unordered_set<std::uint32_t> materials;
    for (std::uint32_t shape = 0u;
         shape < handle.model.shapes.size();
         ++shape) {
        if (std::ranges::find(
                bodies,
                handle.model.shapes[shape].bodyIndex
            ) == bodies.end()) {
            continue;
        }
        asset.shapeIndices.push_back(shape);
        materials.insert(handle.model.shapes[shape].materialIndex);
    }
    asset.materialIndices.assign(materials.begin(), materials.end());
    std::ranges::sort(asset.materialIndices);
    return asset;
}

std::unique_ptr<MRTaskVisualRuntime> compileTaskVisualRuntime(
    MRTaskRolloutHandle& handle,
    const metalrobo::VisualSensorProgram& program,
    const bool deviceObservation
) {
    if (program.assets.empty() || program.cameraParentBody.empty() ||
        program.width == 0u || program.height == 0u ||
        program.minimumVisiblePixels == 0u ||
        !std::isfinite(program.verticalFieldOfViewDegrees) ||
        program.verticalFieldOfViewDegrees < 0.0f ||
        program.verticalFieldOfViewDegrees >= 180.0f ||
        !std::isfinite(program.nominalRateHz) ||
        !(program.nominalRateHz > 0.0f)) {
        throw std::invalid_argument(
            "visual observation packs, articulated camera, and dimensions are required"
        );
    }
    const std::string selectedProfile =
        program.rendererProfile.empty()
        ? "sensor_fast"
        : program.rendererProfile;
    if (selectedProfile != "sensor_fast") {
        throw std::invalid_argument(
            "closed-loop visual observation currently requires sensor_fast"
        );
    }

    const auto parentBody = std::ranges::find(
        handle.model.bodyNames,
        program.cameraParentBody
    );
    if (parentBody == handle.model.bodyNames.end()) {
        throw std::invalid_argument(
            "head-camera parent body is not present in the compiled world"
        );
    }
    const std::uint32_t parentBodyIndex =
        static_cast<std::uint32_t>(
            parentBody - handle.model.bodyNames.begin()
        );
    if (handle.model.bodies[parentBodyIndex].articulationIndex ==
        MR_INVALID_INDEX) {
        throw std::invalid_argument(
            "visual camera must be attached to an articulated link"
        );
    }

    const auto sceneIndices = handle.world.sceneBodyIndices();
    std::map<std::uint32_t, std::uint32_t> trackedOffsets;
    std::map<std::uint32_t, float> positionScales;
    std::map<std::uint32_t, float> velocityScales;
    std::unordered_set<std::uint32_t> trackedSceneBodies;
    std::uint32_t maskedDepthOffset = MR_INVALID_INDEX;
    std::uint32_t maskedDepthCount = 0u;
    if (deviceObservation) {
        const auto actorOperators = handle.taskProgram.actorOperators();
        for (std::uint32_t offset = 0u;
             offset < actorOperators.size();
             ++offset) {
            const MRTaskObservationOperatorGPU& operation =
                actorOperators[offset];
            if (operation.source.x == MR_TASK_OBSERVE_MASKED_DEPTH) {
                if (maskedDepthOffset == MR_INVALID_INDEX) {
                    maskedDepthOffset = offset;
                }
                if (offset != maskedDepthOffset + operation.source.z) {
                    throw std::logic_error(
                        "compiled masked-depth actor slots are not contiguous"
                    );
                }
                ++maskedDepthCount;
                continue;
            }
            if (operation.source.x != MR_TASK_OBSERVE_OBJECT_TRACK) {
                continue;
            }
            if (operation.source.y >= sceneIndices.size()) {
                throw std::logic_error(
                    "compiled object-track scene index is invalid"
                );
            }
            if (operation.source.z == 0u) {
                trackedOffsets.emplace(operation.source.y, offset);
                trackedSceneBodies.insert(operation.source.y);
            } else if (operation.source.z == 1u) {
                positionScales.emplace(
                    operation.source.y,
                    operation.transform.x
                );
            } else if (operation.source.z == 4u) {
                velocityScales.emplace(
                    operation.source.y,
                    operation.transform.x
                );
            }
        }
        for (const MRTaskObservationOperatorGPU& operation :
             handle.taskProgram.criticOperators()) {
            if (operation.source.x == MR_TASK_OBSERVE_OBJECT_TRACK) {
                if (operation.source.y >= sceneIndices.size()) {
                    throw std::logic_error(
                        "compiled critic object-track scene index is invalid"
                    );
                }
                trackedSceneBodies.insert(operation.source.y);
            }
        }
    }
    const MRTaskProgramHeaderGPU& taskHeader =
        handle.taskProgram.header();
    const std::uint64_t expectedMaskedDepthPixels =
        static_cast<std::uint64_t>(taskHeader.visualLayout.x) *
        taskHeader.visualLayout.y * taskHeader.visualLayout.z;
    const std::uint32_t maskedDepthFeatureCount =
        (taskHeader.schedule.w &
         MR_TASK_PROGRAM_MASKED_DEPTH_FEATURES) != 0u
        ? MR_TASK_MASKED_DEPTH_FEATURE_COUNT
        : 0u;
    const std::uint64_t expectedMaskedDepth =
        expectedMaskedDepthPixels + maskedDepthFeatureCount;
    if (deviceObservation &&
        (maskedDepthCount != 0u || expectedMaskedDepth != 0u) &&
        (maskedDepthCount != expectedMaskedDepth ||
         maskedDepthOffset == MR_INVALID_INDEX)) {
        throw std::invalid_argument(
            "SensorPack masked-depth actor layout is incomplete"
        );
    }
    const bool deviceObservationEnabled =
        deviceObservation &&
        (!trackedOffsets.empty() || maskedDepthCount != 0u);
    if (deviceObservation && !deviceObservationEnabled &&
        (program.captureWidth == 0u || program.captureHeight == 0u)) {
        throw std::invalid_argument(
            "visual runtime requires device observations or presentation capture"
        );
    }

    metalrobo::EpisodeTwin episode;
    episode.id = handle.model.name + ".visual_rollout";
    std::vector<std::uint32_t> articulatedBodies;
    for (std::uint32_t body = 0u;
         body < handle.model.bodies.size();
         ++body) {
        if (handle.model.bodies[body].articulationIndex !=
            MR_INVALID_INDEX) {
            articulatedBodies.push_back(body);
        }
    }
    episode.assets.push_back(makeRolloutVisualAsset(
        handle,
        "robot",
        articulatedBodies,
        MR_WORLD_ASSET_ROBOT,
        MR_WORLD_RENDER_MESH_PBR,
        MR_WORLD_DYNAMICS_ARTICULATED,
        handle.model.bodies[parentBodyIndex].articulationIndex
    ));

    std::string manipulatedAsset;
    std::string targetAsset;
    for (std::uint32_t local = 0u;
         local < sceneIndices.size();
         ++local) {
        const std::uint32_t body = sceneIndices[local];
        const std::string& id = handle.model.bodyNames[body];
        const std::uint32_t motion = handle.model.bodies[body].motionType;
        const bool tracked = trackedSceneBodies.contains(local);
        const std::uint32_t role = tracked && manipulatedAsset.empty()
            ? MR_WORLD_ASSET_MANIPULATED
            : motion == MR_MOTION_STATIC
                ? MR_WORLD_ASSET_FIXTURE
                : MR_WORLD_ASSET_CLUTTER;
        const std::uint32_t dynamics = motion == MR_MOTION_STATIC
            ? MR_WORLD_DYNAMICS_STATIC
            : motion == MR_MOTION_KINEMATIC
                ? MR_WORLD_DYNAMICS_KINEMATIC
                : MR_WORLD_DYNAMICS_RIGID;
        metalrobo::WorldAsset asset = makeRolloutVisualAsset(
            handle,
            id,
            {body},
            role,
            (tracked || !deviceObservation)
                ? MR_WORLD_RENDER_MESH_PBR
                : MR_WORLD_RENDER_NONE,
            dynamics
        );
        if (tracked) {
            const auto sphereShape = std::ranges::find_if(
                asset.shapeIndices,
                [&](const std::uint32_t shape) {
                    return handle.model.shapes[shape].shapeType ==
                        MR_SHAPE_SPHERE;
                }
            );
            if (sphereShape != asset.shapeIndices.end()) {
                // The bundled visual sphere is unit radius; imported tracked
                // assets retain their authored presentation scale.
                asset.uniformScale =
                    handle.model.shapes[*sphereShape].dimensions.x;
            }
        }
        if (local < handle.defaultSceneBodies.size()) {
            const MRBodyStateGPU& state =
                handle.defaultSceneBodies[local];
            asset.initialPose.position = {
                state.position.x,
                state.position.y,
                state.position.z,
                0.0f,
            };
            asset.initialPose.orientation = state.orientation;
        }
        if (role == MR_WORLD_ASSET_MANIPULATED) {
            manipulatedAsset = id;
        }
        if (motion == MR_MOTION_STATIC && targetAsset.empty()) {
            targetAsset = id;
            asset.anchors.push_back({
                "visual_rollout_target",
                {},
                0.0f,
                0u,
            });
        }
        episode.assets.push_back(std::move(asset));
    }
    if (deviceObservationEnabled &&
        (manipulatedAsset.empty() || targetAsset.empty())) {
        throw std::invalid_argument(
            "visual object tracking requires a tracked rigid object and static scene asset"
        );
    }

    metalrobo::SensorSpec camera;
    camera.id = "head_rgbd";
    camera.parentAssetId = "robot";
    camera.parentKind = MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
    camera.parentBodyIndex = parentBodyIndex;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose.position = {
        program.cameraPosition.x,
        program.cameraPosition.y,
        program.cameraPosition.z,
        0.0f,
    };
    camera.localPose.orientation = program.cameraOrientation;
    camera.width = program.width;
    camera.height = program.height;
    const float focalLengthPixels =
        program.verticalFieldOfViewDegrees > 0.0f
        ? 0.5f * static_cast<float>(program.height) /
            std::tan(
                0.5f * program.verticalFieldOfViewDegrees *
                std::numbers::pi_v<float> / 180.0f
            )
        : 0.875f * static_cast<float>(program.width);
    if (!std::isfinite(focalLengthPixels) ||
        !(focalLengthPixels > 0.0f)) {
        throw std::invalid_argument(
            "visual observation field of view produces invalid intrinsics"
        );
    }
    camera.intrinsics = {
        focalLengthPixels,
        focalLengthPixels,
        0.5f * static_cast<float>(program.width),
        0.5f * static_cast<float>(program.height),
    };
    camera.nominalRateHz = program.nominalRateHz;
    camera.exposureSeconds = 1.0f / 240.0f;
    camera.minimumDepthMeters = 0.05f;
    camera.maximumDepthMeters = 8.0f;
    episode.sensors.push_back(std::move(camera));
    if (program.capturePolicyCamera &&
        !(program.verticalFieldOfViewDegrees > 0.0f)) {
        throw std::invalid_argument(
            "policy-camera capture requires an authored vertical field of view"
        );
    }
    if (program.captureWidth != 0u && program.captureHeight != 0u) {
        metalrobo::SensorSpec presentation;
        presentation.id = program.capturePolicyCamera
            ? "foundation_policy_camera"
            : "presentation_camera";
        presentation.parentAssetId = "robot";
        presentation.parentKind = program.capturePolicyCamera
            ? MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK
            : MR_WORLD_SENSOR_PARENT_WORLD;
        presentation.parentBodyIndex = program.capturePolicyCamera
            ? parentBodyIndex
            : MR_INVALID_INDEX;
        presentation.kind = MR_WORLD_SENSOR_RGBD;
        presentation.localPose.position = program.capturePolicyCamera
            ? program.cameraPosition
            : mr_float4{1.05f, -1.30f, 1.05f, 0.0f};
        // External media keeps the qualified Studio Small 03 view. Teacher
        // capture instead inherits the deployable policy sensor pose.
        presentation.localPose.orientation = program.capturePolicyCamera
            ? program.cameraOrientation
            : mr_float4{
                  -0.73858354f,
                  -0.26102070f,
                  0.20711737f,
                  0.58605883f,
              };
        presentation.width = program.captureWidth;
        presentation.height = program.captureHeight;
        const float captureFocalLengthPixels =
            program.capturePolicyCamera
            ? 0.5f * static_cast<float>(program.captureHeight) /
                std::tan(
                    0.5f * program.verticalFieldOfViewDegrees *
                    std::numbers::pi_v<float> / 180.0f
                )
            : 0.90f * static_cast<float>(program.captureHeight);
        presentation.intrinsics = {
            captureFocalLengthPixels,
            captureFocalLengthPixels,
            0.5f * static_cast<float>(program.captureWidth),
            0.5f * static_cast<float>(program.captureHeight),
        };
        presentation.nominalRateHz = 50.0f;
        presentation.exposureSeconds = 1.0f / 240.0f;
        presentation.minimumDepthMeters = 0.05f;
        presentation.maximumDepthMeters = 12.0f;
        episode.sensors.push_back(std::move(presentation));
    }
    episode.task = deviceObservationEnabled
        ? metalrobo::TaskSpec{
              "compiled_task_visual_observation",
              "robot",
              manipulatedAsset,
              targetAsset,
              "visual_rollout_target",
              handle.stepConfig.timestepSeconds,
              20.0,
          }
        : metalrobo::TaskSpec{
              deviceObservation
                  ? "compiled_task_presentation_capture"
                  : "inspection_presentation",
              "robot",
              {},
              {},
              {},
              handle.stepConfig.timestepSeconds,
              20.0,
          };

    auto runtime = std::make_unique<MRTaskVisualRuntime>();
    const metalrobo::WorldCompileResult worldStatus =
        metalrobo::compileEpisodeTwin(
            episode,
            handle.model,
            runtime->worldTemplate
        );
    if (!worldStatus.succeeded()) {
        throw std::runtime_error(
            "visual WorldTemplate compile failed: " +
            worldStatus.message
        );
    }
    metalrobo::WorldProgram worldProgram;
    worldProgram.id = episode.id + ".program";
    const metalrobo::WorldCompileResult familyStatus =
        metalrobo::compileWorldFamily(
            runtime->worldTemplate,
            worldProgram,
            runtime->family
        );
    if (!familyStatus.succeeded()) {
        throw std::runtime_error(
            "visual WorldFamily compile failed: " +
            familyStatus.message
        );
    }

    std::vector<metalrobo::VisualAssetReferenceV3> references;
    references.reserve(program.assets.size());
    std::unordered_map<std::string, std::uint32_t> trackedInstances;
    metalrobo::MetalHybridDevicePresentationProgram presentationProgram;
    std::uint32_t deformableAssetCount = 0u;
    for (std::size_t index = 0u;
         index < program.assets.size();
         ++index) {
        const metalrobo::VisualSensorAssetProgram& source =
            program.assets[index];
        const std::uint32_t assetIndex =
            runtime->worldTemplate.assetIndex(source.assetId);
        if (assetIndex == MR_INVALID_INDEX) {
            throw std::invalid_argument(
                std::string{"visual pack references unknown asset: "} +
                source.assetId
            );
        }
        metalrobo::VisualAssetPackV2 pack;
        std::string reason;
        if (!metalrobo::readVisualAssetPackIndex(
                source.path,
                pack,
                &reason
            )) {
            throw std::runtime_error(
                "visual pack load failed: " + reason
            );
        }
        if (pack.contentHash != source.contentHash) {
            throw std::runtime_error(
                "visual pack content changed after run compilation: " +
                source.path
            );
        }
        if (source.deformationSource != static_cast<std::uint32_t>(
                metalrobo::VisualDeformationSource::none
            )) {
            if (source.deformationSource != static_cast<std::uint32_t>(
                    metalrobo::VisualDeformationSource::measuredSurface
                ) ||
                ++deformableAssetCount != 1u ||
                handle.measuredSurfaceRuntime == nullptr ||
                source.assetId != "robot" || pack.materials.empty()) {
                throw std::invalid_argument(
                    "measured-surface visual deformation requires exactly one robot pack and an owning mechanics runtime"
                );
            }
            const auto* measuredBinding =
                handle.run.measuredSurfaceBinding();
            if (measuredBinding == nullptr ||
                !std::ranges::any_of(
                    pack.symbolicBindings,
                    [&](const metalrobo::VisualSymbolicBindingV2& binding) {
                        return binding.bodyIndex == measuredBinding->bodyIndex &&
                            binding.binding ==
                                MR_VISUAL_BINDING_ARTICULATED_LINK;
                    }
                )) {
                throw std::invalid_argument(
                    "measured-surface visual pack is not bound to its owning articulated link"
                );
            }
            const MRVisualMaterialGPUV2& material = pack.materials.front();
            presentationProgram =
                handle.measuredSurfaceRuntime->presentationProgram({
                    .semanticId = source.semanticId,
                    .instanceId = source.instanceId,
                    .baseColorAndOpacity = material.baseColorAndOpacity,
                    .perceptualRoughness = material.surface.x,
                    .metallic = material.surface.y,
                });
            if (!presentationProgram.valid()) {
                throw std::runtime_error(
                    "measured-surface visual presentation program is invalid"
                );
            }
            // The indexed pack remains the authored material, identity,
            // binding, and provenance contract. Its frozen phase-zero
            // triangles must not be compiled beside the live surface.
            continue;
        }
        const bool robotPresentationPack =
            program.captureWidth != 0u &&
            source.instanceId == 1u &&
            source.assetId == "robot";
        const std::uint32_t instanceId =
            robotPresentationPack && index != 0u
            ? 1'000u + static_cast<std::uint32_t>(index)
            : source.instanceId;
        references.push_back({
            source.path,
            pack.contentHash,
            assetIndex,
            source.semanticId,
            instanceId,
        });
        const auto sceneName = std::ranges::find(
            handle.model.bodyNames,
            source.assetId
        );
        if (sceneName != handle.model.bodyNames.end()) {
            const std::uint32_t body =
                static_cast<std::uint32_t>(
                    sceneName - handle.model.bodyNames.begin()
                );
            const auto local = std::ranges::find(sceneIndices, body);
            if (local != sceneIndices.end() &&
                trackedSceneBodies.contains(
                    static_cast<std::uint32_t>(
                        local - sceneIndices.begin()
                    )
                )) {
                const bool rigidBound = std::ranges::any_of(
                    pack.symbolicBindings,
                    [&](const metalrobo::VisualSymbolicBindingV2& binding) {
                        return binding.bodyIndex == body &&
                            binding.binding ==
                                MR_VISUAL_BINDING_RIGID_BODY;
                    }
                );
                if (!rigidBound) {
                    throw std::invalid_argument(
                        std::string{
                            "tracked visual pack is not bound to physics body: "
                        } + source.assetId
                    );
                }
                trackedInstances[source.assetId] = source.instanceId;
            }
        }
    }

    metalrobo::VisualEnvironmentReferenceV2 environment =
        metalrobo::makeNeutralStudioEnvironmentV2();
    if (!program.environmentPath.empty()) {
        metalrobo::VisualEnvironmentPackV2 pack;
        std::string reason;
        if (!metalrobo::readVisualEnvironmentPackIndex(
                program.environmentPath,
                pack,
                &reason
            )) {
            throw std::runtime_error(
                "visual environment pack load failed: " + reason
            );
        }
        if (pack.contentHash != program.environmentContentHash) {
            throw std::runtime_error(
                "visual environment content changed after run compilation"
            );
        }
        environment.id = pack.id;
        environment.packPath = program.environmentPath;
        environment.contentHash = pack.contentHash;
        environment.intensity = 0.12f;
    }
    metalrobo::VisualSceneManifestV3 manifest;
    std::string reason;
    if (!metalrobo::compileVisualSceneManifestV3(
            runtime->worldTemplate,
            references,
            environment,
            metalrobo::makeIndoorAreaLightRigV1(),
            manifest,
            &reason
        )) {
        throw std::runtime_error(
            "visual scene compile failed: " + reason
        );
    }
    auto captureScene = manifest.renderScene;

    metalrobo::MetalWorldFamilyConfig familyConfig;
    familyConfig.metallibPath = handle.metallibPath;
    runtime->worlds = metalrobo::MetalWorldFamilyContext{
        std::move(familyConfig)
    };
    const auto metalFamily = runtime->worlds.compile(
        runtime->family,
        handle.environmentCount
    );
    if (!metalFamily.succeeded()) {
        throw worldFamilyError("visual WorldFamily Metal compile", metalFamily);
    }
    const auto sampled = runtime->worlds.sample(
        handle.environmentCount,
        handle.stepConfig.taskSeed
    );
    if (!sampled.succeeded()) {
        throw worldFamilyError("visual WorldFamily sample", sampled);
    }

    if (!deviceObservation) {
        metalrobo::MetalRunInspectorConfig inspectionConfig;
        inspectionConfig.metallibPath = handle.metallibPath;
        inspectionConfig.width = program.width;
        inspectionConfig.height = program.height;
        inspectionConfig.environmentIndex = 0u;
        inspectionConfig.maximumFramesInFlight = 3u;
        if (program.maximumRetainedBytes != 0u) {
            if (program.maximumRetainedBytes >
                std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument(
                    "inspection retained-memory budget exceeds size_t"
                );
            }
            inspectionConfig.maximumRetainedBytes =
                static_cast<std::size_t>(program.maximumRetainedBytes);
        }
        runtime->inspector = std::make_unique<
            metalrobo::MetalRunInspector
        >(std::move(inspectionConfig));
        const auto inspectionCompiled = runtime->inspector->compile(
            std::move(manifest.renderScene),
            metalrobo::VisualRendererProfileV1::sensorFast(),
            runtime->worlds
        );
        if (!inspectionCompiled.succeeded()) {
            throw std::runtime_error(
                std::string{"inspection renderer compile failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    inspectionCompiled.status
                ) + "]: " + inspectionCompiled.message
            );
        }
        if (presentationProgram.valid()) {
            runtime->inspector->setDevicePresentationProgram(
                presentationProgram
            );
        }
        runtime->sceneFingerprint = manifest.fingerprint ^
            (presentationProgram.valid() ? program.fingerprint : 0u);
        return runtime;
    }

    metalrobo::MetalHybridRendererConfig rendererConfig;
    rendererConfig.metallibPath = handle.metallibPath;
    rendererConfig.width = program.width;
    rendererConfig.height = program.height;
    // Closed-loop masked depth consumes exact geometric winners directly.
    // Media capture has its own retained renderer below.
    rendererConfig.retainObservationBuffers = false;
    rendererConfig.geometricObservationsOnly = true;
    if (program.maximumRetainedBytes != 0u) {
        if (program.maximumRetainedBytes >
            std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument(
                "visual retained-memory budget exceeds size_t"
            );
        }
        rendererConfig.maximumRetainedBytes =
            static_cast<std::size_t>(
                program.maximumRetainedBytes
            );
    }
    runtime->renderer = metalrobo::MetalHybridRenderer{
        std::move(rendererConfig)
    };
    const auto rendered = runtime->renderer.compile(
        std::move(manifest.renderScene),
        metalrobo::VisualRendererProfileV1::sensorFast(),
        handle.environmentCount
    );
    if (!rendered.succeeded()) {
        throw std::runtime_error(
            std::string{"visual renderer compile failed ["} +
            metalrobo::metalHybridRendererStatusName(rendered.status) +
            "]: " + rendered.message
        );
    }
    if (presentationProgram.valid()) {
        runtime->renderer.setDevicePresentationProgram(
            presentationProgram
        );
    }
    if (program.captureWidth != 0u) {
        metalrobo::MetalHybridRendererConfig captureConfig;
        captureConfig.metallibPath = handle.metallibPath;
        captureConfig.width = program.captureWidth;
        captureConfig.height = program.captureHeight;
        captureConfig.retainObservationBuffers = true;
        runtime->captureRenderer = metalrobo::MetalHybridRenderer{
            std::move(captureConfig)
        };
        const auto captureCompiled = runtime->captureRenderer.compile(
            std::move(captureScene),
            presentationProgram.valid()
                ? metalrobo::VisualRendererProfileV1::sensorFast()
                : metalrobo::VisualRendererProfileV1::sensorReference(),
            handle.environmentCount
        );
        if (!captureCompiled.succeeded()) {
            throw std::runtime_error(
                std::string{"visual capture renderer compile failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    captureCompiled.status
                ) + "]: " + captureCompiled.message
            );
        }
        if (presentationProgram.valid()) {
            runtime->captureRenderer.setDevicePresentationProgram(
                presentationProgram
            );
        }
        runtime->captureEnabled = true;
        runtime->capturePolicyCamera = program.capturePolicyCamera;
        runtime->captureUsesLiveDeformation = presentationProgram.valid();
    }

    metalrobo::MetalHybridObjectTrackerConfig trackerConfig;
    trackerConfig.capacity = handle.environmentCount;
    trackerConfig.cameraIndex = 0u;
    trackerConfig.rootBodyIndex =
        handle.model.articulations[handle.world.articulationIndex()].rootBody;
    trackerConfig.maximumActorHistoryLength =
        handle.taskProgram.layout().actorHistoryLength;
    trackerConfig.timestepSeconds = handle.stepConfig.timestepSeconds;
    trackerConfig.nominalSensorRateHz = program.nominalRateHz;
    trackerConfig.maximumTrackSpeedMetersPerSecond = 10.0f;
    for (const auto& [sceneIndex, offset] : trackedOffsets) {
        const std::string& assetId =
            handle.model.bodyNames[sceneIndices[sceneIndex]];
        const auto instance = trackedInstances.find(assetId);
        if (instance == trackedInstances.end() ||
            !positionScales.contains(sceneIndex) ||
            !velocityScales.contains(sceneIndex)) {
            throw std::invalid_argument(
                "every SensorPack object track requires one authored visual pack binding"
            );
        }
        trackerConfig.bindings.push_back({
            .instanceId = instance->second,
            .actorFrameOffset = offset,
            .positionScale = positionScales[sceneIndex],
            .velocityScale = velocityScales[sceneIndex],
            .minimumVisiblePixels = program.minimumVisiblePixels,
        });
    }
    if (maskedDepthCount != 0u) {
        trackerConfig.maskedDepthWidth = taskHeader.visualLayout.x;
        trackerConfig.maskedDepthHeight = taskHeader.visualLayout.y;
        trackerConfig.maskedDepthActorFrameOffset =
            handle.taskProgram.layout().actorFrameSize *
            handle.taskProgram.layout().actorHistoryLength +
            taskHeader.counts3.w;
        trackerConfig.maskedDepthNearMeters = taskHeader.visualRange.x;
        trackerConfig.maskedDepthFarMeters = taskHeader.visualRange.y;
        trackerConfig.maskedDepthEdgeFlickerProbability =
            taskHeader.visualRange.z;
        trackerConfig.maskedDepthCurriculumCorruptionGain =
            taskHeader.visualRange.w;
        trackerConfig.maskedDepthFullDropoutProbability =
            taskHeader.visualCorruption.x;
        trackerConfig.maskedDepthPixelDropoutProbability =
            taskHeader.visualCorruption.y;
        trackerConfig.maskedDepthJitterMeters =
            taskHeader.visualCorruption.z;
        trackerConfig.maskedDepthNoiseSigmaMeters =
            taskHeader.visualCorruption.w;
        const std::array offsets{
            taskHeader.visualHistory.x,
            taskHeader.visualHistory.y,
            taskHeader.visualHistory.z,
            taskHeader.visualHistory.w,
        };
        trackerConfig.maskedDepthFrameOffsets.assign(
            offsets.begin(),
            offsets.begin() + taskHeader.visualLayout.z
        );
        trackerConfig.maskedDepthCurriculumLevelCount =
            taskHeader.schedule.z;
        trackerConfig.maskedDepthFeatureCount =
            maskedDepthFeatureCount;
        for (const std::uint32_t sceneIndex : trackedSceneBodies) {
            const std::string& assetId =
                handle.model.bodyNames[sceneIndices[sceneIndex]];
            const auto instance = trackedInstances.find(assetId);
            if (instance == trackedInstances.end()) {
                throw std::invalid_argument(
                    "every masked-depth object requires an authored visual binding"
                );
            }
            trackerConfig.maskedDepthInstanceIds.push_back(
                instance->second
            );
        }
    }
    if (deviceObservationEnabled) {
        const auto trackerStatus = runtime->tracker.compile(
            runtime->renderer,
            runtime->worlds,
            std::move(trackerConfig)
        );
        if (!trackerStatus.succeeded()) {
            throw std::runtime_error(
                std::string{"visual tracker compile failed ["} +
                metalrobo::metalHybridRendererStatusName(trackerStatus.status) +
                "]: " + trackerStatus.message
            );
        }
    }
    runtime->deviceObservationEnabled = deviceObservationEnabled;
    runtime->sceneFingerprint = manifest.fingerprint ^
        (presentationProgram.valid() ? program.fingerprint : 0u);
    return runtime;
}

void installTaskVisualRuntime(
    MRTaskRolloutHandle& handle,
    const metalrobo::VisualSensorProgram& program
) {
    if (handle.residentState.valid()) {
        throw std::logic_error(
            "SensorPack must compile before resident initialization"
        );
    }
    std::unique_ptr<MRTaskVisualRuntime> candidate =
        compileTaskVisualRuntime(handle, program, true);
    if (candidate->deviceObservationEnabled) {
        const metalrobo::MetalWorldDeviceObservationProgram observationProgram =
            candidate->tracker.observationProgram();
        if (!observationProgram.valid()) {
            throw std::runtime_error(
                "compiled SensorPack visual program is invalid"
            );
        }
        handle.stepConfig.deviceObservationProgram = observationProgram;
    }
    handle.visualRuntime = std::move(candidate);
}

void installTaskInspectionRuntime(
    MRTaskRolloutHandle& handle,
    const metalrobo::VisualSensorProgram& program
) {
    if (handle.residentState.valid()) {
        throw std::logic_error(
            "inspection scene must compile before resident initialization"
        );
    }
    std::unique_ptr<MRTaskVisualRuntime> candidate =
        compileTaskVisualRuntime(handle, program, false);
    if (candidate->inspector == nullptr) {
        throw std::logic_error("inspection renderer did not compile");
    }
    const metalrobo::MetalWorldInspectionProgram inspectionProgram =
        candidate->inspector->inspectionProgram();
    if (!inspectionProgram.valid()) {
        throw std::logic_error("inspection program is invalid");
    }
    handle.inspectionVisualRuntime = std::move(candidate);
    handle.stepConfig.inspectionProgram = inspectionProgram;
}

std::runtime_error worldFamilyError(
    const char* operation,
    const metalrobo::MetalWorldFamilyDiagnostics& diagnostics
) {
    return std::runtime_error(
        std::string{operation} + " failed [" +
        metalrobo::metalWorldFamilyStatusName(diagnostics.status) +
        "]: " + diagnostics.message
    );
}

} // namespace

extern "C" {

const char* mr_version(void) {
    return "0.4.0";
}

uint64_t mr_runtime_abi_fingerprint(void) {
    return metalrobo::runtimeAbiFingerprint();
}

const char* mr_last_error(void) {
    return gLastError.c_str();
}

const char* mr_unitree_g1_deployment_contract_json(void) {
    try {
        static const std::string contract =
            unitreeG1DeploymentContractJSON();
        gLastError.clear();
        return contract.c_str();
    } catch (const std::exception& error) {
        gLastError = error.what();
    } catch (...) {
        gLastError =
            "failed to construct the Unitree G1 deployment contract";
    }
    return nullptr;
}

int mr_write_policy_pack(
    const MRPolicyPackC* policy,
    const char* policy_pack_path
) {
    if (policy == nullptr ||
        policy_pack_path == nullptr ||
        policy_pack_path[0] == '\0') {
        gLastError = "PolicyPack and output path are required.";
        return -1;
    }
    return translateErrors([&] {
        const metalrobo::PolicyPack authored =
            policyPackFromC(*policy);
        const metalrobo::LearningPackResult written =
            metalrobo::writePolicyPack(
                authored,
                policy_pack_path
            );
        if (!written.succeeded()) {
            throw std::invalid_argument(
                std::string{"PolicyPack write failed ["} +
                metalrobo::learningPackStatusName(
                    written.status
                ) + "]: " + written.message
            );
        }
    });
}

uint64_t mr_learning_pack_content_hash(
    const void* payload,
    const size_t byte_count
) {
    if (payload == nullptr && byte_count != 0u) {
        return 0u;
    }
    return metalrobo::learningPackContentHash(
        {
            static_cast<const std::byte*>(payload),
            byte_count,
        }
    );
}

int mr_compile_episode_manifest(
    const char* manifest_path,
    const char* output_pack_path,
    const char* artifact_store_path
) {
    if (manifest_path == nullptr || manifest_path[0] == '\0' ||
        output_pack_path == nullptr || output_pack_path[0] == '\0') {
        gLastError =
            "manifest_path and output_pack_path must be nonempty.";
        return -1;
    }
    return translateErrors([&] {
        metalrobo::CaptureManifest manifest;
        const metalrobo::EpisodeTwinCompilerResult loaded =
            metalrobo::loadCaptureManifestJSON(
                manifest_path,
                manifest
            );
        if (!loaded.succeeded()) {
            throw std::runtime_error(
                std::string{"capture manifest load failed ["} +
                metalrobo::episodeTwinCompilerStatusName(loaded.status) +
                "]: " + loaded.message
            );
        }
        if (manifest.engineModelId != "franka_pick_place" ||
            manifest.worldProgramId != "franka_pick_place") {
            throw std::runtime_error(
                "capture manifest references an unregistered engine "
                "model or world program"
            );
        }

        const std::filesystem::path outputPath{output_pack_path};
        metalrobo::EpisodeTwinCompilerConfig config;
        config.artifactStore =
            artifact_store_path != nullptr &&
                artifact_store_path[0] != '\0'
            ? std::filesystem::path{artifact_store_path}
            : outputPath.parent_path() /
                (outputPath.stem().string() + ".artifacts");
        metalrobo::EpisodeTwinCompiler compiler{std::move(config)};
        metalrobo::CompiledEpisodeTwin compiled;
        const metalrobo::EpisodeTwinCompilerResult result =
            compiler.compile(
                manifest,
                metalrobo::makeFrankaPickPlaceEngineModel(),
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                compiled
            );
        if (!result.succeeded()) {
            throw std::runtime_error(
                std::string{"episode twin compilation failed ["} +
                metalrobo::episodeTwinCompilerStatusName(result.status) +
                "]: " + result.message
            );
        }
        const metalrobo::WorldPackResult written =
            metalrobo::writeWorldPack(
                compiled.worldPack,
                outputPath
            );
        if (!written.succeeded()) {
            throw std::runtime_error(
                std::string{"world-pack write failed ["} +
                metalrobo::worldPackStatusName(written.status) +
                "]: " + written.message
            );
        }
    });
}

static MRTaskRolloutHandle* createUnitreeG1Run(
    const MRTaskRolloutConfigC* config,
    const uint32_t surface_value,
    const uint32_t task_value,
    const MRTaskVisualObservationConfigC* visual_sensor,
    const char* metallib_path
) {
    if (config == nullptr) {
        gLastError = "task-rollout config is null.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        validateTaskRolloutConfiguration(*config);
        const metalrobo::LocomotionSurface surface =
            locomotionSurface(surface_value);
        auto handle = createCompiledRunTaskRollout(
            makeUnitreeG1RunManifest(
                surface,
                unitreeG1Task(task_value),
                *config
            ),
            metallib_path,
            "bundled G1",
            visual_sensor
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

static MRTaskRolloutHandle* createFrankaRun(
    const MRTaskRolloutConfigC* config,
    const MRTaskVisualObservationConfigC* visual_sensor,
    const char* metallib_path
) {
    if (config == nullptr) {
        gLastError = "task-rollout config is null.";
        return nullptr;
    }
    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        validateTaskRolloutConfiguration(*config);
        auto handle = createCompiledRunTaskRollout(
            makeFrankaPickPlaceRunManifest(*config),
            metallib_path,
            "bundled Franka pick/place",
            visual_sensor
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

static MRTaskRolloutHandle* createPX4X500Run(
    const MRTaskRolloutConfigC* config,
    const char* metallib_path
) {
    if (config == nullptr) {
        gLastError = "task-rollout config is null.";
        return nullptr;
    }
    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        validateTaskRolloutConfiguration(*config);
        auto robot = metalrobo::builtinRobotPack("px4_x500");
        if (!robot) {
            throw std::logic_error(
                "bundled PX4 X500 RobotPack is unavailable"
            );
        }
        metalrobo::RunManifest manifest;
        manifest.id = "px4_x500_hover_run";
        manifest.robot = std::move(*robot);
        manifest.scene = metalrobo::makePX4X500HoverScenePack();
        manifest.sensors.id = "px4_x500_state_sensors";
        manifest.task = metalrobo::makePX4X500HoverTaskPack(
            manifest.sensors.observation,
            manifest.reality.reset
        );
        manifest.reality.id = "px4_x500_nominal_reality";
        manifest.teacher.id = "no_teacher";
        applyRunProfile(manifest, *config);
        manifest.profile.capacities = manifest.task.capacities;
        auto handle = createCompiledRunTaskRollout(
            std::move(manifest),
            metallib_path,
            "PX4 X500",
            nullptr
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

static MRTaskRolloutHandle* createMeasuredDoveRun(
    const MRTaskRolloutConfigC* config,
    const char* measured_surface_manifest_path,
    const uint32_t task_value,
    const char* metallib_path
) {
    if (config == nullptr || measured_surface_manifest_path == nullptr ||
        measured_surface_manifest_path[0] == '\0') {
        gLastError = "measured-dove manifest and rollout config are required.";
        return nullptr;
    }
    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        validateTaskRolloutConfiguration(*config);
        if (task_value > MR_MEASURED_SURFACE_TASK_FOOD_NAVIGATION) {
            throw std::invalid_argument("measured-dove task is invalid");
        }
        metalrobo::MeasuredSurfaceRobotPack surface =
            metalrobo::loadDeetjenMeasuredDoveRobotPack(
                measured_surface_manifest_path);
        const bool dropRecovery = task_value ==
            MR_MEASURED_SURFACE_TASK_FATAL_DROP_RECOVERY;
        const bool cruise = task_value == MR_MEASURED_SURFACE_TASK_CRUISE;
        const bool foodNavigation = task_value ==
            MR_MEASURED_SURFACE_TASK_FOOD_NAVIGATION;
        surface.normalizedActionBias =
            metalrobo::measuredSurfaceRecoveryTrimActions();
        metalrobo::RunManifest manifest;
        manifest.id = foodNavigation ? "deetjen_f03_food_navigation_run"
            : (cruise ? "deetjen_f03_forward_agility_run"
            : (dropRecovery ? "deetjen_f03_fatal_drop_recovery_run"
                            : "deetjen_f03_flight_trim_run"));
        manifest.robot = metalrobo::makeMeasuredSurfaceRobotPack(
            std::move(surface), "deetjen_f03_robot");
        manifest.scene = foodNavigation
            ? metalrobo::makeMeasuredSurfaceFoodNavigationScenePack(
                manifest.robot)
            : ((dropRecovery || cruise)
            ? metalrobo::makeMeasuredSurfaceDropRecoveryScenePack(
                manifest.robot)
            : metalrobo::ScenePack{.id = "unbounded_air"});
        manifest.sensors.id = "measured_surface_flight_state";
        manifest.reality.id = "measured_surface_nominal_reality";
        manifest.teacher.id = "no_teacher";
        manifest.task = foodNavigation
            ? metalrobo::makeMeasuredSurfaceFoodNavigationTaskPack(
                manifest.robot, manifest.sensors.observation,
                manifest.reality.reset)
            : (cruise
            ? metalrobo::makeMeasuredSurfaceCruiseTaskPack(
                manifest.robot, manifest.sensors.observation,
                manifest.reality.reset)
            : (dropRecovery ? metalrobo::makeMeasuredSurfaceDropRecoveryTaskPack(
                manifest.robot, manifest.sensors.observation,
                manifest.reality.reset)
            : metalrobo::makeMeasuredSurfaceFlightTaskPack(
                manifest.robot, manifest.sensors.observation,
                manifest.reality.reset)));
        applyRunProfile(manifest, *config);
        manifest.profile.capacities = manifest.task.capacities;
        auto handle = createCompiledRunTaskRollout(
            std::move(manifest), metallib_path, "measured Deetjen dove",
            nullptr);
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

enum class NumiflyRunMode {
    hover,
    forwardFlight,
    noLegs,
};

static MRTaskRolloutHandle* createNumiflyRun(
    const MRTaskRolloutConfigC* config,
    const char* measured_surface_manifest_path,
    const uint32_t surface_value,
    const MRTaskVisualObservationConfigC* visual_sensor,
    const char* metallib_path,
    const NumiflyRunMode mode
) {
    if (config == nullptr || measured_surface_manifest_path == nullptr ||
        measured_surface_manifest_path[0] == '\0') {
        gLastError = "Numifly wing manifest and rollout config are required.";
        return nullptr;
    }
    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        validateTaskRolloutConfiguration(*config);
        const metalrobo::LocomotionSurface surface =
            locomotionSurface(surface_value);
        metalrobo::MeasuredSurfaceRobotPack wings =
            metalrobo::loadNumiflyMaedaWingPack(
                measured_surface_manifest_path);
        const bool noLegs = mode == NumiflyRunMode::noLegs;
        const bool forwardFlight = mode == NumiflyRunMode::forwardFlight;
        metalrobo::RunManifest manifest;
        manifest.id = noLegs ? "numifly_no_legs_flight_run"
            : (forwardFlight ? "numifly_forward_flight_run"
                             : "numifly_flight_run");
        manifest.robot = noLegs
            ? metalrobo::makeNumiflyNoLegsRobotPack(std::move(wings))
            : metalrobo::makeNumiflyRobotPack(std::move(wings));
        manifest.scene.id = surface == metalrobo::LocomotionSurface::ground
            ? (noLegs ? "numifly_no_legs_ground_scene"
                      : (forwardFlight ? "numifly_forward_ground_scene"
                                       : "numifly_ground_scene"))
            : (noLegs ? "numifly_no_legs_terrain_scene"
                      : (forwardFlight ? "numifly_forward_terrain_scene"
                                       : "numifly_terrain_scene"));
        manifest.sensors.id = noLegs ? "numifly_no_legs_flight_state"
            : (forwardFlight ? "numifly_forward_flight_state"
                             : "numifly_flight_state");
        manifest.reality.id = noLegs ? "numifly_no_legs_nominal_reality"
            : (forwardFlight ? "numifly_forward_nominal_reality"
                             : "numifly_nominal_reality");
        manifest.teacher.id = "no_teacher";
        manifest.task = noLegs
            ? metalrobo::makeNumiflyNoLegsFlightTaskPack(
                manifest.robot, manifest.sensors.observation,
                manifest.reality.reset)
            : (forwardFlight
                ? metalrobo::makeNumiflyForwardFlightTaskPack(
                    manifest.robot, surface, manifest.sensors.observation,
                    manifest.reality.reset)
                : metalrobo::makeNumiflyFlightTaskPack(
                    manifest.robot, surface, manifest.sensors.observation,
                    manifest.reality.reset));
        applyRunProfile(manifest, *config);
        manifest.profile.capacities = manifest.task.capacities;
        if (config->disable_task_terminations != 0u) {
            manifest.task.terminations.clear();
        }
        const metalrobo::LocomotionSceneComponent surfaceComponent =
            metalrobo::makeLocomotionSurfaceComponent(
                manifest.robot.mechanics, surface);
        manifest.scene.objects.push_back({
            .id = surface == metalrobo::LocomotionSurface::ground
                ? (noLegs ? "numifly_no_legs_ground"
                          : (forwardFlight ? "numifly_forward_ground"
                                           : "numifly_ground"))
                : (noLegs ? "numifly_no_legs_terrain"
                          : (forwardFlight ? "numifly_forward_terrain"
                                           : "numifly_terrain")),
            .semanticClass = "support_surface",
            .role = MR_WORLD_ASSET_FIXTURE,
            .render = MR_WORLD_RENDER_NONE,
            .collision = surface == metalrobo::LocomotionSurface::ground
                ? MR_WORLD_COLLISION_PRIMITIVES
                : MR_WORLD_COLLISION_TRIANGLE_MESH,
            .dynamics = MR_WORLD_DYNAMICS_STATIC,
            .mechanics = surfaceComponent.mechanics,
            .defaultBodyStates = surfaceComponent.defaultBodyStates,
        });
        auto handle = createCompiledRunTaskRollout(
            std::move(manifest), metallib_path,
            noLegs ? "Numifly No Legs"
                   : (forwardFlight ? "Numifly Forward Flight" : "Numifly"),
            visual_sensor);
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

static MRTaskRolloutHandle* createUnitreeG1TeacherRun(
    const MRTaskRolloutConfigC* config,
    const uint32_t surface_value,
    const uint32_t task_value,
    const char* interaction_pack_path,
    const char* interaction_clip_id,
    const MRTaskVisualObservationConfigC* visual_sensor,
    const char* metallib_path
) {
    if (config == nullptr || interaction_pack_path == nullptr ||
        interaction_pack_path[0] == '\0' ||
        interaction_clip_id == nullptr ||
        interaction_clip_id[0] == '\0') {
        gLastError =
            "InteractionPack path, clip identity, and task-rollout config are required.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        validateTaskRolloutConfiguration(*config);
        metalrobo::InteractionPack interactions;
        const metalrobo::LearningPackResult loaded =
            metalrobo::readInteractionPack(
                interaction_pack_path,
                interactions
            );
        if (!loaded.succeeded()) {
            throw std::invalid_argument(
                std::string{"InteractionPack load failed ["} +
                metalrobo::learningPackStatusName(loaded.status) +
                "]: " + loaded.message
            );
        }
        const metalrobo::InteractionClip& clip =
            selectedInteractionClip(
                interactions,
                interaction_clip_id
            );
        const metalrobo::UnitreeG1Task task =
            unitreeG1Task(task_value);
        if (task != metalrobo::UnitreeG1Task::velocity &&
            task != metalrobo::UnitreeG1Task::ballDodge &&
            task != metalrobo::UnitreeG1Task::supineGetUpDiscovery) {
            throw std::invalid_argument(
                "InteractionPack task composition supports velocity, ball-dodge, or supine-get-up."
            );
        }
        metalrobo::RunManifest manifest = makeUnitreeG1RunManifest(
            locomotionSurface(surface_value),
            task,
            *config
        );
        if (task == metalrobo::UnitreeG1Task::velocity) {
            authorG1InteractionTrackingTask(
                manifest.task,
                manifest.sensors.observation,
                manifest.reality.reset,
                interactions,
                clip,
                *config
            );
        } else {
            authorG1ImaginedTask(
                manifest.task,
                manifest.sensors.observation,
                interactions,
                clip,
                *config,
                task != metalrobo::UnitreeG1Task::ballDodge
            );
        }
        if (config->interaction_reference_mode ==
            MR_INTERACTION_REFERENCE_GUIDE) {
            manifest.task.interactionControlReference = true;
        } else if (config->interaction_reference_mode ==
                   MR_INTERACTION_REFERENCE_RESET_ONLY) {
            manifest.task.interactionControlReference = false;
        }
        if (config->override_interaction_student_authority != 0u) {
            manifest.task.interactionStudentAuthority =
                config->interaction_student_authority;
        }
        if (config->override_interaction_reset_phase_fraction != 0u) {
            manifest.task.interactionResetPhaseFraction =
                config->interaction_reset_phase_fraction;
        }
        if (config->override_interaction_reset_phase_probability != 0u) {
            manifest.task.interactionResetPhaseProbability =
                config->interaction_reset_phase_probability;
        }
        if (config->override_interaction_reset_maximum_phase != 0u) {
            manifest.task.interactionResetMaximumPhase =
                config->interaction_reset_maximum_phase;
        }
        manifest.teacher = {
            .id = interactions.id,
            .kind = metalrobo::TeacherKind::motionImagination,
            .provider = "interaction_pack",
            .model = clip.id,
            .artifact = interaction_pack_path,
            .artifactFingerprint = loaded.contentHash,
            .interactions = interactions,
            .interactionClip = clip.id,
        };
        auto handle = createCompiledRunTaskRollout(
            std::move(manifest),
            metallib_path,
            "bundled G1 interaction",
            visual_sensor
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

static MRTaskRolloutHandle* createImportedURDFRun(
    const char* urdf_path,
    const char* srdf_path,
    const char* task_pack_path,
    const char* robot_actuator_pack_path,
    const char* sensor_program_pack_path,
    const char* reality_program_pack_path,
    const MRTaskRolloutConfigC* config,
    const uint32_t surface_value,
    const char* interaction_pack_path,
    const char* interaction_clip_id,
    const MRTaskVisualObservationConfigC* visual_sensor,
    const char* metallib_path
) {
    if (config == nullptr ||
        urdf_path == nullptr || urdf_path[0] == '\0' ||
        task_pack_path == nullptr ||
        task_pack_path[0] == '\0' ||
        robot_actuator_pack_path == nullptr ||
        robot_actuator_pack_path[0] == '\0' ||
        sensor_program_pack_path == nullptr ||
        sensor_program_pack_path[0] == '\0' ||
        reality_program_pack_path == nullptr ||
        reality_program_pack_path[0] == '\0') {
        gLastError =
            "URDF, TaskPack, RobotActuatorPack, SensorProgramPack, "
            "RealityProgramPack, and task-rollout config are required.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        const metalrobo::LocomotionSurface surface =
            locomotionSurface(surface_value);
        metalrobo::TaskPack task;
        metalrobo::RobotActuatorPack actuators;
        metalrobo::SensorProgramPack sensors;
        metalrobo::RealityProgramPack reality;
        const metalrobo::LearningPackResult taskLoaded =
            metalrobo::readTaskPack(task_pack_path, task);
        const metalrobo::LearningPackResult actuatorLoaded =
            metalrobo::readRobotActuatorPack(
                robot_actuator_pack_path, actuators);
        const metalrobo::LearningPackResult sensorLoaded =
            metalrobo::readSensorProgramPack(
                sensor_program_pack_path, sensors);
        const metalrobo::LearningPackResult realityLoaded =
            metalrobo::readRealityProgramPack(
                reality_program_pack_path, reality);
        if (!taskLoaded.succeeded() || !actuatorLoaded.succeeded() ||
            !sensorLoaded.succeeded() || !realityLoaded.succeeded()) {
            throw std::invalid_argument(
                "one or more imported run owner artifacts failed to load"
            );
        }
        const bool hasInteraction = interaction_pack_path != nullptr &&
            interaction_pack_path[0] != '\0';
        if (hasInteraction != (interaction_clip_id != nullptr &&
                               interaction_clip_id[0] != '\0')) {
            throw std::invalid_argument(
                "imported InteractionPack requires both a path and clip identity"
            );
        }
        metalrobo::InteractionPack interactions;
        metalrobo::LearningPackResult interactionLoaded;
        const metalrobo::InteractionClip* interactionClip = nullptr;
        std::string interactionClipId;
        if (hasInteraction) {
            interactionLoaded = metalrobo::readInteractionPack(
                interaction_pack_path,
                interactions
            );
            if (!interactionLoaded.succeeded()) {
                throw std::invalid_argument(
                    std::string{"InteractionPack load failed ["} +
                    metalrobo::learningPackStatusName(
                        interactionLoaded.status) + "]: " +
                    interactionLoaded.message
                );
            }
            interactionClip = &selectedInteractionClip(
                interactions,
                interaction_clip_id
            );
            interactionClipId = interactionClip->id;
            // AI Sapiens mimic actors already emit the source's absolute
            // joint-position commands.  The InteractionPack is their
            // motion-conditioned observation/reset authority, not a second
            // residual controller layered onto those commands.
            task.interactionControlReference = false;
        }

        metalrobo::RobotDescriptionCookOptions options;
        options.rootMode =
            metalrobo::RobotDescriptionRootMode::floating;
        options.meshMode =
            metalrobo::RobotDescriptionMeshMode::convexHull;
        const bool aiSapiensK1 =
            task.id == "robotis_ai_sapiens_k1_velocity_v1" ||
            task.id == "robotis_ai_sapiens_k1_mimic_v1";
        if (aiSapiensK1) {
            // The pinned K1 MuJoCo floor is condim=3 with sliding friction
            // 0.8.  It has no torsional or rolling constraint rows, so map
            // that Coulomb coefficient to both MetalWorld sliding modes and
            // leave the unsupported rows at zero.
            options.friction = {0.8F, 0.8F, 0.0F, 0.0F};
        }
        metalrobo::LocomotionWorld authored;
        const metalrobo::RobotDescriptionDiagnostics cooked =
            metalrobo::cookRobotDescriptionFiles(
                urdf_path,
                srdf_path != nullptr && srdf_path[0] != '\0'
                    ? std::filesystem::path{srdf_path}
                    : std::filesystem::path{},
                authored.model,
                options
            );
        if (!cooked.succeeded()) {
            throw std::invalid_argument(
                std::string{"URDF/SRDF cook failed ["} +
                metalrobo::robotDescriptionStatusName(
                    cooked.status
                ) + "]: " + cooked.element + ": " +
                cooked.message
            );
        }
        metalrobo::LocomotionSceneComponent surfaceComponent =
            metalrobo::makeLocomotionSurfaceComponent(
                authored.model,
                surface
            );
        metalrobo::RunManifest manifest = makeImportedRunManifest(
            std::move(authored.model),
            std::move(authored.sceneBodies),
            authored.articulationIndex,
            std::move(task),
            std::move(actuators),
            std::move(sensors),
            std::move(reality),
            *config,
            "imported_urdf_run"
        );
        manifest.scene.objects.push_back({
            .id = "imported_locomotion_surface",
            .semanticClass = "support_surface",
            .role = MR_WORLD_ASSET_FIXTURE,
            .render = MR_WORLD_RENDER_NONE,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_STATIC,
            .mechanics = std::move(surfaceComponent.mechanics),
            .defaultBodyStates = std::move(surfaceComponent.defaultBodyStates),
        });
        const auto spheres = locomotionDynamicSpheres(*config);
        if (!spheres.empty()) {
            metalrobo::LocomotionSceneComponent sphereComponent =
                metalrobo::makeLocomotionDynamicSphereComponent(
                    manifest.robot.mechanics,
                    spheres
                );
            manifest.scene.objects.push_back({
                .id = "locomotion_dynamic_spheres",
                .semanticClass = "dynamic_projectile",
                .role = MR_WORLD_ASSET_MANIPULATED,
                .render = MR_WORLD_RENDER_NONE,
                .collision = MR_WORLD_COLLISION_PRIMITIVES,
                .dynamics = MR_WORLD_DYNAMICS_RIGID,
                .mechanics = std::move(sphereComponent.mechanics),
                .defaultBodyStates =
                    std::move(sphereComponent.defaultBodyStates),
            });
        }
        if (interactionClip != nullptr) {
            manifest.teacher = {
                .id = interactions.id,
                .kind = metalrobo::TeacherKind::motionImagination,
                .provider = "interaction_pack",
                .model = interactionClipId,
                .artifact = interaction_pack_path,
                .artifactFingerprint = interactionLoaded.contentHash,
                .interactions = std::move(interactions),
                .interactionClip = interactionClipId,
            };
        }
        auto handle = createCompiledRunTaskRollout(
            std::move(manifest),
            metallib_path,
            "imported URDF",
            visual_sensor
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

static MRTaskRolloutHandle* createWorldPackRun(
    const char* world_pack_path,
    const char* task_pack_path,
    const char* robot_actuator_pack_path,
    const char* sensor_program_pack_path,
    const char* reality_program_pack_path,
    const MRTaskRolloutConfigC* config,
    const MRTaskVisualObservationConfigC* visual_sensor,
    const char* metallib_path
) {
    if (config == nullptr ||
        world_pack_path == nullptr ||
        world_pack_path[0] == '\0' ||
        task_pack_path == nullptr ||
        task_pack_path[0] == '\0' ||
        robot_actuator_pack_path == nullptr ||
        robot_actuator_pack_path[0] == '\0' ||
        sensor_program_pack_path == nullptr ||
        sensor_program_pack_path[0] == '\0' ||
        reality_program_pack_path == nullptr ||
        reality_program_pack_path[0] == '\0') {
        gLastError =
            "MRWorldPack and all four executable owner artifacts are required.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::TaskPack task;
        metalrobo::RobotActuatorPack actuators;
        metalrobo::SensorProgramPack sensors;
        metalrobo::RealityProgramPack reality;
        const metalrobo::LearningPackResult taskLoaded =
            metalrobo::readTaskPack(task_pack_path, task);
        const metalrobo::LearningPackResult actuatorLoaded =
            metalrobo::readRobotActuatorPack(
                robot_actuator_pack_path, actuators);
        const metalrobo::LearningPackResult sensorLoaded =
            metalrobo::readSensorProgramPack(
                sensor_program_pack_path, sensors);
        const metalrobo::LearningPackResult realityLoaded =
            metalrobo::readRealityProgramPack(
                reality_program_pack_path, reality);
        if (!taskLoaded.succeeded() || !actuatorLoaded.succeeded() ||
            !sensorLoaded.succeeded() || !realityLoaded.succeeded()) {
            throw std::invalid_argument(
                "one or more WorldPack run owner artifacts failed to load"
            );
        }
        metalrobo::MRWorldPack worldPack;
        const metalrobo::WorldPackResult worldLoaded =
            metalrobo::readWorldPack(
                world_pack_path,
                worldPack
            );
        if (!worldLoaded.succeeded()) {
            throw std::invalid_argument(
                std::string{"MRWorldPack load failed ["} +
                metalrobo::worldPackStatusName(
                    worldLoaded.status
                ) + "]: " + worldLoaded.message
            );
        }
        metalrobo::LocomotionWorld materialized =
            metalrobo::makeWorldPackLocomotionWorld(worldPack);
        const auto spheres = locomotionDynamicSpheres(*config);
        if (!spheres.empty()) {
            metalrobo::appendLocomotionDynamicSpheres(materialized, spheres);
        }
        metalrobo::RunManifest run = makeImportedRunManifest(
            std::move(materialized.model),
            std::move(materialized.sceneBodies),
            materialized.articulationIndex,
            std::move(task),
            std::move(actuators),
            std::move(sensors),
            std::move(reality),
            *config,
            "world_pack_run"
        );
        run.robot.id = worldPack.family.worldTemplate.task.robotAssetId;
        run.scene.authoredAssets =
            worldPack.family.worldTemplate.assets;
        run.sensors.worldSensors =
            worldPack.family.worldTemplate.sensors;
        auto handle = createCompiledRunTaskRollout(
            std::move(run),
            metallib_path,
            "MRWorldPack",
            visual_sensor
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

MRTaskRolloutHandle* mr_create_task_rollout(
    const MRRunManifestC* manifest
) {
    if (manifest == nullptr) {
        gLastError = "run manifest is null.";
        return nullptr;
    }
    MRTaskRolloutHandle* result = nullptr;
    switch (manifest->source) {
    case MR_RUN_SOURCE_UNITREE_G1:
        if (manifest->teacher_pack_path != nullptr &&
            manifest->teacher_pack_path[0] != '\0') {
            result = createUnitreeG1TeacherRun(
                &manifest->profile,
                manifest->surface,
                manifest->task,
                manifest->teacher_pack_path,
                manifest->teacher_clip_id,
                manifest->visual_sensor_program,
                manifest->metallib_path
            );
            break;
        }
        result = createUnitreeG1Run(
            &manifest->profile,
            manifest->surface,
            manifest->task,
            manifest->visual_sensor_program,
            manifest->metallib_path
        );
        break;
    case MR_RUN_SOURCE_FRANKA_PICK_PLACE:
        result = createFrankaRun(
            &manifest->profile,
            manifest->visual_sensor_program,
            manifest->metallib_path
        );
        break;
    case MR_RUN_SOURCE_IMPORTED_URDF:
        result = createImportedURDFRun(
            manifest->urdf_path,
            manifest->srdf_path,
            manifest->task_pack_path,
            manifest->robot_actuator_pack_path,
            manifest->sensor_program_pack_path,
            manifest->reality_program_pack_path,
            &manifest->profile,
            manifest->surface,
            manifest->teacher_pack_path,
            manifest->teacher_clip_id,
            manifest->visual_sensor_program,
            manifest->metallib_path
        );
        break;
    case MR_RUN_SOURCE_WORLD_PACK:
        result = createWorldPackRun(
            manifest->world_pack_path,
            manifest->task_pack_path,
            manifest->robot_actuator_pack_path,
            manifest->sensor_program_pack_path,
            manifest->reality_program_pack_path,
            &manifest->profile,
            manifest->visual_sensor_program,
            manifest->metallib_path
        );
        break;
    case MR_RUN_SOURCE_PX4_X500:
        result = createPX4X500Run(
            &manifest->profile,
            manifest->metallib_path
        );
        break;
    case MR_RUN_SOURCE_MEASURED_DOVE:
        result = createMeasuredDoveRun(
            &manifest->profile,
            manifest->measured_surface_manifest_path,
            manifest->task,
            manifest->metallib_path
        );
        break;
    case MR_RUN_SOURCE_NUMIFLY:
        result = createNumiflyRun(
            &manifest->profile,
            manifest->measured_surface_manifest_path,
            manifest->surface,
            manifest->visual_sensor_program,
            manifest->metallib_path,
            NumiflyRunMode::hover
        );
        break;
    case MR_RUN_SOURCE_NUMIFLY_NO_LEGS:
        result = createNumiflyRun(
            &manifest->profile,
            manifest->measured_surface_manifest_path,
            manifest->surface,
            manifest->visual_sensor_program,
            manifest->metallib_path,
            NumiflyRunMode::noLegs
        );
        break;
    case MR_RUN_SOURCE_NUMIFLY_FORWARD_FLIGHT:
        result = createNumiflyRun(
            &manifest->profile,
            manifest->measured_surface_manifest_path,
            manifest->surface,
            manifest->visual_sensor_program,
            manifest->metallib_path,
            NumiflyRunMode::forwardFlight
        );
        break;
    default:
        gLastError = "run manifest source is invalid.";
        return nullptr;
    }
    if (result != nullptr && manifest->inspection_visual_program != nullptr) {
        const int inspectionStatus = translateErrors([&] {
            installTaskInspectionRuntime(
                *result,
                visualSensorProgram(*manifest->inspection_visual_program)
            );
        });
        if (inspectionStatus != 0) {
            delete result;
            return nullptr;
        }
    }
    return result;
}

void mr_task_rollout_destroy(MRTaskRolloutHandle* handle) {
    delete handle;
}

int mr_task_rollout_reset(
    MRTaskRolloutHandle* handle,
    const uint64_t seed
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    return translateErrors(
        [&] { resetTaskRolloutState(*handle, seed); }
    );
}

int mr_task_rollout_evidence_telemetry(
    const MRTaskRolloutHandle* handle,
    MRTaskEvidenceTelemetryC* telemetry
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    if (telemetry == nullptr) {
        gLastError = "task evidence telemetry output is null.";
        return -1;
    }
    return translateErrors([&] {
        if (!handle->residentState.valid()) {
            throw std::logic_error(
                "task evidence telemetry requires a completed resident submission"
            );
        }
        const MRTaskEvidenceStateGPU& state =
            handle->result.evidenceState;
        telemetry->control_steps = state.controlSteps;
        telemetry->evidence_windows = state.evidenceWindows;
        telemetry->pending_completed_episode_count =
            state.completedEpisodeCount;
        telemetry->pending_timeout_episode_count =
            state.timeoutEpisodeCount;
        telemetry->last_completed_episode_count =
            state.lastCompletedEpisodeCount;
        telemetry->last_contact_rate = state.lastWindow.x;
        telemetry->last_clean_miss_rate = state.lastWindow.y;
        telemetry->last_balance_failure_rate = state.lastWindow.z;
        telemetry->last_mean_tracking_per_million =
            state.lastWindow.w;
    });
}

int mr_task_rollout_set_policy(
    MRTaskRolloutHandle* handle,
    const MRPolicyPackC* policy
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    if (policy == nullptr) {
        gLastError = "task-rollout policy is null.";
        return -1;
    }
    return translateErrors([&] {
        metalrobo::PolicyPack authored =
            policyPackFromC(*policy);
        installPolicyPack(*handle, authored);
    });
}

int mr_task_rollout_load_policy_pack(
    MRTaskRolloutHandle* handle,
    const char* policy_pack_path
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    if (policy_pack_path == nullptr ||
        policy_pack_path[0] == '\0') {
        gLastError = "PolicyPack path is empty.";
        return -1;
    }
    return translateErrors([&] {
        metalrobo::PolicyPack authored;
        const metalrobo::LearningPackResult loaded =
            metalrobo::readPolicyPack(
                policy_pack_path,
                authored
            );
        if (!loaded.succeeded()) {
            throw std::invalid_argument(
                std::string{"PolicyPack load failed ["} +
                metalrobo::learningPackStatusName(
                    loaded.status
                ) + "]: " + loaded.message
            );
        }
        installPolicyPack(*handle, authored);
    });
}

uint64_t mr_task_rollout_policy_revision(
    const MRTaskRolloutHandle* handle
) {
    if (!requireTaskRolloutHandle(handle) ||
        !handle->stepConfig.policyProgram.valid()) {
        return 0u;
    }
    return handle->stepConfig.policyProgram.revision();
}

int mr_task_rollout_clear_policy(
    MRTaskRolloutHandle* handle
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    handle->stepConfig.policyProgram = {};
    gLastError.clear();
    return 0;
}

size_t mr_task_rollout_copy_visual_rgba(
    MRTaskRolloutHandle* handle,
    float* destination,
    const size_t destination_count,
    uint32_t* width,
    uint32_t* height
) {
    if (!requireTaskRolloutHandle(handle) ||
        handle->visualRuntime == nullptr ||
        !handle->visualRuntime->captureEnabled) {
        gLastError = "task rollout has no native presentation capture.";
        return 0u;
    }
    const std::uint32_t environments = handle->environmentCount;
    const std::size_t bodyCount = handle->model.bodies.size();
    const std::size_t nq = handle->world.nq();
    const std::size_t nv = handle->world.nv();
    if (handle->result.finalQ.size() != environments * nq ||
        handle->result.finalV.size() != environments * nv) {
        gLastError = "task visual capture requires final-state readback.";
        return 0u;
    }
    std::vector<MRBodyStateGPU> bodies;
    std::string compositionReason;
    if (!metalrobo::composeVisualBodyStates(
            handle->model,
            environments,
            handle->result.finalQ,
            handle->result.finalV,
            handle->result.finalSceneBodies,
            bodies,
            &compositionReason
        )) {
        gLastError =
            "task visual body composition failed: " +
            compositionReason;
        return 0u;
    }
    for (std::uint32_t environment = 0u;
         environment < environments;
         ++environment) {
        for (std::size_t body = 0u; body < bodyCount; ++body) {
            if (!std::string_view{handle->model.bodyNames[body]}
                    .starts_with("locomotion_dynamic_sphere_")) {
                continue;
            }
            MRBodyStateGPU& projectile =
                bodies[environment * bodyCount + body];
            const float speedSquared =
                projectile.linearVelocityAndInverseMass.x *
                    projectile.linearVelocityAndInverseMass.x +
                projectile.linearVelocityAndInverseMass.y *
                    projectile.linearVelocityAndInverseMass.y +
                projectile.linearVelocityAndInverseMass.z *
                    projectile.linearVelocityAndInverseMass.z;
            if (!handle->visualRuntime->capturePolicyCamera &&
                speedSquared <= 0.25f) {
                // Staged and expired projectiles remain real simulator state,
                // but are not active presentation subjects. The onboard
                // sensor path above continues to consume untouched states.
                projectile.position.z = -100.0f;
            }
        }
    }
    auto& runtime = *handle->visualRuntime;
    metalrobo::VisualMotionSampleBatchV1 motion;
    motion.environmentCount = environments;
    motion.bodyCount = static_cast<std::uint32_t>(bodyCount);
    motion.sampleCount = 2u;
    motion.exposureOpenSeconds = 0.0;
    motion.exposureCloseSeconds = 1.0 / 120.0;
    motion.timestampsSeconds = {
        motion.exposureOpenSeconds,
        motion.exposureCloseSeconds,
    };
    const auto& previous = runtime.previousCaptureBodies.empty()
        ? bodies
        : runtime.previousCaptureBodies;
    motion.bodyStates.reserve(2u * bodies.size());
    motion.bodyStates.insert(
        motion.bodyStates.end(),
        previous.begin(),
        previous.end()
    );
    motion.bodyStates.insert(
        motion.bodyStates.end(),
        bodies.begin(),
        bodies.end()
    );
    motion.frameIndex = ++runtime.captureFrameIndex;
    motion.sensorSequence =
        static_cast<std::uint32_t>(runtime.captureFrameIndex);
    motion.source = MR_VISUAL_SOURCE_SIMULATION;
    metalrobo::MetalHybridRendererDiagnostics render;
    if (runtime.captureUsesLiveDeformation) {
        metalrobo::HybridLiveStateBatch live;
        live.environmentCount = environments;
        live.bodyCount = static_cast<std::uint32_t>(bodyCount);
        live.currentBodies = bodies;
        live.previousBodies = previous;
        live.frameIndex = motion.frameIndex;
        live.sensorSequence = motion.sensorSequence;
        live.source = motion.source;
        render = runtime.captureRenderer.renderLive(
            runtime.worlds,
            live,
            1u
        );
    } else {
        render = runtime.captureRenderer.renderFrame(
            runtime.worlds,
            motion,
            1u
        );
    }
    if (!render.succeeded()) {
        gLastError = std::string{"task visual capture render failed: "} +
            render.message;
        return 0u;
    }
    runtime.previousCaptureBodies = bodies;
    metalrobo::HybridObservationBatch frame;
    const auto diagnostics =
        runtime.captureRenderer.readback(frame);
    if (!diagnostics.succeeded()) {
        gLastError =
            std::string{"task visual readback failed ["} +
            metalrobo::metalHybridRendererStatusName(
                diagnostics.status
            ) + "]: " + diagnostics.message;
        return 0u;
    }
    if (width != nullptr) {
        *width = frame.width;
    }
    if (height != nullptr) {
        *height = frame.height;
    }
    const size_t required = frame.rgb.size() * 4u;
    const bool anyColor = std::ranges::any_of(
        frame.rgb,
        [](const mr_float4 value) {
            return value.x != 0.0f || value.y != 0.0f || value.z != 0.0f;
        }
    );
    if (!anyColor) {
        const std::size_t visible = std::ranges::count_if(
            frame.segmentation,
            [](const std::uint32_t value) { return value != 0u; }
        );
        gLastError = "native presentation capture is empty; visible pixels=" +
            std::to_string(visible) + ", root=" +
            std::to_string(bodies.front().position.x) + "," +
            std::to_string(bodies.front().position.y) + "," +
            std::to_string(bodies.front().position.z);
        return 0u;
    }
    if (destination == nullptr) {
        gLastError.clear();
        return required;
    }
    if (destination_count < required) {
        gLastError = "task visual RGBA destination is too small.";
        return 0u;
    }
    static_assert(sizeof(mr_float4) == 4u * sizeof(float));
    std::memcpy(destination, frame.rgb.data(), required * sizeof(float));
    gLastError.clear();
    return required;
}

int mr_task_rollout_set_state_readback(
    MRTaskRolloutHandle* handle,
    const uint32_t enabled
) {
    if (!requireTaskRolloutHandle(handle) || enabled > 1u) {
        if (enabled > 1u) {
            gLastError = "task-rollout state-readback flag is invalid.";
        }
        return -1;
    }
    handle->stepConfig.publishFinalState = enabled != 0u;
    gLastError.clear();
    return 0;
}

int mr_task_rollout_advance(
    MRTaskRolloutHandle* handle,
    const float* normalized_actions,
    const size_t normalized_action_count,
    const uint32_t* reset_masks,
    const size_t reset_mask_count,
    const uint32_t control_step_count,
    const uint64_t policy_revision,
    const uint32_t evaluate_final_policy,
    MRTaskRolloutAdvanceC* advance
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    if (advance == nullptr) {
        gLastError = "task-rollout advance result is null.";
        return -1;
    }
    *advance = {};
    advance->first_failing_environment = UINT32_MAX;
    advance->first_failing_control_step = UINT32_MAX;
    if (control_step_count == 0u) {
        gLastError =
            "task-rollout control_step_count must be positive.";
        return -1;
    }

    return translateErrors([&] {
        handle->stepConfig.evaluateFinalPolicy =
            evaluate_final_policy != 0u;
        const std::size_t actionCount =
            handle->taskProgram.layout().actionCount;
        const std::size_t environmentCount =
            handle->environmentCount;
        const bool nativePolicy =
            handle->stepConfig.policyProgram.valid();
        const std::size_t requiredActions =
            nativePolicy
            ? 0u
            : checkedProduct(
                  {
                      control_step_count,
                      environmentCount,
                      actionCount,
                  },
                  "task-rollout action"
              );
        const std::size_t requiredMasks = checkedProduct(
            {control_step_count, environmentCount},
            "task-rollout reset mask"
        );
        if ((!nativePolicy &&
             normalized_actions == nullptr) ||
            normalized_action_count != requiredActions ||
            (nativePolicy &&
             normalized_actions != nullptr)) {
            throw std::invalid_argument(
                nativePolicy
                ? "native-policy rollout does not accept a host action stream"
                : "normalized task action count must equal step_count * environment_count * compiled action_count"
            );
        }
        if ((reset_masks == nullptr && reset_mask_count != 0u) ||
            (reset_masks != nullptr &&
             reset_mask_count != requiredMasks)) {
            throw std::invalid_argument(
                "task reset mask count must be zero or "
                "step_count * environment_count"
            );
        }

        handle->resetMasks.assign(requiredMasks, 0u);
        if (reset_masks != nullptr) {
            std::copy(
                reset_masks,
                reset_masks + reset_mask_count,
                handle->resetMasks.begin()
            );
        }
        advance->host_requested_resets =
            static_cast<std::uint32_t>(
                std::count_if(
                    handle->resetMasks.begin(),
                    handle->resetMasks.end(),
                    [](const std::uint32_t value) {
                        return value != 0u;
                    }
                )
            );

        const bool initializeResidentState =
            !handle->residentState.valid();
        const metalrobo::MetalWorldBatch batch{
            .environmentCount = environmentCount,
            .controlStepCount = control_step_count,
            .initialQ =
                initializeResidentState
                ? std::span<const float>{handle->resetQ}
                : std::span<const float>{},
            .initialV =
                initializeResidentState
                ? std::span<const float>{handle->resetV}
                : std::span<const float>{},
            .actions =
                nativePolicy
                ? std::span<const float>{}
                : std::span<const float>{
                      normalized_actions,
                      normalized_action_count
                  },
            .policyRevision = policy_revision,
            .resetMasks = handle->resetMasks,
            .initialSceneBodies =
                initializeResidentState
                ? std::span<const MRBodyStateGPU>{
                      handle->resetSceneBodies
                  }
                : std::span<const MRBodyStateGPU>{},
        };
        metalrobo::MetalWorldSubmission submission;
        metalrobo::MetalWorldResult published;
        metalrobo::MetalWorldDiagnostics diagnostics =
            initializeResidentState
            ? handle->context.initializeResidentState(
                handle->world,
                batch,
                handle->stepConfig,
                handle->residentState,
                submission
            )
            : handle->context.submitResident(
                handle->world,
                batch,
                handle->stepConfig,
                handle->residentState,
                submission
            );
        if (diagnostics.succeeded()) {
            diagnostics = submission.wait(published);
        }
        advance->control_step_count = control_step_count;
        advance->successful_environment_steps =
            diagnostics.successfulStepCount;
        advance->failed_environment_steps =
            diagnostics.failedStepCount;
        advance->first_failing_environment =
            diagnostics.firstFailingEnvironment;
        advance->first_failing_control_step =
            diagnostics.firstFailingControlStep;
        advance->first_gpu_status_code =
            diagnostics.firstGPUStatusCode;
        advance->gpu_milliseconds =
            diagnostics.gpuElapsedMilliseconds;
        advance->submission_milliseconds =
            diagnostics.submissionElapsedMilliseconds;

        handle->statusCodes.assign(
            requiredMasks,
            MR_STEP_SUCCESS
        );
        handle->activeContacts.assign(requiredMasks, 0u);
        for (std::size_t index = 0u;
             index < requiredMasks;
             ++index) {
            if (index < published.statuses.size()) {
                handle->statusCodes[index] =
                    published.statuses[index].code;
            }
            if (index < published.contactStatuses.size()) {
                const auto& contact =
                    published.contactStatuses[index];
                if (handle->statusCodes[index] ==
                    MR_STEP_SUCCESS) {
                    handle->statusCodes[index] = contact.code;
                }
                handle->activeContacts[index] =
                    contact.activeContacts;
                advance->maximum_active_contacts = std::max(
                    advance->maximum_active_contacts,
                    contact.activeContacts
                );
                advance->maximum_manifolds = std::max(
                    advance->maximum_manifolds,
                    contact.requiredManifolds
                );
            }
        }
        for (const metalrobo::MetalWorldStatus& status :
             published.environmentStatuses) {
            accumulateTaskStageHighWater(
                advance->high_water,
                status.highWater
            );
        }

        if (!diagnostics.published ||
            published.statuses.size() != requiredMasks ||
            published.contactStatuses.size() !=
                requiredMasks) {
            throw std::runtime_error(
                std::string{"task-rollout publication failed ["} +
                metalrobo::metalWorldHostStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->result = std::move(published);
        handle->outcomeValues.clear();
        handle->outcomeValues.reserve(
            handle->result.transitions.size() *
            handle->outcomes.size()
        );
        for (const MRTaskTransitionGPU& native :
             handle->result.transitions) {
            const auto& transition =
                reinterpret_cast<const MRTaskTransitionC&>(native);
            for (const MRTaskOutcomeDescriptor& outcome :
                 handle->outcomes) {
                handle->outcomeValues.push_back(
                    taskOutcomeValue(transition, outcome.source)
                );
            }
        }
        handle->deviceName = diagnostics.deviceName;
        handle->submittedControlSteps += control_step_count;
        handle->completedEnvironmentSteps +=
            diagnostics.successfulStepCount;
        ++handle->submissionCount;
        handle->totalGPUMilliseconds +=
            diagnostics.gpuElapsedMilliseconds;
        handle->totalSubmissionMilliseconds +=
            diagnostics.submissionElapsedMilliseconds;
        // Per-environment failures are transactional: the rejected state is
        // rolled back, the transition is published with physics_error, and
        // healthy environments remain valid. Returning the published advance
        // lets schedulers count/reset those environments instead of aborting
        // the whole batch. Host/publication failures still throw above.
    });
}

MRTaskRolloutLayoutC mr_task_rollout_layout(
    const MRTaskRolloutHandle* handle
) {
    MRTaskRolloutLayoutC result{};
    if (!requireTaskRolloutHandle(handle)) {
        return result;
    }
    const metalrobo::MetalWorldContextStats stats =
        handle->context.stats();
    result.environment_count = handle->environmentCount;
    result.nq = static_cast<std::uint32_t>(handle->world.nq());
    result.nv = static_cast<std::uint32_t>(handle->world.nv());
    result.action_count =
        handle->taskProgram.layout().actionCount;
    result.actor_observation_count =
        handle->taskProgram.layout().actorObservationSize;
    result.critic_observation_count =
        handle->taskProgram.layout().criticObservationSize;
    result.scene_body_count =
        static_cast<std::uint32_t>(
            handle->defaultSceneBodies.size()
        );
    result.motion_feature_count =
        handle->taskProgram.layout().motionFeatureCount;
    result.maximum_episode_steps =
        handle->taskProgram.header().schedule.x;
    result.world_fingerprint =
        handle->taskProgram.worldFingerprint();
    result.task_fingerprint =
        handle->taskProgram.fingerprint();
    result.observation_fingerprint =
        handle->taskProgram.observationFingerprint();
    result.action_fingerprint =
        handle->taskProgram.actionFingerprint();
    result.run_fingerprint = handle->run.fingerprint();
    result.robot_fingerprint = handle->run.robotFingerprint();
    result.sensor_fingerprint = handle->run.sensorFingerprint();
    result.reality_fingerprint = handle->run.realityFingerprint();
    result.teacher_fingerprint = handle->run.teacherFingerprint();
    result.submitted_control_steps =
        handle->submittedControlSteps;
    result.completed_environment_steps =
        handle->completedEnvironmentSteps;
    result.submission_count = handle->submissionCount;
    result.retained_buffer_bytes = stats.retainedBufferBytes;
    result.immutable_private_bytes =
        stats.memoryPlan.immutablePrivateBytes;
    result.persistent_state_private_bytes =
        stats.memoryPlan.persistentStatePrivateBytes;
    result.transient_private_bytes =
        stats.memoryPlan.transientPrivateBytes;
    result.shared_boundary_bytes =
        stats.memoryPlan.sharedBoundaryBytes;
    result.peak_aliased_bytes =
        stats.memoryPlan.peakAliasedBytes;
    result.total_gpu_milliseconds =
        handle->totalGPUMilliseconds;
    result.total_submission_milliseconds =
        handle->totalSubmissionMilliseconds;
    return result;
}

const char* mr_task_rollout_task_id(
    const MRTaskRolloutHandle* handle
) {
    if (!requireTaskRolloutHandle(handle)) {
        return nullptr;
    }
    gLastError.clear();
    return handle->taskId.c_str();
}

const char* mr_task_rollout_device_name(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle)
        ? handle->deviceName.c_str()
        : "";
}

uint64_t mr_task_rollout_visual_scene_fingerprint(
    const MRTaskRolloutHandle* handle
) {
    if (!requireTaskRolloutHandle(handle)) {
        return 0u;
    }
    return handle->visualRuntime == nullptr
        ? 0u
        : handle->visualRuntime->sceneFingerprint;
}

int mr_task_rollout_acquire_inspection_frame(
    MRTaskRolloutHandle* handle,
    MRTaskInspectionFrameC* frame
) {
    if (!requireTaskRolloutHandle(handle) || frame == nullptr) {
        return -1;
    }
    if (handle->inspectionVisualRuntime == nullptr ||
        handle->inspectionVisualRuntime->inspector == nullptr) {
        gLastError = "task rollout has no presentation inspector.";
        return -1;
    }
    metalrobo::MetalRunInspectorFrame native;
    if (!handle->inspectionVisualRuntime->inspector->acquireLatestFrame(
            native
        )) {
        gLastError.clear();
        return 0;
    }
    frame->rgb_buffer = native.rgb;
    frame->slot_index = native.slotIndex;
    frame->width = native.width;
    frame->height = native.height;
    frame->frame_index = native.frameIndex;
    frame->submission_index = native.submissionIndex;
    frame->environment_index = native.environmentIndex;
    frame->dropped_frames = native.droppedFrames;
    gLastError.clear();
    return 1;
}

int mr_task_rollout_set_inspection_enabled(
    MRTaskRolloutHandle* handle,
    const uint32_t enabled
) {
    if (!requireTaskRolloutHandle(handle) || enabled > 1u ||
        handle->inspectionVisualRuntime == nullptr ||
        handle->inspectionVisualRuntime->inspector == nullptr) {
        return -1;
    }
    handle->inspectionVisualRuntime->inspector->setEnabled(enabled != 0u);
    gLastError.clear();
    return 0;
}

int mr_task_rollout_set_inspection_camera(
    MRTaskRolloutHandle* handle,
    const float translation_x,
    const float translation_y,
    const float translation_z,
    const float quaternion_x,
    const float quaternion_y,
    const float quaternion_z,
    const float quaternion_w
) {
    if (!requireTaskRolloutHandle(handle) ||
        handle->inspectionVisualRuntime == nullptr ||
        handle->inspectionVisualRuntime->inspector == nullptr) {
        return -1;
    }
    const float norm = std::sqrt(
        quaternion_x * quaternion_x + quaternion_y * quaternion_y +
        quaternion_z * quaternion_z + quaternion_w * quaternion_w
    );
    if (!std::isfinite(translation_x) || !std::isfinite(translation_y) ||
        !std::isfinite(translation_z) || !std::isfinite(norm) ||
        norm < 1.0e-6f) {
        gLastError = "inspection camera transform is not finite";
        return -1;
    }
    handle->inspectionVisualRuntime->inspector->setCameraOverride(
        {translation_x, translation_y, translation_z, 1.0f},
        {quaternion_x / norm, quaternion_y / norm,
         quaternion_z / norm, quaternion_w / norm}
    );
    gLastError.clear();
    return 0;
}

int mr_task_rollout_release_inspection_frame(
    MRTaskRolloutHandle* handle,
    const uint32_t slot_index
) {
    if (!requireTaskRolloutHandle(handle) ||
        handle->inspectionVisualRuntime == nullptr ||
        handle->inspectionVisualRuntime->inspector == nullptr) {
        return -1;
    }
    handle->inspectionVisualRuntime->inspector->releaseFrame(slot_index);
    gLastError.clear();
    return 0;
}

uint32_t mr_task_rollout_impact_event_count(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle)
        ? static_cast<uint32_t>(
              handle->taskProgram.impactEvents().size()
          )
        : 0u;
}

const uint32_t* mr_task_rollout_status_codes(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->statusCodes.empty()
        ? handle->statusCodes.data()
        : nullptr;
}

const uint32_t* mr_task_rollout_active_contacts(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->activeContacts.empty()
        ? handle->activeContacts.data()
        : nullptr;
}

const float* mr_task_rollout_actor_observations(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.actorObservations.empty()
        ? handle->result.actorObservations.data()
        : nullptr;
}

const float* mr_task_rollout_critic_observations(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.criticObservations.empty()
        ? handle->result.criticObservations.data()
        : nullptr;
}

const float* mr_task_rollout_motion_features(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.motionFeatures.empty()
        ? handle->result.motionFeatures.data()
        : nullptr;
}

const float* mr_task_rollout_teacher_actions(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        (handle->taskProgram.header().schedule.w &
         MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u &&
        !handle->result.teacherActions.empty()
        ? handle->result.teacherActions.data()
        : nullptr;
}

const MRTaskTransitionC* mr_task_rollout_transitions(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.transitions.empty()
        ? reinterpret_cast<const MRTaskTransitionC*>(
              handle->result.transitions.data()
          )
        : nullptr;
}

uint32_t mr_task_rollout_outcome_count(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle)
        ? static_cast<uint32_t>(handle->outcomes.size())
        : 0u;
}

const char* mr_task_rollout_outcome_id(
    const MRTaskRolloutHandle* handle,
    const uint32_t outcome_index
) {
    if (!requireTaskRolloutHandle(handle) ||
        outcome_index >= handle->outcomes.size()) {
        gLastError = "task outcome index is out of range.";
        return nullptr;
    }
    gLastError.clear();
    return handle->outcomes[outcome_index].id.c_str();
}

const char* mr_task_rollout_outcome_unit(
    const MRTaskRolloutHandle* handle,
    const uint32_t outcome_index
) {
    if (!requireTaskRolloutHandle(handle) ||
        outcome_index >= handle->outcomes.size()) {
        gLastError = "task outcome index is out of range.";
        return nullptr;
    }
    gLastError.clear();
    return handle->outcomes[outcome_index].unit.c_str();
}

uint32_t mr_task_rollout_outcome_direction(
    const MRTaskRolloutHandle* handle,
    const uint32_t outcome_index
) {
    if (!requireTaskRolloutHandle(handle) ||
        outcome_index >= handle->outcomes.size()) {
        gLastError = "task outcome index is out of range.";
        return MR_TASK_OUTCOME_NEUTRAL;
    }
    gLastError.clear();
    return handle->outcomes[outcome_index].direction;
}

const float* mr_task_rollout_outcome_values(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->outcomeValues.empty()
        ? handle->outcomeValues.data()
        : nullptr;
}

const float* mr_task_rollout_policy_latents(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.policyLatents.empty()
        ? handle->result.policyLatents.data()
        : nullptr;
}

const float* mr_task_rollout_policy_log_probabilities(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.policyLogProbabilities.empty()
        ? handle->result.policyLogProbabilities.data()
        : nullptr;
}

const float* mr_task_rollout_policy_values(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.policyValues.empty()
        ? handle->result.policyValues.data()
        : nullptr;
}

const float* mr_task_rollout_bootstrap_policy_values(
    const MRTaskRolloutHandle* handle
) {
    if (!requireTaskRolloutHandle(handle)) {
        return nullptr;
    }
    const std::size_t offset =
        handle->result.transitions.size();
    const std::size_t required =
        offset + handle->environmentCount;
    return handle->result.policyValues.size() >= required
        ? handle->result.policyValues.data() + offset
        : nullptr;
}

const float* mr_task_rollout_final_q(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle) &&
        !handle->result.finalQ.empty()
        ? handle->result.finalQ.data()
        : nullptr;
}

int mr_task_rollout_copy_final_scene_states(
    const MRTaskRolloutHandle* handle,
    float* output,
    const size_t output_count
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    constexpr std::size_t stride = 13u;
    const std::size_t required =
        handle->result.finalSceneBodies.size() * stride;
    if (required == 0u || output == nullptr ||
        output_count != required) {
        gLastError =
            "final scene-state output has the wrong size";
        return -1;
    }
    std::size_t cursor = 0u;
    for (const MRBodyStateGPU& state :
         handle->result.finalSceneBodies) {
        for (const float value : {
            state.position.x, state.position.y, state.position.z,
            state.orientation.x, state.orientation.y,
            state.orientation.z, state.orientation.w,
            state.linearVelocityAndInverseMass.x,
            state.linearVelocityAndInverseMass.y,
            state.linearVelocityAndInverseMass.z,
            state.angularVelocity.x, state.angularVelocity.y,
            state.angularVelocity.z,
        }) {
            output[cursor++] = value;
        }
    }
    return 0;
}

int mr_task_rollout_write_policy_rollout_pack(
    const MRTaskRolloutHandle* handle,
    const MRPolicyRolloutBatchC* batch,
    const char* batch_id,
    const char* output_path
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    if (batch == nullptr ||
        batch_id == nullptr || batch_id[0] == '\0' ||
        output_path == nullptr || output_path[0] == '\0') {
        gLastError =
            "policy rollout batch, identity, and output path are required.";
        return -1;
    }
    return translateErrors([&] {
        if (!handle->stepConfig.policyProgram.valid()) {
            throw std::invalid_argument(
                "a compiled policy must be installed before publishing a rollout pack"
            );
        }
        const auto floats = [](
            const float* values,
            const std::size_t count,
            const char* label
        ) {
            if (values == nullptr && count != 0u) {
                throw std::invalid_argument(
                    std::string{label} + " pointer is null"
                );
            }
            return std::span<const float>{values, count};
        };
        if (batch->transition_count != 0u &&
            batch->transitions == nullptr) {
            throw std::invalid_argument(
                "rollout transition pointer is null"
            );
        }
        std::vector<MRLearningTransitionGPU> transitions(
            batch->transition_count
        );
        std::vector<metalrobo::PolicyOutcomeDescriptor> outcomes;
        outcomes.reserve(handle->outcomes.size());
        for (const MRTaskOutcomeDescriptor& outcome : handle->outcomes) {
            outcomes.push_back({
                .id = outcome.id,
                .unit = outcome.unit,
                .direction = outcome.direction,
            });
        }
        std::vector<float> outcomeValues;
        outcomeValues.reserve(
            batch->transition_count * handle->outcomes.size()
        );
        for (std::size_t index = 0u;
             index < batch->transition_count;
             ++index) {
            const MRTaskTransitionC& source = batch->transitions[index];
            MRLearningTransitionGPU& destination = transitions[index];
            destination.rewardAndBootstrap = {
                source.reward,
                source.timeout_bootstrap_value,
                0.0f,
                0.0f,
            };
            destination.termination = {
                source.done,
                source.timeout,
                source.physics_error,
                source.termination_reason,
            };
            destination.context = {
                source.difficulty_band,
                source.terrain_level,
                source.impact_sequence_index,
                source.impact_event_flags,
            };
            destination.policyRevision = source.policy_revision;
            for (const MRTaskOutcomeDescriptor& outcome :
                 handle->outcomes) {
                outcomeValues.push_back(
                    taskOutcomeValue(source, outcome.source)
                );
            }
        }
        const metalrobo::PolicyRolloutPackView authored{
            .id = batch_id,
            .taskFingerprint =
                handle->taskProgram.fingerprint(),
            .policyFingerprint =
                handle->stepConfig.policyProgram.fingerprint(),
            .policyRevision =
                handle->stepConfig.policyProgram.revision(),
            .environmentCount =
                handle->environmentCount,
            .controlStepCount =
                batch->control_step_count,
            .actorObservationCount =
                handle->taskProgram.layout()
                    .actorObservationSize,
            .criticObservationCount =
                handle->taskProgram.layout()
                    .criticObservationSize,
            .actionCount =
                handle->taskProgram.layout().actionCount,
            .motionFeatureCount =
                handle->taskProgram.layout().motionFeatureCount,
            .actorObservations = floats(
                batch->actor_observations,
                batch->actor_observation_count,
                "rollout actor observations"
            ),
            .criticObservations = floats(
                batch->critic_observations,
                batch->critic_observation_count,
                "rollout critic observations"
            ),
            .motionFeatures = floats(
                batch->motion_features,
                batch->motion_feature_count,
                "rollout motion features"
            ),
            .teacherActions = floats(
                batch->teacher_actions,
                batch->teacher_action_count,
                "rollout teacher actions"
            ),
            .latents = floats(
                batch->latents,
                batch->latent_count,
                "rollout latents"
            ),
            .logProbabilities = floats(
                batch->log_probabilities,
                batch->log_probability_count,
                "rollout log probabilities"
            ),
            .values = floats(
                batch->values,
                batch->value_count,
                "rollout values"
            ),
            .bootstrapValues = floats(
                batch->bootstrap_values,
                batch->bootstrap_value_count,
                "rollout bootstrap values"
            ),
            .outcomes = outcomes,
            .outcomeValues = outcomeValues,
            .transitions = transitions,
        };
        const metalrobo::LearningPackResult written =
            metalrobo::writePolicyRolloutPack(
                authored,
                output_path
            );
        if (!written.succeeded()) {
            throw std::runtime_error(
                std::string{
                    "PolicyRolloutPack write failed ["
                } +
                metalrobo::learningPackStatusName(
                    written.status
                ) + "]: " + written.message
            );
        }
    });
}

MRWorldFamilyHandle* mr_create_franka_pick_place_world_family(
    const uint32_t capacity,
    const char* metallib_path
) {
    if (capacity == 0u) {
        gLastError = "world-family capacity must be greater than zero.";
        return nullptr;
    }

    MRWorldFamilyHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::WorldTemplate worldTemplate;
        const metalrobo::WorldCompileResult twin =
            metalrobo::compileEpisodeTwin(
                metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                metalrobo::makeFrankaPickPlaceEngineModel(),
                worldTemplate
            );
        if (!twin.succeeded()) {
            throw std::runtime_error(
                "Franka episode compilation failed: " + twin.message
            );
        }

        metalrobo::WorldFamily family;
        const metalrobo::WorldCompileResult compiled =
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            );
        if (!compiled.succeeded()) {
            throw std::runtime_error(
                "Franka world-family compilation failed: " +
                compiled.message
            );
        }

        metalrobo::MetalWorldFamilyConfig config;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRWorldFamilyHandle>();
        handle->context = metalrobo::MetalWorldFamilyContext{
            std::move(config)
        };
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.compile(family, capacity);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError("world-family compile", diagnostics);
        }
        handle->family = std::move(family);
        handle->scenarioSchema =
            metalrobo::compileScenarioSchema(handle->family);
        handle->deviceName = diagnostics.deviceName;
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

MRWorldFamilyHandle* mr_load_world_family_pack(
    const char* pack_path,
    const uint32_t capacity,
    const char* metallib_path
) {
    if (pack_path == nullptr || pack_path[0] == '\0') {
        gLastError = "world-pack path must be nonempty.";
        return nullptr;
    }
    if (capacity == 0u) {
        gLastError = "world-family capacity must be greater than zero.";
        return nullptr;
    }

    MRWorldFamilyHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::MRWorldPack pack;
        const metalrobo::WorldPackResult loaded =
            metalrobo::readWorldPack(pack_path, pack);
        if (!loaded.succeeded()) {
            throw std::runtime_error(
                std::string{"world-pack load failed ["} +
                metalrobo::worldPackStatusName(loaded.status) +
                "]: " + loaded.message
            );
        }
        metalrobo::MetalWorldFamilyConfig config;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRWorldFamilyHandle>();
        handle->context = metalrobo::MetalWorldFamilyContext{
            std::move(config)
        };
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.compile(pack, capacity);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError(
                "packed world-family compile",
                diagnostics
            );
        }
        handle->family = std::move(pack.family);
        handle->scenarioSchema =
            metalrobo::compileScenarioSchema(handle->family);
        handle->authoredPackHash = pack.contentHash;
        handle->deviceName = diagnostics.deviceName;
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_world_family_retain(MRWorldFamilyHandle* handle) {
    if (handle != nullptr) {
        handle->references.fetch_add(1u, std::memory_order_relaxed);
    }
}

void mr_world_family_destroy(MRWorldFamilyHandle* handle) {
    if (handle != nullptr &&
        handle->references.fetch_sub(
            1u,
            std::memory_order_acq_rel
        ) == 1u) {
        delete handle;
    }
}

int mr_world_family_sample(
    MRWorldFamilyHandle* handle,
    const uint32_t instance_count,
    const uint64_t seed
) {
    if (!requireWorldFamilyHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.sample(instance_count, seed);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError("world-family sample", diagnostics);
        }
        handle->lastSampleMilliseconds =
            diagnostics.elapsedMilliseconds;
        handle->readback = {};
    });
}

int mr_world_family_sample_ex(
    MRWorldFamilyHandle* handle,
    const uint32_t instance_count,
    const uint64_t seed,
    const uint32_t sampling_mode,
    const uint64_t episode_counter
) {
    if (!requireWorldFamilyHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        if (sampling_mode > MR_WORLD_SAMPLING_REPLAY) {
            throw std::invalid_argument(
                "world-family sampling mode is invalid"
            );
        }
        const auto diagnostics = handle->context.sample(
            instance_count,
            seed,
            static_cast<MRWorldSamplingMode>(sampling_mode),
            episode_counter
        );
        if (!diagnostics.succeeded()) {
            throw worldFamilyError(
                "adaptive world-family sample",
                diagnostics
            );
        }
        handle->lastSampleMilliseconds =
            diagnostics.elapsedMilliseconds;
        handle->readback = {};
    });
}

int mr_world_family_configure_sampling(
    MRWorldFamilyHandle* handle,
    const uint64_t alignment_fingerprint,
    const float* particle_quantiles,
    const float* particle_weights,
    const float* particle_residuals,
    const uint32_t particle_count,
    const uint64_t feedback_fingerprint,
    const uint32_t* region_kinds,
    const float* region_weights,
    const float* region_bounds,
    const uint32_t region_count,
    const float broad_weight,
    const float failure_weight,
    const float uncertainty_weight,
    const float alignment_jitter
) {
    if (!requireWorldFamilyHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        const std::size_t featureCount =
            handle->scenarioSchema.features.size();
        if (!handle->scenarioSchema.valid() ||
            featureCount == 0u ||
            particle_count > metalrobo::kMaximumAlignmentParticles ||
            region_count > metalrobo::kMaximumFeedbackRegions ||
            (particle_count != 0u &&
             (alignment_fingerprint == 0u ||
              particle_quantiles == nullptr ||
              particle_weights == nullptr)) ||
            (region_count != 0u &&
             (feedback_fingerprint == 0u ||
              region_kinds == nullptr ||
              region_weights == nullptr ||
              region_bounds == nullptr))) {
            throw std::invalid_argument(
                "adaptive sampling arrays or fingerprints are invalid"
            );
        }
        metalrobo::CompiledWorldSamplingProgram program;
        program.schemaFingerprint =
            handle->scenarioSchema.fingerprint;
        program.alignmentFingerprint = alignment_fingerprint;
        program.feedbackFingerprint = feedback_fingerprint;
        program.broadWeight = broad_weight;
        program.failureWeight = failure_weight;
        program.uncertaintyWeight = uncertainty_weight;
        program.alignmentJitter = alignment_jitter;

        double particleTotal = 0.0;
        for (std::uint32_t particle = 0u;
             particle < particle_count;
             ++particle) {
            if (!std::isfinite(particle_weights[particle]) ||
                particle_weights[particle] < 0.0f) {
                throw std::invalid_argument(
                    "alignment particle weight is invalid"
                );
            }
            particleTotal += particle_weights[particle];
        }
        if (particle_count != 0u && !(particleTotal > 0.0)) {
            throw std::invalid_argument(
                "alignment particle weights have no mass"
            );
        }
        double particleCDF = 0.0;
        program.alignmentParticles.reserve(particle_count);
        program.alignmentQuantiles.reserve(
            static_cast<std::size_t>(particle_count) * featureCount
        );
        for (std::uint32_t particle = 0u;
             particle < particle_count;
             ++particle) {
            const double weight =
                particle_weights[particle] / particleTotal;
            particleCDF += weight;
            MRWorldAlignmentParticleGPU record{};
            record.statistics = {
                static_cast<float>(weight),
                static_cast<float>(
                    particle + 1u == particle_count
                    ? 1.0
                    : particleCDF
                ),
                particle_residuals == nullptr
                    ? 0.0f
                    : particle_residuals[particle],
                0.0f,
            };
            record.identity = {particle, 0u, 0u, 0u};
            program.alignmentParticles.push_back(record);
            for (std::size_t feature = 0u;
                 feature < featureCount;
                 ++feature) {
                program.alignmentQuantiles.push_back(
                    particle_quantiles[
                        static_cast<std::size_t>(particle) *
                            featureCount +
                        feature
                    ]
                );
            }
        }

        std::array<double, 2> regionTotals{};
        for (std::uint32_t region = 0u;
             region < region_count;
             ++region) {
            if (region_kinds[region] >
                    MR_FEEDBACK_REGION_UNCERTAINTY ||
                !std::isfinite(region_weights[region]) ||
                region_weights[region] < 0.0f) {
                throw std::invalid_argument(
                    "feedback region kind or weight is invalid"
                );
            }
            regionTotals[region_kinds[region]] +=
                region_weights[region];
        }
        std::array<double, 2> regionCDF{};
        program.feedbackRegions.reserve(region_count);
        program.feedbackBounds.reserve(
            static_cast<std::size_t>(region_count) * featureCount
        );
        for (std::uint32_t region = 0u;
             region < region_count;
             ++region) {
            const std::uint32_t kind = region_kinds[region];
            const double total = regionTotals[kind];
            const double weight =
                total > 0.0 ? region_weights[region] / total : 0.0;
            regionCDF[kind] += weight;
            MRWorldFeedbackRegionGPU record{};
            record.statistics = {
                static_cast<float>(weight),
                static_cast<float>(
                    std::abs(regionCDF[kind] - 1.0) < 1.0e-12
                    ? 1.0
                    : regionCDF[kind]
                ),
                0.0f,
                0.0f,
            };
            record.identity = {kind, region, 0u, 0u};
            program.feedbackRegions.push_back(record);
            for (std::size_t feature = 0u;
                 feature < featureCount;
                 ++feature) {
                const std::size_t offset =
                    (
                        static_cast<std::size_t>(region) *
                            featureCount +
                        feature
                    ) * 2u;
                program.feedbackBounds.push_back({
                    region_bounds[offset],
                    region_bounds[offset + 1u],
                    0.0f,
                    0.0f,
                });
            }
        }
        std::string reason;
        if (!program.valid(handle->scenarioSchema, &reason)) {
            throw std::invalid_argument(reason);
        }
        const auto diagnostics =
            handle->context.configureSamplingProgram(
                handle->scenarioSchema,
                program
            );
        if (!diagnostics.succeeded()) {
            throw worldFamilyError(
                "world-family sampling configuration",
                diagnostics
            );
        }
    });
}

uint64_t mr_world_family_scenario_fingerprint(
    const MRWorldFamilyHandle* handle
) {
    return requireWorldFamilyHandle(handle)
        ? handle->scenarioSchema.fingerprint
        : 0u;
}

uint64_t mr_world_family_authored_pack_hash(
    const MRWorldFamilyHandle* handle
) {
    return requireWorldFamilyHandle(handle)
        ? handle->authoredPackHash
        : 0u;
}

const char* mr_world_family_scenario_id(
    const MRWorldFamilyHandle* handle
) {
    return requireWorldFamilyHandle(handle)
        ? handle->scenarioSchema.id.c_str()
        : "";
}

const char* mr_world_family_scenario_feature_id(
    const MRWorldFamilyHandle* handle,
    const uint32_t feature
) {
    if (!requireWorldFamilyHandle(handle) ||
        feature >= handle->scenarioSchema.features.size()) {
        gLastError = "scenario feature index is invalid.";
        return "";
    }
    return handle->scenarioSchema.features[feature].id.c_str();
}

const char* mr_world_family_scenario_target_id(
    const MRWorldFamilyHandle* handle,
    const uint32_t feature
) {
    if (!requireWorldFamilyHandle(handle) ||
        feature >= handle->scenarioSchema.features.size()) {
        gLastError = "scenario feature index is invalid.";
        return "";
    }
    return handle->scenarioSchema.features[feature].targetId.c_str();
}

MRScenarioFeatureC mr_world_family_scenario_feature(
    const MRWorldFamilyHandle* handle,
    const uint32_t feature
) {
    MRScenarioFeatureC result{};
    if (!requireWorldFamilyHandle(handle) ||
        feature >= handle->scenarioSchema.features.size()) {
        gLastError = "scenario feature index is invalid.";
        return result;
    }
    const auto& source = handle->scenarioSchema.features[feature];
    result.axis = source.axis;
    result.distribution = source.distribution;
    result.target = source.target;
    result.ordinal = source.ordinal;
    result.parameters[0] = source.parameters.x;
    result.parameters[1] = source.parameters.y;
    result.parameters[2] = source.parameters.z;
    result.parameters[3] = source.parameters.w;
    return result;
}

int mr_world_family_readback(MRWorldFamilyHandle* handle) {
    if (!requireWorldFamilyHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::WorldInstanceBatch staged;
        const metalrobo::MetalWorldFamilyDiagnostics diagnostics =
            handle->context.readback(staged);
        if (!diagnostics.succeeded()) {
            throw worldFamilyError("world-family readback", diagnostics);
        }
        handle->readback = std::move(staged);
    });
}

MRWorldFamilyLayoutC mr_world_family_layout(
    const MRWorldFamilyHandle* handle
) {
    MRWorldFamilyLayoutC result{};
    if (!requireWorldFamilyHandle(handle)) {
        return result;
    }
    const metalrobo::MetalWorldFamilyLayout layout =
        handle->context.layout();
    result.capacity = layout.capacity;
    result.active_instance_count = layout.activeInstanceCount;
    result.asset_count_per_instance =
        layout.assetCountPerInstance;
    result.sensor_count_per_instance =
        layout.sensorCountPerInstance;
    result.appearance_count_per_instance =
        layout.appearanceCountPerInstance;
    result.variation_count = layout.variationCount;
    result.categorical_value_count =
        layout.categoricalValueCount;
    result.asset_binding_count = layout.assetBindingCount;
    result.binding_index_count = layout.bindingIndexCount;
    result.primary_articulation_index =
        layout.primaryArticulationIndex;
    result.nq = layout.nq;
    result.nv = layout.nv;
    result.body_count = layout.bodyCount;
    result.scene_body_count = layout.sceneBodyCount;
    result.articulation_count = layout.articulationCount;
    result.retained_private_bytes = layout.totalPrivateBytes();
    return result;
}

MRWorldFamilyStatsC mr_world_family_stats(
    const MRWorldFamilyHandle* handle
) {
    MRWorldFamilyStatsC result{};
    if (!requireWorldFamilyHandle(handle)) {
        return result;
    }
    const metalrobo::MetalWorldFamilyStats stats =
        handle->context.stats();
    result.compile_count = stats.compileCount;
    result.sample_count = stats.sampleCount;
    result.readback_count = stats.readbackCount;
    result.last_sample_milliseconds =
        handle->lastSampleMilliseconds;
    return result;
}

const char* mr_world_family_device_name(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle)) {
        return "";
    }
    return handle->deviceName.c_str();
}

void* mr_world_family_native_buffer(
    const MRWorldFamilyHandle* handle,
    const uint32_t buffer_kind
) {
    if (!requireWorldFamilyHandle(handle) || buffer_kind > 12u) {
        if (buffer_kind > 12u) {
            gLastError = "world-family buffer kind is invalid.";
        }
        return nullptr;
    }
    return handle->context.nativeBuffer(
        static_cast<metalrobo::MetalWorldFamilyBuffer>(buffer_kind)
    );
}

const MRWorldInstanceHeaderGPU* mr_world_family_instance_headers(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.instances.empty()) {
        return nullptr;
    }
    return handle->readback.instances.data();
}

const MRWorldAssetInstanceGPU* mr_world_family_asset_instances(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.assets.empty()) {
        return nullptr;
    }
    return handle->readback.assets.data();
}

const MRWorldSensorInstanceGPU* mr_world_family_sensor_instances(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.sensors.empty()) {
        return nullptr;
    }
    return handle->readback.sensors.data();
}

const MRWorldAppearanceInstanceGPU*
mr_world_family_appearance_instances(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.appearances.empty()) {
        return nullptr;
    }
    return handle->readback.appearances.data();
}

const MRWorldScenarioHeaderGPU* mr_world_family_scenario_headers(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.scenarioHeaders.empty()) {
        return nullptr;
    }
    return handle->readback.scenarioHeaders.data();
}

const MRWorldScenarioValueGPU* mr_world_family_scenario_values(
    const MRWorldFamilyHandle* handle
) {
    if (!requireWorldFamilyHandle(handle) ||
        handle->readback.scenarioValues.empty()) {
        return nullptr;
    }
    return handle->readback.scenarioValues.data();
}

MRHybridRendererHandle* mr_hybrid_renderer_create_v3(
    const MRHybridGaussianC* gaussians,
    const size_t gaussian_count,
    const char* visual_pack_path,
    const char* environment_pack_path,
    const uint32_t asset_count,
    const uint32_t body_count,
    const uint32_t visual_asset_index,
    const uint32_t semantic_id,
    const uint32_t instance_id,
    const char* light_rig,
    const char* renderer_profile,
    const uint32_t capacity,
    const uint32_t width,
    const uint32_t height,
    const uint32_t retain_observation_buffers,
    const char* metallib_path
) {
    const bool hasGaussians = gaussian_count != 0u;
    const bool hasPack =
        visual_pack_path != nullptr &&
        visual_pack_path[0] != '\0';
    const bool hasEnvironmentPack =
        environment_pack_path != nullptr &&
        environment_pack_path[0] != '\0';
    if ((!hasGaussians && !hasPack) ||
        (hasGaussians && gaussians == nullptr) ||
        gaussian_count >
            std::numeric_limits<std::uint32_t>::max() ||
        asset_count == 0u ||
        visual_asset_index >= asset_count ||
        semantic_id == 0u ||
        semantic_id == MR_INVALID_INDEX ||
        instance_id == 0u ||
        instance_id == MR_INVALID_INDEX ||
        capacity == 0u || width == 0u || height == 0u ||
        retain_observation_buffers > 1u) {
        gLastError =
            "V3 visual scene, bindings, dimensions, and capacity "
            "must be valid and nonempty.";
        return nullptr;
    }
    MRHybridRendererHandle* result = nullptr;
    const int status = translateErrors([&] {
        std::vector<MRHybridGaussianGPU> copiedGaussians(
            gaussian_count
        );
        for (std::size_t index = 0u;
             index < gaussian_count;
             ++index) {
            const MRHybridGaussianC& source = gaussians[index];
            MRHybridGaussianGPU& destination =
                copiedGaussians[index];
            std::memcpy(
                &destination.meanAndOpacity,
                source.mean_and_opacity,
                sizeof(source.mean_and_opacity)
            );
            std::memcpy(
                &destination.scaleAndImportance,
                source.scale_and_importance,
                sizeof(source.scale_and_importance)
            );
            std::memcpy(
                &destination.orientation,
                source.orientation,
                sizeof(source.orientation)
            );
            std::memcpy(
                &destination.colorAndEmission,
                source.color_and_emission,
                sizeof(source.color_and_emission)
            );
            std::memcpy(
                &destination.binding,
                source.binding,
                sizeof(source.binding)
            );
        }
        metalrobo::VisualRenderSceneV3 scene;
        scene.id = "c_api_visual_scene_v3";
        scene.assetCount = asset_count;
        scene.bodyCount = body_count;
        scene.gaussians = std::move(copiedGaussians);
        scene.environment =
            metalrobo::makeNeutralStudioEnvironmentV2();
        scene.lightRig =
            metalrobo::makeStudioKeyLightRigV1();
        if (hasPack) {
            metalrobo::VisualAssetPackV2 pack;
            std::string reason;
            if (!metalrobo::readVisualAssetPackIndex(
                    visual_pack_path,
                    pack,
                    &reason
                )) {
                throw std::runtime_error(
                    "could not load V3 visual pack: " + reason
                );
            }
            scene.visualPacks.push_back({
                visual_pack_path,
                pack.contentHash,
                visual_asset_index,
                semantic_id,
                instance_id,
            });
        }
        if (hasEnvironmentPack) {
            metalrobo::VisualEnvironmentPackV2 environmentPack;
            std::string reason;
            if (!metalrobo::readVisualEnvironmentPackIndex(
                    environment_pack_path,
                    environmentPack,
                    &reason
                )) {
                throw std::runtime_error(
                    "could not load V3 environment pack: " + reason
                );
            }
            scene.environment.id = environmentPack.id;
            scene.environment.packPath = environment_pack_path;
            scene.environment.contentHash =
                environmentPack.contentHash;
        }

        const std::string selectedLightRig =
            light_rig == nullptr || light_rig[0] == '\0'
            ? "studio_key"
            : light_rig;
        if (selectedLightRig == "indoor_area") {
            scene.lightRig =
                metalrobo::makeIndoorAreaLightRigV1();
        } else if (selectedLightRig != "studio_key") {
            throw std::runtime_error(
                "unsupported V3 light rig: " + selectedLightRig
            );
        }

        const std::string selectedProfile =
            renderer_profile == nullptr ||
                renderer_profile[0] == '\0'
            ? "sensor_fast"
            : renderer_profile;
        metalrobo::VisualRendererProfileV1 profile;
        if (selectedProfile == "sensor_fast") {
            profile =
                metalrobo::VisualRendererProfileV1::sensorFast();
        } else if (selectedProfile == "sensor_reference") {
            profile =
                metalrobo::VisualRendererProfileV1::
                    sensorReference();
        } else {
            throw std::runtime_error(
                "unsupported V3 renderer profile: " +
                selectedProfile
            );
        }
        if (profile.rayQueryVisibility && body_count == 0u) {
            throw std::runtime_error(
                "sensor_reference requires a nonzero body_count"
            );
        }
        scene.fingerprint =
            metalrobo::computeVisualRenderSceneV3Fingerprint(scene);

        metalrobo::MetalHybridRendererConfig config;
        config.width = width;
        config.height = height;
        config.retainObservationBuffers =
            retain_observation_buffers != 0u;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRHybridRendererHandle>();
        handle->renderer = metalrobo::MetalHybridRenderer{
            std::move(config)
        };
        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.compile(
                std::move(scene),
                profile,
                capacity
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"V2 visual renderer compile failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->deviceName = diagnostics.deviceName;
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_hybrid_renderer_retain(MRHybridRendererHandle* handle) {
    if (handle != nullptr) {
        handle->references.fetch_add(1u, std::memory_order_relaxed);
    }
}

void mr_hybrid_renderer_destroy(MRHybridRendererHandle* handle) {
    if (handle != nullptr &&
        handle->references.fetch_sub(
            1u,
            std::memory_order_acq_rel
        ) == 1u) {
        delete handle;
    }
}

int mr_hybrid_renderer_render(
    MRHybridRendererHandle* handle,
    const MRWorldFamilyHandle* worlds,
    const uint32_t environment_count,
    const uint32_t camera_index
) {
    if (!requireHybridRendererHandle(handle) ||
        !requireWorldFamilyHandle(worlds)) {
        return -1;
    }
    return translateErrors([&] {
        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.render(
                worlds->context,
                environment_count,
                camera_index
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"hybrid render failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->activeEnvironmentCount = environment_count;
        handle->lastRenderMilliseconds =
            diagnostics.elapsedMilliseconds;
    });
}

int mr_hybrid_renderer_encode_graph(
    MRHybridRendererHandle* handle,
    const MRWorldFamilyHandle* worlds,
    const void* current_body_states,
    const void* previous_body_states,
    const size_t current_body_offset,
    const size_t previous_body_offset,
    const uint32_t environment_count,
    const uint32_t body_count,
    const uint64_t frame_index,
    const uint32_t sensor_sequence,
    const uint32_t camera_index,
    const MRMetalComputeEncoderCallbacksC* encoder,
    const MRHybridObservationBuffersC* outputs
) {
    if (!requireHybridRendererHandle(handle) ||
        !requireWorldFamilyHandle(worlds) ||
        current_body_states == nullptr ||
        encoder == nullptr ||
        outputs == nullptr) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::HybridDeviceStateBatch live;
        live.currentBodyStates =
            const_cast<void*>(current_body_states);
        live.previousBodyStates =
            const_cast<void*>(previous_body_states);
        live.currentBodyOffset = current_body_offset;
        live.previousBodyOffset = previous_body_offset;
        live.environmentCount = environment_count;
        live.bodyCount = body_count;
        live.frameIndex = frame_index;
        live.sensorSequence = sensor_sequence;
        live.source = MR_VISUAL_SOURCE_SIMULATION;

        metalrobo::MetalHybridComputeEncoderCallbacks callbacks;
        callbacks.context = encoder->context;
        callbacks.setLabel = encoder->set_label;
        callbacks.useHeap = encoder->use_heap;
        callbacks.useResidencySet =
            encoder->use_residency_set;
        callbacks.setPipeline = encoder->set_pipeline;
        callbacks.setBuffer = encoder->set_buffer;
        callbacks.setBytes = encoder->set_bytes;
        callbacks.dispatchThreads = encoder->dispatch_threads;
        callbacks.dispatchThreadgroups =
            encoder->dispatch_threadgroups;
        callbacks.dispatchThreadgroupsIndirect =
            encoder->dispatch_threadgroups_indirect;

        metalrobo::HybridDeviceObservationBuffers destination;
        destination.rgb = outputs->rgb;
        destination.depth = outputs->depth;
        destination.segmentation = outputs->segmentation;
        destination.identities = outputs->identities;
        destination.normals = outputs->normals;
        destination.motion = outputs->motion;
        destination.validity = outputs->validity;
        destination.outputMask = outputs->output_mask;

        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.encodeGraph(
                worlds->context,
                live,
                camera_index,
                callbacks,
                destination
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"hybrid graph encode failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->activeEnvironmentCount = environment_count;
    });
}

int mr_hybrid_renderer_readback(
    MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::HybridObservationBatch candidate;
        const metalrobo::MetalHybridRendererDiagnostics diagnostics =
            handle->renderer.readback(candidate);
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"hybrid readback failed ["} +
                metalrobo::metalHybridRendererStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->readback = std::move(candidate);
    });
}

MRHybridRendererLayoutC mr_hybrid_renderer_layout(
    const MRHybridRendererHandle* handle
) {
    MRHybridRendererLayoutC result{};
    if (!requireHybridRendererHandle(handle)) {
        return result;
    }
    const metalrobo::MetalHybridRendererLayout layout =
        handle->renderer.layout();
    result.capacity = layout.capacity;
    result.active_environment_count =
        handle->activeEnvironmentCount;
    result.width = layout.width;
    result.height = layout.height;
    result.tile_count_x = layout.tileCountX;
    result.tile_count_y = layout.tileCountY;
    result.gaussian_count = layout.gaussianCount;
    result.maximum_gaussians_per_tile =
        layout.maximumGaussiansPerTile;
    result.maximum_mesh_triangles_per_tile =
        layout.maximumMeshTrianglesPerTile;
    result.mesh_vertex_count = layout.meshVertexCount;
    result.mesh_triangle_count = layout.meshTriangleCount;
    result.mesh_cluster_count = layout.meshClusterCount;
    result.mesh_primitive_count = layout.meshPrimitiveCount;
    result.mesh_instance_count = layout.meshInstanceCount;
    result.mesh_index_count = layout.meshIndexCount;
    result.material_count = layout.materialCount;
    result.texture_count = layout.textureCount;
    result.light_count = layout.lightCount;
    result.body_count = layout.bodyCount;
    result.sensor_binding_count = layout.sensorBindingCount;
    result.shadow_layer_capacity = layout.shadowLayerCapacity;
    result.ray_instance_count = layout.rayInstanceCount;
    result.shadow_workspace_bytes = layout.shadowWorkspaceBytes;
    result.acceleration_structure_bytes =
        layout.accelerationStructureBytes;
    result.retained_private_bytes = layout.retainedPrivateBytes;
    result.last_render_milliseconds =
        handle->lastRenderMilliseconds;
    return result;
}

const char* mr_hybrid_renderer_device_name(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle)) {
        return "";
    }
    return handle->deviceName.c_str();
}

void* mr_hybrid_renderer_native_buffer(
    const MRHybridRendererHandle* handle,
    const uint32_t buffer_kind
) {
    if (!requireHybridRendererHandle(handle) ||
        buffer_kind >
            static_cast<std::uint32_t>(
                metalrobo::MetalHybridRendererBuffer::
                    validity
            )) {
        if (handle != nullptr) {
            gLastError =
                "hybrid renderer buffer kind is invalid.";
        }
        return nullptr;
    }
    return handle->renderer.nativeBuffer(
        static_cast<metalrobo::MetalHybridRendererBuffer>(
            buffer_kind
        )
    );
}

const float* mr_hybrid_renderer_rgb(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.rgb.empty()) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(
        handle->readback.rgb.data()
    );
}

const float* mr_hybrid_renderer_depth(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.depth.empty()) {
        return nullptr;
    }
    return handle->readback.depth.data();
}

const uint32_t* mr_hybrid_renderer_segmentation(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.segmentation.empty()) {
        return nullptr;
    }
    return handle->readback.segmentation.data();
}

const uint32_t* mr_hybrid_renderer_identities(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.identities.empty()) {
        return nullptr;
    }
    return reinterpret_cast<const uint32_t*>(
        handle->readback.identities.data()
    );
}

const float* mr_hybrid_renderer_normals(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.normals.empty()) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(
        handle->readback.normals.data()
    );
}

const float* mr_hybrid_renderer_motion(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.motion.empty()) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(
        handle->readback.motion.data()
    );
}

const uint32_t* mr_hybrid_renderer_validity(
    const MRHybridRendererHandle* handle
) {
    if (!requireHybridRendererHandle(handle) ||
        handle->readback.validity.empty()) {
        return nullptr;
    }
    return handle->readback.validity.data();
}

MRVisualFrameMetadataC mr_hybrid_renderer_frame_metadata(
    const MRHybridRendererHandle* handle
) {
    MRVisualFrameMetadataC result{};
    if (!requireHybridRendererHandle(handle)) {
        return result;
    }
    std::memcpy(
        &result,
        &handle->readback.metadata,
        sizeof(result)
    );
    return result;
}

MRTactileHandle* mr_tactile_create_world_pack(
    const char* world_pack_path,
    const uint32_t capacity,
    const uint32_t contact_capacity_per_environment,
    const char* metallib_path
) {
    if (world_pack_path == nullptr ||
        world_pack_path[0] == '\0' ||
        capacity == 0u ||
        contact_capacity_per_environment == 0u) {
        gLastError =
            "authored tactile world path and capacities are required.";
        return nullptr;
    }
    MRTactileHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::MRWorldPack pack;
        const auto loaded = metalrobo::readWorldPack(
            world_pack_path,
            pack
        );
        if (!loaded.succeeded()) {
            throw std::runtime_error(
                std::string{"authored tactile world load failed ["} +
                metalrobo::worldPackStatusName(loaded.status) +
                "]: " + loaded.message
            );
        }
        const metalrobo::WorldTemplate& world =
            pack.family.worldTemplate;
        std::string reason;
        if (!world.valid(&reason)) {
            throw std::runtime_error(
                "authored tactile world is incomplete: " + reason
            );
        }
        if (world.tactileSystem.sensors.empty()) {
            throw std::runtime_error(
                "authored world pack contains no tactile sensors"
            );
        }
        metalrobo::MetalTactileConfig config;
        config.contactCapacityPerEnvironment =
            contact_capacity_per_environment;
        if (metallib_path != nullptr) {
            config.metallibPath = metallib_path;
        }
        auto handle = std::make_unique<MRTactileHandle>();
        handle->context = metalrobo::MetalTactileContext{
            std::move(config)
        };
        const auto diagnostics = handle->context.compile(
            world.tactileSystem,
            world.engineModel,
            capacity
        );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"authored tactile Metal compile failed ["} +
                metalrobo::metalTactileStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->deviceName = diagnostics.deviceName;
        handle->observationMetadataJSON =
            metalrobo::tactileObservationMetadataJSON(
                world.tactileSystem
            );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_tactile_destroy(MRTactileHandle* handle) {
    delete handle;
}

int mr_tactile_encode(
    MRTactileHandle* handle,
    void* body_states,
    void* contacts,
    void* contact_counts,
    void* reset_mask,
    const uint32_t environment_count,
    const uint32_t body_count,
    const uint32_t contact_capacity_per_environment,
    const float observation_timestep_seconds,
    const float contact_impulse_timestep_seconds,
    const uint64_t frame_index,
    const double timestamp_seconds,
    void* metal_compute_command_encoder
) {
    if (!requireTactileHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::MetalTactileDeviceFrame frame;
        frame.bodyStates = body_states;
        frame.contacts = contacts;
        frame.contactCounts = contact_counts;
        frame.resetMask = reset_mask;
        frame.environmentCount = environment_count;
        frame.bodyCount = body_count;
        frame.contactCapacityPerEnvironment =
            contact_capacity_per_environment;
        frame.observationTimestepSeconds =
            observation_timestep_seconds;
        frame.contactImpulseTimestepSeconds =
            contact_impulse_timestep_seconds;
        frame.frameIndex = frame_index;
        frame.timestampSeconds = timestamp_seconds;
        const auto diagnostics = handle->context.encode(
            frame,
            metal_compute_command_encoder
        );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"tactile encode failed ["} +
                metalrobo::metalTactileStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->activeEnvironmentCount = environment_count;
        handle->lastObserveMilliseconds =
            diagnostics.elapsedMilliseconds;
    });
}

int mr_tactile_readback(MRTactileHandle* handle) {
    if (!requireTactileHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        metalrobo::TactileObservationBatch candidate;
        const auto diagnostics = handle->context.readback(
            handle->activeEnvironmentCount,
            candidate
        );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"tactile readback failed ["} +
                metalrobo::metalTactileStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
        handle->readback = std::move(candidate);
    });
}

MRTactileLayoutC mr_tactile_layout(
    const MRTactileHandle* handle
) {
    MRTactileLayoutC result{};
    if (!requireTactileHandle(handle)) {
        return result;
    }
    const metalrobo::MetalTactileLayout layout =
        handle->context.layout();
    result.capacity = layout.environmentCapacity;
    result.active_environment_count =
        handle->activeEnvironmentCount;
    result.body_count = layout.bodyCount;
    result.shape_count = layout.shapeCount;
    result.sensor_count = layout.sensorCount;
    result.sample_count = layout.sampleCount;
    result.target_count = layout.targetCount;
    result.contact_capacity_per_environment =
        layout.contactCapacityPerEnvironment;
    result.query_backend =
        static_cast<uint32_t>(layout.queryBackend);
    result.hardware_ray_queries_available =
        layout.hardwareRayQueriesAvailable ? 1u : 0u;
    result.retained_bytes = layout.retainedBytes;
    result.bytes_per_environment = layout.bytesPerEnvironment;
    result.last_observe_milliseconds =
        handle->lastObserveMilliseconds;
    return result;
}

const char* mr_tactile_device_name(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle)
        ? handle->deviceName.c_str()
        : "";
}

const char* mr_tactile_observation_metadata_json(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle)
        ? handle->observationMetadataJSON.c_str()
        : "";
}

void* mr_tactile_native_buffer(
    const MRTactileHandle* handle,
    const uint32_t buffer_kind
) {
    if (!requireTactileHandle(handle) ||
        buffer_kind >
            static_cast<uint32_t>(
                metalrobo::MetalTactileBuffer::tangentialMotion
            )) {
        if (handle != nullptr) {
            gLastError = "tactile buffer kind is invalid.";
        }
        return nullptr;
    }
    return handle->context.nativeBuffer(
        static_cast<metalrobo::MetalTactileBuffer>(buffer_kind)
    );
}

const float* mr_tactile_depth(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle) &&
        !handle->readback.penetrationDepthMeters.empty()
        ? handle->readback.penetrationDepthMeters.data()
        : nullptr;
}

const float* mr_tactile_depth_velocity(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle) &&
        !handle->readback.depthVelocityMetersPerSecond.empty()
        ? handle->readback.depthVelocityMetersPerSecond.data()
        : nullptr;
}

const float* mr_tactile_tangential_motion(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle) &&
        !handle->readback.tangentialMotion.empty()
        ? reinterpret_cast<const float*>(
            handle->readback.tangentialMotion.data()
        )
        : nullptr;
}

const uint32_t* mr_tactile_validity(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle) &&
        !handle->readback.validity.empty()
        ? handle->readback.validity.data()
        : nullptr;
}

const uint32_t* mr_tactile_object_shape_ids(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle) &&
        !handle->readback.objectShapeIds.empty()
        ? handle->readback.objectShapeIds.data()
        : nullptr;
}

const MRTactileSummaryC* mr_tactile_summaries(
    const MRTactileHandle* handle
) {
    return requireTactileHandle(handle) &&
        !handle->readback.summaries.empty()
        ? reinterpret_cast<const MRTactileSummaryC*>(
            handle->readback.summaries.data()
        )
        : nullptr;
}

} // extern "C"
