#include "metalrobo/c_api.h"

#include "metalrobo/EpisodeTwinCompiler.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalTactile.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/Model.hpp"
#include "metalrobo/Runtime.hpp"
#include "metalrobo/RuntimeAbi.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
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
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct MRRuntimeHandle {
    std::unique_ptr<metalrobo::Runtime> runtime;
    std::string deviceName;
};

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

struct MRTaskRolloutHandle {
    explicit MRTaskRolloutHandle(
        metalrobo::MetalWorldConfig config
    ) : context(std::move(config)) {}

    metalrobo::EngineModel model;
    metalrobo::CompiledWorld world;
    metalrobo::CompiledTaskProgram taskProgram;
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
    std::string deviceName;
    std::uint32_t environmentCount = 0u;
    std::uint64_t submittedControlSteps = 0u;
    std::uint64_t completedEnvironmentSteps = 0u;
    std::uint64_t submissionCount = 0u;
    double totalGPUMilliseconds = 0.0;
    double totalSubmissionMilliseconds = 0.0;
};

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
    offsetof(MRTaskTransitionC, curriculum_level) ==
    offsetof(MRTaskTransitionGPU, taskProgress)
);

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

bool requireHandle(const MRRuntimeHandle* handle) {
    if (handle != nullptr && handle->runtime != nullptr) {
        return true;
    }
    gLastError = "MetalRobo runtime handle is null.";
    return false;
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
    if (config.solver != MR_TASK_ROLLOUT_SOLVER_PGS &&
        config.solver != MR_TASK_ROLLOUT_SOLVER_TGS) {
        throw std::invalid_argument(
            "task-rollout solver is invalid"
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

std::unique_ptr<MRTaskRolloutHandle>
createCompiledTaskRollout(
    metalrobo::LocomotionWorld authored,
    const MRTaskRolloutConfigC& config,
    const char* metallibPath,
    const std::string_view source
) {
    validateTaskRolloutConfiguration(config);

    metalrobo::MetalWorldConfig worldConfig;
    if (metallibPath != nullptr &&
        metallibPath[0] != '\0') {
        worldConfig.metallibPath = metallibPath;
    }
    worldConfig.maximumInFlightSubmissions = 1u;
    auto handle =
        std::make_unique<MRTaskRolloutHandle>(
            std::move(worldConfig)
        );

    metalrobo::CompiledLocomotionWorld compiled;
    const metalrobo::LocomotionWorldCompileDiagnostics
        compiledStatus = metalrobo::compileLocomotionWorld(
            authored,
            authored.articulationIndex,
            compiled
        );
    if (!compiledStatus.world.succeeded()) {
        throw std::runtime_error(
            std::string{source} +
            " Metal world compile failed [" +
            metalrobo::metalWorldHostStatusName(
                compiledStatus.world.status
            ) + "]: " + compiledStatus.world.message
        );
    }
    if (!compiledStatus.task.succeeded()) {
        throw std::runtime_error(
            std::string{source} +
            " TaskPack compile failed [" +
            metalrobo::taskCompileStatusName(
                compiledStatus.task.status
            ) + "]: " + compiledStatus.task.element +
            ": " + compiledStatus.task.message
        );
    }

    handle->model = std::move(authored.model);
    handle->defaultSceneBodies =
        std::move(authored.sceneBodies);
    handle->world = std::move(compiled.world);
    handle->taskProgram = std::move(compiled.task);
    if (handle->world.sceneBodyCount() !=
            handle->defaultSceneBodies.size()) {
        throw std::runtime_error(
            std::string{source} +
            " scene-state count does not match compiled topology"
        );
    }

    handle->environmentCount = config.environment_count;
    handle->stepConfig.timestepSeconds =
        config.control_timestep_seconds;
    handle->stepConfig.physicsSubsteps =
        config.physics_substeps;
    handle->stepConfig.solverMode =
        config.solver == MR_TASK_ROLLOUT_SOLVER_TGS
        ? metalrobo::MetalWorldSolverMode::throughputTGS
        : metalrobo::MetalWorldSolverMode::throughputPGS;
    handle->stepConfig.actuationMode =
        metalrobo::MetalWorldActuationMode::
            implicitPositionDrive;
    handle->stepConfig.velocityIterations =
        config.velocity_iterations;
    handle->stepConfig.finalVelocityIterations =
        config.final_velocity_iterations;
    handle->stepConfig.ccdMode =
        metalrobo::MetalWorldCCDMode::disabled;
    handle->stepConfig.applyBodyDamping = true;
    handle->stepConfig.deterministic = true;
    handle->stepConfig.warmStart = true;
    handle->stepConfig.captureContactEvidence = false;
    handle->stepConfig.publishFinalState = false;
    handle->stepConfig.publishStateTrajectory = false;
    handle->stepConfig.taskProgram = handle->taskProgram;
    resetTaskRolloutState(*handle, config.seed);
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

MRRuntimeHandle* mr_create_franka(
    const uint32_t environment_count,
    const uint64_t seed,
    const char* metallib_path
) {
    if (environment_count == 0) {
        gLastError = "environment_count must be greater than zero.";
        return nullptr;
    }

    MRRuntimeHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::RuntimeDescriptor descriptor;
        descriptor.environmentCount = environment_count;
        descriptor.seed = seed;
        descriptor.autoReset = true;
        descriptor.captureBodyPoses = true;
        if (metallib_path != nullptr) {
            descriptor.metallibPath = metallib_path;
        }

        auto handle = std::make_unique<MRRuntimeHandle>();
        handle->runtime = metalrobo::makeMetalRuntime(
            metalrobo::makeFrankaPandaModel(),
            descriptor
        );
        handle->deviceName = handle->runtime->deviceName();
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

void mr_destroy(MRRuntimeHandle* handle) {
    delete handle;
}

int mr_reset(MRRuntimeHandle* handle, const uint64_t seed) {
    if (!requireHandle(handle)) {
        return -1;
    }
    return translateErrors([&] { handle->runtime->reset(seed); });
}

int mr_step(
    MRRuntimeHandle* handle,
    const float* normalized_actions,
    const size_t action_count
) {
    if (!requireHandle(handle)) {
        return -1;
    }
    const std::size_t required =
        static_cast<std::size_t>(handle->runtime->environmentCount()) *
        handle->runtime->model().gpu.actionCount;
    if (normalized_actions == nullptr) {
        gLastError = "normalized_actions is null.";
        return -1;
    }
    if (action_count != required) {
        gLastError =
            "action_count does not match environment_count * action_count.";
        return -1;
    }
    return translateErrors([&] {
        handle->runtime->step(
            std::span<const float>(normalized_actions, action_count)
        );
    });
}

uint32_t mr_environment_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle) ? handle->runtime->environmentCount() : 0;
}

uint32_t mr_action_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle) ? handle->runtime->model().gpu.actionCount : 0;
}

uint32_t mr_observation_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle)
        ? handle->runtime->model().gpu.observationCount
        : 0;
}

uint32_t mr_link_count(const MRRuntimeHandle* handle) {
    return requireHandle(handle) ? handle->runtime->model().gpu.linkCount : 0;
}

const float* mr_observations(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->observations().data();
}

const float* mr_rewards(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->rewards().data();
}

const uint8_t* mr_terminated(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->terminated().data();
}

const float* mr_body_positions(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->bodyPositions().data();
}

const float* mr_body_rotations(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return nullptr;
    }
    return handle->runtime->bodyRotations().data();
}

MRRuntimeStatsC mr_stats(const MRRuntimeHandle* handle) {
    MRRuntimeStatsC result{};
    if (!requireHandle(handle)) {
        return result;
    }
    const metalrobo::RuntimeStats stats = handle->runtime->stats();
    result.last_gpu_milliseconds = stats.lastGpuMilliseconds;
    result.total_gpu_milliseconds = stats.totalGpuMilliseconds;
    result.control_steps = stats.controlSteps;
    result.physics_steps = stats.physicsSteps;
    return result;
}

const char* mr_device_name(const MRRuntimeHandle* handle) {
    if (!requireHandle(handle)) {
        return "";
    }
    return handle->deviceName.c_str();
}

MRTaskRolloutHandle* mr_create_unitree_g1_locomotion_rollout(
    const MRTaskRolloutConfigC* config,
    const uint32_t surface_value,
    const char* metallib_path
) {
    if (config == nullptr) {
        gLastError = "task-rollout config is null.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        const metalrobo::LocomotionSurface surface =
            locomotionSurface(surface_value);
        auto handle =
            createCompiledTaskRollout(
                metalrobo::makeUnitreeG1LocomotionWorld(
                    surface
                ),
                *config,
                metallib_path,
                "bundled G1"
            );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

MRTaskRolloutHandle* mr_create_urdf_locomotion_rollout(
    const char* urdf_path,
    const char* srdf_path,
    const char* task_pack_path,
    const MRTaskRolloutConfigC* config,
    const uint32_t surface_value,
    const char* metallib_path
) {
    if (config == nullptr ||
        urdf_path == nullptr || urdf_path[0] == '\0' ||
        task_pack_path == nullptr ||
        task_pack_path[0] == '\0') {
        gLastError =
            "URDF, TaskPack, and task-rollout config are required.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        const metalrobo::LocomotionSurface surface =
            locomotionSurface(surface_value);
        metalrobo::TaskPack task;
        const metalrobo::LearningPackResult loaded =
            metalrobo::readTaskPack(task_pack_path, task);
        if (!loaded.succeeded()) {
            throw std::invalid_argument(
                std::string{"TaskPack load failed ["} +
                metalrobo::learningPackStatusName(
                    loaded.status
                ) + "]: " + loaded.message
            );
        }

        metalrobo::RobotDescriptionCookOptions options;
        options.rootMode =
            metalrobo::RobotDescriptionRootMode::floating;
        options.meshMode =
            metalrobo::RobotDescriptionMeshMode::convexHull;
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
        authored.task = std::move(task);
        metalrobo::appendLocomotionSurface(
            authored.model,
            authored.sceneBodies,
            surface
        );
        auto handle = createCompiledTaskRollout(
            std::move(authored),
            *config,
            metallib_path,
            "imported URDF"
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
}

MRTaskRolloutHandle* mr_create_world_pack_locomotion_rollout(
    const char* world_pack_path,
    const char* task_pack_path,
    const MRTaskRolloutConfigC* config,
    const char* metallib_path
) {
    if (config == nullptr ||
        world_pack_path == nullptr ||
        world_pack_path[0] == '\0' ||
        task_pack_path == nullptr ||
        task_pack_path[0] == '\0') {
        gLastError =
            "MRWorldPack, TaskPack, and task-rollout config are required.";
        return nullptr;
    }

    MRTaskRolloutHandle* result = nullptr;
    const int status = translateErrors([&] {
        metalrobo::TaskPack task;
        const metalrobo::LearningPackResult taskLoaded =
            metalrobo::readTaskPack(task_pack_path, task);
        if (!taskLoaded.succeeded()) {
            throw std::invalid_argument(
                std::string{"TaskPack load failed ["} +
                metalrobo::learningPackStatusName(
                    taskLoaded.status
                ) + "]: " + taskLoaded.message
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
        auto handle = createCompiledTaskRollout(
            metalrobo::makeWorldPackLocomotionWorld(
                worldPack,
                std::move(task)
            ),
            *config,
            metallib_path,
            "MRWorldPack"
        );
        result = handle.release();
    });
    return status == 0 ? result : nullptr;
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

int mr_task_rollout_set_curriculum_level(
    MRTaskRolloutHandle* handle,
    const uint32_t level
) {
    if (!requireTaskRolloutHandle(handle)) {
        return -1;
    }
    return translateErrors([&] {
        if (handle->residentState.valid()) {
            throw std::logic_error(
                "task curriculum must be restored before resident initialization"
            );
        }
        if (level >= handle->taskProgram.header().schedule.z) {
            throw std::invalid_argument(
                "task curriculum level exceeds the compiled TaskPack"
            );
        }
        handle->stepConfig.taskCurriculumLevel = level;
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
        installPolicyPack(
            *handle,
            policyPackFromC(*policy)
        );
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
        handle->deviceName = diagnostics.deviceName;
        handle->submittedControlSteps += control_step_count;
        handle->completedEnvironmentSteps +=
            diagnostics.successfulStepCount;
        ++handle->submissionCount;
        handle->totalGPUMilliseconds +=
            diagnostics.gpuElapsedMilliseconds;
        handle->totalSubmissionMilliseconds +=
            diagnostics.submissionElapsedMilliseconds;
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"task-rollout GPU step failed ["} +
                metalrobo::metalWorldHostStatusName(
                    diagnostics.status
                ) + "]: " + diagnostics.message
            );
        }
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

uint64_t mr_task_rollout_task_fingerprint(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle)
        ? handle->taskProgram.fingerprint()
        : 0u;
}

const char* mr_task_rollout_device_name(
    const MRTaskRolloutHandle* handle
) {
    return requireTaskRolloutHandle(handle)
        ? handle->deviceName.c_str()
        : "";
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
        std::vector<MRTaskTransitionGPU> transitions(
            batch->transition_count
        );
        if (!transitions.empty()) {
            std::memcpy(
                transitions.data(),
                batch->transitions,
                transitions.size() *
                    sizeof(MRTaskTransitionGPU)
            );
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
