#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(METALROBO_BUILDING_LIBRARY)
#define MR_API __declspec(dllexport)
#else
#define MR_API __declspec(dllimport)
#endif
#else
#define MR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MRWorldFamilyHandle MRWorldFamilyHandle;
typedef struct MRHybridRendererHandle MRHybridRendererHandle;
typedef struct MRTactileHandle MRTactileHandle;
typedef struct MRTaskRolloutHandle MRTaskRolloutHandle;
typedef struct MRWorldInstanceHeaderGPU MRWorldInstanceHeaderGPU;
typedef struct MRWorldAssetInstanceGPU MRWorldAssetInstanceGPU;
typedef struct MRWorldSensorInstanceGPU MRWorldSensorInstanceGPU;
typedef struct MRWorldAppearanceInstanceGPU
    MRWorldAppearanceInstanceGPU;
typedef struct MRWorldScenarioHeaderGPU MRWorldScenarioHeaderGPU;
typedef struct MRWorldScenarioValueGPU MRWorldScenarioValueGPU;

typedef enum MRLocomotionSurfaceC {
    MR_LOCOMOTION_SURFACE_GROUND = 0,
    MR_LOCOMOTION_SURFACE_TERRAIN = 1,
} MRLocomotionSurfaceC;

typedef enum MRUnitreeG1TaskC {
    MR_UNITREE_G1_TASK_VELOCITY = 0,
    MR_UNITREE_G1_TASK_DISTURBANCE_RECOVERY = 1,
    MR_UNITREE_G1_TASK_SUPINE_GET_UP_DISCOVERY = 2,
    MR_UNITREE_G1_TASK_BALL_DISTURBANCE_RECOVERY = 3,
    MR_UNITREE_G1_TASK_BALL_DODGE = 4,
} MRUnitreeG1TaskC;

typedef enum MRInteractionReferenceModeC {
    MR_INTERACTION_REFERENCE_TASK_DEFAULT = 0,
    MR_INTERACTION_REFERENCE_GUIDE = 1,
    MR_INTERACTION_REFERENCE_RESET_ONLY = 2,
} MRInteractionReferenceModeC;

typedef struct MRTaskRolloutDynamicSphereC {
    float position[3];
    float linear_velocity[3];
    float radius;
    float mass;
    uint32_t launch_step;
} MRTaskRolloutDynamicSphereC;

typedef struct MRTaskRolloutConfigC {
    uint32_t environment_count;
    uint32_t physics_substeps;
    uint32_t velocity_iterations;
    uint32_t final_velocity_iterations;
    float control_timestep_seconds;
    uint64_t seed;
    const MRTaskRolloutDynamicSphereC* dynamic_spheres;
    uint32_t dynamic_sphere_count;
    uint32_t disable_task_terminations;
    uint32_t interaction_reference_mode;
    float interaction_student_authority;
    uint32_t override_interaction_student_authority;
    float interaction_reset_phase_fraction;
    uint32_t override_interaction_reset_phase_fraction;
    // Diagnostic/correctness path: materialize inverse-ABA response columns
    // before contact solve instead of consuming the direct streamed path.
    uint32_t materialize_articulated_contact_responses;
    // Optional inclusive reset-band range within the compiled TaskPack.
    uint32_t override_difficulty_band_range;
    uint32_t minimum_difficulty_band;
    uint32_t maximum_difficulty_band;
    float interaction_reset_phase_probability;
    uint32_t override_interaction_reset_phase_probability;
    float interaction_reset_maximum_phase;
    uint32_t override_interaction_reset_maximum_phase;
} MRTaskRolloutConfigC;

typedef struct MRTaskVisualPackC {
    const char* path;
    // Authored WorldTemplate asset identity. Multiple packs may bind the
    // same asset (for example, one pack per articulated robot link).
    const char* asset_id;
    uint32_t semantic_id;
    uint32_t instance_id;
} MRTaskVisualPackC;

typedef struct MRTaskVisualObservationConfigC {
    const MRTaskVisualPackC* packs;
    uint32_t pack_count;
    const char* environment_pack_path;
    const char* renderer_profile;
    const char* camera_parent_body;
    float camera_position[3];
    float camera_orientation[4];
    uint32_t width;
    uint32_t height;
    uint32_t minimum_visible_pixels;
    // Zero retains the legacy focal-length rule. A positive value authors
    // the physical vertical field of view in degrees.
    float vertical_field_of_view_degrees;
    // Authored sensor cadence. Physics and inference retain their independent
    // control rate and hold the latest private observation between samples.
    float nominal_rate_hz;
    // Zero selects the renderer default. Large batched training callers may
    // raise this explicit preflight bound after accounting for their device.
    uint64_t maximum_retained_bytes;
    // Optional presentation capture. Zero keeps the training-only sensor
    // path and allocates no second renderer.
    uint32_t capture_width;
    uint32_t capture_height;
    // Capture a high-resolution camera with the policy sensor's authored
    // parent and pose instead of the external presentation camera.
    uint32_t capture_policy_camera;
} MRTaskVisualObservationConfigC;

typedef struct MRTaskRolloutLayoutC {
    uint32_t environment_count;
    uint32_t nq;
    uint32_t nv;
    uint32_t action_count;
    uint32_t actor_observation_count;
    uint32_t critic_observation_count;
    uint32_t scene_body_count;
    uint32_t motion_feature_count;
    uint32_t maximum_episode_steps;
    uint64_t world_fingerprint;
    uint64_t task_fingerprint;
    uint64_t observation_fingerprint;
    uint64_t action_fingerprint;
    uint64_t run_fingerprint;
    uint64_t robot_fingerprint;
    uint64_t sensor_fingerprint;
    uint64_t reality_fingerprint;
    uint64_t teacher_fingerprint;
    uint64_t submitted_control_steps;
    uint64_t completed_environment_steps;
    uint64_t submission_count;
    size_t retained_buffer_bytes;
    size_t immutable_private_bytes;
    size_t persistent_state_private_bytes;
    size_t transient_private_bytes;
    size_t shared_boundary_bytes;
    size_t peak_aliased_bytes;
    double total_gpu_milliseconds;
    double total_submission_milliseconds;
} MRTaskRolloutLayoutC;

enum MRRunManifestSourceC {
    MR_RUN_SOURCE_UNITREE_G1 = 0u,
    MR_RUN_SOURCE_FRANKA_PICK_PLACE = 1u,
    MR_RUN_SOURCE_IMPORTED_URDF = 2u,
    MR_RUN_SOURCE_WORLD_PACK = 3u,
};

// Single native construction boundary for training, evaluation and
// deployment. Source paths are compile inputs; after return the executor owns
// one immutable CompiledRun and never dispatches by source kind again.
typedef struct MRRunManifestC {
    MRTaskRolloutConfigC profile;
    uint32_t source;
    uint32_t surface;
    uint32_t task;
    const char* urdf_path;
    const char* srdf_path;
    const char* world_pack_path;
    const char* task_pack_path;
    const char* teacher_pack_path;
    const char* teacher_clip_id;
    const MRTaskVisualObservationConfigC* visual_sensor_program;
    const char* metallib_path;
} MRRunManifestC;

typedef struct MRTaskRolloutStageHighWaterC {
    uint32_t candidate_pairs;
    uint32_t raw_contacts;
    uint32_t manifolds;
    uint32_t constraint_blocks;
    uint32_t constraint_rows;
    uint32_t islands;
    uint32_t hard_convex_pairs;
    uint32_t mesh_triangle_candidates;
    uint32_t solver_tiles;
    uint32_t spill_rows;
    uint32_t ccd_candidates;
    uint32_t ccd_events;
    uint32_t endpoint_runtime_records;
    uint32_t articulation_point_queries;
    uint32_t rod_candidate_pairs;
    uint32_t rod_raw_contacts;
    uint32_t rod_manifolds;
    uint32_t rod_ccd_events;
    uint32_t quality_generalized_velocities;
    uint32_t quality_rows;
    uint32_t quality_krylov_vectors;
    uint32_t quality_direct_tiles;
    uint32_t dynamic_nodes;
    uint32_t island_node_references;
    uint32_t island_constraint_references;
    uint32_t rod_factor_blocks;
    uint32_t operator_velocity_elements;
} MRTaskRolloutStageHighWaterC;

typedef struct MRTaskRolloutAdvanceC {
    uint32_t control_step_count;
    uint32_t successful_environment_steps;
    uint32_t failed_environment_steps;
    uint32_t first_failing_environment;
    uint32_t first_failing_control_step;
    uint32_t first_gpu_status_code;
    uint32_t host_requested_resets;
    uint32_t maximum_active_contacts;
    uint32_t maximum_manifolds;
    MRTaskRolloutStageHighWaterC high_water;
    double gpu_milliseconds;
    double submission_milliseconds;
} MRTaskRolloutAdvanceC;

typedef struct MRTaskEvidenceTelemetryC {
    uint64_t control_steps;
    uint64_t evidence_windows;
    uint32_t pending_completed_episode_count;
    uint32_t pending_timeout_episode_count;
    uint32_t last_completed_episode_count;
    uint32_t last_contact_rate;
    uint32_t last_clean_miss_rate;
    uint32_t last_balance_failure_rate;
    uint32_t last_mean_tracking_per_million;
} MRTaskEvidenceTelemetryC;

typedef struct MRTaskTransitionC {
    float reward;
    float tracking_score;
    float root_height;
    float tilt;
    uint32_t done;
    uint32_t timeout;
    uint32_t physics_error;
    uint32_t termination_reason;
    float task_reward;
    float base_reward;
    float joint_velocity_reward;
    float joint_acceleration_reward;
    float control_reward;
    float posture_reward;
    float energy_reward;
    float contact_reward;
    float task_outcome_channel_0;
    float task_outcome_channel_1;
    float task_outcome_channel_2;
    float task_outcome_channel_3;
    float task_outcome_channel_4;
    float task_outcome_channel_5;
    float task_outcome_channel_6;
    float task_outcome_channel_7;
    uint64_t policy_revision;
    float timeout_bootstrap_value;
    float episode_tracking_score;
    uint32_t difficulty_band;
    uint32_t terrain_level;
    // One-based authored impact-sequence index, or zero outside an event.
    uint32_t impact_sequence_index;
    // MR_TASK_IMPACT_* transition flags.
    uint32_t impact_event_flags;
} MRTaskTransitionC;

// Task-dependent typed outcome view. MRTaskTransitionC remains an internal
// executor publication boundary for existing Swift aggregation; persisted
// learner artifacts use MRLearningTransitionGPU plus this typed schema.
typedef enum MRTaskOutcomeDirectionC {
    MR_TASK_OUTCOME_NEUTRAL = 0,
    MR_TASK_OUTCOME_HIGHER_IS_BETTER = 1,
    MR_TASK_OUTCOME_LOWER_IS_BETTER = 2,
} MRTaskOutcomeDirectionC;

typedef enum MRTaskOutcomeSourceC {
    MR_TASK_OUTCOME_REWARD = 0,
    MR_TASK_OUTCOME_TASK_REWARD = 1,
    MR_TASK_OUTCOME_TRACKING_SCORE = 2,
    MR_TASK_OUTCOME_ROOT_HEIGHT = 3,
    MR_TASK_OUTCOME_TILT = 4,
    MR_TASK_OUTCOME_DONE = 5,
    MR_TASK_OUTCOME_TIMEOUT = 6,
    MR_TASK_OUTCOME_PHYSICS_ERROR = 7,
    MR_TASK_OUTCOME_CONTACT_REWARD = 8,
    MR_TASK_OUTCOME_CHANNEL_0 = 9,
    MR_TASK_OUTCOME_CHANNEL_1 = 10,
    MR_TASK_OUTCOME_CHANNEL_2 = 11,
    MR_TASK_OUTCOME_CHANNEL_3 = 12,
    MR_TASK_OUTCOME_CHANNEL_4 = 13,
    MR_TASK_OUTCOME_CHANNEL_5 = 14,
    MR_TASK_OUTCOME_CHANNEL_6 = 15,
    MR_TASK_OUTCOME_CHANNEL_7 = 16,
} MRTaskOutcomeSourceC;

typedef enum MRPolicyActivationC {
    MR_POLICY_ACTIVATION_C_IDENTITY = 0,
    MR_POLICY_ACTIVATION_C_RELU = 1,
    MR_POLICY_ACTIVATION_C_TANH = 2,
    MR_POLICY_ACTIVATION_C_ELU = 3,
    MR_POLICY_ACTIVATION_C_SILU = 4,
} MRPolicyActivationC;

typedef struct MRPolicyDenseLayerC {
    uint32_t input_count;
    uint32_t output_count;
    uint32_t activation;
    const float* weights;
    size_t weight_count;
    const float* bias;
    size_t bias_count;
} MRPolicyDenseLayerC;

typedef struct MRPolicyPackC {
    const char* id;
    uint64_t revision;
    uint64_t contract_version;
    uint64_t world_fingerprint;
    uint64_t task_fingerprint;
    uint64_t observation_fingerprint;
    uint64_t action_fingerprint;
    const float* observation_mean;
    size_t observation_mean_count;
    const float* observation_inverse_standard_deviation;
    size_t observation_inverse_standard_deviation_count;
    const MRPolicyDenseLayerC* layers;
    size_t layer_count;
    const float* critic_observation_mean;
    size_t critic_observation_mean_count;
    const float* critic_observation_inverse_standard_deviation;
    size_t critic_observation_inverse_standard_deviation_count;
    const MRPolicyDenseLayerC* critic_layers;
    size_t critic_layer_count;
    const float* action_log_standard_deviation;
    size_t action_log_standard_deviation_count;
    const float* action_bias;
    size_t action_bias_count;
    const float* action_scale;
    size_t action_scale_count;
    float observation_clip;
    float action_clip;
} MRPolicyPackC;

typedef struct MRPolicyRolloutBatchC {
    uint32_t control_step_count;
    const float* actor_observations;
    size_t actor_observation_count;
    const float* critic_observations;
    size_t critic_observation_count;
    const float* motion_features;
    size_t motion_feature_count;
    const float* teacher_actions;
    size_t teacher_action_count;
    const float* latents;
    size_t latent_count;
    const float* log_probabilities;
    size_t log_probability_count;
    const float* values;
    size_t value_count;
    const float* bootstrap_values;
    size_t bootstrap_value_count;
    const MRTaskTransitionC* transitions;
    size_t transition_count;
} MRPolicyRolloutBatchC;

typedef struct MRWorldFamilyLayoutC {
    uint32_t capacity;
    uint32_t active_instance_count;
    uint32_t asset_count_per_instance;
    uint32_t sensor_count_per_instance;
    uint32_t appearance_count_per_instance;
    uint32_t variation_count;
    uint32_t categorical_value_count;
    uint32_t asset_binding_count;
    uint32_t binding_index_count;
    uint32_t primary_articulation_index;
    uint32_t nq;
    uint32_t nv;
    uint32_t body_count;
    uint32_t scene_body_count;
    uint32_t articulation_count;
    size_t retained_private_bytes;
} MRWorldFamilyLayoutC;

typedef struct MRWorldFamilyStatsC {
    uint64_t compile_count;
    uint64_t sample_count;
    uint64_t readback_count;
    double last_sample_milliseconds;
} MRWorldFamilyStatsC;

typedef struct MRScenarioFeatureC {
    uint32_t axis;
    uint32_t distribution;
    uint32_t target;
    uint32_t ordinal;
    float parameters[4];
} MRScenarioFeatureC;

typedef struct MRHybridRendererLayoutC {
    uint32_t capacity;
    uint32_t active_environment_count;
    uint32_t width;
    uint32_t height;
    uint32_t tile_count_x;
    uint32_t tile_count_y;
    uint32_t gaussian_count;
    uint32_t maximum_gaussians_per_tile;
    uint32_t maximum_mesh_triangles_per_tile;
    uint32_t mesh_vertex_count;
    uint32_t mesh_triangle_count;
    uint32_t mesh_cluster_count;
    uint32_t mesh_primitive_count;
    uint32_t mesh_instance_count;
    uint32_t mesh_index_count;
    uint32_t material_count;
    uint32_t texture_count;
    uint32_t light_count;
    uint32_t body_count;
    uint32_t sensor_binding_count;
    uint32_t shadow_layer_capacity;
    uint32_t ray_instance_count;
    size_t shadow_workspace_bytes;
    size_t acceleration_structure_bytes;
    size_t retained_private_bytes;
    double last_render_milliseconds;
} MRHybridRendererLayoutC;

typedef struct MRMetalComputeEncoderCallbacksC {
    void* context;
    void (*set_label)(void* context, const char* label);
    void (*use_heap)(void* context, void* heap);
    void (*use_residency_set)(
        void* context,
        void* residency_set
    );
    void (*set_pipeline)(void* context, void* pipeline);
    void (*set_buffer)(
        void* context,
        void* buffer,
        size_t offset,
        uint32_t index
    );
    void (*set_bytes)(
        void* context,
        const void* bytes,
        size_t length,
        uint32_t index
    );
    void (*dispatch_threads)(
        void* context,
        size_t thread_count,
        size_t threads_per_threadgroup
    );
    void (*dispatch_threadgroups)(
        void* context,
        size_t threadgroup_count,
        size_t threads_per_threadgroup
    );
    void (*dispatch_threadgroups_indirect)(
        void* context,
        void* arguments,
        size_t offset,
        size_t threads_per_threadgroup
    );
} MRMetalComputeEncoderCallbacksC;

typedef struct MRHybridObservationBuffersC {
    void* rgb;
    void* depth;
    void* segmentation;
    void* identities;
    void* normals;
    void* motion;
    void* validity;
    uint32_t output_mask;
} MRHybridObservationBuffersC;

typedef struct MRVisualFrameMetadataC {
    uint32_t dimensions[4];
    uint32_t identity[4];
    float timing[4];
    uint32_t contract[4];
} MRVisualFrameMetadataC;

typedef struct MRHybridGaussianC {
    float mean_and_opacity[4];
    float scale_and_importance[4];
    float orientation[4];
    float color_and_emission[4];
    uint32_t binding[4];
} MRHybridGaussianC;

typedef struct MRTactileLayoutC {
    uint32_t capacity;
    uint32_t active_environment_count;
    uint32_t body_count;
    uint32_t shape_count;
    uint32_t sensor_count;
    uint32_t sample_count;
    uint32_t target_count;
    uint32_t contact_capacity_per_environment;
    uint32_t query_backend;
    uint32_t hardware_ray_queries_available;
    size_t retained_bytes;
    size_t bytes_per_environment;
    double last_observe_milliseconds;
} MRTactileLayoutC;

typedef struct MRTactileSummaryC {
    float pose_position_and_timestamp[4];
    float pose_orientation[4];
    float net_force_and_contact_area[4];
    float net_torque_and_maximum_depth[4];
    float centroid_local_and_mean_depth[4];
    float centroid_world_and_active_count[4];
    float center_of_pressure_local_and_force_weight[4];
    float center_of_pressure_world_and_contact_count[4];
    float tangential_motion_and_friction[4];
    uint32_t statistics_and_identity[4];
} MRTactileSummaryC;

MR_API const char* mr_version(void);
// Host/Metal record fingerprint for native-extension compatibility checks.
// A consumer must reject a mismatch before submitting GPU work.
MR_API uint64_t mr_runtime_abi_fingerprint(void);
MR_API const char* mr_last_error(void);
// Process-lifetime JSON contract for the bundled Unitree G1 mechanics/task
// preset. Deployment comparators use this instead of duplicating joint order,
// nominal drives, limits, or action scaling in another language.
MR_API const char* mr_unitree_g1_deployment_contract_json(void);
// Writes the same deterministic PolicyPack artifact consumed by Swift and
// native Metal rollout. All caller-owned spans are copied before return.
MR_API int mr_write_policy_pack(
    const MRPolicyPackC* policy,
    const char* policy_pack_path
);
// Hashes one already-mapped learning-pack payload with the exact native wire
// algorithm. Null is valid only when byte_count is zero.
MR_API uint64_t mr_learning_pack_content_hash(
    const void* payload,
    size_t byte_count
);

// Compile an Apple-native capture manifest into a portable MRWorldPack.
// artifact_store_path may be null to place the CAS beside the output pack.
MR_API int mr_compile_episode_manifest(
    const char* manifest_path,
    const char* output_pack_path,
    const char* artifact_store_path
);

MR_API MRTaskRolloutHandle* mr_create_task_rollout(
    const MRRunManifestC* manifest
);
MR_API void mr_task_rollout_destroy(MRTaskRolloutHandle* handle);
MR_API int mr_task_rollout_reset(
    MRTaskRolloutHandle* handle,
    uint64_t seed
);
// Returns the task-wide state published by the most recent successful native
// submission. This fixed record is independent of environment count.
MR_API int mr_task_rollout_evidence_telemetry(
    const MRTaskRolloutHandle* handle,
    MRTaskEvidenceTelemetryC* telemetry
);
// Installs one immutable compiled policy artifact. The call copies all
// caller-owned spans; subsequent advances take no action stream and run
// inference between native observation construction and action application.
MR_API int mr_task_rollout_set_policy(
    MRTaskRolloutHandle* handle,
    const MRPolicyPackC* policy
);
// Loads, validates, compiles, and atomically installs a persisted PolicyPack.
// A failed load leaves the currently installed policy unchanged.
MR_API int mr_task_rollout_load_policy_pack(
    MRTaskRolloutHandle* handle,
    const char* policy_pack_path
);
MR_API int mr_task_rollout_clear_policy(
    MRTaskRolloutHandle* handle
);
// Copies the most recently completed native visual-observation frame as
// environment-major linear RGBA floats. Returns the required float count;
// pass a null destination to query it without copying.
MR_API size_t mr_task_rollout_copy_visual_rgba(
    MRTaskRolloutHandle* handle,
    float* destination,
    size_t destination_count,
    uint32_t* width,
    uint32_t* height
);
// Opt-in inspection/export readback. Disabled by default so training keeps
// simulator state device-resident. When enabled, final_q aliases the last
// accepted post-step state until the next advance, reset, or destroy.
MR_API int mr_task_rollout_set_state_readback(
    MRTaskRolloutHandle* handle,
    uint32_t enabled
);
MR_API int mr_task_rollout_advance(
    MRTaskRolloutHandle* handle,
    const float* normalized_actions,
    size_t normalized_action_count,
    const uint32_t* reset_masks,
    size_t reset_mask_count,
    uint32_t control_step_count,
    uint64_t policy_revision,
    uint32_t evaluate_final_policy,
    MRTaskRolloutAdvanceC* advance
);
MR_API MRTaskRolloutLayoutC mr_task_rollout_layout(
    const MRTaskRolloutHandle* handle
);
MR_API const char* mr_task_rollout_task_id(
    const MRTaskRolloutHandle* handle
);
MR_API const char* mr_task_rollout_device_name(
    const MRTaskRolloutHandle* handle
);
// Returns zero when no Visual Presentation scene is attached.
MR_API uint64_t mr_task_rollout_visual_scene_fingerprint(
    const MRTaskRolloutHandle* handle
);
MR_API uint32_t mr_task_rollout_impact_event_count(
    const MRTaskRolloutHandle* handle
);
// Diagnostic status spans alias handle-owned publication memory until the
// next advance, reset, or destroy. Simulator state is never exposed here.
MR_API const uint32_t* mr_task_rollout_status_codes(
    const MRTaskRolloutHandle* handle
);
MR_API const uint32_t* mr_task_rollout_active_contacts(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_actor_observations(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_critic_observations(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_motion_features(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_teacher_actions(
    const MRTaskRolloutHandle* handle
);
MR_API const MRTaskTransitionC* mr_task_rollout_transitions(
    const MRTaskRolloutHandle* handle
);
MR_API uint32_t mr_task_rollout_outcome_count(
    const MRTaskRolloutHandle* handle
);
MR_API const char* mr_task_rollout_outcome_id(
    const MRTaskRolloutHandle* handle,
    uint32_t outcome_index
);
MR_API const char* mr_task_rollout_outcome_unit(
    const MRTaskRolloutHandle* handle,
    uint32_t outcome_index
);
MR_API uint32_t mr_task_rollout_outcome_direction(
    const MRTaskRolloutHandle* handle,
    uint32_t outcome_index
);
// Latest submission values packed [transition][outcome].
MR_API const float* mr_task_rollout_outcome_values(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_policy_latents(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_policy_log_probabilities(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_policy_values(
    const MRTaskRolloutHandle* handle
);
// Value estimates for the accepted post-rollout state. Native policy
// evaluation is encoded in the same submission without advancing physics.
MR_API const float* mr_task_rollout_bootstrap_policy_values(
    const MRTaskRolloutHandle* handle
);
MR_API const float* mr_task_rollout_final_q(
    const MRTaskRolloutHandle* handle
);
// Copies accepted final scene-body states as packed
// [environment][scene body][position xyz, orientation xyzw,
// linear velocity xyz, angular velocity xyz].
MR_API int mr_task_rollout_copy_final_scene_states(
    const MRTaskRolloutHandle* handle,
    float* output,
    size_t output_count
);
// Transactionally publishes an aggregated Swift-owned collection boundary.
// Dimensions and fingerprints are derived from the installed task/policy;
// caller spans are copied before return.
MR_API int mr_task_rollout_write_policy_rollout_pack(
    const MRTaskRolloutHandle* handle,
    const MRPolicyRolloutBatchC* batch,
    const char* batch_id,
    const char* output_path
);

// Canonical first world-family frontend. Sampling writes private Metal buffers
// that may be bound directly by Objective-C++/MLX through
// mr_world_family_native_buffer. Readback is an explicit inspection path.
MR_API MRWorldFamilyHandle*
mr_create_franka_pick_place_world_family(
    uint32_t capacity,
    const char* metallib_path
);
MR_API MRWorldFamilyHandle* mr_load_world_family_pack(
    const char* pack_path,
    uint32_t capacity,
    const char* metallib_path
);
MR_API void mr_world_family_retain(MRWorldFamilyHandle* handle);
MR_API void mr_world_family_destroy(MRWorldFamilyHandle* handle);
MR_API int mr_world_family_sample(
    MRWorldFamilyHandle* handle,
    uint32_t instance_count,
    uint64_t seed
);
MR_API int mr_world_family_sample_ex(
    MRWorldFamilyHandle* handle,
    uint32_t instance_count,
    uint64_t seed,
    // MRWorldSamplingMode. Replay maps particle i to environment i.
    uint32_t sampling_mode,
    uint64_t episode_counter
);
// Configures an already content-addressed alignment/feedback sampler.
// particle_quantiles is [particle, feature], region_bounds is
// [region, feature, lower/upper], and weights are normalized internally.
MR_API int mr_world_family_configure_sampling(
    MRWorldFamilyHandle* handle,
    uint64_t alignment_fingerprint,
    const float* particle_quantiles,
    const float* particle_weights,
    const float* particle_residuals,
    uint32_t particle_count,
    uint64_t feedback_fingerprint,
    const uint32_t* region_kinds,
    const float* region_weights,
    const float* region_bounds,
    uint32_t region_count,
    float broad_weight,
    float failure_weight,
    float uncertainty_weight,
    float alignment_jitter
);
MR_API uint64_t mr_world_family_scenario_fingerprint(
    const MRWorldFamilyHandle* handle
);
// Nonzero only for an explicitly loaded MRWorldPack. This is the single
// authored-artifact identity used to reject physics/render graph mismatches.
MR_API uint64_t mr_world_family_authored_pack_hash(
    const MRWorldFamilyHandle* handle
);
MR_API const char* mr_world_family_scenario_id(
    const MRWorldFamilyHandle* handle
);
MR_API const char* mr_world_family_scenario_feature_id(
    const MRWorldFamilyHandle* handle,
    uint32_t feature
);
MR_API const char* mr_world_family_scenario_target_id(
    const MRWorldFamilyHandle* handle,
    uint32_t feature
);
MR_API MRScenarioFeatureC mr_world_family_scenario_feature(
    const MRWorldFamilyHandle* handle,
    uint32_t feature
);
MR_API int mr_world_family_readback(MRWorldFamilyHandle* handle);
MR_API MRWorldFamilyLayoutC mr_world_family_layout(
    const MRWorldFamilyHandle* handle
);
MR_API MRWorldFamilyStatsC mr_world_family_stats(
    const MRWorldFamilyHandle* handle
);
MR_API const char* mr_world_family_device_name(
    const MRWorldFamilyHandle* handle
);
// buffer_kind: 0 headers, 1 assets, 2 sensors, 3 appearances, 4 immutable
// asset bindings, 5 immutable binding-index arena, 6 reset q, 7 reset v,
// 8 scene-body resets, 9 body parameters, 10 controller parameters,
// 11 scenario headers, 12 environment-major scenario values. The returned
// value is a borrowed id<MTLBuffer> for native graph composition.
MR_API void* mr_world_family_native_buffer(
    const MRWorldFamilyHandle* handle,
    uint32_t buffer_kind
);
// Readback pointers remain valid until the next sample or readback call.
MR_API const MRWorldInstanceHeaderGPU*
mr_world_family_instance_headers(const MRWorldFamilyHandle* handle);
MR_API const MRWorldAssetInstanceGPU*
mr_world_family_asset_instances(const MRWorldFamilyHandle* handle);
MR_API const MRWorldSensorInstanceGPU*
mr_world_family_sensor_instances(const MRWorldFamilyHandle* handle);
MR_API const MRWorldAppearanceInstanceGPU*
mr_world_family_appearance_instances(const MRWorldFamilyHandle* handle);
MR_API const MRWorldScenarioHeaderGPU*
mr_world_family_scenario_headers(const MRWorldFamilyHandle* handle);
MR_API const MRWorldScenarioValueGPU*
mr_world_family_scenario_values(const MRWorldFamilyHandle* handle);

// V3 presentation entry point. At least one of gaussians or visual_pack_path
// must be supplied. environment_pack_path accepts a cooked .mrenv pack.
// light_rig accepts "studio_key" or "indoor_area"; renderer_profile accepts
// "sensor_fast" or "sensor_reference".
MR_API MRHybridRendererHandle* mr_hybrid_renderer_create_v3(
    const MRHybridGaussianC* gaussians,
    size_t gaussian_count,
    const char* visual_pack_path,
    const char* environment_pack_path,
    uint32_t asset_count,
    uint32_t body_count,
    uint32_t visual_asset_index,
    uint32_t semantic_id,
    uint32_t instance_id,
    const char* light_rig,
    const char* renderer_profile,
    uint32_t capacity,
    uint32_t width,
    uint32_t height,
    uint32_t retain_observation_buffers,
    const char* metallib_path
);
MR_API void mr_hybrid_renderer_retain(
    MRHybridRendererHandle* handle
);
MR_API void mr_hybrid_renderer_destroy(
    MRHybridRendererHandle* handle
);
MR_API int mr_hybrid_renderer_render(
    MRHybridRendererHandle* handle,
    const MRWorldFamilyHandle* worlds,
    uint32_t environment_count,
    uint32_t camera_index
);
// Direct lazy-graph path. Inputs and outputs borrow MTLBuffer objects and the
// callbacks append to the graph runtime's active compute encoder.
MR_API int mr_hybrid_renderer_encode_graph(
    MRHybridRendererHandle* handle,
    const MRWorldFamilyHandle* worlds,
    const void* current_body_states,
    const void* previous_body_states,
    size_t current_body_offset,
    size_t previous_body_offset,
    uint32_t environment_count,
    uint32_t body_count,
    uint64_t frame_index,
    uint32_t sensor_sequence,
    uint32_t camera_index,
    const MRMetalComputeEncoderCallbacksC* encoder,
    const MRHybridObservationBuffersC* outputs
);
MR_API int mr_hybrid_renderer_readback(
    MRHybridRendererHandle* handle
);
MR_API MRHybridRendererLayoutC mr_hybrid_renderer_layout(
    const MRHybridRendererHandle* handle
);
MR_API const char* mr_hybrid_renderer_device_name(
    const MRHybridRendererHandle* handle
);
// buffer_kind: 0 RGB float4, 1 depth float, 2 semantic uint,
// 3 projected Gaussian records, 4 per-world tile overflow counts,
// 5 semantic/instance/link/primitive uint4, 6 normals float4,
// 7 motion float4, 8 validity uint. Validity bits are: bit 0 frame
// produced, bit 1 usable sensor depth, bit 2 rendered geometry/truth.
// Returned values borrow id<MTLBuffer>.
MR_API void* mr_hybrid_renderer_native_buffer(
    const MRHybridRendererHandle* handle,
    uint32_t buffer_kind
);
// Readback pointers remain valid until the next renderer readback or destroy.
MR_API const float* mr_hybrid_renderer_rgb(
    const MRHybridRendererHandle* handle
);
MR_API const float* mr_hybrid_renderer_depth(
    const MRHybridRendererHandle* handle
);
MR_API const uint32_t* mr_hybrid_renderer_segmentation(
    const MRHybridRendererHandle* handle
);
MR_API const uint32_t* mr_hybrid_renderer_identities(
    const MRHybridRendererHandle* handle
);
MR_API const float* mr_hybrid_renderer_normals(
    const MRHybridRendererHandle* handle
);
MR_API const float* mr_hybrid_renderer_motion(
    const MRHybridRendererHandle* handle
);
MR_API const uint32_t* mr_hybrid_renderer_validity(
    const MRHybridRendererHandle* handle
);
MR_API MRVisualFrameMetadataC mr_hybrid_renderer_frame_metadata(
    const MRHybridRendererHandle* handle
);

// Canonical authored-world tactile frontend. encode() borrows id<MTLBuffer>
// inputs and a live id<MTLComputeCommandEncoder>; it neither commits nor
// waits. Passing null contact buffers produces geometry-only observations.
// The pack must contain explicit cooked tactile sensors.
MR_API MRTactileHandle* mr_tactile_create_world_pack(
    const char* world_pack_path,
    uint32_t capacity,
    uint32_t contact_capacity_per_environment,
    const char* metallib_path
);
MR_API void mr_tactile_destroy(MRTactileHandle* handle);
MR_API int mr_tactile_encode(
    MRTactileHandle* handle,
    void* body_states,
    void* contacts,
    void* contact_counts,
    void* reset_mask,
    uint32_t environment_count,
    uint32_t body_count,
    uint32_t contact_capacity_per_environment,
    float observation_timestep_seconds,
    float contact_impulse_timestep_seconds,
    uint64_t frame_index,
    double timestamp_seconds,
    void* metal_compute_command_encoder
);
MR_API int mr_tactile_readback(MRTactileHandle* handle);
MR_API MRTactileLayoutC mr_tactile_layout(
    const MRTactileHandle* handle
);
MR_API const char* mr_tactile_device_name(
    const MRTactileHandle* handle
);
MR_API const char* mr_tactile_observation_metadata_json(
    const MRTactileHandle* handle
);
// buffer_kind: 0 depth float, 1 depth velocity float, 2 validity uint,
// 3 object shape uint, 4 optional debug hit, 5 summary, 6 status,
// 7 tangential-motion float4. A headless context returns null for kind 4.
MR_API void* mr_tactile_native_buffer(
    const MRTactileHandle* handle,
    uint32_t buffer_kind
);
// Readback pointers remain valid until the next readback or destroy.
MR_API const float* mr_tactile_depth(
    const MRTactileHandle* handle
);
MR_API const float* mr_tactile_depth_velocity(
    const MRTactileHandle* handle
);
MR_API const float* mr_tactile_tangential_motion(
    const MRTactileHandle* handle
);
MR_API const uint32_t* mr_tactile_validity(
    const MRTactileHandle* handle
);
MR_API const uint32_t* mr_tactile_object_shape_ids(
    const MRTactileHandle* handle
);
MR_API const MRTactileSummaryC* mr_tactile_summaries(
    const MRTactileHandle* handle
);

#ifdef __cplusplus
}
#endif
