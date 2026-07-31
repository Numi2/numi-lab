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
typedef struct MRSimulationHandle MRSimulationHandle;
typedef struct MRWorldInstanceHeaderGPU MRWorldInstanceHeaderGPU;
typedef struct MRWorldAssetInstanceGPU MRWorldAssetInstanceGPU;
typedef struct MRWorldSensorInstanceGPU MRWorldSensorInstanceGPU;
typedef struct MRWorldAppearanceInstanceGPU
    MRWorldAppearanceInstanceGPU;
typedef struct MRWorldScenarioHeaderGPU MRWorldScenarioHeaderGPU;
typedef struct MRWorldScenarioValueGPU MRWorldScenarioValueGPU;

// Convenience environment used only by bundled/imported robot constructors.
// WorldPack simulations author their complete scene instead.
typedef enum MRBuiltinSurfaceC {
    MR_BUILTIN_SURFACE_GROUND = 0,
    MR_BUILTIN_SURFACE_TERRAIN = 1,
} MRBuiltinSurfaceC;

typedef enum MRG1ActuatorPresetC {
    MR_G1_ACTUATOR_UNITREE_URDF_REV_1_0 = 0,
    MR_G1_ACTUATOR_UNITREE_MJCF_REV_1_0 = 1,
    MR_G1_ACTUATOR_UNITREE_RL_LAB_4960B84 = 2,
} MRG1ActuatorPresetC;

typedef enum MRNumiIterationPolicyC {
    MR_NUMI_ITERATION_FIXED_BUDGET = 0,
    MR_NUMI_ITERATION_RESIDUAL_CONVERGED = 1,
} MRNumiIterationPolicyC;

typedef enum MRRobotRootModeC {
    MR_ROBOT_ROOT_FIXED = 0,
    MR_ROBOT_ROOT_FLOATING = 1,
} MRRobotRootModeC;

typedef struct MRSimulationConfigC {
    uint32_t environment_count;
    uint32_t numi_iteration_policy;
    uint32_t physics_substeps;
    uint32_t temporal_substeps;
    float control_timestep_seconds;
    uint64_t seed;
} MRSimulationConfigC;

typedef struct MRSimulationLayoutC {
    uint32_t environment_count;
    uint32_t nq;
    uint32_t nv;
    uint32_t action_count;
    uint32_t actor_observation_count;
    uint32_t critic_observation_count;
    uint32_t scene_body_count;
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
} MRSimulationLayoutC;

typedef struct MRSimulationStageHighWaterC {
    uint32_t candidate_pairs;
    uint32_t raw_contacts;
    uint32_t manifolds;
    uint32_t constraint_blocks;
    uint32_t constraint_rows;
    uint32_t islands;
    uint32_t hard_convex_pairs;
    uint32_t mesh_triangle_candidates;
    uint32_t ccd_candidates;
    uint32_t ccd_events;
    uint32_t endpoint_runtime_records;
    uint32_t articulation_point_queries;
    uint32_t rod_candidate_pairs;
    uint32_t rod_raw_contacts;
    uint32_t rod_manifolds;
    uint32_t rod_ccd_events;
    uint32_t numi_generalized_velocities;
    uint32_t numi_rows;
    uint32_t numi_krylov_vectors;
    uint32_t numi_direct_tiles;
    uint32_t dynamic_nodes;
    uint32_t island_node_references;
    uint32_t island_constraint_references;
    uint32_t rod_factor_blocks;
    uint32_t operator_velocity_elements;
} MRSimulationStageHighWaterC;

typedef struct MRSimulationAdvanceC {
    uint32_t control_step_count;
    uint32_t successful_environment_steps;
    uint32_t failed_environment_steps;
    uint32_t first_failing_environment;
    uint32_t first_failing_control_step;
    uint32_t first_gpu_status_code;
    uint32_t host_requested_resets;
    uint32_t maximum_active_contacts;
    uint32_t maximum_manifolds;
    MRSimulationStageHighWaterC high_water;
    double gpu_milliseconds;
    double submission_milliseconds;
} MRSimulationAdvanceC;

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
    uint64_t policy_revision;
    float timeout_bootstrap_value;
    float episode_tracking_score;
    uint32_t curriculum_level;
    uint32_t terrain_level;
    uint32_t reserved[2];
} MRTaskTransitionC;

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

// Creates the bundled G1 mechanics and locomotion TaskPack through the same
// compiled task-program route used by imported robots. The explicit actuator
// preset participates in compiled world/task fingerprints. The returned
// executor is robot-independent: the caller owns rollout chunking and supplies packed
// normalized [step][environment][compiled action] values plus an optional
// [step][environment] reset mask. One advance call submits and waits for
// exactly one native Metal command buffer.
MR_API MRSimulationHandle*
mr_create_unitree_g1_simulation(
    const MRSimulationConfigC* config,
    uint32_t actuator_preset,
    uint32_t surface,
    const char* metallib_path
);
// Cooks a fixed- or floating-base URDF/SRDF, loads its authored TaskPack,
// resolves every semantic binding, and creates the same generic native
// executor used by bundled simulations.
// srdf_path and metallib_path may be null; all other pointers are required.
MR_API MRSimulationHandle* mr_create_urdf_simulation(
    const char* urdf_path,
    const char* srdf_path,
    const char* task_pack_path,
    const MRSimulationConfigC* config,
    uint32_t root_mode,
    uint32_t surface,
    const char* metallib_path
);
// Loads complete mechanics/scene composition from MRWorldPack and resolves a
// separate TaskPack against it. The authored pack, not a runtime preset,
// owns terrain and other scene bodies.
MR_API MRSimulationHandle* mr_create_world_pack_simulation(
    const char* world_pack_path,
    const char* task_pack_path,
    const MRSimulationConfigC* config,
    const char* metallib_path
);
MR_API void mr_simulation_destroy(MRSimulationHandle* handle);
MR_API int mr_simulation_reset(
    MRSimulationHandle* handle,
    uint64_t seed
);
// Restores the compact task-wide curriculum checkpoint before the first
// resident submission. Per-environment simulator state remains native.
MR_API int mr_simulation_set_curriculum_level(
    MRSimulationHandle* handle,
    uint32_t level
);
// Installs one immutable compiled policy artifact. The call copies all
// caller-owned spans; subsequent advances take no action stream and run
// inference between native observation construction and action application.
MR_API int mr_simulation_set_policy(
    MRSimulationHandle* handle,
    const MRPolicyPackC* policy
);
// Loads, validates, compiles, and atomically installs a persisted PolicyPack.
// A failed load leaves the currently installed policy unchanged.
MR_API int mr_simulation_load_policy_pack(
    MRSimulationHandle* handle,
    const char* policy_pack_path
);
MR_API int mr_simulation_clear_policy(
    MRSimulationHandle* handle
);
MR_API int mr_simulation_advance(
    MRSimulationHandle* handle,
    const float* normalized_actions,
    size_t normalized_action_count,
    const uint32_t* reset_masks,
    size_t reset_mask_count,
    uint32_t control_step_count,
    uint64_t policy_revision,
    uint32_t evaluate_final_policy,
    MRSimulationAdvanceC* advance
);
MR_API MRSimulationLayoutC mr_simulation_layout(
    const MRSimulationHandle* handle
);
// Stable compiled TaskPack fingerprint used by learner checkpoints to reject
// semantically incompatible resume attempts.
MR_API uint64_t mr_simulation_task_fingerprint(
    const MRSimulationHandle* handle
);
MR_API const char* mr_simulation_device_name(
    const MRSimulationHandle* handle
);
// Diagnostic status spans alias handle-owned publication memory until the
// next advance, reset, or destroy. Simulator state is never exposed here.
MR_API const uint32_t* mr_simulation_status_codes(
    const MRSimulationHandle* handle
);
MR_API const uint32_t* mr_simulation_active_contacts(
    const MRSimulationHandle* handle
);
MR_API const float* mr_simulation_actor_observations(
    const MRSimulationHandle* handle
);
MR_API const float* mr_simulation_critic_observations(
    const MRSimulationHandle* handle
);
MR_API const MRTaskTransitionC* mr_simulation_transitions(
    const MRSimulationHandle* handle
);
MR_API const float* mr_simulation_policy_latents(
    const MRSimulationHandle* handle
);
MR_API const float* mr_simulation_policy_log_probabilities(
    const MRSimulationHandle* handle
);
MR_API const float* mr_simulation_policy_values(
    const MRSimulationHandle* handle
);
// Value estimates for the accepted post-rollout state. Native policy
// evaluation is encoded in the same submission without advancing physics.
MR_API const float* mr_simulation_bootstrap_policy_values(
    const MRSimulationHandle* handle
);
// Transactionally publishes an aggregated Swift-owned collection boundary.
// Dimensions and fingerprints are derived from the installed task/policy;
// caller spans are copied before return.
MR_API int mr_simulation_write_policy_rollout_pack(
    const MRSimulationHandle* handle,
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
