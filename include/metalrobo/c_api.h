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

typedef struct MRRuntimeHandle MRRuntimeHandle;
typedef struct MRWorldFamilyHandle MRWorldFamilyHandle;
typedef struct MRHybridRendererHandle MRHybridRendererHandle;
typedef struct MRWorldInstanceHeaderGPU MRWorldInstanceHeaderGPU;
typedef struct MRWorldAssetInstanceGPU MRWorldAssetInstanceGPU;
typedef struct MRWorldSensorInstanceGPU MRWorldSensorInstanceGPU;
typedef struct MRWorldAppearanceInstanceGPU
    MRWorldAppearanceInstanceGPU;
typedef struct MRWorldScenarioHeaderGPU MRWorldScenarioHeaderGPU;
typedef struct MRWorldScenarioValueGPU MRWorldScenarioValueGPU;

typedef struct MRRuntimeStatsC {
    double last_gpu_milliseconds;
    double total_gpu_milliseconds;
    uint64_t control_steps;
    uint64_t physics_steps;
} MRRuntimeStatsC;

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
    uint32_t mesh_vertex_count;
    uint32_t mesh_triangle_count;
    uint32_t material_count;
    uint32_t body_count;
    uint32_t sensor_binding_count;
    size_t retained_private_bytes;
    double last_render_milliseconds;
} MRHybridRendererLayoutC;

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

MR_API const char* mr_version(void);
MR_API const char* mr_last_error(void);

// Compile an Apple-native capture manifest into a portable MRWorldPack.
// artifact_store_path may be null to place the CAS beside the output pack.
MR_API int mr_compile_episode_manifest(
    const char* manifest_path,
    const char* output_pack_path,
    const char* artifact_store_path
);

MR_API MRRuntimeHandle* mr_create_franka(
    uint32_t environment_count,
    uint64_t seed,
    const char* metallib_path
);
MR_API void mr_destroy(MRRuntimeHandle* handle);

MR_API int mr_reset(MRRuntimeHandle* handle, uint64_t seed);
MR_API int mr_step(
    MRRuntimeHandle* handle,
    const float* normalized_actions,
    size_t action_count
);

MR_API uint32_t mr_environment_count(const MRRuntimeHandle* handle);
MR_API uint32_t mr_action_count(const MRRuntimeHandle* handle);
MR_API uint32_t mr_observation_count(const MRRuntimeHandle* handle);
MR_API uint32_t mr_link_count(const MRRuntimeHandle* handle);

// Returned spans alias shared simulator memory and remain valid until destroy.
MR_API const float* mr_observations(const MRRuntimeHandle* handle);
MR_API const float* mr_rewards(const MRRuntimeHandle* handle);
MR_API const uint8_t* mr_terminated(const MRRuntimeHandle* handle);
MR_API const float* mr_body_positions(const MRRuntimeHandle* handle);
MR_API const float* mr_body_rotations(const MRRuntimeHandle* handle);

MR_API MRRuntimeStatsC mr_stats(const MRRuntimeHandle* handle);
MR_API const char* mr_device_name(const MRRuntimeHandle* handle);

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

MR_API MRHybridRendererHandle* mr_hybrid_renderer_create(
    const MRHybridGaussianC* gaussians,
    size_t gaussian_count,
    uint32_t asset_count,
    uint32_t capacity,
    uint32_t width,
    uint32_t height,
    const char* metallib_path
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

#ifdef __cplusplus
}
#endif
