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
typedef struct MRWorldInstanceHeaderGPU MRWorldInstanceHeaderGPU;
typedef struct MRWorldAssetInstanceGPU MRWorldAssetInstanceGPU;
typedef struct MRWorldSensorInstanceGPU MRWorldSensorInstanceGPU;
typedef struct MRWorldAppearanceInstanceGPU
    MRWorldAppearanceInstanceGPU;

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
    size_t retained_private_bytes;
} MRWorldFamilyLayoutC;

typedef struct MRWorldFamilyStatsC {
    uint64_t compile_count;
    uint64_t sample_count;
    uint64_t readback_count;
    double last_sample_milliseconds;
} MRWorldFamilyStatsC;

MR_API const char* mr_version(void);
MR_API const char* mr_last_error(void);

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
MR_API void mr_world_family_destroy(MRWorldFamilyHandle* handle);
MR_API int mr_world_family_sample(
    MRWorldFamilyHandle* handle,
    uint32_t instance_count,
    uint64_t seed
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
// buffer_kind: 0 headers, 1 assets, 2 sensors, 3 appearances. The returned
// value is a borrowed id<MTLBuffer> and is intended for native extensions.
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

#ifdef __cplusplus
}
#endif
