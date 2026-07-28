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

typedef struct MRRuntimeStatsC {
    double last_gpu_milliseconds;
    double total_gpu_milliseconds;
    uint64_t control_steps;
    uint64_t physics_steps;
} MRRuntimeStatsC;

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

#ifdef __cplusplus
}
#endif
