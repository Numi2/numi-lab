#ifndef NUMI_BRAIN_HUMAN_STANDING_ABI_H
#define NUMI_BRAIN_HUMAN_STANDING_ABI_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define NB_HUMAN_STANDING_ABI_VERSION 1u
typedef struct NBHumanStandingReceptor {
    uint32_t modality;
    uint32_t receptor_count;
    uint32_t feature_dimension;
    uint32_t reserved0;
    uint64_t receptor_timestamp_microseconds;
    void* values;
    void* validity;
} NBHumanStandingReceptor;
typedef struct NBHumanStandingInfo {
    uint64_t model_source_fingerprint;
    uint64_t locomotor_program_fingerprint;
    uint64_t baseline_program_fingerprint;
    uint64_t compiled_species_fingerprint;
    uint64_t sensory_profile_fingerprint;
    uint64_t parameter_version_fingerprint;
    uint64_t committed_generation;
    uint64_t last_joint_commit_fingerprint;
} NBHumanStandingInfo;
/* All Metal objects are unretained borrowed Objective-C handles. The native
   owner submits and waits. Source JSON comes from its immutable anatomy and
   prepared calibration. Optional program JSON must match that exact source.
   These functions never own, advance, or rewrite physical state. */
void* nb_human_standing_create_v1(void* device, const char* source_json,
    const char* program_json, const char* output_directory,
    uint32_t timestep_microseconds, uint64_t epoch_microseconds, uint32_t seed,
    char* error, size_t error_capacity);
uint32_t nb_human_standing_encode_motor_v1(void* handle, void* encoder,
    uint32_t step_index, const NBHumanStandingReceptor* receptors, uint32_t receptor_count,
    void* muscle_states, uint32_t muscle_count, char* error, size_t error_capacity);
/* Optional stage split on the same native command buffer for GPU phase timing. */
uint32_t nb_human_standing_encode_motor_decision_v1(void* handle, void* encoder,
    uint32_t step_index, const NBHumanStandingReceptor* receptors, uint32_t receptor_count,
    char* error, size_t error_capacity);
uint32_t nb_human_standing_encode_motor_tissue_v1(void* handle, void* encoder,
    void* muscle_states, uint32_t muscle_count, char* error, size_t error_capacity);
/* Optional inner GPU phase timing. The Brain borrows this exact owner command
   buffer and encodes successive passes without submitting or waiting. */
uint32_t nb_human_standing_encode_motor_tissue_phased_v1(void* handle,
    void* command_buffer, void* muscle_states, uint32_t muscle_count,
    char* error, size_t error_capacity);
uint32_t nb_human_standing_encode_accepted_v1(void* handle, void* encoder,
    uint64_t physical_state_fingerprint, uint32_t completed_step_count,
    const NBHumanStandingReceptor* receptors, uint32_t receptor_count,
    char* error, size_t error_capacity);
/* Optional split of accepted fast systems and cognition on one native command. */
uint32_t nb_human_standing_encode_accepted_fast_v1(void* handle, void* encoder,
    uint64_t physical_state_fingerprint, uint32_t completed_step_count,
    char* error, size_t error_capacity);
uint32_t nb_human_standing_encode_accepted_cognitive_v1(void* handle, void* encoder,
    const NBHumanStandingReceptor* receptors, uint32_t receptor_count,
    char* error, size_t error_capacity);
uint32_t nb_human_standing_publish_v1(void* handle, double gpu_start_seconds,
    double gpu_end_seconds, char* error, size_t error_capacity);
uint32_t nb_human_standing_abort_v1(void* handle, char* error, size_t error_capacity);
uint32_t nb_human_standing_info_v1(void* handle, NBHumanStandingInfo* info);
void nb_human_standing_destroy_v1(void* handle);
#ifdef __cplusplus
}
static_assert(sizeof(NBHumanStandingReceptor) == 40u);
static_assert(sizeof(NBHumanStandingInfo) == 64u);
#endif
#endif
