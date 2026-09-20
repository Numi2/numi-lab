#ifndef METALROBO_MRNX_HUMAN_BEHAVIOR_V1_H
#define METALROBO_MRNX_HUMAN_BEHAVIOR_V1_H
#include "metalrobo/mrnx_bridge_v1.h"
#if defined(_MSC_VER)
#  define MRNX_BEHAVIOR_ALIGN16 __declspec(align(16))
#else
#  define MRNX_BEHAVIOR_ALIGN16 __attribute__((aligned(16)))
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define MRNX_BEHAVIOR_TRACE_ABI_V1 1u
#define MRNX_BEHAVIOR_TRACE_STATUS_DISABLED_V1 0u
#define MRNX_BEHAVIOR_TRACE_STATUS_READY_V1 1u
#define MRNX_BEHAVIOR_TRACE_STATUS_OVERFLOW_V1 2u
#define MRNX_BEHAVIOR_TRACE_STATUS_INVALID_V1 3u
#define MRNX_BEHAVIOR_TRACE_STATUS_FINALIZED_V1 4u
#define MRNX_BEHAVIOR_TRACE_JOINTLY_PUBLISHED_V1 1u
#define MRNX_BEHAVIOR_TRACE_REJECTED_RELEASED_V1 2u
#define MRNX_BEHAVIOR_TRACE_METRIC_UNAVAILABLE_V1 0u
#define MRNX_BEHAVIOR_TRACE_METRIC_ACCEPTED_V1 1u
#define MRNX_BEHAVIOR_TRACE_METRIC_REJECTED_CANDIDATE_V1 2u
#define MRNX_BEHAVIOR_TRACE_TERMINAL_COMPLETED_V1 1u
#define MRNX_BEHAVIOR_TRACE_TERMINAL_CANCELLED_V1 2u
#define MRNX_BEHAVIOR_TRACE_TERMINAL_FAILED_V1 3u
#define MRNX_BEHAVIOR_TRACE_CAPTURE_COMPLETE_V1 1u
#define MRNX_BEHAVIOR_TRACE_CAPTURE_PARTIAL_V1 2u
#define MRNX_BEHAVIOR_TRACE_CAPTURE_FAILED_V1 3u
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_SOURCE_BOUND_METRIC_V1 (1u << 0u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_NATIVE_AUDIT_V1 (1u << 1u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_FORBIDDEN_CONTACT_V1 (1u << 2u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_ACCEPTED_ROOT_PROOF_V1 (1u << 3u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_FULL_BEHAVIOR_V1 (1u << 4u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_PHYSICAL_V1 (1u << 5u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_BIOLOGICAL_V1 (1u << 6u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_PERFORMANCE_V1 (1u << 7u)
#define MRNX_BEHAVIOR_TRACE_EVIDENCE_PRODUCTION_V1 (1u << 8u)

typedef struct mrnx_behavior_trace_config_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t record_capacity;
    uint32_t flags;
    uint64_t expected_accepted_roots;
    uint64_t reserved0;
} mrnx_behavior_trace_config_v1;

// One reducer-validated terminal attempt. Fingerprints are canonical FNV-1a
// identities used by the native transaction ABIs, not cryptographic digests.
// A rejected record's after_* identity must equal its base_* identity.
typedef struct MRNX_BEHAVIOR_ALIGN16 mrnx_behavior_trace_record_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t disposition;
    uint32_t metric_kind;
    uint32_t control_step;
    uint32_t runtime_failure_stage;
    uint32_t candidate_status;
    uint32_t posture_valid;
    uint32_t settled;
    uint32_t audit_covered_mask;
    uint32_t audit_violation_mask;
    uint32_t forbidden_contact_coverage;
    uint32_t forbidden_contact_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint64_t attempt_index;
    uint64_t transaction_fingerprint;
    uint64_t linearization_epoch;
    uint64_t slot_generation;
    uint64_t base_publication_epoch;
    uint64_t base_physics_generation;
    uint64_t base_accepted_timestamp_nanoseconds;
    uint64_t base_accepted_token_fingerprint;
    uint64_t candidate_physics_generation;
    uint64_t candidate_timestamp_nanoseconds;
    uint64_t candidate_state_proof_fingerprint;
    uint64_t candidate_accepted_token_fingerprint;
    uint64_t candidate_publication_fingerprint;
    uint64_t after_publication_epoch;
    uint64_t after_physics_generation;
    uint64_t after_accepted_timestamp_nanoseconds;
    uint64_t after_accepted_token_fingerprint;
    uint64_t after_publication_fingerprint;
    uint64_t joint_fence_fingerprint;
    float value_high[4];
    float value_low[4];
    uint64_t reserved_tail;
    uint64_t previous_record_fingerprint;
    uint64_t record_fingerprint;
} mrnx_behavior_trace_record_v1;

typedef struct MRNX_BEHAVIOR_ALIGN16 mrnx_behavior_trace_chunk_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t trace_status;
    uint32_t record_capacity;
    uint64_t trace_instance_fingerprint;
    uint64_t expected_accepted_roots;
    uint64_t chunk_index;
    uint64_t record_count;
    uint64_t first_attempt_index;
    uint64_t last_attempt_index;
    uint64_t previous_record_fingerprint;
    uint64_t last_record_fingerprint;
    uint64_t observed_attempt_count;
    uint64_t total_record_count;
    uint64_t dropped_record_count;
    uint64_t drained_record_count;
    uint8_t metric_program_sha256[32];
    uint64_t behavior_program_fingerprint;
    uint64_t model_source_fingerprint;
    uint64_t accepted_state_proof_program_fingerprint;
    uint64_t timestep_nanoseconds;
    uint64_t initial_timestamp_nanoseconds;
    uint64_t initial_physics_generation;
    uint32_t clock_domain;
    uint32_t clock_quantum_nanoseconds;
    uint32_t reserved0;
    uint32_t reserved1;
} mrnx_behavior_trace_chunk_v1;

typedef struct mrnx_behavior_trace_terminal_request_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t reason;
    int32_t caller_exit_code;
    uint64_t reserved0;
    uint64_t reserved1;
} mrnx_behavior_trace_terminal_request_v1;

// CAPTURE_COMPLETE closes a bounded exact-runtime capture. It does not imply
// full behavior, physical, biological, performance, or production qualification;
// those evidence classes remain independently flagged below.
typedef struct MRNX_BEHAVIOR_ALIGN16 mrnx_behavior_trace_terminal_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t capture_status;
    uint32_t reason;
    int32_t caller_exit_code;
    uint32_t qualification_flags;
    uint32_t unavailable_evidence_flags;
    uint32_t reserved0;
    uint64_t expected_accepted_roots;
    uint64_t observed_attempt_count;
    uint64_t total_record_count;
    uint64_t drained_record_count;
    uint64_t dropped_record_count;
    uint64_t chunk_count;
    uint64_t accepted_root_count;
    uint64_t rejected_attempt_count;
    uint64_t completed_attempt_count;
    uint64_t initial_timestamp_nanoseconds;
    uint64_t end_timestamp_nanoseconds;
    uint64_t timestep_nanoseconds;
    uint64_t final_publication_epoch;
    uint64_t final_physics_generation;
    uint64_t final_brain_generation;
    uint64_t final_sensor_generation;
    uint64_t final_timestamp_nanoseconds;
    uint64_t final_accepted_token_fingerprint;
    uint64_t final_publication_fingerprint;
    uint64_t behavior_program_fingerprint;
    uint64_t last_record_fingerprint;
    uint64_t terminal_fingerprint;
} mrnx_behavior_trace_terminal_v1;
#ifdef __cplusplus
static_assert(sizeof(mrnx_behavior_trace_config_v1) == 32u);
static_assert(sizeof(mrnx_behavior_trace_record_v1) == 272u);
static_assert(alignof(mrnx_behavior_trace_record_v1) == 16u);
static_assert(offsetof(mrnx_behavior_trace_record_v1, record_fingerprint) == 264u);
static_assert(sizeof(mrnx_behavior_trace_chunk_v1) == 208u);
static_assert(alignof(mrnx_behavior_trace_chunk_v1) == 16u);
static_assert(sizeof(mrnx_behavior_trace_terminal_request_v1) == 32u);
static_assert(sizeof(mrnx_behavior_trace_terminal_v1) == 208u);
static_assert(alignof(mrnx_behavior_trace_terminal_v1) == 16u);
static_assert(offsetof(mrnx_behavior_trace_terminal_v1, terminal_fingerprint) == 200u);
#else
_Static_assert(sizeof(mrnx_behavior_trace_config_v1) == 32u,
               "mrnx_behavior_trace_config_v1 ABI");
_Static_assert(sizeof(mrnx_behavior_trace_record_v1) == 272u,
               "mrnx_behavior_trace_record_v1 ABI");
_Static_assert(_Alignof(mrnx_behavior_trace_record_v1) == 16u,
               "mrnx_behavior_trace_record_v1 alignment");
_Static_assert(offsetof(mrnx_behavior_trace_record_v1, record_fingerprint) == 264u,
               "mrnx_behavior_trace_record_v1 fingerprint offset");
_Static_assert(sizeof(mrnx_behavior_trace_chunk_v1) == 208u,
               "mrnx_behavior_trace_chunk_v1 ABI");
_Static_assert(_Alignof(mrnx_behavior_trace_chunk_v1) == 16u,
               "mrnx_behavior_trace_chunk_v1 alignment");
_Static_assert(sizeof(mrnx_behavior_trace_terminal_request_v1) == 32u,
               "mrnx_behavior_trace_terminal_request_v1 ABI");
_Static_assert(sizeof(mrnx_behavior_trace_terminal_v1) == 208u,
               "mrnx_behavior_trace_terminal_v1 ABI");
_Static_assert(_Alignof(mrnx_behavior_trace_terminal_v1) == 16u,
               "mrnx_behavior_trace_terminal_v1 alignment");
_Static_assert(offsetof(mrnx_behavior_trace_terminal_v1, terminal_fingerprint) == 200u,
               "mrnx_behavior_trace_terminal_v1 fingerprint offset");
#endif
// Optional privileged measurement channel; never part of Brain observations.
// Attach once before any physical attempt. The expected SHA256 is 64 lower-case
// hex characters. Source semantic criteria are verified against actual assets.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_behavior_attach(
    mrnx_runtime_v1* runtime,const char* metric_program_path,const char* expected_sha256,
    uint64_t initial_committed_timestamp_nanoseconds);
// Explicit final collector. Uses the original owner queue and advances no
// physical state/time. Return required bytes INCLUDING NUL, or0 on failure.
// Output includes explicit unknown audit/contact/reset coverage. This is not
// an accepted-root behavior-trial.v1 trace or a generic TaskPack receipt.
MRNX_BRIDGE_EXPORT size_t mrnx_bridge_v1_runtime_behavior_flush_json(
    mrnx_runtime_v1* runtime,char* output,size_t capacity);

// Optional exact-runtime qualification sidecar. Attach after the source-bound
// metric and before the first attempt. The fixed page is drained only while the
// runtime is quiescent. Insufficient output capacity leaves the page unchanged.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_behavior_trace_attach(
    mrnx_runtime_v1* runtime,const mrnx_behavior_trace_config_v1* config);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_behavior_trace_drain(
    mrnx_runtime_v1* runtime,mrnx_behavior_trace_chunk_v1* chunk,
    mrnx_behavior_trace_record_v1* records,uint32_t record_capacity);

// Finalization is idempotent for an identical request and closes this runtime
// to further physical-root admission so the terminal record cannot go stale.
// A true return means a typed terminal record was produced; inspect
// capture_status rather than treating caller_exit_code as qualification.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_behavior_trace_finalize(
    mrnx_runtime_v1* runtime,
    const mrnx_behavior_trace_terminal_request_v1* request,
    mrnx_behavior_trace_terminal_v1* terminal);
#ifdef __cplusplus
}
#endif
#undef MRNX_BEHAVIOR_ALIGN16
#endif
