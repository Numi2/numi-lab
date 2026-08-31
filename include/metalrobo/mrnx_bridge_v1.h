#ifndef METALROBO_MRNX_BRIDGE_V1_H
#define METALROBO_MRNX_BRIDGE_V1_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(METALROBO_BUILDING_LIBRARY)
#    define MRNX_BRIDGE_EXPORT __declspec(dllexport)
#  else
#    define MRNX_BRIDGE_EXPORT __declspec(dllimport)
#  endif
#else
#  define MRNX_BRIDGE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MRNX_BRIDGE_ABI_V1 1u
#define MRNX_OWNER_WIRE_ABI_V4 4u
#define MRNX_FULL_BODY_NQ 129u
#define MRNX_FULL_BODY_NV 128u
#define MRNX_FULL_BODY_MUSCLE_COUNT 416u
#define MRNX_BRAIN_MOTOR_CANDIDATE_VERSION_V1 7u
#define MRNX_BRAIN_MOTOR_CANDIDATE_VALID_V1 (1u << 0u)
#define MRNX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW_V1 (1u << 1u)
#define MRNX_BRAIN_MOTOR_READY_ABI_VERSION_V1 1u
#define MRNX_BRAIN_MOTOR_READY_GATE_BYTES_V1 160u

typedef struct mrnx_runtime_v1 mrnx_runtime_v1;
typedef struct mrnx_prepared_v1 mrnx_prepared_v1;
typedef struct mrnx_candidate_v1 mrnx_candidate_v1;

typedef enum mrnx_element_type_v1 {
    MRNX_ELEMENT_RAW_BYTES_V1 = 0u,
    MRNX_ELEMENT_FLOAT32_V1 = 1u,
    MRNX_ELEMENT_UINT32_V1 = 2u,
} mrnx_element_type_v1;

typedef enum mrnx_completion_status_v1 {
    MRNX_COMPLETION_READY_V1 = 1u,
    MRNX_COMPLETION_ACCEPTED_PENDING_PUBLICATION_V1 = 2u,
    MRNX_COMPLETION_REJECTED_RELEASED_V1 = 3u,
    MRNX_COMPLETION_TERMINAL_NO_TOUCH_V1 = 4u,
    MRNX_COMPLETION_COMMAND_BUFFER_FAILURE_V1 = 5u,
    MRNX_COMPLETION_TIMEOUT_QUARANTINED_V1 = 6u,
} mrnx_completion_status_v1;

typedef enum mrnx_publication_disposition_v1 {
    MRNX_PUBLICATION_RELEASED_V1 = 1u,
    MRNX_PUBLICATION_REJECTED_V1 = 2u,
    MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1 = 3u,
} mrnx_publication_disposition_v1;

// Canonical disposition of the completed owner apply command. These values
// intentionally differ from mrnx_completion_status_v1 and match the Brain
// joint-resolution SPI exactly.
typedef enum mrnx_command_disposition_v1 {
    MRNX_COMMAND_ACCEPTED_PENDING_PUBLICATION_V1 = 1u,
    MRNX_COMMAND_REJECTED_RELEASED_V1 = 2u,
    MRNX_COMMAND_TERMINAL_NO_TOUCH_V1 = 3u,
} mrnx_command_disposition_v1;

typedef struct mrnx_root_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t owner_wire_abi_version;
    uint32_t environment_count;
    uint32_t environment;
    uint32_t transaction_slot;
    uint32_t step_index;
    uint32_t control_step;
    uint32_t substep_index;
    uint32_t physics_substep_count;
    uint32_t q_coordinate_count;
    uint32_t dof_count;
    uint32_t dof_layout_version;
    uint32_t reserved0;
    uint64_t program_fingerprint;
    uint64_t transaction_fingerprint;
    uint64_t linearization_epoch;
    uint64_t slot_generation;
    uint64_t device_registry_id;
} mrnx_root_v1;

typedef struct mrnx_metal_range_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void* metal_buffer;
    uint64_t gpu_address;
    uint64_t byte_offset;
    uint64_t byte_count;
    uint32_t element_type;
    uint32_t element_byte_count;
} mrnx_metal_range_v1;

// Every metal_buffer/shared_event descriptor is borrowed at +0. It remains
// valid only while the corresponding opaque candidate/prepared handle is
// retained AND its phase-specific copy API still succeeds. All descriptor
// authority expires on terminal or resolved state even if a scalar opaque
// reference remains alive. Callback descriptors expire when the callback
// returns unless the receiver retains that opaque handle and copies the
// fixed-width descriptor before terminal settlement.

typedef struct mrnx_event_point_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void* shared_event;
    uint64_t value;
    uint64_t device_registry_id;
} mrnx_event_point_v1;

typedef struct mrnx_candidate_key_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t transaction_fingerprint;
    uint64_t program_fingerprint;
    uint64_t sensor_fingerprint;
    uint64_t transaction_instance_fingerprint;
    uint64_t sensor_generation;
    uint64_t command_buffer_identity;
    uint64_t fingerprint;
} mrnx_candidate_key_v1;

typedef struct mrnx_candidate_view_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    mrnx_candidate_key_v1 key;
    uint64_t accepted_brain_generation;
    uint64_t candidate_publication_fingerprint;
    uint64_t candidate_identity_fingerprint;
    uint64_t device_registry_id;
    uint32_t channel_count;
    uint32_t reserved0;
} mrnx_candidate_view_v1;

#define MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1 (1u << 0u)
// Direct parity with NumiBrain SensoryModality.proprioception.
#define MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1 4u
// Direct parity with NumiBrain SensoryModality.interoception.
#define MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1 8u

typedef struct mrnx_candidate_channel_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t modality;
    uint32_t flags;
    uint64_t receptor_timestamp_microseconds;
    uint32_t receptor_count;
    uint32_t feature_dimension;
    mrnx_metal_range_v1 values;
    mrnx_metal_range_v1 validity;
} mrnx_candidate_channel_v1;

typedef struct mrnx_wire_lease_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    mrnx_root_v1 root;
    mrnx_metal_range_v1 record;
    mrnx_event_point_v1 ready;
} mrnx_wire_lease_v1;

typedef struct mrnx_proposal_view_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    mrnx_root_v1 root;
    mrnx_metal_range_v1 proposal;
    mrnx_metal_range_v1 proposed_token;
    mrnx_metal_range_v1 publication_fence;
    mrnx_event_point_v1 ready;
} mrnx_proposal_view_v1;

typedef struct mrnx_applied_view_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    mrnx_root_v1 root;
    mrnx_metal_range_v1 applied;
    mrnx_metal_range_v1 final_token;
    mrnx_event_point_v1 ready;
    // One mrnx_command_disposition_v1 value.
    uint32_t command_disposition;
    uint32_t reserved0;
} mrnx_applied_view_v1;

typedef struct mrnx_completion_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t status;
    uint32_t metal_status;
    uint64_t slot_generation;
} mrnx_completion_v1;

typedef struct mrnx_publication_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t joint_commit_fingerprint;
    uint64_t brain_generation;
} mrnx_publication_v1;

typedef enum mrnx_runtime_status_v1 {
    MRNX_RUNTIME_READY_V1 = 1u,
    MRNX_RUNTIME_INVALID_CONFIGURATION_V1 = 2u,
    MRNX_RUNTIME_ASSET_FAILURE_V1 = 3u,
    MRNX_RUNTIME_METAL_FAILURE_V1 = 4u,
    MRNX_RUNTIME_MATTER_FAILURE_V1 = 5u,
    MRNX_RUNTIME_BUSY_V1 = 6u,
    MRNX_RUNTIME_INVALID_REQUEST_V1 = 7u,
    MRNX_RUNTIME_SUBMISSION_FAILURE_V1 = 8u,
    MRNX_RUNTIME_TERMINAL_QUARANTINE_V1 = 9u,
    MRNX_RUNTIME_CONTINUATION_UNAVAILABLE_V1 = 10u,
} mrnx_runtime_status_v1;

// Trusted, process-local runtime construction. All strings are copied before
// create returns. The configured device and every subsequently imported Metal
// object must identify the same Apple GPU. The first production revision is
// deliberately exact: one 129-q/128-DoF floating Human, 416 MyoSim muscles,
// one environment, one physical substep, and one attached non-adaptive Matter
// world. Runtime-owned generations and transaction slots are never supplied
// by the caller.
typedef struct mrnx_runtime_config_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void* metal_device;
    const char* rigid_payload_path;
    const char* muscle_payload_path;
    const char* metalrobo_metallib_path;
    const char* matter_metallib_path;
    const char* matter_material_path;
    uint64_t timestep_microseconds;
    uint64_t maximum_retained_bytes;
    uint32_t transaction_slot_count;
    uint32_t reserved0;
} mrnx_runtime_config_v1;

typedef struct mrnx_runtime_info_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t status;
    uint32_t body_count;
    uint32_t q_coordinate_count;
    uint32_t dof_count;
    uint32_t muscle_count;
    uint32_t transaction_slot_count;
    uint32_t request_failure_stage;
    uint32_t resident_continuation_count;
    uint64_t device_registry_id;
    uint64_t accepted_state_proof_program_fingerprint;
    uint64_t model_source_fingerprint;
} mrnx_runtime_info_v1;

// One aggregate public tuple copied while the bridge's shared reader gate is
// held. A poisoned or never-published runtime returns false and zeroes the
// caller-provided output. The two Metal ranges remain borrowed at +0 only
// while the runtime is retained and until the next successful publication.
typedef struct mrnx_aggregate_snapshot_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t publication_epoch;
    uint64_t brain_generation;
    uint64_t physics_generation;
    uint64_t sensor_generation;
    mrnx_root_v1 root;
    mrnx_candidate_view_v1 sensor;
    mrnx_candidate_channel_v1 proprioception;
    mrnx_candidate_channel_v1 interoception;
} mrnx_aggregate_snapshot_v1;

typedef struct mrnx_brain_joint_transaction_v1 {
    uint32_t format_version;
    uint32_t environment_identifier;
    uint64_t episode_identifier;
    uint64_t control_step_identifier;
    uint64_t parameter_version_fingerprint;
    uint64_t base_brain_generation;
    uint64_t base_physics_generation;
    uint64_t committed_timestamp_microseconds;
    uint64_t target_timestamp_microseconds;
    uint64_t shadow_generation;
    uint64_t random_counter_generation;
    uint32_t flags;
    uint32_t reserved;
    uint64_t transaction_fingerprint;
} mrnx_brain_joint_transaction_v1;

typedef struct mrnx_brain_joint_substep_v1 {
    uint64_t transaction_fingerprint;
    uint32_t substep_index;
    uint32_t attempt_index;
    uint64_t start_timestamp_microseconds;
    uint64_t duration_microseconds;
    uint64_t candidate_timestamp_microseconds;
    uint64_t shadow_generation;
    uint64_t random_counter_generation;
    uint32_t flags;
    uint32_t reserved;
    uint64_t substep_fingerprint;
} mrnx_brain_joint_substep_v1;

typedef struct mrnx_brain_motor_candidate_v1 {
    uint32_t format_version;
    uint32_t flags;
    uint64_t transaction_fingerprint;
    uint64_t substep_fingerprint;
    uint64_t accepted_brain_timestamp_microseconds;
    uint64_t brain_generation;
    uint64_t motor_profile_fingerprint;
    uint64_t motor_output_header_gpu_address;
    uint64_t muscle_excitation_gpu_address;
    uint64_t random_counter_generation;
    uint32_t motor_output_header_byte_count;
    uint32_t muscle_excitation_byte_count;
    uint32_t muscle_count;
    uint32_t environment_identifier;
    uint64_t autonomic_command_gpu_address;
    uint32_t autonomic_command_byte_count;
    uint32_t autonomic_command_count;
    uint64_t active_sensing_command_gpu_address;
    uint32_t active_sensing_command_byte_count;
    uint32_t active_sensing_command_count;
    uint32_t actuator_command_kind;
    uint32_t reserved;
    uint64_t species_template_fingerprint;
    uint64_t compiled_species_template_fingerprint;
    uint64_t candidate_fingerprint;
} mrnx_brain_motor_candidate_v1;

typedef struct mrnx_brain_motor_ready_gate_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    uint32_t status;
    uint32_t environment;
    uint32_t substep_index;
    uint32_t attempt_index;
    uint32_t muscle_count;
    uint32_t actuator_command_kind;
    uint64_t control_step;
    uint64_t transaction_fingerprint;
    uint64_t substep_fingerprint;
    uint64_t candidate_fingerprint;
    uint64_t motor_output_fingerprint;
    uint64_t motor_profile_fingerprint;
    uint64_t brain_generation;
    uint64_t accepted_brain_timestamp_microseconds;
    uint64_t random_counter_generation;
    uint64_t species_template_fingerprint;
    uint64_t compiled_species_template_fingerprint;
    uint64_t brain_program_fingerprint;
    uint64_t fast_program_fingerprint;
    uint64_t decision_gate_fingerprint;
    uint64_t reserved64_0;
    uint64_t gate_fingerprint;
} mrnx_brain_motor_ready_gate_v1;

// Exact asynchronous Brain authority offered to the future native full-body
// begin call. Every range is an exact checked slice of a retained same-device
// MTLBuffer; object identity, base+offset GPU address, count, type, and pairwise
// non-overlap are all authenticated before encoding. The event is liveness
// only; HumanIO validates motor_ready_gate on the GPU before reading any
// payload range. Model dimensions, sensor generation, slot, and physics
// identity are derived by the long-lived native runtime.
typedef struct mrnx_physical_root_request_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    mrnx_brain_joint_transaction_v1 root;
    mrnx_brain_joint_substep_v1 substep;
    mrnx_brain_motor_candidate_v1 candidate;
    mrnx_metal_range_v1 motor_header;
    mrnx_metal_range_v1 muscle_excitation;
    mrnx_metal_range_v1 autonomic_command;
    mrnx_metal_range_v1 active_sensing_command;
    mrnx_metal_range_v1 motor_ready_gate;
    mrnx_event_point_v1 motor_ready;
} mrnx_physical_root_request_v1;

// candidate_handle and every view passed here are borrowed for the callback
// invocation. On success candidate_handle is non-null and carries an internal
// lifecycle self-hold; call mrnx_bridge_v1_candidate_retain before returning
// only if the receiver needs an independent external reference. Failure
// completion supplies no candidate authority. Callbacks run outside native
// runtime locks and may synchronously re-enter the bridge.
typedef void (*mrnx_candidate_settled_callback_v1)(
    void* context,
    mrnx_candidate_v1* candidate_handle,
    const mrnx_completion_v1* completion,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    uint32_t channel_count);

typedef void (*mrnx_proposal_settled_callback_v1)(
    void* context,
    const mrnx_completion_v1* completion,
    const mrnx_proposal_view_v1* proposal);

typedef void (*mrnx_apply_settled_callback_v1)(
    void* context,
    const mrnx_completion_v1* completion,
    const mrnx_applied_view_v1* applied);

// Once begin is armed this callback is delivered exactly once, outside all
// runtime/owner/HumanIO mutexes, after both the original physical command and
// the HumanIO candidate have independently settled. It may run synchronously
// before begin returns. READY supplies borrowed prepared+candidate handles;
// retain them inside the callback to continue proposal/ACK/application.
// Failure supplies no usable authority. The runtime self-retains through the
// exact native terminal resolution even if the external runtime is dropped.
typedef void (*mrnx_physical_root_settled_callback_v1)(
    void* context,
    mrnx_prepared_v1* prepared,
    mrnx_candidate_v1* candidate,
    const mrnx_completion_v1* completion,
    const mrnx_root_v1* root);

// Borrowed synchronous callback. It must not be retained or invoked after the
// release function returns. The Brain runtime lock is already held; callback
// code may latch only the supplied generation and must not call a Brain getter.
typedef bool (*mrnx_brain_generation_latch_v1)(
    void* context,
    uint64_t publishing_brain_generation);

MRNX_BRIDGE_EXPORT uint32_t mrnx_bridge_v1_abi_version(void);

MRNX_BRIDGE_EXPORT mrnx_runtime_v1* mrnx_bridge_v1_runtime_create(
    const mrnx_runtime_config_v1* config,
    mrnx_runtime_info_v1* info);
MRNX_BRIDGE_EXPORT void mrnx_bridge_v1_runtime_retain(
    mrnx_runtime_v1* runtime);
MRNX_BRIDGE_EXPORT void mrnx_bridge_v1_runtime_drop(
    mrnx_runtime_v1* runtime);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_copy_info(
    const mrnx_runtime_v1* runtime,
    mrnx_runtime_info_v1* info);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_begin_physical_root(
    mrnx_runtime_v1* runtime,
    const mrnx_physical_root_request_v1* request,
    void* completion_context,
    mrnx_physical_root_settled_callback_v1 completion);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v1* snapshot);

MRNX_BRIDGE_EXPORT void mrnx_bridge_v1_prepared_retain(
    mrnx_prepared_v1* prepared);
MRNX_BRIDGE_EXPORT void mrnx_bridge_v1_prepared_drop(
    mrnx_prepared_v1* prepared);
MRNX_BRIDGE_EXPORT void mrnx_bridge_v1_candidate_retain(
    mrnx_candidate_v1* candidate);
MRNX_BRIDGE_EXPORT void mrnx_bridge_v1_candidate_drop(
    mrnx_candidate_v1* candidate);

MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_prepared_copy_root(
    const mrnx_prepared_v1* prepared,
    mrnx_root_v1* output);
// Exact GPU-read-only accepted-physics gate: 64-byte prepared token plus the
// owner physical-ready event. Descriptor authority expires on every terminal
// or resolved phase even if an external opaque reference remains alive.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_prepared_copy_physical_gate(
    const mrnx_prepared_v1* prepared,
    mrnx_wire_lease_v1* output);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_candidate_copy_view(
    const mrnx_candidate_v1* candidate,
    mrnx_candidate_view_v1* output);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_candidate_copy_channel(
    const mrnx_candidate_v1* candidate,
    uint32_t channel_index,
    mrnx_candidate_channel_v1* output);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_bind_candidate(
    mrnx_prepared_v1* prepared,
    mrnx_candidate_v1* candidate);
// A successful bind makes the prepared handle retain the exact candidate and
// its move-only HumanIO publication lease through terminal root resolution.
// Dropping the caller's candidate reference cannot release that authority.

// All command buffers are borrowed at +0, must be same-device and
// NotEnqueued, and are retained by the opaque handle through settlement or an
// exact abort. The bridge encodes only: it never creates a queue, commits,
// waits, polls, reads payload bytes, or performs a readback/blit. Brain wire
// record/event objects are strongly retained after successful admission.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_submit_proposal(
    mrnx_prepared_v1* prepared,
    void* command_buffer,
    const mrnx_wire_lease_v1* brain_witness,
    void* completion_context,
    mrnx_proposal_settled_callback_v1 completion);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_abort_proposal(
    mrnx_prepared_v1* prepared,
    void* command_buffer);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_reserve_application(
    mrnx_prepared_v1* prepared,
    const mrnx_wire_lease_v1* brain_preflight);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_submit_apply(
    mrnx_prepared_v1* prepared,
    void* command_buffer,
    const mrnx_wire_lease_v1* brain_ack,
    void* completion_context,
    mrnx_apply_settled_callback_v1 completion);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_abort_apply(
    mrnx_prepared_v1* prepared,
    void* command_buffer);

// Timeout is sticky and blocks every ordinary witness/ACK transition above.
// These three explicitly named operations are the sole pre-apply teardown
// path after mrnx_bridge_v1_quarantine_timeout succeeds. They encode the
// owner's canonical forceReject proposal/apply modes and therefore never
// consume a Brain witness or ACK. The application reservation still requires
// an exact, GPU-produced Brain preflight lease; the bridge never fabricates a
// Brain record. A caller may use the forced reservation/apply after a late
// ordinary proposal completes, or use all three operations after timing out
// before proposal. A late native REJECT is released only by
// mrnx_bridge_v1_release_rejected. A normal ACCEPT apply already in flight
// when timeout wins is permanently quarantined and cannot be published. The
// generic abort functions reject a timed-out ordinary command; they may abort
// only an explicit timeout-reject command that is still NotEnqueued, preserving
// the same forced-reject phase for an exact retry.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_submit_timeout_reject_proposal(
    mrnx_prepared_v1* prepared,
    void* command_buffer,
    void* completion_context,
    mrnx_proposal_settled_callback_v1 completion);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_reserve_timeout_reject_application(
    mrnx_prepared_v1* prepared,
    const mrnx_wire_lease_v1* brain_preflight);
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_submit_timeout_reject_apply(
    mrnx_prepared_v1* prepared,
    void* command_buffer,
    void* completion_context,
    mrnx_apply_settled_callback_v1 completion);

MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_reserve_publication(
    mrnx_prepared_v1* prepared,
    const mrnx_publication_v1* publication);

// Called synchronously while the Brain runtime lock is already held. The
// bridge takes its sole public-reader writer gate, invokes generation_latch
// exactly once before any Matter/HumanIO visibility, and keeps the gate held
// through the native root release. A false latch invokes no native release and
// permanently quarantines the generation. No callback is retained. After a
// true latch the bridge performs no queue operation, GPU host wait, allocation,
// payload readback, or blit. The release is not lock-free: it acquires the
// predeclared owner -> adapter -> Matter -> HumanIO mutex chain while the
// writer gate remains held.
MRNX_BRIDGE_EXPORT uint32_t mrnx_bridge_v1_release_accepted(
    mrnx_prepared_v1* prepared,
    const mrnx_publication_v1* publication,
    void* latch_context,
    mrnx_brain_generation_latch_v1 generation_latch);
MRNX_BRIDGE_EXPORT uint32_t mrnx_bridge_v1_release_rejected(
    mrnx_prepared_v1* prepared);

// Host timeout is observation only: it never cancels, restores, reuses, or
// destroys the unresolved native capability. It is sticky and immediately
// disables all ordinary ACCEPT-capable forward operations. Only the explicit
// timeout-reject operations above may continue a pre-apply root. The lifecycle
// self-hold remains until an exact later REJECT release, otherwise for runtime
// teardown.
MRNX_BRIDGE_EXPORT bool mrnx_bridge_v1_quarantine_timeout(
    mrnx_prepared_v1* prepared);

// Rejects an unbound HumanIO candidate via its exact native one-shot program.
// A bound candidate can be resolved only by its owning Prepared root.
MRNX_BRIDGE_EXPORT uint32_t mrnx_bridge_v1_reject_unbound_candidate(
    mrnx_candidate_v1* candidate);

// Opaque handles are created only by the bridge-owned runtime implementation;
// no C entrypoint adopts arbitrary C++ Prepared or HumanIO callback/program
// objects.

#ifdef __cplusplus
} // extern "C"

static_assert(sizeof(mrnx_root_v1) == 96u);
static_assert(alignof(mrnx_root_v1) == 8u);
static_assert(offsetof(mrnx_root_v1, program_fingerprint) == 56u);
static_assert(offsetof(mrnx_root_v1, device_registry_id) == 88u);
static_assert(sizeof(mrnx_metal_range_v1) == 48u);
static_assert(offsetof(mrnx_metal_range_v1, metal_buffer) == 8u);
static_assert(offsetof(mrnx_metal_range_v1, gpu_address) == 16u);
static_assert(offsetof(mrnx_metal_range_v1, element_type) == 40u);
static_assert(sizeof(mrnx_event_point_v1) == 32u);
static_assert(offsetof(mrnx_event_point_v1, shared_event) == 8u);
static_assert(sizeof(mrnx_candidate_key_v1) == 64u);
static_assert(offsetof(mrnx_candidate_key_v1, fingerprint) == 56u);
static_assert(sizeof(mrnx_candidate_view_v1) == 112u);
static_assert(offsetof(mrnx_candidate_view_v1,
                       candidate_publication_fingerprint) == 80u);
static_assert(sizeof(mrnx_candidate_channel_v1) == 128u);
static_assert(offsetof(mrnx_candidate_channel_v1, values) == 32u);
static_assert(offsetof(mrnx_candidate_channel_v1, validity) == 80u);
static_assert(sizeof(mrnx_wire_lease_v1) == 184u);
static_assert(offsetof(mrnx_wire_lease_v1, record) == 104u);
static_assert(offsetof(mrnx_wire_lease_v1, ready) == 152u);
static_assert(sizeof(mrnx_proposal_view_v1) == 280u);
static_assert(offsetof(mrnx_proposal_view_v1, proposal) == 104u);
static_assert(offsetof(mrnx_proposal_view_v1, ready) == 248u);
static_assert(sizeof(mrnx_applied_view_v1) == 240u);
static_assert(offsetof(mrnx_applied_view_v1, applied) == 104u);
static_assert(offsetof(mrnx_applied_view_v1, command_disposition) == 232u);
static_assert(sizeof(mrnx_completion_v1) == 24u);
static_assert(offsetof(mrnx_completion_v1, slot_generation) == 16u);
static_assert(sizeof(mrnx_publication_v1) == 24u);
static_assert(offsetof(mrnx_publication_v1,
                       joint_commit_fingerprint) == 8u);
static_assert(sizeof(mrnx_runtime_config_v1) == 80u);
static_assert(offsetof(mrnx_runtime_config_v1,
                       timestep_microseconds) == 56u);
static_assert(offsetof(mrnx_runtime_config_v1,
                       maximum_retained_bytes) == 64u);
static_assert(sizeof(mrnx_runtime_info_v1) == 64u);
static_assert(offsetof(mrnx_runtime_info_v1,
                       device_registry_id) == 40u);
static_assert(sizeof(mrnx_aggregate_snapshot_v1) == 504u);
static_assert(offsetof(mrnx_aggregate_snapshot_v1, root) == 40u);
static_assert(offsetof(mrnx_aggregate_snapshot_v1, sensor) == 136u);
static_assert(offsetof(mrnx_aggregate_snapshot_v1,
                       proprioception) == 248u);
static_assert(offsetof(mrnx_aggregate_snapshot_v1,
                       interoception) == 376u);
static_assert(sizeof(mrnx_brain_joint_transaction_v1) == 96u);
static_assert(alignof(mrnx_brain_joint_transaction_v1) == 8u);
static_assert(offsetof(mrnx_brain_joint_transaction_v1,
                       transaction_fingerprint) == 88u);
static_assert(sizeof(mrnx_brain_joint_substep_v1) == 72u);
static_assert(alignof(mrnx_brain_joint_substep_v1) == 8u);
static_assert(offsetof(mrnx_brain_joint_substep_v1,
                       substep_fingerprint) == 64u);
static_assert(sizeof(mrnx_brain_motor_candidate_v1) == 152u);
static_assert(alignof(mrnx_brain_motor_candidate_v1) == 8u);
static_assert(offsetof(mrnx_brain_motor_candidate_v1,
                       candidate_fingerprint) == 144u);
static_assert(sizeof(mrnx_brain_motor_ready_gate_v1) == 160u);
static_assert(alignof(mrnx_brain_motor_ready_gate_v1) == 8u);
static_assert(offsetof(mrnx_brain_motor_ready_gate_v1,
                       gate_fingerprint) == 152u);
static_assert(sizeof(mrnx_physical_root_request_v1) == 600u);
static_assert(alignof(mrnx_physical_root_request_v1) == 8u);
static_assert(offsetof(mrnx_physical_root_request_v1, root) == 8u);
static_assert(offsetof(mrnx_physical_root_request_v1,
                       motor_header) == 328u);
static_assert(offsetof(mrnx_physical_root_request_v1,
                       motor_ready_gate) == 520u);
static_assert(offsetof(mrnx_physical_root_request_v1,
                       motor_ready) == 568u);
#else
_Static_assert(sizeof(mrnx_root_v1) == 96u, "mrnx_root_v1 ABI");
_Static_assert(_Alignof(mrnx_root_v1) == 8u, "mrnx_root_v1 alignment");
_Static_assert(offsetof(mrnx_root_v1, program_fingerprint) == 56u,
               "mrnx_root_v1 program offset");
_Static_assert(offsetof(mrnx_root_v1, device_registry_id) == 88u,
               "mrnx_root_v1 device offset");
_Static_assert(sizeof(mrnx_metal_range_v1) == 48u,
               "mrnx_metal_range_v1 ABI");
_Static_assert(_Alignof(mrnx_metal_range_v1) == 8u,
               "mrnx_metal_range_v1 alignment");
_Static_assert(offsetof(mrnx_metal_range_v1, metal_buffer) == 8u,
               "mrnx_metal_range_v1 object offset");
_Static_assert(offsetof(mrnx_metal_range_v1, element_type) == 40u,
               "mrnx_metal_range_v1 type offset");
_Static_assert(sizeof(mrnx_event_point_v1) == 32u,
               "mrnx_event_point_v1 ABI");
_Static_assert(offsetof(mrnx_event_point_v1, shared_event) == 8u,
               "mrnx_event_point_v1 event offset");
_Static_assert(sizeof(mrnx_candidate_key_v1) == 64u,
               "mrnx_candidate_key_v1 ABI");
_Static_assert(offsetof(mrnx_candidate_key_v1, fingerprint) == 56u,
               "mrnx_candidate_key_v1 fingerprint offset");
_Static_assert(sizeof(mrnx_candidate_view_v1) == 112u,
               "mrnx_candidate_view_v1 ABI");
_Static_assert(offsetof(mrnx_candidate_view_v1,
                        candidate_publication_fingerprint) == 80u,
               "mrnx_candidate_view_v1 publication offset");
_Static_assert(sizeof(mrnx_candidate_channel_v1) == 128u,
               "mrnx_candidate_channel_v1 ABI");
_Static_assert(offsetof(mrnx_candidate_channel_v1, values) == 32u,
               "mrnx_candidate_channel_v1 values offset");
_Static_assert(sizeof(mrnx_wire_lease_v1) == 184u,
               "mrnx_wire_lease_v1 ABI");
_Static_assert(offsetof(mrnx_wire_lease_v1, ready) == 152u,
               "mrnx_wire_lease_v1 ready offset");
_Static_assert(sizeof(mrnx_proposal_view_v1) == 280u,
               "mrnx_proposal_view_v1 ABI");
_Static_assert(offsetof(mrnx_proposal_view_v1, ready) == 248u,
               "mrnx_proposal_view_v1 ready offset");
_Static_assert(sizeof(mrnx_applied_view_v1) == 240u,
               "mrnx_applied_view_v1 ABI");
_Static_assert(offsetof(mrnx_applied_view_v1, command_disposition) == 232u,
               "mrnx_applied_view_v1 disposition offset");
_Static_assert(sizeof(mrnx_completion_v1) == 24u,
               "mrnx_completion_v1 ABI");
_Static_assert(offsetof(mrnx_completion_v1, slot_generation) == 16u,
               "mrnx_completion_v1 generation offset");
_Static_assert(sizeof(mrnx_publication_v1) == 24u,
               "mrnx_publication_v1 ABI");
_Static_assert(sizeof(mrnx_runtime_config_v1) == 80u,
               "mrnx_runtime_config_v1 ABI");
_Static_assert(offsetof(mrnx_runtime_config_v1,
                        timestep_microseconds) == 56u,
               "mrnx_runtime_config_v1 timestep offset");
_Static_assert(offsetof(mrnx_runtime_config_v1,
                        maximum_retained_bytes) == 64u,
               "mrnx_runtime_config_v1 retained offset");
_Static_assert(sizeof(mrnx_runtime_info_v1) == 64u,
               "mrnx_runtime_info_v1 ABI");
_Static_assert(sizeof(mrnx_aggregate_snapshot_v1) == 504u,
               "mrnx_aggregate_snapshot_v1 ABI");
_Static_assert(offsetof(mrnx_aggregate_snapshot_v1, root) == 40u,
               "mrnx_aggregate_snapshot_v1 root offset");
_Static_assert(offsetof(mrnx_aggregate_snapshot_v1,
                        proprioception) == 248u,
               "mrnx_aggregate_snapshot_v1 channel offset");
_Static_assert(offsetof(mrnx_aggregate_snapshot_v1,
                        interoception) == 376u,
               "mrnx_aggregate_snapshot_v1 interoception offset");
_Static_assert(sizeof(mrnx_brain_joint_transaction_v1) == 96u,
               "mrnx_brain_joint_transaction_v1 ABI");
_Static_assert(_Alignof(mrnx_brain_joint_transaction_v1) == 8u,
               "mrnx_brain_joint_transaction_v1 alignment");
_Static_assert(offsetof(mrnx_brain_joint_transaction_v1,
                        transaction_fingerprint) == 88u,
               "mrnx_brain_joint_transaction_v1 fingerprint offset");
_Static_assert(sizeof(mrnx_brain_joint_substep_v1) == 72u,
               "mrnx_brain_joint_substep_v1 ABI");
_Static_assert(_Alignof(mrnx_brain_joint_substep_v1) == 8u,
               "mrnx_brain_joint_substep_v1 alignment");
_Static_assert(offsetof(mrnx_brain_joint_substep_v1,
                        substep_fingerprint) == 64u,
               "mrnx_brain_joint_substep_v1 fingerprint offset");
_Static_assert(sizeof(mrnx_brain_motor_candidate_v1) == 152u,
               "mrnx_brain_motor_candidate_v1 ABI");
_Static_assert(_Alignof(mrnx_brain_motor_candidate_v1) == 8u,
               "mrnx_brain_motor_candidate_v1 alignment");
_Static_assert(offsetof(mrnx_brain_motor_candidate_v1,
                        candidate_fingerprint) == 144u,
               "mrnx_brain_motor_candidate_v1 fingerprint offset");
_Static_assert(sizeof(mrnx_brain_motor_ready_gate_v1) == 160u,
               "mrnx_brain_motor_ready_gate_v1 ABI");
_Static_assert(offsetof(mrnx_brain_motor_ready_gate_v1,
                        gate_fingerprint) == 152u,
               "mrnx_brain_motor_ready_gate_v1 fingerprint offset");
_Static_assert(sizeof(mrnx_physical_root_request_v1) == 600u,
               "mrnx_physical_root_request_v1 ABI");
_Static_assert(offsetof(mrnx_physical_root_request_v1,
                        motor_header) == 328u,
               "mrnx_physical_root_request_v1 motor header offset");
_Static_assert(offsetof(mrnx_physical_root_request_v1,
                        motor_ready_gate) == 520u,
               "mrnx_physical_root_request_v1 gate offset");
_Static_assert(offsetof(mrnx_physical_root_request_v1,
                        motor_ready) == 568u,
               "mrnx_physical_root_request_v1 event offset");
#endif

#endif // METALROBO_MRNX_BRIDGE_V1_H
