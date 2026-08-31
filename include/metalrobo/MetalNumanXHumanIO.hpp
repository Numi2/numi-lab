#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numanx_human_io_gpu.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace metalrobo {

namespace detail {
struct MetalNumanXHumanIOState;
} // namespace detail

// Authoritative little-endian FNV-1a mirrors of NumiBrainABI. These helpers
// intentionally hash candidate metadata and GPU addresses, not mutable device
// payload bytes; the latter are deterministically validated again on the
// trusted command-buffer timeline by the motor-output kernel. This is not
// cryptographic authentication.
[[nodiscard]] std::uint64_t metalNumanXBrainJointTransactionFingerprint(
    const MRNumanXBrainJointTransactionToken& token
) noexcept;
[[nodiscard]] std::uint64_t metalNumanXBrainJointSubstepFingerprint(
    const MRNumanXBrainJointSubstepToken& token
) noexcept;
[[nodiscard]] std::uint64_t metalNumanXBrainMotorCandidateFingerprint(
    const MRNumanXBrainMotorCandidate& candidate
) noexcept;

struct MetalNumanXHumanIOConfig {
    // Explicit library selection is intentional: the adapter never falls
    // back to a process-default metallib or runtime source compilation.
    std::string metallibPath;
    std::size_t maximumRetainedBytes =
        1024ull * 1024ull * 1024ull;
};

// One exact NumiBrain/NumanX motor candidate. NumiBrain's root, substep, and
// motor-candidate records are single-environment values, so this first
// authoritative adapter revision rejects environmentCount != 1. A future
// cohort path requires a separately versioned canonical batch identity.
//
// All four Metal buffers are borrowed native id<MTLBuffer> lease identities;
// the context never retains them. The caller must keep every object and slice
// alive and unchanged through command-buffer completion. Backing buffers may
// be larger, but each offset/count/address triple is exact and is rechecked at
// prepare and every transaction phase. Only the somatic payload enters MyoSim;
// autonomic and active-sensing leases remain provenance-bound parts of the
// complete candidate handoff.
struct MetalNumanXHumanIOInput {
    MRNumanXBrainJointTransactionToken root{};
    MRNumanXBrainJointSubstepToken substep{};
    MRNumanXBrainMotorCandidate candidate{};

    void* motorOutputHeaderMetalBuffer = nullptr;
    std::size_t motorOutputHeaderByteOffset = 0u;
    std::size_t motorOutputHeaderByteCount = 0u;
    std::size_t motorOutputHeaderEnvironmentStride = 0u;
    std::uint64_t expectedMotorOutputHeaderGPUAddress = 0u;

    void* excitationMetalBuffer = nullptr;
    std::size_t excitationByteOffset = 0u;
    std::size_t excitationByteCount = 0u;
    std::size_t excitationEnvironmentStride = 0u;
    std::uint64_t expectedExcitationGPUAddress = 0u;

    void* autonomicCommandMetalBuffer = nullptr;
    std::size_t autonomicCommandByteOffset = 0u;
    std::size_t autonomicCommandByteCount = 0u;
    std::uint64_t expectedAutonomicCommandGPUAddress = 0u;

    void* activeSensingCommandMetalBuffer = nullptr;
    std::size_t activeSensingCommandByteOffset = 0u;
    std::size_t activeSensingCommandByteCount = 0u;
    std::uint64_t expectedActiveSensingCommandGPUAddress = 0u;

    // Required for ABI-v7 DECISION_SHADOW candidates. The event is liveness
    // only; beginStep waits on it and the GPU validates the exact terminal
    // SUCCESS gate before touching any motor payload. Legacy synchronous
    // VALID-only candidates must leave all fields zero.
    void* motorReadyGateMetalBuffer = nullptr;
    std::size_t motorReadyGateByteOffset = 0u;
    std::size_t motorReadyGateByteCount = 0u;
    std::uint64_t expectedMotorReadyGateGPUAddress = 0u;
    void* motorReadySharedEvent = nullptr;
    std::uint64_t motorReadySharedEventValue = 0u;

    std::uint32_t environmentCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t stepCount = 0u;
    float timestepSeconds = 0.0f;

    // Exact integer authority for the receptor sample at horizon step zero.
    // Step s is evaluated at this timestamp plus s * timestepSeconds and
    // delivered after the stand update at one timestep later. The public sensor
    // view derives seconds from this integer; callers never have to reproduce a
    // particular binary double representation. MyoSim geometry/result features
    // are not post-stand features.
    std::uint64_t receptorTimestampMicroseconds = 0u;

    // The candidate owns all brain/transaction provenance. A sensor generation
    // is MetalRobo-owned and must advance relative to the published view.
    std::uint64_t candidateSensorGeneration = 0u;
};

enum class MetalNumanXHumanIOViewState : std::uint32_t {
    candidate = 0u,
    published = 1u,
};

// Zero-copy native publication descriptor. The context retains all four
// modality buffers
// objects; these void* values are borrowed id<MTLBuffer> identities and must
// not be released by the caller. A candidate view is quarantined storage, not
// an accepted sensor packet. A published view remains byte- and
// identity-stable until an exact reserved publication atomically swaps slots.
struct MetalNumanXHumanIOSensorView {
    void* proprioceptionMetalBuffer = nullptr;
    void* validityMetalBuffer = nullptr;
    void* interoceptionMetalBuffer = nullptr;
    void* interoceptionValidityMetalBuffer = nullptr;
    std::uint64_t proprioceptionGPUAddress = 0u;
    std::uint64_t validityGPUAddress = 0u;
    std::uint64_t interoceptionGPUAddress = 0u;
    std::uint64_t interoceptionValidityGPUAddress = 0u;
    std::size_t proprioceptionByteCount = 0u;
    std::size_t validityByteCount = 0u;
    std::size_t interoceptionByteCount = 0u;
    std::size_t interoceptionValidityByteCount = 0u;

    std::uint32_t environmentCount = 0u;
    std::uint32_t stepCount = 0u;
    std::uint32_t receptorCount = 0u;
    std::uint32_t featureCount = 0u;
    std::size_t proprioceptionEnvironmentStrideElements = 0u;
    std::size_t proprioceptionStepStrideElements = 0u;
    std::size_t proprioceptionReceptorStrideElements = 0u;
    std::size_t validityEnvironmentStrideElements = 0u;
    std::size_t validityStepStrideElements = 0u;
    std::size_t validityReceptorStrideElements = 0u;
    std::size_t interoceptionEnvironmentStrideElements = 0u;
    std::size_t interoceptionStepStrideElements = 0u;
    std::size_t interoceptionReceptorStrideElements = 0u;
    std::size_t interoceptionValidityEnvironmentStrideElements = 0u;
    std::size_t interoceptionValidityStepStrideElements = 0u;
    std::size_t interoceptionValidityReceptorStrideElements = 0u;

    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t motorCandidateFingerprint = 0u;
    std::uint64_t acceptedBrainGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t programFingerprint = 0u;
    // Includes all provenance above plus exact excitation GPU address and
    // tensor/timing dimensions. Stable before command-buffer binding.
    std::uint64_t sensorFingerprint = 0u;
    // Adds the command-buffer identity to sensorFingerprint. It is zero in
    // the prepare() storage view and nonzero after beginStep is encoded.
    std::uint64_t transactionInstanceFingerprint = 0u;
    std::uint64_t excitationGPUAddress = 0u;
    std::uint64_t motorOutputHeaderGPUAddress = 0u;
    std::uintptr_t commandBufferIdentity = 0u;

    // Exact sensor-clock authority. The double-valued fields below are
    // convenience projections only and must never be rounded back into an
    // integer transaction identity.
    std::uint64_t receptorTimestampMicroseconds = 0u;
    std::uint64_t deliveryTimestampMicroseconds = 0u;
    std::uint32_t latencyMicroseconds = 0u;
    std::uint32_t stepTimeStrideMicroseconds = 0u;
    double receptorTimeSeconds = 0.0;
    double deliveryTimeSeconds = 0.0;
    double latencySeconds = 0.0;
    double stepTimeStrideSeconds = 0.0;
    MetalNumanXHumanIOViewState state =
        MetalNumanXHumanIOViewState::candidate;
};

// Completion/acceptance key. Pointer identity alone is deliberately
// insufficient: every provenance and generation fingerprint contributes to
// transactionInstanceFingerprint.
struct MetalNumanXHumanIOTransactionKey {
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t sensorFingerprint = 0u;
    std::uint64_t transactionInstanceFingerprint = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uintptr_t commandBufferIdentity = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return transactionFingerprint != 0u &&
            programFingerprint != 0u && sensorFingerprint != 0u &&
            transactionInstanceFingerprint != 0u &&
            sensorGeneration != 0u && commandBufferIdentity != 0u;
    }
};

enum class MetalNumanXHumanIOCandidateCompletionStatus : std::uint32_t {
    succeeded = 1u,
    commandBufferFailure = 2u,
};

// Borrowed synchronous notification. The key and view are valid only for the
// callback invocation. The HumanIO mutex is never held while this callback
// runs, so it may immediately reserve the exact candidate publication lease.
using MetalNumanXHumanIOCandidateCompletion = void (*)(
    void* context,
    MetalNumanXHumanIOCandidateCompletionStatus status,
    const MetalNumanXHumanIOTransactionKey& key,
    const MetalNumanXHumanIOSensorView& candidateView
) noexcept;

enum class MetalNumanXHumanIOSensorElementType : std::uint32_t {
    float32 = 1u,
    uint32 = 2u,
};

// Exact native range metadata retained by a candidate-publication lease.
// metalBuffer is a borrowed id<MTLBuffer> identity; the move-only lease keeps
// the owning HumanIO state and both adapter-owned buffers alive. Offsets are
// explicit even though ABI1 admits only whole adapter-owned buffers at offset
// zero, so a future sliced layout cannot silently change the identity.
struct MetalNumanXHumanIOSensorBufferLease {
    void* metalBuffer = nullptr;
    std::uint64_t gpuAddress = 0u;
    std::size_t byteOffset = 0u;
    std::size_t byteCount = 0u;
    MetalNumanXHumanIOSensorElementType elementType =
        MetalNumanXHumanIOSensorElementType::float32;
    std::uint32_t elementByteCount = 0u;
};

struct MetalNumanXHumanIOCandidatePublicationView {
    std::uint32_t abiVersion =
        kMetalNumanXHumanIOPublicationABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanIOCandidatePublicationView);
    MetalNumanXHumanIOSensorView sensor{};
    MetalNumanXHumanIOSensorBufferLease proprioception{};
    MetalNumanXHumanIOSensorBufferLease validity{};
    MetalNumanXHumanIOSensorBufferLease interoception{};
    MetalNumanXHumanIOSensorBufferLease interoceptionValidity{};
    std::uint64_t deviceRegistryID = 0u;
    std::uint64_t candidatePublicationFingerprint = 0u;
};

[[nodiscard]] std::uint64_t
metalNumanXHumanIOPublicationBindingFingerprint(
    const MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept;

[[nodiscard]] std::uint64_t
metalNumanXHumanIOCandidatePublicationIdentityFingerprint(
    const MetalNumanXHumanIOCandidatePublicationProgram& program
) noexcept;

class MetalNumanXHumanIOCandidatePublicationLease {
public:
    MetalNumanXHumanIOCandidatePublicationLease() = default;
    ~MetalNumanXHumanIOCandidatePublicationLease() = default;

    MetalNumanXHumanIOCandidatePublicationLease(
        MetalNumanXHumanIOCandidatePublicationLease&& other
    ) noexcept = default;
    MetalNumanXHumanIOCandidatePublicationLease& operator=(
        MetalNumanXHumanIOCandidatePublicationLease&& other
    ) noexcept = default;

    MetalNumanXHumanIOCandidatePublicationLease(
        const MetalNumanXHumanIOCandidatePublicationLease&
    ) = delete;
    MetalNumanXHumanIOCandidatePublicationLease& operator=(
        const MetalNumanXHumanIOCandidatePublicationLease&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const MetalNumanXHumanIOCandidatePublicationProgram&
    program() const noexcept {
        return program_;
    }
    [[nodiscard]] const MetalNumanXHumanIOCandidatePublicationView&
    view() const noexcept {
        return view_;
    }

private:
    friend class MetalNumanXHumanIOContext;
    std::shared_ptr<detail::MetalNumanXHumanIOState> state_;
    MetalNumanXHumanIOCandidatePublicationProgram program_{};
    MetalNumanXHumanIOCandidatePublicationView view_{};
};

enum class MetalNumanXHumanIOStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidInput,
    arithmeticOverflow,
    metalDeviceUnavailable,
    metallibUnavailable,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    incompatibleDevice,
    incompatibleTransaction,
    contextBusy,
    candidateUnavailable,
    candidateNotEncoded,
    candidateInFlight,
    commandBufferFailure,
    humanDiagnosticsRejected,
    internalFailure,
};

struct MetalNumanXHumanIODiagnostics {
    MetalNumanXHumanIOStatus status = MetalNumanXHumanIOStatus::success;
    bool encoded = false;
    bool commandBufferCompleted = false;
    bool commandBufferSucceeded = false;
    bool published = false;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uintptr_t commandBufferIdentity = 0u;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalNumanXHumanIOStatus::success;
    }
};

// Move-only, reusable adapter. prepare() allocates/selects a private candidate
// slot and returns a transaction program for MetalNumiHumanStandInput. The
// context and all four borrowed motor-candidate buffers must outlive the
// enclosing submitted Human command buffer. The callback only encodes: it
// never commits, waits, reads payloads, or retains borrowed pass resources.
//
// prepare()'s candidateView exposes storage for same-device composition but is
// explicitly not accepted. After submit(), pendingCandidate() supplies the
// full command-buffer-bound key. reserveCandidatePublication() validates the
// HumanIO completion and freezes a provisional move-only candidate capability
// without changing the published slot. The Human owner separately certifies
// physical success while binding that capability to its prepared root. Only
// the resulting exact root callbacks can publish or reject the generation.
class MetalNumanXHumanIOContext {
public:
    explicit MetalNumanXHumanIOContext(
        MetalNumanXHumanIOConfig config = {}
    );
    ~MetalNumanXHumanIOContext();

    MetalNumanXHumanIOContext(
        MetalNumanXHumanIOContext&& other
    ) noexcept;
    MetalNumanXHumanIOContext& operator=(
        MetalNumanXHumanIOContext&& other
    ) noexcept;

    MetalNumanXHumanIOContext(
        const MetalNumanXHumanIOContext&
    ) = delete;
    MetalNumanXHumanIOContext& operator=(
        const MetalNumanXHumanIOContext&
    ) = delete;

    [[nodiscard]] MetalNumanXHumanIODiagnostics prepare(
        const MetalNumanXHumanIOInput& input,
        MetalNumanXTransactionProgram& program,
        MetalNumanXHumanIOSensorView& candidateView
    );

    [[nodiscard]] MetalNumanXHumanIODiagnostics pendingCandidate(
        MetalNumanXHumanIOTransactionKey& key,
        MetalNumanXHumanIOSensorView& candidateView
    ) const;

    // Registers exactly once for the exact command-buffer-bound candidate.
    // Completion first settles HumanIO state under its mutex, then invokes the
    // callback outside the mutex. If completion already won the race, this
    // call invokes synchronously before returning. No handler-order, polling,
    // host wait, or retry assumption is required.
    [[nodiscard]] MetalNumanXHumanIODiagnostics
    registerCandidateCompletion(
        const MetalNumanXHumanIOTransactionKey& key,
        void* completionContext,
        MetalNumanXHumanIOCandidateCompletion completion
    ) noexcept;

    // Recovery for an enclosing Human submission rejected by host validation
    // before any transaction phase is offered (and therefore before the seam
    // can invoke abort). It cannot cancel an encoding/in-flight command.
    [[nodiscard]] MetalNumanXHumanIODiagnostics cancelPrepared(
        std::uint64_t transactionFingerprint,
        std::uint64_t programFingerprint
    );

    [[nodiscard]] MetalNumanXHumanIODiagnostics
    reserveCandidatePublication(
        const MetalNumanXHumanIOTransactionKey& key,
        MetalNumanXHumanIOCandidatePublicationLease& lease
    );

    [[nodiscard]] MetalNumanXHumanIODiagnostics reject(
        const MetalNumanXHumanIOTransactionKey& key
    );

    [[nodiscard]] MetalNumanXHumanIODiagnostics publishedView(
        MetalNumanXHumanIOSensorView& view
    ) const;

private:
    std::shared_ptr<detail::MetalNumanXHumanIOState> state_;
};

[[nodiscard]] const char* metalNumanXHumanIOStatusName(
    MetalNumanXHumanIOStatus status
) noexcept;

static_assert(sizeof(MetalNumanXHumanIOCandidatePublicationBinding) == 192u);
static_assert(alignof(MetalNumanXHumanIOCandidatePublicationBinding) == 8u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationBinding,
    ownerProgramFingerprint
) == 32u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationBinding,
    bindingFingerprint
) == 184u);
static_assert(sizeof(MetalNumanXHumanIOCandidatePublicationCommit) == 56u);
static_assert(alignof(MetalNumanXHumanIOCandidatePublicationCommit) == 8u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationCommit,
    candidatePublicationFingerprint
) == 16u);
static_assert(sizeof(
    MetalNumanXHumanIOCandidatePublicationProgram
) == 120u);
static_assert(alignof(
    MetalNumanXHumanIOCandidatePublicationProgram
) == 8u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationProgram,
    transactionFingerprint
) == 48u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationProgram,
    identityFingerprint
) == 112u);

} // namespace metalrobo
