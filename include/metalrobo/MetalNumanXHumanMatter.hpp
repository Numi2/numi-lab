#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"

#include <cstdint>
#include <memory>
#include <string>

namespace numi::matter {
class Runtime;
}

namespace metalrobo {

namespace detail {
struct MetalNumanXHumanMatterState;
}

// Borrowed proof-production surface offered only during postDynamics, after
// Human stand and after Matter has materialized its success-surviving prepared
// state on the same command buffer. A producer may encode into commandBuffer
// and write acceptedStateProofs; it must not create a queue, commit, wait,
// read back, retain, or replace any borrowed resource. It must hash the final
// candidate Human bytes and the prepared Matter bytes it owns. The adapter
// validates the proof and writes only a quarantined physical-prepare token;
// root publication remains behind the later Brain-ACK apply and exact joint
// COMMITTED publication fence.
// Hashing only identities or status metadata is not a valid content proof.
struct MetalNumanXHumanMatterStateProofPass {
    std::uint32_t abiVersion = MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterStateProofPass);
    std::uint32_t environmentCount = 0u;
    std::uint32_t environmentIdentifierBase = 0u;

    void* commandBuffer = nullptr;
    void* q = nullptr;
    void* v = nullptr;
    void* mujocoStates = nullptr;
    void* matterGeneralizedReaction = nullptr;
    // Exact provisional Human MRMetalWorldStatusGPU stream written earlier
    // on commandBuffer. Matter consumes it as a proof admission witness.
    void* environmentStatuses = nullptr;
    void* matterStatuses = nullptr;
    void* acceptedStateProofs = nullptr;

    std::uint64_t qGPUAddress = 0u;
    std::uint64_t vGPUAddress = 0u;
    std::uint64_t mujocoStatesGPUAddress = 0u;
    std::uint64_t matterGeneralizedReactionGPUAddress = 0u;
    std::uint64_t environmentStatusesGPUAddress = 0u;
    std::uint64_t matterStatusesGPUAddress = 0u;
    std::uint64_t acceptedStateProofsGPUAddress = 0u;

    std::uint64_t qElementCount = 0u;
    std::uint64_t vElementCount = 0u;
    std::uint64_t mujocoStateCount = 0u;
    std::uint64_t matterGeneralizedReactionElementCount = 0u;
    std::uint64_t environmentStatusElementCount = 0u;
    std::uint64_t matterStatusElementCount = 0u;
    std::uint64_t acceptedStateProofElementCount = 0u;

    std::uint32_t qStride = 0u;
    std::uint32_t vStride = 0u;
    std::uint32_t mujocoStateStride = 0u;
    std::uint32_t reactionStride = 0u;
    std::uint32_t environmentStatusStride = 0u;
    std::uint32_t matterStatusStride = 0u;
    std::uint32_t acceptedStateProofStride = 0u;
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    // Exact adapter/owner slot retained in Matter's private prepared binding
    // and checked again against the later device-side final decision.
    std::uint32_t transactionSlot = 0u;

    std::uint64_t programFingerprint = 0u;
    std::uint64_t stateProofProgramFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t substepFingerprint = 0u;
    std::uint64_t acceptedTimestampMicroseconds = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t matterSourcePhysicsFingerprint = 0u;
    std::uint64_t matterDeviceProgramFingerprint = 0u;
};

using MetalNumanXHumanMatterEncodeStateProof = bool (*)(
    void* context,
    const MetalNumanXHumanMatterStateProofPass& pass
) noexcept;

struct MetalNumanXHumanMatterStateProofProgram {
    std::uint32_t abiVersion = MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterStateProofProgram);
    void* context = nullptr;
    MetalNumanXHumanMatterEncodeStateProof encode = nullptr;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return abiVersion == MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION &&
            structSize == sizeof(MetalNumanXHumanMatterStateProofProgram) &&
            context != nullptr && encode != nullptr && fingerprint != 0u;
    }

    [[nodiscard]] bool configured() const noexcept {
        return abiVersion != MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION ||
            structSize != sizeof(MetalNumanXHumanMatterStateProofProgram) ||
            context != nullptr || encode != nullptr || fingerprint != 0u;
    }
};

struct MetalNumanXHumanMatterTransaction {
    std::uint32_t environmentCount = 0u;
    std::uint32_t environmentIdentifierBase = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t physicsSubsteps = 1u;
    std::uint32_t expectedMatterCompletedMicrosteps = 1u;
    // Exact logical Human shape. Defaults preserve the maximum-capacity
    // source ABI; production reference assets set 129/128 explicitly.
    std::uint32_t qCoordinateCount = MR_NUMANX_COUPLED_HUMAN_MAX_Q;
    std::uint32_t dofCount = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
    std::uint32_t dofLayoutVersion =
        kMetalNumanXHumanMatterDofLayoutVersion;
    std::uint32_t reserved0 = 0u;

    std::uint64_t seed = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t substepFingerprint = 0u;
    std::uint64_t acceptedTimestampMicroseconds = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

struct MetalNumanXHumanMatterConfig {
    // Non-owning Runtime identity. It must outlive this context and every
    // command buffer encoded through a returned program.
    numi::matter::Runtime* matterRuntime = nullptr;
    std::string coupledHumanMetallibPath;
    std::string adapterMetallibPath;
    std::uint32_t environmentCapacity = 1u;
    std::uint32_t pointCapacity = 0u;
    std::uint32_t transactionSlotCount = 2u;
    std::uint32_t reserved0 = 0u;
    std::uint64_t maximumRetainedBytes = 1024ull * 1024ull * 1024ull;
    // Optional fail-closed proof authority. Production callers should bridge
    // Runtime::encodeAcceptedStateProof and use the exact fingerprint returned
    // by Runtime::acceptedStateProofProgramFingerprint(). When absent or when
    // it emits an invalid record, the prepared token stays exactly zero and
    // the later ACK-gated apply must reject and restore the transaction.
    MetalNumanXHumanMatterStateProofProgram stateProofProgram{};
};

enum class MetalNumanXHumanMatterHostStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    arithmeticOverflow,
    matterRuntimeUnavailable,
    matterRuntimeIncompatible,
    metallibUnavailable,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    coupledHumanFailure,
    uninitialized,
    invalidTransaction,
    slotBusy,
};

struct MetalNumanXHumanMatterDiagnostics {
    MetalNumanXHumanMatterHostStatus status =
        MetalNumanXHumanMatterHostStatus::success;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t matterSourcePhysicsFingerprint = 0u;
    std::uint64_t matterDeviceProgramFingerprint = 0u;
    std::uint64_t retainedBytes = 0u;
    bool acceptedStateProofAvailable = false;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalNumanXHumanMatterHostStatus::success;
    }
};

// Settled scalar diagnostics for qualification and learning provenance. This
// copies no q/v/MyoSim or proof authority and is available only while the exact
// slot generation still names the queried transaction.
struct MetalNumanXHumanMatterPhysicalOutcome {
    std::uint32_t jointDecision = 0u;
    std::uint32_t humanCode = 0u;
    std::uint32_t matterCode = 0u;
    std::uint32_t humanCompletedSteps = 0u;
    std::uint32_t humanFailingIndex = 0u;
    std::uint32_t humanActiveContactCount = 0u;
    std::uint32_t humanContactIterations = 0u;
    std::array<float, 4u> humanContactAndAcceleration{};
    std::array<float, 4u> humanFactorAndAssistance{};
    std::uint32_t matterCompletedMicrosteps = 0u;
    std::uint32_t worldCode = 0u;
    std::uint32_t worldSuccessfulSubsteps = 0u;
    std::uint32_t worldABACode = 0u;
    std::uint32_t matterObjectIndex = 0u;
    std::uint32_t matterFailingIndex = 0u;
    std::uint32_t matterFGMRESIterations = 0u;
    std::uint32_t matterContactCount = 0u;
    std::array<float, 4u> matterDiagnostics{};
};

// Owns the adapter pipelines and fixed-capacity slot arenas. It never owns a
// command queue and never retains resources borrowed through an owner pass or
// prepare lease. A slot remains quarantined across the physical-prepare and
// proposal/ACK/apply command buffers. REJECT becomes reusable only after exact
// restored release; ACCEPT remains quarantined until an exact COMMITTED joint
// publication fence releases the published root. slotGeneration must advance
// on every use.
class MetalNumanXHumanMatterContext {
public:
    explicit MetalNumanXHumanMatterContext(
        MetalNumanXHumanMatterConfig config = {}
    );
    ~MetalNumanXHumanMatterContext();

    MetalNumanXHumanMatterContext(
        MetalNumanXHumanMatterContext&& other
    ) noexcept;
    MetalNumanXHumanMatterContext& operator=(
        MetalNumanXHumanMatterContext&& other
    ) noexcept;

    MetalNumanXHumanMatterContext(
        const MetalNumanXHumanMatterContext&
    ) = delete;
    MetalNumanXHumanMatterContext& operator=(
        const MetalNumanXHumanMatterContext&
    ) = delete;

    [[nodiscard]] MetalNumanXHumanMatterDiagnostics initialize();

    // Reserves scalar identity for one transaction and returns its exact
    // owner program. An invalid all-zero program is returned on rejection.
    [[nodiscard]] MetalNumanXHumanMatterProgram program(
        const MetalNumanXHumanMatterTransaction& transaction
    ) noexcept;

    [[nodiscard]] bool physicalOutcome(
        std::uint32_t transactionSlot,
        std::uint64_t transactionFingerprint,
        std::uint64_t slotGeneration,
        MetalNumanXHumanMatterPhysicalOutcome& outcome
    ) const noexcept;

private:
    // A submitted command buffer retains this owned state through its
    // completion handler. It never captures borrowed Human buffers.
    std::shared_ptr<detail::MetalNumanXHumanMatterState> state_;
};

[[nodiscard]] const char* metalNumanXHumanMatterHostStatusName(
    MetalNumanXHumanMatterHostStatus status
) noexcept;

} // namespace metalrobo
