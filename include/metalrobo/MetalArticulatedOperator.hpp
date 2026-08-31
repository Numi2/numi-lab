#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/millard_muscle_gpu.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include "metalrobo/numanx_human_matter_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {
struct MetalArticulatedOperatorContextState;
struct MetalArticulatedOperatorSubmissionState;
struct MetalNumanXHumanMatterPreparedState;

[[nodiscard]] constexpr std::size_t
articulatedOperatorThreadgroupBytes(
    const std::size_t bodyCount,
    const std::size_t dofCount,
    const bool includeDenseDynamics = true
) noexcept {
    const auto aligned16 = [](const std::size_t value) {
        return (value + 15u) & ~std::size_t{15u};
    };
    std::size_t bytes = 0u;
    const auto append = [&bytes, &aligned16](
        const std::size_t value
    ) {
        bytes = aligned16(bytes);
        bytes += value;
    };
    // float3 occupies a 16-byte slot in Metal threadgroup memory.
    append(16u * bodyCount); // body position
    append(16u * bodyCount); // body rotation
    append(16u * bodyCount); // joint position
    append(16u * bodyCount); // joint axis
    append(sizeof(std::uint32_t) * bodyCount); // inbound joint
    append(sizeof(std::uint32_t) * bodyCount); // parent body
    append(sizeof(std::uint8_t) * bodyCount); // topology-known flags
    if (includeDenseDynamics) {
        append(sizeof(float) * dofCount * dofCount); // dense factor
        append(sizeof(float) * dofCount); // right-hand side
        append(sizeof(float) * dofCount); // forward solve
        append(sizeof(float) * dofCount); // solution
    }
    return aligned16(bytes);
}
} // namespace detail

// Immutable source program and mutable state sidecar for the Millard
// reference pass. It is optional: all spans empty preserves the articulated
// operator's historical kinematics/mass behavior. When present, the muscle
// pass executes after the FunctionBased kinematics/Jacobian kernel in the
// same command buffer and receives no CPU-restaged pose or Jacobian data.
struct MetalMillardReferenceInput {
    std::span<const MRMillardMuscleGPU> muscles{};
    std::span<const MRMillardMuscleStateGPU> states{};
    std::span<const MRMillardPathPointGPU> pathPoints{};
    std::span<const MRMillardSourceCurveGPU> curves{};
    std::span<const MRMillardCylinderWrapGPU> cylinderWraps{};

    [[nodiscard]] bool enabled() const noexcept {
        return !muscles.empty();
    }
};

// Immutable MyoSim source program and environment-major activation state for
// the MuJoCo general-muscle device reference. Its path kernel reads the
// articulated pose stream produced immediately before it in the same command
// buffer; no CPU-restaged geometry is admitted.
struct MetalMujocoMuscleReferenceInput {
    std::span<const MRMujocoMuscleGPU> muscles{};
    std::span<const MRMujocoMuscleStateGPU> states{};
    std::span<const MRMujocoMuscleSiteGPU> sites{};
    std::span<const MRMujocoMuscleWrapGPU> wraps{};
    std::span<const MRMujocoMuscleRouteNodeGPU> routeNodes{};
    // Index in the enclosing point-query stream of a dense per-body probe
    // block: COM, local +X, local +Y, local +Z. The source route kernel uses
    // it to reconstruct a world-space spatial Jacobian at each authored site
    // and tangent point, then emits source generalized muscle force on Metal.
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;

    [[nodiscard]] bool enabled() const noexcept {
        return !muscles.empty();
    }
};

// Borrowed device view shared by the pre-dynamics continuum pass and its
// post-validation reconciliation in every persistent-Human step. The first
// callback runs after tendon transfer but before articulated dynamics, so an
// accepted continuum reaction may replace (never duplicate) a declared share
// of the source J^T force. The second callback runs after stand validation and
// must commit or roll back the matching continuum transaction from device
// status. Neither callback may commit, wait, retain, or replace borrowed
// objects. Transfer records are environment-major [environment][endpoint].
struct MetalNumiHumanTendonLoadPass {
    void* commandBuffer = nullptr;
    void* bindings = nullptr;
    void* envelopes = nullptr;
    void* transfers = nullptr;
    void* generalizedCorrections = nullptr;
    void* generalizedForces = nullptr;
    void* bodyPoses = nullptr;
    void* pointJacobians = nullptr;
    void* standStatuses = nullptr;
    std::uint32_t stepIndex = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t endpointCount = 0u;
    std::uint32_t envelopeCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t generalizedForceStride = 0u;
    std::uint32_t generalizedForceOffset = 0u;
    std::uint32_t pointJacobianStride = 0u;
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;
    std::uint32_t bodyPoseStride = 0u;
    std::uint32_t articulationFirstBody = 0u;
};

using MetalNumiHumanTendonLoadEncode = bool (*)(
    void* context,
    const MetalNumiHumanTendonLoadPass& pass
);

// Called only if a downstream load pass was encoded but the enclosing Human
// command buffer is abandoned before commit. It releases consumer-side
// transaction ownership and must not touch the borrowed Metal resources.
using MetalNumiHumanTendonLoadAbort = void (*)(
    void* context,
    void* commandBuffer
);

struct MetalNumiHumanTendonLoadProgram {
    void* context = nullptr;
    MetalNumiHumanTendonLoadEncode encodePreDynamics = nullptr;
    MetalNumiHumanTendonLoadEncode encodePostValidation = nullptr;
    MetalNumiHumanTendonLoadAbort abort = nullptr;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return context != nullptr && encodePreDynamics != nullptr &&
            encodePostValidation != nullptr && abort != nullptr &&
            fingerprint != 0u;
    }

    [[nodiscard]] bool configured() const noexcept {
        return context != nullptr || encodePreDynamics != nullptr ||
            encodePostValidation != nullptr || abort != nullptr ||
            fingerprint != 0u;
    }
};

enum class MetalNumanXTransactionPhase : std::uint32_t {
    beginStep = 0u,
    preDynamics = 1u,
    postDynamics = 2u,
};

inline constexpr std::uint32_t kMetalNumanXTransactionABIVersion = 1u;

enum MetalNumanXTransactionAccessFlag : std::uint32_t {
    MetalNumanXTransactionReadBorrowedState = 1u << 0u,
    // beginStep only: x in each MRMujocoMuscleStateGPU may be replaced by an
    // authenticated motor excitation before current-step kinematics.
    MetalNumanXTransactionWriteMujocoExcitation = 1u << 1u,
    // postDynamics only: a consumer may change a successful stand status to
    // a defined failure code. It may never turn failure into success.
    MetalNumanXTransactionWriteStandFailure = 1u << 2u,
};

// Borrowed device view of one persistent NumanX step. The callback is an
// encoder hook: it may append work to commandBuffer but must not inspect GPU
// results synchronously, commit, wait, retain, or replace any borrowed object.
// beginStep precedes current-step kinematics; preDynamics follows current-step
// MyoSim activation and tendon transfer; postDynamics follows stand dynamics.
// At postDynamics, q/v/status are post-stand while poses, point streams, MyoSim
// results, and tendon transfers remain that step's pre-dynamics evaluation.
struct MetalNumanXTransactionPass {
    std::uint32_t abiVersion = kMetalNumanXTransactionABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXTransactionPass);
    std::uint32_t accessFlags = 0u;
    std::uint32_t reserved0 = 0u;

    void* commandBuffer = nullptr;
    void* q = nullptr;
    void* v = nullptr;
    void* bodyPoses = nullptr;
    void* pointWorld = nullptr;
    void* pointJacobians = nullptr;
    void* mujocoMuscles = nullptr;
    void* mujocoStates = nullptr;
    void* mujocoSites = nullptr;
    void* mujocoWraps = nullptr;
    void* mujocoRouteNodes = nullptr;
    void* mujocoResults = nullptr;
    void* mujocoGeneralizedForceArena = nullptr;
    void* tendonBindings = nullptr;
    void* tendonEnvelopes = nullptr;
    void* tendonTransfers = nullptr;
    void* tendonGeneralizedCorrections = nullptr;
    void* standStatuses = nullptr;

    MetalNumanXTransactionPhase phase =
        MetalNumanXTransactionPhase::beginStep;
    std::uint64_t programFingerprint = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t stepCount = 0u;
    float timestepSeconds = 0.0f;
    std::uint32_t articulationFirstBody = 0u;
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;

    std::uint64_t environmentCount = 0u;
    std::uint64_t qCoordinateCount = 0u;
    std::uint64_t qElementCount = 0u;
    std::uint64_t qStride = 0u;
    std::uint64_t dofCount = 0u;
    std::uint64_t vElementCount = 0u;
    std::uint64_t vStride = 0u;
    std::uint64_t bodyCount = 0u;
    std::uint64_t bodyPoseElementCount = 0u;
    std::uint64_t bodyPoseStride = 0u;
    std::uint64_t pointCount = 0u;
    std::uint64_t pointWorldElementCount = 0u;
    std::uint64_t pointWorldStride = 0u;
    std::uint64_t pointJacobianElementCount = 0u;
    std::uint64_t pointJacobianStride = 0u;

    std::uint64_t mujocoMuscleCount = 0u;
    std::uint64_t mujocoStateElementCount = 0u;
    std::uint64_t mujocoStateStride = 0u;
    std::uint64_t mujocoSiteCount = 0u;
    std::uint64_t mujocoWrapCount = 0u;
    std::uint64_t mujocoRouteNodeCount = 0u;
    std::uint64_t mujocoResultElementCount = 0u;
    std::uint64_t mujocoResultStride = 0u;
    std::uint64_t mujocoMuscleGeneralizedForceElementCount = 0u;
    std::uint64_t mujocoMuscleGeneralizedForceRowStride = 0u;
    std::uint64_t mujocoMuscleGeneralizedForceEnvironmentStride = 0u;
    std::uint64_t mujocoGeneralizedForceElementCount = 0u;
    std::uint64_t mujocoGeneralizedForceOffset = 0u;
    std::uint64_t mujocoGeneralizedForceStride = 0u;
    std::uint64_t mujocoGeneralizedForceArenaElementCount = 0u;

    std::uint64_t tendonBindingCount = 0u;
    std::uint64_t tendonEnvelopeCount = 0u;
    std::uint64_t tendonTransferElementCount = 0u;
    std::uint64_t tendonTransferStride = 0u;
    std::uint64_t tendonCorrectionElementCount = 0u;
    std::uint64_t tendonCorrectionStride = 0u;
    std::uint64_t standStatusElementCount = 0u;
    std::uint64_t standStatusStride = 0u;
};

using MetalNumanXTransactionEncode = bool (*)(
    void* context,
    const MetalNumanXTransactionPass& pass
) noexcept;

// Called exactly once if at least one transaction phase was offered to the
// consumer and the enclosing command buffer is then abandoned before commit.
// It must release consumer-side transaction ownership without touching the
// borrowed Metal resources.
using MetalNumanXTransactionAbort = void (*)(
    void* context,
    void* commandBuffer
) noexcept;

struct MetalNumanXTransactionProgram {
    std::uint32_t abiVersion = kMetalNumanXTransactionABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXTransactionProgram);
    void* context = nullptr;
    MetalNumanXTransactionEncode encode = nullptr;
    MetalNumanXTransactionAbort abort = nullptr;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid() const noexcept {
        return abiVersion == kMetalNumanXTransactionABIVersion &&
            structSize == sizeof(MetalNumanXTransactionProgram) &&
            context != nullptr && encode != nullptr && abort != nullptr &&
            fingerprint != 0u;
    }

    [[nodiscard]] bool configured() const noexcept {
        return abiVersion != kMetalNumanXTransactionABIVersion ||
            structSize != sizeof(MetalNumanXTransactionProgram) ||
            context != nullptr || encode != nullptr || abort != nullptr ||
            fingerprint != 0u;
    }
};

enum class MetalNumanXHumanMatterPhase : std::uint32_t {
    beginStep = 0u,
    preDynamics = 1u,
    postDynamics = 2u,
};

inline constexpr std::uint32_t kMetalNumanXHumanMatterABIVersion =
    MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
inline constexpr std::uint32_t kMetalNumanXHumanMatterDofLayoutVersion = 1u;

enum MetalNumanXHumanMatterAccessFlag : std::uint32_t {
    MetalNumanXHumanMatterReadLiveHumanState = 1u << 0u,
    MetalNumanXHumanMatterReadHumanCheckpoints = 1u << 1u,
    MetalNumanXHumanMatterReadSourceEffectiveTangent = 1u << 2u,
    MetalNumanXHumanMatterMayEncodeExactCandidate = 1u << 3u,
    MetalNumanXHumanMatterWriteStagedReaction = 1u << 4u,
    MetalNumanXHumanMatterWriteJointStatus = 1u << 5u,
    MetalNumanXHumanMatterWritePreparedPhysicsToken = 1u << 6u,
};

inline constexpr std::uint32_t kMetalNumanXHumanMatterKnownAccess =
    MetalNumanXHumanMatterReadLiveHumanState |
    MetalNumanXHumanMatterReadHumanCheckpoints |
    MetalNumanXHumanMatterReadSourceEffectiveTangent |
    MetalNumanXHumanMatterMayEncodeExactCandidate |
    MetalNumanXHumanMatterWriteStagedReaction |
    MetalNumanXHumanMatterWriteJointStatus |
    MetalNumanXHumanMatterWritePreparedPhysicsToken;

enum MetalNumanXHumanMatterCapability : std::uint32_t {
    // Exact q/body/attachment kinematics are encoded by the Human owner from
    // the supplied delta-v; no first-order pose approximation is admitted.
    MetalNumanXHumanMatterExactCandidateKinematics = 1u << 0u,
    // The exposed lower factor is the frozen source-step discrete velocity
    // tangent A0 = M(q0) + armature + h*D.  It is intentionally not named or
    // represented as a pure mass matrix.
    MetalNumanXHumanMatterSourceEffectiveTangent = 1u << 1u,
    // A separate staged generalized reaction A0*deltaV/h is consumed once
    // before stand.  It never aliases MyoSim's per-muscle or reduced rows.
    MetalNumanXHumanMatterStagedReaction = 1u << 2u,
    MetalNumanXHumanMatterJointDecision = 1u << 3u,
    MetalNumanXHumanMatterPreparedPhysicsGate = 1u << 4u,
};

inline constexpr std::uint32_t kMetalNumanXHumanMatterKnownCapabilities =
    MetalNumanXHumanMatterExactCandidateKinematics |
    MetalNumanXHumanMatterSourceEffectiveTangent |
    MetalNumanXHumanMatterStagedReaction |
    MetalNumanXHumanMatterJointDecision |
    MetalNumanXHumanMatterPreparedPhysicsGate;

struct MetalNumanXHumanMatterPass;

enum MetalNumanXHumanMatterCandidateAccessFlag : std::uint32_t {
    MetalNumanXHumanMatterCandidateReadDeltaVelocity = 1u << 0u,
    MetalNumanXHumanMatterCandidateWriteQ = 1u << 1u,
    MetalNumanXHumanMatterCandidateWriteBodies = 1u << 2u,
    MetalNumanXHumanMatterCandidateReadPointQueries = 1u << 3u,
    MetalNumanXHumanMatterCandidateWritePointWorld = 1u << 4u,
    MetalNumanXHumanMatterCandidateWritePointJacobians = 1u << 5u,
};

inline constexpr std::uint32_t
    kMetalNumanXHumanMatterCandidateKnownAccess =
        MetalNumanXHumanMatterCandidateReadDeltaVelocity |
        MetalNumanXHumanMatterCandidateWriteQ |
        MetalNumanXHumanMatterCandidateWriteBodies |
        MetalNumanXHumanMatterCandidateReadPointQueries |
        MetalNumanXHumanMatterCandidateWritePointWorld |
        MetalNumanXHumanMatterCandidateWritePointJacobians;

// One exact candidate requested by the Matter adapter during preDynamics.
// deltaVelocity is environment-major [environment][160]. candidateQ and
// candidateBodies are authoritative outputs. Optional attachment queries are
// evaluated in the same generic kinematics dispatch as the four immutable
// Human body probes; pointWorld and pointJacobians name only the attachment
// suffix, not the private combined stream.
struct MetalNumanXHumanMatterCandidateQuery {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterCandidateQuery);
    std::uint32_t accessFlags = 0u;
    std::uint32_t reserved0 = 0u;

    void* deltaVelocity = nullptr;
    void* candidateQ = nullptr;
    void* candidateBodies = nullptr;
    void* pointQueries = nullptr;
    void* pointWorld = nullptr;
    void* pointJacobians = nullptr;

    std::uint64_t deltaVelocityGPUAddress = 0u;
    std::uint64_t candidateQGPUAddress = 0u;
    std::uint64_t candidateBodiesGPUAddress = 0u;
    std::uint64_t pointQueriesGPUAddress = 0u;
    std::uint64_t pointWorldGPUAddress = 0u;
    std::uint64_t pointJacobiansGPUAddress = 0u;

    std::uint32_t deltaVelocityStride = 0u;
    std::uint32_t candidateQStride = 0u;
    std::uint32_t candidateBodyStride = 0u;
    std::uint32_t pointCount = 0u;
    std::uint32_t pointStride = 0u;
    std::uint32_t pointWorldStride = 0u;
    std::uint32_t pointJacobianStride = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t transactionSlot = 0u;
    // ABI4 is one atomic control-step root. Matter must not expose an
    // intermediate physics substep that a later substep can roll back.
    std::uint32_t physicsSubstepCount = 0u;

    // Global adapter/Matter control root. This is not the local Human stand
    // step, which remains exactly zero in the one-step owner.
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

using MetalNumanXHumanMatterEncodeExactCandidate = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPass& pass,
    const MetalNumanXHumanMatterCandidateQuery& query
) noexcept;

// Borrowed view of one owner transaction. All native objects are unretained
// id<MTLBuffer>/id<MTLCommandBuffer> identities. The callback may encode only
// on commandBuffer; it must not commit, wait, retain, read back, or replace a
// resource. beginStep follows owner checkpointing and precedes motor mutation.
// preDynamics follows source kinematics, MyoSim, and exact A0 factorization;
// the owner consumes the staged reaction immediately after it returns.
// postDynamics follows stand and produces a quarantined physical prepare;
// later proposal/ACK/apply commands own root resolution.
struct MetalNumanXHumanMatterPass {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterPass);
    std::uint32_t accessFlags = 0u;
    std::uint32_t capabilities = 0u;

    MetalNumanXHumanMatterPhase phase =
        MetalNumanXHumanMatterPhase::beginStep;
    std::uint32_t stepIndex = 0u;
    std::uint32_t stepCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;

    // Global adapter/Matter control root, independent of local stepIndex.
    std::uint32_t controlStep = 0u;

    void* commandBuffer = nullptr;
    void* q = nullptr;
    void* v = nullptr;
    void* mujocoStates = nullptr;
    void* mujocoGeneralizedForceArena = nullptr;
    void* bodyPoses = nullptr;
    void* pointQueries = nullptr;
    void* pointWorld = nullptr;
    void* pointJacobians = nullptr;
    void* standStatuses = nullptr;

    void* qCheckpoint = nullptr;
    void* vCheckpoint = nullptr;
    void* mujocoStateCheckpoint = nullptr;
    // Environment-major lower Cholesky of A0. The stand kernel subsequently
    // rebuilds and factors the same source-step bytes before its solve.
    void* sourceEffectiveTangentFactor = nullptr;
    void* ownerStatuses = nullptr;

    // Adapter-owned, same-device arenas. matterGeneralizedReaction is
    // [environment][160] force and is accumulated exactly once before stand.
    // jointStatuses use the 32-byte MRNumanXCoupledHumanStatusGPU prefix.
    // acceptedPhysicsStateTokens retain their source-compatible spelling but
    // are ABI4 *prepared* tokens: begin clears them, a physical reject clears
    // them, and physical prepare preserves the adapter's 64-byte record only
    // inside the quarantined slot.  They are never a root-commit publication.
    void* matterGeneralizedReaction = nullptr;
    void* jointStatuses = nullptr;
    void* acceptedPhysicsStateTokens = nullptr;

    void* exactCandidateContext = nullptr;
    MetalNumanXHumanMatterEncodeExactCandidate encodeExactCandidate = nullptr;

    std::uint64_t qGPUAddress = 0u;
    std::uint64_t vGPUAddress = 0u;
    std::uint64_t mujocoStatesGPUAddress = 0u;
    std::uint64_t mujocoGeneralizedForceArenaGPUAddress = 0u;
    std::uint64_t bodyPosesGPUAddress = 0u;
    std::uint64_t pointQueriesGPUAddress = 0u;
    std::uint64_t pointWorldGPUAddress = 0u;
    std::uint64_t pointJacobiansGPUAddress = 0u;
    std::uint64_t standStatusesGPUAddress = 0u;
    std::uint64_t qCheckpointGPUAddress = 0u;
    std::uint64_t vCheckpointGPUAddress = 0u;
    std::uint64_t mujocoStateCheckpointGPUAddress = 0u;
    std::uint64_t sourceEffectiveTangentFactorGPUAddress = 0u;
    std::uint64_t ownerStatusesGPUAddress = 0u;
    std::uint64_t matterGeneralizedReactionGPUAddress = 0u;
    std::uint64_t jointStatusesGPUAddress = 0u;
    std::uint64_t acceptedPhysicsStateTokensGPUAddress = 0u;

    std::uint64_t environmentCount = 0u;
    std::uint64_t qCoordinateCount = 0u;
    std::uint64_t dofCount = 0u;
    std::uint64_t bodyCount = 0u;
    std::uint64_t pointCount = 0u;
    std::uint64_t mujocoStateCount = 0u;
    std::uint64_t qStride = 0u;
    std::uint64_t vStride = 0u;
    std::uint64_t bodyPoseStride = 0u;
    std::uint64_t pointStride = 0u;
    std::uint64_t pointWorldStride = 0u;
    std::uint64_t pointJacobianStride = 0u;
    std::uint64_t mujocoStateStride = 0u;
    std::uint64_t factorStride = 0u;
    std::uint64_t generalizedForceOffset = 0u;
    std::uint64_t generalizedForceStride = 0u;
    std::uint64_t generalizedForceArenaElementCount = 0u;
    std::uint64_t reactionStride = 0u;
    std::uint64_t jointStatusStride = 0u;
    std::uint64_t acceptedTokenStrideBytes = 0u;
    std::uint64_t bodyJacobianPointOffset = 0u;

    float timestepSeconds = 0.0f;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t articulationFirstBody = 0u;
    std::uint32_t reserved0 = 0u;

    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

using MetalNumanXHumanMatterEncode = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPass& pass
) noexcept;

using MetalNumanXHumanMatterAbort = void (*)(
    void* context,
    void* commandBuffer
) noexcept;

inline constexpr std::uint32_t kMetalNumanXHumanIOPublicationABIVersion = 1u;

enum class MetalNumanXHumanIOCandidatePublicationDisposition : std::uint32_t {
    released = 1u,
    rejected = 2u,
    terminalNoTouch = 3u,
};

struct MetalNumanXHumanIOCandidatePublicationBinding {
    std::uint32_t abiVersion =
        kMetalNumanXHumanIOPublicationABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanIOCandidatePublicationBinding);
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t ownerProgramFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t physicsTokenFingerprint = 0u;
    std::uint64_t proposalFingerprint = 0u;
    std::uint64_t ackFingerprint = 0u;
    std::uint64_t appliedDecisionFingerprint = 0u;
    std::uint64_t jointCommitFingerprint = 0u;
    std::uint64_t brainGeneration = 0u;
    std::uint64_t candidateKeyFingerprint = 0u;
    std::uint64_t acceptedBrainGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t humanIOProgramFingerprint = 0u;
    std::uint64_t sensorFingerprint = 0u;
    std::uint64_t transactionInstanceFingerprint = 0u;
    std::uint64_t candidatePublicationFingerprint = 0u;
    std::uint64_t deviceRegistryID = 0u;
    std::uint64_t humanIOIdentityFingerprint = 0u;
    std::uint64_t bindingFingerprint = 0u;
};

enum class MetalNumanXHumanIOCandidatePublicationCommitStatus :
    std::uint32_t {
    committed = 1u,
};

struct MetalNumanXHumanIOCandidatePublicationCommit {
    std::uint32_t abiVersion =
        kMetalNumanXHumanIOPublicationABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanIOCandidatePublicationCommit);
    MetalNumanXHumanIOCandidatePublicationCommitStatus status =
        MetalNumanXHumanIOCandidatePublicationCommitStatus::committed;
    std::uint32_t reserved0 = 0u;
    std::uint64_t candidatePublicationFingerprint = 0u;
    std::uint64_t bindingFingerprint = 0u;
    std::uint64_t jointCommitFingerprint = 0u;
    std::uint64_t brainGeneration = 0u;
    std::uint64_t fenceFingerprint = 0u;
};

using MetalNumanXHumanIOReservePublishedRoot = bool (*)(
    void* context,
    std::uint64_t candidatePublicationFingerprint,
    const MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept;

using MetalNumanXHumanIOPublishCandidate =
    MetalNumanXHumanIOCandidatePublicationDisposition (*)(
    void* context,
    std::uint64_t candidatePublicationFingerprint,
    const MetalNumanXHumanIOCandidatePublicationCommit& commit
) noexcept;

using MetalNumanXHumanIORejectCandidate =
    MetalNumanXHumanIOCandidatePublicationDisposition (*)(
    void* context,
    std::uint64_t candidatePublicationFingerprint
) noexcept;

// Pointer-bearing host capability for one already-materialized, unpublished
// HumanIO sensor candidate. Its FNV identity deliberately excludes context and
// callback addresses; candidatePublicationFingerprint binds the exact local
// buffer objects, GPU ranges, scalar types, and layout inside HumanIO.
struct MetalNumanXHumanIOCandidatePublicationProgram {
    std::uint32_t abiVersion =
        kMetalNumanXHumanIOPublicationABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanIOCandidatePublicationProgram);
    void* context = nullptr;
    MetalNumanXHumanIOReservePublishedRoot reservePublishedRoot = nullptr;
    MetalNumanXHumanIOPublishCandidate publishCandidate = nullptr;
    MetalNumanXHumanIORejectCandidate rejectCandidate = nullptr;
    std::uint64_t candidateKeyFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t acceptedBrainGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t humanIOProgramFingerprint = 0u;
    std::uint64_t sensorFingerprint = 0u;
    std::uint64_t transactionInstanceFingerprint = 0u;
    std::uint64_t candidatePublicationFingerprint = 0u;
    std::uint64_t deviceRegistryID = 0u;
    std::uint64_t identityFingerprint = 0u;

    [[nodiscard]] std::uint64_t computedIdentityFingerprint()
        const noexcept {
        std::uint64_t hash = 14695981039346656037ull;
        constexpr std::uint64_t prime = 1099511628211ull;
        const auto mix32 = [&hash](std::uint32_t value) noexcept {
            for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
                hash ^= static_cast<std::uint8_t>(value >> (8u * byte));
                hash *= prime;
            }
        };
        const auto mix64 = [&hash](std::uint64_t value) noexcept {
            for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
                hash ^= static_cast<std::uint8_t>(value >> (8u * byte));
                hash *= prime;
            }
        };
        mix32(abiVersion);
        mix32(structSize);
        mix64(candidateKeyFingerprint);
        mix64(transactionFingerprint);
        mix64(acceptedBrainGeneration);
        mix64(sensorGeneration);
        mix64(humanIOProgramFingerprint);
        mix64(sensorFingerprint);
        mix64(transactionInstanceFingerprint);
        mix64(candidatePublicationFingerprint);
        mix64(deviceRegistryID);
        return hash == 0u ? 14695981039346656037ull : hash;
    }

    [[nodiscard]] bool valid() const noexcept {
        return abiVersion == kMetalNumanXHumanIOPublicationABIVersion &&
            structSize == sizeof(*this) && context != nullptr &&
            reservePublishedRoot != nullptr && publishCandidate != nullptr &&
            rejectCandidate != nullptr && candidateKeyFingerprint != 0u &&
            transactionFingerprint != 0u && acceptedBrainGeneration != 0u &&
            sensorGeneration != 0u && humanIOProgramFingerprint != 0u &&
            sensorFingerprint != 0u && transactionInstanceFingerprint != 0u &&
            candidatePublicationFingerprint != 0u &&
            deviceRegistryID != 0u &&
            identityFingerprint != 0u &&
            identityFingerprint == computedIdentityFingerprint();
    }

    [[nodiscard]] bool configured() const noexcept {
        return abiVersion != kMetalNumanXHumanIOPublicationABIVersion ||
            structSize != sizeof(*this) || context != nullptr ||
            reservePublishedRoot != nullptr || publishCandidate != nullptr ||
            rejectCandidate != nullptr || candidateKeyFingerprint != 0u ||
            transactionFingerprint != 0u || acceptedBrainGeneration != 0u ||
            sensorGeneration != 0u || humanIOProgramFingerprint != 0u ||
            sensorFingerprint != 0u || transactionInstanceFingerprint != 0u ||
            candidatePublicationFingerprint != 0u ||
            deviceRegistryID != 0u || identityFingerprint != 0u;
    }
};

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
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationCommit,
    candidatePublicationFingerprint
) == 16u);
static_assert(sizeof(MetalNumanXHumanIOCandidatePublicationProgram) == 120u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationProgram,
    transactionFingerprint
) == 48u);
static_assert(offsetof(
    MetalNumanXHumanIOCandidatePublicationProgram,
    identityFingerprint
) == 112u);

struct MetalNumanXHumanMatterPrepareLease {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterPrepareLease);
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t dofLayoutVersion = 0u;
    std::uint32_t reservedDofLayout = 0u;

    std::uint32_t preparedTokenStrideBytes = 0u;
    std::uint32_t proposalStride = 0u;
    std::uint32_t proposedTokenStrideBytes = 0u;
    std::uint32_t applyActionStride = 0u;
    std::uint32_t matterApplyOutcomeStride = 0u;
    std::uint32_t appliedOutcomeStride = 0u;
    std::uint32_t finalTokenStrideBytes = 0u;
    std::uint32_t publicationFenceStride = 0u;

    void* preparedPhysicsStateTokens = nullptr;
    void* proposals = nullptr;
    void* proposedPhysicsStateTokens = nullptr;
    void* applyActions = nullptr;
    void* matterApplyOutcomes = nullptr;
    void* appliedOutcomes = nullptr;
    void* finalAcceptedPhysicsStateTokens = nullptr;
    void* publicationFences = nullptr;
    void* physicalPreparedEvent = nullptr;

    std::uint64_t preparedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t proposalsGPUAddress = 0u;
    std::uint64_t proposedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t applyActionsGPUAddress = 0u;
    std::uint64_t matterApplyOutcomesGPUAddress = 0u;
    std::uint64_t appliedOutcomesGPUAddress = 0u;
    std::uint64_t finalAcceptedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t publicationFencesGPUAddress = 0u;

    std::uint64_t preparedPhysicsStateTokenByteCount = 0u;
    std::uint64_t proposalElementCount = 0u;
    std::uint64_t proposedPhysicsStateTokenByteCount = 0u;
    std::uint64_t applyActionElementCount = 0u;
    std::uint64_t matterApplyOutcomeElementCount = 0u;
    std::uint64_t appliedOutcomeElementCount = 0u;
    std::uint64_t finalAcceptedPhysicsStateTokenByteCount = 0u;
    std::uint64_t publicationFenceElementCount = 0u;

    std::uint64_t physicalPreparedEventValue = 0u;
    std::uint64_t proposalEventValue = 0u;
    std::uint64_t appliedEventValue = 0u;

    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    MetalNumanXHumanIOCandidatePublicationProgram humanIOCandidate{};
};

using MetalNumanXHumanMatterAcquirePrepareLease = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept;

// The exact HumanIO publication capability does not exist until the first
// physical command has completed and HumanIO has certified its materialized
// candidate. This one-shot callback immutably attaches that post-command
// capability to the already-quarantined adapter generation. The staged lease
// contains the candidate; the originally acquired lease contains an empty
// candidate program.
using MetalNumanXHumanMatterBindHumanIOCandidatePublication = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanIOCandidatePublicationProgram& candidate
) noexcept;

// Host disposition returned only after the adapter has inspected its exact
// completed finalization generation. `released` is the sole authority for the
// Human owner to invalidate/reap the prepared root. `restoreRequired` retains
// the immutable checkpoints and permits only a fresh forceReject command.
// `terminalNoTouch` permanently quarantines the generation: neither owner may
// restore, publish, release, nor reuse it before context teardown.
enum class MetalNumanXHumanMatterPrepareLeaseDisposition : std::uint32_t {
    released = 1u,
    restoreRequired = 2u,
    terminalNoTouch = 3u,
    acceptedPendingPublication = 4u,
};

using MetalNumanXHumanMatterReleasePrepareLease =
    MetalNumanXHumanMatterPrepareLeaseDisposition (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    void* finalCommandBuffer,
    bool completed
) noexcept;

struct MetalNumanXHumanMatterProposalView {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterProposalView);
    void* proposals = nullptr;
    void* proposedPhysicsStateTokens = nullptr;
    std::uint64_t proposalsGPUAddress = 0u;
    std::uint64_t proposedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t proposalElementCount = 0u;
    std::uint64_t proposedPhysicsStateTokenByteCount = 0u;
    std::uint32_t proposalStride = 0u;
    std::uint32_t proposedTokenStrideBytes = 0u;
    std::uint64_t proposalEventValue = 0u;
};

struct MetalNumanXHumanMatterBrainPreflightView {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanMatterBrainPreflightView);
    void* brainCommitPreflights = nullptr;
    void* preflightReadyEvent = nullptr;
    std::uint64_t brainCommitPreflightsGPUAddress = 0u;
    std::uint64_t brainCommitPreflightElementCount = 0u;
    std::uint64_t preflightReadyEventValue = 0u;
    std::uint32_t brainCommitPreflightStride = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

using MetalNumanXHumanMatterReservePreparedApplication = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterProposalView& proposal,
    const MetalNumanXHumanMatterBrainPreflightView& preflight
) noexcept;

enum class MetalNumanXHumanMatterApplyMode : std::uint32_t {
    validateBrainAck = 0u,
    forceReject = 1u,
};

enum class MetalNumanXHumanMatterApplyTerminalStatus : std::uint32_t {
    acceptedPendingPublication = 1u,
    rejectedReleased = 2u,
    terminalNoTouch = 3u,
};

using MetalNumanXHumanMatterApplyCompletion = void (*)(
    void* context,
    MetalNumanXHumanMatterApplyTerminalStatus status,
    std::uint64_t slotGeneration
) noexcept;

struct MetalNumanXHumanMatterApplyPass {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterApplyPass);
    MetalNumanXHumanMatterApplyMode mode =
        MetalNumanXHumanMatterApplyMode::validateBrainAck;
    std::uint32_t reserved0 = 0u;
    void* commandBuffer = nullptr;
    void* brainAcks = nullptr;
    void* brainAckEvent = nullptr;
    std::uint64_t brainAcksGPUAddress = 0u;
    std::uint64_t brainAckElementCount = 0u;
    std::uint64_t brainAckEventValue = 0u;
    std::uint32_t brainAckStride = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

using MetalNumanXHumanMatterEncodePreparedApply = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterApplyPass& pass
) noexcept;

using MetalNumanXHumanMatterAbortPreparedApply = void (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterApplyPass& pass
) noexcept;

struct MetalNumanXHumanMatterPublicationFenceView {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanMatterPublicationFenceView);
    void* publicationFences = nullptr;
    std::uint64_t publicationFencesGPUAddress = 0u;
    std::uint64_t publicationFenceElementCount = 0u;
    std::uint32_t publicationFenceStride = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

struct MetalNumanXHumanMatterPublicationReservationView {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanMatterPublicationReservationView);
    MetalNumanXHumanMatterProposalView proposal{};
    MetalNumanXHumanMatterPublicationFenceView fence{};
    void* appliedOutcomes = nullptr;
    void* finalAcceptedPhysicsStateTokens = nullptr;
    std::uint64_t appliedOutcomesGPUAddress = 0u;
    std::uint64_t finalAcceptedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t appliedOutcomeElementCount = 0u;
    std::uint64_t finalAcceptedPhysicsStateTokenByteCount = 0u;
    std::uint32_t appliedOutcomeStride = 0u;
    std::uint32_t finalTokenStrideBytes = 0u;
    std::uint64_t jointCommitFingerprint = 0u;
    std::uint64_t brainGeneration = 0u;
    MetalNumanXHumanIOCandidatePublicationBinding humanIOBinding{};
};

using MetalNumanXHumanMatterReservePublishedRoot = bool (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterPublicationReservationView& reservation
) noexcept;

using MetalNumanXHumanMatterReleasePublishedRoot =
    MetalNumanXHumanMatterPrepareLeaseDisposition (*)(
    void* context,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterPublicationFenceView& fence
) noexcept;

struct MetalNumanXHumanMatterProgram {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterProgram);
    std::uint32_t capabilities = 0u;
    std::uint32_t accessFlags = 0u;

    void* context = nullptr;
    MetalNumanXHumanMatterEncode encode = nullptr;
    MetalNumanXHumanMatterAbort abort = nullptr;
    // Required ABI4 lifetime handshake. acquirePrepareLease must make the exact
    // adapter token slot immutable before the first physical command commits.
    // releasePrepareLease is called after an apply command completes, before
    // the Human owner clears any prepared state. Its explicit disposition is
    // then applied only after the owner reacquires and revalidates the exact
    // generation/apply attempt. It is called with completed=false for a GPU
    // error, pre-commit unwind, or terminal context teardown; that return value
    // cannot authorize publication or slot reuse.
    MetalNumanXHumanMatterAcquirePrepareLease acquirePrepareLease = nullptr;
    MetalNumanXHumanMatterBindHumanIOCandidatePublication
        bindHumanIOCandidatePublication = nullptr;
    MetalNumanXHumanMatterReleasePrepareLease releasePrepareLease = nullptr;
    // ABI4: proposal is mutation-free. The adapter reserves the exact proposal
    // generation before Brain ACK, consumes the owner apply action on the later
    // borrowed command buffer, writes its integrity-checked Matter outcome, and
    // releases an accepted root only after a COMMITTED publication fence.
    MetalNumanXHumanMatterReservePreparedApplication
        reservePreparedApplication = nullptr;
    MetalNumanXHumanMatterEncodePreparedApply encodePreparedApply = nullptr;
    MetalNumanXHumanMatterAbortPreparedApply abortPreparedApply = nullptr;
    MetalNumanXHumanMatterReservePublishedRoot reservePublishedRoot = nullptr;
    MetalNumanXHumanMatterReleasePublishedRoot releasePublishedRoot = nullptr;
    std::uint64_t fingerprint = 0u;

    void* matterGeneralizedReaction = nullptr;
    void* jointStatuses = nullptr;
    void* acceptedPhysicsStateTokens = nullptr;
    void* matterApplyOutcomes = nullptr;
    std::uint64_t matterGeneralizedReactionGPUAddress = 0u;
    std::uint64_t jointStatusesGPUAddress = 0u;
    std::uint64_t acceptedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t matterApplyOutcomesGPUAddress = 0u;
    std::uint64_t matterGeneralizedReactionElementCount = 0u;
    std::uint64_t jointStatusElementCount = 0u;
    std::uint64_t acceptedPhysicsStateTokenByteCount = 0u;
    std::uint64_t matterApplyOutcomeElementCount = 0u;

    std::uint32_t environmentCount = 0u;
    std::uint32_t reactionStride = 0u;
    std::uint32_t jointStatusStride = 0u;
    std::uint32_t acceptedTokenStrideBytes = 0u;
    std::uint32_t matterApplyOutcomeStride = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t substepIndex = 0u;
    // The current owner ABI admits exactly one physics substep and therefore
    // one start-of-control checkpoint. A future multi-substep ABI must retain
    // one prepared root across all substeps instead of publishing per-substep.
    std::uint32_t physicsSubstepCount = 0u;
    // Maximum attachment-query suffix admitted by the owner. The private
    // generic-kinematics stream also carries four immutable probes per Human
    // body, so validation requires 4*bodyCount+candidatePointCapacity <= the
    // generic operator point ceiling. Zero remains valid for body-only exact
    // candidate calls.
    std::uint32_t candidatePointCapacity = 0u;

    // Global control-root identity supplied by the adapter. Zero is a valid
    // first root; every later pass/lease/witness/applied record must match it.
    std::uint32_t controlStep = 0u;
    // Logical articulation shape. The adapter's reaction arena remains
    // capacity-strided, while live q/v/A0 are exact logical strides.
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t dofLayoutVersion = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;

    [[nodiscard]] bool valid() const noexcept {
        constexpr std::uint32_t requiredCapabilities =
            MetalNumanXHumanMatterSourceEffectiveTangent |
            MetalNumanXHumanMatterStagedReaction |
            MetalNumanXHumanMatterJointDecision |
            MetalNumanXHumanMatterPreparedPhysicsGate;
        constexpr std::uint32_t requiredAccess =
            MetalNumanXHumanMatterReadLiveHumanState |
            MetalNumanXHumanMatterReadHumanCheckpoints |
            MetalNumanXHumanMatterReadSourceEffectiveTangent |
            MetalNumanXHumanMatterWriteStagedReaction |
            MetalNumanXHumanMatterWriteJointStatus |
            MetalNumanXHumanMatterWritePreparedPhysicsToken;
        const bool exactCapability =
            (capabilities &
             MetalNumanXHumanMatterExactCandidateKinematics) != 0u;
        const bool exactAccess =
            (accessFlags &
             MetalNumanXHumanMatterMayEncodeExactCandidate) != 0u;
        return abiVersion == kMetalNumanXHumanMatterABIVersion &&
            structSize == sizeof(MetalNumanXHumanMatterProgram) &&
            (capabilities &
             ~kMetalNumanXHumanMatterKnownCapabilities) == 0u &&
            (capabilities & requiredCapabilities) == requiredCapabilities &&
            (accessFlags & ~kMetalNumanXHumanMatterKnownAccess) == 0u &&
            (accessFlags & requiredAccess) == requiredAccess &&
            exactCapability == exactAccess &&
            (exactCapability
                 ? candidatePointCapacity <=
                       MR_ARTICULATED_OPERATOR_MAX_POINTS
                 : candidatePointCapacity == 0u) &&
            context != nullptr && encode != nullptr && abort != nullptr &&
            acquirePrepareLease != nullptr &&
            bindHumanIOCandidatePublication != nullptr &&
            releasePrepareLease != nullptr &&
            reservePreparedApplication != nullptr &&
            encodePreparedApply != nullptr && abortPreparedApply != nullptr &&
            reservePublishedRoot != nullptr &&
            releasePublishedRoot != nullptr &&
            fingerprint != 0u && transactionFingerprint != 0u &&
            linearizationEpoch != 0u && slotGeneration != 0u &&
            dofLayoutVersion == kMetalNumanXHumanMatterDofLayoutVersion &&
            dofCount != 0u &&
            dofCount <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
            qCoordinateCount == dofCount + 1u &&
            qCoordinateCount <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
            physicsSubstepCount == 1u && substepIndex == 0u &&
            environmentCount == 1u &&
            matterGeneralizedReaction != nullptr && jointStatuses != nullptr &&
            acceptedPhysicsStateTokens != nullptr &&
            matterApplyOutcomes != nullptr &&
            matterGeneralizedReactionGPUAddress != 0u &&
            jointStatusesGPUAddress != 0u &&
            acceptedPhysicsStateTokensGPUAddress != 0u &&
            matterApplyOutcomesGPUAddress != 0u &&
            reactionStride == MR_NUMI_HUMAN_STAND_MAX_DOFS &&
            jointStatusStride != 0u &&
            acceptedTokenStrideBytes ==
                MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES &&
            matterApplyOutcomeStride != 0u &&
            matterGeneralizedReactionElementCount >=
                static_cast<std::uint64_t>(environmentCount) *
                    reactionStride &&
            jointStatusElementCount >=
                static_cast<std::uint64_t>(environmentCount - 1u) *
                    jointStatusStride + 1u &&
            acceptedPhysicsStateTokenByteCount >=
                static_cast<std::uint64_t>(environmentCount) *
                    acceptedTokenStrideBytes &&
            matterApplyOutcomeElementCount >=
                static_cast<std::uint64_t>(environmentCount - 1u) *
                    matterApplyOutcomeStride + 1u;
    }

    [[nodiscard]] bool configured() const noexcept {
        return abiVersion != kMetalNumanXHumanMatterABIVersion ||
            structSize != sizeof(MetalNumanXHumanMatterProgram) ||
            capabilities != 0u || accessFlags != 0u || context != nullptr ||
            encode != nullptr || abort != nullptr ||
            acquirePrepareLease != nullptr ||
            bindHumanIOCandidatePublication != nullptr ||
            releasePrepareLease != nullptr ||
            reservePreparedApplication != nullptr ||
            encodePreparedApply != nullptr || abortPreparedApply != nullptr ||
            reservePublishedRoot != nullptr ||
            releasePublishedRoot != nullptr || fingerprint != 0u ||
            matterGeneralizedReaction != nullptr || jointStatuses != nullptr ||
            acceptedPhysicsStateTokens != nullptr ||
            matterApplyOutcomes != nullptr ||
            matterGeneralizedReactionGPUAddress != 0u ||
            jointStatusesGPUAddress != 0u ||
            acceptedPhysicsStateTokensGPUAddress != 0u ||
            matterApplyOutcomesGPUAddress != 0u ||
            matterGeneralizedReactionElementCount != 0u ||
            jointStatusElementCount != 0u ||
            acceptedPhysicsStateTokenByteCount != 0u ||
            matterApplyOutcomeElementCount != 0u ||
            environmentCount != 0u || reactionStride != 0u ||
            jointStatusStride != 0u || acceptedTokenStrideBytes != 0u ||
            matterApplyOutcomeStride != 0u ||
            qCoordinateCount != 0u || dofCount != 0u ||
            dofLayoutVersion != 0u ||
            transactionSlot != 0u || substepIndex != 0u ||
            physicsSubstepCount != 0u ||
            candidatePointCapacity != 0u ||
            controlStep != 0u ||
            transactionFingerprint != 0u || linearizationEpoch != 0u ||
            slotGeneration != 0u;
    }
};

enum class MetalNumanXHumanMatterProposalMode : std::uint32_t {
    validateBrainWitness = 0u,
    forceReject = 1u,
};

struct MetalNumanXHumanMatterProposalRequest {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterProposalRequest);
    MetalNumanXHumanMatterProposalMode mode =
        MetalNumanXHumanMatterProposalMode::validateBrainWitness;
    std::uint32_t reserved1 = 0u;
    void* commandBuffer = nullptr;
    void* brainCommitWitnesses = nullptr;
    void* brainPrepareCompleteEvent = nullptr;
    std::uint64_t brainPrepareCompleteEventValue = 0u;
    std::uint64_t brainCommitWitnessesGPUAddress = 0u;
    std::uint64_t brainCommitWitnessElementCount = 0u;
    std::uint32_t brainCommitWitnessStride = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

struct MetalNumanXHumanMatterApplyRequest {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterApplyRequest);
    MetalNumanXHumanMatterApplyMode mode =
        MetalNumanXHumanMatterApplyMode::validateBrainAck;
    std::uint32_t reserved0 = 0u;
    void* commandBuffer = nullptr;
    void* brainAcks = nullptr;
    void* brainAckEvent = nullptr;
    // Exactly once after command completion and adapter disposition, never at
    // GPU event-signal time. Resources remain quarantined through callback
    // return. abortApply on an uncommitted command produces no callback.
    void* completionContext = nullptr;
    MetalNumanXHumanMatterApplyCompletion completion = nullptr;
    std::uint64_t brainAckEventValue = 0u;
    std::uint64_t brainAcksGPUAddress = 0u;
    std::uint64_t brainAckElementCount = 0u;
    std::uint32_t brainAckStride = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

struct MetalNumanXHumanMatterPublicationReservationRequest {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanMatterPublicationReservationRequest);
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t jointCommitFingerprint = 0u;
    std::uint64_t brainGeneration = 0u;
};

struct MetalNumanXHumanMatterPublicationReleaseRequest {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(
        MetalNumanXHumanMatterPublicationReleaseRequest);
    void* publicationFences = nullptr;
    std::uint64_t publicationFencesGPUAddress = 0u;
    std::uint64_t publicationFenceElementCount = 0u;
    std::uint32_t publicationFenceStride = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t jointCommitFingerprint = 0u;
    std::uint64_t brainGeneration = 0u;
};

struct MetalNumanXHumanMatterPreparedView {
    std::uint32_t abiVersion = kMetalNumanXHumanMatterABIVersion;
    std::uint32_t structSize = sizeof(MetalNumanXHumanMatterPreparedView);
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t finalTokenStrideBytes = 0u;

    std::uint32_t proposalStride = 0u;
    std::uint32_t proposedTokenStrideBytes = 0u;
    std::uint32_t applyActionStride = 0u;
    std::uint32_t matterApplyOutcomeStride = 0u;
    std::uint32_t appliedOutcomeStride = 0u;
    std::uint32_t publicationFenceStride = 0u;

    // Low-level transaction SPI. All arenas are persistent exact same-device
    // MTLBuffer identities retained by the prepared capability. Applied bytes
    // are quarantined and are not externally authoritative before an exact
    // COMMITTED publication fence and successful releasePublishedRoot.
    // physicalPreparedEvent is owner-produced only: consumers wait for the
    // physicalPreparedEventValue before Brain preparation and wait for
    // proposalEventValue / appliedEventValue are liveness order only; every
    // consumer must also validate the corresponding status and fingerprint.
    void* preparedPhysicsStateTokens = nullptr;
    void* finalAcceptedPhysicsStateTokens = nullptr;
    void* proposals = nullptr;
    void* proposedPhysicsStateTokens = nullptr;
    void* applyActions = nullptr;
    void* matterApplyOutcomes = nullptr;
    void* appliedOutcomes = nullptr;
    void* publicationFences = nullptr;
    void* physicalPreparedEvent = nullptr;
    std::uint64_t preparedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t finalAcceptedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t proposalsGPUAddress = 0u;
    std::uint64_t proposedPhysicsStateTokensGPUAddress = 0u;
    std::uint64_t applyActionsGPUAddress = 0u;
    std::uint64_t matterApplyOutcomesGPUAddress = 0u;
    std::uint64_t appliedOutcomesGPUAddress = 0u;
    std::uint64_t publicationFencesGPUAddress = 0u;
    std::uint64_t preparedPhysicsStateTokenByteCount = 0u;
    std::uint64_t finalAcceptedPhysicsStateTokenByteCount = 0u;
    std::uint64_t proposalElementCount = 0u;
    std::uint64_t proposedPhysicsStateTokenByteCount = 0u;
    std::uint64_t applyActionElementCount = 0u;
    std::uint64_t matterApplyOutcomeElementCount = 0u;
    std::uint64_t appliedOutcomeElementCount = 0u;
    std::uint64_t publicationFenceElementCount = 0u;
    std::uint64_t physicalPreparedEventValue = 0u;
    std::uint64_t proposalEventValue = 0u;
    std::uint64_t appliedEventValue = 0u;

    std::uint32_t controlStep = 0u;
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t dofLayoutVersion = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    // Populated only after the post-physical one-shot candidate bind. These
    // are pointer-free identities; the callback program remains private to the
    // quarantined prepared lease.
    std::uint64_t humanIOCandidateKeyFingerprint = 0u;
    std::uint64_t acceptedBrainGeneration = 0u;
    std::uint64_t humanIOSensorGeneration = 0u;
    std::uint64_t humanIOProgramFingerprint = 0u;
    std::uint64_t humanIOSensorFingerprint = 0u;
    std::uint64_t humanIOTransactionInstanceFingerprint = 0u;
    std::uint64_t humanIOCandidatePublicationFingerprint = 0u;
    std::uint64_t humanIODeviceRegistryID = 0u;
    std::uint64_t humanIOIdentityFingerprint = 0u;
};

enum class MetalNumanXHumanMatterOperationStatus : std::uint32_t {
    success = 0u,
    invalidHandle,
    invalidRequest,
    alreadyFinalizing,
    metalEncoderFailure,
    terminalNoTouch,
};

struct MetalNumanXHumanMatterOperationDiagnostics {
    MetalNumanXHumanMatterOperationStatus status =
        MetalNumanXHumanMatterOperationStatus::success;
    bool encoded = false;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalNumanXHumanMatterOperationStatus::success;
    }
};

enum class MetalNumanXHumanMatterProposalCompletionStatus : std::uint32_t {
    ready = 1u,
    terminalNoTouch = 2u,
};

enum class MetalNumanXHumanMatterPhysicalCompletionStatus : std::uint32_t {
    ready = 1u,
    terminalNoTouch = 2u,
};

// One-shot borrowed notification for the original physical command. Owner
// state and the device-written physical status are settled before delivery;
// the owner mutex is never held while the callback runs. Late registration
// invokes synchronously before returning.
using MetalNumanXHumanMatterPhysicalCompletion = void (*)(
    void* context,
    MetalNumanXHumanMatterPhysicalCompletionStatus status,
    std::uint64_t slotGeneration
) noexcept;

// One-shot borrowed callback for the exact prepared generation. The owner
// mutex is not held while it runs. A READY callback may synchronously reserve
// the prepared application from inside the callback.
using MetalNumanXHumanMatterProposalCompletion = void (*)(
    void* context,
    MetalNumanXHumanMatterProposalCompletionStatus status,
    std::uint64_t slotGeneration
) noexcept;

// Move-only capability for one quarantined physical prepare. It adopts the
// original submission and retains owner checkpoints without retaining any old
// borrowed pass or command buffer. Proposal is mutation-free; apply can mutate
// Human/Matter only after exact Brain ACK. ACCEPT remains quarantined until a
// COMMITTED publication fence is released. Any terminal-no-touch disposition
// permanently prevents restore, publication, and slot reuse.
class MetalNumanXHumanMatterPrepared {
public:
    MetalNumanXHumanMatterPrepared() noexcept;
    ~MetalNumanXHumanMatterPrepared();
    MetalNumanXHumanMatterPrepared(
        MetalNumanXHumanMatterPrepared&& other
    ) noexcept;
    MetalNumanXHumanMatterPrepared& operator=(
        MetalNumanXHumanMatterPrepared&& other
    ) noexcept;
    MetalNumanXHumanMatterPrepared(
        const MetalNumanXHumanMatterPrepared&
    ) = delete;
    MetalNumanXHumanMatterPrepared& operator=(
        const MetalNumanXHumanMatterPrepared&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool view(
        MetalNumanXHumanMatterPreparedView& output
    ) const noexcept;
    [[nodiscard]] bool encodeWaitForPhysicalPrepare(
        void* commandBuffer
    ) noexcept;
    [[nodiscard]] bool registerPhysicalCompletion(
        void* completionContext,
        MetalNumanXHumanMatterPhysicalCompletion completion
    ) noexcept;
    // Installs the exact unpublished HumanIO sensor candidate after the first
    // physical command has completed successfully. Legal exactly once and
    // only before proposal; wrong, stale, duplicate, or late capabilities are
    // rejected without replacing the installed generation.
    [[nodiscard]] bool bindHumanIOCandidatePublication(
        const MetalNumanXHumanIOCandidatePublicationProgram& candidate
    ) noexcept;
    [[nodiscard]] MetalNumanXHumanMatterOperationDiagnostics
    proposePrepared(
        const MetalNumanXHumanMatterProposalRequest& request
    ) noexcept;
    // Registers after proposePrepared succeeds. Internal proposal state and
    // the owner completion event are settled before the callback. If the
    // proposal command already completed, registration invokes synchronously.
    [[nodiscard]] bool registerProposalCompletion(
        void* completionContext,
        MetalNumanXHumanMatterProposalCompletion completion
    ) noexcept;
    [[nodiscard]] bool abortProposal(void* commandBuffer) noexcept;
    // Called after proposal-command completion and before the Brain ACK is
    // submitted. It performs the adapter's fallible exact-generation
    // reservation; no later ACCEPT ACK is admissible without it.
    [[nodiscard]] bool reservePreparedApplication(
        const MetalNumanXHumanMatterBrainPreflightView& preflight
    ) noexcept;
    [[nodiscard]] MetalNumanXHumanMatterOperationDiagnostics
    applyPrepared(
        const MetalNumanXHumanMatterApplyRequest& request
    ) noexcept;
    [[nodiscard]] bool abortApply(void* commandBuffer) noexcept;
    // Prevalidates the exact owner-owned PENDING fence range and the completed
    // applied-output range before any Brain pointer flip.
    [[nodiscard]] bool reservePublishedRoot(
        const MetalNumanXHumanMatterPublicationReservationRequest& request
    ) noexcept;
    // Nonthrow release after the caller overwrites the reserved fence with the
    // exact COMMITTED record. Only adapter `released` clears the quarantine.
    [[nodiscard]] MetalNumanXHumanMatterPrepareLeaseDisposition
    releasePublishedRoot(
        const MetalNumanXHumanMatterPublicationReleaseRequest& request
    ) noexcept;

private:
    friend class MetalArticulatedOperatorContext;
    friend class MetalArticulatedOperatorSubmission;
    std::shared_ptr<detail::MetalArticulatedOperatorContextState> state_;
    std::shared_ptr<detail::MetalNumanXHumanMatterPreparedState> capability_;
    std::uint64_t slotGeneration_ = 0u;
};

// Persistent large-state Human horizon. This is deliberately coupled to the
// MyoSim sidecar: each device step recomputes source routes and J^T muscle
// force from the current q before advancing activation and dynamics. Support
// contacts address source-authored point queries in the enclosing stream.
// Root assistance is a world wrench on the floating base, never joint torque.
struct MetalNumiHumanStandInput {
    std::span<const float> v{};
    std::span<const MRNumiHumanStandContactGPU> contacts{};
    // Exact scalar joint manifold imported from the source model. These rows
    // carry bilateral reaction impulses during dynamics; dependent q/v are
    // projected back onto the same polynomial after each accepted step.
    std::span<const MRNumiHumanJointEqualityGPU> jointEqualities{};
    // Optional NHTENDON2/3 program. When present, one terminal-load transaction
    // executes from the current MyoSim force field before every dynamics step.
    // Transfer alone retains MyoSim's original J^T wrench. A configured
    // two-phase consumer may replace only its explicitly declared share with
    // a solved continuum reaction before dynamics. Transfer generalized
    // corrections remain diagnostics and are never added as direct torque.
    std::span<const MRNumiHumanTendonBindingGPU> tendonBindings{};
    std::span<const MRNumiHumanTendonEnvelopeGPU> tendonEnvelopes{};
    MetalNumiHumanTendonLoadProgram tendonLoadProgram{};
    MetalNumanXTransactionProgram numanXTransactionProgram{};
    MetalNumanXHumanMatterProgram numanXHumanMatterProgram{};
    std::uint32_t stepCount = 0u;
    std::uint32_t contactIterationCount = 12u;
    bool enableContact = true;
    bool enableRootAssistance = false;
    mr_float4 groundPoint{0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 groundNormal{0.0f, 1.0f, 0.0f, 0.0f};
    mr_float4 targetRootPosition{0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 targetRootOrientation{0.0f, 0.0f, 0.0f, 1.0f};
    // linear stiffness, linear damping, angular stiffness, angular damping.
    mr_float4 assistanceGains{0.0f, 0.0f, 0.0f, 0.0f};

    [[nodiscard]] bool enabled() const noexcept {
        return stepCount != 0u;
    }
};

// Exact authorization to advance the device-resident state published by the
// preceding NumanX Human/Matter root. A valid continuation never supplies a
// host copy of q, v, or MyoSim state: those three streams remain in the
// context-owned Metal arena and are consumed in place. The owner records this
// identity only after a canonical accepted root has crossed the joint
// publication fence. Once resident state exists, an implicit host reset is
// rejected; callers must either present the exact preceding identity or use a
// fresh context.
struct MetalArticulatedOperatorResidentStateContinuation {
    std::uint64_t previousTransactionFingerprint = 0u;
    std::uint64_t previousPhysicsGeneration = 0u;

    [[nodiscard]] bool configured() const noexcept {
        return previousTransactionFingerprint != 0u ||
            previousPhysicsGeneration != 0u;
    }

    [[nodiscard]] bool valid() const noexcept {
        return previousTransactionFingerprint != 0u &&
            previousPhysicsGeneration != 0u;
    }
};

// Packed, environment-major input for the synchronous Metal articulated
// operator. q contains environmentCount * articulation.nq floats. points
// contains environmentCount * pointCount records. All query body indices are
// global EngineModel body indices. A Millard source program indexes `points`
// directly and therefore retains one authoritative device kinematics stream.
struct MetalArticulatedOperatorInput {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::size_t pointCount = 0u;
    std::span<const float> q{};
    // Optional environment-major articulation velocity. Required only when a
    // caller needs nonzero MyoSim path velocity; stand horizons source the
    // current device-resident velocity sidecar directly.
    std::span<const float> v{};
    std::span<const MRArticulatedPointImpulseGPU> points{};
    MetalMillardReferenceInput millard{};
    MetalMujocoMuscleReferenceInput mujoco{};
    MetalNumiHumanStandInput stand{};
    MetalArticulatedOperatorResidentStateContinuation residentContinuation{};
};

struct MetalArticulatedOperatorConfig {
    // The dense mass matrix is a correctness diagnostic. The factor-backed
    // impulse solve always runs; disabling this avoids its output bandwidth.
    bool writeDiagnosticMassMatrix = false;
    // Emits body/point kinematics and analytic point Jacobians without
    // assembling or factorizing the mass matrix. Generalized impulse and
    // delta-velocity outputs are deterministically zero. This mode is used by
    // the multi-articulation contact frontend before batched inverse ABA.
    bool pointJacobiansOnly = false;
    // When non-zero, advance each valid MyoSim activation after its force
    // reference and reduction pass in the same Metal command buffer. This is
    // one explicit-Euler step in seconds; callers retain the returned state
    // sidecar and feed it into their next transaction. Zero preserves the
    // historical force-only behavior.
    float mujocoActivationTimestepSeconds = 0.0f;
    // Empty discovers the co-installed metallib relative to the loaded
    // MetalRobo dylib, with the configured build-tree path as a fallback.
    // A non-empty path is an explicit trusted ABI-compatible override.
    std::string metallibPath;
};

enum class MetalArticulatedOperatorHostStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    capacityOverflow,
    arithmeticOverflow,
    nonfiniteInput,
    invalidPointQuery,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    gpuEnvironmentFailure,
    internalFailure,
    contextBusy,
    externalProgramFailure,
};

// The host always derives compact strides; callers cannot smuggle unchecked
// buffer layouts into the raw kernel. Counts and byte sizes are retained so a
// batch can be audited without reconstructing overflow-sensitive arithmetic.
struct MetalArticulatedOperatorLayout {
    MRArticulatedOperatorDispatchGPU dispatch{};
    std::size_t qElements = 0u;
    std::size_t qBytes = 0u;
    std::size_t pointElements = 0u;
    std::size_t pointBytes = 0u;
    std::size_t bodyPoseElements = 0u;
    std::size_t bodyPoseBytes = 0u;
    std::size_t pointWorldElements = 0u;
    std::size_t pointWorldBytes = 0u;
    std::size_t massMatrixElements = 0u;
    std::size_t massMatrixBytes = 0u;
    std::size_t pointJacobianElements = 0u;
    std::size_t pointJacobianBytes = 0u;
    std::size_t generalizedElements = 0u;
    std::size_t generalizedBytes = 0u;
    std::size_t statusElements = 0u;
    std::size_t statusBytes = 0u;
    std::size_t millardMuscleElements = 0u;
    std::size_t millardMuscleBytes = 0u;
    std::size_t millardStateElements = 0u;
    std::size_t millardStateBytes = 0u;
    std::size_t millardPathPointElements = 0u;
    std::size_t millardPathPointBytes = 0u;
    std::size_t millardCurveElements = 0u;
    std::size_t millardCurveBytes = 0u;
    std::size_t millardWrapElements = 0u;
    std::size_t millardWrapBytes = 0u;
    std::size_t millardResultElements = 0u;
    std::size_t millardResultBytes = 0u;
    std::size_t millardGeneralizedForceElements = 0u;
    std::size_t millardGeneralizedForceBytes = 0u;
    std::size_t mujocoMuscleElements = 0u;
    std::size_t mujocoMuscleBytes = 0u;
    std::size_t mujocoStateElements = 0u;
    std::size_t mujocoStateBytes = 0u;
    std::size_t mujocoSiteElements = 0u;
    std::size_t mujocoSiteBytes = 0u;
    std::size_t mujocoWrapElements = 0u;
    std::size_t mujocoWrapBytes = 0u;
    std::size_t mujocoRouteNodeElements = 0u;
    std::size_t mujocoRouteNodeBytes = 0u;
    std::size_t mujocoResultElements = 0u;
    std::size_t mujocoResultBytes = 0u;
    std::size_t mujocoMuscleGeneralizedForceElements = 0u;
    std::size_t mujocoMuscleGeneralizedForceBytes = 0u;
    std::size_t mujocoGeneralizedForceElements = 0u;
    std::size_t mujocoGeneralizedForceBytes = 0u;
    // Private suballocation in the mutually-exclusive source-muscle force
    // workspace: per-muscle rows followed by their environment reduction.
    std::size_t mujocoForceWorkspaceElements = 0u;
    std::size_t standVelocityElements = 0u;
    std::size_t standVelocityBytes = 0u;
    std::size_t standContactElements = 0u;
    std::size_t standContactBytes = 0u;
    std::size_t standJointEqualityElements = 0u;
    std::size_t standJointEqualityBytes = 0u;
    std::size_t standSpatialJacobianElements = 0u;
    std::size_t standBodyMotionElements = 0u;
    std::size_t standFactorElements = 0u;
    std::size_t standVectorElements = 0u;
    std::size_t standResponseElements = 0u;
    std::size_t standScratchBytes = 0u;
    std::size_t standStatusElements = 0u;
    std::size_t standStatusBytes = 0u;
    std::size_t standTendonBindingElements = 0u;
    std::size_t standTendonBindingBytes = 0u;
    std::size_t standTendonEnvelopeElements = 0u;
    std::size_t standTendonEnvelopeBytes = 0u;
    std::size_t standTendonTransferElements = 0u;
    std::size_t standTendonTransferBytes = 0u;
    std::size_t standTendonCorrectionElements = 0u;
    std::size_t standTendonCorrectionBytes = 0u;
    // Private owner-transaction arena. These bytes exist only when a valid
    // MetalNumanXHumanMatterProgram is attached to a one-step stand input.
    std::size_t humanMatterQCheckpointElements = 0u;
    std::size_t humanMatterQCheckpointBytes = 0u;
    std::size_t humanMatterVCheckpointElements = 0u;
    std::size_t humanMatterVCheckpointBytes = 0u;
    std::size_t humanMatterMujocoCheckpointElements = 0u;
    std::size_t humanMatterMujocoCheckpointBytes = 0u;
    std::size_t humanMatterSourceFactorElements = 0u;
    std::size_t humanMatterSourceFactorBytes = 0u;
    std::size_t humanMatterOwnerStatusElements = 0u;
    std::size_t humanMatterOwnerStatusBytes = 0u;
    std::size_t humanMatterCandidatePointElements = 0u;
    std::size_t humanMatterCandidatePointBytes = 0u;
    std::size_t humanMatterCandidateBodyPoseElements = 0u;
    std::size_t humanMatterCandidateBodyPoseBytes = 0u;
    std::size_t humanMatterCandidatePointWorldElements = 0u;
    std::size_t humanMatterCandidatePointWorldBytes = 0u;
    std::size_t humanMatterCandidatePointJacobianElements = 0u;
    std::size_t humanMatterCandidatePointJacobianBytes = 0u;
    std::size_t humanMatterCandidateOperatorScratchElements = 0u;
    std::size_t humanMatterCandidateOperatorScratchBytes = 0u;
    std::size_t humanMatterCandidateOperatorStatusElements = 0u;
    std::size_t humanMatterCandidateOperatorStatusBytes = 0u;
    std::size_t humanMatterProposalElements = 0u;
    std::size_t humanMatterProposalBytes = 0u;
    std::size_t humanMatterProposedTokenElements = 0u;
    std::size_t humanMatterProposedTokenBytes = 0u;
    std::size_t humanMatterApplyActionElements = 0u;
    std::size_t humanMatterApplyActionBytes = 0u;
    std::size_t humanMatterAppliedOutcomeElements = 0u;
    std::size_t humanMatterAppliedOutcomeBytes = 0u;
    std::size_t humanMatterFinalAcceptedTokenElements = 0u;
    std::size_t humanMatterFinalAcceptedTokenBytes = 0u;
    std::size_t humanMatterPublicationFenceElements = 0u;
    std::size_t humanMatterPublicationFenceBytes = 0u;
    // Includes immutable model buffers and one-element placeholders required
    // to bind logically empty Metal buffers.
    std::size_t totalAllocatedBytes = 0u;
};

struct MetalArticulatedOperatorResult {
    MetalArticulatedOperatorLayout layout{};
    std::vector<MRArticulatedBodyPoseGPU> bodyPoses;
    std::vector<MRArticulatedPointWorldGPU> pointWorld;
    std::vector<float> diagnosticMassMatrix;
    std::vector<float> pointJacobians;
    std::vector<float> generalizedImpulse;
    std::vector<float> deltaVelocity;
    std::vector<MRArticulatedOperatorStatusGPU> statuses;
    std::vector<MRMillardMuscleResultGPU> millardResults;
    std::vector<float> millardGeneralizedForces;
    std::vector<MRMujocoMuscleResultGPU> mujocoResults;
    // Device-published activation sidecar after the optional explicit step.
    // Excitation is preserved; invalid reference records retain their prior
    // state so a caller can decide whether to retry or reject the transaction.
    std::vector<MRMujocoMuscleStateGPU> mujocoActivationStates;
    // Device-produced [environment][muscle][dof] source contributions and
    // their deterministic [environment][dof] reduction.
    std::vector<float> mujocoMuscleGeneralizedForces;
    std::vector<float> mujocoGeneralizedForces;
    // Final device state and cumulative horizon diagnostics. Empty for the
    // historical one-pass operator path.
    std::vector<float> standQ;
    std::vector<float> standV;
    std::vector<MRNumiHumanStandStatusGPU> standStatuses;
    // Final accepted step's exact endpoint-to-node transaction and its
    // wrench-equivalence diagnostic. Earlier steps remain device-resident and
    // are available to an optional per-step borrowed consumer.
    std::vector<MRNumiHumanTendonTransferResultGPU> standTendonTransfers;
    std::vector<float> standTendonGeneralizedCorrections;
};

struct MetalArticulatedOperatorDiagnostics {
    MetalArticulatedOperatorHostStatus status =
        MetalArticulatedOperatorHostStatus::success;
    MetalArticulatedOperatorLayout layout{};
    bool dispatched = false;
    bool published = false;
    std::uint32_t successfulEnvironmentCount = 0u;
    std::uint32_t failedEnvironmentCount = 0u;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode =
        MR_ARTICULATED_OPERATOR_SUCCESS;
    std::uint32_t completedStandSteps = 0u;
    std::uint32_t firstStandGPUStatusCode =
        MR_NUMI_HUMAN_STAND_SUCCESS;
    // Exact owning command-buffer and external-program identities. Downstream
    // candidate publishers must match both before accepting a zero-copy view.
    std::uint64_t numanXProgramFingerprint = 0u;
    std::uintptr_t commandBufferIdentity = 0u;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalArticulatedOperatorHostStatus::success;
    }
};

// Lifetime counters for a reusable operator context. retainedBufferBytes is
// the physical capacity of the grow-only Metal buffer arena; an individual
// result layout continues to report only that batch's required bytes.
struct MetalArticulatedOperatorContextStats {
    std::uint64_t pipelineCreationCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    // Submissions that consumed q/v/MyoSim directly from the exact preceding
    // published root instead of uploading a host reset image.
    std::uint64_t residentContinuationSubmissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    // Destruction fallback waits only while Metal still reports nonterminal
    // work. Prepared ACCEPT/REJECT and late terminal cleanup increment the
    // nonwaiting counter instead.
    std::uint64_t submissionDestructorWaitCount = 0u;
    std::uint64_t terminalSubmissionNonwaitingReapCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    bool hasInFlightSubmission = false;
};

class MetalArticulatedOperatorContext;

// A committed Metal batch. submit() copies all caller-owned spans before
// returning, so their storage can immediately be reused. wait() may be called
// once and transactionally publishes the completed typed result. Destroying a
// live submission waits for the GPU and discards its result, making context
// and submission destruction safe in either order.
class MetalArticulatedOperatorSubmission {
public:
    MetalArticulatedOperatorSubmission() noexcept;
    ~MetalArticulatedOperatorSubmission();

    MetalArticulatedOperatorSubmission(
        MetalArticulatedOperatorSubmission&& other
    ) noexcept;
    MetalArticulatedOperatorSubmission& operator=(
        MetalArticulatedOperatorSubmission&& other
    ) noexcept;

    MetalArticulatedOperatorSubmission(
        const MetalArticulatedOperatorSubmission&
    ) = delete;
    MetalArticulatedOperatorSubmission& operator=(
        const MetalArticulatedOperatorSubmission&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;

    // Transfers this submission into the prepared capability without waiting
    // on the first command buffer. On success this submission becomes empty;
    // the capability (and the final-CB completion handler) jointly own the
    // submission until the event-ordered final command completes.
    [[nodiscard]] bool extractPreparedHumanMatter(
        MetalNumanXHumanMatterPrepared& output
    ) noexcept;

    [[nodiscard]] MetalArticulatedOperatorDiagnostics wait(
        MetalArticulatedOperatorResult& result
    );

private:
    friend class MetalArticulatedOperatorContext;
    std::unique_ptr<
        detail::MetalArticulatedOperatorSubmissionState
    > state_;
};

// Reusable execution context for steady-state simulation. Device discovery,
// metallib loading, command-queue creation, and pipeline compilation happen at
// most once. Its fixed-binding buffer arena is reused and grows geometrically as batch
// sizes increase. Calls are thread-safe, but a context deliberately admits
// only one in-flight submission because every batch shares the same arena;
// independent contexts provide safe overlap when multiple queues are useful.
class MetalArticulatedOperatorContext {
public:
    explicit MetalArticulatedOperatorContext(
        MetalArticulatedOperatorConfig config = {}
    );
    ~MetalArticulatedOperatorContext();

    MetalArticulatedOperatorContext(
        MetalArticulatedOperatorContext&& other
    ) noexcept;
    MetalArticulatedOperatorContext& operator=(
        MetalArticulatedOperatorContext&& other
    ) noexcept;

    MetalArticulatedOperatorContext(
        const MetalArticulatedOperatorContext&
    ) = delete;
    MetalArticulatedOperatorContext& operator=(
        const MetalArticulatedOperatorContext&
    ) = delete;

    // Encodes and commits without waiting for GPU completion. Host validation,
    // capacity growth, and all input copies are complete before this returns.
    // submission must be empty and remains unchanged on rejection.
    [[nodiscard]] MetalArticulatedOperatorDiagnostics submit(
        const EngineModel& model,
        const MetalArticulatedOperatorInput& input,
        MetalArticulatedOperatorSubmission& submission
    );

    // Convenience synchronous path using submit() followed by wait().
    [[nodiscard]] MetalArticulatedOperatorDiagnostics run(
        const EngineModel& model,
        const MetalArticulatedOperatorInput& input,
        MetalArticulatedOperatorResult& result
    );

    [[nodiscard]] MetalArticulatedOperatorContextStats stats()
        const noexcept;

private:
    std::shared_ptr<
        detail::MetalArticulatedOperatorContextState
    > state_;
};

// Executes one synchronous command buffer. The EngineModel and every layout,
// span, query, required element count, byte count, Metal allocation, and
// pipeline capability are checked before encoding. The raw kernel binding
// table remains private to the Objective-C++ implementation.
//
// Host-side failure leaves result bit-for-bit unchanged. Once a command
// completes, the entire typed batch and its per-environment GPU statuses are
// published together. A GPU environment failure therefore returns
// gpuEnvironmentFailure with published=true so the status stream remains
// inspectable; failed environments have zeroed payload slots.
[[nodiscard]] MetalArticulatedOperatorDiagnostics
runMetalArticulatedOperator(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorResult& result,
    const MetalArticulatedOperatorConfig& config = {}
);

[[nodiscard]] const char* metalArticulatedOperatorHostStatusName(
    MetalArticulatedOperatorHostStatus status
) noexcept;

} // namespace metalrobo
