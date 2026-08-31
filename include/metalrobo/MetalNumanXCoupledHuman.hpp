#pragma once

#include "metalrobo/numanx_coupled_human_gpu.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace metalrobo {

namespace detail {
struct MetalNumanXCoupledHumanState;
} // namespace detail

enum class MetalNumanXCoupledHumanOperation : std::uint32_t {
    candidateKinematics = MR_NUMANX_COUPLED_HUMAN_CANDIDATE_KINEMATICS,
    massAction = MR_NUMANX_COUPLED_HUMAN_MASS_ACTION,
    inverseMassPreconditioner =
        MR_NUMANX_COUPLED_HUMAN_INVERSE_MASS_PRECONDITIONER,
    publishCandidate = MR_NUMANX_COUPLED_HUMAN_STAGED_PUBLISH,
};

enum class MetalNumanXCoupledHumanPhase : std::uint32_t {
    preDynamics = 0u,
    postDynamics = 1u,
};

enum MetalNumanXCoupledHumanAccessFlag : std::uint32_t {
    // q/v and immutable Human model state may be read by the owning exact
    // kinematics callback.  Neither the context nor the callback may mutate
    // source q/v.
    MetalNumanXCoupledHumanReadSourceState = 1u << 0u,
    // The service reads a same-timeline row-major lower Cholesky factor L0 of
    // the Stand source effective tangent A0=M+armature+h*D=L0*L0^T. The exact
    // candidate callback may refresh this arena before it returns.
    MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor = 1u << 1u,
    // The service may read an explicit row-major tangent projector P.
    MetalNumanXCoupledHumanReadTangentProjector = 1u << 2u,
    // postDynamics only: read Human stand statuses; never turn failure into
    // success or mutate the owning status stream.
    MetalNumanXCoupledHumanReadStandStatus = 1u << 3u,
    // The owning callback may write only the candidate buffers supplied by
    // the query and the pass's source effective-tangent factor. It shares
    // command order.
    MetalNumanXCoupledHumanEncodeExactCandidate = 1u << 4u,
    // The service writes its own separate Matter-reaction arena.
    MetalNumanXCoupledHumanWriteReaction = 1u << 5u,
    // The service writes its own pending/accept/reject status arena.
    MetalNumanXCoupledHumanWriteJointStatus = 1u << 6u,
};

inline constexpr std::uint32_t kMetalNumanXCoupledHumanKnownAccess =
    MetalNumanXCoupledHumanReadSourceState |
    MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor |
    MetalNumanXCoupledHumanReadTangentProjector |
    MetalNumanXCoupledHumanReadStandStatus |
    MetalNumanXCoupledHumanEncodeExactCandidate |
    MetalNumanXCoupledHumanWriteReaction |
    MetalNumanXCoupledHumanWriteJointStatus;

enum MetalNumanXCoupledHumanQueryAccessFlag : std::uint32_t {
    MetalNumanXCoupledHumanQueryReadInput = 1u << 0u,
    MetalNumanXCoupledHumanQueryWriteOutput = 1u << 1u,
    MetalNumanXCoupledHumanQueryWriteCandidateQ = 1u << 2u,
    MetalNumanXCoupledHumanQueryWriteCandidateBodies = 1u << 3u,
    MetalNumanXCoupledHumanQueryWriteInverseStatus = 1u << 4u,
    MetalNumanXCoupledHumanQueryReadPointQueries = 1u << 5u,
    MetalNumanXCoupledHumanQueryWritePointJacobians = 1u << 6u,
    MetalNumanXCoupledHumanQueryReadMatterOutcome = 1u << 7u,
};

inline constexpr std::uint32_t kMetalNumanXCoupledHumanKnownQueryAccess =
    MetalNumanXCoupledHumanQueryReadInput |
    MetalNumanXCoupledHumanQueryWriteOutput |
    MetalNumanXCoupledHumanQueryWriteCandidateQ |
    MetalNumanXCoupledHumanQueryWriteCandidateBodies |
    MetalNumanXCoupledHumanQueryWriteInverseStatus |
    MetalNumanXCoupledHumanQueryReadPointQueries |
    MetalNumanXCoupledHumanQueryWritePointJacobians |
    MetalNumanXCoupledHumanQueryReadMatterOutcome;

struct MetalNumanXCoupledHumanPass;

// The leading fields, operation values, and stride semantics intentionally
// match numi::matter::CoupledCandidateQuery.  An adapter may copy that prefix
// directly, then must populate every version/access/provenance/address field
// below it before calling the Human service.
struct MetalNumanXCoupledHumanQuery {
    void* input = nullptr;
    void* output = nullptr;
    void* candidateQ = nullptr;
    void* candidateBodies = nullptr;
    // inverseMassPreconditioner only: MRInverseMassStatusGPU records laid out
    // [articulation][statusStride].  The Human record for one environment is
    // statuses[pass.articulationIndex * statusStride + environment].
    void* statuses = nullptr;
    void* pointQueries = nullptr;
    void* pointJacobians = nullptr;
    MetalNumanXCoupledHumanOperation operation =
        MetalNumanXCoupledHumanOperation::candidateKinematics;
    std::uint32_t generalizedVectorStride = 0u;
    std::uint32_t candidateQStride = 0u;
    std::uint32_t candidateBodyStride = 0u;
    std::uint32_t statusStride = 0u;
    std::uint32_t pointCount = 0u;
    std::uint32_t pointStride = 0u;
    std::uint32_t pointJacobianStride = 0u;

    std::uint32_t abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    std::uint32_t structSize = sizeof(MetalNumanXCoupledHumanQuery);
    std::uint32_t accessFlags = 0u;
    std::uint32_t requiredCapabilities = 0u;

    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;

    // Exact base GPU virtual addresses.  This first slice accepts no hidden
    // byte offsets.  A null optional buffer requires a zero address.
    std::uint64_t inputGPUAddress = 0u;
    std::uint64_t outputGPUAddress = 0u;
    std::uint64_t candidateQGPUAddress = 0u;
    std::uint64_t candidateBodiesGPUAddress = 0u;
    std::uint64_t statusesGPUAddress = 0u;
    std::uint64_t pointQueriesGPUAddress = 0u;
    std::uint64_t pointJacobiansGPUAddress = 0u;
    std::uint64_t matterOutcomeGPUAddress = 0u;

    // Required only by publishCandidate. Layout is
    // MRNumanXCoupledMatterOutcomeGPU[environment], with record stride below.
    // Publish is the pre-progress phase: both this expected count and the
    // outcome's completedMicrosteps must be exactly zero. The later resolve
    // query independently requires its exact nonzero finalized count.
    void* matterOutcomes = nullptr;
    std::uint32_t matterOutcomeStride = 0u;
    std::uint32_t matterSuccessCode = 0u;
    std::uint32_t expectedMatterCompletedMicrosteps = 0u;
    std::uint32_t transactionSlot = 0u;
    // Monotonic nonzero generation owned by the enclosing transaction adapter.
    // Reusing a slot without advancing this value is rejected before encoding.
    std::uint64_t slotGeneration = 0u;
};

using MetalNumanXCoupledHumanEncodeExactKinematics = bool (*)(
    void* context,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept;

// One borrowed Human linearization.  All pointers are unretained native
// id<MTLBuffer>/id<MTLCommandBuffer> identities and all GPU addresses name the
// beginning of those buffers.  The context may encode only on commandBuffer;
// it never commits, waits, reads back, creates a queue, or stores these values
// after the callback returns.
struct MetalNumanXCoupledHumanPass {
    std::uint32_t abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    std::uint32_t structSize = sizeof(MetalNumanXCoupledHumanPass);
    std::uint32_t accessFlags = 0u;
    std::uint32_t capabilities = 0u;

    void* commandBuffer = nullptr;
    void* sourceQ = nullptr;
    void* sourceV = nullptr;
    // [environment][nv][nv] row-major lower Cholesky L0 of the owning Stand
    // solve's source effective tangent A0=M+armature+h*D=L0*L0^T. Exact
    // kinematics must refresh it on this command-buffer timeline before
    // returning. This field is never a pure-M claim.
    void* sourceEffectiveTangentFactor = nullptr;
    // Optional [environment][nv][nv] row-major P. The capability promises a
    // finite symmetric/idempotent projector. Actions evaluate P^T*A0*P and the
    // preconditioner evaluates P*A0^-1*P^T; they are not mutual inverses in
    // general. Contact/equality constrained passes fail closed in ABI v4.
    void* tangentProjector = nullptr;
    void* standStatuses = nullptr;

    void* exactKinematicsContext = nullptr;
    MetalNumanXCoupledHumanEncodeExactKinematics encodeExactKinematics =
        nullptr;

    std::uint64_t sourceQGPUAddress = 0u;
    std::uint64_t sourceVGPUAddress = 0u;
    std::uint64_t sourceEffectiveTangentFactorGPUAddress = 0u;
    std::uint64_t tangentProjectorGPUAddress = 0u;
    std::uint64_t standStatusesGPUAddress = 0u;

    std::uint64_t sourceQElementCount = 0u;
    std::uint64_t sourceVElementCount = 0u;
    std::uint64_t sourceEffectiveTangentFactorElementCount = 0u;
    std::uint64_t tangentProjectorElementCount = 0u;
    std::uint64_t standStatusElementCount = 0u;

    MetalNumanXCoupledHumanPhase phase =
        MetalNumanXCoupledHumanPhase::preDynamics;
    std::uint32_t environmentCount = 0u;
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t qStride = 0u;
    std::uint32_t vStride = 0u;
    std::uint32_t factorStride = 0u;
    std::uint32_t projectorStride = 0u;
    std::uint32_t standStatusStride = 0u;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t standFlags = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t stepCount = 0u;
    std::uint32_t transactionSlot = 0u;
    float timestepSeconds = 0.0f;
    std::uint32_t reserved0 = 0u;

    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

struct MetalNumanXCoupledHumanResolveQuery {
    std::uint32_t abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    std::uint32_t structSize = sizeof(MetalNumanXCoupledHumanResolveQuery);
    std::uint32_t accessFlags =
        MetalNumanXCoupledHumanQueryReadMatterOutcome;
    std::uint32_t reserved0 = 0u;

    void* matterOutcomes = nullptr;
    std::uint64_t matterOutcomeGPUAddress = 0u;
    std::uint64_t matterOutcomeElementCount = 0u;
    std::uint32_t matterOutcomeStride = 0u;
    std::uint32_t matterSuccessCode = 0u;
    std::uint32_t expectedMatterCompletedMicrosteps = 0u;
    std::uint32_t transactionSlot = 0u;

    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

using MetalNumanXCoupledHumanBegin = bool (*)(
    void* context,
    const MetalNumanXCoupledHumanPass& pass
) noexcept;

using MetalNumanXCoupledHumanEncode = bool (*)(
    void* context,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept;

using MetalNumanXCoupledHumanResolve = bool (*)(
    void* context,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanResolveQuery& query
) noexcept;

struct MetalNumanXCoupledHumanProgram {
    std::uint32_t abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    std::uint32_t structSize = sizeof(MetalNumanXCoupledHumanProgram);
    std::uint32_t capabilities = 0u;
    std::uint32_t environmentCapacity = 0u;
    std::uint32_t pointCapacity = 0u;
    std::uint32_t transactionSlotCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;

    void* context = nullptr;
    MetalNumanXCoupledHumanBegin begin = nullptr;
    MetalNumanXCoupledHumanEncode encode = nullptr;
    MetalNumanXCoupledHumanResolve resolve = nullptr;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid() const noexcept {
        constexpr std::uint32_t required =
            MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION |
            MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER |
            MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH |
            MR_NUMANX_COUPLED_HUMAN_CAP_JOINT_DECISION;
        return abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
            structSize == sizeof(MetalNumanXCoupledHumanProgram) &&
            (capabilities & ~MR_NUMANX_COUPLED_HUMAN_KNOWN_CAPABILITIES) ==
                0u &&
            (capabilities & required) == required &&
            environmentCapacity != 0u &&
            pointCapacity <= MR_NUMANX_COUPLED_HUMAN_MAX_POINTS &&
            transactionSlotCount != 0u &&
            transactionSlotCount <=
                MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS &&
            context != nullptr && begin != nullptr && encode != nullptr &&
            resolve != nullptr && fingerprint != 0u;
    }
};

struct MetalNumanXCoupledHumanArenaView {
    void* matterGeneralizedReaction = nullptr;
    void* jointStatuses = nullptr;
    // Private per-slot transaction witness storage. This is exposed only so
    // enclosing adapters can reject caller-controlled buffers that overlap
    // any CoupledHuman authority; callers must never read or encode it.
    void* transactionMetadata = nullptr;
    std::uint64_t reactionGPUAddress = 0u;
    std::uint64_t jointStatusGPUAddress = 0u;
    std::uint64_t transactionMetadataGPUAddress = 0u;
    std::uint64_t reactionByteCount = 0u;
    std::uint64_t jointStatusByteCount = 0u;
    std::uint64_t transactionMetadataByteCount = 0u;
    std::uint32_t environmentCapacity = 0u;
    std::uint32_t reactionStride = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
    std::uint32_t jointStatusStride = 1u;
    std::uint32_t transactionSlot = 0u;
    std::uint64_t programFingerprint = 0u;
};

struct MetalNumanXCoupledHumanConfig {
    // Explicit library selection: no process-default library and no runtime
    // source compilation fallback.
    std::string metallibPath;
    std::uint32_t environmentCapacity = 1u;
    std::uint32_t pointCapacity = 0u;
    std::uint32_t transactionSlotCount = 2u;
    std::uint32_t reserved0 = 0u;
    std::uint64_t maximumRetainedBytes = 1024ull * 1024ull * 1024ull;
};

enum class MetalNumanXCoupledHumanHostStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    arithmeticOverflow,
    metalDeviceUnavailable,
    metallibUnavailable,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    incompatibleDevice,
    uninitialized,
    invalidSlot,
};

struct MetalNumanXCoupledHumanDiagnostics {
    MetalNumanXCoupledHumanHostStatus status =
        MetalNumanXCoupledHumanHostStatus::success;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t retainedBytes = 0u;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalNumanXCoupledHumanHostStatus::success;
    }
};

// Owns only pipeline state plus per-slot reaction/status arenas.  The object
// must outlive every invocation and enclosing command buffer using program().
// Callers own asynchronous slot reuse: a slot may not be reused until its
// enclosing command buffer has completed or was abandoned before commit, and
// every reuse must carry a strictly newer nonzero slotGeneration.  The context
// stores only scalar transaction metadata; it never installs completion
// handlers or retains borrowed transaction resources.
class MetalNumanXCoupledHumanContext {
public:
    explicit MetalNumanXCoupledHumanContext(
        MetalNumanXCoupledHumanConfig config = {}
    );
    ~MetalNumanXCoupledHumanContext();

    MetalNumanXCoupledHumanContext(
        MetalNumanXCoupledHumanContext&& other
    ) noexcept;
    MetalNumanXCoupledHumanContext& operator=(
        MetalNumanXCoupledHumanContext&& other
    ) noexcept;

    MetalNumanXCoupledHumanContext(
        const MetalNumanXCoupledHumanContext&
    ) = delete;
    MetalNumanXCoupledHumanContext& operator=(
        const MetalNumanXCoupledHumanContext&
    ) = delete;

    [[nodiscard]] MetalNumanXCoupledHumanDiagnostics initialize();
    [[nodiscard]] MetalNumanXCoupledHumanProgram program() const noexcept;
    [[nodiscard]] MetalNumanXCoupledHumanDiagnostics arenaView(
        std::uint32_t transactionSlot,
        MetalNumanXCoupledHumanArenaView& view
    ) const;

private:
    std::unique_ptr<detail::MetalNumanXCoupledHumanState> state_;
};

[[nodiscard]] const char* metalNumanXCoupledHumanHostStatusName(
    MetalNumanXCoupledHumanHostStatus status
) noexcept;

static_assert(
    static_cast<std::uint32_t>(
        MetalNumanXCoupledHumanOperation::candidateKinematics
    ) == 0u
);
static_assert(
    static_cast<std::uint32_t>(MetalNumanXCoupledHumanOperation::massAction) ==
        1u
);
static_assert(
    static_cast<std::uint32_t>(
        MetalNumanXCoupledHumanOperation::inverseMassPreconditioner
    ) == 2u
);
static_assert(
    static_cast<std::uint32_t>(
        MetalNumanXCoupledHumanOperation::publishCandidate
    ) == 3u
);
static_assert(std::is_standard_layout_v<MetalNumanXCoupledHumanQuery>);
static_assert(offsetof(MetalNumanXCoupledHumanQuery, input) == 0u);
static_assert(offsetof(MetalNumanXCoupledHumanQuery, operation) == 56u);
static_assert(
    offsetof(
        MetalNumanXCoupledHumanQuery,
        generalizedVectorStride
    ) == 60u
);
static_assert(offsetof(MetalNumanXCoupledHumanQuery, abiVersion) == 88u);

} // namespace metalrobo
