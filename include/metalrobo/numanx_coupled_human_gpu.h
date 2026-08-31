#pragma once

// Pointer-free Human/Matter coupling ABI shared by C++, Objective-C++, and
// Metal.  The owning Human runtime remains the only authority allowed to
// materialize articulated candidates or publish generalized reaction force.

#include "metalrobo/numi_human_stand_gpu.h"

#define MR_NUMANX_COUPLED_HUMAN_ABI_VERSION 4u
#define MR_NUMANX_COUPLED_HUMAN_MAX_DOFS MR_NUMI_HUMAN_STAND_MAX_DOFS
#define MR_NUMANX_COUPLED_HUMAN_MAX_Q MR_NUMI_HUMAN_STAND_MAX_Q
#define MR_NUMANX_COUPLED_HUMAN_MAX_POINTS \
    MR_ARTICULATED_OPERATOR_MAX_POINTS
#define MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS 8u

enum MRNumanXCoupledHumanOperation : mr_u32 {
    // Operation ordinals intentionally match
    // numi::matter::CoupledCandidateOperation and
    // metalrobo::MetalWorldCoupledCandidateOperation.
    MR_NUMANX_COUPLED_HUMAN_CANDIDATE_KINEMATICS = 0u,
    MR_NUMANX_COUPLED_HUMAN_MASS_ACTION = 1u,
    MR_NUMANX_COUPLED_HUMAN_INVERSE_MASS_PRECONDITIONER = 2u,
    MR_NUMANX_COUPLED_HUMAN_STAGED_PUBLISH = 3u,
};

enum MRNumanXCoupledHumanCapability : mr_u32 {
    // An owning callback integrates q and materializes exact Human body/point
    // kinematics on the borrowed command-buffer timeline.
    MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS = 1u << 0u,
    // The service consumes the owning Stand solve's source effective tangent
    // A0 = M + armature + h*D and evaluates P^T*A0*P. This is not a pure mass
    // operator when armature or passive damping is nonzero.
    MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION = 1u << 1u,
    // The service evaluates P*A0^-1*P^T as a preconditioner through triangular
    // solves. It is not claimed to invert P^T*A0*P.
    MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER =
        1u << 2u,
    // Accepted Matter delta-v is staged as a separate effective generalized
    // reaction; live q/v and the MyoSim force arena are never written.
    MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH = 1u << 3u,
    // A same-buffer Human/Matter status is initialized pending and resolved
    // only after both runtimes have produced their device statuses.
    MR_NUMANX_COUPLED_HUMAN_CAP_JOINT_DECISION = 1u << 4u,
    // The pass supplies an explicit row-major projector P for an optional
    // projected preconditioner. ABI v4 rejects support-contact/equality passes:
    // a full-space projector is not a nullspace or KKT constrained solve.
    MR_NUMANX_COUPLED_HUMAN_CAP_TANGENT_PROJECTOR = 1u << 5u,
};

#define MR_NUMANX_COUPLED_HUMAN_KNOWN_CAPABILITIES \
    (MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS | \
     MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION | \
     MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER | \
     MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH | \
     MR_NUMANX_COUPLED_HUMAN_CAP_JOINT_DECISION | \
     MR_NUMANX_COUPLED_HUMAN_CAP_TANGENT_PROJECTOR)

enum MRNumanXCoupledHumanDispatchFlags : mr_u32 {
    MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR = 1u << 0u,
};

enum MRNumanXCoupledHumanDecision : mr_u32 {
    // Pending is deliberately zero so cleared/unresolved storage fails closed.
    MR_NUMANX_COUPLED_HUMAN_PENDING = 0u,
    MR_NUMANX_COUPLED_HUMAN_ACCEPT = 1u,
    MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN = 2u,
    MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER = 3u,
};

enum MRNumanXCoupledHumanTransactionStage : mr_u32 {
    MR_NUMANX_COUPLED_HUMAN_STAGE_UNINITIALIZED = 0u,
    MR_NUMANX_COUPLED_HUMAN_STAGE_BEGUN = 1u,
    MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED = 2u,
    MR_NUMANX_COUPLED_HUMAN_STAGE_PUBLISHED = 3u,
    MR_NUMANX_COUPLED_HUMAN_STAGE_RESOLVED = 4u,
};

enum MRNumanXCoupledHumanOperationWitness : mr_u32 {
    MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE = 1u << 0u,
    MR_NUMANX_COUPLED_HUMAN_WITNESS_PUBLISH = 1u << 1u,
    MR_NUMANX_COUPLED_HUMAN_WITNESS_RESOLVE = 1u << 2u,
};

enum MRNumanXCoupledHumanAdvanceFlags : mr_u32 {
    MR_NUMANX_COUPLED_HUMAN_ADVANCE_REQUIRE_PENDING = 1u << 0u,
};

// Service-side failures occupy a disjoint range from
// MRNumiHumanStandStatusCode and remain device-visible in humanCode.
#define MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT 0x80000000u
enum MRNumanXCoupledHumanServiceCode : mr_u32 {
    MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_DISPATCH =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 1u,
    MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_INPUT =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 2u,
    MR_NUMANX_COUPLED_HUMAN_SERVICE_INVALID_FACTOR =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 3u,
    MR_NUMANX_COUPLED_HUMAN_SERVICE_NONFINITE_RESULT =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 4u,
    MR_NUMANX_COUPLED_HUMAN_SERVICE_KINEMATICS_FAILED =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 5u,
};

// Exactly 32 bytes, one record per environment and transaction slot.  No
// consumer may publish while decision is pending or rejected.
typedef struct MR_ALIGN16 MRNumanXCoupledHumanStatusGPU {
    mr_u32 abiVersion;
    mr_u32 decision;
    mr_u32 environment;
    mr_u32 stepIndex;

    mr_u32 humanCode;
    mr_u32 matterCode;
    mr_u32 humanCompletedSteps;
    mr_u32 matterCompletedMicrosteps;
} MRNumanXCoupledHumanStatusGPU;

// Minimal device outcome written by the Matter adapter before staged publish
// and reused during post-Human resolution.  A nonzero reserved word is an
// invalid outcome, never an implicit acceptance.
typedef struct MR_ALIGN16 MRNumanXCoupledMatterOutcomeGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 completedMicrosteps;
    mr_u32 reserved0;
} MRNumanXCoupledMatterOutcomeGPU;

// Private per-slot transaction identity.  This record is never exposed through
// arenaView; every device operation verifies it before touching reaction or
// joint status storage.
typedef struct MR_ALIGN16 MRNumanXCoupledHumanSlotMetadataGPU {
    mr_u32 abiVersion;
    mr_u32 structSize;
    mr_u32 stage;
    mr_u32 transactionSlot;

    mr_u32 stepIndex;
    mr_u32 environmentCount;
    mr_u32 operationWitnesses;
    mr_u32 qCoordinateCount;

    mr_u32 dofCount;
    mr_u32 sourceQStride;
    mr_u32 sourceVStride;
    mr_u32 factorStride;

    mr_u32 reactionStride;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXCoupledHumanSlotMetadataGPU;

// Constants for candidate validation, projected dense effective-tangent
// operators, and staged publication. Every stride is an element count, never
// bytes. factorStride and projectorStride are exactly nv*nv for the active
// Human.
typedef struct MR_ALIGN16 MRNumanXCoupledHumanDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 operation;
    mr_u32 environmentCount;
    mr_u32 flags;

    mr_u32 nq;
    mr_u32 nv;
    mr_u32 sourceQStride;
    mr_u32 sourceVStride;

    mr_u32 generalizedVectorStride;
    mr_u32 candidateQStride;
    mr_u32 candidateBodyStride;
    mr_u32 statusStride;

    mr_u32 pointCount;
    mr_u32 pointStride;
    mr_u32 pointJacobianStride;
    mr_u32 factorStride;

    mr_u32 projectorStride;
    mr_u32 reactionStride;
    mr_u32 articulationIndex;
    mr_u32 transactionSlot;

    mr_u32 stepIndex;
    mr_u32 bodyCount;
    mr_u32 standFlags;
    mr_u32 reserved0;

    // x = timestep seconds, y = reciprocal timestep, zw required zero.
    mr_float4 timestepAndInverse;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXCoupledHumanDispatchGPU;

typedef struct MR_ALIGN16 MRNumanXCoupledHumanResolveDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 stepIndex;
    mr_u32 dofCount;

    mr_u32 statusStride;
    mr_u32 standStatusStride;
    mr_u32 matterOutcomeStride;
    mr_u32 reactionStride;

    mr_u32 expectedHumanCompletedSteps;
    mr_u32 expectedMatterCompletedMicrosteps;
    mr_u32 matterSuccessCode;
    mr_u32 transactionSlot;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXCoupledHumanResolveDispatchGPU;

typedef struct MR_ALIGN16 MRNumanXCoupledHumanAdvanceDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 expectedStage;
    mr_u32 nextStage;
    mr_u32 witness;

    mr_u32 flags;
    mr_u32 environmentCount;
    mr_u32 transactionSlot;
    mr_u32 stepIndex;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;

    mr_u32 qCoordinateCount;
    mr_u32 dofCount;
    mr_u32 reactionStride;
    mr_u32 reserved0;
} MRNumanXCoupledHumanAdvanceDispatchGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(MR_NUMANX_COUPLED_HUMAN_MAX_DOFS == 160u);
static_assert(MR_NUMANX_COUPLED_HUMAN_MAX_Q == 161u);
static_assert(sizeof(MRNumanXCoupledHumanStatusGPU) == 32u);
static_assert(alignof(MRNumanXCoupledHumanStatusGPU) == 16u);
static_assert(offsetof(MRNumanXCoupledHumanStatusGPU, humanCode) == 16u);
static_assert(sizeof(MRNumanXCoupledMatterOutcomeGPU) == 16u);
static_assert(alignof(MRNumanXCoupledMatterOutcomeGPU) == 16u);
static_assert(sizeof(MRNumanXCoupledHumanSlotMetadataGPU) == 96u);
static_assert(alignof(MRNumanXCoupledHumanSlotMetadataGPU) == 16u);
static_assert(
    offsetof(
        MRNumanXCoupledHumanSlotMetadataGPU,
        programFingerprint
    ) == 64u
);
static_assert(offsetof(MRNumanXCoupledHumanSlotMetadataGPU,
                       qCoordinateCount) == 28u);
static_assert(offsetof(MRNumanXCoupledHumanSlotMetadataGPU,
                       reactionStride) == 48u);
static_assert(sizeof(MRNumanXCoupledHumanDispatchGPU) == 144u);
static_assert(alignof(MRNumanXCoupledHumanDispatchGPU) == 16u);
static_assert(
    offsetof(MRNumanXCoupledHumanDispatchGPU, timestepAndInverse) == 96u
);
static_assert(
    offsetof(MRNumanXCoupledHumanDispatchGPU, programFingerprint) == 112u
);
static_assert(sizeof(MRNumanXCoupledHumanResolveDispatchGPU) == 80u);
static_assert(alignof(MRNumanXCoupledHumanResolveDispatchGPU) == 16u);
static_assert(
    offsetof(
        MRNumanXCoupledHumanResolveDispatchGPU,
        programFingerprint
    ) == 48u
);
static_assert(sizeof(MRNumanXCoupledHumanAdvanceDispatchGPU) == 80u);
static_assert(alignof(MRNumanXCoupledHumanAdvanceDispatchGPU) == 16u);
static_assert(
    offsetof(
        MRNumanXCoupledHumanAdvanceDispatchGPU,
        programFingerprint
    ) == 32u
);
#endif
