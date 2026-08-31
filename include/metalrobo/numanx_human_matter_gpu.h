#pragma once

// Pointer-free owner-side Human/Matter transaction ABI shared by C++,
// Objective-C++, and Metal.  This ABI deliberately names the discrete Human
// operator it exposes: A0 = M(q0) + armature + h*D.  It is not the pure mass
// matrix ABI used by the older generic CoupledHuman service.

#include "metalrobo/numanx_coupled_human_gpu.h"

#define MR_NUMANX_HUMAN_MATTER_ABI_VERSION 4u
#define MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES 64u
#define MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_BYTES 64u
#define MR_NUMANX_HUMAN_MATTER_PREPARED_TOKEN_FINGERPRINT_OFFSET 56u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_ABI_VERSION 1u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_MAGIC 0x4e584257u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_WITNESS_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_PHYSICS_SUBSTEP_COUNT 1u
#define MR_NUMANX_HUMAN_MATTER_PROPOSAL_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION 1u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION 1u
#define MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_APPLY_ACTION_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_MATTER_APPLY_OUTCOME_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_APPLIED_OUTCOME_BYTES 128u
#define MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION 1u
#define MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES 128u

enum MRNumanXHumanMatterOwnerStage : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_STAGE_UNINITIALIZED = 0u,
    MR_NUMANX_HUMAN_MATTER_STAGE_BEGUN = 1u,
    MR_NUMANX_HUMAN_MATTER_STAGE_REACTION_CONSUMED = 2u,
    MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_PREPARED = 3u,
    MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_REJECT_RESTORED = 4u,
    MR_NUMANX_HUMAN_MATTER_STAGE_PROPOSED = 5u,
    MR_NUMANX_HUMAN_MATTER_STAGE_APPLIED_ACCEPT_QUARANTINED = 6u,
    MR_NUMANX_HUMAN_MATTER_STAGE_APPLIED_RESTORE = 7u,
    MR_NUMANX_HUMAN_MATTER_STAGE_ROOT_PUBLISHED = 8u,
    MR_NUMANX_HUMAN_MATTER_STAGE_FAILED = 9u,
};

enum MRNumanXHumanMatterOwnerCode : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS = 0u,
    MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_DISPATCH = 1u,
    MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_ORDER = 2u,
    MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_JOINT_STATUS = 3u,
    MR_NUMANX_HUMAN_MATTER_OWNER_NONFINITE_REACTION = 4u,
    MR_NUMANX_HUMAN_MATTER_OWNER_INVALID_BRAIN_WITNESS = 5u,
};

// Written by the last owner kernel on the first physical command buffer, or
// overwritten with FAILED by the owner completion handler before its liveness
// fallback advances the shared event. A later apply pass must never restore or
// publish from UNKNOWN/FAILED because the checkpoint writes may be partial.
enum MRNumanXHumanMatterPhysicalCommandStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_UNKNOWN = 0u,
    MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE = 1u,
    MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_FAILED = 2u,
};

enum MRNumanXHumanMatterDispatchFlags : mr_u32 {
    // The adapter-owned token is a provisional physical-prepare record.  It
    // is never the root commit token.  The public program requires it.
    MR_NUMANX_HUMAN_MATTER_HAS_PREPARED_TOKEN = 1u << 0u,
};

enum MRNumanXHumanMatterProposalFlags : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_VALIDATE_BRAIN_WITNESS = 1u << 0u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCE_REJECT = 1u << 1u,
};

enum MRNumanXHumanMatterApplyFlags : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_APPLY_VALIDATE_BRAIN_ACK = 1u << 0u,
    MR_NUMANX_HUMAN_MATTER_APPLY_FORCE_REJECT = 1u << 1u,
};

enum MRNumanXHumanMatterBrainCommitStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_UNINITIALIZED = 0u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_COMPLETE = 1u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_COMMIT_PREPARE_FAILED = 2u,
};

enum MRNumanXHumanMatterRootDecision : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_ROOT_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT = 1u,
    MR_NUMANX_HUMAN_MATTER_ROOT_REJECT = 2u,
};

enum MRNumanXHumanMatterCandidateFlags : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_CANDIDATE_HAS_POINT_WORLD = 1u << 0u,
};

enum MRNumanXHumanMatterProposalStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY = 1u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_FAILURE = 2u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_TERMINAL_NO_TOUCH = 3u,
};

enum MRNumanXHumanMatterProposalCode : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS = 0u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT = 1u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_OWNER = 2u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS = 3u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH = 4u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT = 5u,
    MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT = 6u,
};

enum MRNumanXHumanMatterBrainPreflightStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS = 1u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_FAILURE = 2u,
};

enum MRNumanXHumanMatterBrainAckStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT = 1u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT = 2u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_INVALID = 3u,
};

enum MRNumanXHumanMatterBrainAckCode : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS = 0u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT = 1u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_INVALID_PROPOSAL = 2u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_INVALID_WITNESS = 3u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_INVALID_FAST_GATE = 4u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_INVALID_PREFLIGHT = 5u,
    MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_TOKEN_MISMATCH = 6u,
};

enum MRNumanXHumanMatterApplyStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_APPLY_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT = 1u,
    MR_NUMANX_HUMAN_MATTER_APPLY_REJECT = 2u,
    MR_NUMANX_HUMAN_MATTER_APPLY_TERMINAL_NO_TOUCH = 3u,
};

enum MRNumanXHumanMatterAppliedStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_APPLIED_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED = 1u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED = 2u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH = 3u,
};

enum MRNumanXHumanMatterAppliedCode : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS = 0u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT = 1u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT = 2u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER = 3u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_BRAIN_ACK = 4u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_TOKEN_MISMATCH = 5u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_MATTER_REJECT = 6u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_MATTER_OUTCOME = 7u,
    MR_NUMANX_HUMAN_MATTER_APPLIED_BRAIN_REJECT = 8u,
};

enum MRNumanXHumanMatterPublicationFenceStatus : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING = 0u,
    MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED = 1u,
    MR_NUMANX_HUMAN_MATTER_PUBLICATION_FAILURE = 2u,
};

// Private owner witness, one record per environment.  It is initialized on
// the borrowed command-buffer timeline before any motor mutation.  The
// reactionConsumed word is the authoritative exactly-once witness.
typedef struct MR_ALIGN16 MRNumanXHumanMatterOwnerStatusGPU {
    mr_u32 abiVersion;
    mr_u32 stage;
    mr_u32 environment;
    mr_u32 stepIndex;

    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 code;
    mr_u32 reactionConsumed;

    mr_u32 restored;
    mr_u32 preparedTokenPreserved;
    mr_u32 appliedOutcomePublished;
    mr_u32 physicsSubstepCount;

    mr_u32 physicalCommandStatus;
    // Logical articulation shape is part of the prepared transaction
    // identity. Capacity-backed transport strides are carried separately.
    mr_u32 qCoordinateCount;
    // Global adapter/Matter control root, distinct from local stepIndex.
    mr_u32 controlStep;
    mr_u32 dofCount;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXHumanMatterOwnerStatusGPU;

// Exact compact layout for begin, pre-stand reaction consumption, and final
// accepted-or-restored publication.  Strides are element counts except the
// explicitly byte-named acceptedTokenStrideBytes.
typedef struct MR_ALIGN16 MRNumanXHumanMatterDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 stepIndex;
    mr_u32 substepIndex;

    mr_u32 flags;
    mr_u32 transactionSlot;
    mr_u32 nq;
    mr_u32 nv;

    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 mujocoStateStride;
    mr_u32 mujocoStateCount;

    mr_u32 generalizedForceStride;
    mr_u32 generalizedForceOffset;
    mr_u32 reactionStride;
    mr_u32 jointStatusStride;

    mr_u32 acceptedTokenStrideBytes;
    mr_u32 ownerStatusStride;

    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;

    // x = timestep seconds, y = reciprocal timestep, zw must be zero.
    mr_float4 timestepAndInverse;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXHumanMatterDispatchGPU;

// One exact candidate invocation. The generic articulated operator consumes
// a private combined point stream whose prefix is the four canonical Human
// body probes and whose suffix contains the adapter's attachment queries.
// Every stride is an element count. The source state is the owner checkpoint;
// candidate q integrates v0+delta-v for exactly one timestep.
typedef struct MR_ALIGN16 MRNumanXHumanMatterCandidateDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 flags;
    mr_u32 environmentCount;
    mr_u32 articulationIndex;

    mr_u32 nq;
    mr_u32 nv;
    mr_u32 bodyCount;
    mr_u32 articulationFirstBody;

    mr_u32 sourcePointStride;
    mr_u32 sourceBodyProbeOffset;
    mr_u32 bodyProbeCount;
    mr_u32 candidatePointCount;

    mr_u32 combinedPointStride;
    mr_u32 candidateQStride;
    mr_u32 candidateBodyStride;
    mr_u32 candidatePointStride;

    mr_u32 candidatePointWorldStride;
    mr_u32 candidatePointJacobianStride;
    mr_u32 privateBodyPoseStride;
    mr_u32 privatePointWorldStride;

    mr_u32 privatePointJacobianStride;
    mr_u32 privateOperatorScratchStride;
    mr_u32 deltaVelocityStride;
    mr_u32 substepIndex;

    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u32 reserved3;

    // x = timestep seconds, y = reciprocal timestep, zw must be zero.
    mr_float4 timestepAndInverse;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXHumanMatterCandidateDispatchGPU;

// Written only by the final NumiBrain accepted-consequence preparation
// kernel.  The earlier NBAcceptedPhysicsGateResult is intentionally neither
// layout- nor magic-compatible with this proof record.
typedef struct MR_ALIGN16 MRNumanXHumanMatterBrainCommitWitnessGPU {
    mr_u32 magic;
    mr_u32 abiVersion;
    mr_u32 structBytes;
    mr_u32 status;

    mr_u32 decision;
    mr_u32 environment;
    mr_u32 stepIndex;
    mr_u32 substepIndex;

    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u32 reserved0;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;

    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 brainProgramFingerprint;

    mr_u64 brainShadowStateFingerprint;
    // FNV-1a-64 over every field in declaration order except this field.
    // Integer bytes are mixed least-significant byte first. All reserved
    // fields participate and must be zero. The impossible-for-production
    // zero result is represented by the FNV offset basis.
    mr_u64 witnessFingerprint;
    mr_u64 reserved1[2];
} MRNumanXHumanMatterBrainCommitWitnessGPU;

// Mutation-free owner proposal. The exact natural layout is u32x4 at bytes
// 0...15, u64x10 at bytes 16...95, u32x6 at bytes 96...119, and the terminal
// fingerprint at byte 120. proposalFingerprint is nonzero FNV-1a-64 over every
// prior field in declaration order, with little-endian integer bytes.
typedef struct MR_ALIGN16 MRNumanXHumanMatterProposalGPU {
    mr_u32 abiVersion;
    mr_u32 status;
    mr_u32 decision;
    mr_u32 code;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 brainProgramFingerprint;
    mr_u64 brainShadowStateFingerprint;
    mr_u64 brainWitnessFingerprint;
    // Exact unpublished HumanIO candidate and the pointer-free identity of
    // its retained publication program. These remain transitively bound by
    // proposal/ACK/applied/fence fingerprints without enlarging the record.
    mr_u64 candidatePublicationFingerprint;
    mr_u64 humanIOIdentityFingerprint;
    mr_u32 environment;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u64 proposalFingerprint;
} MRNumanXHumanMatterProposalGPU;

// Host-written only after all fallible fast/cognitive publication preflight
// succeeds. The Brain ACK kernel requires exact SUCCESS and recomputes the FNV
// over bytes 0...119. PENDING/FAILURE can never authorize an ACCEPT ACK.
typedef struct MR_ALIGN16 MRNumanXHumanMatterBrainCommitPreflightGPU {
    mr_u32 abiVersion;
    mr_u32 structBytes;
    mr_u32 status;
    mr_u32 environment;
    mr_u32 controlStep;
    mr_u32 substepIndex;
    mr_u32 physicsSubstepCount;
    mr_u32 transactionSlot;
    mr_u64 ownerProgramFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 substepFingerprint;
    mr_u64 physicsTokenFingerprint;
    mr_u64 fastTargetGeneration;
    mr_u64 cognitiveTargetGeneration;
    mr_u64 jointReceiptFingerprint;
    mr_u64 fastProgramFingerprint;
    mr_u64 brainProgramFingerprint;
    mr_u64 preflightFingerprint;
} MRNumanXHumanMatterBrainCommitPreflightGPU;

// Brain-owned result produced after validating proposal + witness + fast gate
// + host preflight. Fast/preflight fingerprints are carried explicitly so an
// applied outcome binds the complete Brain close transitively through ack.
typedef struct MR_ALIGN16 MRNumanXHumanMatterBrainAckGPU {
    mr_u32 abiVersion;
    mr_u32 status;
    mr_u32 decision;
    mr_u32 code;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 proposalFingerprint;
    mr_u64 preflightFingerprint;
    mr_u64 fastGateFingerprint;
    mr_u64 brainWitnessFingerprint;
    mr_u64 brainProgramFingerprint;
    mr_u32 environment;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u64 ackFingerprint;
} MRNumanXHumanMatterBrainAckGPU;

// Context-owned action written before the adapter/Matter apply callback. The
// callback may consume it but cannot replace or alias it.
typedef struct MR_ALIGN16 MRNumanXHumanMatterApplyActionGPU {
    mr_u32 abiVersion;
    mr_u32 status;
    mr_u32 decision;
    mr_u32 code;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 proposalFingerprint;
    mr_u64 ackFingerprint;
    mr_u64 preflightFingerprint;
    mr_u64 fastGateFingerprint;
    mr_u64 brainWitnessFingerprint;
    mr_u32 environment;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u64 actionFingerprint;
} MRNumanXHumanMatterApplyActionGPU;

// Adapter/Matter-owned completion of its same-command-buffer accept/restore.
// The Human owner validates this record after the adapter encoder finishes and
// only then mutates/restores Human q/v/MyoSim.
typedef struct MR_ALIGN16 MRNumanXHumanMatterMatterApplyOutcomeGPU {
    mr_u32 abiVersion;
    mr_u32 status;
    mr_u32 decision;
    mr_u32 code;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 proposalFingerprint;
    mr_u64 ackFingerprint;
    mr_u64 actionFingerprint;
    mr_u64 matterProgramFingerprint;
    mr_u64 reserved0;
    mr_u32 environment;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u64 outcomeFingerprint;
} MRNumanXHumanMatterMatterApplyOutcomeGPU;

// Last GPU record of the owner apply command. ACCEPT remains quarantined until
// the later COMMITTED publication fence passes integrity/replay validation and
// is released.
typedef struct MR_ALIGN16 MRNumanXHumanMatterAppliedOutcomeGPU {
    mr_u32 abiVersion;
    mr_u32 status;
    mr_u32 decision;
    mr_u32 code;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 proposalFingerprint;
    mr_u64 ackFingerprint;
    mr_u64 preflightFingerprint;
    mr_u64 fastGateFingerprint;
    // Exact Matter-owned retained-checkpoint apply outcome fingerprint.
    mr_u64 matterApplyFingerprint;
    mr_u32 environment;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u64 appliedFingerprint;
} MRNumanXHumanMatterAppliedOutcomeGPU;

// Owner-owned shared visibility fence. High-level Brain code may overwrite the
// exact PENDING reservation with COMMITTED only after all fallible preflight
// and nonthrow fast+cognitive pointer flips. Fence FNV covers bytes 0...119.
typedef struct MR_ALIGN16 MRNumanXHumanMatterJointPublicationFenceGPU {
    mr_u32 abiVersion;
    mr_u32 structBytes;
    mr_u32 status;
    mr_u32 environment;
    mr_u32 controlStep;
    mr_u32 substepIndex;
    mr_u32 physicsSubstepCount;
    mr_u32 reserved0;
    mr_u64 ownerProgramFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 physicsTokenFingerprint;
    mr_u64 brainProgramFingerprint;
    mr_u64 brainShadowStateFingerprint;
    mr_u64 brainWitnessFingerprint;
    mr_u64 appliedDecisionFingerprint;
    mr_u64 jointCommitFingerprint;
    mr_u64 brainGeneration;
    mr_u64 fenceFingerprint;
} MRNumanXHumanMatterJointPublicationFenceGPU;

typedef struct MR_ALIGN16 MRNumanXHumanMatterProposalDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 flags;
    mr_u32 environmentCount;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 ownerStatusStride;
    mr_u32 brainWitnessStride;
    mr_u32 preparedTokenStrideBytes;
    mr_u32 proposalStride;
    mr_u32 proposedTokenStrideBytes;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 candidatePublicationFingerprint;
    mr_u64 humanIOIdentityFingerprint;
} MRNumanXHumanMatterProposalDispatchGPU;

typedef struct MR_ALIGN16 MRNumanXHumanMatterApplyDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 flags;
    mr_u32 environmentCount;
    mr_u32 stepIndex;
    mr_u32 substepIndex;
    mr_u32 transactionSlot;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 mujocoStateStride;
    mr_u32 mujocoStateCount;
    mr_u32 ownerStatusStride;
    mr_u32 brainAckStride;
    // Immutable token copied by the proposal kernel, never the mutable
    // adapter prepared-token arena.
    mr_u32 proposedTokenStrideBytes;
    mr_u32 applyActionStride;
    mr_u32 matterOutcomeStride;
    mr_u32 appliedOutcomeStride;
    mr_u32 finalTokenStrideBytes;
    mr_u32 physicsSubstepCount;
    mr_u32 controlStep;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
} MRNumanXHumanMatterApplyDispatchGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(MRNumanXHumanMatterOwnerStatusGPU) == 96u);
static_assert(alignof(MRNumanXHumanMatterOwnerStatusGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterOwnerStatusGPU, programFingerprint) == 64u
);
static_assert(
    offsetof(MRNumanXHumanMatterOwnerStatusGPU, controlStep) == 56u
);
static_assert(
    offsetof(MRNumanXHumanMatterOwnerStatusGPU, qCoordinateCount) == 52u
);
static_assert(
    offsetof(MRNumanXHumanMatterOwnerStatusGPU, dofCount) == 60u
);
static_assert(sizeof(MRNumanXHumanMatterDispatchGPU) == 128u);
static_assert(alignof(MRNumanXHumanMatterDispatchGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterDispatchGPU, timestepAndInverse) == 80u
);
static_assert(
    offsetof(MRNumanXHumanMatterDispatchGPU, controlStep) == 76u
);
static_assert(
    offsetof(MRNumanXHumanMatterDispatchGPU, programFingerprint) == 96u
);
static_assert(sizeof(MRNumanXHumanMatterCandidateDispatchGPU) == 160u);
static_assert(alignof(MRNumanXHumanMatterCandidateDispatchGPU) == 16u);
static_assert(
    offsetof(
        MRNumanXHumanMatterCandidateDispatchGPU,
        timestepAndInverse
    ) == 112u
);
static_assert(
    offsetof(MRNumanXHumanMatterCandidateDispatchGPU, controlStep) == 104u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterCandidateDispatchGPU,
        programFingerprint
    ) == 128u
);
static_assert(sizeof(MRNumanXHumanMatterBrainCommitWitnessGPU) == 128u);
static_assert(
    alignof(MRNumanXHumanMatterBrainCommitWitnessGPU) == 16u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterBrainCommitWitnessGPU,
        programFingerprint
    ) == 48u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterBrainCommitWitnessGPU,
        controlStep
    ) == 40u
);
static_assert(sizeof(MRNumanXHumanMatterProposalGPU) == 128u);
static_assert(alignof(MRNumanXHumanMatterProposalGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterProposalGPU, programFingerprint) == 16u
);
static_assert(
    offsetof(MRNumanXHumanMatterProposalGPU, environment) == 96u
);
static_assert(
    offsetof(MRNumanXHumanMatterProposalGPU, proposalFingerprint) == 120u
);
static_assert(
    sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU) == 128u
);
static_assert(
    alignof(MRNumanXHumanMatterBrainCommitPreflightGPU) == 16u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterBrainCommitPreflightGPU,
        ownerProgramFingerprint
    ) == 32u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterBrainCommitPreflightGPU,
        preflightFingerprint
    ) == 120u
);
static_assert(sizeof(MRNumanXHumanMatterBrainAckGPU) == 128u);
static_assert(alignof(MRNumanXHumanMatterBrainAckGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterBrainAckGPU, programFingerprint) == 16u
);
static_assert(
    offsetof(MRNumanXHumanMatterBrainAckGPU, environment) == 96u
);
static_assert(
    offsetof(MRNumanXHumanMatterBrainAckGPU, ackFingerprint) == 120u
);
static_assert(sizeof(MRNumanXHumanMatterApplyActionGPU) == 128u);
static_assert(alignof(MRNumanXHumanMatterApplyActionGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterApplyActionGPU, programFingerprint) == 16u
);
static_assert(
    offsetof(MRNumanXHumanMatterApplyActionGPU, environment) == 96u
);
static_assert(
    offsetof(MRNumanXHumanMatterApplyActionGPU, actionFingerprint) == 120u
);
static_assert(
    sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU) == 128u
);
static_assert(
    alignof(MRNumanXHumanMatterMatterApplyOutcomeGPU) == 16u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterMatterApplyOutcomeGPU,
        programFingerprint
    ) == 16u
);
static_assert(
    offsetof(MRNumanXHumanMatterMatterApplyOutcomeGPU, environment) == 96u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterMatterApplyOutcomeGPU,
        outcomeFingerprint
    ) == 120u
);
static_assert(sizeof(MRNumanXHumanMatterAppliedOutcomeGPU) == 128u);
static_assert(alignof(MRNumanXHumanMatterAppliedOutcomeGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterAppliedOutcomeGPU, programFingerprint) == 16u
);
static_assert(
    offsetof(MRNumanXHumanMatterAppliedOutcomeGPU, environment) == 96u
);
static_assert(
    offsetof(MRNumanXHumanMatterAppliedOutcomeGPU, appliedFingerprint) == 120u
);
static_assert(
    sizeof(MRNumanXHumanMatterJointPublicationFenceGPU) == 128u
);
static_assert(
    alignof(MRNumanXHumanMatterJointPublicationFenceGPU) == 16u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterJointPublicationFenceGPU,
        ownerProgramFingerprint
    ) == 32u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterJointPublicationFenceGPU,
        fenceFingerprint
    ) == 120u
);
static_assert(sizeof(MRNumanXHumanMatterProposalDispatchGPU) == 112u);
static_assert(alignof(MRNumanXHumanMatterProposalDispatchGPU) == 16u);
static_assert(
    offsetof(
        MRNumanXHumanMatterProposalDispatchGPU,
        programFingerprint
    ) == 64u
);
static_assert(
    offsetof(
        MRNumanXHumanMatterProposalDispatchGPU,
        candidatePublicationFingerprint
    ) == 96u
);
static_assert(sizeof(MRNumanXHumanMatterApplyDispatchGPU) == 128u);
static_assert(alignof(MRNumanXHumanMatterApplyDispatchGPU) == 16u);
static_assert(
    offsetof(MRNumanXHumanMatterApplyDispatchGPU, programFingerprint) == 96u
);
#endif
