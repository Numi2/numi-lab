#pragma once

#include "numi/matter/shared.h"

#define NM_MATTER_PREPARED_STATE_ABI_VERSION 1u
#define NM_MATTER_OWNER_APPLY_ABI_VERSION 4u
#define NM_MATTER_OWNER_BRAIN_ACK_ABI_VERSION 1u
#define NM_MATTER_OWNER_APPLY_RECORD_BYTES 128u
#define NM_MATTER_PUBLICATION_FENCE_ABI_VERSION 1u

enum NMPreparedStateBindingStatus : nm_u32 {
    NM_PREPARED_STATE_BINDING_PENDING = 0u,
    NM_PREPARED_STATE_BINDING_VALID = 1u,
    NM_PREPARED_STATE_BINDING_REJECTED = 2u,
};

enum NMPreparedStateAction : nm_u32 {
    NM_PREPARED_STATE_ACTION_PENDING = 0u,
    NM_PREPARED_STATE_ACTION_ACCEPT = 1u,
    NM_PREPARED_STATE_ACTION_RESTORE = 2u,
    // The physical prepare CB failed and its checkpoints may be partial.
    // Every accepted-state restore/commit kernel must leave authority alone.
    NM_PREPARED_STATE_ACTION_TERMINAL_NO_TOUCH = 3u,
};

enum NMPreparedStateApplyOutcome : nm_u32 {
    NM_PREPARED_STATE_APPLY_PENDING = 0u,
    NM_PREPARED_STATE_APPLY_RESOLVED = 1u,
    NM_PREPARED_STATE_APPLY_TERMINAL_NO_TOUCH = 2u,
    NM_PREPARED_STATE_APPLY_ACCEPTED_PENDING_PUBLICATION = 3u,
};

enum NMOwnerRootDecision : nm_u32 {
    NM_OWNER_ROOT_PENDING = 0u,
    NM_OWNER_ROOT_ACCEPT = 1u,
    NM_OWNER_ROOT_REJECT = 2u,
};

enum NMOwnerProposalStatus : nm_u32 {
    NM_OWNER_PROPOSAL_PENDING = 0u,
    NM_OWNER_PROPOSAL_READY = 1u,
    NM_OWNER_PROPOSAL_FAILURE = 2u,
    NM_OWNER_PROPOSAL_TERMINAL_NO_TOUCH = 3u,
};

enum NMOwnerProposalCode : nm_u32 {
    NM_OWNER_PROPOSAL_SUCCESS = 0u,
    NM_OWNER_PROPOSAL_PHYSICAL_REJECT = 1u,
    NM_OWNER_PROPOSAL_INVALID_OWNER = 2u,
    NM_OWNER_PROPOSAL_INVALID_BRAIN_WITNESS = 3u,
    NM_OWNER_PROPOSAL_TOKEN_MISMATCH = 4u,
    NM_OWNER_PROPOSAL_FORCED_REJECT = 5u,
    NM_OWNER_PROPOSAL_BRAIN_REJECT = 6u,
};

enum NMOwnerBrainAckStatus : nm_u32 {
    NM_OWNER_BRAIN_ACK_PENDING = 0u,
    NM_OWNER_BRAIN_ACK_ACCEPT = 1u,
    NM_OWNER_BRAIN_ACK_REJECT = 2u,
    NM_OWNER_BRAIN_ACK_INVALID = 3u,
};

enum NMOwnerBrainAckCode : nm_u32 {
    NM_OWNER_BRAIN_ACK_SUCCESS = 0u,
    NM_OWNER_BRAIN_ACK_PROPOSAL_REJECT = 1u,
    NM_OWNER_BRAIN_ACK_INVALID_PROPOSAL = 2u,
    NM_OWNER_BRAIN_ACK_INVALID_WITNESS = 3u,
    NM_OWNER_BRAIN_ACK_INVALID_FAST_GATE = 4u,
    NM_OWNER_BRAIN_ACK_INVALID_PREFLIGHT = 5u,
    NM_OWNER_BRAIN_ACK_TOKEN_MISMATCH = 6u,
};

enum NMOwnerApplyStatus : nm_u32 {
    NM_OWNER_APPLY_PENDING = 0u,
    NM_OWNER_APPLY_ACCEPT = 1u,
    NM_OWNER_APPLY_REJECT = 2u,
    NM_OWNER_APPLY_TERMINAL_NO_TOUCH = 3u,
};

enum NMOwnerAppliedCode : nm_u32 {
    NM_OWNER_APPLIED_SUCCESS = 0u,
    NM_OWNER_APPLIED_FORCED_REJECT = 1u,
    NM_OWNER_APPLIED_PHYSICAL_REJECT = 2u,
    NM_OWNER_APPLIED_INVALID_OWNER = 3u,
    NM_OWNER_APPLIED_INVALID_BRAIN_ACK = 4u,
    NM_OWNER_APPLIED_TOKEN_MISMATCH = 5u,
    NM_OWNER_APPLIED_MATTER_REJECT = 6u,
    NM_OWNER_APPLIED_INVALID_MATTER_OUTCOME = 7u,
    NM_OWNER_APPLIED_BRAIN_REJECT = 8u,
};

enum NMMatterApplyCode : nm_u32 {
    NM_MATTER_APPLY_SUCCESS = 0u,
    NM_MATTER_APPLY_FORCED_REJECT = 1u,
    NM_MATTER_APPLY_PHYSICAL_REJECT = 2u,
    NM_MATTER_APPLY_INVALID_OWNER = 3u,
    NM_MATTER_APPLY_INVALID_BRAIN_ACK = 4u,
    NM_MATTER_APPLY_TOKEN_MISMATCH = 5u,
    NM_MATTER_APPLY_REJECT = 6u,
    NM_MATTER_APPLY_INVALID_OUTCOME = 7u,
    NM_MATTER_APPLY_BRAIN_REJECT = 8u,
};

enum NMJointPublicationFenceStatus : nm_u32 {
    NM_JOINT_PUBLICATION_PENDING = 0u,
    NM_JOINT_PUBLICATION_COMMITTED = 1u,
    NM_JOINT_PUBLICATION_FAILURE = 2u,
};

enum NMPreparedStatePublicationFactsStatus : nm_u32 {
    NM_PREPARED_STATE_PUBLICATION_FACTS_PENDING = 0u,
    NM_PREPARED_STATE_PUBLICATION_FACTS_VALID = 1u,
    NM_PREPARED_STATE_PUBLICATION_FACTS_REJECTED = 2u,
};

// Matter-local mirrors of the owner ABI4 proposal/ACK/apply records.  They are
// intentionally kept independent of MetalRobo headers and are statically
// compared by the adapter translation unit.
typedef struct NM_ALIGN16 NMOwnerProposalGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 decision;
    nm_u32 code;
    nm_u64 programFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 physicsTokenFingerprint;
    nm_u64 brainProgramFingerprint;
    nm_u64 brainShadowStateFingerprint;
    nm_u64 brainWitnessFingerprint;
    nm_u64 candidatePublicationFingerprint;
    nm_u64 humanIOIdentityFingerprint;
    nm_u32 environment;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 physicsSubstepCount;
    nm_u32 controlStep;
    nm_u64 proposalFingerprint;
} NMOwnerProposalGPU;

typedef struct NM_ALIGN16 NMOwnerBrainAckGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 decision;
    nm_u32 code;
    nm_u64 programFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 physicsTokenFingerprint;
    nm_u64 proposalFingerprint;
    nm_u64 preflightFingerprint;
    nm_u64 fastGateFingerprint;
    nm_u64 brainWitnessFingerprint;
    nm_u64 brainProgramFingerprint;
    nm_u32 environment;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 physicsSubstepCount;
    nm_u32 controlStep;
    nm_u64 ackFingerprint;
} NMOwnerBrainAckGPU;

typedef struct NM_ALIGN16 NMOwnerApplyActionGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 decision;
    nm_u32 code;
    nm_u64 programFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 physicsTokenFingerprint;
    nm_u64 proposalFingerprint;
    nm_u64 ackFingerprint;
    nm_u64 preflightFingerprint;
    nm_u64 fastGateFingerprint;
    nm_u64 brainWitnessFingerprint;
    nm_u32 environment;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 physicsSubstepCount;
    nm_u32 controlStep;
    nm_u64 actionFingerprint;
} NMOwnerApplyActionGPU;

typedef struct NM_ALIGN16 NMMatterApplyOutcomeGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 decision;
    nm_u32 code;
    nm_u64 programFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 physicsTokenFingerprint;
    nm_u64 proposalFingerprint;
    nm_u64 ackFingerprint;
    nm_u64 actionFingerprint;
    nm_u64 matterProgramFingerprint;
    nm_u64 reserved0;
    nm_u32 environment;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 physicsSubstepCount;
    nm_u32 controlStep;
    nm_u64 outcomeFingerprint;
} NMMatterApplyOutcomeGPU;

typedef struct NM_ALIGN16 NMOwnerAppliedOutcomeGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 decision;
    nm_u32 code;
    nm_u64 programFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 physicsTokenFingerprint;
    nm_u64 proposalFingerprint;
    nm_u64 ackFingerprint;
    nm_u64 preflightFingerprint;
    nm_u64 fastGateFingerprint;
    nm_u64 matterApplyFingerprint;
    nm_u32 environment;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 physicsSubstepCount;
    nm_u32 controlStep;
    nm_u64 appliedFingerprint;
} NMOwnerAppliedOutcomeGPU;

typedef struct NM_ALIGN16 NMJointPublicationFenceGPU {
    nm_u32 abiVersion;
    nm_u32 structBytes;
    nm_u32 status;
    nm_u32 environment;
    nm_u32 controlStep;
    nm_u32 substepIndex;
    nm_u32 physicsSubstepCount;
    nm_u32 reserved0;
    nm_u64 ownerProgramFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 physicsTokenFingerprint;
    nm_u64 brainProgramFingerprint;
    nm_u64 brainShadowStateFingerprint;
    nm_u64 brainWitnessFingerprint;
    nm_u64 appliedDecisionFingerprint;
    nm_u64 jointCommitFingerprint;
    nm_u64 brainGeneration;
    nm_u64 fenceFingerprint;
} NMJointPublicationFenceGPU;

// Matter-owned shared scalar result. It is populated only by an independently
// validated ACCEPT application and consumed by Runtime's completion helper;
// no borrowed proposal/ACK bytes are read on the CPU.
typedef struct NM_ALIGN16 NMPreparedStatePublicationFactsGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u64 physicsTokenFingerprint;
    nm_u64 brainProgramFingerprint;
    nm_u64 brainShadowStateFingerprint;
    nm_u64 brainWitnessFingerprint;
    nm_u64 matterApplyFingerprint;
    nm_u64 factsFingerprint;
} NMPreparedStatePublicationFactsGPU;

typedef struct NM_ALIGN16 NMPreparedStateBindingGPU {
    nm_u32 abiVersion;
    nm_u32 status;
    nm_u32 environment;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 controlStep;
    nm_u32 reserved2;

    nm_u64 transactionFingerprint;
    nm_u64 substepFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 adapterProgramFingerprint;
    nm_u64 transactionPolicyFingerprint;
    nm_u64 physicsTokenFingerprint;
    nm_u64 proofFingerprint;
} NMPreparedStateBindingGPU;

typedef struct NM_ALIGN16 NMPreparedStateApplyGPU {
    nm_u32 abiVersion;
    nm_u32 environmentCount;
    nm_u32 proposalStride;
    nm_u32 brainAckStride;
    nm_u32 applyActionStride;
    nm_u32 matterApplyOutcomeStride;
    nm_u32 proposedTokenStrideBytes;
    nm_u32 stepIndex;
    nm_u32 substepIndex;
    nm_u32 transactionSlot;
    nm_u32 physicsSubstepCount;
    nm_u32 controlStep;
    nm_u32 forceRestore;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u32 reserved2;
    nm_u64 ownerProgramFingerprint;
    nm_u64 transactionFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 matterProgramFingerprint;
    nm_u64 reserved3;
} NMPreparedStateApplyGPU;

enum NMPreparedStateRestoreFlags : nm_u32 {
    NM_PREPARED_STATE_RESTORE_SHARED = 1u << 0u,
};

typedef struct NM_ALIGN16 NMPreparedStateRestoreGPU {
    nm_u32 environmentCount;
    nm_u32 actionStride;
    nm_u32 flags;
    nm_u32 reserved0;
    nm_u64 bytesPerEnvironment;
    nm_u64 byteCount;
} NMPreparedStateRestoreGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(NMPreparedStateBindingGPU) == 96u);
static_assert(sizeof(NMPreparedStateRestoreGPU) == 32u);
static_assert(sizeof(NMOwnerProposalGPU) == NM_MATTER_OWNER_APPLY_RECORD_BYTES);
static_assert(sizeof(NMOwnerBrainAckGPU) == NM_MATTER_OWNER_APPLY_RECORD_BYTES);
static_assert(sizeof(NMOwnerApplyActionGPU) == NM_MATTER_OWNER_APPLY_RECORD_BYTES);
static_assert(sizeof(NMMatterApplyOutcomeGPU) == NM_MATTER_OWNER_APPLY_RECORD_BYTES);
static_assert(sizeof(NMOwnerAppliedOutcomeGPU) == NM_MATTER_OWNER_APPLY_RECORD_BYTES);
static_assert(sizeof(NMJointPublicationFenceGPU) ==
              NM_MATTER_OWNER_APPLY_RECORD_BYTES);
static_assert(sizeof(NMPreparedStatePublicationFactsGPU) == 64u);
static_assert(offsetof(NMPreparedStatePublicationFactsGPU,
                       factsFingerprint) == 56u);
static_assert(offsetof(NMOwnerProposalGPU, proposalFingerprint) == 120u);
static_assert(offsetof(NMOwnerBrainAckGPU, ackFingerprint) == 120u);
static_assert(offsetof(NMOwnerApplyActionGPU, actionFingerprint) == 120u);
static_assert(offsetof(NMMatterApplyOutcomeGPU, outcomeFingerprint) == 120u);
static_assert(offsetof(NMOwnerAppliedOutcomeGPU, appliedFingerprint) == 120u);
static_assert(offsetof(NMJointPublicationFenceGPU, fenceFingerprint) == 120u);
static_assert(sizeof(NMPreparedStateApplyGPU) == 112u);
#endif
