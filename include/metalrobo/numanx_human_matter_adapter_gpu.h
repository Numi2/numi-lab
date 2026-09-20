#pragma once

// Pointer-free NumanX Human/Matter adapter ABI.  The adapter owns transaction
// plumbing only: the Human owner computes exact candidates and A0 actions,
// Matter owns continuum acceptance, and an external proof producer hashes the
// accepted Human + Matter bytes on the same Metal timeline.

#include "metalrobo/numanx_coupled_human_gpu.h"

#define MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION 2u
#define MR_NUMANX_ACCEPTED_STATE_PROOF_BYTES 128u
#define MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES 64u

enum MRNumanXAcceptedStateProofStatus : mr_u32 {
    // Cleared storage is always pending and therefore fail-closed.
    MR_NUMANX_ACCEPTED_STATE_PROOF_PENDING = 0u,
    MR_NUMANX_ACCEPTED_STATE_PROOF_VALID = 1u,
    MR_NUMANX_ACCEPTED_STATE_PROOF_REJECTED = 2u,
};

enum MRNumanXHumanMatterAdapterCode : mr_u32 {
    MR_NUMANX_HUMAN_MATTER_ADAPTER_SUCCESS = 0u,
    MR_NUMANX_HUMAN_MATTER_ADAPTER_MISSING_STATE_PROOF =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 0x100u,
    MR_NUMANX_HUMAN_MATTER_ADAPTER_INVALID_STATE_PROOF =
        MR_NUMANX_COUPLED_HUMAN_SERVICE_CODE_BIT | 0x101u,
};

// GPU-derived content proof.  humanStateFingerprint and
// matterStateFingerprint must each hash the accepted state bytes owned by that
// runtime, not transaction metadata.  physicsStateFingerprint is the adapter
// domain-separated combination of those content hashes and the exact runtime
// identities. proofFingerprint is a deterministic integrity/replay witness
// for this complete record on a trusted same-device timeline; it is not a
// cryptographic authenticator.
typedef struct MR_ALIGN16 MRNumanXAcceptedStateProofGPU {
    mr_u32 abiVersion;
    mr_u32 structSize;
    mr_u32 status;
    mr_u32 environment;

    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 acceptedTimestampMicroseconds;
    mr_u64 physicsGeneration;

    mr_u64 humanStateFingerprint;
    mr_u64 matterStateFingerprint;
    mr_u64 physicsStateFingerprint;
    mr_u64 matterSourcePhysicsFingerprint;

    mr_u64 matterDeviceProgramFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 proofFingerprint;

    // Bound identity of the exact adapter program that requested this proof.
    // This occupies the former ABI-v1 reserved0 word.
    mr_u64 adapterProgramFingerprint;
    // Exact completion-boundary execution policy. It is folded into
    // physicsStateFingerprint; the canonical accepted token keeps reserved=0.
    mr_u64 transactionPolicyFingerprint;
} MRNumanXAcceptedStateProofGPU;

// Byte-for-byte compatible with NumiBrain's 64-byte
// NBAcceptedPhysicsStateToken.  This local definition intentionally avoids a
// source dependency between the two repositories while retaining static ABI
// verification in both.
typedef struct MR_ALIGN16 MRNumanXAcceptedPhysicsStateTokenGPU {
    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 physicsStateFingerprint;
    mr_u64 acceptedTimestampMicroseconds;
    mr_u64 physicsGeneration;
    mr_u32 environmentIdentifier;
    mr_u32 flags;
    mr_u64 reserved;
    mr_u64 tokenFingerprint;
} MRNumanXAcceptedPhysicsStateTokenGPU;

typedef struct MR_ALIGN16 MRNumanXHumanMatterAdapterDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 expectedMatterCompletedMicrosteps;
    mr_u32 matterSuccessCode;

    mr_u32 jointStatusStride;
    mr_u32 standStatusStride;
    mr_u32 matterOutcomeStride;
    mr_u32 acceptedTokenStrideBytes;

    mr_u32 worldStatusStride;
    mr_u32 acceptedStateProofStride;
    mr_u32 environmentIdentifierBase;
    mr_u32 flags;

    mr_u32 controlStep;
    mr_u32 physicsSubstep;
    mr_u32 physicsSubsteps;
    mr_u32 reserved0;

    mr_u64 programFingerprint;
    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 acceptedTimestampMicroseconds;
    mr_u64 physicsGeneration;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 matterSourcePhysicsFingerprint;
    mr_u64 matterDeviceProgramFingerprint;
    mr_u64 stateProofProgramFingerprint;
} MRNumanXHumanMatterAdapterDispatchGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(MRNumanXAcceptedStateProofGPU) ==
              MR_NUMANX_ACCEPTED_STATE_PROOF_BYTES);
static_assert(alignof(MRNumanXAcceptedStateProofGPU) == 16u);
static_assert(offsetof(MRNumanXAcceptedStateProofGPU, proofFingerprint) == 104u);
static_assert(sizeof(MRNumanXAcceptedPhysicsStateTokenGPU) ==
              MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES);
static_assert(alignof(MRNumanXAcceptedPhysicsStateTokenGPU) == 16u);
static_assert(offsetof(MRNumanXAcceptedPhysicsStateTokenGPU,
                       tokenFingerprint) == 56u);
static_assert(sizeof(MRNumanXHumanMatterAdapterDispatchGPU) == 144u);
static_assert(alignof(MRNumanXHumanMatterAdapterDispatchGPU) == 16u);
static_assert(offsetof(MRNumanXHumanMatterAdapterDispatchGPU,
                       programFingerprint) == 64u);
#endif
