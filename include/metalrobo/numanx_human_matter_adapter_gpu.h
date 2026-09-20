#pragma once

// Pointer-free NumanX Human/Matter adapter ABI.  The adapter owns transaction
// plumbing only: the Human owner computes exact candidates and A0 actions,
// Matter owns continuum acceptance, and an external proof producer hashes the
// accepted Human + Matter bytes on the same Metal timeline.

#include "metalrobo/numanx_coupled_human_gpu.h"

#define MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION 2u
#define MR_NUMANX_ACCEPTED_STATE_PROOF_BYTES 128u
#define MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES 64u
#define MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION 3u
#define MR_NUMANX_ACCEPTED_STATE_PROOF_V2_BYTES 160u
#define MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_V2_BYTES 64u
#define MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_VERSION_V2 2u
#define MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS 2u
#define MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS 1u
#define MR_NUMANX_FINGERPRINT_DOMAIN_PHYSICS_STATE_V2 0x4e585053u
#define MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_STATE_PROOF_V2 0x4e584150u
#define MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_PHYSICS_TOKEN_V2 0x4e584154u

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

// Additive exact-clock proof. Its terminal fingerprint follows every explicit
// field, including the state-proof producer, canonical inbound Brain authority,
// and both clock words; no padding or host pointer contributes to the identity.
// The v1 proof remains byte-for-byte unchanged for existing microsecond
// transactions.
typedef struct MR_ALIGN16 MRNumanXAcceptedStateProofGPUV2 {
    mr_u32 abiVersion;
    mr_u32 structSize;
    mr_u32 status;
    mr_u32 environment;

    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 acceptedTimestampNanoseconds;
    mr_u64 physicsGeneration;

    mr_u32 clockDomain;
    mr_u32 clockQuantumNanoseconds;

    mr_u64 humanStateFingerprint;
    mr_u64 matterStateFingerprint;
    mr_u64 physicsStateFingerprint;
    mr_u64 matterSourcePhysicsFingerprint;
    mr_u64 matterDeviceProgramFingerprint;
    mr_u64 stateProofProgramFingerprint;
    mr_u64 adapterProgramFingerprint;
    mr_u64 transactionPolicyFingerprint;
    mr_u64 linearizationEpoch;
    mr_u64 slotGeneration;
    mr_u64 motorCandidateFingerprint;
    mr_u64 inboundAuthorityFingerprint;
    mr_u64 proofFingerprint;
} MRNumanXAcceptedStateProofGPUV2;

// Exact-clock successor to NBAcceptedPhysicsStateToken. Clock domain and
// quantum occupy the legacy reserved 64-bit word, preserving the 64-byte,
// 16-byte-aligned lease shape while giving the fingerprint a distinct version
// and an unambiguous time unit.
typedef struct MR_ALIGN16 MRNumanXAcceptedPhysicsStateTokenGPUV2 {
    mr_u64 transactionFingerprint;
    mr_u64 substepFingerprint;
    mr_u64 physicsStateFingerprint;
    mr_u64 acceptedTimestampNanoseconds;
    mr_u64 physicsGeneration;
    mr_u32 environmentIdentifier;
    mr_u32 flags;
    mr_u32 clockDomain;
    mr_u32 clockQuantumNanoseconds;
    mr_u64 tokenFingerprint;
} MRNumanXAcceptedPhysicsStateTokenGPUV2;

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
static_assert(sizeof(MRNumanXAcceptedStateProofGPUV2) ==
              MR_NUMANX_ACCEPTED_STATE_PROOF_V2_BYTES);
static_assert(alignof(MRNumanXAcceptedStateProofGPUV2) == 16u);
static_assert(offsetof(MRNumanXAcceptedStateProofGPUV2,
                       clockDomain) == 48u);
static_assert(offsetof(MRNumanXAcceptedStateProofGPUV2,
                       motorCandidateFingerprint) == 136u);
static_assert(offsetof(MRNumanXAcceptedStateProofGPUV2,
                       proofFingerprint) == 152u);
static_assert(sizeof(MRNumanXAcceptedPhysicsStateTokenGPUV2) ==
              MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_V2_BYTES);
static_assert(alignof(MRNumanXAcceptedPhysicsStateTokenGPUV2) == 16u);
static_assert(offsetof(MRNumanXAcceptedPhysicsStateTokenGPUV2,
                       clockDomain) == 48u);
static_assert(offsetof(MRNumanXAcceptedPhysicsStateTokenGPUV2,
                       tokenFingerprint) == 56u);
static_assert(sizeof(MRNumanXHumanMatterAdapterDispatchGPU) == 144u);
static_assert(alignof(MRNumanXHumanMatterAdapterDispatchGPU) == 16u);
static_assert(offsetof(MRNumanXHumanMatterAdapterDispatchGPU,
                       programFingerprint) == 64u);
#endif
