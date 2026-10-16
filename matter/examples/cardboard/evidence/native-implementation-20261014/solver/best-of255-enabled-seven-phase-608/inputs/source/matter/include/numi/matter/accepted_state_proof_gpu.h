#pragma once

#include "numi/matter/shared.h"

#define NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION 2u
#define NM_MATTER_ACCEPTED_STATE_PROOF_BYTES 128u
#define NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION_V2 3u
#define NM_MATTER_ACCEPTED_STATE_PROOF_V2_BYTES 160u
#define NM_MATTER_EXACT_INBOUND_AUTHORITY_ABI_VERSION_V2 2u
#define NM_MATTER_EXACT_INBOUND_AUTHORITY_V2_BYTES 112u
#define NM_MATTER_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS 2u
#define NM_MATTER_EXACT_CLOCK_QUANTUM_NANOSECONDS 1u
#define NM_MATTER_FINGERPRINT_DOMAIN_EXACT_INBOUND_AUTHORITY_V2 0x4e584941u
#define NM_MATTER_FINGERPRINT_DOMAIN_PHYSICS_STATE_V2 0x4e585053u
#define NM_MATTER_FINGERPRINT_DOMAIN_ACCEPTED_STATE_PROOF_V2 0x4e584150u
#define NM_MATTER_FINGERPRINT_DOMAIN_ACCEPTED_PHYSICS_TOKEN_V2 0x4e584154u
#define NM_MATTER_ACCEPTED_PHYSICS_TOKEN_VERSION_V2 2u

enum NMAcceptedStateProofStatus : nm_u32 {
    NM_ACCEPTED_STATE_PROOF_PENDING = 0u,
    NM_ACCEPTED_STATE_PROOF_VALID = 1u,
    NM_ACCEPTED_STATE_PROOF_REJECTED = 2u,
};

// Matter-local, pointer-free mirror of the joint Human/Matter proof record.
// Both repositories assert exact layout parity; neither needs to import the
// other's C++ API to produce or validate device bytes.
typedef struct NM_ALIGN16 NMAcceptedStateProofGPU {
    nm_u32 abiVersion;
    nm_u32 structSize;
    nm_u32 status;
    nm_u32 environment;

    nm_u64 transactionFingerprint;
    nm_u64 substepFingerprint;
    nm_u64 acceptedTimestampMicroseconds;
    nm_u64 physicsGeneration;

    nm_u64 humanStateFingerprint;
    nm_u64 matterStateFingerprint;
    nm_u64 physicsStateFingerprint;
    nm_u64 matterSourcePhysicsFingerprint;

    nm_u64 matterDeviceProgramFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 proofFingerprint;

    // Exact adapter program identity. Together with the record FNV this is a
    // deterministic integrity/replay witness on a trusted same-device
    // timeline; it is not cryptographic authentication.
    nm_u64 adapterProgramFingerprint;
    // Exact completion-boundary execution policy used to produce the state.
    // It is already folded into physicsStateFingerprint; the canonical
    // 64-byte accepted-state token keeps its reserved word zero.
    nm_u64 transactionPolicyFingerprint;
} NMAcceptedStateProofGPU;

// Matter-local mirror of HumanIO's device-private exact admission receipt.
// The host may bind this record but cannot read or synthesize its terminal
// fingerprint; the V2 finalizer validates every field on the same Metal
// command-buffer timeline that produced it.
typedef struct NM_ALIGN16 NMExactInboundAuthorityGPUV2 {
    nm_u32 abiVersion;
    nm_u32 structSize;
    nm_u32 clockDomain;
    nm_u32 clockQuantumNanoseconds;
    nm_u64 acceptedBrainTimestampNanoseconds;
    nm_u64 brainGeneration;
    nm_u64 transactionFingerprint;
    nm_u64 substepFingerprint;
    nm_u64 motorCandidateFingerprint;
    nm_u64 motorOutputFingerprint;
    nm_u64 motorProfileFingerprint;
    nm_u64 motorReadyGateFingerprint;
    nm_u64 brainProgramFingerprint;
    nm_u64 fastProgramFingerprint;
    nm_u64 decisionGateFingerprint;
    nm_u64 inboundAuthorityFingerprint;
} NMExactInboundAuthorityGPUV2;

// Exact-clock proof family. This is intentionally a distinct record rather
// than a mode bit on the microsecond proof: clocks, HumanIO receipt identity,
// and the motor candidate all participate in the terminal fingerprint.
typedef struct NM_ALIGN16 NMAcceptedStateProofGPUV2 {
    nm_u32 abiVersion;
    nm_u32 structSize;
    nm_u32 status;
    nm_u32 environment;

    nm_u64 transactionFingerprint;
    nm_u64 substepFingerprint;
    nm_u64 acceptedTimestampNanoseconds;
    nm_u64 physicsGeneration;

    nm_u32 clockDomain;
    nm_u32 clockQuantumNanoseconds;

    nm_u64 humanStateFingerprint;
    nm_u64 matterStateFingerprint;
    nm_u64 physicsStateFingerprint;
    nm_u64 matterSourcePhysicsFingerprint;
    nm_u64 matterDeviceProgramFingerprint;
    nm_u64 stateProofProgramFingerprint;
    nm_u64 adapterProgramFingerprint;
    nm_u64 transactionPolicyFingerprint;
    nm_u64 linearizationEpoch;
    nm_u64 slotGeneration;
    nm_u64 motorCandidateFingerprint;
    nm_u64 inboundAuthorityFingerprint;
    nm_u64 proofFingerprint;
} NMAcceptedStateProofGPUV2;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(NMAcceptedStateProofGPU) ==
              NM_MATTER_ACCEPTED_STATE_PROOF_BYTES);
static_assert(alignof(NMAcceptedStateProofGPU) == 16u);
static_assert(offsetof(NMAcceptedStateProofGPU, proofFingerprint) == 104u);
static_assert(sizeof(NMExactInboundAuthorityGPUV2) ==
              NM_MATTER_EXACT_INBOUND_AUTHORITY_V2_BYTES);
static_assert(alignof(NMExactInboundAuthorityGPUV2) == 16u);
static_assert(offsetof(NMExactInboundAuthorityGPUV2,
                       inboundAuthorityFingerprint) == 104u);
static_assert(sizeof(NMAcceptedStateProofGPUV2) ==
              NM_MATTER_ACCEPTED_STATE_PROOF_V2_BYTES);
static_assert(alignof(NMAcceptedStateProofGPUV2) == 16u);
static_assert(offsetof(NMAcceptedStateProofGPUV2, clockDomain) == 48u);
static_assert(offsetof(NMAcceptedStateProofGPUV2,
                       motorCandidateFingerprint) == 136u);
static_assert(offsetof(NMAcceptedStateProofGPUV2,
                       proofFingerprint) == 152u);
#endif
