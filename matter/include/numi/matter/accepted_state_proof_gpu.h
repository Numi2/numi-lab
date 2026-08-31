#pragma once

#include "numi/matter/shared.h"

#define NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION 2u
#define NM_MATTER_ACCEPTED_STATE_PROOF_BYTES 128u

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

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(NMAcceptedStateProofGPU) ==
              NM_MATTER_ACCEPTED_STATE_PROOF_BYTES);
static_assert(alignof(NMAcceptedStateProofGPU) == 16u);
static_assert(offsetof(NMAcceptedStateProofGPU, proofFingerprint) == 104u);
#endif
