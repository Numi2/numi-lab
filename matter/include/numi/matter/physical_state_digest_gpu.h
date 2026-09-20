#pragma once

#include "numi/matter/sha256_gpu.h"

#define NM_MATTER_PHYSICAL_STATE_DIGEST_ABI_VERSION 1u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_BYTES 160u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_SCHEMA_VERSION 1u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_MANIFEST_VERSION 1u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT 31u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_HUMAN_SOURCE_COUNT 4u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_MATTER_SOURCE_COUNT 27u
#define NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES 1024u

#define NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN 0u
#define NM_MATTER_PHYSICAL_STATE_TARGET_MATTER 1u
#define NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED (1u << 0u)
#define NM_MATTER_PHYSICAL_STATE_ENCODING_RAW_BYTES_V1 1u

// Stable source identities shared by the existing accepted-state proof and
// the direct-byte physical-state digest. The manifest table below is the
// canonical V1 order; changing an identity, target, flag, or ordinal requires
// a new physical-state manifest version.
enum NMPhysicalStateDigestSourceID : nm_u32 {
    NM_PHYSICAL_STATE_SOURCE_HUMAN_ROOT_TRANSLATION = 0x1004u,
    NM_PHYSICAL_STATE_SOURCE_HUMAN_Q = 0x1001u,
    NM_PHYSICAL_STATE_SOURCE_HUMAN_V = 0x1002u,
    NM_PHYSICAL_STATE_SOURCE_HUMAN_MUJOCO = 0x1003u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_PARTICLES = 0x2001u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_PARTICLE_MATERIAL = 0x2002u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_NODES = 0x2003u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_MATERIAL = 0x2004u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_FIELDS = 0x2005u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_VASCULAR_STATE = 0x201au,
    NM_PHYSICAL_STATE_SOURCE_MATTER_VASCULAR_CLOCK = 0x201bu,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_TETRAHEDRA = 0x2006u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_TOPOLOGY_NODES = 0x2007u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_COHESIVE_FACES = 0x2008u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_PUNCTURE_CHANNELS = 0x2009u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_TOPOLOGY_STATES = 0x200au,
    NM_PHYSICAL_STATE_SOURCE_MATTER_LEARNED_WEIGHTS = 0x200bu,
    NM_PHYSICAL_STATE_SOURCE_MATTER_LEARNED_REVISION = 0x200cu,
    NM_PHYSICAL_STATE_SOURCE_MATTER_CONTACT_HISTORIES = 0x200du,
    NM_PHYSICAL_STATE_SOURCE_MATTER_HUMAN_SUPPORT_HISTORIES = 0x2018u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_HUMAN_SUPPORT_CONSEQUENCES = 0x2019u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_DEFORMABLE_CONTACT_HISTORIES = 0x200eu,
    NM_PHYSICAL_STATE_SOURCE_MATTER_GENERALIZED_CANDIDATE = 0x200fu,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FRAME_REACTIONS = 0x2010u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_RIGID_STATES = 0x2017u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_ADAPTIVE_STATE = 0x2011u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_SCHEDULERS = 0x2012u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_IDENTIFICATION = 0x2013u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_ENVIRONMENT_PARAMETERS = 0x2014u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_NODE_INCIDENCE = 0x2015u,
    NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_NODE_RANGES = 0x2016u,
};

typedef struct NM_ALIGN16 NMPhysicalStateDigestManifestIdentityGPU {
    nm_u32 source;
    nm_u32 target;
    nm_u32 flags;
    nm_u32 reserved0;
} NMPhysicalStateDigestManifestIdentityGPU;

#if defined(__METAL_VERSION__)
constant NMPhysicalStateDigestManifestIdentityGPU
    kNMPhysicalStateDigestManifestV1[
        NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT] = {
#else
inline constexpr NMPhysicalStateDigestManifestIdentityGPU
    kNMPhysicalStateDigestManifestV1[
        NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT] = {
#endif
    {NM_PHYSICAL_STATE_SOURCE_HUMAN_ROOT_TRANSLATION,
     NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_HUMAN_Q,
     NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_HUMAN_V,
     NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_HUMAN_MUJOCO,
     NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_PARTICLES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_PARTICLE_MATERIAL,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_NODES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_MATERIAL,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_FIELDS,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_VASCULAR_STATE,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_VASCULAR_CLOCK,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_TETRAHEDRA,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_TOPOLOGY_NODES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_COHESIVE_FACES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_PUNCTURE_CHANNELS,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_TOPOLOGY_STATES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_LEARNED_WEIGHTS,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER,
     NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_LEARNED_REVISION,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER,
     NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_CONTACT_HISTORIES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_HUMAN_SUPPORT_HISTORIES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_HUMAN_SUPPORT_CONSEQUENCES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_DEFORMABLE_CONTACT_HISTORIES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_GENERALIZED_CANDIDATE,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FRAME_REACTIONS,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_RIGID_STATES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_ADAPTIVE_STATE,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_SCHEDULERS,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_IDENTIFICATION,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER,
     NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_ENVIRONMENT_PARAMETERS,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_NODE_INCIDENCE,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
    {NM_PHYSICAL_STATE_SOURCE_MATTER_FEM_NODE_RANGES,
     NM_MATTER_PHYSICAL_STATE_TARGET_MATTER, 0u, 0u},
};

enum NMPhysicalStateDigestStatus : nm_u32 {
    NM_PHYSICAL_STATE_DIGEST_PENDING = 0u,
    NM_PHYSICAL_STATE_DIGEST_VALID = 1u,
    NM_PHYSICAL_STATE_DIGEST_INVALID = 2u,
};

// Pointer-free canonical description of one logical state source. The source
// IDs are the stable AcceptedStateProofSource values. A zero element count is
// still emitted and hashed, so an inactive state family cannot disappear from
// the manifest silently. byteCount is logical bytes per environment for an
// environment-local source and total logical bytes for a shared source.
typedef struct NM_ALIGN16 NMPhysicalStateDigestSourceGPU {
    nm_u32 source;
    nm_u32 target;
    nm_u32 flags;
    nm_u32 encoding;
    nm_u64 elementCount;
    nm_u32 elementBytes;
    nm_u32 reserved0;
    nm_u64 byteCount;
    nm_u64 reserved1;
} NMPhysicalStateDigestSourceGPU;

typedef struct NM_ALIGN16 NMPhysicalStateDigestChunkGPU {
    nm_u32 environmentCount;
    nm_u32 environmentIdentifierBase;
    nm_u32 sourceOrdinal;
    nm_u32 chunkBytes;
    nm_u32 chunkCount;
    nm_u32 scratchStride;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u64 sourceCapacityBytes;
} NMPhysicalStateDigestChunkGPU;

// Canonical little-endian leaf metadata. Every byte is explicitly initialized
// before hashing; Apple Metal targets are little-endian by platform contract.
typedef struct NM_ALIGN16 NMPhysicalStateDigestLeafMetadataGPU {
    nm_u32 domainFamily;
    nm_u32 domainAlgorithm;
    nm_u32 schemaVersion;
    nm_u32 domainRole;
    NMPhysicalStateDigestSourceGPU source;
    nm_u32 sourceOrdinal;
    nm_u32 chunk;
    nm_u32 environment;
    nm_u32 reserved0;
    nm_u64 actualBytes;
    nm_u64 reserved1;
} NMPhysicalStateDigestLeafMetadataGPU;

typedef struct NM_ALIGN16 NMPhysicalStateDigestLeafFinalizeGPU {
    nm_u32 environmentCount;
    nm_u32 chunkCount;
    nm_u32 scratchStride;
    nm_u32 reserved0;
} NMPhysicalStateDigestLeafFinalizeGPU;

typedef struct NM_ALIGN16 NMPhysicalStateDigestReduceGPU {
    nm_u32 environmentCount;
    nm_u32 sourceOrdinal;
    nm_u32 inputCount;
    nm_u32 outputCount;
    nm_u32 scratchStride;
    nm_u32 level;
    nm_u32 reserved0;
    nm_u32 reserved1;
} NMPhysicalStateDigestReduceGPU;

typedef struct NM_ALIGN16 NMPhysicalStateDigestStoreGPU {
    nm_u32 environmentCount;
    nm_u32 sourceOrdinal;
    nm_u32 sourceCount;
    nm_u32 scratchStride;
} NMPhysicalStateDigestStoreGPU;

// Cursor fields are recorded beside the content digest for provenance. They
// are not hashed: transaction identity, generation, timestamps, GPU addresses
// and allocation slack must not prevent byte-identical retries or restorations
// from reproducing the same physical-state digest.
typedef struct NM_ALIGN16 NMPhysicalStateDigestFinalizeGPU {
    nm_u32 abiVersion;
    nm_u32 structSize;
    nm_u32 schemaVersion;
    nm_u32 manifestVersion;
    nm_u32 environmentCount;
    nm_u32 environmentIdentifierBase;
    nm_u32 sourceCount;
    nm_u32 clockDomain;
    nm_u32 clockQuantumNanoseconds;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u32 reserved2;
    nm_u64 acceptedTimestampNanoseconds;
    nm_u64 physicsGeneration;
    nm_u64 matterSourcePhysicsFingerprint;
    nm_u64 matterDeviceProgramFingerprint;
} NMPhysicalStateDigestFinalizeGPU;

// Direct-byte Merkle SHA-256 result. humanSHA256 and matterSHA256 are
// independently domain-separated owner roots; physicalSHA256 binds them under
// the complete manifest/domain header. The two 64-bit fingerprints are
// provenance only and are not substitutes for cryptographic source evidence.
typedef struct NM_ALIGN16 NMPhysicalStateDigestGPU {
    nm_u32 abiVersion;
    nm_u32 structSize;
    nm_u32 status;
    nm_u32 environment;
    nm_u32 schemaVersion;
    nm_u32 manifestVersion;
    nm_u32 sourceCount;
    nm_u32 reserved0;
    nm_u64 acceptedTimestampNanoseconds;
    nm_u64 physicsGeneration;
    nm_u64 matterSourcePhysicsFingerprint;
    nm_u64 matterDeviceProgramFingerprint;
    NMSHA256DigestGPU humanSHA256;
    NMSHA256DigestGPU matterSHA256;
    NMSHA256DigestGPU physicalSHA256;
} NMPhysicalStateDigestGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(NMPhysicalStateDigestSourceGPU) == 48u);
static_assert(sizeof(NMPhysicalStateDigestManifestIdentityGPU) == 16u);
static_assert(alignof(NMPhysicalStateDigestManifestIdentityGPU) == 16u);
static_assert(sizeof(kNMPhysicalStateDigestManifestV1) /
                  sizeof(kNMPhysicalStateDigestManifestV1[0]) ==
              NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT);
static_assert(alignof(NMPhysicalStateDigestSourceGPU) == 16u);
static_assert(offsetof(NMPhysicalStateDigestSourceGPU, byteCount) == 32u);
static_assert(offsetof(NMPhysicalStateDigestSourceGPU, reserved1) == 40u);
static_assert(sizeof(NMPhysicalStateDigestChunkGPU) == 48u);
static_assert(sizeof(NMPhysicalStateDigestLeafMetadataGPU) == 96u);
static_assert(sizeof(NMPhysicalStateDigestLeafFinalizeGPU) == 16u);
static_assert(sizeof(NMPhysicalStateDigestReduceGPU) == 32u);
static_assert(sizeof(NMPhysicalStateDigestStoreGPU) == 16u);
static_assert(sizeof(NMPhysicalStateDigestFinalizeGPU) == 80u);
static_assert(sizeof(NMPhysicalStateDigestGPU) ==
              NM_MATTER_PHYSICAL_STATE_DIGEST_BYTES);
static_assert(alignof(NMPhysicalStateDigestGPU) == 16u);
static_assert(offsetof(NMPhysicalStateDigestGPU, humanSHA256) == 64u);
static_assert(offsetof(NMPhysicalStateDigestGPU, physicalSHA256) == 128u);
#endif
