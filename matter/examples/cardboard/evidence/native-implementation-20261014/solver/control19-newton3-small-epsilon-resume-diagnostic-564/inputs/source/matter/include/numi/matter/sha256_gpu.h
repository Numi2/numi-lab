#pragma once

#include "numi/matter/shared.h"

#define NM_MATTER_SHA256_GPU_ABI_VERSION 1u
#define NM_MATTER_SHA256_CONTEXT_BYTES 112u
#define NM_MATTER_SHA256_DIGEST_BYTES 32u

enum NMSHA256ContextStatus : nm_u32 {
    NM_SHA256_CONTEXT_UNINITIALIZED = 0u,
    NM_SHA256_CONTEXT_READY = 1u,
    NM_SHA256_CONTEXT_FINALIZED = 2u,
    NM_SHA256_CONTEXT_INVALID = 3u,
};

// Fixed-size, allocation-free incremental SHA-256 state. blockWords holds the
// pending message bytes in SHA-256 (big-endian) word order. totalBytes counts
// message bytes only; final padding never changes it.
typedef struct NM_ALIGN16 NMSHA256ContextGPU {
    nm_u32 state[8];
    nm_u32 blockWords[16];
    nm_u64 totalBytes;
    nm_u32 blockBytes;
    nm_u32 status;
} NMSHA256ContextGPU;

// Canonical digest bytes, in the byte order used by FIPS 180-4 and common
// hexadecimal SHA-256 renderings. Do not reinterpret these as host-endian
// words.
#if defined(__METAL_VERSION__)
typedef uchar nm_sha256_u8;
#else
#include <cstdint>
typedef std::uint8_t nm_sha256_u8;
#endif

typedef struct NM_ALIGN16 NMSHA256DigestGPU {
    nm_sha256_u8 bytes[NM_MATTER_SHA256_DIGEST_BYTES];
} NMSHA256DigestGPU;

// One exact source range to append to the context with the same index. A
// sequence of update dispatches hashes the bytewise concatenation of their
// spans; a zero-byte span is a no-op.
typedef struct NM_ALIGN16 NMSHA256SpanGPU {
    nm_u64 sourceOffsetBytes;
    nm_u64 byteCount;
} NMSHA256SpanGPU;

typedef struct NM_ALIGN16 NMSHA256DispatchGPU {
    nm_u32 contextCount;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u32 reserved2;
} NMSHA256DispatchGPU;

typedef struct NM_ALIGN16 NMSHA256UpdateGPU {
    nm_u32 contextCount;
    nm_u32 reserved0;
    nm_u64 sourceCapacityBytes;
} NMSHA256UpdateGPU;

#if !defined(__METAL_VERSION__)
#include <cstddef>
static_assert(sizeof(NMSHA256ContextGPU) == NM_MATTER_SHA256_CONTEXT_BYTES);
static_assert(alignof(NMSHA256ContextGPU) == 16u);
static_assert(offsetof(NMSHA256ContextGPU, totalBytes) == 96u);
static_assert(offsetof(NMSHA256ContextGPU, status) == 108u);
static_assert(sizeof(NMSHA256DigestGPU) == NM_MATTER_SHA256_DIGEST_BYTES);
static_assert(alignof(NMSHA256DigestGPU) == 16u);
static_assert(sizeof(NMSHA256SpanGPU) == 16u);
static_assert(alignof(NMSHA256SpanGPU) == 16u);
static_assert(sizeof(NMSHA256DispatchGPU) == 16u);
static_assert(sizeof(NMSHA256UpdateGPU) == 16u);
#endif
