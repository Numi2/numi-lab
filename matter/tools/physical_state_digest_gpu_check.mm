#include "numi/matter/accepted_state_proof_gpu.h"
#include "numi/matter/physical_state_digest_gpu.h"

#import <CommonCrypto/CommonDigest.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;
using Digest = std::array<std::uint8_t, NM_MATTER_SHA256_DIGEST_BYTES>;

constexpr std::uint32_t kDomainFamily = 0x5350584eu;
constexpr std::uint32_t kDomainAlgorithm = 0x32414853u;
constexpr std::uint32_t kLeafDomain = 1u;
constexpr std::uint32_t kNodeDomain = 2u;
constexpr std::uint32_t kOwnerDomain = 3u;
constexpr std::uint32_t kRootDomain = 4u;
constexpr std::uint32_t kEnvironmentCount = 2u;
constexpr std::uint32_t kEnvironmentIdentifierBase = 41u;

constexpr std::array<std::uint32_t,
    NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT> kSourceIdentifiers{{
    0x1004u, 0x1001u, 0x1002u, 0x1003u,
    0x2001u, 0x2002u, 0x2003u, 0x2004u, 0x2005u, 0x201au,
    0x201bu, 0x2006u, 0x2007u, 0x2008u, 0x2009u, 0x200au,
    0x200bu, 0x200cu, 0x200du, 0x2018u, 0x2019u, 0x200eu,
    0x200fu, 0x2010u, 0x2017u, 0x2011u, 0x2012u, 0x2013u,
    0x2014u, 0x2015u, 0x2016u,
}};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string errorText(NSError* error) {
    if (error == nil || error.localizedDescription == nil ||
        error.localizedDescription.UTF8String == nullptr) {
        return "unknown";
    }
    return error.localizedDescription.UTF8String;
}

void appendU32(Bytes& bytes, const std::uint32_t value) {
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }
}

void appendU64(Bytes& bytes, const std::uint64_t value) {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }
}

void appendDigest(Bytes& bytes, const Digest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

void appendDomain(Bytes& bytes, const std::uint32_t role) {
    appendU32(bytes, kDomainFamily);
    appendU32(bytes, kDomainAlgorithm);
    appendU32(bytes, NM_MATTER_PHYSICAL_STATE_DIGEST_SCHEMA_VERSION);
    appendU32(bytes, role);
}

void appendSource(
    Bytes& bytes,
    const NMPhysicalStateDigestSourceGPU& source
) {
    appendU32(bytes, source.source);
    appendU32(bytes, source.target);
    appendU32(bytes, source.flags);
    appendU32(bytes, source.encoding);
    appendU64(bytes, source.elementCount);
    appendU32(bytes, source.elementBytes);
    appendU32(bytes, source.reserved0);
    appendU64(bytes, source.byteCount);
    appendU64(bytes, source.reserved1);
}

void appendHeader(
    Bytes& bytes,
    const NMPhysicalStateDigestFinalizeGPU& pass,
    const std::uint32_t environment
) {
    appendU32(bytes, pass.schemaVersion);
    appendU32(bytes, pass.manifestVersion);
    appendU32(bytes, pass.environmentCount);
    appendU32(bytes, pass.environmentIdentifierBase + environment);
    appendU32(bytes, pass.sourceCount);
    appendU32(bytes, pass.clockDomain);
    appendU32(bytes, pass.clockQuantumNanoseconds);
}

Digest sha256(const Bytes& bytes) {
    require(bytes.size() <= std::numeric_limits<CC_LONG>::max(),
        "CPU SHA-256 oracle input is too large");
    Digest digest{};
    static constexpr std::uint8_t kEmpty = 0u;
    const void* input = bytes.empty()
        ? static_cast<const void*>(&kEmpty)
        : static_cast<const void*>(bytes.data());
    require(CC_SHA256(input, static_cast<CC_LONG>(bytes.size()),
                digest.data()) != nullptr,
        "CommonCrypto SHA-256 failed");
    return digest;
}

struct Fixture {
    NMPhysicalStateDigestSourceGPU descriptor{};
    Bytes bytes;
};

std::array<Fixture, NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT>
makeFixtures() {
    std::array<Fixture, NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT>
        fixtures{};
    for (std::size_t index = 0u; index < fixtures.size(); ++index) {
        const bool shared = index == 16u || index == 17u || index == 27u;
        const std::uint32_t elementBytes =
            std::array<std::uint32_t, 4u>{{1u, 2u, 4u, 8u}}[index % 4u];
        std::uint64_t elementCount = 3u + index;
        if (index == 1u || index == 27u) elementCount = 0u;
        if (index == 2u) elementCount = 513u;
        if (index == 3u) elementCount = 256u;
        if (index == 16u) elementCount = 2057u;
        const std::uint64_t byteCount =
            elementCount * static_cast<std::uint64_t>(elementBytes);
        fixtures[index].descriptor = NMPhysicalStateDigestSourceGPU{
            .source = kSourceIdentifiers[index],
            .target = index < NM_MATTER_PHYSICAL_STATE_DIGEST_HUMAN_SOURCE_COUNT
                ? NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN
                : NM_MATTER_PHYSICAL_STATE_TARGET_MATTER,
            .flags = shared ? NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED : 0u,
            .encoding = NM_MATTER_PHYSICAL_STATE_ENCODING_RAW_BYTES_V1,
            .elementCount = elementCount,
            .elementBytes = elementBytes,
            .reserved0 = 0u,
            .byteCount = byteCount,
            .reserved1 = 0u,
        };
        const std::uint64_t totalBytes = shared
            ? byteCount : byteCount * kEnvironmentCount;
        require(totalBytes <= std::numeric_limits<std::size_t>::max(),
            "synthetic source is too large");
        fixtures[index].bytes.resize(static_cast<std::size_t>(totalBytes));
        for (std::size_t byte = 0u; byte < fixtures[index].bytes.size(); ++byte) {
            const std::uint64_t environment = shared || byteCount == 0u
                ? 0u : byte / byteCount;
            const std::uint64_t local = byteCount == 0u
                ? 0u : byte % byteCount;
            fixtures[index].bytes[byte] = static_cast<std::uint8_t>(
                (kSourceIdentifiers[index] * 17u + environment * 71u +
                    local * 29u + (local >> 3u)) & 0xffu);
        }
    }
    return fixtures;
}

Digest sourceRoot(
    const Fixture& fixture,
    const std::uint32_t sourceOrdinal,
    const std::uint32_t environment
) {
    const auto& source = fixture.descriptor;
    const bool shared =
        (source.flags & NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED) != 0u;
    const std::uint64_t sourceBase = shared
        ? 0u : static_cast<std::uint64_t>(environment) * source.byteCount;
    const std::uint32_t chunkCount = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(1u,
            source.byteCount / NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES +
                (source.byteCount %
                    NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES != 0u)));
    std::vector<Digest> levelDigests;
    levelDigests.reserve(chunkCount);
    for (std::uint32_t chunk = 0u; chunk < chunkCount; ++chunk) {
        const std::uint64_t chunkBase =
            static_cast<std::uint64_t>(chunk) *
            NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES;
        const std::uint64_t actualBytes = chunkBase < source.byteCount
            ? std::min<std::uint64_t>(
                  source.byteCount - chunkBase,
                  NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES)
            : 0u;
        Bytes message;
        message.reserve(static_cast<std::size_t>(96u + actualBytes));
        appendDomain(message, kLeafDomain);
        appendSource(message, source);
        appendU32(message, sourceOrdinal);
        appendU32(message, chunk);
        appendU32(message, kEnvironmentIdentifierBase + environment);
        appendU32(message, 0u);
        appendU64(message, actualBytes);
        appendU64(message, 0u);
        require(message.size() == sizeof(NMPhysicalStateDigestLeafMetadataGPU),
            "CPU leaf metadata serialization is not 96 bytes");
        if (actualBytes != 0u) {
            const auto begin = fixture.bytes.begin() +
                static_cast<std::ptrdiff_t>(sourceBase + chunkBase);
            message.insert(message.end(), begin,
                begin + static_cast<std::ptrdiff_t>(actualBytes));
        }
        levelDigests.push_back(sha256(message));
    }

    std::uint32_t level = 0u;
    while (levelDigests.size() > 1u) {
        std::vector<Digest> next;
        next.reserve((levelDigests.size() + 1u) / 2u);
        for (std::size_t node = 0u; node < levelDigests.size(); node += 2u) {
            const bool hasRight = node + 1u < levelDigests.size();
            Bytes message;
            appendDomain(message, kNodeDomain);
            appendU32(message,
                NM_MATTER_PHYSICAL_STATE_DIGEST_SCHEMA_VERSION);
            appendSource(message, source);
            appendU32(message, sourceOrdinal);
            appendU32(message, level);
            appendU32(message, hasRight ? 2u : 1u);
            appendDigest(message, levelDigests[node]);
            if (hasRight) appendDigest(message, levelDigests[node + 1u]);
            next.push_back(sha256(message));
        }
        levelDigests = std::move(next);
        ++level;
    }
    return levelDigests.front();
}

struct OracleRoots {
    Digest human{};
    Digest matter{};
    Digest physical{};
};

OracleRoots oracleRoots(
    const std::array<Fixture,
        NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT>& fixtures,
    const NMPhysicalStateDigestFinalizeGPU& pass,
    const std::uint32_t environment
) {
    std::array<Digest, NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT> roots{};
    for (std::size_t source = 0u; source < fixtures.size(); ++source) {
        roots[source] = sourceRoot(fixtures[source],
            static_cast<std::uint32_t>(source), environment);
    }
    const auto ownerRoot = [&](const std::uint32_t target,
                               const std::uint32_t expectedCount) {
        Bytes message;
        appendDomain(message, kOwnerDomain);
        appendHeader(message, pass, environment);
        appendU32(message, target);
        appendU32(message, expectedCount);
        std::uint32_t count = 0u;
        for (std::size_t source = 0u; source < fixtures.size(); ++source) {
            if (fixtures[source].descriptor.target != target) continue;
            appendSource(message, fixtures[source].descriptor);
            appendDigest(message, roots[source]);
            ++count;
        }
        require(count == expectedCount, "CPU owner source count mismatch");
        return sha256(message);
    };

    OracleRoots result;
    result.human = ownerRoot(NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN,
        NM_MATTER_PHYSICAL_STATE_DIGEST_HUMAN_SOURCE_COUNT);
    result.matter = ownerRoot(NM_MATTER_PHYSICAL_STATE_TARGET_MATTER,
        NM_MATTER_PHYSICAL_STATE_DIGEST_MATTER_SOURCE_COUNT);
    Bytes physical;
    appendDomain(physical, kRootDomain);
    appendHeader(physical, pass, environment);
    appendDigest(physical, result.human);
    appendDigest(physical, result.matter);
    result.physical = sha256(physical);
    return result;
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    NSString* qualified = [@"numi_matter_metal::"
        stringByAppendingString:name];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    require(function != nil,
        "missing physical digest Metal function " +
            std::string(name.UTF8String));
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil,
        "failed to create physical digest pipeline " +
            std::string(name.UTF8String) + ": " + errorText(error));
    return result;
}

void dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipelineState,
    const std::size_t count
) {
    require(count != 0u, "physical digest dispatch cannot be empty");
    [encoder setComputePipelineState:pipelineState];
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            std::min<std::size_t>(
                count, pipelineState.maxTotalThreadsPerThreadgroup),
            1u, 1u)];
}

void requireDigest(
    const NMSHA256DigestGPU& actual,
    const Digest& expected,
    const std::string& label
) {
    require(std::memcmp(actual.bytes, expected.data(), expected.size()) == 0,
        label + " SHA-256 mismatch");
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            auto fixtures = makeFixtures();
            for (std::size_t index = 0u; index < fixtures.size(); ++index) {
                const bool shared =
                    index == 16u || index == 17u || index == 27u;
                const std::uint32_t target =
                    index < NM_MATTER_PHYSICAL_STATE_DIGEST_HUMAN_SOURCE_COUNT
                    ? NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN
                    : NM_MATTER_PHYSICAL_STATE_TARGET_MATTER;
                const auto& identity =
                    kNMPhysicalStateDigestManifestV1[index];
                require(identity.source == kSourceIdentifiers[index] &&
                            identity.target == target &&
                            identity.flags ==
                                (shared
                                    ? NM_MATTER_PHYSICAL_STATE_SOURCE_SHARED
                                    : 0u) &&
                            identity.reserved0 == 0u,
                    "public physical-state manifest differs from the "
                    "independent oracle identity");
            }
            std::uint32_t scratchStride = 1u;
            bool exercisedOddChunkTree = false;
            for (const auto& fixture : fixtures) {
                const auto byteCount = fixture.descriptor.byteCount;
                const auto chunks = static_cast<std::uint32_t>(
                    std::max<std::uint64_t>(1u,
                        byteCount /
                                NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES +
                            (byteCount %
                                NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES !=
                             0u)));
                scratchStride = std::max(scratchStride, chunks);
                exercisedOddChunkTree = exercisedOddChunkTree ||
                    (chunks > 1u && (chunks & 1u) != 0u);
            }
            require(exercisedOddChunkTree,
                "fixtures do not exercise an odd-width chunk tree");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:@NUMI_PHYSICAL_DIGEST_CHECK_METALLIB]
                error:&error];
            require(library != nil,
                "physical digest Metal library unavailable: " +
                    errorText(error));
            id<MTLComputePipelineState> leafInitialize = pipeline(
                device, library, @"nm_physical_state_digest_leaf_initialize");
            id<MTLComputePipelineState> leafMetadata = pipeline(
                device, library, @"nm_physical_state_digest_leaf_metadata");
            id<MTLComputePipelineState> leafBegin = pipeline(
                device, library, @"nm_physical_state_digest_leaf_begin");
            id<MTLComputePipelineState> chunks = pipeline(
                device, library, @"nm_physical_state_digest_chunks");
            id<MTLComputePipelineState> leafFinalize = pipeline(
                device, library, @"nm_physical_state_digest_leaf_finalize");
            id<MTLComputePipelineState> reduce = pipeline(
                device, library, @"nm_physical_state_digest_reduce");
            id<MTLComputePipelineState> store = pipeline(
                device, library, @"nm_physical_state_digest_store");
            id<MTLComputePipelineState> finalize = pipeline(
                device, library, @"nm_physical_state_digest_finalize");

            const auto sharedBuffer = [device](const void* bytes,
                                                const std::size_t size) {
                id<MTLBuffer> result = [device newBufferWithBytes:bytes
                    length:size options:MTLResourceStorageModeShared];
                require(result != nil, "Metal shared-buffer allocation failed");
                return result;
            };
            std::array<id<MTLBuffer>,
                NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT> sourceBuffers{};
            static constexpr std::uint8_t kEmptySource = 0xa5u;
            for (std::size_t index = 0u; index < fixtures.size(); ++index) {
                const void* bytes = fixtures[index].bytes.empty()
                    ? static_cast<const void*>(&kEmptySource)
                    : static_cast<const void*>(fixtures[index].bytes.data());
                sourceBuffers[index] = sharedBuffer(bytes,
                    std::max<std::size_t>(fixtures[index].bytes.size(), 1u));
            }

            const std::size_t scratchCount =
                static_cast<std::size_t>(kEnvironmentCount) * scratchStride;
            id<MTLBuffer> contexts = [device newBufferWithLength:
                scratchCount * sizeof(NMSHA256ContextGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> metadata = [device newBufferWithLength:
                scratchCount * sizeof(NMPhysicalStateDigestLeafMetadataGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> scratchA = [device newBufferWithLength:
                scratchCount * sizeof(NMSHA256DigestGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> scratchB = [device newBufferWithLength:
                scratchCount * sizeof(NMSHA256DigestGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> sourceRoots = [device newBufferWithLength:
                kEnvironmentCount * fixtures.size() * sizeof(NMSHA256DigestGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> output = [device newBufferWithLength:
                kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU)
                options:MTLResourceStorageModeShared];
            require(contexts != nil && metadata != nil && scratchA != nil &&
                    scratchB != nil && sourceRoots != nil && output != nil,
                "physical digest working-buffer allocation failed");
            std::memset(metadata.contents, 0xa5,
                scratchCount * sizeof(NMPhysicalStateDigestLeafMetadataGPU));
            std::memset(sourceRoots.contents, 0xa5,
                kEnvironmentCount * fixtures.size() *
                    sizeof(NMSHA256DigestGPU));
            std::memset(output.contents, 0,
                kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU));

            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> baselineEncoder =
                [command computeCommandEncoder];
            require(queue != nil && command != nil && baselineEncoder != nil,
                "physical digest command allocation failed");

            const auto encodeSource = [&](id<MTLComputeCommandEncoder> encoder,
                                           const std::size_t sourceIndex) {
                const auto barrier = [&]() {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                };
                const auto& fixture = fixtures[sourceIndex];
                const auto& source = fixture.descriptor;
                const std::uint32_t chunkCount = static_cast<std::uint32_t>(
                    std::max<std::uint64_t>(1u,
                        source.byteCount /
                                NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES +
                            (source.byteCount %
                                NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES !=
                             0u)));
                const NMPhysicalStateDigestChunkGPU chunkPass{
                    .environmentCount = kEnvironmentCount,
                    .environmentIdentifierBase = kEnvironmentIdentifierBase,
                    .sourceOrdinal = static_cast<std::uint32_t>(sourceIndex),
                    .chunkBytes = NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES,
                    .chunkCount = chunkCount,
                    .scratchStride = scratchStride,
                    .reserved0 = 0u,
                    .reserved1 = 0u,
                    .sourceCapacityBytes = sourceBuffers[sourceIndex].length,
                };
                const NMPhysicalStateDigestLeafFinalizeGPU leafPass{
                    .environmentCount = kEnvironmentCount,
                    .chunkCount = chunkCount,
                    .scratchStride = scratchStride,
                    .reserved0 = 0u,
                };
                [encoder setBuffer:contexts offset:0u atIndex:0u];
                [encoder setBytes:&leafPass length:sizeof(leafPass) atIndex:1u];
                dispatch(encoder, leafInitialize,
                    static_cast<std::size_t>(kEnvironmentCount) * chunkCount);
                barrier();

                [encoder setBuffer:metadata offset:0u atIndex:0u];
                [encoder setBytes:&source length:sizeof(source) atIndex:1u];
                [encoder setBytes:&chunkPass length:sizeof(chunkPass) atIndex:2u];
                dispatch(encoder, leafMetadata,
                    static_cast<std::size_t>(kEnvironmentCount) * chunkCount);
                barrier();

                [encoder setBuffer:metadata offset:0u atIndex:0u];
                [encoder setBuffer:contexts offset:0u atIndex:1u];
                [encoder setBytes:&leafPass length:sizeof(leafPass) atIndex:2u];
                dispatch(encoder, leafBegin,
                    static_cast<std::size_t>(kEnvironmentCount) * chunkCount);
                barrier();

                [encoder setBuffer:sourceBuffers[sourceIndex]
                             offset:0u atIndex:0u];
                [encoder setBuffer:contexts offset:0u atIndex:1u];
                [encoder setBytes:&source length:sizeof(source) atIndex:2u];
                [encoder setBytes:&chunkPass length:sizeof(chunkPass) atIndex:3u];
                dispatch(encoder, chunks,
                    static_cast<std::size_t>(kEnvironmentCount) * chunkCount);
                barrier();

                [encoder setBuffer:contexts offset:0u atIndex:0u];
                [encoder setBuffer:scratchA offset:0u atIndex:1u];
                [encoder setBytes:&leafPass length:sizeof(leafPass) atIndex:2u];
                dispatch(encoder, leafFinalize,
                    static_cast<std::size_t>(kEnvironmentCount) * chunkCount);
                barrier();

                id<MTLBuffer> reductionInput = scratchA;
                id<MTLBuffer> reductionOutput = scratchB;
                std::uint32_t inputCount = chunkCount;
                std::uint32_t level = 0u;
                while (inputCount > 1u) {
                    const std::uint32_t outputCount =
                        inputCount / 2u + inputCount % 2u;
                    const NMPhysicalStateDigestReduceGPU reducePass{
                        .environmentCount = kEnvironmentCount,
                        .sourceOrdinal =
                            static_cast<std::uint32_t>(sourceIndex),
                        .inputCount = inputCount,
                        .outputCount = outputCount,
                        .scratchStride = scratchStride,
                        .level = level,
                        .reserved0 = 0u,
                        .reserved1 = 0u,
                    };
                    [encoder setBuffer:reductionInput offset:0u atIndex:0u];
                    [encoder setBuffer:reductionOutput offset:0u atIndex:1u];
                    [encoder setBytes:&source length:sizeof(source) atIndex:2u];
                    [encoder setBytes:&reducePass
                               length:sizeof(reducePass) atIndex:3u];
                    dispatch(encoder, reduce,
                        static_cast<std::size_t>(kEnvironmentCount) *
                            outputCount);
                    barrier();
                    std::swap(reductionInput, reductionOutput);
                    inputCount = outputCount;
                    ++level;
                }
                const NMPhysicalStateDigestStoreGPU storePass{
                    .environmentCount = kEnvironmentCount,
                    .sourceOrdinal = static_cast<std::uint32_t>(sourceIndex),
                    .sourceCount = NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT,
                    .scratchStride = scratchStride,
                };
                [encoder setBuffer:reductionInput offset:0u atIndex:0u];
                [encoder setBuffer:sourceRoots offset:0u atIndex:1u];
                [encoder setBytes:&storePass length:sizeof(storePass) atIndex:2u];
                dispatch(encoder, store, kEnvironmentCount);
                barrier();
            };
            for (std::size_t sourceIndex = 0u;
                 sourceIndex < fixtures.size(); ++sourceIndex) {
                encodeSource(baselineEncoder, sourceIndex);
            }

            std::array<NMPhysicalStateDigestSourceGPU,
                NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT> descriptors{};
            for (std::size_t index = 0u; index < fixtures.size(); ++index) {
                descriptors[index] = fixtures[index].descriptor;
            }
            const NMPhysicalStateDigestFinalizeGPU finalizePass{
                .abiVersion = NM_MATTER_PHYSICAL_STATE_DIGEST_ABI_VERSION,
                .structSize = NM_MATTER_PHYSICAL_STATE_DIGEST_BYTES,
                .schemaVersion = NM_MATTER_PHYSICAL_STATE_DIGEST_SCHEMA_VERSION,
                .manifestVersion =
                    NM_MATTER_PHYSICAL_STATE_DIGEST_MANIFEST_VERSION,
                .environmentCount = kEnvironmentCount,
                .environmentIdentifierBase = kEnvironmentIdentifierBase,
                .sourceCount = NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT,
                .clockDomain = NM_MATTER_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS,
                .clockQuantumNanoseconds =
                    NM_MATTER_EXACT_CLOCK_QUANTUM_NANOSECONDS,
                .reserved0 = 0u,
                .reserved1 = 0u,
                .reserved2 = 0u,
                .acceptedTimestampNanoseconds = 1'234'567'890'123u,
                .physicsGeneration = 77u,
                .matterSourcePhysicsFingerprint = 0x1020304050607080u,
                .matterDeviceProgramFingerprint = 0x8877665544332211u,
            };
            [baselineEncoder setBuffer:sourceRoots offset:0u atIndex:0u];
            [baselineEncoder setBytes:descriptors.data()
                       length:sizeof(descriptors) atIndex:1u];
            [baselineEncoder setBuffer:output offset:0u atIndex:2u];
            [baselineEncoder setBytes:&finalizePass
                       length:sizeof(finalizePass) atIndex:3u];
            dispatch(baselineEncoder, finalize, kEnvironmentCount);
            [baselineEncoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted,
                "physical digest Metal command failed: " +
                    errorText(command.error));

            const auto* records =
                static_cast<const NMPhysicalStateDigestGPU*>(output.contents);
            std::array<OracleRoots, kEnvironmentCount> expected{};
            for (std::uint32_t environment = 0u;
                 environment < kEnvironmentCount; ++environment) {
                expected[environment] = oracleRoots(
                    fixtures, finalizePass, environment);
                const auto& record = records[environment];
                require(record.abiVersion == finalizePass.abiVersion &&
                        record.structSize == finalizePass.structSize &&
                        record.status == NM_PHYSICAL_STATE_DIGEST_VALID &&
                        record.environment ==
                            kEnvironmentIdentifierBase + environment &&
                        record.schemaVersion == finalizePass.schemaVersion &&
                        record.manifestVersion == finalizePass.manifestVersion &&
                        record.sourceCount == finalizePass.sourceCount,
                    "physical digest record header mismatch");
                require(record.acceptedTimestampNanoseconds ==
                            finalizePass.acceptedTimestampNanoseconds &&
                        record.physicsGeneration ==
                            finalizePass.physicsGeneration &&
                        record.matterSourcePhysicsFingerprint ==
                            finalizePass.matterSourcePhysicsFingerprint &&
                        record.matterDeviceProgramFingerprint ==
                            finalizePass.matterDeviceProgramFingerprint,
                    "physical digest provenance mismatch");
                requireDigest(record.humanSHA256, expected[environment].human,
                    "Human owner root");
                requireDigest(record.matterSHA256, expected[environment].matter,
                    "Matter owner root");
                requireDigest(record.physicalSHA256,
                    expected[environment].physical, "physical root");
            }

            id<MTLBuffer> provenanceOutput = [device newBufferWithLength:
                kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU)
                options:MTLResourceStorageModeShared];
            require(provenanceOutput != nil,
                "provenance result allocation failed");
            std::memset(provenanceOutput.contents, 0,
                kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU));
            auto changedProvenance = finalizePass;
            changedProvenance.acceptedTimestampNanoseconds += 9u;
            changedProvenance.physicsGeneration += 5u;
            changedProvenance.matterSourcePhysicsFingerprint ^= 0x55u;
            changedProvenance.matterDeviceProgramFingerprint ^= 0xaau;
            id<MTLCommandBuffer> provenanceCommand = [queue commandBuffer];
            id<MTLComputeCommandEncoder> provenanceEncoder =
                [provenanceCommand computeCommandEncoder];
            require(provenanceCommand != nil && provenanceEncoder != nil,
                "provenance command allocation failed");
            [provenanceEncoder setBuffer:sourceRoots offset:0u atIndex:0u];
            [provenanceEncoder setBytes:descriptors.data()
                                length:sizeof(descriptors) atIndex:1u];
            [provenanceEncoder setBuffer:provenanceOutput
                                  offset:0u atIndex:2u];
            [provenanceEncoder setBytes:&changedProvenance
                                  length:sizeof(changedProvenance) atIndex:3u];
            dispatch(provenanceEncoder, finalize, kEnvironmentCount);
            [provenanceEncoder endEncoding];
            [provenanceCommand commit];
            [provenanceCommand waitUntilCompleted];
            require(provenanceCommand.status == MTLCommandBufferStatusCompleted,
                "provenance-only command failed: " +
                    errorText(provenanceCommand.error));
            const auto* changed = static_cast<const NMPhysicalStateDigestGPU*>(
                provenanceOutput.contents);
            for (std::uint32_t environment = 0u;
                 environment < kEnvironmentCount; ++environment) {
                require(changed[environment].status ==
                            NM_PHYSICAL_STATE_DIGEST_VALID,
                    "changed-provenance record is invalid");
                require(std::memcmp(&changed[environment].humanSHA256,
                            &records[environment].humanSHA256,
                            sizeof(NMSHA256DigestGPU)) == 0 &&
                        std::memcmp(&changed[environment].matterSHA256,
                            &records[environment].matterSHA256,
                            sizeof(NMSHA256DigestGPU)) == 0 &&
                        std::memcmp(&changed[environment].physicalSHA256,
                            &records[environment].physicalSHA256,
                            sizeof(NMSHA256DigestGPU)) == 0,
                    "provenance changed a content digest");
                require(changed[environment].acceptedTimestampNanoseconds ==
                            changedProvenance.acceptedTimestampNanoseconds &&
                        changed[environment].physicsGeneration ==
                            changedProvenance.physicsGeneration &&
                        changed[environment].matterSourcePhysicsFingerprint ==
                            changedProvenance.matterSourcePhysicsFingerprint &&
                        changed[environment].matterDeviceProgramFingerprint ==
                            changedProvenance.matterDeviceProgramFingerprint,
                    "changed provenance was not preserved beside the digest");
            }

            const auto requireInvalidDescriptors = [&]
                (const std::array<
                     NMPhysicalStateDigestSourceGPU,
                     NM_MATTER_PHYSICAL_STATE_DIGEST_SOURCE_COUNT>& invalidSet,
                 const std::string& label) {
                id<MTLBuffer> invalidOutput = [device newBufferWithLength:
                    kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU)
                    options:MTLResourceStorageModeShared];
                require(invalidOutput != nil,
                    label + " result allocation failed");
                std::memset(invalidOutput.contents, 0xa5,
                    kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU));
                id<MTLCommandBuffer> invalidCommand = [queue commandBuffer];
                id<MTLComputeCommandEncoder> invalidEncoder =
                    [invalidCommand computeCommandEncoder];
                require(invalidCommand != nil && invalidEncoder != nil,
                    label + " command allocation failed");
                [invalidEncoder setBuffer:sourceRoots offset:0u atIndex:0u];
                [invalidEncoder setBytes:invalidSet.data()
                                   length:sizeof(invalidSet) atIndex:1u];
                [invalidEncoder setBuffer:invalidOutput offset:0u atIndex:2u];
                [invalidEncoder setBytes:&finalizePass
                                  length:sizeof(finalizePass) atIndex:3u];
                dispatch(invalidEncoder, finalize, kEnvironmentCount);
                [invalidEncoder endEncoding];
                [invalidCommand commit];
                [invalidCommand waitUntilCompleted];
                require(invalidCommand.status ==
                            MTLCommandBufferStatusCompleted,
                    label + " command failed: " +
                        errorText(invalidCommand.error));
                const auto* invalid =
                    static_cast<const NMPhysicalStateDigestGPU*>(
                        invalidOutput.contents);
                const NMSHA256DigestGPU zeroDigest{};
                for (std::uint32_t environment = 0u;
                     environment < kEnvironmentCount; ++environment) {
                    require(invalid[environment].status ==
                                NM_PHYSICAL_STATE_DIGEST_INVALID &&
                            std::memcmp(&invalid[environment].humanSHA256,
                                &zeroDigest, sizeof(zeroDigest)) == 0 &&
                            std::memcmp(&invalid[environment].matterSHA256,
                                &zeroDigest, sizeof(zeroDigest)) == 0 &&
                            std::memcmp(&invalid[environment].physicalSHA256,
                                &zeroDigest, sizeof(zeroDigest)) == 0,
                        label + " was not retained fail-closed");
                }
            };
            auto invalidDescriptors = descriptors;
            invalidDescriptors[17u].reserved1 = 1u;
            requireInvalidDescriptors(
                invalidDescriptors, "invalid reserved descriptor");
            auto reorderedDescriptors = descriptors;
            std::swap(
                reorderedDescriptors[0u].source,
                reorderedDescriptors[1u].source);
            requireInvalidDescriptors(
                reorderedDescriptors, "reordered manifest identity");

            const auto refreshSourceBuffer = [&](const std::size_t source) {
                require(!fixtures[source].bytes.empty(),
                    "mutation fixture unexpectedly has no bytes");
                require(sourceBuffers[source].length ==
                            fixtures[source].bytes.size(),
                    "mutation fixture changed source capacity");
                std::memcpy(sourceBuffers[source].contents,
                    fixtures[source].bytes.data(),
                    fixtures[source].bytes.size());
            };
            const auto allocateDigestOutput = [&](const std::string& label) {
                id<MTLBuffer> result = [device newBufferWithLength:
                    kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU)
                    options:MTLResourceStorageModeShared];
                require(result != nil, label + " result allocation failed");
                std::memset(result.contents, 0,
                    kEnvironmentCount * sizeof(NMPhysicalStateDigestGPU));
                return result;
            };
            const auto recomputeSourceAndFinalize = [&]
                (const std::size_t source,
                 id<MTLBuffer> result,
                 const std::string& label) {
                id<MTLCommandBuffer> mutationCommand = [queue commandBuffer];
                id<MTLComputeCommandEncoder> mutationEncoder =
                    [mutationCommand computeCommandEncoder];
                require(mutationCommand != nil && mutationEncoder != nil,
                    label + " command allocation failed");
                encodeSource(mutationEncoder, source);
                [mutationEncoder setBuffer:sourceRoots
                                     offset:0u atIndex:0u];
                [mutationEncoder setBytes:descriptors.data()
                                    length:sizeof(descriptors) atIndex:1u];
                [mutationEncoder setBuffer:result offset:0u atIndex:2u];
                [mutationEncoder setBytes:&finalizePass
                                    length:sizeof(finalizePass) atIndex:3u];
                dispatch(mutationEncoder, finalize, kEnvironmentCount);
                [mutationEncoder endEncoding];
                [mutationCommand commit];
                [mutationCommand waitUntilCompleted];
                require(mutationCommand.status ==
                            MTLCommandBufferStatusCompleted,
                    label + " command failed: " +
                        errorText(mutationCommand.error));
            };
            const auto requireMutationParity = [&]
                (id<MTLBuffer> result,
                 const std::array<OracleRoots, kEnvironmentCount>& oracle,
                 const std::string& label) {
                const auto* mutationRecords =
                    static_cast<const NMPhysicalStateDigestGPU*>(
                        result.contents);
                for (std::uint32_t environment = 0u;
                     environment < kEnvironmentCount; ++environment) {
                    require(mutationRecords[environment].status ==
                                NM_PHYSICAL_STATE_DIGEST_VALID,
                        label + " produced an invalid record");
                    const std::string suffix = " environment " +
                        std::to_string(environment);
                    requireDigest(mutationRecords[environment].humanSHA256,
                        oracle[environment].human,
                        label + " Human owner root" + suffix);
                    requireDigest(mutationRecords[environment].matterSHA256,
                        oracle[environment].matter,
                        label + " Matter owner root" + suffix);
                    requireDigest(mutationRecords[environment].physicalSHA256,
                        oracle[environment].physical,
                        label + " physical root" + suffix);
                }
            };

            // The Human mutation lands in the unpaired third leaf of a
            // three-chunk environment-local source. This exercises the odd
            // reduction branch and verifies owner/environment locality.
            constexpr std::size_t kHumanMutationSource = 2u;
            constexpr std::size_t kHumanMutationByte =
                2u * NM_MATTER_PHYSICAL_STATE_DIGEST_CHUNK_BYTES + 1u;
            require(fixtures[kHumanMutationSource].descriptor.target ==
                        NM_MATTER_PHYSICAL_STATE_TARGET_HUMAN &&
                    fixtures[kHumanMutationSource].descriptor.byteCount >
                        kHumanMutationByte,
                "Human odd-tree mutation source is malformed");
            const auto originalHumanByte =
                fixtures[kHumanMutationSource].bytes[kHumanMutationByte];
            fixtures[kHumanMutationSource].bytes[kHumanMutationByte] ^=
                0x5au;
            refreshSourceBuffer(kHumanMutationSource);
            id<MTLBuffer> humanMutationOutput =
                allocateDigestOutput("Human mutation");
            recomputeSourceAndFinalize(kHumanMutationSource,
                humanMutationOutput, "Human mutation");
            std::array<OracleRoots, kEnvironmentCount> humanMutationOracle{};
            for (std::uint32_t environment = 0u;
                 environment < kEnvironmentCount; ++environment) {
                humanMutationOracle[environment] = oracleRoots(
                    fixtures, finalizePass, environment);
            }
            requireMutationParity(humanMutationOutput,
                humanMutationOracle, "Human mutation");
            require(humanMutationOracle[0u].human != expected[0u].human &&
                    humanMutationOracle[0u].matter == expected[0u].matter &&
                    humanMutationOracle[0u].physical != expected[0u].physical,
                "Human mutation did not change only the Human owner root");
            require(humanMutationOracle[1u].human == expected[1u].human &&
                    humanMutationOracle[1u].matter == expected[1u].matter &&
                    humanMutationOracle[1u].physical == expected[1u].physical,
                "Human mutation escaped its environment");

            fixtures[kHumanMutationSource].bytes[kHumanMutationByte] =
                originalHumanByte;
            refreshSourceBuffer(kHumanMutationSource);
            id<MTLBuffer> humanRestoreOutput =
                allocateDigestOutput("Human restoration");
            recomputeSourceAndFinalize(kHumanMutationSource,
                humanRestoreOutput, "Human restoration");
            requireMutationParity(humanRestoreOutput,
                expected, "Human restoration");

            // Mutate only environment 1 of a Matter-local source and prove
            // that environment 0 and both Human roots remain unchanged.
            constexpr std::size_t kMatterMutationSource = 4u;
            const std::size_t matterMutationByte =
                static_cast<std::size_t>(
                    fixtures[kMatterMutationSource].descriptor.byteCount) + 2u;
            require(fixtures[kMatterMutationSource].descriptor.target ==
                        NM_MATTER_PHYSICAL_STATE_TARGET_MATTER &&
                    fixtures[kMatterMutationSource].bytes.size() >
                        matterMutationByte,
                "Matter mutation source is malformed");
            fixtures[kMatterMutationSource].bytes[matterMutationByte] ^=
                0xa5u;
            refreshSourceBuffer(kMatterMutationSource);
            id<MTLBuffer> matterMutationOutput =
                allocateDigestOutput("Matter mutation");
            recomputeSourceAndFinalize(kMatterMutationSource,
                matterMutationOutput, "Matter mutation");
            std::array<OracleRoots, kEnvironmentCount> matterMutationOracle{};
            for (std::uint32_t environment = 0u;
                 environment < kEnvironmentCount; ++environment) {
                matterMutationOracle[environment] = oracleRoots(
                    fixtures, finalizePass, environment);
            }
            requireMutationParity(matterMutationOutput,
                matterMutationOracle, "Matter mutation");
            require(matterMutationOracle[0u].human == expected[0u].human &&
                    matterMutationOracle[0u].matter == expected[0u].matter &&
                    matterMutationOracle[0u].physical == expected[0u].physical,
                "Matter mutation escaped its environment");
            require(matterMutationOracle[1u].human == expected[1u].human &&
                    matterMutationOracle[1u].matter != expected[1u].matter &&
                    matterMutationOracle[1u].physical != expected[1u].physical,
                "Matter mutation did not change only the Matter owner root");

            std::cout
                << "matter_physical_state_digest_schema=passed"
                << " environments=" << kEnvironmentCount
                << " sources=" << fixtures.size()
                << " zero_length_sources=2"
                << " shared_sources=3"
                << " environment_local_sources=28"
                << " multichunk_sources=3"
                << " odd_chunk_tree=3_leaves"
                << " human_mutation_locality=passed"
                << " matter_mutation_locality=passed"
                << " cpu_oracle=commoncrypto_sha256"
                << " provenance_excluded=true"
                << " invalid_descriptor=fail_closed"
                << " manifest_identity=exact_order_fail_closed"
                << " algorithm_schema_evidence=true"
                << " physical_state_qualified=false\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr
                << "matter_physical_state_digest_schema=failed reason=\""
                << exception.what()
                << "\" algorithm_schema_evidence=false"
                << " physical_state_qualified=false\n";
            return 1;
        }
    }
}
