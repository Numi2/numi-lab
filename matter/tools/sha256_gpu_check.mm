#include "numi/matter/sha256_gpu.h"

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

std::array<std::uint8_t, 32u> decodeHex(const std::string_view text) {
    require(text.size() == 64u, "SHA-256 vector must contain 64 hex digits");
    std::array<std::uint8_t, 32u> result{};
    const auto nibble = [](const char value) -> std::uint8_t {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(value - 'a' + 10);
        }
        throw std::runtime_error("invalid SHA-256 vector hex digit");
    };
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(
            (nibble(text[index * 2u]) << 4u) |
            nibble(text[index * 2u + 1u]));
    }
    return result;
}

struct Vector {
    std::string label;
    std::vector<std::uint8_t> message;
    std::array<std::uint8_t, 32u> expected{};
    std::vector<std::size_t> chunks;
};

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    require(function != nil,
        "missing Metal SHA-256 function " + std::string(name.UTF8String));
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil,
        "failed to create Metal SHA-256 pipeline " +
            std::string(name.UTF8String) + ": " +
            errorText(error));
    return result;
}

void dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipelineState,
    const std::size_t count
) {
    [encoder setComputePipelineState:pipelineState];
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            std::min<std::size_t>(count, pipelineState.maxTotalThreadsPerThreadgroup),
            1u, 1u)];
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            std::vector<Vector> vectors;
            vectors.push_back({
                "empty", {},
                decodeHex("e3b0c44298fc1c149afbf4c8996fb924"
                          "27ae41e4649b934ca495991b7852b855"),
                {0u, 0u},
            });
            vectors.push_back({
                "abc", {'a', 'b', 'c'},
                decodeHex("ba7816bf8f01cfea414140de5dae2223"
                          "b00361a396177a9cb410ff61f20015ad"),
                {1u, 0u, 2u},
            });
            const std::string quick =
                "The quick brown fox jumps over the lazy dog";
            vectors.push_back({
                "quick-brown-fox",
                std::vector<std::uint8_t>(quick.begin(), quick.end()),
                decodeHex("d7a8fbb307d7809469ca9abcb0082e4f"
                          "8d5651e46d3cdb762d02d0bf37c9e592"),
                {7u, 13u, 1u, 22u},
            });
            vectors.push_back({
                "64-byte-block", std::vector<std::uint8_t>(64u, 'a'),
                decodeHex("ffe054fe7ae0cb6dc65c3af9b61d5209"
                          "f439851db43d0ba5997337df154668eb"),
                {31u, 1u, 32u},
            });
            vectors.push_back({
                "padding-two-blocks", std::vector<std::uint8_t>(56u, 'a'),
                decodeHex("b35439a4ac6f0948b6d6f9e3c6af0f5f"
                          "590ce20f1bde7090ef7970686ec6738a"),
                {55u, 1u},
            });
            vectors.push_back({
                "million-a", std::vector<std::uint8_t>(1'000'000u, 'a'),
                decodeHex("cdc76e5c9914fb9281a1c7e284d73e67"
                          "f1809a48a497200e046d39ccc7112cd0"),
                {63u, 1u, 65u, 4096u, 17u, 995'758u},
            });

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:@NUMI_SHA256_CHECK_METALLIB]
                error:&error];
            require(library != nil,
                "Metal SHA-256 test library unavailable: " +
                errorText(error));
            id<MTLComputePipelineState> initialize = pipeline(
                device, library, @"nm_sha256_context_initialize");
            id<MTLComputePipelineState> update = pipeline(
                device, library, @"nm_sha256_context_update");
            id<MTLComputePipelineState> finalize = pipeline(
                device, library, @"nm_sha256_context_finalize");

            std::vector<std::uint8_t> source;
            std::vector<std::size_t> bases;
            for (const auto& vector : vectors) {
                bases.push_back(source.size());
                source.insert(source.end(), vector.message.begin(), vector.message.end());
            }
            if (source.empty()) source.push_back(0u);
            const auto sharedBuffer = [&](const void* bytes, const std::size_t size) {
                id<MTLBuffer> result = [device newBufferWithBytes:bytes
                    length:size options:MTLResourceStorageModeShared];
                require(result != nil, "Metal SHA-256 buffer allocation failed");
                return result;
            };
            id<MTLBuffer> sourceBuffer = sharedBuffer(source.data(), source.size());
            id<MTLBuffer> contexts = [device newBufferWithLength:
                vectors.size() * sizeof(NMSHA256ContextGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> digests = [device newBufferWithLength:
                vectors.size() * sizeof(NMSHA256DigestGPU)
                options:MTLResourceStorageModeShared];
            require(contexts != nil && digests != nil,
                "Metal SHA-256 result allocation failed");

            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(queue != nil && command != nil && encoder != nil,
                "Metal SHA-256 command allocation failed");

            const NMSHA256DispatchGPU dispatchPass{
                static_cast<nm_u32>(vectors.size()), 0u, 0u, 0u};
            [encoder setBuffer:contexts offset:0 atIndex:0];
            [encoder setBytes:&dispatchPass length:sizeof(dispatchPass) atIndex:1];
            dispatch(encoder, initialize, vectors.size());

            std::vector<std::size_t> positions(vectors.size(), 0u);
            std::size_t round = 0u;
            for (;;) {
                bool pending = false;
                std::vector<NMSHA256SpanGPU> spans(vectors.size());
                for (std::size_t index = 0u; index < vectors.size(); ++index) {
                    const auto& vector = vectors[index];
                    std::size_t count = 0u;
                    if (round < vector.chunks.size()) {
                        count = vector.chunks[round];
                        pending = true;
                    }
                    require(positions[index] + count <= vector.message.size(),
                        "invalid SHA-256 test chunk partition");
                    spans[index] = {
                        static_cast<nm_u64>(bases[index] + positions[index]),
                        static_cast<nm_u64>(count),
                    };
                    positions[index] += count;
                }
                if (!pending) break;
                id<MTLBuffer> spanBuffer = sharedBuffer(
                    spans.data(), spans.size() * sizeof(spans[0]));
                const NMSHA256UpdateGPU updatePass{
                    static_cast<nm_u32>(vectors.size()), 0u,
                    static_cast<nm_u64>(source.size()),
                };
                [encoder setBuffer:sourceBuffer offset:0 atIndex:0];
                [encoder setBuffer:contexts offset:0 atIndex:1];
                [encoder setBuffer:spanBuffer offset:0 atIndex:2];
                [encoder setBytes:&updatePass length:sizeof(updatePass) atIndex:3];
                dispatch(encoder, update, vectors.size());
                ++round;
            }
            for (std::size_t index = 0u; index < vectors.size(); ++index) {
                require(positions[index] == vectors[index].message.size(),
                    "SHA-256 test chunks did not cover the message");
            }

            [encoder setBuffer:contexts offset:0 atIndex:0];
            [encoder setBuffer:digests offset:0 atIndex:1];
            [encoder setBytes:&dispatchPass length:sizeof(dispatchPass) atIndex:2];
            dispatch(encoder, finalize, vectors.size());
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted,
                "Metal SHA-256 command failed: " +
                errorText(command.error));

            const auto* actual = static_cast<const NMSHA256DigestGPU*>(digests.contents);
            const auto* states = static_cast<const NMSHA256ContextGPU*>(contexts.contents);
            for (std::size_t index = 0u; index < vectors.size(); ++index) {
                require(states[index].status == NM_SHA256_CONTEXT_FINALIZED,
                    vectors[index].label + " context did not finalize");
                require(states[index].totalBytes == vectors[index].message.size(),
                    vectors[index].label + " byte count changed");
                require(std::memcmp(actual[index].bytes,
                    vectors[index].expected.data(),
                    vectors[index].expected.size()) == 0,
                    vectors[index].label + " digest mismatch");
            }

            // A descriptor that reaches beyond the declared source capacity
            // must poison the context and finalize to an all-zero digest.
            id<MTLBuffer> invalidContext = [device newBufferWithLength:
                sizeof(NMSHA256ContextGPU) options:MTLResourceStorageModeShared];
            id<MTLBuffer> invalidDigest = [device newBufferWithLength:
                sizeof(NMSHA256DigestGPU) options:MTLResourceStorageModeShared];
            const NMSHA256SpanGPU invalidSpan{
                static_cast<nm_u64>(source.size()), 1u};
            id<MTLBuffer> invalidSpanBuffer = sharedBuffer(
                &invalidSpan, sizeof(invalidSpan));
            const NMSHA256DispatchGPU oneContext{1u, 0u, 0u, 0u};
            const NMSHA256UpdateGPU boundedSource{
                1u, 0u, static_cast<nm_u64>(source.size())};
            id<MTLCommandBuffer> invalidCommand = [queue commandBuffer];
            id<MTLComputeCommandEncoder> invalidEncoder =
                [invalidCommand computeCommandEncoder];
            require(invalidContext != nil && invalidDigest != nil &&
                    invalidCommand != nil && invalidEncoder != nil,
                "Metal SHA-256 invalid-span command allocation failed");
            [invalidEncoder setBuffer:invalidContext offset:0 atIndex:0];
            [invalidEncoder setBytes:&oneContext length:sizeof(oneContext)
                atIndex:1];
            dispatch(invalidEncoder, initialize, 1u);
            [invalidEncoder setBuffer:sourceBuffer offset:0 atIndex:0];
            [invalidEncoder setBuffer:invalidContext offset:0 atIndex:1];
            [invalidEncoder setBuffer:invalidSpanBuffer offset:0 atIndex:2];
            [invalidEncoder setBytes:&boundedSource length:sizeof(boundedSource)
                atIndex:3];
            dispatch(invalidEncoder, update, 1u);
            [invalidEncoder setBuffer:invalidContext offset:0 atIndex:0];
            [invalidEncoder setBuffer:invalidDigest offset:0 atIndex:1];
            [invalidEncoder setBytes:&oneContext length:sizeof(oneContext)
                atIndex:2];
            dispatch(invalidEncoder, finalize, 1u);
            [invalidEncoder endEncoding];
            [invalidCommand commit];
            [invalidCommand waitUntilCompleted];
            require(invalidCommand.status == MTLCommandBufferStatusCompleted,
                "Metal SHA-256 invalid-span command failed: " +
                errorText(invalidCommand.error));
            const auto* invalidState =
                static_cast<const NMSHA256ContextGPU*>(invalidContext.contents);
            const auto* invalidOutput =
                static_cast<const NMSHA256DigestGPU*>(invalidDigest.contents);
            require(invalidState->status == NM_SHA256_CONTEXT_INVALID,
                "out-of-bounds SHA-256 span did not poison its context");
            for (const auto byte : invalidOutput->bytes) {
                require(byte == 0u,
                    "invalid SHA-256 context emitted a nonzero digest");
            }
            std::cout << "matter_gpu_sha256=passed vectors=" << vectors.size()
                      << " fragmented_updates=" << round
                      << " largest_message_bytes=" << vectors.back().message.size()
                      << " invalid_span=fail_closed"
                      << " physical_state_qualified=false\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "matter_gpu_sha256=failed " << exception.what() << '\n';
            return 1;
        }
    }
}
