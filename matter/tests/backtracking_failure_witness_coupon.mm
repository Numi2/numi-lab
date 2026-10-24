#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Input {
    float determinantValue;
    float currentJ;
    std::uint32_t globalTet;
    std::uint32_t constitutiveValid;
    float lowerJ;
    float upperJ;
};
struct alignas(16) Float4 { float x, y, z, w; };
static_assert(sizeof(Input) == 24);
static_assert(sizeof(Float4) == 16);

static void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static std::uint32_t floatBits(const float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
int main(int argc, const char* argv[]) {
    @autoreleasepool {
        require(argc == 2, "usage: coupon METALLIB");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device available");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!library) {
            std::fprintf(stderr, "library: %s\n",
                error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"nm_fem_backtracking_witness_coupon"];
        require(function != nil, "witness coupon kernel exists");
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::fprintf(stderr, "pipeline: %s\n",
                error.localizedDescription.UTF8String);
            return 1;
        }
        std::array<Input, 288> inputs{};
        for (std::uint32_t group = 0; group < 9; ++group) {
            for (std::uint32_t lane = 0; lane < 32; ++lane) {
                auto& in = inputs[group * 32 + lane];
                in = {1.0f + 0.001f * static_cast<float>(lane), 1.0f,
                      1000u + group * 32u + lane, 1u, 0.2f, 2.5f};
            }
        }
        inputs[0] = {2.6f, 2.4f, 0xfffffffeu, 1u, 0.2f, 2.5f};
        inputs[1] = {NAN, 1.0f, 16777217u, 1u, 0.2f, 2.5f};
        inputs[2] = {0.1f, 1.0f, 16777216u, 1u, 0.2f, 2.5f};
        inputs[32] = {NAN, 1.25f, 16777217u, 1u, 0.2f, 2.5f};
        inputs[96] = {2.6f, 2.4f, 0xfffffffeu, 1u, 0.2f, 2.5f};
        inputs[128] = {1.0f, 1.0f, 1200u, 1u, NAN, 2.5f};
        inputs[160] = {1.0f, 1.0f, 1300u, 1u, 0.2f, NAN};
        inputs[192] = {1.0f, 1.0f, 1400u, 1u, -INFINITY, INFINITY};
        inputs[224] = {1.0f, 1.0f, 1500u, 1u, INFINITY, INFINITY};
        inputs[256] = {1.0f, 1.0f, 1600u, 1u, -INFINITY, -INFINITY};
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputs.data()
            length:sizeof(inputs) options:MTLResourceStorageModeShared];
        std::array<Float4, 10> initial{};
        id<MTLBuffer> outputBuffer = [device newBufferWithBytes:initial.data()
            length:sizeof(initial) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(inputBuffer && outputBuffer && queue, "Metal buffers and queue");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        require(command && encoder, "Metal command encoder");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(288, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(288, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Metal coupon dispatch completes");
        const auto* out = static_cast<const Float4*>(outputBuffer.contents);
        require(floatBits(out[0].x) == 16777216u && out[0].w == 2.0f &&
                out[0].y == 0.1f && out[0].z == 1.0f,
                "lowest uint tet owns upper/lower mixed witness");
        require(floatBits(out[1].x) == 16777217u && out[1].w == 1.0f &&
                std::isnan(out[1].y) && out[1].z == 1.25f,
                "nonfinite determinant and adjacent large ID survive");
        require(out[2].x == 1.0f && out[2].w == 0.0f,
                "valid group retains minimum J semantics");
        require(floatBits(out[3].x) == 0xfffffffeu && out[3].w == 4.0f &&
                out[3].y == 2.6f && out[3].z == 2.4f,
                "near-uint-max global tet survives as raw bits");
        require(floatBits(out[4].x) == 1200u && out[4].w == 2.0f,
                "NaN lower bound preserves legacy rejection");
        require(floatBits(out[5].x) == 1300u && out[5].w == 4.0f,
                "NaN upper bound preserves legacy rejection");
        require(out[6].x == 1.0f && out[6].w == 0.0f,
                "negative/positive infinite bounds accept finite determinant");
        require(floatBits(out[7].x) == 1500u && out[7].w == 2.0f,
                "positive infinite lower bound rejects finite determinant");
        require(floatBits(out[8].x) == 1600u && out[8].w == 4.0f,
                "negative infinite upper bound rejects finite determinant");
        require(out[9].x == 0.0f && out[9].y == 0.0f &&
                std::isinf(out[9].z) && out[9].z > 0.0f && out[9].w == 0.0f,
                "zero-trial line search starts with initialized empty witness");
        std::puts("PASS: Metal zero-trial initialization, legacy predicate parity, deterministic uint witness and minimum-J path");
    }
    return 0;
}
