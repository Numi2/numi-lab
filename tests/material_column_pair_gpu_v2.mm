#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

struct alignas(16) U4 { std::uint32_t x, y, z, w; };
struct alignas(16) F4 { float x, y, z, w; };
struct alignas(16) Dispatch { std::uint32_t objectCount, gridNodeCount, femNodeCount, tetrahedronCount, cohesiveFaceCount; };
struct alignas(16) Primitive { U4 nodesAndObject, identity; F4 boundsMinimum, boundsMaximum; };
struct alignas(16) TopologyNode { U4 identity; };
struct alignas(16) Object { std::uint32_t representation, materialIndex, flags, schedulerIndex; };
struct alignas(16) Range { std::uint32_t first, count, objectIndex, reserved; };
struct alignas(16) CohesiveFace { U4 nodesAndFirst, adjacency, lawIdentity; };
struct alignas(16) U4x { std::uint32_t x, y, z, w; };

static constexpr std::uint32_t kActive = 1u;
static constexpr std::uint32_t kDisableSelf = 1u << 7u;

int main(int argc, char** argv) {
    if (argc != 2) return 64;
    std::ifstream input(argv[1], std::ios::binary);
    const std::string source((std::istreambuf_iterator<char>(input)), {});
    if (source.empty()) return 65;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 66;
        NSError* error = nil;
        NSString* text = [[NSString alloc] initWithBytes:source.data()
                                                 length:source.size()
                                               encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [device newLibraryWithSource:text options:nil error:&error];
        if (!library) {
            std::cerr << "Metal source compile failed: " << [[error description] UTF8String] << "\n";
            return 67;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"testMaterialColumnPairEligibility"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::cerr << "Metal pipeline creation failed: " << [[error description] UTF8String] << "\n";
            return 68;
        }

        constexpr std::uint32_t pairCount = 8u;
        Dispatch dispatch{3u, 0u, pairCount * 6u, 0u, 0u};
        std::array<Object, 3u> objects{};
        objects[2].flags = kDisableSelf;
        std::array<Primitive, pairCount> left{}, right{};
        std::vector<TopologyNode> nodes(dispatch.femNodeCount * 2u);
        for (std::uint32_t pair = 0u; pair < pairCount; ++pair) {
            const std::uint32_t base = pair * 6u;
            const std::uint32_t leftOwner = pair == 4u ? 2u : 0u;
            const std::uint32_t rightOwner = pair == 2u ? 1u : leftOwner;
            left[pair].nodesAndObject = {base, base + 1u, base + 2u, leftOwner};
            right[pair].nodesAndObject = {base + 3u, base + 4u, base + 5u, rightOwner};
            left[pair].identity = {pair * 2u, 2u, 1u, 0u};
            right[pair].identity = {pair * 2u + 1u, 2u, 1u, 0u};
            for (std::uint32_t local = 0u; local < 6u; ++local) {
                std::uint32_t column = local % 3u + 1u;
                if (pair == 1u && local >= 3u) column += 3u;
                if (pair == 3u || pair == 4u || pair == 6u || pair == 7u) column = 0u;
                nodes[base + local].identity = {
                    base + local, leftOwner, 1u,
                    kActive | (column << 10u)};
            }
            for (std::uint32_t local = 3u; local < 6u; ++local)
                nodes[base + local].identity.y = rightOwner;
            if (pair == 3u)
                nodes[base + 3u].identity.x = nodes[base].identity.x;
            if (pair == 5u)
                right[pair].nodesAndObject.x = left[pair].nodesAndObject.x;
        }
        for (std::uint32_t i = 0u; i < dispatch.femNodeCount; ++i)
            nodes[dispatch.femNodeCount + i] = nodes[i];
        const std::uint32_t offsetPairBase = 7u * 6u;
        for (std::uint32_t local = 0u; local < 6u; ++local) {
            const std::uint32_t column = local < 3u ? local + 1u : local - 2u;
            nodes[dispatch.femNodeCount + offsetPairBase + local].identity = {
                offsetPairBase + local, 0u, 1u, kActive | (column << 10u)};
        }
        std::array<std::uint32_t, pairCount> environments{};
        environments[7] = 1u;
        std::array<Range, 1u> ranges{};
        std::array<U4x, 1u> refs{};
        std::array<CohesiveFace, 1u> cohesive{};
        std::array<std::uint32_t, pairCount> output{};
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        id<MTLBuffer> buffers[] = {
            [device newBufferWithBytes:&dispatch length:sizeof(dispatch) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:left.data() length:sizeof(left) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:right.data() length:sizeof(right) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:objects.data() length:sizeof(objects) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:nodes.data() length:nodes.size() * sizeof(TopologyNode) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:ranges.data() length:sizeof(ranges) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:refs.data() length:sizeof(refs) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:cohesive.data() length:sizeof(cohesive) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:environments.data() length:sizeof(environments) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:output.data() length:sizeof(output) options:MTLResourceStorageModeShared],
        };
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger i = 0u; i < 10u; ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(pairCount, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "Metal dispatch failed: " << [[command.error description] UTF8String] << "\n";
            return 69;
        }
        const auto* actual = static_cast<const std::uint32_t*>(buffers[9].contents);
        const std::array<std::uint32_t, pairCount> expected{0u, 1u, 1u, 0u, 0u, 0u, 1u, 0u};
        for (std::uint32_t i = 0u; i < pairCount; ++i) {
            std::cout << "pair[" << i << "]=" << actual[i] << " expected=" << expected[i] << "\n";
            if (actual[i] != expected[i]) return 70;
        }
        std::cout << "MATERIAL_COLUMN_PAIR_GPU_PASS device=" << [[device.name description] UTF8String] << "\n";
    }
    return 0;
}
