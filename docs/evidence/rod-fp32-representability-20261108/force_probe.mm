#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
struct Pair { float high, low; };
struct Input { Pair left, center, right, restLeft, restRight, stiffnessLeft, stiffnessRight, velocityOld, velocityNew, timestepOverMass; };
struct Output { float naive, high, low; };
static_assert(sizeof(Input) == 80);
static_assert(sizeof(Output) == 12);
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 4) { std::cerr << "usage: probe library input output\n"; return 2; }
        std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
        auto size = file.tellg();
        if (!file || size <= 0 || size % sizeof(Input)) return 3;
        std::vector<Input> input(size / sizeof(Input)); file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(input.data()), size)) return 4;
        uint32_t count = static_cast<uint32_t>(input.size());
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 5;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!library) { std::cerr << error.localizedDescription.UTF8String; return 6; }
        id<MTLFunction> function = [library newFunctionWithName:@"rod_precision_probe"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) { std::cerr << error.localizedDescription.UTF8String; return 7; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> in = [device newBufferWithBytes:input.data() length:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:count * sizeof(Output) options:MTLResourceStorageModeShared];
        if (!queue || !in || !out) return 8;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];
        [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::cerr << command.error.localizedDescription.UTF8String; return 9;
        }
        std::ofstream result(argv[3], std::ios::binary);
        result.write(static_cast<const char*>(out.contents), count*sizeof(Output));
        if (!result) return 10;
        std::cout << "device=" << device.name.UTF8String << " rows=" << count << "\n";
    }
    return 0;
}
