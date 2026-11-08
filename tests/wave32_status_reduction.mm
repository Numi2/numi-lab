#include "metalrobo/engine_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"mr_world_reduce_wave32_status"];
        id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline) { std::cerr << "missing status-reduction pipeline\n"; return 3; }
        constexpr unsigned count = 5, stride = 2;
        MRMetalWorldContactDispatchGPU dispatch{}; dispatch.environmentCount = count; dispatch.islandStride = stride;
        std::array<MRMetalWorldContactStatusGPU, count> statuses{};
        for (auto& status : statuses) { status.islandCount = 2; status.firstFailingConstraint = MR_INVALID_INDEX; }
        statuses[3].islandCount = 1;
        statuses[4].code = MR_STEP_CONTACT_CAPACITY_OVERFLOW; statuses[4].solverIterations = 77;
        std::array<MRWave32IslandStatusGPU, count * stride> islands{};
        islands[0].iterations = 5; islands[1].iterations = 9;
        islands[2].code = MR_STEP_DID_NOT_CONVERGE; islands[2].iterations = 2; islands[2].residuals = {0.3f, 0.4f, 0.5f, 7.0f};
        islands[4].iterations = 12; islands[5].code = MR_STEP_DID_NOT_CONVERGE; islands[5].iterations = 2; islands[5].residuals = {0.1f, 0.2f, 0.3f, 3.0f};
        islands[6].code = MR_STEP_UNSUPPORTED; islands[6].iterations = MR_INVALID_INDEX; islands[6].environment = 9; islands[6].islandIndex = 4;
        std::array<MRWorkQueueHeaderGPU, MR_WORLD_WORK_CLASS_COUNT> headers{};
        auto buffer = [&](const void* data, NSUInteger size) { return [device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared]; };
        id<MTLBuffer> buffers[] = {buffer(&dispatch,sizeof dispatch),buffer(islands.data(),sizeof islands),buffer(statuses.data(),sizeof statuses),buffer(headers.data(),sizeof headers)};
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (unsigned i=0;i<4;++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(count,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) { std::cerr << "GPU execution failed\n"; return 4; }
        std::memcpy(statuses.data(),buffers[2].contents,sizeof statuses);
        auto check = [&](bool condition,const char* message){ if (!condition) std::cerr << message << '\n'; return condition; };
        bool ok = true;
        ok &= check(statuses[0].code == MR_STEP_SUCCESS && statuses[0].solverIterations == 9,"successful islands must retain maximum iteration count");
        ok &= check(statuses[1].code == MR_STEP_DID_NOT_CONVERGE && statuses[1].solverIterations == 2 && statuses[1].firstFailingConstraint == 7 && statuses[1].residuals.y == 0.4f,"first failing island must retain its count and diagnostics");
        ok &= check(statuses[2].code == MR_STEP_DID_NOT_CONVERGE && statuses[2].solverIterations == 12 && statuses[2].firstFailingConstraint == 3,"failure after success must retain maximum processed count");
        ok &= check(statuses[3].code == MR_STEP_UNSUPPORTED && statuses[3].solverIterations == 0 && statuses[3].firstFailingConstraint == MR_INVALID_INDEX && statuses[3].firstFailingStableKeyLow == 4 && statuses[3].firstFailingStableKeyHigh == 9,"invalid-index sentinel must not become an iteration count");
        ok &= check(statuses[4].code == MR_STEP_CONTACT_CAPACITY_OVERFLOW && statuses[4].solverIterations == 77,"previously failed environment must remain unchanged");
        if (!ok) return 1;
        std::cout << "wave32 status reduction: 5 cases passed\n"; return 0;
    }
}
