#include "metalrobo/engine_types.h"
#include "metalrobo/hyper_policy_program_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "expected HyperPolicy metallib path");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      require(device != nil, "Metal device unavailable");
      NSError *error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:
              [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                      error:&error];
      require(library != nil, "cannot load HyperPolicy metallib");
      id<MTLFunction> function =
          [library newFunctionWithName:@"mr_hyper_policy_restore_failed_step"];
      require(function != nil, "missing restore kernel");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      require(pipeline != nil, "cannot create restore pipeline");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      require(queue != nil, "cannot create command queue");
      constexpr unsigned environments = 4, actionCount = 7;
      auto buffer = [&](size_t bytes) {
        return [device newBufferWithLength:bytes
                                   options:MTLResourceStorageModeShared];
      };
      id<MTLBuffer> phases =
          buffer((environments + 1) * sizeof(MRHyperPolicyPhaseStateGPU));
      id<MTLBuffer> savedPhases =
          buffer((environments + 1) * sizeof(MRHyperPolicyPhaseStateGPU));
      id<MTLBuffer> actions =
          buffer((environments * actionCount + 1) * sizeof(float));
      id<MTLBuffer> savedActions =
          buffer((environments * actionCount + 1) * sizeof(float));
      id<MTLBuffer> world =
          buffer(environments * sizeof(MRMetalWorldStatusGPU));
      id<MTLBuffer> contact =
          buffer(environments * sizeof(MRMetalWorldContactStatusGPU));
      require(phases && savedPhases && actions && savedActions && world &&
                  contact,
              "buffer allocation");
      for (unsigned contactMode : {0u, 1u}) {
        std::memset(phases.contents, 0x37, phases.length);
        std::memset(savedPhases.contents, 0x52, savedPhases.length);
        auto *values = static_cast<float *>(actions.contents);
        auto *savedValues = static_cast<float *>(savedActions.contents);
        for (unsigned i = 0; i <= environments * actionCount; ++i) {
          values[i] = static_cast<float>(100 + i);
          savedValues[i] = static_cast<float>(200 + i);
        }
        std::memset(world.contents, 0, world.length);
        std::memset(contact.contents, 0, contact.length);
        auto *worldStatuses =
            static_cast<MRMetalWorldStatusGPU *>(world.contents);
        auto *contactStatuses =
            static_cast<MRMetalWorldContactStatusGPU *>(contact.contents);
        worldStatuses[1].code = worldStatuses[3].code = 5;
        contactStatuses[2].code = contactStatuses[3].code = 5;
        const mr_uint4 counts{environments, actionCount, contactMode, 0};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        require(command && encoder, "command encoding");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:phases offset:0 atIndex:0];
        [encoder setBuffer:savedPhases offset:0 atIndex:1];
        [encoder setBuffer:actions offset:0 atIndex:2];
        [encoder setBuffer:savedActions offset:0 atIndex:3];
        [encoder setBuffer:world offset:0 atIndex:4];
        [encoder setBuffer:contact offset:0 atIndex:5];
        [encoder setBytes:&counts length:sizeof(counts) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(environments, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(environments, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "GPU restore command failed");
        auto *phaseBytes = static_cast<unsigned char *>(phases.contents);
        for (unsigned e = 0; e < environments; ++e) {
          bool failed = e == 1 || e == 3 || (contactMode && e == 2);
          for (size_t i = 0; i < sizeof(MRHyperPolicyPhaseStateGPU); ++i)
            require(phaseBytes[e * sizeof(MRHyperPolicyPhaseStateGPU) + i] ==
                        (failed ? 0x52 : 0x37),
                    "phase restore mismatch");
          for (unsigned i = 0; i < actionCount; ++i) {
            unsigned index = e * actionCount + i;
            require(values[index] ==
                        static_cast<float>((failed ? 200 : 100) + index),
                    "action restore mismatch");
          }
        }
        require(phaseBytes[environments * sizeof(MRHyperPolicyPhaseStateGPU)] ==
                    0x37,
                "phase canary changed");
        require(values[environments * actionCount] ==
                    static_cast<float>(100 + environments * actionCount),
                "action canary changed");
      }
      std::cout << "HyperPolicy restore status combinations passed\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << "\n";
      return 1;
    }
  }
}
