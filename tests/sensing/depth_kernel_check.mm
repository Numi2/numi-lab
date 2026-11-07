#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/task_program_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

// Synthetic image regression. It does not qualify the full renderer, policy,
// sensor timing, or transactional behavior under failed physical steps.
int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 2) { std::cerr << "usage: depth_kernel_check metallib\n"; return 1; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
    id<MTLFunction> fn = [lib newFunctionWithName:@"mr_hybrid_masked_depth_history"];
    id<MTLComputePipelineState> pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeline) { std::cerr << [[err description] UTF8String] << "\n"; return 2; }
    auto buffer = [&](NSUInteger bytes) { return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; };
    std::array<id<MTLBuffer>, 16> b{};
    b[0] = buffer(8*sizeof(float));
    b[1] = buffer(8*sizeof(mr_uint4));
    b[2] = buffer(8*sizeof(uint32_t));
    b[3] = buffer(sizeof(uint32_t));
    b[4] = buffer(24*sizeof(float));
    b[5] = buffer(sizeof(uint32_t));
    b[6] = buffer(16*sizeof(float));
    b[7] = buffer(16*sizeof(float));
    b[8] = buffer(sizeof(MRTaskStateGPU));
    b[10] = buffer(8*sizeof(uint64_t));
    for (int i=11; i<=15; ++i) b[i] = buffer(4096);
    float depths[8] = {0.5f,1,2,3,4,5,0.2f,1.5f};
    uint32_t valid[8] = {1,1,1,0,1,1,1,1};
    mr_uint4 ids[8]{};
    for (int i=0;i<8;++i) ids[i].y = (i%2) ? 42u : 99u;
    memcpy([b[0] contents], depths, sizeof depths);
    memcpy([b[1] contents], ids, sizeof ids);
    memcpy([b[2] contents], valid, sizeof valid);
    *static_cast<uint32_t*>([b[3] contents]) = 42u;
    *static_cast<uint32_t*>([b[5] contents]) = 1u;
    MRHybridMaskedDepthUniformsGPU u{};
    u.counts = {1,2,4,1}; u.layout = {16,1,0,1};
    u.history = {3,2,0,1}; u.frameOffsets = {0,2,0,0};
    u.range = {0.1f,5.0f,1.0f/4.9f,0.0f};
    MRHybridRenderUniformsGPU ru{};
    auto run = [&](bool raw, bool direct, bool dropout) {
      u.curriculum.w = raw ? 1.0f : 0.0f;
      u.curriculum.y = direct ? 1.0f : 0.0f;
      u.corruption.x = dropout ? 1.0f : 0.0f;
      id<MTLCommandQueue> queue = [dev newCommandQueue];
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipeline];
      for (NSUInteger i=0;i<9;++i) [enc setBuffer:b[i] offset:0 atIndex:i];
      [enc setBytes:&u length:sizeof u atIndex:9];
      for (NSUInteger i=10;i<=15;++i) [enc setBuffer:b[i] offset:0 atIndex:i];
      [enc setBytes:&ru length:sizeof ru atIndex:16];
      [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
      [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted) return std::array<float,16>{};
      std::array<float,16> out{}; memcpy(out.data(), [b[7] contents], sizeof(float)*16); return out;
    };
    auto first = run(true,false,false);
    for (int i=0;i<8;++i) {
      float expected = valid[i] ? (depths[i]-0.1f)/4.9f : 1.0f;
      if (std::abs(first[i]-expected)>1e-6f || first[i]!=first[i+8]) return 3;
      auto history = static_cast<float*>([b[4] contents]);
      for (int slot=0;slot<3;++slot) if (history[slot*8+i]!=first[i]) return 3;
    }
    auto oracle = run(false,false,false);
    for (int i=0;i<8;++i) ids[i].y = ids[i].y==42u ? 99u : 42u;
    memcpy([b[1] contents], ids, sizeof ids);
    auto second = run(true,false,false);
    auto oracleChanged = run(false,false,false);
    if (first != second || oracle == oracleChanged) return 4;
    auto dropped = run(true,false,true);
    for (float v : dropped) if (v != 1.0f) return 5;

    // Exercise the geometric-winner route without consulting triangle IDs.
    auto winners = static_cast<uint64_t*>([b[10] contents]);
    for (int i=0;i<8;++i) {
      uint32_t bits; memcpy(&bits,&depths[i],sizeof bits);
      winners[i] = valid[i] ? (uint64_t(bits)<<32) : UINT64_MAX;
    }
    ru.live.w = 1;
    auto geometric = run(true,true,false);
    if (geometric != first) return 6;
    std::cout << "PASS raw identity invariance, oracle sensitivity, validity, reset, dropout, direct winners\n";
  }
}
