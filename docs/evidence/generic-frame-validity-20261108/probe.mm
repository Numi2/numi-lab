#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/visual_platform_types.h"
#include "metalrobo/world_compiler_types.h"
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
    uint32_t valid[8] = {7,7,7,0,7,7,7,7};
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
    std::cout << "baseline synthetic depth gates passed\n";
    // The renderer may leave FRAME|GEOMETRY set after sensor dropout while
    // clearing DEPTH. Its depth plane is cleared to far. Mask-level noise
    // must never turn these depth-invalid pixels into valid near samples.
    const uint32_t geometryOnly=MR_VISUAL_VALIDITY_FRAME|MR_VISUAL_VALIDITY_GEOMETRY;
    for(int i=0;i<8;++i){valid[i]=geometryOnly;depths[i]=5.0f;ids[i].y=42u;}
    memcpy([b[0] contents],depths,sizeof depths);
    memcpy([b[1] contents],ids,sizeof ids);
    memcpy([b[2] contents],valid,sizeof valid);
    u.corruption.w=1.0f;
    auto invalidRaw=run(true,false,false);
    auto invalidOracle=run(false,false,false);
    // Same sensor dropout through direct geometric winners.
    std::memset([b[14] contents],0,[b[14] length]);
    std::memset([b[15] contents],0,[b[15] length]);
    auto* sensor=static_cast<MRWorldSensorInstanceGPU*>([b[15] contents]);
    sensor->noiseAndLatency.z=1.0f;
    for(int i=0;i<8;++i){float z=.5f;uint32_t bits;memcpy(&bits,&z,sizeof bits);winners[i]=uint64_t(bits)<<32;}
    ru.sensorRangeAndResponse={.1f,5.f,0.f,0.f};
    ru.clearColorAndDepth.w=5.f;ru.shadowBatch.w=1u;
    auto invalidDirect=run(true,true,false);
    unsigned rawFailures=0,oracleFailures=0,directFailures=0;
    for(int i=0;i<8;++i){
      rawFailures+=invalidRaw[i]!=1.0f;
      oracleFailures+=invalidOracle[i]!=1.0f;
      directFailures+=invalidDirect[i]!=1.0f;
      std::cout<<"pixel="<<i<<" raw="<<invalidRaw[i]<<" oracle="<<invalidOracle[i]<<" direct="<<invalidDirect[i]<<'\n';
    }
    std::cout<<"invalid_depth_near_count raw="<<rawFailures<<" oracle="<<oracleFailures<<" direct="<<directFailures<<'\n';
    if(rawFailures || oracleFailures || directFailures) return 7;
    // A missing frame is not an edge-fill destination. Geometry-only
    // pixels cannot donate depth, but frame-present missing pixels may
    // receive explicitly authored edge flicker from a measured neighbor.
    u.corruption.w=0.f;u.range.w=1.f;
    for(int i=0;i<8;++i){depths[i]=.5f;valid[i]=MR_VISUAL_VALIDITY_FRAME|MR_VISUAL_VALIDITY_GEOMETRY;ids[i].y=42;}
    valid[0]=MR_VISUAL_VALIDITY_FRAME|MR_VISUAL_VALIDITY_DEPTH;
    valid[1]=0;
    memcpy([b[0] contents],depths,sizeof depths);memcpy([b[1] contents],ids,sizeof ids);memcpy([b[2] contents],valid,sizeof valid);
    auto mixed=run(true,false,false);
    const float validValue=(.5f-.1f)/4.9f;
    std::cout<<"mixed valid="<<mixed[0]<<" missing="<<mixed[1]<<" frame_present_fill="<<mixed[4]<<" no_valid_neighbor="<<mixed[3]<<'\n';
    bool mixedOK=std::abs(mixed[0]-validValue)<1e-6 && mixed[1]==1.f && std::abs(mixed[4]-validValue)<1e-6 && mixed[3]==1.f;
    for(int i=0;i<8;++i)valid[i]=0;
    memcpy([b[2] contents],valid,sizeof valid);
    auto missing=run(true,false,false);unsigned missingErrors=0;for(float v:missing)missingErrors+=v!=1.f;
    std::cout<<"fully_missing_near_count="<<missingErrors<<'\n';
    if(!mixedOK || missingErrors) return 8;
    std::cout << "PASS raw identity invariance, oracle sensitivity, validity, reset, dropout, direct winners\n";
  }
}
