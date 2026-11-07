#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "research_cadence.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
using research_cadence::Record;
static void check(bool ok,const char* s) {if(!ok) throw std::runtime_error(s);}
int main(int argc,char** argv) {
 @autoreleasepool { try {
  check(argc==2,"expected metallib path");
  id<MTLDevice> device=MTLCreateSystemDefaultDevice(); check(device!=nil,"device");
  NSError* error=nil;
  id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
  check(lib!=nil,"library");
  id<MTLComputePipelineState> pipe=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_commit_visual_clock"] error:&error];
  check(pipe!=nil,"pipeline");
  constexpr uint32_t n=4;
  auto allocate = ^id<MTLBuffer>(NSUInteger bytes) {return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];};
  id<MTLBuffer> old=allocate(n*sizeof(Record)), candidate=allocate(n*sizeof(Record));
  id<MTLBuffer> world=allocate(n*sizeof(MRMetalWorldStatusGPU)), contact=allocate(n*sizeof(MRMetalWorldContactStatusGPU));
  check(old && candidate && world && contact,"buffers");
  auto* a=static_cast<Record*>(old.contents);auto* c=static_cast<Record*>(candidate.contents);
  auto* w=static_cast<MRMetalWorldStatusGPU*>(world.contents);auto* k=static_cast<MRMetalWorldContactStatusGPU*>(contact.contents);
  std::memset(w,0,world.length); std::memset(k,0,contact.length);
  Record expected[n];
  for(uint32_t i=0;i<n;++i) {a[i]={i+5,10+i,20+i,30+i}; c[i]={i+6,40+i,50+i,60+i}; expected[i]=i==0?c[i]:a[i];}
  w[1].code=5; k[2].code=5; w[3].code=5; k[3].code=5;
  id<MTLCommandQueue> q=[device newCommandQueue]; id<MTLCommandBuffer> cmd=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
  check(enc!=nil,"command");
  [enc setComputePipelineState:pipe];[enc setBuffer:old offset:0 atIndex:0];[enc setBuffer:candidate offset:0 atIndex:1];[enc setBuffer:world offset:0 atIndex:2];[enc setBuffer:contact offset:0 atIndex:3];[enc setBytes:&n length:sizeof(n) atIndex:4];
  [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
  check(cmd.status==MTLCommandBufferStatusCompleted,"GPU command failed");
  for(uint32_t i=0;i<n;++i) {
   bool equal=std::memcmp(&a[i],&expected[i],sizeof(Record))==0;
   std::cout<<"env="<<i<<" world="<<w[i].code<<" contact="<<k[i].code<<" expected="<<(i==0?"candidate":"prior")<<" equal="<<equal<<" words="<<a[i].acceptedSteps<<","<<a[i].lastSensor<<","<<a[i].captureStep<<","<<a[i].captureTimeBits<<"\n";
   check(equal,"incorrect commit");
  }
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;} }
}
