#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/world_compiler_types.h"
#include "metalrobo/hybrid_renderer_types.h"
#include "ResearchVisualCadence.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
static void check(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
static uint32_t reference(uint32_t x,uint32_t y,uint32_t frame,uint32_t high,uint32_t sensor,uint32_t camera,uint32_t pixel) {
 return x ^ y ^ frame ^ high ^ (sensor * uint32_t(0x85ebca6b)) ^ (camera * uint32_t(0xc2b2ae35)) ^ (pixel * uint32_t(0x9e3779b9));
}
int main(int argc,char** argv) {
 @autoreleasepool {try {
  check(argc==2,"metallib argument");
  id<MTLDevice> d=MTLCreateSystemDefaultDevice();check(d!=nil,"device");NSError* err=nil;
  id<MTLLibrary> l=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];check(l!=nil,"library");
  id<MTLComputePipelineState> pipe=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"mr_research_seed_probe"] error:&err];check(pipe!=nil,"pipeline");
  auto alloc=^id<MTLBuffer>(NSUInteger n){return [d newBufferWithLength:n options:MTLResourceStorageModeShared];};
  auto instances=alloc(2*sizeof(MRWorldInstanceHeaderGPU));auto uniforms=alloc(2*sizeof(MRHybridRenderUniformsGPU));
  auto pixels=alloc(2*sizeof(uint32_t));auto result=alloc(2*sizeof(uint32_t));
  check(instances && uniforms && pixels && result,"buffers");
  std::memset(instances.contents,0,instances.length);std::memset(uniforms.contents,0,uniforms.length);
  auto* h=static_cast<MRWorldInstanceHeaderGPU*>(instances.contents);auto* u=static_cast<MRHybridRenderUniformsGPU*>(uniforms.contents);auto* px=static_cast<uint32_t*>(pixels.contents);auto* got=static_cast<uint32_t*>(result.contents);
  research_cadence::Record clocks[2]={{0,research_cadence::none,research_cadence::none,0},{3,1,2,0}};research_cadence::Candidate proposed[2]{};
  for(int i=0;i<2;++i)check(research_cadence::propose(clocks[i],.02f,25.f,0,false,proposed[i]),"proposal");
  check(proposed[0].sample && !proposed[1].sample,"sample/hold prereq");
  h[0].identity={101,401,0,0};h[1].identity={102,402,0,0};px[0]=73;px[1]=217;
  for(int i=0;i<2;++i){u[i].timing={proposed[i].accepted.captureStep,0,proposed[i].accepted.lastSensor,0};u[i].render.x=0;}
  id<MTLCommandQueue> q=[d newCommandQueue];check(q!=nil,"queue");
  auto dispatch=^(uint32_t base,uint32_t count) {
   id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];check(e!=nil,"encoder");
   [e setComputePipelineState:pipe];[e setBuffer:instances offset:0 atIndex:0];[e setBuffer:uniforms offset:0 atIndex:1];[e setBuffer:pixels offset:0 atIndex:2];[e setBuffer:result offset:0 atIndex:3];[e setBytes:&base length:sizeof(base) atIndex:4];
   [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(count,1,1)];[e endEncoding];[cmd commit];[cmd waitUntilCompleted];check(cmd.status==MTLCommandBufferStatusCompleted,"GPU command");
  };
  dispatch(0,2);uint32_t batch[2]={got[0],got[1]};got[0]=0;got[1]=0;
  dispatch(0,1);uint32_t ref0=got[0];dispatch(1,1);uint32_t ref1=got[1];
  for(int i=0;i<2;++i){uint32_t independent=reference(h[i].identity.x,h[i].identity.y,u[i].timing.x,u[i].timing.y,u[i].timing.z,u[i].render.x,px[i]);
   uint32_t single=i?ref1:ref0;
   std::cout<<"env="<<i<<" sample="<<proposed[i].sample<<" frame="<<u[i].timing.x<<" sensor="<<u[i].timing.z<<" globalPixel="<<px[i]<<" batch="<<batch[i]<<" single="<<single<<" cpu="<<independent<<'\n';
   check(batch[i]==single && single==independent,"seed mismatch");}
  check(batch[0]!=batch[1],"seed distinction");
  check(reference(h[1].identity.x,h[1].identity.y,u[1].timing.x,0,u[1].timing.z,0,73)!=batch[1],"pixel-index negative control");
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
}
