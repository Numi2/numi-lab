// Verify the actual slider wrench preparation, not a coupled transaction.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <iostream>
#include <limits>
#include <stdexcept>
static void check(bool b,const char* text){if(!b)throw std::runtime_error(text);}
int main(int argc,char** argv){@autoreleasepool {try {
 check(argc==2,"expected MetalWorld metallib");id<MTLDevice> device=MTLCreateSystemDefaultDevice();
 check(device!=nil,"no device");NSError* error=nil;
 id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
 check(lib!=nil,"no library");id<MTLFunction> fn=[lib newFunctionWithName:@"mr_world_slider_midpoint_prepare_wrench"];
 check(fn!=nil,"kernel missing");id<MTLComputePipelineState> pipe=[device newComputePipelineStateWithFunction:fn error:&error];
 check(pipe!=nil,"no pipeline");id<MTLCommandQueue> queue=[device newCommandQueue];check(queue!=nil,"no queue");
 auto run=[&](const MRABABodyWrenchGPU (&wrench)[2],float effort,std::uint32_t prior,
              std::uint32_t expected,float output) {
  MRMetalWorldContactStatusGPU status{};status.code=prior;status.firstFailingConstraint=7u;
  id<MTLBuffer> wr=[device newBufferWithBytes:wrench length:sizeof(wrench) options:MTLResourceStorageModeShared];
  id<MTLBuffer> ef=[device newBufferWithBytes:&effort length:sizeof(effort) options:MTLResourceStorageModeShared];
  id<MTLBuffer> st=[device newBufferWithBytes:&status length:sizeof(status) options:MTLResourceStorageModeShared];
  check(wr&&ef&&st,"buffer failure");id<MTLCommandBuffer> cmd=[queue commandBuffer];
  id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];check(enc!=nil,"encoder failure");
  [enc setComputePipelineState:pipe];[enc setBuffer:wr offset:0 atIndex:0];
  [enc setBuffer:ef offset:0 atIndex:1];[enc setBuffer:st offset:0 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];check(cmd.status==MTLCommandBufferStatusCompleted,"command failed");
  const auto got=*static_cast<const MRMetalWorldContactStatusGPU*>(st.contents);
  check(got.code==expected && *static_cast<const float*>(ef.contents)==output,"incorrect status or effort");
  if(expected!=MR_STEP_SUCCESS && prior==MR_STEP_SUCCESS)check(got.firstFailingConstraint==0u,"failure index");
  if(prior!=MR_STEP_SUCCESS)check(got.firstFailingConstraint==7u,"previous failure changed");
 };
 MRABABodyWrenchGPU wrench[2]{};run(wrench,4.f,MR_STEP_SUCCESS,MR_STEP_SUCCESS,4.f);
 wrench[1].force.z=-2.f;wrench[1].torque.x=.4f;
 run(wrench,4.f,MR_STEP_SUCCESS,MR_STEP_SUCCESS,6.f);
 wrench[1].force.z=2.f;run(wrench,4.f,MR_STEP_SUCCESS,MR_STEP_SUCCESS,2.f);
 wrench[1].force.x=1.f;run(wrench,4.f,MR_STEP_SUCCESS,MR_STEP_UNSUPPORTED,4.f);
 wrench[1].force.x=0;wrench[0].torque.y=1.f;run(wrench,4.f,MR_STEP_SUCCESS,MR_STEP_UNSUPPORTED,4.f);
 wrench[0].torque.y=0;wrench[1].torque.x=std::numeric_limits<float>::quiet_NaN();
 run(wrench,4.f,MR_STEP_SUCCESS,MR_STEP_NONFINITE_INPUT,4.f);
 run(wrench,4.f,MR_STEP_FACTORIZATION_FAILED,MR_STEP_FACTORIZATION_FAILED,4.f);
 std::cout<<"seven slider wrench preparation cases passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }}
