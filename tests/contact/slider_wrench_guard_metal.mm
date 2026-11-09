// Exercise the actual owner guard. This does not qualify a coupled step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static void check(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
int main(int argc,char** argv) { @autoreleasepool { try {
    check(argc==2,"expected MetalWorld metallib");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); check(device!=nil,"no Metal device");
    NSError* error=nil;
    id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:
        [NSString stringWithUTF8String:argv[1]]] error:&error];
    check(lib!=nil,"cannot load MetalWorld library");
    id<MTLFunction> fn=[lib newFunctionWithName:@"mr_world_slider_midpoint_promote_status_with_wrench_guard"];
    check(fn!=nil,"wrench guard missing");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
    check(pipeline!=nil,"cannot create wrench guard pipeline");
    id<MTLCommandQueue> queue=[device newCommandQueue];check(queue!=nil,"no queue");
    MRMetalWorldPassGPU pass{};pass.physicsSubstep=3u;
    auto run=[&](const MRABABodyWrenchGPU (&wrench)[2],std::uint32_t enabled,
                 std::uint32_t candidateCode,std::uint32_t worldCode,
                 std::uint32_t expected,std::uint32_t expectedIndex) {
        MRMetalWorldContactStatusGPU candidate{};candidate.code=candidateCode;
        candidate.firstFailingConstraint=7u;
        MRMetalWorldStatusGPU initial{};initial.code=worldCode;initial.failingIndex=11u;
        id<MTLBuffer> cb=[device newBufferWithBytes:&candidate length:sizeof(candidate) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wb=[device newBufferWithBytes:&initial length:sizeof(initial) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wr=[device newBufferWithBytes:wrench length:sizeof(wrench) options:MTLResourceStorageModeShared];
        check(cb&&wb&&wr,"buffer allocation failed");
        id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
        check(enc!=nil,"encoder failed");[enc setComputePipelineState:pipeline];
        [enc setBytes:&pass length:sizeof(pass) atIndex:0];[enc setBuffer:cb offset:0 atIndex:1];
        [enc setBuffer:wb offset:0 atIndex:2];[enc setBuffer:wr offset:0 atIndex:3];
        [enc setBytes:&enabled length:sizeof(enabled) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
        check(cmd.status==MTLCommandBufferStatusCompleted,"guard command failed");
        const auto result=*static_cast<const MRMetalWorldStatusGPU*>(wb.contents);
        check(result.code==expected && result.failingIndex==expectedIndex,"wrong guard status or index");
        if (expected!=MR_STEP_SUCCESS && worldCode==MR_STEP_SUCCESS)
            check(result.failingSubstep==3u,"failure substep not promoted");
    };
    MRABABodyWrenchGPU wrench[2]{};
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_SUCCESS,11u);
    wrench[1].torque.x=0.3f; wrench[1].torque.y=-0.2f;
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_SUCCESS,11u);
    wrench[1].force.z=0.1f;
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_SUCCESS,11u);
    wrench[1].force.x=0.1f;
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_UNSUPPORTED,1u);
    run(wrench,0u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_SUCCESS,11u);
    wrench[1].force.x=0;wrench[1].force.z=0;wrench[0].torque.x=0.2f;
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_UNSUPPORTED,0u);
    wrench[0].torque.x=0;wrench[1].torque.w=1;
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_UNSUPPORTED,1u);
    wrench[1].torque.w=0;wrench[1].torque.x=std::numeric_limits<float>::quiet_NaN();
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_SUCCESS,MR_STEP_NONFINITE_INPUT,1u);
    run(wrench,1u,MR_STEP_FACTORIZATION_FAILED,MR_STEP_SUCCESS,MR_STEP_FACTORIZATION_FAILED,7u);
    run(wrench,1u,MR_STEP_SUCCESS,MR_STEP_FACTORIZATION_FAILED,MR_STEP_FACTORIZATION_FAILED,11u);
    std::cout<<"ten slider owner wrench guard cases passed\n";
    return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;} } }
