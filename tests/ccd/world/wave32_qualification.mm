#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/constraint_ir_shared.h"
#include <cmath>
#include <cstdio>
#include <cstring>
int main(int argc,char** argv) {
    @autoreleasepool {
        if(argc!=2)return 2;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        if(!lib){std::fprintf(stderr,"library: %s\n",error.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"numi_wave32_qualification"] error:&error];
        if(!pipeline){std::fprintf(stderr,"pipeline: %s\n",error.localizedDescription.UTF8String);return 2;}
        constexpr unsigned count=7;
        float inputs[count][16]={};
        MREvaluatedConstraintIRConeGPU cones[count]{};
        auto setup=[&](unsigned i,float d0,float d1,float d2,float a,float b,float c,float mu0,float mu1,unsigned iterations,float maximum=0.f){
            inputs[i][0]=d0;inputs[i][4]=d1;inputs[i][8]=d2;
            inputs[i][9]=a;inputs[i][10]=b;inputs[i][11]=c;inputs[i][12]=iterations;
            cones[i].effectiveFrictionU=mu0;cones[i].effectiveFrictionV=mu1;cones[i].maximumNormalImpulse=maximum;
        };
        setup(0,1e-12f,1,1,1,0,0,0,0,1);
        setup(1,1,2,1,2,-.1f,-.2f,0,0,128);
        setup(2,1,2,1,2,-.1f,-.2f,.5f,.5f,128);
        setup(3,1,1,1,2,0,0,0,0,128,1);
        setup(4,1,1,1,2,-3,-4,.8f,.2f,128);
        setup(5,1,1,1,2,-3,0,.5f,0,128);
        setup(6,1,1,1,1,0,0,0,0,3);inputs[6][13]=1;
        id<MTLBuffer> in=[device newBufferWithBytes:inputs length:sizeof(inputs) options:MTLResourceStorageModeShared];
        id<MTLBuffer> co=[device newBufferWithBytes:cones length:sizeof(cones) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out=[device newBufferWithLength:count*4*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[[device newCommandQueue] commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:co offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(count,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)return 3;
        const float* v=(const float*)out.contents;
        std::printf("{\"cases\":[");
        for(unsigned i=0;i<count;++i){
            if(i)std::printf(",");
            std::printf("{\"id\":%u,\"impulse\":[%.9g,%.9g,%.9g],\"normal_response\":%.9g}",i,v[4*i],v[4*i+1],v[4*i+2],v[4*i+3]);
        }
        std::printf("]}\n");
        return 0;
    }
}
