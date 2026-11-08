#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#include <cstdio>
struct Input { simd_float4 p[17],target[17],parameters; simd_uint4 counts; };
struct Output { simd_float4 position[17],reaction[17],info; };
static_assert(sizeof(Input)==576); static_assert(sizeof(Output)==560);
int main(int argc,const char**argv) {
    @autoreleasepool {
        if(argc!=4){std::fprintf(stderr,"usage: host shader input output\n");return 2;}
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        if(!dev){std::fprintf(stderr,"no Metal device\n");return 3;}
        NSError* error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if(!lib){std::fprintf(stderr,"library: %s\n",error.localizedDescription.UTF8String);return 4;}
        id<MTLFunction> fn=[lib newFunctionWithName:@"rod_block"];
        id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:fn error:&error];
        if(!pipe){std::fprintf(stderr,"pipeline: %s\n",error.localizedDescription.UTF8String);return 5;}
        NSData* data=[NSData dataWithContentsOfFile:@(argv[2])];
        if(!data || data.length==0 || data.length%sizeof(Input)){std::fprintf(stderr,"invalid input size\n");return 6;}
        NSUInteger count=data.length/sizeof(Input);
        id<MTLBuffer> input=[dev newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> output=[dev newBufferWithLength:count*sizeof(Output) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        if(!input||!output||!queue||!command||!enc){std::fprintf(stderr,"allocation or encoder failure\n");return 7;}
        [enc setComputePipelineState:pipe];[enc setBuffer:input offset:0 atIndex:0];[enc setBuffer:output offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];
        [command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted){std::fprintf(stderr,"command failed: %s\n",command.error.localizedDescription.UTF8String);return 8;}
        NSData* saved=[NSData dataWithBytes:output.contents length:output.length];
        if(![saved writeToFile:@(argv[3]) atomically:YES]){std::fprintf(stderr,"output write failed\n");return 9;}
        std::printf("device=%s cases=%lu input_stride=%zu output_stride=%zu\n",dev.name.UTF8String,(unsigned long)count,sizeof(Input),sizeof(Output));
        return 0;
    }
}
