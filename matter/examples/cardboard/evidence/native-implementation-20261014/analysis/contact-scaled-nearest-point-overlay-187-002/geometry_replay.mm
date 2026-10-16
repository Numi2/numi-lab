#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>
static void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
int main(int argc,const char** argv){@autoreleasepool{try{
    require(argc==3,"usage: geometry-replay <metallib> <fresh-output>");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"no Metal device");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];require(library!=nil,"cannot load metallib");
    id<MTLFunction> function=[library newFunctionWithName:@"numi_matter_metal::nm_contact_scaled_geometry_replay"];require(function!=nil,"missing geometry kernel");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];require(pipeline!=nil,"cannot create pipeline");
    constexpr NSUInteger bytes=48u*16u*16u;
    id<MTLBuffer> output=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];require(output!=nil,"no output arena");
    id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"no command queue");
    id<MTLCommandBuffer> command=[queue commandBuffer];require(command!=nil,"no command buffer");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];require(encoder!=nil,"no encoder");
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:output offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(48,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
    int fd=::open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0600);require(fd>=0,"output must be fresh");
    const auto* cursor=static_cast<const char*>(output.contents);NSUInteger left=bytes;
    while(left){ssize_t count=::write(fd,cursor,left);require(count>0,"output write failed");cursor+=count;left-=static_cast<NSUInteger>(count);}
    require(::fsync(fd)==0,"output fsync failed");require(::close(fd)==0,"output close failed");
    std::printf("geometry_rows=48\nphysical_transactions=0\n");return 0;
}catch(const std::exception& ex){std::fprintf(stderr,"geometry replay failed: %s\n",ex.what());return 2;}}}
