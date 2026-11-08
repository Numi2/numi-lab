#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(int argc,char**argv){@autoreleasepool{try{
 need(argc==4,"Usage: certificate METALLIB SNAPSHOT NEW_OUTPUT");
 need(std::filesystem::file_size(argv[2])==sizeof(MRBoundedDriveFailureGPU),"Snapshot size mismatch");
 need(!std::filesystem::exists(argv[3]),"Output already exists");
 MRBoundedDriveFailureGPU snapshot{};std::ifstream in(argv[2],std::ios::binary);in.read(reinterpret_cast<char*>(&snapshot),sizeof(snapshot));need(bool(in),"Snapshot read failed");
 need(snapshot.valid==0x42444631u && snapshot.nv>0 && snapshot.nv<=MR_ARTICULATED_ABA_MAX_DOFS,"Snapshot identity invalid");
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");NSError* error=nil;
 id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];need(library!=nil,"Explicit Metal library unavailable");
 id<MTLFunction> function=[library newFunctionWithName:@"mr_bounded_drive_certificate_probe"];need(function!=nil,"Certificate kernel missing");
 id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];need(pipeline!=nil,"Certificate pipeline unavailable");
 id<MTLBuffer> input=[device newBufferWithBytes:&snapshot length:sizeof(snapshot) options:MTLResourceStorageModeShared];
 const auto size=std::size_t(snapshot.nv)*sizeof(mr_float4);id<MTLBuffer> output=[device newBufferWithLength:size options:MTLResourceStorageModeShared];
 id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
 need(input!=nil && output!=nil && queue!=nil && command!=nil && encoder!=nil,"Certificate resources unavailable");
 [encoder setComputePipelineState:pipeline];[encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
 [encoder dispatchThreads:MTLSizeMake(snapshot.nv,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
 need(command.status==MTLCommandBufferStatusCompleted,"Certificate GPU command failed");
 const int fd=::open(argv[3],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0644);need(fd>=0,"New certificate output unavailable");
 const auto written=::write(fd,output.contents,size);const auto closed=::close(fd);need(written==ssize_t(size)&&closed==0,"Certificate output incomplete");
 std::cout<<"{\"device\":\""<<device.name.UTF8String<<"\",\"rows\":"<<snapshot.nv<<",\"native_bytes\":"<<size<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
