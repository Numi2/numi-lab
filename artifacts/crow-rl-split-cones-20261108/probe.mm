#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
static std::vector<char> read(const char*p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("input");return {std::istreambuf_iterator<char>(f),{}};}
static void write(std::filesystem::path p,const void*d,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)d,n);if(!f)throw std::runtime_error("write");}
int main(int argc,char**argv){@autoreleasepool{try{
 if(argc!=4)throw std::runtime_error("METALLIB INPUT NEW_OUTPUT");
 auto raw=read(argv[2]);if(raw.size()<32)throw std::runtime_error("header");auto c=(uint32_t*)raw.data();
 uint32_t nv=c[2],rows=c[3],nb=c[4];
 if(c[0]!=0x53504331||c[1]!=1||!nv||nv>64||rows>768||nb>192||c[5]>20||c[6]>1||
    raw.size()!=32+128*nb+4*(nv*nv+rows*nv+rows+2*nv))throw std::runtime_error("dimensions");
 id<MTLDevice>dev=MTLCreateSystemDefaultDevice();if(!dev)throw std::runtime_error("Metal");
 NSError*e=nil;auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];if(!lib)throw std::runtime_error(e.localizedDescription.UTF8String);
 auto fn=[lib newFunctionWithName:@"solve"];auto pipe=[dev newComputePipelineStateWithFunction:fn error:&e];if(!pipe)throw std::runtime_error(e.localizedDescription.UTF8String);
 std::vector<id<MTLBuffer>>buffers;buffers.push_back([dev newBufferWithBytes:raw.data() length:32 options:MTLResourceStorageModeShared]);size_t at=32;
 for(size_t bytes:{size_t(128*nb),size_t(4*nv*nv),size_t(4*rows*nv),size_t(4*rows),size_t(4*nv),size_t(4*nv)}) {
  if(bytes)buffers.push_back([dev newBufferWithBytes:raw.data()+at length:bytes options:MTLResourceStorageModeShared]);
  else buffers.push_back([dev newBufferWithLength:16 options:MTLResourceStorageModeShared]);at+=bytes;
 }
 size_t scratch=8*(4*nv+2*rows+36*nb+nv*nv+1);
 buffers.push_back([dev newBufferWithLength:scratch options:MTLResourceStorageModeShared]);buffers.push_back([dev newBufferWithLength:128 options:MTLResourceStorageModeShared]);
 for(auto b:buffers)if(!b)throw std::runtime_error("allocation");
 memset(buffers[7].contents,0xCD,buffers[7].length);memset(buffers[8].contents,0xCD,buffers[8].length);
 auto queue=[dev newCommandQueue];auto cmd=[queue commandBuffer];auto enc=[cmd computeCommandEncoder];[enc setComputePipelineState:pipe];
 for(NSUInteger i=0;i<buffers.size();++i)[enc setBuffer:buffers[i] offset:0 atIndex:i];
 [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
 if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error.localizedDescription.UTF8String);
 std::filesystem::path out(argv[3]);if(!std::filesystem::create_directory(out))throw std::runtime_error("output exists");
 write(out/"scratch.f32x2",buffers[7].contents,scratch);write(out/"status.bin",buffers[8].contents,128);
 printf("{\"device\":\"%s\",\"gpu_seconds\":%.9g,\"scratch_bytes\":%zu,\"code\":%u}\n",dev.name.UTF8String,cmd.GPUEndTime-cmd.GPUStartTime,scratch,((uint32_t*)buffers[8].contents)[0]);return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}}
