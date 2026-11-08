#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
static std::vector<char> read(const char* p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("input open");return {std::istreambuf_iterator<char>(f),{}};}
static void write(const std::filesystem::path& p,const void* d,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)d,n);if(!f)throw std::runtime_error("output write");}
int main(int argc,char**argv){@autoreleasepool{try {
    if(argc<4)throw std::runtime_error("usage MODE METALLIB OUTPUT [INPUT]");
    bool calibration=std::string(argv[1])=="calibrate";
    if(!calibration&&argc!=5)throw std::runtime_error("input required");
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();if(!dev)throw std::runtime_error("no Metal device");
    NSError* error=nil;id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error];
    if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLFunction> fn=[lib newFunctionWithName:calibration?@"calibrate":@"solve"];
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:fn error:&error];
    if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
    std::vector<id<MTLBuffer>> buffers;uint32_t nv=0,rows=0;size_t scratchBytes=64;
    if(calibration)buffers.push_back([dev newBufferWithLength:64 options:MTLResourceStorageModeShared]);
    else {
        auto raw=read(argv[4]);if(raw.size()<32)throw std::runtime_error("short input");
        auto c=(const uint32_t*)raw.data();nv=c[2];rows=c[3];uint32_t blocks=c[4];
        if(c[0]!=0x53504c31||c[1]!=1||!nv||nv>64||rows>768||blocks>256||c[5]>4||c[6]>1)throw std::runtime_error("bad dimensions");
        size_t expected=32+128*blocks+4*(nv*nv+rows*nv+rows+2*nv);
        if(raw.size()!=expected)throw std::runtime_error("bad input bytes");
        buffers.push_back([dev newBufferWithBytes:raw.data() length:32 options:MTLResourceStorageModeShared]);
        size_t offset=32;
        for(size_t bytes:{size_t(128*blocks),size_t(4*nv*nv),size_t(4*rows*nv),size_t(4*rows),size_t(4*nv),size_t(4*nv)}) {
            if(bytes)buffers.push_back([dev newBufferWithBytes:raw.data()+offset length:bytes options:MTLResourceStorageModeShared]);
            else buffers.push_back([dev newBufferWithLength:16 options:MTLResourceStorageModeShared]);
            offset+=bytes;
        }
        scratchBytes=8*(3*nv+3*rows+nv*nv);
        buffers.push_back([dev newBufferWithLength:scratchBytes options:MTLResourceStorageModeShared]);
        buffers.push_back([dev newBufferWithLength:64 options:MTLResourceStorageModeShared]);
    }
    for(auto b:buffers)if(!b)throw std::runtime_error("allocation");
    if(calibration)memset(buffers[0].contents,0xCD,buffers[0].length);
    else for(size_t k:{size_t(7),size_t(8)})memset(buffers[k].contents,0xCD,buffers[k].length);
    id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cmd=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[cmd computeCommandEncoder];[encoder setComputePipelineState:pipeline];
    for(NSUInteger i=0;i<buffers.size();++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [encoder endEncoding];[cmd commit];[cmd waitUntilCompleted];
    if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error.localizedDescription.UTF8String);
    std::filesystem::path out(argv[3]);if(!std::filesystem::create_directory(out))throw std::runtime_error("output exists");
    if(calibration)write(out/"arithmetic.f32x2",buffers[0].contents,64);
    else {
        write(out/"scratch.f32x2",buffers[7].contents,scratchBytes);
        write(out/"status.bin",buffers[8].contents,64);
    }
    printf("{\"device\":\"%s\",\"gpu_seconds\":%.9g,\"thread_width\":%lu,\"scratch_bytes\":%zu,\"code\":%u}\n",
        dev.name.UTF8String,cmd.GPUEndTime-cmd.GPUStartTime,(unsigned long)pipeline.threadExecutionWidth,scratchBytes,
        calibration?0:((uint32_t*)buffers[8].contents)[0]);
    return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}}
