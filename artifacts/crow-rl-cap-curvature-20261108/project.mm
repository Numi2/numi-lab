#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <filesystem>
static std::vector<char> read(const char* p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("input");return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char**argv){@autoreleasepool{try{
 if(argc!=4)throw std::runtime_error("METALLIB INPUT OUTPUT");
 auto raw=read(argv[2]);if(raw.size()<16)throw std::runtime_error("short");
 auto h=(const uint32_t*)raw.data();uint32_t n=h[2];if(h[0]!=0x43415031||h[1]!=1||!n||n>256||raw.size()!=16+152*n)throw std::runtime_error("header");
 std::vector<char> blocks(128*n);std::vector<float>w(6*n);
 for(uint32_t i=0;i<n;++i){memcpy(blocks.data()+128*i,raw.data()+16+152*i,128);memcpy(w.data()+6*i,raw.data()+16+152*i+128,24);}
 id<MTLDevice>dev=MTLCreateSystemDefaultDevice();if(!dev)throw std::runtime_error("Metal");
 NSError*e=nil;auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];if(!lib)throw std::runtime_error(e.localizedDescription.UTF8String);
 auto fn=[lib newFunctionWithName:@"cap_projection_probe"];auto pipeline=[dev newComputePipelineStateWithFunction:fn error:&e];if(!pipeline)throw std::runtime_error(e.localizedDescription.UTF8String);
 auto bb=[dev newBufferWithBytes:blocks.data() length:blocks.size() options:MTLResourceStorageModeShared];
 auto wb=[dev newBufferWithBytes:w.data() length:w.size()*4 options:MTLResourceStorageModeShared];
 auto ob=[dev newBufferWithLength:48*n*4 options:MTLResourceStorageModeShared];if(!bb||!wb||!ob)throw std::runtime_error("allocation");memset(ob.contents,0xCD,ob.length);
 auto queue=[dev newCommandQueue];auto cmd=[queue commandBuffer];auto encoder=[cmd computeCommandEncoder];[encoder setComputePipelineState:pipeline];
 [encoder setBuffer:bb offset:0 atIndex:0];[encoder setBuffer:wb offset:0 atIndex:1];[encoder setBuffer:ob offset:0 atIndex:2];[encoder setBytes:&n length:4 atIndex:3];
 [encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[cmd commit];[cmd waitUntilCompleted];
 if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error.localizedDescription.UTF8String);
 if(std::filesystem::exists(argv[3]))throw std::runtime_error("output exists");
 std::ofstream out(argv[3],std::ios::binary);out.write((const char*)ob.contents,ob.length);if(!out)throw std::runtime_error("write");
 printf("{\"device\":\"%s\",\"cases\":%u,\"gpu_seconds\":%.9g}\n",dev.name.UTF8String,n,cmd.GPUEndTime-cmd.GPUStartTime);return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}}
