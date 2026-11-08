#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/unified_quality_shared.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <stdexcept>
static std::vector<char> read(const char*p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("input");return {std::istreambuf_iterator<char>(f),{}};}
static void write(std::filesystem::path p,const void*d,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)d,n);if(!f)throw std::runtime_error("write");}
int main(int argc,char**argv){@autoreleasepool{try{
 if(argc!=5)throw std::runtime_error("METALLIB CAPTURE NEW_OUTPUT LEGACY_ARENA");
 auto raw=read(argv[2]);if(raw.size()<256)throw std::runtime_error("header");
 uint32_t h[8];memcpy(h,raw.data(),32);if(h[0]!=0x51464c31||h[1]!=1||h[5]!=raw.size())throw std::runtime_error("capture identity");
 MRUnifiedQualityDispatchGPU d;memcpy(&d,raw.data()+32,sizeof(d));uint32_t nv=d.generalizedVelocityCount,rs=d.rowCount,nb=d.blockCount;
 if(!nv||nv>64||!rs||rs>768||!nb||nb>192)throw std::runtime_error("dimensions");
 d.problemCount=1;d.blockStride=0;d.dynamicsStride=nv*nv;d.jacobianStride=rs*nv;d.vectorStride=std::max(nv,rs);d.derivativeStride=36*nb;
 d.hessianStride=std::string(argv[4])=="1"?nv*nv:mrUnifiedQualityRefinementScratchFloats(nv,rs,nb);
 size_t at=256;auto take=[&](size_t n){if(at+n>raw.size())throw std::runtime_error("truncated");const char*p=raw.data()+at;at+=n;return p;};
 auto blocks=take(128*nb),A=take(4*nv*nv),J=take(4*rs*nv),bias=take(4*rs),freev=take(4*nv),warm=take(4*nv),wp=take(4*rs);
 id<MTLDevice>dev=MTLCreateSystemDefaultDevice();NSError*error=nil;
 auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto fn=[lib newFunctionWithName:@"mr_unified_quality_solve"];auto pipe=[dev newComputePipelineStateWithFunction:fn error:&error];if(!pipe)throw std::runtime_error(error.localizedDescription.UTF8String);
 if(pipe.threadExecutionWidth!=32||pipe.maxTotalThreadsPerThreadgroup<32||pipe.staticThreadgroupMemoryLength>dev.maxThreadgroupMemoryLength)throw std::runtime_error("SIMD32 capacity");
 std::vector<id<MTLBuffer>>buf;
 auto add=[&](const void*p,size_t count){auto b=[dev newBufferWithLength:std::max(count,size_t(16)) options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("allocation");memset(b.contents,0,count);if(p)memcpy(b.contents,p,count);buf.push_back(b);};
 add(&d,sizeof(d));add(blocks,128*nb);add(A,4*nv*nv);add(J,4*rs*nv);add(bias,4*rs);
 for(auto p:{freev,warm}){add(nullptr,4*d.vectorStride);memcpy(buf.back().contents,p,4*nv);}
 add(wp,4*rs);
 add(nullptr,4*d.vectorStride);add(nullptr,4*d.vectorStride);add(nullptr,4*d.derivativeStride);add(nullptr,4*d.hessianStride);add(nullptr,sizeof(MRUnifiedQualityStatusGPU));
 for(unsigned i:{8u,9u,10u,11u,12u})memset(buf[i].contents,0xCD,buf[i].length);
 auto queue=[dev newCommandQueue];auto cmd=[queue commandBuffer];auto enc=[cmd computeCommandEncoder];[enc setComputePipelineState:pipe];
 for(NSUInteger i=0;i<buf.size();++i)[enc setBuffer:buf[i] offset:0 atIndex:i];
 [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
 if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cmd.error.localizedDescription.UTF8String);
 std::filesystem::path out(argv[3]);if(!std::filesystem::create_directory(out))throw std::runtime_error("output exists");
 write(out/"dispatch.bin",&d,sizeof(d));write(out/"status.bin",buf[12].contents,sizeof(MRUnifiedQualityStatusGPU));
 write(out/"velocity.f32",buf[8].contents,4*d.vectorStride);write(out/"impulse.f32",buf[9].contents,4*d.vectorStride);
 printf("{\"gpu_seconds\":%.9g,\"static_threadgroup_bytes\":%zu,\"hessian_bytes\":%u,\"code\":%u}\n",cmd.GPUEndTime-cmd.GPUStartTime,(size_t)pipe.staticThreadgroupMemoryLength,4*d.hessianStride,((MRUnifiedQualityStatusGPU*)buf[12].contents)->code);
 return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}}
