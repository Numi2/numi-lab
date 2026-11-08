#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/unified_quality_capture_shared.h"
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
int main(int argc,char**argv){@autoreleasepool{try{
    require(argc>=4 && argc<=6,"Usage: probe INPUT METALLIB NEW_OUTPUT [WARM_F32 [--polish-scratch]]");
    std::ifstream input(argv[1],std::ios::binary|std::ios::ate);
    require(bool(input),"Missing input");const auto length=input.tellg();
    require(length>=256 && length<=4*1024*1024,"Input size");
    std::vector<char> raw(static_cast<std::size_t>(length));input.seekg(0);input.read(raw.data(),length);
    require(bool(input),"Truncated input");MRUnifiedQualityFailureHeaderGPU header{};
    std::memcpy(&header,raw.data(),sizeof(header));auto d=header.dispatch;
    if(argc==6){require(std::string(argv[5])=="--polish-scratch","Unknown scratch selector");d.hessianStride=d.generalizedVelocityCount*d.generalizedVelocityCount;}
    require(header.valid==MR_UNIFIED_QUALITY_FAILURE_MAGIC && header.version==1u &&
        header.environment==0u && header.packet.identity.x==0u &&
        header.recordBytes==raw.size() && header.recordBytes==
            mrUnifiedQualityFailureRecordBytes(d.generalizedVelocityCount,d.rowCount,d.blockCount),
        "Invalid first-lane capture");
    require(d.problemCount<=64u,"Problem capacity");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"No Metal device");
    NSError* error=nil;id<MTLLibrary> library=[device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];
    require(library!=nil,"No shader library");
    id<MTLFunction> function=[library newFunctionWithName:@"mr_unified_quality_solve_queued_capture"];
    require(function!=nil,"Missing capture kernel");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline!=nil,"Pipeline failed");
    const char* cursor=raw.data()+sizeof(header);
    auto take=[&](std::size_t bytes){id<MTLBuffer> b=[device newBufferWithBytes:cursor length:bytes
        options:MTLResourceStorageModeShared];cursor+=bytes;require(b!=nil,"Input allocation");return b;};
    auto empty=[&](std::size_t bytes){id<MTLBuffer>b=[device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];require(b!=nil,"Scratch allocation");std::memset(b.contents,0,bytes);return b;};
    const auto nv=d.generalizedVelocityCount,rows=d.rowCount;
    id<MTLBuffer> buffers[16]={};
    buffers[0]=[device newBufferWithBytes:&d length:sizeof(d) options:MTLResourceStorageModeShared];
    buffers[1]=take(d.blockCount*sizeof(MRUnifiedQualityBlockGPU));
    buffers[2]=take(nv*nv*sizeof(float));buffers[3]=take(rows*nv*sizeof(float));
    buffers[4]=take(rows*sizeof(float));buffers[5]=take(nv*sizeof(float));
    buffers[6]=take(nv*sizeof(float));buffers[7]=take(rows*sizeof(float));
    std::vector<float> suppliedWarm;
    if(argc>=5){std::ifstream f(argv[4],std::ios::binary|std::ios::ate);require(bool(f),"Missing warm state");
        require(f.tellg()==std::streamoff(nv*sizeof(float)),"Warm state size");suppliedWarm.resize(nv);
        f.seekg(0);f.read(reinterpret_cast<char*>(suppliedWarm.data()),nv*sizeof(float));require(bool(f),"Truncated warm state");
        for(float v:suppliedWarm)require(std::isfinite(v),"Nonfinite warm state");
        buffers[6]=[device newBufferWithBytes:suppliedWarm.data() length:nv*sizeof(float) options:MTLResourceStorageModeShared];}
    buffers[8]=empty(d.vectorStride*sizeof(float));buffers[9]=empty(d.vectorStride*sizeof(float));
    buffers[10]=empty(d.derivativeStride*sizeof(float));buffers[11]=empty(d.hessianStride*sizeof(float));
    buffers[12]=empty(d.problemCount*sizeof(MRUnifiedQualityStatusGPU));
    buffers[13]=[device newBufferWithBytes:&header.packet length:sizeof(header.packet)
        options:MTLResourceStorageModeShared];
    MRUnifiedQualityWorkQueueGPU queue{};queue.count=queue.capacity=queue.required=1u;
    buffers[14]=[device newBufferWithBytes:&queue length:sizeof(queue) options:MTLResourceStorageModeShared];
    buffers[15]=empty(header.recordBytes*d.problemCount);
    id<MTLCommandQueue> commandQueue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(encoder!=nil,"No encoder");[encoder setComputePipelineState:pipeline];
    for(unsigned i=0;i<16;++i){require(buffers[i]!=nil,"Missing buffer");
        [encoder setBuffer:buffers[i] offset:0u atIndex:i];}
    const MRUnifiedQualityFailurePassGPU pass{header.controlStep,header.physicsSubstep,header.recordBytes,0u};
    [encoder setBytes:&pass length:sizeof(pass) atIndex:16u];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Native command failed");
    MRUnifiedQualityStatusGPU result{};std::memcpy(&result,buffers[12].contents,sizeof(result));
    require(result.code==0u || result.code==7u,"Unexpected solver failure");
    const std::filesystem::path directory(argv[3]);require(std::filesystem::create_directory(directory),"Output exists");
    auto save=[&](const char* name,const void* bytes,std::size_t size){std::ofstream f(directory/name,std::ios::binary);
        f.write(static_cast<const char*>(bytes),size);require(bool(f),"Write failed");};
    save("status.bin",&result,sizeof(result));
    if(result.code==0u){save("velocity.f32",buffers[8].contents,nv*sizeof(float));save("impulse.f32",buffers[9].contents,rows*sizeof(float));}
    else {MRUnifiedQualityFailureHeaderGPU witness{};std::memcpy(&witness,buffers[15].contents,sizeof(witness));
        require(witness.valid==MR_UNIFIED_QUALITY_FAILURE_MAGIC,"Missing failure witness");save("capture.bin",buffers[15].contents,header.recordBytes);}
    std::cout<<"{\"device\":\""<<device.name.UTF8String<<"\",\"code\":"<<result.code
        <<",\"newton\":"<<result.newtonIterations<<",\"pcg\":"<<result.pcgIterations
        <<",\"optimality\":"<<result.certificates0.x<<",\"feasibility\":"<<result.certificates0.y
        <<",\"gpu_ms\":"<<(command.GPUEndTime-command.GPUStartTime)*1000.<<"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
