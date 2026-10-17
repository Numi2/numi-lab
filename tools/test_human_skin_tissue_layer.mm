#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

static void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
static std::string read(const char* path) {
    std::ifstream f(path);require(f.good(),"source file unavailable");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
static std::string structure(const std::string& header,const char* name) {
    const auto end=header.find(std::string("} ")+name+";");
    require(end!=std::string::npos,"native structure missing");
    const auto begin=header.rfind("typedef struct ",end);
    require(begin!=std::string::npos,"native structure start missing");
    return header.substr(begin,end+std::strlen(name)+3-begin);
}
int main(int argc,char** argv) { @autoreleasepool { try {
    require(argc==4,"usage: test_human_skin_tissue_layer KERNEL VISUAL_HEADER RESTING_HEADER");
    static_assert(sizeof(MRHumanRestingSkinLayerVertexGPU)==64);
    static_assert(sizeof(MRVisualVertexGPUV2)==80);
    const auto metal=read(argv[1]);
    const std::string marker="// BEGIN material-frame skin tissue layer";
    const auto begin=metal.find(marker);
    require(begin!=std::string::npos,"native kernel marker missing");
    const auto source=std::string("#include <metal_stdlib>\nusing namespace metal;\n"
        "#define MR_ALIGN16 alignas(16)\n"
        "using mr_float4=float4; using mr_uint4=uint4;\n")+
        structure(read(argv[2]),"MRVisualVertexGPUV2")+"\n"+
        structure(read(argv[3]),"MRHumanRestingSkinLayerVertexGPU")+"\n"+
        metal.substr(begin);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");
    NSError* error=nil;
    auto lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                   options:nil error:&error];
    if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
    auto pipeline=[device newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"nm_human_resting_skin_tissue_layer"] error:&error];
    require(pipeline!=nil,"native skin layer pipeline failed");
    MRVisualVertexGPUV2 vertices[4]{};
    vertices[0].position={0,0,0,1};vertices[1].position={1,0,0,1};
    vertices[2].position={0,0,1,1};vertices[3].color={.1f,.2f,.3f,1};
    vertices[3].texcoord01={.4f,.5f,.6f,.7f};
    MRHumanRestingSkinLayerVertexGPU map{};
    map.parentAndOutput={0,1,2,3};map.barycentric={.2f,.3f,.5f,0};
    map.offset={0,0,-.001f,0};map.normal={0,0,-1,0};
    auto vb=[device newBufferWithBytes:vertices length:sizeof(vertices) options:MTLResourceStorageModeShared];
    auto mb=[device newBufferWithBytes:&map length:sizeof(map) options:MTLResourceStorageModeShared];
    auto queue=[device newCommandQueue];
    auto run=[&]{
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        unsigned count=1;
        [e setComputePipelineState:pipeline];[e setBytes:&count length:sizeof(count) atIndex:0];
        [e setBuffer:mb offset:0 atIndex:1];[e setBuffer:vb offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        require(cb.status==MTLCommandBufferStatusCompleted,"skin layer GPU command failed");
    };
    run();
    auto* out=static_cast<MRVisualVertexGPUV2*>(vb.contents);
    auto near=[](float a,float b){return std::abs(a-b)<=2e-7f;};
    require(near(out[3].position.x,.3f)&&near(out[3].position.y,.001f)&&
            near(out[3].position.z,.5f),"source thickness failed under rigid rotation");
    require(near(out[3].normalAndTangentSign.x,0)&&
            near(out[3].normalAndTangentSign.y,1)&&
            near(out[3].normalAndTangentSign.z,0),"inner surface normal failed");
    require(std::memcmp(&out[3].color,&vertices[3].color,sizeof(mr_float4))==0&&
            std::memcmp(&out[3].texcoord01,&vertices[3].texcoord01,sizeof(mr_float4))==0,
            "transport changed unrelated skin attributes");
    const auto prior=out[3];run();
    require(std::memcmp(&prior,&out[3],sizeof(prior))==0,"unchanged accepted state drifted");
    out[1].position={2,0,0,1};out[2].position={0,0,3,1};run();
    require(near(out[3].position.x,.6f)&&near(out[3].position.y,.001f)&&
            near(out[3].position.z,1.5f),"stretch changed source layer thickness");
    out[2].position=out[1].position;run();
    require(!std::isfinite(out[3].position.x),"collapsed parent was silently accepted");
    std::cout<<"PASS skin-layer Metal rotation, thickness, attributes, replay, collapse on "
             <<device.name.UTF8String<<"\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} } }
