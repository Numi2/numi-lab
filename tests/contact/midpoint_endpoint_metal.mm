#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
static float component(const nm_float4& x, int i) {
    return i == 0 ? x.x : i == 1 ? x.y : x.z;
}
static NMFEMNodeStateGPU node(float x, float y, float z) {
    NMFEMNodeStateGPU a{};
    a.positionAndMass={x,y,z,1.0f};
    a.velocityAndInverseMass={0,0,0,1.0f};
    a.restAndFixed={x,y,z,0};
    a.referenceDisplacementAndMode={0,0,0,1};
    return a;
}
struct Fixture {
    const char* name;
    NMFEMNodeStateGPU accepted, mid;
    nm_float4 plane{0,0,1,0};
    float clearance=1e-5f;
    std::uint32_t expected=0;
};
int main(int argc, const char** argv) {
 @autoreleasepool { try {
    require(argc==2,"expected metallib path");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device != nil,"Metal device unavailable");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    require(library != nil, error ? error.localizedDescription.UTF8String : "library error");
    id<MTLFunction> function=[library newFunctionWithName:@"numi_matter_metal::nm_fem_midpoint_reflect_endpoint_scratch"];
    require(function != nil,"endpoint kernel missing");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil,error ? error.localizedDescription.UTF8String : "pipeline error");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    require(queue != nil,"queue unavailable");
    Fixture f[12];
    for(auto& x:f) { x.accepted=node(0,0,.001f); x.mid=x.accepted; }
    f[0].name="free"; f[0].mid.referenceDisplacementAndMode.z=-.0001f;
    f[0].accepted.velocityAndInverseMass.z=-.1f; f[0].mid.velocityAndInverseMass.z=-.2f;
    f[1].name="fixed"; f[1].accepted.restAndFixed.w=f[1].mid.restAndFixed.w=1;
    f[1].accepted.velocityAndInverseMass.w=f[1].mid.velocityAndInverseMass.w=0;
    f[2].name="reflected-crossing"; f[2].mid.referenceDisplacementAndMode.z=-.0006f; f[2].expected=2;
    f[3].name="initially-unsafe"; f[3].accepted=node(0,0,-.001f); f[3].mid=f[3].accepted; f[3].expected=2;
    f[4].name="tangential"; f[4].mid.referenceDisplacementAndMode.x=.01f;
    f[5].name="tilted"; f[5].accepted=node(.0005f,0,.0005f); f[5].mid=f[5].accepted;
    f[5].plane={1,0,1,0}; f[5].mid.referenceDisplacementAndMode.x=.0001f;
    f[6].name="large-origin"; f[6].accepted=node(0,0,1024.f); f[6].mid=f[6].accepted;
    f[6].accepted.referenceDisplacementAndMode.z=.01f; f[6].mid.referenceDisplacementAndMode.z=.011f;
    f[6].plane={0,0,1,1024.f};
    f[7].name="exact-floor"; f[7].accepted=node(0,0,1e-5f); f[7].mid=f[7].accepted; f[7].expected=2;
    f[8].name="nonfinite"; f[8].mid.velocityAndInverseMass.x=std::numeric_limits<float>::infinity(); f[8].expected=1;
    f[9].name="overflow"; f[9].mid.velocityAndInverseMass.x=std::numeric_limits<float>::max(); f[9].expected=1;
    f[10].name="metadata"; f[10].mid.restAndFixed.x=1; f[10].expected=1;
    f[11].name="zero-normal"; f[11].plane={0,0,0,0}; f[11].expected=1;
    std::uint32_t count=1;
    for(const auto& test:f) {
        NMFEMNodeStateGPU sentinel; std::memset(&sentinel,0xa5,sizeof(sentinel));
        std::uint32_t missing=0xffffffffu;
        auto buffer=[&](const void* source, size_t size) {
            id<MTLBuffer> b=[device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared];
            require(b != nil,"allocation failed");
            return b;
        };
        id<MTLBuffer> accepted=buffer(&test.accepted,sizeof(test.accepted));
        id<MTLBuffer> mid=buffer(&test.mid,sizeof(test.mid));
        id<MTLBuffer> end=buffer(&sentinel,sizeof(sentinel));
        id<MTLBuffer> status=buffer(&missing,sizeof(missing));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        require(encoder != nil,"encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&test.plane length:sizeof(test.plane) atIndex:0];
        [encoder setBytes:&test.clearance length:sizeof(test.clearance) atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBuffer:accepted offset:0 atIndex:3];
        [encoder setBuffer:mid offset:0 atIndex:4];
        [encoder setBuffer:end offset:0 atIndex:5];
        [encoder setBuffer:status offset:0 atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"command failed");
        const auto actual=*static_cast<const std::uint32_t*>(status.contents);
        require(actual==test.expected,std::string(test.name)+": unexpected result "+std::to_string(actual));
        require(std::memcmp(accepted.contents,&test.accepted,sizeof(test.accepted))==0,"accepted mutated");
        require(std::memcmp(mid.contents,&test.mid,sizeof(test.mid))==0,"midpoint mutated");
        if(actual!=0) require(std::memcmp(end.contents,&sentinel,sizeof(sentinel))==0,
                              std::string(test.name)+": rejected output modified");
        else {
            const auto& e=*static_cast<const NMFEMNodeStateGPU*>(end.contents);
            for(int k=0;k<3;++k) {
                const double u0=double(component(test.accepted.referenceDisplacementAndMode,k))+
                    double(component(test.accepted.referenceDisplacementLow,k));
                const double um=double(component(test.mid.referenceDisplacementAndMode,k))+
                    double(component(test.mid.referenceDisplacementLow,k));
                const double u1=double(component(e.referenceDisplacementAndMode,k))+
                    double(component(e.referenceDisplacementLow,k));
                const bool fixed=test.accepted.restAndFixed.w==1;
                const double expectedU=fixed ? u0 : 2*um-u0;
                const double expectedV=fixed ? double(component(test.accepted.velocityAndInverseMass,k)) :
                    2*double(component(test.mid.velocityAndInverseMass,k))-
                    double(component(test.accepted.velocityAndInverseMass,k));
                require(std::abs(u1-expectedU)<1e-10,std::string(test.name)+": displacement mismatch");
                require(std::abs(double(component(e.velocityAndInverseMass,k))-expectedV)<1e-7,
                        std::string(test.name)+": velocity mismatch");
            }
        }
        std::cout<<test.name<<" result="<<actual<<"\n";
    }
    std::cout<<"device="<<device.name.UTF8String<<" endpoint_cases=12 PASS\n";
    return 0;
 } catch(const std::exception& e) {
    std::cerr<<"endpoint gate: "<<e.what()<<"\n"; return 1;
 } }
}
