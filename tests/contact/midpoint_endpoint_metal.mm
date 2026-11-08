#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <array>
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

    // Mixed outcomes in one dispatch; FP64 checks evaluate the actual output
    // limbs, independently of the shader's interval bound.
    constexpr std::size_t batchSize=257;
    std::array<NMFEMNodeStateGPU,batchSize> aa{},mm{},ee{},sentinels{};
    std::array<std::uint32_t,batchSize> rr{};
    for(std::size_t i=0;i<batchSize;++i) {
        const float x=float(int(i%13)-6)*.0125f;
        const float desired=1e-5f+(int(i%9)-4)*.0001f;
        const float z=(desired-.6f*x)/.8f;
        aa[i]=node(x,0,z); mm[i]=aa[i];
        mm[i].referenceDisplacementAndMode.z=float(int(i%7)-3)*.00008f;
        mm[i].velocityAndInverseMass.x=float(int(i%11)-5)*.02f;
        std::memset(&sentinels[i],0xa5,sizeof(sentinels[i])); ee[i]=sentinels[i];
        rr[i]=0xffffffffu;
    }
    const auto expectedAA=aa,expectedMM=mm;
    id<MTLBuffer> ab=[device newBufferWithBytes:aa.data() length:sizeof(aa) options:MTLResourceStorageModeShared];
    id<MTLBuffer> mb=[device newBufferWithBytes:mm.data() length:sizeof(mm) options:MTLResourceStorageModeShared];
    id<MTLBuffer> eb=[device newBufferWithBytes:ee.data() length:sizeof(ee) options:MTLResourceStorageModeShared];
    id<MTLBuffer> rb=[device newBufferWithBytes:rr.data() length:sizeof(rr) options:MTLResourceStorageModeShared];
    require(ab && mb && eb && rb,"batch allocation failed");
    const nm_float4 plane{.6f,0,.8f,0}; const float floor=1e-5f;
    const std::uint32_t batchCount=batchSize;
    id<MTLCommandBuffer> cmd=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    require(enc != nil,"batch encoder unavailable");
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&plane length:sizeof(plane) atIndex:0];
    [enc setBytes:&floor length:sizeof(floor) atIndex:1];
    [enc setBytes:&batchCount length:sizeof(batchCount) atIndex:2];
    [enc setBuffer:ab offset:0 atIndex:3]; [enc setBuffer:mb offset:0 atIndex:4];
    [enc setBuffer:eb offset:0 atIndex:5]; [enc setBuffer:rb offset:0 atIndex:6];
    [enc dispatchThreads:MTLSizeMake(batchSize,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    require(cmd.status==MTLCommandBufferStatusCompleted,"batch command failed");
    require(std::memcmp(ab.contents,expectedAA.data(),sizeof(aa))==0,"batch accepted mutated");
    require(std::memcmp(mb.contents,expectedMM.data(),sizeof(mm))==0,"batch midpoint mutated");
    const auto* ends=static_cast<const NMFEMNodeStateGPU*>(eb.contents);
    const auto* codes=static_cast<const std::uint32_t*>(rb.contents);
    unsigned passed=0,rejected=0;
    for(std::size_t i=0;i<batchSize;++i) {
        require(codes[i]==0 || codes[i]==2,"batch invalid result");
        if(codes[i]==2) {
            ++rejected;
            require(std::memcmp(&ends[i],&sentinels[i],sizeof(ends[i]))==0,"batch rejected output modified");
            continue;
        }
        ++passed;
        const auto gap=[&](const NMFEMNodeStateGPU& n) {
            const double x=double(n.restAndFixed.x)+n.referenceDisplacementAndMode.x+n.referenceDisplacementLow.x;
            const double y=double(n.restAndFixed.y)+n.referenceDisplacementAndMode.y+n.referenceDisplacementLow.y;
            const double z=double(n.restAndFixed.z)+n.referenceDisplacementAndMode.z+n.referenceDisplacementLow.z;
            return double(plane.x)*x+double(plane.y)*y+double(plane.z)*z-double(plane.w);
        };
        const double threshold=double(floor)*std::sqrt(double(plane.x)*plane.x+
            double(plane.y)*plane.y+double(plane.z)*plane.z);
        require(gap(aa[i])>threshold && gap(ends[i])>threshold,"batch falsely certified crossing");
        for(int k=0;k<3;++k) {
            const double old=double(component(aa[i].referenceDisplacementAndMode,k))+
                component(aa[i].referenceDisplacementLow,k);
            const double mid=double(component(mm[i].referenceDisplacementAndMode,k))+
                component(mm[i].referenceDisplacementLow,k);
            const double end=double(component(ends[i].referenceDisplacementAndMode,k))+
                component(ends[i].referenceDisplacementLow,k);
            const double velocity=2*double(component(mm[i].velocityAndInverseMass,k))-
                component(aa[i].velocityAndInverseMass,k);
            require(std::abs(end-(2*mid-old))<1e-10,"batch displacement mismatch");
            require(std::abs(double(component(ends[i].velocityAndInverseMass,k))-velocity)<1e-7,
                    "batch velocity mismatch");
        }
    }
    require(passed>0 && rejected>0,"batch did not exercise both outcomes");
    std::cout<<"mixed_batch certified="<<passed<<" rejected="<<rejected<<"\n";
    std::cout<<"device="<<device.name.UTF8String<<" endpoint_cases=12 PASS\n";
    return 0;
 } catch(const std::exception& e) {
    std::cerr<<"endpoint gate: "<<e.what()<<"\n"; return 1;
 } }
}
