#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "support/fem_body_triangle_contact_math.hpp"
#include "metalrobo/engine_types.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace numi::matter::fem_body_triangle_contact;
namespace {
void require(bool ok,const char* why) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",why); std::exit(1); }
}
template<class T> id<MTLBuffer> makeBuffer(id<MTLDevice> d,const T* p) {
    return [d newBufferWithBytes:p length:sizeof(T)
                         options:MTLResourceStorageModeShared];
}
id<MTLComputePipelineState> makePipeline(id<MTLDevice> d,id<MTLLibrary> lib,
                                         NSString* name) {
    NSError* error=nil;
    id<MTLFunction> fn=[lib newFunctionWithName:name];
    require(fn!=nil,"sparse and dense-oracle functions exist");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:fn
        error:&error];
    if (!p) { std::fprintf(stderr,"pipeline: %s\n",
        error.localizedDescription.UTF8String); std::exit(1); }
    return p;
}
nm_float4 run(id<MTLDevice> d,id<MTLCommandQueue> q,
    id<MTLComputePipelineState> p,id<MTLBuffer> db,id<MTLBuffer> sb,
    id<MTLBuffer> nb,id<MTLBuffer> pb,id<MTLBuffer> rb,id<MTLBuffer> tb,
    nm_float4 initial) {
    id<MTLBuffer> out=[d newBufferWithBytes:&initial length:sizeof(initial)
        options:MTLResourceStorageModeShared];
    require(out!=nil,"output buffer");
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    require(cb && e,"command encoder");
    [e setComputePipelineState:p];
    [e setBuffer:db offset:0 atIndex:0]; [e setBuffer:sb offset:0 atIndex:1];
    [e setBuffer:nb offset:0 atIndex:2]; [e setBuffer:pb offset:0 atIndex:3];
    [e setBuffer:out offset:0 atIndex:4]; [e setBuffer:rb offset:0 atIndex:5];
    [e setBuffer:tb offset:0 atIndex:6];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.status==MTLCommandBufferStatusCompleted,"dispatch completes");
    const auto* values=static_cast<const nm_float4*>(out.contents);
    require(std::isfinite(values[0].x)&&std::isfinite(values[0].y)&&
        std::isfinite(values[0].z)&&values[0].w==initial.w,
        "finite force and unchanged w lane");
    return values[0];
}
}
int main(int argc,const char* argv[]) {
 @autoreleasepool {
    require(argc==3,"usage: arithmetic_contraction_coupon METALLIB match|mismatch");
    const bool expectMatch=std::strcmp(argv[2],"match")==0;
    const bool expectMismatch=std::strcmp(argv[2],"mismatch")==0;
    require(expectMatch||expectMismatch,"expectation is match or mismatch");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil,"Metal device");
    NSError* error=nil;
    id<MTLLibrary> lib=[device newLibraryWithURL:
        [NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!lib) { std::fprintf(stderr,"library: %s\n",
        error.localizedDescription.UTF8String); return 1; }
    auto sparse=makePipeline(device,lib,
        @"numi_matter_metal::nm_numi_human_assemble_fem_body_contact_loads");
    auto dense=makePipeline(device,lib,
        @"numi_matter_metal::nm_numi_human_assemble_fem_body_contact_loads_dense_oracle");

    NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
    dispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
    dispatch.environmentCount=1u; dispatch.femNodeCount=1u;
    dispatch.bodyPoseStride=2u; dispatch.articulationFirstBody=0u;
    dispatch.femBodyContactSampleCount=1u;
    dispatch.femBodyContactCandidateTriangleCount=0u;
    NMNumiHumanFEMBodyContactSampleGPU sample{};
    sample.slaveNode=0u; sample.bodyIndex=1u;
    sample.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE|
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
    sample.bodyLocalPointAndArea={0.0f,0.0f,0.0f,1.0f};
    sample.bodyLocalNormalAndReferenceSeparation={0.0f,0.0f,1.0f,0.0f};
    sample.stiffnessAndNormalStrainPerPressure={1000.0f,1.0e-6f,0.0f,0.0f};
    NMFEMNodeStateGPU node{};
    node.positionAndMass={0.4f,0.4f,-0.05f,1.0f};
    node.velocityAndInverseMass={0.0f,0.0f,0.0f,1.0f};
    node.restAndFixed={0.0f,0.0f,0.0f,0.0f};
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU range{};
    NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU dummyTriangle{};
    std::array<MRArticulatedBodyPoseGPU,2> poses{};
    for (auto& pose:poses) {
        pose.position={0.0f,0.0f,0.0f,0.0f};
        pose.orientation={0.0f,0.0f,0.0f,1.0f};
    }
    const nm_float4 initial={145.375f,-67.125f,9.75f,77.0f};
    auto db=makeBuffer(device,&dispatch); auto sb=makeBuffer(device,&sample);
    auto nb=makeBuffer(device,&node);
    auto pb=[device newBufferWithBytes:poses.data() length:sizeof(poses)
        options:MTLResourceStorageModeShared];
    auto rb=makeBuffer(device,&range); auto tb=makeBuffer(device,&dummyTriangle);
    auto queue=[device newCommandQueue];
    require(db&&sb&&nb&&pb&&rb&&tb&&queue,"input buffers and command queue");
    const auto s=run(device,queue,sparse,db,sb,nb,pb,rb,tb,initial);
    const auto o=run(device,queue,dense,db,sb,nb,pb,rb,tb,initial);
    const bool exact=std::memcmp(&s,&o,sizeof(s))==0;
    const bool changed=s.x!=initial.x||s.y!=initial.y||s.z!=initial.z;
    require(changed,"fixture produces nonzero contact force");
    std::printf("incoming=(%.9g,%.9g,%.9g,%.9g) sparse=(%.9g,%.9g,%.9g,%.9g) dense=(%.9g,%.9g,%.9g,%.9g) xbits=%08x/%08x exact=%s\n",
        initial.x,initial.y,initial.z,initial.w,s.x,s.y,s.z,s.w,o.x,o.y,o.z,o.w,
        std::bit_cast<std::uint32_t>(s.x),std::bit_cast<std::uint32_t>(o.x),
        exact?"true":"false");
    if (expectMatch) {
        require(exact,"patched sparse accumulation is bitwise equal to dense oracle");
        std::puts("PASS: arithmetic-contraction regression");
    } else {
        require(!exact,"original direct += is distinguished from dense oracle");
        std::puts("PASS: legacy direct-add mismatch reproduced");
    }
 }
 return 0;
}

