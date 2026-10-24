#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/fem_body_contact_static_bvh.hpp"
#include "support/fem_body_triangle_contact_math.hpp"
#include "metalrobo/engine_types.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace numi::matter::fem_body_triangle_contact;
using namespace numi::matter::fem_body_contact_bvh;

namespace {
void require(bool value,const char* message) {
    if (!value) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
}

int main(int argc,const char* argv[]) {
    @autoreleasepool {
        require(argc==2,"expected metallib path");
        constexpr std::uint32_t kSlaves=4808u,kBodies=86u;
        constexpr std::uint32_t kNodes=7233u;
        constexpr std::uint64_t kExpected=std::uint64_t(kSlaves)*kBodies;
        constexpr float kArea=1.0e-4f,kStiffness=1000.0f,kStrain=1.0e-7f;
        const std::array<Vec3,3> local{{{-0.1,-0.1,0.0},{0.1,-0.1,0.0},{0.0,0.1,0.0}}};

        std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles;
        triangles.reserve(kBodies);
        for (std::uint32_t body=0;body<kBodies;++body) {
            NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU t{};
            t.bodyLocalVertex0={-0.1f,-0.1f,0.0f,0.0f};
            t.bodyLocalVertex1={0.1f,-0.1f,0.0f,0.0f};
            t.bodyLocalVertex2={0.0f,0.1f,0.0f,0.0f};
            t.authoredOutwardNormal={0.0f,0.0f,1.0f,0.0f};
            t.bodyIndex=body;t.boneStableId=1000u+body;
            t.triangleStableId=2000u+body;
            triangles.push_back(t);
        }
        std::vector<NMNumiHumanFEMBodyContactSampleGPU> templates;
        templates.reserve(kSlaves);
        for (std::uint32_t slave=0;slave<kSlaves;++slave)
            templates.push_back(makeSample(slave,0u,local[0],local[1],local[2],
                                           kArea,kStiffness,kStrain));
        PreparationOptions preparation{};
        preparation.expandAllTriangleSamplesAcrossBodies=true;
        const auto asset=prepareStaticCompleteBodyContactAsset(
            templates,triangles,kBodies,0.05f,preparation);
        require(asset.succeeded(),"all-body complete-BVH sample expansion");
        require(asset.samples.size()==kExpected,"414k pair capacity");
        require(asset.ranges.size()==kExpected,"per-pair range capacity");
        require(asset.triangles.size()==kBodies,"full synthetic body inventory");
        require(asset.nodes.size()==kBodies,"one-leaf complete tree per body");
        require(validCompleteStaticBVHSpans(asset.samples,asset.ranges,
                    asset.triangles,asset.nodes),"expanded full-bank spans validate");
        for (std::uint32_t s=0;s<kSlaves;++s) {
            for (std::uint32_t b=0;b<kBodies;++b) {
                const std::size_t i=std::size_t(s)*kBodies+b;
                require(asset.samples[i].slaveNode==s && asset.samples[i].bodyIndex==b,
                        "deterministic slave-major/body-major rows");
            }
        }

        NSError* error=nil;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        require(device!=nil,"Metal device");
        id<MTLLibrary> library=[device newLibraryWithURL:
            [NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!library) {
            std::fprintf(stderr,"library: %s\n",error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function=[library newFunctionWithName:
            @"nm_numi_human_fem_all_body_moving_capacity_coupon"];
        require(function!=nil,"414k all-body coupon kernel");
        id<MTLComputePipelineState> pipeline=[device
            newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::fprintf(stderr,"pipeline: %s\n",error.localizedDescription.UTF8String);
            return 1;
        }

        NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
        dispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        dispatch.environmentCount=1u;dispatch.femNodeCount=kNodes;
        dispatch.bodyPoseStride=kBodies;dispatch.articulationFirstBody=0u;
        dispatch.femBodyContactSampleCount=static_cast<nm_u32>(kExpected);
        dispatch.femBodyContactCandidateTriangleCount=kBodies;
        std::vector<NMFEMNodeStateGPU> nodeStates(kNodes);
        for (auto& node:nodeStates) {
            node.positionAndMass={0.02f,0.01f,0.02f,1.0f};
            node.velocityAndInverseMass={0.0f,0.0f,0.0f,1.0f};
            node.restAndFixed={0.0f,0.0f,0.0f,0.0f};
        }
        std::vector<MRArticulatedBodyPoseGPU> poses(kBodies);
        for (std::uint32_t b=0;b<kBodies;++b) {
            poses[b].position={0.0f,0.0f,0.0f,0.0f};
            poses[b].orientation={0.0f,0.0f,0.0f,1.0f};
        }
        // All body pairs start outside the authored +Z surface. Body 85 then
        // moves into the contact side; this body was not the original template
        // owner and represents a newly approaching master.
        poses[85].position={0.0f,0.0f,0.0f,0.0f};
        const NSUInteger outBytes=static_cast<NSUInteger>(kExpected*8u*sizeof(nm_float4));
        auto makeBuffer=[&](const void* data,NSUInteger length) {
            return [device newBufferWithBytes:data length:length
                options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> dispatchBuffer=makeBuffer(&dispatch,sizeof(dispatch));
        id<MTLBuffer> sampleBuffer=makeBuffer(asset.samples.data(),
            asset.samples.size()*sizeof(asset.samples[0]));
        id<MTLBuffer> nodeBuffer=makeBuffer(nodeStates.data(),
            nodeStates.size()*sizeof(nodeStates[0]));
        id<MTLBuffer> poseBuffer=makeBuffer(poses.data(),
            poses.size()*sizeof(poses[0]));
        id<MTLBuffer> rangeBuffer=makeBuffer(asset.ranges.data(),
            asset.ranges.size()*sizeof(asset.ranges[0]));
        id<MTLBuffer> triangleBuffer=makeBuffer(asset.triangles.data(),
            asset.triangles.size()*sizeof(asset.triangles[0]));
        id<MTLBuffer> bvhBuffer=makeBuffer(asset.nodes.data(),
            asset.nodes.size()*sizeof(asset.nodes[0]));
        id<MTLBuffer> outputBuffer=[device newBufferWithLength:outBytes
            options:MTLResourceStorageModeShared];
        require(dispatchBuffer&&sampleBuffer&&nodeBuffer&&poseBuffer&&rangeBuffer&&
                triangleBuffer&&bvhBuffer&&outputBuffer,"414k Metal buffers");
        auto run=[&]() {
            id<MTLCommandQueue> queue=[device newCommandQueue];
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:sampleBuffer offset:0 atIndex:1];
            [encoder setBuffer:nodeBuffer offset:0 atIndex:2];
            [encoder setBuffer:poseBuffer offset:0 atIndex:3];
            [encoder setBuffer:rangeBuffer offset:0 atIndex:4];
            [encoder setBuffer:triangleBuffer offset:0 atIndex:5];
            [encoder setBuffer:bvhBuffer offset:0 atIndex:6];
            [encoder setBuffer:outputBuffer offset:0 atIndex:7];
            [encoder dispatchThreads:MTLSizeMake(kExpected,1,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,
                    "414k GPU command completed");
        };
        const auto* out=static_cast<const nm_float4*>(outputBuffer.contents);
        run();
        std::uint64_t nonfinite0=0u,nonzero0=0u;
        for (std::uint64_t i=0;i<kExpected;++i) {
            for (std::uint32_t k=0;k<8u;++k) {
                const auto v=out[i*8u+k];
                if (!std::isfinite(v.x)||!std::isfinite(v.y)||
                    !std::isfinite(v.z)||!std::isfinite(v.w)) ++nonfinite0;
            }
            if (out[i*8u].x!=0.0f||out[i*8u].y!=0.0f||out[i*8u].z!=0.0f)
                ++nonzero0;
        }
        require(nonfinite0==0u,"finite force/reaction/tangent in initial state");
        require(nonzero0==0u,"no force before any master approaches");

        poses[85].position.z=0.04f;
        std::memcpy(poseBuffer.contents,poses.data(),poses.size()*sizeof(poses[0]));
        run();
        std::uint64_t nonfinite1=0u,nonzeroOtherBody=0u,contactRows=0u;
        double forceNormSquared=0.0,reactionError=0.0,tangentAsymmetry=0.0;
        for (std::uint64_t i=0;i<kExpected;++i) {
            const auto force=out[i*8u],reaction=out[i*8u+1u];
            for (std::uint32_t k=0;k<8u;++k) {
                const auto v=out[i*8u+k];
                if (!std::isfinite(v.x)||!std::isfinite(v.y)||
                    !std::isfinite(v.z)||!std::isfinite(v.w)) ++nonfinite1;
            }
            const std::uint32_t body=asset.samples[i].bodyIndex;
            const double f2=double(force.x)*force.x+double(force.y)*force.y+
                            double(force.z)*force.z;
            forceNormSquared+=f2;
            reactionError=std::max(reactionError,std::sqrt(
                std::pow(double(force.x)+reaction.x,2)+
                std::pow(double(force.y)+reaction.y,2)+
                std::pow(double(force.z)+reaction.z,2)));
            const auto tx=out[i*8u+3u],ty=out[i*8u+4u],tz=out[i*8u+5u];
            tangentAsymmetry=std::max(tangentAsymmetry,std::max({
                std::abs(double(tx.y)-ty.x),std::abs(double(tx.z)-tz.x),
                std::abs(double(ty.z)-tz.y)}));
            if (body==85u && force.w>0.0f) ++contactRows;
            if (body!=85u && f2>0.0) ++nonzeroOtherBody;
        }
        require(nonfinite1==0u,"finite force/reaction/tangent after master motion");
        require(contactRows==kSlaves,"every slave sees the newly approaching body");
        require(nonzeroOtherBody==0u,"no spurious force from other master banks");
        require(forceNormSquared>0.0,"approaching master generates nonzero FEM force");
        require(reactionError<1.0e-7,"slave/body reaction is equal and opposite");
        require(tangentAsymmetry<1.0e-6,"contact tangent is symmetric");
        const auto witness=std::uint64_t(123u)*kBodies+85u;
        std::printf("PASS samples=%llu nodes=%u bodies=%u triangles=%zu initial_nonfinite=%llu initial_nonzero=%llu moved_nonfinite=%llu contact_rows=%llu other_body_nonzero=%llu force_l2=%.9g max_reaction_error=%.9g max_tangent_asymmetry=%.9g witness_force=(%.9g,%.9g,%.9g) witness_tangent_zz=%.9g\n",
            static_cast<unsigned long long>(kExpected),kNodes,kBodies,asset.triangles.size(),
            static_cast<unsigned long long>(nonfinite0),static_cast<unsigned long long>(nonzero0),
            static_cast<unsigned long long>(nonfinite1),static_cast<unsigned long long>(contactRows),
            static_cast<unsigned long long>(nonzeroOtherBody),std::sqrt(forceNormSquared),
            reactionError,tangentAsymmetry,out[witness*8u].x,out[witness*8u].y,
            out[witness*8u].z,out[witness*8u+5u].z);
        return 0;
    }
}
