#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "support/fem_body_triangle_contact_math.hpp"
#include "metalrobo/engine_types.h"
#include "numi/matter/fem_body_contact_sample_grouping.hpp"
#include "numi/matter/fem_body_contact_static_bvh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <limits>
#include <tuple>

using namespace numi::matter::fem_body_triangle_contact;

namespace {
void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
nm_float4 f4(float x,float y,float z,float w) { return {x,y,z,w}; }
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        const bool staticBVHOnly=argc==3 &&
            std::strcmp(argv[2],"--static-bvh-only")==0;
        require(argc==2 || staticBVHOnly,
                "expected isolated metallib path [--static-bvh-only]");
        NSError* error=nil;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        require(device!=nil,"Metal device");
        NSURL* url=[NSURL fileURLWithPath:@(argv[1])];
        id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
        if (library==nil) {
            std::fprintf(stderr,"Metal library: %s\n",
                error.localizedDescription.UTF8String);
            return 1;
        }
        require([library newFunctionWithName:
            @"numi_matter_metal::nm_numi_human_contact_triangle_abi17_marker"] != nil,
            "ABI 17 marker is exported by the matched metallib");
        require([library newFunctionWithName:
            @"numi_matter_metal::nm_numi_human_contact_triangle_abi16_marker"] == nil,
            "stale ABI 16 host rejects the ABI 17 metallib marker");
        id<MTLFunction> function=[library
            newFunctionWithName:@"nm_numi_human_fem_body_triangle_contact_coupon"];
        require(function!=nil,"triangle coupon kernel");
        id<MTLComputePipelineState> pipeline=
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline==nil) {
            std::fprintf(stderr,"pipeline: %s\n",
                error.localizedDescription.UTF8String);
            return 1;
        }

        constexpr std::size_t count=9u;
        constexpr float area=0.25f, legacyGap=0.10f;
        constexpr float k=1000.0f, strain=1.0e-6f;
        const std::array<Vec3,3> triangle{{{0,0,0},{2,0,0},{0,2,0}}};
        const std::array<double,4> rotatedQ{{0.0,0.0,0.7071067811865476,
                                             0.7071067811865476}};
        const Vec3 rotatedOrigin{0.3,-0.1,0.2};
        auto rotatedPoint=[&](Vec3 v) {
            return add(rotatedOrigin,quaternionRotate(rotatedQ,v));
        };
        const std::array<Vec3,3> rotatedTriangle{{
            rotatedPoint(triangle[0]),rotatedPoint(triangle[1]),
            rotatedPoint(triangle[2])}};
        const std::array<Vec3,count> points{{
            {0.4,0.4,0.3},             // zero force outside
            {0.4,0.4,-0.05},           // face penetration
            rotatedPoint({0.7,-0.03,-0.04}), // rotated closest edge
            {-0.03,-0.04,-0.02},       // closest vertex
            {0.4,0.4,-0.02},           // authored-winding outward force
            {0.2,0.2,0.05},            // legacy tangent-plane behavior
            {0.05,0,0},                // legacy elastic attachment behavior
            {0.4,0.4,-0.02},           // illegal flag combination
            {0.4,0.4,-0.02},           // non-unit body quaternion
        }};
        std::array<NMNumiHumanFEMBodyContactSampleGPU,count> samples{};
        std::array<NMFEMNodeStateGPU,count> nodes{};
        std::array<MRArticulatedBodyPoseGPU,count> poses{};
        for (std::size_t i=0;i<5u;++i) {
            samples[i]=makeSample(static_cast<std::uint32_t>(i),1u,
                triangle[0],triangle[1],triangle[2],area,k,strain);
        }
        samples[7]=samples[4];
        samples[7].slaveNode=7u;
        samples[7].flags |= NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
        samples[8]=samples[4];
        samples[8].slaveNode=8u;
        samples[5].slaveNode=5u;
        samples[5].bodyIndex=1u;
        samples[5].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE;
        samples[5].bodyLocalPointAndArea=f4(0,0,0,1);
        samples[5].bodyLocalNormalAndReferenceSeparation=f4(0,0,1,legacyGap);
        samples[5].stiffnessAndNormalStrainPerPressure=f4(k,strain,0,0);
        samples[6].slaveNode=6u;
        samples[6].bodyIndex=1u;
        samples[6].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
            NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
        samples[6].bodyLocalPointAndArea=f4(0,0,0,1);
        samples[6].bodyLocalNormalAndReferenceSeparation=f4(0,0,1,0);
        samples[6].stiffnessAndNormalStrainPerPressure=f4(k,strain,0,0);
        for (std::size_t i=0;i<count;++i) {
            nodes[i].positionAndMass=f4(float(points[i].x),float(points[i].y),
                                        float(points[i].z),1.0f);
            nodes[i].velocityAndInverseMass=f4(0,0,0,1);
            nodes[i].restAndFixed=f4(0,0,0,0);
            poses[i].position={0,0,0,0};
            poses[i].orientation={0,0,0,1};
        }
        poses[2].position={float(rotatedOrigin.x),float(rotatedOrigin.y),
                            float(rotatedOrigin.z),0.0f};
        poses[2].orientation={float(rotatedQ[0]),float(rotatedQ[1]),
                              float(rotatedQ[2]),float(rotatedQ[3])};
        poses[8].orientation={0,0,0,2};
        const NSUInteger inputBytes=count*sizeof(samples[0]);
        const NSUInteger nodeBytes=count*sizeof(nodes[0]);
        const NSUInteger poseBytes=count*sizeof(poses[0]);
        const NSUInteger outputBytes=count*8u*sizeof(nm_float4);
        id<MTLBuffer> sampleBuffer=[device newBufferWithBytes:samples.data()
            length:inputBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> nodeBuffer=[device newBufferWithBytes:nodes.data()
            length:nodeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> poseBuffer=[device newBufferWithBytes:poses.data()
            length:poseBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer=[device newBufferWithLength:outputBytes
            options:MTLResourceStorageModeShared];
        require(sampleBuffer && nodeBuffer && poseBuffer && outputBuffer,
                "Metal buffers");
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:sampleBuffer offset:0 atIndex:0];
        [encoder setBuffer:nodeBuffer offset:0 atIndex:1];
        [encoder setBuffer:poseBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,
                "Metal coupon command completed");
        const auto* out=static_cast<const nm_float4*>(outputBuffer.contents);
        for (std::size_t i=0;i<7u;++i)
            require(std::isfinite(out[i*8u].x) &&
                    std::isfinite(out[i*8u].y) &&
                    std::isfinite(out[i*8u].z) &&
                    std::isfinite(out[i*8u].w),
                    "finite valid owner output");

        require(out[0].x==0.0f && out[0].y==0.0f &&
                out[0].z==0.0f && out[0].w==0.0f,
                "triangle sample is exactly force-free outside");
        require(std::isnan(out[7u*8u].x) && std::isnan(out[8u*8u].x),
                "illegal flag combinations and non-unit poses are rejected");
        for (std::size_t i=1;i<5u;++i) {
            const bool rotated=i==2u;
            const auto expected=evaluate(
                points[i],rotated?rotatedOrigin:Vec3{0,0,0},
                rotated?rotatedTriangle:triangle,area,k,strain);
            const double maxDifference=std::max({
                std::abs(double(out[i*8u+0u].x)-expected.slaveForce.x),
                std::abs(double(out[i*8u+0u].y)-expected.slaveForce.y),
                std::abs(double(out[i*8u+0u].z)-expected.slaveForce.z)});
            if (!(maxDifference<1.0e-3)) {
                std::fprintf(stderr,"legacy triangle mismatch row=%zu got=(%.9g,%.9g,%.9g) expected=(%.17g,%.17g,%.17g) diff=%.9g rotated=%d point=(%.9g,%.9g,%.9g) closure=%.9g body=(%.9g,%.9g,%.9g) normal=(%.9g,%.9g,%.9g)\n",
                    i,out[i*8u].x,out[i*8u].y,out[i*8u].z,
                    expected.slaveForce.x,expected.slaveForce.y,expected.slaveForce.z,
                    maxDifference,int(rotated),points[i].x,points[i].y,points[i].z,
                    out[i*8u].w,out[i*8u+2u].x,out[i*8u+2u].y,out[i*8u+2u].z,
                    out[i*8u+7u].x,out[i*8u+7u].y,out[i*8u+7u].z);
            }
            require(maxDifference<1.0e-3,"Metal force matches CPU triangle query");
            for (std::size_t col=0;col<3u;++col) {
                const nm_float4 got=out[i*8u+3u+col];
                for (std::size_t row=0;row<3u;++row) {
                    const float g=row==0?got.x:(row==1?got.y:got.z);
                    require(std::abs(double(g)-expected.positiveStiffness[row][col])<2.0e-3,
                            "Metal feature tangent matches CPU oracle");
                }
            }
            require(std::abs(out[i*8u+1u].x+out[i*8u+0u].x)<1.0e-5 &&
                    std::abs(out[i*8u+1u].y+out[i*8u+0u].y)<1.0e-5 &&
                    std::abs(out[i*8u+1u].z+out[i*8u+0u].z)<1.0e-5,
                    "Metal body reaction balances the FEM node force");
        }
        // Legacy plane and elastic-attachment resultants remain on the exact
        // pre-existing branches; this coupon guards accidental flag overlap.
        require(std::abs(out[5u*8u+0u].z-49.9999f)<2.0e-3f &&
                out[5u*8u+0u].x==0.0f && out[5u*8u+0u].y==0.0f,
                "legacy plane contact branch unchanged");
        require(std::abs(out[6u*8u+0u].x+49.9999f)<2.0e-3f &&
                out[6u*8u+0u].y==0.0f && out[6u*8u+0u].z==0.0f,
                "legacy elastic attachment branch unchanged");

        // Exercise the actual owner load, body-wrench, and audit kernels on
        // one valid triangle sample, using the same single Metal command path.
        NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
        dispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        dispatch.environmentCount=1u;
        dispatch.femNodeCount=static_cast<nm_u32>(count);
        dispatch.bodyPoseStride=static_cast<nm_u32>(count);
        dispatch.articulationFirstBody=0u;
        dispatch.femBodyContactSampleCount=1u;
        dispatch.femBodyContactCandidateTriangleCount=2u;
        const auto ownerSample=samples[1];
        const std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,1>
            ownerRanges{{{0u,2u,0.05f,1u,{0.4f,0.4f,-0.05f,0.01f}}}};
        std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,2>
            ownerCandidates{};
        for (std::uint32_t i=0;i<ownerCandidates.size();++i) {
            const float z=i==0u ? 0.0f : -0.04f;
            auto& candidate=ownerCandidates[i];
            candidate.bodyLocalVertex0={0,0,z,0};
            candidate.bodyLocalVertex1={2,0,z,0};
            candidate.bodyLocalVertex2={0,2,z,0};
            candidate.authoredOutwardNormal={0,0,1,0};
            candidate.bodyIndex=ownerSample.bodyIndex;
            candidate.boneStableId=10u+i;
            candidate.triangleStableId=20u+i;
        }
        std::array<nm_float4,count> ownerForces{};
        std::array<NMNumiHumanBodyWrenchGPU,count> ownerWrenches{};
        NMNumiHumanArticularContactAuditGPU ownerAudit{};
        auto dispatchBuffer=[&](const void* bytes, NSUInteger length) {
            return [device newBufferWithBytes:bytes length:length
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> dispatchBufferObject=dispatchBuffer(
            &dispatch,sizeof(dispatch));
        id<MTLBuffer> ownerSampleBuffer=dispatchBuffer(
            &ownerSample,sizeof(ownerSample));
        id<MTLBuffer> ownerForceBuffer=dispatchBuffer(
            ownerForces.data(),sizeof(ownerForces));
        id<MTLBuffer> ownerWrenchBuffer=dispatchBuffer(
            ownerWrenches.data(),sizeof(ownerWrenches));
        id<MTLBuffer> ownerAuditBuffer=dispatchBuffer(
            &ownerAudit,sizeof(ownerAudit));
        id<MTLBuffer> ownerRangeBuffer=dispatchBuffer(
            ownerRanges.data(),sizeof(ownerRanges));
        id<MTLBuffer> ownerCandidateBuffer=dispatchBuffer(
            ownerCandidates.data(),sizeof(ownerCandidates));
        require(dispatchBufferObject && ownerSampleBuffer &&
                ownerForceBuffer && ownerWrenchBuffer && ownerAuditBuffer &&
                ownerRangeBuffer && ownerCandidateBuffer,
                "actual owner kernel buffers");

        // Two nearby prepared triangles exercise winner switching; the second
        // query exceeds the certified radius and must fail closed.
        NMNumiHumanTendonFEMLoadDispatchGPU selectorDispatch=dispatch;
        selectorDispatch.femBodyContactSampleCount=4u;
        selectorDispatch.femBodyContactCandidateTriangleCount=6u;
        std::array<NMNumiHumanFEMBodyContactSampleGPU,4> selectorSamples{{
            ownerSample,ownerSample,ownerSample,samples[5]}};
        selectorSamples[1].slaveNode=2u;
        selectorSamples[2].slaveNode=3u;
        const nm_float4 selectorRest{0.4f,0.4f,-0.05f,0.01f};
        std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,4>
            selectorRanges{{{0u,2u,0.05f,1u,selectorRest},
                            {2u,2u,0.05f,1u,selectorRest},
                            {4u,2u,0.05f,1u,{0.4f,0.4f,-0.20f,0.01f}},
                            {6u,0u,0.0f,0u,{}}}};
        std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,6>
            selectorCandidates{{ownerCandidates[0],ownerCandidates[1],
                                ownerCandidates[0],ownerCandidates[1],
                                ownerCandidates[0],ownerCandidates[1]}};
        std::array<NMFEMNodeStateGPU,count> selectorNodes=nodes;
        // Row 2 moves tangentially beyond the declared envelope while still
        // lying on the same broad face and inside maximumDistanceMeters.
        selectorNodes[2].positionAndMass=f4(0.2f,0.4f,-0.05f,1.0f);
        // Row 3 starts inside its own envelope but beyond the prepared bone
        // distance, isolating the existing distance-bound rejection.
        selectorNodes[3].positionAndMass=f4(0.4f,0.4f,-0.20f,1.0f);
        auto selectorNodeBuffer=dispatchBuffer(
            selectorNodes.data(),sizeof(selectorNodes));
        auto selectorDispatchBuffer=dispatchBuffer(
            &selectorDispatch,sizeof(selectorDispatch));
        auto selectorSampleBuffer=dispatchBuffer(
            selectorSamples.data(),sizeof(selectorSamples));
        auto selectorRangeBuffer=dispatchBuffer(
            selectorRanges.data(),sizeof(selectorRanges));
        auto selectorCandidateBuffer=dispatchBuffer(
            selectorCandidates.data(),sizeof(selectorCandidates));
        const NMNumiHumanFEMBodyContactBVHNodeGPU emptySelectorBVHNode{};
        auto selectorBVHNodeBuffer=dispatchBuffer(
            &emptySelectorBVHNode,sizeof(emptySelectorBVHNode));
        std::array<nm_float4,8> selectorOutput{};
        auto selectorOutputBuffer=dispatchBuffer(
            selectorOutput.data(),sizeof(selectorOutput));
        id<MTLFunction> selectorFunction=[library
            newFunctionWithName:@"nm_numi_human_fem_body_contact_neighborhood_coupon"];
        require(selectorFunction!=nil,"bounded selector coupon kernel");
        NSError* selectorError=nil;
        id<MTLComputePipelineState> selectorPipeline=
            [device newComputePipelineStateWithFunction:selectorFunction
                                                  error:&selectorError];
        if (selectorPipeline==nil) {
            std::fprintf(stderr,"selector pipeline: %s\n",
                selectorError.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandBuffer> selectorCommand=[queue commandBuffer];
        id<MTLComputeCommandEncoder> selectorEncoder=
            [selectorCommand computeCommandEncoder];
        [selectorEncoder setComputePipelineState:selectorPipeline];
        [selectorEncoder setBuffer:selectorDispatchBuffer offset:0 atIndex:0];
        [selectorEncoder setBuffer:selectorSampleBuffer offset:0 atIndex:1];
        [selectorEncoder setBuffer:selectorNodeBuffer offset:0 atIndex:2];
        [selectorEncoder setBuffer:poseBuffer offset:0 atIndex:3];
        [selectorEncoder setBuffer:selectorOutputBuffer offset:0 atIndex:4];
        [selectorEncoder setBuffer:selectorRangeBuffer offset:0 atIndex:5];
        [selectorEncoder setBuffer:selectorCandidateBuffer offset:0 atIndex:6];
        [selectorEncoder setBuffer:selectorBVHNodeBuffer offset:0 atIndex:7];
        [selectorEncoder dispatchThreads:MTLSizeMake(4,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [selectorEncoder endEncoding];
        [selectorCommand commit];
        [selectorCommand waitUntilCompleted];
        require(selectorCommand.status==MTLCommandBufferStatusCompleted,
                "bounded selector coupon completes");
        const auto* selected=static_cast<const nm_float4*>(
            selectorOutputBuffer.contents);
        require(std::isfinite(selected[0].x) &&
                std::abs(selected[0].z-2.499975f)<2.0e-3f &&
                std::abs(selected[1].z+0.04f)<1.0e-6f,
                "closest triangle switches to nearer prepared face");
        require(std::isnan(selected[2].x) && std::isnan(selected[3].x),
                "tangential query outside motion envelope fails closed although it remains within triangle distance");
        require(std::isnan(selected[4].x) && std::isnan(selected[5].x),
                "query outside prepared distance range fails closed");
        require(std::isfinite(selected[6].x) &&
                std::abs(selected[6].z-49.9999f)<2.0e-3f &&
                selected[7].z==0.0f,
                "explicit zero-count range keeps the legacy plane branch");

        // Static complete BVH traversal is compared with a known exhaustive
        // nearest-face result on a multi-level tree, including non-first leaves.
        std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,5>
            staticTriangles{};
        const auto makeStaticTriangle=[](float x,float y,float z,
                                         std::uint32_t bone,
                                         std::uint32_t stable) {
            NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU t{};
            t.bodyLocalVertex0={x,y,z,0.0f};
            t.bodyLocalVertex1={x+2.0f,y,z,0.0f};
            t.bodyLocalVertex2={x,y+2.0f,z,0.0f};
            t.authoredOutwardNormal={0.0f,0.0f,1.0f,0.0f};
            t.bodyIndex=1u;t.boneStableId=bone;t.triangleStableId=stable;
            return t;
        };
        staticTriangles[0]=makeStaticTriangle(0.0f,0.0f,0.0f,11u,41u);
        staticTriangles[1]=makeStaticTriangle(0.0f,0.0f,-0.04f,11u,42u);
        staticTriangles[2]=makeStaticTriangle(2.0f,2.0f,-0.01f,12u,43u);
        staticTriangles[3]=makeStaticTriangle(-20.0f,-20.0f,0.0f,13u,44u);
        staticTriangles[4]=makeStaticTriangle(20.0f,20.0f,0.0f,14u,45u);
        const auto staticTree=numi::matter::fem_body_contact_bvh::
            buildStaticCompleteBodyBVH(staticTriangles,staticTriangles.size());
        require(staticTree.succeeded() && staticTree.bodies.size()==1u,
                "construct static complete body-local BVH for GPU query");
        std::array<NMNumiHumanFEMBodyContactSampleGPU,2> staticSamples{{
            makeSample(0u,1u,triangle[0],triangle[1],triangle[2],area,k,strain),
            makeSample(1u,1u,triangle[0],triangle[1],triangle[2],area,k,strain)}};
        std::array<NMFEMNodeStateGPU,2> staticSlaveNodes{};
        staticSlaveNodes[0].positionAndMass=f4(0.4f,0.4f,-0.05f,1.0f);
        staticSlaveNodes[1].positionAndMass=f4(2.2f,2.2f,-0.025f,1.0f);
        for (auto& n:staticSlaveNodes) {
            n.velocityAndInverseMass=f4(0,0,0,1);
            n.restAndFixed=f4(0,0,0,0);
        }
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>
            staticRanges;
        require(numi::matter::fem_body_contact_bvh::
                makeStaticCompleteBodyContactRanges(
                    staticSamples,staticTree.bodies,0.05f,staticRanges),
                "prepare complete per-sample BVH ranges");
        NMNumiHumanTendonFEMLoadDispatchGPU staticDispatch{};
        staticDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        staticDispatch.environmentCount=1u;
        staticDispatch.femNodeCount=2u;
        staticDispatch.bodyPoseStride=static_cast<nm_u32>(count);
        staticDispatch.articulationFirstBody=0u;
        staticDispatch.femBodyContactSampleCount=2u;
        staticDispatch.femBodyContactCandidateTriangleCount=
            static_cast<nm_u32>(staticTree.triangles.size());
        auto staticDispatchBuffer=dispatchBuffer(
            &staticDispatch,sizeof(staticDispatch));
        auto staticSampleBuffer=dispatchBuffer(
            staticSamples.data(),sizeof(staticSamples));
        auto staticNodeBuffer=dispatchBuffer(
            staticSlaveNodes.data(),sizeof(staticSlaveNodes));
        auto staticRangeBuffer=dispatchBuffer(
            staticRanges.data(),staticRanges.size()*sizeof(staticRanges[0]));
        auto staticTriangleBuffer=dispatchBuffer(
            staticTree.triangles.data(),
            staticTree.triangles.size()*sizeof(staticTree.triangles[0]));
        auto staticBVHNodeBuffer=dispatchBuffer(
            staticTree.nodes.data(),staticTree.nodes.size()*sizeof(staticTree.nodes[0]));
        std::array<nm_float4,4> staticOutput{};
        auto staticOutputBuffer=dispatchBuffer(staticOutput.data(),sizeof(staticOutput));
        require(staticDispatchBuffer && staticSampleBuffer && staticNodeBuffer &&
                staticRangeBuffer && staticTriangleBuffer && staticBVHNodeBuffer &&
                staticOutputBuffer,
                "static BVH GPU query buffers");
        id<MTLFunction> staticFunction=[library
            newFunctionWithName:@"nm_numi_human_static_complete_bvh_coupon"];
        require(staticFunction!=nil,"static complete BVH query kernel");
        NSError* staticError=nil;
        id<MTLComputePipelineState> staticPipeline=
            [device newComputePipelineStateWithFunction:staticFunction
                                                  error:&staticError];
        if (staticPipeline==nil) {
            std::fprintf(stderr,"static BVH pipeline: %s\n",
                staticError.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandBuffer> staticCommand=[queue commandBuffer];
        id<MTLComputeCommandEncoder> staticEncoder=
            [staticCommand computeCommandEncoder];
        [staticEncoder setComputePipelineState:staticPipeline];
        [staticEncoder setBuffer:staticDispatchBuffer offset:0 atIndex:0];
        [staticEncoder setBuffer:staticSampleBuffer offset:0 atIndex:1];
        [staticEncoder setBuffer:staticNodeBuffer offset:0 atIndex:2];
        [staticEncoder setBuffer:poseBuffer offset:0 atIndex:3];
        [staticEncoder setBuffer:staticOutputBuffer offset:0 atIndex:4];
        [staticEncoder setBuffer:staticRangeBuffer offset:0 atIndex:5];
        [staticEncoder setBuffer:staticTriangleBuffer offset:0 atIndex:6];
        [staticEncoder setBuffer:staticBVHNodeBuffer offset:0 atIndex:7];
        [staticEncoder dispatchThreads:MTLSizeMake(2u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [staticEncoder endEncoding];
        [staticCommand commit];
        [staticCommand waitUntilCompleted];
        require(staticCommand.status==MTLCommandBufferStatusCompleted,
                "static complete BVH query command completed");
        const auto* staticResult=static_cast<const nm_float4*>(
            staticOutputBuffer.contents);
        require(std::isfinite(staticResult[0].x) &&
                std::abs(staticResult[0].x-0.4f)<2.0e-5f &&
                std::abs(staticResult[0].y-0.4f)<2.0e-5f &&
                std::abs(staticResult[0].z+0.04f)<2.0e-5f &&
                std::abs(staticResult[1].z-1.0f)<2.0e-5f,
                "BVH finds nearest non-prefix triangle for first query");
        require(std::isfinite(staticResult[2].x) &&
                std::abs(staticResult[2].x-2.2f)<2.0e-5f &&
                std::abs(staticResult[2].y-2.2f)<2.0e-5f &&
                std::abs(staticResult[2].z+0.01f)<2.0e-5f &&
                std::abs(staticResult[3].z-1.0f)<2.0e-5f,
                "BVH finds nearest triangle in a distant branch");
        // ABI 16 must be rejected before the ABI 17 contact/status bindings are read.
        // Null contact-table arguments make any accidental table access fail.
        NMNumiHumanTendonFEMLoadDispatchGPU oldABIDispatch{};
        oldABIDispatch.abiVersion = 16u;
        oldABIDispatch.environmentCount = 1u;
        oldABIDispatch.femNodeCount = 1u;
        oldABIDispatch.bodyPoseStride = static_cast<nm_u32>(count);
        oldABIDispatch.femBodyContactSampleCount = 1u;
        oldABIDispatch.femBodyContactCandidateTriangleCount = 1u;
        auto oldABIDispatchBuffer = dispatchBuffer(
            &oldABIDispatch, sizeof(oldABIDispatch));
        std::array<nm_float4,1> oldABIOutput{{f4(-1.0f,-1.0f,-1.0f,-1.0f)}};
        auto oldABIOutputBuffer = dispatchBuffer(
            oldABIOutput.data(), sizeof(oldABIOutput));
        require(oldABIDispatchBuffer && oldABIOutputBuffer,
                "old ABI mismatch test buffers");
        id<MTLFunction> oldABIFunction = [library
            newFunctionWithName:@"nm_numi_human_contact_abi16_mismatch_coupon"];
        require(oldABIFunction != nil, "ABI mismatch guard kernel");
        NSError* oldABIPipelineError = nil;
        id<MTLComputePipelineState> oldABIPipeline = [device
            newComputePipelineStateWithFunction:oldABIFunction
                                          error:&oldABIPipelineError];
        require(oldABIPipeline != nil, "ABI mismatch guard pipeline");
        id<MTLCommandBuffer> oldABICommand = [queue commandBuffer];
        id<MTLComputeCommandEncoder> oldABIEncoder =
            [oldABICommand computeCommandEncoder];
        [oldABIEncoder setComputePipelineState:oldABIPipeline];
        [oldABIEncoder setBuffer:oldABIDispatchBuffer offset:0 atIndex:0];
        [oldABIEncoder setBuffer:oldABIOutputBuffer offset:0 atIndex:1];
        [oldABIEncoder dispatchThreads:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [oldABIEncoder endEncoding];
        [oldABICommand commit];
        [oldABICommand waitUntilCompleted];
        require(oldABICommand.status == MTLCommandBufferStatusCompleted,
                "stale ABI 16 dispatch is rejected before unbound contact tables are read");
        const auto* oldABIResult = static_cast<const nm_float4*>(
            oldABIOutputBuffer.contents);
        require(oldABIResult[0].x == 0.0f &&
                oldABIResult[0].y == 16.0f &&
                oldABIResult[0].z == 1.0f,
                "ABI 15 dispatch returns invalid before triangle access");

        // Adversarial precision and coverage cases for complete static banks.
        // The two body-4 planes have different true distances below one Float32
        // ULP at z=1, but both rounded absolute closest points equal the query.
        // Stable IDs intentionally favor the farther first plane if ranking is
        // done by subtracting those rounded absolute points.
        const float zOneULP=std::nextafter(1.0f,2.0f);
        std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,7>
            precisionTriangles{};
        std::copy(staticTriangles.begin(),staticTriangles.end(),
                  precisionTriangles.begin());
        auto farther=makeStaticTriangle(0.0f,0.0f,1.0f,50u,10u);
        farther.bodyIndex=4u;
        farther.bodyLocalVertex1.z=zOneULP;
        farther.bodyLocalVertex2.z=zOneULP;
        auto nearer=makeStaticTriangle(0.0f,0.0f,1.0f,50u,20u);
        nearer.bodyIndex=4u;
        nearer.bodyLocalVertex1.z=zOneULP;
        precisionTriangles[5]=farther;
        precisionTriangles[6]=nearer;
        const auto preciseTree=numi::matter::fem_body_contact_bvh::
            buildStaticCompleteBodyBVH(precisionTriangles,
                                       precisionTriangles.size());
        require(preciseTree.succeeded() && preciseTree.bodies.size()==2u,
                "build complete two-body precision bank");
        const Vec3 preciseQuery{0.4,0.4,1.0};
        const Vec3 farPoint0{0.0,0.0,1.0};
        const Vec3 farPoint1{2.0,0.0,double(zOneULP)};
        const Vec3 farPoint2{0.0,2.0,double(zOneULP)};
        const Vec3 nearPoint0{0.0,0.0,1.0};
        const Vec3 nearPoint1{2.0,0.0,double(zOneULP)};
        const Vec3 nearPoint2{0.0,2.0,1.0};
        const auto farCPU=closestPointOnTriangle(
            preciseQuery,farPoint0,farPoint1,farPoint2);
        const auto nearCPU=closestPointOnTriangle(
            preciseQuery,nearPoint0,nearPoint1,nearPoint2);
        const Vec3 farResidual=sub(preciseQuery,farCPU.point);
        const Vec3 nearResidual=sub(preciseQuery,nearCPU.point);
        require(norm2(nearResidual)<norm2(farResidual),
                "double CPU oracle ranks the second sub-ULP face closer");
        require(float(farCPU.point.x)==0.4f && float(farCPU.point.y)==0.4f &&
                float(farCPU.point.z)==1.0f &&
                float(nearCPU.point.x)==0.4f && float(nearCPU.point.y)==0.4f &&
                float(nearCPU.point.z)==1.0f,
                "both CPU oracle closest points collapse to the same Float32 query");

        std::array<NMNumiHumanFEMBodyContactSampleGPU,3> precisionSamples{{
            makeSample(0u,4u,triangle[0],triangle[1],triangle[2],area,k,strain),
            makeSample(1u,1u,triangle[0],triangle[1],triangle[2],area,k,strain),
            makeSample(2u,1u,triangle[0],triangle[1],triangle[2],area,k,strain)}};
        std::array<NMFEMNodeStateGPU,3> precisionNodes{};
        precisionNodes[0].positionAndMass=f4(0.4f,0.4f,1.0f,1.0f);
        precisionNodes[1].positionAndMass=f4(0.4f,0.4f,1.0f,1.0f);
        precisionNodes[2].positionAndMass=f4(0.4f,0.4f,-0.2f,1.0f);
        for (auto& n:precisionNodes) {
            n.velocityAndInverseMass=f4(0,0,0,1);
            n.restAndFixed=f4(0,0,0,0);
        }
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>
            precisionRanges;
        require(numi::matter::fem_body_contact_bvh::
                makeStaticCompleteBodyContactRanges(
                    precisionSamples,preciseTree.bodies,0.05f,
                    precisionRanges),
                "prepare complete precision and far-field ranges");
        NMNumiHumanTendonFEMLoadDispatchGPU precisionDispatch{};
        precisionDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        precisionDispatch.environmentCount=1u;
        precisionDispatch.femNodeCount=3u;
        precisionDispatch.bodyPoseStride=static_cast<nm_u32>(count);
        precisionDispatch.articulationFirstBody=0u;
        precisionDispatch.femBodyContactSampleCount=3u;
        precisionDispatch.femBodyContactCandidateTriangleCount=
            static_cast<nm_u32>(preciseTree.triangles.size());
        auto precisionDispatchBuffer=dispatchBuffer(
            &precisionDispatch,sizeof(precisionDispatch));
        auto precisionSampleBuffer=dispatchBuffer(
            precisionSamples.data(),sizeof(precisionSamples));
        auto precisionNodeBuffer=dispatchBuffer(
            precisionNodes.data(),sizeof(precisionNodes));
        auto precisionRangeBuffer=dispatchBuffer(
            precisionRanges.data(),precisionRanges.size()*sizeof(precisionRanges[0]));
        auto precisionTriangleBuffer=dispatchBuffer(
            preciseTree.triangles.data(),
            preciseTree.triangles.size()*sizeof(preciseTree.triangles[0]));
        auto precisionBVHNodeBuffer=dispatchBuffer(
            preciseTree.nodes.data(),preciseTree.nodes.size()*sizeof(preciseTree.nodes[0]));
        std::array<nm_float4,6> precisionOutput{};
        auto precisionOutputBuffer=dispatchBuffer(
            precisionOutput.data(),sizeof(precisionOutput));
        require(precisionDispatchBuffer && precisionSampleBuffer &&
                precisionNodeBuffer && precisionRangeBuffer &&
                precisionTriangleBuffer && precisionBVHNodeBuffer &&
                precisionOutputBuffer,
                "precision and far-field GPU buffers");
        id<MTLCommandBuffer> precisionCommand=[queue commandBuffer];
        id<MTLComputeCommandEncoder> precisionEncoder=
            [precisionCommand computeCommandEncoder];
        [precisionEncoder setComputePipelineState:staticPipeline];
        [precisionEncoder setBuffer:precisionDispatchBuffer offset:0 atIndex:0];
        [precisionEncoder setBuffer:precisionSampleBuffer offset:0 atIndex:1];
        [precisionEncoder setBuffer:precisionNodeBuffer offset:0 atIndex:2];
        [precisionEncoder setBuffer:poseBuffer offset:0 atIndex:3];
        [precisionEncoder setBuffer:precisionOutputBuffer offset:0 atIndex:4];
        [precisionEncoder setBuffer:precisionRangeBuffer offset:0 atIndex:5];
        [precisionEncoder setBuffer:precisionTriangleBuffer offset:0 atIndex:6];
        [precisionEncoder setBuffer:precisionBVHNodeBuffer offset:0 atIndex:7];
        [precisionEncoder dispatchThreads:MTLSizeMake(3u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [precisionEncoder endEncoding];
        [precisionCommand commit];
        [precisionCommand waitUntilCompleted];
        require(precisionCommand.status==MTLCommandBufferStatusCompleted,
                "precision/far-field query command completes");
        const auto* preciseResult=static_cast<const nm_float4*>(
            precisionOutputBuffer.contents);
        require(std::isfinite(preciseResult[0].x) &&
                preciseResult[0].x==0.4f && preciseResult[0].y==0.4f &&
                preciseResult[0].z==1.0f &&
                std::abs(preciseResult[1].y)<1.0e-8f,
                "GPU ranks the truly nearer sub-ULP face despite rounded-point tie");
        // The bounded legacy linear-range path uses the same precise ranking
        // rule. Keep this as a separate execution of that branch so a future
        // static-BVH-only fix cannot leave neighborhood callers regressed.
        NMNumiHumanTendonFEMLoadDispatchGPU legacyPrecisionDispatch{};
        legacyPrecisionDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        legacyPrecisionDispatch.environmentCount=1u;
        legacyPrecisionDispatch.femNodeCount=1u;
        legacyPrecisionDispatch.bodyPoseStride=static_cast<nm_u32>(count);
        legacyPrecisionDispatch.articulationFirstBody=0u;
        legacyPrecisionDispatch.femBodyContactSampleCount=1u;
        legacyPrecisionDispatch.femBodyContactCandidateTriangleCount=
            static_cast<nm_u32>(preciseTree.triangles.size());
        const auto legacyPrecisionSample=precisionSamples[0];
        const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU legacyPrecisionRange{
            5u,2u,0.05f,0u,{0.0f,0.0f,0.0f,0.0f}};
        std::array<nm_float4,2> legacyPrecisionOutput{};
        auto legacyPrecisionDispatchBuffer=dispatchBuffer(
            &legacyPrecisionDispatch,sizeof(legacyPrecisionDispatch));
        auto legacyPrecisionSampleBuffer=dispatchBuffer(
            &legacyPrecisionSample,sizeof(legacyPrecisionSample));
        auto legacyPrecisionNodeBuffer=dispatchBuffer(
            &precisionNodes[0],sizeof(precisionNodes[0]));
        auto legacyPrecisionRangeBuffer=dispatchBuffer(
            &legacyPrecisionRange,sizeof(legacyPrecisionRange));
        auto legacyPrecisionOutputBuffer=dispatchBuffer(
            legacyPrecisionOutput.data(),sizeof(legacyPrecisionOutput));
        require(legacyPrecisionDispatchBuffer && legacyPrecisionSampleBuffer &&
                legacyPrecisionNodeBuffer && legacyPrecisionRangeBuffer &&
                legacyPrecisionOutputBuffer,
                "legacy precision ranking GPU buffers");
        id<MTLFunction> legacyPrecisionFunction=[library
            newFunctionWithName:@"nm_numi_human_legacy_precision_coupon"];
        require(legacyPrecisionFunction!=nil,
                "legacy precision-ranking coupon kernel");
        NSError* legacyPrecisionError=nil;
        id<MTLComputePipelineState> legacyPrecisionPipeline=
            [device newComputePipelineStateWithFunction:legacyPrecisionFunction
                                                  error:&legacyPrecisionError];
        if (legacyPrecisionPipeline==nil) {
            std::fprintf(stderr,"legacy precision pipeline: %s\n",
                legacyPrecisionError.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandBuffer> legacyPrecisionCommand=[queue commandBuffer];
        id<MTLComputeCommandEncoder> legacyPrecisionEncoder=
            [legacyPrecisionCommand computeCommandEncoder];
        [legacyPrecisionEncoder setComputePipelineState:legacyPrecisionPipeline];
        [legacyPrecisionEncoder setBuffer:legacyPrecisionDispatchBuffer offset:0 atIndex:0];
        [legacyPrecisionEncoder setBuffer:legacyPrecisionSampleBuffer offset:0 atIndex:1];
        [legacyPrecisionEncoder setBuffer:legacyPrecisionNodeBuffer offset:0 atIndex:2];
        [legacyPrecisionEncoder setBuffer:poseBuffer offset:0 atIndex:3];
        [legacyPrecisionEncoder setBuffer:legacyPrecisionOutputBuffer offset:0 atIndex:4];
        [legacyPrecisionEncoder setBuffer:legacyPrecisionRangeBuffer offset:0 atIndex:5];
        [legacyPrecisionEncoder setBuffer:precisionTriangleBuffer offset:0 atIndex:6];
        [legacyPrecisionEncoder setBuffer:precisionBVHNodeBuffer offset:0 atIndex:7];
        [legacyPrecisionEncoder dispatchThreads:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [legacyPrecisionEncoder endEncoding];
        [legacyPrecisionCommand commit];
        [legacyPrecisionCommand waitUntilCompleted];
        require(legacyPrecisionCommand.status==MTLCommandBufferStatusCompleted,
                "legacy precise-range command completes");
        const auto* legacyPrecise=static_cast<const nm_float4*>(
            legacyPrecisionOutputBuffer.contents);
        require(std::isfinite(legacyPrecise[0].x) &&
                legacyPrecise[0].x==0.4f && legacyPrecise[0].y==0.4f &&
                legacyPrecise[0].z==1.0f &&
                std::abs(legacyPrecise[1].y)<1.0e-8f,
                "legacy range ranks the truly nearer sub-ULP face");
        require(std::isfinite(preciseResult[2].x) &&
                std::isfinite(preciseResult[3].x) &&
                preciseResult[2].w==0.0f,
                "complete BVH treats a far exterior query as valid and inactive");
        require(std::isnan(preciseResult[4].x) &&
                std::isnan(preciseResult[5].x),
                "complete BVH keeps far interior penetration fail-closed");

        if (staticBVHOnly) {
            std::puts("static complete body BVH GPU query PASS");
            return 0;
        }

        // The final candidate wins a complete maximum-size bank. This catches
        // CPU/Metal capacity disagreement and accidental prefix truncation.
        constexpr nm_u32 bankLimit=
            NM_NUMI_HUMAN_FEM_BODY_CONTACT_MAXIMUM_NEIGHBORHOOD_TRIANGLES;
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
            maximumCandidates(bankLimit+1u,ownerCandidates[0]);
        for (nm_u32 i=0u;i<maximumCandidates.size();++i) {
            maximumCandidates[i].triangleStableId=1000u+i;
            maximumCandidates[i].boneStableId=1u;
        }
        maximumCandidates[bankLimit-1u].bodyLocalVertex0.z=-0.04f;
        maximumCandidates[bankLimit-1u].bodyLocalVertex1.z=-0.04f;
        maximumCandidates[bankLimit-1u].bodyLocalVertex2.z=-0.04f;
        NMNumiHumanTendonFEMLoadDispatchGPU maximumDispatch=dispatch;
        maximumDispatch.femBodyContactSampleCount=1u;
        maximumDispatch.femBodyContactCandidateTriangleCount=bankLimit;
        NMNumiHumanFEMBodyContactNeighborhoodRangeGPU maximumRange{
            0u,bankLimit,0.05f,1u,{0.4f,0.4f,-0.05f,0.01f}};
        std::array<nm_float4,2> maximumOutput{};
        auto maximumDispatchBuffer=dispatchBuffer(&maximumDispatch,sizeof(maximumDispatch));
        auto maximumRangeBuffer=dispatchBuffer(&maximumRange,sizeof(maximumRange));
        auto maximumCandidateBuffer=dispatchBuffer(maximumCandidates.data(),
            maximumCandidates.size()*sizeof(maximumCandidates[0]));
        auto maximumOutputBuffer=dispatchBuffer(maximumOutput.data(),sizeof(maximumOutput));
        for (nm_u32 trial=0u;trial<2u;++trial) {
            if (trial==1u) {
                maximumDispatch.femBodyContactCandidateTriangleCount=bankLimit+1u;
                maximumRange.triangleCount=bankLimit+1u;
                std::memcpy(maximumDispatchBuffer.contents,&maximumDispatch,sizeof(maximumDispatch));
                std::memcpy(maximumRangeBuffer.contents,&maximumRange,sizeof(maximumRange));
            }
            auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:selectorPipeline];
            [encoder setBuffer:maximumDispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:ownerSampleBuffer offset:0 atIndex:1];
            [encoder setBuffer:nodeBuffer offset:0 atIndex:2];
            [encoder setBuffer:poseBuffer offset:0 atIndex:3];
            [encoder setBuffer:maximumOutputBuffer offset:0 atIndex:4];
            [encoder setBuffer:maximumRangeBuffer offset:0 atIndex:5];
            [encoder setBuffer:maximumCandidateBuffer offset:0 atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,"maximum bank query completes");
            const auto* out=static_cast<const nm_float4*>(maximumOutputBuffer.contents);
            if (trial==0u)
                require(std::isfinite(out[0].x) && std::abs(out[1].z+0.04f)<1.0e-6f &&
                    std::abs(out[0].z-2.499975f)<2.0e-3f,
                    "last triangle of complete maximum bank wins the GPU query");
            else require(std::isnan(out[0].x)&&std::isnan(out[1].x),
                    "GPU bank capacity overflow fails closed");
        }

        // A genuinely empty optional triangle table uses typed dummy buffers,
        // while a zero dispatch count selects the exact legacy behavior.
        NMNumiHumanTendonFEMLoadDispatchGPU emptyDispatch=selectorDispatch;
        emptyDispatch.femBodyContactSampleCount=1u;
        emptyDispatch.femBodyContactCandidateTriangleCount=0u;
        const auto emptySample=samples[5];
        const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU emptyRange{};
        const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU emptyTriangle{};
        std::array<nm_float4,2> emptyOutput{};
        auto emptyDispatchBuffer=dispatchBuffer(&emptyDispatch,sizeof(emptyDispatch));
        auto emptySampleBuffer=dispatchBuffer(&emptySample,sizeof(emptySample));
        auto emptyRangeBuffer=dispatchBuffer(&emptyRange,sizeof(emptyRange));
        auto emptyTriangleBuffer=dispatchBuffer(&emptyTriangle,sizeof(emptyTriangle));
        auto emptyOutputBuffer=dispatchBuffer(emptyOutput.data(),sizeof(emptyOutput));
        require(emptyDispatchBuffer && emptySampleBuffer && emptyRangeBuffer &&
                emptyTriangleBuffer && emptyOutputBuffer,
                "typed empty neighborhood sidecar buffers");
        id<MTLCommandBuffer> emptyCommand=[queue commandBuffer];
        id<MTLComputeCommandEncoder> emptyEncoder=[emptyCommand computeCommandEncoder];
        [emptyEncoder setComputePipelineState:selectorPipeline];
        [emptyEncoder setBuffer:emptyDispatchBuffer offset:0 atIndex:0];
        [emptyEncoder setBuffer:emptySampleBuffer offset:0 atIndex:1];
        [emptyEncoder setBuffer:nodeBuffer offset:0 atIndex:2];
        [emptyEncoder setBuffer:poseBuffer offset:0 atIndex:3];
        [emptyEncoder setBuffer:emptyOutputBuffer offset:0 atIndex:4];
        [emptyEncoder setBuffer:emptyRangeBuffer offset:0 atIndex:5];
        [emptyEncoder setBuffer:emptyTriangleBuffer offset:0 atIndex:6];
        [emptyEncoder dispatchThreads:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [emptyEncoder endEncoding];
        [emptyCommand commit];
        [emptyCommand waitUntilCompleted];
        const auto* emptySelected=static_cast<const nm_float4*>(
            emptyOutputBuffer.contents);
        require(emptyCommand.status==MTLCommandBufferStatusCompleted &&
                std::isfinite(emptySelected[0].x) &&
                std::abs(emptySelected[0].z-49.9999f)<2.0e-3f &&
                emptySelected[1].z==0.0f,
                "empty optional triangle table preserves legacy contact behavior");

        // Provisional admission uses the same accepted FEM node and body pose,
        // so it validates the same deterministic closest-triangle winner.
        NMNumiHumanTendonFEMLoadDispatchGPU statusDispatch=dispatch;
        statusDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        statusDispatch.endpointCount=1u;
        statusDispatch.transferStride=1u;
        statusDispatch.dofCount=1u;
        statusDispatch.generalizedForceStride=1u;
        statusDispatch.muscleCount=1u;
        statusDispatch.pointJacobianStride=1u;
        statusDispatch.bodyJacobianPointOffset=0u;
        statusDispatch.femBodyContactCandidateTriangleCount=2u;
        std::array<NMNumiHumanTendonFEMNodeLoadGPU,count> statusLoads{};
        std::array<NMNumiHumanTendonFEMNodeAnchorGPU,count> statusAnchors{};
        for (std::size_t i=0;i<count;++i) {
            for (auto& endpoint:statusLoads[i].endpointIndex)
                endpoint=NM_INVALID_INDEX;
            statusAnchors[i].bodyIndex=MR_INVALID_INDEX;
        }
        MRMetalWorldStatusGPU provisionalStatus{};
        auto statusDispatchBuffer=dispatchBuffer(
            &statusDispatch,sizeof(statusDispatch));
        auto statusLoadsBuffer=dispatchBuffer(
            statusLoads.data(),sizeof(statusLoads));
        auto statusAnchorsBuffer=dispatchBuffer(
            statusAnchors.data(),sizeof(statusAnchors));
        auto provisionalStatusBuffer=dispatchBuffer(
            &provisionalStatus,sizeof(provisionalStatus));
        const nm_float4 typedEmpty{};
        const NMNumiHumanFEMBodyContactBVHNodeGPU emptyStatusBVHNode{};
        auto statusBVHBuffer=dispatchBuffer(
            &emptyStatusBVHNode,sizeof(emptyStatusBVHNode));
        std::array<std::uint32_t,1> sampleValidity{};
        std::array<std::uint32_t,1> blockFirstInvalid{};
        auto sampleValidityBuffer=dispatchBuffer(
            sampleValidity.data(),sizeof(sampleValidity));
        auto blockFirstInvalidBuffer=dispatchBuffer(
            blockFirstInvalid.data(),sizeof(blockFirstInvalid));
        require(statusBVHBuffer && sampleValidityBuffer &&
                blockFirstInvalidBuffer,
                "parallel provisional-status validation scratch buffers");
        auto statusKernel=[&](NSString* name) {
            id<MTLFunction> function=[library newFunctionWithName:
                [NSString stringWithFormat:@"numi_matter_metal::%@",name]];
            require(function!=nil,"parallel status validation kernel exists");
            NSError* error=nil;
            id<MTLComputePipelineState> pipeline=
                [device newComputePipelineStateWithFunction:function error:&error];
            if (pipeline==nil) {
                std::fprintf(stderr,"parallel status pipeline %s: %s\n",
                    name.UTF8String,error.localizedDescription.UTF8String);
                std::exit(1);
            }
            return pipeline;
        };
        id<MTLComputePipelineState> sampleValidationPipeline=statusKernel(
            @"nm_numi_human_validate_fem_body_contact_samples");
        id<MTLComputePipelineState> validationReducePipeline=statusKernel(
            @"nm_numi_human_reduce_fem_body_contact_sample_validation");
        auto encodeStatusValidation=[&](id<MTLCommandBuffer> command,
                                        id<MTLBuffer> nodeStates) {
            id<MTLComputeCommandEncoder> validate=[command computeCommandEncoder];
            [validate setComputePipelineState:sampleValidationPipeline];
            [validate setBuffer:statusDispatchBuffer offset:0 atIndex:0];
            [validate setBuffer:ownerSampleBuffer offset:0 atIndex:1];
            [validate setBuffer:ownerRangeBuffer offset:0 atIndex:2];
            [validate setBuffer:ownerCandidateBuffer offset:0 atIndex:3];
            [validate setBuffer:statusBVHBuffer offset:0 atIndex:4];
            [validate setBuffer:nodeStates offset:0 atIndex:5];
            [validate setBuffer:poseBuffer offset:0 atIndex:6];
            [validate setBuffer:sampleValidityBuffer offset:0 atIndex:7];
            [validate dispatchThreads:MTLSizeMake(1,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [validate endEncoding];
            id<MTLComputeCommandEncoder> reduce=[command computeCommandEncoder];
            [reduce setComputePipelineState:validationReducePipeline];
            [reduce setBuffer:statusDispatchBuffer offset:0 atIndex:0];
            [reduce setBuffer:sampleValidityBuffer offset:0 atIndex:1];
            [reduce setBuffer:blockFirstInvalidBuffer offset:0 atIndex:2];
            [reduce dispatchThreads:MTLSizeMake(1,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [reduce endEncoding];
        };
        id<MTLFunction> statusFunction=[library newFunctionWithName:
            @"numi_matter_metal::nm_numi_human_prepare_provisional_status"];
        require(statusFunction!=nil,"production provisional-status kernel");
        NSError* statusError=nil;
        id<MTLComputePipelineState> statusPipeline=
            [device newComputePipelineStateWithFunction:statusFunction
                                                  error:&statusError];
        if (statusPipeline==nil) {
            std::fprintf(stderr,"status pipeline: %s\n",
                statusError.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandBuffer> statusCommand=[queue commandBuffer];
        encodeStatusValidation(statusCommand,nodeBuffer);
        id<MTLComputeCommandEncoder> statusEncoder=
            [statusCommand computeCommandEncoder];
        [statusEncoder setComputePipelineState:statusPipeline];
        [statusEncoder setBuffer:statusDispatchBuffer offset:0 atIndex:0];
        [statusEncoder setBuffer:statusLoadsBuffer offset:0 atIndex:1];
        [statusEncoder setBuffer:statusAnchorsBuffer offset:0 atIndex:2];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:3];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:4];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:5];
        [statusEncoder setBuffer:poseBuffer offset:0 atIndex:6];
        [statusEncoder setBuffer:provisionalStatusBuffer offset:0 atIndex:7];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:8];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:9];
        [statusEncoder setBuffer:ownerSampleBuffer offset:0 atIndex:10];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:11];
        [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:12];
        [statusEncoder setBuffer:nodeBuffer offset:0 atIndex:13];
        [statusEncoder setBuffer:ownerRangeBuffer offset:0 atIndex:14];
        [statusEncoder setBuffer:ownerCandidateBuffer offset:0 atIndex:15];
        [statusEncoder setBuffer:statusBVHBuffer offset:0 atIndex:16];
        [statusEncoder setBuffer:blockFirstInvalidBuffer offset:0 atIndex:17];
        [statusEncoder dispatchThreads:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [statusEncoder endEncoding];
        [statusCommand commit];
        [statusCommand waitUntilCompleted];
        require(statusCommand.status==MTLCommandBufferStatusCompleted &&
                static_cast<const MRMetalWorldStatusGPU*>(
                    provisionalStatusBuffer.contents)[0].code==MR_STEP_SUCCESS,
                "production provisional status accepts the in-range selected face");

        std::array<NMFEMNodeStateGPU,count> outsideAcceptedNodes=nodes;
        // This point is tangentially outside the motion ball but remains only
        // 10 mm from the prepared z=-40 mm face (inside the 50 mm distance cap).
        outsideAcceptedNodes[ownerSample.slaveNode].positionAndMass=
            f4(0.2f,0.4f,-0.05f,1.0f);
        MRMetalWorldStatusGPU rejectedProvisionalStatus{};
        auto outsideAcceptedNodeBuffer=dispatchBuffer(
            outsideAcceptedNodes.data(),sizeof(outsideAcceptedNodes));
        auto rejectedStatusBuffer=dispatchBuffer(
            &rejectedProvisionalStatus,sizeof(rejectedProvisionalStatus));
        id<MTLCommandBuffer> rejectedStatusCommand=[queue commandBuffer];
        encodeStatusValidation(rejectedStatusCommand,outsideAcceptedNodeBuffer);
        id<MTLComputeCommandEncoder> rejectedStatusEncoder=
            [rejectedStatusCommand computeCommandEncoder];
        [rejectedStatusEncoder setComputePipelineState:statusPipeline];
        [rejectedStatusEncoder setBuffer:statusDispatchBuffer offset:0 atIndex:0];
        [rejectedStatusEncoder setBuffer:statusLoadsBuffer offset:0 atIndex:1];
        [rejectedStatusEncoder setBuffer:statusAnchorsBuffer offset:0 atIndex:2];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:3];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:4];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:5];
        [rejectedStatusEncoder setBuffer:poseBuffer offset:0 atIndex:6];
        [rejectedStatusEncoder setBuffer:rejectedStatusBuffer offset:0 atIndex:7];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:8];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:9];
        [rejectedStatusEncoder setBuffer:ownerSampleBuffer offset:0 atIndex:10];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:11];
        [rejectedStatusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:12];
        [rejectedStatusEncoder setBuffer:outsideAcceptedNodeBuffer offset:0 atIndex:13];
        [rejectedStatusEncoder setBuffer:ownerRangeBuffer offset:0 atIndex:14];
        [rejectedStatusEncoder setBuffer:ownerCandidateBuffer offset:0 atIndex:15];
        [rejectedStatusEncoder setBuffer:statusBVHBuffer offset:0 atIndex:16];
        [rejectedStatusEncoder setBuffer:blockFirstInvalidBuffer offset:0 atIndex:17];
        [rejectedStatusEncoder dispatchThreads:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [rejectedStatusEncoder endEncoding];
        [rejectedStatusCommand commit];
        [rejectedStatusCommand waitUntilCompleted];
        require(rejectedStatusCommand.status==MTLCommandBufferStatusCompleted &&
                static_cast<const MRMetalWorldStatusGPU*>(
                    rejectedStatusBuffer.contents)[0].code!=MR_STEP_SUCCESS,
                "provisional status rejects tangentially out-of-envelope nodes still near a prepared face");

        auto runValidationBatch=[&](
            const std::uint32_t sampleCount,
            const std::uint32_t environmentCount,
            const std::vector<std::uint32_t>& invalidSamples,
            const std::uint32_t nanEnvironment,
            const std::uint32_t nanSample,
            const bool staleABI,
            const bool failEarlierGate,
            const bool badInputPointers) {
            const std::uint32_t nodeCount=std::max(1u,sampleCount);
            NMNumiHumanTendonFEMLoadDispatchGPU batchDispatch=statusDispatch;
            batchDispatch.abiVersion=staleABI
                ? NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION-1u
                : NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
            batchDispatch.environmentCount=environmentCount;
            batchDispatch.femNodeCount=nodeCount;
            batchDispatch.bodyPoseStride=nodeCount;
            batchDispatch.femBodyContactSampleCount=sampleCount;
            batchDispatch.endpointCount=failEarlierGate ? 0u : 1u;
            batchDispatch.transferStride=1u;
            batchDispatch.generalizedForceStride=1u;
            batchDispatch.dofCount=1u;
            batchDispatch.muscleCount=1u;
            batchDispatch.pointJacobianStride=1u;
            batchDispatch.bodyJacobianPointOffset=0u;
            auto batchSamples=std::vector<NMNumiHumanFEMBodyContactSampleGPU>(
                std::max(1u,sampleCount),ownerSample);
            for (std::uint32_t i=0u;i<sampleCount;++i) {
                batchSamples[i].slaveNode=i;
            }
            for (const std::uint32_t invalid:invalidSamples) {
                require(invalid<sampleCount,"validation fixture bad sample is in range");
                batchSamples[invalid].slaveNode=nodeCount;
            }
            auto batchRanges=std::vector<
                NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>(
                    std::max(1u,sampleCount),ownerRanges[0]);
            auto batchNodes=std::vector<NMFEMNodeStateGPU>(
                static_cast<std::size_t>(environmentCount)*nodeCount,nodes[1]);
            if (nanEnvironment<environmentCount && nanSample<sampleCount) {
                batchNodes[static_cast<std::size_t>(nanEnvironment)*nodeCount+
                    nanSample].positionAndMass.x=
                        std::numeric_limits<float>::quiet_NaN();
            }
            auto batchPoses=std::vector<MRArticulatedBodyPoseGPU>(
                static_cast<std::size_t>(environmentCount)*nodeCount,poses[1]);
            const std::uint32_t blockSize=
                NM_NUMI_HUMAN_FEM_BODY_CONTACT_VALIDATION_BLOCK_SIZE;
            const std::uint32_t blockCount=sampleCount/blockSize+
                (sampleCount%blockSize!=0u ? 1u : 0u);
            auto sampleValidity=std::vector<std::uint32_t>(
                std::max<std::size_t>(1u,
                    static_cast<std::size_t>(environmentCount)*sampleCount),
                0xdeadbeefu);
            auto blockFirstInvalid=std::vector<std::uint32_t>(
                std::max<std::size_t>(1u,
                    static_cast<std::size_t>(environmentCount)*blockCount),
                0xdeadbeefu);
            std::array<NMNumiHumanFEMBodyContactBVHNodeGPU,1>
                batchBVH{{{}}};
            std::vector<NMNumiHumanTendonFEMNodeLoadGPU> batchLoads(nodeCount);
            for (auto& load:batchLoads) {
                for (auto& endpoint:load.endpointIndex)
                    endpoint=NM_INVALID_INDEX;
            }
            std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> batchAnchors(nodeCount);
            for (auto& anchor:batchAnchors) anchor.bodyIndex=MR_INVALID_INDEX;
            auto batchStatus=std::vector<MRMetalWorldStatusGPU>(environmentCount);

            const auto batchDispatchBuffer=dispatchBuffer(
                &batchDispatch,sizeof(batchDispatch));
            const auto batchSamplesBuffer=badInputPointers
                ? dispatchBuffer(&ownerSample,sizeof(ownerSample))
                : dispatchBuffer(batchSamples.data(),
                    batchSamples.size()*sizeof(batchSamples[0]));
            const auto batchRangesBuffer=badInputPointers
                ? dispatchBuffer(&ownerRanges[0],sizeof(ownerRanges[0]))
                : dispatchBuffer(batchRanges.data(),
                    batchRanges.size()*sizeof(batchRanges[0]));
            const auto batchNodesBuffer=badInputPointers
                ? dispatchBuffer(&nodes[1],sizeof(nodes[1]))
                : dispatchBuffer(batchNodes.data(),
                    batchNodes.size()*sizeof(batchNodes[0]));
            const auto batchPosesBuffer=badInputPointers
                ? dispatchBuffer(&poses[1],sizeof(poses[1]))
                : dispatchBuffer(batchPoses.data(),
                    batchPoses.size()*sizeof(batchPoses[0]));
            const auto batchValidityBuffer=dispatchBuffer(sampleValidity.data(),
                sampleValidity.size()*sizeof(sampleValidity[0]));
            const auto batchBlockBuffer=dispatchBuffer(blockFirstInvalid.data(),
                blockFirstInvalid.size()*sizeof(blockFirstInvalid[0]));
            const auto batchTriangleBuffer=dispatchBuffer(
                ownerCandidates.data(),sizeof(ownerCandidates));
            const auto batchBVHBuffer=dispatchBuffer(
                batchBVH.data(),sizeof(batchBVH));
            const auto batchLoadsBuffer=dispatchBuffer(batchLoads.data(),
                batchLoads.size()*sizeof(batchLoads[0]));
            const auto batchAnchorsBuffer=dispatchBuffer(batchAnchors.data(),
                batchAnchors.size()*sizeof(batchAnchors[0]));
            const auto batchStatusBuffer=dispatchBuffer(batchStatus.data(),
                batchStatus.size()*sizeof(batchStatus[0]));
            require(batchDispatchBuffer && batchSamplesBuffer &&
                    batchRangesBuffer && batchNodesBuffer && batchPosesBuffer &&
                    batchValidityBuffer && batchBlockBuffer &&
                    batchTriangleBuffer && batchBVHBuffer && batchLoadsBuffer &&
                    batchAnchorsBuffer && batchStatusBuffer,
                    "parallel batch validation buffers");
            id<MTLCommandBuffer> batchCommand=[queue commandBuffer];
            if (sampleCount!=0u) {
                id<MTLComputeCommandEncoder> validate=
                    [batchCommand computeCommandEncoder];
                [validate setComputePipelineState:sampleValidationPipeline];
                [validate setBuffer:batchDispatchBuffer offset:0 atIndex:0];
                [validate setBuffer:batchSamplesBuffer offset:0 atIndex:1];
                [validate setBuffer:batchRangesBuffer offset:0 atIndex:2];
                [validate setBuffer:batchTriangleBuffer offset:0 atIndex:3];
                [validate setBuffer:batchBVHBuffer offset:0 atIndex:4];
                [validate setBuffer:batchNodesBuffer offset:0 atIndex:5];
                [validate setBuffer:batchPosesBuffer offset:0 atIndex:6];
                [validate setBuffer:batchValidityBuffer offset:0 atIndex:7];
                [validate dispatchThreads:MTLSizeMake(
                    static_cast<NSUInteger>(sampleCount)*environmentCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                [validate endEncoding];
                id<MTLComputeCommandEncoder> reduce=
                    [batchCommand computeCommandEncoder];
                [reduce setComputePipelineState:validationReducePipeline];
                [reduce setBuffer:batchDispatchBuffer offset:0 atIndex:0];
                [reduce setBuffer:batchValidityBuffer offset:0 atIndex:1];
                [reduce setBuffer:batchBlockBuffer offset:0 atIndex:2];
                [reduce dispatchThreads:MTLSizeMake(
                    static_cast<NSUInteger>(blockCount)*environmentCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                [reduce endEncoding];
            }
            id<MTLComputeCommandEncoder> statusEncoder=
                [batchCommand computeCommandEncoder];
            [statusEncoder setComputePipelineState:statusPipeline];
            [statusEncoder setBuffer:batchDispatchBuffer offset:0 atIndex:0];
            [statusEncoder setBuffer:batchLoadsBuffer offset:0 atIndex:1];
            [statusEncoder setBuffer:batchAnchorsBuffer offset:0 atIndex:2];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:3];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:4];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:5];
            [statusEncoder setBuffer:batchPosesBuffer offset:0 atIndex:6];
            [statusEncoder setBuffer:batchStatusBuffer offset:0 atIndex:7];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:8];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:9];
            [statusEncoder setBuffer:batchSamplesBuffer offset:0 atIndex:10];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:11];
            [statusEncoder setBytes:&typedEmpty length:sizeof(typedEmpty) atIndex:12];
            [statusEncoder setBuffer:batchNodesBuffer offset:0 atIndex:13];
            [statusEncoder setBuffer:batchRangesBuffer offset:0 atIndex:14];
            [statusEncoder setBuffer:batchTriangleBuffer offset:0 atIndex:15];
            [statusEncoder setBuffer:batchBVHBuffer offset:0 atIndex:16];
            if (sampleCount!=0u) {
                [statusEncoder setBuffer:batchBlockBuffer offset:0 atIndex:17];
            } else {
                const std::uint32_t emptyFirstInvalid=MR_INVALID_INDEX;
                [statusEncoder setBytes:&emptyFirstInvalid
                    length:sizeof(emptyFirstInvalid) atIndex:17];
            }
            [statusEncoder dispatchThreads:MTLSizeMake(environmentCount,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [statusEncoder endEncoding];
            [batchCommand commit];
            [batchCommand waitUntilCompleted];
            require(batchCommand.status==MTLCommandBufferStatusCompleted,
                    "parallel validation batch command completed");
            const auto* validOut=static_cast<const std::uint32_t*>(
                batchValidityBuffer.contents);
            const auto* firstInvalidOut=static_cast<const std::uint32_t*>(
                batchBlockBuffer.contents);
            const auto* statusOut=static_cast<const MRMetalWorldStatusGPU*>(
                batchStatusBuffer.contents);
            for (std::uint32_t environment=0u;
                 environment<environmentCount;++environment) {
                for (std::uint32_t sample=0u;sample<sampleCount;++sample) {
                    const bool expectedValid=!staleABI &&
                        std::find(invalidSamples.begin(),invalidSamples.end(),sample)==
                            invalidSamples.end() &&
                        !(environment==nanEnvironment && sample==nanSample);
                    require(validOut[environment*sampleCount+sample]==
                                (expectedValid ? 1u : 0u),
                            "parallel validator writes every environment/sample result");
                }
                for (std::uint32_t block=0u;block<blockCount;++block) {
                    std::uint32_t expectedFirst=MR_INVALID_INDEX;
                    const std::uint32_t first=block*blockSize;
                    const std::uint32_t end=std::min(first+blockSize,sampleCount);
                    for (std::uint32_t sample=first;sample<end;++sample) {
                        const bool expectedValid=!staleABI &&
                            std::find(invalidSamples.begin(),invalidSamples.end(),sample)==
                                invalidSamples.end() &&
                            !(environment==nanEnvironment && sample==nanSample);
                        if (!expectedValid) { expectedFirst=sample; break; }
                    }
                    require(firstInvalidOut[environment*blockCount+block]==
                                expectedFirst,
                            "block reduction preserves first invalid sample at boundaries");
                }
                const bool statusExpected=!staleABI && !failEarlierGate &&
                    std::none_of(invalidSamples.begin(),invalidSamples.end(),
                        [&](const std::uint32_t sample) {
                            return sample<sampleCount;
                        }) && !(environment==nanEnvironment && nanSample<sampleCount);
                require((statusOut[environment].code==MR_STEP_SUCCESS)==statusExpected,
                        "single provisional-status owner consumes reduced validation");
            }
        };
        runValidationBatch(0u,1u,{},NM_INVALID_INDEX,NM_INVALID_INDEX,
                           false,false,false);
        runValidationBatch(255u,1u,{0u,127u,254u},
                           NM_INVALID_INDEX,NM_INVALID_INDEX,false,false,false);
        runValidationBatch(256u,1u,{0u,127u,255u},
                           NM_INVALID_INDEX,NM_INVALID_INDEX,false,false,false);
        runValidationBatch(257u,2u,{0u,127u,255u,256u},
                           1u,128u,false,false,false);
        runValidationBatch(257u,2u,{},1u,128u,false,true,false);
        runValidationBatch(257u,2u,{},NM_INVALID_INDEX,NM_INVALID_INDEX,
                           true,false,true);

        id<MTLCommandBuffer> ownerCommand=[queue commandBuffer];
        auto encodeOwner=[&](NSString* name, NSUInteger threadCount,
                             id<MTLBuffer> fifthBuffer) {
            id<MTLFunction> ownerFunction=[library newFunctionWithName:[NSString stringWithFormat:
                @"numi_matter_metal::%@",name]];
            require(ownerFunction!=nil,"actual owner kernel exists");
            NSError* ownerError=nil;
            id<MTLComputePipelineState> ownerPipeline=
                [device newComputePipelineStateWithFunction:ownerFunction
                                                      error:&ownerError];
            if (ownerPipeline==nil) {
                std::fprintf(stderr,"owner pipeline %s: %s\\n",
                    name.UTF8String,ownerError.localizedDescription.UTF8String);
                std::exit(1);
            }
            id<MTLComputeCommandEncoder> ownerEncoder=
                [ownerCommand computeCommandEncoder];
            [ownerEncoder setComputePipelineState:ownerPipeline];
            [ownerEncoder setBuffer:dispatchBufferObject offset:0 atIndex:0];
            [ownerEncoder setBuffer:ownerSampleBuffer offset:0 atIndex:1];
            [ownerEncoder setBuffer:nodeBuffer offset:0 atIndex:2];
            [ownerEncoder setBuffer:poseBuffer offset:0 atIndex:3];
            [ownerEncoder setBuffer:fifthBuffer offset:0 atIndex:4];
            if ([name isEqualToString:@"nm_numi_human_audit_fem_body_contact"]) {
                const NMNumiHumanFEMSurfaceAttachmentGPU empty{};
                [ownerEncoder setBytes:&empty length:sizeof(empty) atIndex:5];
                [ownerEncoder setBuffer:ownerRangeBuffer offset:0 atIndex:6];
                [ownerEncoder setBuffer:ownerCandidateBuffer offset:0 atIndex:7];
            } else {
                [ownerEncoder setBuffer:ownerRangeBuffer offset:0 atIndex:5];
                [ownerEncoder setBuffer:ownerCandidateBuffer offset:0 atIndex:6];
            }
            [ownerEncoder dispatchThreads:MTLSizeMake(threadCount,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [ownerEncoder endEncoding];
        };
        encodeOwner(@"nm_numi_human_assemble_fem_body_contact_loads",
                    dispatch.femBodyContactSampleCount,ownerForceBuffer);
        encodeOwner(@"nm_numi_human_assemble_fem_body_contact_wrenches",
                    count,ownerWrenchBuffer);
        encodeOwner(@"nm_numi_human_audit_fem_body_contact",
                    1u,ownerAuditBuffer);
        [ownerCommand commit];
        [ownerCommand waitUntilCompleted];
        require(ownerCommand.status==MTLCommandBufferStatusCompleted,
                "actual owner kernels complete");
        const auto* actualLoads=
            static_cast<const nm_float4*>(ownerForceBuffer.contents);
        const auto* actualWrenches=
            static_cast<const NMNumiHumanBodyWrenchGPU*>(
                ownerWrenchBuffer.contents);
        const auto* actualAudit=
            static_cast<const NMNumiHumanArticularContactAuditGPU*>(
                ownerAuditBuffer.contents);
        require(std::abs(actualLoads[1].x-selected[0].x)<1.0e-5f &&
                std::abs(actualLoads[1].y-selected[0].y)<1.0e-5f &&
                std::abs(actualLoads[1].z-selected[0].z)<1.0e-5f,
                "actual owner FEM load uses the same selected triangle");
        const Vec3 selectedPoint{selected[1].x,selected[1].y,selected[1].z};
        const Vec3 selectedReaction{-selected[0].x,-selected[0].y,-selected[0].z};
        const Vec3 selectedTorque=cross(selectedPoint,selectedReaction);
        require(std::abs(actualWrenches[1].force.x+selected[0].x)<1.0e-5f &&
                std::abs(actualWrenches[1].force.y+selected[0].y)<1.0e-5f &&
                std::abs(actualWrenches[1].force.z+selected[0].z)<1.0e-5f &&
                std::abs(actualWrenches[1].torque.x-selectedTorque.x)<1.0e-5f &&
                std::abs(actualWrenches[1].torque.y-selectedTorque.y)<1.0e-5f &&
                std::abs(actualWrenches[1].torque.z-selectedTorque.z)<1.0e-5f,
                "actual owner body wrench uses selected closest point");
        require(std::abs(actualAudit[0].forceResidualAndL1.x)<1.0e-5f &&
                std::abs(actualAudit[0].forceResidualAndL1.y)<1.0e-5f &&
                std::abs(actualAudit[0].forceResidualAndL1.z)<1.0e-5f &&
                std::abs(actualAudit[0].normalForceAreaAndCounts.y-area)<1.0e-6f &&
                std::abs(actualAudit[0].normalForceAreaAndCounts.z-1.0f)<1.0e-6f,
                "actual owner accepted-audit path preserves slave area/resultants");


        // Compare the sparse owner to the preserved pre-change GPU owner.
        // Unsorted slave IDs, inactive gaps, nonzero incoming loads and three
        // environments expose indexing/overwrite errors that one node cannot.
        constexpr unsigned envs=3u, nodeCount=17u, sampleCount=5u;
        NMNumiHumanTendonFEMLoadDispatchGPU sparseDispatch=dispatch;
        sparseDispatch.environmentCount=envs;
        sparseDispatch.femNodeCount=nodeCount;
        sparseDispatch.bodyPoseStride=2u;
        sparseDispatch.femBodyContactSampleCount=sampleCount;
        sparseDispatch.femBodyContactCandidateTriangleCount=0u;
        std::array<NMNumiHumanFEMBodyContactSampleGPU,sampleCount> sparseSamples{};
        const std::array<unsigned,sampleCount> slaveIds{{12u,1u,16u,4u,9u}};
        for (unsigned i=0;i<sampleCount;++i) {
            sparseSamples[i]=samples[i%3u==0u ? 5u : (i%3u==1u ? 6u : 1u)];
            sparseSamples[i].slaveNode=slaveIds[i];
            sparseSamples[i].bodyIndex=1u;
        }
        std::array<NMFEMNodeStateGPU,envs*nodeCount> sparseNodes{};
        std::array<MRArticulatedBodyPoseGPU,envs*2u> sparsePoses{};
        std::array<nm_float4,envs*nodeCount> initialLoads{};
        for (unsigned e=0;e<envs;++e) {
            for (unsigned b=0;b<2u;++b) {
                sparsePoses[e*2u+b].position={0,0,0,0};
                sparsePoses[e*2u+b].orientation={0,0,0,1};
            }
            for (unsigned n=0;n<nodeCount;++n) {
                const unsigned row=e*nodeCount+n;
                sparseNodes[row].positionAndMass=f4(.4f,.4f,
                    .01f*float(int(n%5u)-2)-.004f*float(e),1);
                initialLoads[row]=f4(.125f*float(n),-.25f*float(e),.03125f,77.0f);
            }
        }
        auto sdBuffer=dispatchBuffer(&sparseDispatch,sizeof(sparseDispatch));
        auto ssBuffer=dispatchBuffer(sparseSamples.data(),sizeof(sparseSamples));
        auto snBuffer=dispatchBuffer(sparseNodes.data(),sizeof(sparseNodes));
        auto spBuffer=dispatchBuffer(sparsePoses.data(),sizeof(sparsePoses));
        auto sparseLoads=dispatchBuffer(initialLoads.data(),sizeof(initialLoads));
        auto denseLoads=dispatchBuffer(initialLoads.data(),sizeof(initialLoads));
        id<MTLCommandBuffer> parityCommand=[queue commandBuffer];
        auto parityEncode=[&](NSString* name,id<MTLBuffer> result,unsigned threads) {
            NSError* error=nil;
            auto fn=[library newFunctionWithName:[@"numi_matter_metal::" stringByAppendingString:name]];
            auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
            require(pipeline!=nil,"sparse/dense oracle pipelines");
            auto encoder=[parityCommand computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:sdBuffer offset:0 atIndex:0];
            [encoder setBuffer:ssBuffer offset:0 atIndex:1];
            [encoder setBuffer:snBuffer offset:0 atIndex:2];
            [encoder setBuffer:spBuffer offset:0 atIndex:3];
            [encoder setBuffer:result offset:0 atIndex:4];
            [encoder setBuffer:ownerRangeBuffer offset:0 atIndex:5];
            [encoder setBuffer:ownerCandidateBuffer offset:0 atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(threads,1,1)
                threadsPerThreadgroup:MTLSizeMake(8,1,1)];
            [encoder endEncoding];
        };
        parityEncode(@"nm_numi_human_assemble_fem_body_contact_loads",sparseLoads,envs*sampleCount);
        parityEncode(@"nm_numi_human_assemble_fem_body_contact_loads_dense_oracle",denseLoads,envs*nodeCount);
        [parityCommand commit];[parityCommand waitUntilCompleted];
        require(parityCommand.status==MTLCommandBufferStatusCompleted,"sparse/dense oracle completes");
        const auto* sparseResult=static_cast<const nm_float4*>(sparseLoads.contents);
        const auto* denseResult=static_cast<const nm_float4*>(denseLoads.contents);
        unsigned changedNodes=0;
        for (unsigned row=0;row<envs*nodeCount;++row) {
            for (unsigned axis=0;axis<4u;++axis) {
                const auto* a=reinterpret_cast<const float*>(&sparseResult[row]);
                const auto* b=reinterpret_cast<const float*>(&denseResult[row]);
                require(std::isfinite(a[axis]) && std::abs(a[axis]-b[axis])<=1.0e-6f,
                    "sparse force matches old GPU kernel including untouched rows and w");
            }
            changedNodes+=sparseResult[row].x!=initialLoads[row].x ||
                sparseResult[row].y!=initialLoads[row].y || sparseResult[row].z!=initialLoads[row].z;
        }
        require(changedNodes>=envs,"oracle must exercise actual nonzero contact/attachment forces");
        std::printf("sparse_dense_gpu_parity=PASS environments=%u nodes=%u contacts=%u changed=%u\n",
                    envs,nodeCount,sampleCount,changedNodes);


        // Two articulated bodies may now each contribute one triangle query
        // to the same slave. The grouped sparse owner must match the preserved
        // dense GPU oracle, keep body wrenches balanced, and pass the
        // force/moment audit across multiple environments.
        constexpr unsigned groupEnvs=3u, groupNodeCount=7u, groupSampleCount=3u;
        NMNumiHumanTendonFEMLoadDispatchGPU groupDispatch=dispatch;
        groupDispatch.environmentCount=groupEnvs;
        groupDispatch.femNodeCount=groupNodeCount;
        groupDispatch.bodyPoseStride=2u;
        groupDispatch.articulationFirstBody=0u;
        groupDispatch.femBodyContactSampleCount=groupSampleCount;
        groupDispatch.femBodyContactCandidateTriangleCount=groupSampleCount;
        groupDispatch.femSurfaceAttachmentSampleCount=0u;
        const std::array<Vec3,3> groupTri{{{0,0,0},{2,0,0},{0,2,0}}};
        std::array<NMNumiHumanFEMBodyContactSampleGPU,groupSampleCount>
            originalGroupSamples{{
                makeSample(3u,0u,groupTri[0],groupTri[1],groupTri[2],
                           0.25f,1000.0f,1.0e-6f),
                makeSample(5u,1u,groupTri[0],groupTri[1],groupTri[2],
                           0.25f,1000.0f,1.0e-6f),
                makeSample(3u,1u,groupTri[0],groupTri[1],groupTri[2],
                           0.25f,1000.0f,1.0e-6f)}};
        std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,
                   groupSampleCount> originalGroupRanges{};
        std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,
                   groupSampleCount> groupTriangles{};
        for (unsigned i=0u;i<groupSampleCount;++i) {
            originalGroupRanges[i].firstTriangle=i;
            originalGroupRanges[i].triangleCount=1u;
            originalGroupRanges[i].maximumDistanceMeters=0.05f;
            const float z=0.0f;
            groupTriangles[i].bodyLocalVertex0={0,0,z,0};
            groupTriangles[i].bodyLocalVertex1={2,0,z,0};
            groupTriangles[i].bodyLocalVertex2={0,2,z,0};
            groupTriangles[i].authoredOutwardNormal={0,0,1,0};
            groupTriangles[i].bodyIndex=originalGroupSamples[i].bodyIndex;
            groupTriangles[i].boneStableId=100u+i;
            groupTriangles[i].triangleStableId=200u+i;
        }
        std::array<NMNumiHumanTendonFEMNodeAnchorGPU,groupNodeCount>
            groupAnchors{};
        const auto groupAdmission =
            numi::matter::fem_body_contact_grouping::
                validateDistinctBodyTriangleSlaves(
                    originalGroupSamples,groupAnchors);
        require(groupAdmission.accepted() &&
                groupAdmission.hasDistinctBodyTriangleDuplicates,
                "two distinct-body triangle contacts pass CPU admission");
        std::vector<NMNumiHumanFEMBodyContactSampleGPU> groupedSamples;
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> groupedRanges;
        std::vector<std::uint32_t> groupCounts;
        require(numi::matter::fem_body_contact_grouping::stableGroupRowsBySlave(
                    originalGroupSamples,originalGroupRanges,
                    groupedSamples,groupedRanges,groupCounts) &&
                groupedSamples.size()==groupSampleCount &&
                groupCounts==std::vector<std::uint32_t>({2u,0u,1u}) &&
                groupedSamples[0].slaveNode==3u &&
                groupedSamples[0].bodyIndex==0u &&
                groupedSamples[1].slaveNode==3u &&
                groupedSamples[1].bodyIndex==1u &&
                groupedRanges[0].firstTriangle==0u &&
                groupedRanges[1].firstTriangle==2u,
                "grouped GPU rows keep deterministic per-slave source order and original triangle offsets");
        std::array<NMFEMNodeStateGPU,groupEnvs*groupNodeCount> groupNodes{};
        std::array<MRArticulatedBodyPoseGPU,groupEnvs*2u> groupPoses{};
        std::array<nm_float4,groupEnvs*groupNodeCount> groupInitialLoads{};
        std::array<NMNumiHumanBodyWrenchGPU,groupEnvs*2u> groupWrenches{};
        std::array<NMNumiHumanArticularContactAuditGPU,groupEnvs> groupAudits{};
        for (unsigned e=0u;e<groupEnvs;++e) {
            for (unsigned b=0u;b<2u;++b) {
                auto& pose=groupPoses[e*2u+b];
                pose.position={b==0u?0.0f:0.1f,0.0f,b==0u?0.0f:-0.005f,0.0f};
                pose.orientation={0,0,0,1};
            }
            for (unsigned n=0u;n<groupNodeCount;++n) {
                const unsigned row=e*groupNodeCount+n;
                groupNodes[row].positionAndMass=f4(
                    n==5u?0.5f:0.4f,0.4f,
                    n==3u?-0.02f:(n==5u?-0.03f:0.2f),1.0f);
                groupNodes[row].velocityAndInverseMass=f4(0,0,0,1);
                groupInitialLoads[row]=f4(0.01f*float(n),
                    -0.02f*float(e),0.03f,float(70u+n));
            }
        }
        auto groupDispatchBuffer=dispatchBuffer(&groupDispatch,sizeof(groupDispatch));
        auto originalRangeBuffer=dispatchBuffer(
            originalGroupRanges.data(),sizeof(originalGroupRanges));
        auto originalGroupSampleBuffer=dispatchBuffer(
            originalGroupSamples.data(),sizeof(originalGroupSamples));
        auto groupedSampleBuffer=dispatchBuffer(
            groupedSamples.data(),groupedSamples.size()*sizeof(groupedSamples[0]));
        auto groupedRangeBuffer=dispatchBuffer(
            groupedRanges.data(),groupedRanges.size()*sizeof(groupedRanges[0]));
        auto groupCountBuffer=dispatchBuffer(
            groupCounts.data(),groupCounts.size()*sizeof(groupCounts[0]));
        auto groupNodeBuffer=dispatchBuffer(groupNodes.data(),sizeof(groupNodes));
        auto groupPoseBuffer=dispatchBuffer(groupPoses.data(),sizeof(groupPoses));
        auto groupTriangleBuffer=dispatchBuffer(
            groupTriangles.data(),sizeof(groupTriangles));
        const NMNumiHumanFEMBodyContactBVHNodeGPU emptyGroupBVHNode{};
        id<MTLBuffer> groupBVHNodeBuffer=dispatchBuffer(
            &emptyGroupBVHNode,sizeof(emptyGroupBVHNode));
        auto groupedLoadBuffer=dispatchBuffer(
            groupInitialLoads.data(),sizeof(groupInitialLoads));
        auto denseGroupLoadBuffer=dispatchBuffer(
            groupInitialLoads.data(),sizeof(groupInitialLoads));
        auto groupWrenchBuffer=dispatchBuffer(groupWrenches.data(),sizeof(groupWrenches));
        auto groupAuditBuffer=dispatchBuffer(groupAudits.data(),sizeof(groupAudits));
        require(groupDispatchBuffer && originalGroupSampleBuffer &&
                groupedSampleBuffer && groupedRangeBuffer && groupCountBuffer &&
                groupNodeBuffer && groupPoseBuffer && groupTriangleBuffer &&
                groupedLoadBuffer && denseGroupLoadBuffer && groupWrenchBuffer &&
                groupAuditBuffer,"multi-master GPU test buffers");
        auto makePipeline=[&](NSString* name) {
            auto fn=[library newFunctionWithName:
                [@"numi_matter_metal::" stringByAppendingString:name]];
            require(fn!=nil,"multi-master kernel exists");
            NSError* pipelineError=nil;
            auto result=[device newComputePipelineStateWithFunction:fn
                                                               error:&pipelineError];
            if (result==nil) {
                std::fprintf(stderr,"multi-master pipeline %s: %s\n",
                    name.UTF8String,pipelineError.localizedDescription.UTF8String);
                std::exit(1);
            }
            return result;
        };
        auto groupedPipeline=makePipeline(
            @"nm_numi_human_assemble_fem_body_contact_loads_grouped");
        auto densePipeline=makePipeline(
            @"nm_numi_human_assemble_fem_body_contact_loads_dense_oracle");
        auto wrenchPipeline=makePipeline(
            @"nm_numi_human_assemble_fem_body_contact_wrenches");
        auto auditPipeline=makePipeline(
            @"nm_numi_human_audit_fem_body_contact");
        id<MTLCommandBuffer> groupCommand=[queue commandBuffer];
        auto encodeGroupKernel=[&](id<MTLComputePipelineState> groupPipeline,
                                   unsigned threads,
                                   bool grouped, id<MTLBuffer> result) {
            auto encoder=[groupCommand computeCommandEncoder];
            [encoder setComputePipelineState:groupPipeline];
            [encoder setBuffer:groupDispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:(grouped?groupedSampleBuffer:originalGroupSampleBuffer)
                        offset:0 atIndex:1];
            [encoder setBuffer:groupNodeBuffer offset:0 atIndex:2];
            [encoder setBuffer:groupPoseBuffer offset:0 atIndex:3];
            [encoder setBuffer:result offset:0 atIndex:4];
            [encoder setBuffer:(grouped?groupedRangeBuffer:originalRangeBuffer)
                        offset:0 atIndex:5];
            [encoder setBuffer:groupTriangleBuffer offset:0 atIndex:6];
            [encoder setBuffer:groupBVHNodeBuffer offset:0
                       atIndex:(grouped?8u:7u)];
            if (grouped) [encoder setBuffer:groupCountBuffer offset:0 atIndex:9];
            [encoder dispatchThreads:MTLSizeMake(threads,1,1)
                threadsPerThreadgroup:MTLSizeMake(8,1,1)];
            [encoder endEncoding];
        };
        encodeGroupKernel(groupedPipeline,groupEnvs*groupSampleCount,true,
                          groupedLoadBuffer);
        encodeGroupKernel(densePipeline,groupEnvs*groupNodeCount,false,
                          denseGroupLoadBuffer);
        auto wrenchEncoder=[groupCommand computeCommandEncoder];
        [wrenchEncoder setComputePipelineState:wrenchPipeline];
        [wrenchEncoder setBuffer:groupDispatchBuffer offset:0 atIndex:0];
        [wrenchEncoder setBuffer:originalGroupSampleBuffer offset:0 atIndex:1];
        [wrenchEncoder setBuffer:groupNodeBuffer offset:0 atIndex:2];
        [wrenchEncoder setBuffer:groupPoseBuffer offset:0 atIndex:3];
        [wrenchEncoder setBuffer:groupWrenchBuffer offset:0 atIndex:4];
        [wrenchEncoder setBuffer:originalRangeBuffer offset:0 atIndex:5];
        [wrenchEncoder setBuffer:groupTriangleBuffer offset:0 atIndex:6];
        [wrenchEncoder setBuffer:groupBVHNodeBuffer offset:0 atIndex:7];
        [wrenchEncoder dispatchThreads:MTLSizeMake(groupEnvs*2u,1,1)
            threadsPerThreadgroup:MTLSizeMake(2,1,1)];
        [wrenchEncoder endEncoding];
        auto auditEncoder=[groupCommand computeCommandEncoder];
        [auditEncoder setComputePipelineState:auditPipeline];
        [auditEncoder setBuffer:groupDispatchBuffer offset:0 atIndex:0];
        [auditEncoder setBuffer:originalGroupSampleBuffer offset:0 atIndex:1];
        [auditEncoder setBuffer:groupNodeBuffer offset:0 atIndex:2];
        [auditEncoder setBuffer:groupPoseBuffer offset:0 atIndex:3];
        [auditEncoder setBuffer:groupAuditBuffer offset:0 atIndex:4];
        const NMNumiHumanFEMSurfaceAttachmentGPU emptyAttachment{};
        [auditEncoder setBytes:&emptyAttachment length:sizeof(emptyAttachment) atIndex:5];
        [auditEncoder setBuffer:originalRangeBuffer offset:0 atIndex:6];
        [auditEncoder setBuffer:groupTriangleBuffer offset:0 atIndex:7];
        [auditEncoder setBuffer:groupBVHNodeBuffer offset:0 atIndex:8];
        [auditEncoder dispatchThreads:MTLSizeMake(groupEnvs,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [auditEncoder endEncoding];
        [groupCommand commit];[groupCommand waitUntilCompleted];
        require(groupCommand.status==MTLCommandBufferStatusCompleted,
                "multi-master grouped force, dense oracle, wrench and audit complete");
        const auto* groupedResult=
            static_cast<const nm_float4*>(groupedLoadBuffer.contents);
        const auto* denseGroupResult=
            static_cast<const nm_float4*>(denseGroupLoadBuffer.contents);
        for (unsigned row=0u;row<groupEnvs*groupNodeCount;++row) {
            const auto* a=reinterpret_cast<const float*>(&groupedResult[row]);
            const auto* b=reinterpret_cast<const float*>(&denseGroupResult[row]);
            for (unsigned axis=0u;axis<4u;++axis)
                require(std::isfinite(a[axis]) && a[axis]==b[axis],
                        "grouped two-body force matches dense oracle bitwise on all rows");
            if (row%groupNodeCount!=3u && row%groupNodeCount!=5u) {
                require(std::memcmp(&groupedResult[row],&groupInitialLoads[row],
                            sizeof(nm_float4))==0,
                        "multi-master path preserves untouched loads exactly");
            }
        }
        const auto* actualGroupWrenches=
            static_cast<const NMNumiHumanBodyWrenchGPU*>(groupWrenchBuffer.contents);
        const auto* actualGroupAudits=
            static_cast<const NMNumiHumanArticularContactAuditGPU*>(groupAuditBuffer.contents);
        const auto maxAbsVec3=[](Vec3 v) {
            return std::max({std::abs(v.x),std::abs(v.y),std::abs(v.z)});
        };
        for (unsigned e=0u;e<groupEnvs;++e) {
            const unsigned row3=e*groupNodeCount+3u;
            const unsigned row5=e*groupNodeCount+5u;
            const Vec3 f3{
                groupedResult[row3].x-groupInitialLoads[row3].x,
                groupedResult[row3].y-groupInitialLoads[row3].y,
                groupedResult[row3].z-groupInitialLoads[row3].z};
            const Vec3 f5{
                groupedResult[row5].x-groupInitialLoads[row5].x,
                groupedResult[row5].y-groupInitialLoads[row5].y,
                groupedResult[row5].z-groupInitialLoads[row5].z};
            const auto body0=actualGroupWrenches[e*2u];
            const auto body1=actualGroupWrenches[e*2u+1u];
            const Vec3 netForce{f3.x+f5.x+body0.force.x+body1.force.x,
                                f3.y+f5.y+body0.force.y+body1.force.y,
                                f3.z+f5.z+body0.force.z+body1.force.z};
            require(maxAbsVec3(netForce)<1.0e-5,
                    "two-body FEM loads and body wrenches conserve total force");
            require(std::abs(body0.force.z)>0.0f &&
                    std::abs(body1.force.z)>0.0f,
                    "both distinct master bodies receive reaction wrench");
            const Vec3 node3{groupNodes[row3].positionAndMass.x,
                             groupNodes[row3].positionAndMass.y,
                             groupNodes[row3].positionAndMass.z};
            const Vec3 node5{groupNodes[row5].positionAndMass.x,
                             groupNodes[row5].positionAndMass.y,
                             groupNodes[row5].positionAndMass.z};
            const Vec3 body1Origin{0.1,0.0,-0.005};
            const Vec3 moment = add(
                add(cross(node3,f3),cross(node5,f5)),
                add(add(Vec3{body0.torque.x,body0.torque.y,body0.torque.z},
                        Vec3{body1.torque.x,body1.torque.y,body1.torque.z}),
                    cross(body1Origin,Vec3{body1.force.x,body1.force.y,body1.force.z})));
            require(maxAbsVec3(moment)<2.0e-5,
                    "two-body FEM loads and body wrenches conserve total moment");
            require(std::abs(actualGroupAudits[e].forceResidualAndL1.x)<1.0e-5f &&
                    std::abs(actualGroupAudits[e].forceResidualAndL1.y)<1.0e-5f &&
                    std::abs(actualGroupAudits[e].forceResidualAndL1.z)<1.0e-5f &&
                    std::abs(actualGroupAudits[e].normalForceAreaAndCounts.y-0.75f)<1.0e-6f &&
                    std::abs(actualGroupAudits[e].normalForceAreaAndCounts.z-3.0f)<1.0e-6f,
                    "multi-master contact audit includes both body contacts and unique row");
        }
        std::printf("multi_master_grouped_gpu=PASS envs=%u samples=%u node_wrenches=2 oracle=bitwise\n",
                    groupEnvs,groupSampleCount);


        // Couple two moving rigid masters to one FEM slave through the
        // production grouped-force, original-row wrench, and generalized-load
        // kernels; validate the shared contact tangent by master-wise FD.
        constexpr unsigned ccDofCount=12u;
        NMNumiHumanTendonFEMLoadDispatchGPU ccDispatch=groupDispatch;
        ccDispatch.environmentCount=1u;
        ccDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        ccDispatch.dofCount=ccDofCount;
        ccDispatch.reserved0=ccDofCount;
        ccDispatch.pointJacobianStride=2u*4u*3u*ccDofCount;
        ccDispatch.bodyJacobianPointOffset=0u;
        auto ccDispatchBuffer=dispatchBuffer(&ccDispatch,sizeof(ccDispatch));
        std::array<float,2u*4u*3u*ccDofCount> ccJacobians{};
        for (unsigned body=0u;body<2u;++body) {
            const unsigned q=6u*body, base=body*4u*3u*ccDofCount;
            // Angular extraction uses every axis probe relative to the center.
            // Translation must therefore be identical at center and all probes.
            for (unsigned point=0u;point<4u;++point)
                for (unsigned axis=0u;axis<3u;++axis)
                    ccJacobians[base+point*3u*ccDofCount+
                        axis*ccDofCount+q+axis]=1.0f;
            // Encode each angular basis as omega x e_i at all three probes.
            for (unsigned rotation=0u;rotation<3u;++rotation) {
                const Vec3 omega=rotation==0u?Vec3{1,0,0}:
                    (rotation==1u?Vec3{0,1,0}:Vec3{0,0,1});
                for (unsigned probe=0u;probe<3u;++probe) {
                    const Vec3 basis=probe==0u?Vec3{1,0,0}:
                        (probe==1u?Vec3{0,1,0}:Vec3{0,0,1});
                    const Vec3 derivative=cross(omega,basis);
                    const float values[3]={float(derivative.x),
                        float(derivative.y),float(derivative.z)};
                    for (unsigned component=0u;component<3u;++component)
                        ccJacobians[base+(probe+1u)*3u*ccDofCount+
                            component*ccDofCount+q+3u+rotation]=values[component];
                }
            }
        }
        std::array<float,ccDofCount> ccSourceQ{},ccPendingQ{},ccPendingSourceQ{};
        auto ccJacobianBuffer=dispatchBuffer(ccJacobians.data(),sizeof(ccJacobians));
        auto ccSourceQBuffer=dispatchBuffer(ccSourceQ.data(),sizeof(ccSourceQ));
        auto ccPendingQBuffer=dispatchBuffer(ccPendingQ.data(),sizeof(ccPendingQ));
        auto ccPendingSourceQBuffer=dispatchBuffer(
            ccPendingSourceQ.data(),sizeof(ccPendingSourceQ));
        auto ccWrenchBuffer=dispatchBuffer(
            groupWrenches.data(),2u*sizeof(NMNumiHumanBodyWrenchGPU));
        auto ccGroupedPipeline=makePipeline(
            @"nm_numi_human_assemble_fem_body_contact_loads_grouped");
        auto ccWrenchPipeline=makePipeline(
            @"nm_numi_human_assemble_fem_body_contact_wrenches");
        auto ccGeneralizedPipeline=makePipeline(
            @"nm_numi_human_prepare_fem_body_generalized_load");
        require(ccDispatchBuffer && ccJacobianBuffer && ccSourceQBuffer &&
                ccPendingQBuffer && ccPendingSourceQBuffer && ccWrenchBuffer,
                "two-master generalized-load regression buffers");
        struct CoupledResult {
            std::array<nm_float4,groupNodeCount> loads{};
            std::array<NMNumiHumanBodyWrenchGPU,2u> wrenches{};
            std::array<float,ccDofCount> q{};
        };
        auto runCoupled=[&](std::array<MRArticulatedBodyPoseGPU,2u> poses) {
            std::memcpy(groupPoseBuffer.contents,poses.data(),sizeof(poses));
            std::memcpy(groupedLoadBuffer.contents,groupInitialLoads.data(),
                        sizeof(groupInitialLoads));
            auto command=[queue commandBuffer];
            auto force=[command computeCommandEncoder];
            [force setComputePipelineState:ccGroupedPipeline];
            [force setBuffer:ccDispatchBuffer offset:0 atIndex:0];
            [force setBuffer:groupedSampleBuffer offset:0 atIndex:1];
            [force setBuffer:groupNodeBuffer offset:0 atIndex:2];
            [force setBuffer:groupPoseBuffer offset:0 atIndex:3];
            [force setBuffer:groupedLoadBuffer offset:0 atIndex:4];
            [force setBuffer:groupedRangeBuffer offset:0 atIndex:5];
            [force setBuffer:groupTriangleBuffer offset:0 atIndex:6];
            [force setBuffer:groupBVHNodeBuffer offset:0 atIndex:8];
            [force setBuffer:groupCountBuffer offset:0 atIndex:9];
            [force dispatchThreads:MTLSizeMake(groupSampleCount,1,1)
                threadsPerThreadgroup:MTLSizeMake(4,1,1)];
            [force endEncoding];
            auto wrench=[command computeCommandEncoder];
            [wrench setComputePipelineState:ccWrenchPipeline];
            [wrench setBuffer:ccDispatchBuffer offset:0 atIndex:0];
            [wrench setBuffer:originalGroupSampleBuffer offset:0 atIndex:1];
            [wrench setBuffer:groupNodeBuffer offset:0 atIndex:2];
            [wrench setBuffer:groupPoseBuffer offset:0 atIndex:3];
            [wrench setBuffer:ccWrenchBuffer offset:0 atIndex:4];
            [wrench setBuffer:originalRangeBuffer offset:0 atIndex:5];
            [wrench setBuffer:groupTriangleBuffer offset:0 atIndex:6];
            [wrench setBuffer:groupBVHNodeBuffer offset:0 atIndex:7];
            [wrench dispatchThreads:MTLSizeMake(2u,1,1)
                threadsPerThreadgroup:MTLSizeMake(2,1,1)];
            [wrench endEncoding];
            auto generalized=[command computeCommandEncoder];
            [generalized setComputePipelineState:ccGeneralizedPipeline];
            [generalized setBuffer:ccDispatchBuffer offset:0 atIndex:0];
            [generalized setBuffer:ccWrenchBuffer offset:0 atIndex:1];
            [generalized setBuffer:groupPoseBuffer offset:0 atIndex:2];
            [generalized setBuffer:ccJacobianBuffer offset:0 atIndex:3];
            [generalized setBuffer:ccSourceQBuffer offset:0 atIndex:4];
            [generalized setBuffer:ccPendingQBuffer offset:0 atIndex:5];
            [generalized setBuffer:ccPendingSourceQBuffer offset:0 atIndex:6];
            [generalized dispatchThreads:MTLSizeMake(ccDofCount,1,1)
                threadsPerThreadgroup:MTLSizeMake(8,1,1)];
            [generalized endEncoding];
            [command commit];[command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,
                    "grouped contact, source-row wrenches and q contraction complete");
            CoupledResult result{};
            std::memcpy(result.loads.data(),groupedLoadBuffer.contents,
                        sizeof(result.loads));
            std::memcpy(result.wrenches.data(),ccWrenchBuffer.contents,
                        sizeof(result.wrenches));
            std::memcpy(result.q.data(),ccPendingQBuffer.contents,sizeof(result.q));
            return result;
        };
        std::array<MRArticulatedBodyPoseGPU,2u> ccBasePoses{{
            groupPoses[0],groupPoses[1]}};
        const auto ccBase=runCoupled(ccBasePoses);
        for (unsigned body=0u;body<2u;++body) {
            const unsigned q=6u*body;
            const auto w=ccBase.wrenches[body];
            const std::array<float,6u> expected{{
                w.force.x,w.force.y,w.force.z,w.torque.x,w.torque.y,w.torque.z}};
            for (unsigned axis=0u;axis<6u;++axis) {
                if (!(std::abs(ccBase.q[q+axis]-expected[axis])<2.0e-5f)) {
                    std::fprintf(stderr,"cc q mismatch body=%u axis=%u actual=%.9g expected=%.9g force=(%.9g,%.9g,%.9g) torque=(%.9g,%.9g,%.9g)\n",
                        body,axis,ccBase.q[q+axis],expected[axis],
                        w.force.x,w.force.y,w.force.z,w.torque.x,w.torque.y,w.torque.z);
                }
                require(std::abs(ccBase.q[q+axis]-expected[axis])<2.0e-5f,
                    "production q contraction uses each original master wrench row and Jacobian block");
            }
        }

        // The coupon state entrypoint exposes each row's production K. Rows
        // 0 and 2 share slave 3 but belong to bodies 0 and 1 respectively.
        std::array<NMFEMNodeStateGPU,groupSampleCount> ccTangentNodes{{
            groupNodes[3u],groupNodes[5u],groupNodes[3u]}};
        std::array<MRArticulatedBodyPoseGPU,groupSampleCount> ccTangentPoses{{
            ccBasePoses[0],ccBasePoses[1],ccBasePoses[1]}};
        std::array<nm_float4,groupSampleCount*8u> ccTangentOut{};
        auto ccTangentSamples=dispatchBuffer(
            originalGroupSamples.data(),sizeof(originalGroupSamples));
        auto ccTangentNodeBuffer=dispatchBuffer(
            ccTangentNodes.data(),sizeof(ccTangentNodes));
        auto ccTangentPoseBuffer=dispatchBuffer(
            ccTangentPoses.data(),sizeof(ccTangentPoses));
        auto ccTangentBuffer=dispatchBuffer(ccTangentOut.data(),sizeof(ccTangentOut));
        auto tangentCommand=[queue commandBuffer];
        auto tangentEncoder=[tangentCommand computeCommandEncoder];
        [tangentEncoder setComputePipelineState:pipeline];
        [tangentEncoder setBuffer:ccTangentSamples offset:0 atIndex:0];
        [tangentEncoder setBuffer:ccTangentNodeBuffer offset:0 atIndex:1];
        [tangentEncoder setBuffer:ccTangentPoseBuffer offset:0 atIndex:2];
        [tangentEncoder setBuffer:ccTangentBuffer offset:0 atIndex:3];
        [tangentEncoder dispatchThreads:MTLSizeMake(groupSampleCount,1,1)
            threadsPerThreadgroup:MTLSizeMake(4,1,1)];
        [tangentEncoder endEncoding];[tangentCommand commit];
        [tangentCommand waitUntilCompleted];
        require(tangentCommand.status==MTLCommandBufferStatusCompleted,
                "production tangent states for three original rows complete");
        const auto* tangentRows=static_cast<const nm_float4*>(ccTangentBuffer.contents);
        const float kShared0=tangentRows[5u].z;
        const float kBody1Extra=tangentRows[8u+5u].z;
        const float kShared1=tangentRows[16u+5u].z;
        require(std::abs(kShared0-250.0f)<1.0e-3f &&
                std::abs(kShared1-250.0f)<1.0e-3f &&
                std::abs(kBody1Extra-250.0f)<1.0e-3f,
                "contact tangents remain distinct on original body/slave rows");

        constexpr float ccEps=1.0e-4f;
        for (unsigned body=0u;body<2u;++body) {
            auto plus=ccBasePoses,minus=ccBasePoses;
            plus[body].position.z+=ccEps;
            minus[body].position.z-=ccEps;
            const auto p=runCoupled(plus),m=runCoupled(minus);
            const float dSlave=(p.loads[3u].z-m.loads[3u].z)/(2.0f*ccEps);
            const float expectedK=body==0u?kShared0:kShared1;
            const float dQ=(p.q[6u*body+2u]-m.q[6u*body+2u])/
                (2.0f*ccEps);
            const float qK=body==0u?kShared0:(kShared1+kBody1Extra);
            require(std::abs(dSlave-expectedK)<0.08f,
                    "shared-slave grouped-force FD matches each moving-master tangent row");
            require(std::abs(dQ+qK)<0.08f,
                    "contracted generalized tangent matches body-specific original-row wrench sum");
        }
        auto bothPlus=ccBasePoses,bothMinus=ccBasePoses;
        for (unsigned body=0u;body<2u;++body) {
            bothPlus[body].position.z+=ccEps;
            bothMinus[body].position.z-=ccEps;
        }
        const auto bp=runCoupled(bothPlus),bm=runCoupled(bothMinus);
        const float dShared=(bp.loads[3u].z-bm.loads[3u].z)/(2.0f*ccEps);
        const float dQsum=(bp.q[2u]+bp.q[8u]-bm.q[2u]-bm.q[8u])/
            (2.0f*ccEps);
        require(std::abs(dShared-(kShared0+kShared1))<0.08f &&
                std::abs(dQsum+(kShared0+kShared1+kBody1Extra))<0.08f,
                "two moving masters sum shared-slave tangent and source-row q contraction");
        std::printf("multi_master_coupled_tangent_gpu=PASS masters=2 shared_slave=3\n");

        // Strict closed convex and concave source banks feed the production
        // feature-aware triangle state. Face/edge/vertex signs and active
        // force/tangent finite differences are checked on the GPU.
        id<MTLFunction> pseudoFunction=[library
            newFunctionWithName:@"nm_numi_human_closed_pseudonormal_state_coupon"];
        require(pseudoFunction!=nil,"closed pseudonormal state kernel");
        id<MTLComputePipelineState> pseudoPipeline=
            [device newComputePipelineStateWithFunction:pseudoFunction error:&error];
        require(pseudoPipeline!=nil,"closed pseudonormal state pipeline");
        auto makeOutwardTriangle=[](std::uint32_t body,std::uint32_t bone,
                                    std::uint32_t id,Vec3 a,Vec3 b,Vec3 c) {
            NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU t{};
            t.bodyIndex=body;t.boneStableId=bone;t.triangleStableId=id;
            t.bodyLocalVertex0=f4(float(a.x),float(a.y),float(a.z),0.0f);
            t.bodyLocalVertex1=f4(float(b.x),float(b.y),float(b.z),0.0f);
            t.bodyLocalVertex2=f4(float(c.x),float(c.y),float(c.z),0.0f);
            Vec3 n{};require(scaledFaceNormal(a,b,c,n),"nonzero coupon face normal");
            t.authoredOutwardNormal=f4(float(n.x),float(n.y),float(n.z),0.0f);
            return t;
        };
        const std::array<Vec3,4> tetraVertices{{
            {0,0,0},{1,0,0},{0,1,0},{0,0,1}}};
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> closedInput;
        closedInput.push_back(makeOutwardTriangle(31u,7u,0u,tetraVertices[0],tetraVertices[2],tetraVertices[1]));
        closedInput.push_back(makeOutwardTriangle(31u,7u,1u,tetraVertices[0],tetraVertices[3],tetraVertices[2]));
        closedInput.push_back(makeOutwardTriangle(31u,7u,2u,tetraVertices[0],tetraVertices[1],tetraVertices[3]));
        closedInput.push_back(makeOutwardTriangle(31u,7u,3u,tetraVertices[1],tetraVertices[2],tetraVertices[3]));
        constexpr std::array<std::array<double,2>,7> polygon{{
            {{0,0}},{{2,0}},{{2,1}},{{1,1}},{{1,2}},{{0,2}},{{0,1}}}};
        std::vector<unsigned> ring{0u,1u,2u,3u,4u,5u,6u};
        std::vector<std::array<unsigned,3>> caps;
        const auto cross2=[&](unsigned a,unsigned b,unsigned c) {
            const auto& A=polygon[a];const auto& B=polygon[b];const auto& C=polygon[c];
            return (B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0]);
        };
        while(ring.size()>3u) {
            bool clipped=false;
            for(std::size_t i=0;i<ring.size();++i) {
                const unsigned a=ring[(i+ring.size()-1u)%ring.size()];
                const unsigned b=ring[i],c=ring[(i+1u)%ring.size()];
                if(cross2(a,b,c)<=0.0)continue;
                bool contains=false;
                for(const unsigned q:ring)if(q!=a&&q!=b&&q!=c)
                    if(cross2(a,b,q)>=0.0&&cross2(b,c,q)>=0.0&&cross2(c,a,q)>=0.0)
                        {contains=true;break;}
                if(contains)continue;
                caps.push_back({a,b,c});
                ring.erase(ring.begin()+static_cast<std::ptrdiff_t>(i));
                clipped=true;break;
            }
            require(clipped,"concave L-prism cap triangulates");
        }
        caps.push_back({ring[0],ring[1],ring[2]});
        std::array<Vec3,14> prismVertices{};
        for(unsigned i=0;i<7u;++i) {
            prismVertices[i]={polygon[i][0],polygon[i][1],0.0};
            prismVertices[7u+i]={polygon[i][0],polygon[i][1],1.0};
        }
        std::uint32_t faceId=0u;
        for(const auto& f:caps) {
            closedInput.push_back(makeOutwardTriangle(32u,8u,faceId++,
                prismVertices[7u+f[0]],prismVertices[7u+f[1]],prismVertices[7u+f[2]]));
            closedInput.push_back(makeOutwardTriangle(32u,8u,faceId++,
                prismVertices[f[2]],prismVertices[f[1]],prismVertices[f[0]]));
        }
        for(unsigned i=0;i<7u;++i) {
            const unsigned j=(i+1u)%7u;
            closedInput.push_back(makeOutwardTriangle(32u,8u,faceId++,
                prismVertices[i],prismVertices[j],prismVertices[7u+j]));
            closedInput.push_back(makeOutwardTriangle(32u,8u,faceId++,
                prismVertices[i],prismVertices[7u+j],prismVertices[7u+i]));
        }
        const auto closed=numi::matter::fem_body_contact_bvh::
            buildStaticCompleteBodyBVH(closedInput,closedInput.size(),true);
        require(closed.succeeded(),"closed convex tetra and concave L-prism admit");
        for(const auto& t:closed.triangles)
            require(t.featureNormalFlags==NM_NUMI_HUMAN_FEM_BODY_CONTACT_ALL_FEATURE_PSEUDONORMALS,
                "strict banks emit all feature pseudonormals");

        struct Query {
            const char* label;std::uint32_t body;Vec3 point;Feature expected;
            int group;bool expectedAabbInactive;bool rotated;
        };
        std::vector<Query> queries;
        const auto addQuery=[&](const char* label,std::uint32_t body,Vec3 q,
                                Feature f,int group=-1,
                                bool expectedAabbInactive=false,bool rotated=false) {
            queries.push_back({label,body,q,f,group,expectedAabbInactive,rotated});
        };
        addQuery("convex-face-inside",31u,{0.2,0.2,0.58},Feature::face,0);
        addQuery("convex-edge-outside",31u,{0.5,-0.02,-0.02},Feature::edge);
        addQuery("convex-vertex-outside",31u,{-0.02,-0.02,-0.02},Feature::vertex);
        addQuery("convex-face-outside",31u,{0.2,0.2,0.62},Feature::face);
        addQuery("concave-face-inside",32u,{0.3,0.4,0.98},Feature::face,1);
        addQuery("concave-notch-face-outside",32u,{1.02,1.02,0.5},Feature::face);
        addQuery("concave-edge-inside",32u,{0.99,0.99,0.5},Feature::edge,2);
        addQuery("concave-notch-edge-outside",32u,{1.02,1.02,1.02},Feature::edge);
        addQuery("closed-aabb-face-outside",32u,{-0.2,0.5,0.5},
                 Feature::face,-1,true);
        addQuery("closed-aabb-edge-outside",32u,{-0.04,-0.04,0.5},
                 Feature::edge,-1,true);
        addQuery("closed-aabb-corner-outside-rotated",31u,{-0.2,-0.2,-0.2},
                 Feature::vertex,-1,true,true);
        addQuery("closed-aabb-boundary-ulp",32u,
                 {double(-std::numeric_limits<float>::denorm_min()),0.5,0.5},
                 Feature::face);
        addQuery("closed-aabb-band-boundary",32u,{-0.05,0.5,0.5},
                 Feature::face);
        addQuery("concave-outside-inside-aabb",32u,{1.02,1.02,0.0},
                 Feature::edge);
        constexpr double fdStep=1.0e-4;
        const std::array<Vec3,3> axes{{{1,0,0},{0,1,0},{0,0,1}}};
        const std::array<Query,3> bases{{queries[0],queries[4],queries[6]}};
        for(const auto& base:bases)for(unsigned axisIndex=0;axisIndex<3u;++axisIndex) {
            addQuery("fd-plus",base.body,add(base.point,mul(axes[axisIndex],fdStep)),
                     base.expected,base.group);
            addQuery("fd-minus",base.body,sub(base.point,mul(axes[axisIndex],fdStep)),
                     base.expected,base.group);
        }
        const auto choose=[&](std::uint32_t body,Vec3 q) {
            const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU* winner=nullptr;
            double best=std::numeric_limits<double>::infinity();
            std::uint32_t bestBone=UINT32_MAX,bestId=UINT32_MAX;
            for(const auto& t:closed.triangles)if(t.bodyIndex==body) {
                const auto cp=closestPointOnTriangle(q,read(t.bodyLocalVertex0),
                    read(t.bodyLocalVertex1),read(t.bodyLocalVertex2));
                const double d2=norm2(sub(q,cp.point));
                if(d2<best||(d2==best&&std::tie(t.boneStableId,t.triangleStableId)<
                                             std::tie(bestBone,bestId))) {
                    winner=&t;best=d2;bestBone=t.boneStableId;bestId=t.triangleStableId;
                }
            }
            require(winner!=nullptr,"complete source bank chooses query winner");
            return std::pair{*winner,closestPointOnTriangle(q,
                read(winner->bodyLocalVertex0),read(winner->bodyLocalVertex1),
                read(winner->bodyLocalVertex2))};
        };
        // Exercise the CPU mirror at the two previously broken boundaries:
        // legacy edge/vertex queries have zero flags and zero optional vectors,
        // while strict face queries have no face-bit (bit zero is reserved).
        auto legacyFeatureFixture=closed.triangles[0];
        legacyFeatureFixture.featureNormalFlags=0u;
        legacyFeatureFixture.edgePseudoNormal01={0,0,0,0};
        legacyFeatureFixture.edgePseudoNormal12={0,0,0,0};
        legacyFeatureFixture.edgePseudoNormal20={0,0,0,0};
        legacyFeatureFixture.vertexPseudoNormal0={0,0,0,0};
        legacyFeatureFixture.vertexPseudoNormal1={0,0,0,0};
        legacyFeatureFixture.vertexPseudoNormal2={0,0,0,0};
        ClosestPoint edgeFallback{};edgeFallback.feature=Feature::edge;
        edgeFallback.barycentric={0.5,0.5,0.0};
        Vec3 selectedNormal{};
        const Vec3 legacyFace=read(legacyFeatureFixture.authoredOutwardNormal);
        require(featureOrientationNormal(legacyFeatureFixture,edgeFallback,
                    legacyFace,selectedNormal) && norm(sub(selectedNormal,legacyFace))==0.0,
                "zero feature flags preserve legacy face-normal fallback at edges");
        ClosestPoint vertexFallback{};vertexFallback.feature=Feature::vertex;
        vertexFallback.barycentric={1.0,0.0,0.0};
        require(featureOrientationNormal(legacyFeatureFixture,vertexFallback,
                    legacyFace,selectedNormal) && norm(sub(selectedNormal,legacyFace))==0.0,
                "zero feature flags preserve legacy face-normal fallback at vertices");
        const Vec3 rotatedLegacyFace=quaternionRotate(rotatedQ,legacyFace);
        require(featureOrientationNormal(legacyFeatureFixture,edgeFallback,
                    rotatedLegacyFace,selectedNormal) &&
                norm(sub(selectedNormal,rotatedLegacyFace))==0.0,
                "zero feature flags preserve rotated legacy face-normal fallback");
        auto strictFaceFixture=closed.triangles[0];
        require(featureOrientationNormal(strictFaceFixture,ClosestPoint{},
                    read(strictFaceFixture.authoredOutwardNormal),selectedNormal),
                "strict face projection does not require a nonexistent face-bit");
        auto strictMissingEdge=closed.triangles[0];
        strictMissingEdge.featureNormalFlags &=
            ~NM_NUMI_HUMAN_FEM_BODY_CONTACT_EDGE_01_PSEUDONORMAL;
        require(!featureOrientationNormal(strictMissingEdge,edgeFallback,
                    read(strictMissingEdge.authoredOutwardNormal),selectedNormal),
                "strict edge projection rejects a missing feature bit");
        std::printf("feature_normal_cpu_contract=PASS legacy_edge_vertex_fallback strict_face_bitless strict_missing_edge_reject\n");

        struct Ref { Vec3 force{},point{};double closure=0.0;Matrix3 tangent{}; };
        const auto reference=[&](Vec3 q,const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU& t) {
            Ref r{};
            const auto cp=closestPointOnTriangle(q,read(t.bodyLocalVertex0),
                read(t.bodyLocalVertex1),read(t.bodyLocalVertex2));
            require(finite(cp.point),"CPU coupon closest point finite");
            Vec3 pseudo{};
            require(featureOrientationNormal(t,cp,read(t.authoredOutwardNormal),pseudo),
                    "CPU reference selects only the valid feature normal");
            const Vec3 delta=sub(q,cp.point);const double d=norm(delta);
            const double sign=dot(delta,pseudo)<0.0?-1.0:1.0;
            const Vec3 n=d>1.0e-12?mul(delta,sign/d):pseudo;
            r.point=cp.point;r.closure=std::max(0.0,sign<0.0?d-1.0e-7:-d-1.0e-7);
            constexpr double stiffness=5000.0,area=0.25;
            const double scale=stiffness*area;r.force=mul(n,scale*r.closure);
            if(r.closure>0.0) {
                for(unsigned row=0;row<3u;++row)for(unsigned col=0;col<3u;++col) {
                    const double nr=row==0?n.x:(row==1?n.y:n.z);
                    const double nc=col==0?n.x:(col==1?n.y:n.z);
                    double projection=0.0;
                    if(cp.feature==Feature::edge) {
                        const double er=row==0?cp.edgeAxis.x:(row==1?cp.edgeAxis.y:cp.edgeAxis.z);
                        const double ec=col==0?cp.edgeAxis.x:(col==1?cp.edgeAxis.y:cp.edgeAxis.z);
                        projection=er*ec;
                    }
                    const double identity=row==col?1.0:0.0;
                    const double curvature=d>1.0e-12?sign/std::max(d,1.0e-12):
                        1.0/std::max(d,1.0e-12);
                    const double h=cp.feature==Feature::face?0.0:
                        curvature*(identity-projection-nr*nc);
                    r.tangent[row][col]=scale*(nr*nc-r.closure*h);
                }
            }
            return std::pair{r,cp.feature};
        };
        const std::size_t n=queries.size();
        std::vector<NMNumiHumanFEMBodyContactSampleGPU> ps(n);
        std::vector<NMFEMNodeStateGPU> pn(n);
        std::vector<MRArticulatedBodyPoseGPU> pp(n);
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges(n);
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> pt(n);
        std::vector<Ref> pr(n);
        std::vector<Vec3> worldQueries(n);
        std::array<std::uint32_t,3> baseIndex{{UINT32_MAX,UINT32_MAX,UINT32_MAX}};
        std::array<std::uint32_t,9> plusIndex{},minusIndex{};
        std::array<unsigned,3> counts{};
        for(std::size_t i=0;i<n;++i) {
            const auto& q=queries[i];
            const Vec3 qf{double(float(q.point.x)),double(float(q.point.y)),double(float(q.point.z))};
            const auto [t,cp]=choose(q.body,qf);
            if(cp.feature!=q.expected) {
                std::fprintf(stderr,
                    "feature_witness label=%s body=%u expected=%u actual=%u point=(%.9g,%.9g,%.9g) triangle=%u\n",
                    q.label,q.body,unsigned(q.expected),unsigned(cp.feature),
                    qf.x,qf.y,qf.z,t.triangleStableId);
            }
            require(cp.feature==q.expected,"CPU complete-bank winner has expected face/edge/vertex feature");
            pt[i]=t;
            ps[i]=makeSample(std::uint32_t(i),q.body,read(t.bodyLocalVertex0),
                read(t.bodyLocalVertex1),read(t.bodyLocalVertex2),0.25f,5000.0f,1.0e-6f);
            const Vec3 origin=q.rotated?rotatedOrigin:Vec3{0,0,0};
            const auto orientation=q.rotated?rotatedQ:std::array<double,4>{{0,0,0,1}};
            const Vec3 worldQ=add(origin,quaternionRotate(orientation,qf));
            worldQueries[i]=worldQ;
            pn[i].positionAndMass=f4(float(worldQ.x),float(worldQ.y),float(worldQ.z),1.0f);
            pn[i].velocityAndInverseMass=f4(0,0,0,1);
            pn[i].restAndFixed=f4(0,0,0,0);
            pp[i].position={float(origin.x),float(origin.y),float(origin.z),0};
            pp[i].orientation={float(orientation[0]),float(orientation[1]),
                               float(orientation[2]),float(orientation[3])};
            const auto body=std::find_if(closed.bodies.begin(),closed.bodies.end(),
                [&](const auto& range){return range.bodyIndex==q.body;});
            require(body!=closed.bodies.end(),"query body owns a complete BVH range");
            ranges[i].firstTriangle=body->firstNode;
            ranges[i].triangleCount=body->nodeCount;
            ranges[i].maximumDistanceMeters=0.05f;
            ranges[i].reserved0=body->closedOrientedCertified
                ? numi::matter::fem_body_contact_bvh::kRangeFlagStaticClosedOrientedBVH
                : numi::matter::fem_body_contact_bvh::kRangeFlagStaticCompleteBVH;
            const auto [r,f]=reference(qf,t);require(f==q.expected,"reference feature agrees");
            pr[i]=r;
            if(q.group>=0) {
                const auto g=unsigned(q.group);
                if(std::strncmp(q.label,"fd-",3)!=0)baseIndex[g]=std::uint32_t(i);
                else {
                    const unsigned ordinal=counts[g]++;
                    if(ordinal%2u==0u)plusIndex[g*3u+ordinal/2u]=std::uint32_t(i);
                    else minusIndex[g*3u+ordinal/2u]=std::uint32_t(i);
                    require(t.triangleStableId==pt[baseIndex[g]].triangleStableId,
                        "finite-difference query retains its selected triangle");
                }
            }
        }
        require(std::all_of(baseIndex.begin(),baseIndex.end(),[](auto v){return v!=UINT32_MAX;})&&
                std::all_of(counts.begin(),counts.end(),[](auto c){return c==6u;})&&
                std::all_of(plusIndex.begin(),plusIndex.end(),[](auto v){return v!=0u;})&&
                std::all_of(minusIndex.begin(),minusIndex.end(),[](auto v){return v!=0u;}),
                "central-difference query inventory complete");
        NMNumiHumanTendonFEMLoadDispatchGPU pseudoDispatch{};
        pseudoDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        pseudoDispatch.environmentCount=1u;
        pseudoDispatch.femNodeCount=static_cast<nm_u32>(n);
        pseudoDispatch.bodyPoseStride=static_cast<nm_u32>(n);
        pseudoDispatch.articulationFirstBody=0u;
        pseudoDispatch.femBodyContactSampleCount=static_cast<nm_u32>(n);
        pseudoDispatch.femBodyContactCandidateTriangleCount=
            static_cast<nm_u32>(closed.triangles.size());
        require(numi::matter::fem_body_contact_bvh::validCompleteStaticBVHSpans(
                ps,ranges,closed.triangles,closed.nodes),
                "strict closed source-bank ranges validate before GPU upload");
        auto pDispatch=dispatchBuffer(&pseudoDispatch,sizeof(pseudoDispatch));
        auto pSamples=dispatchBuffer(ps.data(),n*sizeof(ps[0]));
        auto pNodes=dispatchBuffer(pn.data(),n*sizeof(pn[0]));
        auto pPoses=dispatchBuffer(pp.data(),n*sizeof(pp[0]));
        auto pRanges=dispatchBuffer(ranges.data(),n*sizeof(ranges[0]));
        auto pTriangles=dispatchBuffer(closed.triangles.data(),
            closed.triangles.size()*sizeof(closed.triangles[0]));
        auto pBvhNodes=dispatchBuffer(closed.nodes.data(),
            closed.nodes.size()*sizeof(closed.nodes[0]));
        std::vector<nm_float4> pOut(n*8u);
        auto pOutput=dispatchBuffer(pOut.data(),pOut.size()*sizeof(pOut[0]));
        require(pDispatch&&pSamples&&pNodes&&pPoses&&pRanges&&pTriangles&&
                pBvhNodes&&pOutput,"strict closed-bank GPU buffers");
        auto pCmd=[queue commandBuffer];auto pEnc=[pCmd computeCommandEncoder];
        [pEnc setComputePipelineState:pseudoPipeline];
        [pEnc setBuffer:pDispatch offset:0 atIndex:0];
        [pEnc setBuffer:pSamples offset:0 atIndex:1];
        [pEnc setBuffer:pNodes offset:0 atIndex:2];
        [pEnc setBuffer:pPoses offset:0 atIndex:3];
        [pEnc setBuffer:pOutput offset:0 atIndex:4];
        [pEnc setBuffer:pRanges offset:0 atIndex:5];
        [pEnc setBuffer:pTriangles offset:0 atIndex:6];
        [pEnc setBuffer:pBvhNodes offset:0 atIndex:7];
        [pEnc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)];
        [pEnc endEncoding];[pCmd commit];[pCmd waitUntilCompleted];
        require(pCmd.status==MTLCommandBufferStatusCompleted,"pseudonormal production kernel completes");
        const auto* po=static_cast<const nm_float4*>(pOutput.contents);
        double maxCpuTangentError=0.0;
        std::size_t maxCpuTangentQuery=0;
        unsigned maxCpuTangentRow=0,maxCpuTangentCol=0;
        double maxCpuTangentGot=0.0,maxCpuTangentExpected=0.0;
        for(std::size_t i=0;i<n;++i) {
            const auto force=po[i*8u];
            require(std::isfinite(force.x)&&std::isfinite(force.y)&&
                std::isfinite(force.z)&&std::isfinite(force.w),"finite feature contact output");
            require(std::abs(double(force.x)-pr[i].force.x)<3.0e-3&&
                std::abs(double(force.y)-pr[i].force.y)<3.0e-3&&
                std::abs(double(force.z)-pr[i].force.z)<3.0e-3&&
                std::abs(double(force.w)-pr[i].closure)<2.0e-6,
                "GPU signed force matches independent CPU feature-pseudonormal oracle");
            const auto reaction=po[i*8u+1u];
            require(std::abs(reaction.x+force.x)<1.0e-5&&
                std::abs(reaction.y+force.y)<1.0e-5&&std::abs(reaction.z+force.z)<1.0e-5,
                "GPU body reaction balances the slave force");
            const auto point=po[i*8u+2u];
            if (queries[i].expectedAabbInactive) {
                require(force.w==0.0f&&
                        std::abs(double(point.x)-double(float(worldQueries[i].x)))<1.0e-7&&
                        std::abs(double(point.y)-double(float(worldQueries[i].y)))<1.0e-7&&
                        std::abs(double(point.z)-double(float(worldQueries[i].z)))<1.0e-7&&
                        std::isfinite(po[i*8u+1u].x)&&
                        std::abs(double(po[i*8u+1u].w)-0.25)<1.0e-7,
                        "certified exterior AABB fast path returns finite zero-force state");
            } else {
                require(std::abs(double(point.x)-pr[i].point.x)<2.0e-5&&
                    std::abs(double(point.y)-pr[i].point.y)<2.0e-5&&
                    std::abs(double(point.z)-pr[i].point.z)<2.0e-5,
                    "GPU closest point matches full-bank CPU query");
            }
            if (std::string_view(queries[i].label)=="concave-outside-inside-aabb")
                require(norm(sub(Vec3{point.x,point.y,point.z},worldQueries[i]))>0.01,
                    "inside-AABB concave exterior follows full closest-face query");
            for(unsigned col=0;col<3u;++col) {
                const auto got=po[i*8u+3u+col];
                const std::array<double,3> values{{got.x,got.y,got.z}};
                for(unsigned row=0;row<3u;++row) {
                    const double errorValue=std::abs(values[row]-pr[i].tangent[row][col]);
                    if(errorValue>maxCpuTangentError) {
                        maxCpuTangentError=errorValue;
                        maxCpuTangentQuery=i;maxCpuTangentRow=row;maxCpuTangentCol=col;
                        maxCpuTangentGot=values[row];
                        maxCpuTangentExpected=pr[i].tangent[row][col];
                    }
                }
            }
        }
        std::printf("feature_tangent_cpu_max_abs=%.9g label=%s query=%zu feature=%u row=%u col=%u got=%.9g expected=%.9g\n",
            maxCpuTangentError,queries[maxCpuTangentQuery].label,maxCpuTangentQuery,
            unsigned(queries[maxCpuTangentQuery].expected),maxCpuTangentRow,maxCpuTangentCol,
            maxCpuTangentGot,maxCpuTangentExpected);
        constexpr double fdTolerance=0.35;
        double maxFdError=0.0;
        unsigned maxFdGroup=0,maxFdAxis=0,maxFdRow=0;
        double maxFdValue=0.0,maxFdTangent=0.0;
        for(unsigned g=0;g<3u;++g)for(unsigned c=0;c<3u;++c) {
            const auto plus=plusIndex[g*3u+c],minus=minusIndex[g*3u+c];
            const auto fp=po[plus*8u],fm=po[minus*8u];
            const std::array<double,3> fd{{
                -(double(fp.x)-fm.x)/(2.0*fdStep),
                -(double(fp.y)-fm.y)/(2.0*fdStep),
                -(double(fp.z)-fm.z)/(2.0*fdStep)}};
            const auto k=po[baseIndex[g]*8u+3u+c];
            const std::array<double,3> tangent{{k.x,k.y,k.z}};
            for(unsigned row=0;row<3u;++row) {
                const double errorValue=std::abs(fd[row]-tangent[row]);
                if(errorValue>maxFdError) {
                    maxFdError=errorValue;maxFdGroup=g;maxFdAxis=c;maxFdRow=row;
                    maxFdValue=fd[row];maxFdTangent=tangent[row];
                }
            }
        }
        std::printf("feature_tangent_fd_max_abs=%.9g group=%u axis=%u row=%u fd=%.9g tangent=%.9g\n",
            maxFdError,maxFdGroup,maxFdAxis,maxFdRow,maxFdValue,maxFdTangent);
        require(maxCpuTangentError<2.0e-2,"GPU tangent matches CPU feature Hessian");
        require(maxFdError<fdTolerance,"face/reflex-edge tangent agrees with GPU force finite difference");
        std::printf("closed_pseudonormal_gpu=PASS convex_face_edge_vertex=3 concave_face_edge_vertex=3 active_fd=face+reflex_edge\n");
        std::printf("closed_aabb_gpu=PASS outside_face_edge_corner=3 rotated=1 boundary_ulp_full_query=1 band_boundary_full_query=1 concave_inside_aabb_full_query=1\n");

        std::printf("{\"status\":\"PASS\",\"device\":\"%s\","
                    "\"triangle_cases\":5,\"legacy_cases\":3,"
                    "\"outside_force\":[%.8g,%.8g,%.8g],"
                    "\"face_force\":[%.8g,%.8g,%.8g],"
                    "\"edge_force\":[%.8g,%.8g,%.8g],"
                    "\"vertex_force\":[%.8g,%.8g,%.8g]}\n",
                    device.name.UTF8String,out[0].x,out[0].y,out[0].z,
                    out[8u].x,out[8u].y,out[8u].z,
                    out[16u].x,out[16u].y,out[16u].z,
                    out[24u].x,out[24u].y,out[24u].z);
    }
    return 0;
}
