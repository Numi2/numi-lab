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
        [rejectedStatusEncoder dispatchThreads:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [rejectedStatusEncoder endEncoding];
        [rejectedStatusCommand commit];
        [rejectedStatusCommand waitUntilCompleted];
        require(rejectedStatusCommand.status==MTLCommandBufferStatusCompleted &&
                static_cast<const MRMetalWorldStatusGPU*>(
                    rejectedStatusBuffer.contents)[0].code!=MR_STEP_SUCCESS,
                "provisional status rejects tangentially out-of-envelope nodes still near a prepared face");

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
