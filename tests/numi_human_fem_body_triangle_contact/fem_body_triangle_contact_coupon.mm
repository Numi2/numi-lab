#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "support/fem_body_triangle_contact_math.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
        require(argc==2,"expected isolated metallib path");
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
