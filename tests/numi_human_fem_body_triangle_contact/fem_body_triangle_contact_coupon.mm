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
        const auto ownerSample=samples[1];
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
        require(dispatchBufferObject && ownerSampleBuffer &&
                ownerForceBuffer && ownerWrenchBuffer && ownerAuditBuffer,
                "actual owner kernel buffers");
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
            }
            [ownerEncoder dispatchThreads:MTLSizeMake(threadCount,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [ownerEncoder endEncoding];
        };
        encodeOwner(@"nm_numi_human_assemble_fem_body_contact_loads",
                    count,ownerForceBuffer);
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
        require(std::abs(actualLoads[1].x-out[8u].x)<1.0e-5f &&
                std::abs(actualLoads[1].y-out[8u].y)<1.0e-5f &&
                std::abs(actualLoads[1].z-out[8u].z)<1.0e-5f,
                "actual owner FEM load matches triangle state coupon");
        require(std::abs(actualWrenches[1].force.x+out[8u].x)<1.0e-5f &&
                std::abs(actualWrenches[1].force.y+out[8u].y)<1.0e-5f &&
                std::abs(actualWrenches[1].force.z+out[8u].z)<1.0e-5f &&
                std::abs(actualWrenches[1].torque.x-out[8u+6u].x)<1.0e-5f &&
                std::abs(actualWrenches[1].torque.y-out[8u+6u].y)<1.0e-5f &&
                std::abs(actualWrenches[1].torque.z-out[8u+6u].z)<1.0e-5f,
                "actual owner body wrench balances force at closest point");
        require(std::abs(actualAudit[0].forceResidualAndL1.x)<1.0e-5f &&
                std::abs(actualAudit[0].forceResidualAndL1.y)<1.0e-5f &&
                std::abs(actualAudit[0].forceResidualAndL1.z)<1.0e-5f &&
                std::abs(actualAudit[0].normalForceAreaAndCounts.y-area)<1.0e-6f &&
                std::abs(actualAudit[0].normalForceAreaAndCounts.z-1.0f)<1.0e-6f,
                "actual owner accepted-audit path preserves slave area/resultants");

        std::printf("{\"status\":\"PASS\",\"device\":\"%s\","
                    "\"triangle_cases\":5,\"legacy_cases\":2,"
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
