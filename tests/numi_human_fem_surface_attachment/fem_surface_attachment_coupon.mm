#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/numi_human_stand_gpu.h"
#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <string>

namespace {
void require(bool value,const char* message){
    if(!value){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
nm_float4 f4(float x,float y,float z,float w){return {x,y,z,w};}
mr_float4 mr4(float x,float y,float z,float w){return {x,y,z,w};}
NMNumiHumanFEMSurfaceAttachmentGPU surfaceSample(){
    NMNumiHumanFEMSurfaceAttachmentGPU s{};
    s.node[0]=0;s.node[1]=1;s.node[2]=2;
    s.bodyIndex=1;s.stableID=73;
    s.flags=NM_NUMI_HUMAN_FEM_SURFACE_ATTACHMENT_ACTIVE;
    s.barycentric=f4(.2f,.3f,.5f,0);
    s.bodyLocalPointAndArea=f4(.3f,.5f,0,.02f);
    s.stiffnessAndStrainPerPressure=f4(100000.0f,1.0e-5f,0,0);
    return s;
}
NMNumiHumanFEMBodyContactSampleGPU legacySample(){
    NMNumiHumanFEMBodyContactSampleGPU s{};
    s.slaveNode=0;s.bodyIndex=1;
    s.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
    s.bodyLocalPointAndArea=f4(0,0,0,.01f);
    s.bodyLocalNormalAndReferenceSeparation=f4(0,0,1,0);
    s.stiffnessAndNormalStrainPerPressure=f4(30000.0f,1.0e-6f,0,0);
    return s;
}
}

int main(int argc,const char* argv[]){
 @autoreleasepool {
    require(argc==2,"expected isolated metallib path");
    NSError* error=nil;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil,"Metal device");
    id<MTLLibrary> library=[device newLibraryWithURL:
        [NSURL fileURLWithPath:@(argv[1])] error:&error];
    if(library==nil){
        std::fprintf(stderr,"load metallib: %s\n",
            error.localizedDescription.UTF8String);return 1;
    }
    id<MTLCommandQueue> queue=[device newCommandQueue];
    require(queue!=nil,"Metal queue");
    auto pipeline=[&](const char* name){
        NSString* qualified=[NSString stringWithFormat:@"numi_matter_metal::%s",name];
        id<MTLFunction> f=[library newFunctionWithName:qualified];
        require(f!=nil,name);
        NSError* e=nil;
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&e];
        if(p==nil){std::fprintf(stderr,"%s: %s\n",name,e.localizedDescription.UTF8String);std::exit(1);}
        return p;
    };
    auto buffer=[&](const void* data,NSUInteger length){
        require(length>0,"nonzero Metal buffer length");
        return [device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
    };

    // The state helper must reject malformed barycentrics, a degenerate
    // triangle, an out-of-range body, and a non-unit pose before force use.
    NMNumiHumanTendonFEMLoadDispatchGPU stateDispatch{};
    stateDispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
    stateDispatch.environmentCount=1;stateDispatch.femNodeCount=15;
    stateDispatch.bodyPoseStride=3;stateDispatch.articulationFirstBody=0;
    std::array<NMNumiHumanFEMSurfaceAttachmentGPU,5> stateSamples{};
    std::array<NMFEMNodeStateGPU,15> stateNodes{};
    std::array<MRArticulatedBodyPoseGPU,3> statePoses{};
    const std::array<std::array<nm_float4,3>,5> tri{{
        {{{0,0,0,1},{1,0,0,1},{0,1,0,1}}},
        {{{0,0,0,1},{1,0,0,1},{2,0,0,1}}},
        {{{0,0,0,1},{1,0,0,1},{0,1,0,1}}},
        {{{0,0,0,1},{1,0,0,1},{0,1,0,1}}},
        {{{0,0,0,1},{1,0,0,1},{0,1,0,1}}},
    }};
    for(std::size_t g=0;g<5;++g){
        auto s=surfaceSample();
        s.node[0]=static_cast<nm_u32>(3*g);
        s.node[1]=static_cast<nm_u32>(3*g+1);
        s.node[2]=static_cast<nm_u32>(3*g+2);
        s.stableID=static_cast<nm_u32>(100+g);
        if(g==2)s.barycentric=f4(.2f,.3f,.4f,0);
        if(g==3)s.bodyIndex=3;
        if(g==4)s.bodyIndex=2;
        stateSamples[g]=s;
        for(std::size_t j=0;j<3;++j)
            stateNodes[3*g+j].positionAndMass=tri[g][j];
    }
    for(auto& p:statePoses){p.position=mr4(0,0,.1f,0);p.orientation=mr4(0,0,0,1);}
    statePoses[2].orientation=mr4(0,0,0,2);
    std::array<nm_float4,5> stateOutput{};
    auto stateDispatchBuffer=buffer(&stateDispatch,sizeof(stateDispatch));
    auto stateSampleBuffer=buffer(stateSamples.data(),sizeof(stateSamples));
    auto stateNodeBuffer=buffer(stateNodes.data(),sizeof(stateNodes));
    auto statePoseBuffer=buffer(statePoses.data(),sizeof(statePoses));
    auto stateOutputBuffer=[device newBufferWithLength:sizeof(stateOutput)
        options:MTLResourceStorageModeShared];
    id<MTLFunction> stateFn=[library newFunctionWithName:
        @"nm_numi_human_surface_attachment_state_coupon"];
    require(stateFn!=nil,"state validation coupon kernel");
    id<MTLComputePipelineState> statePSO=[device newComputePipelineStateWithFunction:stateFn error:&error];
    require(statePSO!=nil,"state validation coupon PSO");
    id<MTLCommandBuffer> stateCommand=[queue commandBuffer];
    id<MTLComputeCommandEncoder> stateEncoder=[stateCommand computeCommandEncoder];
    [stateEncoder setComputePipelineState:statePSO];
    [stateEncoder setBuffer:stateDispatchBuffer offset:0 atIndex:0];
    [stateEncoder setBuffer:stateSampleBuffer offset:0 atIndex:1];
    [stateEncoder setBuffer:stateNodeBuffer offset:0 atIndex:2];
    [stateEncoder setBuffer:statePoseBuffer offset:0 atIndex:3];
    [stateEncoder setBuffer:stateOutputBuffer offset:0 atIndex:4];
    [stateEncoder dispatchThreads:MTLSizeMake(5,1,1)
        threadsPerThreadgroup:MTLSizeMake(5,1,1)];
    [stateEncoder endEncoding];[stateCommand commit];[stateCommand waitUntilCompleted];
    require(stateCommand.status==MTLCommandBufferStatusCompleted,"state helper GPU run");
    const auto* stateOut=static_cast<const nm_float4*>(stateOutputBuffer.contents);
    require(stateOut[0].x==1.0f && std::abs(stateOut[0].y-10.0f)<1e-4f,
        "state helper valid sample energy");
    for(std::size_t i=1;i<5;++i)
        require(stateOut[i].x==0.0f && std::isnan(stateOut[i].y),
            "state helper rejects malformed barycentric, degenerate, bad body, and bad pose");

    auto runOwner=[&](bool useLegacy,bool useSurface){
        require(useLegacy||useSurface,"empty owner case not needed");
        NMNumiHumanTendonFEMLoadDispatchGPU d{};
        d.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        d.environmentCount=1;d.femNodeCount=3;d.endpointCount=1;d.transferStride=1;
        d.stepIndex=0;d.dofCount=1;d.bodyPoseStride=2;d.articulationFirstBody=0;
        d.pointJacobianStride=24;d.bodyJacobianPointOffset=0;
        d.generalizedForceStride=1;d.muscleCount=1;
        d.femBodyContactSampleCount=useLegacy?1u:0u;
        d.femSurfaceAttachmentSampleCount=useSurface?1u:0u;
        d.reserved0=1u; // source-q count used by the owner projection/commit kernels.
        const auto oldSample=useLegacy?legacySample():NMNumiHumanFEMBodyContactSampleGPU{};
        const auto newSample=useSurface?surfaceSample():NMNumiHumanFEMSurfaceAttachmentGPU{};
        d.femBodyContactCandidateTriangleCount=0u;
        const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU emptyContactRange{};
        const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU emptyContactTriangle{};
        std::array<NMFEMNodeStateGPU,3> nodes{};
        nodes[0].positionAndMass=f4(0,0,0,1);
        nodes[1].positionAndMass=f4(1,0,0,1);
        nodes[2].positionAndMass=f4(0,1,0,1);
        std::array<MRArticulatedBodyPoseGPU,2> poses{};
        poses[0].position=mr4(0,0,0,0);poses[0].orientation=mr4(0,0,0,1);
        poses[1].position=mr4(0,0,.1f,0);poses[1].orientation=mr4(0,0,0,1);
        // Body 1 has unit z-translation Jacobian and zero angular Jacobian.
        std::array<float,24> jac{};
        jac[14]=1.0f;jac[17]=1.0f;jac[20]=1.0f;jac[23]=1.0f;
        std::array<nm_float4,3> forces{};
        std::array<NMNumiHumanBodyWrenchGPU,2> wrenches{};
        wrenches[0].force=f4(9,8,7,0);wrenches[1].force=f4(9,8,7,0);
        wrenches[0].torque=f4(6,5,4,0);wrenches[1].torque=f4(6,5,4,0);
        NMNumiHumanArticularContactAuditGPU audit{};
        std::array<NMNumiHumanArticularContactAuditGPU,
            NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS> history{};
        MRNumiHumanStandStatusGPU status{};
        status.code=MR_NUMI_HUMAN_STAND_SUCCESS;
        status.environment=0;status.completedSteps=1;
        float sourceQ=.25f,pendingLoad=0,pendingQ=0,acceptedLoad=0,acceptedQ=0;
        auto db=buffer(&d,sizeof(d));
        auto oldb=buffer(&oldSample,sizeof(oldSample));
        auto newb=buffer(&newSample,sizeof(newSample));
        auto nb=buffer(nodes.data(),sizeof(nodes));
        auto pb=buffer(poses.data(),sizeof(poses));
        auto fb=buffer(forces.data(),sizeof(forces));
        auto wb=buffer(wrenches.data(),sizeof(wrenches));
        auto ab=buffer(&audit,sizeof(audit));
        auto hb=buffer(history.data(),sizeof(history));
        auto jb=buffer(jac.data(),sizeof(jac));
        auto qb=buffer(&sourceQ,sizeof(sourceQ));
        auto plb=buffer(&pendingLoad,sizeof(pendingLoad));
        auto pqb=buffer(&pendingQ,sizeof(pendingQ));
        auto alb=buffer(&acceptedLoad,sizeof(acceptedLoad));
        auto aqb=buffer(&acceptedQ,sizeof(acceptedQ));
        auto sb=buffer(&status,sizeof(status));
        auto emptyRangeBuffer=buffer(&emptyContactRange,sizeof(emptyContactRange));
        auto emptyTriangleBuffer=buffer(&emptyContactTriangle,sizeof(emptyContactTriangle));
        require(db&&oldb&&newb&&nb&&pb&&fb&&wb&&ab&&hb&&jb&&qb&&plb&&pqb&&alb&&aqb&&sb&&
                emptyRangeBuffer&&emptyTriangleBuffer,
            "owner test buffers");
        id<MTLCommandBuffer> cmd=[queue commandBuffer];
        auto encode=[&](const char* name,NSUInteger n,
            const std::function<void(id<MTLComputeCommandEncoder>)>& bind){
            auto p=pipeline(name);
            auto e=[cmd computeCommandEncoder];
            require(e!=nil,"owner encoder");
            [e setComputePipelineState:p];
            [e setBuffer:db offset:0 atIndex:0];
            bind(e);
            [e dispatchThreads:MTLSizeMake(n,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [e endEncoding];
        };
        if(useLegacy) encode("nm_numi_human_assemble_fem_body_contact_loads",3,
            [&](id<MTLComputeCommandEncoder> e){
                [e setBuffer:oldb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];
                [e setBuffer:pb offset:0 atIndex:3];[e setBuffer:fb offset:0 atIndex:4];
                [e setBuffer:emptyRangeBuffer offset:0 atIndex:5];
                [e setBuffer:emptyTriangleBuffer offset:0 atIndex:6];
            });
        // This production kernel resets every body wrench even when the old
        // span is empty; the surface-only case proves the empty-span path.
        encode("nm_numi_human_assemble_fem_body_contact_wrenches",2,
            [&](id<MTLComputeCommandEncoder> e){
                [e setBuffer:oldb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];
                [e setBuffer:pb offset:0 atIndex:3];[e setBuffer:wb offset:0 atIndex:4];
                [e setBuffer:emptyRangeBuffer offset:0 atIndex:5];
                [e setBuffer:emptyTriangleBuffer offset:0 atIndex:6];
            });
        if(useSurface){
            encode("nm_numi_human_assemble_fem_surface_attachment_loads",3,
                [&](id<MTLComputeCommandEncoder> e){
                    [e setBuffer:newb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];
                    [e setBuffer:pb offset:0 atIndex:3];[e setBuffer:fb offset:0 atIndex:4];
                });
            encode("nm_numi_human_add_fem_surface_attachment_wrenches",2,
                [&](id<MTLComputeCommandEncoder> e){
                    [e setBuffer:newb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];
                    [e setBuffer:pb offset:0 atIndex:3];[e setBuffer:wb offset:0 atIndex:4];
                });
        }
        encode("nm_numi_human_audit_fem_body_contact",1,
            [&](id<MTLComputeCommandEncoder> e){
                [e setBuffer:oldb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];
                [e setBuffer:pb offset:0 atIndex:3];[e setBuffer:ab offset:0 atIndex:4];
                [e setBuffer:newb offset:0 atIndex:5];
                [e setBuffer:emptyRangeBuffer offset:0 atIndex:6];
                [e setBuffer:emptyTriangleBuffer offset:0 atIndex:7];
            });
        encode("nm_numi_human_prepare_fem_body_generalized_load",1,
            [&](id<MTLComputeCommandEncoder> e){
                [e setBuffer:wb offset:0 atIndex:1];[e setBuffer:pb offset:0 atIndex:2];
                [e setBuffer:jb offset:0 atIndex:3];[e setBuffer:qb offset:0 atIndex:4];
                [e setBuffer:plb offset:0 atIndex:5];[e setBuffer:pqb offset:0 atIndex:6];
            });
        encode("nm_numi_human_commit_fem_body_generalized_load",1,
            [&](id<MTLComputeCommandEncoder> e){
                [e setBuffer:sb offset:0 atIndex:1];[e setBuffer:plb offset:0 atIndex:2];
                [e setBuffer:pqb offset:0 atIndex:3];[e setBuffer:alb offset:0 atIndex:4];
                [e setBuffer:aqb offset:0 atIndex:5];
            });
        encode("nm_numi_human_commit_articular_contact_audit",1,
            [&](id<MTLComputeCommandEncoder> e){
                [e setBuffer:sb offset:0 atIndex:1];[e setBuffer:ab offset:0 atIndex:2];
                [e setBuffer:hb offset:0 atIndex:3];
            });
        [cmd commit];[cmd waitUntilCompleted];
        require(cmd.status==MTLCommandBufferStatusCompleted,"actual owner command buffer");
        const auto* gotF=static_cast<const nm_float4*>(fb.contents);
        const auto* gotW=static_cast<const NMNumiHumanBodyWrenchGPU*>(wb.contents);
        const auto* gotA=static_cast<const NMNumiHumanArticularContactAuditGPU*>(ab.contents);
        const auto* gotH=static_cast<const NMNumiHumanArticularContactAuditGPU*>(hb.contents);
        const auto* gotL=static_cast<const float*>(alb.contents);
        const float total=(useSurface?200.0f:0.0f)+(useLegacy?30.0f:0.0f);
        require(std::abs(gotF[0].z-(useSurface?40.0f:0.0f)-(useLegacy?30.0f:0.0f))<2e-4f,
            "owner force at first barycentric node");
        require(std::abs(gotF[1].z-(useSurface?60.0f:0.0f))<2e-4f &&
                std::abs(gotF[2].z-(useSurface?100.0f:0.0f))<2e-4f,
            "owner B-transpose force scatter");
        require(std::abs(gotW[0].force.x)<1e-5f && std::abs(gotW[0].force.y)<1e-5f &&
                std::abs(gotW[0].force.z)<1e-5f &&
                std::abs(gotW[1].force.z+total)<2e-4f,
            "body wrench reset and surface/legacy force sum");
        require(std::abs(gotW[1].torque.x+(useSurface?100.0f:0.0f))<2e-4f &&
                std::abs(gotW[1].torque.y-(useSurface?60.0f:0.0f))<2e-4f,
            "balanced body moment at the material point");
        const float expectedEnergy=(useSurface?10.0f:0.0f)+(useLegacy?1.5f:0.0f);
        const float expectedArea=(useSurface?.02f:0.0f)+(useLegacy?.01f:0.0f);
        require(std::abs(gotA->energyStrainClosureAndAccepted.x-expectedEnergy)<2e-4f &&
                std::abs(gotA->normalForceAreaAndCounts.y-expectedArea)<1e-6f &&
                std::abs(gotA->normalForceAreaAndCounts.w-float(useLegacy+useSurface))<1e-6f &&
                gotH[0].energyStrainClosureAndAccepted.w==1.0f,
            "owner energy, counts, and accepted audit commit");
        require(std::abs(gotA->forceResidualAndL1.x)<2e-4f &&
                std::abs(gotA->forceResidualAndL1.y)<2e-4f &&
                std::abs(gotA->forceResidualAndL1.z)<2e-4f &&
                std::abs(gotA->momentResidualAndMaximumPressure.x)<2e-4f &&
                std::abs(gotA->momentResidualAndMaximumPressure.y)<2e-4f &&
                std::abs(gotA->momentResidualAndMaximumPressure.z)<2e-4f,
            "owner audit force and world-origin moment balance");
        require(std::abs(gotL[0]+total)<3e-4f,
            "surface-only/mixed/legacy body-wrench generalized projection");
        return std::array<double,3>{double(total),double(expectedEnergy),
            double(gotL[0])};
    };
    const auto surfaceOnly=runOwner(false,true);
    const auto mixed=runOwner(true,true);
    const auto legacyOnly=runOwner(true,false);
    require(surfaceOnly[0]==200.0 && surfaceOnly[1]==10.0 &&
            mixed[0]==230.0 && mixed[1]==11.5 &&
            legacyOnly[0]==30.0 && legacyOnly[1]==1.5,
        "surface-only, mixed, and legacy-only frozen law values");
    std::printf("{\"status\":\"PASS\",\"device\":\"%s\","
        "\"surface_only_force_N\":%.6g,\"surface_only_energy_J\":%.6g,"
        "\"mixed_force_N\":%.6g,\"mixed_energy_J\":%.6g,"
        "\"legacy_only_force_N\":%.6g,\"legacy_only_energy_J\":%.6g,"
        "\"generalized_projection\":[%.6g,%.6g,%.6g],"
        "\"state_rejections\":4}\n",device.name.UTF8String,
        surfaceOnly[0],surfaceOnly[1],mixed[0],mixed[1],legacyOnly[0],legacyOnly[1],
        surfaceOnly[2],mixed[2],legacyOnly[2]);
 }
 return 0;
}
