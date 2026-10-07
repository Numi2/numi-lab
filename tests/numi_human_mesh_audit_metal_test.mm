#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "numi/matter/human_respiration.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

static void check(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
int main() { @autoreleasepool { try {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    check(device!=nil,"Metal device unavailable");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:
        [NSURL fileURLWithPath:@NUMI_HUMAN_RESPIRATION_METALLIB] error:&error];
    check(library!=nil,"compiled HumanRespiration library unavailable");
    id<MTLComputePipelineState> partial=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_audit_mesh_triangles"] error:&error];
    id<MTLComputePipelineState> reduce=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_reduce_mesh_audit"] error:&error];
    check(partial&&reduce,"compiled mesh audit kernels unavailable");
    constexpr unsigned groupSize=256,trianglesPerLane=8;
    constexpr unsigned trianglesPerGroup=groupSize*trianglesPerLane;
    const unsigned count=16391,groups=count/trianglesPerGroup+
        unsigned(count%trianglesPerGroup!=0u);
    std::vector<MRVisualVertexGPUV2> vertices(3*count);
    std::vector<unsigned> indices(3*count);
    std::iota(indices.begin(),indices.end(),0u);
    for(unsigned t=0;t<count;++t) {
        const float x=float(t)*.001f;
        vertices[3*t].position={x,0,0,1};
        vertices[3*t+1].position={x+.5f,0,0,1};
        vertices[3*t+2].position={x,.25f,0,1};
    }
    const auto clean=vertices;
    id<MTLBuffer> vb=[device newBufferWithBytes:vertices.data() length:vertices.size()*sizeof(vertices[0]) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib=[device newBufferWithBytes:indices.data() length:indices.size()*sizeof(indices[0]) options:MTLResourceStorageModeShared];
    id<MTLBuffer> parts=[device newBufferWithLength:groups*sizeof(mr_uint4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> result=[device newBufferWithLength:sizeof(mr_uint4)+sizeof(MRHumanRestingSurfaceFailureGPU) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    check(vb&&ib&&parts&&result&&queue,"Metal fixture allocation failed");
    auto run=[&](unsigned triangles,unsigned zero,unsigned nonfinite,unsigned first,unsigned kind) {
        std::memcpy(vb.contents,vertices.data(),vb.length);
        std::memset(parts.contents,0xab,parts.length);
        const mr_uint4 d={triangles,groups,0,0};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
        [e setComputePipelineState:partial];[e setBytes:&d length:sizeof(d) atIndex:0];
        [e setBuffer:ib offset:0 atIndex:1];[e setBuffer:vb offset:0 atIndex:2];
        [e setBuffer:parts offset:0 atIndex:3];
        [e dispatchThreads:MTLSizeMake(groups*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e setComputePipelineState:reduce];[e setBytes:&d length:sizeof(d) atIndex:0];
        [e setBuffer:parts offset:0 atIndex:1];[e setBuffer:ib offset:0 atIndex:2];
        [e setBuffer:vb offset:0 atIndex:3];[e setBuffer:result offset:0 atIndex:4];
        [e setBuffer:result offset:sizeof(mr_uint4) atIndex:5];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e endEncoding];[command commit];[command waitUntilCompleted];
        check(command.status==MTLCommandBufferStatusCompleted,"mesh audit GPU command failed");
        const auto r=*static_cast<const mr_uint4*>(result.contents);
        const auto& f=*reinterpret_cast<const MRHumanRestingSurfaceFailureGPU*>(
            static_cast<const unsigned char*>(result.contents)+sizeof(mr_uint4));
        check(r.x==zero&&r.y==nonfinite&&r.z==first&&r.w==triangles,"mesh audit counts/first triangle differ");
        if(first<triangles) {
            check(f.surfaceTriangleKind.y==first&&f.surfaceTriangleKind.z==kind,"first failure identity differs");
            check(f.vertexIndices.x==3*first&&f.vertexIndices.y==3*first+1&&f.vertexIndices.z==3*first+2,"witness vertex indices differ");
            for(unsigned k=0;k<3;++k)
                check(std::memcmp(&f.renderedPositions[k].x,&vertices[3*first+k].position.x,3*sizeof(float))==0,
                    "witness positions differ from submitted binary32 vertices");
        }else check(f.surfaceTriangleKind.x==MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE&&
            f.surfaceTriangleKind.z==MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,"clean audit retained a stale witness");
        check(std::memcmp(vb.contents,vertices.data(),vb.length)==0&&
            std::memcmp(ib.contents,indices.data(),ib.length)==0,"mesh audit mutated geometry");
    };
    vertices[16].position=vertices[15].position; // Exact coincident vertex.
    vertices[3*255+2].position={1.255f,0,0,1}; // Distinct collinear triangle.
    vertices[3*(count-1)].position.x=std::numeric_limits<float>::quiet_NaN(); // Last triangle in the partial tail.
    run(count,2,1,5,MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA);
    vertices=clean;
    run(count,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    vertices[3*(count-1)].position.x=std::numeric_limits<float>::quiet_NaN();
    run(count,0,1,count-1,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA);
    vertices=clean;
    run(1,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    run(0,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    id<MTLComputePipelineState> volumePartial=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_audit_volume_partials"] error:&error];
    id<MTLComputePipelineState> volumeReduce=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_reduce_volume_audits"] error:&error];
    check(volumePartial&&volumeReduce,"parallel functional volume audit kernels unavailable");
    std::vector<MRVisualVertexGPUV2> volumeVertices;
    std::vector<unsigned> volumeIndices;
    std::vector<MRHumanRestingSurfaceAuditGPU> volumeSurfaces;
    auto addTetraSurface=[&](unsigned repeats,float scale) {
        const unsigned first=unsigned(volumeIndices.size());
        const std::array<mr_float4,4> points{{
            {3.0f,2.0f,-1.0f,1.0f},{3.0f+scale,2.0f,-1.0f,1.0f},
            {3.0f,2.0f+scale,-1.0f,1.0f},{3.0f,2.0f,-1.0f+scale,1.0f}}};
        const std::array<std::array<unsigned,3>,4> faces{{
            {{0,2,1}},{{0,1,3}},{{0,3,2}},{{1,2,3}}}};
        for(unsigned repetition=0;repetition<repeats;++repetition)for(const auto& face:faces)
            for(const unsigned point:face) {
                MRVisualVertexGPUV2 vertex{};vertex.position=points[point];
                volumeIndices.push_back(unsigned(volumeVertices.size()));
                volumeVertices.push_back(vertex);
            }
        const unsigned row=unsigned(volumeSurfaces.size());
        volumeSurfaces.push_back({{first,unsigned(volumeIndices.size())-first,2u,row},{0,0,0,0}});
    };
    addTetraSurface(257u,.75f);
    addTetraSurface(19u,.375f);
    std::vector<MRHumanRestingVolumeAuditGroupGPU> volumeGroups;
    std::vector<MRHumanRestingVolumeAuditRangeGPU> volumeRanges;
    for(unsigned row=0;row<volumeSurfaces.size();++row) {
        const unsigned firstGroup=unsigned(volumeGroups.size());
        const unsigned triangles=volumeSurfaces[row].indicesAndOwner.y/3u;
        for(unsigned firstTriangle=0;firstTriangle<triangles;
            firstTriangle+=MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS) {
            const unsigned triangleCount=std::min<unsigned>(
                MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS,triangles-firstTriangle);
            volumeGroups.push_back({{row,firstTriangle,triangleCount,0u}});
        }
        volumeRanges.push_back({{firstGroup,unsigned(volumeGroups.size())-firstGroup,0u,0u}});
    }
    auto oldScalarAndExact=[&](unsigned row) {
        const auto& owner=volumeSurfaces[row].indicesAndOwner;
        const auto origin=volumeVertices[volumeIndices[owner.x]].position;
        float sum=0,compensation=0;
        double exact=0;
        for(unsigned j=owner.x;j<owner.x+owner.y;j+=3u) {
            const auto pa=volumeVertices[volumeIndices[j]].position;
            const auto pb=volumeVertices[volumeIndices[j+1u]].position;
            const auto pc=volumeVertices[volumeIndices[j+2u]].position;
            const float ax=pa.x-origin.x,ay=pa.y-origin.y,az=pa.z-origin.z;
            const float bx=pb.x-origin.x,by=pb.y-origin.y,bz=pb.z-origin.z;
            const float cx=pc.x-origin.x,cy=pc.y-origin.y,cz=pc.z-origin.z;
            const float term=(ax*(by*cz-bz*cy)+ay*(bz*cx-bx*cz)+az*(bx*cy-by*cx))/6.0f;
            const float y=term-compensation,next=sum+y;
            compensation=(next-sum)-y;sum=next;
            const double dax=double(pa.x)-origin.x,day=double(pa.y)-origin.y,daz=double(pa.z)-origin.z;
            const double dbx=double(pb.x)-origin.x,dby=double(pb.y)-origin.y,dbz=double(pb.z)-origin.z;
            const double dcx=double(pc.x)-origin.x,dcy=double(pc.y)-origin.y,dcz=double(pc.z)-origin.z;
            exact+=(dax*(dby*dcz-dbz*dcy)+day*(dbz*dcx-dbx*dcz)+daz*(dbx*dcy-dby*dcx))/6.0;
        }
        return std::pair<float,double>{sum,exact};
    };
    auto expected0=oldScalarAndExact(0u),expected1=oldScalarAndExact(1u);
    NMHumanRespirationState volumeState{};
    volumeState.chamberVolumes={expected0.first,expected1.first,0,0};
    MRHumanRestingAnatomyGPU volumeAnatomy{};
    MRHumanRestingCardiacWallGPU volumeWall{};
    MRHumanRestingCommonFieldGPU volumeCommon{};
    MRHumanRestingCommonCoordinatesGPU volumeCoordinates{};
    id<MTLBuffer> volumeVB=[device newBufferWithBytes:volumeVertices.data()
        length:volumeVertices.size()*sizeof(volumeVertices.front()) options:MTLResourceStorageModeShared];
    id<MTLBuffer> volumeIB=[device newBufferWithBytes:volumeIndices.data()
        length:volumeIndices.size()*sizeof(volumeIndices.front()) options:MTLResourceStorageModeShared];
    id<MTLBuffer> surfaceBuffer=[device newBufferWithBytes:volumeSurfaces.data()
        length:volumeSurfaces.size()*sizeof(volumeSurfaces.front()) options:MTLResourceStorageModeShared];
    id<MTLBuffer> groupBuffer=[device newBufferWithBytes:volumeGroups.data()
        length:volumeGroups.size()*sizeof(volumeGroups.front()) options:MTLResourceStorageModeShared];
    id<MTLBuffer> rangeBuffer=[device newBufferWithBytes:volumeRanges.data()
        length:volumeRanges.size()*sizeof(volumeRanges.front()) options:MTLResourceStorageModeShared];
    id<MTLBuffer> partialBuffer=[device newBufferWithLength:volumeGroups.size()*
        sizeof(MRHumanRestingVolumeAuditPartialGPU) options:MTLResourceStorageModeShared];
    id<MTLBuffer> stateBuffer=[device newBufferWithBytes:&volumeState length:sizeof(volumeState) options:MTLResourceStorageModeShared];
    id<MTLBuffer> anatomyBuffer=[device newBufferWithBytes:&volumeAnatomy length:sizeof(volumeAnatomy) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wallBuffer=[device newBufferWithBytes:&volumeWall length:sizeof(volumeWall) options:MTLResourceStorageModeShared];
    id<MTLBuffer> commonBuffer=[device newBufferWithBytes:&volumeCommon length:sizeof(volumeCommon) options:MTLResourceStorageModeShared];
    id<MTLBuffer> coordinatesBuffer=[device newBufferWithBytes:&volumeCoordinates length:sizeof(volumeCoordinates) options:MTLResourceStorageModeShared];
    const std::size_t failureOffset=(volumeSurfaces.size()+2u)*sizeof(mr_float4);
    id<MTLBuffer> volumeOutput=[device newBufferWithLength:failureOffset+
        volumeSurfaces.size()*sizeof(MRHumanRestingSurfaceFailureGPU) options:MTLResourceStorageModeShared];
    check(volumeVB&&volumeIB&&surfaceBuffer&&groupBuffer&&rangeBuffer&&partialBuffer&&stateBuffer&&
        anatomyBuffer&&wallBuffer&&commonBuffer&&coordinatesBuffer&&volumeOutput,
        "parallel functional volume audit fixture allocation failed");
    auto runVolumeAudit=[&](unsigned expectedStatus0,unsigned expectedStatus1,
                            unsigned expectedFailureSurface,unsigned expectedFailureTriangle,
                            unsigned expectedFailureKind,float chamber0Target) {
        std::memcpy(volumeVB.contents,volumeVertices.data(),volumeVB.length);
        volumeState.chamberVolumes.x=chamber0Target;volumeState.chamberVolumes.y=expected1.first;
        std::memcpy(stateBuffer.contents,&volumeState,sizeof(volumeState));
        const mr_uint4 dimensions={unsigned(volumeGroups.size()),unsigned(volumeSurfaces.size()),0,0};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:volumePartial];
        [encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:0];
        [encoder setBuffer:surfaceBuffer offset:0 atIndex:1];
        [encoder setBuffer:volumeIB offset:0 atIndex:2];[encoder setBuffer:volumeVB offset:0 atIndex:3];
        [encoder setBuffer:groupBuffer offset:0 atIndex:4];[encoder setBuffer:partialBuffer offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(volumeGroups.size()*
            MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS,1,1)
            threadsPerThreadgroup:MTLSizeMake(MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS,1,1)];
        [encoder setComputePipelineState:volumeReduce];
        [encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:0];
        [encoder setBuffer:surfaceBuffer offset:0 atIndex:1];
        [encoder setBuffer:volumeIB offset:0 atIndex:2];[encoder setBuffer:volumeVB offset:0 atIndex:3];
        [encoder setBuffer:rangeBuffer offset:0 atIndex:4];[encoder setBuffer:partialBuffer offset:0 atIndex:5];
        [encoder setBuffer:stateBuffer offset:0 atIndex:6];[encoder setBuffer:anatomyBuffer offset:0 atIndex:7];
        [encoder setBuffer:volumeOutput offset:0 atIndex:8];[encoder setBuffer:wallBuffer offset:0 atIndex:9];
        [encoder setBuffer:volumeOutput offset:failureOffset atIndex:10];
        [encoder setBuffer:commonBuffer offset:0 atIndex:11];[encoder setBuffer:coordinatesBuffer offset:0 atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(volumeSurfaces.size(),1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        check(command.status==MTLCommandBufferStatusCompleted,"parallel volume audit GPU command failed");
        const auto* rows=static_cast<const mr_float4*>(volumeOutput.contents);
        const auto* witnesses=reinterpret_cast<const MRHumanRestingSurfaceFailureGPU*>(
            static_cast<const unsigned char*>(volumeOutput.contents)+failureOffset);
        check(unsigned(rows[0].w)==expectedStatus0&&unsigned(rows[1].w)==expectedStatus1,
            "parallel volume audit status differs from exact geometry checks");
        if(expectedFailureSurface<volumeSurfaces.size()) {
            const auto& witness=witnesses[expectedFailureSurface];
            check(witness.surfaceTriangleKind.x==expectedFailureSurface&&
                witness.surfaceTriangleKind.y==expectedFailureTriangle&&
                witness.surfaceTriangleKind.z==expectedFailureKind,
                "parallel volume audit first-failure witness differs");
        }
    };
    runVolumeAudit(0u,0u,volumeSurfaces.size(),0u,0u,expected0.first);
    const auto checkVolumeResult=[&](unsigned row,const std::pair<float,double>& expected) {
        const auto* result=static_cast<const mr_float4*>(volumeOutput.contents);
        const double denom=std::max(1e-30,std::abs(expected.second));
        check(std::abs(double(result[row].x)-std::abs(double(expected.first)))/denom<2e-4,
            "parallel volume audit differs from legacy compensated FP32 result");
        check(std::abs(double(result[row].x)-std::abs(expected.second))/denom<2e-4,
            "parallel volume audit differs from offline FP64 triangle volume");
    };
    checkVolumeResult(0u,expected0);checkVolumeResult(1u,expected1);
    const auto originalVolumeVertices=volumeVertices;
    const unsigned zeroTriangleIndex=volumeIndices[1];
    const unsigned zeroMatchIndex=volumeIndices[2];
    volumeVertices[zeroTriangleIndex].position=volumeVertices[zeroMatchIndex].position;
    runVolumeAudit(2u,0u,0u,0u,MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,expected0.first);
    checkVolumeResult(0u,expected0);checkVolumeResult(1u,expected1);
    volumeVertices=originalVolumeVertices;
    const unsigned nonfiniteTriangle=263u;
    const unsigned nonfiniteVertex=volumeIndices[3u*nonfiniteTriangle];
    volumeVertices[nonfiniteVertex].position.x=std::numeric_limits<float>::quiet_NaN();
    runVolumeAudit(3u,0u,0u,nonfiniteTriangle,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA,expected0.first);
    volumeVertices=originalVolumeVertices;
    runVolumeAudit(1u,0u,volumeSurfaces.size(),0u,0u,0.0f);
    std::cout<<"parallel_volume_audit_metal_test=passed surfaces=2 triangles="<<volumeIndices.size()/3u
        <<" cases=4 groups="<<volumeGroups.size()<<"\n";
    std::cout<<"whole_native_mesh_metal_test=passed triangles="<<count<<" cases=5\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} } }

