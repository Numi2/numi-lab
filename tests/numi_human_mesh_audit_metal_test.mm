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
    const unsigned count=16391,groups=64;
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
    id<MTLComputePipelineState> skinReference=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_audit_skin"] error:&error];
    id<MTLComputePipelineState> skinPartial=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_audit_skin_partials"] error:&error];
    id<MTLComputePipelineState> skinReduce=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"nm_human_resting_reduce_skin_audit"] error:&error];
    check(skinReference&&skinPartial&&skinReduce,"scalar/parallel skin audit kernels unavailable");
    auto runSkinAudit=[&](unsigned vertexCount,
            const std::vector<std::pair<unsigned,float>>& zOverrides,
            const std::vector<unsigned>& nanNormalIndices,bool allSkin,
            float expectedMinimum,unsigned expectedOwner,unsigned expectedBelow,unsigned expectedInvalid) {
        std::vector<MRHumanRestingVertexMap> skinMap(vertexCount);
        std::vector<MRVisualVertexGPUV2> skinVertices(vertexCount);
        for(unsigned i=0;i<vertexCount;++i) {
            skinMap[i].deformationKind=allSkin?3u:0u;
            skinVertices[i].position={0.1f,0.2f,1.0f,1.0f};
            skinVertices[i].normalAndTangentSign={0.0f,0.0f,1.0f,1.0f};
        }
        for(const auto& [index,z]:zOverrides) {
            check(index<vertexCount,"skin audit fixture override index out of range");
            skinMap[index].deformationKind=3u;
            skinVertices[index].position.z=z;
        }
        const float nan=std::numeric_limits<float>::quiet_NaN();
        for(const unsigned index:nanNormalIndices) {
            check(index<vertexCount,"skin audit fixture normal index out of range");
            skinVertices[index].normalAndTangentSign.x=nan;
        }
        const unsigned groups=(vertexCount+255u)/256u;
        id<MTLBuffer> skinMapBuffer=[device newBufferWithBytes:skinMap.data()
            length:skinMap.size()*sizeof(skinMap.front()) options:MTLResourceStorageModeShared];
        id<MTLBuffer> skinVerticesBuffer=[device newBufferWithBytes:skinVertices.data()
            length:skinVertices.size()*sizeof(skinVertices.front()) options:MTLResourceStorageModeShared];
        id<MTLBuffer> skinPartialBuffer=[device newBufferWithLength:groups*sizeof(mr_uint4)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> skinResultBuffer=[device newBufferWithLength:2u*sizeof(mr_float4)
            options:MTLResourceStorageModeShared];
        check(skinMapBuffer&&skinVerticesBuffer&&skinPartialBuffer&&skinResultBuffer,
            "skin scalar/parallel Metal fixture allocation failed");
        std::memset(skinPartialBuffer.contents,0xa5,skinPartialBuffer.length);
        std::memset(skinResultBuffer.contents,0x5a,skinResultBuffer.length);
        const mr_uint4 referenceDimensions={vertexCount,0,0,0};
        const mr_uint4 partialDimensions={vertexCount,groups,0,0};
        const mr_uint4 reduceDimensions={groups,1u,0,0};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:skinReference];
        [encoder setBytes:&referenceDimensions length:sizeof(referenceDimensions) atIndex:0];
        [encoder setBuffer:skinMapBuffer offset:0 atIndex:1];
        [encoder setBuffer:skinVerticesBuffer offset:0 atIndex:2];
        [encoder setBuffer:skinResultBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:skinPartial];
        [encoder setBytes:&partialDimensions length:sizeof(partialDimensions) atIndex:0];
        [encoder setBuffer:skinMapBuffer offset:0 atIndex:1];
        [encoder setBuffer:skinVerticesBuffer offset:0 atIndex:2];
        [encoder setBuffer:skinPartialBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(groups*256u,1,1)
            threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder setComputePipelineState:skinReduce];
        [encoder setBytes:&reduceDimensions length:sizeof(reduceDimensions) atIndex:0];
        [encoder setBuffer:skinPartialBuffer offset:0 atIndex:1];
        [encoder setBuffer:skinResultBuffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        check(command.status==MTLCommandBufferStatusCompleted,"scalar/parallel skin audit GPU command failed");
        const auto* rows=static_cast<const mr_float4*>(skinResultBuffer.contents);
        const mr_float4 expected={expectedMinimum,float(expectedOwner),float(expectedBelow),float(expectedInvalid)};
        check(std::memcmp(&rows[0],&rows[1],sizeof(mr_float4))==0,
            "parallel skin audit differs bitwise from retained scalar kernel");
        check(std::memcmp(&rows[0],&expected,sizeof(expected))==0,
            "skin audit minimum bits, exact counts, or witness ID differs from fixture");
        check(std::memcmp(skinMapBuffer.contents,skinMap.data(),skinMapBuffer.length)==0&&
            std::memcmp(skinVerticesBuffer.contents,skinVertices.data(),skinVerticesBuffer.length)==0,
            "skin audit mutated submitted maps or vertices");
    };
    const float positiveInfiniteGap=std::numeric_limits<float>::infinity();
    const float negativeInfiniteGap=-std::numeric_limits<float>::infinity();
    runSkinAudit(257u,{{256u,-0.01f}},{255u},true,-0.01f,256u,1u,1u);
    runSkinAudit(513u,{{512u,-0.01f}},{511u},true,-0.01f,512u,1u,1u);
    runSkinAudit(513u,{{37u,0.0f},{176u,-0.0f}},{},true,-0.0f,176u,0u,0u);
    runSkinAudit(513u,{{256u,negativeInfiniteGap},{512u,negativeInfiniteGap}},{},false,
        negativeInfiniteGap,256u,2u,2u);
    runSkinAudit(257u,{{37u,positiveInfiniteGap},{176u,positiveInfiniteGap}},{},false,
        positiveInfiniteGap,MR_INVALID_INDEX,0u,2u);
    std::cout<<"scalar_parallel_skin_audit_metal_test=passed cases=5 sizes=257,513 ties=signed-zero,negative-infinity positive-infinity=unselected tail=256,512 nan-normal=255,511\n";
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
        if(!(r.x==zero&&r.y==nonfinite&&r.z==first&&r.w==triangles))
            std::cerr<<"mesh audit actual="<<r.x<<","<<r.y<<","<<r.z<<","<<r.w
                <<" expected="<<zero<<","<<nonfinite<<","<<first<<","<<triangles<<"\n";
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
    vertices[3*255].position={0,0,0,1};
    vertices[3*255+1].position={1,0,0,1};
    vertices[3*255+2].position={2,0,0,1}; // Distinct exactly representable collinear triangle.
    vertices[3*16386].position.x=std::numeric_limits<float>::quiet_NaN(); // Beyond one grid stride.
    run(count,2,1,5,MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA);
    vertices=clean;
    run(count,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    vertices[3*16386].position.x=std::numeric_limits<float>::quiet_NaN();
    run(count,0,1,16386,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA);
    vertices=clean;
    run(1,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    auto runTriangle=[&](const std::array<mr_float4,3>& points,unsigned expectedZero,unsigned expectedFirst) {
        vertices=clean;
        for(unsigned i=0;i<3;++i)vertices[i].position=points[i];
        run(1,expectedZero,0,expectedFirst,
            expectedZero?MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA:
                MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    };
    const std::array<mr_float4,3> fmaWitness{{
        {0.03748244047164917f,-0.39636877179145813f,0.09854736924171448f,1},
        {0.03771711140871048f,-0.3969446122646332f,0.09887465834617615f,1},
        {0.03795178234577179f,-0.3975204527378082f,0.09920194745063782f,1}}};
    runTriangle(fmaWitness,1,0);
    auto oneUlp=fmaWitness;
    oneUlp[2].x=std::nextafter(oneUlp[2].x,std::numeric_limits<float>::infinity());
    runTriangle(oneUlp,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    oneUlp=fmaWitness;
    oneUlp[2].y=std::nextafter(oneUlp[2].y,std::numeric_limits<float>::infinity());
    runTriangle(oneUlp,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    oneUlp=fmaWitness;
    oneUlp[2].z=std::nextafter(oneUlp[2].z,std::numeric_limits<float>::infinity());
    runTriangle(oneUlp,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    const std::array<mr_float4,3> powerTwoScaled{{
        {0.07496488094329834f,-0.7927375435829163f,0.19709473848342896f,1},
        {0.07543422281742096f,-0.7938892245292664f,0.1977493166923523f,1},
        {0.07590356469154358f,-0.7950409054756165f,0.19840389490127563f,1}}};
    runTriangle(powerTwoScaled,1,0);
    const std::array<mr_float4,3> translatedScaled{{
        {-0.23125877976417542f,0.051815614104270935f,-0.07572631537914276f,1},
        {-0.23114144802093506f,0.05152769386768341f,-0.07556267082691193f,1},
        {-0.2310241162776947f,0.051239773631095886f,-0.07539902627468109f,1}}};
    runTriangle(translatedScaled,1,0);
    const std::array<mr_float4,3> roundedTranslated{{
        {0.5749648809432983f,-1.7927374839782715f,0.44709473848342896f,1},
        {0.5754342079162598f,-1.7938892841339111f,0.4477493166923523f,1},
        {0.5759035348892212f,-1.7950408458709717f,0.44840389490127563f,1}}};
    runTriangle(roundedTranslated,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    const float tinyEdge=0x1p-80f;
    const std::array<mr_float4,3> underflowNonzero{{
        {0,0,0,1},{tinyEdge,0,0,1},{0,tinyEdge,0,1}}};
    runTriangle(underflowNonzero,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    const std::array<mr_float4,3> underflowCollinear{{
        {0,0,0,1},{tinyEdge,tinyEdge,tinyEdge,1},{2*tinyEdge,2*tinyEdge,2*tinyEdge,1}}};
    runTriangle(underflowCollinear,1,0);
    const float large=0x1p60f;
    const std::array<mr_float4,3> inexactOriginNonzero{{
        {large,0,0,1},{1,1,0,1},{-large,2,0,1}}};
    runTriangle(inexactOriginNonzero,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE);
    const std::array<mr_float4,3> inexactOriginCollinear{{
        {large,large,0,1},{1,1,0,1},{-large,-large,0,1}}};
    runTriangle(inexactOriginCollinear,1,0);
    std::cout<<"fma_resistant_mesh_area_audit_metal_test=passed cases=11 exact_collinear=5 one_ulp_controls=3 rounded_translation_control=1 underflow_cases=2 inexact_origin_controls=2\n";
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
    const auto originalExpected0=expected0;
    const std::array<mr_float4,3> fmaVolumeWitness{{
        {0.03748244047164917f,-0.39636877179145813f,0.09854736924171448f,1},
        {0.03771711140871048f,-0.3969446122646332f,0.09887465834617615f,1},
        {0.03795178234577179f,-0.3975204527378082f,0.09920194745063782f,1}}};
    for(unsigned i=0;i<3u;++i)volumeVertices[volumeIndices[i]].position=fmaVolumeWitness[i];
    expected0=oldScalarAndExact(0u);
    runVolumeAudit(2u,0u,0u,0u,MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,expected0.first);
    checkVolumeResult(0u,expected0);checkVolumeResult(1u,expected1);
    volumeVertices=originalVolumeVertices;expected0=originalExpected0;
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
        <<" cases=5 groups="<<volumeGroups.size()<<"\n";
    std::cout<<"whole_native_mesh_metal_test=passed triangles="<<count<<" cases=5\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} } }

