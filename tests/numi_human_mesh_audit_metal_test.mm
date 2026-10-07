#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
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
    vertices[3*16386].position.x=std::numeric_limits<float>::quiet_NaN(); // Beyond one grid stride.
    run(count,2,1,5,MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA);
    vertices=clean;
    run(count,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    vertices[3*16386].position.x=std::numeric_limits<float>::quiet_NaN();
    run(count,0,1,16386,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA);
    vertices=clean;
    run(1,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    run(0,0,0,MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE,0);
    std::cout<<"whole_native_mesh_metal_test=passed triangles="<<count<<" cases=5\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} } }

