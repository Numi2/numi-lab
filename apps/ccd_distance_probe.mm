#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Case { std::array<mr_float4,7> rows; };
struct Result { mr_float4 normalSeparation,pointA,pointB;mr_uint4 status; };
static_assert(sizeof(Case)==112 && sizeof(Result)==64);
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
template<class T>std::vector<T> read(const char* name){std::ifstream f(name,std::ios::binary|std::ios::ate);require(bool(f),"Missing input");auto n=f.tellg();require(n>0 && n%sizeof(T)==0,"Bad input dimensions");std::vector<T> v(std::size_t(n)/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),n);require(bool(f),"Truncated input");return v;}
int main(int argc,char** argv){@autoreleasepool{try{
    require(argc==4,"Usage: metalrobo_ccd_distance_probe METALLIB INPUT_CASES NEW_OUTPUT");
    require(!std::filesystem::exists(argv[3]),"Output must be new");const auto inputs=read<Case>(argv[2]);
    require(inputs.size()<=1000000,"Too many cases");const unsigned count=unsigned(inputs.size());
    std::vector<MRShapeGPU> shapes(2*count);std::vector<MRProjectedColliderGPU> current(2*count),future(2*count);
    for(unsigned i=0;i<count;++i){const auto& x=inputs[i].rows;
      for(unsigned role=0;role<2;++role){const bool sphere=role==1;const unsigned j=2*i+(x[2].w!=0?1-role:role);
        auto& s=shapes[j];s.shapeType=sphere?MR_SHAPE_SPHERE:MR_SHAPE_BOX;s.bodyIndex=j;
        s.dimensions=sphere?mr_float4{x[0].w,0,0,0}:mr_float4{x[0].x,x[0].y,x[0].z,0};
        const float radius=sphere?x[0].w:std::sqrt(x[0].x*x[0].x+x[0].y*x[0].y+x[0].z*x[0].z);
        s.contactRestAndBoundingRadius={0,0,radius,0};
        for(unsigned end=0;end<2;++end){auto& p=end?future[j]:current[j];const auto center=x[sphere?5+end:1+end];
          p.centerAndRadius={center.x,center.y,center.z,radius};p.rotation=sphere?mr_float4{0,0,0,1}:x[3+end];
          p.lowerAndHalfLength={center.x-radius,center.y-radius,center.z-radius,0};
          p.upperAndContactOffset={center.x+radius,center.y+radius,center.z+radius,0};}
      }
    }
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    require(library!=nil,"Cannot load probe metallib");id<MTLFunction> fn=[library newFunctionWithName:@"mr_ccd_distance_probe"];
    require(fn!=nil,"Missing probe entry");auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];require(pipeline!=nil,"Cannot build probe pipeline");
    auto buffer=[&](const void* p,std::size_t n){return [device newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    const MRGeometryHeaderGPU header{};const mr_float4 vertex{};const MRMeshBVHNodeGPU node{};const MRMeshTriangleGPU triangle{};
    std::array<id<MTLBuffer>,9> b{buffer(inputs.data(),inputs.size()*sizeof(Case)),buffer(shapes.data(),shapes.size()*sizeof(MRShapeGPU)),
      buffer(current.data(),current.size()*sizeof(MRProjectedColliderGPU)),buffer(future.data(),future.size()*sizeof(MRProjectedColliderGPU)),
      buffer(&header,sizeof(header)),buffer(&vertex,sizeof(vertex)),buffer(&node,sizeof(node)),buffer(&triangle,sizeof(triangle)),
      [device newBufferWithLength:count*sizeof(Result) options:MTLResourceStorageModeShared]};
    for(auto p:b)require(p!=nil,"Probe allocation failed");auto queue=[device newCommandQueue];auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];for(unsigned i=0;i<b.size();++i)[encoder setBuffer:b[i] offset:0 atIndex:i];
    [encoder setBytes:&count length:sizeof(count) atIndex:9];
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<unsigned>(64,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"Probe GPU command failed");
    std::ofstream f(argv[3],std::ios::binary);f.write(static_cast<const char*>(b[8].contents),count*sizeof(Result));require(bool(f),"Cannot write probe results");
    std::cout<<"{\"cases\":"<<count<<",\"GPU_seconds\":"<<(command.GPUEndTime-command.GPUStartTime)<<",\"device\":\""<<device.name.UTF8String<<"\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
