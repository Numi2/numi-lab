#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "NumiHumanAcceptedFEMBoundaryRenderPass.hpp"

using namespace numi::human;

struct EncoderContext {
    id<MTLComputeCommandEncoder> encoder;
};
static void useHeap(void*, void*) {}
static void setPipeline(void* c, void* p) {
    [static_cast<EncoderContext*>(c)->encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)p];
}
static void setBuffer(void* c, void* b, std::size_t off, std::uint32_t i) {
    [static_cast<EncoderContext*>(c)->encoder setBuffer:(__bridge id<MTLBuffer>)b offset:off atIndex:i];
}
static void setBytes(void* c, const void* b, std::size_t n, std::uint32_t i) {
    [static_cast<EncoderContext*>(c)->encoder setBytes:b length:n atIndex:i];
}
static void dispatchThreads(void* c, std::size_t n, std::size_t width) {
    [static_cast<EncoderContext*>(c)->encoder dispatchThreads:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(width,1,1)];
}
static void dispatchThreadgroups(void* c, std::size_t n, std::size_t width) {
    [static_cast<EncoderContext*>(c)->encoder dispatchThreadgroups:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(width,1,1)];
}

static void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static bool near(float a, float b, float eps=2.0e-5f) {
    return std::isfinite(a) && std::fabs(a-b)<=eps;
}
static float norm3(const mr_float4& a) {
    return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}
static void set4(mr_float4& v, float x,float y,float z,float w) { v={x,y,z,w}; }

int main(){
 @autoreleasepool{
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"device");
  id<MTLCommandQueue> queue=[device newCommandQueue];
  std::vector<AcceptedFEMBoundaryRenderPass::Face> faces={{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
  std::vector<AcceptedFEMBoundaryRenderPass::Point> points={
   {0,0,0,1},{1,0,0,1},{0,1,0,1},{0,0,1,1},
   {0,0,2,1},{1,0,2,1},{0,1,2,1},{0,0,3,1}};
  std::vector<AcceptedFEMBoundaryRenderPass::Face> fatFaces={{0,2,4},{2,1,4},{1,0,4},{0,1,3},{0,3,2},{1,2,3}};
  points.push_back({1.f/3.f,1.f/3.f,2,1});
  std::string error;
  AcceptedFEMBoundaryRenderPass pass;
  require(pass.initialize(device,faces,4,fatFaces,5,2,9,&error),error.c_str());
  require(pass.updateAcceptedState(points,&error),error.c_str());
  std::array<MRVisualVertexGPUV2,14> original{};
  for(auto& v:original){
   set4(v.position,-10,-20,-30,1);set4(v.normalAndTangentSign,0,0,1,1);
   set4(v.tangent,1,0,0,0);set4(v.texcoord01,.1f,.2f,.3f,.4f);set4(v.color,.8f,.6f,.4f,1);
  }
  id<MTLBuffer> mesh=[device newBufferWithBytes:original.data() length:sizeof(original) options:MTLResourceStorageModeShared];
  auto run=[&]{
   id<MTLCommandBuffer> cb=[queue commandBuffer];
   id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
   EncoderContext context{enc};metalrobo::MetalHybridComputeEncoderCallbacks callbacks{};
   callbacks.context=&context;callbacks.useHeap=useHeap;callbacks.setPipeline=setPipeline;
   callbacks.setBuffer=setBuffer;callbacks.setBytes=setBytes;callbacks.dispatchThreads=dispatchThreads;
   callbacks.dispatchThreadgroups=dispatchThreadgroups;
   require(!pass.encode(callbacks,(__bridge void*)mesh,13),"undersized output accepted");
   require(pass.encode(callbacks,(__bridge void*)mesh,14),"valid output refused");
   [enc endEncoding];[cb commit];[cb waitUntilCompleted];
   require(cb.status==MTLCommandBufferStatusCompleted,"dispatch failed");
  };
  auto check=[&]{
   const auto* actual=static_cast<const MRVisualVertexGPUV2*>(mesh.contents);
   for(unsigned i=0;i<14;i++){
    if(i==0||i==1||i==6||i==7||i==8){require(std::memcmp(&actual[i],&original[i],sizeof(original[i]))==0,"outside output changed");continue;}
    unsigned row=i<6?i-2:i-9+4;
    require(near(actual[i].position.x,points[row][0],0)&&near(actual[i].position.y,points[row][1],0)&&near(actual[i].position.z,points[row][2],0),"accepted point changed");
    require(near(norm3(actual[i].normalAndTangentSign),1),"normal");
    require(near(norm3(actual[i].tangent),1),"tangent");
    auto n=actual[i].normalAndTangentSign,t=actual[i].tangent;
    require(std::abs(n.x*t.x+n.y*t.y+n.z*t.z)<5e-5,"basis orthogonality");
    require(std::memcmp(&actual[i].color,&original[i].color,sizeof(mr_float4))==0,"color");
    require(std::memcmp(&actual[i].texcoord01,&original[i].texcoord01,sizeof(mr_float4))==0,"UV");
   }
  };
  run();check();
  points[3][2]+=.25f;points[6][0]+=.1f;
  require(pass.updateAcceptedState(points,&error),error.c_str());run();check();
  auto invalid=points;invalid[2][1]=NAN;
  require(!pass.updateAcceptedState(invalid,&error),"nonfinite update accepted");
  require(!pass.updateAcceptedState(std::span<const AcceptedFEMBoundaryRenderPass::Point>(points.data(),7),&error),"wrong count accepted");
  run();check();
  AcceptedFEMBoundaryRenderPass bad;
  require(!bad.initialize(device,faces,4,2,4,&error),"overlapping output accepted");
  require(!bad.initialize(device,faces,4,fatFaces,5,2,5,&error),"independent overlap accepted");
  auto wrongFat=fatFaces;wrongFat[0][0]=5;
  require(!bad.initialize(device,faces,4,wrongFat,5,2,9,&error),"role-local invalid ID accepted");
  std::printf("PASS accepted FEM boundary render device=%s layers=2 independent_node_counts=4,5 independent_face_counts=4,6 nodes=9 exact_positions=1 finite_unit_basis=1 unchanged_uv_color=1 outside_ranges_unchanged=1 invalid_update_atomic=1\n",device.name.UTF8String);
 }
 return 0;
}
