#pragma once
// Presentation only. Copy the two exact accepted FEM boundaries at the existing
// synchronized presentation cadence; no force, integration or alternate state.
#import <Metal/Metal.h>
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/visual_platform_types.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>
namespace numi::human {
class AcceptedFEMBoundaryRenderPass {
public:
 using Point=std::array<float,4>;
 using Face=std::array<std::uint32_t,3>;
 bool initialize(id<MTLDevice> device,std::span<const Face> faces,std::uint32_t n,
                 std::uint32_t firstInner,std::uint32_t firstFat,std::string* error) {
  reset();
  auto fail=[&](const char* m){if(error)*error=m;return false;};
  if(!device||n<4||n>16384||faces.empty()||faces.size()>32768)
   return fail("accepted boundary render topology is empty or outside reduced bounds");
  if(firstInner>UINT32_MAX-n||firstFat>UINT32_MAX-n||
     (firstInner<firstFat+n&&firstFat<firstInner+n))
   return fail("accepted boundary render output ranges overflow or overlap");
  std::vector<std::vector<std::uint32_t>> incident(n);
  std::vector<std::array<std::uint32_t,4>> triangles;triangles.reserve(faces.size());
  for(std::uint32_t i=0;i<faces.size();i++){
   auto f=faces[i];
   if(f[0]>=n||f[1]>=n||f[2]>=n||f[0]==f[1]||f[1]==f[2]||f[2]==f[0])
    return fail("accepted boundary render face has invalid vertices");
   triangles.push_back({f[0],f[1],f[2],0});
   for(auto v:f)incident[v].push_back(i);
  }
  std::vector<std::array<std::uint32_t,2>> ranges(n);
  std::vector<std::uint32_t> refs;refs.reserve(3*faces.size());
  for(std::uint32_t i=0;i<n;i++){
   if(incident[i].empty())return fail("accepted boundary render has an isolated node");
   ranges[i]={static_cast<std::uint32_t>(refs.size()),static_cast<std::uint32_t>(incident[i].size())};
   refs.insert(refs.end(),incident[i].begin(),incident[i].end());
  }
  static_assert(sizeof(MRVisualVertexGPUV2)==80);
  static_assert(offsetof(MRVisualVertexGPUV2,position)==0);
  static_assert(offsetof(MRVisualVertexGPUV2,normalAndTangentSign)==16);
  static_assert(offsetof(MRVisualVertexGPUV2,tangent)==32);
  static_assert(offsetof(MRVisualVertexGPUV2,texcoord01)==48);
  static_assert(offsetof(MRVisualVertexGPUV2,color)==64);
  NSError* e=nil;
  MTLCompileOptions* opts=[MTLCompileOptions new];
  opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;
  id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:metalSource()] options:opts error:&e];
  if(!lib){if(error)*error=e?std::string(e.localizedDescription.UTF8String):"boundary Metal library unavailable";return false;}
  pipeline_=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_accepted_fem_boundaries"] error:&e];
  if(!pipeline_){if(error)*error=e?std::string(e.localizedDescription.UTF8String):"boundary Metal pipeline unavailable";return false;}
  triangles_=[device newBufferWithBytes:triangles.data() length:triangles.size()*sizeof(triangles[0]) options:MTLResourceStorageModeShared];
  ranges_=[device newBufferWithBytes:ranges.data() length:ranges.size()*sizeof(ranges[0]) options:MTLResourceStorageModeShared];
  refs_=[device newBufferWithBytes:refs.data() length:refs.size()*sizeof(refs[0]) options:MTLResourceStorageModeShared];
  positions_=[device newBufferWithLength:2*n*sizeof(Point) options:MTLResourceStorageModeShared];
  if(!triangles_||!ranges_||!refs_||!positions_){reset();return fail("boundary Metal buffer allocation failed");}
  n_=n;firstInner_=firstInner;firstFat_=firstFat;
  if(error)error->clear();
  return true;
 }
 // Shared buffers are reused only between synchronous native presentations.
 bool updateAcceptedState(std::span<const Point> positions,std::string* error){
  if(!positions_||positions.size()!=2ull*n_){if(error)*error="accepted boundary point count mismatch";return false;}
  for(const auto& p:positions)for(unsigned c=0;c<3;c++)if(!std::isfinite(p[c])){
   if(error)*error="accepted boundary point is nonfinite";return false;
  }
  auto* bytes=positions_.contents;
  if(!bytes){if(error)*error="accepted boundary buffer is not accessible";return false;}
  std::memcpy(bytes,positions.data(),positions.size_bytes());ready_=true;
  if(error)error->clear();return true;
 }
 template<class Encoder>
 bool encode(const Encoder& e,void* vertices,std::uint32_t meshVertexCount)const{
  if(!ready_||!pipeline_||!vertices||!e.valid()||firstInner_>meshVertexCount||
     n_>meshVertexCount-firstInner_||firstFat_>meshVertexCount||n_>meshVertexCount-firstFat_)return false;
  const std::uint32_t p[4]={n_,firstInner_,firstFat_,2*n_};
  e.setPipeline(e.context,(__bridge void*)pipeline_);
  e.setBytes(e.context,p,sizeof(p),0);
  e.setBuffer(e.context,(__bridge void*)positions_,0,1);
  e.setBuffer(e.context,(__bridge void*)triangles_,0,2);
  e.setBuffer(e.context,(__bridge void*)ranges_,0,3);
  e.setBuffer(e.context,(__bridge void*)refs_,0,4);
  e.setBuffer(e.context,vertices,0,5);
  e.dispatchThreads(e.context,2*n_,64);
  return true;
 }
 void reset(){triangles_=nil;ranges_=nil;refs_=nil;positions_=nil;pipeline_=nil;n_=firstInner_=firstFat_=0;ready_=false;}
 static const char* metalSource(){return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct MRVisualVertexGPUV2{
 float4 position;float4 normalAndTangentSign;float4 tangent;float4 texcoord01;float4 color;
};
kernel void nm_human_accepted_fem_boundaries(
 constant uint4& p [[buffer(0)]],device const float4* points [[buffer(1)]],
 device const uint4* triangles [[buffer(2)]],device const uint2* ranges [[buffer(3)]],
 device const uint* refs [[buffer(4)]],device MRVisualVertexGPUV2* outputs [[buffer(5)]],
 uint i [[thread_position_in_grid]]){
 if(i>=p.w)return;
 uint layer=i/p.x,v=i%p.x,base=layer*p.x;
 uint out=(layer==0?p.y:p.z)+v;
 float3 normal=float3(0.0f);
 uint2 range=ranges[v];
 for(uint j=0;j<range.y;j++){
  uint3 f=triangles[refs[range.x+j]].xyz;
  normal+=cross(points[base+f.y].xyz-points[base+f.x].xyz,
                points[base+f.z].xyz-points[base+f.x].xyz);
 }
 outputs[out].position=float4(points[i].xyz,1.0f);
 float n2=dot(normal,normal);
 if(n2>1.0e-30f&&isfinite(n2)){
  normal*=rsqrt(n2);
  float3 axis=fabs(normal.x)<0.8f?float3(1,0,0):float3(0,1,0);
  float3 tangent=normalize(cross(normal,axis));
  outputs[out].normalAndTangentSign.xyz=normal;
  outputs[out].tangent.xyz=tangent;
 }
}
)MSL";}
private:
 __strong id<MTLBuffer> triangles_=nil,ranges_=nil,refs_=nil,positions_=nil;
 __strong id<MTLComputePipelineState> pipeline_=nil;
 std::uint32_t n_=0,firstInner_=0,firstFat_=0;
 bool ready_=false;
};
}
