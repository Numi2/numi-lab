#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "metalrobo/visual_platform_types.h"
#include "metalrobo/numi_human_resting_rope_gpu.h"
#include "metalrobo/numi_human_resting_rope_visual_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct V3 { float x,y,z; };
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static std::string readFile(const char* path){std::ifstream f(path,std::ios::binary);require(f.good(),"assembled Metal source unavailable");return {std::istreambuf_iterator<char>(f),{}};}
static V3 add(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 sub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 mul(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static float len(V3 a){return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);}
static bool near3(V3 a,V3 b,float eps=2.0e-6f){return len(sub(a,b))<=eps;}
static V3 meanRing(const MRVisualVertexGPUV2* vertices,std::uint32_t first,std::uint32_t count){
 V3 r{};for(std::uint32_t i=0;i<count;++i){const auto p=vertices[first+i].position;r=add(r,{p.x,p.y,p.z});}return mul(r,1.0f/float(count));
}
static bool finite4(mr_float4 v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z)&&std::isfinite(v.w);}
static V3 rotate(mr_float4 q,V3 v){const V3 qv{q.x,q.y,q.z};return add(v,mul(cross(qv,add(cross(qv,v),mul(v,q.w))),2.0f));}
int main(int argc,char** argv){@autoreleasepool{try{
 require(argc==2,"usage: test_human_resting_rope_visual ASSEMBLED_METAL_SOURCE");
 const auto source=readFile(argv[1]);
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device unavailable");
 NSError* error=nil;MTLCompileOptions* options=[MTLCompileOptions new];
 options.mathMode=MTLMathModeSafe;options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
 options.languageVersion=MTLLanguageVersion4_0;
 id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
 if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 id<MTLFunction> function=[library newFunctionWithName:@"nm_human_resting_rope_visual"];
 require(function!=nil,"visual kernel missing from assembled source");
 id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
 if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);

 constexpr std::uint32_t bodyIndex=20,ropeRings=11,ropeSides=7,harnessMajor=8,harnessMinor=6;
 constexpr std::uint32_t ropeCount=ropeRings*ropeSides,harnessCount=harnessMajor*harnessMinor;
 constexpr std::uint32_t dynamicFirst=2,harnessFirst=dynamicFirst+ropeCount,totalVertices=harnessFirst+harnessCount+3;
 MRHumanRestingRopeGPU rope{};
 rope.indices={0,0,bodyIndex,18};rope.strides={1,0,1,0};rope.counts={1,3,0,1};
 rope.anchorAndLength={0.4f,1.0f,0.5f,3.2f};rope.bodyLocalSite={0.2f,0.1f,0.3f,0.0f};
 MRHumanRestingRopeVisualGPU visual{};
 visual.ranges={dynamicFirst,ropeRings,ropeSides,harnessFirst};
 visual.counts={ropeCount,harnessMajor,harnessMinor,harnessCount};
 visual.radiiAndSag={0.01f,0.05f,0.01f,0.8f};
 std::vector<MRBodyStateGPU> bodies(bodyIndex+1);
 for(auto& body:bodies){body.position={0,0,0,1};body.orientation={0,0,0,1};}
 const float s=std::sqrt(0.5f);bodies[bodyIndex].position={1.0f,2.0f,3.0f,1.0f};
 bodies[bodyIndex].orientation={0,0,s,s};
 MRVisualVertexGPUV2 sentinel{};sentinel.position={913.0f,-407.0f,211.0f,1.0f};
 sentinel.normalAndTangentSign={0,1,0,1};sentinel.tangent={1,0,0,0};
 sentinel.texcoord01={0.25f,0.75f,0.125f,0.875f};sentinel.color={0.2f,0.3f,0.4f,0.5f};
 std::vector<MRVisualVertexGPUV2> vertices(totalVertices,sentinel);
 auto makeBuffer=[&](const void* data,NSUInteger size){id<MTLBuffer> b=[device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];require(b!=nil,"Metal shared buffer allocation failed");return b;};
 id<MTLBuffer> ropeBuffer=makeBuffer(&rope,sizeof(rope));
 id<MTLBuffer> visualBuffer=makeBuffer(&visual,sizeof(visual));
 id<MTLBuffer> bodiesBuffer=makeBuffer(bodies.data(),bodies.size()*sizeof(MRBodyStateGPU));
 id<MTLBuffer> verticesBuffer=makeBuffer(vertices.data(),vertices.size()*sizeof(MRVisualVertexGPUV2));
 id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"Metal command queue unavailable");
 auto dispatch=[&]{
  std::memcpy(ropeBuffer.contents,&rope,sizeof(rope));std::memcpy(visualBuffer.contents,&visual,sizeof(visual));
  std::memcpy(bodiesBuffer.contents,bodies.data(),bodies.size()*sizeof(MRBodyStateGPU));
  id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  [enc setComputePipelineState:pipeline];[enc setBuffer:ropeBuffer offset:0 atIndex:0];
  [enc setBuffer:visualBuffer offset:0 atIndex:1];[enc setBuffer:bodiesBuffer offset:0 atIndex:2];
  [enc setBuffer:verticesBuffer offset:0 atIndex:3];
  [enc dispatchThreads:MTLSizeMake(ropeCount+harnessCount,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
  [enc endEncoding];[cb commit];[cb waitUntilCompleted];
  require(cb.status==MTLCommandBufferStatusCompleted,"visual GPU dispatch failed");
  std::memcpy(vertices.data(),verticesBuffer.contents,vertices.size()*sizeof(MRVisualVertexGPUV2));
 };
 auto checkDynamicFinite=[&]{
  for(std::uint32_t i=dynamicFirst;i<harnessFirst+harnessCount;++i){const auto& v=vertices[i];
   require(finite4(v.position)&&finite4(v.normalAndTangentSign)&&finite4(v.tangent),"nonfinite dynamic visual vertex");
   const V3 n{v.normalAndTangentSign.x,v.normalAndTangentSign.y,v.normalAndTangentSign.z};
   const V3 t{v.tangent.x,v.tangent.y,v.tangent.z};
   require(std::abs(len(n)-1.0f)<2.0e-4f&&std::abs(len(t)-1.0f)<2.0e-4f,"visual normal/tangent is not unit length");
  }
 };
 const V3 anchor{rope.anchorAndLength.x,rope.anchorAndLength.y,rope.anchorAndLength.z};
 const auto expectedEndpoint=[&]{const auto& b=bodies[bodyIndex];return add({b.position.x,b.position.y,b.position.z},rotate(b.orientation,{rope.bodyLocalSite.x,rope.bodyLocalSite.y,rope.bodyLocalSite.z}));};
 const auto ropeMidIndex=dynamicFirst+(ropeRings/2)*ropeSides;
 dispatch();checkDynamicFinite();
 auto* out=static_cast<MRVisualVertexGPUV2*>(verticesBuffer.contents);
 const V3 endpoint0=expectedEndpoint();
 require(near3(meanRing(out,dynamicFirst,ropeSides),anchor),"first rope ring detached from fixed anchor");
 require(near3(meanRing(out,dynamicFirst+(ropeRings-1)*ropeSides,ropeSides),endpoint0),"rope endpoint does not match COM plus rotated body-local site");
 require(near3(meanRing(out,harnessFirst,harnessCount),endpoint0,5.0e-6f),"moving harness ring detached from force endpoint");
 const V3 slackMid5=meanRing(out,ropeMidIndex,ropeSides);
 std::vector<MRVisualVertexGPUV2> first(vertices.begin()+dynamicFirst,vertices.begin()+harnessFirst+harnessCount);
 dispatch();require(std::memcmp(first.data(),static_cast<MRVisualVertexGPUV2*>(verticesBuffer.contents)+dynamicFirst,first.size()*sizeof(MRVisualVertexGPUV2))==0,"same accepted body pose does not replay byte-identically");
 rope.anchorAndLength.w=3.5f;dispatch();checkDynamicFinite();out=static_cast<MRVisualVertexGPUV2*>(verticesBuffer.contents);
 require(near3(meanRing(out,dynamicFirst,ropeSides),anchor),"slack update moved the fixed anchor");
 require(near3(meanRing(out,dynamicFirst+(ropeRings-1)*ropeSides,ropeSides),endpoint0),"slack update moved the harness endpoint");
 const V3 slackMid6=meanRing(out,ropeMidIndex,ropeSides);
 require(len(sub(slackMid6,slackMid5))>1.0e-4f,"slack change did not alter visual sag");
 rope.anchorAndLength.w=len(sub(endpoint0,anchor));dispatch();out=static_cast<MRVisualVertexGPUV2*>(verticesBuffer.contents);
 require(near3(meanRing(out,dynamicFirst+(ropeRings-1)*ropeSides,ropeSides),endpoint0),"taut endpoint changed");
 const V3 tautMid=meanRing(out,ropeMidIndex,ropeSides);
 require(near3(tautMid,mul(add(anchor,endpoint0),0.5f),5.0e-6f),"taut rope midpoint is not on the anchor-endpoint chord");
 bodies[bodyIndex].position={2.0f,-1.0f,4.0f,1.0f};
 bodies[bodyIndex].orientation={s,0,0,s};rope.anchorAndLength.w=6.0f;dispatch();checkDynamicFinite();out=static_cast<MRVisualVertexGPUV2*>(verticesBuffer.contents);
 const V3 endpoint1=expectedEndpoint();
 require(near3(meanRing(out,dynamicFirst,ropeSides),anchor),"body motion moved fixed ceiling anchor");
 require(near3(meanRing(out,dynamicFirst+(ropeRings-1)*ropeSides,ropeSides),endpoint1),"rope endpoint did not follow rotated/translated accepted body site");
 require(near3(meanRing(out,harnessFirst,harnessCount),endpoint1,5.0e-6f),"harness ring did not follow rotated/translated accepted body site");
 for(std::uint32_t i=0;i<dynamicFirst;++i)require(std::memcmp(&vertices[i],&sentinel,sizeof(sentinel))==0,"kernel overwrote leading guard vertex");
 for(std::uint32_t i=harnessFirst+harnessCount;i<totalVertices;++i)require(std::memcmp(&vertices[i],&sentinel,sizeof(sentinel))==0,"kernel overwrote trailing guard vertex");
 std::cout<<"PASS visual dispatch on "<<device.name.UTF8String<<"; endpoint="<<endpoint1.x<<","<<endpoint1.y<<","<<endpoint1.z<<" anchor="<<anchor.x<<","<<anchor.y<<","<<anchor.z<<" slack-sag-delta="<<len(sub(slackMid6,slackMid5))<<"\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
