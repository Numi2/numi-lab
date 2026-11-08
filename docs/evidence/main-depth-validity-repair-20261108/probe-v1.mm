#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/visual_platform_types.h"
#include "metalrobo/world_compiler_types.h"
#include "metalrobo/task_program_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main(int argc,char** argv) { @autoreleasepool {
 if(argc!=2){std::cerr<<"usage: probe shader.metallib\n";return 2;}
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
 id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
 id<MTLFunction> fn=[lib newFunctionWithName:@"mr_hybrid_masked_depth_history"];
 id<MTLComputePipelineState> ps=fn?[dev newComputePipelineStateWithFunction:fn error:&err]:nil;
 if(!ps){std::cerr<<"pipeline: "<<(err?[[err description] UTF8String]:"missing")<<'\n';return 3;}
 unsigned failures=0;
 auto test=[&](unsigned w,unsigned h,bool direct) {
  const unsigned n=w*h, envs=2,total=n*envs;
  id<MTLBuffer> b[16]{};
  auto buf=[&](NSUInteger len){id<MTLBuffer> q=[dev newBufferWithLength:len options:MTLResourceStorageModeShared];std::memset(q.contents,0,len);return q;};
  b[0]=buf(total*sizeof(float));b[1]=buf(total*sizeof(mr_uint4));b[2]=buf(total*sizeof(uint32_t));
  b[3]=buf(sizeof(uint32_t));b[4]=buf(total*3*sizeof(float));b[5]=buf(envs*sizeof(uint32_t));
  b[6]=buf(16*sizeof(float));b[7]=buf(total*sizeof(float));b[8]=buf(envs*sizeof(MRTaskStateGPU));
  b[10]=buf(total*sizeof(uint64_t));for(int i=11;i<=15;++i)b[i]=buf(4096);
  auto depths=(float*)b[0].contents;auto ids=(mr_uint4*)b[1].contents;auto val=(uint32_t*)b[2].contents;
  auto masks=(uint32_t*)b[5].contents;auto winners=(uint64_t*)b[10].contents;
  *(uint32_t*)b[3].contents=42; masks[0]=masks[1]=1;
  ((MRVisualPrimitiveGPUV2*)b[12].contents)->identity.y=42;
  auto sensor=(MRWorldSensorInstanceGPU*)b[15].contents;
  MRHybridMaskedDepthUniformsGPU u{};u.counts={envs,h,w,1};u.layout={n,1,0,envs};u.history={3,1,0,1};u.range={.1f,5.f,1.f/4.9f,0.f};u.curriculum.y=direct?1.f:0.f;
  MRHybridRenderUniformsGPU ru{};ru.live.w=1;ru.clearColorAndDepth.w=5.f;ru.sensorRangeAndResponse={.1f,5.f,0.f,0.f};
  auto run=[&](const char* phase) {
   id<MTLCommandQueue> q=[dev newCommandQueue];id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
   [e setComputePipelineState:ps];for(unsigned i=0;i<9;++i)[e setBuffer:b[i] offset:0 atIndex:i];[e setBytes:&u length:sizeof u atIndex:9];for(unsigned i=10;i<=15;++i)[e setBuffer:b[i] offset:0 atIndex:i];[e setBytes:&ru length:sizeof ru atIndex:16];
   [e dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
   if(cb.status!=MTLCommandBufferStatusCompleted){std::cerr<<"command failed "<<phase<<'\n';++failures;}
   std::vector<float> out(total);std::memcpy(out.data(),b[7].contents,total*sizeof(float));return out;
  };
  for(unsigned i=0;i<total;++i){depths[i]=.5f;ids[i].y=i<n?42u:99u;val[i]=MR_VISUAL_VALIDITY_FRAME|MR_VISUAL_VALIDITY_GEOMETRY|MR_VISUAL_VALIDITY_DEPTH;uint32_t bits;float d=.5f;std::memcpy(&bits,&d,sizeof bits);winners[i]=(uint64_t(bits)<<32);}
  if(direct){for(unsigned i=n;i<total;++i)winners[i]=UINT64_MAX;ru.shadowBatch.w=1;}
  auto valid=run("selected");unsigned badValid=0,badUnselected=0;
  for(unsigned i=0;i<n;++i)badValid+=std::abs(valid[i]-(.4f/4.9f))>1e-6f;
  for(unsigned i=n;i<total;++i)badUnselected+=std::abs(valid[i]-1.f)>1e-6f;
  failures+=badValid+badUnselected;std::cout<<"valid "<<w<<'x'<<h<<" direct="<<direct<<" selected_bad="<<badValid<<" unselected_bad="<<badUnselected<<'\n';
  for(unsigned i=0;i<total;++i){depths[i]=5.f;ids[i].y=42;val[i]=MR_VISUAL_VALIDITY_FRAME|MR_VISUAL_VALIDITY_GEOMETRY;float d=.5f;uint32_t bits;std::memcpy(&bits,&d,sizeof bits);winners[i]=uint64_t(bits)<<32;}
  if(direct)sensor->noiseAndLatency.z=1.f;
  u.corruption.w=1.f;
  auto invalid=run("sensor-depth-invalid");unsigned falseNear=0,nonfinite=0;
  for(float x:invalid){nonfinite+=!std::isfinite(x);falseNear+=x<1.f-1e-6f;}
  failures+=falseNear+nonfinite;std::cout<<"invalid "<<w<<'x'<<h<<" direct="<<direct<<" false_near="<<falseNear<<" nonfinite="<<nonfinite<<" total="<<total<<'\n';
 };
 test(4,2,false);test(20,20,false);test(4,2,true);
 std::cout<<"failures="<<failures<<'\n';return failures?1:0;
} }
