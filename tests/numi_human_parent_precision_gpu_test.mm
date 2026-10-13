#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "NumiHumanRestingSupportGeometry.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>
static void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
int main(int argc,char**argv) { @autoreleasepool { try {
 require(argc>=2&&argc<=4,"production source, optional published baseline and term count required");
 NSError* error=nil;id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"native Metal device required");
 NSString* source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];require(source!=nil,"source read failed");
 MTLCompileOptions* options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
 id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
 if(!library) { std::cerr<<error.localizedDescription.UTF8String<<"\n";return 1; }
 id<MTLFunction> function=numiHumanRestingMakeSkinInfluenceFunction(library,@"nm_human_resting_skin",{false,true,NumiHumanSkinInfluenceLayout::vertexMajor},&error);require(function!=nil,"production kernel missing");
 id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];require(pipeline!=nil,"production pipeline failed");
 constexpr unsigned cases=256;const unsigned terms=argc==4?unsigned(std::stoul(argv[3])):32u;require(terms>0&&terms<=512,"term count exceeds admitted parent domain");std::vector<MRHumanRestingVertexMap> maps(cases*2);std::vector<MRHumanRestingInfluence> influences(cases*2*terms);std::array<std::array<double,3>,cases> oracle{};std::array<std::array<float,3>,cases> legacy{};unsigned state=0x4e554d49u;
 for(unsigned i=0;i<cases;++i) {
  for(unsigned j=0;j<terms;++j) {
   state=state*1664525u+1013904223u;const unsigned bits=state>>16;
   const float x=1.0f+float(bits%32)*std::ldexp(1.0f,-23);
   const float y=0.5f+float((bits>>5)%32)*std::ldexp(1.0f,-24);
   const float z=0.25f+float((bits>>10)%32)*std::ldexp(1.0f,-25);
   for(unsigned flag=0;flag<2;++flag) {
    auto& f=influences[(i*2+flag)*terms+j];f.positionAndWeight={x,y,z,1.0f/terms};f.normal={0,0,1,0};f.body={i%2?0u:MR_INVALID_INDEX,flag,0,0};
   }
   const float values[3]={x,y,z};for(unsigned axis=0;axis<3;++axis) {const float term=values[axis]*(1.0f/terms);oracle[i][axis]+=double(term);legacy[i][axis]=std::fma(values[axis],1.0f/terms,legacy[i][axis]);}
  }
  for(unsigned flag=0;flag<2;++flag) {auto& m=maps[i*2+flag];m.firstInfluence=(i*2+flag)*terms;m.influenceCount=terms;}
 }
 constexpr unsigned normalCases=128;
 for(unsigned i=0;i<normalCases;++i)for(unsigned flag=0;flag<2;++flag) {
  MRHumanRestingVertexMap m{};m.firstInfluence=unsigned(influences.size());m.influenceCount=terms;
  for(unsigned j=0;j<terms;++j) {
   const unsigned bits=(i*1664525u+j*1013904223u)>>8;
   MRHumanRestingInfluence f{};f.positionAndWeight={0.17f+float(bits%1024)/4096.0f,-0.13f+float((bits>>5)%1024)/8192.0f,0.31f+float((bits>>10)%1024)/4096.0f,float(j+1)/float(terms*(terms+1)/2)};
   f.normal={0.271828f+float(bits%1024)/4096.0f,-0.14159f+float((bits>>5)%1024)/8192.0f,0.57721f+float((bits>>10)%1024)/4096.0f,0};f.body={i%2?1u:MR_INVALID_INDEX,flag,0,0};influences.push_back(f);
  }
  maps.push_back(m);
 }
 mr_uint4 dimensions={unsigned(maps.size()),0,0,0};std::array<MRBodyStateGPU,2> bodies{};bodies[0].orientation.w=1;bodies[1].orientation={0.18257418f,0.36514837f,0.54772256f,0.73029673f};bodies[1].position={0.11f,0.23f,-0.07f,0};
 std::array<id<MTLBuffer>,13> buffers{};for(auto& b:buffers)b=[device newBufferWithLength:16384 options:MTLResourceStorageModeShared];
 buffers[0]=[device newBufferWithBytes:&dimensions length:sizeof(dimensions) options:MTLResourceStorageModeShared];buffers[1]=[device newBufferWithBytes:maps.data() length:maps.size()*sizeof(maps[0]) options:MTLResourceStorageModeShared];buffers[2]=[device newBufferWithBytes:influences.data() length:influences.size()*sizeof(influences[0]) options:MTLResourceStorageModeShared];buffers[3]=[device newBufferWithBytes:bodies.data() length:sizeof(bodies) options:MTLResourceStorageModeShared];buffers[4]=[device newBufferWithLength:maps.size()*sizeof(MRVisualVertexGPUV2) options:MTLResourceStorageModeShared];
 for(unsigned index=5;index<13;++index)std::memset(buffers[index].contents,0,buffers[index].length);
 id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];for(unsigned index=0;index<13;++index)[encoder setBuffer:buffers[index] offset:0 atIndex:index];[encoder dispatchThreads:MTLSizeMake(maps.size(),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"native dispatch failed");
 auto* output=static_cast<MRVisualVertexGPUV2*>(buffers[4].contents);unsigned improved=0;double worst=0;
 for(unsigned i=0;i<cases;++i) {
  const auto a=output[i*2],b=output[i*2+1];const float av[3]={a.position.x,a.position.y,a.position.z},bv[3]={b.position.x,b.position.y,b.position.z};
  require(std::memcmp(&a.normalAndTangentSign,&b.normalAndTangentSign,sizeof(a.normalAndTangentSign))==0,"compensation changed normal");require(std::memcmp(&a.tangent,&b.tangent,sizeof(a.tangent))==0,"compensation changed tangent");
  for(unsigned axis=0;axis<3;++axis) {
   require(av[axis]==legacy[i][axis],"unflagged production path differs from ordered fused FP32 oracle");const float rounded=float(oracle[i][axis]);const double ulp=double(std::nextafter(rounded,INFINITY))-double(rounded);const double error=std::abs(double(bv[axis])-oracle[i][axis]);require(std::isfinite(bv[axis])&&error<=ulp,"compensated production path exceeds one ULP of independent FP64 sum");worst=std::max(worst,error);if(error<std::abs(double(av[axis])-oracle[i][axis]))++improved;
  }
 }
 for(unsigned i=cases;i<cases+normalCases;++i) {
  require(std::memcmp(&output[i*2].normalAndTangentSign,&output[i*2+1].normalAndTangentSign,sizeof(output[i*2].normalAndTangentSign))==0,"unequal-weight compensation changed normal");
  require(std::memcmp(&output[i*2].tangent,&output[i*2+1].tangent,sizeof(output[i*2].tangent))==0,"unequal-weight compensation changed tangent");
 }
 if(argc>=3) {
  NSString* legacySource=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];require(legacySource!=nil,"published baseline source read failed");
  id<MTLLibrary> legacyLibrary=[device newLibraryWithSource:legacySource options:options error:&error];require(legacyLibrary!=nil,"published baseline source compilation failed");
  id<MTLFunction> legacyFunction=numiHumanRestingMakeSkinInfluenceFunction(legacyLibrary,@"nm_human_resting_skin",{false,true,NumiHumanSkinInfluenceLayout::vertexMajor},&error);require(legacyFunction!=nil,"published baseline function specialization failed");
  id<MTLComputePipelineState> legacyPipeline=[device newComputePipelineStateWithFunction:legacyFunction error:&error];require(legacyPipeline!=nil,"published baseline pipeline failed");
  id<MTLBuffer> legacyOutput=[device newBufferWithLength:buffers[4].length options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> legacyCommand=[queue commandBuffer];id<MTLComputeCommandEncoder> legacyEncoder=[legacyCommand computeCommandEncoder];[legacyEncoder setComputePipelineState:legacyPipeline];
  for(unsigned index=0;index<13;++index)[legacyEncoder setBuffer:index==4?legacyOutput:buffers[index] offset:0 atIndex:index];
  [legacyEncoder dispatchThreads:MTLSizeMake(maps.size(),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[legacyEncoder endEncoding];[legacyCommand commit];[legacyCommand waitUntilCompleted];require(legacyCommand.status==MTLCommandBufferStatusCompleted,"published baseline native dispatch failed");
  const auto* reference=static_cast<const MRVisualVertexGPUV2*>(legacyOutput.contents);unsigned normalDifferences=0,tangentDifferences=0;
  for(unsigned i=0;i<maps.size();++i) {
   if(i%2==0)require(std::memcmp(&reference[i].position,&output[i].position,sizeof(reference[i].position))==0,"compensation changed unflagged published-baseline position");
   normalDifferences+=std::memcmp(&reference[i].normalAndTangentSign,&output[i].normalAndTangentSign,sizeof(reference[i].normalAndTangentSign))!=0;
   tangentDifferences+=std::memcmp(&reference[i].tangent,&output[i].tangent,sizeof(reference[i].tangent))!=0;
  }
  std::cout<<"publishedBaselineNormalDifferences="<<normalDifferences<<" publishedBaselineTangentDifferences="<<tangentDifferences<<" unequalWeightNormalFixtures="<<normalCases<<"\n";
  require(normalDifferences==0&&tangentDifferences==0,"position compensation changed published-baseline normals or tangents");
 }
 if(terms==32)require(improved>0,"fixture does not demonstrate retained low-order contributions");std::cout<<"{\"status\":\"PASS\",\"nativeProductionKernel\":true,\"syntheticParentFixtures\":"<<cases<<",\"termsPerFixture\":"<<terms<<",\"improvedComponents\":"<<improved<<",\"worstAbsoluteSummationError\":"<<worst<<",\"normalAndTangentByteIdentical\":true,\"scope\":\"synthetic presentation arithmetic; not anatomical admission\"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} } }
