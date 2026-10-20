#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/engine_types.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
static void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(int argc,char** argv){try{@autoreleasepool{
 check(argc==2,"usage: check NumiMatter.metallib");
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();check(device!=nil,"Metal unavailable");
 NSError* error=nil;
 id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
 check(lib!=nil,error?error.localizedDescription.UTF8String:"library unavailable");
 id<MTLFunction> function=[lib newFunctionWithName:@"numi_matter_metal::nm_numi_human_audit_passive_ligaments"];
 check(function!=nil,"existing ligament audit kernel unavailable");
 id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
 check(pipeline!=nil,"pipeline unavailable");id<MTLCommandQueue> queue=[device newCommandQueue];
 NMNumiHumanTendonFEMLoadDispatchGPU d{};d.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
 d.environmentCount=1;d.bodyPoseStride=2;d.passiveLigamentCount=1;
 NMNumiHumanPassiveLigamentGPU r{};r.firstBodyIndex=0;r.secondBodyIndex=1;r.flags=NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE;
 r.material={3e10f,.001f,float(3e7*std::exp(.001*(double(1.05f)-1))),1.05f};
 r.reference={1,3e-5f,1,0};
 id<MTLBuffer> rows=[device newBufferWithLength:sizeof(r) options:MTLResourceStorageModeShared];
 id<MTLBuffer> bodies=[device newBufferWithLength:2*sizeof(MRArticulatedBodyPoseGPU) options:MTLResourceStorageModeShared];
 id<MTLBuffer> out=[device newBufferWithLength:sizeof(NMNumiHumanPassiveLigamentAuditGPU) options:MTLResourceStorageModeShared];
 *static_cast<NMNumiHumanPassiveLigamentGPU*>(rows.contents)=r;
 unsigned cases=0;double maxRelative=0;
 for(unsigned rotation=0;rotation<2;++rotation)for(float lambda:{.99f,1.f,1.00000011920928955078125f,1.00001f,1.0001f,1.001f,1.049f,1.05f,1.1f}){
  auto* poses=static_cast<MRArticulatedBodyPoseGPU*>(bodies.contents);
  poses[0]={{0,0,0,0},{0,0,0,1}};
  poses[1]={{rotation?0:lambda,rotation?lambda:0,0,0},{0,0,0,1}};
  id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  [e setComputePipelineState:pipeline];[e setBytes:&d length:sizeof(d) atIndex:0];
  [e setBuffer:rows offset:0 atIndex:1];[e setBuffer:bodies offset:0 atIndex:2];[e setBuffer:out offset:0 atIndex:3];
  [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
  check(cb.status==MTLCommandBufferStatusCompleted,"GPU dispatch failed");
  const auto& a=*static_cast<NMNumiHumanPassiveLigamentAuditGPU*>(out.contents);
  const double stretch=a.stretchCountAndAccepted.x;double stress=0;
  if(stretch>1&&stretch<r.material.w)stress=double(r.material.x)*std::expm1(double(r.material.y)*(stretch-1));
  else if(stretch>=r.material.w)stress=double(r.material.x)*std::expm1(double(r.material.y)*(double(r.material.w)-1))+double(r.material.z)*(stretch-double(r.material.w));
  const double expected=stress*double(r.reference.y),actual=a.momentResidualAndMaximumTension.w;
  const double relative=std::abs(actual-expected)/std::max(expected,1e-12);maxRelative=std::max(maxRelative,relative);
  if(!std::isfinite(actual)||relative>=2e-5)
   std::fprintf(stderr,"lambda=%.9g actual_N=%.9g reference_N=%.9g relative_error=%.9g\n",stretch,actual,expected,relative);
  check(std::isfinite(actual)&&relative<2e-5,"GPU tension differs from analytic reference");
  check(expected==0||actual>0,"positive small-strain force rounded to zero");
  check(a.forceResidualAndL1.x==0&&a.forceResidualAndL1.y==0&&a.forceResidualAndL1.z==0,"internal force residual");
  check(std::abs(a.momentResidualAndMaximumTension.x)+std::abs(a.momentResidualAndMaximumTension.y)+std::abs(a.momentResidualAndMaximumTension.z)<1e-6,"internal moment residual");
  check(a.stretchCountAndAccepted.z==1,"row count changed");++cases;
 }
 std::printf("PASS existing_Metal_ligament_owner device=%s cases=%u max_relative_force_error=%.9g compression_zero=1 low_strain_nonzero=1 force_moment_closed=1\n",device.name.UTF8String,cases,maxRelative);
}}catch(const std::exception& e){
 std::fprintf(stderr,"FAIL %s\n",e.what());return 1;
}}
