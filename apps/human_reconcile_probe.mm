#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "numi/matter/numi_human_shared.h"
#include <array>
#include <cstring>
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
id<MTLComputePipelineState> pipeline(id<MTLDevice> device, const char* path, NSString* name) {
    NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
    require(library!=nil,"metallib missing");
    auto function=[library newFunctionWithName:name]; require(function!=nil,"kernel missing");
    auto result=[device newComputePipelineStateWithFunction:function error:&error];
    require(result!=nil,"pipeline creation failed"); return result;
}
}
int main() {
 @autoreleasepool {
  try {
    auto device=MTLCreateSystemDefaultDevice(); require(device!=nil,"Metal unavailable");
    auto adapt=pipeline(device,NUMI_MATTER_METALLIB,@"numi_matter_metal::nm_numi_human_adapt_stand_status");
    auto restore=pipeline(device,METALROBO_METALLIB,@"mr_numi_human_stand_reconcile");
    auto queue=[device newCommandQueue];
    constexpr unsigned count=6, nq=7, nv=6, muscles=2, vectors=18;
    for(unsigned step: {0u,7u}) {
      std::array<float,count*nq> q{}, q0{};
      std::array<float,count*nv> v{}, v0{};
      std::array<float,count*vectors> scratch{}, scratch0{};
      std::array<MRMujocoMuscleStateGPU,count*muscles> m{}, m0{};
      std::array<MRNumiHumanStandStatusGPU,count> status{}, status0{};
      std::array<NMMatterStatusGPU,count> matter{};
      for(unsigned i=0;i<q.size();++i) {q[i]=100.f+i;q0[i]=-100.f-i;}
      for(unsigned i=0;i<v.size();++i) {v[i]=200.f+i;v0[i]=-200.f-i;}
      for(unsigned i=0;i<scratch.size();++i) {scratch[i]=300.f+i;scratch0[i]=-300.f-i;}
      std::memset(m.data(),0x35,sizeof(m)); std::memset(m0.data(),0x43,sizeof(m0));
      for(unsigned e=0;e<count;++e) {
        status[e].environment=e; status[e].completedSteps=step+1;
        status[e].tendonTransferCount=100; status0[e].environment=e;
        status0[e].completedSteps=step; status0[e].tendonTransferCount=step?20:0;
        matter[e].environment=e; matter[e].completedMicrosteps=1;
      }
      matter[1].code=NM_STATUS_CONTACT_FAILURE; matter[1].failingIndex=1040; matter[1].objectIndex=17;
      status[2].code=MR_NUMI_HUMAN_STAND_CONTACT_FAILED;
      matter[3].completedMicrosteps=0; matter[4].environment=0;
      status[5].completedSteps=step;
      const auto buffer=[&](const void* data,std::size_t bytes) {
        auto b=[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        require(b!=nil,"buffer allocation failed"); return b;
      };
      std::array<id<MTLBuffer>,10> b = {
        buffer(q.data(),sizeof(q)),buffer(v.data(),sizeof(v)),buffer(m.data(),sizeof(m)),
        buffer(status.data(),sizeof(status)),buffer(scratch.data(),sizeof(scratch)),
        buffer(q0.data(),sizeof(q0)),buffer(v0.data(),sizeof(v0)),buffer(m0.data(),sizeof(m0)),
        buffer(status0.data(),sizeof(status0)),buffer(scratch0.data(),sizeof(scratch0))};
      auto matterBuffer=buffer(matter.data(),sizeof(matter));
      auto world=[device newBufferWithLength:count*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
      auto command=[queue commandBuffer]; auto encoder=[command computeCommandEncoder];
      require(encoder!=nil,"encoder unavailable");
      NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
      dispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
      dispatch.environmentCount=count; dispatch.stepIndex=step; dispatch.muscleCount=muscles;
      [encoder setComputePipelineState:adapt];
      [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
      [encoder setBuffer:b[3] offset:0 atIndex:1]; [encoder setBuffer:world offset:0 atIndex:2];
      [encoder setBuffer:matterBuffer offset:0 atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
      [encoder endEncoding]; encoder=[command computeCommandEncoder];
      const mr_uint4 shape={count,step,muscles,vectors}, strides={nq,nv,0,0};
      [encoder setComputePipelineState:restore];
      [encoder setBytes:&shape length:sizeof(shape) atIndex:0];
      [encoder setBytes:&strides length:sizeof(strides) atIndex:1];
      for(unsigned i=0;i<b.size();++i) [encoder setBuffer:b[i] offset:0 atIndex:i+2];
      [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      require(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
      const auto* actual=static_cast<const MRNumiHumanStandStatusGPU*>(b[3].contents);
      const auto* worlds=static_cast<const MRMetalWorldStatusGPU*>(world.contents);
      const std::array<std::size_t,5> sizes={nq*sizeof(float),nv*sizeof(float),muscles*sizeof(MRMujocoMuscleStateGPU),sizeof(MRNumiHumanStandStatusGPU),vectors*sizeof(float)};
      const std::array<const void*,5> candidate={q.data(),v.data(),m.data(),status.data(),scratch.data()};
      for(unsigned e=0;e<count;++e) {
        require((worlds[e].code==MR_STEP_SUCCESS)==(e==0),"joint admission mismatch");
        for(unsigned field=0;field<5;++field) {
          if(field==3) continue;
          const auto* expected=static_cast<const char*>(e==0?candidate[field]:b[field+5].contents)+e*sizes[field];
          require(std::memcmp(static_cast<const char*>(b[field].contents)+e*sizes[field],expected,sizes[field])==0,"physical checkpoint mismatch");
        }
        require(actual[e].completedSteps==(e==0?step+1:step),"accepted step count mismatch");
        require(actual[e].tendonTransferCount==(e==0?100:(step?20:0)),"accepted tendon count mismatch");
        require((actual[e].code==MR_NUMI_HUMAN_STAND_SUCCESS)==(e==0),"Human rejection missing");
        if(e==1) require(actual[e].failingIndex==17,"first failure identity lost");
        std::printf("PASS human_joint_reconcile step=%u environment=%u code=%u accepted_steps=%u\n",step,e,actual[e].code,actual[e].completedSteps);
      }
    }
    std::puts("human_joint_reconcile_cases=12 passed=12 failed=0"); return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"human_reconcile_probe: %s\n",e.what());return 1;}
 }
}
