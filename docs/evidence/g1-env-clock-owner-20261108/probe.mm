#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
// Bounded native composition of actual task, tracker, rendering and physics.
// Compares an intended mixed-overflow arm with an all-clear control.
#include "/Users/numi/numi-g1-transaction-candidate-owner-20261107/src/c_api_retained_research.cpp"
#include "ResearchVisualCadence.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using ResearchHook = bool (*)(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t);
extern "C" void mr_research_set_observation_hooks(void*,ResearchHook,ResearchHook) noexcept;

struct SavedBuffer {
 __strong id<MTLBuffer> buffer=nil;
 __strong id<MTLBuffer> backup=nil;
 std::uint32_t words=0;
};
static std::vector<SavedBuffer> saved;
static metalrobo::MetalHybridObjectTracker* tracker=nullptr;
static __strong id<MTLDevice> researchDevice=nil;
static __strong id<MTLComputePipelineState> restorePipeline=nil;
static __strong id<MTLBuffer> actorReadback=nil;
static __strong id<MTLBuffer> cadenceBuffer=nil;
static __strong id<MTLComputePipelineState> clockPipeline=nil;
static __strong id<MTLBuffer> envClock=nil;
static __strong id<MTLBuffer> envCandidate=nil;
static __strong id<MTLComputePipelineState> envClockPipeline=nil;
static int hookPhase=0;
static int guardRefusals=0;
static bool prepareHook(void*,void* rawcmd,void* rawhist,void* rawactor,void*,void*,void*,void*,std::uint32_t envs,std::uint32_t frame,std::uint32_t length,std::uint32_t obs) {
 if(hookPhase!=0 || !tracker || envs!=2 || !cadenceBuffer) return false;
 auto* clocks=static_cast<std::uint32_t*>(cadenceBuffer.contents);
 if(clocks[0]!=clocks[2]) {++guardRefusals; return false;}
 id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)rawcmd;
 if(!restorePipeline) {
  if(!researchDevice) researchDevice=MTLCreateSystemDefaultDevice();
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-clock-owner-build-20261108/shaders/restore_fresh.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];
  if(!lib) return false;
  restorePipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_selective_restore"] error:&error];
  if(!restorePipeline) return false;
  clockPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_accept_sensor_clock"] error:&error];
  if(!clockPipeline) return false;
 }
 if (envClock && !envClockPipeline) {
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-clock-owner-build-20261108/shaders/numi-research-cadence-20261108.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];
  if(!lib) return false;
  envClockPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_commit_visual_clock"] error:&error];
  if(!envClockPipeline) return false;
 }
 auto borrowed=tracker->researchBorrowBuffers();
 if(borrowed.size()!=6) return false;
 saved.clear();
 auto add=[&](void* raw,std::uint32_t words)->bool {
  id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)raw;
  const std::size_t bytes=std::size_t(words)*envs*sizeof(std::uint32_t);
  if(!buffer || buffer.length<bytes || !words) return false;
  id<MTLBuffer> backup=[researchDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  if(!backup) return false;
  saved.push_back({buffer,backup,words}); return true;
 };
 // This fixture has six tracks, one frame of history and no masked-depth ring.
 if(frame!=140 || length!=1 || obs!=140) return false;
 if(!add(rawhist,frame*length) || !add(rawactor,obs) ||
    !add(borrowed[0],6*length*8) || !add(borrowed[2],16*9) ||
    !add(borrowed[3],16*9*4) || !add(borrowed[4],16*9) ||
    !add(borrowed[5],16)) return false;
 actorReadback=[researchDevice newBufferWithLength:frame*length*envs*4 options:MTLResourceStorageModeShared];
 if(!actorReadback) return false;
 id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder];
 if(!blit) return false;
 for(const auto& b:saved) [blit copyFromBuffer:b.buffer sourceOffset:0 toBuffer:b.backup destinationOffset:0 size:std::size_t(b.words)*envs*4];
 [blit endEncoding]; hookPhase=1; return true;
}
static bool finishHook(void*,void* rawcmd,void*,void*,void*,void*,void* rawworld,void* rawcontact,std::uint32_t envs,std::uint32_t,std::uint32_t,std::uint32_t) {
 if(hookPhase!=1 || envs!=2 || !rawworld || !rawcontact) return false;
 id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)rawcmd;
 id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
 if(!enc) return false;
 [enc setComputePipelineState:restorePipeline];
 for(const auto& b:saved) {
  [enc setBuffer:b.buffer offset:0 atIndex:0]; [enc setBuffer:b.backup offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)rawworld offset:0 atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)rawcontact offset:0 atIndex:3];
  [enc setBytes:&b.words length:sizeof(b.words) atIndex:4];
  [enc setBytes:&envs length:sizeof(envs) atIndex:5];
  NSUInteger threads=std::min<NSUInteger>(restorePipeline.maxTotalThreadsPerThreadgroup,128);
  [enc dispatchThreads:MTLSizeMake(std::size_t(b.words)*envs,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
 }
 [enc setComputePipelineState:clockPipeline];
 [enc setBuffer:cadenceBuffer offset:0 atIndex:0];
 [enc setBuffer:(__bridge id<MTLBuffer>)rawworld offset:0 atIndex:1];
 [enc setBuffer:(__bridge id<MTLBuffer>)rawcontact offset:0 atIndex:2];
 [enc setBytes:&envs length:sizeof(envs) atIndex:3];
 [enc dispatchThreads:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(envs,1,1)];
 if(envClock) {
  [enc setComputePipelineState:envClockPipeline];
  [enc setBuffer:envClock offset:0 atIndex:0]; [enc setBuffer:envCandidate offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)rawworld offset:0 atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)rawcontact offset:0 atIndex:3];
  [enc setBytes:&envs length:sizeof(envs) atIndex:4];
  [enc dispatchThreads:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(envs,1,1)];
 }
 [enc endEncoding];
 id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder]; if(!blit) return false;
 [blit copyFromBuffer:saved[0].buffer sourceOffset:0 toBuffer:actorReadback destinationOffset:0 size:actorReadback.length];
 [blit endEncoding]; hookPhase=2; return true;
}
static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
static std::unique_ptr<MRTaskRolloutHandle> make(const char* metallib,const char* assetDir,bool failing) {
 std::array<MRTaskRolloutDynamicSphereC,6> balls{};
 for(std::size_t i=0;i<balls.size();++i) {balls[i].position[0]=4.f+float(i); balls[i].position[2]=2.f; balls[i].radius=.1f; balls[i].mass=1.f; balls[i].launch_step=0;}
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=4081015u;cfg.dynamic_spheres=balls.data();cfg.dynamic_sphere_count=6;
 balls[5].position[0]=.08f; balls[5].position[1]=0.f; balls[5].position[2]=2.f; balls[5].linear_velocity[2]=-1.f; balls[5].linear_velocity[0]=2.f;
 auto manifest=makeUnitreeG1RunManifest(metalrobo::LocomotionSurface::ground,metalrobo::UnitreeG1Task::ballDisturbanceRecovery,cfg);
 manifest.task.capacities.rawContacts=8;
 // Preserve deterministic authored positions for the overflow intervention.
 std::erase_if(manifest.reality.reset.operators,[](const auto& op) {return op.operation==metalrobo::TaskRandomizationOperator::sceneBodyPosition || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyVelocity || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyLaunchStep || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyEventImpact;});
 std::array<std::string,6> names,paths;
 std::array<MRTaskVisualPackC,6> packs{};
 for(std::size_t i=0;i<6;++i) {
  names[i]="locomotion_dynamic_sphere_"+std::to_string(i);
  paths[i]=(std::filesystem::path(assetDir)/(names[i]+".mrvpack")).string();
  packs[i]={paths[i].c_str(),names[i].c_str(),2u,static_cast<std::uint32_t>(100u+i)};
 }
 MRTaskVisualObservationConfigC visual{};visual.packs=packs.data();visual.pack_count=6;
 visual.camera_parent_body="torso_link";visual.camera_position[0]=.08f;visual.camera_position[2]=.45f;
 visual.camera_orientation[3]=1.f;
 visual.width=16;visual.height=9;visual.minimum_visible_pixels=1;visual.vertical_field_of_view_degrees=54;visual.nominal_rate_hz=50;
 auto h=createCompiledRunTaskRollout(std::move(manifest),metallib,"visual rejection fixture",&visual);
 require(h->environmentCount==2 && h->visualRuntime!=nullptr,"visual owner unavailable");
 h->stepConfig.publishStateTrajectory=true; h->stepConfig.publishFinalState=true;
 // Both robots start high enough to avoid ordinary floor contact. The failing
 // arm alone places five converging rigid spheres away from the elevated robot.
 const auto layout=mr_task_rollout_layout(h.get());
 h->resetQ[2]+=1.f;h->resetQ[layout.nq+2]+=1.f;
 const auto n=h->defaultSceneBodies.size();
 require(n>=7,"ground plus six rigid spheres required");
 if(failing) {
  for(std::size_t i=1;i<=5;++i) {
   const float angle=float(i-1)*2.f*3.14159265358979323846f/5.f;
   const float x=std::cos(angle), y=std::sin(angle);
   h->resetSceneBodies[i].position.x=4.f+.38f*x;
   h->resetSceneBodies[i].position.y=.38f*y;
   h->resetSceneBodies[i].position.z=2.f;
   h->resetSceneBodies[i].linearVelocityAndInverseMass.x=-5.f*x;
   h->resetSceneBodies[i].linearVelocityAndInverseMass.y=-5.f*y;
  }
 }
 metalrobo::PolicyPack policy;
 policy.id="visual_rejection_zero_policy";
 const auto& policyLayout=h->taskProgram.layout();
 policy.layers.push_back({.inputCount=policyLayout.actorObservationSize,
  .outputCount=policyLayout.actionCount,
  .activation=metalrobo::PolicyActivation::identity,
  .weights=std::vector<float>(policyLayout.actorObservationSize*policyLayout.actionCount,0.f),
  .bias=std::vector<float>(policyLayout.actionCount,0.f)});
 metalrobo::bindPolicyPack(policy,h->taskProgram);
 installPolicyPack(*h,policy);
 return h;
}
int main(int argc,char** argv) {
 try {
  require(argc==4,"usage: tracker_probe METALLIB ASSET_DIRECTORY SNAPSHOT_TSV");
  const bool instrumented=std::string(argv[3])!="-";
  std::ofstream snapshotFile;
  if(instrumented) snapshotFile.open(argv[3]);
  require(!instrumented || bool(snapshotFile),"snapshot file creation failed");
  if(instrumented) snapshotFile << "arm\tphase\tarray\tindex\tbits\n";
  auto capture=[&](MRTaskRolloutHandle* h,const char* arm,int phase) {
   if(!instrumented) return;
   metalrobo::TrackerProbeSnapshot x;
   require(h->visualRuntime->tracker.researchSnapshot(x),"snapshot failed");
   std::cerr << "snapshot arm="<<arm<<" phase="<<phase
    <<" sequence="<<x.observationSequence<<" sensor="<<x.lastSensorSequence
    <<" envs="<<x.environments<<" tracks="<<x.tracks
    <<" track_history="<<x.trackHistoryLength<<" depth_width="<<x.depthWidth
    <<" depth_height="<<x.depthHeight<<" depth_slots="<<x.depthSlots<<'\n';
   for(std::size_t i=0;i<x.trackHistory.size();++i)
    snapshotFile<<arm<<'\t'<<phase<<"\ttrack\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.trackHistory[i])<<'\n';
   for(std::size_t i=0;i<x.cameraState.size();++i)
    snapshotFile<<arm<<'\t'<<phase<<"\tcamera\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.cameraState[i])<<'\n';
   auto dump=[&](const char* label,const std::vector<std::uint32_t>& values) {
    for(std::size_t i=0;i<values.size();++i)
     snapshotFile<<arm<<'\t'<<phase<<'\t'<<label<<'\t'<<i<<'\t'<<values[i]<<'\n';
   };
   dump("frameDepth",x.frameDepthBits); dump("frameIdentity",x.frameIdentityBits);
   dump("frameValidity",x.frameValidityBits); dump("binding",x.bindingIds);
   for(std::size_t i=0;i<x.depthHistory.size();++i)
    snapshotFile<<arm<<'\t'<<phase<<"\tdepth\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.depthHistory[i])<<'\n';
  };
  auto control=make(argv[1],argv[2],false); auto failure=make(argv[1],argv[2],true);
  const auto c=mr_task_rollout_layout(control.get()), f=mr_task_rollout_layout(failure.get());
  require(c.world_fingerprint==f.world_fingerprint && c.task_fingerprint==f.task_fingerprint && c.observation_fingerprint==f.observation_fingerprint && c.action_fingerprint==f.action_fingerprint,"arm contract mismatch");
  std::cerr << "world="<<f.world_fingerprint<<" task="<<f.task_fingerprint<<" observation="<<f.observation_fingerprint<<" action="<<f.action_fingerprint<<" policy="<<failure->stepConfig.policyProgram.fingerprint()<<" sensor_offset="<<(failure->taskProgram.layout().actorFrameSize*failure->taskProgram.layout().actorHistoryLength+failure->taskProgram.header().counts3.w)<<" actor_count="<<f.actor_observation_count<<'\n';
  std::cout << "arm\tstep\tenv\tstatus\tphysics_error\tdone\tactor_component\tvalue\n" << std::setprecision(9);
  for(int arm=0;arm<2;++arm) {
   auto* h=arm?failure.get():control.get();
   tracker=&h->visualRuntime->tracker;
   envClock=nil; envCandidate=nil; envClockPipeline=nil;
   researchDevice=MTLCreateSystemDefaultDevice();
   cadenceBuffer=[researchDevice newBufferWithLength:4*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
   require(cadenceBuffer!=nil,"cadence allocation failed");
   auto* initialClocks=static_cast<std::uint32_t*>(cadenceBuffer.contents);
   initialClocks[0]=0;initialClocks[1]=UINT32_MAX;initialClocks[2]=0;initialClocks[3]=UINT32_MAX;
   mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepareHook,finishHook);
   capture(h,arm?"failure":"control",0);
   
   for(int step=1;step<=4;++step) {
    if (step==3) {
     envClock=[researchDevice newBufferWithLength:2*sizeof(research_cadence::Record) options:MTLResourceStorageModeShared];
     envCandidate=[researchDevice newBufferWithLength:2*sizeof(research_cadence::Record) options:MTLResourceStorageModeShared];
     require(envClock && envCandidate,"new clock buffers");
     auto* v=static_cast<research_cadence::Record*>(envClock.contents);
     for(int e=0;e<2;++e) v[e]={2,1,1,0};
     require(tracker->researchSetCadenceBuffer((__bridge void*)envClock),"attach accepted clock");
    }
    if (step>=3 && !(arm && step==4)) {
     auto* v=static_cast<research_cadence::Record*>(envClock.contents);
     auto* candidates=static_cast<research_cadence::Record*>(envCandidate.contents);
     for(int e=0;e<2;++e) {
      research_cadence::Candidate next{};
      require(research_cadence::propose(v[e],.02f,50.f,1000+step,false,next),"clock proposal");
      candidates[e]=next.accepted;
      std::cerr<<"clock_proposal arm="<<(arm?"failure":"control")<<" step="<<step<<" env="<<e<<" sensor="<<next.sensor<<" sample="<<next.sample<<'\n';
     }
    }
    if (arm && step==4) {
     hookPhase=0; const int beforeGuard=guardRefusals; MRTaskRolloutAdvanceC blocked{};
     const int denied=mr_task_rollout_advance(h,nullptr,0,nullptr,0,1,0,0,&blocked);
     std::cerr<<"continuation_guard rc="<<denied<<" hook_phase="<<hookPhase<<" error="<<mr_last_error()<<'\n';
     require(denied!=0 && hookPhase==0 && guardRefusals==beforeGuard+1,"divergence not refused before observation");
     capture(h,"failure",4);
     break;
    }
    hookPhase=0; MRTaskRolloutAdvanceC advance{}; int rc=mr_task_rollout_advance(h,nullptr,0,nullptr,0,1,0,0,&advance);
    require(hookPhase==2,"prepare/finish hooks missing or out of order");
    if(rc!=0) std::cerr << (arm?"failure":"control") << " step=" << step << " advance error: " << mr_last_error() << std::endl;
    for(std::size_t e=0;e<h->result.contactStatuses.size();++e) {
     const auto& contact=h->result.contactStatuses[e];
     std::cerr<<"contact arm="<<(arm?"failure":"control")<<" step="<<step<<" env="<<e<<" code="<<contact.code<<" raw_required="<<contact.requiredRawContacts<<" active="<<contact.activeContacts<<'\n';
    }
    const auto count=h->defaultSceneBodies.size();
    for(std::size_t e=0;e<2;++e) for(std::size_t i=1;i<count;++i) {
     if(e*count+i>=h->result.finalSceneBodies.size()) continue;
     const auto& b=h->result.finalSceneBodies[e*count+i];
     std::cerr<<"body arm="<<(arm?"failure":"control")<<" step="<<step<<" env="<<e<<" index="<<i<<" p="<<b.position.x<<","<<b.position.y<<","<<b.position.z<<" v="<<b.linearVelocityAndInverseMass.x<<","<<b.linearVelocityAndInverseMass.y<<","<<b.linearVelocityAndInverseMass.z<<'\n';
    }
    auto* values=mr_task_rollout_actor_observations(h);auto* status=mr_task_rollout_status_codes(h);auto* trans=mr_task_rollout_transitions(h);
    require(values && status && trans,"missing outputs");
    auto* clocks=static_cast<std::uint32_t*>(cadenceBuffer.contents);
    std::cerr<<"cadence_after arm="<<(arm?"failure":"control")<<" step="<<step<<" env0="<<clocks[0]<<","<<clocks[1]<<" env1="<<clocks[2]<<","<<clocks[3]<<std::endl;
    for(int e=0;e<2;++e) {
     const auto* bits=static_cast<const std::uint32_t*>(actorReadback.contents);
     for(std::uint32_t j=0;j<140;++j)
      std::cerr<<"actorbits "<<(arm?"failure":"control")<<" "<<step<<" "<<e<<" "<<j<<" "<<bits[e*140+j]<<std::endl;
    }

    for(int e=0;e<2;++e) for(std::uint32_t j=0;j<c.actor_observation_count;++j)
     std::cout << (arm?"failure":"control") << '\t' << step << '\t' << e << '\t' << status[e] << '\t' << trans[e].physics_error << '\t' << trans[e].done << '\t' << j << '\t' << values[e*c.actor_observation_count+j] << '\n';
    capture(h,arm?"failure":"control",step);
    if (envClock) {
     auto* v=static_cast<research_cadence::Record*>(envClock.contents);
     for(int e=0;e<2;++e) std::cerr<<"env_clock arm="<<(arm?"failure":"control")<<" step="<<step<<" env="<<e<<" words="<<v[e].acceptedSteps<<","<<v[e].lastSensor<<","<<v[e].captureStep<<","<<v[e].captureTimeBits<<'\n';
    }
    if(rc!=0 && !(arm && step==3)) break;
   }
  }
  mr_research_set_observation_hooks(nullptr,nullptr,nullptr);
  return 0;
 } catch(const std::exception& e) {std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}
}
