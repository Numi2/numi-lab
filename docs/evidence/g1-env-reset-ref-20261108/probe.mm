#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
// Bounded native composition of actual task, tracker, rendering and physics.
// Compares an intended mixed-overflow arm with an all-clear control.
#include "/Users/numi/numi-g1-env-reset-ref-owner-20261108/src/c_api_retained_research.cpp"
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
static __strong id<MTLBuffer> acceptedMetadata=nil;
static __strong id<MTLBuffer> candidateMetadata=nil;
static __strong id<MTLBuffer> metadataFlags=nil;
static __strong id<MTLComputePipelineState> metadataPipeline=nil;
static __strong id<MTLBuffer> resetReadback=nil;
static bool testPermitDivergence=false;
static int hookPhase=0;
static int guardRefusals=0;
static bool prepareHook(void*,void* rawcmd,void* rawhist,void* rawactor,void*,void*,void*,void*,std::uint32_t envs,std::uint32_t frame,std::uint32_t length,std::uint32_t obs) {
 if(hookPhase!=0 || !tracker || envs!=2 || !cadenceBuffer) return false;
 auto* clocks=static_cast<std::uint32_t*>(cadenceBuffer.contents);
 if(clocks[0]!=clocks[2] && !testPermitDivergence) {++guardRefusals; return false;}
 id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)rawcmd;
 if(!restorePipeline) {
  if(!researchDevice) researchDevice=MTLCreateSystemDefaultDevice();
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-reset-ref-build-20261108/shaders/restore_fresh.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];
  if(!lib) return false;
  restorePipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_selective_restore"] error:&error];
  if(!restorePipeline) return false;
  clockPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_accept_sensor_clock"] error:&error];
  if(!clockPipeline) return false;
 }
 if (envClock && !envClockPipeline) {
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-reset-ref-build-20261108/shaders/numi-research-cadence-20261108.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];
  if(!lib) return false;
  envClockPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_commit_visual_clock"] error:&error];
  if(!envClockPipeline) return false;
 }
 if(acceptedMetadata && !metadataPipeline) {
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-reset-ref-build-20261108/shaders/select.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];if(!lib)return false;
  metadataPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_rendered_select"] error:&error];if(!metadataPipeline)return false;
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
static bool finishHook(void*,void* rawcmd,void*,void*,void* rawReset,void*,void* rawworld,void* rawcontact,std::uint32_t envs,std::uint32_t,std::uint32_t,std::uint32_t) {
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
 if(acceptedMetadata) {
  const std::uint32_t stride=16;
  [enc setComputePipelineState:metadataPipeline];
  [enc setBuffer:acceptedMetadata offset:0 atIndex:0];[enc setBuffer:candidateMetadata offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)rawworld offset:0 atIndex:2];[enc setBuffer:(__bridge id<MTLBuffer>)rawcontact offset:0 atIndex:3];
  [enc setBuffer:metadataFlags offset:0 atIndex:4];[enc setBuffer:acceptedMetadata offset:0 atIndex:5];
  [enc setBytes:&stride length:sizeof(stride) atIndex:6];[enc setBytes:&envs length:sizeof(envs) atIndex:7];
  [enc dispatchThreads:MTLSizeMake(stride*envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
 }
 [enc endEncoding];
 id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder]; if(!blit) return false;
 [blit copyFromBuffer:saved[0].buffer sourceOffset:0 toBuffer:actorReadback destinationOffset:0 size:actorReadback.length];
 if(resetReadback) { if(!rawReset) return false; [blit copyFromBuffer:(__bridge id<MTLBuffer>)rawReset sourceOffset:0 toBuffer:resetReadback destinationOffset:0 size:envs*4]; }
 [blit endEncoding]; hookPhase=2; return true;
}
static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
static void savePartial(const char* output,const char* phase,const metalrobo::TrackerProbeSnapshot& x) {
 std::ofstream f(std::string(output)+".partial.tsv",std::ios::app);
 if(!f) throw std::runtime_error("partial output");
 auto emit=[&](const char* name,const auto& v){for(std::size_t i=0;i<v.size();++i)f<<phase<<'\t'<<name<<'\t'<<i<<'\t'<<v[i]<<'\n';};
 emit("depth",x.frameDepthBits);emit("identity",x.frameIdentityBits);emit("validity",x.frameValidityBits);emit("metadata",x.frameMetadataBits);
 for(std::size_t i=0;i<x.cameraState.size();++i)f<<phase<<"\tcamera\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.cameraState[i])<<'\n';
 for(std::size_t i=0;i<x.trackHistory.size();++i)f<<phase<<"\ttrack\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.trackHistory[i])<<'\n';
}
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
 visual.width=16;visual.height=9;visual.minimum_visible_pixels=1;visual.vertical_field_of_view_degrees=54;visual.nominal_rate_hz=25;
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
   h->resetSceneBodies[i].position.x=4.f+.58f*x;
   h->resetSceneBodies[i].position.y=.58f*y;
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
int main(int argc,char** argv) {@autoreleasepool {try {
 require(argc==4,"usage: probe METALLIB ASSETS OUT_TSV");
 auto h=make(argv[1],argv[2],true);tracker=&h->visualRuntime->tracker;tracker->researchEnableBodyCapture(true);researchDevice=MTLCreateSystemDefaultDevice();
 cadenceBuffer=[researchDevice newBufferWithLength:4*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
 envClock=[researchDevice newBufferWithLength:2*sizeof(research_cadence::Record) options:MTLResourceStorageModeShared];
 envCandidate=[researchDevice newBufferWithLength:2*sizeof(research_cadence::Record) options:MTLResourceStorageModeShared];
 require(cadenceBuffer && envClock && envCandidate,"clock buffers");
 auto* guard=static_cast<std::uint32_t*>(cadenceBuffer.contents);guard[0]=0;guard[1]=UINT32_MAX;guard[2]=0;guard[3]=UINT32_MAX;
 auto* accepted=static_cast<research_cadence::Record*>(envClock.contents);auto* candidates=static_cast<research_cadence::Record*>(envCandidate.contents);
 accepted[0]={};accepted[1]={};
 mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepareHook,finishHook);
 metalrobo::TrackerProbeSnapshot warmup,before,after;
 std::array<std::array<std::uint32_t,280>,2> actorBits{};
 for(int step=1;step<=5;++step) {
  for(unsigned e=0;e<2;++e){research_cadence::Candidate p{};require(research_cadence::propose(accepted[e],.02f,25.f,0,false,p),"proposal");
   candidates[e]=p.accepted;
   if(p.sample){candidates[e].captureStep=accepted[e].acceptedSteps;candidates[e].captureTimeBits=std::bit_cast<std::uint32_t>(float(double(accepted[e].acceptedSteps)*double(.02f)));}
   std::cerr<<"proposal step="<<step<<" env="<<e<<" count="<<accepted[e].acceptedSteps<<" sensor="<<p.sensor<<" sample="<<p.sample<<'\n';}
  if(step==3) {
   MRVisualFrameMetadataGPU prior{};std::memcpy(&prior,warmup.frameMetadataBits.data(),sizeof(prior));
   MRVisualFrameMetadataGPU priors[2]={prior,prior};const std::uint32_t flags[2]={1,1};
   acceptedMetadata=[researchDevice newBufferWithBytes:priors length:sizeof(priors) options:MTLResourceStorageModeShared];
   candidateMetadata=[researchDevice newBufferWithLength:sizeof(priors) options:MTLResourceStorageModeShared];
   metadataFlags=[researchDevice newBufferWithBytes:flags length:sizeof(flags) options:MTLResourceStorageModeShared];
   require(acceptedMetadata && candidateMetadata && metadataFlags,"metadata buffers");
   require(tracker->researchSetCadenceBuffer((__bridge void*)envClock),"cadence attachment");
   require(tracker->researchSetMixedCandidateBuffers((__bridge void*)acceptedMetadata,(__bridge void*)candidateMetadata),"candidate attachment");
  }
  hookPhase=0;MRTaskRolloutAdvanceC a{};int rc=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&a);
  require(hookPhase==2 && (step<5?rc==0:rc!=0),"expected status missing");
  auto* st=mr_task_rollout_status_codes(h.get());auto* tr=mr_task_rollout_transitions(h.get());require(st && tr,"missing status");
  if(step<5)require(st[0]==0 && st[1]==0 && !tr[0].done && !tr[1].done,"warmup failed");
  else {std::cerr<<"mixed_status world="<<st[0]<<","<<st[1]<<" contact="<<h->result.contactStatuses[0].code<<","<<h->result.contactStatuses[1].code<<" required0="<<h->result.contactStatuses[0].requiredRawContacts<<'\n';
   require(st[0]!=0 && st[1]==0 && h->result.contactStatuses[0].code!=0 && h->result.contactStatuses[1].code==0 && tr[0].done && !tr[1].done,"mixed status/terminal prerequisite");}
  for(unsigned e=0;e<2;++e)std::cerr<<"accepted step="<<step<<" env="<<e<<" words="<<accepted[e].acceptedSteps<<","<<accepted[e].lastSensor<<","<<accepted[e].captureStep<<","<<accepted[e].captureTimeBits<<'\n';
  if(step>=4) {
   require(actorReadback && actorReadback.length>=280*sizeof(std::uint32_t),"actor readback");
   std::memcpy(actorBits[step-4].data(),actorReadback.contents,280*sizeof(std::uint32_t));
  }
  if(step==2){require(tracker->researchSnapshot(warmup),"warmup snapshot");savePartial(argv[3],"warmup",warmup);}
  if(step==4){require(tracker->researchSnapshot(before),"before snapshot");savePartial(argv[3],"before",before);}
  if(step==5){require(tracker->researchSnapshot(after),"after snapshot");savePartial(argv[3],"after",after);}
 }
 require(accepted[0].acceptedSteps==4 && accepted[0].lastSensor==1 && accepted[1].acceptedSteps==5 && accepted[1].lastSensor==2 && accepted[1].captureStep==4 && accepted[1].captureTimeBits==1034147594u,"clock result");
 for(unsigned e=0;e<2;++e){research_cadence::Candidate p{};require(research_cadence::propose(accepted[e],.02f,25.f,0,false,p),"next proposal");
  std::cout<<"next env="<<e<<" count="<<accepted[e].acceptedSteps<<" sensor="<<p.sensor<<" sample="<<p.sample<<'\n';
  require(p.sensor==2 && p.sample==(e==0),"expected differing phases");}
 hookPhase=0;const int guardBefore=guardRefusals;MRTaskRolloutAdvanceC blocked{};
 const int denied=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&blocked);
 require(denied!=0 && guardRefusals==guardBefore+1 && hookPhase==0,"divergence guard failed");
 metalrobo::TrackerProbeSnapshot refused;require(tracker->researchSnapshot(refused),"guard snapshot");
 require(refused.frameDepthBits==after.frameDepthBits && refused.frameIdentityBits==after.frameIdentityBits && refused.frameValidityBits==after.frameValidityBits && refused.trackHistory==after.trackHistory && refused.cameraState==after.cameraState,"guard changed visual state");
 std::cout<<"divergence_guard=refused\n";
 std::array<std::uint32_t,32> metadataPrior{};
 std::memcpy(metadataPrior.data(),acceptedMetadata.contents,sizeof(metadataPrior));
 research_cadence::Record working[2]={research_cadence::Record{},accepted[1]};
 id<MTLBuffer> workingBuffer=[researchDevice newBufferWithBytes:working length:sizeof(working) options:MTLResourceStorageModeShared];
 resetReadback=[researchDevice newBufferWithLength:2*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
 require(workingBuffer && resetReadback,"working clock or reset readback");
 for(unsigned e=0;e<2;++e){research_cadence::Candidate p{};require(research_cadence::propose(working[e],.02f,25.f,0,false,p),"reset proposal");candidates[e]=p.accepted;
  if(p.sample)candidates[e].captureTimeBits=std::bit_cast<std::uint32_t>(float(double(working[e].acceptedSteps)*double(.02f)));
  std::cout<<"reset_proposal env="<<e<<" count="<<working[e].acceptedSteps<<" sensor="<<p.sensor<<" sample="<<p.sample<<'\n';
  require(p.sample==(e==0),"unexpected reset sample phases");}
 require(tracker->researchSetCadenceBuffer((__bridge void*)workingBuffer),"working cadence attachment");
 tracker->researchAllowMixedForOneStep();testPermitDivergence=true;hookPhase=0;MRTaskRolloutAdvanceC continuedStep{};
 const int continuation=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&continuedStep);testPermitDivergence=false;
 auto* resetWords=static_cast<const std::uint32_t*>(resetReadback.contents);
 auto* stepStatus=mr_task_rollout_status_codes(h.get());
 std::cout<<"continuation rc="<<continuation<<" hook="<<hookPhase<<" reset="<<resetWords[0]<<","<<resetWords[1];
 if(stepStatus)std::cout<<" world="<<stepStatus[0]<<","<<stepStatus[1]<<" contact="<<h->result.contactStatuses[0].code<<","<<h->result.contactStatuses[1].code;
 std::cout<<'\n';
 require(hookPhase==2 && continuation==0 && stepStatus && stepStatus[0]==0 && stepStatus[1]==0 && h->result.contactStatuses[0].code==0 && h->result.contactStatuses[1].code==0,"continuation status prerequisite");
 require(resetWords[0]==1 && resetWords[1]==0,"automatic reset mask prerequisite");
 require(accepted[0].acceptedSteps==1 && accepted[0].lastSensor==0 && accepted[0].captureStep==0 && accepted[0].captureTimeBits==0 && accepted[1].acceptedSteps==6 && accepted[1].lastSensor==2 && accepted[1].captureStep==4 && accepted[1].captureTimeBits==1034147594u,"reset clock selection");
 metalrobo::TrackerProbeSnapshot continued;require(tracker->researchSnapshot(continued),"continuation snapshot");savePartial(argv[3],"continued",continued);
 std::array<std::uint32_t,280> continuedActor{};
 std::memcpy(continuedActor.data(),actorReadback.contents,sizeof(continuedActor));
 unsigned changedDepth=0,changedMetadata=0;
 const auto* acceptedWords=static_cast<const std::uint32_t*>(acceptedMetadata.contents);
 for(unsigned i=0;i<144;++i)changedDepth+=after.frameDepthBits[i]!=continued.frameDepthBits[i];
 for(unsigned i=0;i<16;++i)changedMetadata+=metadataPrior[i]!=acceptedWords[i];
 {std::ofstream partial(std::string(argv[3])+".partial.tsv",std::ios::app);require(bool(partial),"metadata partial output");
  for(unsigned i=0;i<32;++i){partial<<"prior\taccepted_metadata\t"<<i<<'\t'<<metadataPrior[i]<<'\n';partial<<"continued\taccepted_metadata\t"<<i<<'\t'<<acceptedWords[i]<<'\n';}
  for(unsigned i=0;i<280;++i)partial<<"continued\tactor\t"<<i<<'\t'<<continuedActor[i]<<'\n';}
 std::cout<<"challenge reset_depth_changed="<<changedDepth<<" reset_metadata_changed="<<changedMetadata<<'\n';
 require(changedDepth>0 && changedMetadata>0,"fresh-reset challenge prerequisite");
 mr_research_set_observation_hooks(nullptr,nullptr,nullptr);
 auto* rt=h->visualRuntime.get();void* bodies=tracker->researchCapturedBodies();void* laterBodies=tracker->researchCurrentBodies();require(bodies && laterBodies,"missing body buffers");
 id<MTLCommandQueue> q=[researchDevice newCommandQueue];
 id<MTLBuffer> captured=(__bridge id<MTLBuffer>)bodies;id<MTLBuffer> later=(__bridge id<MTLBuffer>)laterBodies;
 const std::size_t bodyBytes=std::size_t(2)*rt->renderer.layout().bodyCount*sizeof(MRBodyStateGPU);
 require(bodyBytes && captured.length>=bodyBytes && later.length>=bodyBytes,"body buffer shape");
 id<MTLBuffer> captureRead=[researchDevice newBufferWithLength:bodyBytes options:MTLResourceStorageModeShared];
 id<MTLBuffer> laterRead=[researchDevice newBufferWithLength:bodyBytes options:MTLResourceStorageModeShared];
 id<MTLCommandBuffer> readcmd=[q commandBuffer];id<MTLBlitCommandEncoder> read=[readcmd blitCommandEncoder];require(captureRead && laterRead && read,"body readback");
 [read copyFromBuffer:captured sourceOffset:0 toBuffer:captureRead destinationOffset:0 size:bodyBytes];
 [read copyFromBuffer:later sourceOffset:0 toBuffer:laterRead destinationOffset:0 size:bodyBytes];
 [read endEncoding];[readcmd commit];[readcmd waitUntilCompleted];require(readcmd.status==MTLCommandBufferStatusCompleted,"body readback command");
 id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];require(enc!=nil,"encoder");
 metalrobo::HybridDeviceStateBatch live{};live.currentBodyStates=bodies;live.previousBodyStates=bodies;live.environmentCount=2;live.bodyCount=rt->renderer.layout().bodyCount;live.frameIndex=0;live.sensorSequence=0;live.source=MR_VISUAL_SOURCE_SIMULATION;live.captureTimestampSeconds=0.0;
 auto rs=rt->renderer.encode(rt->worlds,live,0,(__bridge void*)enc);[enc endEncoding];require(rs.succeeded(),"reference render encoding");[cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,"reference command");
 metalrobo::TrackerProbeSnapshot reference;require(tracker->researchSnapshot(reference),"reference snapshot");
 // A distinct tracker instance, with copied bindings but fresh private history,
 // receives the captured reset-time bodies and the actual reset-mask words.
 // It uses the same renderer/shader, so this is an independent state reference,
 // not an independent implementation of the tracker equations.
 const auto refConfig=tracker->researchCopyConfig();
 metalrobo::MetalHybridObjectTracker referenceTracker;
 auto refStatus=referenceTracker.compile(rt->renderer,rt->worlds,refConfig);
 require(refStatus.succeeded(),"reference tracker compile");
 auto referenceProgram=referenceTracker.observationProgram();require(referenceProgram.valid(),"reference tracker program");
 id<MTLBuffer> refHistory=[researchDevice newBufferWithLength:280*4 options:MTLResourceStorageModeShared];
 id<MTLBuffer> refActor=[researchDevice newBufferWithLength:280*4 options:MTLResourceStorageModeShared];
 require(refHistory && refActor,"reference actor buffers");
 std::memset(refHistory.contents,0,refHistory.length);std::memset(refActor.contents,0,refActor.length);
 id<MTLCommandBuffer> trackCmd=[q commandBuffer];
 metalrobo::MetalWorldDeviceObservationPass pass{};
 pass.commandBuffer=(__bridge void*)trackCmd;pass.currentBodies=bodies;pass.resetMasks=(__bridge void*)resetReadback;
 pass.actorHistory=(__bridge void*)refHistory;pass.actorObservations=(__bridge void*)refActor;
 pass.environmentCount=2;pass.bodyCount=rt->renderer.layout().bodyCount;pass.actorFrameSize=140;
 pass.actorHistoryLength=1;pass.actorObservationSize=140;
 require(referenceProgram.encode(referenceProgram.context,pass),"reference tracker encoding");
 [trackCmd commit];[trackCmd waitUntilCompleted];require(trackCmd.status==MTLCommandBufferStatusCompleted,"reference tracker command");
 metalrobo::TrackerProbeSnapshot refTracks;require(referenceTracker.researchSnapshot(refTracks),"reference tracker snapshot");
 savePartial(argv[3],"reference_tracker",refTracks);
 unsigned trackMismatch=0,actorMismatch=0,slotCount=0;
 require(continued.trackHistory.size()>=48 && refTracks.trackHistory.size()>=48,"tracker reference shape");
 for(unsigned i=0;i<48;++i)trackMismatch+=std::bit_cast<std::uint32_t>(continued.trackHistory[i])!=std::bit_cast<std::uint32_t>(refTracks.trackHistory[i]);
 auto* refActorWords=static_cast<const std::uint32_t*>(refHistory.contents);
 for(const auto& binding:refConfig.bindings)for(unsigned j=0;j<7;++j){auto index=binding.actorFrameOffset+j;require(index<140,"reference slot shape");actorMismatch+=continuedActor[index]!=refActorWords[index];++slotCount;}
 std::cout<<"tracker_reference_mismatch="<<trackMismatch<<" actor_slot_mismatch="<<actorMismatch<<" actor_slots="<<slotCount<<'\n';
 {std::ofstream partial(std::string(argv[3])+".partial.tsv",std::ios::app);require(bool(partial),"actor partial output");
  for(unsigned i=0;i<280;++i)partial<<"reference_tracker\tactor\t"<<i<<'\t'<<refActorWords[i]<<'\n';}
 require(slotCount==42 && !trackMismatch && !actorMismatch,"independent tracker state mismatch");
 std::ofstream file(argv[3]);require(bool(file),"output");file<<"phase\tarray\tindex\tbits\n";
 auto dump=[&](const char* phase,const metalrobo::TrackerProbeSnapshot& x){
  auto emit=[&](const char* name,const auto& v){for(std::size_t i=0;i<v.size();++i)file<<phase<<'\t'<<name<<'\t'<<i<<'\t'<<v[i]<<'\n';};
  emit("depth",x.frameDepthBits);emit("identity",x.frameIdentityBits);emit("validity",x.frameValidityBits);emit("metadata",x.frameMetadataBits);
  for(std::size_t i=0;i<x.cameraState.size();++i)file<<phase<<"\tcamera\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.cameraState[i])<<'\n';
  for(std::size_t i=0;i<x.trackHistory.size();++i)file<<phase<<"\ttrack\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.trackHistory[i])<<'\n';};
 dump("before",before);dump("after",after);dump("continued",continued);dump("reference",reference);dump("reference_tracker",refTracks);
 for(unsigned i=0;i<280;++i){file<<"continued\tactor\t"<<i<<'\t'<<continuedActor[i]<<'\n';file<<"reference_tracker\tactor\t"<<i<<'\t'<<refActorWords[i]<<'\n';}
 const auto* metadataWords=static_cast<const std::uint32_t*>(acceptedMetadata.contents);
 const auto* candidateWords=static_cast<const std::uint32_t*>(candidateMetadata.contents);
 unsigned meta0=0,meta1=0;
 for(unsigned i=0;i<16;++i){meta0+=metadataWords[i]!=reference.frameMetadataBits[i];meta1+=metadataWords[16+i]!=metadataPrior[16+i];
  file<<"prior\tmetadata\t"<<i<<'\t'<<metadataPrior[i]<<'\n';file<<"prior\tmetadata\t"<<16+i<<'\t'<<metadataPrior[16+i]<<'\n';
  file<<"accepted\tmetadata\t"<<i<<'\t'<<metadataWords[i]<<'\n';file<<"accepted\tmetadata\t"<<16+i<<'\t'<<metadataWords[16+i]<<'\n';
  file<<"candidate\tmetadata\t"<<i<<'\t'<<candidateWords[i]<<'\n';file<<"candidate\tmetadata\t"<<16+i<<'\t'<<candidateWords[16+i]<<'\n';}
 std::cout<<"metadata reset_reference_mismatch="<<meta0<<" neighbor_held_mismatch="<<meta1<<'\n';require(!meta0 && !meta1,"metadata selection mismatch");
 const auto* capturedWords=static_cast<const std::uint32_t*>(captureRead.contents);const auto* laterWords=static_cast<const std::uint32_t*>(laterRead.contents);
 const std::size_t wordsPerEnv=bodyBytes/8;
 for(unsigned e=0;e<2;++e){unsigned changed=0;for(std::size_t j=0;j<wordsPerEnv;++j){auto i=e*wordsPerEnv+j;changed+=capturedWords[i]!=laterWords[i];file<<"captured\tbody\t"<<i<<'\t'<<capturedWords[i]<<'\n';file<<"later\tbody\t"<<i<<'\t'<<laterWords[i]<<'\n';}std::cout<<"body env="<<e<<" changed_words="<<changed<<'\n';}
 for(unsigned phase=0;phase<2;++phase)for(unsigned i=0;i<280;++i)file<<(phase?"after":"before")<<"\tactor\t"<<i<<'\t'<<actorBits[phase][i]<<'\n';
 unsigned actorRejected=0,actorAcceptedChanged=0;
 for(unsigned i=0;i<140;++i){actorRejected+=actorBits[0][i]!=actorBits[1][i];actorAcceptedChanged+=actorBits[0][140+i]!=actorBits[1][140+i];}
 std::cout<<"actor rejected_mismatch="<<actorRejected<<" accepted_changed="<<actorAcceptedChanged<<'\n';
 require(actorRejected==0,"rejected actor history changed");
 auto compare=[&](const char* name,const auto& prev,const auto& now,const auto& ref,unsigned stride) {
  require(prev.size()==stride*2 && now.size()==prev.size() && ref.size()==prev.size(),"frame shape");unsigned bad0=0,bad1=0,changed0=0;
  for(unsigned i=0;i<stride;++i){bad0+=now[i]!=ref[i];bad1+=now[stride+i]!=prev[stride+i];changed0+=prev[i]!=now[i];}
  std::cout<<name<<" reset_reference_mismatch="<<bad0<<" neighbor_held_mismatch="<<bad1<<" reset_changed="<<changed0<<'\n';require(!bad0 && !bad1,"frame selection mismatch");};
 compare("depth",after.frameDepthBits,continued.frameDepthBits,reference.frameDepthBits,144);
 compare("identity",after.frameIdentityBits,continued.frameIdentityBits,reference.frameIdentityBits,576);
 compare("validity",after.frameValidityBits,continued.frameValidityBits,reference.frameValidityBits,144);
 auto asBits=[](const auto& v){std::vector<std::uint32_t> b;for(auto x:v)b.push_back(std::bit_cast<std::uint32_t>(x));return b;};
 compare("camera",asBits(after.cameraState),asBits(continued.cameraState),asBits(reference.cameraState),16);
 for(unsigned e=0;e<2;++e){unsigned visible=0;for(unsigned i=e*144;i<(e+1)*144;++i)visible+=before.frameIdentityBits[i*4+1]==105 && before.frameValidityBits[i]!=0;std::cout<<"visible env="<<e<<" pixels="<<visible<<'\n';require(visible>0,"visibility prerequisite");}
 return 0;
}catch(const std::exception& e){std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}}}
