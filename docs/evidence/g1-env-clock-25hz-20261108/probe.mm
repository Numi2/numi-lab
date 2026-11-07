#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
// Bounded native composition of actual task, tracker, rendering and physics.
// Compares an intended mixed-overflow arm with an all-clear control.
#include "/Users/numi/numi-g1-env-render-ref-owner-20261108/src/c_api_retained_research.cpp"
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
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-clock-25hz-build-20261108/shaders/restore_fresh.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];
  if(!lib) return false;
  restorePipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_selective_restore"] error:&error];
  if(!restorePipeline) return false;
  clockPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_accept_sensor_clock"] error:&error];
  if(!clockPipeline) return false;
 }
 if (envClock && !envClockPipeline) {
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-clock-25hz-build-20261108/shaders/numi-research-cadence-20261108.metallib"];
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
int main(int argc,char** argv) {@autoreleasepool {try {
 require(argc==4,"usage: probe METALLIB ASSETS OUT_TSV");
 auto h=make(argv[1],argv[2],true);tracker=&h->visualRuntime->tracker;researchDevice=MTLCreateSystemDefaultDevice();
 cadenceBuffer=[researchDevice newBufferWithLength:4*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
 envClock=[researchDevice newBufferWithLength:2*sizeof(research_cadence::Record) options:MTLResourceStorageModeShared];
 envCandidate=[researchDevice newBufferWithLength:2*sizeof(research_cadence::Record) options:MTLResourceStorageModeShared];
 require(cadenceBuffer && envClock && envCandidate,"clock buffers");
 auto* guard=static_cast<std::uint32_t*>(cadenceBuffer.contents);guard[0]=0;guard[1]=UINT32_MAX;guard[2]=0;guard[3]=UINT32_MAX;
 auto* accepted=static_cast<research_cadence::Record*>(envClock.contents);auto* candidates=static_cast<research_cadence::Record*>(envCandidate.contents);
 accepted[0]={};accepted[1]={};
 mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepareHook,finishHook);
 metalrobo::TrackerProbeSnapshot before,after;
 std::array<std::array<std::uint32_t,280>,2> actorBits{};
 for(int step=1;step<=3;++step) {
  for(unsigned e=0;e<2;++e){research_cadence::Candidate p{};require(research_cadence::propose(accepted[e],.02f,25.f,0,false,p),"proposal");
   // Existing single-step batch renderer records frame 0 / time 0; do not
   // substitute a not-yet-rendered per-environment timestamp.
   candidates[e]=p.accepted;if(p.sample){candidates[e].captureStep=0;candidates[e].captureTimeBits=0;}
   std::cerr<<"proposal step="<<step<<" env="<<e<<" count="<<accepted[e].acceptedSteps<<" sensor="<<p.sensor<<" sample="<<p.sample<<'\n';}
  hookPhase=0;MRTaskRolloutAdvanceC a{};int rc=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&a);
  require(hookPhase==2 && (step<3?rc==0:rc!=0),"expected status missing");
  auto* st=mr_task_rollout_status_codes(h.get());auto* tr=mr_task_rollout_transitions(h.get());require(st && tr,"missing status");
  if(step<3)require(st[0]==0 && st[1]==0 && !tr[0].done && !tr[1].done,"warmup failed");
  else {std::cerr<<"mixed_status world="<<st[0]<<","<<st[1]<<" contact="<<h->result.contactStatuses[0].code<<","<<h->result.contactStatuses[1].code<<" required0="<<h->result.contactStatuses[0].requiredRawContacts<<'\n';
   require(st[0]!=0 && st[1]==0 && h->result.contactStatuses[0].code!=0 && h->result.contactStatuses[1].code==0,"mixed status");}
  for(unsigned e=0;e<2;++e)std::cerr<<"accepted step="<<step<<" env="<<e<<" words="<<accepted[e].acceptedSteps<<","<<accepted[e].lastSensor<<","<<accepted[e].captureStep<<","<<accepted[e].captureTimeBits<<'\n';
  if(step>=2) {
   require(actorReadback && actorReadback.length>=280*sizeof(std::uint32_t),"actor readback");
   std::memcpy(actorBits[step-2].data(),actorReadback.contents,280*sizeof(std::uint32_t));
  }
  if(step==2)require(tracker->researchSnapshot(before),"before snapshot");
  if(step==3)require(tracker->researchSnapshot(after),"after snapshot");
 }
 require(accepted[0].acceptedSteps==2 && accepted[0].lastSensor==0 && accepted[1].acceptedSteps==3 && accepted[1].lastSensor==1,"clock result");
 for(unsigned e=0;e<2;++e){research_cadence::Candidate p{};require(research_cadence::propose(accepted[e],.02f,25.f,0,false,p),"next proposal");
  std::cout<<"next env="<<e<<" count="<<accepted[e].acceptedSteps<<" sensor="<<p.sensor<<" sample="<<p.sample<<'\n';
  require(p.sensor==1 && p.sample==(e==0),"expected differing phases");}
 mr_research_set_observation_hooks(nullptr,nullptr,nullptr);
 auto* rt=h->visualRuntime.get();void* bodies=tracker->researchCurrentBodies();require(bodies,"missing bodies");
 id<MTLCommandQueue> q=[researchDevice newCommandQueue];id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];require(enc!=nil,"encoder");
 metalrobo::HybridDeviceStateBatch live{};live.currentBodyStates=bodies;live.previousBodyStates=bodies;live.environmentCount=2;live.bodyCount=rt->renderer.layout().bodyCount;live.frameIndex=0;live.sensorSequence=1;live.source=MR_VISUAL_SOURCE_SIMULATION;
 auto rs=rt->renderer.encode(rt->worlds,live,0,(__bridge void*)enc);[enc endEncoding];require(rs.succeeded(),"reference render encoding");[cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,"reference command");
 metalrobo::TrackerProbeSnapshot reference;require(tracker->researchSnapshot(reference),"reference snapshot");
 std::ofstream file(argv[3]);require(bool(file),"output");file<<"phase\tarray\tindex\tbits\n";
 auto dump=[&](const char* phase,const metalrobo::TrackerProbeSnapshot& x){
  auto emit=[&](const char* name,const auto& v){for(std::size_t i=0;i<v.size();++i)file<<phase<<'\t'<<name<<'\t'<<i<<'\t'<<v[i]<<'\n';};
  emit("depth",x.frameDepthBits);emit("identity",x.frameIdentityBits);emit("validity",x.frameValidityBits);emit("metadata",x.frameMetadataBits);
  for(std::size_t i=0;i<x.cameraState.size();++i)file<<phase<<"\tcamera\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.cameraState[i])<<'\n';
  for(std::size_t i=0;i<x.trackHistory.size();++i)file<<phase<<"\ttrack\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.trackHistory[i])<<'\n';};
 dump("before",before);dump("after",after);dump("reference",reference);
 for(unsigned phase=0;phase<2;++phase)for(unsigned i=0;i<280;++i)file<<(phase?"after":"before")<<"\tactor\t"<<i<<'\t'<<actorBits[phase][i]<<'\n';
 unsigned actorRejected=0,actorAcceptedChanged=0;
 for(unsigned i=0;i<140;++i){actorRejected+=actorBits[0][i]!=actorBits[1][i];actorAcceptedChanged+=actorBits[0][140+i]!=actorBits[1][140+i];}
 std::cout<<"actor rejected_mismatch="<<actorRejected<<" accepted_changed="<<actorAcceptedChanged<<'\n';
 require(actorRejected==0,"rejected actor history changed");
 auto compare=[&](const char* name,const auto& prev,const auto& now,const auto& ref,unsigned stride) {
  require(prev.size()==stride*2 && now.size()==prev.size() && ref.size()==prev.size(),"frame shape");unsigned bad0=0,bad1=0,changed1=0;
  for(unsigned i=0;i<stride;++i){bad0+=prev[i]!=now[i];bad1+=now[stride+i]!=ref[stride+i];changed1+=prev[stride+i]!=now[stride+i];}
  std::cout<<name<<" rejected_mismatch="<<bad0<<" accepted_reference_mismatch="<<bad1<<" accepted_changed="<<changed1<<'\n';require(!bad0 && !bad1,"frame selection mismatch");};
 compare("depth",before.frameDepthBits,after.frameDepthBits,reference.frameDepthBits,144);
 compare("identity",before.frameIdentityBits,after.frameIdentityBits,reference.frameIdentityBits,576);
 compare("validity",before.frameValidityBits,after.frameValidityBits,reference.frameValidityBits,144);
 auto asBits=[](const auto& v){std::vector<std::uint32_t> b;for(auto x:v)b.push_back(std::bit_cast<std::uint32_t>(x));return b;};
 compare("camera",asBits(before.cameraState),asBits(after.cameraState),asBits(reference.cameraState),16);
 for(unsigned e=0;e<2;++e){unsigned visible=0;for(unsigned i=e*144;i<(e+1)*144;++i)visible+=before.frameIdentityBits[i*4+1]==105 && before.frameValidityBits[i]!=0;std::cout<<"visible env="<<e<<" pixels="<<visible<<'\n';require(visible>0,"visibility prerequisite");}
 return 0;
}catch(const std::exception& e){std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}}}
