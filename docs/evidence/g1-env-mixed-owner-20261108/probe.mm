#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
// Bounded native composition of actual task, tracker, rendering and physics.
// Compares an intended mixed-overflow arm with an all-clear control.
#include "/Users/numi/numi-g1-env-mixed-owner-20261108/src/c_api_retained_research.cpp"
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
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-mixed-build-20261108/shaders/restore_fresh.metallib"];
  id<MTLLibrary> lib=[researchDevice newLibraryWithURL:url error:&error];
  if(!lib) return false;
  restorePipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_selective_restore"] error:&error];
  if(!restorePipeline) return false;
  clockPipeline=[researchDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_accept_sensor_clock"] error:&error];
  if(!clockPipeline) return false;
 }
 if (envClock && !envClockPipeline) {
  NSError* error=nil;
  NSURL* url=[NSURL fileURLWithPath:@"/Users/numi/numi-g1-env-mixed-build-20261108/shaders/numi-research-cadence-20261108.metallib"];
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
int main(int argc,char** argv) {
 @autoreleasepool {try {
  require(argc==4,"usage: probe METALLIB ASSETS OUT_TSV");
  auto h=make(argv[1],argv[2],false); tracker=&h->visualRuntime->tracker;
  researchDevice=MTLCreateSystemDefaultDevice();
  cadenceBuffer=[researchDevice newBufferWithLength:4*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
  require(cadenceBuffer!=nil,"cadence buffer");auto* clocks=static_cast<std::uint32_t*>(cadenceBuffer.contents);
  clocks[0]=0;clocks[1]=UINT32_MAX;clocks[2]=0;clocks[3]=UINT32_MAX;
  mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepareHook,finishHook);
  for(int step=1;step<=2;++step) {
   hookPhase=0;MRTaskRolloutAdvanceC a{};int rc=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&a);
   require(rc==0 && hookPhase==2,"warmup step failed");
   auto* status=mr_task_rollout_status_codes(h.get());auto* trans=mr_task_rollout_transitions(h.get());
   require(status && trans && status[0]==0 && status[1]==0 && !trans[0].done && !trans[1].done,"warmup not accepted nonterminal");
  }
  mr_research_set_observation_hooks(nullptr,nullptr,nullptr);
  auto* rt=h->visualRuntime.get();void* bodies=tracker->researchCurrentBodies();require(bodies!=nullptr,"missing retained observation bodies");
  id<MTLCommandQueue> q=[researchDevice newCommandQueue];require(q!=nil,"queue");
  auto render=[&](std::uint64_t frame,std::uint32_t sensor) {
   metalrobo::HybridDeviceStateBatch live{};live.currentBodyStates=bodies;live.previousBodyStates=bodies;
   live.environmentCount=2;live.bodyCount=rt->renderer.layout().bodyCount;live.frameIndex=frame;live.sensorSequence=sensor;
   live.source=MR_VISUAL_SOURCE_SIMULATION;live.captureTimestampSeconds=double(frame)*double(.02f);live.frameAgeSeconds=0;
   id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];require(enc!=nil,"encoder");
   auto status=rt->renderer.encode(rt->worlds,live,0,(__bridge void*)enc);[enc endEncoding];require(status.succeeded(),"renderer encode");
   [cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,"render command failed");
   metalrobo::TrackerProbeSnapshot shot;require(tracker->researchSnapshot(shot),"snapshot");
   std::cerr<<"render frame="<<frame<<" sensor="<<sensor<<" meta_frame="<<shot.frameMetadataBits[4]<<" meta_sensor="<<shot.frameMetadataBits[6]<<" time_bits="<<shot.frameMetadataBits[8]<<'\n';
   return shot;
  };
  auto held=render(2,1);auto fresh=render(0,0);auto heldRef=render(2,1);auto freshRef=render(0,0);
  auto exact=[](const auto& a,const auto& b){return a.frameDepthBits==b.frameDepthBits && a.frameIdentityBits==b.frameIdentityBits && a.frameValidityBits==b.frameValidityBits && a.cameraState==b.cameraState && a.frameMetadataBits==b.frameMetadataBits;};
  require(exact(held,heldRef) && exact(fresh,freshRef),"reference renders not deterministic");
  std::ofstream file(argv[3]);require(bool(file),"output path");file<<"frame\tarray\tindex\tbits\n";
  auto dump=[&](const char* label,const metalrobo::TrackerProbeSnapshot& snap) {
   auto emit=[&](const char* name,const auto& v){for(std::size_t i=0;i<v.size();++i)file<<label<<'\t'<<name<<'\t'<<i<<'\t'<<v[i]<<'\n';};
   emit("depth",snap.frameDepthBits);emit("identity",snap.frameIdentityBits);emit("validity",snap.frameValidityBits);emit("metadata",snap.frameMetadataBits);
  };
  dump("held",held);dump("fresh",fresh);dump("heldReference",heldRef);dump("freshReference",freshRef);
  for(unsigned env=0;env<2;++env) {
   unsigned visible=0,changed=0;
   for(unsigned pixel=env*144;pixel<(env+1)*144;++pixel) {
    if(held.frameIdentityBits[pixel*4+1]==105 && fresh.frameIdentityBits[pixel*4+1]==105 && held.frameValidityBits[pixel] && fresh.frameValidityBits[pixel]) {
     ++visible;changed+=held.frameDepthBits[pixel]!=fresh.frameDepthBits[pixel];
    }
   }
   std::cout<<"env="<<env<<" common_visible_target_pixels="<<visible<<" sequence_sensitive_depth_pixels="<<changed<<'\n';
   require(visible>0 && changed>0,"visible or noise-seed sensitivity prerequisite");
  }
  std::cout<<"separate_reference_renders=bitwise_equal\n";
  MRVisualFrameMetadataGPU heldMeta{};std::memcpy(&heldMeta,held.frameMetadataBits.data(),sizeof(heldMeta));
  MRVisualFrameMetadataGPU freshMeta{};std::memcpy(&freshMeta,fresh.frameMetadataBits.data(),sizeof(freshMeta));
  const auto bitCamera=[](const auto& v){std::vector<std::uint32_t> b;for(float x:v)b.push_back(std::bit_cast<std::uint32_t>(x));return b;};
  const auto compare=[&](const char* field,const auto& got,const auto& a,const auto& b,std::uint32_t stride,int scenario) {
   unsigned mismatch=0;for(unsigned e=0;e<2;++e){const bool takeFresh=(scenario==0?e==0:e==1);const auto& ref=takeFresh?a:b;
    for(unsigned i=0;i<stride;++i){const auto k=e*stride+i;mismatch+=got[k]!=ref[k];file<<"mixed"<<scenario<<'\t'<<field<<'\t'<<k<<'\t'<<got[k]<<'\n';}}
   std::cout<<"scenario="<<scenario<<" field="<<field<<" mismatches="<<mismatch<<'\n';require(mismatch==0,"mixed renderer mismatch");
  };
  for(int scenario=0;scenario<2;++scenario) {
   // Re-establish held frame in both environments before each candidate.
   auto initial=render(2,1);require(exact(initial,held),"held reference changed");
   research_cadence::Record clocks[2];
   for(int e=0;e<2;++e) clocks[e]=(scenario==0?e==0:e==1)
     ? research_cadence::Record{} : research_cadence::Record{3,1,2,1025758986u};
   MRVisualFrameMetadataGPU acceptedMetadata[2]={heldMeta,heldMeta},candidateMetadata[2]{};
   id<MTLCommandBuffer> mixedCommand=[q commandBuffer];require(mixedCommand!=nil,"mixed command");
   require(tracker->researchEncodeMixedFrames((__bridge void*)mixedCommand,bodies,2,rt->renderer.layout().bodyCount,clocks,acceptedMetadata,candidateMetadata),"mixed owner encoding");
   [mixedCommand commit];[mixedCommand waitUntilCompleted];require(mixedCommand.status==MTLCommandBufferStatusCompleted,"mixed owner command");
   metalrobo::TrackerProbeSnapshot result;require(tracker->researchSnapshot(result),"mixed snapshot");
   compare("depth",result.frameDepthBits,fresh.frameDepthBits,held.frameDepthBits,144,scenario);
   compare("identity",result.frameIdentityBits,fresh.frameIdentityBits,held.frameIdentityBits,576,scenario);
   compare("validity",result.frameValidityBits,fresh.frameValidityBits,held.frameValidityBits,144,scenario);
   compare("camera",bitCamera(result.cameraState),bitCamera(fresh.cameraState),bitCamera(held.cameraState),16,scenario);
   for(unsigned e=0;e<2;++e){bool takeFresh=(scenario==0?e==0:e==1);const auto& ref=takeFresh?freshMeta:heldMeta;
    require(std::memcmp(&candidateMetadata[e],&ref,sizeof(ref))==0,"per-environment metadata mismatch");
    const auto* bytes=reinterpret_cast<const std::uint32_t*>(&candidateMetadata[e]);for(unsigned i=0;i<16;++i)file<<"mixed"<<scenario<<"\tmetadata\t"<<e*16+i<<'\t'<<bytes[i]<<'\n';
   }
   std::cout<<"scenario="<<scenario<<" metadata=exact\n";
  }
  return 0;
 }catch(const std::exception& e){std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}}
}
