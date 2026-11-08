#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "/Users/numi/numi-franka-presubmit-owner-20261108/src/c_api_retained_research.cpp"
#include "CandidatePublication.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Hook=bool (*)(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t);
extern "C" void mr_research_set_observation_hooks(void*,Hook,Hook,void(*)(void*)) noexcept;
static metalrobo::MetalHybridObjectTracker* tracker=nullptr;
static metalrobo::TrackerHostCheckpoint checkpoint{};
static bool failFinish=false;static unsigned aborts=0;
static bool prepare(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t,std::uint32_t,std::uint32_t) {
 return n==2 && tracker && tracker->researchCheckpointHost(checkpoint);
}
static id<MTLBuffer> actorCapture=nil;
static bool finish(void*,void* cb,void* actor,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t frame,std::uint32_t history,std::uint32_t) {
 if(n!=2)return false;
 id<MTLBuffer> source=(__bridge id<MTLBuffer>)actor; id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)cb;
 size_t bytes=size_t(n)*frame*history*sizeof(float);
 if(!source || source.length<bytes || !cmd || !bytes)return false;
 actorCapture=[source.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
 id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder]; if(!actorCapture || !blit)return false;
 [blit copyFromBuffer:source sourceOffset:0 toBuffer:actorCapture destinationOffset:0 size:bytes]; [blit endEncoding];return true;
}
static void abortGuard(void*) {tracker->researchRestoreHost(checkpoint);++aborts;}
void req(bool b,const char* message){if(!b)throw std::runtime_error(message);}
struct Outcome {metalrobo::MetalWorldDiagnostics diagnostic; metalrobo::MetalWorldResult result;};
Outcome step(MRTaskRolloutHandle& h) {
 const bool init=!h.residentState.valid();
 const std::array<std::uint32_t,2> reset{0,0};
 metalrobo::MetalWorldBatch b{.environmentCount=h.environmentCount,.controlStepCount=1,
  .initialQ=init?std::span<const float>{h.resetQ}:std::span<const float>{},
  .initialV=init?std::span<const float>{h.resetV}:std::span<const float>{},
  .actions={},.policyRevision=0,.resetMasks=reset,
  .initialSceneBodies=init?std::span<const MRBodyStateGPU>{h.resetSceneBodies}:std::span<const MRBodyStateGPU>{}};
 metalrobo::MetalWorldSubmission submit;Outcome out;
 out.diagnostic=init?h.context.initializeResidentState(h.world,b,h.stepConfig,h.residentState,submit):
                     h.context.submitResident(h.world,b,h.stepConfig,h.residentState,submit);
 if(out.diagnostic.succeeded())out.diagnostic=submit.wait(out.result);
 return out;
}
int main(int argc,char**argv){try{
 req(argc==5,"usage: probe metallib asset output control|mixed");
 const bool mixed=std::string(argv[4])=="mixed";req(mixed||std::string(argv[4])=="control","arm");
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=20261108;
 auto manifest=makeFrankaPickPlaceRunManifest(cfg);manifest.task.capacities.rawContacts=1;
 std::erase_if(manifest.reality.reset.operators,[](const auto& x){return x.operation==metalrobo::TaskRandomizationOperator::sceneBodyPosition||x.operation==metalrobo::TaskRandomizationOperator::sceneBodyVelocity;});
 auto& obs=manifest.sensors.observation;obs.visual.width=16;obs.visual.height=9;obs.visual.frameOffsets={0};
 for(unsigned i=0;i<144;++i)obs.actorFrame.push_back({.source=metalrobo::TaskObservationSource::maskedDepth,.component=i});
 obs.critic.push_back({.source=metalrobo::TaskObservationSource::objectTrack,.target="pick_object",.component=0});
 MRTaskVisualPackC pack{argv[2],"pick_object",2u,105u};MRTaskVisualObservationConfigC visual{};
 visual.packs=&pack;visual.pack_count=1;visual.camera_parent_body="panda_hand";visual.camera_position[2]=20.f;visual.camera_orientation[3]=1.f;visual.width=16;visual.height=9;visual.minimum_visible_pixels=1;visual.vertical_field_of_view_degrees=54;visual.nominal_rate_hz=50;
 auto h=createCompiledRunTaskRollout(std::move(manifest),argv[1],"franka staged publication",&visual);
 req(h->visualRuntime && h->environmentCount==2,"visual owner missing");const auto n=h->defaultSceneBodies.size();req(n==4 && h->resetSceneBodies.size()==2*n,"scene shape");
 for(unsigned e=0;e<2;++e){h->resetSceneBodies[e*n].position={3.f,0.f,3.f,0.f};h->resetSceneBodies[e*n+3].position={4.f,0.f,3.f,0.f};}
 if(mixed){
 h->resetSceneBodies[0].position={3.f,0.f,.070f,0.f};h->resetSceneBodies[3].position={4.f,0.f,.075f,0.f};
 h->resetSceneBodies[0].linearVelocityAndInverseMass.z=-1.f;h->resetSceneBodies[3].linearVelocityAndInverseMass.z=-1.f;
 }
 h->stepConfig.publishFinalState=true; const auto& l=h->taskProgram.layout();metalrobo::PolicyPack p;p.id="stage_zero_policy";
 p.layers.push_back({.inputCount=l.actorObservationSize,.outputCount=l.actionCount,.activation=metalrobo::PolicyActivation::identity,.weights=std::vector<float>(l.actorObservationSize*l.actionCount,0.f),.bias=std::vector<float>(l.actionCount,0.f)});
 metalrobo::bindPolicyPack(p,h->taskProgram);installPolicyPack(*h,p);tracker=&h->visualRuntime->tracker;req(tracker->researchDirectWinnerEnabled(),"direct winner prerequisite");
 mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepare,finish,abortGuard);
 std::ofstream output(argv[3],std::ios::binary);req(bool(output),"output");
 auto write=[&](const auto& v){std::uint64_t count=v.size();output.write((const char*)&count,8);if(count)output.write((const char*)v.data(),count*sizeof(v[0]));};
 auto snap=[&](const char* name){metalrobo::TrackerProbeSnapshot x;req(tracker->researchSnapshot(x),"tracker readback");req(actorCapture!=nil,"actor readback");
 std::cout<<"snapshot="<<name<<" depth_history="<<x.depthHistory.size()<<" track_history="<<x.trackHistory.size()<<" frame_depth="<<x.frameDepthBits.size()<<" frame_id="<<x.frameIdentityBits.size()<<" frame_valid="<<x.frameValidityBits.size()<<" camera="<<x.cameraState.size()<<" actor_history_bytes="<<actorCapture.length<<" sequence="<<x.observationSequence<<" sensor="<<x.lastSensorSequence<<" bindings="<<x.bindingIds.size()<<"\n";
 write(x.depthHistory);write(x.trackHistory);write(x.frameDepthBits);write(x.frameIdentityBits);write(x.frameValidityBits);write(x.cameraState);write(x.frameMetadataBits);std::vector<std::uint64_t> cadence{x.observationSequence,x.lastSensorSequence};write(cadence);
 std::vector<unsigned char> actor((unsigned char*)actorCapture.contents,(unsigned char*)actorCapture.contents+actorCapture.length);write(actor);output.flush();};
 auto log=[&](const char* name,const Outcome& x){std::cout<<name<<" status="<<(unsigned)x.diagnostic.status<<" published="<<x.diagnostic.published<<" success="<<x.diagnostic.successfulStepCount<<" failed="<<x.diagnostic.failedStepCount<<"\n";for(const auto& y:x.result.statuses)std::cout<<name<<" world env="<<y.environment<<" code="<<y.code<<"\n";for(const auto& y:x.result.contactStatuses)std::cout<<name<<" contact env="<<y.environment<<" code="<<y.code<<" raw_required="<<y.requiredRawContacts<<"\n";};
 auto scene=[&](const char* name,const auto& v){std::cout<<name<<" scene_count="<<v.size()<<"\n"; for(size_t i=0;i<v.size();++i){const auto& x=v[i];std::cout<<name<<" scene="<<i<<" xyz="<<std::hexfloat<<x.position.x<<","<<x.position.y<<","<<x.position.z<<" velocity="<<x.linearVelocityAndInverseMass.x<<","<<x.linearVelocityAndInverseMass.y<<","<<x.linearVelocityAndInverseMass.z<<std::defaultfloat<<" motion="<<x.flagsAndIndices[0]<<"\n";}};
 scene("input",h->resetSceneBodies);
 auto warm=step(*h);log("warmup",warm);scene("warmup",warm.result.finalSceneBodies);req(warm.diagnostic.published && warm.result.statuses.size()==2 && warm.result.contactStatuses.size()==2 && warm.result.statuses[0].code==0 && warm.result.statuses[1].code==0 && warm.result.contactStatuses[0].code==0 && warm.result.contactStatuses[1].code==0,"accepted warmup prerequisite");snap("warmup");
 auto next=step(*h);log("test",next);scene("test",next.result.finalSceneBodies);req(next.diagnostic.published && next.result.statuses.size()==2 && next.result.contactStatuses.size()==2,"validated test prerequisite");snap("test");
 bool expected=mixed ? (next.result.statuses[0].code!=0 && next.result.statuses[1].code==0 && next.result.contactStatuses[0].code!=0 && next.result.contactStatuses[1].code==0) : (next.result.statuses[0].code==0 && next.result.statuses[1].code==0 && next.result.contactStatuses[0].code==0 && next.result.contactStatuses[1].code==0);
 req(expected,"mixed or control contact prerequisite");
 if(mixed){auto retry=step(*h);log("continuation",retry);if(retry.diagnostic.published)snap("continuation");}
 return 0;
}catch(const std::exception& e){std::cerr<<"error="<<e.what()<<"\n";return 2;}}
