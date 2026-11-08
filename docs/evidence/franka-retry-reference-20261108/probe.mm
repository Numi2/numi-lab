#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "/Users/numi/numi-franka-presubmit-owner-20261108/src/c_api_retained_research.cpp"
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using Hook=bool (*)(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t);
extern "C" void mr_research_set_observation_hooks(void*,Hook,Hook,void(*)(void*)) noexcept;
// Research-only admission contract: the caller supplies immutable, already
// synchronized capture statuses before submission. Missing means reject the
// entire batch before any visual or physics command is submitted.
static std::array<std::uint32_t,2> frameStatus={1,1};
static metalrobo::MetalHybridObjectTracker* guardedTracker=nullptr; static metalrobo::TrackerHostCheckpoint hostCheckpoint{}; static int aborted=0;
static void abortGuard(void*) { guardedTracker->researchRestoreHost(hostCheckpoint); ++aborted; }
static int admitted=0,rejected=0,finished=0; static bool rejectFinish=false;
static bool prepare(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t,std::uint32_t,std::uint32_t) {
 if(n!=2) return false;
 for(auto v:frameStatus) if(v!=1u) {++rejected;return false;}
 if(!guardedTracker || !guardedTracker->researchCheckpointHost(hostCheckpoint)) return false; ++admitted;return true;
}
static __strong id<MTLBuffer> actorCapture=nil;
static bool finish(void*,void* rawcmd,void* rawhist,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t,std::uint32_t,std::uint32_t obs){
 if(n!=2)return false;++finished;if(rejectFinish){rejectFinish=false;return false;}
 id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)rawcmd;id<MTLBuffer> hist=(__bridge id<MTLBuffer>)rawhist;
 NSUInteger bytes=n*obs*sizeof(float);if(!hist || hist.length<bytes)return false;
 if(!actorCapture)actorCapture=[cmd.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
 id<MTLBlitCommandEncoder> b=[cmd blitCommandEncoder];if(!actorCapture||!b)return false;
 [b copyFromBuffer:hist sourceOffset:0 toBuffer:actorCapture destinationOffset:0 size:bytes];[b endEncoding];return true;
}
static void req(bool x,const char* m){if(!x)throw std::runtime_error(m);}
template<class V> static bool exact(const V& a,const V& b) {return a.size()==b.size() && (!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0);}
int main(int argc,char** argv){@autoreleasepool{try{
 req(argc==5,"usage: probe METALLIB VISUAL_PACK OUTPUT_PREFIX control|retry"); const bool trial=std::string(argv[4])=="retry"; req(trial || std::string(argv[4])=="control","unknown arm");
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=20261108;
 auto manifest=makeFrankaPickPlaceRunManifest(cfg);
 auto& obs=manifest.sensors.observation;
 obs.visual.width=16;obs.visual.height=9;obs.visual.frameOffsets={0};
 for(unsigned i=0;i<144;++i)obs.actorFrame.push_back({.source=metalrobo::TaskObservationSource::maskedDepth,.component=i});
 obs.critic.push_back({.source=metalrobo::TaskObservationSource::objectTrack,.target="pick_object",.component=0});
 MRTaskVisualPackC pack{argv[2],"pick_object",2u,105u};
 MRTaskVisualObservationConfigC visual{};visual.packs=&pack;visual.pack_count=1;visual.camera_parent_body="panda_hand";visual.camera_position[2]=20.f;visual.camera_orientation[3]=1.f;visual.width=16;visual.height=9;visual.minimum_visible_pixels=1;visual.vertical_field_of_view_degrees=54.f;visual.nominal_rate_hz=50.f;
 std::ofstream out(argv[3],std::ios::binary);req(bool(out),"output");
 auto h=createCompiledRunTaskRollout(std::move(manifest),argv[1],"franka explicit frame status fixture",&visual);
 req(h->visualRuntime && h->environmentCount==2,"visual owner unavailable"); req(mr_task_rollout_set_state_readback(h.get(),1)==0,"state readback");
 auto& tracker=h->visualRuntime->tracker;
 std::cout<<"direct_winner_enabled="<<tracker.researchDirectWinnerEnabled()<<'\n';
 req(tracker.researchDirectWinnerEnabled(),"direct winner prerequisite");
 metalrobo::PolicyPack policy;policy.id="fixture_zero_policy";const auto& l=h->taskProgram.layout();
 policy.layers.push_back({.inputCount=l.actorObservationSize,.outputCount=l.actionCount,.activation=metalrobo::PolicyActivation::identity,.weights=std::vector<float>(l.actorObservationSize*l.actionCount,0.f),.bias=std::vector<float>(l.actionCount,0.f)});
 metalrobo::bindPolicyPack(policy,h->taskProgram);installPolicyPack(*h,policy);
 guardedTracker=&tracker; mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepare,finish,abortGuard);
 MRTaskRolloutAdvanceC first{};int a=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&first);
 auto* s=mr_task_rollout_status_codes(h.get());std::cout<<"present_empty rc="<<a<<" admitted="<<admitted<<" finished="<<finished;if(s)std::cout<<" world="<<s[0]<<","<<s[1];std::cout<<'\n';
 req(a==0 && s && s[0]==0 && s[1]==0 && admitted==1 && finished==1,"present step failed");
 metalrobo::TrackerProbeSnapshot before,after;
 req(tracker.researchSnapshot(before),"before snapshot");
 auto save=[&](const auto& v){std::uint64_t n=v.size();out.write(reinterpret_cast<const char*>(&n),sizeof(n));if(n)out.write(reinterpret_cast<const char*>(v.data()),n*sizeof(v[0]));};
 auto snapshot=[&](const metalrobo::TrackerProbeSnapshot& x){save(x.depthHistory);save(x.trackHistory);save(x.frameDepthBits);save(x.frameIdentityBits);save(x.frameValidityBits);save(x.cameraState);out.write(reinterpret_cast<const char*>(x.frameMetadataBits.data()),sizeof(x.frameMetadataBits));out.write(reinterpret_cast<const char*>(&x.observationSequence),sizeof(x.observationSequence));out.write(reinterpret_cast<const char*>(&x.lastSensorSequence),sizeof(x.lastSensorSequence));};
 snapshot(before);out.flush();
 id<MTLBuffer> winners=(__bridge id<MTLBuffer>)h->visualRuntime->renderer.nativeBuffer(metalrobo::MetalHybridRendererBuffer::meshWinners);
 const size_t bytes=2*144*sizeof(std::uint64_t);req(winners && winners.length>=bytes,"winner buffer unavailable");
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();id<MTLBuffer> read=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
 id<MTLCommandQueue> q=[dev newCommandQueue];id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder];req(read && blit,"winner readback");[blit copyFromBuffer:winners sourceOffset:0 toBuffer:read destinationOffset:0 size:bytes];[blit endEncoding];[cmd commit];[cmd waitUntilCompleted];req(cmd.status==MTLCommandBufferStatusCompleted,"winner command");
 auto* words=static_cast<const std::uint64_t*>(read.contents);unsigned nonempty=0;for(unsigned i=0;i<288;++i)nonempty+=static_cast<std::uint32_t>(words[i])!=UINT32_MAX;
 std::cout<<"explicit_present=1,1 nonempty_winners="<<nonempty<<'\n';req(nonempty==0,"empty capture prerequisite");
 if(trial) {
 rejectFinish=true; MRTaskRolloutAdvanceC denied{};int b=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&denied);
 std::cout<<"finish_reject rc="<<b<<" admitted="<<admitted<<" finished="<<finished<<"\n";
 req(b!=0 && admitted==2 && finished==2 && aborted==1,"finish hook did not reject and abort");
 req(tracker.researchSnapshot(after),"after snapshot");snapshot(after);out.flush();
 bool sameHistory=exact(before.depthHistory,after.depthHistory)&&exact(before.trackHistory,after.trackHistory)&&exact(before.frameDepthBits,after.frameDepthBits)&&exact(before.frameIdentityBits,after.frameIdentityBits)&&exact(before.frameValidityBits,after.frameValidityBits)&&exact(before.cameraState,after.cameraState);
 bool sameMetadata=std::memcmp(before.frameMetadataBits.data(),after.frameMetadataBits.data(),sizeof(before.frameMetadataBits))==0;
 bool sameCadence=before.observationSequence==after.observationSequence && before.lastSensorSequence==after.lastSensorSequence;
 std::cout<<"unchanged_history_and_frames="<<sameHistory<<" unchanged_metadata="<<sameMetadata<<" unchanged_cadence="<<sameCadence<<" sequence="<<before.observationSequence<<","<<after.observationSequence<<" sensor="<<before.lastSensorSequence<<","<<after.lastSensorSequence<<"\n";
 req(sameHistory && sameMetadata && sameCadence,"pre-submit rejection changed retained state");
 }
 MRTaskRolloutAdvanceC continued{};int c=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&continued);
 s=mr_task_rollout_status_codes(h.get());std::cout<<"retry rc="<<c<<" admitted="<<admitted<<" finished="<<finished;if(s)std::cout<<" world="<<s[0]<<","<<s[1];std::cout<<"\n";
 req(c==0 && s && s[0]==0 && s[1]==0 && admitted==(trial?3:2) && finished==(trial?3:2),"final step failed");
 metalrobo::TrackerProbeSnapshot retried;req(tracker.researchSnapshot(retried),"retry snapshot");snapshot(retried);out.flush();
 std::cout<<"retry_sequence="<<retried.observationSequence<<" retry_sensor="<<retried.lastSensorSequence<<" aborted="<<aborted<<"\n";
 req(retried.observationSequence==2 && retried.lastSensorSequence==1 && aborted==(trial?1:0),"final cadence or acceptance wrong");
 out.write(reinterpret_cast<const char*>(words),bytes);out.flush();
 auto writeField=[&](const std::string& name,const void* ptr,size_t count,size_t size){
   req(ptr && count>0,"empty comparison field");std::ofstream f(std::string(argv[3])+"."+name,std::ios::binary);req(bool(f),"field output");f.write((const char*)ptr,count*size);req(bool(f),"field write");
 };
 auto vectorField=[&](const std::string& name,const auto& v){writeField(name,v.data(),v.size(),sizeof(v[0]));};
 vectorField("q",h->result.finalQ);vectorField("v",h->result.finalV);vectorField("actor",h->result.actorObservations);
 vectorField("depth_history",retried.depthHistory);vectorField("track_history",retried.trackHistory);vectorField("camera",retried.cameraState);
 writeField("metadata",retried.frameMetadataBits.data(),retried.frameMetadataBits.size(),sizeof(std::uint32_t));
 writeField("actor_history",actorCapture.contents,actorCapture.length,1);
 std::vector<float> scene; for(const auto& body:h->result.finalSceneBodies) {
  for(float x:{body.position.x,body.position.y,body.position.z,body.orientation.x,body.orientation.y,body.orientation.z,body.orientation.w,body.linearVelocityAndInverseMass.x,body.linearVelocityAndInverseMass.y,body.linearVelocityAndInverseMass.z,body.angularVelocity.x,body.angularVelocity.y,body.angularVelocity.z})scene.push_back(x);
 }
 vectorField("scene",scene);writeField("status",s,2,sizeof(std::uint32_t));
 id<MTLCommandBuffer> finalcmd=[q commandBuffer];id<MTLBlitCommandEncoder> finalblit=[finalcmd blitCommandEncoder];req(finalblit,"final winner blit");[finalblit copyFromBuffer:winners sourceOffset:0 toBuffer:read destinationOffset:0 size:bytes];[finalblit endEncoding];[finalcmd commit];[finalcmd waitUntilCompleted];req(finalcmd.status==MTLCommandBufferStatusCompleted,"final winner readback");
 writeField("winners",read.contents,bytes,1);
 std::cout<<"comparison_fields=11 actor_history_bytes="<<actorCapture.length<<" scene_values="<<scene.size()<<"\n";

 return 0;
}catch(const std::exception& e){std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}}}

