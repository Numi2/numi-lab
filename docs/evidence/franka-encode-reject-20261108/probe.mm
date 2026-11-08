#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "/Users/numi/numi-franka-frame-gate-owner-20261108/src/c_api_retained_research.cpp"
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using Hook=bool (*)(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t);
extern "C" void mr_research_set_observation_hooks(void*,Hook,Hook) noexcept;
// Research-only admission contract: the caller supplies immutable, already
// synchronized capture statuses before submission. Missing means reject the
// entire batch before any visual or physics command is submitted.
static std::array<std::uint32_t,2> frameStatus={1,1};
static int admitted=0,rejected=0,finished=0; static bool rejectFinish=false;
static bool prepare(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t,std::uint32_t,std::uint32_t) {
 if(n!=2) return false;
 for(auto v:frameStatus) if(v!=1u) {++rejected;return false;}
 ++admitted;return true;
}
static bool finish(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t,std::uint32_t,std::uint32_t){if(n!=2)return false;++finished;if(rejectFinish){rejectFinish=false;return false;}return true;}
static void req(bool x,const char* m){if(!x)throw std::runtime_error(m);}
template<class V> static bool exact(const V& a,const V& b) {return a.size()==b.size() && (!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0);}
int main(int argc,char** argv){@autoreleasepool{try{
 req(argc==4,"usage: probe METALLIB VISUAL_PACK OUTPUT");
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
 req(h->visualRuntime && h->environmentCount==2,"visual owner unavailable");
 auto& tracker=h->visualRuntime->tracker;
 std::cout<<"direct_winner_enabled="<<tracker.researchDirectWinnerEnabled()<<'\n';
 req(tracker.researchDirectWinnerEnabled(),"direct winner prerequisite");
 metalrobo::PolicyPack policy;policy.id="fixture_zero_policy";const auto& l=h->taskProgram.layout();
 policy.layers.push_back({.inputCount=l.actorObservationSize,.outputCount=l.actionCount,.activation=metalrobo::PolicyActivation::identity,.weights=std::vector<float>(l.actorObservationSize*l.actionCount,0.f),.bias=std::vector<float>(l.actionCount,0.f)});
 metalrobo::bindPolicyPack(policy,h->taskProgram);installPolicyPack(*h,policy);
 mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepare,finish);
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
 rejectFinish=true; MRTaskRolloutAdvanceC denied{};int b=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&denied);
 std::cout<<"finish_reject rc="<<b<<" admitted="<<admitted<<" finished="<<finished<<"\n";
 req(b!=0 && admitted==2 && finished==2,"finish hook did not reject");
 req(tracker.researchSnapshot(after),"after snapshot");snapshot(after);out.flush();
 bool sameHistory=exact(before.depthHistory,after.depthHistory)&&exact(before.trackHistory,after.trackHistory)&&exact(before.frameDepthBits,after.frameDepthBits)&&exact(before.frameIdentityBits,after.frameIdentityBits)&&exact(before.frameValidityBits,after.frameValidityBits)&&exact(before.cameraState,after.cameraState);
 bool sameMetadata=std::memcmp(before.frameMetadataBits.data(),after.frameMetadataBits.data(),sizeof(before.frameMetadataBits))==0;
 bool sameCadence=before.observationSequence==after.observationSequence && before.lastSensorSequence==after.lastSensorSequence;
 std::cout<<"unchanged_history_and_frames="<<sameHistory<<" unchanged_metadata="<<sameMetadata<<" unchanged_cadence="<<sameCadence<<" sequence="<<before.observationSequence<<","<<after.observationSequence<<" sensor="<<before.lastSensorSequence<<","<<after.lastSensorSequence<<"\n";
 MRTaskRolloutAdvanceC continued{};int c=mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,0,0,&continued);
 s=mr_task_rollout_status_codes(h.get());std::cout<<"retry rc="<<c<<" admitted="<<admitted<<" finished="<<finished;if(s)std::cout<<" world="<<s[0]<<","<<s[1];std::cout<<"\n";
 req(c==0 && s && s[0]==0 && s[1]==0 && admitted==3 && finished==3,"retry failed");
 metalrobo::TrackerProbeSnapshot retried;req(tracker.researchSnapshot(retried),"retry snapshot");snapshot(retried);out.flush();
 std::cout<<"retry_sequence="<<retried.observationSequence<<" retry_sensor="<<retried.lastSensorSequence<<"\n";
 out.write(reinterpret_cast<const char*>(words),bytes);out.flush();
 return 0;
}catch(const std::exception& e){std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}}}

