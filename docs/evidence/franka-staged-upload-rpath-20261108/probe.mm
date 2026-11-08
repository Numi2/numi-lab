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
static bool finish(void*,void*,void*,void*,void*,void*,void*,void*,std::uint32_t n,std::uint32_t,std::uint32_t,std::uint32_t) {
 if(n!=2)return false;if(failFinish){failFinish=false;return false;}return true;
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
 if(mixed){h->resetSceneBodies[0].position={.45f,0.f,.01f,0.f};h->resetSceneBodies[3].position={.45f,0.f,.01f,0.f};}
 const auto& l=h->taskProgram.layout();metalrobo::PolicyPack p;p.id="stage_zero_policy";
 p.layers.push_back({.inputCount=l.actorObservationSize,.outputCount=l.actionCount,.activation=metalrobo::PolicyActivation::identity,.weights=std::vector<float>(l.actorObservationSize*l.actionCount,0.f),.bias=std::vector<float>(l.actionCount,0.f)});
 metalrobo::bindPolicyPack(p,h->taskProgram);installPolicyPack(*h,p);tracker=&h->visualRuntime->tracker;req(tracker->researchDirectWinnerEnabled(),"direct winner prerequisite");
 mr_research_set_observation_hooks(h->stepConfig.deviceObservationProgram.context,prepare,finish,abortGuard);
 CandidatePublication store(2,l.actorObservationSize);auto staged=store.begin();
 failFinish=true;auto denied=step(*h);CandidatePublication::Certificate failure{};bool rejected=!store.publish(staged,failure);
 std::cout<<"pre_submit_status="<<(unsigned)denied.diagnostic.status<<" published="<<denied.diagnostic.published<<" aborts="<<aborts<<" host_refused="<<rejected<<" generation="<<store.generation()<<"\n";
 req(!denied.diagnostic.succeeded()&&!denied.diagnostic.published&&aborts==1&&rejected&&store.generation()==0,"pre-submit failure prerequisite");
 staged=store.begin(); auto result=step(*h);
 std::cout<<"step_status="<<(unsigned)result.diagnostic.status<<" published="<<result.diagnostic.published<<" success="<<result.diagnostic.successfulStepCount<<" failed="<<result.diagnostic.failedStepCount<<" statuses="<<result.result.statuses.size()<<" contacts="<<result.result.contactStatuses.size()<<" actor="<<result.result.actorObservations.size()<<"\n";
 for(const auto& x:result.result.statuses)std::cout<<"world env="<<x.environment<<" code="<<x.code<<"\n";
 for(const auto& x:result.result.contactStatuses)std::cout<<"contact env="<<x.environment<<" code="<<x.code<<" raw_required="<<x.requiredRawContacts<<"\n";
 req(result.diagnostic.published && result.result.statuses.size()==2 && result.result.contactStatuses.size()==2 && result.result.actorObservations.size()==2*l.actorObservationSize,"validated completed result unavailable");
 CandidatePublication::Certificate cert{true,true,{},{}};
 for(unsigned e=0;e<2;++e){const auto& w=result.result.statuses[e];const auto& c=result.result.contactStatuses[e];
  req(w.environment==e && c.environment==e && w.controlStep==0 && c.controlStep==0,"status identity");cert.worldStatus.push_back(w.code);cert.contactStatus.push_back(c.code);
  staged.environments[e].actor.assign(result.result.actorObservations.begin()+e*l.actorObservationSize,result.result.actorObservations.begin()+(e+1)*l.actorObservationSize);
 }
 std::ofstream output(argv[3],std::ios::binary);req(bool(output),"output");auto write=[&](const auto& v){std::uint64_t n=v.size();output.write((const char*)&n,sizeof(n));output.write((const char*)v.data(),n*sizeof(v[0]));};
 write(staged.environments[0].actor);write(staged.environments[1].actor);
 bool expected=mixed?((cert.worldStatus[0]!=0 || cert.contactStatus[0]!=0) && cert.worldStatus[1]==0 && cert.contactStatus[1]==0):(cert.worldStatus[0]==0&&cert.worldStatus[1]==0&&cert.contactStatus[0]==0&&cert.contactStatus[1]==0);
 bool committed=store.publish(staged,cert);std::cout<<"expected_physics_outcome="<<expected<<" host_committed="<<committed<<" generation="<<store.generation()<<" env_steps="<<store.records()[0].acceptedSteps<<","<<store.records()[1].acceptedSteps<<"\n";
 write(store.records()[0].actor);write(store.records()[1].actor);output.flush();
 req(expected,"mixed/control physics prerequisite not reached");req(committed && store.generation()==1 && store.records()[1].acceptedSteps==1 && store.records()[0].acceptedSteps==(mixed?0u:1u),"publication incorrect");
 return 0;
}catch(const std::exception& e){std::cerr<<"error="<<e.what()<<"\n";return 2;}}
