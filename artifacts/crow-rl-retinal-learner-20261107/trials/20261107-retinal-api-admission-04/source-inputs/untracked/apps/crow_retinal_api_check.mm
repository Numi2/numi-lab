#import <Foundation/Foundation.h>
#include "metalrobo/c_api.h"
#include "metalrobo/CollisionVisualProxy.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace metalrobo;
namespace {
constexpr unsigned N=2,T=16,F=1251;
void require(bool value,const std::string& reason) {if (!value) throw std::runtime_error(reason);}
template<class R> void check(const R& result,const char* reason) {require(result.succeeded(),std::string(reason)+": "+result.message);}
void api(int result,const char* reason) {require(result==0,std::string(reason)+": "+mr_last_error());}
template<class V> void save(const std::filesystem::path& path,const std::vector<V>& values) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(V));require(bool(f),"Evidence write failed");
}
using CrowRunOwner=std::unique_ptr<MRTaskRolloutHandle,decltype(&mr_task_rollout_destroy)>;
struct Fixture {
    RunManifest manifest;CompiledRun run;VisualSensorProgram visual;
    MRRunManifestC c{};MRTaskVisualObservationConfigC camera{};MRRetinalSensorConfigC retina{};
    std::array<std::string,2> paths;std::array<MRTaskVisualPackC,2> packs{};
    Fixture(const std::filesystem::path& out) {
        manifest.id="american_crow_retinal_feeding_v13_run";manifest.robot=makeAmericanCrowActiveHeadRobotPack();
        manifest.scene=makeBirdFlowAmericanCrowFlightScenePack();manifest.sensors.id="american_crow_retinal_v13_rgb_proprioception_nutrition";
        manifest.task=makeAmericanCrowFeedingStationTaskPack(manifest.sensors.observation,manifest.reality.reset);
        manifest.profile.id="native_task_rollout";manifest.profile.environmentCount=N;manifest.profile.controlSteps=1;
        manifest.profile.physicsSubsteps=4;manifest.profile.velocityIterations=4;manifest.profile.finalVelocityIterations=2;
        manifest.profile.controlTimestepSeconds=.02f;manifest.profile.seed=2651108801;manifest.profile.capacities=manifest.task.capacities;
        manifest.profile.minimumDifficultyBand=manifest.profile.maximumDifficultyBand=0;
        manifest.reality.id="birdflow_american_crow_estimated_hybrid_nominal_reality";manifest.teacher.id="no_teacher";
        CompiledRun base;check(compileRun(manifest,base),"compile asset source world");
        std::vector<unsigned> robot;
        for (unsigned body=0;body<base.model().bodies.size();++body) if (base.model().bodies[body].articulationIndex!=MR_INVALID_INDEX) robot.push_back(body);
        const auto ground=base.world().sceneBodyIndices();require(ground.size()==1,"Fixture is not the canonical supported station");
        const std::array<std::vector<unsigned>,2> selections{robot,std::vector<unsigned>{ground.front()}};
        const std::array<std::string,2> ids{"robot",base.model().bodyNames[ground.front()]};
        for (unsigned i=0;i<2;++i) {
            const auto pack=makeCollisionProxyVisualAssetPack(base.model(),selections[i],base.world().fingerprint(),ids[i]+"_explicit_development_proxy",
                i ? mr_float4{.25f,.29f,.25f,1} : mr_float4{.035f,.035f,.04f,1},16);
            paths[i]=(out/(ids[i]+".mrvpack")).string();std::string reason;
            require(writeVisualAssetPack(pack,paths[i],&reason),reason);
            visual.assets.push_back({paths[i],ids[i],pack.contentHash,i+1,i+1});
        }
        visual.cameraParentBody="crow_head";visual.cameraPosition={.018f,.026f,.012f,0};visual.cameraOrientation={-.5f,.5f,-.5f,.5f};
        visual.width=160;visual.height=120;visual.minimumVisiblePixels=1;visual.verticalFieldOfViewDegrees=45;
        visual.nominalRateHz=15;visual.retinal=RetinalSensorProgram{};visual.fingerprint=visualSensorProgramFingerprint(visual);
        manifest.sensors.observation={};manifest.reality.reset={};
        manifest.task=makeAmericanCrowRetinalFeedingTaskPack(manifest.sensors.observation,manifest.reality.reset,*visual.retinal);
        manifest.sensors.deviceVisual=visual;manifest.profile.capacities=manifest.task.capacities;
        check(compileRun(manifest,run),"compile expected retinal task");
        for (unsigned i=0;i<2;++i) packs[i]={paths[i].c_str(),visual.assets[i].assetId.c_str(),i+1,i+1};
        camera.packs=packs.data();camera.pack_count=2;camera.renderer_profile="sensor_fast";camera.camera_parent_body="crow_head";
        camera.camera_position[0]=.018f;camera.camera_position[1]=.026f;camera.camera_position[2]=.012f;
        const float orientation[4]={-.5f,.5f,-.5f,.5f};std::copy_n(orientation,4,camera.camera_orientation);
        camera.width=160;camera.height=120;camera.minimum_visible_pixels=1;camera.vertical_field_of_view_degrees=45;camera.nominal_rate_hz=15;
        retina.struct_size=sizeof(retina);retina.abi_version=1;retina.right_camera_position[0]=.018f;retina.right_camera_position[1]=-.026f;retina.right_camera_position[2]=.012f;
        std::copy_n(orientation,4,retina.right_camera_orientation);retina.width=16;retina.height=12;
        retina.sample_numerator=3;retina.sample_denominator=10;retina.latency_ticks=2;retina.minimum_depth_meters=.01f;retina.maximum_depth_meters=5;
        c.source=MR_RUN_SOURCE_BIRDFLOW_AMERICAN_CROW_JOURNEY;c.visual_sensor_program=&camera;
        c.profile.environment_count=N;c.profile.physics_substeps=4;c.profile.velocity_iterations=4;c.profile.final_velocity_iterations=2;
        c.profile.control_timestep_seconds=.02f;c.profile.seed=2651108801;c.profile.override_difficulty_band_range=1;
        c.profile.minimum_difficulty_band=c.profile.maximum_difficulty_band=0;c.profile.birdflow_journey_variant=MR_BIRDFLOW_JOURNEY_V13_RETINAL_FEEDING;
        std::ofstream artifact(out/"retinal-observation.json");
        artifact<<R"({"format":"numi.retinal-observation.v1","id":"explicit-supported-station-proxy-development-only","packs":[)";
        for (unsigned i=0;i<2;++i) artifact<<(i ? "," : "")<<"{\"path\":"<<std::quoted(std::filesystem::path(paths[i]).filename().string())
            <<",\"asset_id\":"<<std::quoted(visual.assets[i].assetId)<<",\"semantic_id\":"<<i+1<<",\"instance_id\":"<<i+1<<"}";
        artifact<<R"(],"camera":{"parent_body":"crow_head","position":[0.018,0.026,0.012],"orientation":[-0.5,0.5,-0.5,0.5],"width":160,"height":120,"minimum_visible_pixels":1,"vertical_field_of_view_degrees":45,"nominal_rate_hz":15,"maximum_retained_bytes":536870912},"retinal":{"right_position":[0.018,-0.026,0.012],"right_orientation":[-0.5,0.5,-0.5,0.5],"width":16,"height":12,"sample_numerator":3,"sample_denominator":10,"latency_ticks":2,"minimum_depth_meters":0.01,"maximum_depth_meters":5}})";
    }
    CrowRunOwner create() {
        CrowRunOwner h(mr_create_task_rollout_retinal(&c,&retina),mr_task_rollout_destroy);require(bool(h),std::string("Create retinal C handle: ")+mr_last_error());
        const auto layout=mr_task_rollout_layout(h.get());
        require(layout.actor_observation_count==F && layout.critic_observation_count==97 && layout.action_count==18 &&
            layout.world_fingerprint==run.world().fingerprint() && layout.observation_fingerprint==run.task().observationFingerprint() &&
            layout.task_fingerprint==run.task().fingerprint(),"C entry differs from exact native task contract");
        return h;
    }
    std::filesystem::path policy(const std::filesystem::path& out,const std::string& id,float jaw=0) {
        PolicyPack p;p.id=id;p.observationMean.assign(F,0);p.observationInverseStandardDeviation.assign(F,1);
        PolicyDenseLayer a{.inputCount=F,.outputCount=18,.activation=PolicyActivation::identity,.weights=std::vector<float>(F*18,0),.bias=std::vector<float>(18,0)};
        a.bias[17]=jaw;p.layers.push_back(a);p.criticObservationMean.assign(97,0);p.criticObservationInverseStandardDeviation.assign(97,1);
        p.criticLayers={{.inputCount=97,.outputCount=1,.activation=PolicyActivation::identity,.weights=std::vector<float>(97,0),.bias={.25f}}};
        bindPolicyPack(p,run.task());const auto path=out/(id+".policypack");check(writePolicyPack(p,path),"save explicit control policy");return path;
    }
};
struct Stream {std::vector<float> actor,actions,latents,q,v;std::vector<MRTaskTransitionC> transitions;};
Stream execute(Fixture& f,const std::filesystem::path& out,unsigned chunk,const std::filesystem::path& policy) {@autoreleasepool {
    require(std::filesystem::create_directory(out),"Evidence path exists");auto h=f.create();api(mr_task_rollout_load_policy_pack(h.get(),policy.c_str()),"install control actor");
    api(mr_task_rollout_set_state_readback(h.get(),1),"enable final state inspection");Stream result;std::vector<float> previous;
    for (unsigned start=0;start<T;start+=chunk) {
        const unsigned count=std::min(chunk,T-start);std::vector<unsigned> masks(count*N,0);MRTaskRolloutAdvanceC a{};
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),count,1,1,&a),"terminal-enabled advance");
        require(a.successful_environment_steps==count*N && !a.failed_environment_steps,"C rollout failed physics");
        const auto* actor=mr_task_rollout_actor_observations(h.get());const auto* actions=mr_task_rollout_policy_actions(h.get());
        const auto* latents=mr_task_rollout_policy_latents(h.get());const auto* transitions=mr_task_rollout_transitions(h.get());
        require(actor && actions && latents && transitions,"Missing learning publication");
        if (!previous.empty()) for (unsigned env=0;env<N;++env)
            require(!std::memcmp(&previous[env*F+97],actor+env*F+97,1154*sizeof(float)),"Terminal retinal features differ from next chunk's first actor input");
        previous.assign(actor+count*N*F,actor+(count+1)*N*F);
        save(out/("terminal-"+std::to_string(start+count)+".f32"),previous);
        for (unsigned t=0;t<=count;++t) for (unsigned env=0;env<N;++env) {
            const auto index=(t*N+env)*F;const unsigned tick=start+t;
            require(actor[index+1249]==(tick<2 ? 0 : 1),"Terminal/live validity differs from two-tick delay");
            double sum=0;for (unsigned c=97;c<1249;++c) {require(std::isfinite(actor[index+c]),"Nonfinite actor image");sum+=actor[index+c];}
            require(tick<2 ? sum==0 : sum>1,"Terminal/live slot silently lost its rendered RGB");
        }
        result.actor.insert(result.actor.end(),actor,actor+count*N*F);result.actions.insert(result.actions.end(),actions,actions+count*N*18);
        result.latents.insert(result.latents.end(),latents,latents+count*N*18);result.transitions.insert(result.transitions.end(),transitions,transitions+count*N);
    }
    result.q.assign(mr_task_rollout_final_q(h.get()),mr_task_rollout_final_q(h.get())+N*23);
    result.v.assign(mr_task_rollout_final_v(h.get()),mr_task_rollout_final_v(h.get())+N*22);
    save(out/"actor.f32",result.actor);save(out/"actions.f32",result.actions);save(out/"latents.f32",result.latents);
    save(out/"transitions.bin",result.transitions);save(out/"q.f32",result.q);save(out/"v.f32",result.v);return result;
}}
}
int main(int argc,char** argv) {@autoreleasepool {bool owns=false;try {
    require(argc==2,"Usage: metalrobo_crow_retinal_api_check NEW_OUTPUT");const std::filesystem::path out=argv[1];
    require(std::filesystem::create_directory(out),"Output exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-retinal-api-admission.v4","parent":"01 and02 exposed partition-dependent retinal frames with exact physics/actions; phase0 alone did not fix it. Terminal pass used the regular pre-step state selector, whose guard rejects controlStep==count and leaves stale projection input. v3 projects accepted terminal state directly and passed partition/terminal checks, but its immediate open/negative-close gesture consumed nothing. v4 uses established feeding-control settling64 ticks, open32 and neutral-close32; no timing/partition thresholds changed","unit":"two native lanes at50Hz in canonical fixed supported station; explicit generated collision proxies, no biological or visual-learning claim","seed":2651108801,"predictions":{"negative_configs_rejected":6,"terminal_and_next_retina_bit_identical":true,"chunk1_8_complete_learning_stream_bit_identical":true,"cold_start_blind_ticks":2,"zero_sensor_and_physics_errors":true,"world_reset_preserves_consumed_food":true},"stopping":"two16-tick partition runs plus bounded64-settle/32-open/32-neutral-close feeding and8-tick world-reset control; retain failed attempts, no threshold changes"})";
    Fixture f(out);std::ofstream negatives(out/"negative-controls.jsonl");
    for (unsigned test=0;test<6;++test) {
        auto c=f.c;auto retinal=f.retina;const MRRetinalSensorConfigC* input=&retinal;
        if (test==0) retinal.struct_size=0;if (test==1) retinal.abi_version=2;
        if (test==2) input=nullptr;if (test==3) c.visual_sensor_program=nullptr;
        if (test==4) c.source=MR_RUN_SOURCE_BIRDFLOW_DOVE;
        CrowRunOwner h(test==5 ? mr_create_task_rollout(&c) : mr_create_task_rollout_retinal(&c,input),mr_task_rollout_destroy);
        negatives<<"{\"case\":"<<test<<",\"rejected\":"<<(!h ? "true" : "false")<<",\"reason\":"<<std::quoted(mr_last_error())<<"}\n";negatives.flush();require(!h,"Malformed retinal C configuration was accepted");
    }
    const auto zero=f.policy(out,"zero-actor-with-critic");const auto one=execute(f,out/"chunk1",1,zero),eight=execute(f,out/"chunk8",8,zero);
    const auto same=[](const auto& a,const auto& b){return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]));};
    require(same(one.actor,eight.actor) && same(one.actions,eight.actions) && same(one.latents,eight.latents) && same(one.transitions,eight.transitions) && same(one.q,eight.q) && same(one.v,eight.v),"Terminal-enabled learning stream depends on submission partition");
    {auto h=f.create();const auto open=f.policy(out,"open-beak",1),close=f.policy(out,"close-beak",0);
        std::array<MRTaskResourceAccountingC,N> before{},after{};
        std::vector<unsigned> masks(32*N,0);MRTaskRolloutAdvanceC a{};
        api(mr_task_rollout_load_policy_pack(h.get(),close.c_str()),"install settling control");
        for (unsigned i=0;i<2;++i) api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),32,1,1,&a),"settle supported body before feeding");
        api(mr_task_rollout_load_policy_pack(h.get(),open.c_str()),"install open-beak control");
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),32,1,1,&a),"open beak");
        api(mr_task_rollout_load_policy_pack(h.get(),close.c_str()),"install close-beak control");
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),32,1,1,&a),"close beak");
        api(mr_task_rollout_copy_resource_accounting(h.get(),before.data(),N),"publish pre-reset food");
        api(mr_task_rollout_reset(h.get(),2651108802),"reset world and retina");masks.resize(8*N);
        api(mr_task_rollout_advance(h.get(),nullptr,0,masks.data(),masks.size(),8,1,1,&a),"advance after world reset");
        api(mr_task_rollout_copy_resource_accounting(h.get(),after.data(),N),"publish post-reset food");
        save(out/"resource-before-reset.bin",std::vector(before.begin(),before.end()));save(out/"resource-after-reset.bin",std::vector(after.begin(),after.end()));
        for (unsigned env=0;env<N;++env) {
            require(before[env].consumed_micrograms>0,"Controlled gesture did not consume food before reset");
            require(before[env].remaining_micrograms==after[env].remaining_micrograms && before[env].crop_micrograms==after[env].crop_micrograms &&
                before[env].consumed_micrograms==after[env].consumed_micrograms && after[env].accepted_control_ticks==before[env].accepted_control_ticks+8,"World reset refilled food or changed lifetime accounting");
        }
        const auto* actor=mr_task_rollout_actor_observations(h.get());
        for (unsigned t=0;t<=8;++t) for (unsigned env=0;env<N;++env) require(actor[(t*N+env)*F+1249]==(t<2 ? 0 : 1),"Whole-world reset retained stale retinal frames");
    }
    const std::string result=R"({"schema":"numi.crow-retinal-api-evidence.v1","status":"passed","malformed_configs_rejected":6,"terminal_retinal_slots_valid":true,"terminal_next_retina_bit_identical":true,"chunk1_8_bit_identical":true,"world_reset_preserves_food":true,"visual_learning_qualified":false,"ecological_environment_qualified":false})";
    std::ofstream(out/"result.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
} catch (const std::exception& e) {if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
