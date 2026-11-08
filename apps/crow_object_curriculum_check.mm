#import <Foundation/Foundation.h>
#include "metalrobo/CrowManipulation.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/c_api.h"
#include "metalrobo/detail/Checkpoint.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <iomanip>
#include <memory>
#include <set>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
extern char** environ;
using namespace metalrobo;
namespace cp=metalrobo::checkpoint;
namespace {
constexpr unsigned N=33,F=1247,T=70,S=35;
constexpr std::uint64_t seed=2651316001ull;
void require(bool v,const std::string& reason){if(!v)throw std::runtime_error(reason);}
template<class R>void check(const R&r,const char*s){require(r.succeeded(),std::string(s)+": "+r.message);}
void api(int status,const char*s){require(status==0,std::string(s)+": "+mr_last_error());}
using Owner=std::unique_ptr<MRTaskRolloutHandle,decltype(&mr_task_rollout_destroy)>;
struct Fixture {
    CompiledRun run;VisualSensorProgram visual;
    std::array<std::string,4> paths,ids{"robot","locomotion_ground","object_support","crow_manipulandum"};
    std::array<MRTaskVisualPackC,4> packs{};
    MRRunManifestC manifest{};MRTaskVisualObservationConfigC camera{};MRRetinalSensorConfigC retina{};
    MRCrowObjectPushCurriculumC curriculum{sizeof(MRCrowObjectPushCurriculumC),1,32,0,.223f,.227f,.028f,.042f};
    explicit Fixture(const std::filesystem::path& assets) {
        visual.cameraParentBody="crow_head";visual.cameraPosition={.018f,.026f,.012f,0};
        visual.cameraOrientation=crowObjectPushCameraOrientation;
        visual.width=160;visual.height=120;visual.verticalFieldOfViewDegrees=45;
        visual.minimumVisiblePixels=1;visual.nominalRateHz=15;visual.retinal=RetinalSensorProgram{};
        visual.retinal->rightCameraOrientation=visual.cameraOrientation;
        for(unsigned i=0;i<4;++i) {
            paths[i]=(assets/(ids[i]+".mrvpack")).string();VisualAssetPackV2 p;std::string reason;
            require(readVisualAssetPack(paths[i],p,&reason),"Read canonical visual asset: "+reason);
            visual.assets.push_back({paths[i],ids[i],p.contentHash,i+1,i+1});
            packs[i]={paths[i].c_str(),ids[i].c_str(),i+1,i+1};
        }
        visual.fingerprint=visualSensorProgramFingerprint(visual);
        auto m=makeAmericanCrowObjectPushFixtureManifest();m.sensors.observation={};
        m.task=makeAmericanCrowObjectPushTaskPack(m.sensors.observation,m.reality.reset,&*visual.retinal);
        m.sensors.deviceVisual=visual;applyAmericanCrowForagingFixtureLimits(m);
        applyAmericanCrowObjectPushCurriculum(m,{32,.223f,.227f,.028f,.042f});
        check(compileRun(m,run),"Compile expected native curriculum");
        camera.packs=packs.data();camera.pack_count=4;camera.renderer_profile="sensor_fast";
        camera.camera_parent_body="crow_head";camera.camera_position[0]=.018f;camera.camera_position[1]=.026f;camera.camera_position[2]=.012f;
        std::copy_n(&visual.cameraOrientation.x,4,camera.camera_orientation);
        camera.width=160;camera.height=120;camera.minimum_visible_pixels=1;
        camera.vertical_field_of_view_degrees=45;camera.nominal_rate_hz=15;
        retina.struct_size=sizeof(retina);retina.abi_version=1;retina.right_camera_position[0]=.018f;
        retina.right_camera_position[1]=-.026f;retina.right_camera_position[2]=.012f;
        std::copy_n(&visual.cameraOrientation.x,4,retina.right_camera_orientation);
        retina.width=16;retina.height=12;retina.sample_numerator=3;retina.sample_denominator=10;
        retina.latency_ticks=2;retina.minimum_depth_meters=.01f;retina.maximum_depth_meters=5;
        manifest.source=MR_RUN_SOURCE_BIRDFLOW_AMERICAN_CROW_JOURNEY;manifest.visual_sensor_program=&camera;
        auto& p=manifest.profile;p.environment_count=N;p.physics_substeps=32;p.velocity_iterations=4;p.final_velocity_iterations=2;
        p.control_timestep_seconds=.02f;p.seed=seed;p.override_difficulty_band_range=1;
        p.birdflow_journey_variant=MR_BIRDFLOW_JOURNEY_V14_OBJECT_PUSH;
    }
    Owner create() {
        Owner h(mr_create_task_rollout_object_push_curriculum(&manifest,&retina,&curriculum),mr_task_rollout_destroy);
        require(bool(h),std::string("Create curriculum: ")+mr_last_error());const auto l=mr_task_rollout_layout(h.get());
        require(l.environment_count==N && l.actor_observation_count==F && l.critic_observation_count==100 &&
            l.maximum_episode_steps==32 && l.world_fingerprint==run.world().fingerprint() &&
            l.task_fingerprint==run.task().fingerprint() && l.observation_fingerprint==run.task().observationFingerprint(),
            "C and compiled curriculum contracts differ");
        return h;
    }
    void policy(const std::filesystem::path& path) {
        PolicyPack p;p.id="neutral-object-curriculum-instrument";p.revision=1;
        p.observationMean.assign(F,0);p.observationInverseStandardDeviation.assign(F,1);
        p.layers={{.inputCount=F,.outputCount=18,.activation=PolicyActivation::identity,
            .weights=std::vector<float>(F*18,0),.bias=std::vector<float>(18,0)}};
        p.criticObservationMean.assign(100,0);p.criticObservationInverseStandardDeviation.assign(100,1);
        p.criticLayers={{.inputCount=100,.outputCount=1,.activation=PolicyActivation::identity,
            .weights=std::vector<float>(100,0),.bias={.25f}}};
        bindPolicyPack(p,run.task());check(writePolicyPack(p,path),"Write measurement policy");
    }
};
struct Hash {
    CC_SHA256_CTX state{};Hash(){CC_SHA256_Init(&state);}
    template<class V>void add(const V* ptr,std::size_t count){const auto bytes=count*sizeof(V);require(bytes<UINT32_MAX,"Hash span too large");CC_SHA256_Update(&state,ptr,static_cast<CC_LONG>(bytes));}
    cp::Digest finish(){cp::Digest d;CC_SHA256_Final(d.data(),&state);return d;}
};
void clock(MRTaskRolloutHandle*h,unsigned want) {
    std::array<std::uint64_t,N> ticks{};api(mr_task_rollout_copy_lifetime_ticks(h,ticks.data(),N),"Read accepted life clock");
    require(std::all_of(ticks.begin(),ticks.end(),[&](auto v){return v==want;}),"Episode reset changed lifetime clock");
}
std::array<cp::Bytes,3> parts(const std::filesystem::path& path) {
    const auto bytes=cp::readBounded(path,64ull*1024*1024);cp::Reader envelope{bytes};const auto h=envelope.get<cp::Header>();
    cp::Reader r{cp::open(bytes,{'N','U','M','I','L','F','0','1'},h.identity)};r.take(40);
    std::array<cp::Bytes,3> values;for(auto& v:values){const auto data=r.blob();v.assign(data.begin(),data.end());}r.end();
    require(values[2].empty(),"Resource-free life contains food state");return values;
}
std::array<cp::Digest,2> execute(Fixture& f,const std::filesystem::path& parent,const std::filesystem::path& out,unsigned chunk,bool restored=false) {
    require(std::filesystem::create_directory(out),"Execution output exists");auto h=f.create();
    api(mr_task_rollout_load_policy_pack(h.get(),(parent/"neutral.policypack").c_str()),"Install neutral policy");
    std::array<std::uint64_t,N> sentinel;sentinel.fill(0xababababababababull);
    require(mr_task_rollout_copy_lifetime_ticks(h.get(),sentinel.data(),N)!=0 &&
        std::all_of(sentinel.begin(),sentinel.end(),[](auto v){return v==0xababababababababull;}),"Unadvanced clock published or modified output");
    if(restored){api(mr_task_rollout_restore_life_checkpoint(h.get(),(parent/"prefix.life").c_str()),"Restore curriculum life");clock(h.get(),S);}
    api(mr_task_rollout_set_state_readback(h.get(),1),"Enable bounded state instrument");
    Hash complete,future;std::ofstream trace(out/"object-state.tsv");trace<<"tick\tlane\tx\ty\tz\tretinal_valid\n"<<std::setprecision(10);
    std::array<std::array<float,2>,N> last{};unsigned resetEpochs=0;bool imageDiversity=false;std::set<std::array<float,2>> allPlacements;
    for(unsigned start=restored?S:0;start<T;){
        unsigned count=std::min(chunk,T-start);
        MRTaskRolloutAdvanceC a{};api(mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,count,1,1,&a),"Advance curriculum");
        require(a.successful_environment_steps==count*N && !a.failed_environment_steps && !a.host_requested_resets,"Curriculum rejected physics or used a host reset");
        const auto* actor=mr_task_rollout_actor_observations(h.get());const auto* tr=mr_task_rollout_transitions(h.get());
        for(unsigned j=0;j<count;++j)for(unsigned e=0;e<N;++e) {
            const unsigned tick=start+j,at=j*N+e;require(!tr[at].physics_error,"Physics error in publication");
            complete.add(actor+at*F,F);complete.add(tr+at,1);
            if(tick>=S){future.add(actor+at*F,F);future.add(tr+at,1);}
            require(bool(tr[at].timeout)==(tick%32==31),"Native episode timeout phase differs");
            const auto valid=actor[at*F+1245];
            if(tick%32<2)require(valid==0,"Reset leaked a stale camera frame");
            else require(valid==1,"Reset camera did not reacquire on schedule");
        }
        // Only the one-tick reference publishes trajectories. Other schedules
        // must reproduce its whole hidden state and future hashes.
        if(chunk==1 && !restored) {
            std::vector<float> states(N*3*13);api(mr_task_rollout_copy_final_scene_states(h.get(),states.data(),states.size()),"Read actual object states");
            std::set<std::array<float,2>> placements;
            for(unsigned e=0;e<N;++e){const auto* p=states.data()+(e*3+2)*13;
                require(std::abs(p[2]-.257f)<.0005f,"Reset object exceeded settling-depth control");
                if(start%32==0){
                    require(p[0]>=f.curriculum.minimum_x && p[0]<=f.curriculum.maximum_x && p[1]>=f.curriculum.minimum_y && p[1]<=f.curriculum.maximum_y,"Native placement outside authored bounds");
                    placements.insert({p[0],p[1]});allPlacements.insert({p[0],p[1]});
                    if(start)require(last[e]!=std::array<float,2>{p[0],p[1]},"Episode repeated the same object placement");
                    last[e]={p[0],p[1]};
                }else require(std::hypot(p[0]-last[e][0],p[1]-last[e][1])<1e-5,"Neutral crow or reset launched the object");
                trace<<start<<'\t'<<e<<'\t'<<p[0]<<'\t'<<p[1]<<'\t'<<p[2]<<'\t'<<actor[e*F+1245]<<'\n';
                if(start==2 && e)for(unsigned k=93;k<1245;++k)imageDiversity|=actor[e*F+k]!=actor[k];
            }
            if(start%32==0){require(placements.size()==N,"Placement RNG aliases native lanes");++resetEpochs;}
        }
        start+=count;clock(h.get(),start);
        if(!restored && chunk==1 && start==S)api(mr_task_rollout_save_life_checkpoint(h.get(),(parent/"prefix.life").c_str()),"Save pending-camera prefix");
    }
    if(chunk==1 && !restored)require(resetEpochs==3 && allPlacements.size()==99 && imageDiversity,"Missing reset/RGB placement diversity evidence");
    api(mr_task_rollout_save_life_checkpoint(h.get(),(out/"final.life").c_str()),"Save final curriculum life");
    return {complete.finish(),future.finish()};
}
void child(const std::filesystem::path& program,const std::filesystem::path& parent,const std::filesystem::path& assets,unsigned chunk) {
    const auto out=parent/("resumed"+std::to_string(chunk));std::vector<std::string> args{program.string(),"--restore",parent.string(),out.string(),std::to_string(chunk),assets.string()};
    std::vector<char*> pointers;for(auto& x:args)pointers.push_back(x.data());pointers.push_back(nullptr);
    posix_spawn_file_actions_t actions;require(!posix_spawn_file_actions_init(&actions),"Spawn setup failed");
    const auto log=parent/("child"+std::to_string(chunk)+".log");require(!posix_spawn_file_actions_addopen(&actions,1,log.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600),"Child log exists");
    require(!posix_spawn_file_actions_adddup2(&actions,1,2),"Child log redirect failed");pid_t pid=0;
    const auto result=posix_spawn(&pid,program.c_str(),&actions,nullptr,pointers.data(),environ);posix_spawn_file_actions_destroy(&actions);require(!result,"Spawn failed");
    const auto began=std::chrono::steady_clock::now();int status=0;
    while(true){const auto wait=waitpid(pid,&status,WNOHANG);if(wait==pid)break;require(wait==0,"Child observation failed");
        if(std::chrono::steady_clock::now()-began>std::chrono::seconds(60)){kill(pid,SIGTERM);waitpid(pid,&status,0);throw std::runtime_error("Owned child timed out");}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));}
    require(WIFEXITED(status) && WEXITSTATUS(status)==0,"Fresh child failed; retained log");
}
}
int main(int argc,char**argv){@autoreleasepool{std::filesystem::path out;bool owns=false;try{
    if(argc==6 && std::string(argv[1])=="--restore") {
        Fixture f(argv[5]);const auto hashes=execute(f,argv[2],argv[3],std::stoul(argv[4]),true);
        cp::writeExclusive(std::filesystem::path(argv[3])/"stream.sha256",std::as_bytes(std::span(hashes[0])));return 0;
    }
    require(argc==3,"Usage: metalrobo_crow_object_curriculum_check NEW_OUTPUT CANONICAL_ASSETS");
    out=std::filesystem::absolute(argv[1]);const auto assets=std::filesystem::absolute(argv[2]);require(std::filesystem::create_directory(out),"Output exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"question":"Does native object practice renew actual placements without a host reset and preserve exact physical/camera/lifetime state across two automatic episode resets?","units":"33 native lanes across a SIMD boundary, three episodes in one fixed apparatus; not biological replication","conditions":{"seed":2651316001,"episode_ticks":32,"total_ticks":70,"split_tick":35,"x_range_m":[0.223,0.227],"y_range_m":[0.028,0.042],"physics_substeps":32},"prediction":"99 distinct lane/episode reset placements within bounds; neutral horizontal drift<10um, resting-depth error<0.5mm; blank camera first2 ticks then fresh RGB with lane diversity; clocks70, chunk1/8 complete streams and native components exact; two fresh35tick futures exact across reset64","budget":"one reference70, partition70, two35tick fresh futures, 33lanes; six malformed configs and incompatible legacy policy; each child60s, no retries, no video, under32MiB raw","scope":"engineering reset/RNG/camera/continuation admission, not learned manipulation or general ecology"})";
    Fixture f(assets);f.policy(out/"neutral.policypack");
    const auto reference=execute(f,out,out/"reference",1),partition=execute(f,out,out/"partition",8);
    require(reference[0]==partition[0],"Complete streams depend on chunk size");
    require(parts(out/"reference/final.life")==parts(out/"partition/final.life"),"Partition native hidden state differs");
    for(unsigned chunk:{1u,8u}){child(std::filesystem::absolute(argv[0]),out,assets,chunk);const auto d=out/("resumed"+std::to_string(chunk));
        require(cp::readBounded(d/"stream.sha256",32)==cp::Bytes(reinterpret_cast<const std::byte*>(reference[1].data()),reinterpret_cast<const std::byte*>(reference[1].data()+32)),"Fresh future stream differs");
        require(parts(out/"reference/final.life")==parts(d/"final.life"),"Fresh future hidden state differs");}
    const auto original=f.curriculum;
    for(unsigned i=0;i<6;++i){f.curriculum=original;
        if(i==0)f.curriculum.abi_version=2;if(i==1)f.curriculum.reserved=1;
        if(i==2)f.curriculum.episode_steps=31;if(i==3)f.curriculum.minimum_x=.3f;
        if(i==4)f.curriculum.maximum_y=NAN;if(i==5)f.curriculum.minimum_y=.02f;
        Owner h(mr_create_task_rollout_object_push_curriculum(&f.manifest,&f.retina,&f.curriculum),mr_task_rollout_destroy);
        require(!h,"Malformed curriculum accepted");}
    f.curriculum=original;Owner old(mr_create_task_rollout_retinal(&f.manifest,&f.retina),mr_task_rollout_destroy);require(bool(old),"Construct unchanged legacy fixture");
    require(mr_task_rollout_layout(old.get()).maximum_episode_steps==100000,"Optional curriculum changed legacy horizon");
    require(mr_task_rollout_load_policy_pack(old.get(),(out/"neutral.policypack").c_str())!=0,"Curriculum policy silently rebound to legacy task");
    std::ofstream(out/"summary.json")<<R"({"status":"passed_engineering_controls","accepted_transitions":6930,"native_lane_episode_placements":99,"episode_resets_host_requested":0,"accepted_lifetime_ticks":70,"chunk1_8_stream_and_native_state_exact":true,"fresh_process_future_exact":true,"reset_camera_cold_start_and_actual_RGB_diversity":true,"malformed_config_rejections":6,"legacy_policy_contract_rejection":true,"physics_rejections":0,"learned_manipulation_qualified":false,"ecological_environment_qualified":false})";
    return 0;
}catch(const std::exception& e){if(owns)std::ofstream(out/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
