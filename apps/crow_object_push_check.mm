#import <Foundation/Foundation.h>
#include "metalrobo/CrowManipulation.hpp"
#include "metalrobo/RetinalObserver.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace metalrobo;
namespace {
constexpr unsigned lanes=8,ticks=160,split=48,features=1247,criticFeatures=100;
constexpr std::uint64_t seed=2651313001ull;
void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
template<class R>void check(const R&r,const char*s){require(r.succeeded(),std::string(s)+": "+r.message);}
template<class T>void save(const std::filesystem::path&p,const std::vector<T>&v){
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));require(bool(f),"Evidence write failed");
}
std::string hex(const checkpoint::Digest& d){std::ostringstream o;o<<std::hex<<std::setfill('0');for(auto b:d)o<<std::setw(2)<<unsigned(b);return o.str();}
struct Stream {
    CC_SHA256_CTX ctx{};std::uint64_t bytes=0;
    Stream(){CC_SHA256_Init(&ctx);}
    template<class T>void add(const std::vector<T>&v){const auto b=std::as_bytes(std::span(v));
        for(std::size_t i=0;i<b.size();){const auto n=std::min<std::size_t>(b.size()-i,1u<<20);CC_SHA256_Update(&ctx,b.data()+i,static_cast<CC_LONG>(n));i+=n;}bytes+=b.size();}
    checkpoint::Digest finish(){checkpoint::Digest d;CC_SHA256_Final(d.data(),&ctx);return d;}
};
struct Fixture {CompiledRun run;CrowForagingVisual visual;unsigned object,local,head,beak,shape,goalChannel;};
unsigned findBody(const CompiledRun& r,const std::string& name){const auto& n=r.model().bodyNames;const auto i=std::find(n.begin(),n.end(),name);require(i!=n.end(),"Missing body "+name);return unsigned(i-n.begin());}
Fixture fixture(const std::filesystem::path& dir,bool contact){
    require(std::filesystem::create_directory(dir),"Fixture directory exists");auto m=makeAmericanCrowObjectPushFixtureManifest(contact);
    CompiledRun base;check(compileRun(m,base),"Compile physical fixture");
    auto visual=authorAmericanCrowForagingVisual(base,dir,false,crowObjectPushCameraOrientation);
    auto p=authorAmericanCrowObjectPushRetinalProgram(base,visual);m.sensors.observation={};
    m.task=makeAmericanCrowObjectPushTaskPack(m.sensors.observation,m.reality.reset,&*p.retinal);
    m.sensors.deviceVisual=p;m.task.capacities=m.profile.capacities;

    unsigned privileged=0;for(const auto& o:m.sensors.observation.actorCurrent)require(o.source!=TaskObservationSource::objectTrack,"Object truth leaked to actor");
    for(const auto& o:m.sensors.observation.actorFrame)require(o.source!=TaskObservationSource::objectTrack,"Object truth leaked to actor history");
    for(const auto& o:m.sensors.observation.critic)privileged+=o.source==TaskObservationSource::objectTrack;
    require(privileged==7,"Critic requires exactly seven visibility/relative position/velocity channels");
    Fixture f;check(compileRun(m,f.run),"Compile retinal object task");f.visual=std::move(visual);
    require(f.run.task().layout().actorObservationSize==features && f.run.task().layout().criticObservationSize==criticFeatures,"Incorrect compiled actor/critic schema");
    require(base.world().fingerprint()==f.run.world().fingerprint(),"Sensor authoring changed physics");
    f.object=findBody(f.run,"crow_manipulandum");f.head=findBody(f.run,"crow_head");f.beak=findBody(f.run,"crow_lower_beak");
    const auto ids=f.run.world().sceneBodyIndices();const auto i=std::find(ids.begin(),ids.end(),f.object);require(i!=ids.end(),"Object is not scene-owned");f.local=unsigned(i-ids.begin());
    f.shape=MR_INVALID_INDEX;for(unsigned j=0;j<f.run.model().shapes.size();++j)if(f.run.model().shapes[j].bodyIndex==f.object)f.shape=j;
    require(f.shape!=MR_INVALID_INDEX,"Missing object shape");
    f.goalChannel=MR_INVALID_INDEX;for(const auto& o:f.run.task().outcomes())if(o.id=="object_goal_score")f.goalChannel=o.source-9;
    require(f.goalChannel<8,"Missing goal outcome channel");return f;
}
struct Evidence {
    checkpoint::Bytes world,retina,splitWorld,splitRetina;
    std::array<checkpoint::Digest,3> streams{},future{};
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    std::array<double,lanes> impulse{},motion{},goalDistance{};
    std::array<unsigned,lanes> contacts{},precontacts{};
    unsigned objectPixels=0;double rewardError=0;
};
double distance(mr_float4 a,mr_float4 b){const double x=a.x-b.x,y=a.y-b.y,z=a.z-b.z;return std::sqrt(x*x+y*y+z*z);}
Evidence execute(const Fixture& f,const std::filesystem::path& dir,unsigned arm,unsigned chunk,const Evidence* restore=nullptr){@autoreleasepool{
    require(std::filesystem::create_directory(dir),"Arm directory exists");const unsigned begin=restore?split:0;
    RetinalObserverContext retina(f.run,f.visual.family,f.visual.manifest,lanes,seed);
    MetalWorldStepConfig step;step.timestepSeconds=.02f;step.physicsSubsteps=f.run.profile().physicsSubsteps;step.velocityIterations=4;step.finalVelocityIterations=2;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=f.run.task();step.taskSeed=seed;step.minimumDifficultyBand=step.maximumDifficultyBand=0;
    step.ccdMode=MetalWorldCCDMode::hybrid;
    step.publishFinalState=true;step.captureContactEvidence=true;step.deviceObservationProgram=retina.deviceProgram();
    if(f.run.flappingWingProgram())step.flappingWingProgram=*f.run.flappingWingProgram();
    // Retinal execution always has native policy authority. These immutable
    // constant policies implement the declared three-phase motor control;
    // chunk boundaries only select a revision, never restage physical state.
    std::array<CompiledPolicyProgram,3> controls;
    for(unsigned phase=0;phase<3;++phase){PolicyPack p;p.id="object_push_motor_control";p.revision=phase+1;
        p.observationMean.assign(features,0);p.observationInverseStandardDeviation.assign(features,1);
        PolicyDenseLayer layer{.inputCount=features,.outputCount=18,.activation=PolicyActivation::identity,
            .weights=std::vector<float>(features*18,0),.bias=std::vector<float>(18,0)};
        if(phase==1)layer.bias[15]=arm==1?-.6f:arm==2?0:.6f;
        p.layers.push_back(std::move(layer));bindPolicyPack(p,f.run.task());
        check(compilePolicyProgram(p,f.run.task(),controls[phase]),"Compile native motor control");}
    std::vector<float> q,v;std::vector<MRBodyStateGPU> initial;
    for(unsigned e=0;e<lanes;++e){q.insert(q.end(),f.run.model().defaultQ.begin(),f.run.model().defaultQ.end());v.insert(v.end(),f.run.model().defaultV.begin(),f.run.model().defaultV.end());initial.insert(initial.end(),f.run.defaultSceneBodies().begin(),f.run.defaultSceneBodies().end());}
    MetalWorldConfig config;config.enableResidentCheckpoints=true;config.maximumInFlightSubmissions=1;
    MetalWorldContext world(config);MetalWorldResidentState resident;MetalWorldResult r;Evidence e;std::array<Stream,3> streams,future;
    std::vector<MRBodyStateGPU> trajectory;std::vector<float> fullq,fullv;
    std::ofstream trace(dir/"contact-reward.tsv");trace<<"tick\tlane\tobject_x\tobject_y\tobject_z\tgoal_score_per_second\tgoal_score_fp64\tbeak_contact_samples\tnormal_impulse_sum\n"<<std::setprecision(10);
    if(restore)retina.restoreState(restore->splitRetina);
    for(unsigned start=begin;start<ticks;start+=chunk){const auto count=std::min(chunk,ticks-start);std::vector<unsigned>resets(count*lanes,0);
        step.policyProgram=controls[start<32?0:start<96?1:2];
        MetalWorldBatch b{.environmentCount=lanes,.controlStepCount=count,.resetMasks=resets};MetalWorldSubmission ticket;
        if(start==begin){b.initialQ=q;b.initialV=v;b.initialSceneBodies=initial;
            if(restore){world.validateResidentCheckpoint(f.run.world(),b,step,restore->splitWorld);check(world.resumeResident(f.run.world(),b,step,restore->splitWorld,resident,ticket),"Resume object arm");}
            else check(world.initializeResidentState(f.run.world(),b,step,resident,ticket),"Initialize object arm");
        }else check(world.submitResident(f.run.world(),b,step,resident,ticket),"Submit object arm");
        const auto status=ticket.wait(r);check(status,"Complete object arm");require(!status.failedStepCount,"Physics rejected object arm");
        for(const auto& tr:r.transitions)require(!tr.termination.x,"Object arm terminated physically");
        require(r.actorObservations.size()==count*lanes*features && r.criticObservations.size()==count*lanes*criticFeatures && r.transitions.size()==count*lanes,"Incomplete native task streams");
        streams[0].add(r.actorObservations);streams[1].add(r.criticObservations);streams[2].add(r.transitions);
        if(start>=split){future[0].add(r.actorObservations);future[1].add(r.criticObservations);future[2].add(r.transitions);}
        if(start+count==split){e.splitWorld=world.checkpointResident(resident,step);e.splitRetina=retina.checkpointState();}
        if(!restore && start==0){const auto images=retina.inspectImages();for(const auto& image:images)for(const auto id:image.identities)if(id.w==f.shape)++e.objectPixels;}
        if(chunk==1){trajectory.insert(trajectory.end(),r.finalSceneBodies.begin(),r.finalSceneBodies.end());fullq.insert(fullq.end(),r.finalQ.begin(),r.finalQ.end());fullv.insert(fullv.end(),r.finalV.begin(),r.finalV.end());
            const auto stride=r.contactEvidence.contacts.size()/lanes,scenes=f.run.defaultSceneBodies().size();
            for(unsigned lane=0;lane<lanes;++lane){unsigned contacts=0;double impulse=0;
                for(unsigned c=0;c<r.contactStatuses[lane].activeContacts;++c){const auto& x=r.contactEvidence.contacts[lane*stride+c];const auto other=x.bodyA==f.object?x.bodyB:x.bodyB==f.object?x.bodyA:MR_INVALID_INDEX;
                    if(other==f.head||other==f.beak){++contacts;impulse+=std::abs(x.impulses.x);}}
                e.contacts[lane]+=contacts;e.impulse[lane]+=impulse;if(start<32)e.precontacts[lane]+=contacts;
                const auto& pos=r.finalSceneBodies[lane*scenes+f.local].position;
                const double d=distance(pos,{.19f,.25f,.257f,1});const double want=std::exp(-d*d/.01f);
                const auto& tr=r.transitions[lane];const auto channel=f.goalChannel;const auto values=channel<4?tr.outcomeChannels0:tr.outcomeChannels1;
                const double got=(&values.x)[channel%4]/double(step.timestepSeconds);
                e.rewardError=std::max(e.rewardError,std::abs(got-want));
                trace<<start<<'\t'<<lane<<'\t'<<pos.x<<'\t'<<pos.y<<'\t'<<pos.z<<'\t'<<got<<'\t'<<want<<'\t'<<contacts<<'\t'<<impulse<<'\n';
            }
        }
    }
    for(const auto error:retina.snapshot().errors)require(!error,"Retinal sticky error");
    e.world=world.checkpointResident(resident,step);e.retina=retina.checkpointState();e.q=r.finalQ;e.v=r.finalV;e.scene=r.finalSceneBodies;
    for(unsigned i=0;i<3;++i){e.streams[i]=streams[i].finish();e.future[i]=future[i].finish();}
    const auto scenes=f.run.defaultSceneBodies().size();for(unsigned lane=0;lane<lanes;++lane){const auto pos=e.scene[lane*scenes+f.local].position;e.motion[lane]=distance(pos,initial[lane*scenes+f.local].position);e.goalDistance[lane]=distance(pos,{.19f,.25f,.257f,1});}
    save(dir/"world.bin",e.world);save(dir/"retina.bin",e.retina);save(dir/"final-q.f32",e.q);save(dir/"final-v.f32",e.v);save(dir/"final-scene.bin",e.scene);
    if(!trajectory.empty()){save(dir/"scene-trajectory.bin",trajectory);save(dir/"q-trajectory.f32",fullq);save(dir/"v-trajectory.f32",fullv);}
    if(!e.splitWorld.empty()){save(dir/"tick48.world",e.splitWorld);save(dir/"tick48.retina",e.splitRetina);}
    std::ofstream report(dir/"summary.json");report<<std::setprecision(12)<<"{\"arm\":"<<arm<<",\"chunk\":"<<chunk<<",\"begin_tick\":"<<begin<<",\"accepted_transitions\":"<<(ticks-begin)*lanes<<",\"initial_object_pixels\":"<<e.objectPixels<<",\"maximum_goal_reward_fp64_error\":"<<e.rewardError<<",\"world_sha256\":\""<<hex(checkpoint::digest(e.world))<<"\",\"retina_sha256\":\""<<hex(checkpoint::digest(e.retina))<<"\",\"streams\":[";
    for(unsigned i=0;i<3;++i){if(i)report<<',';report<<'"'<<hex(e.streams[i])<<'"';}report<<"],\"future_streams\":[";
    for(unsigned i=0;i<3;++i){if(i)report<<',';report<<'"'<<hex(e.future[i])<<'"';}report<<"],\"lanes\":[";
    for(unsigned lane=0;lane<lanes;++lane){if(lane)report<<',';report<<"{\"motion_m\":"<<e.motion[lane]<<",\"goal_distance_m\":"<<e.goalDistance[lane]<<",\"beak_contact_samples\":"<<e.contacts[lane]<<",\"pre_pulse_contacts\":"<<e.precontacts[lane]<<",\"normal_impulse_sum\":"<<e.impulse[lane]<<'}';}report<<"],\"physics_rejections\":0,\"terminations\":0}\n";
    return e;
}}
void exactEnd(const Evidence&a,const Evidence&b){require(a.world==b.world && a.retina==b.retina && a.q==b.q && a.v==b.v && a.scene.size()==b.scene.size() && !std::memcmp(a.scene.data(),b.scene.data(),a.scene.size()*sizeof(MRBodyStateGPU)),"Final native world/retina/state differs");}
}
int main(int argc,char**argv){@autoreleasepool{std::filesystem::path out;bool owns=false;try{
    require(argc==2,"Usage: metalrobo_crow_object_push_check NEW_OUTPUT");out=std::filesystem::absolute(argv[1]);require(std::filesystem::create_directory(out),"Output exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"purpose":"native object-push contact/reward/sensor/partition/continuation admission","lanes":8,"physics_substeps":32,"solver_mode":"hybrid CCD","ticks":160,"split_tick":48,"arms":["positive yaw0.6 ticks32..95","negative yaw-0.6","neutral","positive without crow/object contact","positive chunk8","positive restored from tick48 into fresh world+retina"],"evidence":"streaming actor/critic/transition SHA256, bounded accepted scene/q/v traces for chunk1, final state and world+retina checkpoints; no video","requirements":"positive solved beak impulses and improved goal distance; neutral/opposite/contact-disabled miss; initial object visible through RGB; reward agrees with independent FP64 formula; partition/future continuation exact","scope":"estimated engineering task only, no learned manipulation or biological calibration"})";
    const auto normal=fixture(out/"normal-fixture",true),disabled=fixture(out/"no-crow-contact-fixture",false);
    require(normal.run.world().fingerprint()!=disabled.run.world().fingerprint(),"Contact control did not change physical identity");
    const auto positive=execute(normal,out/"positive-chunk1",0,1);
    const auto negative=execute(normal,out/"negative",1,1);
    const auto neutral=execute(normal,out/"neutral",2,1);
    const auto noContact=execute(disabled,out/"no-crow-contact",3,1);
    const auto partition=execute(normal,out/"positive-chunk8",0,8);
    const auto continuation=execute(normal,out/"restored-tick48",0,8,&positive);
    exactEnd(positive,partition);require(positive.streams==partition.streams,"Task streams depend on chunk partition");
    exactEnd(positive,continuation);require(positive.future==continuation.streams,"Restored future task streams differ");
    require(positive.objectPixels>0 && neutral.objectPixels==positive.objectPixels,"Initial physical object invisible/inconsistent in RGB renderer");
    const auto start=normal.run.defaultSceneBodies()[normal.local].position;const double initialDistance=distance(start,{.19f,.25f,.257f,1});
    for(unsigned lane=0;lane<lanes;++lane){require(!positive.precontacts[lane],"Object already contacts beak before action pulse");
        require(positive.impulse[lane]>0 && positive.motion[lane]>.01 && positive.goalDistance[lane]<initialDistance*.5,"Positive arm failed solved-contact/progress criterion");
        const auto pos=positive.scene[lane*normal.run.defaultSceneBodies().size()+normal.local].position;
        require(std::abs(pos.z-.257)<.0005 && pos.x>.16 && pos.x<.80 && std::abs(pos.y)<.50,"Pushed object left its support");
        for(const auto* miss:{&negative,&neutral,&noContact}) {
            const auto p=miss->scene[lane*normal.run.defaultSceneBodies().size()+normal.local].position;
            require(std::abs(p.x-start.x)<1e-5 && std::abs(p.y-start.y)<1e-5 && std::abs(p.z-start.z)<.0005 && !miss->contacts[lane] && miss->impulse[lane]==0,"Miss/contact-disabled control displaced object horizontally, exceeded settling tolerance or contacted beak");
        }
    }
    for(const auto* e:{&positive,&negative,&neutral,&noContact})require(e->rewardError<2e-6,"Goal reward differs from accepted position FP64 reference");
    std::ofstream(out/"summary.json")<<"{\"status\":\"passed_engineering_controls\",\"accepted_transitions\":7296,\"physical_contact_and_miss_controls\":true,\"actor_has_object_truth\":false,\"critic_object_channels\":7,\"goal_reward_fp64_agrees\":true,\"world_retina_stream_partition_exact\":true,\"world_retina_future_restore_exact\":true,\"physics_rejections\":0,\"terminations\":0,\"learned_manipulation\":false,\"biological_calibration\":false}\n";
    std::cout<<"Native object-push engineering controls passed\n";return 0;
}catch(const std::exception&e){if(owns)std::ofstream(out/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
