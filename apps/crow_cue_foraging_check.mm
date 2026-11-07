#import <Foundation/Foundation.h>
#include "CrowCameraFixture.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/RetinalObserver.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/LearningPacks.hpp"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
using namespace crowCameraFixture;
namespace {
constexpr unsigned lanes=8,features=1251;
bool memoryChoice=false;
template<class T> void save(const std::filesystem::path& p,const std::vector<T>& a) {
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(bool(f),"Evidence write failed");
}
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(T));
}
mr_float4 rotate(mr_float4 q,mr_float4 v) {
    const mr_float4 t{2*(q.y*v.z-q.z*v.y),2*(q.z*v.x-q.x*v.z),2*(q.x*v.y-q.y*v.x),0};
    return {v.x+q.w*t.x+q.y*t.z-q.z*t.y,v.y+q.w*t.y+q.z*t.x-q.x*t.z,v.z+q.w*t.z+q.x*t.y-q.y*t.x,0};
}
RunManifest manifest(bool blocked) {
    auto m=makeManifest(blocked);m.profile.environmentCount=lanes;m.profile.controlSteps=320;
    applyAmericanCrowForagingFixtureLimits(m);m.reality.reset={};return m;
}
MetalWorldStepConfig stepConfig(const CompiledRun& run) {
    MetalWorldStepConfig s;s.timestepSeconds=.02f;s.physicsSubsteps=4;s.velocityIterations=4;s.finalVelocityIterations=2;
    s.solverMode=MetalWorldSolverMode::temporalCone;s.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    s.taskProgram=run.task();s.taskSeed=2651109803u;s.minimumDifficultyBand=s.maximumDifficultyBand=0;
    s.publishFinalState=true;s.publishStateTrajectory=false;
    if (run.flappingWingProgram()) s.flappingWingProgram=*run.flappingWingProgram();return s;
}
struct Initial {
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    explicit Initial(const CompiledRun& run) {
        for (unsigned e=0;e<lanes;++e) {
            q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
            scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());
        }
    }
};
void healthy(const MetalWorldDiagnostics& status,const MetalWorldResult& r) {
    check(status,"native accepted run");require(!status.failedStepCount,"Rejected native physics step");
    for (const auto& t:r.transitions) require(!t.termination.x,"Physical task termination");
}
std::array<mr_float4,2> calibrate(const std::filesystem::path& output) {
    std::filesystem::create_directory(output);
    std::ofstream(output/"plan.json")<<R"({"purpose":"native fixture geometry calibration before food-port admission","prediction":"canonical supported crow reaches two distinct stable mouth locations with yaw commands +0.6 and -0.6; calibrate only centers, keep intake radius0.015m fixed; no food consumption, rejected steps or terminations","budget":"one128tick native run, eight lanes alternating commanded side, no parameter search or retries; accepted q/v composed into body poses through native owner","limits":"head-reachable two-port fixture, not free locomotion, biological feeding or a learned controller"})";
    auto m=manifest(false);CompiledRun run;check(compileRun(m,run),"compile calibration");Initial init(run);
    FiniteResourceContext food(makeAmericanCrowFeedingResourceConfig(run,lanes));auto step=stepConfig(run);step.devicePhysicsProgram=food.deviceProgram();
    std::vector<float> actions(128*lanes*18,0);std::vector<unsigned> resets(128*lanes,0);
    for (unsigned t=0;t<128;++t) for (unsigned e=0;e<lanes;++e) actions[(t*lanes+e)*18+15]=(e%2 ? -.6f : .6f);
    MetalWorldBatch batch{.environmentCount=lanes,.controlStepCount=128,.initialQ=init.q,.initialV=init.v,.actions=actions,.resetMasks=resets,.initialSceneBodies=init.scene};
    MetalWorldContext world;MetalWorldResidentState resident;MetalWorldSubmission ticket;
    check(world.initializeResidentState(run.world(),batch,step,resident,ticket),"initialize calibration");MetalWorldResult r;healthy(ticket.wait(r),r);
    save(output/"final-q.f32",r.finalQ);save(output/"final-v.f32",r.finalV);save(output/"final-scene.bin",r.finalSceneBodies);
    std::vector<MRBodyStateGPU> bodies;std::string reason;
    require(composeVisualBodyStates(run.model(),lanes,r.finalQ,r.finalV,r.finalSceneBodies,bodies,&reason),reason);save(output/"accepted-bodies.bin",bodies);
    std::array<mr_float4,2> ports{};
    for (unsigned e=0;e<2;++e) {
        const auto& head=bodies[e*run.model().bodies.size()+15];const auto& ground=bodies[e*run.model().bodies.size()+17];
        auto mouth=rotate(head.orientation,{.065f,0,0,0});mouth.x+=head.position.x-ground.position.x;mouth.y+=head.position.y-ground.position.y;mouth.z+=head.position.z-ground.position.z;
        const auto q=ground.orientation;ports[e]=rotate({-q.x,-q.y,-q.z,q.w},mouth);ports[e].w=.015f;
    }
    const auto a=ports[0],b=ports[1];const float separation=std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
    save(output/"ports.bin",std::vector<mr_float4>(ports.begin(),ports.end()));
    std::ofstream report(output/"ports.json");report<<std::setprecision(10)<<"{\"ground_body\":17,\"mouth_body\":15,\"mouth_local_point\":[0.065,0,0],\"centers_radius_m\":[";
    for (unsigned i=0;i<2;++i) {if(i)report<<',';const auto p=ports[i];report<<'['<<p.x<<','<<p.y<<','<<p.z<<','<<p.w<<']';}
    report<<"],\"separation_m\":"<<separation<<",\"source\":\"accepted native canonical yaw plus/minus0.6 at tick128\"}\n";report.flush();
    require(separation>.060f,"Calibrated food ports are not disjoint at fixed15mm intake radius");
    for (const auto& s:food.snapshot()) require(!s.accounting.y,"Calibration consumed food");return ports;
}
struct Evidence {
    std::vector<float> actor,q,v;std::vector<MRBodyStateGPU> scene;std::vector<MRTaskTransitionGPU> transitions;
    std::vector<MRFiniteResourceStateGPU> food;std::vector<std::byte> retinal;
    std::array<std::array<unsigned,lanes>,4> cuePixels{};
};
CompiledPolicyProgram fixedPolicy(const CompiledRun& run,float yaw,float jaw,unsigned revision,const std::filesystem::path& output) {
    PolicyPack policy;policy.id="authored_constant_port_controller_no_hidden_labels";policy.revision=revision;
    policy.observationMean.assign(features,0);policy.observationInverseStandardDeviation.assign(features,1);
    PolicyDenseLayer layer{.inputCount=features,.outputCount=18,.activation=PolicyActivation::identity,
        .weights=std::vector<float>(features*18,0),.bias=std::vector<float>(18,0)};
    layer.bias[15]=yaw;layer.bias[17]=jaw;policy.layers.push_back(std::move(layer));bindPolicyPack(policy,run.task());
    CompiledPolicyProgram compiled;check(compilePolicyProgram(policy,run.task(),compiled),"compile authored port controller");
    check(writePolicyPack(policy,output/("controller-revision"+std::to_string(revision)+".policypack")),"save native controller");return compiled;
}
Evidence execute(const std::array<mr_float4,2>& ports,const std::filesystem::path& output,unsigned ticks,unsigned chunk,
    std::uint64_t seed,bool blocked=false,int motor=0,bool reset=false,bool rawImages=false) {@autoreleasepool {
    require(std::filesystem::create_directory(output),"Trial output exists");auto m=manifest(blocked);CompiledRun base;check(compileRun(m,base),"compile cue fixture");
    const auto visual=makeVisual(base,output,true);
    const auto authored=authorAmericanCrowCuedForagingProgram(base,visual,ports,lanes,seed);
    const auto& resources=authored.resources;const auto& p=authored.sensor;m.sensors.observation={};m.reality.reset={};
    m.task=makeAmericanCrowCuedForagingTaskPack(m.sensors.observation,m.reality.reset,authored);m.task.capacities=m.profile.capacities;m.task.maximumEpisodeSteps=100000;m.reality.reset={};
    if(memoryChoice)applyAmericanCrowMemoryChoiceCurriculum(m.task,64);
    m.sensors.deviceVisual=p;CompiledRun run;check(compileRun(m,run),"compile cue retinal task");require(base.world().fingerprint()==run.world().fingerprint(),"Cue authoring changed physical world");
    FiniteResourceContext food(resources);
    if(rawImages && memoryChoice) {
        require(writeTaskPack(m.task,output/"memory-choice.taskpack").succeeded(),"Write gated TaskPack");
        TaskPack decoded;require(readTaskPack(output/"memory-choice.taskpack",decoded).succeeded(),"Read gated TaskPack");
        auto roundtrip=m;roundtrip.task=decoded;CompiledRun restored;check(compileRun(roundtrip,restored),"Compile gated TaskPack roundtrip");
        require(run.task().fingerprint()==restored.task().fingerprint() &&
            run.task().actionFingerprint()==restored.task().actionFingerprint() &&
            run.task().observationFingerprint()==restored.task().observationFingerprint(),"Gated TaskPack changed identity on roundtrip");
        std::ofstream controls(output/"memory-choice-negative-controls.jsonl");
        for(unsigned test=0;test<7;++test) {
            auto bad=m;
            if(test==0)bad.task.actionAvailableAfterSteps=0;
            if(test==1)bad.task.allowedActionMask=0;
            if(test==2)bad.task.actionAvailableAfterSteps=bad.task.maximumEpisodeSteps;
            if(test==3)bad.task.allowedActionMask=1u<<31u;
            if(test==4)bad.task.deviceStateProgramFingerprint=0;
            if(test==5)bad.sensors.observation.retinalFeatureCount=0;
            if(test==6)bad.task.actions.resize(33,bad.task.actions.front());
            CompiledRun invalid;const auto status=compileRun(bad,invalid);require(!status.succeeded(),"Malformed memory choice admitted");
            controls<<"{\"case\":"<<test<<",\"rejected\":true,\"reason\":"<<std::quoted(status.message)<<"}\n";
        }
    }
    if(rawImages) {
        std::ofstream authoringControls(output/"authoring-negative-controls.jsonl");
        for(unsigned test=0;test<11;++test) {
            auto badPorts=ports;auto badVisual=visual;unsigned badEnvironments=lanes;
            if(test==0)badEnvironments=0;
            if(test==1)badPorts[0].x=std::numeric_limits<float>::quiet_NaN();
            if(test==2)badPorts[0].w=0;
            if(test==3)badPorts[1]=badPorts[0];
            if(test==4)badVisual.resourceCues.clear();
            if(test==5)badVisual.resourceCues[1].binding.x=badVisual.resourceCues[0].binding.x;
            if(test==6)badVisual.resourceCues[0].binding.w=1;
            if(test==7)++badVisual.manifest.fingerprint;
            if(test==8)badVisual.head=MR_INVALID_INDEX;
            bool rejected=false;std::string reason;
            try {
                if(test==9) {CompiledRun invalid;const auto unused=authorAmericanCrowCuedForagingProgram(invalid,badVisual,badPorts,badEnvironments,seed);(void)unused;}
                else if(test==10) {const auto unused=makeVisual(base,output,true);(void)unused;}
                else {const auto unused=authorAmericanCrowCuedForagingProgram(base,badVisual,badPorts,badEnvironments,seed);(void)unused;}
            }catch(const std::invalid_argument& error){rejected=true;reason=error.what();}
            require(rejected,"Malformed shared authoring program admitted");
            authoringControls<<"{\"case\":"<<test<<",\"rejected\":true,\"reason\":"<<std::quoted(reason)<<"}\n";
        }
        std::ofstream controls(output/"negative-controls.jsonl");
        for(unsigned test=0;test<6;++test){auto bad=m;auto& cue=bad.sensors.deviceVisual->retinal->resourceCues;
            if(test==0)cue.resize(9,cue.front());if(test==1)cue[1].binding.x=cue[0].binding.x;
            if(test==2)cue[0].binding.y=8;if(test==3)cue[0].binding.z=0;
            if(test==4)cue[0].binding.w=1;if(test==5)cue[0].binding.x=MR_INVALID_INDEX;
            bad.sensors.deviceVisual->fingerprint=visualSensorProgramFingerprint(*bad.sensors.deviceVisual);
            CompiledRun invalid;const auto status=compileRun(bad,invalid);require(!status.succeeded(),"Malformed cue compiler program admitted");
            controls<<"{\"case\":"<<test<<",\"rejected\":true,\"reason\":"<<std::quoted(status.message)<<"}\n";
        }
        auto changed=p;changed.retinal->resourceCues[0].binding.z=31;
        require(visualSensorProgramFingerprint(changed)!=p.fingerprint,"Changing visual cue timing did not bind a new sensor fingerprint");
        for(unsigned test=0;test<3;++test){bool rejected=false;std::string reason;
            try{
                if(test==0){RetinalObserverContext invalid(run,visual.family,visual.manifest,lanes,2651109803u);}
                else{auto bad=resources;if(test==1)++bad.worldFingerprint;else bad.allocation.concealProximity=0;
                    FiniteResourceContext wrong(bad);RetinalObserverContext invalid(run,visual.family,visual.manifest,lanes,2651109803u,&wrong);}
            }catch(const std::invalid_argument& error){rejected=true;reason=error.what();}
            require(rejected,"Mismatched cue resource owner admitted");controls<<"{\"case\":"<<test+6<<",\"rejected\":true,\"reason\":"<<std::quoted(reason)<<"}\n";
        }
    }
    RetinalObserverContext retina(run,visual.family,visual.manifest,lanes,2651109803u,&food);
    auto step=stepConfig(run);step.devicePhysicsProgram=food.deviceProgram();step.deviceObservationProgram=retina.deviceProgram();Initial init(run);
    MetalWorldContext world;MetalWorldResidentState resident;MetalWorldResult r;Evidence e;
    unsigned revision=1;float lastYaw=0,lastJaw=0;step.policyProgram=fixedPolicy(run,0,0,revision,output);
    for (unsigned start=0;start<ticks;start+=chunk) {
        const auto count=std::min(chunk,ticks-start);std::vector<unsigned> resets(count*lanes,0);
        // All lanes receive the same native policy. Food allocation is never
        // consulted to author actions. The opposite-side run is the paired miss.
        const float yaw=motor && start>=64 ? (motor==1 ? .6f : -.6f) : 0.0f;
        const float jaw=motor && ((start>=192 && start<224)||(start>=256 && start<288)) ? .9f : 0.0f;
        if(yaw!=lastYaw || jaw!=lastJaw){step.policyProgram=fixedPolicy(run,yaw,jaw,++revision,output);lastYaw=yaw;lastJaw=jaw;}
        for(unsigned t=0;t<count;++t)for(unsigned lane=0;lane<lanes;++lane){const auto time=start+t;
            if(reset && time==48)resets[t*lanes+lane]=1;
        }
        MetalWorldBatch batch{.environmentCount=lanes,.controlStepCount=count,.resetMasks=resets};MetalWorldSubmission ticket;
        if(!start){batch.initialQ=init.q;batch.initialV=init.v;batch.initialSceneBodies=init.scene;check(world.initializeResidentState(run.world(),batch,step,resident,ticket),"initialize cue world");}
        else check(world.submitResident(run.world(),batch,step,resident,ticket),"submit cue world");healthy(ticket.wait(r),r);
        e.actor.insert(e.actor.end(),r.actorObservations.begin(),r.actorObservations.end());e.transitions.insert(e.transitions.end(),r.transitions.begin(),r.transitions.end());
        const unsigned end=start+count;const std::array<unsigned,4> samples{16,48,64,80};
        for(unsigned k=0;k<samples.size();++k)if(end==samples[k]){
            const auto images=retina.inspectImages();
            for(unsigned eye=0;eye<2;++eye){
                if(rawImages && k<2)save(output/("image-tick"+std::to_string(end)+"-eye"+std::to_string(eye)+".rgba.f32"),images[eye].rgb);
                for(unsigned lane=0;lane<lanes;++lane)for(unsigned pixel=0;pixel<W*H;++pixel){
                    const auto id=images[eye].identities[lane*W*H+pixel];
                    if(id.w==1000 || id.w==1001)++e.cuePixels[k][lane];
                }
            }
        }
    }
    e.q=r.finalQ;e.v=r.finalV;e.scene=r.finalSceneBodies;e.food=food.snapshot();e.retinal=retina.checkpointState();
    save(output/"actor.f32",e.actor);save(output/"final-q.f32",e.q);save(output/"final-v.f32",e.v);save(output/"final-scene.bin",e.scene);save(output/"food.bin",e.food);save(output/"retina.bin",e.retinal);save(output/"transitions.bin",e.transitions);
    require(e.actor.size()==std::size_t(ticks)*lanes*features,"Cue actor stream shape differs");
    std::ofstream report(output/"cue-pixels.json");report<<"{\"ticks\":[16,48,64,80],\"pixels_across_two_eyes\":[";
    for(unsigned k=0;k<4;++k){if(k)report<<',';report<<'[';for(unsigned lane=0;lane<lanes;++lane){if(lane)report<<',';report<<e.cuePixels[k][lane];}report<<']';}report<<"]}\n";
    for(auto error:retina.snapshot().errors)require(!error,"Sticky retinal sensor error");
    return e;
}}
}
int main(int argc,char** argv){@autoreleasepool{bool owns=false;try{
    require(argc==2 || (argc==3 && std::string(argv[2])=="--memory-choice"),"Usage: metalrobo_crow_cue_foraging_check NEW_OUTPUT [--memory-choice]");
    memoryChoice=argc==3;const std::filesystem::path output=argv[1];require(std::filesystem::create_directory(output),"Output exists");owns=true;
    if(memoryChoice)std::ofstream(output/"memory-choice-plan.json")<<R"({"purpose":"native memory-choice motor/history and reward-path admission","decision_tick":64,"neutral_actions_before_decision":true,"allowed_action_indices_after_decision":[15,17],"cue_end_tick":32,"retinal_flush_tick":40,"intake_start_tick":192,"ordinary_reset":"restarts episode motor hold but never refills/rerolls resources or reveals cue","prediction":"same frozen constant-bias phase controllers still get40mg at matching physical ports and0 at opposite ports; TaskPack roundtrip preserves exact identity; seven malformed programs reject; chunk1/16 and8/32 remain exact","budget":"one calibration and eight native arms, no retries","scope":"staged supported head/beak engineering task, no learned memory or free locomotion"})";
    const auto ports=calibrate(output/"calibration");
    std::ofstream(output/"plan.json")<<R"({"purpose":"integrated native two-port foraging and disappearing visual cue admission, no learned memory claim","units":"eight supported estimated crow lanes; physical port centers calibrated once before this admission, fixed15mm radius; no subsequent parameter search","cue":"one of two authored noncolliding indicator meshes at a distant static fixture represents its selected food port; normal native projection, material and occlusion; visible only at initialized accepted resource lifetime ticks below32; food intake forbidden through first192 ticks","actor":"97 ordinary features plus two delayed16x12 RGB streams and validity/age; no hidden food location/seed/clock label","revision":"Attempt01 reached two disjoint mouth positions, then rejected the unsupported open-loop retinal execution. Preserve it. Bind actual native constant-bias policies, with higher revisions only at fixed yaw/jaw phase boundaries. Left/right arms each use the same actions in every lane, never hidden allocation labels. No physics bounds or site radius changed.","prediction":"both ports consume exactly40000ug at its matching side under the same native constant-bias policy for all lanes; the opposite-side native policy consumes0 in those lanes; neutral cue is visible at tick16 and absent at48/64/80; physical wall occludes cue; five neutral80tick arms cover chunk1/16 exactness, paired hidden allocation seed, occlusion and reset at48; all97 non-image features byte-exact under changed hidden allocation, full actors byte-exact after tick40 while some early image values differ; episode reset cannot reveal cue or reroll food; left-side320tick chunk8/32 final states and streams exact","seeds":[2651109801,2651109802],"budget":"one calibration plus eight fixed native arms, no retry or threshold search; retain original failures, abort first invalidity","scope":"scripted intake controls and native sensing mechanism; no autonomous search, trained memory, solid food, biological fidelity, complete life recovery or production throughput qualification"})";
    const auto one=execute(ports,output/"visible-chunk1",80,1,2651109801,false,0,false,true);
    const auto many=execute(ports,output/"visible-chunk16",80,16,2651109801);
    require(exact(one.actor,many.actor)&&exact(one.q,many.q)&&exact(one.v,many.v)&&exact(one.food,many.food)&&exact(one.retinal,many.retinal)&&exact(one.transitions,many.transitions),"Native cue state depends on chunk partition");
    const auto paired=execute(ports,output/"paired-seed",80,16,2651109802);
    const auto blocked=execute(ports,output/"occluded",80,16,2651109801,true);
    const auto reset=execute(ports,output/"post-cue-reset",80,16,2651109801,false,0,true);
    unsigned changed=0,earlyDifferences=0;
    for(unsigned lane=0;lane<lanes;++lane){
        require(one.cuePixels[0][lane]>0 && !blocked.cuePixels[0][lane],"Native rendered cue absent or physical occluder failed");
        for(unsigned k=1;k<4;++k)require(!one.cuePixels[k][lane]&&!paired.cuePixels[k][lane]&&!reset.cuePixels[k][lane],"Cue remained visible or reset its lifetime");
        require(one.food[lane].clock.x==80 && reset.food[lane].clock.x==80 && one.food[lane].clock.w==reset.food[lane].clock.w,"Episode reset reset lifetime or rerolled food");
        if(one.food[lane].clock.w!=paired.food[lane].clock.w)++changed;
    }
    for(unsigned t=0;t<80;++t)for(unsigned lane=0;lane<lanes;++lane){const auto first=(t*lanes+lane)*features;
        require(!std::memcmp(one.actor.data()+first,paired.actor.data()+first,97*sizeof(float)),"Hidden food state leaked into non-image actor features");
        if(t>=40)require(!std::memcmp(one.actor.data()+first,paired.actor.data()+first,features*sizeof(float)),"Expired cue still exposes hidden location after retinal flush");
        else if(std::memcmp(one.actor.data()+first+97,paired.actor.data()+first+97,1152*sizeof(float)))++earlyDifferences;
    }
    require(changed && earlyDifferences,"Paired resource assignment did not change a real camera cue");
    const auto correct=execute(ports,output/"left-port",320,32,2651109801,false,1);
    const auto wrong=execute(ports,output/"right-port",320,32,2651109801,false,2);
    const auto partition=execute(ports,output/"left-partition",320,8,2651109801,false,1);
    require(exact(correct.actor,partition.actor)&&exact(correct.q,partition.q)&&exact(correct.v,partition.v)&&exact(correct.food,partition.food)&&exact(correct.retinal,partition.retinal),"Integrated feeding depends on chunk partition");
    unsigned left=0,right=0;
    for(unsigned lane=0;lane<lanes;++lane){const auto& a=correct.food[lane];const auto& b=wrong.food[lane];
        const unsigned expectedLeft=a.clock.w==1 ? 40000u : 0u,expectedRight=a.clock.w==2 ? 40000u : 0u;
        require(a.accounting.y==expectedLeft && a.accounting.x==expectedLeft && a.accounting.z==(expectedLeft ? 2u : 0u) &&
            b.accounting.y==expectedRight && b.accounting.x==expectedRight && b.accounting.z==(expectedRight ? 2u : 0u) &&
            a.clock.x==320 && b.clock.w==a.clock.w,"Matched/opposite physical port intake disagrees");
        if(a.clock.w==1)++left;else if(a.clock.w==2)++right;else require(false,"Invalid allocation mask");
    }require(left&&right,"Native reachability did not cover both ports");
    std::ofstream report(output/"summary.json");if(memoryChoice)report<<"{\"memory_choice_decision_tick\":64,";else report<<'{';report<<"\"status\":\"passed\",\"lanes\":8,\"native_arms\":8,\"cue_active_until_lifetime_tick\":32,\"intake_delayed_ticks\":192,\"left_port_lanes\":"<<left<<",\"right_port_lanes\":"<<right<<",\"correct_consumed_ug_per_lane\":40000,\"wrong_consumed_ug_per_lane\":0,\"paired_changed_lanes\":"<<changed<<",\"early_image_differences\":"<<earlyDifferences<<",\"non_image_actor_features_byte_exact\":true,\"post_cue_actor_byte_exact\":true,\"native_partition_byte_exact\":true,\"physical_terminations\":0,\"failed_physics_steps\":0,\"learned_memory_qualified\":false}\n";report.close();std::ifstream result(output/"summary.json");std::cout<<result.rdbuf();return 0;
}catch(const std::exception& error){if(owns)std::ofstream(std::filesystem::path(argv[1])/"failure.json")<<"{\"error\":"<<std::quoted(error.what())<<"}\n";std::cerr<<error.what()<<'\n';return 1;}}}
