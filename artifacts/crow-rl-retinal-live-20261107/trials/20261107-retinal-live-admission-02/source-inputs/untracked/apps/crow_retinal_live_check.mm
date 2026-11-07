#import <Foundation/Foundation.h>
#include "CrowCameraFixture.hpp"
#include "metalrobo/RetinalObserver.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <sstream>
using namespace crowCameraFixture;
namespace {
constexpr unsigned steps=32,features=1251,first=97,valid=1249,age=1250;
constexpr unsigned red=first+(6*16+8)*3,blue=red+2;
template<class T> void save(const std::filesystem::path& p,const std::vector<T>& data) {
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));require(bool(f),"Evidence write failed");
}
struct Evidence {std::vector<float> actor,actions,latents,q,v;std::vector<MRTaskTransitionGPU> transitions;};
Evidence execute(bool blocked,bool connected,unsigned chunk,const std::filesystem::path& out) {
    require(std::filesystem::create_directory(out),"Run directory exists");
    auto m=makeManifest(blocked);CompiledRun base;check(compileRun(m,base),"compile physical camera fixture");
    const auto visual=makeVisual(base,out);
    VisualSensorProgram p;p.cameraParentBody="crow_head";p.cameraPosition={.018f,.026f,.012f,0};p.cameraOrientation={-.5f,.5f,-.5f,.5f};
    p.width=W;p.height=H;p.verticalFieldOfViewDegrees=45;p.nominalRateHz=15;p.minimumVisiblePixels=1;
    p.retinal=RetinalSensorProgram{};
    for (const auto& pack:visual.manifest.renderScene.visualPacks) p.assets.push_back({pack.packPath.string(),
        visual.family.worldTemplate.assets[pack.assetIndex].id,pack.contentHash,pack.semanticId,pack.instanceId});
    p.fingerprint=visualSensorProgramFingerprint(p);m.sensors.observation={};m.reality.reset={};m.task=makeAmericanCrowRetinalFeedingTaskPack(m.sensors.observation,m.reality.reset,*p.retinal);
    m.sensors.deviceVisual=p;CompiledRun run;check(compileRun(m,run),"compile actual retinal run");
    require(base.world().fingerprint()==run.world().fingerprint(),"Sensor changed physical world");
    RetinalObserverContext retina(run,visual.family,visual.manifest,N,2651108701u);
    PolicyPack policy;policy.id=connected ? "authored_pixel_red_minus_blue_actor" : "disconnected_pixel_control";
    policy.observationMean.assign(features,0);policy.observationInverseStandardDeviation.assign(features,1);
    PolicyDenseLayer layer{.inputCount=features,.outputCount=18,.activation=PolicyActivation::identity,
        .weights=std::vector<float>(features*18,0),.bias=std::vector<float>(18,0)};
    if (connected) {layer.weights[15*features+red]=2;layer.weights[15*features+blue]=-2;}
    policy.layers.push_back(std::move(layer));bindPolicyPack(policy,run.task());CompiledPolicyProgram compiled;
    check(compilePolicyProgram(policy,run.task(),compiled),"compile pixel actor");
    check(writePolicyPack(policy,out/"authored-actor.policypack"),"save authored pixel actor");
    MetalWorldContext world;MetalWorldResidentState resident;FiniteResourceContext food(makeAmericanCrowFeedingResourceConfig(run,N));
    MetalWorldStepConfig step;step.timestepSeconds=.02f;step.physicsSubsteps=4;step.velocityIterations=4;step.finalVelocityIterations=2;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=run.task();step.policyProgram=compiled;step.deviceObservationProgram=retina.deviceProgram();
    step.devicePhysicsProgram=food.deviceProgram();step.taskSeed=2651108701u;step.minimumDifficultyBand=step.maximumDifficultyBand=0;
    step.publishStateTrajectory=true;step.publishFinalState=true;
    if (run.flappingWingProgram()) step.flappingWingProgram=*run.flappingWingProgram();
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    for (unsigned env=0;env<N;++env) {q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());
        v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
        scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());}
    Evidence e;std::vector<float> trajectory;MetalWorldResult result;
    for (unsigned start=0;start<steps;start+=chunk) {
        const unsigned count=std::min(chunk,steps-start);std::vector<unsigned> resets(count*N,0);
        MetalWorldBatch batch{.environmentCount=N,.controlStepCount=count,.resetMasks=resets};
        MetalWorldSubmission ticket;
        if (!start) {batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;
            check(world.initializeResidentState(run.world(),batch,step,resident,ticket),"initialize live retinal world");}
        else check(world.submitResident(run.world(),batch,step,resident,ticket),"submit live retinal world");
        const auto status=ticket.wait(result);check(status,"accept live retinal world");
        require(!status.failedStepCount,"Live retinal physics rejected a step");
        e.actor.insert(e.actor.end(),result.actorObservations.begin(),result.actorObservations.end());
        e.actions.insert(e.actions.end(),result.policyActions.begin(),result.policyActions.end());
        e.latents.insert(e.latents.end(),result.policyLatents.begin(),result.policyLatents.end());
        e.transitions.insert(e.transitions.end(),result.transitions.begin(),result.transitions.end());
        trajectory.insert(trajectory.end(),result.observations.begin(),result.observations.end());
    }
    e.q=result.finalQ;e.v=result.finalV;const auto sensor=retina.snapshot();
    save(out/"actor.f32",e.actor);save(out/"actions.f32",e.actions);save(out/"latents.f32",e.latents);
    save(out/"transitions.bin",e.transitions);save(out/"trajectory.f32",trajectory);save(out/"final-q.f32",e.q);save(out/"final-v.f32",e.v);
    save(out/"retinal-state.bin",sensor.states);save(out/"retinal-errors.u32",sensor.errors);
    require(e.actor.size()==steps*N*features && e.actions.size()==steps*N*18 && e.latents.size()==steps*N*18,"Native observation/action publication dimensions differ");
    for (auto error:sensor.errors) require(error==0,"Retinal GPU reported a sticky sensor error");
    for (const auto& t:e.transitions) require(t.termination.x==0,"Live retinal control terminated");
    double maximum=0;
    for (unsigned t=0;t<steps;++t) for (unsigned env=0;env<N;++env) {
        const auto a=(t*N+env)*features,b=(t*N+env)*18;
        require(e.actor[a+valid]==(t<2 ? 0 : 1),"Retinal cold-start latency was not two control ticks");
        unsigned capture=0;
        if (t>=2) for (unsigned tick=1;tick<=t-2;++tick) if ((tick*3)/10>((tick-1)*3)/10) capture=tick;
        require(std::abs(e.actor[a+age]-(t<2 ? 0.0f : (t-capture)*.02f))<2e-6,"Live retinal age differs from integer timing oracle");
        const float expected=connected ? 2*(e.actor[a+red]-e.actor[a+blue]) : 0;
        maximum=std::max(maximum,double(std::abs(expected-e.latents[b+15])));
        require(std::abs(std::clamp(expected,-1.0f,1.0f)-e.actions[b+15])<2e-6,"Pixel coefficient did not cause the native action");
        for (unsigned action=0;action<18;++action) if (action!=15) require(e.actions[b+action]==0,"Pixel control leaked into another action");
        for (unsigned c=first;c<features;++c) require(std::isfinite(e.actor[a+c]),"Nonfinite live retinal input");
        if (t<2) for (unsigned c=first;c<features;++c) require(e.actor[a+c]==0,"Undelayed image leaked into actor");
    }
    require(maximum<2e-6,"GPU actor latent differs from independent pixel coefficient evaluation");
    for (const auto& s:food.snapshot()) require(s.accounting.y==0,"Yaw-only instrument consumed food");
    std::ofstream(out/"evidence.json")<<"{\"run_fingerprint\":"<<run.fingerprint()<<",\"observation_fingerprint\":"<<run.task().observationFingerprint()
        <<",\"policy_fingerprint\":"<<compiled.fingerprint()<<",\"maximum_actor_coefficient_error\":"<<maximum<<",\"actor_features\":1251,\"transition_record_bytes\":"<<sizeof(MRTaskTransitionGPU)<<"}\n";
    return e;
}
}
int main(int argc,char** argv) {@autoreleasepool {bool owns=false;try {
    require(argc==2,"Usage: metalrobo_crow_retinal_live_check NEW_OUTPUT");const std::filesystem::path out=argv[1];
    require(std::filesystem::create_directory(out),"Evidence directory exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-retinal-live-admission.v1","unit":"authored pixel-dependent actor, two physical head cameras, 2 native lanes, 32 ticks, dt=.02; no learning","seed":2651108701,"source_shape":[2,120,160,4],"actor_shape":[32,2,1251],"retinal":"2x12x16x3 area-mean RGB response c/(1+c), 15 Hz, 40 ms latency","actor":"yaw=clamp(2*(left[6,8].red-left[6,8].blue),-1,1); all other controls zero","controls":["visible chunk1","visible chunk8","occluded chunk8","pixel disconnected chunk8"],"predictions":{"cold_start_image_ticks":2,"sensor_errors":0,"max_pixel_to_latent_error":0.000002,"chunk_outputs_bit_identical":true,"visible_yaw_at_tick2_positive":true,"occluded_yaw_at_tick2_negative":true,"disconnected_actions_zero":true,"visible_physical_yaw_changes":true},"stopping":"four bounded runs, retain failed attempts, no seed or threshold search","limits":"instantaneous estimated parallel cameras; renders every tick; no biological calibration, visual learning, restored sensory state or whole-environment qualification"})";
    const auto one=execute(false,true,1,out/"visible-chunk1"),eight=execute(false,true,8,out/"visible-chunk8"),
        hidden=execute(true,true,8,out/"occluded"),off=execute(false,false,8,out/"disconnected");
    const auto same=[](const auto& a,const auto& b) {return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]));};
    require(same(one.actor,eight.actor) && same(one.actions,eight.actions) && same(one.latents,eight.latents) &&
        same(one.transitions,eight.transitions) && same(one.q,eight.q) && same(one.v,eight.v),"Live retinal outputs depend on submission partition");
    const float visible=one.actions[(2*N)*18+15],occluded=hidden.actions[(2*N)*18+15];
    require(visible>0 && occluded<0,"Physical occlusion did not change the pixel-driven action in the predicted direction");
    require(std::all_of(off.actions.begin(),off.actions.end(),[](float x){return x==0;}),"Disconnected control produced a nonzero action");
    require(one.q!=off.q,"Pixel-dependent actions did not change accepted physical state");
    std::ostringstream summary;summary<<"{\"schema\":\"numi.crow-retinal-live-evidence.v1\",\"status\":\"passed\",\"accepted_native_transitions\":256,\"cold_start_ticks\":2,\"chunk_bit_identical\":true,\"visible_tick2_yaw\":"<<visible<<",\"occluded_tick2_yaw\":"<<occluded<<",\"visual_learning_qualified\":false,\"environment_qualified\":false}\n";
    std::ofstream(out/"result.json")<<summary.str();std::cout<<summary.str();return 0;
} catch (const std::exception& e) {if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
