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
struct Evidence {std::vector<float> actor,actions,latents,q,v,trajectory;std::vector<MRBodyStateGPU> scene;std::vector<MRTaskTransitionGPU> transitions;};
Evidence execute(bool blocked,bool connected,unsigned chunk,const std::filesystem::path& out,bool resetLane=false) {@autoreleasepool {
    require(std::filesystem::create_directory(out),"Run directory exists");
    auto m=makeManifest(blocked);m.profile.capacities=m.task.capacities;
    // Trial02 required100 blocks in its first substep. Keep the original
    // crow envelope elsewhere and round this measured need to128 blocks.
    m.profile.capacities.constraintBlocks=128;m.profile.capacities.constraintRows=384;
    m.profile.capacities.endpointRuntimeRecords=256;m.profile.capacities.qualityRows=384;
    m.profile.capacities.islandConstraintReferences=128;m.task.capacities=m.profile.capacities;CompiledRun base;check(compileRun(m,base),"compile physical camera fixture");
    const auto visual=makeVisual(base,out);
    VisualSensorProgram p;p.cameraParentBody="crow_head";p.cameraPosition={.018f,.026f,.012f,0};p.cameraOrientation={-.5f,.5f,-.5f,.5f};
    p.width=W;p.height=H;p.verticalFieldOfViewDegrees=45;p.nominalRateHz=15;p.minimumVisiblePixels=1;
    p.retinal=RetinalSensorProgram{};
    for (const auto& pack:visual.manifest.renderScene.visualPacks) p.assets.push_back({pack.packPath.string(),
        visual.family.worldTemplate.assets[pack.assetIndex].id,pack.contentHash,pack.semanticId,pack.instanceId});
    p.fingerprint=visualSensorProgramFingerprint(p);m.sensors.observation={};m.reality.reset={};m.task=makeAmericanCrowRetinalFeedingTaskPack(m.sensors.observation,m.reality.reset,*p.retinal);
    m.task.capacities=m.profile.capacities;m.sensors.deviceVisual=p;CompiledRun run;check(compileRun(m,run),"compile actual retinal run");
    require(base.world().fingerprint()==run.world().fingerprint(),"Sensor changed physical world");
    if (!blocked && connected && chunk==1 && !resetLane) {
        std::ofstream controls(out/"observer-negative-controls.jsonl");
        for (unsigned test=0;test<8;++test) {
            EpisodeTwin bad;bad.id=visual.family.worldTemplate.id;bad.assets=visual.family.worldTemplate.assets;
            bad.sensors=visual.family.worldTemplate.sensors;bad.task=visual.family.worldTemplate.task;
            if (test==0) bad.sensors[0].localPose.position.z+=.005f;
            if (test==1) bad.sensors[1].localPose.orientation={0,0,0,1};
            if (test==2) bad.sensors[0].colorNoiseSigma=.01f;
            if (test==3) bad.sensors[0].exposureSeconds=1.0f/120;
            if (test==4) bad.sensors[0].latencySeconds=.02f;
            if (test==5) bad.sensors[0].focalScale=2;
            if (test==6) bad.assets.back().initialPose.position.y+=.01f;
            WorldTemplate badTemplate;check(compileEpisodeTwin(bad,run.model(),badTemplate),"compile negative fixture");
            WorldProgram fixed;fixed.id=visual.family.program.id;fixed.instanceFlags=visual.family.program.instanceFlags;
            WorldFamily badFamily;check(compileWorldFamily(badTemplate,fixed,badFamily),"compile negative family");
            if (test==7) ++badFamily.worldTemplate.fingerprint;
            bool rejected=false;std::string reason;
            try {RetinalObserverContext invalid(run,badFamily,visual.manifest,N,2651108701u);}
            catch (const std::invalid_argument& error) {rejected=true;reason=error.what();}
            controls<<"{\"case\":"<<test<<",\"rejected\":"<<(rejected ? "true" : "false")<<",\"reason\":"<<std::quoted(reason)<<"}\n";controls.flush();
            require(rejected,"Observer admitted a physical sensor/scene different from its bound task");
        }
    }
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
    std::vector<std::vector<double>> oracle(steps);
    double imageError=0;unsigned pairs=0,raw=0,manifolds=0,blocks=0,rows=0;
    for (unsigned start=0;start<steps;start+=chunk) {
        const unsigned count=std::min(chunk,steps-start);std::vector<unsigned> resets(count*N,0);
        for (unsigned t=0;t<count;++t) if (resetLane && start+t==17) resets[t*N+1]=1;
        MetalWorldBatch batch{.environmentCount=N,.controlStepCount=count,.resetMasks=resets};
        MetalWorldSubmission ticket;
        if (!start) {batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;
            check(world.initializeResidentState(run.world(),batch,step,resident,ticket),"initialize live retinal world");}
        else check(world.submitResident(run.world(),batch,step,resident,ticket),"submit live retinal world");
        const auto status=ticket.wait(result);check(status,"accept live retinal world");
        for (const auto& s:result.environmentStatuses) {
            pairs=std::max(pairs,s.highWater.candidatePairs);raw=std::max(raw,s.highWater.rawContacts);
            manifolds=std::max(manifolds,s.highWater.manifolds);blocks=std::max(blocks,s.highWater.constraintBlocks);rows=std::max(rows,s.highWater.constraintRows);
        }
        require(!status.failedStepCount,"Live retinal physics rejected a step");
        if (chunk==1) {
            const auto images=retina.inspectImages();
            std::vector<double> reduced(N*1152,0);
            for (unsigned eye=0;eye<2;++eye) {
                const auto& image=images[eye];require(image.rgb.size()==N*W*H,"Live RGB publication dimensions differ");
                // Save raw RGB before comparison, so failed instrument evidence survives.
                save(out/("tick-"+std::to_string(start)+"-eye-"+std::to_string(eye)+".rgba.f32"),image.rgb);
                for (unsigned env=0;env<N;++env) for (unsigned y=0;y<12;++y) for (unsigned x=0;x<16;++x) {
                    double sums[3]{};
                    for (unsigned dy=0;dy<10;++dy) for (unsigned dx=0;dx<10;++dx) {
                        const auto& pixel=image.rgb[env*W*H+(y*10+dy)*W+x*10+dx];
                        sums[0]+=pixel.x;sums[1]+=pixel.y;sums[2]+=pixel.z;
                    }
                    for (unsigned channel=0;channel<3;++channel) {
                        const double mean=sums[channel]/100;
                        reduced[(env*2+eye)*576+(y*16+x)*3+channel]=mean/(1+mean);
                    }
                }
            }
            oracle[start]=std::move(reduced);
            for (unsigned env=0;env<N;++env) for (unsigned channel=0;channel<1152;++channel) {
                const unsigned epochStart=resetLane && env==1 && start>=17 ? 17 : 0;
                const unsigned local=start-epochStart;
                unsigned localCapture=0;
                if (local>=2) for (unsigned tick=1;tick<=local-2;++tick) if ((tick*3)/10>((tick-1)*3)/10) localCapture=tick;
                const double expected=local<2 ? 0.0 : oracle[epochStart+localCapture][env*1152+channel];
                imageError=std::max(imageError,std::abs(expected-result.actorObservations[env*features+first+channel]));
            }
        }
        e.actor.insert(e.actor.end(),result.actorObservations.begin(),result.actorObservations.end());
        e.actions.insert(e.actions.end(),result.policyActions.begin(),result.policyActions.end());
        e.latents.insert(e.latents.end(),result.policyLatents.begin(),result.policyLatents.end());
        e.transitions.insert(e.transitions.end(),result.transitions.begin(),result.transitions.end());
        trajectory.insert(trajectory.end(),result.observations.begin(),result.observations.end());
    }
    e.q=result.finalQ;e.v=result.finalV;e.scene=result.finalSceneBodies;e.trajectory=trajectory;const auto sensor=retina.snapshot();
    save(out/"actor.f32",e.actor);save(out/"actions.f32",e.actions);save(out/"latents.f32",e.latents);
    save(out/"transitions.bin",e.transitions);save(out/"trajectory.f32",trajectory);save(out/"final-q.f32",e.q);save(out/"final-v.f32",e.v);save(out/"final-scene.bin",e.scene);
    save(out/"retinal-state.bin",sensor.states);save(out/"retinal-errors.u32",sensor.errors);
    require(e.actor.size()==steps*N*features && e.actions.size()==steps*N*18 && e.latents.size()==steps*N*18,"Native observation/action publication dimensions differ");
    for (auto error:sensor.errors) require(error==0,"Retinal GPU reported a sticky sensor error");
    for (unsigned t=0;t<steps;++t) for (unsigned env=0;env<N;++env) {
        const auto& transition=e.transitions[t*N+env];
        // A declared explicit reset is a boundary, never a physical failure.
        require(transition.termination.x==0,"Live retinal control terminated");
    }
    double maximum=0;
    for (unsigned t=0;t<steps;++t) for (unsigned env=0;env<N;++env) {
        const auto a=(t*N+env)*features,b=(t*N+env)*18;
        const unsigned local=resetLane && env==1 && t>=17 ? t-17 : t;
        require(e.actor[a+valid]==(local<2 ? 0 : 1),"Retinal cold-start latency was not two control ticks");
        unsigned capture=0;
        if (local>=2) for (unsigned tick=1;tick<=local-2;++tick) if ((tick*3)/10>((tick-1)*3)/10) capture=tick;
        require(std::abs(e.actor[a+age]-(local<2 ? 0.0f : (local-capture)*.02f))<2e-6,"Live retinal age differs from integer timing oracle");
        const float expected=connected ? 2*(e.actor[a+red]-e.actor[a+blue]) : 0;
        maximum=std::max(maximum,double(std::abs(expected-e.latents[b+15])));
        require(std::abs(std::clamp(expected,-1.0f,1.0f)-e.actions[b+15])<2e-6,"Pixel coefficient did not cause the native action");
        for (unsigned action=0;action<18;++action) if (action!=15) require(e.actions[b+action]==0,"Pixel control leaked into another action");
        for (unsigned c=first;c<features;++c) require(std::isfinite(e.actor[a+c]),"Nonfinite live retinal input");
        if (local<2) for (unsigned c=first;c<features;++c) require(e.actor[a+c]==0,"Undelayed image leaked into actor");
    }
    require(imageError<3e-6,"Live renderer RGB differs from independent FP64 delayed area-reduction oracle");
    require(maximum<2e-6,"GPU actor latent differs from independent pixel coefficient evaluation");
    for (const auto& s:food.snapshot()) require(s.accounting.y==0,"Yaw-only instrument consumed food");
    std::ofstream(out/"capacity-high-water.json")<<"{\"candidate_pairs\":"<<pairs<<",\"raw_contacts\":"<<raw<<",\"manifolds\":"<<manifolds<<",\"constraint_blocks\":"<<blocks<<",\"constraint_rows\":"<<rows<<"}\n";
    std::ofstream(out/"evidence.json")<<"{\"run_fingerprint\":"<<run.fingerprint()<<",\"observation_fingerprint\":"<<run.task().observationFingerprint()
        <<",\"policy_fingerprint\":"<<compiled.fingerprint()<<",\"maximum_delayed_rgb_error\":"<<imageError<<",\"maximum_actor_coefficient_error\":"<<maximum<<",\"actor_features\":1251,\"transition_record_bytes\":"<<sizeof(MRTaskTransitionGPU)<<"}\n";
    return e;
}}
}
int main(int argc,char** argv) {@autoreleasepool {bool owns=false;try {
    require(argc==2,"Usage: metalrobo_crow_retinal_live_check NEW_OUTPUT");const std::filesystem::path out=argv[1];
    require(std::filesystem::create_directory(out),"Evidence directory exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-retinal-live-admission.v10","parent_failures":"01 duplicated observation setup rejected policy compile; 02 default 96-constraint envelope rolled back at required100; v2 authors 256 pairs,512 raw contacts,128 manifolds,256 blocks; 03 rejected mismatched TaskPack capacity contract before physics; v3 binds the same allocation in both profile and task; behavioral predictions unchanged","instrument_extension":"05 passed strict provenance; v4 adds all live RGB features versus an independent FP64 frame-history/area-mean oracle for chunk1; behavioral thresholds unchanged","reset_extension":"one explicit lane1 reset at global tick17; lane0 uninterrupted; reset local clock repeats two blind ticks and restarts rational cadence; compare chunk1/8","resource_lifecycle_extension":"06 stopped at fifth scene construction after sample showed dispatch thread limit64 blocked in Metal IO queue; 07 repeated the same stall despite per-run autorelease pools; v7 reuses one bounded serial Metal IO loader per device for scene geometry/textures; all completed raw outputs and stop sample retained; no behavioral thresholds changed","instrument_boundary_extension":"08 passed all six runs with bounded shared IO; v8 additionally checks eight malformed observer inputs, uninterrupted-lane isolation, complete q/v trajectories and final scene partition equality; behavioral thresholds unchanged","capacity_instrument_extension":"09 passed all stricter checks; v9 retains compact native contact high-water counts to replace exploratory capacity envelope with measured bounds; behavioral predictions unchanged","capacity_refinement":"10 passed high-water instrumentation; trial02 required100 blocks within first substep, so v10 keeps the original crow envelope and authors128 blocks/384 rows/256 endpoints plus matched quality/island bounds; per-control publication high-waters are retained separately; behavioral thresholds unchanged","unit":"authored pixel-dependent actor, two physical head cameras, 2 native lanes, 32 ticks, dt=.02; no learning","seed":2651108701,"source_shape":[2,120,160,4],"actor_shape":[32,2,1251],"retinal":"2x12x16x3 area-mean RGB response c/(1+c), 15 Hz, 40 ms latency","actor":"yaw=clamp(2*(left[6,8].red-left[6,8].blue),-1,1); all other controls zero","controls":["visible chunk1","visible chunk8","occluded chunk8","pixel disconnected chunk8","lane1 reset chunk1","lane1 reset chunk8"],"predictions":{"cold_start_image_ticks":2,"sensor_errors":0,"malformed_observers_rejected":8,"reset_does_not_contaminate_other_lane":true,"max_pixel_to_latent_error":0.000002,"max_delayed_rgb_error":0.000003,"chunk_outputs_bit_identical":true,"visible_yaw_at_tick2_positive":true,"occluded_yaw_at_tick2_negative":true,"disconnected_actions_zero":true,"visible_physical_yaw_changes":true},"stopping":"six bounded runs, retain failed attempts, no seed or threshold search","limits":"instantaneous estimated parallel cameras; renders every tick; no biological calibration, visual learning, restored sensory state or whole-environment qualification"})";
    const auto one=execute(false,true,1,out/"visible-chunk1"),eight=execute(false,true,8,out/"visible-chunk8"),
        hidden=execute(true,true,8,out/"occluded"),off=execute(false,false,8,out/"disconnected"),resetOne=execute(false,true,1,out/"reset-chunk1",true),
        resetEight=execute(false,true,8,out/"reset-chunk8",true);
    const auto same=[](const auto& a,const auto& b) {return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]));};
    require(same(one.actor,eight.actor) && same(one.actions,eight.actions) && same(one.latents,eight.latents) &&
        same(one.transitions,eight.transitions) && same(one.q,eight.q) && same(one.v,eight.v) && same(one.scene,eight.scene) && same(one.trajectory,eight.trajectory),"Live retinal outputs depend on submission partition");
    require(same(resetOne.actor,resetEight.actor) && same(resetOne.actions,resetEight.actions) && same(resetOne.latents,resetEight.latents) &&
        same(resetOne.transitions,resetEight.transitions) && same(resetOne.q,resetEight.q) && same(resetOne.v,resetEight.v) && same(resetOne.scene,resetEight.scene) && same(resetOne.trajectory,resetEight.trajectory),"Live retinal per-lane reset depends on submission partition");
    for (unsigned t=0;t<steps;++t) {
        require(!std::memcmp(&one.actor[t*N*features],&resetOne.actor[t*N*features],features*sizeof(float)) &&
            !std::memcmp(&one.actions[t*N*18],&resetOne.actions[t*N*18],18*sizeof(float)) &&
            !std::memcmp(&one.transitions[t*N],&resetOne.transitions[t*N],sizeof(MRTaskTransitionGPU)),
            "Reset of lane1 contaminated uninterrupted lane0");
    }
    const float visible=one.actions[(2*N)*18+15],occluded=hidden.actions[(2*N)*18+15];
    require(visible>0 && occluded<0,"Physical occlusion did not change the pixel-driven action in the predicted direction");
    require(std::all_of(off.actions.begin(),off.actions.end(),[](float x){return x==0;}),"Disconnected control produced a nonzero action");
    require(one.q!=off.q,"Pixel-dependent actions did not change accepted physical state");
    std::ostringstream summary;summary<<"{\"schema\":\"numi.crow-retinal-live-evidence.v1\",\"status\":\"passed\",\"accepted_native_transitions\":384,\"cold_start_ticks\":2,\"chunk_bit_identical\":true,\"per_lane_reset_bit_identical\":true,\"visible_tick2_yaw\":"<<visible<<",\"occluded_tick2_yaw\":"<<occluded<<",\"visual_learning_qualified\":false,\"environment_qualified\":false}\n";
    std::ofstream(out/"result.json")<<summary.str();std::cout<<summary.str();return 0;
} catch (const std::exception& e) {if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
