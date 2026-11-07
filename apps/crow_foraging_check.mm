#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace {
using namespace metalrobo;
void require(bool valid,const char* message) { if (!valid) throw std::runtime_error(message); }
bool equal(const std::vector<MRFiniteResourceStateGPU>& a,const std::vector<MRFiniteResourceStateGPU>& b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0;
}
CompiledRun crowRun() {
    RunManifest m;
    m.id="crow_resource_admission"; m.robot=makeAmericanCrowActiveHeadRobotPack();
    m.scene=makeBirdFlowAmericanCrowFlightScenePack(); m.sensors.id="proprioception";
    m.task=makeAmericanCrowEmbodiedTaskPack(m.sensors.observation,m.reality.reset);
    // Transaction control has identical canonical starts. Randomized search
    // belongs to a behavioral experiment, not this intake-geometry admission.
    m.reality.reset={};
    m.task.maximumEpisodeSteps=100000u;
    m.reality.id="resource_reality";m.teacher.id="no_teacher";m.profile.id="resource_profile";
    m.profile.environmentCount=8;m.profile.controlSteps=448;
    m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=0.02f;
    CompiledRun run;
    const auto status=compileRun(m,run);
    if (!status.succeeded()) throw std::runtime_error(status.message);
    return run;
}
MetalWorldStepConfig config(const CompiledRun& run) {
    MetalWorldStepConfig c;
    c.timestepSeconds=0.02f;c.physicsSubsteps=4;
    c.solverMode=MetalWorldSolverMode::temporalCone;
    c.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    c.taskProgram=run.task();c.taskSeed=2651107401u;
    c.minimumDifficultyBand=0;c.maximumDifficultyBand=0;
    c.velocityIterations=4;c.finalVelocityIterations=2;
    c.publishStateTrajectory=false;c.publishFinalState=true;
    if (run.flappingWingProgram()) c.flappingWingProgram=*run.flappingWingProgram();
    return c;
}
MetalWorldBatch initial(const CompiledRun& run,unsigned count,std::vector<float>& q,std::vector<float>& v,
                        std::vector<MRBodyStateGPU>& scene,std::vector<float>& actions,std::vector<unsigned>& resets) {
    for (unsigned env=0;env<8;++env) {
        q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());
        v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
        scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());
    }
    actions.resize(count*8u*18u,0.0f);
    resets.resize(count*8u,0u);
    return {.environmentCount=8,.controlStepCount=count,.initialQ=q,.initialV=v,.actions=actions,
            .resetMasks=resets,.initialSceneBodies=scene};
}
mr_float4 intakeCenter(const CompiledRun& run) {
    std::vector<float> q,v,actions;std::vector<MRBodyStateGPU> scene;
    std::vector<unsigned> resets;
    const auto batch=initial(run,128,q,v,scene,actions,resets);
    MetalWorldContext world;MetalWorldResult result;
    const auto status=world.run(run.world(),batch,config(run),result);
    if (!status.succeeded() || status.failedStepCount!=0)
        throw std::runtime_error("Intake authoring control failed: "+status.message);
    std::vector<double> cq(result.finalQ.begin(),result.finalQ.begin()+run.model().defaultQ.size());
    std::vector<double> cv(result.finalV.begin(),result.finalV.begin()+run.model().defaultV.size());
    std::vector<ArticulatedBodyKinematics> bodies(run.model().articulations[0].bodyCount);
    require(computeArticulatedBodyKinematics(run.model(),0,cq,cv,bodies).succeeded(),"Intake authoring FK failed");
    // Reference FK authors a fixed intake volume once. Runtime feeding uses
    // only native accepted body poses, never this host oracle.
    const auto& head=bodies[15];
    // The physical upper beak extends along the head's local +x.
    const auto& rotation=head.orientation;
    const double x=rotation[0],y=rotation[1],z=rotation[2],w=rotation[3];
    const auto& ground=run.defaultSceneBodies().front();
    require(ground.orientation.x==0 && ground.orientation.y==0 && ground.orientation.z==0 && ground.orientation.w==1,
            "Authoring fixture expects an axis-aligned ground frame");
    return {float(head.centerOfMassPosition[0]+0.065*(1-2*(y*y+z*z))-ground.position.x),
            float(head.centerOfMassPosition[1]+0.065*2*(x*y+w*z)-ground.position.y),
            float(head.centerOfMassPosition[2]+0.065*2*(x*z-w*y)-ground.position.z),0.030f};
}
FiniteResourceConfig resources(const CompiledRun& run,mr_float4 center) {
    FiniteResourceConfig c;
    c.worldFingerprint=run.world().fingerprint();c.environmentCount=8;
    c.bodyCount=run.model().bodies.size();c.nq=run.model().defaultQ.size();
    c.program.binding={15,22,1,100000};
    c.program.mouthClosed={0.065f,0,0,0.08f};c.program.opening={0.20f,0,0,0};
    c.patches.push_back({{run.world().sceneBodyIndices().front(),45000,20000,0},center});
    return c;
}
struct Trial { std::vector<MRFiniteResourceStateGPU> state; unsigned physicalTerminations=0; };
void saveTrial(const std::filesystem::path& path,const Trial& trial) {
    std::ofstream out(path);
    out<<"{\"physical_terminations\":"<<trial.physicalTerminations<<",\"lanes\":[";
    for (unsigned env=0;env<trial.state.size();++env) {
        if (env) out<<',';
        const auto& state=trial.state[env];
        out<<"{\"crop_micrograms\":"<<state.accounting.x<<",\"bites\":"<<state.accounting.z
           <<",\"remaining_micrograms\":"<<state.remaining[0].x<<",\"accepted_ticks\":"<<state.clock.x<<'}';
    }
    out<<"]}\n";
}
Trial physicalTrial(const CompiledRun& run,const FiniteResourceConfig& authored,unsigned chunk,
                    bool miss,bool reset,const std::filesystem::path& checkpoint={}) {
    FiniteResourceContext food(authored);
    MetalWorldContext context;MetalWorldResidentState resident;
    auto step=config(run);step.devicePhysicsProgram=food.deviceProgram();
    std::vector<float> initialQ,initialV,firstActions;std::vector<MRBodyStateGPU> scene;
    Trial result;
    for (unsigned start=0;start<448;start+=chunk) {
        const unsigned count=std::min(chunk,448u-start);
        std::vector<float> actions(count*8u*18u,0);std::vector<unsigned> resets(count*8u,0);
        for (unsigned t=0;t<count;++t) for (unsigned env=0;env<8;++env) {
            const unsigned time=start+t;
            actions[(t*8u+env)*18u+15u]=miss ? 1.0f : 0.0f;
            if (time>=64 && time<320 && ((time-64u)/32u)%2u==0u)
                actions[(t*8u+env)*18u+17u]=0.9f;
            if (reset && time==352u) resets[t*8u+env]=1;
        }
        MetalWorldBatch batch{.environmentCount=8,.controlStepCount=count,.actions=actions,.resetMasks=resets};
        MetalWorldSubmission submission;
        MetalWorldDiagnostics status;
        if (start==0) {
            auto seed=initial(run,count,initialQ,initialV,scene,firstActions,resets);
            seed.actions=actions;seed.resetMasks=resets;
            status=context.initializeResidentState(run.world(),seed,step,resident,submission);
        } else status=context.submitResident(run.world(),batch,step,resident,submission);
        if (!status.succeeded()) throw std::runtime_error("Resource native submit: "+status.message);
        MetalWorldResult native;
        status=submission.wait(native);
        if (!status.succeeded()) throw std::runtime_error("Resource native wait: "+status.message);
        require(status.failedStepCount==0,"Resource test encountered a rejected physics step");
        for (const auto& transition:native.transitions) {
            const unsigned reason=transition.termination.w;
            if (transition.termination.x && reason!=4u && reason!=7u) ++result.physicalTerminations;
        }
    }
    result.state=food.snapshot();
    if (!checkpoint.empty()) {
        food.saveCheckpoint(checkpoint);
        FiniteResourceContext restored(authored);restored.restoreCheckpoint(checkpoint);
        require(equal(result.state,restored.snapshot()),"Resource checkpoint did not restore exactly");
        // Truncation, corruption, a different world contract, and trailing data
        // must fail without changing an already accepted resource snapshot.
        std::ifstream source(checkpoint,std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(source)),{});
        for (unsigned mutation=0;mutation<4;++mutation) {
            auto changed=bytes;
            if (mutation==0) changed.resize(changed.size()-1);
            if (mutation==1) changed.back()^=1;
            if (mutation==2) changed.push_back('x');
            if (mutation==3) {
                // A valid checksum is insufficient: this invents one
                // microgram in the first lane's crop and recomputes SHA-256.
                changed[64+32]^=1;
                CC_SHA256(changed.data()+64,static_cast<CC_LONG>(changed.size()-64),
                          reinterpret_cast<unsigned char*>(changed.data()+32));
            }
            const auto corrupt=checkpoint.string()+".negative-"+std::to_string(mutation);
            std::ofstream out(corrupt,std::ios::binary);out.write(changed.data(),changed.size());out.close();
            bool rejected=false;
            try { restored.restoreCheckpoint(corrupt); } catch (const std::exception&) { rejected=true; }
            require(rejected && equal(result.state,restored.snapshot()),"Invalid resource restore mutated accepted state");
        }
        auto other=authored;other.worldFingerprint^=1u;FiniteResourceContext mismatch(other);
        bool rejected=false;
        try { mismatch.restoreCheckpoint(checkpoint); } catch (const std::exception&) { rejected=true; }
        require(rejected,"Resource checkpoint rebound to a different world");
    }
    return result;
}
void fixtureControls(const std::filesystem::path& output) {
    // Synthetic accepted-state fixtures exercise edge cases independently of
    // animal behavior. They execute the same compiled Metal production kernel.
    FiniteResourceConfig c;
    c.worldFingerprint=12345;c.environmentCount=8;c.bodyCount=2;c.nq=1;
    c.program.binding={0,0,2,100};c.program.mouthClosed={0,0,0,0.08};c.program.opening={0.20,0,0,0};
    c.patches={{{1,45,20,0},{0,0,0,0.05}},{{1,45,20,0},{0,0,0,0.05}}};
    FiniteResourceContext food(c);
    const auto device=MTLCreateSystemDefaultDevice();const auto queue=[device newCommandQueue];
    std::array<MRBodyStateGPU,16> bodies{};
    for (auto& body:bodies) body.orientation.w=1;
    std::array<float,8> angles{};std::array<MRMetalWorldStatusGPU,8> statuses{};
    id<MTLBuffer> body=[device newBufferWithBytes:bodies.data() length:sizeof(bodies) options:MTLResourceStorageModeShared];
    id<MTLBuffer> q=[device newBufferWithBytes:angles.data() length:sizeof(angles) options:MTLResourceStorageModeShared];
    id<MTLBuffer> status=[device newBufferWithBytes:statuses.data() length:sizeof(statuses) options:MTLResourceStorageModeShared];
    auto extension=food.deviceProgram();
    const auto advance=[&](float angle,bool reject,unsigned phase=1) {
        for (auto& value:angles) value=angle;std::memcpy(q.contents,angles.data(),sizeof(angles));
        for (auto& value:statuses) value.code=reject ? MR_STEP_NONFINITE_INPUT : MR_STEP_SUCCESS;
        std::memcpy(status.contents,statuses.data(),sizeof(statuses));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        MetalWorldDevicePhysicsPass pass;
        pass.commandBuffer=(__bridge void*)command;pass.q=(__bridge void*)q;pass.currentBodies=(__bridge void*)body;
        pass.environmentStatuses=(__bridge void*)status;pass.worldFingerprint=c.worldFingerprint;
        pass.phase=phase ? MetalWorldDevicePhysicsPhase::postCommit : MetalWorldDevicePhysicsPhase::preDynamics;
        pass.environmentCount=8;pass.bodyCount=2;pass.nq=1;pass.bodyStateStride=2;pass.qStride=1;
        pass.physicsSubstep=3;pass.physicsSubsteps=4;
        require(extension.encode(extension.context,pass),"Synthetic native resource encoding rejected");
        [command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"Synthetic resource GPU command failed");
    };
    advance(0,false);
    auto previous=food.snapshot();
    advance(0.3,true);require(equal(previous,food.snapshot()),"Rejected physics advanced resources");
    advance(0.3,false,0);require(equal(previous,food.snapshot()),"Pre-dynamics advanced resources");
    advance(0.3,false);advance(0,false);
    auto one=food.snapshot();
    require(one[0].accounting.x==20 && one[0].remaining[0].x==25 && one[0].remaining[0].y==45,
            "Overlapping volumes double-debited a bite");
    advance(0,false);require(food.snapshot()[0].accounting.x==20,"Holding the jaw closed repeated a bite");
    food.saveCheckpoint(output/"synthetic-resources.checkpoint");
    FiniteResourceContext restored(c);restored.restoreCheckpoint(output/"synthetic-resources.checkpoint");
    for (unsigned bite=0;bite<10;++bite) { advance(0.3,false);advance(0,false); }
    for (const auto& state:food.snapshot()) require(state.accounting.x==90 && state.accounting.z==6 &&
            state.remaining[0].x==0 && state.remaining[0].y==0,"Finite inventory was duplicated or lost");
    extension=restored.deviceProgram();
    for (unsigned bite=0;bite<10;++bite) { advance(0.3,false);advance(0,false); }
    require(equal(food.snapshot(),restored.snapshot()),"Restored resource trajectory diverged from continuous execution");
    c.program.binding.w=27;
    FiniteResourceContext capped(c);extension=capped.deviceProgram();
    for (unsigned bite=0;bite<4;++bite) { advance(0.3,false);advance(0,false); }
    for (const auto& state:capped.snapshot()) require(state.accounting.x==27 &&
            state.remaining[0].x==18 && state.remaining[0].y==45,"Crop capacity broke exact conservation");
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        bool ownsOutput=false;
        try {
            require(argc==2,"Usage: metalrobo_crow_foraging_check NEW_OUTPUT_DIRECTORY");
            const std::filesystem::path output=argv[1];
            require(std::filesystem::create_directory(output),"Foraging evidence directory already exists");
            ownsOutput=true;
            const auto start=std::chrono::steady_clock::now();
            std::ofstream(output/"plan.json")<<R"({"schema":"numi.finite-resource-admission-plan.v2","seed":2651107401,"environments":8,"steps":448,"dt_seconds":0.02,"starting_distribution":"canonical identical transaction fixture; randomized search is separate","patch_micrograms":45000,"bite_micrograms":20000,"expected_bites":3,"expected_consumption_micrograms":45000,"controls":["native-miss","chunk-8-vs-64","ordinary-task-reset-no-refill","resource-only-restore","corrupt-restore-rejection","synthetic-physics-rejection","synthetic-overlap","hold-closed","exhaustion","crop-capacity","resource-only-restored-continuation"],"boundary":"nutritional intake-volume tracer; no solid food mechanics, swallowing, physiology or complete world restore"})";
            fixtureControls(output);
            const auto run=crowRun();const auto authored=resources(run,intakeCenter(run));
            std::ofstream(output/"authored-intake.json")<<"{\"ground_local_center_radius_m\":["
                <<authored.patches[0].centerRadius.x<<','<<authored.patches[0].centerRadius.y<<','
                <<authored.patches[0].centerRadius.z<<','<<authored.patches[0].centerRadius.w<<"]}\n";
            const auto ordinary=physicalTrial(run,authored,64,false,false,output/"resources.checkpoint");
            saveTrial(output/"ordinary.json",ordinary);
            const auto partition=physicalTrial(run,authored,8,false,false);
            saveTrial(output/"partition.json",partition);
            const auto missed=physicalTrial(run,authored,64,true,false);
            saveTrial(output/"missed.json",missed);
            const auto reset=physicalTrial(run,authored,64,false,true);
            saveTrial(output/"reset.json",reset);
            require(equal(ordinary.state,partition.state),"Native resource state depends on submission partition");
            for (unsigned env=0;env<8;++env) {
                require(ordinary.state[env].accounting.x==45000 && ordinary.state[env].accounting.z==3 &&
                        ordinary.state[env].remaining[0].x==0 && ordinary.state[env].clock.x==448,
                        "Native mouth motion failed exact food accounting");
                require(missed.state[env].accounting.x==0,"Native mouth miss consumed food");
                require(reset.state[env].accounting.x==45000 && reset.state[env].remaining[0].x==0,
                        "Ordinary task reset refilled or erased resources");
            }
            require(ordinary.physicalTerminations==0 && partition.physicalTerminations==0 &&
                    missed.physicalTerminations==0 && reset.physicalTerminations==0,
                    "Physical termination invalidates the native foraging control");
            const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const std::string result="{\"schema\":\"numi.crow-foraging-admission.v1\",\"status\":\"passed\","
                "\"native_environments\":8,\"native_steps_per_trial\":448,\"native_trials\":4,"
                "\"consumed_micrograms_per_lane\":45000,\"bites_per_lane\":3,\"physical_terminations\":0,"
                "\"failed_physics_steps\":0,\"exact_partition_replay\":true,\"resource_only_restore\":true,"
                "\"ordinary_reset_refills_resources\":false,\"synthetic_negative_controls\":true,"
                "\"wall_seconds\":"+std::to_string(seconds)+",\"boundary\":\"estimated nutritional intake-volume tracer; no solid food mechanics or whole-world restore\"}";
            std::ofstream(output/"summary.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
        } catch (const std::exception& error) {
            if (ownsOutput)
                std::ofstream(std::filesystem::path(argv[1])/"failure.json")<<"{\"error\":"<<std::quoted(error.what())<<"}\n";
            std::cerr<<error.what()<<'\n';return 1;
        }
    }
}
