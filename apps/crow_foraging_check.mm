#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/c_api.h"
#include "metalrobo/detail/Checkpoint.hpp"
#include <cmath>
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
void cAPIAccountingControls(const std::filesystem::path& output) {
    for (const bool feeding : {false,true}) {
        MRRunManifestC manifest{};
        manifest.source=MR_RUN_SOURCE_BIRDFLOW_AMERICAN_CROW_JOURNEY;
        manifest.profile.environment_count=2u;manifest.profile.physics_substeps=4u;
        manifest.profile.velocity_iterations=4u;manifest.profile.final_velocity_iterations=2u;
        manifest.profile.control_timestep_seconds=0.02f;manifest.profile.seed=2651107402u;
        manifest.profile.override_difficulty_band_range=1u;
        manifest.profile.birdflow_journey_variant=feeding ? MR_BIRDFLOW_JOURNEY_V12_FEEDING_STATION
                                                       : MR_BIRDFLOW_JOURNEY_V11_EMBODIED;
        std::unique_ptr<MRTaskRolloutHandle,decltype(&mr_task_rollout_destroy)> handle(
            mr_create_task_rollout(&manifest),&mr_task_rollout_destroy);
        if (!handle) throw std::runtime_error(mr_last_error());
        std::array<MRTaskResourceAccountingC,2> before;
        std::memset(before.data(),0x5a,sizeof(before));
        auto actual=before;
        require(mr_task_rollout_copy_resource_accounting(handle.get(),actual.data(),2u)!=0 &&
                std::memcmp(actual.data(),before.data(),sizeof(before))==0,
                "Uninitialized or absent resources returned fabricated accounting");
        require(mr_task_rollout_copy_resource_accounting(handle.get(),nullptr,2u)!=0 &&
                mr_task_rollout_copy_resource_accounting(handle.get(),actual.data(),1u)!=0 &&
                std::memcmp(actual.data(),before.data(),sizeof(before))==0,
                "Invalid accounting output mutated caller memory");
        std::array<float,36> actions{};std::array<std::uint32_t,2> resets{};
        MRTaskRolloutAdvanceC advance{};
        require(mr_task_rollout_advance(handle.get(),actions.data(),actions.size(),
                resets.data(),resets.size(),1u,0u,0u,&advance)==0 && advance.failed_environment_steps==0u,
                "C API accounting control failed its native physical step");
        const int status=mr_task_rollout_copy_resource_accounting(handle.get(),actual.data(),2u);
        if (!feeding) {
            require(status!=0 && std::memcmp(actual.data(),before.data(),sizeof(before))==0,
                    "A completed task without resources fabricated a zero ledger");
        } else {
            require(status==0,"Completed feeding task could not publish resource accounting");
            for (const auto& row : actual) require(row.remaining_micrograms==45000u &&
                    row.crop_micrograms==0u && row.consumed_micrograms==0u && row.bite_count==0u &&
                    row.initialized==1u && row.accepted_control_ticks==1u,
                    "C API initial finite ledger has the wrong inventory or time");
            auto again=before;
            require(mr_task_rollout_copy_resource_accounting(handle.get(),again.data(),2u)==0 &&
                    std::memcmp(actual.data(),again.data(),sizeof(actual))==0,
                    "Repeated publication changed accepted resource accounting");
        }
    }
    std::ofstream(output/"c-api-accounting-controls.json")
        <<"{\"status\":\"passed\",\"environments_per_variant\":2,\"native_steps_per_variant\":1,"
        "\"absent_and_uninitialized_rejected\":true,\"invalid_outputs_untouched\":true,"
        "\"exact_initial_ledger\":true,\"repeated_readback_stable\":true}\n";
}
FiniteResourceConfig spatialConfig(FiniteResourceConfig c) {
    c.program.binding.z=2;
    auto second=c.patches.front();second.centerRadius.y=.25f;
    c.patches.push_back(second);c.allocation={1,1,192,1};c.allocationSeed=2651109701ULL;
    return c;
}
CompiledRun crowRun(bool spatial=false) {
    RunManifest m;
    m.id="crow_resource_admission"; m.robot=makeAmericanCrowActiveHeadRobotPack();
    m.scene=makeBirdFlowAmericanCrowFlightScenePack(); m.sensors.id="proprioception";
    m.task=makeAmericanCrowFeedingStationTaskPack(m.sensors.observation,m.reality.reset);
    if (spatial) {
        // Author the same physical head/ground bindings as the baseline. Only
        // resource semantics change; no hidden site is an observation or reward.
        FiniteResourceConfig c;c.program={{15,22,1,100000},{.065f,0,0,.08f},{.20f,0,0,0}};
        c.patches={{{17,45000,20000,0},{.225585f,0,.271365f,.030f}}};c=spatialConfig(c);
        // Keep the canonical physical support semantics. The resource program
        // has its own new fingerprint; renaming this legacy TaskPack would
        // silently drop the compiler's avian-ground flag.
        m.task.deviceStateProgramFingerprint=finiteResourceProgramFingerprint(c.program,c.patches,true,c.allocation);
    }
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
struct Trial {
    std::vector<MRFiniteResourceStateGPU> state;
    unsigned physicalTerminations=0;
    std::array<double,8> utility{};
    std::array<double,8> resetStepUtility{};
    std::array<std::array<float,4>,8> sensed{};
};
void saveTrial(const std::filesystem::path& path,const Trial& trial) {
    std::ofstream out(path);
    out<<"{\"physical_terminations\":"<<trial.physicalTerminations<<",\"lanes\":[";
    for (unsigned env=0;env<trial.state.size();++env) {
        if (env) out<<',';
        const auto& state=trial.state[env];
        out<<"{\"crop_micrograms\":"<<state.accounting.x<<",\"bites\":"<<state.accounting.z
           <<",\"remaining_micrograms\":"<<state.remaining[0].x<<",\"accepted_ticks\":"<<state.clock.x
           <<",\"utility\":"<<trial.utility[env]<<",\"nutrition_observation\":[";
        for (unsigned component=0;component<4;++component) {
            if (component) out<<',';out<<trial.sensed[env][component];
        }
        out<<"]}";
    }
    out<<"]}\n";
}
Trial physicalTrial(const CompiledRun& run,const FiniteResourceConfig& authored,unsigned chunk,
                    bool miss,bool reset,const std::filesystem::path& checkpoint={},unsigned resetStep=352u,
                    std::vector<float>* preIntakeObservations=nullptr) {
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
            if (reset && time==resetStep) resets[t*8u+env]=1;
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
        for (unsigned index=0;index<native.transitions.size();++index) {
            const auto& transition=native.transitions[index];
            result.utility[index%8u]+=transition.rewardAndState.x;
            if (reset && start+index/8u==resetStep) result.resetStepUtility[index%8u]+=transition.rewardAndState.x;
            const unsigned reason=transition.termination.w;
            if (transition.termination.x && reason!=4u && reason!=7u) ++result.physicalTerminations;
        }
        const auto actorSize=run.task().layout().actorObservationSize;
        require(actorSize==97 && native.actorObservations.size()>=8u*actorSize,"Nutrition actor evidence missing");
        if (preIntakeObservations && start<192) {
            const auto rows=std::min(count,192u-start)*8u*actorSize;
            require(native.actorObservations.size()>=rows,"Pre-intake actor stream is truncated");
            preIntakeObservations->insert(preIntakeObservations->end(),native.actorObservations.begin(),native.actorObservations.begin()+rows);
        }
        const auto last=native.actorObservations.size()-8u*actorSize;
        for (unsigned env=0;env<8;++env) for (unsigned component=0;component<4;++component)
            result.sensed[env][component]=native.actorObservations[last+env*actorSize+93u+component];
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
    std::array<unsigned,8> resetMasks{};
    id<MTLBuffer> body=[device newBufferWithBytes:bodies.data() length:sizeof(bodies) options:MTLResourceStorageModeShared];
    id<MTLBuffer> q=[device newBufferWithBytes:angles.data() length:sizeof(angles) options:MTLResourceStorageModeShared];
    id<MTLBuffer> status=[device newBufferWithBytes:statuses.data() length:sizeof(statuses) options:MTLResourceStorageModeShared];
    id<MTLBuffer> resets=[device newBufferWithBytes:resetMasks.data() length:sizeof(resetMasks) options:MTLResourceStorageModeShared];
    id<MTLBuffer> tasks=[device newBufferWithLength:8*sizeof(MRTaskStateGPU) options:MTLResourceStorageModeShared];
    std::memset(tasks.contents,0,tasks.length);
    auto extension=food.deviceProgram();
    const auto advance=[&](float angle,bool reject,unsigned phase=1,bool resetting=false) {
        for (auto& value:angles) value=angle;std::memcpy(q.contents,angles.data(),sizeof(angles));
        for (auto& value:statuses) value.code=reject ? MR_STEP_NONFINITE_INPUT : MR_STEP_SUCCESS;
        std::memcpy(status.contents,statuses.data(),sizeof(statuses));
        resetMasks.fill(resetting ? 1u : 0u);std::memcpy(resets.contents,resetMasks.data(),sizeof(resetMasks));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        MetalWorldDevicePhysicsPass pass;
        pass.commandBuffer=(__bridge void*)command;pass.q=(__bridge void*)q;pass.currentBodies=(__bridge void*)body;
        pass.environmentStatuses=(__bridge void*)status;pass.worldFingerprint=c.worldFingerprint;
        pass.resetMasks=(__bridge void*)resets;
        pass.taskStates=(__bridge void*)tasks;
        pass.taskStateProgramFingerprint=finiteResourceProgramFingerprint(c.program,c.patches,c.publishTaskState,c.allocation);
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
    FiniteResourceContext resetControl(c);extension=resetControl.deviceProgram();
    advance(0.3,false);advance(0,false,1,true);advance(0,false);
    require(resetControl.snapshot()[0].accounting.x==0,"A reset manufactured a closing gesture");
    advance(0.3,false);advance(0,false);
    require(resetControl.snapshot()[0].accounting.x==20,"A valid post-reset bite was suppressed");

    c.publishTaskState=true;c.allocation={1,1,5,1};c.allocationSeed=2651109701ULL;
    c.program.binding.w=100;c.patches[1].centerRadius.x=.5f;
    FiniteResourceContext allocated(c);extension=allocated.deviceProgram();
    advance(0,false);
    auto selected=allocated.snapshot();unsigned left=0,right=0;
    for (unsigned env=0;env<8;++env) {
        const auto mask=mrResourceAllocationMask(c.allocationSeed,env,2,1);
        require(selected[env].clock.w==mask && selected[env].remaining[0].x==(mask==1 ? 45u : 0u) &&
            selected[env].remaining[0].y==(mask==2 ? 45u : 0u),"Native allocation disagrees with seeded fixed-lifetime selection");
        mask==1 ? ++left : ++right;
        bodies[2*env].position.x=mask==1 ? 0.0f : .5f;
    }
    require(left && right,"Allocation admission needs both spatial sites");
    std::memcpy(body.contents,bodies.data(),sizeof(bodies));
    advance(.3f,false);advance(0,false); // Accepted ticks2/3: food is not available.
    require(allocated.snapshot()[0].accounting.x==0,"Delay admitted an early close");
    advance(0,false);advance(0,false);advance(0,false); // Cross delay while closed.
    for (const auto& row:allocated.snapshot()) require(row.accounting.x==0,"Delay expiration banked a previous jaw edge");
    advance(.3f,false);advance(0,false); // New physical edge after delay.
    for (const auto& row:allocated.snapshot()) require(row.accounting.x==20 && row.accounting.y==20 &&
        row.accounting.z==1 && row.clock.x==8,"Delayed spatial intake failed exact debit/time");
    const auto* task=static_cast<const MRTaskStateGPU*>(tasks.contents);
    std::ofstream nutritionDebug(output/"allocation-nutrition-debug.json");nutritionDebug<<std::setprecision(10)<<'[';
    for (unsigned env=0;env<8;++env) {if (env) nutritionDebug<<',';const auto n=task[env].nutrition;
        nutritionDebug<<'['<<n.x<<','<<n.y<<','<<n.z<<','<<n.w<<']';}
    nutritionDebug<<"]\n";nutritionDebug.close();
    for (unsigned env=0;env<8;++env) require(std::abs(task[env].nutrition.x-.2f)<1e-6f && task[env].nutrition.y>0 &&
        task[env].nutrition.z==0 && task[env].nutrition.w==0,"Concealed resource sites leaked through nutrition");
    const auto accepted=allocated.snapshot();advance(.3f,true);
    require(equal(accepted,allocated.snapshot()),"Rejected physics changed spatial food state");
    advance(.3f,false);advance(0,false,1,true);advance(0,false);
    for (unsigned env=0;env<8;++env) require(allocated.snapshot()[env].clock.w==selected[env].clock.w &&
        allocated.snapshot()[env].accounting.x==20,"Episode reset rerolled/refilled spatial food");
    allocated.saveCheckpoint(output/"allocated-resources.checkpoint");
    FiniteResourceContext recovered(c);recovered.restoreCheckpoint(output/"allocated-resources.checkpoint");
    require(equal(allocated.snapshot(),recovered.snapshot()),"Spatial resource checkpoint did not preserve lifetime selection");
    unsigned invalid=0;
    for (unsigned mode=0;mode<2;++mode) {
        auto forged=allocated.snapshot();auto& row=forged[0];
        if (mode==0) row.clock.w^=3u;
        else { if (row.clock.w==1) {row.remaining[0].x-=5;row.remaining[0].y=5;}
               else {row.remaining[0].y-=5;row.remaining[0].x=5;} }
        const auto bytes=checkpoint::seal({'N','U','M','I','F','B','0','1'},allocated.fingerprint(),std::as_bytes(std::span(forged)));
        bool rejected=false;try {recovered.restoreState(bytes);}catch (const std::exception&) {rejected=true;}
        require(rejected && equal(allocated.snapshot(),recovered.snapshot()),"Forged allocation checkpoint mutated accepted state");++invalid;
    }
    auto differentSeed=c;differentSeed.allocationSeed++;
    require(finiteResourceProgramFingerprint(c.program,c.patches,true,c.allocation)==
        finiteResourceProgramFingerprint(c.program,c.patches,true,differentSeed.allocation),"Runtime seed leaked into policy task identity");
    FiniteResourceContext changedSeed(differentSeed);bool rejected=false;
    try {changedSeed.restoreCheckpoint(output/"allocated-resources.checkpoint");}catch (const std::exception&) {rejected=true;}
    require(rejected && changedSeed.fingerprint()!=allocated.fingerprint(),"A different lifetime seed admitted spatial state rebinding");++invalid;
    auto noFood=c;noFood.allocation.activePatchCount=0;
    FiniteResourceContext empty(noFood);extension=empty.deviceProgram();c=noFood;
    advance(.3f,false);for (unsigned i=0;i<8;++i) advance(0,false);
    for (const auto& row:empty.snapshot()) require(row.clock.w==0 && row.accounting.x==0 &&
        row.remaining[0].x==0 && row.remaining[0].y==0,"Zero-stock allocation manufactured food");
    auto emptyState=empty.checkpointState();FiniteResourceContext emptyRestored(c);emptyRestored.restoreState(emptyState);
    require(equal(empty.snapshot(),emptyRestored.snapshot()),"Zero-stock spatial lifetime failed restore");
    c=differentSeed;c.allocationSeed=2651109701ULL;
    FiniteResourceContext rollover(c);auto oldAge=allocated.snapshot();
    for (auto& row:oldAge) {row.clock.x=0xffffffffu;row.clock.y=0;}
    const auto ageFixture=checkpoint::seal({'N','U','M','I','F','B','0','1'},rollover.fingerprint(),std::as_bytes(std::span(oldAge)));
    rollover.restoreState(ageFixture);extension=rollover.deviceProgram();
    advance(.3f,false);advance(0,false);
    for (const auto& row:rollover.snapshot()) require(row.clock.x==1 && row.clock.y==1 && row.accounting.x==40,
        "Accepted lifetime clock carry reactivated the delay or lost intake");
    unsigned invalidPrograms=0;
    for (unsigned mode=0;mode<5;++mode) {
        auto bad=c;
        if (mode==0) bad.allocation.version=2;
        if (mode==1) bad.allocation.activePatchCount=3;
        if (mode==2) bad.allocation.concealProximity=2;
        if (mode==3) bad.allocation.version=0;
        if (mode==4) {bad.allocation={};bad.allocationSeed=1;}
        bool failed=false;try {FiniteResourceContext invalidProgram(bad);}catch (const std::exception&) {failed=true;}
        require(failed,"Malformed allocation program was admitted");++invalidPrograms;
    }
    // Admission covers the complete authored active-count range, rather
    // than inferring multi-site behavior from a one-of-two fixture.
    std::vector<unsigned> breadthMasks;
    for (unsigned active=0;active<=8;++active) {
        c.program.binding.z=8;c.patches.resize(8,c.patches.front());
        for (unsigned patch=0;patch<8;++patch) c.patches[patch].centerRadius.x=.5f*patch;
        c.allocation.activePatchCount=active;c.allocationSeed=2651109703ULL;
        FiniteResourceContext breadth(c);extension=breadth.deviceProgram();advance(0,false);
        const auto rows=breadth.snapshot();
        for (unsigned env=0;env<8;++env) {
            const auto mask=rows[env].clock.w;breadthMasks.push_back(mask);
            require(unsigned(__builtin_popcount(mask))==active && mask==mrResourceAllocationMask(c.allocationSeed,env,8,active),
                "Native allocation did not select the authored number of sites");
            const auto* amounts=reinterpret_cast<const unsigned*>(rows[env].remaining);
            for (unsigned patch=0;patch<8;++patch) require(amounts[patch]==((mask&(1u<<patch)) ? 45u : 0u),
                "Native multi-site initialization lost or duplicated stock");
        }
        const auto bytes=breadth.checkpointState();FiniteResourceContext back(c);back.restoreState(bytes);
        require(equal(rows,back.snapshot()),"Multi-site allocation count failed checkpoint conservation");
    }
    std::ofstream evidence(output/"allocation-controls.json");
    evidence<<"{\"status\":\"passed\",\"seed\":2651109701,\"left_lanes\":"<<left<<",\"right_lanes\":"<<right
        <<",\"rejected_forged_or_rebound_checkpoints\":"<<invalid<<",\"rejected_invalid_programs\":"<<invalidPrograms<<",\"clock_carry_preserves_availability\":true,\"native_masks\":[";
    for (unsigned env=0;env<8;++env) {if (env) evidence<<',';evidence<<selected[env].clock.w;}
    evidence<<"],\"breadth_seed\":2651109703,\"breadth_masks_by_active_count\":[";
    for (unsigned i=0;i<breadthMasks.size();++i) {if (i) evidence<<',';evidence<<breadthMasks[i];}
    evidence<<"],\"delayed_edge_and_concealment\":true,\"reset_preserves_selection_and_stock\":true,\"zero_stock_control\":true}\n";
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
            std::ofstream(output/"plan.json")<<R"({"schema":"numi.finite-resource-admission-plan.v4","seed":2651107401,"environments":8,"steps":448,"dt_seconds":0.02,"starting_distribution":"canonical identical transaction fixture; randomized search is separate","patch_micrograms":45000,"bite_micrograms":20000,"expected_bites":3,"expected_utility_per_lane":4.5,"expected_consumption_micrograms":45000,"controls":["native-miss","chunk-8-vs-64","ordinary-task-reset-no-refill","resource-only-restore","corrupt-restore-rejection","c-api-null-size-absent-uninitialized-rejections","c-api-readback-purity","synthetic-physics-rejection","synthetic-overlap","hold-closed","exhaustion","crop-capacity","resource-only-restored-continuation"],"boundary":"nutritional intake-volume tracer; no solid food mechanics, swallowing, physiology or complete world restore"})";
            std::ofstream(output/"spatial-plan.json")<<R"({"schema":"numi.spatial-resource-plan.v5","ABI_revision":"Attempt04 passed all physical, causal-observation and0..8-site controls. Preserve the legacy three-argument exported C++ fingerprint symbol as a wrapper over the zero-allocation descriptor; retain exact legacy hash bytes. Remove an unsigned-popcount comparison warning; repeat unchanged numerical admission.","breadth_calibration":"Attempt03 passed. Add72 native lanes covering every active count0..8 over eight sites at fixed seed2651109703, exact stock/popcount and checkpoint restoration; repeat the same physical and causality controls without model or thresholds changes.","instrument_revision":"Attempt02 passed every synthetic allocation and all five baseline physical controls, then a renamed TaskPack silently lost the existing compiler avian-ground flag and hit100 contact constraints versus capacity96. Preserve failure. Inherit the canonical feeding TaskPack identity/physical flags and distinguish resource variants through their executable state-program fingerprint; assert inherited flags. Add a fixed-seed paired pre-intake observation control, same mechanism and no relaxed capacity/physics bounds.","parent":"Attempt01/diagnostic01 concealed proximity lanes were exactly zero; crop fraction0.1999999881 differs by one FP32 rounding step from the asserted literal0.2. Use the existing1e-6 nutrition-fraction bound and retain exact-zero concealment. Code review also found shadowed availability variable disabling the legacy presence loop; rename the gate and rerun all legacy controls. Preserve prior runtime/source and failed attempts.","purpose":"native randomized food allocation and delayed-access engineering admission, not a memory or biological experiment","prediction":"Exactly one of two sites is stocked once per lane at seed2651109701; no seed or site/proximity label is published in concealed nutrition. First192 accepted ticks forbid intake; existing close gestures at96/160 consume nothing and new closes at224/288 debit exactly40000ug only in lanes whose central site is stocked. Chunk8/64 are byte exact; episode reset preserves mask, stock, debit and accepted clock; chosen stock and zero-stock controls conserve mass and survive resource restore; forged masks/inactive stock and changed lifetime seeds reject without mutation.","units":"eight supported estimated native crow lanes; synthetic accepted-state controls separately identified","budget":"five existing448tick controls plus four448tick spatial trials and exact192tick pre-intake actor comparison; abort first invalidity, no retries or threshold changes","limits":"Second site is a distant miss control, not a demonstrated reachable alternative. No visible cue, learned search, caching, physical food mechanics or complete spatial-life checkpoint admission."})";
            fixtureControls(output);
            cAPIAccountingControls(output);
            const auto run=crowRun();const auto authored=makeAmericanCrowFeedingResourceConfig(run,8);
            TaskObservationProgram authoredObservations;TaskResetProgram authoredReset;
            const auto authoredTask=makeAmericanCrowFeedingStationTaskPack(authoredObservations,authoredReset);
            require(writeTaskPack(authoredTask,output/"feeding.taskpack").succeeded(),"Feeding TaskPack could not be saved");
            TaskPack loadedTask;
            require(readTaskPack(output/"feeding.taskpack",loadedTask).succeeded() &&
                    loadedTask.deviceStateProgramFingerprint==authoredTask.deviceStateProgramFingerprint,
                    "TaskPack did not preserve the native resource identity");
            auto changed=authored;
            changed.patches[0].binding.y+=1u;
            require(finiteResourceProgramFingerprint(changed.program,changed.patches,true)!=
                    run.task().header().deviceStateProgramFingerprint,
                    "Changing finite food did not change the bound native state-program identity");
            FiniteResourceContext mismatch(changed);
            MetalWorldDevicePhysicsPass identityControl;
            identityControl.worldFingerprint=run.world().fingerprint();identityControl.environmentCount=8;
            identityControl.bodyCount=run.model().bodies.size();identityControl.nq=run.model().defaultQ.size();
            identityControl.taskStateProgramFingerprint=run.task().header().deviceStateProgramFingerprint;
            const auto mismatchProgram=mismatch.deviceProgram();
            require(!mismatchProgram.encode(mismatchProgram.context,identityControl),
                    "A changed food program could execute under the existing policy contract");
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
            const auto resetClosing=physicalTrial(run,authored,64,false,true,{},96);
            saveTrial(output/"reset-closing.json",resetClosing);
            require(equal(ordinary.state,partition.state),"Native resource state depends on submission partition");
            for (unsigned env=0;env<8;++env) {
                require(ordinary.state[env].accounting.x==45000 && ordinary.state[env].accounting.z==3 &&
                        ordinary.state[env].remaining[0].x==0 && ordinary.state[env].clock.x==448,
                        "Native mouth motion failed exact food accounting");
                require(missed.state[env].accounting.x==0,"Native mouth miss consumed food");
                require(std::abs(ordinary.utility[env]-4.5)<2.0e-6 && std::abs(reset.utility[env]-4.5)<2.0e-6 &&
                        missed.utility[env]==0,"Feeding utility was duplicated, lost or granted without food");
                require(std::abs(ordinary.sensed[env][0]-0.45f)<1e-6 && ordinary.sensed[env][1]==0 &&
                        ordinary.sensed[env][2]==0 && ordinary.sensed[env][3]==0,
                        "Nutrition observations disagree with depleted accepted state");
                require(reset.state[env].accounting.x==45000 && reset.state[env].remaining[0].x==0,
                        "Ordinary task reset refilled or erased resources");
                require(resetClosing.resetStepUtility[env]==0 &&
                        std::abs(resetClosing.utility[env]-4.5)<2e-6,
                        "A native task reset manufactured feeding utility from a closing gesture");
            }
            require(ordinary.physicalTerminations==0 && partition.physicalTerminations==0 &&
                    missed.physicalTerminations==0 && reset.physicalTerminations==0 && resetClosing.physicalTerminations==0,
                    "Physical termination invalidates the native foraging control");
            const auto spatialRun=crowRun(true);auto spatial=spatialConfig(authored);
            spatial.worldFingerprint=spatialRun.world().fingerprint();
            require(spatialRun.task().header().deviceStateProgramFingerprint==
                finiteResourceProgramFingerprint(spatial.program,spatial.patches,true,spatial.allocation),"Spatial resources do not bind their compiled task contract");
            require(spatialRun.task().header().schedule.w==run.task().header().schedule.w,"Resource authoring changed the physical task support semantics");
            std::vector<float> preA,preOther;
            const auto spatialA=physicalTrial(spatialRun,spatial,64,false,false,output/"spatial-physical.checkpoint",352,&preA);
            const auto spatialB=physicalTrial(spatialRun,spatial,8,false,false);
            const auto spatialReset=physicalTrial(spatialRun,spatial,64,false,true);
            auto otherSeed=spatial;otherSeed.allocationSeed++;
            const auto spatialOther=physicalTrial(spatialRun,otherSeed,64,false,false,{},352,&preOther);
            require(preA.size()==192u*8u*97u && preA.size()==preOther.size() &&
                std::memcmp(preA.data(),preOther.data(),preA.size()*sizeof(float))==0,
                "Changing hidden spatial allocation leaked into pre-intake actor observations");
            unsigned changedLanes=0;
            for (unsigned env=0;env<8;++env) if (spatialA.state[env].clock.w!=spatialOther.state[env].clock.w) ++changedLanes;
            require(changedLanes>0 && spatialOther.physicalTerminations==0,"Paired hidden-layout control did not change layouts or terminated physically");
            for (const auto& pair: {std::pair{"pre-intake-actor.f32",&preA},std::pair{"other-seed-pre-intake-actor.f32",&preOther}}) {
                std::ofstream data(output/pair.first,std::ios::binary);data.write(reinterpret_cast<const char*>(pair.second->data()),pair.second->size()*sizeof(float));
            }
            std::ofstream(output/"pre-intake-observation-control.json")<<"{\"status\":\"passed\",\"ticks\":192,\"lanes\":8,\"actor_features\":97,\"changed_hidden_layout_lanes\":"<<changedLanes<<",\"full_actor_stream_byte_exact\":true,\"seeds\":[2651109701,2651109702]}\n";
            require(equal(spatialA.state,spatialB.state),"Spatial resources changed under chunk partition");
            for (unsigned env=0;env<8;++env) {
                const auto& a=spatialA.state[env];const auto& b=spatialReset.state[env];
                require(std::memcmp(a.remaining,b.remaining,sizeof(a.remaining))==0 &&
                    std::memcmp(&a.accounting,&b.accounting,sizeof(a.accounting))==0 &&
                    a.clock.x==b.clock.x && a.clock.y==b.clock.y && a.clock.w==b.clock.w,
                    "Episode reset rerolled/refilled spatial food or reset its accepted lifetime clock");
            }
            require(spatialA.physicalTerminations==0 && spatialB.physicalTerminations==0 && spatialReset.physicalTerminations==0,"Spatial native control terminated physically");
            std::ofstream spatialEvidence(output/"spatial-physical.json");spatialEvidence<<"{\"status\":\"passed\",\"lanes\":[";
            for (unsigned env=0;env<8;++env) {
                const auto mask=mrResourceAllocationMask(spatial.allocationSeed,env,2,1);
                const auto& row=spatialA.state[env];const unsigned expected=mask==1 ? 40000 : 0;
                require(row.clock.w==mask && row.accounting.x==expected && row.accounting.y==expected &&
                    row.accounting.z==(mask==1 ? 2u : 0u) && row.clock.x==448 && row.clock.y==0 &&
                    row.remaining[0].x==(mask==1 ? 5000u : 0u) && row.remaining[0].y==(mask==2 ? 45000u : 0u),"Physical spatial food debit or hidden layout differs");
                require(std::abs(spatialA.utility[env]-expected*.0001)<2e-6 &&
                    spatialA.sensed[env][2]==0 && spatialA.sensed[env][3]==0,"Spatial nutrition leaked or reward violated accepted debit");
                if (env) spatialEvidence<<',';
                spatialEvidence<<"{\"mask\":"<<mask<<",\"consumed_ug\":"<<expected<<",\"central_remaining_ug\":"<<row.remaining[0].x
                    <<",\"distant_remaining_ug\":"<<row.remaining[0].y<<",\"ticks\":"<<row.clock.x<<'}';
            }
            spatialEvidence<<"],\"chunk8_64_exact\":true,\"reset_stock_allocation_and_clock_exact\":true,\"failed_steps\":0,\"physical_terminations\":0}\n";
            const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const std::string result="{\"schema\":\"numi.crow-foraging-admission.v1\",\"status\":\"passed\","
                "\"native_environments\":8,\"native_steps_per_trial\":448,\"native_trials\":9,\"spatial_allocation_controls\":true,"
                "\"consumed_micrograms_per_lane\":45000,\"bites_per_lane\":3,\"physical_terminations\":0,"
                "\"failed_physics_steps\":0,\"exact_partition_replay\":true,\"resource_only_restore\":true,"
                "\"accepted_intake_utility_per_lane\":4.5,\"nutrition_actor_contract\":true,\"c_api_accounting_controls\":true,"
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
