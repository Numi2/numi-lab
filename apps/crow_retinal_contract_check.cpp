#include "metalrobo/RunProgram.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace metalrobo;
void require(bool yes,const std::string& why) {if (!yes) throw std::runtime_error(why);}
RunManifest manifest(bool retinal=true) {
    RunManifest m;m.id="crow_retinal_contract_instrument";
    m.robot=makeAmericanCrowActiveHeadRobotPack();m.scene=makeBirdFlowAmericanCrowFlightScenePack();
    m.sensors.id="compiler_only_retinal_sensor_metadata";
    RetinalSensorProgram r;
    m.task=retinal ? makeAmericanCrowRetinalFeedingTaskPack(m.sensors.observation,m.reality.reset,r)
        : makeAmericanCrowFeedingStationTaskPack(m.sensors.observation,m.reality.reset);
    if (retinal) {
        VisualSensorProgram p;p.assets={{"compiler-only-not-a-runtime-asset.mrvpack","robot",std::string(64,'a'),1,1},
            {"compiler-only-not-a-runtime-ground.mrvpack","ground",std::string(64,'a'),2,2}};
        p.cameraParentBody="crow_head";p.cameraPosition={.018f,.026f,.012f,0};p.cameraOrientation={-.5f,.5f,-.5f,.5f};
        p.width=160;p.height=120;p.minimumVisiblePixels=1;p.verticalFieldOfViewDegrees=45;p.nominalRateHz=15;
        p.retinal=r;p.fingerprint=visualSensorProgramFingerprint(p);m.sensors.deviceVisual=p;
    }
    m.reality.id="canonical_contract_fixture";m.teacher.id="no_teacher";m.profile.id="metadata_and_rejection_only";
    m.profile.environmentCount=2;m.profile.controlSteps=1;m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;
    m.profile.minimumDifficultyBand=0;m.profile.maximumDifficultyBand=0;return m;
}
CompiledRun compile(const RunManifest& m) {
    CompiledRun r;const auto s=compileRun(m,r);require(s.succeeded(),s.element+": "+s.message);return r;
}
PolicyPack zeroPolicy(const CompiledTaskProgram& task) {
    PolicyPack p;p.id="retinal_contract_control_actor";
    const auto n=task.layout().actorObservationSize,a=task.layout().actionCount;
    p.observationMean.assign(n,0);p.observationInverseStandardDeviation.assign(n,1);
    p.layers={{.inputCount=n,.outputCount=a,.activation=PolicyActivation::identity,
        .weights=std::vector<float>(std::size_t(n)*a,0),.bias=std::vector<float>(a,0)}};
    bindPolicyPack(p,task);return p;
}
struct Callback {bool called=false,layout=false;};
MetalWorldDiagnostics rejectedRuntime(const CompiledRun& run,const CompiledPolicyProgram& policy,
    const std::string& shader,bool callback,Callback& reached,bool exactIdentity=true) {
    MetalWorldConfig config;config.metallibPath=shader;
    MetalWorldContext context(config);FiniteResourceContext food(makeAmericanCrowFeedingResourceConfig(run,2));
    MetalWorldStepConfig step;step.timestepSeconds=.02f;step.physicsSubsteps=4;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=run.task();step.policyProgram=policy;step.minimumDifficultyBand=step.maximumDifficultyBand=0;
    step.taskSeed=2651108601u;step.devicePhysicsProgram=food.deviceProgram();
    if (run.flappingWingProgram()) step.flappingWingProgram=*run.flappingWingProgram();
    if (callback) step.deviceObservationProgram={&reached,[](void* opaque,const MetalWorldDeviceObservationPass& pass) {
        auto& state=*static_cast<Callback*>(opaque);state.called=true;state.layout=pass.actorObservationSize==1251 && pass.taskStates && pass.actorObservations;
        return false; // Instrument intentionally refuses to supply an image.
    }};
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    for (unsigned env=0;env<2;++env) {
        q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());
        v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
        scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());
    }
    if (callback) step.deviceObservationProgram.observationFingerprint=exactIdentity ? run.task().observationFingerprint() : 1u;
    const std::array<unsigned,2> resets{};
    MetalWorldBatch batch{.environmentCount=2,.controlStepCount=1,.initialQ=q,.initialV=v,.resetMasks=resets,.initialSceneBodies=scene};
    MetalWorldSubmission ticket;const auto status=context.submit(run.world(),batch,step,ticket);
    require(!status.succeeded() && !ticket.valid(),"Rejected sensory contract created a live physics submission");return status;
}
}
int main(int argc,char** argv) {
    bool owns=false;
    try {
        require(argc==2 || argc==3,"Usage: metalrobo_crow_retinal_contract_check NEW_OUTPUT [OLD_METALLIB]");
        const std::filesystem::path out=argv[1];require(std::filesystem::create_directory(out),"Contract evidence directory must be new");owns=true;
        std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-retinal-contract-admission.v1","question":"Can native retinal semantics bind camera/timing/content to a new actor without changing legacy feeding contracts or admitting zero-filled vision?","unit":"compiler metadata and native pre-submission rejection controls; no rendered images or learned behavior","actor_features":1251,"critic_features":97,"actions":18,"legacy_task_abi":51,"retinal_task_abi":52,"predictions":{"malformed_manifest_cases_rejected":12,"sensor_semantic_changes_reject_old_policy":9,"relocated_asset_paths_change_fingerprint":false,"missing_observer_creates_submission":false,"stale_shader_reaches_observer":false,"mismatched_observer_reaches_execution":false,"current_shader_reaches_declared_layout":true},"asset_boundary":"compiler-only paths and declared hashes; no visual asset exists or is rendered in this instrument","stopping":"one base retinal/legacy compile, twelve malformed manifests, nine semantic mutations, one path relocation and four native rejection probes when old shader supplied; no retries or relaxed thresholds","limits":"typed contract only; camera-to-retina-to-actor implementation remains pending"})";
        const auto m=manifest();const auto run=compile(m),legacy=compile(manifest(false));
        require(run.task().layout().actorObservationSize==1251 && run.task().layout().criticObservationSize==97 && run.task().layout().actionCount==18,"Unexpected retinal actor contract");
        require(run.task().header().articulation.z==MR_TASK_PROGRAM_RETINAL_ABI_VERSION && legacy.task().header().articulation.z==MR_TASK_PROGRAM_ABI_VERSION,"Unexpected retinal/base executable ABI");
        require(legacy.task().layout().actorObservationSize==97 && run.world().fingerprint()==legacy.world().fingerprint() &&
            run.task().actionFingerprint()==legacy.task().actionFingerprint() &&
            run.task().header().deviceStateProgramFingerprint==legacy.task().header().deviceStateProgramFingerprint,"Retinal task altered mechanics, actions or the finite resource program");
        std::ofstream rows(out/"controls.jsonl");unsigned rejected=0;
        for (unsigned test=0;test<12;++test) {
            auto bad=m;auto& obs=bad.sensors.observation;auto& p=*bad.sensors.deviceVisual;
            if (test==0) bad.sensors.deviceVisual.reset();
            if (test==1) p.retinal.reset();
            if (test==2) --obs.retinalFeatureCount;
            if (test==3) obs.actorCurrent.pop_back();
            if (test==4) std::swap(obs.actorCurrent[obs.actorCurrent.size()-1],obs.actorCurrent[obs.actorCurrent.size()-2]);
            if (test==5) obs.actorCurrent.back().scale=2;
            if (test==6) obs.critic.push_back(obs.actorCurrent.back());
            if (test==7) p.retinal->rightCameraOrientation={0,0,0,0};
            if (test==8) p.retinal->sampleDenominator=0;
            if (test==9) p.retinal->latencyTicks=17;
            if (test==10) p.nominalRateHz=20;
            if (test==11) obs.deviceSensorProgramFingerprint=1;
            if (bad.sensors.deviceVisual) p.fingerprint=visualSensorProgramFingerprint(p);
            CompiledRun unused;const auto status=compileRun(bad,unused);require(!status.succeeded() && !unused.valid(),"Malformed retinal manifest accepted");
            ++rejected;rows<<"{\"malformed_case\":"<<test<<",\"element\":"<<std::quoted(status.element)<<",\"reason\":"<<std::quoted(status.message)<<"}\n";
        }
        require(rejected==12,"Not all malformed retinal cases were checked");
        const auto policy=zeroPolicy(run.task());CompiledPolicyProgram compiled;
        require(compilePolicyProgram(policy,run.task(),compiled).succeeded(),"Base retinal policy did not compile");
        for (unsigned test=0;test<9;++test) {
            auto changed=m;auto& p=*changed.sensors.deviceVisual;auto& r=*p.retinal;
            if (test==0) r.rightCameraPosition.z+=.005f;
            if (test==1) ++r.latencyTicks;
            if (test==2) {r.sampleNumerator=1;p.nominalRateHz=5;}
            if (test==3) p.assets[0].contentHash=std::string(64,'b');
            if (test==4) p.verticalFieldOfViewDegrees=60;
            if (test==5) r.maximumDepthMeters=6;
            if (test==6) r.rightCameraOrientation={0,0,0,1};
            if (test==7) {p.width=80;p.height=60;}
            if (test==8) p.cameraPosition.z+=.005f;
            p.fingerprint=visualSensorProgramFingerprint(p);const auto other=compile(changed);
            require(other.task().layout().actorObservationSize==1251 && other.task().observationFingerprint()!=run.task().observationFingerprint() &&
                other.task().fingerprint()!=run.task().fingerprint(),"Camera semantics changed without changing the exact actor/task identity");
            CompiledPolicyProgram forbidden;require(compilePolicyProgram(policy,other.task(),forbidden).status==PolicyCompileStatus::incompatibleContract,"Old retinal policy admitted under changed sensory semantics");
            rows<<"{\"semantic_change\":"<<test<<",\"old_policy_rejected\":true}\n";
        }
        auto relocated=m;for (auto& asset:relocated.sensors.deviceVisual->assets) asset.path="relocated/"+asset.path;
        relocated.sensors.deviceVisual->fingerprint=visualSensorProgramFingerprint(*relocated.sensors.deviceVisual);
        const auto moved=compile(relocated);require(moved.task().fingerprint()==run.task().fingerprint() && moved.sensorFingerprint()==run.sensorFingerprint(),"Asset paths changed content identity");
        Callback missing;const auto noObserver=rejectedRuntime(run,compiled,"",false,missing);
        require(!missing.called && noObserver.message.find("zero-filled")!=std::string::npos,"Missing observer was not rejected at native admission");
        Callback mismatch;const auto mismatchStatus=rejectedRuntime(run,compiled,"",true,mismatch,false);
        require(!mismatch.called && mismatchStatus.message.find("identity does not match")!=std::string::npos,"Mismatched observer identity admitted");
        Callback current;const auto currentStatus=rejectedRuntime(run,compiled,"",true,current);
        require(current.called && current.layout,"Current shader did not reach the declared retinal callback layout");
        std::string staleMessage;
        if (argc==3) {Callback stale;const auto status=rejectedRuntime(run,compiled,argv[2],true,stale);
            require(!stale.called && (status.message.find("ABI 52 capability")!=std::string::npos || status.message.find("metallib does not contain mr_locomotion_task_observe_history2")!=std::string::npos),"Stale shader admitted retinal semantics");staleMessage=status.message;}
        std::ofstream(out/"native-rejections.json")<<"{\"missing_observer\":"<<std::quoted(noObserver.message)<<",\"mismatched_observer\":"<<std::quoted(mismatchStatus.message)<<",\"current_refusing_observer\":"<<std::quoted(currentStatus.message)<<",\"stale_shader\":"<<std::quoted(staleMessage)<<"}\n";
        const std::string result=std::string(R"({"schema":"numi.crow-retinal-contract-evidence.v1","status":"passed","actor_features":1251,"critic_features":97,"actions":18,"retinal_abi":52,"legacy_abi":51,"malformed_manifests_rejected":12,"same_dimension_sensor_changes_reject_policy":9,"asset_path_relocation_exact":true,"missing_observer_rejected":true,"mismatched_observer_identity_rejected":true,"current_shader_reaches_declared_callback":true,"stale_shader_rejected":)")+(argc==3 ? "true" : "false")+R"(,"rendered_actor_images":false,"ecological_environment_qualified":false})";
        std::ofstream(out/"summary.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
    } catch(const std::exception& e) {
        if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.json")<<"{\"error\":"<<std::quoted(e.what())<<"}\n";
        std::cerr<<e.what()<<'\n';return 1;
    }
}
