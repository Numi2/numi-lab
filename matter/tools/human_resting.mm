#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "human_respiration_parameters.hpp"
#include "metalrobo/engine_types.h"
#include <NumiBrainRespiratoryChemoreflexV1.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

using namespace numi::matter;
namespace {
void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
id<MTLBuffer> buffer(void* value){return (__bridge id<MTLBuffer>)value;}
struct Respiration {
    id<MTLDevice> device;
    id<MTLBuffer> accepted,candidate,excitation;
    id<MTLComputePipelineState> predict,exchange,resolve;
    NMHumanRespirationParameters parameters;
    NMHumanRespirationDispatch dispatch{};
    std::function<bool(AcceptedStepExtensionPhase,const AcceptedStepExtensionView&)> brain;
    explicit Respiration(id<MTLDevice> dev,const CompiledWorld& world,const char* config,float dt):device(dev) {
        parameters=readRespirationParameters(config,dt);
        dispatch={world.dispatch.environmentCount,45,0,0};
        NSError* error=nil;
        auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&error];
        need(library!=nil,"respiration library: "+std::string(error?error.localizedDescription.UTF8String:"missing"));
        auto pipeline=[&](NSString* name){
            id<MTLFunction> f=[library newFunctionWithName:name];
            auto result=f?[device newComputePipelineStateWithFunction:f error:&error]:nil;
            need(result!=nil,std::string("respiration pipeline: ")+name.UTF8String);return result;
        };
        predict=pipeline(@"nm_human_respiration_predict");
        exchange=pipeline(@"nm_human_respiration_exchange");
        resolve=pipeline(@"nm_human_respiration_resolve");
        auto s=initializeRespiration(parameters,world);
        accepted=[device newBufferWithLength:sizeof(s)*dispatch.environmentCount options:MTLResourceStorageModeShared];
        candidate=[device newBufferWithLength:accepted.length options:MTLResourceStorageModeShared];
        excitation=[device newBufferWithLength:sizeof(nm_float4)*dispatch.environmentCount options:MTLResourceStorageModeShared];
        need(accepted&&candidate&&excitation,"respiration allocation");
        for(unsigned e=0;e<dispatch.environmentCount;++e) static_cast<NMHumanRespirationState*>(accepted.contents)[e]=s;
        std::memcpy(candidate.contents,accepted.contents,accepted.length);
        std::memset(excitation.contents,0,excitation.length);
    }
    bool encode(AcceptedStepExtensionPhase phase,const AcceptedStepExtensionView& v) {
        if(v.microtickCount!=1||v.environmentCount!=dispatch.environmentCount||v.vascularStateStride!=45||
           v.microstepTimestepSeconds!=parameters.environment.w) return false;
        if(brain&&!brain(phase,v)) return false;
        id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)v.commandBuffer;
        if(phase==AcceptedStepExtensionPhase::microstepComplete) return true;
        auto enc=[cb computeCommandEncoder];if(!enc)return false;
        if(phase==AcceptedStepExtensionPhase::frameBegin) {
            [enc setComputePipelineState:predict];
            [enc setBytes:&parameters length:sizeof(parameters) atIndex:0];
            [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:1];
            [enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:candidate offset:0 atIndex:3];
            [enc setBuffer:excitation offset:0 atIndex:4];
        } else if(phase==AcceptedStepExtensionPhase::candidateReady) {
            [enc setComputePipelineState:exchange];
            [enc setBytes:&parameters length:sizeof(parameters) atIndex:0];
            [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:1];
            [enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:candidate offset:0 atIndex:3];
            [enc setBuffer:buffer(v.vascularAccepted) offset:0 atIndex:4];
            [enc setBuffer:buffer(v.vascularCandidate) offset:0 atIndex:5];
            [enc setBuffer:buffer(v.vascularUnknowns) offset:0 atIndex:6];
            [enc setBuffer:buffer(v.vascularConnections) offset:0 atIndex:7];
            [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:8];
            [enc setBuffer:buffer(v.vascularCompartments) offset:0 atIndex:9];
            [enc setBuffer:buffer(v.vascularElastance) offset:0 atIndex:10];
        } else if(phase==AcceptedStepExtensionPhase::frameComplete) {
            [enc setComputePipelineState:resolve];
            [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
            [enc setBuffer:candidate offset:0 atIndex:1];
            [enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:3];
        } else {[enc endEncoding];return false;}
        [enc dispatchThreads:MTLSizeMake(dispatch.environmentCount,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];return true;
    }
    static bool callback(void* ctx,AcceptedStepExtensionPhase phase,const AcceptedStepExtensionView& v) {
        return static_cast<Respiration*>(ctx)->encode(phase,v);
    }
};

struct RespiratoryBrain {
    id<MTLBuffer> input,accepted,candidate,output;
    id<MTLComputePipelineState> observe,advance,deliver,resolve;
    NBNumiRespiratoryChemoreflexParametersV1 parameters{};
    Respiration& body;
    std::uint64_t rootProgramIdentity;
    double interventionStart=-1,interventionEnd=-1;
    float interventionScale=1;
    std::uint64_t root(unsigned step)const {
        return (rootProgramIdentity^(std::uint64_t(step)*0x9e3779b97f4a7c15ull))|1ull;
    }
    RespiratoryBrain(Respiration& physiology,const CompiledWorld& world,const char* config):body(physiology) {
        NSDictionary* json=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(config)] options:0 error:nil];
        NSDictionary* values=json[@"brain_parameters"];
        need([values isKindOfClass:NSDictionary.class],"reference Brain parameters missing");
        const std::array<NSString*,16> keys={@"restingMinuteVentilationLitresPerMinute",@"CO2ResponseSlopeLitresPerMinutePerMmHg",@"centralCO2Fraction",@"centralTimeConstantSeconds",@"peripheralTimeConstantSeconds",@"hypoxicReferencePaO2MillimetersMercury",@"hypoxicScalePaO2MillimetersMercury",@"hypoxicIncrementAtPaO2FortyLitresPerMinute",@"restingFrequencyBreathsPerMinute",@"restingTidalVolumeLitres",@"diaphragmExcitationGain",@"intercostalExcitationGain",@"inspirationFractionOfCycle",@"minimumMinuteVentilationLitresPerMinute",@"maximumMinuteVentilationLitresPerMinute",@"reserved"};
        for(unsigned i=0;i<keys.size();++i) {
            id value=values[keys[i]];
            need([value isKindOfClass:NSNumber.class]&&std::isfinite([value doubleValue]),"invalid Brain parameter");
            reinterpret_cast<float*>(&parameters)[i]=[value floatValue];
        }
        rootProgramIdentity=world.fingerprint;
        auto mix=[&](const void* bytes,std::size_t count) {
            auto p=static_cast<const unsigned char*>(bytes);
            for(std::size_t i=0;i<count;++i){rootProgramIdentity^=p[i];rootProgramIdentity*=1099511628211ull;}
        };
        mix(&body.parameters,sizeof(body.parameters));mix(&parameters,sizeof(parameters));
        NSData* libraryBytes=[NSData dataWithContentsOfFile:@(NUMI_HUMAN_RESPIRATION_METALLIB)];
        mix(libraryBytes.bytes,libraryBytes.length);
        NSError* error=nil;
        auto library=[body.device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&error];
        auto pipeline=[&](NSString* name){auto p=[body.device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];need(p!=nil,"Brain pipeline unavailable");return p;};
        observe=pipeline(@"nm_human_respiration_brain_observe");
        advance=pipeline(@"advance_numi_respiratory_chemoreflex_v1");
        deliver=pipeline(@"nm_human_respiration_brain_deliver");
        resolve=pipeline(@"nm_human_respiration_brain_resolve");
        const auto count=body.dispatch.environmentCount;
        auto allocate=[&](std::size_t n){auto b=[body.device newBufferWithLength:n*count options:MTLResourceStorageModeShared];need(b!=nil,"Brain buffer allocation");std::memset(b.contents,0,b.length);return b;};
        input=allocate(sizeof(NBNumiRespiratoryChemoreflexInputV1));
        accepted=allocate(sizeof(NBNumiRespiratoryChemoreflexStateV1));candidate=allocate(sizeof(NBNumiRespiratoryChemoreflexStateV1));
        output=allocate(sizeof(NBNumiRespiratoryChemoreflexOutputV1));
        for(unsigned e=0;e<count;++e) {
            auto& s=static_cast<NBNumiRespiratoryChemoreflexStateV1*>(accepted.contents)[e];
            s.acceptedRootFingerprint=root(0);s.flags=NB_NUMI_RESPIRATORY_RECORD_VALID;
        }
        body.brain=[this](auto phase,const auto& v){return encode(phase,v);};
    }
    ~RespiratoryBrain(){body.brain={};}
    bool encode(AcceptedStepExtensionPhase phase,const AcceptedStepExtensionView& v) {
        if(phase!=AcceptedStepExtensionPhase::frameBegin&&phase!=AcceptedStepExtensionPhase::frameComplete)return true;
        const double dt=body.parameters.environment.w;
        NMHumanRespirationBrainDispatch d{};
        // Scheduling timestamps are rounded from Matter's exact fixed binary32
        // cadence, never accumulated in floating point by the controller.
        d.sourceTimeMicroseconds=std::llround(double(v.controlStep)*dt*1e6);
        d.targetTimeMicroseconds=std::llround(double(v.controlStep+1u)*dt*1e6);
        d.sourceRootIdentity=root(v.controlStep);d.targetRootIdentity=root(v.controlStep+1);
        d.sourceStep=v.controlStep;d.environmentCount=v.environmentCount;
        const double t=double(v.controlStep)*dt;
        d.driveScale=t>=interventionStart&&t<interventionEnd?interventionScale:1;
        auto cb=(__bridge id<MTLCommandBuffer>)v.commandBuffer;
        auto enc=[cb computeCommandEncoder];if(!enc)return false;
        auto launch=[&](){[enc dispatchThreads:MTLSizeMake(v.environmentCount,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];};
        if(phase==AcceptedStepExtensionPhase::frameBegin) {
            [enc setComputePipelineState:observe];[enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBuffer:body.accepted offset:0 atIndex:1];[enc setBuffer:input offset:0 atIndex:2];launch();
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:advance];[enc setBytes:&parameters length:sizeof(parameters) atIndex:0];
            [enc setBuffer:input offset:0 atIndex:1];[enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:candidate offset:0 atIndex:3];[enc setBuffer:output offset:0 atIndex:4];launch();
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:deliver];[enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBuffer:output offset:0 atIndex:1];[enc setBuffer:body.excitation offset:0 atIndex:2];launch();
        }else {
            [enc setComputePipelineState:resolve];[enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBuffer:candidate offset:0 atIndex:1];[enc setBuffer:accepted offset:0 atIndex:2];
            [enc setBuffer:buffer(v.matterStatuses) offset:0 atIndex:3];[enc setBuffer:body.candidate offset:0 atIndex:4];launch();
        }
        [enc endEncoding];return true;
    }
};

struct RestingRun {
    CompiledWorld world;
    Runtime runtime;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> statuses;
    std::unique_ptr<Respiration> respiration;
    explicit RestingRun(const char* network,const char* configuration,float dt) {
        VascularNetworkSource n;std::string error;
        need(readHumanPhysiologyNetwork(network,n,&error),error);
        WorldSource source;source.environmentCount=1;source.frameTimestep=dt;
        source.gravity={0,0,0};source.vascular=n;
        source.mixedSolver.relativeResidual=1.e-7;
        source.mixedSolver.newtonIterations=12;source.mixedSolver.fgmresIterations=64;
        auto compiled=compileWorld(source,{.maximumRateExponent=0});
        for(const auto& d:compiled.diagnostics)if(d.severity==Diagnostic::Severity::error)error+=d.message+"; ";
        need(compiled.succeeded(),error);world=std::move(compiled.world);
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple GPU required");
        queue=[device newCommandQueue];statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        RuntimeConfiguration config;config.metallib=NUMI_MATTER_METALLIB;
        config.environmentCount=1;config.captureEvents=false;config.captureDiagnostics=false;config.adaptiveTransfer=false;
        auto init=runtime.initialize(world,config);need(init.encoded,init.message);
        respiration=std::make_unique<Respiration>(device,world,configuration,runtime.timestepSeconds());
        std::cout<<"device="<<[device name].UTF8String<<" world_fingerprint="<<world.fingerprint<<" timestep_s="<<runtime.timestepSeconds()<<'\n';
    }
    double batch(unsigned first,unsigned count,bool expectRejected=false) {
        *static_cast<MRMetalWorldStatusGPU*>(statuses.contents)={};
        auto cb=[queue commandBuffer];need(cb!=nil,"command buffer");
        for(unsigned step=first;step<first+count;++step) {
            EncodeRequest r;r.commandBuffer=(__bridge void*)cb;r.environmentStatuses=(__bridge void*)statuses;
            r.controlStep=step;r.physicsSubsteps=1;r.timestepSeconds=runtime.timestepSeconds();r.runAdaptiveTransfer=false;
            r.acceptedStepExtensionContext=respiration.get();r.encodeAcceptedStepExtension=&Respiration::callback;
            r.phase=EncodePhase::preDynamics;auto result=runtime.encode(r);need(result.encoded,result.message);
            r.phase=EncodePhase::postCommit;result=runtime.encode(r);need(result.encoded,result.message);
        }
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal execution failure");
        auto* state=static_cast<const NMHumanRespirationState*>(respiration->accepted.contents);
        auto* staged=static_cast<const NMHumanRespirationState*>(respiration->candidate.contents);
        if(expectRejected) {
            need(state->status.x==first&&staged->status.w!=0,"forced physical rejection did not preserve accepted count");
        }else if(state->status.x!=first+count) {
            const auto failed=runtime.snapshot();
            const auto status=failed.statuses.empty()?NMMatterStatusGPU{}:failed.statuses.front();
            throw std::runtime_error("rejected physical/respiratory step at accepted="+std::to_string(state->status.x)+
                " respiratory_code="+std::to_string(staged->status.w)+" candidate_step="+std::to_string(staged->status.x)+
                " matter_code="+std::to_string(status.code)+" failing_index="+std::to_string(status.failingIndex)+
                " residual="+std::to_string(status.diagnostics.z));
        }
        return cb.GPUEndTime-cb.GPUStartTime;
    }
};
}

int main(int argc,const char* argv[]) {@autoreleasepool {try {
    need(argc>=4,"usage: numi-human-resting NETWORK.json RESPIRATION.json OUTPUT.csv [--steps N] [--dt SECONDS] [--instrument-excitation D I]");
    unsigned steps=100;float dt=.001f;float excitationD=0,excitationI=0;bool instrument=false;
    double interventionStart=-1,interventionEnd=-1;float interventionScale=1;
    bool verifyTransaction=false;
    for(int a=4;a<argc;++a) {
        std::string key=argv[a];
        if(key=="--steps"&&a+1<argc)steps=unsigned(std::stoul(argv[++a]));
        else if(key=="--dt"&&a+1<argc)dt=std::stof(argv[++a]);
        else if(key=="--instrument-excitation"&&a+2<argc) {instrument=true;excitationD=std::stof(argv[++a]);excitationI=std::stof(argv[++a]);}
        else if(key=="--drive-intervention"&&a+3<argc) {interventionStart=std::stod(argv[++a]);interventionEnd=std::stod(argv[++a]);interventionScale=std::stof(argv[++a]);}
        else if(key=="--verify-transaction")verifyTransaction=true;
        else need(false,"unknown or incomplete argument "+key);
    }
    need(steps>0&&steps<=1000000,"bounded step count required");
    need(std::isfinite(interventionStart)&&std::isfinite(interventionEnd)&&std::isfinite(interventionScale)&&interventionScale>=0&&interventionScale<=2,"invalid respiratory intervention");
    std::cout<<std::setprecision(12)<<std::unitbuf;
    const auto start=std::chrono::steady_clock::now();
    RestingRun run(argv[1],argv[2],dt);
    std::unique_ptr<RespiratoryBrain> brain;
    if(instrument)*static_cast<nm_float4*>(run.respiration->excitation.contents)={excitationD,excitationI,0,0};
    else {
        brain=std::make_unique<RespiratoryBrain>(*run.respiration,run.world,argv[2]);
        brain->interventionStart=interventionStart;brain->interventionEnd=interventionEnd;brain->interventionScale=interventionScale;
    }
    if(verifyTransaction) {
        need(brain!=nullptr,"transaction check requires the coupled Brain controller");
        run.batch(0,1);
        const auto before=run.runtime.snapshot();need(before.available,before.message);
        const auto respiratoryBefore=*static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
        const auto brainBefore=*static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(brain->accepted.contents);
        run.respiration->dispatch.reject=1;run.batch(1,1,true);
        const auto rejected=run.runtime.snapshot();need(rejected.available,rejected.message);
        need(before.vascularState.size()==rejected.vascularState.size()&&
             std::memcmp(before.vascularState.data(),rejected.vascularState.data(),before.vascularState.size()*sizeof(nm_float4))==0&&
             std::memcmp(before.vascularClock.data(),rejected.vascularClock.data(),before.vascularClock.size()*sizeof(NMVascularClockGPU))==0,
             "rejected respiration advanced circulation or its clock");
        need(std::memcmp(&respiratoryBefore,run.respiration->accepted.contents,sizeof(respiratoryBefore))==0&&
             std::memcmp(&brainBefore,brain->accepted.contents,sizeof(brainBefore))==0,
             "rejected body advanced respiration/controller history");
        run.respiration->dispatch.reject=0;run.batch(1,1);
        const auto accepted=run.runtime.snapshot();need(accepted.available,accepted.message);
        const auto respiratoryAccepted=*static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
        const auto brainAccepted=*static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(brain->accepted.contents);
        const auto restored=run.runtime.restore(before);need(restored.encoded,restored.message);
        std::memcpy(run.respiration->accepted.contents,&respiratoryBefore,sizeof(respiratoryBefore));
        std::memcpy(brain->accepted.contents,&brainBefore,sizeof(brainBefore));
        run.batch(1,1);
        const auto replay=run.runtime.snapshot();need(replay.available,replay.message);
        need(std::memcmp(accepted.vascularState.data(),replay.vascularState.data(),accepted.vascularState.size()*sizeof(nm_float4))==0&&
             std::memcmp(accepted.vascularClock.data(),replay.vascularClock.data(),accepted.vascularClock.size()*sizeof(NMVascularClockGPU))==0&&
             std::memcmp(&respiratoryAccepted,run.respiration->accepted.contents,sizeof(respiratoryAccepted))==0&&
             std::memcmp(&brainAccepted,brain->accepted.contents,sizeof(brainAccepted))==0,
             "coupled respiratory/circulatory/controller replay differs");
        std::cout<<"coupled_rejection=bitwise_unchanged circulation_clock=unchanged controller_history=unchanged accepted_replay=bitwise\n";
        return 0;
    }
    std::ofstream trace(argv[3]);need(trace.good(),"trace path is not writable");
    trace<<std::setprecision(12)<<"time_s,lung_volume_ml,airflow_ml_s,alveolar_pa,pleural_pa,diaphragm_mm,rib_mm,PaO2_mmhg,PaCO2_mmhg,SaO2,oxygen_balance_error_stpd_ml,co2_balance_error_stpd_ml,breaths,tidal_ml,lv_mmhg,rv_mmhg,aorta_mmhg,pulmonary_artery_mmhg,lv_ml,rv_ml,blood_ml,blood_error_ml,aortic_ejected_ml,pulmonary_ejected_ml,complete_filling_ejection_cycles,last_lv_stroke_ml\n";
    double gpuSeconds=0;
    for(unsigned first=0;first<steps;) {@autoreleasepool {
        unsigned count=std::min(16u,steps-first);gpuSeconds+=run.batch(first,count);first+=count;
        const auto s=*static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
        trace<<double(s.status.x)*run.runtime.timestepSeconds()<<','<<s.mechanics.x*1e6<<','<<s.mechanics.w*1e6<<','<<s.mechanics.y<<','<<s.mechanics.z<<','
             <<s.motion.x/run.respiration->parameters.geometry.z*1000<<','<<s.motion.y/run.respiration->parameters.geometry.w*1000<<','
             <<s.observation.x<<','<<s.observation.y<<','<<s.observation.z<<','<<s.gasBudget.z*1e6<<','<<s.gasBudget.w*1e6<<','<<s.status.y<<','<<s.breath.z*1e6<<','
             <<s.cardiacPressure.x/133.322387415<<','<<s.cardiacPressure.y/133.322387415<<','<<s.cardiacPressure.z/133.322387415<<','<<s.cardiacPressure.w/133.322387415<<','
             <<s.circulation.y*1e6<<','<<s.circulation.z*1e6<<','<<s.circulation.x*1e6<<','<<s.circulation.w*1e6<<','<<s.cardiacFlow.x*1e6<<','<<s.cardiacFlow.y*1e6<<','<<s.cardiacStatus.x<<','<<s.cardiacFlow.w*1e6<<'\n';
        if(first%1000==0)std::cout<<"accepted="<<first<<" time_s="<<first*double(run.runtime.timestepSeconds())<<" PaCO2="<<s.observation.y<<'\n';
    }}
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double simulated=steps*double(run.runtime.timestepSeconds());
    std::cout<<"accepted_steps="<<steps<<" simulated_s="<<simulated<<" wall_s="<<wall<<" gpu_s="<<gpuSeconds<<" real_time_factor="<<simulated/wall<<" Brain_control="<<!instrument<<" no_anatomy_or_resting_claim=true\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"human_resting=failed reason="<<e.what()<<'\n';return 1;}}}
