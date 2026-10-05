#pragma once
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

namespace numi::human {
using namespace numi::matter;
inline void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
inline id<MTLBuffer> buffer(void* value){return (__bridge id<MTLBuffer>)value;}
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
    RespiratoryBrain(Respiration& physiology,const CompiledWorld& world,const char* config,std::uint64_t bodySourceIdentity=0):body(physiology) {
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
        if(bodySourceIdentity)mix(&bodySourceIdentity,sizeof(bodySourceIdentity));
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
    explicit RestingRun(const char* network,const char* configuration,float dt,bool vascularDense45=false) {
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
        config.environmentCount=1;config.captureEvents=false;config.captureDiagnostics=false;config.adaptiveTransfer=false;config.enableVascularDense45=vascularDense45;
        auto init=runtime.initialize(world,config);need(init.encoded,init.message);
        respiration=std::make_unique<Respiration>(device,world,configuration,runtime.timestepSeconds());
        std::cout<<"runtime="<<init.message<<" device="<<[device name].UTF8String<<" world_fingerprint="<<world.fingerprint<<" timestep_s="<<runtime.timestepSeconds()<<'\n';
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
