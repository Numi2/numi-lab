#pragma once
#include "../matter/tools/human_resting_runtime.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_stand_gpu.h"

// The existing articulated owner remains responsible for gravity, MyoSim,
// tendons, equalities and unilateral support. Its existing post-validation
// reconciliation restores q/v/root/muscle state if this transaction fails.
// Matter remains responsible for circulation and its accepted clock. Both
// owners see the same final failure gate on the same borrowed command buffer.
class NumiHumanRestingCoupling {
public:
    numi::human::RestingRun physiology;
    numi::human::RespiratoryBrain brain;
    id<MTLBuffer> presentationBodies, presentationRespiration;
    id<MTLBuffer> presentationCandidateBodies, presentationCandidateRespiration;
    unsigned presentedStep = 0;
    std::string error;
private:
    id<MTLComputePipelineState> prepareWorld, validateBody, validateMatter, capture, publishFrame;
    id<MTLBuffer> bodyTemplate;
    std::uint64_t fingerprint;
public:
    NumiHumanRestingCoupling(const char* network,const char* parameters,float dt,
                            const metalrobo::EngineModel& model,std::uint64_t bodySourceIdentity,bool denseVascular=false):
        physiology(network,parameters,dt,denseVascular),brain(*physiology.respiration,physiology.world,parameters,bodySourceIdentity) {
        using numi::human::need;
        auto device=physiology.device;NSError* e=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&e];
        auto pipeline=[&](NSString* name) {
            auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
            need(p!=nil,std::string("resting transaction pipeline: ")+name.UTF8String);return p;
        };
        prepareWorld=pipeline(@"nm_human_resting_prepare_world");
        validateBody=pipeline(@"nm_human_resting_validate_body");
        validateMatter=pipeline(@"nm_human_resting_validate_matter");
        capture=pipeline(@"nm_human_resting_capture");
        publishFrame=pipeline(@"nm_human_resting_present_commit");
        std::vector<MRBodyStateGPU> initial(model.bodies.size());
        for(unsigned i=0;i<initial.size();++i) {
            initial[i].linearVelocityAndInverseMass.w=model.bodies[i].massAndInverseMass.y;
            initial[i].flagsAndIndices[0]=model.bodies[i].motionType;
            initial[i].flagsAndIndices[1]=model.bodies[i].articulationIndex;
            initial[i].flagsAndIndices[2]=i;
        }
        bodyTemplate=[device newBufferWithBytes:initial.data() length:initial.size()*sizeof(MRBodyStateGPU) options:MTLResourceStorageModeShared];
        presentationBodies=[device newBufferWithLength:bodyTemplate.length options:MTLResourceStorageModeShared];
        presentationRespiration=[device newBufferWithLength:sizeof(NMHumanRespirationState) options:MTLResourceStorageModeShared];
        presentationCandidateBodies=[device newBufferWithLength:bodyTemplate.length options:MTLResourceStorageModeShared];
        presentationCandidateRespiration=[device newBufferWithLength:sizeof(NMHumanRespirationState) options:MTLResourceStorageModeShared];
        need(bodyTemplate&&presentationBodies&&presentationRespiration&&presentationCandidateBodies&&presentationCandidateRespiration,
             "resting presentation allocation");
        fingerprint=brain.rootProgramIdentity^0x4e48524553543031ull;
        if(!fingerprint)fingerprint=1;
    }
    metalrobo::MetalNumanXTransactionProgram program() {
        metalrobo::MetalNumanXTransactionProgram p;
        p.context=this;p.encode=&encodeCallback;p.abort=&abortCallback;p.fingerprint=fingerprint;return p;
    }
    static void abortCallback(void* context,void* commandBuffer) noexcept {
        auto& self=*static_cast<NumiHumanRestingCoupling*>(context);
        self.physiology.runtime.cancel(commandBuffer);
    }
    static bool encodeCallback(void* context,const metalrobo::MetalNumanXTransactionPass& p) noexcept {
        auto& self=*static_cast<NumiHumanRestingCoupling*>(context);
        try{return self.encode(p);}catch(const std::exception& e){self.error=e.what();return false;}
    }
    bool encode(const metalrobo::MetalNumanXTransactionPass& p) {
        using namespace numi::matter;
        using numi::human::need;
        need(p.environmentCount==1&&p.timestepSeconds==physiology.runtime.timestepSeconds(),
             "resting body/physiology cadence differs");
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::beginStep)return true;
        auto cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
        const mr_uint4 d={p.stepIndex,static_cast<unsigned>(p.bodyCount),0,0};
        auto bridge=[&](id<MTLComputePipelineState> pipeline) {
            auto e=[cb computeCommandEncoder];need(e!=nil,"resting status encoder");
            [e setComputePipelineState:pipeline];[e setBytes:&d length:sizeof(d) atIndex:0];
            [e setBuffer:numi::human::buffer(p.standStatuses) offset:0 atIndex:1];
            [e setBuffer:physiology.statuses offset:0 atIndex:2];
            [e setBuffer:numi::human::buffer(physiology.runtime.statusBuffer()) offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];
        };
        EncodeRequest r;r.commandBuffer=p.commandBuffer;
        r.environmentStatuses=(__bridge void*)physiology.statuses;
        r.controlStep=p.stepIndex;r.physicsSubsteps=1;r.timestepSeconds=p.timestepSeconds;r.runAdaptiveTransfer=false;
        r.deferAcceptedControlCursorPublication=true;
        r.acceptedStepExtensionContext=physiology.respiration.get();r.encodeAcceptedStepExtension=&numi::human::Respiration::callback;
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::preDynamics) {
            bridge(prepareWorld);
            // Poses here are the exact current accepted q at t_n. Capture the
            // matching accepted respiratory state into a frame candidate.
            // Publication waits for the complete physical acceptance gate.
            auto e=[cb computeCommandEncoder];need(e!=nil,"resting accepted frame encoder");
            [e setComputePipelineState:capture];[e setBytes:&d length:sizeof(d) atIndex:0];
            [e setBuffer:numi::human::buffer(p.bodyPoses) offset:0 atIndex:1];
            [e setBuffer:bodyTemplate offset:0 atIndex:2];[e setBuffer:presentationCandidateBodies offset:0 atIndex:3];
            [e setBuffer:physiology.respiration->accepted offset:0 atIndex:4];
            [e setBuffer:presentationCandidateRespiration offset:0 atIndex:5];
            [e setBuffer:numi::human::buffer(p.standStatuses) offset:0 atIndex:6];
            [e dispatchThreads:MTLSizeMake(p.bodyCount,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];
            r.phase=EncodePhase::preDynamics;
        } else {
            need(p.phase==metalrobo::MetalNumanXTransactionPhase::postDynamics,"invalid resting transaction phase");
            bridge(validateBody);r.phase=EncodePhase::postCommit;
        }
        auto result=physiology.runtime.encode(r);need(result.encoded,result.message);
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::postDynamics) {
            bridge(validateMatter);
            AcceptedControlCursorGate gate;
            gate.statusCodes=p.standStatuses;gate.recordCount=1;
            gate.recordStrideBytes=sizeof(MRNumiHumanStandStatusGPU);
            gate.acceptedStatusCode=MR_NUMI_HUMAN_STAND_SUCCESS;
            const auto cursor=physiology.runtime.encodeAcceptedControlCursorPublication(
                p.commandBuffer,p.stepIndex,0,gate);
            need(cursor.encoded,cursor.message);
            auto e=[cb computeCommandEncoder];need(e!=nil,"resting accepted frame commit encoder");
            [e setComputePipelineState:publishFrame];[e setBytes:&d length:sizeof(d) atIndex:0];
            [e setBuffer:presentationCandidateBodies offset:0 atIndex:1];
            [e setBuffer:presentationCandidateRespiration offset:0 atIndex:2];
            [e setBuffer:presentationBodies offset:0 atIndex:3];
            [e setBuffer:presentationRespiration offset:0 atIndex:4];
            [e setBuffer:numi::human::buffer(p.standStatuses) offset:0 atIndex:5];
            [e dispatchThreads:MTLSizeMake(p.bodyCount,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];
        }
        return true;
    }
};
