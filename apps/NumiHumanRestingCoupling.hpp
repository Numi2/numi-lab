#pragma once
#include "../matter/tools/human_resting_runtime.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"

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
    id<MTLBuffer> presentationCommonCoordinates, presentationCandidateCommonCoordinates;
    id<MTLBuffer> acceptedCommonCoordinates, presentationFrameCommonCoordinates;
    bool commonAcceptedCoordinatesInitialized = false;
    unsigned presentedStep = 0;
    std::string error;
    id<MTLBuffer> commonFailureCaptureBuffer=nil;
private:
    id<MTLComputePipelineState> prepareWorld, validateBody, validateMatter, capture, publishFrame;
    id<MTLComputePipelineState> latchCommonFailure=nil;
    bool commonFailureCaptureEnabled=false;
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
        latchCommonFailure=pipeline(@"nm_human_resting_latch_common_failure");
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
        const MRHumanRestingCommonCoordinatesGPU emptyCommonCoordinates{};
        presentationCommonCoordinates=[device newBufferWithBytes:&emptyCommonCoordinates length:sizeof(emptyCommonCoordinates) options:MTLResourceStorageModeShared];
        presentationCandidateCommonCoordinates=[device newBufferWithBytes:&emptyCommonCoordinates length:sizeof(emptyCommonCoordinates) options:MTLResourceStorageModeShared];
        acceptedCommonCoordinates=[device newBufferWithBytes:&emptyCommonCoordinates length:sizeof(emptyCommonCoordinates) options:MTLResourceStorageModeShared];
        presentationFrameCommonCoordinates=[device newBufferWithBytes:&emptyCommonCoordinates length:sizeof(emptyCommonCoordinates) options:MTLResourceStorageModeShared];
        const char* failureCaptureSetting=std::getenv("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT");
        commonFailureCaptureEnabled=failureCaptureSetting&&failureCaptureSetting[0];
        if(commonFailureCaptureEnabled) {
            const MRHumanRestingCommonFailureGPU emptyFailure{};
            commonFailureCaptureBuffer=[device newBufferWithBytes:&emptyFailure length:sizeof(emptyFailure) options:MTLResourceStorageModeShared];
        }
        need(bodyTemplate&&presentationBodies&&presentationRespiration&&presentationCandidateBodies&&presentationCandidateRespiration&&
             presentationCommonCoordinates&&presentationCandidateCommonCoordinates&&acceptedCommonCoordinates&&presentationFrameCommonCoordinates&&
             (!commonFailureCaptureEnabled||commonFailureCaptureBuffer),
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
        const mr_uint4 d={p.stepIndex,static_cast<unsigned>(p.bodyCount),physiology.respiration->commonGeometryGateEnabled?1u:0u,0};
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
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::preDynamics) {
            auto& respiration=*physiology.respiration;
            if(respiration.commonGeometryGateEnabled) {
                if(!commonAcceptedCoordinatesInitialized) {
                    need(respiration.commonCoordinatesSolvePipeline&&respiration.commonCoordinateStatusPipeline&&
                         respiration.commonGeometryParameters&&respiration.commonGeometryBoxes,
                         "initial accepted common coordinates lack the registered solver");
                    auto solve=[cb computeCommandEncoder];need(solve!=nil,"initial accepted common-coordinate solver encoder");
                    [solve setComputePipelineState:respiration.commonCoordinatesSolvePipeline];
                    [solve setBuffer:presentationCandidateRespiration offset:0 atIndex:0];
                    [solve setBuffer:respiration.commonGeometryParameters offset:0 atIndex:1];
                    [solve setBuffer:respiration.commonGeometryBoxes offset:0 atIndex:2];
                    [solve setBuffer:acceptedCommonCoordinates offset:0 atIndex:3];
                    [solve dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                    [solve setComputePipelineState:respiration.commonCoordinateStatusPipeline];
                    [solve setBuffer:acceptedCommonCoordinates offset:0 atIndex:0];
                    [solve setBuffer:numi::human::buffer(physiology.runtime.statusBuffer()) offset:0 atIndex:1];
                    [solve dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                    [solve endEncoding];
                    commonAcceptedCoordinatesInitialized=true;
                }
            }
        }
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::postDynamics) {
            bridge(validateMatter);
            if(commonFailureCaptureEnabled) {
                auto latch=[cb computeCommandEncoder];need(latch!=nil,"common failure latch encoder");
                [latch setComputePipelineState:latchCommonFailure];[latch setBytes:&d length:sizeof(d) atIndex:0];
                [latch setBuffer:numi::human::buffer(p.standStatuses) offset:0 atIndex:1];
                [latch setBuffer:physiology.statuses offset:0 atIndex:2];
                [latch setBuffer:physiology.respiration->candidate offset:0 atIndex:3];
                [latch setBuffer:physiology.respiration->accepted offset:0 atIndex:4];
                [latch setBuffer:presentationCandidateCommonCoordinates offset:0 atIndex:5];
                [latch setBuffer:brain.input offset:0 atIndex:6];
                [latch setBuffer:brain.accepted offset:0 atIndex:7];
                [latch setBuffer:brain.candidate offset:0 atIndex:8];
                [latch setBuffer:brain.output offset:0 atIndex:9];
                [latch setBuffer:physiology.respiration->excitation offset:0 atIndex:10];
                [latch setBuffer:commonFailureCaptureBuffer offset:0 atIndex:11];
                [latch setBuffer:physiology.respiration->commonGeometryParameters offset:0 atIndex:12];
                [latch setBuffer:physiology.respiration->commonGeometryBoxes offset:0 atIndex:13];
                [latch dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[latch endEncoding];
            }
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
            [e setBuffer:presentationFrameCommonCoordinates offset:0 atIndex:6];
            [e setBuffer:presentationCommonCoordinates offset:0 atIndex:7];
            [e setBuffer:presentationCandidateCommonCoordinates offset:0 atIndex:8];
            [e setBuffer:acceptedCommonCoordinates offset:0 atIndex:9];
            [e setBuffer:numi::human::buffer(physiology.runtime.statusBuffer()) offset:0 atIndex:10];
            [e dispatchThreads:MTLSizeMake(p.bodyCount,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];
        }
        return true;
    }
};
