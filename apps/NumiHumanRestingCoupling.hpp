#pragma once
#include "../matter/tools/human_resting_runtime.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>

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
    // Optional probe-only raw snapshot for an in-flight multi-root rejection.
    // The public accepted-state observer is intentionally unavailable when a
    // submission fails before publishing a new resident-generation identity.
    id<MTLBuffer> transactionProbeRoots=nil, transactionProbeQ=nil;
    id<MTLBuffer> transactionProbeV=nil, transactionProbeMuscles=nil;
    // Retained source buffers captured by the callback. The transaction
    // callback runs before Human/Matter physical prepare/apply can restore a
    // rejected candidate, so the probe blits these only after run() completes.
    __strong id<MTLBuffer> transactionProbeRawRoots=nil, transactionProbeRawQ=nil;
    __strong id<MTLBuffer> transactionProbeRawV=nil, transactionProbeRawMuscles=nil;
    std::size_t transactionProbeRawRootCount=0u, transactionProbeRawQCount=0u;
    std::size_t transactionProbeRawVCount=0u, transactionProbeRawMuscleCount=0u;
    std::uint32_t transactionProbeCaptureControlStep=MR_INVALID_INDEX;
    bool transactionProbeCaptured=false;
    std::uint32_t transactionProbeCapturedStep=MR_INVALID_INDEX;
    bool commonAcceptedCoordinatesInitialized = false;
    unsigned presentedStep = 0;
    std::string error;
    id<MTLBuffer> commonFailureCaptureBuffer=nil;
private:
    id<MTLComputePipelineState> prepareWorld, validateBody, validateMatter, capture, publishFrame, publishTerminalFrame;
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
        publishTerminalFrame=pipeline(@"nm_human_resting_terminal_present_commit");
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
    void initializeInitialCommonCoordinates() {
        using numi::human::need;
        auto& respiration=*physiology.respiration;
        if(!respiration.commonGeometryGateEnabled)return;
        if(commonAcceptedCoordinatesInitialized)return;
        need(respiration.commonCoordinatesSolvePipeline&&respiration.commonCoordinateStatusPipeline&&
             respiration.commonGeometryParameters&&respiration.commonGeometryBoxes&&respiration.accepted&&
             acceptedCommonCoordinates&&presentationCommonCoordinates&&presentationCandidateCommonCoordinates&&
             presentationFrameCommonCoordinates,
             "initial accepted common coordinates lack the registered solver or buffers");
        need(respiration.accepted.length>=sizeof(NMHumanRespirationState)&&
             respiration.commonGeometryParameters.length==sizeof(MRHumanRestingCommonFieldGPU)&&
             respiration.commonGeometryBoxes.length%sizeof(MRHumanRestingCommonCoordinateBoxGPU)==0,
             "initial accepted common-coordinate buffers have invalid sizes");
        const auto* parameters=static_cast<const MRHumanRestingCommonFieldGPU*>(
            respiration.commonGeometryParameters.contents);
        const auto boxCount=respiration.commonGeometryBoxes.length/sizeof(MRHumanRestingCommonCoordinateBoxGPU);
        need(parameters->countsAndFlags.x==boxCount&&boxCount>0,
             "initial common-coordinate domain boxes do not match their admitted count");
        id<MTLCommandQueue> queue=[physiology.device newCommandQueue];
        need(queue!=nil,"initial accepted common-coordinate command queue is unavailable");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        need(command!=nil,"initial accepted common-coordinate command buffer is unavailable");
        auto encoder=[command computeCommandEncoder];
        need(encoder!=nil,"initial accepted common-coordinate solver encoder is unavailable");
        [encoder setComputePipelineState:respiration.commonCoordinatesSolvePipeline];
        [encoder setBuffer:respiration.accepted offset:0 atIndex:0];
        [encoder setBuffer:respiration.commonGeometryParameters offset:0 atIndex:1];
        [encoder setBuffer:respiration.commonGeometryBoxes offset:0 atIndex:2];
        [encoder setBuffer:acceptedCommonCoordinates offset:0 atIndex:3];
        const NSUInteger initialSolverWidth=respiration.commonCoordinatesSolvePipeline.threadExecutionWidth;
        need(initialSolverWidth>=7u,"common-coordinate SIMD width is below seven rows");
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(initialSolverWidth,1,1)];
        [encoder setComputePipelineState:respiration.commonCoordinateStatusPipeline];
        [encoder setBuffer:acceptedCommonCoordinates offset:0 atIndex:0];
        [encoder setBuffer:numi::human::buffer(physiology.runtime.statusBuffer()) offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        auto blit=[command blitCommandEncoder];
        need(blit!=nil,"initial accepted common-coordinate buffer-copy encoder is unavailable");
        [blit copyFromBuffer:acceptedCommonCoordinates sourceOffset:0 toBuffer:presentationCommonCoordinates destinationOffset:0 size:sizeof(MRHumanRestingCommonCoordinatesGPU)];
        [blit copyFromBuffer:acceptedCommonCoordinates sourceOffset:0 toBuffer:presentationCandidateCommonCoordinates destinationOffset:0 size:sizeof(MRHumanRestingCommonCoordinatesGPU)];
        [blit copyFromBuffer:acceptedCommonCoordinates sourceOffset:0 toBuffer:presentationFrameCommonCoordinates destinationOffset:0 size:sizeof(MRHumanRestingCommonCoordinatesGPU)];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             std::string("initial accepted common-coordinate GPU solve failed: ")+
                 (command.error?command.error.localizedDescription.UTF8String:"command buffer did not complete"));
        const auto solved=*static_cast<const MRHumanRestingCommonCoordinatesGPU*>(acceptedCommonCoordinates.contents);
        const std::array<float,7> coordinates{{solved.first.x,solved.first.y,solved.first.z,solved.first.w,
            solved.second.x,solved.second.y,solved.second.z}};
        const bool finiteCoordinates=std::all_of(coordinates.begin(),coordinates.end(),
            [](float value){return std::isfinite(value);});
        need(solved.status.x==0u&&solved.status.z<boxCount&&finiteCoordinates&&
             std::isfinite(solved.diagnostics.x)&&std::isfinite(parameters->solver.x)&&parameters->solver.x>0.0f&&
             solved.diagnostics.x<=parameters->solver.x,
             "initial accepted common-coordinate solve failed status/domain/residual validation");
        need(presentationCommonCoordinates.length==sizeof(solved)&&
             presentationCandidateCommonCoordinates.length==sizeof(solved)&&
             presentationFrameCommonCoordinates.length==sizeof(solved)&&
             std::memcmp(presentationCommonCoordinates.contents,&solved,sizeof(solved))==0&&
             std::memcmp(presentationCandidateCommonCoordinates.contents,&solved,sizeof(solved))==0&&
             std::memcmp(presentationFrameCommonCoordinates.contents,&solved,sizeof(solved))==0,
             "initial accepted common-coordinate GPU copies differ from the validated solution");
        commonAcceptedCoordinatesInitialized=true;
        std::cout<<"resting_common_initialization=accepted_t0 solver_status="<<solved.status.x
            <<" iterations="<<solved.status.y<<" domain_box="<<solved.status.z
            <<" normalized_residual="<<solved.diagnostics.x
            <<" coordinates=["<<coordinates[0]<<','<<coordinates[1]<<','<<coordinates[2]<<','<<coordinates[3]
            <<','<<coordinates[4]<<','<<coordinates[5]<<','<<coordinates[6]<<"] physical_steps_advanced=0\n";
    }
    void publishTerminalSnapshot(
        const std::uint32_t expectedTerminalStep,
        const std::span<const MRArticulatedBodyPoseGPU> acceptedBodyPoses,
        const MRNumiHumanStandStatusGPU& acceptedStandStatus
    ) {
        using numi::human::need;
        auto& respiration=*physiology.respiration;
        need(expectedTerminalStep>0u&&acceptedStandStatus.completedSteps==expectedTerminalStep&&
             acceptedStandStatus.code==MR_NUMI_HUMAN_STAND_SUCCESS&&
             respiration.accepted&&respiration.accepted.length>=sizeof(NMHumanRespirationState)&&
             respiration.accepted.contents,
             "terminal frame requires the successful exact-N accepted physical state");
        const NMHumanRespirationState acceptedRespirationSnapshot=
            *static_cast<const NMHumanRespirationState*>(respiration.accepted.contents);
        need(acceptedRespirationSnapshot.status.x==expectedTerminalStep&&
             acceptedRespirationSnapshot.status.w==0u,
             "terminal frame requires the exact-N accepted respiratory state");
        need(bodyTemplate&&acceptedBodyPoses.size()==bodyTemplate.length/sizeof(MRBodyStateGPU)&&
             presentationBodies&&presentationCandidateBodies&&presentationRespiration&&
             presentationCandidateRespiration&&
             presentationBodies.length==bodyTemplate.length&&
             presentationCandidateBodies.length==bodyTemplate.length&&
             presentationRespiration.length==sizeof(NMHumanRespirationState)&&
             presentationCandidateRespiration.length==sizeof(NMHumanRespirationState),
             "terminal GPU pose capture does not match the registered body presentation layout");

        MRHumanRestingCommonCoordinatesGPU acceptedCommonSnapshot{};
        const bool hasTerminalCommon=respiration.commonGeometryGateEnabled;
        if(hasTerminalCommon) {
            need(commonAcceptedCoordinatesInitialized&&respiration.commonGeometryParameters&&
                 respiration.commonGeometryParameters.contents&&respiration.commonGeometryBoxes&&
                 presentationFrameCommonCoordinates&&presentationCommonCoordinates&&
                 acceptedCommonCoordinates,
                 "terminal common-coordinate publication lacks its registered owner or buffers");
            need(presentationFrameCommonCoordinates.length==sizeof(MRHumanRestingCommonCoordinatesGPU)&&
                 presentationCommonCoordinates.length==sizeof(MRHumanRestingCommonCoordinatesGPU)&&
                 acceptedCommonCoordinates.length==sizeof(MRHumanRestingCommonCoordinatesGPU)&&
                 respiration.commonGeometryParameters.length==sizeof(MRHumanRestingCommonFieldGPU)&&
                 respiration.commonGeometryBoxes.length%sizeof(MRHumanRestingCommonCoordinateBoxGPU)==0&&
                 acceptedCommonCoordinates.contents!=nullptr,
                 "terminal common-coordinate buffers have invalid sizes or are not host-visible");
            const auto* parameters=static_cast<const MRHumanRestingCommonFieldGPU*>(
                respiration.commonGeometryParameters.contents);
            const auto boxCount=respiration.commonGeometryBoxes.length/
                sizeof(MRHumanRestingCommonCoordinateBoxGPU);
            need(parameters->countsAndFlags.x==boxCount&&boxCount>0u,
                 "terminal common-coordinate boxes do not match their admitted count");
            acceptedCommonSnapshot=*static_cast<const MRHumanRestingCommonCoordinatesGPU*>(
                acceptedCommonCoordinates.contents);
            const std::array<float,7> coordinates{{acceptedCommonSnapshot.first.x,
                acceptedCommonSnapshot.first.y,acceptedCommonSnapshot.first.z,
                acceptedCommonSnapshot.first.w,acceptedCommonSnapshot.second.x,
                acceptedCommonSnapshot.second.y,acceptedCommonSnapshot.second.z}};
            const bool finiteCoordinates=std::all_of(coordinates.begin(),coordinates.end(),
                [](float value){return std::isfinite(value);});
            need(acceptedCommonSnapshot.status.x==0u&&acceptedCommonSnapshot.status.z<boxCount&&
                 finiteCoordinates&&std::isfinite(acceptedCommonSnapshot.diagnostics.x)&&
                 std::isfinite(parameters->solver.x)&&parameters->solver.x>0.0f&&
                 acceptedCommonSnapshot.diagnostics.x<=parameters->solver.x,
                 "terminal accepted common coordinates fail status/domain/residual validation");
        }

        id<MTLBuffer> poseBuffer=[physiology.device newBufferWithBytes:acceptedBodyPoses.data()
            length:acceptedBodyPoses.size_bytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> statusBuffer=[physiology.device newBufferWithBytes:&acceptedStandStatus
            length:sizeof(acceptedStandStatus) options:MTLResourceStorageModeShared];
        need(poseBuffer&&statusBuffer,"terminal accepted pose/status buffer allocation failed");
        id<MTLCommandQueue> queue=[physiology.device newCommandQueue];
        need(queue!=nil,"terminal accepted presentation command queue is unavailable");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        need(command!=nil,"terminal accepted presentation command buffer is unavailable");
        const mr_uint4 d={expectedTerminalStep,
            static_cast<std::uint32_t>(acceptedBodyPoses.size()),hasTerminalCommon?1u:0u,0u};
        auto captureEncoder=[command computeCommandEncoder];
        need(captureEncoder!=nil,"terminal body capture encoder is unavailable");
        [captureEncoder setComputePipelineState:capture];
        [captureEncoder setBytes:&d length:sizeof(d) atIndex:0];
        [captureEncoder setBuffer:poseBuffer offset:0 atIndex:1];
        [captureEncoder setBuffer:bodyTemplate offset:0 atIndex:2];
        [captureEncoder setBuffer:presentationCandidateBodies offset:0 atIndex:3];
        [captureEncoder setBuffer:respiration.accepted offset:0 atIndex:4];
        [captureEncoder setBuffer:presentationCandidateRespiration offset:0 atIndex:5];
        [captureEncoder setBuffer:statusBuffer offset:0 atIndex:6];
        [captureEncoder dispatchThreads:MTLSizeMake(acceptedBodyPoses.size(),1,1)
            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [captureEncoder endEncoding];
        auto commitEncoder=[command computeCommandEncoder];
        need(commitEncoder!=nil,"terminal frame commit encoder is unavailable");
        [commitEncoder setComputePipelineState:publishTerminalFrame];
        [commitEncoder setBytes:&d length:sizeof(d) atIndex:0];
        [commitEncoder setBuffer:presentationCandidateBodies offset:0 atIndex:1];
        [commitEncoder setBuffer:presentationCandidateRespiration offset:0 atIndex:2];
        [commitEncoder setBuffer:presentationBodies offset:0 atIndex:3];
        [commitEncoder setBuffer:presentationRespiration offset:0 atIndex:4];
        [commitEncoder setBuffer:statusBuffer offset:0 atIndex:5];
        [commitEncoder setBuffer:presentationFrameCommonCoordinates offset:0 atIndex:6];
        [commitEncoder setBuffer:presentationCommonCoordinates offset:0 atIndex:7];
        [commitEncoder setBuffer:acceptedCommonCoordinates offset:0 atIndex:8];
        [commitEncoder dispatchThreads:MTLSizeMake(acceptedBodyPoses.size(),1,1)
            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [commitEncoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             std::string("terminal accepted presentation failed: ")+
                 (command.error?command.error.localizedDescription.UTF8String:
                  "command buffer did not complete"));
        need(std::memcmp(presentationBodies.contents,presentationCandidateBodies.contents,
                         presentationBodies.length)==0&&
             std::memcmp(presentationRespiration.contents,presentationCandidateRespiration.contents,
                         sizeof(NMHumanRespirationState))==0,
             "terminal accepted presentation commit did not publish the GPU-captured frame");
        const auto& published=*static_cast<const NMHumanRespirationState*>(
            presentationRespiration.contents);
        need(published.status.x==expectedTerminalStep&&published.status.w==0u&&
             std::memcmp(&published,&acceptedRespirationSnapshot,sizeof(published))==0&&
             std::memcmp(respiration.accepted.contents,&acceptedRespirationSnapshot,
                         sizeof(acceptedRespirationSnapshot))==0,
             "terminal presentation differs from, or mutated, the final accepted respiration state");
        if(hasTerminalCommon) {
            need(std::memcmp(presentationFrameCommonCoordinates.contents,&acceptedCommonSnapshot,
                             sizeof(acceptedCommonSnapshot))==0&&
                 std::memcmp(presentationCommonCoordinates.contents,&acceptedCommonSnapshot,
                             sizeof(acceptedCommonSnapshot))==0&&
                 std::memcmp(acceptedCommonCoordinates.contents,&acceptedCommonSnapshot,
                             sizeof(acceptedCommonSnapshot))==0,
                 "terminal common-coordinate presentation differs from, or mutated, the accepted state");
        }
        std::cout<<"resting_terminal_presentation=accepted step="<<expectedTerminalStep
            <<" body_count="<<acceptedBodyPoses.size()
            <<" respiratory_status="<<published.status.x
            <<" common_coordinates="<<(hasTerminalCommon?"accepted_buffer_copied":"disabled")
            <<" physical_steps_advanced=0 controller_steps_advanced=0 fk_owner=MetalArticulatedOperator_query_only\n";
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
                    const NSUInteger initialSolverWidth=respiration.commonCoordinatesSolvePipeline.threadExecutionWidth;
                    need(initialSolverWidth>=7u,"common-coordinate SIMD width is below seven rows");
                    [solve dispatchThreadgroups:MTLSizeMake(1,1,1)
                        threadsPerThreadgroup:MTLSizeMake(initialSolverWidth,1,1)];
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
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::postDynamics&&
           p.stepIndex==transactionProbeCaptureControlStep) {
            const auto rootBytes=static_cast<std::size_t>(p.rootTranslationElementCount)*
                sizeof(MRCompensatedRootTranslationGPU);
            const auto qBytes=static_cast<std::size_t>(p.qElementCount)*sizeof(float);
            const auto vBytes=static_cast<std::size_t>(p.vElementCount)*sizeof(float);
            const auto muscleBytes=static_cast<std::size_t>(p.mujocoStateElementCount)*
                sizeof(MRMujocoMuscleStateGPU);
            need(transactionProbeRoots&&transactionProbeQ&&transactionProbeV&&
                     transactionProbeMuscles&&p.rootTranslation&&p.q&&p.v&&
                     p.mujocoStates&&p.commandBuffer&&
                     transactionProbeRoots.length==rootBytes&&
                     transactionProbeQ.length==qBytes&&
                     transactionProbeV.length==vBytes&&
                     transactionProbeMuscles.length==muscleBytes,
                 "multi-step rejection probe raw physical snapshot is incomplete");
            transactionProbeRawRoots=numi::human::buffer(p.rootTranslation);
            transactionProbeRawQ=numi::human::buffer(p.q);
            transactionProbeRawV=numi::human::buffer(p.v);
            transactionProbeRawMuscles=numi::human::buffer(p.mujocoStates);
            need(transactionProbeRawRoots&&transactionProbeRawQ&&
                     transactionProbeRawV&&transactionProbeRawMuscles&&
                     transactionProbeRawRoots.length>=rootBytes&&
                     transactionProbeRawQ.length>=qBytes&&
                     transactionProbeRawV.length>=vBytes&&
                     transactionProbeRawMuscles.length>=muscleBytes,
                 "multi-step rejection probe could not retain raw physical owner buffers");
            transactionProbeRawRootCount=p.rootTranslationElementCount;
            transactionProbeRawQCount=p.qElementCount;
            transactionProbeRawVCount=p.vElementCount;
            transactionProbeRawMuscleCount=p.mujocoStateElementCount;
            transactionProbeCaptured=true;
            transactionProbeCapturedStep=p.stepIndex;
        }
        return true;
    }
};
