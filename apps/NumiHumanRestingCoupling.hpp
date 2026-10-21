#pragma once
#include "../matter/tools/human_resting_runtime.hpp"
#include "NumiHumanVesselWallCoupling.hpp"
#include "NumiHumanImplicitSpringRows.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "numi/matter/numi_human.hpp"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iostream>

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
    // Optional post-command-completion handoff for the FEM body's accepted
    // generalized wrench. It is produced only by the adapter's final
    // accepted-audit gate and is never read while a command buffer is active.
    id<MTLBuffer> acceptedFEMBodyGeneralizedLoadCapture=nil;
    id<MTLBuffer> acceptedFEMBodySourceQCapture=nil;
    std::uint32_t acceptedFEMBodyGeneralizedLoadDofCount=0u;
    std::uint32_t acceptedFEMBodySourceQCoordinateCount=0u;
    std::uint32_t acceptedFEMBodyGeneralizedLoadSourceStep=MR_INVALID_INDEX;
private:
    id<MTLComputePipelineState> prepareWorld, validateBody, validateMatter, capture, publishFrame, publishTerminalFrame;
    id<MTLComputePipelineState> latchCommonFailure=nil;
    id<MTLComputePipelineState> releaseSeamDofProperties=nil;
    id<MTLBuffer> seamDofPatchIndices=nil, seamDofPatchRecords=nil;
    std::uint32_t seamDofReleaseStep=MR_INVALID_INDEX;
    std::uint32_t seamDofIndex=MR_INVALID_INDEX;
    std::uint32_t seamDofPatchCount=0u;
    bool commonFailureCaptureEnabled=false;
    id<MTLBuffer> bodyTemplate;
    metalrobo::MetalNumiHumanGravityScaleRamp gravityScaleRamp{};
    std::uint64_t fingerprint;
    std::uint64_t baseFingerprint=0u;
    numi::matter::NumiHumanTendonFEMLoadAdapter tendonFEMAdapter;
    metalrobo::MetalNumiHumanTendonLoadProgram tendonFEMAdapterProgram{};
    std::uint64_t tendonSourceRootIdentity=0u;
    bool tendonFEMBindingAttempted=false;
    bool tendonFEMBindingEnabled=false;
    bool tendonFEMBindingFailed=false;
    bool programRequested=false;
    bool acceptedFEMBodyGeneralizedLoadCaptureRequested=false;
public:
    NumiHumanRestingCoupling(const char* network,const char* parameters,float dt,
                            const metalrobo::EngineModel& model,std::uint64_t bodySourceIdentity,bool denseVascular=false,
                            std::uint32_t releaseStep=MR_INVALID_INDEX,
                            std::uint32_t sourceSeamDofIndex=MR_INVALID_INDEX,
                            std::span<const std::uint32_t> releasePatchIndices={},
                            std::span<const MRDofPropertiesGPU> releasePatchRecords={},
                            const numi::matter::WorldSource* tissueSource=nullptr,
                            const numi::human::vessel_wall_coupling::Built* aorticWallBinding=nullptr,
                            std::function<void(numi::matter::CompiledWorld&)> solverPrepare={},
                            metalrobo::MetalNumiHumanGravityScaleRamp gravityRamp={}):
        physiology(network,parameters,dt,denseVascular,tissueSource,
                   std::move(solverPrepare)),
        brain(*physiology.respiration,physiology.world,parameters,bodySourceIdentity),
        seamDofReleaseStep(releaseStep),seamDofIndex(sourceSeamDofIndex),
        gravityScaleRamp(gravityRamp) {
        using numi::human::need;
        if(aorticWallBinding) {
            numi::human::vessel_wall_coupling::requireCompiledAbdominalAortaBinding(
                physiology.world,*aorticWallBinding);
            if(!aorticWallBinding->object.femDisabledContactSurfaceFaces.empty()) {
                const auto audit=numi::human::vessel_wall_coupling::auditOuterLateralContactSurfaceFaces(
                    physiology.world,*aorticWallBinding,
                    aorticWallBinding->object.femContactNodes);
                std::cout<<"resting_aortic_wall_runtime_contact_verified=1"
                    <<" selected_faces="<<audit.retainedWallFaceCount
                    <<" disabled_faces="<<audit.excludedWallFaceCount
                    <<" canonical_faces="<<audit.canonicalSurfaceFaceCount
                    <<" mpm_particles="<<physiology.world.dispatch.particleCount
                    <<std::endl;
            }
        }
        need((seamDofReleaseStep==MR_INVALID_INDEX)==(seamDofIndex==MR_INVALID_INDEX),
             "seam DoF release requires both a cut step and mapped source DoF");
        auto device=physiology.device;NSError* e=nil;
        if(seamDofReleaseStep!=MR_INVALID_INDEX) {
            need(!releasePatchIndices.empty()&&releasePatchIndices.size()==releasePatchRecords.size()&&
                 releasePatchIndices.size()<=UINT32_MAX,
                 "seam release requires an exact post-cut DoF patch table");
            seamDofPatchCount=static_cast<std::uint32_t>(releasePatchIndices.size());
            for(const auto index:releasePatchIndices)need(index<model.dofs.size(),
                 "seam release post-cut patch index is outside the admitted model");
            seamDofPatchIndices=[device newBufferWithBytes:releasePatchIndices.data()
                length:releasePatchIndices.size_bytes() options:MTLResourceStorageModeShared];
            seamDofPatchRecords=[device newBufferWithBytes:releasePatchRecords.data()
                length:releasePatchRecords.size_bytes() options:MTLResourceStorageModeShared];
            need(seamDofPatchIndices&&seamDofPatchRecords,
                 "seam release post-cut patch buffers could not be allocated");
            need(seamDofIndex<model.dofs.size()&&model.dofs[seamDofIndex].jointIndex==21u&&
                 (model.dofs[seamDofIndex].flags&MR_DOF_FLAG_POSITION_LIMIT)!=0u&&
                 model.dofs[seamDofIndex].drive.y>=0.0f,
                 "seam release must target the admitted remapped source revolute and preserve its pre-cut limit");
        }
        using numi::human::need;
        #ifdef NUMI_HUMAN_RESPIRATION_SOURCE
        NSString* transactionSource=[NSString stringWithContentsOfFile:@(NUMI_HUMAN_RESPIRATION_SOURCE) encoding:NSUTF8StringEncoding error:&e];
        need(transactionSource!=nil,"resting Human source library unavailable for the seam event");
        if(seamDofReleaseStep!=MR_INVALID_INDEX) {
            NSString* seamKernel=[NSString stringWithUTF8String:R"MSL(kernel void nm_human_release_seam_dof_properties_v6(
    constant uint4& event [[buffer(0)]],
    device MRDofPropertiesGPU* dofs [[buffer(1)]],
    device const uint* patchIndices [[buffer(2)]],
    device const MRDofPropertiesGPU* patchRecords [[buffer(3)]],
    device const MRDofPropertiesGPU* acceptedDofs [[buffer(4)]],
    uint lane [[thread_position_in_grid]]
) {
    if(event.x<event.y || lane>=event.z) return;
    const uint index=patchIndices[lane];
    if(index>=event.w) return;
    // Start from the exact accepted record so unrelated DoF metadata remains
    // rollback-owned; install only precomputed source-derived post-cut fields.
    MRDofPropertiesGPU d=acceptedDofs[index];
    const MRDofPropertiesGPU target=patchRecords[lane];
    d.flags=target.flags;
    d.limits=target.limits;
    d.drive.y=target.drive.y;
    d.drive.z=target.drive.z;
    dofs[index]=d;
}
)MSL"];
            transactionSource=[transactionSource stringByAppendingString:seamKernel];
        }
        MTLCompileOptions* transactionOptions=[MTLCompileOptions new];transactionOptions.languageVersion=MTLLanguageVersion4_0;transactionOptions.mathMode=MTLMathModeSafe;transactionOptions.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        auto lib=[device newLibraryWithSource:transactionSource options:transactionOptions error:&e];
#else
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&e];
#endif
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
        if(seamDofReleaseStep!=MR_INVALID_INDEX)
            releaseSeamDofProperties=pipeline(@"nm_human_release_seam_dof_properties_v6");
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
        const char* femLoadCaptureSetting=std::getenv(
            "NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE");
        need(femLoadCaptureSetting==nullptr ||
                 std::strcmp(femLoadCaptureSetting,"0")==0 ||
                 std::strcmp(femLoadCaptureSetting,"1")==0,
             "NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE must be exactly 0 or 1");
        acceptedFEMBodyGeneralizedLoadCaptureRequested=
            femLoadCaptureSetting!=nullptr &&
            std::strcmp(femLoadCaptureSetting,"1")==0;
        need(bodyTemplate&&presentationBodies&&presentationRespiration&&presentationCandidateBodies&&presentationCandidateRespiration&&
             presentationCommonCoordinates&&presentationCandidateCommonCoordinates&&acceptedCommonCoordinates&&presentationFrameCommonCoordinates&&
             (!commonFailureCaptureEnabled||commonFailureCaptureBuffer),
             "resting presentation allocation");
        fingerprint=brain.rootProgramIdentity^0x4e48524553543031ull;
        fingerprint^=(std::uint64_t(seamDofReleaseStep)<<32u)|seamDofIndex;
        if (gravityScaleRamp.configured()) {
            fingerprint=combineFingerprint(fingerprint,
                gravityScaleRamp.firstAcceptedStep);
            fingerprint=combineFingerprint(fingerprint,
                gravityScaleRamp.fullGravityAcceptedStep);
            fingerprint=combineFingerprint(fingerprint,
                std::bit_cast<std::uint32_t>(gravityScaleRamp.initialScale));
            fingerprint=combineFingerprint(fingerprint,
                std::bit_cast<std::uint32_t>(gravityScaleRamp.fullGravityScale));
        }
        if(!fingerprint)fingerprint=1;
        baseFingerprint=fingerprint;
    }
    static std::uint64_t combineFingerprint(std::uint64_t seed,
                                            std::uint64_t value) noexcept {
        std::uint64_t h=seed?seed:1469598103934665603ull;
        for(unsigned byte=0;byte<8u;++byte) {
            h^=static_cast<std::uint8_t>(value>>(byte*8u));
            h*=1099511628211ull;
        }
        return h?h:1u;
    }
    // Bind a source-owned FEM load map only after this RestingRun has compiled
    // that FEM world. The adapter is never pointed at a second Runtime.
    bool bindTendonFEMSource(
        const numi::matter::NumiHumanTendonFEMLoadSource& source,
        const std::filesystem::path& metallib,
        std::uint64_t sourceRootIdentity) {
        using numi::human::need;
        if(tendonFEMBindingAttempted || programRequested) {
            tendonFEMBindingFailed=true;
            error="Human FEM adapter binding is duplicate or too late";
            return false;
        }
        tendonFEMBindingAttempted=true;
        if(sourceRootIdentity==0u || !physiology.runtime.valid() ||
           physiology.runtime.environmentCount()!=1u ||
           physiology.runtime.femNodeCount()==0u ||
           source.nodeLoads.size()!=physiology.runtime.femNodeCount() ||
           source.nodeAnchors.size()!=source.nodeLoads.size()) {
            tendonFEMBindingFailed=true;
            error="Human FEM adapter requires a source-root-pinned map matching this RestingRun's compiled FEM world";
            return false;
        }
        numi::matter::NumiHumanTendonFEMLoadConfiguration configuration;
        configuration.metallib=metallib;
        configuration.executionMode=
            numi::matter::NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime;
        configuration.reserved0=
            numi::matter::kNumiHumanTendonFEMLoadDeferAcceptedAuditPublication;
        configuration.expectedMicrotickCount=
            1u<<physiology.world.dispatch.maximumRateExponent;
        std::uint32_t implicitSpringCount = 0u;
        need(numi::human::implicit_spring_rows::parseRequestedCount(
                 std::getenv("NUMI_HUMAN_IMPLICIT_BODY_SPRINGS"),
                 numi::matter::kNumiHumanTendonFEMImplicitSpringMaximum,
                 implicitSpringCount),
             "NUMI_HUMAN_IMPLICIT_BODY_SPRINGS must be an integer from 1 through 20");
        if (implicitSpringCount != 0u) {
            tendonFEMBindingFailed=true;
            error="standalone RestingRun cannot admit implicit body springs without the coupled-candidate owner; unset NUMI_HUMAN_IMPLICIT_BODY_SPRINGS";
            return false;
        }
        configuration.implicitElasticBodySpringCount = 0u;
        configuration.acceptedStepExtensionContext=physiology.respiration.get();
        configuration.encodeAcceptedStepExtension=
            &numi::human::Respiration::callback;
        configuration.acceptedStepExtensionFingerprint=
            combineFingerprint(baseFingerprint,sourceRootIdentity);
        configuration.deferAcceptedControlCursorPublication=true;
        configuration.deferAcceptedStepExtensionFinalization=true;
        if(!tendonFEMAdapter.initialize(physiology.runtime,source,configuration)) {
            tendonFEMBindingFailed=true;
            const auto diagnostics=tendonFEMAdapter.diagnostics();
            error=diagnostics.message.empty()
                ? "Human FEM adapter initialization failed for the owning RestingRun"
                : diagnostics.message;
            return false;
        }
        tendonFEMAdapterProgram=tendonFEMAdapter.program();
        const auto diagnostics=tendonFEMAdapter.diagnostics();
        if(!tendonFEMAdapterProgram.valid() || !diagnostics.initialized ||
           diagnostics.fingerprint==0u) {
            tendonFEMBindingFailed=true;
            error="Human FEM adapter produced no valid source/runtime program identity";
            return false;
        }
        tendonSourceRootIdentity=sourceRootIdentity;
        fingerprint=combineFingerprint(
            combineFingerprint(baseFingerprint,sourceRootIdentity),
            diagnostics.fingerprint);
        tendonFEMBindingEnabled=true;
        return true;
    }
    // Return the accepted pre-dynamics FEM-to-body load and the exact source
    // q from which its body poses/Jacobians were evaluated. Output is left
    // untouched on failure; this is not the post-step q.
    // Return the accepted pre-dynamics load and the exact source q from which
    // its body poses/Jacobians were evaluated. Caller must use this captured q
    // as the load reference; it is not the post-step q.
    bool copyAcceptedFEMBodyGeneralizedLoad(
        const std::uint32_t sourceStep,
        std::vector<double>& output,
        std::vector<double>& capturedSourceQ) const {
        if(!acceptedFEMBodyGeneralizedLoadCaptureRequested ||
           acceptedFEMBodyGeneralizedLoadCapture==nil ||
           acceptedFEMBodySourceQCapture==nil ||
           acceptedFEMBodyGeneralizedLoadDofCount==0u ||
           acceptedFEMBodySourceQCoordinateCount==0u ||
           acceptedFEMBodyGeneralizedLoadSourceStep!=sourceStep ||
           sourceStep==MR_INVALID_INDEX ||
           physiology.respiration==nullptr ||
           physiology.respiration->accepted.contents==nullptr) return false;
        const auto* accepted=static_cast<const NMHumanRespirationState*>(
            physiology.respiration->accepted.contents);
        const NSUInteger loadBytes=
            static_cast<NSUInteger>(acceptedFEMBodyGeneralizedLoadDofCount)*sizeof(float);
        const NSUInteger qBytes=
            static_cast<NSUInteger>(acceptedFEMBodySourceQCoordinateCount)*sizeof(float);
        if(accepted->status.x!=sourceStep+1u || accepted->status.w!=0u ||
           acceptedFEMBodyGeneralizedLoadCapture.storageMode!=MTLStorageModeShared ||
           acceptedFEMBodySourceQCapture.storageMode!=MTLStorageModeShared ||
           acceptedFEMBodyGeneralizedLoadCapture.contents==nullptr ||
           acceptedFEMBodySourceQCapture.contents==nullptr ||
           acceptedFEMBodyGeneralizedLoadCapture.length<loadBytes ||
           acceptedFEMBodySourceQCapture.length<qBytes) return false;
        const auto* capturedLoad=static_cast<const float*>(
            acceptedFEMBodyGeneralizedLoadCapture.contents);
        const auto* capturedQ=static_cast<const float*>(
            acceptedFEMBodySourceQCapture.contents);
        std::vector<double> candidateLoad(acceptedFEMBodyGeneralizedLoadDofCount);
        std::vector<double> candidateQ(acceptedFEMBodySourceQCoordinateCount);
        for(std::size_t i=0u;i<candidateLoad.size();++i) {
            if(!std::isfinite(capturedLoad[i])) return false;
            candidateLoad[i]=static_cast<double>(capturedLoad[i]);
        }
        for(std::size_t i=0u;i<candidateQ.size();++i) {
            if(!std::isfinite(capturedQ[i])) return false;
            candidateQ[i]=static_cast<double>(capturedQ[i]);
        }
        output=std::move(candidateLoad);
        capturedSourceQ=std::move(candidateQ);
        return true;
    }
    // Existing terminal evidence code borrows this adapter for snapshots only.
    // Its physical callbacks remain scheduled exclusively by this coupling.
    numi::matter::NumiHumanTendonFEMLoadAdapter* boundTendonFEMAdapter() noexcept {
        return tendonFEMBindingEnabled ? &tendonFEMAdapter : nullptr;
    }
    static metalrobo::MetalNumiHumanTendonLoadPass makeTendonFEMPass(
        const metalrobo::MetalNumanXTransactionPass& p,
        const float matterGravityScale = 1.0f,
        id<MTLBuffer> acceptedGeneralizedLoadCapture = nil,
        const std::uint64_t acceptedGeneralizedLoadElementCount = 0u,
        id<MTLBuffer> acceptedSourceQCapture = nil,
        const std::uint64_t acceptedSourceQElementCount = 0u) {
        using numi::human::need;
        auto narrow=[](std::uint64_t value,const char* name) {
            need(value<=UINT32_MAX,std::string("Human FEM pass ")+name+
                 " exceeds the adapter's 32-bit dimensions");
            return static_cast<std::uint32_t>(value);
        };
        metalrobo::MetalNumiHumanTendonLoadPass pass{};
        pass.commandBuffer=p.commandBuffer;
        pass.mujocoMuscles=p.mujocoMuscles;
        pass.mujocoSites=p.mujocoSites;
        pass.mujocoWraps=p.mujocoWraps;
        pass.mujocoRouteNodes=p.mujocoRouteNodes;
        pass.mujocoResults=p.mujocoResults;
        pass.bindings=p.tendonBindings;
        pass.envelopes=p.tendonEnvelopes;
        pass.transfers=p.tendonTransfers;
        pass.generalizedCorrections=p.tendonGeneralizedCorrections;
        pass.generalizedForces=p.mujocoGeneralizedForceArena;
        pass.bodyPoses=p.bodyPoses;
        pass.bodyPositionLow=p.bodyPositionLow;
        pass.bodyPositionLowGPUAddress=p.bodyPositionLowGPUAddress;
        pass.bodyPositionLowElementCount=p.bodyPositionLowElementCount;
        pass.pointJacobians=p.pointJacobians;
        pass.standStatuses=p.standStatuses;
        pass.stepIndex=p.stepIndex;
        pass.environmentCount=narrow(p.environmentCount,"environment count");
        pass.endpointCount=narrow(p.tendonBindingCount,"endpoint count");
        pass.envelopeCount=narrow(p.tendonEnvelopeCount,"envelope count");
        pass.dofCount=narrow(p.dofCount,"DoF count");
        pass.muscleCount=narrow(p.mujocoMuscleCount,"muscle count");
        pass.siteCount=narrow(p.mujocoSiteCount,"site count");
        pass.wrapCount=narrow(p.mujocoWrapCount,"wrap count");
        pass.routeNodeCount=narrow(p.mujocoRouteNodeCount,"route count");
        pass.generalizedForceStride=narrow(p.mujocoGeneralizedForceStride,"force stride");
        pass.generalizedForceOffset=narrow(p.mujocoGeneralizedForceOffset,"force offset");
        pass.pointJacobianStride=narrow(p.pointJacobianStride,"Jacobian stride");
        pass.bodyJacobianPointOffset=p.bodyJacobianPointOffset;
        pass.bodyPoseStride=narrow(p.bodyPoseStride,"body-pose stride");
        pass.articulationFirstBody=p.articulationFirstBody;
        pass.matterGravityScale=matterGravityScale;
        if (p.abiVersion ==
                metalrobo::kMetalNumanXTransactionCandidateKinematicsABIVersion &&
            p.phase == metalrobo::MetalNumanXTransactionPhase::preDynamics &&
            (p.accessFlags &
             metalrobo::MetalNumanXTransactionEncodeExactCandidateKinematics) != 0u) {
            pass.coupledCandidateContext =
                const_cast<metalrobo::MetalNumanXTransactionPass*>(&p);
            pass.encodeCoupledCandidate =
                &NumiHumanRestingCoupling::encodeTransactionCandidateKinematics;
        }
        if(acceptedSourceQCapture!=nil) {
            pass.sourceQ=p.q;
            pass.sourceQCoordinateCount=narrow(
                p.qCoordinateCount,"source q coordinate count");
            pass.sourceQElementCount=p.qElementCount;
        }
        if(acceptedGeneralizedLoadCapture!=nil) {
            pass.acceptedFEMBodyGeneralizedLoadCapture=
                (__bridge void*)acceptedGeneralizedLoadCapture;
            pass.acceptedFEMBodyGeneralizedLoadGPUAddress=
                acceptedGeneralizedLoadCapture.gpuAddress;
            pass.acceptedFEMBodyGeneralizedLoadElementCount=
                acceptedGeneralizedLoadElementCount;
        }
        if(acceptedSourceQCapture!=nil) {
            pass.acceptedFEMBodySourceQCapture=
                (__bridge void*)acceptedSourceQCapture;
            pass.acceptedFEMBodySourceQGPUAddress=
                acceptedSourceQCapture.gpuAddress;
            pass.acceptedFEMBodySourceQElementCount=
                acceptedSourceQElementCount;
        }
        return pass;
    }
    void ensureAcceptedFEMBodyGeneralizedLoadCapture(
        const std::uint32_t dofCount,
        const std::uint32_t environmentCount,
        const std::uint32_t sourceQCount) {
        using numi::human::need;
        if(!acceptedFEMBodyGeneralizedLoadCaptureRequested) return;
        need(tendonFEMBindingEnabled && dofCount>0u && sourceQCount>0u &&
                 environmentCount==1u,
             "accepted FEM load/source-q capture requires the bound one-environment Human adapter");
        const std::uint64_t elementCount=
            static_cast<std::uint64_t>(dofCount)*environmentCount;
        need(elementCount<=std::numeric_limits<NSUInteger>::max()/sizeof(float),
             "accepted FEM body-load capture extent overflows Metal buffer size");
        const NSUInteger bytes=static_cast<NSUInteger>(elementCount*sizeof(float));
        if(acceptedFEMBodyGeneralizedLoadCapture==nil) {
            acceptedFEMBodyGeneralizedLoadCapture=[physiology.device
                newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            need(acceptedFEMBodyGeneralizedLoadCapture!=nil,
                 "accepted FEM body-load capture buffer allocation failed");
            std::memset(acceptedFEMBodyGeneralizedLoadCapture.contents,0,bytes);
            acceptedFEMBodyGeneralizedLoadDofCount=dofCount;
        }
        need(acceptedFEMBodyGeneralizedLoadDofCount==dofCount &&
                 acceptedFEMBodyGeneralizedLoadCapture.length==bytes,
             "accepted FEM body-load capture dimensions changed within the owner");
        const NSUInteger qBytes=static_cast<NSUInteger>(sourceQCount)*sizeof(float);
        if(acceptedFEMBodySourceQCapture==nil) {
            acceptedFEMBodySourceQCapture=[physiology.device
                newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
            need(acceptedFEMBodySourceQCapture!=nil,
                 "accepted FEM source-q capture buffer allocation failed");
            std::memset(acceptedFEMBodySourceQCapture.contents,0,qBytes);
            acceptedFEMBodySourceQCoordinateCount=sourceQCount;
        }
        need(acceptedFEMBodySourceQCoordinateCount==sourceQCount &&
                 acceptedFEMBodySourceQCapture.length==qBytes,
             "accepted FEM source-q dimensions changed within the owner");
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
        auto acceptedRespirationCopyEncoder=[command blitCommandEncoder];
        need(acceptedRespirationCopyEncoder!=nil,
             "terminal accepted respiration blit encoder is unavailable");
        [acceptedRespirationCopyEncoder copyFromBuffer:respiration.accepted sourceOffset:0
            toBuffer:presentationCandidateRespiration destinationOffset:0
            size:sizeof(NMHumanRespirationState)];
        [acceptedRespirationCopyEncoder endEncoding];
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
        auto terminalCopyEncoder=[command blitCommandEncoder];
        need(terminalCopyEncoder!=nil,
             "terminal accepted exact-state blit encoder is unavailable");
        [terminalCopyEncoder copyFromBuffer:presentationCandidateBodies sourceOffset:0
            toBuffer:presentationBodies destinationOffset:0 size:presentationCandidateBodies.length];
        [terminalCopyEncoder copyFromBuffer:presentationCandidateRespiration sourceOffset:0
            toBuffer:presentationRespiration destinationOffset:0
            size:sizeof(NMHumanRespirationState)];
        [terminalCopyEncoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             std::string("terminal accepted presentation failed: ")+
                 (command.error?command.error.localizedDescription.UTF8String:
                  "command buffer did not complete"));
        const bool terminalBodiesMatch=std::memcmp(
            presentationBodies.contents,presentationCandidateBodies.contents,
            presentationBodies.length)==0;
        const bool terminalRespirationMatch=std::memcmp(
            presentationRespiration.contents,presentationCandidateRespiration.contents,
            sizeof(NMHumanRespirationState))==0;
        if(!terminalBodiesMatch||!terminalRespirationMatch) {
            const auto firstMismatch=[](const void* lhs,const void* rhs,std::size_t bytes) {
                const auto* a=static_cast<const unsigned char*>(lhs);
                const auto* b=static_cast<const unsigned char*>(rhs);
                for(std::size_t i=0;i<bytes;++i) if(a[i]!=b[i]) return i;
                return bytes;
            };
            const auto* candidateRespiration=static_cast<const NMHumanRespirationState*>(
                presentationCandidateRespiration.contents);
            const auto* publishedRespiration=static_cast<const NMHumanRespirationState*>(
                presentationRespiration.contents);
            std::cerr<<"terminal_present_copy_mismatch body_equal="<<terminalBodiesMatch
                <<" body_first_byte="<<firstMismatch(presentationBodies.contents,
                    presentationCandidateBodies.contents,presentationBodies.length)
                <<" respiration_equal="<<terminalRespirationMatch
                <<" respiration_first_byte="<<firstMismatch(presentationRespiration.contents,
                    presentationCandidateRespiration.contents,sizeof(NMHumanRespirationState))
                <<" expected_step="<<expectedTerminalStep
                <<" accepted_body_steps="<<acceptedStandStatus.completedSteps
                <<" accepted_body_code="<<acceptedStandStatus.code
                <<" candidate_resp_step="<<candidateRespiration->status.x
                <<" candidate_resp_failure="<<candidateRespiration->status.w
                <<" published_resp_step="<<publishedRespiration->status.x
                <<" published_resp_failure="<<publishedRespiration->status.w<<"\n";
        }
        need(terminalBodiesMatch&&terminalRespirationMatch,
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
    static bool encodeTransactionCandidateKinematics(
        void* context,
        const void* rawQuery
    ) noexcept {
        const auto* pass = static_cast<
            const metalrobo::MetalNumanXTransactionPass*>(context);
        const auto* query = static_cast<
            const numi::matter::CoupledCandidateQuery*>(rawQuery);
        if (pass == nullptr || query == nullptr ||
            pass->abiVersion != metalrobo::kMetalNumanXTransactionCandidateKinematicsABIVersion ||
            pass->phase != metalrobo::MetalNumanXTransactionPhase::preDynamics ||
            pass->encodeExactCandidateKinematics == nullptr ||
            pass->exactCandidateKinematicsContext == nullptr ||
            query->operation != numi::matter::CoupledCandidateOperation::candidateKinematics ||
            query->legacyHighOnly || query->candidateRootTranslation == nullptr ||
            query->candidateBodyPositionLow == nullptr ||
            query->pointPositionLow != nullptr || query->output != nullptr ||
            query->statuses != nullptr || query->statusStride != 0u) {
            return false;
        }
        auto address = [](void* raw) -> std::uint64_t {
            if (raw == nullptr) return 0u;
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)raw;
            return buffer == nil ? 0u : buffer.gpuAddress;
        };
        metalrobo::MetalNumanXHumanMatterCandidateQuery candidate{};
        candidate.accessFlags =
            metalrobo::MetalNumanXHumanMatterCandidateReadDeltaVelocity |
            metalrobo::MetalNumanXHumanMatterCandidateWriteQ |
            metalrobo::MetalNumanXHumanMatterCandidateWriteBodies;
        if (query->pointCount != 0u) {
            candidate.accessFlags |=
                metalrobo::MetalNumanXHumanMatterCandidateReadPointQueries |
                metalrobo::MetalNumanXHumanMatterCandidateWritePointJacobians;
        }
        candidate.deltaVelocity = query->input;
        candidate.candidateQ = query->candidateQ;
        candidate.candidateBodies = query->candidateBodies;
        candidate.pointQueries = query->pointQueries;
        candidate.pointJacobians = query->pointJacobians;
        candidate.deltaVelocityGPUAddress = address(query->input);
        candidate.candidateQGPUAddress = address(query->candidateQ);
        candidate.candidateBodiesGPUAddress = address(query->candidateBodies);
        candidate.pointQueriesGPUAddress = address(query->pointQueries);
        candidate.pointJacobiansGPUAddress = address(query->pointJacobians);
        candidate.deltaVelocityStride = query->generalizedVectorStride;
        candidate.candidateQStride = query->candidateQStride;
        candidate.candidateBodyStride = query->candidateBodyStride;
        candidate.pointCount = query->pointCount;
        candidate.pointStride = query->pointStride;
        candidate.pointWorldStride = 0u; // CoupledCandidateQuery exposes no pointWorld buffer.
        candidate.pointJacobianStride = query->pointJacobianStride;
        candidate.substepIndex = pass->candidateSubstepIndex;
        candidate.transactionSlot = pass->candidateTransactionSlot;
        candidate.physicsSubstepCount = pass->candidatePhysicsSubstepCount;
        candidate.controlStep = pass->candidateControlStep;
        candidate.programFingerprint = pass->programFingerprint;
        candidate.transactionFingerprint = pass->candidateTransactionFingerprint;
        candidate.linearizationEpoch = pass->candidateLinearizationEpoch;
        candidate.slotGeneration = pass->candidateSlotGeneration;
        candidate.candidateRootTranslation = query->candidateRootTranslation;
        candidate.candidateBodyPositionLow = query->candidateBodyPositionLow;
        candidate.pointPositionLow = query->pointPositionLow;
        candidate.candidateRootTranslationGPUAddress =
            query->candidateRootTranslationGPUAddress;
        candidate.candidateBodyPositionLowGPUAddress =
            query->candidateBodyPositionLowGPUAddress;
        candidate.pointPositionLowGPUAddress = query->pointPositionLowGPUAddress;
        candidate.candidateRootTranslationElementCount =
            query->candidateRootTranslationElementCount;
        candidate.candidateBodyPositionLowElementCount =
            query->candidateBodyPositionLowElementCount;
        candidate.pointPositionLowElementCount = query->pointPositionLowElementCount;
        return pass->encodeExactCandidateKinematics(
            pass->exactCandidateKinematicsContext, *pass, candidate);
    }

    metalrobo::MetalNumanXTransactionProgram program() {
        metalrobo::MetalNumanXTransactionProgram p;
        if(tendonFEMBindingFailed)return p;
        programRequested=true;
        p.context=this;p.encode=&encodeCallback;p.abort=&abortCallback;p.fingerprint=fingerprint;
        p.writeBodyProperties=true;
        p.deferAcceptedPublication=true;
        if(seamDofReleaseStep!=MR_INVALID_INDEX) p.writeDofProperties=true;
        const char* candidateSetting =
            std::getenv("NUMI_HUMAN_EXACT_CANDIDATE_KINEMATICS");
        if (candidateSetting != nullptr &&
            std::strcmp(candidateSetting, "1") == 0) {
            const std::uint32_t capacity =
                physiology.runtime.femHumanAttachmentCount();
            if (!tendonFEMBindingEnabled || capacity == 0u ||
                capacity > MR_ARTICULATED_OPERATOR_MAX_POINTS) {
                return {};
            }
            p.abiVersion =
                metalrobo::kMetalNumanXTransactionCandidateKinematicsABIVersion;
            p.writeExactCandidateKinematics = true;
            p.candidatePointCapacity = capacity;
        }
        return p;
    }
    static void abortCallback(void* context,void* commandBuffer) noexcept {
        auto& self=*static_cast<NumiHumanRestingCoupling*>(context);
        const bool adapterOwnsCancel=self.tendonFEMBindingEnabled&&
            self.tendonFEMAdapterProgram.abort!=nullptr;
        if(adapterOwnsCancel)
            self.tendonFEMAdapterProgram.abort(
                self.tendonFEMAdapterProgram.context,commandBuffer);
        self.physiology.respiration->discardBodyPropertiesReferenceCapture(commandBuffer);
        self.physiology.respiration->clearPhysicsLease(commandBuffer);
        if(!adapterOwnsCancel)self.physiology.runtime.cancel(commandBuffer);
    }
    static bool encodeCallback(void* context,const metalrobo::MetalNumanXTransactionPass& p) noexcept {
        auto& self=*static_cast<NumiHumanRestingCoupling*>(context);
        try{return self.encode(p);}catch(const std::exception& e){self.error=e.what();fprintf(stderr,"resting_transaction_encode_failure phase=%u step=%u error=%s\n",static_cast<unsigned>(p.phase),p.stepIndex,self.error.c_str());return false;}
    }
    bool encode(const metalrobo::MetalNumanXTransactionPass& p) {
        using namespace numi::matter;
        using numi::human::need;
        need(!tendonFEMBindingFailed,
             error.empty()?"Human FEM adapter binding failed":error);
        need(p.environmentCount==1&&p.timestepSeconds==physiology.runtime.timestepSeconds(),
             "resting body/physiology cadence differs");
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::beginStep) {
            return physiology.respiration->captureBodyPropertiesReference(
                p.commandBuffer,p.bodyPropertiesCheckpoint,p.bodyPropertyCount);
        }
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
        need(p.bodyPoseStride<=UINT32_MAX&&p.pointCount<=UINT32_MAX&&
             p.pointJacobianStride<=UINT32_MAX&&p.dofCount<=UINT32_MAX&&
             p.vStride<=UINT32_MAX&&p.mujocoGeneralizedForceOffset<=UINT32_MAX&&
             p.mujocoGeneralizedForceStride<=UINT32_MAX&&p.bodyJacobianPointOffset<=UINT32_MAX&&
             p.bodyCount<=UINT32_MAX,
             "resting outlet metadata exceeds 32-bit shader dimensions");
        r.controlStep=p.stepIndex;r.physicsSubsteps=1;r.timestepSeconds=p.timestepSeconds;r.runAdaptiveTransfer=false;
        r.deferAcceptedControlCursorPublication=true;
        r.deferAcceptedStepExtensionFinalization=true;
        r.acceptedStepExtensionContext=physiology.respiration.get();r.encodeAcceptedStepExtension=&numi::human::Respiration::callback;
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::acceptedPublication) {
            need(p.standStatuses&&p.environmentCount==1,
                 "accepted publication requires the current one-environment stand status");
            AcceptedStepFinalizationGate finalGate;
            finalGate.statusRecords=p.standStatuses;
            finalGate.recordCount=static_cast<std::uint32_t>(p.environmentCount);
            finalGate.recordStrideBytes=sizeof(MRNumiHumanStandStatusGPU);
            finalGate.acceptedStatusCode=MR_NUMI_HUMAN_STAND_SUCCESS;
            finalGate.completedCountOffsetBytes=offsetof(MRNumiHumanStandStatusGPU,completedSteps);
            finalGate.expectedCompletedCount=p.stepIndex+1u;
            finalGate.environmentIndexOffsetBytes=offsetof(MRNumiHumanStandStatusGPU,environment);
            finalGate.expectedEnvironmentBase=0u;
            const auto finalized=physiology.runtime.finalizeAcceptedStepExtension(
                p.commandBuffer,p.stepIndex,0u,finalGate);
            need(finalized.encoded,finalized.message);
            if(tendonFEMBindingEnabled) {
                // The Runtime finalizer can still fail its completion-count
                // gate. Merge that final Matter status into Stand before any
                // adapter accepted-audit history is allowed to publish.
                bridge(validateMatter);
                const auto adapterPass=makeTendonFEMPass(
                    p,1.0f,acceptedFEMBodyGeneralizedLoadCapture,
                    acceptedFEMBodyGeneralizedLoadDofCount,
                    acceptedFEMBodySourceQCapture,
                    acceptedFEMBodySourceQCoordinateCount);
                need(tendonFEMAdapter.finalizeAcceptedAudits(adapterPass),
                     tendonFEMAdapter.diagnostics().message);
                if(acceptedFEMBodyGeneralizedLoadCaptureRequested)
                    acceptedFEMBodyGeneralizedLoadSourceStep=p.stepIndex;
            }

            AcceptedControlCursorGate cursorGate;
            cursorGate.statusCodes=p.standStatuses;
            cursorGate.recordCount=static_cast<std::uint32_t>(p.environmentCount);
            cursorGate.recordStrideBytes=sizeof(MRNumiHumanStandStatusGPU);
            cursorGate.acceptedStatusCode=MR_NUMI_HUMAN_STAND_SUCCESS;
            const auto cursor=physiology.runtime.encodeAcceptedControlCursorPublication(
                p.commandBuffer,p.stepIndex,0u,cursorGate);
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
            return true;
        }
        if(p.phase==metalrobo::MetalNumanXTransactionPhase::preDynamics) {
            need(p.dofCount>0u&&p.dofCount<=UINT32_MAX&&
                     p.environmentCount>0u&&p.environmentCount<=UINT32_MAX,
                 "accepted FEM body-load capture dimensions exceed 32-bit bounds");
            if(acceptedFEMBodyGeneralizedLoadCaptureRequested) {
                need(p.q != nullptr && p.qCoordinateCount > 0u &&
                         p.qCoordinateCount <= UINT32_MAX &&
                         p.qElementCount >= p.qCoordinateCount,
                     "accepted FEM source-q capture requires a valid borrowed q view");
                ensureAcceptedFEMBodyGeneralizedLoadCapture(
                    static_cast<std::uint32_t>(p.dofCount),
                    static_cast<std::uint32_t>(p.environmentCount),
                    static_cast<std::uint32_t>(p.qCoordinateCount));
            }
            if(seamDofReleaseStep!=MR_INVALID_INDEX && p.stepIndex>=seamDofReleaseStep) {
                need((p.accessFlags&metalrobo::MetalNumanXTransactionWriteDofProperties)!=0u&&
                     p.abiVersion==metalrobo::kMetalNumanXTransactionABIVersion&&
                     p.dofProperties&&p.dofPropertiesCheckpoint&&
                     p.dofPropertyCount>seamDofIndex&&
                     p.dofPropertyCount<=UINT32_MAX&&
                     p.commandBuffer&&releaseSeamDofProperties,
                     "seam release lacks the ABI6 live DoF-property lease");
                need(seamDofPatchCount>0u&&seamDofPatchIndices&&seamDofPatchRecords,
                     "seam release post-cut patch buffers are unavailable");
                const mr_uint4 event={p.stepIndex,seamDofReleaseStep,seamDofPatchCount,
                                      static_cast<mr_u32>(p.dofPropertyCount)};
                auto release=[cb computeCommandEncoder];need(release!=nil,"seam DoF release encoder");
                [release setComputePipelineState:releaseSeamDofProperties];
                [release setBytes:&event length:sizeof(event) atIndex:0];
                [release setBuffer:numi::human::buffer(p.dofProperties) offset:0 atIndex:1];
                [release setBuffer:seamDofPatchIndices offset:0 atIndex:2];
                [release setBuffer:seamDofPatchRecords offset:0 atIndex:3];
                [release setBuffer:numi::human::buffer(p.dofPropertiesCheckpoint) offset:0 atIndex:4];
                [release dispatchThreads:MTLSizeMake(seamDofPatchCount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64u,releaseSeamDofProperties.maxTotalThreadsPerThreadgroup),1,1)];
                [release endEncoding];
            }
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
        const float matterGravityScale =
            gravityScaleRamp.scaleAtAcceptedStep(p.stepIndex + 1u);
        auto& respiration=*physiology.respiration;
        numi::human::Respiration::PhysicsLease lease{};
        lease.commandBuffer=p.commandBuffer;
        lease.controlStep=p.stepIndex;
        lease.bodyPropertiesAccepted=p.bodyPropertiesCheckpoint;
        lease.bodyPropertiesCandidate=p.bodyProperties;
        lease.bodyPropertiesCheckpoint=p.bodyPropertiesCheckpoint;
        lease.bodyPropertiesReference=respiration.bodyPropertiesReferenceHandle();
        lease.bodyPosesAccepted=p.bodyPoses;
        lease.bodyPositionLow=p.bodyPositionLow;
        lease.pointJacobians=p.pointJacobians;
        lease.generalizedVelocity=p.v;
        lease.generalizedForceArena=p.mujocoGeneralizedForceArena;
        lease.bodyPropertyCount=p.bodyPropertyCount;
        lease.bodyPoseElementCount=p.bodyPoseElementCount;
        lease.bodyPositionLowElementCount=p.bodyPositionLowElementCount;
        lease.pointJacobianElementCount=p.pointJacobianElementCount;
        lease.generalizedVelocityElementCount=p.vElementCount;
        lease.generalizedForceArenaElementCount=p.mujocoGeneralizedForceArenaElementCount;
        lease.bodyPoseStride=static_cast<std::uint32_t>(p.bodyPoseStride);
        lease.pointCount=static_cast<std::uint32_t>(p.pointCount);
        lease.pointJacobianStride=static_cast<std::uint32_t>(p.pointJacobianStride);
        lease.bodyJacobianPointOffset=static_cast<std::uint32_t>(p.bodyJacobianPointOffset);
        lease.articulationFirstBody=p.articulationFirstBody;
        lease.dofCount=static_cast<std::uint32_t>(p.dofCount);
        lease.vStride=static_cast<std::uint32_t>(p.vStride);
        lease.generalizedForceOffset=static_cast<std::uint32_t>(p.mujocoGeneralizedForceOffset);
        lease.generalizedForceStride=static_cast<std::uint32_t>(p.mujocoGeneralizedForceStride);
        lease.bodyCount=static_cast<std::uint32_t>(p.bodyCount);
        need(respiration.stagePhysicsLease(lease),"resting owner physics lease already active or invalid");
        struct PhysicsLeaseScope {
            numi::human::Respiration& owner;
            void* commandBuffer;
            ~PhysicsLeaseScope(){owner.clearPhysicsLease(commandBuffer);}
        } leaseScope{respiration,p.commandBuffer};
        if(tendonFEMBindingEnabled) {
            const auto adapterPass=makeTendonFEMPass(
                p, matterGravityScale,
                acceptedFEMBodyGeneralizedLoadCapture,
                acceptedFEMBodyGeneralizedLoadDofCount,
                acceptedFEMBodySourceQCapture,
                acceptedFEMBodySourceQCoordinateCount);
            const auto encodeAdapter=p.phase==
                    metalrobo::MetalNumanXTransactionPhase::preDynamics
                ? tendonFEMAdapterProgram.encodePreDynamics
                : tendonFEMAdapterProgram.encodePostValidation;
            const bool adapterEncoded=encodeAdapter!=nullptr&&
                encodeAdapter(tendonFEMAdapterProgram.context,adapterPass);
            const auto adapterDiagnostic=tendonFEMAdapter.diagnostics();
            need(adapterEncoded,adapterDiagnostic.message.empty()
                    ? "Human FEM adapter rejected the borrowed RestingRun pass"
                    : adapterDiagnostic.message);
        } else {
            auto result=physiology.runtime.encodeWithGravityScale(
                r, matterGravityScale);
            need(result.encoded,result.message);
        }
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
                [latch setBuffer:numi::human::buffer(physiology.runtime.statusBuffer()) offset:0 atIndex:2];
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
