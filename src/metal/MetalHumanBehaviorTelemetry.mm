#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/MetalHumanBehaviorTelemetry.hpp"
#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace metalrobo {
namespace {
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
std::string describe(NSError* e){const char* text=e.localizedDescription.UTF8String;return text!=nullptr?text:"Metal telemetry failure";}
id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString* name){
    id<MTLFunction> f=[l newFunctionWithName:name];require(f!=nil,"missing telemetry kernel");NSError* error=nil;
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(p==nil)throw std::runtime_error(describe(error));return p;
}
void dispatch(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> p,std::uint32_t n){
    [encoder setComputePipelineState:p];[encoder dispatchThreads:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(n,p.maxTotalThreadsPerThreadgroup),1,1)];[encoder endEncoding];
}
template<class T> std::vector<T> copy(id<MTLBuffer> b,std::uint32_t n){std::vector<T> out(n);std::memcpy(out.data(),b.contents,n*sizeof(T));return out;}
}
struct MetalHumanBehaviorTelemetry::State {
    id<MTLDevice> device=nil;id<MTLComputePipelineState> measure=nil,initial=nil,reduce=nil;
    id<MTLBuffer> program=nil,candidate=nil,release=nil,fences=nil,reduction=nil,audit=nil;
    std::uint32_t environments=0;std::uint64_t fp=0,terminalSerial=0,initialPhysicsGeneration=0,initialTimestampNanoseconds=0;
    bool initialEncoded=false;
    MRHumanBehaviorProgramGPU cooked{};
};
MetalHumanBehaviorTelemetry::MetalHumanBehaviorTelemetry(void* device,const CompiledHumanBehaviorProgram& program,const std::string& path,std::uint32_t environments,std::uint64_t initialPhysicsGeneration,std::uint64_t initialTimestampNanoseconds):state_(std::make_unique<State>()) {
    auto& s=*state_;s.device=(__bridge id<MTLDevice>)device;
    require(s.device!=nil&&environments==1&&program.gpu.abiVersion==MR_HUMAN_BEHAVIOR_ABI_VERSION&&program.gpu.fingerprint!=0,"telemetry currently requires one bound root environment");
    s.environments=environments;s.cooked=program.gpu;s.fp=program.gpu.fingerprint;s.initialPhysicsGeneration=initialPhysicsGeneration;s.initialTimestampNanoseconds=initialTimestampNanoseconds;
    NSError* error=nil;id<MTLLibrary> library=[s.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&error];
    if(library==nil)throw std::runtime_error(describe(error));
    s.measure=pipeline(s.device,library,@"human_behavior_measure");s.initial=pipeline(s.device,library,@"human_behavior_seed_initial");s.reduce=pipeline(s.device,library,@"human_behavior_reduce");
    auto allocate=[&](NSUInteger size){id<MTLBuffer> b=[s.device newBufferWithLength:size options:MTLResourceStorageModeShared];require(b!=nil,"telemetry buffer allocation failed");std::memset(b.contents,0,size);return b;};
    s.program=allocate(sizeof(MRHumanBehaviorProgramGPU));std::memcpy(s.program.contents,&program.gpu,sizeof(program.gpu));
    s.candidate=allocate(sizeof(MRHumanBehaviorCandidateGPU)*environments);s.release=allocate(sizeof(MRHumanBehaviorReleaseGPU)*environments);
    s.fences=allocate(sizeof(MRNumanXHumanMatterJointPublicationFenceGPU)*environments);s.reduction=allocate(sizeof(MRHumanBehaviorReductionGPU)*environments);
    s.audit=allocate(sizeof(MRHumanBehaviorAuditGPU)*environments);
    // Coverage remains explicitly unknown until native audit/contact owners
    // implement it. A zero violation count with coverage0 is never qualification.
    reset();
}
MetalHumanBehaviorTelemetry::~MetalHumanBehaviorTelemetry()=default;
std::uint64_t MetalHumanBehaviorTelemetry::fingerprint()const noexcept{return state_->fp;}
std::uint64_t MetalHumanBehaviorTelemetry::completedAttempts()const noexcept{return state_->terminalSerial;}
bool MetalHumanBehaviorTelemetry::initialObservationEncoded()const noexcept{return state_->initialEncoded;}
void MetalHumanBehaviorTelemetry::reset(){auto& s=*state_;s.terminalSerial=0;s.initialEncoded=false;
    for(auto b:{s.candidate,s.release,s.fences,s.reduction})std::memset(b.contents,0,b.length);
    auto* out=static_cast<MRHumanBehaviorReductionGPU*>(s.reduction.contents);
    for(std::uint32_t i=0;i<s.environments;++i){out[i].abiVersion=MR_HUMAN_BEHAVIOR_ABI_VERSION;out[i].programFingerprint=s.fp;
        out[i].initialPhysicsGeneration=s.initialPhysicsGeneration;out[i].initialTimestampNanoseconds=s.initialTimestampNanoseconds;out[i].endNanoseconds=s.initialTimestampNanoseconds;
        out[i].initialPostureValid=2;out[i].initialSettled=2;} // unmeasured reset, explicit unknown
}
bool MetalHumanBehaviorTelemetry::encodeInitial(const MetalNumanXHumanMatterPass& pass,std::string& error)noexcept{
    try{auto& s=*state_;require(!s.initialEncoded&&pass.abiVersion==kMetalNumanXHumanMatterPassABIVersion&&pass.structSize==sizeof(pass)&&pass.phase==MetalNumanXHumanMatterPhase::preDynamics,"invalid initial behavior pass");require(pass.environmentCount==s.environments&&pass.physicsSubstepCount==1,"invalid initial behavior root shape");require(pass.bodyPoses!=nullptr&&pass.bodyPositionLow!=nullptr&&pass.pointJacobians!=nullptr&&pass.v!=nullptr,"missing initial behavior geometry");require(pass.bodyPoseStride>=s.cooked.bodyCount&&pass.vStride>=s.cooked.dofCount&&pass.dofCount==s.cooked.dofCount,"initial behavior shape drift");
        MRHumanBehaviorDispatchGPU d{};d.environmentCount=s.environments;d.bodyPoseStride=static_cast<std::uint32_t>(pass.bodyPoseStride);d.pointJacobianStride=static_cast<std::uint32_t>(pass.pointJacobianStride);d.vStride=static_cast<std::uint32_t>(pass.vStride);d.bodyJacobianPointOffset=static_cast<std::uint32_t>(pass.bodyJacobianPointOffset);d.transactionFingerprint=pass.transactionFingerprint;d.linearizationEpoch=pass.linearizationEpoch;d.slotGeneration=pass.slotGeneration;d.physicsGeneration=s.initialPhysicsGeneration;d.acceptedTimestampNanoseconds=s.initialTimestampNanoseconds;
        require(pass.bodyPoseStride<=UINT32_MAX&&pass.pointJacobianStride<=UINT32_MAX&&pass.vStride<=UINT32_MAX&&pass.bodyJacobianPointOffset<=UINT32_MAX,"initial behavior stride overflow");
        auto buffer=[&](void* p,NSUInteger bytes){id<MTLBuffer> b=(__bridge id<MTLBuffer>)p;require(b!=nil&&b.device==s.device&&b.length>=bytes,"initial behavior buffer shape/device mismatch");return b;};
        id<MTLBuffer> bodies=buffer(pass.bodyPoses,sizeof(MRArticulatedBodyPoseGPU)*pass.bodyPoseStride*s.environments);id<MTLBuffer> low=buffer(pass.bodyPositionLow,sizeof(mr_float4)*pass.bodyPoseStride*s.environments);id<MTLBuffer> jac=buffer(pass.pointJacobians,sizeof(float)*pass.pointJacobianStride*s.environments);id<MTLBuffer> v=buffer(pass.v,sizeof(float)*pass.vStride*s.environments);id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;require(cb!=nil&&cb.device==s.device,"initial behavior command buffer mismatch");
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];require(e!=nil,"initial behavior encoder failed");[e setBuffer:s.program offset:0 atIndex:0];[e setBytes:&d length:sizeof(d) atIndex:1];[e setBuffer:bodies offset:0 atIndex:2];[e setBuffer:low offset:0 atIndex:3];[e setBuffer:jac offset:0 atIndex:4];[e setBuffer:v offset:0 atIndex:5];[e setBuffer:s.audit offset:0 atIndex:6];[e setBuffer:s.candidate offset:0 atIndex:7];dispatch(e,s.measure,s.environments);
        e=[cb computeCommandEncoder];require(e!=nil,"initial behavior seed encoder failed");[e setBuffer:s.program offset:0 atIndex:0];[e setBytes:&d length:sizeof(d) atIndex:1];[e setBuffer:s.candidate offset:0 atIndex:2];[e setBuffer:s.reduction offset:0 atIndex:3];dispatch(e,s.initial,1u);s.initialEncoded=true;error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
bool MetalHumanBehaviorTelemetry::encodeFlush(void* commandBuffer,std::string& error)noexcept{
    try{auto& s=*state_;id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)commandBuffer;
        require(cb!=nil&&cb.device==s.device,"telemetry flush requires original compatible borrowed command buffer");
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];require(e!=nil,"telemetry reduction encoder failed");
        [e setBuffer:s.program offset:0 atIndex:0];[e setBytes:&s.environments length:sizeof(s.environments) atIndex:1];
        [e setBuffer:s.candidate offset:0 atIndex:2];[e setBuffer:s.release offset:0 atIndex:3];[e setBuffer:s.fences offset:0 atIndex:4];[e setBuffer:s.reduction offset:0 atIndex:5];
        dispatch(e,s.reduce,s.environments);error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
bool MetalHumanBehaviorTelemetry::encodeCandidate(const MetalNumanXHumanMatterPass& pass,std::uint64_t physicsGeneration,std::uint64_t acceptedTimestampNanoseconds,std::string& error)noexcept{
    try{auto& s=*state_;require(pass.abiVersion==kMetalNumanXHumanMatterPassABIVersion&&pass.structSize==sizeof(pass)&&pass.phase==MetalNumanXHumanMatterPhase::postDynamics,"telemetry requires exact final-candidate owner phase");require(pass.environmentCount==s.environments&&pass.physicsSubstepCount==1&&pass.substepIndex==0,"invalid telemetry candidate root shape");
        require(pass.bodyPoses!=nullptr&&pass.bodyPositionLow!=nullptr&&pass.pointJacobians!=nullptr&&pass.v!=nullptr,"missing paired candidate geometry");
        require(pass.bodyPoseStride>=s.cooked.bodyCount&&pass.vStride>=s.cooked.dofCount&&pass.dofCount==s.cooked.dofCount,"telemetry candidate shape drift");
        MRHumanBehaviorDispatchGPU d{};d.environmentCount=s.environments;d.bodyPoseStride=static_cast<std::uint32_t>(pass.bodyPoseStride);d.pointJacobianStride=static_cast<std::uint32_t>(pass.pointJacobianStride);d.vStride=static_cast<std::uint32_t>(pass.vStride);d.bodyJacobianPointOffset=static_cast<std::uint32_t>(pass.bodyJacobianPointOffset);
        d.transactionFingerprint=pass.transactionFingerprint;d.linearizationEpoch=pass.linearizationEpoch;d.slotGeneration=pass.slotGeneration;
        d.physicsGeneration=physicsGeneration;d.acceptedTimestampNanoseconds=acceptedTimestampNanoseconds;
        require(pass.bodyPoseStride<=UINT32_MAX&&pass.pointJacobianStride<=UINT32_MAX&&pass.vStride<=UINT32_MAX&&pass.bodyJacobianPointOffset<=UINT32_MAX,"telemetry stride overflow");
        auto buffer=[&](void* p,NSUInteger bytes){id<MTLBuffer> b=(__bridge id<MTLBuffer>)p;require(b!=nil&&b.device==s.device&&b.length>=bytes,"candidate telemetry buffer shape/device mismatch");return b;};
        id<MTLBuffer> bodies=buffer(pass.bodyPoses,sizeof(MRArticulatedBodyPoseGPU)*pass.bodyPoseStride*s.environments);
        id<MTLBuffer> low=buffer(pass.bodyPositionLow,sizeof(mr_float4)*pass.bodyPoseStride*s.environments);
        id<MTLBuffer> jac=buffer(pass.pointJacobians,sizeof(float)*pass.pointJacobianStride*s.environments);
        id<MTLBuffer> v=buffer(pass.v,sizeof(float)*pass.vStride*s.environments);
        id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];require(e!=nil,"telemetry candidate encoder failed");
        [e setBuffer:s.program offset:0 atIndex:0];[e setBytes:&d length:sizeof(d) atIndex:1];[e setBuffer:bodies offset:0 atIndex:2];[e setBuffer:low offset:0 atIndex:3];
        [e setBuffer:jac offset:0 atIndex:4];[e setBuffer:v offset:0 atIndex:5];[e setBuffer:s.audit offset:0 atIndex:6];[e setBuffer:s.candidate offset:0 atIndex:7];dispatch(e,s.measure,s.environments);
        error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
bool MetalHumanBehaviorTelemetry::terminal(const MRHumanBehaviorReleaseGPU& r,const MRNumanXHumanMatterJointPublicationFenceGPU* fence,std::string& error)noexcept{
    try{auto& s=*state_;require(r.programFingerprint==s.fp&&r.publicationSerial==s.terminalSerial+1&&s.terminalSerial!=std::numeric_limits<std::uint64_t>::max()&&r.transactionFingerprint!=0&&r.slotGeneration!=0&&(r.released==1||r.released==2)&&r.reserved0==0&&r.reserved1==0&&r.reserved2==0,"invalid telemetry terminal identity");
        require(r.released!=1||(fence!=nullptr&&
            (fence->abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION||
             fence->abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2)&&
            fence->structBytes==sizeof(*fence)&&
            fence->status==MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED&&
            fence->fenceFingerprint==r.jointFenceFingerprint&&
            r.jointFenceFingerprint!=0),"telemetry acceptance requires released COMMITTED root");
        if(fence)std::memcpy(s.fences.contents,fence,sizeof(*fence));else std::memset(s.fences.contents,0,s.fences.length);
        std::memcpy(s.release.contents,&r,sizeof(r));s.terminalSerial=r.publicationSerial;error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
HumanBehaviorTelemetrySnapshot MetalHumanBehaviorTelemetry::snapshot()const{const auto& s=*state_;return {s.fp,copy<MRHumanBehaviorCandidateGPU>(s.candidate,s.environments),copy<MRHumanBehaviorReleaseGPU>(s.release,s.environments),copy<MRNumanXHumanMatterJointPublicationFenceGPU>(s.fences,s.environments),copy<MRHumanBehaviorReductionGPU>(s.reduction,s.environments)};}
bool MetalHumanBehaviorTelemetry::restore(const HumanBehaviorTelemetrySnapshot& snapshot,std::string& error)noexcept{
    try{auto& s=*state_;require(snapshot.programFingerprint==s.fp&&snapshot.candidate.size()==s.environments&&snapshot.release.size()==s.environments&&snapshot.fences.size()==s.environments&&snapshot.reduction.size()==s.environments,"telemetry checkpoint shape/source mismatch");
        const auto& r=snapshot.reduction[0];const auto& release=snapshot.release[0];
        require(r.abiVersion==MR_HUMAN_BEHAVIOR_ABI_VERSION&&r.programFingerprint==s.fp&&r.status==0&&r.initialPhysicsGeneration==s.initialPhysicsGeneration&&r.initialTimestampNanoseconds==s.initialTimestampNanoseconds&&r.initialPostureValid<=2&&r.initialSettled<=2&&r.metricSampleCount==r.acceptedRootCount&&r.auditCoveredRootCount<=r.metricSampleCount&&r.publicationSerial<=release.publicationSerial&&release.publicationSerial-r.publicationSerial<=1,"invalid telemetry checkpoint counters/pending root");
        require(r.rejectedAttemptCount<=UINT64_MAX-r.acceptedRootCount&&r.publicationSerial==r.acceptedRootCount+r.rejectedAttemptCount&&
            r.acceptedRootCount<=(UINT64_MAX-r.initialTimestampNanoseconds)/s.cooked.timestepNanoseconds&&
            r.endNanoseconds==r.initialTimestampNanoseconds+r.acceptedRootCount*s.cooked.timestepNanoseconds&&
            r.postureViolationCount<=r.acceptedRootCount&&r.settledSuffixSteps<=r.acceptedRootCount&&r.auditCoveredAttemptCount<=r.publicationSerial&&
            (r.speedErrorSampleCount==0||r.speedErrorSampleCount==r.acceptedRootCount),"telemetry checkpoint count/clock inconsistency");
        if(release.publicationSerial){require(release.programFingerprint==s.fp&&(release.released==1||release.released==2)&&
            (release.released==2||release.transactionFingerprint==snapshot.candidate[0].transactionFingerprint),"telemetry checkpoint pending identity mismatch");}
        for(auto count:r.auditViolations)require(count<=r.publicationSerial,"telemetry checkpoint audit count inconsistency");
        s.initialEncoded = r.initialPostureValid != 2u || r.initialSettled != 2u;
        std::memcpy(s.candidate.contents,snapshot.candidate.data(),s.candidate.length);std::memcpy(s.release.contents,snapshot.release.data(),s.release.length);
        std::memcpy(s.fences.contents,snapshot.fences.data(),s.fences.length);std::memcpy(s.reduction.contents,snapshot.reduction.data(),s.reduction.length);s.terminalSerial=release.publicationSerial;
        error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
} // namespace metalrobo
