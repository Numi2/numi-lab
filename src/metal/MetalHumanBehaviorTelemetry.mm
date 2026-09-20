#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/MetalHumanBehaviorTelemetry.hpp"
#include <cstring>
#include <algorithm>
#include <array>
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
std::uint64_t fnv(const void* raw,std::size_t bytes,std::uint64_t seed=14695981039346656037ull){
    const auto* p=static_cast<const std::uint8_t*>(raw);auto h=seed;
    for(std::size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ull;}return h;
}
template<class T> void mix(std::uint64_t& h,const T& value){h=fnv(&value,sizeof(value),h);}
bool nonzero(const std::array<std::uint8_t,32>& value){return std::any_of(value.begin(),value.end(),[](auto x){return x!=0;});}
constexpr std::uint32_t kAllTraceEvidence=
    MRNX_BEHAVIOR_TRACE_EVIDENCE_SOURCE_BOUND_METRIC_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_NATIVE_AUDIT_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_FORBIDDEN_CONTACT_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_ACCEPTED_ROOT_PROOF_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_FULL_BEHAVIOR_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_PHYSICAL_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_BIOLOGICAL_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_PERFORMANCE_V1|
    MRNX_BEHAVIOR_TRACE_EVIDENCE_PRODUCTION_V1;
static_assert(sizeof(MRHumanBehaviorTraceRecordGPU)==sizeof(mrnx_behavior_trace_record_v1));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,abiVersion)==offsetof(mrnx_behavior_trace_record_v1,abi_version));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,attemptIndex)==offsetof(mrnx_behavior_trace_record_v1,attempt_index));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,basePublicationEpoch)==offsetof(mrnx_behavior_trace_record_v1,base_publication_epoch));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,candidateStateProofFingerprint)==offsetof(mrnx_behavior_trace_record_v1,candidate_state_proof_fingerprint));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,afterPublicationEpoch)==offsetof(mrnx_behavior_trace_record_v1,after_publication_epoch));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,valueHigh)==offsetof(mrnx_behavior_trace_record_v1,value_high));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,previousRecordFingerprint)==offsetof(mrnx_behavior_trace_record_v1,previous_record_fingerprint));
static_assert(offsetof(MRHumanBehaviorTraceRecordGPU,recordFingerprint)==offsetof(mrnx_behavior_trace_record_v1,record_fingerprint));
}
struct MetalHumanBehaviorTelemetry::State {
    id<MTLDevice> device=nil;id<MTLComputePipelineState> measure=nil,initial=nil,reduce=nil;
    id<MTLBuffer> program=nil,candidate=nil,release=nil,fences=nil,reduction=nil,audit=nil;
    id<MTLBuffer> tracePage=nil,traceContext=nil,traceRecords=nil;
    std::uint32_t environments=0;std::uint64_t fp=0,terminalSerial=0,initialPhysicsGeneration=0,initialTimestampNanoseconds=0;
    bool initialEncoded=false;
    bool traceEnabled=false,traceClosed=false,traceTerminalStored=false;
    HumanBehaviorTraceBinding traceBinding{};
    mrnx_behavior_trace_terminal_request_v1 traceTerminalRequest{};
    mrnx_behavior_trace_terminal_v1 traceTerminal{};
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
    s.tracePage=allocate(sizeof(MRHumanBehaviorTracePageGPU));
    s.traceContext=allocate(sizeof(MRHumanBehaviorTraceAttemptContextGPU));
    s.traceRecords=allocate(sizeof(MRHumanBehaviorTraceRecordGPU));
    auto* trace=static_cast<MRHumanBehaviorTracePageGPU*>(s.tracePage.contents);
    trace->abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;
    trace->structSize=sizeof(*trace);
    trace->status=MR_HUMAN_BEHAVIOR_TRACE_STATUS_DISABLED;
    // Candidate-stage coverage starts unknown. Authoritative terminal owners
    // may add independently proven bits to the trace context; missing bits and
    // contact coverage remain unavailable, never implicit zero violations.
    reset();
}
MetalHumanBehaviorTelemetry::~MetalHumanBehaviorTelemetry()=default;
std::uint64_t MetalHumanBehaviorTelemetry::fingerprint()const noexcept{return state_->fp;}
std::uint64_t MetalHumanBehaviorTelemetry::completedAttempts()const noexcept{return state_->terminalSerial;}
bool MetalHumanBehaviorTelemetry::initialObservationEncoded()const noexcept{return state_->initialEncoded;}
bool MetalHumanBehaviorTelemetry::traceAttached()const noexcept{return state_->traceEnabled;}
bool MetalHumanBehaviorTelemetry::traceFinalized()const noexcept{return state_->traceClosed;}
void MetalHumanBehaviorTelemetry::reset(){auto& s=*state_;s.terminalSerial=0;s.initialEncoded=false;
    for(auto b:{s.candidate,s.release,s.fences,s.reduction})std::memset(b.contents,0,b.length);
    auto* out=static_cast<MRHumanBehaviorReductionGPU*>(s.reduction.contents);
    for(std::uint32_t i=0;i<s.environments;++i){out[i].abiVersion=MR_HUMAN_BEHAVIOR_ABI_VERSION;out[i].programFingerprint=s.fp;
        out[i].initialPhysicsGeneration=s.initialPhysicsGeneration;out[i].initialTimestampNanoseconds=s.initialTimestampNanoseconds;out[i].endNanoseconds=s.initialTimestampNanoseconds;
        out[i].initialPostureValid=2;out[i].initialSettled=2;} // unmeasured reset, explicit unknown
    if(s.traceEnabled){auto* page=static_cast<MRHumanBehaviorTracePageGPU*>(s.tracePage.contents);const auto capacity=page->recordCapacity;const auto instance=page->traceInstanceFingerprint;const auto expected=page->expectedAcceptedRoots;
        std::memset(page,0,sizeof(*page));page->abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;page->structSize=sizeof(*page);page->status=MR_HUMAN_BEHAVIOR_TRACE_STATUS_READY;page->recordCapacity=capacity;page->traceInstanceFingerprint=instance;page->expectedAcceptedRoots=expected;page->pagePreviousRecordFingerprint=instance;page->lastRecordFingerprint=instance;page->lastAfterPhysicsGeneration=s.initialPhysicsGeneration;page->lastAfterAcceptedTimestampNanoseconds=s.initialTimestampNanoseconds;
        std::memset(s.traceContext.contents,0,s.traceContext.length);std::memset(s.traceRecords.contents,0,s.traceRecords.length);s.traceClosed=false;s.traceTerminalStored=false;s.traceTerminalRequest={};s.traceTerminal={};}
}
bool MetalHumanBehaviorTelemetry::traceAttach(const mrnx_behavior_trace_config_v1& config,const HumanBehaviorTraceBinding& binding,std::string& error)noexcept{
    try{auto& s=*state_;require(!s.traceEnabled&&!s.traceClosed&&s.terminalSerial==0,"behavior trace must attach before the first terminal attempt");
        require(config.abi_version==MRNX_BEHAVIOR_TRACE_ABI_V1&&config.struct_size==sizeof(config)&&config.record_capacity>0&&config.record_capacity<=65536u&&config.flags==0&&config.expected_accepted_roots>0&&config.reserved0==0,"invalid behavior trace configuration");
        require(nonzero(binding.metricProgramSHA256)&&binding.modelSourceFingerprint!=0&&binding.acceptedStateProofProgramFingerprint!=0&&binding.clockDomain==MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS&&binding.clockQuantumNanoseconds==MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS,"invalid behavior trace source binding");
        const NSUInteger bytes=static_cast<NSUInteger>(config.record_capacity)*sizeof(MRHumanBehaviorTraceRecordGPU);
        id<MTLBuffer> records=[s.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];require(records!=nil,"behavior trace allocation failed");std::memset(records.contents,0,records.length);
        std::uint64_t instance=14695981039346656037ull;constexpr std::uint32_t domain=0x4e484254u;mix(instance,domain);mix(instance,s.fp);mix(instance,s.cooked.timestepNanoseconds);mix(instance,s.initialPhysicsGeneration);mix(instance,s.initialTimestampNanoseconds);mix(instance,config.record_capacity);mix(instance,config.expected_accepted_roots);mix(instance,binding.metricProgramSHA256);mix(instance,binding.modelSourceFingerprint);mix(instance,binding.acceptedStateProofProgramFingerprint);mix(instance,binding.clockDomain);mix(instance,binding.clockQuantumNanoseconds);require(instance!=0,"zero behavior trace identity");
        MRHumanBehaviorTracePageGPU page{};page.abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;page.structSize=sizeof(page);page.status=MR_HUMAN_BEHAVIOR_TRACE_STATUS_READY;page.recordCapacity=config.record_capacity;page.traceInstanceFingerprint=instance;page.expectedAcceptedRoots=config.expected_accepted_roots;page.pagePreviousRecordFingerprint=instance;page.lastRecordFingerprint=instance;page.lastAfterPhysicsGeneration=s.initialPhysicsGeneration;page.lastAfterAcceptedTimestampNanoseconds=s.initialTimestampNanoseconds;
        std::memcpy(s.tracePage.contents,&page,sizeof(page));std::memset(s.traceContext.contents,0,s.traceContext.length);s.traceRecords=records;s.traceBinding=binding;s.traceEnabled=true;s.traceClosed=false;s.traceTerminalStored=false;s.traceTerminalRequest={};s.traceTerminal={};error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
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
        [e setBuffer:s.tracePage offset:0 atIndex:6];[e setBuffer:s.traceContext offset:0 atIndex:7];[e setBuffer:s.traceRecords offset:0 atIndex:8];
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
bool MetalHumanBehaviorTelemetry::terminal(const MRHumanBehaviorReleaseGPU& r,const MRNumanXHumanMatterJointPublicationFenceGPU* fence,std::string& error)noexcept{return terminal(r,fence,nullptr,error);}
bool MetalHumanBehaviorTelemetry::terminal(const MRHumanBehaviorReleaseGPU& r,const MRNumanXHumanMatterJointPublicationFenceGPU* fence,const HumanBehaviorTraceAttemptContext* trace,std::string& error)noexcept{
    try{auto& s=*state_;require(r.programFingerprint==s.fp&&r.publicationSerial==s.terminalSerial+1&&s.terminalSerial!=std::numeric_limits<std::uint64_t>::max()&&r.transactionFingerprint!=0&&r.slotGeneration!=0&&(r.released==1||r.released==2)&&r.reserved0==0&&r.reserved1==0&&r.reserved2==0,"invalid telemetry terminal identity");
        require(r.released!=1||(fence!=nullptr&&
            (fence->abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION||
             fence->abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2)&&
            fence->structBytes==sizeof(*fence)&&
            fence->status==MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED&&
            fence->fenceFingerprint==r.jointFenceFingerprint&&
            r.jointFenceFingerprint!=0),"telemetry acceptance requires released COMMITTED root");
        MRHumanBehaviorTraceAttemptContextGPU context{};
        if(s.traceEnabled&&!s.traceClosed){context.abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;context.structSize=sizeof(context);
            context.present=trace!=nullptr&&trace->abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&trace->structSize==sizeof(*trace)?1u:2u;
            if(trace!=nullptr){context.controlStep=trace->controlStep;context.runtimeFailureStage=trace->runtimeFailureStage;context.auditCoveredMask=trace->auditCoveredMask;context.auditViolationMask=trace->auditViolationMask;context.forbiddenContactCoverage=trace->forbiddenContactCoverage;context.forbiddenContactCount=trace->forbiddenContactCount;context.reserved0=trace->reserved0;context.reserved1=trace->reserved1;context.reserved2=trace->reserved2;context.basePublicationEpoch=trace->basePublicationEpoch;context.basePhysicsGeneration=trace->basePhysicsGeneration;context.baseAcceptedTimestampNanoseconds=trace->baseAcceptedTimestampNanoseconds;context.baseAcceptedTokenFingerprint=trace->baseAcceptedTokenFingerprint;context.basePublicationFingerprint=trace->basePublicationFingerprint;context.candidateStateProofFingerprint=trace->candidateStateProofFingerprint;context.candidateAcceptedTokenFingerprint=trace->candidateAcceptedTokenFingerprint;context.candidatePublicationFingerprint=trace->candidatePublicationFingerprint;context.afterPublicationEpoch=trace->afterPublicationEpoch;context.afterPhysicsGeneration=trace->afterPhysicsGeneration;context.afterAcceptedTimestampNanoseconds=trace->afterAcceptedTimestampNanoseconds;context.afterAcceptedTokenFingerprint=trace->afterAcceptedTokenFingerprint;context.afterPublicationFingerprint=trace->afterPublicationFingerprint;}}
        if(fence)std::memcpy(s.fences.contents,fence,sizeof(*fence));else std::memset(s.fences.contents,0,s.fences.length);
        std::memcpy(s.release.contents,&r,sizeof(r));std::memcpy(s.traceContext.contents,&context,sizeof(context));s.terminalSerial=r.publicationSerial;error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
bool MetalHumanBehaviorTelemetry::traceDrain(mrnx_behavior_trace_chunk_v1& output,std::span<mrnx_behavior_trace_record_v1> records,std::string& error)noexcept{
    try{auto& s=*state_;require(s.traceEnabled,"behavior trace is not attached");auto* page=static_cast<MRHumanBehaviorTracePageGPU*>(s.tracePage.contents);
        require(page->abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&page->structSize==sizeof(*page)&&page->recordCapacity>0&&page->traceInstanceFingerprint!=0&&page->expectedAcceptedRoots>0&&page->recordCount<=page->recordCapacity&&page->recordCount<=records.size()&&page->drainedRecordCount<=page->totalRecordCount&&page->totalRecordCount-page->drainedRecordCount==page->recordCount&&page->droppedRecordCount<=page->observedAttemptCount&&page->totalRecordCount<=page->observedAttemptCount-page->droppedRecordCount&&page->totalRecordCount<=std::numeric_limits<std::uint64_t>::max()-page->droppedRecordCount&&page->totalRecordCount+page->droppedRecordCount==page->observedAttemptCount&&((page->lastAfterPublicationEpoch==0&&page->lastAfterAcceptedTokenFingerprint==0&&page->lastAfterPublicationFingerprint==0)||(page->lastAfterPublicationEpoch!=0&&page->lastAfterAcceptedTokenFingerprint!=0&&page->lastAfterPublicationFingerprint!=0)),"invalid or undersized behavior trace drain");
        require(page->status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_READY||page->status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_OVERFLOW||page->status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID||page->status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_FINALIZED,"invalid behavior trace state");
        const auto* source=static_cast<const MRHumanBehaviorTraceRecordGPU*>(s.traceRecords.contents);std::uint64_t prior=page->pagePreviousRecordFingerprint;
        for(std::uint64_t i=0;i<page->recordCount;++i){const auto& record=source[i];require(record.abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&record.structSize==sizeof(record)&&record.previousRecordFingerprint==prior&&record.recordFingerprint!=0&&record.recordFingerprint==fnv(&record,offsetof(MRHumanBehaviorTraceRecordGPU,recordFingerprint)),"invalid behavior trace record chain");prior=record.recordFingerprint;}
        require((page->recordCount==0&&page->firstAttemptIndex==0&&page->lastAttemptIndex==0&&page->pagePreviousRecordFingerprint==page->lastRecordFingerprint)||(page->recordCount>0&&page->firstAttemptIndex==source[0].attemptIndex&&page->lastAttemptIndex==source[page->recordCount-1].attemptIndex&&prior==page->lastRecordFingerprint),"invalid behavior trace page bounds");
        mrnx_behavior_trace_chunk_v1 chunk{};chunk.abi_version=MRNX_BEHAVIOR_TRACE_ABI_V1;chunk.struct_size=sizeof(chunk);chunk.trace_status=page->status;chunk.record_capacity=page->recordCapacity;chunk.trace_instance_fingerprint=page->traceInstanceFingerprint;chunk.expected_accepted_roots=page->expectedAcceptedRoots;chunk.chunk_index=page->chunkIndex;chunk.record_count=page->recordCount;chunk.first_attempt_index=page->firstAttemptIndex;chunk.last_attempt_index=page->lastAttemptIndex;chunk.previous_record_fingerprint=page->pagePreviousRecordFingerprint;chunk.last_record_fingerprint=page->lastRecordFingerprint;chunk.observed_attempt_count=page->observedAttemptCount;chunk.total_record_count=page->totalRecordCount;chunk.dropped_record_count=page->droppedRecordCount;chunk.drained_record_count=page->drainedRecordCount+page->recordCount;std::memcpy(chunk.metric_program_sha256,s.traceBinding.metricProgramSHA256.data(),s.traceBinding.metricProgramSHA256.size());chunk.behavior_program_fingerprint=s.fp;chunk.model_source_fingerprint=s.traceBinding.modelSourceFingerprint;chunk.accepted_state_proof_program_fingerprint=s.traceBinding.acceptedStateProofProgramFingerprint;chunk.timestep_nanoseconds=s.cooked.timestepNanoseconds;chunk.initial_timestamp_nanoseconds=s.initialTimestampNanoseconds;chunk.initial_physics_generation=s.initialPhysicsGeneration;chunk.clock_domain=s.traceBinding.clockDomain;chunk.clock_quantum_nanoseconds=s.traceBinding.clockQuantumNanoseconds;
        if(page->recordCount>0)std::memcpy(records.data(),source,page->recordCount*sizeof(MRHumanBehaviorTraceRecordGPU));output=chunk;
        if(page->recordCount>0){page->drainedRecordCount+=page->recordCount;++page->chunkIndex;page->recordCount=0;page->firstAttemptIndex=0;page->lastAttemptIndex=0;page->pagePreviousRecordFingerprint=page->lastRecordFingerprint;std::memset(s.traceRecords.contents,0,s.traceRecords.length);}error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}}
bool MetalHumanBehaviorTelemetry::traceFinalize(const mrnx_behavior_trace_terminal_request_v1& request,const HumanBehaviorTraceFinalContext& finalContext,mrnx_behavior_trace_terminal_v1& output,std::string& error)noexcept{
    try{auto& s=*state_;require(s.traceEnabled,"behavior trace is not attached");require(request.abi_version==MRNX_BEHAVIOR_TRACE_ABI_V1&&request.struct_size==sizeof(request)&&(request.reason==MRNX_BEHAVIOR_TRACE_TERMINAL_COMPLETED_V1||request.reason==MRNX_BEHAVIOR_TRACE_TERMINAL_CANCELLED_V1||request.reason==MRNX_BEHAVIOR_TRACE_TERMINAL_FAILED_V1)&&request.reserved0==0&&request.reserved1==0,"invalid behavior trace terminal request");
        if(s.traceTerminalStored){require(std::memcmp(&request,&s.traceTerminalRequest,sizeof(request))==0,"behavior trace already finalized with a different request");output=s.traceTerminal;error.clear();return true;}
        require(finalContext.abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&finalContext.structSize==sizeof(finalContext),"invalid behavior trace final context");auto* page=static_cast<MRHumanBehaviorTracePageGPU*>(s.tracePage.contents);const auto* reduction=static_cast<const MRHumanBehaviorReductionGPU*>(s.reduction.contents);const auto& r=reduction[0];
        const bool pageShape=page->abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&page->structSize==sizeof(*page)&&page->recordCapacity>0&&page->traceInstanceFingerprint!=0&&page->expectedAcceptedRoots>0&&page->recordCount<=page->recordCapacity&&page->drainedRecordCount<=page->totalRecordCount&&page->totalRecordCount-page->drainedRecordCount==page->recordCount&&page->droppedRecordCount<=page->observedAttemptCount&&page->totalRecordCount<=page->observedAttemptCount-page->droppedRecordCount&&page->totalRecordCount<=std::numeric_limits<std::uint64_t>::max()-page->droppedRecordCount&&page->totalRecordCount+page->droppedRecordCount==page->observedAttemptCount&&((page->lastAfterPublicationEpoch==0&&page->lastAfterAcceptedTokenFingerprint==0&&page->lastAfterPublicationFingerprint==0)||(page->lastAfterPublicationEpoch!=0&&page->lastAfterAcceptedTokenFingerprint!=0&&page->lastAfterPublicationFingerprint!=0));
        bool chainValid=pageShape;if(chainValid){const auto* source=static_cast<const MRHumanBehaviorTraceRecordGPU*>(s.traceRecords.contents);auto prior=page->pagePreviousRecordFingerprint;for(std::uint64_t i=0;i<page->recordCount;++i){const auto& record=source[i];if(record.abiVersion!=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION||record.structSize!=sizeof(record)||record.previousRecordFingerprint!=prior||record.recordFingerprint==0||record.recordFingerprint!=fnv(&record,offsetof(MRHumanBehaviorTraceRecordGPU,recordFingerprint))){chainValid=false;break;}prior=record.recordFingerprint;}if(chainValid)chainValid=page->recordCount>0?page->firstAttemptIndex==source[0].attemptIndex&&page->lastAttemptIndex==source[page->recordCount-1].attemptIndex&&prior==page->lastRecordFingerprint:page->firstAttemptIndex==0&&page->lastAttemptIndex==0&&page->pagePreviousRecordFingerprint==page->lastRecordFingerprint;}
        const bool countClosure=r.status==0&&r.publicationSerial==s.terminalSerial&&r.publicationSerial==r.acceptedRootCount+r.rejectedAttemptCount&&page->observedAttemptCount==r.publicationSerial&&page->totalRecordCount==page->observedAttemptCount&&page->droppedRecordCount==0&&page->acceptedRecordCount==r.acceptedRootCount&&page->rejectedRecordCount==r.rejectedAttemptCount;
        const bool clockClosure=r.acceptedRootCount<=(std::numeric_limits<std::uint64_t>::max()-r.initialTimestampNanoseconds)/s.cooked.timestepNanoseconds&&r.endNanoseconds==r.initialTimestampNanoseconds+r.acceptedRootCount*s.cooked.timestepNanoseconds;
        const bool finalClosure=page->observedAttemptCount>0&&page->lastAfterPublicationEpoch==finalContext.publicationEpoch&&page->lastAfterPhysicsGeneration==finalContext.physicsGeneration&&page->lastAfterAcceptedTimestampNanoseconds==finalContext.timestampNanoseconds&&page->lastAfterAcceptedTokenFingerprint==finalContext.acceptedTokenFingerprint&&page->lastAfterPublicationFingerprint==finalContext.publicationFingerprint&&finalContext.physicsGeneration==r.initialPhysicsGeneration+r.acceptedRootCount&&finalContext.timestampNanoseconds==r.endNanoseconds&&finalContext.brainGeneration!=0&&finalContext.sensorGeneration!=0&&finalContext.acceptedTokenFingerprint!=0&&finalContext.publicationFingerprint!=0;
        const bool complete=request.reason==MRNX_BEHAVIOR_TRACE_TERMINAL_COMPLETED_V1&&request.caller_exit_code==0&&finalContext.quiescent&&!finalContext.terminalQuarantine&&page->status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_READY&&page->recordCount==0&&page->drainedRecordCount==page->totalRecordCount&&chainValid&&countClosure&&clockClosure&&finalClosure&&r.acceptedRootCount==page->expectedAcceptedRoots;
        constexpr std::uint32_t qualification=MRNX_BEHAVIOR_TRACE_EVIDENCE_SOURCE_BOUND_METRIC_V1;
        mrnx_behavior_trace_terminal_v1 terminal{};terminal.abi_version=MRNX_BEHAVIOR_TRACE_ABI_V1;terminal.struct_size=sizeof(terminal);terminal.capture_status=complete?MRNX_BEHAVIOR_TRACE_CAPTURE_COMPLETE_V1:((request.reason==MRNX_BEHAVIOR_TRACE_TERMINAL_FAILED_V1||request.caller_exit_code!=0||finalContext.terminalQuarantine||r.status!=0||page->status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID||!pageShape||!chainValid)?MRNX_BEHAVIOR_TRACE_CAPTURE_FAILED_V1:MRNX_BEHAVIOR_TRACE_CAPTURE_PARTIAL_V1);terminal.reason=request.reason;terminal.caller_exit_code=request.caller_exit_code;terminal.qualification_flags=qualification;terminal.unavailable_evidence_flags=kAllTraceEvidence&~qualification;terminal.expected_accepted_roots=page->expectedAcceptedRoots;terminal.observed_attempt_count=page->observedAttemptCount;terminal.total_record_count=page->totalRecordCount;terminal.drained_record_count=page->drainedRecordCount;terminal.dropped_record_count=page->droppedRecordCount;terminal.chunk_count=page->chunkIndex;terminal.accepted_root_count=r.acceptedRootCount;terminal.rejected_attempt_count=r.rejectedAttemptCount;terminal.completed_attempt_count=r.publicationSerial;terminal.initial_timestamp_nanoseconds=r.initialTimestampNanoseconds;terminal.end_timestamp_nanoseconds=r.endNanoseconds;terminal.timestep_nanoseconds=s.cooked.timestepNanoseconds;terminal.final_publication_epoch=finalContext.publicationEpoch;terminal.final_physics_generation=finalContext.physicsGeneration;terminal.final_brain_generation=finalContext.brainGeneration;terminal.final_sensor_generation=finalContext.sensorGeneration;terminal.final_timestamp_nanoseconds=finalContext.timestampNanoseconds;terminal.final_accepted_token_fingerprint=finalContext.acceptedTokenFingerprint;terminal.final_publication_fingerprint=finalContext.publicationFingerprint;terminal.behavior_program_fingerprint=s.fp;terminal.last_record_fingerprint=page->lastRecordFingerprint;terminal.terminal_fingerprint=fnv(&terminal,offsetof(mrnx_behavior_trace_terminal_v1,terminal_fingerprint));require(terminal.terminal_fingerprint!=0,"zero behavior trace terminal fingerprint");
        page->status=MR_HUMAN_BEHAVIOR_TRACE_STATUS_FINALIZED;s.traceClosed=true;s.traceTerminalStored=true;s.traceTerminalRequest=request;s.traceTerminal=terminal;output=terminal;error.clear();return true;
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
