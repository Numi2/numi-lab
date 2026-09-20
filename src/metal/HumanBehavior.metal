#include <metal_stdlib>
using namespace metal;
#include "metalrobo/human_behavior_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/numanx_human_matter_gpu.h"

namespace {
// Error-free transforms require precise compilation (no fast-math reassociation).
inline float2 addPair(float2 a,float2 b) { auto r=mrCompensatedAdd({a.x,a.y},{b.x,b.y});return {r.high,r.low}; }
inline float2 multiplyPair(float2 a,float b) { auto r=mrCompensatedAdd(mrCompensatedProduct(a.x,b),mrCompensatedProduct(a.y,b));return {r.high,r.low}; }
inline bool lessPair(float2 a,float2 b) { return a.x<b.x||(a.x==b.x&&a.y<b.y); }
inline float3 rotate(float4 q,float3 v) { float3 t=2.0f*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t); }
inline bool finitePair(float2 x){return all(isfinite(x));}
inline ulong fenceHash(const device MRNumanXHumanMatterJointPublicationFenceGPU& f) {
    const device uchar* bytes=reinterpret_cast<const device uchar*>(&f);ulong h=14695981039346656037ul;
    for(uint i=0;i<120u;++i){h^=bytes[i];h*=1099511628211ul;}return h;
}
inline ulong traceRecordHash(const thread MRHumanBehaviorTraceRecordGPU& r) {
    const thread uchar* bytes=reinterpret_cast<const thread uchar*>(&r);ulong h=14695981039346656037ul;
    for(uint i=0;i<264u;++i){h^=bytes[i];h*=1099511628211ul;}return h;
}
inline void traceDrop(device MRHumanBehaviorTracePageGPU& page,ulong attempt,uint status) {
    page.observedAttemptCount=attempt;
    if(page.droppedRecordCount!=ULONG_MAX)++page.droppedRecordCount;
    if(status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID||page.status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID)
        page.status=MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID;
    else page.status=MR_HUMAN_BEHAVIOR_TRACE_STATUS_OVERFLOW;
}
inline void appendTrace(
    constant MRHumanBehaviorProgramGPU& p,
    const thread MRHumanBehaviorCandidateGPU& sample,
    const thread MRHumanBehaviorReleaseGPU& release,
    const device MRNumanXHumanMatterJointPublicationFenceGPU& fence,
    device MRHumanBehaviorTracePageGPU& page,
    const device MRHumanBehaviorTraceAttemptContextGPU& context,
    device MRHumanBehaviorTraceRecordGPU* records) {
    if(page.status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_DISABLED||page.status==MR_HUMAN_BEHAVIOR_TRACE_STATUS_FINALIZED)return;
    const bool pageValid=page.abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&
        page.structSize==sizeof(MRHumanBehaviorTracePageGPU)&&page.recordCapacity>0&&
        page.traceInstanceFingerprint!=0&&page.expectedAcceptedRoots>0&&
        page.recordCount<=page.recordCapacity&&page.drainedRecordCount<=page.totalRecordCount&&
        page.totalRecordCount-page.drainedRecordCount==page.recordCount&&
        page.droppedRecordCount<=page.observedAttemptCount&&
        page.totalRecordCount<=page.observedAttemptCount-page.droppedRecordCount&&
        page.totalRecordCount<=ULONG_MAX-page.droppedRecordCount&&
        page.totalRecordCount+page.droppedRecordCount==page.observedAttemptCount&&
        ((page.lastAfterPublicationEpoch==0u&&page.lastAfterAcceptedTokenFingerprint==0u&&page.lastAfterPublicationFingerprint==0u)||
         (page.lastAfterPublicationEpoch!=0u&&page.lastAfterAcceptedTokenFingerprint!=0u&&page.lastAfterPublicationFingerprint!=0u))&&
        page.observedAttemptCount!=ULONG_MAX&&release.publicationSerial==page.observedAttemptCount+1ul;
    if(!pageValid){traceDrop(page,release.publicationSerial,MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID);return;}
    const bool sampleMatches=sample.abiVersion==MR_HUMAN_BEHAVIOR_ABI_VERSION&&sample.programFingerprint==p.fingerprint&&
        sample.transactionFingerprint==release.transactionFingerprint&&sample.linearizationEpoch==release.linearizationEpoch&&
        sample.slotGeneration==release.slotGeneration&&sample.physicsGeneration==release.physicsGeneration&&
        sample.acceptedTimestampNanoseconds==release.acceptedTimestampNanoseconds;
    bool contextValid=context.abiVersion==MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION&&
        context.structSize==sizeof(MRHumanBehaviorTraceAttemptContextGPU)&&context.present==1u&&context.controlStep!=0u&&
        context.reserved0==0u&&context.reserved1==0u&&context.reserved2==0u&&
        context.basePublicationEpoch==page.lastAfterPublicationEpoch&&
        context.basePhysicsGeneration==page.lastAfterPhysicsGeneration&&
        context.baseAcceptedTimestampNanoseconds==page.lastAfterAcceptedTimestampNanoseconds&&
        context.baseAcceptedTokenFingerprint==page.lastAfterAcceptedTokenFingerprint&&
        context.basePublicationFingerprint==page.lastAfterPublicationFingerprint&&
        context.basePhysicsGeneration!=ULONG_MAX&&release.physicsGeneration==context.basePhysicsGeneration+1ul&&
        context.baseAcceptedTimestampNanoseconds<=ULONG_MAX-p.timestepNanoseconds&&
        release.acceptedTimestampNanoseconds==context.baseAcceptedTimestampNanoseconds+p.timestepNanoseconds;
    if(release.released==1u) {
        contextValid=contextValid&&context.basePublicationEpoch!=ULONG_MAX&&
            fence.abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2&&
            fence.structBytes==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES&&
            fence.status==MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED&&
            context.afterPublicationEpoch==context.basePublicationEpoch+1ul&&
            context.afterPhysicsGeneration==release.physicsGeneration&&
            context.afterAcceptedTimestampNanoseconds==release.acceptedTimestampNanoseconds&&
            context.candidateStateProofFingerprint!=0&&context.candidateAcceptedTokenFingerprint!=0&&
            context.candidatePublicationFingerprint!=0&&context.afterPublicationFingerprint!=0&&
            context.afterAcceptedTokenFingerprint==context.candidateAcceptedTokenFingerprint&&
            context.candidateAcceptedTokenFingerprint==fence.physicsTokenFingerprint&&
            fence.controlStep==context.controlStep&&release.jointFenceFingerprint==fence.fenceFingerprint;
    } else {
        const bool noCandidateProof=context.candidateStateProofFingerprint==0&&
            context.candidateAcceptedTokenFingerprint==0&&context.candidatePublicationFingerprint==0;
        const bool completeCandidateProof=context.candidateStateProofFingerprint!=0&&
            context.candidateAcceptedTokenFingerprint!=0&&context.candidatePublicationFingerprint!=0;
        contextValid=contextValid&&(noCandidateProof||completeCandidateProof)&&
            context.afterPublicationEpoch==context.basePublicationEpoch&&
            context.afterPhysicsGeneration==context.basePhysicsGeneration&&
            context.afterAcceptedTimestampNanoseconds==context.baseAcceptedTimestampNanoseconds&&
            context.afterAcceptedTokenFingerprint==context.baseAcceptedTokenFingerprint&&
            context.afterPublicationFingerprint==context.basePublicationFingerprint&&
            ((context.basePublicationEpoch==0u&&context.basePublicationFingerprint==0u)||
             (context.basePublicationEpoch!=0u&&context.basePublicationFingerprint!=0u))&&
            release.jointFenceFingerprint==0u;
    }
    if(!contextValid){traceDrop(page,release.publicationSerial,MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID);return;}
    page.lastAfterPublicationEpoch=context.afterPublicationEpoch;
    page.lastAfterPhysicsGeneration=context.afterPhysicsGeneration;
    page.lastAfterAcceptedTimestampNanoseconds=context.afterAcceptedTimestampNanoseconds;
    page.lastAfterAcceptedTokenFingerprint=context.afterAcceptedTokenFingerprint;
    page.lastAfterPublicationFingerprint=context.afterPublicationFingerprint;
    if(page.recordCount>=page.recordCapacity||page.totalRecordCount==ULONG_MAX){traceDrop(page,release.publicationSerial,MR_HUMAN_BEHAVIOR_TRACE_STATUS_OVERFLOW);return;}
    MRHumanBehaviorTraceRecordGPU record={};record.abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;record.structSize=sizeof(record);
    record.disposition=release.released==1u?MR_HUMAN_BEHAVIOR_TRACE_JOINTLY_PUBLISHED:MR_HUMAN_BEHAVIOR_TRACE_REJECTED_RELEASED;
    record.metricKind=release.released==1u?MR_HUMAN_BEHAVIOR_TRACE_METRIC_ACCEPTED:
        (sampleMatches&&sample.status==0u?MR_HUMAN_BEHAVIOR_TRACE_METRIC_REJECTED_CANDIDATE:MR_HUMAN_BEHAVIOR_TRACE_METRIC_UNAVAILABLE);
    record.controlStep=context.controlStep;record.runtimeFailureStage=context.runtimeFailureStage;
    record.candidateStatus=sampleMatches?sample.status:2u;
    if(record.metricKind!=MR_HUMAN_BEHAVIOR_TRACE_METRIC_UNAVAILABLE){record.postureValid=sample.postureValid;record.settled=sample.settled;record.auditCoveredMask=sample.audit.coveredMask;record.auditViolationMask=sample.audit.violationMask;record.forbiddenContactCoverage=sample.audit.forbiddenContactCoverage;record.forbiddenContactCount=sample.audit.forbiddenContactCount;record.valueHigh[0]=sample.valueHigh.x;record.valueHigh[1]=sample.valueHigh.y;record.valueHigh[2]=sample.valueHigh.z;record.valueHigh[3]=sample.valueHigh.w;record.valueLow[0]=sample.valueLow.x;record.valueLow[1]=sample.valueLow.y;record.valueLow[2]=sample.valueLow.z;record.valueLow[3]=sample.valueLow.w;}
    record.attemptIndex=release.publicationSerial;record.transactionFingerprint=release.transactionFingerprint;record.linearizationEpoch=release.linearizationEpoch;record.slotGeneration=release.slotGeneration;
    record.basePublicationEpoch=context.basePublicationEpoch;record.basePhysicsGeneration=context.basePhysicsGeneration;record.baseAcceptedTimestampNanoseconds=context.baseAcceptedTimestampNanoseconds;record.baseAcceptedTokenFingerprint=context.baseAcceptedTokenFingerprint;
    record.candidatePhysicsGeneration=release.physicsGeneration;record.candidateTimestampNanoseconds=release.acceptedTimestampNanoseconds;record.candidateStateProofFingerprint=context.candidateStateProofFingerprint;record.candidateAcceptedTokenFingerprint=context.candidateAcceptedTokenFingerprint;record.candidatePublicationFingerprint=context.candidatePublicationFingerprint;
    record.afterPublicationEpoch=context.afterPublicationEpoch;record.afterPhysicsGeneration=context.afterPhysicsGeneration;record.afterAcceptedTimestampNanoseconds=context.afterAcceptedTimestampNanoseconds;record.afterAcceptedTokenFingerprint=context.afterAcceptedTokenFingerprint;record.afterPublicationFingerprint=context.afterPublicationFingerprint;record.jointFenceFingerprint=release.jointFenceFingerprint;
    record.previousRecordFingerprint=page.lastRecordFingerprint;record.recordFingerprint=traceRecordHash(record);
    if(record.recordFingerprint==0u){traceDrop(page,release.publicationSerial,MR_HUMAN_BEHAVIOR_TRACE_STATUS_INVALID);return;}
    records[page.recordCount]=record;if(page.recordCount==0u)page.firstAttemptIndex=release.publicationSerial;
    page.lastAttemptIndex=release.publicationSerial;++page.recordCount;++page.totalRecordCount;page.observedAttemptCount=release.publicationSerial;page.lastRecordFingerprint=record.recordFingerprint;
    if(release.released==1u){++page.acceptedRecordCount;++page.acceptedProofRecordCount;if(sample.audit.forbiddenContactCoverage==1u)++page.forbiddenContactCoveredAcceptedCount;}else ++page.rejectedRecordCount;
    if(sampleMatches&&sample.status==0u&&sample.audit.coveredMask==MR_HUMAN_BEHAVIOR_COMPLETE_AUDIT_MASK)++page.nativeAuditRecordCount;
}
}

kernel void human_behavior_measure(
    constant MRHumanBehaviorProgramGPU& p [[buffer(0)]],
    constant MRHumanBehaviorDispatchGPU& d [[buffer(1)]],
    const device MRArticulatedBodyPoseGPU* bodies [[buffer(2)]],
    const device float4* bodyLow [[buffer(3)]],
    const device float* jacobians [[buffer(4)]],
    const device float* velocity [[buffer(5)]],
    const device MRHumanBehaviorAuditGPU* audit [[buffer(6)]],
    device MRHumanBehaviorCandidateGPU* samples [[buffer(7)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    MRHumanBehaviorCandidateGPU sample={};sample.abiVersion=MR_HUMAN_BEHAVIOR_ABI_VERSION;sample.status=2;
    sample.programFingerprint=p.fingerprint;sample.transactionFingerprint=d.transactionFingerprint;
    sample.linearizationEpoch=d.linearizationEpoch;sample.slotGeneration=d.slotGeneration;
    sample.physicsGeneration=d.physicsGeneration;sample.acceptedTimestampNanoseconds=d.acceptedTimestampNanoseconds;
    sample.audit=audit[env];
    if(p.abiVersion!=MR_HUMAN_BEHAVIOR_ABI_VERSION||p.fingerprint==0||p.dofCount==0||p.task>2||
       p.rootBody>=p.bodyCount||p.trunkBody>=p.bodyCount||p.velocityBody>=p.bodyCount||d.bodyPoseStride<p.bodyCount||
       d.vStride<p.dofCount||d.reserved0||d.reserved1||d.reserved2||
       ulong(d.pointJacobianStride)<(ulong(d.bodyJacobianPointOffset)+4ul*p.bodyCount)*3ul*p.dofCount) {samples[env]=sample;return;}
    const uint bodyBase=env*d.bodyPoseStride;
    const auto root=bodies[bodyBase+p.rootBody],trunk=bodies[bodyBase+p.trunkBody];
    const float4 low=bodyLow[bodyBase+p.rootBody];
    if(!all(isfinite(root.position))||!all(isfinite(root.orientation))||!all(isfinite(trunk.orientation))||
       !all(isfinite(low))||low.w!=0||fabs(dot(root.orientation,root.orientation)-1.0f)>1e-4f||
       fabs(dot(trunk.orientation,trunk.orientation)-1.0f)>1e-4f){samples[env]=sample;return;}
    const float3 offsetHigh=rotate(root.orientation,p.rootOriginHigh.xyz);
    const float3 offsetLow=rotate(root.orientation,p.rootOriginLow.xyz);
    float2 height={0,0};
    for(uint axis=0;axis<3;++axis) {
        float2 relative=addPair({root.position[axis],low[axis]},{-p.worldOriginHigh[axis],-p.worldOriginLow[axis]});
        relative=addPair(relative,{offsetHigh[axis],offsetLow[axis]});
        height=addPair(height,multiplyPair(relative,p.worldUp[axis]));
    }
    const float3 trunkUp=rotate(trunk.orientation,p.trunkUp.xyz);
    const float tilt=acos(clamp(dot(trunkUp,p.worldUp.xyz),-1.0f,1.0f));
    float2 v[3]={{0,0},{0,0},{0,0}};
    const ulong probeBase=ulong(env)*d.pointJacobianStride+(ulong(d.bodyJacobianPointOffset)+4ul*p.velocityBody)*3ul*p.dofCount;
    for(uint axis=0;axis<3;++axis)for(uint j=0;j<p.dofCount;++j) {
        const float a=jacobians[probeBase+axis*p.dofCount+j],b=velocity[ulong(env)*d.vStride+j];
        const float product=a*b;v[axis]=addPair(v[axis],{product,fma(a,b,-product)});
    }
    float2 vertical={0,0},forward={0,0};
    for(uint axis=0;axis<3;++axis){vertical=addPair(vertical,multiplyPair(v[axis],p.worldUp[axis]));forward=addPair(forward,multiplyPair(v[axis],p.worldForward[axis]));}
    float2 planar2={0,0};
    for(uint axis=0;axis<3;++axis){float2 a=addPair(v[axis],multiplyPair(vertical,-p.worldUp[axis]));float product=a.x*a.x;
        planar2=addPair(planar2,{product,fma(a.x,a.x,-product)+2.0f*a.x*a.y+a.y*a.y});}
    const float planar=sqrt(max(0.0f,planar2.x+planar2.y));
    if(!finitePair(height)||!isfinite(tilt)||!isfinite(planar)||!finitePair(forward)||
       (sample.audit.coveredMask&~MR_HUMAN_BEHAVIOR_COMPLETE_AUDIT_MASK)!=0||
       (sample.audit.violationMask&~sample.audit.coveredMask)!=0||sample.audit.forbiddenContactCoverage>1){samples[env]=sample;return;}
    const bool geometric= !lessPair(height,{p.limitsHigh.x,p.limitsLow.x})&&!lessPair({p.limitsHigh.y,p.limitsLow.y},{tilt,0});
    sample.postureValid=geometric&&sample.audit.forbiddenContactCount==0;
    sample.settled=sample.postureValid&&(p.task==2||!lessPair({p.limitsHigh.z,p.limitsLow.z},{planar,0}));
    sample.valueHigh={height.x,tilt,planar,forward.x};sample.valueLow={height.y,0,0,forward.y};sample.status=0;
    samples[env]=sample;
}

// Seeds only the reset observations in the reducer. A reset sample is
// authoritative only when it is produced by the same owner pass and exact
// source-bound program as later candidates; it never contributes to accepted
// root counts or outcomes.
kernel void human_behavior_seed_initial(
    constant MRHumanBehaviorProgramGPU& p [[buffer(0)]],
    constant MRHumanBehaviorDispatchGPU& d [[buffer(1)]],
    const device MRHumanBehaviorCandidateGPU* samples [[buffer(2)]],
    device MRHumanBehaviorReductionGPU* reductions [[buffer(3)]],
    uint env [[thread_position_in_grid]]) {
    if (env >= 1u) return;
    auto out = reductions[env];
    const auto s = samples[env];
    const bool valid = out.status == 0u && out.programFingerprint == p.fingerprint &&
        out.initialPostureValid == 2u && out.initialSettled == 2u &&
        d.physicsGeneration == out.initialPhysicsGeneration &&
        d.acceptedTimestampNanoseconds == out.initialTimestampNanoseconds &&
        s.abiVersion == MR_HUMAN_BEHAVIOR_ABI_VERSION && s.status == 0u &&
        s.programFingerprint == p.fingerprint &&
        s.physicsGeneration == out.initialPhysicsGeneration &&
        s.acceptedTimestampNanoseconds == out.initialTimestampNanoseconds;
    if (!valid) {
        out.status = 3u;
    } else {
        out.initialPostureValid = s.postureValid ? 1u : 0u;
        out.initialSettled = s.settled ? 1u : 0u;
    }
    reductions[env] = out;
}

kernel void human_behavior_reduce(
    constant MRHumanBehaviorProgramGPU& p [[buffer(0)]],
    constant uint& environments [[buffer(1)]],
    const device MRHumanBehaviorCandidateGPU* samples [[buffer(2)]],
    const device MRHumanBehaviorReleaseGPU* release [[buffer(3)]],
    const device MRNumanXHumanMatterJointPublicationFenceGPU* fences [[buffer(4)]],
    device MRHumanBehaviorReductionGPU* reductions [[buffer(5)]],
    device MRHumanBehaviorTracePageGPU* tracePage [[buffer(6)]],
    const device MRHumanBehaviorTraceAttemptContextGPU* traceContext [[buffer(7)]],
    device MRHumanBehaviorTraceRecordGPU* traceRecords [[buffer(8)]],uint env [[thread_position_in_grid]]) {
    if(env>=environments)return;
    const auto r=release[env];if(r.released==0)return;
    auto out=reductions[env];if(out.status!=0)return;
    const auto s=samples[env];
    bool valid=out.abiVersion==MR_HUMAN_BEHAVIOR_ABI_VERSION&&out.programFingerprint==p.fingerprint&&
        r.released<=2&&r.programFingerprint==p.fingerprint&&r.publicationSerial!=0&&r.reserved0==0&&r.reserved1==0&&r.reserved2==0;
    const bool sampleMatches=s.abiVersion==MR_HUMAN_BEHAVIOR_ABI_VERSION&&s.programFingerprint==p.fingerprint&&
        s.transactionFingerprint==r.transactionFingerprint&&s.linearizationEpoch==r.linearizationEpoch&&s.slotGeneration==r.slotGeneration&&
        s.physicsGeneration==r.physicsGeneration&&s.acceptedTimestampNanoseconds==r.acceptedTimestampNanoseconds;
    if(r.publicationSerial==out.publicationSerial) {
        // Repeated final flush is a no-op only for the exact same root.
        if(!valid||r.transactionFingerprint!=out.lastTransactionFingerprint||r.jointFenceFingerprint!=out.lastJointFenceFingerprint)reductions[env].status=3;
        return;
    }
    valid=valid&&r.publicationSerial==out.publicationSerial+1&&out.publicationSerial!=ULONG_MAX;
    if(!valid){reductions[env].status=3;return;}
    if(r.released==2) {
        if(r.jointFenceFingerprint!=0||out.rejectedAttemptCount==ULONG_MAX){reductions[env].status=4;return;}
        ++out.rejectedAttemptCount;
    } else {
        const device auto& f=fences[env];
        valid=sampleMatches&&s.status==0&&
            (f.abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION||
             f.abiVersion==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2)&&
            f.structBytes==MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES&&f.status==MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED&&
            f.environment==env&&f.substepIndex==0&&f.physicsSubstepCount==1&&f.reserved0==0&&
            f.transactionFingerprint==r.transactionFingerprint&&f.linearizationEpoch==r.linearizationEpoch&&f.slotGeneration==r.slotGeneration&&
            f.fenceFingerprint!=0&&f.fenceFingerprint==r.jointFenceFingerprint&&f.fenceFingerprint==fenceHash(f)&&
            f.physicsTokenFingerprint!=0&&f.brainShadowStateFingerprint!=0&&f.brainWitnessFingerprint!=0&&f.jointCommitFingerprint!=0&&
            out.acceptedRootCount<ULONG_MAX-out.initialPhysicsGeneration&&r.physicsGeneration==out.initialPhysicsGeneration+out.acceptedRootCount+1&&
            out.endNanoseconds<=ULONG_MAX-p.timestepNanoseconds&&r.acceptedTimestampNanoseconds==out.endNanoseconds+p.timestepNanoseconds;
        if(!valid){reductions[env].status=3;return;}
        ++out.acceptedRootCount;++out.metricSampleCount;out.endNanoseconds=r.acceptedTimestampNanoseconds;
        if(s.audit.coveredMask==MR_HUMAN_BEHAVIOR_COMPLETE_AUDIT_MASK&&s.audit.forbiddenContactCoverage==1)++out.auditCoveredRootCount;
        if(!s.postureValid)++out.postureViolationCount;
        out.settledSuffixSteps=s.settled?out.settledSuffixSteps+1:0;
        if(out.acceptedRootCount==1||lessPair(float2(s.valueHigh.x,s.valueLow.x),{out.extremaHigh.x,out.extremaLow.x})) {
            out.extremaHigh.x=s.valueHigh.x;out.extremaLow.x=s.valueLow.x;
        }
        out.extremaHigh.y=max(out.extremaHigh.y,s.valueHigh.y);out.extremaHigh.z=max(out.extremaHigh.z,s.valueHigh.z);
        if(p.task==2) {
            float2 e=addPair({s.valueHigh.w,s.valueLow.w},{-p.limitsHigh.w,-p.limitsLow.w});float product=e.x*e.x;
            float2 squared={product,fma(e.x,e.x,-product)+2.0f*e.x*e.y+e.y*e.y};
            float2 sum=addPair({out.extremaHigh.w,out.extremaLow.w},squared);
            if(!finitePair(sum)){reductions[env].status=4;return;}
            out.extremaHigh.w=sum.x;out.extremaLow.w=sum.y;++out.speedErrorSampleCount;
        }
    }
    if(sampleMatches&&s.status==0) {
        if(s.audit.coveredMask==MR_HUMAN_BEHAVIOR_COMPLETE_AUDIT_MASK)++out.auditCoveredAttemptCount;
        for(uint bit=0;bit<8;++bit)if(s.audit.violationMask&(1u<<bit))++out.auditViolations[bit];
    }
    out.publicationSerial=r.publicationSerial;out.lastTransactionFingerprint=r.transactionFingerprint;out.lastJointFenceFingerprint=r.jointFenceFingerprint;
    reductions[env]=out;
    appendTrace(p,s,r,fences[env],tracePage[env],traceContext[env],traceRecords);
}
