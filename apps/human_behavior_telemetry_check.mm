// Synthetic native measurement/publication controls only; no physical stepping.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/MetalHumanBehaviorTelemetry.hpp"
#include "metalrobo/compensated_translation_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace metalrobo;
namespace {
void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
std::uint64_t hashFence(const MRNumanXHumanMatterJointPublicationFenceGPU& f){
    const auto* p=reinterpret_cast<const std::uint8_t*>(&f);std::uint64_t h=14695981039346656037ull;
    for(unsigned i=0;i<120;++i){h^=p[i];h*=1099511628211ull;}return h;
}
CompiledHumanBehaviorProgram program(){HumanBehaviorProgramSource s;s.bodyCount=1;s.nq=7;s.nv=6;s.task=2;s.timestepNanoseconds=25000;
    for(auto& h:s.sourceSHA256)h.fill(1);s.numerical={0,0,0,0,0,1,1,0,0,0,0,1,1.0+std::ldexp(1.0,-24),.3,1,.5};
    s.forbiddenBodies={{{0,0}}};HumanBehaviorCompileBinding b;b.bodyCount=1;b.nq=7;b.nv=6;b.timestepNanoseconds=25000;
    b.sourceArchiveSHA256=s.sourceSHA256[0];b.rigidSHA256=s.sourceSHA256[1];b.sourceToCore={{{0,0}}};b.cookedCOMOffset={{{0,0,0}}};
    CompiledHumanBehaviorProgram out;std::string error;need(compileHumanBehaviorProgram(s,b,out,error),error.c_str());return out;
}
struct Driver {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue;
    id<MTLBuffer> bodies,low,jacobian,velocity;std::string path,error;CompiledHumanBehaviorProgram cooked=program();
    Driver(const char* p):queue([device newCommandQueue]),path(p){need(device!=nil&&queue!=nil,"physical Metal device unavailable");
        auto buffer=[&](NSUInteger size){id<MTLBuffer> b=[device newBufferWithLength:size options:MTLResourceStorageModeShared];need(b!=nil,"test buffer allocation");std::memset(b.contents,0,size);return b;};
        bodies=buffer(sizeof(MRArticulatedBodyPoseGPU));low=buffer(sizeof(mr_float4));jacobian=buffer(72*sizeof(float));velocity=buffer(6*sizeof(float));
        auto* pose=static_cast<MRArticulatedBodyPoseGPU*>(bodies.contents);pose->orientation={0,0,0,1};
        auto* j=static_cast<float*>(jacobian.contents);j[0]=j[7]=j[14]=1;static_cast<float*>(velocity.contents)[0]=.75f;
    }
    MetalHumanBehaviorTelemetry telemetry(std::uint64_t timestamp=100000){return MetalHumanBehaviorTelemetry((__bridge void*)device,cooked,path,1,4,timestamp);}
    void command(const auto& encode){id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"test command allocation");need(encode((__bridge void*)cb),error.c_str());[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"test Metal command failure");}
    void flush(MetalHumanBehaviorTelemetry& t){command([&](void* cb){return t.encodeFlush(cb,error);});}
    void measure(MetalHumanBehaviorTelemetry& t,std::uint64_t serial,float correction,std::uint64_t acceptedOrdinal=0){
        if(acceptedOrdinal==0)acceptedOrdinal=serial;
        MRCompensatedRootTranslationGPU root{};root.reference={0,0,1,0};root.displacement={0,0,std::ldexp(1.0f,-24),0};root.correction={0,0,correction,0};
        const auto position=mrCompensatedTranslationPosition(root,{0,0,0,0});
        static_cast<MRArticulatedBodyPoseGPU*>(bodies.contents)->position=position.high;*static_cast<mr_float4*>(low.contents)=position.low;
        command([&](void* cb){if(!t.encodeFlush(cb,error))return false;MetalNumanXHumanMatterPass p{};
            p.phase=MetalNumanXHumanMatterPhase::postDynamics;p.commandBuffer=cb;p.environmentCount=1;p.stepCount=1;p.physicsSubstepCount=1;
            p.bodyPoses=(__bridge void*)bodies;p.bodyPositionLow=(__bridge void*)low;p.pointJacobians=(__bridge void*)jacobian;p.v=(__bridge void*)velocity;
            p.bodyPoseStride=1;p.bodyCount=1;p.qCoordinateCount=7;p.dofCount=6;p.vStride=6;p.pointWorldStride=4;p.pointJacobianStride=72;
            p.transactionFingerprint=100+serial;p.linearizationEpoch=20+serial;p.slotGeneration=serial;
            return t.encodeCandidate(p,4+acceptedOrdinal,100000+acceptedOrdinal*25000,error);});
    }
    auto fence(std::uint64_t serial){MRNumanXHumanMatterJointPublicationFenceGPU f{};
        f.abiVersion=1;f.structBytes=sizeof(f);f.status=MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED;f.controlStep=static_cast<std::uint32_t>(serial);f.physicsSubstepCount=1;
        f.ownerProgramFingerprint=10;f.transactionFingerprint=100+serial;f.linearizationEpoch=20+serial;f.slotGeneration=serial;
        f.physicsTokenFingerprint=11;f.brainProgramFingerprint=12;f.brainShadowStateFingerprint=13;f.brainWitnessFingerprint=14;f.appliedDecisionFingerprint=15;f.jointCommitFingerprint=16;f.brainGeneration=serial;
        f.fenceFingerprint=hashFence(f);return f;
    }
    auto release(MetalHumanBehaviorTelemetry& t,std::uint64_t serial,bool accepted,std::uint64_t acceptedOrdinal=0){if(acceptedOrdinal==0)acceptedOrdinal=serial;MRHumanBehaviorReleaseGPU r{};
        r.programFingerprint=t.fingerprint();r.transactionFingerprint=100+serial;r.linearizationEpoch=20+serial;r.slotGeneration=serial;
        r.physicsGeneration=4+acceptedOrdinal;r.acceptedTimestampNanoseconds=100000+acceptedOrdinal*25000;r.publicationSerial=t.completedAttempts()+1;r.released=accepted?1:2;
        if(accepted)r.jointFenceFingerprint=fence(serial).fenceFingerprint;return r;
    }
};
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
    need(argc==2,"usage: human_behavior_telemetry_check MetalRobo.metallib");Driver d(argv[1]);auto t=d.telemetry();
    d.command([&](void* cb){MRCompensatedRootTranslationGPU root{};root.reference={0,0,1,0};root.displacement={0,0,std::ldexp(1.0f,-24),0};root.correction={0,0,0,0};const auto position=mrCompensatedTranslationPosition(root,{0,0,0,0});static_cast<MRArticulatedBodyPoseGPU*>(d.bodies.contents)->position=position.high;*static_cast<mr_float4*>(d.low.contents)=position.low;MetalNumanXHumanMatterPass p{};p.phase=MetalNumanXHumanMatterPhase::beginStep;p.commandBuffer=cb;p.environmentCount=1;p.physicsSubstepCount=1;p.bodyPoses=(__bridge void*)d.bodies;p.bodyPositionLow=(__bridge void*)d.low;p.pointJacobians=(__bridge void*)d.jacobian;p.v=(__bridge void*)d.velocity;p.bodyPoseStride=1;p.bodyCount=1;p.qCoordinateCount=7;p.dofCount=6;p.vStride=6;p.pointWorldStride=4;p.pointJacobianStride=72;p.transactionFingerprint=99;p.linearizationEpoch=19;p.slotGeneration=1;return t.encodeInitial(p,d.error);});
    d.measure(t,1,std::ldexp(1.0f,-50));auto first=d.fence(1);auto release=d.release(t,1,true);need(t.terminal(release,&first,d.error),d.error.c_str());
    const auto pending=t.snapshot();need(pending.reduction[0].acceptedRootCount==0,"pending sample prematurely counted");
    d.flush(t);const auto accepted=t.snapshot();const auto& a=accepted.reduction[0];
    need(a.status==0&&a.acceptedRootCount==1&&a.metricSampleCount==1&&a.endNanoseconds==125000,"accepted nonzero epoch reduction failed");
    need(a.postureViolationCount==0&&a.speedErrorSampleCount==1&&std::abs(double(a.extremaHigh.w)+a.extremaLow.w-.0625)<1e-12,"threshold/SSE measurement failed");
    need(a.auditCoveredRootCount==0&&a.auditCoveredAttemptCount==0&&a.initialPostureValid==1&&a.initialSettled==1,"reset posture/settled measurement missing");
    d.flush(t);auto repeated=t.snapshot();need(std::memcmp(&a,&repeated.reduction[0],sizeof(a))==0,"duplicate flush changed metrics");
    need(t.restore(pending,d.error),d.error.c_str());d.flush(t);auto replay=t.snapshot();need(std::memcmp(&a,&replay.reduction[0],sizeof(a))==0,"pending checkpoint replay changed metrics");
    // Current root rejects before postDynamics: beginStep consumes pending,
    // then terminal records a rejection with no matching candidate sample.
    need(t.restore(pending,d.error),d.error.c_str());d.flush(t);auto rejected=d.release(t,2,false);need(t.terminal(rejected,nullptr,d.error),d.error.c_str());d.flush(t);
    const auto afterReject=t.snapshot();need(afterReject.reduction[0].status==0&&afterReject.reduction[0].acceptedRootCount==1&&afterReject.reduction[0].rejectedAttemptCount==1&&afterReject.reduction[0].metricSampleCount==1,"early rejection lost accepted sample or counted stale metrics");
    need(t.restore(afterReject,d.error),"checkpoint after early rejected attempt was denied");d.flush(t);
    d.measure(t,3,std::ldexp(1.0f,-50),2);auto retryFence=d.fence(3);auto retryRelease=d.release(t,3,true,2);
    need(t.terminal(retryRelease,&retryFence,d.error),d.error.c_str());d.flush(t);const auto retry=t.snapshot();
    need(retry.reduction[0].status==0&&retry.reduction[0].acceptedRootCount==2&&retry.reduction[0].rejectedAttemptCount==1&&retry.reduction[0].endNanoseconds==150000,"rejection retry clock/count drift");
    auto wrongEpoch=d.telemetry(200000);need(!wrongEpoch.restore(pending,d.error),"checkpoint from mismatched reset epoch admitted");
    auto below=d.telemetry();d.measure(below,1,-std::ldexp(1.0f,-50));auto belowRelease=d.release(below,1,true);need(below.terminal(belowRelease,&first,d.error),d.error.c_str());d.flush(below);need(below.snapshot().reduction[0].postureViolationCount==1,"negative low threshold side lost");
    auto bad=d.telemetry();d.measure(bad,1,std::ldexp(1.0f,-50));auto corrupted=first;corrupted.brainShadowStateFingerprint=0;corrupted.fenceFingerprint=hashFence(corrupted);auto badRelease=d.release(bad,1,true);badRelease.jointFenceFingerprint=corrupted.fenceFingerprint;need(bad.terminal(badRelease,&corrupted,d.error),d.error.c_str());d.flush(bad);need(bad.snapshot().reduction[0].status!=0&&bad.snapshot().reduction[0].acceptedRootCount==0,"invalid root fence counted");
    std::printf("human_behavior_telemetry=pass controls=10 pending_publication=denied committed=once early_reject=preserved retry=pass checkpoint_replay=bitwise double_flush=bitwise nonzero_epoch=pass mismatched_reset=denied paired_threshold=pass audit_coverage=unknown physical_steps=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"human_behavior_telemetry=fail %s\n",e.what());return 1;}}}
