#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_stand_gpu.h"
#include <array>
#include <vector>
#include <cstring>
#include <iostream>
#include <stdexcept>
using Phase=metalrobo::MetalNumanXTransactionPhase;
using Pass=metalrobo::MetalNumanXTransactionPass;
constexpr unsigned kStepCount=1;constexpr float kTimestepSeconds=0.0001f;
constexpr std::uint64_t kProgramFingerprint=0x4e48524f52443215ull;
void require(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
struct TransactionAudit{
 bool reject=false;bool finalWriteAllowed=false;unsigned finalCount=0;std::string error;
 std::array<id<MTLBuffer>,4> raw{},initial{},candidate{};
};
bool encodeTransaction(void* context,const Pass& p) noexcept{
 auto& a=*static_cast<TransactionAudit*>(context);
 @try{
 auto cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
 std::array<void*,4> ptr={p.q,p.v,p.mujocoStates,p.rootTranslation};
 auto copy=[&](std::array<id<MTLBuffer>,4>& dst){
  auto e=[cb blitCommandEncoder];if(!e)return false;
  for(unsigned i=0;i<4;++i){a.raw[i]=(__bridge id<MTLBuffer>)ptr[i];if(!a.raw[i])return false;
   dst[i]=[cb.device newBufferWithLength:a.raw[i].length options:MTLResourceStorageModeShared];
   if(!dst[i])return false;[e copyFromBuffer:a.raw[i] sourceOffset:0 toBuffer:dst[i] destinationOffset:0 size:a.raw[i].length];}
  [e endEncoding];return true;
 };
 if(p.phase==Phase::beginStep)return copy(a.initial);
 if(p.phase==Phase::postDynamics)return copy(a.candidate);
 if(p.phase==Phase::acceptedPublication){++a.finalCount;a.finalWriteAllowed=(p.accessFlags & metalrobo::MetalNumanXTransactionWriteStandFailure)!=0u;if(a.reject){auto e=[cb blitCommandEncoder];if(!e)return false;
  // Inject only after physical integration and the regular validation phases.
  const std::uint32_t code=MR_NUMI_HUMAN_STAND_CONTACT_FAILED; auto failure=[cb.device newBufferWithBytes:&code length:sizeof(code) options:MTLResourceStorageModeShared]; if(!failure)return false;
  [e copyFromBuffer:failure sourceOffset:0 toBuffer:(__bridge id<MTLBuffer>)p.standStatuses destinationOffset:offsetof(MRNumiHumanStandStatusGPU,code) size:sizeof(code)];[e endEncoding];}}
 return true;
 }@catch(NSException* e){a.error=e.reason.UTF8String;return false;}
}
void abortTransaction(void*,void*) noexcept{}
std::vector<MRArticulatedPointImpulseGPU> makeBodyProbes() {
    constexpr std::array<mr_float4, 4u> points{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    std::vector<MRArticulatedPointImpulseGPU> result;
    result.reserve(points.size());
    for (const mr_float4 point : points) {
        MRArticulatedPointImpulseGPU query{};
        query.bodyIndex = 1u;
        query.localPoint = point;
        result.push_back(query);
    }
    return result;
}

struct SyntheticHumanFixture {
    metalrobo::EngineModel model = metalrobo::makeFreeSphereEngineModel();
    std::vector<float> q = model.defaultQ;
    std::vector<float> v = model.defaultV;
    std::vector<MRArticulatedPointImpulseGPU> points = makeBodyProbes();
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleRouteNodeGPU> routeNodes;

    SyntheticHumanFixture() {
        v[0]=0.1f;
        MRMujocoMuscleSiteGPU origin{};
        origin.bodyIndex = 1u;
        origin.localPoint = {-0.05f, 0.0f, 0.0f, 0.0f};
        sites.push_back(origin);
        MRMujocoMuscleSiteGPU insertion{};
        insertion.bodyIndex = 1u;
        insertion.localPoint = {0.05f, 0.0f, 0.0f, 0.0f};
        sites.push_back(insertion);

        MRMujocoMuscleRouteNodeGPU first{};
        first.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        first.targetIndex = 0u;
        first.sideSiteIndex = MR_INVALID_INDEX;
        routeNodes.push_back(first);
        MRMujocoMuscleRouteNodeGPU second{};
        second.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        second.targetIndex = 1u;
        second.sideSiteIndex = MR_INVALID_INDEX;
        routeNodes.push_back(second);

        MRMujocoMuscleGPU muscle{};
        muscle.route = {0u, 2u, 0u, 0u};
        muscle.lengthRangeAndAcceleration = {0.05f, 0.15f, 1.0f, 0.0f};
        muscle.controlRange = {0.0f, 1.0f, 0.0f, 0.0f};
        muscles.push_back(muscle);

        MRMujocoMuscleStateGPU state{};
        state.excitationAndActivation = {0.6f, 0.0f, 0.0f, 0.0f};
        states.push_back(state);
    }

    metalrobo::MetalArticulatedOperatorInput input(
        TransactionAudit& audit
    ) const {
        metalrobo::MetalArticulatedOperatorInput result{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = points.size(),
            .q = q,
            .v = v,
            .points = points,
            .mujoco = {
                .muscles = muscles,
                .states = states,
                .sites = sites,
                .wraps = {},
                .routeNodes = routeNodes,
                .bodyJacobianPointOffset = 0u,
            },
            .stand = {
                .v = v,
                .contacts = {},
                .jointEqualities = {},
                .tendonBindings = {},
                .tendonEnvelopes = {},
                .tendonLoadProgram = {},
                .numanXTransactionProgram = {
                    .context = &audit,
                    .encode = &encodeTransaction,
                    .abort = &abortTransaction,
                    .fingerprint = kProgramFingerprint,
                    .deferAcceptedPublication = true,
                },
                .stepCount = kStepCount,
                .contactIterationCount = 1u,
                .enableContact = false,
                .enableRootAssistance = false,
            },
        };
        return result;
    }
};

std::vector<unsigned char> bytes(const std::array<id<MTLBuffer>,4>& b){std::vector<unsigned char> r;for(auto v:b){require(v!=nil,"snapshot missing");auto p=(const unsigned char*)v.contents;r.insert(r.end(),p,p+v.length);}return r;}
int main(){@autoreleasepool{try{
 SyntheticHumanFixture f;
 auto make=[](){metalrobo::MetalArticulatedOperatorConfig c;c.pointJacobiansOnly=true;c.mujocoActivationTimestepSeconds=kTimestepSeconds;c.metallibPath=METALROBO_DEFAULT_METALLIB;return metalrobo::MetalArticulatedOperatorContext(c);};
 auto context=make();TransactionAudit rejected;rejected.reject=true;metalrobo::MetalArticulatedOperatorResult rr;
 auto rd=context.run(f.model,f.input(rejected),rr);require(rd.dispatched&&!rd.published&&!rd.succeeded(),"late rejection was not executed: "+rd.message);
 require(rejected.finalCount==1&&rejected.error.empty(),"final gate callback missing");
 auto initial=bytes(rejected.initial),candidate=bytes(rejected.candidate),restored=bytes(rejected.raw);
 require(candidate!=initial,"physical candidate did not move");require(restored==initial,"late rejection leaked body state");require(rejected.finalWriteAllowed,"final phase lacks status downgrade permission");
 TransactionAudit retry;metalrobo::MetalArticulatedOperatorResult retryResult;auto tr=context.run(f.model,f.input(retry),retryResult);
 require(tr.succeeded()&&tr.published,"retry failed: "+tr.message);
 auto cleanContext=make();TransactionAudit clean;metalrobo::MetalArticulatedOperatorResult cleanResult;auto cr=cleanContext.run(f.model,f.input(clean),cleanResult);
 require(cr.succeeded()&&cr.published,"clean control failed: "+cr.message);
 require(bytes(retry.raw)==bytes(clean.raw),"retry differs from clean control");
 require(bytes(clean.raw)!=bytes(clean.initial),"accepted control did not move");
 std::cout<<"late_gate_order=pass rollback_exact=1 candidate_changed=1 retry_matches_clean=1 states=q,v,myosim,compensated_root dt_s="<<kTimestepSeconds<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
