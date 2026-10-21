#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_cavity_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <sstream>

using namespace numi::matter;
namespace fixture=vascular_cavity_fixture;
namespace {
float& cavityComponent(nm_float4& value, unsigned axis) {
    return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
}
float cavityComponent(const nm_float4& value, unsigned axis) {
    return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
}
void need(bool okay,const std::string& why){if(!okay)throw std::runtime_error(why);}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
template<class T> void paired(const std::vector<T>& a,const char* name){need(a.size()%2==0,std::string(name)+" pair arity");const auto n=a.size()/2;need(!n||std::memcmp(a.data(),a.data()+n,n*sizeof(T))==0,std::string(name)+" paired environments differ");}
template<class T> void unchangedEnvironment(const std::vector<T>& a,const std::vector<T>& b,unsigned env,const char* name){
    need(a.size()==b.size()&&a.size()%2==0,std::string(name)+" snapshot arity");const auto n=a.size()/2;
    need(!n||std::memcmp(a.data()+env*n,b.data()+env*n,n*sizeof(T))==0,std::string(name)+" rejected/reset state differs");
}
bool samePhysical(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){
    return same(a.femNodes,b.femNodes)&&same(a.femFields,b.femFields)&&same(a.vascularState,b.vascularState)&&same(a.vascularClock,b.vascularClock);
}
bool sameBloodMomentumState(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){
    return same(a.femNodes,b.femNodes)&&same(a.vascularState,b.vascularState)&&same(a.vascularClock,b.vascularClock);
}
struct ScopedEnvironment {
    std::string key;
    bool wasSet=false;
    std::string prior;
    ScopedEnvironment(const char* name,const char* value):key(name){
        if(const char* current=std::getenv(name)){wasSet=true;prior=current;}
        if(value){if(setenv(name,value,1)!=0)throw std::runtime_error("setenv failed for "+key);}
        else if(unsetenv(name)!=0)throw std::runtime_error("unsetenv failed for "+key);
    }
    ~ScopedEnvironment(){if(wasSet)setenv(key.c_str(),prior.c_str(),1);else unsetenv(key.c_str());}
};
struct ScopedVascularPolicy {
    ScopedEnvironment block;
    ScopedEnvironment predictor;
    ScopedVascularPolicy(const char* blockValue,const char* predictorValue):
        block("NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER",blockValue),
        predictor("NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR",predictorValue){}
};
std::vector<fixture::Vec> positions(const RuntimeStateSnapshot& s,unsigned env,unsigned nodes){
    need(s.femNodes.size()==2*nodes,"accepted FEM node coverage");std::vector<fixture::Vec> x(nodes);
    for(unsigned i=0;i<nodes;++i){const auto& p=s.femNodes[env*nodes+i].positionAndMass;x[i]={p.x,p.y,p.z};}
    return x;
}
struct Run {
    CompiledWorld world;Runtime runtime;
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLBuffer> statuses;
    explicit Run(const WorldSource& source){
        auto compiled=compileWorld(source,{.maximumRateExponent=0});std::string message;
        for(const auto& d:compiled.diagnostics)if(d.severity==Diagnostic::Severity::error)message+=d.message+"; ";
        need(compiled.succeeded(),"hollow-cavity compile: "+message);
        const auto path=std::filesystem::temp_directory_path()/(std::string("numi-hollow-cavity-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        need(writePackage(compiled,path,&message),"package write: "+message);
        const bool loaded=readPackage(path,world,nullptr,&message);std::error_code removal;std::filesystem::remove(path,removal);
        need(loaded&&!removal&&world.fingerprint==compiled.world.fingerprint,"package roundtrip: "+message);
        double mass=0;for(const auto& node:world.fem.nodes)mass+=node.positionAndMass.w;
        double expectedMass=.026;for(const auto& tissue:world.vascular.tissues)expectedMass+=tissue.physical.z;
        need(std::abs(mass-expectedMass)<1e-8,"registered blood mass was not partitioned into FEM nodal inertia");
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue];statuses=[device newBufferWithLength:2*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        RuntimeConfiguration config;config.metallib=NUMI_MATTER_METALLIB;config.environmentCount=2;
        config.captureEvents=false;config.captureDiagnostics=true;config.adaptiveTransfer=false;
        const auto result=runtime.initialize(world,config);need(result.encoded,result.message);
        std::cout<<"cavity_device="<<[device name].UTF8String<<" abi="<<NM_MATTER_ABI_VERSION<<" world_fingerprint="<<world.fingerprint<<'\n';
    }
    RuntimeStateSnapshot state(){auto out=runtime.snapshot();need(out.available,out.message);return out;}
    void restore(const RuntimeStateSnapshot& snapshot){const auto result=runtime.restore(snapshot);need(result.encoded,"restore: "+result.message);}
    void step(unsigned index,int reject=-1,int reset=-1){
        auto* incoming=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        for(unsigned env=0;env<2;++env){incoming[env]={};incoming[env].environment=env;}
        auto command=[queue commandBuffer];need(command!=nil,"command buffer unavailable");
        EncodeRequest request;request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=index;request.physicsSubsteps=1;request.timestepSeconds=runtime.timestepSeconds();request.runAdaptiveTransfer=false;
        id<MTLBuffer> resets=nil,failure=nil;
        if(reset>=0){std::vector<std::uint32_t> mask((index+1u)*2u);mask[index*2u+unsigned(reset)]=1;
            resets=[device newBufferWithBytes:mask.data() length:mask.size()*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
            request.resetMasks=(__bridge void*)resets;request.resetMaskStepStride=2;}
        request.phase=EncodePhase::preDynamics;auto result=runtime.encode(request);need(result.encoded,"preDynamics: "+result.message);
        if(reject>=0){MRMetalWorldStatusGPU denied{};denied.code=MR_STEP_DID_NOT_CONVERGE;denied.environment=unsigned(reject);
            failure=[device newBufferWithBytes:&denied length:sizeof(denied) options:MTLResourceStorageModeShared];
            auto blit=[command blitCommandEncoder];[blit copyFromBuffer:failure sourceOffset:0 toBuffer:statuses destinationOffset:unsigned(reject)*sizeof(denied) size:sizeof(denied)];[blit endEncoding];}
        request.phase=EncodePhase::postCommit;result=runtime.encode(request);need(result.encoded,"postCommit: "+result.message);
        [command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"Metal command failed");
        const auto after=state();need(after.statuses.size()==2,"environment status coverage");
        for(unsigned env=0;env<2;++env)if(int(env)!=reject &&
                after.statuses[env].code!=NM_STATUS_SUCCESS) {
            const auto& status=after.statuses[env];
            std::cerr << std::setprecision(17) << "cavity_failure step=" << index
                      << " env=" << env << " object=" << status.objectIndex
                      << " index=" << status.failingIndex << " diagnostics=["
                      << status.diagnostics.x << "," << status.diagnostics.y << ","
                      << status.diagnostics.z << "," << status.diagnostics.w << "]\n";
        }
        for(unsigned env=0;env<2;++env)if(int(env)!=reject)need(after.statuses[env].code==NM_STATUS_SUCCESS,
            "cavity step="+std::to_string(index)+" env="+std::to_string(env)+" status="+std::to_string(after.statuses[env].code)+
            " residual="+std::to_string(after.statuses[env].diagnostics.z));
        if(reject>=0)need(after.statuses[unsigned(reject)].code!=NM_STATUS_SUCCESS,"injected transaction rejection was lost");
        (void)failure;(void)resets;
    }
    double physical(const RuntimeStateSnapshot& s,unsigned row,unsigned env=0)const{
        need(row<world.vascular.unknowns.size()&&s.vascularState.size()==2*world.vascular.unknowns.size(),"vascular unknown arity");
        const double value=double(s.vascularState[env*world.vascular.unknowns.size()+row].x)*world.vascular.unknowns[row].initialAndScaling.y;
        need(std::isfinite(value),"nonfinite physical vascular coordinate");return value;
    }
};
RuntimeStateSnapshot noOwnerTangentEvolution(const char* policy) {
    ScopedEnvironment tangentPolicy("NM_VASCULAR_BLOOD_MASS_TANGENT", policy);
    Run run(fixture::world());
    need(run.world.vascular.tissues.empty() &&
         run.world.vascular.tissueBindings.empty(),
         "no-owner tangent fixture unexpectedly has tissue bindings");
    const auto initial = run.state();
    need(initial.available && initial.message.empty(),
         "no-owner tangent initial snapshot unavailable");
    run.step(0u);
    return run.state();
}
void bloodMassTangentNoOwnerAdmission() {
    const auto legacy = noOwnerTangentEvolution("0");
    const auto requestedNoOp = noOwnerTangentEvolution("1");
    need(legacy.available && requestedNoOp.available &&
         samePhysical(legacy, requestedNoOp) &&
         same(legacy.statuses, requestedNoOp.statuses),
         "owner-free blood-mass tangent request changed physical state or status");
    std::cout << "blood_mass_tangent_no_owner=pass requested=1 effective=0 "
              << "eligible_bindings=0 no_op_logged=1 state_match=bitwise\n";
}
struct Metrics {double geometry=0,blood=0,tracer=0,balance=0,flow=0,work=0,motion=0,flowMaximum=0;};
template<class T> id<MTLBuffer> gpuBuffer(id<MTLDevice> device,const std::vector<T>& values){
    need(!values.empty(),"empty direct operator input");
    return [device newBufferWithBytes:values.data() length:values.size()*sizeof(T) options:MTLResourceStorageModeShared];
}
template<class T> id<MTLBuffer> gpuBufferOrDummy(id<MTLDevice> device,const std::vector<T>& values){
    if(!values.empty())return gpuBuffer(device,values);
    return [device newBufferWithLength:sizeof(T) options:MTLResourceStorageModeShared];
}
id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> library,NSString* name){
    NSError* error=nil;auto function=[library newFunctionWithName:name];need(function!=nil,"production cavity kernel missing");
    auto result=[device newComputePipelineStateWithFunction:function error:&error];need(result!=nil,"cavity pipeline creation failed");return result;
}
void completed(id<MTLCommandBuffer> command){[command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"direct cavity operator failed");}

void frozenStaticLoad(Run& run,const FrozenFEMStaticLoadSource& sealedSource){
    const RuntimeStateSnapshot& source=sealedSource.state;
    const auto before=run.state();
    need(before.available&&before.message.empty(),
         "ordinary pre-evaluation Matter snapshot unavailable");
    const std::uint32_t nodes=run.world.dispatch.femNodeCount;
    const std::uint32_t environments=run.world.dispatch.environmentCount;
    const std::size_t total=std::size_t(nodes)*environments;
    need(environments==2&&source.vascularClock.size()==environments,
         "frozen static fixture requires two explicit pressure phases");
    std::vector<nm_float4> sentinel(total,nm_float4{91.f,92.f,93.f,0.f});
    std::vector<NMMatterStatusGPU> statusSeed(environments);
    id<MTLBuffer> result=[run.device newBufferWithBytes:sentinel.data()
        length:sentinel.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> reaction=[run.device newBufferWithBytes:sentinel.data()
        length:sentinel.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> statuses=[run.device newBufferWithBytes:statusSeed.data()
        length:statusSeed.size()*sizeof(NMMatterStatusGPU) options:MTLResourceStorageModeShared];
    need(result&&reaction&&statuses,"frozen static result buffers unavailable");
    const auto command=[run.queue commandBuffer];
    FrozenFEMStaticLoadRequest request;
    request.commandBuffer=(__bridge void*)command;
    request.source=&sealedSource;
    request.declaredPressurePhase=std::span<const NMVascularClockGPU>(source.vascularClock);
    request.nodalResultants=(__bridge void*)result;
    request.anchorReactions=(__bridge void*)reaction;
    request.statuses=(__bridge void*)statuses;
    request.gravityScale=1.f;
    FrozenFEMStaticLoadCertificate certificate;
    const auto encoded=run.runtime.encodeFrozenFEMStaticLoad(request,certificate);
    need(encoded.encoded,"frozen static load encoding: "+encoded.message);
    need(certificate.nodalVelocityZeroed&&certificate.pressurePhaseMatchesSourceClock&&
         certificate.environmentCount==environments&&certificate.femNodeCount==nodes&&
         certificate.gravityScale==1.f&&!certificate.frozenQSHA256.empty()&&
         !certificate.frozenStateSHA256.empty(),
         "frozen static load certificate is incomplete");
    completed(command);
    const auto* deviceStatus=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    const auto* resultValues=static_cast<const nm_float4*>(result.contents);
    const auto* reactionValues=static_cast<const nm_float4*>(reaction.contents);
    std::size_t nonzeroResultants=0u,nonzeroReactions=0u;
    for(std::size_t i=0;i<total;++i){
        const auto finite=[](const nm_float4& v){
            return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z)&&
                   std::isfinite(v.w);
        };
        need(finite(resultValues[i])&&finite(reactionValues[i]),
             "frozen static load returned a nonfinite nodal value");
        const double r=double(resultValues[i].x)*resultValues[i].x+
            double(resultValues[i].y)*resultValues[i].y+
            double(resultValues[i].z)*resultValues[i].z;
        const double a=double(reactionValues[i].x)*reactionValues[i].x+
            double(reactionValues[i].y)*reactionValues[i].y+
            double(reactionValues[i].z)*reactionValues[i].z;
        nonzeroResultants+=r>0.0;
        nonzeroReactions+=a>0.0;
    }
    for(std::uint32_t env=0;env<environments;++env)
        need(deviceStatus[env].code==NM_STATUS_SUCCESS,
             "frozen static load GPU status="+std::to_string(deviceStatus[env].code));
    need(run.world.dispatch.deformableContactCapacity==0u&&
         run.world.dispatch.contactPairCount==0u,
         "frozen static coupon must compile with explicit zero contact capacity");
    need(nonzeroResultants>0u&&nonzeroReactions>0u,
         "frozen gravity/cavity/elastic fixture produced no static resultant or anchor reaction");

    // Global force closure: internal elastic forces and a closed-cavity
    // pressure field must cancel under summation, leaving only nodal gravity.
    for(std::uint32_t env=0;env<environments;++env){
        double sum[3]{},absoluteSum=0.0,totalMass=0.0;
        for(std::uint32_t node=0;node<nodes;++node){
            const std::size_t index=std::size_t(env)*nodes+node;
            const auto& f=resultValues[index];
            sum[0]+=f.x;sum[1]+=f.y;sum[2]+=f.z;
            absoluteSum+=std::abs(double(f.x))+std::abs(double(f.y))+std::abs(double(f.z));
            totalMass+=run.world.fem.nodes[node].positionAndMass.w;
        }
        const double gravity[3]={run.world.dispatch.gravityAndTimestep.x,
            run.world.dispatch.gravityAndTimestep.y,run.world.dispatch.gravityAndTimestep.z};
        const double weight[3]={totalMass*gravity[0],totalMass*gravity[1],totalMass*gravity[2]};
        const double error=std::sqrt((sum[0]-weight[0])*(sum[0]-weight[0])+
            (sum[1]-weight[1])*(sum[1]-weight[1])+
            (sum[2]-weight[2])*(sum[2]-weight[2]));
        const double scale=std::max({1.0,absoluteSum,
            std::sqrt(weight[0]*weight[0]+weight[1]*weight[1]+weight[2]*weight[2])});
        need(error<=2.0e-5*scale,
            "frozen static internal/cavity force closure failed env="+std::to_string(env)+
            " error="+std::to_string(error)+" scale="+std::to_string(scale));
    }

    // Re-evaluate with an independently uploaded nonuniform external field.
    // The output must change by that field exactly once; accepted state remains
    // a frozen input and the force buffer is never modified by the API.
    std::vector<nm_float4> external(total),externalSentinel(total,nm_float4{81.f,82.f,83.f,0.f});
    for(std::size_t i=0;i<total;++i){
        const float k=float(i%13u);
        external[i]={2.0f+0.125f*k,-1.5f+0.0625f*k,0.75f-0.03125f*k,0.0f};
    }
    id<MTLBuffer> externalBuffer=[run.device newBufferWithBytes:external.data()
        length:external.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> externalResult=[run.device newBufferWithBytes:externalSentinel.data()
        length:externalSentinel.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> externalReaction=[run.device newBufferWithBytes:externalSentinel.data()
        length:externalSentinel.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> externalStatuses=[run.device newBufferWithBytes:statusSeed.data()
        length:statusSeed.size()*sizeof(NMMatterStatusGPU) options:MTLResourceStorageModeShared];
    const auto externalCommand=[run.queue commandBuffer];
    request.commandBuffer=(__bridge void*)externalCommand;
    request.externalForces=(__bridge void*)externalBuffer;
    request.nodalResultants=(__bridge void*)externalResult;
    request.anchorReactions=(__bridge void*)externalReaction;
    request.statuses=(__bridge void*)externalStatuses;
    FrozenFEMStaticLoadCertificate externalCertificate;
    const auto externalEncoded=run.runtime.encodeFrozenFEMStaticLoad(request,externalCertificate);
    need(externalEncoded.encoded,"frozen external-load encoding: "+externalEncoded.message);
    completed(externalCommand);
    const auto* externalOutput=static_cast<const nm_float4*>(externalResult.contents);
    const auto* externalStatus=static_cast<const NMMatterStatusGPU*>(externalStatuses.contents);
    double externalSum[2][3]{};
    for(std::uint32_t env=0;env<environments;++env)
        need(externalStatus[env].code==NM_STATUS_SUCCESS,
            "frozen external-load GPU status="+std::to_string(externalStatus[env].code));
    for(std::size_t i=0;i<total;++i){
        const float* base=&resultValues[i].x;
        const float* withExternal=&externalOutput[i].x;
        const float* applied=&external[i].x;
        for(unsigned axis=0;axis<3;++axis){
            const double difference=double(withExternal[axis])-double(base[axis]);
            const double rounding=64.0*std::numeric_limits<float>::epsilon()*
                (1.0+std::abs(double(base[axis]))+std::abs(double(withExternal[axis]))+
                 std::abs(double(applied[axis])));
            need(std::abs(difference-double(applied[axis]))<=rounding,
                "frozen external force was omitted or applied more than once");
            externalSum[i/nodes][axis]+=applied[axis];
        }
    }
    for(std::uint32_t env=0;env<environments;++env){
        double sum[3]{};
        for(std::uint32_t node=0;node<nodes;++node){
            const auto& f=externalOutput[std::size_t(env)*nodes+node];
            sum[0]+=f.x;sum[1]+=f.y;sum[2]+=f.z;
        }
        const double fieldSum[3]={externalSum[env][0],
            externalSum[env][1],externalSum[env][2]};
        const double mass=std::accumulate(run.world.fem.nodes.begin(),
            run.world.fem.nodes.end(),0.0,[](double total,const NMFEMNodeStateGPU& node){
                return total+node.positionAndMass.w;
            });
        const double gravity[3]={run.world.dispatch.gravityAndTimestep.x,
            run.world.dispatch.gravityAndTimestep.y,run.world.dispatch.gravityAndTimestep.z};
        const double expected[3]={mass*gravity[0]+fieldSum[0],
            mass*gravity[1]+fieldSum[1],mass*gravity[2]+fieldSum[2]};
        const double error=std::sqrt((sum[0]-expected[0])*(sum[0]-expected[0])+
            (sum[1]-expected[1])*(sum[1]-expected[1])+
            (sum[2]-expected[2])*(sum[2]-expected[2]));
        const double scale=std::max({1.0,std::abs(expected[0])+std::abs(expected[1])+std::abs(expected[2])});
        need(error<=3.0e-5*scale,
            "frozen external field global resultant is not included exactly once");
    }
    request.externalForces=nullptr;
    const auto after=run.state();
    need(same(before.particles,after.particles)&&
         same(before.femNodes,after.femNodes)&&same(before.femFields,after.femFields)&&
         same(before.femMaterialState,after.femMaterialState)&&
         same(before.particleMaterialState,after.particleMaterialState)&&
         same(before.topologyStates,after.topologyStates)&&
         same(before.statuses,after.statuses)&&
         same(before.solverCertificates,after.solverCertificates)&&
         same(before.rigidGeneralizedCandidate,after.rigidGeneralizedCandidate)&&
         same(before.mpmActiveNodeIndices,after.mpmActiveNodeIndices)&&
         same(before.mpmNodeToActive,after.mpmNodeToActive)&&
         same(before.mpmActiveNodeCounts,after.mpmActiveNodeCounts)&&
         same(before.identification,after.identification)&&
         before.identificationGeneration==after.identificationGeneration&&
         before.identificationCheckpoint==after.identificationCheckpoint&&
         before.identificationAdvanced==after.identificationAdvanced&&
         before.sutureProxyEdges==after.sutureProxyEdges&&
         before.sutureProxyBindingRevision==after.sutureProxyBindingRevision&&
         before.coupledTimestepMultiplier==after.coupledTimestepMultiplier&&
         before.coupledTimestepDivisor==after.coupledTimestepDivisor&&
         before.fgmresIterationBudgetOverride==after.fgmresIterationBudgetOverride&&
         before.newtonIterationBudgetOverride==after.newtonIterationBudgetOverride&&
         before.allocationGeneration==after.allocationGeneration&&
         before.learnedWeightRevision==after.learnedWeightRevision&&
         before.materialStateStride==after.materialStateStride&&
         same(before.vascularState,after.vascularState)&&
         same(before.vascularClock,after.vascularClock)&&
         same(before.femTopologyNodes,after.femTopologyNodes)&&
         same(before.femTopologyTetrahedra,after.femTopologyTetrahedra)&&
         same(before.cohesiveFaces,after.cohesiveFaces)&&
         same(before.punctureChannels,after.punctureChannels)&&
         same(before.schedulers,after.schedulers)&&same(before.adaptive,after.adaptive)&&
         same(before.environmentParameters,after.environmentParameters)&&
         same(before.learnedWeights,after.learnedWeights)&&
         same(before.contactSamples,after.contactSamples)&&
         same(before.contactHistories,after.contactHistories)&&
         same(before.humanSupportHistories,after.humanSupportHistories)&&
         same(before.humanSupportConsequences,after.humanSupportConsequences)&&
         same(before.deformableContactHistories,after.deformableContactHistories)&&
         same(before.deformableContactFailures,after.deformableContactFailures)&&
         same(before.rigidStates,after.rigidStates)&&
         same(before.reactions,after.reactions)&&
         before.controlStep==after.controlStep&&before.physicsSubstep==after.physicsSubstep,
         "frozen static load evaluation mutated or advanced Matter state");

    // The explicit source seal rejects even a copied snapshot whose arrays
    // were altered after capture; validation occurs before any GPU encoding.
    std::vector<nm_float4> untouched(total,nm_float4{71.f,72.f,73.f,0.f});
    RuntimeStateSnapshot mutated=source;
    mutated.femNodes[0].positionAndMass.x+=0.001f;
    auto mutatedSealedSource=sealedSource;
    mutatedSealedSource.state=std::move(mutated);
    id<MTLBuffer> mutatedResult=[run.device newBufferWithBytes:untouched.data()
        length:untouched.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    request.commandBuffer=(__bridge void*)[run.queue commandBuffer];
    request.source=&mutatedSealedSource;
    request.declaredPressurePhase=std::span<const NMVascularClockGPU>(source.vascularClock);
    request.nodalResultants=(__bridge void*)mutatedResult;
    FrozenFEMStaticLoadCertificate mutatedCertificate;
    const auto mutationRejected=run.runtime.encodeFrozenFEMStaticLoad(
        request,mutatedCertificate);
    need(!mutationRejected.encoded&&
         std::string(mutationRejected.message).find("content proof")!=std::string::npos,
         "mutated frozen source snapshot was not rejected");
    const auto* mutatedUntouched=static_cast<const nm_float4*>(mutatedResult.contents);
    need(std::memcmp(untouched.data(),mutatedUntouched,
             untouched.size()*sizeof(nm_float4))==0,
         "mutated-source rejection modified output storage");
    request.source=&sealedSource;

    // Invalid pressure phase is rejected before encoding and leaves caller
    // output storage untouched.
    auto wrongPhase=source.vascularClock;
    wrongPhase[0].low+=1u;
    id<MTLBuffer> rejectedResult=[run.device newBufferWithBytes:untouched.data()
        length:untouched.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
    request.commandBuffer=(__bridge void*)[run.queue commandBuffer];
    request.declaredPressurePhase=std::span<const NMVascularClockGPU>(wrongPhase);
    request.nodalResultants=(__bridge void*)rejectedResult;
    FrozenFEMStaticLoadCertificate rejectedCertificate;
    const auto rejected=run.runtime.encodeFrozenFEMStaticLoad(request,rejectedCertificate);
    need(!rejected.encoded&&std::string(rejected.message).find("declared cardiac phase must exactly match the frozen pressure snapshot clock")!=std::string::npos,
         "mismatched frozen pressure phase was not rejected");
    const auto* untouchedAfter=static_cast<const nm_float4*>(rejectedResult.contents);
    need(std::memcmp(untouched.data(),untouchedAfter,untouched.size()*sizeof(nm_float4))==0,
         "rejected frozen phase modified output storage");
    std::cout<<"frozen_static_load=pass environments="<<environments
        <<" fem_nodes="<<nodes<<" nonzero_resultants="<<nonzeroResultants
        <<" nonzero_anchor_reactions="<<nonzeroReactions
        <<" source_state_unchanged=1 contact_scope=excluded force_closure=pass external_once=pass\n";
}

void productionOperators(Run& run){
    // Independent manufactured states; the production force and tangent
    // kernels are called directly, and never replaced by the FP64 oracle.
    const auto shell=fixture::hollowShell();const auto& w=run.world;const unsigned nodes=w.dispatch.femNodeCount;
    need(nodes==64&&w.dispatch.environmentCount==2,"operator fixture layout differs");
    NSError* error=nil;auto library=[run.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]] error:&error];
    need(library!=nil,"metallib unavailable for production operator checks");
    const auto forcePipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_cavity_forces");
    const auto operatorPipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_cavity_operator");
    NMMicrostepGPU micro{};micro.time={float(run.runtime.timestepSeconds()),float(1./run.runtime.timestepSeconds()),0,0};
    micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;
    NMFGMRESLayoutGPU layout{};layout.vascularUnknownCount=unsigned(w.vascular.unknowns.size());
    layout.vascularBase=4*nodes;layout.unknownCount=layout.vascularBase+2*layout.vascularUnknownCount;
    auto accepted=run.state().femNodes,candidate=accepted;
    std::vector<nm_float4> vascular=run.state().vascularState,direction(layout.unknownCount),borrowed(2*nodes);
    auto compartments=w.vascular.compartments;compartments[0].compliance.w=20;
    for(unsigned env=0;env<2;++env){
        vascular[env*layout.vascularUnknownCount+w.vascular.layout.cavities.z].x=env==0?1.2f:2.4f;
        direction[layout.vascularBase+env*layout.vascularUnknownCount+w.vascular.layout.cavities.z].x=env==0?.2f:-.3f;
        for(unsigned i=0;i<nodes;++i)borrowed[env*nodes+i]={.0003f,-.0002f,.0001f,0};
        for(unsigned i=0;i<shell.innerNodes.size();++i){const unsigned n=env*nodes+shell.innerNodes[i];
            candidate[n].positionAndMass.x+=float(.0001*std::sin(double(i)+.3));
            candidate[n].positionAndMass.y+=float(.0002*std::cos(double(i)+.2));
            candidate[n].positionAndMass.z+=float(.0001*std::sin(2*double(i)+.1));
            direction[n]={float(.08*std::cos(double(i)+.2)),float(.06*std::sin(double(i)+.4)),float(.07*std::cos(2*double(i)+.1)),0};
        }
    }
    const auto unknownBuffer=gpuBuffer(run.device,w.vascular.unknowns),cavityBuffer=gpuBuffer(run.device,w.vascular.cavities),faceBuffer=gpuBuffer(run.device,w.vascular.cavityFaces);
    const auto incidenceBuffer=gpuBuffer(run.device,w.vascular.cavityNodeIncidence),rangeBuffer=gpuBuffer(run.device,w.vascular.cavityNodeRanges);
    const auto bloodTissueBuffer=gpuBufferOrDummy(run.device,w.vascular.tissues),bloodBindingBuffer=gpuBufferOrDummy(run.device,w.vascular.tissueBindings);
    std::vector<float> fixtureElastance(w.dispatch.environmentCount*w.vascular.compartments.size(),1.f);
    for(unsigned env=0;env<w.dispatch.environmentCount;++env) for(unsigned i=0;i<w.vascular.compartments.size();++i)
        if(w.vascular.compartments[i].identity.w==0u||w.vascular.compartments[i].identity.w==3u) fixtureElastance[env*w.vascular.compartments.size()+i]=float(1./w.vascular.compartments[i].compliance.z);
    const auto elastanceBuffer=gpuBufferOrDummy(run.device,fixtureElastance);
    auto acceptedBuffer=gpuBuffer(run.device,accepted);
    NMVascularLayoutGPU equationVascularLayout = w.vascular.layout;
    const auto cavityConnectionBuffer=gpuBufferOrDummy(run.device,w.vascular.connections);
    const auto cavityAcceptedVascularBuffer=gpuBuffer(run.device,run.state().vascularState);
    auto encodeCommon=[&](id<MTLComputeCommandEncoder> encoder,id<MTLBuffer> compartmentBuffer,id<MTLBuffer> candidateBuffer,id<MTLBuffer> vascularBuffer){
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&equationVascularLayout length:sizeof(equationVascularLayout) atIndex:1];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];[encoder setBuffer:incidenceBuffer offset:0 atIndex:7];
        [encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:acceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:vascularBuffer offset:0 atIndex:11];
        [encoder setBuffer:cavityConnectionBuffer offset:0 atIndex:19];[encoder setBuffer:cavityAcceptedVascularBuffer offset:0 atIndex:20];
    };
    auto forces=[&](const std::vector<NMFEMNodeStateGPU>& positions,const std::vector<nm_float4>& states,
                    const std::vector<NMVascularCompartmentGPU>& pressureData,bool hasBorrowed){
        const auto cb=[run.queue commandBuffer];auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:forcePipeline];
        const auto c=gpuBuffer(run.device,positions),v=gpuBuffer(run.device,states),p=gpuBuffer(run.device,pressureData),b=gpuBuffer(run.device,borrowed);
        const auto output=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes,nm_float4{111,222,333,0}));
        const auto status=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));const std::uint32_t include=hasBorrowed?1:0;
        encodeCommon(encoder,p,c,v);[encoder setBuffer:b offset:0 atIndex:12];[encoder setBytes:&include length:sizeof(include) atIndex:13];
        [encoder setBuffer:output offset:0 atIndex:14];[encoder setBuffer:status offset:0 atIndex:15];
        [encoder setBuffer:bloodTissueBuffer offset:0 atIndex:16];[encoder setBuffer:bloodBindingBuffer offset:0 atIndex:17];
        [encoder setBuffer:elastanceBuffer offset:0 atIndex:18];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(cb);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(status.contents)[env].code==NM_STATUS_SUCCESS,"manufactured cavity force state rejected");
        const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);
    };
    auto apply=[&](const std::vector<NMFEMNodeStateGPU>& positions,const std::vector<NMFGMRESStateGPU>& solverStates,
                    nm_float4 initialValue=nm_float4{.0001f,-.0002f,.0003f,0}){
        const auto cb=[run.queue commandBuffer];auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:operatorPipeline];
        const auto c=gpuBuffer(run.device,positions),v=gpuBuffer(run.device,vascular),p=gpuBuffer(run.device,compartments),d=gpuBuffer(run.device,direction);
        const auto initial=std::vector<nm_float4>(2*nodes,initialValue);
        const auto output=gpuBuffer(run.device,initial),solver=gpuBuffer(run.device,solverStates),status=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));
        encodeCommon(encoder,p,c,v);[encoder setBytes:&layout length:sizeof(layout) atIndex:30];[encoder setBuffer:d offset:0 atIndex:12];
        [encoder setBuffer:output offset:0 atIndex:13];[encoder setBuffer:solver offset:0 atIndex:14];[encoder setBuffer:status offset:0 atIndex:15];
        [encoder setBuffer:bloodTissueBuffer offset:0 atIndex:16];[encoder setBuffer:bloodBindingBuffer offset:0 atIndex:17];[encoder setBuffer:elastanceBuffer offset:0 atIndex:18];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(cb);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(status.contents)[env].code==NM_STATUS_SUCCESS,"manufactured cavity tangent state rejected");
        const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);
    };
    const auto raw=forces(candidate,vascular,compartments,false),merged=forces(candidate,vascular,compartments,true);
    double forceError=0,forceScale=0,workError=0;
    for(unsigned env=0;env<2;++env){
        std::vector<fixture::Vec> a(nodes),b(nodes);for(unsigned i=0;i<nodes;++i){const auto& p=accepted[env*nodes+i].positionAndMass;const auto& q=candidate[env*nodes+i].positionAndMass;a[i]={p.x,p.y,p.z};b[i]={q.x,q.y,q.z};}
        const auto dg=fixture::discreteGradient(a,b,shell.lumenFaces);
        const unsigned row=w.vascular.layout.cavities.z;
        const double pressure=double(vascular[env*layout.vascularUnknownCount+row].x)*w.vascular.unknowns[row].initialAndScaling.y-compartments[0].compliance.w;
        double work=0;
        for(unsigned i=0;i<nodes;++i){const unsigned global=env*nodes+i;const fixture::Vec f{raw[global].x,raw[global].y,raw[global].z};
            const fixture::Vec external{borrowed[global].x,borrowed[global].y,borrowed[global].z},both{merged[global].x,merged[global].y,merged[global].z};
            for(unsigned j=0;j<3;++j){need(std::isfinite(f[j])&&std::isfinite(both[j]),"nonfinite production force");forceError=std::max(forceError,std::abs(f[j]-pressure*dg[i][j]));forceScale=std::max(forceScale,std::abs(pressure*dg[i][j]));need(std::abs(both[j]-f[j]-external[j])<1e-8,"borrowed external force was dropped or double counted");}
            work+=fixture::dot(f,fixture::subtract(b[i],a[i]));
        }
        const double expected=pressure*(fixture::volume(b,shell.lumenFaces)-fixture::volume(a,shell.lumenFaces));
        workError=std::max(workError,std::abs(work-expected)/std::max(1e-18,std::abs(expected)));
    }
    need(forceError/forceScale<3e-6&&workError<1e-5,"production force does not match independent volume/work oracle");
    auto shiftedCompartment=compartments;shiftedCompartment[0].compliance.w+=1024;
    auto shiftedPressure=vascular;for(unsigned env=0;env<2;++env)shiftedPressure[env*layout.vascularUnknownCount+w.vascular.layout.cavities.z].x+=1024/w.vascular.unknowns[w.vascular.layout.cavities.z].initialAndScaling.y;
    const auto shifted=forces(candidate,shiftedPressure,shiftedCompartment,false);
    for(unsigned i=0;i<raw.size();++i)need(std::max({std::abs(raw[i].x-shifted[i].x),std::abs(raw[i].y-shifted[i].y),std::abs(raw[i].z-shifted[i].z)})<forceScale*3e-6,"wall force uses absolute rather than transmural pressure");
    constexpr double epsilon=1./32.;auto plus=candidate,minus=candidate;auto vp=vascular,vm=vascular;
    for(unsigned i=0;i<2*nodes;++i){
        plus[i].positionAndMass.x+=float(epsilon*micro.time.x*direction[i].x);minus[i].positionAndMass.x-=float(epsilon*micro.time.x*direction[i].x);
        plus[i].positionAndMass.y+=float(epsilon*micro.time.x*direction[i].y);minus[i].positionAndMass.y-=float(epsilon*micro.time.x*direction[i].y);
        plus[i].positionAndMass.z+=float(epsilon*micro.time.x*direction[i].z);minus[i].positionAndMass.z-=float(epsilon*micro.time.x*direction[i].z);
    }
    for(unsigned env=0;env<2;++env){const unsigned i=env*layout.vascularUnknownCount+w.vascular.layout.cavities.z;const auto d=direction[layout.vascularBase+i].x;vp[i].x+=float(epsilon*d);vm[i].x-=float(epsilon*d);}
    const auto fp=forces(plus,vp,compartments,false),fm=forces(minus,vm,compartments,false),tangent=apply(candidate,std::vector<NMFGMRESStateGPU>(2));
    const nm_float4 baseline{.0001f,-.0002f,.0003f,0};double tangentError=0,tangentScale=0;
    for(unsigned i=0;i<2*nodes;++i){const fixture::Vec fd{-(fp[i].x-fm[i].x)*micro.time.x/(2*epsilon),-(fp[i].y-fm[i].y)*micro.time.x/(2*epsilon),-(fp[i].z-fm[i].z)*micro.time.x/(2*epsilon)};
        const fixture::Vec actual{double(tangent[i].x)-baseline.x,double(tangent[i].y)-baseline.y,double(tangent[i].z)-baseline.z};
        for(unsigned j=0;j<3;++j){need(std::isfinite(actual[j]),"nonfinite production cavity Jv");tangentError=std::max(tangentError,std::abs(actual[j]-fd[j]));tangentScale=std::max(tangentScale,std::abs(fd[j]));}}
    need(tangentScale>1e-9&&tangentError/tangentScale<3e-3,"production cavity Jv disagrees with finite differences");
    // Translate both configurations, including the origin used by production.
    // Recompute the FP64 oracle from the actual translated FP32 coordinates;
    // this separates world-origin dependence from coordinate rounding.
    auto translatedAccepted=accepted,translatedCandidate=candidate;
    for(unsigned i=0;i<2*nodes;++i){
        translatedAccepted[i].positionAndMass.x+=.25f;translatedCandidate[i].positionAndMass.x+=.25f;
        translatedAccepted[i].positionAndMass.y+=.125f;translatedCandidate[i].positionAndMass.y+=.125f;
        translatedAccepted[i].positionAndMass.z-=.5f;translatedCandidate[i].positionAndMass.z-=.5f;
    }
    acceptedBuffer=gpuBuffer(run.device,translatedAccepted);
    const auto translatedForce=forces(translatedCandidate,vascular,compartments,false);
    double translatedForceError=0,translatedWorkError=0;
    for(unsigned env=0;env<2;++env){
        std::vector<fixture::Vec> a(nodes),b(nodes);for(unsigned i=0;i<nodes;++i){
            const auto& pa=translatedAccepted[env*nodes+i].positionAndMass;const auto& pb=translatedCandidate[env*nodes+i].positionAndMass;
            a[i]={pa.x,pa.y,pa.z};b[i]={pb.x,pb.y,pb.z};}
        const auto dg=fixture::discreteGradient(a,b,shell.lumenFaces);const auto row=w.vascular.layout.cavities.z;
        const double pressure=double(vascular[env*layout.vascularUnknownCount+row].x)*w.vascular.unknowns[row].initialAndScaling.y-compartments[0].compliance.w;
        double work=0;for(unsigned i=0;i<nodes;++i){const auto& value=translatedForce[env*nodes+i];const fixture::Vec f{value.x,value.y,value.z};
            for(unsigned j=0;j<3;++j)translatedForceError=std::max(translatedForceError,std::abs(f[j]-pressure*dg[i][j]));
            work+=fixture::dot(f,fixture::subtract(b[i],a[i]));}
        const double expected=pressure*(fixture::volume(b,shell.lumenFaces)-fixture::volume(a,shell.lumenFaces));
        translatedWorkError=std::max(translatedWorkError,std::abs(work-expected)/std::max(1e-18,std::abs(expected)));
    }
    need(translatedForceError/forceScale<5e-6&&translatedWorkError<2e-5,"translated production force/work disagrees with closed-surface oracle");
    acceptedBuffer=gpuBuffer(run.device,accepted);

    // Exercise the actual volume/flow/transport residual and lifted Jacobian,
    // including the independent signed pressure row and FEM velocity entries.
    const auto residualPipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_residual");
    const auto vascularOperatorPipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_operator");
    const auto preparePipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_prepare_elastance");
    auto padded=[&]<class T>(const std::vector<T>& values){return gpuBuffer(run.device,values.empty()?std::vector<T>(1):values);};
    const auto connectionBuffer=padded(w.vascular.connections),tissueBuffer=padded(w.vascular.tissues),exchangeBuffer=padded(w.vascular.exchanges);
    const auto connectionsIncidence=padded(w.vascular.connectionIncidence),connectionsRanges=padded(w.vascular.connectionRanges);
    const auto bloodIncidence=padded(w.vascular.bloodExchangeIncidence),bloodRanges=padded(w.vascular.bloodExchangeRanges);
    const auto tissueIncidence=padded(w.vascular.tissueExchangeIncidence),tissueRanges=padded(w.vascular.tissueExchangeRanges);
    const auto mapping=padded(w.vascular.compartmentCavity),vascularAccepted=gpuBuffer(run.device,run.state().vascularState);
    auto coupledV=vascular,coupledDirection=direction;
    const unsigned pressureRow=w.vascular.layout.cavities.z,flowRow=w.vascular.layout.offsets.y,amountRow=w.vascular.layout.offsets.z;
    for(unsigned env=0;env<2;++env){const unsigned base=env*layout.vascularUnknownCount;
        coupledV[base+flowRow].x=.5f;coupledV[base+amountRow].x=.2f;coupledV[base+amountRow+1].x=1.8f;
        coupledV[base+pressureRow].x=env==0?1.2f:-1.3f;
        coupledDirection[layout.vascularBase+base].x=.003f;coupledDirection[layout.vascularBase+base+1].x=-.002f;
        coupledDirection[layout.vascularBase+base+flowRow].x=.07f;
        coupledDirection[layout.vascularBase+base+amountRow].x=.02f;coupledDirection[layout.vascularBase+base+amountRow+1].x=-.03f;
    }
    auto equationWithUnknownBuffer=[&](const std::vector<NMFEMNodeStateGPU>& fem,const std::vector<nm_float4>& states,
                      const std::vector<nm_float4>& lifted,bool derivative,id<MTLBuffer> equationUnknownBuffer){
        auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];
        const auto p=gpuBuffer(run.device,compartments),v=gpuBuffer(run.device,states),d=gpuBuffer(run.device,lifted),f=gpuBuffer(run.device,fem);
        const auto out=gpuBuffer(run.device,std::vector<nm_float4>(layout.unknownCount,nm_float4{17,18,19,20}));
        const auto solver=gpuBuffer(run.device,std::vector<NMFGMRESStateGPU>(2)),status=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));
        const auto elastance=gpuBuffer(run.device,std::vector<float>(4)),working=gpuBuffer(run.device,std::vector<std::uint32_t>(2*layout.vascularUnknownCount));
        const std::uint32_t useWorking=0;
        const auto clock=gpuBuffer(run.device,run.state().vascularClock);
        [encoder setComputePipelineState:preparePipeline];
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&equationVascularLayout length:sizeof(equationVascularLayout) atIndex:1];
        [encoder setBuffer:p offset:0 atIndex:2];[encoder setBuffer:clock offset:0 atIndex:3];[encoder setBuffer:elastance offset:0 atIndex:4];[encoder setBuffer:status offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(2*w.vascular.layout.counts.x,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
        encoder=[command computeCommandEncoder];[encoder setComputePipelineState:derivative?vascularOperatorPipeline:residualPipeline];
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&equationVascularLayout length:sizeof(equationVascularLayout) atIndex:1];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:equationUnknownBuffer offset:0 atIndex:3];[encoder setBuffer:p offset:0 atIndex:4];
        [encoder setBuffer:connectionBuffer offset:0 atIndex:5];[encoder setBuffer:tissueBuffer offset:0 atIndex:6];[encoder setBuffer:exchangeBuffer offset:0 atIndex:7];
        [encoder setBuffer:connectionsIncidence offset:0 atIndex:8];[encoder setBuffer:connectionsRanges offset:0 atIndex:9];
        [encoder setBuffer:bloodIncidence offset:0 atIndex:10];[encoder setBuffer:bloodRanges offset:0 atIndex:11];
        [encoder setBuffer:tissueIncidence offset:0 atIndex:12];[encoder setBuffer:tissueRanges offset:0 atIndex:13];
        [encoder setBuffer:vascularAccepted offset:0 atIndex:14];[encoder setBuffer:v offset:0 atIndex:15];[encoder setBuffer:d offset:0 atIndex:16];
        [encoder setBuffer:out offset:0 atIndex:17];[encoder setBuffer:solver offset:0 atIndex:18];[encoder setBuffer:status offset:0 atIndex:19];
        [encoder setBuffer:elastance offset:0 atIndex:20];[encoder setBuffer:working offset:0 atIndex:21];[encoder setBytes:&useWorking length:sizeof(useWorking) atIndex:22];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:23];[encoder setBuffer:faceBuffer offset:0 atIndex:24];[encoder setBuffer:mapping offset:0 atIndex:25];
        [encoder setBuffer:f offset:0 atIndex:26];[encoder setBuffer:acceptedBuffer offset:0 atIndex:27];[encoder setBytes:&layout length:sizeof(layout) atIndex:30];
        [encoder dispatchThreads:MTLSizeMake(2*layout.vascularUnknownCount,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(status.contents)[env].code==NM_STATUS_SUCCESS,"signed-pressure manufactured vascular equation rejected");
        const auto* values=static_cast<nm_float4*>(out.contents);
        for(unsigned i=0;i<layout.vascularBase;++i)need(values[i].x==17&&values[i].y==18&&values[i].z==19&&values[i].w==20,"vascular equation overwrote unrelated FEM row");
        std::vector<nm_float4> result(values+layout.vascularBase,values+layout.unknownCount);
        for(const auto& value:result)need(std::isfinite(value.x)&&value.y==0&&value.z==0&&value.w==0,"nonfinite or malformed vascular equation output");
        return result;
    };
    auto equation=[&](const std::vector<NMFEMNodeStateGPU>& fem,const std::vector<nm_float4>& states,
                      const std::vector<nm_float4>& lifted,bool derivative){
        return equationWithUnknownBuffer(fem,states,lifted,derivative,unknownBuffer);
    };
    const auto residual=equation(candidate,coupledV,coupledDirection,false),jacobian=equation(candidate,coupledV,coupledDirection,true);
    auto coupledPlus=coupledV,coupledMinus=coupledV;
    for(unsigned i=0;i<coupledV.size();++i){const auto delta=coupledDirection[layout.vascularBase+i].x;coupledPlus[i].x+=float(epsilon*delta);coupledMinus[i].x-=float(epsilon*delta);}
    const auto residualPlus=equation(plus,coupledPlus,coupledDirection,false),residualMinus=equation(minus,coupledMinus,coupledDirection,false);
    double equationFD=0,geometryAnalytic=0,flowAnalytic=0;
    for(unsigned i=0;i<jacobian.size();++i){const double fd=-(double(residualPlus[i].x)-residualMinus[i].x)/(2*epsilon);
        const double error=std::abs(jacobian[i].x-fd);equationFD=std::max(equationFD,error/std::max(1e-3,std::abs(fd)));
        need(error<5e-6+3e-3*std::abs(fd),"coupled production vascular Jv disagrees with rowwise finite differences row="+std::to_string(i));}
    for(unsigned env=0;env<2;++env){const unsigned base=env*layout.vascularUnknownCount;
        std::vector<fixture::Vec> x(nodes);for(unsigned i=0;i<nodes;++i){const auto& point=candidate[env*nodes+i].positionAndMass;x[i]={point.x,point.y,point.z};}
        const auto gradient=fixture::gradient(x,shell.lumenFaces);double geometryDirection=0;
        for(unsigned i=0;i<nodes;++i){const auto& d=coupledDirection[env*nodes+i];geometryDirection+=fixture::dot(gradient[i],{double(micro.time.x)*d.x,double(micro.time.x)*d.y,double(micro.time.x)*d.z});}
        auto physical=[&](unsigned row){return double(coupledV[base+row].x)*w.vascular.unknowns[row].initialAndScaling.y;};
        auto delta=[&](unsigned row){return double(coupledDirection[layout.vascularBase+base+row].x)*w.vascular.unknowns[row].initialAndScaling.y;};
        const double geometryR=(physical(0)-fixture::volume(x,shell.lumenFaces))/w.vascular.unknowns[pressureRow].initialAndScaling.z;
        const double geometryJ=(delta(0)-geometryDirection)/w.vascular.unknowns[pressureRow].initialAndScaling.z;
        need(std::abs(double(residual[base+pressureRow].x)+geometryR)<2e-6,"production cavity geometry residual differs from FP64 closed volume");
        geometryAnalytic=std::max(geometryAnalytic,std::abs(jacobian[base+pressureRow].x-geometryJ));
        need(geometryAnalytic<2e-6&&std::abs(geometryDirection)>1e-11,"cavity geometry Jv omitted manufactured wall velocity");
        const auto& c=compartments[1].compliance;const double sourcePressure=c.w+c.y+(physical(1)-c.x)/c.z;
        const double flowR=(w.vascular.connections[0].physical.x*physical(flowRow)-sourcePressure+physical(pressureRow))/w.vascular.unknowns[flowRow].initialAndScaling.z;
        const double flowJ=(w.vascular.connections[0].physical.x*delta(flowRow)-delta(1)/c.z+delta(pressureRow))/w.vascular.unknowns[flowRow].initialAndScaling.z;
        flowAnalytic=std::max(flowAnalytic,std::abs(jacobian[base+flowRow].x-flowJ));
        need(std::abs(double(residual[base+flowRow].x)+flowR)<2e-6&&flowAnalytic<2e-6,"flow pressure-law residual/Jv did not use independent signed pressure");
    }
    // A pure independent pressure direction changes flow, never geometry.
    std::vector<nm_float4> pressureOnly(layout.unknownCount);for(unsigned env=0;env<2;++env)pressureOnly[layout.vascularBase+env*layout.vascularUnknownCount+pressureRow].x=1;
    const auto pressureJv=equation(candidate,coupledV,pressureOnly,true);
    for(unsigned env=0;env<2;++env){const auto base=env*layout.vascularUnknownCount;
        need(pressureJv[base+pressureRow].x==0&&std::abs(pressureJv[base+flowRow].x-w.vascular.unknowns[pressureRow].initialAndScaling.y/w.vascular.unknowns[flowRow].initialAndScaling.z)<1e-7,
             "independent cavity pressure leaked into geometric volume equation or failed to drive flow");}
    std::cout<<"cavity_coupled_equations=pass signed_pressure=pass geometry_residual_fp64=pass geometry_jv_absolute_error="<<geometryAnalytic
             <<" flow_pressure_jv_absolute_error="<<flowAnalytic<<" rowwise_jv_fd_relative_max="<<equationFD
             <<" translated_force_relative_error="<<translatedForceError/forceScale<<" translated_work_relative_error="<<translatedWorkError<<'\n';
    auto staticNode=candidate;const auto inactiveNode=shell.innerNodes[0];staticNode[inactiveNode].velocityAndInverseMass.w=0;staticNode[inactiveNode].restAndFixed.w=1;
    std::vector<NMFGMRESStateGPU> converged(2);converged[1].diagnostics.z=1;const auto skipped=apply(staticNode,converged);
    need(std::memcmp(&skipped[inactiveNode],&baseline,sizeof(baseline))==0,"static wall tangent row was modified");
    for(unsigned i=nodes;i<2*nodes;++i)need(std::memcmp(&skipped[i],&baseline,sizeof(baseline))==0,"converged environment tangent row was modified");
    std::cout<<"cavity_production_operators=pass force_fp64_relative_max="<<forceError/forceScale<<" pressure_work_relative_max="<<workError
             <<" jv_finite_difference_relative_max="<<tangentError/tangentScale<<" external_load_merge=pass pressure_gauge_invariance=pass static_and_converged_rows=unchanged\n";
    {
        const auto savedLayout = equationVascularLayout;
        const auto& cavity = w.vascular.cavities[0];
        need(cavity.identity.y != cavity.identity.w,
            "compensated cavity scale regression requires distinct volume and pressure rows");
        auto compensatedUnknowns = w.vascular.unknowns;
        const float pressureEquationScale = 0.125e-6f;
        const float volumeEquationScale = 2.0e-6f;
        compensatedUnknowns[cavity.identity.w].initialAndScaling.z = pressureEquationScale;
        compensatedUnknowns[cavity.identity.y].initialAndScaling.z = volumeEquationScale;
        auto compensatedUnknownBuffer = gpuBuffer(run.device, compensatedUnknowns);
        need(std::abs(double(pressureEquationScale) - double(volumeEquationScale)) > 8.0 *
                std::max(double(pressureEquationScale), 1.0e-12),
            "compensated cavity scale regression did not separate equation scales");
        auto wordAccepted = accepted;
        auto wordCandidate = candidate;
        const double translation[3] = {65536.0, -32768.0, 16384.0};
        auto installReferencePosition = [&](NMFEMNodeStateGPU& state,
                                            const NMFEMNodeStateGPU& source,
                                            const double low[3]) {
            for (unsigned axis = 0u; axis < 3u; ++axis) {
                const float rest = cavityComponent(source.positionAndMass, axis);
                cavityComponent(state.restAndFixed, axis) = rest;
                cavityComponent(state.referenceDisplacementAndMode, axis) = float(translation[axis]);
                cavityComponent(state.referenceDisplacementLow, axis) = float(low[axis]);
                state.referenceDisplacementAndMode.w = 1.0f;
                cavityComponent(state.positionAndMass, axis) = float(double(rest) + translation[axis] + low[axis]);
            }
        };
        for (unsigned global = 0u; global < 2u * nodes; ++global) {
            const double zero[3] = {0.0, 0.0, 0.0};
            installReferencePosition(wordAccepted[global], accepted[global], zero);
            double candidateDelta[3];
            for (unsigned axis = 0u; axis < 3u; ++axis)
                candidateDelta[axis] = double(cavityComponent(candidate[global].positionAndMass, axis)) -
                    double(cavityComponent(accepted[global].positionAndMass, axis));
            installReferencePosition(wordCandidate[global], accepted[global], candidateDelta);
        }
        auto representedPositions = [&](const std::vector<NMFEMNodeStateGPU>& values,
                                        unsigned environment) {
            // Evaluate the FP64 oracle in a common translated frame. Absolute
            // triple products at 65 km cancel before the small cavity survives.
            std::vector<fixture::Vec> result(nodes);
            for (unsigned node = 0u; node < nodes; ++node) {
                const auto& state = values[environment * nodes + node];
                result[node] = {
                    double(state.restAndFixed.x) + (double(state.referenceDisplacementAndMode.x) - translation[0]) + double(state.referenceDisplacementLow.x),
                    double(state.restAndFixed.y) + (double(state.referenceDisplacementAndMode.y) - translation[1]) + double(state.referenceDisplacementLow.y),
                    double(state.restAndFixed.z) + (double(state.referenceDisplacementAndMode.z) - translation[2]) + double(state.referenceDisplacementLow.z)};
            }
            return result;
        };
        equationVascularLayout.ranges.w |=
            NM_VASCULAR_LAYOUT_PERSISTENT_LOW_WORDS |
            NM_VASCULAR_LAYOUT_COMPENSATED_CAVITY_WORDS;
        acceptedBuffer = gpuBuffer(run.device, wordAccepted);
        const auto wordResidual = equationWithUnknownBuffer(wordCandidate, coupledV, coupledDirection, false, compensatedUnknownBuffer);
        const auto wordJacobian = equationWithUnknownBuffer(wordCandidate, coupledV, coupledDirection, true, compensatedUnknownBuffer);
        constexpr double wordEpsilon = 1.0 / 64.0;
        auto wordPlus = wordCandidate, wordMinus = wordCandidate;
        for (unsigned global = 0u; global < 2u * nodes; ++global) {
            for (unsigned axis = 0u; axis < 3u; ++axis) {
                const double delta = wordEpsilon * double(micro.time.x) * double(cavityComponent(coupledDirection[global], axis));
                cavityComponent(wordPlus[global].referenceDisplacementLow, axis) += float(delta);
                cavityComponent(wordMinus[global].referenceDisplacementLow, axis) -= float(delta);
                cavityComponent(wordPlus[global].positionAndMass, axis) = float(
                    double(cavityComponent(wordPlus[global].restAndFixed, axis)) +
                    double(cavityComponent(wordPlus[global].referenceDisplacementAndMode, axis)) +
                    double(cavityComponent(wordPlus[global].referenceDisplacementLow, axis)));
                cavityComponent(wordMinus[global].positionAndMass, axis) = float(
                    double(cavityComponent(wordMinus[global].restAndFixed, axis)) +
                    double(cavityComponent(wordMinus[global].referenceDisplacementAndMode, axis)) +
                    double(cavityComponent(wordMinus[global].referenceDisplacementLow, axis)));
            }
        }
        auto wordVascularPlus = coupledV, wordVascularMinus = coupledV;
        for (unsigned env = 0u; env < 2u; ++env) {
            const unsigned base = env * layout.vascularUnknownCount;
            for (unsigned row = 0u; row < layout.vascularUnknownCount; ++row) {
                const float delta = float(wordEpsilon) *
                    coupledDirection[layout.vascularBase + base + row].x;
                wordVascularPlus[base + row].x += delta;
                wordVascularMinus[base + row].x -= delta;
            }
        }
        const auto wordResidualPlus = equationWithUnknownBuffer(wordPlus, wordVascularPlus, coupledDirection, false, compensatedUnknownBuffer);
        const auto wordResidualMinus = equationWithUnknownBuffer(wordMinus, wordVascularMinus, coupledDirection, false, compensatedUnknownBuffer);
        double oracleError = 0.0, wordFiniteDifferenceError = 0.0, wordFiniteDifferenceScale = 0.0;
        for (unsigned env = 0u; env < 2u; ++env) {
            const auto exact = representedPositions(wordCandidate, env);
            const double oracleVolume = fixture::volume(exact, shell.lumenFaces);
            const unsigned base = env * layout.vascularUnknownCount;
            const double physicalVolume = double(coupledV[base + cavity.identity.y].x) *
                double(w.vascular.unknowns[cavity.identity.y].initialAndScaling.y);
            const double equationScale = double(compensatedUnknowns[pressureRow].initialAndScaling.z);
            const double expectedResidual = (oracleVolume - physicalVolume) / equationScale;
            const double incorrectlyVolumeScaled = (oracleVolume - physicalVolume) /
                double(volumeEquationScale);
            need(std::abs(expectedResidual - incorrectlyVolumeScaled) > 2.0e-7,
                 "distinct-scale fixture cannot detect volume-row normalization");
            const unsigned outputRow = base + pressureRow;
            oracleError = std::max(oracleError,
                std::abs(double(wordResidual[outputRow].x) - expectedResidual));
            if (std::abs(double(wordResidual[outputRow].x) - expectedResidual) >= 2.0e-7)
                std::cerr << "cavity_word_oracle_diagnostic env=" << env
                          << " actual=" << std::setprecision(17) << wordResidual[outputRow].x
                          << " expected=" << expectedResidual << " volume=" << oracleVolume
                          << " physical=" << physicalVolume << " scale=" << equationScale << "\n";
            need(std::abs(double(wordResidual[outputRow].x) - expectedResidual) < 2.0e-7,
                 "compensated cavity volume differs from the same-state FP64 closed-surface oracle");
            const auto gradient = fixture::gradient(exact, shell.lumenFaces);
            double geometryDirection = 0.0;
            for (unsigned node = 0u; node < nodes; ++node) {
                const auto& v = coupledDirection[env * nodes + node];
                geometryDirection += fixture::dot(gradient[node],
                    {double(micro.time.x) * double(v.x),
                     double(micro.time.x) * double(v.y),
                     double(micro.time.x) * double(v.z)});
            }
            const double pressureDirection = double(coupledDirection[
                layout.vascularBase + base + cavity.identity.y].x) *
                double(w.vascular.unknowns[cavity.identity.y].initialAndScaling.y);
            const double expectedJvp = (pressureDirection - geometryDirection) / equationScale;
            const double finiteDifference = -(
                double(wordResidualPlus[outputRow].x) -
                double(wordResidualMinus[outputRow].x)) / (2.0 * wordEpsilon);
            const double absFD = std::abs(double(wordJacobian[outputRow].x) - finiteDifference);
            const double absOracle = std::abs(double(wordJacobian[outputRow].x) - expectedJvp);
            wordFiniteDifferenceError = std::max(wordFiniteDifferenceError, absFD);
            wordFiniteDifferenceScale = std::max(wordFiniteDifferenceScale,
                std::max(std::abs(finiteDifference), std::abs(expectedJvp)));
            need(absOracle < 3.0e-7,
                 "compensated cavity JVP differs from the FP64 closed-surface gradient");
            need(absFD < 2.0e-6 + 2.0e-3 * std::abs(finiteDifference),
                 "compensated cavity JVP differs from random-direction finite differences");
        }
        // The common translation is represented only in the high displacement
        // limb; every local perturbation remains in the low limb and is retained.
        need(oracleError < 2.0e-7 && wordFiniteDifferenceError <
                2.0e-6 + 2.0e-3 * wordFiniteDifferenceScale,
             "compensated cavity cancellation audit exceeded its bound");
        std::cout << "cavity_compensated_words=pass same_state_fp64_oracle=pass "
                  << "random_direction_fd=pass translation_cancellation=pass "
                  << "distinct_equation_scales=pass pressure_scale=" << pressureEquationScale
                  << " volume_scale=" << volumeEquationScale << " oracle_abs=" << oracleError << " jvp_fd_abs="
                  << wordFiniteDifferenceError << "\n";

        // Exercise the pressure force and mechanical cavity operator with the
        // same high/low geometry consumed by the compensated volume/JVP.
        const auto wordForces = forces(wordCandidate, coupledV, compartments, false);
        auto coupledPlus = coupledV, coupledMinus = coupledV;
        for (unsigned env = 0u; env < 2u; ++env) {
            const unsigned stateBase = env * layout.vascularUnknownCount;
            for (unsigned local = 0u; local < layout.vascularUnknownCount; ++local) {
                const float delta = coupledDirection[layout.vascularBase + stateBase + local].x;
                coupledPlus[stateBase + local].x += float(wordEpsilon * double(delta));
                coupledMinus[stateBase + local].x -= float(wordEpsilon * double(delta));
            }
        }
        const auto wordForcesPlus = forces(wordPlus, coupledPlus, compartments, false);
        const auto wordForcesMinus = forces(wordMinus, coupledMinus, compartments, false);
        auto savedDirection = direction;
        auto savedVascular = vascular;
        direction = coupledDirection;
        vascular = coupledV;
        const auto wordOperator = apply(wordCandidate,
            std::vector<NMFGMRESStateGPU>(2), nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
        direction = savedDirection;
        vascular = savedVascular;
        double wordForceOracleError = 0.0, wordForceScale = 0.0;
        double wordPressureWorkError = 0.0, wordPressureWorkScale = 0.0;
        double wordForceJVPError = 0.0, wordForceJVPScale = 0.0;
        for (unsigned env = 0u; env < 2u; ++env) {
            const auto a = representedPositions(wordAccepted, env);
            const auto b = representedPositions(wordCandidate, env);
            const auto discrete = fixture::discreteGradient(a, b, shell.lumenFaces);
            const double pressure = double(coupledV[env * layout.vascularUnknownCount + pressureRow].x) *
                double(w.vascular.unknowns[pressureRow].initialAndScaling.y) - compartments[0].compliance.w;
            double work = 0.0;
            for (unsigned node = 0u; node < nodes; ++node) {
                const unsigned global = env * nodes + node;
                const auto& force = wordForces[global];
                for (unsigned axis = 0u; axis < 3u; ++axis) {
                    const double expected = pressure * discrete[node][axis];
                    wordForceOracleError = std::max(wordForceOracleError,
                        std::abs(double(cavityComponent(force, axis)) - expected));
                    wordForceScale = std::max(wordForceScale, std::abs(expected));
                }
                work += fixture::dot({double(force.x), double(force.y), double(force.z)},
                    fixture::subtract(b[node], a[node]));
                const auto& plusForce = wordForcesPlus[global];
                const auto& minusForce = wordForcesMinus[global];
                const auto& actual = wordOperator[global];
                for (unsigned axis = 0u; axis < 3u; ++axis) {
                    const double finiteDifference = -double(micro.time.x) *
                        (double(cavityComponent(plusForce, axis)) - double(cavityComponent(minusForce, axis))) /
                        (2.0 * wordEpsilon);
                    wordForceJVPError = std::max(wordForceJVPError,
                        std::abs(double(cavityComponent(actual, axis)) - finiteDifference));
                    wordForceJVPScale = std::max(wordForceJVPScale,
                        std::max(std::abs(double(cavityComponent(actual, axis))), std::abs(finiteDifference)));
                }
            }
            const double expectedWork = pressure *
                (fixture::volume(b, shell.lumenFaces) - fixture::volume(a, shell.lumenFaces));
            wordPressureWorkError = std::max(wordPressureWorkError, std::abs(work - expectedWork));
            wordPressureWorkScale = std::max(wordPressureWorkScale, std::abs(expectedWork));
        }
        need(wordForceScale > 0.0 && wordForceOracleError / wordForceScale < 2.0e-4,
            "compensated cavity pressure force differs from high/low FP64 geometry oracle");
        need(wordPressureWorkError / std::max(1.0e-18, wordPressureWorkScale) < 3.0e-4,
            "compensated pressure force violates closed-surface work identity");
        need(wordForceJVPScale > 1.0e-15 &&
                wordForceJVPError < 2.0e-7 + 3.0e-3 * wordForceJVPScale,
            "compensated cavity mechanical JVP differs from force finite differences");
        std::cout << "cavity_compensated_force=pass same_high_low_geometry=pass "
                  << "pressure_work_closure=pass mechanical_jvp_fd=pass force_rel="
                  << wordForceOracleError / wordForceScale << " work_rel="
                  << wordPressureWorkError / std::max(1.0e-18, wordPressureWorkScale)
                  << " jvp_rel=" << wordForceJVPError / wordForceJVPScale << "\n";
        equationVascularLayout = savedLayout;
        acceptedBuffer = gpuBuffer(run.device, accepted);
    }
}
void inspectStep(const Run& run,const RuntimeStateSnapshot& first,const RuntimeStateSnapshot& before,
                 const RuntimeStateSnapshot& after,const fixture::Shell& shell,Metrics& metrics){
    const auto& layout=run.world.vascular.layout;
    const unsigned pressureRow=layout.cavities.z;
    need(run.world.vascular.unknowns.size()==6,"expected two volumes,flow,two amounts,one wall pressure");
    paired(after.femNodes,"FEM state");paired(after.femFields,"FEM fields");paired(after.vascularState,"vascular state");paired(after.vascularClock,"clock");
    const double dt=run.runtime.timestepSeconds();
    const auto exponent=std::bit_cast<std::int32_t>(layout.clock.x);
    const auto ticks=static_cast<std::uint64_t>(std::ldexp(dt,-exponent));
    for(unsigned env=0;env<2;++env){
        const auto& previousClock=before.vascularClock[env];const auto& currentClock=after.vascularClock[env];
        const std::uint64_t expectedLow=previousClock.low+ticks;
        need(currentClock.low==expectedLow&&currentClock.high==previousClock.high+(expectedLow<previousClock.low?1u:0u),"accepted cavity clock did not advance exactly");
        auto x=positions(after,env,unsigned(shell.object.femNodes.size()));
        const auto old=positions(before,env,unsigned(shell.object.femNodes.size()));
        const auto origin=positions(first,env,unsigned(shell.object.femNodes.size()));
        for(unsigned i=0;i<x.size();++i)for(unsigned j=0;j<3;++j)need(std::isfinite(x[i][j]),"nonfinite FEM state");
        for(unsigned i=0;i<x.size();++i){const unsigned global=env*unsigned(x.size())+i;
            if(after.femNodes[global].positionAndMass.w!=first.femNodes[global].positionAndMass.w){
                std::ostringstream diagnostic;diagnostic<<std::setprecision(17)<<"cavity pressure changed mechanical nodal mass env="<<env<<" node="<<i
                    <<" initial="<<first.femNodes[global].positionAndMass.w<<" previous="<<before.femNodes[global].positionAndMass.w<<" current="<<after.femNodes[global].positionAndMass.w
                    <<" initial_bits="<<std::hex<<std::bit_cast<std::uint32_t>(first.femNodes[global].positionAndMass.w)<<" current_bits="<<std::bit_cast<std::uint32_t>(after.femNodes[global].positionAndMass.w)<<std::dec
                    <<" initial_inverse_mass="<<first.femNodes[global].velocityAndInverseMass.w<<" current_inverse_mass="<<after.femNodes[global].velocityAndInverseMass.w
                    <<" initial_fixed="<<first.femNodes[global].restAndFixed.w<<" current_fixed="<<after.femNodes[global].restAndFixed.w<<" status="<<after.statuses[env].code;
                need(false,diagnostic.str());}}

        for(const auto i:shell.object.femFixedNodes)need(x[i]==origin[i],"fixed outer wall moved");
        for(const auto i:shell.innerNodes)metrics.motion=std::max(metrics.motion,std::sqrt(fixture::dot(fixture::subtract(x[i],origin[i]),fixture::subtract(x[i],origin[i]))));
        const double geometricVolume=fixture::volume(x,shell.lumenFaces);
        const double lumen=run.physical(after,0,env),reservoir=run.physical(after,1,env),q=run.physical(after,layout.offsets.y,env);
        const double m0=run.physical(after,layout.offsets.z,env),m1=run.physical(after,layout.offsets.z+1,env);
        need(lumen>0&&reservoir>0&&geometricVolume>0&&m0>=0&&m1>=0,"nonpositive volume or negative tracer");
        const double initialVolume=run.physical(first,0,env)+run.physical(first,1,env);
        const double initialTracer=run.physical(first,layout.offsets.z,env)+run.physical(first,layout.offsets.z+1,env);
        metrics.geometry=std::max(metrics.geometry,std::abs(lumen-geometricVolume)/run.world.vascular.unknowns[0].initialAndScaling.y);
        metrics.blood=std::max(metrics.blood,std::abs(lumen+reservoir-initialVolume)/initialVolume);
        metrics.tracer=std::max(metrics.tracer,std::abs(m0+m1-initialTracer)/initialTracer);
        metrics.balance=std::max(metrics.balance,std::abs(lumen-run.physical(before,0,env)-dt*q)/run.world.vascular.unknowns[0].initialAndScaling.y);
        const auto& c=run.world.vascular.compartments[1].compliance;
        const double reservoirPressure=c.w+c.y+(reservoir-c.x)/c.z;
        const double pressure=run.physical(after,pressureRow,env);
        const double resistance=run.world.vascular.connections[0].physical.x;
        metrics.flow=std::max(metrics.flow,std::abs(resistance*q-(reservoirPressure-pressure))/run.world.vascular.unknowns[layout.offsets.y].initialAndScaling.z);
        metrics.flowMaximum=std::max(metrics.flowMaximum,std::abs(q));
        const auto dg=fixture::discreteGradient(old,x,shell.lumenFaces);double work=0;
        const double transmural=pressure-run.world.vascular.compartments[0].compliance.w;
        for(unsigned i=0;i<x.size();++i)work+=transmural*fixture::dot(dg[i],fixture::subtract(x[i],old[i]));
        const double pressureWork=transmural*(geometricVolume-fixture::volume(old,shell.lumenFaces));
        metrics.work=std::max(metrics.work,std::abs(work-pressureWork)/std::max(1e-18,std::abs(pressureWork)));
    }
    need(metrics.geometry<2e-5,"cavity volume and accepted FEM geometry diverged");
    need(metrics.blood<1e-4&&metrics.tracer<1e-4,"closed blood/tracer conservation failed");
    need(metrics.balance<2e-5&&metrics.flow<2e-5,"accepted hydraulic residual disagrees with FP64 balance");
    need(metrics.work<2e-8,"independent discrete wall-work oracle failed");
}
void geometryOracle(){
    const auto shell=fixture::hollowShell();const auto& old=shell.object.femNodes;auto next=old;
    double tissueVolume=0;for(const auto& tet:shell.object.tetrahedra){const auto& t=tet.nodes;
        const double v=fixture::dot(fixture::subtract(old[t[1]],old[t[0]]),fixture::cross(fixture::subtract(old[t[2]],old[t[0]]),fixture::subtract(old[t[3]],old[t[0]])))/6.;
        need(v>0,"nonpositive shell tetrahedron");tissueVolume+=v;}
    need(std::abs(tissueVolume-26e-6)<1e-18&&std::abs(fixture::volume(old,shell.lumenFaces)-1e-6)<1e-19,"shell and lumen overlap or volume differs");
    for(unsigned i=0;i<shell.innerNodes.size();++i){auto& x=next[shell.innerNodes[i]];x[0]+=1e-4*std::sin(double(i)+.3);x[1]+=2e-4*std::cos(double(i)+.2);x[2]+=1e-4*std::sin(2*double(i)+.1);}
    const auto dg=fixture::discreteGradient(old,next,shell.lumenFaces);double work=0;fixture::Vec force{};
    for(unsigned i=0;i<old.size();++i){work+=fixture::dot(dg[i],fixture::subtract(next[i],old[i]));force=fixture::add(force,dg[i]);}
    const double change=fixture::volume(next,shell.lumenFaces)-fixture::volume(old,shell.lumenFaces);
    need(std::abs(work-change)<1e-20&&std::sqrt(fixture::dot(force,force))<1e-18,"closed surface work/force geometry oracle failed");
    std::cout<<"cavity_geometry_oracle=pass tissue_tets=156 outer_fixed_nodes=56 lumen_free_nodes=8 lumen_faces=12 tissue_volume_m3="<<tissueVolume<<" lumen_volume_m3="<<fixture::volume(old,shell.lumenFaces)<<" pressure_work_error_m3="<<std::abs(work-change)<<'\n';
}
void bloodMassOwner(){
    const auto shell=fixture::hollowShell();
    auto staticLoadWorld=fixture::world(false,.001,true);
    // This coupon intentionally has no deformable-contact pairs. Disable both
    // contact modes at the source so the compiler emits zero deformable
    // contact capacity; production Human/contact admission remains unchanged.
    staticLoadWorld.objects.front().deformableContact=false;
    staticLoadWorld.objects.front().deformableSelfContact=false;
    ScopedEnvironment ownerTangentPolicy("NM_VASCULAR_BLOOD_MASS_TANGENT", "1");
    Run run(staticLoadWorld);const auto& w=run.world;
    need(w.vascular.tissues.size()==1&&w.vascular.tissueBindings.size()==shell.innerNodes.size(),"blood owner fixture did not cook its FEM region");
    const auto& owner=w.vascular.tissues[0];need(owner.identity.w!=0u&&owner.physical.y>1000.f&&owner.physical.z>0.f,"blood owner identity, density or initial mechanical mass was not cooked");
    need(std::abs(double(owner.physical.z)-4e-6*1060.0)<1e-8,"blood owner initial mechanical mass disagrees with hydraulic volume");
    auto contactEnabledSource=fixture::world(false,.001,true);
    const auto contactEnabled=compileWorld(contactEnabledSource);
    need(contactEnabled.succeeded()&&
         contactEnabled.world.dispatch.deformableContactCapacity>0u,
         "enabled FEM self/deformable contact did not reserve candidate capacity");
    auto contactDisabledSource=fixture::world(false,.001,true);
    contactDisabledSource.objects.front().deformableContact=false;
    contactDisabledSource.objects.front().deformableSelfContact=false;
    const auto contactDisabled=compileWorld(contactDisabledSource);
    need(contactDisabled.succeeded()&&
         contactDisabled.world.dispatch.deformableContactCapacity==0u&&
         contactDisabled.world.dispatch.contactPairCount==0u&&
         (contactDisabled.world.objects.front().flags&
             (NM_OBJECT_DISABLE_DEFORMABLE_CONTACT|NM_OBJECT_DISABLE_SELF_CONTACT))==
             (NM_OBJECT_DISABLE_DEFORMABLE_CONTACT|NM_OBJECT_DISABLE_SELF_CONTACT),
         "explicitly disabled deformable contact retained a contact work domain");
    need(same(contactDisabled.world.fem.nodes,w.fem.nodes)&&
         same(contactDisabled.world.fem.tetrahedra,w.fem.tetrahedra),
         "contact capacity opt-out changed coupon FEM geometry or mass");
    auto explicitContactReserveSource=contactDisabledSource;
    explicitContactReserveSource.objects.front().femCapacity.deformableContacts=7u;
    const auto explicitContactReserve=compileWorld(explicitContactReserveSource);
    need(explicitContactReserve.succeeded()&&
         explicitContactReserve.world.dispatch.deformableContactCapacity==7u,
         "explicit deformable contact capacity was discarded when both contact modes were disabled");
    const auto baseline=compileWorld(fixture::world());need(baseline.succeeded(),"blood owner baseline compile failed");
    for (const auto node:shell.innerNodes) {
        const double delta=double(w.fem.nodes[node].positionAndMass.w)-double(baseline.world.fem.nodes[node].positionAndMass.w);
        need(std::abs(delta-double(owner.physical.z)/double(shell.innerNodes.size()))<2e-8,"blood mass partition was not normalized over FEM nodes");
    }
    need(std::abs(owner.spatialFirst.x)<1e-7f&&std::abs(owner.spatialFirst.y)<1e-7f&&std::abs(owner.spatialFirst.z)<1e-7f&&
         std::abs(owner.spatialSecond0.x-2.5e-5f)<1e-8f&&std::abs(owner.spatialSecond1.x-2.5e-5f)<1e-8f&&std::abs(owner.spatialSecond1.z-2.5e-5f)<1e-8f,
         "blood owner spatial moments were not symmetric or source-bound");
    double weightSum=0;for(const auto& binding:w.vascular.tissueBindings)weightSum+=binding.physical.x;
    need(std::abs(weightSum-1.)<1e-7,"blood owner FEM weights are not normalized");
    unsigned compartment=NM_INVALID_INDEX;for(unsigned i=0;i<w.vascular.compartments.size();++i)if(w.vascular.compartments[i].identity.x==12u)compartment=i;
    need(compartment!=NM_INVALID_INDEX&&owner.identity.w==compartment+1u,"blood owner compartment was not canonically resolved");
    const auto initialSnapshot=run.state();
    const auto staticSource=run.runtime.snapshotForFrozenFEMStaticLoad();
    need(initialSnapshot.available&&initialSnapshot.message.empty()&&
         staticSource.state.available&&staticSource.state.message.empty()&&
         !staticSource.contentSHA256.empty()&&
         same(staticSource.state.femNodes,initialSnapshot.femNodes)&&
         same(staticSource.state.vascularState,initialSnapshot.vascularState),
         "explicit frozen-load snapshot differs from ordinary state");
    need(!run.world.vascular.cavities.empty(),
         "frozen static fixture lacks a compiled cavity owner");
    std::cout<<"frozen_static_admission_fixture={\"mutable_flags\":[";
    for(std::size_t i=0;i<run.world.objects.size();++i){
        if(i)std::cout<<",";
        std::cout<<((run.world.objects[i].flags&NM_OBJECT_MUTABLE_TOPOLOGY)!=0u?1:0);
    }
    std::cout<<"],\"nodes\":"<<run.world.dispatch.femNodeCount
        <<",\"tets\":"<<run.world.dispatch.tetrahedronCount
        <<",\"particles\":"<<run.world.dispatch.particleCount
        <<",\"cohesive\":"<<run.world.dispatch.cohesiveFaceCount
        <<",\"mutation_commands\":"<<run.world.dispatch.mutationCommandCount
        <<",\"rigid_contact_pairs\":"<<run.world.dispatch.contactPairCount
        <<",\"deformable_contact_capacity\":"<<run.world.dispatch.deformableContactCapacity
        <<"}\n";
    frozenStaticLoad(run,staticSource);
    const auto accepted=initialSnapshot.femNodes;const auto vascular=initialSnapshot.vascularState;const unsigned nodes=w.dispatch.femNodeCount;
    NSError* error=nil;auto library=[run.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]] error:&error];
    need(library!=nil,"metallib unavailable for blood owner operator check");
    const auto forcePipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_cavity_forces");
    const auto momentPipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_blood_moments");
    NMMicrostepGPU micro{};micro.time={float(run.runtime.timestepSeconds()),float(1./run.runtime.timestepSeconds()),0,0};
    const auto unknownBuffer=gpuBuffer(run.device,w.vascular.unknowns),compartmentBuffer=gpuBuffer(run.device,w.vascular.compartments);
    const auto cavityBuffer=gpuBuffer(run.device,w.vascular.cavities),faceBuffer=gpuBuffer(run.device,w.vascular.cavityFaces);
    const auto incidenceBuffer=gpuBuffer(run.device,w.vascular.cavityNodeIncidence),rangeBuffer=gpuBuffer(run.device,w.vascular.cavityNodeRanges);
    const auto acceptedBuffer=gpuBuffer(run.device,accepted),tissueBuffer=gpuBuffer(run.device,w.vascular.tissues),bindingBuffer=gpuBuffer(run.device,w.vascular.tissueBindings);
    std::vector<float> fixtureElastance(w.dispatch.environmentCount*w.vascular.compartments.size(),1.f);
    for(unsigned env=0;env<w.dispatch.environmentCount;++env) for(unsigned i=0;i<w.vascular.compartments.size();++i)
        if(w.vascular.compartments[i].identity.w==0u||w.vascular.compartments[i].identity.w==3u) fixtureElastance[env*w.vascular.compartments.size()+i]=float(1./w.vascular.compartments[i].compliance.z);
    const auto elastanceBuffer=gpuBufferOrDummy(run.device,fixtureElastance);
    const auto cavityConnectionBuffer=gpuBufferOrDummy(run.device,w.vascular.connections);
    const auto cavityAcceptedVascularBuffer=gpuBuffer(run.device,vascular);
    auto encode=[&](const std::vector<nm_float4>& state,
                    const std::vector<NMFEMNodeStateGPU>& candidateNodes){
        const auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:forcePipeline];
        const auto vascularBuffer=gpuBuffer(run.device,state),candidateBuffer=gpuBuffer(run.device,candidateNodes),borrowed=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes,nm_float4{0,0,0,0}));
        const auto output=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes,nm_float4{99,99,99,0}));
        const auto statuses=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));const std::uint32_t include=0;
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];[encoder setBuffer:incidenceBuffer offset:0 atIndex:7];
        [encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:acceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:vascularBuffer offset:0 atIndex:11];
        [encoder setBuffer:borrowed offset:0 atIndex:12];[encoder setBytes:&include length:sizeof(include) atIndex:13];[encoder setBuffer:output offset:0 atIndex:14];[encoder setBuffer:statuses offset:0 atIndex:15];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:16];[encoder setBuffer:bindingBuffer offset:0 atIndex:17];[encoder setBuffer:elastanceBuffer offset:0 atIndex:18];
        [encoder setBuffer:cavityConnectionBuffer offset:0 atIndex:19];[encoder setBuffer:cavityAcceptedVascularBuffer offset:0 atIndex:20];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(statuses.contents)[env].code==NM_STATUS_SUCCESS,"blood owner force state rejected");
        const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);
    };
    const auto encodeMoments=[&](const std::vector<nm_float4>& state,
                                 const std::vector<NMFEMNodeStateGPU>& nodes){
        const auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:momentPipeline];
        const auto nodeBuffer=gpuBuffer(run.device,nodes),vascularBuffer=gpuBuffer(run.device,state);
        const auto output=gpuBuffer(run.device,std::vector<NMVascularTissueMomentGPU>(2*w.vascular.tissues.size()));
        const auto statuses=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];
        [encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];
        [encoder setBuffer:nodeBuffer offset:0 atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:4];[encoder setBuffer:bindingBuffer offset:0 atIndex:5];
        [encoder setBuffer:vascularBuffer offset:0 atIndex:6];[encoder setBuffer:output offset:0 atIndex:7];
        [encoder setBuffer:statuses offset:0 atIndex:8];
        [encoder dispatchThreads:MTLSizeMake(2*w.vascular.tissues.size(),1,1)
            threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(statuses.contents)[env].code==NM_STATUS_SUCCESS,"blood moment state rejected");
        const auto* values=static_cast<NMVascularTissueMomentGPU*>(output.contents);
        return std::vector<NMVascularTissueMomentGPU>(values,values+2*w.vascular.tissues.size());
    };
    const auto initialMoments=encodeMoments(vascular,accepted),initialReplay=encodeMoments(vascular,accepted);
    need(same(initialMoments,initialReplay),"blood spatial moment replay changed output");
    const auto& initialMoment=initialMoments[0];
    need(std::abs(double(initialMoment.firstMassMoment.x))<2e-9&&
         std::abs(double(initialMoment.firstMassMoment.y))<2e-9&&
         std::abs(double(initialMoment.firstMassMoment.z))<2e-9&&
         std::abs(double(initialMoment.firstMassMoment.w)-double(owner.physical.z))<2e-8,
         "initial blood first mass moment disagreed with the symmetric region");
    need(std::abs(double(initialMoment.secondMassMoment0.x)-double(owner.physical.z)*2.5e-5)<2e-12&&
         std::abs(double(initialMoment.secondMassMoment1.x)-double(owner.physical.z)*2.5e-5)<2e-12&&
         std::abs(double(initialMoment.secondMassMoment1.z)-double(owner.physical.z)*2.5e-5)<2e-12,
         "initial blood second mass moment disagreed with the reference spatial moment");
    const auto first=encode(vascular,accepted),second=encode(vascular,accepted);need(same(first,second),"blood owner correction replay changed output");
    const double volume=run.physical(run.state(),compartment),density=owner.physical.y,weight=1./double(shell.innerNodes.size());
    double total=0;for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){
        const auto& force=first[env*nodes+node];
        need(std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7&&std::abs(double(force.z))<2e-7,"initial blood mass was double-counted as an external force");
        total+=force.z;
    }
    need(std::abs(total)<4e-7,"initial blood mass correction was not zero");
    auto changed=vascular;
    const unsigned vascularStride=unsigned(w.vascular.unknowns.size());
    for (unsigned env=0; env<2; ++env) changed[env*vascularStride+compartment].x+=0.5f;
    const auto dynamic=encode(changed,accepted);const double deltaVolume=0.5*double(w.vascular.unknowns[compartment].initialAndScaling.y);
    const double expectedDelta=double(w.dispatch.gravityAndTimestep.z)*density*deltaVolume*weight;
    double dynamicTotal=0;
    for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){
        const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();
        const double expected=owned?expectedDelta:0.;
        const auto& force=dynamic[env*nodes+node];
        need(std::abs(double(force.z)-expected)<2e-6&&std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7,"dynamic blood mass correction disagreed with current volume");
        dynamicTotal+=force.z;
    }
    need(std::abs(dynamicTotal-2.*double(w.dispatch.gravityAndTimestep.z)*density*deltaVolume)<4e-6,"dynamic blood mass correction did not conserve total force");
    auto moving=accepted;
    const double imposedVelocity=0.003;
    for(unsigned env=0;env<2;++env) for(const auto node:shell.innerNodes)
        moving[env*nodes+node].velocityAndInverseMass.z+=float(imposedVelocity);
    const auto inertial=encode(changed,moving);
    const double acceleration=imposedVelocity/double(run.runtime.timestepSeconds());
    const double expectedInertial=(double(w.dispatch.gravityAndTimestep.z)+acceleration)*density*deltaVolume*weight;
    double inertialTotal=0;
    for(unsigned env=0;env<2;++env) for(unsigned node=0;node<nodes;++node){
        const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();
        const double expected=owned?expectedInertial:0.;
        const auto& force=inertial[env*nodes+node];
        need(std::abs(double(force.z)-expected)<2e-6&&std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7,"co-moving blood inertia correction disagreed with tissue acceleration");
        inertialTotal+=force.z;
    }
    need(std::abs(inertialTotal-2.*(double(w.dispatch.gravityAndTimestep.z)+acceleration)*density*deltaVolume)<4e-6,"co-moving blood inertia correction did not conserve total force");
    const auto dynamicMoments=encodeMoments(changed,moving);
    const double dynamicMass=density*(volume+deltaVolume);
    for(unsigned env=0;env<2;++env){
        fixture::Vec firstMoment{},second0{},second1{},velocity{};
        for(const auto& binding:w.vascular.tissueBindings){
            const double weight=binding.physical.x;const auto& node=moving[env*nodes+binding.identity.y];
            const fixture::Vec x{node.positionAndMass.x,node.positionAndMass.y,node.positionAndMass.z};
            const fixture::Vec v{node.velocityAndInverseMass.x,node.velocityAndInverseMass.y,node.velocityAndInverseMass.z};
            for(unsigned k=0;k<3;++k){firstMoment[k]+=weight*x[k];velocity[k]+=weight*v[k];}
            second0[0]+=weight*x[0]*x[0];second0[1]+=weight*x[0]*x[1];second0[2]+=weight*x[0]*x[2];
            second1[0]+=weight*x[1]*x[1];second1[1]+=weight*x[1]*x[2];second1[2]+=weight*x[2]*x[2];
        }
        const auto& actual=dynamicMoments[env];
        need(std::abs(double(actual.firstMassMoment.x)-dynamicMass*firstMoment[0])<3e-10&&
             std::abs(double(actual.firstMassMoment.y)-dynamicMass*firstMoment[1])<3e-10&&
             std::abs(double(actual.firstMassMoment.z)-dynamicMass*firstMoment[2])<3e-10&&
             std::abs(double(actual.firstMassMoment.w)-dynamicMass)<3e-8,
             "dynamic blood first mass moment disagreed with the moving FEM region");
        need(std::abs(double(actual.secondMassMoment0.x)-dynamicMass*second0[0])<3e-12&&
             std::abs(double(actual.secondMassMoment0.y)-dynamicMass*second0[1])<3e-12&&
             std::abs(double(actual.secondMassMoment0.z)-dynamicMass*second0[2])<3e-12&&
             std::abs(double(actual.secondMassMoment1.x)-dynamicMass*second1[0])<3e-12&&
             std::abs(double(actual.secondMassMoment1.y)-dynamicMass*second1[1])<3e-12&&
             std::abs(double(actual.secondMassMoment1.z)-dynamicMass*second1[2])<3e-12,
             "dynamic blood second mass moment disagreed with the moving FEM region");
        need(std::abs(double(actual.linearMomentum.x)-dynamicMass*velocity[0])<3e-8&&
             std::abs(double(actual.linearMomentum.y)-dynamicMass*velocity[1])<3e-8&&
             std::abs(double(actual.linearMomentum.z)-dynamicMass*velocity[2])<3e-8,
             "dynamic blood co-moving momentum disagreed with the moving FEM region");
    }
    // A bounded time-integrated closure uses the production moment kernel on
    // two consecutive co-moving configurations. With fixed current mass,
    // the finite difference of the first mass moment must equal the reported
    // co-moving linear momentum. This is a moment/transfer audit only; it
    // does not invent a fluid pressure-gradient state.
    auto movingNext=moving;
    const double dt=double(run.runtime.timestepSeconds());
    for(unsigned env=0;env<2;++env) for(const auto node:shell.innerNodes){
        auto& position=movingNext[env*nodes+node].positionAndMass;
        const auto& velocityState=moving[env*nodes+node].velocityAndInverseMass;
        position.x+=float(dt*double(velocityState.x));
        position.y+=float(dt*double(velocityState.y));
        position.z+=float(dt*double(velocityState.z));
    }
    const auto nextMoments=encodeMoments(changed,movingNext);
    double momentDerivativeError=0,momentDerivativeScale=0;
    for(unsigned env=0;env<2;++env){
        const auto& beforeMoment=dynamicMoments[env];const auto& afterMoment=nextMoments[env];
        for(unsigned component=0;component<3;++component){
            const float* before=&beforeMoment.firstMassMoment.x;
            const float* after=&afterMoment.firstMassMoment.x;
            const float* momentum=&beforeMoment.linearMomentum.x;
            const double derivative=(double(after[component])-double(before[component]))/dt;
            momentDerivativeError=std::max(momentDerivativeError,std::abs(derivative-double(momentum[component])));
            momentDerivativeScale=std::max(momentDerivativeScale,std::abs(double(momentum[component])));
        }
        need(std::memcmp(&beforeMoment.firstMassMoment.w,&afterMoment.firstMassMoment.w,sizeof(float))==0,
             "time-integrated blood moment changed mass without a volume change");
    }
    need(momentDerivativeScale>1e-12&&momentDerivativeError/momentDerivativeScale<3e-4,
         "time-integrated blood first-moment derivative disagreed with co-moving momentum");
    // Restore the coupled FEM velocity and vascular state as one accepted
    // transaction, then replay the owner moments. This proves that a
    // checkpoint cannot restore one side of the blood momentum state while
    // leaving the other side at a different continuation point.
    auto movingSnapshot=run.state();
    for(unsigned env=0;env<2;++env) for(const auto node:shell.innerNodes)
        movingSnapshot.femNodes[env*nodes+node].velocityAndInverseMass.z+=float(imposedVelocity);
    const auto expectedRestoredMoments=encodeMoments(movingSnapshot.vascularState,movingSnapshot.femNodes);
    need(run.runtime.restore(movingSnapshot).encoded,"coupled blood momentum restore was rejected");
    const auto restoredMomentum=run.state();
    need(sameBloodMomentumState(restoredMomentum,movingSnapshot),"coupled blood momentum restore was not atomic");
    need(same(encodeMoments(restoredMomentum.vascularState,restoredMomentum.femNodes),expectedRestoredMoments),
         "coupled blood momentum moments changed across restore");
    const auto restoredCheckpoint=restoredMomentum;
    need(run.runtime.restore(initialSnapshot).encoded,"blood momentum rewind failed");
    need(run.runtime.restore(restoredCheckpoint).encoded,"blood momentum replay restore failed");
    const auto replayedMomentum=run.state();
    need(sameBloodMomentumState(replayedMomentum,restoredCheckpoint),"coupled blood momentum replay changed state");
    auto invalidMomentum=restoredCheckpoint;
    invalidMomentum.vascularState[0].x=std::numeric_limits<float>::quiet_NaN();
    const auto deniedMomentum=run.runtime.restore(invalidMomentum);
    need(!deniedMomentum.encoded&&sameBloodMomentumState(run.state(),replayedMomentum),
         "invalid coupled blood momentum restore changed accepted state");
    need(run.runtime.restore(initialSnapshot).encoded,"blood momentum final rewind failed");
    // Uniform cavity pressure has no net force on a closed surface. Remove
    // the separately known current-volume mass correction and retain the
    // residual pressure impulse as an explicit closed-surface audit.
    const auto pressureForces=dynamic;
    fixture::Vec pressureTotal{0,0,0};
    for(unsigned env=0;env<2;++env) for(unsigned node=0;node<nodes;++node){
        const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();
        const double correction=owned?double(w.dispatch.gravityAndTimestep.z)*density*deltaVolume*weight:0.;
        const auto& force=pressureForces[env*nodes+node];
        pressureTotal[0]+=double(force.x);pressureTotal[1]+=double(force.y);
        pressureTotal[2]+=double(force.z)-correction;
    }
    const double pressureImpulseNorm=std::sqrt(fixture::dot(pressureTotal,pressureTotal))*dt;
    need(pressureImpulseNorm<2e-9,"uniform closed-surface pressure impulse did not cancel");
    std::cout<<"blood_mass_owner=pass compartment_stable_id=12 density_kg_m3="<<density<<" initial_volume_m3="<<volume
             <<" initial_mass_kg="<<owner.physical.z<<" dynamic_volume_delta_m3="<<deltaVolume
             <<" dynamic_force_total_N="<<dynamicTotal<<" normalized_region_weight="<<weight
             <<" partitioned_inertia=pass dynamic_gravity_correction=pass dynamic_spatial_moments=pass co_moving_inertia=pass"
             <<" replay=bitwise time_integrated_moment_closure=pass pressure_impulse_closure=pass"
             <<" atomic_full_momentum_restore=pass pressure_driven_momentum=unqualified subject_calibration=unqualified\n";
}
void pressureMomentumOwner(){
    const auto shell=fixture::hollowShell();Run run(fixture::world(false,.001,true,true));const auto& w=run.world;
    need(w.vascular.tissues.size()==1,"pressure reaction fixture did not cook its FEM region");
    const auto& owner=w.vascular.tissues[0];need(owner.region.z!=0u&&owner.region.w!=0u&&owner.physical.w>0.f&&
        (owner.identity.w & NM_VASCULAR_TISSUE_MOMENTUM_TRANSFER)!=0u,"pressure/momentum reaction contract was not cooked");
    need(std::abs(double(owner.spatialFirst.w))<1e-7&&std::abs(double(owner.spatialSecond0.w))<1e-7&&std::abs(double(owner.spatialSecond1.w)-1.)<1e-7,"pressure reaction direction was not preserved");
    const auto accepted=run.state().femNodes;const auto vascular=run.state().vascularState;const unsigned nodes=w.dispatch.femNodeCount;
    NSError* error=nil;auto library=[run.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]] error:&error];
    need(library!=nil,"metallib unavailable for pressure reaction check");
    const auto forcePipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_cavity_forces");
    const auto operatorPipeline=pipeline(run.device,library,@"numi_matter_metal::nm_vascular_cavity_operator");
    NMMicrostepGPU micro{};micro.time={float(run.runtime.timestepSeconds()),float(1./run.runtime.timestepSeconds()),0,0};micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;
    const auto unknownBuffer=gpuBuffer(run.device,w.vascular.unknowns),compartmentBuffer=gpuBuffer(run.device,w.vascular.compartments);
    const auto cavityBuffer=gpuBufferOrDummy(run.device,w.vascular.cavities),faceBuffer=gpuBufferOrDummy(run.device,w.vascular.cavityFaces);
    const auto incidenceBuffer=gpuBufferOrDummy(run.device,w.vascular.cavityNodeIncidence),rangeBuffer=gpuBufferOrDummy(run.device,w.vascular.cavityNodeRanges);
    auto tissueBuffer=gpuBuffer(run.device,w.vascular.tissues);const auto bindingBuffer=gpuBuffer(run.device,w.vascular.tissueBindings);
    const auto connectionBuffer=gpuBuffer(run.device,w.vascular.connections),femAcceptedBuffer=gpuBuffer(run.device,accepted);
    std::vector<float> fixtureElastance(w.dispatch.environmentCount*w.vascular.compartments.size(),1.f);
    for(unsigned env=0;env<w.dispatch.environmentCount;++env) for(unsigned i=0;i<w.vascular.compartments.size();++i)
        if(w.vascular.compartments[i].identity.w==0u||w.vascular.compartments[i].identity.w==3u) fixtureElastance[env*w.vascular.compartments.size()+i]=float(1./w.vascular.compartments[i].compliance.z);
    const auto elastanceBuffer=gpuBufferOrDummy(run.device,fixtureElastance);
    auto pressureLayout=w.vascular.layout;
    std::vector<NMFEMNodeStateGPU> forceCandidateNodes=accepted;
    auto encodeForce=[&](const std::vector<nm_float4>& candidateState,const std::vector<nm_float4>& acceptedState){
        const auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:forcePipeline];
        const auto stateBuffer=gpuBuffer(run.device,candidateState),acceptedVascularBuffer=gpuBuffer(run.device,acceptedState),candidateBuffer=gpuBuffer(run.device,forceCandidateNodes),borrowed=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes));
        const auto output=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes,nm_float4{99,99,99,0})),statuses=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));const std::uint32_t include=0;
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&pressureLayout length:sizeof(pressureLayout) atIndex:1];[encoder setBytes:&micro length:sizeof(micro) atIndex:2];
        [encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];[encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];
        [encoder setBuffer:incidenceBuffer offset:0 atIndex:7];[encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:femAcceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:stateBuffer offset:0 atIndex:11];
        [encoder setBuffer:borrowed offset:0 atIndex:12];[encoder setBytes:&include length:sizeof(include) atIndex:13];[encoder setBuffer:output offset:0 atIndex:14];[encoder setBuffer:statuses offset:0 atIndex:15];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:16];[encoder setBuffer:bindingBuffer offset:0 atIndex:17];[encoder setBuffer:elastanceBuffer offset:0 atIndex:18];[encoder setBuffer:connectionBuffer offset:0 atIndex:19];[encoder setBuffer:acceptedVascularBuffer offset:0 atIndex:20];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(statuses.contents)[env].code==NM_STATUS_SUCCESS,"pressure/momentum reaction force state rejected");
        const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);
    };
    const auto first=encodeForce(vascular,vascular),second=encodeForce(vascular,vascular);need(same(first,second),"pressure/momentum reaction replay changed output");
    const std::uint32_t baseMicroFlags=micro.flags;
    auto forceAtPressureScale=[&](float scale,const std::vector<nm_float4>& candidateState,
                                  const std::vector<nm_float4>& acceptedState){
        micro.flags=baseMicroFlags|NM_MICROSTEP_VASCULAR_PRESSURE_LOAD_SCALE;
        micro.initializationLoadScales={scale,0.0f,0.0f,0.0f};
        return encodeForce(candidateState,acceptedState);
    };
    const auto pressureForce0=forceAtPressureScale(0.0f,vascular,vascular);
    const auto pressureForceHalf=forceAtPressureScale(0.5f,vascular,vascular);
    const auto pressureForce1=forceAtPressureScale(1.0f,vascular,vascular);
    double pressureScaleError=0.0,pressureScaleMagnitude=0.0;
    for(std::size_t i=0;i<pressureForce1.size();++i)for(unsigned axis=0;axis<3;++axis){
        const auto component=[&](const nm_float4& value){return axis==0?value.x:axis==1?value.y:value.z;};
        const double f0=component(pressureForce0[i]),fh=component(pressureForceHalf[i]),f1=component(pressureForce1[i]);
        pressureScaleError=std::max(pressureScaleError,std::abs(fh-.5*(f0+f1)));
        pressureScaleMagnitude=std::max(pressureScaleMagnitude,std::abs(f1-f0));
    }
    need(same(first,pressureForce1),"pressure scale 1 changed the established mechanical load");
    need(pressureScaleMagnitude>1e-6&&pressureScaleError<2e-6*pressureScaleMagnitude,
         "mechanical pressure force did not scale linearly at 0/0.5/1");
    micro.flags=baseMicroFlags;micro.initializationLoadScales={};
    unsigned from=NM_INVALID_INDEX,to=NM_INVALID_INDEX;for(unsigned i=0;i<w.vascular.compartments.size();++i){if(w.vascular.compartments[i].identity.x==12u)from=i;if(w.vascular.compartments[i].identity.x==11u)to=i;}
    need(from!=NM_INVALID_INDEX&&to!=NM_INVALID_INDEX,"pressure reaction compartments were not resolved");
    const double volume=run.physical(run.state(),from);const auto& source=w.vascular.compartments[from];const double pFrom=source.compliance.y+(volume-source.compliance.x)/source.compliance.z;const double pTo=run.physical(run.state(),w.vascular.cavities[0].identity.w);const double totalReaction=-(pFrom-pTo)*double(owner.physical.w),weight=1./double(shell.innerNodes.size());
    double observedTotal=0;for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();const auto& force=first[env*nodes+node];const double expected=owned?totalReaction*weight:0.;need(std::abs(double(force.z)-expected)<2e-7&&std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7,"pressure reaction force disagreed with the authored pressure-area direction");observedTotal+=force.z;}
    need(std::abs(observedTotal-2.*totalReaction)<2e-6,"pressure reaction did not conserve the registered wall force");
    const unsigned flowRow=w.vascular.layout.offsets.y;const double flowDelta=double(w.vascular.unknowns[flowRow].initialAndScaling.y);
    auto momentumCandidate=vascular;for(unsigned env=0;env<2;++env) momentumCandidate[env*w.vascular.layout.ranges.z+flowRow].x+=1.f;
    const auto momentumForces=encodeForce(momentumCandidate,vascular);
    const auto momentum0=forceAtPressureScale(0.0f,momentumCandidate,vascular);
    const auto momentumBase0=forceAtPressureScale(0.0f,vascular,vascular);
    const auto momentumHalf=forceAtPressureScale(0.5f,momentumCandidate,vascular);
    const auto momentumBaseHalf=forceAtPressureScale(0.5f,vascular,vascular);
    const auto momentum1=forceAtPressureScale(1.0f,momentumCandidate,vascular);
    const auto momentumBase1=forceAtPressureScale(1.0f,vascular,vascular);
    double momentumScaleError=0.0,momentumScaleMagnitude=0.0;
    for(std::size_t i=0;i<momentumForces.size();++i)for(unsigned axis=0;axis<3;++axis){
        const auto component=[&](const nm_float4& value){return axis==0?value.x:axis==1?value.y:value.z;};
        const double d0=double(component(momentum0[i]))-component(momentumBase0[i]);
        const double dh=double(component(momentumHalf[i]))-component(momentumBaseHalf[i]);
        const double d1=double(component(momentum1[i]))-component(momentumBase1[i]);
        momentumScaleError=std::max(momentumScaleError,std::max(std::abs(dh-d0),std::abs(d1-d0)));
        momentumScaleMagnitude=std::max(momentumScaleMagnitude,std::abs(d0));
    }
    need(momentumScaleMagnitude>1e-6&&momentumScaleError<2e-6*momentumScaleMagnitude,
         "vascular momentum reaction changed under the pressure-only scale");
    micro.flags=baseMicroFlags;micro.initializationLoadScales={};
    const double momentumReaction=-double(owner.physical.y)*double(owner.physical.x)/double(owner.physical.w)*flowDelta/double(micro.time.x);
    const double combinedReaction=totalReaction+momentumReaction;double momentumObserved=0;
    for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();const auto& force=momentumForces[env*nodes+node];const double expected=owned?combinedReaction*weight:0.;need(std::abs(double(force.z)-expected)<3e-7&&std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7,"fluid momentum transfer force disagreed with the finite-volume impulse");momentumObserved+=force.z;}
    need(std::abs(momentumObserved-2.*combinedReaction)<3e-6&&std::abs(momentumReaction)>1e-6,"fluid momentum transfer was not exercised or conserved");
    NMFGMRESLayoutGPU layout{};layout.vascularUnknownCount=unsigned(w.vascular.unknowns.size());layout.vascularBase=4*nodes;layout.unknownCount=layout.vascularBase+2*layout.vascularUnknownCount;
    auto direction=std::vector<nm_float4>(layout.unknownCount);for(unsigned env=0;env<2;++env){direction[layout.vascularBase+env*w.vascular.unknowns.size()+from].x=.5f;direction[layout.vascularBase+env*w.vascular.unknowns.size()+flowRow].x=.25f;}
    const auto acceptedBuffer=gpuBuffer(run.device,accepted);
    std::vector<NMFEMNodeStateGPU> operatorCandidateNodes=accepted;
    std::vector<nm_float4> operatorCandidateVascularState=vascular;
    auto encodeOperator=[&](const std::vector<nm_float4>& actionDirection){const auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:operatorPipeline];
        const auto candidateBuffer=gpuBuffer(run.device,operatorCandidateNodes);
        const auto stateBuffer=gpuBuffer(run.device,operatorCandidateVascularState);
        const auto acceptedVascularBuffer=gpuBuffer(run.device,vascular);
        const auto output=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes)),solver=gpuBuffer(run.device,std::vector<NMFGMRESStateGPU>(2)),statuses=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));const auto actionDirectionBuffer=gpuBuffer(run.device,actionDirection);
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&pressureLayout length:sizeof(pressureLayout) atIndex:1];[encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];[encoder setBuffer:incidenceBuffer offset:0 atIndex:7];[encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:acceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:stateBuffer offset:0 atIndex:11];[encoder setBuffer:actionDirectionBuffer offset:0 atIndex:12];[encoder setBuffer:output offset:0 atIndex:13];[encoder setBuffer:solver offset:0 atIndex:14];[encoder setBuffer:statuses offset:0 atIndex:15];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:16];[encoder setBuffer:bindingBuffer offset:0 atIndex:17];[encoder setBuffer:elastanceBuffer offset:0 atIndex:18];[encoder setBuffer:connectionBuffer offset:0 atIndex:19];[encoder setBuffer:acceptedVascularBuffer offset:0 atIndex:20];[encoder setBytes:&layout length:sizeof(layout) atIndex:30];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(statuses.contents)[env].code==NM_STATUS_SUCCESS,"pressure reaction tangent state rejected");const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);};
    const auto firstOperator=encodeOperator(direction),secondOperator=encodeOperator(direction);need(same(firstOperator,secondOperator),"pressure reaction tangent replay changed output");
    const double pressureDelta=.5*double(w.vascular.unknowns[from].initialAndScaling.y)/source.compliance.z;const double tangentTotal=micro.time.x*pressureDelta*double(owner.physical.w);const double momentumTangentTotal=double(owner.physical.y)*double(owner.physical.x)/double(owner.physical.w)*.25*double(w.vascular.unknowns[flowRow].initialAndScaling.y);const double combinedTangentTotal=tangentTotal+momentumTangentTotal;for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();const double expected=owned?combinedTangentTotal*weight:0.;const auto& value=firstOperator[env*nodes+node];need(std::abs(double(value.z)-expected)<4e-7&&std::abs(double(value.x))<3e-7&&std::abs(double(value.y))<3e-7,"pressure/momentum reaction tangent disagreed with the coupled derivative");}
    std::vector<nm_float4> flowOnly(layout.unknownCount);for(unsigned env=0;env<2;++env)
        flowOnly[layout.vascularBase+env*w.vascular.layout.ranges.z+flowRow].x=.25f;
    micro.flags=baseMicroFlags|NM_MICROSTEP_VASCULAR_PRESSURE_LOAD_SCALE;micro.initializationLoadScales={0.0f,0,0,0};
    const auto j0=encodeOperator(direction),flowJ0=encodeOperator(flowOnly);
    micro.initializationLoadScales.x=.5f;const auto jHalf=encodeOperator(direction),flowJHalf=encodeOperator(flowOnly);
    micro.initializationLoadScales.x=1.0f;const auto j1=encodeOperator(direction),flowJ1=encodeOperator(flowOnly);
    double tangentScaleError=0.0,tangentScaleMagnitude=0.0;
    for(std::size_t i=0;i<j1.size();++i)for(unsigned axis=0;axis<3;++axis){
        const auto component=[&](const nm_float4& value){return axis==0?value.x:axis==1?value.y:value.z;};
        const double a=component(j0[i]),b=component(jHalf[i]),c=component(j1[i]);
        tangentScaleError=std::max(tangentScaleError,std::abs(b-.5*(a+c)));
        tangentScaleMagnitude=std::max(tangentScaleMagnitude,std::abs(c-a));
    }
    need(tangentScaleMagnitude>1e-7&&tangentScaleError<3e-5*tangentScaleMagnitude,
         "mechanical pressure JVP did not scale linearly at 0/0.5/1");
    need(same(flowJ0,flowJHalf)&&same(flowJ0,flowJ1),
         "flow-momentum JVP changed under the pressure-only scale");
    constexpr double pressureFDStep=1.0/64.0;
    auto pressurePlus=vascular,pressureMinus=vascular;
    for(unsigned env=0;env<2;++env){
        const unsigned base=env*w.vascular.layout.ranges.z;
        pressurePlus[base+from].x+=float(pressureFDStep*direction[layout.vascularBase+base+from].x);
        pressureMinus[base+from].x-=float(pressureFDStep*direction[layout.vascularBase+base+from].x);
        pressurePlus[base+flowRow].x+=float(pressureFDStep*direction[layout.vascularBase+base+flowRow].x);
        pressureMinus[base+flowRow].x-=float(pressureFDStep*direction[layout.vascularBase+base+flowRow].x);
    }
    micro.initializationLoadScales.x=.5f;
    const auto forceHalfPlus=encodeForce(pressurePlus,vascular),
        forceHalfMinus=encodeForce(pressureMinus,vascular);
    const auto forceZeroPlus=forceAtPressureScale(0.0f,pressurePlus,vascular),
        forceZeroMinus=forceAtPressureScale(0.0f,pressureMinus,vascular);
    double pressureJvpError=0.0,pressureJvpMagnitude=0.0;
    double worstFD=0.0,worstActual=0.0;std::size_t worstIndex=0;unsigned worstAxis=0;
    for(std::size_t i=0;i<jHalf.size();++i)for(unsigned axis=0;axis<3;++axis){
        const auto component=[&](const nm_float4& value){return axis==0?value.x:axis==1?value.y:value.z;};
        // Isolate only the pressure-scaled force owner. The complete force
        // also contains the scale-invariant blood-mass correction and flow
        // momentum transfer, each with a distinct Jacobian owner.
        const double scaledForceDifference =
            (double(component(forceHalfPlus[i]))-component(forceHalfMinus[i]))-
            (double(component(forceZeroPlus[i]))-component(forceZeroMinus[i]));
        const double fd=-double(micro.time.x)*scaledForceDifference/(2.0*pressureFDStep);
        const double actual=double(component(jHalf[i]))-component(j0[i]);
        const double error=std::abs(actual-fd);
        if(error>pressureJvpError){pressureJvpError=error;worstFD=fd;worstActual=actual;worstIndex=i;worstAxis=axis;}
        pressureJvpMagnitude=std::max(pressureJvpMagnitude,std::abs(fd));
    }
    const double pressureJvpTolerance=4e-3*pressureJvpMagnitude;
    std::fprintf(stderr,
        "pressure_jvp_fd_diagnostic=error:%g tolerance:%g magnitude:%g worst_index:%zu axis:%u fd:%g operator_delta:%g\n",
        pressureJvpError,pressureJvpTolerance,pressureJvpMagnitude,worstIndex,worstAxis,worstFD,worstActual);
    need(pressureJvpMagnitude>1e-7&&pressureJvpError<pressureJvpTolerance,
         "scaled production pressure JVP disagreed with finite differences");
    auto noPressureTissues=w.vascular.tissues;
    for(auto& tissue:noPressureTissues){tissue.region.z=0u;tissue.region.w=0u;}
    tissueBuffer=gpuBufferOrDummy(run.device,noPressureTissues);
    const auto savedCavityCount=pressureLayout.cavities.x;pressureLayout.cavities.x=0u;
    const auto ownerCode=owner.identity.w&~NM_VASCULAR_TISSUE_MOMENTUM_TRANSFER;
    need(ownerCode>0u&&ownerCode-1u<w.vascular.layout.counts.x,"blood-mass fixture owner compartment is invalid");
    auto massCandidate=vascular;const unsigned massRow=ownerCode-1u;
    for(unsigned env=0;env<2;++env)massCandidate[env*w.vascular.layout.ranges.z+massRow].x+=1.0e-3f;
    const auto mass0=forceAtPressureScale(0.0f,massCandidate,vascular);
    const auto massHalf=forceAtPressureScale(.5f,massCandidate,vascular);
    const auto mass1=forceAtPressureScale(1.0f,massCandidate,vascular);
    need(same(mass0,massHalf)&&same(mass0,mass1),
         "blood-mass correction changed under mechanical pressure scale");
    double massForceMagnitude=0.0;for(const auto& value:mass0)
        massForceMagnitude=std::max(massForceMagnitude,std::sqrt(double(value.x)*value.x+double(value.y)*value.y+double(value.z)*value.z));
    need(massForceMagnitude>1e-6,"blood-mass scale invariance test did not exercise a nonzero force");
    // Exercise the exact Jacobian owner for both volume-dependent gravity
    // transfer and co-moving acceleration. Keep pressure traction disabled so
    // the finite difference isolates the blood-mass correction.
    auto acceleratedCandidate=accepted;
    const fixture::Vec baseAcceleration{0.375,-0.25,0.125};
    for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){
        auto& velocity=acceleratedCandidate[env*nodes+node].velocityAndInverseMass;
        velocity.x+=float(micro.time.x*baseAcceleration[0]);
        velocity.y+=float(micro.time.x*baseAcceleration[1]);
        velocity.z+=float(micro.time.x*baseAcceleration[2]);
    }
    forceCandidateNodes=acceleratedCandidate;
    operatorCandidateNodes=acceleratedCandidate;
    operatorCandidateVascularState=massCandidate;
    micro.flags=baseMicroFlags|NM_MICROSTEP_VASCULAR_PRESSURE_LOAD_SCALE|
        NM_MICROSTEP_VASCULAR_BLOOD_MASS_TANGENT;
    micro.initializationLoadScales={0.0f,0.0f,0.0f,0.0f};
    constexpr double bloodFDStep=1.0/64.0;
    std::vector<nm_float4> bloodVolumeDirection(layout.unknownCount);
    constexpr float normalizedVolumeDirection=.375f;
    for(unsigned env=0;env<2;++env)
        bloodVolumeDirection[layout.vascularBase+env*w.vascular.layout.ranges.z+massRow].x=
            normalizedVolumeDirection;
    const auto bloodVolumeJVP=encodeOperator(bloodVolumeDirection);
    auto bloodVolumePlus=massCandidate,bloodVolumeMinus=massCandidate;
    for(unsigned env=0;env<2;++env){
        const unsigned base=env*w.vascular.layout.ranges.z+massRow;
        bloodVolumePlus[base].x+=float(bloodFDStep*normalizedVolumeDirection);
        bloodVolumeMinus[base].x-=float(bloodFDStep*normalizedVolumeDirection);
    }
    const auto bloodVolumeForcePlus=forceAtPressureScale(0.0f,bloodVolumePlus,vascular);
    const auto bloodVolumeForceMinus=forceAtPressureScale(0.0f,bloodVolumeMinus,vascular);
    double bloodVolumeError=0.0,bloodVolumeMagnitude=0.0;
    for(std::size_t i=0;i<bloodVolumeJVP.size();++i){
        const auto& node=accepted[i];
        if((node.restAndFixed.w>0.5f&&node.restAndFixed.w<1.5f)||
            !(node.positionAndMass.w>0.0f)) continue;
        for(unsigned axis=0;axis<3;++axis){
            const auto component=[&](const nm_float4& value){return axis==0?value.x:axis==1?value.y:value.z;};
            const double fd=-double(micro.time.x)*
                (double(component(bloodVolumeForcePlus[i]))-component(bloodVolumeForceMinus[i]))/
                (2.0*bloodFDStep);
            const double actual=component(bloodVolumeJVP[i]);
            bloodVolumeError=std::max(bloodVolumeError,std::abs(actual-fd));
            bloodVolumeMagnitude=std::max(bloodVolumeMagnitude,std::abs(fd));
        }
    }
    const double bloodVolumeTolerance=5e-3*bloodVolumeMagnitude;
    std::fprintf(stderr,
        "blood_mass_volume_jvp=error:%g tolerance:%g magnitude:%g\n",
        bloodVolumeError,bloodVolumeTolerance,bloodVolumeMagnitude);
    need(bloodVolumeMagnitude>1e-7&&bloodVolumeError<bloodVolumeTolerance,
         "blood-mass volume/gravity JVP disagreed with full-force finite differences");

    std::vector<nm_float4> translationDirection(layout.unknownCount);
    // A unit-scale direction keeps the known nonzero mass response above the fixture floor.
    const fixture::Vec translation{2.0,-1.0,.5};
    for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){
        auto& value=translationDirection[env*nodes+node];
        value.x=float(translation[0]);value.y=float(translation[1]);value.z=float(translation[2]);
    }
    // forceAtPressureScale updates `micro.flags` for the residual-only calls
    // above. Restore the blood-mass tangent bit before evaluating this JVP.
    // Otherwise the finite-difference path exercises the co-moving force
    // while the production operator silently omits its Jacobian term.
    micro.flags=baseMicroFlags|NM_MICROSTEP_VASCULAR_PRESSURE_LOAD_SCALE|
        NM_MICROSTEP_VASCULAR_BLOOD_MASS_TANGENT;
    micro.initializationLoadScales={0.0f,0.0f,0.0f,0.0f};
    const auto bloodVelocityJVP=encodeOperator(translationDirection);
    constexpr double velocityFDStep=1.0/64.0;
    auto velocityPlus=acceleratedCandidate,velocityMinus=acceleratedCandidate;
    for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){
        auto& plus=velocityPlus[env*nodes+node].velocityAndInverseMass;
        auto& minus=velocityMinus[env*nodes+node].velocityAndInverseMass;
        plus.x+=float(velocityFDStep*translation[0]);minus.x-=float(velocityFDStep*translation[0]);
        plus.y+=float(velocityFDStep*translation[1]);minus.y-=float(velocityFDStep*translation[1]);
        plus.z+=float(velocityFDStep*translation[2]);minus.z-=float(velocityFDStep*translation[2]);
    }
    forceCandidateNodes=velocityPlus;
    const auto bloodVelocityForcePlus=forceAtPressureScale(0.0f,massCandidate,vascular);
    forceCandidateNodes=velocityMinus;
    const auto bloodVelocityForceMinus=forceAtPressureScale(0.0f,massCandidate,vascular);
    double bloodVelocityError=0.0,bloodVelocityMagnitude=0.0;
    for(std::size_t i=0;i<bloodVelocityJVP.size();++i){
        const auto& node=accepted[i];
        if((node.restAndFixed.w>0.5f&&node.restAndFixed.w<1.5f)||
            !(node.positionAndMass.w>0.0f)) continue;
        for(unsigned axis=0;axis<3;++axis){
            const auto component=[&](const nm_float4& value){return axis==0?value.x:axis==1?value.y:value.z;};
            const double fd=-double(micro.time.x)*
                (double(component(bloodVelocityForcePlus[i]))-component(bloodVelocityForceMinus[i]))/
                (2.0*velocityFDStep);
            const double actual=component(bloodVelocityJVP[i]);
            bloodVelocityError=std::max(bloodVelocityError,std::abs(actual-fd));
            bloodVelocityMagnitude=std::max(bloodVelocityMagnitude,std::abs(fd));
        }
    }
    const double bloodVelocityTolerance=5e-3*bloodVelocityMagnitude;
    std::fprintf(stderr,
        "blood_mass_velocity_jvp=error:%g tolerance:%g magnitude:%g\n",
        bloodVelocityError,bloodVelocityTolerance,bloodVelocityMagnitude);
    need(bloodVelocityMagnitude>1e-7&&bloodVelocityError<bloodVelocityTolerance,
         "blood-mass co-moving velocity JVP disagreed with full-force finite differences");
    forceCandidateNodes=accepted;
    operatorCandidateNodes=accepted;
    operatorCandidateVascularState=vascular;
    pressureLayout.cavities.x=savedCavityCount;tissueBuffer=gpuBuffer(run.device,w.vascular.tissues);
    micro.flags=baseMicroFlags;micro.initializationLoadScales={};
    std::cout<<"pressure_gradient_reaction=pass from_compartment=12 to_compartment=11 area_m2="<<owner.physical.w<<" total_force_per_environment_N="<<totalReaction<<" momentum_force_per_environment_N="<<momentumReaction<<" replay=bitwise jacobian=pass pressure_driven_fluid_momentum=pass anatomical_registration=unqualified subject_calibration=unqualified\n";
}

void cavitySchurBlockPreconditioner() {
    ScopedVascularPolicy vascularPolicy("0", "0");
    ScopedEnvironment schurPolicy("NUMI_MATTER_VASCULAR_BLOCK_CAVITY_SCHUR", "0");
    const auto source = fixture::world();
    auto compiled = compileWorld(source, {.maximumRateExponent=0});
    need(compiled.succeeded(), "cavity-Schur fixture compile failed");
    const auto& w = compiled.world;
    auto device = MTLCreateSystemDefaultDevice();
    need(device != nil && [[device name] rangeOfString:@"Apple"].location != NSNotFound &&
        [[device name] rangeOfString:@"Paravirtual"].location == NSNotFound,
        "physical Apple Metal required");
    auto queue = [device newCommandQueue];
    const unsigned envs = w.dispatch.environmentCount, nodes = w.dispatch.femNodeCount;
    const unsigned cavityCount = w.vascular.layout.cavities.x;
    need(envs == 2u && nodes == 64u && cavityCount != 0u,
        "cavity-Schur block fixture layout differs");
    NSError* error = nil;
    auto library = [device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]]
        error:&error];
    need(library != nil, "cavity-Schur metallib unavailable");
    const auto scalarPSO = pipeline(device, library,
        @"numi_matter_metal::nm_vascular_cavity_schur_preconditioner");
    const auto blockPSO = pipeline(device, library,
        @"numi_matter_metal::nm_vascular_cavity_schur_preconditioner_coupled_fem_block");
    std::vector<NMFEMNodeStateGPU> accepted;
    for (unsigned e = 0u; e < envs; ++e)
        accepted.insert(accepted.end(), w.fem.nodes.begin(), w.fem.nodes.end());
    auto candidate = accepted;
    for (unsigned e = 0; e < envs; ++e) for (unsigned n = 0; n < nodes; ++n) {
        auto& p = candidate[e * nodes + n].positionAndMass;
        const double phase = double(n + 1u) * (0.37 + 0.11 * e);
        p.x += float(1.1e-4 * std::sin(phase));
        p.y += float(0.9e-4 * std::cos(1.3 * phase));
        p.z += float(0.7e-4 * std::sin(0.71 * phase + 0.2));
    }
    const double dt = source.frameTimestep;
    NMMicrostepGPU micro{};
    micro.time = {float(dt), float(1.0 / dt), 0.0f, 0.0f};
    const auto dispatchB = gpuBuffer(device,
        std::vector<NMMatterDispatchGPU>{w.dispatch});
    const auto layoutB = gpuBuffer(device,
        std::vector<NMVascularLayoutGPU>{w.vascular.layout});
    const auto microB = gpuBuffer(device, std::vector<NMMicrostepGPU>{micro});
    const auto unknownB = gpuBuffer(device, w.vascular.unknowns);
    const auto cavityB = gpuBuffer(device, w.vascular.cavities);
    const auto faceB = gpuBuffer(device, w.vascular.cavityFaces);
    const auto acceptedB = gpuBuffer(device, accepted);
    const auto candidateB = gpuBuffer(device, candidate);
    const std::size_t nodeTotal = std::size_t(envs) * nodes;
    constexpr float scalar = 0.8f;
    std::vector<nm_float4> diagonal(nodeTotal, nm_float4{scalar, scalar, scalar, 0});
    std::vector<nm_float4> isotropic(nodeTotal * 3u), anisotropic(nodeTotal * 3u),
        invalid(nodeTotal * 3u);
    const double M[3][3] = {{2.0, 0.40, -0.10}, {0.40, 1.30, 0.25},
                            {-0.10, 0.25, 0.90}};
    for (std::size_t node = 0; node < nodeTotal; ++node) {
        const std::size_t b = node * 3u;
        isotropic[b] = {scalar, 0, 0, 1};
        isotropic[b + 1u] = {0, scalar, 0, 0};
        isotropic[b + 2u] = {0, 0, scalar, 0};
        for (unsigned row = 0; row < 3; ++row)
            anisotropic[b + row] = {float(M[row][0]), float(M[row][1]),
                float(M[row][2]), row == 0u ? 1.0f : 0.0f};
        const float nan = std::numeric_limits<float>::quiet_NaN();
        invalid[b] = {nan, nan, nan, 0};
        invalid[b + 1u] = {nan, nan, nan, 0};
        invalid[b + 2u] = {nan, nan, nan, 0};
    }
    const auto execute = [&](id<MTLComputePipelineState> pso,
        const std::vector<nm_float4>& diag, const std::vector<nm_float4>* blocks) {
        const auto diagB = gpuBuffer(device, diag);
        const auto blockB = gpuBuffer(device,
            blocks ? *blocks : std::vector<nm_float4>(nodeTotal * 3u));
        const auto outputB = gpuBuffer(device,
            std::vector<float>(std::size_t(envs) * cavityCount, -1.0f));
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:dispatchB offset:0 atIndex:0];
        [encoder setBuffer:layoutB offset:0 atIndex:1];
        [encoder setBuffer:microB offset:0 atIndex:2];
        [encoder setBuffer:unknownB offset:0 atIndex:3];
        [encoder setBuffer:cavityB offset:0 atIndex:4];
        [encoder setBuffer:faceB offset:0 atIndex:5];
        [encoder setBuffer:acceptedB offset:0 atIndex:6];
        [encoder setBuffer:candidateB offset:0 atIndex:7];
        [encoder setBuffer:diagB offset:0 atIndex:8];
        if (blocks) {
            [encoder setBuffer:blockB offset:0 atIndex:9];
            [encoder setBuffer:outputB offset:0 atIndex:10];
        } else {
            [encoder setBuffer:outputB offset:0 atIndex:9];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(envs * cavityCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding]; completed(command);
        const auto* out = static_cast<const float*>(outputB.contents);
        return std::vector<float>(out, out + std::size_t(envs) * cavityCount);
    };
    const auto oldScalar = execute(scalarPSO, diagonal, nullptr);
    const auto isotropicResult = execute(blockPSO, diagonal, &isotropic);
    for (std::size_t i = 0; i < oldScalar.size(); ++i)
        need(std::isfinite(isotropicResult[i]) &&
            std::abs(double(isotropicResult[i]) - oldScalar[i]) <=
                2e-6 * std::max(1e-30, std::abs(double(oldScalar[i]))),
            "isotropic 3x3 blocks changed the scalar cavity Schur estimate");

    const auto fullBlock = execute(blockPSO, diagonal, &anisotropic);
    double maxOracleError = 0.0, maxChange = 0.0;
    const auto grad = [](const fixture::Vec& a, const fixture::Vec& b) {
        return fixture::scaled(fixture::cross(a, b), 1.0 / 6.0);
    };
    const auto midpoint = [](const fixture::Vec& a, const fixture::Vec& b) {
        return fixture::scaled(fixture::add(a, b), 0.5);
    };
    for (unsigned e = 0; e < envs; ++e) {
        for (unsigned ci = 0; ci < cavityCount; ++ci) {
            const auto& cavity = w.vascular.cavities[ci];
            const auto first = w.vascular.cavityFaces[cavity.faces.x];
            const auto& op = accepted[e * nodes + first.nodes.x].positionAndMass;
            const fixture::Vec origin{op.x, op.y, op.z};
            const auto pos = [&](const std::vector<NMFEMNodeStateGPU>& x, unsigned n) {
                const auto& p = x[e * nodes + n].positionAndMass;
                return fixture::subtract(fixture::Vec{p.x, p.y, p.z}, origin);
            };
            double sum = 0.0;
            for (unsigned faceIndex = 0; faceIndex < cavity.faces.y; ++faceIndex) {
                const auto face = w.vascular.cavityFaces[cavity.faces.x + faceIndex];
                const unsigned n[3] = {face.nodes.x, face.nodes.y, face.nodes.z};
                const auto a0 = pos(accepted, n[0]), b0 = pos(accepted, n[1]);
                const auto c0 = pos(accepted, n[2]), a1 = pos(candidate, n[0]);
                const auto b1 = pos(candidate, n[1]), c1 = pos(candidate, n[2]);
                const auto am = midpoint(a0, a1), bm = midpoint(b0, b1);
                const auto cm = midpoint(c0, c1);
                const fixture::Vec gv[3] = {grad(b1, c1), grad(c1, a1), grad(a1, b1)};
                const fixture::Vec gp[3] = {
                    fixture::scaled(fixture::add(fixture::add(grad(b0,c0),
                        fixture::scaled(grad(bm,cm),4)),grad(b1,c1)),1.0/6.0),
                    fixture::scaled(fixture::add(fixture::add(grad(c0,a0),
                        fixture::scaled(grad(cm,am),4)),grad(c1,a1)),1.0/6.0),
                    fixture::scaled(fixture::add(fixture::add(grad(a0,b0),
                        fixture::scaled(grad(am,bm),4)),grad(a1,b1)),1.0/6.0)};
                for (unsigned corner = 0; corner < 3; ++corner) {
                    fixture::Vec applied{};
                    for (unsigned row = 0; row < 3; ++row)
                        for (unsigned col = 0; col < 3; ++col)
                            applied[row] += M[row][col] * gp[corner][col];
                    sum += std::max(fixture::dot(gv[corner], applied), 0.0);
                }
            }
            const unsigned pressureRow = cavity.identity.w;
            const double scale = dt * dt *
                double(w.vascular.unknowns[pressureRow].initialAndScaling.y) /
                double(w.vascular.unknowns[pressureRow].initialAndScaling.z);
            const double expected = sum * scale > 1e-12 ? sum * scale : 1.0;
            const std::size_t i = std::size_t(e) * cavityCount + ci;
            maxOracleError = std::max(maxOracleError,
                std::abs(double(fullBlock[i]) - expected) /
                    std::max(1e-30, std::abs(expected)));
            maxChange = std::max(maxChange,
                std::abs(double(fullBlock[i]) - oldScalar[i]) /
                    std::max(1e-30, std::abs(double(oldScalar[i]))));
        }
    }
    need(maxOracleError < 2e-4,
        "anisotropic block Schur differs from independent face-gradient reference");
    need(maxChange > 1e-3,
        "anisotropic block Schur did not exercise off-diagonal inverse entries");
    const auto invalidFallback = execute(blockPSO, diagonal, &invalid);
    for (std::size_t i = 0; i < oldScalar.size(); ++i)
        need(std::isfinite(invalidFallback[i]) &&
            std::abs(double(invalidFallback[i]) - oldScalar[i]) <=
                2e-6 * std::max(1e-30, std::abs(double(oldScalar[i]))),
            "invalid block marker did not preserve scalar fallback");
    std::cout << "cavity_schur_3x3=pass isotropic=pass anisotropic_fp64_relative_error="
        << maxOracleError << " scalar_difference=" << maxChange
        << " invalid_marker_fallback=pass operator=unchanged\n";
}

void compensatedCavityOptionFingerprint(){
    ScopedEnvironment persistent("NM_VASCULAR_PERSISTENT_LOW_WORDS", "1");
    const auto source = fixture::world();
    std::uint64_t disabledFingerprint = 0u;
    {
        ScopedEnvironment option("NM_VASCULAR_COMPENSATED_CAVITY_WORDS", nullptr);
        Run disabled(source);
        need((disabled.world.vascular.layout.ranges.w &
              NM_VASCULAR_LAYOUT_COMPENSATED_CAVITY_WORDS) == 0u,
             "compensated cavity geometry was not default-off");
        disabledFingerprint = disabled.runtime.deviceProgramFingerprint();
    }
    {
        ScopedEnvironment option("NM_VASCULAR_COMPENSATED_CAVITY_WORDS", "1");
        Run enabled(source);
        // Execution options change the runtime's private layout; compilation
        // metadata remains immutable. The production-equation fixture below
        // directly exercises that private flag's shader semantics.
        need((enabled.world.vascular.layout.ranges.w &
              NM_VASCULAR_LAYOUT_COMPENSATED_CAVITY_WORDS) == 0u,
             "runtime option mutated the immutable compiled world");
        need(enabled.runtime.deviceProgramFingerprint() != disabledFingerprint,
             "compensated cavity opt-in was omitted from the execution fingerprint");
    }
    {
        ScopedEnvironment missingPersistent("NM_VASCULAR_PERSISTENT_LOW_WORDS", "0");
        ScopedEnvironment option("NM_VASCULAR_COMPENSATED_CAVITY_WORDS", "1");
        const auto compiled = compileWorld(source, {.maximumRateExponent=0});
        need(compiled.succeeded(), "compensated option dependency fixture compile");
        Runtime rejected; RuntimeConfiguration config;
        config.metallib=NUMI_MATTER_METALLIB; config.environmentCount=2;
        const auto result=rejected.initialize(compiled.world, config);
        need(!result.encoded && result.message.find("require persistent") != std::string::npos,
             "compensated cavity option admitted an incompatible legacy equation path");
    }
    std::cout << "cavity_compensated_option=pass default_off=pass strict_fingerprint=pass dependency=pass\n";
}

bool compensatedCavityWordsRequested(){
    const char* value=std::getenv("NM_VASCULAR_COMPENSATED_CAVITY_WORDS");
    return value!=nullptr&&std::strcmp(value,"1")==0;
}
WorldSource movingWallWorld(bool pinInner,bool& referenceDisplacementGradient){
    auto source=fixture::world(pinInner);
    referenceDisplacementGradient=compensatedCavityWordsRequested();
    if(!referenceDisplacementGradient)return source;
    need(source.objects.size()==1u,"reference-displacement moving-wall fixture object count");
    auto& object=source.objects[0];
    need(object.representation==Representation::fem&&object.femReferenceNodes.empty()&&
         object.femMaterialIndices.empty()&&!object.femReferenceDisplacementGradient,
         "moving-wall reference fixture did not start from the legacy source object");
    // The same authored mesh is its stress-free reference. These stable
    // synthetic identities bind the fixture's reference/material mapping;
    // enabling the compensated option never rewrites compiled/runtime state.
    object.femReferenceNodes=object.femNodes;
    object.femReferenceSourceIdentity={
        0x766173635f726566ull,0x6772616469656e74ull,0x0000000000000001ull,0x0000000000000001ull};
    object.femMaterialIndices.assign(object.tetrahedra.size(),object.materialIndex);
    object.femMaterialSourceIdentity={
        0x766173635f6d6174ull,0x657269616c5f3031ull,0x0000000000000001ull,0x0000000000000001ull};
    object.femReferenceDisplacementGradient=true;
    return source;
}
void movingWall(){
    constexpr unsigned steps=32;const auto shell=fixture::hollowShell();
    bool referenceDisplacementGradient=false;
    const auto source=movingWallWorld(false,referenceDisplacementGradient);Run run(source);
    const bool compiledReferenceDisplacementGradient=!run.world.objects.empty()&&
        (run.world.objects[0].flags&NM_OBJECT_FEM_REFERENCE_DISPLACEMENT_GRADIENT)!=0u;
    need(compiledReferenceDisplacementGradient==referenceDisplacementGradient,
         "full moving-wall fixture did not compile the authored reference-displacement mode selected by the option");
    productionOperators(run);
    const auto first=run.state();auto signedPressure=first;
    signedPressure.vascularState[run.world.vascular.layout.cavities.z].x=-1;
    run.restore(signedPressure);const auto signedRestored=run.state();
    need(signedRestored.vascularState[run.world.vascular.layout.cavities.z].x==-1&&same(first.femNodes,signedRestored.femNodes),"finite negative cavity pressure restore was rejected or changed geometry");
    run.restore(first);auto before=first;RuntimeStateSnapshot oneStep;Metrics metrics;
    for(unsigned i=0;i<steps;++i){if(referenceDisplacementGradient&&i==0u){ScopedEnvironment vascularTrace("NM_VASCULAR_TRACE_ROOT","0");ScopedEnvironment femTrace("NM_FEM_NEWTON_TRACE_ROOT","0");run.step(i);}else run.step(i);auto after=run.state();inspectStep(run,first,before,after,shell,metrics);if(i==0)oneStep=after;before=std::move(after);}
    const auto evolved=before;
    std::size_t nonzeroLowWordComponents=0u;
    if(referenceDisplacementGradient){
        for(const auto& node:evolved.femNodes){
            nonzeroLowWordComponents+=node.referenceDisplacementLow.x!=0.0f?1u:0u;
            nonzeroLowWordComponents+=node.referenceDisplacementLow.y!=0.0f?1u:0u;
            nonzeroLowWordComponents+=node.referenceDisplacementLow.z!=0.0f?1u:0u;
        }
        need(nonzeroLowWordComponents>0u,
             "full moving-wall reference-displacement run never exercised a nonzero low word");
    }
    need(metrics.motion>1e-8&&metrics.flowMaximum>1e-10,"moving-wall response not exercised");
    need(run.physical(evolved,run.world.vascular.layout.offsets.z)>0,"flow did not transport tracer into moving cavity");
    run.restore(first);for(unsigned i=0;i<steps;++i)run.step(i);
    need(samePhysical(evolved,run.state()),"full FEM/vascular snapshot replay differs");
    if(referenceDisplacementGradient){
        run.restore(evolved);
        need(samePhysical(evolved,run.state()),
             "compensated reference-displacement snapshot restore changed the accepted cavity state");
    }
    const auto accepted=run.state();run.step(steps,1);const auto rejected=run.state();
    unchangedEnvironment(accepted.femNodes,rejected.femNodes,1,"FEM");unchangedEnvironment(accepted.femFields,rejected.femFields,1,"FEM fields");
    unchangedEnvironment(accepted.vascularState,rejected.vascularState,1,"vascular pressure/volume/amounts");unchangedEnvironment(accepted.vascularClock,rejected.vascularClock,1,"vascular clock");
    need(std::memcmp(&accepted.vascularClock[0],&rejected.vascularClock[0],sizeof(NMVascularClockGPU))!=0,"accepted peer clock stalled");
    auto corrupt=accepted;corrupt.vascularState[run.world.vascular.layout.cavities.z].x=std::numeric_limits<float>::quiet_NaN();
    const auto denied=run.runtime.restore(corrupt);need(!denied.encoded&&samePhysical(rejected,run.state()),"invalid cavity pressure restore changed accepted state");
    corrupt=accepted;corrupt.vascularState[0].x+=.01f;
    const auto geometryDenied=run.runtime.restore(corrupt);need(!geometryDenied.encoded&&samePhysical(rejected,run.state()),"inconsistent positive cavity volume/FEM geometry restore was accepted");
    corrupt=accepted;need(!corrupt.schedulers.empty(),"cavity scheduler snapshot absent");++corrupt.schedulers[0].activeExponent;
    const auto schedulerDenied=run.runtime.restore(corrupt);need(!schedulerDenied.encoded&&samePhysical(rejected,run.state()),"cavity scheduler drift restore was accepted");
    corrupt=accepted;corrupt.schedulers[0].numerical.w=std::numeric_limits<float>::infinity();
    const auto nonfiniteSchedulerDenied=run.runtime.restore(corrupt);need(!nonfiniteSchedulerDenied.encoded&&samePhysical(rejected,run.state()),"nonfinite cavity scheduler enable restore was accepted");
    run.step(steps+1,-1,1);const auto reset=run.state();
    unchangedEnvironment(oneStep.femNodes,reset.femNodes,1,"reset FEM");unchangedEnvironment(oneStep.femFields,reset.femFields,1,"reset FEM fields");
    unchangedEnvironment(oneStep.vascularState,reset.vascularState,1,"reset vascular");unchangedEnvironment(oneStep.vascularClock,reset.vascularClock,1,"reset clock");
    bool fixedReferenceDisplacementGradient=false;
    Run fixed(movingWallWorld(true,fixedReferenceDisplacementGradient));
    need(fixedReferenceDisplacementGradient==referenceDisplacementGradient,
         "fixed-wall control did not preserve the selected authored reference mode");
    const auto fixedFirst=fixed.state();fixed.step(0);const auto fixedState=fixed.state();
    const double fixedFlow=std::abs(fixed.physical(fixedState,fixed.world.vascular.layout.offsets.y));
    need(fixedFlow<1e-10&&fixedFlow<metrics.flowMaximum*.01,"fixed-wall control did not suppress cavity inflow");
    need(positions(fixedState,0,64)==positions(fixedFirst,0,64),"fixed-wall control geometry moved");
    std::cout<<"cavity_moving_wall=pass accepted_steps="<<steps<<" environments=2 max_displacement_m="<<metrics.motion<<" max_flow_m3_per_s="<<metrics.flowMaximum
             <<" fixed_wall_flow_m3_per_s="<<fixedFlow<<" normalized_geometry_residual="<<metrics.geometry<<" normalized_volume_balance="<<metrics.balance
             <<" normalized_flow_residual="<<metrics.flow<<" relative_blood_conservation="<<metrics.blood<<" relative_tracer_conservation="<<metrics.tracer
             <<" fp64_work_identity_relative_error="<<metrics.work<<" clock=exact replay=bitwise isolated_rollback=pass reset=bitwise signed_pressure_restore=accepted invalid_pressure_restore=denied geometry_restore=denied scheduler_restore=denied"
             <<" reference_displacement_gradient="<<(referenceDisplacementGradient?1:0)
             <<" evolved_nonzero_low_word_components="<<nonzeroLowWordComponents
             <<" evolved_snapshot_restore="<<(referenceDisplacementGradient?"pass":"not_applicable")
             <<" qualification=synthetic_moving_wall_interface blood_mechanical_mass=absent biological_calibration=unqualified\n";
    if(referenceDisplacementGradient)
        std::cout<<"cavity_full_moving_words=pass authored_ObjectSource_reference=1 "
                 <<"persistent_low_word_components="<<nonzeroLowWordComponents
                 <<" accepted_snapshot_restore=pass\n";
}
struct FP64Factor {
    unsigned n=0;
    std::vector<double> lu;
    std::vector<unsigned> permutation;
    explicit FP64Factor(const std::vector<double>& matrix,unsigned size):n(size),lu(matrix),permutation(size){
        need(matrix.size()==std::size_t(size)*size,"FP64 matrix arity");
        for(unsigned i=0;i<n;++i)permutation[i]=i;
        for(unsigned k=0;k<n;++k){
            unsigned pivot=k;double maximum=0;
            for(unsigned r=k;r<n;++r){const double value=std::abs(lu[std::size_t(r)*n+k]);if(value>maximum){maximum=value;pivot=r;}}
            need(std::isfinite(maximum)&&maximum>0,"FP64 principal Jacobian is singular");
            if(pivot!=k){for(unsigned c=0;c<n;++c)std::swap(lu[std::size_t(k)*n+c],lu[std::size_t(pivot)*n+c]);std::swap(permutation[k],permutation[pivot]);}
            const double diagonal=lu[std::size_t(k)*n+k];
            for(unsigned r=k+1;r<n;++r){const auto row=std::size_t(r)*n;const double multiplier=lu[row+k]/diagonal;lu[row+k]=multiplier;
                for(unsigned c=k+1;c<n;++c)lu[row+c]-=multiplier*lu[std::size_t(k)*n+c];}
        }
    }
    std::vector<double> solve(const std::vector<float>& input)const{
        need(input.size()==n,"FP64 RHS arity");std::vector<double> x(n);
        for(unsigned i=0;i<n;++i)x[i]=double(input[permutation[i]]);
        for(unsigned i=0;i<n;++i)for(unsigned j=0;j<i;++j)x[i]-=lu[std::size_t(i)*n+j]*x[j];
        for(unsigned ii=n;ii-->0;){for(unsigned j=ii+1;j<n;++j)x[ii]-=lu[std::size_t(ii)*n+j]*x[j];x[ii]/=lu[std::size_t(ii)*n+ii];}
        return x;
    }
};

void blockPreconditioner(const std::filesystem::path& physiologyPath){
    // This fixture retains the source 21-compartment/24-edge CVSim topology,
    // but overlays one explicitly synthetic cavity on a linear-compliance
    // compartment so the production factor also exercises its cavity row.
    WorldSource source=fixture::world();VascularNetworkSource network;std::string message;
    need(readHumanPhysiologyNetwork(physiologyPath,network,&message),"reference physiology: "+message);
    need(network.compartments.size()==21&&network.connections.size()==24&&network.species.empty()&&
        network.tissues.empty()&&network.exchanges.empty()&&network.cavities.empty(),
        "expected source 21-compartment/24-edge graph without transport or cavity rows");
    auto compartment=std::min_element(network.compartments.begin(),network.compartments.end(),[](const auto& a,const auto& b){
        const bool av=a.pressureLaw==VascularPressureLaw::linearCompliance&&a.storageKind==VascularStorageKind::absoluteVolume&&a.compliance>0&&a.initialVolume>0;
        const bool bv=b.pressureLaw==VascularPressureLaw::linearCompliance&&b.storageKind==VascularStorageKind::absoluteVolume&&b.compliance>0&&b.initialVolume>0;
        if(av!=bv)return av;return a.initialVolume<b.initialVolume;
    });
    need(compartment!=network.compartments.end()&&compartment->pressureLaw==VascularPressureLaw::linearCompliance&&
        compartment->storageKind==VascularStorageKind::absoluteVolume&&compartment->compliance>0&&compartment->initialVolume>0,
        "no supported linear-compliance source row for synthetic cavity overlay");
    const auto selectedCompartment=compartment->stableIdentifier;
    const double targetVolume=compartment->initialVolume;
    const double initialPressure=compartment->referencePressure+
        (compartment->initialVolume-compartment->referenceVolume)/compartment->compliance;
    need(std::isfinite(initialPressure),"source compartment pressure is nonfinite");
    compartment->pressureLaw=VascularPressureLaw::deformingCavity;
    compartment->referenceVolume=0;compartment->referencePressure=0;compartment->compliance=0;compartment->externalPressure=0;
    compartment->elastanceMin=compartment->elastanceMax=compartment->periodSeconds=0;
    compartment->activationStart=compartment->activationEnd=compartment->sourcePi=compartment->phaseDelay=0;
    compartment->maximumVolumeDisplacement=0;compartment->periodNumeratorSeconds=compartment->periodDenominator=0;
    auto edge=std::find_if(network.connections.begin(),network.connections.end(),[](const auto& value){return value.flowLaw==VascularFlowLaw::oneWayResistance;});
    need(edge!=network.connections.end(),"source graph has no one-way edge for a closed-valve working-set row");
    const auto constrainedStableEdge=edge->stableIdentifier;edge->initialFlow=0;
    network.cavities.clear();source.vascular=std::move(network);source.environmentCount=2;source.frameTimestep=.001;

    const auto shell=fixture::hollowShell();source.objects.at(0)=shell.object;
    const double baseLumen=fixture::volume(shell.object.femNodes,shell.lumenFaces);
    need(baseLumen>0&&std::isfinite(baseLumen),"synthetic cavity reference volume");
    const double geometryScale=std::cbrt(targetVolume/baseLumen);
    need(geometryScale>0&&std::isfinite(geometryScale),"synthetic cavity scale");
    for(auto& p:source.objects[0].femNodes)for(double& x:p)x*=geometryScale;
    source.objects[0].characteristicLength*=geometryScale;
    VascularCavitySource cavity;cavity.stableIdentifier=0x7f000001u;cavity.compartment=selectedCompartment;
    cavity.objectIndex=0;cavity.initialPressure=initialPressure;cavity.pressureScale=std::max(1000.0,std::abs(initialPressure));
    cavity.geometryResidualTolerance=1e-5;cavity.sourceIdentity={0x73796e7468657469ULL,0x632d626c6f636b31ULL,1,0};
    cavity.mechanicalIdentity={0x73796e7468657469ULL,0x632d686f6c6c6f77ULL,1,0};
    for(unsigned i=0;i<shell.lumenFaces.size();++i)cavity.faces.push_back({0x7f001000u+i,shell.lumenFaces[i],VascularCavityFaceRole::materialWall});
    source.vascular.cavities.push_back(cavity);
    auto compiled=compileWorld(source,{.maximumRateExponent=0});
    std::string diagnostics;for(const auto& d:compiled.diagnostics)if(d.severity==Diagnostic::Severity::error)diagnostics+=d.message+"; ";
    need(compiled.succeeded(),"block fixture compile: "+diagnostics);
    const auto& w=compiled.world;const auto& v=w.vascular;const unsigned n=v.layout.ranges.z,envs=w.dispatch.environmentCount;
    need(n==46&&envs==2&&v.layout.counts.x==21&&v.layout.counts.y==24&&v.layout.cavities.x==1,
        "synthetic overlay did not compile to expected 46-row source-topology block");
    const unsigned flowRow=v.layout.offsets.y;unsigned constrainedEdge=NM_INVALID_INDEX;
    for(unsigned i=0;i<v.connections.size();++i)if(v.connections[i].identity.x==constrainedStableEdge)constrainedEdge=i;
    need(constrainedEdge!=NM_INVALID_INDEX&&v.connections[constrainedEdge].identity.w==std::uint32_t(VascularFlowLaw::oneWayResistance),
        "compiled closed-valve source edge identity was not retained");

    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
    need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"Metal command queue unavailable");
    NSError* error=nil;auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]] error:&error];
    need(library!=nil,"metallib unavailable for principal block fixture");
    const auto operatorPipeline=pipeline(device,library,@"numi_matter_metal::nm_vascular_operator");
    const auto factorPipeline=pipeline(device,library,@"numi_matter_metal::nm_vascular_factor_principal_block");
    const auto applyPipeline=pipeline(device,library,@"numi_matter_metal::nm_vascular_apply_principal_block");
    const auto diagonalPipeline=pipeline(device,library,@"numi_matter_metal::nm_vascular_precondition");
    const auto elastancePipeline=pipeline(device,library,@"numi_matter_metal::nm_vascular_prepare_elastance");
    NMMicrostepGPU micro{};micro.time={float(source.frameTimestep),float(1./source.frameTimestep),0,0};micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;
    NMFGMRESLayoutGPU solverLayout{};solverLayout.vascularUnknownCount=n;solverLayout.vascularBase=envs*w.dispatch.femNodeCount+7;solverLayout.unknownCount=solverLayout.vascularBase+envs*n+5;
    std::vector<nm_float4> accepted(envs*n),candidate(envs*n),zeroDirection(solverLayout.unknownCount);
    for(unsigned e=0;e<envs;++e)for(unsigned i=0;i<n;++i){const auto& q=v.unknowns[i].initialAndScaling;accepted[e*n+i]=candidate[e*n+i]={q.x/q.y,0,0,0};}
    std::vector<std::uint32_t> workingSet(envs*n,0);workingSet[flowRow+constrainedEdge]=1;
    auto fem=w.fem.nodes;need(fem.size()==w.dispatch.femNodeCount,"compiled synthetic shell FEM node count");
    std::vector<NMFEMNodeStateGPU> femState;femState.reserve(envs*fem.size());for(unsigned e=0;e<envs;++e)femState.insert(femState.end(),fem.begin(),fem.end());
    std::vector<NMVascularClockGPU> clocks(envs);std::vector<NMMatterStatusGPU> statusValues(envs);
    const auto solverStateBuffer=gpuBuffer(device,std::vector<NMFGMRESStateGPU>(envs));
    const auto dispatchBuffer=gpuBuffer(device,std::vector<NMMatterDispatchGPU>{w.dispatch});
    const auto graphBuffer=gpuBuffer(device,std::vector<NMVascularLayoutGPU>{v.layout});
    const auto microBuffer=gpuBuffer(device,std::vector<NMMicrostepGPU>{micro});
    const auto unknownBuffer=gpuBuffer(device,v.unknowns),compartmentBuffer=gpuBuffer(device,v.compartments);
    const auto connectionBuffer=gpuBufferOrDummy(device,v.connections),tissueBuffer=gpuBufferOrDummy(device,v.tissues),exchangeBuffer=gpuBufferOrDummy(device,v.exchanges);
    const auto connIncidence=gpuBufferOrDummy(device,v.connectionIncidence),connRanges=gpuBufferOrDummy(device,v.connectionRanges);
    const auto bloodIncidence=gpuBufferOrDummy(device,v.bloodExchangeIncidence),bloodRanges=gpuBufferOrDummy(device,v.bloodExchangeRanges);
    const auto tissueIncidence=gpuBufferOrDummy(device,v.tissueExchangeIncidence),tissueRanges=gpuBufferOrDummy(device,v.tissueExchangeRanges);
    const auto acceptedBuffer=gpuBuffer(device,accepted),candidateBuffer=gpuBuffer(device,candidate),zeroDirectionBuffer=gpuBuffer(device,zeroDirection);
    const auto workingBuffer=gpuBuffer(device,workingSet),cavityBuffer=gpuBuffer(device,v.cavities),faceBuffer=gpuBuffer(device,v.cavityFaces);
    const auto mappingBuffer=gpuBuffer(device,v.compartmentCavity),femAcceptedBuffer=gpuBuffer(device,femState),femCandidateBuffer=gpuBuffer(device,femState);
    const auto clockBuffer=gpuBuffer(device,clocks),statusBuffer=gpuBuffer(device,statusValues);
    const auto elastanceBuffer=[device newBufferWithLength:std::size_t(envs)*v.layout.counts.x*sizeof(float) options:MTLResourceStorageModeShared];
    need(elastanceBuffer!=nil,"elastance allocation failed");
    {auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:elastancePipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];[encoder setBuffer:graphBuffer offset:0 atIndex:1];[encoder setBuffer:compartmentBuffer offset:0 atIndex:2];
        [encoder setBuffer:clockBuffer offset:0 atIndex:3];[encoder setBuffer:elastanceBuffer offset:0 atIndex:4];[encoder setBuffer:statusBuffer offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(envs*v.layout.counts.x,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);}
    const auto* preparedStatuses=static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
    for(unsigned e=0;e<envs;++e)need(preparedStatuses[e].code==NM_STATUS_SUCCESS,"source graph elastance preparation failed");

    const std::uint32_t useWorkingSet=1;
    auto bindOperator=[&](id<MTLComputeCommandEncoder> encoder,id<MTLBuffer> direction,id<MTLBuffer> output){
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];[encoder setBuffer:graphBuffer offset:0 atIndex:1];[encoder setBuffer:microBuffer offset:0 atIndex:2];
        [encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];[encoder setBuffer:connectionBuffer offset:0 atIndex:5];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:6];[encoder setBuffer:exchangeBuffer offset:0 atIndex:7];[encoder setBuffer:connIncidence offset:0 atIndex:8];
        [encoder setBuffer:connRanges offset:0 atIndex:9];[encoder setBuffer:bloodIncidence offset:0 atIndex:10];[encoder setBuffer:bloodRanges offset:0 atIndex:11];
        [encoder setBuffer:tissueIncidence offset:0 atIndex:12];[encoder setBuffer:tissueRanges offset:0 atIndex:13];[encoder setBuffer:acceptedBuffer offset:0 atIndex:14];
        [encoder setBuffer:candidateBuffer offset:0 atIndex:15];[encoder setBuffer:direction offset:0 atIndex:16];[encoder setBuffer:output offset:0 atIndex:17];
        [encoder setBuffer:solverStateBuffer offset:0 atIndex:18];[encoder setBuffer:statusBuffer offset:0 atIndex:19];
        [encoder setBuffer:elastanceBuffer offset:0 atIndex:20];[encoder setBuffer:workingBuffer offset:0 atIndex:21];[encoder setBytes:&useWorkingSet length:sizeof(useWorkingSet) atIndex:22];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:23];[encoder setBuffer:faceBuffer offset:0 atIndex:24];[encoder setBuffer:mappingBuffer offset:0 atIndex:25];
        [encoder setBuffer:femCandidateBuffer offset:0 atIndex:26];[encoder setBuffer:femAcceptedBuffer offset:0 atIndex:27];
        [encoder setBytes:&solverLayout length:sizeof(solverLayout) atIndex:30];
    };
    // Extract J columns through the existing production matrix-free operator.
    // Closed-valve row replacement is then applied exactly as the factor kernel does.
    std::vector<std::vector<double>> matrices(envs,std::vector<double>(std::size_t(n)*n));
    std::vector<id<MTLBuffer>> directionBuffers,outputBuffers;directionBuffers.reserve(n);outputBuffers.reserve(n);
    auto matrixCommand=[queue commandBuffer];
    for(unsigned column=0;column<n;++column){std::vector<nm_float4> basis(solverLayout.unknownCount);
        for(unsigned e=0;e<envs;++e)basis[solverLayout.vascularBase+e*n+column].x=1;
        auto direction=gpuBuffer(device,basis);auto output=[device newBufferWithLength:std::size_t(solverLayout.unknownCount)*sizeof(nm_float4) options:MTLResourceStorageModeShared];
        need(output!=nil,"operator output allocation failed");directionBuffers.push_back(direction);outputBuffers.push_back(output);
        auto encoder=[matrixCommand computeCommandEncoder];[encoder setComputePipelineState:operatorPipeline];bindOperator(encoder,direction,output);
        [encoder dispatchThreads:MTLSizeMake(envs*n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
    }
    completed(matrixCommand);
    for(unsigned e=0;e<envs;++e)need(preparedStatuses[e].code==NM_STATUS_SUCCESS,"production principal-J operator rejected the source state");
    for(unsigned column=0;column<n;++column){const auto* values=static_cast<const nm_float4*>(outputBuffers[column].contents);
        for(unsigned e=0;e<envs;++e)for(unsigned row=0;row<n;++row)matrices[e][std::size_t(row)*n+column]=values[solverLayout.vascularBase+e*n+row].x;}
    const unsigned constrainedRow=flowRow+constrainedEdge;
    need(std::abs(matrices[0][std::size_t(constrainedRow)*n+constrainedRow])<1e-6,"closed source valve operator did not remove its active row");
    for(unsigned column=0;column<n;++column)matrices[0][std::size_t(constrainedRow)*n+column]=column==constrainedRow?1.0:0.0;
    FP64Factor oracle0(matrices[0],n),oracle1(matrices[1],n);

    const auto factorBuffer=[device newBufferWithLength:std::size_t(envs)*64*64*sizeof(float) options:MTLResourceStorageModeShared];
    const auto pivotBuffer=[device newBufferWithLength:std::size_t(envs)*64*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    const auto validBuffer=[device newBufferWithLength:std::size_t(envs)*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    need(factorBuffer&&pivotBuffer&&validBuffer,"principal factor arena allocation failed");
    {auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:factorPipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];[encoder setBuffer:graphBuffer offset:0 atIndex:1];[encoder setBuffer:microBuffer offset:0 atIndex:2];
        [encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];[encoder setBuffer:connectionBuffer offset:0 atIndex:5];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:6];[encoder setBuffer:exchangeBuffer offset:0 atIndex:7];[encoder setBuffer:connIncidence offset:0 atIndex:8];
        [encoder setBuffer:connRanges offset:0 atIndex:9];[encoder setBuffer:bloodIncidence offset:0 atIndex:10];[encoder setBuffer:bloodRanges offset:0 atIndex:11];
        [encoder setBuffer:tissueIncidence offset:0 atIndex:12];[encoder setBuffer:tissueRanges offset:0 atIndex:13];[encoder setBuffer:acceptedBuffer offset:0 atIndex:14];
        [encoder setBuffer:candidateBuffer offset:0 atIndex:15];[encoder setBuffer:zeroDirectionBuffer offset:0 atIndex:16];[encoder setBuffer:elastanceBuffer offset:0 atIndex:20];
        [encoder setBuffer:workingBuffer offset:0 atIndex:21];[encoder setBytes:&useWorkingSet length:sizeof(useWorkingSet) atIndex:22];[encoder setBuffer:cavityBuffer offset:0 atIndex:23];
        [encoder setBuffer:faceBuffer offset:0 atIndex:24];[encoder setBuffer:mappingBuffer offset:0 atIndex:25];[encoder setBuffer:femCandidateBuffer offset:0 atIndex:26];
        [encoder setBuffer:femAcceptedBuffer offset:0 atIndex:27];[encoder setBuffer:factorBuffer offset:0 atIndex:28];[encoder setBuffer:pivotBuffer offset:0 atIndex:29];
        [encoder setBuffer:validBuffer offset:0 atIndex:30];[encoder dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];completed(command);}
    const auto* valid=static_cast<const std::uint32_t*>(validBuffer.contents);need(valid[0]==1&&valid[1]==1,"production 46-row principal factor rejected a valid source-topology fixture");
    const auto* factoredValues=static_cast<const float*>(factorBuffer.contents);
    const auto factorSnapshot=std::vector<float>(factoredValues,factoredValues+std::size_t(envs)*64*64);
    const auto pivotSnapshot=std::vector<std::uint32_t>(static_cast<const std::uint32_t*>(pivotBuffer.contents),static_cast<const std::uint32_t*>(pivotBuffer.contents)+std::size_t(envs)*64);
    const auto solverStateSnapshot=std::vector<NMFGMRESStateGPU>(static_cast<const NMFGMRESStateGPU*>(solverStateBuffer.contents),static_cast<const NMFGMRESStateGPU*>(solverStateBuffer.contents)+envs);
    std::vector<nm_float4> rhsA(solverLayout.unknownCount),rhsB(solverLayout.unknownCount);
    for(unsigned e=0;e<envs;++e)for(unsigned i=0;i<n;++i){rhsA[solverLayout.vascularBase+e*n+i].x=float(std::sin(double(i+1)*.31+e*.7));rhsB[solverLayout.vascularBase+e*n+i].x=float(std::cos(double(i+2)*.17-e*.3));}
    rhsA[solverLayout.vascularBase+constrainedRow].x=3.25f;
    auto rhsABuffer=gpuBuffer(device,rhsA),rhsBBuffer=gpuBuffer(device,rhsB);
    std::vector<nm_float4> initialOutput(solverLayout.unknownCount,nm_float4{13,17,19,23});
    auto outputABuffer=gpuBuffer(device,initialOutput),outputBBuffer=gpuBuffer(device,initialOutput);
    auto apply=[&](id<MTLBuffer> basis,id<MTLBuffer> output,id<MTLBuffer> factorValid){
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:applyPipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];[encoder setBuffer:graphBuffer offset:0 atIndex:1];[encoder setBytes:&solverLayout length:sizeof(solverLayout) atIndex:2];
        [encoder setBuffer:basis offset:0 atIndex:3];[encoder setBuffer:output offset:0 atIndex:4];[encoder setBuffer:solverStateBuffer offset:0 atIndex:5];[encoder setBuffer:workingBuffer offset:0 atIndex:6];
        [encoder setBuffer:factorBuffer offset:0 atIndex:7];[encoder setBuffer:pivotBuffer offset:0 atIndex:8];[encoder setBuffer:factorValid offset:0 atIndex:9];
        [encoder setBytes:&useWorkingSet length:sizeof(useWorkingSet) atIndex:10];[encoder dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);
    };
    apply(rhsABuffer,outputABuffer,validBuffer);apply(rhsBBuffer,outputBBuffer,validBuffer);
    const auto checkSolution=[&](const std::vector<nm_float4>& input,id<MTLBuffer> output,unsigned e,const FP64Factor& oracle){
        const auto* actual=static_cast<const nm_float4*>(output.contents);std::vector<float> rhs(n);for(unsigned i=0;i<n;++i)rhs[i]=(e==0&&i==constrainedRow)?0.f:input[solverLayout.vascularBase+e*n+i].x;
        const auto expected=oracle.solve(rhs);double maxExpected=0,maxError=0,maxResidual=0,maxRhs=0;
        for(unsigned row=0;row<n;++row){const double got=actual[solverLayout.vascularBase+e*n+row].x;maxExpected=std::max(maxExpected,std::abs(expected[row]));maxError=std::max(maxError,std::abs(got-expected[row]));maxRhs=std::max(maxRhs,std::abs(double(rhs[row])));
            double applied=0;for(unsigned c=0;c<n;++c)applied+=matrices[e][std::size_t(row)*n+c]*double(actual[solverLayout.vascularBase+e*n+c].x);
            maxResidual=std::max(maxResidual,std::abs(applied-double(rhs[row])));}
        need(maxError/std::max(1e-30,maxExpected)<2e-3,"GPU principal apply disagrees with independent FP64 LU");
        need(maxResidual/std::max(1e-30,maxRhs)<3e-3,"GPU principal apply has excessive FP64 principal residual");
        if(e==0)need(actual[solverLayout.vascularBase+constrainedRow].x==0,"closed-valve constrained RHS was not eliminated");
        return std::array<double,2>{maxError/std::max(1e-30,maxExpected),maxResidual/std::max(1e-30,maxRhs)};
    };
    const auto a0=checkSolution(rhsA,outputABuffer,0,oracle0),a1=checkSolution(rhsA,outputABuffer,1,oracle1);
    const auto b0=checkSolution(rhsB,outputBBuffer,0,oracle0),b1=checkSolution(rhsB,outputBBuffer,1,oracle1);
    need(std::memcmp(factorSnapshot.data(),factorBuffer.contents,factorSnapshot.size()*sizeof(float))==0&&
        std::memcmp(pivotSnapshot.data(),pivotBuffer.contents,pivotSnapshot.size()*sizeof(std::uint32_t))==0,
        "RHS applications changed the reused principal factor");
    need(std::memcmp(solverStateSnapshot.data(),solverStateBuffer.contents,solverStateSnapshot.size()*sizeof(NMFGMRESStateGPU))==0,
        "principal block apply changed read-only FGMRES solver states");
    for(unsigned i=0;i<solverLayout.vascularBase;++i)need(std::memcmp(&static_cast<const nm_float4*>(outputABuffer.contents)[i],&initialOutput[i],sizeof(nm_float4))==0,"principal apply touched nonvascular prefix");
    for(unsigned i=solverLayout.vascularBase+envs*n;i<solverLayout.unknownCount;++i)need(std::memcmp(&static_cast<const nm_float4*>(outputABuffer.contents)[i],&initialOutput[i],sizeof(nm_float4))==0,"principal apply touched nonvascular suffix");

    // Run the existing rowwise diagonal path first. A deliberately invalid
    // factor must leave that exact fallback result intact.
    auto fallbackBuffer=gpuBuffer(device,initialOutput);const auto cavitySchur=gpuBuffer(device,std::vector<float>(envs,1.f));
    {auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:diagonalPipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];[encoder setBuffer:graphBuffer offset:0 atIndex:1];[encoder setBuffer:microBuffer offset:0 atIndex:2];
        [encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:connectionBuffer offset:0 atIndex:4];[encoder setBuffer:rhsABuffer offset:0 atIndex:5];
        [encoder setBuffer:fallbackBuffer offset:0 atIndex:6];[encoder setBuffer:solverStateBuffer offset:0 atIndex:7];[encoder setBuffer:candidateBuffer offset:0 atIndex:8];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:9];[encoder setBuffer:exchangeBuffer offset:0 atIndex:10];[encoder setBuffer:connIncidence offset:0 atIndex:11];
        [encoder setBuffer:connRanges offset:0 atIndex:12];[encoder setBuffer:bloodIncidence offset:0 atIndex:13];[encoder setBuffer:bloodRanges offset:0 atIndex:14];
        [encoder setBuffer:tissueIncidence offset:0 atIndex:15];[encoder setBuffer:tissueRanges offset:0 atIndex:16];[encoder setBuffer:statusBuffer offset:0 atIndex:17];
        [encoder setBuffer:compartmentBuffer offset:0 atIndex:18];[encoder setBuffer:elastanceBuffer offset:0 atIndex:19];[encoder setBuffer:workingBuffer offset:0 atIndex:20];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:21];[encoder setBuffer:mappingBuffer offset:0 atIndex:22];[encoder setBuffer:cavitySchur offset:0 atIndex:23];
        [encoder setBytes:&solverLayout length:sizeof(solverLayout) atIndex:30];[encoder dispatchThreads:MTLSizeMake(envs*n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];completed(command);}
    const auto fallbackSnapshot=std::vector<nm_float4>(static_cast<const nm_float4*>(fallbackBuffer.contents),static_cast<const nm_float4*>(fallbackBuffer.contents)+solverLayout.unknownCount);
    const std::vector<std::uint32_t> invalidFactors(envs,0);auto invalidBuffer=gpuBuffer(device,invalidFactors);apply(rhsBBuffer,fallbackBuffer,invalidBuffer);
    need(std::memcmp(fallbackSnapshot.data(),fallbackBuffer.contents,fallbackSnapshot.size()*sizeof(nm_float4))==0,
        "invalid factor did not preserve existing rowwise diagonal fallback output");
    need(same(rhsA,std::vector<nm_float4>(static_cast<const nm_float4*>(rhsABuffer.contents),static_cast<const nm_float4*>(rhsABuffer.contents)+rhsA.size()))&&
        same(rhsB,std::vector<nm_float4>(static_cast<const nm_float4*>(rhsBBuffer.contents),static_cast<const nm_float4*>(rhsBBuffer.contents)+rhsB.size()))&&
        same(workingSet,std::vector<std::uint32_t>(static_cast<const std::uint32_t*>(workingBuffer.contents),static_cast<const std::uint32_t*>(workingBuffer.contents)+workingSet.size())),
        "principal preconditioner modified input RHS or active-set state");
    need(same(accepted,std::vector<nm_float4>(static_cast<const nm_float4*>(acceptedBuffer.contents),static_cast<const nm_float4*>(acceptedBuffer.contents)+accepted.size()))&&
        same(candidate,std::vector<nm_float4>(static_cast<const nm_float4*>(candidateBuffer.contents),static_cast<const nm_float4*>(candidateBuffer.contents)+candidate.size())),
        "principal fixture modified accepted/candidate vascular state");
    std::cout<<"vascular_principal_block=pass source_compartments=21 source_edges=24 synthetic_cavity_overlay=1 rows="<<n
        <<" vascular_base="<<solverLayout.vascularBase<<" closed_valve_edge_stable_id="<<constrainedStableEdge
        <<" fp64_lu_relative_solution_error="<<std::max({a0[0],a1[0],b0[0],b1[0]})
        <<" fp64_lu_relative_residual="<<std::max({a0[1],a1[1],b0[1],b1[1]})
        <<" rhs_per_factor=2 invalid_factor_diagonal_fallback=preserved state=bitwise biological_qualification=none\n";
}

struct PredictorProbe {
    std::uint64_t fingerprint=0;
    RuntimeStateSnapshot initial,accepted;
    double femMass=0,initialCavityVolume=0,acceptedCavityVolume=0;
};
PredictorProbe predictorProbe(const char* blockPolicy,const char* predictorPolicy){
    ScopedVascularPolicy policy(blockPolicy,predictorPolicy);
    Run run(fixture::world());PredictorProbe probe;probe.fingerprint=run.world.fingerprint;
    for(const auto& node:run.world.fem.nodes)probe.femMass+=double(node.positionAndMass.w);
    probe.initial=run.state();
    const unsigned volumeRow=run.world.vascular.cavities.at(0).identity.y;
    probe.initialCavityVolume=run.physical(probe.initial,volumeRow);
    run.step(0);probe.accepted=run.state();
    probe.acceptedCavityVolume=run.physical(probe.accepted,volumeRow);
    need(std::isfinite(probe.initialCavityVolume)&&std::isfinite(probe.acceptedCavityVolume)&&
        probe.initialCavityVolume>0&&probe.acceptedCavityVolume>0,
        "block predictor fixture lost its positive finite cavity blood volume");
    return probe;
}
void blockPredictorOnly(){
    // The caller policy is restored after each initialization. This tests the
    // actual Runtime capture point without leaking the opt-in to other probes.
    const auto unset=predictorProbe("1",nullptr);
    const auto explicitOff=predictorProbe("1","0");
    const auto enabled=predictorProbe("1","1");
    need(unset.fingerprint==explicitOff.fingerprint&&unset.fingerprint==enabled.fingerprint,
        "vascular predictor policy changed the compiled physical world");
    need(samePhysical(unset.initial,explicitOff.initial)&&samePhysical(unset.accepted,explicitOff.accepted),
        "unset and explicit-off vascular predictor changed accepted snapshot bits");
    need(samePhysical(unset.initial,enabled.initial)&&unset.femMass==explicitOff.femMass&&
        unset.femMass==enabled.femMass,
        "vascular predictor initialization changed initial physical state or FEM mass");
    need(unset.initial.vascularState.size()==enabled.initial.vascularState.size()&&
        unset.initial.femNodes.size()==enabled.initial.femNodes.size(),
        "vascular predictor added physical state storage");
    const bool enabledStepMatchesOff=samePhysical(unset.accepted,enabled.accepted);
    std::cout<<"vascular_block_predictor_policy=pass preconditioner=1 predictor_unset_zero_initial_and_first_step=bitwise enabled_initial_state=bitwise fem_mass_kg="
        <<enabled.femMass<<" initial_cavity_volume_m3="<<enabled.initialCavityVolume
        <<" first_step_cavity_volume_m3="<<enabled.acceptedCavityVolume
        <<" enabled_step_matches_off="<<(enabledStepMatchesOff?"bitwise":"different")
        <<" enabled_minus_off_cavity_volume_m3="<<(enabled.acceptedCavityVolume-unset.acceptedCavityVolume)
        <<" prediction_improvement=not_claimed\n";
    {ScopedVascularPolicy policy("1","1");movingWall();}
}


void semismoothUpwindWorkingSet(VascularFlowLaw flowLaw) {
    ScopedEnvironment semismoothPolicy("NM_VASCULAR_SEMISMOOTH_TRANSPORT", "1");
    auto source = fixture::world();
    auto& authoredEdge = source.vascular.connections.at(0);
    authoredEdge.flowLaw = flowLaw;
    authoredEdge.resistance = 1.0e9;
    authoredEdge.inertance = 0.0;
    authoredEdge.reverseResistance = flowLaw == VascularFlowLaw::directionalResistance
        ? 2.0e9 : 0.0;
    authoredEdge.initialFlow = 0.0;
    // This is a focused production-kernel regression. Full Runtime ownership,
    // rollback and replay remain exercised by movingWall(); compiling every
    // unrelated simulation pipeline is unnecessary for these authored inputs.
    auto compiled = compileWorld(source, {.maximumRateExponent = 0});
    need(compiled.succeeded(), "semismooth focused fixture compile failed");
    struct FocusedInputs {
        CompiledWorld world;
        id<MTLDevice> device;
        id<MTLCommandQueue> queue;
    } run;
    run.world = std::move(compiled.world);
    run.device = MTLCreateSystemDefaultDevice();
    need(run.device != nil &&
        [[run.device name] rangeOfString:@"Apple"].location != NSNotFound &&
        [[run.device name] rangeOfString:@"Paravirtual"].location == NSNotFound,
        "physical Apple Metal required");
    run.queue = [run.device newCommandQueue];
    need(run.queue != nil, "focused fixture command queue unavailable");
    const auto& world = run.world;
    RuntimeStateSnapshot snapshot;
    for (std::uint32_t e = 0; e < world.dispatch.environmentCount; ++e) {
        snapshot.femNodes.insert(snapshot.femNodes.end(),
            world.fem.nodes.begin(), world.fem.nodes.end());
        for (const auto& unknown : world.vascular.unknowns)
            snapshot.vascularState.push_back({
                unknown.initialAndScaling.x / unknown.initialAndScaling.y,
                0.0f, 0.0f, 0.0f});
    }
    const std::uint32_t environments = world.dispatch.environmentCount;
    const std::uint32_t nodes = world.dispatch.femNodeCount;
    NMVascularLayoutGPU graph = world.vascular.layout;
    graph.ranges.w |= NM_VASCULAR_LAYOUT_SEMISMOOTH_TRANSPORT;
    const std::uint32_t unknownCount = graph.ranges.z;
    need(environments == 2u && graph.counts.z == 1u &&
        graph.counts.y == 1u && world.vascular.connections.size() == 1u &&
        world.vascular.connections[0].identity.w == static_cast<std::uint32_t>(flowLaw),
        "semismooth transport fixture layout differs");
    const std::uint32_t flowRow = graph.offsets.y;
    const std::uint32_t cavity = world.vascular.connections[0].identity.z;
    const std::uint32_t reservoir = world.vascular.connections[0].identity.y;
    const std::uint32_t cavityAmountRow = graph.offsets.z + cavity * graph.counts.z;
    const std::uint32_t reservoirAmountRow = graph.offsets.z + reservoir * graph.counts.z;
    const double dt = source.frameTimestep;
    const std::uint32_t vascularBase = 4u * nodes;
    NMFGMRESLayoutGPU solverLayout{};
    solverLayout.unknownCount = vascularBase + environments * unknownCount;
    solverLayout.vascularUnknownCount = unknownCount;
    solverLayout.vascularBase = vascularBase;
    NMMicrostepGPU micro{};
    micro.time = {float(dt), float(1.0 / dt), 0.0f, 0.0f};
    micro.solverIteration = 0u;

    std::vector<nm_float4> candidate = snapshot.vascularState;
    const auto setPhysical = [&](std::vector<nm_float4>& state,
        std::uint32_t environment, std::uint32_t row, double physical) {
        auto& value = state[environment * unknownCount + row];
        value.x = float(physical /
            world.vascular.unknowns[row].initialAndScaling.y);
        value.y = 0.0f;
    };
    for (std::uint32_t e = 0; e < environments; ++e) {
        setPhysical(candidate, e, flowRow, 0.0);
        setPhysical(candidate, e, cavityAmountRow, 0.0);
        setPhysical(candidate, e, reservoirAmountRow, 2.0e-6);
    }

    std::vector<float> elastance(std::size_t(environments) *
        graph.counts.x, 0.0f);
    std::vector<NMFGMRESStateGPU> states(environments);
    std::vector<NMMatterStatusGPU> initialStatuses(environments);
    for (std::uint32_t e = 0; e < environments; ++e) {
        initialStatuses[e] = {};
        initialStatuses[e].environment = e;
        initialStatuses[e].code = NM_STATUS_SUCCESS;
    }
    const auto dispatchB = gpuBuffer(run.device,
        std::vector<NMMatterDispatchGPU>{world.dispatch});
    auto graphB = gpuBuffer(run.device, std::vector<NMVascularLayoutGPU>{graph});
    const auto microB = gpuBuffer(run.device, std::vector<NMMicrostepGPU>{micro});
    const auto unknownB = gpuBuffer(run.device, world.vascular.unknowns);
    const auto compartmentB = gpuBuffer(run.device, world.vascular.compartments);
    const auto connectionB = gpuBuffer(run.device, world.vascular.connections);
    const auto tissueB = gpuBufferOrDummy(run.device, world.vascular.tissues);
    const auto exchangeB = gpuBufferOrDummy(run.device, world.vascular.exchanges);
    const auto connectionIncidenceB = gpuBuffer(run.device,
        world.vascular.connectionIncidence);
    const auto connectionRangesB = gpuBuffer(run.device,
        world.vascular.connectionRanges);
    const auto bloodIncidenceB = gpuBufferOrDummy(run.device,
        world.vascular.bloodExchangeIncidence);
    const auto bloodRangesB = gpuBufferOrDummy(run.device,
        world.vascular.bloodExchangeRanges);
    const auto tissueIncidenceB = gpuBufferOrDummy(run.device,
        world.vascular.tissueExchangeIncidence);
    const auto tissueRangesB = gpuBufferOrDummy(run.device,
        world.vascular.tissueExchangeRanges);
    const auto elastanceB = gpuBuffer(run.device, elastance);
    const auto cavityB = gpuBuffer(run.device, world.vascular.cavities);
    const auto cavityFaceB = gpuBuffer(run.device, world.vascular.cavityFaces);
    const auto compartmentCavityB = gpuBuffer(run.device,
        world.vascular.compartmentCavity);
    const auto femB = gpuBuffer(run.device, snapshot.femNodes);
    const auto candidateB = gpuBuffer(run.device, candidate);
    const auto workingPipeline = pipeline(run.device,
        [run.device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]]
            error:nil],
        @"numi_matter_metal::nm_vascular_begin_working_set");
    auto workingSet = std::vector<std::uint32_t>(
        std::size_t(environments) * unknownCount, 0u);
    auto workingB = gpuBuffer(run.device, workingSet);
    {
        auto command = [run.queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:workingPipeline];
        [encoder setBuffer:dispatchB offset:0 atIndex:0];
        [encoder setBuffer:graphB offset:0 atIndex:1];
        [encoder setBuffer:microB offset:0 atIndex:2];
        [encoder setBuffer:unknownB offset:0 atIndex:3];
        [encoder setBuffer:compartmentB offset:0 atIndex:4];
        [encoder setBuffer:connectionB offset:0 atIndex:5];
        [encoder setBuffer:candidateB offset:0 atIndex:6];
        [encoder setBuffer:elastanceB offset:0 atIndex:7];
        [encoder setBuffer:workingB offset:0 atIndex:8];
        [encoder setBuffer:cavityB offset:0 atIndex:9];
        [encoder setBuffer:compartmentCavityB offset:0 atIndex:10];
        [encoder dispatchThreads:MTLSizeMake(std::size_t(environments) * unknownCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        completed(command);
    }
    workingSet.assign(static_cast<const std::uint32_t*>(workingB.contents),
        static_cast<const std::uint32_t*>(workingB.contents) +
            std::size_t(environments) * unknownCount);
    const std::uint32_t forwardWord =
        NM_VASCULAR_WORKING_SET_TRANSPORT_BRANCH_VALID;
    const auto& compiledEdge = world.vascular.connections[0];
    const double flowScale = world.vascular.unknowns[flowRow].initialAndScaling.y;
    const double flowResidualScale =
        world.vascular.unknowns[flowRow].initialAndScaling.z;
    const double cavityResidualScale =
        world.vascular.unknowns[cavityAmountRow].initialAndScaling.z;
    const double reservoirResidualScale =
        world.vascular.unknowns[reservoirAmountRow].initialAndScaling.z;
    const auto expectedFlowSlope = [&](bool reverse, int directionSign) {
        const double resistance = flowLaw == VascularFlowLaw::directionalResistance && reverse
            ? compiledEdge.directional.x
            : compiledEdge.physical.x + compiledEdge.physical.y / dt;
        return resistance * flowScale * double(directionSign) / flowResidualScale;
    };
    const std::uint32_t reverseWord = forwardWord |
        NM_VASCULAR_WORKING_SET_TRANSPORT_REVERSE;
    need(workingSet[flowRow] == forwardWord,
        "begin working set did not seed the positive-pressure branch at Q=0");

    const auto operatorPipeline = pipeline(run.device,
        [run.device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]]
            error:nil],
        @"numi_matter_metal::nm_vascular_operator");
    const auto resolvePipeline = pipeline(run.device,
        [run.device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]]
            error:nil],
        @"numi_matter_metal::nm_vascular_resolve_working_set");
    const auto directionFor = [&](int sign) {
        std::vector<nm_float4> direction(solverLayout.unknownCount);
        direction[vascularBase + flowRow].x = float(sign);
        return direction;
    };
    const auto applyOperator = [&](const std::vector<nm_float4>& candidateState,
        const std::vector<nm_float4>& direction,
        const std::vector<std::uint32_t>& words) {
        auto candidateInputB = gpuBuffer(run.device, candidateState);
        auto acceptedInputB = gpuBuffer(run.device, candidateState);
        auto directionB = gpuBuffer(run.device, direction);
        auto wordsB = gpuBuffer(run.device, words);
        auto outputB = gpuBuffer(run.device,
            std::vector<nm_float4>(solverLayout.unknownCount));
        auto stateB = gpuBuffer(run.device, states);
        auto statusB = gpuBuffer(run.device, initialStatuses);
        const std::uint32_t useWorkingSet = 1u;
        auto command = [run.queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:operatorPipeline];
        [encoder setBuffer:dispatchB offset:0 atIndex:0];
        [encoder setBuffer:graphB offset:0 atIndex:1];
        [encoder setBuffer:microB offset:0 atIndex:2];
        [encoder setBuffer:unknownB offset:0 atIndex:3];
        [encoder setBuffer:compartmentB offset:0 atIndex:4];
        [encoder setBuffer:connectionB offset:0 atIndex:5];
        [encoder setBuffer:tissueB offset:0 atIndex:6];
        [encoder setBuffer:exchangeB offset:0 atIndex:7];
        [encoder setBuffer:connectionIncidenceB offset:0 atIndex:8];
        [encoder setBuffer:connectionRangesB offset:0 atIndex:9];
        [encoder setBuffer:bloodIncidenceB offset:0 atIndex:10];
        [encoder setBuffer:bloodRangesB offset:0 atIndex:11];
        [encoder setBuffer:tissueIncidenceB offset:0 atIndex:12];
        [encoder setBuffer:tissueRangesB offset:0 atIndex:13];
        [encoder setBuffer:acceptedInputB offset:0 atIndex:14];
        [encoder setBuffer:candidateInputB offset:0 atIndex:15];
        [encoder setBuffer:directionB offset:0 atIndex:16];
        [encoder setBuffer:outputB offset:0 atIndex:17];
        [encoder setBuffer:stateB offset:0 atIndex:18];
        [encoder setBuffer:statusB offset:0 atIndex:19];
        [encoder setBuffer:elastanceB offset:0 atIndex:20];
        [encoder setBuffer:wordsB offset:0 atIndex:21];
        [encoder setBytes:&useWorkingSet length:sizeof(useWorkingSet) atIndex:22];
        [encoder setBuffer:cavityB offset:0 atIndex:23];
        [encoder setBuffer:cavityFaceB offset:0 atIndex:24];
        [encoder setBuffer:compartmentCavityB offset:0 atIndex:25];
        [encoder setBuffer:femB offset:0 atIndex:26];
        [encoder setBuffer:femB offset:0 atIndex:27];
        [encoder setBytes:&solverLayout length:sizeof(solverLayout) atIndex:30];
        [encoder dispatchThreads:MTLSizeMake(
            std::size_t(environments) * unknownCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        completed(command);
        const auto* statusValues =
            static_cast<const NMMatterStatusGPU*>(statusB.contents);
        for (std::uint32_t e = 0; e < environments; ++e)
            need(statusValues[e].code == NM_STATUS_SUCCESS,
                "semismooth operator rejected environment");
        const auto* values = static_cast<const nm_float4*>(outputB.contents);
        return std::vector<nm_float4>(values,
            values + solverLayout.unknownCount);
    };
    const auto resolve = [&](const std::vector<nm_float4>& candidateState,
        const std::vector<nm_float4>& direction,
        std::vector<std::uint32_t> words) {
        auto candidateInputB = gpuBuffer(run.device, candidateState);
        auto wordsB = gpuBuffer(run.device, words);
        auto changedB = gpuBuffer(run.device,
            std::vector<std::uint32_t>(environments, 0u));
        auto solutionB = gpuBuffer(run.device, direction);
        auto statusB = gpuBuffer(run.device, initialStatuses);
        auto command = [run.queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:resolvePipeline];
        [encoder setBuffer:dispatchB offset:0 atIndex:0];
        [encoder setBuffer:graphB offset:0 atIndex:1];
        [encoder setBuffer:connectionB offset:0 atIndex:2];
        [encoder setBuffer:candidateInputB offset:0 atIndex:3];
        [encoder setBuffer:solutionB offset:0 atIndex:4];
        [encoder setBuffer:wordsB offset:0 atIndex:5];
        [encoder setBuffer:changedB offset:0 atIndex:6];
        [encoder setBuffer:statusB offset:0 atIndex:7];
        [encoder setBuffer:unknownB offset:0 atIndex:8];
        [encoder setBuffer:compartmentB offset:0 atIndex:9];
        [encoder setBuffer:elastanceB offset:0 atIndex:10];
        [encoder setBuffer:cavityB offset:0 atIndex:11];
        [encoder setBuffer:compartmentCavityB offset:0 atIndex:12];
        [encoder setBytes:&solverLayout length:sizeof(solverLayout) atIndex:30];
        [encoder dispatchThreadgroups:MTLSizeMake(environments, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        completed(command);
        const auto* changed =
            static_cast<const std::uint32_t*>(changedB.contents);
        const auto* nextWords =
            static_cast<const std::uint32_t*>(wordsB.contents);
        const auto* discarded = static_cast<const nm_float4*>(solutionB.contents);
        for (std::uint32_t e = 0; e < environments; ++e) {
            if (changed[e] == 0u) continue;
            for (std::uint32_t i = 0; i < solverLayout.unknownCount; ++i)
                need(discarded[i].x == 0.0f && discarded[i].y == 0.0f &&
                    discarded[i].z == 0.0f && discarded[i].w == 0.0f,
                    "working-set branch change did not discard the old Newton direction");
        }
        return std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>>{
            std::vector<std::uint32_t>(nextWords,
                nextWords + std::size_t(environments) * unknownCount),
            std::vector<std::uint32_t>(changed, changed + environments)};
    };
    const auto expectedTracerDerivative = [&](const std::vector<nm_float4>& state,
        std::uint32_t environment, int qDirection,
        std::uint32_t donorCompartment, std::uint32_t equationCompartment) {
        const auto physical = [&](std::uint32_t row) {
            return (double(state[environment * unknownCount + row].x) +
                double(state[environment * unknownCount + row].y)) *
                double(world.vascular.unknowns[row].initialAndScaling.y);
        };
        const double volume = physical(donorCompartment);
        const std::uint32_t amountRow = graph.offsets.z +
            donorCompartment * graph.counts.z;
        const double concentration = physical(amountRow) / volume;
        const double dq = double(qDirection) *
            double(world.vascular.unknowns[flowRow].initialAndScaling.y);
        const double sign = world.vascular.connections[0].identity.y ==
            equationCompartment ? 1.0 : -1.0;
        return dt * sign * dq * concentration /
            double(world.vascular.unknowns[graph.offsets.z +
                equationCompartment * graph.counts.z].initialAndScaling.z);
    };
    const auto setDirection = directionFor;
    auto negative = setDirection(-1);
    auto positive = setDirection(1);
    auto words = workingSet;
    const auto base = std::size_t(0) * unknownCount;
    words[base + flowRow] = forwardWord;
    auto jv = applyOperator(candidate, negative, words);
    const double forwardReservoirEquation = expectedTracerDerivative(
        candidate, 0, -1, reservoir, reservoir);
    const double forwardCavityEquation = expectedTracerDerivative(
        candidate, 0, -1, reservoir, cavity);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x) -
            forwardCavityEquation) < 1e-7 &&
        std::abs(double(jv[vascularBase + reservoirAmountRow].x) -
            forwardReservoirEquation) < 1e-7 &&
        std::abs(double(jv[vascularBase + cavityAmountRow].x) * cavityResidualScale +
            double(jv[vascularBase + reservoirAmountRow].x) * reservoirResidualScale) < 1e-14 &&
        std::abs(double(jv[vascularBase + flowRow].x) -
            expectedFlowSlope(false, -1)) < 1e-6,
        "zero-flow forward branch failed hydraulic or equal-and-opposite tracer tangent");
    const auto switchedReverse = resolve(candidate, negative, words);
    need(switchedReverse.second[0] == 1u &&
        switchedReverse.first[flowRow] == reverseWord,
        "negative Newton flow correction did not switch the zero-flow donor branch");
    jv = applyOperator(candidate, negative, switchedReverse.first);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x)) < 1e-8 &&
        std::abs(double(jv[vascularBase + reservoirAmountRow].x)) < 1e-8 &&
        std::abs(double(jv[vascularBase + flowRow].x) -
            expectedFlowSlope(true, -1)) < 1e-6,
        "reverse Q=0 working set failed zero-cavity transport or hydraulic branch");

    auto switchedForward = resolve(candidate, positive, switchedReverse.first);
    need(switchedForward.second[0] == 1u &&
        switchedForward.first[flowRow] == forwardWord,
        "positive Newton flow correction did not switch the zero-flow donor branch");
    jv = applyOperator(candidate, positive, switchedForward.first);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x) -
            expectedTracerDerivative(candidate, 0, 1, reservoir, cavity)) < 1e-7 &&
        std::abs(double(jv[vascularBase + reservoirAmountRow].x) -
            expectedTracerDerivative(candidate, 0, 1, reservoir, reservoir)) < 1e-7 &&
        std::abs(double(jv[vascularBase + flowRow].x) -
            expectedFlowSlope(false, 1)) < 1e-6,
        "positive-flow reversal failed the reservoir donor or hydraulic tangent");

    // Give both donors nonzero, unequal concentrations. The derivative must
    // use the working-set donor selected by the Newton flow sign and conserve
    // the amount transfer across both adjacent compartment rows.
    setPhysical(candidate, 0, cavityAmountRow, 0.25e-6);
    const double reverseCavityEquation = expectedTracerDerivative(
        candidate, 0, -1, cavity, cavity);
    const double reverseReservoirEquation = expectedTracerDerivative(
        candidate, 0, -1, cavity, reservoir);
    jv = applyOperator(candidate, negative, switchedReverse.first);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x) -
            reverseCavityEquation) < 1e-7 &&
        std::abs(double(jv[vascularBase + reservoirAmountRow].x) -
            reverseReservoirEquation) < 1e-7 &&
        std::abs(double(jv[vascularBase + cavityAmountRow].x) * cavityResidualScale +
            double(jv[vascularBase + reservoirAmountRow].x) * reservoirResidualScale) < 1e-14 &&
        std::abs(reverseReservoirEquation - forwardReservoirEquation) > 1e-8,
        "reverse-flow donor concentration or conservative exchange is wrong");

    // Away from Q=0, the physical flow sign owns the donor regardless of a
    // stale working-set branch bit.
    auto nonzero = candidate;
    setPhysical(nonzero, 0, flowRow, -0.01 * world.vascular.unknowns[flowRow].initialAndScaling.y);
    jv = applyOperator(nonzero, positive, switchedForward.first);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x) -
            expectedTracerDerivative(nonzero, 0, 1, cavity, cavity)) < 1e-7,
        "negative nonzero flow did not select the cavity donor");
    setPhysical(nonzero, 0, flowRow, 0.01 * world.vascular.unknowns[flowRow].initialAndScaling.y);
    jv = applyOperator(nonzero, negative, switchedReverse.first);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x) -
            expectedTracerDerivative(nonzero, 0, -1, reservoir, cavity)) < 1e-7,
        "positive nonzero flow did not select the reservoir donor");

    auto legacyGraph = graph;
    legacyGraph.ranges.w &= ~NM_VASCULAR_LAYOUT_SEMISMOOTH_TRANSPORT;
    const auto legacyGraphB = gpuBuffer(run.device,
        std::vector<NMVascularLayoutGPU>{legacyGraph});
    auto command = [run.queue commandBuffer];
    auto blit = [command blitCommandEncoder];
    [blit copyFromBuffer:legacyGraphB sourceOffset:0 toBuffer:graphB
        destinationOffset:0 size:sizeof(NMVascularLayoutGPU)];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    auto legacyWords = switchedReverse.first;
    jv = applyOperator(candidate, negative, legacyWords);
    need(std::abs(double(jv[vascularBase + cavityAmountRow].x) -
            forwardCavityEquation) < 1e-7,
        "default-off legacy pressure-tie donor behavior changed");
    std::cout << "vascular_semismooth_upwind=pass flow_law="
        << static_cast<std::uint32_t>(flowLaw) << " default=legacy "
        << "zero_flow_branch_switch=bidirectional "
        << "zero_cavity_negative_tracer_direction=zero "
        << "nonzero_Q_sign_reversal=pass amount_conservation=pass "
        << "physical_qualification=none\n";
}
} // namespace
int main(int argc,char** argv){@autoreleasepool{try{std::cout<<std::setprecision(17);
if(argc==3&&std::string(argv[1])=="--block-preconditioner-only"){blockPreconditioner(argv[2]);return 0;}
if(argc==2&&std::string(argv[1])=="--block-predictor-only"){blockPredictorOnly();return 0;}
if(argc==2&&std::string(argv[1])=="--cavity-schur-block-only"){cavitySchurBlockPreconditioner();return 0;}
if(argc==2&&std::string(argv[1])=="--semismooth-upwind-only"){
    semismoothUpwindWorkingSet(VascularFlowLaw::resistanceInertance);
    semismoothUpwindWorkingSet(VascularFlowLaw::directionalResistance);
    return 0;
}
need(argc==1,"usage: vascular_cavity_check [--block-preconditioner-only human-physiology.json | --block-predictor-only | --cavity-schur-block-only | --semismooth-upwind-only]");
geometryOracle();bloodMassTangentNoOwnerAdmission();bloodMassOwner();pressureMomentumOwner();compensatedCavityOptionFingerprint();movingWall();return 0;}
catch(const std::exception& error){std::cerr<<"cavity_check=failed reason="<<error.what()<<'\n';return 1;}}}
