#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_cavity_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>

using namespace numi::matter;
namespace fixture=vascular_cavity_fixture;
namespace {
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
    auto encodeCommon=[&](id<MTLComputeCommandEncoder> encoder,id<MTLBuffer> compartmentBuffer,id<MTLBuffer> candidateBuffer,id<MTLBuffer> vascularBuffer){
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];[encoder setBuffer:incidenceBuffer offset:0 atIndex:7];
        [encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:acceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:vascularBuffer offset:0 atIndex:11];
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
    auto apply=[&](const std::vector<NMFEMNodeStateGPU>& positions,const std::vector<NMFGMRESStateGPU>& solverStates){
        const auto cb=[run.queue commandBuffer];auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:operatorPipeline];
        const auto c=gpuBuffer(run.device,positions),v=gpuBuffer(run.device,vascular),p=gpuBuffer(run.device,compartments),d=gpuBuffer(run.device,direction);
        const auto initial=std::vector<nm_float4>(2*nodes,nm_float4{.0001f,-.0002f,.0003f,0});
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
    auto equation=[&](const std::vector<NMFEMNodeStateGPU>& fem,const std::vector<nm_float4>& states,
                      const std::vector<nm_float4>& lifted,bool derivative){
        auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];
        const auto p=gpuBuffer(run.device,compartments),v=gpuBuffer(run.device,states),d=gpuBuffer(run.device,lifted),f=gpuBuffer(run.device,fem);
        const auto out=gpuBuffer(run.device,std::vector<nm_float4>(layout.unknownCount,nm_float4{17,18,19,20}));
        const auto solver=gpuBuffer(run.device,std::vector<NMFGMRESStateGPU>(2)),status=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));
        const auto elastance=gpuBuffer(run.device,std::vector<float>(4)),working=gpuBuffer(run.device,std::vector<std::uint32_t>(2*layout.vascularUnknownCount));
        const std::uint32_t useWorking=0;
        const auto clock=gpuBuffer(run.device,run.state().vascularClock);
        [encoder setComputePipelineState:preparePipeline];
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];
        [encoder setBuffer:p offset:0 atIndex:2];[encoder setBuffer:clock offset:0 atIndex:3];[encoder setBuffer:elastance offset:0 atIndex:4];[encoder setBuffer:status offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(2*w.vascular.layout.counts.x,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
        encoder=[command computeCommandEncoder];[encoder setComputePipelineState:derivative?vascularOperatorPipeline:residualPipeline];
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:p offset:0 atIndex:4];
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
    const auto shell=fixture::hollowShell();Run run(fixture::world(false,.001,true));const auto& w=run.world;
    need(w.vascular.tissues.size()==1&&w.vascular.tissueBindings.size()==shell.innerNodes.size(),"blood owner fixture did not cook its FEM region");
    const auto& owner=w.vascular.tissues[0];need(owner.identity.w!=0u&&owner.physical.y>1000.f&&owner.physical.z>0.f,"blood owner identity, density or initial mechanical mass was not cooked");
    need(std::abs(double(owner.physical.z)-4e-6*1060.0)<1e-8,"blood owner initial mechanical mass disagrees with hydraulic volume");
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
    const auto tissueBuffer=gpuBuffer(run.device,w.vascular.tissues),bindingBuffer=gpuBuffer(run.device,w.vascular.tissueBindings);
    const auto connectionBuffer=gpuBuffer(run.device,w.vascular.connections),femAcceptedBuffer=gpuBuffer(run.device,accepted);
    std::vector<float> fixtureElastance(w.dispatch.environmentCount*w.vascular.compartments.size(),1.f);
    for(unsigned env=0;env<w.dispatch.environmentCount;++env) for(unsigned i=0;i<w.vascular.compartments.size();++i)
        if(w.vascular.compartments[i].identity.w==0u||w.vascular.compartments[i].identity.w==3u) fixtureElastance[env*w.vascular.compartments.size()+i]=float(1./w.vascular.compartments[i].compliance.z);
    const auto elastanceBuffer=gpuBufferOrDummy(run.device,fixtureElastance);
    auto encodeForce=[&](const std::vector<nm_float4>& candidateState,const std::vector<nm_float4>& acceptedState){
        const auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:forcePipeline];
        const auto stateBuffer=gpuBuffer(run.device,candidateState),acceptedVascularBuffer=gpuBuffer(run.device,acceptedState),candidateBuffer=gpuBuffer(run.device,accepted),borrowed=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes));
        const auto output=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes,nm_float4{99,99,99,0})),statuses=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));const std::uint32_t include=0;
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];[encoder setBytes:&micro length:sizeof(micro) atIndex:2];
        [encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];[encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];
        [encoder setBuffer:incidenceBuffer offset:0 atIndex:7];[encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:femAcceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:stateBuffer offset:0 atIndex:11];
        [encoder setBuffer:borrowed offset:0 atIndex:12];[encoder setBytes:&include length:sizeof(include) atIndex:13];[encoder setBuffer:output offset:0 atIndex:14];[encoder setBuffer:statuses offset:0 atIndex:15];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:16];[encoder setBuffer:bindingBuffer offset:0 atIndex:17];[encoder setBuffer:elastanceBuffer offset:0 atIndex:18];[encoder setBuffer:connectionBuffer offset:0 atIndex:19];[encoder setBuffer:acceptedVascularBuffer offset:0 atIndex:20];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);
        for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(statuses.contents)[env].code==NM_STATUS_SUCCESS,"pressure/momentum reaction force state rejected");
        const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);
    };
    const auto first=encodeForce(vascular,vascular),second=encodeForce(vascular,vascular);need(same(first,second),"pressure/momentum reaction replay changed output");
    unsigned from=NM_INVALID_INDEX,to=NM_INVALID_INDEX;for(unsigned i=0;i<w.vascular.compartments.size();++i){if(w.vascular.compartments[i].identity.x==12u)from=i;if(w.vascular.compartments[i].identity.x==11u)to=i;}
    need(from!=NM_INVALID_INDEX&&to!=NM_INVALID_INDEX,"pressure reaction compartments were not resolved");
    const double volume=run.physical(run.state(),from);const auto& source=w.vascular.compartments[from];const double pFrom=source.compliance.y+(volume-source.compliance.x)/source.compliance.z;const double pTo=run.physical(run.state(),w.vascular.cavities[0].identity.w);const double totalReaction=-(pFrom-pTo)*double(owner.physical.w),weight=1./double(shell.innerNodes.size());
    double observedTotal=0;for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();const auto& force=first[env*nodes+node];const double expected=owned?totalReaction*weight:0.;need(std::abs(double(force.z)-expected)<2e-7&&std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7,"pressure reaction force disagreed with the authored pressure-area direction");observedTotal+=force.z;}
    need(std::abs(observedTotal-2.*totalReaction)<2e-6,"pressure reaction did not conserve the registered wall force");
    const unsigned flowRow=w.vascular.layout.offsets.y;const double flowDelta=double(w.vascular.unknowns[flowRow].initialAndScaling.y);
    auto momentumCandidate=vascular;for(unsigned env=0;env<2;++env) momentumCandidate[env*w.vascular.layout.ranges.z+flowRow].x+=1.f;
    const auto momentumForces=encodeForce(momentumCandidate,vascular);
    const double momentumReaction=-double(owner.physical.y)*double(owner.physical.x)/double(owner.physical.w)*flowDelta/double(micro.time.x);
    const double combinedReaction=totalReaction+momentumReaction;double momentumObserved=0;
    for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();const auto& force=momentumForces[env*nodes+node];const double expected=owned?combinedReaction*weight:0.;need(std::abs(double(force.z)-expected)<3e-7&&std::abs(double(force.x))<2e-7&&std::abs(double(force.y))<2e-7,"fluid momentum transfer force disagreed with the finite-volume impulse");momentumObserved+=force.z;}
    need(std::abs(momentumObserved-2.*combinedReaction)<3e-6&&std::abs(momentumReaction)>1e-6,"fluid momentum transfer was not exercised or conserved");
    NMFGMRESLayoutGPU layout{};layout.vascularUnknownCount=unsigned(w.vascular.unknowns.size());layout.vascularBase=4*nodes;layout.unknownCount=layout.vascularBase+2*layout.vascularUnknownCount;
    auto direction=std::vector<nm_float4>(layout.unknownCount);for(unsigned env=0;env<2;++env){direction[layout.vascularBase+env*w.vascular.unknowns.size()+from].x=.5f;direction[layout.vascularBase+env*w.vascular.unknowns.size()+flowRow].x=.25f;}
    const auto directionBuffer=gpuBuffer(run.device,direction),acceptedBuffer=gpuBuffer(run.device,accepted),candidateBuffer=gpuBuffer(run.device,accepted),stateBuffer=gpuBuffer(run.device,vascular);
    auto encodeOperator=[&](){const auto command=[run.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:operatorPipeline];
        const auto output=gpuBuffer(run.device,std::vector<nm_float4>(2*nodes)),solver=gpuBuffer(run.device,std::vector<NMFGMRESStateGPU>(2)),statuses=gpuBuffer(run.device,std::vector<NMMatterStatusGPU>(2));
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&w.vascular.layout length:sizeof(w.vascular.layout) atIndex:1];[encoder setBytes:&micro length:sizeof(micro) atIndex:2];[encoder setBuffer:unknownBuffer offset:0 atIndex:3];[encoder setBuffer:compartmentBuffer offset:0 atIndex:4];
        [encoder setBuffer:cavityBuffer offset:0 atIndex:5];[encoder setBuffer:faceBuffer offset:0 atIndex:6];[encoder setBuffer:incidenceBuffer offset:0 atIndex:7];[encoder setBuffer:rangeBuffer offset:0 atIndex:8];[encoder setBuffer:acceptedBuffer offset:0 atIndex:9];[encoder setBuffer:candidateBuffer offset:0 atIndex:10];[encoder setBuffer:stateBuffer offset:0 atIndex:11];[encoder setBuffer:directionBuffer offset:0 atIndex:12];[encoder setBuffer:output offset:0 atIndex:13];[encoder setBuffer:solver offset:0 atIndex:14];[encoder setBuffer:statuses offset:0 atIndex:15];
        [encoder setBuffer:tissueBuffer offset:0 atIndex:16];[encoder setBuffer:bindingBuffer offset:0 atIndex:17];[encoder setBuffer:elastanceBuffer offset:0 atIndex:18];[encoder setBuffer:connectionBuffer offset:0 atIndex:19];[encoder setBuffer:stateBuffer offset:0 atIndex:20];[encoder setBytes:&layout length:sizeof(layout) atIndex:30];
        [encoder dispatchThreads:MTLSizeMake(2*nodes,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];completed(command);for(unsigned env=0;env<2;++env)need(static_cast<NMMatterStatusGPU*>(statuses.contents)[env].code==NM_STATUS_SUCCESS,"pressure reaction tangent state rejected");const auto* values=static_cast<nm_float4*>(output.contents);return std::vector<nm_float4>(values,values+2*nodes);};
    const auto firstOperator=encodeOperator(),secondOperator=encodeOperator();need(same(firstOperator,secondOperator),"pressure reaction tangent replay changed output");
    const double pressureDelta=.5*double(w.vascular.unknowns[from].initialAndScaling.y)/source.compliance.z;const double tangentTotal=micro.time.x*pressureDelta*double(owner.physical.w);const double momentumTangentTotal=double(owner.physical.y)*double(owner.physical.x)/double(owner.physical.w)*.25*double(w.vascular.unknowns[flowRow].initialAndScaling.y);const double combinedTangentTotal=tangentTotal+momentumTangentTotal;for(unsigned env=0;env<2;++env)for(unsigned node=0;node<nodes;++node){const bool owned=std::find(shell.innerNodes.begin(),shell.innerNodes.end(),node)!=shell.innerNodes.end();const double expected=owned?combinedTangentTotal*weight:0.;const auto& value=firstOperator[env*nodes+node];need(std::abs(double(value.z)-expected)<4e-7&&std::abs(double(value.x))<3e-7&&std::abs(double(value.y))<3e-7,"pressure/momentum reaction tangent disagreed with the coupled derivative");}
    std::cout<<"pressure_gradient_reaction=pass from_compartment=12 to_compartment=11 area_m2="<<owner.physical.w<<" total_force_per_environment_N="<<totalReaction<<" momentum_force_per_environment_N="<<momentumReaction<<" replay=bitwise jacobian=pass pressure_driven_fluid_momentum=pass anatomical_registration=unqualified subject_calibration=unqualified\n";
}
void movingWall(){
    constexpr unsigned steps=32;const auto shell=fixture::hollowShell();Run run(fixture::world());
    productionOperators(run);
    const auto first=run.state();auto signedPressure=first;
    signedPressure.vascularState[run.world.vascular.layout.cavities.z].x=-1;
    run.restore(signedPressure);const auto signedRestored=run.state();
    need(signedRestored.vascularState[run.world.vascular.layout.cavities.z].x==-1&&same(first.femNodes,signedRestored.femNodes),"finite negative cavity pressure restore was rejected or changed geometry");
    run.restore(first);auto before=first;RuntimeStateSnapshot oneStep;Metrics metrics;
    for(unsigned i=0;i<steps;++i){run.step(i);auto after=run.state();inspectStep(run,first,before,after,shell,metrics);if(i==0)oneStep=after;before=std::move(after);}
    const auto evolved=before;
    need(metrics.motion>1e-8&&metrics.flowMaximum>1e-10,"moving-wall response not exercised");
    need(run.physical(evolved,run.world.vascular.layout.offsets.z)>0,"flow did not transport tracer into moving cavity");
    run.restore(first);for(unsigned i=0;i<steps;++i)run.step(i);
    need(samePhysical(evolved,run.state()),"full FEM/vascular snapshot replay differs");
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
    Run fixed(fixture::world(true));const auto fixedFirst=fixed.state();fixed.step(0);const auto fixedState=fixed.state();
    const double fixedFlow=std::abs(fixed.physical(fixedState,fixed.world.vascular.layout.offsets.y));
    need(fixedFlow<1e-10&&fixedFlow<metrics.flowMaximum*.01,"fixed-wall control did not suppress cavity inflow");
    need(positions(fixedState,0,64)==positions(fixedFirst,0,64),"fixed-wall control geometry moved");
    std::cout<<"cavity_moving_wall=pass accepted_steps="<<steps<<" environments=2 max_displacement_m="<<metrics.motion<<" max_flow_m3_per_s="<<metrics.flowMaximum
             <<" fixed_wall_flow_m3_per_s="<<fixedFlow<<" normalized_geometry_residual="<<metrics.geometry<<" normalized_volume_balance="<<metrics.balance
             <<" normalized_flow_residual="<<metrics.flow<<" relative_blood_conservation="<<metrics.blood<<" relative_tracer_conservation="<<metrics.tracer
             <<" fp64_work_identity_relative_error="<<metrics.work<<" clock=exact replay=bitwise isolated_rollback=pass reset=bitwise signed_pressure_restore=accepted invalid_pressure_restore=denied geometry_restore=denied scheduler_restore=denied"
             <<" qualification=synthetic_moving_wall_interface blood_mechanical_mass=absent biological_calibration=unqualified\n";
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

} // namespace
int main(int argc,char** argv){@autoreleasepool{try{std::cout<<std::setprecision(17);
if(argc==3&&std::string(argv[1])=="--block-preconditioner-only"){blockPreconditioner(argv[2]);return 0;}
if(argc==2&&std::string(argv[1])=="--block-predictor-only"){blockPredictorOnly();return 0;}
need(argc==1,"usage: vascular_cavity_check [--block-preconditioner-only human-physiology.json | --block-predictor-only]");
geometryOracle();bloodMassOwner();pressureMomentumOwner();movingWall();return 0;}
catch(const std::exception& error){std::cerr<<"cavity_check=failed reason="<<error.what()<<'\n';return 1;}}}
