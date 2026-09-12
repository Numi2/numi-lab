#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_cavity_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
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
        need(std::abs(mass-.026)<1e-8,"hydraulic lumen added mechanical mass to the hollow tissue shell");
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
} // namespace
int main(){@autoreleasepool{try{std::cout<<std::setprecision(17);geometryOracle();movingWall();return 0;}
catch(const std::exception& error){std::cerr<<"cavity_check=failed reason="<<error.what()<<'\n';return 1;}}}
