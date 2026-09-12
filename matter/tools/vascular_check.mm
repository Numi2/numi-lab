#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_fixture.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace numi::matter;
namespace {
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
bool same(const std::vector<nm_float4>& a,const std::vector<nm_float4>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(nm_float4))==0);}
WorldSource sourceFor(VascularNetworkSource n,double dt,bool fem=false) {
    WorldSource s;s.environmentCount=2;s.frameTimestep=dt;s.gravity={0,0,0};s.vascular=std::move(n);
    s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
    if(fem){
        const auto p=parseMatterFile(NUMI_MATTER_MATERIAL);need(p.succeeded(),"fixture material parse");s.materials.push_back(p.material);
        ObjectSource o;o.name="fixture:organ-region";o.representation=Representation::fem;o.characteristicLength=.01;
        o.deformableContact=false;o.mixedFEM=false;o.femNodes={{0,0,0},{.01,0,0},{0,.01,0},{0,0,.01}};
        o.femFixedNodes={0,1,2,3};o.tetrahedra.push_back({{0,1,2,3}});s.objects.push_back(o);
        s.vascular.tissues[0].objectIndex=0;s.vascular.tissues[0].femRegion={{0,.25},{1,.25},{2,.25},{3,.25}};
    }
    return s;
}
struct Run {
    CompiledWorld world;Runtime runtime;
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLBuffer> statuses;
    explicit Run(const WorldSource& s) {
        auto c=compileWorld(s,{.maximumRateExponent=0});
        for(const auto& d:c.diagnostics)if(d.severity==Diagnostic::Severity::error)std::cerr<<d.message<<'\n';
        need(c.succeeded(),"vascular fixture compile failed");
        const auto package=std::filesystem::temp_directory_path()/
            (std::string("numi-vascular-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        std::string error;need(writePackage(c,package,&error),error);
        const bool loaded=readPackage(package,world,nullptr,&error);
        std::error_code removal;std::filesystem::remove(package,removal);
        need(loaded && !removal && world.fingerprint==c.world.fingerprint,"runtime package roundtrip failed: "+error);
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device missing");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound && [[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue];statuses=[device newBufferWithLength:2*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        RuntimeConfiguration cfg;cfg.metallib=NUMI_MATTER_METALLIB;cfg.environmentCount=2;cfg.captureEvents=false;cfg.captureDiagnostics=true;cfg.adaptiveTransfer=false;
        const auto init=runtime.initialize(world,cfg);need(init.encoded,init.message);
    }
    RuntimeStateSnapshot state(){auto s=runtime.snapshot();need(s.available,s.message);return s;}
    void step(unsigned index,int reject=-1,int reset=-1) {
        auto* values=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        for(unsigned e=0;e<2;++e){values[e]={};values[e].environment=e;}
        id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"command buffer unavailable");
        EncodeRequest req;req.commandBuffer=(__bridge void*)cb;req.environmentStatuses=(__bridge void*)statuses;req.controlStep=index;
        req.physicsSubsteps=1;req.timestepSeconds=runtime.timestepSeconds();req.runAdaptiveTransfer=false;
        id<MTLBuffer> resetBuffer=nil;
        if(reset>=0){
            std::vector<std::uint32_t> masks((index+1)*2u,0u);masks[index*2u+reset]=1u;
            resetBuffer=[device newBufferWithBytes:masks.data() length:masks.size()*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
            req.resetMasks=(__bridge void*)resetBuffer;req.resetMaskStepStride=2u;
        }
        req.phase=EncodePhase::preDynamics;auto r=runtime.encode(req);need(r.encoded,"pre: "+r.message);
        id<MTLBuffer> failure=nil;
        if(reject>=0){
            MRMetalWorldStatusGPU fail{};fail.code=MR_STEP_DID_NOT_CONVERGE;fail.environment=reject;
            failure=[device newBufferWithBytes:&fail length:sizeof(fail) options:MTLResourceStorageModeShared];
            id<MTLBlitCommandEncoder> b=[cb blitCommandEncoder];
            [b copyFromBuffer:failure sourceOffset:0 toBuffer:statuses destinationOffset:reject*sizeof(fail) size:sizeof(fail)];[b endEncoding];
        }
        req.phase=EncodePhase::postCommit;r=runtime.encode(req);need(r.encoded,"post: "+r.message);
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal command failed");(void)failure;(void)resetBuffer;
        auto end=state();need(end.statuses.size()==2,"missing environment certificates");
        for(unsigned e=0;e<2;++e)if(static_cast<int>(e)!=reject) need(end.statuses[e].code==NM_STATUS_SUCCESS,
            "vascular step "+std::to_string(index)+" environment "+std::to_string(e)+" rejected code="+std::to_string(end.statuses[e].code)+" residual="+std::to_string(end.statuses[e].diagnostics.z));
    }
};

// Independent FP64 elimination for the two-compartment case. Hydraulics is
// solved analytically; transported species use a separate dense amount matrix.
// This reference does not call native material/kernels or their derivatives.
struct Reference {double a,b,q;std::array<double,3> amount;};
Reference initial(const VascularNetworkSource& n){return{n.compartments[0].initialVolume,n.compartments[1].initialVolume,n.connections[0].initialFlow,{n.compartments[0].initialSpeciesAmounts[0],n.compartments[1].initialSpeciesAmounts[0],n.tissues[0].initialSpeciesAmounts[0]}};}
Reference advance(Reference old,const VascularNetworkSource& n,double dt){
    auto pressure=[](double v,const VascularCompartmentSource& c){return c.externalPressure+c.referencePressure+(v-c.referenceVolume)/c.compliance;};
    const auto& edge=n.connections[0];
    const double q=(pressure(old.a,n.compartments[0])-pressure(old.b,n.compartments[1])+edge.inertance*old.q/dt)/(edge.resistance+edge.inertance/dt+dt*(1/n.compartments[0].compliance+1/n.compartments[1].compliance));
    Reference out{old.a-dt*q,old.b+dt*q,q,{}};
    double m[3][4]={{1,0,0,old.amount[0]},{0,1,0,old.amount[1]},{0,0,1,old.amount[2]}};
    const unsigned donor=q>=0?0:1,receiver=1-donor;const double adv=dt*std::abs(q)/(donor==0?out.a:out.b);
    m[donor][donor]+=adv;m[receiver][donor]-=adv;
    const auto& ex=n.exchanges[0];const unsigned blood=ex.compartment==n.compartments[0].stableIdentifier?0:1;
    const double into=dt*ex.permeabilitySurface/(blood==0?out.a:out.b);
    const double back=dt*ex.permeabilitySurface/(n.tissues[0].volume*ex.partitionCoefficient);
    m[blood][blood]+=into;m[blood][2]-=back;m[2][blood]-=into;m[2][2]+=back;
    for(unsigned k=0;k<3;++k){unsigned pivot=k;for(unsigned i=k+1;i<3;++i)if(std::abs(m[i][k])>std::abs(m[pivot][k]))pivot=i;for(unsigned j=0;j<4;++j)std::swap(m[k][j],m[pivot][j]);need(std::abs(m[k][k])>1e-12,"singular reference");double d=m[k][k];for(unsigned j=k;j<4;++j)m[k][j]/=d;for(unsigned i=0;i<3;++i)if(i!=k){double f=m[i][k];for(unsigned j=k;j<4;++j)m[i][j]-=f*m[k][j];}}
    for(unsigned i=0;i<3;++i)out.amount[i]=m[i][3];return out;
}
double value(const Run& r,const RuntimeStateSnapshot& s,unsigned i,unsigned e=0){return double(s.vascularState[e*r.world.vascular.unknowns.size()+i].x)*r.world.vascular.unknowns[i].initialAndScaling.y;}
void check(const Run& r,const RuntimeStateSnapshot& s,const Reference& ref,const Reference& start,double& maxError,double& maxConservation){
    const auto& l=r.world.vascular.layout;
    std::array<unsigned,6> indices={l.offsets.x,l.offsets.x+1,l.offsets.y,l.offsets.z,l.offsets.z+1,l.offsets.w};
    std::array<double,6> expected={ref.a,ref.b,ref.q,ref.amount[0],ref.amount[1],ref.amount[2]};
    for(unsigned e=0;e<2;++e){
        for(unsigned j=0;j<6;++j){auto i=indices[j];double v=value(r,s,i,e);need(std::isfinite(v),"nonfinite vascular state");if(j!=2)need(v>=0,"negative conserved variable");double err=std::abs(v-expected[j])/r.world.vascular.unknowns[i].initialAndScaling.y;maxError=std::max(maxError,err);need(err<8e-5,"FP64 reference disagreement row="+std::to_string(j)+" normalized_error="+std::to_string(err));}
        const double dv=std::abs(value(r,s,l.offsets.x,e)+value(r,s,l.offsets.x+1,e)-start.a-start.b)/(start.a+start.b);
        const double initialM=start.amount[0]+start.amount[1]+start.amount[2];
        const double dm=std::abs(value(r,s,l.offsets.z,e)+value(r,s,l.offsets.z+1,e)+value(r,s,l.offsets.w,e)-initialM)/initialM;
        maxConservation=std::max({maxConservation,dv,dm});need(dv<3e-5&&dm<3e-5,"closed volume/species conservation failed");
    }
}
void qualify(VascularNetworkSource n,const char* label,bool fem=false){
    constexpr unsigned steps=16;constexpr double dt=.01;
    Run run(sourceFor(n,dt,fem));auto start=run.state();auto ref=initial(n);const auto first=ref;
    double maxError=0,maxConservation=0;RuntimeStateSnapshot firstStep;
    for(unsigned i=0;i<steps;++i){run.step(i);ref=advance(ref,n,run.runtime.timestepSeconds());auto current=run.state();check(run,current,ref,first,maxError,maxConservation);if(i==0)firstStep=current;}
    auto evolved=run.state();need(!same(start.vascularState,evolved.vascularState),"state did not evolve");
    auto restored=run.runtime.restore(start);need(restored.encoded,restored.message);
    for(unsigned i=0;i<steps;++i)run.step(i);
    need(same(evolved.vascularState,run.state().vascularState),"snapshot replay differs");
    const auto before=run.state();run.step(steps,1);const auto rejected=run.state();const auto stride=run.world.vascular.unknowns.size();
    need(std::memcmp(before.vascularState.data()+stride,rejected.vascularState.data()+stride,stride*sizeof(nm_float4))==0,"rejected environment leaked future vascular state");
    need(std::memcmp(before.vascularState.data(),rejected.vascularState.data(),stride*sizeof(nm_float4))!=0,"accepted environment did not advance");
    auto corrupt=before;corrupt.vascularState[0].x=-1;auto denied=run.runtime.restore(corrupt);need(!denied.encoded,"negative volume restore accepted");need(same(rejected.vascularState,run.state().vascularState),"invalid restore changed state");
    run.step(steps+1,-1,1);const auto resetState=run.state();
    need(std::memcmp(firstStep.vascularState.data()+stride,resetState.vascularState.data()+stride,stride*sizeof(nm_float4))==0,"episode reset did not restore authored vascular origin");
    need(std::memcmp(rejected.vascularState.data(),resetState.vascularState.data(),stride*sizeof(nm_float4))!=0,"episode reset stalled other environment");
    std::cout<<"vascular_case="<<label<<" result=pass accepted_steps="<<steps<<" environments=2 fp64_normalized_max="<<maxError<<" relative_conservation_max="<<maxConservation<<" replay=bitwise isolated_rollback=pass invalid_restore=denied isolated_reset=pass fem_region="<<(fem?"bound_identity_only":"none")<<'\n';
}
void branched(){
    auto n=vascularFixture();n.species.push_back({10,"fixture:second-tracer",2e-6,1e-5});
    n.compartments[0].initialSpeciesAmounts={2e-6,1e-6};n.compartments[1].initialSpeciesAmounts={0,3e-6};
    auto third=n.compartments[0];third.stableIdentifier=7;third.anatomicalIdentifier="fixture:blood-c";third.initialVolume=1.5e-6;third.initialSpeciesAmounts={3e-6,.5e-6};n.compartments.push_back(third);
    auto e=n.connections[0];e.stableIdentifier=8;e.fromCompartment=3;e.toCompartment=7;n.connections.push_back(e);e.stableIdentifier=11;e.fromCompartment=7;e.toCompartment=2;n.connections.push_back(e);
    n.tissues[0].initialSpeciesAmounts={0,0};auto tissue=n.tissues[0];tissue.stableIdentifier=9;tissue.anatomicalIdentifier="fixture:second-organ";n.tissues.push_back(tissue);
    n.exchanges.push_back({12,2,9,10,3e-7,1});n.exchanges.push_back({13,7,5,10,2e-7,1.5});
    Run r(sourceFor(n,.01));auto start=r.state();auto totals=[&](const RuntimeStateSnapshot& s,unsigned env){
        std::array<double,3> t{};const auto l=r.world.vascular.layout;
        for(unsigned i=0;i<l.counts.x;++i)t[0]+=value(r,s,l.offsets.x+i,env);
        for(unsigned species=0;species<2;++species){for(unsigned i=0;i<l.counts.x;++i)t[species+1]+=value(r,s,l.offsets.z+i*2+species,env);for(unsigned i=0;i<l.counts.w;++i)t[species+1]+=value(r,s,l.offsets.w+i*2+species,env);}return t;};
    const auto initialTotals=totals(start,0);double worst=0;
    for(unsigned step=0;step<8;++step){r.step(step);auto s=r.state();for(unsigned env=0;env<2;++env){auto t=totals(s,env);for(unsigned i=0;i<3;++i){double err=std::abs(t[i]-initialTotals[i])/initialTotals[i];worst=std::max(worst,err);need(err<3e-5,"branched multiple-species conservation failed");}}}
    auto end=r.state();const auto l=r.world.vascular.layout;need(value(r,end,l.offsets.w)>0 && value(r,end,l.offsets.w+1)>0 && value(r,end,l.offsets.w+3)>0,"species exchange did not reach target reservoirs");
    need(r.runtime.restore(start).encoded,"branched restore failed");for(unsigned i=0;i<8;++i)r.step(i);need(same(end.vascularState,r.state().vascularState),"branched replay failed");
    std::cout<<"vascular_branched=pass compartments=3 edges=3 species=2 tissues=2 exchanges=3 relative_conservation_max="<<worst<<" replay=bitwise\n";
}
void refinement(){
    auto n=vascularFixture();n.exchanges[0].permeabilitySurface=1e-7;
    const double duration=.16;std::array<double,3> errors{};
    for(unsigned k=0;k<3;++k){double dt=.02/(1u<<k);unsigned steps=8u<<k;Run r(sourceFor(n,dt));for(unsigned i=0;i<steps;++i)r.step(i);
        auto s=r.state();double diff=value(r,s,r.world.vascular.layout.offsets.x)-value(r,s,r.world.vascular.layout.offsets.x+1);
        errors[k]=std::abs(diff-1e-6*std::exp(-2*duration));
    }
    need(errors[2]<errors[1] && errors[1]<errors[0] && errors[0]/errors[1]>1.7 && errors[1]/errors[2]>1.7,"hydraulic timestep refinement failed");
    std::cout<<"vascular_refinement=pass equal_duration_s="<<duration<<" errors_m3="<<errors[0]<<','<<errors[1]<<','<<errors[2]<<'\n';
}
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
    std::cout<<std::setprecision(10);
    if(argc==3 && std::string(argv[1])=="--human-input") {VascularNetworkSource n;std::string error;need(readHumanPhysiologyNetwork(argv[2],n,&error),error);need(n.compartments.size()==2&&n.connections.size()==1&&n.species.size()==1&&n.tissues.size()==1&&n.exchanges.size()==1,"reference requires two-pool fixture");qualify(n,"human_compiled_fixture");}
    else {need(argc==1,"usage: numi-matter-vascular-check [--human-input FIXTURE.json]");
        qualify(vascularFixture(),"resistive_exchange");auto reverse=vascularFixture();std::swap(reverse.compartments[0].initialVolume,reverse.compartments[1].initialVolume);qualify(reverse,"reverse_flow");
        auto inertial=vascularFixture();inertial.connections[0].inertance=1e7;qualify(inertial,"inertial_flow");qualify(vascularFixture(),"fem_region",true);branched();refinement();}
    std::cout<<"vascular_native_qualification=pass runtime_input=nmatterpack biological_calibration=unqualified\n";return 0;
}catch(const std::exception& e){std::cerr<<"vascular_native_qualification=failed reason="<<e.what()<<'\n';return 1;}}}
