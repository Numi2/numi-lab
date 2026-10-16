#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_fixture.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace numi::matter;
namespace {
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
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
    void step(unsigned index,int reject=-1,int reset=-1,int expectedFailure=-1) {
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
        for(unsigned e=0;e<2;++e)if(static_cast<int>(e)!=reject && static_cast<int>(e)!=expectedFailure) need(end.statuses[e].code==NM_STATUS_SUCCESS,
            "vascular step "+std::to_string(index)+" environment "+std::to_string(e)+" rejected code="+std::to_string(end.statuses[e].code)+
            " object="+std::to_string(end.statuses[e].objectIndex)+" failing_index="+std::to_string(end.statuses[e].failingIndex)+
            " diagnostics=["+std::to_string(end.statuses[e].diagnostics.x)+","+std::to_string(end.statuses[e].diagnostics.y)+","+
            std::to_string(end.statuses[e].diagnostics.z)+","+std::to_string(end.statuses[e].diagnostics.w)+"]");
        if(expectedFailure>=0)need(end.statuses[expectedFailure].code!=NM_STATUS_SUCCESS,"expected failure not reported");
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
    const double dp=pressure(old.a,n.compartments[0])-pressure(old.b,n.compartments[1]);
    const double storageSlope=dt*(1/n.compartments[0].compliance+1/n.compartments[1].compliance);
    double q;
    if(edge.flowLaw==VascularFlowLaw::oneWayOrifice){
        // Independently eliminate both backward-Euler volume equations:
        // q*q/CV^2 + storageSlope*q - dp = 0, q >= 0. The rationalized
        // positive root avoids cancellation; adverse pressure gives q=0.
        const double cv=edge.orificeCoefficient;
        q=dp<=0 ? 0 : 2*dp*cv/(std::sqrt(std::pow(storageSlope*cv,2)+4*dp)+storageSlope*cv);
    } else q=(dp+edge.inertance*old.q/dt)/(edge.resistance+edge.inertance/dt+storageSlope);
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
    for(unsigned i=0;i<steps;++i){
        run.step(i);ref=advance(ref,n,run.runtime.timestepSeconds());auto current=run.state();
        check(run,current,ref,first,maxError,maxConservation);
        if(n.connections[0].flowLaw==VascularFlowLaw::oneWayOrifice){
            const auto l=run.world.vascular.layout;
            for(unsigned env=0;env<2;++env){
                const double q=value(run,current,l.offsets.y,env);
                need(ref.q==0 ? q==0 : q>0,"valve did not reach the exact closed/open branch");
                if(ref.q==0 && ref.amount[1]==0)
                    need(value(run,current,l.offsets.z+1,env)==0,"closed valve leaked tracer into isolated blood pool");
                need(value(run,current,l.offsets.w,env)>0,"valve fixture did not exercise tissue exchange");
            }
        }
        if(i==0)firstStep=current;
    }
    auto evolved=run.state();need(!same(start.vascularState,evolved.vascularState),"state did not evolve");
    auto restored=run.runtime.restore(start);need(restored.encoded,restored.message);
    for(unsigned i=0;i<steps;++i)run.step(i);
    need(same(evolved.vascularState,run.state().vascularState),"snapshot replay differs");
    need(same(evolved.vascularClock,run.state().vascularClock),"snapshot replay clock differs");
    const auto before=run.state();run.step(steps,1);const auto rejected=run.state();const auto stride=run.world.vascular.unknowns.size();
    need(std::memcmp(before.vascularState.data()+stride,rejected.vascularState.data()+stride,stride*sizeof(nm_float4))==0,"rejected environment leaked future vascular state");
    need(std::memcmp(&before.vascularClock[1],&rejected.vascularClock[1],sizeof(NMVascularClockGPU))==0,"rejected future advanced clock");
    need(before.vascularClock[0].low!=rejected.vascularClock[0].low,"accepted clock stalled");
    need(std::memcmp(before.vascularState.data(),rejected.vascularState.data(),stride*sizeof(nm_float4))!=0,"accepted environment did not advance");
    auto corrupt=before;corrupt.vascularState[0].x=-1;auto denied=run.runtime.restore(corrupt);need(!denied.encoded,"negative volume restore accepted");need(same(rejected.vascularState,run.state().vascularState),"invalid restore changed state");
    run.step(steps+1,-1,1);const auto resetState=run.state();
    need(std::memcmp(firstStep.vascularState.data()+stride,resetState.vascularState.data()+stride,stride*sizeof(nm_float4))==0,"episode reset did not restore authored vascular origin");
    need(std::memcmp(&firstStep.vascularClock[1],&resetState.vascularClock[1],sizeof(NMVascularClockGPU))==0,"episode reset did not restore clock origin");
    need(std::memcmp(rejected.vascularState.data(),resetState.vascularState.data(),stride*sizeof(nm_float4))!=0,"episode reset stalled other environment");
    std::cout<<"vascular_case="<<label<<" result=pass accepted_steps="<<steps<<" environments=2 fp64_normalized_max="<<maxError<<" relative_conservation_max="<<maxConservation<<" replay=bitwise isolated_rollback=pass invalid_restore=denied isolated_reset=pass fem_region="<<(fem?"bound_identity_only":"none")<<'\n';
}
// Synthetic absolute-volume valve cases exercise the species product rule
// and tissue exchange independently of the hydraulics-only cardiac source.
void valveTransport(){
    auto closing=vascularFixture();
    auto& edge=closing.connections[0];
    edge.flowLaw=VascularFlowLaw::oneWayOrifice;
    edge.resistance=0;edge.inertance=0;edge.orificeCoefficient=1e-7;
    edge.initialFlow=2e-6;
    closing.compartments[0].initialVolume=1e-6;
    closing.compartments[1].initialVolume=2e-6;
    closing.exchanges[0].compartment=closing.compartments[0].stableIdentifier;
    // At this authored initial guess, q*pressureScale/flowScale = 2000 Pa,
    // q^2/CV^2-dp = 1400 Pa: the original min law selects its open branch.
    // Its unconstrained hydraulic Newton correction crosses q<0. Closing
    // must remove that edge from BOTH conservation rows and species tangents,
    // while the independent blood-A/tissue exchange remains active.
    qualify(closing,"valve_closing_species_exchange");
    auto opening=vascularFixture();
    opening.connections[0].flowLaw=VascularFlowLaw::oneWayOrifice;
    opening.connections[0].resistance=0;opening.connections[0].inertance=0;
    opening.connections[0].orificeCoefficient=1e-7;
    qualify(opening,"valve_opening_species_exchange");
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
void clockLifecycle(){
    Run r(sourceFor(vascularFixture(),.01));auto initial=r.state();
    need(initial.vascularClock.size()==2,"clock logical width");
    const auto quantum=std::bit_cast<std::int32_t>(r.world.vascular.layout.clock.x);
    const auto dt=static_cast<std::uint64_t>(std::ldexp(double(r.runtime.timestepSeconds()),-quantum));
    r.step(0);auto one=r.state();need(one.vascularClock[0].low==dt&&one.vascularClock[0].high==0,"clock did not advance exact authored dt");
    auto malformed=one;malformed.vascularClock.pop_back();need(!r.runtime.restore(malformed).encoded,"clock arity restore accepted");
    need(same(one.vascularClock,r.state().vascularClock),"malformed clock restore mutated state");
    auto huge=one;huge.vascularClock[0]={0xfffffffffffffffcul,17};huge.vascularClock[1]={123,0x100000001ul};
    need(r.runtime.restore(huge).encoded,"large clock restore failed");r.step(1);auto large=r.state();
    need(large.vascularClock[0].low==std::uint64_t(0xfffffffffffffffcul+dt)&&large.vascularClock[0].high==18,"runtime clock carry failed");
    need(large.vascularClock[1].low==123+dt&&large.vascularClock[1].high==0x100000001ul,"runtime high-word continuation failed");
    auto overflow=large;overflow.vascularClock[0]={std::numeric_limits<std::uint64_t>::max(),std::numeric_limits<std::uint64_t>::max()};
    need(r.runtime.restore(overflow).encoded,"overflow-boundary restore failed");r.step(2,-1,-1,0);auto denied=r.state();
    const auto stride=r.world.vascular.unknowns.size();
    need(std::memcmp(&overflow.vascularClock[0],&denied.vascularClock[0],sizeof(NMVascularClockGPU))==0&&std::memcmp(overflow.vascularState.data(),denied.vascularState.data(),stride*sizeof(nm_float4))==0,"overflow failed to roll back physical state and clock");
    need(denied.vascularClock[1].low==overflow.vascularClock[1].low+dt,"overflow contaminated healthy environment");
    need(r.runtime.restore(initial).encoded,"clock lifecycle initial restore failed");
    auto cb=[r.queue commandBuffer];EncodeRequest req;req.commandBuffer=(__bridge void*)cb;req.environmentStatuses=(__bridge void*)r.statuses;
    req.timestepSeconds=std::nextafter(r.runtime.timestepSeconds(),INFINITY);req.physicsSubsteps=1;req.runAdaptiveTransfer=false;
    const auto invalid=r.runtime.encode(req);need(!invalid.encoded,"unrepresentable timestep override accepted");
    need(same(initial.vascularClock,r.state().vascularClock)&&same(initial.vascularState,r.state().vascularState),"invalid timestep mutated authority");
    std::cout<<"vascular_clock_lifecycle=pass exact_dt_ticks="<<dt<<" carry=exact huge_snapshot=pass overflow_rollback=pass malformed_arity=denied timestep_override=denied\n";
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
void humanRegional(const std::string& path) {
    VascularNetworkSource network;
    std::string error;
    need(readHumanPhysiologyNetwork(path, network, &error), error);
    need(network.compartments.size() == 21u && network.connections.size() == 24u &&
         network.species.empty() && network.tissues.empty() && network.exchanges.empty(),
         "CVSim21 source graph topology changed");
    const auto nonzeroIdentity = [](const std::array<std::uint64_t, 4>& identity) {
        return std::any_of(identity.begin(), identity.end(), [](std::uint64_t value) {
            return value != 0u;
        });
    };
    need(nonzeroIdentity(network.contentIdentity) &&
         nonzeroIdentity(network.sourceIdentity) &&
         nonzeroIdentity(network.authoredIdentity),
         "CVSim21 source identities are absent");
    const auto compartment = [&](const char* suffix) {
        const std::string wanted = suffix;
        for (const auto& candidate : network.compartments) {
            if (candidate.anatomicalIdentifier.size() >= wanted.size() &&
                candidate.anatomicalIdentifier.compare(
                    candidate.anatomicalIdentifier.size() - wanted.size(),
                    wanted.size(), wanted) == 0) {
                return candidate.stableIdentifier;
            }
        }
        throw std::runtime_error(std::string("CVSim21 source compartment missing: ") + suffix);
    };
    network.species.push_back({1001u, "oxygen_amount", 1.0e-3, 1.0e-5});
    for (auto& candidate : network.compartments) {
        candidate.initialSpeciesAmounts = {0.20 * candidate.initialVolume};
    }
    struct Bed {
        const char* region;
        const char* arterial;
        const char* venous;
        double candidateVolume;
    };
    constexpr std::array<Bed, 7> beds{{
        {"right_lung", "pulmonary_arteries", "pulmonary_veins", 8.735690598687958e-05},
        {"left_lung", "pulmonary_arteries", "pulmonary_veins", 8.340118835074067e-05},
        {"right_kidney", "renal_arteries", "renal_veins", 1.0687769971363567e-04},
        {"left_kidney", "renal_arteries", "renal_veins", 1.0374115500169420e-04},
        {"stomach", "splanchnic_arteries", "splanchnic_veins", 5.664554569713276e-04},
        {"pancreas", "splanchnic_arteries", "splanchnic_veins", 1.2977908046225777e-04},
        {"liver", "splanchnic_arteries", "splanchnic_veins", 1.1080525283765639e-03},
    }};
    for (std::size_t i = 0; i < beds.size(); ++i) {
        const auto arterial = compartment(beds[i].arterial);
        const auto venous = compartment(beds[i].venous);
        need(arterial != venous, "regional bed arterial and venous owners alias");
        VascularTissueSource tissue;
        tissue.stableIdentifier = 2001u + static_cast<std::uint32_t>(i);
        tissue.anatomicalIdentifier = std::string("Human/BodyParts3D/") + beds[i].region;
        tissue.volume = beds[i].candidateVolume;
        tissue.initialSpeciesAmounts = {0.05 * tissue.volume};
        network.tissues.push_back(std::move(tissue));
        network.exchanges.push_back({
            3001u + static_cast<std::uint32_t>(i), arterial,
            2001u + static_cast<std::uint32_t>(i), 1001u, 1.0e-5, 1.0
        });
    }
    constexpr unsigned steps = 512u;
    constexpr unsigned rejectedStep = 37u;
    constexpr double timestep = 1.25e-5;
    constexpr double bloodDensity = 1060.0;
    Run run(sourceFor(std::move(network), timestep));
    need(run.world.vascular.layout.counts.x == 21u &&
         run.world.vascular.layout.counts.z == 1u &&
         run.world.vascular.layout.counts.w == beds.size(),
         "regional source layout lost blood or tissue rows");
    const auto initial = run.state();
    need(std::abs(run.runtime.timestepSeconds() - timestep) < 1.0e-5 * timestep,
         "regional exchange did not retain the exact 12.5 us timestep");
    const auto total = [&](const RuntimeStateSnapshot& snapshot, unsigned environment) {
        struct Totals {
            double volume = 0.0;
            double bloodMass = 0.0;
            double oxygen = 0.0;
        };
        Totals result;
        const auto& layout = run.world.vascular.layout;
        for (std::uint32_t i = 0; i < layout.counts.x; ++i) {
            result.volume += value(run, snapshot, layout.offsets.x + i, environment);
            result.oxygen += value(run, snapshot,
                layout.offsets.z + i * layout.counts.z, environment);
        }
        result.bloodMass = result.volume * bloodDensity;
        for (std::uint32_t i = 0; i < layout.counts.w; ++i) {
            result.oxygen += value(run, snapshot,
                layout.offsets.w + i * layout.counts.z, environment);
        }
        return result;
    };
    const auto initialTotals = total(initial, 0u);
    need(initialTotals.volume > 0.005 && initialTotals.oxygen > 0.0,
         "regional source initial conserved totals are empty");
    double maximumVolumeError = 0.0;
    double maximumBloodMassError = 0.0;
    double maximumOxygenError = 0.0;
    const auto verify = [&](const RuntimeStateSnapshot& snapshot) {
        for (unsigned environment = 0; environment < 2u; ++environment) {
            const auto current = total(snapshot, environment);
            const double volumeError = std::abs(current.volume - initialTotals.volume) /
                initialTotals.volume;
            const double bloodMassError =
                std::abs(current.bloodMass - initialTotals.bloodMass) /
                initialTotals.bloodMass;
            const double oxygenError = std::abs(current.oxygen - initialTotals.oxygen) /
                initialTotals.oxygen;
            maximumVolumeError = std::max(maximumVolumeError, volumeError);
            maximumBloodMassError = std::max(maximumBloodMassError, bloodMassError);
            maximumOxygenError = std::max(maximumOxygenError, oxygenError);
            need(volumeError < 5.0e-5 && bloodMassError < 5.0e-5 &&
                     oxygenError < 5.0e-5,
                 "regional blood mass, volume, or oxygen amount was not conserved");
        }
    };
    const auto runSequence = [&]() {
        for (unsigned step = 0; step < steps; ++step) {
            if (step == rejectedStep) {
                const auto before = run.state();
                run.step(step, 0);
                const auto after = run.state();
                const auto stride = run.world.vascular.unknowns.size();
                need(std::memcmp(before.vascularState.data(), after.vascularState.data(),
                                 stride * sizeof(nm_float4)) == 0 &&
                     std::memcmp(before.vascularClock.data(), after.vascularClock.data(),
                                 sizeof(NMVascularClockGPU)) == 0,
                     "rejected regional candidate changed accepted environment");
            } else {
                run.step(step);
            }
            verify(run.state());
        }
        return run.state();
    };
    const auto finalState = runSequence();
    need(!same(initial.vascularState, finalState.vascularState),
         "regional blood/tissue state did not evolve");
    const auto tissueOffset = run.world.vascular.layout.offsets.w;
    need(std::abs(value(run, finalState, tissueOffset, 0u) -
                  value(run, initial, tissueOffset, 0u)) > 0.0,
         "regional tissue exchange did not change a tissue amount");
    const auto quantum = std::bit_cast<std::int32_t>(run.world.vascular.layout.clock.x);
    const auto ticks = static_cast<std::uint64_t>(
        std::ldexp(run.runtime.timestepSeconds(), -quantum));
    need(finalState.vascularClock[0].low == (steps - 1u) * ticks &&
         finalState.vascularClock[1].low == steps * ticks &&
         finalState.vascularClock[0].high == 0u &&
         finalState.vascularClock[1].high == 0u,
         "regional exact clock did not match accepted/rejected steps c0=" +
             std::to_string(finalState.vascularClock[0].low) + " c1=" +
             std::to_string(finalState.vascularClock[1].low) + " ticks=" +
             std::to_string(ticks) + " h0=" +
             std::to_string(finalState.vascularClock[0].high) + " h1=" +
             std::to_string(finalState.vascularClock[1].high));
    need(run.runtime.restore(initial).encoded, "regional snapshot restore failed");
    const auto replayState = runSequence();
    need(same(finalState.vascularState, replayState.vascularState) &&
         same(finalState.vascularClock, replayState.vascularClock),
         "regional source replay was not bitwise identical");
    std::cout << "human_regional_exchange=pass source_compartments=21 source_edges=24 "
              << "regional_beds=7 attempted_steps=" << steps
              << " accepted_environment0=" << (steps - 1u)
              << " rejected_step=" << rejectedStep
              << " timestep_ns=12500 blood_density_candidate_kg_m3=" << bloodDensity
              << " volume_conservation_max=" << maximumVolumeError
              << " blood_mass_conservation_max=" << maximumBloodMassError
              << " oxygen_conservation_max=" << maximumOxygenError
              << " rollback=bitwise replay=bitwise source_identity=retained "
                 "density_material_calibration=unqualified\n";
}
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
    std::cout<<std::setprecision(10);
    if(argc==3 && std::string(argv[1])=="--human-input") {VascularNetworkSource n;std::string error;need(readHumanPhysiologyNetwork(argv[2],n,&error),error);need(n.compartments.size()==2&&n.connections.size()==1&&n.species.size()==1&&n.tissues.size()==1&&n.exchanges.size()==1,"reference requires two-pool fixture");qualify(n,"human_compiled_fixture");}
    else if(argc==3 && std::string(argv[1])=="--human-regional") {humanRegional(argv[2]);}
    else {need(argc==1,"usage: numi-matter-vascular-check [--human-input FIXTURE.json | --human-regional CVSIM21.json]");
        qualify(vascularFixture(),"resistive_exchange");auto reverse=vascularFixture();std::swap(reverse.compartments[0].initialVolume,reverse.compartments[1].initialVolume);qualify(reverse,"reverse_flow");
        auto inertial=vascularFixture();inertial.connections[0].inertance=1e7;qualify(inertial,"inertial_flow");qualify(vascularFixture(),"fem_region",true);valveTransport();branched();refinement();clockLifecycle();}
    std::cout<<"vascular_native_qualification=pass runtime_input=nmatterpack biological_calibration=unqualified\n";return 0;
}catch(const std::exception& e){std::cerr<<"vascular_native_qualification=failed reason="<<e.what()<<'\n';return 1;}}}
