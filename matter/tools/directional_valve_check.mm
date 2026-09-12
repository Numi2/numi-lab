#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/detail.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_fixture.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace numi::matter;
namespace {
// Only the valve coefficients and declared mmHg conversion reproduce source
// numbers. Volumes, compliance, tracer, tissue and loads are synthetic controls.
constexpr double mmHgSI=133.322387415, resistanceSI=mmHgSI*1e6;
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));}
WorldSource sourceFor(VascularNetworkSource n,double dt=.01){
 WorldSource s;s.environmentCount=2;s.frameTimestep=dt;s.gravity={0,0,0};s.vascular=std::move(n);
 s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
 return s;
}
VascularNetworkSource fixture(double forward,double reverse,double sign=1,double initialQ=0){
 auto n=vascularFixture();
 n.contentIdentity={0x646972656374696full,0x6e616c2d76616c76ull,0x652d666978747572ull,0x652d303030303031ull};
 n.sourceIdentity={1,2,3,4};n.authoredIdentity={5,6,7,8};
 for(auto& c:n.compartments){c.referenceVolume=1e-4;c.initialVolume=1e-4;c.volumeScale=1e-4;c.compliance=1e-8;c.volumeResidualTolerance=1e-6;c.referencePressure=0;}
 n.compartments[0].referencePressure=sign*1000;
 n.compartments[0].initialSpeciesAmounts={2e-6};n.compartments[1].initialSpeciesAmounts={7e-6};
 n.species[0].amountResidualTolerance=1e-6;n.tissues[0].volume=5e-5;n.tissues[0].initialSpeciesAmounts={3e-6};
 n.exchanges[0].permeabilitySurface=2e-5;
 auto& e=n.connections[0];e.flowLaw=VascularFlowLaw::directionalResistance;e.resistance=forward*resistanceSI;
 e.reverseResistance=reverse*resistanceSI;e.initialFlow=initialQ;e.flowScale=1e-4;e.pressureScale=1000;e.flowResidualTolerance=1e-6;
 return n;
}
CompileResult compiled(const WorldSource& s){
 auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});
 std::string d;for(const auto& x:c.diagnostics)d+=x.message+"; ";
 need(c.succeeded(),"compile: "+d);return c;
}
std::filesystem::path temporary(const char* label){
 return std::filesystem::temp_directory_path()/(std::string("numi-directional-")+label+"-"+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
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
    } else if(edge.flowLaw==VascularFlowLaw::directionalResistance) q=dp/((dp<0?edge.reverseResistance:edge.resistance)+storageSlope);
    else q=(dp+edge.inertance*old.q/dt)/(edge.resistance+edge.inertance/dt+storageSlope);
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

unsigned cpuCount=0;
void resealedPackage(const CompileResult& good,CompiledWorld changed,bool accept){
 const auto path=temporary("reseal");std::string error;need(writePackage(good,path,&error),error);
 std::ifstream in(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});in.close();
 struct Header{std::array<char,16> magic;std::uint32_t version,endian,abi,sections;std::uint64_t physics,fingerprint,hash;};
 struct Section{std::uint32_t id,size;std::uint64_t count,bytes,hash;};
 static_assert(sizeof(Header)==56&&sizeof(Section)==32);
 Header h{};need(bytes.size()>=sizeof(h),"short generated package");std::memcpy(&h,bytes.data(),sizeof(h));
 std::size_t cursor=sizeof(h);bool found=false;
 for(unsigned i=0;i<h.sections;++i){need(cursor+sizeof(Section)<=bytes.size(),"section header overflow");Section s{};std::memcpy(&s,bytes.data()+cursor,sizeof(s));
  const auto start=cursor+sizeof(s);need(s.bytes<=bytes.size()-start,"section extent overflow");
  if(s.id==47u){need(s.size==sizeof(NMVascularConnectionGPU)&&s.count==changed.vascular.connections.size(),"connection section schema changed");
   need(s.bytes==s.count*s.size,"connection section arity");std::memcpy(bytes.data()+start,changed.vascular.connections.data(),s.bytes);
   s.hash=detail::hashBytes(bytes.data()+start,s.bytes);std::memcpy(bytes.data()+cursor,&s,sizeof(s));found=true;}
  cursor=start+s.bytes;
 }
 need(found&&cursor==bytes.size(),"connection section absent or trailing payload");
 changed.fingerprint=compiledWorldFingerprint(changed);h.fingerprint=changed.fingerprint;h.hash=0;h.hash=detail::hashBytes(&h,sizeof(h));std::memcpy(bytes.data(),&h,sizeof(h));
 {std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write(bytes.data(),bytes.size());need(bool(out),"reseal write failed");}
 CompiledWorld output=good.world;const auto before=output.fingerprint;const bool loaded=readPackage(path,output,nullptr,&error);std::filesystem::remove(path);
 need(loaded==accept,"resealed package admission: "+error);
 if(!accept){need(output.fingerprint==before,"rejected package mutated destination");need(error.find("layout")!=std::string::npos,"forged package rejected by checksum, not layout: "+error);}++cpuCount;
}
VascularNetworkSource three(bool cycle,double r=0){
 auto n=fixture(r,10000);auto c=n.compartments[1];c.stableIdentifier=7;c.anatomicalIdentifier="fixture:third";c.referencePressure=-1000;n.compartments.push_back(c);
 auto e=n.connections[0];e.stableIdentifier=8;e.fromCompartment=3;e.toCompartment=7;n.connections.push_back(e);
 if(cycle){e.stableIdentifier=9;e.fromCompartment=7;e.toCompartment=2;n.connections.push_back(e);}return n;
}
void cpuChecks(){
 const auto good=compiled(sourceFor(fixture(0,10000,-1,-1e-8)));const auto& w=good.world;
 need(w.objects.empty()&&w.materials.empty()&&w.fem.nodes.empty(),"invented mechanical owner");
 need(w.vascular.connections[0].physical.x==0&&w.vascular.connections[0].directional.x==float(10000*resistanceSI),"SI coefficient cook mismatch");
 need(w.vascular.unknowns[w.vascular.layout.offsets.y].initialAndScaling.x<0,"signed initial flow lost");++cpuCount;
 resealedPackage(good,w,true);
 auto negative=[&](const char* label,const std::function<void(WorldSource&)>& edit){auto s=sourceFor(fixture(0,10000));edit(s);auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(!c.succeeded(),std::string("source accepted ")+label);++cpuCount;};
 negative("negative forward",[](auto& s){s.vascular.connections[0].resistance=-1;});
 negative("zero reverse",[](auto& s){s.vascular.connections[0].reverseResistance=0;});
 negative("reverse below forward",[](auto& s){s.vascular.connections[0].resistance=2e12;});
 negative("nonfinite reverse",[](auto& s){s.vascular.connections[0].reverseResistance=INFINITY;});
 negative("nonfinite forward",[](auto& s){s.vascular.connections[0].resistance=NAN;});
 negative("inertance",[](auto& s){s.vascular.connections[0].inertance=1;});
 negative("orifice coefficient",[](auto& s){s.vascular.connections[0].orificeCoefficient=1;});
 negative("pressure floor",[](auto& s){s.vascular.connections[0].downstreamPressureFloor=1;});
 negative("positive forward underflows to ideal",[](auto& s){s.vascular.connections[0].resistance=1e-100;});
 negative("reverse underflow",[](auto& s){s.vascular.connections[0].reverseResistance=1e-100;});
 negative("flow NaN",[](auto& s){s.vascular.connections[0].initialFlow=NAN;});
 negative("old law reverse payload",[](auto& s){s.vascular.connections[0].flowLaw=VascularFlowLaw::resistanceInertance;s.vascular.connections[0].resistance=1;});
 negative("unknown law",[](auto& s){s.vascular.connections[0].flowLaw=static_cast<VascularFlowLaw>(5);});
 auto tree=compiled(sourceFor(three(false)));need(tree.world.vascular.connections.size()==2,"zero forward tree lost");++cpuCount;
 negative("zero cycle",[](auto& s){s.vascular=three(true);});
 negative("zero parallel",[](auto& s){auto e=s.vascular.connections[0];e.stableIdentifier=8;s.vascular.connections.push_back(e);});
 negative("zero antiparallel",[](auto& s){auto e=s.vascular.connections[0];e.stableIdentifier=8;std::swap(e.fromCompartment,e.toCompartment);s.vascular.connections.push_back(e);});
 auto boundedCycle=compiled(sourceFor(three(true,.05)));++cpuCount;
 const auto rejectCooked=[&](const char* label,const std::function<void(CompiledWorld&)>& edit){
  auto bad=w;edit(bad);bad.fingerprint=compiledWorldFingerprint(bad);std::string error;
  need(!validateCompiledWorldLayout(bad,&error),std::string("cooked accepted ")+label);++cpuCount;resealedPackage(good,bad,false);
 };
 rejectCooked("zero reverse",[](auto& c){c.vascular.connections[0].directional.x=0;});
 rejectCooked("reverse below forward",[](auto& c){c.vascular.connections[0].physical.x=2e12f;});
 rejectCooked("unused reverse padding",[](auto& c){c.vascular.connections[0].directional.y=1;});
 rejectCooked("nonfinite reverse",[](auto& c){c.vascular.connections[0].directional.x=NAN;});
 rejectCooked("inertance",[](auto& c){c.vascular.connections[0].physical.y=1;});
 rejectCooked("orifice",[](auto& c){c.vascular.connections[0].physical.z=1;});
 rejectCooked("floor",[](auto& c){c.vascular.connections[0].physical.w=1;});
 rejectCooked("old law payload",[](auto& c){c.vascular.connections[0].identity.w=0;c.vascular.connections[0].physical.x=1;});
 auto badCycle=boundedCycle.world;for(auto& e:badCycle.vascular.connections)e.physical.x=0;
 badCycle.fingerprint=compiledWorldFingerprint(badCycle);std::string error;need(!validateCompiledWorldLayout(badCycle,&error),"cooked zero cycle accepted");++cpuCount;
 resealedPackage(boundedCycle,badCycle,false);
 auto changed=w;changed.vascular.connections[0].directional.x*=.5f;need(compiledWorldFingerprint(changed)!=w.fingerprint,"reverse coefficient absent from identity");++cpuCount;
 // Writer rejection must preserve an existing destination byte-for-byte.
 const auto path=temporary("atomic");{std::ofstream out(path);out<<"preserved";}
 auto invalid=good;invalid.world.vascular.connections[0].directional.x=0;invalid.world.fingerprint=compiledWorldFingerprint(invalid.world);
 need(!writePackage(invalid,path,&error),"writer accepted invalid direction");std::ifstream kept(path);std::string text;kept>>text;std::filesystem::remove(path);need(text=="preserved","invalid writer destroyed output");++cpuCount;
 std::cout<<"directional_compiler=pass checks="<<cpuCount<<" zero_forward_tree=pass cycles=denied resealed_packages=checked signed_initial_flow=pass positive_underflow=denied SI_Pa_per_mmHg="<<mmHgSI<<" source_Rr10000_cooked="<<w.vascular.connections[0].directional.x<<"\n";
}
VascularNetworkSource cookedReference(const Run& run,VascularNetworkSource n){
 const auto& v=run.world.vascular;
 for(unsigned i=0;i<2;++i){auto& c=n.compartments[i];const auto& p=v.compartments[i].compliance;
  c.referenceVolume=p.x;c.referencePressure=p.y;c.compliance=p.z;c.externalPressure=p.w;
  c.initialVolume=double(float(v.unknowns[i].initialAndScaling.x/v.unknowns[i].initialAndScaling.y))*v.unknowns[i].initialAndScaling.y;
  const auto& a=v.unknowns[v.layout.offsets.z+i].initialAndScaling;c.initialSpeciesAmounts[0]=double(float(a.x/a.y))*a.y;}
 auto& e=n.connections[0];e.resistance=v.connections[0].physical.x;e.reverseResistance=v.connections[0].directional.x;e.inertance=v.connections[0].physical.y;
 const auto& q=v.unknowns[v.layout.offsets.y].initialAndScaling;e.initialFlow=double(float(q.x/q.y))*q.y;
 n.tissues[0].volume=v.tissues[0].physical.x;const auto& t=v.unknowns[v.layout.offsets.w].initialAndScaling;n.tissues[0].initialSpeciesAmounts[0]=double(float(t.x/t.y))*t.y;
 n.exchanges[0].permeabilitySurface=v.exchanges[0].physical.x;n.exchanges[0].partitionCoefficient=v.exchanges[0].physical.y;return n;
}
void qualify(VascularNetworkSource authored,const char* label){
 Run r(sourceFor(authored));const auto n=cookedReference(r,authored);auto ref=initial(n),origin=ref;
 const auto start=r.state();RuntimeStateSnapshot first;double error=0,conservation=0,flowRelative=0,lawError=0;
 constexpr unsigned steps=16;
 for(unsigned i=0;i<steps;++i){r.step(i);ref=advance(ref,n,r.runtime.timestepSeconds());const auto s=r.state();check(r,s,ref,origin,error,conservation);
  const auto l=r.world.vascular.layout;
  for(unsigned env=0;env<2;++env){const double q=value(r,s,l.offsets.y,env);if(std::abs(ref.q)>1e-10){need(q*ref.q>0,"signed flow direction lost");
    const double relative=std::abs(q-ref.q)/std::abs(ref.q);flowRelative=std::max(flowRelative,relative);need(relative<3e-4,"signed flow reference relative error="+std::to_string(relative));}
   const auto pressure=[&](unsigned c){return n.compartments[c].referencePressure+n.compartments[c].externalPressure+(value(r,s,c,env)-n.compartments[c].referenceVolume)/n.compartments[c].compliance;};
   const double dp=pressure(0)-pressure(1),R=q<0?n.connections[0].reverseResistance:n.connections[0].resistance;
   const double e=std::abs(R*q-dp)/1000;lawError=std::max(lawError,e);need(e<3e-5,"directional pressure equation mismatch");
  }if(i==0)first=s;
 }
 const auto evolved=r.state();need(r.runtime.restore(start).encoded,"initial restore");
 for(unsigned i=0;i<steps;++i)r.step(i);need(same(evolved.vascularState,r.state().vascularState)&&same(evolved.vascularClock,r.state().vascularClock),"replay mismatch");
 const auto before=r.state();r.step(steps,1);const auto rejected=r.state();const auto stride=r.world.vascular.unknowns.size();
 need(std::memcmp(before.vascularState.data()+stride,rejected.vascularState.data()+stride,stride*sizeof(nm_float4))==0&&std::memcmp(&before.vascularClock[1],&rejected.vascularClock[1],sizeof(NMVascularClockGPU))==0,"rollback leaked state/clock");
 need(before.vascularClock[0].low!=rejected.vascularClock[0].low,"healthy clock stalled");
 auto negative=rejected;negative.vascularState[r.world.vascular.layout.offsets.y].x=-.125f;
 need(r.runtime.restore(negative).encoded&&same(negative.vascularState,r.state().vascularState),"finite negative flow restore denied");
 need(r.runtime.restore(rejected).encoded,"restore checkpoint");
 const auto deny=[&](const char* why,const std::function<void(RuntimeStateSnapshot&)>& edit){auto b=rejected;edit(b);need(!r.runtime.restore(b).encoded,std::string("invalid restore ")+why);need(same(rejected.vascularState,r.state().vascularState),"invalid restore mutated state");};
 deny("negative volume",[](auto& s){s.vascularState[0].x=-1;});
 deny("nonfinite flow",[&](auto& s){s.vascularState[r.world.vascular.layout.offsets.y].x=NAN;});
 deny("negative amount",[&](auto& s){s.vascularState[r.world.vascular.layout.offsets.z].x=-1;});
 r.step(steps+1,-1,1);const auto reset=r.state();
 need(std::memcmp(first.vascularState.data()+stride,reset.vascularState.data()+stride,stride*sizeof(nm_float4))==0&&std::memcmp(&first.vascularClock[1],&reset.vascularClock[1],sizeof(NMVascularClockGPU))==0,"episode reset mismatch");
 std::cout<<"directional_case="<<label<<" result=pass steps="<<steps<<" environments=2 fp64_normalized_max="<<error<<" signed_flow_relative_max="<<flowRelative<<" conservation_relative_max="<<conservation<<" pressure_normalized_max="<<lawError<<" replay=bitwise rollback=isolated reset=bitwise signed_restore=pass invalid_restore=denied\n";
}

template<class T> id<MTLBuffer> buffer(id<MTLDevice> device,const std::vector<T>& v){
 T zero{};const void* data=v.empty()?&zero:v.data();const auto bytes=v.empty()?sizeof(T):v.size()*sizeof(T);
 auto b=[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];need(b!=nil,"direct buffer allocation");return b;
}
id<MTLComputePipelineState> productionPipeline(id<MTLDevice> device,id<MTLLibrary> library,NSString* name){
 NSString* qualified=[@"numi_matter_metal::" stringByAppendingString:name];
 auto function=[library newFunctionWithName:qualified];
 need(function!=nil,std::string("production function missing: ")+qualified.UTF8String);
 NSError* error=nil;auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];
 need(pipeline!=nil,std::string("production pipeline failed: ")+qualified.UTF8String+(error?std::string(": ")+error.localizedDescription.UTF8String:""));return pipeline;
}
struct Direct {
 Run run;id<MTLLibrary> library;id<MTLComputePipelineState> residual,op,prepareElastance;
 explicit Direct(VascularNetworkSource n):run(sourceFor(std::move(n))){
  NSError* error=nil;library=[run.device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];need(library!=nil,"load production metallib");
  residual=productionPipeline(run.device,library,@"nm_vascular_residual");
  op=productionPipeline(run.device,library,@"nm_vascular_operator");prepareElastance=productionPipeline(run.device,library,@"nm_vascular_prepare_elastance");need(residual&&op&&prepareElastance,"production equation kernels unavailable");
 }
 void encodeElastance(id<MTLCommandBuffer> command,const NMMatterDispatchGPU& dispatch,id<MTLBuffer> compartments,id<MTLBuffer> elastance,id<MTLBuffer> statuses){
  const auto& graph=run.world.vascular.layout;auto clock=buffer(run.device,run.state().vascularClock);
  auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:prepareElastance];
  [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[encoder setBytes:&graph length:sizeof(graph) atIndex:1];
  [encoder setBuffer:compartments offset:0 atIndex:2];[encoder setBuffer:clock offset:0 atIndex:3];
  [encoder setBuffer:elastance offset:0 atIndex:4];[encoder setBuffer:statuses offset:0 atIndex:5];
  [encoder dispatchThreads:MTLSizeMake(dispatch.environmentCount*graph.counts.x,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
 }
 std::vector<double> evaluate(const std::vector<nm_float4>& current,const std::vector<nm_float4>& direction,bool derivative){
  const auto& w=run.world;const auto& v=w.vascular;const unsigned N=unsigned(v.unknowns.size());
  need(current.size()==2*N&&direction.size()==2*N,"direct state width");
  NMFGMRESLayoutGPU layout{};layout.vascularBase=3;layout.vascularUnknownCount=N;layout.unknownCount=3+2*N;
  NMMicrostepGPU micro{};micro.time.x=run.runtime.timestepSeconds();
  std::vector<nm_float4> lifted(layout.unknownCount);std::copy(direction.begin(),direction.end(),lifted.begin()+3);
  std::vector<nm_float4> accepted(2*N);for(unsigned e=0;e<2;++e)for(unsigned i=0;i<N;++i)accepted[e*N+i].x=v.unknowns[i].initialAndScaling.x/v.unknowns[i].initialAndScaling.y;
  std::array<id<MTLBuffer>,28> b{};
  b[3]=buffer(run.device,v.unknowns);b[4]=buffer(run.device,v.compartments);b[5]=buffer(run.device,v.connections);
  b[6]=buffer(run.device,v.tissues);b[7]=buffer(run.device,v.exchanges);b[8]=buffer(run.device,v.connectionIncidence);b[9]=buffer(run.device,v.connectionRanges);
  b[10]=buffer(run.device,v.bloodExchangeIncidence);b[11]=buffer(run.device,v.bloodExchangeRanges);b[12]=buffer(run.device,v.tissueExchangeIncidence);b[13]=buffer(run.device,v.tissueExchangeRanges);
  b[14]=buffer(run.device,accepted);b[15]=buffer(run.device,current);b[16]=buffer(run.device,lifted);
  b[17]=buffer(run.device,std::vector<nm_float4>(layout.unknownCount,nm_float4{17,18,19,20}));
  b[18]=buffer(run.device,std::vector<NMFGMRESStateGPU>(2));b[19]=buffer(run.device,std::vector<NMMatterStatusGPU>(2));
  b[20]=buffer(run.device,std::vector<float>(2*v.compartments.size()));b[21]=buffer(run.device,std::vector<unsigned>(2*N));
  b[23]=buffer(run.device,v.cavities);b[24]=buffer(run.device,v.cavityFaces);b[25]=buffer(run.device,v.compartmentCavity);
  b[26]=buffer(run.device,std::vector<NMFEMNodeStateGPU>(1));b[27]=buffer(run.device,std::vector<NMFEMNodeStateGPU>(1));
  const unsigned useWorking=0;auto command=[run.queue commandBuffer];encodeElastance(command,w.dispatch,b[4],b[20],b[19]);auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:derivative?op:residual];
  [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&v.layout length:sizeof(v.layout) atIndex:1];[encoder setBytes:&micro length:sizeof(micro) atIndex:2];
  for(unsigned i=3;i<=27;++i)if(i!=22)[encoder setBuffer:b[i] offset:0 atIndex:i];
  [encoder setBytes:&useWorking length:sizeof(useWorking) atIndex:22];[encoder setBytes:&layout length:sizeof(layout) atIndex:30];
  [encoder dispatchThreads:MTLSizeMake(2*N,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
  need(command.status==MTLCommandBufferStatusCompleted,"direct command failed");
  for(unsigned e=0;e<2;++e){const auto status=static_cast<NMMatterStatusGPU*>(b[19].contents)[e];need(status.code==NM_STATUS_SUCCESS,"direct equation rejected code="+std::to_string(status.code));}
  const auto* values=static_cast<nm_float4*>(b[17].contents);for(unsigned i=0;i<3;++i)need(values[i].x==17&&values[i].y==18&&values[i].z==19&&values[i].w==20,"direct kernel corrupted prefix");
  std::vector<double> result(2*N);for(unsigned i=0;i<2*N;++i){const auto x=values[3+i];need(std::isfinite(x.x)&&x.y==0&&x.z==0&&x.w==0,"direct malformed output");result[i]=(derivative?1.:-1.)*x.x;}return result;
 }
};
// Independent physical residual and selected generalized tangent. At Q0 the
// directional branch selects the SAME donor for every coupled transport row.
std::vector<double> oracle(const Direct& direct,const std::vector<nm_float4>& x,const std::vector<nm_float4>& d,bool tangent){
 const auto& v=direct.run.world.vascular;const unsigned N=unsigned(v.unknowns.size());const double dt=direct.run.runtime.timestepSeconds();std::vector<double> out(2*N);
 for(unsigned env=0;env<2;++env){
  const auto val=[&](unsigned r){return double(x[env*N+r].x)*v.unknowns[r].initialAndScaling.y;};
  const auto delta=[&](unsigned r){return double(d[env*N+r].x)*v.unknowns[r].initialAndScaling.y;};
  const auto old=[&](unsigned r){const auto p=v.unknowns[r].initialAndScaling;return double(float(p.x/p.y))*p.y;};
  const unsigned flow=v.layout.offsets.y,amount=v.layout.offsets.z,tissue=v.layout.offsets.w;
  const auto p=[&](unsigned c){const auto a=v.compartments[c].compliance;return a.w+a.y+(val(c)-a.x)/a.z;};
  const double q=val(flow),dq=delta(flow),dp=p(0)-p(1);const bool reverse=q<0||(q==0&&dp<0);
  const double R=reverse?v.connections[0].directional.x:v.connections[0].physical.x;
  out[env*N]=tangent?delta(0)+dt*dq:val(0)-old(0)+dt*q;
  out[env*N+1]=tangent?delta(1)-dt*dq:val(1)-old(1)-dt*q;
  out[env*N+flow]=tangent?R*dq-delta(0)/v.compartments[0].compliance.z+delta(1)/v.compartments[1].compliance.z:R*q-dp;
  const unsigned donor=reverse?1:0;const double C=val(amount+donor)/val(donor);
  const double flux=tangent?dq*C+q*(delta(amount+donor)-C*delta(donor))/val(donor):q*C;
  out[env*N+amount]=(tangent?delta(amount):val(amount)-old(amount))+dt*flux;
  out[env*N+amount+1]=(tangent?delta(amount+1):val(amount+1)-old(amount+1))-dt*flux;
  if(!v.tissues.empty()){
   const auto ex=v.exchanges[0];const unsigned pool=ex.identity.y;const double b=val(amount+pool)/val(pool),back=val(tissue)/(v.tissues[0].physical.x*double(ex.physical.y));
   const double J=tangent?double(ex.physical.x)*((delta(amount+pool)-b*delta(pool))/val(pool)-delta(tissue)/(v.tissues[0].physical.x*double(ex.physical.y))):double(ex.physical.x)*(b-back);
   out[env*N+amount+pool]+=dt*J;out[env*N+tissue]=(tangent?delta(tissue):val(tissue)-old(tissue))-dt*J;
  }
  for(unsigned i=0;i<N;++i)out[env*N+i]/=v.unknowns[i].initialAndScaling.z;
 }return out;
}
void directChecks(){
 double worst=0,fdWorst=0;
 for(unsigned test=0;test<4;++test){
  const bool zero=test<3;auto n=fixture(test==1?0:.05,test==1?10000:1000,test==0?-1:test==2?0:1);
  if(zero){n.tissues.clear();n.exchanges.clear();}
  Direct direct(n);auto x=direct.run.state().vascularState;const auto& v=direct.run.world.vascular;const unsigned N=unsigned(v.unknowns.size()),q=v.layout.offsets.y;
  std::vector<nm_float4> d(x.size());
  for(unsigned e=0;e<2;++e){const auto base=e*N;
   d[base+q].x=test==0?-1.f:test==3?2e-3f:1.f;
   if(!zero){x[base+q].x=e?-.01f:.01f;d[base].x=.004f;d[base+1].x=-.003f;d[base+v.layout.offsets.z].x=.015f;d[base+v.layout.offsets.z+1].x=-.02f;d[base+v.layout.offsets.w].x=.01f;}
  }
  const auto r=direct.evaluate(x,d,false),j=direct.evaluate(x,d,true),expectedR=oracle(direct,x,d,false),expectedJ=oracle(direct,x,d,true);
  for(unsigned i=0;i<r.size();++i){const double re=std::abs(r[i]-expectedR[i])/std::max(1.,std::abs(expectedR[i])),je=std::abs(j[i]-expectedJ[i])/std::max(1.,std::abs(expectedJ[i]));worst=std::max({worst,re,je});need(re<2e-5&&je<2e-5,"production residual/Jv oracle mismatch test="+std::to_string(test)+" row="+std::to_string(i));}
  // Interior flow perturbations stay on their branch and resolve the high-R FP32 residual.
  const float h=.001f;auto plus=x,minus=x;for(unsigned i=0;i<x.size();++i){plus[i].x+=h*d[i].x;if(!zero)minus[i].x-=h*d[i].x;}
  const auto rp=direct.evaluate(plus,d,false),rm=zero?r:direct.evaluate(minus,d,false);
  for(unsigned i=0;i<j.size();++i){const double fd=(rp[i]-rm[i])/(zero?h:2*h);const double error=std::abs(fd-j[i])/std::max(1.,std::abs(j[i]));fdWorst=std::max(fdWorst,error);need(error<3e-3,"production selected-branch FD mismatch test="+std::to_string(test)+" row="+std::to_string(i));}
  if(test==0){const double target=-double(d[q].x)*v.unknowns[q].initialAndScaling.y*direct.run.runtime.timestepSeconds()*(double(x[v.layout.offsets.z+1].x)*v.unknowns[v.layout.offsets.z+1].initialAndScaling.y/(double(x[1].x)*v.unknowns[1].initialAndScaling.y))/v.unknowns[v.layout.offsets.z+1].initialAndScaling.z;
   need(std::abs(j[v.layout.offsets.z+1]-target)<1e-6,"Q0 reverse used wrong donor concentration");}
  if(test==1)need(j[q]==0,"zero-forward pressure constraint acquired an epsilon resistance");
 }
 std::cout<<"directional_direct=pass cases=4 environments=2 residual_Jv_normalized_max="<<worst<<" one_sided_or_interior_FD_max="<<fdWorst<<" Q0_reverse_donor=downstream Q0_equal_pressure_tie=forward exact_zero_forward_diagonal=true\n";
}

float firstCrossing(float q,float dq){
 // Independent CPU binary32 witness: bracket the first representable alpha
 // whose fused affine value reaches the new branch. No physical Q projection.
 float a=float(-double(q)/double(dq));
 const auto crossed=[&](float x){const float v=std::fma(x,dq,q);return q>0?v<=0:v>=0;};
 for(unsigned i=0;i<8&&!crossed(a);++i)a=std::nextafter(a,INFINITY);
 need(a>0&&std::isfinite(a)&&crossed(a),"unrepresentable direct crossing");
 for(unsigned i=0;i<8;++i){const float prior=std::nextafter(a,0.f);if(!crossed(prior))return a;a=prior;}
 throw std::runtime_error("CPU crossing bracket did not terminate");
}
void sharedBreakpointChecks(){
 unsigned rowsChecked=0;
 for(unsigned test=0;test<2;++test){
  auto n=fixture(.05,1000,-1);auto edge=n.connections[0];edge.stableIdentifier=8;n.connections.push_back(edge);
  Direct direct(n);auto& run=direct.run;const auto& w=run.world;const auto& v=w.vascular;
  const unsigned N=unsigned(v.unknowns.size()),q=v.layout.offsets.y;
  auto x=run.state().vascularState;std::vector<nm_float4> d(2*N);
  std::array<float,2> expected{};
  for(unsigned e=0;e<2;++e){
   const unsigned b=e*N;
   x[b+q].x=test==0?(e?.75f:.5f):(e?.9f:.7f);
   x[b+q+1].x=test==0?(e?.25f:.75f):(e?.31f:.9f);
   d[b+q].x=test==0?-1.f:(e?-1.1f:-1.3f);
   d[b+q+1].x=test==0?-1.f:(e?-.9f:-1.1f);
   for(unsigned i=0;i<N;++i)if(i<q||i>=v.layout.offsets.z)d[b+i].x=(i%2?-.00025f:.0005f);
   expected[e]=std::min(firstCrossing(x[b+q].x,d[b+q].x),firstCrossing(x[b+q+1].x,d[b+q+1].x));
  }
  auto dispatch=w.dispatch;dispatch.objectCount=2; // Two owner slots, no invented physical objects.
  NMFGMRESLayoutGPU layout{};layout.vascularBase=3;layout.vascularUnknownCount=N;layout.unknownCount=3+2*N;
  NMMicrostepGPU micro{};micro.time.x=run.runtime.timestepSeconds();
  std::vector<nm_float4> lifted(layout.unknownCount);std::copy(d.begin(),d.end(),lifted.begin()+3);
  std::vector<nm_float4> accepted(2*N);for(unsigned e=0;e<2;++e)for(unsigned i=0;i<N;++i)accepted[e*N+i].x=v.unknowns[i].initialAndScaling.x/v.unknowns[i].initialAndScaling.y;
  std::array<id<MTLBuffer>,30> b{};
  b[2]=buffer(run.device,x);b[3]=buffer(run.device,lifted);
  b[4]=buffer(run.device,std::vector<nm_float4>(4,nm_float4{1,17,18,19}));
  b[5]=buffer(run.device,std::vector<float>(2,-1.f));b[6]=buffer(run.device,std::vector<NMMatterStatusGPU>(2));
  b[7]=buffer(run.device,v.compartments);b[8]=buffer(run.device,v.connections);
  b[9]=buffer(run.device,std::vector<unsigned>(2*N));b[10]=buffer(run.device,std::vector<unsigned>(2));
  b[12]=buffer(run.device,v.unknowns);b[13]=buffer(run.device,accepted);b[14]=buffer(run.device,v.tissues);b[15]=buffer(run.device,v.exchanges);
  b[16]=buffer(run.device,v.connectionIncidence);b[17]=buffer(run.device,v.connectionRanges);
  b[18]=buffer(run.device,v.bloodExchangeIncidence);b[19]=buffer(run.device,v.bloodExchangeRanges);
  b[20]=buffer(run.device,v.tissueExchangeIncidence);b[21]=buffer(run.device,v.tissueExchangeRanges);
  b[22]=buffer(run.device,std::vector<float>(2*v.compartments.size()));b[23]=buffer(run.device,std::vector<nm_float4>(2*N,nm_float4{31,32,33,34}));
  b[25]=buffer(run.device,v.cavities);b[26]=buffer(run.device,v.cavityFaces);b[27]=buffer(run.device,v.compartmentCavity);
  b[28]=buffer(run.device,std::vector<NMFEMNodeStateGPU>(1));b[29]=buffer(run.device,std::vector<NMFEMNodeStateGPU>(1));
  auto limiter=productionPipeline(run.device,direct.library,@"nm_vascular_limit_line_search");
  auto apply=productionPipeline(run.device,direct.library,@"nm_vascular_apply_solution");need(limiter&&apply,"direct limiter kernels missing");
  auto command=[run.queue commandBuffer];direct.encodeElastance(command,dispatch,b[7],b[22],b[6]);auto encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:limiter];[encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[encoder setBytes:&v.layout length:sizeof(v.layout) atIndex:1];
  for(unsigned i=2;i<30;++i)if(i!=11&&i!=24)[encoder setBuffer:b[i] offset:0 atIndex:i];
  [encoder setBytes:&micro length:sizeof(micro) atIndex:11];[encoder setBytes:&w.mixedSolver length:sizeof(w.mixedSolver) atIndex:24];
  [encoder setBytes:&layout length:sizeof(layout) atIndex:30];[encoder dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
  encoder=[command computeCommandEncoder];[encoder setComputePipelineState:apply];
  [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[encoder setBytes:&v.layout length:sizeof(v.layout) atIndex:1];
  [encoder setBuffer:b[3] offset:0 atIndex:2];[encoder setBuffer:b[5] offset:0 atIndex:3];[encoder setBuffer:b[2] offset:0 atIndex:4];
  [encoder setBuffer:b[6] offset:0 atIndex:5];[encoder setBuffer:b[9] offset:0 atIndex:6];[encoder setBuffer:b[10] offset:0 atIndex:7];
  [encoder setBuffer:b[8] offset:0 atIndex:8];[encoder setBytes:&layout length:sizeof(layout) atIndex:30];
  [encoder dispatchThreads:MTLSizeMake(2*N,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
  need(command.status==MTLCommandBufferStatusCompleted,"direct shared-breakpoint command failed");
  const auto* applied=static_cast<nm_float4*>(b[2].contents);const auto* trial=static_cast<nm_float4*>(b[23].contents);
  const auto* alpha=static_cast<float*>(b[5].contents);const auto* slots=static_cast<nm_float4*>(b[4].contents);
  const auto* mask=static_cast<unsigned*>(b[9].contents);
  for(unsigned e=0;e<2;++e){
   const auto status=static_cast<NMMatterStatusGPU*>(b[6].contents)[e];
   need(status.code==NM_STATUS_SUCCESS,"direct shared-breakpoint rejection code="+std::to_string(status.code));
   need(alpha[e]==expected[e],"limiter did not select earliest representable shared breakpoint");
   for(unsigned o=0;o<2;++o){const auto s=slots[e*2+o];need(s.x==alpha[e]&&s.y==17&&s.z==18&&s.w==19,"shared owner alpha publication differs");}
   unsigned crossed=0;
   for(unsigned i=0;i<N;++i){
    const unsigned row=e*N+i;need(std::memcmp(applied+row,trial+row,sizeof(nm_float4))==0,"line trial/apply arithmetic differs");
    need(mask[row]==0,"directional limiter invented an active-set mask");
    if(i>=q&&i<v.layout.offsets.z){
     const float wanted=std::fma(alpha[e],d[row].x,x[row].x);
     need(applied[row].x==wanted,"directional affine update differs from CPU fma");
     if(applied[row].x<=0)++crossed;
    }else{
     const double wanted=double(x[row].x)+double(alpha[e])*d[row].x;
     need(std::abs(double(applied[row].x)-wanted)<=2*std::numeric_limits<float>::epsilon()*std::max(1.,std::abs(wanted)),"nonflow coordinate did not use common alpha");
    }
    ++rowsChecked;
   }
   need(crossed==1,"more than the earliest flow edge crossed");
  }
  need(std::memcmp(lifted.data(),b[3].contents,lifted.size()*sizeof(nm_float4))==0,"limiter altered full Newton direction");
  need(std::memcmp(accepted.data(),b[13].contents,accepted.size()*sizeof(nm_float4))==0,"limiter changed accepted state");
 }
 std::cout<<"directional_shared_breakpoint=pass cases=2 environments=2 edges=2 rows="<<rowsChecked<<" exact_and_rounded_crossings=pass trial_apply=bitwise common_owner_alpha=pass no_flow_projection=true\n";
}

void legacyEquivalence(){
 double worst=0;
 for(double sign:{-1.,1.}){auto n=fixture(.05,.05,sign);Run directional(sourceFor(n));n.connections[0].flowLaw=VascularFlowLaw::resistanceInertance;n.connections[0].reverseResistance=0;Run legacy(sourceFor(n));
  for(unsigned step=0;step<16;++step){directional.step(step);legacy.step(step);const auto a=directional.state(),b=legacy.state();need(same(a.vascularClock,b.vascularClock),"legacy clock differs");
   for(unsigned i=0;i<a.vascularState.size();++i){const double e=std::abs(double(a.vascularState[i].x)-b.vascularState[i].x);worst=std::max(worst,e);need(e<3e-5,"Rf=Rr differs from legacy resistance");}}
 }
 std::cout<<"directional_legacy_equivalence=pass signs=2 steps=16 normalized_max="<<worst<<"\n";
}
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
 need(argc==1||(argc==2&&std::string(argv[1])=="--cpu-only"),"usage: directional-valve-check [--cpu-only]");
 std::cout<<std::setprecision(17)<<"directional_abi="<<NM_MATTER_ABI_VERSION<<"\n";cpuChecks();
 if(argc==2){std::cout<<"directional_compiler_qualification=pass physical_steps=0\n";return 0;}
 // Run the hard crossings first, preserving the original 12/64 nonlinear/Krylov budgets.
 qualify(fixture(.05,1000,-1,1e-4),"mitral_forward_guess_to_reverse");
 qualify(fixture(.05,1000,1,-1e-4),"mitral_reverse_guess_to_forward");
 qualify(fixture(0,10000,-1,1e-4),"aortic_forward_guess_to_reverse");
 qualify(fixture(0,10000,1,-1e-4),"aortic_reverse_guess_to_forward");
 qualify(fixture(.05,1000,-1),"mitral_zero_to_reverse");qualify(fixture(0,10000,1),"aortic_zero_to_forward");
 directChecks();sharedBreakpointChecks();legacyEquivalence();
 std::cout<<"directional_valve_qualification=pass runtime_input=nmatterpack source_coefficients_only=true anatomical_cardiac_cycle=false biological_calibration=false\n";return 0;
}catch(const std::exception& e){std::cerr<<"directional_valve_qualification=failed reason="<<e.what()<<"\n";return 1;}}}
