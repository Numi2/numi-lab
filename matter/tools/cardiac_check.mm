#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
#include "shi_hose_reference/shi_hose_dopri.hpp"
#include "shi_hose_reference/shi_hose_observables.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace numi::matter;
namespace ref = numi_shi_hose_reference;
namespace {
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
constexpr std::array<unsigned,20> sourceToNative={0,1,7,8,9,5,6,2,3,4,10,11,17,18,19,15,16,12,13,14};
constexpr std::array<const char*,10> anatomy={"CellML:shi_hose_2009:ModelHeart:LA","CellML:shi_hose_2009:ModelHeart:LV","CellML:shi_hose_2009:ModelPul:Pas","CellML:shi_hose_2009:ModelPul:Pat","CellML:shi_hose_2009:ModelPul:Pvn","CellML:shi_hose_2009:ModelHeart:RA","CellML:shi_hose_2009:ModelHeart:RV","CellML:shi_hose_2009:ModelSys:Sas","CellML:shi_hose_2009:ModelSys:Sat","CellML:shi_hose_2009:ModelSys:Svn"};
struct Run {
    CompiledWorld world;Runtime runtime;id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLBuffer> statuses;
    explicit Run(const VascularNetworkSource& n,double dt,unsigned newton=12,unsigned krylov=64) {
        need(n.compartments.size()==10&&n.connections.size()==10&&n.species.empty()&&n.tissues.empty()&&n.exchanges.empty(),"expected pinned full-loop source topology");
        for(unsigned i=0;i<10;++i)need(n.compartments[i].anatomicalIdentifier==anatomy[i],"source-to-oracle mapping differs");
        WorldSource s;s.environmentCount=2;s.frameTimestep=dt;s.gravity={0,0,0};s.vascular=n;
        s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=newton;s.mixedSolver.fgmresIterations=krylov;
        auto c=compileWorld(s,{.maximumRateExponent=0});std::string error;
        for(const auto& d:c.diagnostics)if(d.severity==Diagnostic::Severity::error)error+=d.message+"; ";
        need(c.succeeded(),"source compile failed: "+error);
        const auto package=std::filesystem::temp_directory_path()/(std::string("numi-cardiac-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        need(writePackage(c,package,&error),error);const bool loaded=readPackage(package,world,nullptr,&error);
        std::error_code removal;std::filesystem::remove(package,removal);need(loaded&&!removal&&world.fingerprint==c.world.fingerprint,"package reload: "+error);
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device missing");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue];statuses=[device newBufferWithLength:2*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        RuntimeConfiguration cfg;cfg.metallib=NUMI_MATTER_METALLIB;cfg.environmentCount=2;cfg.captureEvents=false;cfg.captureDiagnostics=true;cfg.adaptiveTransfer=false;
        const auto init=runtime.initialize(world,cfg);need(init.encoded,init.message);
        std::cout<<"cardiac_device="<<[device name].UTF8String<<" abi="<<NM_MATTER_ABI_VERSION<<" world_fingerprint="<<world.fingerprint<<'\n';
    }
    RuntimeStateSnapshot state(){auto s=runtime.snapshot();need(s.available,s.message);return s;}
    void step(unsigned index) {
        auto* values=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);for(unsigned e=0;e<2;++e){values[e]={};values[e].environment=e;}
        id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"command buffer missing");
        EncodeRequest req;req.commandBuffer=(__bridge void*)cb;req.environmentStatuses=(__bridge void*)statuses;req.controlStep=index;req.physicsSubsteps=1;req.timestepSeconds=runtime.timestepSeconds();req.runAdaptiveTransfer=false;
        req.phase=EncodePhase::preDynamics;auto r=runtime.encode(req);need(r.encoded,"pre: "+r.message);
        req.phase=EncodePhase::postCommit;r=runtime.encode(req);need(r.encoded,"post: "+r.message);
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal command failed");
    }
};
std::array<double,20> physical(const Run& run,const RuntimeStateSnapshot& s,unsigned env=0){
    std::array<double,20> out{};need(s.vascularState.size()==40,"vascular state arity");
    for(unsigned i=0;i<20;++i){const auto row=sourceToNative[i];out[i]=double(s.vascularState[env*20+row].x)*run.world.vascular.unknowns[row].initialAndScaling.y;need(std::isfinite(out[i]),"nonfinite hydraulic state");}
    return out;
}
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
    need(argc>=2,"usage: numi-matter-cardiac-check INPUT [--steps N] [--dt SECONDS] [--trace FILE]");
    unsigned steps=64,newton=12,krylov=64;double dt=.001;std::string tracePath,checkpointPath,resumePath;
    for(int i=2;i<argc;i+=2){need(i+1<argc,"missing option value");std::string key=argv[i];if(key=="--steps")steps=unsigned(std::stoul(argv[i+1]));else if(key=="--dt")dt=std::stod(argv[i+1]);else if(key=="--trace")tracePath=argv[i+1];else if(key=="--checkpoint")checkpointPath=argv[i+1];else if(key=="--resume")resumePath=argv[i+1];else if(key=="--newton")newton=unsigned(std::stoul(argv[i+1]));else if(key=="--krylov")krylov=unsigned(std::stoul(argv[i+1]));else need(false,"unknown option");}
    need(steps>0&&steps<=1000000&&std::isfinite(dt)&&dt>0&&dt<=.01,"invalid bounded qualification request");
    VascularNetworkSource n;std::string error;need(readHumanPhysiologyNetwork(argv[1],n,&error),error);
    std::cout<<std::setprecision(12)<<std::unitbuf;
    const auto begin=std::chrono::steady_clock::now();Run run(n,dt,newton,krylov);ref::Integrator reference;
    reference.relative_tolerance=1e-12;reference.absolute_tolerance=1e-12;
    ref::State dy{};ref::Values values{};ref::evaluate(0,reference.state,dy,&values);
    const auto initialReference=ref::native_hydraulic_state(values);const auto start=run.state();const auto first=physical(run,start);
    need(start.vascularClock.size()==2,"accepted clock absent");
    for(unsigned i=0;i<20;++i)need(std::abs(first[i]-initialReference[i])/run.world.vascular.unknowns[sourceToNative[i]].initialAndScaling.y<5e-6,"initial source state mismatch row="+std::to_string(i));
    const double initialStorage=std::accumulate(first.begin(),first.begin()+10,0.0);
    const int exponent=std::bit_cast<std::int32_t>(run.world.vascular.layout.clock.x);
    const auto ticks=static_cast<std::uint64_t>(std::ldexp(double(run.runtime.timestepSeconds()),-exponent));
    unsigned offset=0;auto before=start;
    if(!resumePath.empty()) {
        std::ifstream input(resumePath,std::ios::binary);std::uint64_t magic=0;float savedDt=0;
        input.read(reinterpret_cast<char*>(&magic),sizeof(magic));input.read(reinterpret_cast<char*>(&offset),sizeof(offset));input.read(reinterpret_cast<char*>(&savedDt),sizeof(savedDt));
        input.read(reinterpret_cast<char*>(before.vascularState.data()),40*sizeof(nm_float4));input.read(reinterpret_cast<char*>(before.vascularClock.data()),2*sizeof(NMVascularClockGPU));
        need(input.good()&&magic==0x4e554d4943415244ULL&&savedDt==run.runtime.timestepSeconds(),"invalid diagnostic continuation");
        const auto restored=run.runtime.restore(before);need(restored.encoded,"continuation restore: "+restored.message);
        std::cout<<"diagnostic_continuation_step="<<offset<<" newton_budget="<<newton<<" krylov_budget="<<krylov<<'\n';
    }
    double maxVolume=0,maxStorage=0,maxFlow=0,maxInvariant=0,squareError=0;unsigned long long observations=0;
    std::ofstream trace;if(!tracePath.empty()){trace.open(tracePath);need(trace.good(),"trace open failed");trace<<std::setprecision(17)<<"time_seconds";for(unsigned i=0;i<20;++i)trace<<",native_"<<i<<",cellml_"<<i;trace<<'\n';}
    for(unsigned step=0;step<steps;++step){@autoreleasepool{
        run.step(offset+step);const auto current=run.state();
        if(current.statuses[0].code!=NM_STATUS_SUCCESS && !checkpointPath.empty()) {
            std::ofstream output(checkpointPath,std::ios::binary);const std::uint64_t magic=0x4e554d4943415244ULL;const unsigned at=offset+step;const float savedDt=run.runtime.timestepSeconds();
            output.write(reinterpret_cast<const char*>(&magic),sizeof(magic));output.write(reinterpret_cast<const char*>(&at),sizeof(at));output.write(reinterpret_cast<const char*>(&savedDt),sizeof(savedDt));
            output.write(reinterpret_cast<const char*>(before.vascularState.data()),40*sizeof(nm_float4));output.write(reinterpret_cast<const char*>(before.vascularClock.data()),2*sizeof(NMVascularClockGPU));need(output.good(),"diagnostic checkpoint write failed");
            const auto& st=current.statuses[0];std::cout<<"cardiac_failure_checkpoint="<<checkpointPath<<" status_diagnostics="<<st.diagnostics.x<<','<<st.diagnostics.y<<','<<st.diagnostics.z<<','<<st.diagnostics.w<<'\n';
        }
        for(unsigned env=0;env<2;++env)need(current.statuses[env].code==NM_STATUS_SUCCESS,"source step "+std::to_string(offset+step)+" env "+std::to_string(env)+" rejected code="+std::to_string(current.statuses[env].code)+" residual="+std::to_string(current.statuses[env].diagnostics.z));
        const unsigned __int128 expected=static_cast<unsigned __int128>(ticks)*(offset+step+1u);
        for(const auto& clock:current.vascularClock)need(clock.low==std::uint64_t(expected)&&clock.high==std::uint64_t(expected>>64u),"accepted cardiac phase did not advance exactly");
        need(std::memcmp(current.vascularState.data(),current.vascularState.data()+20,20*sizeof(nm_float4))==0,"identical source environments diverged");
        const double time=double(offset+step+1u)*run.runtime.timestepSeconds();reference.advance(time);ref::evaluate(time,reference.state,dy,&values);
        const auto expectedState=ref::native_hydraulic_state(values),actual=physical(run,current);
        if(trace.is_open())trace<<time;
        for(unsigned i=0;i<20;++i){const double err=std::abs(actual[i]-expectedState[i]);const auto row=sourceToNative[i];
            if(i<10){if(run.world.vascular.compartments[row].identity.z==0)maxVolume=std::max(maxVolume,err);else maxStorage=std::max(maxStorage,err);}
            else maxFlow=std::max(maxFlow,err);
            const double scaled=err/run.world.vascular.unknowns[row].initialAndScaling.y;squareError+=scaled*scaled;++observations;
            if(trace.is_open())trace<<','<<actual[i]<<','<<expectedState[i];
        }
        if(trace.is_open())trace<<'\n';
        const double storage=std::accumulate(actual.begin(),actual.begin()+10,0.0);
        maxInvariant=std::max(maxInvariant,std::abs(storage-initialStorage)/std::abs(initialStorage));need(maxInvariant<1e-4,"closed hydraulic storage conservation exceeds numerical gate");
        before=current;
        if((step+1)%500u==0u)std::cout<<"cardiac_progress accepted_steps="<<step+1<<" time_seconds="<<time<<" max_chamber_volume_error_m3="<<maxVolume<<" max_flow_error_m3_per_s="<<maxFlow<<'\n';
    }}
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"cardiac_source_run=pass accepted_steps="<<steps<<" environments=2 failed_steps=0 dt_seconds="<<run.runtime.timestepSeconds()<<" duration_seconds="<<steps*double(run.runtime.timestepSeconds())<<" clock=exact_binary_128 replay_pair=bitwise max_chamber_volume_error_m3="<<maxVolume<<" max_storage_error_m3="<<maxStorage<<" max_flow_error_m3_per_s="<<maxFlow<<" scaled_state_rms_error="<<std::sqrt(squareError/observations)<<" relative_storage_invariant_error="<<maxInvariant<<" reference_accepted_steps="<<reference.accepted_steps<<" reference_rejected_steps="<<reference.rejected_steps<<" wall_seconds="<<elapsed<<" qualification=source_model_numerical_comparison absolute_vascular_blood_volume=unqualified biological_calibration=unqualified\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"cardiac_source_run=failed reason="<<e.what()<<'\n';return 1;}}}
