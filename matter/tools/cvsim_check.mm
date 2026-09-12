#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
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
#include <sstream>
#include <stdexcept>
using namespace numi::matter;
namespace {
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
constexpr unsigned count=45;
constexpr std::array<const char*,21> labels={"ascending_aorta","brachiocephalic_arteries","upper_body_arteries","upper_body_veins","superior_vena_cava","descending_thoracic_aorta","abdominal_aorta","renal_arteries","renal_veins","splanchnic_arteries","splanchnic_veins","lower_body_arteries","lower_body_veins","abdominal_veins","inferior_vena_cava","right_atrium","right_ventricle","pulmonary_arteries","pulmonary_veins","left_atrium","left_ventricle"};
struct Run {
    CompiledWorld world;Runtime runtime;id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLBuffer> statuses;
    explicit Run(const VascularNetworkSource& n,double dt,unsigned newton=12,unsigned krylov=64) {
        need(n.compartments.size()==21&&n.connections.size()==24&&n.species.empty()&&n.tissues.empty()&&n.exchanges.empty(),"expected pinned full-loop source topology");
        for(unsigned i=0;i<21;++i)need(n.compartments[i].anatomicalIdentifier==std::string("source_aggregate:CVSim21:")+labels[i],"source-to-reference mapping differs");
        WorldSource s;s.environmentCount=2;s.frameTimestep=dt;s.gravity={0,0,0};s.vascular=n;
        s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=newton;s.mixedSolver.fgmresIterations=krylov;
        auto c=compileWorld(s,{.maximumRateExponent=0});std::string error;
        for(const auto& d:c.diagnostics)if(d.severity==Diagnostic::Severity::error)error+=d.message+"; ";
        need(c.succeeded(),"source compile failed: "+error);
        const auto package=std::filesystem::temp_directory_path()/(std::string("numi-cvsim-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        need(writePackage(c,package,&error),error);const bool loaded=readPackage(package,world,nullptr,&error);
        std::error_code removal;std::filesystem::remove(package,removal);need(loaded&&!removal&&world.fingerprint==c.world.fingerprint,"package reload: "+error);
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device missing");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue];statuses=[device newBufferWithLength:2*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        RuntimeConfiguration cfg;cfg.metallib=NUMI_MATTER_METALLIB;cfg.environmentCount=2;cfg.captureEvents=false;cfg.captureDiagnostics=true;cfg.adaptiveTransfer=false;
        const auto init=runtime.initialize(world,cfg);need(init.encoded,init.message);
        std::cout<<"cvsim_device="<<[device name].UTF8String<<" abi="<<NM_MATTER_ABI_VERSION<<" world_fingerprint="<<world.fingerprint<<'\n';
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
std::array<double,count> physical(const Run& run,const RuntimeStateSnapshot& s,unsigned env=0){
    std::array<double,count> out{};need(s.vascularState.size()==2*count,"vascular state arity");
    for(unsigned i=0;i<count;++i){out[i]=double(s.vascularState[env*count+i].x)*run.world.vascular.unknowns[i].initialAndScaling.y;need(std::isfinite(out[i]),"nonfinite hydraulic state");}
    return out;
}
// Offline independent C++ reference is an input artifact, never a runtime owner.
struct Reference {
    std::ifstream input; double time=-1; std::array<double,count> state{}; double translation=0;
    explicit Reference(const std::string& path,bool aligned):input(path),translation(aligned?184e-6:0){
        need(input.good(),"reference open failed");std::string line;need(bool(std::getline(input,line)),"reference header missing");
        std::string expected="time_s";
        for(unsigned i=0;i<21;++i)expected+=",P_"+std::to_string(i)+"_mmHg";
        for(unsigned i=0;i<21;++i)expected+=",V_"+std::to_string(i)+"_mL";
        for(unsigned i=0;i<24;++i)expected+=",Q_"+std::to_string(i)+"_mL_per_s";
        expected+=",volume_sum_mL,source_atrial_phase_s,source_ventricular_phase_s";
        if(!line.empty()&&line.back()=='\r')line.pop_back();need(line==expected,"reference columns or units differ");next();need(time==0,"reference must start at zero");
    }
    void next(){
        std::string line;need(bool(std::getline(input,line)),"reference exhausted");std::istringstream stream(line);std::string field;
        std::array<double,70> values{};unsigned n=0;
        while(std::getline(stream,field,',')){need(n<values.size(),"extra reference column");std::size_t used=0;values[n]=std::stod(field,&used);need(used==field.size()&&std::isfinite(values[n]),"invalid reference value");++n;}
        need(n==values.size()&&values[0]>time,"reference arity or time order differs");time=values[0];
        double sum=0;for(unsigned i=0;i<21;++i){need(values[22+i]>0,"nonpositive reference volume");state[i]=values[22+i]*1e-6;sum+=values[22+i];}
        need(std::abs(sum-values[67])<1e-7&&std::abs(sum-5150)<1e-5,"reference blood volume does not conserve source total");
        state[2]+=translation;state[5]-=translation;
        for(unsigned i=0;i<24;++i)state[21+i]=values[43+i]*1e-6;
    }
    void advance(double target){while(time<target)next();need(time==target,"native/reference sample clocks differ; use an exact binary32 grid subset");}
};
}
int main(int argc,const char* argv[]){@autoreleasepool{try{
    need(argc>=3,"usage: numi-matter-cvsim-check INPUT REFERENCE.csv [--steps N] [--dt SECONDS] [--trace FILE] [--volume-coordinates upstream_equation|heldt_table_aligned]");
    unsigned steps=64;double dt=.001;std::string tracePath,checkpointPath;bool aligned=false;
    for(int i=3;i<argc;i+=2){need(i+1<argc,"missing option value");std::string key=argv[i];
        if(key=="--steps")steps=unsigned(std::stoul(argv[i+1]));else if(key=="--dt")dt=std::stod(argv[i+1]);
        else if(key=="--trace")tracePath=argv[i+1];else if(key=="--checkpoint")checkpointPath=argv[i+1];
        else if(key=="--volume-coordinates"){std::string value=argv[i+1];need(value=="upstream_equation"||value=="heldt_table_aligned","unknown volume coordinates");aligned=value=="heldt_table_aligned";}
        else need(false,"unknown option");}
    need(steps>0&&steps<=1000000&&std::isfinite(dt)&&dt>0&&dt<=.01,"invalid bounded qualification request");
    VascularNetworkSource n;std::string error;need(readHumanPhysiologyNetwork(argv[1],n,&error),error);
    std::cout<<std::setprecision(17)<<std::unitbuf;const auto begin=std::chrono::steady_clock::now();Run run(n,dt);Reference reference(argv[2],aligned);
    const auto start=run.state();const auto first=physical(run,start);need(start.vascularClock.size()==2,"accepted clock absent");
    for(unsigned i=0;i<count;++i)need(std::abs(first[i]-reference.state[i])/run.world.vascular.unknowns[i].initialAndScaling.y<5e-6,"initial source state mismatch row="+std::to_string(i));
    const double initialVolume=std::accumulate(first.begin(),first.begin()+21,0.0);
    need(std::abs(initialVolume-.00515)<1e-9,"initial source absolute blood budget differs");
    const int exponent=std::bit_cast<std::int32_t>(run.world.vascular.layout.clock.x);
    const auto ticks=static_cast<std::uint64_t>(std::ldexp(double(run.runtime.timestepSeconds()),-exponent));
    double maxVolume=0,maxFlow=0,maxInvariant=0,squareError=0;unsigned long long observations=0;auto before=start;
    std::ofstream trace;if(!tracePath.empty()){trace.open(tracePath);need(trace.good(),"trace open failed");trace<<std::setprecision(17)<<"time_seconds";for(unsigned i=0;i<count;++i)trace<<",native_"<<i<<",reference_"<<i;trace<<'\n';}
    for(unsigned step=0;step<steps;++step){@autoreleasepool{
        run.step(step);const auto current=run.state();
        if(current.statuses[0].code!=NM_STATUS_SUCCESS&&!checkpointPath.empty()){
            std::ofstream output(checkpointPath,std::ios::binary);const std::uint64_t magic=0x435653494d323131ULL;const float savedDt=run.runtime.timestepSeconds();
            output.write(reinterpret_cast<const char*>(&magic),sizeof(magic));output.write(reinterpret_cast<const char*>(&step),sizeof(step));output.write(reinterpret_cast<const char*>(&savedDt),sizeof(savedDt));
            output.write(reinterpret_cast<const char*>(before.vascularState.data()),2*count*sizeof(nm_float4));output.write(reinterpret_cast<const char*>(before.vascularClock.data()),2*sizeof(NMVascularClockGPU));need(output.good(),"diagnostic checkpoint write failed");
            std::cout<<"cvsim_failure_checkpoint="<<checkpointPath<<" pre_step="<<step<<'\n';
        }
        for(unsigned env=0;env<2;++env)need(current.statuses[env].code==NM_STATUS_SUCCESS,"source step "+std::to_string(step)+" env "+std::to_string(env)+" rejected code="+std::to_string(current.statuses[env].code)+" residual="+std::to_string(current.statuses[env].diagnostics.z));
        const unsigned __int128 expected=static_cast<unsigned __int128>(ticks)*(step+1u);
        for(const auto& clock:current.vascularClock)need(clock.low==std::uint64_t(expected)&&clock.high==std::uint64_t(expected>>64u),"accepted phase did not advance exactly");
        need(std::memcmp(current.vascularState.data(),current.vascularState.data()+count,count*sizeof(nm_float4))==0,"identical source environments diverged");
        const double time=double(step+1u)*run.runtime.timestepSeconds();reference.advance(time);const auto actual=physical(run,current);
        if(trace.is_open())trace<<time;
        for(unsigned i=0;i<count;++i){const double err=std::abs(actual[i]-reference.state[i]);
            if(i<21)maxVolume=std::max(maxVolume,err);else maxFlow=std::max(maxFlow,err);
            const double scaled=err/run.world.vascular.unknowns[i].initialAndScaling.y;squareError+=scaled*scaled;++observations;
            if(trace.is_open())trace<<','<<actual[i]<<','<<reference.state[i];}
        if(trace.is_open())trace<<'\n';
        const double volume=std::accumulate(actual.begin(),actual.begin()+21,0.0);
        maxInvariant=std::max(maxInvariant,std::abs(volume-initialVolume)/initialVolume);need(maxInvariant<1e-4,"absolute blood conservation exceeds numerical gate");
        // Exercise production snapshot restore mid-trajectory; compare the replay
        // to the same committed physical state and exact clock without stepping C.
        if(step==steps/2){const auto restored=run.runtime.restore(before);need(restored.encoded,"restore before: "+restored.message);run.step(step);const auto replay=run.state();
            need(std::memcmp(replay.vascularState.data(),current.vascularState.data(),2*count*sizeof(nm_float4))==0&&std::memcmp(replay.vascularClock.data(),current.vascularClock.data(),2*sizeof(NMVascularClockGPU))==0,"snapshot replay differs");
            for(const auto& status:replay.statuses)need(status.code==NM_STATUS_SUCCESS,"replayed source step failed");}
        before=current;
        if((step+1)%500u==0u)std::cout<<"cvsim_progress accepted_steps="<<step+1<<" time_seconds="<<time<<" max_volume_error_m3="<<maxVolume<<" max_flow_error_m3_per_s="<<maxFlow<<'\n';
    }}
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"cvsim_source_run=pass accepted_steps="<<steps<<" environments=2 failed_steps=0 dt_seconds="<<run.runtime.timestepSeconds()<<" duration_seconds="<<steps*double(run.runtime.timestepSeconds())<<" clock=exact_binary_128_rational_period replay_pair=bitwise snapshot_replay=bitwise max_volume_error_m3="<<maxVolume<<" max_flow_error_m3_per_s="<<maxFlow<<" scaled_state_rms_error="<<std::sqrt(squareError/observations)<<" relative_blood_volume_invariant_error="<<maxInvariant<<" wall_seconds="<<elapsed<<" qualification=source_variant_numerical_comparison biological_calibration=unqualified\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"cvsim_source_run=failed reason="<<e.what()<<'\n';return 1;}}}
