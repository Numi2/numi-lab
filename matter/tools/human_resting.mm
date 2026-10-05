#include "human_resting_runtime.hpp"
using namespace numi::human;

int main(int argc,const char* argv[]) {@autoreleasepool {try {
    need(argc>=4,"usage: numi-human-resting NETWORK.json RESPIRATION.json OUTPUT.csv [--steps N] [--dt SECONDS] [--instrument-excitation D I]");
    unsigned steps=100;float dt=.001f;float excitationD=0,excitationI=0;bool instrument=false;bool vascularDense45=false;
    double interventionStart=-1,interventionEnd=-1;float interventionScale=1;
    bool verifyTransaction=false;
    for(int a=4;a<argc;++a) {
        std::string key=argv[a];
        if(key=="--steps"&&a+1<argc)steps=unsigned(std::stoul(argv[++a]));
        else if(key=="--dt"&&a+1<argc)dt=std::stof(argv[++a]);
        else if(key=="--instrument-excitation"&&a+2<argc) {instrument=true;excitationD=std::stof(argv[++a]);excitationI=std::stof(argv[++a]);}
        else if(key=="--drive-intervention"&&a+3<argc) {interventionStart=std::stod(argv[++a]);interventionEnd=std::stod(argv[++a]);interventionScale=std::stof(argv[++a]);}
        else if(key=="--verify-transaction")verifyTransaction=true;
        else if(key=="--vascular-dense45")vascularDense45=true;
        else need(false,"unknown or incomplete argument "+key);
    }
    need(steps>0&&steps<=1000000,"bounded step count required");
    need(std::isfinite(interventionStart)&&std::isfinite(interventionEnd)&&std::isfinite(interventionScale)&&interventionScale>=0&&interventionScale<=2,"invalid respiratory intervention");
    std::cout<<std::setprecision(12)<<std::unitbuf;
    const auto start=std::chrono::steady_clock::now();
    RestingRun run(argv[1],argv[2],dt,vascularDense45);
    std::unique_ptr<RespiratoryBrain> brain;
    if(instrument)*static_cast<nm_float4*>(run.respiration->excitation.contents)={excitationD,excitationI,0,0};
    else {
        brain=std::make_unique<RespiratoryBrain>(*run.respiration,run.world,argv[2]);
        brain->interventionStart=interventionStart;brain->interventionEnd=interventionEnd;brain->interventionScale=interventionScale;
    }
    if(verifyTransaction) {
        need(brain!=nullptr,"transaction check requires the coupled Brain controller");
        run.batch(0,1);
        const auto before=run.runtime.snapshot();need(before.available,before.message);
        const auto respiratoryBefore=*static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
        const auto brainBefore=*static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(brain->accepted.contents);
        run.respiration->dispatch.reject=1;run.batch(1,1,true);
        const auto rejected=run.runtime.snapshot();need(rejected.available,rejected.message);
        need(before.vascularState.size()==rejected.vascularState.size()&&
             std::memcmp(before.vascularState.data(),rejected.vascularState.data(),before.vascularState.size()*sizeof(nm_float4))==0&&
             std::memcmp(before.vascularClock.data(),rejected.vascularClock.data(),before.vascularClock.size()*sizeof(NMVascularClockGPU))==0,
             "rejected respiration advanced circulation or its clock");
        need(std::memcmp(&respiratoryBefore,run.respiration->accepted.contents,sizeof(respiratoryBefore))==0&&
             std::memcmp(&brainBefore,brain->accepted.contents,sizeof(brainBefore))==0,
             "rejected body advanced respiration/controller history");
        run.respiration->dispatch.reject=0;run.batch(1,1);
        const auto accepted=run.runtime.snapshot();need(accepted.available,accepted.message);
        const auto respiratoryAccepted=*static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
        const auto brainAccepted=*static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(brain->accepted.contents);
        const auto restored=run.runtime.restore(before);need(restored.encoded,restored.message);
        std::memcpy(run.respiration->accepted.contents,&respiratoryBefore,sizeof(respiratoryBefore));
        std::memcpy(brain->accepted.contents,&brainBefore,sizeof(brainBefore));
        run.batch(1,1);
        const auto replay=run.runtime.snapshot();need(replay.available,replay.message);
        need(std::memcmp(accepted.vascularState.data(),replay.vascularState.data(),accepted.vascularState.size()*sizeof(nm_float4))==0&&
             std::memcmp(accepted.vascularClock.data(),replay.vascularClock.data(),accepted.vascularClock.size()*sizeof(NMVascularClockGPU))==0&&
             std::memcmp(&respiratoryAccepted,run.respiration->accepted.contents,sizeof(respiratoryAccepted))==0&&
             std::memcmp(&brainAccepted,brain->accepted.contents,sizeof(brainAccepted))==0,
             "coupled respiratory/circulatory/controller replay differs");
        std::cout<<"coupled_rejection=bitwise_unchanged circulation_clock=unchanged controller_history=unchanged accepted_replay=bitwise\n";
        return 0;
    }
    std::ofstream trace(argv[3]);need(trace.good(),"trace path is not writable");
    trace<<std::setprecision(12)<<"time_s,lung_volume_ml,airflow_ml_s,alveolar_pa,pleural_pa,diaphragm_mm,rib_mm,PaO2_mmhg,PaCO2_mmhg,SaO2,oxygen_balance_error_stpd_ml,co2_balance_error_stpd_ml,breaths,tidal_ml,lv_mmhg,rv_mmhg,aorta_mmhg,pulmonary_artery_mmhg,lv_ml,rv_ml,blood_ml,blood_error_ml,aortic_ejected_ml,pulmonary_ejected_ml,complete_filling_ejection_cycles,last_lv_stroke_ml\n";
    double gpuSeconds=0;
    for(unsigned first=0;first<steps;) {@autoreleasepool {
        unsigned count=std::min(16u,steps-first);gpuSeconds+=run.batch(first,count);first+=count;
        const auto s=*static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
        trace<<double(s.status.x)*run.runtime.timestepSeconds()<<','<<s.mechanics.x*1e6<<','<<s.mechanics.w*1e6<<','<<s.mechanics.y<<','<<s.mechanics.z<<','
             <<s.motion.x/run.respiration->parameters.geometry.z*1000<<','<<s.motion.y/run.respiration->parameters.geometry.w*1000<<','
             <<s.observation.x<<','<<s.observation.y<<','<<s.observation.z<<','<<s.gasBudget.z*1e6<<','<<s.gasBudget.w*1e6<<','<<s.status.y<<','<<s.breath.z*1e6<<','
             <<s.cardiacPressure.x/133.322387415<<','<<s.cardiacPressure.y/133.322387415<<','<<s.cardiacPressure.z/133.322387415<<','<<s.cardiacPressure.w/133.322387415<<','
             <<s.circulation.y*1e6<<','<<s.circulation.z*1e6<<','<<s.circulation.x*1e6<<','<<s.circulation.w*1e6<<','<<s.cardiacFlow.x*1e6<<','<<s.cardiacFlow.y*1e6<<','<<s.cardiacStatus.x<<','<<s.cardiacFlow.w*1e6<<'\n';
        if(first%1000==0)std::cout<<"accepted="<<first<<" time_s="<<first*double(run.runtime.timestepSeconds())<<" PaCO2="<<s.observation.y<<'\n';
    }}
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double simulated=steps*double(run.runtime.timestepSeconds());
    std::cout<<"accepted_steps="<<steps<<" simulated_s="<<simulated<<" wall_s="<<wall<<" gpu_s="<<gpuSeconds<<" real_time_factor="<<simulated/wall<<" Brain_control="<<!instrument<<" no_anatomy_or_resting_claim=true\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"human_resting=failed reason="<<e.what()<<'\n';return 1;}}}
