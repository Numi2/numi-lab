#import <Foundation/Foundation.h>
#include "CrowCameraFixture.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/RetinalObserver.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace crowCameraFixture;
namespace {
constexpr unsigned ticks=65,features=1251;
template<class T> void save(const std::filesystem::path& p,const std::vector<T>& a) {
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(bool(f),"Evidence write failed");
}
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(T));
}
struct Stream {
    CC_SHA256_CTX ctx{};std::uint64_t bytes=0;
    Stream(){CC_SHA256_Init(&ctx);}
    template<class T> void add(const std::vector<T>& values) {
        const auto data=std::as_bytes(std::span(values));
        for(std::size_t at=0;at<data.size();) {const auto n=std::min<std::size_t>(data.size()-at,1u<<20);CC_SHA256_Update(&ctx,data.data()+at,static_cast<CC_LONG>(n));at+=n;}
        bytes+=data.size();
    }
    checkpoint::Digest finish(){checkpoint::Digest out;CC_SHA256_Final(out.data(),&ctx);return out;}
};
struct Evidence {
    std::vector<std::byte> world,retina;
    std::vector<MRFiniteResourceStateGPU> food;
    std::array<checkpoint::Digest,3> streams;
    std::uint64_t checkedHistoryFloats=0;
};
Evidence execute(const std::array<mr_float4,2>& ports,const std::filesystem::path& output,unsigned lanes,unsigned chunk) {@autoreleasepool {
    require(std::filesystem::create_directory(output),"Arm output already exists");
    auto m=makeManifest(false);applyAmericanCrowForagingFixtureLimits(m);
    m.profile.environmentCount=lanes;m.profile.controlSteps=ticks;m.reality.reset={};
    CompiledRun base;check(compileRun(m,base),"Compile physical fixture");
    const auto visual=makeVisual(base,output,true);
    const auto authored=authorAmericanCrowCuedForagingProgram(base,visual,ports,lanes,2651113001ull);
    m.sensors.observation={};m.reality.reset={};
    m.task=makeAmericanCrowCuedForagingTaskPack(m.sensors.observation,m.reality.reset,authored);
    m.task.capacities=m.profile.capacities;m.task.maximumEpisodeSteps=100000;m.reality.reset={};
    applyAmericanCrowMemoryChoiceCurriculum(m.task,64,256,16);m.sensors.deviceVisual=authored.sensor;
    CompiledRun run;check(compileRun(m,run),"Compile held-action task");
    require(base.world().fingerprint()==run.world().fingerprint(),"Task authoring altered physical fixture");
    FiniteResourceContext food(authored.resources);
    RetinalObserverContext retina(run,visual.family,visual.manifest,lanes,2651113002ull,&food);
    PolicyPack p;p.id="constant_temporal_history_control";p.revision=1;
    p.observationMean.assign(features,0);p.observationInverseStandardDeviation.assign(features,1);
    PolicyDenseLayer layer{.inputCount=features,.outputCount=18,.activation=PolicyActivation::identity,
        .weights=std::vector<float>(features*18,0),.bias=std::vector<float>(18,0)};
    layer.bias[15]=.6f;layer.bias[17]=.9f;p.layers.push_back(std::move(layer));bindPolicyPack(p,run.task());
    CompiledPolicyProgram policy;check(compilePolicyProgram(p,run.task(),policy),"Compile constant control");
    MetalWorldStepConfig step;step.timestepSeconds=.02f;step.physicsSubsteps=4;step.velocityIterations=4;step.finalVelocityIterations=2;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=run.task();step.taskSeed=2651113002ull;step.minimumDifficultyBand=step.maximumDifficultyBand=0;step.publishFinalState=true;
    step.policyProgram=policy;step.devicePhysicsProgram=food.deviceProgram();step.deviceObservationProgram=retina.deviceProgram();
    if(run.flappingWingProgram())step.flappingWingProgram=*run.flappingWingProgram();
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    for(unsigned e=0;e<lanes;++e) {
        q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
        scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());
    }
    MetalWorldConfig config;config.enableResidentCheckpoints=true;config.maximumInFlightSubmissions=1;
    MetalWorldContext world(config);MetalWorldResidentState resident;MetalWorldResult result;
    std::array<Stream,3> streams;
    for(unsigned start=0;start<ticks;start+=chunk) {
        const auto count=std::min(chunk,ticks-start);std::vector<unsigned> resets(count*lanes,0);
        MetalWorldBatch batch{.environmentCount=lanes,.controlStepCount=count,.resetMasks=resets};MetalWorldSubmission ticket;
        if(!start){batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;check(world.initializeResidentState(run.world(),batch,step,resident,ticket),"Initialize history arm");}
        else check(world.submitResident(run.world(),batch,step,resident,ticket),"Submit history arm");
        const auto status=ticket.wait(result);check(status,"Complete history arm");require(!status.failedStepCount,"Native physics rejected");
        for(const auto& t:result.transitions)require(!t.termination.x,"Physical termination");
        streams[0].add(result.actorObservations);streams[1].add(result.criticObservations);streams[2].add(result.transitions);
    }
    Evidence e;e.world=world.checkpointResident(resident,step);e.retina=retina.checkpointState();e.food=food.snapshot();
    for(unsigned i=0;i<3;++i)e.streams[i]=streams[i].finish();
    for(const auto error:retina.snapshot().errors)require(!error,"Sticky retinal error");
    const auto h=checkpoint::Reader{e.world}.get<checkpoint::Header>();
    checkpoint::Reader reader{checkpoint::open(e.world,{'N','U','M','I','W','R','0','1'},h.identity)};
    require(reader.get<std::uint64_t>()==run.world().fingerprint(),"Checkpoint physical world differs");
    require(reader.get<std::uint64_t>()==lanes,"Checkpoint lane count differs");
    reader.take(sizeof(checkpoint::Digest)+9*sizeof(std::uint64_t)+sizeof(std::uint32_t));
    std::vector<float> history;
    while(!reader.remaining.empty()) {
        const auto index=reader.get<std::uint32_t>();const auto bytes=reader.blob();
        if(index==200){require(history.empty(),"Duplicate temporal history section");require(bytes.size()%sizeof(float)==0,"Bad history section size");history.resize(bytes.size()/sizeof(float));std::memcpy(history.data(),bytes.data(),bytes.size());}
    }
    const auto bindings=run.task().actionBindings();const auto actions=bindings.size();
    require(history.size()==lanes*actions*2,"Temporal history does not contain two independent arrays per lane");
    require(result.finalQ.size()==q.size() && result.finalV.size()==v.size(),"Published final state has wrong shape");
    unsigned distinct=0;
    for(unsigned lane=0;lane<lanes;++lane)for(unsigned action=0;action<actions;++action) {
        const auto& b=bindings[action];const float wantV=b.indices.w==MR_INVALID_INDEX?0:result.finalV.at(lane*run.model().defaultV.size()+b.indices.w);
        const float wantQ=b.indices.z==MR_INVALID_INDEX?0:result.finalQ.at(lane*run.model().defaultQ.size()+b.indices.z);
        const float gotV=history[lane*2*actions+action],gotQ=history[lane*2*actions+actions+action];
        require(!std::memcmp(&wantV,&gotV,sizeof(float)) && !std::memcmp(&wantQ,&gotQ,sizeof(float)),"Temporal history differs from accepted joint state at lane "+std::to_string(lane)+" action "+std::to_string(action));
        distinct+=std::memcmp(&wantV,&wantQ,sizeof(float))!=0;e.checkedHistoryFloats+=2;
    }
    require(distinct>=lanes,"History control failed to distinguish position and velocity");
    save(output/"world.bin",e.world);save(output/"retina.bin",e.retina);save(output/"food.bin",e.food);
    save(output/"final-q.f32",result.finalQ);save(output/"final-v.f32",result.finalV);
    save(output/"action-bindings.bin",std::vector<MRTaskActionBindingGPU>(bindings.begin(),bindings.end()));
    std::ofstream report(output/"summary.json");report<<"{\"lanes\":"<<lanes<<",\"chunk\":"<<chunk<<",\"ticks\":"<<ticks<<",\"history_float_checks\":"<<e.checkedHistoryFloats<<",\"distinct_q_v_pairs\":"<<distinct<<",\"history_exact_to_final_state\":true,\"stream_bytes\":[";
    for(unsigned i=0;i<3;++i){if(i)report<<',';report<<streams[i].bytes;}report<<"],\"stream_sha256\":[";
    for(unsigned i=0;i<3;++i){if(i)report<<',';report<<'"';for(auto byte:e.streams[i])report<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte);report<<'"';}
    report<<"],\"physics_rejections\":0,\"terminations\":0}\n";return e;
}}
}
int main(int argc,char** argv){@autoreleasepool{try {
    require(argc==3,"Usage: metalrobo_crow_temporal_history_check NEW_OUTPUT CALIBRATED_PORTS");
    const std::filesystem::path output=argv[1];require(std::filesystem::create_directory(output),"Output already exists");
    std::array<mr_float4,2> ports;std::ifstream input(argv[2],std::ios::binary);
    require(bool(input.read(reinterpret_cast<char*>(ports.data()),sizeof(ports))) && input.peek()==std::char_traits<char>::eof(),"Port file shape/read error");
    std::ofstream(output/"plan.json")<<R"({"purpose":"actual native temporal-history ownership and partition regression","lanes":[32,33,64,65],"ticks":65,"chunks":[1,8],"policy":"constant native head yaw0.6 beak0.9; accepted action holds256/16 after decision64","requirements":"every velocity and position history word equals its own lane/action final accepted v/q, validated SHA256 world envelope; complete world/retina/food checkpoints and full actor/critic/transition streams exact across chunks","evidence":"streaming hashes, final snapshots only; no images or video","scope":"engineering control, no learned policy or biological qualification","retry":false})";
    std::uint64_t checked=0;
    for(unsigned lanes:{32u,33u,64u,65u}) {
        const auto a=execute(ports,output/("lanes"+std::to_string(lanes)+"-chunk1"),lanes,1);
        const auto b=execute(ports,output/("lanes"+std::to_string(lanes)+"-chunk8"),lanes,8);
        require(exact(a.world,b.world)&&exact(a.retina,b.retina)&&exact(a.food,b.food)&&a.streams==b.streams,"Native state/streams depend on submission partition at "+std::to_string(lanes)+" lanes");
        checked+=a.checkedHistoryFloats+b.checkedHistoryFloats;
        std::cout<<"lanes="<<lanes<<" history and partition exact\n"<<std::flush;
    }
    std::ofstream(output/"summary.json")<<"{\"status\":\"passed\",\"arms\":8,\"accepted_ticks_per_arm\":65,\"history_float_checks\":"<<checked<<",\"full_checkpoint_and_stream_partition_exact\":true,\"physics_rejections\":0,\"terminations\":0}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
