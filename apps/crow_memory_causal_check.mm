#import <Foundation/Foundation.h>
#include "CrowCameraFixture.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/RetinalObserver.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace crowCameraFixture;
namespace {
constexpr unsigned lanes=64,ticks=320,features=1251;
using Bytes=checkpoint::Bytes;
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(T));
}
std::string hex(const checkpoint::Digest& d) {
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto b:d)out<<std::setw(2)<<unsigned(b);return out.str();
}
// Preserve every resident field and selector while isolating only the final GRU
// envelope. Parse lengths/checksums rather than masking guessed byte offsets.
std::pair<Bytes,Bytes> splitMemory(const Bytes& world) {
    checkpoint::Reader outer{world};const auto h=outer.get<checkpoint::Header>();
    const auto payload=checkpoint::open(world,{'N','U','M','I','W','R','0','1'},h.identity);
    checkpoint::Reader r{payload};r.take(2*sizeof(std::uint64_t)+sizeof(checkpoint::Digest)+9*sizeof(std::uint64_t)+sizeof(std::uint32_t));
    while(r.remaining.size()>=16 && std::memcmp(r.remaining.data()+8,"NUMIGM01",8)) {
        r.get<std::uint32_t>();r.blob();
    }
    const auto prefix=payload.first(payload.size()-r.remaining.size());const auto memory=r.blob();r.end();
    checkpoint::Reader mr{memory};const auto mh=mr.get<checkpoint::Header>();
    checkpoint::open(memory,{'N','U','M','I','G','M','0','1'},mh.identity);
    return {{prefix.begin(),prefix.end()},{memory.begin(),memory.end()}};
}
void requireCleared(const Bytes& memory) {
    checkpoint::Reader outer{memory};const auto h=outer.get<checkpoint::Header>();
    checkpoint::Reader r{checkpoint::open(memory,{'N','U','M','I','G','M','0','1'},h.identity)};
    r.take(sizeof(checkpoint::Digest));
    for(unsigned i=0;i<4;++i)for(auto b:r.blob())require(b==std::byte{0},"Memory intervention did not clear GRU buffers");
    r.end();
}
struct Evidence {
    std::vector<float> prechoiceQ,prechoiceV,head;
    std::vector<MRFiniteResourceStateGPU> food;
    checkpoint::Digest postcueRGB{},proprioception{};
    unsigned intake=0,correctSide=0;std::uint64_t consumed=0;
};
Evidence execute(const PolicyPack& policy,const std::array<mr_float4,2>& ports,
    const std::filesystem::path& output,std::uint64_t seed,unsigned arm) {@autoreleasepool {
    require(std::filesystem::create_directory(output),"Causal arm output exists");
    auto m=makeManifest(false);applyAmericanCrowForagingFixtureLimits(m);
    m.profile.environmentCount=lanes;m.profile.controlSteps=ticks;m.reality.reset={};
    CompiledRun base;check(compileRun(m,base),"Compile physical fixture");
    const auto visual=makeVisual(base,output,true);
    const auto authored=authorAmericanCrowCuedForagingProgram(base,visual,ports,lanes,seed);
    m.sensors.observation={};m.reality.reset={};
    m.task=makeAmericanCrowCuedForagingTaskPack(m.sensors.observation,m.reality.reset,authored);
    m.task.capacities=m.profile.capacities;m.task.maximumEpisodeSteps=100000;m.reality.reset={};
    applyAmericanCrowMemoryChoiceCurriculum(m.task,64,256,16);m.sensors.deviceVisual=authored.sensor;
    CompiledRun run;check(compileRun(m,run),"Compile causal task");
    CompiledPolicyProgram compiled;check(compilePolicyProgram(policy,run.task(),compiled),"Compile unchanged input policy contract");
    require(compiled.actorMemory()!=nullptr && compiled.layout().stochastic,"Causal admission requires trained stochastic recurrent policy");
    FiniteResourceContext food(authored.resources);
    const auto mode=arm==2 ? RetinalCueEvaluationIntervention::absent : arm==3 ? RetinalCueEvaluationIntervention::swapBinary : RetinalCueEvaluationIntervention::none;
    RetinalObserverContext retina(run,visual.family,visual.manifest,lanes,seed,&food,mode);
    MetalWorldStepConfig step;step.timestepSeconds=.02f;step.physicsSubsteps=4;step.velocityIterations=4;step.finalVelocityIterations=2;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=run.task();step.taskSeed=seed;step.minimumDifficultyBand=step.maximumDifficultyBand=0;step.publishFinalState=true;
    step.policyProgram=compiled;step.devicePhysicsProgram=food.deviceProgram();step.deviceObservationProgram=retina.deviceProgram();
    if(run.flappingWingProgram())step.flappingWingProgram=*run.flappingWingProgram();
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    for(unsigned e=0;e<lanes;++e) {
        q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
        scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());
    }
    MetalWorldConfig config;config.enableResidentCheckpoints=true;config.maximumInFlightSubmissions=1;
    MetalWorldContext world(config);MetalWorldResidentState resident;MetalWorldResult result;Evidence evidence;
    CC_SHA256_CTX rgb{},proprio{};CC_SHA256_Init(&rgb);CC_SHA256_Init(&proprio);
    for(unsigned start=0;start<ticks;start+=8) {
        if(start==40 && arm==1) {
            const auto before=world.checkpointResident(resident,step),retinal=retina.checkpointState();const auto resources=food.checkpointState();
            MetalWorldContext other(config);bool rejected=false;
            try{other.clearResidentPolicyMemoryForEvaluation(resident,step);}catch(const std::runtime_error&){rejected=true;}
            require(rejected,"Foreign context admitted a memory intervention");
            world.clearResidentPolicyMemoryForEvaluation(resident,step);
            const auto after=world.checkpointResident(resident,step);const auto a=splitMemory(before),b=splitMemory(after);
            require(a.first==b.first && a.second!=b.second,"GRU intervention changed resident physics/task state or did not change memory");
            requireCleared(b.second);
            require(retinal==retina.checkpointState() && resources==food.checkpointState(),"GRU intervention changed retinal/resource state");
            world.clearResidentPolicyMemoryForEvaluation(resident,step);
            require(after==world.checkpointResident(resident,step),"GRU clearing is not idempotent");
            checkpoint::writeExclusive(output/"before.world",before);checkpoint::writeExclusive(output/"after.world",after);
            std::ofstream(output/"intervention.json")<<"{\"tick\":40,\"resident_except_GRU_exact\":true,\"retina_and_food_exact\":true,\"GRU_cleared\":true,\"resident_prefix_sha256\":\""<<hex(checkpoint::digest(a.first))<<"\"}\n";
        }
        std::vector<unsigned> resets(8*lanes,0);
        MetalWorldBatch batch{.environmentCount=lanes,.controlStepCount=8,.resetMasks=resets};MetalWorldSubmission ticket;
        if(!start){batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;check(world.initializeResidentState(run.world(),batch,step,resident,ticket),"Initialize causal arm");}
        else check(world.submitResident(run.world(),batch,step,resident,ticket),"Submit causal arm");
        if(!start && arm==1) {
            bool rejected=false;try{world.clearResidentPolicyMemoryForEvaluation(resident,step);}catch(const std::runtime_error&){rejected=true;}
            require(rejected,"Pending submission admitted memory intervention before completion");
        }
        const auto status=ticket.wait(result);check(status,"Complete causal arm");require(!status.failedStepCount,"Causal arm rejected physics");
        for(const auto& t:result.transitions)require(!t.termination.x,"Causal arm physical termination");
        require(result.actorObservations.size()==8*lanes*features && result.policyActions.size()==8*lanes*18,"Incomplete native causal publication");
        if(start<64)for(unsigned t=0;t<8;++t)for(unsigned e=0;e<lanes;++e) {
            const auto* row=result.actorObservations.data()+(t*lanes+e)*features;
            CC_SHA256_Update(&proprio,row,97*sizeof(float));
            if(start>=40)CC_SHA256_Update(&rgb,row+97,1154*sizeof(float));
        }
        if(start==8) {
            const auto images=retina.inspectImages();const auto state=food.snapshot();std::ofstream pixels(output/"cue-pixels.tsv");pixels<<"lane\tallocation_mask\tleft_pixels\tright_pixels\n";
            for(unsigned e=0;e<lanes;++e) {
                std::array<unsigned,2> counts{};
                for(const auto& image:images)for(unsigned p=0;p<W*H;++p) {
                    const auto id=image.identities[e*W*H+p].w;if(id==1000 || id==1001)++counts[id-1000];
                }
                const auto mask=state[e].clock.w;require(mask==1 || mask==2,"Nonbinary allocation");const unsigned side=mask==1 ? 0 : 1;
                if(arm==2)require(!counts[0] && !counts[1],"Absent cue rendered an indicator");
                else {const auto shown=arm==3 ? 1-side : side;require(counts[shown]>0 && counts[1-shown]==0,"Rendered cue intervention does not match allocation");}
                pixels<<e<<'\t'<<mask<<'\t'<<counts[0]<<'\t'<<counts[1]<<'\n';
            }
        }
        if(start==56){evidence.prechoiceQ=result.finalQ;evidence.prechoiceV=result.finalV;}
        if(start==64)for(unsigned e=0;e<lanes;++e)evidence.head.push_back(result.policyActions[e*18+15]);
        if(start==40)for(unsigned e=0;e<lanes;++e) {
            require(result.policyMemoryResets[e]==unsigned(arm==1),"Memory reset trace differs from intervention");
            if(arm==1)for(unsigned h=0;h<compiled.actorMemory()->cell.hidden;++h)
                require(result.policyMemoryBefore[e*compiled.actorMemory()->cell.hidden+h]==0,"Post-intervention hidden input is not zero");
        }
    }
    CC_SHA256_Final(evidence.postcueRGB.data(),&rgb);CC_SHA256_Final(evidence.proprioception.data(),&proprio);
    evidence.food=food.snapshot();std::ofstream rows(output/"lifetimes.tsv");rows<<"lane\tallocation_mask\thead_action_tick64\tconsumed_micrograms\tbites\n"<<std::setprecision(10);
    for(unsigned e=0;e<lanes;++e) {
        const auto& s=evidence.food[e];require(s.clock.w==1 || s.clock.w==2,"Final allocation invalid");
        require(std::uint64_t(s.remaining[0].x)+s.remaining[0].y+s.accounting.y==45000,"Food conservation failed");
        evidence.intake+=s.accounting.y>0;evidence.consumed+=s.accounting.y;
        evidence.correctSide+=s.clock.w==1 ? evidence.head[e]>0 : evidence.head[e]<0;
        rows<<e<<'\t'<<s.clock.w<<'\t'<<evidence.head[e]<<'\t'<<s.accounting.y<<'\t'<<s.accounting.z<<'\n';
    }
    if(arm>=2) {
        bool rejected=false;try{(void)retina.checkpointState();}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Intervened retina published a canonical checkpoint");
    }
    std::ofstream(output/"summary.json")<<"{\"status\":\"completed\",\"seed\":"<<seed<<",\"arm\":"<<arm<<",\"lifetimes\":64,\"intake_lifetimes\":"<<evidence.intake<<",\"correct_head_side\":"<<evidence.correctSide<<",\"consumed_micrograms\":"<<evidence.consumed<<",\"postcue_RGB_sha256\":\""<<hex(evidence.postcueRGB)<<"\",\"prechoice_proprioception_sha256\":\""<<hex(evidence.proprioception)<<"\"}\n";
    return evidence;
}}
}
int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==4,"Usage: crow_memory_causal_check NEW_OUTPUT POLICY PORTS");
    const std::filesystem::path out=argv[1];require(std::filesystem::create_directory(out),"Output exists");
    PolicyPack policy;check(readPolicyPack(argv[2],policy),"Read unchanged input policy");
    std::array<mr_float4,2> ports;std::ifstream input(argv[3],std::ios::binary);require(bool(input.read(reinterpret_cast<char*>(ports.data()),sizeof(ports))) && input.peek()==std::char_traits<char>::eof(),"Invalid food ports");
    std::ofstream(out/"plan.json")<<R"({"budget":"Exactly3 held-out allocation/action seeds x4 matched64-lane320-tick arms=245760 accepted transitions, no learning or retries; bounded tabular evidence, no video/full rollouts","arms":["intact","clear_GRU_only_at40","absent_rendered_cue","swapped_rendered_cue"],"seeds":[2651213402,2651213403,2651213404],"prediction":"Native memory intervention preserves every resident field except GRU, plus complete retina/food bytes. Pre-choice physics/proprioception and postcue RGB agree across arms. Cue interventions change only rendered indicator visibility; trained stochastic policy bytes/contracts are unchanged.","limits":"Fixed head-reachable fixture and allocation/action-seed generalization only; broader geometry, natural ecology, manipulation/social learning and biological fidelity unqualified"})";
    std::array<unsigned,4> intake{},correct{};std::array<std::uint64_t,4> food{};
    for(std::uint64_t seed:{2651213402ull,2651213403ull,2651213404ull}) {
        std::array<Evidence,4> arms;
        for(unsigned arm=0;arm<4;++arm) {
            arms[arm]=execute(policy,ports,out/(std::to_string(seed)+"-arm"+std::to_string(arm)),seed,arm);
            intake[arm]+=arms[arm].intake;correct[arm]+=arms[arm].correctSide;food[arm]+=arms[arm].consumed;
            if(arm) {
                require(exact(arms[0].prechoiceQ,arms[arm].prechoiceQ) && exact(arms[0].prechoiceV,arms[arm].prechoiceV),"Cue/memory intervention changed pre-choice physics");
                require(arms[0].postcueRGB==arms[arm].postcueRGB && arms[0].proprioception==arms[arm].proprioception,"Cue/memory intervention changed postcue RGB or pre-choice proprioception");
                for(unsigned e=0;e<lanes;++e)require(arms[0].food[e].clock.w==arms[arm].food[e].clock.w,"Intervention changed food allocation");
            }
        }
    }
    std::ofstream report(out/"summary.json");report<<"{\"status\":\"passed_engineering_controls\",\"accepted_transitions\":245760,\"policy_revision\":"<<policy.revision<<",\"policy_sha256\":\""<<hex(checkpoint::fileDigest(argv[2]))<<"\",\"lifetimes_per_arm\":192,\"intake_lifetimes\":[";
    for(unsigned a=0;a<4;++a){if(a)report<<',';report<<intake[a];}report<<"],\"correct_head_side\":[";
    for(unsigned a=0;a<4;++a){if(a)report<<',';report<<correct[a];}report<<"],\"consumed_micrograms\":[";
    for(unsigned a=0;a<4;++a){if(a)report<<',';report<<food[a];}report<<"],\"learned_memory_qualified\":false}\n";
    std::cout<<"Completed native causal controls; see summary.json and all paired per-lane outcomes.\n";return 0;
}catch(const std::exception& error){std::cerr<<"crow_memory_causal_check failed: "<<error.what()<<'\n';return 1;}}}
