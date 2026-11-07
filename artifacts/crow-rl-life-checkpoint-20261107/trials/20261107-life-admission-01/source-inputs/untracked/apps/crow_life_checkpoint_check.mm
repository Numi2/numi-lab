#import <Foundation/Foundation.h>
#include "CrowRetinalFixture.hpp"
#include "metalrobo/finite_resource_types.h"
#include "metalrobo/detail/Checkpoint.hpp"
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <thread>
#include <chrono>
extern char** environ;
using namespace crow_retinal_fixture;
namespace cp=metalrobo::checkpoint;
namespace {
constexpr std::uint64_t seed=2651109001;
constexpr std::array<char,8> kind{'N','U','M','I','L','F','0','1'};
struct Evidence {std::vector<float> actor,actions,latents,q,v;std::vector<MRTaskTransitionC> transitions;};
void advance(MRTaskRolloutHandle* h,unsigned steps,const std::vector<unsigned>& masks={}) {
    MRTaskRolloutAdvanceC a{};api(mr_task_rollout_advance(h,nullptr,0,masks.empty()?nullptr:masks.data(),masks.size(),steps,19,1,&a),"advance life");
    require(a.successful_environment_steps==steps*N && !a.failed_environment_steps,"Life physics failed");
}
void install(MRTaskRolloutHandle* h,const std::filesystem::path& path) {api(mr_task_rollout_load_policy_pack(h,path.c_str()),"install exact life policy");}
void continueLife(MRTaskRolloutHandle* h,const std::filesystem::path& out,unsigned chunk) {
    require(std::filesystem::create_directory(out),"Continuation output exists");api(mr_task_rollout_set_state_readback(h,1),"enable state instrument");Evidence e;
    for (unsigned start=0;start<32;start+=chunk) {
        const unsigned count=std::min(chunk,32-start);std::vector<unsigned> masks(count*N,0);
        if (start<=9 && 9<start+count) masks[(9-start)*N+1]=1;advance(h,count,masks);
        const auto* actor=mr_task_rollout_actor_observations(h);
        e.actor.insert(e.actor.end(),actor,actor+count*N*F);
        if (start+count==32) e.actor.insert(e.actor.end(),actor+count*N*F,actor+(count+1)*N*F);
        const auto* a=mr_task_rollout_policy_actions(h);const auto* l=mr_task_rollout_policy_latents(h);const auto* t=mr_task_rollout_transitions(h);
        e.actions.insert(e.actions.end(),a,a+count*N*18);e.latents.insert(e.latents.end(),l,l+count*N*18);e.transitions.insert(e.transitions.end(),t,t+count*N);
    }
    e.q.assign(mr_task_rollout_final_q(h),mr_task_rollout_final_q(h)+N*23);e.v.assign(mr_task_rollout_final_v(h),mr_task_rollout_final_v(h)+N*22);
    save(out/"actor.f32",e.actor);save(out/"actions.f32",e.actions);save(out/"latents.f32",e.latents);save(out/"transitions.bin",e.transitions);save(out/"q.f32",e.q);save(out/"v.f32",e.v);
    std::array<MRTaskResourceAccountingC,N> food{};api(mr_task_rollout_copy_resource_accounting(h,food.data(),N),"publish life resources");save(out/"food.bin",std::vector(food.begin(),food.end()));
    api(mr_task_rollout_save_life_checkpoint(h,(out/"final.lifecheckpoint").c_str()),"save continued life");
}
std::array<cp::Bytes,3> components(const cp::Bytes& bytes) {
    cp::Reader envelope{bytes};const auto header=envelope.get<cp::Header>();cp::Reader r{cp::open(bytes,kind,header.identity)};
    r.take(3*sizeof(std::uint64_t)+2*sizeof(double));std::array<cp::Bytes,3> result;
    for (auto& item:result) {auto data=r.blob();item.assign(data.begin(),data.end());}r.end();return result;
}
void compare(const std::filesystem::path& left,const std::filesystem::path& right) {
    for (const auto* name: {"actor.f32","actions.f32","latents.f32","transitions.bin","q.f32","v.f32","food.bin"})
        require(cp::readBounded(left/name,64ull*1024*1024)==cp::readBounded(right/name,64ull*1024*1024),std::string("Life continuation differs: ")+name);
    const auto a=components(cp::readBounded(left/"final.lifecheckpoint",64ull*1024*1024));
    const auto b=components(cp::readBounded(right/"final.lifecheckpoint",64ull*1024*1024));
    for (unsigned i=0;i<3;++i) require(a[i]==b[i],"Continuing hidden native/sensory/resource state differs");
}
void child(const std::filesystem::path& program,const std::filesystem::path& out,unsigned chunk) {
    const auto destination=out/("restored-chunk"+std::to_string(chunk));
    std::vector<std::string> args{program.string(),"--restore",out.string(),destination.string(),std::to_string(chunk)};
    std::vector<char*> argv;for (auto& s:args) argv.push_back(s.data());argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;require(!posix_spawn_file_actions_init(&actions),"Spawn setup failed");
    const auto log=out/("child-chunk"+std::to_string(chunk)+".log");
    require(!posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,log.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600),"Child log exists");
    require(!posix_spawn_file_actions_adddup2(&actions,STDOUT_FILENO,STDERR_FILENO),"Child log redirect failed");
    pid_t pid=0;const auto spawned=posix_spawn(&pid,program.c_str(),&actions,nullptr,argv.data(),environ);posix_spawn_file_actions_destroy(&actions);require(!spawned,"Life child spawn failed");
    const auto start=std::chrono::steady_clock::now();int status=0;
    while (true) {
        const auto waited=waitpid(pid,&status,WNOHANG);if (waited==pid) break;require(waited==0,"Life child observation failed");
        if (std::chrono::steady_clock::now()-start>std::chrono::seconds(40)) {kill(pid,SIGTERM);waitpid(pid,&status,0);throw std::runtime_error("Life child timed out; owned child terminated, evidence retained");}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    require(WIFEXITED(status) && WEXITSTATUS(status)==0,"Life child failed; inspect retained log");
}
cp::Bytes rewriteComponent(const cp::Bytes& original,unsigned component,const cp::Bytes& replacement) {
    cp::Reader envelope{original};const auto header=envelope.get<cp::Header>();cp::Reader r{cp::open(original,kind,header.identity)};
    cp::Bytes payload;const auto counters=r.take(3*sizeof(std::uint64_t)+2*sizeof(double));payload.insert(payload.end(),counters.begin(),counters.end());
    for (unsigned i=0;i<3;++i) {const auto old=r.blob();cp::appendBlob(payload,i==component?std::span<const std::byte>(replacement):old);}r.end();return cp::seal(kind,header.identity,payload);
}
void negatives(Fixture& f,const std::filesystem::path& out,const std::filesystem::path& policy) {
    const auto original=cp::readBounded(out/"saved.lifecheckpoint",64ull*1024*1024);const auto sections=components(original);
    std::ofstream rows(out/"negative-controls.jsonl");
    for (unsigned test=0;test<10;++test) {@autoreleasepool {
        auto bytes=original;auto h=f.create();install(h.get(),policy);
        if (test==0) bytes.resize(20);
        if (test==1) bytes.push_back(std::byte{1});
        if (test==2) bytes.back()^=std::byte{1};
        if (test==3) install(h.get(),out/"neutral.policypack");
        if (test==4) {api(mr_task_rollout_reset(h.get(),seed+1),"change seed for negative control");}
        if (test==5) {auto c=f.c;c.profile.environment_count=1;h=CrowRunOwner(mr_create_task_rollout_retinal(&c,&f.retina),mr_task_rollout_destroy);require(bool(h),"Construct changed lane count");install(h.get(),policy);}
        if (test==6 || test==7) {
            const unsigned part=test==6?1:2;cp::Reader r{sections[part]};auto header=r.get<cp::Header>();cp::Bytes payload(r.remaining.begin(),r.remaining.end());
            // Both checksums remain valid: the component must reject semantics.
            if (part==1) payload[0]^=std::byte{1};
            else {MRFiniteResourceStateGPU state;std::memcpy(&state,payload.data(),sizeof(state));state.accounting.x+=1;std::memcpy(payload.data(),&state,sizeof(state));}
            bytes=rewriteComponent(original,part,cp::seal(header.kind,header.identity,payload));
        }
        if (test==8) {auto broken=sections[0];broken.resize(broken.size()-1);bytes=rewriteComponent(original,0,broken);}
        cp::Bytes before;
        if (test==9) {advance(h.get(),1);const auto path=out/"advanced-before.lifecheckpoint";api(mr_task_rollout_save_life_checkpoint(h.get(),path.c_str()),"save negative before");before=cp::readBounded(path,64ull*1024*1024);}
        const auto path=out/("invalid-"+std::to_string(test)+".lifecheckpoint");cp::writeExclusive(path,bytes);
        const auto status=mr_task_rollout_restore_life_checkpoint(h.get(),path.c_str());const auto reason=std::string(mr_last_error());
        rows<<"{\"case\":"<<test<<",\"rejected\":"<<(status!=0?"true":"false")<<",\"reason\":"<<std::quoted(reason)<<"}\n";rows.flush();require(status!=0,"Invalid life restore accepted");
        if (test==9) {const auto after=out/"advanced-after.lifecheckpoint";api(mr_task_rollout_save_life_checkpoint(h.get(),after.c_str()),"save negative after");require(before==cp::readBounded(after,64ull*1024*1024),"Rejected restore mutated existing life");}
        else if (test!=4 && test!=5) {advance(h.get(),8);const auto* actor=mr_task_rollout_actor_observations(h.get());require(actor[1249]==0 && actor[N*F+1249]==0 && actor[2*N*F+1249]==1,"Rejected restore changed fresh retinal history");std::array<MRTaskResourceAccountingC,N> food{};api(mr_task_rollout_copy_resource_accounting(h.get(),food.data(),N),"inspect rejection purity");for (const auto& item:food) require(item.crop_micrograms==0 && item.remaining_micrograms==45000 && item.accepted_control_ticks==8,"Rejected restore mutated resource state");}
    }}
}
}
int main(int argc,char** argv) {@autoreleasepool {bool owns=false;try {
    if (argc==5 && std::string(argv[1])=="--restore") {
        const std::filesystem::path out=argv[2],destination=argv[3];Fixture f(out,seed,true);auto h=f.create();install(h.get(),out/"stochastic-vision.policypack");
        api(mr_task_rollout_restore_life_checkpoint(h.get(),(out/"saved.lifecheckpoint").c_str()),"restore life in fresh process");
        continueLife(h.get(),destination,std::stoul(argv[4]));return 0;
    }
    require(argc==2,"Usage: metalrobo_crow_life_checkpoint_check NEW_OUTPUT");const std::filesystem::path out=argv[1];require(std::filesystem::create_directory(out),"Output exists");owns=true;
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-life-checkpoint-plan.v1","seed":2651109001,"question":"Does one v13 checkpoint close native physical/contact/task/RNG, retinal pending/visible history and resource state across process interruption?","unit":"two lanes in fixed supported development proxy scene; one authored stochastic image-dependent actor, not learned biological memory","prediction":{"all_future_actor_action_latent_transition_final_q_v_and_resource_bytes_identical":true,"all_three_final_hidden_state_components_identical":true,"negative_controls_rejected_without_published_mutation":10},"conditions":"64 neutral settle,32 jaw-open,32 neutral-close then7 stochastic ticks; snapshot at135 with latency2 pending capture134; uninterrupted32 ticks vs fresh processes at chunk1 and8, lane1 reset at local9","validity":"exact code/shader/device/OS and world/task/policy/sensor identities, consumed food before save, zero failed native/sensor steps, no restore warmup transition","stopping":"exactly one reference, two bounded40s child continuations and ten negative controls; preserve every failure; no threshold changes","scope":"native v13 supported feeding fixture only; learner optimizer remains an exact external input; no persistent trainable memory, ecology or biological qualification"})";
    Fixture f(out,seed);auto h=f.create();const auto neutral=f.policy(out,"neutral"),open=f.policy(out,"open",1);
    install(h.get(),neutral);advance(h.get(),32);advance(h.get(),32);install(h.get(),open);advance(h.get(),32);install(h.get(),neutral);advance(h.get(),32);
    std::array<MRTaskResourceAccountingC,N> food{};api(mr_task_rollout_copy_resource_accounting(h.get(),food.data(),N),"publish consumed food");for (const auto& item:food) require(item.consumed_micrograms>0,"Source life did not consume food");save(out/"source-food.bin",std::vector(food.begin(),food.end()));
    PolicyPack policy;check(readPolicyPack(neutral,policy),"read policy source");policy.id="stochastic-image-dependent-life-control";policy.revision=19;policy.actionLogStandardDeviation.assign(18,-1.5f);
    const unsigned cell=97+(6*16+8)*3;policy.layers[0].weights[15*F+cell]=.3f;policy.layers[0].weights[15*F+cell+2]=-.3f;
    check(writePolicyPack(policy,out/"stochastic-vision.policypack"),"write stochastic vision control");install(h.get(),out/"stochastic-vision.policypack");advance(h.get(),7);
    api(mr_task_rollout_save_life_checkpoint(h.get(),(out/"saved.lifecheckpoint").c_str()),"save native life");
    require(mr_task_rollout_save_life_checkpoint(h.get(),(out/"saved.lifecheckpoint").c_str())!=0,"Save overwrote an existing checkpoint");
    continueLife(h.get(),out/"uninterrupted",8);
    child(std::filesystem::absolute(argv[0]),out,1);compare(out/"uninterrupted",out/"restored-chunk1");
    child(std::filesystem::absolute(argv[0]),out,8);compare(out/"uninterrupted",out/"restored-chunk8");negatives(f,out,out/"stochastic-vision.policypack");
    const std::string result=R"({"schema":"numi.crow-life-checkpoint-evidence.v1","status":"passed","fresh_process_continuations":2,"chunk1_8_future_stream_bit_identical":true,"all_hidden_native_retinal_resource_state_bit_identical":true,"negative_controls_rejected":10,"rejected_restore_pure":true,"restore_advances_physics":false,"whole_v13_development_life_restore":true,"trainable_memory_qualified":false,"ecological_environment_qualified":false})";
    std::ofstream(out/"result.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
} catch (const std::exception& e) {if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}}}
