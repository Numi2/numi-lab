#import <Foundation/Foundation.h>
#include "CrowRetinalFixture.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <chrono>
#include <signal.h>
#include <sstream>
extern char** environ;
using namespace crow_retinal_fixture;
namespace cp=metalrobo::checkpoint;
namespace fs=std::filesystem;
namespace {
constexpr unsigned I=5,H=7,steps=40,split=18;
constexpr std::uint64_t seed=2651109401;
struct Record {std::vector<float> actor,actions,before,q,v;std::vector<unsigned> reset;std::vector<MRTaskTransitionC> transitions;};
PolicyPack authored(const Fixture& f) {
    PolicyPack p;p.id="native-recurrent-crow-control";p.revision=19;
    PolicyDenseLayer encoder{.inputCount=F,.outputCount=I,.activation=PolicyActivation::tanh,
        .weights=std::vector<float>(F*I,0),.bias=std::vector<float>(I,0)};
    for (unsigned i=0;i<I;++i) {encoder.bias[i]=.02f*float(i+1);encoder.weights[i*F+(17*i)%97]=.03f;encoder.weights[i*F+97+31*i]=.04f;}
    PolicyDenseLayer head{.inputCount=H,.outputCount=18,.activation=PolicyActivation::identity,
        .weights=std::vector<float>(18*H,0),.bias=std::vector<float>(18,0)};
    head.bias[17]=-.2f;head.weights[17*H]=.15f;head.weights[17*H+2]=-.08f;
    p.layers={encoder,head};PolicyGRUMemory m;m.afterLayer=1;m.cell.inputs=I;m.cell.hidden=H;m.cell.maximumAbsoluteInput=1.000001f;
    auto weights=[](unsigned n,double scale) {std::vector<float> a(n);for (unsigned k=0;k<n;++k) a[k]=float(scale*std::sin(.17*(k+1)));return a;};
    m.cell.inputWeights=weights(3*H*I,.12);m.cell.recurrentWeights=weights(3*H*H,.09);
    m.cell.inputBias=weights(3*H,.11);m.cell.recurrentBias=weights(3*H,.07);
    p.actorMemory=m;p.criticLayers={{.inputCount=97,.outputCount=1,.activation=PolicyActivation::identity,.weights=std::vector<float>(97,0),.bias={.25f}}};
    p.actionClip=1;bindPolicyPack(p,f.run.task());return p;
}
void install(MRTaskRolloutHandle* h,const fs::path& file) {api(mr_task_rollout_load_policy_pack(h,file.c_str()),"install recurrent actor");}
void collect(MRTaskRolloutHandle* h,unsigned begin,unsigned end,unsigned chunk,bool previews,Record& r,bool matchEndpoint=false) {
    for (unsigned start=begin;start<end;start+=chunk) {
        const auto count=std::min(chunk,end-start);std::vector<unsigned> resets(count*N,0);
        if (start<=9 && 9<start+count) resets[(9-start)*N+1]=1;
        MRTaskRolloutAdvanceC a{};api(mr_task_rollout_advance(h,nullptr,0,resets.data(),resets.size(),count,mr_task_rollout_policy_revision(h),previews || (matchEndpoint && start+count==steps),&a),"advance recurrent actor");
        require(!a.failed_environment_steps && a.successful_environment_steps==count*N,"Recurrent actor physics failed");
        require(mr_task_rollout_policy_memory_width(h)==H,"Native recurrent trace width lost");
        const auto* actor=mr_task_rollout_actor_observations(h);const auto* action=mr_task_rollout_policy_actions(h);
        const auto* before=mr_task_rollout_policy_memory_before(h);const auto* reset=mr_task_rollout_policy_memory_resets(h);
        require(actor && action && before && reset,"Native recurrent traces missing");
        r.actor.insert(r.actor.end(),actor,actor+count*N*F);r.actions.insert(r.actions.end(),action,action+count*N*18);
        r.before.insert(r.before.end(),before,before+count*N*H);r.reset.insert(r.reset.end(),reset,reset+count*N);
        auto t=mr_task_rollout_transitions(h);r.transitions.insert(r.transitions.end(),t,t+count*N);
    }
    r.q.assign(mr_task_rollout_final_q(h),mr_task_rollout_final_q(h)+N*23);r.v.assign(mr_task_rollout_final_v(h),mr_task_rollout_final_v(h)+N*22);
}
void saveRecord(const fs::path& out,const Record& r) {
    fs::create_directory(out);save(out/"actor.f32",r.actor);save(out/"actions.f32",r.actions);save(out/"before.f32",r.before);save(out/"reset.u32",r.reset);
    save(out/"q.f32",r.q);save(out/"v.f32",r.v);save(out/"transitions.bin",r.transitions);
}
void compare(const Record& a,const Record& b) {
    require(a.actor==b.actor && a.actions==b.actions && a.before==b.before && a.reset==b.reset && a.q==b.q && a.v==b.v &&
        a.transitions.size()==b.transitions.size() && !std::memcmp(a.transitions.data(),b.transitions.data(),a.transitions.size()*sizeof(MRTaskTransitionC)),
        "Recurrent actor future depends on chunks, previews or memory restore");
}
std::vector<double> dense(const PolicyDenseLayer& p,const float* x) {
    std::vector<double> out(p.outputCount);
    for (unsigned i=0;i<p.outputCount;++i) {double a=p.bias[i];for (unsigned j=0;j<p.inputCount;++j) a+=double(p.weights[i*p.inputCount+j])*std::clamp(double(x[j]),-100.,100.);out[i]=p.activation==PolicyActivation::tanh ? std::tanh(a) : a;}return out;
}
std::vector<double> gru(const GRUCellParameters& p,const std::vector<double>& x,const std::vector<double>& h) {
    std::vector<double> out(H);
    for (unsigned i=0;i<H;++i) {
        double a[3],b[3];for (unsigned g=0;g<3;++g) {const auto row=g*H+i;a[g]=p.inputBias[row];b[g]=p.recurrentBias[row];
            for (unsigned j=0;j<I;++j) a[g]+=p.inputWeights[row*I+j]*x[j];for (unsigned j=0;j<H;++j) b[g]+=p.recurrentWeights[row*H+j]*h[j];}
        const double r=1/(1+std::exp(-a[0]-b[0])),z=1/(1+std::exp(-a[1]-b[1]));out[i]=(1-z)*std::tanh(a[2]+r*b[2])+z*h[i];
    }return out;
}
double oracle(const Record& r,const PolicyPack& first,const PolicyPack& next,const fs::path& out) {
    std::array<std::vector<double>,N> hidden;for (auto& h:hidden) h.assign(H,0);double maximum=0;
    std::vector<double> expectedBefore,expectedActions;
    for (unsigned t=0;t<steps;++t) for (unsigned n=0;n<N;++n) {
        const bool reset=t==0 || (t==9 && n==1);require(r.reset[t*N+n]==unsigned(reset),"Episode reset trace disagrees with authored native reset");
        if (reset) hidden[n].assign(H,0);
        for (unsigned j=0;j<H;++j) {maximum=std::max(maximum,std::abs(hidden[n][j]-r.before[(t*N+n)*H+j]));expectedBefore.push_back(hidden[n][j]);}
        const auto& p=t<split ? first : next;const auto input=dense(p.layers[0],r.actor.data()+(t*N+n)*F);hidden[n]=gru(p.actorMemory->cell,input,hidden[n]);
        for (unsigned a=0;a<18;++a) {double value=p.layers[1].bias[a];for (unsigned j=0;j<H;++j) value+=p.layers[1].weights[a*H+j]*hidden[n][j];
            value=std::clamp(value,-1.,1.);maximum=std::max(maximum,std::abs(value-r.actions[(t*N+n)*18+a]));expectedActions.push_back(value);}
    }
    save(out/"oracle-before.f64",expectedBefore);save(out/"oracle-actions.f64",expectedActions);require(maximum<=5e-6,"Real crow actor/FP64 recurrent sequence exceeds prospective tolerance");return maximum;
}
void negatives(const Fixture& f,const PolicyPack& good) {
    CompiledPolicyProgram compiled;check(compilePolicyProgram(good,f.run.task(),compiled),"compile recurrent reference");const auto fp=compiled.fingerprint();
    for (unsigned test=0;test<7;++test) {
        auto bad=good;
        if (test==0) bad.actorMemory->afterLayer=0;if (test==1) bad.actorMemory->afterLayer=bad.layers.size();
        if (test==2) bad.actorMemory->cell.inputWeights.pop_back();if (test==3) bad.actorMemory->cell.maximumAbsoluteInput=.5f;
        if (test==4) bad.actorMemory->cell.recurrentWeights[0]=std::numeric_limits<float>::quiet_NaN();
        if (test==5) bad.layers[1].inputCount=H+1;if (test==6) bad.layers[0].outputCount=I+1;
        require(!compilePolicyProgram(bad,f.run.task(),compiled).succeeded() && compiled.fingerprint()==fp,"Invalid recurrent contract mutated compiled policy");
    }
}
void publicationNegatives(MRTaskRolloutHandle* handle,const Fixture& f,const fs::path& out) {
    auto dense=authored(f);dense.actorMemory.reset();dense.layers[1].inputCount=I;dense.layers[1].weights.assign(18*I,0);
    const auto file=out/"dense-drop-control.policypack";check(writePolicyPack(dense,file),"write dense-drop control");
    require(mr_task_rollout_load_policy_pack(handle,file.c_str())!=0 && mr_task_rollout_policy_revision(handle)==19,
        "Resident recurrent architecture drop was published");
    require(mr_task_rollout_clear_policy(handle)!=0 && mr_task_rollout_policy_revision(handle)==19,
        "Resident recurrent policy clear was published");
    MRPolicyRolloutBatchC batch{};
    require(mr_task_rollout_write_policy_rollout_pack(handle,&batch,"forbidden-legacy-memory",(out/"forbidden-v7.rolloutpack").c_str())!=0 &&
        std::string(mr_last_error()).find("explicit hidden/reset traces")!=std::string::npos && !fs::exists(out/"forbidden-v7.rolloutpack"),
        "Legacy rollout publication silently discarded recurrent history");
}
void staleShaderNegative(Fixture& f,const fs::path& out,const fs::path& shader) {
    const auto path=shader.string();f.c.metallib_path=path.c_str();auto h=f.create();f.c.metallib_path=nullptr;
    install(h.get(),out/"first.policypack");MRTaskRolloutAdvanceC result{};
    require(mr_task_rollout_advance(h.get(),nullptr,0,nullptr,0,1,19,1,&result)!=0 &&
        std::string(mr_last_error()).find("policy ABI4")!=std::string::npos && mr_task_rollout_policy_memory_width(h.get())==0,
        "Stale recurrent policy shader reached execution or rejected for the wrong reason");
}
void child(const fs::path& out,const fs::path& destination) {
    Fixture f(out,seed,true);auto h=f.create();install(h.get(),out/"next.policypack");api(mr_task_rollout_set_state_readback(h.get(),1),"enable state");
    api(mr_task_rollout_restore_life_checkpoint(h.get(),(out/"saved.lifecheckpoint").c_str()),"restore recurrent life");
    Record tail;collect(h.get(),split,steps,8,true,tail);saveRecord(destination,tail);api(mr_task_rollout_save_life_checkpoint(h.get(),(destination/"final.lifecheckpoint").c_str()),"save restored recurrent life");
}
void spawnChild(const fs::path& executable,const fs::path& out) {
    const auto destination=out/"fresh";std::array<std::string,4> a{executable.string(),"--restore",out.string(),destination.string()};std::array<char*,5> args{a[0].data(),a[1].data(),a[2].data(),a[3].data(),nullptr};
    pid_t pid;require(!posix_spawn(&pid,executable.c_str(),nullptr,nullptr,args.data(),environ),"Native recurrent child spawn failed");
    int status;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(40);
    while (true) {const auto wait=waitpid(pid,&status,WNOHANG);if (wait==pid) break;require(wait==0,"Native recurrent child observation failed");
        if (std::chrono::steady_clock::now()>deadline) {kill(pid,SIGKILL);waitpid(pid,&status,0);throw std::runtime_error("Owned recurrent child timed out");}std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    require(WIFEXITED(status) && WEXITSTATUS(status)==0,"Native recurrent child rejected restore");
}
}
int main(int argc,char** argv) { @autoreleasepool {try {
    if (argc==4 && std::string(argv[1])=="--restore") {child(fs::absolute(argv[2]),fs::absolute(argv[3]));return 0;}
    require(argc==3,"Usage: crow_recurrent_actor_check NEW_OUTPUT OLD_POLICY_ABI3_METALLIB");const auto out=fs::absolute(argv[1]);require(!fs::exists(out),"Output must be new");fs::create_directories(out);
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.crow-recurrent-actor-plan.v4","parent":"01 matched all future streams and physical/GRU/food components but rejected endpoint retinal byte comparison because reference deferred the final camera capture while continuation requested it; v2 aligns only the final endpoint observation in all arms, with no additional physical tick or changed numerical tolerance; v3 adds three live publication rejection controls and explicit native component memory accounting; v4 requires a retained policyABI3 shader to reject for the declared ABI4 cause before execution","seed":2651109401,"unit":"two supported native v13 crow lanes, authored encoder5-GRU7-action18 actor; not learned behavior","prediction":"FP64 actor/pre-cell error<=5e-6; all real actor/action/pre-cell/reset/transition/final-state bytes match chunks1/8 despite extra intermediate terminal previews; publish next weights at18 without a physical step then fresh-process restore matches future and all final hidden life components; stale policyABI3 shader rejects before native execution for the declared ABI4 cause; seven invalid contracts reject without compiled output mutation; resident architecture-drop, clear and legacy-v7 publication reject without changing the policy or future","conditions":"40 native ticks; lane1 reset at9; policy19 then20 after18; actual head RGB/proprioception; final endpoint camera capture aligned; native physical/contact/task/RNG,retinal,food and GRU state checkpoint","stopping":"one chunk1 reference,one chunk8 preview comparison and one bounded40s fresh continuation; preserve every failure,no threshold changes","limitations":"authored controller and supported engineering body; recurrent rollout/sequence PPO and embodied memory tasks still required"})";
    Fixture f(out,seed);auto first=authored(f),next=first;next.revision=20;next.actorMemory->cell.inputBias[2*H]+=.02f;
    check(writePolicyPack(first,out/"first.policypack"),"write recurrent format5");check(writePolicyPack(next,out/"next.policypack"),"write next recurrent format5");
    PolicyPack loaded;check(readPolicyPack(out/"first.policypack",loaded),"read recurrent format5");CompiledPolicyProgram c;check(compilePolicyProgram(loaded,f.run.task(),c),"compile recurrent format5");
    require(c.header().abi.x==4 && c.layout().memoryHiddenCount==H && c.memoryParameterFingerprint()==gruCellFingerprint(first.actorMemory->cell),"Compiled memory descriptor lost");negatives(f,first);
    staleShaderNegative(f,out,fs::absolute(argv[2]));
    auto a=f.create(),b=f.create();for (auto* h:{a.get(),b.get()}) {install(h,out/"first.policypack");api(mr_task_rollout_set_state_readback(h,1),"enable state");}
    Record aa,bb;collect(a.get(),0,split,1,false,aa);collect(b.get(),0,split,8,true,bb);compare(aa,bb);publicationNegatives(a.get(),f,out);
    install(a.get(),out/"next.policypack");install(b.get(),out/"next.policypack");
    api(mr_task_rollout_save_life_checkpoint(a.get(),(out/"saved.lifecheckpoint").c_str()),"save next-policy recurrent life");
    Record tail,other;collect(a.get(),split,steps,1,false,tail,true);collect(b.get(),split,steps,8,true,other);compare(tail,other);
    saveRecord(out/"tail",tail);api(mr_task_rollout_save_life_checkpoint(a.get(),(out/"tail/final.lifecheckpoint").c_str()),"save reference recurrent life");
    aa.actor.insert(aa.actor.end(),tail.actor.begin(),tail.actor.end());aa.actions.insert(aa.actions.end(),tail.actions.begin(),tail.actions.end());aa.before.insert(aa.before.end(),tail.before.begin(),tail.before.end());aa.reset.insert(aa.reset.end(),tail.reset.begin(),tail.reset.end());
    save(out/"actor.f32",aa.actor);save(out/"actions.f32",aa.actions);save(out/"before.f32",aa.before);save(out/"reset.u32",aa.reset);const auto error=oracle(aa,first,next,out);
    spawnChild(fs::absolute(argv[0]),out);
    for (const auto* name:{"actor.f32","actions.f32","before.f32","reset.u32","transitions.bin","q.f32","v.f32"})
        require(cp::readBounded(out/"tail"/name,1ull<<26)==cp::readBounded(out/"fresh"/name,1ull<<26),"Fresh recurrent life future differs");
    // Outer life envelope includes scheduler timing, so compare its three
    // exact state-component blobs rather than wall-clock counters.
    auto components=[](const fs::path& file) {auto bytes=cp::readBounded(file,1ull<<26);cp::Reader head{bytes};auto header=head.get<cp::Header>();cp::Reader r{cp::open(bytes,{'N','U','M','I','L','F','0','1'},header.identity)};r.take(3*sizeof(std::uint64_t)+2*sizeof(double));std::vector<cp::Bytes> out;for (unsigned i=0;i<3;++i) {auto b=r.blob();out.emplace_back(b.begin(),b.end());}r.end();return out;};
    require(components(out/"tail/final.lifecheckpoint")==components(out/"fresh/final.lifecheckpoint"),"Fresh recurrent hidden life components differ");
    std::ostringstream result;result<<std::setprecision(17)<<"{\"schema\":\"numi.crow-recurrent-actor-evidence.v3\",\"status\":\"passed\",\"native_control_ticks\":40,\"lanes\":2,\"maximum_FP64_error\":"<<error<<",\"chunk_preview_trace_exact\":true,\"fresh_process_full_life_exact\":true,\"invalid_contract_controls\":7,\"publication_rejection_controls\":3,\"stale_policy_shader_rejected\":true,\"recurrent_PPO_integrated\":false,\"embodied_memory_qualified\":false}";
    std::ofstream(out/"summary.json")<<result.str()<<'\n';std::cout<<result.str()<<'\n';return 0;
} catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;} } }
