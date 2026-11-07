#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/GRUMemory.hpp"
#include "metalrobo/task_program_types.h"
#include "metalrobo/detail/Checkpoint.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <chrono>
#include <unistd.h>
extern char** environ;
using namespace metalrobo;
namespace fs=std::filesystem;
namespace {
void require(bool yes,const char* message) {if (!yes) throw std::runtime_error(message);}
template<class T,std::size_t Extent> void save(const fs::path& p,std::span<T,Extent> data) {
    checkpoint::writeExclusive(p,std::as_bytes(data));
}
template<class T> void save(const fs::path& p,const std::vector<T>& data) {save(p,std::span(data));}
template<class T> std::vector<T> load(const fs::path& p) {
    auto bytes=checkpoint::readBounded(p,128ull*1024*1024);require(bytes.size()%sizeof(T)==0,"Invalid fixture payload size");
    std::vector<T> values(bytes.size()/sizeof(T));std::memcpy(values.data(),bytes.data(),bytes.size());return values;
}
struct Fixture {
    unsigned N,I,H,T=40,split=18;
    GRUCellParameters first,next;
    std::vector<float> inputs;
    std::vector<MRTaskStateGPU> tasks;
    Fixture(unsigned n,unsigned i,unsigned h):N(n),I(i),H(h) {
        first.inputs=I;first.hidden=H;first.maximumAbsoluteInput=2;
        const auto weights=[](std::size_t count,double scale,unsigned salt) {std::vector<float> a(count);for (std::size_t k=0;k<count;++k) a[k]=float(scale*std::sin(double(k+1)*(0.127+salt*0.023)));return a;};
        first.inputWeights=weights(std::size_t(3)*H*I,.04,1);first.recurrentWeights=weights(std::size_t(3)*H*H,.035,2);
        first.inputBias=weights(3*H,.1,3);first.recurrentBias=weights(3*H,.07,4);
        next=first;next.inputBias[0]+=.01f;
        inputs.resize(std::size_t(T)*N*I);tasks.resize(std::size_t(T)*N);
        for (unsigned t=0;t<T;++t) for (unsigned n=0;n<N;++n) {
            auto& state=tasks[t*N+n];state.status.x=1;
            state.episode.y=1+(n==1 && t>=11 ? 1u : 0u)+(n==2 && t>=23 ? 1u : 0u);
            state.episode.x=n==1 && t>=11 ? t-11 : n==2 && t>=23 ? t-23 : t;
            for (unsigned j=0;j<I;++j) inputs[(std::size_t(t)*N+n)*I+j]=float(.8*std::sin(.19*(t+1)+.13*n+.017*j));
        }
    }
    void write(const fs::path& out) const {
        fs::create_directory(out);const std::array<unsigned,5> dimensions{N,I,H,T,split};save(out/"dimensions.u32",std::span(dimensions));
        save(out/"inputs.f32",inputs);save(out/"tasks.bin",tasks);
        for (auto item : {std::pair{"first",&first},std::pair{"next",&next}}) {
            save(out/(std::string(item.first)+"-input-weights.f32"),item.second->inputWeights);
            save(out/(std::string(item.first)+"-recurrent-weights.f32"),item.second->recurrentWeights);
            save(out/(std::string(item.first)+"-input-bias.f32"),item.second->inputBias);
            save(out/(std::string(item.first)+"-recurrent-bias.f32"),item.second->recurrentBias);
        }
    }
};
GRUMemoryConfig config(unsigned N) {GRUMemoryConfig c;c.environments=N;return c;}
std::vector<double> cell(const GRUCellParameters& p,const float* input,std::span<const double> hidden) {
    std::vector<double> output(p.hidden);
    for (unsigned h=0;h<p.hidden;++h) {
        double x[3],r[3];
        for (unsigned g=0;g<3;++g) {
            const auto row=g*p.hidden+h;x[g]=p.inputBias[row];r[g]=p.recurrentBias[row];
            for (unsigned j=0;j<p.inputs;++j) x[g]=std::fma(double(p.inputWeights[row*p.inputs+j]),input[j],x[g]);
            for (unsigned j=0;j<p.hidden;++j) r[g]=std::fma(double(p.recurrentWeights[row*p.hidden+j]),hidden[j],r[g]);
        }
        const double reset=1/(1+std::exp(-(x[0]+r[0]))),update=1/(1+std::exp(-(x[1]+r[1])));
        const double proposed=std::tanh(x[2]+reset*r[2]);output[h]=(1-update)*proposed+update*hidden[h];
    }
    return output;
}
struct Output {std::vector<float> hidden,before,preview;std::vector<unsigned> reset;};
Output execute(GRUMemoryContext& context,const Fixture& f,unsigned begin,unsigned end,unsigned chunk,bool previews,bool duplicates,bool previewOnly=false) {
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
    auto taskTimeline=[device newBufferWithBytes:f.tasks.data() length:f.tasks.size()*sizeof(MRTaskStateGPU) options:MTLResourceStorageModeShared];
    auto task=[device newBufferWithLength:f.N*sizeof(MRTaskStateGPU) options:MTLResourceStorageModePrivate];
    auto inputs=[device newBufferWithBytes:f.inputs.data() length:f.inputs.size()*sizeof(float) options:MTLResourceStorageModeShared];
    const auto elements=std::size_t(f.T)*f.N*f.H;
    auto output=[device newBufferWithLength:elements*sizeof(float) options:MTLResourceStorageModeShared];
    auto trace=[device newBufferWithLength:elements*sizeof(float) options:MTLResourceStorageModeShared];
    auto preview=[device newBufferWithLength:elements*sizeof(float) options:MTLResourceStorageModeShared];
    auto scratchTrace=[device newBufferWithLength:f.N*f.H*sizeof(float) options:MTLResourceStorageModePrivate];
    auto reset=[device newBufferWithLength:std::size_t(f.T)*f.N*sizeof(unsigned) options:MTLResourceStorageModeShared];
    auto scratchReset=[device newBufferWithLength:f.N*sizeof(unsigned) options:MTLResourceStorageModePrivate];
    require(queue && task && taskTimeline && inputs && output && trace && preview && scratchTrace && reset && scratchReset,"Fixture GPU allocation failed");
    std::memset(output.contents,0,output.length);std::memset(trace.contents,0,trace.length);std::memset(preview.contents,0,preview.length);std::memset(reset.contents,0,reset.length);
    for (unsigned start=begin;start<end;start+=chunk) {
        auto command=[queue commandBuffer];
        for (unsigned t=start;t<std::min(end,start+chunk);++t) {
            auto blit=[command blitCommandEncoder];[blit copyFromBuffer:taskTimeline sourceOffset:std::size_t(t)*f.N*sizeof(MRTaskStateGPU) toBuffer:task destinationOffset:0 size:task.length];[blit endEncoding];
            if (previews) for (unsigned repeat=0;repeat<2;++repeat)
                require(context.encode((__bridge void*)command,(__bridge void*)task,(__bridge void*)inputs,f.I,(__bridge void*)preview,(__bridge void*)scratchTrace,(__bridge void*)scratchReset,
                    true,std::size_t(t)*f.N*f.I,std::size_t(t)*f.N*f.H),"Preview encoding failed");
            if (!previewOnly) require(context.encode((__bridge void*)command,(__bridge void*)task,(__bridge void*)inputs,f.I,(__bridge void*)output,(__bridge void*)trace,(__bridge void*)reset,
                false,std::size_t(t)*f.N*f.I,std::size_t(t)*f.N*f.H,std::size_t(t)*f.N*f.H,std::size_t(t)*f.N),"Real memory encoding failed");
            if (duplicates) require(context.encode((__bridge void*)command,(__bridge void*)task,(__bridge void*)inputs,f.I,(__bridge void*)output,(__bridge void*)trace,(__bridge void*)scratchReset,
                false,std::size_t(t)*f.N*f.I,std::size_t(t)*f.N*f.H,std::size_t(t)*f.N*f.H),"Duplicate encoding failed");
        }
        [command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted && context.healthy(),"Memory GPU publication failed");
    }
    Output result;const auto offset=std::size_t(begin)*f.N*f.H,count=std::size_t(end-begin)*f.N*f.H;
    result.hidden.assign(static_cast<float*>(output.contents)+offset,static_cast<float*>(output.contents)+offset+count);
    result.before.assign(static_cast<float*>(trace.contents)+offset,static_cast<float*>(trace.contents)+offset+count);
    result.preview.assign(static_cast<float*>(preview.contents)+offset,static_cast<float*>(preview.contents)+offset+count);
    if (previews && !previewOnly) require(result.preview==result.hidden,"Preview and real neural outputs differ");
    result.reset.assign(static_cast<unsigned*>(reset.contents)+begin*f.N,static_cast<unsigned*>(reset.contents)+end*f.N);return result;
}
bool equalState(const GRUMemorySnapshot& a,const GRUMemorySnapshot& b) {
    return a.before==b.before && a.proposed==b.proposed && a.errors==b.errors &&
        a.metadata.size()==b.metadata.size() && std::memcmp(a.metadata.data(),b.metadata.data(),a.metadata.size()*sizeof(mr_uint4))==0;
}
void saveOutput(const fs::path& out,const std::string& name,const Output& v) {save(out/(name+"-hidden.f32"),v.hidden);save(out/(name+"-before.f32"),v.before);save(out/(name+"-reset.u32"),v.reset);}
void compare(const Output& a,const Output& b) {require(a.hidden==b.hidden && a.before==b.before && a.reset==b.reset,"Memory future depends on partition/preview/restore");}
std::vector<double> oracle(const Fixture& f,std::vector<unsigned>& masks) {
    std::vector<double> hidden(f.N*f.H,0),values;std::vector<unsigned> episode(f.N,0);masks.clear();
    for (unsigned t=0;t<f.T;++t) for (unsigned n=0;n<f.N;++n) {
        const bool reset=episode[n]!=f.tasks[t*f.N+n].episode.y;
        if (reset) std::fill_n(hidden.begin()+n*f.H,f.H,0.0);episode[n]=f.tasks[t*f.N+n].episode.y;masks.push_back(reset ? 1u : 0u);
        const auto next=cell(t<f.split ? f.first : f.next,&f.inputs[(std::size_t(t)*f.N+n)*f.I],std::span(hidden).subspan(n*f.H,f.H));
        std::copy(next.begin(),next.end(),hidden.begin()+n*f.H);values.insert(values.end(),next.begin(),next.end());
    }
    return values;
}
void boundedChild(const fs::path& program,const fs::path& out) {
    std::array<std::string,3> args{program.string(),"--child",out.string()};std::array<char*,4> raw{args[0].data(),args[1].data(),args[2].data(),nullptr};pid_t child=0;
    require(posix_spawn(&child,program.c_str(),nullptr,nullptr,raw.data(),environ)==0,"Fresh process spawn failed");
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(40);int status=0;
    while (waitpid(child,&status,WNOHANG)==0) {
        if (std::chrono::steady_clock::now()>deadline) {kill(child,SIGKILL);waitpid(child,&status,0);throw std::runtime_error("Fresh memory continuation timed out");}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require(WIFEXITED(status) && WEXITSTATUS(status)==0,"Fresh memory continuation rejected");
}
void child(const fs::path& out) {
    Fixture f(4,257,65);GRUMemoryContext context(config(f.N),f.next);
    context.restoreState(checkpoint::readBounded(out/"saved.memorycheckpoint",1ull<<26));
    auto tail=execute(context,f,f.split,f.T,8,true,true);saveOutput(out,"fresh-tail",tail);
    checkpoint::writeExclusive(out/"fresh-final.memorycheckpoint",context.checkpointState());
}
void hostNegatives(GRUMemoryContext& context,const Fixture& f) {
    const auto state=context.snapshot();unsigned rejected=0;
    for (unsigned test=0;test<5;++test) {
        auto bad=f.first;if (test==0) bad.inputWeights.pop_back();if (test==1) bad.inputBias[0]=std::numeric_limits<float>::quiet_NaN();
        if (test==2) bad.maximumAbsoluteInput=-1;if (test==3) bad.inputWeights[0]=std::numeric_limits<float>::max();if (test==4) bad.hidden=0;
        try {context.install(bad);} catch (...) {++rejected;}
        require(equalState(state,context.snapshot()),"Rejected weights mutated working memory");
    }
    auto saved=context.checkpointState();auto corrupt=saved;corrupt.back()^=std::byte{1};
    try {context.restoreState(corrupt);} catch (...) {++rejected;}
    require(rejected==6 && equalState(state,context.snapshot()),"Memory parameter/checksum rejection failed");
}
unsigned boundaryNegatives(const Fixture& f) {
    GRUMemoryContext context(config(f.N),f.first);const auto initial=context.checkpointState();
    auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
    auto task=[device newBufferWithBytes:f.tasks.data() length:f.N*sizeof(MRTaskStateGPU) options:MTLResourceStorageModeShared];
    auto input=[device newBufferWithBytes:f.inputs.data() length:f.N*f.I*sizeof(float) options:MTLResourceStorageModeShared];
    auto output=[device newBufferWithLength:f.N*f.H*sizeof(float) options:MTLResourceStorageModeShared];
    auto trace=[device newBufferWithLength:output.length options:MTLResourceStorageModeShared];
    auto reset=[device newBufferWithLength:f.N*sizeof(unsigned) options:MTLResourceStorageModeShared];
    auto shortTask=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    unsigned rejected=0;
    for (unsigned test=0;test<9;++test) {
        auto command=[queue commandBuffer];if (test==8) {[command commit];[command waitUntilCompleted];}
        const auto huge=std::numeric_limits<std::size_t>::max();
        const bool accepted=context.encode(test==0 ? nullptr : (__bridge void*)command,
            (__bridge void*)(test==7 ? shortTask : task),(__bridge void*)input,test==1 ? f.I-1 : f.I,
            (__bridge void*)(test==2 ? input : output),(__bridge void*)trace,(__bridge void*)reset,true,
            test==3 ? huge : 0,test==4 ? huge : 0,test==5 ? huge : 0,test==6 ? huge : 0);
        require(!accepted,"Malformed memory encode was accepted");++rejected;
        require(context.checkpointState()==initial,"Rejected encode mutated ownership/state");
    }
    auto owner=[queue commandBuffer],other=[queue commandBuffer];
    require(context.encode((__bridge void*)owner,(__bridge void*)task,(__bridge void*)input,f.I,(__bridge void*)output,(__bridge void*)trace,(__bridge void*)reset,true),"Valid owner preview rejected");
    require(!context.encode((__bridge void*)other,(__bridge void*)task,(__bridge void*)input,f.I,(__bridge void*)output,(__bridge void*)trace,(__bridge void*)reset,true),"Concurrent command stole memory owner");++rejected;
    for (unsigned test=0;test<5;++test) {
        bool caught=false;try {
            if (test==0) (void)context.snapshot();if (test==1) (void)context.checkpointState();if (test==2) context.reset();
            if (test==3) context.install(f.first);if (test==4) context.restoreState(initial);
        } catch (...) {caught=true;}require(caught,"Busy memory boundary operation was accepted");++rejected;
    }
    require(context.encode((__bridge void*)owner,(__bridge void*)task,(__bridge void*)input,f.I,(__bridge void*)output,(__bridge void*)trace,(__bridge void*)reset,true),"Same-command reuse rejected");
    [owner commit];[owner waitUntilCompleted];require(context.healthy() && context.checkpointState()==initial,"Read-only owner previews mutated state or leaked ownership");
    auto tiny=config(f.N);tiny.maximumRetainedBytes=1;bool caught=false;
    try {GRUMemoryContext denied(tiny,f.first);} catch (...) {caught=true;}
    require(caught,"Memory storage budget bypassed");return rejected+1;
}
void gpuNegative(const Fixture& f,bool nonfinite) {
    GRUMemoryContext context(config(f.N),f.first);execute(context,f,0,1,1,false,false);const auto before=context.snapshot();
    auto broken=f;for (unsigned n=0;n<f.N;++n) broken.tasks[f.N+n].episode.x=nonfinite ? 1u : 3u;
    if (nonfinite) broken.inputs[f.N*f.I]=std::numeric_limits<float>::quiet_NaN();
    bool rejected=false;try {execute(context,broken,1,2,1,false,false);} catch (...) {rejected=true;}
    require(rejected && !context.healthy(),"Invalid GPU memory input/clock was published");auto after=context.snapshot();
    require(before.before==after.before && before.proposed==after.proposed && std::memcmp(before.metadata.data(),after.metadata.data(),before.metadata.size()*sizeof(mr_uint4))==0,
        "Invalid GPU input partially mutated hidden memory");
    require(after.errors[0] && std::any_of(after.errors.begin()+1,after.errors.end(),[](auto x){return x!=0;}),"Memory GPU error was lost");
    context.reset();require(context.healthy(),"Explicit memory reset did not recover poison");execute(context,f,1,2,1,false,false);
}
}
int main(int argc,char** argv) { @autoreleasepool {try {
    if (argc==3 && std::string(argv[1])=="--child") {child(fs::absolute(argv[2]));return 0;}
    require(argc==2,"Usage: gru_memory_check NEW_OUTPUT");const auto out=fs::absolute(argv[1]);require(!fs::exists(out),"Output must be new");fs::create_directories(out);
    std::ofstream(out/"plan.json")<<R"({"schema":"numi.gru-memory-plan.v2","question":"Can native working memory retain causal GRU state without preview, duplicate, partition or process-interruption drift?","conditions":"reset-after r/z/n GRU; I5/H7/N3 and I257/H65/N4;40 authored neural-instrument ticks; lane1 reset11 and lane2 reset23; next weights published at18 without step; chunks1/8 plus duplicate real inference and extra read-only previews; dedicated preview-only calls before initialization and at next physical tick; fresh process restores next-weight-bound checkpoint","prediction":"FP64 error<=3e-6; all future vectors, masks and hidden component checkpoints identical; six malformed parameter/checksum, sixteen buffer/ownership/budget and two GPU-invalid controls reject; previews and installs leave hidden state unchanged","validity":"GPU-only state transition operator with borrowed native TaskState clock; no host wait/readback in encode; host fixture clock is a declared synthetic instrument, not crow simulation","stopping":"two40-tick cases,24 negative controls,one bounded40s independent continuation; retain failure,no threshold changes","scope":"native GRU component admission only; PolicyPack/MetalWorld/recurrent PPO and embodied behavior still required"})";
    double maximum=0;unsigned cases=0;
    for (auto dims : {std::array<unsigned,3>{3,5,7},std::array<unsigned,3>{4,257,65}}) {
        Fixture f(dims[0],dims[1],dims[2]);const auto directory=out/(f.I==5 ? "small" : "large");f.write(directory);
        GRUMemoryContext a(config(f.N),f.first),b(config(f.N),f.first);
        const auto empty=a.checkpointState();execute(a,f,0,1,1,true,false,true);require(a.checkpointState()==empty,"Uninitialized preview advanced memory");
        auto prefix=execute(a,f,0,f.split,1,false,false),other=execute(b,f,0,f.split,8,true,true);compare(prefix,other);
        const auto atSplit=a.checkpointState();execute(a,f,f.split,f.split+1,1,true,false,true);require(a.checkpointState()==atSplit,"Next-clock preview advanced memory");
        const auto beforeInstall=a.snapshot();a.install(f.next);b.install(f.next);require(equalState(beforeInstall,a.snapshot()),"Weight publication advanced memory");
        const auto saved=a.checkpointState();checkpoint::writeExclusive(directory/"saved.memorycheckpoint",saved);
        GRUMemoryContext resumed(config(f.N),f.next);resumed.restoreState(saved);
        auto tail=execute(a,f,f.split,f.T,1,false,false),continued=execute(b,f,f.split,f.T,8,true,true),restored=execute(resumed,f,f.split,f.T,8,true,true);
        compare(tail,continued);compare(tail,restored);
        require(a.checkpointState()==b.checkpointState() && a.checkpointState()==resumed.checkpointState(),"Hidden memory checkpoint differs");
        saveOutput(directory,"prefix",prefix);saveOutput(directory,"tail",tail);
        checkpoint::writeExclusive(directory/"final.memorycheckpoint",a.checkpointState());
        std::vector<unsigned> expectedMasks;auto expected=oracle(f,expectedMasks);std::vector<float> values=prefix.hidden;values.insert(values.end(),tail.hidden.begin(),tail.hidden.end());
        std::vector<unsigned> masks=prefix.reset;masks.insert(masks.end(),tail.reset.begin(),tail.reset.end());require(masks==expectedMasks,"Memory reset mask mismatch");
        for (std::size_t i=0;i<values.size();++i) maximum=std::max(maximum,std::abs(double(values[i])-expected[i]));
        require(maximum<=3e-6,"Native GRU exceeds prospective FP64 tolerance");
        save(directory/"oracle.f64",expected);if (f.I==5) {hostNegatives(a,f);require(boundaryNegatives(f)==16,"Boundary control count mismatch");gpuNegative(f,false);gpuNegative(f,true);}
        ++cases;
    }
    boundedChild(fs::absolute(argv[0]),out/"large");
    for (auto suffix : {"hidden.f32","before.f32","reset.u32"}) require(checkpoint::readBounded(out/"large"/("tail-"+std::string(suffix)),1ull<<26)==checkpoint::readBounded(out/"large"/("fresh-tail-"+std::string(suffix)),1ull<<26),"Fresh-process memory future differs");
    require(checkpoint::readBounded(out/"large/final.memorycheckpoint",1ull<<26)==checkpoint::readBounded(out/"large/fresh-final.memorycheckpoint",1ull<<26),"Fresh hidden memory checkpoint differs");
    std::ostringstream precise;precise<<std::setprecision(17)<<maximum;
    const auto result=std::string("{\"schema\":\"numi.gru-memory-evidence.v2\",\"status\":\"passed\",\"cases\":")+std::to_string(cases)+",\"maximum_FP64_error\":"+precise.str()+",\"partition_preview_duplicate_exact\":true,\"fresh_process_future_exact\":true,\"negative_controls\":24,\"recurrent_policy_integrated\":false,\"trainable_crow_memory_qualified\":false}";
    std::ofstream(out/"summary.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} } }
