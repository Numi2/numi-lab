#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/GRUMemory.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include "metalrobo/task_program_types.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <stdexcept>
namespace metalrobo {
namespace {
void require(bool yes,const char* message) {if (!yes) throw std::runtime_error(message);}
std::vector<float> validatedArena(const GRUCellParameters& p) {
    const std::uint64_t I=p.inputs,H=p.hidden,rows=3*H;
    require(I && H && I<=65536 && H<=4096 && std::isfinite(p.maximumAbsoluteInput) && p.maximumAbsoluteInput>0,
        "Invalid GRU dimensions or input bound");
    const std::uint64_t count=rows*(I+H+2);
    require(count<=std::numeric_limits<unsigned>::max()/sizeof(float) &&
        p.inputWeights.size()==rows*I && p.recurrentWeights.size()==rows*H &&
        p.inputBias.size()==rows && p.recurrentBias.size()==rows,"GRU tensor shape/index mismatch");
    std::vector<float> arena;arena.reserve(count);
    for (const auto* values : {&p.inputWeights,&p.recurrentWeights,&p.inputBias,&p.recurrentBias}) {
        require(std::all_of(values->begin(),values->end(),[](float x){return std::isfinite(x);}),"GRU tensors must be finite");
        arena.insert(arena.end(),values->begin(),values->end());
    }
    constexpr double maximum=double(std::numeric_limits<float>::max())/8;
    for (std::uint64_t row=0;row<rows;++row) {
        double x=std::abs(double(p.inputBias[row])),h=std::abs(double(p.recurrentBias[row]));
        for (std::uint64_t i=0;i<I;++i) x+=std::abs(double(p.inputWeights[row*I+i]))*p.maximumAbsoluteInput;
        for (std::uint64_t i=0;i<H;++i) h+=std::abs(double(p.recurrentWeights[row*H+i]))*1.000001;
        require(x<=maximum && h<=maximum,"GRU FP32 accumulator bound exceeded");
    }
    return arena;
}
std::uint64_t fingerprint(const GRUCellParameters& p,std::span<const float> arena) {
    checkpoint::Bytes bytes;checkpoint::append(bytes,std::uint32_t(MR_GRU_MEMORY_ABI));
    checkpoint::append(bytes,p.inputs);checkpoint::append(bytes,p.hidden);checkpoint::append(bytes,p.maximumAbsoluteInput);
    checkpoint::appendBlob(bytes,std::as_bytes(arena));const auto h=checkpoint::digest(bytes);
    std::uint64_t result=0;std::memcpy(&result,h.data(),8);return result ? result : 1;
}
std::filesystem::path libraryPath(const GRUMemoryConfig& c) {
    if (!c.metallibPath.empty()) return c.metallibPath;
    static const char anchor=0;Dl_info image{};
    require(dladdr(&anchor,&image)!=0 && image.dli_fname,"GRU native library identity unavailable");
    auto directory=std::filesystem::path(image.dli_fname).parent_path();
    for (auto p : {directory/"metalrobo/GRUMemory.metallib",directory.parent_path()/"shaders/GRUMemory.metallib"})
        if (std::filesystem::is_regular_file(p)) return p;
    throw std::runtime_error("GRU metallib unavailable");
}
}
std::uint64_t gruCellFingerprint(const GRUCellParameters& p) {const auto arena=validatedArena(p);return fingerprint(p,arena);}
namespace detail {
struct GRUMemoryRuntime {
    GRUMemoryConfig config;std::uint32_t inputs=0,hidden=0;float inputBound=0;
    std::uint64_t parameterFingerprint=0;checkpoint::Digest fixedIdentity{};
    __strong id<MTLDevice> device=nil;__strong id<MTLCommandQueue> queue=nil;
    __strong id<MTLComputePipelineState> validate=nil,prepare=nil,cell=nil;
    __strong id<MTLBuffer> parameters=nil,metadata=nil,before=nil,proposed=nil,errors=nil;
    mutable std::mutex mutex;std::uintptr_t activeToken=0;__weak id<MTLCommandBuffer> activeCommand=nil;bool poisoned=false;
};
}
namespace {
using Runtime=detail::GRUMemoryRuntime;
std::array<id<MTLBuffer>,4> stateBuffers(const Runtime& s) {return {s.metadata,s.before,s.proposed,s.errors};}
checkpoint::Digest identity(const Runtime& s) {
    checkpoint::Bytes bytes;checkpoint::append(bytes,s.fixedIdentity);checkpoint::append(bytes,s.parameterFingerprint);
    checkpoint::append(bytes,s.config.environments);return checkpoint::digest(bytes);
}
id<MTLBuffer> privateBuffer(Runtime& s,std::size_t bytes) {
    auto result=[s.device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];require(result!=nil,"GRU private allocation failed");return result;
}
void wait(id<MTLCommandBuffer> command) {[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"GRU boundary GPU operation failed");}
}
GRUMemoryContext::GRUMemoryContext(GRUMemoryConfig c,GRUCellParameters p) {
    const auto arena=validatedArena(p);
    const std::uint64_t elements=std::uint64_t(c.environments)*p.hidden;
    const std::uint64_t retained=2*elements*sizeof(float)+std::uint64_t(c.environments)*sizeof(mr_uint4)+
        (std::uint64_t(c.environments)+1)*sizeof(unsigned)+arena.size()*sizeof(float);
    require(c.environments && c.environments<=65536 && elements<=std::numeric_limits<unsigned>::max() && retained<=c.maximumRetainedBytes,
        "GRU retained storage/index admission exceeded");
    auto s=std::make_shared<Runtime>();s->config=std::move(c);s->inputs=p.inputs;s->hidden=p.hidden;
    s->device=MTLCreateSystemDefaultDevice();require(s->device!=nil,"GRU requires Metal");s->queue=[s->device newCommandQueue];require(s->queue!=nil,"GRU queue unavailable");
    const auto path=libraryPath(s->config);checkpoint::Bytes fixed;checkpoint::append(fixed,checkpoint::fileDigest(path));
    static const char anchor=0;Dl_info image{};require(dladdr(&anchor,&image)!=0 && image.dli_fname,"GRU library identity unavailable");
    checkpoint::append(fixed,checkpoint::fileDigest(image.dli_fname));checkpoint::append(fixed,std::uint64_t(s->device.registryID));
    const auto os=std::string([NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
    checkpoint::appendBlob(fixed,std::as_bytes(std::span(os.data(),os.size())));s->fixedIdentity=checkpoint::digest(fixed);
    NSError* error=nil;auto library=[s->device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&error];
    require(library!=nil,"GRU shader library load failed");
    const auto pipeline=[&](NSString* name) {auto f=[library newFunctionWithName:name];require(f!=nil,"GRU shader entry missing");
        auto result=[s->device newComputePipelineStateWithFunction:f error:&error];require(result!=nil,"GRU shader pipeline failed");return result;};
    s->validate=pipeline(@"mr_gru_memory_validate");s->prepare=pipeline(@"mr_gru_memory_prepare");s->cell=pipeline(@"mr_gru_memory_cell");
    require(s->validate.maxTotalThreadsPerThreadgroup>=32 && s->prepare.maxTotalThreadsPerThreadgroup>=32,"GRU requires SIMD32 group capacity");
    s->metadata=privateBuffer(*s,c.environments*sizeof(mr_uint4));s->before=privateBuffer(*s,elements*sizeof(float));s->proposed=privateBuffer(*s,elements*sizeof(float));
    s->errors=[s->device newBufferWithLength:(c.environments+1)*sizeof(unsigned) options:MTLResourceStorageModeShared];require(s->errors!=nil,"GRU error storage failed");
    runtime_=s;install(std::move(p));reset();
}
GRUMemoryContext::~GRUMemoryContext()=default;
void GRUMemoryContext::install(GRUCellParameters p) {
    auto arena=validatedArena(p);auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->activeToken && !s->poisoned,"GRU weight installation requires idle healthy owner");
    require(p.inputs==s->inputs && p.hidden==s->hidden,"GRU weight installation cannot change memory shape");
    auto parameters=privateBuffer(*s,arena.size()*sizeof(float));auto staging=[s->device newBufferWithBytes:arena.data() length:arena.size()*sizeof(float) options:MTLResourceStorageModeShared];
    auto command=[s->queue commandBuffer];auto blit=[command blitCommandEncoder];require(staging && command && blit,"GRU parameter upload allocation failed");
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:parameters destinationOffset:0 size:staging.length];[blit endEncoding];wait(command);
    s->parameters=parameters;s->inputBound=p.maximumAbsoluteInput;s->parameterFingerprint=fingerprint(p,arena);
}
void GRUMemoryContext::reset() {
    auto s=runtime_;const std::lock_guard lock(s->mutex);require(!s->activeToken,"GRU reset requires idle ownership");
    auto command=[s->queue commandBuffer];auto blit=[command blitCommandEncoder];require(command && blit,"GRU reset encoder failed");
    for (auto b:stateBuffers(*s)) [blit fillBuffer:b range:NSMakeRange(0,b.length) value:0];[blit endEncoding];
    try {wait(command);s->poisoned=false;} catch (...) {s->poisoned=true;throw;}
}
bool GRUMemoryContext::encode(void* rawCommand,void* rawTasks,void* rawInput,unsigned stride,
    void* rawOutput,void* rawTrace,void* rawReset,bool preview,std::size_t inputBase,std::size_t outputBase,std::size_t traceBase,std::size_t resetBase) {
    auto s=runtime_;if (!rawCommand || !rawTasks || !rawInput || !rawOutput || !rawTrace || !rawReset || stride<s->inputs) return false;
    id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)rawCommand;
    const std::array<id<MTLBuffer>,5> buffers{(__bridge id<MTLBuffer>)rawTasks,(__bridge id<MTLBuffer>)rawInput,
        (__bridge id<MTLBuffer>)rawOutput,(__bridge id<MTLBuffer>)rawTrace,(__bridge id<MTLBuffer>)rawReset};
    if (command.status>MTLCommandBufferStatusEnqueued || command.device.registryID!=s->device.registryID) return false;
    for (unsigned i=0;i<buffers.size();++i) {
        if (buffers[i].device.registryID!=s->device.registryID) return false;
        for (unsigned j=0;j<i;++j) if (buffers[i]==buffers[j]) return false;
    }
    const auto fits=[](std::size_t base,std::uint64_t count,std::size_t bytes) {return base<=bytes/sizeof(float) && count<=bytes/sizeof(float)-base;};
    const auto N=s->config.environments;const std::uint64_t features=std::uint64_t(N)*s->hidden;
    if (buffers[0].length<std::size_t(N)*sizeof(MRTaskStateGPU) ||
        !fits(inputBase,std::uint64_t(N-1)*stride+s->inputs,buffers[1].length) || !fits(outputBase,features,buffers[2].length) ||
        !fits(traceBase,features,buffers[3].length) || !fits(resetBase,N,buffers[4].length)) return false;
    const auto token=reinterpret_cast<std::uintptr_t>(rawCommand);const std::lock_guard lock(s->mutex);
    if (s->poisoned || (s->activeToken && s->activeCommand!=command)) return false;
    auto encoder=[command computeCommandEncoder];if (!encoder) return false;
    if (!s->activeToken) {
        s->activeToken=token;s->activeCommand=command;
        [command addCompletedHandler:^(id<MTLCommandBuffer> done) {
            const std::lock_guard finished(s->mutex);
            const auto* errors=static_cast<const unsigned*>(s->errors.contents);
            if (done.status!=MTLCommandBufferStatusCompleted || errors[0]) s->poisoned=true;
            if (s->activeToken==token) {s->activeToken=0;s->activeCommand=nil;}
        }];
    }
    MRGRUMemoryDispatchGPU p{};p.counts={N,s->inputs,s->hidden,MR_GRU_MEMORY_ABI};p.mode={preview ? 1u : 0u,stride,0,0};
    p.inputBase=inputBase;p.outputBase=outputBase;p.traceBase=traceBase;p.resetBase=resetBase;p.parameterFingerprint=s->parameterFingerprint;p.limits={s->inputBound,0,0,0};
    [encoder setBuffer:buffers[1] offset:0 atIndex:0];[encoder setBuffer:buffers[0] offset:0 atIndex:1];
    [encoder setBuffer:s->metadata offset:0 atIndex:2];[encoder setBuffer:s->before offset:0 atIndex:3];[encoder setBuffer:s->proposed offset:0 atIndex:4];
    [encoder setBuffer:s->errors offset:0 atIndex:5];[encoder setBuffer:s->parameters offset:0 atIndex:6];
    [encoder setBuffer:buffers[2] offset:0 atIndex:7];[encoder setBuffer:buffers[3] offset:0 atIndex:8];[encoder setBuffer:buffers[4] offset:0 atIndex:9];[encoder setBytes:&p length:sizeof(p) atIndex:10];
    const auto group=[&](id<MTLComputePipelineState> pipeline) {[encoder setComputePipelineState:pipeline];[encoder dispatchThreadgroups:MTLSizeMake(N,1,1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,pipeline.maxTotalThreadsPerThreadgroup),1,1)];[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];};
    group(s->validate);if (!preview) group(s->prepare);
    [encoder setComputePipelineState:s->cell];[encoder dispatchThreads:MTLSizeMake(features,1,1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,s->cell.maxTotalThreadsPerThreadgroup),1,1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];[encoder endEncoding];return true;
}
bool GRUMemoryContext::healthy() const {const auto s=runtime_;const std::lock_guard lock(s->mutex);return !s->activeToken && !s->poisoned;}
std::uint64_t GRUMemoryContext::parameterFingerprint() const {const auto s=runtime_;const std::lock_guard lock(s->mutex);return s->parameterFingerprint;}
namespace {
GRUMemorySnapshot snapshotLocked(const std::shared_ptr<Runtime>& s) {
    require(!s->activeToken,"GRU snapshot requires idle owner");
    const auto buffers=stateBuffers(*s);std::size_t total=0;for (auto b:buffers) total+=b.length;
    auto staging=[s->device newBufferWithLength:total options:MTLResourceStorageModeShared];auto command=[s->queue commandBuffer];auto blit=[command blitCommandEncoder];
    require(staging && command && blit,"GRU snapshot allocation failed");std::size_t offset=0;
    for (auto b:buffers) {[blit copyFromBuffer:b sourceOffset:0 toBuffer:staging destinationOffset:offset size:b.length];offset+=b.length;}[blit endEncoding];wait(command);
    GRUMemorySnapshot result;result.metadata.resize(s->config.environments);result.before.resize(std::size_t(s->config.environments)*s->hidden);result.proposed.resize(result.before.size());result.errors.resize(s->config.environments+1);
    offset=0;for (auto destination : {std::as_writable_bytes(std::span(result.metadata)),std::as_writable_bytes(std::span(result.before)),std::as_writable_bytes(std::span(result.proposed)),std::as_writable_bytes(std::span(result.errors))}) {
        std::memcpy(destination.data(),static_cast<const char*>(staging.contents)+offset,destination.size());offset+=destination.size();}
    return result;
}
} // namespace
GRUMemorySnapshot GRUMemoryContext::snapshot() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);return snapshotLocked(s);
}
std::vector<std::byte> GRUMemoryContext::checkpointState() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);const auto state=snapshotLocked(s);
    require(!s->activeToken && !s->poisoned && std::all_of(state.errors.begin(),state.errors.end(),[](auto x){return x==0;}),"GRU checkpoint requires healthy quiescent owner");
    checkpoint::Bytes bytes;checkpoint::append(bytes,identity(*s));
    for (auto data : {std::as_bytes(std::span(state.metadata)),std::as_bytes(std::span(state.before)),std::as_bytes(std::span(state.proposed)),std::as_bytes(std::span(state.errors))}) checkpoint::appendBlob(bytes,data);
    return checkpoint::seal({'N','U','M','I','G','M','0','1'},MR_GRU_MEMORY_ABI,bytes);
}
void GRUMemoryContext::restoreState(std::span<const std::byte> bytes) {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);require(!s->activeToken && !s->poisoned,"GRU restore requires healthy idle owner");
    checkpoint::Reader reader{checkpoint::open(bytes,{'N','U','M','I','G','M','0','1'},MR_GRU_MEMORY_ABI)};
    require(reader.get<checkpoint::Digest>()==identity(*s),"GRU checkpoint execution/parameter identity mismatch");
    const auto old=stateBuffers(*s);std::array<std::span<const std::byte>,4> payload;
    for (unsigned i=0;i<4;++i) {payload[i]=reader.blob();require(payload[i].size()==old[i].length,"GRU checkpoint state dimensions mismatch");}reader.end();
    for (unsigned n=0;n<s->config.environments;++n) {mr_uint4 metadata;std::memcpy(&metadata,payload[0].data()+n*sizeof(metadata),sizeof(metadata));
        require(metadata.z<=1 && !metadata.w && (metadata.z || (!metadata.x && !metadata.y)),"Invalid GRU checkpoint clock metadata");
        for (unsigned h=0;h<s->hidden;++h) for (unsigned b=1;b<=2;++b) {float value;std::memcpy(&value,payload[b].data()+(std::size_t(n)*s->hidden+h)*sizeof(float),sizeof(value));
            require(std::isfinite(value) && std::abs(value)<=1.000001f && (metadata.z || value==0),"Invalid GRU checkpoint hidden value");}}
    for (std::size_t i=0;i<payload[3].size();i+=sizeof(unsigned)) {unsigned error;std::memcpy(&error,payload[3].data()+i,sizeof(error));require(!error,"GRU checkpoint contains sticky errors");}
    std::array<id<MTLBuffer>,4> restored;auto command=[s->queue commandBuffer];auto blit=[command blitCommandEncoder];require(command && blit,"GRU restore encoder failed");
    for (unsigned i=0;i<4;++i) {restored[i]=i==3 ? [s->device newBufferWithLength:payload[i].size() options:MTLResourceStorageModeShared] : privateBuffer(*s,payload[i].size());
        auto upload=[s->device newBufferWithBytes:payload[i].data() length:payload[i].size() options:MTLResourceStorageModeShared];require(restored[i] && upload,"GRU restore allocation failed");
        [blit copyFromBuffer:upload sourceOffset:0 toBuffer:restored[i] destinationOffset:0 size:payload[i].size()];}
    [blit endEncoding];wait(command);s->metadata=restored[0];s->before=restored[1];s->proposed=restored[2];s->errors=restored[3];
}
}
