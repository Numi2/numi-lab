#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/RetinalStream.hpp"
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
void require(bool yes,const char* reason) {if (!yes) throw std::runtime_error(reason);}
std::filesystem::path defaultLibrary() {
    static const char anchor=0;Dl_info image{};
    if (dladdr(&anchor,&image)!=0 && image.dli_fname) {
        const auto directory=std::filesystem::path(image.dli_fname).parent_path();
        for (const auto& p : {directory/"metalrobo/RetinalStream.metallib",directory.parent_path()/"shaders/RetinalStream.metallib"})
            if (std::filesystem::is_regular_file(p)) return p;
    }
    return {};
}
}
namespace detail {
struct RetinalStreamRuntime : std::enable_shared_from_this<RetinalStreamRuntime> {
    RetinalStreamConfig config;
    checkpoint::Digest checkpointIdentity{};
    __strong id<MTLDevice> device=nil;
    __strong id<MTLComputePipelineState> prepare=nil,sample=nil,publish=nil;
    __strong id<MTLBuffer> states=nil,pending=nil,pixels=nil,visible=nil,errors=nil;
    mutable std::mutex mutex;
    std::uintptr_t activeToken=0;
    bool poisoned=false;
};
}
RetinalStreamContext::RetinalStreamContext(RetinalStreamConfig config) {
    const auto& c=config;
    if (!c.environments || c.environments>65536 || !c.width || !c.height || c.width>4096 || c.height>4096 ||
        c.sourceWidth<c.width || c.sourceHeight<c.height || c.sourceWidth>4096 || c.sourceHeight>4096 ||
        c.sourceWidth%c.width || c.sourceHeight%c.height || !c.eyes || c.eyes>4 ||
        !c.sampleNumerator || !c.sampleDenominator || c.sampleNumerator>c.sampleDenominator ||
        c.latencyTicks>16 || !std::isfinite(c.tickSeconds) || !(c.tickSeconds>0))
        throw std::invalid_argument("Invalid retinal dimensions, rational cadence or latency");
    const std::uint64_t pixels=std::uint64_t(c.environments)*c.width*c.height*c.eyes;
    const std::uint64_t ring=c.latencyTicks+1;
    const std::uint64_t retained=pixels*(ring+1)*sizeof(mr_float4)+
        std::uint64_t(c.environments)*(ring*sizeof(mr_uint4)+sizeof(MRRetinalStateGPU)+sizeof(unsigned));
    if (std::uint64_t(c.environments)*c.sourceWidth*c.sourceHeight>std::numeric_limits<unsigned>::max() ||
        pixels*ring>std::numeric_limits<unsigned>::max() ||
        std::uint64_t(c.environments)*(std::uint64_t(c.width)*c.height*c.eyes*3+2)>std::numeric_limits<unsigned>::max() ||
        retained>c.maximumRetainedBytes) throw std::invalid_argument("Retinal storage/index admission exceeded");
    auto s=std::make_shared<detail::RetinalStreamRuntime>();s->config=std::move(config);
    s->device=MTLCreateSystemDefaultDevice();require(s->device!=nil,"Retinal stream requires Metal");
    const auto path=s->config.metallibPath.empty() ? defaultLibrary() : s->config.metallibPath;
    checkpoint::Bytes identity;
    for (const auto value : {c.environments,c.sourceWidth,c.sourceHeight,c.width,c.height,c.eyes,
                            c.sampleNumerator,c.sampleDenominator,c.latencyTicks}) checkpoint::append(identity,value);
    checkpoint::append(identity,c.tickSeconds);checkpoint::append(identity,std::uint32_t(MR_RETINAL_STREAM_ABI));
    checkpoint::append(identity,checkpoint::fileDigest(path));
    s->checkpointIdentity=checkpoint::digest(identity);
    NSError* error=nil;
    id<MTLLibrary> library=[s->device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&error];
    require(library!=nil,"Retinal metallib could not be loaded");
    const auto pipeline=[&](NSString* name) {
        id<MTLFunction> function=[library newFunctionWithName:name];
        require(function!=nil,"Retinal kernel missing");
        id<MTLComputePipelineState> result=[s->device newComputePipelineStateWithFunction:function error:&error];
        require(result!=nil,"Retinal pipeline could not be compiled");return result;
    };
    s->prepare=pipeline(@"mr_retinal_prepare");s->sample=pipeline(@"mr_retinal_sample");s->publish=pipeline(@"mr_retinal_publish");
    const auto allocate=[&](std::size_t bytes) {id<MTLBuffer> b=[s->device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];require(b!=nil,"Retinal storage allocation failed");return b;};
    s->states=allocate(c.environments*sizeof(MRRetinalStateGPU));
    s->pending=allocate(c.environments*ring*sizeof(mr_uint4));
    s->pixels=allocate(pixels*ring*sizeof(mr_float4));s->visible=allocate(pixels*sizeof(mr_float4));
    s->errors=allocate(c.environments*sizeof(unsigned));
    id<MTLCommandQueue> queue=[s->device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    require(queue!=nil && command!=nil && blit!=nil,"Retinal initialization failed");
    for (id<MTLBuffer> b : {s->states,s->pending,s->pixels,s->visible,s->errors}) [blit fillBuffer:b range:NSMakeRange(0,b.length) value:0];
    [blit endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Retinal initialization GPU failure");runtime_=std::move(s);
}
RetinalStreamContext::~RetinalStreamContext()=default;
std::uint32_t RetinalStreamContext::featureCount() const noexcept {
    const auto& c=runtime_->config;return c.eyes*c.width*c.height*3+2;
}
bool RetinalStreamContext::encode(void* commandBuffer,void* taskStates,std::span<void* const> eyeRGB,
    void* actorOutput,unsigned actorStride,unsigned featureOffset,std::size_t outputBaseElements) {
    auto s=runtime_;const auto& c=s->config;
    if (!commandBuffer || !taskStates || !actorOutput || eyeRGB.size()!=c.eyes ||
        std::uint64_t(featureOffset)+featureCount()>actorStride) return false;
    id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)commandBuffer;
    id<MTLBuffer> tasks=(__bridge id<MTLBuffer>)taskStates,output=(__bridge id<MTLBuffer>)actorOutput;
    const std::uint64_t extra=std::uint64_t(c.environments-1)*actorStride+featureOffset+featureCount();
    if (command.status>MTLCommandBufferStatusEnqueued || outputBaseElements>std::numeric_limits<std::size_t>::max()/sizeof(float)-extra ||
        command.device.registryID!=s->device.registryID || tasks.device.registryID!=s->device.registryID ||
        output.device.registryID!=s->device.registryID || tasks.length<std::size_t(c.environments)*sizeof(MRTaskStateGPU) ||
        output.length<(outputBaseElements+extra)*sizeof(float)) return false;
    for (void* raw : eyeRGB) {
        if (!raw) return false;
        id<MTLBuffer> image=(__bridge id<MTLBuffer>)raw;
        if (image.device.registryID!=s->device.registryID ||
            image.length<std::size_t(c.environments)*c.sourceWidth*c.sourceHeight*sizeof(mr_float4)) return false;
    }
    // Exclusively own this stream until the caller's command completes. Never
    // retain the borrowed command or buffers. An abandoned command keeps the
    // context unavailable rather than admitting unordered state mutation.
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];if (encoder==nil) return false;
    const auto token=reinterpret_cast<std::uintptr_t>(commandBuffer);
    {
        const std::lock_guard lock(s->mutex);
        if (s->poisoned || (s->activeToken && s->activeToken!=token)) {[encoder endEncoding];return false;}
        if (!s->activeToken) {
            s->activeToken=token;
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                const std::lock_guard done(s->mutex);
                if (completed.status!=MTLCommandBufferStatusCompleted) s->poisoned=true;
                if (s->activeToken==token) s->activeToken=0;
            }];
        }
    }
    MRRetinalDispatchGPU p{{c.sourceWidth,c.sourceHeight,c.width,c.height},
        {c.environments,c.eyes,featureOffset,actorStride},{c.sampleNumerator,c.sampleDenominator,c.latencyTicks,MR_RETINAL_STREAM_ABI},
        {static_cast<unsigned>(outputBaseElements),static_cast<unsigned>(outputBaseElements>>32),0,0},{c.tickSeconds,0,0,0}};
    const auto dispatch=[&](id<MTLComputePipelineState> pipeline,unsigned count) {
        [encoder setComputePipelineState:pipeline];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<unsigned>(128,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    encoder.label=@"native retinal timing and radiance";
    [encoder setBuffer:tasks offset:0 atIndex:0];[encoder setBuffer:s->states offset:0 atIndex:1];
    [encoder setBuffer:s->pending offset:0 atIndex:2];[encoder setBuffer:s->errors offset:0 atIndex:3];
    [encoder setBytes:&p length:sizeof(p) atIndex:4];dispatch(s->prepare,c.environments);
    for (unsigned eye=0;eye<c.eyes;++eye) {
        p.output.z=eye;
        [encoder setBuffer:(__bridge id<MTLBuffer>)eyeRGB[eye] offset:0 atIndex:0];
        [encoder setBuffer:s->states offset:0 atIndex:1];[encoder setBuffer:s->pixels offset:0 atIndex:2];
        [encoder setBuffer:s->visible offset:0 atIndex:3];[encoder setBuffer:s->errors offset:0 atIndex:4];
        [encoder setBytes:&p length:sizeof(p) atIndex:5];dispatch(s->sample,c.environments*c.width*c.height);
    }
    [encoder setBuffer:s->states offset:0 atIndex:0];[encoder setBuffer:s->visible offset:0 atIndex:1];
    [encoder setBuffer:s->errors offset:0 atIndex:2];[encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&p length:sizeof(p) atIndex:4];dispatch(s->publish,c.environments*featureCount());
    [encoder endEncoding];return true;
}
RetinalStreamSnapshot RetinalStreamContext::snapshot() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal publication requires a quiescent successful command");
    const auto bytes=s->states.length+s->errors.length;
    id<MTLBuffer> staging=[s->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[s->device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    require(staging!=nil && queue!=nil && command!=nil && blit!=nil,"Retinal publication allocation failed");
    [blit copyFromBuffer:s->states sourceOffset:0 toBuffer:staging destinationOffset:0 size:s->states.length];
    [blit copyFromBuffer:s->errors sourceOffset:0 toBuffer:staging destinationOffset:s->states.length size:s->errors.length];
    [blit endEncoding];[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"Retinal publication GPU failure");
    RetinalStreamSnapshot result;result.states.resize(s->config.environments);result.errors.resize(s->config.environments);
    std::memcpy(result.states.data(),staging.contents,s->states.length);
    std::memcpy(result.errors.data(),static_cast<const char*>(staging.contents)+s->states.length,s->errors.length);return result;
}

std::vector<std::byte> RetinalStreamContext::checkpointState() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal checkpoint requires quiescent successful work");
    const std::array<id<MTLBuffer>,5> buffers{s->states,s->pending,s->pixels,s->visible,s->errors};
    std::size_t total=0;for (auto b:buffers) total+=b.length;
    id<MTLBuffer> staging=[s->device newBufferWithLength:total options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[s->device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];require(staging && queue && command && blit,"Retinal checkpoint allocation failed");
    std::size_t offset=0;for (auto b:buffers) {[blit copyFromBuffer:b sourceOffset:0 toBuffer:staging destinationOffset:offset size:b.length];offset+=b.length;}
    [blit endEncoding];[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"Retinal checkpoint GPU readback failed");
    const auto* data=static_cast<const std::byte*>(staging.contents);
    const auto* errors=reinterpret_cast<const unsigned*>(data+total-s->errors.length);
    for (unsigned env=0;env<s->config.environments;++env) require(errors[env]==0,"Cannot checkpoint a rejected retinal stream");
    checkpoint::Bytes bytes;checkpoint::append(bytes,s->checkpointIdentity);offset=0;
    for (auto b:buffers) {checkpoint::appendBlob(bytes,{data+offset,b.length});offset+=b.length;}
    return checkpoint::seal({'N','U','M','I','R','S','0','1'},MR_RETINAL_STREAM_ABI,bytes);
}
void RetinalStreamContext::restoreState(std::span<const std::byte> bytes) {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal restore requires quiescent successful work");
    checkpoint::Reader reader{checkpoint::open(bytes,{'N','U','M','I','R','S','0','1'},MR_RETINAL_STREAM_ABI)};
    require(reader.get<checkpoint::Digest>()==s->checkpointIdentity,"Retinal checkpoint configuration/shader mismatch");
    const std::array<id<MTLBuffer>,5> buffers{s->states,s->pending,s->pixels,s->visible,s->errors};
    std::array<std::span<const std::byte>,5> sections;
    for (unsigned i=0;i<buffers.size();++i) {sections[i]=reader.blob();require(sections[i].size()==buffers[i].length,"Retinal checkpoint layout mismatch");}reader.end();
    for (unsigned env=0;env<s->config.environments;++env) {unsigned error=0;std::memcpy(&error,sections[4].data()+env*sizeof(unsigned),sizeof(error));require(!error,"Retinal checkpoint contains rejected state");}
    // Validate everything before the first mutation; no per-step capture/readback.
    id<MTLCommandQueue> queue=[s->device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];require(queue && command && blit,"Retinal restore encoder failed");
    for (unsigned i=0;i<buffers.size();++i) {
        id<MTLBuffer> upload=[s->device newBufferWithBytes:sections[i].data() length:sections[i].size() options:MTLResourceStorageModeShared];
        require(upload!=nil,"Retinal restore staging failed");[blit copyFromBuffer:upload sourceOffset:0 toBuffer:buffers[i] destinationOffset:0 size:sections[i].size()];
    }
    [blit endEncoding];[command commit];[command waitUntilCompleted];
    if (command.status!=MTLCommandBufferStatusCompleted) {s->poisoned=true;throw std::runtime_error("Retinal restore GPU failure");}
}
}
