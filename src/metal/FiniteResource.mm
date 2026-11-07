#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include "metalrobo/task_program_types.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {
void require(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest(std::span<const std::byte> bytes) {
    require(bytes.size()<=std::numeric_limits<CC_LONG>::max(),"Resource digest payload too large");
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> result{};
    CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),result.data());
    return result;
}
template<class T> void append(std::vector<std::byte>& bytes,const T& value) {
    const auto* begin=reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(),begin,begin+sizeof(T));
}
bool finite(mr_float4 value) {
    return std::isfinite(value.x)&&std::isfinite(value.y)&&std::isfinite(value.z)&&std::isfinite(value.w);
}
std::filesystem::path resourceMetallib() {
    static const char imageAnchor=0;
    Dl_info image{};
    if (dladdr(&imageAnchor,&image)!=0 && image.dli_fname) {
        const auto directory=std::filesystem::path(image.dli_fname).parent_path();
        for (const auto& candidate : {directory/"metalrobo/MetalRobo.metallib",
                                      directory.parent_path()/"shaders/MetalRobo.metallib"})
            if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return METALROBO_DEFAULT_METALLIB;
}
struct CheckpointHeader {
    std::array<char,8> magic{'N','U','M','I','F','R','0','1'};
    std::uint32_t abi=MR_FINITE_RESOURCE_ABI, environments=0;
    std::uint64_t fingerprint=0, payloadBytes=0;
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> sha{};
};
static_assert(sizeof(CheckpointHeader)==64);
static_assert(std::endian::native==std::endian::little);
}

std::uint64_t finiteResourceProgramFingerprint(const MRFiniteResourceProgramGPU& program,
    std::span<const MRFiniteResourcePatchGPU> patches,bool publishTaskState,
    const MRFiniteResourceAllocationGPU& allocation) {
    std::vector<std::byte> bytes;
    append(bytes,std::uint32_t(MR_FINITE_RESOURCE_ABI));append(bytes,program);
    append(bytes,std::uint32_t(publishTaskState));
    for (const auto& patch:patches) append(bytes,patch);
    if (allocation.version) append(bytes,allocation);
    const auto hash=digest(bytes);std::uint64_t result=0;
    std::memcpy(&result,hash.data(),sizeof(result));return result;
}
// Preserve the existing exported C++ symbol and its exact legacy hash bytes.
std::uint64_t finiteResourceProgramFingerprint(const MRFiniteResourceProgramGPU& program,
    std::span<const MRFiniteResourcePatchGPU> patches,bool publishTaskState) {
    return finiteResourceProgramFingerprint(program,patches,publishTaskState,{});
}
namespace detail {
struct FiniteResourceRuntime : std::enable_shared_from_this<FiniteResourceRuntime> {
    FiniteResourceConfig config;
    std::uint64_t fingerprint=0;
    std::uint64_t programFingerprint=0;
    mutable std::mutex mutex;
    std::uintptr_t activeToken=0;
    bool poisoned=false;
    __strong id<MTLDevice> device=nil;
    __strong id<MTLComputePipelineState> pipeline=nil;
    __strong id<MTLBuffer> program=nil, patches=nil, state=nil;
    __strong id<MTLBuffer> unusedTask=nil;
    __strong id<MTLBuffer> noResets=nil;

    static bool encode(void* opaque,const MetalWorldDevicePhysicsPass& pass) {
        auto& runtime=*static_cast<FiniteResourceRuntime*>(opaque);
        if (pass.worldFingerprint!=runtime.config.worldFingerprint ||
            pass.environmentCount!=runtime.config.environmentCount ||
            pass.bodyCount!=runtime.config.bodyCount || pass.nq!=runtime.config.nq ||
            (runtime.config.publishTaskState && pass.taskStateProgramFingerprint!=runtime.programFingerprint)) return false;
        if (pass.phase!=MetalWorldDevicePhysicsPhase::postCommit ||
            pass.physicsSubstep+1u!=pass.physicsSubsteps) return true;
        if (!pass.commandBuffer || !pass.currentBodies || !pass.q || !pass.environmentStatuses ||
            pass.bodyStateStride<pass.bodyCount || pass.qStride<pass.nq) return false;
        id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> bodies=(__bridge id<MTLBuffer>)pass.currentBodies;
        id<MTLBuffer> q=(__bridge id<MTLBuffer>)pass.q;
        id<MTLBuffer> statuses=(__bridge id<MTLBuffer>)pass.environmentStatuses;
        id<MTLBuffer> task=runtime.config.publishTaskState
            ? (__bridge id<MTLBuffer>)pass.taskStates : runtime.unusedTask;
        id<MTLBuffer> resets=pass.resetMasks ? (__bridge id<MTLBuffer>)pass.resetMasks : runtime.noResets;
        const unsigned resetStride=pass.resetMasks ? pass.resetMaskStepStride : 0u;
        if (command.device.registryID!=runtime.device.registryID ||
            task==nil || task.length<std::size_t(pass.environmentCount)*sizeof(MRTaskStateGPU) ||
            resets==nil || resets.length<(std::size_t(pass.controlStep)*resetStride+pass.environmentCount)*sizeof(unsigned) ||
            bodies.length<std::size_t(pass.environmentCount)*pass.bodyStateStride*sizeof(MRBodyStateGPU) ||
            q.length<std::size_t(pass.environmentCount)*pass.qStride*sizeof(float) ||
            statuses.length<std::size_t(pass.environmentCount)*sizeof(MRMetalWorldStatusGPU)) return false;
        const auto token=reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
        {
            std::lock_guard lock(runtime.mutex);
            if (runtime.poisoned || (runtime.activeToken!=0 && runtime.activeToken!=token)) return false;
            if (runtime.activeToken==0) {
                runtime.activeToken=token;
                const auto retained=runtime.shared_from_this();
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    std::lock_guard completedLock(retained->mutex);
                    if (completed.status==MTLCommandBufferStatusError) retained->poisoned=true;
                    if (retained->activeToken==token) retained->activeToken=0;
                }];
            }
        }
        const MRFiniteResourceDispatchGPU dispatch{
            {pass.environmentCount,pass.bodyStateStride,pass.qStride,pass.bodyCount},
            {pass.nq,MR_FINITE_RESOURCE_KERNEL_ABI,runtime.config.publishTaskState ? 1u : 0u,0u},
            {resetStride,pass.controlStep,0u,0u}};
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (encoder==nil) return false;
        [encoder setComputePipelineState:runtime.pipeline];
        [encoder setBuffer:runtime.program offset:0 atIndex:0];
        [encoder setBuffer:runtime.patches offset:0 atIndex:1];
        [encoder setBuffer:runtime.state offset:0 atIndex:2];
        [encoder setBuffer:bodies offset:0 atIndex:3];
        [encoder setBuffer:q offset:0 atIndex:4];
        [encoder setBuffer:statuses offset:0 atIndex:5];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:6];
        [encoder setBuffer:task offset:0 atIndex:7];
        [encoder setBuffer:resets offset:0 atIndex:8];
        [encoder setBytes:&runtime.config.allocation length:sizeof(MRFiniteResourceAllocationGPU) atIndex:9];
        [encoder setBytes:&runtime.config.allocationSeed length:sizeof(std::uint64_t) atIndex:10];
        const NSUInteger width=std::min<NSUInteger>(32,runtime.pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(pass.environmentCount,1,1)
             threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [encoder endEncoding];
        return true;
    }
    static void abort(void* opaque,void* command) {
        auto& runtime=*static_cast<FiniteResourceRuntime*>(opaque);
        std::lock_guard lock(runtime.mutex);
        if (runtime.activeToken==reinterpret_cast<std::uintptr_t>(command)) runtime.activeToken=0;
    }
};
}

FiniteResourceContext::FiniteResourceContext(FiniteResourceConfig config)
    : runtime_(std::make_shared<detail::FiniteResourceRuntime>()) {
    const auto& p=config.program;
    const auto& allocation=config.allocation;
    require((allocation.version==0u && allocation.activePatchCount==0u &&
             allocation.availableAfterTicks==0u && allocation.concealProximity==0u && config.allocationSeed==0u) ||
            (allocation.version==1u && allocation.activePatchCount<=p.binding.z && allocation.concealProximity<=1u),
            "Invalid finite-resource allocation program");
    require(config.worldFingerprint!=0 && config.environmentCount>0 && config.environmentCount<=1048576u &&
            config.bodyCount>0 && config.nq>0 && p.binding.x<config.bodyCount && p.binding.y<config.nq &&
            p.binding.z>0 && p.binding.z<=MR_FINITE_RESOURCE_MAX_PATCHES &&
            config.patches.size()==p.binding.z && p.binding.w>0 && finite(p.mouthClosed) && finite(p.opening) &&
            p.mouthClosed.w>=0 && p.opening.x>p.mouthClosed.w &&
            p.opening.y==0 && p.opening.z==0 && p.opening.w==0,
            "Invalid finite-resource program");
    std::uint64_t total=0;
    for (const auto& patch:config.patches) {
        require(patch.binding.x<config.bodyCount && patch.binding.y>0 && patch.binding.z>0 &&
                patch.binding.w==0 && finite(patch.centerRadius) && patch.centerRadius.w>0,
                "Invalid finite-resource patch");
        total+=patch.binding.y;
    }
    require(total<=std::numeric_limits<std::uint32_t>::max(),"Resource accounting would overflow");
    std::vector<std::byte> identity;
    runtime_->programFingerprint=finiteResourceProgramFingerprint(p,config.patches,config.publishTaskState,config.allocation);
    append(identity,std::uint32_t(MR_FINITE_RESOURCE_ABI));
    append(identity,config.worldFingerprint); append(identity,config.environmentCount);
    append(identity,config.bodyCount); append(identity,config.nq); append(identity,p);
    append(identity,std::uint32_t(config.publishTaskState));
    for (const auto& patch:config.patches) append(identity,patch);
    if (allocation.version) { append(identity,allocation);append(identity,config.allocationSeed); }
    const auto hashed=digest(identity);
    std::memcpy(&runtime_->fingerprint,hashed.data(),sizeof(runtime_->fingerprint));
    require(runtime_->fingerprint!=0,"Invalid resource identity");
    runtime_->config=std::move(config);
    runtime_->device=MTLCreateSystemDefaultDevice();
    require(runtime_->device!=nil,"Metal device unavailable");
    NSError* error=nil;
    const auto libraryPath=resourceMetallib();
    id<MTLLibrary> library=[runtime_->device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath.c_str()]] error:&error];
    if (library==nil) throw std::runtime_error("Resource metallib load failed: "+std::string(error.localizedDescription.UTF8String));
    // Fail before allocation/encoding when the dylib and shader argument layouts
    // disagree; an older unversioned kernel would silently ignore bindings9/10.
    id<MTLFunction> function=[library newFunctionWithName:@"mr_finite_resource_commit_v4"];
    require(function!=nil,"Finite-resource kernel ABI 4 unavailable: incompatible MetalRobo.metallib");
    runtime_->pipeline=[runtime_->device newComputePipelineStateWithFunction:function error:&error];
    require(runtime_->pipeline!=nil,"Resource pipeline creation failed");
    runtime_->program=[runtime_->device newBufferWithBytes:&runtime_->config.program
        length:sizeof(MRFiniteResourceProgramGPU) options:MTLResourceStorageModeShared];
    runtime_->patches=[runtime_->device newBufferWithBytes:runtime_->config.patches.data()
        length:runtime_->config.patches.size()*sizeof(MRFiniteResourcePatchGPU) options:MTLResourceStorageModeShared];
    runtime_->state=[runtime_->device newBufferWithLength:
        std::size_t(runtime_->config.environmentCount)*sizeof(MRFiniteResourceStateGPU) options:MTLResourceStorageModeShared];
    require(runtime_->program!=nil && runtime_->patches!=nil && runtime_->state!=nil,"Resource buffer allocation failed");
    runtime_->noResets=[runtime_->device newBufferWithLength:
        std::size_t(runtime_->config.environmentCount)*sizeof(unsigned) options:MTLResourceStorageModeShared];
    require(runtime_->noResets!=nil,"Resource reset buffer allocation failed");
    std::memset(runtime_->noResets.contents,0,runtime_->noResets.length);
    if (!runtime_->config.publishTaskState)
        runtime_->unusedTask=[runtime_->device newBufferWithLength:
            std::size_t(runtime_->config.environmentCount)*sizeof(MRTaskStateGPU) options:MTLResourceStorageModeShared];
    std::memset(runtime_->state.contents,0,runtime_->state.length);
}
FiniteResourceContext::~FiniteResourceContext()=default;
std::uint64_t FiniteResourceContext::fingerprint() const noexcept { return runtime_->fingerprint; }
MetalWorldDevicePhysicsProgram FiniteResourceContext::deviceProgram() {
    return {.context=runtime_.get(),.encode=&detail::FiniteResourceRuntime::encode,
        .abort=&detail::FiniteResourceRuntime::abort,.fingerprint=runtime_->fingerprint};
}
std::vector<MRFiniteResourceStateGPU> FiniteResourceContext::snapshot() const {
    std::lock_guard lock(runtime_->mutex);
    require(runtime_->activeToken==0 && !runtime_->poisoned,"Resource state is in flight or device-poisoned");
    std::vector<MRFiniteResourceStateGPU> result(runtime_->config.environmentCount);
    std::memcpy(result.data(),runtime_->state.contents,runtime_->state.length);
    return result;
}
void FiniteResourceContext::saveCheckpoint(const std::filesystem::path& path) const {
    const auto states=snapshot();
    CheckpointHeader header;
    header.environments=runtime_->config.environmentCount; header.fingerprint=runtime_->fingerprint;
    header.payloadBytes=states.size()*sizeof(MRFiniteResourceStateGPU);
    header.sha=digest(std::as_bytes(std::span(states)));
    // Exclusive creation preserves earlier evidence; failed writes are retained.
    const int descriptor=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    require(descriptor>=0,"Cannot exclusively create resource checkpoint");
    const auto writeAll=[descriptor](const void* data,std::size_t count) {
        const auto* bytes=static_cast<const char*>(data);
        while (count) {
            const auto written=write(descriptor,bytes,count);
            if (written<=0) return false;
            bytes+=written; count-=std::size_t(written);
        }
        return true;
    };
    const bool written=writeAll(&header,sizeof(header)) && writeAll(states.data(),header.payloadBytes);
    const bool synced=written && fsync(descriptor)==0;
    const int closed=close(descriptor);
    require(synced && closed==0,"Resource checkpoint write failed");
}
void FiniteResourceContext::restoreCheckpoint(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    CheckpointHeader header, expected;
    require(bool(input.read(reinterpret_cast<char*>(&header),sizeof(header))),"Truncated resource checkpoint header");
    require(header.magic==expected.magic && header.abi==MR_FINITE_RESOURCE_ABI &&
            header.environments==runtime_->config.environmentCount && header.fingerprint==runtime_->fingerprint &&
            header.payloadBytes==std::size_t(header.environments)*sizeof(MRFiniteResourceStateGPU),
            "Resource checkpoint contract mismatch");
    std::vector<MRFiniteResourceStateGPU> states(header.environments);
    require(bool(input.read(reinterpret_cast<char*>(states.data()),header.payloadBytes)),"Truncated resource checkpoint state");
    require(input.peek()==std::char_traits<char>::eof() && header.sha==digest(std::as_bytes(std::span(states))),
            "Resource checkpoint checksum or length mismatch");
    restoreValidatedStates(states);
}
std::vector<std::byte> FiniteResourceContext::checkpointState() const {
    const auto states=snapshot();return checkpoint::seal({'N','U','M','I','F','B','0','1'},runtime_->fingerprint,std::as_bytes(std::span(states)));
}
void FiniteResourceContext::restoreState(std::span<const std::byte> bytes) {
    const auto payload=checkpoint::open(bytes,{'N','U','M','I','F','B','0','1'},runtime_->fingerprint);
    require(payload.size()==std::size_t(runtime_->config.environmentCount)*sizeof(MRFiniteResourceStateGPU),"Resource checkpoint lane/layout mismatch");
    std::vector<MRFiniteResourceStateGPU> states(runtime_->config.environmentCount);std::memcpy(states.data(),payload.data(),payload.size());restoreValidatedStates(states);
}
void FiniteResourceContext::restoreValidatedStates(std::span<const MRFiniteResourceStateGPU> states) {
    for (unsigned environment=0;environment<states.size();++environment) {
        const auto& state=states[environment];
        const auto& c=runtime_->config;
        const unsigned selected=c.allocation.version ? mrResourceAllocationMask(c.allocationSeed,
            environment,c.program.binding.z,c.allocation.activePatchCount) : (1u<<c.program.binding.z)-1u;
        const unsigned expectedMask=c.allocation.version ? selected : 0u;
        std::uint64_t initial=0,remaining=0;
        require(state.accounting.w<=1u && state.clock.z<=1u &&
                (state.accounting.w==0u ? state.clock.w==0u : state.clock.w==expectedMask) &&
                finite(state.jaw) && state.jaw.y==0 && state.jaw.z==0 && state.jaw.w==0,
                "Invalid resource checkpoint state or allocation mask");
        const auto* amounts=reinterpret_cast<const std::uint32_t*>(state.remaining);
        for (unsigned patch=0;patch<8;++patch) {
            const unsigned limit=patch<c.patches.size() && (selected & (1u<<patch)) ? c.patches[patch].binding.y : 0u;
            require(amounts[patch]<=limit,"Invalid resource checkpoint inventory");
            initial+=limit;remaining+=amounts[patch];
        }
        if (state.accounting.w==0u) {
            const MRFiniteResourceStateGPU zero{};
            require(std::memcmp(&state,&zero,sizeof(state))==0,"Uninitialized resource state contains history");
        } else require(remaining+state.accounting.x==initial && state.accounting.x==state.accounting.y &&
                       state.accounting.x<=c.program.binding.w && state.accounting.z<=state.accounting.y,
                       "Resource checkpoint violates conservation");
    }
    std::lock_guard lock(runtime_->mutex);
    require(runtime_->activeToken==0 && !runtime_->poisoned,"Cannot restore in-flight or device-poisoned resources");
    std::memcpy(runtime_->state.contents,states.data(),runtime_->state.length);
}
} // namespace metalrobo
