#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNeuronCulture.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifndef METALROBO_NEURON_METALLIB
#define METALROBO_NEURON_METALLIB "NumiNeuron.metallib"
#endif

namespace metalrobo {
namespace {

template <typename T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, std::span<const T> values, NSString* label) {
    const NSUInteger bytes = std::max<NSUInteger>(1u, values.size_bytes());
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) return nil;
    buffer.label = label;
    if (!values.empty()) std::memcpy(buffer.contents, values.data(), values.size_bytes());
    return buffer;
}

id<MTLBuffer> makeBytes(id<MTLDevice> device, NSUInteger bytes, NSString* label) {
    id<MTLBuffer> buffer = [device newBufferWithLength:std::max<NSUInteger>(1u, bytes)
                                              options:MTLResourceStorageModeShared];
    if (buffer) {
        buffer.label = label;
        std::memset(buffer.contents, 0, buffer.length);
    }
    return buffer;
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) return nil;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
    return error ? nil : pipeline;
}

void copyBuffer(id<MTLBlitCommandEncoder> blit, id<MTLBuffer> source, id<MTLBuffer> target) {
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:target destinationOffset:0
                    size:std::min(source.length, target.length)];
}

void dispatch(id<MTLComputeCommandEncoder> encoder,
              id<MTLComputePipelineState> pipeline,
              NSUInteger count) {
    [encoder setComputePipelineState:pipeline];
    const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256u);
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

} // namespace

struct MetalNeuronCultureTicket::State {
    id<MTLCommandBuffer> commandBuffer = nil;
    std::atomic<MetalNeuronCultureStatus> status{MetalNeuronCultureStatus::pending};
};

struct MetalNeuronCultureRuntime::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> tickPipeline = nil;
    id<MTLComputePipelineState> plasticityPipeline = nil;
    id<MTLComputePipelineState> recordPipeline = nil;
    id<MTLComputePipelineState> historyPipeline = nil;
    id<MTLComputePipelineState> growthPipeline = nil;
    CompiledNeuronCulture culture;
    id<MTLBuffer> header = nil;
    id<MTLBuffer> growth = nil;
    id<MTLBuffer> neurons = nil;
    id<MTLBuffer> synapses = nil;
    id<MTLBuffer> electrodes = nil;

    struct Buffers {
        id<MTLBuffer> membrane = nil;
        id<MTLBuffer> refractory = nil;
        id<MTLBuffer> preTrace = nil;
        id<MTLBuffer> postTrace = nil;
        id<MTLBuffer> weights = nil;
        id<MTLBuffer> depression = nil;
        id<MTLBuffer> spikes = nil;
        id<MTLBuffer> spikeHistory = nil;
        id<MTLBuffer> electrodeCounts = nil;
        id<MTLBuffer> phase = nil;
        id<MTLBuffer> tubulin = nil;
    } accepted, working;
    id<MTLBuffer> phaseScratch = nil;
    id<MTLBuffer> tubulinScratch = nil;

    std::shared_ptr<MetalNeuronCultureTicket::State> active;
    mutable std::mutex mutex;
    bool hasPrepared = false;
    enum class PreparedKind { none, ticks, growth } preparedKind = PreparedKind::none;
    std::uint64_t acceptedTick = 0u;
    std::uint64_t acceptedGrowthIteration = 0u;
    std::uint64_t preparedTick = 0u;
    std::uint64_t preparedGrowthIteration = 0u;

    bool pipelinesValid() const noexcept {
        return device && queue && library && tickPipeline && plasticityPipeline &&
            recordPipeline && historyPipeline && growthPipeline;
    }
};

MetalNeuronCultureTicket::MetalNeuronCultureTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
MetalNeuronCultureTicket::MetalNeuronCultureTicket(MetalNeuronCultureTicket&&) noexcept = default;
MetalNeuronCultureTicket& MetalNeuronCultureTicket::operator=(MetalNeuronCultureTicket&&) noexcept = default;
MetalNeuronCultureTicket::~MetalNeuronCultureTicket() = default;

bool MetalNeuronCultureTicket::valid() const noexcept { return state_ && state_->commandBuffer; }
bool MetalNeuronCultureTicket::completed() const noexcept {
    return state_ && state_->status.load(std::memory_order_acquire) != MetalNeuronCultureStatus::pending;
}
MetalNeuronCultureStatus MetalNeuronCultureTicket::wait() noexcept {
    if (!valid()) return MetalNeuronCultureStatus::invalidArgument;
    [state_->commandBuffer waitUntilCompleted];
    return state_->status.load(std::memory_order_acquire);
}

MetalNeuronCultureRuntime::MetalNeuronCultureRuntime() = default;
MetalNeuronCultureRuntime::MetalNeuronCultureRuntime(MetalNeuronCultureRuntime&&) noexcept = default;
MetalNeuronCultureRuntime& MetalNeuronCultureRuntime::operator=(MetalNeuronCultureRuntime&&) noexcept = default;
MetalNeuronCultureRuntime::~MetalNeuronCultureRuntime() = default;

MetalNeuronCultureRuntime MetalNeuronCultureRuntime::create(
    const CompiledNeuronCulture& culture,
    void* metalDevice
) {
    MetalNeuronCultureRuntime runtime;
    if (!culture.valid()) return runtime;
    id<MTLDevice> device = metalDevice ? (__bridge id<MTLDevice>)metalDevice : MTLCreateSystemDefaultDevice();
    if (!device) return runtime;
    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->queue = [device newCommandQueue];
    NSURL* url = [NSURL fileURLWithPath:@METALROBO_NEURON_METALLIB];
    NSError* error = nil;
    impl->library = [device newLibraryWithURL:url error:&error];
    if (error || !impl->library || !impl->queue) return runtime;
    impl->tickPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_tick");
    impl->plasticityPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_plasticity");
    impl->recordPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_record");
    impl->historyPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_store_history");
    impl->growthPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_growth");
    if (!impl->pipelinesValid()) return runtime;
    impl->culture = culture;
    const auto& h = culture.header();
    impl->header = makeBuffer(device, std::span(&h, 1u), @"Neuron culture header");
    const auto& g = culture.growth();
    impl->growth = makeBuffer(device, std::span(&g, 1u), @"Neuron growth parameters");
    impl->neurons = makeBuffer(device, culture.neurons(), @"Neuron culture neurons");
    impl->synapses = makeBuffer(device, culture.synapses(), @"Neuron culture incoming CSR");
    impl->electrodes = makeBuffer(device, culture.electrodes(), @"Neuron culture virtual MEA");

    NeuronCultureReference reference(culture);
    const auto& initial = reference.accepted();
    auto allocate = [&](Impl::Buffers& buffers, NSString* prefix) {
        buffers.membrane = makeBuffer(device, std::span(initial.membrane),
                                      [prefix stringByAppendingString:@" membrane"]);
        buffers.refractory = makeBuffer(device, std::span(initial.refractory),
                                        [prefix stringByAppendingString:@" refractory"]);
        buffers.preTrace = makeBuffer(device, std::span(initial.preTrace),
                                      [prefix stringByAppendingString:@" pre trace"]);
        buffers.postTrace = makeBuffer(device, std::span(initial.postTrace),
                                       [prefix stringByAppendingString:@" post trace"]);
        buffers.weights = makeBuffer(device, std::span(initial.weights),
                                     [prefix stringByAppendingString:@" weights"]);
        buffers.depression = makeBuffer(device, std::span(initial.depression),
                                        [prefix stringByAppendingString:@" depression"]);
        buffers.spikes = makeBuffer(device, std::span(initial.spikes),
                                    [prefix stringByAppendingString:@" spikes"]);
        buffers.spikeHistory = makeBuffer(device, std::span(initial.spikeHistory),
                                          [prefix stringByAppendingString:@" spike history"]);
        buffers.electrodeCounts = makeBuffer(device, std::span(initial.electrodeSpikeCounts),
                                             [prefix stringByAppendingString:@" MEA counts"]);
        buffers.phase = makeBuffer(device, std::span(initial.phase),
                                   [prefix stringByAppendingString:@" phase"]);
        buffers.tubulin = makeBuffer(device, std::span(initial.tubulin),
                                     [prefix stringByAppendingString:@" tubulin"]);
    };
    allocate(impl->accepted, @"Accepted");
    allocate(impl->working, @"Prepared");
    impl->phaseScratch = makeBytes(device, impl->working.phase.length, @"Neuron phase scratch");
    impl->tubulinScratch = makeBytes(device, impl->working.tubulin.length, @"Neuron tubulin scratch");
    if (!impl->header || !impl->growth || !impl->neurons || !impl->synapses ||
        !impl->electrodes || !impl->accepted.membrane || !impl->working.membrane ||
        !impl->phaseScratch || !impl->tubulinScratch) return runtime;
    runtime.impl_ = std::move(impl);
    return runtime;
}

bool MetalNeuronCultureRuntime::valid() const noexcept {
    return impl_ && impl_->pipelinesValid() && impl_->culture.valid();
}
std::uint64_t MetalNeuronCultureRuntime::fingerprint() const noexcept {
    return valid() ? impl_->culture.fingerprint() : 0u;
}
std::string MetalNeuronCultureRuntime::deviceName() const {
    if (!valid()) return {};
    const char* name = impl_->device.name.UTF8String;
    return name ? std::string(name) : std::string{};
}

MetalNeuronCultureTicket MetalNeuronCultureRuntime::prepareTicks(
    std::uint32_t tickCount,
    std::uint32_t stimulationElectrode,
    float stimulationCurrent
) {
    if (!valid() || tickCount == 0u || !std::isfinite(stimulationCurrent) ||
        (stimulationElectrode != std::numeric_limits<std::uint32_t>::max() &&
         stimulationElectrode >= impl_->culture.header().electrodeCount)) return {};
    std::scoped_lock lock(impl_->mutex);
    if (impl_->active && impl_->active->status.load() == MetalNeuronCultureStatus::pending) return {};
    if (impl_->hasPrepared) return {};
    id<MTLCommandBuffer> commandBuffer = [impl_->queue commandBuffer];
    if (!commandBuffer) return {};
    commandBuffer.label = @"NumiNeuron transactional spike prepare";
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    copyBuffer(blit, impl_->accepted.membrane, impl_->working.membrane);
    copyBuffer(blit, impl_->accepted.refractory, impl_->working.refractory);
    copyBuffer(blit, impl_->accepted.preTrace, impl_->working.preTrace);
    copyBuffer(blit, impl_->accepted.postTrace, impl_->working.postTrace);
    copyBuffer(blit, impl_->accepted.weights, impl_->working.weights);
    copyBuffer(blit, impl_->accepted.depression, impl_->working.depression);
    copyBuffer(blit, impl_->accepted.spikes, impl_->working.spikes);
    copyBuffer(blit, impl_->accepted.spikeHistory, impl_->working.spikeHistory);
    copyBuffer(blit, impl_->accepted.electrodeCounts, impl_->working.electrodeCounts);
    [blit endEncoding];
    const auto& h = impl_->culture.header();
    for (std::uint32_t offset = 0u; offset < tickCount; ++offset) {
        MRNeuronCultureTickGPU tick{};
        tick.tick = static_cast<std::uint32_t>((impl_->acceptedTick + offset) & 0xffffffffu);
        tick.stimulationElectrode = stimulationElectrode;
        tick.stimulationEnabled = stimulationElectrode < h.electrodeCount;
        tick.status = MR_NEURON_CULTURE_STATUS_PENDING;
        tick.stimulationCurrent = stimulationCurrent;
        tick.traceDecay = std::exp(-h.neuralTimestepSeconds / h.traceTimeConstantSeconds);
        tick.depressionRecovery = std::min(1.0f,
            h.neuralTimestepSeconds / h.depressionRecoverySeconds);
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setBuffer:impl_->header offset:0 atIndex:0];
        [encoder setBuffer:impl_->neurons offset:0 atIndex:1];
        [encoder setBuffer:impl_->synapses offset:0 atIndex:2];
        [encoder setBuffer:impl_->electrodes offset:0 atIndex:3];
        [encoder setBytes:&tick length:sizeof(tick) atIndex:4];
        [encoder setBuffer:impl_->working.membrane offset:0 atIndex:5];
        [encoder setBuffer:impl_->working.refractory offset:0 atIndex:6];
        [encoder setBuffer:impl_->working.preTrace offset:0 atIndex:7];
        [encoder setBuffer:impl_->working.postTrace offset:0 atIndex:8];
        [encoder setBuffer:impl_->working.weights offset:0 atIndex:9];
        [encoder setBuffer:impl_->working.depression offset:0 atIndex:10];
        [encoder setBuffer:impl_->working.spikeHistory offset:0 atIndex:11];
        [encoder setBuffer:impl_->working.spikes offset:0 atIndex:12];
        dispatch(encoder, impl_->tickPipeline, h.neuronCount);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setBuffer:impl_->header offset:0 atIndex:0];
        [encoder setBuffer:impl_->synapses offset:0 atIndex:1];
        [encoder setBytes:&tick length:sizeof(tick) atIndex:2];
        [encoder setBuffer:impl_->working.spikes offset:0 atIndex:3];
        [encoder setBuffer:impl_->working.preTrace offset:0 atIndex:4];
        [encoder setBuffer:impl_->working.postTrace offset:0 atIndex:5];
        [encoder setBuffer:impl_->working.weights offset:0 atIndex:6];
        [encoder setBuffer:impl_->working.depression offset:0 atIndex:7];
        dispatch(encoder, impl_->plasticityPipeline, h.synapseCount);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setBuffer:impl_->header offset:0 atIndex:0];
        [encoder setBuffer:impl_->neurons offset:0 atIndex:1];
        [encoder setBuffer:impl_->electrodes offset:0 atIndex:2];
        [encoder setBuffer:impl_->working.spikes offset:0 atIndex:3];
        [encoder setBuffer:impl_->working.electrodeCounts offset:0 atIndex:4];
        dispatch(encoder, impl_->recordPipeline, h.electrodeCount);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setBuffer:impl_->header offset:0 atIndex:0];
        [encoder setBytes:&tick length:sizeof(tick) atIndex:1];
        [encoder setBuffer:impl_->working.spikes offset:0 atIndex:2];
        [encoder setBuffer:impl_->working.spikeHistory offset:0 atIndex:3];
        dispatch(encoder, impl_->historyPipeline, h.neuronCount);
        [encoder endEncoding];
    }
    auto state = std::make_shared<MetalNeuronCultureTicket::State>();
    state->commandBuffer = commandBuffer;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        state->status.store(completed.status == MTLCommandBufferStatusCompleted ?
            MetalNeuronCultureStatus::success : MetalNeuronCultureStatus::commandFailure,
            std::memory_order_release);
    }];
    impl_->active = state;
    impl_->hasPrepared = true;
    impl_->preparedKind = Impl::PreparedKind::ticks;
    impl_->preparedTick = impl_->acceptedTick + tickCount;
    impl_->preparedGrowthIteration = impl_->acceptedGrowthIteration;
    [commandBuffer commit];
    return MetalNeuronCultureTicket(state);
}

MetalNeuronCultureTicket MetalNeuronCultureRuntime::prepareGrowth(std::uint32_t iterationCount) {
    if (!valid() || iterationCount == 0u) return {};
    std::scoped_lock lock(impl_->mutex);
    if (impl_->active && impl_->active->status.load() == MetalNeuronCultureStatus::pending) return {};
    if (impl_->hasPrepared) return {};
    id<MTLCommandBuffer> commandBuffer = [impl_->queue commandBuffer];
    if (!commandBuffer) return {};
    commandBuffer.label = @"NumiNeuron transactional growth prepare";
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    copyBuffer(blit, impl_->accepted.phase, impl_->working.phase);
    copyBuffer(blit, impl_->accepted.tubulin, impl_->working.tubulin);
    [blit endEncoding];
    id<MTLBuffer> phaseSource = impl_->working.phase;
    id<MTLBuffer> tubulinSource = impl_->working.tubulin;
    id<MTLBuffer> phaseTarget = impl_->phaseScratch;
    id<MTLBuffer> tubulinTarget = impl_->tubulinScratch;
    const NSUInteger cells = static_cast<NSUInteger>(impl_->culture.header().growthWidth) *
        impl_->culture.header().growthHeight;
    for (std::uint32_t iteration = 0u; iteration < iterationCount; ++iteration) {
        MRNeuronCultureGrowthGPU parameters = impl_->culture.growth();
        parameters.iteration = static_cast<std::uint32_t>(
            (impl_->acceptedGrowthIteration + iteration) & 0xffffffffu);
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder setBuffer:phaseSource offset:0 atIndex:1];
        [encoder setBuffer:tubulinSource offset:0 atIndex:2];
        [encoder setBuffer:phaseTarget offset:0 atIndex:3];
        [encoder setBuffer:tubulinTarget offset:0 atIndex:4];
        dispatch(encoder, impl_->growthPipeline, cells);
        [encoder endEncoding];
        std::swap(phaseSource, phaseTarget);
        std::swap(tubulinSource, tubulinTarget);
    }
    if (phaseSource != impl_->working.phase) {
        id<MTLBlitCommandEncoder> finalBlit = [commandBuffer blitCommandEncoder];
        copyBuffer(finalBlit, phaseSource, impl_->working.phase);
        copyBuffer(finalBlit, tubulinSource, impl_->working.tubulin);
        [finalBlit endEncoding];
    }
    auto state = std::make_shared<MetalNeuronCultureTicket::State>();
    state->commandBuffer = commandBuffer;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        state->status.store(completed.status == MTLCommandBufferStatusCompleted ?
            MetalNeuronCultureStatus::success : MetalNeuronCultureStatus::commandFailure,
            std::memory_order_release);
    }];
    impl_->active = state;
    impl_->hasPrepared = true;
    impl_->preparedKind = Impl::PreparedKind::growth;
    impl_->preparedTick = impl_->acceptedTick;
    impl_->preparedGrowthIteration = impl_->acceptedGrowthIteration + iterationCount;
    [commandBuffer commit];
    return MetalNeuronCultureTicket(state);
}

MetalNeuronCultureStatus MetalNeuronCultureRuntime::publishPrepared() noexcept {
    if (!valid()) return MetalNeuronCultureStatus::invalidArgument;
    std::scoped_lock lock(impl_->mutex);
    if (!impl_->active || impl_->active->status.load() == MetalNeuronCultureStatus::pending)
        return MetalNeuronCultureStatus::busy;
    if (impl_->active->status.load() != MetalNeuronCultureStatus::success)
        return impl_->active->status.load();
    id<MTLCommandBuffer> commandBuffer = [impl_->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (impl_->preparedKind == Impl::PreparedKind::ticks) {
        copyBuffer(blit, impl_->working.membrane, impl_->accepted.membrane);
        copyBuffer(blit, impl_->working.refractory, impl_->accepted.refractory);
        copyBuffer(blit, impl_->working.preTrace, impl_->accepted.preTrace);
        copyBuffer(blit, impl_->working.postTrace, impl_->accepted.postTrace);
        copyBuffer(blit, impl_->working.weights, impl_->accepted.weights);
        copyBuffer(blit, impl_->working.depression, impl_->accepted.depression);
        copyBuffer(blit, impl_->working.spikes, impl_->accepted.spikes);
        copyBuffer(blit, impl_->working.spikeHistory, impl_->accepted.spikeHistory);
        copyBuffer(blit, impl_->working.electrodeCounts, impl_->accepted.electrodeCounts);
    } else if (impl_->preparedKind == Impl::PreparedKind::growth) {
        copyBuffer(blit, impl_->working.phase, impl_->accepted.phase);
        copyBuffer(blit, impl_->working.tubulin, impl_->accepted.tubulin);
    } else {
        [blit endEncoding];
        return MetalNeuronCultureStatus::invalidArgument;
    }
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted)
        return MetalNeuronCultureStatus::commandFailure;
    impl_->acceptedTick = impl_->preparedTick;
    impl_->acceptedGrowthIteration = impl_->preparedGrowthIteration;
    impl_->preparedKind = Impl::PreparedKind::none;
    impl_->active.reset();
    impl_->hasPrepared = false;
    return MetalNeuronCultureStatus::success;
}

void MetalNeuronCultureRuntime::rejectPrepared() noexcept {
    if (!valid()) return;
    std::scoped_lock lock(impl_->mutex);
    if (impl_->active && impl_->active->status.load() == MetalNeuronCultureStatus::pending) return;
    impl_->preparedKind = Impl::PreparedKind::none;
    impl_->active.reset();
    impl_->hasPrepared = false;
}

NeuronCultureState MetalNeuronCultureRuntime::snapshotAcceptedForTesting() const {
    NeuronCultureState result;
    if (!valid()) return result;
    std::scoped_lock lock(impl_->mutex);
    const auto& h = impl_->culture.header();
    auto copyFloat = [](id<MTLBuffer> buffer, std::size_t count) {
        std::vector<float> values(count);
        std::memcpy(values.data(), buffer.contents, count * sizeof(float));
        return values;
    };
    auto copyU32 = [](id<MTLBuffer> buffer, std::size_t count) {
        std::vector<std::uint32_t> values(count);
        std::memcpy(values.data(), buffer.contents, count * sizeof(std::uint32_t));
        return values;
    };
    result.membrane = copyFloat(impl_->accepted.membrane, h.neuronCount);
    result.refractory = copyFloat(impl_->accepted.refractory, h.neuronCount);
    result.preTrace = copyFloat(impl_->accepted.preTrace, h.neuronCount);
    result.postTrace = copyFloat(impl_->accepted.postTrace, h.neuronCount);
    result.weights = copyFloat(impl_->accepted.weights, h.synapseCount);
    result.depression = copyFloat(impl_->accepted.depression, h.synapseCount);
    result.spikes = copyU32(impl_->accepted.spikes, h.neuronCount);
    result.spikeHistory = copyU32(impl_->accepted.spikeHistory,
                                  static_cast<std::size_t>(h.neuronCount) * 256u);
    result.electrodeSpikeCounts = copyU32(impl_->accepted.electrodeCounts, h.electrodeCount);
    const std::size_t cells = static_cast<std::size_t>(h.growthWidth) * h.growthHeight;
    result.phase = copyFloat(impl_->accepted.phase, cells);
    result.tubulin = copyFloat(impl_->accepted.tubulin, cells);
    result.tick = impl_->acceptedTick;
    result.growthIteration = impl_->acceptedGrowthIteration;
    return result;
}

const char* metalNeuronCultureStatusName(MetalNeuronCultureStatus status) noexcept {
    switch (status) {
        case MetalNeuronCultureStatus::pending: return "pending";
        case MetalNeuronCultureStatus::success: return "success";
        case MetalNeuronCultureStatus::invalidArgument: return "invalid_argument";
        case MetalNeuronCultureStatus::busy: return "busy";
        case MetalNeuronCultureStatus::commandFailure: return "command_failure";
        case MetalNeuronCultureStatus::internalFailure: return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
