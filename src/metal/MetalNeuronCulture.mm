#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNeuronCulture.hpp"
#include "metalrobo/numanx_human_io_gpu.h"

#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifndef METALROBO_NEURON_METALLIB
#define METALROBO_NEURON_METALLIB "NumiNeuron.metallib"
#endif

namespace metalrobo {
namespace {

void neuronCultureMetallibAnchor() noexcept {}

NSURL* neuronCultureMetallibURL() {
    NSFileManager* files = NSFileManager.defaultManager;
    if (const char* overridePath = std::getenv("NUMI_NEURON_METALLIB");
        overridePath != nullptr && overridePath[0] != '\0') {
        NSString* path = [NSString stringWithUTF8String:overridePath];
        if ([files fileExistsAtPath:path]) return [NSURL fileURLWithPath:path];
    }
    Dl_info image{};
    if (dladdr(reinterpret_cast<const void*>(&neuronCultureMetallibAnchor),
               &image) != 0 && image.dli_fname != nullptr) {
        NSURL* library = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:image.dli_fname]];
        NSURL* installed = [[library URLByDeletingLastPathComponent]
            URLByAppendingPathComponent:@"metalrobo/NumiNeuron.metallib"];
        if ([files fileExistsAtPath:installed.path]) return installed;
    }
    NSString* buildPath = @METALROBO_NEURON_METALLIB;
    return [files fileExistsAtPath:buildPath] ?
        [NSURL fileURLWithPath:buildPath] : nil;
}

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
    id<MTLBuffer> retainedSchedule = nil;
    id<MTLSharedEvent> completionEvent = nil;
    std::uint64_t completionValue = 0u;
    std::atomic<MetalNeuronCultureStatus> status{MetalNeuronCultureStatus::pending};
    std::mutex completionMutex;
    void* completionContext = nullptr;
    MetalNeuronCultureCompletion completion = nullptr;
    bool completionDelivered = false;
};

struct MetalNeuronCultureAcceptedView::State {
    std::uint64_t cultureFingerprint = 0u;
    std::uint64_t generation = 0u;
    std::uint64_t tick = 0u;
    std::uint64_t growthIteration = 0u;
    id<MTLSharedEvent> completionEvent = nil;
    std::uint64_t completionValue = 0u;
    std::array<id<MTLBuffer>, 11u> retained{};
    std::array<MetalNeuronCultureBufferView, 11u> views{};
};

template <typename TicketState>
void settleTicket(
    const std::shared_ptr<TicketState>& state,
    const MetalNeuronCultureStatus status
) noexcept {
    state->status.store(status, std::memory_order_release);
    MetalNeuronCultureCompletion completion = nullptr;
    void* context = nullptr;
    {
        std::scoped_lock lock(state->completionMutex);
        completion = state->completion;
        context = state->completionContext;
        state->completion = nullptr;
        state->completionContext = nullptr;
        state->completionDelivered = completion != nullptr;
    }
    if (completion != nullptr) completion(context, status);
}

struct MetalNeuronCultureRuntime::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> tickPipeline = nil;
    id<MTLComputePipelineState> plasticityPipeline = nil;
    id<MTLComputePipelineState> recordPipeline = nil;
    id<MTLComputePipelineState> historyPipeline = nil;
    id<MTLComputePipelineState> growthPipeline = nil;
    id<MTLComputePipelineState> windowPipeline = nil;
    id<MTLComputePipelineState> supportPipeline = nil;
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
    std::uint64_t acceptedGeneration = 0u;
    id<MTLSharedEvent> acceptedEvent = nil;
    id<MTLSharedEvent> preparedEvent = nil;
    std::uint64_t nextPreparedEventValue = 1u;
    std::uint64_t baseResidentBytes = 0u;
    std::uint64_t peakBytes = 0u;

    bool renewWorking() {
        working.membrane = makeBytes(device, accepted.membrane.length, @"Prepared membrane");
        working.refractory = makeBytes(device, accepted.refractory.length, @"Prepared refractory");
        working.preTrace = makeBytes(device, accepted.preTrace.length, @"Prepared pre trace");
        working.postTrace = makeBytes(device, accepted.postTrace.length, @"Prepared post trace");
        working.weights = makeBytes(device, accepted.weights.length, @"Prepared weights");
        working.depression = makeBytes(device, accepted.depression.length, @"Prepared depression");
        working.spikes = makeBytes(device, accepted.spikes.length, @"Prepared spikes");
        working.spikeHistory = makeBytes(device, accepted.spikeHistory.length, @"Prepared spike history");
        working.electrodeCounts = makeBytes(device, accepted.electrodeCounts.length, @"Prepared MEA counts");
        working.phase = makeBytes(device, accepted.phase.length, @"Prepared phase");
        working.tubulin = makeBytes(device, accepted.tubulin.length, @"Prepared tubulin");
        return working.membrane && working.refractory && working.preTrace &&
            working.postTrace && working.weights && working.depression && working.spikes &&
            working.spikeHistory && working.electrodeCounts && working.phase && working.tubulin;
    }

    bool pipelinesValid() const noexcept {
        return device && queue && library && tickPipeline && plasticityPipeline &&
            recordPipeline && historyPipeline && growthPipeline && windowPipeline &&
            supportPipeline;
    }
};

MetalNeuronCultureTicket::MetalNeuronCultureTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
MetalNeuronCultureTicket::MetalNeuronCultureTicket(MetalNeuronCultureTicket&&) noexcept = default;
MetalNeuronCultureTicket& MetalNeuronCultureTicket::operator=(MetalNeuronCultureTicket&&) noexcept = default;
MetalNeuronCultureTicket::~MetalNeuronCultureTicket() = default;

MetalNeuronCultureAcceptedView::MetalNeuronCultureAcceptedView(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
bool MetalNeuronCultureAcceptedView::valid() const noexcept {
    return state_ && state_->cultureFingerprint != 0u && state_->completionEvent;
}
std::uint64_t MetalNeuronCultureAcceptedView::cultureFingerprint() const noexcept {
    return valid() ? state_->cultureFingerprint : 0u;
}
std::uint64_t MetalNeuronCultureAcceptedView::generation() const noexcept {
    return valid() ? state_->generation : 0u;
}
std::uint64_t MetalNeuronCultureAcceptedView::tick() const noexcept {
    return valid() ? state_->tick : 0u;
}
std::uint64_t MetalNeuronCultureAcceptedView::growthIteration() const noexcept {
    return valid() ? state_->growthIteration : 0u;
}
void* MetalNeuronCultureAcceptedView::completionEvent() const noexcept {
    return valid() ? (__bridge void*)state_->completionEvent : nullptr;
}
std::uint64_t MetalNeuronCultureAcceptedView::completionValue() const noexcept {
    return valid() ? state_->completionValue : 0u;
}
std::span<const MetalNeuronCultureBufferView>
MetalNeuronCultureAcceptedView::buffers() const noexcept {
    return valid() ? std::span<const MetalNeuronCultureBufferView>(state_->views) :
        std::span<const MetalNeuronCultureBufferView>{};
}

bool MetalNeuronCultureTicket::valid() const noexcept { return state_ && state_->commandBuffer; }
bool MetalNeuronCultureTicket::completed() const noexcept {
    return state_ && state_->status.load(std::memory_order_acquire) != MetalNeuronCultureStatus::pending;
}
MetalNeuronCultureStatus MetalNeuronCultureTicket::wait() noexcept {
    if (!valid()) return MetalNeuronCultureStatus::invalidArgument;
    [state_->commandBuffer waitUntilCompleted];
    return state_->status.load(std::memory_order_acquire);
}
void* MetalNeuronCultureTicket::completionEvent() const noexcept {
    return valid() ? (__bridge void*)state_->completionEvent : nullptr;
}
std::uint64_t MetalNeuronCultureTicket::completionValue() const noexcept {
    return valid() ? state_->completionValue : 0u;
}

bool MetalNeuronCultureTicket::onCompleted(
    void* context, MetalNeuronCultureCompletion completion) noexcept {
    if (!valid() || completion == nullptr) return false;
    MetalNeuronCultureStatus settled = MetalNeuronCultureStatus::pending;
    {
        std::scoped_lock lock(state_->completionMutex);
        if (state_->completion != nullptr || state_->completionDelivered) return false;
        settled = state_->status.load(std::memory_order_acquire);
        if (settled == MetalNeuronCultureStatus::pending) {
            state_->completionContext = context;
            state_->completion = completion;
            return true;
        }
        state_->completionDelivered = true;
    }
    completion(context, settled);
    return true;
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
    impl->acceptedEvent = [device newSharedEvent];
    impl->preparedEvent = [device newSharedEvent];
    NSURL* url = neuronCultureMetallibURL();
    NSError* error = nil;
    impl->library = url ? [device newLibraryWithURL:url error:&error] : nil;
    if (error || !impl->library || !impl->queue || !impl->acceptedEvent ||
        !impl->preparedEvent) return runtime;
    impl->tickPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_tick");
    impl->plasticityPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_plasticity");
    impl->recordPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_record");
    impl->historyPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_store_history");
    impl->growthPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_growth");
    impl->windowPipeline = makePipeline(device, impl->library, @"mr_neuron_culture_window");
    impl->supportPipeline = makePipeline(
        device, impl->library, @"mr_neuron_culture_support_schedule");
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
    const auto sumBuffer = [](std::uint64_t& total, id<MTLBuffer> buffer) {
        total += buffer != nil ? static_cast<std::uint64_t>(buffer.length) : 0u;
    };
    auto& resident = runtime.impl_->baseResidentBytes;
    sumBuffer(resident, runtime.impl_->header);
    sumBuffer(resident, runtime.impl_->growth);
    sumBuffer(resident, runtime.impl_->neurons);
    sumBuffer(resident, runtime.impl_->synapses);
    sumBuffer(resident, runtime.impl_->electrodes);
    const auto sumState = [&](const Impl::Buffers& buffers) {
        sumBuffer(resident, buffers.membrane); sumBuffer(resident, buffers.refractory);
        sumBuffer(resident, buffers.preTrace); sumBuffer(resident, buffers.postTrace);
        sumBuffer(resident, buffers.weights); sumBuffer(resident, buffers.depression);
        sumBuffer(resident, buffers.spikes); sumBuffer(resident, buffers.spikeHistory);
        sumBuffer(resident, buffers.electrodeCounts); sumBuffer(resident, buffers.phase);
        sumBuffer(resident, buffers.tubulin);
    };
    sumState(runtime.impl_->accepted);
    sumState(runtime.impl_->working);
    sumBuffer(resident, runtime.impl_->phaseScratch);
    sumBuffer(resident, runtime.impl_->tubulinScratch);
    runtime.impl_->peakBytes = resident;
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

std::uint64_t MetalNeuronCultureRuntime::residentBytes() const noexcept {
    if (!valid()) return 0u;
    std::scoped_lock lock(impl_->mutex);
    return impl_->baseResidentBytes +
        (impl_->active && impl_->active->retainedSchedule != nil
            ? static_cast<std::uint64_t>(impl_->active->retainedSchedule.length) : 0u);
}

std::uint64_t MetalNeuronCultureRuntime::peakResidentBytes() const noexcept {
    if (!valid()) return 0u;
    std::scoped_lock lock(impl_->mutex);
    return impl_->peakBytes;
}

MetalNeuronCultureTicket MetalNeuronCultureRuntime::prepareTicks(
    std::uint32_t tickCount,
    std::uint32_t stimulationElectrode,
    float stimulationCurrent
) {
    if (!valid() || tickCount == 0u || !std::isfinite(stimulationCurrent) ||
        (stimulationElectrode != std::numeric_limits<std::uint32_t>::max() &&
         stimulationElectrode >= impl_->culture.header().electrodeCount)) return {};
    NeuronCultureWindowRequest request{
        .cultureFingerprint = impl_->culture.fingerprint(),
        .rootFingerprint = impl_->culture.fingerprint(),
        .tickCount = tickCount,
    };
    if (stimulationElectrode < impl_->culture.header().electrodeCount) {
        request.pulses.push_back({
            .electrode = stimulationElectrode,
            .startTick = 0u,
            .durationTicks = tickCount,
            .source = NeuronCultureStimulusSource::authored,
            .current = stimulationCurrent,
            .sourceFingerprint = impl_->culture.fingerprint(),
        });
    }
    return prepareWindow(request);
}

MetalNeuronCultureTicket MetalNeuronCultureRuntime::prepareWindow(
    const NeuronCultureWindowRequest& request
) {
    if (!valid() || !validateNeuronCultureWindow(impl_->culture, request)) return {};
    std::scoped_lock lock(impl_->mutex);
    if (impl_->active && impl_->active->status.load() == MetalNeuronCultureStatus::pending) return {};
    if (impl_->hasPrepared) return {};
    if (!impl_->renewWorking()) return {};
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
    const std::size_t currentCount = static_cast<std::size_t>(request.tickCount) *
        h.electrodeCount;
    std::vector<float> allCurrents(currentCount, 0.0f);
    for (std::uint32_t offset = 0u; offset < request.tickCount; ++offset) {
        auto activeCurrents = std::span<float>(allCurrents.data() +
            static_cast<std::size_t>(offset) * h.electrodeCount, h.electrodeCount);
        if (!neuronCultureStimulusCurrents(
                impl_->culture, request, offset, activeCurrents)) return {};
    }
    id<MTLBuffer> currents = makeBuffer(impl_->device, std::span<const float>(allCurrents),
                                        @"Neuron culture stimulus schedule");
    if (!currents || impl_->windowPipeline.maxTotalThreadsPerThreadgroup < 256u) return {};
    const std::uint32_t recordingDuration = request.recordingDurationTicks == 0u ?
        request.tickCount - request.recordingStartTick : request.recordingDurationTicks;
    const MRNeuronCultureWindowGPU window{
        .startTick = static_cast<std::uint32_t>(impl_->acceptedTick & 0xffffffffu),
        .tickCount = request.tickCount,
        .recordingStartTick = request.recordingStartTick,
        .recordingDurationTicks = recordingDuration,
        .traceDecay = std::exp(-h.neuralTimestepSeconds / h.traceTimeConstantSeconds),
        .depressionRecovery = std::min(1.0f,
            h.neuralTimestepSeconds / h.depressionRecoverySeconds),
        .status = MR_NEURON_CULTURE_STATUS_PENDING,
        .flags = request.plasticityEnabled ? 0u :
            MR_NEURON_CULTURE_WINDOW_DISABLE_PLASTICITY,
    };
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:impl_->windowPipeline];
    [encoder setBuffer:impl_->header offset:0 atIndex:0];
    [encoder setBuffer:impl_->neurons offset:0 atIndex:1];
    [encoder setBuffer:impl_->synapses offset:0 atIndex:2];
    [encoder setBuffer:impl_->electrodes offset:0 atIndex:3];
    [encoder setBytes:&window length:sizeof(window) atIndex:4];
    [encoder setBuffer:currents offset:0 atIndex:5];
    [encoder setBuffer:impl_->working.membrane offset:0 atIndex:6];
    [encoder setBuffer:impl_->working.refractory offset:0 atIndex:7];
    [encoder setBuffer:impl_->working.preTrace offset:0 atIndex:8];
    [encoder setBuffer:impl_->working.postTrace offset:0 atIndex:9];
    [encoder setBuffer:impl_->working.weights offset:0 atIndex:10];
    [encoder setBuffer:impl_->working.depression offset:0 atIndex:11];
    [encoder setBuffer:impl_->working.spikeHistory offset:0 atIndex:12];
    [encoder setBuffer:impl_->working.spikes offset:0 atIndex:13];
    [encoder setBuffer:impl_->working.electrodeCounts offset:0 atIndex:14];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (impl_->nextPreparedEventValue ==
        std::numeric_limits<std::uint64_t>::max()) return {};
    const std::uint64_t preparedEventValue = impl_->nextPreparedEventValue++;
    [commandBuffer encodeSignalEvent:impl_->preparedEvent value:preparedEventValue];
    auto state = std::make_shared<MetalNeuronCultureTicket::State>();
    state->commandBuffer = commandBuffer;
    state->retainedSchedule = currents;
    state->completionEvent = impl_->preparedEvent;
    state->completionValue = preparedEventValue;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        settleTicket(state, completed.status == MTLCommandBufferStatusCompleted ?
            MetalNeuronCultureStatus::success : MetalNeuronCultureStatus::commandFailure);
    }];
    impl_->active = state;
    impl_->peakBytes = std::max(
        impl_->peakBytes,
        impl_->baseResidentBytes + static_cast<std::uint64_t>(currents.length));
    impl_->hasPrepared = true;
    impl_->preparedKind = Impl::PreparedKind::ticks;
    impl_->preparedTick = impl_->acceptedTick + request.tickCount;
    impl_->preparedGrowthIteration = impl_->acceptedGrowthIteration;
    [commandBuffer commit];
    return MetalNeuronCultureTicket(state);
}

MetalNeuronCultureTicket MetalNeuronCultureRuntime::prepareSupportWindow(
    const MetalNeuronCultureSupportRequest& request
) {
    if (!valid() || request.cultureFingerprint != impl_->culture.fingerprint() ||
        request.rootFingerprint == 0u || request.supportConsequencesBuffer == nullptr ||
        request.supportConsequencesGPUAddress == 0u || request.supportCount != 10u ||
        request.supportStride != 10u || request.tickCount == 0u ||
        request.tickCount > kNeuronCultureMaximumWindowTicks ||
        !std::isfinite(request.physicsTimestepSeconds) ||
        request.physicsTimestepSeconds <= 0.0f ||
        !std::isfinite(request.currentPerNewton) || request.currentPerNewton < 0.0f) {
        return {};
    }
    __unsafe_unretained id object = (__bridge id)request.supportConsequencesBuffer;
    if (![object conformsToProtocol:@protocol(MTLBuffer)]) return {};
    __unsafe_unretained id<MTLBuffer> support =
        (__bridge id<MTLBuffer>)request.supportConsequencesBuffer;
    const std::uint64_t supportBytes = request.supportCount *
        sizeof(MRNumanXHumanSupportConsequenceGPU);
    const std::uint64_t supportBase = support.gpuAddress;
    if (support.device != impl_->device || supportBase == 0u ||
        request.supportConsequencesGPUAddress < supportBase ||
        request.supportConsequencesGPUAddress - supportBase > support.length ||
        supportBytes > support.length -
            (request.supportConsequencesGPUAddress - supportBase)) return {};

    std::scoped_lock lock(impl_->mutex);
    if ((impl_->active && impl_->active->status.load() ==
            MetalNeuronCultureStatus::pending) || impl_->hasPrepared) return {};
    const std::uint64_t supportEnd = request.supportConsequencesGPUAddress + supportBytes;
    if (supportEnd < request.supportConsequencesGPUAddress) return {};
    const auto overlaps = [&](id<MTLBuffer> buffer) {
        if (buffer == nil || buffer.gpuAddress == 0u) return true;
        const std::uint64_t end = buffer.gpuAddress + buffer.length;
        return end < buffer.gpuAddress ||
            (request.supportConsequencesGPUAddress < end &&
             buffer.gpuAddress < supportEnd);
    };
    const id<MTLBuffer> protectedBuffers[] = {
        impl_->header, impl_->growth, impl_->neurons, impl_->synapses,
        impl_->electrodes, impl_->accepted.membrane, impl_->accepted.refractory,
        impl_->accepted.preTrace, impl_->accepted.postTrace, impl_->accepted.weights,
        impl_->accepted.depression, impl_->accepted.spikes,
        impl_->accepted.spikeHistory, impl_->accepted.electrodeCounts,
        impl_->accepted.phase, impl_->accepted.tubulin};
    for (id<MTLBuffer> buffer : protectedBuffers) if (overlaps(buffer)) return {};
    if (!impl_->renewWorking()) return {};
    id<MTLCommandBuffer> commandBuffer = [impl_->queue commandBuffer];
    if (commandBuffer == nil) return {};
    commandBuffer.label = @"NumiNeuron NHCNT culture prepare";
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
    const auto& header = impl_->culture.header();
    const std::uint64_t currentCount =
        static_cast<std::uint64_t>(request.tickCount) * header.electrodeCount;
    if (currentCount > std::numeric_limits<NSUInteger>::max() / sizeof(float)) return {};
    id<MTLBuffer> currents = makeBytes(
        impl_->device, static_cast<NSUInteger>(currentCount * sizeof(float)),
        @"Neuron culture NHCNT schedule");
    if (currents == nil) return {};
    const MRNeuronCultureSupportDispatchGPU supportDispatch{
        .supportCount = request.supportCount,
        .supportStride = request.supportStride,
        .electrodeCount = header.electrodeCount,
        .tickCount = request.tickCount,
        .physicsTimestepSeconds = request.physicsTimestepSeconds,
        .currentPerNewton = request.currentPerNewton,
        .status = MR_NEURON_CULTURE_STATUS_PENDING,
        .reserved0 = 0u,
    };
    id<MTLComputeCommandEncoder> supportEncoder = [commandBuffer computeCommandEncoder];
    [supportEncoder setComputePipelineState:impl_->supportPipeline];
    [supportEncoder setBuffer:impl_->header offset:0u atIndex:0u];
    [supportEncoder setBuffer:impl_->electrodes offset:0u atIndex:1u];
    [supportEncoder setBytes:&supportDispatch length:sizeof(supportDispatch) atIndex:2u];
    [supportEncoder setBuffer:support
        offset:static_cast<NSUInteger>(request.supportConsequencesGPUAddress - supportBase)
        atIndex:3u];
    [supportEncoder setBuffer:currents offset:0u atIndex:4u];
    [supportEncoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [supportEncoder endEncoding];
    const MRNeuronCultureWindowGPU window{
        .startTick = static_cast<std::uint32_t>(impl_->acceptedTick & 0xffffffffu),
        .tickCount = request.tickCount,
        .recordingStartTick = 0u,
        .recordingDurationTicks = request.tickCount,
        .traceDecay = std::exp(-header.neuralTimestepSeconds /
            header.traceTimeConstantSeconds),
        .depressionRecovery = std::min(1.0f, header.neuralTimestepSeconds /
            header.depressionRecoverySeconds),
        .status = MR_NEURON_CULTURE_STATUS_PENDING,
        .flags = 0u,
    };
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:impl_->windowPipeline];
    [encoder setBuffer:impl_->header offset:0u atIndex:0u];
    [encoder setBuffer:impl_->neurons offset:0u atIndex:1u];
    [encoder setBuffer:impl_->synapses offset:0u atIndex:2u];
    [encoder setBuffer:impl_->electrodes offset:0u atIndex:3u];
    [encoder setBytes:&window length:sizeof(window) atIndex:4u];
    [encoder setBuffer:currents offset:0u atIndex:5u];
    [encoder setBuffer:impl_->working.membrane offset:0u atIndex:6u];
    [encoder setBuffer:impl_->working.refractory offset:0u atIndex:7u];
    [encoder setBuffer:impl_->working.preTrace offset:0u atIndex:8u];
    [encoder setBuffer:impl_->working.postTrace offset:0u atIndex:9u];
    [encoder setBuffer:impl_->working.weights offset:0u atIndex:10u];
    [encoder setBuffer:impl_->working.depression offset:0u atIndex:11u];
    [encoder setBuffer:impl_->working.spikeHistory offset:0u atIndex:12u];
    [encoder setBuffer:impl_->working.spikes offset:0u atIndex:13u];
    [encoder setBuffer:impl_->working.electrodeCounts offset:0u atIndex:14u];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (impl_->nextPreparedEventValue ==
        std::numeric_limits<std::uint64_t>::max()) return {};
    const std::uint64_t preparedEventValue = impl_->nextPreparedEventValue++;
    [commandBuffer encodeSignalEvent:impl_->preparedEvent value:preparedEventValue];
    auto state = std::make_shared<MetalNeuronCultureTicket::State>();
    state->commandBuffer = commandBuffer;
    state->retainedSchedule = currents;
    state->completionEvent = impl_->preparedEvent;
    state->completionValue = preparedEventValue;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        settleTicket(state, completed.status == MTLCommandBufferStatusCompleted ?
            MetalNeuronCultureStatus::success : MetalNeuronCultureStatus::commandFailure);
    }];
    impl_->active = state;
    impl_->peakBytes = std::max(
        impl_->peakBytes,
        impl_->baseResidentBytes + static_cast<std::uint64_t>(currents.length));
    impl_->hasPrepared = true;
    impl_->preparedKind = Impl::PreparedKind::ticks;
    impl_->preparedTick = impl_->acceptedTick + request.tickCount;
    impl_->preparedGrowthIteration = impl_->acceptedGrowthIteration;
    [commandBuffer commit];
    return MetalNeuronCultureTicket(state);
}

MetalNeuronCultureTicket MetalNeuronCultureRuntime::prepareGrowth(std::uint32_t iterationCount) {
    if (!valid() || iterationCount == 0u) return {};
    std::scoped_lock lock(impl_->mutex);
    if (impl_->active && impl_->active->status.load() == MetalNeuronCultureStatus::pending) return {};
    if (impl_->hasPrepared) return {};
    if (!impl_->renewWorking()) return {};
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
    if (impl_->nextPreparedEventValue ==
        std::numeric_limits<std::uint64_t>::max()) return {};
    const std::uint64_t preparedEventValue = impl_->nextPreparedEventValue++;
    [commandBuffer encodeSignalEvent:impl_->preparedEvent value:preparedEventValue];
    auto state = std::make_shared<MetalNeuronCultureTicket::State>();
    state->commandBuffer = commandBuffer;
    state->completionEvent = impl_->preparedEvent;
    state->completionValue = preparedEventValue;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        settleTicket(state, completed.status == MTLCommandBufferStatusCompleted ?
            MetalNeuronCultureStatus::success : MetalNeuronCultureStatus::commandFailure);
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
    if (impl_->preparedKind == Impl::PreparedKind::ticks) {
        std::swap(impl_->working.membrane, impl_->accepted.membrane);
        std::swap(impl_->working.refractory, impl_->accepted.refractory);
        std::swap(impl_->working.preTrace, impl_->accepted.preTrace);
        std::swap(impl_->working.postTrace, impl_->accepted.postTrace);
        std::swap(impl_->working.weights, impl_->accepted.weights);
        std::swap(impl_->working.depression, impl_->accepted.depression);
        std::swap(impl_->working.spikes, impl_->accepted.spikes);
        std::swap(impl_->working.spikeHistory, impl_->accepted.spikeHistory);
        std::swap(impl_->working.electrodeCounts, impl_->accepted.electrodeCounts);
    } else if (impl_->preparedKind == Impl::PreparedKind::growth) {
        std::swap(impl_->working.phase, impl_->accepted.phase);
        std::swap(impl_->working.tubulin, impl_->accepted.tubulin);
    } else {
        return MetalNeuronCultureStatus::invalidArgument;
    }
    impl_->acceptedTick = impl_->preparedTick;
    impl_->acceptedGrowthIteration = impl_->preparedGrowthIteration;
    ++impl_->acceptedGeneration;
    impl_->acceptedEvent.signaledValue = impl_->acceptedGeneration;
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

MetalNeuronCultureStatus MetalNeuronCultureRuntime::restoreAccepted(
    const NeuronCultureState& state
) noexcept {
    if (!valid()) return MetalNeuronCultureStatus::invalidArgument;
    NeuronCultureReference validator(impl_->culture);
    if (!validator.restoreAccepted(state)) return MetalNeuronCultureStatus::invalidArgument;
    std::scoped_lock lock(impl_->mutex);
    if (impl_->hasPrepared ||
        (impl_->active && impl_->active->status.load() == MetalNeuronCultureStatus::pending)) {
        return MetalNeuronCultureStatus::busy;
    }
    if (state.generation < impl_->acceptedGeneration ||
        (state.generation == impl_->acceptedGeneration &&
         impl_->acceptedGeneration != 0u)) {
        return MetalNeuronCultureStatus::invalidArgument;
    }
    auto copy = [](id<MTLBuffer> target, const auto& values) {
        std::memcpy(target.contents, values.data(), values.size() * sizeof(values.front()));
    };
    copy(impl_->accepted.membrane, state.membrane);
    copy(impl_->accepted.refractory, state.refractory);
    copy(impl_->accepted.preTrace, state.preTrace);
    copy(impl_->accepted.postTrace, state.postTrace);
    copy(impl_->accepted.weights, state.weights);
    copy(impl_->accepted.depression, state.depression);
    copy(impl_->accepted.spikes, state.spikes);
    copy(impl_->accepted.spikeHistory, state.spikeHistory);
    copy(impl_->accepted.electrodeCounts, state.electrodeSpikeCounts);
    copy(impl_->accepted.phase, state.phase);
    copy(impl_->accepted.tubulin, state.tubulin);
    impl_->acceptedTick = state.tick;
    impl_->acceptedGrowthIteration = state.growthIteration;
    impl_->acceptedGeneration = state.generation;
    impl_->acceptedEvent.signaledValue = state.generation;
    return MetalNeuronCultureStatus::success;
}

MetalNeuronCultureAcceptedView MetalNeuronCultureRuntime::acceptedView() const noexcept {
    try {
        if (!valid()) return {};
        std::scoped_lock lock(impl_->mutex);
        auto state = std::make_shared<MetalNeuronCultureAcceptedView::State>();
        state->cultureFingerprint = impl_->culture.fingerprint();
        state->generation = impl_->acceptedGeneration;
        state->tick = impl_->acceptedTick;
        state->growthIteration = impl_->acceptedGrowthIteration;
        state->completionEvent = impl_->acceptedEvent;
        state->completionValue = impl_->acceptedGeneration;
        state->retained = {impl_->accepted.membrane, impl_->accepted.refractory,
            impl_->accepted.preTrace, impl_->accepted.postTrace, impl_->accepted.weights,
            impl_->accepted.depression, impl_->accepted.spikes, impl_->accepted.spikeHistory,
            impl_->accepted.electrodeCounts, impl_->accepted.phase, impl_->accepted.tubulin};
        for (std::size_t index = 0u; index < state->retained.size(); ++index) {
            id<MTLBuffer> buffer = state->retained[index];
            state->views[index] = {
                .kind = static_cast<MetalNeuronCultureAcceptedBuffer>(index),
                .metalBuffer = (__bridge void*)buffer,
                .gpuAddress = buffer.gpuAddress,
                .byteLength = buffer.length,
            };
        }
        return MetalNeuronCultureAcceptedView(std::move(state));
    } catch (...) {
        return {};
    }
}

MetalNeuronCultureAcceptedView
MetalNeuronCultureRuntime::preparedAcceptedView() const noexcept {
    try {
        if (!valid()) return {};
        std::scoped_lock lock(impl_->mutex);
        if (!impl_->hasPrepared || !impl_->active ||
            impl_->active->status.load(std::memory_order_acquire) !=
                MetalNeuronCultureStatus::success ||
            impl_->acceptedGeneration == std::numeric_limits<std::uint64_t>::max()) {
            return {};
        }
        auto state = std::make_shared<MetalNeuronCultureAcceptedView::State>();
        state->cultureFingerprint = impl_->culture.fingerprint();
        state->generation = impl_->acceptedGeneration + 1u;
        state->tick = impl_->preparedTick;
        state->growthIteration = impl_->preparedGrowthIteration;
        state->completionEvent = impl_->acceptedEvent;
        state->completionValue = state->generation;
        state->retained = {impl_->working.membrane, impl_->working.refractory,
            impl_->working.preTrace, impl_->working.postTrace, impl_->working.weights,
            impl_->working.depression, impl_->working.spikes, impl_->working.spikeHistory,
            impl_->working.electrodeCounts, impl_->working.phase, impl_->working.tubulin};
        for (std::size_t index = 0u; index < state->retained.size(); ++index) {
            id<MTLBuffer> buffer = state->retained[index];
            state->views[index] = {
                .kind = static_cast<MetalNeuronCultureAcceptedBuffer>(index),
                .metalBuffer = (__bridge void*)buffer,
                .gpuAddress = buffer.gpuAddress,
                .byteLength = buffer.length,
            };
        }
        return MetalNeuronCultureAcceptedView(std::move(state));
    } catch (...) {
        return {};
    }
}

std::vector<std::uint32_t>
MetalNeuronCultureRuntime::acceptedElectrodeCountsTelemetry() const {
    if (!valid()) return {};
    std::scoped_lock lock(impl_->mutex);
    std::vector<std::uint32_t> counts(impl_->culture.header().electrodeCount);
    std::memcpy(counts.data(), impl_->accepted.electrodeCounts.contents,
                counts.size() * sizeof(std::uint32_t));
    return counts;
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
    result.generation = impl_->acceptedGeneration;
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
