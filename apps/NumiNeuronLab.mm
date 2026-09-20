#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "metalrobo/MetalNeuronCulture.hpp"
#include "metalrobo/NeuronCultureArtifacts.hpp"
#include "metalrobo/NeuronCultureProtocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef METALROBO_NEURON_METALLIB
#define METALROBO_NEURON_METALLIB "NumiNeuron.metallib"
#endif

namespace {
struct VisualParams {
    std::uint32_t width, height, growthWidth, growthHeight;
    std::uint32_t neuronCount, synapseCount, electrodeCount, acceptedGeneration;
    float animatX, animatY;
    std::uint32_t protocolPhase, transactionStatus;
    float minimumWeight, maximumWeight;
    std::uint32_t histogramBins, reserved0;
};
static_assert(sizeof(VisualParams) == 64u);
constexpr std::uint32_t kHistogramBins = 256u;
constexpr std::uint32_t kVisualPointMapSize = 256u;

NSURL* neuronMetallibURL() {
    NSFileManager* files = NSFileManager.defaultManager;
    if (const char* overridePath = std::getenv("NUMI_NEURON_METALLIB");
        overridePath != nullptr && overridePath[0] != '\0') {
        NSString* path = [NSString stringWithUTF8String:overridePath];
        if ([files fileExistsAtPath:path]) return [NSURL fileURLWithPath:path];
    }
    NSURL* executable = NSBundle.mainBundle.executableURL;
    NSURL* prefix = [[executable URLByDeletingLastPathComponent]
        URLByDeletingLastPathComponent];
    NSURL* installed = [prefix URLByAppendingPathComponent:
        @"lib/metalrobo/NumiNeuron.metallib"];
    if ([files fileExistsAtPath:installed.path]) return installed;
    NSString* buildPath = @METALROBO_NEURON_METALLIB;
    return [files fileExistsAtPath:buildPath] ?
        [NSURL fileURLWithPath:buildPath] : nil;
}

id<MTLBuffer> bufferFor(const metalrobo::MetalNeuronCultureAcceptedView& view,
                        metalrobo::MetalNeuronCultureAcceptedBuffer kind) {
    for (const auto& item : view.buffers())
        if (item.kind == kind) return (__bridge id<MTLBuffer>)item.metalBuffer;
    return nil;
}

metalrobo::CompiledNeuronCulture compileCulture(std::uint64_t seed) {
    auto pack = metalrobo::makePotterReferenceCulture(1000u, 50000u, seed);
    metalrobo::CompiledNeuronCulture culture;
    const auto result = metalrobo::compileNeuronCulture(pack, culture);
    if (!result.succeeded()) throw std::runtime_error(result.message);
    return culture;
}

template <typename Item>
id<MTLBuffer> makeVisualPointMap(
    id<MTLDevice> device, const std::span<const Item> items,
    const int radius, NSString* label
) {
    std::vector<std::int32_t> map(
        static_cast<std::size_t>(kVisualPointMapSize) * kVisualPointMapSize,
        -1);
    for (std::size_t index = 0u; index < items.size(); ++index) {
        const int centerX = std::clamp(static_cast<int>(std::lround(
            items[index].x * static_cast<float>(kVisualPointMapSize - 1u) /
            3.0f)), 0, static_cast<int>(kVisualPointMapSize - 1u));
        const int centerY = std::clamp(static_cast<int>(std::lround(
            items[index].y * static_cast<float>(kVisualPointMapSize - 1u) /
            3.0f)), 0, static_cast<int>(kVisualPointMapSize - 1u));
        for (int y = std::max(0, centerY - radius);
             y <= std::min(static_cast<int>(kVisualPointMapSize - 1u),
                           centerY + radius); ++y) {
            for (int x = std::max(0, centerX - radius);
                 x <= std::min(static_cast<int>(kVisualPointMapSize - 1u),
                               centerX + radius); ++x) {
                const int dx = x - centerX;
                const int dy = y - centerY;
                if (dx * dx + dy * dy > radius * radius) continue;
                map[static_cast<std::size_t>(y) * kVisualPointMapSize + x] =
                    static_cast<std::int32_t>(index);
            }
        }
    }
    id<MTLBuffer> buffer = [device newBufferWithBytes:map.data()
        length:map.size() * sizeof(map.front())
        options:MTLResourceStorageModeShared];
    buffer.label = label;
    return buffer;
}

std::string companionBase(std::string path) {
    constexpr std::string_view suffix = ".ncrun.json";
    if (path.ends_with(suffix)) path.resize(path.size() - suffix.size());
    return path;
}

const char* protocolPhaseName(const metalrobo::PotterProtocolPhase phase) {
    switch (phase) {
        case metalrobo::PotterProtocolPhase::calibration: return "calibration";
        case metalrobo::PotterProtocolPhase::baseline: return "baseline";
        case metalrobo::PotterProtocolPhase::postSwitch: return "post-switch";
        case metalrobo::PotterProtocolPhase::complete: return "complete";
    }
    return "invalid";
}

int renderCapture(const std::string& path, const std::string& runPath) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    NSURL* metallib = neuronMetallibURL();
    id<MTLLibrary> library = metallib ?
        [device newLibraryWithURL:metallib error:&error] : nil;
    id<MTLFunction> function = [library newFunctionWithName:@"mr_neuron_culture_visualize"];
    id<MTLFunction> histogramFunction =
        [library newFunctionWithName:@"mr_neuron_culture_visual_histogram"];
    id<MTLComputePipelineState> pipeline = function ?
        [device newComputePipelineStateWithFunction:function error:&error] : nil;
    id<MTLComputePipelineState> histogramPipeline = histogramFunction ?
        [device newComputePipelineStateWithFunction:histogramFunction error:&error] : nil;
    auto culture = compileCulture(2056u);
    metalrobo::NeuronCultureState runState;
    if (!runPath.empty()) {
        const auto base = companionBase(runPath);
        if (!metalrobo::readCompiledNeuronCulture(
                base + ".nculture", culture).succeeded() ||
            !metalrobo::validateNeuronCultureRunManifest(
                runPath, culture.fingerprint(), "potter-switch-v1").succeeded() ||
            !metalrobo::readNeuronCultureCheckpoint(
                culture, base + ".ncstate", runState).succeeded()) {
            return 1;
        }
    }
    auto runtime = metalrobo::MetalNeuronCultureRuntime::create(
        culture, (__bridge void*)device);
    id<MTLBuffer> weightHistogram = [device newBufferWithLength:
        kHistogramBins * sizeof(std::uint32_t) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> depressionHistogram = [device newBufferWithLength:
        kHistogramBins * sizeof(std::uint32_t) options:MTLResourceStorageModePrivate];
    if (!device || !queue || !pipeline || !histogramPipeline || !weightHistogram ||
        !depressionHistogram || error || !runtime.valid()) return 1;
    id<MTLBuffer> neurons = [device newBufferWithBytes:culture.neurons().data()
        length:culture.neurons().size_bytes() options:MTLResourceStorageModeShared];
    id<MTLBuffer> electrodes = [device newBufferWithBytes:culture.electrodes().data()
        length:culture.electrodes().size_bytes() options:MTLResourceStorageModeShared];
    id<MTLBuffer> neuronMap = makeVisualPointMap(
        device, culture.neurons(), 1, @"Neuron visual topology map");
    id<MTLBuffer> electrodeMap = makeVisualPointMap(
        device, culture.electrodes(), 3, @"MEA visual topology map");
    if (!runPath.empty()) {
        if (runtime.restoreAccepted(runState) !=
            metalrobo::MetalNeuronCultureStatus::success) return 1;
    } else {
        auto growth = runtime.prepareGrowth(8u);
        if (!growth.valid() ||
            growth.wait() != metalrobo::MetalNeuronCultureStatus::success ||
            runtime.publishPrepared() != metalrobo::MetalNeuronCultureStatus::success) return 1;
        auto activity = runtime.prepareTicks(
            400u, 7u, metalrobo::PotterProtocolConfig{}.stimulationCurrent);
        if (!activity.valid() ||
            activity.wait() != metalrobo::MetalNeuronCultureStatus::success ||
            runtime.publishPrepared() != metalrobo::MetalNeuronCultureStatus::success) return 1;
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:1024u height:640u mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    const auto accepted = runtime.acceptedView();
    if (!neurons || !electrodes || !neuronMap || !electrodeMap || !texture ||
        !accepted.valid()) return 1;
    VisualParams params{1024u, 640u, culture.header().growthWidth,
        culture.header().growthHeight, culture.header().neuronCount,
        culture.header().synapseCount, culture.header().electrodeCount,
        static_cast<std::uint32_t>(accepted.generation()), 0.0f, 0.0f,
        static_cast<std::uint32_t>(metalrobo::PotterProtocolPhase::complete), 0u,
        culture.header().minimumWeight, culture.header().maximumWeight,
        kHistogramBins, 0u};
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
    [clear fillBuffer:weightHistogram range:NSMakeRange(0u, weightHistogram.length) value:0u];
    [clear fillBuffer:depressionHistogram range:NSMakeRange(0u, depressionHistogram.length) value:0u];
    [clear endEncoding];
    id<MTLComputeCommandEncoder> histogram = [command computeCommandEncoder];
    [histogram setComputePipelineState:histogramPipeline];
    [histogram setBytes:&params length:sizeof(params) atIndex:0];
    [histogram setBuffer:bufferFor(accepted,
        metalrobo::MetalNeuronCultureAcceptedBuffer::weights) offset:0 atIndex:1];
    [histogram setBuffer:bufferFor(accepted,
        metalrobo::MetalNeuronCultureAcceptedBuffer::depression) offset:0 atIndex:2];
    [histogram setBuffer:weightHistogram offset:0 atIndex:3];
    [histogram setBuffer:depressionHistogram offset:0 atIndex:4];
    [histogram dispatchThreads:MTLSizeMake(params.synapseCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [histogram endEncoding];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:neurons offset:0 atIndex:1];
    [encoder setBuffer:electrodes offset:0 atIndex:2];
    const metalrobo::MetalNeuronCultureAcceptedBuffer kinds[] = {
        metalrobo::MetalNeuronCultureAcceptedBuffer::phase,
        metalrobo::MetalNeuronCultureAcceptedBuffer::tubulin,
        metalrobo::MetalNeuronCultureAcceptedBuffer::spikes,
        metalrobo::MetalNeuronCultureAcceptedBuffer::spikeHistory,
        metalrobo::MetalNeuronCultureAcceptedBuffer::electrodeSpikeCounts};
    for (NSUInteger index = 0u; index < 5u; ++index)
        [encoder setBuffer:bufferFor(accepted, kinds[index]) offset:0 atIndex:index + 3u];
    [encoder setBuffer:weightHistogram offset:0 atIndex:8];
    [encoder setBuffer:depressionHistogram offset:0 atIndex:9];
    [encoder setBuffer:neuronMap offset:0 atIndex:10];
    [encoder setBuffer:electrodeMap offset:0 atIndex:11];
    [encoder setTexture:texture atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1024u, 640u, 1u)
       threadsPerThreadgroup:MTLSizeMake(16u, 16u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 1;
    std::vector<unsigned char> bgra(1024u * 640u * 4u);
    [texture getBytes:bgra.data() bytesPerRow:1024u * 4u
        fromRegion:MTLRegionMake2D(0u, 0u, 1024u, 640u) mipmapLevel:0u];
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "P6\n1024 640\n255\n";
    for (std::size_t pixel = 0u; pixel < 1024u * 640u; ++pixel) {
        output.put(static_cast<char>(bgra[pixel * 4u + 2u]));
        output.put(static_cast<char>(bgra[pixel * 4u + 1u]));
        output.put(static_cast<char>(bgra[pixel * 4u + 0u]));
    }
    return output.good() ? 0 : 1;
}
} // namespace

@interface NumiNeuronLabController : NSObject <NSApplicationDelegate, MTKViewDelegate> {
    NSWindow* _window;
    MTKView* _view;
    NSTextField* _status;
    NSPopUpButton* _protocolSelector;
    NSPopUpButton* _seedSelector;
    NSTimer* _timer;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _renderQueue;
    id<MTLComputePipelineState> _visualPipeline;
    id<MTLComputePipelineState> _histogramPipeline;
    id<MTLBuffer> _weightHistogram;
    id<MTLBuffer> _depressionHistogram;
    id<MTLBuffer> _neurons;
    id<MTLBuffer> _electrodes;
    id<MTLBuffer> _neuronMap;
    id<MTLBuffer> _electrodeMap;
    metalrobo::CompiledNeuronCulture _culture;
    metalrobo::MetalNeuronCultureRuntime _runtime;
    std::optional<metalrobo::MetalNeuronCultureTicket> _ticket;
    std::unique_ptr<metalrobo::PotterProtocolSession> _protocol;
    bool _playing;
    bool _singleStep;
    bool _stimulate;
    bool _quarantined;
    std::uint32_t _electrode;
    std::uint64_t _seed;
    std::string _checkpointPath;
    std::string _runPath;
}
@end

@implementation NumiNeuronLabController

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    (void)notification;
    _seed = 2056u;
    _checkpointPath = "/tmp/NumiNeuronLab.ncstate";
    NSArray<NSString*>* arguments = NSProcessInfo.processInfo.arguments;
    for (NSUInteger i = 1u; i < arguments.count; ++i)
        if ([arguments[i] isEqualToString:@"--checkpoint"] && i + 1u < arguments.count)
            _checkpointPath = arguments[++i].UTF8String;
        else if ([arguments[i] isEqualToString:@"--run"] && i + 1u < arguments.count)
            _runPath = arguments[++i].UTF8String;
        else if ([arguments[i] isEqualToString:@"--live"])
            _runPath.clear();
    if (!_runPath.empty()) {
        std::ifstream input(_runPath, std::ios::binary | std::ios::ate);
        if (!input.good() || input.tellg() <= 0 || input.tellg() > 1024 * 1024) {
            [NSApp terminate:nil]; return;
        }
        const auto size = static_cast<std::size_t>(input.tellg());
        std::string manifest(size, '\0');
        input.seekg(0);
        input.read(manifest.data(), static_cast<std::streamsize>(size));
        if (!input.good() || manifest.find("\"schema\": \"numi.neuron-culture.run.v1\"") ==
                std::string::npos) {
            [NSApp terminate:nil]; return;
        }
    }
    _device = MTLCreateSystemDefaultDevice();
    _renderQueue = [_device newCommandQueue];
    NSError* error = nil;
    NSURL* metallib = neuronMetallibURL();
    id<MTLLibrary> library = metallib ?
        [_device newLibraryWithURL:metallib error:&error] : nil;
    id<MTLFunction> function = [library newFunctionWithName:@"mr_neuron_culture_visualize"];
    id<MTLFunction> histogramFunction =
        [library newFunctionWithName:@"mr_neuron_culture_visual_histogram"];
    _visualPipeline = function ? [_device newComputePipelineStateWithFunction:function error:&error] : nil;
    _histogramPipeline = histogramFunction ?
        [_device newComputePipelineStateWithFunction:histogramFunction error:&error] : nil;
    _weightHistogram = [_device newBufferWithLength:kHistogramBins * sizeof(std::uint32_t)
        options:MTLResourceStorageModePrivate];
    _depressionHistogram = [_device newBufferWithLength:kHistogramBins * sizeof(std::uint32_t)
        options:MTLResourceStorageModePrivate];
    if (!_device || !_renderQueue || !_visualPipeline || !_histogramPipeline ||
        !_weightHistogram || !_depressionHistogram || error) {
        [NSApp terminate:nil]; return;
    }

    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(80, 80, 1280, 820)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                  NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
        backing:NSBackingStoreBuffered defer:NO];
    _window.title = @"Numi Neuron Lab — Simulation-Only Metal Research Platform";
    NSView* content = _window.contentView;
    _view = [[MTKView alloc] initWithFrame:NSMakeRect(0, 0, 1280, 770) device:_device];
    _view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    _view.framebufferOnly = NO;
    _view.preferredFramesPerSecond = 30;
    _view.paused = NO;
    _view.enableSetNeedsDisplay = NO;
    _view.delegate = self;
    [content addSubview:_view];
    NSView* controls = [[NSView alloc] initWithFrame:NSMakeRect(0, 770, 1280, 50)];
    controls.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    [content addSubview:controls];
    NSArray<NSString*>* titles = @[@"Play / Pause", @"Step", @"Stimulate", @"Save", @"Load", @"Replay"];
    SEL actions[] = {@selector(togglePlay:), @selector(step:), @selector(stimulate:),
                     @selector(save:), @selector(load:), @selector(replay:)};
    for (NSUInteger i = 0u; i < titles.count; ++i) {
        NSButton* button = [NSButton buttonWithTitle:titles[i] target:self action:actions[i]];
        button.frame = NSMakeRect(12 + i * 104, 10, 96, 30);
        [controls addSubview:button];
    }
    _protocolSelector = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(650, 10, 155, 30)];
    [_protocolSelector addItemsWithTitles:@[@"Free stimulation", @"Potter switch v1"]];
    _protocolSelector.target = self; _protocolSelector.action = @selector(protocolChanged:);
    [controls addSubview:_protocolSelector];
    _seedSelector = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(815, 10, 105, 30)];
    [_seedSelector addItemsWithTitles:@[@"Seed 2056", @"Seed 4099", @"Seed 8191"]];
    _seedSelector.target = self; _seedSelector.action = @selector(seedChanged:);
    [controls addSubview:_seedSelector];
    _status = [NSTextField labelWithString:@"Initializing accepted culture…"];
    _status.frame = NSMakeRect(930, 14, 340, 24);
    _status.textColor = NSColor.secondaryLabelColor;
    [controls addSubview:_status];
    [self resetRuntime];
    _playing = _runPath.empty() && _runtime.valid();
    _timer = [NSTimer scheduledTimerWithTimeInterval:0.01 target:self
        selector:@selector(advanceSimulation:) userInfo:nil repeats:YES];
    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)resetRuntime {
    try {
        _ticket.reset(); _protocol.reset();
        _quarantined = false;
        if (_runPath.empty()) {
            _culture = compileCulture(_seed);
        } else {
            const auto base = companionBase(_runPath);
            const auto packPath = base + ".nculture";
            const auto pack = metalrobo::readCompiledNeuronCulture(
                packPath, _culture);
            if (!pack.succeeded() ||
                !metalrobo::validateNeuronCultureRunManifest(
                    _runPath, _culture.fingerprint(),
                    "potter-switch-v1").succeeded()) {
                throw std::runtime_error(
                    "run manifest or companion culture is invalid");
            }
            _checkpointPath = base + ".ncstate";
        }
        _runtime = metalrobo::MetalNeuronCultureRuntime::create(_culture, (__bridge void*)_device);
        if (!_runtime.valid()) throw std::runtime_error("Metal culture runtime unavailable");
        _neurons = [_device newBufferWithBytes:_culture.neurons().data()
            length:_culture.neurons().size_bytes() options:MTLResourceStorageModeShared];
        _electrodes = [_device newBufferWithBytes:_culture.electrodes().data()
            length:_culture.electrodes().size_bytes() options:MTLResourceStorageModeShared];
        _neuronMap = makeVisualPointMap(
            _device, _culture.neurons(), 1, @"Neuron visual topology map");
        _electrodeMap = makeVisualPointMap(
            _device, _culture.electrodes(), 3, @"MEA visual topology map");
        if (!_neurons || !_electrodes || !_neuronMap || !_electrodeMap)
            throw std::runtime_error("visual topology allocation failed");
        NSString* checkpoint = [NSString stringWithUTF8String:_checkpointPath.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:checkpoint]) {
            metalrobo::NeuronCultureState state;
            if (!metalrobo::readNeuronCultureCheckpoint(
                    _culture, _checkpointPath, state).succeeded() ||
                _runtime.restoreAccepted(state) !=
                    metalrobo::MetalNeuronCultureStatus::success) {
                throw std::runtime_error("accepted checkpoint restore failed");
            }
        } else if (!_runPath.empty()) {
            throw std::runtime_error("run companion checkpoint is missing");
        }
        _status.stringValue = _runPath.empty() ?
            @"live • accepted • prepared buffers private" :
            @"run manifest • accepted checkpoint restored • paused";
    } catch (const std::exception& error) {
        _quarantined = true;
        _status.stringValue = [NSString stringWithUTF8String:error.what()]; _playing = false;
    }
}

- (void)advanceSimulation:(NSTimer*)timer {
    (void)timer;
    [_view draw];
    if (_ticket) {
        if (!_ticket->completed()) return;
        const bool success = _ticket->wait() == metalrobo::MetalNeuronCultureStatus::success &&
            _runtime.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success;
        _ticket.reset();
        if (!success) { _quarantined = true;
            _status.stringValue = @"quarantined • no publication"; _playing = false; return; }
        if (_protocol && !_protocol->observe(_runtime.acceptedElectrodeCountsTelemetry())) {
            _quarantined = true;
            _status.stringValue = @"quarantined • protocol observation"; _playing = false; return;
        }
        _singleStep = false;
    }
    if ((!_playing && !_singleStep) || _ticket) return;
    if (_protocol) {
        if (_protocol->complete()) { _playing = false; return; }
        _ticket.emplace(_runtime.prepareWindow(_protocol->nextWindow().request));
    } else {
        _ticket.emplace(_runtime.prepareTicks(
            50u, _stimulate ? _electrode : UINT32_MAX,
            _stimulate ? metalrobo::PotterProtocolConfig{}.stimulationCurrent : 0.0f));
        _stimulate = false;
    }
    if (!_ticket->valid()) { _ticket.reset(); _playing = false; _quarantined = true;
        _status.stringValue = @"quarantined • prepare rejected"; }
}

- (void)drawInMTKView:(MTKView*)view {
    if (!(_window.occlusionState & NSWindowOcclusionStateVisible)) return;
    id<CAMetalDrawable> drawable = view.currentDrawable;
    const auto accepted = _runtime.acceptedView();
    if (!drawable || !accepted.valid()) return;
    const auto protocolResult = _protocol ? _protocol->result() :
        metalrobo::PotterProtocolResult{};
    VisualParams p{static_cast<std::uint32_t>(drawable.texture.width),
        static_cast<std::uint32_t>(drawable.texture.height), _culture.header().growthWidth,
        _culture.header().growthHeight, _culture.header().neuronCount,
        _culture.header().synapseCount, _culture.header().electrodeCount,
        static_cast<std::uint32_t>(accepted.generation()),
        protocolResult.finalX, protocolResult.finalY,
        static_cast<std::uint32_t>(_protocol ? _protocol->phase() :
            metalrobo::PotterProtocolPhase::complete),
        _quarantined ? 2u : (_ticket ? 1u : 0u),
        _culture.header().minimumWeight, _culture.header().maximumWeight,
        kHistogramBins, 0u};
    id<MTLCommandBuffer> command = [_renderQueue commandBuffer];
    id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
    [clear fillBuffer:_weightHistogram
        range:NSMakeRange(0u, _weightHistogram.length) value:0u];
    [clear fillBuffer:_depressionHistogram
        range:NSMakeRange(0u, _depressionHistogram.length) value:0u];
    [clear endEncoding];
    id<MTLComputeCommandEncoder> histogram = [command computeCommandEncoder];
    [histogram setComputePipelineState:_histogramPipeline];
    [histogram setBytes:&p length:sizeof(p) atIndex:0];
    [histogram setBuffer:bufferFor(accepted,
        metalrobo::MetalNeuronCultureAcceptedBuffer::weights) offset:0 atIndex:1];
    [histogram setBuffer:bufferFor(accepted,
        metalrobo::MetalNeuronCultureAcceptedBuffer::depression) offset:0 atIndex:2];
    [histogram setBuffer:_weightHistogram offset:0 atIndex:3];
    [histogram setBuffer:_depressionHistogram offset:0 atIndex:4];
    [histogram dispatchThreads:MTLSizeMake(p.synapseCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [histogram endEncoding];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:_visualPipeline];
    [encoder setBytes:&p length:sizeof(p) atIndex:0];
    [encoder setBuffer:_neurons offset:0 atIndex:1]; [encoder setBuffer:_electrodes offset:0 atIndex:2];
    const metalrobo::MetalNeuronCultureAcceptedBuffer kinds[] = {
        metalrobo::MetalNeuronCultureAcceptedBuffer::phase,
        metalrobo::MetalNeuronCultureAcceptedBuffer::tubulin,
        metalrobo::MetalNeuronCultureAcceptedBuffer::spikes,
        metalrobo::MetalNeuronCultureAcceptedBuffer::spikeHistory,
        metalrobo::MetalNeuronCultureAcceptedBuffer::electrodeSpikeCounts};
    for (NSUInteger i = 0u; i < 5u; ++i)
        [encoder setBuffer:bufferFor(accepted, kinds[i]) offset:0 atIndex:i + 3u];
    [encoder setBuffer:_weightHistogram offset:0 atIndex:8];
    [encoder setBuffer:_depressionHistogram offset:0 atIndex:9];
    [encoder setBuffer:_neuronMap offset:0 atIndex:10];
    [encoder setBuffer:_electrodeMap offset:0 atIndex:11];
    [encoder setTexture:drawable.texture atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(p.width, p.height, 1)
       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding]; [command presentDrawable:drawable]; [command commit];
    NSString* transaction = _quarantined ? @"quarantined" :
        (_ticket ? @"prepared" : @"accepted");
    if (_protocol) {
        const auto result = _protocol->result();
        _status.stringValue = [NSString stringWithFormat:
            @"%@ • %s • animat %.2f,%.2f • gen %llu",
            transaction, protocolPhaseName(_protocol->phase()),
            result.finalX, result.finalY, accepted.generation()];
    } else {
        _status.stringValue = [NSString stringWithFormat:
            @"%@ • free/embodiment idle • gen %llu • tick %llu",
            transaction, accepted.generation(), accepted.tick()];
    }
}

- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size { (void)view; (void)size; }
- (void)togglePlay:(id)sender { (void)sender;
    if (!_quarantined) _playing = !_playing; }
- (void)step:(id)sender { (void)sender;
    if (!_quarantined) _singleStep = true; }
- (void)stimulate:(id)sender { (void)sender;
    if (_quarantined) return;
    _stimulate = true;
    _electrode = (_electrode + 1u) % 60u; _singleStep = true; }
- (void)save:(id)sender { (void)sender;
    const auto result = metalrobo::writeNeuronCultureCheckpoint(
        _culture, _runtime.snapshotAcceptedForTesting(), _checkpointPath);
    _status.stringValue = result.succeeded() ? @"checkpoint saved" : @"checkpoint save rejected"; }
- (void)load:(id)sender { (void)sender;
    if (_ticket) { _status.stringValue = @"busy • checkpoint load deferred"; return; }
    NSString* checkpoint = [NSString stringWithUTF8String:_checkpointPath.c_str()];
    if (![[NSFileManager defaultManager] fileExistsAtPath:checkpoint]) {
        _status.stringValue = @"checkpoint load rejected • file missing";
        return;
    }
    _playing = false; _singleStep = false;
    [self resetRuntime];
    [_protocolSelector selectItemAtIndex:0];
    if (!_quarantined) _status.stringValue = @"checkpoint restored • paused"; }
- (void)replay:(id)sender { (void)sender;
    if (_ticket) { _status.stringValue = @"busy • replay deferred"; return; }
    _playing = false; _singleStep = false;
    [self resetRuntime];
    [_protocolSelector selectItemAtIndex:0];
    if (!_quarantined) _status.stringValue = @"accepted state replayed • paused"; }
- (void)protocolChanged:(id)sender { (void)sender;
    if (_ticket) { _status.stringValue = @"busy • protocol change deferred"; return; }
    if (_protocolSelector.indexOfSelectedItem == 1) {
        _runPath.clear();
        [self resetRuntime];
        _protocol = std::make_unique<metalrobo::PotterProtocolSession>(_culture);
        _playing = _protocol->valid();
    } else {
        _protocol.reset();
    }
}
- (void)seedChanged:(id)sender { (void)sender;
    if (_ticket) { _status.stringValue = @"busy • seed change deferred"; return; }
    constexpr std::array<std::uint64_t, 3u> seeds{2056u, 4099u, 8191u};
    const NSInteger selected = _seedSelector.indexOfSelectedItem;
    if (selected < 0 || static_cast<std::size_t>(selected) >= seeds.size()) {
        _status.stringValue = @"seed selection rejected";
        return;
    }
    _seed = seeds[static_cast<std::size_t>(selected)];
    _runPath.clear();
    [self resetRuntime]; }
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    (void)sender; return YES; }
@end

static NumiNeuronLabController* gNeuronLabDelegate;

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::string capturePath;
        std::string runPath;
        for (int index = 1; index + 1 < argc; ++index) {
            if (std::string_view(argv[index]) == "--capture")
                capturePath = argv[++index];
            else if (std::string_view(argv[index]) == "--run")
                runPath = argv[++index];
        }
        if (!capturePath.empty()) return renderCapture(capturePath, runPath);
        NSApplication* application = NSApplication.sharedApplication;
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        gNeuronLabDelegate = [NumiNeuronLabController new];
        application.delegate = gNeuronLabDelegate;
        [application finishLaunching];
        [application run];
    }
    return 0;
}
