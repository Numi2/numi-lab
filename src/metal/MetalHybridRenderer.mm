#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalHybridRenderer.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <system_error>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kMinimumAllocationBytes = 16u;
const char kMetalHybridRendererImageAnchor = 0;

struct RendererBuffers {
    __strong id<MTLBuffer> gaussians = nil;
    __strong id<MTLBuffer> uniforms = nil;
    __strong id<MTLBuffer> projected = nil;
    __strong id<MTLBuffer> tileCounts = nil;
    __strong id<MTLBuffer> tileIndices = nil;
    __strong id<MTLBuffer> tileOverflowCounts = nil;
    __strong id<MTLBuffer> rgb = nil;
    __strong id<MTLBuffer> depth = nil;
    __strong id<MTLBuffer> segmentation = nil;
};

std::string nsString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return std::string{value.UTF8String};
}

std::string describeError(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const std::string description = nsString(error.localizedDescription);
    return description.empty() ? nsString(error.description) : description;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalHybridRendererImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory / "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() / "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }
    const std::filesystem::path configured{METALROBO_DEFAULT_METALLIB};
    return regularFile(configured) ? configured.string() : std::string{};
}

MetalHybridRendererDiagnostics
reject(MetalHybridRendererDiagnostics diagnostics,
       const MetalHybridRendererStatus status, std::string message) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool checkedMultiply(const std::size_t left, const std::size_t right,
                     std::size_t& result) {
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(const std::size_t left, const std::size_t right,
                std::size_t& result) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

template <typename Value>
bool checkedBytes(const std::size_t count, std::size_t& result) {
    return checkedMultiply(count, sizeof(Value), result) &&
           result <= std::numeric_limits<NSUInteger>::max();
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::size_t logicalBytes,
                         const MTLResourceOptions options, NSString* label) {
    const NSUInteger bytes = static_cast<NSUInteger>(
        std::max<std::size_t>(logicalBytes, kMinimumAllocationBytes));
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:options];
    buffer.label = label;
    return buffer;
}

id<MTLBuffer> makePrivateBuffer(id<MTLDevice> device, const std::size_t bytes,
                                NSString* label) {
    return makeBuffer(device, bytes, MTLResourceStorageModePrivate, label);
}

id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device, const std::size_t bytes,
                               NSString* label) {
    return makeBuffer(device, bytes, MTLResourceStorageModeShared, label);
}

bool finite4(const mr_float4& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

} // namespace

namespace detail {

struct MetalHybridRendererState {
    explicit MetalHybridRendererState(MetalHybridRendererConfig configured)
        : config(std::move(configured)) {}

    MetalHybridRendererConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool compiled = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> clearPipeline = nil;
    __strong id<MTLComputePipelineState> binPipeline = nil;
    __strong id<MTLComputePipelineState> renderPipeline = nil;
    RendererBuffers buffers;
    MetalHybridRendererLayout layout;
    std::uint32_t assetCount = 0u;
    std::uint32_t activeEnvironmentCount = 0u;
};

} // namespace detail

namespace {

MetalHybridRendererDiagnostics
initialize(detail::MetalHybridRendererState& state,
           MetalHybridRendererDiagnostics diagnostics) {
    if (state.initialized) {
        diagnostics.deviceName = nsString(state.device.name);
        return diagnostics;
    }
    state.device = MTLCreateSystemDefaultDevice();
    if (state.device == nil) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalDeviceUnavailable,
                      "no Metal device is available");
    }
    state.queue = [state.device newCommandQueue];
    if (state.queue == nil) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalDeviceUnavailable,
                      "could not create the hybrid-render command queue");
    }
    const std::string path = state.config.metallibPath.empty()
                                 ? defaultMetallibPath()
                                 : state.config.metallibPath;
    if (path.empty() || !regularFile(path)) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metallibUnavailable,
                      "MetalRobo.metallib is unavailable");
    }
    NSError* error = nil;
    state.library =
        [state.device newLibraryWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                                  error:&error];
    if (state.library == nil) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalLibraryFailure,
                      "could not load MetalRobo.metallib: " +
                          describeError(error));
    }

    const auto pipeline =
        [&state, &error](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> function = [state.library newFunctionWithName:name];
        if (function == nil) {
            return nil;
        }
        return [state.device newComputePipelineStateWithFunction:function
                                                           error:&error];
    };
    state.clearPipeline = pipeline(@"mr_hybrid_clear_tiles");
    state.binPipeline = pipeline(@"mr_hybrid_bin_gaussians");
    state.renderPipeline = pipeline(@"mr_hybrid_render_tiles");
    if (state.clearPipeline == nil || state.binPipeline == nil ||
        state.renderPipeline == nil) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalPipelineFailure,
                      "could not create hybrid-render pipelines: " +
                          describeError(error));
    }
    if (state.renderPipeline.maxTotalThreadsPerThreadgroup <
        MR_HYBRID_MAX_GAUSSIANS_PER_TILE) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalPipelineFailure,
                      "device cannot dispatch the 16x16 Gaussian tile kernel");
    }
    state.initialized = true;
    diagnostics.deviceName = nsString(state.device.name);
    return diagnostics;
}

} // namespace

bool HybridGaussianScene::valid(std::string* reason) const {
    const auto invalid = [reason](const std::string& message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (id.empty() || assetCount == 0u || gaussians.empty()) {
        return invalid(
            "hybrid Gaussian scene identity, assets, or points are empty");
    }
    for (const MRHybridGaussianGPU& gaussian : gaussians) {
        if (!finite4(gaussian.meanAndOpacity) ||
            !finite4(gaussian.scaleAndImportance) ||
            !finite4(gaussian.orientation) ||
            !finite4(gaussian.colorAndEmission) ||
            gaussian.meanAndOpacity.w < 0.0f ||
            gaussian.meanAndOpacity.w > 1.0f ||
            gaussian.scaleAndImportance.x <= 0.0f ||
            gaussian.scaleAndImportance.y <= 0.0f ||
            gaussian.scaleAndImportance.z <= 0.0f ||
            gaussian.orientation.x * gaussian.orientation.x +
                    gaussian.orientation.y * gaussian.orientation.y +
                    gaussian.orientation.z * gaussian.orientation.z +
                    gaussian.orientation.w * gaussian.orientation.w <=
                1.0e-12f ||
            gaussian.binding.w > MR_HYBRID_GAUSSIAN_WORLD ||
            (gaussian.binding.w == MR_HYBRID_GAUSSIAN_ASSET_LOCAL &&
             gaussian.binding.x >= assetCount) ||
            gaussian.binding.w == MR_HYBRID_GAUSSIAN_BODY_LOCAL) {
            return invalid(
                "hybrid scene contains an invalid or unsupported Gaussian");
        }
    }
    return true;
}

MetalHybridRenderer::MetalHybridRenderer(MetalHybridRendererConfig config)
    : state_(std::make_shared<detail::MetalHybridRendererState>(
          std::move(config))) {}

MetalHybridRenderer::~MetalHybridRenderer() = default;

MetalHybridRenderer::MetalHybridRenderer(MetalHybridRenderer&& other) noexcept =
    default;

MetalHybridRenderer&
MetalHybridRenderer::operator=(MetalHybridRenderer&& other) noexcept = default;

MetalHybridRendererDiagnostics
MetalHybridRenderer::compile(const HybridGaussianScene& scene,
                             const std::uint32_t capacity) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::internalFailure,
                      "hybrid renderer has no state");
    }
    try {
        const std::lock_guard lock(state_->mutex);
        std::string reason;
        if (!scene.valid(&reason)) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::invalidScene,
                          std::move(reason));
        }
        if (capacity == 0u || state_->config.width == 0u ||
            state_->config.height == 0u ||
            state_->config.maximumGaussiansPerTile == 0u ||
            state_->config.maximumGaussiansPerTile >
                MR_HYBRID_MAX_GAUSSIANS_PER_TILE ||
            !finite4(state_->config.clearColorAndDepth) ||
            state_->config.clearColorAndDepth.w <= 0.0f) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::invalidConfiguration,
                          "hybrid renderer dimensions, tile capacity, clear "
                          "values, or world capacity are invalid");
        }
        diagnostics = initialize(*state_, std::move(diagnostics));
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        MetalHybridRendererLayout layout;
        layout.capacity = capacity;
        layout.width = state_->config.width;
        layout.height = state_->config.height;
        layout.tileCountX =
            (layout.width + MR_HYBRID_TILE_SIZE - 1u) / MR_HYBRID_TILE_SIZE;
        layout.tileCountY =
            (layout.height + MR_HYBRID_TILE_SIZE - 1u) / MR_HYBRID_TILE_SIZE;
        if (scene.gaussians.size() >
            std::numeric_limits<std::uint32_t>::max()) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::capacityOverflow,
                          "hybrid scene has too many Gaussians");
        }
        layout.gaussianCount =
            static_cast<std::uint32_t>(scene.gaussians.size());
        layout.maximumGaussiansPerTile = state_->config.maximumGaussiansPerTile;

        std::size_t pixelCount = 0u;
        std::size_t tileCount = 0u;
        std::size_t projectedCount = 0u;
        std::size_t tileIndexCount = 0u;
        if (!checkedMultiply(layout.width, layout.height, pixelCount) ||
            !checkedMultiply(pixelCount, capacity, pixelCount) ||
            !checkedMultiply(layout.tileCountX, layout.tileCountY, tileCount) ||
            !checkedMultiply(tileCount, capacity, tileCount) ||
            !checkedMultiply(scene.gaussians.size(), capacity,
                             projectedCount) ||
            !checkedMultiply(tileCount, layout.maximumGaussiansPerTile,
                             tileIndexCount)) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::capacityOverflow,
                          "hybrid renderer element counts overflow");
        }

        std::size_t gaussianBytes = 0u;
        std::size_t projectedBytes = 0u;
        std::size_t tileCountBytes = 0u;
        std::size_t tileIndexBytes = 0u;
        std::size_t tileOverflowBytes = 0u;
        std::size_t rgbBytes = 0u;
        std::size_t depthBytes = 0u;
        std::size_t segmentationBytes = 0u;
        if (!checkedBytes<MRHybridGaussianGPU>(scene.gaussians.size(),
                                               gaussianBytes) ||
            !checkedBytes<MRHybridProjectedGaussianGPU>(projectedCount,
                                                        projectedBytes) ||
            !checkedBytes<std::uint32_t>(tileCount, tileCountBytes) ||
            !checkedBytes<std::uint32_t>(tileIndexCount, tileIndexBytes) ||
            !checkedBytes<std::uint32_t>(capacity, tileOverflowBytes) ||
            !checkedBytes<mr_float4>(pixelCount, rgbBytes) ||
            !checkedBytes<float>(pixelCount, depthBytes) ||
            !checkedBytes<std::uint32_t>(pixelCount, segmentationBytes)) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::capacityOverflow,
                          "hybrid renderer buffer sizes overflow Metal limits");
        }

        std::size_t retained = 0u;
        for (const std::size_t bytes : {
                 gaussianBytes,
                 projectedBytes,
                 tileCountBytes,
                 tileIndexBytes,
                 tileOverflowBytes,
                 rgbBytes,
                 depthBytes,
                 segmentationBytes,
             }) {
            if (!checkedAdd(retained, bytes, retained)) {
                return reject(std::move(diagnostics),
                              MetalHybridRendererStatus::capacityOverflow,
                              "hybrid renderer retained byte count overflows");
            }
        }
        layout.retainedPrivateBytes = retained;

        RendererBuffers buffers;
        buffers.gaussians = makePrivateBuffer(state_->device, gaussianBytes,
                                              @"MetalRobo hybrid Gaussians");
        buffers.uniforms =
            makeSharedBuffer(state_->device, sizeof(MRHybridRenderUniformsGPU),
                             @"MetalRobo hybrid uniforms");
        buffers.projected = makePrivateBuffer(state_->device, projectedBytes,
                                              @"MetalRobo projected Gaussians");
        buffers.tileCounts = makePrivateBuffer(
            state_->device, tileCountBytes, @"MetalRobo Gaussian tile counts");
        buffers.tileIndices = makePrivateBuffer(
            state_->device, tileIndexBytes, @"MetalRobo Gaussian tile indices");
        buffers.tileOverflowCounts =
            makePrivateBuffer(state_->device, tileOverflowBytes,
                              @"MetalRobo Gaussian tile overflow counts");
        buffers.rgb = makePrivateBuffer(state_->device, rgbBytes,
                                        @"MetalRobo hybrid RGB");
        buffers.depth = makePrivateBuffer(state_->device, depthBytes,
                                          @"MetalRobo hybrid depth");
        buffers.segmentation =
            makePrivateBuffer(state_->device, segmentationBytes,
                              @"MetalRobo hybrid segmentation");
        if (buffers.gaussians == nil || buffers.uniforms == nil ||
            buffers.projected == nil || buffers.tileCounts == nil ||
            buffers.tileIndices == nil || buffers.tileOverflowCounts == nil ||
            buffers.rgb == nil || buffers.depth == nil ||
            buffers.segmentation == nil) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalBufferFailure,
                          "could not allocate hybrid renderer buffers");
        }
        id<MTLBuffer> upload = makeSharedBuffer(
            state_->device, gaussianBytes, @"MetalRobo hybrid Gaussian upload");
        if (upload == nil) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalBufferFailure,
                          "could not allocate Gaussian upload buffer");
        }
        std::memcpy(upload.contents, scene.gaussians.data(), gaussianBytes);
        id<MTLCommandBuffer> command = [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalCommandFailure,
                          "could not create Gaussian upload command");
        }
        [blit copyFromBuffer:upload
                 sourceOffset:0u
                     toBuffer:buffers.gaussians
            destinationOffset:0u
                         size:gaussianBytes];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalCommandFailure,
                          "Gaussian upload failed: " +
                              describeError(command.error));
        }

        state_->buffers = std::move(buffers);
        state_->layout = layout;
        state_->assetCount = scene.assetCount;
        state_->activeEnvironmentCount = 0u;
        state_->compiled = true;
        diagnostics.layout = layout;
        diagnostics.deviceName = nsString(state_->device.name);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalBufferFailure,
                      "host allocation failed while compiling hybrid renderer");
    } catch (const std::exception& error) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::internalFailure, error.what());
    }
}

MetalHybridRendererDiagnostics
MetalHybridRenderer::render(const MetalWorldFamilyContext& worlds,
                            const std::uint32_t environmentCount,
                            const std::uint32_t cameraIndex) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::internalFailure,
                      "hybrid renderer has no state");
    }
    try {
        const std::lock_guard lock(state_->mutex);
        diagnostics.layout = state_->layout;
        diagnostics.deviceName = nsString(state_->device.name);
        if (!state_->compiled) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::notCompiled,
                          "compile the hybrid renderer before rendering");
        }
        const MetalWorldFamilyLayout worldLayout = worlds.layout();
        if (environmentCount == 0u ||
            environmentCount > state_->layout.capacity ||
            environmentCount > worldLayout.activeInstanceCount ||
            worldLayout.assetCountPerInstance < state_->assetCount ||
            cameraIndex >= worldLayout.sensorCountPerInstance) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::incompatibleWorldFamily,
                          "sampled world count, assets, or camera are "
                          "incompatible with the hybrid scene");
        }
        id<MTLBuffer> instances = (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::instanceHeaders);
        id<MTLBuffer> assets = (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::assetInstances);
        id<MTLBuffer> sensors = (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::sensorInstances);
        id<MTLBuffer> appearances = (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::appearanceInstances);
        if (instances == nil || assets == nil || sensors == nil ||
            appearances == nil || instances.device != state_->device ||
            assets.device != state_->device ||
            sensors.device != state_->device ||
            appearances.device != state_->device) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::incompatibleWorldFamily,
                          "world-family Metal buffers are unavailable or on "
                          "a different device");
        }

        MRHybridRenderUniformsGPU uniforms{};
        uniforms.counts = {
            environmentCount,
            state_->layout.gaussianCount,
            worldLayout.assetCountPerInstance,
            worldLayout.sensorCountPerInstance,
        };
        uniforms.image = {
            state_->layout.width,
            state_->layout.height,
            state_->layout.tileCountX,
            state_->layout.tileCountY,
        };
        uniforms.render = {
            cameraIndex,
            state_->layout.maximumGaussiansPerTile,
            MR_HYBRID_TILE_SIZE,
            MR_HYBRID_RENDERER_ABI_VERSION,
        };
        uniforms.clearColorAndDepth = state_->config.clearColorAndDepth;
        std::memcpy(state_->buffers.uniforms.contents, &uniforms,
                    sizeof(uniforms));
        const NSUInteger tileCount = static_cast<NSUInteger>(environmentCount) *
                                     state_->layout.tileCountX *
                                     state_->layout.tileCountY;
        const NSUInteger projectedCount =
            static_cast<NSUInteger>(environmentCount) *
            state_->layout.gaussianCount;
        const auto start = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command = [state_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalCommandFailure,
                          "could not create hybrid render command");
        }
        encoder.label = @"MetalRobo hybrid observations";
        [encoder setComputePipelineState:state_->clearPipeline];
        [encoder setBuffer:state_->buffers.tileCounts offset:0u atIndex:0u];
        [encoder setBuffer:state_->buffers.uniforms offset:0u atIndex:2u];
        [encoder setBuffer:state_->buffers.tileOverflowCounts
                    offset:0u
                   atIndex:1u];
        const NSUInteger clearThreads = std::min<NSUInteger>(
            state_->clearPipeline.maxTotalThreadsPerThreadgroup, 256u);
        [encoder dispatchThreads:MTLSizeMake(tileCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(clearThreads, 1u, 1u)];

        [encoder setComputePipelineState:state_->binPipeline];
        [encoder setBuffer:state_->buffers.gaussians offset:0u atIndex:0u];
        [encoder setBuffer:instances offset:0u atIndex:1u];
        [encoder setBuffer:assets offset:0u atIndex:2u];
        [encoder setBuffer:sensors offset:0u atIndex:3u];
        [encoder setBuffer:state_->buffers.projected offset:0u atIndex:4u];
        [encoder setBuffer:state_->buffers.tileCounts offset:0u atIndex:5u];
        [encoder setBuffer:state_->buffers.tileIndices offset:0u atIndex:6u];
        [encoder setBuffer:state_->buffers.uniforms offset:0u atIndex:8u];
        [encoder setBuffer:state_->buffers.tileOverflowCounts
                    offset:0u
                   atIndex:7u];
        const NSUInteger binThreads = std::min<NSUInteger>(
            state_->binPipeline.maxTotalThreadsPerThreadgroup, 256u);
        [encoder dispatchThreads:MTLSizeMake(projectedCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(binThreads, 1u, 1u)];

        [encoder setComputePipelineState:state_->renderPipeline];
        [encoder setBuffer:state_->buffers.projected offset:0u atIndex:0u];
        [encoder setBuffer:state_->buffers.tileCounts offset:0u atIndex:1u];
        [encoder setBuffer:state_->buffers.tileIndices offset:0u atIndex:2u];
        [encoder setBuffer:instances offset:0u atIndex:3u];
        [encoder setBuffer:sensors offset:0u atIndex:4u];
        [encoder setBuffer:appearances offset:0u atIndex:5u];
        [encoder setBuffer:state_->buffers.rgb offset:0u atIndex:6u];
        [encoder setBuffer:state_->buffers.depth offset:0u atIndex:7u];
        [encoder setBuffer:state_->buffers.segmentation offset:0u atIndex:8u];
        [encoder setBuffer:state_->buffers.uniforms offset:0u atIndex:9u];
        [encoder
             dispatchThreadgroups:MTLSizeMake(tileCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(MR_HYBRID_MAX_GAUSSIANS_PER_TILE,
                                              1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(end - start).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalCommandFailure,
                          "hybrid render failed: " +
                              describeError(command.error));
        }
        state_->activeEnvironmentCount = environmentCount;
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::internalFailure, error.what());
    }
}

MetalHybridRendererDiagnostics
MetalHybridRenderer::readback(HybridObservationBatch& output) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::internalFailure,
                      "hybrid renderer has no state");
    }
    try {
        const std::lock_guard lock(state_->mutex);
        diagnostics.layout = state_->layout;
        diagnostics.deviceName = nsString(state_->device.name);
        if (!state_->compiled || state_->activeEnvironmentCount == 0u) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::notCompiled,
                          "render observations before reading them back");
        }
        std::size_t pixelCount = 0u;
        checkedMultiply(state_->layout.width, state_->layout.height,
                        pixelCount);
        checkedMultiply(pixelCount, state_->activeEnvironmentCount, pixelCount);
        HybridObservationBatch candidate;
        candidate.environmentCount = state_->activeEnvironmentCount;
        candidate.width = state_->layout.width;
        candidate.height = state_->layout.height;
        candidate.rgb.resize(pixelCount);
        candidate.depth.resize(pixelCount);
        candidate.segmentation.resize(pixelCount);

        const std::size_t rgbBytes = pixelCount * sizeof(mr_float4);
        const std::size_t depthBytes = pixelCount * sizeof(float);
        const std::size_t segmentationBytes =
            pixelCount * sizeof(std::uint32_t);
        id<MTLBuffer> rgb = makeSharedBuffer(state_->device, rgbBytes,
                                             @"MetalRobo hybrid RGB readback");
        id<MTLBuffer> depth = makeSharedBuffer(
            state_->device, depthBytes, @"MetalRobo hybrid depth readback");
        id<MTLBuffer> segmentation =
            makeSharedBuffer(state_->device, segmentationBytes,
                             @"MetalRobo hybrid segmentation readback");
        if (rgb == nil || depth == nil || segmentation == nil) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalBufferFailure,
                          "could not allocate hybrid observation readback");
        }
        id<MTLCommandBuffer> command = [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create hybrid observation readback command");
        }
        [blit copyFromBuffer:state_->buffers.rgb
                 sourceOffset:0u
                     toBuffer:rgb
            destinationOffset:0u
                         size:rgbBytes];
        [blit copyFromBuffer:state_->buffers.depth
                 sourceOffset:0u
                     toBuffer:depth
            destinationOffset:0u
                         size:depthBytes];
        [blit copyFromBuffer:state_->buffers.segmentation
                 sourceOffset:0u
                     toBuffer:segmentation
            destinationOffset:0u
                         size:segmentationBytes];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(std::move(diagnostics),
                          MetalHybridRendererStatus::metalCommandFailure,
                          "hybrid observation readback failed: " +
                              describeError(command.error));
        }
        std::memcpy(candidate.rgb.data(), rgb.contents, rgbBytes);
        std::memcpy(candidate.depth.data(), depth.contents, depthBytes);
        std::memcpy(candidate.segmentation.data(), segmentation.contents,
                    segmentationBytes);
        output = std::move(candidate);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::metalBufferFailure,
                      "host allocation failed during hybrid readback");
    } catch (const std::exception& error) {
        return reject(std::move(diagnostics),
                      MetalHybridRendererStatus::internalFailure, error.what());
    }
}

MetalHybridRendererLayout MetalHybridRenderer::layout() const noexcept {
    if (state_ == nullptr) {
        return {};
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->layout;
    } catch (...) {
        return {};
    }
}

void* MetalHybridRenderer::nativeBuffer(
    const MetalHybridRendererBuffer buffer) const noexcept {
    if (state_ == nullptr) {
        return nullptr;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!state_->compiled) {
            return nullptr;
        }
        id<MTLBuffer> selected = nil;
        switch (buffer) {
        case MetalHybridRendererBuffer::rgb:
            selected = state_->buffers.rgb;
            break;
        case MetalHybridRendererBuffer::depth:
            selected = state_->buffers.depth;
            break;
        case MetalHybridRendererBuffer::segmentation:
            selected = state_->buffers.segmentation;
            break;
        case MetalHybridRendererBuffer::projectedGaussians:
            selected = state_->buffers.projected;
            break;
        case MetalHybridRendererBuffer::tileOverflowCounts:
            selected = state_->buffers.tileOverflowCounts;
            break;
        }
        return (__bridge void*)selected;
    } catch (...) {
        return nullptr;
    }
}

const char*
metalHybridRendererStatusName(const MetalHybridRendererStatus status) noexcept {
    switch (status) {
    case MetalHybridRendererStatus::success:
        return "success";
    case MetalHybridRendererStatus::invalidScene:
        return "invalid_scene";
    case MetalHybridRendererStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalHybridRendererStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalHybridRendererStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalHybridRendererStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalHybridRendererStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalHybridRendererStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalHybridRendererStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalHybridRendererStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalHybridRendererStatus::incompatibleWorldFamily:
        return "incompatible_world_family";
    case MetalHybridRendererStatus::notCompiled:
        return "not_compiled";
    case MetalHybridRendererStatus::internalFailure:
        return "internal_failure";
    }
}

} // namespace metalrobo
