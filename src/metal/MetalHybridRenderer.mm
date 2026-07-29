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
#include <ranges>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kMinimumAllocationBytes = 16u;
constexpr std::uint32_t kLiveCurrent = 1u << 0u;
constexpr std::uint32_t kLivePrevious = 1u << 1u;
const char kMetalHybridRendererImageAnchor = 0;

struct RendererBuffers {
    __strong id<MTLBuffer> gaussians = nil;
    __strong id<MTLBuffer> meshVertices = nil;
    __strong id<MTLBuffer> meshTriangles = nil;
    __strong id<MTLBuffer> materials = nil;
    __strong id<MTLBuffer> sensorBindings = nil;
    __strong id<MTLBuffer> currentBodies = nil;
    __strong id<MTLBuffer> previousBodies = nil;
    __strong id<MTLBuffer> projected = nil;
    __strong id<MTLBuffer> tileCounts = nil;
    __strong id<MTLBuffer> tileIndices = nil;
    __strong id<MTLBuffer> tileOverflowCounts = nil;
    __strong id<MTLBuffer> meshWinners = nil;
    __strong id<MTLBuffer> rgb = nil;
    __strong id<MTLBuffer> depth = nil;
    __strong id<MTLBuffer> segmentation = nil;
    __strong id<MTLBuffer> identities = nil;
    __strong id<MTLBuffer> normals = nil;
    __strong id<MTLBuffer> motion = nil;
    __strong id<MTLBuffer> validity = nil;
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
    const std::string description =
        nsString(error.localizedDescription);
    return description.empty()
        ? nsString(error.description)
        : description;
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
            libraryDirectory.parent_path() /
                "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }
    const std::filesystem::path configured{
        METALROBO_DEFAULT_METALLIB
    };
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

MetalHybridRendererDiagnostics reject(
    MetalHybridRendererDiagnostics diagnostics,
    const MetalHybridRendererStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

template <typename Value>
bool checkedBytes(
    const std::size_t count,
    std::size_t& result
) {
    return checkedMultiply(count, sizeof(Value), result) &&
        result <= std::numeric_limits<NSUInteger>::max();
}

id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::size_t logicalBytes,
    const MTLResourceOptions options,
    NSString* label
) {
    const NSUInteger bytes = static_cast<NSUInteger>(
        std::max<std::size_t>(
            logicalBytes,
            kMinimumAllocationBytes
        )
    );
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:options];
    buffer.label = label;
    return buffer;
}

id<MTLBuffer> makePrivateBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    return makeBuffer(
        device,
        bytes,
        MTLResourceStorageModePrivate,
        label
    );
}

id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    return makeBuffer(
        device,
        bytes,
        MTLResourceStorageModeShared,
        label
    );
}

bool finite4(const mr_float4& value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool validMaterial(const MRVisualMaterialGPU& material) {
    return finite4(material.baseColorAndOpacity) &&
        finite4(material.emissionAndStrength) &&
        finite4(material.surface) &&
        finite4(material.coating) &&
        material.baseColorAndOpacity.w >= 0.0f &&
        material.baseColorAndOpacity.w <= 1.0f &&
        material.surface.x >= 0.0f &&
        material.surface.x <= 1.0f &&
        material.surface.y >= 0.0f &&
        material.surface.y <= 1.0f;
}

} // namespace

namespace detail {

struct MetalHybridRendererState {
    explicit MetalHybridRendererState(
        MetalHybridRendererConfig configured
    )
        : config(std::move(configured)) {}

    MetalHybridRendererConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool compiled = false;
    bool requiresLiveState = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> clearPipeline = nil;
    __strong id<MTLComputePipelineState> clearMeshPipeline = nil;
    __strong id<MTLComputePipelineState> binPipeline = nil;
    __strong id<MTLComputePipelineState> renderPipeline = nil;
    __strong id<MTLComputePipelineState> rasterMeshPipeline = nil;
    __strong id<MTLComputePipelineState> selectMeshPipeline = nil;
    __strong id<MTLComputePipelineState> compositeMeshPipeline = nil;
    RendererBuffers buffers;
    MetalHybridRendererLayout layout;
    MRVisualFrameMetadataGPU activeMetadata{};
    std::vector<MRVisualSensorBindingGPU> sensorProfiles;
    std::uint32_t assetCount = 0u;
    std::uint32_t activeEnvironmentCount = 0u;
};

} // namespace detail

namespace {

MetalHybridRendererDiagnostics initialize(
    detail::MetalHybridRendererState& state,
    MetalHybridRendererDiagnostics diagnostics
) {
    if (state.initialized) {
        diagnostics.deviceName = nsString(state.device.name);
        return diagnostics;
    }
    state.device = MTLCreateSystemDefaultDevice();
    if (state.device == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalDeviceUnavailable,
            "no Metal device is available"
        );
    }
    state.queue = [state.device newCommandQueue];
    if (state.queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalDeviceUnavailable,
            "could not create the hybrid-render command queue"
        );
    }
    const std::string path = state.config.metallibPath.empty()
        ? defaultMetallibPath()
        : state.config.metallibPath;
    if (path.empty() || !regularFile(path)) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metallibUnavailable,
            "MetalRobo.metallib is unavailable"
        );
    }
    NSError* error = nil;
    state.library = [state.device
        newLibraryWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                    error:&error];
    if (state.library == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalLibraryFailure,
            "could not load MetalRobo.metallib: " +
                describeError(error)
        );
    }

    const auto pipeline =
        [&state, &error](
            NSString* name
        ) -> id<MTLComputePipelineState> {
        id<MTLFunction> function =
            [state.library newFunctionWithName:name];
        if (function == nil) {
            return nil;
        }
        return [state.device
            newComputePipelineStateWithFunction:function
                                          error:&error];
    };
    state.clearPipeline =
        pipeline(@"mr_hybrid_clear_tiles");
    state.clearMeshPipeline =
        pipeline(@"mr_hybrid_clear_mesh_winners");
    state.binPipeline =
        pipeline(@"mr_hybrid_bin_gaussians");
    state.renderPipeline =
        pipeline(@"mr_hybrid_render_tiles");
    state.rasterMeshPipeline =
        pipeline(@"mr_hybrid_rasterize_mesh");
    state.selectMeshPipeline =
        pipeline(@"mr_hybrid_select_mesh");
    state.compositeMeshPipeline =
        pipeline(@"mr_hybrid_composite_mesh");
    if (state.clearPipeline == nil ||
        state.clearMeshPipeline == nil ||
        state.binPipeline == nil ||
        state.renderPipeline == nil ||
        state.rasterMeshPipeline == nil ||
        state.selectMeshPipeline == nil ||
        state.compositeMeshPipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalPipelineFailure,
            "could not create visual sensor pipelines: " +
                describeError(error)
        );
    }
    if (state.renderPipeline.maxTotalThreadsPerThreadgroup <
        MR_HYBRID_MAX_GAUSSIANS_PER_TILE) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalPipelineFailure,
            "device cannot dispatch the 16x16 Gaussian tile kernel"
        );
    }
    state.initialized = true;
    diagnostics.deviceName = nsString(state.device.name);
    return diagnostics;
}

bool upload(
    id<MTLDevice> device,
    id<MTLBlitCommandEncoder> blit,
    const void* source,
    const std::size_t bytes,
    id<MTLBuffer> destination,
    NSString* label,
    std::vector<id<MTLBuffer>>& staging
) {
    if (bytes == 0u) {
        return true;
    }
    id<MTLBuffer> buffer =
        makeSharedBuffer(device, bytes, label);
    if (buffer == nil) {
        return false;
    }
    std::memcpy(buffer.contents, source, bytes);
    [blit copyFromBuffer:buffer
            sourceOffset:0u
                toBuffer:destination
       destinationOffset:0u
                    size:bytes];
    staging.push_back(buffer);
    return true;
}

MetalHybridRendererDiagnostics encodeLocked(
    detail::MetalHybridRendererState& state,
    const MetalWorldFamilyContext& worlds,
    const HybridDeviceStateBatch& liveState,
    const std::uint32_t cameraIndex,
    id<MTLComputeCommandEncoder> encoder
) {
    MetalHybridRendererDiagnostics diagnostics;
    diagnostics.layout = state.layout;
    diagnostics.deviceName = nsString(state.device.name);
    if (!state.compiled) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::notCompiled,
            "compile the visual sensor runtime before rendering"
        );
    }
    if (encoder == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "visual sensor encoding requires a Metal compute encoder"
        );
    }
    const MetalWorldFamilyLayout worldLayout = worlds.layout();
    const std::uint32_t environmentCount =
        liveState.environmentCount;
    if (environmentCount == 0u ||
        environmentCount > state.layout.capacity ||
        environmentCount > worldLayout.activeInstanceCount ||
        worldLayout.assetCountPerInstance < state.assetCount ||
        cameraIndex >= worldLayout.sensorCountPerInstance ||
        (state.layout.sensorBindingCount != 0u &&
         cameraIndex >= state.layout.sensorBindingCount)) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::incompatibleWorldFamily,
            "sampled world count, assets, sensor bindings, or camera "
            "are incompatible with the visual scene"
        );
    }

    id<MTLBuffer> instances =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::instanceHeaders
        );
    id<MTLBuffer> assets =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::assetInstances
        );
    id<MTLBuffer> sensors =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::sensorInstances
        );
    id<MTLBuffer> appearances =
        (__bridge id<MTLBuffer>)worlds.nativeBuffer(
            MetalWorldFamilyBuffer::appearanceInstances
        );
    if (instances == nil || assets == nil || sensors == nil ||
        appearances == nil ||
        instances.device != state.device ||
        assets.device != state.device ||
        sensors.device != state.device ||
        appearances.device != state.device) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::incompatibleWorldFamily,
            "world-family Metal buffers are unavailable or on a "
            "different device"
        );
    }

    id<MTLBuffer> currentBodies =
        liveState.currentBodyStates == nullptr
        ? state.buffers.currentBodies
        : (__bridge id<MTLBuffer>)liveState.currentBodyStates;
    id<MTLBuffer> previousBodies =
        liveState.previousBodyStates == nullptr
        ? currentBodies
        : (__bridge id<MTLBuffer>)liveState.previousBodyStates;
    std::uint32_t liveFlags = 0u;
    if (liveState.currentBodyStates != nullptr) {
        liveFlags |= kLiveCurrent;
    }
    if (liveState.previousBodyStates != nullptr) {
        liveFlags |= kLivePrevious;
    }
    if (state.requiresLiveState &&
        (liveFlags & kLiveCurrent) == 0u) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "body/link-bound visual geometry requires live body state"
        );
    }
    if ((liveFlags & kLiveCurrent) != 0u &&
        liveState.bodyCount != state.layout.bodyCount) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "live visual body count does not match the compiled scene"
        );
    }
    std::size_t liveBodyCount = 0u;
    std::size_t liveBodyBytes = 0u;
    if (!checkedMultiply(
            environmentCount,
            state.layout.bodyCount,
            liveBodyCount
        ) ||
        !checkedBytes<MRBodyStateGPU>(
            liveBodyCount,
            liveBodyBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::capacityOverflow,
            "live visual body-state size overflows"
        );
    }
    if (currentBodies == nil || previousBodies == nil ||
        currentBodies.device != state.device ||
        previousBodies.device != state.device ||
        ((liveFlags & kLiveCurrent) != 0u &&
         (currentBodies.length < liveBodyBytes ||
          previousBodies.length < liveBodyBytes))) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "live visual body buffers are unavailable, undersized, or "
            "on another device"
        );
    }

    MRHybridRenderUniformsGPU uniforms{};
    uniforms.counts = {
        environmentCount,
        state.layout.gaussianCount,
        worldLayout.assetCountPerInstance,
        worldLayout.sensorCountPerInstance,
    };
    uniforms.image = {
        state.layout.width,
        state.layout.height,
        state.layout.tileCountX,
        state.layout.tileCountY,
    };
    uniforms.render = {
        cameraIndex,
        state.layout.maximumGaussiansPerTile,
        MR_HYBRID_TILE_SIZE,
        MR_HYBRID_RENDERER_ABI_VERSION,
    };
    uniforms.live = {
        state.layout.bodyCount,
        liveFlags,
        state.layout.sensorBindingCount,
        state.layout.meshTriangleCount,
    };
    uniforms.timing = {
        static_cast<std::uint32_t>(liveState.frameIndex),
        static_cast<std::uint32_t>(
            liveState.frameIndex >> 32u
        ),
        liveState.sensorSequence,
        static_cast<std::uint32_t>(liveState.source),
    };
    uniforms.clearColorAndDepth =
        state.config.clearColorAndDepth;
    const MRVisualSensorBindingGPU profile =
        cameraIndex < state.sensorProfiles.size()
        ? state.sensorProfiles[cameraIndex]
        : MRVisualSensorBindingGPU{
              {},
              {15.0f, 1.0f / 120.0f, 0.0f, 0.0f},
              {0.05f, 10.0f, 0.001f, 0.0f},
          };
    uniforms.sensorTiming = profile.timing;
    uniforms.sensorRangeAndResponse =
        profile.rangeAndResponse;

    const NSUInteger tileCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.tileCountX *
        state.layout.tileCountY;
    const NSUInteger pixelCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.width * state.layout.height;
    const NSUInteger projectedCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.gaussianCount;
    const NSUInteger triangleCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.meshTriangleCount;

    encoder.label = @"MetalRobo visual sensor runtime";
    [encoder setComputePipelineState:state.clearPipeline];
    [encoder setBuffer:state.buffers.tileCounts
                offset:0u
               atIndex:0u];
    [encoder setBuffer:state.buffers.tileOverflowCounts
                offset:0u
               atIndex:1u];
    [encoder setBytes:&uniforms
               length:sizeof(uniforms)
              atIndex:2u];
    const NSUInteger clearThreads = std::min<NSUInteger>(
        state.clearPipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    [encoder dispatchThreads:MTLSizeMake(
                                 std::max<NSUInteger>(
                                     tileCount,
                                     environmentCount
                                 ),
                                 1u,
                                 1u
                             )
        threadsPerThreadgroup:MTLSizeMake(
                                  clearThreads,
                                  1u,
                                  1u
                              )];

    [encoder setComputePipelineState:state.clearMeshPipeline];
    [encoder setBuffer:state.buffers.meshWinners
                offset:0u
               atIndex:0u];
    [encoder setBytes:&uniforms
               length:sizeof(uniforms)
              atIndex:1u];
    const NSUInteger clearMeshThreads = std::min<NSUInteger>(
        state.clearMeshPipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    [encoder dispatchThreads:MTLSizeMake(pixelCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
                                  clearMeshThreads,
                                  1u,
                                  1u
                              )];

    if (projectedCount != 0u) {
        [encoder setComputePipelineState:state.binPipeline];
        [encoder setBuffer:state.buffers.gaussians
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:instances offset:0u atIndex:1u];
        [encoder setBuffer:assets offset:0u atIndex:2u];
        [encoder setBuffer:sensors offset:0u atIndex:3u];
        [encoder setBuffer:state.buffers.sensorBindings
                    offset:0u
                   atIndex:4u];
        [encoder setBuffer:currentBodies offset:0u atIndex:5u];
        [encoder setBuffer:previousBodies offset:0u atIndex:6u];
        [encoder setBuffer:state.buffers.projected
                    offset:0u
                   atIndex:7u];
        [encoder setBuffer:state.buffers.tileCounts
                    offset:0u
                   atIndex:8u];
        [encoder setBuffer:state.buffers.tileIndices
                    offset:0u
                   atIndex:9u];
        [encoder setBuffer:state.buffers.tileOverflowCounts
                    offset:0u
                   atIndex:10u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:11u];
        const NSUInteger binThreads = std::min<NSUInteger>(
            state.binPipeline.maxTotalThreadsPerThreadgroup,
            256u
        );
        [encoder dispatchThreads:MTLSizeMake(
                                     projectedCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      binThreads,
                                      1u,
                                      1u
                                  )];
    }

    [encoder setComputePipelineState:state.renderPipeline];
    [encoder setBuffer:state.buffers.projected
                offset:0u
               atIndex:0u];
    [encoder setBuffer:state.buffers.tileCounts
                offset:0u
               atIndex:1u];
    [encoder setBuffer:state.buffers.tileIndices
                offset:0u
               atIndex:2u];
    [encoder setBuffer:instances offset:0u atIndex:3u];
    [encoder setBuffer:sensors offset:0u atIndex:4u];
    [encoder setBuffer:appearances offset:0u atIndex:5u];
    [encoder setBuffer:state.buffers.rgb
                offset:0u
               atIndex:6u];
    [encoder setBuffer:state.buffers.depth
                offset:0u
               atIndex:7u];
    [encoder setBuffer:state.buffers.segmentation
                offset:0u
               atIndex:8u];
    [encoder setBuffer:state.buffers.identities
                offset:0u
               atIndex:9u];
    [encoder setBuffer:state.buffers.normals
                offset:0u
               atIndex:10u];
    [encoder setBuffer:state.buffers.motion
                offset:0u
               atIndex:11u];
    [encoder setBuffer:state.buffers.validity
                offset:0u
               atIndex:12u];
    [encoder setBytes:&uniforms
               length:sizeof(uniforms)
              atIndex:13u];
    [encoder
         dispatchThreadgroups:MTLSizeMake(tileCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
                                  MR_HYBRID_MAX_GAUSSIANS_PER_TILE,
                                  1u,
                                  1u
                              )];

    if (triangleCount != 0u) {
        [encoder setComputePipelineState:state.rasterMeshPipeline];
        [encoder setBuffer:state.buffers.meshVertices
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:state.buffers.meshTriangles
                    offset:0u
                   atIndex:1u];
        [encoder setBuffer:instances offset:0u atIndex:2u];
        [encoder setBuffer:assets offset:0u atIndex:3u];
        [encoder setBuffer:sensors offset:0u atIndex:4u];
        [encoder setBuffer:state.buffers.sensorBindings
                    offset:0u
                   atIndex:5u];
        [encoder setBuffer:currentBodies offset:0u atIndex:6u];
        [encoder setBuffer:state.buffers.meshWinners
                    offset:0u
                   atIndex:7u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:8u];
        const NSUInteger rasterThreads = std::min<NSUInteger>(
            state.rasterMeshPipeline
                .maxTotalThreadsPerThreadgroup,
            128u
        );
        [encoder dispatchThreads:MTLSizeMake(
                                     triangleCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      rasterThreads,
                                      1u,
                                      1u
                                  )];

        [encoder setComputePipelineState:state.selectMeshPipeline];
        const NSUInteger selectThreads = std::min<NSUInteger>(
            state.selectMeshPipeline
                .maxTotalThreadsPerThreadgroup,
            128u
        );
        [encoder dispatchThreads:MTLSizeMake(
                                     triangleCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      selectThreads,
                                      1u,
                                      1u
                                  )];

        [encoder
            setComputePipelineState:state.compositeMeshPipeline];
        [encoder setBuffer:state.buffers.meshVertices
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:state.buffers.meshTriangles
                    offset:0u
                   atIndex:1u];
        [encoder setBuffer:state.buffers.materials
                    offset:0u
                   atIndex:2u];
        [encoder setBuffer:instances offset:0u atIndex:3u];
        [encoder setBuffer:assets offset:0u atIndex:4u];
        [encoder setBuffer:sensors offset:0u atIndex:5u];
        [encoder setBuffer:appearances offset:0u atIndex:6u];
        [encoder setBuffer:state.buffers.sensorBindings
                    offset:0u
                   atIndex:7u];
        [encoder setBuffer:currentBodies offset:0u atIndex:8u];
        [encoder setBuffer:previousBodies offset:0u atIndex:9u];
        [encoder setBuffer:state.buffers.meshWinners
                    offset:0u
                   atIndex:10u];
        [encoder setBuffer:state.buffers.rgb
                    offset:0u
                   atIndex:11u];
        [encoder setBuffer:state.buffers.depth
                    offset:0u
                   atIndex:12u];
        [encoder setBuffer:state.buffers.segmentation
                    offset:0u
                   atIndex:13u];
        [encoder setBuffer:state.buffers.identities
                    offset:0u
                   atIndex:14u];
        [encoder setBuffer:state.buffers.normals
                    offset:0u
                   atIndex:15u];
        [encoder setBuffer:state.buffers.motion
                    offset:0u
                   atIndex:16u];
        [encoder setBuffer:state.buffers.validity
                    offset:0u
                   atIndex:17u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:18u];
        const NSUInteger compositeThreads =
            std::min<NSUInteger>(
                state.compositeMeshPipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        [encoder dispatchThreads:MTLSizeMake(
                                     pixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      compositeThreads,
                                      1u,
                                      1u
                                  )];
    }

    state.activeEnvironmentCount = environmentCount;
    state.activeMetadata.dimensions = {
        environmentCount,
        1u,
        state.layout.width,
        state.layout.height,
    };
    state.activeMetadata.identity = uniforms.timing;
    state.activeMetadata.timing = {
        static_cast<float>(
            liveState.captureTimestampSeconds
        ),
        static_cast<float>(liveState.frameAgeSeconds),
        profile.timing.y,
        profile.timing.z,
    };
    state.activeMetadata.contract = {
        MR_VISUAL_MODALITY_RGB |
            MR_VISUAL_MODALITY_DEPTH |
            MR_VISUAL_MODALITY_DEPTH_VALIDITY |
            MR_VISUAL_MODALITY_NORMAL |
            MR_VISUAL_MODALITY_MOTION |
            MR_VISUAL_MODALITY_SEMANTIC |
            MR_VISUAL_MODALITY_INSTANCE |
            MR_VISUAL_MODALITY_LINK,
        MR_VISUAL_FRAME_CAMERA,
        MR_VISUAL_PLATFORM_ABI_VERSION,
        0u,
    };
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
    if (id.empty() || assetCount == 0u ||
        (gaussians.empty() && meshTriangles.empty())) {
        return invalid(
            "visual scene identity, assets, and geometry are incomplete"
        );
    }
    if ((!meshVertices.empty() || !meshTriangles.empty()) &&
        (meshVertices.empty() || meshTriangles.empty() ||
         materials.empty())) {
        return invalid(
            "mesh visual scene requires vertices, triangles, and materials"
        );
    }
    for (const MRHybridGaussianGPU& gaussian : gaussians) {
        const double orientationSquared =
            static_cast<double>(gaussian.orientation.x) *
                gaussian.orientation.x +
            static_cast<double>(gaussian.orientation.y) *
                gaussian.orientation.y +
            static_cast<double>(gaussian.orientation.z) *
                gaussian.orientation.z +
            static_cast<double>(gaussian.orientation.w) *
                gaussian.orientation.w;
        if (!finite4(gaussian.meanAndOpacity) ||
            !finite4(gaussian.scaleAndImportance) ||
            !finite4(gaussian.orientation) ||
            !finite4(gaussian.colorAndEmission) ||
            gaussian.meanAndOpacity.w < 0.0f ||
            gaussian.meanAndOpacity.w > 1.0f ||
            gaussian.scaleAndImportance.x <= 0.0f ||
            gaussian.scaleAndImportance.y <= 0.0f ||
            gaussian.scaleAndImportance.z <= 0.0f ||
            orientationSquared <= 1.0e-12 ||
            gaussian.binding.w > MR_HYBRID_GAUSSIAN_WORLD ||
            gaussian.binding.x >= assetCount ||
            (gaussian.binding.w ==
                 MR_HYBRID_GAUSSIAN_BODY_LOCAL &&
             (bodyCount == 0u ||
              gaussian.binding.y >= bodyCount))) {
            return invalid(
                "visual scene contains an invalid Gaussian"
            );
        }
    }
    for (const MRVisualMeshVertexGPU& vertex : meshVertices) {
        if (!finite4(vertex.position) ||
            !finite4(vertex.normalAndU) ||
            !finite4(vertex.tangentAndV) ||
            vertex.position.w != 1.0f) {
            return invalid(
                "visual scene contains an invalid mesh vertex"
            );
        }
    }
    for (const MRVisualMeshTriangleGPU& triangle :
         meshTriangles) {
        if (triangle.verticesAndMaterial.x >= meshVertices.size() ||
            triangle.verticesAndMaterial.y >= meshVertices.size() ||
            triangle.verticesAndMaterial.z >= meshVertices.size() ||
            triangle.verticesAndMaterial.w >= materials.size() ||
            triangle.binding.x >= assetCount ||
            triangle.binding.z >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            ((triangle.binding.z ==
                  MR_VISUAL_BINDING_RIGID_BODY ||
              triangle.binding.z ==
                  MR_VISUAL_BINDING_ARTICULATED_LINK) &&
             triangle.binding.y >= bodyCount) ||
            triangle.identity.x == 0u ||
            triangle.identity.y == 0u ||
            !finite4(triangle.colorAndOpacity)) {
            return invalid(
                "visual scene contains an invalid mesh triangle"
            );
        }
    }
    if (!std::ranges::all_of(materials, validMaterial)) {
        return invalid("visual scene contains an invalid material");
    }
    for (const MRVisualSensorBindingGPU& binding :
         sensorBindings) {
        const bool assetBinding =
            binding.identity.x == MR_VISUAL_BINDING_ASSET;
        const bool bodyBinding =
            binding.identity.x == MR_VISUAL_BINDING_RIGID_BODY ||
            binding.identity.x ==
                MR_VISUAL_BINDING_ARTICULATED_LINK;
        if (binding.identity.x >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            (assetBinding && binding.identity.z >= assetCount) ||
            (bodyBinding && binding.identity.y >= bodyCount) ||
            !finite4(binding.timing) ||
            !finite4(binding.rangeAndResponse) ||
            binding.timing.x <= 0.0f ||
            binding.rangeAndResponse.x < 0.0f ||
            binding.rangeAndResponse.y <=
                binding.rangeAndResponse.x) {
            return invalid(
                "visual scene contains an invalid sensor binding"
            );
        }
    }
    return true;
}

MetalHybridRenderer::MetalHybridRenderer(
    MetalHybridRendererConfig config
)
    : state_(
          std::make_shared<detail::MetalHybridRendererState>(
              std::move(config)
          )
      ) {}

MetalHybridRenderer::~MetalHybridRenderer() = default;

MetalHybridRenderer::MetalHybridRenderer(
    MetalHybridRenderer&& other
) noexcept = default;

MetalHybridRenderer& MetalHybridRenderer::operator=(
    MetalHybridRenderer&& other
) noexcept = default;

MetalHybridRendererDiagnostics MetalHybridRenderer::compile(
    const HybridGaussianScene& scene,
    const std::uint32_t capacity
) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        std::string reason;
        if (!scene.valid(&reason)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidScene,
                std::move(reason)
            );
        }
        if (capacity == 0u ||
            state_->config.width == 0u ||
            state_->config.height == 0u ||
            state_->config.maximumGaussiansPerTile == 0u ||
            state_->config.maximumGaussiansPerTile >
                MR_HYBRID_MAX_GAUSSIANS_PER_TILE ||
            !finite4(state_->config.clearColorAndDepth) ||
            state_->config.clearColorAndDepth.w <= 0.0f) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidConfiguration,
                "visual sensor dimensions, tile capacity, clear "
                "values, or world capacity are invalid"
            );
        }
        diagnostics = initialize(
            *state_,
            std::move(diagnostics)
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        MetalHybridRendererLayout layout;
        layout.capacity = capacity;
        layout.width = state_->config.width;
        layout.height = state_->config.height;
        layout.tileCountX =
            (layout.width + MR_HYBRID_TILE_SIZE - 1u) /
            MR_HYBRID_TILE_SIZE;
        layout.tileCountY =
            (layout.height + MR_HYBRID_TILE_SIZE - 1u) /
            MR_HYBRID_TILE_SIZE;
        const auto fitsUint32 = [](const std::size_t value) {
            return value <=
                std::numeric_limits<std::uint32_t>::max();
        };
        if (!fitsUint32(scene.gaussians.size()) ||
            !fitsUint32(scene.meshVertices.size()) ||
            !fitsUint32(scene.meshTriangles.size()) ||
            !fitsUint32(scene.materials.size()) ||
            !fitsUint32(scene.sensorBindings.size())) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual scene element count exceeds uint32"
            );
        }
        layout.gaussianCount =
            static_cast<std::uint32_t>(
                scene.gaussians.size()
            );
        layout.meshVertexCount =
            static_cast<std::uint32_t>(
                scene.meshVertices.size()
            );
        layout.meshTriangleCount =
            static_cast<std::uint32_t>(
                scene.meshTriangles.size()
            );
        layout.materialCount =
            static_cast<std::uint32_t>(
                scene.materials.size()
            );
        layout.bodyCount = scene.bodyCount;
        layout.sensorBindingCount =
            static_cast<std::uint32_t>(
                scene.sensorBindings.size()
            );
        layout.maximumGaussiansPerTile =
            state_->config.maximumGaussiansPerTile;

        std::size_t pixelCount = 0u;
        std::size_t tileCount = 0u;
        std::size_t projectedCount = 0u;
        std::size_t tileIndexCount = 0u;
        std::size_t bodyStateCount = 0u;
        if (!checkedMultiply(
                layout.width,
                layout.height,
                pixelCount
            ) ||
            !checkedMultiply(pixelCount, capacity, pixelCount) ||
            !checkedMultiply(
                layout.tileCountX,
                layout.tileCountY,
                tileCount
            ) ||
            !checkedMultiply(tileCount, capacity, tileCount) ||
            !checkedMultiply(
                scene.gaussians.size(),
                capacity,
                projectedCount
            ) ||
            !checkedMultiply(
                tileCount,
                layout.maximumGaussiansPerTile,
                tileIndexCount
            ) ||
            !checkedMultiply(
                capacity,
                layout.bodyCount,
                bodyStateCount
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual sensor element counts overflow"
            );
        }

        std::size_t gaussianBytes = 0u;
        std::size_t meshVertexBytes = 0u;
        std::size_t meshTriangleBytes = 0u;
        std::size_t materialBytes = 0u;
        std::size_t sensorBindingBytes = 0u;
        std::size_t bodyStateBytes = 0u;
        std::size_t projectedBytes = 0u;
        std::size_t tileCountBytes = 0u;
        std::size_t tileIndexBytes = 0u;
        std::size_t tileOverflowBytes = 0u;
        std::size_t meshWinnerBytes = 0u;
        std::size_t rgbBytes = 0u;
        std::size_t depthBytes = 0u;
        std::size_t uintBytes = 0u;
        std::size_t float4Bytes = 0u;
        std::size_t uint4Bytes = 0u;
        if (!checkedBytes<MRHybridGaussianGPU>(
                scene.gaussians.size(),
                gaussianBytes
            ) ||
            !checkedBytes<MRVisualMeshVertexGPU>(
                scene.meshVertices.size(),
                meshVertexBytes
            ) ||
            !checkedBytes<MRVisualMeshTriangleGPU>(
                scene.meshTriangles.size(),
                meshTriangleBytes
            ) ||
            !checkedBytes<MRVisualMaterialGPU>(
                scene.materials.size(),
                materialBytes
            ) ||
            !checkedBytes<MRVisualSensorBindingGPU>(
                scene.sensorBindings.size(),
                sensorBindingBytes
            ) ||
            !checkedBytes<MRBodyStateGPU>(
                bodyStateCount,
                bodyStateBytes
            ) ||
            !checkedBytes<MRHybridProjectedGaussianGPU>(
                projectedCount,
                projectedBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                tileCount,
                tileCountBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                tileIndexCount,
                tileIndexBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                capacity,
                tileOverflowBytes
            ) ||
            !checkedMultiply(
                pixelCount,
                2u * sizeof(std::uint32_t),
                meshWinnerBytes
            ) ||
            meshWinnerBytes >
                std::numeric_limits<NSUInteger>::max() ||
            !checkedBytes<mr_float4>(
                pixelCount,
                rgbBytes
            ) ||
            !checkedBytes<float>(
                pixelCount,
                depthBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                pixelCount,
                uintBytes
            ) ||
            !checkedBytes<mr_float4>(
                pixelCount,
                float4Bytes
            ) ||
            !checkedBytes<mr_uint4>(
                pixelCount,
                uint4Bytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual sensor buffers exceed Metal limits"
            );
        }

        RendererBuffers buffers;
        buffers.gaussians = makePrivateBuffer(
            state_->device,
            gaussianBytes,
            @"MetalRobo visual Gaussians"
        );
        buffers.meshVertices = makePrivateBuffer(
            state_->device,
            meshVertexBytes,
            @"MetalRobo visual mesh vertices"
        );
        buffers.meshTriangles = makePrivateBuffer(
            state_->device,
            meshTriangleBytes,
            @"MetalRobo visual mesh triangles"
        );
        buffers.materials = makePrivateBuffer(
            state_->device,
            materialBytes,
            @"MetalRobo visual materials"
        );
        buffers.sensorBindings = makePrivateBuffer(
            state_->device,
            sensorBindingBytes,
            @"MetalRobo visual sensor bindings"
        );
        buffers.currentBodies = makeSharedBuffer(
            state_->device,
            bodyStateBytes,
            @"MetalRobo visual current bodies"
        );
        buffers.previousBodies = makeSharedBuffer(
            state_->device,
            bodyStateBytes,
            @"MetalRobo visual previous bodies"
        );
        buffers.projected = makePrivateBuffer(
            state_->device,
            projectedBytes,
            @"MetalRobo projected Gaussians"
        );
        buffers.tileCounts = makePrivateBuffer(
            state_->device,
            tileCountBytes,
            @"MetalRobo Gaussian tile counts"
        );
        buffers.tileIndices = makePrivateBuffer(
            state_->device,
            tileIndexBytes,
            @"MetalRobo Gaussian tile indices"
        );
        buffers.tileOverflowCounts = makePrivateBuffer(
            state_->device,
            tileOverflowBytes,
            @"MetalRobo Gaussian tile overflows"
        );
        buffers.meshWinners = makePrivateBuffer(
            state_->device,
            meshWinnerBytes,
            @"MetalRobo mesh pixel winners"
        );
        buffers.rgb = makePrivateBuffer(
            state_->device,
            rgbBytes,
            @"MetalRobo visual RGB"
        );
        buffers.depth = makePrivateBuffer(
            state_->device,
            depthBytes,
            @"MetalRobo visual depth"
        );
        buffers.segmentation = makePrivateBuffer(
            state_->device,
            uintBytes,
            @"MetalRobo visual semantics"
        );
        buffers.identities = makePrivateBuffer(
            state_->device,
            uint4Bytes,
            @"MetalRobo visual identities"
        );
        buffers.normals = makePrivateBuffer(
            state_->device,
            float4Bytes,
            @"MetalRobo visual normals"
        );
        buffers.motion = makePrivateBuffer(
            state_->device,
            float4Bytes,
            @"MetalRobo visual motion"
        );
        buffers.validity = makePrivateBuffer(
            state_->device,
            uintBytes,
            @"MetalRobo visual validity"
        );
        const std::array allBuffers{
            buffers.gaussians,
            buffers.meshVertices,
            buffers.meshTriangles,
            buffers.materials,
            buffers.sensorBindings,
            buffers.currentBodies,
            buffers.previousBodies,
            buffers.projected,
            buffers.tileCounts,
            buffers.tileIndices,
            buffers.tileOverflowCounts,
            buffers.meshWinners,
            buffers.rgb,
            buffers.depth,
            buffers.segmentation,
            buffers.identities,
            buffers.normals,
            buffers.motion,
            buffers.validity,
        };
        if (std::ranges::any_of(
                allBuffers,
                [](id<MTLBuffer> buffer) {
                    return buffer == nil;
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate visual sensor buffers"
            );
        }
        std::memset(
            buffers.currentBodies.contents,
            0,
            buffers.currentBodies.length
        );
        std::memset(
            buffers.previousBodies.contents,
            0,
            buffers.previousBodies.length
        );

        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual scene upload command"
            );
        }
        std::vector<id<MTLBuffer>> staging;
        if (!upload(
                state_->device,
                blit,
                scene.gaussians.data(),
                gaussianBytes,
                buffers.gaussians,
                @"MetalRobo Gaussian upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                scene.meshVertices.data(),
                meshVertexBytes,
                buffers.meshVertices,
                @"MetalRobo mesh vertex upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                scene.meshTriangles.data(),
                meshTriangleBytes,
                buffers.meshTriangles,
                @"MetalRobo mesh triangle upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                scene.materials.data(),
                materialBytes,
                buffers.materials,
                @"MetalRobo material upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                scene.sensorBindings.data(),
                sensorBindingBytes,
                buffers.sensorBindings,
                @"MetalRobo sensor binding upload",
                staging
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate visual scene upload buffers"
            );
        }
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "visual scene upload failed: " +
                    describeError(command.error)
            );
        }

        std::size_t retained = 0u;
        for (const std::size_t bytes : {
                 gaussianBytes,
                 meshVertexBytes,
                 meshTriangleBytes,
                 materialBytes,
                 sensorBindingBytes,
                 2u * bodyStateBytes,
                 projectedBytes,
                 tileCountBytes,
                 tileIndexBytes,
                 tileOverflowBytes,
                 meshWinnerBytes,
                 rgbBytes,
                 depthBytes,
                 3u * uintBytes,
                 2u * float4Bytes,
                 uint4Bytes,
             }) {
            if (!checkedAdd(retained, bytes, retained)) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::capacityOverflow,
                    "visual sensor retained byte count overflows"
                );
            }
        }
        layout.retainedPrivateBytes = retained;
        state_->buffers = std::move(buffers);
        state_->layout = layout;
        state_->assetCount = scene.assetCount;
        state_->sensorProfiles = scene.sensorBindings;
        state_->requiresLiveState =
            std::ranges::any_of(
                scene.gaussians,
                [](const MRHybridGaussianGPU& gaussian) {
                    return gaussian.binding.w ==
                        MR_HYBRID_GAUSSIAN_BODY_LOCAL;
                }
            ) ||
            std::ranges::any_of(
                scene.meshTriangles,
                [](const MRVisualMeshTriangleGPU& triangle) {
                    return triangle.binding.z ==
                            MR_VISUAL_BINDING_RIGID_BODY ||
                        triangle.binding.z ==
                            MR_VISUAL_BINDING_ARTICULATED_LINK;
                }
            ) ||
            std::ranges::any_of(
                scene.sensorBindings,
                [](const MRVisualSensorBindingGPU& binding) {
                    return binding.identity.x ==
                            MR_VISUAL_BINDING_RIGID_BODY ||
                        binding.identity.x ==
                            MR_VISUAL_BINDING_ARTICULATED_LINK;
                }
            );
        state_->activeEnvironmentCount = 0u;
        state_->activeMetadata = {};
        state_->compiled = true;
        diagnostics.layout = layout;
        diagnostics.deviceName = nsString(state_->device.name);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalBufferFailure,
            "host allocation failed while compiling visual sensor runtime"
        );
    } catch (const std::exception& error) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::render(
    const MetalWorldFamilyContext& worlds,
    const std::uint32_t environmentCount,
    const std::uint32_t cameraIndex
) {
    HybridDeviceStateBatch state;
    state.environmentCount = environmentCount;
    state.source = MR_VISUAL_SOURCE_SIMULATION;
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual sensor command"
            );
        }
        const auto start = std::chrono::steady_clock::now();
        MetalHybridRendererDiagnostics diagnostics =
            encodeLocked(
                *state_,
                worlds,
                state,
                cameraIndex,
                encoder
            );
        if (!diagnostics.succeeded()) {
            [encoder endEncoding];
            return diagnostics;
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "visual sensor render failed: " +
                    describeError(command.error)
            );
        }
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::renderLive(
    const MetalWorldFamilyContext& worlds,
    const HybridLiveStateBatch& liveState,
    const std::uint32_t cameraIndex
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        const std::size_t expected =
            static_cast<std::size_t>(
                liveState.environmentCount
            ) * liveState.bodyCount;
        if (liveState.environmentCount == 0u ||
            liveState.bodyCount != state_->layout.bodyCount ||
            liveState.currentBodies.size() != expected ||
            (!liveState.previousBodies.empty() &&
             liveState.previousBodies.size() != expected)) {
            return reject(
                {},
                MetalHybridRendererStatus::missingLiveState,
                "host live-state dimensions do not match the visual scene"
            );
        }
        const std::size_t bytes =
            expected * sizeof(MRBodyStateGPU);
        if (bytes > state_->buffers.currentBodies.length ||
            bytes > state_->buffers.previousBodies.length) {
            return reject(
                {},
                MetalHybridRendererStatus::capacityOverflow,
                "host live-state batch exceeds compiled capacity"
            );
        }
        std::memcpy(
            state_->buffers.currentBodies.contents,
            liveState.currentBodies.data(),
            bytes
        );
        if (liveState.previousBodies.empty()) {
            std::memcpy(
                state_->buffers.previousBodies.contents,
                liveState.currentBodies.data(),
                bytes
            );
        } else {
            std::memcpy(
                state_->buffers.previousBodies.contents,
                liveState.previousBodies.data(),
                bytes
            );
        }
        HybridDeviceStateBatch deviceState;
        deviceState.currentBodyStates =
            (__bridge void*)state_->buffers.currentBodies;
        deviceState.previousBodyStates =
            liveState.previousBodies.empty()
            ? nullptr
            : (__bridge void*)state_->buffers.previousBodies;
        deviceState.environmentCount =
            liveState.environmentCount;
        deviceState.bodyCount = liveState.bodyCount;
        deviceState.frameIndex = liveState.frameIndex;
        deviceState.sensorSequence =
            liveState.sensorSequence;
        deviceState.source = liveState.source;
        deviceState.captureTimestampSeconds =
            liveState.captureTimestampSeconds;
        deviceState.frameAgeSeconds =
            liveState.frameAgeSeconds;

        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create live visual sensor command"
            );
        }
        const auto start = std::chrono::steady_clock::now();
        MetalHybridRendererDiagnostics diagnostics =
            encodeLocked(
                *state_,
                worlds,
                deviceState,
                cameraIndex,
                encoder
            );
        if (!diagnostics.succeeded()) {
            [encoder endEncoding];
            return diagnostics;
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "live visual sensor render failed: " +
                    describeError(command.error)
            );
        }
        return diagnostics;
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::encode(
    const MetalWorldFamilyContext& worlds,
    const HybridDeviceStateBatch& liveState,
    const std::uint32_t cameraIndex,
    void* metalComputeCommandEncoder
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return encodeLocked(
            *state_,
            worlds,
            liveState,
            cameraIndex,
            (__bridge id<MTLComputeCommandEncoder>)
                metalComputeCommandEncoder
        );
    } catch (const std::exception& error) {
        return reject(
            {},
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
    }
}

MetalHybridRendererDiagnostics MetalHybridRenderer::readback(
    HybridObservationBatch& output
) {
    MetalHybridRendererDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            "visual sensor runtime has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        diagnostics.layout = state_->layout;
        diagnostics.deviceName = nsString(state_->device.name);
        if (!state_->compiled ||
            state_->activeEnvironmentCount == 0u) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::notCompiled,
                "render observations before reading them back"
            );
        }
        std::size_t pixelCount = 0u;
        if (!checkedMultiply(
                state_->layout.width,
                state_->layout.height,
                pixelCount
            ) ||
            !checkedMultiply(
                pixelCount,
                state_->activeEnvironmentCount,
                pixelCount
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual readback pixel count overflows"
            );
        }
        HybridObservationBatch candidate;
        candidate.environmentCount =
            state_->activeEnvironmentCount;
        candidate.width = state_->layout.width;
        candidate.height = state_->layout.height;
        candidate.rgb.resize(pixelCount);
        candidate.depth.resize(pixelCount);
        candidate.segmentation.resize(pixelCount);
        candidate.identities.resize(pixelCount);
        candidate.normals.resize(pixelCount);
        candidate.motion.resize(pixelCount);
        candidate.validity.resize(pixelCount);
        candidate.metadata = state_->activeMetadata;

        const std::size_t rgbBytes =
            pixelCount * sizeof(mr_float4);
        const std::size_t depthBytes =
            pixelCount * sizeof(float);
        const std::size_t uintBytes =
            pixelCount * sizeof(std::uint32_t);
        const std::size_t uint4Bytes =
            pixelCount * sizeof(mr_uint4);
        id<MTLBuffer> rgb = makeSharedBuffer(
            state_->device,
            rgbBytes,
            @"MetalRobo visual RGB readback"
        );
        id<MTLBuffer> depth = makeSharedBuffer(
            state_->device,
            depthBytes,
            @"MetalRobo visual depth readback"
        );
        id<MTLBuffer> segmentation = makeSharedBuffer(
            state_->device,
            uintBytes,
            @"MetalRobo visual semantic readback"
        );
        id<MTLBuffer> identities = makeSharedBuffer(
            state_->device,
            uint4Bytes,
            @"MetalRobo visual identity readback"
        );
        id<MTLBuffer> normals = makeSharedBuffer(
            state_->device,
            rgbBytes,
            @"MetalRobo visual normal readback"
        );
        id<MTLBuffer> motion = makeSharedBuffer(
            state_->device,
            rgbBytes,
            @"MetalRobo visual motion readback"
        );
        id<MTLBuffer> validity = makeSharedBuffer(
            state_->device,
            uintBytes,
            @"MetalRobo visual validity readback"
        );
        const std::array readbacks{
            rgb,
            depth,
            segmentation,
            identities,
            normals,
            motion,
            validity,
        };
        if (std::ranges::any_of(
                readbacks,
                [](id<MTLBuffer> buffer) {
                    return buffer == nil;
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                "could not allocate visual observation readback"
            );
        }
        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual observation readback command"
            );
        }
        const auto copy =
            [blit](
                id<MTLBuffer> source,
                id<MTLBuffer> destination,
                const std::size_t bytes
            ) {
            [blit copyFromBuffer:source
                    sourceOffset:0u
                        toBuffer:destination
               destinationOffset:0u
                            size:bytes];
        };
        copy(state_->buffers.rgb, rgb, rgbBytes);
        copy(state_->buffers.depth, depth, depthBytes);
        copy(
            state_->buffers.segmentation,
            segmentation,
            uintBytes
        );
        copy(
            state_->buffers.identities,
            identities,
            uint4Bytes
        );
        copy(state_->buffers.normals, normals, rgbBytes);
        copy(state_->buffers.motion, motion, rgbBytes);
        copy(state_->buffers.validity, validity, uintBytes);
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalCommandFailure,
                "visual observation readback failed: " +
                    describeError(command.error)
            );
        }
        std::memcpy(
            candidate.rgb.data(),
            rgb.contents,
            rgbBytes
        );
        std::memcpy(
            candidate.depth.data(),
            depth.contents,
            depthBytes
        );
        std::memcpy(
            candidate.segmentation.data(),
            segmentation.contents,
            uintBytes
        );
        std::memcpy(
            candidate.identities.data(),
            identities.contents,
            uint4Bytes
        );
        std::memcpy(
            candidate.normals.data(),
            normals.contents,
            rgbBytes
        );
        std::memcpy(
            candidate.motion.data(),
            motion.contents,
            rgbBytes
        );
        std::memcpy(
            candidate.validity.data(),
            validity.contents,
            uintBytes
        );
        output = std::move(candidate);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalBufferFailure,
            "host allocation failed during visual readback"
        );
    } catch (const std::exception& error) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::internalFailure,
            error.what()
        );
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
    const MetalHybridRendererBuffer buffer
) const noexcept {
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
        case MetalHybridRendererBuffer::identities:
            selected = state_->buffers.identities;
            break;
        case MetalHybridRendererBuffer::normals:
            selected = state_->buffers.normals;
            break;
        case MetalHybridRendererBuffer::motion:
            selected = state_->buffers.motion;
            break;
        case MetalHybridRendererBuffer::validity:
            selected = state_->buffers.validity;
            break;
        case MetalHybridRendererBuffer::meshWinners:
            selected = state_->buffers.meshWinners;
            break;
        }
        return (__bridge void*)selected;
    } catch (...) {
        return nullptr;
    }
}

const char* metalHybridRendererStatusName(
    const MetalHybridRendererStatus status
) noexcept {
    switch (status) {
    case MetalHybridRendererStatus::success:
        return "success";
    case MetalHybridRendererStatus::invalidScene:
        return "invalid_scene";
    case MetalHybridRendererStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalHybridRendererStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalHybridRendererStatus::missingLiveState:
        return "missing_live_state";
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
    return "unknown";
}

} // namespace metalrobo
