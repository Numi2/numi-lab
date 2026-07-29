#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalHybridRenderer.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
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
    __strong id<MTLBuffer> meshIndices = nil;
    __strong id<MTLBuffer> meshTriangles = nil;
    __strong id<MTLBuffer> meshPrimitives = nil;
    __strong id<MTLBuffer> meshInstances = nil;
    __strong id<MTLBuffer> materials = nil;
    __strong id<MTLBuffer> textureDescriptors = nil;
    __strong id<MTLBuffer> textureTexels = nil;
    __strong id<MTLBuffer> lights = nil;
    __strong id<MTLBuffer> environmentSH = nil;
    __strong id<MTLBuffer> rayVisibleInstances = nil;
    __strong id<MTLBuffer> rayBlasIndices = nil;
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
    __strong id<MTLBuffer> shadowAtlas = nil;
    __strong id<MTLBuffer> temporalAccumulation = nil;
};

struct RuntimeVisualScene {
    std::vector<MRHybridGaussianGPU> gaussians;
    std::vector<MRVisualVertexGPUV2> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MRVisualTriangleGPUV2> triangles;
    std::vector<MRVisualPrimitiveGPUV2> primitives;
    std::vector<MRVisualInstanceGPUV2> instances;
    std::vector<MRVisualMaterialGPUV2> materials;
    std::vector<MRVisualTextureGPUV1> textureDescriptors;
    std::vector<std::uint32_t> textureTexels;
    std::vector<MRVisualLightGPUV1> lights;
    std::array<mr_float4, 9u> environmentSH{};
    std::vector<MRVisualSensorBindingGPU> sensors;
};

using RayGeometryKey =
    std::vector<std::array<std::uint32_t, 3u>>;

struct CompactedPrimitiveAccelerationStructures {
    __strong NSArray<id<MTLAccelerationStructure>>* structures = nil;
    std::size_t retainedBytes = 0u;
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

mr_float4 interpolate4(
    const mr_float4 a,
    const mr_float4 b,
    const float fraction
) {
    return {
        a.x + (b.x - a.x) * fraction,
        a.y + (b.y - a.y) * fraction,
        a.z + (b.z - a.z) * fraction,
        a.w + (b.w - a.w) * fraction,
    };
}

mr_float4 interpolateQuaternion(
    mr_float4 a,
    mr_float4 b,
    const float fraction
) {
    const float dot =
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (dot < 0.0f) {
        b = {-b.x, -b.y, -b.z, -b.w};
    }
    mr_float4 result = interpolate4(a, b, fraction);
    const float length = std::sqrt(
        result.x * result.x + result.y * result.y +
        result.z * result.z + result.w * result.w
    );
    if (!(length > 1.0e-12f)) {
        return a;
    }
    result.x /= length;
    result.y /= length;
    result.z /= length;
    result.w /= length;
    return result;
}

MRBodyStateGPU interpolateBodyState(
    const MRBodyStateGPU& a,
    const MRBodyStateGPU& b,
    const float fraction
) {
    MRBodyStateGPU result{};
    result.position = interpolate4(a.position, b.position, fraction);
    result.orientation =
        interpolateQuaternion(a.orientation, b.orientation, fraction);
    result.linearVelocityAndInverseMass = interpolate4(
        a.linearVelocityAndInverseMass,
        b.linearVelocityAndInverseMass,
        fraction
    );
    result.angularVelocity =
        interpolate4(a.angularVelocity, b.angularVelocity, fraction);
    result.inverseInertiaWorldRow0 = interpolate4(
        a.inverseInertiaWorldRow0,
        b.inverseInertiaWorldRow0,
        fraction
    );
    result.inverseInertiaWorldRow1 = interpolate4(
        a.inverseInertiaWorldRow1,
        b.inverseInertiaWorldRow1,
        fraction
    );
    result.inverseInertiaWorldRow2 = interpolate4(
        a.inverseInertiaWorldRow2,
        b.inverseInertiaWorldRow2,
        fraction
    );
    std::copy(
        fraction < 0.5f
            ? std::begin(a.flagsAndIndices)
            : std::begin(b.flagsAndIndices),
        fraction < 0.5f
            ? std::end(a.flagsAndIndices)
            : std::end(b.flagsAndIndices),
        std::begin(result.flagsAndIndices)
    );
    return result;
}

bool sampleMotionStates(
    const VisualMotionSampleBatchV1& motion,
    const double timestamp,
    std::span<MRBodyStateGPU> output
) {
    const std::size_t perSample =
        static_cast<std::size_t>(motion.environmentCount) *
        motion.bodyCount;
    if (output.size() != perSample ||
        motion.timestampsSeconds.empty()) {
        return false;
    }
    const auto upper = std::upper_bound(
        motion.timestampsSeconds.begin(),
        motion.timestampsSeconds.end(),
        timestamp
    );
    if (upper == motion.timestampsSeconds.begin()) {
        std::ranges::copy(motion.sample(0u), output.begin());
        return true;
    }
    if (upper == motion.timestampsSeconds.end()) {
        std::ranges::copy(
            motion.sample(motion.sampleCount - 1u),
            output.begin()
        );
        return true;
    }
    const std::size_t right =
        static_cast<std::size_t>(
            upper - motion.timestampsSeconds.begin()
        );
    const std::size_t left = right - 1u;
    const double duration =
        motion.timestampsSeconds[right] -
        motion.timestampsSeconds[left];
    const float fraction = duration <= 0.0
        ? 0.0f
        : static_cast<float>(
              (timestamp - motion.timestampsSeconds[left]) /
              duration
          );
    const auto leftStates =
        motion.sample(static_cast<std::uint32_t>(left));
    const auto rightStates =
        motion.sample(static_cast<std::uint32_t>(right));
    for (std::size_t index = 0u; index < perSample; ++index) {
        output[index] = interpolateBodyState(
            leftStates[index],
            rightStates[index],
            fraction
        );
    }
    return true;
}

bool flattenScene(
    VisualRenderSceneV2&& scene,
    RuntimeVisualScene& output,
    std::string& reason
) {
    output.gaussians = std::move(scene.gaussians);
    output.vertices = std::move(scene.vertices);
    output.indices = std::move(scene.indices);
    output.primitives = std::move(scene.primitives);
    output.instances = std::move(scene.instances);
    output.materials = std::move(scene.materials);
    output.lights = std::move(scene.lightRig.lights);
    output.environmentSH = scene.environment.diffuseSH;
    output.sensors = std::move(scene.sensorBindings);
    output.textureDescriptors.reserve(scene.textures.size());
    std::size_t textureTexelCount = 0u;
    for (const VisualTextureImageV1& texture : scene.textures) {
        if (!checkedAdd(
                textureTexelCount,
                texture.rgba8.size() / 4u,
                textureTexelCount
            )) {
            reason = "visual texture atlas size overflows";
            return false;
        }
    }
    output.textureTexels.reserve(textureTexelCount);
    for (VisualTextureImageV1& texture : scene.textures) {
        if (output.textureTexels.size() >
            std::numeric_limits<std::uint32_t>::max() -
                texture.rgba8.size() / 4u) {
            reason = "visual texture atlas exceeds uint32";
            return false;
        }
        MRVisualTextureGPUV1 descriptor{};
        descriptor.dimensions = {
            texture.width,
            texture.height,
            static_cast<std::uint32_t>(
                texture.mipTexelOffsets.size()
            ),
            texture.flags,
        };
        const std::uint32_t base =
            static_cast<std::uint32_t>(
                output.textureTexels.size()
            );
        descriptor.storage = {
            base,
            static_cast<std::uint32_t>(texture.rgba8.size() / 4u),
            static_cast<std::uint32_t>(
                output.textureDescriptors.size() + 1u
            ),
            0u,
        };
        descriptor.mipOffsets0 = {
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
        };
        descriptor.mipOffsets1 = {
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
        };
        for (std::size_t level = 1u;
             level < texture.mipTexelOffsets.size();
             ++level) {
            const std::uint32_t offset =
                base + texture.mipTexelOffsets[level];
            if (level <= 4u) {
                (&descriptor.mipOffsets0.x)[level - 1u] = offset;
            } else if (level <= 8u) {
                (&descriptor.mipOffsets1.x)[level - 5u] = offset;
            }
        }
        output.textureDescriptors.push_back(descriptor);
        for (std::size_t byte = 0u;
             byte < texture.rgba8.size();
             byte += 4u) {
            output.textureTexels.push_back(
                static_cast<std::uint32_t>(texture.rgba8[byte]) |
                static_cast<std::uint32_t>(
                    texture.rgba8[byte + 1u]
                ) << 8u |
                static_cast<std::uint32_t>(
                    texture.rgba8[byte + 2u]
                ) << 16u |
                static_cast<std::uint32_t>(
                    texture.rgba8[byte + 3u]
                ) << 24u
            );
        }
        std::vector<std::uint8_t>{}.swap(texture.rgba8);
    }
    if (!output.primitives.empty() && output.materials.empty()) {
        reason = "V2 render scene has no materials";
        return false;
    }
    for (std::uint32_t primitiveIndex = 0u;
         primitiveIndex < output.primitives.size();
         ++primitiveIndex) {
        const MRVisualPrimitiveGPUV2& primitive =
            output.primitives[primitiveIndex];
        const MRVisualInstanceGPUV2& instance =
            output.instances[primitive.geometry.w];
        if ((instance.binding.w &
             MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR) == 0u) {
            continue;
        }
        const std::uint32_t first = primitive.geometry.x;
        const std::uint32_t end = first + primitive.geometry.y;
        for (std::uint32_t index = first;
             index < end;
             index += 3u) {
            if (output.triangles.size() ==
                    std::numeric_limits<std::uint32_t>::max()) {
                reason = "indexed visual triangle count exceeds uint32";
                return false;
            }
            MRVisualTriangleGPUV2 triangle{};
            triangle.verticesAndPrimitive = {
                output.indices[index],
                output.indices[index + 1u],
                output.indices[index + 2u],
                primitiveIndex,
            };
            output.triangles.push_back(triangle);
        }
    }
    return true;
}

} // namespace

namespace detail {

struct ReferenceFrameWorkspace {
    std::atomic_bool inUse{false};
    __strong id<MTLBuffer> motionBodies = nil;
    __strong id<MTLBuffer> instanceDescriptors = nil;
    __strong id<MTLBuffer> motionTransforms = nil;
    __strong id<MTLBuffer> buildScratch = nil;
    __strong id<MTLAccelerationStructure> instanceStructure = nil;
    std::size_t retainedBytes = 0u;
    std::size_t motionBodyBytes = 0u;
    std::size_t structureBytes = 0u;
    std::size_t scratchBytes = 0u;
    std::uint32_t instanceCount = 0u;
    std::uint32_t keyframeCount = 0u;
    bool built = false;
    std::uint32_t refitCount = 0u;
};

struct ExposureFrameWorkspace {
    std::atomic_bool inUse{false};
    __strong id<MTLBuffer> motionBodies = nil;
    std::size_t retainedBytes = 0u;
};

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
    __strong id<MTLComputePipelineState> clearObservationPipeline = nil;
    __strong id<MTLComputePipelineState> clearMeshPipeline = nil;
    __strong id<MTLComputePipelineState> binPipeline = nil;
    __strong id<MTLComputePipelineState> renderPipeline = nil;
    __strong id<MTLComputePipelineState> rasterMeshPipeline = nil;
    __strong id<MTLComputePipelineState> selectMeshPipeline = nil;
    __strong id<MTLComputePipelineState> compositeMeshPipeline = nil;
    __strong id<MTLComputePipelineState> clearShadowPipeline = nil;
    __strong id<MTLComputePipelineState> rasterShadowPipeline = nil;
    __strong id<MTLComputePipelineState> clearAccumulationPipeline = nil;
    __strong id<MTLComputePipelineState> accumulatePipeline = nil;
    __strong id<MTLComputePipelineState> resolveAccumulationPipeline = nil;
    __strong id<MTLComputePipelineState> applySensorPipeline = nil;
    __strong id<MTLComputePipelineState> prepareRayInstancesPipeline = nil;
    __strong id<MTLComputePipelineState> referenceRenderPipeline = nil;
    __strong NSArray<id<MTLAccelerationStructure>>*
        primitiveAccelerationStructures = nil;
    RendererBuffers buffers;
    MetalHybridRendererLayout layout;
    MRVisualFrameMetadataGPU activeMetadata{};
    std::vector<MRVisualSensorBindingGPU> sensorProfiles;
    VisualRendererProfileV1 rendererProfile =
        VisualRendererProfileV1::sensorFast();
    VisualEnvironmentV1 environment;
    std::uint64_t renderSceneFingerprint = 0u;
    std::uint32_t shadowLightIndex = MR_INVALID_INDEX;
    std::vector<std::uint32_t> rayVisibleInstances;
    std::vector<std::uint32_t> rayBlasIndices;
    std::vector<std::shared_ptr<ReferenceFrameWorkspace>>
        referenceWorkspaces;
    std::vector<std::shared_ptr<ExposureFrameWorkspace>>
        exposureWorkspaces;
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
    state.clearObservationPipeline =
        pipeline(@"mr_hybrid_clear_observations");
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
    state.clearShadowPipeline =
        pipeline(@"mr_hybrid_clear_shadow_atlas");
    state.rasterShadowPipeline =
        pipeline(@"mr_hybrid_rasterize_shadow_atlas");
    state.clearAccumulationPipeline =
        pipeline(@"mr_hybrid_clear_temporal_accumulation");
    state.accumulatePipeline =
        pipeline(@"mr_hybrid_accumulate_temporal_sample");
    state.resolveAccumulationPipeline =
        pipeline(@"mr_hybrid_resolve_temporal_accumulation");
    state.applySensorPipeline =
        pipeline(@"mr_hybrid_apply_sensor");
    state.prepareRayInstancesPipeline =
        pipeline(@"mr_hybrid_prepare_ray_instances");
    state.referenceRenderPipeline =
        pipeline(@"mr_hybrid_render_reference");
    if (state.clearPipeline == nil ||
        state.clearObservationPipeline == nil ||
        state.clearMeshPipeline == nil ||
        state.binPipeline == nil ||
        state.renderPipeline == nil ||
        state.rasterMeshPipeline == nil ||
        state.selectMeshPipeline == nil ||
        state.compositeMeshPipeline == nil ||
        state.clearShadowPipeline == nil ||
        state.rasterShadowPipeline == nil ||
        state.clearAccumulationPipeline == nil ||
        state.accumulatePipeline == nil ||
        state.resolveAccumulationPipeline == nil ||
        state.applySensorPipeline == nil) {
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

bool buildCompactedPrimitiveAccelerationStructures(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLBuffer> vertices,
    id<MTLBuffer> indices,
    const std::span<const MRVisualMaterialGPUV2> materials,
    const std::span<const RayGeometryKey> geometryKeys,
    const std::size_t maximumRetainedBytes,
    CompactedPrimitiveAccelerationStructures& output,
    std::string& reason
) {
    @autoreleasepool {
    output = {};
    if (geometryKeys.empty()) {
        output.structures = @[];
        return true;
    }
    if (![device supportsRaytracing]) {
        reason =
            "sensor_reference requires Metal ray-tracing support";
        return false;
    }

    NSMutableArray<MTLPrimitiveAccelerationStructureDescriptor*>*
        descriptors = [NSMutableArray
            arrayWithCapacity:geometryKeys.size()];
    std::vector<MTLAccelerationStructureSizes> sizes;
    sizes.reserve(geometryKeys.size());
    NSUInteger maximumScratchBytes = 0u;
    for (const RayGeometryKey& key : geometryKeys) {
        NSMutableArray<MTLAccelerationStructureGeometryDescriptor*>*
            geometryDescriptors = [NSMutableArray
                arrayWithCapacity:key.size()];
        for (const std::array<std::uint32_t, 3u>& geometry : key) {
            MTLAccelerationStructureTriangleGeometryDescriptor*
                triangle =
                    [MTLAccelerationStructureTriangleGeometryDescriptor
                        descriptor];
            triangle.vertexBuffer = vertices;
            triangle.vertexBufferOffset = 0u;
            // Positions are the first xyz values in the authored V2 vertex;
            // ray hardware walks the shared interleaved stream directly.
            triangle.vertexFormat = MTLAttributeFormatFloat3;
            triangle.vertexStride = sizeof(MRVisualVertexGPUV2);
            triangle.indexBuffer = indices;
            triangle.indexBufferOffset =
                static_cast<NSUInteger>(geometry[0]) *
                sizeof(std::uint32_t);
            triangle.indexType = MTLIndexTypeUInt32;
            triangle.triangleCount = geometry[1] / 3u;
            triangle.opaque =
                geometry[2] >= materials.size() ||
                materials[geometry[2]].flags.x ==
                    MR_VISUAL_ALPHA_OPAQUE;
            [geometryDescriptors addObject:triangle];
        }
        MTLPrimitiveAccelerationStructureDescriptor* descriptor =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        descriptor.geometryDescriptors = geometryDescriptors;
        descriptor.usage =
            MTLAccelerationStructureUsagePreferFastIntersection;
        [descriptors addObject:descriptor];
        const MTLAccelerationStructureSizes measured =
            [device accelerationStructureSizesWithDescriptor:descriptor];
        if (measured.accelerationStructureSize == 0u ||
            measured.buildScratchBufferSize == 0u) {
            reason =
                "Metal returned an invalid primitive acceleration "
                "structure size";
            return false;
        }
        sizes.push_back(measured);
        maximumScratchBytes = std::max(
            maximumScratchBytes,
            measured.buildScratchBufferSize
        );
    }

    id<MTLBuffer> scratch = makePrivateBuffer(
        device,
        maximumScratchBytes,
        @"MetalRobo reusable BLAS build scratch"
    );
    if (scratch == nil) {
        reason =
            "could not allocate the reusable BLAS build workspace";
        return false;
    }

    // Conservative BLAS builds are transient and can be substantially larger
    // than their compacted result. Build only a bounded group at a time,
    // reusing one scratch allocation across the entire scene.
    constexpr std::size_t kMaximumTransientBuildBatchBytes =
        256ull * 1024ull * 1024ull;
    const std::size_t transientBatchBudget = std::max<std::size_t>(
        1u,
        std::min(
            kMaximumTransientBuildBatchBytes,
            maximumRetainedBytes / 4u
        )
    );
    NSMutableArray<id<MTLAccelerationStructure>>* compacted =
        [NSMutableArray arrayWithCapacity:geometryKeys.size()];
    std::size_t retainedBytes = 0u;
    std::size_t first = 0u;
    while (first < sizes.size()) {
        @autoreleasepool {
        std::size_t end = first;
        std::size_t batchBytes = 0u;
        while (end < sizes.size()) {
            const std::size_t candidate =
                sizes[end].accelerationStructureSize;
            if (end != first &&
                candidate > transientBatchBudget - std::min(
                    transientBatchBudget,
                    batchBytes
                )) {
                break;
            }
            if (!checkedAdd(batchBytes, candidate, batchBytes)) {
                reason =
                    "primitive acceleration structure size overflows";
                return false;
            }
            ++end;
            if (batchBytes >= transientBatchBudget) {
                break;
            }
        }

        NSMutableArray<id<MTLAccelerationStructure>>* conservative =
            [NSMutableArray arrayWithCapacity:end - first];
        id<MTLBuffer> compactedSizes = makeSharedBuffer(
            device,
            (end - first) * sizeof(std::uint64_t),
            @"MetalRobo BLAS compacted sizes"
        );
        id<MTLCommandBuffer> buildCommand = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEncoder =
            [buildCommand accelerationStructureCommandEncoder];
        if (compactedSizes == nil ||
            buildCommand == nil ||
            buildEncoder == nil) {
            reason =
                "could not allocate a primitive acceleration "
                "structure build batch";
            return false;
        }
        for (std::size_t index = first; index < end; ++index) {
            id<MTLAccelerationStructure> structure =
                [device newAccelerationStructureWithSize:
                    sizes[index].accelerationStructureSize];
            if (structure == nil) {
                reason =
                    "could not allocate a primitive acceleration "
                    "structure";
                return false;
            }
            [conservative addObject:structure];
            [buildEncoder
                buildAccelerationStructure:structure
                descriptor:descriptors[index]
                scratchBuffer:scratch
                scratchBufferOffset:0u];
            [buildEncoder
                writeCompactedAccelerationStructureSize:structure
                toBuffer:compactedSizes
                offset:(index - first) * sizeof(std::uint64_t)
                sizeDataType:MTLDataTypeULong];
        }
        [buildEncoder endEncoding];
        [buildCommand commit];
        [buildCommand waitUntilCompleted];
        if (buildCommand.status != MTLCommandBufferStatusCompleted) {
            reason =
                "primitive acceleration structure build failed: " +
                describeError(buildCommand.error);
            return false;
        }

        const auto* compactedByteCounts =
            static_cast<const std::uint64_t*>(
                compactedSizes.contents
            );
        NSMutableArray<id<MTLAccelerationStructure>>*
            compactedBatch =
                [NSMutableArray arrayWithCapacity:end - first];
        for (std::size_t index = first; index < end; ++index) {
            const std::uint64_t byteCount =
                compactedByteCounts[index - first];
            if (byteCount == 0u ||
                byteCount >
                    std::numeric_limits<NSUInteger>::max() ||
                !checkedAdd(
                    retainedBytes,
                    static_cast<std::size_t>(byteCount),
                    retainedBytes
                ) ||
                retainedBytes > maximumRetainedBytes) {
                reason =
                    "compacted acceleration structures exceed the "
                    "configured retained-memory budget";
                return false;
            }
            id<MTLAccelerationStructure> structure =
                [device newAccelerationStructureWithSize:
                    static_cast<NSUInteger>(byteCount)];
            if (structure == nil) {
                reason =
                    "could not allocate a compacted primitive "
                    "acceleration structure";
                return false;
            }
            [compactedBatch addObject:structure];
        }

        id<MTLCommandBuffer> compactCommand = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> compactEncoder =
            [compactCommand accelerationStructureCommandEncoder];
        if (compactCommand == nil || compactEncoder == nil) {
            reason =
                "could not create the BLAS compaction command";
            return false;
        }
        for (std::size_t local = 0u;
             local < compactedBatch.count;
             ++local) {
            [compactEncoder
                copyAndCompactAccelerationStructure:
                    conservative[local]
                toAccelerationStructure:compactedBatch[local]];
        }
        [compactEncoder endEncoding];
        [compactCommand commit];
        [compactCommand waitUntilCompleted];
        if (compactCommand.status != MTLCommandBufferStatusCompleted) {
            reason =
                "primitive acceleration structure compaction failed: " +
                describeError(compactCommand.error);
            return false;
        }
        [compacted addObjectsFromArray:compactedBatch];
        first = end;
        }
    }

    output.structures = [compacted copy];
    output.retainedBytes = retainedBytes;
    return true;
    }
}

struct AcquiredReferenceWorkspace {
    std::shared_ptr<detail::ReferenceFrameWorkspace> workspace;
    __strong MTLInstanceAccelerationStructureDescriptor* descriptor =
        nil;
};

AcquiredReferenceWorkspace acquireReferenceWorkspace(
    detail::MetalHybridRendererState& state,
    const std::size_t motionBodyBytes,
    const std::uint32_t instanceCount,
    const std::uint32_t keyframeCount,
    std::string& reason
) {
    AcquiredReferenceWorkspace result;
    std::size_t transformCount = 0u;
    std::size_t descriptorBytes = 0u;
    std::size_t transformBytes = 0u;
    if (instanceCount == 0u || keyframeCount < 2u ||
        !checkedMultiply(
            instanceCount,
            keyframeCount,
            transformCount
        ) ||
        !checkedBytes<MTLAccelerationStructureMotionInstanceDescriptor>(
            instanceCount,
            descriptorBytes
        ) ||
        !checkedBytes<MTLComponentTransform>(
            transformCount,
            transformBytes
        )) {
        reason =
            "reference instance or motion-transform count overflows";
        return result;
    }

    for (const auto& candidate : state.referenceWorkspaces) {
        bool expected = false;
        if (candidate != nullptr &&
            candidate->inUse.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel
            )) {
            result.workspace = candidate;
            break;
        }
    }
    if (result.workspace == nullptr) {
        if (state.referenceWorkspaces.size() >=
            state.config.maximumReferenceFramesInFlight) {
            reason =
                "all recycled sensor_reference frame workspaces are "
                "still in flight";
            return result;
        }
        result.workspace =
            std::make_shared<detail::ReferenceFrameWorkspace>();
        result.workspace->inUse.store(
            true,
            std::memory_order_release
        );
        state.referenceWorkspaces.push_back(result.workspace);
    }
    const auto releaseOnFailure = [&result]() {
        result.workspace->inUse.store(
            false,
            std::memory_order_release
        );
        result.workspace.reset();
    };

    const bool topologyChanged =
        result.workspace->instanceCount != instanceCount ||
        result.workspace->keyframeCount != keyframeCount;
    if (result.workspace->motionBodies == nil ||
        result.workspace->motionBodies.length < motionBodyBytes) {
        result.workspace->motionBodies = makeSharedBuffer(
            state.device,
            motionBodyBytes,
            @"MetalRobo recycled reference body history"
        );
    }
    if (result.workspace->instanceDescriptors == nil ||
        result.workspace->instanceDescriptors.length <
            descriptorBytes) {
        result.workspace->instanceDescriptors = makePrivateBuffer(
            state.device,
            descriptorBytes,
            @"MetalRobo recycled ray instance descriptors"
        );
    }
    if (result.workspace->motionTransforms == nil ||
        result.workspace->motionTransforms.length <
            transformBytes) {
        result.workspace->motionTransforms = makePrivateBuffer(
            state.device,
            transformBytes,
            @"MetalRobo recycled component motion transforms"
        );
    }
    if (result.workspace->motionBodies == nil ||
        result.workspace->instanceDescriptors == nil ||
        result.workspace->motionTransforms == nil) {
        reason =
            "could not allocate recycled reference frame buffers";
        releaseOnFailure();
        return result;
    }

    MTLInstanceAccelerationStructureDescriptor* descriptor =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instancedAccelerationStructures =
        state.primitiveAccelerationStructures;
    descriptor.instanceDescriptorBuffer =
        result.workspace->instanceDescriptors;
    descriptor.instanceDescriptorBufferOffset = 0u;
    descriptor.instanceDescriptorStride =
        sizeof(MTLAccelerationStructureMotionInstanceDescriptor);
    descriptor.instanceCount = instanceCount;
    descriptor.instanceDescriptorType =
        MTLAccelerationStructureInstanceDescriptorTypeMotion;
    descriptor.motionTransformBuffer =
        result.workspace->motionTransforms;
    descriptor.motionTransformBufferOffset = 0u;
    descriptor.motionTransformCount = transformCount;
    descriptor.motionTransformType = MTLTransformTypeComponent;
    descriptor.motionTransformStride =
        sizeof(MTLComponentTransform);
    descriptor.usage =
        MTLAccelerationStructureUsageRefit |
        MTLAccelerationStructureUsagePreferFastIntersection;
    const MTLAccelerationStructureSizes sizes =
        [state.device
            accelerationStructureSizesWithDescriptor:descriptor];
    const std::size_t scratchBytes = std::max(
        sizes.buildScratchBufferSize,
        sizes.refitScratchBufferSize
    );
    if (sizes.accelerationStructureSize == 0u ||
        scratchBytes == 0u) {
        reason =
            "Metal returned an invalid reference acceleration "
            "structure size";
        releaseOnFailure();
        return result;
    }
    if (result.workspace->instanceStructure == nil ||
        result.workspace->structureBytes <
            sizes.accelerationStructureSize) {
        result.workspace->instanceStructure =
            [state.device newAccelerationStructureWithSize:
                sizes.accelerationStructureSize];
        result.workspace->structureBytes =
            sizes.accelerationStructureSize;
        result.workspace->built = false;
    }
    if (result.workspace->buildScratch == nil ||
        result.workspace->scratchBytes < scratchBytes) {
        result.workspace->buildScratch = makePrivateBuffer(
            state.device,
            scratchBytes,
            @"MetalRobo recycled TLAS build scratch"
        );
        result.workspace->scratchBytes = scratchBytes;
    }
    if (result.workspace->instanceStructure == nil ||
        result.workspace->buildScratch == nil) {
        reason =
            "could not allocate the recycled reference TLAS workspace";
        releaseOnFailure();
        return result;
    }
    if (topologyChanged) {
        result.workspace->built = false;
        result.workspace->refitCount = 0u;
    }
    result.workspace->motionBodyBytes = motionBodyBytes;
    result.workspace->instanceCount = instanceCount;
    result.workspace->keyframeCount = keyframeCount;
    result.workspace->retainedBytes =
        result.workspace->motionBodies.length +
        result.workspace->instanceDescriptors.length +
        result.workspace->motionTransforms.length +
        result.workspace->scratchBytes +
        result.workspace->structureBytes;

    std::size_t totalRetained = state.layout.retainedPrivateBytes;
    for (const auto& workspace : state.referenceWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "reference workspace retained byte count overflows";
            releaseOnFailure();
            return result;
        }
    }
    for (const auto& workspace : state.exposureWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "visual frame workspace retained byte count "
                "overflows";
            releaseOnFailure();
            return result;
        }
    }
    if (totalRetained > state.config.maximumRetainedBytes) {
        reason =
            "recycled reference workspaces exceed the configured "
            "unified-memory budget";
        releaseOnFailure();
        return result;
    }
    result.descriptor = descriptor;
    return result;
}

std::shared_ptr<detail::ExposureFrameWorkspace>
acquireExposureWorkspace(
    detail::MetalHybridRendererState& state,
    const std::size_t motionBodyBytes,
    std::string& reason
) {
    std::shared_ptr<detail::ExposureFrameWorkspace> result;
    for (const auto& candidate : state.exposureWorkspaces) {
        bool expected = false;
        if (candidate != nullptr &&
            candidate->inUse.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel
            )) {
            result = candidate;
            break;
        }
    }
    if (result == nullptr) {
        if (state.exposureWorkspaces.size() >=
            state.config.maximumReferenceFramesInFlight) {
            reason =
                "all recycled physical-exposure workspaces are "
                "still in flight";
            return nullptr;
        }
        result =
            std::make_shared<detail::ExposureFrameWorkspace>();
        result->inUse.store(true, std::memory_order_release);
        state.exposureWorkspaces.push_back(result);
    }
    if (result->motionBodies == nil ||
        result->motionBodies.length < motionBodyBytes) {
        result->motionBodies = makeSharedBuffer(
            state.device,
            motionBodyBytes,
            @"MetalRobo recycled fast-shutter body history"
        );
    }
    if (result->motionBodies == nil) {
        reason =
            "could not allocate the recycled physical-exposure "
            "body history";
        result->inUse.store(false, std::memory_order_release);
        return nullptr;
    }
    result->retainedBytes = result->motionBodies.length;
    std::size_t totalRetained = state.layout.retainedPrivateBytes;
    for (const auto& workspace : state.referenceWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "visual frame workspace retained byte count "
                "overflows";
            result->inUse.store(false, std::memory_order_release);
            return nullptr;
        }
    }
    for (const auto& workspace : state.exposureWorkspaces) {
        if (workspace == nullptr ||
            !checkedAdd(
                totalRetained,
                workspace->retainedBytes,
                totalRetained
            )) {
            reason =
                "physical-exposure workspace retained byte count "
                "overflows";
            result->inUse.store(false, std::memory_order_release);
            return nullptr;
        }
    }
    if (totalRetained > state.config.maximumRetainedBytes) {
        reason =
            "recycled physical-exposure workspaces exceed the "
            "configured unified-memory budget";
        result->inUse.store(false, std::memory_order_release);
        return nullptr;
    }
    return result;
}

struct EncodePassOptions {
    NSUInteger currentBodyOffset = 0u;
    NSUInteger previousBodyOffset = 0u;
    std::uint32_t temporalSample = 0u;
    std::uint32_t temporalSampleCount = 1u;
    bool physicalExposure = false;
    bool clearAccumulation = false;
    bool accumulateRadiance = false;
    bool resolveAccumulation = false;
    bool applySensor = true;
    bool renderMeshes = true;
    bool truthOnly = false;
    bool updateShadows = true;
    MRHybridRenderUniformsGPU* encodedUniforms = nullptr;
    std::uint32_t bandFirst = 0u;
    std::uint32_t bandCount = 0u;
    std::uint32_t bandAxis = 0u;
    std::uint32_t motionKeyframes = 0u;
    float exposureFraction = -1.0f;
    float previousExposureFraction = 0.0f;
    float shutterWindowSeconds = 0.0f;
    bool interpolateMotion = false;
};

MetalHybridRendererDiagnostics encodeLocked(
    detail::MetalHybridRendererState& state,
    const MetalWorldFamilyContext& worlds,
    const HybridDeviceStateBatch& liveState,
    const std::uint32_t cameraIndex,
    id<MTLComputeCommandEncoder> encoder,
    const EncodePassOptions options = {}
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
         (options.currentBodyOffset >
              currentBodies.length ||
          liveBodyBytes >
              currentBodies.length -
                  options.currentBodyOffset ||
          options.previousBodyOffset >
              previousBodies.length ||
          liveBodyBytes >
              previousBodies.length -
                  options.previousBodyOffset))) {
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
              {
                  MR_VISUAL_SHUTTER_GLOBAL,
                  MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
                  0u,
                  0u,
              },
          };
    uniforms.sensorTiming = profile.timing;
    uniforms.sensorRangeAndResponse =
        profile.rangeAndResponse;
    if (options.physicalExposure) {
        uniforms.sensorRangeAndResponse.w = 0.0f;
    }
    uniforms.presentation = {
        state.layout.textureCount,
        state.layout.lightCount,
        static_cast<std::uint32_t>(
            state.rendererProfile.kind
        ),
        state.environment.textureIndex,
    };
    uniforms.shutter = {
        profile.shutter.x,
        profile.shutter.y,
        options.temporalSample,
        options.temporalSampleCount,
    };
    const float temporalFraction =
        options.temporalSampleCount <= 1u
        ? 0.5f
        : (
              static_cast<float>(options.temporalSample) + 0.5f
          ) /
              static_cast<float>(options.temporalSampleCount);
    uniforms.exposure = {
        temporalFraction,
        static_cast<float>(
            state.rendererProfile.rollingShutterBands
        ),
        state.environment.intensity,
        state.environment.rotationRadians,
    };
    uniforms.shadow = {
        state.rendererProfile.shadowMapResolution,
        state.rendererProfile.shadowMapResolution,
        state.rendererProfile.kind ==
                MR_VISUAL_RENDERER_SENSOR_REFERENCE
            ? 2u
            : 1u,
        state.shadowLightIndex,
    };
    uniforms.shadowBatch = {
        0u,
        environmentCount,
        state.layout.shadowLayerCapacity,
        0u,
    };
    uniforms.band = {
        options.bandFirst,
        options.bandCount == 0u
            ? (
                  options.bandAxis == 0u
                  ? state.layout.height
                  : state.layout.width
              )
            : options.bandCount,
        options.bandAxis,
        options.truthOnly ? 1u : 0u,
    };
    uniforms.ray = {
        state.layout.rayInstanceCount,
        options.motionKeyframes,
        state.rendererProfile.areaLightSamples,
        0u,
    };
    uniforms.rayTiming = {
        options.shutterWindowSeconds > 0.0f
            ? options.shutterWindowSeconds
            : profile.timing.y + profile.timing.z,
        0.5f,
        options.previousExposureFraction,
        options.interpolateMotion ? 1.0f : 0.0f,
    };
    if (options.exposureFraction >= 0.0f) {
        uniforms.exposure.x = options.exposureFraction;
    }
    if (options.encodedUniforms != nullptr) {
        *options.encodedUniforms = uniforms;
    }

    const NSUInteger bandPixelsPerEnvironment =
        static_cast<NSUInteger>(uniforms.band.y) *
        (
            uniforms.band.z == 0u
            ? state.layout.width
            : state.layout.height
        );
    const NSUInteger bandPixelCount =
        static_cast<NSUInteger>(environmentCount) *
        bandPixelsPerEnvironment;
    const std::uint32_t firstBandTile =
        uniforms.band.x / MR_HYBRID_TILE_SIZE;
    const std::uint32_t lastBandTile =
        (
            uniforms.band.x + uniforms.band.y - 1u
        ) / MR_HYBRID_TILE_SIZE;
    const std::uint32_t bandTilesOnAxis =
        lastBandTile - firstBandTile + 1u;
    const NSUInteger bandTilesPerEnvironment =
        uniforms.band.z == 0u
        ? static_cast<NSUInteger>(state.layout.tileCountX) *
            bandTilesOnAxis
        : static_cast<NSUInteger>(state.layout.tileCountY) *
            bandTilesOnAxis;
    const NSUInteger bandTileCount =
        static_cast<NSUInteger>(environmentCount) *
        bandTilesPerEnvironment;
    const NSUInteger projectedCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.gaussianCount;
    const NSUInteger triangleCount =
        static_cast<NSUInteger>(environmentCount) *
        state.layout.meshTriangleCount;

    encoder.label = @"MetalRobo visual sensor runtime";
    if (options.clearAccumulation) {
        [encoder
            setComputePipelineState:state.clearAccumulationPipeline];
        [encoder setBuffer:state.buffers.temporalAccumulation
                    offset:0u
                   atIndex:0u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:1u];
        const NSUInteger accumulationThreads =
            std::min<NSUInteger>(
                state.clearAccumulationPipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        [encoder dispatchThreads:MTLSizeMake(
                                     bandPixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      accumulationThreads,
                                      1u,
                                      1u
                                  )];
    }
    if (projectedCount != 0u) {
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
                                         bandTileCount,
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
    } else {
        [encoder
            setComputePipelineState:state.clearObservationPipeline];
        [encoder setBuffer:instances offset:0u atIndex:0u];
        [encoder setBuffer:appearances offset:0u atIndex:1u];
        [encoder setBuffer:state.buffers.rgb
                    offset:0u
                   atIndex:2u];
        [encoder setBuffer:state.buffers.depth
                    offset:0u
                   atIndex:3u];
        [encoder setBuffer:state.buffers.segmentation
                    offset:0u
                   atIndex:4u];
        [encoder setBuffer:state.buffers.identities
                    offset:0u
                   atIndex:5u];
        [encoder setBuffer:state.buffers.normals
                    offset:0u
                   atIndex:6u];
        [encoder setBuffer:state.buffers.motion
                    offset:0u
                   atIndex:7u];
        [encoder setBuffer:state.buffers.validity
                    offset:0u
                   atIndex:8u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:9u];
        const NSUInteger clearObservationThreads =
            std::min<NSUInteger>(
                state.clearObservationPipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        [encoder dispatchThreads:MTLSizeMake(
                                     bandPixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      clearObservationThreads,
                                      1u,
                                      1u
                                  )];
    }

    if (options.renderMeshes && triangleCount != 0u) {
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
        [encoder dispatchThreads:MTLSizeMake(
                                     bandPixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      clearMeshThreads,
                                      1u,
                                      1u
                                  )];
    }

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
        [encoder setBuffer:currentBodies
                    offset:options.currentBodyOffset
                   atIndex:5u];
        [encoder setBuffer:previousBodies
                    offset:options.previousBodyOffset
                   atIndex:6u];
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

    if (projectedCount != 0u) {
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
             dispatchThreadgroups:MTLSizeMake(
                                      bandTileCount,
                                      1u,
                                      1u
                                  )
            threadsPerThreadgroup:MTLSizeMake(
                                      MR_HYBRID_MAX_GAUSSIANS_PER_TILE,
                                      1u,
                                      1u
                                  )];
    }

    if (options.renderMeshes && triangleCount != 0u) {
        [encoder setComputePipelineState:state.rasterMeshPipeline];
        [encoder setBuffer:state.buffers.meshVertices
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:state.buffers.meshTriangles
                    offset:0u
                   atIndex:1u];
        [encoder setBuffer:state.buffers.meshPrimitives
                    offset:0u
                   atIndex:2u];
        [encoder setBuffer:state.buffers.meshInstances
                    offset:0u
                   atIndex:3u];
        [encoder setBuffer:instances offset:0u atIndex:4u];
        [encoder setBuffer:assets offset:0u atIndex:5u];
        [encoder setBuffer:sensors offset:0u atIndex:6u];
        [encoder setBuffer:state.buffers.sensorBindings
                    offset:0u
                   atIndex:7u];
        [encoder setBuffer:currentBodies
                    offset:options.currentBodyOffset
                   atIndex:8u];
        [encoder setBuffer:state.buffers.meshWinners
                    offset:0u
                   atIndex:9u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:10u];
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

        const bool shadowResourcesAvailable =
            !options.truthOnly &&
            state.shadowLightIndex != MR_INVALID_INDEX &&
            state.layout.shadowLayerCapacity != 0u;
        const std::uint32_t batchCapacity =
            shadowResourcesAvailable
            ? state.layout.shadowLayerCapacity
            : environmentCount;
        const bool updateShadowAtlas =
            shadowResourcesAvailable &&
            (
                options.updateShadows ||
                environmentCount >
                    state.layout.shadowLayerCapacity
            );
        const NSUInteger clearShadowThreads =
            std::min<NSUInteger>(
                state.clearShadowPipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        const NSUInteger shadowThreads =
            std::min<NSUInteger>(
                state.rasterShadowPipeline
                    .maxTotalThreadsPerThreadgroup,
                128u
            );
        const NSUInteger compositeThreads =
            std::min<NSUInteger>(
                state.compositeMeshPipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        for (std::uint32_t environmentStart = 0u;
             environmentStart < environmentCount;
             environmentStart += batchCapacity) {
            const std::uint32_t batchCount = std::min(
                batchCapacity,
                environmentCount - environmentStart
            );
            uniforms.shadowBatch = {
                environmentStart,
                batchCount,
                state.layout.shadowLayerCapacity,
                0u,
            };
            if (updateShadowAtlas) {
                const NSUInteger shadowPixelCount =
                    static_cast<NSUInteger>(batchCount) *
                    uniforms.shadow.x * uniforms.shadow.y;
                [encoder
                    setComputePipelineState:state.clearShadowPipeline];
                [encoder setBuffer:state.buffers.shadowAtlas
                            offset:0u
                           atIndex:0u];
                [encoder setBytes:&uniforms
                           length:sizeof(uniforms)
                          atIndex:1u];
                [encoder dispatchThreads:MTLSizeMake(
                                             shadowPixelCount,
                                             1u,
                                             1u
                                         )
                    threadsPerThreadgroup:MTLSizeMake(
                                              clearShadowThreads,
                                              1u,
                                              1u
                                          )];

                [encoder
                    setComputePipelineState:state.rasterShadowPipeline];
                [encoder setBuffer:state.buffers.meshVertices
                            offset:0u
                           atIndex:0u];
                [encoder setBuffer:state.buffers.meshTriangles
                            offset:0u
                           atIndex:1u];
                [encoder setBuffer:state.buffers.meshPrimitives
                            offset:0u
                           atIndex:2u];
                [encoder setBuffer:state.buffers.meshInstances
                            offset:0u
                           atIndex:3u];
                [encoder setBuffer:instances offset:0u atIndex:4u];
                [encoder setBuffer:assets offset:0u atIndex:5u];
                [encoder setBuffer:currentBodies
                            offset:options.currentBodyOffset
                           atIndex:6u];
                [encoder setBuffer:state.buffers.lights
                            offset:0u
                           atIndex:7u];
                [encoder setBuffer:state.buffers.shadowAtlas
                            offset:0u
                           atIndex:8u];
                [encoder setBytes:&uniforms
                           length:sizeof(uniforms)
                          atIndex:9u];
                const NSUInteger batchTriangleCount =
                    static_cast<NSUInteger>(batchCount) *
                    state.layout.meshTriangleCount;
                [encoder dispatchThreads:MTLSizeMake(
                                             batchTriangleCount,
                                             1u,
                                             1u
                                         )
                    threadsPerThreadgroup:MTLSizeMake(
                                              shadowThreads,
                                              1u,
                                              1u
                                          )];
            }

            [encoder
                setComputePipelineState:state.compositeMeshPipeline];
            [encoder setBuffer:state.buffers.meshVertices
                        offset:0u
                       atIndex:0u];
            [encoder setBuffer:state.buffers.meshTriangles
                        offset:0u
                       atIndex:1u];
            [encoder setBuffer:state.buffers.meshPrimitives
                        offset:0u
                       atIndex:2u];
            [encoder setBuffer:state.buffers.meshInstances
                        offset:0u
                       atIndex:3u];
            [encoder setBuffer:state.buffers.materials
                        offset:0u
                       atIndex:4u];
            [encoder setBuffer:instances offset:0u atIndex:5u];
            [encoder setBuffer:assets offset:0u atIndex:6u];
            [encoder setBuffer:sensors offset:0u atIndex:7u];
            [encoder setBuffer:appearances offset:0u atIndex:8u];
            [encoder setBuffer:state.buffers.sensorBindings
                        offset:0u
                       atIndex:9u];
            [encoder setBuffer:currentBodies
                        offset:options.currentBodyOffset
                       atIndex:10u];
            [encoder setBuffer:previousBodies
                        offset:options.previousBodyOffset
                       atIndex:11u];
            [encoder setBuffer:state.buffers.meshWinners
                        offset:0u
                       atIndex:12u];
            [encoder setBuffer:state.buffers.rgb
                        offset:0u
                       atIndex:13u];
            [encoder setBuffer:state.buffers.depth
                        offset:0u
                       atIndex:14u];
            [encoder setBuffer:state.buffers.segmentation
                        offset:0u
                       atIndex:15u];
            [encoder setBuffer:state.buffers.identities
                        offset:0u
                       atIndex:16u];
            [encoder setBuffer:state.buffers.normals
                        offset:0u
                       atIndex:17u];
            [encoder setBuffer:state.buffers.motion
                        offset:0u
                       atIndex:18u];
            [encoder setBuffer:state.buffers.validity
                        offset:0u
                       atIndex:19u];
            [encoder setBuffer:state.buffers.textureDescriptors
                        offset:0u
                       atIndex:20u];
            [encoder setBuffer:state.buffers.textureTexels
                        offset:0u
                       atIndex:21u];
            [encoder setBuffer:state.buffers.lights
                        offset:0u
                       atIndex:22u];
            [encoder setBuffer:state.buffers.environmentSH
                        offset:0u
                       atIndex:23u];
            [encoder setBuffer:state.buffers.shadowAtlas
                        offset:0u
                       atIndex:24u];
            [encoder setBytes:&uniforms
                       length:sizeof(uniforms)
                      atIndex:25u];
            [encoder dispatchThreads:MTLSizeMake(
                                         static_cast<NSUInteger>(
                                             batchCount
                                         ) * bandPixelsPerEnvironment,
                                         1u,
                                         1u
                                     )
                threadsPerThreadgroup:MTLSizeMake(
                                          compositeThreads,
                                          1u,
                                          1u
                                      )];
        }
    }

    if (options.accumulateRadiance) {
        [encoder setComputePipelineState:state.accumulatePipeline];
        [encoder setBuffer:state.buffers.rgb
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:state.buffers.temporalAccumulation
                    offset:0u
                   atIndex:1u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:2u];
        const NSUInteger accumulateThreads =
            std::min<NSUInteger>(
                state.accumulatePipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        [encoder dispatchThreads:MTLSizeMake(
                                     bandPixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      accumulateThreads,
                                      1u,
                                      1u
                                  )];
    }
    if (options.resolveAccumulation) {
        [encoder
            setComputePipelineState:state.resolveAccumulationPipeline];
        [encoder setBuffer:state.buffers.temporalAccumulation
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:state.buffers.rgb
                    offset:0u
                   atIndex:1u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:2u];
        const NSUInteger resolveThreads =
            std::min<NSUInteger>(
                state.resolveAccumulationPipeline
                    .maxTotalThreadsPerThreadgroup,
                256u
            );
        [encoder dispatchThreads:MTLSizeMake(
                                     bandPixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      resolveThreads,
                                      1u,
                                      1u
                                  )];
    }
    if (options.applySensor) {
        [encoder setComputePipelineState:state.applySensorPipeline];
        [encoder setBuffer:instances offset:0u atIndex:0u];
        [encoder setBuffer:sensors offset:0u atIndex:1u];
        [encoder setBuffer:state.buffers.rgb
                    offset:0u
                   atIndex:2u];
        [encoder setBuffer:state.buffers.depth
                    offset:0u
                   atIndex:3u];
        [encoder setBuffer:state.buffers.validity
                    offset:0u
                   atIndex:4u];
        [encoder setBytes:&uniforms
                   length:sizeof(uniforms)
                  atIndex:5u];
        const NSUInteger sensorThreads = std::min<NSUInteger>(
            state.applySensorPipeline.maxTotalThreadsPerThreadgroup,
            256u
        );
        [encoder dispatchThreads:MTLSizeMake(
                                     bandPixelCount,
                                     1u,
                                     1u
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      sensorThreads,
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

MetalHybridRendererDiagnostics encodeReferenceFrameLocked(
    detail::MetalHybridRendererState& state,
    const MetalWorldFamilyContext& worlds,
    const VisualMotionSampleBatchV1& motion,
    const std::uint32_t cameraIndex,
    id<MTLCommandBuffer> command
) {
    MetalHybridRendererDiagnostics diagnostics;
    diagnostics.layout = state.layout;
    diagnostics.deviceName = nsString(state.device.name);
    const MetalWorldFamilyLayout worldLayout = worlds.layout();
    if (!state.compiled ||
        !state.rendererProfile.rayQueryVisibility ||
        state.primitiveAccelerationStructures == nil ||
        state.prepareRayInstancesPipeline == nil ||
        state.referenceRenderPipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::notCompiled,
            "sensor_reference was not compiled for ray-query rendering"
        );
    }
    if (motion.environmentCount == 0u ||
        motion.environmentCount > state.layout.capacity ||
        motion.environmentCount >
            worldLayout.activeInstanceCount ||
        worldLayout.assetCountPerInstance < state.assetCount ||
        cameraIndex >= worldLayout.sensorCountPerInstance ||
        (state.layout.sensorBindingCount != 0u &&
         cameraIndex >= state.layout.sensorBindingCount)) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::incompatibleWorldFamily,
            "motion samples, assets, or camera are incompatible "
            "with the reference visual scene"
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
    if (instances == nil || assets == nil ||
        sensors == nil || appearances == nil ||
        instances.device != state.device ||
        assets.device != state.device ||
        sensors.device != state.device ||
        appearances.device != state.device) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::incompatibleWorldFamily,
            "reference world-family buffers are unavailable or on "
            "another Metal device"
        );
    }

    const std::uint32_t keyframeCount =
        state.rendererProfile.temporalSamples;
    std::size_t statesPerKeyframe = 0u;
    std::size_t retainedStateCount = 0u;
    std::size_t retainedStateBytes = 0u;
    std::size_t descriptorCountValue = 0u;
    if (keyframeCount < 2u ||
        !checkedMultiply(
            motion.environmentCount,
            motion.bodyCount,
            statesPerKeyframe
        ) ||
        !checkedMultiply(
            statesPerKeyframe,
            static_cast<std::size_t>(keyframeCount) + 1u,
            retainedStateCount
        ) ||
        !checkedBytes<MRBodyStateGPU>(
            retainedStateCount,
            retainedStateBytes
        ) ||
        !checkedMultiply(
            motion.environmentCount,
            state.layout.rayInstanceCount,
            descriptorCountValue
        ) ||
        descriptorCountValue >
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::capacityOverflow,
            "reference motion or instance count overflows"
        );
    }
    const std::uint32_t descriptorCount =
        static_cast<std::uint32_t>(descriptorCountValue);
    if (descriptorCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::invalidScene,
            "sensor_reference has no ray-visible or shadow-casting "
            "mesh instances"
        );
    }

    std::string reason;
    AcquiredReferenceWorkspace acquired =
        acquireReferenceWorkspace(
            state,
            retainedStateBytes,
            descriptorCount,
            keyframeCount,
            reason
        );
    if (acquired.workspace == nullptr ||
        acquired.descriptor == nil) {
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalBufferFailure,
            std::move(reason)
        );
    }
    const auto releaseWorkspace = [&acquired]() {
        acquired.workspace->inUse.store(
            false,
            std::memory_order_release
        );
    };

    std::span<MRBodyStateGPU> sampledStates{
        static_cast<MRBodyStateGPU*>(
            acquired.workspace->motionBodies.contents
        ),
        retainedStateCount,
    };
    const double shutterOpen = motion.exposureOpenSeconds;
    const double shutterClose = motion.exposureCloseSeconds;
    for (std::uint32_t keyframe = 0u;
         keyframe < keyframeCount;
         ++keyframe) {
        const double fraction =
            static_cast<double>(keyframe) /
            static_cast<double>(keyframeCount - 1u);
        const double timestamp =
            shutterOpen +
            fraction * (shutterClose - shutterOpen);
        if (!sampleMotionStates(
                motion,
                timestamp,
                {
                    sampledStates.data() +
                        statesPerKeyframe * keyframe,
                    statesPerKeyframe,
                }
            )) {
            releaseWorkspace();
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::missingLiveState,
                "could not interpolate reference motion keyframes"
            );
        }
    }
    const double truthTimestamp =
        0.5 * (shutterOpen + shutterClose);
    if (!sampleMotionStates(
            motion,
            truthTimestamp,
            {
                sampledStates.data() +
                    statesPerKeyframe * keyframeCount,
                statesPerKeyframe,
            }
        )) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::missingLiveState,
            "could not interpolate center-exposure reference truth"
        );
    }

    HybridDeviceStateBatch live;
    live.currentBodyStates =
        (__bridge void*)acquired.workspace->motionBodies;
    live.previousBodyStates =
        (__bridge void*)acquired.workspace->motionBodies;
    live.environmentCount = motion.environmentCount;
    live.bodyCount = motion.bodyCount;
    live.frameIndex = motion.frameIndex;
    live.sensorSequence = motion.sensorSequence;
    live.source = motion.source;
    live.captureTimestampSeconds = truthTimestamp;
    live.frameAgeSeconds = 0.0;

    id<MTLComputeCommandEncoder> baseEncoder =
        [command computeCommandEncoder];
    if (baseEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the reference Gaussian-base encoder"
        );
    }
    MRHybridRenderUniformsGPU uniforms{};
    EncodePassOptions baseOptions;
    baseOptions.currentBodyOffset =
        statesPerKeyframe * keyframeCount *
        sizeof(MRBodyStateGPU);
    baseOptions.previousBodyOffset =
        baseOptions.currentBodyOffset;
    baseOptions.physicalExposure = true;
    baseOptions.applySensor = false;
    baseOptions.renderMeshes = false;
    baseOptions.encodedUniforms = &uniforms;
    diagnostics = encodeLocked(
        state,
        worlds,
        live,
        cameraIndex,
        baseEncoder,
        baseOptions
    );
    [baseEncoder endEncoding];
    if (!diagnostics.succeeded()) {
        releaseWorkspace();
        return diagnostics;
    }

    uniforms.ray = {
        state.layout.rayInstanceCount,
        keyframeCount,
        state.rendererProfile.areaLightSamples,
        0u,
    };
    uniforms.shutter = {
        cameraIndex < state.sensorProfiles.size()
            ? state.sensorProfiles[cameraIndex].shutter.x
            : MR_VISUAL_SHUTTER_GLOBAL,
        cameraIndex < state.sensorProfiles.size()
            ? state.sensorProfiles[cameraIndex].shutter.y
            : MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
        0u,
        state.rendererProfile.temporalSamples,
    };
    uniforms.rayTiming = {
        static_cast<float>(
            std::max(0.0, shutterClose - shutterOpen)
        ),
        0.5f,
        1.0f / static_cast<float>(
            state.rendererProfile.temporalSamples
        ),
        0.0f,
    };

    id<MTLComputeCommandEncoder> prepareEncoder =
        [command computeCommandEncoder];
    if (prepareEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the ray-instance preparation encoder"
        );
    }
    prepareEncoder.label =
        @"MetalRobo reference motion-instance preparation";
    [prepareEncoder
        setComputePipelineState:state.prepareRayInstancesPipeline];
    [prepareEncoder setBuffer:state.buffers.meshInstances
                       offset:0u
                      atIndex:0u];
    [prepareEncoder setBuffer:state.buffers.rayVisibleInstances
                       offset:0u
                      atIndex:1u];
    [prepareEncoder setBuffer:state.buffers.rayBlasIndices
                       offset:0u
                      atIndex:2u];
    [prepareEncoder setBuffer:instances offset:0u atIndex:3u];
    [prepareEncoder setBuffer:assets offset:0u atIndex:4u];
    [prepareEncoder setBuffer:acquired.workspace->motionBodies
                       offset:0u
                      atIndex:5u];
    [prepareEncoder
        setBuffer:acquired.workspace->instanceDescriptors
        offset:0u
        atIndex:6u];
    [prepareEncoder setBuffer:acquired.workspace->motionTransforms
                       offset:0u
                      atIndex:7u];
    [prepareEncoder setBytes:&uniforms
                      length:sizeof(uniforms)
                     atIndex:8u];
    const NSUInteger prepareThreads =
        std::min<NSUInteger>(
            state.prepareRayInstancesPipeline
                .maxTotalThreadsPerThreadgroup,
            128u
        );
    [prepareEncoder dispatchThreads:MTLSizeMake(
                                        descriptorCount,
                                        1u,
                                        1u
                                    )
        threadsPerThreadgroup:MTLSizeMake(
                                  prepareThreads,
                                  1u,
                                  1u
                              )];
    [prepareEncoder endEncoding];

    id<MTLAccelerationStructureCommandEncoder> buildEncoder =
        [command accelerationStructureCommandEncoder];
    if (buildEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the reference TLAS encoder"
        );
    }
    buildEncoder.label = @"MetalRobo reference motion TLAS";
    // Refit recycles the existing TLAS for stable topology. Periodic rebuilds
    // restore traversal quality after large accumulated object motion.
    if (acquired.workspace->built &&
        acquired.workspace->refitCount < 32u) {
        [buildEncoder
            refitAccelerationStructure:
                acquired.workspace->instanceStructure
            descriptor:acquired.descriptor
            destination:nil
            scratchBuffer:acquired.workspace->buildScratch
            scratchBufferOffset:0u];
        ++acquired.workspace->refitCount;
    } else {
        [buildEncoder
            buildAccelerationStructure:
                acquired.workspace->instanceStructure
            descriptor:acquired.descriptor
            scratchBuffer:acquired.workspace->buildScratch
            scratchBufferOffset:0u];
        acquired.workspace->built = true;
        acquired.workspace->refitCount = 0u;
    }
    [buildEncoder endEncoding];

    id<MTLComputeCommandEncoder> referenceEncoder =
        [command computeCommandEncoder];
    if (referenceEncoder == nil) {
        releaseWorkspace();
        return reject(
            std::move(diagnostics),
            MetalHybridRendererStatus::metalCommandFailure,
            "could not create the reference ray-query encoder"
        );
    }
    referenceEncoder.label =
        @"MetalRobo exact-shutter reference renderer";
    [referenceEncoder
        setComputePipelineState:state.referenceRenderPipeline];
    [referenceEncoder setBuffer:state.buffers.meshVertices
                         offset:0u
                        atIndex:0u];
    [referenceEncoder setBuffer:state.buffers.meshIndices
                         offset:0u
                        atIndex:1u];
    [referenceEncoder setBuffer:state.buffers.meshPrimitives
                         offset:0u
                        atIndex:2u];
    [referenceEncoder setBuffer:state.buffers.meshInstances
                         offset:0u
                        atIndex:3u];
    [referenceEncoder setBuffer:state.buffers.rayVisibleInstances
                         offset:0u
                        atIndex:4u];
    [referenceEncoder setBuffer:state.buffers.materials
                         offset:0u
                        atIndex:5u];
    [referenceEncoder setBuffer:state.buffers.textureDescriptors
                         offset:0u
                        atIndex:6u];
    [referenceEncoder setBuffer:state.buffers.textureTexels
                         offset:0u
                        atIndex:7u];
    [referenceEncoder setBuffer:state.buffers.lights
                         offset:0u
                        atIndex:8u];
    [referenceEncoder setBuffer:state.buffers.environmentSH
                         offset:0u
                        atIndex:9u];
    [referenceEncoder setBuffer:instances offset:0u atIndex:10u];
    [referenceEncoder setBuffer:assets offset:0u atIndex:11u];
    [referenceEncoder setBuffer:sensors offset:0u atIndex:12u];
    [referenceEncoder setBuffer:appearances offset:0u atIndex:13u];
    [referenceEncoder setBuffer:state.buffers.sensorBindings
                         offset:0u
                        atIndex:14u];
    [referenceEncoder setBuffer:acquired.workspace->motionBodies
                         offset:0u
                        atIndex:15u];
    [referenceEncoder setBuffer:state.buffers.rgb
                         offset:0u
                        atIndex:16u];
    [referenceEncoder setBuffer:state.buffers.depth
                         offset:0u
                        atIndex:17u];
    [referenceEncoder setBuffer:state.buffers.segmentation
                         offset:0u
                        atIndex:18u];
    [referenceEncoder setBuffer:state.buffers.identities
                         offset:0u
                        atIndex:19u];
    [referenceEncoder setBuffer:state.buffers.normals
                         offset:0u
                        atIndex:20u];
    [referenceEncoder setBuffer:state.buffers.motion
                         offset:0u
                        atIndex:21u];
    [referenceEncoder setBuffer:state.buffers.validity
                         offset:0u
                        atIndex:22u];
    [referenceEncoder setBytes:&uniforms
                        length:sizeof(uniforms)
                       atIndex:23u];
    [referenceEncoder
        setAccelerationStructure:
            acquired.workspace->instanceStructure
        atBufferIndex:24u];
    for (id<MTLAccelerationStructure> primitive :
         state.primitiveAccelerationStructures) {
        [referenceEncoder
            useResource:primitive
            usage:MTLResourceUsageRead];
    }
    const NSUInteger pixelCount =
        static_cast<NSUInteger>(motion.environmentCount) *
        state.layout.width * state.layout.height;
    const NSUInteger referenceThreads =
        std::min<NSUInteger>(
            state.referenceRenderPipeline
                .maxTotalThreadsPerThreadgroup,
            128u
        );
    [referenceEncoder dispatchThreads:MTLSizeMake(
                                          pixelCount,
                                          1u,
                                          1u
                                      )
        threadsPerThreadgroup:MTLSizeMake(
                                  referenceThreads,
                                  1u,
                                  1u
                              )];

    [referenceEncoder
        setComputePipelineState:state.applySensorPipeline];
    [referenceEncoder setBuffer:instances offset:0u atIndex:0u];
    [referenceEncoder setBuffer:sensors offset:0u atIndex:1u];
    [referenceEncoder setBuffer:state.buffers.rgb
                         offset:0u
                        atIndex:2u];
    [referenceEncoder setBuffer:state.buffers.depth
                         offset:0u
                        atIndex:3u];
    [referenceEncoder setBuffer:state.buffers.validity
                         offset:0u
                        atIndex:4u];
    [referenceEncoder setBytes:&uniforms
                        length:sizeof(uniforms)
                       atIndex:5u];
    const NSUInteger sensorThreads =
        std::min<NSUInteger>(
            state.applySensorPipeline.maxTotalThreadsPerThreadgroup,
            256u
        );
    [referenceEncoder dispatchThreads:MTLSizeMake(
                                          pixelCount,
                                          1u,
                                          1u
                                      )
        threadsPerThreadgroup:MTLSizeMake(
                                  sensorThreads,
                                  1u,
                                  1u
                              )];
    [referenceEncoder endEncoding];

    const auto retainedWorkspace = acquired.workspace;
    [command addCompletedHandler:^(
        id<MTLCommandBuffer> completed
    ) {
        if (completed.status != MTLCommandBufferStatusCompleted) {
            retainedWorkspace->built = false;
            retainedWorkspace->refitCount = 0u;
        }
        retainedWorkspace->inUse.store(
            false,
            std::memory_order_release
        );
    }];
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
        assetCount >= MR_INVALID_INDEX ||
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
            gaussian.scaleAndImportance.w < 0.0f ||
            gaussian.colorAndEmission.x < 0.0f ||
            gaussian.colorAndEmission.y < 0.0f ||
            gaussian.colorAndEmission.z < 0.0f ||
            orientationSquared <= 1.0e-12 ||
            gaussian.binding.z == 0u ||
            gaussian.binding.z == MR_INVALID_INDEX ||
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
    VisualRenderSceneV2&& scene,
    const VisualRendererProfileV1& profile,
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
        if (!scene.valid(&reason) || !profile.valid(&reason)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidScene,
                std::move(reason)
            );
        }
        const std::uint32_t sceneAssetCount = scene.assetCount;
        const std::uint32_t sceneBodyCount = scene.bodyCount;
        VisualEnvironmentV1 sceneEnvironment = scene.environment;
        const std::uint64_t sceneFingerprint =
            scene.fingerprint != 0u
            ? scene.fingerprint
            : computeVisualRenderSceneV2Fingerprint(scene);
        RuntimeVisualScene runtime;
        if (!flattenScene(std::move(scene), runtime, reason)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidScene,
                std::move(reason)
            );
        }
        std::uint32_t shadowLightIndex = MR_INVALID_INDEX;
        std::uint32_t shadowLightPriority = 0u;
        for (std::uint32_t lightIndex = 0u;
             lightIndex < runtime.lights.size();
             ++lightIndex) {
            const MRVisualLightGPUV1& light =
                runtime.lights[lightIndex];
            if (light.shadow.x == 0u) {
                continue;
            }
            if (shadowLightIndex == MR_INVALID_INDEX ||
                light.identity.z > shadowLightPriority) {
                shadowLightIndex = lightIndex;
                shadowLightPriority = light.identity.z;
            }
        }
        std::vector<RayGeometryKey> rayGeometryKeys;
        std::vector<std::uint32_t> rayVisibleInstances;
        std::vector<std::uint32_t> rayBlasIndices;
        if (profile.rayQueryVisibility) {
            for (std::uint32_t instanceIndex = 0u;
                 instanceIndex < runtime.instances.size();
                 ++instanceIndex) {
                const MRVisualInstanceGPUV2& instance =
                    runtime.instances[instanceIndex];
                if ((instance.binding.w & (
                         MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR |
                         MR_VISUAL_INSTANCE_CASTS_SHADOW
                     )) == 0u ||
                    instance.geometry.y == 0u) {
                    continue;
                }
                RayGeometryKey key;
                key.reserve(instance.geometry.y);
                for (std::uint32_t geometry = 0u;
                     geometry < instance.geometry.y;
                     ++geometry) {
                    const MRVisualPrimitiveGPUV2& primitive =
                        runtime.primitives[
                            instance.geometry.x + geometry
                        ];
                    key.push_back({
                        primitive.geometry.x,
                        primitive.geometry.y,
                        primitive.geometry.z,
                    });
                }
                const auto found =
                    std::ranges::find(rayGeometryKeys, key);
                const std::uint32_t blasIndex =
                    found == rayGeometryKeys.end()
                    ? static_cast<std::uint32_t>(
                          rayGeometryKeys.size()
                      )
                    : static_cast<std::uint32_t>(
                          found - rayGeometryKeys.begin()
                      );
                if (found == rayGeometryKeys.end()) {
                    rayGeometryKeys.push_back(std::move(key));
                }
                rayVisibleInstances.push_back(instanceIndex);
                rayBlasIndices.push_back(blasIndex);
            }
        }
        if (capacity == 0u ||
            state_->config.width == 0u ||
            state_->config.height == 0u ||
            state_->config.maximumGaussiansPerTile == 0u ||
            state_->config.maximumGaussiansPerTile >
                MR_HYBRID_MAX_GAUSSIANS_PER_TILE ||
            state_->config.shadowLayerBatchSize == 0u ||
            state_->config.maximumReferenceFramesInFlight == 0u ||
            state_->config.maximumRetainedBytes == 0u ||
            state_->config.maximumShadowAtlasBytes == 0u ||
            !finite4(state_->config.clearColorAndDepth) ||
            state_->config.clearColorAndDepth.w <= 0.0f) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::invalidConfiguration,
                "visual sensor dimensions, tile capacity, clear "
                "values, memory budget, or world capacity are invalid"
            );
        }
        diagnostics = initialize(
            *state_,
            std::move(diagnostics)
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        if (profile.rayQueryVisibility &&
            (state_->prepareRayInstancesPipeline == nil ||
             state_->referenceRenderPipeline == nil)) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalPipelineFailure,
                "sensor_reference ray-query pipelines are unavailable"
            );
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
        if (!fitsUint32(runtime.gaussians.size()) ||
            !fitsUint32(runtime.vertices.size()) ||
            !fitsUint32(runtime.triangles.size()) ||
            !fitsUint32(runtime.primitives.size()) ||
            !fitsUint32(runtime.instances.size()) ||
            !fitsUint32(runtime.indices.size()) ||
            !fitsUint32(runtime.materials.size()) ||
            !fitsUint32(runtime.textureDescriptors.size()) ||
            !fitsUint32(runtime.textureTexels.size()) ||
            !fitsUint32(runtime.lights.size()) ||
            !fitsUint32(runtime.sensors.size())) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual scene element count exceeds uint32"
            );
        }
        layout.gaussianCount =
            static_cast<std::uint32_t>(
                runtime.gaussians.size()
            );
        layout.meshVertexCount =
            static_cast<std::uint32_t>(
                runtime.vertices.size()
            );
        layout.meshTriangleCount =
            static_cast<std::uint32_t>(
                runtime.triangles.size()
            );
        layout.meshPrimitiveCount =
            static_cast<std::uint32_t>(runtime.primitives.size());
        layout.meshInstanceCount =
            static_cast<std::uint32_t>(runtime.instances.size());
        layout.meshIndexCount =
            static_cast<std::uint32_t>(runtime.indices.size());
        layout.materialCount =
            static_cast<std::uint32_t>(
                runtime.materials.size()
            );
        layout.textureCount =
            static_cast<std::uint32_t>(
                runtime.textureDescriptors.size()
            );
        layout.lightCount =
            static_cast<std::uint32_t>(
                runtime.lights.size()
            );
        layout.bodyCount = sceneBodyCount;
        layout.sensorBindingCount =
            static_cast<std::uint32_t>(
                runtime.sensors.size()
            );
        layout.rayInstanceCount =
            static_cast<std::uint32_t>(
                rayVisibleInstances.size()
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
                runtime.gaussians.size(),
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
        std::size_t meshIndexBytes = 0u;
        std::size_t meshTriangleBytes = 0u;
        std::size_t meshPrimitiveBytes = 0u;
        std::size_t meshInstanceBytes = 0u;
        std::size_t materialBytes = 0u;
        std::size_t textureDescriptorBytes = 0u;
        std::size_t textureTexelBytes = 0u;
        std::size_t lightBytes = 0u;
        std::size_t environmentBytes = 0u;
        std::size_t rayVisibleInstanceBytes = 0u;
        std::size_t rayBlasIndexBytes = 0u;
        std::size_t sensorBindingBytes = 0u;
        std::size_t bodyStateBytes = 0u;
        std::size_t projectedBytes = 0u;
        std::size_t tileCountBytes = 0u;
        std::size_t tileIndexBytes = 0u;
        std::size_t tileOverflowBytes = 0u;
        std::size_t meshWinnerBytes = 0u;
        std::size_t rgbBytes = 0u;
        std::size_t temporalAccumulationBytes = 0u;
        std::size_t depthBytes = 0u;
        std::size_t uintBytes = 0u;
        std::size_t float4Bytes = 0u;
        std::size_t uint4Bytes = 0u;
        std::size_t shadowPixelsPerLayer = 0u;
        std::size_t shadowBytesPerLayer = 0u;
        std::size_t shadowPixelCount = 0u;
        std::size_t shadowBytes = 0u;
        if (!checkedMultiply(
                profile.shadowMapResolution,
                profile.shadowMapResolution,
                shadowPixelsPerLayer
            ) ||
            !checkedBytes<std::uint32_t>(
                shadowPixelsPerLayer,
                shadowBytesPerLayer
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "one shadow layer exceeds Metal limits"
            );
        }
        const bool usesShadowAtlas =
            !profile.rayQueryVisibility &&
            !runtime.triangles.empty() &&
            shadowLightIndex != MR_INVALID_INDEX &&
            profile.shadowMapResolution != 0u;
        if (usesShadowAtlas) {
            const std::size_t budgetLayers =
                state_->config.maximumShadowAtlasBytes /
                shadowBytesPerLayer;
            if (budgetLayers == 0u) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::capacityOverflow,
                    "one shadow layer exceeds the configured shadow "
                    "workspace budget"
                );
            }
            layout.shadowLayerCapacity =
                static_cast<std::uint32_t>(std::min<std::size_t>({
                    capacity,
                    state_->config.shadowLayerBatchSize,
                    budgetLayers,
                }));
        }
        if (!checkedBytes<MRHybridGaussianGPU>(
                runtime.gaussians.size(),
                gaussianBytes
            ) ||
            !checkedBytes<MRVisualVertexGPUV2>(
                runtime.vertices.size(),
                meshVertexBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                profile.rayQueryVisibility
                    ? runtime.indices.size()
                    : 0u,
                meshIndexBytes
            ) ||
            !checkedBytes<MRVisualTriangleGPUV2>(
                runtime.triangles.size(),
                meshTriangleBytes
            ) ||
            !checkedBytes<MRVisualPrimitiveGPUV2>(
                runtime.primitives.size(),
                meshPrimitiveBytes
            ) ||
            !checkedBytes<MRVisualInstanceGPUV2>(
                runtime.instances.size(),
                meshInstanceBytes
            ) ||
            !checkedBytes<MRVisualMaterialGPUV2>(
                runtime.materials.size(),
                materialBytes
            ) ||
            !checkedBytes<MRVisualTextureGPUV1>(
                runtime.textureDescriptors.size(),
                textureDescriptorBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.textureTexels.size(),
                textureTexelBytes
            ) ||
            !checkedBytes<MRVisualLightGPUV1>(
                runtime.lights.size(),
                lightBytes
            ) ||
            !checkedBytes<mr_float4>(
                runtime.environmentSH.size(),
                environmentBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                rayVisibleInstances.size(),
                rayVisibleInstanceBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                rayBlasIndices.size(),
                rayBlasIndexBytes
            ) ||
            !checkedBytes<MRVisualSensorBindingGPU>(
                runtime.sensors.size(),
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
                runtime.gaussians.empty() ? 0u : tileCount,
                tileCountBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.gaussians.empty() ? 0u : tileIndexCount,
                tileIndexBytes
            ) ||
            !checkedBytes<std::uint32_t>(
                runtime.gaussians.empty() ? 0u : capacity,
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
            !checkedMultiply(
                profile.rayQueryVisibility
                    ? 0u
                    : pixelCount,
                4u * sizeof(std::uint16_t),
                temporalAccumulationBytes
            ) ||
            temporalAccumulationBytes >
                std::numeric_limits<NSUInteger>::max() ||
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
            ) ||
            !checkedMultiply(
                shadowPixelsPerLayer,
                layout.shadowLayerCapacity,
                shadowPixelCount
            ) ||
            !checkedBytes<std::uint32_t>(
                shadowPixelCount,
                shadowBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual sensor buffers exceed Metal limits"
            );
        }
        layout.shadowWorkspaceBytes = shadowBytes;

        std::size_t requestedRetention = 0u;
        for (const std::size_t bytes : {
                 gaussianBytes,
                 meshVertexBytes,
                 meshIndexBytes,
                 meshTriangleBytes,
                 meshPrimitiveBytes,
                 meshInstanceBytes,
                 materialBytes,
                 textureDescriptorBytes,
                 textureTexelBytes,
                 lightBytes,
                 environmentBytes,
                 rayVisibleInstanceBytes,
                 rayBlasIndexBytes,
                 sensorBindingBytes,
                 2u * bodyStateBytes,
                 projectedBytes,
                 tileCountBytes,
                 tileIndexBytes,
                 tileOverflowBytes,
                 meshWinnerBytes,
                 rgbBytes,
                 temporalAccumulationBytes,
                 depthBytes,
                 3u * uintBytes,
                 2u * float4Bytes,
                 uint4Bytes,
                 shadowBytes,
             }) {
            if (!checkedAdd(
                    requestedRetention,
                    bytes,
                    requestedRetention
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::capacityOverflow,
                    "visual sensor retained byte count overflows"
                );
            }
        }
        if (shadowBytes >
                state_->config.maximumShadowAtlasBytes ||
            requestedRetention >
                state_->config.maximumRetainedBytes) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual profile exceeds the configured unified-memory "
                "budget; lower shadow resolution or compile capacity"
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
        buffers.meshIndices = makePrivateBuffer(
            state_->device,
            meshIndexBytes,
            @"MetalRobo visual ray indices"
        );
        buffers.meshTriangles = makePrivateBuffer(
            state_->device,
            meshTriangleBytes,
            @"MetalRobo visual mesh triangles"
        );
        buffers.meshPrimitives = makePrivateBuffer(
            state_->device,
            meshPrimitiveBytes,
            @"MetalRobo visual mesh primitives"
        );
        buffers.meshInstances = makePrivateBuffer(
            state_->device,
            meshInstanceBytes,
            @"MetalRobo visual mesh instances"
        );
        buffers.materials = makePrivateBuffer(
            state_->device,
            materialBytes,
            @"MetalRobo visual materials"
        );
        buffers.textureDescriptors = makePrivateBuffer(
            state_->device,
            textureDescriptorBytes,
            @"MetalRobo visual texture descriptors"
        );
        buffers.textureTexels = makePrivateBuffer(
            state_->device,
            textureTexelBytes,
            @"MetalRobo visual texture texels"
        );
        buffers.lights = makePrivateBuffer(
            state_->device,
            lightBytes,
            @"MetalRobo visual light rig"
        );
        buffers.environmentSH = makePrivateBuffer(
            state_->device,
            environmentBytes,
            @"MetalRobo visual environment SH"
        );
        buffers.rayVisibleInstances = makePrivateBuffer(
            state_->device,
            rayVisibleInstanceBytes,
            @"MetalRobo ray-visible instance indices"
        );
        buffers.rayBlasIndices = makePrivateBuffer(
            state_->device,
            rayBlasIndexBytes,
            @"MetalRobo ray BLAS indices"
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
        buffers.shadowAtlas = makePrivateBuffer(
            state_->device,
            shadowBytes,
            @"MetalRobo environment-major shadow atlas"
        );
        buffers.temporalAccumulation = makePrivateBuffer(
            state_->device,
            temporalAccumulationBytes,
            @"MetalRobo temporal radiance accumulation"
        );
        const std::array allBuffers{
            buffers.gaussians,
            buffers.meshVertices,
            buffers.meshIndices,
            buffers.meshTriangles,
            buffers.meshPrimitives,
            buffers.meshInstances,
            buffers.materials,
            buffers.textureDescriptors,
            buffers.textureTexels,
            buffers.lights,
            buffers.environmentSH,
            buffers.rayVisibleInstances,
            buffers.rayBlasIndices,
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
            buffers.shadowAtlas,
            buffers.temporalAccumulation,
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
                runtime.gaussians.data(),
                gaussianBytes,
                buffers.gaussians,
                @"MetalRobo Gaussian upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.vertices.data(),
                meshVertexBytes,
                buffers.meshVertices,
                @"MetalRobo mesh vertex upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.indices.data(),
                meshIndexBytes,
                buffers.meshIndices,
                @"MetalRobo ray index upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.triangles.data(),
                meshTriangleBytes,
                buffers.meshTriangles,
                @"MetalRobo mesh triangle upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.primitives.data(),
                meshPrimitiveBytes,
                buffers.meshPrimitives,
                @"MetalRobo mesh primitive upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.instances.data(),
                meshInstanceBytes,
                buffers.meshInstances,
                @"MetalRobo mesh instance upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.materials.data(),
                materialBytes,
                buffers.materials,
                @"MetalRobo material upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.textureDescriptors.data(),
                textureDescriptorBytes,
                buffers.textureDescriptors,
                @"MetalRobo texture descriptor upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.textureTexels.data(),
                textureTexelBytes,
                buffers.textureTexels,
                @"MetalRobo texture texel upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.lights.data(),
                lightBytes,
                buffers.lights,
                @"MetalRobo light rig upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.environmentSH.data(),
                environmentBytes,
                buffers.environmentSH,
                @"MetalRobo environment SH upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                rayVisibleInstances.data(),
                rayVisibleInstanceBytes,
                buffers.rayVisibleInstances,
                @"MetalRobo ray-visible instance upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                rayBlasIndices.data(),
                rayBlasIndexBytes,
                buffers.rayBlasIndices,
                @"MetalRobo ray BLAS-index upload",
                staging
            ) ||
            !upload(
                state_->device,
                blit,
                runtime.sensors.data(),
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
        // Upload staging is no longer referenced once the blit completes.
        // Release it before BLAS construction so unified-memory peaks at one
        // scene representation instead of source + staging + acceleration
        // structures.
        std::vector<id<MTLBuffer>>{}.swap(staging);
        const bool requiresLiveState =
            std::ranges::any_of(
                runtime.gaussians,
                [](const MRHybridGaussianGPU& gaussian) {
                    return gaussian.binding.w ==
                        MR_HYBRID_GAUSSIAN_BODY_LOCAL;
                }
            ) ||
            std::ranges::any_of(
                runtime.instances,
                [](const MRVisualInstanceGPUV2& instance) {
                    return instance.binding.z ==
                            MR_VISUAL_BINDING_RIGID_BODY ||
                        instance.binding.z ==
                            MR_VISUAL_BINDING_ARTICULATED_LINK;
                }
            ) ||
            std::ranges::any_of(
                runtime.sensors,
                [](const MRVisualSensorBindingGPU& binding) {
                    return binding.identity.x ==
                            MR_VISUAL_BINDING_RIGID_BODY ||
                        binding.identity.x ==
                            MR_VISUAL_BINDING_ARTICULATED_LINK;
                }
            );
        auto sensorProfiles = std::move(runtime.sensors);
        std::vector<MRHybridGaussianGPU>{}.swap(runtime.gaussians);
        std::vector<MRVisualVertexGPUV2>{}.swap(runtime.vertices);
        std::vector<std::uint32_t>{}.swap(runtime.indices);
        std::vector<MRVisualTriangleGPUV2>{}.swap(runtime.triangles);
        std::vector<MRVisualPrimitiveGPUV2>{}.swap(runtime.primitives);
        std::vector<MRVisualInstanceGPUV2>{}.swap(runtime.instances);
        std::vector<MRVisualTextureGPUV1>{}.swap(
            runtime.textureDescriptors
        );
        std::vector<std::uint32_t>{}.swap(runtime.textureTexels);
        std::vector<MRVisualLightGPUV1>{}.swap(runtime.lights);

        CompactedPrimitiveAccelerationStructures
            primitiveAccelerationStructures;
        const std::size_t accelerationStructureBudget =
            state_->config.maximumRetainedBytes -
            requestedRetention;
        if (!buildCompactedPrimitiveAccelerationStructures(
                state_->device,
                state_->queue,
                buffers.meshVertices,
                buffers.meshIndices,
                runtime.materials,
                rayGeometryKeys,
                accelerationStructureBudget,
                primitiveAccelerationStructures,
                reason
            )) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::metalBufferFailure,
                std::move(reason)
            );
        }
        std::vector<MRVisualMaterialGPUV2>{}.swap(runtime.materials);
        layout.accelerationStructureBytes =
            primitiveAccelerationStructures.retainedBytes;
        if (!checkedAdd(
                requestedRetention,
                layout.accelerationStructureBytes,
                layout.retainedPrivateBytes
            ) ||
            layout.retainedPrivateBytes >
                state_->config.maximumRetainedBytes) {
            return reject(
                std::move(diagnostics),
                MetalHybridRendererStatus::capacityOverflow,
                "visual scene plus compacted acceleration structures "
                "exceed the configured retained-memory budget"
            );
        }
        state_->buffers = std::move(buffers);
        state_->primitiveAccelerationStructures =
            primitiveAccelerationStructures.structures;
        state_->referenceWorkspaces.clear();
        state_->exposureWorkspaces.clear();
        state_->layout = layout;
        state_->assetCount = sceneAssetCount;
        state_->sensorProfiles = std::move(sensorProfiles);
        state_->rendererProfile = profile;
        state_->environment = std::move(sceneEnvironment);
        state_->renderSceneFingerprint = sceneFingerprint;
        state_->shadowLightIndex = shadowLightIndex;
        state_->rayVisibleInstances =
            std::move(rayVisibleInstances);
        state_->rayBlasIndices = std::move(rayBlasIndices);
        state_->requiresLiveState = requiresLiveState;
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
        if (command == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual sensor command"
            );
        }
        const auto start = std::chrono::steady_clock::now();
        if (state_->rendererProfile.rayQueryVisibility) {
            if (state_->requiresLiveState) {
                return reject(
                    {},
                    MetalHybridRendererStatus::missingLiveState,
                    "body/link-bound sensor_reference scenes require "
                    "encodeFrame motion samples"
                );
            }
            const MRVisualSensorBindingGPU sensor =
                cameraIndex < state_->sensorProfiles.size()
                ? state_->sensorProfiles[cameraIndex]
                : MRVisualSensorBindingGPU{
                      {},
                      {15.0f, 1.0f / 120.0f, 0.0f, 0.0f},
                      {0.05f, 10.0f, 0.001f, 0.0f},
                      {
                          MR_VISUAL_SHUTTER_GLOBAL,
                          MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
                          0u,
                          0u,
                      },
                  };
            const double shutterWindow = std::max(
                1.0e-9,
                static_cast<double>(sensor.timing.y) +
                    (
                        sensor.shutter.x ==
                            MR_VISUAL_SHUTTER_ROLLING
                        ? static_cast<double>(sensor.timing.z)
                        : 0.0
                    )
            );
            VisualMotionSampleBatchV1 motion;
            motion.environmentCount = environmentCount;
            motion.bodyCount = state_->layout.bodyCount;
            motion.sampleCount = 2u;
            motion.exposureOpenSeconds = 0.0;
            motion.exposureCloseSeconds = shutterWindow;
            motion.timestampsSeconds = {0.0, shutterWindow};
            std::size_t motionBodyCount = 0u;
            if (motion.bodyCount == 0u ||
                !checkedMultiply(
                    static_cast<std::size_t>(environmentCount),
                    motion.bodyCount,
                    motionBodyCount
                ) ||
                !checkedMultiply(
                    motionBodyCount,
                    motion.sampleCount,
                    motionBodyCount
                )) {
                return reject(
                    {},
                    MetalHybridRendererStatus::capacityOverflow,
                    "static reference motion dimensions overflow"
                );
            }
            motion.bodyStates.resize(motionBodyCount);
            for (MRBodyStateGPU& body : motion.bodyStates) {
                body.orientation.w = 1.0f;
            }
            motion.scenarioIdentity = 1u;
            motion.sensorIdentity =
                static_cast<std::uint64_t>(cameraIndex) + 1u;
            motion.frameIndex = 1u;
            MetalHybridRendererDiagnostics diagnostics =
                encodeReferenceFrameLocked(
                    *state_,
                    worlds,
                    motion,
                    cameraIndex,
                    command
                );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }
            [command commit];
            [command waitUntilCompleted];
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start
                ).count();
            if (command.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalHybridRendererStatus::metalCommandFailure,
                    "reference visual sensor render failed: " +
                        describeError(command.error)
                );
            }
            return diagnostics;
        }
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "could not create visual sensor command"
            );
        }
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

MetalHybridRendererDiagnostics MetalHybridRenderer::encodeFrame(
    const MetalWorldFamilyContext& worlds,
    const VisualMotionSampleBatchV1& motion,
    const std::uint32_t cameraIndex,
    const MetalHybridFrameCommandContext& commandContext
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
        std::string reason;
        if (!motion.valid(&reason) ||
            !state_->compiled ||
            motion.bodyCount != state_->layout.bodyCount ||
            motion.environmentCount > state_->layout.capacity) {
            return reject(
                {},
                MetalHybridRendererStatus::missingLiveState,
                reason.empty()
                    ? "motion samples do not match the compiled scene"
                    : std::move(reason)
            );
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)
                commandContext.commandBuffer;
        if (command == nil ||
            command.commandQueue.device != state_->device) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "encodeFrame requires a caller-owned Metal command "
                "buffer on the renderer device"
            );
        }
        if (state_->rendererProfile.rayQueryVisibility) {
            if (commandContext.activeComputeCommandEncoder != nullptr) {
                return reject(
                    {},
                    MetalHybridRendererStatus::metalCommandFailure,
                    "sensor_reference requires a command buffer "
                    "without a caller-owned active encoder"
                );
            }
            return encodeReferenceFrameLocked(
                *state_,
                worlds,
                motion,
                cameraIndex,
                command
            );
        }
        id<MTLComputeCommandEncoder> encoder =
            (__bridge id<MTLComputeCommandEncoder>)
                commandContext.activeComputeCommandEncoder;
        const bool ownsEncoder = encoder == nil;
        if (ownsEncoder) {
            encoder = [command computeCommandEncoder];
        }
        if (encoder == nil) {
            return reject(
                {},
                MetalHybridRendererStatus::metalCommandFailure,
                "encodeFrame could not create a compute encoder"
            );
        }

        const std::uint32_t temporalSamples =
            state_->rendererProfile.temporalSamples;
        const std::size_t statesPerSample =
            static_cast<std::size_t>(motion.environmentCount) *
            motion.bodyCount;
        std::size_t retainedStateCount = 0u;
        std::size_t retainedStateBytes = 0u;
        if (!checkedMultiply(
                statesPerSample,
                temporalSamples,
                retainedStateCount
            ) ||
            !checkedBytes<MRBodyStateGPU>(
                retainedStateCount,
                retainedStateBytes
            ) ||
            retainedStateBytes >
                state_->config.maximumRetainedBytes / 2u) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            return reject(
                {},
                MetalHybridRendererStatus::capacityOverflow,
                "physical exposure samples exceed the unified-memory "
                "budget"
            );
        }
        std::string workspaceReason;
        const auto exposureWorkspace = acquireExposureWorkspace(
            *state_,
            retainedStateBytes,
            workspaceReason
        );
        if (exposureWorkspace == nullptr) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            return reject(
                {},
                MetalHybridRendererStatus::metalBufferFailure,
                std::move(workspaceReason)
            );
        }
        const auto releaseExposureWorkspace =
            [&exposureWorkspace]() {
                exposureWorkspace->inUse.store(
                    false,
                    std::memory_order_release
                );
            };
        id<MTLBuffer> motionBuffer =
            exposureWorkspace->motionBodies;
        std::span<MRBodyStateGPU> sampledStates{
            static_cast<MRBodyStateGPU*>(motionBuffer.contents),
            retainedStateCount,
        };
        const MRVisualSensorBindingGPU sensorProfile =
            cameraIndex < state_->sensorProfiles.size()
            ? state_->sensorProfiles[cameraIndex]
            : MRVisualSensorBindingGPU{
                  {},
                  {15.0f, 1.0f / 120.0f, 0.0f, 0.0f},
                  {0.05f, 10.0f, 0.001f, 0.0f},
                  {
                      MR_VISUAL_SHUTTER_GLOBAL,
                      MR_VISUAL_SHUTTER_TOP_TO_BOTTOM,
                      0u,
                      0u,
                  },
              };
        const double exposureDuration = std::max(
            0.0,
            static_cast<double>(sensorProfile.timing.y)
        );
        const bool rollingShutter =
            sensorProfile.shutter.x ==
                MR_VISUAL_SHUTTER_ROLLING &&
            sensorProfile.timing.z > 0.0f;
        const double readoutDuration = rollingShutter
            ? static_cast<double>(sensorProfile.timing.z)
            : 0.0;
        const double shutterWindow =
            exposureDuration + readoutDuration;
        const double availableWindow =
            motion.exposureCloseSeconds -
            motion.exposureOpenSeconds;
        if (!(shutterWindow > 0.0) ||
            availableWindow + 1.0e-9 < shutterWindow) {
            if (ownsEncoder) {
                [encoder endEncoding];
            }
            releaseExposureWorkspace();
            return reject(
                {},
                MetalHybridRendererStatus::missingLiveState,
                "motion history does not cover the complete shutter "
                "exposure and readout interval"
            );
        }
        for (std::uint32_t keyframe = 0u;
             keyframe < temporalSamples;
             ++keyframe) {
            const double fraction =
                temporalSamples <= 1u
                ? 0.5
                : static_cast<double>(keyframe) /
                    static_cast<double>(temporalSamples - 1u);
            const double timestamp =
                motion.exposureOpenSeconds +
                fraction * shutterWindow;
            if (!sampleMotionStates(
                    motion,
                    timestamp,
                    {
                        sampledStates.data() +
                            statesPerSample * keyframe,
                        statesPerSample,
                    }
                )) {
                if (ownsEncoder) {
                    [encoder endEncoding];
                }
                releaseExposureWorkspace();
                return reject(
                    {},
                    MetalHybridRendererStatus::missingLiveState,
                    "could not interpolate physical exposure state"
                );
            }
        }
        const double truthTimestamp =
            motion.exposureOpenSeconds +
            0.5 * shutterWindow;
        HybridDeviceStateBatch live;
        live.currentBodyStates = (__bridge void*)motionBuffer;
        live.previousBodyStates = (__bridge void*)motionBuffer;
        live.environmentCount = motion.environmentCount;
        live.bodyCount = motion.bodyCount;
        live.frameIndex = motion.frameIndex;
        live.sensorSequence = motion.sensorSequence;
        live.source = motion.source;
        live.captureTimestampSeconds = truthTimestamp;
        live.frameAgeSeconds = 0.0;

        const bool horizontalScan =
            sensorProfile.shutter.y ==
                MR_VISUAL_SHUTTER_LEFT_TO_RIGHT ||
            sensorProfile.shutter.y ==
                MR_VISUAL_SHUTTER_RIGHT_TO_LEFT;
        const std::uint32_t bandAxis =
            horizontalScan ? 1u : 0u;
        const std::uint32_t bandDimension =
            horizontalScan
            ? state_->layout.width
            : state_->layout.height;
        const std::uint32_t bandCount = rollingShutter
            ? std::max<std::uint32_t>(
                  1u,
                  std::min(
                      state_->rendererProfile.rollingShutterBands,
                      bandDimension
                  )
              )
            : 1u;

        MetalHybridRendererDiagnostics diagnostics;
        for (std::uint32_t band = 0u;
             band < bandCount;
             ++band) {
            const std::uint32_t first =
                static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(band) *
                    bandDimension / bandCount
                );
            const std::uint32_t end =
                static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(band + 1u) *
                    bandDimension / bandCount
                );
            const std::uint32_t pixelsInBand = end - first;
            float scanFraction =
                (
                    static_cast<float>(first) +
                    0.5f * static_cast<float>(pixelsInBand)
                ) /
                static_cast<float>(bandDimension);
            if (sensorProfile.shutter.y ==
                    MR_VISUAL_SHUTTER_BOTTOM_TO_TOP ||
                sensorProfile.shutter.y ==
                    MR_VISUAL_SHUTTER_RIGHT_TO_LEFT) {
                scanFraction = 1.0f - scanFraction;
            }

            float previousFraction = std::clamp(
                static_cast<float>(
                    (
                        scanFraction * readoutDuration +
                        0.5 * exposureDuration /
                            std::max<std::uint32_t>(
                                temporalSamples,
                                1u
                            )
                    ) / shutterWindow
                ),
                0.0f,
                1.0f
            );
            for (std::uint32_t sample = 0u;
                 sample < temporalSamples;
                 ++sample) {
                const double sampleFraction =
                    (
                        static_cast<double>(sample) + 0.5
                    ) /
                    temporalSamples;
                const float exposureFraction = std::clamp(
                    static_cast<float>(
                        (
                            scanFraction * readoutDuration +
                            sampleFraction * exposureDuration
                        ) / shutterWindow
                    ),
                    0.0f,
                    1.0f
                );
                EncodePassOptions options;
                options.temporalSample = sample;
                options.temporalSampleCount = temporalSamples;
                options.physicalExposure = true;
                options.clearAccumulation = sample == 0u;
                options.accumulateRadiance = true;
                options.applySensor = false;
                options.updateShadows = sample == 0u;
                options.bandFirst = first;
                options.bandCount = pixelsInBand;
                options.bandAxis = bandAxis;
                options.motionKeyframes = temporalSamples;
                options.exposureFraction = exposureFraction;
                options.previousExposureFraction =
                    sample == 0u
                    ? exposureFraction
                    : previousFraction;
                options.shutterWindowSeconds =
                    static_cast<float>(shutterWindow);
                options.interpolateMotion =
                    temporalSamples >= 2u;
                diagnostics = encodeLocked(
                    *state_,
                    worlds,
                    live,
                    cameraIndex,
                    encoder,
                    options
                );
                if (!diagnostics.succeeded()) {
                    if (ownsEncoder) {
                        [encoder endEncoding];
                    }
                    releaseExposureWorkspace();
                    return diagnostics;
                }
                previousFraction = exposureFraction;
            }

            // Geometry channels are evaluated at the band's row/column
            // exposure midpoint while radiance resolves from the two
            // temporal samples accumulated above.
            const float truthFraction = std::clamp(
                static_cast<float>(
                    (
                        scanFraction * readoutDuration +
                        0.5 * exposureDuration
                    ) / shutterWindow
                ),
                0.0f,
                1.0f
            );
            EncodePassOptions truthOptions;
            truthOptions.temporalSample = temporalSamples / 2u;
            truthOptions.temporalSampleCount = temporalSamples;
            truthOptions.physicalExposure = true;
            truthOptions.resolveAccumulation = true;
            truthOptions.applySensor = true;
            truthOptions.truthOnly = true;
            truthOptions.updateShadows = false;
            truthOptions.bandFirst = first;
            truthOptions.bandCount = pixelsInBand;
            truthOptions.bandAxis = bandAxis;
            truthOptions.motionKeyframes = temporalSamples;
            truthOptions.exposureFraction = truthFraction;
            truthOptions.previousExposureFraction =
                previousFraction;
            truthOptions.shutterWindowSeconds =
                static_cast<float>(shutterWindow);
            truthOptions.interpolateMotion =
                temporalSamples >= 2u;
            diagnostics = encodeLocked(
                *state_,
                worlds,
                live,
                cameraIndex,
                encoder,
                truthOptions
            );
            if (!diagnostics.succeeded()) {
                if (ownsEncoder) {
                    [encoder endEncoding];
                }
                releaseExposureWorkspace();
                return diagnostics;
            }
        }
        if (ownsEncoder) {
            [encoder endEncoding];
        }
        if (!diagnostics.succeeded()) {
            releaseExposureWorkspace();
            return diagnostics;
        }
        [command addCompletedHandler:^(
            id<MTLCommandBuffer> completed
        ) {
            (void)completed;
            exposureWorkspace->inUse.store(
                false,
                std::memory_order_release
            );
        }];
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            {},
            MetalHybridRendererStatus::metalBufferFailure,
            "host allocation failed for physical exposure samples"
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
        case MetalHybridRendererBuffer::shadowAtlas:
            selected = state_->buffers.shadowAtlas;
            break;
        case MetalHybridRendererBuffer::temporalAccumulation:
            selected = state_->buffers.temporalAccumulation;
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
