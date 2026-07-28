#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalWorldFamily.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
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
const char kMetalWorldFamilyImageAnchor = 0;

struct FamilyBuffers {
    __strong id<MTLBuffer> baseAssets = nil;
    __strong id<MTLBuffer> baseSensors = nil;
    __strong id<MTLBuffer> baseAppearances = nil;
    __strong id<MTLBuffer> variations = nil;
    __strong id<MTLBuffer> categoricalValues = nil;
    __strong id<MTLBuffer> uniforms = nil;
    __strong id<MTLBuffer> instances = nil;
    __strong id<MTLBuffer> assets = nil;
    __strong id<MTLBuffer> sensors = nil;
    __strong id<MTLBuffer> appearances = nil;
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
    std::string result = nsString(error.localizedDescription);
    if (result.empty()) {
        result = nsString(error.description);
    }
    return result.empty() ? "unknown Metal error" : result;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalWorldFamilyImageAnchor, &image) != 0 &&
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
    const std::filesystem::path configured{METALROBO_DEFAULT_METALLIB};
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

MetalWorldFamilyDiagnostics reject(
    MetalWorldFamilyDiagnostics diagnostics,
    const MetalWorldFamilyStatus status,
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

template <typename T>
bool checkedByteCount(
    const std::size_t count,
    std::size_t& bytes
) {
    return checkedMultiply(count, sizeof(T), bytes) &&
        bytes <= std::numeric_limits<NSUInteger>::max();
}

std::uint32_t low32(const std::uint64_t value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t high32(const std::uint64_t value) {
    return static_cast<std::uint32_t>(value >> 32u);
}

id<MTLBuffer> makePrivateBuffer(
    id<MTLDevice> device,
    const std::size_t logicalBytes,
    NSString* label
) {
    const NSUInteger allocationBytes = static_cast<NSUInteger>(
        std::max<std::size_t>(
            logicalBytes,
            kMinimumAllocationBytes
        )
    );
    id<MTLBuffer> buffer = [device
        newBufferWithLength:allocationBytes
                   options:MTLResourceStorageModePrivate];
    buffer.label = label;
    return buffer;
}

id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const std::size_t logicalBytes,
    NSString* label
) {
    const NSUInteger allocationBytes = static_cast<NSUInteger>(
        std::max<std::size_t>(
            logicalBytes,
            kMinimumAllocationBytes
        )
    );
    id<MTLBuffer> buffer = [device
        newBufferWithLength:allocationBytes
                   options:MTLResourceStorageModeShared];
    buffer.label = label;
    return buffer;
}

template <typename T>
id<MTLBuffer> makeUploadBuffer(
    id<MTLDevice> device,
    const std::span<const T> values,
    NSString* label
) {
    std::size_t logicalBytes = 0u;
    if (!checkedByteCount<T>(values.size(), logicalBytes)) {
        return nil;
    }
    id<MTLBuffer> buffer =
        makeSharedBuffer(device, logicalBytes, label);
    if (buffer == nil) {
        return nil;
    }
    if (logicalBytes != 0u) {
        std::memcpy(buffer.contents, values.data(), logicalBytes);
    } else {
        std::memset(buffer.contents, 0, kMinimumAllocationBytes);
    }
    return buffer;
}

template <typename T>
void copySharedBuffer(
    std::vector<T>& destination,
    id<MTLBuffer> buffer
) {
    if (!destination.empty()) {
        std::memcpy(
            destination.data(),
            buffer.contents,
            destination.size() * sizeof(T)
        );
    }
}

} // namespace

namespace detail {

struct MetalWorldFamilyContextState {
    explicit MetalWorldFamilyContextState(
        MetalWorldFamilyConfig configured
    )
        : config(std::move(configured)) {}

    MetalWorldFamilyConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool compiled = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> pipeline = nil;
    FamilyBuffers buffers{};
    MetalWorldFamilyLayout layout{};
    MetalWorldFamilyStats stats{};
    std::uint32_t instanceFlags = 0u;
    std::uint64_t familyFingerprint = 0u;
};

} // namespace detail

namespace {

MetalWorldFamilyDiagnostics initialize(
    detail::MetalWorldFamilyContextState& state,
    MetalWorldFamilyDiagnostics diagnostics
) {
    if (state.initialized) {
        diagnostics.deviceName = nsString(state.device.name);
        return diagnostics;
    }

    const std::string metallibPath = state.config.metallibPath.empty()
        ? defaultMetallibPath()
        : state.config.metallibPath;
    if (metallibPath.empty()) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metallibUnavailable,
            "MetalRobo.metallib could not be discovered"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalDeviceUnsupported,
            "world-family sampling requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalDeviceUnavailable,
            "failed to create the world-family Metal command queue"
        );
    }
    queue.label = @"MetalRobo world-family queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metallibUnavailable,
            "metallib path is not valid UTF-8"
        );
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalLibraryFailure,
            "failed to load metallib: " + describeError(error)
        );
    }
    id<MTLFunction> function = [library
        newFunctionWithName:@"mr_world_family_sample"];
    if (function == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalLibraryFailure,
            "metallib does not contain mr_world_family_sample"
        );
    }
    error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                       error:&error];
    if (pipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalPipelineFailure,
            "failed to create world-family pipeline: " +
                describeError(error)
        );
    }

    state.device = device;
    state.queue = queue;
    state.library = library;
    state.pipeline = pipeline;
    state.initialized = true;
    return diagnostics;
}

bool validateFamily(
    const WorldFamily& family,
    std::string& reason
) {
    if (family.fingerprint == 0u ||
        family.program.fingerprint == 0u ||
        family.program.id.empty() ||
        !family.worldTemplate.valid(&reason)) {
        if (reason.empty()) {
            reason = "world family identity or compiled program is empty";
        }
        return false;
    }
    const std::size_t assetCount = family.worldTemplate.assets.size();
    const std::size_t sensorCount = family.worldTemplate.sensors.size();
    const std::size_t appearanceCount =
        family.worldTemplate.appearances.size();
    for (const MRWorldVariationGPU& variation :
         family.program.variations) {
        const std::uint32_t target = variation.binding.z;
        const std::uint32_t index = variation.binding.w;
        if (variation.binding.x > MR_WORLD_VARIATION_CAMERA ||
            variation.binding.y > MR_WORLD_DISTRIBUTION_CATEGORICAL ||
            target > MR_WORLD_TARGET_CLUTTER_SET ||
            (target <= MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE &&
             index >= assetCount) ||
            (target == MR_WORLD_TARGET_CLUTTER_SET &&
             index >= assetCount) ||
            (target >= MR_WORLD_TARGET_SENSOR_POSITION_X &&
             target <= MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT &&
             index >= sensorCount) ||
            (target >= MR_WORLD_TARGET_APPEARANCE_EXPOSURE &&
             target <= MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY &&
             index >= appearanceCount)) {
            reason = "compiled world program contains an invalid binding";
            return false;
        }
        if (variation.binding.y == MR_WORLD_DISTRIBUTION_CATEGORICAL &&
            (variation.categorical.y == 0u ||
             variation.categorical.x >
                 family.program.categoricalValues.size() ||
             variation.categorical.y >
                 family.program.categoricalValues.size() -
                     variation.categorical.x)) {
            reason =
                "compiled world program categorical range is invalid";
            return false;
        }
    }
    return true;
}

} // namespace

MetalWorldFamilyContext::MetalWorldFamilyContext(
    MetalWorldFamilyConfig config
)
    : state_(std::make_shared<
          detail::MetalWorldFamilyContextState
      >(std::move(config))) {}

MetalWorldFamilyContext::~MetalWorldFamilyContext() = default;

MetalWorldFamilyContext::MetalWorldFamilyContext(
    MetalWorldFamilyContext&& other
) noexcept = default;

MetalWorldFamilyContext& MetalWorldFamilyContext::operator=(
    MetalWorldFamilyContext&& other
) noexcept = default;

MetalWorldFamilyDiagnostics MetalWorldFamilyContext::compile(
    const WorldFamily& family,
    const std::uint32_t capacity
) {
    MetalWorldFamilyDiagnostics diagnostics;
    diagnostics.familyFingerprint = family.fingerprint;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            "world-family context has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        std::string reason;
        if (!validateFamily(family, reason)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                std::move(reason)
            );
        }
        if (capacity == 0u) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidCapacity,
                "world-family capacity must be nonzero"
            );
        }

        diagnostics = initialize(*state_, std::move(diagnostics));
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        const std::size_t assetCount =
            family.worldTemplate.assets.size();
        const std::size_t sensorCount =
            family.worldTemplate.sensors.size();
        const std::size_t appearanceCount =
            family.worldTemplate.appearances.size();
        constexpr std::size_t addressLimit =
            std::numeric_limits<std::uint32_t>::max();
        if (assetCount > addressLimit ||
            sensorCount > addressLimit ||
            appearanceCount > addressLimit ||
            family.program.variations.size() > addressLimit ||
            family.program.categoricalValues.size() > addressLimit ||
            capacity > addressLimit /
                std::max<std::size_t>(assetCount, 1u) ||
            capacity > addressLimit /
                std::max<std::size_t>(sensorCount, 1u) ||
            capacity > addressLimit /
                std::max<std::size_t>(appearanceCount, 1u)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::capacityOverflow,
                "world-family streams exceed 32-bit GPU addressing"
            );
        }

        WorldProgram baseProgram;
        baseProgram.id = family.program.id + ".immutable-base";
        baseProgram.instanceFlags = family.program.instanceFlags;
        WorldFamily baseFamily;
        const WorldCompileResult baseCompile = compileWorldFamily(
            family.worldTemplate,
            baseProgram,
            baseFamily
        );
        if (!baseCompile.succeeded()) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                "failed to materialize immutable world base: " +
                    baseCompile.message
            );
        }
        const WorldInstanceBatch base = baseFamily.sample(1u, 0u);
        std::string batchReason;
        if (!base.valid(&batchReason)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                "immutable world base is invalid: " + batchReason
            );
        }

        MetalWorldFamilyLayout layout;
        layout.capacity = capacity;
        layout.assetCountPerInstance =
            static_cast<std::uint32_t>(assetCount);
        layout.sensorCountPerInstance =
            static_cast<std::uint32_t>(sensorCount);
        layout.appearanceCountPerInstance =
            static_cast<std::uint32_t>(appearanceCount);
        layout.variationCount = static_cast<std::uint32_t>(
            family.program.variations.size()
        );
        layout.categoricalValueCount = static_cast<std::uint32_t>(
            family.program.categoricalValues.size()
        );

        std::size_t baseAssetBytes = 0u;
        std::size_t baseSensorBytes = 0u;
        std::size_t baseAppearanceBytes = 0u;
        std::size_t variationBytes = 0u;
        std::size_t categoricalBytes = 0u;
        std::size_t instanceElements = 0u;
        std::size_t assetElements = 0u;
        std::size_t sensorElements = 0u;
        std::size_t appearanceElements = 0u;
        if (!checkedByteCount<MRWorldAssetInstanceGPU>(
                base.assets.size(),
                baseAssetBytes
            ) ||
            !checkedByteCount<MRWorldSensorInstanceGPU>(
                base.sensors.size(),
                baseSensorBytes
            ) ||
            !checkedByteCount<MRWorldAppearanceInstanceGPU>(
                base.appearances.size(),
                baseAppearanceBytes
            ) ||
            !checkedByteCount<MRWorldVariationGPU>(
                family.program.variations.size(),
                variationBytes
            ) ||
            !checkedByteCount<std::uint32_t>(
                family.program.categoricalValues.size(),
                categoricalBytes
            ) ||
            !checkedMultiply(capacity, 1u, instanceElements) ||
            !checkedMultiply(capacity, assetCount, assetElements) ||
            !checkedMultiply(capacity, sensorCount, sensorElements) ||
            !checkedMultiply(
                capacity,
                appearanceCount,
                appearanceElements
            ) ||
            !checkedByteCount<MRWorldInstanceHeaderGPU>(
                instanceElements,
                layout.instancePrivateBytes
            ) ||
            !checkedByteCount<MRWorldAssetInstanceGPU>(
                assetElements,
                layout.assetPrivateBytes
            ) ||
            !checkedByteCount<MRWorldSensorInstanceGPU>(
                sensorElements,
                layout.sensorPrivateBytes
            ) ||
            !checkedByteCount<MRWorldAppearanceInstanceGPU>(
                appearanceElements,
                layout.appearancePrivateBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "world-family byte layout overflowed"
            );
        }
        layout.immutablePrivateBytes =
            baseAssetBytes + baseSensorBytes + baseAppearanceBytes +
            variationBytes + categoricalBytes;
        const std::size_t maxBufferLength = static_cast<std::size_t>(
            state_->device.maxBufferLength
        );
        if (layout.instancePrivateBytes > maxBufferLength ||
            layout.assetPrivateBytes > maxBufferLength ||
            layout.sensorPrivateBytes > maxBufferLength ||
            layout.appearancePrivateBytes > maxBufferLength ||
            baseAssetBytes > maxBufferLength ||
            baseSensorBytes > maxBufferLength ||
            baseAppearanceBytes > maxBufferLength ||
            variationBytes > maxBufferLength ||
            categoricalBytes > maxBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "one or more world-family streams exceed "
                "device.maxBufferLength"
            );
        }

        FamilyBuffers candidate;
        candidate.baseAssets = makePrivateBuffer(
            state_->device,
            baseAssetBytes,
            @"MetalRobo base world assets"
        );
        candidate.baseSensors = makePrivateBuffer(
            state_->device,
            baseSensorBytes,
            @"MetalRobo base world sensors"
        );
        candidate.baseAppearances = makePrivateBuffer(
            state_->device,
            baseAppearanceBytes,
            @"MetalRobo base world appearances"
        );
        candidate.variations = makePrivateBuffer(
            state_->device,
            variationBytes,
            @"MetalRobo world variations"
        );
        candidate.categoricalValues = makePrivateBuffer(
            state_->device,
            categoricalBytes,
            @"MetalRobo categorical values"
        );
        candidate.uniforms = makeSharedBuffer(
            state_->device,
            sizeof(MRWorldFamilySampleUniformsGPU),
            @"MetalRobo world-family uniforms"
        );
        candidate.instances = makePrivateBuffer(
            state_->device,
            layout.instancePrivateBytes,
            @"MetalRobo world instance headers"
        );
        candidate.assets = makePrivateBuffer(
            state_->device,
            layout.assetPrivateBytes,
            @"MetalRobo world asset instances"
        );
        candidate.sensors = makePrivateBuffer(
            state_->device,
            layout.sensorPrivateBytes,
            @"MetalRobo world sensor instances"
        );
        candidate.appearances = makePrivateBuffer(
            state_->device,
            layout.appearancePrivateBytes,
            @"MetalRobo world appearance instances"
        );
        if (candidate.baseAssets == nil ||
            candidate.baseSensors == nil ||
            candidate.baseAppearances == nil ||
            candidate.variations == nil ||
            candidate.categoricalValues == nil ||
            candidate.uniforms == nil ||
            candidate.instances == nil ||
            candidate.assets == nil ||
            candidate.sensors == nil ||
            candidate.appearances == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "failed to allocate persistent world-family buffers"
            );
        }

        id<MTLBuffer> uploadBaseAssets = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldAssetInstanceGPU>{base.assets},
            @"MetalRobo upload base assets"
        );
        id<MTLBuffer> uploadBaseSensors = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldSensorInstanceGPU>{base.sensors},
            @"MetalRobo upload base sensors"
        );
        id<MTLBuffer> uploadBaseAppearances = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldAppearanceInstanceGPU>{
                base.appearances
            },
            @"MetalRobo upload base appearances"
        );
        id<MTLBuffer> uploadVariations = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldVariationGPU>{
                family.program.variations
            },
            @"MetalRobo upload variations"
        );
        id<MTLBuffer> uploadCategorical = makeUploadBuffer(
            state_->device,
            std::span<const std::uint32_t>{
                family.program.categoricalValues
            },
            @"MetalRobo upload categorical values"
        );
        if (uploadBaseAssets == nil ||
            uploadBaseSensors == nil ||
            uploadBaseAppearances == nil ||
            uploadVariations == nil ||
            uploadCategorical == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "failed to allocate world-family upload buffers"
            );
        }

        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "failed to create world-family upload command"
            );
        }
        const auto copyUpload = ^(
            id<MTLBuffer> source,
            id<MTLBuffer> destination,
            std::size_t logicalBytes
        ) {
            [blit copyFromBuffer:source
                   sourceOffset:0u
                       toBuffer:destination
              destinationOffset:0u
                           size:static_cast<NSUInteger>(
                               std::max<std::size_t>(
                                   logicalBytes,
                                   kMinimumAllocationBytes
                               )
                           )];
        };
        copyUpload(
            uploadBaseAssets,
            candidate.baseAssets,
            baseAssetBytes
        );
        copyUpload(
            uploadBaseSensors,
            candidate.baseSensors,
            baseSensorBytes
        );
        copyUpload(
            uploadBaseAppearances,
            candidate.baseAppearances,
            baseAppearanceBytes
        );
        copyUpload(
            uploadVariations,
            candidate.variations,
            variationBytes
        );
        copyUpload(
            uploadCategorical,
            candidate.categoricalValues,
            categoricalBytes
        );
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "world-family upload failed: " +
                    describeError(command.error)
            );
        }

        state_->buffers = std::move(candidate);
        state_->layout = layout;
        state_->instanceFlags = family.program.instanceFlags;
        state_->familyFingerprint = family.fingerprint;
        state_->compiled = true;
        state_->stats.compileCount += 1u;
        state_->stats.retainedPrivateBytes =
            layout.totalPrivateBytes();
        state_->stats.activeInstanceCount = 0u;
        state_->stats.familyFingerprint = family.fingerprint;
        diagnostics.layout = layout;
        diagnostics.deviceName = nsString(state_->device.name);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalBufferFailure,
            "host allocation failed while compiling world family"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldFamilyDiagnostics MetalWorldFamilyContext::sample(
    const std::uint32_t instanceCount,
    const std::uint64_t seed
) {
    MetalWorldFamilyDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            "world-family context has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        diagnostics.familyFingerprint = state_->familyFingerprint;
        diagnostics.deviceName = nsString(state_->device.name);
        diagnostics.layout = state_->layout;
        if (!state_->compiled) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::notCompiled,
                "compile a world family before sampling it"
            );
        }
        if (instanceCount == 0u ||
            instanceCount > state_->layout.capacity) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidCapacity,
                "requested instance count is zero or exceeds capacity"
            );
        }

        MRWorldFamilySampleUniformsGPU uniforms{};
        uniforms.counts = {
            instanceCount,
            state_->layout.assetCountPerInstance,
            state_->layout.sensorCountPerInstance,
            state_->layout.appearanceCountPerInstance,
        };
        uniforms.program = {
            state_->layout.variationCount,
            state_->layout.categoricalValueCount,
            state_->instanceFlags,
            MR_WORLD_COMPILER_ABI_VERSION,
        };
        uniforms.identity = {
            low32(seed),
            high32(seed),
            low32(state_->familyFingerprint),
            high32(state_->familyFingerprint),
        };
        std::memcpy(
            state_->buffers.uniforms.contents,
            &uniforms,
            sizeof(uniforms)
        );

        const auto start = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "failed to create world-family sample command"
            );
        }
        encoder.label = @"MetalRobo sample world family";
        [encoder setComputePipelineState:state_->pipeline];
        [encoder setBuffer:state_->buffers.baseAssets
                    offset:0u
                   atIndex:0u];
        [encoder setBuffer:state_->buffers.baseSensors
                    offset:0u
                   atIndex:1u];
        [encoder setBuffer:state_->buffers.baseAppearances
                    offset:0u
                   atIndex:2u];
        [encoder setBuffer:state_->buffers.variations
                    offset:0u
                   atIndex:3u];
        [encoder setBuffer:state_->buffers.categoricalValues
                    offset:0u
                   atIndex:4u];
        [encoder setBuffer:state_->buffers.uniforms
                    offset:0u
                   atIndex:5u];
        [encoder setBuffer:state_->buffers.instances
                    offset:0u
                   atIndex:6u];
        [encoder setBuffer:state_->buffers.assets
                    offset:0u
                   atIndex:7u];
        [encoder setBuffer:state_->buffers.sensors
                    offset:0u
                   atIndex:8u];
        [encoder setBuffer:state_->buffers.appearances
                    offset:0u
                   atIndex:9u];
        const NSUInteger threadsPerGroup = std::min<NSUInteger>(
            state_->pipeline.maxTotalThreadsPerThreadgroup,
            std::max<NSUInteger>(
                state_->pipeline.threadExecutionWidth,
                32u
            )
        );
        [encoder dispatchThreads:MTLSizeMake(instanceCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(
                threadsPerGroup,
                1u,
                1u
            )];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "world-family sample failed: " +
                    describeError(command.error)
            );
        }

        state_->layout.activeInstanceCount = instanceCount;
        state_->stats.activeInstanceCount = instanceCount;
        state_->stats.sampleCount += 1u;
        diagnostics.layout = state_->layout;
        return diagnostics;
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldFamilyDiagnostics MetalWorldFamilyContext::readback(
    WorldInstanceBatch& output
) {
    MetalWorldFamilyDiagnostics diagnostics;
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            "world-family context has no state"
        );
    }
    try {
        const std::lock_guard lock(state_->mutex);
        diagnostics.familyFingerprint = state_->familyFingerprint;
        diagnostics.deviceName = nsString(state_->device.name);
        diagnostics.layout = state_->layout;
        if (!state_->compiled ||
            state_->layout.activeInstanceCount == 0u) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::notCompiled,
                "sample the compiled world family before readback"
            );
        }

        const std::size_t instanceCount =
            state_->layout.activeInstanceCount;
        const std::size_t assetElements =
            instanceCount * state_->layout.assetCountPerInstance;
        const std::size_t sensorElements =
            instanceCount * state_->layout.sensorCountPerInstance;
        const std::size_t appearanceElements =
            instanceCount *
            state_->layout.appearanceCountPerInstance;
        const std::size_t instanceBytes =
            instanceCount * sizeof(MRWorldInstanceHeaderGPU);
        const std::size_t assetBytes =
            assetElements * sizeof(MRWorldAssetInstanceGPU);
        const std::size_t sensorBytes =
            sensorElements * sizeof(MRWorldSensorInstanceGPU);
        const std::size_t appearanceBytes =
            appearanceElements *
            sizeof(MRWorldAppearanceInstanceGPU);

        id<MTLBuffer> instanceReadback = makeSharedBuffer(
            state_->device,
            instanceBytes,
            @"MetalRobo instance readback"
        );
        id<MTLBuffer> assetReadback = makeSharedBuffer(
            state_->device,
            assetBytes,
            @"MetalRobo asset readback"
        );
        id<MTLBuffer> sensorReadback = makeSharedBuffer(
            state_->device,
            sensorBytes,
            @"MetalRobo sensor readback"
        );
        id<MTLBuffer> appearanceReadback = makeSharedBuffer(
            state_->device,
            appearanceBytes,
            @"MetalRobo appearance readback"
        );
        if (instanceReadback == nil || assetReadback == nil ||
            sensorReadback == nil || appearanceReadback == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "failed to allocate world-family readback buffers"
            );
        }

        const auto start = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command =
            [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit =
            [command blitCommandEncoder];
        if (command == nil || blit == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "failed to create world-family readback command"
            );
        }
        const auto copyOutput = ^(
            id<MTLBuffer> source,
            id<MTLBuffer> destination,
            std::size_t bytes
        ) {
            if (bytes != 0u) {
                [blit copyFromBuffer:source
                       sourceOffset:0u
                           toBuffer:destination
                  destinationOffset:0u
                               size:static_cast<NSUInteger>(bytes)];
            }
        };
        copyOutput(
            state_->buffers.instances,
            instanceReadback,
            instanceBytes
        );
        copyOutput(
            state_->buffers.assets,
            assetReadback,
            assetBytes
        );
        copyOutput(
            state_->buffers.sensors,
            sensorReadback,
            sensorBytes
        );
        copyOutput(
            state_->buffers.appearances,
            appearanceReadback,
            appearanceBytes
        );
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "world-family readback failed: " +
                    describeError(command.error)
            );
        }

        WorldInstanceBatch staged;
        staged.familyFingerprint = state_->familyFingerprint;
        staged.instances.resize(instanceCount);
        staged.assets.resize(assetElements);
        staged.sensors.resize(sensorElements);
        staged.appearances.resize(appearanceElements);
        copySharedBuffer(staged.instances, instanceReadback);
        copySharedBuffer(staged.assets, assetReadback);
        copySharedBuffer(staged.sensors, sensorReadback);
        copySharedBuffer(staged.appearances, appearanceReadback);
        std::string reason;
        if (!staged.valid(&reason)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "GPU produced an invalid world batch: " + reason
            );
        }
        output = std::move(staged);
        state_->stats.readbackCount += 1u;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalBufferFailure,
            "host allocation failed during world-family readback"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldFamilyLayout MetalWorldFamilyContext::layout() const noexcept {
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

MetalWorldFamilyStats MetalWorldFamilyContext::stats() const noexcept {
    if (state_ == nullptr) {
        return {};
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->stats;
    } catch (...) {
        return {};
    }
}

void* MetalWorldFamilyContext::nativeBuffer(
    const MetalWorldFamilyBuffer buffer
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
        case MetalWorldFamilyBuffer::instanceHeaders:
            selected = state_->buffers.instances;
            break;
        case MetalWorldFamilyBuffer::assetInstances:
            selected = state_->buffers.assets;
            break;
        case MetalWorldFamilyBuffer::sensorInstances:
            selected = state_->buffers.sensors;
            break;
        case MetalWorldFamilyBuffer::appearanceInstances:
            selected = state_->buffers.appearances;
            break;
        }
        return (__bridge void*)selected;
    } catch (...) {
        return nullptr;
    }
}

const char* metalWorldFamilyStatusName(
    const MetalWorldFamilyStatus status
) noexcept {
    switch (status) {
    case MetalWorldFamilyStatus::success:
        return "success";
    case MetalWorldFamilyStatus::invalidFamily:
        return "invalid_family";
    case MetalWorldFamilyStatus::invalidCapacity:
        return "invalid_capacity";
    case MetalWorldFamilyStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalWorldFamilyStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalWorldFamilyStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalWorldFamilyStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalWorldFamilyStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalWorldFamilyStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalWorldFamilyStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalWorldFamilyStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalWorldFamilyStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalWorldFamilyStatus::notCompiled:
        return "not_compiled";
    case MetalWorldFamilyStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
