#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/WorldPack.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
    __strong id<MTLBuffer> assetBindings = nil;
    __strong id<MTLBuffer> bindingIndices = nil;
    __strong id<MTLBuffer> baseQ = nil;
    __strong id<MTLBuffer> baseV = nil;
    __strong id<MTLBuffer> baseSceneBodies = nil;
    __strong id<MTLBuffer> bodyToScene = nil;
    __strong id<MTLBuffer> bodyProperties = nil;
    __strong id<MTLBuffer> uniforms = nil;
    __strong id<MTLBuffer> materializeUniforms = nil;
    __strong id<MTLBuffer> instances = nil;
    __strong id<MTLBuffer> assets = nil;
    __strong id<MTLBuffer> sensors = nil;
    __strong id<MTLBuffer> appearances = nil;
    __strong id<MTLBuffer> scenarioHeaders = nil;
    __strong id<MTLBuffer> scenarioValues = nil;
    __strong id<MTLBuffer> adaptiveUniforms = nil;
    __strong id<MTLBuffer> alignmentParticles = nil;
    __strong id<MTLBuffer> alignmentQuantiles = nil;
    __strong id<MTLBuffer> feedbackRegions = nil;
    __strong id<MTLBuffer> feedbackBounds = nil;
    __strong id<MTLBuffer> resetQ = nil;
    __strong id<MTLBuffer> resetV = nil;
    __strong id<MTLBuffer> resetSceneBodies = nil;
    __strong id<MTLBuffer> bodyParameters = nil;
    __strong id<MTLBuffer> controllerParameters = nil;
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

bool appendAlignedReadbackRegion(
    const std::size_t bytes,
    std::size_t& cursor,
    std::size_t& offset
) {
    constexpr std::size_t kAlignment = 256u;
    const std::size_t remainder = cursor % kAlignment;
    const std::size_t padding =
        remainder == 0u ? 0u : kAlignment - remainder;
    if (padding >
            std::numeric_limits<std::size_t>::max() - cursor) {
        return false;
    }
    offset = cursor + padding;
    if (bytes >
            std::numeric_limits<std::size_t>::max() - offset) {
        return false;
    }
    cursor = offset + bytes;
    return cursor <= std::numeric_limits<NSUInteger>::max();
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

template <typename T>
id<MTLBuffer> makePrivateTypedBuffer(
    id<MTLDevice> device,
    const std::size_t logicalBytes,
    NSString* label
) {
    return makePrivateBuffer(
        device,
        std::max<std::size_t>(logicalBytes, sizeof(T)),
        label
    );
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
    id<MTLBuffer> buffer,
    const std::size_t offset
) {
    if (!destination.empty()) {
        const auto* contents =
            static_cast<const std::byte*>(buffer.contents);
        std::memcpy(
            destination.data(),
            contents + offset,
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
    __strong id<MTLComputePipelineState> materializePipeline = nil;
    FamilyBuffers buffers{};
    MetalWorldFamilyLayout layout{};
    MetalWorldFamilyStats stats{};
    std::uint32_t instanceFlags = 0u;
    std::uint64_t familyFingerprint = 0u;
    ScenarioSchema scenarioSchema;
    CompiledWorldSamplingProgram samplingProgram;
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
    id<MTLFunction> materializeFunction = [library
        newFunctionWithName:@"mr_world_family_materialize_physics"];
    if (materializeFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalLibraryFailure,
            "metallib does not contain "
            "mr_world_family_materialize_physics"
        );
    }
    error = nil;
    id<MTLComputePipelineState> materializePipeline = [device
        newComputePipelineStateWithFunction:materializeFunction
                                       error:&error];
    if (materializePipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalPipelineFailure,
            "failed to create world-family physics materialization "
            "pipeline: " + describeError(error)
        );
    }

    state.device = device;
    state.queue = queue;
    state.library = library;
    state.pipeline = pipeline;
    state.materializePipeline = materializePipeline;
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
            target > MR_WORLD_TARGET_ASSET_COLLISION_ALTERNATIVE ||
            (target <= MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE &&
             index >= assetCount) ||
            ((target == MR_WORLD_TARGET_CLUTTER_SET ||
              target ==
                  MR_WORLD_TARGET_ASSET_COLLISION_ALTERNATIVE) &&
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

struct PhysicsBase {
    std::uint32_t primaryArticulation = MR_INVALID_INDEX;
    std::vector<float> q;
    std::vector<float> v;
    std::vector<MRBodyStateGPU> sceneBodies;
    std::vector<std::uint32_t> bodyToScene;
};

bool buildPhysicsBase(
    const WorldTemplate& worldTemplate,
    PhysicsBase& output,
    std::string& reason
) {
    const std::uint32_t robotAsset =
        worldTemplate.assetIndex(worldTemplate.task.robotAssetId);
    if (robotAsset == MR_INVALID_INDEX) {
        reason = "task robot asset is missing";
        return false;
    }
    const WorldAsset& robot = worldTemplate.assets[robotAsset];
    if (robot.articulationIndex == MR_INVALID_INDEX ||
        robot.articulationIndex >=
            worldTemplate.engineModel.articulations.size()) {
        reason = "task robot has no executable articulation";
        return false;
    }
    const EngineModel& model = worldTemplate.engineModel;
    const MRArticulationGPU& articulation =
        model.articulations[robot.articulationIndex];
    if (articulation.qOffset > model.defaultQ.size() ||
        articulation.nq > model.defaultQ.size() -
            articulation.qOffset ||
        articulation.vOffset > model.defaultV.size() ||
        articulation.nv > model.defaultV.size() -
            articulation.vOffset) {
        reason = "task robot reset range is invalid";
        return false;
    }

    PhysicsBase staged;
    staged.primaryArticulation = robot.articulationIndex;
    staged.q.assign(
        model.defaultQ.begin() + articulation.qOffset,
        model.defaultQ.begin() + articulation.qOffset +
            articulation.nq
    );
    staged.v.assign(
        model.defaultV.begin() + articulation.vOffset,
        model.defaultV.begin() + articulation.vOffset +
            articulation.nv
    );
    staged.bodyToScene.assign(
        model.bodies.size(),
        MR_INVALID_INDEX
    );
    for (std::uint32_t body = 0u;
         body < model.bodies.size();
         ++body) {
        const MRBodyPropertiesGPU& properties = model.bodies[body];
        if (properties.articulationIndex != MR_INVALID_INDEX) {
            continue;
        }
        staged.bodyToScene[body] =
            static_cast<std::uint32_t>(staged.sceneBodies.size());
        MRBodyStateGPU state{};
        state.position.w = 1.0f;
        state.orientation.w = 1.0f;
        state.flagsAndIndices[0] = properties.motionType;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = body;
        staged.sceneBodies.push_back(state);
    }
    for (const WorldAsset& asset : worldTemplate.assets) {
        for (const std::uint32_t body : asset.bodyIndices) {
            if (body >= staged.bodyToScene.size()) {
                reason = "asset body mapping exceeds physics topology";
                return false;
            }
            const std::uint32_t localScene =
                staged.bodyToScene[body];
            if (localScene == MR_INVALID_INDEX) {
                continue;
            }
            MRBodyStateGPU& state =
                staged.sceneBodies[localScene];
            state.position = {
                asset.initialPose.position.x,
                asset.initialPose.position.y,
                asset.initialPose.position.z,
                1.0f,
            };
            state.orientation = asset.initialPose.orientation;
        }
    }
    output = std::move(staged);
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

        const ScenarioSchema scenarioSchema =
            compileScenarioSchema(family);
        CompiledWorldSamplingProgram samplingProgram;
        if (!scenarioSchema.valid(&reason) ||
            !compileWorldSamplingProgram(
                scenarioSchema,
                nullptr,
                nullptr,
                samplingProgram,
                &reason
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                "failed to compile scenario schema: " + reason
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
        const std::size_t bodyCount =
            family.worldTemplate.engineModel.bodies.size();
        const std::size_t articulationCount =
            family.worldTemplate.engineModel.articulations.size();
        constexpr std::size_t addressLimit =
            std::numeric_limits<std::uint32_t>::max();
        if (assetCount > addressLimit ||
            sensorCount > addressLimit ||
            appearanceCount > addressLimit ||
            bodyCount > addressLimit ||
            articulationCount > addressLimit ||
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
        PhysicsBase physicsBase;
        if (!buildPhysicsBase(
                family.worldTemplate,
                physicsBase,
                batchReason
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                "failed to compile physics reset base: " +
                    batchReason
            );
        }
        if (physicsBase.sceneBodies.size() > addressLimit ||
            capacity > addressLimit /
                std::max<std::size_t>(bodyCount, 1u) ||
            capacity > addressLimit /
                std::max<std::size_t>(articulationCount, 1u) ||
            capacity > addressLimit /
                std::max<std::size_t>(physicsBase.q.size(), 1u) ||
            capacity > addressLimit /
                std::max<std::size_t>(physicsBase.v.size(), 1u) ||
            capacity > addressLimit /
                std::max<std::size_t>(
                    physicsBase.sceneBodies.size(),
                    1u
                )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::capacityOverflow,
                "world-family physics streams exceed 32-bit GPU "
                "addressing"
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
        layout.assetBindingCount = static_cast<std::uint32_t>(
            family.worldTemplate.assetBindings.size()
        );
        layout.bindingIndexCount = static_cast<std::uint32_t>(
            family.worldTemplate.bindingIndices.size()
        );
        layout.primaryArticulationIndex =
            physicsBase.primaryArticulation;
        layout.nq = static_cast<std::uint32_t>(physicsBase.q.size());
        layout.nv = static_cast<std::uint32_t>(physicsBase.v.size());
        layout.bodyCount = static_cast<std::uint32_t>(bodyCount);
        layout.sceneBodyCount = static_cast<std::uint32_t>(
            physicsBase.sceneBodies.size()
        );
        layout.articulationCount = static_cast<std::uint32_t>(
            articulationCount
        );

        std::size_t baseAssetBytes = 0u;
        std::size_t baseSensorBytes = 0u;
        std::size_t baseAppearanceBytes = 0u;
        std::size_t variationBytes = 0u;
        std::size_t categoricalBytes = 0u;
        std::size_t assetBindingBytes = 0u;
        std::size_t bindingIndexBytes = 0u;
        std::size_t baseQBytes = 0u;
        std::size_t baseVBytes = 0u;
        std::size_t baseSceneBodyBytes = 0u;
        std::size_t bodyToSceneBytes = 0u;
        std::size_t bodyPropertyBytes = 0u;
        std::size_t instanceElements = 0u;
        std::size_t assetElements = 0u;
        std::size_t sensorElements = 0u;
        std::size_t appearanceElements = 0u;
        std::size_t scenarioHeaderElements = 0u;
        std::size_t scenarioValueElements = 0u;
        std::size_t resetQElements = 0u;
        std::size_t resetVElements = 0u;
        std::size_t resetSceneBodyElements = 0u;
        std::size_t bodyParameterElements = 0u;
        std::size_t controllerParameterElements = 0u;
        std::size_t resetQBytes = 0u;
        std::size_t resetVBytes = 0u;
        std::size_t resetSceneBodyBytes = 0u;
        std::size_t bodyParameterBytes = 0u;
        std::size_t controllerParameterBytes = 0u;
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
            !checkedByteCount<MRWorldAssetBindingGPU>(
                family.worldTemplate.assetBindings.size(),
                assetBindingBytes
            ) ||
            !checkedByteCount<std::uint32_t>(
                family.worldTemplate.bindingIndices.size(),
                bindingIndexBytes
            ) ||
            !checkedByteCount<float>(
                physicsBase.q.size(),
                baseQBytes
            ) ||
            !checkedByteCount<float>(
                physicsBase.v.size(),
                baseVBytes
            ) ||
            !checkedByteCount<MRBodyStateGPU>(
                physicsBase.sceneBodies.size(),
                baseSceneBodyBytes
            ) ||
            !checkedByteCount<std::uint32_t>(
                physicsBase.bodyToScene.size(),
                bodyToSceneBytes
            ) ||
            !checkedByteCount<MRBodyPropertiesGPU>(
                family.worldTemplate.engineModel.bodies.size(),
                bodyPropertyBytes
            ) ||
            !checkedMultiply(capacity, 1u, instanceElements) ||
            !checkedMultiply(capacity, assetCount, assetElements) ||
            !checkedMultiply(capacity, sensorCount, sensorElements) ||
            !checkedMultiply(
                capacity,
                appearanceCount,
                appearanceElements
            ) ||
            !checkedMultiply(
                capacity,
                1u,
                scenarioHeaderElements
            ) ||
            !checkedMultiply(
                capacity,
                family.program.variations.size(),
                scenarioValueElements
            ) ||
            !checkedMultiply(
                capacity,
                physicsBase.q.size(),
                resetQElements
            ) ||
            !checkedMultiply(
                capacity,
                physicsBase.v.size(),
                resetVElements
            ) ||
            !checkedMultiply(
                capacity,
                physicsBase.sceneBodies.size(),
                resetSceneBodyElements
            ) ||
            !checkedMultiply(
                capacity,
                bodyCount,
                bodyParameterElements
            ) ||
            !checkedMultiply(
                capacity,
                articulationCount,
                controllerParameterElements
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
            ) ||
            !checkedByteCount<MRWorldScenarioHeaderGPU>(
                scenarioHeaderElements,
                layout.scenarioHeaderPrivateBytes
            ) ||
            !checkedByteCount<MRWorldScenarioValueGPU>(
                scenarioValueElements,
                layout.scenarioValuePrivateBytes
            ) ||
            !checkedByteCount<float>(
                resetQElements,
                resetQBytes
            ) ||
            !checkedByteCount<float>(
                resetVElements,
                resetVBytes
            ) ||
            !checkedByteCount<MRBodyStateGPU>(
                resetSceneBodyElements,
                resetSceneBodyBytes
            ) ||
            !checkedByteCount<MRWorldBodyParametersGPU>(
                bodyParameterElements,
                bodyParameterBytes
            ) ||
            !checkedByteCount<MRWorldControllerParametersGPU>(
                controllerParameterElements,
                controllerParameterBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "world-family byte layout overflowed"
            );
        }
        layout.immutablePrivateBytes =
            baseAssetBytes + baseSensorBytes + baseAppearanceBytes +
            variationBytes + categoricalBytes + assetBindingBytes +
            bindingIndexBytes + baseQBytes + baseVBytes +
            baseSceneBodyBytes + bodyToSceneBytes +
            bodyPropertyBytes;
        layout.physicsResetPrivateBytes =
            resetQBytes + resetVBytes + resetSceneBodyBytes +
            bodyParameterBytes + controllerParameterBytes;
        const std::size_t maxBufferLength = static_cast<std::size_t>(
            state_->device.maxBufferLength
        );
        if (layout.instancePrivateBytes > maxBufferLength ||
            layout.assetPrivateBytes > maxBufferLength ||
            layout.sensorPrivateBytes > maxBufferLength ||
            layout.appearancePrivateBytes > maxBufferLength ||
            layout.scenarioHeaderPrivateBytes > maxBufferLength ||
            layout.scenarioValuePrivateBytes > maxBufferLength ||
            baseAssetBytes > maxBufferLength ||
            baseSensorBytes > maxBufferLength ||
            baseAppearanceBytes > maxBufferLength ||
            variationBytes > maxBufferLength ||
            categoricalBytes > maxBufferLength ||
            assetBindingBytes > maxBufferLength ||
            bindingIndexBytes > maxBufferLength ||
            baseQBytes > maxBufferLength ||
            baseVBytes > maxBufferLength ||
            baseSceneBodyBytes > maxBufferLength ||
            bodyToSceneBytes > maxBufferLength ||
            bodyPropertyBytes > maxBufferLength ||
            resetQBytes > maxBufferLength ||
            resetVBytes > maxBufferLength ||
            resetSceneBodyBytes > maxBufferLength ||
            bodyParameterBytes > maxBufferLength ||
            controllerParameterBytes > maxBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "one or more world-family streams exceed "
                "device.maxBufferLength"
            );
        }

        FamilyBuffers candidate;
        candidate.baseAssets = makePrivateTypedBuffer<MRWorldAssetInstanceGPU>(
            state_->device,
            baseAssetBytes,
            @"MetalRobo base world assets"
        );
        candidate.baseSensors = makePrivateTypedBuffer<MRWorldSensorInstanceGPU>(
            state_->device,
            baseSensorBytes,
            @"MetalRobo base world sensors"
        );
        candidate.baseAppearances = makePrivateTypedBuffer<MRWorldAppearanceInstanceGPU>(
            state_->device,
            baseAppearanceBytes,
            @"MetalRobo base world appearances"
        );
        candidate.variations = makePrivateTypedBuffer<MRWorldVariationGPU>(
            state_->device,
            variationBytes,
            @"MetalRobo world variations"
        );
        candidate.categoricalValues = makePrivateTypedBuffer<std::uint32_t>(
            state_->device,
            categoricalBytes,
            @"MetalRobo categorical values"
        );
        candidate.assetBindings = makePrivateTypedBuffer<MRWorldAssetBindingGPU>(
            state_->device,
            assetBindingBytes,
            @"MetalRobo world asset bindings"
        );
        candidate.bindingIndices = makePrivateTypedBuffer<std::uint32_t>(
            state_->device,
            bindingIndexBytes,
            @"MetalRobo world binding indices"
        );
        candidate.baseQ = makePrivateTypedBuffer<float>(
            state_->device,
            baseQBytes,
            @"MetalRobo world-family base q"
        );
        candidate.baseV = makePrivateTypedBuffer<float>(
            state_->device,
            baseVBytes,
            @"MetalRobo world-family base v"
        );
        candidate.baseSceneBodies = makePrivateTypedBuffer<MRBodyStateGPU>(
            state_->device,
            baseSceneBodyBytes,
            @"MetalRobo world-family base scene bodies"
        );
        candidate.bodyToScene = makePrivateTypedBuffer<std::uint32_t>(
            state_->device,
            bodyToSceneBytes,
            @"MetalRobo world-family body-to-scene map"
        );
        candidate.bodyProperties = makePrivateTypedBuffer<MRBodyPropertiesGPU>(
            state_->device,
            bodyPropertyBytes,
            @"MetalRobo world-family body properties"
        );
        candidate.uniforms = makeSharedBuffer(
            state_->device,
            sizeof(MRWorldFamilySampleUniformsGPU),
            @"MetalRobo world-family uniforms"
        );
        candidate.materializeUniforms = makeSharedBuffer(
            state_->device,
            sizeof(MRWorldFamilyMaterializeUniformsGPU),
            @"MetalRobo world-family physics uniforms"
        );
        candidate.instances = makePrivateTypedBuffer<MRWorldInstanceHeaderGPU>(
            state_->device,
            layout.instancePrivateBytes,
            @"MetalRobo world instance headers"
        );
        candidate.assets = makePrivateTypedBuffer<MRWorldAssetInstanceGPU>(
            state_->device,
            layout.assetPrivateBytes,
            @"MetalRobo world asset instances"
        );
        candidate.sensors = makePrivateTypedBuffer<MRWorldSensorInstanceGPU>(
            state_->device,
            layout.sensorPrivateBytes,
            @"MetalRobo world sensor instances"
        );
        candidate.appearances = makePrivateTypedBuffer<MRWorldAppearanceInstanceGPU>(
            state_->device,
            layout.appearancePrivateBytes,
            @"MetalRobo world appearance instances"
        );
        candidate.scenarioHeaders = makePrivateTypedBuffer<MRWorldScenarioHeaderGPU>(
            state_->device,
            layout.scenarioHeaderPrivateBytes,
            @"MetalRobo world scenario headers"
        );
        candidate.scenarioValues = makePrivateTypedBuffer<MRWorldScenarioValueGPU>(
            state_->device,
            layout.scenarioValuePrivateBytes,
            @"MetalRobo world scenario values"
        );
        candidate.adaptiveUniforms = makeSharedBuffer(
            state_->device,
            sizeof(MRWorldAdaptiveSampleUniformsGPU),
            @"MetalRobo adaptive sampling uniforms"
        );
        candidate.alignmentParticles = makePrivateTypedBuffer<MRWorldAlignmentParticleGPU>(
            state_->device,
            sizeof(MRWorldAlignmentParticleGPU),
            @"MetalRobo alignment particles"
        );
        candidate.alignmentQuantiles = makePrivateTypedBuffer<float>(
            state_->device,
            0u,
            @"MetalRobo alignment quantiles"
        );
        candidate.feedbackRegions = makePrivateTypedBuffer<MRWorldFeedbackRegionGPU>(
            state_->device,
            sizeof(MRWorldFeedbackRegionGPU),
            @"MetalRobo feedback regions"
        );
        candidate.feedbackBounds = makePrivateTypedBuffer<mr_float4>(
            state_->device,
            0u,
            @"MetalRobo feedback bounds"
        );
        candidate.resetQ = makePrivateTypedBuffer<float>(
            state_->device,
            resetQBytes,
            @"MetalRobo world-family reset q"
        );
        candidate.resetV = makePrivateTypedBuffer<float>(
            state_->device,
            resetVBytes,
            @"MetalRobo world-family reset v"
        );
        candidate.resetSceneBodies = makePrivateTypedBuffer<MRBodyStateGPU>(
            state_->device,
            resetSceneBodyBytes,
            @"MetalRobo world-family reset scene bodies"
        );
        candidate.bodyParameters = makePrivateTypedBuffer<MRWorldBodyParametersGPU>(
            state_->device,
            bodyParameterBytes,
            @"MetalRobo world-family body parameters"
        );
        candidate.controllerParameters = makePrivateTypedBuffer<MRWorldControllerParametersGPU>(
            state_->device,
            controllerParameterBytes,
            @"MetalRobo world-family controller parameters"
        );
        if (candidate.baseAssets == nil ||
            candidate.baseSensors == nil ||
            candidate.baseAppearances == nil ||
            candidate.variations == nil ||
            candidate.categoricalValues == nil ||
            candidate.assetBindings == nil ||
            candidate.bindingIndices == nil ||
            candidate.baseQ == nil ||
            candidate.baseV == nil ||
            candidate.baseSceneBodies == nil ||
            candidate.bodyToScene == nil ||
            candidate.bodyProperties == nil ||
            candidate.uniforms == nil ||
            candidate.materializeUniforms == nil ||
            candidate.instances == nil ||
            candidate.assets == nil ||
            candidate.sensors == nil ||
            candidate.appearances == nil ||
            candidate.scenarioHeaders == nil ||
            candidate.scenarioValues == nil ||
            candidate.adaptiveUniforms == nil ||
            candidate.alignmentParticles == nil ||
            candidate.alignmentQuantiles == nil ||
            candidate.feedbackRegions == nil ||
            candidate.feedbackBounds == nil ||
            candidate.resetQ == nil ||
            candidate.resetV == nil ||
            candidate.resetSceneBodies == nil ||
            candidate.bodyParameters == nil ||
            candidate.controllerParameters == nil) {
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
        id<MTLBuffer> uploadAssetBindings = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldAssetBindingGPU>{
                family.worldTemplate.assetBindings
            },
            @"MetalRobo upload asset bindings"
        );
        id<MTLBuffer> uploadBindingIndices = makeUploadBuffer(
            state_->device,
            std::span<const std::uint32_t>{
                family.worldTemplate.bindingIndices
            },
            @"MetalRobo upload binding indices"
        );
        id<MTLBuffer> uploadBaseQ = makeUploadBuffer(
            state_->device,
            std::span<const float>{physicsBase.q},
            @"MetalRobo upload base q"
        );
        id<MTLBuffer> uploadBaseV = makeUploadBuffer(
            state_->device,
            std::span<const float>{physicsBase.v},
            @"MetalRobo upload base v"
        );
        id<MTLBuffer> uploadBaseSceneBodies = makeUploadBuffer(
            state_->device,
            std::span<const MRBodyStateGPU>{
                physicsBase.sceneBodies
            },
            @"MetalRobo upload base scene bodies"
        );
        id<MTLBuffer> uploadBodyToScene = makeUploadBuffer(
            state_->device,
            std::span<const std::uint32_t>{
                physicsBase.bodyToScene
            },
            @"MetalRobo upload body-to-scene map"
        );
        id<MTLBuffer> uploadBodyProperties = makeUploadBuffer(
            state_->device,
            std::span<const MRBodyPropertiesGPU>{
                family.worldTemplate.engineModel.bodies
            },
            @"MetalRobo upload body properties"
        );
        if (uploadBaseAssets == nil ||
            uploadBaseSensors == nil ||
            uploadBaseAppearances == nil ||
            uploadVariations == nil ||
            uploadCategorical == nil ||
            uploadAssetBindings == nil ||
            uploadBindingIndices == nil ||
            uploadBaseQ == nil ||
            uploadBaseV == nil ||
            uploadBaseSceneBodies == nil ||
            uploadBodyToScene == nil ||
            uploadBodyProperties == nil) {
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
        copyUpload(
            uploadAssetBindings,
            candidate.assetBindings,
            assetBindingBytes
        );
        copyUpload(
            uploadBindingIndices,
            candidate.bindingIndices,
            bindingIndexBytes
        );
        copyUpload(
            uploadBaseQ,
            candidate.baseQ,
            baseQBytes
        );
        copyUpload(
            uploadBaseV,
            candidate.baseV,
            baseVBytes
        );
        copyUpload(
            uploadBaseSceneBodies,
            candidate.baseSceneBodies,
            baseSceneBodyBytes
        );
        copyUpload(
            uploadBodyToScene,
            candidate.bodyToScene,
            bodyToSceneBytes
        );
        copyUpload(
            uploadBodyProperties,
            candidate.bodyProperties,
            bodyPropertyBytes
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
        state_->scenarioSchema = scenarioSchema;
        state_->samplingProgram = std::move(samplingProgram);
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
    return sample(
        instanceCount,
        seed,
        MR_WORLD_SAMPLING_COVERAGE,
        0u
    );
}

MetalWorldFamilyDiagnostics MetalWorldFamilyContext::sample(
    const std::uint32_t instanceCount,
    const std::uint64_t seed,
    const MRWorldSamplingMode mode,
    const std::uint64_t episodeCounterBase
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
        if (mode > MR_WORLD_SAMPLING_REPLAY) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                "world-family sampling mode is invalid"
            );
        }
        if (episodeCounterBase >
            std::numeric_limits<std::uint64_t>::max() -
                (static_cast<std::uint64_t>(instanceCount) - 1u)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "per-environment episode counters overflow uint64"
            );
        }
        if (mode == MR_WORLD_SAMPLING_REPLAY &&
            (state_->samplingProgram.alignmentParticles.empty() ||
             instanceCount >
                 state_->samplingProgram.alignmentParticles.size())) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidCapacity,
                "replay sampling requires one alignment particle per "
                "active environment"
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
        MRWorldAdaptiveSampleUniformsGPU adaptive{};
        adaptive.counts = {
            static_cast<std::uint32_t>(
                state_->samplingProgram.alignmentParticles.size()
            ),
            static_cast<std::uint32_t>(
                state_->samplingProgram.feedbackRegions.size()
            ),
            state_->layout.variationCount,
            mode,
        };
        adaptive.identity = {
            low32(episodeCounterBase),
            high32(episodeCounterBase),
            low32(state_->samplingProgram.alignmentFingerprint),
            high32(state_->samplingProgram.alignmentFingerprint),
        };
        adaptive.provenance = {
            low32(state_->samplingProgram.feedbackFingerprint),
            high32(state_->samplingProgram.feedbackFingerprint),
            0u,
            0u,
        };
        adaptive.mixture = {
            static_cast<float>(
                mode != MR_WORLD_SAMPLING_CURRICULUM
                ? 1.0
                : state_->samplingProgram.broadWeight
            ),
            static_cast<float>(
                mode != MR_WORLD_SAMPLING_CURRICULUM
                ? 0.0
                : state_->samplingProgram.failureWeight
            ),
            static_cast<float>(
                mode != MR_WORLD_SAMPLING_CURRICULUM
                ? 0.0
                : state_->samplingProgram.uncertaintyWeight
            ),
            mode == MR_WORLD_SAMPLING_REPLAY
                ? 0.0f
                : state_->samplingProgram.alignmentJitter,
        };
        adaptive.abi = {
            MR_R2S2R_ABI_VERSION,
            0u,
            0u,
            0u,
        };
        std::memcpy(
            state_->buffers.adaptiveUniforms.contents,
            &adaptive,
            sizeof(adaptive)
        );
        MRWorldFamilyMaterializeUniformsGPU materializeUniforms{};
        materializeUniforms.stateCounts = {
            instanceCount,
            state_->layout.nq,
            state_->layout.nv,
            state_->layout.sceneBodyCount,
        };
        materializeUniforms.topology = {
            state_->layout.bodyCount,
            state_->layout.articulationCount,
            state_->layout.assetCountPerInstance,
            state_->layout.primaryArticulationIndex,
        };
        materializeUniforms.identity = {
            MR_WORLD_COMPILER_ABI_VERSION,
            0u,
            0u,
            0u,
        };
        std::memcpy(
            state_->buffers.materializeUniforms.contents,
            &materializeUniforms,
            sizeof(materializeUniforms)
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
        [encoder setBuffer:state_->buffers.scenarioHeaders
                    offset:0u
                   atIndex:10u];
        [encoder setBuffer:state_->buffers.scenarioValues
                    offset:0u
                   atIndex:11u];
        [encoder setBuffer:state_->buffers.adaptiveUniforms
                    offset:0u
                   atIndex:12u];
        [encoder setBuffer:state_->buffers.alignmentParticles
                    offset:0u
                   atIndex:13u];
        [encoder setBuffer:state_->buffers.alignmentQuantiles
                    offset:0u
                   atIndex:14u];
        [encoder setBuffer:state_->buffers.feedbackRegions
                    offset:0u
                   atIndex:15u];
        [encoder setBuffer:state_->buffers.feedbackBounds
                    offset:0u
                   atIndex:16u];
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

        id<MTLComputeCommandEncoder> materializeEncoder =
            [command computeCommandEncoder];
        if (materializeEncoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "failed to create world-family physics materialization "
                "encoder"
            );
        }
        materializeEncoder.label =
            @"MetalRobo materialize family physics";
        [materializeEncoder
            setComputePipelineState:state_->materializePipeline];
        [materializeEncoder setBuffer:state_->buffers.baseQ
                              offset:0u
                             atIndex:0u];
        [materializeEncoder setBuffer:state_->buffers.baseV
                              offset:0u
                             atIndex:1u];
        [materializeEncoder
            setBuffer:state_->buffers.baseSceneBodies
               offset:0u
              atIndex:2u];
        [materializeEncoder setBuffer:state_->buffers.bodyToScene
                              offset:0u
                             atIndex:3u];
        [materializeEncoder setBuffer:state_->buffers.assetBindings
                              offset:0u
                             atIndex:4u];
        [materializeEncoder setBuffer:state_->buffers.bindingIndices
                              offset:0u
                             atIndex:5u];
        [materializeEncoder setBuffer:state_->buffers.assets
                              offset:0u
                             atIndex:6u];
        [materializeEncoder
            setBuffer:state_->buffers.materializeUniforms
               offset:0u
              atIndex:7u];
        [materializeEncoder setBuffer:state_->buffers.resetQ
                              offset:0u
                             atIndex:8u];
        [materializeEncoder setBuffer:state_->buffers.resetV
                              offset:0u
                             atIndex:9u];
        [materializeEncoder
            setBuffer:state_->buffers.resetSceneBodies
               offset:0u
              atIndex:10u];
        [materializeEncoder
            setBuffer:state_->buffers.bodyParameters
               offset:0u
              atIndex:11u];
        [materializeEncoder
            setBuffer:state_->buffers.controllerParameters
               offset:0u
              atIndex:12u];
        [materializeEncoder
            setBuffer:state_->buffers.bodyProperties
               offset:0u
              atIndex:13u];
        const NSUInteger materializeThreadsPerGroup =
            std::min<NSUInteger>(
                state_->materializePipeline.maxTotalThreadsPerThreadgroup,
                std::max<NSUInteger>(
                    state_->materializePipeline.threadExecutionWidth,
                    32u
                )
            );
        [materializeEncoder
            dispatchThreads:MTLSizeMake(instanceCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(
                materializeThreadsPerGroup,
                1u,
                1u
            )];
        [materializeEncoder endEncoding];
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

MetalWorldFamilyDiagnostics
MetalWorldFamilyContext::configureSamplingProgram(
    const ScenarioSchema& schema,
    const CompiledWorldSamplingProgram& program
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
                "compile a world family before configuring sampling"
            );
        }
        std::string reason;
        if (schema.fingerprint != state_->scenarioSchema.fingerprint ||
            !program.valid(schema, &reason)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::invalidFamily,
                reason.empty()
                    ? "sampling schema does not match the compiled family"
                    : std::move(reason)
            );
        }
        std::size_t particleBytes = 0u;
        std::size_t quantileBytes = 0u;
        std::size_t regionBytes = 0u;
        std::size_t boundBytes = 0u;
        if (!checkedByteCount<MRWorldAlignmentParticleGPU>(
                program.alignmentParticles.size(),
                particleBytes
            ) ||
            !checkedByteCount<float>(
                program.alignmentQuantiles.size(),
                quantileBytes
            ) ||
            !checkedByteCount<MRWorldFeedbackRegionGPU>(
                program.feedbackRegions.size(),
                regionBytes
            ) ||
            !checkedByteCount<mr_float4>(
                program.feedbackBounds.size(),
                boundBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "adaptive sampling program byte layout overflowed"
            );
        }
        const std::size_t maxBufferLength =
            static_cast<std::size_t>(
                state_->device.maxBufferLength
            );
        if (particleBytes > maxBufferLength ||
            quantileBytes > maxBufferLength ||
            regionBytes > maxBufferLength ||
            boundBytes > maxBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "adaptive sampling program exceeds device.maxBufferLength"
            );
        }

        id<MTLBuffer> particles = makePrivateBuffer(
            state_->device,
            std::max(
                particleBytes,
                sizeof(MRWorldAlignmentParticleGPU)
            ),
            @"MetalRobo alignment particles"
        );
        id<MTLBuffer> quantiles = makePrivateBuffer(
            state_->device,
            quantileBytes,
            @"MetalRobo alignment quantiles"
        );
        id<MTLBuffer> regions = makePrivateBuffer(
            state_->device,
            std::max(
                regionBytes,
                sizeof(MRWorldFeedbackRegionGPU)
            ),
            @"MetalRobo feedback regions"
        );
        id<MTLBuffer> bounds = makePrivateBuffer(
            state_->device,
            boundBytes,
            @"MetalRobo feedback bounds"
        );
        id<MTLBuffer> uploadParticles = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldAlignmentParticleGPU>{
                program.alignmentParticles
            },
            @"MetalRobo upload alignment particles"
        );
        id<MTLBuffer> uploadQuantiles = makeUploadBuffer(
            state_->device,
            std::span<const float>{program.alignmentQuantiles},
            @"MetalRobo upload alignment quantiles"
        );
        id<MTLBuffer> uploadRegions = makeUploadBuffer(
            state_->device,
            std::span<const MRWorldFeedbackRegionGPU>{
                program.feedbackRegions
            },
            @"MetalRobo upload feedback regions"
        );
        id<MTLBuffer> uploadBounds = makeUploadBuffer(
            state_->device,
            std::span<const mr_float4>{program.feedbackBounds},
            @"MetalRobo upload feedback bounds"
        );
        if (particles == nil || quantiles == nil || regions == nil ||
            bounds == nil || uploadParticles == nil ||
            uploadQuantiles == nil || uploadRegions == nil ||
            uploadBounds == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "failed to allocate adaptive sampling buffers"
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
                "failed to create adaptive sampling upload command"
            );
        }
        const auto copy = ^(
            id<MTLBuffer> source,
            id<MTLBuffer> destination,
            const std::size_t bytes
        ) {
            if (bytes != 0u) {
                [blit copyFromBuffer:source
                       sourceOffset:0u
                           toBuffer:destination
                  destinationOffset:0u
                               size:static_cast<NSUInteger>(bytes)];
            }
        };
        copy(uploadParticles, particles, particleBytes);
        copy(uploadQuantiles, quantiles, quantileBytes);
        copy(uploadRegions, regions, regionBytes);
        copy(uploadBounds, bounds, boundBytes);
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "adaptive sampling upload failed: " +
                    describeError(command.error)
            );
        }

        state_->buffers.alignmentParticles = particles;
        state_->buffers.alignmentQuantiles = quantiles;
        state_->buffers.feedbackRegions = regions;
        state_->buffers.feedbackBounds = bounds;
        state_->samplingProgram = program;
        state_->layout.alignmentParticleCount =
            static_cast<std::uint32_t>(
                program.alignmentParticles.size()
            );
        state_->layout.feedbackRegionCount =
            static_cast<std::uint32_t>(
                program.feedbackRegions.size()
            );
        state_->layout.samplingPrivateBytes =
            particleBytes + quantileBytes + regionBytes + boundBytes;
        state_->stats.retainedPrivateBytes =
            state_->layout.totalPrivateBytes();
        diagnostics.layout = state_->layout;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalBufferFailure,
            "host allocation failed while configuring sampling"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldFamilyDiagnostics MetalWorldFamilyContext::compile(
    const MRWorldPack& pack,
    const std::uint32_t capacity
) {
    std::string reason;
    if (!pack.valid(&reason)) {
        return reject(
            {},
            MetalWorldFamilyStatus::invalidFamily,
            "invalid MRWorldPack: " + reason
        );
    }
    return compile(pack.family, capacity);
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
        std::size_t assetElements = 0u;
        std::size_t sensorElements = 0u;
        std::size_t appearanceElements = 0u;
        std::size_t scenarioValueElements = 0u;
        if (!checkedMultiply(
                instanceCount,
                state_->layout.assetCountPerInstance,
                assetElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.sensorCountPerInstance,
                sensorElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.appearanceCountPerInstance,
                appearanceElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.variationCount,
                scenarioValueElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "world-family readback element count overflows"
            );
        }
        enum ReadbackSection : std::size_t {
            instanceSection,
            assetSection,
            sensorSection,
            appearanceSection,
            scenarioHeaderSection,
            scenarioValueSection,
            sectionCount,
        };
        std::array<std::size_t, sectionCount> sectionBytes{};
        if (!checkedByteCount<MRWorldInstanceHeaderGPU>(
                instanceCount,
                sectionBytes[instanceSection]
            ) ||
            !checkedByteCount<MRWorldAssetInstanceGPU>(
                assetElements,
                sectionBytes[assetSection]
            ) ||
            !checkedByteCount<MRWorldSensorInstanceGPU>(
                sensorElements,
                sectionBytes[sensorSection]
            ) ||
            !checkedByteCount<MRWorldAppearanceInstanceGPU>(
                appearanceElements,
                sectionBytes[appearanceSection]
            ) ||
            !checkedByteCount<MRWorldScenarioHeaderGPU>(
                instanceCount,
                sectionBytes[scenarioHeaderSection]
            ) ||
            !checkedByteCount<MRWorldScenarioValueGPU>(
                scenarioValueElements,
                sectionBytes[scenarioValueSection]
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "world-family readback byte count overflows"
            );
        }
        std::array<std::size_t, sectionCount> sectionOffsets{};
        std::size_t readbackBytes = 0u;
        for (std::size_t section = 0u;
             section < sectionCount;
             ++section) {
            if (!appendAlignedReadbackRegion(
                    sectionBytes[section],
                    readbackBytes,
                    sectionOffsets[section]
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldFamilyStatus::arithmeticOverflow,
                    "world-family readback layout overflows"
                );
            }
        }
        output.instances.reserve(instanceCount);
        output.scenarioHeaders.reserve(instanceCount);
        output.scenarioValues.reserve(scenarioValueElements);
        output.assets.reserve(assetElements);
        output.sensors.reserve(sensorElements);
        output.appearances.reserve(appearanceElements);
        id<MTLBuffer> readback = makeSharedBuffer(
            state_->device,
            readbackBytes,
            @"MetalRobo world-family readback"
        );
        if (readback == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "failed to allocate world-family readback buffer"
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
            const std::size_t destinationOffset,
            std::size_t bytes
        ) {
            if (bytes != 0u) {
                [blit copyFromBuffer:source
                       sourceOffset:0u
                           toBuffer:readback
                  destinationOffset:destinationOffset
                               size:static_cast<NSUInteger>(bytes)];
            }
        };
        copyOutput(
            state_->buffers.instances,
            sectionOffsets[instanceSection],
            sectionBytes[instanceSection]
        );
        copyOutput(
            state_->buffers.assets,
            sectionOffsets[assetSection],
            sectionBytes[assetSection]
        );
        copyOutput(
            state_->buffers.sensors,
            sectionOffsets[sensorSection],
            sectionBytes[sensorSection]
        );
        copyOutput(
            state_->buffers.appearances,
            sectionOffsets[appearanceSection],
            sectionBytes[appearanceSection]
        );
        copyOutput(
            state_->buffers.scenarioHeaders,
            sectionOffsets[scenarioHeaderSection],
            sectionBytes[scenarioHeaderSection]
        );
        copyOutput(
            state_->buffers.scenarioValues,
            sectionOffsets[scenarioValueSection],
            sectionBytes[scenarioValueSection]
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

        output.familyFingerprint = state_->familyFingerprint;
        output.instances.resize(instanceCount);
        output.scenarioHeaders.resize(instanceCount);
        output.scenarioValues.resize(scenarioValueElements);
        output.assets.resize(assetElements);
        output.sensors.resize(sensorElements);
        output.appearances.resize(appearanceElements);
        copySharedBuffer(
            output.instances,
            readback,
            sectionOffsets[instanceSection]
        );
        copySharedBuffer(
            output.scenarioHeaders,
            readback,
            sectionOffsets[scenarioHeaderSection]
        );
        copySharedBuffer(
            output.scenarioValues,
            readback,
            sectionOffsets[scenarioValueSection]
        );
        copySharedBuffer(
            output.assets,
            readback,
            sectionOffsets[assetSection]
        );
        copySharedBuffer(
            output.sensors,
            readback,
            sectionOffsets[sensorSection]
        );
        copySharedBuffer(
            output.appearances,
            readback,
            sectionOffsets[appearanceSection]
        );
        std::string reason;
        if (!output.valid(&reason)) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "GPU produced an invalid world batch: " + reason
            );
        }
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

MetalWorldFamilyDiagnostics MetalWorldFamilyContext::readbackPhysics(
    MetalWorldFamilyPhysicsBatch& output
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
                "sample the compiled world family before physics "
                "readback"
            );
        }

        const std::size_t instanceCount =
            state_->layout.activeInstanceCount;
        std::size_t qElements = 0u;
        std::size_t vElements = 0u;
        std::size_t sceneElements = 0u;
        std::size_t bodyElements = 0u;
        std::size_t controllerElements = 0u;
        if (!checkedMultiply(
                instanceCount,
                state_->layout.nq,
                qElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.nv,
                vElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.sceneBodyCount,
                sceneElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.bodyCount,
                bodyElements
            ) ||
            !checkedMultiply(
                instanceCount,
                state_->layout.articulationCount,
                controllerElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "physics readback element count overflows"
            );
        }
        enum ReadbackSection : std::size_t {
            qSection,
            vSection,
            sceneSection,
            bodySection,
            controllerSection,
            sectionCount,
        };
        std::array<std::size_t, sectionCount> sectionBytes{};
        if (!checkedByteCount<float>(
                qElements,
                sectionBytes[qSection]
            ) ||
            !checkedByteCount<float>(
                vElements,
                sectionBytes[vSection]
            ) ||
            !checkedByteCount<MRBodyStateGPU>(
                sceneElements,
                sectionBytes[sceneSection]
            ) ||
            !checkedByteCount<MRWorldBodyParametersGPU>(
                bodyElements,
                sectionBytes[bodySection]
            ) ||
            !checkedByteCount<MRWorldControllerParametersGPU>(
                controllerElements,
                sectionBytes[controllerSection]
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::arithmeticOverflow,
                "physics readback byte count overflows"
            );
        }
        std::array<std::size_t, sectionCount> sectionOffsets{};
        std::size_t readbackBytes = 0u;
        for (std::size_t section = 0u;
             section < sectionCount;
             ++section) {
            if (!appendAlignedReadbackRegion(
                    sectionBytes[section],
                    readbackBytes,
                    sectionOffsets[section]
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldFamilyStatus::arithmeticOverflow,
                    "physics readback layout overflows"
                );
            }
        }
        output.resetQ.reserve(qElements);
        output.resetV.reserve(vElements);
        output.resetSceneBodies.reserve(sceneElements);
        output.bodyParameters.reserve(bodyElements);
        output.controllerParameters.reserve(controllerElements);
        id<MTLBuffer> readback = makeSharedBuffer(
            state_->device,
            readbackBytes,
            @"MetalRobo physics readback"
        );
        if (readback == nil) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalBufferFailure,
                "failed to allocate physics readback buffer"
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
                "failed to create physics readback command"
            );
        }
        const auto copyOutput = ^(
            id<MTLBuffer> source,
            const std::size_t destinationOffset,
            const std::size_t bytes
        ) {
            if (bytes != 0u) {
                [blit copyFromBuffer:source
                       sourceOffset:0u
                           toBuffer:readback
                  destinationOffset:destinationOffset
                               size:static_cast<NSUInteger>(bytes)];
            }
        };
        copyOutput(
            state_->buffers.resetQ,
            sectionOffsets[qSection],
            sectionBytes[qSection]
        );
        copyOutput(
            state_->buffers.resetV,
            sectionOffsets[vSection],
            sectionBytes[vSection]
        );
        copyOutput(
            state_->buffers.resetSceneBodies,
            sectionOffsets[sceneSection],
            sectionBytes[sceneSection]
        );
        copyOutput(
            state_->buffers.bodyParameters,
            sectionOffsets[bodySection],
            sectionBytes[bodySection]
        );
        copyOutput(
            state_->buffers.controllerParameters,
            sectionOffsets[controllerSection],
            sectionBytes[controllerSection]
        );
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "physics readback failed: " +
                    describeError(command.error)
            );
        }
        output.instanceCount =
            static_cast<std::uint32_t>(instanceCount);
        output.primaryArticulationIndex =
            state_->layout.primaryArticulationIndex;
        output.nq = state_->layout.nq;
        output.nv = state_->layout.nv;
        output.bodyCount = state_->layout.bodyCount;
        output.sceneBodyCount = state_->layout.sceneBodyCount;
        output.articulationCount =
            state_->layout.articulationCount;
        output.resetQ.resize(qElements);
        output.resetV.resize(vElements);
        output.resetSceneBodies.resize(sceneElements);
        output.bodyParameters.resize(bodyElements);
        output.controllerParameters.resize(controllerElements);
        copySharedBuffer(
            output.resetQ,
            readback,
            sectionOffsets[qSection]
        );
        copySharedBuffer(
            output.resetV,
            readback,
            sectionOffsets[vSection]
        );
        copySharedBuffer(
            output.resetSceneBodies,
            readback,
            sectionOffsets[sceneSection]
        );
        copySharedBuffer(
            output.bodyParameters,
            readback,
            sectionOffsets[bodySection]
        );
        copySharedBuffer(
            output.controllerParameters,
            readback,
            sectionOffsets[controllerSection]
        );
        const auto finiteFloat = [](const float value) {
            return std::isfinite(value);
        };
        if (!std::all_of(
                output.resetQ.begin(),
                output.resetQ.end(),
                finiteFloat
            ) ||
            !std::all_of(
                output.resetV.begin(),
                output.resetV.end(),
                finiteFloat
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldFamilyStatus::metalCommandFailure,
                "GPU produced non-finite physics reset coordinates"
            );
        }
        state_->stats.readbackCount += 1u;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldFamilyStatus::metalBufferFailure,
            "host allocation failed during physics readback"
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
        case MetalWorldFamilyBuffer::assetBindings:
            selected = state_->buffers.assetBindings;
            break;
        case MetalWorldFamilyBuffer::bindingIndices:
            selected = state_->buffers.bindingIndices;
            break;
        case MetalWorldFamilyBuffer::resetQ:
            selected = state_->buffers.resetQ;
            break;
        case MetalWorldFamilyBuffer::resetV:
            selected = state_->buffers.resetV;
            break;
        case MetalWorldFamilyBuffer::resetSceneBodies:
            selected = state_->buffers.resetSceneBodies;
            break;
        case MetalWorldFamilyBuffer::bodyParameters:
            selected = state_->buffers.bodyParameters;
            break;
        case MetalWorldFamilyBuffer::controllerParameters:
            selected = state_->buffers.controllerParameters;
            break;
        case MetalWorldFamilyBuffer::scenarioHeaders:
            selected = state_->buffers.scenarioHeaders;
            break;
        case MetalWorldFamilyBuffer::scenarioValues:
            selected = state_->buffers.scenarioValues;
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
