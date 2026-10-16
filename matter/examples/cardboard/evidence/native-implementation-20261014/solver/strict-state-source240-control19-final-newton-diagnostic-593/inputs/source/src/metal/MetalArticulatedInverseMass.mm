#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedInverseMass.hpp"

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
#include <string>
#include <system_error>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr std::size_t kRawBufferCount = 10u;
constexpr NSUInteger kThreadsPerThreadgroup = 32u;
constexpr float kQuaternionHostTolerance = 1.9e-5f;
constexpr std::uint64_t kShaderAddressableElements =
    static_cast<std::uint64_t>(
        std::numeric_limits<mr_u32>::max()
    ) + 1u;
const char kMetalRoboInverseMassImageAnchor = 0;

struct BufferRequirement {
    const char* label = "";
    std::size_t logicalElements = 0u;
    std::size_t logicalBytes = 0u;
    std::size_t allocationBytes = 0u;
};

struct RequiredBuffers {
    std::array<BufferRequirement, kRawBufferCount> entries{};
};

} // namespace

namespace detail {

struct MetalArticulatedInverseMassContextState {
    explicit MetalArticulatedInverseMassContextState(
        MetalArticulatedInverseMassConfig configured
    )
        : config(std::move(configured)) {}

    MetalArticulatedInverseMassConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool inFlight = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> pipeline = nil;
    __strong id<MTLBuffer> buffers[kRawBufferCount] = {};
    std::array<std::size_t, kRawBufferCount> capacities{};
    MetalArticulatedInverseMassContextStats stats{};
};

struct MetalArticulatedInverseMassSubmissionState {
    ~MetalArticulatedInverseMassSubmissionState() {
        if (!ownsInFlight || context == nullptr) {
            return;
        }
        @autoreleasepool {
            [commandBuffer waitUntilCompleted];
        }
        try {
            const std::lock_guard lock(context->mutex);
            context->inFlight = false;
            context->stats.hasInFlightSubmission = false;
            ++context->stats.completedSubmissionCount;
        } catch (...) {
            // Destruction cannot throw. Waiting first guarantees the shared
            // arena is no longer visible to GPU execution.
        }
        ownsInFlight = false;
    }

    std::shared_ptr<
        MetalArticulatedInverseMassContextState
    > context;
    __strong id<MTLCommandBuffer> commandBuffer = nil;
    MetalArticulatedInverseMassDiagnostics diagnostics{};
    std::chrono::steady_clock::time_point start{};
    MRArticulationGPU articulation{};
    bool ownsInFlight = false;
};

} // namespace detail

namespace {

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
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalRoboInverseMassImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory / "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() /
                "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate :
             candidates) {
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

MetalArticulatedInverseMassDiagnostics reject(
    MetalArticulatedInverseMassDiagnostics diagnostics,
    const MetalArticulatedInverseMassHostStatus status,
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

template <typename T>
bool makeRequirement(
    const char* label,
    const std::size_t logicalElements,
    BufferRequirement& result
) {
    std::size_t logicalBytes = 0u;
    if (!checkedMultiply(
            logicalElements,
            sizeof(T),
            logicalBytes
        )) {
        return false;
    }
    result.label = label;
    result.logicalElements = logicalElements;
    result.logicalBytes = logicalBytes;
    result.allocationBytes =
        logicalBytes == 0u ? sizeof(T) : logicalBytes;
    return result.allocationBytes <=
        std::numeric_limits<NSUInteger>::max();
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool finiteFloats(const std::span<const float> values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

bool validQ(
    const MRArticulationGPU& articulation,
    const std::size_t environmentCount,
    const std::span<const float> q
) {
    if (!finiteFloats(q)) {
        return false;
    }
    if (articulation.rootType != MR_ROOT_FLOATING) {
        return true;
    }
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        const std::size_t base =
            environment * articulation.nq + 3u;
        const double x = q[base + 0u];
        const double y = q[base + 1u];
        const double z = q[base + 2u];
        const double w = q[base + 3u];
        const double normSquared =
            x * x + y * y + z * z + w * w;
        if (!(normSquared > 1.0e-12) ||
            !std::isfinite(normSquared)) {
            return false;
        }
        const double norm = std::sqrt(normSquared);
        if (!std::isfinite(norm) ||
            std::abs(norm - 1.0) >
                kQuaternionHostTolerance) {
            return false;
        }
    }
    return true;
}

bool supportedTopology(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    std::string& reason
) {
    if (articulation.bodyCount == 0u ||
        articulation.bodyCount >
            MR_ARTICULATED_ABA_MAX_BODIES ||
        articulation.nv == 0u ||
        articulation.nv > MR_ARTICULATED_ABA_MAX_DOFS ||
        articulation.nq > MR_ARTICULATED_ABA_MAX_Q) {
        reason =
            "articulation exceeds the Metal inverse-mass "
            "body/DoF/q bucket";
        return false;
    }
    if (articulation.rootType != MR_ROOT_FIXED &&
        articulation.rootType != MR_ROOT_FLOATING) {
        reason = "Metal inverse mass root type is unsupported";
        return false;
    }

    const std::size_t jointEnd =
        static_cast<std::size_t>(articulation.firstJoint) +
        articulation.jointCount;
    for (std::size_t jointIndex = articulation.firstJoint;
         jointIndex < jointEnd;
         ++jointIndex) {
        const MRJointDescriptorGPU& joint =
            model.joints[jointIndex];
        if (joint.flags != 0u) {
            reason =
                "Metal inverse-mass joints require zero "
                "reserved flags";
            return false;
        }
        if (joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC &&
            joint.jointType != MR_JOINT_FIXED) {
            reason =
                "Metal inverse mass supports revolute, prismatic, "
                "continuous, and fixed joints";
            return false;
        }
    }

    const std::size_t bodyEnd =
        static_cast<std::size_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (std::size_t bodyIndex = articulation.firstBody;
         bodyIndex < bodyEnd;
         ++bodyIndex) {
        if (model.bodies[bodyIndex].motionType !=
            MR_MOTION_DYNAMIC) {
            reason =
                "every body owned by a Metal inverse-mass "
                "articulation must be dynamic";
            return false;
        }
    }
    return true;
}

bool buildRequirements(
    const EngineModel& model,
    const MetalArticulatedInverseMassLayout& layout,
    RequiredBuffers& requirements,
    std::size_t& totalRequiredBytes
) {
    const std::size_t jointElements =
        std::max<std::size_t>(model.joints.size(), 1u);
    if (!makeRequirement<MRWorldGPU>(
            "world",
            1u,
            requirements.entries[0]
        ) ||
        !makeRequirement<MRArticulationGPU>(
            "articulations",
            model.articulations.size(),
            requirements.entries[1]
        ) ||
        !makeRequirement<MRJointDescriptorGPU>(
            "joints",
            jointElements,
            requirements.entries[2]
        ) ||
        !makeRequirement<MRDofPropertiesGPU>(
            "DoF properties",
            model.dofs.size(),
            requirements.entries[3]
        ) ||
        !makeRequirement<MRBodyPropertiesGPU>(
            "body properties",
            model.bodies.size(),
            requirements.entries[4]
        ) ||
        !makeRequirement<MRInverseMassDispatchGPU>(
            "inverse-mass dispatch",
            1u,
            requirements.entries[5]
        ) ||
        !makeRequirement<float>(
            "q",
            layout.qElements,
            requirements.entries[6]
        ) ||
        !makeRequirement<float>(
            "right-hand sides",
            layout.rhsElements,
            requirements.entries[7]
        ) ||
        !makeRequirement<float>(
            "inverse-mass output",
            layout.outputElements,
            requirements.entries[8]
        ) ||
        !makeRequirement<MRInverseMassStatusGPU>(
            "inverse-mass statuses",
            layout.statusElements,
            requirements.entries[9]
        )) {
        return false;
    }

    totalRequiredBytes = 0u;
    for (const BufferRequirement& requirement :
         requirements.entries) {
        if (!checkedAdd(
                totalRequiredBytes,
                requirement.allocationBytes,
                totalRequiredBytes
            )) {
            return false;
        }
    }
    return true;
}

MetalArticulatedInverseMassDiagnostics validateAndBuildLayout(
    const EngineModel& model,
    const MetalArticulatedInverseMassInput& input,
    RequiredBuffers& requirements
) {
    MetalArticulatedInverseMassDiagnostics diagnostics{};

    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::invalidModel,
            "invalid EngineModel: " + modelReason
        );
    }
    if (input.articulationIndex >=
        model.articulations.size()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                invalidDimensions,
            "articulation index is outside the canonical model"
        );
    }
    if (input.environmentCount == 0u ||
        input.rhsCount == 0u ||
        input.rhsCount >
            MR_ARTICULATED_INVERSE_MASS_MAX_RHS) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                invalidDimensions,
            "environmentCount must be positive and rhsCount "
            "must be in [1, 3]"
        );
    }
    if (input.environmentCount >
            std::numeric_limits<mr_u32>::max() ||
        input.rhsCount >
            std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                arithmeticOverflow,
            "batch counts do not fit the GPU dispatch ABI"
        );
    }

    const MRArticulationGPU& articulation =
        model.articulations[input.articulationIndex];
    std::string topologyReason;
    if (!supportedTopology(
            model,
            articulation,
            topologyReason
        )) {
        const bool capacity =
            articulation.bodyCount >
                MR_ARTICULATED_ABA_MAX_BODIES ||
            articulation.nv >
                MR_ARTICULATED_ABA_MAX_DOFS ||
            articulation.nq > MR_ARTICULATED_ABA_MAX_Q;
        return reject(
            std::move(diagnostics),
            capacity
                ? MetalArticulatedInverseMassHostStatus::
                      capacityOverflow
                : MetalArticulatedInverseMassHostStatus::
                      unsupportedTopology,
            std::move(topologyReason)
        );
    }

    MetalArticulatedInverseMassLayout layout{};
    std::size_t vectorsPerEnvironment = 0u;
    if (!checkedMultiply(
            input.rhsCount,
            articulation.nv,
            vectorsPerEnvironment
        ) ||
        vectorsPerEnvironment >
            std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                arithmeticOverflow,
            "per-environment inverse-mass vector span overflow"
        );
    }
    MRInverseMassDispatchGPU& dispatch = layout.dispatch;
    dispatch.articulationIndex = input.articulationIndex;
    dispatch.environmentCount =
        static_cast<mr_u32>(input.environmentCount);
    dispatch.rhsCount = static_cast<mr_u32>(input.rhsCount);
    dispatch.qStride = articulation.nq;
    dispatch.rhsEnvironmentStride =
        static_cast<mr_u32>(vectorsPerEnvironment);
    dispatch.rhsVectorStride = articulation.nv;
    dispatch.outputEnvironmentStride =
        static_cast<mr_u32>(vectorsPerEnvironment);
    dispatch.outputVectorStride = articulation.nv;

    if (!checkedMultiply(
            input.environmentCount,
            articulation.nq,
            layout.qElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            vectorsPerEnvironment,
            layout.rhsElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                arithmeticOverflow,
            "derived inverse-mass element-count overflow"
        );
    }
    layout.outputElements = layout.rhsElements;
    layout.statusElements = input.environmentCount;

    const auto exceedsShaderAddressing =
        [](const std::size_t elements) {
            return static_cast<std::uint64_t>(elements) >
                kShaderAddressableElements;
        };
    if (exceedsShaderAddressing(model.articulations.size()) ||
        exceedsShaderAddressing(model.joints.size()) ||
        exceedsShaderAddressing(model.dofs.size()) ||
        exceedsShaderAddressing(model.bodies.size()) ||
        exceedsShaderAddressing(layout.qElements) ||
        exceedsShaderAddressing(layout.rhsElements) ||
        exceedsShaderAddressing(layout.outputElements) ||
        exceedsShaderAddressing(layout.statusElements)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                arithmeticOverflow,
            "compact buffer exceeds the shader's 32-bit "
            "element-addressing contract"
        );
    }

    std::size_t totalRequiredBytes = 0u;
    if (!buildRequirements(
            model,
            layout,
            requirements,
            totalRequiredBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                arithmeticOverflow,
            "required Metal buffer byte-count overflow"
        );
    }
    layout.qBytes = requirements.entries[6].logicalBytes;
    layout.rhsBytes = requirements.entries[7].logicalBytes;
    layout.outputBytes =
        requirements.entries[8].logicalBytes;
    layout.statusBytes =
        requirements.entries[9].logicalBytes;
    layout.totalRequiredBytes = totalRequiredBytes;
    diagnostics.layout = layout;

    if (input.q.size() != layout.qElements ||
        input.rightHandSides.size() != layout.rhsElements) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                invalidDimensions,
            "packed q or right-hand-side span has the wrong "
            "element count"
        );
    }
    if (!validQ(
            articulation,
            input.environmentCount,
            input.q
        ) ||
        !finiteFloats(input.rightHandSides)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                nonfiniteInput,
            "q or right-hand side contains a non-finite "
            "value or invalid root quaternion"
        );
    }
    return diagnostics;
}

NSString* bufferLabel(const std::size_t index) {
    switch (index) {
    case 0u:
        return @"inverse mass world";
    case 1u:
        return @"inverse mass articulations";
    case 2u:
        return @"inverse mass joints";
    case 3u:
        return @"inverse mass DoF properties";
    case 4u:
        return @"inverse mass body properties";
    case 5u:
        return @"inverse mass dispatch";
    case 6u:
        return @"inverse mass q";
    case 7u:
        return @"inverse mass RHS";
    case 8u:
        return @"inverse mass output";
    case 9u:
        return @"inverse mass statuses";
    default:
        return @"inverse mass buffer";
    }
}

MetalArticulatedInverseMassDiagnostics initializeContext(
    detail::MetalArticulatedInverseMassContextState& context,
    MetalArticulatedInverseMassDiagnostics diagnostics
) {
    if (context.initialized) {
        diagnostics.deviceName = nsString(context.device.name);
        return diagnostics;
    }

    std::string metallibPath = context.config.metallibPath;
    if (metallibPath.empty()) {
        metallibPath = defaultMetallibPath();
    }
    if (metallibPath.empty()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metallibUnavailable,
            "no inverse-mass metallib path is available"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalDeviceUnsupported,
            "Metal inverse mass requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalDeviceUnavailable,
            "failed to create a Metal command queue"
        );
    }
    queue.label = @"MetalRobo inverse-mass queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metallibUnavailable,
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
            MetalArticulatedInverseMassHostStatus::
                metalLibraryFailure,
            "failed to load metallib: " +
                describeError(error)
        );
    }
    id<MTLFunction> function = [library
        newFunctionWithName:@"mr_articulated_inverse_mass"];
    if (function == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalLibraryFailure,
            "metallib does not contain "
            "mr_articulated_inverse_mass"
        );
    }
    error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                       error:&error];
    if (pipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalPipelineFailure,
            "failed to create inverse-mass pipeline: " +
                describeError(error)
        );
    }
    if (pipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup ||
        pipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalDeviceUnsupported,
            "device cannot execute the 32-lane inverse-mass "
            "threadgroup"
        );
    }

    context.device = device;
    context.queue = queue;
    context.library = library;
    context.pipeline = pipeline;
    context.initialized = true;
    ++context.stats.pipelineCreationCount;
    return diagnostics;
}

std::size_t growthCapacity(
    const std::size_t current,
    const std::size_t required,
    const std::size_t maximum
) {
    if (current >= required) {
        return current;
    }
    if (current == 0u) {
        return required;
    }
    const std::size_t half = current / 2u;
    const std::size_t grown =
        half <= maximum - current
            ? current + half
            : maximum;
    return std::max(required, grown);
}

MetalArticulatedInverseMassDiagnostics ensureBufferArena(
    detail::MetalArticulatedInverseMassContextState& context,
    const RequiredBuffers& requirements,
    MetalArticulatedInverseMassDiagnostics diagnostics
) {
    const std::size_t maximumBufferLength =
        static_cast<std::size_t>(
            context.device.maxBufferLength
        );
    std::array<std::size_t, kRawBufferCount> proposed =
        context.capacities;
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        const BufferRequirement& requirement =
            requirements.entries[index];
        if (requirement.allocationBytes >
            maximumBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedInverseMassHostStatus::
                    metalBufferFailure,
                std::string(requirement.label) +
                    " exceeds device.maxBufferLength"
            );
        }
        proposed[index] = growthCapacity(
            context.capacities[index],
            requirement.allocationBytes,
            maximumBufferLength
        );
    }

    std::size_t projectedBytes = 0u;
    for (const std::size_t capacity : proposed) {
        if (!checkedAdd(
                projectedBytes,
                capacity,
                projectedBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedInverseMassHostStatus::
                    arithmeticOverflow,
                "persistent inverse-mass arena byte-count "
                "overflow"
            );
        }
    }
    const std::uint64_t recommendedWorkingSet =
        context.device.recommendedMaxWorkingSetSize;
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        projectedBytes = 0u;
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            proposed[index] = std::max(
                context.capacities[index],
                requirements.entries[index].allocationBytes
            );
            if (!checkedAdd(
                    projectedBytes,
                    proposed[index],
                    projectedBytes
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedInverseMassHostStatus::
                        arithmeticOverflow,
                    "persistent inverse-mass arena byte-count "
                    "overflow"
                );
            }
        }
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalBufferFailure,
            "persistent inverse-mass arena exceeds "
            "device.recommendedMaxWorkingSetSize"
        );
    }

    __strong id<MTLBuffer> replacements[kRawBufferCount] = {};
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        if (proposed[index] == context.capacities[index]) {
            continue;
        }
        replacements[index] = [context.device
            newBufferWithLength:static_cast<NSUInteger>(
                proposed[index]
            )
                       options:MTLResourceStorageModeShared];
        if (replacements[index] == nil ||
            replacements[index].contents == nullptr ||
            replacements[index].length < proposed[index]) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedInverseMassHostStatus::
                    metalBufferFailure,
                std::string(
                    "persistent Metal buffer growth failed for "
                ) +
                    requirements.entries[index].label
            );
        }
        replacements[index].label = bufferLabel(index);
    }

    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        if (replacements[index] == nil) {
            continue;
        }
        if (context.capacities[index] != 0u) {
            ++context.stats.bufferGrowthCount;
        }
        ++context.stats.bufferAllocationCount;
        context.buffers[index] = replacements[index];
        context.capacities[index] = proposed[index];
    }
    context.stats.retainedBufferBytes = projectedBytes;
    return diagnostics;
}

void copyToBuffer(
    id<MTLBuffer> destination,
    const void* source,
    const BufferRequirement& requirement
) {
    if (requirement.logicalBytes == 0u) {
        std::memset(
            destination.contents,
            0,
            requirement.allocationBytes
        );
        return;
    }
    std::memcpy(
        destination.contents,
        source,
        requirement.logicalBytes
    );
}

void uploadBatch(
    detail::MetalArticulatedInverseMassContextState& context,
    const EngineModel& model,
    const MetalArticulatedInverseMassInput& input,
    const MetalArticulatedInverseMassLayout& layout,
    const RequiredBuffers& requirements
) {
    MRJointDescriptorGPU emptyJoint{};
    const std::array<const void*, 8u> sources{
        &model.world,
        model.articulations.data(),
        model.joints.empty()
            ? static_cast<const void*>(&emptyJoint)
            : static_cast<const void*>(model.joints.data()),
        model.dofs.data(),
        model.bodies.data(),
        &layout.dispatch,
        input.q.data(),
        input.rightHandSides.data(),
    };
    for (std::size_t index = 0u;
         index < sources.size();
         ++index) {
        copyToBuffer(
            context.buffers[index],
            sources[index],
            requirements.entries[index]
        );
    }
    for (std::size_t index = sources.size();
         index < kRawBufferCount;
         ++index) {
        std::memset(
            context.buffers[index].contents,
            0,
            requirements.entries[index].allocationBytes
        );
    }
}

template <typename T>
void copyOutput(
    std::vector<T>& destination,
    id<MTLBuffer> source
) {
    if (!destination.empty()) {
        std::memcpy(
            destination.data(),
            source.contents,
            destination.size() * sizeof(T)
        );
    }
}

bool zeroEnvironmentPayload(
    const MetalArticulatedInverseMassResult& result,
    const std::size_t environment
) {
    const std::size_t begin =
        environment *
        result.layout.dispatch.outputEnvironmentStride;
    const std::size_t end =
        begin +
        result.layout.dispatch.outputEnvironmentStride;
    return std::all_of(
        result.output.begin() +
            static_cast<std::ptrdiff_t>(begin),
        result.output.begin() +
            static_cast<std::ptrdiff_t>(end),
        [](const float value) {
            return value == 0.0f;
        }
    );
}

bool finiteEnvironmentPayload(
    const MetalArticulatedInverseMassResult& result,
    const std::size_t environment
) {
    const std::size_t begin =
        environment *
        result.layout.dispatch.outputEnvironmentStride;
    const std::size_t end =
        begin +
        result.layout.dispatch.outputEnvironmentStride;
    return std::all_of(
        result.output.begin() +
            static_cast<std::ptrdiff_t>(begin),
        result.output.begin() +
            static_cast<std::ptrdiff_t>(end),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

MetalArticulatedInverseMassDiagnostics validateAndPublish(
    const MRArticulationGPU& articulation,
    MetalArticulatedInverseMassResult&& staged,
    MetalArticulatedInverseMassDiagnostics diagnostics,
    MetalArticulatedInverseMassResult& result
) {
    const MRInverseMassDispatchGPU& dispatch =
        diagnostics.layout.dispatch;
    for (std::size_t environment = 0u;
         environment < staged.statuses.size();
         ++environment) {
        const MRInverseMassStatusGPU& status =
            staged.statuses[environment];
        if (status.environment != environment ||
            status.articulationIndex !=
                dispatch.articulationIndex ||
            status.code >
                MR_INVERSE_MASS_UNSUPPORTED_TOPOLOGY ||
            status.bodyCount != articulation.bodyCount ||
            status.nq != articulation.nq ||
            status.nv != articulation.nv ||
            status.rhsCount != dispatch.rhsCount) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedInverseMassHostStatus::
                    internalFailure,
                "GPU returned a malformed inverse-mass status"
            );
        }
        if (status.code == MR_INVERSE_MASS_SUCCESS) {
            if (status.failingIndex != MR_INVALID_INDEX ||
                !finite(status.diagnostics) ||
                !(status.diagnostics.x > 0.0f) ||
                status.diagnostics.y < status.diagnostics.x ||
                !finiteEnvironmentPayload(staged, environment)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedInverseMassHostStatus::
                        internalFailure,
                    "GPU success status has invalid diagnostics "
                    "or payload"
                );
            }
            ++diagnostics.successfulEnvironmentCount;
        } else {
            if (!zeroEnvironmentPayload(staged, environment)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedInverseMassHostStatus::
                        internalFailure,
                    "failed GPU environment published a partial "
                    "inverse-mass result"
                );
            }
            if (diagnostics.failedEnvironmentCount == 0u) {
                diagnostics.firstFailingEnvironment =
                    static_cast<std::uint32_t>(environment);
                diagnostics.firstGPUStatusCode = status.code;
            }
            ++diagnostics.failedEnvironmentCount;
        }
    }

    result = std::move(staged);
    diagnostics.published = true;
    if (diagnostics.failedEnvironmentCount != 0u) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                gpuEnvironmentFailure,
            "one or more GPU environments rejected the "
            "inverse-mass action"
        );
    }
    diagnostics.status =
        MetalArticulatedInverseMassHostStatus::success;
    diagnostics.message.clear();
    return diagnostics;
}

} // namespace

MetalArticulatedInverseMassSubmission::
    MetalArticulatedInverseMassSubmission() noexcept = default;

MetalArticulatedInverseMassSubmission::
    ~MetalArticulatedInverseMassSubmission() = default;

MetalArticulatedInverseMassSubmission::
    MetalArticulatedInverseMassSubmission(
        MetalArticulatedInverseMassSubmission&& other
    ) noexcept = default;

MetalArticulatedInverseMassSubmission&
MetalArticulatedInverseMassSubmission::operator=(
    MetalArticulatedInverseMassSubmission&& other
) noexcept = default;

bool MetalArticulatedInverseMassSubmission::valid()
    const noexcept {
    return state_ != nullptr;
}

MetalArticulatedInverseMassDiagnostics
MetalArticulatedInverseMassSubmission::wait(
    MetalArticulatedInverseMassResult& result
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalArticulatedInverseMassHostStatus::
                metalCommandFailure,
            "inverse-mass submission is empty or already consumed"
        );
    }

    std::unique_ptr<
        detail::MetalArticulatedInverseMassSubmissionState
    > pending = std::move(state_);
    MetalArticulatedInverseMassDiagnostics diagnostics =
        pending->diagnostics;
    try {
        MetalArticulatedInverseMassResult staged{};
        @autoreleasepool {
            [pending->commandBuffer waitUntilCompleted];
            const auto end = std::chrono::steady_clock::now();
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - pending->start
                ).count();
            if (pending->commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedInverseMassHostStatus::
                        metalCommandFailure,
                    "Metal inverse-mass command failed: " +
                        describeError(
                            pending->commandBuffer.error
                        )
                );
            }

            staged.layout = diagnostics.layout;
            staged.output.resize(
                staged.layout.outputElements
            );
            staged.statuses.resize(
                staged.layout.statusElements
            );
            const auto& buffers =
                pending->context->buffers;
            copyOutput(staged.output, buffers[8]);
            copyOutput(staged.statuses, buffers[9]);
        }

        return validateAndPublish(
            pending->articulation,
            std::move(staged),
            std::move(diagnostics),
            result
        );
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalBufferFailure,
            "host allocation failed while publishing "
            "inverse-mass results"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedInverseMassContext::
    MetalArticulatedInverseMassContext(
        MetalArticulatedInverseMassConfig config
    )
    : state_(std::make_shared<
          detail::MetalArticulatedInverseMassContextState
      >(std::move(config))) {}

MetalArticulatedInverseMassContext::
    ~MetalArticulatedInverseMassContext() = default;

MetalArticulatedInverseMassContext::
    MetalArticulatedInverseMassContext(
        MetalArticulatedInverseMassContext&& other
    ) noexcept = default;

MetalArticulatedInverseMassContext&
MetalArticulatedInverseMassContext::operator=(
    MetalArticulatedInverseMassContext&& other
) noexcept = default;

MetalArticulatedInverseMassDiagnostics
MetalArticulatedInverseMassContext::submit(
    const EngineModel& model,
    const MetalArticulatedInverseMassInput& input,
    MetalArticulatedInverseMassSubmission& submission
) {
    MetalArticulatedInverseMassDiagnostics diagnostics{};
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                internalFailure,
            "inverse-mass context was moved from"
        );
    }
    if (submission.valid()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::contextBusy,
            "submission output already owns an in-flight batch"
        );
    }

    RequiredBuffers requirements{};
    try {
        diagnostics = validateAndBuildLayout(
            model,
            input,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        const std::lock_guard lock(state_->mutex);
        if (state_->inFlight) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedInverseMassHostStatus::
                    contextBusy,
                "inverse-mass context already has an "
                "in-flight batch"
            );
        }
        @autoreleasepool {
            diagnostics = initializeContext(
                *state_,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }
            diagnostics = ensureBufferArena(
                *state_,
                requirements,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }

            uploadBatch(
                *state_,
                model,
                input,
                diagnostics.layout,
                requirements
            );

            id<MTLCommandBuffer> commandBuffer =
                [state_->queue commandBuffer];
            if (commandBuffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedInverseMassHostStatus::
                        metalCommandFailure,
                    "failed to create inverse-mass command "
                    "buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo persistent articulated inverse mass";
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedInverseMassHostStatus::
                        metalCommandFailure,
                    "failed to create inverse-mass compute "
                    "encoder"
                );
            }
            [encoder setComputePipelineState:state_->pipeline];
            for (NSUInteger index = 0u;
                 index < kRawBufferCount;
                 ++index) {
                [encoder
                    setBuffer:state_->buffers[index]
                       offset:0u
                      atIndex:index];
            }
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                    static_cast<NSUInteger>(
                        input.environmentCount
                    ),
                    1u,
                    1u
                )
                threadsPerThreadgroup:MTLSizeMake(
                    kThreadsPerThreadgroup,
                    1u,
                    1u
                )];
            [encoder endEncoding];

            auto pending = std::make_unique<
                detail::
                    MetalArticulatedInverseMassSubmissionState
            >();
            diagnostics.dispatched = true;
            pending->context = state_;
            pending->commandBuffer = commandBuffer;
            pending->diagnostics = diagnostics;
            pending->articulation =
                model.articulations[input.articulationIndex];
            pending->start = std::chrono::steady_clock::now();
            pending->ownsInFlight = true;

            state_->inFlight = true;
            state_->stats.hasInFlightSubmission = true;
            ++state_->stats.submissionCount;
            [commandBuffer commit];
            submission.state_ = std::move(pending);
        }
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                metalBufferFailure,
            "host allocation failed while preparing "
            "inverse-mass submission"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedInverseMassHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedInverseMassDiagnostics
MetalArticulatedInverseMassContext::run(
    const EngineModel& model,
    const MetalArticulatedInverseMassInput& input,
    MetalArticulatedInverseMassResult& result
) {
    MetalArticulatedInverseMassSubmission submission;
    MetalArticulatedInverseMassDiagnostics diagnostics = submit(
        model,
        input,
        submission
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return submission.wait(result);
}

MetalArticulatedInverseMassContextStats
MetalArticulatedInverseMassContext::stats() const noexcept {
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

MetalArticulatedInverseMassDiagnostics
runMetalArticulatedInverseMass(
    const EngineModel& model,
    const MetalArticulatedInverseMassInput& input,
    MetalArticulatedInverseMassResult& result,
    const MetalArticulatedInverseMassConfig& config
) {
    try {
        MetalArticulatedInverseMassContext context(config);
        return context.run(model, input, result);
    } catch (const std::bad_alloc&) {
        return reject(
            {},
            MetalArticulatedInverseMassHostStatus::
                metalBufferFailure,
            "host allocation failed while creating "
            "inverse-mass context"
        );
    } catch (const std::exception& exception) {
        return reject(
            {},
            MetalArticulatedInverseMassHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

const char* metalArticulatedInverseMassHostStatusName(
    const MetalArticulatedInverseMassHostStatus status
) noexcept {
    switch (status) {
    case MetalArticulatedInverseMassHostStatus::success:
        return "success";
    case MetalArticulatedInverseMassHostStatus::invalidModel:
        return "invalid_model";
    case MetalArticulatedInverseMassHostStatus::
            unsupportedTopology:
        return "unsupported_topology";
    case MetalArticulatedInverseMassHostStatus::
            invalidDimensions:
        return "invalid_dimensions";
    case MetalArticulatedInverseMassHostStatus::
            capacityOverflow:
        return "capacity_overflow";
    case MetalArticulatedInverseMassHostStatus::
            arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalArticulatedInverseMassHostStatus::
            nonfiniteInput:
        return "nonfinite_input";
    case MetalArticulatedInverseMassHostStatus::
            metallibUnavailable:
        return "metallib_unavailable";
    case MetalArticulatedInverseMassHostStatus::
            metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalArticulatedInverseMassHostStatus::
            metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalArticulatedInverseMassHostStatus::
            metalLibraryFailure:
        return "metal_library_failure";
    case MetalArticulatedInverseMassHostStatus::
            metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalArticulatedInverseMassHostStatus::
            metalBufferFailure:
        return "metal_buffer_failure";
    case MetalArticulatedInverseMassHostStatus::
            metalCommandFailure:
        return "metal_command_failure";
    case MetalArticulatedInverseMassHostStatus::
            gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalArticulatedInverseMassHostStatus::
            internalFailure:
        return "internal_failure";
    case MetalArticulatedInverseMassHostStatus::contextBusy:
        return "context_busy";
    }
    return "unknown";
}

} // namespace metalrobo
