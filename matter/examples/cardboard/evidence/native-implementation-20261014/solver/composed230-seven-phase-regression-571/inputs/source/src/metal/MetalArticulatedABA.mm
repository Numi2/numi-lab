#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedABA.hpp"

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

constexpr std::size_t kRawBufferCount = 14u;
constexpr NSUInteger kThreadsPerThreadgroup = 32u;
constexpr float kQuaternionHostTolerance = 1.9e-5f;
constexpr std::uint64_t kShaderAddressableElements =
    static_cast<std::uint64_t>(
        std::numeric_limits<mr_u32>::max()
    ) + 1u;
const char kMetalRoboABAImageAnchor = 0;

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

struct MetalArticulatedABAContextState {
    explicit MetalArticulatedABAContextState(
        MetalArticulatedABAConfig configured
    )
        : config(std::move(configured)) {}

    MetalArticulatedABAConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool inFlight = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> pipeline = nil;
    __strong id<MTLBuffer> buffers[kRawBufferCount] = {};
    std::array<std::size_t, kRawBufferCount> capacities{};
    MetalArticulatedABAContextStats stats{};
};

struct MetalArticulatedABASubmissionState {
    ~MetalArticulatedABASubmissionState() {
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
            // Destruction must not throw. The command is complete, so no GPU
            // work can access the arena even if platform locking fails.
        }
        ownsInFlight = false;
    }

    std::shared_ptr<MetalArticulatedABAContextState> context;
    __strong id<MTLCommandBuffer> commandBuffer = nil;
    MetalArticulatedABADiagnostics diagnostics{};
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
    if (dladdr(&kMetalRoboABAImageAnchor, &image) != 0 &&
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

MetalArticulatedABADiagnostics reject(
    MetalArticulatedABADiagnostics diagnostics,
    const MetalArticulatedABAHostStatus status,
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
            "articulation exceeds the Metal ABA body/DoF/q bucket";
        return false;
    }
    if (articulation.rootType != MR_ROOT_FIXED &&
        articulation.rootType != MR_ROOT_FLOATING) {
        reason = "Metal ABA root type is unsupported";
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
            reason = "Metal ABA joints require zero reserved flags";
            return false;
        }
        if (joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC &&
            joint.jointType != MR_JOINT_FIXED) {
            reason =
                "Metal ABA supports revolute, prismatic, continuous, "
                "and fixed joints";
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
                "every body owned by a Metal ABA articulation "
                "must be dynamic";
            return false;
        }
    }
    return true;
}

bool validQ(
    const MRArticulationGPU& articulation,
    const std::size_t environmentCount,
    const std::span<const float> q
) {
    for (const float value : q) {
        if (!std::isfinite(value)) {
            return false;
        }
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

bool finiteFloats(const std::span<const float> values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

bool validWrenches(
    const std::span<const MRABABodyWrenchGPU> wrenches
) {
    return std::all_of(
        wrenches.begin(),
        wrenches.end(),
        [](const MRABABodyWrenchGPU& wrench) {
            return finite(wrench.force) &&
                finite(wrench.torque) &&
                wrench.force.w == 0.0f &&
                wrench.torque.w == 0.0f;
        }
    );
}

bool buildRequirements(
    const EngineModel& model,
    const MetalArticulatedABALayout& layout,
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
        !makeRequirement<MRABADispatchGPU>(
            "ABA dispatch",
            1u,
            requirements.entries[5]
        ) ||
        !makeRequirement<float>(
            "q",
            layout.qElements,
            requirements.entries[6]
        ) ||
        !makeRequirement<float>(
            "v",
            layout.vElements,
            requirements.entries[7]
        ) ||
        !makeRequirement<float>(
            "effort",
            layout.effortElements,
            requirements.entries[8]
        ) ||
        !makeRequirement<MRABABodyWrenchGPU>(
            "body wrenches",
            layout.wrenchElements,
            requirements.entries[9]
        ) ||
        !makeRequirement<float>(
            "acceleration",
            layout.accelerationElements,
            requirements.entries[10]
        ) ||
        !makeRequirement<float>(
            "next v",
            layout.nextVElements,
            requirements.entries[11]
        ) ||
        !makeRequirement<float>(
            "next q",
            layout.nextQElements,
            requirements.entries[12]
        ) ||
        !makeRequirement<MRABAStatusGPU>(
            "ABA statuses",
            layout.statusElements,
            requirements.entries[13]
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

MetalArticulatedABADiagnostics validateAndBuildLayout(
    const EngineModel& model,
    const MetalArticulatedABAInput& input,
    RequiredBuffers& requirements
) {
    MetalArticulatedABADiagnostics diagnostics{};

    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::invalidModel,
            "invalid EngineModel: " + modelReason
        );
    }
    if (input.articulationIndex >=
        model.articulations.size()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::invalidDimensions,
            "articulation index is outside the canonical model"
        );
    }
    if (input.environmentCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::invalidDimensions,
            "environmentCount must be greater than zero"
        );
    }
    if (input.environmentCount >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::arithmeticOverflow,
            "environmentCount does not fit the GPU dispatch ABI"
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
                ? MetalArticulatedABAHostStatus::capacityOverflow
                : MetalArticulatedABAHostStatus::
                      unsupportedTopology,
            std::move(topologyReason)
        );
    }

    MetalArticulatedABALayout layout{};
    MRABADispatchGPU& dispatch = layout.dispatch;
    dispatch.articulationIndex = input.articulationIndex;
    dispatch.environmentCount =
        static_cast<mr_u32>(input.environmentCount);
    dispatch.flags =
        (input.bodyWrenches.empty()
             ? 0u
             : static_cast<mr_u32>(
                   MR_ABA_HAS_BODY_WRENCHES
               )) |
        (input.applyBodyDamping
             ? static_cast<mr_u32>(
                   MR_ABA_APPLY_BODY_DAMPING
               )
             : 0u);
    dispatch.qStride = articulation.nq;
    dispatch.vStride = articulation.nv;
    dispatch.effortStride = articulation.nv;
    dispatch.wrenchStride = input.bodyWrenches.empty()
        ? 0u
        : articulation.bodyCount;
    dispatch.accelerationStride = articulation.nv;
    dispatch.nextVStride = articulation.nv;
    dispatch.nextQStride = articulation.nq;

    if (!checkedMultiply(
            input.environmentCount,
            dispatch.qStride,
            layout.qElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.vStride,
            layout.vElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.effortStride,
            layout.effortElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.wrenchStride,
            layout.wrenchElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.accelerationStride,
            layout.accelerationElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.nextVStride,
            layout.nextVElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.nextQStride,
            layout.nextQElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::arithmeticOverflow,
            "derived ABA element-count overflow"
        );
    }
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
        exceedsShaderAddressing(layout.vElements) ||
        exceedsShaderAddressing(layout.effortElements) ||
        exceedsShaderAddressing(layout.wrenchElements) ||
        exceedsShaderAddressing(layout.accelerationElements) ||
        exceedsShaderAddressing(layout.nextVElements) ||
        exceedsShaderAddressing(layout.nextQElements) ||
        exceedsShaderAddressing(layout.statusElements)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::arithmeticOverflow,
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
            MetalArticulatedABAHostStatus::arithmeticOverflow,
            "required Metal buffer byte-count overflow"
        );
    }
    layout.qBytes = requirements.entries[6].logicalBytes;
    layout.vBytes = requirements.entries[7].logicalBytes;
    layout.effortBytes = requirements.entries[8].logicalBytes;
    layout.wrenchBytes = requirements.entries[9].logicalBytes;
    layout.accelerationBytes =
        requirements.entries[10].logicalBytes;
    layout.nextVBytes = requirements.entries[11].logicalBytes;
    layout.nextQBytes = requirements.entries[12].logicalBytes;
    layout.statusBytes = requirements.entries[13].logicalBytes;
    layout.totalRequiredBytes = totalRequiredBytes;
    diagnostics.layout = layout;

    if (input.q.size() != layout.qElements ||
        input.v.size() != layout.vElements ||
        input.effort.size() != layout.effortElements ||
        input.bodyWrenches.size() != layout.wrenchElements) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::invalidDimensions,
            "packed q, v, effort, or wrench span has the "
            "wrong element count"
        );
    }
    if (!validQ(
            articulation,
            input.environmentCount,
            input.q
        ) ||
        !finiteFloats(input.v) ||
        !finiteFloats(input.effort)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::nonfiniteInput,
            "state or effort contains a non-finite value or "
            "invalid root quaternion"
        );
    }
    if (!validWrenches(input.bodyWrenches)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::invalidBodyWrench,
            "body wrench is non-finite or has a nonzero reserved w component"
        );
    }
    return diagnostics;
}

NSString* bufferLabel(const std::size_t index) {
    switch (index) {
    case 0u:
        return @"ABA world";
    case 1u:
        return @"ABA articulations";
    case 2u:
        return @"ABA joints";
    case 3u:
        return @"ABA DoF properties";
    case 4u:
        return @"ABA body properties";
    case 5u:
        return @"ABA dispatch";
    case 6u:
        return @"ABA q";
    case 7u:
        return @"ABA v";
    case 8u:
        return @"ABA effort";
    case 9u:
        return @"ABA body wrenches";
    case 10u:
        return @"ABA acceleration output";
    case 11u:
        return @"ABA next v output";
    case 12u:
        return @"ABA next q output";
    case 13u:
        return @"ABA status output";
    default:
        return @"ABA buffer";
    }
}

MetalArticulatedABADiagnostics initializeContext(
    detail::MetalArticulatedABAContextState& context,
    MetalArticulatedABADiagnostics diagnostics
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
            MetalArticulatedABAHostStatus::metallibUnavailable,
            "no ABA metallib path is available"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::
                metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::
                metalDeviceUnsupported,
            "Metal ABA requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::
                metalDeviceUnavailable,
            "failed to create a Metal command queue"
        );
    }
    queue.label = @"MetalRobo ABA queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::metallibUnavailable,
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
            MetalArticulatedABAHostStatus::metalLibraryFailure,
            "failed to load metallib: " + describeError(error)
        );
    }
    id<MTLFunction> function = [library
        newFunctionWithName:@"mr_articulated_aba_step"];
    if (function == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::metalLibraryFailure,
            "metallib does not contain mr_articulated_aba_step"
        );
    }
    error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                       error:&error];
    if (pipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::metalPipelineFailure,
            "failed to create ABA pipeline: " +
                describeError(error)
        );
    }
    if (pipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup ||
        pipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::
                metalDeviceUnsupported,
            "device cannot execute the 32-lane ABA threadgroup"
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

MetalArticulatedABADiagnostics ensureBufferArena(
    detail::MetalArticulatedABAContextState& context,
    const RequiredBuffers& requirements,
    MetalArticulatedABADiagnostics diagnostics
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
                MetalArticulatedABAHostStatus::metalBufferFailure,
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
                MetalArticulatedABAHostStatus::arithmeticOverflow,
                "persistent ABA arena byte-count overflow"
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
                    MetalArticulatedABAHostStatus::
                        arithmeticOverflow,
                    "persistent ABA arena byte-count overflow"
                );
            }
        }
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::metalBufferFailure,
            "persistent ABA arena exceeds "
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
                MetalArticulatedABAHostStatus::metalBufferFailure,
                std::string("persistent Metal buffer growth failed for ") +
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
    detail::MetalArticulatedABAContextState& context,
    const EngineModel& model,
    const MetalArticulatedABAInput& input,
    const MetalArticulatedABALayout& layout,
    const RequiredBuffers& requirements
) {
    MRJointDescriptorGPU emptyJoint{};
    MRABABodyWrenchGPU emptyWrench{};
    const std::array<const void*, 10u> sources{
        &model.world,
        model.articulations.data(),
        model.joints.empty()
            ? static_cast<const void*>(&emptyJoint)
            : static_cast<const void*>(model.joints.data()),
        model.dofs.data(),
        model.bodies.data(),
        &layout.dispatch,
        input.q.data(),
        input.v.data(),
        input.effort.data(),
        input.bodyWrenches.empty()
            ? static_cast<const void*>(&emptyWrench)
            : static_cast<const void*>(
                  input.bodyWrenches.data()
              ),
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
    const MetalArticulatedABAResult& result,
    const std::size_t environment
) {
    const MRABADispatchGPU& dispatch = result.layout.dispatch;
    const auto zeroRange = [](
                               const std::vector<float>& values,
                               const std::size_t begin,
                               const std::size_t count
                           ) {
        return std::all_of(
            values.begin() +
                static_cast<std::ptrdiff_t>(begin),
            values.begin() +
                static_cast<std::ptrdiff_t>(begin + count),
            [](const float value) {
                return value == 0.0f;
            }
        );
    };
    return zeroRange(
               result.acceleration,
               environment * dispatch.accelerationStride,
               dispatch.accelerationStride
           ) &&
        zeroRange(
               result.nextV,
               environment * dispatch.nextVStride,
               dispatch.nextVStride
           ) &&
        zeroRange(
               result.nextQ,
               environment * dispatch.nextQStride,
               dispatch.nextQStride
           );
}

bool finiteEnvironmentPayload(
    const MetalArticulatedABAResult& result,
    const std::size_t environment
) {
    const MRABADispatchGPU& dispatch = result.layout.dispatch;
    const auto finiteRange = [](
                                 const std::vector<float>& values,
                                 const std::size_t begin,
                                 const std::size_t count
                             ) {
        return std::all_of(
            values.begin() +
                static_cast<std::ptrdiff_t>(begin),
            values.begin() +
                static_cast<std::ptrdiff_t>(begin + count),
            [](const float value) {
                return std::isfinite(value);
            }
        );
    };
    return finiteRange(
               result.acceleration,
               environment * dispatch.accelerationStride,
               dispatch.accelerationStride
           ) &&
        finiteRange(
               result.nextV,
               environment * dispatch.nextVStride,
               dispatch.nextVStride
           ) &&
        finiteRange(
               result.nextQ,
               environment * dispatch.nextQStride,
               dispatch.nextQStride
           );
}

MetalArticulatedABADiagnostics validateAndPublish(
    const MRArticulationGPU& articulation,
    MetalArticulatedABAResult&& staged,
    MetalArticulatedABADiagnostics diagnostics,
    MetalArticulatedABAResult& result
) {
    const MRABADispatchGPU& dispatch =
        diagnostics.layout.dispatch;
    for (std::size_t environment = 0u;
         environment < staged.statuses.size();
         ++environment) {
        const MRABAStatusGPU& status =
            staged.statuses[environment];
        if (status.environment != environment ||
            status.articulationIndex !=
                dispatch.articulationIndex ||
            status.code > MR_ABA_UNSUPPORTED_TOPOLOGY ||
            status.bodyCount != articulation.bodyCount ||
            status.nq != articulation.nq ||
            status.nv != articulation.nv ||
            status.flags != dispatch.flags) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedABAHostStatus::internalFailure,
                "GPU returned a malformed ABA status record"
            );
        }
        if (status.code == MR_ABA_SUCCESS) {
            if (status.failingIndex != MR_INVALID_INDEX ||
                !finite(status.diagnostics) ||
                !finiteEnvironmentPayload(staged, environment)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedABAHostStatus::internalFailure,
                    "GPU success status has invalid diagnostics "
                    "or payload"
                );
            }
            ++diagnostics.successfulEnvironmentCount;
        } else {
            if (!zeroEnvironmentPayload(staged, environment)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedABAHostStatus::internalFailure,
                    "failed GPU environment published partial ABA state"
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
            MetalArticulatedABAHostStatus::gpuEnvironmentFailure,
            "one or more GPU environments rejected ABA execution"
        );
    }
    diagnostics.status = MetalArticulatedABAHostStatus::success;
    diagnostics.message.clear();
    return diagnostics;
}

} // namespace

MetalArticulatedABASubmission::
    MetalArticulatedABASubmission() noexcept = default;

MetalArticulatedABASubmission::
    ~MetalArticulatedABASubmission() = default;

MetalArticulatedABASubmission::
    MetalArticulatedABASubmission(
        MetalArticulatedABASubmission&& other
    ) noexcept = default;

MetalArticulatedABASubmission&
MetalArticulatedABASubmission::operator=(
    MetalArticulatedABASubmission&& other
) noexcept = default;

bool MetalArticulatedABASubmission::valid() const noexcept {
    return state_ != nullptr;
}

MetalArticulatedABADiagnostics
MetalArticulatedABASubmission::wait(
    MetalArticulatedABAResult& result
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalArticulatedABAHostStatus::metalCommandFailure,
            "ABA submission is empty or has already been consumed"
        );
    }

    std::unique_ptr<
        detail::MetalArticulatedABASubmissionState
    > pending = std::move(state_);
    MetalArticulatedABADiagnostics diagnostics =
        pending->diagnostics;
    try {
        MetalArticulatedABAResult staged{};
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
                    MetalArticulatedABAHostStatus::
                        metalCommandFailure,
                    "Metal ABA command failed: " +
                        describeError(pending->commandBuffer.error)
                );
            }

            staged.layout = diagnostics.layout;
            staged.acceleration.resize(
                staged.layout.accelerationElements
            );
            staged.nextV.resize(staged.layout.nextVElements);
            staged.nextQ.resize(staged.layout.nextQElements);
            staged.statuses.resize(staged.layout.statusElements);
            const auto& buffers = pending->context->buffers;
            copyOutput(staged.acceleration, buffers[10]);
            copyOutput(staged.nextV, buffers[11]);
            copyOutput(staged.nextQ, buffers[12]);
            copyOutput(staged.statuses, buffers[13]);
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
            MetalArticulatedABAHostStatus::metalBufferFailure,
            "host allocation failed while publishing ABA results"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedABAContext::MetalArticulatedABAContext(
    MetalArticulatedABAConfig config
)
    : state_(std::make_shared<
          detail::MetalArticulatedABAContextState
      >(std::move(config))) {}

MetalArticulatedABAContext::~MetalArticulatedABAContext() =
    default;

MetalArticulatedABAContext::MetalArticulatedABAContext(
    MetalArticulatedABAContext&& other
) noexcept = default;

MetalArticulatedABAContext&
MetalArticulatedABAContext::operator=(
    MetalArticulatedABAContext&& other
) noexcept = default;

MetalArticulatedABADiagnostics MetalArticulatedABAContext::submit(
    const EngineModel& model,
    const MetalArticulatedABAInput& input,
    MetalArticulatedABASubmission& submission
) {
    MetalArticulatedABADiagnostics diagnostics{};
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::internalFailure,
            "ABA context was moved from"
        );
    }
    if (submission.valid()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::contextBusy,
            "ABA submission output already owns an in-flight batch"
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
                MetalArticulatedABAHostStatus::contextBusy,
                "ABA context already has an in-flight batch"
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
                    MetalArticulatedABAHostStatus::
                        metalCommandFailure,
                    "failed to create ABA command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo persistent articulated ABA";
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedABAHostStatus::
                        metalCommandFailure,
                    "failed to create ABA compute encoder"
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
                detail::MetalArticulatedABASubmissionState
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
            MetalArticulatedABAHostStatus::metalBufferFailure,
            "host allocation failed while preparing ABA submission"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedABAHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedABADiagnostics MetalArticulatedABAContext::run(
    const EngineModel& model,
    const MetalArticulatedABAInput& input,
    MetalArticulatedABAResult& result
) {
    MetalArticulatedABASubmission submission;
    MetalArticulatedABADiagnostics diagnostics = submit(
        model,
        input,
        submission
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return submission.wait(result);
}

MetalArticulatedABAContextStats
MetalArticulatedABAContext::stats() const noexcept {
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

MetalArticulatedABADiagnostics runMetalArticulatedABA(
    const EngineModel& model,
    const MetalArticulatedABAInput& input,
    MetalArticulatedABAResult& result,
    const MetalArticulatedABAConfig& config
) {
    try {
        MetalArticulatedABAContext context(config);
        return context.run(model, input, result);
    } catch (const std::bad_alloc&) {
        return reject(
            {},
            MetalArticulatedABAHostStatus::metalBufferFailure,
            "host allocation failed while creating ABA context"
        );
    } catch (const std::exception& exception) {
        return reject(
            {},
            MetalArticulatedABAHostStatus::internalFailure,
            exception.what()
        );
    }
}

const char* metalArticulatedABAHostStatusName(
    const MetalArticulatedABAHostStatus status
) noexcept {
    switch (status) {
    case MetalArticulatedABAHostStatus::success:
        return "success";
    case MetalArticulatedABAHostStatus::invalidModel:
        return "invalid_model";
    case MetalArticulatedABAHostStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalArticulatedABAHostStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalArticulatedABAHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalArticulatedABAHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalArticulatedABAHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalArticulatedABAHostStatus::invalidBodyWrench:
        return "invalid_body_wrench";
    case MetalArticulatedABAHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalArticulatedABAHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalArticulatedABAHostStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalArticulatedABAHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalArticulatedABAHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalArticulatedABAHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalArticulatedABAHostStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalArticulatedABAHostStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalArticulatedABAHostStatus::internalFailure:
        return "internal_failure";
    case MetalArticulatedABAHostStatus::contextBusy:
        return "context_busy";
    }
    return "unknown";
}

} // namespace metalrobo
