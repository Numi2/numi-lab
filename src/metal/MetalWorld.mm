#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalWorld.hpp"

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

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr std::size_t kRawBufferCount = 27u;
constexpr NSUInteger kABAThreadsPerThreadgroup = 32u;
constexpr NSUInteger kWorldThreadsPerThreadgroup = 64u;
constexpr mr_u32 kSmallABAMaxBodies = 12u;
constexpr mr_u32 kSmallABAMaxDofs = 16u;
constexpr mr_u32 kSmallABAMaxQ = 17u;
constexpr float kQuaternionHostTolerance = 1.9e-5f;
constexpr std::uint64_t kShaderAddressableElements =
    static_cast<std::uint64_t>(
        std::numeric_limits<mr_u32>::max()
    ) + 1u;
constexpr std::uint64_t kFNVOffset =
    14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
const char kMetalRoboWorldImageAnchor = 0;

enum BufferIndex : std::size_t {
    kWorld = 0u,
    kArticulations = 1u,
    kJoints = 2u,
    kDofs = 3u,
    kBodies = 4u,
    kABADispatch = 5u,
    kStateQA = 6u,
    kStateVA = 7u,
    kWorkingEffort = 8u,
    kBodyWrenchPlaceholder = 9u,
    kCandidateAcceleration = 10u,
    kCandidateV = 11u,
    kCandidateQ = 12u,
    kABAStatuses = 13u,
    kStateQB = 14u,
    kStateVB = 15u,
    kEffortTrajectory = 16u,
    kResetMasks = 17u,
    kResetQ = 18u,
    kResetV = 19u,
    kObservations = 20u,
    kAccelerationTrajectory = 21u,
    kPublicStatuses = 22u,
    kWorldDispatch = 23u,
    kEnvironmentStatuses = 24u,
    kCheckpointQ = 25u,
    kCheckpointV = 26u,
};

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

struct MetalWorldContextState {
    explicit MetalWorldContextState(MetalWorldConfig configured)
        : config(std::move(configured)) {}

    MetalWorldConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool inFlight = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> abaPipeline = nil;
    __strong id<MTLComputePipelineState> smallABAPipeline = nil;
    __strong id<MTLComputePipelineState> preparePipeline = nil;
    __strong id<MTLComputePipelineState> commitPipeline = nil;
    __strong id<MTLComputePipelineState> capturePipeline = nil;
    __strong id<MTLBuffer> buffers[kRawBufferCount] = {};
    std::array<std::size_t, kRawBufferCount> capacities{};
    std::uint64_t boundModelFingerprint = 0u;
    MetalWorldContextStats stats{};
};

struct MetalWorldSubmissionState {
    ~MetalWorldSubmissionState() {
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
            // Destruction cannot throw. Completion makes the shared arena
            // safe even if platform locking itself were to fail.
        }
        ownsInFlight = false;
    }

    std::shared_ptr<MetalWorldContextState> context;
    __strong id<MTLCommandBuffer> commandBuffer = nil;
    MetalWorldDiagnostics diagnostics{};
    std::chrono::steady_clock::time_point start{};
    MRArticulationGPU articulation{};
    std::size_t finalQBuffer = kStateQA;
    std::size_t finalVBuffer = kStateVA;
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

std::string thermalStateName(
    const NSProcessInfoThermalState state
) {
    switch (state) {
    case NSProcessInfoThermalStateNominal:
        return "nominal";
    case NSProcessInfoThermalStateFair:
        return "fair";
    case NSProcessInfoThermalStateSerious:
        return "serious";
    case NSProcessInfoThermalStateCritical:
        return "critical";
    }
    return "unknown";
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalRoboWorldImageAnchor, &image) != 0 &&
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

MetalWorldDiagnostics reject(
    MetalWorldDiagnostics diagnostics,
    const MetalWorldHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

MetalWorldCompileDiagnostics rejectCompile(
    MetalWorldCompileDiagnostics diagnostics,
    const MetalWorldHostStatus status,
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
        right >
            std::numeric_limits<std::size_t>::max() / left) {
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
        articulation.nq == 0u ||
        articulation.nq > MR_ARTICULATED_ABA_MAX_Q) {
        reason =
            "articulation exceeds the Metal ABA body, DoF, or q bucket";
        return false;
    }
    if (articulation.rootType != MR_ROOT_FIXED &&
        articulation.rootType != MR_ROOT_FLOATING) {
        reason = "Metal world root type is unsupported";
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
                "Metal world joints require zero reserved flags";
            return false;
        }
        if (joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC &&
            joint.jointType != MR_JOINT_FIXED) {
            reason =
                "free-motion Metal world supports revolute, "
                "continuous, prismatic, and fixed joints";
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
                "every body in the selected articulation must be dynamic";
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

void hashBytes(
    std::uint64_t& hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFNVPrime;
    }
}

template <typename T>
void hashValue(std::uint64_t& hash, const T& value) {
    hashBytes(hash, &value, sizeof(value));
}

template <typename T>
void hashVector(
    std::uint64_t& hash,
    const std::vector<T>& values
) {
    hashValue(hash, values.size());
    if (!values.empty()) {
        hashBytes(
            hash,
            values.data(),
            values.size() * sizeof(T)
        );
    }
}

std::uint64_t fingerprint(const EngineModel& model) {
    std::uint64_t hash = kFNVOffset;
    hashValue(hash, MR_ENGINE_ABI_VERSION);
    hashValue(hash, model.world);
    hashVector(hash, model.articulations);
    hashVector(hash, model.joints);
    hashVector(hash, model.dofs);
    hashVector(hash, model.bodies);
    hashVector(hash, model.shapes);
    hashVector(hash, model.materials);
    hashVector(hash, model.defaultQ);
    hashVector(hash, model.defaultV);
    hashValue(hash, model.name.size());
    hashBytes(hash, model.name.data(), model.name.size());
    return hash == 0u ? 1u : hash;
}

bool buildRequirements(
    const CompiledWorld& world,
    const MetalWorldLayout& layout,
    RequiredBuffers& requirements,
    std::size_t& totalRequiredBytes
) {
    const EngineModel& model = world.model();
    const std::size_t jointElements =
        std::max<std::size_t>(model.joints.size(), 1u);
    const std::size_t resetMaskElements =
        layout.resetMaskElements;
    const std::size_t resetQElements = layout.resetQElements;
    const std::size_t resetVElements = layout.resetVElements;
    if (!makeRequirement<MRWorldGPU>(
            "runtime world",
            1u,
            requirements.entries[kWorld]
        ) ||
        !makeRequirement<MRArticulationGPU>(
            "articulations",
            model.articulations.size(),
            requirements.entries[kArticulations]
        ) ||
        !makeRequirement<MRJointDescriptorGPU>(
            "joints",
            jointElements,
            requirements.entries[kJoints]
        ) ||
        !makeRequirement<MRDofPropertiesGPU>(
            "DoF properties",
            model.dofs.size(),
            requirements.entries[kDofs]
        ) ||
        !makeRequirement<MRBodyPropertiesGPU>(
            "body properties",
            model.bodies.size(),
            requirements.entries[kBodies]
        ) ||
        !makeRequirement<MRABADispatchGPU>(
            "ABA dispatch",
            1u,
            requirements.entries[kABADispatch]
        ) ||
        !makeRequirement<float>(
            "state q A",
            layout.initialQElements,
            requirements.entries[kStateQA]
        ) ||
        !makeRequirement<float>(
            "state v A",
            layout.initialVElements,
            requirements.entries[kStateVA]
        ) ||
        !makeRequirement<float>(
            "working effort",
            layout.initialVElements,
            requirements.entries[kWorkingEffort]
        ) ||
        !makeRequirement<MRABABodyWrenchGPU>(
            "body-wrench placeholder",
            0u,
            requirements.entries[kBodyWrenchPlaceholder]
        ) ||
        !makeRequirement<float>(
            "candidate acceleration",
            layout.initialVElements,
            requirements.entries[kCandidateAcceleration]
        ) ||
        !makeRequirement<float>(
            "candidate v",
            layout.initialVElements,
            requirements.entries[kCandidateV]
        ) ||
        !makeRequirement<float>(
            "candidate q",
            layout.initialQElements,
            requirements.entries[kCandidateQ]
        ) ||
        !makeRequirement<MRABAStatusGPU>(
            "ABA statuses",
            layout.dispatch.environmentCount,
            requirements.entries[kABAStatuses]
        ) ||
        !makeRequirement<float>(
            "state q B",
            layout.initialQElements,
            requirements.entries[kStateQB]
        ) ||
        !makeRequirement<float>(
            "state v B",
            layout.initialVElements,
            requirements.entries[kStateVB]
        ) ||
        !makeRequirement<float>(
            "effort trajectory",
            layout.effortElements,
            requirements.entries[kEffortTrajectory]
        ) ||
        !makeRequirement<mr_u32>(
            "reset masks",
            resetMaskElements,
            requirements.entries[kResetMasks]
        ) ||
        !makeRequirement<float>(
            "reset q",
            resetQElements,
            requirements.entries[kResetQ]
        ) ||
        !makeRequirement<float>(
            "reset v",
            resetVElements,
            requirements.entries[kResetV]
        ) ||
        !makeRequirement<float>(
            "observations",
            layout.observationElements,
            requirements.entries[kObservations]
        ) ||
        !makeRequirement<float>(
            "acceleration trajectory",
            layout.accelerationElements,
            requirements.entries[kAccelerationTrajectory]
        ) ||
        !makeRequirement<MRMetalWorldStatusGPU>(
            "public statuses",
            layout.statusElements,
            requirements.entries[kPublicStatuses]
        ) ||
        !makeRequirement<MRMetalWorldDispatchGPU>(
            "world dispatch",
            1u,
            requirements.entries[kWorldDispatch]
        ) ||
        !makeRequirement<MRMetalWorldStatusGPU>(
            "environment statuses",
            layout.dispatch.environmentCount,
            requirements.entries[kEnvironmentStatuses]
        ) ||
        !makeRequirement<float>(
            "checkpoint q",
            layout.initialQElements,
            requirements.entries[kCheckpointQ]
        ) ||
        !makeRequirement<float>(
            "checkpoint v",
            layout.initialVElements,
            requirements.entries[kCheckpointV]
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

MetalWorldDiagnostics validateAndBuildLayout(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    RequiredBuffers& requirements
) {
    MetalWorldDiagnostics diagnostics{};
    if (!world.valid()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidModel,
            "CompiledWorld is empty or invalid"
        );
    }
    if (config.solverMode !=
        MetalWorldSolverMode::freeMotionABA) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::unsupportedSolverMode,
            "contact PGS/TGS are reserved but not executable in "
            "MetalWorld ABI v1"
        );
    }
    if (!std::isfinite(config.timestepSeconds) ||
        !(config.timestepSeconds > 0.0f) ||
        config.physicsSubsteps == 0u ||
        config.physicsSubsteps >
            MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "timestepSeconds must be finite and positive and "
            "physicsSubsteps must be in the supported range"
        );
    }
    const float substepTimestep =
        config.timestepSeconds /
        static_cast<float>(config.physicsSubsteps);
    if (!std::isfinite(substepTimestep) ||
        substepTimestep <
            std::numeric_limits<float>::min()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "derived physics substep timestep is not a positive "
            "normal FP32 value"
        );
    }
    if (batch.environmentCount == 0u ||
        batch.controlStepCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "environmentCount and controlStepCount must be nonzero"
        );
    }
    if (batch.environmentCount >
            std::numeric_limits<mr_u32>::max() ||
        batch.controlStepCount >
            std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "batch dimensions do not fit the GPU ABI"
        );
    }

    const MRArticulationGPU& articulation =
        world.model().articulations[world.articulationIndex()];
    MetalWorldLayout layout{};
    MRMetalWorldDispatchGPU& dispatch = layout.dispatch;
    dispatch.abiVersion = MR_METAL_WORLD_ABI_VERSION;
    dispatch.articulationIndex = world.articulationIndex();
    dispatch.environmentCount =
        static_cast<mr_u32>(batch.environmentCount);
    dispatch.controlStepCount =
        static_cast<mr_u32>(batch.controlStepCount);
    dispatch.physicsSubsteps = config.physicsSubsteps;
    dispatch.flags = MR_METAL_WORLD_FREE_MOTION_ONLY |
        (config.applyBodyDamping
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_APPLY_BODY_DAMPING
               )
             : 0u) |
        (config.deterministic
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_DETERMINISTIC
               )
             : 0u) |
        (!batch.resetMasks.empty()
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_HAS_RESETS
               )
             : 0u);
    dispatch.nq = articulation.nq;
    dispatch.nv = articulation.nv;
    dispatch.qStride = articulation.nq;
    dispatch.vStride = articulation.nv;
    dispatch.effortEnvironmentStride = articulation.nv;
    dispatch.observationEnvironmentStride =
        articulation.nq + articulation.nv;

    std::size_t effortStepStride = 0u;
    std::size_t resetMaskStepStride = 0u;
    std::size_t observationStepStride = 0u;
    std::size_t accelerationStepStride = 0u;
    if (!checkedMultiply(
            batch.environmentCount,
            dispatch.effortEnvironmentStride,
            effortStepStride
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            dispatch.observationEnvironmentStride,
            observationStepStride
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            articulation.nv,
            accelerationStepStride
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived Metal world step-stride overflow"
        );
    }
    resetMaskStepStride = batch.resetMasks.empty()
        ? 0u
        : batch.environmentCount;
    const auto fitsU32 = [](const std::size_t value) {
        return value <= std::numeric_limits<mr_u32>::max();
    };
    if (!fitsU32(effortStepStride) ||
        !fitsU32(resetMaskStepStride) ||
        !fitsU32(observationStepStride) ||
        !fitsU32(accelerationStepStride)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived Metal world stride does not fit 32-bit ABI"
        );
    }
    dispatch.effortStepStride =
        static_cast<mr_u32>(effortStepStride);
    dispatch.resetMaskStepStride =
        static_cast<mr_u32>(resetMaskStepStride);
    dispatch.observationStepStride =
        static_cast<mr_u32>(observationStepStride);
    dispatch.accelerationStepStride =
        static_cast<mr_u32>(accelerationStepStride);

    MRABADispatchGPU& aba = layout.abaDispatch;
    aba.articulationIndex = world.articulationIndex();
    aba.environmentCount = dispatch.environmentCount;
    aba.flags = config.applyBodyDamping
        ? static_cast<mr_u32>(
              MR_ABA_APPLY_BODY_DAMPING
          )
        : 0u;
    aba.qStride = articulation.nq;
    aba.vStride = articulation.nv;
    aba.effortStride = articulation.nv;
    aba.wrenchStride = 0u;
    aba.accelerationStride = articulation.nv;
    aba.nextVStride = articulation.nv;
    aba.nextQStride = articulation.nq;

    if (!checkedMultiply(
            batch.environmentCount,
            articulation.nq,
            layout.initialQElements
        ) ||
        !checkedMultiply(
            batch.environmentCount,
            articulation.nv,
            layout.initialVElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            effortStepStride,
            layout.effortElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            resetMaskStepStride,
            layout.resetMaskElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            observationStepStride,
            layout.observationElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            accelerationStepStride,
            layout.accelerationElements
        ) ||
        !checkedMultiply(
            batch.controlStepCount,
            batch.environmentCount,
            layout.statusElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "derived Metal world element-count overflow"
        );
    }
    layout.resetQElements = batch.resetMasks.empty()
        ? 0u
        : layout.initialQElements;
    layout.resetVElements = batch.resetMasks.empty()
        ? 0u
        : layout.initialVElements;

    const std::array shaderElementCounts{
        world.model().articulations.size(),
        world.model().joints.size(),
        world.model().dofs.size(),
        world.model().bodies.size(),
        layout.initialQElements,
        layout.initialVElements,
        layout.effortElements,
        layout.resetMaskElements,
        layout.resetQElements,
        layout.resetVElements,
        layout.observationElements,
        layout.accelerationElements,
        layout.statusElements,
    };
    if (std::any_of(
            shaderElementCounts.begin(),
            shaderElementCounts.end(),
            [](const std::size_t count) {
                return static_cast<std::uint64_t>(count) >
                    kShaderAddressableElements;
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "compact buffer exceeds the shader's 32-bit "
            "element-addressing contract"
        );
    }

    std::size_t totalRequiredBytes = 0u;
    if (!buildRequirements(
            world,
            layout,
            requirements,
            totalRequiredBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "required persistent Metal arena byte-count overflow"
        );
    }
    layout.totalRequiredBytes = totalRequiredBytes;
    diagnostics.layout = layout;

    if (batch.initialQ.size() != layout.initialQElements ||
        batch.initialV.size() != layout.initialVElements ||
        batch.efforts.size() != layout.effortElements) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "initial q/v or effort trajectory has the wrong "
            "packed element count"
        );
    }
    if (batch.resetMasks.empty()) {
        if (!batch.resetQ.empty() || !batch.resetV.empty()) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidReset,
                "reset states require a control-step reset mask"
            );
        }
    } else if (
        batch.resetMasks.size() != layout.resetMaskElements ||
        batch.resetQ.size() != layout.resetQElements ||
        batch.resetV.size() != layout.resetVElements) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset mask or environment reset state has the "
            "wrong packed element count"
        );
    }
    if (!std::all_of(
            batch.resetMasks.begin(),
            batch.resetMasks.end(),
            [](const std::uint32_t value) {
                return value <= 1u;
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset masks must contain only zero or one"
        );
    }
    if (!validQ(
            articulation,
            batch.environmentCount,
            batch.initialQ
        ) ||
        !finiteFloats(batch.initialV) ||
        !finiteFloats(batch.efforts)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::nonfiniteInput,
            "initial state or effort contains a non-finite value "
            "or invalid floating-root quaternion"
        );
    }
    if (!batch.resetMasks.empty() &&
        (!validQ(
             articulation,
             batch.environmentCount,
             batch.resetQ
         ) ||
         !finiteFloats(batch.resetV))) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidReset,
            "reset state is non-finite or has an invalid "
            "floating-root quaternion"
        );
    }
    return diagnostics;
}

NSString* bufferLabel(const std::size_t index) {
    switch (index) {
    case kWorld:
        return @"MetalWorld runtime world";
    case kArticulations:
        return @"MetalWorld articulations";
    case kJoints:
        return @"MetalWorld joints";
    case kDofs:
        return @"MetalWorld DoF properties";
    case kBodies:
        return @"MetalWorld body properties";
    case kABADispatch:
        return @"MetalWorld ABA dispatch";
    case kStateQA:
        return @"MetalWorld state q A";
    case kStateVA:
        return @"MetalWorld state v A";
    case kWorkingEffort:
        return @"MetalWorld working effort";
    case kBodyWrenchPlaceholder:
        return @"MetalWorld body-wrench placeholder";
    case kCandidateAcceleration:
        return @"MetalWorld candidate acceleration";
    case kCandidateV:
        return @"MetalWorld candidate v";
    case kCandidateQ:
        return @"MetalWorld candidate q";
    case kABAStatuses:
        return @"MetalWorld ABA statuses";
    case kStateQB:
        return @"MetalWorld state q B";
    case kStateVB:
        return @"MetalWorld state v B";
    case kEffortTrajectory:
        return @"MetalWorld effort trajectory";
    case kResetMasks:
        return @"MetalWorld reset masks";
    case kResetQ:
        return @"MetalWorld reset q";
    case kResetV:
        return @"MetalWorld reset v";
    case kObservations:
        return @"MetalWorld observations";
    case kAccelerationTrajectory:
        return @"MetalWorld acceleration trajectory";
    case kPublicStatuses:
        return @"MetalWorld public statuses";
    case kWorldDispatch:
        return @"MetalWorld dispatch";
    case kEnvironmentStatuses:
        return @"MetalWorld environment statuses";
    case kCheckpointQ:
        return @"MetalWorld checkpoint q";
    case kCheckpointV:
        return @"MetalWorld checkpoint v";
    default:
        return @"MetalWorld buffer";
    }
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* functionName,
    NSError** error
) {
    id<MTLFunction> function =
        [library newFunctionWithName:functionName];
    if (function == nil) {
        if (error != nullptr) {
            *error = [NSError
                errorWithDomain:@"MetalRobo.MetalWorld"
                           code:1
                       userInfo:@{
                           NSLocalizedDescriptionKey:
                               [NSString
                                   stringWithFormat:
                                       @"metallib does not contain %@",
                                       functionName]
                       }];
        }
        return nil;
    }
    return [device
        newComputePipelineStateWithFunction:function
                                      error:error];
}

MetalWorldDiagnostics initializeContext(
    detail::MetalWorldContextState& context,
    MetalWorldDiagnostics diagnostics
) {
    if (context.initialized) {
        diagnostics.deviceName = nsString(context.device.name);
        diagnostics.thermalState = thermalStateName(
            [NSProcessInfo processInfo].thermalState
        );
        return diagnostics;
    }

    std::string metallibPath = context.config.metallibPath;
    if (metallibPath.empty()) {
        metallibPath = defaultMetallibPath();
    }
    if (metallibPath.empty()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metallibUnavailable,
            "no MetalWorld metallib path is available"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    diagnostics.thermalState = thermalStateName(
        [NSProcessInfo processInfo].thermalState
    );
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnsupported,
            "MetalWorld requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnavailable,
            "failed to create MetalWorld command queue"
        );
    }
    queue.label = @"MetalRobo persistent world queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metallibUnavailable,
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
            MetalWorldHostStatus::metalLibraryFailure,
            "failed to load MetalWorld metallib: " +
                describeError(error)
        );
    }

    error = nil;
    id<MTLComputePipelineState> aba = makePipeline(
        device,
        library,
        @"mr_articulated_aba_step",
        &error
    );
    if (aba == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create ABA pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> smallABA = makePipeline(
        device,
        library,
        @"mr_articulated_aba_step_small",
        &error
    );
    if (smallABA == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create small-capacity ABA pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> prepare = makePipeline(
        device,
        library,
        @"mr_metal_world_prepare",
        &error
    );
    if (prepare == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld prepare pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> commit = makePipeline(
        device,
        library,
        @"mr_metal_world_commit",
        &error
    );
    if (commit == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld commit pipeline: " +
                describeError(error)
        );
    }
    error = nil;
    id<MTLComputePipelineState> capture = makePipeline(
        device,
        library,
        @"mr_metal_world_capture",
        &error
    );
    if (capture == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create MetalWorld capture pipeline: " +
                describeError(error)
        );
    }

    if (aba.maxTotalThreadsPerThreadgroup <
            kABAThreadsPerThreadgroup ||
        smallABA.maxTotalThreadsPerThreadgroup <
            kABAThreadsPerThreadgroup ||
        aba.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        smallABA.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        prepare.maxTotalThreadsPerThreadgroup == 0u ||
        commit.maxTotalThreadsPerThreadgroup == 0u ||
        capture.maxTotalThreadsPerThreadgroup == 0u) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalDeviceUnsupported,
            "device cannot execute the MetalWorld kernel geometry"
        );
    }

    context.device = device;
    context.queue = queue;
    context.library = library;
    context.abaPipeline = aba;
    context.smallABAPipeline = smallABA;
    context.preparePipeline = prepare;
    context.commitPipeline = commit;
    context.capturePipeline = capture;
    context.initialized = true;
    context.stats.pipelineCreationCount += 5u;
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

MetalWorldDiagnostics ensureBufferArena(
    detail::MetalWorldContextState& context,
    const RequiredBuffers& requirements,
    MetalWorldDiagnostics diagnostics
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
                MetalWorldHostStatus::metalBufferFailure,
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
                MetalWorldHostStatus::arithmeticOverflow,
                "persistent MetalWorld arena byte-count overflow"
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
                    MetalWorldHostStatus::arithmeticOverflow,
                    "persistent MetalWorld arena byte-count overflow"
                );
            }
        }
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "persistent MetalWorld arena exceeds "
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
                MetalWorldHostStatus::metalBufferFailure,
                std::string("persistent Metal buffer growth failed for ") +
                    requirements.entries[index].label
            );
        }
        replacements[index].label = bufferLabel(index);
    }

    bool immutableBufferReplaced = false;
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
        immutableBufferReplaced =
            immutableBufferReplaced ||
            (index >= kArticulations && index <= kBodies);
    }
    if (immutableBufferReplaced) {
        context.boundModelFingerprint = 0u;
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

void zeroBuffer(
    id<MTLBuffer> destination,
    const BufferRequirement& requirement
) {
    std::memset(
        destination.contents,
        0,
        requirement.allocationBytes
    );
}

void uploadBatch(
    detail::MetalWorldContextState& context,
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    const MetalWorldLayout& layout,
    const RequiredBuffers& requirements
) {
    const EngineModel& model = world.model();
    MRWorldGPU runtimeWorld = model.world;
    runtimeWorld.gravityAndTimestep.w =
        config.timestepSeconds /
        static_cast<float>(config.physicsSubsteps);
    copyToBuffer(
        context.buffers[kWorld],
        &runtimeWorld,
        requirements.entries[kWorld]
    );

    if (context.boundModelFingerprint !=
        world.fingerprint()) {
        MRJointDescriptorGPU emptyJoint{};
        const std::array<const void*, 4u> sources{
            model.articulations.data(),
            model.joints.empty()
                ? static_cast<const void*>(&emptyJoint)
                : static_cast<const void*>(
                      model.joints.data()
                  ),
            model.dofs.data(),
            model.bodies.data(),
        };
        for (std::size_t offset = 0u;
             offset < sources.size();
             ++offset) {
            const std::size_t index =
                kArticulations + offset;
            copyToBuffer(
                context.buffers[index],
                sources[offset],
                requirements.entries[index]
            );
        }
        context.boundModelFingerprint = world.fingerprint();
        ++context.stats.modelUploadCount;
    }

    copyToBuffer(
        context.buffers[kABADispatch],
        &layout.abaDispatch,
        requirements.entries[kABADispatch]
    );
    copyToBuffer(
        context.buffers[kStateQA],
        batch.initialQ.data(),
        requirements.entries[kStateQA]
    );
    copyToBuffer(
        context.buffers[kStateVA],
        batch.initialV.data(),
        requirements.entries[kStateVA]
    );
    copyToBuffer(
        context.buffers[kEffortTrajectory],
        batch.efforts.data(),
        requirements.entries[kEffortTrajectory]
    );
    copyToBuffer(
        context.buffers[kResetMasks],
        batch.resetMasks.data(),
        requirements.entries[kResetMasks]
    );
    copyToBuffer(
        context.buffers[kResetQ],
        batch.resetQ.data(),
        requirements.entries[kResetQ]
    );
    copyToBuffer(
        context.buffers[kResetV],
        batch.resetV.data(),
        requirements.entries[kResetV]
    );
    copyToBuffer(
        context.buffers[kWorldDispatch],
        &layout.dispatch,
        requirements.entries[kWorldDispatch]
    );

    const std::array<std::size_t, 13u> scratch{
        kWorkingEffort,
        kBodyWrenchPlaceholder,
        kCandidateAcceleration,
        kCandidateV,
        kCandidateQ,
        kABAStatuses,
        kStateQB,
        kStateVB,
        kObservations,
        kAccelerationTrajectory,
        kPublicStatuses,
        kEnvironmentStatuses,
        kCheckpointQ,
    };
    for (const std::size_t index : scratch) {
        zeroBuffer(
            context.buffers[index],
            requirements.entries[index]
        );
    }
    zeroBuffer(
        context.buffers[kCheckpointV],
        requirements.entries[kCheckpointV]
    );
}

NSUInteger worldThreadWidth(
    id<MTLComputePipelineState> pipeline
) {
    return std::max<NSUInteger>(
        1u,
        std::min<NSUInteger>(
            kWorldThreadsPerThreadgroup,
            pipeline.maxTotalThreadsPerThreadgroup
        )
    );
}

void dispatchWorldThreads(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const std::size_t environmentCount
) {
    [encoder
        dispatchThreads:MTLSizeMake(
            static_cast<NSUInteger>(environmentCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            worldThreadWidth(pipeline),
            1u,
            1u
        )];
}

bool encodePrepare(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld prepare/reset";
    [encoder setComputePipelineState:context.preparePipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[kEffortTrajectory]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[kResetMasks]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kResetQ]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kResetV]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kCheckpointQ]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kCheckpointV]
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:context.buffers[kWorkingEffort]
                 offset:0u
                atIndex:10u];
    [encoder setBuffer:context.buffers[kEnvironmentStatuses]
                 offset:0u
                atIndex:11u];
    dispatchWorldThreads(
        encoder,
        context.preparePipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
}

bool encodeABA(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld ABA";
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:context.buffers[kWorld]
                 offset:0u
                atIndex:0u];
    [encoder setBuffer:context.buffers[kArticulations]
                 offset:0u
                atIndex:1u];
    [encoder setBuffer:context.buffers[kJoints]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[kDofs]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kBodies]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kABADispatch]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kWorkingEffort]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kBodyWrenchPlaceholder]
                 offset:0u
                atIndex:9u];
    [encoder setBuffer:context.buffers[kCandidateAcceleration]
                 offset:0u
                atIndex:10u];
    [encoder setBuffer:context.buffers[kCandidateV]
                 offset:0u
                atIndex:11u];
    [encoder setBuffer:context.buffers[kCandidateQ]
                 offset:0u
                atIndex:12u];
    [encoder setBuffer:context.buffers[kABAStatuses]
                 offset:0u
                atIndex:13u];
    [encoder
        dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(environmentCount),
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(
            kABAThreadsPerThreadgroup,
            1u,
            1u
        )];
    [encoder endEncoding];
    return true;
}

bool encodeCommit(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t destinationQ,
    const std::size_t destinationV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld transactional commit";
    [encoder setComputePipelineState:context.commitPipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[kABAStatuses]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[kCandidateQ]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kCandidateV]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[destinationQ]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[destinationV]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[kEnvironmentStatuses]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kCheckpointQ]
                 offset:0u
                atIndex:8u];
    [encoder setBuffer:context.buffers[kCheckpointV]
                 offset:0u
                atIndex:9u];
    dispatchWorldThreads(
        encoder,
        context.commitPipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
}

bool encodeCapture(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceQ,
    const std::size_t sourceV,
    const std::size_t environmentCount
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = @"MetalWorld observation capture";
    [encoder setComputePipelineState:context.capturePipeline];
    [encoder setBuffer:context.buffers[kWorldDispatch]
                 offset:0u
                atIndex:0u];
    [encoder setBytes:&pass length:sizeof(pass) atIndex:1u];
    [encoder setBuffer:context.buffers[sourceQ]
                 offset:0u
                atIndex:2u];
    [encoder setBuffer:context.buffers[sourceV]
                 offset:0u
                atIndex:3u];
    [encoder setBuffer:context.buffers[kCandidateAcceleration]
                 offset:0u
                atIndex:4u];
    [encoder setBuffer:context.buffers[kEnvironmentStatuses]
                 offset:0u
                atIndex:5u];
    [encoder setBuffer:context.buffers[kObservations]
                 offset:0u
                atIndex:6u];
    [encoder setBuffer:context.buffers[kAccelerationTrajectory]
                 offset:0u
                atIndex:7u];
    [encoder setBuffer:context.buffers[kPublicStatuses]
                 offset:0u
                atIndex:8u];
    dispatchWorldThreads(
        encoder,
        context.capturePipeline,
        environmentCount
    );
    [encoder endEncoding];
    return true;
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

bool finiteRange(
    const std::vector<float>& values,
    const std::size_t begin,
    const std::size_t count
) {
    return std::all_of(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() +
            static_cast<std::ptrdiff_t>(begin + count),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

bool zeroRange(
    const std::vector<float>& values,
    const std::size_t begin,
    const std::size_t count
) {
    return std::all_of(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() +
            static_cast<std::ptrdiff_t>(begin + count),
        [](const float value) {
            return value == 0.0f;
        }
    );
}

MetalWorldDiagnostics validateAndPublish(
    MetalWorldResult&& staged,
    MetalWorldDiagnostics diagnostics,
    MetalWorldResult& result
) {
    const MRMetalWorldDispatchGPU& dispatch =
        diagnostics.layout.dispatch;
    const std::size_t observationWidth =
        static_cast<std::size_t>(dispatch.nq) + dispatch.nv;
    for (std::size_t controlStep = 0u;
         controlStep < dispatch.controlStepCount;
         ++controlStep) {
        for (std::size_t environment = 0u;
             environment < dispatch.environmentCount;
             ++environment) {
            const std::size_t statusIndex =
                controlStep * dispatch.environmentCount +
                environment;
            const MRMetalWorldStatusGPU& status =
                staged.statuses[statusIndex];
            if (status.environment != environment ||
                status.controlStep != controlStep ||
                status.code > MR_STEP_UNSUPPORTED ||
                status.abaCode >
                    MR_ABA_UNSUPPORTED_TOPOLOGY ||
                status.flags != dispatch.flags ||
                status.successfulSubsteps >
                    dispatch.physicsSubsteps ||
                !finite(status.diagnostics)) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "GPU returned a malformed MetalWorld status record"
                );
            }
            const std::size_t observationBase =
                controlStep * dispatch.observationStepStride +
                environment *
                    dispatch.observationEnvironmentStride;
            const std::size_t accelerationBase =
                controlStep * dispatch.accelerationStepStride +
                environment * dispatch.nv;
            if (!finiteRange(
                    staged.observations,
                    observationBase,
                    observationWidth
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "GPU published a non-finite MetalWorld observation"
                );
            }

            if (status.code == MR_STEP_SUCCESS) {
                if (status.successfulSubsteps !=
                        dispatch.physicsSubsteps ||
                    status.abaCode != MR_ABA_SUCCESS ||
                    status.failingSubstep != MR_INVALID_INDEX ||
                    status.failingIndex != MR_INVALID_INDEX ||
                    !finiteRange(
                        staged.accelerations,
                        accelerationBase,
                        dispatch.nv
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::internalFailure,
                        "GPU success status has invalid substep "
                        "accounting or payload"
                    );
                }
                ++diagnostics.successfulStepCount;
            } else {
                if (status.successfulSubsteps >=
                        dispatch.physicsSubsteps ||
                    status.abaCode == MR_ABA_SUCCESS ||
                    status.failingSubstep >=
                        dispatch.physicsSubsteps ||
                    !zeroRange(
                        staged.accelerations,
                        accelerationBase,
                        dispatch.nv
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::internalFailure,
                        "failed GPU step violated rollback or "
                        "failure-accounting semantics"
                    );
                }
                if (diagnostics.failedStepCount == 0u) {
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(environment);
                    diagnostics.firstFailingControlStep =
                        static_cast<std::uint32_t>(controlStep);
                    diagnostics.firstGPUStatusCode = status.code;
                }
                ++diagnostics.failedStepCount;
            }
        }
    }

    if (!finiteFloats(staged.finalQ) ||
        !finiteFloats(staged.finalV)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "GPU published a non-finite final MetalWorld state"
        );
    }
    const std::size_t lastStep =
        static_cast<std::size_t>(dispatch.controlStepCount) - 1u;
    for (std::size_t environment = 0u;
         environment < dispatch.environmentCount;
         ++environment) {
        const std::size_t observationBase =
            lastStep * dispatch.observationStepStride +
            environment * dispatch.observationEnvironmentStride;
        const std::size_t qBase = environment * dispatch.qStride;
        const std::size_t vBase = environment * dispatch.vStride;
        if (std::memcmp(
                staged.finalQ.data() + qBase,
                staged.observations.data() + observationBase,
                dispatch.nq * sizeof(float)
            ) != 0 ||
            std::memcmp(
                staged.finalV.data() + vBase,
                staged.observations.data() +
                    observationBase + dispatch.nq,
                dispatch.nv * sizeof(float)
            ) != 0) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::internalFailure,
                "final state does not match the last captured observation"
            );
        }
    }

    result = std::move(staged);
    diagnostics.published = true;
    if (diagnostics.failedStepCount != 0u) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::gpuEnvironmentFailure,
            "one or more MetalWorld control steps rolled back "
            "after a GPU environment failure"
        );
    }
    diagnostics.status = MetalWorldHostStatus::success;
    diagnostics.message.clear();
    return diagnostics;
}

} // namespace

bool CompiledWorld::valid() const noexcept {
    return fingerprint_ != 0u &&
        articulationIndex_ < model_.articulations.size() &&
        capacityClass_ != MetalWorldCapacityClass::uncompiled;
}

const EngineModel& CompiledWorld::model() const noexcept {
    return model_;
}

std::uint32_t CompiledWorld::articulationIndex() const noexcept {
    return articulationIndex_;
}

std::uint32_t CompiledWorld::nq() const noexcept {
    return valid()
        ? model_.articulations[articulationIndex_].nq
        : 0u;
}

std::uint32_t CompiledWorld::nv() const noexcept {
    return valid()
        ? model_.articulations[articulationIndex_].nv
        : 0u;
}

std::uint32_t CompiledWorld::bodyCount() const noexcept {
    return valid()
        ? model_.articulations[articulationIndex_].bodyCount
        : 0u;
}

MetalWorldCapacityClass CompiledWorld::capacityClass()
    const noexcept {
    return capacityClass_;
}

std::uint64_t CompiledWorld::fingerprint() const noexcept {
    return fingerprint_;
}

MetalWorldCompileDiagnostics compileMetalWorld(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    CompiledWorld& compiled
) {
    MetalWorldCompileDiagnostics diagnostics{};
    try {
        std::string modelReason;
        if (!model.valid(&modelReason)) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidModel,
                "invalid EngineModel: " + modelReason
            );
        }
        if (model.articulations.size() != 1u ||
            articulationIndex != 0u) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::unsupportedTopology,
                "MetalWorld ABI v1 requires exactly one selected "
                "articulation"
            );
        }
        std::string topologyReason;
        if (!supportedTopology(
                model,
                model.articulations[articulationIndex],
                topologyReason
            )) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            const bool capacity =
                articulation.bodyCount >
                    MR_ARTICULATED_ABA_MAX_BODIES ||
                articulation.nv >
                    MR_ARTICULATED_ABA_MAX_DOFS ||
                articulation.nq >
                    MR_ARTICULATED_ABA_MAX_Q;
            return rejectCompile(
                std::move(diagnostics),
                capacity
                    ? MetalWorldHostStatus::capacityOverflow
                    : MetalWorldHostStatus::unsupportedTopology,
                std::move(topologyReason)
            );
        }

        CompiledWorld staged;
        staged.model_ = model;
        staged.articulationIndex_ = articulationIndex;
        const MRArticulationGPU& articulation =
            staged.model_.articulations[articulationIndex];
        staged.capacityClass_ =
            articulation.bodyCount <= kSmallABAMaxBodies &&
                articulation.nv <= kSmallABAMaxDofs &&
                articulation.nq <= kSmallABAMaxQ
            ? MetalWorldCapacityClass::compactABA12
            : MetalWorldCapacityClass::fullABA32;
        staged.fingerprint_ = fingerprint(staged.model_);
        diagnostics.fingerprint = staged.fingerprint_;
        compiled = std::move(staged);
        diagnostics.status = MetalWorldHostStatus::success;
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return rejectCompile(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while compiling MetalWorld"
        );
    } catch (const std::exception& exception) {
        return rejectCompile(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldSubmission::MetalWorldSubmission() noexcept = default;

MetalWorldSubmission::~MetalWorldSubmission() = default;

MetalWorldSubmission::MetalWorldSubmission(
    MetalWorldSubmission&& other
) noexcept = default;

MetalWorldSubmission& MetalWorldSubmission::operator=(
    MetalWorldSubmission&& other
) noexcept = default;

bool MetalWorldSubmission::valid() const noexcept {
    return state_ != nullptr;
}

MetalWorldDiagnostics MetalWorldSubmission::wait(
    MetalWorldResult& result
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalWorldHostStatus::metalCommandFailure,
            "MetalWorld submission is empty or already consumed"
        );
    }

    std::unique_ptr<detail::MetalWorldSubmissionState> pending =
        std::move(state_);
    MetalWorldDiagnostics diagnostics = pending->diagnostics;
    try {
        MetalWorldResult staged{};
        @autoreleasepool {
            [pending->commandBuffer waitUntilCompleted];
            const auto end = std::chrono::steady_clock::now();
            diagnostics.submissionElapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - pending->start
                ).count();
            diagnostics.thermalState = thermalStateName(
                [NSProcessInfo processInfo].thermalState
            );
            if (pending->commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::metalCommandFailure,
                    "MetalWorld command failed: " +
                        describeError(
                            pending->commandBuffer.error
                        )
                );
            }
            const CFTimeInterval gpuStart =
                pending->commandBuffer.GPUStartTime;
            const CFTimeInterval gpuEnd =
                pending->commandBuffer.GPUEndTime;
            if (std::isfinite(gpuStart) &&
                std::isfinite(gpuEnd) &&
                gpuEnd >= gpuStart) {
                diagnostics.gpuElapsedMilliseconds =
                    1000.0 * (gpuEnd - gpuStart);
            }

            staged.layout = diagnostics.layout;
            staged.finalQ.resize(staged.layout.initialQElements);
            staged.finalV.resize(staged.layout.initialVElements);
            staged.observations.resize(
                staged.layout.observationElements
            );
            staged.accelerations.resize(
                staged.layout.accelerationElements
            );
            staged.statuses.resize(
                staged.layout.statusElements
            );
            const auto& buffers = pending->context->buffers;
            copyOutput(
                staged.finalQ,
                buffers[pending->finalQBuffer]
            );
            copyOutput(
                staged.finalV,
                buffers[pending->finalVBuffer]
            );
            copyOutput(
                staged.observations,
                buffers[kObservations]
            );
            copyOutput(
                staged.accelerations,
                buffers[kAccelerationTrajectory]
            );
            copyOutput(
                staged.statuses,
                buffers[kPublicStatuses]
            );
        }

        return validateAndPublish(
            std::move(staged),
            std::move(diagnostics),
            result
        );
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while publishing MetalWorld results"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldContext::MetalWorldContext(MetalWorldConfig config)
    : state_(
          std::make_shared<detail::MetalWorldContextState>(
              std::move(config)
          )
      ) {}

MetalWorldContext::~MetalWorldContext() = default;

MetalWorldContext::MetalWorldContext(
    MetalWorldContext&& other
) noexcept = default;

MetalWorldContext& MetalWorldContext::operator=(
    MetalWorldContext&& other
) noexcept = default;

MetalWorldDiagnostics MetalWorldContext::submit(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldSubmission& submission
) {
    MetalWorldDiagnostics diagnostics{};
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "MetalWorld context was moved from"
        );
    }
    if (submission.valid()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::contextBusy,
            "submission output already owns a live MetalWorld batch"
        );
    }

    RequiredBuffers requirements{};
    try {
        diagnostics = validateAndBuildLayout(
            world,
            batch,
            config,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }
        const std::lock_guard lock(state_->mutex);
        if (state_->inFlight) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::contextBusy,
                "MetalWorld context already has an in-flight batch"
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
                world,
                batch,
                config,
                diagnostics.layout,
                requirements
            );

            id<MTLCommandBuffer> commandBuffer =
                [state_->queue commandBuffer];
            if (commandBuffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::metalCommandFailure,
                    "failed to create MetalWorld command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo persistent batched world graph";

            id<MTLComputePipelineState> selectedABAPipeline =
                world.capacityClass() ==
                    MetalWorldCapacityClass::compactABA12
                ? state_->smallABAPipeline
                : state_->abaPipeline;
            std::size_t sourceQ = kStateQA;
            std::size_t sourceV = kStateVA;
            std::size_t destinationQ = kStateQB;
            std::size_t destinationV = kStateVB;
            for (std::uint32_t controlStep = 0u;
                 controlStep <
                     diagnostics.layout.dispatch.controlStepCount;
                 ++controlStep) {
                MRMetalWorldPassGPU pass{};
                pass.controlStep = controlStep;
                pass.physicsSubstep = MR_INVALID_INDEX;
                if (!encodePrepare(
                        *state_,
                        commandBuffer,
                        pass,
                        sourceQ,
                        sourceV,
                        batch.environmentCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode MetalWorld prepare pass"
                    );
                }

                for (std::uint32_t physicsSubstep = 0u;
                     physicsSubstep < config.physicsSubsteps;
                     ++physicsSubstep) {
                    pass.physicsSubstep = physicsSubstep;
                    if (!encodeABA(
                            *state_,
                            commandBuffer,
                            selectedABAPipeline,
                            sourceQ,
                            sourceV,
                            batch.environmentCount
                        ) ||
                        !encodeCommit(
                            *state_,
                            commandBuffer,
                            pass,
                            destinationQ,
                            destinationV,
                            batch.environmentCount
                        )) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode MetalWorld ABA/commit pass"
                        );
                    }
                    std::swap(sourceQ, destinationQ);
                    std::swap(sourceV, destinationV);
                }

                pass.physicsSubstep = MR_INVALID_INDEX;
                if (!encodeCapture(
                        *state_,
                        commandBuffer,
                        pass,
                        sourceQ,
                        sourceV,
                        batch.environmentCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode MetalWorld capture pass"
                    );
                }
            }

            auto pending =
                std::make_unique<
                    detail::MetalWorldSubmissionState
                >();
            diagnostics.dispatched = true;
            pending->context = state_;
            pending->commandBuffer = commandBuffer;
            pending->diagnostics = diagnostics;
            pending->articulation =
                world.model().articulations[
                    world.articulationIndex()
                ];
            pending->finalQBuffer = sourceQ;
            pending->finalVBuffer = sourceV;
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
            MetalWorldHostStatus::metalBufferFailure,
            "host allocation failed while preparing MetalWorld submission"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            exception.what()
        );
    }
}

MetalWorldDiagnostics MetalWorldContext::run(
    const CompiledWorld& world,
    const MetalWorldBatch& batch,
    const MetalWorldStepConfig& config,
    MetalWorldResult& result
) {
    MetalWorldSubmission submission;
    MetalWorldDiagnostics diagnostics = submit(
        world,
        batch,
        config,
        submission
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return submission.wait(result);
}

MetalWorldContextStats MetalWorldContext::stats()
    const noexcept {
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

const char* metalWorldHostStatusName(
    const MetalWorldHostStatus status
) noexcept {
    switch (status) {
    case MetalWorldHostStatus::success:
        return "success";
    case MetalWorldHostStatus::invalidModel:
        return "invalid_model";
    case MetalWorldHostStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalWorldHostStatus::unsupportedSolverMode:
        return "unsupported_solver_mode";
    case MetalWorldHostStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalWorldHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalWorldHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalWorldHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalWorldHostStatus::invalidReset:
        return "invalid_reset";
    case MetalWorldHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalWorldHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalWorldHostStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalWorldHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalWorldHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalWorldHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalWorldHostStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalWorldHostStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalWorldHostStatus::internalFailure:
        return "internal_failure";
    case MetalWorldHostStatus::contextBusy:
        return "context_busy";
    }
    return "unknown";
}

} // namespace metalrobo
