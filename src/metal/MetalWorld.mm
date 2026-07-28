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
#include <initializer_list>
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

constexpr std::size_t kRawBufferCount = 81u;
constexpr NSUInteger kABAThreadsPerThreadgroup = 32u;
constexpr NSUInteger kOperatorThreadsPerThreadgroup = 32u;
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
    kShapes = 27u,
    kMaterials = 28u,
    kSceneBodyIndices = 29u,
    kEligiblePairs = 30u,
    kContactDispatch = 31u,
    kOperatorKinematicsDispatch = 32u,
    kOperatorFactorDispatch = 33u,
    kInitialSceneBodies = 34u,
    kResetSceneBodies = 35u,
    kKinematicTargets = 36u,
    kSceneBodiesA = 37u,
    kSceneBodiesB = 38u,
    kCheckpointSceneBodies = 39u,
    kBodyPoses = 40u,
    kPointWorld = 41u,
    kFactorMatrix = 42u,
    kPointJacobians = 43u,
    kGeneralizedImpulse = 44u,
    kDeltaVelocity = 45u,
    kOperatorStatuses = 46u,
    kCurrentBodies = 47u,
    kCandidateBodies = 48u,
    kManifoldHeadersA = 49u,
    kManifoldPointsA = 50u,
    kManifoldCountsA = 51u,
    kManifoldHeadersB = 52u,
    kManifoldPointsB = 53u,
    kManifoldCountsB = 54u,
    kCandidateManifoldHeaders = 55u,
    kCandidateManifoldPoints = 56u,
    kCandidateManifoldCounts = 57u,
    kCheckpointManifoldHeaders = 58u,
    kCheckpointManifoldPoints = 59u,
    kCheckpointManifoldCounts = 60u,
    kCandidatePairs = 61u,
    kRawContacts = 62u,
    kRawPairIndices = 63u,
    kContacts = 64u,
    kContactMetadata = 65u,
    kIRBlocks = 66u,
    kIREndpoints = 67u,
    kIRRows = 68u,
    kIRCones = 69u,
    kPointQueries = 70u,
    kEvaluatedRows = 71u,
    kEvaluatedCones = 72u,
    kFactorCaches = 73u,
    kIslands = 74u,
    kResponseColumns = 75u,
    kContactStatuses = 76u,
    kPublicContactStatuses = 77u,
    kActiveIndirectDispatch = 78u,
    kProjectedColliders = 79u,
    kPairOverlapFlags = 80u,
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
    __strong id<MTLComputePipelineState> operatorPipeline = nil;
    __strong id<MTLComputePipelineState> contactPreparePipeline = nil;
    __strong id<MTLComputePipelineState> bodyProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> scenePredictionPipeline = nil;
    __strong id<MTLComputePipelineState> colliderProjectionPipeline = nil;
    __strong id<MTLComputePipelineState> pairFlagPipeline = nil;
    __strong id<MTLComputePipelineState> collisionCompilePipeline = nil;
    __strong id<MTLComputePipelineState> factorDispatchPipeline = nil;
    __strong id<MTLComputePipelineState> pointQueryTailPipeline = nil;
    __strong id<MTLComputePipelineState> evaluateIRPipeline = nil;
    __strong id<MTLComputePipelineState> islandPipeline = nil;
    __strong id<MTLComputePipelineState> contactSolvePipeline = nil;
    __strong id<MTLComputePipelineState> contactIntegratePipeline = nil;
    __strong id<MTLComputePipelineState> contactLatchPipeline = nil;
    __strong id<MTLComputePipelineState> contactCommitPipeline = nil;
    __strong id<MTLComputePipelineState> contactCapturePipeline = nil;
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
    std::size_t finalSceneBuffer = kSceneBodiesA;
    std::size_t finalManifoldHeaderBuffer = kManifoldHeadersA;
    std::size_t finalManifoldPointBuffer = kManifoldPointsA;
    std::size_t finalManifoldCountBuffer = kManifoldCountsA;
    bool contactMode = false;
    bool captureContactEvidence = false;
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

bool validSceneStates(
    const CompiledWorld& world,
    const std::size_t environmentCount,
    const std::span<const MRBodyStateGPU> states
) {
    if (states.empty() && world.sceneBodyCount() == 0u) {
        return true;
    }
    if (states.size() !=
        environmentCount * world.sceneBodyCount()) {
        return false;
    }
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        std::size_t localScene = 0u;
        for (std::uint32_t globalBody = 0u;
             globalBody < world.model().bodies.size();
             ++globalBody) {
            if (world.model().bodies[globalBody].articulationIndex !=
                MR_INVALID_INDEX) {
                continue;
            }
            const MRBodyPropertiesGPU& properties =
                world.model().bodies[globalBody];
            const MRBodyStateGPU& state =
                states[
                    environment * world.sceneBodyCount() +
                    localScene
                ];
            const double quaternionNormSquared =
                static_cast<double>(state.orientation.x) *
                    state.orientation.x +
                static_cast<double>(state.orientation.y) *
                    state.orientation.y +
                static_cast<double>(state.orientation.z) *
                    state.orientation.z +
                static_cast<double>(state.orientation.w) *
                    state.orientation.w;
            if (!finite(state.position) ||
                !finite(state.orientation) ||
                !finite(state.linearVelocityAndInverseMass) ||
                !finite(state.angularVelocity) ||
                !finite(state.inverseInertiaWorldRow0) ||
                !finite(state.inverseInertiaWorldRow1) ||
                !finite(state.inverseInertiaWorldRow2) ||
                !(quaternionNormSquared > 1.0e-12) ||
                std::abs(std::sqrt(quaternionNormSquared) - 1.0) >
                    kQuaternionHostTolerance ||
                state.flagsAndIndices[0] != properties.motionType ||
                state.flagsAndIndices[1] != MR_INVALID_INDEX ||
                (state.flagsAndIndices[2] != globalBody &&
                 state.flagsAndIndices[2] != MR_INVALID_INDEX)) {
                return false;
            }
            ++localScene;
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

std::uint32_t compiledPairClass(
    const std::uint32_t typeA,
    const std::uint32_t typeB
) {
    if (typeA == MR_SHAPE_SPHERE &&
        typeB == MR_SHAPE_SPHERE) {
        return MR_COLLISION_PAIR_SPHERE_SPHERE;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_SPHERE &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_SPHERE_PLANE;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_CAPSULE &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_CAPSULE_PLANE;
    }
    if ((typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_BOX &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_BOX_PLANE;
    }
    if ((typeA == MR_SHAPE_CYLINDER &&
         typeB == MR_SHAPE_PLANE) ||
        (typeB == MR_SHAPE_CYLINDER &&
         typeA == MR_SHAPE_PLANE)) {
        return MR_COLLISION_PAIR_CYLINDER_PLANE;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_CAPSULE) ||
        (typeB == MR_SHAPE_SPHERE &&
         typeA == MR_SHAPE_CAPSULE)) {
        return MR_COLLISION_PAIR_SPHERE_CAPSULE;
    }
    if (typeA == MR_SHAPE_CAPSULE &&
        typeB == MR_SHAPE_CAPSULE) {
        return MR_COLLISION_PAIR_CAPSULE_CAPSULE;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_BOX) ||
        (typeB == MR_SHAPE_SPHERE &&
         typeA == MR_SHAPE_BOX)) {
        return MR_COLLISION_PAIR_SPHERE_BOX;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_BOX) ||
        (typeB == MR_SHAPE_CAPSULE &&
         typeA == MR_SHAPE_BOX)) {
        return MR_COLLISION_PAIR_CAPSULE_BOX;
    }
    if (typeA == MR_SHAPE_BOX &&
        typeB == MR_SHAPE_BOX) {
        return MR_COLLISION_PAIR_BOX_BOX;
    }
    return MR_COLLISION_PAIR_UNSUPPORTED;
}

std::uint64_t compiledFingerprint(
    const EngineModel& model,
    const MetalWorldCapacityProfile& capacities,
    const std::vector<std::uint32_t>& sceneBodyIndices,
    const std::vector<MRCompiledCollisionPairGPU>& eligiblePairs
) {
    std::uint64_t hash = fingerprint(model);
    hashValue(hash, capacities);
    hashVector(hash, sceneBodyIndices);
    hashVector(hash, eligiblePairs);
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
    const std::size_t environments =
        layout.dispatch.environmentCount;
    const std::size_t contactEnvironments =
        (layout.dispatch.flags & MR_METAL_WORLD_CONTACTS) != 0u
        ? environments
        : 0u;
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

    const MRMetalWorldContactDispatchGPU& contact =
        layout.contactDispatch;
    std::size_t bodyPoseElements = 0u;
    std::size_t bodyStateElements = 0u;
    std::size_t projectedColliderElements = 0u;
    std::size_t eligiblePairFlagElements = 0u;
    std::size_t pairElements = 0u;
    std::size_t rawContactElements = 0u;
    std::size_t pointQueryElements = 0u;
    std::size_t factorElements = 0u;
    std::size_t pointJacobianElements = 0u;
    std::size_t endpointElements = 0u;
    std::size_t coneElements = 0u;
    std::size_t responseElements = 0u;
    if (!checkedMultiply(
            contactEnvironments,
            world.bodyCount(),
            bodyPoseElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            model.bodies.size(),
            bodyStateElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            model.shapes.size(),
            projectedColliderElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            world.eligiblePairCount(),
            eligiblePairFlagElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.pairStride,
            pairElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.rawContactStride,
            rawContactElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.pointQueryStride,
            pointQueryElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.factorStride,
            factorElements
        ) ||
        !checkedMultiply(
            pointQueryElements,
            3u * static_cast<std::size_t>(contact.nv),
            pointJacobianElements
        ) ||
        !checkedMultiply(
            layout.contactConstraintElements,
            2u,
            endpointElements
        ) ||
        !checkedMultiply(
            contactEnvironments,
            contact.constraintStride,
            coneElements
        ) ||
        !checkedMultiply(
            layout.contactConstraintElements,
            3u * static_cast<std::size_t>(contact.nv),
            responseElements
        )) {
        return false;
    }
    const std::size_t immutableShapeElements =
        std::max<std::size_t>(model.shapes.size(), 1u);
    const std::size_t immutableMaterialElements =
        std::max<std::size_t>(model.materials.size(), 1u);
    const std::size_t immutableSceneIndexElements =
        std::max<std::size_t>(world.sceneBodyCount(), 1u);
    const std::size_t immutablePairElements =
        std::max<std::size_t>(world.eligiblePairCount(), 1u);
    if (!makeRequirement<MRShapeGPU>(
            "shapes",
            immutableShapeElements,
            requirements.entries[kShapes]
        ) ||
        !makeRequirement<MRMaterialGPU>(
            "materials",
            immutableMaterialElements,
            requirements.entries[kMaterials]
        ) ||
        !makeRequirement<mr_u32>(
            "scene body indices",
            immutableSceneIndexElements,
            requirements.entries[kSceneBodyIndices]
        ) ||
        !makeRequirement<MRCompiledCollisionPairGPU>(
            "eligible collision pairs",
            immutablePairElements,
            requirements.entries[kEligiblePairs]
        ) ||
        !makeRequirement<MRMetalWorldContactDispatchGPU>(
            "contact dispatch",
            1u,
            requirements.entries[kContactDispatch]
        ) ||
        !makeRequirement<MRArticulatedOperatorDispatchGPU>(
            "kinematics operator dispatch",
            1u,
            requirements.entries[kOperatorKinematicsDispatch]
        ) ||
        !makeRequirement<MRArticulatedOperatorDispatchGPU>(
            "factor operator dispatch",
            1u,
            requirements.entries[kOperatorFactorDispatch]
        ) ||
        !makeRequirement<MRIndirectDispatchArgumentsGPU>(
            "active contact indirect dispatch arguments",
            contactEnvironments == 0u ? 0u : 2u,
            requirements.entries[kActiveIndirectDispatch]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "initial scene bodies",
            layout.initialSceneBodyElements,
            requirements.entries[kInitialSceneBodies]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "reset scene bodies",
            layout.resetSceneBodyElements,
            requirements.entries[kResetSceneBodies]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "kinematic targets",
            layout.kinematicTargetElements,
            requirements.entries[kKinematicTargets]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "scene bodies A",
            layout.initialSceneBodyElements,
            requirements.entries[kSceneBodiesA]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "scene bodies B",
            layout.initialSceneBodyElements,
            requirements.entries[kSceneBodiesB]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "checkpoint scene bodies",
            layout.initialSceneBodyElements,
            requirements.entries[kCheckpointSceneBodies]
        ) ||
        !makeRequirement<MRArticulatedBodyPoseGPU>(
            "articulation body poses",
            bodyPoseElements,
            requirements.entries[kBodyPoses]
        ) ||
        !makeRequirement<MRArticulatedPointWorldGPU>(
            "articulated point world",
            pointQueryElements,
            requirements.entries[kPointWorld]
        ) ||
        !makeRequirement<float>(
            "articulation factor matrix",
            factorElements,
            requirements.entries[kFactorMatrix]
        ) ||
        !makeRequirement<float>(
            "point Jacobians",
            pointJacobianElements,
            requirements.entries[kPointJacobians]
        ) ||
        !makeRequirement<float>(
            "generalized impulse",
            contactEnvironments == 0u
                ? 0u
                : layout.initialVElements,
            requirements.entries[kGeneralizedImpulse]
        ) ||
        !makeRequirement<float>(
            "operator delta velocity",
            contactEnvironments == 0u
                ? 0u
                : layout.initialVElements,
            requirements.entries[kDeltaVelocity]
        ) ||
        !makeRequirement<MRArticulatedOperatorStatusGPU>(
            "articulated operator statuses",
            contactEnvironments,
            requirements.entries[kOperatorStatuses]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "current global body states",
            bodyStateElements,
            requirements.entries[kCurrentBodies]
        ) ||
        !makeRequirement<MRBodyStateGPU>(
            "candidate global body states",
            bodyStateElements,
            requirements.entries[kCandidateBodies]
        ) ||
        !makeRequirement<MRProjectedColliderGPU>(
            "projected colliders and AABBs",
            projectedColliderElements,
            requirements.entries[kProjectedColliders]
        ) ||
        !makeRequirement<mr_u32>(
            "eligible-pair overlap flags",
            eligiblePairFlagElements,
            requirements.entries[kPairOverlapFlags]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "manifold headers A",
            layout.manifoldHeaderElements,
            requirements.entries[kManifoldHeadersA]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "manifold points A",
            layout.manifoldPointElements,
            requirements.entries[kManifoldPointsA]
        ) ||
        !makeRequirement<mr_u32>(
            "manifold counts A",
            contactEnvironments,
            requirements.entries[kManifoldCountsA]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "manifold headers B",
            layout.manifoldHeaderElements,
            requirements.entries[kManifoldHeadersB]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "manifold points B",
            layout.manifoldPointElements,
            requirements.entries[kManifoldPointsB]
        ) ||
        !makeRequirement<mr_u32>(
            "manifold counts B",
            contactEnvironments,
            requirements.entries[kManifoldCountsB]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "candidate manifold headers",
            layout.manifoldHeaderElements,
            requirements.entries[kCandidateManifoldHeaders]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "candidate manifold points",
            layout.manifoldPointElements,
            requirements.entries[kCandidateManifoldPoints]
        ) ||
        !makeRequirement<mr_u32>(
            "candidate manifold counts",
            contactEnvironments,
            requirements.entries[kCandidateManifoldCounts]
        ) ||
        !makeRequirement<MRManifoldHeaderGPU>(
            "checkpoint manifold headers",
            layout.manifoldHeaderElements,
            requirements.entries[kCheckpointManifoldHeaders]
        ) ||
        !makeRequirement<MRManifoldPointGPU>(
            "checkpoint manifold points",
            layout.manifoldPointElements,
            requirements.entries[kCheckpointManifoldPoints]
        ) ||
        !makeRequirement<mr_u32>(
            "checkpoint manifold counts",
            contactEnvironments,
            requirements.entries[kCheckpointManifoldCounts]
        ) ||
        !makeRequirement<MRCandidatePairGPU>(
            "candidate pairs",
            pairElements,
            requirements.entries[kCandidatePairs]
        ) ||
        !makeRequirement<MRRawContactGPU>(
            "raw contacts",
            rawContactElements,
            requirements.entries[kRawContacts]
        ) ||
        !makeRequirement<mr_u32>(
            "raw contact pair indices",
            rawContactElements,
            requirements.entries[kRawPairIndices]
        ) ||
        !makeRequirement<MRContactConstraintGPU>(
            "contact constraints",
            layout.contactConstraintElements,
            requirements.entries[kContacts]
        ) ||
        !makeRequirement<MRContactPointMetaGPU>(
            "contact metadata",
            layout.contactConstraintElements,
            requirements.entries[kContactMetadata]
        ) ||
        !makeRequirement<MRConstraintIRBlockGPU>(
            "ConstraintIR blocks",
            layout.contactConstraintElements,
            requirements.entries[kIRBlocks]
        ) ||
        !makeRequirement<MRConstraintIREndpointGPU>(
            "ConstraintIR endpoints",
            endpointElements,
            requirements.entries[kIREndpoints]
        ) ||
        !makeRequirement<MRConstraintIRRowGPU>(
            "ConstraintIR rows",
            layout.constraintRowElements,
            requirements.entries[kIRRows]
        ) ||
        !makeRequirement<MRConstraintIRConeGPU>(
            "ConstraintIR cones",
            coneElements,
            requirements.entries[kIRCones]
        ) ||
        !makeRequirement<MRArticulatedPointImpulseGPU>(
            "articulated point queries",
            pointQueryElements,
            requirements.entries[kPointQueries]
        ) ||
        !makeRequirement<MREvaluatedConstraintIRRowGPU>(
            "evaluated ConstraintIR rows",
            layout.constraintRowElements,
            requirements.entries[kEvaluatedRows]
        ) ||
        !makeRequirement<MREvaluatedConstraintIRConeGPU>(
            "evaluated ConstraintIR cones",
            coneElements,
            requirements.entries[kEvaluatedCones]
        ) ||
        !makeRequirement<MRArticulationFactorCacheGPU>(
            "articulation factor cache",
            contactEnvironments,
            requirements.entries[kFactorCaches]
        ) ||
        !makeRequirement<MRContactIslandGPU>(
            "contact islands",
            layout.islandElements,
            requirements.entries[kIslands]
        ) ||
        !makeRequirement<float>(
            "contact response columns",
            responseElements,
            requirements.entries[kResponseColumns]
        ) ||
        !makeRequirement<MRMetalWorldContactStatusGPU>(
            "contact statuses",
            contactEnvironments,
            requirements.entries[kContactStatuses]
        ) ||
        !makeRequirement<MRMetalWorldContactStatusGPU>(
            "public contact statuses",
            layout.contactStatusElements,
            requirements.entries[kPublicContactStatuses]
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
            MetalWorldSolverMode::freeMotionABA &&
        config.solverMode !=
            MetalWorldSolverMode::throughputPGS &&
        config.solverMode !=
            MetalWorldSolverMode::throughputTGS) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::unsupportedSolverMode,
            "unknown MetalWorld solver mode"
        );
    }
    const bool contactMode =
        config.solverMode != MetalWorldSolverMode::freeMotionABA;
    if (!std::isfinite(config.timestepSeconds) ||
        !(config.timestepSeconds > 0.0f) ||
        config.physicsSubsteps == 0u ||
        config.physicsSubsteps >
            MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS ||
        (contactMode &&
         (config.velocityIterations == 0u ||
          config.velocityIterations > 128u ||
          config.finalVelocityIterations > 128u)) ||
        !std::isfinite(config.manifoldBreakingSeparation) ||
        !std::isfinite(config.manifoldBreakingTangential) ||
        !std::isfinite(config.manifoldMergeDistance) ||
        !std::isfinite(config.manifoldNormalCosine) ||
        config.manifoldBreakingSeparation < 0.0f ||
        config.manifoldBreakingTangential < 0.0f ||
        config.manifoldMergeDistance < 0.0f ||
        config.manifoldNormalCosine < -1.0f ||
        config.manifoldNormalCosine > 1.0f) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "timestepSeconds must be finite and positive and "
            "physicsSubsteps, solver iterations, or manifold "
            "thresholds are outside the supported range"
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
    dispatch.flags = (
        contactMode
            ? static_cast<mr_u32>(MR_METAL_WORLD_CONTACTS)
            : static_cast<mr_u32>(
                  MR_METAL_WORLD_FREE_MOTION_ONLY
              )
        ) |
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
    const std::size_t observationEnvironmentStride =
        static_cast<std::size_t>(articulation.nq) +
        articulation.nv +
        (contactMode
             ? 13u * static_cast<std::size_t>(
                   world.sceneBodyCount()
               )
             : 0u);
    if (observationEnvironmentStride >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::arithmeticOverflow,
            "observation environment stride exceeds the GPU ABI"
        );
    }
    dispatch.observationEnvironmentStride =
        static_cast<mr_u32>(observationEnvironmentStride);

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

    MRMetalWorldContactDispatchGPU& contact =
        layout.contactDispatch;
    contact.abiVersion = MR_METAL_WORLD_CONTACT_ABI_VERSION;
    contact.environmentCount = dispatch.environmentCount;
    contact.articulationIndex = world.articulationIndex();
    contact.solverType =
        config.solverMode == MetalWorldSolverMode::throughputPGS
        ? MR_SOLVER_THROUGHPUT_PGS
        : MR_SOLVER_THROUGHPUT_TGS;
    contact.bodyCount =
        static_cast<mr_u32>(world.model().bodies.size());
    contact.sceneBodyCount = world.sceneBodyCount();
    contact.shapeCount = world.colliderCount();
    contact.eligiblePairCount = world.eligiblePairCount();
    contact.pairCapacity = world.capacities().candidatePairs;
    contact.rawContactCapacity = world.capacities().rawContacts;
    contact.manifoldCapacity = world.capacities().manifolds;
    contact.constraintCapacity =
        world.capacities().constraintBlocks;
    contact.rowCapacity = world.capacities().constraintRows;
    contact.islandCapacity = world.capacities().islands;
    contact.sceneBodyStride = world.sceneBodyCount();
    contact.bodyStateStride = contact.bodyCount;
    contact.pairStride = contact.pairCapacity;
    contact.rawContactStride = contact.rawContactCapacity;
    contact.manifoldStride = contact.manifoldCapacity;
    contact.constraintStride = contact.constraintCapacity;
    contact.rowStride = contact.rowCapacity;
    contact.islandStride = contact.islandCapacity;
    contact.pointQueryStride =
        2u * contact.constraintCapacity;
    contact.factorStride = articulation.nv * articulation.nv;
    contact.nv = articulation.nv;
    contact.flags =
        (config.deterministic
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_DETERMINISTIC
               )
             : 0u) |
        (config.warmStart
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_WARM_START
               )
             : 0u) |
        (config.captureContactEvidence
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_CAPTURE_EVIDENCE
               )
             : 0u) |
        (!batch.kinematicTargets.empty()
             ? static_cast<mr_u32>(
                   MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS
               )
             : 0u);
    contact.velocityIterations = config.velocityIterations;
    contact.finalVelocityIterations =
        config.finalVelocityIterations;
    contact.timestepAndBias = {
        substepTimestep,
        world.model().world.solverScales.w,
        world.model().world.solverScales.z,
        config.warmStart ? 1.0f : 0.0f,
    };
    contact.manifoldThresholds = {
        config.manifoldBreakingSeparation,
        config.manifoldBreakingTangential,
        config.manifoldMergeDistance,
        config.manifoldNormalCosine,
    };

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
    if (contactMode) {
        if (!checkedMultiply(
                batch.environmentCount,
                world.sceneBodyCount(),
                layout.initialSceneBodyElements
            ) ||
            !checkedMultiply(
                batch.controlStepCount,
                batch.environmentCount,
                layout.contactStatusElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.manifoldStride,
                layout.manifoldHeaderElements
            ) ||
            !checkedMultiply(
                layout.manifoldHeaderElements,
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY,
                layout.manifoldPointElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.constraintStride,
                layout.contactConstraintElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.rowStride,
                layout.constraintRowElements
            ) ||
            !checkedMultiply(
                batch.environmentCount,
                contact.islandStride,
                layout.islandElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "derived contact-world element-count overflow"
            );
        }
        layout.resetSceneBodyElements =
            batch.resetMasks.empty()
            ? 0u
            : layout.initialSceneBodyElements;
        if (!batch.kinematicTargets.empty() &&
            !checkedMultiply(
                batch.controlStepCount,
                layout.initialSceneBodyElements,
                layout.kinematicTargetElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "kinematic-target element-count overflow"
            );
        }
    }

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
        layout.initialSceneBodyElements,
        layout.resetSceneBodyElements,
        layout.kinematicTargetElements,
        layout.contactStatusElements,
        layout.manifoldHeaderElements,
        layout.manifoldPointElements,
        layout.contactConstraintElements,
        layout.constraintRowElements,
        layout.islandElements,
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
        batch.efforts.size() != layout.effortElements ||
        batch.initialSceneBodies.size() !=
            layout.initialSceneBodyElements ||
        batch.kinematicTargets.size() !=
            layout.kinematicTargetElements) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "initial state, effort, or kinematic trajectory has the wrong "
            "packed element count"
        );
    }
    if (!contactMode &&
        (!batch.initialSceneBodies.empty() ||
         !batch.resetSceneBodies.empty() ||
         !batch.kinematicTargets.empty())) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::invalidDimensions,
            "scene-body inputs require a contact solver mode"
        );
    }
    if (batch.resetMasks.empty()) {
        if (!batch.resetQ.empty() ||
            !batch.resetV.empty() ||
            !batch.resetSceneBodies.empty()) {
            return reject(
                std::move(diagnostics),
                MetalWorldHostStatus::invalidReset,
                "reset states require a control-step reset mask"
            );
        }
    } else if (
        batch.resetMasks.size() != layout.resetMaskElements ||
        batch.resetQ.size() != layout.resetQElements ||
        batch.resetV.size() != layout.resetVElements ||
        batch.resetSceneBodies.size() !=
            layout.resetSceneBodyElements) {
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
    if (contactMode &&
        (!validSceneStates(
             world,
             batch.environmentCount,
             batch.initialSceneBodies
         ) ||
         (!batch.resetMasks.empty() &&
          !validSceneStates(
              world,
              batch.environmentCount,
              batch.resetSceneBodies
          )))) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::nonfiniteInput,
            "scene-body state is non-finite, has an invalid "
            "quaternion, or does not match compiled body identity"
        );
    }
    if (contactMode && !batch.kinematicTargets.empty()) {
        const std::size_t stateCount =
            layout.initialSceneBodyElements;
        for (std::size_t controlStep = 0u;
             controlStep < batch.controlStepCount;
             ++controlStep) {
            if (!validSceneStates(
                    world,
                    batch.environmentCount,
                    batch.kinematicTargets.subspan(
                        controlStep * stateCount,
                        stateCount
                    )
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::nonfiniteInput,
                    "kinematic target state is invalid"
                );
            }
        }
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

    __strong id<MTLComputePipelineState> operatorPipeline = nil;
    __strong id<MTLComputePipelineState> contactPrepare = nil;
    __strong id<MTLComputePipelineState> bodyProjection = nil;
    __strong id<MTLComputePipelineState> scenePrediction = nil;
    __strong id<MTLComputePipelineState> colliderProjection = nil;
    __strong id<MTLComputePipelineState> pairFlags = nil;
    __strong id<MTLComputePipelineState> collisionCompile = nil;
    __strong id<MTLComputePipelineState> factorDispatch = nil;
    __strong id<MTLComputePipelineState> pointQueryTail = nil;
    __strong id<MTLComputePipelineState> evaluateIR = nil;
    __strong id<MTLComputePipelineState> islands = nil;
    __strong id<MTLComputePipelineState> contactSolve = nil;
    __strong id<MTLComputePipelineState> contactIntegrate = nil;
    __strong id<MTLComputePipelineState> contactLatch = nil;
    __strong id<MTLComputePipelineState> contactCommit = nil;
    __strong id<MTLComputePipelineState> contactCapture = nil;
    auto createContactPipeline = [&](
        NSString* functionName
    ) {
        error = nil;
        return makePipeline(
            device,
            library,
            functionName,
            &error
        );
    };
    operatorPipeline =
        createContactPipeline(@"mr_articulated_operator");
    contactPrepare =
        createContactPipeline(@"mr_world_prepare_contact_step");
    bodyProjection =
        createContactPipeline(@"mr_world_build_body_states");
    scenePrediction =
        createContactPipeline(@"mr_world_predict_scene");
    colliderProjection =
        createContactPipeline(@"mr_world_project_colliders");
    pairFlags =
        createContactPipeline(@"mr_world_flag_eligible_pairs");
    collisionCompile =
        createContactPipeline(@"mr_world_collide_compile");
    factorDispatch = createContactPipeline(
        @"mr_world_finalize_factor_dispatch"
    );
    pointQueryTail = createContactPipeline(
        @"mr_world_fill_point_query_tail"
    );
    evaluateIR =
        createContactPipeline(@"mr_world_evaluate_constraint_ir");
    islands =
        createContactPipeline(@"mr_world_build_contact_islands");
    contactSolve =
        createContactPipeline(@"mr_world_solve_contact_islands");
    contactIntegrate =
        createContactPipeline(@"mr_world_integrate_contact_state");
    contactLatch =
        createContactPipeline(@"mr_world_latch_contact_status");
    contactCommit =
        createContactPipeline(@"mr_world_commit_contact_state");
    contactCapture =
        createContactPipeline(@"mr_world_capture_contact");
    if (operatorPipeline == nil ||
        contactPrepare == nil ||
        bodyProjection == nil ||
        scenePrediction == nil ||
        colliderProjection == nil ||
        pairFlags == nil ||
        collisionCompile == nil ||
        factorDispatch == nil ||
        pointQueryTail == nil ||
        evaluateIR == nil ||
        islands == nil ||
        contactSolve == nil ||
        contactIntegrate == nil ||
        contactLatch == nil ||
        contactCommit == nil ||
        contactCapture == nil) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::metalPipelineFailure,
            "failed to create device-resident contact pipeline: " +
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
        operatorPipeline.maxTotalThreadsPerThreadgroup <
            kOperatorThreadsPerThreadgroup ||
        operatorPipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength ||
        prepare.maxTotalThreadsPerThreadgroup == 0u ||
        commit.maxTotalThreadsPerThreadgroup == 0u ||
        capture.maxTotalThreadsPerThreadgroup == 0u ||
        contactPrepare.maxTotalThreadsPerThreadgroup == 0u ||
        bodyProjection.maxTotalThreadsPerThreadgroup == 0u ||
        scenePrediction.maxTotalThreadsPerThreadgroup == 0u ||
        colliderProjection.maxTotalThreadsPerThreadgroup == 0u ||
        pairFlags.maxTotalThreadsPerThreadgroup == 0u ||
        collisionCompile.maxTotalThreadsPerThreadgroup == 0u ||
        factorDispatch.maxTotalThreadsPerThreadgroup == 0u ||
        pointQueryTail.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        evaluateIR.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        islands.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        contactSolve.maxTotalThreadsPerThreadgroup <
            kWorldThreadsPerThreadgroup ||
        contactIntegrate.maxTotalThreadsPerThreadgroup == 0u ||
        contactLatch.maxTotalThreadsPerThreadgroup == 0u ||
        contactCommit.maxTotalThreadsPerThreadgroup == 0u ||
        contactCapture.maxTotalThreadsPerThreadgroup == 0u) {
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
    context.operatorPipeline = operatorPipeline;
    context.contactPreparePipeline = contactPrepare;
    context.bodyProjectionPipeline = bodyProjection;
    context.scenePredictionPipeline = scenePrediction;
    context.colliderProjectionPipeline = colliderProjection;
    context.pairFlagPipeline = pairFlags;
    context.collisionCompilePipeline = collisionCompile;
    context.factorDispatchPipeline = factorDispatch;
    context.pointQueryTailPipeline = pointQueryTail;
    context.evaluateIRPipeline = evaluateIR;
    context.islandPipeline = islands;
    context.contactSolvePipeline = contactSolve;
    context.contactIntegratePipeline = contactIntegrate;
    context.contactLatchPipeline = contactLatch;
    context.contactCommitPipeline = contactCommit;
    context.contactCapturePipeline = contactCapture;
    context.initialized = true;
    context.stats.pipelineCreationCount += 21u;
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
            (index >= kArticulations && index <= kBodies) ||
            index == kShapes ||
            index == kMaterials ||
            index == kSceneBodyIndices ||
            index == kEligiblePairs;
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
        MRShapeGPU emptyShape{};
        MRMaterialGPU emptyMaterial{};
        mr_u32 emptySceneIndex = 0u;
        MRCompiledCollisionPairGPU emptyPair{};
        const std::array<std::pair<std::size_t, const void*>, 4u>
            contactSources{{
                {
                    kShapes,
                    model.shapes.empty()
                        ? static_cast<const void*>(&emptyShape)
                        : static_cast<const void*>(
                              model.shapes.data()
                          ),
                },
                {
                    kMaterials,
                    model.materials.empty()
                        ? static_cast<const void*>(&emptyMaterial)
                        : static_cast<const void*>(
                              model.materials.data()
                          ),
                },
                {
                    kSceneBodyIndices,
                    world.sceneBodyIndices().empty()
                        ? static_cast<const void*>(
                              &emptySceneIndex
                          )
                        : static_cast<const void*>(
                              world.sceneBodyIndices().data()
                          ),
                },
                {
                    kEligiblePairs,
                    world.eligiblePairs().empty()
                        ? static_cast<const void*>(&emptyPair)
                        : static_cast<const void*>(
                              world.eligiblePairs().data()
                          ),
                },
            }};
        for (const auto& [index, source] : contactSources) {
            copyToBuffer(
                context.buffers[index],
                source,
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
    copyToBuffer(
        context.buffers[kContactDispatch],
        &layout.contactDispatch,
        requirements.entries[kContactDispatch]
    );
    MRArticulatedOperatorDispatchGPU kinematicsDispatch{};
    kinematicsDispatch.articulationIndex =
        world.articulationIndex();
    kinematicsDispatch.environmentCount =
        layout.dispatch.environmentCount;
    kinematicsDispatch.flags =
        MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY;
    kinematicsDispatch.qStride = layout.dispatch.qStride;
    kinematicsDispatch.bodyPoseStride = world.bodyCount();
    kinematicsDispatch.generalizedStride =
        layout.dispatch.nv;
    MRArticulatedOperatorDispatchGPU factorDispatch =
        kinematicsDispatch;
    factorDispatch.pointCount =
        layout.contactDispatch.pointQueryStride;
    factorDispatch.flags =
        MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR;
    factorDispatch.pointStride = factorDispatch.pointCount;
    factorDispatch.pointWorldStride = factorDispatch.pointCount;
    factorDispatch.massMatrixStride =
        layout.contactDispatch.factorStride;
    factorDispatch.pointJacobianStride =
        factorDispatch.pointCount * 3u * layout.dispatch.nv;
    copyToBuffer(
        context.buffers[kOperatorKinematicsDispatch],
        &kinematicsDispatch,
        requirements.entries[kOperatorKinematicsDispatch]
    );
    copyToBuffer(
        context.buffers[kOperatorFactorDispatch],
        &factorDispatch,
        requirements.entries[kOperatorFactorDispatch]
    );
    copyToBuffer(
        context.buffers[kInitialSceneBodies],
        batch.initialSceneBodies.data(),
        requirements.entries[kInitialSceneBodies]
    );
    copyToBuffer(
        context.buffers[kSceneBodiesA],
        batch.initialSceneBodies.data(),
        requirements.entries[kSceneBodiesA]
    );
    copyToBuffer(
        context.buffers[kResetSceneBodies],
        batch.resetSceneBodies.data(),
        requirements.entries[kResetSceneBodies]
    );
    copyToBuffer(
        context.buffers[kKinematicTargets],
        batch.kinematicTargets.data(),
        requirements.entries[kKinematicTargets]
    );

    const std::array scratch{
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
        kSceneBodiesB,
        kCheckpointSceneBodies,
        kBodyPoses,
        kPointWorld,
        kFactorMatrix,
        kPointJacobians,
        kGeneralizedImpulse,
        kDeltaVelocity,
        kOperatorStatuses,
        kCurrentBodies,
        kCandidateBodies,
        kManifoldHeadersA,
        kManifoldPointsA,
        kManifoldCountsA,
        kManifoldHeadersB,
        kManifoldPointsB,
        kManifoldCountsB,
        kCandidateManifoldHeaders,
        kCandidateManifoldPoints,
        kCandidateManifoldCounts,
        kCheckpointManifoldHeaders,
        kCheckpointManifoldPoints,
        kCheckpointManifoldCounts,
        kCandidatePairs,
        kRawContacts,
        kRawPairIndices,
        kContacts,
        kContactMetadata,
        kIRBlocks,
        kIREndpoints,
        kIRRows,
        kIRCones,
        kPointQueries,
        kEvaluatedRows,
        kEvaluatedCones,
        kFactorCaches,
        kIslands,
        kResponseColumns,
        kContactStatuses,
        kPublicContactStatuses,
        kActiveIndirectDispatch,
        kProjectedColliders,
        kPairOverlapFlags,
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

struct MetalBufferBinding {
    NSUInteger argument = 0u;
    std::size_t buffer = 0u;
};

bool encodeContactThreadKernel(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    NSString* label,
    const std::initializer_list<MetalBufferBinding> bindings,
    const MRMetalWorldPassGPU* pass,
    const NSUInteger passArgument,
    const std::size_t environmentCount,
    const bool indirectDispatch = false,
    const NSUInteger indirectOffset = 0u
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = label;
    [encoder setComputePipelineState:pipeline];
    for (const MetalBufferBinding& binding : bindings) {
        [encoder setBuffer:context.buffers[binding.buffer]
                     offset:0u
                    atIndex:binding.argument];
    }
    if (pass != nullptr) {
        [encoder setBytes:pass
                   length:sizeof(*pass)
                  atIndex:passArgument];
    }
    if (indirectDispatch) {
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kActiveIndirectDispatch]
            indirectBufferOffset:indirectOffset
            threadsPerThreadgroup:MTLSizeMake(
                kWorldThreadsPerThreadgroup,
                1u,
                1u
            )];
    } else {
        dispatchWorldThreads(
            encoder,
            pipeline,
            environmentCount
        );
    }
    [encoder endEncoding];
    return true;
}

bool encodeArticulatedOperator(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const std::size_t dispatchBuffer,
    const std::size_t qBuffer,
    const std::size_t pointBuffer,
    const std::size_t environmentCount,
    NSString* label,
    const bool indirectDispatch
) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return false;
    }
    encoder.label = label;
    [encoder setComputePipelineState:context.operatorPipeline];
    const std::array<std::size_t, 15u> buffers{{
        kWorld,
        kArticulations,
        kJoints,
        kDofs,
        kBodies,
        dispatchBuffer,
        qBuffer,
        pointBuffer,
        kBodyPoses,
        kPointWorld,
        kFactorMatrix,
        kPointJacobians,
        kGeneralizedImpulse,
        kDeltaVelocity,
        kOperatorStatuses,
    }};
    for (NSUInteger argument = 0u;
         argument < buffers.size();
         ++argument) {
        [encoder setBuffer:context.buffers[buffers[argument]]
                     offset:0u
                    atIndex:argument];
    }
    const MTLSize threadgroupSize = MTLSizeMake(
        kOperatorThreadsPerThreadgroup,
        1u,
        1u
    );
    if (indirectDispatch) {
        [encoder
            dispatchThreadgroupsWithIndirectBuffer:
                context.buffers[kActiveIndirectDispatch]
            indirectBufferOffset:0u
            threadsPerThreadgroup:threadgroupSize];
    } else {
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(environmentCount),
                1u,
                1u
            )
            threadsPerThreadgroup:threadgroupSize];
    }
    [encoder endEncoding];
    return true;
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

bool encodeContactControlPrepare(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sourceScene,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.contactPreparePipeline,
        @"MetalWorld contact checkpoint/reset",
        {
            {0u, kWorldDispatch},
            {1u, kContactDispatch},
            {3u, kResetMasks},
            {4u, kResetSceneBodies},
            {5u, kKinematicTargets},
            {6u, kBodies},
            {7u, kSceneBodyIndices},
            {8u, sourceScene},
            {9u, kCheckpointSceneBodies},
            {10u, sourceManifoldHeaders},
            {11u, sourceManifoldPoints},
            {12u, sourceManifoldCounts},
            {13u, kCheckpointManifoldHeaders},
            {14u, kCheckpointManifoldPoints},
            {15u, kCheckpointManifoldCounts},
            {16u, kContactStatuses},
        },
        &pass,
        2u,
        environmentCount
    );
}

bool encodeContactSubstep(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const bool finalPhysicsSubstep,
    const std::size_t sourceQ,
    const std::size_t destinationQ,
    const std::size_t destinationV,
    const std::size_t sourceScene,
    const std::size_t destinationScene,
    const std::size_t sourceManifoldHeaders,
    const std::size_t sourceManifoldPoints,
    const std::size_t sourceManifoldCounts,
    const std::size_t destinationManifoldHeaders,
    const std::size_t destinationManifoldPoints,
    const std::size_t destinationManifoldCounts,
    const std::size_t environmentCount,
    const std::size_t colliderThreadCount,
    const std::size_t pairFlagThreadCount
) {
    MRMetalWorldPassGPU solverPass = pass;
    solverPass.reserved0 = finalPhysicsSubstep ? 1u : 0u;
    if (!encodeArticulatedOperator(
            context,
            commandBuffer,
            kOperatorKinematicsDispatch,
            sourceQ,
            kPointQueries,
            environmentCount,
            @"MetalWorld articulation kinematics",
            false
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.bodyProjectionPipeline,
            @"MetalWorld body/collider projection",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kBodies},
                {3u, kSceneBodyIndices},
                {4u, kBodyPoses},
                {5u, kOperatorStatuses},
                {6u, sourceScene},
                {7u, kCurrentBodies},
                {8u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.scenePredictionPipeline,
            @"MetalWorld scene prediction",
            {
                {0u, kWorld},
                {1u, kContactDispatch},
                {2u, kBodies},
                {3u, kSceneBodyIndices},
                {4u, kCurrentBodies},
                {5u, kCandidateBodies},
                {6u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.colliderProjectionPipeline,
            @"MetalWorld collider transform/AABB projection",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kCurrentBodies},
                {3u, kProjectedColliders},
            },
            nullptr,
            0u,
            colliderThreadCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.pairFlagPipeline,
            @"MetalWorld compiled-pair broadphase flags",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kEligiblePairs},
                {3u, kProjectedColliders},
                {4u, kPairOverlapFlags},
            },
            nullptr,
            0u,
            pairFlagThreadCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.collisionCompilePipeline,
            @"MetalWorld collision/manifold/IR compile",
            {
                {0u, kContactDispatch},
                {1u, kShapes},
                {2u, kMaterials},
                {3u, kCurrentBodies},
                {4u, kArticulations},
                {5u, kEligiblePairs},
                {6u, sourceManifoldCounts},
                {7u, sourceManifoldHeaders},
                {8u, sourceManifoldPoints},
                {9u, kCandidatePairs},
                {10u, kRawContacts},
                {11u, kRawPairIndices},
                {12u, kCandidateManifoldHeaders},
                {13u, kCandidateManifoldPoints},
                {14u, kCandidateManifoldCounts},
                {15u, kContacts},
                {16u, kContactMetadata},
                {17u, kIRBlocks},
                {18u, kIREndpoints},
                {19u, kIRRows},
                {20u, kIRCones},
                {21u, kPointQueries},
                {22u, kContactStatuses},
                {24u, kProjectedColliders},
                {25u, kPairOverlapFlags},
            },
            &pass,
            23u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.factorDispatchPipeline,
            @"MetalWorld active point-query reduction",
            {
                {0u, kContactDispatch},
                {1u, kContactStatuses},
                {2u, kOperatorFactorDispatch},
                {3u, kActiveIndirectDispatch},
            },
            nullptr,
            0u,
            1u
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.pointQueryTailPipeline,
            @"MetalWorld point-query tail fill",
            {
                {0u, kContactDispatch},
                {1u, kOperatorFactorDispatch},
                {2u, kArticulations},
                {3u, kContactStatuses},
                {4u, kPointQueries},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        !encodeArticulatedOperator(
            context,
            commandBuffer,
            kOperatorFactorDispatch,
            sourceQ,
            kPointQueries,
            environmentCount,
            @"MetalWorld articulated factor/Jacobians",
            true
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.evaluateIRPipeline,
            @"MetalWorld ConstraintIR evaluation",
            {
                {0u, kContactDispatch},
                {1u, kContacts},
                {2u, kContacts},
                {3u, kIRBlocks},
                {4u, kIRRows},
                {5u, kIRCones},
                {6u, kCandidateBodies},
                {7u, kCandidateV},
                {8u, kPointJacobians},
                {9u, kOperatorStatuses},
                {10u, kEvaluatedRows},
                {11u, kEvaluatedCones},
                {12u, kFactorCaches},
                {13u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.islandPipeline,
            @"MetalWorld mixed contact islands",
            {
                {0u, kContactDispatch},
                {1u, kCandidateBodies},
                {2u, kContacts},
                {3u, kIRBlocks},
                {4u, kIslands},
                {5u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactSolvePipeline,
            @"MetalWorld exact-cone contact solve",
            {
                {0u, kContactDispatch},
                {1u, kFactorMatrix},
                {2u, kPointJacobians},
                {3u, kCandidateV},
                {4u, kCandidateBodies},
                {5u, kContacts},
                {6u, kContactMetadata},
                {7u, kEvaluatedRows},
                {8u, kEvaluatedCones},
                {9u, kResponseColumns},
                {10u, kCandidateManifoldPoints},
                {11u, kContactStatuses},
            },
            &solverPass,
            12u,
            environmentCount,
            true,
            sizeof(MRIndirectDispatchArgumentsGPU)
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactIntegratePipeline,
            @"MetalWorld constrained integration",
            {
                {0u, kContactDispatch},
                {1u, kArticulations},
                {2u, kJoints},
                {3u, kBodies},
                {4u, kSceneBodyIndices},
                {5u, sourceQ},
                {6u, kCandidateV},
                {7u, kCandidateQ},
                {8u, kCandidateBodies},
                {9u, kContactStatuses},
            },
            nullptr,
            0u,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactLatchPipeline,
            @"MetalWorld contact failure latch",
            {
                {0u, kWorldDispatch},
                {2u, kContactStatuses},
                {3u, kEnvironmentStatuses},
            },
            &pass,
            1u,
            environmentCount
        ) ||
        !encodeCommit(
            context,
            commandBuffer,
            pass,
            destinationQ,
            destinationV,
            environmentCount
        ) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.contactCommitPipeline,
            @"MetalWorld contact transactional commit",
            {
                {0u, kWorldDispatch},
                {1u, kContactDispatch},
                {3u, kEnvironmentStatuses},
                {4u, kSceneBodyIndices},
                {5u, kCandidateBodies},
                {6u, kCheckpointSceneBodies},
                {7u, destinationScene},
                {8u, kCandidateManifoldHeaders},
                {9u, kCandidateManifoldPoints},
                {10u, kCandidateManifoldCounts},
                {11u, kCheckpointManifoldHeaders},
                {12u, kCheckpointManifoldPoints},
                {13u, kCheckpointManifoldCounts},
                {14u, destinationManifoldHeaders},
                {15u, destinationManifoldPoints},
                {16u, destinationManifoldCounts},
            },
            &pass,
            2u,
            environmentCount
        )) {
        return false;
    }
    return true;
}

bool encodeContactCapture(
    detail::MetalWorldContextState& context,
    id<MTLCommandBuffer> commandBuffer,
    const MRMetalWorldPassGPU& pass,
    const std::size_t sceneState,
    const std::size_t environmentCount
) {
    return encodeContactThreadKernel(
        context,
        commandBuffer,
        context.contactCapturePipeline,
        @"MetalWorld contact observation/status capture",
        {
            {0u, kWorldDispatch},
            {1u, kContactDispatch},
            {3u, sceneState},
            {4u, kContactStatuses},
            {5u, kObservations},
            {6u, kPublicContactStatuses},
        },
        &pass,
        2u,
        environmentCount
    );
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
    const bool contactMode =
        (dispatch.flags & MR_METAL_WORLD_CONTACTS) != 0u;
    const MRMetalWorldContactDispatchGPU& contactDispatch =
        diagnostics.layout.contactDispatch;
    const std::size_t observationWidth =
        dispatch.observationEnvironmentStride;
    staged.environmentStatuses.resize(
        dispatch.environmentCount
    );
    std::vector<std::uint64_t> retainedPointTotals(
        dispatch.environmentCount,
        0u
    );
    std::vector<std::uint64_t> observedPointTotals(
        dispatch.environmentCount,
        0u
    );
    for (std::uint32_t environment = 0u;
         environment < dispatch.environmentCount;
         ++environment) {
        staged.environmentStatuses[environment].environment =
            environment;
    }
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
            const MRMetalWorldContactStatusGPU* contactStatus =
                contactMode
                ? &staged.contactStatuses[statusIndex]
                : nullptr;
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
            if (contactStatus != nullptr &&
                (
                    contactStatus->environment != environment ||
                    contactStatus->controlStep != controlStep ||
                    contactStatus->physicsSubstep >=
                        dispatch.physicsSubsteps ||
                    contactStatus->code > MR_STEP_UNSUPPORTED ||
                    !finite(contactStatus->residuals) ||
                    !finite(contactStatus->diagnostics) ||
                    contactStatus->activePairs >
                        contactStatus->requiredPairs ||
                    contactStatus->activeContacts >
                        contactStatus->requiredConstraints ||
                    contactStatus->islandCount >
                        contactStatus->requiredIslands
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalWorldHostStatus::internalFailure,
                    "GPU returned a malformed contact-world status record"
                );
            }
            MetalWorldStatus& summary =
                staged.environmentStatuses[environment];
            if (status.code == MR_STEP_SUCCESS) {
                ++summary.successfulControlSteps;
            } else {
                ++summary.failedControlSteps;
                if (summary.code == MR_STEP_SUCCESS) {
                    summary.code = status.code;
                    summary.firstFailingControlStep =
                        static_cast<std::uint32_t>(controlStep);
                    if (contactStatus != nullptr) {
                        summary.firstFailingPair =
                            contactStatus->firstFailingPair;
                        summary.firstFailingConstraint =
                            contactStatus
                                ->firstFailingConstraint;
                        if (summary.firstFailingPair !=
                            MR_INVALID_INDEX) {
                            summary.firstFailingStableKey =
                                summary.firstFailingPair;
                        } else if (
                            summary.firstFailingConstraint !=
                            MR_INVALID_INDEX
                        ) {
                            summary.firstFailingStableKey =
                                (std::uint64_t{1} << 63u) |
                                summary.firstFailingConstraint;
                        }
                    }
                }
            }
            if (contactStatus != nullptr) {
                auto updateMaximum = [](
                    std::uint32_t& target,
                    const std::uint32_t value
                ) {
                    target = std::max(target, value);
                };
                updateMaximum(
                    summary.required.candidatePairs,
                    contactStatus->requiredPairs
                );
                updateMaximum(
                    summary.required.rawContacts,
                    contactStatus->requiredRawContacts
                );
                updateMaximum(
                    summary.required.manifolds,
                    contactStatus->requiredManifolds
                );
                updateMaximum(
                    summary.required.constraintBlocks,
                    contactStatus->requiredConstraints
                );
                updateMaximum(
                    summary.required.constraintRows,
                    contactStatus->requiredRows
                );
                updateMaximum(
                    summary.required.islands,
                    contactStatus->requiredIslands
                );
                updateMaximum(
                    summary.required.spillRows,
                    contactStatus->spillRows
                );
                updateMaximum(
                    summary.highWater.candidatePairs,
                    std::min(
                        contactStatus->activePairs,
                        contactDispatch.pairCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.rawContacts,
                    std::min(
                        contactStatus->requiredRawContacts,
                        contactDispatch.rawContactCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.manifolds,
                    std::min(
                        contactStatus->requiredManifolds,
                        contactDispatch.manifoldCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.constraintBlocks,
                    std::min(
                        contactStatus->activeContacts,
                        contactDispatch.constraintCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.constraintRows,
                    std::min(
                        contactStatus->requiredRows,
                        contactDispatch.rowCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.islands,
                    std::min(
                        contactStatus->islandCount,
                        contactDispatch.islandCapacity
                    )
                );
                updateMaximum(
                    summary.highWater.spillRows,
                    contactStatus->spillRows
                );
                updateMaximum(
                    summary.maximumSolverIterations,
                    contactStatus->solverIterations
                );
                const std::array residuals{
                    contactStatus->residuals.x,
                    contactStatus->residuals.y,
                    contactStatus->residuals.z,
                    contactStatus->residuals.w,
                };
                for (std::size_t index = 0u;
                     index < residuals.size();
                     ++index) {
                    summary.maximumResiduals[index] = std::max(
                        summary.maximumResiduals[index],
                        std::abs(residuals[index])
                    );
                }
                retainedPointTotals[environment] +=
                    contactStatus->retainedPoints;
                observedPointTotals[environment] +=
                    static_cast<std::uint64_t>(
                        contactStatus->retainedPoints
                    ) + contactStatus->newPoints;
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
                    (contactStatus != nullptr &&
                     (
                         contactStatus->code != MR_STEP_SUCCESS ||
                         contactStatus->requiredPairs >
                             contactDispatch.pairCapacity ||
                         contactStatus->requiredRawContacts >
                             contactDispatch.rawContactCapacity ||
                         contactStatus->requiredManifolds >
                             contactDispatch.manifoldCapacity ||
                         contactStatus->requiredConstraints >
                             contactDispatch.constraintCapacity ||
                         contactStatus->requiredRows >
                             contactDispatch.rowCapacity ||
                         contactStatus->requiredIslands >
                             contactDispatch.islandCapacity
                     )) ||
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
                    (!contactMode &&
                     status.abaCode == MR_ABA_SUCCESS) ||
                    (contactStatus != nullptr &&
                     contactStatus->code == MR_STEP_SUCCESS) ||
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
    for (std::size_t environment = 0u;
         environment < dispatch.environmentCount;
         ++environment) {
        staged.environmentStatuses[environment]
            .manifoldRetention =
            observedPointTotals[environment] == 0u
            ? 1.0f
            : static_cast<float>(
                  retainedPointTotals[environment]
              ) /
                static_cast<float>(
                    observedPointTotals[environment]
                );
    }

    if (!finiteFloats(staged.finalQ) ||
        !finiteFloats(staged.finalV)) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "GPU published a non-finite final MetalWorld state"
        );
    }
    if (contactMode &&
        !std::all_of(
            staged.finalSceneBodies.begin(),
            staged.finalSceneBodies.end(),
            [](const MRBodyStateGPU& state) {
                return finite(state.position) &&
                    finite(state.orientation) &&
                    finite(
                        state.linearVelocityAndInverseMass
                    ) &&
                    finite(state.angularVelocity) &&
                    finite(state.inverseInertiaWorldRow0) &&
                    finite(state.inverseInertiaWorldRow1) &&
                    finite(state.inverseInertiaWorldRow2);
            }
        )) {
        return reject(
            std::move(diagnostics),
            MetalWorldHostStatus::internalFailure,
            "GPU published a non-finite final scene-body state"
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

std::uint32_t CompiledWorld::sceneBodyCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(sceneBodyIndices_.size())
        : 0u;
}

std::uint32_t CompiledWorld::colliderCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(model_.shapes.size())
        : 0u;
}

std::uint32_t CompiledWorld::eligiblePairCount() const noexcept {
    return valid()
        ? static_cast<std::uint32_t>(eligiblePairs_.size())
        : 0u;
}

std::span<const std::uint32_t>
CompiledWorld::sceneBodyIndices() const noexcept {
    return sceneBodyIndices_;
}

std::span<const MRCompiledCollisionPairGPU>
CompiledWorld::eligiblePairs() const noexcept {
    return eligiblePairs_;
}

const MetalWorldCapacityProfile& CompiledWorld::capacities()
    const noexcept {
    return capacities_;
}

const MetalWorldCapacityProfile&
CompiledWorld::minimumCapacities() const noexcept {
    return minimumCapacities_;
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
    CompiledWorld& compiled,
    const MetalWorldCapacityProfile& requestedCapacities
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

        for (std::uint32_t body = 0u;
             body < staged.model_.bodies.size();
             ++body) {
            if (staged.model_.bodies[body].articulationIndex ==
                MR_INVALID_INDEX) {
                staged.sceneBodyIndices_.push_back(body);
            }
        }
        for (std::uint32_t colliderA = 0u;
             colliderA < staged.model_.shapes.size();
             ++colliderA) {
            const MRShapeGPU& shapeA =
                staged.model_.shapes[colliderA];
            if ((shapeA.flags &
                 MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u) {
                continue;
            }
            for (std::uint32_t colliderB = colliderA + 1u;
                 colliderB < staged.model_.shapes.size();
                 ++colliderB) {
                const MRShapeGPU& shapeB =
                    staged.model_.shapes[colliderB];
                if ((shapeB.flags &
                     MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
                    shapeA.bodyIndex == shapeB.bodyIndex ||
                    (shapeA.collisionGroup &
                     shapeB.collisionMask) == 0u ||
                    (shapeB.collisionGroup &
                     shapeA.collisionMask) == 0u) {
                    continue;
                }
                const MRBodyPropertiesGPU& bodyA =
                    staged.model_.bodies[shapeA.bodyIndex];
                const MRBodyPropertiesGPU& bodyB =
                    staged.model_.bodies[shapeB.bodyIndex];
                if (bodyA.motionType != MR_MOTION_DYNAMIC &&
                    bodyB.motionType != MR_MOTION_DYNAMIC) {
                    continue;
                }
                // Directly joint-connected links are a canonical cooker
                // exclusion. Solving their intentionally overlapping
                // collision proxies creates rank-deficient self constraints
                // and fights the authored joint.
                if (bodyA.articulationIndex != MR_INVALID_INDEX &&
                    bodyA.articulationIndex ==
                        bodyB.articulationIndex &&
                    (bodyA.parentBody == shapeB.bodyIndex ||
                     bodyB.parentBody == shapeA.bodyIndex)) {
                    continue;
                }
                staged.eligiblePairs_.push_back({
                    .colliderA = colliderA,
                    .colliderB = colliderB,
                    .pairClass = compiledPairClass(
                        shapeA.shapeType,
                        shapeB.shapeType
                    ),
                    .flags = 0u,
                });
            }
        }

        const auto inferred = [](
            const std::uint32_t requested,
            const std::uint32_t fallback
        ) {
            return requested == 0u ? fallback : requested;
        };
        const std::uint64_t eligibleCount =
            staged.eligiblePairs_.size();
        const std::uint32_t eligibleU32 =
            static_cast<std::uint32_t>(eligibleCount);
        const std::uint32_t defaultRaw =
            static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    std::max<std::uint64_t>(
                        staged.model_.world.contactCapacity,
                        MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR *
                            eligibleCount
                    ),
                    std::numeric_limits<std::uint32_t>::max()
                )
            );
        const std::uint32_t defaultConstraints =
            static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    std::max<std::uint64_t>(
                        staged.model_.world.contactCapacity,
                        4u * eligibleCount
                    ),
                    MR_ARTICULATED_OPERATOR_MAX_POINTS / 2u
                )
            );
        staged.capacities_.candidatePairs = inferred(
            requestedCapacities.candidatePairs,
            std::max(
                staged.model_.world.pairCapacity,
                eligibleU32
            )
        );
        staged.capacities_.rawContacts = inferred(
            requestedCapacities.rawContacts,
            defaultRaw
        );
        staged.capacities_.manifolds = inferred(
            requestedCapacities.manifolds,
            std::max(
                staged.model_.world.contactCapacity,
                eligibleU32
            )
        );
        staged.capacities_.constraintBlocks = inferred(
            requestedCapacities.constraintBlocks,
            defaultConstraints
        );
        const std::uint64_t requiredRows =
            3ull * staged.capacities_.constraintBlocks;
        if (requiredRows >
            std::numeric_limits<std::uint32_t>::max()) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::arithmeticOverflow,
                "contact row capacity overflows the GPU ABI"
            );
        }
        staged.capacities_.constraintRows = inferred(
            requestedCapacities.constraintRows,
            static_cast<std::uint32_t>(requiredRows)
        );
        staged.capacities_.islands = inferred(
            requestedCapacities.islands,
            std::max(
                staged.model_.world.islandCapacity,
                static_cast<std::uint32_t>(
                    std::min<std::size_t>(
                        staged.model_.bodies.size(),
                        MR_ARTICULATED_OPERATOR_MAX_BODIES
                    )
                )
            )
        );
        staged.capacities_.spillRows =
            requestedCapacities.spillRows;
        staged.minimumCapacities_ = {
            .candidatePairs = eligibleU32,
            .rawContacts = defaultRaw,
            .manifolds = std::max(
                staged.model_.world.contactCapacity,
                eligibleU32
            ),
            .constraintBlocks = defaultConstraints,
            .constraintRows = 3u * defaultConstraints,
            .islands = std::max(
                staged.model_.world.islandCapacity,
                static_cast<std::uint32_t>(
                    std::min<std::size_t>(
                        staged.model_.bodies.size(),
                        MR_ARTICULATED_OPERATOR_MAX_BODIES
                    )
                )
            ),
            .spillRows = 0u,
        };
        if (staged.capacities_.candidatePairs == 0u ||
            staged.capacities_.rawContacts == 0u ||
            staged.capacities_.manifolds == 0u ||
            staged.capacities_.constraintBlocks == 0u ||
            staged.capacities_.constraintBlocks >
                MR_ARTICULATED_OPERATOR_MAX_POINTS / 2u ||
            staged.capacities_.constraintRows <
                requiredRows ||
            staged.capacities_.islands == 0u ||
            staged.model_.bodies.size() >
                MR_ARTICULATED_OPERATOR_MAX_BODIES ||
            staged.model_.shapes.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            staged.sceneBodyIndices_.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            staged.eligiblePairs_.size() >
                std::numeric_limits<std::uint32_t>::max()) {
            return rejectCompile(
                std::move(diagnostics),
                MetalWorldHostStatus::capacityOverflow,
                "contact capacity profile is empty, internally "
                "inconsistent, or exceeds the current "
                "64-body/512-contact bucket"
            );
        }

        staged.fingerprint_ = compiledFingerprint(
            staged.model_,
            staged.capacities_,
            staged.sceneBodyIndices_,
            staged.eligiblePairs_
        );
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
            if (pending->contactMode) {
                staged.finalSceneBodies.resize(
                    staged.layout.initialSceneBodyElements
                );
                staged.contactStatuses.resize(
                    staged.layout.contactStatusElements
                );
                if (pending->captureContactEvidence) {
                    auto& evidence = staged.contactEvidence;
                    evidence.manifoldHeaders.resize(
                        staged.layout.manifoldHeaderElements
                    );
                    evidence.manifoldPoints.resize(
                        staged.layout.manifoldPointElements
                    );
                    evidence.manifoldCounts.resize(
                        staged.layout.dispatch.environmentCount
                    );
                    evidence.contacts.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.contactMetadata.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.blocks.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.endpoints.resize(
                        2u *
                            staged.layout
                                .contactConstraintElements
                    );
                    evidence.rows.resize(
                        staged.layout.constraintRowElements
                    );
                    evidence.cones.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.evaluatedRows.resize(
                        staged.layout.constraintRowElements
                    );
                    evidence.evaluatedCones.resize(
                        staged.layout.contactConstraintElements
                    );
                    evidence.islands.resize(
                        staged.layout.islandElements
                    );
                }
            }
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
            if (pending->contactMode) {
                copyOutput(
                    staged.finalSceneBodies,
                    buffers[pending->finalSceneBuffer]
                );
                copyOutput(
                    staged.contactStatuses,
                    buffers[kPublicContactStatuses]
                );
                if (pending->captureContactEvidence) {
                    auto& evidence = staged.contactEvidence;
                    copyOutput(
                        evidence.manifoldHeaders,
                        buffers[
                            pending->finalManifoldHeaderBuffer
                        ]
                    );
                    copyOutput(
                        evidence.manifoldPoints,
                        buffers[
                            pending->finalManifoldPointBuffer
                        ]
                    );
                    copyOutput(
                        evidence.manifoldCounts,
                        buffers[
                            pending->finalManifoldCountBuffer
                        ]
                    );
                    copyOutput(
                        evidence.contacts,
                        buffers[kContacts]
                    );
                    copyOutput(
                        evidence.contactMetadata,
                        buffers[kContactMetadata]
                    );
                    copyOutput(
                        evidence.blocks,
                        buffers[kIRBlocks]
                    );
                    copyOutput(
                        evidence.endpoints,
                        buffers[kIREndpoints]
                    );
                    copyOutput(
                        evidence.rows,
                        buffers[kIRRows]
                    );
                    copyOutput(
                        evidence.cones,
                        buffers[kIRCones]
                    );
                    copyOutput(
                        evidence.evaluatedRows,
                        buffers[kEvaluatedRows]
                    );
                    copyOutput(
                        evidence.evaluatedCones,
                        buffers[kEvaluatedCones]
                    );
                    copyOutput(
                        evidence.islands,
                        buffers[kIslands]
                    );
                }
            }
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
            const bool contactMode =
                config.solverMode !=
                    MetalWorldSolverMode::freeMotionABA;
            std::size_t sourceScene = kSceneBodiesA;
            std::size_t destinationScene = kSceneBodiesB;
            std::size_t sourceManifoldHeaders =
                kManifoldHeadersA;
            std::size_t sourceManifoldPoints =
                kManifoldPointsA;
            std::size_t sourceManifoldCounts =
                kManifoldCountsA;
            std::size_t destinationManifoldHeaders =
                kManifoldHeadersB;
            std::size_t destinationManifoldPoints =
                kManifoldPointsB;
            std::size_t destinationManifoldCounts =
                kManifoldCountsB;
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
                if (contactMode &&
                    !encodeContactControlPrepare(
                        *state_,
                        commandBuffer,
                        pass,
                        sourceScene,
                        sourceManifoldHeaders,
                        sourceManifoldPoints,
                        sourceManifoldCounts,
                        batch.environmentCount
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalWorldHostStatus::metalCommandFailure,
                        "failed to encode contact checkpoint/reset pass"
                    );
                }

                for (std::uint32_t physicsSubstep = 0u;
                     physicsSubstep < config.physicsSubsteps;
                     ++physicsSubstep) {
                    pass.physicsSubstep = physicsSubstep;
                    const bool encodedABA = encodeABA(
                            *state_,
                            commandBuffer,
                            selectedABAPipeline,
                            sourceQ,
                            sourceV,
                            batch.environmentCount
                        );
                    const bool encodedPublication =
                        contactMode
                        ? encodeContactSubstep(
                              *state_,
                              commandBuffer,
                              pass,
                              physicsSubstep + 1u ==
                                  config.physicsSubsteps,
                              sourceQ,
                              destinationQ,
                              destinationV,
                              sourceScene,
                              destinationScene,
                              sourceManifoldHeaders,
                              sourceManifoldPoints,
                              sourceManifoldCounts,
                              destinationManifoldHeaders,
                              destinationManifoldPoints,
                              destinationManifoldCounts,
                              batch.environmentCount,
                              requirements.entries[
                                  kProjectedColliders
                              ].logicalElements,
                              std::max<std::size_t>(
                                  requirements.entries[
                                      kPairOverlapFlags
                                  ].logicalElements,
                                  1u
                              )
                          )
                        : encodeCommit(
                              *state_,
                              commandBuffer,
                              pass,
                              destinationQ,
                              destinationV,
                              batch.environmentCount
                          );
                    if (!encodedABA || !encodedPublication) {
                        return reject(
                            std::move(diagnostics),
                            MetalWorldHostStatus::metalCommandFailure,
                            "failed to encode MetalWorld substep graph"
                        );
                    }
                    std::swap(sourceQ, destinationQ);
                    std::swap(sourceV, destinationV);
                    if (contactMode) {
                        std::swap(sourceScene, destinationScene);
                        std::swap(
                            sourceManifoldHeaders,
                            destinationManifoldHeaders
                        );
                        std::swap(
                            sourceManifoldPoints,
                            destinationManifoldPoints
                        );
                        std::swap(
                            sourceManifoldCounts,
                            destinationManifoldCounts
                        );
                    }
                }

                pass.physicsSubstep = MR_INVALID_INDEX;
                if (!encodeCapture(
                        *state_,
                        commandBuffer,
                        pass,
                        sourceQ,
                        sourceV,
                        batch.environmentCount
                    ) ||
                    (contactMode &&
                     !encodeContactCapture(
                         *state_,
                         commandBuffer,
                         pass,
                         sourceScene,
                         batch.environmentCount
                     ))) {
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
            pending->finalSceneBuffer = sourceScene;
            pending->finalManifoldHeaderBuffer =
                sourceManifoldHeaders;
            pending->finalManifoldPointBuffer =
                sourceManifoldPoints;
            pending->finalManifoldCountBuffer =
                sourceManifoldCounts;
            pending->contactMode = contactMode;
            pending->captureContactEvidence =
                config.captureContactEvidence;
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
