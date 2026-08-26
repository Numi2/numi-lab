#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"

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

// The final stream carries one canonical packed OpenSim SpatialTransform per
// global joint. Non-FunctionBased slots are all-zero and are never consumed.
// Shared kernel ABI. The first sixteen slots are the established articulated
// operator stream; the Millard sidecar consumes those private outputs and
// occupies slots 16..23 in the same command buffer.
constexpr std::size_t kRawBufferCount = 24u;
constexpr std::size_t kMillardDispatchBuffer = 16u;
constexpr std::size_t kMillardMusclesBuffer = 17u;
constexpr std::size_t kMillardStatesBuffer = 18u;
constexpr std::size_t kMillardPathPointsBuffer = 19u;
constexpr std::size_t kMillardCurvesBuffer = 20u;
constexpr std::size_t kMillardWrapsBuffer = 21u;
constexpr std::size_t kMillardResultsBuffer = 22u;
constexpr std::size_t kMillardForcesBuffer = 23u;
constexpr NSUInteger kThreadsPerThreadgroup = 32u;
constexpr float kQuaternionHostTolerance = 1.9e-5f;
constexpr std::uint64_t kShaderAddressableElements =
    static_cast<std::uint64_t>(
        std::numeric_limits<mr_u32>::max()
    ) + 1u;
const char kMetalRoboImageAnchor = 0;

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

struct MetalArticulatedOperatorContextState {
    explicit MetalArticulatedOperatorContextState(
        MetalArticulatedOperatorConfig configured
    )
        : config(std::move(configured)) {}

    MetalArticulatedOperatorConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool inFlight = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> pipeline = nil;
    __strong id<MTLComputePipelineState> millardPipeline = nil;
    __strong id<MTLBuffer> buffers[kRawBufferCount] = {};
    std::array<std::size_t, kRawBufferCount> capacities{};
    MetalArticulatedOperatorContextStats stats{};
};

struct MetalArticulatedOperatorSubmissionState {
    ~MetalArticulatedOperatorSubmissionState() {
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
            // Destructors must not throw. The command is complete, so even a
            // platform mutex failure cannot leave GPU work accessing memory.
        }
        ownsInFlight = false;
    }

    std::shared_ptr<MetalArticulatedOperatorContextState> context;
    __strong id<MTLCommandBuffer> commandBuffer = nil;
    MetalArticulatedOperatorDiagnostics diagnostics{};
    std::chrono::steady_clock::time_point start{};
    MRArticulationGPU articulation{};
    std::uint32_t articulationIndex = 0u;
    std::size_t pointCount = 0u;
    bool hasMillardReference = false;
    std::size_t millardMuscleCount = 0u;
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
    if (dladdr(&kMetalRoboImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory /
                "metalrobo/MetalRobo.metallib",
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

MetalArticulatedOperatorDiagnostics reject(
    MetalArticulatedOperatorDiagnostics diagnostics,
    const MetalArticulatedOperatorHostStatus status,
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
            MR_ARTICULATED_OPERATOR_MAX_BODIES ||
        articulation.nv == 0u ||
        articulation.nv >
            MR_ARTICULATED_OPERATOR_MAX_DOFS) {
        reason =
            "articulation exceeds the Metal operator body/DoF bucket";
        return false;
    }
    const std::size_t jointEnd =
        static_cast<std::size_t>(articulation.firstJoint) +
        articulation.jointCount;
    for (std::size_t jointIndex = articulation.firstJoint;
         jointIndex < jointEnd;
         ++jointIndex) {
        const MRJointDescriptorGPU& joint = model.joints[jointIndex];
        if (joint.flags != 0u) {
            reason =
                "Metal articulated joints require zero reserved flags";
            return false;
        }
        if (joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC &&
            joint.jointType != MR_JOINT_FIXED &&
            joint.jointType != MR_JOINT_FUNCTION_BASED) {
            reason =
                "Metal articulated operator supports revolute, prismatic, "
                "continuous, fixed, and FunctionBased joints";
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
                "every body owned by a Metal articulation must be dynamic";
            return false;
        }
    }
    return true;
}

bool packFunctionPrograms(
    const EngineModel& model,
    std::vector<MROpenSimSpatialTransformGPU>& packed,
    std::string* reason = nullptr
) {
    packed.assign(std::max<std::size_t>(model.joints.size(), 1u), {});
    for (const FunctionBasedJointProgram& program :
         model.functionBasedJointPrograms) {
        if (program.jointIndex >= model.joints.size()) {
            if (reason != nullptr) {
                *reason = "FunctionBased program joint index is outside model";
            }
            return false;
        }
        const OpenSimSpatialTransformStatus status =
            packOpenSimSpatialTransformGPU(
                program.transform,
                packed[program.jointIndex]
            );
        if (status != OpenSimSpatialTransformStatus::success) {
            if (reason != nullptr) {
                *reason = std::string("FunctionBased program packing failed: ") +
                    openSimSpatialTransformStatusName(status);
            }
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
        const float x = q[base + 0u];
        const float y = q[base + 1u];
        const float z = q[base + 2u];
        const float w = q[base + 3u];
        const float normSquared =
            x * x + y * y + z * z + w * w;
        if (!(normSquared > 1.0e-12f) ||
            !std::isfinite(normSquared)) {
            return false;
        }
        const float norm = std::sqrt(normSquared);
        if (!std::isfinite(norm) ||
            std::abs(norm - 1.0f) >
                kQuaternionHostTolerance) {
            return false;
        }
    }
    return true;
}

bool validPoints(
    const MRArticulationGPU& articulation,
    const std::span<const MRArticulatedPointImpulseGPU> points
) {
    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (const MRArticulatedPointImpulseGPU& point : points) {
        if (point.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(point.bodyIndex) >= bodyEnd ||
            point.flags != 0u ||
            point.reserved0 != 0u ||
            point.reserved1 != 0u ||
            !finite(point.localPoint) ||
            !finite(point.worldImpulse) ||
            point.localPoint.w != 0.0f ||
            point.worldImpulse.w != 0.0f) {
            return false;
        }
    }
    return true;
}

bool validMillardReference(
    const MRArticulationGPU& articulation,
    const std::size_t environmentCount,
    const std::span<const MRArticulatedPointImpulseGPU> operatorPoints,
    const MetalMillardReferenceInput& millard,
    std::string& reason
) {
    const bool enabled = millard.enabled();
    if (!enabled) {
        if (!millard.states.empty() || !millard.pathPoints.empty() ||
            !millard.curves.empty() || !millard.cylinderWraps.empty()) {
            reason = "Millard sidecar has data but no muscle definitions";
            return false;
        }
        return true;
    }
    if (millard.muscles.size() > std::numeric_limits<mr_u32>::max() ||
        millard.pathPoints.size() > std::numeric_limits<mr_u32>::max() ||
        millard.cylinderWraps.size() > std::numeric_limits<mr_u32>::max() ||
        millard.curves.size() != millard.muscles.size()) {
        reason = "Millard source dimensions do not fit the device ABI";
        return false;
    }
    std::size_t expectedStateCount = 0u;
    if (!checkedMultiply(
            environmentCount,
            millard.muscles.size(),
            expectedStateCount
        ) || millard.states.size() != expectedStateCount) {
        reason = "Millard state stream is not environment-major";
        return false;
    }
    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (std::size_t index = 0u; index < millard.muscles.size(); ++index) {
        const MRMillardMuscleGPU& muscle = millard.muscles[index];
        if (!finite(muscle.forceAndLengths) ||
            !finite(muscle.dampingAndActivation) ||
            muscle.dampingAndActivation.w != 0.0f ||
            muscle.pathAndWrap.y < 2u ||
            muscle.pathAndWrap.x > millard.pathPoints.size() ||
            muscle.pathAndWrap.y >
                millard.pathPoints.size() - muscle.pathAndWrap.x ||
            muscle.pathAndWrap.z > millard.cylinderWraps.size() ||
            muscle.pathAndWrap.w >
                millard.cylinderWraps.size() - muscle.pathAndWrap.z ||
            muscle.pathAndWrap.w >
                MR_MILLARD_REFERENCE_MAX_WRAPS_PER_MUSCLE ||
            muscle.flags.y != 0u || muscle.flags.z != 0u ||
            muscle.flags.w != 0u) {
            reason = "Millard muscle definition is malformed";
            return false;
        }
        const MRMillardSourceCurveGPU& curve = millard.curves[index];
        for (const mr_float4 block : curve.values) {
            if (!finite(block)) {
                reason = "Millard source curves contain non-finite values";
                return false;
            }
        }
        if (curve.values[5u].z != 0.0f || curve.values[5u].w != 0.0f) {
            reason = "Millard source curve padding is nonzero";
            return false;
        }
    }
    for (const MRMillardMuscleStateGPU& state : millard.states) {
        if (!finite(state.activationAndVelocity) ||
            state.activationAndVelocity.z != 0.0f ||
            state.activationAndVelocity.w != 0.0f) {
            reason = "Millard state stream is non-finite or noncanonical";
            return false;
        }
    }
    for (const MRMillardPathPointGPU& point : millard.pathPoints) {
        if (point.pointQueryIndex >= operatorPoints.size() ||
            point.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(point.bodyIndex) >= bodyEnd ||
            point.bodyIndex != operatorPoints[point.pointQueryIndex].bodyIndex ||
            point.reserved0 != 0u || point.reserved1 != 0u) {
            reason = "Millard path point does not match an operator query";
            return false;
        }
    }
    for (const MRMillardCylinderWrapGPU& wrap : millard.cylinderWraps) {
        if (wrap.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(wrap.bodyIndex) >= bodyEnd ||
            wrap.reserved0 != 0u || wrap.reserved1 != 0u ||
            wrap.reserved2 != 0u || !finite(wrap.center) ||
            !finite(wrap.rotationAndRadius) || !finite(wrap.length) ||
            wrap.center.w != 0.0f || wrap.length.y != 0.0f ||
            wrap.length.z != 0.0f || wrap.length.w != 0.0f ||
            !(wrap.rotationAndRadius.w > 0.0f) || !(wrap.length.x > 0.0f)) {
            reason = "Millard cylinder wrap is malformed";
            return false;
        }
    }
    return true;
}

bool buildRequirements(
    const EngineModel& model,
    const MetalArticulatedOperatorLayout& layout,
    RequiredBuffers& requirements,
    std::size_t& totalAllocatedBytes
) {
    std::size_t jointElements =
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
        !makeRequirement<MRArticulatedOperatorDispatchGPU>(
            "dispatch",
            1u,
            requirements.entries[5]
        ) ||
        !makeRequirement<float>(
            "q",
            layout.qElements,
            requirements.entries[6]
        ) ||
        !makeRequirement<MRArticulatedPointImpulseGPU>(
            "point queries",
            layout.pointElements,
            requirements.entries[7]
        ) ||
        !makeRequirement<MRArticulatedBodyPoseGPU>(
            "body poses",
            layout.bodyPoseElements,
            requirements.entries[8]
        ) ||
        !makeRequirement<MRArticulatedPointWorldGPU>(
            "point world",
            layout.pointWorldElements,
            requirements.entries[9]
        ) ||
        !makeRequirement<float>(
            "diagnostic mass",
            layout.massMatrixElements,
            requirements.entries[10]
        ) ||
        !makeRequirement<float>(
            "point Jacobians",
            layout.pointJacobianElements,
            requirements.entries[11]
        ) ||
        !makeRequirement<float>(
            "generalized impulse",
            layout.generalizedElements,
            requirements.entries[12]
        ) ||
        !makeRequirement<float>(
            "delta velocity",
            layout.generalizedElements,
            requirements.entries[13]
        ) ||
        !makeRequirement<MRArticulatedOperatorStatusGPU>(
            "statuses",
            layout.statusElements,
            requirements.entries[14]
        ) ||
        !makeRequirement<MROpenSimSpatialTransformGPU>(
            "FunctionBased programs",
            jointElements,
            requirements.entries[15]
        ) ||
        !makeRequirement<MRMillardReferenceDispatchGPU>(
            "Millard dispatch",
            1u,
            requirements.entries[kMillardDispatchBuffer]
        ) ||
        !makeRequirement<MRMillardMuscleGPU>(
            "Millard muscles",
            layout.millardMuscleElements,
            requirements.entries[kMillardMusclesBuffer]
        ) ||
        !makeRequirement<MRMillardMuscleStateGPU>(
            "Millard states",
            layout.millardStateElements,
            requirements.entries[kMillardStatesBuffer]
        ) ||
        !makeRequirement<MRMillardPathPointGPU>(
            "Millard path points",
            layout.millardPathPointElements,
            requirements.entries[kMillardPathPointsBuffer]
        ) ||
        !makeRequirement<MRMillardSourceCurveGPU>(
            "Millard source curves",
            layout.millardCurveElements,
            requirements.entries[kMillardCurvesBuffer]
        ) ||
        !makeRequirement<MRMillardCylinderWrapGPU>(
            "Millard cylinder wraps",
            layout.millardWrapElements,
            requirements.entries[kMillardWrapsBuffer]
        ) ||
        !makeRequirement<MRMillardMuscleResultGPU>(
            "Millard results",
            layout.millardResultElements,
            requirements.entries[kMillardResultsBuffer]
        ) ||
        !makeRequirement<float>(
            "Millard generalized forces",
            layout.millardGeneralizedForceElements,
            requirements.entries[kMillardForcesBuffer]
        )) {
        return false;
    }

    totalAllocatedBytes = 0u;
    for (const BufferRequirement& requirement :
         requirements.entries) {
        if (!checkedAdd(
                totalAllocatedBytes,
                requirement.allocationBytes,
                totalAllocatedBytes
            )) {
            return false;
        }
    }
    return true;
}

MetalArticulatedOperatorDiagnostics validateAndBuildLayout(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorConfig& config,
    RequiredBuffers& requirements
) {
    MetalArticulatedOperatorDiagnostics diagnostics{};

    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidModel,
            "invalid EngineModel: " + modelReason
        );
    }
    std::vector<MROpenSimSpatialTransformGPU> packedPrograms;
    if (!packFunctionPrograms(model, packedPrograms, &modelReason)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidModel,
            "invalid FunctionBased program stream: " + modelReason
        );
    }
    if (input.articulationIndex >=
        model.articulations.size()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "articulation index is outside the canonical model"
        );
    }
    if (input.environmentCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "environmentCount must be greater than zero"
        );
    }
    if (input.environmentCount >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "environmentCount does not fit the GPU dispatch ABI"
        );
    }
    if (input.pointCount >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "pointCount does not fit the GPU dispatch ABI"
        );
    }
    if (input.pointCount >
        MR_ARTICULATED_OPERATOR_MAX_POINTS) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::capacityOverflow,
            "pointCount exceeds the compiled operator capacity"
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
                MR_ARTICULATED_OPERATOR_MAX_BODIES ||
            articulation.nv >
                MR_ARTICULATED_OPERATOR_MAX_DOFS;
        return reject(
            std::move(diagnostics),
            capacity
                ? MetalArticulatedOperatorHostStatus::
                      capacityOverflow
                : MetalArticulatedOperatorHostStatus::
                      unsupportedTopology,
            std::move(topologyReason)
        );
    }

    MetalArticulatedOperatorLayout layout{};
    MRArticulatedOperatorDispatchGPU& dispatch =
        layout.dispatch;
    dispatch.articulationIndex = input.articulationIndex;
    dispatch.environmentCount =
        static_cast<mr_u32>(input.environmentCount);
    dispatch.pointCount =
        static_cast<mr_u32>(input.pointCount);
    dispatch.flags = 0u;
    if (config.writeDiagnosticMassMatrix) {
        dispatch.flags |=
            MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS;
    }
    if (config.pointJacobiansOnly) {
        dispatch.flags |=
            MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY;
    }
    if (config.writeDiagnosticMassMatrix &&
        config.pointJacobiansOnly) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                invalidDimensions,
            "point-Jacobian-only mode cannot request a "
            "diagnostic mass matrix"
        );
    }
    dispatch.qStride = articulation.nq;
    dispatch.pointStride = dispatch.pointCount;
    dispatch.bodyPoseStride = articulation.bodyCount;
    dispatch.pointWorldStride = dispatch.pointCount;
    dispatch.generalizedStride = articulation.nv;

    std::size_t massStride = 0u;
    std::size_t jacobianStride = 0u;
    if (!checkedMultiply(
            articulation.nv,
            articulation.nv,
            massStride
        ) ||
        !checkedMultiply(
            input.pointCount,
            3u,
            jacobianStride
        ) ||
        !checkedMultiply(
            jacobianStride,
            articulation.nv,
            jacobianStride
        ) ||
        massStride > std::numeric_limits<mr_u32>::max() ||
        jacobianStride > std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "derived GPU stride overflow"
        );
    }
    dispatch.massMatrixStride =
        config.writeDiagnosticMassMatrix
            ? static_cast<mr_u32>(massStride)
            : 0u;
    dispatch.pointJacobianStride =
        static_cast<mr_u32>(jacobianStride);

    if (!checkedMultiply(
            input.environmentCount,
            dispatch.qStride,
            layout.qElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.pointStride,
            layout.pointElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.bodyPoseStride,
            layout.bodyPoseElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.pointWorldStride,
            layout.pointWorldElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.massMatrixStride,
            layout.massMatrixElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.pointJacobianStride,
            layout.pointJacobianElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.generalizedStride,
            layout.generalizedElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "derived GPU element-count overflow"
        );
    }
    layout.statusElements = input.environmentCount;

    if (input.millard.enabled()) {
        layout.millardMuscleElements = input.millard.muscles.size();
        layout.millardStateElements = input.millard.states.size();
        layout.millardPathPointElements = input.millard.pathPoints.size();
        layout.millardCurveElements = input.millard.curves.size();
        layout.millardWrapElements = input.millard.cylinderWraps.size();
        if (!checkedMultiply(
                input.environmentCount,
                layout.millardMuscleElements,
                layout.millardResultElements
            ) || !checkedMultiply(
                layout.millardResultElements,
                articulation.nv,
                layout.millardGeneralizedForceElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "derived Millard output element-count overflow"
            );
        }
    }

    const auto exceedsShaderAddressing =
        [](const std::size_t elements) {
            return static_cast<std::uint64_t>(elements) >
                kShaderAddressableElements;
        };
    if (exceedsShaderAddressing(layout.qElements) ||
        exceedsShaderAddressing(layout.pointElements) ||
        exceedsShaderAddressing(layout.bodyPoseElements) ||
        exceedsShaderAddressing(layout.pointWorldElements) ||
        exceedsShaderAddressing(layout.massMatrixElements) ||
        exceedsShaderAddressing(
            layout.pointJacobianElements
        ) ||
        exceedsShaderAddressing(layout.generalizedElements) ||
        exceedsShaderAddressing(layout.statusElements) ||
        exceedsShaderAddressing(layout.millardMuscleElements) ||
        exceedsShaderAddressing(layout.millardStateElements) ||
        exceedsShaderAddressing(layout.millardPathPointElements) ||
        exceedsShaderAddressing(layout.millardCurveElements) ||
        exceedsShaderAddressing(layout.millardWrapElements) ||
        exceedsShaderAddressing(layout.millardResultElements) ||
        exceedsShaderAddressing(layout.millardGeneralizedForceElements)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "compact buffer exceeds the shader's 32-bit element "
            "addressing contract"
        );
    }

    std::size_t totalAllocatedBytes = 0u;
    if (!buildRequirements(
            model,
            layout,
            requirements,
            totalAllocatedBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "required Metal buffer byte-count overflow"
        );
    }
    layout.qBytes = requirements.entries[6].logicalBytes;
    layout.pointBytes = requirements.entries[7].logicalBytes;
    layout.bodyPoseBytes =
        requirements.entries[8].logicalBytes;
    layout.pointWorldBytes =
        requirements.entries[9].logicalBytes;
    layout.massMatrixBytes =
        requirements.entries[10].logicalBytes;
    layout.pointJacobianBytes =
        requirements.entries[11].logicalBytes;
    layout.generalizedBytes =
        requirements.entries[12].logicalBytes;
    layout.statusBytes =
        requirements.entries[14].logicalBytes;
    layout.millardMuscleBytes =
        requirements.entries[kMillardMusclesBuffer].logicalBytes;
    layout.millardStateBytes =
        requirements.entries[kMillardStatesBuffer].logicalBytes;
    layout.millardPathPointBytes =
        requirements.entries[kMillardPathPointsBuffer].logicalBytes;
    layout.millardCurveBytes =
        requirements.entries[kMillardCurvesBuffer].logicalBytes;
    layout.millardWrapBytes =
        requirements.entries[kMillardWrapsBuffer].logicalBytes;
    layout.millardResultBytes =
        requirements.entries[kMillardResultsBuffer].logicalBytes;
    layout.millardGeneralizedForceBytes =
        requirements.entries[kMillardForcesBuffer].logicalBytes;
    layout.totalAllocatedBytes = totalAllocatedBytes;
    diagnostics.layout = layout;

    if (input.q.size() != layout.qElements ||
        input.points.size() != layout.pointElements) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "packed q or point-query span has the wrong element count"
        );
    }
    if (!validQ(
            articulation,
            input.environmentCount,
            input.q
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::nonfiniteInput,
            "q contains a non-finite value or invalid root quaternion"
        );
    }
    if (!validPoints(articulation, input.points)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidPointQuery,
            "point query is non-finite, reserved, or outside articulation"
        );
    }
    std::string millardReason;
    if (!validMillardReference(
            articulation,
            input.environmentCount,
            input.points,
            input.millard,
            millardReason
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "invalid Millard reference program: " + millardReason
        );
    }
    return diagnostics;
}

NSString* bufferLabel(const std::size_t index) {
    switch (index) {
    case 0u:
        return @"articulated world";
    case 1u:
        return @"articulation descriptors";
    case 2u:
        return @"joint descriptors";
    case 3u:
        return @"DoF properties";
    case 4u:
        return @"body properties";
    case 5u:
        return @"articulated dispatch";
    case 6u:
        return @"articulated q";
    case 7u:
        return @"point impulses";
    case 8u:
        return @"body pose output";
    case 9u:
        return @"point world output";
    case 10u:
        return @"diagnostic mass output";
    case 11u:
        return @"point Jacobian output";
    case 12u:
        return @"generalized impulse output";
    case 13u:
        return @"delta velocity output";
    case 14u:
        return @"articulated status output";
    case 15u:
        return @"FunctionBased programs";
    case kMillardDispatchBuffer:
        return @"Millard dispatch";
    case kMillardMusclesBuffer:
        return @"Millard muscles";
    case kMillardStatesBuffer:
        return @"Millard states";
    case kMillardPathPointsBuffer:
        return @"Millard path points";
    case kMillardCurvesBuffer:
        return @"Millard source curves";
    case kMillardWrapsBuffer:
        return @"Millard cylinder wraps";
    case kMillardResultsBuffer:
        return @"Millard results";
    case kMillardForcesBuffer:
        return @"Millard generalized forces";
    default:
        return @"articulated buffer";
    }
}

MetalArticulatedOperatorDiagnostics initializeContext(
    detail::MetalArticulatedOperatorContextState& context,
    MetalArticulatedOperatorDiagnostics diagnostics
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
            MetalArticulatedOperatorHostStatus::
                metallibUnavailable,
            "no articulated-operator metallib path is available"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnsupported,
            "articulated operator requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnavailable,
            "failed to create a Metal command queue"
        );
    }
    queue.label = @"MetalRobo articulated operator queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
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
            MetalArticulatedOperatorHostStatus::
                metalLibraryFailure,
            "failed to load metallib: " +
                describeError(error)
        );
    }
    id<MTLFunction> function = [library
        newFunctionWithName:@"mr_articulated_operator"];
    if (function == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalLibraryFailure,
            "metallib does not contain the articulated operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                       error:&error];
    if (pipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalPipelineFailure,
            "failed to create articulated pipeline: " +
                describeError(error)
        );
    }
    if (pipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup ||
        pipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnsupported,
            "device cannot execute the articulated threadgroup"
        );
    }
    id<MTLFunction> millardFunction = [library
        newFunctionWithName:@"mr_millard_reference"];
    if (millardFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the Millard reference operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> millardPipeline = [device
        newComputePipelineStateWithFunction:millardFunction
                                       error:&error];
    if (millardPipeline == nil ||
        millardPipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create Millard reference pipeline: " +
                describeError(error)
        );
    }

    context.device = device;
    context.queue = queue;
    context.library = library;
    context.pipeline = pipeline;
    context.millardPipeline = millardPipeline;
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

MetalArticulatedOperatorDiagnostics ensureBufferArena(
    detail::MetalArticulatedOperatorContextState& context,
    const RequiredBuffers& requirements,
    MetalArticulatedOperatorDiagnostics diagnostics
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
                MetalArticulatedOperatorHostStatus::
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
                MetalArticulatedOperatorHostStatus::
                    arithmeticOverflow,
                "persistent Metal arena byte-count overflow"
            );
        }
    }
    const std::uint64_t recommendedWorkingSet =
        context.device.recommendedMaxWorkingSetSize;
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        // Geometric slack is optional. Retry at the smallest safe retained
        // capacities before rejecting a batch near the working-set budget.
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
                    MetalArticulatedOperatorHostStatus::
                        arithmeticOverflow,
                    "persistent Metal arena byte-count overflow"
                );
            }
        }
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalBufferFailure,
            "persistent articulated buffer arena exceeds "
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
                MetalArticulatedOperatorHostStatus::
                    metalBufferFailure,
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
    detail::MetalArticulatedOperatorContextState& context,
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorLayout& layout,
    const RequiredBuffers& requirements
) {
    MRJointDescriptorGPU emptyJoint{};
    MRArticulatedPointImpulseGPU emptyPoint{};
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
        input.points.empty()
            ? static_cast<const void*>(&emptyPoint)
            : static_cast<const void*>(input.points.data()),
    };
    for (std::size_t index = 0u; index < sources.size(); ++index) {
        copyToBuffer(
            context.buffers[index],
            sources[index],
            requirements.entries[index]
        );
    }
    for (std::size_t index = sources.size();
         index < 15u;
         ++index) {
        std::memset(
            context.buffers[index].contents,
            0,
            requirements.entries[index].allocationBytes
        );
    }
    std::vector<MROpenSimSpatialTransformGPU> packedPrograms;
    const bool packed = packFunctionPrograms(model, packedPrograms);
    // validateAndBuildLayout() has already validated this exact immutable
    // stream. Retaining the defensive branch makes a future model mutation
    // between validation and upload fail closed rather than dispatch zeros.
    if (!packed) {
        std::memset(
            context.buffers[15u].contents,
            0,
            requirements.entries[15u].allocationBytes
        );
        return;
    }
    copyToBuffer(
        context.buffers[15u],
        packedPrograms.data(),
        requirements.entries[15u]
    );

    MRMillardReferenceDispatchGPU millardDispatch{};
    if (input.millard.enabled()) {
        const MRArticulationGPU& articulation =
            model.articulations[input.articulationIndex];
        millardDispatch.abiVersion = MR_MILLARD_REFERENCE_GPU_ABI_VERSION;
        millardDispatch.muscleCount = static_cast<mr_u32>(
            input.millard.muscles.size()
        );
        millardDispatch.pathPointCount = static_cast<mr_u32>(
            input.millard.pathPoints.size()
        );
        millardDispatch.wrapCount = static_cast<mr_u32>(
            input.millard.cylinderWraps.size()
        );
        millardDispatch.environmentCount = static_cast<mr_u32>(
            input.environmentCount
        );
        millardDispatch.dofCount = articulation.nv;
        millardDispatch.pointWorldStride = layout.dispatch.pointWorldStride;
        millardDispatch.pointJacobianStride =
            layout.dispatch.pointJacobianStride;
        millardDispatch.bodyPoseStride = layout.dispatch.bodyPoseStride;
        millardDispatch.articulationFirstBody = articulation.firstBody;
    }
    copyToBuffer(
        context.buffers[kMillardDispatchBuffer],
        &millardDispatch,
        requirements.entries[kMillardDispatchBuffer]
    );
    const auto uploadMillard = [&](const std::size_t index, const void* source) {
        copyToBuffer(
            context.buffers[index],
            source,
            requirements.entries[index]
        );
    };
    uploadMillard(
        kMillardMusclesBuffer,
        input.millard.muscles.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.muscles.data())
    );
    uploadMillard(
        kMillardStatesBuffer,
        input.millard.states.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.states.data())
    );
    uploadMillard(
        kMillardPathPointsBuffer,
        input.millard.pathPoints.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.pathPoints.data())
    );
    uploadMillard(
        kMillardCurvesBuffer,
        input.millard.curves.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.curves.data())
    );
    uploadMillard(
        kMillardWrapsBuffer,
        input.millard.cylinderWraps.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.cylinderWraps.data())
    );
    std::memset(
        context.buffers[kMillardResultsBuffer].contents,
        0,
        requirements.entries[kMillardResultsBuffer].allocationBytes
    );
    std::memset(
        context.buffers[kMillardForcesBuffer].contents,
        0,
        requirements.entries[kMillardForcesBuffer].allocationBytes
    );
}

// The standalone compatibility entry point still uses an isolated arena so
// that its historical lifetime and transaction semantics remain unchanged.
// Reusable callers should use MetalArticulatedOperatorContext.
id<MTLBuffer> makeInputBuffer(
    id<MTLDevice> device,
    const void* source,
    const BufferRequirement& requirement,
    NSString* label
) {
    id<MTLBuffer> result = [device
        newBufferWithLength:static_cast<NSUInteger>(
            requirement.allocationBytes
        )
                   options:MTLResourceStorageModeShared];
    if (result != nil && result.contents != nullptr) {
        copyToBuffer(result, source, requirement);
    }
    result.label = label;
    return result;
}

id<MTLBuffer> makeOutputBuffer(
    id<MTLDevice> device,
    const BufferRequirement& requirement,
    NSString* label
) {
    id<MTLBuffer> result = [device
        newBufferWithLength:static_cast<NSUInteger>(
            requirement.allocationBytes
        )
                   options:MTLResourceStorageModeShared];
    if (result != nil && result.contents != nullptr) {
        std::memset(
            result.contents,
            0,
            requirement.allocationBytes
        );
    }
    result.label = label;
    return result;
}

bool validBuffer(
    id<MTLBuffer> buffer,
    const BufferRequirement& requirement
) {
    return buffer != nil &&
        buffer.contents != nullptr &&
        buffer.length >= requirement.allocationBytes;
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

bool finitePayload(
    const MetalArticulatedOperatorResult& result
) {
    return
        std::all_of(
            result.bodyPoses.begin(),
            result.bodyPoses.end(),
            [](const MRArticulatedBodyPoseGPU& pose) {
                return finite(pose.position) &&
                    finite(pose.orientation);
            }
        ) &&
        std::all_of(
            result.pointWorld.begin(),
            result.pointWorld.end(),
            [](const MRArticulatedPointWorldGPU& point) {
                return finite(point.position);
            }
        ) &&
        std::all_of(
            result.diagnosticMassMatrix.begin(),
            result.diagnosticMassMatrix.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.pointJacobians.begin(),
            result.pointJacobians.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.generalizedImpulse.begin(),
            result.generalizedImpulse.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.deltaVelocity.begin(),
            result.deltaVelocity.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.millardResults.begin(),
            result.millardResults.end(),
            [](const MRMillardMuscleResultGPU& value) {
                return finite(value.pathFiberTendonResidual);
            }
        ) &&
        std::all_of(
            result.millardGeneralizedForces.begin(),
            result.millardGeneralizedForces.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        );
}

} // namespace

MetalArticulatedOperatorSubmission::
    MetalArticulatedOperatorSubmission() noexcept = default;

MetalArticulatedOperatorSubmission::
    ~MetalArticulatedOperatorSubmission() = default;

MetalArticulatedOperatorSubmission::
    MetalArticulatedOperatorSubmission(
        MetalArticulatedOperatorSubmission&& other
    ) noexcept = default;

MetalArticulatedOperatorSubmission&
MetalArticulatedOperatorSubmission::operator=(
    MetalArticulatedOperatorSubmission&& other
) noexcept = default;

bool MetalArticulatedOperatorSubmission::valid() const noexcept {
    return state_ != nullptr;
}

MetalArticulatedOperatorDiagnostics
MetalArticulatedOperatorSubmission::wait(
    MetalArticulatedOperatorResult& result
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalArticulatedOperatorHostStatus::
                metalCommandFailure,
            "submission is empty or has already been consumed"
        );
    }

    std::unique_ptr<
        detail::MetalArticulatedOperatorSubmissionState
    > pending = std::move(state_);
    MetalArticulatedOperatorDiagnostics diagnostics =
        pending->diagnostics;
    try {
        MetalArticulatedOperatorResult staged{};
        @autoreleasepool {
            [pending->commandBuffer waitUntilCompleted];
            const auto end =
                std::chrono::steady_clock::now();
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - pending->start
                ).count();
            if (pending->commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "Metal articulated command failed: " +
                        describeError(
                            pending->commandBuffer.error
                        )
                );
            }

            const MetalArticulatedOperatorLayout& layout =
                diagnostics.layout;
            staged.layout = layout;
            staged.bodyPoses.resize(layout.bodyPoseElements);
            staged.pointWorld.resize(layout.pointWorldElements);
            staged.diagnosticMassMatrix.resize(
                layout.massMatrixElements
            );
            staged.pointJacobians.resize(
                layout.pointJacobianElements
            );
            staged.generalizedImpulse.resize(
                layout.generalizedElements
            );
            staged.deltaVelocity.resize(
                layout.generalizedElements
            );
            staged.statuses.resize(layout.statusElements);
            staged.millardResults.resize(layout.millardResultElements);
            staged.millardGeneralizedForces.resize(
                layout.millardGeneralizedForceElements
            );

            const auto& buffers = pending->context->buffers;
            copyOutput(staged.bodyPoses, buffers[8]);
            copyOutput(staged.pointWorld, buffers[9]);
            copyOutput(
                staged.diagnosticMassMatrix,
                buffers[10]
            );
            copyOutput(staged.pointJacobians, buffers[11]);
            copyOutput(
                staged.generalizedImpulse,
                buffers[12]
            );
            copyOutput(staged.deltaVelocity, buffers[13]);
            copyOutput(staged.statuses, buffers[14]);
            copyOutput(
                staged.millardResults,
                buffers[kMillardResultsBuffer]
            );
            copyOutput(
                staged.millardGeneralizedForces,
                buffers[kMillardForcesBuffer]
            );
        }

        for (std::size_t environment = 0u;
             environment < staged.statuses.size();
             ++environment) {
            const MRArticulatedOperatorStatusGPU& status =
                staged.statuses[environment];
            if (status.environment != environment ||
                status.articulationIndex !=
                    pending->articulationIndex ||
                status.code >
                    MR_ARTICULATED_OPERATOR_ACCURACY_FAILED) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        internalFailure,
                    "GPU returned a malformed articulated status record"
                );
            }
            if (status.code ==
                MR_ARTICULATED_OPERATOR_SUCCESS) {
                if (status.bodyCount !=
                        pending->articulation.bodyCount ||
                    status.nq != pending->articulation.nq ||
                    status.nv != pending->articulation.nv ||
                    status.pointCount != pending->pointCount ||
                    !finite(status.diagnostics)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            internalFailure,
                        "GPU success status has invalid dimensions or "
                        "diagnostics"
                    );
                }
                ++diagnostics.successfulEnvironmentCount;
            } else {
                if (diagnostics.failedEnvironmentCount == 0u) {
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(
                            environment
                        );
                    diagnostics.firstGPUStatusCode =
                        status.code;
                }
                ++diagnostics.failedEnvironmentCount;
            }
        }
        if (pending->hasMillardReference) {
            for (std::size_t index = 0u;
                 index < staged.millardResults.size();
                 ++index) {
                const MRMillardMuscleResultGPU& millard =
                    staged.millardResults[index];
                const std::size_t environment =
                    index / pending->millardMuscleCount;
                const std::size_t muscle =
                    index - environment * pending->millardMuscleCount;
                if (millard.status != MR_MILLARD_REFERENCE_SUCCESS ||
                    millard.environment != environment ||
                    millard.muscleIndex != muscle ||
                    !finite(millard.pathFiberTendonResidual)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected a Millard source-reference muscle"
                    );
                }
            }
        }
        if (!finitePayload(staged)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    internalFailure,
                "GPU batch contained non-finite typed payload"
            );
        }

        result = std::move(staged);
        diagnostics.published = true;
        if (diagnostics.failedEnvironmentCount != 0u) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    gpuEnvironmentFailure,
                "one or more GPU environments rejected execution"
            );
        }
        diagnostics.status =
            MetalArticulatedOperatorHostStatus::success;
        diagnostics.message.clear();
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalBufferFailure,
            "host allocation failed while publishing Metal results"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedOperatorContext::
    MetalArticulatedOperatorContext(
        MetalArticulatedOperatorConfig config
    )
    : state_(std::make_shared<
          detail::MetalArticulatedOperatorContextState
      >(std::move(config))) {}

MetalArticulatedOperatorContext::
    ~MetalArticulatedOperatorContext() = default;

MetalArticulatedOperatorContext::
    MetalArticulatedOperatorContext(
        MetalArticulatedOperatorContext&& other
    ) noexcept = default;

MetalArticulatedOperatorContext&
MetalArticulatedOperatorContext::operator=(
    MetalArticulatedOperatorContext&& other
) noexcept = default;

MetalArticulatedOperatorDiagnostics
MetalArticulatedOperatorContext::submit(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorSubmission& submission
) {
    MetalArticulatedOperatorDiagnostics diagnostics{};
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                internalFailure,
            "operator context was moved from"
        );
    }
    if (submission.valid()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::contextBusy,
            "submission output already owns an in-flight batch"
        );
    }

    RequiredBuffers requirements{};
    try {
        diagnostics = validateAndBuildLayout(
            model,
            input,
            state_->config,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        const std::lock_guard lock(state_->mutex);
        if (state_->inFlight) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::contextBusy,
                "operator context already has an in-flight batch"
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
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo persistent articulated operator";
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal compute encoder"
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
            const MRArticulationGPU& articulation =
                model.articulations[input.articulationIndex];
            [encoder
                setThreadgroupMemoryLength:
                    detail::articulatedOperatorThreadgroupBytes(
                        articulation.bodyCount,
                        articulation.nv
                    )
                atIndex:0u];
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

            if (input.millard.enabled()) {
                id<MTLComputeCommandEncoder> millardEncoder =
                    [commandBuffer computeCommandEncoder];
                if (millardEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create Millard reference encoder"
                    );
                }
                [millardEncoder setComputePipelineState:state_->millardPipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [millardEncoder
                        setBuffer:state_->buffers[index]
                           offset:0u
                          atIndex:index];
                }
                const std::size_t threadCount =
                    diagnostics.layout.millardResultElements;
                [millardEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (threadCount + kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [millardEncoder endEncoding];
            }

            auto pending = std::make_unique<
                detail::MetalArticulatedOperatorSubmissionState
            >();
            diagnostics.dispatched = true;
            pending->context = state_;
            pending->commandBuffer = commandBuffer;
            pending->diagnostics = diagnostics;
            pending->articulation =
                model.articulations[input.articulationIndex];
            pending->articulationIndex =
                input.articulationIndex;
            pending->pointCount = input.pointCount;
            pending->hasMillardReference = input.millard.enabled();
            pending->millardMuscleCount = input.millard.muscles.size();
            pending->start =
                std::chrono::steady_clock::now();
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
            MetalArticulatedOperatorHostStatus::
                metalBufferFailure,
            "host allocation failed while preparing Metal submission"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedOperatorDiagnostics
MetalArticulatedOperatorContext::run(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorResult& result
) {
    MetalArticulatedOperatorSubmission submission;
    MetalArticulatedOperatorDiagnostics diagnostics = submit(
        model,
        input,
        submission
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return submission.wait(result);
}

MetalArticulatedOperatorContextStats
MetalArticulatedOperatorContext::stats() const noexcept {
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

MetalArticulatedOperatorDiagnostics runMetalArticulatedOperator(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorResult& result,
    const MetalArticulatedOperatorConfig& config
) {
    RequiredBuffers requirements{};
    MetalArticulatedOperatorDiagnostics diagnostics{};
    try {
        diagnostics = validateAndBuildLayout(
            model,
            input,
            config,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        const MetalArticulatedOperatorLayout layout =
            diagnostics.layout;

        MetalArticulatedOperatorResult staged{};
        MRJointDescriptorGPU emptyJoint{};
        MRArticulatedPointImpulseGPU emptyPoint{};

        @autoreleasepool {
            std::string metallibPath = config.metallibPath;
            if (metallibPath.empty()) {
                metallibPath = defaultMetallibPath();
            }
            if (metallibPath.empty()) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metallibUnavailable,
                    "no articulated-operator metallib path is available"
                );
            }

            id<MTLDevice> device =
                MTLCreateSystemDefaultDevice();
            if (device == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnavailable,
                    "no Metal-capable device is available"
                );
            }
            diagnostics.deviceName = nsString(device.name);
            if (!device.hasUnifiedMemory) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnsupported,
                    "articulated operator requires unified-memory Metal"
                );
            }
            for (const BufferRequirement& requirement :
                 requirements.entries) {
                if (requirement.allocationBytes >
                    device.maxBufferLength) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            metalBufferFailure,
                        std::string(requirement.label) +
                            " exceeds device.maxBufferLength"
                    );
                }
            }
            const std::uint64_t recommendedWorkingSet =
                device.recommendedMaxWorkingSetSize;
            if (recommendedWorkingSet != 0u &&
                static_cast<std::uint64_t>(
                    layout.totalAllocatedBytes
                ) > recommendedWorkingSet) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalBufferFailure,
                    "aggregate articulated buffers exceed "
                    "device.recommendedMaxWorkingSetSize"
                );
            }

            id<MTLCommandQueue> queue =
                [device newCommandQueue];
            if (queue == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnavailable,
                    "failed to create a Metal command queue"
                );
            }
            queue.label =
                @"MetalRobo articulated operator queue";

            NSString* path = [NSString
                stringWithUTF8String:metallibPath.c_str()];
            if (path == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
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
                    MetalArticulatedOperatorHostStatus::
                        metalLibraryFailure,
                    "failed to load metallib: " +
                        describeError(error)
                );
            }
            id<MTLFunction> function = [library
                newFunctionWithName:@"mr_articulated_operator"];
            if (function == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalLibraryFailure,
                    "metallib does not contain the articulated operator"
                );
            }
            error = nil;
            id<MTLComputePipelineState> pipeline = [device
                newComputePipelineStateWithFunction:function
                                               error:&error];
            if (pipeline == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalPipelineFailure,
                    "failed to create articulated pipeline: " +
                        describeError(error)
                );
            }
            if (pipeline.maxTotalThreadsPerThreadgroup <
                    kThreadsPerThreadgroup ||
                pipeline.staticThreadgroupMemoryLength >
                    device.maxThreadgroupMemoryLength) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnsupported,
                        "device cannot execute the articulated threadgroup"
                );
            }

            id<MTLComputePipelineState> millardPipeline = nil;
            if (input.millard.enabled()) {
                id<MTLFunction> millardFunction = [library
                    newFunctionWithName:@"mr_millard_reference"];
                if (millardFunction == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalLibraryFailure,
                        "metallib does not contain the Millard reference operator"
                    );
                }
                error = nil;
                millardPipeline = [device
                    newComputePipelineStateWithFunction:millardFunction
                                                   error:&error];
                if (millardPipeline == nil ||
                    millardPipeline.maxTotalThreadsPerThreadgroup <
                        kThreadsPerThreadgroup) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalPipelineFailure,
                        "failed to create Millard reference pipeline: " +
                            describeError(error)
                    );
                }
            }

            id<MTLBuffer> buffers[kRawBufferCount] = {};
            buffers[0] = makeInputBuffer(
                device,
                &model.world,
                requirements.entries[0],
                @"articulated world"
            );
            buffers[1] = makeInputBuffer(
                device,
                model.articulations.data(),
                requirements.entries[1],
                @"articulation descriptors"
            );
            buffers[2] = makeInputBuffer(
                device,
                model.joints.empty()
                    ? static_cast<const void*>(&emptyJoint)
                    : static_cast<const void*>(
                          model.joints.data()
                      ),
                requirements.entries[2],
                @"joint descriptors"
            );
            buffers[3] = makeInputBuffer(
                device,
                model.dofs.data(),
                requirements.entries[3],
                @"DoF properties"
            );
            buffers[4] = makeInputBuffer(
                device,
                model.bodies.data(),
                requirements.entries[4],
                @"body properties"
            );
            buffers[5] = makeInputBuffer(
                device,
                &layout.dispatch,
                requirements.entries[5],
                @"articulated dispatch"
            );
            buffers[6] = makeInputBuffer(
                device,
                input.q.data(),
                requirements.entries[6],
                @"articulated q"
            );
            buffers[7] = makeInputBuffer(
                device,
                input.points.empty()
                    ? static_cast<const void*>(&emptyPoint)
                    : static_cast<const void*>(input.points.data()),
                requirements.entries[7],
                @"point impulses"
            );
            buffers[8] = makeOutputBuffer(
                device,
                requirements.entries[8],
                @"body pose output"
            );
            buffers[9] = makeOutputBuffer(
                device,
                requirements.entries[9],
                @"point world output"
            );
            buffers[10] = makeOutputBuffer(
                device,
                requirements.entries[10],
                @"diagnostic mass output"
            );
            buffers[11] = makeOutputBuffer(
                device,
                requirements.entries[11],
                @"point Jacobian output"
            );
            buffers[12] = makeOutputBuffer(
                device,
                requirements.entries[12],
                @"generalized impulse output"
            );
            buffers[13] = makeOutputBuffer(
                device,
                requirements.entries[13],
                @"delta velocity output"
            );
            buffers[14] = makeOutputBuffer(
                device,
                requirements.entries[14],
                @"articulated status output"
            );
            std::vector<MROpenSimSpatialTransformGPU> packedPrograms;
            if (!packFunctionPrograms(model, packedPrograms)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::invalidModel,
                    "FunctionBased program packing failed before Metal allocation"
                );
            }
            buffers[15] = makeInputBuffer(
                device,
                packedPrograms.data(),
                requirements.entries[15],
                @"FunctionBased programs"
            );
            MRMillardReferenceDispatchGPU millardDispatch{};
            if (input.millard.enabled()) {
                const MRArticulationGPU& millardArticulation =
                    model.articulations[input.articulationIndex];
                millardDispatch.abiVersion =
                    MR_MILLARD_REFERENCE_GPU_ABI_VERSION;
                millardDispatch.muscleCount = static_cast<mr_u32>(
                    input.millard.muscles.size()
                );
                millardDispatch.pathPointCount = static_cast<mr_u32>(
                    input.millard.pathPoints.size()
                );
                millardDispatch.wrapCount = static_cast<mr_u32>(
                    input.millard.cylinderWraps.size()
                );
                millardDispatch.environmentCount = static_cast<mr_u32>(
                    input.environmentCount
                );
                millardDispatch.dofCount = millardArticulation.nv;
                millardDispatch.pointWorldStride =
                    layout.dispatch.pointWorldStride;
                millardDispatch.pointJacobianStride =
                    layout.dispatch.pointJacobianStride;
                millardDispatch.bodyPoseStride =
                    layout.dispatch.bodyPoseStride;
                millardDispatch.articulationFirstBody =
                    millardArticulation.firstBody;
            }
            buffers[kMillardDispatchBuffer] = makeInputBuffer(
                device,
                &millardDispatch,
                requirements.entries[kMillardDispatchBuffer],
                @"Millard dispatch"
            );
            buffers[kMillardMusclesBuffer] = makeInputBuffer(
                device,
                input.millard.muscles.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.muscles.data()),
                requirements.entries[kMillardMusclesBuffer],
                @"Millard muscles"
            );
            buffers[kMillardStatesBuffer] = makeInputBuffer(
                device,
                input.millard.states.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.states.data()),
                requirements.entries[kMillardStatesBuffer],
                @"Millard states"
            );
            buffers[kMillardPathPointsBuffer] = makeInputBuffer(
                device,
                input.millard.pathPoints.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.pathPoints.data()),
                requirements.entries[kMillardPathPointsBuffer],
                @"Millard path points"
            );
            buffers[kMillardCurvesBuffer] = makeInputBuffer(
                device,
                input.millard.curves.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.curves.data()),
                requirements.entries[kMillardCurvesBuffer],
                @"Millard source curves"
            );
            buffers[kMillardWrapsBuffer] = makeInputBuffer(
                device,
                input.millard.cylinderWraps.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.cylinderWraps.data()),
                requirements.entries[kMillardWrapsBuffer],
                @"Millard cylinder wraps"
            );
            buffers[kMillardResultsBuffer] = makeOutputBuffer(
                device,
                requirements.entries[kMillardResultsBuffer],
                @"Millard results"
            );
            buffers[kMillardForcesBuffer] = makeOutputBuffer(
                device,
                requirements.entries[kMillardForcesBuffer],
                @"Millard generalized forces"
            );

            for (std::size_t index = 0u;
                 index < kRawBufferCount;
                 ++index) {
                if (!validBuffer(
                        buffers[index],
                        requirements.entries[index]
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            metalBufferFailure,
                        std::string("Metal buffer allocation/length "
                                    "check failed for ") +
                            requirements.entries[index].label
                    );
                }
            }

            id<MTLCommandBuffer> commandBuffer =
                [queue commandBuffer];
            if (commandBuffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo articulated operator";
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal compute encoder"
                );
            }
            [encoder setComputePipelineState:pipeline];
            for (NSUInteger index = 0u;
                 index < kRawBufferCount;
                 ++index) {
                [encoder
                    setBuffer:buffers[index]
                       offset:0u
                      atIndex:index];
            }
            const MRArticulationGPU& articulation =
                model.articulations[input.articulationIndex];
            [encoder
                setThreadgroupMemoryLength:
                    detail::articulatedOperatorThreadgroupBytes(
                        articulation.bodyCount,
                        articulation.nv
                    )
                atIndex:0u];
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

            if (input.millard.enabled()) {
                id<MTLComputeCommandEncoder> millardEncoder =
                    [commandBuffer computeCommandEncoder];
                if (millardEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create Millard reference encoder"
                    );
                }
                [millardEncoder setComputePipelineState:millardPipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [millardEncoder setBuffer:buffers[index] offset:0u atIndex:index];
                }
                const std::size_t threadCount = layout.millardResultElements;
                [millardEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (threadCount + kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [millardEncoder endEncoding];
            }

            const auto start =
                std::chrono::steady_clock::now();
            diagnostics.dispatched = true;
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            const auto end =
                std::chrono::steady_clock::now();
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();
            if (commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "Metal articulated command failed: " +
                        describeError(commandBuffer.error)
                );
            }

            // Allocate the host-side transactional snapshot only after the
            // device has accepted every individual and aggregate buffer and
            // the command has completed. Until this point result is untouched.
            staged.layout = layout;
            staged.bodyPoses.resize(layout.bodyPoseElements);
            staged.pointWorld.resize(layout.pointWorldElements);
            staged.diagnosticMassMatrix.resize(
                layout.massMatrixElements
            );
            staged.pointJacobians.resize(
                layout.pointJacobianElements
            );
            staged.generalizedImpulse.resize(
                layout.generalizedElements
            );
            staged.deltaVelocity.resize(
                layout.generalizedElements
            );
            staged.statuses.resize(layout.statusElements);
            staged.millardResults.resize(layout.millardResultElements);
            staged.millardGeneralizedForces.resize(
                layout.millardGeneralizedForceElements
            );
            copyOutput(staged.bodyPoses, buffers[8]);
            copyOutput(staged.pointWorld, buffers[9]);
            copyOutput(
                staged.diagnosticMassMatrix,
                buffers[10]
            );
            copyOutput(staged.pointJacobians, buffers[11]);
            copyOutput(
                staged.generalizedImpulse,
                buffers[12]
            );
            copyOutput(staged.deltaVelocity, buffers[13]);
            copyOutput(staged.statuses, buffers[14]);
            copyOutput(
                staged.millardResults,
                buffers[kMillardResultsBuffer]
            );
            copyOutput(
                staged.millardGeneralizedForces,
                buffers[kMillardForcesBuffer]
            );
        }

        for (std::size_t environment = 0u;
             environment < staged.statuses.size();
             ++environment) {
            const MRArticulatedOperatorStatusGPU& status =
                staged.statuses[environment];
            const MRArticulationGPU& articulation =
                model.articulations[input.articulationIndex];
            if (status.environment != environment ||
                status.articulationIndex !=
                    input.articulationIndex ||
                status.code >
                    MR_ARTICULATED_OPERATOR_ACCURACY_FAILED) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        internalFailure,
                    "GPU returned a malformed articulated status record"
                );
            }
            if (status.code ==
                MR_ARTICULATED_OPERATOR_SUCCESS) {
                if (status.bodyCount !=
                        articulation.bodyCount ||
                    status.nq != articulation.nq ||
                    status.nv != articulation.nv ||
                    status.pointCount != input.pointCount ||
                    !finite(status.diagnostics)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            internalFailure,
                        "GPU success status has invalid dimensions or "
                        "diagnostics"
                    );
                }
                ++diagnostics.successfulEnvironmentCount;
            } else {
                if (diagnostics.failedEnvironmentCount == 0u) {
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(environment);
                    diagnostics.firstGPUStatusCode = status.code;
                }
                ++diagnostics.failedEnvironmentCount;
            }
        }
        if (input.millard.enabled()) {
            for (std::size_t index = 0u;
                 index < staged.millardResults.size();
                 ++index) {
                const MRMillardMuscleResultGPU& millard =
                    staged.millardResults[index];
                const std::size_t environment =
                    index / input.millard.muscles.size();
                const std::size_t muscle =
                    index - environment * input.millard.muscles.size();
                if (millard.status != MR_MILLARD_REFERENCE_SUCCESS ||
                    millard.environment != environment ||
                    millard.muscleIndex != muscle ||
                    !finite(millard.pathFiberTendonResidual)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected a Millard source-reference muscle"
                    );
                }
            }
        }
        if (!finitePayload(staged)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    internalFailure,
                "GPU batch contained non-finite typed payload"
            );
        }

        result = std::move(staged);
        diagnostics.published = true;
        if (diagnostics.failedEnvironmentCount != 0u) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    gpuEnvironmentFailure,
                "one or more GPU environments rejected execution"
            );
        }
        diagnostics.status =
            MetalArticulatedOperatorHostStatus::success;
        diagnostics.message.clear();
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalBufferFailure,
            "host allocation failed while staging Metal buffers"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::internalFailure,
            exception.what()
        );
    }
}

const char* metalArticulatedOperatorHostStatusName(
    const MetalArticulatedOperatorHostStatus status
) noexcept {
    switch (status) {
    case MetalArticulatedOperatorHostStatus::success:
        return "success";
    case MetalArticulatedOperatorHostStatus::invalidModel:
        return "invalid_model";
    case MetalArticulatedOperatorHostStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalArticulatedOperatorHostStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalArticulatedOperatorHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalArticulatedOperatorHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalArticulatedOperatorHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalArticulatedOperatorHostStatus::invalidPointQuery:
        return "invalid_point_query";
    case MetalArticulatedOperatorHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalArticulatedOperatorHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalArticulatedOperatorHostStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalArticulatedOperatorHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalArticulatedOperatorHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalArticulatedOperatorHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalArticulatedOperatorHostStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalArticulatedOperatorHostStatus::internalFailure:
        return "internal_failure";
    case MetalArticulatedOperatorHostStatus::contextBusy:
        return "context_busy";
    }
    return "unknown";
}

} // namespace metalrobo
