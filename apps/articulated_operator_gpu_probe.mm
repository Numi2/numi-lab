#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace {

constexpr std::uint8_t kSentinel = 0xa5u;

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

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
    return description.empty() ? nsString(error.description) : description;
}

template <typename T>
id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const T* data,
    const std::size_t count,
    NSString* label
) {
    require(data != nullptr && count != 0u, "empty Metal buffer");
    require(
        count <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer byte count overflow"
    );
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:data
                    length:static_cast<NSUInteger>(count * sizeof(T))
                   options:MTLResourceStorageModeShared];
    require(
        buffer != nil,
        "failed to allocate Metal buffer '" + nsString(label) + "'"
    );
    buffer.label = label;
    return buffer;
}

template <typename T>
std::vector<T> sentinelVector(const std::size_t count) {
    std::vector<T> result(count);
    std::memset(result.data(), kSentinel, result.size() * sizeof(T));
    return result;
}

template <typename T>
bool allSentinel(const std::vector<T>& values) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(values.data());
    return std::all_of(
        bytes,
        bytes + values.size() * sizeof(T),
        [](const std::uint8_t value) {
            return value == kSentinel;
        }
    );
}

struct MetalResult {
    MRArticulatedOperatorDispatchGPU dispatch{};
    std::vector<MRArticulatedBodyPoseGPU> bodyPoses;
    std::vector<MRArticulatedPointWorldGPU> pointWorld;
    std::vector<float> massMatrix;
    std::vector<float> pointJacobians;
    std::vector<float> generalizedImpulse;
    std::vector<float> deltaVelocity;
    std::vector<MRArticulatedOperatorStatusGPU> statuses;
    std::string deviceName;
    double elapsedMilliseconds = 0.0;

    [[nodiscard]] bool payloadUntouched() const {
        return allSentinel(bodyPoses) &&
            allSentinel(pointWorld) &&
            allSentinel(massMatrix) &&
            allSentinel(pointJacobians) &&
            allSentinel(generalizedImpulse) &&
            allSentinel(deltaVelocity);
    }
};

MetalResult runMetal(
    const metalrobo::EngineModel& model,
    const std::vector<std::vector<float>>& environmentsQ,
    const std::vector<
        std::vector<MRArticulatedPointImpulseGPU>
    >& environmentPoints,
    const mr_u32 dispatchReserved0 = 0u,
    const bool writeDiagnosticMass = true
) {
    @autoreleasepool {
        require(
            model.articulations.size() == 1u,
            "probe expects one articulation"
        );
        require(
            !environmentsQ.empty() &&
                environmentPoints.size() == environmentsQ.size(),
            "invalid environment batch"
        );
        const MRArticulationGPU& articulation =
            model.articulations[0];
        const std::size_t environmentCount = environmentsQ.size();
        const std::size_t pointCount = environmentPoints[0].size();
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            require(
                environmentsQ[environment].size() == articulation.nq,
                "q environment has wrong size"
            );
            require(
                environmentPoints[environment].size() == pointCount,
                "point environment has wrong size"
            );
        }

        MRArticulatedOperatorDispatchGPU dispatch{};
        dispatch.articulationIndex = 0u;
        dispatch.environmentCount =
            static_cast<mr_u32>(environmentCount);
        dispatch.pointCount = static_cast<mr_u32>(pointCount);
        dispatch.flags = writeDiagnosticMass
            ? MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS
            : 0u;
        dispatch.qStride = articulation.nq;
        dispatch.pointStride = dispatch.pointCount;
        dispatch.bodyPoseStride = articulation.bodyCount;
        dispatch.pointWorldStride = dispatch.pointCount;
        dispatch.massMatrixStride =
            articulation.nv * articulation.nv;
        dispatch.pointJacobianStride =
            dispatch.pointCount * 3u * articulation.nv;
        dispatch.generalizedStride = articulation.nv;
        dispatch.reserved0 = dispatchReserved0;

        std::vector<float> flatQ(
            environmentCount * dispatch.qStride
        );
        std::vector<MRArticulatedPointImpulseGPU> flatPoints(
            environmentCount * dispatch.pointStride
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                environmentsQ[environment].begin(),
                environmentsQ[environment].end(),
                flatQ.begin() +
                    environment * dispatch.qStride
            );
            std::copy(
                environmentPoints[environment].begin(),
                environmentPoints[environment].end(),
                flatPoints.begin() +
                    environment * dispatch.pointStride
            );
        }

        MetalResult result;
        result.dispatch = dispatch;
        result.bodyPoses =
            sentinelVector<MRArticulatedBodyPoseGPU>(
                environmentCount * dispatch.bodyPoseStride
            );
        result.pointWorld =
            sentinelVector<MRArticulatedPointWorldGPU>(
                environmentCount * dispatch.pointWorldStride
            );
        result.massMatrix = sentinelVector<float>(
            environmentCount * dispatch.massMatrixStride
        );
        result.pointJacobians = sentinelVector<float>(
            environmentCount * dispatch.pointJacobianStride
        );
        result.generalizedImpulse = sentinelVector<float>(
            environmentCount * dispatch.generalizedStride
        );
        result.deltaVelocity = sentinelVector<float>(
            environmentCount * dispatch.generalizedStride
        );
        result.statuses =
            sentinelVector<MRArticulatedOperatorStatusGPU>(
                environmentCount
            );

        std::vector<MRJointDescriptorGPU> jointRecords = model.joints;
        if (jointRecords.empty()) {
            jointRecords.resize(1u);
        }
        require(
            model.dofs.size() == model.world.nv &&
                !model.dofs.empty(),
            "model has no canonical DoF records"
        );
        require(!model.bodies.empty(), "model has no bodies");

        const std::string metallibPath = METALROBO_DEFAULT_METALLIB;
        require(
            !metallibPath.empty(),
            "METALROBO_DEFAULT_METALLIB is empty"
        );
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal-capable device is available");
        const std::size_t threadgroupBytes =
            metalrobo::detail::articulatedOperatorThreadgroupBytes(
                articulation.bodyCount,
                articulation.nv
            );
        require(
            threadgroupBytes <= device.maxThreadgroupMemoryLength,
            "Metal device has insufficient threadgroup memory"
        );
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create Metal command queue");

        NSString* path =
            [NSString stringWithUTF8String:metallibPath.c_str()];
        require(path != nil, "metallib path is not valid UTF-8");
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                error:&error];
        require(
            library != nil,
            "failed to load metallib: " + describeError(error)
        );
        id<MTLFunction> function =
            [library newFunctionWithName:@"mr_articulated_operator"];
        require(
            function != nil,
            "metallib does not contain mr_articulated_operator"
        );
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        require(
            pipeline != nil,
            "failed to create articulated pipeline: " +
                describeError(error)
        );

        id<MTLBuffer> worldBuffer = makeSharedBuffer(
            device,
            &model.world,
            1u,
            @"articulated world"
        );
        id<MTLBuffer> articulationBuffer = makeSharedBuffer(
            device,
            model.articulations.data(),
            model.articulations.size(),
            @"articulation descriptors"
        );
        id<MTLBuffer> jointBuffer = makeSharedBuffer(
            device,
            jointRecords.data(),
            jointRecords.size(),
            @"joint descriptors"
        );
        id<MTLBuffer> dofBuffer = makeSharedBuffer(
            device,
            model.dofs.data(),
            model.dofs.size(),
            @"DoF properties"
        );
        id<MTLBuffer> bodyBuffer = makeSharedBuffer(
            device,
            model.bodies.data(),
            model.bodies.size(),
            @"body properties"
        );
        id<MTLBuffer> dispatchBuffer = makeSharedBuffer(
            device,
            &dispatch,
            1u,
            @"articulated dispatch"
        );
        id<MTLBuffer> qBuffer = makeSharedBuffer(
            device,
            flatQ.data(),
            flatQ.size(),
            @"articulated q"
        );
        id<MTLBuffer> pointBuffer = makeSharedBuffer(
            device,
            flatPoints.data(),
            flatPoints.size(),
            @"point impulses"
        );
        id<MTLBuffer> poseBuffer = makeSharedBuffer(
            device,
            result.bodyPoses.data(),
            result.bodyPoses.size(),
            @"body pose output"
        );
        id<MTLBuffer> pointWorldBuffer = makeSharedBuffer(
            device,
            result.pointWorld.data(),
            result.pointWorld.size(),
            @"point world output"
        );
        id<MTLBuffer> massBuffer = makeSharedBuffer(
            device,
            result.massMatrix.data(),
            result.massMatrix.size(),
            @"diagnostic mass output"
        );
        id<MTLBuffer> jacobianBuffer = makeSharedBuffer(
            device,
            result.pointJacobians.data(),
            result.pointJacobians.size(),
            @"point Jacobian output"
        );
        id<MTLBuffer> generalizedBuffer = makeSharedBuffer(
            device,
            result.generalizedImpulse.data(),
            result.generalizedImpulse.size(),
            @"generalized impulse output"
        );
        id<MTLBuffer> deltaBuffer = makeSharedBuffer(
            device,
            result.deltaVelocity.data(),
            result.deltaVelocity.size(),
            @"delta velocity output"
        );
        id<MTLBuffer> statusBuffer = makeSharedBuffer(
            device,
            result.statuses.data(),
            result.statuses.size(),
            @"articulated status"
        );

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        require(
            commandBuffer != nil && encoder != nil,
            "failed to create Metal command encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:worldBuffer offset:0 atIndex:0];
        [encoder setBuffer:articulationBuffer offset:0 atIndex:1];
        [encoder setBuffer:jointBuffer offset:0 atIndex:2];
        [encoder setBuffer:dofBuffer offset:0 atIndex:3];
        [encoder setBuffer:bodyBuffer offset:0 atIndex:4];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:5];
        [encoder setBuffer:qBuffer offset:0 atIndex:6];
        [encoder setBuffer:pointBuffer offset:0 atIndex:7];
        [encoder setBuffer:poseBuffer offset:0 atIndex:8];
        [encoder setBuffer:pointWorldBuffer offset:0 atIndex:9];
        [encoder setBuffer:massBuffer offset:0 atIndex:10];
        [encoder setBuffer:jacobianBuffer offset:0 atIndex:11];
        [encoder setBuffer:generalizedBuffer offset:0 atIndex:12];
        [encoder setBuffer:deltaBuffer offset:0 atIndex:13];
        [encoder setBuffer:statusBuffer offset:0 atIndex:14];
        [encoder
            setThreadgroupMemoryLength:threadgroupBytes
                              atIndex:0u];
        [encoder
            dispatchThreadgroups:MTLSizeMake(environmentCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder endEncoding];

        const auto start = std::chrono::steady_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();
        require(
            commandBuffer.status == MTLCommandBufferStatusCompleted,
            "articulated command failed: " +
                describeError(commandBuffer.error)
        );
        result.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(end - start)
                .count();

        std::memcpy(
            result.bodyPoses.data(),
            poseBuffer.contents,
            result.bodyPoses.size() *
                sizeof(MRArticulatedBodyPoseGPU)
        );
        std::memcpy(
            result.pointWorld.data(),
            pointWorldBuffer.contents,
            result.pointWorld.size() *
                sizeof(MRArticulatedPointWorldGPU)
        );
        std::memcpy(
            result.massMatrix.data(),
            massBuffer.contents,
            result.massMatrix.size() * sizeof(float)
        );
        std::memcpy(
            result.pointJacobians.data(),
            jacobianBuffer.contents,
            result.pointJacobians.size() * sizeof(float)
        );
        std::memcpy(
            result.generalizedImpulse.data(),
            generalizedBuffer.contents,
            result.generalizedImpulse.size() * sizeof(float)
        );
        std::memcpy(
            result.deltaVelocity.data(),
            deltaBuffer.contents,
            result.deltaVelocity.size() * sizeof(float)
        );
        std::memcpy(
            result.statuses.data(),
            statusBuffer.contents,
            result.statuses.size() *
                sizeof(MRArticulatedOperatorStatusGPU)
        );
        result.deviceName = nsString(device.name);
        return result;
    }
}

MRBodyPropertiesGPU makeBody(
    const std::uint32_t parent,
    const std::uint32_t inbound,
    const float mass,
    const std::array<float, 3> inertia
) {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = 0u;
    body.parentBody = parent;
    body.inboundJoint = inbound;
    body.motionType = MR_MOTION_DYNAMIC;
    body.massAndInverseMass = f4(mass, 1.0f / mass, 0.0f);
    body.inertiaRow0 = f4(inertia[0], 0.0f, 0.0f);
    body.inertiaRow1 = f4(0.0f, inertia[1], 0.0f);
    body.inertiaRow2 = f4(0.0f, 0.0f, inertia[2]);
    body.inverseInertiaRow0 =
        f4(1.0f / inertia[0], 0.0f, 0.0f);
    body.inverseInertiaRow1 =
        f4(0.0f, 1.0f / inertia[1], 0.0f);
    body.inverseInertiaRow2 =
        f4(0.0f, 0.0f, 1.0f / inertia[2]);
    body.dampingAndSpeedLimits =
        f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);
    return body;
}

metalrobo::EngineModel makeFixedPendulumModel() {
    metalrobo::EngineModel model;
    model.name = "fixed_pendulum_gpu_reference";
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = 2u;
    model.world.articulationCount = 1u;
    model.world.jointCount = 1u;
    model.world.nq = 1u;
    model.world.nv = 1u;
    model.world.pairCapacity = 1u;
    model.world.contactCapacity = 1u;
    model.world.constraintCapacity = 1u;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_REFERENCE_FP64;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep =
        f4(0.0f, 0.0f, -9.81f, 1.0f / 1000.0f);
    model.world.solverScales =
        f4(1.0e-8f, 1.0e-9f, 2.0f, 1.0e-4f);

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount = 2u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 1u;
    articulation.qOffset = 0u;
    articulation.nq = 1u;
    articulation.vOffset = 0u;
    articulation.nv = 1u;
    model.articulations.push_back(articulation);

    model.bodies.push_back(
        makeBody(
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            2.0f,
            {0.3f, 0.4f, 0.5f}
        )
    );
    model.bodies.push_back(
        makeBody(0u, 0u, 3.0f, {0.2f, 0.35f, 0.4f})
    );

    MRJointDescriptorGPU joint{};
    joint.parentBody = 0u;
    joint.childBody = 1u;
    joint.jointType = MR_JOINT_REVOLUTE;
    joint.qOffset = 0u;
    joint.nq = 1u;
    joint.vOffset = 0u;
    joint.nv = 1u;
    joint.axis0 = f4(0.0f, 1.0f, 0.0f);
    joint.parentAnchor = f4(0.0f, 0.0f, 0.0f);
    joint.childAnchor = f4(0.0f, 0.0f, -0.4f);
    joint.parentRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    joint.childRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    model.joints.push_back(joint);

    MRDofPropertiesGPU dof{};
    dof.articulationIndex = 0u;
    dof.jointIndex = 0u;
    dof.qIndex = 0u;
    dof.vIndex = 0u;
    dof.localDof = 0u;
    // Armature is valid even without an enabled actuator/drive.
    dof.drive = f4(0.0f, 0.0f, 0.025f, 0.0f);
    model.dofs.push_back(dof);

    model.defaultQ = {0.37f};
    model.defaultV = {0.0f};
    return model;
}

MRArticulatedPointImpulseGPU makePoint(
    const std::uint32_t body,
    const std::array<float, 3> local,
    const std::array<float, 3> impulse
) {
    MRArticulatedPointImpulseGPU result{};
    result.bodyIndex = body;
    result.localPoint = f4(local[0], local[1], local[2], 0.0f);
    result.worldImpulse =
        f4(impulse[0], impulse[1], impulse[2], 0.0f);
    return result;
}

struct CpuReference {
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies;
    std::vector<metalrobo::ArticulatedPointKinematics> points;
    std::vector<double> mass;
    std::vector<double> jacobian;
    std::vector<double> generalizedImpulse;
    std::vector<double> deltaVelocity;
    double solveResidual = 0.0;
};

std::vector<double> solvePositiveDefinite(
    const std::vector<double>& matrix,
    const std::vector<double>& rightHandSide
) {
    const std::size_t n = rightHandSide.size();
    require(matrix.size() == n * n, "invalid CPU solve dimensions");
    std::vector<double> factor(n * n, 0.0);
    for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * n + column];
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -=
                    factor[row * n + inner] *
                    factor[column * n + inner];
            }
            if (row == column) {
                require(
                    value > 1.0e-14 && std::isfinite(value),
                    "CPU mass factorization failed"
                );
                factor[row * n + row] = std::sqrt(value);
            } else {
                factor[row * n + column] =
                    value / factor[column * n + column];
            }
        }
    }
    std::vector<double> intermediate(n, 0.0);
    std::vector<double> solution(n, 0.0);
    for (std::size_t row = 0u; row < n; ++row) {
        double value = rightHandSide[row];
        for (std::size_t column = 0u; column < row; ++column) {
            value -=
                factor[row * n + column] * intermediate[column];
        }
        intermediate[row] = value / factor[row * n + row];
    }
    for (std::size_t reverse = 0u; reverse < n; ++reverse) {
        const std::size_t row = n - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u; column < n; ++column) {
            value -=
                factor[column * n + row] * solution[column];
        }
        solution[row] = value / factor[row * n + row];
    }
    return solution;
}

CpuReference buildCpuReference(
    const metalrobo::EngineModel& model,
    const std::vector<float>& qFloat,
    const std::vector<MRArticulatedPointImpulseGPU>& points
) {
    const MRArticulationGPU& articulation = model.articulations[0];
    std::vector<double> q(qFloat.begin(), qFloat.end());
    std::vector<double> zeroVelocity(articulation.nv, 0.0);
    CpuReference result;
    result.bodies.resize(articulation.bodyCount);
    auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            0u,
            q,
            zeroVelocity,
            result.bodies
        );
    require(
        diagnostics.succeeded(),
        "CPU body kinematics reference failed"
    );

    std::vector<metalrobo::ArticulatedPointQuery> queries(
        points.size()
    );
    for (std::size_t point = 0u; point < points.size(); ++point) {
        queries[point].bodyIndex = points[point].bodyIndex;
        queries[point].localPoint = {
            points[point].localPoint.x,
            points[point].localPoint.y,
            points[point].localPoint.z,
        };
    }
    result.points.resize(points.size());
    result.jacobian.resize(
        points.size() * 3u * articulation.nv
    );
    diagnostics =
        metalrobo::computeArticulatedPointJacobians(
            model,
            0u,
            q,
            zeroVelocity,
            queries,
            result.points,
            result.jacobian
        );
    require(
        diagnostics.succeeded(),
        "CPU point Jacobian reference failed"
    );

    result.mass.resize(articulation.nv * articulation.nv);
    diagnostics = metalrobo::computeArticulatedMassMatrix(
        model,
        0u,
        q,
        result.mass
    );
    require(
        diagnostics.succeeded(),
        "CPU mass matrix reference failed"
    );

    result.generalizedImpulse.assign(articulation.nv, 0.0);
    for (std::size_t point = 0u; point < points.size(); ++point) {
        const std::array<double, 3> impulse{
            points[point].worldImpulse.x,
            points[point].worldImpulse.y,
            points[point].worldImpulse.z,
        };
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                result.generalizedImpulse[dof] +=
                    result.jacobian[
                        (point * 3u + axis) * articulation.nv + dof
                    ] *
                    impulse[axis];
            }
        }
    }
    result.deltaVelocity = solvePositiveDefinite(
        result.mass,
        result.generalizedImpulse
    );
    for (std::size_t row = 0u; row < articulation.nv; ++row) {
        double action = 0.0;
        for (std::size_t column = 0u;
             column < articulation.nv;
             ++column) {
            action +=
                result.mass[row * articulation.nv + column] *
                result.deltaVelocity[column];
        }
        result.solveResidual = std::max(
            result.solveResidual,
            std::abs(action - result.generalizedImpulse[row])
        );
    }
    return result;
}

struct ParityMetrics {
    double pose = 0.0;
    double orientation = 0.0;
    double point = 0.0;
    double mass = 0.0;
    double massScaled = 0.0;
    double jacobian = 0.0;
    double generalizedImpulse = 0.0;
    double deltaVelocity = 0.0;
    double deltaVelocityScaled = 0.0;
    double gpuEquationResidual = 0.0;
    double statusRelativeResidual = 0.0;
};

void compareEnvironment(
    const metalrobo::EngineModel& model,
    const std::size_t environment,
    const CpuReference& cpu,
    const MetalResult& gpu,
    ParityMetrics& metrics
) {
    const MRArticulationGPU& articulation = model.articulations[0];
    const MRArticulatedOperatorStatusGPU& status =
        gpu.statuses[environment];
    require(
        status.code == MR_ARTICULATED_OPERATOR_SUCCESS &&
            status.environment == environment &&
            status.articulationIndex == 0u &&
            status.bodyCount == articulation.bodyCount &&
            status.nq == articulation.nq &&
            status.nv == articulation.nv &&
            status.pointCount == gpu.dispatch.pointCount,
        "Metal articulated status failed for environment " +
            std::to_string(environment) +
            " code=" + std::to_string(status.code) +
            " failingIndex=" + std::to_string(status.failingIndex)
    );
    require(
        std::isfinite(status.diagnostics.x) &&
            status.diagnostics.x > 0.0f &&
            std::isfinite(status.diagnostics.y) &&
            status.diagnostics.y >= status.diagnostics.x &&
            std::isfinite(status.diagnostics.z) &&
            std::isfinite(status.diagnostics.w),
        "Metal articulated diagnostics are invalid"
    );
    metrics.statusRelativeResidual = std::max(
        metrics.statusRelativeResidual,
        static_cast<double>(status.diagnostics.z)
    );

    const std::size_t poseBase =
        environment * gpu.dispatch.bodyPoseStride;
    for (std::size_t body = 0u;
         body < articulation.bodyCount;
         ++body) {
        const auto& gpuPose = gpu.bodyPoses[poseBase + body];
        const auto& cpuPose = cpu.bodies[body];
        const std::array<double, 3> gpuPosition{
            gpuPose.position.x,
            gpuPose.position.y,
            gpuPose.position.z,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            metrics.pose = std::max(
                metrics.pose,
                std::abs(
                    gpuPosition[axis] -
                    cpuPose.centerOfMassPosition[axis]
                )
            );
        }
        const std::array<double, 4> gpuQuaternion{
            gpuPose.orientation.x,
            gpuPose.orientation.y,
            gpuPose.orientation.z,
            gpuPose.orientation.w,
        };
        double dot = 0.0;
        for (std::size_t component = 0u; component < 4u; ++component) {
            dot += gpuQuaternion[component] *
                cpuPose.orientation[component];
        }
        const double sign = dot < 0.0 ? -1.0 : 1.0;
        for (std::size_t component = 0u; component < 4u; ++component) {
            metrics.orientation = std::max(
                metrics.orientation,
                std::abs(
                    sign * gpuQuaternion[component] -
                    cpuPose.orientation[component]
                )
            );
        }
    }

    const std::size_t pointBase =
        environment * gpu.dispatch.pointWorldStride;
    for (std::size_t point = 0u;
         point < gpu.dispatch.pointCount;
         ++point) {
        const auto& gpuPoint =
            gpu.pointWorld[pointBase + point].position;
        const std::array<double, 3> components{
            gpuPoint.x, gpuPoint.y, gpuPoint.z,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            metrics.point = std::max(
                metrics.point,
                std::abs(
                    components[axis] -
                    cpu.points[point].position[axis]
                )
            );
        }
    }

    const std::size_t massBase =
        environment * gpu.dispatch.massMatrixStride;
    for (std::size_t index = 0u; index < cpu.mass.size(); ++index) {
        const double error =
            std::abs(gpu.massMatrix[massBase + index] - cpu.mass[index]);
        metrics.mass = std::max(metrics.mass, error);
        metrics.massScaled = std::max(
            metrics.massScaled,
            error / (1.0 + std::abs(cpu.mass[index]))
        );
    }
    const std::size_t jacobianBase =
        environment * gpu.dispatch.pointJacobianStride;
    for (std::size_t index = 0u;
         index < cpu.jacobian.size();
         ++index) {
        metrics.jacobian = std::max(
            metrics.jacobian,
            std::abs(
                gpu.pointJacobians[jacobianBase + index] -
                cpu.jacobian[index]
            )
        );
    }

    const std::size_t generalizedBase =
        environment * gpu.dispatch.generalizedStride;
    for (std::size_t dof = 0u;
         dof < articulation.nv;
         ++dof) {
        metrics.generalizedImpulse = std::max(
            metrics.generalizedImpulse,
            std::abs(
                gpu.generalizedImpulse[generalizedBase + dof] -
                cpu.generalizedImpulse[dof]
            )
        );
        const double deltaError = std::abs(
            gpu.deltaVelocity[generalizedBase + dof] -
            cpu.deltaVelocity[dof]
        );
        metrics.deltaVelocity =
            std::max(metrics.deltaVelocity, deltaError);
        metrics.deltaVelocityScaled = std::max(
            metrics.deltaVelocityScaled,
            deltaError /
                (1.0 + std::abs(cpu.deltaVelocity[dof]))
        );
    }
    for (std::size_t row = 0u; row < articulation.nv; ++row) {
        double action = 0.0;
        for (std::size_t column = 0u;
             column < articulation.nv;
             ++column) {
            action +=
                cpu.mass[row * articulation.nv + column] *
                gpu.deltaVelocity[generalizedBase + column];
        }
        metrics.gpuEquationResidual = std::max(
            metrics.gpuEquationResidual,
            std::abs(action - cpu.generalizedImpulse[row])
        );
    }
}

void validateMetrics(
    const ParityMetrics& metrics,
    const std::string& label
) {
    require(
        metrics.pose < 3.0e-5 &&
            metrics.orientation < 3.0e-5 &&
            metrics.point < 4.0e-5 &&
            metrics.massScaled < 8.0e-4 &&
            metrics.jacobian < 5.0e-5 &&
            metrics.generalizedImpulse < 1.0e-3 &&
            metrics.deltaVelocityScaled < 8.0e-3 &&
            metrics.statusRelativeResidual <
                MR_ARTICULATED_OPERATOR_MAX_RELATIVE_RESIDUAL,
        label + " CPU FP64/Metal FP32 parity exceeded tolerance"
    );
}

std::vector<std::vector<float>> makeG1QBatch(
    const metalrobo::EngineModel& model
) {
    std::vector<std::vector<float>> result(
        3u,
        model.defaultQ
    );
    for (std::size_t environment = 0u;
         environment < result.size();
         ++environment) {
        auto& q = result[environment];
        q[0] += 0.13f * static_cast<float>(environment);
        q[1] -= 0.07f * static_cast<float>(environment);
        q[2] += 0.04f * static_cast<float>(environment);
        if (environment != 0u) {
            const float x = 0.035f * static_cast<float>(environment);
            const float y = -0.021f * static_cast<float>(environment);
            const float z = 0.017f * static_cast<float>(environment);
            q[3] = x;
            q[4] = y;
            q[5] = z;
            q[6] = std::sqrt(1.0f - x * x - y * y - z * z);
        }
        for (std::size_t joint = 7u; joint < q.size(); ++joint) {
            q[joint] +=
                0.045f *
                std::sin(
                    0.37f * static_cast<float>(
                        (joint - 6u) * (environment + 1u)
                    )
                );
        }
    }
    return result;
}

void printMetrics(
    const std::string& label,
    const ParityMetrics& metrics,
    const double milliseconds
) {
    std::cout
        << "  " << label
        << ": pose=" << metrics.pose
        << " orientation=" << metrics.orientation
        << " point=" << metrics.point
        << " mass=" << metrics.mass
        << " mass_scaled=" << metrics.massScaled
        << " J=" << metrics.jacobian
        << " JTp=" << metrics.generalizedImpulse
        << " dv=" << metrics.deltaVelocity
        << " dv_scaled=" << metrics.deltaVelocityScaled
        << " equation_residual=" << metrics.gpuEquationResidual
        << " status_backward_error="
        << metrics.statusRelativeResidual
        << " elapsed_ms=" << milliseconds
        << '\n';
}

} // namespace

int main() {
    try {
        std::cout << std::scientific << std::setprecision(6);

        const metalrobo::EngineModel freeModel =
            metalrobo::makeFreeSphereEngineModel();
        std::vector<std::vector<float>> freeQ{freeModel.defaultQ};
        freeQ[0][0] = 0.3f;
        freeQ[0][1] = -0.2f;
        freeQ[0][2] = 1.1f;
        freeQ[0][3] = 0.12f;
        freeQ[0][4] = -0.08f;
        freeQ[0][5] = 0.04f;
        freeQ[0][6] = std::sqrt(
            1.0f -
            freeQ[0][3] * freeQ[0][3] -
            freeQ[0][4] * freeQ[0][4] -
            freeQ[0][5] * freeQ[0][5]
        );
        std::vector<
            std::vector<MRArticulatedPointImpulseGPU>
        > freePoints{{
            makePoint(
                1u,
                {0.20f, -0.10f, 0.15f},
                {1.5f, -0.7f, 2.2f}
            ),
        }};
        const MetalResult freeGpu =
            runMetal(freeModel, freeQ, freePoints);
        const CpuReference freeCpu =
            buildCpuReference(freeModel, freeQ[0], freePoints[0]);
        ParityMetrics freeMetrics;
        compareEnvironment(
            freeModel,
            0u,
            freeCpu,
            freeGpu,
            freeMetrics
        );
        validateMetrics(freeMetrics, "floating analytic");

        const metalrobo::EngineModel fixedModel =
            makeFixedPendulumModel();
        std::string fixedReason;
        require(
            fixedModel.valid(&fixedReason),
            "fixed test model invalid: " + fixedReason
        );
        std::vector<std::vector<float>> fixedQ{
            fixedModel.defaultQ,
        };
        std::vector<
            std::vector<MRArticulatedPointImpulseGPU>
        > fixedPoints{{
            makePoint(
                1u,
                {0.18f, 0.07f, -0.12f},
                {0.6f, -0.3f, 1.1f}
            ),
        }};
        const MetalResult fixedGpu =
            runMetal(fixedModel, fixedQ, fixedPoints);
        const CpuReference fixedCpu =
            buildCpuReference(fixedModel, fixedQ[0], fixedPoints[0]);
        ParityMetrics fixedMetrics;
        compareEnvironment(
            fixedModel,
            0u,
            fixedCpu,
            fixedGpu,
            fixedMetrics
        );
        validateMetrics(fixedMetrics, "fixed pendulum");

        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        const auto g1Q = makeG1QBatch(g1);
        const auto& metadata = metalrobo::unitreeG1Metadata();
        std::vector<
            std::vector<MRArticulatedPointImpulseGPU>
        > g1Points(g1Q.size());
        for (std::size_t environment = 0u;
             environment < g1Points.size();
             ++environment) {
            const float scale =
                1.0f + 0.2f * static_cast<float>(environment);
            g1Points[environment] = {
                makePoint(
                    metadata.feet[0].bodyIndex,
                    {
                        metadata.feet[0].solePosition.x,
                        metadata.feet[0].solePosition.y,
                        metadata.feet[0].solePosition.z,
                    },
                    {1.2f * scale, -0.7f, 18.0f * scale}
                ),
                makePoint(
                    metadata.feet[1].bodyIndex,
                    {
                        metadata.feet[1].solePosition.x,
                        metadata.feet[1].solePosition.y,
                        metadata.feet[1].solePosition.z,
                    },
                    {-0.9f, 0.5f * scale, 16.0f}
                ),
                makePoint(
                    22u,
                    {0.03f, -0.02f, 0.01f},
                    {0.7f, -1.1f * scale, 0.4f}
                ),
                makePoint(
                    29u,
                    {-0.02f, 0.01f, 0.04f},
                    {-0.5f * scale, 0.8f, -0.3f}
                ),
            };
        }
        const MetalResult g1Gpu = runMetal(g1, g1Q, g1Points);
        ParityMetrics g1Metrics;
        for (std::size_t environment = 0u;
             environment < g1Q.size();
             ++environment) {
            const CpuReference cpu = buildCpuReference(
                g1,
                g1Q[environment],
                g1Points[environment]
            );
            compareEnvironment(
                g1,
                environment,
                cpu,
                g1Gpu,
                g1Metrics
            );
        }
        validateMetrics(g1Metrics, "actual G1 batch");

        const MetalResult g1Replay = runMetal(g1, g1Q, g1Points);
        require(
            std::memcmp(
                g1Gpu.bodyPoses.data(),
                g1Replay.bodyPoses.data(),
                g1Gpu.bodyPoses.size() *
                    sizeof(MRArticulatedBodyPoseGPU)
            ) == 0 &&
                std::memcmp(
                    g1Gpu.pointWorld.data(),
                    g1Replay.pointWorld.data(),
                    g1Gpu.pointWorld.size() *
                        sizeof(MRArticulatedPointWorldGPU)
                ) == 0 &&
                g1Gpu.massMatrix == g1Replay.massMatrix &&
                g1Gpu.pointJacobians == g1Replay.pointJacobians &&
                g1Gpu.generalizedImpulse ==
                    g1Replay.generalizedImpulse &&
                g1Gpu.deltaVelocity == g1Replay.deltaVelocity &&
                std::memcmp(
                    g1Gpu.statuses.data(),
                    g1Replay.statuses.data(),
                    g1Gpu.statuses.size() *
                        sizeof(MRArticulatedOperatorStatusGPU)
                ) == 0,
            "G1 Metal replay was not bitwise deterministic"
        );

        constexpr std::size_t throughputEnvironmentCount = 1024u;
        std::vector<std::vector<float>> throughputQ(
            throughputEnvironmentCount
        );
        std::vector<
            std::vector<MRArticulatedPointImpulseGPU>
        > throughputPoints(throughputEnvironmentCount);
        for (std::size_t environment = 0u;
             environment < throughputEnvironmentCount;
             ++environment) {
            const std::size_t source = environment % g1Q.size();
            throughputQ[environment] = g1Q[source];
            throughputPoints[environment] = g1Points[source];
        }
        const MetalResult throughputGpu =
            runMetal(
                g1,
                throughputQ,
                throughputPoints,
                0u,
                false
            );
        require(
            std::all_of(
                throughputGpu.statuses.begin(),
                throughputGpu.statuses.end(),
                [](const MRArticulatedOperatorStatusGPU& status) {
                    return status.code ==
                        MR_ARTICULATED_OPERATOR_SUCCESS;
                }
            ),
            "parallel G1 throughput batch failed"
        );

        auto invalidPoints = freePoints;
        invalidPoints[0][0].bodyIndex = MR_INVALID_INDEX;
        const MetalResult invalidGpu =
            runMetal(freeModel, freeQ, invalidPoints);
        require(
            invalidGpu.statuses[0].code ==
                MR_ARTICULATED_OPERATOR_NONFINITE_INPUT &&
                invalidGpu.payloadUntouched(),
            "invalid point did not preserve transactional output"
        );
        auto overflowQ = freeQ;
        auto overflowPoints = freePoints;
        overflowQ[0][0] = std::numeric_limits<float>::max();
        overflowPoints[0][0].localPoint.x =
            std::numeric_limits<float>::max();
        overflowPoints[0][0].worldImpulse = {};
        const MetalResult overflowPublicationGpu =
            runMetal(freeModel, overflowQ, overflowPoints);
        require(
            overflowPublicationGpu.statuses[0].code ==
                MR_ARTICULATED_OPERATOR_NONFINITE_RESULT &&
                overflowPublicationGpu.payloadUntouched(),
            "nonfinite derived publication escaped transaction gate"
        );
        const MetalResult reservedDispatchGpu =
            runMetal(freeModel, freeQ, freePoints, 1u);
        require(
            std::all_of(
                reservedDispatchGpu.statuses.begin(),
                reservedDispatchGpu.statuses.end(),
                [](const MRArticulatedOperatorStatusGPU& status) {
                    return status.code ==
                        MR_ARTICULATED_OPERATOR_INVALID_DISPATCH;
                }
            ) &&
                reservedDispatchGpu.payloadUntouched(),
            "nonzero dispatch reserved field was silently accepted"
        );
        metalrobo::EngineModel invalidArmature = g1;
        invalidArmature.dofs[6].drive.z = -0.01f;
        const MetalResult invalidArmatureGpu =
            runMetal(invalidArmature, g1Q, g1Points);
        require(
            std::all_of(
                invalidArmatureGpu.statuses.begin(),
                invalidArmatureGpu.statuses.end(),
                [](const MRArticulatedOperatorStatusGPU& status) {
                    return status.code ==
                        MR_ARTICULATED_OPERATOR_INVALID_MODEL;
                }
            ) &&
                invalidArmatureGpu.payloadUntouched(),
            "invalid armature did not preserve transactional output"
        );

        metalrobo::EngineModel invalidInverseInertia = g1;
        invalidInverseInertia.bodies[6].inverseInertiaRow0 = {};
        std::string invalidInverseReason;
        require(
            !invalidInverseInertia.valid(&invalidInverseReason),
            "CPU model validator accepted a singular inverse inertia"
        );
        const MetalResult invalidInverseGpu =
            runMetal(invalidInverseInertia, g1Q, g1Points);
        require(
            std::all_of(
                invalidInverseGpu.statuses.begin(),
                invalidInverseGpu.statuses.end(),
                [](const MRArticulatedOperatorStatusGPU& status) {
                    return status.code ==
                        MR_ARTICULATED_OPERATOR_INVALID_MODEL;
                }
            ) &&
                invalidInverseGpu.payloadUntouched(),
            "Metal accepted a singular inverse inertia"
        );

        metalrobo::EngineModel indefiniteInertia = g1;
        MRBodyPropertiesGPU& indefiniteBody =
            indefiniteInertia.bodies[6];
        indefiniteBody.inertiaRow0 = f4(1.0f, 2.0f, 0.0f);
        indefiniteBody.inertiaRow1 = f4(2.0f, 1.0f, 0.0f);
        indefiniteBody.inertiaRow2 = f4(0.0f, 0.0f, 1.0f);
        indefiniteBody.inverseInertiaRow0 =
            f4(-1.0f / 3.0f, 2.0f / 3.0f, 0.0f);
        indefiniteBody.inverseInertiaRow1 =
            f4(2.0f / 3.0f, -1.0f / 3.0f, 0.0f);
        indefiniteBody.inverseInertiaRow2 =
            f4(0.0f, 0.0f, 1.0f);
        std::string indefiniteReason;
        require(
            !indefiniteInertia.valid(&indefiniteReason),
            "CPU model validator accepted indefinite inertia"
        );
        const MetalResult indefiniteGpu =
            runMetal(indefiniteInertia, g1Q, g1Points);
        require(
            std::all_of(
                indefiniteGpu.statuses.begin(),
                indefiniteGpu.statuses.end(),
                [](const MRArticulatedOperatorStatusGPU& status) {
                    return status.code ==
                        MR_ARTICULATED_OPERATOR_INVALID_MODEL;
                }
            ) &&
                indefiniteGpu.payloadUntouched(),
            "Metal accepted indefinite inertia with positive diagonal"
        );

        std::cout
            << "MetalRobo generic articulated operator probe passed on "
            << g1Gpu.deviceName << '\n';
        printMetrics(
            "floating 6-DoF analytic",
            freeMetrics,
            freeGpu.elapsedMilliseconds
        );
        printMetrics(
            "fixed 1-DoF analytic",
            fixedMetrics,
            fixedGpu.elapsedMilliseconds
        );
        printMetrics(
            "Unitree G1 3-env x 35-DoF",
            g1Metrics,
            g1Gpu.elapsedMilliseconds
        );
        std::cout
            << "  G1 topology: bodies="
            << g1.articulations[0].bodyCount
            << " joints=" << g1.articulations[0].jointCount
            << " nq=" << g1.articulations[0].nq
            << " nv=" << g1.articulations[0].nv
            << " point_impulses=" << g1Points[0].size()
            << " deterministic_replay=yes"
            << " transactional_failure=yes"
            << " armature_canary=yes"
            << " inertia_validation_parity=yes\n";
        std::cout
            << "  G1 throughput: environments="
            << throughputEnvironmentCount
            << " threadgroup_lanes=32"
            << " diagnostic_mass=no"
            << " elapsed_ms=" << throughputGpu.elapsedMilliseconds
            << " environments_per_s="
            << (
                1000.0 *
                static_cast<double>(throughputEnvironmentCount) /
                throughputGpu.elapsedMilliseconds
            )
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "MetalRobo generic articulated operator probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
