#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/G1.hpp"

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
#include <vector>

#ifndef METALROBO_ABA_METALLIB
#define METALROBO_ABA_METALLIB ""
#endif

namespace {

constexpr std::uint8_t kSentinel = 0xa5u;
constexpr std::uint32_t kABAHasBodyWrenches =
    MR_ABA_HAS_BODY_WRENCHES;
constexpr std::uint32_t kABAApplyBodyDamping =
    MR_ABA_APPLY_BODY_DAMPING;
constexpr std::uint32_t kABASuccess = MR_ABA_SUCCESS;
constexpr std::uint32_t kABANonfiniteInput =
    MR_ABA_NONFINITE_INPUT;
constexpr std::uint32_t kABAFactorizationFailed =
    MR_ABA_FACTORIZATION_FAILED;

using ABADispatchGPU = MRABADispatchGPU;
using ABABodyWrenchGPU = MRABABodyWrenchGPU;
using ABAStatusGPU = MRABAStatusGPU;

static_assert(sizeof(ABADispatchGPU) == 48u);
static_assert(sizeof(ABABodyWrenchGPU) == 32u);
static_assert(sizeof(ABAStatusGPU) == 48u);

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
    const T* values,
    const std::size_t count,
    NSString* label
) {
    require(values != nullptr && count != 0u, "empty Metal buffer");
    require(
        count <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer byte count overflow"
    );
    id<MTLBuffer> result = [device
        newBufferWithBytes:values
                    length:static_cast<NSUInteger>(count * sizeof(T))
                   options:MTLResourceStorageModeShared];
    require(result != nil, "failed to allocate " + nsString(label));
    result.label = label;
    return result;
}

template <typename T>
std::vector<T> sentinelVector(const std::size_t count) {
    std::vector<T> values(count);
    std::memset(values.data(), kSentinel, values.size() * sizeof(T));
    return values;
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

struct ABAResult {
    ABADispatchGPU dispatch{};
    std::vector<float> acceleration;
    std::vector<float> nextV;
    std::vector<float> nextQ;
    std::vector<ABAStatusGPU> statuses;
    std::string deviceName;
    double elapsedMilliseconds = 0.0;

    [[nodiscard]] bool payloadUntouched() const {
        return allSentinel(acceleration) &&
            allSentinel(nextV) &&
            allSentinel(nextQ);
    }
};

ABAResult runMetal(
    const metalrobo::EngineModel& model,
    const std::vector<std::vector<float>>& q,
    const std::vector<std::vector<float>>& v,
    const std::vector<std::vector<float>>& effort,
    const std::vector<
        std::vector<ABABodyWrenchGPU>
    >& bodyWrenches,
    const bool applyDamping
) {
    @autoreleasepool {
        require(
            model.articulations.size() == 1u,
            "ABA probe expects one articulation"
        );
        require(
            !q.empty() &&
                v.size() == q.size() &&
                effort.size() == q.size(),
            "invalid ABA environment batch"
        );
        const auto& articulation = model.articulations[0];
        const std::size_t environmentCount = q.size();
        const bool hasWrenches = !bodyWrenches.empty();
        if (hasWrenches) {
            require(
                bodyWrenches.size() == environmentCount,
                "wrong wrench environment count"
            );
        }
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            require(
                q[environment].size() == articulation.nq &&
                    v[environment].size() == articulation.nv &&
                    effort[environment].size() == articulation.nv,
                "wrong ABA input dimensions"
            );
            if (hasWrenches) {
                require(
                    bodyWrenches[environment].size() ==
                        articulation.bodyCount,
                    "wrong ABA wrench dimensions"
                );
            }
        }

        ABADispatchGPU dispatch{};
        dispatch.articulationIndex = 0u;
        dispatch.environmentCount =
            static_cast<std::uint32_t>(environmentCount);
        dispatch.flags =
            (hasWrenches ? kABAHasBodyWrenches : 0u) |
            (applyDamping ? kABAApplyBodyDamping : 0u);
        dispatch.qStride = articulation.nq;
        dispatch.vStride = articulation.nv;
        dispatch.effortStride = articulation.nv;
        dispatch.wrenchStride = articulation.bodyCount;
        dispatch.accelerationStride = articulation.nv;
        dispatch.nextVStride = articulation.nv;
        dispatch.nextQStride = articulation.nq;

        std::vector<float> flatQ(
            environmentCount * dispatch.qStride
        );
        std::vector<float> flatV(
            environmentCount * dispatch.vStride
        );
        std::vector<float> flatEffort(
            environmentCount * dispatch.effortStride
        );
        std::vector<ABABodyWrenchGPU> flatWrenches(
            std::max<std::size_t>(
                1u,
                environmentCount * dispatch.wrenchStride
            )
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                q[environment].begin(),
                q[environment].end(),
                flatQ.begin() + environment * dispatch.qStride
            );
            std::copy(
                v[environment].begin(),
                v[environment].end(),
                flatV.begin() + environment * dispatch.vStride
            );
            std::copy(
                effort[environment].begin(),
                effort[environment].end(),
                flatEffort.begin() +
                    environment * dispatch.effortStride
            );
            if (hasWrenches) {
                std::copy(
                    bodyWrenches[environment].begin(),
                    bodyWrenches[environment].end(),
                    flatWrenches.begin() +
                        environment * dispatch.wrenchStride
                );
            }
        }

        ABAResult result;
        result.dispatch = dispatch;
        result.acceleration = sentinelVector<float>(
            environmentCount * dispatch.accelerationStride
        );
        result.nextV = sentinelVector<float>(
            environmentCount * dispatch.nextVStride
        );
        result.nextQ = sentinelVector<float>(
            environmentCount * dispatch.nextQStride
        );
        result.statuses = sentinelVector<ABAStatusGPU>(
            environmentCount
        );

        std::vector<MRJointDescriptorGPU> jointRecords = model.joints;
        if (jointRecords.empty()) {
            jointRecords.resize(1u);
        }
        require(
            !model.articulations.empty() &&
                !model.dofs.empty() &&
                !model.bodies.empty(),
            "empty ABA model stream"
        );

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create command queue");
        NSString* path =
            [NSString stringWithUTF8String:METALROBO_ABA_METALLIB];
        require(
            path != nil && path.length != 0u,
            "ABA metallib path is empty"
        );
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:path]
                       error:&error];
        require(
            library != nil,
            "failed to load ABA metallib: " + describeError(error)
        );
        id<MTLFunction> function =
            [library newFunctionWithName:@"mr_articulated_aba_step"];
        require(function != nil, "ABA kernel is missing");
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        require(
            pipeline != nil,
            "failed to create ABA pipeline: " +
                describeError(error)
        );
        require(
            pipeline.staticThreadgroupMemoryLength <=
                device.maxThreadgroupMemoryLength,
            "ABA threadgroup scratch exceeds device capacity"
        );

        id<MTLBuffer> worldBuffer = makeSharedBuffer(
            device, &model.world, 1u, @"ABA world"
        );
        id<MTLBuffer> articulationBuffer = makeSharedBuffer(
            device,
            model.articulations.data(),
            model.articulations.size(),
            @"ABA articulations"
        );
        id<MTLBuffer> jointBuffer = makeSharedBuffer(
            device,
            jointRecords.data(),
            jointRecords.size(),
            @"ABA joints"
        );
        id<MTLBuffer> dofBuffer = makeSharedBuffer(
            device,
            model.dofs.data(),
            model.dofs.size(),
            @"ABA dofs"
        );
        id<MTLBuffer> bodyBuffer = makeSharedBuffer(
            device,
            model.bodies.data(),
            model.bodies.size(),
            @"ABA bodies"
        );
        id<MTLBuffer> dispatchBuffer = makeSharedBuffer(
            device, &dispatch, 1u, @"ABA dispatch"
        );
        id<MTLBuffer> qBuffer = makeSharedBuffer(
            device, flatQ.data(), flatQ.size(), @"ABA q"
        );
        id<MTLBuffer> vBuffer = makeSharedBuffer(
            device, flatV.data(), flatV.size(), @"ABA v"
        );
        id<MTLBuffer> effortBuffer = makeSharedBuffer(
            device,
            flatEffort.data(),
            flatEffort.size(),
            @"ABA effort"
        );
        id<MTLBuffer> wrenchBuffer = makeSharedBuffer(
            device,
            flatWrenches.data(),
            flatWrenches.size(),
            @"ABA body wrenches"
        );
        id<MTLBuffer> accelerationBuffer = makeSharedBuffer(
            device,
            result.acceleration.data(),
            result.acceleration.size(),
            @"ABA acceleration"
        );
        id<MTLBuffer> nextVBuffer = makeSharedBuffer(
            device,
            result.nextV.data(),
            result.nextV.size(),
            @"ABA next v"
        );
        id<MTLBuffer> nextQBuffer = makeSharedBuffer(
            device,
            result.nextQ.data(),
            result.nextQ.size(),
            @"ABA next q"
        );
        id<MTLBuffer> statusBuffer = makeSharedBuffer(
            device,
            result.statuses.data(),
            result.statuses.size(),
            @"ABA status"
        );

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        require(
            commandBuffer != nil && encoder != nil,
            "failed to create ABA command encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:worldBuffer offset:0 atIndex:0];
        [encoder setBuffer:articulationBuffer offset:0 atIndex:1];
        [encoder setBuffer:jointBuffer offset:0 atIndex:2];
        [encoder setBuffer:dofBuffer offset:0 atIndex:3];
        [encoder setBuffer:bodyBuffer offset:0 atIndex:4];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:5];
        [encoder setBuffer:qBuffer offset:0 atIndex:6];
        [encoder setBuffer:vBuffer offset:0 atIndex:7];
        [encoder setBuffer:effortBuffer offset:0 atIndex:8];
        [encoder setBuffer:wrenchBuffer offset:0 atIndex:9];
        [encoder setBuffer:accelerationBuffer offset:0 atIndex:10];
        [encoder setBuffer:nextVBuffer offset:0 atIndex:11];
        [encoder setBuffer:nextQBuffer offset:0 atIndex:12];
        [encoder setBuffer:statusBuffer offset:0 atIndex:13];
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
            "ABA command failed: " +
                describeError(commandBuffer.error)
        );
        result.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        std::memcpy(
            result.acceleration.data(),
            accelerationBuffer.contents,
            result.acceleration.size() * sizeof(float)
        );
        std::memcpy(
            result.nextV.data(),
            nextVBuffer.contents,
            result.nextV.size() * sizeof(float)
        );
        std::memcpy(
            result.nextQ.data(),
            nextQBuffer.contents,
            result.nextQ.size() * sizeof(float)
        );
        std::memcpy(
            result.statuses.data(),
            statusBuffer.contents,
            result.statuses.size() * sizeof(ABAStatusGPU)
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
        f4(0.015f, 0.01f, 1.0e6f, 1.0e6f);
    return body;
}

MRJointDescriptorGPU makeJoint(
    const std::uint32_t parent,
    const std::uint32_t child,
    const std::uint32_t type,
    const std::uint32_t qOffset,
    const std::uint32_t vOffset,
    const std::array<float, 3> axis,
    const std::array<float, 3> parentAnchor,
    const std::array<float, 3> childAnchor
) {
    MRJointDescriptorGPU joint{};
    joint.parentBody = parent;
    joint.childBody = child;
    joint.jointType = type;
    joint.qOffset = qOffset;
    joint.nq = type == MR_JOINT_FIXED ? 0u : 1u;
    joint.vOffset = vOffset;
    joint.nv = type == MR_JOINT_FIXED ? 0u : 1u;
    joint.axis0 = f4(axis[0], axis[1], axis[2]);
    joint.parentAnchor = f4(
        parentAnchor[0],
        parentAnchor[1],
        parentAnchor[2]
    );
    joint.childAnchor = f4(
        childAnchor[0],
        childAnchor[1],
        childAnchor[2]
    );
    joint.parentRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    joint.childRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    return joint;
}

MRDofPropertiesGPU makeDof(
    const std::uint32_t joint,
    const std::uint32_t q,
    const std::uint32_t v,
    const float armature
) {
    MRDofPropertiesGPU dof{};
    dof.articulationIndex = 0u;
    dof.jointIndex = joint;
    dof.qIndex = q;
    dof.vIndex = v;
    dof.localDof = 0u;
    dof.drive = f4(0.0f, 0.0f, armature, 0.0f);
    return dof;
}

void setWorldDefaults(
    metalrobo::EngineModel& model,
    const std::uint32_t bodies,
    const std::uint32_t joints,
    const std::uint32_t nq,
    const std::uint32_t nv
) {
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = bodies;
    model.world.articulationCount = 1u;
    model.world.jointCount = joints;
    model.world.nq = nq;
    model.world.nv = nv;
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
}

metalrobo::EngineModel makeFixedPendulumModel() {
    metalrobo::EngineModel model;
    model.name = "aba_fixed_pendulum";
    setWorldDefaults(model, 2u, 1u, 1u, 1u);
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount = 2u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 1u;
    articulation.nq = 1u;
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
    model.joints.push_back(
        makeJoint(
            0u,
            1u,
            MR_JOINT_REVOLUTE,
            0u,
            0u,
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, -0.4f}
        )
    );
    model.dofs.push_back(makeDof(0u, 0u, 0u, 0.025f));
    model.defaultQ = {0.37f};
    model.defaultV = {0.0f};
    return model;
}

metalrobo::EngineModel makeBranchedModel() {
    metalrobo::EngineModel model;
    model.name = "aba_branched_tree";
    setWorldDefaults(model, 4u, 3u, 2u, 2u);
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount = 4u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 3u;
    articulation.nq = 2u;
    articulation.nv = 2u;
    model.articulations.push_back(articulation);
    model.bodies.push_back(
        makeBody(
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            3.0f,
            {0.5f, 0.6f, 0.7f}
        )
    );
    model.bodies.push_back(
        makeBody(0u, 0u, 1.7f, {0.13f, 0.17f, 0.2f})
    );
    model.bodies.push_back(
        makeBody(0u, 1u, 1.2f, {0.09f, 0.11f, 0.14f})
    );
    model.bodies.push_back(
        makeBody(1u, 2u, 0.8f, {0.06f, 0.07f, 0.08f})
    );
    model.joints.push_back(
        makeJoint(
            0u,
            1u,
            MR_JOINT_REVOLUTE,
            0u,
            0u,
            {0.0f, 1.0f, 0.0f},
            {0.15f, 0.0f, 0.1f},
            {0.0f, 0.0f, -0.3f}
        )
    );
    model.joints.push_back(
        makeJoint(
            0u,
            2u,
            MR_JOINT_FIXED,
            1u,
            1u,
            {1.0f, 0.0f, 0.0f},
            {-0.2f, 0.1f, 0.0f},
            {0.0f, 0.0f, -0.2f}
        )
    );
    model.joints.push_back(
        makeJoint(
            1u,
            3u,
            MR_JOINT_CONTINUOUS,
            1u,
            1u,
            {1.0f, 0.0f, 0.0f},
            {0.0f, 0.05f, -0.3f},
            {0.0f, 0.0f, -0.2f}
        )
    );
    model.dofs.push_back(makeDof(0u, 0u, 0u, 0.01f));
    model.dofs.push_back(makeDof(2u, 1u, 1u, 0.02f));
    model.defaultQ = {0.31f, -0.42f};
    model.defaultV = {0.0f, 0.0f};
    return model;
}

struct ParityMetrics {
    double acceleration = 0.0;
    double accelerationScaled = 0.0;
    double nextV = 0.0;
    double nextQ = 0.0;
};

std::vector<metalrobo::ArticulatedBodyWrench> cpuWrenches(
    const metalrobo::EngineModel& model,
    const std::span<const ABABodyWrenchGPU> local
) {
    if (local.empty()) {
        return {};
    }
    const auto& articulation = model.articulations[0];
    std::vector<metalrobo::ArticulatedBodyWrench> result(
        model.bodies.size()
    );
    for (std::size_t localBody = 0u;
         localBody < local.size();
         ++localBody) {
        auto& output =
            result[articulation.firstBody + localBody];
        output.force = {
            local[localBody].force.x,
            local[localBody].force.y,
            local[localBody].force.z,
        };
        output.torque = {
            local[localBody].torque.x,
            local[localBody].torque.y,
            local[localBody].torque.z,
        };
    }
    return result;
}

void compareEnvironment(
    const metalrobo::EngineModel& model,
    const std::size_t environment,
    const std::vector<float>& q,
    const std::vector<float>& v,
    const std::vector<float>& effort,
    const std::span<const ABABodyWrenchGPU> localWrenches,
    const bool applyDamping,
    const ABAResult& gpu,
    ParityMetrics& metrics
) {
    const auto& articulation = model.articulations[0];
    const auto& status = gpu.statuses[environment];
    require(
        status.code == kABASuccess &&
            status.environment == environment &&
            status.bodyCount == articulation.bodyCount &&
            status.nq == articulation.nq &&
            status.nv == articulation.nv,
        "ABA status failed for environment " +
            std::to_string(environment) +
            " code=" + std::to_string(status.code) +
            " index=" + std::to_string(status.failingIndex)
    );
    require(
        std::isfinite(status.diagnostics.x) &&
            std::isfinite(status.diagnostics.y) &&
            std::isfinite(status.diagnostics.z) &&
            std::isfinite(status.diagnostics.w),
        "ABA diagnostics are nonfinite"
    );

    std::vector<double> qCpu(q.begin(), q.end());
    std::vector<double> vCpu(v.begin(), v.end());
    std::vector<double> effortCpu(effort.begin(), effort.end());
    auto wrenchCpu = cpuWrenches(model, localWrenches);
    metalrobo::ArticulatedDynamicsConfig config;
    config.gravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
    };
    config.timestep = model.world.gravityAndTimestep.w;
    config.applyBodyDamping = applyDamping;
    std::vector<double> accelerationCpu(articulation.nv);
    auto diagnostics =
        metalrobo::computeArticulatedForwardDynamics(
            model,
            0u,
            qCpu,
            vCpu,
            effortCpu,
            wrenchCpu,
            accelerationCpu,
            config
        );
    require(diagnostics.succeeded(), "CPU ABA oracle failed");

    const std::size_t accelerationBase =
        environment * gpu.dispatch.accelerationStride;
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        const double error = std::abs(
            gpu.acceleration[accelerationBase + dof] -
            accelerationCpu[dof]
        );
        metrics.acceleration =
            std::max(metrics.acceleration, error);
        metrics.accelerationScaled = std::max(
            metrics.accelerationScaled,
            error / (1.0 + std::abs(accelerationCpu[dof]))
        );
    }

    diagnostics = metalrobo::integrateArticulatedState(
        model,
        0u,
        qCpu,
        vCpu,
        effortCpu,
        wrenchCpu,
        config
    );
    require(
        diagnostics.succeeded(),
        "CPU symplectic oracle failed"
    );
    const std::size_t nextVBase =
        environment * gpu.dispatch.nextVStride;
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        metrics.nextV = std::max(
            metrics.nextV,
            std::abs(gpu.nextV[nextVBase + dof] - vCpu[dof])
        );
    }
    const std::size_t nextQBase =
        environment * gpu.dispatch.nextQStride;
    for (std::size_t coordinate = 0u;
         coordinate < articulation.nq;
         ++coordinate) {
        metrics.nextQ = std::max(
            metrics.nextQ,
            std::abs(
                gpu.nextQ[nextQBase + coordinate] -
                qCpu[coordinate]
            )
        );
    }
}

void requireParity(
    const ParityMetrics& metrics,
    const std::string& label
) {
    require(
        metrics.accelerationScaled < 5.0e-5 &&
            metrics.nextV < 1.0e-4 &&
            metrics.nextQ < 5.0e-5,
        label + " ABA parity exceeded tolerance: a_scaled=" +
            std::to_string(metrics.accelerationScaled) +
            " v=" + std::to_string(metrics.nextV) +
            " q=" + std::to_string(metrics.nextQ)
    );
}

void printMetrics(
    const std::string& label,
    const ParityMetrics& metrics,
    const double elapsedMilliseconds
) {
    std::cout
        << "  " << label
        << ": a=" << metrics.acceleration
        << " a_scaled=" << metrics.accelerationScaled
        << " next_v=" << metrics.nextV
        << " next_q=" << metrics.nextQ
        << " elapsed_ms=" << elapsedMilliseconds
        << '\n';
}

std::vector<std::vector<float>> makeG1QBatch(
    const metalrobo::EngineModel& model
) {
    std::vector<std::vector<float>> result(3u, model.defaultQ);
    for (std::size_t environment = 0u;
         environment < result.size();
         ++environment) {
        auto& q = result[environment];
        q[0] += 0.04f * static_cast<float>(environment);
        q[1] -= 0.025f * static_cast<float>(environment);
        if (environment != 0u) {
            const float x = 0.02f * static_cast<float>(environment);
            const float y = -0.012f * static_cast<float>(environment);
            const float z = 0.009f * static_cast<float>(environment);
            q[3] = x;
            q[4] = y;
            q[5] = z;
            q[6] = std::sqrt(1.0f - x * x - y * y - z * z);
        }
        for (std::size_t joint = 7u; joint < q.size(); ++joint) {
            q[joint] +=
                0.025f *
                std::sin(
                    0.31f *
                    static_cast<float>(
                        (joint - 6u) * (environment + 1u)
                    )
                );
        }
    }
    return result;
}

std::vector<std::vector<float>> makeG1VBatch(
    const metalrobo::EngineModel& model
) {
    std::vector<std::vector<float>> result(
        3u,
        std::vector<float>(model.articulations[0].nv)
    );
    for (std::size_t environment = 0u;
         environment < result.size();
         ++environment) {
        for (std::size_t dof = 0u;
             dof < result[environment].size();
             ++dof) {
            result[environment][dof] =
                0.08f *
                std::sin(
                    0.19f *
                    static_cast<float>(
                        (dof + 1u) * (environment + 1u)
                    )
                );
        }
    }
    return result;
}

std::vector<std::vector<float>> makeG1EffortBatch(
    const metalrobo::EngineModel& model
) {
    std::vector<std::vector<float>> result(
        3u,
        std::vector<float>(model.articulations[0].nv)
    );
    for (std::size_t environment = 0u;
         environment < result.size();
         ++environment) {
        for (std::size_t dof = 0u;
             dof < result[environment].size();
             ++dof) {
            result[environment][dof] =
                0.4f *
                std::cos(
                    0.13f *
                    static_cast<float>(
                        (dof + 1u) * (environment + 2u)
                    )
                );
        }
    }
    return result;
}

} // namespace

int main() {
    try {
        std::cout << std::scientific << std::setprecision(6);

        metalrobo::EngineModel freeModel =
            metalrobo::makeFreeSphereEngineModel();
        freeModel.world.gravityAndTimestep =
            f4(0.0f, -9.81f, 0.0f, 1.0f / 1000.0f);
        std::vector<std::vector<float>> freeQ{
            {0.2f, -0.1f, 0.8f, 0.0f, 0.0f, 0.0f, 1.0f},
        };
        std::vector<std::vector<float>> freeV{
            {0.1f, -0.2f, 0.3f, 0.0f, 0.0f, 0.0f},
        };
        std::vector<std::vector<float>> freeEffort{
            {1.2f, -0.7f, 0.4f, 0.03f, -0.02f, 0.05f},
        };
        std::vector<std::vector<ABABodyWrenchGPU>> freeWrenches{
            {{
                f4(0.3f, 0.2f, -0.1f),
                f4(-0.01f, 0.04f, 0.02f),
            }},
        };
        const ABAResult freeGpu = runMetal(
            freeModel,
            freeQ,
            freeV,
            freeEffort,
            freeWrenches,
            false
        );
        ParityMetrics freeMetrics;
        compareEnvironment(
            freeModel,
            0u,
            freeQ[0],
            freeV[0],
            freeEffort[0],
            freeWrenches[0],
            false,
            freeGpu,
            freeMetrics
        );
        requireParity(freeMetrics, "free analytic");
        require(
            std::abs(freeGpu.acceleration[0] - 1.5f) < 2.0e-6f &&
                std::abs(
                    freeGpu.acceleration[1] - (-10.31f)
                ) < 2.0e-5f &&
                std::abs(freeGpu.acceleration[2] - 0.3f) <
                    2.0e-6f &&
                std::abs(freeGpu.acceleration[3] - 0.2f) <
                    2.0e-6f &&
                std::abs(freeGpu.acceleration[4] - 0.2f) <
                    2.0e-6f &&
                std::abs(freeGpu.acceleration[5] - 0.7f) <
                    2.0e-6f,
            "free-body analytic acceleration failed"
        );

        const auto pendulum = makeFixedPendulumModel();
        std::string reason;
        require(
            pendulum.valid(&reason),
            "pendulum model invalid: " + reason
        );
        std::vector<std::vector<float>> pendulumQ{{0.43f}};
        std::vector<std::vector<float>> pendulumV{{-0.27f}};
        std::vector<std::vector<float>> pendulumEffort{{0.65f}};
        const ABAResult pendulumGpu = runMetal(
            pendulum,
            pendulumQ,
            pendulumV,
            pendulumEffort,
            {},
            true
        );
        ParityMetrics pendulumMetrics;
        compareEnvironment(
            pendulum,
            0u,
            pendulumQ[0],
            pendulumV[0],
            pendulumEffort[0],
            {},
            true,
            pendulumGpu,
            pendulumMetrics
        );
        requireParity(pendulumMetrics, "fixed pendulum");

        const auto branched = makeBranchedModel();
        require(
            branched.valid(&reason),
            "branched model invalid: " + reason
        );
        std::vector<std::vector<float>> branchedQ{
            branched.defaultQ,
        };
        std::vector<std::vector<float>> branchedV{
            {0.34f, -0.22f},
        };
        std::vector<std::vector<float>> branchedEffort{
            {0.8f, -0.45f},
        };
        std::vector<std::vector<ABABodyWrenchGPU>> branchedWrenches{
            std::vector<ABABodyWrenchGPU>(4u),
        };
        branchedWrenches[0][3].force =
            f4(0.3f, -0.2f, 0.5f);
        branchedWrenches[0][3].torque =
            f4(-0.05f, 0.04f, 0.02f);
        const ABAResult branchedGpu = runMetal(
            branched,
            branchedQ,
            branchedV,
            branchedEffort,
            branchedWrenches,
            true
        );
        ParityMetrics branchedMetrics;
        compareEnvironment(
            branched,
            0u,
            branchedQ[0],
            branchedV[0],
            branchedEffort[0],
            branchedWrenches[0],
            true,
            branchedGpu,
            branchedMetrics
        );
        requireParity(branchedMetrics, "branched tree");

        const auto g1 = metalrobo::makeUnitreeG1EngineModel();
        const auto g1Q = makeG1QBatch(g1);
        const auto g1V = makeG1VBatch(g1);
        const auto g1Effort = makeG1EffortBatch(g1);
        const ABAResult g1Gpu = runMetal(
            g1,
            g1Q,
            g1V,
            g1Effort,
            {},
            true
        );
        ParityMetrics g1Metrics;
        for (std::size_t environment = 0u;
             environment < g1Q.size();
             ++environment) {
            compareEnvironment(
                g1,
                environment,
                g1Q[environment],
                g1V[environment],
                g1Effort[environment],
                {},
                true,
                g1Gpu,
                g1Metrics
            );
        }
        requireParity(g1Metrics, "actual G1");
        const ABAResult g1Replay = runMetal(
            g1,
            g1Q,
            g1V,
            g1Effort,
            {},
            true
        );
        require(
            g1Gpu.acceleration == g1Replay.acceleration &&
                g1Gpu.nextV == g1Replay.nextV &&
                g1Gpu.nextQ == g1Replay.nextQ &&
                std::memcmp(
                    g1Gpu.statuses.data(),
                    g1Replay.statuses.data(),
                    g1Gpu.statuses.size() * sizeof(ABAStatusGPU)
                ) == 0,
            "G1 ABA replay was not bitwise deterministic"
        );

        auto invalidQ = g1Q;
        invalidQ.resize(1u);
        invalidQ[0][0] =
            std::numeric_limits<float>::quiet_NaN();
        auto invalidV = g1V;
        invalidV.resize(1u);
        auto invalidEffort = g1Effort;
        invalidEffort.resize(1u);
        const ABAResult invalidGpu = runMetal(
            g1,
            invalidQ,
            invalidV,
            invalidEffort,
            {},
            true
        );
        require(
            invalidGpu.statuses[0].code == kABANonfiniteInput &&
                invalidGpu.payloadUntouched(),
            "ABA nonfinite input escaped transaction gate"
        );
        auto tinyPivotModel = freeModel;
        auto& tinyBody = tinyPivotModel.bodies[
            tinyPivotModel.articulations[0].rootBody
        ];
        tinyBody.massAndInverseMass =
            f4(1.0e-20f, 1.0e20f, 0.0f);
        tinyBody.inertiaRow0 = f4(1.0e-20f, 0.0f, 0.0f);
        tinyBody.inertiaRow1 = f4(0.0f, 1.0e-20f, 0.0f);
        tinyBody.inertiaRow2 = f4(0.0f, 0.0f, 1.0e-20f);
        tinyBody.inverseInertiaRow0 =
            f4(1.0e20f, 0.0f, 0.0f);
        tinyBody.inverseInertiaRow1 =
            f4(0.0f, 1.0e20f, 0.0f);
        tinyBody.inverseInertiaRow2 =
            f4(0.0f, 0.0f, 1.0e20f);
        const ABAResult tinyPivotGpu = runMetal(
            tinyPivotModel,
            freeQ,
            freeV,
            freeEffort,
            freeWrenches,
            false
        );
        require(
            tinyPivotGpu.statuses[0].code ==
                kABAFactorizationFailed &&
                tinyPivotGpu.payloadUntouched(),
            "ABA tiny pivot was clamped instead of rejected"
        );

        constexpr std::size_t throughputEnvironmentCount = 1024u;
        std::vector<std::vector<float>> throughputQ(
            throughputEnvironmentCount
        );
        std::vector<std::vector<float>> throughputV(
            throughputEnvironmentCount
        );
        std::vector<std::vector<float>> throughputEffort(
            throughputEnvironmentCount
        );
        for (std::size_t environment = 0u;
             environment < throughputEnvironmentCount;
             ++environment) {
            const std::size_t source = environment % g1Q.size();
            throughputQ[environment] = g1Q[source];
            throughputV[environment] = g1V[source];
            throughputEffort[environment] = g1Effort[source];
        }
        const ABAResult throughputGpu = runMetal(
            g1,
            throughputQ,
            throughputV,
            throughputEffort,
            {},
            true
        );
        require(
            std::all_of(
                throughputGpu.statuses.begin(),
                throughputGpu.statuses.end(),
                [](const ABAStatusGPU& status) {
                    return status.code == kABASuccess;
                }
            ),
            "G1 ABA throughput batch failed"
        );

        std::cout
            << "MetalRobo O(n) ABA probe passed on "
            << g1Gpu.deviceName << '\n';
        printMetrics(
            "free analytic",
            freeMetrics,
            freeGpu.elapsedMilliseconds
        );
        printMetrics(
            "fixed pendulum",
            pendulumMetrics,
            pendulumGpu.elapsedMilliseconds
        );
        printMetrics(
            "branched tree",
            branchedMetrics,
            branchedGpu.elapsedMilliseconds
        );
        printMetrics(
            "Unitree G1 3-env x 35-DoF",
            g1Metrics,
            g1Gpu.elapsedMilliseconds
        );
        std::cout
            << "  G1 ABA throughput: environments="
            << throughputEnvironmentCount
            << " threadgroup_lanes=32"
            << " elapsed_ms=" << throughputGpu.elapsedMilliseconds
            << " environments_per_s="
            << (
                1000.0 *
                static_cast<double>(throughputEnvironmentCount) /
                throughputGpu.elapsedMilliseconds
            )
            << " deterministic_replay=yes"
            << " transactional_failure=yes"
            << " pivot_rejection=yes"
            << " body_wrench=yes"
            << " damping=yes"
            << " armature=yes"
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "MetalRobo O(n) ABA probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
