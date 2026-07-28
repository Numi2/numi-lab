#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedContact.hpp"
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

#ifndef METALROBO_INVERSE_MASS_METALLIB
#define METALROBO_INVERSE_MASS_METALLIB ""
#endif

namespace {

constexpr std::uint8_t kSentinel = 0xa5u;

using InverseMassDispatchGPU = MRInverseMassDispatchGPU;
using InverseMassStatusGPU = MRInverseMassStatusGPU;
using RhsVectors = std::vector<std::vector<float>>;
using RhsBatch = std::vector<RhsVectors>;

static_assert(sizeof(InverseMassDispatchGPU) == 48u);
static_assert(sizeof(InverseMassStatusGPU) == 48u);

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

struct InverseMassResult {
    InverseMassDispatchGPU dispatch{};
    std::vector<float> output;
    std::vector<InverseMassStatusGPU> statuses;
    std::string deviceName;
    double elapsedMilliseconds = 0.0;

    [[nodiscard]] bool payloadUntouched() const {
        return allSentinel(output);
    }
};

InverseMassResult runMetal(
    const metalrobo::EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::vector<std::vector<float>>& q,
    const RhsBatch& rightHandSides
) {
    @autoreleasepool {
        require(
            articulationIndex < model.articulations.size(),
            "inverse-mass articulation index is out of range"
        );
        require(
            !q.empty() && rightHandSides.size() == q.size(),
            "invalid inverse-mass environment batch"
        );
        const auto& articulation =
            model.articulations[articulationIndex];
        const std::size_t environmentCount = q.size();
        require(
            !rightHandSides.front().empty() &&
                rightHandSides.front().size() <=
                    MR_ARTICULATED_INVERSE_MASS_MAX_RHS,
            "inverse-mass RHS count is out of range"
        );
        const std::size_t rhsCount =
            rightHandSides.front().size();
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            require(
                q[environment].size() == articulation.nq &&
                    rightHandSides[environment].size() == rhsCount,
                "wrong inverse-mass environment dimensions"
            );
            for (const auto& rhs : rightHandSides[environment]) {
                require(
                    rhs.size() == articulation.nv,
                    "wrong inverse-mass vector dimensions"
                );
            }
        }

        InverseMassDispatchGPU dispatch{};
        dispatch.articulationIndex = articulationIndex;
        dispatch.environmentCount =
            static_cast<std::uint32_t>(environmentCount);
        dispatch.rhsCount = static_cast<std::uint32_t>(rhsCount);
        dispatch.qStride = articulation.nq + 2u;
        dispatch.rhsVectorStride = articulation.nv + 3u;
        dispatch.rhsEnvironmentStride =
            dispatch.rhsCount * dispatch.rhsVectorStride + 5u;
        dispatch.outputVectorStride = articulation.nv + 4u;
        dispatch.outputEnvironmentStride =
            dispatch.rhsCount * dispatch.outputVectorStride + 7u;

        std::vector<float> flatQ(
            environmentCount * dispatch.qStride,
            19.0f
        );
        std::vector<float> flatRhs(
            environmentCount * dispatch.rhsEnvironmentStride,
            -23.0f
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                q[environment].begin(),
                q[environment].end(),
                flatQ.begin() + environment * dispatch.qStride
            );
            for (std::size_t rhs = 0u; rhs < rhsCount; ++rhs) {
                std::copy(
                    rightHandSides[environment][rhs].begin(),
                    rightHandSides[environment][rhs].end(),
                    flatRhs.begin() +
                        environment *
                            dispatch.rhsEnvironmentStride +
                        rhs * dispatch.rhsVectorStride
                );
            }
        }

        InverseMassResult result;
        result.dispatch = dispatch;
        result.output = sentinelVector<float>(
            environmentCount * dispatch.outputEnvironmentStride
        );
        result.statuses = sentinelVector<InverseMassStatusGPU>(
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
            "empty inverse-mass model stream"
        );

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create command queue");
        NSString* path = [NSString
            stringWithUTF8String:METALROBO_INVERSE_MASS_METALLIB];
        require(
            path != nil && path.length != 0u,
            "inverse-mass metallib path is empty"
        );
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:path]
                       error:&error];
        require(
            library != nil,
            "failed to load inverse-mass metallib: " +
                describeError(error)
        );
        id<MTLFunction> function = [library
            newFunctionWithName:@"mr_articulated_inverse_mass"];
        require(function != nil, "inverse-mass kernel is missing");
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        require(
            pipeline != nil,
            "failed to create inverse-mass pipeline: " +
                describeError(error)
        );
        require(
            pipeline.staticThreadgroupMemoryLength <=
                device.maxThreadgroupMemoryLength,
            "inverse-mass threadgroup scratch exceeds device capacity"
        );

        id<MTLBuffer> worldBuffer = makeSharedBuffer(
            device, &model.world, 1u, @"inverse mass world"
        );
        id<MTLBuffer> articulationBuffer = makeSharedBuffer(
            device,
            model.articulations.data(),
            model.articulations.size(),
            @"inverse mass articulations"
        );
        id<MTLBuffer> jointBuffer = makeSharedBuffer(
            device,
            jointRecords.data(),
            jointRecords.size(),
            @"inverse mass joints"
        );
        id<MTLBuffer> dofBuffer = makeSharedBuffer(
            device,
            model.dofs.data(),
            model.dofs.size(),
            @"inverse mass dofs"
        );
        id<MTLBuffer> bodyBuffer = makeSharedBuffer(
            device,
            model.bodies.data(),
            model.bodies.size(),
            @"inverse mass bodies"
        );
        id<MTLBuffer> dispatchBuffer = makeSharedBuffer(
            device, &dispatch, 1u, @"inverse mass dispatch"
        );
        id<MTLBuffer> qBuffer = makeSharedBuffer(
            device, flatQ.data(), flatQ.size(), @"inverse mass q"
        );
        id<MTLBuffer> rhsBuffer = makeSharedBuffer(
            device,
            flatRhs.data(),
            flatRhs.size(),
            @"inverse mass RHS"
        );
        id<MTLBuffer> outputBuffer = makeSharedBuffer(
            device,
            result.output.data(),
            result.output.size(),
            @"inverse mass output"
        );
        id<MTLBuffer> statusBuffer = makeSharedBuffer(
            device,
            result.statuses.data(),
            result.statuses.size(),
            @"inverse mass status"
        );

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        require(
            commandBuffer != nil && encoder != nil,
            "failed to create inverse-mass command encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:worldBuffer offset:0 atIndex:0];
        [encoder setBuffer:articulationBuffer offset:0 atIndex:1];
        [encoder setBuffer:jointBuffer offset:0 atIndex:2];
        [encoder setBuffer:dofBuffer offset:0 atIndex:3];
        [encoder setBuffer:bodyBuffer offset:0 atIndex:4];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:5];
        [encoder setBuffer:qBuffer offset:0 atIndex:6];
        [encoder setBuffer:rhsBuffer offset:0 atIndex:7];
        [encoder setBuffer:outputBuffer offset:0 atIndex:8];
        [encoder setBuffer:statusBuffer offset:0 atIndex:9];
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
            "inverse-mass command failed: " +
                describeError(commandBuffer.error)
        );
        result.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
        std::memcpy(
            result.output.data(),
            outputBuffer.contents,
            result.output.size() * sizeof(float)
        );
        std::memcpy(
            result.statuses.data(),
            statusBuffer.contents,
            result.statuses.size() *
                sizeof(InverseMassStatusGPU)
        );
        result.deviceName = nsString(device.name);
        return result;
    }
}

bool outputPaddingUntouched(const InverseMassResult& result) {
    const auto& dispatch = result.dispatch;
    for (std::size_t environment = 0u;
         environment < dispatch.environmentCount;
         ++environment) {
        for (std::size_t offset = 0u;
             offset < dispatch.outputEnvironmentStride;
             ++offset) {
            bool active = false;
            for (std::size_t rhs = 0u;
                 rhs < dispatch.rhsCount;
                 ++rhs) {
                const std::size_t begin =
                    rhs * dispatch.outputVectorStride;
                if (offset >= begin &&
                    offset < begin +
                        result.statuses[environment].nv) {
                    active = true;
                    break;
                }
            }
            if (!active) {
                const auto* bytes =
                    reinterpret_cast<const std::uint8_t*>(
                        &result.output[
                            environment *
                                dispatch.outputEnvironmentStride +
                            offset
                        ]
                    );
                for (std::size_t byte = 0u;
                     byte < sizeof(float);
                     ++byte) {
                    if (bytes[byte] != kSentinel) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
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
    dof.drive = f4(0.0f, 0.0f, armature);
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
    model.name = "inverse_mass_fixed_pendulum";
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
    model.name = "inverse_mass_branched_tree";
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

metalrobo::EngineModel makeOffsetModel() {
    const auto prefix = makeFixedPendulumModel();
    const auto source = makeBranchedModel();
    metalrobo::EngineModel model;
    model.name = "inverse_mass_nonzero_offsets";
    setWorldDefaults(
        model,
        static_cast<std::uint32_t>(
            prefix.bodies.size() + source.bodies.size()
        ),
        static_cast<std::uint32_t>(
            prefix.joints.size() + source.joints.size()
        ),
        static_cast<std::uint32_t>(
            prefix.defaultQ.size() + source.defaultQ.size()
        ),
        static_cast<std::uint32_t>(
            prefix.defaultV.size() + source.defaultV.size()
        )
    );
    model.world.articulationCount = 2u;
    model.articulations = prefix.articulations;
    model.bodies = prefix.bodies;
    model.joints = prefix.joints;
    model.dofs = prefix.dofs;
    model.defaultQ = prefix.defaultQ;
    model.defaultV = prefix.defaultV;

    const std::uint32_t bodyOffset =
        static_cast<std::uint32_t>(prefix.bodies.size());
    const std::uint32_t jointOffset =
        static_cast<std::uint32_t>(prefix.joints.size());
    const std::uint32_t qOffset =
        static_cast<std::uint32_t>(prefix.defaultQ.size());
    const std::uint32_t vOffset =
        static_cast<std::uint32_t>(prefix.defaultV.size());

    auto articulation = source.articulations[0];
    articulation.rootBody += bodyOffset;
    articulation.firstBody += bodyOffset;
    articulation.firstJoint += jointOffset;
    articulation.qOffset += qOffset;
    articulation.vOffset += vOffset;
    model.articulations.push_back(articulation);
    for (auto body : source.bodies) {
        body.articulationIndex = 1u;
        if (body.parentBody != MR_INVALID_INDEX) {
            body.parentBody += bodyOffset;
        }
        if (body.inboundJoint != MR_INVALID_INDEX) {
            body.inboundJoint += jointOffset;
        }
        model.bodies.push_back(body);
    }
    for (auto joint : source.joints) {
        joint.parentBody += bodyOffset;
        joint.childBody += bodyOffset;
        joint.qOffset += qOffset;
        joint.vOffset += vOffset;
        model.joints.push_back(joint);
    }
    for (auto dof : source.dofs) {
        dof.articulationIndex = 1u;
        dof.jointIndex += jointOffset;
        dof.qIndex += qOffset;
        dof.vIndex += vOffset;
        model.dofs.push_back(dof);
    }
    model.defaultQ.insert(
        model.defaultQ.end(),
        source.defaultQ.begin(),
        source.defaultQ.end()
    );
    model.defaultV.insert(
        model.defaultV.end(),
        source.defaultV.begin(),
        source.defaultV.end()
    );
    return model;
}

std::vector<double> solveRetainedCholesky(
    const std::span<const double> lower,
    const std::span<const float> rhs
) {
    const std::size_t dimension = rhs.size();
    require(
        lower.size() == dimension * dimension,
        "retained Cholesky dimensions are wrong"
    );
    std::vector<double> intermediate(dimension);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = rhs[row];
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            value -=
                lower[row * dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value / lower[row * dimension + row];
    }
    std::vector<double> output(dimension);
    for (std::size_t reverse = 0u;
         reverse < dimension;
         ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                lower[column * dimension + row] *
                output[column];
        }
        output[row] =
            value / lower[row * dimension + row];
    }
    return output;
}

struct ParityMetrics {
    double absoluteError = 0.0;
    double scaledError = 0.0;
    double scaledResidual = 0.0;
};

void compareBatch(
    const metalrobo::EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::vector<std::vector<float>>& q,
    const RhsBatch& rightHandSides,
    const InverseMassResult& gpu,
    ParityMetrics& metrics
) {
    const auto& articulation =
        model.articulations[articulationIndex];
    for (std::size_t environment = 0u;
         environment < q.size();
         ++environment) {
        const auto& status = gpu.statuses[environment];
        require(
            status.code == MR_INVERSE_MASS_SUCCESS &&
                status.environment == environment &&
                status.articulationIndex == articulationIndex &&
                status.bodyCount == articulation.bodyCount &&
                status.nq == articulation.nq &&
                status.nv == articulation.nv &&
                status.rhsCount ==
                    rightHandSides[environment].size(),
            "inverse-mass GPU status failed for environment " +
                std::to_string(environment) +
                " code=" + std::to_string(status.code) +
                " index=" + std::to_string(status.failingIndex)
        );
        require(
            std::isfinite(status.diagnostics.x) &&
                std::isfinite(status.diagnostics.y) &&
                std::isfinite(status.diagnostics.z) &&
                std::isfinite(status.diagnostics.w) &&
                status.diagnostics.x > 0.0f &&
                status.diagnostics.y >= status.diagnostics.x,
            "inverse-mass diagnostics are invalid"
        );

        std::vector<double> qCpu(
            q[environment].begin(),
            q[environment].end()
        );
        std::vector<double> zeroVelocity(articulation.nv);
        metalrobo::ArticulatedContactProblem factorProblem;
        const auto factorDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                model,
                articulationIndex,
                qCpu,
                zeroVelocity,
                std::span<const metalrobo::ArticulatedContact>{},
                factorProblem
            );
        require(
            factorDiagnostics.succeeded() &&
                factorProblem.massCholeskyLower.size() ==
                    static_cast<std::size_t>(articulation.nv) *
                        articulation.nv,
            "CPU retained-Cholesky oracle failed"
        );
        std::vector<double> massMatrix(
            static_cast<std::size_t>(articulation.nv) *
            articulation.nv
        );
        const auto massDiagnostics =
            metalrobo::computeArticulatedMassMatrix(
                model,
                articulationIndex,
                qCpu,
                massMatrix
            );
        require(
            massDiagnostics.succeeded(),
            "CPU mass matrix oracle failed"
        );

        for (std::size_t rhsIndex = 0u;
             rhsIndex < rightHandSides[environment].size();
             ++rhsIndex) {
            const auto& rhs =
                rightHandSides[environment][rhsIndex];
            const auto expected = solveRetainedCholesky(
                factorProblem.massCholeskyLower,
                rhs
            );
            const std::size_t outputBase =
                environment *
                    gpu.dispatch.outputEnvironmentStride +
                rhsIndex * gpu.dispatch.outputVectorStride;
            double maximumRhs = 0.0;
            double maximumResidual = 0.0;
            for (std::size_t row = 0u;
                 row < articulation.nv;
                 ++row) {
                const double actual =
                    gpu.output[outputBase + row];
                const double error =
                    std::abs(actual - expected[row]);
                metrics.absoluteError =
                    std::max(metrics.absoluteError, error);
                metrics.scaledError = std::max(
                    metrics.scaledError,
                    error / (1.0 + std::abs(expected[row]))
                );
                double action = 0.0;
                for (std::size_t column = 0u;
                     column < articulation.nv;
                     ++column) {
                    action +=
                        massMatrix[
                            row * articulation.nv + column
                        ] *
                        gpu.output[outputBase + column];
                }
                maximumResidual = std::max(
                    maximumResidual,
                    std::abs(action - rhs[row])
                );
                maximumRhs = std::max(
                    maximumRhs,
                    std::abs(static_cast<double>(rhs[row]))
                );
            }
            metrics.scaledResidual = std::max(
                metrics.scaledResidual,
                maximumResidual / (1.0 + maximumRhs)
            );
        }
    }
    require(
        outputPaddingUntouched(gpu),
        "inverse-mass kernel overwrote output padding"
    );
}

void requireParity(
    const ParityMetrics& metrics,
    const std::string& label
) {
    require(
        metrics.scaledError < 5.0e-5 &&
            metrics.scaledResidual < 8.0e-5,
        label + " inverse-mass parity exceeded tolerance: error=" +
            std::to_string(metrics.scaledError) +
            " residual=" +
            std::to_string(metrics.scaledResidual)
    );
}

RhsBatch makeRhsBatch(
    const std::size_t environmentCount,
    const std::size_t rhsCount,
    const std::size_t nv
) {
    RhsBatch result(
        environmentCount,
        RhsVectors(rhsCount, std::vector<float>(nv))
    );
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        for (std::size_t rhs = 0u; rhs < rhsCount; ++rhs) {
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                result[environment][rhs][dof] =
                    0.7f *
                        std::sin(
                            0.17f *
                            static_cast<float>(
                                (environment + 1u) *
                                (dof + 1u)
                            ) +
                            0.31f *
                            static_cast<float>(rhs + 1u)
                        ) +
                    0.2f *
                        std::cos(
                            0.11f *
                            static_cast<float>(
                                (rhs + 1u) * (dof + 2u)
                            )
                        );
            }
        }
    }
    return result;
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
            const float x =
                0.02f * static_cast<float>(environment);
            const float y =
                -0.012f * static_cast<float>(environment);
            const float z =
                0.009f * static_cast<float>(environment);
            q[3] = x;
            q[4] = y;
            q[5] = z;
            q[6] =
                std::sqrt(1.0f - x * x - y * y - z * z);
        }
        for (std::size_t joint = 7u;
             joint < q.size();
             ++joint) {
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

void printMetrics(
    const std::string& label,
    const ParityMetrics& metrics,
    const double elapsedMilliseconds
) {
    std::cout
        << "  " << label
        << ": absolute_error=" << metrics.absoluteError
        << " scaled_error=" << metrics.scaledError
        << " scaled_residual=" << metrics.scaledResidual
        << " elapsed_ms=" << elapsedMilliseconds
        << '\n';
}

} // namespace

int main() {
    try {
        std::cout << std::scientific << std::setprecision(6);
        std::string reason;

        const auto freeModel =
            metalrobo::makeFreeSphereEngineModel();
        require(
            freeModel.valid(&reason),
            "free model invalid: " + reason
        );
        std::vector<std::vector<float>> freeQ{
            {0.2f, -0.1f, 0.8f, 0.0f, 0.0f, 0.0f, 1.0f},
        };
        RhsBatch freeRhs{{
            {1.2f, -0.7f, 0.4f, 0.03f, -0.02f, 0.05f},
            {-0.2f, 0.6f, 0.8f, -0.07f, 0.04f, 0.01f},
            {0.5f, 0.3f, -0.9f, 0.02f, 0.08f, -0.06f},
        }};
        const auto freeGpu = runMetal(
            freeModel, 0u, freeQ, freeRhs
        );
        ParityMetrics freeMetrics;
        compareBatch(
            freeModel,
            0u,
            freeQ,
            freeRhs,
            freeGpu,
            freeMetrics
        );
        requireParity(freeMetrics, "free body");
        require(
            std::abs(freeGpu.output[0] - 1.2f) < 2.0e-6f &&
                std::abs(freeGpu.output[1] + 0.7f) <
                    2.0e-6f &&
                std::abs(freeGpu.output[2] - 0.4f) <
                    2.0e-6f &&
                std::abs(freeGpu.output[3] - 0.3f) <
                    2.0e-6f &&
                std::abs(freeGpu.output[4] + 0.2f) <
                    2.0e-6f &&
                std::abs(freeGpu.output[5] - 0.5f) <
                    2.0e-6f,
            "free-body analytic inverse mass failed"
        );

        const auto pendulum = makeFixedPendulumModel();
        require(
            pendulum.valid(&reason),
            "pendulum model invalid: " + reason
        );
        std::vector<std::vector<float>> pendulumQ{{0.43f}};
        RhsBatch pendulumRhs{{{0.65f}}};
        const auto pendulumGpu = runMetal(
            pendulum, 0u, pendulumQ, pendulumRhs
        );
        ParityMetrics pendulumMetrics;
        compareBatch(
            pendulum,
            0u,
            pendulumQ,
            pendulumRhs,
            pendulumGpu,
            pendulumMetrics
        );
        requireParity(pendulumMetrics, "fixed pendulum");
        require(
            std::abs(
                pendulumGpu.output[0] -
                0.65f / (0.35f + 3.0f * 0.16f + 0.025f)
            ) < 2.0e-6f,
            "pendulum armature was not applied exactly"
        );

        const auto branched = makeBranchedModel();
        require(
            branched.valid(&reason),
            "branched model invalid: " + reason
        );
        std::vector<std::vector<float>> branchedQ{
            branched.defaultQ,
        };
        auto branchedRhs = makeRhsBatch(1u, 2u, 2u);
        const auto branchedGpu = runMetal(
            branched, 0u, branchedQ, branchedRhs
        );
        ParityMetrics branchedMetrics;
        compareBatch(
            branched,
            0u,
            branchedQ,
            branchedRhs,
            branchedGpu,
            branchedMetrics
        );
        requireParity(branchedMetrics, "branched tree");

        const auto offset = makeOffsetModel();
        require(
            offset.valid(&reason),
            "offset model invalid: " + reason
        );
        const auto& offsetArticulation = offset.articulations[1];
        std::vector<std::vector<float>> offsetQ{{
            offset.defaultQ.begin() + offsetArticulation.qOffset,
            offset.defaultQ.begin() +
                offsetArticulation.qOffset +
                offsetArticulation.nq,
        }};
        auto offsetRhs = makeRhsBatch(
            1u, 3u, offsetArticulation.nv
        );
        const auto offsetGpu = runMetal(
            offset, 1u, offsetQ, offsetRhs
        );
        ParityMetrics offsetMetrics;
        compareBatch(
            offset,
            1u,
            offsetQ,
            offsetRhs,
            offsetGpu,
            offsetMetrics
        );
        requireParity(offsetMetrics, "nonzero offsets");

        const auto g1 = metalrobo::makeUnitreeG1EngineModel();
        require(
            g1.valid(&reason),
            "G1 model invalid: " + reason
        );
        const auto g1Q = makeG1QBatch(g1);
        const auto g1Rhs = makeRhsBatch(
            g1Q.size(),
            3u,
            g1.articulations[0].nv
        );
        const auto g1Gpu = runMetal(g1, 0u, g1Q, g1Rhs);
        ParityMetrics g1Metrics;
        compareBatch(
            g1,
            0u,
            g1Q,
            g1Rhs,
            g1Gpu,
            g1Metrics
        );
        requireParity(g1Metrics, "actual G1");
        const auto g1Replay = runMetal(g1, 0u, g1Q, g1Rhs);
        require(
            g1Gpu.output == g1Replay.output &&
                std::memcmp(
                    g1Gpu.statuses.data(),
                    g1Replay.statuses.data(),
                    g1Gpu.statuses.size() *
                        sizeof(InverseMassStatusGPU)
                ) == 0,
            "G1 inverse-mass replay was not bitwise deterministic"
        );

        auto invalidRhs = g1Rhs;
        invalidRhs.resize(1u);
        invalidRhs[0][2][7] =
            std::numeric_limits<float>::quiet_NaN();
        auto invalidQ = g1Q;
        invalidQ.resize(1u);
        const auto invalidGpu = runMetal(
            g1, 0u, invalidQ, invalidRhs
        );
        require(
            invalidGpu.statuses[0].code ==
                MR_INVERSE_MASS_NONFINITE_INPUT &&
                invalidGpu.payloadUntouched(),
            "nonfinite inverse-mass input escaped transaction gate"
        );

        auto tinyPivot = freeModel;
        auto& tinyBody = tinyPivot.bodies[
            tinyPivot.articulations[0].rootBody
        ];
        tinyBody.massAndInverseMass =
            f4(1.0e-20f, 1.0e20f, 0.0f);
        tinyBody.inertiaRow0 = f4(1.0e-20f, 0.0f, 0.0f);
        tinyBody.inertiaRow1 = f4(0.0f, 1.0e-20f, 0.0f);
        tinyBody.inertiaRow2 = f4(0.0f, 0.0f, 1.0e-20f);
        const auto tinyPivotGpu = runMetal(
            tinyPivot, 0u, freeQ, freeRhs
        );
        require(
            tinyPivotGpu.statuses[0].code ==
                MR_INVERSE_MASS_FACTORIZATION_FAILED &&
                tinyPivotGpu.payloadUntouched(),
            "tiny inverse-mass pivot was clamped"
        );

        constexpr std::size_t kThroughputEnvironments = 1024u;
        std::vector<std::vector<float>> throughputQ(
            kThroughputEnvironments
        );
        RhsBatch throughputRhs(kThroughputEnvironments);
        for (std::size_t environment = 0u;
             environment < kThroughputEnvironments;
             ++environment) {
            const std::size_t source = environment % g1Q.size();
            throughputQ[environment] = g1Q[source];
            throughputRhs[environment] = g1Rhs[source];
        }
        const auto throughputGpu = runMetal(
            g1, 0u, throughputQ, throughputRhs
        );
        require(
            std::all_of(
                throughputGpu.statuses.begin(),
                throughputGpu.statuses.end(),
                [](const InverseMassStatusGPU& status) {
                    return status.code == MR_INVERSE_MASS_SUCCESS;
                }
            ),
            "G1 inverse-mass throughput batch failed"
        );

        std::cout
            << "MetalRobo O(n) inverse-mass probe passed on "
            << g1Gpu.deviceName << '\n';
        printMetrics(
            "free body 3-RHS",
            freeMetrics,
            freeGpu.elapsedMilliseconds
        );
        printMetrics(
            "fixed pendulum 1-RHS",
            pendulumMetrics,
            pendulumGpu.elapsedMilliseconds
        );
        printMetrics(
            "branched tree 2-RHS",
            branchedMetrics,
            branchedGpu.elapsedMilliseconds
        );
        printMetrics(
            "nonzero articulation/body/q/v offsets",
            offsetMetrics,
            offsetGpu.elapsedMilliseconds
        );
        printMetrics(
            "Unitree G1 3-env x 3-RHS x 35-DoF",
            g1Metrics,
            g1Gpu.elapsedMilliseconds
        );
        std::cout
            << "  G1 inverse-mass throughput: environments="
            << kThroughputEnvironments
            << " rhs_per_environment=3"
            << " elapsed_ms="
            << throughputGpu.elapsedMilliseconds
            << " environment_actions_per_s="
            << (
                1000.0 *
                static_cast<double>(kThroughputEnvironments) /
                throughputGpu.elapsedMilliseconds
            )
            << " rhs_actions_per_s="
            << (
                3000.0 *
                static_cast<double>(kThroughputEnvironments) /
                throughputGpu.elapsedMilliseconds
            )
            << " retained_factor_oracle=yes"
            << " armature=yes"
            << " replay=bitwise"
            << " transaction=pass"
            << " stride_canaries=pass"
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "MetalRobo O(n) inverse-mass probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
