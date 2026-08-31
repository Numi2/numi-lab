#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "NUMI_MATTER_METALLIB must name the build-tree Matter Metal library"
#endif

#ifndef NUMI_MATTER_MATERIAL
#error "NUMI_MATTER_MATERIAL must name a production Matter material"
#endif

namespace {

constexpr std::uint32_t kDofs = 6u;
constexpr std::uint32_t kBodyCount = 1u;
constexpr float kTimestep = 0.25f;
constexpr float kFiniteDifferenceStep = 1.0e-3f;

static_assert(sizeof(NMFEMHumanAttachmentGPU) == 32u);
static_assert(sizeof(NMFEMNodeStateGPU) == 64u);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string stringValue(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) return {};
    return value.UTF8String;
}

std::string errorValue(NSError* error) {
    return error == nil ? "unknown Metal error" :
        stringValue(error.localizedDescription);
}

bool close(const float lhs, const float rhs, const float tolerance) {
    return std::isfinite(lhs) && std::isfinite(rhs) &&
        std::abs(lhs - rhs) <= tolerance *
            std::max({1.0f, std::abs(lhs), std::abs(rhs)});
}

void requireClose(
    const float lhs,
    const float rhs,
    const float tolerance,
    const std::string& message
) {
    require(close(lhs, rhs, tolerance), message + ": " +
        std::to_string(lhs) + " != " + std::to_string(rhs));
}

nm_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

nm_uint4 u4(
    const std::uint32_t x,
    const std::uint32_t y,
    const std::uint32_t z,
    const std::uint32_t w
) {
    return {x, y, z, w};
}

std::array<float, 3> cross(
    const std::array<float, 3>& lhs,
    const std::array<float, 3>& rhs
) {
    return {
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    };
}

std::array<float, 3> add(
    const std::array<float, 3>& lhs,
    const std::array<float, 3>& rhs
) {
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

std::array<float, 3> scale(
    const float value,
    const std::array<float, 3>& vector
) {
    return {value * vector[0], value * vector[1], value * vector[2]};
}

float dot(
    const std::array<float, 3>& lhs,
    const std::array<float, 3>& rhs
) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<float, 3> rotate(
    const std::array<float, 4>& quaternion,
    const std::array<float, 3>& point
) {
    const std::array<float, 3> imaginary{
        quaternion[0], quaternion[1], quaternion[2]
    };
    const auto twiceCross = scale(2.0f, cross(imaginary, point));
    return add(point, add(
        scale(quaternion[3], twiceCross),
        cross(imaginary, twiceCross)
    ));
}

std::array<float, 4> rotationIncrement(
    const std::array<float, 3>& angular,
    const float timestep
) {
    const float speed = std::sqrt(dot(angular, angular));
    const float angle = timestep * speed;
    if (speed <= 1.0e-12f) return {0.0f, 0.0f, 0.0f, 1.0f};
    const float sineScale = std::sin(0.5f * angle) / speed;
    return {
        sineScale * angular[0],
        sineScale * angular[1],
        sineScale * angular[2],
        std::cos(0.5f * angle),
    };
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::span<const T> values,
    NSString* label
) {
    require(device != nil && !values.empty(), "invalid Metal buffer input");
    require(
        values.size() <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer size overflow"
    );
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:values.data()
                    length:values.size_bytes()
                   options:MTLResourceStorageModeShared];
    require(buffer != nil, "failed to allocate " + stringValue(label));
    buffer.label = label;
    return buffer;
}

template <typename T, std::size_t Count>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::array<T, Count>& values,
    NSString* label
) {
    return makeBuffer<T>(device, std::span<const T>(values), label);
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const T& value,
    NSString* label
) {
    return makeBuffer<T>(device, std::span<const T>(&value, 1u), label);
}

template <typename T>
T value(id<MTLBuffer> buffer, const std::size_t index = 0u) {
    require(
        buffer != nil && buffer.length >= (index + 1u) * sizeof(T),
        "readback buffer is undersized"
    );
    T result{};
    std::memcpy(
        &result,
        static_cast<const std::byte*>(buffer.contents) + index * sizeof(T),
        sizeof(T)
    );
    return result;
}

void appendBytes(std::vector<std::byte>& destination, id<MTLBuffer> buffer) {
    require(buffer != nil && buffer.contents != nullptr,
        "cannot snapshot a non-shared Metal buffer");
    const auto* begin = static_cast<const std::byte*>(buffer.contents);
    destination.insert(destination.end(), begin, begin + buffer.length);
}

struct Harness {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> drive = nil;
    id<MTLComputePipelineState> map = nil;
    id<MTLComputePipelineState> captureResidual = nil;
    id<MTLComputePipelineState> scatterResidual = nil;
    id<MTLComputePipelineState> captureOperator = nil;
    id<MTLComputePipelineState> scatterOperator = nil;
    id<MTLComputePipelineState> maskReactions = nil;
};

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    NSString* qualified = [@"numi_matter_metal::"
        stringByAppendingString:name];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    require(function != nil, "missing Matter function " + stringValue(name));
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil,
        "failed to create " + stringValue(name) + ": " + errorValue(error));
    return result;
}

Harness makeHarness() {
    Harness result;
    result.device = MTLCreateSystemDefaultDevice();
    require(result.device != nil, "no Metal device is available");
    result.queue = [result.device newCommandQueue];
    require(result.queue != nil, "failed to create Metal command queue");
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
    result.library = [result.device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    require(result.library != nil,
        "failed to load Matter metallib: " + errorValue(error));
    result.drive = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_drive_candidate");
    result.map = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_map_direction");
    result.captureResidual = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_capture_residual");
    result.scatterResidual = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_scatter_residual");
    result.captureOperator = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_capture_operator");
    result.scatterOperator = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_scatter_operator");
    result.maskReactions = pipeline(result.device, result.library,
        @"nm_fem_human_attachment_mask_reactions");
    return result;
}

template <typename Binder>
void encode(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> state,
    const NSUInteger threadCount,
    Binder&& bind
) {
    require(commandBuffer != nil && state != nil && threadCount != 0u,
        "invalid Metal dispatch");
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    require(encoder != nil, "failed to create Metal compute encoder");
    [encoder setComputePipelineState:state];
    bind(encoder);
    const NSUInteger width = std::min(
        threadCount, state.maxTotalThreadsPerThreadgroup
    );
    [encoder
        dispatchThreads:MTLSizeMake(threadCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
}

void finish(id<MTLCommandBuffer> commandBuffer) {
    require(commandBuffer != nil, "missing Metal command buffer");
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    require(
        commandBuffer.status == MTLCommandBufferStatusCompleted,
        "Metal command failed: " + errorValue(commandBuffer.error)
    );
}

NMMatterDispatchGPU attachmentDispatch(const std::uint32_t count = 1u) {
    NMMatterDispatchGPU dispatch{};
    dispatch.abiVersion = NM_MATTER_ABI_VERSION;
    dispatch.environmentCount = 1u;
    dispatch.objectCount = 1u;
    dispatch.femNodeCount = 1u;
    dispatch.femHumanAttachmentCount = count;
    dispatch.rigidGeneralizedCapacity = kDofs;
    dispatch.rigidQCapacity = kDofs + 1u;
    dispatch.femHumanAttachmentPointJacobianStride = count * 3u * kDofs;
    dispatch.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, kTimestep);
    return dispatch;
}

NMFEMHumanAttachmentGPU attachment(const std::uint32_t body = 0u) {
    NMFEMHumanAttachmentGPU result{};
    result.identity = u4(0u, body, 0u, 0x41545431u);
    result.localPoint = f4(0.3f, -0.2f, 0.1f, 0.0f);
    return result;
}

NMFEMNodeStateGPU node() {
    NMFEMNodeStateGPU result{};
    result.positionAndMass = f4(-3.0f, -2.0f, -1.0f, 2.0f);
    result.velocityAndInverseMass = f4(0.1f, 0.2f, 0.3f, 0.5f);
    result.restAndFixed = f4(-3.0f, -2.0f, -1.0f, 2.0f);
    result.deltaVelocity = f4(0.0f, 0.0f, 0.0f, 0.0f);
    return result;
}

MRBodyStateGPU body() {
    MRBodyStateGPU result{};
    result.position = {1.0f, 2.0f, 3.0f, 1.0f};
    result.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    result.linearVelocityAndInverseMass = {0.4f, -0.3f, 0.2f, 1.0f};
    result.angularVelocity = {0.5f, -0.25f, 0.75f, 0.0f};
    result.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
    result.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    result.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    result.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    return result;
}

std::array<float, 3u * kDofs> pointJacobian() {
    // [I, -skew(r)] for r = (0.3, -0.2, 0.1), component-major.
    return {
        1.0f, 0.0f, 0.0f,  0.0f,  0.1f,  0.2f,
        0.0f, 1.0f, 0.0f, -0.1f,  0.0f,  0.3f,
        0.0f, 0.0f, 1.0f, -0.2f, -0.3f,  0.0f,
    };
}

std::array<float, kDofs> generalizedDirection() {
    return {0.4f, -0.3f, 0.2f, 0.5f, -0.25f, 0.75f};
}

std::array<float, 3> applyJacobian(
    const std::array<float, 3u * kDofs>& jacobian,
    const std::array<float, kDofs>& direction
) {
    std::array<float, 3> result{};
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        for (std::uint32_t dof = 0u; dof < kDofs; ++dof) {
            result[component] +=
                jacobian[component * kDofs + dof] * direction[dof];
        }
    }
    return result;
}

struct ValidRun {
    NMFEMNodeStateGPU candidate{};
    NMMatterStatusGPU status{};
    std::array<nm_float4, 8u> residual{};
    std::array<nm_float4, 8u> mapped{};
    std::array<nm_float4, 8u> operatorWork{};
    nm_float4 residualImpulse{};
    nm_float4 rawReaction{};
    nm_float4 attachmentOperator{};
    std::vector<std::byte> replayBytes;
};

ValidRun runValid(const Harness& harness) {
    const NMMatterDispatchGPU dispatch = attachmentDispatch();
    const std::uint32_t bodyCount = kBodyCount;
    const std::uint32_t bodyStride = kBodyCount;
    const std::uint32_t articulatedNv = kDofs;
    const NMFEMHumanAttachmentGPU binding = attachment();
    const NMFEMNodeStateGPU accepted = node();
    const NMFEMNodeStateGPU initialCandidate = accepted;
    const MRBodyStateGPU candidateBody = body();
    const NMIncidenceRangeGPU range{0u, 0u, 0u, 0u};
    const NMContinuumObjectGPU object{};
    const NMSchedulerStateGPU scheduler{};
    NMMatterStatusGPU status{};
    status.code = NM_STATUS_SUCCESS;
    status.environment = 0u;
    const auto jacobian = pointJacobian();
    const auto p = generalizedDirection();
    std::array<float, kDofs> deltaV{};

    constexpr std::size_t kUnknownCount = 2u + kDofs;
    std::array<nm_float4, kUnknownCount> residual{};
    residual[0] = f4(2.0f, -3.0f, 4.0f, 7.0f);
    std::array<nm_float4, kUnknownCount> preconditioned{};
    std::array<nm_float4, kUnknownCount> direction{};
    for (std::uint32_t dof = 0u; dof < kDofs; ++dof) {
        direction[2u + dof].x = p[dof];
    }
    const auto mapped = applyJacobian(jacobian, p);
    std::array<nm_float4, kUnknownCount> operatorWork{};
    operatorWork[0] = f4(
        2.0f * mapped[0], 3.0f * mapped[1], 4.0f * mapped[2], 0.0f
    );
    const nm_float4 zero{};

    id<MTLBuffer> dispatchBuffer = makeBuffer(
        harness.device, dispatch, @"attachment dispatch");
    id<MTLBuffer> attachmentBuffer = makeBuffer(
        harness.device, binding, @"attachment record");
    id<MTLBuffer> bodyBuffer = makeBuffer(
        harness.device, candidateBody, @"candidate body");
    id<MTLBuffer> jacobianBuffer = makeBuffer(
        harness.device, jacobian, @"attachment point Jacobian");
    id<MTLBuffer> generalizedBuffer = makeBuffer(
        harness.device, deltaV, @"candidate delta-v");
    id<MTLBuffer> acceptedBuffer = makeBuffer(
        harness.device, accepted, @"accepted attachment node");
    id<MTLBuffer> candidateBuffer = makeBuffer(
        harness.device, initialCandidate, @"candidate attachment node");
    id<MTLBuffer> rangeBuffer = makeBuffer(
        harness.device, range, @"attachment node range");
    id<MTLBuffer> objectBuffer = makeBuffer(
        harness.device, object, @"attachment object");
    id<MTLBuffer> schedulerBuffer = makeBuffer(
        harness.device, scheduler, @"attachment scheduler");
    id<MTLBuffer> statusBuffer = makeBuffer(
        harness.device, status, @"attachment status");
    id<MTLBuffer> residualBuffer = makeBuffer(
        harness.device, residual, @"attachment residual");
    id<MTLBuffer> preconditionedBuffer = makeBuffer(
        harness.device, preconditioned, @"attachment preconditioned");
    id<MTLBuffer> directionBuffer = makeBuffer(
        harness.device, direction, @"attachment mapped direction");
    id<MTLBuffer> captureDirectionBuffer = makeBuffer(
        harness.device, preconditioned, @"attachment capture direction");
    id<MTLBuffer> residualImpulseBuffer = makeBuffer(
        harness.device, zero, @"attachment residual impulse");
    id<MTLBuffer> rawReactionBuffer = makeBuffer(
        harness.device, zero, @"attachment raw reaction");
    id<MTLBuffer> operatorWorkBuffer = makeBuffer(
        harness.device, operatorWork, @"attachment operator work");
    id<MTLBuffer> attachmentOperatorBuffer = makeBuffer(
        harness.device, zero, @"attachment operator row");

    require(
        residualImpulseBuffer.gpuAddress != rawReactionBuffer.gpuAddress,
        "raw attachment reaction aliases the reduced residual impulse"
    );

    id<MTLCommandBuffer> commandBuffer = [harness.queue commandBuffer];
    require(commandBuffer != nil, "failed to create attachment command buffer");
    encode(commandBuffer, harness.drive, 1u,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&bodyCount length:sizeof(bodyCount) atIndex:1u];
            [encoder setBytes:&bodyStride length:sizeof(bodyStride) atIndex:2u];
            [encoder setBytes:&articulatedNv
                       length:sizeof(articulatedNv) atIndex:3u];
            [encoder setBuffer:attachmentBuffer offset:0u atIndex:4u];
            [encoder setBuffer:bodyBuffer offset:0u atIndex:5u];
            [encoder setBuffer:jacobianBuffer offset:0u atIndex:6u];
            [encoder setBuffer:generalizedBuffer offset:0u atIndex:7u];
            [encoder setBuffer:acceptedBuffer offset:0u atIndex:8u];
            [encoder setBuffer:candidateBuffer offset:0u atIndex:9u];
            [encoder setBuffer:rangeBuffer offset:0u atIndex:10u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:11u];
        });
    encode(commandBuffer, harness.map, 1u,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&articulatedNv
                       length:sizeof(articulatedNv) atIndex:1u];
            [encoder setBuffer:attachmentBuffer offset:0u atIndex:2u];
            [encoder setBuffer:jacobianBuffer offset:0u atIndex:3u];
            [encoder setBuffer:directionBuffer offset:0u atIndex:4u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:5u];
        });
    encode(commandBuffer, harness.captureResidual, 1u,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:attachmentBuffer offset:0u atIndex:1u];
            [encoder setBuffer:objectBuffer offset:0u atIndex:2u];
            [encoder setBuffer:rangeBuffer offset:0u atIndex:3u];
            [encoder setBuffer:schedulerBuffer offset:0u atIndex:4u];
            [encoder setBuffer:candidateBuffer offset:0u atIndex:5u];
            [encoder setBuffer:residualBuffer offset:0u atIndex:6u];
            [encoder setBuffer:preconditionedBuffer offset:0u atIndex:7u];
            [encoder setBuffer:captureDirectionBuffer offset:0u atIndex:8u];
            [encoder setBuffer:residualImpulseBuffer offset:0u atIndex:9u];
            [encoder setBuffer:rawReactionBuffer offset:0u atIndex:10u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:11u];
        });
    encode(commandBuffer, harness.scatterResidual, kDofs,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&articulatedNv
                       length:sizeof(articulatedNv) atIndex:1u];
            [encoder setBuffer:jacobianBuffer offset:0u atIndex:2u];
            [encoder setBuffer:residualImpulseBuffer offset:0u atIndex:3u];
            [encoder setBuffer:residualBuffer offset:0u atIndex:4u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:5u];
        });
    encode(commandBuffer, harness.captureOperator, 1u,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:attachmentBuffer offset:0u atIndex:1u];
            [encoder setBuffer:operatorWorkBuffer offset:0u atIndex:2u];
            [encoder setBuffer:attachmentOperatorBuffer offset:0u atIndex:3u];
        });
    encode(commandBuffer, harness.scatterOperator, kDofs,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&articulatedNv
                       length:sizeof(articulatedNv) atIndex:1u];
            [encoder setBuffer:jacobianBuffer offset:0u atIndex:2u];
            [encoder setBuffer:attachmentOperatorBuffer offset:0u atIndex:3u];
            [encoder setBuffer:operatorWorkBuffer offset:0u atIndex:4u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:5u];
        });
    finish(commandBuffer);

    ValidRun result;
    result.candidate = value<NMFEMNodeStateGPU>(candidateBuffer);
    result.status = value<NMMatterStatusGPU>(statusBuffer);
    std::memcpy(result.residual.data(), residualBuffer.contents,
        residualBuffer.length);
    std::memcpy(result.mapped.data(), directionBuffer.contents,
        directionBuffer.length);
    std::memcpy(result.operatorWork.data(), operatorWorkBuffer.contents,
        operatorWorkBuffer.length);
    result.residualImpulse = value<nm_float4>(residualImpulseBuffer);
    result.rawReaction = value<nm_float4>(rawReactionBuffer);
    result.attachmentOperator = value<nm_float4>(attachmentOperatorBuffer);
    for (id<MTLBuffer> buffer in @[
             candidateBuffer, statusBuffer, residualBuffer, directionBuffer,
             operatorWorkBuffer, residualImpulseBuffer, rawReactionBuffer,
             attachmentOperatorBuffer]) {
        appendBytes(result.replayBytes, buffer);
    }
    return result;
}

void verifyValidRun(const ValidRun& run) {
    require(run.status.code == NM_STATUS_SUCCESS,
        "valid attachment dispatch failed");
    requireClose(run.candidate.positionAndMass.x, 1.3f, 1.0e-6f,
        "attached position x");
    requireClose(run.candidate.positionAndMass.y, 1.8f, 1.0e-6f,
        "attached position y");
    requireClose(run.candidate.positionAndMass.z, 3.1f, 1.0e-6f,
        "attached position z");

    const auto jacobian = pointJacobian();
    const auto direction = generalizedDirection();
    const auto pointDirection = applyJacobian(jacobian, direction);
    requireClose(run.candidate.velocityAndInverseMass.x,
        pointDirection[0], 1.0e-6f, "attached velocity x");
    requireClose(run.candidate.velocityAndInverseMass.y,
        pointDirection[1], 1.0e-6f, "attached velocity y");
    requireClose(run.candidate.velocityAndInverseMass.z,
        pointDirection[2], 1.0e-6f, "attached velocity z");
    requireClose(run.candidate.deltaVelocity.x,
        pointDirection[0] - 0.1f, 1.0e-6f, "attached delta velocity x");
    requireClose(run.candidate.deltaVelocity.y,
        pointDirection[1] - 0.2f, 1.0e-6f, "attached delta velocity y");
    requireClose(run.candidate.deltaVelocity.z,
        pointDirection[2] - 0.3f, 1.0e-6f, "attached delta velocity z");

    // Exact body-point kinematics: the analytic point velocity must be the
    // finite difference of the returned candidate pose, including rotation.
    const std::array<float, 3> localPoint{0.3f, -0.2f, 0.1f};
    const std::array<float, 3> basePosition{1.0f, 2.0f, 3.0f};
    const std::array<float, 3> linear{
        direction[0], direction[1], direction[2]
    };
    const std::array<float, 3> angular{
        direction[3], direction[4], direction[5]
    };
    const auto increment = rotationIncrement(angular, kFiniteDifferenceStep);
    const auto basePoint = add(basePosition, localPoint);
    const auto perturbedPoint = add(
        add(basePosition, scale(kFiniteDifferenceStep, linear)),
        rotate(increment, localPoint)
    );
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        const float finiteDifference =
            (perturbedPoint[component] - basePoint[component]) /
            kFiniteDifferenceStep;
        requireClose(finiteDifference, pointDirection[component], 8.0e-4f,
            "finite-difference attachment kinematics");
    }

    const std::array<float, 3> impulse{2.0f, -3.0f, 4.0f};
    requireClose(run.residualImpulse.x, impulse[0], 1.0e-7f,
        "attachment impulse x");
    requireClose(run.residualImpulse.y, impulse[1], 1.0e-7f,
        "attachment impulse y");
    requireClose(run.residualImpulse.z, impulse[2], 1.0e-7f,
        "attachment impulse z");
    requireClose(run.rawReaction.x, impulse[0] / kTimestep, 1.0e-7f,
        "raw reaction x");
    requireClose(run.rawReaction.y, impulse[1] / kTimestep, 1.0e-7f,
        "raw reaction y");
    requireClose(run.rawReaction.z, impulse[2] / kTimestep, 1.0e-7f,
        "raw reaction z");
    requireClose(run.residual[0].x, 0.0f, 0.0f,
        "eliminated residual x");
    requireClose(run.residual[0].y, 0.0f, 0.0f,
        "eliminated residual y");
    requireClose(run.residual[0].z, 0.0f, 0.0f,
        "eliminated residual z");
    requireClose(run.residual[0].w, 7.0f, 0.0f,
        "mechanical elimination changed field row");
    requireClose(run.mapped[0].x, pointDirection[0], 1.0e-6f,
        "J p x");
    requireClose(run.mapped[0].y, pointDirection[1], 1.0e-6f,
        "J p y");
    requireClose(run.mapped[0].z, pointDirection[2], 1.0e-6f,
        "J p z");

    std::array<float, kDofs> transposedImpulse{};
    std::array<float, kDofs> transposedOperator{};
    for (std::uint32_t dof = 0u; dof < kDofs; ++dof) {
        transposedImpulse[dof] =
            jacobian[dof] * impulse[0] +
            jacobian[kDofs + dof] * impulse[1] +
            jacobian[2u * kDofs + dof] * impulse[2];
        requireClose(run.residual[2u + dof].x,
            transposedImpulse[dof], 1.0e-6f, "J^T residual");
        const std::array<float, 3> operatorRow{
            2.0f * pointDirection[0],
            3.0f * pointDirection[1],
            4.0f * pointDirection[2],
        };
        transposedOperator[dof] =
            jacobian[dof] * operatorRow[0] +
            jacobian[kDofs + dof] * operatorRow[1] +
            jacobian[2u * kDofs + dof] * operatorRow[2];
        requireClose(run.operatorWork[2u + dof].x,
            transposedOperator[dof], 1.0e-6f, "J^T A J operator");
    }
    requireClose(run.operatorWork[0].x, 0.0f, 0.0f,
        "operator attachment row was not eliminated");
    requireClose(run.operatorWork[0].y, 0.0f, 0.0f,
        "operator attachment row was not eliminated");
    requireClose(run.operatorWork[0].z, 0.0f, 0.0f,
        "operator attachment row was not eliminated");

    float generalizedVirtualWork = 0.0f;
    float generalizedQuadraticWork = 0.0f;
    for (std::uint32_t dof = 0u; dof < kDofs; ++dof) {
        generalizedVirtualWork +=
            direction[dof] * transposedImpulse[dof];
        generalizedQuadraticWork +=
            direction[dof] * transposedOperator[dof];
    }
    requireClose(generalizedVirtualWork,
        dot(pointDirection, impulse), 1.0e-6f,
        "attachment virtual work");
    const std::array<float, 3> operatorRow{
        2.0f * pointDirection[0],
        3.0f * pointDirection[1],
        4.0f * pointDirection[2],
    };
    requireClose(generalizedQuadraticWork,
        dot(pointDirection, operatorRow), 1.0e-6f,
        "attachment quadratic virtual work");
}

struct DriveRun {
    NMFEMNodeStateGPU before{};
    NMFEMNodeStateGPU after{};
    NMMatterStatusGPU beforeStatus{};
    NMMatterStatusGPU afterStatus{};
};

DriveRun runDriveCase(
    const Harness& harness,
    const std::uint32_t count,
    const std::uint32_t bodyIndex
) {
    const NMMatterDispatchGPU dispatch = attachmentDispatch(count);
    const std::uint32_t bodyCount = kBodyCount;
    const std::uint32_t bodyStride = kBodyCount;
    const std::uint32_t articulatedNv = kDofs;
    const NMFEMHumanAttachmentGPU binding = attachment(bodyIndex);
    const NMFEMNodeStateGPU accepted = node();
    NMFEMNodeStateGPU candidate = node();
    candidate.positionAndMass = f4(91.0f, 92.0f, 93.0f, 2.0f);
    const MRBodyStateGPU candidateBody = body();
    const auto jacobian = pointJacobian();
    std::array<float, kDofs> generalized{};
    const NMIncidenceRangeGPU range{0u, 0u, 0u, 0u};
    NMMatterStatusGPU status{};
    status.code = NM_STATUS_SUCCESS;
    status.environment = 0u;

    id<MTLBuffer> dispatchBuffer = makeBuffer(
        harness.device, dispatch, @"drive-case dispatch");
    id<MTLBuffer> attachmentBuffer = makeBuffer(
        harness.device, binding, @"drive-case attachment");
    id<MTLBuffer> bodyBuffer = makeBuffer(
        harness.device, candidateBody, @"drive-case body");
    id<MTLBuffer> jacobianBuffer = makeBuffer(
        harness.device, jacobian, @"drive-case Jacobian");
    id<MTLBuffer> generalizedBuffer = makeBuffer(
        harness.device, generalized, @"drive-case generalized");
    id<MTLBuffer> acceptedBuffer = makeBuffer(
        harness.device, accepted, @"drive-case accepted");
    id<MTLBuffer> candidateBuffer = makeBuffer(
        harness.device, candidate, @"drive-case candidate");
    id<MTLBuffer> rangeBuffer = makeBuffer(
        harness.device, range, @"drive-case range");
    id<MTLBuffer> statusBuffer = makeBuffer(
        harness.device, status, @"drive-case status");

    id<MTLCommandBuffer> commandBuffer = [harness.queue commandBuffer];
    encode(commandBuffer, harness.drive, 1u,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&bodyCount length:sizeof(bodyCount) atIndex:1u];
            [encoder setBytes:&bodyStride length:sizeof(bodyStride) atIndex:2u];
            [encoder setBytes:&articulatedNv
                       length:sizeof(articulatedNv) atIndex:3u];
            [encoder setBuffer:attachmentBuffer offset:0u atIndex:4u];
            [encoder setBuffer:bodyBuffer offset:0u atIndex:5u];
            [encoder setBuffer:jacobianBuffer offset:0u atIndex:6u];
            [encoder setBuffer:generalizedBuffer offset:0u atIndex:7u];
            [encoder setBuffer:acceptedBuffer offset:0u atIndex:8u];
            [encoder setBuffer:candidateBuffer offset:0u atIndex:9u];
            [encoder setBuffer:rangeBuffer offset:0u atIndex:10u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:11u];
        });
    finish(commandBuffer);
    return {
        .before = candidate,
        .after = value<NMFEMNodeStateGPU>(candidateBuffer),
        .beforeStatus = status,
        .afterStatus = value<NMMatterStatusGPU>(statusBuffer),
    };
}

void verifyZeroAndMalformed(const Harness& harness) {
    const DriveRun zero = runDriveCase(harness, 0u, NM_INVALID_INDEX);
    require(std::memcmp(&zero.before, &zero.after, sizeof(zero.before)) == 0,
        "zero-attachment dispatch changed the candidate node");
    require(std::memcmp(
        &zero.beforeStatus, &zero.afterStatus, sizeof(zero.beforeStatus)) == 0,
        "zero-attachment dispatch changed status");

    const DriveRun malformed = runDriveCase(harness, 1u, 1u);
    require(std::memcmp(
        &malformed.before, &malformed.after, sizeof(malformed.before)) == 0,
        "malformed attachment partially changed the candidate node");
    require(malformed.afterStatus.code == NM_STATUS_INVALID_DISPATCH,
        "malformed attachment did not fail closed");
}

void verifyFailureMask(const Harness& harness) {
    const NMMatterDispatchGPU dispatch = attachmentDispatch();
    NMMatterStatusGPU status{};
    status.code = NM_STATUS_NONFINITE_RESULT;
    const nm_float4 impulse = f4(1.0f, 2.0f, 3.0f, 4.0f);
    const nm_float4 raw = f4(5.0f, 6.0f, 7.0f, 8.0f);
    id<MTLBuffer> dispatchBuffer = makeBuffer(
        harness.device, dispatch, @"mask dispatch");
    id<MTLBuffer> statusBuffer = makeBuffer(
        harness.device, status, @"mask status");
    id<MTLBuffer> impulseBuffer = makeBuffer(
        harness.device, impulse, @"mask impulse");
    id<MTLBuffer> rawBuffer = makeBuffer(
        harness.device, raw, @"mask raw reaction");
    id<MTLCommandBuffer> commandBuffer = [harness.queue commandBuffer];
    encode(commandBuffer, harness.maskReactions, 1u,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:1u];
            [encoder setBuffer:impulseBuffer offset:0u atIndex:2u];
            [encoder setBuffer:rawBuffer offset:0u atIndex:3u];
        });
    finish(commandBuffer);
    const nm_float4 maskedImpulse = value<nm_float4>(impulseBuffer);
    const nm_float4 maskedRaw = value<nm_float4>(rawBuffer);
    require(maskedImpulse.x == 0.0f && maskedImpulse.y == 0.0f &&
            maskedImpulse.z == 0.0f && maskedImpulse.w == 0.0f &&
            maskedRaw.x == 0.0f && maskedRaw.y == 0.0f &&
            maskedRaw.z == 0.0f && maskedRaw.w == 0.0f,
        "failed transaction retained attachment reactions");
}

numi::matter::CompiledWorld compileRuntimeWorld(const bool withAttachment) {
    const auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "Matter probe material did not parse");

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 240.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.articulatedDofCapacity = kDofs;
    source.articulatedQCapacity = kDofs + 1u;
    source.materials.push_back(parsed.material);

    numi::matter::ObjectSource object;
    object.name = withAttachment ? "attached_fem" : "unattached_fem";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.characteristicLength = 0.01;
    object.mixedFEM = false;
    object.femNodes = {
        {-0.005, -0.005, -0.005},
        { 0.005, -0.005, -0.005},
        {-0.005,  0.005, -0.005},
        {-0.005, -0.005,  0.005},
    };
    object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    if (withAttachment) {
        numi::matter::FEMHumanAttachmentSource binding;
        binding.node = 0u;
        binding.bodyIndex = 0u;
        binding.stableIdentifier = 0x41545431u;
        // The accepted Runtime fixture begins exactly at rest: an identity
        // root body owns node zero at this body-local point.
        binding.localPoint = {-0.005, -0.005, -0.005};
        object.femHumanAttachments.push_back(binding);
    }
    source.objects.push_back(std::move(object));

    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), withAttachment
        ? "attached Matter world did not compile"
        : "zero-attachment Matter world did not compile");
    std::string layoutError;
    require(numi::matter::validateCompiledWorldLayout(
        compiled.world, &layoutError),
        "compiled Matter attachment layout failed validation: " + layoutError);
    return std::move(compiled.world);
}

struct CallbackAudit {
    bool called = false;
    numi::matter::CoupledCandidateOperation operation =
        numi::matter::CoupledCandidateOperation::massAction;
};

struct AcceptedCandidateService {
    id<MTLCommandBuffer> commandBuffer = nil;
    id<MTLBuffer> candidateQ = nil;
    id<MTLBuffer> candidateBody = nil;
    id<MTLBuffer> pointJacobian = nil;
    id<MTLBuffer> inverseStatus = nil;
    std::array<std::uint32_t, 4u> operationCounts{};
    std::uint32_t attachmentKinematics = 0u;
    std::uint32_t genericPointKinematics = 0u;
    void* attachmentPointJacobians = nullptr;
    bool invalidQuery = false;
    std::string diagnostic;
};

bool copyWithBlit(
    id<MTLBlitCommandEncoder> blit,
    id<MTLBuffer> source,
    void* destinationPointer,
    const NSUInteger bytes
) {
    id<MTLBuffer> destination = destinationPointer == nullptr
        ? nil
        : (__bridge id<MTLBuffer>)destinationPointer;
    if (blit == nil || source == nil || destination == nil || bytes == 0u ||
        source.device.registryID != destination.device.registryID ||
        source.length < bytes || destination.length < bytes) {
        return false;
    }
    [blit copyFromBuffer:source
            sourceOffset:0u
                toBuffer:destination
       destinationOffset:0u
                    size:bytes];
    return true;
}

bool zeroWithBlit(
    id<MTLBlitCommandEncoder> blit,
    void* destinationPointer,
    const NSUInteger bytes
) {
    id<MTLBuffer> destination = destinationPointer == nullptr
        ? nil
        : (__bridge id<MTLBuffer>)destinationPointer;
    if (blit == nil || destination == nil || bytes == 0u ||
        destination.length < bytes) {
        return false;
    }
    [blit fillBuffer:destination
               range:NSMakeRange(0u, bytes)
               value:0u];
    return true;
}

bool acceptCandidate(
    void* context,
    const numi::matter::CoupledCandidateQuery& query
) {
    auto* service = static_cast<AcceptedCandidateService*>(context);
    if (service == nullptr || service->commandBuffer == nil) return false;
    const std::uint32_t operation = static_cast<std::uint32_t>(query.operation);
    if (operation >= service->operationCounts.size()) {
        service->invalidQuery = true;
        return false;
    }
    ++service->operationCounts[operation];

    id<MTLBlitCommandEncoder> blit =
        [service->commandBuffer blitCommandEncoder];
    if (blit == nil) {
        service->invalidQuery = true;
        return false;
    }
    bool valid = true;
    switch (query.operation) {
    case numi::matter::CoupledCandidateOperation::candidateKinematics:
        valid = query.input != nullptr &&
            query.generalizedVectorStride == kDofs &&
            query.candidateQStride == kDofs + 1u &&
            query.candidateBodyStride == kBodyCount &&
            copyWithBlit(
                blit,
                service->candidateQ,
                query.candidateQ,
                (kDofs + 1u) * sizeof(float)) &&
            copyWithBlit(
                blit,
                service->candidateBody,
                query.candidateBodies,
                sizeof(MRBodyStateGPU));
        if (valid && query.pointCount != 0u) {
            if (service->attachmentPointJacobians == nullptr) {
                service->attachmentPointJacobians = query.pointJacobians;
            }
            const bool attachmentQuery = query.pointJacobians ==
                service->attachmentPointJacobians;
            if (attachmentQuery) {
                ++service->attachmentKinematics;
            } else {
                ++service->genericPointKinematics;
            }
            const std::uint64_t expectedJacobianStride =
                static_cast<std::uint64_t>(query.pointCount) * 3u * kDofs;
            valid = query.pointStride == query.pointCount &&
                query.pointQueries != nullptr &&
                expectedJacobianStride <=
                    std::numeric_limits<std::uint32_t>::max() &&
                query.pointJacobianStride == expectedJacobianStride &&
                (attachmentQuery
                    ? query.pointCount == 1u && copyWithBlit(
                          blit,
                          service->pointJacobian,
                          query.pointJacobians,
                          3u * kDofs * sizeof(float))
                    : zeroWithBlit(
                          blit,
                          query.pointJacobians,
                          static_cast<NSUInteger>(expectedJacobianStride) *
                              sizeof(float)));
        } else if (valid) {
            valid = query.pointStride == 0u &&
                query.pointJacobianStride == 0u;
        }
        break;
    case numi::matter::CoupledCandidateOperation::massAction:
        valid = query.input != nullptr && query.output != nullptr &&
            query.generalizedVectorStride == kDofs &&
            copyWithBlit(
                blit,
                (__bridge id<MTLBuffer>)query.input,
                query.output,
                kDofs * sizeof(float));
        break;
    case numi::matter::CoupledCandidateOperation::inverseMassPreconditioner:
        valid = query.input != nullptr && query.output != nullptr &&
            query.statuses != nullptr && query.statusStride == 1u &&
            query.generalizedVectorStride == kDofs &&
            copyWithBlit(
                blit,
                (__bridge id<MTLBuffer>)query.input,
                query.output,
                kDofs * sizeof(float)) &&
            copyWithBlit(
                blit,
                service->inverseStatus,
                query.statuses,
                sizeof(MRInverseMassStatusGPU));
        break;
    case numi::matter::CoupledCandidateOperation::publishCandidate:
        valid = query.input != nullptr && query.output != nullptr &&
            query.candidateQ != nullptr &&
            query.generalizedVectorStride == kDofs &&
            query.candidateQStride == kDofs + 1u &&
            copyWithBlit(
                blit,
                (__bridge id<MTLBuffer>)query.input,
                query.output,
                kDofs * sizeof(float));
        break;
    }
    [blit endEncoding];
    service->invalidQuery = service->invalidQuery || !valid;
    if (!valid) {
        id<MTLBuffer> candidateQDestination = query.candidateQ == nullptr
            ? nil
            : (__bridge id<MTLBuffer>)query.candidateQ;
        id<MTLBuffer> candidateBodyDestination =
            query.candidateBodies == nullptr
            ? nil
            : (__bridge id<MTLBuffer>)query.candidateBodies;
        service->diagnostic =
            "operation=" + std::to_string(operation) +
            " input=" + std::to_string(query.input != nullptr) +
            " q=" + std::to_string(query.candidateQ != nullptr) +
            " qBytes=" + std::to_string(candidateQDestination.length) +
            " bodies=" +
                std::to_string(query.candidateBodies != nullptr) +
            " bodyBytes=" +
                std::to_string(candidateBodyDestination.length) +
            " vectorStride=" +
                std::to_string(query.generalizedVectorStride) +
            " qStride=" + std::to_string(query.candidateQStride) +
            " bodyStride=" + std::to_string(query.candidateBodyStride) +
            " statusStride=" + std::to_string(query.statusStride) +
            " pointCount=" + std::to_string(query.pointCount) +
            " pointStride=" + std::to_string(query.pointStride) +
            " pointJacobianStride=" +
                std::to_string(query.pointJacobianStride);
    }
    return valid;
}

bool rejectCandidate(
    void* context,
    const numi::matter::CoupledCandidateQuery& query
) {
    auto* audit = static_cast<CallbackAudit*>(context);
    if (audit != nullptr) {
        audit->called = true;
        audit->operation = query.operation;
    }
    return false;
}

void verifyRuntimeAdmission(const Harness& harness) {
    auto zeroWorld = compileRuntimeWorld(false);
    require(zeroWorld.dispatch.femHumanAttachmentCount == 0u,
        "zero world cooked an attachment");
    numi::matter::RuntimeConfiguration configuration;
    configuration.metallib = NUMI_MATTER_METALLIB;
    configuration.environmentCount = 1u;
    configuration.captureEvents = false;
    configuration.captureDiagnostics = false;
    configuration.adaptiveTransfer = false;

    numi::matter::Runtime zeroRuntime;
    const auto zeroInitialization =
        zeroRuntime.initialize(zeroWorld, configuration);
    require(zeroInitialization.encoded,
        "zero-attachment runtime initialization failed: " +
            zeroInitialization.message);
    require(!zeroRuntime.requiresCoupledCandidate(),
        "zero-attachment world activated coupled candidate authority");
    require(zeroRuntime.coupledCandidatePointCapacity() == 0u,
        "zero-attachment world retained candidate point storage");

    auto attachedWorld = compileRuntimeWorld(true);
    require(attachedWorld.dispatch.femHumanAttachmentCount == 1u,
        "attached world lost its cooked attachment");
    require(attachedWorld.dispatch.rigidGeneralizedCapacity == kDofs &&
            attachedWorld.dispatch.rigidQCapacity == kDofs + 1u,
        "attached world did not retain exact authored candidate capacities");
    numi::matter::Runtime attachedRuntime;
    const auto attachedInitialization =
        attachedRuntime.initialize(attachedWorld, configuration);
    require(attachedInitialization.encoded,
        "attachment runtime initialization failed: " +
            attachedInitialization.message);
    require(attachedRuntime.requiresCoupledCandidate(),
        "attachment world did not activate candidate authority");
    require(attachedRuntime.coupledCandidatePointCapacity() == 1u,
        "attachment runtime reports wrong point capacity");
    require(!attachedRuntime.requiresBodyWrenches(),
        "attachment-only world manufactured rigid proxy wrenches");

    std::array<float, kDofs + 1u> q{};
    q[6] = 1.0f;
    std::array<float, kDofs> v{};
    MRMetalWorldStatusGPU worldStatus{};
    worldStatus.code = MR_STEP_SUCCESS;
    id<MTLBuffer> qBuffer = makeBuffer(harness.device, q, @"runtime q");
    id<MTLBuffer> vBuffer = makeBuffer(harness.device, v, @"runtime v");
    id<MTLBuffer> worldStatusBuffer = makeBuffer(
        harness.device, worldStatus, @"runtime world status");

    CallbackAudit audit;
    id<MTLCommandBuffer> commandBuffer = [harness.queue commandBuffer];
    require(commandBuffer != nil,
        "failed to create runtime-admission command buffer");
    numi::matter::EncodeRequest request;
    request.commandBuffer = (__bridge void*)commandBuffer;
    request.phase = numi::matter::EncodePhase::preDynamics;
    request.rigid.q = (__bridge void*)qBuffer;
    request.rigid.v = (__bridge void*)vBuffer;
    request.rigid.currentBodies = nullptr;
    request.rigid.currentBodyCount = kBodyCount;
    request.rigid.currentBodyStride = kBodyCount;
    request.rigid.qStride = kDofs + 1u;
    request.rigid.vStride = kDofs;
    request.environmentStatuses = (__bridge void*)worldStatusBuffer;
    request.coupledCandidateContext = &audit;
    request.encodeCoupledCandidate = rejectCandidate;
    request.physicsSubstep = 0u;
    request.physicsSubsteps = 1u;
    request.timestepSeconds = static_cast<float>(1.0 / 240.0);
    const auto encoded = attachedRuntime.encode(request);
    require(!encoded.encoded && audit.called,
        "attachment-only runtime did not reach the candidate callback");
    require(audit.operation ==
            numi::matter::CoupledCandidateOperation::candidateKinematics,
        "attachment-only runtime requested the wrong first candidate operation");
    require(encoded.message.find("FEM attachment kinematics") !=
            std::string::npos,
        "attachment callback rejection returned the wrong typed diagnostic: " +
            encoded.message);
    // The callback was reached with no MRBodyStateGPU current-body arena.
    // Only count/stride metadata was needed to allocate its private exact
    // candidate-body output. Complete the deliberately rejected partial
    // command buffer solely at this probe boundary.
    finish(commandBuffer);

    // Exercise the owning Runtime graph, not only the attachment kernels. The
    // synthetic articulated service is an exact identity mass/inverse owner at
    // a rest-state body point and encodes into Runtime's borrowed command
    // buffer. A successful transaction must traverse Newton reassembly,
    // restarted-FGMRES preconditioning/operator work, final certification,
    // staged Human publication, and post-commit reconciliation.
    std::array<float, kDofs + 1u> candidateQ{};
    candidateQ[6] = 1.0f;
    MRBodyStateGPU candidateBody{};
    candidateBody.position = {0.0f, 0.0f, 0.0f, 1.0f};
    candidateBody.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    candidateBody.linearVelocityAndInverseMass =
        {0.0f, 0.0f, 0.0f, 1.0f};
    candidateBody.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
    candidateBody.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    candidateBody.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    candidateBody.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    const std::array<float, 3u * kDofs> runtimeJacobian{
        1.0f, 0.0f, 0.0f,  0.0f, -0.005f,  0.005f,
        0.0f, 1.0f, 0.0f,  0.005f, 0.0f, -0.005f,
        0.0f, 0.0f, 1.0f, -0.005f, 0.005f, 0.0f,
    };
    MRInverseMassStatusGPU inverseStatus{};
    inverseStatus.code = MR_INVERSE_MASS_SUCCESS;
    inverseStatus.environment = 0u;
    inverseStatus.articulationIndex = 0u;
    inverseStatus.failingIndex = MR_INVALID_INDEX;
    inverseStatus.bodyCount = kBodyCount;
    inverseStatus.nq = kDofs + 1u;
    inverseStatus.nv = kDofs;
    inverseStatus.rhsCount = 1u;
    inverseStatus.diagnostics = {1.0f, 1.0f, 0.0f, 0.0f};

    AcceptedCandidateService service;
    service.candidateQ = makeBuffer(
        harness.device, candidateQ, @"runtime candidate q source");
    service.candidateBody = makeBuffer(
        harness.device, candidateBody, @"runtime candidate body source");
    service.pointJacobian = makeBuffer(
        harness.device, runtimeJacobian, @"runtime point Jacobian source");
    service.inverseStatus = makeBuffer(
        harness.device, inverseStatus, @"runtime inverse status source");

    MRMetalWorldStatusGPU acceptedWorldStatus{};
    acceptedWorldStatus.code = MR_STEP_SUCCESS;
    id<MTLBuffer> acceptedWorldStatusBuffer = makeBuffer(
        harness.device, acceptedWorldStatus, @"accepted runtime world status");
    id<MTLCommandBuffer> acceptedCommand = [harness.queue commandBuffer];
    require(acceptedCommand != nil,
        "failed to create accepted Runtime command buffer");
    service.commandBuffer = acceptedCommand;

    numi::matter::EncodeRequest acceptedRequest;
    acceptedRequest.commandBuffer = (__bridge void*)acceptedCommand;
    acceptedRequest.phase = numi::matter::EncodePhase::preDynamics;
    acceptedRequest.rigid.q = (__bridge void*)qBuffer;
    acceptedRequest.rigid.v = (__bridge void*)vBuffer;
    acceptedRequest.rigid.currentBodies = nullptr;
    acceptedRequest.rigid.currentBodyCount = kBodyCount;
    acceptedRequest.rigid.currentBodyStride = kBodyCount;
    acceptedRequest.rigid.qStride = kDofs + 1u;
    acceptedRequest.rigid.vStride = kDofs;
    acceptedRequest.environmentStatuses =
        (__bridge void*)acceptedWorldStatusBuffer;
    acceptedRequest.coupledCandidateContext = &service;
    acceptedRequest.encodeCoupledCandidate = acceptCandidate;
    acceptedRequest.physicsSubstep = 0u;
    acceptedRequest.physicsSubsteps = 1u;
    acceptedRequest.timestepSeconds = attachedRuntime.timestepSeconds();
    const auto acceptedPre = attachedRuntime.encode(acceptedRequest);
    require(acceptedPre.encoded,
        "accepted attachment Runtime pre-dynamics failed: " +
            acceptedPre.message + " (" + service.diagnostic + ")");
    acceptedRequest.phase = numi::matter::EncodePhase::postCommit;
    const auto acceptedPost = attachedRuntime.encode(acceptedRequest);
    require(acceptedPost.encoded,
        "accepted attachment Runtime post-commit failed: " +
            acceptedPost.message);
    finish(acceptedCommand);

    require(!service.invalidQuery && service.attachmentKinematics != 0u,
        "accepted Runtime did not issue a valid nonzero attachment kinematics callback");
    require(
        service.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::candidateKinematics)] > 0u &&
        service.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::massAction)] > 0u &&
        service.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::inverseMassPreconditioner)] > 0u &&
        service.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::publishCandidate)] == 1u,
        "accepted Runtime did not traverse every coupled-candidate operation"
    );
    const auto acceptedSnapshot = attachedRuntime.snapshot();
    require(acceptedSnapshot.available && acceptedSnapshot.statuses.size() == 1u &&
            acceptedSnapshot.statuses.front().code == NM_STATUS_SUCCESS &&
            acceptedSnapshot.statuses.front().completedMicrosteps == 1u,
        "accepted Runtime did not retain one successful completed microstep");
    require(acceptedSnapshot.solverCertificates.size() == 1u &&
            acceptedSnapshot.solverCertificates.front().validity.w > 0.5f,
        "accepted Runtime did not retain an accepted final solver certificate");
    require(acceptedSnapshot.femNodes.size() == 4u,
        "accepted Runtime snapshot has the wrong FEM node count");
    requireClose(acceptedSnapshot.femNodes.front().positionAndMass.x,
        -0.005f, 1.0e-6f, "accepted Runtime attachment position x");
    requireClose(acceptedSnapshot.femNodes.front().positionAndMass.y,
        -0.005f, 1.0e-6f, "accepted Runtime attachment position y");
    requireClose(acceptedSnapshot.femNodes.front().positionAndMass.z,
        -0.005f, 1.0e-6f, "accepted Runtime attachment position z");
    require(value<MRMetalWorldStatusGPU>(acceptedWorldStatusBuffer).code ==
            MR_STEP_SUCCESS,
        "accepted Runtime transaction poisoned enclosing MetalWorld status");

    // Candidate-body capacity is structural and fails before any GPU work.
    CallbackAudit undersizedAudit;
    id<MTLCommandBuffer> undersizedCommand = [harness.queue commandBuffer];
    request.commandBuffer = (__bridge void*)undersizedCommand;
    request.coupledCandidateContext = &undersizedAudit;
    request.rigid.currentBodyCount = 0u;
    request.rigid.currentBodyStride = 0u;
    const auto undersized = attachedRuntime.encode(request);
    require(!undersized.encoded && !undersizedAudit.called &&
            undersized.message.find("candidate-body capacity") !=
                std::string::npos,
        "undersized candidate-body metadata did not fail closed");
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            const Harness harness = makeHarness();
            verifyZeroAndMalformed(harness);
            verifyFailureMask(harness);
            const ValidRun first = runValid(harness);
            const ValidRun second = runValid(harness);
            verifyValidRun(first);
            verifyValidRun(second);
            require(first.replayBytes == second.replayBytes,
                "attachment GPU replay was not byte-identical");
            verifyRuntimeAdmission(harness);
            std::cout
                << "NumanX Matter attachment runtime probe passed on "
                << stringValue(harness.device.name)
                << ": zero-count identity, exact moving point kinematics, "
                   "fail-closed malformed input, separate raw reactions, "
                   "J/J^T virtual work, rollback masking, accepted Runtime "
                   "Newton/certification/publish, and byte replay\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "NumanX Matter attachment runtime probe failed: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
