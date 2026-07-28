#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/FreeBodyDynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace {

constexpr std::size_t kBodyCount = 3u;
constexpr double kStateAbsoluteTolerance = 8.0e-5;
constexpr double kStateRelativeTolerance = 8.0e-5;
constexpr double kConservationTolerance = 5.0e-5;

using Properties = std::array<MRBodyPropertiesGPU, kBodyCount>;
using States = std::array<MRBodyStateGPU, kBodyCount>;
using Wrenches = std::array<MRBodyWrenchGPU, kBodyCount>;
using Statuses = std::array<MRFreeBodyStatusGPU, kBodyCount>;

struct Vec3 {
    double x;
    double y;
    double z;
};

struct MetalResult {
    States states{};
    Statuses statuses{};
    std::string deviceName;
};

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

double norm(const Vec3 value) {
    return std::sqrt(
        value.x * value.x +
        value.y * value.y +
        value.z * value.z
    );
}

double quaternionNorm(const mr_float4 value) {
    return std::sqrt(
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w
    );
}

bool finite4(const mr_float4 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

bool finiteState(const MRBodyStateGPU& state) {
    return finite4(state.position) &&
        finite4(state.orientation) &&
        finite4(state.linearVelocityAndInverseMass) &&
        finite4(state.angularVelocity) &&
        finite4(state.inverseInertiaWorldRow0) &&
        finite4(state.inverseInertiaWorldRow1) &&
        finite4(state.inverseInertiaWorldRow2);
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool nearlyEqual(
    const double left,
    const double right,
    const double absoluteTolerance = kStateAbsoluteTolerance,
    const double relativeTolerance = kStateRelativeTolerance
) {
    return std::abs(left - right) <=
        absoluteTolerance +
        relativeTolerance * std::max(std::abs(left), std::abs(right));
}

MRBodyPropertiesGPU makeProperties(
    const MRMotionType motionType,
    const float mass,
    const Vec3 inertia,
    const float linearDamping = 0.0f,
    const float angularDamping = 0.0f,
    const float maximumLinearSpeed = 1.0e6f,
    const float maximumAngularSpeed = 1.0e6f
) {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    body.motionType = motionType;
    const float inverseMass = mass > 0.0f ? 1.0f / mass : 0.0f;
    body.massAndInverseMass = f4(mass, inverseMass, 0.0f, 0.0f);
    body.inertiaRow0 =
        f4(static_cast<float>(inertia.x), 0.0f, 0.0f);
    body.inertiaRow1 =
        f4(0.0f, static_cast<float>(inertia.y), 0.0f);
    body.inertiaRow2 =
        f4(0.0f, 0.0f, static_cast<float>(inertia.z));
    body.inverseInertiaRow0 =
        f4(static_cast<float>(1.0 / inertia.x), 0.0f, 0.0f);
    body.inverseInertiaRow1 =
        f4(0.0f, static_cast<float>(1.0 / inertia.y), 0.0f);
    body.inverseInertiaRow2 =
        f4(0.0f, 0.0f, static_cast<float>(1.0 / inertia.z));
    body.dampingAndSpeedLimits = f4(
        linearDamping,
        angularDamping,
        maximumLinearSpeed,
        maximumAngularSpeed
    );
    return body;
}

MRBodyStateGPU makeState(
    const MRMotionType motionType,
    const Vec3 position,
    const mr_float4 orientation,
    const Vec3 linearVelocity,
    const Vec3 angularVelocity,
    const float inverseMass,
    const MRBodyPropertiesGPU& properties,
    const std::uint32_t bodyIndex
) {
    MRBodyStateGPU state{};
    state.position = f4(
        static_cast<float>(position.x),
        static_cast<float>(position.y),
        static_cast<float>(position.z),
        1.0f
    );
    state.orientation = orientation;
    state.linearVelocityAndInverseMass = f4(
        static_cast<float>(linearVelocity.x),
        static_cast<float>(linearVelocity.y),
        static_cast<float>(linearVelocity.z),
        inverseMass
    );
    state.angularVelocity = f4(
        static_cast<float>(angularVelocity.x),
        static_cast<float>(angularVelocity.y),
        static_cast<float>(angularVelocity.z)
    );
    // The CPU and Metal paths both refresh this after a moving-body step.
    state.inverseInertiaWorldRow0 = properties.inverseInertiaRow0;
    state.inverseInertiaWorldRow1 = properties.inverseInertiaRow1;
    state.inverseInertiaWorldRow2 = properties.inverseInertiaRow2;
    state.flagsAndIndices[0] = motionType;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = bodyIndex;
    state.flagsAndIndices[3] = 0u;
    return state;
}

std::pair<Properties, States> makeScene(
    const bool useDampingAndCaps
) {
    Properties properties{
        makeProperties(
            MR_MOTION_STATIC,
            0.0f,
            {1.0, 1.0, 1.0}
        ),
        makeProperties(
            MR_MOTION_KINEMATIC,
            0.0f,
            {0.4, 0.5, 0.6}
        ),
        makeProperties(
            MR_MOTION_DYNAMIC,
            2.0f,
            {0.7, 1.1, 1.7},
            useDampingAndCaps ? 0.4f : 0.0f,
            useDampingAndCaps ? 0.3f : 0.0f,
            useDampingAndCaps ? 0.5f : 1.0e6f,
            useDampingAndCaps ? 1.0f : 1.0e6f
        ),
    };

    const float dynamicW = std::sqrt(0.86f);
    States states{
        makeState(
            MR_MOTION_STATIC,
            {0.0, 0.0, -1.0},
            f4(0.0f, 0.0f, 0.0f, 1.0f),
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            0.0f,
            properties[0],
            0u
        ),
        makeState(
            MR_MOTION_KINEMATIC,
            {-0.4, 0.2, 0.6},
            f4(0.0f, 0.0f, 0.258819045f, 0.965925826f),
            {0.35, -0.20, 0.10},
            {0.40, 0.25, -0.30},
            0.0f,
            properties[1],
            1u
        ),
        makeState(
            MR_MOTION_DYNAMIC,
            {0.3, -0.2, 1.4},
            f4(0.2f, -0.1f, 0.3f, dynamicW),
            {0.40, -0.25, 0.15},
            {1.70, -0.90, 2.20},
            0.5f,
            properties[2],
            2u
        ),
    };
    return {properties, states};
}

MRFreeBodyBatchGPU makeBatch(
    const metalrobo::FreeBodyIntegratorConfig& config
) {
    MRFreeBodyBatchGPU batch{};
    batch.bodyOffset = 0u;
    batch.bodyCount = static_cast<mr_u32>(kBodyCount);
    batch.integratorType =
        static_cast<mr_u32>(config.integrator);
    batch.nonlinearIterations = config.nonlinearIterations;
    batch.gravityAndTimestep = f4(
        config.gravity.x,
        config.gravity.y,
        config.gravity.z,
        static_cast<float>(config.timestep)
    );
    batch.convergence =
        f4(static_cast<float>(config.nonlinearTolerance), 0.0f, 0.0f);
    return batch;
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
    const std::string description = nsString(error.localizedDescription);
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

MetalResult integrateOnMetal(
    const Properties& properties,
    const States& initialStates,
    const Wrenches& wrenches,
    const MRFreeBodyBatchGPU& batch
) {
    @autoreleasepool {
        require(
            batch.bodyOffset <= properties.size() &&
                batch.bodyCount <=
                    properties.size() - batch.bodyOffset &&
                batch.bodyOffset <= initialStates.size() &&
                batch.bodyCount <=
                    initialStates.size() - batch.bodyOffset &&
                batch.bodyOffset <= wrenches.size() &&
                batch.bodyCount <=
                    wrenches.size() - batch.bodyOffset,
            "free-body batch exceeds its backing buffers"
        );
        const std::string metallibPath = METALROBO_DEFAULT_METALLIB;
        require(
            !metallibPath.empty(),
            "METALROBO_DEFAULT_METALLIB is empty"
        );
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal-capable device is available");
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
            [library newFunctionWithName:@"mr_integrate_free_bodies"];
        require(
            function != nil,
            "metallib does not contain mr_integrate_free_bodies"
        );
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        require(
            pipeline != nil,
            "failed to create free-body pipeline: " + describeError(error)
        );

        Statuses initialStatuses{};
        for (std::size_t index = 0; index < initialStatuses.size(); ++index) {
            initialStatuses[index].code = MR_STEP_UNSUPPORTED;
            initialStatuses[index].iterations = 0xffffffffu;
            initialStatuses[index].bodyIndex = 0xffffffffu;
        }
        id<MTLBuffer> propertyBuffer = makeSharedBuffer(
            device,
            properties.data(),
            properties.size(),
            @"free body properties"
        );
        id<MTLBuffer> stateBuffer = makeSharedBuffer(
            device,
            initialStates.data(),
            initialStates.size(),
            @"free body states"
        );
        id<MTLBuffer> wrenchBuffer = makeSharedBuffer(
            device,
            wrenches.data(),
            wrenches.size(),
            @"free body wrenches"
        );
        id<MTLBuffer> batchBuffer =
            makeSharedBuffer(device, &batch, 1u, @"free body batch");
        id<MTLBuffer> statusBuffer = makeSharedBuffer(
            device,
            initialStatuses.data(),
            initialStatuses.size(),
            @"free body statuses"
        );

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        require(
            commandBuffer != nil && encoder != nil,
            "failed to create Metal command encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:propertyBuffer offset:0 atIndex:0];
        [encoder setBuffer:stateBuffer offset:0 atIndex:1];
        [encoder setBuffer:wrenchBuffer offset:0 atIndex:2];
        [encoder setBuffer:batchBuffer offset:0 atIndex:3];
        [encoder setBuffer:statusBuffer offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(kBodyCount, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(kBodyCount, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(
            commandBuffer.status == MTLCommandBufferStatusCompleted,
            "free-body command failed: " +
                describeError(commandBuffer.error)
        );

        MetalResult result{};
        std::memcpy(
            result.states.data(),
            stateBuffer.contents,
            sizeof(result.states)
        );
        std::memcpy(
            result.statuses.data(),
            statusBuffer.contents,
            sizeof(result.statuses)
        );
        result.deviceName = nsString(device.name);
        return result;
    }
}

double compareState(
    const MRBodyStateGPU& cpu,
    const MRBodyStateGPU& gpu
) {
    const std::array<mr_float4, 7> cpuValues{
        cpu.position,
        cpu.orientation,
        cpu.linearVelocityAndInverseMass,
        cpu.angularVelocity,
        cpu.inverseInertiaWorldRow0,
        cpu.inverseInertiaWorldRow1,
        cpu.inverseInertiaWorldRow2,
    };
    const std::array<mr_float4, 7> gpuValues{
        gpu.position,
        gpu.orientation,
        gpu.linearVelocityAndInverseMass,
        gpu.angularVelocity,
        gpu.inverseInertiaWorldRow0,
        gpu.inverseInertiaWorldRow1,
        gpu.inverseInertiaWorldRow2,
    };
    double maximumError = 0.0;
    for (std::size_t record = 0; record < cpuValues.size(); ++record) {
        const float* cpuComponents = &cpuValues[record].x;
        const float* gpuComponents = &gpuValues[record].x;
        for (std::size_t component = 0; component < 4u; ++component) {
            maximumError = std::max(
                maximumError,
                std::abs(
                    static_cast<double>(cpuComponents[component]) -
                    gpuComponents[component]
                )
            );
            require(
                nearlyEqual(
                    cpuComponents[component],
                    gpuComponents[component]
                ),
                "CPU/Metal free-body state parity exceeded FP32 tolerance"
            );
        }
    }
    require(
        std::memcmp(
            cpu.flagsAndIndices,
            gpu.flagsAndIndices,
            sizeof(cpu.flagsAndIndices)
        ) == 0,
        "CPU/Metal body flags differ"
    );
    return maximumError;
}

void rotationMatrix(const mr_float4 q, double result[3][3]) {
    const double x = q.x;
    const double y = q.y;
    const double z = q.z;
    const double w = q.w;
    result[0][0] = 1.0 - 2.0 * (y * y + z * z);
    result[0][1] = 2.0 * (x * y - z * w);
    result[0][2] = 2.0 * (x * z + y * w);
    result[1][0] = 2.0 * (x * y + z * w);
    result[1][1] = 1.0 - 2.0 * (x * x + z * z);
    result[1][2] = 2.0 * (y * z - x * w);
    result[2][0] = 2.0 * (x * z - y * w);
    result[2][1] = 2.0 * (y * z + x * w);
    result[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

Vec3 bodyAngularVelocity(const MRBodyStateGPU& state) {
    double rotation[3][3]{};
    rotationMatrix(state.orientation, rotation);
    const double world[3]{
        state.angularVelocity.x,
        state.angularVelocity.y,
        state.angularVelocity.z,
    };
    Vec3 body{};
    double* components = &body.x;
    for (std::size_t row = 0; row < 3u; ++row) {
        for (std::size_t column = 0; column < 3u; ++column) {
            components[row] += rotation[column][row] * world[column];
        }
    }
    return body;
}

double angularEnergy(
    const MRBodyPropertiesGPU& properties,
    const MRBodyStateGPU& state
) {
    const Vec3 omega = bodyAngularVelocity(state);
    return 0.5 * (
        properties.inertiaRow0.x * omega.x * omega.x +
        properties.inertiaRow1.y * omega.y * omega.y +
        properties.inertiaRow2.z * omega.z * omega.z
    );
}

double angularMomentumNorm(
    const MRBodyPropertiesGPU& properties,
    const MRBodyStateGPU& state
) {
    const Vec3 omega = bodyAngularVelocity(state);
    const Vec3 bodyMomentum{
        properties.inertiaRow0.x * omega.x,
        properties.inertiaRow1.y * omega.y,
        properties.inertiaRow2.z * omega.z,
    };
    return norm(bodyMomentum);
}

double relativeDrift(const double initial, const double finalValue) {
    return std::abs(finalValue - initial) /
        std::max(std::abs(initial), 1.0e-12);
}

void validateStatuses(
    const Statuses& statuses,
    const bool implicit
) {
    for (std::size_t index = 0; index < statuses.size(); ++index) {
        const MRFreeBodyStatusGPU& status = statuses[index];
        require(
            status.code == MR_STEP_SUCCESS &&
                status.bodyIndex == index &&
                finite4(status.diagnostics),
            "Metal free-body status is invalid"
        );
    }
    require(
        statuses[0].iterations == 0u &&
            statuses[1].iterations == 0u,
        "static or kinematic body reported nonlinear iterations"
    );
    require(
        implicit
            ? statuses[2].iterations > 0u
            : statuses[2].iterations == 1u,
        "dynamic-body iteration status is incorrect"
    );
    require(
        statuses[1].diagnostics.y <= 2.0e-6f &&
            statuses[2].diagnostics.y <= 2.0e-6f,
        "Metal quaternion normalization status exceeded tolerance"
    );
}

} // namespace

int main() {
    try {
        const auto [implicitProperties, implicitInitial] =
            makeScene(false);
        Wrenches zeroWrenches{};
        metalrobo::FreeBodyIntegratorConfig implicitConfig;
        implicitConfig.timestep = 1.0e-3;
        implicitConfig.gravity = f4(0.0f, 0.0f, 0.0f);
        implicitConfig.integrator =
            metalrobo::FreeBodyIntegrator::implicitMidpoint;
        implicitConfig.nonlinearIterations = 16u;
        implicitConfig.nonlinearTolerance = 2.0e-6;

        States implicitCpu = implicitInitial;
        const auto implicitCpuDiagnostics =
            metalrobo::integrateFreeBodies(
                implicitProperties,
                implicitCpu,
                zeroWrenches,
                implicitConfig
            );
        require(
            implicitCpuDiagnostics.succeeded() &&
                implicitCpuDiagnostics.bodiesIntegrated == 2u,
            "CPU implicit-midpoint integration failed"
        );
        const MetalResult implicitGpu = integrateOnMetal(
            implicitProperties,
            implicitInitial,
            zeroWrenches,
            makeBatch(implicitConfig)
        );
        validateStatuses(implicitGpu.statuses, true);
        require(
            std::memcmp(
                &implicitInitial[0],
                &implicitCpu[0],
                sizeof(MRBodyStateGPU)
            ) == 0 &&
                std::memcmp(
                    &implicitInitial[0],
                    &implicitGpu.states[0],
                    sizeof(MRBodyStateGPU)
                ) == 0,
            "static body changed"
        );

        double implicitMaximumError = 0.0;
        for (std::size_t index = 0; index < kBodyCount; ++index) {
            require(
                finiteState(implicitCpu[index]) &&
                    finiteState(implicitGpu.states[index]),
                "implicit free-body state is non-finite"
            );
            implicitMaximumError = std::max(
                implicitMaximumError,
                compareState(implicitCpu[index], implicitGpu.states[index])
            );
        }
        require(
            implicitCpu[1].linearVelocityAndInverseMass.x ==
                    implicitInitial[1].linearVelocityAndInverseMass.x &&
                implicitGpu.states[1].linearVelocityAndInverseMass.x ==
                    implicitInitial[1].linearVelocityAndInverseMass.x &&
                implicitCpu[1].angularVelocity.x ==
                    implicitInitial[1].angularVelocity.x &&
                implicitGpu.states[1].angularVelocity.x ==
                    implicitInitial[1].angularVelocity.x,
            "kinematic velocity was modified"
        );

        const double initialEnergy =
            angularEnergy(implicitProperties[2], implicitInitial[2]);
        const double initialMomentum =
            angularMomentumNorm(implicitProperties[2], implicitInitial[2]);
        const double cpuEnergyDrift = relativeDrift(
            initialEnergy,
            angularEnergy(implicitProperties[2], implicitCpu[2])
        );
        const double gpuEnergyDrift = relativeDrift(
            initialEnergy,
            angularEnergy(
                implicitProperties[2],
                implicitGpu.states[2]
            )
        );
        const double cpuMomentumDrift = relativeDrift(
            initialMomentum,
            angularMomentumNorm(implicitProperties[2], implicitCpu[2])
        );
        const double gpuMomentumDrift = relativeDrift(
            initialMomentum,
            angularMomentumNorm(
                implicitProperties[2],
                implicitGpu.states[2]
            )
        );
        require(
            std::max(cpuEnergyDrift, gpuEnergyDrift) <=
                    kConservationTolerance &&
                std::max(cpuMomentumDrift, gpuMomentumDrift) <=
                    kConservationTolerance &&
                std::abs(
                    quaternionNorm(implicitGpu.states[2].orientation) -
                    1.0
                ) <= 2.0e-6,
            "torque-free midpoint conservation invariant failed"
        );
        require(
            implicitCpu[2].linearVelocityAndInverseMass.x ==
                    implicitInitial[2].linearVelocityAndInverseMass.x &&
                implicitGpu.states[2].linearVelocityAndInverseMass.x ==
                    implicitInitial[2].linearVelocityAndInverseMass.x,
            "torque-free dynamic linear momentum changed"
        );

        const auto [symplecticProperties, symplecticInitial] =
            makeScene(true);
        Wrenches drivenWrenches{};
        drivenWrenches[2].force = f4(2.0f, -1.0f, 0.5f);
        drivenWrenches[2].torque = f4(0.3f, -0.4f, 0.2f);
        metalrobo::FreeBodyIntegratorConfig symplecticConfig;
        symplecticConfig.timestep = 1.0e-2;
        symplecticConfig.gravity = f4(0.0f, 0.0f, -9.81f);
        symplecticConfig.integrator =
            metalrobo::FreeBodyIntegrator::symplecticEuler;
        symplecticConfig.nonlinearIterations = 1u;
        symplecticConfig.nonlinearTolerance = 1.0e-6;

        States symplecticCpu = symplecticInitial;
        const auto symplecticCpuDiagnostics =
            metalrobo::integrateFreeBodies(
                symplecticProperties,
                symplecticCpu,
                drivenWrenches,
                symplecticConfig
            );
        require(
            symplecticCpuDiagnostics.succeeded() &&
                symplecticCpuDiagnostics.bodiesIntegrated == 2u,
            "CPU symplectic-Euler integration failed"
        );
        const MetalResult symplecticGpu = integrateOnMetal(
            symplecticProperties,
            symplecticInitial,
            drivenWrenches,
            makeBatch(symplecticConfig)
        );
        validateStatuses(symplecticGpu.statuses, false);

        double symplecticMaximumError = 0.0;
        for (std::size_t index = 0; index < kBodyCount; ++index) {
            require(
                finiteState(symplecticCpu[index]) &&
                    finiteState(symplecticGpu.states[index]),
                "symplectic free-body state is non-finite"
            );
            symplecticMaximumError = std::max(
                symplecticMaximumError,
                compareState(
                    symplecticCpu[index],
                    symplecticGpu.states[index]
                )
            );
        }
        const auto speed = [](const mr_float4 value) {
            return std::sqrt(
                static_cast<double>(value.x) * value.x +
                static_cast<double>(value.y) * value.y +
                static_cast<double>(value.z) * value.z
            );
        };
        require(
            speed(symplecticGpu.states[2].linearVelocityAndInverseMass) <=
                    0.50001 &&
                speed(symplecticGpu.states[2].angularVelocity) <= 1.00001 &&
                speed(symplecticCpu[2].linearVelocityAndInverseMass) <=
                    0.50001 &&
                speed(symplecticCpu[2].angularVelocity) <= 1.00001,
            "damping/speed-cap integration invariant failed"
        );

        Properties mismatchedProperties = symplecticProperties;
        mismatchedProperties[2].motionType = MR_MOTION_STATIC;
        States mismatchedCpu = symplecticInitial;
        const auto mismatchedCpuDiagnostics =
            metalrobo::integrateFreeBodies(
                std::span<const MRBodyPropertiesGPU>{
                    mismatchedProperties.data() + 2u,
                    1u
                },
                std::span<MRBodyStateGPU>{
                    mismatchedCpu.data() + 2u,
                    1u
                },
                std::span<const metalrobo::BodyWrench>{
                    drivenWrenches.data() + 2u,
                    1u
                },
                symplecticConfig
            );
        require(
            mismatchedCpuDiagnostics.code == MR_STEP_NONFINITE_INPUT &&
                std::memcmp(
                    &mismatchedCpu[2],
                    &symplecticInitial[2],
                    sizeof(MRBodyStateGPU)
                ) == 0,
            "CPU motion-type mismatch contract regressed"
        );
        MRFreeBodyBatchGPU mismatchedBatch = makeBatch(symplecticConfig);
        mismatchedBatch.bodyOffset = 2u;
        mismatchedBatch.bodyCount = 1u;
        const MetalResult mismatchedGpu = integrateOnMetal(
            mismatchedProperties,
            symplecticInitial,
            drivenWrenches,
            mismatchedBatch
        );
        require(
            mismatchedGpu.statuses[0].code == MR_STEP_NONFINITE_INPUT &&
                mismatchedGpu.statuses[0].bodyIndex == 2u &&
                std::memcmp(
                    &mismatchedGpu.states[2],
                    &symplecticInitial[2],
                    sizeof(MRBodyStateGPU)
                ) == 0,
            "Metal motion-type mismatch was accepted or mutated state"
        );

        Properties zeroMassProperties = symplecticProperties;
        zeroMassProperties[2].massAndInverseMass.x = 0.0f;
        States zeroMassCpu = symplecticInitial;
        const auto zeroMassCpuDiagnostics =
            metalrobo::integrateFreeBodies(
                std::span<const MRBodyPropertiesGPU>{
                    zeroMassProperties.data() + 2u,
                    1u
                },
                std::span<MRBodyStateGPU>{
                    zeroMassCpu.data() + 2u,
                    1u
                },
                std::span<const metalrobo::BodyWrench>{
                    drivenWrenches.data() + 2u,
                    1u
                },
                symplecticConfig
            );
        require(
            zeroMassCpuDiagnostics.code == MR_STEP_NONFINITE_INPUT &&
                std::memcmp(
                    &zeroMassCpu[2],
                    &symplecticInitial[2],
                    sizeof(MRBodyStateGPU)
                ) == 0,
            "CPU non-positive dynamic property mass contract regressed"
        );
        const MetalResult zeroMassGpu = integrateOnMetal(
            zeroMassProperties,
            symplecticInitial,
            drivenWrenches,
            mismatchedBatch
        );
        require(
            zeroMassGpu.statuses[0].code == MR_STEP_NONFINITE_INPUT &&
                std::memcmp(
                    &zeroMassGpu.states[2],
                    &symplecticInitial[2],
                    sizeof(MRBodyStateGPU)
                ) == 0,
            "Metal accepted non-positive dynamic property mass"
        );

        MRFreeBodyBatchGPU wrappingBatch = makeBatch(symplecticConfig);
        wrappingBatch.bodyOffset =
            std::numeric_limits<mr_u32>::max();
        wrappingBatch.bodyCount = 2u;
        bool wrappingBatchRejected = false;
        try {
            static_cast<void>(integrateOnMetal(
                symplecticProperties,
                symplecticInitial,
                drivenWrenches,
                wrappingBatch
            ));
        } catch (const std::runtime_error&) {
            wrappingBatchRejected = true;
        }
        require(
            wrappingBatchRejected,
            "wrapping free-body batch range reached Metal buffers"
        );

        std::cout << std::scientific << std::setprecision(6)
                  << "device=\"" << implicitGpu.deviceName << "\""
                  << " bodies=" << kBodyCount
                  << " implicit_max_error=" << implicitMaximumError
                  << " symplectic_max_error=" << symplecticMaximumError
                  << " energy_drift="
                  << std::max(cpuEnergyDrift, gpuEnergyDrift)
                  << " angular_momentum_drift="
                  << std::max(cpuMomentumDrift, gpuMomentumDrift)
                  << " midpoint_iterations="
                  << implicitGpu.statuses[2].iterations
                  << " midpoint_residual="
                  << implicitGpu.statuses[2].diagnostics.x
                  << " motion_contract=yes"
                  << " range_preflight=yes"
                  << " finite=yes statuses=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_free_body_gpu_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
