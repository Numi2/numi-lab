#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/OpenSimSpatialTransform.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef METALROBO_OPENSIM_SPATIAL_METALLIB
#define METALROBO_OPENSIM_SPATIAL_METALLIB ""
#endif

namespace {

using metalrobo::OpenSimFunctionDefinition;
using metalrobo::OpenSimFunctionKind;
using metalrobo::OpenSimSpatialAxisDefinition;
using metalrobo::OpenSimSpatialTransformDefinition;
using metalrobo::OpenSimSpatialTransformEvaluation;

static_assert(sizeof(MROpenSimSpatialTransformGPU) == 2512u);
static_assert(sizeof(MROpenSimSpatialTransformResultGPU) == 464u);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string text(NSString* value) {
    return value == nil || value.UTF8String == nullptr
        ? std::string{}
        : std::string{value.UTF8String};
}

std::string errorText(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const std::string description = text(error.localizedDescription);
    return description.empty() ? text(error.description) : description;
}

template <typename T>
id<MTLBuffer> buffer(
    id<MTLDevice> device,
    const T* source,
    const std::size_t count,
    NSString* label
) {
    require(source != nullptr && count != 0u, "invalid Metal buffer source");
    require(
        count <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer size overflow"
    );
    id<MTLBuffer> result = [device
        newBufferWithBytes:source
                    length:static_cast<NSUInteger>(count * sizeof(T))
                   options:MTLResourceStorageModeShared];
    require(result != nil, "failed to allocate " + text(label));
    result.label = label;
    return result;
}

OpenSimSpatialAxisDefinition axis(
    const std::array<double, 3> direction,
    const std::uint32_t coordinate,
    const OpenSimFunctionDefinition& function
) {
    return {
        .axis = direction,
        .coordinateIndex = coordinate,
        .function = function,
    };
}

OpenSimSpatialTransformDefinition walkerKnee() {
    OpenSimSpatialTransformDefinition source{};
    source.coordinateCount = 1u;
    source.axes = {
        axis({1.0, 0.0, 0.0}, 0u,
             {.kind = OpenSimFunctionKind::linear, .coefficients = {1.0, 0.0}}),
        axis({0.0, 0.0, 1.0}, 0u,
             {.kind = OpenSimFunctionKind::polynomial,
              .coefficients = {
                  0.010832094539863, -0.025218325501241,
                  -0.032847810398852, 0.079100011967027,
                  -1.473252350900463e-08,
              }}),
        axis({0.0, 1.0, 0.0}, 0u,
             {.kind = OpenSimFunctionKind::polynomial,
              .coefficients = {
                  0.025165762727423, -0.16948005139054,
                  0.369499348688249, -4.430358308836305e-08,
              }}),
        axis({1.0, 0.0, 0.0}, 0u,
             {.kind = OpenSimFunctionKind::polynomial,
              .coefficients = {
                  0.0001590447878850381, -0.001015149915669,
                  0.001817510974968, 2.64142664519923e-05,
                  -7.746563532471892e-07,
              }}),
        axis({0.0, 1.0, 0.0}, 0u,
             {.kind = OpenSimFunctionKind::polynomial,
              .coefficients = {
                  -0.0005796878052338684, 0.005079765745626,
                  -0.011442375726364, 0.003936908668844,
                  -2.516350383213525e-05,
              }}),
        axis({0.0, 0.0, 1.0}, 0u,
             {.kind = OpenSimFunctionKind::polynomial,
              .coefficients = {
                  0.001208086889206, -0.004453611224706,
                  0.000611649407298173, 0.006265429606387,
                  -1.461912533723326e-05,
              }}),
    };
    return source;
}

struct GPUResult {
    MROpenSimSpatialTransformResultGPU result{};
    std::string deviceName;
};

GPUResult runMetal(
    const MROpenSimSpatialTransformGPU& program,
    const std::array<float, MR_OPENSIM_SPATIAL_MAX_COORDINATES>& coordinates,
    const std::array<float, MR_OPENSIM_SPATIAL_MAX_COORDINATES>& velocities
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create Metal command queue");
        NSString* libraryPath =
            [NSString stringWithUTF8String:METALROBO_OPENSIM_SPATIAL_METALLIB];
        require(
            libraryPath != nil && libraryPath.length != 0u,
            "OpenSim spatial-transform metallib path is empty"
        );
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:libraryPath]
                       error:&error];
        require(
            library != nil,
            "failed to load OpenSim spatial-transform metallib: " + errorText(error)
        );
        id<MTLFunction> function = [library
            newFunctionWithName:@"mr_opensim_spatial_transform_evaluate"];
        require(function != nil, "OpenSim spatial-transform kernel is missing");
        error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function error:&error];
        require(
            pipeline != nil,
            "failed to create OpenSim spatial-transform pipeline: " + errorText(error)
        );

        MROpenSimSpatialTransformResultGPU result{};
        id<MTLBuffer> programBuffer = buffer(device, &program, 1u, @"OpenSim program");
        id<MTLBuffer> coordinateBuffer = buffer(
            device, coordinates.data(), coordinates.size(), @"OpenSim coordinates"
        );
        id<MTLBuffer> velocityBuffer = buffer(
            device, velocities.data(), velocities.size(), @"OpenSim velocities"
        );
        id<MTLBuffer> resultBuffer = buffer(device, &result, 1u, @"OpenSim result");
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        require(
            commandBuffer != nil && encoder != nil,
            "failed to create OpenSim spatial-transform command encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:programBuffer offset:0 atIndex:0];
        [encoder setBuffer:coordinateBuffer offset:0 atIndex:1];
        [encoder setBuffer:velocityBuffer offset:0 atIndex:2];
        [encoder setBuffer:resultBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(
            commandBuffer.status == MTLCommandBufferStatusCompleted,
            "OpenSim spatial-transform command failed: " + errorText(commandBuffer.error)
        );
        GPUResult output;
        std::memcpy(&output.result, resultBuffer.contents, sizeof(output.result));
        output.deviceName = text(device.name);
        return output;
    }
}

void requireNear(
    const float actual,
    const double expected,
    const std::string& label
) {
    require(
        std::abs(static_cast<double>(actual) - expected) <= 1.0e-5,
        label + " mismatch"
    );
}

void compare(
    const MROpenSimSpatialTransformResultGPU& gpu,
    const OpenSimSpatialTransformEvaluation& cpu
) {
    require(
        gpu.status == MR_OPENSIM_SPATIAL_SUCCESS && gpu.coordinateCount == 1u,
        "GPU spatial-transform status failed"
    );
    const std::array<mr_float4, 3u> rows{
        gpu.rotationRow0,
        gpu.rotationRow1,
        gpu.rotationRow2,
    };
    for (std::size_t row = 0u; row < 3u; ++row) {
        requireNear(rows[row].x, cpu.rotation[row * 3u], "rotation x");
        requireNear(rows[row].y, cpu.rotation[row * 3u + 1u], "rotation y");
        requireNear(rows[row].z, cpu.rotation[row * 3u + 2u], "rotation z");
    }
    requireNear(gpu.translation.x, cpu.translation[0u], "translation x");
    requireNear(gpu.translation.y, cpu.translation[1u], "translation y");
    requireNear(gpu.translation.z, cpu.translation[2u], "translation z");
    for (std::size_t component = 0u; component < 3u; ++component) {
        const std::array<float, 3u> angular{
            gpu.motionAngular[0u].x,
            gpu.motionAngular[0u].y,
            gpu.motionAngular[0u].z,
        };
        const std::array<float, 3u> linear{
            gpu.motionLinear[0u].x,
            gpu.motionLinear[0u].y,
            gpu.motionLinear[0u].z,
        };
        const std::array<float, 3u> angularDot{
            gpu.motionAngularDot[0u].x,
            gpu.motionAngularDot[0u].y,
            gpu.motionAngularDot[0u].z,
        };
        const std::array<float, 3u> linearDot{
            gpu.motionLinearDot[0u].x,
            gpu.motionLinearDot[0u].y,
            gpu.motionLinearDot[0u].z,
        };
        requireNear(angular[component], cpu.motionSubspace[0u].angular[component], "H angular");
        requireNear(linear[component], cpu.motionSubspace[0u].linear[component], "H linear");
        requireNear(angularDot[component], cpu.motionSubspaceDot[0u].angular[component], "Hdot angular");
        requireNear(linearDot[component], cpu.motionSubspaceDot[0u].linear[component], "Hdot linear");
    }
}

} // namespace

int main() {
    try {
        const auto compiled = metalrobo::compileOpenSimSpatialTransform(walkerKnee());
        require(compiled.succeeded(), "walker knee compilation failed");
        MROpenSimSpatialTransformGPU program{};
        require(
            metalrobo::packOpenSimSpatialTransformGPU(compiled.transform, program) ==
                metalrobo::OpenSimSpatialTransformStatus::success,
            "walker knee GPU packing failed"
        );
        constexpr float coordinate = 0.43f;
        constexpr float velocity = -0.71f;
        const auto cpu = metalrobo::evaluateOpenSimSpatialTransform(
            compiled.transform,
            {static_cast<double>(coordinate)},
            {static_cast<double>(velocity)}
        );
        require(cpu.succeeded(), "walker knee CPU evaluation failed");
        const std::array<float, MR_OPENSIM_SPATIAL_MAX_COORDINATES> coordinates{
            coordinate, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };
        const std::array<float, MR_OPENSIM_SPATIAL_MAX_COORDINATES> velocities{
            velocity, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };
        const GPUResult first = runMetal(program, coordinates, velocities);
        const GPUResult second = runMetal(program, coordinates, velocities);
        compare(first.result, cpu);
        compare(second.result, cpu);
        require(
            std::memcmp(&first.result, &second.result, sizeof(first.result)) == 0,
            "OpenSim spatial-transform GPU output was not deterministic"
        );
        std::cout << "opensim_spatial_transform_gpu=ok"
                  << " device=" << first.deviceName
                  << " tx=" << first.result.translation.x
                  << " h_angular_x=" << first.result.motionAngular[0u].x
                  << " hdot_linear_x=" << first.result.motionLinearDot[0u].x
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "opensim_spatial_transform_gpu=failed " << error.what() << '\n';
        return 1;
    }
}
