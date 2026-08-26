#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/OpenSimSpatialTransform.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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
static_assert(sizeof(MROpenSimSpatialTransformInputGPU) == 64u);
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

float inputScalar(const mr_float4* blocks, const std::size_t index) {
    const mr_float4& block = blocks[index / 4u];
    switch (index % 4u) {
    case 0u:
        return block.x;
    case 1u:
        return block.y;
    case 2u:
        return block.z;
    default:
        return block.w;
    }
}

std::vector<double> inputValues(
    const MROpenSimSpatialTransformInputGPU& input,
    const std::uint32_t coordinateCount,
    const bool velocities
) {
    require(
        coordinateCount > 0u && coordinateCount <= MR_OPENSIM_SPATIAL_MAX_COORDINATES,
        "invalid input coordinate count"
    );
    const mr_float4* blocks = velocities
        ? input.coordinateVelocityBlocks
        : input.coordinateBlocks;
    std::vector<double> values;
    values.reserve(coordinateCount);
    for (std::size_t index = 0u; index < 8u; ++index) {
        const float value = inputScalar(blocks, index);
        require(std::isfinite(value), "non-finite spatial-transform input");
        if (index < coordinateCount) {
            values.push_back(static_cast<double>(value));
        } else {
            require(value == 0.0f, "non-canonical spatial-transform input padding");
        }
    }
    return values;
}

template <typename T>
T readArtifact(const std::string& path, const std::string& label) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.is_open(), "could not open " + label + " artifact: " + path);
    T result{};
    stream.read(reinterpret_cast<char*>(&result), sizeof(result));
    require(
        stream.gcount() == static_cast<std::streamsize>(sizeof(result)),
        label + " artifact has the wrong byte size: " + path
    );
    char extra = '\0';
    stream.read(&extra, 1);
    require(stream.gcount() == 0, label + " artifact has trailing bytes: " + path);
    return result;
}

struct ProbeCase {
    MROpenSimSpatialTransformGPU program{};
    MROpenSimSpatialTransformInputGPU input{};
    std::string source = "built_in_walker_knee";
};

GPUResult runMetal(
    const MROpenSimSpatialTransformGPU& program,
    const MROpenSimSpatialTransformInputGPU& input
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
        id<MTLBuffer> inputBuffer = buffer(device, &input, 1u, @"OpenSim input");
        id<MTLBuffer> resultBuffer = buffer(device, &result, 1u, @"OpenSim result");
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        require(
            commandBuffer != nil && encoder != nil,
            "failed to create OpenSim spatial-transform command encoder"
        );
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:programBuffer offset:0 atIndex:0];
        [encoder setBuffer:inputBuffer offset:0 atIndex:1];
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
        gpu.status == MR_OPENSIM_SPATIAL_SUCCESS &&
            gpu.coordinateCount == cpu.motionSubspace.size(),
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
    for (std::size_t coordinate = 0u; coordinate < cpu.motionSubspace.size(); ++coordinate) {
        const std::array<float, 3u> angular{
            gpu.motionAngular[coordinate].x,
            gpu.motionAngular[coordinate].y,
            gpu.motionAngular[coordinate].z,
        };
        const std::array<float, 3u> linear{
            gpu.motionLinear[coordinate].x,
            gpu.motionLinear[coordinate].y,
            gpu.motionLinear[coordinate].z,
        };
        const std::array<float, 3u> angularDot{
            gpu.motionAngularDot[coordinate].x,
            gpu.motionAngularDot[coordinate].y,
            gpu.motionAngularDot[coordinate].z,
        };
        const std::array<float, 3u> linearDot{
            gpu.motionLinearDot[coordinate].x,
            gpu.motionLinearDot[coordinate].y,
            gpu.motionLinearDot[coordinate].z,
        };
        for (std::size_t component = 0u; component < 3u; ++component) {
            requireNear(
                angular[component], cpu.motionSubspace[coordinate].angular[component], "H angular"
            );
            requireNear(
                linear[component], cpu.motionSubspace[coordinate].linear[component], "H linear"
            );
            requireNear(
                angularDot[component],
                cpu.motionSubspaceDot[coordinate].angular[component],
                "Hdot angular"
            );
            requireNear(
                linearDot[component], cpu.motionSubspaceDot[coordinate].linear[component], "Hdot linear"
            );
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        ProbeCase probe;
        if (argc == 1) {
            const auto compiled = metalrobo::compileOpenSimSpatialTransform(walkerKnee());
            require(compiled.succeeded(), "walker knee compilation failed");
            require(
                metalrobo::packOpenSimSpatialTransformGPU(compiled.transform, probe.program) ==
                    metalrobo::OpenSimSpatialTransformStatus::success,
                "walker knee GPU packing failed"
            );
            probe.input.coordinateBlocks[0].x = 0.43f;
            probe.input.coordinateVelocityBlocks[0].x = -0.71f;
        } else {
            require(
                argc == 5 && std::string(argv[1]) == "--program" &&
                    std::string(argv[3]) == "--input",
                "usage: metalrobo_opensim_spatial_transform_gpu_probe "
                "[--program PATH --input PATH]"
            );
            probe.program = readArtifact<MROpenSimSpatialTransformGPU>(argv[2], "program");
            probe.input = readArtifact<MROpenSimSpatialTransformInputGPU>(argv[4], "input");
            probe.source = argv[2];
        }
        const auto decoded = metalrobo::unpackOpenSimSpatialTransformGPU(probe.program);
        require(decoded.succeeded(), "spatial-transform program ABI decode failed");
        const std::vector<double> coordinates = inputValues(
            probe.input, decoded.transform.coordinateCount, false
        );
        const std::vector<double> velocities = inputValues(
            probe.input, decoded.transform.coordinateCount, true
        );
        const auto cpu = metalrobo::evaluateOpenSimSpatialTransform(
            decoded.transform, coordinates, velocities
        );
        require(cpu.succeeded(), "spatial-transform CPU evaluation failed");
        const GPUResult first = runMetal(probe.program, probe.input);
        const GPUResult second = runMetal(probe.program, probe.input);
        compare(first.result, cpu);
        compare(second.result, cpu);
        require(
            std::memcmp(&first.result, &second.result, sizeof(first.result)) == 0,
            "OpenSim spatial-transform GPU output was not deterministic"
        );
        std::cout << "opensim_spatial_transform_gpu=ok"
                  << " device=" << first.deviceName
                  << " source=" << probe.source
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
