#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalMultiArticulatedInverseMass.hpp"

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
#include <ranges>
#include <string>
#include <system_error>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kThreadgroupWidth = 32u;
constexpr float kQuaternionTolerance = 2.0e-5f;
const char kImageAnchor = 0;

MetalMultiArticulatedInverseMassDiagnostics reject(
    MetalMultiArticulatedInverseMassDiagnostics diagnostics,
    const MetalMultiArticulatedInverseMassStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

std::string string(NSString* value) {
    return value != nil && value.UTF8String != nullptr
        ? std::string{value.UTF8String}
        : std::string{};
}

std::string errorString(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    std::string message = string(error.localizedDescription);
    return message.empty() ? string(error.description) : message;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            directory / "metalrobo/MetalRobo.metallib",
            directory.parent_path() / "shaders/MetalRobo.metallib",
        };
        for (const auto& candidate : candidates) {
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

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& output
) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& output
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

template <typename T>
bool addBytes(
    const std::size_t elements,
    std::size_t& total
) {
    std::size_t bytes = 0u;
    return
        checkedMultiply(
            std::max<std::size_t>(elements, 1u),
            sizeof(T),
            bytes
        ) &&
        checkedAdd(total, bytes, total) &&
        bytes <= std::numeric_limits<NSUInteger>::max();
}

bool validTopology(
    const EngineModel& model,
    const std::size_t articulationIndex
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if ((articulation.rootType != MR_ROOT_FIXED &&
         articulation.rootType != MR_ROOT_FLOATING) ||
        articulation.bodyCount == 0u ||
        articulation.bodyCount > MR_ARTICULATED_ABA_MAX_BODIES ||
        articulation.nv == 0u ||
        articulation.nv > MR_ARTICULATED_ABA_MAX_DOFS ||
        articulation.nq > MR_ARTICULATED_ABA_MAX_Q ||
        articulation.jointCount + 1u != articulation.bodyCount) {
        return false;
    }
    for (std::uint32_t localJoint = 0u;
         localJoint < articulation.jointCount;
         ++localJoint) {
        const std::uint32_t type = model.joints[
            articulation.firstJoint + localJoint
        ].jointType;
        if (type != MR_JOINT_FIXED &&
            type != MR_JOINT_REVOLUTE &&
            type != MR_JOINT_CONTINUOUS &&
            type != MR_JOINT_PRISMATIC) {
            return false;
        }
    }
    return true;
}

template <typename T>
id<MTLBuffer> inputBuffer(
    id<MTLDevice> device,
    const T* data,
    const std::size_t elements
) {
    const NSUInteger bytes = static_cast<NSUInteger>(
        std::max<std::size_t>(elements, 1u) * sizeof(T)
    );
    if (elements == 0u) {
        return [device
            newBufferWithLength:bytes
                        options:MTLResourceStorageModeShared];
    }
    return [device
        newBufferWithBytes:data
                   length:bytes
                  options:MTLResourceStorageModeShared];
}

template <typename T>
id<MTLBuffer> outputBuffer(
    id<MTLDevice> device,
    const std::size_t elements
) {
    return [device
        newBufferWithLength:static_cast<NSUInteger>(
            std::max<std::size_t>(elements, 1u) * sizeof(T)
        )
                    options:MTLResourceStorageModeShared];
}

} // namespace

MetalMultiArticulatedInverseMassDiagnostics
runMetalMultiArticulatedInverseMass(
    const EngineModel& model,
    const MetalMultiArticulatedInverseMassInput& input,
    MetalMultiArticulatedInverseMassResult& output,
    const MetalMultiArticulatedInverseMassConfig& config
) {
    @autoreleasepool {
        MetalMultiArticulatedInverseMassDiagnostics diagnostics{};
        std::string modelReason;
        if (!model.valid(&modelReason)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::invalidModel,
                "invalid EngineModel: " + modelReason
            );
        }
        if (model.articulations.empty()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    unsupportedTopology,
                "inverse mass requires at least one articulation"
            );
        }
        if (model.articulations.size() >
                std::numeric_limits<mr_u32>::max() ||
            input.environmentCount == 0u ||
            input.environmentCount >
                std::numeric_limits<mr_u32>::max() ||
            input.rhsCount == 0u ||
            input.rhsCount >
                MR_ARTICULATED_INVERSE_MASS_MAX_RHS) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    invalidDimensions,
                "environment, articulation, or RHS count is outside "
                "the GPU ABI"
            );
        }
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            if (!validTopology(model, articulationIndex)) {
                diagnostics.firstFailingArticulation =
                    static_cast<std::uint32_t>(articulationIndex);
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedInverseMassStatus::
                        unsupportedTopology,
                    "articulation exceeds the scalar inverse-mass "
                    "topology bucket"
                );
            }
        }

        MetalMultiArticulatedInverseMassLayout layout;
        std::size_t environmentRhs = 0u;
        if (!checkedMultiply(
                input.environmentCount,
                model.world.nq,
                layout.qElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                input.rhsCount,
                environmentRhs
            ) ||
            !checkedMultiply(
                environmentRhs,
                model.world.nv,
                layout.rhsElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.articulations.size(),
                layout.statusElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    arithmeticOverflow,
                "multi-articulation inverse-mass element overflow"
            );
        }
        layout.outputElements = layout.rhsElements;
        if (input.q.size() != layout.qElements ||
            input.rightHandSides.size() != layout.rhsElements) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    invalidDimensions,
                "global q or RHS span has the wrong dimensions"
            );
        }
        const std::size_t rhsEnvironmentStride =
            input.rhsCount * model.world.nv;
        constexpr std::uint64_t shaderElements =
            std::uint64_t{
                std::numeric_limits<mr_u32>::max()
            } + 1u;
        if (model.world.nq >
                std::numeric_limits<mr_u32>::max() ||
            model.world.nv >
                std::numeric_limits<mr_u32>::max() ||
            rhsEnvironmentStride >
                std::numeric_limits<mr_u32>::max() ||
            layout.statusElements >
                std::numeric_limits<mr_u32>::max() ||
            static_cast<std::uint64_t>(layout.qElements) >
                shaderElements ||
            static_cast<std::uint64_t>(layout.rhsElements) >
                shaderElements) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    arithmeticOverflow,
                "global inverse-mass streams exceed 32-bit shader "
                "addressing"
            );
        }
        if (!std::ranges::all_of(
                input.q,
                [](const float value) {
                    return std::isfinite(value);
                }
            ) ||
            !std::ranges::all_of(
                input.rightHandSides,
                [](const float value) {
                    return std::isfinite(value);
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    nonfiniteInput,
                "q or generalized RHS contains a non-finite value"
            );
        }
        for (std::size_t environment = 0u;
             environment < input.environmentCount;
             ++environment) {
            const std::size_t environmentQ =
                environment * model.world.nq;
            for (const MRArticulationGPU& articulation :
                 model.articulations) {
                if (articulation.rootType != MR_ROOT_FLOATING) {
                    continue;
                }
                const std::size_t rotation =
                    environmentQ + articulation.qOffset + 3u;
                const double norm = std::sqrt(
                    double(input.q[rotation + 0u]) *
                        input.q[rotation + 0u] +
                    double(input.q[rotation + 1u]) *
                        input.q[rotation + 1u] +
                    double(input.q[rotation + 2u]) *
                        input.q[rotation + 2u] +
                    double(input.q[rotation + 3u]) *
                        input.q[rotation + 3u]
                );
                if (!std::isfinite(norm) ||
                    std::abs(norm - 1.0) >
                        kQuaternionTolerance) {
                    return reject(
                        std::move(diagnostics),
                        MetalMultiArticulatedInverseMassStatus::
                            nonfiniteInput,
                        "floating-root quaternion is not normalized"
                    );
                }
            }
        }

        layout.dispatches.reserve(model.articulations.size());
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            MRMultiInverseMassDispatchGPU work{};
            work.dispatch.articulationIndex =
                static_cast<std::uint32_t>(articulationIndex);
            work.dispatch.environmentCount =
                static_cast<std::uint32_t>(input.environmentCount);
            work.dispatch.rhsCount =
                static_cast<std::uint32_t>(input.rhsCount);
            work.dispatch.qStride = model.world.nq;
            work.dispatch.rhsEnvironmentStride =
                static_cast<std::uint32_t>(rhsEnvironmentStride);
            work.dispatch.rhsVectorStride = model.world.nv;
            work.dispatch.outputEnvironmentStride =
                static_cast<std::uint32_t>(rhsEnvironmentStride);
            work.dispatch.outputVectorStride = model.world.nv;
            work.qBase = articulation.qOffset;
            work.rhsBase = articulation.vOffset;
            work.outputBase = articulation.vOffset;
            work.statusBase = static_cast<std::uint32_t>(
                articulationIndex * input.environmentCount
            );
            layout.dispatches.push_back(work);
        }

        std::size_t allocatedBytes = 0u;
        if (!addBytes<MRWorldGPU>(1u, allocatedBytes) ||
            !addBytes<MRArticulationGPU>(
                model.articulations.size(),
                allocatedBytes
            ) ||
            !addBytes<MRJointDescriptorGPU>(
                model.joints.size(),
                allocatedBytes
            ) ||
            !addBytes<MRDofPropertiesGPU>(
                model.dofs.size(),
                allocatedBytes
            ) ||
            !addBytes<MRBodyPropertiesGPU>(
                model.bodies.size(),
                allocatedBytes
            ) ||
            !addBytes<MRMultiInverseMassDispatchGPU>(
                layout.dispatches.size(),
                allocatedBytes
            ) ||
            !addBytes<float>(
                layout.qElements,
                allocatedBytes
            ) ||
            !addBytes<float>(
                layout.rhsElements,
                allocatedBytes
            ) ||
            !addBytes<float>(
                layout.outputElements,
                allocatedBytes
            ) ||
            !addBytes<MRInverseMassStatusGPU>(
                layout.statusElements,
                allocatedBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    arithmeticOverflow,
                "multi-articulation inverse-mass byte overflow"
            );
        }
        layout.totalAllocatedBytes = allocatedBytes;
        diagnostics.layout = layout;

        const std::string metallibPath =
            config.metallibPath.empty()
            ? defaultMetallibPath()
            : config.metallibPath;
        if (metallibPath.empty()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalDeviceUnavailable,
                "no Metal device is available"
            );
        }
        diagnostics.deviceName = string(device.name);
        NSString* path = [NSString
            stringWithUTF8String:metallibPath.c_str()];
        if (path == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
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
                MetalMultiArticulatedInverseMassStatus::
                    metalLibraryFailure,
                "failed to load metallib: " + errorString(error)
            );
        }
        id<MTLFunction> function = [library
            newFunctionWithName:
                @"mr_multi_articulated_inverse_mass"];
        if (function == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalLibraryFailure,
                "metallib lacks multi-articulation inverse mass"
            );
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function
                                           error:&error];
        if (pipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalPipelineFailure,
                "failed to create inverse-mass pipeline: " +
                    errorString(error)
            );
        }
        if (pipeline.threadExecutionWidth != kThreadgroupWidth ||
            pipeline.maxTotalThreadsPerThreadgroup <
                kThreadgroupWidth ||
            pipeline.staticThreadgroupMemoryLength >
                device.maxThreadgroupMemoryLength) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalDeviceUnsupported,
                "multi-articulation inverse mass requires SIMD32"
            );
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer =
            [queue commandBuffer];
        if (queue == nil || commandBuffer == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalDeviceUnavailable,
                "failed to create Metal queue or command buffer"
            );
        }
        id<MTLBuffer> buffers[10] = {};
        buffers[0] = inputBuffer(device, &model.world, 1u);
        buffers[1] = inputBuffer(
            device,
            model.articulations.data(),
            model.articulations.size()
        );
        buffers[2] = inputBuffer(
            device,
            model.joints.data(),
            model.joints.size()
        );
        buffers[3] = inputBuffer(
            device,
            model.dofs.data(),
            model.dofs.size()
        );
        buffers[4] = inputBuffer(
            device,
            model.bodies.data(),
            model.bodies.size()
        );
        buffers[5] = inputBuffer(
            device,
            layout.dispatches.data(),
            layout.dispatches.size()
        );
        buffers[6] = inputBuffer(
            device,
            input.q.data(),
            layout.qElements
        );
        buffers[7] = inputBuffer(
            device,
            input.rightHandSides.data(),
            layout.rhsElements
        );
        buffers[8] = outputBuffer<float>(
            device,
            layout.outputElements
        );
        buffers[9] = outputBuffer<MRInverseMassStatusGPU>(
            device,
            layout.statusElements
        );
        for (id<MTLBuffer> buffer : buffers) {
            if (buffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedInverseMassStatus::
                        metalBufferFailure,
                    "failed to allocate inverse-mass buffer"
                );
            }
        }

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalCommandFailure,
                "failed to create inverse-mass encoder"
            );
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < 10u; ++index) {
            [encoder setBuffer:buffers[index]
                        offset:0u
                       atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(input.environmentCount),
                static_cast<NSUInteger>(
                    model.articulations.size()
                ),
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kThreadgroupWidth,
                1u,
                1u
            )];
        [encoder endEncoding];

        const auto start = std::chrono::steady_clock::now();
        diagnostics.dispatched = true;
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    metalCommandFailure,
                "inverse-mass command failed: " +
                    errorString(commandBuffer.error)
            );
        }

        MetalMultiArticulatedInverseMassResult staged;
        staged.layout = layout;
        staged.output.resize(layout.outputElements);
        staged.statuses.resize(layout.statusElements);
        std::memcpy(
            staged.output.data(),
            buffers[8].contents,
            layout.outputElements * sizeof(float)
        );
        std::memcpy(
            staged.statuses.data(),
            buffers[9].contents,
            layout.statusElements *
                sizeof(MRInverseMassStatusGPU)
        );
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            for (std::size_t environment = 0u;
                 environment < input.environmentCount;
                 ++environment) {
                const MRInverseMassStatusGPU& status =
                    staged.statuses[
                        articulationIndex *
                            input.environmentCount +
                        environment
                    ];
                if (status.code != MR_INVERSE_MASS_SUCCESS ||
                    status.articulationIndex != articulationIndex ||
                    status.environment != environment) {
                    diagnostics.firstFailingArticulation =
                        static_cast<std::uint32_t>(
                            articulationIndex
                        );
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(environment);
                    diagnostics.firstGPUStatusCode = status.code;
                    return reject(
                        std::move(diagnostics),
                        MetalMultiArticulatedInverseMassStatus::
                            gpuArticulationFailure,
                        "a GPU inverse-mass packet failed"
                    );
                }
            }
        }
        if (!std::ranges::all_of(
                staged.output,
                [](const float value) {
                    return std::isfinite(value);
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedInverseMassStatus::
                    nonfiniteResult,
                "GPU produced a non-finite global response"
            );
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    }
}

const char* metalMultiArticulatedInverseMassStatusName(
    const MetalMultiArticulatedInverseMassStatus status
) noexcept {
    switch (status) {
    case MetalMultiArticulatedInverseMassStatus::success:
        return "success";
    case MetalMultiArticulatedInverseMassStatus::invalidModel:
        return "invalid_model";
    case MetalMultiArticulatedInverseMassStatus::
            unsupportedTopology:
        return "unsupported_topology";
    case MetalMultiArticulatedInverseMassStatus::
            invalidDimensions:
        return "invalid_dimensions";
    case MetalMultiArticulatedInverseMassStatus::
            arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalMultiArticulatedInverseMassStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalMultiArticulatedInverseMassStatus::
            metallibUnavailable:
        return "metallib_unavailable";
    case MetalMultiArticulatedInverseMassStatus::
            metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalMultiArticulatedInverseMassStatus::
            metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalMultiArticulatedInverseMassStatus::
            metalLibraryFailure:
        return "metal_library_failure";
    case MetalMultiArticulatedInverseMassStatus::
            metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalMultiArticulatedInverseMassStatus::
            metalBufferFailure:
        return "metal_buffer_failure";
    case MetalMultiArticulatedInverseMassStatus::
            metalCommandFailure:
        return "metal_command_failure";
    case MetalMultiArticulatedInverseMassStatus::
            gpuArticulationFailure:
        return "gpu_articulation_failure";
    case MetalMultiArticulatedInverseMassStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
