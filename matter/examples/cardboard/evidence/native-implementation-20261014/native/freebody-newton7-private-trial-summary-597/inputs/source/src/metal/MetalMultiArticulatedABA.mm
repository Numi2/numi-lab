#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalMultiArticulatedABA.hpp"

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
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

constexpr NSUInteger kThreadgroupWidth = 32u;
constexpr float kQuaternionTolerance = 2.0e-5f;
const char kImageAnchor = 0;

MetalMultiArticulatedABADiagnostics reject(
    MetalMultiArticulatedABADiagnostics diagnostics,
    const MetalMultiArticulatedABAStatus status,
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

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
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
        const MRJointDescriptorGPU& joint =
            model.joints[articulation.firstJoint + localJoint];
        if (joint.jointType != MR_JOINT_FIXED &&
            joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC) {
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

MetalMultiArticulatedABADiagnostics
runMetalMultiArticulatedABA(
    const EngineModel& model,
    const MetalMultiArticulatedABAInput& input,
    MetalMultiArticulatedABAResult& output,
    const MetalMultiArticulatedABAConfig& config
) {
    @autoreleasepool {
        MetalMultiArticulatedABADiagnostics diagnostics{};
        std::string modelReason;
        if (!model.valid(&modelReason)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::invalidModel,
                "invalid EngineModel: " + modelReason
            );
        }
        if (model.articulations.empty()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::unsupportedTopology,
                "multi-articulated ABA requires an articulation"
            );
        }
        if (model.articulations.size() >
            std::numeric_limits<mr_u32>::max()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::arithmeticOverflow,
                "articulation count does not fit the GPU grid ABI"
            );
        }
        if (input.environmentCount == 0u ||
            input.environmentCount >
                std::numeric_limits<mr_u32>::max()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::invalidDimensions,
                "environmentCount is outside the GPU ABI"
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
                    MetalMultiArticulatedABAStatus::
                        unsupportedTopology,
                    "articulation exceeds the scalar ABA bucket"
                );
            }
        }

        MetalMultiArticulatedABALayout layout;
        if (!checkedMultiply(
                input.environmentCount,
                model.world.nq,
                layout.qElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.nv,
                layout.vElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.nv,
                layout.effortElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.bodyCount,
                layout.wrenchElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.nv,
                layout.accelerationElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.nv,
                layout.nextVElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.nq,
                layout.nextQElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.articulations.size(),
                layout.statusElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::arithmeticOverflow,
                "multi-articulated ABA element-count overflow"
            );
        }
        if (layout.statusElements >
            std::numeric_limits<mr_u32>::max()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::arithmeticOverflow,
                "articulation status addressing exceeds 32-bit ABI"
            );
        }
        constexpr std::uint64_t shaderElements =
            std::uint64_t{
                std::numeric_limits<mr_u32>::max()
            } + 1u;
        const auto exceedsShaderAddressing =
            [](const std::size_t elements) {
                return static_cast<std::uint64_t>(elements) >
                    shaderElements;
            };
        if (exceedsShaderAddressing(layout.qElements) ||
            exceedsShaderAddressing(layout.vElements) ||
            exceedsShaderAddressing(layout.effortElements) ||
            exceedsShaderAddressing(layout.wrenchElements) ||
            exceedsShaderAddressing(layout.accelerationElements) ||
            exceedsShaderAddressing(layout.nextVElements) ||
            exceedsShaderAddressing(layout.nextQElements)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::arithmeticOverflow,
                "global state exceeds 32-bit shader addressing"
            );
        }
        if (input.q.size() != layout.qElements ||
            input.v.size() != layout.vElements ||
            input.effort.size() != layout.effortElements ||
            (!input.bodyWrenches.empty() &&
             input.bodyWrenches.size() != layout.wrenchElements)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::invalidDimensions,
                "global q/v/effort/wrench spans have wrong dimensions"
            );
        }
        for (const float value : input.q) {
            if (!std::isfinite(value)) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedABAStatus::nonfiniteInput,
                    "q contains a non-finite value"
                );
            }
        }
        for (const float value : input.v) {
            if (!std::isfinite(value)) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedABAStatus::nonfiniteInput,
                    "v contains a non-finite value"
                );
            }
        }
        for (const float value : input.effort) {
            if (!std::isfinite(value)) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedABAStatus::nonfiniteInput,
                    "effort contains a non-finite value"
                );
            }
        }
        for (const MRABABodyWrenchGPU& wrench :
             input.bodyWrenches) {
            if (!finite(wrench.force) ||
                !finite(wrench.torque) ||
                wrench.force.w != 0.0f ||
                wrench.torque.w != 0.0f) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedABAStatus::nonfiniteInput,
                    "body wrench contains a non-finite or reserved value"
                );
            }
        }
        for (std::size_t environment = 0u;
             environment < input.environmentCount;
             ++environment) {
            const std::size_t qBase =
                environment * model.world.nq;
            for (const MRArticulationGPU& articulation :
                 model.articulations) {
                if (articulation.rootType != MR_ROOT_FLOATING) {
                    continue;
                }
                const std::size_t base =
                    qBase + articulation.qOffset + 3u;
                const double norm = std::sqrt(
                    double(input.q[base + 0u]) *
                        input.q[base + 0u] +
                    double(input.q[base + 1u]) *
                        input.q[base + 1u] +
                    double(input.q[base + 2u]) *
                        input.q[base + 2u] +
                    double(input.q[base + 3u]) *
                        input.q[base + 3u]
                );
                if (!std::isfinite(norm) ||
                    std::abs(norm - 1.0) >
                        kQuaternionTolerance) {
                    return reject(
                        std::move(diagnostics),
                        MetalMultiArticulatedABAStatus::nonfiniteInput,
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
            MRMultiABADispatchGPU work{};
            work.dispatch.articulationIndex =
                static_cast<std::uint32_t>(articulationIndex);
            work.dispatch.environmentCount =
                static_cast<std::uint32_t>(input.environmentCount);
            work.dispatch.flags =
                (input.applyBodyDamping
                    ? MR_ABA_APPLY_BODY_DAMPING
                    : 0u) |
                (input.implicitDrives
                    ? MR_ABA_IMPLICIT_DRIVES
                    : 0u) |
                (!input.bodyWrenches.empty()
                    ? MR_ABA_HAS_BODY_WRENCHES
                    : 0u);
            work.dispatch.qStride = model.world.nq;
            work.dispatch.vStride = model.world.nv;
            work.dispatch.effortStride = model.world.nv;
            work.dispatch.wrenchStride = model.world.bodyCount;
            work.dispatch.accelerationStride = model.world.nv;
            work.dispatch.nextVStride = model.world.nv;
            work.dispatch.nextQStride = model.world.nq;
            work.qBase = articulation.qOffset;
            work.vBase = articulation.vOffset;
            work.effortBase = articulation.vOffset;
            work.wrenchBase = input.bodyWrenches.empty()
                ? 0u
                : articulation.firstBody;
            work.accelerationBase = articulation.vOffset;
            work.nextVBase = articulation.vOffset;
            work.nextQBase = articulation.qOffset;
            work.statusBase =
                static_cast<std::uint32_t>(
                    articulationIndex * input.environmentCount
                );
            layout.dispatches.push_back(work);
        }

        std::size_t bytes = 0u;
        if (!addBytes<MRWorldGPU>(1u, bytes) ||
            !addBytes<MRArticulationGPU>(
                model.articulations.size(),
                bytes
            ) ||
            !addBytes<MRJointDescriptorGPU>(
                model.joints.size(),
                bytes
            ) ||
            !addBytes<MRDofPropertiesGPU>(
                model.dofs.size(),
                bytes
            ) ||
            !addBytes<MRBodyPropertiesGPU>(
                model.bodies.size(),
                bytes
            ) ||
            !addBytes<MRMultiABADispatchGPU>(
                layout.dispatches.size(),
                bytes
            ) ||
            !addBytes<float>(layout.qElements, bytes) ||
            !addBytes<float>(layout.vElements, bytes) ||
            !addBytes<float>(layout.effortElements, bytes) ||
            !addBytes<MRABABodyWrenchGPU>(
                input.bodyWrenches.empty()
                    ? 0u
                    : layout.wrenchElements,
                bytes
            ) ||
            !addBytes<float>(
                layout.accelerationElements,
                bytes
            ) ||
            !addBytes<float>(layout.nextVElements, bytes) ||
            !addBytes<float>(layout.nextQElements, bytes) ||
            !addBytes<MRABAStatusGPU>(
                layout.statusElements,
                bytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::arithmeticOverflow,
                "multi-articulated ABA byte-count overflow"
            );
        }
        layout.totalAllocatedBytes = bytes;
        diagnostics.layout = layout;

        std::string metallibPath = config.metallibPath.empty()
            ? defaultMetallibPath()
            : config.metallibPath;
        if (metallibPath.empty()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::
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
                MetalMultiArticulatedABAStatus::metallibUnavailable,
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
                MetalMultiArticulatedABAStatus::metalLibraryFailure,
                "failed to load metallib: " + errorString(error)
            );
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"mr_multi_articulated_aba_step"];
        if (function == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::metalLibraryFailure,
                "metallib lacks multi-articulated ABA"
            );
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function
                                           error:&error];
        if (pipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::metalPipelineFailure,
                "failed to create ABA pipeline: " +
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
                MetalMultiArticulatedABAStatus::
                    metalDeviceUnsupported,
                "multi-articulated ABA requires a SIMD32 pipeline"
            );
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer =
            [queue commandBuffer];
        if (queue == nil || commandBuffer == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::
                    metalDeviceUnavailable,
                "failed to create Metal queue or command buffer"
            );
        }

        id<MTLBuffer> buffers[14] = {};
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
            input.v.data(),
            layout.vElements
        );
        buffers[8] = inputBuffer(
            device,
            input.effort.data(),
            layout.effortElements
        );
        buffers[9] = inputBuffer(
            device,
            input.bodyWrenches.data(),
            input.bodyWrenches.empty()
                ? 0u
                : layout.wrenchElements
        );
        buffers[10] = outputBuffer<float>(
            device,
            layout.accelerationElements
        );
        buffers[11] = outputBuffer<float>(
            device,
            layout.nextVElements
        );
        buffers[12] = outputBuffer<float>(
            device,
            layout.nextQElements
        );
        buffers[13] = outputBuffer<MRABAStatusGPU>(
            device,
            layout.statusElements
        );
        for (id<MTLBuffer> buffer : buffers) {
            if (buffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedABAStatus::
                        metalBufferFailure,
                    "failed to allocate multi-articulated ABA buffer"
                );
            }
        }
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::
                    metalCommandFailure,
                "failed to create ABA command encoder"
            );
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < 14u; ++index) {
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
                MetalMultiArticulatedABAStatus::
                    metalCommandFailure,
                "multi-articulated ABA command failed: " +
                    errorString(commandBuffer.error)
            );
        }

        MetalMultiArticulatedABAResult staged;
        staged.layout = layout;
        staged.acceleration.resize(layout.accelerationElements);
        staged.nextV.resize(layout.nextVElements);
        staged.nextQ.resize(layout.nextQElements);
        staged.statuses.resize(layout.statusElements);
        std::memcpy(
            staged.acceleration.data(),
            buffers[10].contents,
            layout.accelerationElements * sizeof(float)
        );
        std::memcpy(
            staged.nextV.data(),
            buffers[11].contents,
            layout.nextVElements * sizeof(float)
        );
        std::memcpy(
            staged.nextQ.data(),
            buffers[12].contents,
            layout.nextQElements * sizeof(float)
        );
        std::memcpy(
            staged.statuses.data(),
            buffers[13].contents,
            layout.statusElements * sizeof(MRABAStatusGPU)
        );
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            for (std::size_t environment = 0u;
                 environment < input.environmentCount;
                 ++environment) {
                const MRABAStatusGPU& status = staged.statuses[
                    articulationIndex * input.environmentCount +
                    environment
                ];
                if (status.code != MR_ABA_SUCCESS ||
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
                        MetalMultiArticulatedABAStatus::
                            gpuArticulationFailure,
                        "a GPU articulation packet failed"
                    );
                }
            }
        }
        const auto finiteVector = [](const auto& values) {
            return std::ranges::all_of(
                values,
                [](const float value) {
                    return std::isfinite(value);
                }
            );
        };
        if (!finiteVector(staged.acceleration) ||
            !finiteVector(staged.nextV) ||
            !finiteVector(staged.nextQ)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedABAStatus::internalFailure,
                "GPU published a non-finite global state"
            );
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    }
}

const char* metalMultiArticulatedABAStatusName(
    const MetalMultiArticulatedABAStatus status
) noexcept {
    switch (status) {
    case MetalMultiArticulatedABAStatus::success:
        return "success";
    case MetalMultiArticulatedABAStatus::invalidModel:
        return "invalid_model";
    case MetalMultiArticulatedABAStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalMultiArticulatedABAStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalMultiArticulatedABAStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalMultiArticulatedABAStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalMultiArticulatedABAStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalMultiArticulatedABAStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalMultiArticulatedABAStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalMultiArticulatedABAStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalMultiArticulatedABAStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalMultiArticulatedABAStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalMultiArticulatedABAStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalMultiArticulatedABAStatus::gpuArticulationFailure:
        return "gpu_articulation_failure";
    case MetalMultiArticulatedABAStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
