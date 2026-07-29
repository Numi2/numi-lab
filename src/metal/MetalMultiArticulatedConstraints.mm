#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalMultiArticulatedConstraints.hpp"

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

constexpr NSUInteger kWaveWidth = 32u;
constexpr float kQuaternionTolerance = 2.0e-5f;
const char kImageAnchor = 0;

MetalMultiArticulatedConstraintDiagnostics reject(
    MetalMultiArticulatedConstraintDiagnostics diagnostics,
    const MetalMultiArticulatedConstraintStatus status,
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

bool representableFloat(const double value) {
    return
        std::isfinite(value) &&
        std::abs(value) <=
            std::numeric_limits<float>::max();
}

bool validConfiguration(
    const MetalMultiArticulatedConstraintConfig& config
) {
    return
        config.solverIterations > 0u &&
        std::isfinite(config.convergenceTolerance) &&
        config.convergenceTolerance > 0.0f &&
        std::isfinite(config.diagonalFloor) &&
        config.diagonalFloor > 0.0f &&
        representableFloat(config.evaluation.timestep) &&
        config.evaluation.timestep > 0.0 &&
        representableFloat(
            config.evaluation.penetrationSlop
        ) &&
        config.evaluation.penetrationSlop >= 0.0 &&
        representableFloat(
            config.evaluation.maximumDepenetrationVelocity
        ) &&
        config.evaluation.maximumDepenetrationVelocity >= 0.0 &&
        representableFloat(
            config.evaluation.minimumTimeConstantRatio
        ) &&
        config.evaluation.minimumTimeConstantRatio >= 0.0 &&
        representableFloat(
            config.evaluation.stictionTransitionVelocity
        ) &&
        config.evaluation.stictionTransitionVelocity >= 0.0 &&
        representableFloat(
            config.evaluation.minimumRegularization
        ) &&
        config.evaluation.minimumRegularization >= 0.0;
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

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name,
    NSError** error
) {
    id<MTLFunction> function =
        [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    return [device
        newComputePipelineStateWithFunction:function
                                       error:error];
}

} // namespace

MetalMultiArticulatedConstraintDiagnostics
solveMetalMultiArticulatedConstraints(
    const CompiledMetalMultiArticulatedProgram& program,
    const MetalMultiArticulatedConstraintInput& input,
    MetalMultiArticulatedConstraintResult& output,
    const MetalMultiArticulatedConstraintConfig& config
) {
    @autoreleasepool {
        MetalMultiArticulatedConstraintDiagnostics diagnostics{};
        if (!program.valid()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::invalidModel,
                "compiled multi-articulation program is invalid"
            );
        }
        const EngineModel& model = program.model();
        const std::span<const float> jacobian =
            program.generalizedJacobian();
        if (!validConfiguration(config)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    invalidConfiguration,
                "generalized constraint configuration is invalid"
            );
        }
        const std::size_t rowCount = program.rowCount();
        if (model.articulations.size() >
                std::numeric_limits<mr_u32>::max() ||
            input.environmentCount == 0u ||
            input.environmentCount >
                std::numeric_limits<mr_u32>::max()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    invalidDimensions,
                "environment or articulation count is outside "
                "the GPU ABI"
            );
        }
        MetalMultiArticulatedConstraintLayout layout;
        std::size_t environmentRows = 0u;
        std::size_t environmentRowPairs = 0u;
        if (!checkedMultiply(
                input.environmentCount,
                model.world.nq,
                layout.qElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                model.world.nv,
                layout.velocityElements
            ) ||
            !checkedMultiply(
                rowCount,
                model.world.nv,
                layout.jacobianElements
            ) ||
            !checkedMultiply(
                input.environmentCount,
                rowCount,
                environmentRows
            ) ||
            !checkedMultiply(
                environmentRows,
                model.world.nv,
                layout.responseElements
            ) ||
            !checkedMultiply(
                environmentRows,
                rowCount,
                environmentRowPairs
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    arithmeticOverflow,
                "generalized constraint element-count overflow"
            );
        }
        layout.delassusElements = environmentRowPairs;
        layout.impulseElements = environmentRows;
        if (input.q.size() != layout.qElements ||
            input.freeVelocity.size() !=
                layout.velocityElements) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    invalidDimensions,
                "q or free-velocity span has the wrong dimensions"
            );
        }
        if (!std::ranges::all_of(
                input.q,
                [](const float value) {
                    return std::isfinite(value);
                }
            ) ||
            !std::ranges::all_of(
                input.freeVelocity,
                [](const float value) {
                    return std::isfinite(value);
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    nonfiniteInput,
                "q or free velocity contains a non-finite value"
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
                        MetalMultiArticulatedConstraintStatus::
                            nonfiniteInput,
                        "floating-root quaternion is not normalized"
                    );
                }
            }
        }

        const std::size_t rowChunkCount =
            program.rowChunkOffsets().size();
        std::size_t inverseWorkCount = 0u;
        if (!checkedMultiply(
                rowChunkCount,
                model.articulations.size(),
                inverseWorkCount
            ) ||
            !checkedMultiply(
                inverseWorkCount,
                input.environmentCount,
                layout.inverseStatusElements
            ) ||
            inverseWorkCount >
                std::numeric_limits<mr_u32>::max() ||
            layout.inverseStatusElements >
                std::numeric_limits<mr_u32>::max() ||
            model.world.nq >
                std::numeric_limits<mr_u32>::max() ||
            model.world.nv >
                std::numeric_limits<mr_u32>::max() ||
            rowCount * model.world.nv >
                std::numeric_limits<mr_u32>::max()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    arithmeticOverflow,
                "inverse-mass work exceeds the 32-bit GPU ABI"
            );
        }
        layout.inverseMassDispatches.reserve(inverseWorkCount);
        for (std::size_t chunkIndex = 0u;
             chunkIndex < rowChunkCount;
             ++chunkIndex) {
            const std::size_t rowBegin =
                program.rowChunkOffsets()[chunkIndex];
            const std::size_t chunkRows =
                program.rowChunkCounts()[chunkIndex];
            for (std::size_t articulationIndex = 0u;
                 articulationIndex < model.articulations.size();
                 ++articulationIndex) {
                const MRArticulationGPU& articulation =
                    model.articulations[articulationIndex];
                MRMultiInverseMassDispatchGPU work{};
                work.dispatch.articulationIndex =
                    static_cast<std::uint32_t>(
                        articulationIndex
                    );
                work.dispatch.environmentCount =
                    static_cast<std::uint32_t>(
                        input.environmentCount
                    );
                work.dispatch.rhsCount =
                    static_cast<std::uint32_t>(chunkRows);
                work.dispatch.qStride = model.world.nq;
                work.dispatch.rhsEnvironmentStride =
                    static_cast<std::uint32_t>(
                        rowCount * model.world.nv
                    );
                work.dispatch.rhsVectorStride = model.world.nv;
                work.dispatch.outputEnvironmentStride =
                    static_cast<std::uint32_t>(
                        rowCount * model.world.nv
                    );
                work.dispatch.outputVectorStride =
                    model.world.nv;
                work.qBase = articulation.qOffset;
                work.rhsBase = static_cast<std::uint32_t>(
                    rowBegin * model.world.nv +
                    articulation.vOffset
                );
                work.outputBase = work.rhsBase;
                work.statusBase =
                    static_cast<std::uint32_t>(
                        layout.inverseMassDispatches.size() *
                            input.environmentCount
                    );
                layout.inverseMassDispatches.push_back(work);
            }
        }

        layout.dispatch.abiVersion =
            MR_GENERALIZED_CONSTRAINT_ABI_VERSION;
        layout.dispatch.environmentCount =
            static_cast<std::uint32_t>(input.environmentCount);
        layout.dispatch.nv = model.world.nv;
        layout.dispatch.rowCount =
            static_cast<std::uint32_t>(rowCount);
        layout.dispatch.inverseWorkCount =
            static_cast<std::uint32_t>(inverseWorkCount);
        layout.dispatch.solverIterations =
            config.solverIterations;
        layout.dispatch.evaluation0 = {
            static_cast<float>(config.evaluation.timestep),
            static_cast<float>(
                config.evaluation.penetrationSlop
            ),
            static_cast<float>(
                config.evaluation
                    .maximumDepenetrationVelocity
            ),
            static_cast<float>(
                config.evaluation.minimumTimeConstantRatio
            ),
        };
        layout.dispatch.evaluation1 = {
            static_cast<float>(
                config.evaluation.minimumRegularization
            ),
            config.convergenceTolerance,
            config.diagonalFloor,
            0.0f,
        };

        std::vector<float> rhs(layout.responseElements);
        for (std::size_t environment = 0u;
             environment < input.environmentCount;
             ++environment) {
            std::copy(
                jacobian.begin(),
                jacobian.end(),
                rhs.begin() +
                    environment * layout.jacobianElements
            );
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
            !addBytes<MRMultiInverseMassDispatchGPU>(
                layout.inverseMassDispatches.size(),
                bytes
            ) ||
            !addBytes<float>(layout.qElements, bytes) ||
            !addBytes<float>(layout.responseElements, bytes) ||
            !addBytes<float>(layout.responseElements, bytes) ||
            !addBytes<MRInverseMassStatusGPU>(
                layout.inverseStatusElements,
                bytes
            ) ||
            !addBytes<MRGeneralizedConstraintDispatchGPU>(
                1u,
                bytes
            ) ||
            !addBytes<MRConstraintIRRowGPU>(rowCount, bytes) ||
            !addBytes<float>(rowCount, bytes) ||
            !addBytes<float>(
                layout.jacobianElements,
                bytes
            ) ||
            !addBytes<float>(
                layout.velocityElements,
                bytes
            ) ||
            !addBytes<float>(
                layout.delassusElements,
                bytes
            ) ||
            !addBytes<float>(
                layout.impulseElements,
                bytes
            ) ||
            !addBytes<float>(
                layout.velocityElements,
                bytes
            ) ||
            !addBytes<MRGeneralizedConstraintStatusGPU>(
                input.environmentCount,
                bytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    arithmeticOverflow,
                "generalized constraint byte-count overflow"
            );
        }
        layout.totalAllocatedBytes = bytes;
        diagnostics.layout = layout;

        const std::string metallibPath =
            config.metallibPath.empty()
            ? defaultMetallibPath()
            : config.metallibPath;
        if (metallibPath.empty()) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
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
                MetalMultiArticulatedConstraintStatus::
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
                MetalMultiArticulatedConstraintStatus::
                    metalLibraryFailure,
                "failed to load metallib: " + errorString(error)
            );
        }
        error = nil;
        id<MTLComputePipelineState> inversePipeline = pipeline(
            device,
            library,
            @"mr_multi_articulated_inverse_mass",
            &error
        );
        if (inversePipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalPipelineFailure,
                "failed to create inverse-mass pipeline: " +
                    errorString(error)
            );
        }
        error = nil;
        id<MTLComputePipelineState> delassusPipeline = pipeline(
            device,
            library,
            @"mr_generalized_constraint_delassus",
            &error
        );
        if (delassusPipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalPipelineFailure,
                "failed to create Delassus pipeline: " +
                    errorString(error)
            );
        }
        error = nil;
        id<MTLComputePipelineState> solvePipeline = pipeline(
            device,
            library,
            @"mr_generalized_constraint_solve",
            &error
        );
        if (solvePipeline == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalPipelineFailure,
                "failed to create generalized solve pipeline: " +
                    errorString(error)
            );
        }
        if (inversePipeline.threadExecutionWidth != kWaveWidth ||
            solvePipeline.threadExecutionWidth != kWaveWidth ||
            inversePipeline.maxTotalThreadsPerThreadgroup <
                kWaveWidth ||
            solvePipeline.maxTotalThreadsPerThreadgroup <
                kWaveWidth) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalDeviceUnsupported,
                "generalized constraint graph requires SIMD32"
            );
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer =
            [queue commandBuffer];
        if (queue == nil || commandBuffer == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalDeviceUnavailable,
                "failed to create Metal queue or command buffer"
            );
        }

        id<MTLBuffer> worldBuffer =
            inputBuffer(device, &model.world, 1u);
        id<MTLBuffer> articulationBuffer = inputBuffer(
            device,
            model.articulations.data(),
            model.articulations.size()
        );
        id<MTLBuffer> jointBuffer = inputBuffer(
            device,
            model.joints.data(),
            model.joints.size()
        );
        id<MTLBuffer> dofBuffer = inputBuffer(
            device,
            model.dofs.data(),
            model.dofs.size()
        );
        id<MTLBuffer> bodyBuffer = inputBuffer(
            device,
            model.bodies.data(),
            model.bodies.size()
        );
        id<MTLBuffer> inverseDispatchBuffer = inputBuffer(
            device,
            layout.inverseMassDispatches.data(),
            layout.inverseMassDispatches.size()
        );
        id<MTLBuffer> qBuffer = inputBuffer(
            device,
            input.q.data(),
            layout.qElements
        );
        id<MTLBuffer> rhsBuffer = inputBuffer(
            device,
            rhs.data(),
            layout.responseElements
        );
        id<MTLBuffer> responseBuffer = outputBuffer<float>(
            device,
            layout.responseElements
        );
        id<MTLBuffer> inverseStatusBuffer =
            outputBuffer<MRInverseMassStatusGPU>(
                device,
                layout.inverseStatusElements
            );
        id<MTLBuffer> dispatchBuffer = inputBuffer(
            device,
            &layout.dispatch,
            1u
        );
        id<MTLBuffer> sourceRowBuffer = inputBuffer(
            device,
            model.constraintProgram.rows.data(),
            rowCount
        );
        id<MTLBuffer> warmImpulseBuffer = inputBuffer(
            device,
            model.constraintProgram.warmImpulses.data(),
            rowCount
        );
        id<MTLBuffer> jacobianBuffer = inputBuffer(
            device,
            jacobian.data(),
            layout.jacobianElements
        );
        id<MTLBuffer> freeVelocityBuffer = inputBuffer(
            device,
            input.freeVelocity.data(),
            layout.velocityElements
        );
        id<MTLBuffer> delassusBuffer = outputBuffer<float>(
            device,
            layout.delassusElements
        );
        id<MTLBuffer> impulseBuffer = outputBuffer<float>(
            device,
            layout.impulseElements
        );
        id<MTLBuffer> nextVelocityBuffer = outputBuffer<float>(
            device,
            layout.velocityElements
        );
        id<MTLBuffer> statusBuffer =
            outputBuffer<MRGeneralizedConstraintStatusGPU>(
                device,
                input.environmentCount
            );
        const std::array buffers{
            worldBuffer,
            articulationBuffer,
            jointBuffer,
            dofBuffer,
            bodyBuffer,
            inverseDispatchBuffer,
            qBuffer,
            rhsBuffer,
            responseBuffer,
            inverseStatusBuffer,
            dispatchBuffer,
            sourceRowBuffer,
            warmImpulseBuffer,
            jacobianBuffer,
            freeVelocityBuffer,
            delassusBuffer,
            impulseBuffer,
            nextVelocityBuffer,
            statusBuffer,
        };
        if (std::ranges::any_of(
                buffers,
                [](id<MTLBuffer> buffer) {
                    return buffer == nil;
                }
            )) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalBufferFailure,
                "failed to allocate generalized constraint buffers"
            );
        }

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    metalCommandFailure,
                "failed to create generalized constraint encoder"
            );
        }
        [encoder setComputePipelineState:inversePipeline];
        const std::array inverseBuffers{
            worldBuffer,
            articulationBuffer,
            jointBuffer,
            dofBuffer,
            bodyBuffer,
            inverseDispatchBuffer,
            qBuffer,
            rhsBuffer,
            responseBuffer,
            inverseStatusBuffer,
        };
        for (NSUInteger index = 0u;
             index < inverseBuffers.size();
             ++index) {
            [encoder setBuffer:inverseBuffers[index]
                        offset:0u
                       atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(input.environmentCount),
                static_cast<NSUInteger>(inverseWorkCount),
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWaveWidth,
                1u,
                1u
            )];
        const std::array inverseProducts{
            responseBuffer,
            inverseStatusBuffer,
        };
        [encoder
            memoryBarrierWithResources:inverseProducts.data()
                                 count:inverseProducts.size()];

        [encoder setComputePipelineState:delassusPipeline];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:jacobianBuffer offset:0u atIndex:1u];
        [encoder setBuffer:responseBuffer offset:0u atIndex:2u];
        [encoder setBuffer:delassusBuffer offset:0u atIndex:3u];
        [encoder
            dispatchThreads:MTLSizeMake(
                static_cast<NSUInteger>(rowCount),
                static_cast<NSUInteger>(rowCount),
                static_cast<NSUInteger>(input.environmentCount)
            )
            threadsPerThreadgroup:MTLSizeMake(8u, 8u, 1u)];
        const std::array delassusProducts{delassusBuffer};
        [encoder
            memoryBarrierWithResources:delassusProducts.data()
                                 count:delassusProducts.size()];

        [encoder setComputePipelineState:solvePipeline];
        const std::array solveBuffers{
            dispatchBuffer,
            sourceRowBuffer,
            warmImpulseBuffer,
            jacobianBuffer,
            freeVelocityBuffer,
            responseBuffer,
            inverseStatusBuffer,
            delassusBuffer,
            impulseBuffer,
            nextVelocityBuffer,
            statusBuffer,
        };
        for (NSUInteger index = 0u;
             index < solveBuffers.size();
             ++index) {
            [encoder setBuffer:solveBuffers[index]
                        offset:0u
                       atIndex:index];
        }
        [encoder
            dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(input.environmentCount),
                1u,
                1u
            )
            threadsPerThreadgroup:MTLSizeMake(
                kWaveWidth,
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
                MetalMultiArticulatedConstraintStatus::
                    metalCommandFailure,
                "generalized constraint command failed: " +
                    errorString(commandBuffer.error)
            );
        }

        MetalMultiArticulatedConstraintResult staged;
        staged.layout = layout;
        staged.nextVelocity.resize(layout.velocityElements);
        staged.impulses.resize(layout.impulseElements);
        staged.statuses.resize(input.environmentCount);
        staged.inverseMassStatuses.resize(
            layout.inverseStatusElements
        );
        std::memcpy(
            staged.nextVelocity.data(),
            nextVelocityBuffer.contents,
            layout.velocityElements * sizeof(float)
        );
        std::memcpy(
            staged.impulses.data(),
            impulseBuffer.contents,
            layout.impulseElements * sizeof(float)
        );
        std::memcpy(
            staged.statuses.data(),
            statusBuffer.contents,
            input.environmentCount *
                sizeof(MRGeneralizedConstraintStatusGPU)
        );
        std::memcpy(
            staged.inverseMassStatuses.data(),
            inverseStatusBuffer.contents,
            layout.inverseStatusElements *
                sizeof(MRInverseMassStatusGPU)
        );
        for (std::size_t environment = 0u;
             environment < input.environmentCount;
             ++environment) {
            const MRGeneralizedConstraintStatusGPU& status =
                staged.statuses[environment];
            if (status.code !=
                    MR_GENERALIZED_CONSTRAINT_SUCCESS ||
                status.environment != environment) {
                diagnostics.firstFailingEnvironment =
                    static_cast<std::uint32_t>(environment);
                diagnostics.firstGPUStatusCode = status.code;
                diagnostics.firstFailingRow =
                    status.failingRow;
                diagnostics.firstFailingInverseWork =
                    status.failingInverseWork;
                return reject(
                    std::move(diagnostics),
                    MetalMultiArticulatedConstraintStatus::
                        gpuEnvironmentFailure,
                    "a generalized constraint environment failed "
                    "(iterations=" +
                        std::to_string(status.iterations) +
                    ", impulse_delta=" +
                        std::to_string(status.diagnostics.x) +
                    ", natural_residual=" +
                        std::to_string(status.diagnostics.y) +
                    ")"
                );
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
        if (!finiteVector(staged.nextVelocity) ||
            !finiteVector(staged.impulses)) {
            return reject(
                std::move(diagnostics),
                MetalMultiArticulatedConstraintStatus::
                    nonfiniteResult,
                "GPU produced a non-finite constraint result"
            );
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    }
}

MetalMultiArticulatedConstraintDiagnostics
solveMetalMultiArticulatedConstraints(
    const EngineModel& model,
    const MetalMultiArticulatedConstraintInput& input,
    MetalMultiArticulatedConstraintResult& output,
    const MetalMultiArticulatedConstraintConfig& config
) {
    CompiledMetalMultiArticulatedProgram program;
    MetalMultiArticulatedConstraintDiagnostics diagnostics =
        compileMetalMultiArticulatedProgram(model, program);
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return solveMetalMultiArticulatedConstraints(
        program,
        input,
        output,
        config
    );
}

const char* metalMultiArticulatedConstraintStatusName(
    const MetalMultiArticulatedConstraintStatus status
) noexcept {
    switch (status) {
    case MetalMultiArticulatedConstraintStatus::success:
        return "success";
    case MetalMultiArticulatedConstraintStatus::
            invalidConfiguration:
        return "invalid_configuration";
    case MetalMultiArticulatedConstraintStatus::invalidModel:
        return "invalid_model";
    case MetalMultiArticulatedConstraintStatus::
            unsupportedTopology:
        return "unsupported_topology";
    case MetalMultiArticulatedConstraintStatus::
            unsupportedConstraint:
        return "unsupported_constraint";
    case MetalMultiArticulatedConstraintStatus::
            invalidDimensions:
        return "invalid_dimensions";
    case MetalMultiArticulatedConstraintStatus::
            arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalMultiArticulatedConstraintStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalMultiArticulatedConstraintStatus::
            metallibUnavailable:
        return "metallib_unavailable";
    case MetalMultiArticulatedConstraintStatus::
            metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalMultiArticulatedConstraintStatus::
            metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalMultiArticulatedConstraintStatus::
            metalLibraryFailure:
        return "metal_library_failure";
    case MetalMultiArticulatedConstraintStatus::
            metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalMultiArticulatedConstraintStatus::
            metalBufferFailure:
        return "metal_buffer_failure";
    case MetalMultiArticulatedConstraintStatus::
            metalCommandFailure:
        return "metal_command_failure";
    case MetalMultiArticulatedConstraintStatus::
            gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalMultiArticulatedConstraintStatus::
            nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
