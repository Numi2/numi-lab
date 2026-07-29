#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalUnifiedQualitySolver.hpp"

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
const char kUnifiedQualityImageAnchor = 0;

MetalUnifiedQualityDiagnostics fail(
    MetalUnifiedQualityDiagnostics diagnostics,
    const MetalUnifiedQualityHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finiteVector(const std::span<const float> values) {
    return std::ranges::all_of(values, [](const float value) {
        return finite(value);
    });
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
    return description.empty()
        ? nsString(error.description)
        : description;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kUnifiedQualityImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname)
                .parent_path();
        const std::array candidates{
            directory / "metalrobo/MetalRobo.metallib",
            directory.parent_path() /
                "shaders/MetalRobo.metallib",
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
    std::size_t& result
) {
    if (left != 0u &&
        right >
            std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedBytes(
    const std::size_t count,
    const std::size_t elementSize,
    NSUInteger& output
) {
    std::size_t bytes = 0u;
    if (!checkedMultiply(count, elementSize, bytes) ||
        bytes == 0u ||
        bytes >
            std::numeric_limits<NSUInteger>::max()) {
        return false;
    }
    output = static_cast<NSUInteger>(bytes);
    return true;
}

bool validConfig(const MetalUnifiedQualityConfig& config) {
    return
        config.maximumNewtonIterations > 0u &&
        config.maximumPCGIterations > 0u &&
        config.maximumLineSearchIterations > 0u &&
        config.directMaximumGeneralizedVelocities > 0u &&
        config.directMaximumRows > 0u &&
        finite(config.optimalityTolerance) &&
        config.optimalityTolerance > 0.0f &&
        finite(config.feasibilityTolerance) &&
        config.feasibilityTolerance > 0.0f &&
        finite(config.armijoConstant) &&
        config.armijoConstant > 0.0f &&
        config.armijoConstant < 0.5f &&
        finite(config.lineSearchContraction) &&
        config.lineSearchContraction > 0.0f &&
        config.lineSearchContraction < 1.0f &&
        finite(config.complianceFloorMultiplier) &&
        config.complianceFloorMultiplier > 0.0f &&
        finite(config.minimumPivot) &&
        config.minimumPivot > 0.0f &&
        finite(config.minimumPCGDenominator) &&
        config.minimumPCGDenominator > 0.0f &&
        finite(config.regularizationRetryScale) &&
        config.regularizationRetryScale > 1.0f;
}

float blockFloat(
    const mr_float4 first,
    const mr_float4 second,
    const std::size_t index
) {
    const float* values = index < 4u
        ? &first.x
        : &second.x;
    return values[index < 4u ? index : index - 4u];
}

bool validateProblem(
    const MetalUnifiedQualityProblem& problem,
    std::string& message
) {
    const std::size_t problemCount = problem.problemCount;
    const std::size_t nv =
        problem.generalizedVelocityCount;
    const std::size_t rows = problem.rowCount;
    if (problemCount == 0u || nv == 0u || rows == 0u ||
        nv >
            MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES ||
        rows > MR_UNIFIED_QUALITY_MAX_ROWS ||
        problem.blocks.empty() ||
        problem.blocks.size() >
            MR_UNIFIED_QUALITY_MAX_BLOCKS) {
        message =
            "unified quality dimensions exceed the compiled buckets";
        return false;
    }
    std::size_t dynamicsElements = 0u;
    std::size_t jacobianElements = 0u;
    std::size_t vectorElements = 0u;
    if (!checkedMultiply(problemCount, nv * nv,
                         dynamicsElements) ||
        !checkedMultiply(problemCount, rows * nv,
                         jacobianElements) ||
        !checkedMultiply(
            problemCount,
            std::max(nv, rows),
            vectorElements
        )) {
        message = "unified quality dimensions overflow";
        return false;
    }
    const std::size_t velocityElements = problemCount * nv;
    const std::size_t rowElements = problemCount * rows;
    if (problem.dynamics.size() != dynamicsElements ||
        problem.jacobian.size() != jacobianElements ||
        problem.bias.size() != rowElements ||
        problem.freeVelocity.size() != velocityElements ||
        problem.warmVelocity.size() != velocityElements ||
        problem.warmImpulses.size() != rowElements ||
        !finiteVector(problem.dynamics) ||
        !finiteVector(problem.jacobian) ||
        !finiteVector(problem.bias) ||
        !finiteVector(problem.freeVelocity) ||
        !finiteVector(problem.warmVelocity) ||
        !finiteVector(problem.warmImpulses)) {
        message =
            "unified quality arrays are inconsistent or non-finite";
        return false;
    }

    std::vector<bool> covered(rows, false);
    for (std::size_t blockIndex = 0u;
         blockIndex < problem.blocks.size();
         ++blockIndex) {
        const MRUnifiedQualityBlockGPU& block =
            problem.blocks[blockIndex];
        const std::size_t offset = block.layout.x;
        const std::size_t dimension = block.layout.y;
        const std::uint32_t kind = block.layout.z;
        constexpr std::uint32_t knownFlags =
            MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY |
            MR_UNIFIED_QUALITY_BLOCK_REPORT_FLOOR;
        if (dimension == 0u ||
            dimension >
                MR_UNIFIED_QUALITY_MAX_BLOCK_DIMENSION ||
            offset > rows ||
            dimension > rows - offset ||
            (block.layout.w & ~knownFlags) != 0u ||
            ((kind ==
                  MR_UNIFIED_QUALITY_SCALAR_INTERVAL &&
              dimension != 1u) ||
             (kind ==
                  MR_UNIFIED_QUALITY_ELLIPTIC_CONE &&
              dimension < 3u) ||
             (kind !=
                  MR_UNIFIED_QUALITY_SCALAR_INTERVAL &&
              kind !=
                  MR_UNIFIED_QUALITY_ELLIPTIC_CONE))) {
            message =
                "unified quality block topology is invalid at block " +
                std::to_string(blockIndex);
            return false;
        }
        for (std::size_t local = 0u;
             local < dimension;
             ++local) {
            if (covered[offset + local]) {
                message =
                    "unified quality blocks overlap at row " +
                    std::to_string(offset + local);
                return false;
            }
            covered[offset + local] = true;
            const float scale = blockFloat(
                block.scale0,
                block.scale1,
                local
            );
            const float regularization = blockFloat(
                block.regularization0,
                block.regularization1,
                local
            );
            if (!finite(scale) || !(scale > 0.0f) ||
                !finite(regularization) ||
                !(regularization > 0.0f)) {
                message =
                    "unified quality block scale or regularization "
                    "is invalid";
                return false;
            }
        }
        if (kind ==
                MR_UNIFIED_QUALITY_SCALAR_INTERVAL &&
            (!finite(block.boundsAndShift.x) ||
             !finite(block.boundsAndShift.y) ||
             block.boundsAndShift.x >
                 block.boundsAndShift.y)) {
            message =
                "unified scalar interval bounds are invalid";
            return false;
        }
        if (kind ==
                MR_UNIFIED_QUALITY_ELLIPTIC_CONE &&
            (std::abs(block.scale0.x - 1.0f) >
                 8.0f *
                     std::numeric_limits<float>::epsilon() ||
             !finite(block.boundsAndShift.x) ||
             block.boundsAndShift.x < 0.0f ||
             !finite(block.boundsAndShift.y) ||
             block.boundsAndShift.y < 0.0f)) {
            message =
                "unified cone shift, cap, or normal scale is invalid";
            return false;
        }
    }
    if (std::ranges::find(covered, false) != covered.end()) {
        message =
            "unified quality product blocks do not cover every row";
        return false;
    }

    for (std::size_t environment = 0u;
         environment < problemCount;
         ++environment) {
        const std::size_t base = environment * nv * nv;
        for (std::size_t row = 0u; row < nv; ++row) {
            const float diagonal =
                problem.dynamics[base + row * nv + row];
            if (!(diagonal > 0.0f)) {
                message =
                    "unified dynamics has a non-positive diagonal";
                return false;
            }
            for (std::size_t column = row + 1u;
                 column < nv;
                 ++column) {
                const float value =
                    problem.dynamics[
                        base + row * nv + column
                    ];
                const float transpose =
                    problem.dynamics[
                        base + column * nv + row
                    ];
                const float scale =
                    1.0f +
                    std::max(
                        std::abs(value),
                        std::abs(transpose)
                    );
                if (std::abs(value - transpose) >
                    2.0e-5f * scale) {
                    message =
                        "unified dynamics is not symmetric";
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T>
id<MTLBuffer> inputBuffer(
    id<MTLDevice> device,
    const std::span<const T> values
) {
    NSUInteger bytes = 0u;
    if (!checkedBytes(values.size(), sizeof(T), bytes)) {
        return nil;
    }
    return [device
        newBufferWithBytes:values.data()
                   length:bytes
                  options:MTLResourceStorageModeShared];
}

id<MTLBuffer> outputBuffer(
    id<MTLDevice> device,
    const std::size_t elements,
    const std::size_t elementSize
) {
    NSUInteger bytes = 0u;
    if (!checkedBytes(elements, elementSize, bytes)) {
        return nil;
    }
    id<MTLBuffer> result = [device
        newBufferWithLength:bytes
                   options:MTLResourceStorageModeShared];
    if (result != nil && result.contents != nullptr) {
        std::memset(result.contents, 0, bytes);
    }
    return result;
}

} // namespace

MetalUnifiedQualityDiagnostics solveMetalUnifiedQuality(
    const MetalUnifiedQualityProblem& problem,
    MetalUnifiedQualityResult& output,
    const MetalUnifiedQualityConfig& config
) {
    MetalUnifiedQualityDiagnostics diagnostics;
    if (!validConfig(config)) {
        return fail(
            std::move(diagnostics),
            MetalUnifiedQualityHostStatus::
                invalidConfiguration,
            "unified quality solver configuration is invalid"
        );
    }
    std::string problemMessage;
    if (!validateProblem(problem, problemMessage)) {
        return fail(
            std::move(diagnostics),
            MetalUnifiedQualityHostStatus::invalidProblem,
            std::move(problemMessage)
        );
    }

    @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::
                    metalUnavailable,
                "no Metal device is available"
            );
        }
        diagnostics.deviceName = nsString(device.name);
        std::string metallibPath = config.metallibPath;
        if (metallibPath.empty()) {
            metallibPath = defaultMetallibPath();
        }
        if (metallibPath.empty()) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::
                    metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:
                [NSURL fileURLWithPath:
                    [NSString
                        stringWithUTF8String:
                            metallibPath.c_str()]]
                        error:&error];
        if (library == nil) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::
                    metallibUnavailable,
                "failed to load metallib: " +
                    describeError(error)
            );
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"mr_unified_quality_solve"];
        if (function == nil) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::pipelineFailure,
                "metallib has no unified quality kernel"
            );
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function
                                           error:&error];
        if (pipeline == nil ||
            pipeline.threadExecutionWidth != kThreadgroupWidth ||
            pipeline.maxTotalThreadsPerThreadgroup <
                kThreadgroupWidth ||
            pipeline.staticThreadgroupMemoryLength >
                device.maxThreadgroupMemoryLength) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::
                    deviceUnsupported,
                "unified quality SIMD32 pipeline is unsupported: " +
                    describeError(error)
            );
        }

        const std::size_t vectorStride = std::max(
            problem.generalizedVelocityCount,
            problem.rowCount
        );
        const std::size_t derivativeStride =
            problem.blocks.size() * 36u;
        const std::size_t dynamicsStride =
            problem.generalizedVelocityCount *
            problem.generalizedVelocityCount;
        const bool directPath =
            problem.generalizedVelocityCount <=
                config.directMaximumGeneralizedVelocities &&
            problem.rowCount <= config.directMaximumRows;
        const std::size_t hessianStride =
            directPath
            ? dynamicsStride
            : 1u;
        if (problem.problemCount >
                std::numeric_limits<std::uint32_t>::max() ||
            problem.generalizedVelocityCount >
                std::numeric_limits<std::uint32_t>::max() ||
            problem.rowCount >
                std::numeric_limits<std::uint32_t>::max() ||
            problem.blocks.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            hessianStride >
                std::numeric_limits<std::uint32_t>::max() ||
            derivativeStride >
                std::numeric_limits<std::uint32_t>::max() ||
            vectorStride >
                std::numeric_limits<std::uint32_t>::max()) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::
                    arithmeticOverflow,
                "unified quality dispatch exceeds 32-bit indexing"
            );
        }
        MRUnifiedQualityDispatchGPU dispatch{};
        dispatch.abiVersion =
            MR_UNIFIED_QUALITY_ABI_VERSION;
        dispatch.problemCount =
            static_cast<std::uint32_t>(
                problem.problemCount
            );
        dispatch.generalizedVelocityCount =
            static_cast<std::uint32_t>(
                problem.generalizedVelocityCount
            );
        dispatch.rowCount =
            static_cast<std::uint32_t>(problem.rowCount);
        dispatch.blockCount =
            static_cast<std::uint32_t>(
                problem.blocks.size()
            );
        dispatch.dynamicsStride =
            static_cast<std::uint32_t>(dynamicsStride);
        dispatch.jacobianStride =
            static_cast<std::uint32_t>(
                problem.rowCount *
                problem.generalizedVelocityCount
            );
        dispatch.vectorStride =
            static_cast<std::uint32_t>(vectorStride);
        dispatch.maximumNewtonIterations =
            config.maximumNewtonIterations;
        dispatch.maximumPCGIterations =
            config.maximumPCGIterations;
        dispatch.maximumLineSearchIterations =
            config.maximumLineSearchIterations;
        dispatch.directMaximumGeneralizedVelocities =
            config.directMaximumGeneralizedVelocities;
        dispatch.directMaximumRows =
            config.directMaximumRows;
        dispatch.derivativeStride =
            static_cast<std::uint32_t>(derivativeStride);
        dispatch.hessianStride =
            static_cast<std::uint32_t>(hessianStride);
        dispatch.tolerances = {
            config.optimalityTolerance,
            config.feasibilityTolerance,
            config.armijoConstant,
            config.lineSearchContraction,
        };
        dispatch.numerics = {
            config.complianceFloorMultiplier,
            config.minimumPivot,
            config.minimumPCGDenominator,
            config.regularizationRetryScale,
        };

        std::vector<float> packedBias(
            problem.problemCount * vectorStride,
            0.0f
        );
        std::vector<float> packedFree(
            problem.problemCount * vectorStride,
            0.0f
        );
        std::vector<float> packedWarm(
            problem.problemCount * vectorStride,
            0.0f
        );
        std::vector<float> packedWarmImpulses(
            problem.problemCount * vectorStride,
            0.0f
        );
        for (std::size_t environment = 0u;
             environment < problem.problemCount;
             ++environment) {
            std::ranges::copy(
                problem.bias.subspan(
                    environment * problem.rowCount,
                    problem.rowCount
                ),
                packedBias.begin() +
                    environment * vectorStride
            );
            std::ranges::copy(
                problem.freeVelocity.subspan(
                    environment *
                        problem.generalizedVelocityCount,
                    problem.generalizedVelocityCount
                ),
                packedFree.begin() +
                    environment * vectorStride
            );
            std::ranges::copy(
                problem.warmVelocity.subspan(
                    environment *
                        problem.generalizedVelocityCount,
                    problem.generalizedVelocityCount
                ),
                packedWarm.begin() +
                    environment * vectorStride
            );
            std::ranges::copy(
                problem.warmImpulses.subspan(
                    environment * problem.rowCount,
                    problem.rowCount
                ),
                packedWarmImpulses.begin() +
                    environment * vectorStride
            );
        }

        id<MTLBuffer> dispatchBuffer = [device
            newBufferWithBytes:&dispatch
                       length:sizeof(dispatch)
                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> blockBuffer =
            inputBuffer<MRUnifiedQualityBlockGPU>(
                device,
                problem.blocks
            );
        id<MTLBuffer> dynamicsBuffer =
            inputBuffer<float>(device, problem.dynamics);
        id<MTLBuffer> jacobianBuffer =
            inputBuffer<float>(device, problem.jacobian);
        id<MTLBuffer> biasBuffer = inputBuffer<float>(
            device,
            packedBias
        );
        id<MTLBuffer> freeBuffer = inputBuffer<float>(
            device,
            packedFree
        );
        id<MTLBuffer> warmBuffer = inputBuffer<float>(
            device,
            packedWarm
        );
        id<MTLBuffer> warmImpulseBuffer =
            inputBuffer<float>(
                device,
                packedWarmImpulses
            );
        id<MTLBuffer> velocityBuffer = outputBuffer(
            device,
            problem.problemCount * vectorStride,
            sizeof(float)
        );
        id<MTLBuffer> impulseBuffer = outputBuffer(
            device,
            problem.problemCount * vectorStride,
            sizeof(float)
        );
        id<MTLBuffer> derivativeBuffer = outputBuffer(
            device,
            problem.problemCount * derivativeStride,
            sizeof(float)
        );
        id<MTLBuffer> hessianBuffer = outputBuffer(
            device,
            problem.problemCount * hessianStride,
            sizeof(float)
        );
        id<MTLBuffer> statusBuffer = outputBuffer(
            device,
            problem.problemCount,
            sizeof(MRUnifiedQualityStatusGPU)
        );
        const std::array buffers{
            dispatchBuffer,
            blockBuffer,
            dynamicsBuffer,
            jacobianBuffer,
            biasBuffer,
            freeBuffer,
            warmBuffer,
            warmImpulseBuffer,
            velocityBuffer,
            impulseBuffer,
            derivativeBuffer,
            hessianBuffer,
            statusBuffer,
        };
        if (std::ranges::find(buffers, nil) !=
            buffers.end()) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::bufferFailure,
                "failed to allocate unified quality buffers"
            );
        }
        for (id<MTLBuffer> buffer : buffers) {
            diagnostics.allocatedBytes += buffer.length;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command =
            [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (queue == nil || command == nil || encoder == nil) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::commandFailure,
                "failed to create unified quality command objects"
            );
        }
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u;
             index < buffers.size();
             ++index) {
            [encoder setBuffer:buffers[index]
                        offset:0u
                       atIndex:index];
        }
        [encoder
            dispatchThreadgroups:
                MTLSizeMake(problem.problemCount, 1u, 1u)
            threadsPerThreadgroup:
                MTLSizeMake(kThreadgroupWidth, 1u, 1u)];
        [encoder endEncoding];
        diagnostics.dispatched = true;
        [command commit];
        [command waitUntilCompleted];
        diagnostics.elapsedMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
        if (command.status !=
            MTLCommandBufferStatusCompleted) {
            return fail(
                std::move(diagnostics),
                MetalUnifiedQualityHostStatus::commandFailure,
                "unified quality command failed: " +
                    describeError(command.error)
            );
        }

        const auto* gpuStatuses =
            static_cast<const MRUnifiedQualityStatusGPU*>(
                statusBuffer.contents
            );
        for (std::size_t environment = 0u;
             environment < problem.problemCount;
             ++environment) {
            if (gpuStatuses[environment].code !=
                MR_UNIFIED_QUALITY_SUCCESS) {
                diagnostics.firstFailingProblem =
                    static_cast<std::uint32_t>(environment);
                diagnostics.firstGPUStatusCode =
                    gpuStatuses[environment].code;
                return fail(
                    std::move(diagnostics),
                    MetalUnifiedQualityHostStatus::gpuFailure,
                    "unified quality solve failed in problem " +
                        std::to_string(environment) +
                        " with GPU code " +
                        std::to_string(
                            gpuStatuses[environment].code
                        ) +
                        " block=" +
                        std::to_string(
                            gpuStatuses[environment]
                                .failingBlock
                        ) +
                        " iterations=" +
                        std::to_string(
                            gpuStatuses[environment]
                                .newtonIterations
                        ) +
                        "/" +
                        std::to_string(
                            gpuStatuses[environment]
                                .pcgIterations
                        ) +
                        " certificates=" +
                        std::to_string(
                            gpuStatuses[environment]
                                .certificates0.x
                        ) +
                        "/" +
                        std::to_string(
                            gpuStatuses[environment]
                                .certificates0.y
                        ) +
                        " objective=" +
                        std::to_string(
                            gpuStatuses[environment]
                                .certificates1.w
                        )
                );
            }
        }

        const auto* gpuVelocities =
            static_cast<const float*>(
                velocityBuffer.contents
            );
        const auto* gpuImpulses =
            static_cast<const float*>(
                impulseBuffer.contents
            );
        MetalUnifiedQualityResult staged;
        staged.velocity.resize(
            problem.problemCount *
            problem.generalizedVelocityCount
        );
        staged.impulses.resize(
            problem.problemCount * problem.rowCount
        );
        staged.statuses.assign(
            gpuStatuses,
            gpuStatuses + problem.problemCount
        );
        for (std::size_t environment = 0u;
             environment < problem.problemCount;
             ++environment) {
            for (std::size_t dof = 0u;
                 dof <
                     problem.generalizedVelocityCount;
                 ++dof) {
                const float value = gpuVelocities[
                    environment * vectorStride + dof
                ];
                if (!finite(value)) {
                    return fail(
                        std::move(diagnostics),
                        MetalUnifiedQualityHostStatus::
                            nonfiniteResult,
                        "unified quality velocity is non-finite"
                    );
                }
                staged.velocity[
                    environment *
                        problem.generalizedVelocityCount +
                    dof
                ] = value;
            }
            for (std::size_t row = 0u;
                 row < problem.rowCount;
                 ++row) {
                const float value = gpuImpulses[
                    environment * vectorStride + row
                ];
                if (!finite(value)) {
                    return fail(
                        std::move(diagnostics),
                        MetalUnifiedQualityHostStatus::
                            nonfiniteResult,
                        "unified quality impulse is non-finite"
                    );
                }
                staged.impulses[
                    environment * problem.rowCount + row
                ] = value;
            }
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    }
}

const char* metalUnifiedQualityHostStatusName(
    const MetalUnifiedQualityHostStatus status
) noexcept {
    switch (status) {
    case MetalUnifiedQualityHostStatus::success:
        return "success";
    case MetalUnifiedQualityHostStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalUnifiedQualityHostStatus::invalidProblem:
        return "invalid_problem";
    case MetalUnifiedQualityHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalUnifiedQualityHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalUnifiedQualityHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalUnifiedQualityHostStatus::metalUnavailable:
        return "metal_unavailable";
    case MetalUnifiedQualityHostStatus::deviceUnsupported:
        return "device_unsupported";
    case MetalUnifiedQualityHostStatus::pipelineFailure:
        return "pipeline_failure";
    case MetalUnifiedQualityHostStatus::bufferFailure:
        return "buffer_failure";
    case MetalUnifiedQualityHostStatus::commandFailure:
        return "command_failure";
    case MetalUnifiedQualityHostStatus::gpuFailure:
        return "gpu_failure";
    case MetalUnifiedQualityHostStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
