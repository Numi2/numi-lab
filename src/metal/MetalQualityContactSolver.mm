#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalQualityContactSolver.hpp"

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
const char kMetalQualityImageAnchor = 0;

MetalQualityContactDiagnostics fail(
    MetalQualityContactDiagnostics diagnostics,
    const MetalQualityContactHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const float value) {
    return std::isfinite(value);
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
    return
        std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalQualityImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname).parent_path();
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

bool validConfig(
    const MetalQualityContactSolverConfig& config
) {
    return
        config.maximumNewtonIterations > 0u &&
        config.maximumCGIterations > 0u &&
        config.maximumLineSearchIterations > 0u &&
        finite(config.convergenceTolerance) &&
        config.convergenceTolerance > 0.0f &&
        finite(config.armijoCoefficient) &&
        config.armijoCoefficient > 0.0f &&
        config.armijoCoefficient < 0.5f &&
        finite(config.normalEquationRegularization) &&
        config.normalEquationRegularization > 0.0f &&
        finite(config.minimumCGDenominator) &&
        config.minimumCGDenominator > 0.0f;
}

bool checkedByteCount(
    const std::size_t count,
    const std::size_t size,
    NSUInteger& result
) {
    if (count != 0u &&
        size >
            std::numeric_limits<std::size_t>::max() /
            count) {
        return false;
    }
    const std::size_t bytes = count * size;
    if (bytes == 0u ||
        bytes >
            std::numeric_limits<NSUInteger>::max()) {
        return false;
    }
    result = static_cast<NSUInteger>(bytes);
    return true;
}

struct Prepared {
    std::uint32_t contacts = 0u;
    std::uint32_t dimension = 0u;
    std::vector<float> matrix;
    std::vector<float> linear;
    std::vector<float> warm;
    std::vector<double> scales;
};

bool prepare(
    const ContactSpaceConicProblem& problem,
    const MetalQualityContactSolverConfig& config,
    Prepared& prepared,
    MetalQualityContactHostStatus& status,
    std::string& message
) {
    const std::size_t contactCount = problem.contacts.size();
    const std::size_t dimension = 3u * contactCount;
    if (contactCount == 0u ||
        problem.delassus.size() != dimension * dimension ||
        problem.freeContactVelocity.size() != dimension) {
        status = MetalQualityContactHostStatus::invalidProblem;
        message = "contact-space dimensions are inconsistent";
        return false;
    }
    if (contactCount > MR_METAL_QUALITY_MAX_CONTACTS) {
        status = MetalQualityContactHostStatus::capacityOverflow;
        message = "contact count exceeds the Wave32 quality bucket";
        return false;
    }
    prepared.contacts =
        static_cast<std::uint32_t>(contactCount);
    prepared.dimension =
        static_cast<std::uint32_t>(dimension);
    prepared.matrix.assign(dimension * dimension, 0.0f);
    prepared.linear.assign(dimension, 0.0f);
    prepared.warm.assign(dimension, 0.0f);
    prepared.scales.assign(dimension, 1.0);

    for (std::size_t contact = 0u;
         contact < contactCount;
         ++contact) {
        const ContactConicBlock& block =
            problem.contacts[contact];
        if (!(block.friction > 0.0) ||
            !finite(block.friction)) {
            status =
                MetalQualityContactHostStatus::unsupportedProblem;
            message =
                "Metal quality v1 requires strictly positive friction";
            return false;
        }
        prepared.scales[3u * contact] =
            1.0 / block.friction;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const std::size_t row = 3u * contact + axis;
            if (!finite(block.targetVelocity[axis]) ||
                !finite(block.regularization[axis]) ||
                !(block.regularization[axis] > 0.0) ||
                !finite(block.warmImpulse[axis]) ||
                !finite(problem.freeContactVelocity[row])) {
                status =
                    MetalQualityContactHostStatus::nonfiniteInput;
                message =
                    "contact semantics contain non-finite values";
                return false;
            }
            const double linear =
                prepared.scales[row] *
                (
                    problem.freeContactVelocity[row] -
                    block.targetVelocity[axis]
                );
            const double warm =
                config.enableWarmStart
                ? block.warmImpulse[axis] /
                    prepared.scales[row]
                : 0.0;
            if (!finite(linear) || !finite(warm) ||
                std::abs(linear) >
                    std::numeric_limits<float>::max() ||
                std::abs(warm) >
                    std::numeric_limits<float>::max()) {
                status =
                    MetalQualityContactHostStatus::nonfiniteInput;
                message =
                    "scaled contact vector exceeds FP32";
                return false;
            }
            prepared.linear[row] =
                static_cast<float>(linear);
            prepared.warm[row] =
                static_cast<float>(warm);
        }
    }
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            const double physical =
                problem.delassus[row * dimension + column];
            const double transpose =
                problem.delassus[column * dimension + row];
            const double scale =
                1.0 + std::max(
                    std::abs(physical),
                    std::abs(transpose)
                );
            if (!finite(physical) ||
                !finite(transpose) ||
                std::abs(physical - transpose) >
                    2.0e-10 * scale) {
                status =
                    MetalQualityContactHostStatus::invalidProblem;
                message =
                    "Delassus operator is non-finite or asymmetric";
                return false;
            }
            double value =
                prepared.scales[row] *
                physical *
                prepared.scales[column];
            if (row == column) {
                value +=
                    prepared.scales[row] *
                    problem.contacts[row / 3u]
                        .regularization[row % 3u] *
                    prepared.scales[row];
            }
            if (!finite(value) ||
                std::abs(value) >
                    std::numeric_limits<float>::max()) {
                status =
                    MetalQualityContactHostStatus::nonfiniteInput;
                message =
                    "scaled contact Hessian exceeds FP32";
                return false;
            }
            prepared.matrix[row * dimension + column] =
                static_cast<float>(value);
        }
    }
    return true;
}

} // namespace

MetalQualityContactDiagnostics
solveMetalQualityContactSpace(
    const ContactSpaceConicProblem& problem,
    MetalQualityContactSolution& output,
    const MetalQualityContactSolverConfig& config
) {
    MetalQualityContactDiagnostics diagnostics;
    if (!validConfig(config)) {
        return fail(
            std::move(diagnostics),
            MetalQualityContactHostStatus::invalidConfiguration,
            "Metal quality solver configuration is invalid"
        );
    }
    Prepared prepared;
    MetalQualityContactHostStatus preparationStatus =
        MetalQualityContactHostStatus::success;
    std::string preparationMessage;
    if (!prepare(
            problem,
            config,
            prepared,
            preparationStatus,
            preparationMessage
        )) {
        diagnostics.contactCount = prepared.contacts;
        diagnostics.dimension = prepared.dimension;
        return fail(
            std::move(diagnostics),
            preparationStatus,
            std::move(preparationMessage)
        );
    }
    diagnostics.contactCount = prepared.contacts;
    diagnostics.dimension = prepared.dimension;

    @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::metalUnavailable,
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
                MetalQualityContactHostStatus::metallibUnavailable,
                "no MetalRobo metallib is available"
            );
        }
        NSString* path = [NSString
            stringWithUTF8String:metallibPath.c_str()];
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:path]
                        error:&error];
        if (library == nil) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::metallibUnavailable,
                "failed to load metallib: " +
                    describeError(error)
            );
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"mr_quality_contact_solve"];
        if (function == nil) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::pipelineFailure,
                "metallib has no quality contact kernel"
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
                MetalQualityContactHostStatus::pipelineFailure,
                "quality pipeline is unsupported: " +
                    describeError(error)
            );
        }

        MRMetalQualityDispatchGPU dispatch{};
        dispatch.abiVersion =
            MR_METAL_QUALITY_SOLVER_ABI_VERSION;
        dispatch.problemCount = 1u;
        dispatch.contactCount = prepared.contacts;
        dispatch.dimension = prepared.dimension;
        dispatch.matrixStride =
            prepared.dimension * prepared.dimension;
        dispatch.vectorStride = prepared.dimension;
        dispatch.maximumNewtonIterations =
            config.maximumNewtonIterations;
        dispatch.maximumCGIterations =
            config.maximumCGIterations;
        dispatch.maximumLineSearchIterations =
            config.maximumLineSearchIterations;
        dispatch.tolerances = {
            config.convergenceTolerance,
            config.armijoCoefficient,
            config.normalEquationRegularization,
            config.minimumCGDenominator,
        };

        NSUInteger matrixBytes = 0u;
        NSUInteger vectorBytes = 0u;
        if (!checkedByteCount(
                prepared.matrix.size(),
                sizeof(float),
                matrixBytes
            ) ||
            !checkedByteCount(
                prepared.linear.size(),
                sizeof(float),
                vectorBytes
            )) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::capacityOverflow,
                "Metal quality buffer byte count overflow"
            );
        }
        id<MTLBuffer> dispatchBuffer = [device
            newBufferWithBytes:&dispatch
                       length:sizeof(dispatch)
                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> matrixBuffer = [device
            newBufferWithBytes:prepared.matrix.data()
                       length:matrixBytes
                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> linearBuffer = [device
            newBufferWithBytes:prepared.linear.data()
                       length:vectorBytes
                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> warmBuffer = [device
            newBufferWithBytes:prepared.warm.data()
                       length:vectorBytes
                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> impulseBuffer = [device
            newBufferWithLength:vectorBytes
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> statusBuffer = [device
            newBufferWithLength:sizeof(MRMetalQualityStatusGPU)
                       options:MTLResourceStorageModeShared];
        if (dispatchBuffer == nil || matrixBuffer == nil ||
            linearBuffer == nil || warmBuffer == nil ||
            impulseBuffer == nil || statusBuffer == nil ||
            impulseBuffer.contents == nullptr ||
            statusBuffer.contents == nullptr) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::bufferFailure,
                "failed to allocate Metal quality buffers"
            );
        }
        std::memset(impulseBuffer.contents, 0, vectorBytes);
        std::memset(
            statusBuffer.contents,
            0,
            sizeof(MRMetalQualityStatusGPU)
        );

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command =
            [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (queue == nil || command == nil || encoder == nil) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::commandFailure,
                "failed to create Metal quality command objects"
            );
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:matrixBuffer offset:0u atIndex:1u];
        [encoder setBuffer:linearBuffer offset:0u atIndex:2u];
        [encoder setBuffer:warmBuffer offset:0u atIndex:3u];
        [encoder setBuffer:impulseBuffer offset:0u atIndex:4u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:5u];
        [encoder
            dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
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
        if (command.status != MTLCommandBufferStatusCompleted) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::commandFailure,
                "Metal quality command failed: " +
                    describeError(command.error)
            );
        }

        const auto* gpuStatus =
            static_cast<const MRMetalQualityStatusGPU*>(
                statusBuffer.contents
            );
        if (gpuStatus->code != MR_METAL_QUALITY_SUCCESS) {
            return fail(
                std::move(diagnostics),
                MetalQualityContactHostStatus::gpuFailure,
                "Metal quality solve failed with GPU code " +
                    std::to_string(gpuStatus->code)
            );
        }
        const auto* compact =
            static_cast<const float*>(
                impulseBuffer.contents
            );
        MetalQualityContactSolution staged;
        staged.gpuStatus = *gpuStatus;
        staged.impulses.resize(prepared.dimension);
        staged.velocity =
            problem.freeContactVelocity;
        for (std::size_t index = 0u;
             index < prepared.dimension;
             ++index) {
            const double impulse =
                prepared.scales[index] * compact[index];
            if (!finite(impulse)) {
                return fail(
                    std::move(diagnostics),
                    MetalQualityContactHostStatus::nonfiniteResult,
                    "Metal quality impulse is non-finite"
                );
            }
            staged.impulses[index] = impulse;
        }
        for (std::size_t row = 0u;
             row < prepared.dimension;
             ++row) {
            for (std::size_t column = 0u;
                 column < prepared.dimension;
                 ++column) {
                staged.velocity[row] +=
                    problem.delassus[
                        row * prepared.dimension + column
                    ] * staged.impulses[column];
            }
            if (!finite(staged.velocity[row])) {
                return fail(
                    std::move(diagnostics),
                    MetalQualityContactHostStatus::nonfiniteResult,
                    "Metal quality contact velocity is non-finite"
                );
            }
        }
        output = std::move(staged);
        diagnostics.published = true;
        return diagnostics;
    }
}

const char* metalQualityContactHostStatusName(
    const MetalQualityContactHostStatus status
) noexcept {
    switch (status) {
    case MetalQualityContactHostStatus::success:
        return "success";
    case MetalQualityContactHostStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalQualityContactHostStatus::invalidProblem:
        return "invalid_problem";
    case MetalQualityContactHostStatus::unsupportedProblem:
        return "unsupported_problem";
    case MetalQualityContactHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalQualityContactHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalQualityContactHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalQualityContactHostStatus::metalUnavailable:
        return "metal_unavailable";
    case MetalQualityContactHostStatus::pipelineFailure:
        return "pipeline_failure";
    case MetalQualityContactHostStatus::bufferFailure:
        return "buffer_failure";
    case MetalQualityContactHostStatus::commandFailure:
        return "command_failure";
    case MetalQualityContactHostStatus::gpuFailure:
        return "gpu_failure";
    case MetalQualityContactHostStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
