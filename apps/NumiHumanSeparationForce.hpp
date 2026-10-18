#pragma once

// Objective-C++ adapter for the owning Human tendon-load callback. The supplied
// MTLLibrary must be a pinned source-built library containing
// mr_numi_human_separation_release_crossing_forces from
// NumiHumanSeparationForce.metalinc.

#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_tendon_gpu.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace metalrobo {

class NumiHumanSeparationForceAdapter final {
public:
    NumiHumanSeparationForceAdapter() = default;
    ~NumiHumanSeparationForceAdapter() = default;
    NumiHumanSeparationForceAdapter(const NumiHumanSeparationForceAdapter&) = delete;
    NumiHumanSeparationForceAdapter& operator=(
        const NumiHumanSeparationForceAdapter&) = delete;
    NumiHumanSeparationForceAdapter(NumiHumanSeparationForceAdapter&&) = delete;
    NumiHumanSeparationForceAdapter& operator=(
        NumiHumanSeparationForceAdapter&&) = delete;

    // libraryFingerprint binds this callback to the exact source-built library
    // selected by the owning Human run. The adapter retains the library and
    // resolves its function and pipeline once, never in the step loop.
    [[nodiscard]] bool initialize(
        id<MTLLibrary> pinnedMTLLibrary,
        const std::span<const std::uint32_t> crossingMuscleIndices,
        const std::uint32_t releaseStep,
        const std::uint64_t libraryFingerprint
    ) {
        if (pinnedMTLLibrary == nil || crossingMuscleIndices.empty() ||
            crossingMuscleIndices.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            libraryFingerprint == 0u) {
            return false;
        }
        for (std::size_t index = 0u;
             index < crossingMuscleIndices.size(); ++index) {
            for (std::size_t previous = 0u; previous < index; ++previous) {
                if (crossingMuscleIndices[index] ==
                    crossingMuscleIndices[previous]) {
                    return false;
                }
            }
        }

        id<MTLDevice> candidateDevice = pinnedMTLLibrary.device;
        if (candidateDevice == nil) return false;
        id<MTLFunction> function = [pinnedMTLLibrary
            newFunctionWithName:@"mr_numi_human_separation_release_crossing_forces"];
        if (function == nil) return false;
        NSError* error = nil;
        id<MTLComputePipelineState> candidatePipeline =
            [candidateDevice newComputePipelineStateWithFunction:function
                                                            error:&error];
        if (candidatePipeline == nil) return false;

        std::vector<std::uint32_t> candidateIndices(
            crossingMuscleIndices.begin(), crossingMuscleIndices.end());
        id<MTLBuffer> candidateIndexBuffer = [candidateDevice
            newBufferWithBytes:candidateIndices.data()
                        length:candidateIndices.size() * sizeof(std::uint32_t)
                       options:MTLResourceStorageModeShared];
        if (candidateIndexBuffer == nil) return false;

        std::uint64_t hash = 14695981039346656037ull;
        hash = appendFingerprint(hash, &libraryFingerprint,
                                 sizeof(libraryFingerprint));
        hash = appendFingerprint(hash, &releaseStep, sizeof(releaseStep));
        const std::uint64_t indexCount = candidateIndices.size();
        hash = appendFingerprint(hash, &indexCount, sizeof(indexCount));
        hash = appendFingerprint(hash, candidateIndices.data(),
            candidateIndices.size() * sizeof(candidateIndices.front()));
        if (hash == 0u) hash = 1u;

        device_ = candidateDevice;
        library_ = pinnedMTLLibrary;
        pipeline_ = candidatePipeline;
        crossingMuscleIndices_ = std::move(candidateIndices);
        crossingMuscleBuffer_ = candidateIndexBuffer;
        crossingMuscleCount_ = static_cast<std::uint32_t>(
            crossingMuscleIndices_.size());
        releaseStep_ = releaseStep;
        fingerprint_ = hash;
        return true;
    }

    [[nodiscard]] MetalNumiHumanTendonLoadProgram program() noexcept {
        MetalNumiHumanTendonLoadProgram result{};
        if (!initialized()) return result;
        result.context = this;
        result.encodePreDynamics = [](
            void* context, const MetalNumiHumanTendonLoadPass& pass) {
            return context != nullptr &&
                static_cast<NumiHumanSeparationForceAdapter*>(context)
                    ->encodePreDynamics(pass);
        };
        result.encodePostValidation = [](
            void* context, const MetalNumiHumanTendonLoadPass& pass) {
            return context != nullptr &&
                static_cast<NumiHumanSeparationForceAdapter*>(context)
                    ->encodePostValidation(pass);
        };
        result.abort = [](void*, void*) {};
        result.fingerprint = fingerprint_;
        return result;
    }

private:
    struct ConsumedRange {
        std::uint64_t begin = 0u;
        std::uint64_t end = 0u;
    };

    [[nodiscard]] bool initialized() const noexcept {
        return device_ != nil && library_ != nil && pipeline_ != nil &&
            crossingMuscleBuffer_ != nil && crossingMuscleCount_ != 0u &&
            fingerprint_ != 0u;
    }

    [[nodiscard]] bool encodePreDynamics(
        const MetalNumiHumanTendonLoadPass& pass
    ) noexcept {
        @autoreleasepool {
            if (!initialized() || pass.commandBuffer == nullptr ||
                pass.bindings == nullptr || pass.transfers == nullptr ||
                pass.generalizedCorrections == nullptr ||
                pass.generalizedForces == nullptr ||
                pass.environmentCount == 0u || pass.endpointCount == 0u ||
                pass.muscleCount == 0u || pass.dofCount == 0u ||
                pass.generalizedForceStride != pass.dofCount) {
                return false;
            }

            std::uint64_t muscleRows = 0u;
            std::uint64_t muscleRowElements = 0u;
            std::uint64_t reducedPrefix = 0u;
            std::uint64_t reducedEnd = 0u;
            std::uint64_t endpointElements = 0u;
            std::uint64_t correctionElements = 0u;
            std::uint64_t forceWork = 0u;
            std::uint64_t endpointWork = 0u;
            if (!checkedProduct(pass.environmentCount, pass.muscleCount,
                                muscleRows) ||
                !checkedProduct(muscleRows, pass.dofCount,
                                muscleRowElements) ||
                !checkedProduct(pass.environmentCount, pass.endpointCount,
                                endpointElements) ||
                !checkedProduct(endpointElements, pass.dofCount,
                                correctionElements) ||
                !checkedProduct(pass.environmentCount, pass.dofCount,
                                forceWork) ||
                !checkedProduct(pass.environmentCount, pass.endpointCount,
                                endpointWork) ||
                !checkedProduct(pass.environmentCount - 1u,
                                pass.generalizedForceStride, reducedPrefix) ||
                !checkedSum(reducedPrefix, pass.generalizedForceOffset,
                            reducedEnd) ||
                !checkedSum(reducedEnd, pass.dofCount, reducedEnd) ||
                pass.generalizedForceOffset != muscleRowElements ||
                std::max({muscleRowElements, reducedEnd, endpointElements,
                          correctionElements, forceWork, endpointWork}) >
                    std::numeric_limits<std::uint32_t>::max() ||
                std::max(forceWork, endpointWork) == 0u ||
                !std::all_of(crossingMuscleIndices_.begin(),
                             crossingMuscleIndices_.end(),
                    [&](const std::uint32_t index) {
                        return index < pass.muscleCount;
                    })) {
                return false;
            }

            id<MTLCommandBuffer> command =
                (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
            id<MTLBuffer> bindings = (__bridge id<MTLBuffer>)pass.bindings;
            id<MTLBuffer> transfers = (__bridge id<MTLBuffer>)pass.transfers;
            id<MTLBuffer> corrections =
                (__bridge id<MTLBuffer>)pass.generalizedCorrections;
            id<MTLBuffer> generalized =
                (__bridge id<MTLBuffer>)pass.generalizedForces;
            if (command == nil || bindings == nil || transfers == nil ||
                corrections == nil || generalized == nil ||
                command.commandQueue == nil ||
                command.commandQueue.device.registryID != device_.registryID ||
                bindings.device.registryID != device_.registryID ||
                transfers.device.registryID != device_.registryID ||
                corrections.device.registryID != device_.registryID ||
                generalized.device.registryID != device_.registryID ||
                crossingMuscleBuffer_.device.registryID != device_.registryID) {
                return false;
            }

            ConsumedRange bindingRange{};
            ConsumedRange transferRange{};
            ConsumedRange correctionRange{};
            ConsumedRange forceRange{};
            ConsumedRange crossingRange{};
            if (!consumedRange(bindings, pass.endpointCount,
                               sizeof(MRNumiHumanTendonBindingGPU), bindingRange) ||
                !consumedRange(transfers, endpointElements,
                               sizeof(MRNumiHumanTendonTransferResultGPU),
                               transferRange) ||
                !consumedRange(corrections, correctionElements, sizeof(float),
                               correctionRange) ||
                !consumedRange(generalized,
                               std::max(muscleRowElements, reducedEnd),
                               sizeof(float), forceRange) ||
                !consumedRange(crossingMuscleBuffer_, crossingMuscleCount_,
                               sizeof(std::uint32_t), crossingRange) ||
                overlaps(bindingRange, transferRange) ||
                overlaps(bindingRange, correctionRange) ||
                overlaps(bindingRange, forceRange) ||
                overlaps(bindingRange, crossingRange) ||
                overlaps(transferRange, correctionRange) ||
                overlaps(transferRange, forceRange) ||
                overlaps(transferRange, crossingRange) ||
                overlaps(correctionRange, forceRange) ||
                overlaps(correctionRange, crossingRange) ||
                overlaps(forceRange, crossingRange)) {
                return false;
            }

            // This adapter carries no event state. A rejected step retries
            // from the owner's regenerated arena and receives the same edit.
            if (pass.stepIndex < releaseStep_) return true;

            const std::array<std::uint32_t, 4u> shape{{
                pass.environmentCount, pass.muscleCount,
                pass.endpointCount, pass.dofCount,
            }};
            const std::array<std::uint32_t, 4u> forceLayout{{
                pass.generalizedForceOffset, pass.generalizedForceStride,
                pass.stepIndex, releaseStep_,
            }};
            const std::uint32_t crossingCount = crossingMuscleCount_;
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (encoder == nil) return false;
            [encoder setComputePipelineState:pipeline_];
            [encoder setBytes:shape.data() length:sizeof(shape) atIndex:0u];
            [encoder setBytes:forceLayout.data()
                       length:sizeof(forceLayout) atIndex:1u];
            [encoder setBytes:&crossingCount
                       length:sizeof(crossingCount) atIndex:2u];
            [encoder setBuffer:crossingMuscleBuffer_ offset:0u atIndex:3u];
            [encoder setBuffer:bindings offset:0u atIndex:4u];
            [encoder setBuffer:transfers offset:0u atIndex:5u];
            [encoder setBuffer:corrections offset:0u atIndex:6u];
            [encoder setBuffer:generalized offset:0u atIndex:7u];
            const std::uint64_t workCount = std::max(forceWork, endpointWork);
            const NSUInteger width = std::max<NSUInteger>(1u,
                std::min<NSUInteger>(128u,
                    pipeline_.maxTotalThreadsPerThreadgroup));
            const NSUInteger groups = static_cast<NSUInteger>(
                (workCount + width - 1u) / width);
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
            [encoder endEncoding];
            return true;
        }
    }

    [[nodiscard]] bool encodePostValidation(
        const MetalNumiHumanTendonLoadPass& pass
    ) const noexcept {
        // Force release changes only the borrowed pre-dynamics force arena.
        // There is no post-validation state to publish or roll back.
        return initialized() && pass.commandBuffer != nullptr;
    }

    [[nodiscard]] static bool checkedProduct(
        const std::uint64_t a, const std::uint64_t b,
        std::uint64_t& result
    ) noexcept {
        if (b != 0u && a > std::numeric_limits<std::uint64_t>::max() / b)
            return false;
        result = a * b;
        return true;
    }

    [[nodiscard]] static bool checkedSum(
        const std::uint64_t a, const std::uint64_t b,
        std::uint64_t& result
    ) noexcept {
        if (a > std::numeric_limits<std::uint64_t>::max() - b) return false;
        result = a + b;
        return true;
    }

    [[nodiscard]] static bool consumedRange(
        id<MTLBuffer> buffer, const std::uint64_t elements,
        const std::uint64_t elementBytes, ConsumedRange& range
    ) noexcept {
        std::uint64_t bytes = 0u;
        if (buffer == nil || elements == 0u ||
            !checkedProduct(elements, elementBytes, bytes) ||
            bytes > buffer.length || buffer.gpuAddress == 0u) {
            return false;
        }
        range.begin = buffer.gpuAddress;
        return checkedSum(range.begin, bytes, range.end);
    }

    [[nodiscard]] static bool overlaps(
        const ConsumedRange& a, const ConsumedRange& b
    ) noexcept {
        return a.begin < b.end && b.begin < a.end;
    }

    [[nodiscard]] static std::uint64_t appendFingerprint(
        std::uint64_t hash, const void* bytes, const std::size_t count
    ) noexcept {
        constexpr std::uint64_t prime = 1099511628211ull;
        const auto* cursor = static_cast<const std::uint8_t*>(bytes);
        for (std::size_t index = 0u; index < count; ++index) {
            hash ^= cursor[index];
            hash *= prime;
        }
        return hash;
    }

    __strong id<MTLDevice> device_ = nil;
    __strong id<MTLLibrary> library_ = nil;
    __strong id<MTLComputePipelineState> pipeline_ = nil;
    __strong id<MTLBuffer> crossingMuscleBuffer_ = nil;
    std::vector<std::uint32_t> crossingMuscleIndices_;
    std::uint32_t crossingMuscleCount_ = 0u;
    std::uint32_t releaseStep_ = 0u;
    std::uint64_t fingerprint_ = 0u;
};

} // namespace metalrobo
