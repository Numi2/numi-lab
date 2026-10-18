#pragma once

#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_stand_gpu.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace numi::human::event_rollback {

// Source-only failure decorator for testing an organ release at the
// deferred-accept boundary. It delegates every transaction phase, preserving
// the real coupling program/fingerprint, then injects one GPU status failure
// after postDynamics and before acceptedPublication can commit its candidate.
class Probe final {
public:
    Probe(metalrobo::MetalNumanXTransactionProgram delegate,
          const std::uint32_t releaseStep,
          const std::span<const std::uint32_t> dofPatchIndices,
          const std::span<const MRDofPropertiesGPU> dofPatchRecords)
        : delegate_(delegate), releaseStep_(releaseStep),
          dofPatchIndices_(dofPatchIndices.begin(), dofPatchIndices.end()),
          dofPatchRecords_(dofPatchRecords.begin(), dofPatchRecords.end()) {
        if (!delegate_.valid() || releaseStep_ == 0u ||
            delegate_.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
            !delegate_.deferAcceptedPublication ||
            !delegate_.writeBodyProperties ||
            !delegate_.writeDofProperties ||
            dofPatchIndices_.empty() ||
            dofPatchIndices_.size() != dofPatchRecords_.size()) {
            throw std::runtime_error(
                "organ event rollback probe requires a valid deferred-publication delegate and release step");
        }
        for (std::size_t i = 0; i < dofPatchIndices_.size(); ++i) {
            for (std::size_t j = i + 1; j < dofPatchIndices_.size(); ++j) {
                if (dofPatchIndices_[i] == dofPatchIndices_[j])
                    throw std::runtime_error("DoF patch audit indices must be unique");
            }
        }
        static_assert(MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED == 9u,
                      "rollback shader status code must match the ABI enum");
    }

    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    [[nodiscard]] metalrobo::MetalNumanXTransactionProgram program() noexcept {
        auto result = delegate_;
        result.context = this;
        result.encode = &Probe::encodeThunk;
        result.abort = &Probe::abortThunk;
        result.fingerprint = delegate_.fingerprint;
        result.deferAcceptedPublication = delegate_.deferAcceptedPublication;
        return result;
    }

    [[nodiscard]] static Probe* fromProgram(
        const metalrobo::MetalNumanXTransactionProgram* program) noexcept {
        return program != nullptr && program->context != nullptr &&
                program->encode == &Probe::encodeThunk &&
                program->abort == &Probe::abortThunk
            ? static_cast<Probe*>(program->context) : nullptr;
    }

    [[nodiscard]] std::uint32_t releaseStep() const noexcept {
        return releaseStep_;
    }
    [[nodiscard]] std::uint32_t injectedCount() const noexcept {
        return injectedCount_;
    }
    [[nodiscard]] std::uint32_t releaseAcceptedPublicationCount() const noexcept {
        return releaseAcceptedPublicationCount_;
    }
    [[nodiscard]] void* delegateContext() const noexcept {
        return delegate_.context;
    }
    [[nodiscard]] std::uint32_t bodyPropertiesSnapshotCount() const noexcept {
        return bodyPropertiesSnapshotCount_;
    }
    [[nodiscard]] bool bodyPropertiesBytesUnchanged() const noexcept {
        return bodyPropertiesSnapshotCount_ == 2u &&
            beforeBodyProperties_ != nil && retryBodyProperties_ != nil &&
            beforeBodyProperties_.storageMode == MTLStorageModeShared &&
            retryBodyProperties_.storageMode == MTLStorageModeShared &&
            beforeBodyProperties_.contents != nullptr &&
            retryBodyProperties_.contents != nullptr &&
            beforeBodyProperties_.length != 0u &&
            beforeBodyProperties_.length == retryBodyProperties_.length &&
            std::memcmp(beforeBodyProperties_.contents,
                        retryBodyProperties_.contents,
                        beforeBodyProperties_.length) == 0;
    }
    [[nodiscard]] std::uint32_t dofPropertiesSnapshotCount() const noexcept {
        return dofPropertiesSnapshotCount_;
    }
    [[nodiscard]] bool dofPropertiesBytesUnchanged() const noexcept {
        return dofPropertiesSnapshotCount_ == 2u &&
            beforeDofProperties_ != nil && retryDofProperties_ != nil &&
            beforeDofProperties_.storageMode == MTLStorageModeShared &&
            retryDofProperties_.storageMode == MTLStorageModeShared &&
            beforeDofProperties_.contents != nullptr &&
            retryDofProperties_.contents != nullptr &&
            beforeDofProperties_.length != 0u &&
            beforeDofProperties_.length == retryDofProperties_.length &&
            std::memcmp(beforeDofProperties_.contents,
                        retryDofProperties_.contents,
                        beforeDofProperties_.length) == 0;
    }
    [[nodiscard]] std::uint32_t postMutationDofSnapshotCount() const noexcept {
        return postMutationDofSnapshotCount_;
    }
    [[nodiscard]] std::uint32_t dofPatchRecordCount() const noexcept {
        return static_cast<std::uint32_t>(dofPatchIndices_.size());
    }
    [[nodiscard]] std::uint32_t postMutationChangedRecordCount() const noexcept {
        if (beforeDofProperties_ == nil || postMutationDofProperties_ == nil ||
            beforeDofProperties_.contents == nullptr ||
            postMutationDofProperties_.contents == nullptr ||
            beforeDofProperties_.length == 0u ||
            beforeDofProperties_.length != postMutationDofProperties_.length ||
            beforeDofProperties_.length % sizeof(MRDofPropertiesGPU) != 0u)
            return 0u;
        const auto count = beforeDofProperties_.length / sizeof(MRDofPropertiesGPU);
        const auto* before = static_cast<const MRDofPropertiesGPU*>(
            beforeDofProperties_.contents);
        const auto* post = static_cast<const MRDofPropertiesGPU*>(
            postMutationDofProperties_.contents);
        std::uint32_t changed = 0u;
        for (NSUInteger i = 0u; i < count; ++i)
            if (std::memcmp(&before[i], &post[i], sizeof(MRDofPropertiesGPU)) != 0)
                ++changed;
        return changed;
    }
    [[nodiscard]] bool postMutationDofPropertiesMatchExpected() const noexcept {
        if (postMutationDofSnapshotCount_ != 1u ||
            beforeDofProperties_ == nil || postMutationDofProperties_ == nil ||
            beforeDofProperties_.storageMode != MTLStorageModeShared ||
            postMutationDofProperties_.storageMode != MTLStorageModeShared ||
            beforeDofProperties_.contents == nullptr ||
            postMutationDofProperties_.contents == nullptr ||
            beforeDofProperties_.length == 0u ||
            beforeDofProperties_.length != postMutationDofProperties_.length ||
            beforeDofProperties_.length % sizeof(MRDofPropertiesGPU) != 0u)
            return false;
        const auto count = beforeDofProperties_.length / sizeof(MRDofPropertiesGPU);
        const auto* before = static_cast<const MRDofPropertiesGPU*>(
            beforeDofProperties_.contents);
        const auto* post = static_cast<const MRDofPropertiesGPU*>(
            postMutationDofProperties_.contents);
        for (const auto index : dofPatchIndices_)
            if (index >= count) return false;
        std::uint32_t changed = 0u;
        for (NSUInteger index = 0u; index < count; ++index) {
            MRDofPropertiesGPU expected = before[index];
            for (std::size_t patch = 0; patch < dofPatchIndices_.size(); ++patch) {
                if (dofPatchIndices_[patch] != index) continue;
                expected.flags = dofPatchRecords_[patch].flags;
                expected.limits = dofPatchRecords_[patch].limits;
                expected.drive.y = dofPatchRecords_[patch].drive.y;
                expected.drive.z = dofPatchRecords_[patch].drive.z;
                break;
            }
            if (std::memcmp(&post[index], &expected, sizeof(expected)) != 0)
                return false;
            if (std::memcmp(&post[index], &before[index],
                            sizeof(MRDofPropertiesGPU)) != 0)
                ++changed;
        }
        return changed != 0u;
    }

private:
    bool captureBodyPropertiesAtPreDynamics(
        const metalrobo::MetalNumanXTransactionPass& pass,
        const bool retry) noexcept {
        if (pass.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
            pass.structSize != sizeof(pass) ||
            pass.phase != metalrobo::MetalNumanXTransactionPhase::preDynamics ||
            pass.stepIndex != releaseStep_ ||
            pass.environmentCount != 1u ||
            (pass.accessFlags & metalrobo::MetalNumanXTransactionWriteBodyProperties) == 0u ||
            pass.bodyProperties == nullptr || pass.bodyPropertyCount == 0u ||
            pass.bodyPropertyCount > std::numeric_limits<NSUInteger>::max() /
                sizeof(MRBodyPropertiesGPU) || pass.commandBuffer == nullptr)
            return false;
        @autoreleasepool {
            __unsafe_unretained id<MTLCommandBuffer> command =
                (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
            __unsafe_unretained id<MTLBuffer> source =
                (__bridge id<MTLBuffer>)pass.bodyProperties;
            if (command == nil || source == nil ||
                source.device.registryID != command.device.registryID ||
                source.length < pass.bodyPropertyCount * sizeof(MRBodyPropertiesGPU))
                return false;
            const NSUInteger byteCount = static_cast<NSUInteger>(
                pass.bodyPropertyCount * sizeof(MRBodyPropertiesGPU));
            id<MTLBuffer> destination = retry
                ? retryBodyProperties_ : beforeBodyProperties_;
            if (destination != nil) return false;
            destination = [command.device newBufferWithLength:byteCount
                options:MTLResourceStorageModeShared];
            if (destination == nil || destination.storageMode != MTLStorageModeShared)
                return false;
            if (retry)
                retryBodyProperties_ = destination;
            else
                beforeBodyProperties_ = destination;
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (blit == nil) return false;
            [blit copyFromBuffer:source sourceOffset:0u toBuffer:destination
                destinationOffset:0u size:byteCount];
            [blit endEncoding];
            if (bodyPropertiesSnapshotCount_ == 0u && !retry)
                bodyPropertiesSnapshotCount_ = 1u;
            else if (bodyPropertiesSnapshotCount_ == 1u && retry)
                bodyPropertiesSnapshotCount_ = 2u;
            else
                return false;
            return true;
        }
    }

    bool captureDofPropertiesAtPreDynamics(
        const metalrobo::MetalNumanXTransactionPass& pass,
        const bool retry) noexcept {
        if (pass.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
            pass.structSize != sizeof(pass) ||
            pass.phase != metalrobo::MetalNumanXTransactionPhase::preDynamics ||
            pass.stepIndex != releaseStep_ ||
            pass.environmentCount != 1u ||
            (pass.accessFlags & metalrobo::MetalNumanXTransactionWriteDofProperties) == 0u ||
            pass.dofProperties == nullptr || pass.dofPropertiesCheckpoint == nullptr ||
            pass.dofPropertyCount == 0u ||
            pass.dofPropertyCount > std::numeric_limits<NSUInteger>::max() /
                sizeof(MRDofPropertiesGPU) || pass.commandBuffer == nullptr)
            return false;
        @autoreleasepool {
            __unsafe_unretained id<MTLCommandBuffer> command =
                (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
            __unsafe_unretained id<MTLBuffer> source =
                (__bridge id<MTLBuffer>)pass.dofProperties;
            if (command == nil || source == nil ||
                source.device.registryID != command.device.registryID ||
                source.length < pass.dofPropertyCount * sizeof(MRDofPropertiesGPU))
                return false;
            const NSUInteger byteCount = static_cast<NSUInteger>(
                pass.dofPropertyCount * sizeof(MRDofPropertiesGPU));
            id<MTLBuffer> destination = retry
                ? retryDofProperties_ : beforeDofProperties_;
            if (destination != nil) return false;
            destination = [command.device newBufferWithLength:byteCount
                options:MTLResourceStorageModeShared];
            if (destination == nil || destination.storageMode != MTLStorageModeShared)
                return false;
            if (retry)
                retryDofProperties_ = destination;
            else
                beforeDofProperties_ = destination;
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (blit == nil) return false;
            [blit copyFromBuffer:source sourceOffset:0u toBuffer:destination
                destinationOffset:0u size:byteCount];
            [blit endEncoding];
            if (dofPropertiesSnapshotCount_ == 0u && !retry)
                dofPropertiesSnapshotCount_ = 1u;
            else if (dofPropertiesSnapshotCount_ == 1u && retry)
                dofPropertiesSnapshotCount_ = 2u;
            else
                return false;
            return true;
        }
    }

    bool captureDofPropertiesAfterDelegatePreDynamics(
        const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        if (pass.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
            pass.structSize != sizeof(pass) ||
            pass.phase != metalrobo::MetalNumanXTransactionPhase::preDynamics ||
            pass.stepIndex != releaseStep_ ||
            pass.environmentCount != 1u ||
            (pass.accessFlags & metalrobo::MetalNumanXTransactionWriteDofProperties) == 0u ||
            pass.dofProperties == nullptr || pass.dofPropertyCount == 0u ||
            pass.dofPropertyCount > std::numeric_limits<NSUInteger>::max() /
                sizeof(MRDofPropertiesGPU) || pass.commandBuffer == nullptr ||
            postMutationDofSnapshotCount_ != 0u ||
            postMutationDofProperties_ != nil)
            return false;
        @autoreleasepool {
            __unsafe_unretained id<MTLCommandBuffer> command =
                (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
            __unsafe_unretained id<MTLBuffer> source =
                (__bridge id<MTLBuffer>)pass.dofProperties;
            if (command == nil || source == nil ||
                source.device.registryID != command.device.registryID ||
                source.length < pass.dofPropertyCount * sizeof(MRDofPropertiesGPU))
                return false;
            const NSUInteger byteCount = static_cast<NSUInteger>(
                pass.dofPropertyCount * sizeof(MRDofPropertiesGPU));
            postMutationDofProperties_ = [command.device
                newBufferWithLength:byteCount options:MTLResourceStorageModeShared];
            if (postMutationDofProperties_ == nil ||
                postMutationDofProperties_.storageMode != MTLStorageModeShared)
                return false;
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (blit == nil) return false;
            [blit copyFromBuffer:source sourceOffset:0u
                toBuffer:postMutationDofProperties_ destinationOffset:0u
                size:byteCount];
            [blit endEncoding];
            postMutationDofSnapshotCount_ = 1u;
            return true;
        }
    }

    static bool encodeThunk(
        void* context,
        const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        return context != nullptr &&
            static_cast<Probe*>(context)->encode(pass);
    }

    static void abortThunk(void* context, void* commandBuffer) noexcept {
        if (context == nullptr) return;
        auto& self = *static_cast<Probe*>(context);
        if (self.delegate_.abort != nullptr)
            self.delegate_.abort(self.delegate_.context, commandBuffer);
    }

    bool encode(const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        @autoreleasepool {
            try {
                if (pass.phase ==
                        metalrobo::MetalNumanXTransactionPhase::preDynamics &&
                    pass.stepIndex == releaseStep_) {
                    const bool retry = injectedCount_ == 1u;
                    if (!captureBodyPropertiesAtPreDynamics(pass, retry) ||
                        !captureDofPropertiesAtPreDynamics(pass, retry))
                        return false;
                }
                if (delegate_.encode == nullptr ||
                    pass.programFingerprint != delegate_.fingerprint ||
                    !delegate_.encode(delegate_.context, pass))
                    return false;

                if (pass.phase ==
                        metalrobo::MetalNumanXTransactionPhase::preDynamics &&
                    pass.stepIndex == releaseStep_ && injectedCount_ == 0u &&
                    !captureDofPropertiesAfterDelegatePreDynamics(pass))
                    return false;

                if (pass.phase ==
                    metalrobo::MetalNumanXTransactionPhase::acceptedPublication) {
                    constexpr std::uint32_t required =
                        metalrobo::MetalNumanXTransactionReadBorrowedState |
                        metalrobo::MetalNumanXTransactionWriteAcceptedPublication;
                    constexpr std::uint32_t forbidden =
                        metalrobo::MetalNumanXTransactionWriteStandFailure |
                        metalrobo::MetalNumanXTransactionWriteBodyProperties |
                        metalrobo::MetalNumanXTransactionWriteGeneralizedForce;
                    if ((pass.accessFlags & required) != required ||
                        (pass.accessFlags & forbidden) != 0u ||
                        pass.environmentCount != 1u ||
                        pass.standStatusElementCount != 1u ||
                        pass.standStatusStride != 1u ||
                        pass.standStatuses == nullptr)
                        return false;
                    if (pass.stepIndex == releaseStep_)
                        ++releaseAcceptedPublicationCount_;
                    return true;
                }

                if (pass.phase !=
                        metalrobo::MetalNumanXTransactionPhase::postDynamics ||
                    pass.stepIndex != releaseStep_ || injectedCount_ != 0u)
                    return true;

                constexpr std::uint32_t required =
                    metalrobo::MetalNumanXTransactionReadBorrowedState |
                    metalrobo::MetalNumanXTransactionWriteStandFailure;
                if (pass.abiVersion !=
                        metalrobo::kMetalNumanXTransactionABIVersion ||
                    pass.structSize != sizeof(pass) ||
                    (pass.accessFlags & required) != required ||
                    pass.environmentCount != 1u ||
                    pass.standStatusElementCount != 1u ||
                    pass.standStatusStride != 1u ||
                    pass.standStatuses == nullptr ||
                    pass.commandBuffer == nullptr)
                    return false;

                __unsafe_unretained id<MTLCommandBuffer> command =
                    (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
                __unsafe_unretained id<MTLBuffer> status =
                    (__bridge id<MTLBuffer>)pass.standStatuses;
                if (command == nil || status == nil ||
                    status.device.registryID != command.device.registryID ||
                    status.length < sizeof(MRNumiHumanStandStatusGPU))
                    return false;

                if (failurePipeline_ == nil) {
                    NSError* error = nil;
                    NSString* source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void numi_organ_event_rollback_fail(
    device uint* statusWords [[buffer(0)]],
    constant uint& acceptedStepCount [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    // Stand status ABI: code at word 0 and completedSteps at word 2.
    // Reject only a currently successful candidate for this exact step.
    if (tid == 0u && statusWords[0] == 0u &&
        statusWords[2] == acceptedStepCount + 1u) {
        statusWords[0] = 9u;
        statusWords[2] = acceptedStepCount;
    }
}
)METAL";
                    MTLCompileOptions* options = [MTLCompileOptions new];
                    options.mathMode = MTLMathModeSafe;
                    options.mathFloatingPointFunctions =
                        MTLMathFloatingPointFunctionsPrecise;
                    id<MTLLibrary> library = [command.device
                        newLibraryWithSource:source options:options error:&error];
                    if (library == nil) return false;
                    id<MTLFunction> function = [library
                        newFunctionWithName:@"numi_organ_event_rollback_fail"];
                    if (function == nil) return false;
                    failurePipeline_ = [command.device
                        newComputePipelineStateWithFunction:function error:&error];
                    if (failurePipeline_ == nil) return false;
                }

                const std::uint32_t acceptedStepCount = pass.stepIndex;
                id<MTLComputeCommandEncoder> encoder =
                    [command computeCommandEncoder];
                if (encoder == nil) return false;
                [encoder setComputePipelineState:failurePipeline_];
                [encoder setBuffer:status offset:0u atIndex:0u];
                [encoder setBytes:&acceptedStepCount
                           length:sizeof(acceptedStepCount) atIndex:1u];
                [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
                [encoder endEncoding];
                ++injectedCount_;
                return true;
            } catch (...) {
                return false;
            }
        }
    }

    metalrobo::MetalNumanXTransactionProgram delegate_{};
    const std::uint32_t releaseStep_ = 0u;
    std::uint32_t injectedCount_ = 0u;
    std::uint32_t releaseAcceptedPublicationCount_ = 0u;
    std::uint32_t bodyPropertiesSnapshotCount_ = 0u;
    std::uint32_t dofPropertiesSnapshotCount_ = 0u;
    std::uint32_t postMutationDofSnapshotCount_ = 0u;
    std::vector<std::uint32_t> dofPatchIndices_;
    std::vector<MRDofPropertiesGPU> dofPatchRecords_;
    __strong id<MTLBuffer> postMutationDofProperties_ = nil;
    __strong id<MTLBuffer> beforeBodyProperties_ = nil;
    __strong id<MTLBuffer> retryBodyProperties_ = nil;
    __strong id<MTLBuffer> beforeDofProperties_ = nil;
    __strong id<MTLBuffer> retryDofProperties_ = nil;
    __strong id<MTLComputePipelineState> failurePipeline_ = nil;
};

} // namespace numi::human::event_rollback
