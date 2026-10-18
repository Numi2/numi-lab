#pragma once

#import <Metal/Metal.h>
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_stand_gpu.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace numi::human::shared_tissue_rollback {

// Probe-only shared-owner failure decorator. It forwards every transaction
// phase, then changes only the stand status after postDynamics so the real
// deferred Matter/adapter finalizers and the articulated owner see one late
// external rejection on the same command buffer.
class Probe final {
public:
    Probe(metalrobo::MetalNumanXTransactionProgram delegate,
          std::uint32_t rejectStep)
        : delegate_(delegate), rejectStep_(rejectStep) {
        if (!delegate_.valid() || delegate_.abiVersion !=
                metalrobo::kMetalNumanXTransactionABIVersion ||
            !delegate_.deferAcceptedPublication ||
            !delegate_.writeBodyProperties || delegate_.writeDofProperties ||
            rejectStep_ == 0u) {
            throw std::runtime_error(
                "shared tissue rollback probe requires deferred publication, body-property ownership, no DoF mutation, and a post-prefix step");
        }
        static_assert(MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED == 9u,
                      "rollback status must match the stand ABI");
    }

    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    [[nodiscard]] metalrobo::MetalNumanXTransactionProgram program() noexcept {
        auto result = delegate_;
        result.context = this;
        result.encode = &Probe::encodeThunk;
        result.abort = &Probe::abortThunk;
        result.fingerprint = delegate_.fingerprint;
        return result;
    }

    [[nodiscard]] static Probe* fromProgram(
        const metalrobo::MetalNumanXTransactionProgram* program) noexcept {
        return program != nullptr && program->context != nullptr &&
                program->encode == &Probe::encodeThunk &&
                program->abort == &Probe::abortThunk
            ? static_cast<Probe*>(program->context) : nullptr;
    }

    [[nodiscard]] void* delegateContext() const noexcept {
        return delegate_.context;
    }
    [[nodiscard]] std::uint32_t rejectStep() const noexcept { return rejectStep_; }
    [[nodiscard]] std::uint32_t injectedCount() const noexcept { return injectedCount_; }
    [[nodiscard]] std::uint32_t acceptedPublicationCount() const noexcept {
        return acceptedPublicationCount_;
    }
    [[nodiscard]] std::uint32_t bodySnapshotCount() const noexcept {
        return bodySnapshotCount_;
    }
    [[nodiscard]] bool bodyPropertiesRestoredForRetry() const noexcept {
        return bodySnapshotCount_ == 2u && buffersEqual(bodyBefore_, bodyRetry_);
    }
    [[nodiscard]] std::uint32_t dofSnapshotCount() const noexcept {
        return dofSnapshotCount_;
    }
    [[nodiscard]] bool dofPropertiesRestoredForRetry() const noexcept {
        if (dofSnapshotCount_ == 0u)
            return !delegate_.writeDofProperties && dofBefore_ == nil && dofRetry_ == nil;
        return dofSnapshotCount_ == 2u && buffersEqual(dofBefore_, dofRetry_);
    }

private:
    static bool buffersEqual(id<MTLBuffer> first, id<MTLBuffer> second) noexcept {
        return first != nil && second != nil &&
            first.storageMode == MTLStorageModeShared &&
            second.storageMode == MTLStorageModeShared &&
            first.contents != nullptr && second.contents != nullptr &&
            first.length != 0u && first.length == second.length &&
            std::memcmp(first.contents, second.contents, first.length) == 0;
    }

    static bool captureBuffer(const metalrobo::MetalNumanXTransactionPass& pass,
                              const void* rawSource, std::size_t count,
                              std::size_t elementSize, bool retry,
                              __strong id<MTLBuffer>& before,
                              __strong id<MTLBuffer>& after) noexcept {
        if (rawSource == nullptr || count == 0u || elementSize == 0u ||
            count > std::numeric_limits<NSUInteger>::max() / elementSize ||
            pass.commandBuffer == nullptr)
            return false;
        @autoreleasepool {
            __unsafe_unretained id<MTLCommandBuffer> command =
                (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
            __unsafe_unretained id<MTLBuffer> source =
                (__bridge id<MTLBuffer>)rawSource;
            const NSUInteger bytes = static_cast<NSUInteger>(count * elementSize);
            if (command == nil || source == nil ||
                source.device.registryID != command.device.registryID ||
                source.storageMode != MTLStorageModeShared || source.length < bytes)
                return false;
            __strong id<MTLBuffer>& destination = retry ? after : before;
            if (destination != nil) return false;
            destination = [command.device newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
            if (destination == nil || destination.storageMode != MTLStorageModeShared)
                return false;
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (blit == nil) return false;
            [blit copyFromBuffer:source sourceOffset:0u toBuffer:destination
                destinationOffset:0u size:bytes];
            [blit endEncoding];
            return true;
        }
    }

    bool captureProperties(const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        if (pass.phase != metalrobo::MetalNumanXTransactionPhase::preDynamics ||
            pass.stepIndex != rejectStep_ || pass.environmentCount != 1u ||
            (pass.accessFlags & metalrobo::MetalNumanXTransactionWriteBodyProperties) == 0u ||
            pass.bodyProperties == nullptr || pass.bodyPropertyCount == 0u)
            return false;
        const bool retry = injectedCount_ == 1u;
        if (!captureBuffer(pass, pass.bodyProperties, pass.bodyPropertyCount,
                           sizeof(MRBodyPropertiesGPU), retry,
                           bodyBefore_, bodyRetry_))
            return false;
        if (retry) {
            if (bodySnapshotCount_ != 1u) return false;
            ++bodySnapshotCount_;
        } else {
            if (bodySnapshotCount_ != 0u) return false;
            ++bodySnapshotCount_;
        }
        if (pass.dofProperties == nullptr || pass.dofPropertyCount == 0u)
            return !delegate_.writeDofProperties;
        if (!captureBuffer(pass, pass.dofProperties, pass.dofPropertyCount,
                           sizeof(MRDofPropertiesGPU), retry,
                           dofBefore_, dofRetry_))
            return false;
        if (retry) {
            if (dofSnapshotCount_ != 1u) return false;
            ++dofSnapshotCount_;
        } else {
            if (dofSnapshotCount_ != 0u) return false;
            ++dofSnapshotCount_;
        }
        return true;
    }

    static bool encodeThunk(void* context,
                            const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        return context != nullptr && static_cast<Probe*>(context)->encode(pass);
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
                if (pass.phase == metalrobo::MetalNumanXTransactionPhase::preDynamics &&
                    pass.stepIndex == rejectStep_ && !captureProperties(pass))
                    return false;
                if (delegate_.encode == nullptr ||
                    pass.programFingerprint != delegate_.fingerprint ||
                    !delegate_.encode(delegate_.context, pass))
                    return false;
                if (pass.phase == metalrobo::MetalNumanXTransactionPhase::acceptedPublication &&
                    pass.stepIndex == rejectStep_) {
                    ++acceptedPublicationCount_;
                    return true;
                }
                if (pass.phase != metalrobo::MetalNumanXTransactionPhase::postDynamics ||
                    pass.stepIndex != rejectStep_ || injectedCount_ != 0u)
                    return true;
                constexpr std::uint32_t required =
                    metalrobo::MetalNumanXTransactionReadBorrowedState |
                    metalrobo::MetalNumanXTransactionWriteStandFailure;
                if (pass.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
                    pass.structSize != sizeof(pass) || (pass.accessFlags & required) != required ||
                    pass.environmentCount != 1u || pass.standStatusElementCount != 1u ||
                    pass.standStatusStride != 1u || pass.standStatuses == nullptr ||
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
kernel void numi_shared_tissue_rollback_fail(
    device uint* words [[buffer(0)]], constant uint& accepted [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid == 0u && words[0] == 0u && words[2] == accepted + 1u) {
        words[0] = 9u; words[2] = accepted;
    }
}
)METAL";
                    MTLCompileOptions* options = [MTLCompileOptions new];
                    options.mathMode = MTLMathModeSafe;
                    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
                    id<MTLLibrary> library = [command.device
                        newLibraryWithSource:source options:options error:&error];
                    if (library == nil) return false;
                    id<MTLFunction> function = [library
                        newFunctionWithName:@"numi_shared_tissue_rollback_fail"];
                    if (function == nil) return false;
                    failurePipeline_ = [command.device
                        newComputePipelineStateWithFunction:function error:&error];
                    if (failurePipeline_ == nil) return false;
                }
                const std::uint32_t accepted = pass.stepIndex;
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil) return false;
                [encoder setComputePipelineState:failurePipeline_];
                [encoder setBuffer:status offset:0u atIndex:0u];
                [encoder setBytes:&accepted length:sizeof(accepted) atIndex:1u];
                [encoder dispatchThreads:MTLSizeMake(1u,1u,1u)
                    threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
                [encoder endEncoding];
                ++injectedCount_;
                return true;
            } catch (...) { return false; }
        }
    }

    metalrobo::MetalNumanXTransactionProgram delegate_{};
    std::uint32_t rejectStep_ = 1u;
    std::uint32_t injectedCount_ = 0u;
    std::uint32_t acceptedPublicationCount_ = 0u;
    std::uint32_t bodySnapshotCount_ = 0u;
    std::uint32_t dofSnapshotCount_ = 0u;
    __strong id<MTLBuffer> bodyBefore_ = nil, bodyRetry_ = nil;
    __strong id<MTLBuffer> dofBefore_ = nil, dofRetry_ = nil;
    __strong id<MTLComputePipelineState> failurePipeline_ = nil;
};

} // namespace numi::human::shared_tissue_rollback
