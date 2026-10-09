#pragma once

#include <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_motion_observer.hpp"
#include "metalrobo/numi_human_support_geometry_gpu.h"
#include "metalrobo/numi_human_resting_bed_gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

enum class NumiHumanSkinInfluenceLayout : std::uint32_t {
    vertexMajor = 0u,
    tile32 = 1u,
};

struct NumiHumanSkinInfluencePipelineConfig {
    bool validatedStaticSkinInfluences = false;
    bool specializeSkinInfluenceLayout = false;
    NumiHumanSkinInfluenceLayout influenceLayout =
        NumiHumanSkinInfluenceLayout::vertexMajor;
    bool skipNonAncestorSupportBodyDofs = false;
    bool contouredBed = false;
};

inline id<MTLFunction> numiHumanRestingMakeSkinInfluenceFunction(
    id<MTLLibrary> library, NSString* name,
    const NumiHumanSkinInfluencePipelineConfig config,
    NSError** error
) {
    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    // Optional Metal arguments require a specialized function in both modes.
    // Explicit false removes the bed buffers from the legacy pipeline.
    const bool contouredBed = config.contouredBed;
    [constants setConstantValue:&contouredBed type:MTLDataTypeBool atIndex:47u];
    if (config.validatedStaticSkinInfluences) {
        bool validated = true;
        [constants setConstantValue:&validated
                              type:MTLDataTypeBool atIndex:42u];
    }
    if (config.specializeSkinInfluenceLayout) {
        bool tiled = config.influenceLayout ==
            NumiHumanSkinInfluenceLayout::tile32;
        [constants setConstantValue:&tiled
                              type:MTLDataTypeBool atIndex:15u];
    }
    if (config.skipNonAncestorSupportBodyDofs) {
        bool skipNonAncestorSupportBodyDofs = true;
        [constants setConstantValue:&skipNonAncestorSupportBodyDofs
                              type:MTLDataTypeBool atIndex:43u];
    }
    return [library newFunctionWithName:name
                         constantValues:constants
                                  error:error];
}

// Encoder-only client for the operator's accepted-pose support-geometry lease.
// The caller supplies the same preuploaded LBS buffers used by the viewer; this
// class uploads only the derived region assignment and its compact 32-row map.
class NumiHumanRestingSupportGeometry final {
    __strong id<MTLDevice> device_ = nil;
    __strong id<MTLComputePipelineState> positionsPipeline_ = nil;
    __strong id<MTLComputePipelineState> orientationPipeline_ = nil;
    __strong id<MTLComputePipelineState> selectPipeline_ = nil;
    __strong id<MTLComputePipelineState> publishPipeline_ = nil;
    __strong id<MTLComputePipelineState> debugPipeline_ = nil;
    MRHumanRestingBedGPU bed_{};
    __strong id<MTLBuffer> bedHeights_ = nil;
    __strong id<MTLBuffer> vertexMap_ = nil;
    __strong id<MTLBuffer> influences_ = nil;
    __strong id<MTLBuffer> regionForVertex_ = nil;
    __strong id<MTLBuffer> regions_ = nil;
    __strong id<MTLBuffer> worldZKeys_ = nil;
    __strong id<MTLBuffer> minimumHeightKeys_ = nil;
    __strong id<MTLBuffer> selectedVertices_ = nil;
    __strong id<MTLBuffer> invalidRegionFlags_ = nil;
    __strong id<MTLBuffer> normalizedOrientations_ = nil;
    __strong id<MTLBuffer> rotationZBasis_ = nil;
    __strong id<MTLBuffer> bodyDofAncestry_ = nil;
    __strong id<MTLBuffer> debugOutput_ = nil;
    std::uint32_t vertexCount_ = 0u;
    std::uint32_t regionCount_ = 0u;
    std::uint32_t environmentCount_ = 0u;
    std::uint32_t influenceCount_ = 0u;
    NumiHumanSkinInfluenceLayout influenceLayout_ =
        NumiHumanSkinInfluenceLayout::vertexMajor;
    std::uint32_t validatedFirstBody_ = 0u;
    std::uint32_t validatedBodyCount_ = 0u;
    std::uint32_t validatedDofCount_ = 0u;
    bool skipNonAncestorSupportBodyDofs_ = false;
    std::uint32_t orientationStride_ = 0u;
    std::uint64_t orientationElementCount_ = 0u;
    NSUInteger vertexMapOffsetBytes_ = 0u;
    std::uint64_t fingerprint_ = 0u;
    NSUInteger threadsPerGroup_ = 0u;
    NSUInteger debugThreadsPerGroup_ = 0u;
    std::vector<MRHumanRestingSupportRegionGPU> hostRegions_;
    bool diagnosticCapture_ = false;
    bool motionObserverRequested_ = false;
    bool motionObserverCapture_ = false;
    std::uint32_t motionObserverCadenceSteps_ = 0u;
    double motionObserverTimestepSeconds_ = 0.0;
    std::ofstream motionObserverTrace_;

    static void require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }

    static bool encodeCallback(
        void* context, const metalrobo::MetalNumiHumanSupportGeometryPass& pass
    ) {
        if (context == nullptr) return false;
        try {
            return static_cast<NumiHumanRestingSupportGeometry*>(context)
                ->encode(pass);
        } catch (...) {
            return false;
        }
    }

    static void abortCallback(void*, void*) {}

    static id<MTLComputePipelineState> makePipeline(
        id<MTLDevice> device, id<MTLLibrary> library, NSString* name,
        const bool validatedSkinInfluences = false,
        const bool specializeSkinInfluenceLayout = false,
        const NumiHumanSkinInfluenceLayout influenceLayout =
            NumiHumanSkinInfluenceLayout::vertexMajor,
        const bool skipNonAncestorSupportBodyDofs = false,
        const bool contouredBed = false
    ) {
        NSError* functionError = nil;
        const auto function = numiHumanRestingMakeSkinInfluenceFunction(
            library, name,
            {validatedSkinInfluences, specializeSkinInfluenceLayout,
             influenceLayout, skipNonAncestorSupportBodyDofs, contouredBed},
            &functionError);
        require(function != nil,
                functionError.localizedDescription.UTF8String != nullptr
                    ? functionError.localizedDescription.UTF8String
                    : "resting support Metal function specialization failed");
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil,
                error.localizedDescription.UTF8String != nullptr
                    ? error.localizedDescription.UTF8String
                    : "resting support Metal pipeline failed");
        return pipeline;
    }

    static bool sameDevice(id<MTLDevice> expected, id<MTLBuffer> buffer) {
        return buffer != nil && buffer.device != nil &&
            buffer.device.registryID == expected.registryID;
    }

    static id<MTLComputeCommandEncoder> timedSupportEncoder(
        id<MTLCommandBuffer> command, id<MTLDevice> device,
        const char* stage, std::uint32_t step
    ) {
        static const bool enabled = [] {
            const char* requested = std::getenv("NUMI_HUMAN_SUPPORT_GPU_TIMING");
            return requested != nullptr && std::strcmp(requested, "1") == 0;
        }();
        if (!enabled || (step != 0u && step != 100u && step != 1000u) ||
            ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            return [command computeCommandEncoder];
        }
        id<MTLCounterSampleBuffer> timing = nil;
        for (id<MTLCounterSet> set in device.counterSets) {
            if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            MTLCounterSampleBufferDescriptor* descriptor =
                [MTLCounterSampleBufferDescriptor new];
            descriptor.counterSet = set;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.sampleCount = 2u;
            timing = [device newCounterSampleBufferWithDescriptor:descriptor
                                                            error:nil];
            break;
        }
        if (timing == nil) return [command computeCommandEncoder];
        MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer = timing;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0u;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1u;
        [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            NSData* data = [timing resolveCounterRange:NSMakeRange(0u, 2u)];
            if (completed.status != MTLCommandBufferStatusCompleted ||
                data.length != 2u * sizeof(MTLCounterResultTimestamp)) return;
            const auto* samples = static_cast<const MTLCounterResultTimestamp*>(
                data.bytes);
            if (samples[0].timestamp == 0u ||
                samples[1].timestamp < samples[0].timestamp ||
                samples[0].timestamp == MTLCounterErrorValue ||
                samples[1].timestamp == MTLCounterErrorValue) return;
            std::fprintf(stderr,
                "resting_support_gpu_stage=%s step=%u elapsed_ns=%llu\n",
                stage, step, static_cast<unsigned long long>(
                    samples[1].timestamp - samples[0].timestamp));
        }];
        return [command computeCommandEncoderWithDescriptor:pass];
    }

    bool encode(const metalrobo::MetalNumiHumanSupportGeometryPass& pass) {
        constexpr auto maxU32 = std::numeric_limits<std::uint32_t>::max();
        const auto productsFit = [](std::uint64_t a, std::uint64_t b) {
            return b == 0u || a <= std::numeric_limits<std::uint64_t>::max() / b;
        };
        if (!productsFit(pass.pointWorldStride,
                         3ull * static_cast<std::uint64_t>(pass.dofCount)) ||
            !productsFit(4u, pass.bodyCount) ||
            pass.bodyJacobianPointOffset >
                std::numeric_limits<std::uint64_t>::max() - 4u * pass.bodyCount) {
            return false;
        }
        const std::uint64_t requiredPointRows =
            pass.pointWorldStride * 3ull * pass.dofCount;
        const std::uint64_t bodyPoseElements =
            pass.environmentCount * pass.bodyPoseStride;
        const std::uint64_t requiredProbeRows =
            static_cast<std::uint64_t>(pass.bodyJacobianPointOffset) +
            4u * pass.bodyCount;
        const std::uint64_t publishRegionElements64 =
            static_cast<std::uint64_t>(regionCount_) * pass.environmentCount;
        const std::uint64_t publishLanesPerRegion =
            1ull + 3ull * pass.dofCount;
        if (pass.abiVersion != 2u ||
            pass.structSize != sizeof(pass) ||
            pass.commandBuffer == nullptr || pass.bodyPoses == nullptr ||
            pass.bodyPositionLow == nullptr || pass.pointWorld == nullptr ||
            pass.pointPositionLow == nullptr || pass.pointJacobians == nullptr ||
            pass.standContacts == nullptr ||
            pass.environmentCount != environmentCount_ ||
            pass.articulationFirstBody != validatedFirstBody_ ||
            pass.bodyCount != validatedBodyCount_ ||
            (skipNonAncestorSupportBodyDofs_ &&
             pass.dofCount != validatedDofCount_) ||
            pass.standContactCount != regionCount_ ||
            pass.bodyCount == 0u || pass.bodyPoseStride < pass.bodyCount ||
            pass.pointWorldStride < pass.pointCount || pass.dofCount == 0u ||
            pass.bodyPoseStride > maxU32 || pass.pointWorldStride > maxU32 ||
            pass.pointJacobianStride > maxU32 || pass.pointCount > maxU32 ||
            pass.bodyCount > maxU32 || pass.environmentCount > maxU32 ||
            pass.bodyJacobianPointOffset == MR_INVALID_INDEX ||
            pass.pointJacobianStride < requiredPointRows ||
            bodyPoseElements > maxU32 ||
            requiredProbeRows > pass.pointWorldStride ||
            !productsFit(publishRegionElements64, publishLanesPerRegion) ||
            publishRegionElements64 * publishLanesPerRegion > maxU32 ||
            !productsFit(pass.environmentCount, pass.bodyPoseStride) ||
            !productsFit(pass.environmentCount, pass.pointWorldStride) ||
            !productsFit(pass.environmentCount, pass.pointJacobianStride) ||
            pass.environmentCount * pass.bodyPoseStride >
                pass.bodyPoseElementCount ||
            pass.environmentCount * pass.pointWorldStride >
                pass.pointWorldElementCount ||
            pass.environmentCount * pass.pointJacobianStride >
                pass.pointJacobianElementCount) {
            return false;
        }
        for (const auto& region : hostRegions_) {
            const std::uint64_t probeBegin = pass.bodyJacobianPointOffset;
            const std::uint64_t probeEnd = probeBegin + 4ull * pass.bodyCount;
            if (region.pointQueryIndex >= pass.pointCount ||
                (region.pointQueryIndex >= probeBegin &&
                 region.pointQueryIndex < probeEnd)) return false;
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> bodies =
            (__bridge id<MTLBuffer>)pass.bodyPoses;
        id<MTLBuffer> bodyLow =
            (__bridge id<MTLBuffer>)pass.bodyPositionLow;
        id<MTLBuffer> pointWorld =
            (__bridge id<MTLBuffer>)pass.pointWorld;
        id<MTLBuffer> pointLow =
            (__bridge id<MTLBuffer>)pass.pointPositionLow;
        id<MTLBuffer> pointJacobian =
            (__bridge id<MTLBuffer>)pass.pointJacobians;
        id<MTLBuffer> contacts =
            (__bridge id<MTLBuffer>)pass.standContacts;
        if (command == nil || !sameDevice(device_, bodies) ||
            !sameDevice(device_, bodyLow) ||
            !sameDevice(device_, pointWorld) ||
            !sameDevice(device_, pointLow) ||
            !sameDevice(device_, pointJacobian) ||
            !sameDevice(device_, contacts) ||
            bodies.length < pass.bodyPoseElementCount *
                sizeof(MRArticulatedBodyPoseGPU) ||
            bodyLow.length < pass.bodyPoseElementCount * sizeof(mr_float4) ||
            pointWorld.length < pass.pointWorldElementCount *
                sizeof(MRArticulatedPointWorldGPU) ||
            pointLow.length < pass.pointWorldElementCount * sizeof(mr_float4) ||
            pointJacobian.length < pass.pointJacobianElementCount * sizeof(float) ||
            contacts.length < pass.standContactCount *
                sizeof(MRNumiHumanStandContactGPU)) {
            return false;
        }

        if (normalizedOrientations_ == nil) {
            normalizedOrientations_ = [device_ newBufferWithLength:
                static_cast<NSUInteger>(bodyPoseElements) * sizeof(mr_float4)
                options:MTLResourceStorageModePrivate];
            rotationZBasis_ = [device_ newBufferWithLength:
                static_cast<NSUInteger>(bodyPoseElements) *
                    sizeof(MRHumanRestingSupportRotationZBasisGPU)
                options:MTLResourceStorageModePrivate];
            if (normalizedOrientations_ == nil || rotationZBasis_ == nil) return false;
            orientationStride_ =
                static_cast<std::uint32_t>(pass.bodyPoseStride);
            orientationElementCount_ = bodyPoseElements;
        }
        if (orientationStride_ != pass.bodyPoseStride ||
            orientationElementCount_ != bodyPoseElements) return false;

        MRHumanRestingSupportDispatchGPU dispatch{};
        dispatch.counts = {
            vertexCount_, regionCount_, pass.dofCount,
            static_cast<mr_u32>(environmentCount_),
        };
        dispatch.strides = {
            static_cast<mr_u32>(pass.bodyPoseStride),
            pass.bodyJacobianPointOffset,
            static_cast<mr_u32>(pass.pointWorldStride),
            static_cast<mr_u32>(pass.pointJacobianStride),
        };
        dispatch.identity = {
            pass.articulationFirstBody,
            influenceCount_, regionCount_,
            MR_NUMI_HUMAN_SUPPORT_GEOMETRY_ABI_VERSION,
        };

        id<MTLBlitCommandEncoder> reset = [command blitCommandEncoder];
        if (reset == nil) return false;
        [reset fillBuffer:minimumHeightKeys_
                    range:NSMakeRange(0u, minimumHeightKeys_.length)
                    value:0xffu];
        [reset fillBuffer:selectedVertices_
                    range:NSMakeRange(0u, selectedVertices_.length)
                    value:0xffu];
        [reset fillBuffer:invalidRegionFlags_
                    range:NSMakeRange(0u, invalidRegionFlags_.length)
                    value:0u];
        [reset endEncoding];

        id<MTLComputeCommandEncoder> orientation = timedSupportEncoder(
            command, device_, "support_normalize_poses", pass.stepIndex);
        if (orientation == nil) return false;
        orientation.label = @"Numi Human cached normalized orientations";
        [orientation setComputePipelineState:orientationPipeline_];
        [orientation setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [orientation setBuffer:bodies offset:0u atIndex:1u];
        [orientation setBuffer:normalizedOrientations_ offset:0u atIndex:2u];
        [orientation setBuffer:rotationZBasis_ offset:0u atIndex:3u];
        [orientation setBuffer:bodyLow offset:0u atIndex:4u];
        const NSUInteger bodyPoseCount = static_cast<NSUInteger>(bodyPoseElements);
        [orientation dispatchThreadgroups:MTLSizeMake(
             (bodyPoseCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
        [orientation endEncoding];

        id<MTLComputeCommandEncoder> positions = timedSupportEncoder(
            command, device_, "support_positions", pass.stepIndex);
        if (positions == nil) return false;
        positions.label = @"Numi Human full-skin support minima";
        [positions setComputePipelineState:positionsPipeline_];
        [positions setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [positions setBuffer:vertexMap_ offset:vertexMapOffsetBytes_ atIndex:1u];
        [positions setBuffer:influences_ offset:0u atIndex:2u];
        [positions setBuffer:bodies offset:0u atIndex:3u];
        [positions setBuffer:bodyLow offset:0u atIndex:4u];
        [positions setBuffer:regionForVertex_ offset:0u atIndex:5u];
        [positions setBuffer:worldZKeys_ offset:0u atIndex:6u];
        [positions setBuffer:minimumHeightKeys_ offset:0u atIndex:7u];
        [positions setBuffer:invalidRegionFlags_ offset:0u atIndex:8u];
        [positions setBuffer:normalizedOrientations_ offset:0u atIndex:9u];
        [positions setBuffer:rotationZBasis_ offset:0u atIndex:10u];
        if (bed_.counts.z) {
            [positions setBytes:&bed_ length:sizeof(bed_) atIndex:11u];
            [positions setBuffer:bedHeights_ offset:0u atIndex:12u];
        }
        const NSUInteger positionCount =
            static_cast<NSUInteger>(vertexCount_) * environmentCount_;
        [positions dispatchThreadgroups:MTLSizeMake(
             (positionCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
        [positions endEncoding];

        id<MTLComputeCommandEncoder> select = timedSupportEncoder(
            command, device_, "support_select", pass.stepIndex);
        if (select == nil) return false;
        select.label = @"Numi Human deterministic support vertex selection";
        [select setComputePipelineState:selectPipeline_];
        [select setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [select setBuffer:regionForVertex_ offset:0u atIndex:1u];
        [select setBuffer:worldZKeys_ offset:0u atIndex:2u];
        [select setBuffer:minimumHeightKeys_ offset:0u atIndex:3u];
        [select setBuffer:selectedVertices_ offset:0u atIndex:4u];
        [select dispatchThreadgroups:MTLSizeMake(
             (positionCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
        [select endEncoding];

        id<MTLComputeCommandEncoder> publish = timedSupportEncoder(
            command, device_, "support_publish", pass.stepIndex);
        if (publish == nil) return false;
        publish.label = @"Numi Human weighted skin support queries";
        [publish setComputePipelineState:publishPipeline_];
        [publish setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        if (skipNonAncestorSupportBodyDofs_) {
            const std::uint32_t ancestryBodyCount = validatedBodyCount_;
            [publish setBytes:&ancestryBodyCount
                       length:sizeof(ancestryBodyCount) atIndex:15u];
        }
        [publish setBuffer:vertexMap_ offset:vertexMapOffsetBytes_ atIndex:1u];
        [publish setBuffer:influences_ offset:0u atIndex:2u];
        [publish setBuffer:regions_ offset:0u atIndex:3u];
        [publish setBuffer:selectedVertices_ offset:0u atIndex:4u];
        [publish setBuffer:worldZKeys_ offset:0u atIndex:5u];
        [publish setBuffer:contacts offset:0u atIndex:6u];
        [publish setBuffer:pointWorld offset:0u atIndex:7u];
        [publish setBuffer:pointLow offset:0u atIndex:8u];
        [publish setBuffer:pointJacobian offset:0u atIndex:9u];
        [publish setBuffer:invalidRegionFlags_ offset:0u atIndex:10u];
        [publish setBuffer:bodies offset:0u atIndex:11u];
        [publish setBuffer:bodyLow offset:0u atIndex:12u];
        [publish setBuffer:normalizedOrientations_ offset:0u atIndex:13u];
        if (bed_.counts.z) {
            [publish setBytes:&bed_ length:sizeof(bed_) atIndex:16u];
            [publish setBuffer:bedHeights_ offset:0u atIndex:17u];
        }
        if (skipNonAncestorSupportBodyDofs_) {
            [publish setBuffer:bodyDofAncestry_ offset:0u atIndex:14u];
        }
        const NSUInteger regionElements =
            static_cast<NSUInteger>(regionCount_) * environmentCount_;
        const NSUInteger positionWidth = skipNonAncestorSupportBodyDofs_
            ? publishPipeline_.threadExecutionWidth : 1u;
        const NSUInteger publishElements = regionElements *
            (positionWidth + 3u * static_cast<NSUInteger>(pass.dofCount));
        [publish dispatchThreadgroups:MTLSizeMake(
             (publishElements + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
        [publish endEncoding];

        const bool firstStepDiagnostic =
            diagnosticCapture_ && pass.stepIndex == 0u;
        const bool motionObserverSample =
            motionObserverCapture_ &&
            pass.stepIndex % motionObserverCadenceSteps_ ==
                motionObserverCadenceSteps_ - 1u;
        if (firstStepDiagnostic || motionObserverSample) {
            id<MTLComputeCommandEncoder> debug =
                [command computeCommandEncoder];
            if (debug == nil) return false;
            debug.label = motionObserverSample
                ? @"Numi Human accepted support selector observer"
                : @"Numi Human first-step support diagnostics";
            [debug setComputePipelineState:debugPipeline_];
            [debug setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [debug setBuffer:regions_ offset:0u atIndex:1u];
            [debug setBuffer:selectedVertices_ offset:0u atIndex:2u];
            [debug setBuffer:invalidRegionFlags_ offset:0u atIndex:3u];
            [debug setBuffer:pointWorld offset:0u atIndex:4u];
            [debug setBuffer:pointLow offset:0u atIndex:5u];
            [debug setBuffer:pointJacobian offset:0u atIndex:6u];
            [debug setBuffer:debugOutput_ offset:0u atIndex:7u];
            [debug dispatchThreadgroups:MTLSizeMake(
                 (regionElements + debugThreadsPerGroup_ - 1u) / debugThreadsPerGroup_,
                 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(debugThreadsPerGroup_, 1u, 1u)];
            [debug endEncoding];
            id<MTLBuffer> output = debugOutput_;
            const std::uint32_t count = regionCount_;
            const std::uint32_t step = pass.stepIndex;
            const std::uint32_t firstMapIndex = static_cast<std::uint32_t>(
                vertexMapOffsetBytes_ / sizeof(MRHumanRestingVertexMap));
            const auto regions = hostRegions_;
            const bool emitMotionSample = motionObserverSample;
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                if (completed.status != MTLCommandBufferStatusCompleted ||
                    output.contents == nullptr) {
                    std::fprintf(stderr,
                        "resting_support_debug status=%u command_status=%lu\n",
                        step, static_cast<unsigned long>(completed.status));
                    return;
                }
                const auto* records = static_cast<const
                    MRHumanRestingSupportDebugGPU*>(output.contents);
                if (emitMotionSample) {
                    if (!motionObserverTrace_.good() || regions.size() != count) {
                        std::fprintf(stderr,
                            "resting_support_motion_observer_write=failed step=%u\n",
                            step + 1u);
                        return;
                    }
                    for (std::uint32_t region = 0u; region < count; ++region) {
                        const auto& record = records[region];
                        const std::uint32_t selected = record.status.x;
                        motionObserverTrace_ << step + 1u << ','
                            << double(step + 1u) * motionObserverTimestepSeconds_
                            << ',' << step << ','
                            << double(step) * motionObserverTimestepSeconds_
                            << ',' << region << ','
                            << regions[region].sourceGeometryIndex
                            << ',' << regions[region].pointQueryIndex << ','
                            << selected << ',';
                        if (selected < vertexCount_) {
                            motionObserverTrace_ << firstMapIndex + selected;
                        }
                        motionObserverTrace_ << ',' << record.status.y << ','
                            << record.status.z << ',' << record.status.w << ','
                            << std::setprecision(17) << record.position.x << ','
                            << record.position.y << ',' << record.position.z << ','
                            << record.jacobian.x << '\n';
                    }
                    motionObserverTrace_.flush();
                    if (!motionObserverTrace_.good()) {
                        std::fprintf(stderr,
                            "resting_support_motion_observer_write=failed step=%u\n",
                            step + 1u);
                        return;
                    }
                }
                if (firstStepDiagnostic) {
                    for (std::uint32_t region = 0u; region < count; ++region) {
                        const auto& record = records[region];
                        std::fprintf(stderr,
                            "resting_support_debug step=%u region=%u selected=%u region_error=%u global_error=%u nonfinite_j=%u point=(%.9g,%.9g,%.9g) max_abs_j=%.9g\n",
                            step, region, record.status.x, record.status.y,
                            record.status.z, record.status.w,
                            record.position.x, record.position.y,
                            record.position.z, record.jacobian.x);
                    }
                }
            }];
        }
        return true;
    }

public:
    void configureMotionObserver(const std::filesystem::path& outputPath,
                                 const std::uint32_t cadenceSteps,
                                 const double timestepSeconds) {
        require(motionObserverRequested_,
                "support motion observer configuration requires its opt-in");
        require(!motionObserverCapture_ && cadenceSteps > 0u &&
                    cadenceSteps <= 32u && std::isfinite(timestepSeconds) &&
                    timestepSeconds > 0.0 && !outputPath.empty() &&
                    !std::filesystem::exists(outputPath),
                "support motion observer output or cadence is invalid");
        motionObserverTrace_.open(outputPath, std::ios::out);
        require(motionObserverTrace_.good(),
                "support motion observer output could not be opened");
        motionObserverTrace_ << std::setprecision(17)
            << metalrobo::human::observer::kSupportSelectorCsvHeader << '\n';
        motionObserverTrace_.flush();
        require(motionObserverTrace_.good(),
                "support motion observer header write failed");
        motionObserverCadenceSteps_ = cadenceSteps;
        motionObserverTimestepSeconds_ = timestepSeconds;
        motionObserverCapture_ = true;
    }

    NumiHumanRestingSupportGeometry(
        id<MTLDevice> device, id<MTLLibrary> library,
        id<MTLBuffer> vertexMap, std::span<const MRHumanRestingVertexMap> hostMap,
        id<MTLBuffer> influences, std::span<const MRHumanRestingInfluence> hostInfluences,
        std::span<const std::uint32_t> regionForVertex,
        std::span<const MRHumanRestingSupportRegionGPU> regions,
        std::span<const MRNumiHumanStandContactGPU> contacts,
        const std::uint32_t environmentCount,
        const std::uint32_t validatedFirstBody,
        const std::uint32_t validatedBodyCount,
        const std::uint32_t validatedDofCount,
        std::span<const std::uint8_t> bodyDofAncestry,
        const bool skipNonAncestorSupportBodyDofs,
        const std::uint64_t fingerprint,
        const NSUInteger vertexMapOffsetBytes,
        const NumiHumanSkinInfluenceLayout influenceLayout,
        const MRHumanRestingBedGPU& bed = {}, id<MTLBuffer> bedHeights = nil
    ) : device_(device), vertexMap_(vertexMap), influences_(influences),
        vertexCount_(static_cast<std::uint32_t>(hostMap.size())),
        regionCount_(static_cast<std::uint32_t>(regions.size())),
        environmentCount_(environmentCount),
        influenceCount_(static_cast<std::uint32_t>(hostInfluences.size())),
        influenceLayout_(influenceLayout),
        validatedFirstBody_(validatedFirstBody),
        validatedBodyCount_(validatedBodyCount),
        validatedDofCount_(validatedDofCount),
        skipNonAncestorSupportBodyDofs_(skipNonAncestorSupportBodyDofs),
        vertexMapOffsetBytes_(vertexMapOffsetBytes),
        fingerprint_(fingerprint), hostRegions_(regions.begin(), regions.end()) {
        bed_=bed; bedHeights_=bedHeights;
        require(bed_.counts.z==0u ||
            (bed_.counts.z==1u && bed_.counts.w==1u && environmentCount_==1u &&
             bed_.counts.x>=2u && bed_.counts.y>=2u &&
             sameDevice(device_,bedHeights_) &&
             bedHeights_.length==size_t(bed_.counts.x)*bed_.counts.y*sizeof(float)),
             "contoured bed requires one environment and one complete fixed height buffer");
        const char* diagnosticSetting =
            std::getenv("NUMI_HUMAN_SUPPORT_DIAGNOSTICS");
        diagnosticCapture_ = diagnosticSetting != nullptr &&
            std::strcmp(diagnosticSetting, "1") == 0;
        const char* motionSetting =
            std::getenv("NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT");
        require(motionSetting == nullptr || motionSetting[0] == '\0' ||
                    std::strcmp(motionSetting, "0") == 0 ||
                    std::strcmp(motionSetting, "1") == 0,
                "NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT must be 0 or 1");
        motionObserverRequested_ = motionSetting != nullptr &&
            std::strcmp(motionSetting, "1") == 0;
        require(device_ != nil && library != nil && sameDevice(device_, vertexMap_) &&
                    sameDevice(device_, influences_) && environmentCount_ > 0u &&
                    hostMap.size() > 0u &&
                    hostMap.size() <= std::numeric_limits<std::uint32_t>::max() &&
                    hostInfluences.size() <= std::numeric_limits<std::uint32_t>::max() &&
                    static_cast<std::uint64_t>(hostMap.size()) * environmentCount_ <=
                        std::numeric_limits<std::uint32_t>::max() &&
                    !regions.empty() && regions.size() == contacts.size() &&
                    regions.size() <= MR_NUMI_HUMAN_STAND_MAX_CONTACTS &&
                    regionForVertex.size() == hostMap.size() && fingerprint_ != 0u &&
                    !hostInfluences.empty() && validatedFirstBody_ != MR_INVALID_INDEX &&
                    validatedBodyCount_ > 0u &&
                    (!skipNonAncestorSupportBodyDofs_ ||
                     (validatedDofCount_ > 0u &&
                      bodyDofAncestry.size() ==
                          static_cast<std::uint64_t>(validatedBodyCount_) *
                              validatedDofCount_)) &&
                    static_cast<std::uint64_t>(validatedFirstBody_) +
                        validatedBodyCount_ <=
                            static_cast<std::uint64_t>(
                                std::numeric_limits<std::uint32_t>::max()) + 1u,
                "resting support geometry received incomplete source assets");
        if (skipNonAncestorSupportBodyDofs_) {
            bodyDofAncestry_ = [device_ newBufferWithBytes:bodyDofAncestry.data()
                length:bodyDofAncestry.size_bytes()
                options:MTLResourceStorageModeShared];
            require(bodyDofAncestry_ != nil &&
                        sameDevice(device_, bodyDofAncestry_),
                    "body/DoF support ancestry mask could not be uploaded");
        }
        require(vertexMapOffsetBytes_ % alignof(MRHumanRestingVertexMap) == 0u &&
                    vertexMapOffsetBytes_ <= vertexMap_.length &&
                    hostMap.size() <= (vertexMap_.length - vertexMapOffsetBytes_) /
                        sizeof(hostMap.front()) &&
                    hostInfluences.size() <= influences_.length /
                        sizeof(hostInfluences.front()),
                "resting support source buffers do not match their registered assets");
        std::vector<std::uint32_t> regionCounts(regionCount_, 0u);
        for (std::size_t vertex = 0u; vertex < hostMap.size(); ++vertex) {
            const auto& map = hostMap[vertex];
            const std::uint32_t region = regionForVertex[vertex];
            const bool isSkin = map.deformationKind == 3u;
            require((isSkin && region < regionCount_ && map.influenceCount > 0u) ||
                        (!isSkin && region == MR_INVALID_INDEX),
                    "every registered skin vertex must belong to exactly one support region");
            if (!isSkin) continue;
            const std::uint64_t influenceStride =
                influenceLayout_ == NumiHumanSkinInfluenceLayout::tile32
                    ? 32u : 1u;
            const std::uint64_t lastInfluence =
                static_cast<std::uint64_t>(map.firstInfluence) +
                static_cast<std::uint64_t>(map.influenceCount - 1u) *
                    influenceStride;
            require(map.firstInfluence < hostInfluences.size() &&
                        lastInfluence < hostInfluences.size(),
                    "registered skin influence range exceeds its selected layout");
            for (std::uint32_t local = 0u; local < map.influenceCount; ++local) {
                const auto& influence = hostInfluences[
                    map.firstInfluence +
                    static_cast<std::uint64_t>(local) * influenceStride];
                require(influence.body.x >= validatedFirstBody_ &&
                            static_cast<std::uint64_t>(influence.body.x) <
                                static_cast<std::uint64_t>(validatedFirstBody_) +
                                    validatedBodyCount_ &&
                            std::isfinite(influence.positionAndWeight.x) &&
                            std::isfinite(influence.positionAndWeight.y) &&
                            std::isfinite(influence.positionAndWeight.z) &&
                            std::isfinite(influence.positionAndWeight.w) &&
                            influence.positionAndWeight.w > 0.0f,
                        "registered skin influence has an invalid body or static weight");
            }
            ++regionCounts[region];
        }
        require(std::all_of(regionCounts.begin(), regionCounts.end(),
                            [](std::uint32_t count) { return count != 0u; }),
                "a derived support Voronoi region contains no registered skin vertices");
        for (std::size_t region = 0u; region < regions.size(); ++region) {
            require(regions[region].reserved0 == 0u && regions[region].reserved1 == 0u &&
                        regions[region].pointQueryIndex == contacts[region].pointQueryIndex &&
                        regions[region].sourceGeometryIndex == contacts[region].sourceGeometryIndex,
                    "support region lost its source NHCNT row identity");
            for (std::size_t prior = 0u; prior < region; ++prior) {
                require(regions[region].pointQueryIndex != regions[prior].pointQueryIndex,
                        "support regions share a point-query output row");
            }
        }
        regionForVertex_ = [device_ newBufferWithBytes:regionForVertex.data()
            length:regionForVertex.size_bytes() options:MTLResourceStorageModeShared];
        regions_ = [device_ newBufferWithBytes:regions.data()
            length:regions.size_bytes() options:MTLResourceStorageModeShared];
        require(regionForVertex_ != nil && regions_ != nil,
                "resting support region buffers could not be uploaded");
        // The specialized minimum pass may omit only the static influence
        // guards proven above. Its dynamic body-pose/basis checks and all
        // invalid-region publication remain in the shader.
        positionsPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_positions", true, true,
            influenceLayout_, false, bed_.counts.z!=0u);
        orientationPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_normalize_poses");
        selectPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_select");
        publishPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_publish", false, true,
            influenceLayout_, skipNonAncestorSupportBodyDofs_, bed_.counts.z!=0u);
        if (diagnosticCapture_ || motionObserverRequested_) {
            debugPipeline_ = makePipeline(device_, library,
                @"nm_human_resting_support_debug");
        }
        const NSUInteger supportThreadsPerGroup = std::min<NSUInteger>(
            128u, std::min(orientationPipeline_.maxTotalThreadsPerThreadgroup,
                std::min(positionsPipeline_.maxTotalThreadsPerThreadgroup,
                std::min(selectPipeline_.maxTotalThreadsPerThreadgroup,
                         publishPipeline_.maxTotalThreadsPerThreadgroup))));
        debugThreadsPerGroup_ = supportThreadsPerGroup;
        if (debugPipeline_ != nil) {
            debugThreadsPerGroup_ = std::min<NSUInteger>(
                supportThreadsPerGroup, debugPipeline_.maxTotalThreadsPerThreadgroup);
        }
        // Keep the historical diagnostic mode's dispatch specialization exact.
        // The new motion observer's read-only kernel must not constrain or alter
        // the physical support-geometry dispatch size.
        threadsPerGroup_ = diagnosticCapture_ && debugPipeline_ != nil
            ? debugThreadsPerGroup_ : supportThreadsPerGroup;
        require(threadsPerGroup_ > 0u && debugThreadsPerGroup_ > 0u,
                "resting support pipelines expose no dispatch lanes");
        require(!skipNonAncestorSupportBodyDofs_ ||
                    (publishPipeline_.threadExecutionWidth > 0u &&
                     threadsPerGroup_ % publishPipeline_.threadExecutionWidth == 0u),
                "cooperative support positions require complete SIMD groups");
        worldZKeys_ = [device_ newBufferWithLength:
            static_cast<NSUInteger>(environmentCount_) * vertexCount_ *
                sizeof(std::uint32_t)
            options:MTLResourceStorageModePrivate];
        const NSUInteger regionBytes = static_cast<NSUInteger>(environmentCount_) *
            regionCount_ * sizeof(std::uint32_t);
        const NSUInteger invalidFlagBytes =
            static_cast<NSUInteger>(environmentCount_) * (regionCount_ + 1u) *
            sizeof(std::uint32_t);
        minimumHeightKeys_ = [device_ newBufferWithLength:regionBytes
            options:MTLResourceStorageModePrivate];
        selectedVertices_ = [device_ newBufferWithLength:regionBytes
            options:MTLResourceStorageModePrivate];
        invalidRegionFlags_ = [device_ newBufferWithLength:invalidFlagBytes
            options:MTLResourceStorageModePrivate];
        if (diagnosticCapture_ || motionObserverRequested_) {
            debugOutput_ = [device_ newBufferWithLength:
                static_cast<NSUInteger>(regionCount_) *
                    sizeof(MRHumanRestingSupportDebugGPU)
                options:MTLResourceStorageModeShared];
        }
        require(worldZKeys_ != nil && minimumHeightKeys_ != nil &&
                    selectedVertices_ != nil && invalidRegionFlags_ != nil &&
                    (!(diagnosticCapture_ || motionObserverRequested_) ||
                     debugOutput_ != nil),
                "resting support transient GPU buffers could not be allocated");
    }

    [[nodiscard]] metalrobo::MetalNumiHumanSupportGeometryProgram program() {
        return {
            .context = this,
            .encodePreDynamics = &encodeCallback,
            .abort = &abortCallback,
            .fingerprint = fingerprint_,
            .usePerContactSupportPlanes = bed_.counts.z!=0u,
        };
    }
};
