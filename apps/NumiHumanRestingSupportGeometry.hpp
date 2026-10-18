#pragma once

#include <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_motion_observer.hpp"
#include "metalrobo/numi_human_support_geometry_gpu.h"
#include "metalrobo/numi_human_resting_bed_gpu.h"

#include <algorithm>
#include <array>
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
#include "NumiHumanRestingSkinInfluencePipeline.hpp"

// Encoder-only client for the operator's accepted-pose support-geometry lease.
// The caller supplies the same preuploaded LBS buffers used by the viewer; this
// class uploads only the derived region assignment and its compact contact-row map.
class NumiHumanRestingSupportGeometry final {
    __strong id<MTLDevice> device_ = nil;
    __strong id<MTLComputePipelineState> positionsPipeline_ = nil;
    __strong id<MTLComputePipelineState> predictedPositionsPipeline_ = nil;
    __strong id<MTLComputePipelineState> probeVelocitiesPipeline_ = nil;
    __strong id<MTLComputePipelineState> chooseEndpointPipeline_ = nil;
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
    __strong id<MTLBuffer> lineage_ = nil;
    __strong id<MTLBuffer> worldZKeys_ = nil;
    __strong id<MTLBuffer> predictedWorldZKeys_ = nil;
    __strong id<MTLBuffer> minimumHeightKeys_ = nil;
    __strong id<MTLBuffer> predictedMinimumHeightKeys_ = nil;
    __strong id<MTLBuffer> predictedSelectedVertices_ = nil;
    __strong id<MTLBuffer> currentSelectedVertices_ = nil;
    __strong id<MTLBuffer> selectedVertices_ = nil;
    __strong id<MTLBuffer> bodyProbeVelocities_ = nil;
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
        const char* setting =
            std::getenv("NUMI_HUMAN_SUPPORT_ENCODER_DIAGNOSTICS");
        const bool diagnostics = setting != nullptr &&
            std::strcmp(setting, "1") == 0;
        if (context == nullptr) {
            if (diagnostics)
                std::fprintf(stderr, "support_encoder_rejected reason=callback:null_context\n");
            return false;
        }
        try {
            return static_cast<NumiHumanRestingSupportGeometry*>(context)
                ->encode(pass);
        } catch (const std::exception& exception) {
            if (diagnostics)
                std::fprintf(stderr,
                    "support_encoder_rejected reason=callback:exception what=%s\n",
                    exception.what());
            return false;
        } catch (...) {
            if (diagnostics)
                std::fprintf(stderr,
                    "support_encoder_rejected reason=callback:unknown_exception\n");
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
        const bool contouredBed = false,
        const bool predictedEndpoint = false
    ) {
        NSError* functionError = nil;
        const auto function = numiHumanRestingMakeSkinInfluenceFunction(
            library, name,
            {validatedSkinInfluences, specializeSkinInfluenceLayout,
             influenceLayout, skipNonAncestorSupportBodyDofs, contouredBed,
             predictedEndpoint},
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
        const auto reject = [&pass](const char* reason) {
            const char* setting =
                std::getenv("NUMI_HUMAN_SUPPORT_ENCODER_DIAGNOSTICS");
            if (setting != nullptr && std::strcmp(setting, "1") == 0) {
                std::fprintf(stderr,
                    "support_encoder_rejected reason=%s abi=%u size=%u expected_size=%zu phase=%u step=%u first_body=%u body_count=%llu dof=%u env=%llu body_stride=%llu body_elements=%llu point_count=%llu point_stride=%llu point_elements=%llu jacobian_stride=%llu jacobian_elements=%llu contact_count=%llu probe_offset=%u ground=(%.9g,%.9g,%.9g) dt=%.9g\n",
                    reason, pass.abiVersion, pass.structSize, sizeof(pass),
                    static_cast<unsigned>(pass.phase), pass.stepIndex,
                    pass.articulationFirstBody,
                    static_cast<unsigned long long>(pass.bodyCount),
                    pass.dofCount,
                    static_cast<unsigned long long>(pass.environmentCount),
                    static_cast<unsigned long long>(pass.bodyPoseStride),
                    static_cast<unsigned long long>(pass.bodyPoseElementCount),
                    static_cast<unsigned long long>(pass.pointCount),
                    static_cast<unsigned long long>(pass.pointWorldStride),
                    static_cast<unsigned long long>(pass.pointWorldElementCount),
                    static_cast<unsigned long long>(pass.pointJacobianStride),
                    static_cast<unsigned long long>(pass.pointJacobianElementCount),
                    static_cast<unsigned long long>(pass.standContactCount),
                    pass.bodyJacobianPointOffset, pass.groundPoint.x,
                    pass.groundPoint.y, pass.groundPoint.z,
                    pass.timestepSeconds);
            }
            return false;
        };
        constexpr auto maxU32 = std::numeric_limits<std::uint32_t>::max();
        const auto productsFit = [](std::uint64_t a, std::uint64_t b) {
            return b == 0u || a <= std::numeric_limits<std::uint64_t>::max() / b;
        };
        if (!productsFit(pass.pointWorldStride,
                         3ull * static_cast<std::uint64_t>(pass.dofCount)))
            return reject("overflow:point_world_stride_times_dofs");
        if (!productsFit(4u, pass.bodyCount))
            return reject("overflow:four_body_probes");
        if (pass.bodyJacobianPointOffset >
                std::numeric_limits<std::uint64_t>::max() - 4u * pass.bodyCount)
            return reject("overflow:probe_offset_plus_four_bodies");
        const bool predictedEndpoint =
            pass.phase ==
                metalrobo::MetalNumiHumanSupportGeometryPhase::predictedFreeEndpoint;
        const bool currentPose =
            pass.phase ==
                metalrobo::MetalNumiHumanSupportGeometryPhase::currentPose;
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
        if (pass.abiVersion != 3u) return reject("metadata:abi_version");
        if (!currentPose && !predictedEndpoint)
            return reject("metadata:phase");
        if (pass.structSize != sizeof(pass))
            return reject("metadata:struct_size");
        if (pass.commandBuffer == nullptr) return reject("metadata:command_buffer");
        if (pass.bodyPoses == nullptr) return reject("metadata:body_poses");
        if (pass.bodyPositionLow == nullptr)
            return reject("metadata:body_position_low");
        if (pass.pointWorld == nullptr) return reject("metadata:point_world");
        if (pass.pointPositionLow == nullptr)
            return reject("metadata:point_position_low");
        if (pass.pointJacobians == nullptr)
            return reject("metadata:point_jacobians");
        if (pass.standContacts == nullptr)
            return reject("metadata:stand_contacts");
        if (pass.environmentCount != environmentCount_)
            return reject("metadata:environment_count");
        if (predictedEndpoint && environmentCount_ != 1u)
            return reject("predicted:environment_count");
        if (predictedEndpoint && bed_.counts.z != 0u)
            return reject("predicted:contoured_bed_unsupported");
        if (predictedEndpoint && pass.groundNormal.x != 0.0f)
            return reject("predicted:ground_normal_x");
        if (predictedEndpoint && pass.groundNormal.y != 0.0f)
            return reject("predicted:ground_normal_y");
        if (predictedEndpoint && pass.groundNormal.z != 1.0f)
            return reject("predicted:ground_normal_z");
        if (predictedEndpoint && !std::isfinite(pass.groundPoint.z))
            return reject("predicted:ground_point_z_nonfinite");
        if (predictedEndpoint && pass.sourceFreeVelocity == nullptr)
            return reject("predicted:source_free_velocity");
        if (predictedEndpoint && !std::isfinite(pass.timestepSeconds))
            return reject("predicted:timestep_nonfinite");
        if (predictedEndpoint && !(pass.timestepSeconds > 0.0f))
            return reject("predicted:timestep_nonpositive");
        if (predictedEndpoint && pass.sourceFreeVelocityElementOffset > maxU32)
            return reject("predicted:velocity_offset_u32");
        if (predictedEndpoint && pass.sourceFreeVelocityElementCount > maxU32)
            return reject("predicted:velocity_count_u32");
        if (predictedEndpoint && pass.sourceFreeVelocityElementStride > maxU32)
            return reject("predicted:velocity_stride_u32");
        if (predictedEndpoint && pass.sourceFreeVelocityElementStride == 0u)
            return reject("predicted:velocity_stride_zero");
        if (predictedEndpoint && pass.sourceFreeVelocityElementOffset + pass.dofCount >
                pass.sourceFreeVelocityElementStride)
            return reject("predicted:velocity_dofs_exceed_stride");
        if (predictedEndpoint && pass.sourceFreeVelocityElementOffset + pass.dofCount >
                pass.sourceFreeVelocityElementCount)
            return reject("predicted:velocity_dofs_exceed_count");
        if (currentPose && pass.sourceFreeVelocity != nullptr)
            return reject("current_pose:unexpected_velocity");
        if (currentPose && pass.sourceFreeVelocityElementOffset != 0u)
            return reject("current_pose:velocity_offset_nonzero");
        if (currentPose && pass.sourceFreeVelocityElementCount != 0u)
            return reject("current_pose:velocity_count_nonzero");
        if (currentPose && pass.sourceFreeVelocityElementStride != 0u)
            return reject("current_pose:velocity_stride_nonzero");
        if (currentPose && pass.timestepSeconds != 0.0f)
            return reject("current_pose:timestep_nonzero");
        if (pass.articulationFirstBody != validatedFirstBody_)
            return reject("metadata:first_body_mismatch");
        if (pass.bodyCount != validatedBodyCount_)
            return reject("metadata:body_count_mismatch");
        if (skipNonAncestorSupportBodyDofs_ && pass.dofCount != validatedDofCount_)
            return reject("metadata:dof_count_mismatch");
        if (pass.standContactCount != regionCount_)
            return reject("metadata:contact_count_mismatch");
        if (pass.bodyCount == 0u) return reject("metadata:zero_body_count");
        if (pass.bodyPoseStride < pass.bodyCount)
            return reject("metadata:body_pose_stride_too_small");
        if (pass.pointWorldStride < pass.pointCount)
            return reject("metadata:point_stride_too_small");
        if (pass.dofCount == 0u) return reject("metadata:zero_dof_count");
        if (pass.bodyPoseStride > maxU32)
            return reject("metadata:body_pose_stride_u32");
        if (pass.pointWorldStride > maxU32)
            return reject("metadata:point_stride_u32");
        if (pass.pointJacobianStride > maxU32)
            return reject("metadata:jacobian_stride_u32");
        if (pass.pointCount > maxU32) return reject("metadata:point_count_u32");
        if (pass.bodyCount > maxU32) return reject("metadata:body_count_u32");
        if (pass.environmentCount > maxU32)
            return reject("metadata:environment_count_u32");
        if (pass.bodyJacobianPointOffset == MR_INVALID_INDEX)
            return reject("metadata:invalid_probe_offset");
        if (pass.pointJacobianStride < requiredPointRows)
            return reject("metadata:jacobian_stride_too_small");
        if (bodyPoseElements > maxU32)
            return reject("metadata:body_pose_elements_u32");
        if (requiredProbeRows > pass.pointWorldStride)
            return reject("metadata:probe_rows_exceed_point_stride");
        if (!productsFit(publishRegionElements64, publishLanesPerRegion))
            return reject("metadata:publish_elements_overflow");
        if (publishRegionElements64 * publishLanesPerRegion > maxU32)
            return reject("metadata:publish_elements_u32");
        if (!productsFit(pass.environmentCount, pass.bodyPoseStride))
            return reject("metadata:environment_body_pose_overflow");
        if (!productsFit(pass.environmentCount, pass.pointWorldStride))
            return reject("metadata:environment_point_world_overflow");
        if (!productsFit(pass.environmentCount, pass.pointJacobianStride))
            return reject("metadata:environment_jacobian_overflow");
        if (pass.environmentCount * pass.bodyPoseStride > pass.bodyPoseElementCount)
            return reject("metadata:body_pose_elements_too_small");
        if (pass.environmentCount * pass.pointWorldStride > pass.pointWorldElementCount)
            return reject("metadata:point_world_elements_too_small");
        if (pass.environmentCount * pass.pointJacobianStride >
                pass.pointJacobianElementCount)
            return reject("metadata:jacobian_elements_too_small");

        for (const auto& region : hostRegions_) {
            const std::uint64_t probeBegin = pass.bodyJacobianPointOffset;
            const std::uint64_t probeEnd = probeBegin + 4ull * pass.bodyCount;
            if (region.pointQueryIndex >= pass.pointCount)
                return reject("region:query_out_of_range");
            if (region.pointQueryIndex >= probeBegin &&
                region.pointQueryIndex < probeEnd)
                return reject("region:query_overlaps_probe_rows");
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
        id<MTLBuffer> sourceFreeVelocity = predictedEndpoint
            ? (__bridge id<MTLBuffer>)pass.sourceFreeVelocity : nil;
        if (command == nil) return reject("buffer:command_buffer_nil");
        if (!sameDevice(device_, bodies)) return reject("buffer:body_poses_device");
        if (!sameDevice(device_, bodyLow)) return reject("buffer:body_low_device");
        if (!sameDevice(device_, pointWorld)) return reject("buffer:point_world_device");
        if (!sameDevice(device_, pointLow)) return reject("buffer:point_low_device");
        if (!sameDevice(device_, pointJacobian)) return reject("buffer:point_jacobian_device");
        if (!sameDevice(device_, contacts)) return reject("buffer:contacts_device");
        if (predictedEndpoint && !sameDevice(device_, sourceFreeVelocity))
            return reject("buffer:source_velocity_device");
        if (predictedEndpoint && sourceFreeVelocity.length <
                pass.sourceFreeVelocityElementCount * sizeof(float))
            return reject("buffer:source_velocity_length");
        if (bodies.length < pass.bodyPoseElementCount *
                sizeof(MRArticulatedBodyPoseGPU))
            return reject("buffer:body_poses_length");
        if (bodyLow.length < pass.bodyPoseElementCount * sizeof(mr_float4))
            return reject("buffer:body_low_length");
        if (pointWorld.length < pass.pointWorldElementCount *
                sizeof(MRArticulatedPointWorldGPU))
            return reject("buffer:point_world_length");
        if (pointLow.length < pass.pointWorldElementCount * sizeof(mr_float4))
            return reject("buffer:point_low_length");
        if (pointJacobian.length < pass.pointJacobianElementCount * sizeof(float))
            return reject("buffer:point_jacobian_length");
        if (contacts.length < pass.standContactCount *
                sizeof(MRNumiHumanStandContactGPU))
            return reject("buffer:contacts_length");

        if (normalizedOrientations_ == nil) {
            normalizedOrientations_ = [device_ newBufferWithLength:
                static_cast<NSUInteger>(bodyPoseElements) * sizeof(mr_float4)
                options:MTLResourceStorageModePrivate];
            rotationZBasis_ = [device_ newBufferWithLength:
                static_cast<NSUInteger>(bodyPoseElements) *
                    sizeof(MRHumanRestingSupportRotationZBasisGPU)
                options:MTLResourceStorageModePrivate];
            bodyProbeVelocities_ = [device_ newBufferWithLength:
                static_cast<NSUInteger>(bodyPoseElements) * 4u *
                    sizeof(mr_float4)
                options:MTLResourceStorageModePrivate];
            if (normalizedOrientations_ == nil || rotationZBasis_ == nil ||
                bodyProbeVelocities_ == nil) return reject("buffer:transient_allocation");
            orientationStride_ =
                static_cast<std::uint32_t>(pass.bodyPoseStride);
            orientationElementCount_ = bodyPoseElements;
        }
        if (orientationStride_ != pass.bodyPoseStride ||
            orientationElementCount_ != bodyPoseElements)
            return reject("buffer:cached_orientation_stride");

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
        dispatch.prediction.groundPointAndTimestep = {
            pass.groundPoint.x, pass.groundPoint.y, pass.groundPoint.z,
            pass.timestepSeconds,
        };
        dispatch.prediction.sourceVelocity = {
            static_cast<mr_u32>(pass.sourceFreeVelocityElementOffset),
            static_cast<mr_u32>(pass.sourceFreeVelocityElementCount),
            static_cast<mr_u32>(pass.sourceFreeVelocityElementStride),
            static_cast<mr_u32>(pass.bodyCount),
        };

        id<MTLBlitCommandEncoder> reset = [command blitCommandEncoder];
        if (reset == nil) return reject("encoder:reset");
        if (predictedEndpoint) {
            [reset fillBuffer:predictedMinimumHeightKeys_
                        range:NSMakeRange(0u, predictedMinimumHeightKeys_.length)
                        value:0xffu];
            [reset fillBuffer:predictedSelectedVertices_
                        range:NSMakeRange(0u, predictedSelectedVertices_.length)
                        value:0xffu];
            [reset copyFromBuffer:selectedVertices_
                     sourceOffset:0u
                         toBuffer:currentSelectedVertices_
                destinationOffset:0u
                             size:selectedVertices_.length];
        } else {
            [reset fillBuffer:minimumHeightKeys_
                        range:NSMakeRange(0u, minimumHeightKeys_.length)
                        value:0xffu];
            [reset fillBuffer:selectedVertices_
                        range:NSMakeRange(0u, selectedVertices_.length)
                        value:0xffu];
            [reset fillBuffer:invalidRegionFlags_
                        range:NSMakeRange(0u, invalidRegionFlags_.length)
                        value:0u];
        }
        [reset endEncoding];

        const NSUInteger bodyPoseCount = static_cast<NSUInteger>(bodyPoseElements);
        if (!predictedEndpoint) {
            id<MTLComputeCommandEncoder> orientation = timedSupportEncoder(
                command, device_, "support_normalize_poses", pass.stepIndex);
            if (orientation == nil) return reject("encoder:orientation");
            orientation.label = @"Numi Human cached normalized orientations";
            [orientation setComputePipelineState:orientationPipeline_];
            [orientation setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [orientation setBuffer:bodies offset:0u atIndex:1u];
            [orientation setBuffer:normalizedOrientations_ offset:0u atIndex:2u];
            [orientation setBuffer:rotationZBasis_ offset:0u atIndex:3u];
            [orientation setBuffer:bodyLow offset:0u atIndex:4u];
            [orientation dispatchThreadgroups:MTLSizeMake(
                 (bodyPoseCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
            [orientation endEncoding];
        } else {
            const NSUInteger probeCount =
                static_cast<NSUInteger>(pass.bodyCount) * 4u;
            id<MTLComputeCommandEncoder> velocity = timedSupportEncoder(
                command, device_, "support_free_velocity_probes", pass.stepIndex);
            if (velocity == nil) return reject("encoder:probe_velocity");
            velocity.label = @"Numi Human source-free body-probe velocities";
            [velocity setComputePipelineState:probeVelocitiesPipeline_];
            [velocity setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [velocity setBuffer:pointJacobian offset:0u atIndex:1u];
            [velocity setBuffer:sourceFreeVelocity offset:0u atIndex:2u];
            [velocity setBuffer:bodyProbeVelocities_ offset:0u atIndex:3u];
            [velocity dispatchThreadgroups:MTLSizeMake(
                 (probeCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
            [velocity endEncoding];
        }

        const NSUInteger positionCount =
            static_cast<NSUInteger>(vertexCount_) * environmentCount_;
        id<MTLComputeCommandEncoder> positions = timedSupportEncoder(
            command, device_, predictedEndpoint
                ? "support_predicted_endpoint_positions" : "support_positions",
            pass.stepIndex);
        if (positions == nil) return reject("encoder:positions");
        positions.label = predictedEndpoint
            ? @"Numi Human predicted full-skin support minima"
            : @"Numi Human full-skin support minima";
        [positions setComputePipelineState:predictedEndpoint
            ? predictedPositionsPipeline_ : positionsPipeline_];
        [positions setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [positions setBuffer:vertexMap_ offset:vertexMapOffsetBytes_ atIndex:1u];
        [positions setBuffer:influences_ offset:0u atIndex:2u];
        [positions setBuffer:bodies offset:0u atIndex:3u];
        [positions setBuffer:bodyLow offset:0u atIndex:4u];
        [positions setBuffer:regionForVertex_ offset:0u atIndex:5u];
        [positions setBuffer:predictedEndpoint
            ? predictedWorldZKeys_ : worldZKeys_ offset:0u atIndex:6u];
        [positions setBuffer:predictedEndpoint
            ? predictedMinimumHeightKeys_ : minimumHeightKeys_ offset:0u atIndex:7u];
        [positions setBuffer:invalidRegionFlags_ offset:0u atIndex:8u];
        [positions setBuffer:normalizedOrientations_ offset:0u atIndex:9u];
        [positions setBuffer:rotationZBasis_ offset:0u atIndex:10u];
        if (bed_.counts.z) {
            [positions setBytes:&bed_ length:sizeof(bed_) atIndex:11u];
            [positions setBuffer:bedHeights_ offset:0u atIndex:12u];
        }
        if (predictedEndpoint)
            [positions setBuffer:bodyProbeVelocities_ offset:0u atIndex:13u];
        [positions dispatchThreadgroups:MTLSizeMake(
             (positionCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
        [positions endEncoding];

        id<MTLComputeCommandEncoder> select = timedSupportEncoder(
            command, device_, predictedEndpoint
                ? "support_select_predicted_endpoint" : "support_select",
            pass.stepIndex);
        if (select == nil) return reject("encoder:select");
        select.label = predictedEndpoint
            ? @"Numi Human predicted endpoint vertex selection"
            : @"Numi Human deterministic support vertex selection";
        [select setComputePipelineState:selectPipeline_];
        [select setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [select setBuffer:regionForVertex_ offset:0u atIndex:1u];
        [select setBuffer:predictedEndpoint
            ? predictedWorldZKeys_ : worldZKeys_ offset:0u atIndex:2u];
        [select setBuffer:predictedEndpoint
            ? predictedMinimumHeightKeys_ : minimumHeightKeys_ offset:0u atIndex:3u];
        [select setBuffer:predictedEndpoint
            ? predictedSelectedVertices_ : selectedVertices_ offset:0u atIndex:4u];
        [select dispatchThreadgroups:MTLSizeMake(
             (positionCount + threadsPerGroup_ - 1u) / threadsPerGroup_, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
        [select endEncoding];

        if (predictedEndpoint) {
            const NSUInteger regionElements =
                static_cast<NSUInteger>(regionCount_) * environmentCount_;
            id<MTLComputeCommandEncoder> choose = timedSupportEncoder(
                command, device_, "support_choose_endpoint_witness",
                pass.stepIndex);
            if (choose == nil) return reject("encoder:choose_endpoint");
            choose.label = @"Numi Human retain current violations or select predicted endpoint";
            [choose setComputePipelineState:chooseEndpointPipeline_];
            [choose setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [choose setBuffer:currentSelectedVertices_ offset:0u atIndex:1u];
            [choose setBuffer:worldZKeys_ offset:0u atIndex:2u];
            [choose setBuffer:predictedSelectedVertices_ offset:0u atIndex:3u];
            [choose setBuffer:selectedVertices_ offset:0u atIndex:4u];
            [choose setBuffer:predictedWorldZKeys_ offset:0u atIndex:5u];
            [choose dispatchThreadgroups:MTLSizeMake(
                 (regionElements + threadsPerGroup_ - 1u) / threadsPerGroup_,
                 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup_, 1u, 1u)];
            [choose endEncoding];
        }

        id<MTLComputeCommandEncoder> publish = timedSupportEncoder(
            command, device_, "support_publish", pass.stepIndex);
        if (publish == nil) return reject("encoder:publish");
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
        [publish setBuffer:lineage_ offset:0u atIndex:18u];
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
            !predictedEndpoint && diagnosticCapture_ && pass.stepIndex == 0u;
        const bool motionObserverSample =
            !predictedEndpoint && motionObserverCapture_ &&
            pass.stepIndex % motionObserverCadenceSteps_ ==
                motionObserverCadenceSteps_ - 1u;
        if (firstStepDiagnostic || motionObserverSample) {
            id<MTLComputeCommandEncoder> debug =
                [command computeCommandEncoder];
            if (debug == nil) return reject("encoder:debug");
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
        std::span<const MRHumanRestingSupportLineageGPU> lineage,
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
                    regions.size() == lineage.size() &&
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
                const auto role = regions[region].reserved0;
                if (role == MR_HUMAN_RESTING_SUPPORT_REGION_ORGAN_FLOOR ||
                    role == MR_HUMAN_RESTING_SUPPORT_REGION_DERIVED_ORGAN_FLOOR) {
                    require(map.influenceCount == 1u && influence.body.x == contacts[region].bodyIndex &&
                                influence.positionAndWeight.w == 1.0f,
                            "stable anatomy ID 4 support vertices must use one exact dynamic rigid-body influence");
                }
            }
            ++regionCounts[region];
        }
        // Role 1 reserves a source query for a moving boundary and owns no
        // floor vertices. Role 2 is the four-point foot manifold. Roles 3/4
        // are a registered-organ four-bin floor manifold with separate typed
        // source lineage; every selected row must own actual source vertices.
        std::vector<std::array<bool, 4u>> organBins(regions.size());
        for (std::size_t r = 0u; r < regions.size(); ++r) {
            const auto role = regions[r].reserved0;
            const auto& source = lineage[r];
            require(role <= MR_HUMAN_RESTING_SUPPORT_REGION_DERIVED_ORGAN_FLOOR &&
                        (role == MR_HUMAN_RESTING_SUPPORT_REGION_BOUNDARY_ONLY
                            ? regionCounts[r] == 0u : regionCounts[r] > 0u),
                    "support boundary/floor region ownership is inconsistent");
            if (role == MR_HUMAN_RESTING_SUPPORT_REGION_DERIVED_FLOOR) {
                const auto parent = regions[r].reserved1;
                require(source.kind == MR_HUMAN_RESTING_SUPPORT_LINEAGE_NONE &&
                            source.sourceStableId == 0u && source.binIndex == MR_INVALID_INDEX &&
                            (parent == 20u || parent == 30u) && parent < r &&
                            regions[parent].reserved0 == MR_HUMAN_RESTING_SUPPORT_REGION_FLOOR &&
                            regions[parent].sourceGeometryIndex ==
                                regions[r].sourceGeometryIndex &&
                            contacts[parent].sourceGeometryIndex ==
                                contacts[r].sourceGeometryIndex &&
                            contacts[parent].bodyIndex == contacts[r].bodyIndex,
                        "derived foot support region lost its source-row lineage");
            } else if (role == MR_HUMAN_RESTING_SUPPORT_REGION_ORGAN_FLOOR ||
                       role == MR_HUMAN_RESTING_SUPPORT_REGION_DERIVED_ORGAN_FLOOR) {
                const auto parent = source.parentRegion;
                const bool isBase = role == MR_HUMAN_RESTING_SUPPORT_REGION_ORGAN_FLOOR;
                require(source.kind == MR_HUMAN_RESTING_SUPPORT_LINEAGE_ORGAN_SURFACE &&
                            source.sourceStableId == 4u && source.binIndex < 4u &&
                            parent < regions.size() && parent <= r &&
                            source.binIndex < 4u &&
                            (isBase
                                ? (parent == r && regions[r].reserved1 == 0u)
                                : (parent < r && regions[r].reserved1 == parent &&
                                   regions[parent].reserved0 == MR_HUMAN_RESTING_SUPPORT_REGION_ORGAN_FLOOR)) &&
                            regions[parent].sourceGeometryIndex == regions[r].sourceGeometryIndex &&
                            contacts[parent].sourceGeometryIndex == contacts[r].sourceGeometryIndex &&
                            contacts[parent].bodyIndex == contacts[r].bodyIndex &&
                            !organBins[parent][source.binIndex],
                        "organ floor row lost its registered stable-ID/bin lineage");
                organBins[parent][source.binIndex] = true;
            } else {
                require(source.kind == MR_HUMAN_RESTING_SUPPORT_LINEAGE_NONE &&
                            source.sourceStableId == 0u && source.binIndex == MR_INVALID_INDEX &&
                            regions[r].reserved1 == 0u,
                        "base support region has unexpected derived lineage");
            }
        }
        for (std::size_t r = 0u; r < regions.size(); ++r) {
            if (regions[r].reserved0 == MR_HUMAN_RESTING_SUPPORT_REGION_ORGAN_FLOOR)
                require(std::all_of(organBins[r].begin(), organBins[r].end(),
                            [](bool present) { return present; }),
                        "registered organ floor requires all four nonempty spatial bins");
        }
        for (std::size_t region = 0u; region < regions.size(); ++region) {
            require(regions[region].pointQueryIndex == contacts[region].pointQueryIndex &&
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
        lineage_ = [device_ newBufferWithBytes:lineage.data()
            length:lineage.size_bytes() options:MTLResourceStorageModeShared];
        require(regionForVertex_ != nil && regions_ != nil && lineage_ != nil,
                "resting support region and lineage buffers could not be uploaded");
        // The specialized minimum pass may omit only the static influence
        // guards proven above. Its dynamic body-pose/basis checks and all
        // invalid-region publication remain in the shader.
        positionsPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_positions", true, true,
            influenceLayout_, false, bed_.counts.z!=0u);
        predictedPositionsPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_positions", true, true,
            influenceLayout_, false, false, true);
        probeVelocitiesPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_probe_velocities");
        chooseEndpointPipeline_ = makePipeline(device_, library,
            @"nm_human_resting_support_choose_endpoint_witness");
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
                std::min(predictedPositionsPipeline_.maxTotalThreadsPerThreadgroup,
                std::min(probeVelocitiesPipeline_.maxTotalThreadsPerThreadgroup,
                std::min(chooseEndpointPipeline_.maxTotalThreadsPerThreadgroup,
                std::min(positionsPipeline_.maxTotalThreadsPerThreadgroup,
                std::min(selectPipeline_.maxTotalThreadsPerThreadgroup,
                         publishPipeline_.maxTotalThreadsPerThreadgroup)))))));
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
        predictedWorldZKeys_ = [device_ newBufferWithLength:
            static_cast<NSUInteger>(environmentCount_) * vertexCount_ *
                sizeof(std::uint32_t)
            options:MTLResourceStorageModePrivate];
        predictedMinimumHeightKeys_ = [device_ newBufferWithLength:regionBytes
            options:MTLResourceStorageModePrivate];
        predictedSelectedVertices_ = [device_ newBufferWithLength:regionBytes
            options:MTLResourceStorageModePrivate];
        currentSelectedVertices_ = [device_ newBufferWithLength:regionBytes
            options:MTLResourceStorageModePrivate];
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
        require(worldZKeys_ != nil && predictedWorldZKeys_ != nil &&
                    minimumHeightKeys_ != nil &&
                    predictedMinimumHeightKeys_ != nil &&
                    predictedSelectedVertices_ != nil &&
                    currentSelectedVertices_ != nil &&
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
