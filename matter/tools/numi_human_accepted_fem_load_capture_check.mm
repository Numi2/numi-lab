#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error NUMI_MATTER_METALLIB must name the Matter metallib
#endif

namespace {

id<MTLBuffer> buffer(id<MTLDevice> device, const void* data, NSUInteger bytes) {
    if (data == nullptr || bytes == 0u) return nil;
    return [device newBufferWithBytes:data length:bytes
                              options:MTLResourceStorageModeShared];
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSString* qualified = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:qualified];
    if (function == nil)
        throw std::runtime_error(std::string("missing kernel: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (result == nil)
        throw std::runtime_error(error.localizedDescription.UTF8String);
    return result;
}

void runPrepare(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
    NMNumiHumanTendonFEMLoadDispatchGPU dispatch,
    id<MTLBuffer> wrench, id<MTLBuffer> poses, id<MTLBuffer> jacobians,
    id<MTLBuffer> q, id<MTLBuffer> pendingLoad, id<MTLBuffer> pendingQ) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil)
        throw std::runtime_error("prepare command allocation failed");
    [encoder setComputePipelineState:pso];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBuffer:wrench offset:0u atIndex:1u];
    [encoder setBuffer:poses offset:0u atIndex:2u];
    [encoder setBuffer:jacobians offset:0u atIndex:3u];
    [encoder setBuffer:q offset:0u atIndex:4u];
    [encoder setBuffer:pendingLoad offset:0u atIndex:5u];
    [encoder setBuffer:pendingQ offset:0u atIndex:6u];
    const NSUInteger count = std::max<NSUInteger>(
        dispatch.dofCount, dispatch.reserved0);
    const NSUInteger width = std::min<NSUInteger>(
        count, pso.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(std::max<NSUInteger>(width, 1u), 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("prepare command failed");
}

void runCommit(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
    NMNumiHumanTendonFEMLoadDispatchGPU dispatch,
    id<MTLBuffer> statuses, id<MTLBuffer> pendingLoad, id<MTLBuffer> pendingQ,
    id<MTLBuffer> outputLoad, id<MTLBuffer> outputQ) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil)
        throw std::runtime_error("commit command allocation failed");
    [encoder setComputePipelineState:pso];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBuffer:statuses offset:0u atIndex:1u];
    [encoder setBuffer:pendingLoad offset:0u atIndex:2u];
    [encoder setBuffer:pendingQ offset:0u atIndex:3u];
    [encoder setBuffer:outputLoad offset:0u atIndex:4u];
    [encoder setBuffer:outputQ offset:0u atIndex:5u];
    const NSUInteger count = std::max<NSUInteger>(
        dispatch.dofCount, dispatch.reserved0);
    const NSUInteger width = std::min<NSUInteger>(
        count, pso.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(std::max<NSUInteger>(width, 1u), 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("commit command failed");
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            NSString* path = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            if (library == nil)
                throw std::runtime_error(error.localizedDescription.UTF8String);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");
            auto prepare = pipeline(
                device, library,
                "numi_matter_metal::nm_numi_human_prepare_fem_body_generalized_load");
            auto commit = pipeline(
                device, library,
                "numi_matter_metal::nm_numi_human_commit_fem_body_generalized_load");

            NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
            dispatch.abiVersion = NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
            dispatch.environmentCount = 1u;
            dispatch.stepIndex = 4u;
            dispatch.dofCount = 1u;
            dispatch.bodyPoseStride = 1u;
            dispatch.pointJacobianStride = 12u;
            dispatch.bodyJacobianPointOffset = 0u;
            dispatch.reserved0 = 3u;

            NMNumiHumanBodyWrenchGPU wrenchValue{};
            wrenchValue.force.x = 2.0f;
            wrenchValue.force.y = 3.0f;
            wrenchValue.force.z = 4.0f;
            MRArticulatedBodyPoseGPU poseValue{};
            poseValue.position.w = 1.0f;
            poseValue.orientation.w = 1.0f;
            std::array<float, 12u> jacobian{};
            jacobian[0u] = 1.0f;
            jacobian[1u] = 2.0f;
            jacobian[2u] = 3.0f;
            std::array<float, 3u> sourceQ{10.0f, 20.0f, 30.0f};
            std::array<float, 1u> pendingLoad{};
            std::array<float, 3u> pendingQ{};
            std::array<float, 1u> outputLoad{99.0f};
            std::array<float, 3u> outputQ{99.0f, 99.0f, 99.0f};
            MRNumiHumanStandStatusGPU status{};
            status.code = MR_NUMI_HUMAN_STAND_SUCCESS;
            status.environment = 0u;
            status.completedSteps = dispatch.stepIndex + 1u;
            auto wrench = buffer(device, &wrenchValue, sizeof(wrenchValue));
            auto poses = buffer(device, &poseValue, sizeof(poseValue));
            auto jacobians = buffer(device, jacobian.data(), sizeof(jacobian));
            auto q = buffer(device, sourceQ.data(), sizeof(sourceQ));
            auto pendingLoadBuffer = buffer(device, pendingLoad.data(),
                                            sizeof(pendingLoad));
            auto pendingQBuffer = buffer(device, pendingQ.data(), sizeof(pendingQ));
            auto outputLoadBuffer = buffer(device, outputLoad.data(),
                                           sizeof(outputLoad));
            auto outputQBuffer = buffer(device, outputQ.data(), sizeof(outputQ));
            auto statusBuffer = buffer(device, &status, sizeof(status));
            require(wrench && poses && jacobians && q && pendingLoadBuffer &&
                    pendingQBuffer && outputLoadBuffer && outputQBuffer &&
                    statusBuffer, "Metal fixture buffer allocation failed");

            runPrepare(queue, prepare, dispatch, wrench, poses, jacobians, q,
                       pendingLoadBuffer, pendingQBuffer);
            // Simulate later articulated integration mutating every source
            // view. The accepted capture must publish the pre-step pair.
            static_cast<NMNumiHumanBodyWrenchGPU*>(wrench.contents)->force.x = 50.0f;
            static_cast<MRArticulatedBodyPoseGPU*>(poses.contents)->position.x = 4.0f;
            static_cast<float*>(jacobians.contents)[0u] = 50.0f;
            static_cast<float*>(q.contents)[0u] = 500.0f;
            runCommit(queue, commit, dispatch, statusBuffer, pendingLoadBuffer,
                      pendingQBuffer, outputLoadBuffer, outputQBuffer);
            const float acceptedLoad =
                static_cast<const float*>(outputLoadBuffer.contents)[0u];
            require(std::abs(acceptedLoad - 20.0f) <= 1.0e-5f,
                    "accepted commit re-read mutated force/pose/J instead of pending load");
            const auto* acceptedQ =
                static_cast<const float*>(outputQBuffer.contents);
            require(acceptedQ[0u] == 10.0f && acceptedQ[1u] == 20.0f &&
                    acceptedQ[2u] == 30.0f,
                    "accepted commit did not publish matching source q snapshot");

            // A stale completion count must zero both outputs.
            dispatch.stepIndex = 5u;
            status.completedSteps = 5u; // expected is 6
            static_cast<MRNumiHumanStandStatusGPU*>(statusBuffer.contents)[0u] =
                status;
            outputLoad = {77.0f};
            outputQ = {77.0f, 77.0f, 77.0f};
            std::memcpy(outputLoadBuffer.contents, outputLoad.data(),
                        sizeof(outputLoad));
            std::memcpy(outputQBuffer.contents, outputQ.data(), sizeof(outputQ));
            runCommit(queue, commit, dispatch, statusBuffer, pendingLoadBuffer,
                      pendingQBuffer, outputLoadBuffer, outputQBuffer);
            require(static_cast<const float*>(outputLoadBuffer.contents)[0u] == 0.0f,
                    "rejected transaction leaked its load snapshot");
            acceptedQ = static_cast<const float*>(outputQBuffer.contents);
            require(acceptedQ[0u] == 0.0f && acceptedQ[1u] == 0.0f &&
                    acceptedQ[2u] == 0.0f,
                    "rejected transaction leaked its source q snapshot");
            std::cout << "accepted FEM body-load snapshot/commit PASS\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "accepted FEM body-load snapshot/commit FAIL: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
