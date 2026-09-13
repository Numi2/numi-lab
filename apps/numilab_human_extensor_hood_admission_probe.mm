#import <Metal/Metal.h>

#include "metalrobo/NumiHumanExtensorHoodMetal.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using metalrobo::MetalNumiHumanTendonLoadPass;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    std::vector<MRNumiHumanExtensorHoodRayGPU> rays;
    std::vector<MRNumiHumanExtensorHoodNodeGPU> nodes;
    std::vector<MRNumiHumanExtensorHoodElementGPU> elements;
    std::vector<MRNumiHumanExtensorHoodInputGPU> inputs;

    Fixture() {
        // Admission-only topology: no command buffer in this probe is committed.
        // These source indices satisfy the public owner's eight-ray contract.
        for (mr_u32 rayIndex = 0u; rayIndex < 8u; ++rayIndex) {
            const mr_u32 digit = 2u + rayIndex % 4u;
            const mr_u32 nodeCount = digit == 5u ? 12u : 10u;
            const mr_u32 elementCount = digit == 5u ? 14u : 12u;
            const mr_u32 inputCount = digit == 5u ? 5u : 4u;
            const mr_u32 nodeOffset = static_cast<mr_u32>(nodes.size());
            rays.push_back({
                {nodeOffset, nodeCount, rayIndex / 4u, digit},
                {static_cast<mr_u32>(elements.size()), elementCount,
                 static_cast<mr_u32>(inputs.size()), inputCount}});
            for (mr_u32 n = 0u; n < nodeCount; ++n) {
                nodes.push_back({0u,
                    n < 3u ? MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED : 0u,
                    n, 0u, {0.001f * n, 0.0f, 0.0f, 0.0f}});
            }
            for (mr_u32 e = 0u; e < elementCount; ++e) {
                elements.push_back({nodeOffset + e % nodeCount,
                    nodeOffset + (e + 1u) % nodeCount, 0u, 0u,
                    {0.001f, 1.0e8f, 1.0e-6f, 0.0f}});
            }
            for (mr_u32 i = 0u; i < inputCount; ++i) {
                inputs.push_back({nodeOffset + 3u + i, 0u, 0u, 0u, 1u,
                    0u, 0u, 0u, {0.0f, 0.0f, 0.0f, 0.0f}});
            }
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: numilab_human_extensor_hood_admission_probe <metallib>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal queue unavailable");
            constexpr MTLResourceOptions options = MTLResourceStorageModeShared;
            constexpr mr_u32 bodyCount = 64u;
            constexpr NSUInteger lowBytes = bodyCount * sizeof(mr_float4);
            const std::array<NSUInteger, 8u> consumedBytes{
                sizeof(MRMujocoMuscleGPU), sizeof(MRMujocoMuscleSiteGPU),
                sizeof(MRMujocoMuscleWrapGPU), 2u * sizeof(MRMujocoMuscleRouteNodeGPU),
                sizeof(MRMujocoMuscleResultGPU), 2u * sizeof(float),
                bodyCount * sizeof(MRArticulatedBodyPoseGPU),
                bodyCount * 4u * 3u * sizeof(float)};
            // Retain separate ordinary buffers for the valid controls.
            std::array<id<MTLBuffer>, 8u> buffers{};
            for (std::size_t i = 0u; i < buffers.size(); ++i) {
                buffers[i] = [device newBufferWithLength:consumedBytes[i] options:options];
                require(buffers[i] != nil, "ordinary buffer allocation failed");
                std::memset(buffers[i].contents, 0, buffers[i].length);
            }
            id<MTLBuffer> low = [device newBufferWithLength:lowBytes options:options];
            require(low != nil, "low buffer allocation failed");
            std::memset(low.contents, 0, low.length);
            // Retain a nonzero companion so immutability checks include hidden data.
            static_cast<mr_float4*>(low.contents)[0].x = 0x1p-28f;
            MetalNumiHumanTendonLoadPass base{};
            base.mujocoMuscles = (__bridge void*)buffers[0];
            base.mujocoSites = (__bridge void*)buffers[1];
            base.mujocoWraps = (__bridge void*)buffers[2];
            base.mujocoRouteNodes = (__bridge void*)buffers[3];
            base.mujocoResults = (__bridge void*)buffers[4];
            base.generalizedForces = (__bridge void*)buffers[5];
            base.bodyPoses = (__bridge void*)buffers[6];
            base.pointJacobians = (__bridge void*)buffers[7];
            base.bodyPositionLow = (__bridge void*)low;
            base.bodyPositionLowGPUAddress = low.gpuAddress;
            base.bodyPositionLowElementCount = bodyCount;
            base.environmentCount = base.muscleCount = base.siteCount = 1u;
            base.wrapCount = base.dofCount = base.generalizedForceStride = 1u;
            base.routeNodeCount = 2u;
            base.generalizedForceOffset = 1u;
            base.bodyPoseStride = bodyCount;
            base.pointJacobianStride = bodyCount * 4u * 3u;
            base.bodyJacobianPointOffset = 0u;
            Fixture fixture;
            metalrobo::NumiHumanExtensorHoodMetalAdapter owner;
            metalrobo::NumiHumanExtensorHoodMetalConfiguration configuration;
            configuration.metallib = argv[1];
            require(owner.initialize({fixture.rays, fixture.nodes, fixture.elements,
                                      fixture.inputs, 1u}, configuration),
                    "source fixture rejected");
            const auto program = owner.program();
            require(program.valid(), "public hood program unavailable");
            std::uint32_t controls = 0u;
            const auto check = [&](MetalNumiHumanTendonLoadPass pass,
                                   const bool expected, const char* reason,
                                   const char* label) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "command allocation failed");
                pass.commandBuffer = (__bridge void*)command;
                std::vector<std::uint8_t> before(lowBytes);
                std::memcpy(before.data(), low.contents, lowBytes);
                const auto diagnosticsBefore = owner.diagnostics();
                const bool encoded = program.encodePreDynamics(program.context, pass);
                const auto diagnosticsAfter = owner.diagnostics();
                require(encoded == expected, std::string(label) + ": " + diagnosticsAfter.message);
                if (!expected) {
                    require(diagnosticsAfter.message == reason,
                            std::string(label) + ": wrong rejection: " + diagnosticsAfter.message);
                    require(diagnosticsAfter.fingerprint == diagnosticsBefore.fingerprint &&
                            diagnosticsAfter.encodedPassCount == diagnosticsBefore.encodedPassCount &&
                            diagnosticsAfter.abortCount == diagnosticsBefore.abortCount,
                            std::string(label) + ": rejected pass changed owner state");
                } else {
                    program.abort(program.context, (__bridge void*)command);
                }
                require(command.status == MTLCommandBufferStatusNotEnqueued,
                        std::string(label) + ": admission committed GPU work");
                require(std::memcmp(before.data(), low.contents, lowBytes) == 0,
                        std::string(label) + ": admission changed low bytes");
                ++controls;
            };
            constexpr const char* authority = "hood paired body-position authority is invalid";
            constexpr const char* extent = "hood borrowed buffer extent is invalid";
            constexpr const char* alias = "hood paired body-position range overlaps borrowed arena";
            check(base, true, "", "disjoint paired");
            auto mutation = base;
            mutation.bodyPositionLow = nullptr;
            mutation.bodyPositionLowGPUAddress = 0u;
            mutation.bodyPositionLowElementCount = 0u;
            check(mutation, true, "", "legacy null companion");
            mutation.bodyPositionLowElementCount = bodyCount;
            check(mutation, false, authority, "null with count");
            mutation.bodyPositionLowElementCount = 0u;
            mutation.bodyPositionLowGPUAddress = low.gpuAddress;
            check(mutation, false, authority, "null with address");
            mutation = base;
            ++mutation.bodyPositionLowGPUAddress;
            check(mutation, false, authority, "wrong low address");
            mutation = base;
            --mutation.bodyPositionLowElementCount;
            check(mutation, false, authority, "wrong low count");
            id<MTLBuffer> shortLow = [device newBufferWithLength:lowBytes - 1u options:options];
            require(shortLow != nil, "short low allocation failed");
            mutation = base;
            mutation.bodyPositionLow = (__bridge void*)shortLow;
            mutation.bodyPositionLowGPUAddress = shortLow.gpuAddress;
            check(mutation, false, authority, "short low bytes");
            mutation = base;
            mutation.dofCount = mutation.muscleCount = std::numeric_limits<mr_u32>::max();
            mutation.generalizedForceStride = mutation.dofCount;
            check(mutation, false, extent, "overflowed consumed dimensions");
            mutation = base;
            mutation.bodyJacobianPointOffset = std::numeric_limits<mr_u32>::max() - 1u;
            check(mutation, false, extent, "overflowed body Jacobian span");

            using Field = void* MetalNumiHumanTendonLoadPass::*;
            const std::array<Field, 8u> fields{
                &MetalNumiHumanTendonLoadPass::mujocoMuscles,
                &MetalNumiHumanTendonLoadPass::mujocoSites,
                &MetalNumiHumanTendonLoadPass::mujocoWraps,
                &MetalNumiHumanTendonLoadPass::mujocoRouteNodes,
                &MetalNumiHumanTendonLoadPass::mujocoResults,
                &MetalNumiHumanTendonLoadPass::generalizedForces,
                &MetalNumiHumanTendonLoadPass::bodyPoses,
                &MetalNumiHumanTendonLoadPass::pointJacobians};
            for (std::size_t i = 0u; i < fields.size(); ++i) {
                id<MTLBuffer> shortBuffer = [device newBufferWithLength:consumedBytes[i] - 1u options:options];
                require(shortBuffer != nil, "short borrowed allocation failed");
                mutation = base;
                mutation.*fields[i] = (__bridge void*)shortBuffer;
                check(mutation, false, extent, "short borrowed arena");
                const auto alignment = [device heapBufferSizeAndAlignWithLength:
                    std::max(lowBytes, consumedBytes[i]) options:options];
                const NSUInteger partialOffset = alignment.align;
                require(partialOffset < lowBytes, "heap alignment cannot exercise partial overlap");
                MTLHeapDescriptor* descriptor = [MTLHeapDescriptor new];
                descriptor.type = MTLHeapTypePlacement;
                descriptor.storageMode = MTLStorageModeShared;
                descriptor.size = 4u * (alignment.size + lowBytes);
                id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
                require(heap != nil, "placement heap unavailable");
                id<MTLBuffer> heapLow = [heap newBufferWithLength:lowBytes options:options offset:0u];
                require(heapLow != nil, "heap low unavailable");
                [heapLow makeAliasable];
                for (const NSUInteger offset : {NSUInteger{0u}, partialOffset}) {
                    id<MTLBuffer> overlapping = [heap newBufferWithLength:consumedBytes[i]
                        options:options offset:offset];
                    require(overlapping != nil && overlapping != heapLow &&
                            overlapping.gpuAddress == heapLow.gpuAddress + offset,
                            "control did not create distinct overlapping GPU buffers");
                    mutation = base;
                    mutation.bodyPositionLow = (__bridge void*)heapLow;
                    mutation.bodyPositionLowGPUAddress = heapLow.gpuAddress;
                    mutation.*fields[i] = (__bridge void*)overlapping;
                    check(mutation, false, alias, "distinct heap overlap");
                    [overlapping makeAliasable];
                }
                // Identical object pointers are still rejected by the range rule.
                if (lowBytes >= consumedBytes[i]) {
                    mutation = base;
                    mutation.*fields[i] = (__bridge void*)low;
                    check(mutation, false, alias, "same object overlap");
                }
            }
            // Half-open consumed intervals may meet at an endpoint. This is a
            // distinct-buffer control on one heap, with no aliasable resource.
            MTLHeapDescriptor* adjacentDescriptor = [MTLHeapDescriptor new];
            adjacentDescriptor.type = MTLHeapTypePlacement;
            adjacentDescriptor.storageMode = MTLStorageModeShared;
            adjacentDescriptor.size = 4u * lowBytes;
            id<MTLHeap> adjacentHeap = [device newHeapWithDescriptor:adjacentDescriptor];
            require(adjacentHeap != nil, "adjacent heap unavailable");
            id<MTLBuffer> adjacentLow = [adjacentHeap newBufferWithLength:lowBytes
                options:options offset:0u];
            id<MTLBuffer> adjacentOutput = [adjacentHeap newBufferWithLength:consumedBytes[5]
                options:options offset:lowBytes];
            require(adjacentLow != nil && adjacentOutput != nil &&
                    adjacentOutput.gpuAddress == adjacentLow.gpuAddress + lowBytes,
                    "control did not create adjacent GPU buffers");
            mutation = base;
            mutation.bodyPositionLow = (__bridge void*)adjacentLow;
            mutation.bodyPositionLowGPUAddress = adjacentLow.gpuAddress;
            mutation.generalizedForces = (__bridge void*)adjacentOutput;
            check(mutation, true, "", "adjacent consumed GPU ranges");
            check(base, true, "", "paired after rejected aliases");
            std::cout << "numi_human_extensor_hood_admission=passed controls=" << controls
                      << " heap_overlap_controls=16 gpu_commands_committed=0\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "numi_human_extensor_hood_admission=failed message=" << error.what() << '\n';
            return 1;
        }
    }
}
