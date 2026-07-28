#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/Collision.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace {

constexpr std::uint8_t kOutputSentinel = 0xa5u;
constexpr std::uint32_t kEnvironment = 19u;

struct Scene {
    std::vector<MRBodyStateGPU> bodies;
    std::vector<MRShapeGPU> shapes;
    std::vector<metalrobo::CollisionPairExclusion> exclusions;
    std::vector<MRCandidatePairGPU> gpuExclusions;
};

struct MetalRun {
    MRBroadphaseStatusGPU status{};
    std::vector<MRCandidatePairGPU> pairs;
    bool outputUntouched = false;
    bool unusedOutputUntouched = false;
    std::string deviceName;
};

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRBodyStateGPU makeBody(
    const float x,
    const float y,
    const float z,
    const std::uint32_t motion
) {
    MRBodyStateGPU body{};
    body.position = f4(x, y, z, 1.0f);
    body.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    body.flagsAndIndices[0] = motion;
    return body;
}

MRShapeGPU makeShape(
    const std::uint32_t body,
    const std::uint32_t type,
    const mr_float4 dimensions,
    const float boundingRadius
) {
    MRShapeGPU shape{};
    shape.bodyIndex = body;
    shape.shapeType = type;
    shape.collisionGroup = 1u;
    shape.collisionMask = 1u;
    shape.slotGeneration = 5000u + body;
    shape.localPosition = f4(0.0f, 0.0f, 0.0f, 1.0f);
    shape.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    shape.dimensions = dimensions;
    shape.contactRestAndBoundingRadius =
        f4(0.02f, 0.0f, boundingRadius, 0.0f);
    return shape;
}

Scene makeScene() {
    Scene scene;
    scene.bodies.reserve(50u);
    scene.shapes.reserve(50u);

    scene.bodies.push_back(
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC)
    );
    scene.shapes.push_back(
        makeShape(
            0u,
            MR_SHAPE_PLANE,
            f4(0.0f, 0.0f, 0.0f, 0.0f),
            0.0f
        )
    );

    for (std::uint32_t index = 0u; index < 46u; ++index) {
        float x = 0.8f * static_cast<float>(index % 8u);
        const float z =
            0.8f * static_cast<float>(index / 8u);
        if (index == 1u) {
            // Deliberately overlaps the preceding sphere so the stream
            // contains both finite/finite and finite/plane candidates.
            x = 0.45f;
        }
        const std::uint32_t bodyIndex =
            static_cast<std::uint32_t>(scene.bodies.size());
        scene.bodies.push_back(
            makeBody(x, 0.24f, z, MR_MOTION_DYNAMIC)
        );
        scene.shapes.push_back(
            makeShape(
                bodyIndex,
                MR_SHAPE_SPHERE,
                f4(0.25f, 0.0f, 0.0f, 0.0f),
                0.25f
            )
        );
    }

    const std::uint32_t capsuleBody =
        static_cast<std::uint32_t>(scene.bodies.size());
    scene.bodies.push_back(
        makeBody(7.0f, 0.25f, 0.0f, MR_MOTION_DYNAMIC)
    );
    scene.shapes.push_back(
        makeShape(
            capsuleBody,
            MR_SHAPE_CAPSULE,
            f4(0.15f, 0.30f, 0.0f, 0.0f),
            0.45f
        )
    );

    const std::uint32_t boxBody =
        static_cast<std::uint32_t>(scene.bodies.size());
    scene.bodies.push_back(
        makeBody(8.0f, 0.20f, 0.0f, MR_MOTION_DYNAMIC)
    );
    scene.shapes.push_back(
        makeShape(
            boxBody,
            MR_SHAPE_BOX,
            f4(0.20f, 0.20f, 0.20f, 0.0f),
            std::sqrt(3.0f * 0.20f * 0.20f)
        )
    );

    const std::uint32_t disabledBody =
        static_cast<std::uint32_t>(scene.bodies.size());
    scene.bodies.push_back(
        makeBody(9.0f, 0.10f, 0.0f, MR_MOTION_DYNAMIC)
    );
    MRShapeGPU disabled = makeShape(
        disabledBody,
        MR_SHAPE_CYLINDER,
        f4(0.20f, 0.30f, 0.0f, 0.0f),
        0.50f
    );
    disabled.flags = MR_SHAPE_FLAG_SIMULATION_DISABLED;
    scene.shapes.push_back(disabled);

    // Exercise independent mask and explicit-pair suppression.
    scene.shapes[5].collisionMask = 0u;
    scene.exclusions.push_back({0u, 7u});
    scene.gpuExclusions.push_back(
        {kEnvironment, 0u, 7u, 0u}
    );

    require(
        scene.bodies.size() == 50u &&
            scene.shapes.size() == 50u,
        "broadphase probe scene size drifted"
    );
    return scene;
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
    const std::string localized =
        nsString(error.localizedDescription);
    return localized.empty()
        ? nsString(error.description)
        : localized;
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const T* data,
    const std::size_t count,
    NSString* label
) {
    require(
        count <=
            std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer byte count overflow"
    );
    require(
        count == 0u || data != nullptr,
        "non-empty Metal buffer has null source data"
    );
    id<MTLBuffer> buffer = nil;
    if (count == 0u) {
        buffer = [device
            newBufferWithLength:sizeof(T)
                        options:MTLResourceStorageModeShared];
        if (buffer != nil) {
            std::memset(buffer.contents, 0, sizeof(T));
        }
    } else {
        buffer = [device
            newBufferWithBytes:data
                        length:static_cast<NSUInteger>(
                            count * sizeof(T)
                        )
                       options:MTLResourceStorageModeShared];
    }
    require(
        buffer != nil,
        "failed to allocate Metal buffer '" + nsString(label) + "'"
    );
    buffer.label = label;
    return buffer;
}

template <typename T>
std::vector<T> sentinelVector(const std::size_t count) {
    std::vector<T> result(std::max<std::size_t>(count, 1u));
    std::memset(
        result.data(),
        kOutputSentinel,
        result.size() * sizeof(T)
    );
    return result;
}

template <typename T>
std::vector<T> zeroVector(const std::size_t count) {
    return std::vector<T>(std::max<std::size_t>(count, 1u));
}

template <typename T>
bool byteEqual(
    const std::span<const T> left,
    const std::span<const T> right
) {
    return left.size() == right.size() &&
        std::memcmp(
            left.data(),
            right.data(),
            left.size_bytes()
        ) == 0;
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    id<MTLFunction> function =
        [library newFunctionWithName:name];
    require(
        function != nil,
        "metallib does not contain " + nsString(name)
    );
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function
                                               error:&error];
    require(
        pipeline != nil,
        "failed to create " + nsString(name) +
            " pipeline: " + describeError(error)
    );
    return pipeline;
}

void barrier(id<MTLComputeCommandEncoder> encoder) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

MetalRun broadphaseOnMetal(
    const Scene& scene,
    const std::span<const MRAabbGPU> aabbs,
    const std::uint32_t pairCapacity
) {
    @autoreleasepool {
        const std::string metallibPath =
            METALROBO_DEFAULT_METALLIB;
        require(
            !metallibPath.empty(),
            "METALROBO_DEFAULT_METALLIB is empty"
        );
        require(
            aabbs.size() == scene.shapes.size(),
            "AABB count must equal shape count"
        );

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal-capable device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create Metal command queue");

        NSString* path =
            [NSString stringWithUTF8String:metallibPath.c_str()];
        require(path != nil, "metallib path is not valid UTF-8");
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:path]
                        error:&error];
        require(
            library != nil,
            "failed to load metallib: " + describeError(error)
        );

        id<MTLComputePipelineState> preflight =
            makePipeline(
                device,
                library,
                @"mr_broadphase_preflight"
            );
        id<MTLComputePipelineState> classify =
            makePipeline(
                device,
                library,
                @"mr_broadphase_classify_pairs"
            );
        id<MTLComputePipelineState> pairScan =
            makePipeline(
                device,
                library,
                @"mr_broadphase_scan_pair_blocks"
            );
        id<MTLComputePipelineState> blockScan =
            makePipeline(
                device,
                library,
                @"mr_broadphase_scan_block_sums"
            );
        id<MTLComputePipelineState> finalize =
            makePipeline(
                device,
                library,
                @"mr_broadphase_finalize_count"
            );
        id<MTLComputePipelineState> scatter =
            makePipeline(
                device,
                library,
                @"mr_broadphase_scatter_pairs"
            );

        constexpr NSUInteger threads =
            MR_BROADPHASE_SCAN_BLOCK_SIZE;
        require(
            pairScan.maxTotalThreadsPerThreadgroup >= threads &&
                blockScan.maxTotalThreadsPerThreadgroup >= threads &&
                scatter.maxTotalThreadsPerThreadgroup >= threads,
            "device cannot execute required deterministic scan width"
        );

        const std::uint32_t shapeCount =
            static_cast<std::uint32_t>(scene.shapes.size());
        const std::uint32_t logicalPairCount =
            shapeCount < 2u
            ? 0u
            : shapeCount * (shapeCount - 1u) / 2u;
        const std::uint32_t scanBlockCount =
            (
                logicalPairCount +
                MR_BROADPHASE_SCAN_BLOCK_SIZE -
                1u
            ) / MR_BROADPHASE_SCAN_BLOCK_SIZE;
        const MRBroadphaseDispatchGPU dispatch{
            shapeCount,
            static_cast<std::uint32_t>(scene.bodies.size()),
            logicalPairCount,
            scanBlockCount,
            pairCapacity,
            static_cast<std::uint32_t>(
                scene.gpuExclusions.size()
            ),
            kEnvironment,
            0u,
        };
        MRBroadphaseStatusGPU statusSeed{};
        std::memset(
            &statusSeed,
            kOutputSentinel,
            sizeof(statusSeed)
        );

        const auto pairFlags =
            zeroVector<std::uint32_t>(logicalPairCount);
        const auto logicalPairs =
            zeroVector<MRCandidatePairGPU>(logicalPairCount);
        const auto pairOffsets =
            zeroVector<std::uint32_t>(logicalPairCount);
        const auto blockSums =
            zeroVector<std::uint32_t>(scanBlockCount);
        const auto blockOffsets =
            zeroVector<std::uint32_t>(scanBlockCount);
        const auto outputSeed =
            sentinelVector<MRCandidatePairGPU>(pairCapacity);

        id<MTLBuffer> dispatchBuffer = makeBuffer(
            device, &dispatch, 1u, @"broadphase dispatch"
        );
        id<MTLBuffer> shapeBuffer = makeBuffer(
            device,
            scene.shapes.data(),
            scene.shapes.size(),
            @"broadphase shapes"
        );
        id<MTLBuffer> bodyBuffer = makeBuffer(
            device,
            scene.bodies.data(),
            scene.bodies.size(),
            @"broadphase bodies"
        );
        id<MTLBuffer> aabbBuffer = makeBuffer(
            device,
            aabbs.data(),
            aabbs.size(),
            @"broadphase AABBs"
        );
        id<MTLBuffer> exclusionBuffer = makeBuffer(
            device,
            scene.gpuExclusions.data(),
            scene.gpuExclusions.size(),
            @"broadphase exclusions"
        );
        id<MTLBuffer> statusBuffer = makeBuffer(
            device, &statusSeed, 1u, @"broadphase status"
        );
        id<MTLBuffer> flagBuffer = makeBuffer(
            device,
            pairFlags.data(),
            pairFlags.size(),
            @"broadphase flags"
        );
        id<MTLBuffer> logicalPairBuffer = makeBuffer(
            device,
            logicalPairs.data(),
            logicalPairs.size(),
            @"broadphase logical pairs"
        );
        id<MTLBuffer> pairOffsetBuffer = makeBuffer(
            device,
            pairOffsets.data(),
            pairOffsets.size(),
            @"broadphase pair offsets"
        );
        id<MTLBuffer> blockSumBuffer = makeBuffer(
            device,
            blockSums.data(),
            blockSums.size(),
            @"broadphase block sums"
        );
        id<MTLBuffer> blockOffsetBuffer = makeBuffer(
            device,
            blockOffsets.data(),
            blockOffsets.size(),
            @"broadphase block offsets"
        );
        id<MTLBuffer> outputBuffer = makeBuffer(
            device,
            outputSeed.data(),
            outputSeed.size(),
            @"broadphase output"
        );

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        require(
            commandBuffer != nil,
            "failed to create Metal command buffer"
        );
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        require(
            encoder != nil,
            "failed to create Metal compute encoder"
        );

        [encoder setComputePipelineState:preflight];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:shapeBuffer offset:0 atIndex:1];
        [encoder setBuffer:bodyBuffer offset:0 atIndex:2];
        [encoder setBuffer:aabbBuffer offset:0 atIndex:3];
        [encoder setBuffer:exclusionBuffer offset:0 atIndex:4];
        [encoder setBuffer:statusBuffer offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        barrier(encoder);

        if (logicalPairCount > 0u) {
            [encoder setComputePipelineState:classify];
            [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:shapeBuffer offset:0 atIndex:1];
            [encoder setBuffer:bodyBuffer offset:0 atIndex:2];
            [encoder setBuffer:aabbBuffer offset:0 atIndex:3];
            [encoder setBuffer:exclusionBuffer offset:0 atIndex:4];
            [encoder setBuffer:statusBuffer offset:0 atIndex:5];
            [encoder setBuffer:flagBuffer offset:0 atIndex:6];
            [encoder setBuffer:logicalPairBuffer offset:0 atIndex:7];
            [encoder dispatchThreads:
                MTLSizeMake(logicalPairCount, 1u, 1u)
                threadsPerThreadgroup:
                    MTLSizeMake(threads, 1u, 1u)];
            barrier(encoder);

            [encoder setComputePipelineState:pairScan];
            [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:statusBuffer offset:0 atIndex:1];
            [encoder setBuffer:flagBuffer offset:0 atIndex:2];
            [encoder setBuffer:pairOffsetBuffer offset:0 atIndex:3];
            [encoder setBuffer:blockSumBuffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:
                MTLSizeMake(scanBlockCount, 1u, 1u)
                threadsPerThreadgroup:
                    MTLSizeMake(threads, 1u, 1u)];
            barrier(encoder);
        }

        [encoder setComputePipelineState:blockScan];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:statusBuffer offset:0 atIndex:1];
        [encoder setBuffer:blockSumBuffer offset:0 atIndex:2];
        [encoder setBuffer:blockOffsetBuffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        barrier(encoder);

        [encoder setComputePipelineState:finalize];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockSumBuffer offset:0 atIndex:1];
        [encoder setBuffer:blockOffsetBuffer offset:0 atIndex:2];
        [encoder setBuffer:statusBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        barrier(encoder);

        if (logicalPairCount > 0u) {
            [encoder setComputePipelineState:scatter];
            [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:statusBuffer offset:0 atIndex:1];
            [encoder setBuffer:flagBuffer offset:0 atIndex:2];
            [encoder setBuffer:pairOffsetBuffer offset:0 atIndex:3];
            [encoder setBuffer:blockOffsetBuffer offset:0 atIndex:4];
            [encoder setBuffer:logicalPairBuffer offset:0 atIndex:5];
            [encoder setBuffer:outputBuffer offset:0 atIndex:6];
            [encoder dispatchThreads:
                MTLSizeMake(logicalPairCount, 1u, 1u)
                threadsPerThreadgroup:
                    MTLSizeMake(threads, 1u, 1u)];
        }
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(
            commandBuffer.status ==
                MTLCommandBufferStatusCompleted,
            "broadphase dispatch failed: " +
                describeError(commandBuffer.error)
        );

        MetalRun result;
        std::memcpy(
            &result.status,
            statusBuffer.contents,
            sizeof(result.status)
        );
        const auto* output = static_cast<const MRCandidatePairGPU*>(
            outputBuffer.contents
        );
        const std::size_t emitted = std::min<std::size_t>(
            result.status.emittedPairs,
            outputSeed.size()
        );
        result.pairs.assign(output, output + emitted);
        result.outputUntouched =
            std::memcmp(
                output,
                outputSeed.data(),
                outputSeed.size() *
                    sizeof(MRCandidatePairGPU)
            ) == 0;
        if (emitted <= outputSeed.size()) {
            result.unusedOutputUntouched =
                std::memcmp(
                    output + emitted,
                    outputSeed.data() + emitted,
                    (outputSeed.size() - emitted) *
                        sizeof(MRCandidatePairGPU)
                ) == 0;
        }
        result.deviceName = nsString(device.name);
        return result;
    }
}

bool pairEqual(
    const MRCandidatePairGPU& left,
    const MRCandidatePairGPU& right
) {
    return
        left.environment == right.environment &&
        left.colliderA == right.colliderA &&
        left.colliderB == right.colliderB &&
        left.flags == right.flags;
}

} // namespace

int main() {
    try {
        metalrobo::CollisionConfig zeroPairConfig;
        zeroPairConfig.environment = kEnvironment;
        metalrobo::PersistentManifoldCache emptyCache;
        const Scene emptyScene;
        const metalrobo::CollisionFrame emptyCpu =
            metalrobo::collideCpuReference(
                emptyScene.shapes,
                emptyScene.bodies,
                zeroPairConfig,
                emptyCache,
                emptyScene.exclusions
            );
        require(
            emptyCpu.succeeded() &&
                emptyCpu.pairs.empty() &&
                emptyCpu.worldAabbs.empty(),
            "CPU oracle rejected an empty environment"
        );
        const MetalRun emptyMetal = broadphaseOnMetal(
            emptyScene,
            emptyCpu.worldAabbs,
            0u
        );
        require(
            emptyMetal.status.code == MR_STEP_SUCCESS &&
                emptyMetal.status.logicalPairs == 0u &&
                emptyMetal.status.requiredPairs == 0u &&
                emptyMetal.status.emittedPairs == 0u &&
                emptyMetal.outputUntouched,
            "Metal broadphase rejected an empty environment"
        );

        Scene singletonScene;
        singletonScene.bodies.push_back(
            makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC)
        );
        singletonScene.shapes.push_back(
            makeShape(
                0u,
                MR_SHAPE_PLANE,
                f4(0.0f, 0.0f, 0.0f, 0.0f),
                0.0f
            )
        );
        metalrobo::PersistentManifoldCache singletonCache;
        const metalrobo::CollisionFrame singletonCpu =
            metalrobo::collideCpuReference(
                singletonScene.shapes,
                singletonScene.bodies,
                zeroPairConfig,
                singletonCache,
                singletonScene.exclusions
            );
        require(
            singletonCpu.succeeded() &&
                singletonCpu.pairs.empty() &&
                singletonCpu.worldAabbs.size() == 1u,
            "CPU oracle rejected a one-shape environment"
        );
        const MetalRun singletonMetal = broadphaseOnMetal(
            singletonScene,
            singletonCpu.worldAabbs,
            0u
        );
        require(
            singletonMetal.status.code == MR_STEP_SUCCESS &&
                singletonMetal.status.logicalPairs == 0u &&
                singletonMetal.status.requiredPairs == 0u &&
                singletonMetal.status.emittedPairs == 0u &&
                singletonMetal.outputUntouched,
            "Metal broadphase rejected a one-shape environment"
        );

        const Scene scene = makeScene();
        const std::uint32_t shapeCount =
            static_cast<std::uint32_t>(scene.shapes.size());
        const std::uint32_t logicalPairCount =
            shapeCount * (shapeCount - 1u) / 2u;
        const std::uint32_t scanBlockCount =
            (
                logicalPairCount +
                MR_BROADPHASE_SCAN_BLOCK_SIZE -
                1u
            ) / MR_BROADPHASE_SCAN_BLOCK_SIZE;
        require(
            scanBlockCount > 1u,
            "probe must exercise the inter-block scan"
        );

        metalrobo::CollisionConfig config;
        config.environment = kEnvironment;
        config.capacities.pairCapacity = logicalPairCount;
        config.capacities.rawContactCapacity = 4096u;
        config.capacities.manifoldCapacity = logicalPairCount;
        metalrobo::PersistentManifoldCache cache;
        const metalrobo::CollisionFrame cpu =
            metalrobo::collideCpuReference(
                scene.shapes,
                scene.bodies,
                config,
                cache,
                scene.exclusions
            );
        require(
            cpu.succeeded(),
            "FP64 CPU collision oracle rejected probe scene"
        );
        require(
            !cpu.pairs.empty(),
            "probe scene produced no candidate pairs"
        );

        const MetalRun first = broadphaseOnMetal(
            scene,
            cpu.worldAabbs,
            logicalPairCount
        );
        require(
            first.status.code == MR_STEP_SUCCESS,
            "Metal broadphase failed a valid scene"
        );
        require(
            first.status.logicalPairs == logicalPairCount,
            "Metal logical-pair count drifted"
        );
        require(
            first.status.requiredPairs == cpu.pairs.size() &&
                first.status.emittedPairs == cpu.pairs.size(),
            "Metal count/scan result disagrees with CPU oracle"
        );
        require(
            first.pairs.size() == cpu.pairs.size(),
            "Metal candidate stream size disagrees with CPU oracle"
        );
        require(
            std::ranges::equal(
                first.pairs,
                cpu.pairs,
                pairEqual
            ),
            "Metal candidate stream disagrees with CPU oracle"
        );
        require(
            first.unusedOutputUntouched,
            "Metal scatter wrote outside its exact output range"
        );

        const MetalRun replay = broadphaseOnMetal(
            scene,
            cpu.worldAabbs,
            logicalPairCount
        );
        require(
            std::memcmp(
                &first.status,
                &replay.status,
                sizeof(first.status)
            ) == 0 &&
                byteEqual<MRCandidatePairGPU>(
                    first.pairs,
                    replay.pairs
                ),
            "Metal broadphase replay was not bit deterministic"
        );

        const std::uint32_t exactCapacity =
            first.status.requiredPairs;
        const MetalRun exact = broadphaseOnMetal(
            scene,
            cpu.worldAabbs,
            exactCapacity
        );
        require(
            exact.status.code == MR_STEP_SUCCESS &&
                exact.status.emittedPairs == exactCapacity &&
                std::ranges::equal(
                    exact.pairs,
                    cpu.pairs,
                    pairEqual
                ),
            "exact-capacity broadphase dispatch failed"
        );

        const MetalRun overflow = broadphaseOnMetal(
            scene,
            cpu.worldAabbs,
            exactCapacity - 1u
        );
        require(
            overflow.status.code ==
                MR_STEP_PAIR_CAPACITY_OVERFLOW &&
                overflow.status.requiredPairs == exactCapacity &&
                overflow.status.emittedPairs == 0u &&
                overflow.outputUntouched,
            "one-short capacity failure was not transactional"
        );

        std::vector<MRAabbGPU> invalidAabbs = cpu.worldAabbs;
        invalidAabbs[1].lower.x =
            std::numeric_limits<float>::quiet_NaN();
        const MetalRun invalid = broadphaseOnMetal(
            scene,
            invalidAabbs,
            exactCapacity
        );
        require(
            invalid.status.code == MR_STEP_NONFINITE_INPUT &&
                invalid.status.emittedPairs == 0u &&
                invalid.outputUntouched,
            "non-finite preflight failure was not transactional"
        );

        std::vector<MRAabbGPU> subnormalAabbs =
            cpu.worldAabbs;
        subnormalAabbs[1].lower.x =
            std::numeric_limits<float>::denorm_min();
        subnormalAabbs[1].upper.x = 0.0f;
        const MetalRun subnormalAabb = broadphaseOnMetal(
            scene,
            subnormalAabbs,
            exactCapacity
        );
        require(
            subnormalAabb.status.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalAabb.status.emittedPairs == 0u &&
                subnormalAabb.outputUntouched,
            "subnormal inverted AABB was accepted"
        );

        std::vector<MRAabbGPU> outOfDomainAabbs =
            cpu.worldAabbs;
        outOfDomainAabbs[1].lower.x =
            std::numeric_limits<float>::max();
        outOfDomainAabbs[1].upper.x =
            std::numeric_limits<float>::max();
        const MetalRun outOfDomainAabb = broadphaseOnMetal(
            scene,
            outOfDomainAabbs,
            exactCapacity
        );
        require(
            outOfDomainAabb.status.code ==
                MR_STEP_NONFINITE_INPUT &&
                outOfDomainAabb.status.emittedPairs == 0u &&
                outOfDomainAabb.outputUntouched,
            "out-of-domain finite AABB was accepted"
        );

        const auto requireRejected = [](
            const MetalRun& run,
            const std::uint32_t expectedCode,
            const std::string& label
        ) {
            require(
                run.status.code == expectedCode &&
                    run.status.emittedPairs == 0u &&
                    run.outputUntouched,
                label + " was not rejected transactionally"
            );
        };

        Scene invalidBody = scene;
        invalidBody.bodies[1].orientation =
            f4(0.0f, 0.0f, 0.0f, 0.0f);
        requireRejected(
            broadphaseOnMetal(
                invalidBody,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "zero body quaternion"
        );

        Scene invalidUnusedBody = scene;
        MRBodyStateGPU unusedBody =
            makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC);
        unusedBody.orientation =
            f4(0.0f, 0.0f, 0.0f, 0.0f);
        invalidUnusedBody.bodies.push_back(unusedBody);
        metalrobo::PersistentManifoldCache
            invalidUnusedBodyCache;
        const metalrobo::CollisionFrame invalidUnusedBodyCpu =
            metalrobo::collideCpuReference(
                invalidUnusedBody.shapes,
                invalidUnusedBody.bodies,
                config,
                invalidUnusedBodyCache,
                invalidUnusedBody.exclusions
            );
        require(
            invalidUnusedBodyCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidUnusedBodyCpu.pairs.empty(),
            "CPU oracle accepted a malformed unused body"
        );
        requireRejected(
            broadphaseOnMetal(
                invalidUnusedBody,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "malformed unused body"
        );

        Scene invalidShapeRotation = scene;
        invalidShapeRotation.shapes[1].localRotation =
            f4(0.0f, 0.0f, 0.0f, 0.0f);
        requireRejected(
            broadphaseOnMetal(
                invalidShapeRotation,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "zero shape quaternion"
        );

        Scene invalidDimensions = scene;
        invalidDimensions.shapes[1].dimensions.x = -0.25f;
        requireRejected(
            broadphaseOnMetal(
                invalidDimensions,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "negative active shape dimension"
        );

        Scene invalidContactOffset = scene;
        invalidContactOffset
            .shapes[0]
            .contactRestAndBoundingRadius.x = -0.01f;
        requireRejected(
            broadphaseOnMetal(
                invalidContactOffset,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "negative contact offset"
        );

        Scene malformedUnsupported = scene;
        malformedUnsupported.shapes.back().flags = 0u;
        malformedUnsupported.shapes.back().dimensions.x =
            std::numeric_limits<float>::quiet_NaN();
        metalrobo::PersistentManifoldCache
            malformedUnsupportedCache;
        const metalrobo::CollisionFrame malformedUnsupportedCpu =
            metalrobo::collideCpuReference(
                malformedUnsupported.shapes,
                malformedUnsupported.bodies,
                config,
                malformedUnsupportedCache,
                malformedUnsupported.exclusions
            );
        require(
            malformedUnsupportedCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                malformedUnsupportedCpu.pairs.empty(),
            "CPU malformed-unsupported error precedence drifted"
        );
        requireRejected(
            broadphaseOnMetal(
                malformedUnsupported,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "malformed unsupported shape"
        );

        Scene globalPrecedence = scene;
        globalPrecedence.shapes[0].shapeType =
            MR_SHAPE_CYLINDER;
        globalPrecedence.shapes[1].dimensions.x =
            std::numeric_limits<float>::quiet_NaN();
        metalrobo::PersistentManifoldCache
            globalPrecedenceCache;
        const metalrobo::CollisionFrame globalPrecedenceCpu =
            metalrobo::collideCpuReference(
                globalPrecedence.shapes,
                globalPrecedence.bodies,
                config,
                globalPrecedenceCache,
                globalPrecedence.exclusions
            );
        require(
            globalPrecedenceCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                globalPrecedenceCpu.pairs.empty(),
            "CPU common-record error precedence drifted"
        );
        requireRejected(
            broadphaseOnMetal(
                globalPrecedence,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "global common-record error precedence"
        );

        Scene derivedOverflow = scene;
        const std::uint32_t disabledBody =
            derivedOverflow.shapes.back().bodyIndex;
        derivedOverflow.bodies[disabledBody].position.x =
            std::numeric_limits<float>::max();
        derivedOverflow.shapes.back().localPosition.x =
            std::numeric_limits<float>::max();
        metalrobo::PersistentManifoldCache
            derivedOverflowCache;
        const metalrobo::CollisionFrame derivedOverflowCpu =
            metalrobo::collideCpuReference(
                derivedOverflow.shapes,
                derivedOverflow.bodies,
                config,
                derivedOverflowCache,
                derivedOverflow.exclusions
            );
        require(
            derivedOverflowCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                derivedOverflowCpu.pairs.empty(),
            "CPU accepted derived FP32 transform overflow"
        );
        requireRejected(
            broadphaseOnMetal(
                derivedOverflow,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "derived FP32 transform overflow"
        );

        Scene extremeExtent = scene;
        extremeExtent.shapes[1].dimensions.x =
            std::numeric_limits<float>::max();
        extremeExtent.shapes[1]
            .contactRestAndBoundingRadius.x = 1.0e30f;
        metalrobo::PersistentManifoldCache extremeExtentCache;
        const metalrobo::CollisionFrame extremeExtentCpu =
            metalrobo::collideCpuReference(
                extremeExtent.shapes,
                extremeExtent.bodies,
                config,
                extremeExtentCache,
                extremeExtent.exclusions
            );
        require(
            extremeExtentCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                extremeExtentCpu.pairs.empty(),
            "CPU accepted out-of-domain finite extent"
        );
        requireRejected(
            broadphaseOnMetal(
                extremeExtent,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "out-of-domain finite extent"
        );

        Scene subnormalDimension = scene;
        subnormalDimension.shapes[1].dimensions.x =
            std::numeric_limits<float>::denorm_min();
        metalrobo::PersistentManifoldCache
            subnormalDimensionCache;
        const metalrobo::CollisionFrame subnormalDimensionCpu =
            metalrobo::collideCpuReference(
                subnormalDimension.shapes,
                subnormalDimension.bodies,
                config,
                subnormalDimensionCache,
                subnormalDimension.exclusions
            );
        require(
            subnormalDimensionCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalDimensionCpu.pairs.empty(),
            "CPU accepted subnormal active dimension"
        );
        requireRejected(
            broadphaseOnMetal(
                subnormalDimension,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "subnormal active dimension"
        );

        Scene subnormalOffset = scene;
        subnormalOffset.shapes[1]
            .contactRestAndBoundingRadius.x =
                -std::numeric_limits<float>::denorm_min();
        metalrobo::PersistentManifoldCache
            subnormalOffsetCache;
        const metalrobo::CollisionFrame subnormalOffsetCpu =
            metalrobo::collideCpuReference(
                subnormalOffset.shapes,
                subnormalOffset.bodies,
                config,
                subnormalOffsetCache,
                subnormalOffset.exclusions
            );
        require(
            subnormalOffsetCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalOffsetCpu.pairs.empty(),
            "CPU accepted signed subnormal contact offset"
        );
        requireRejected(
            broadphaseOnMetal(
                subnormalOffset,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_NONFINITE_INPUT,
            "signed subnormal contact offset"
        );

        Scene quaternionBoundary = scene;
        const std::uint32_t boundaryBody =
            quaternionBoundary.shapes.back().bodyIndex;
        quaternionBoundary.bodies[boundaryBody].orientation =
            f4(
                0.72941094636917114,
                0.61207860708236694,
                -0.0090453298762440681,
                -0.30533197522163391
            );
        metalrobo::PersistentManifoldCache
            quaternionBoundaryCache;
        const metalrobo::CollisionFrame quaternionBoundaryCpu =
            metalrobo::collideCpuReference(
                quaternionBoundary.shapes,
                quaternionBoundary.bodies,
                config,
                quaternionBoundaryCache,
                quaternionBoundary.exclusions
            );
        const MetalRun quaternionBoundaryMetal =
            broadphaseOnMetal(
                quaternionBoundary,
                cpu.worldAabbs,
                exactCapacity
            );
        require(
            quaternionBoundaryCpu.succeeded() &&
                quaternionBoundaryCpu.pairs.size() ==
                    cpu.pairs.size() &&
                quaternionBoundaryMetal.status.code ==
                    MR_STEP_SUCCESS &&
                quaternionBoundaryMetal.status.requiredPairs ==
                    first.status.requiredPairs &&
                quaternionBoundaryMetal.status.emittedPairs ==
                    first.status.emittedPairs,
            "FP32 quaternion acceptance boundary diverged"
        );

        Scene activeUnsupported = scene;
        activeUnsupported.shapes.back().flags = 0u;
        requireRejected(
            broadphaseOnMetal(
                activeUnsupported,
                cpu.worldAabbs,
                exactCapacity
            ),
            MR_STEP_UNSUPPORTED,
            "active unsupported shape"
        );

        std::cout
            << "device=" << first.deviceName
            << " broadphase=metal_parallel_flag_scan_scatter"
            << " shapes=" << shapeCount
            << " logical_pairs=" << logicalPairCount
            << " scan_blocks=" << scanBlockCount
            << " candidate_pairs=" << exactCapacity
            << " cpu_parity=yes"
            << " deterministic=yes"
            << " exact_capacity=yes"
            << " zero_pair_worlds=yes"
            << " overflow_transactional=yes"
            << " nonfinite_transactional=yes"
            << " shape_validation_transactional=yes"
            << " strict_body_stream=yes"
            << " error_precedence=yes"
            << " derived_transform_validation=yes"
            << " bounded_collision_domain=yes"
            << " subnormal_policy=yes"
            << " quaternion_boundary_parity=yes"
            << " unsupported_transactional=yes"
            << " global_append_atomics=none"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "deterministic broadphase probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
