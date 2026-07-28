#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/Collision.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef METALROBO_COLLISION_METALLIB
#define METALROBO_COLLISION_METALLIB ""
#endif

namespace {

constexpr std::uint8_t kOutputSentinel = 0xa5u;
constexpr double kWitnessTolerance = 8.0e-6;

struct Scene {
    std::vector<MRBodyStateGPU> bodies;
    std::vector<MRShapeGPU> shapes;
    std::vector<metalrobo::CollisionPairExclusion> exclusions;
    std::vector<MRCandidatePairGPU> gpuExclusions;
};

struct MetalRun {
    MRSolverStatusGPU status{};
    std::vector<MRCandidatePairGPU> pairs;
    std::vector<MRRawContactGPU> contacts;
    std::vector<std::uint32_t> contactPairIndices;
    bool outputBuffersUntouched = false;
    bool unusedOutputSlotsUntouched = false;
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
    MRBodyStateGPU result{};
    result.position = f4(x, y, z, 1.0f);
    result.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.flagsAndIndices[0] = motion;
    return result;
}

MRShapeGPU makeShape(
    const std::uint32_t body,
    const std::uint32_t type,
    const float radius,
    const std::uint32_t group = 1u,
    const std::uint32_t mask = 1u,
    const mr_float4 localPosition =
        f4(0.0f, 0.0f, 0.0f, 1.0f)
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = type;
    result.collisionGroup = group;
    result.collisionMask = mask;
    result.slotGeneration = 1000u + body;
    result.localPosition = localPosition;
    result.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.dimensions = f4(radius, 0.0f, 0.0f, 0.0f);
    result.contactRestAndBoundingRadius =
        f4(0.02f, 0.0f, radius, 0.0f);
    return result;
}

MRShapeGPU makeCapsuleShape(
    const std::uint32_t body,
    const float radius,
    const float halfLength,
    const std::uint32_t group,
    const std::uint32_t mask,
    const mr_float4 localRotation
) {
    MRShapeGPU result = makeShape(
        body,
        MR_SHAPE_CAPSULE,
        radius,
        group,
        mask
    );
    result.localRotation = localRotation;
    result.dimensions =
        f4(radius, halfLength, 0.0f, 0.0f);
    result.contactRestAndBoundingRadius =
        f4(0.02f, 0.0f, radius + halfLength, 0.0f);
    return result;
}

MRShapeGPU makeBoxShape(
    const std::uint32_t body,
    const float halfX,
    const float halfY,
    const float halfZ,
    const std::uint32_t group,
    const std::uint32_t mask,
    const mr_float4 localRotation
) {
    MRShapeGPU result = makeShape(
        body,
        MR_SHAPE_BOX,
        1.0f,
        group,
        mask
    );
    result.localRotation = localRotation;
    result.dimensions = f4(halfX, halfY, halfZ, 0.0f);
    result.contactRestAndBoundingRadius = f4(
        0.02f,
        0.0f,
        std::sqrt(
            halfX * halfX +
            halfY * halfY +
            halfZ * halfZ
        ),
        0.0f
    );
    return result;
}

Scene makeScene() {
    Scene scene;
    scene.bodies = {
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(-1.20f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(-0.35f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(0.50f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(2.00f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(2.85f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(4.00f, 0.45f, 0.0f, MR_MOTION_STATIC),
        makeBody(6.00f, 2.00f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(6.90f, 2.90f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(9.00f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(9.50f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(11.00f, 2.00f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(13.00f, 0.20f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(15.00f, 0.00f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(0.00f, 0.00f, 0.0f, MR_MOTION_STATIC),
        makeBody(17.00f, 0.00f, 0.0f, MR_MOTION_DYNAMIC),
    };
    scene.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, 0.0f),
        makeShape(1u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(2u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(3u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(4u, MR_SHAPE_SPHERE, 0.5f, 2u, 2u),
        makeShape(5u, MR_SHAPE_SPHERE, 0.5f, 2u, 2u),
        makeShape(6u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(7u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(8u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(9u, MR_SHAPE_SPHERE, 0.5f, 1u, 0u),
        makeShape(10u, MR_SHAPE_SPHERE, 0.5f),
        makeShape(
            11u,
            MR_SHAPE_SPHERE,
            0.5f,
            1u,
            1u,
            f4(-0.2f, 0.0f, 0.0f, 1.0f)
        ),
        makeShape(
            11u,
            MR_SHAPE_SPHERE,
            0.5f,
            1u,
            1u,
            f4(0.2f, 0.0f, 0.0f, 1.0f)
        ),
        makeCapsuleShape(
            12u,
            0.25f,
            0.50f,
            4u,
            4u,
            f4(
                0.0f,
                0.0f,
                0.7071067811865476f,
                0.7071067811865476f
            )
        ),
        makeBoxShape(
            13u,
            0.35f,
            0.01f,
            0.25f,
            4u,
            4u,
            f4(
                0.0f,
                0.25881904510252074f,
                0.0f,
                0.9659258262890683f
            )
        ),
        makeShape(
            14u,
            MR_SHAPE_PLANE,
            0.0f,
            4u,
            4u
        ),
        makeShape(
            15u,
            MR_SHAPE_CYLINDER,
            0.25f,
            4u,
            4u
        ),
    };
    scene.shapes.back().flags =
        MR_SHAPE_FLAG_SIMULATION_DISABLED;
    scene.exclusions = {
        {2u, 1u},
        {1u, 2u},
        {12u, 12u},
    };
    scene.gpuExclusions.reserve(scene.exclusions.size());
    for (const auto exclusion : scene.exclusions) {
        MRCandidatePairGPU gpu{};
        gpu.colliderA = exclusion.colliderA;
        gpu.colliderB = exclusion.colliderB;
        scene.gpuExclusions.push_back(gpu);
    }
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
        count <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
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
        const NSUInteger byteCount =
            static_cast<NSUInteger>(count * sizeof(T));
        buffer = [device
            newBufferWithBytes:data
                        length:byteCount
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
bool sameBytes(
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

template <typename T>
std::vector<T> copyBuffer(
    id<MTLBuffer> buffer,
    const std::size_t count
) {
    std::vector<T> result(count);
    if (count > 0u) {
        std::memcpy(
            result.data(),
            buffer.contents,
            count * sizeof(T)
        );
    }
    return result;
}

MetalRun collideOnMetal(
    const Scene& scene,
    const std::uint32_t environment,
    const std::uint32_t pairCapacity,
    const std::uint32_t contactCapacity
) {
    @autoreleasepool {
        const std::string metallibPath =
            METALROBO_COLLISION_METALLIB;
        require(
            !metallibPath.empty(),
            "METALROBO_COLLISION_METALLIB is empty"
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
            "failed to load collision metallib: " +
                describeError(error)
        );
        id<MTLFunction> function =
            [library newFunctionWithName:@"mr_collide_baseline"];
        require(
            function != nil,
            "metallib does not contain mr_collide_baseline"
        );
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        require(
            pipeline != nil,
            "failed to create collision pipeline: " +
                describeError(error)
        );

        const auto pairSeed =
            sentinelVector<MRCandidatePairGPU>(pairCapacity);
        const auto contactSeed =
            sentinelVector<MRRawContactGPU>(contactCapacity);
        const auto contactPairSeed =
            sentinelVector<std::uint32_t>(contactCapacity);
        MRSolverStatusGPU statusSeed{};
        std::memset(
            &statusSeed,
            kOutputSentinel,
            sizeof(statusSeed)
        );

        id<MTLBuffer> shapes = makeBuffer(
            device,
            scene.shapes.data(),
            scene.shapes.size(),
            @"collision shapes"
        );
        id<MTLBuffer> bodies = makeBuffer(
            device,
            scene.bodies.data(),
            scene.bodies.size(),
            @"collision bodies"
        );
        id<MTLBuffer> exclusions = makeBuffer(
            device,
            scene.gpuExclusions.data(),
            scene.gpuExclusions.size(),
            @"collision exclusions"
        );
        id<MTLBuffer> pairs = makeBuffer(
            device,
            pairSeed.data(),
            pairSeed.size(),
            @"candidate pairs"
        );
        id<MTLBuffer> contacts = makeBuffer(
            device,
            contactSeed.data(),
            contactSeed.size(),
            @"raw contacts"
        );
        id<MTLBuffer> contactPairIndices = makeBuffer(
            device,
            contactPairSeed.data(),
            contactPairSeed.size(),
            @"contact pair indices"
        );
        id<MTLBuffer> status = makeBuffer(
            device,
            &statusSeed,
            1u,
            @"collision status"
        );

        const std::uint32_t bodyCount =
            static_cast<std::uint32_t>(scene.bodies.size());
        const std::uint32_t shapeCount =
            static_cast<std::uint32_t>(scene.shapes.size());
        const std::uint32_t exclusionCount =
            static_cast<std::uint32_t>(
                scene.gpuExclusions.size()
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
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:shapes offset:0 atIndex:0];
        [encoder setBuffer:bodies offset:0 atIndex:1];
        [encoder setBuffer:exclusions offset:0 atIndex:2];
        [encoder setBuffer:pairs offset:0 atIndex:3];
        [encoder setBuffer:contacts offset:0 atIndex:4];
        [encoder setBuffer:contactPairIndices offset:0 atIndex:5];
        [encoder setBuffer:status offset:0 atIndex:6];
        [encoder setBytes:&bodyCount
                   length:sizeof(bodyCount)
                  atIndex:7];
        [encoder setBytes:&shapeCount
                   length:sizeof(shapeCount)
                  atIndex:8];
        [encoder setBytes:&environment
                   length:sizeof(environment)
                  atIndex:9];
        [encoder setBytes:&exclusionCount
                   length:sizeof(exclusionCount)
                  atIndex:10];
        [encoder setBytes:&pairCapacity
                   length:sizeof(pairCapacity)
                  atIndex:11];
        [encoder setBytes:&contactCapacity
                   length:sizeof(contactCapacity)
                  atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(
            commandBuffer.status ==
                MTLCommandBufferStatusCompleted,
            "collision dispatch failed: " +
                describeError(commandBuffer.error)
        );

        MetalRun result;
        std::memcpy(
            &result.status,
            status.contents,
            sizeof(result.status)
        );
        const auto allPairs =
            copyBuffer<MRCandidatePairGPU>(
                pairs,
                pairSeed.size()
            );
        const auto allContacts =
            copyBuffer<MRRawContactGPU>(
                contacts,
                contactSeed.size()
            );
        const auto allContactPairIndices =
            copyBuffer<std::uint32_t>(
                contactPairIndices,
                contactPairSeed.size()
            );
        result.outputBuffersUntouched =
            sameBytes<MRCandidatePairGPU>(allPairs, pairSeed) &&
            sameBytes<MRRawContactGPU>(
                allContacts,
                contactSeed
            ) &&
            sameBytes<std::uint32_t>(
                allContactPairIndices,
                contactPairSeed
            );

        const bool succeeded =
            result.status.code == MR_STEP_SUCCESS;
        if (succeeded) {
            require(
                result.status.requiredPairs <= pairCapacity &&
                    result.status.requiredContacts <=
                        contactCapacity,
                "successful kernel exceeded declared capacity"
            );
            result.pairs.assign(
                allPairs.begin(),
                allPairs.begin() +
                    result.status.requiredPairs
            );
            result.contacts.assign(
                allContacts.begin(),
                allContacts.begin() +
                    result.status.requiredContacts
            );
            result.contactPairIndices.assign(
                allContactPairIndices.begin(),
                allContactPairIndices.begin() +
                    result.status.requiredContacts
            );

            const auto pairTail = std::span{
                allPairs
            }.subspan(result.status.requiredPairs);
            const auto pairSeedTail = std::span{
                pairSeed
            }.subspan(result.status.requiredPairs);
            const auto contactTail = std::span{
                allContacts
            }.subspan(result.status.requiredContacts);
            const auto contactSeedTail = std::span{
                contactSeed
            }.subspan(result.status.requiredContacts);
            const auto indexTail = std::span{
                allContactPairIndices
            }.subspan(result.status.requiredContacts);
            const auto indexSeedTail = std::span{
                contactPairSeed
            }.subspan(result.status.requiredContacts);
            result.unusedOutputSlotsUntouched =
                sameBytes<MRCandidatePairGPU>(
                    pairTail,
                    pairSeedTail
                ) &&
                sameBytes<MRRawContactGPU>(
                    contactTail,
                    contactSeedTail
                ) &&
                sameBytes<std::uint32_t>(
                    indexTail,
                    indexSeedTail
                );
        }
        result.deviceName = nsString(device.name);
        return result;
    }
}

bool samePair(
    const MRCandidatePairGPU& left,
    const MRCandidatePairGPU& right
) {
    return left.environment == right.environment &&
        left.colliderA == right.colliderA &&
        left.colliderB == right.colliderB &&
        left.flags == right.flags;
}

double componentError(
    const mr_float4 left,
    const mr_float4 right
) {
    return std::max({
        std::abs(static_cast<double>(left.x) - right.x),
        std::abs(static_cast<double>(left.y) - right.y),
        std::abs(static_cast<double>(left.z) - right.z),
        std::abs(static_cast<double>(left.w) - right.w),
    });
}

double compareWithCpu(
    const metalrobo::CollisionFrame& cpu,
    const MetalRun& gpu
) {
    require(
        cpu.pairs.size() == gpu.pairs.size(),
        "CPU/Metal candidate pair counts differ"
    );
    require(
        cpu.rawContacts.size() == gpu.contacts.size(),
        "CPU/Metal raw contact counts differ"
    );
    require(
        cpu.rawContactPairIndices ==
            gpu.contactPairIndices,
        "CPU/Metal raw-contact pair indices differ"
    );

    for (std::size_t index = 0u;
         index < cpu.pairs.size();
         ++index) {
        require(
            samePair(cpu.pairs[index], gpu.pairs[index]),
            "CPU/Metal sorted candidate pairs differ"
        );
        if (index > 0u) {
            const auto& previous = gpu.pairs[index - 1u];
            const auto& current = gpu.pairs[index];
            require(
                std::pair{
                    previous.colliderA,
                    previous.colliderB
                } <
                    std::pair{
                        current.colliderA,
                        current.colliderB
                    },
                "Metal candidate pairs are not strictly sorted"
            );
        }
    }

    double maximumError = 0.0;
    for (std::size_t index = 0u;
         index < cpu.rawContacts.size();
         ++index) {
        const MRRawContactGPU& cpuContact =
            cpu.rawContacts[index];
        const MRRawContactGPU& gpuContact =
            gpu.contacts[index];
        for (std::size_t field = 0u; field < 4u; ++field) {
            require(
                cpuContact.featureAndFlags[field] ==
                    gpuContact.featureAndFlags[field],
                "CPU/Metal stable feature keys differ"
            );
        }
        maximumError = std::max({
            maximumError,
            componentError(
                cpuContact.normalAndSeparation,
                gpuContact.normalAndSeparation
            ),
            componentError(
                cpuContact.pointAWorld,
                gpuContact.pointAWorld
            ),
            componentError(
                cpuContact.pointBWorld,
                gpuContact.pointBWorld
            ),
        });
    }
    require(
        maximumError <= kWitnessTolerance,
        "CPU/Metal contact witness parity exceeded tolerance"
    );
    return maximumError;
}

bool containsPair(
    const std::span<const MRCandidatePairGPU> pairs,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    return std::ranges::any_of(
        pairs,
        [=](const MRCandidatePairGPU& pair) {
            return pair.colliderA == colliderA &&
                pair.colliderB == colliderB;
        }
    );
}

bool pairHasContact(
    const MetalRun& run,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    for (const std::uint32_t pairIndex :
         run.contactPairIndices) {
        require(
            pairIndex < run.pairs.size(),
            "Metal contact references an invalid pair"
        );
        if (run.pairs[pairIndex].colliderA == colliderA &&
            run.pairs[pairIndex].colliderB == colliderB) {
            return true;
        }
    }
    return false;
}

std::vector<MRRawContactGPU> contactsForPair(
    const MetalRun& run,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    std::vector<MRRawContactGPU> result;
    for (std::size_t contactIndex = 0u;
         contactIndex < run.contactPairIndices.size();
         ++contactIndex) {
        const std::uint32_t pairIndex =
            run.contactPairIndices[contactIndex];
        require(
            pairIndex < run.pairs.size(),
            "Metal contact references an invalid pair"
        );
        const MRCandidatePairGPU& pair = run.pairs[pairIndex];
        if (pair.colliderA == colliderA &&
            pair.colliderB == colliderB) {
            result.push_back(run.contacts[contactIndex]);
        }
    }
    return result;
}

std::uint32_t manifoldPointCount(
    const metalrobo::CollisionFrame& frame,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    for (const MRManifoldHeaderGPU& header :
         frame.manifoldHeaders) {
        if (header.pairAndCount[1] == colliderA &&
            header.pairAndCount[2] == colliderB) {
            return header.pairAndCount[3];
        }
    }
    return 0u;
}

bool shapeAppearsInPair(
    const std::span<const MRCandidatePairGPU> pairs,
    const std::uint32_t collider
) {
    return std::ranges::any_of(
        pairs,
        [=](const MRCandidatePairGPU& pair) {
            return pair.colliderA == collider ||
                pair.colliderB == collider;
        }
    );
}

std::uint32_t expectedFeature(
    const std::uint32_t shapeType,
    const std::uint32_t localFeature
) {
    return ((shapeType & 0x0fu) << 28u) |
        (localFeature & 0x0fffffffu);
}

bool sameSuccessfulRun(
    const MetalRun& left,
    const MetalRun& right
) {
    return
        left.status.code == right.status.code &&
        left.status.requiredPairs ==
            right.status.requiredPairs &&
        left.status.requiredContacts ==
            right.status.requiredContacts &&
        sameBytes<MRCandidatePairGPU>(
            left.pairs,
            right.pairs
        ) &&
        sameBytes<MRRawContactGPU>(
            left.contacts,
            right.contacts
        ) &&
        left.contactPairIndices ==
            right.contactPairIndices;
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t environment = 23u;
        const Scene emptyScene;
        metalrobo::CollisionConfig emptyConfig;
        emptyConfig.environment = environment;
        metalrobo::PersistentManifoldCache emptyCache;
        const metalrobo::CollisionFrame emptyCpu =
            metalrobo::collideCpuReference(
                emptyScene.shapes,
                emptyScene.bodies,
                emptyConfig,
                emptyCache,
                emptyScene.exclusions
            );
        const MetalRun emptyMetal = collideOnMetal(
            emptyScene,
            environment,
            0u,
            0u
        );
        require(
            emptyCpu.succeeded() &&
                emptyCpu.pairs.empty() &&
                emptyCpu.rawContacts.empty() &&
                emptyMetal.status.code == MR_STEP_SUCCESS &&
                emptyMetal.status.requiredPairs == 0u &&
                emptyMetal.status.requiredContacts == 0u &&
                emptyMetal.outputBuffersUntouched,
            "zero-shape CPU/Metal collision world diverged"
        );

        const Scene scene = makeScene();

        metalrobo::CollisionConfig cpuConfig;
        cpuConfig.environment = environment;
        cpuConfig.capacities = {
            .pairCapacity = 128u,
            .rawContactCapacity = 128u,
            .manifoldCapacity = 128u,
        };
        metalrobo::PersistentManifoldCache cpuCache;
        const metalrobo::CollisionFrame cpu =
            metalrobo::collideCpuReference(
                scene.shapes,
                scene.bodies,
                cpuConfig,
                cpuCache,
                scene.exclusions
            );
        require(cpu.succeeded(), "CPU collision reference failed");
        require(
            !cpu.pairs.empty() && !cpu.rawContacts.empty(),
            "collision scene did not exercise the pipeline"
        );

        const std::uint32_t pairCapacity =
            static_cast<std::uint32_t>(cpu.pairs.size() + 3u);
        const std::uint32_t contactCapacity =
            static_cast<std::uint32_t>(
                cpu.rawContacts.size() + 3u
            );
        const MetalRun first = collideOnMetal(
            scene,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            first.status.code == MR_STEP_SUCCESS,
            "Metal collision baseline did not succeed"
        );
        require(
            first.status.requiredPairs == cpu.pairs.size() &&
                first.status.requiredContacts ==
                    cpu.rawContacts.size(),
            "Metal preflight counts differ from CPU reference"
        );
        require(
            first.status.activeContacts ==
                first.status.requiredContacts,
            "Metal active-contact diagnostic is incorrect"
        );
        require(
            first.unusedOutputSlotsUntouched,
            "Metal wrote beyond the required output prefix"
        );
        const double maximumWitnessError =
            compareWithCpu(cpu, first);

        require(
            containsPair(first.pairs, 2u, 3u) &&
                pairHasContact(first, 2u, 3u),
            "sphere/sphere analytic narrowphase was not exercised"
        );
        require(
            containsPair(first.pairs, 0u, 1u) &&
                pairHasContact(first, 0u, 1u),
            "sphere/plane analytic narrowphase was not exercised"
        );
        require(
            containsPair(first.pairs, 7u, 8u) &&
                !pairHasContact(first, 7u, 8u),
            "broadphase-only diagonal sphere pair was not preserved"
        );
        require(
            !containsPair(first.pairs, 1u, 2u),
            "explicit pair exclusion was ignored"
        );
        require(
            !containsPair(first.pairs, 9u, 10u),
            "collision group/mask filtering was ignored"
        );
        require(
            !containsPair(first.pairs, 11u, 12u),
            "same-body filtering was ignored"
        );
        require(
            !containsPair(first.pairs, 0u, 6u),
            "static/static filtering was ignored"
        );
        require(
            containsPair(first.pairs, 13u, 15u),
            "capsule/plane analytic pair was not emitted"
        );
        require(
            containsPair(first.pairs, 14u, 15u),
            "box/plane analytic pair was not emitted"
        );

        const std::vector<MRRawContactGPU> capsuleContacts =
            contactsForPair(first, 13u, 15u);
        require(
            capsuleContacts.size() == 2u,
            "horizontal capsule did not emit both endpoint witnesses"
        );
        for (std::uint32_t endpoint = 0u;
             endpoint < capsuleContacts.size();
             ++endpoint) {
            require(
                capsuleContacts[endpoint].featureAndFlags[0] ==
                    expectedFeature(
                        MR_SHAPE_CAPSULE,
                        endpoint
                    ) &&
                    capsuleContacts[endpoint]
                            .featureAndFlags[1] ==
                        expectedFeature(MR_SHAPE_PLANE, 0u),
                "capsule endpoint features were not stable and ordered"
            );
        }
        require(
            std::abs(
                capsuleContacts[0].normalAndSeparation.w -
                capsuleContacts[1].normalAndSeparation.w
            ) <= 2.0e-6f &&
                std::abs(
                    capsuleContacts[0].pointAWorld.x -
                    capsuleContacts[1].pointAWorld.x
                ) >= 0.99f,
            "equal-depth capsule endpoint degeneracy was mishandled"
        );

        const std::vector<MRRawContactGPU> boxContacts =
            contactsForPair(first, 14u, 15u);
        require(
            boxContacts.size() == 8u,
            "box/plane did not preserve all eight raw corner witnesses"
        );
        for (std::uint32_t boxVertex = 0u;
             boxVertex < boxContacts.size();
             ++boxVertex) {
            require(
                boxContacts[boxVertex].featureAndFlags[0] ==
                    expectedFeature(
                        MR_SHAPE_BOX,
                        boxVertex
                    ) &&
                    boxContacts[boxVertex]
                            .featureAndFlags[1] ==
                        expectedFeature(MR_SHAPE_PLANE, 0u),
                "box corner features were not stable and ordered"
            );
        }
        require(
            manifoldPointCount(cpu, 14u, 15u) == 4u,
            "CPU manifold did not reduce eight box witnesses to four"
        );
        require(
            !shapeAppearsInPair(first.pairs, 16u),
            "simulation-disabled unsupported geometry entered pairs"
        );

        const MetalRun replay = collideOnMetal(
            scene,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            sameSuccessfulRun(first, replay),
            "Metal collision replay was not bit deterministic"
        );
        const MetalRun exactCapacity = collideOnMetal(
            scene,
            environment,
            first.status.requiredPairs,
            first.status.requiredContacts
        );
        require(
            sameSuccessfulRun(first, exactCapacity) &&
                exactCapacity.unusedOutputSlotsUntouched,
            "exact declared capacities did not succeed deterministically"
        );

        require(
            first.status.requiredPairs > 0u &&
                first.status.requiredContacts > 0u,
            "overflow probes need nonzero requirements"
        );
        const MetalRun pairOverflow = collideOnMetal(
            scene,
            environment,
            first.status.requiredPairs - 1u,
            contactCapacity
        );
        require(
            pairOverflow.status.code ==
                MR_STEP_PAIR_CAPACITY_OVERFLOW &&
                pairOverflow.status.requiredPairs ==
                    first.status.requiredPairs &&
                pairOverflow.status.requiredContacts ==
                    first.status.requiredContacts &&
                pairOverflow.outputBuffersUntouched,
            "pair-capacity preflight was not exact and transactional"
        );

        const MetalRun contactOverflow = collideOnMetal(
            scene,
            environment,
            pairCapacity,
            first.status.requiredContacts - 1u
        );
        require(
            contactOverflow.status.code ==
                MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
                contactOverflow.status.requiredPairs ==
                    first.status.requiredPairs &&
                contactOverflow.status.requiredContacts ==
                    first.status.requiredContacts &&
                contactOverflow.outputBuffersUntouched,
            "contact-capacity preflight was not exact and transactional"
        );

        Scene invalid = scene;
        invalid.shapes[3].localPosition.x =
            std::numeric_limits<float>::quiet_NaN();
        const MetalRun invalidResult = collideOnMetal(
            invalid,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidResult.status.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidResult.outputBuffersUntouched,
            "non-finite input did not fail transactionally"
        );

        Scene invalidCapsule = scene;
        invalidCapsule.shapes[13].dimensions.y = 0.0f;
        const MetalRun invalidCapsuleResult = collideOnMetal(
            invalidCapsule,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidCapsuleResult.status.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidCapsuleResult.outputBuffersUntouched,
            "degenerate capsule dimensions did not fail transactionally"
        );

        Scene invalidFlags = scene;
        invalidFlags.shapes[1].flags = 1u << 31u;
        metalrobo::PersistentManifoldCache invalidFlagsCache;
        const metalrobo::CollisionFrame invalidFlagsCpu =
            metalrobo::collideCpuReference(
                invalidFlags.shapes,
                invalidFlags.bodies,
                cpuConfig,
                invalidFlagsCache,
                invalidFlags.exclusions
            );
        const MetalRun invalidFlagsMetal = collideOnMetal(
            invalidFlags,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidFlagsCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidFlagsCpu.pairs.empty() &&
                invalidFlagsMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                invalidFlagsMetal.outputBuffersUntouched,
            "unknown shape flags did not fail consistently"
        );

        Scene invalidExclusion = scene;
        const std::uint32_t invalidCollider =
            static_cast<std::uint32_t>(
                invalidExclusion.shapes.size()
            );
        invalidExclusion.exclusions.push_back(
            {0u, invalidCollider}
        );
        invalidExclusion.gpuExclusions.push_back(
            {
                environment,
                0u,
                invalidCollider,
                0u
            }
        );
        metalrobo::PersistentManifoldCache
            invalidExclusionCache;
        const metalrobo::CollisionFrame invalidExclusionCpu =
            metalrobo::collideCpuReference(
                invalidExclusion.shapes,
                invalidExclusion.bodies,
                cpuConfig,
                invalidExclusionCache,
                invalidExclusion.exclusions
            );
        const MetalRun invalidExclusionMetal = collideOnMetal(
            invalidExclusion,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidExclusionCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidExclusionCpu.pairs.empty() &&
                invalidExclusionMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                invalidExclusionMetal.outputBuffersUntouched,
            "out-of-range exclusions did not fail consistently"
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
                cpuConfig,
                invalidUnusedBodyCache,
                invalidUnusedBody.exclusions
            );
        const MetalRun invalidUnusedBodyMetal = collideOnMetal(
            invalidUnusedBody,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            invalidUnusedBodyCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                invalidUnusedBodyCpu.pairs.empty() &&
                invalidUnusedBodyMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                invalidUnusedBodyMetal.outputBuffersUntouched,
            "malformed unused body did not fail consistently"
        );

        Scene malformedUnsupported = scene;
        malformedUnsupported.shapes[16].flags = 0u;
        malformedUnsupported.shapes[16].dimensions.x =
            std::numeric_limits<float>::quiet_NaN();
        metalrobo::PersistentManifoldCache
            malformedUnsupportedCache;
        const metalrobo::CollisionFrame malformedUnsupportedCpu =
            metalrobo::collideCpuReference(
                malformedUnsupported.shapes,
                malformedUnsupported.bodies,
                cpuConfig,
                malformedUnsupportedCache,
                malformedUnsupported.exclusions
            );
        const MetalRun malformedUnsupportedMetal = collideOnMetal(
            malformedUnsupported,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            malformedUnsupportedCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                malformedUnsupportedCpu.pairs.empty() &&
                malformedUnsupportedMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                malformedUnsupportedMetal.outputBuffersUntouched,
            "malformed unsupported shape error precedence diverged"
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
                cpuConfig,
                globalPrecedenceCache,
                globalPrecedence.exclusions
            );
        const MetalRun globalPrecedenceMetal = collideOnMetal(
            globalPrecedence,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            globalPrecedenceCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                globalPrecedenceCpu.pairs.empty() &&
                globalPrecedenceMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                globalPrecedenceMetal.outputBuffersUntouched,
            "global common-record error precedence diverged"
        );

        Scene derivedOverflow = scene;
        const std::uint32_t disabledBody =
            derivedOverflow.shapes[16].bodyIndex;
        derivedOverflow.bodies[disabledBody].position.x =
            std::numeric_limits<float>::max();
        derivedOverflow.shapes[16].localPosition.x =
            std::numeric_limits<float>::max();
        metalrobo::PersistentManifoldCache
            derivedOverflowCache;
        const metalrobo::CollisionFrame derivedOverflowCpu =
            metalrobo::collideCpuReference(
                derivedOverflow.shapes,
                derivedOverflow.bodies,
                cpuConfig,
                derivedOverflowCache,
                derivedOverflow.exclusions
            );
        const MetalRun derivedOverflowMetal = collideOnMetal(
            derivedOverflow,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            derivedOverflowCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                derivedOverflowCpu.pairs.empty() &&
                derivedOverflowMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                derivedOverflowMetal.outputBuffersUntouched,
            "derived FP32 transform overflow was not rejected"
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
                cpuConfig,
                extremeExtentCache,
                extremeExtent.exclusions
            );
        const MetalRun extremeExtentMetal = collideOnMetal(
            extremeExtent,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            extremeExtentCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                extremeExtentCpu.pairs.empty() &&
                extremeExtentMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                extremeExtentMetal.outputBuffersUntouched,
            "out-of-domain finite extent was accepted"
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
                cpuConfig,
                subnormalDimensionCache,
                subnormalDimension.exclusions
            );
        const MetalRun subnormalDimensionMetal = collideOnMetal(
            subnormalDimension,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            subnormalDimensionCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalDimensionCpu.pairs.empty() &&
                subnormalDimensionMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                subnormalDimensionMetal.outputBuffersUntouched,
            "subnormal active dimension was accepted"
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
                cpuConfig,
                subnormalOffsetCache,
                subnormalOffset.exclusions
            );
        const MetalRun subnormalOffsetMetal = collideOnMetal(
            subnormalOffset,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            subnormalOffsetCpu.diagnostics.code ==
                MR_STEP_NONFINITE_INPUT &&
                subnormalOffsetCpu.pairs.empty() &&
                subnormalOffsetMetal.status.code ==
                    MR_STEP_NONFINITE_INPUT &&
                subnormalOffsetMetal.outputBuffersUntouched,
            "signed subnormal contact offset was accepted"
        );

        Scene quaternionBoundary = scene;
        const std::uint32_t boundaryBody =
            quaternionBoundary.shapes[16].bodyIndex;
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
                cpuConfig,
                quaternionBoundaryCache,
                quaternionBoundary.exclusions
            );
        const MetalRun quaternionBoundaryMetal = collideOnMetal(
            quaternionBoundary,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            quaternionBoundaryCpu.succeeded() &&
                quaternionBoundaryCpu.pairs.size() ==
                    cpu.pairs.size() &&
                quaternionBoundaryMetal.status.code ==
                    MR_STEP_SUCCESS &&
                quaternionBoundaryMetal.status.requiredPairs ==
                    first.status.requiredPairs &&
                quaternionBoundaryMetal.status.requiredContacts ==
                    first.status.requiredContacts,
            "FP32 quaternion acceptance boundary diverged"
        );

        Scene activeUnsupported = scene;
        activeUnsupported.shapes[16].flags = 0u;
        const MetalRun unsupportedResult = collideOnMetal(
            activeUnsupported,
            environment,
            pairCapacity,
            contactCapacity
        );
        require(
            unsupportedResult.status.code == MR_STEP_UNSUPPORTED &&
                unsupportedResult.outputBuffersUntouched,
            "active unsupported geometry did not fail transactionally"
        );

        std::cout << std::scientific << std::setprecision(6)
                  << "device=\"" << first.deviceName << "\""
                  << " broadphase=metal_o_n2_baseline"
                  << " shapes=" << scene.shapes.size()
                  << " pairs=" << first.status.requiredPairs
                  << " raw_contacts="
                  << first.status.requiredContacts
                  << " capsule_endpoint_contacts="
                  << capsuleContacts.size()
                  << " box_raw_contacts="
                  << boxContacts.size()
                  << " box_manifold_contacts="
                  << manifoldPointCount(cpu, 14u, 15u)
                  << " max_witness_error="
                  << maximumWitnessError
                  << " canonical_filters=yes"
                  << " stable_features=yes"
                  << " deterministic_replay=yes"
                  << " overflow_transactional=yes"
                  << " finite_validation=yes"
                  << " strict_shape_flags=yes"
                  << " strict_exclusions=yes"
                  << " strict_body_stream=yes"
                  << " error_precedence=yes"
                  << " derived_transform_validation=yes"
                  << " bounded_collision_domain=yes"
                  << " subnormal_policy=yes"
                  << " quaternion_boundary_parity=yes"
                  << " zero_shape_world=yes"
                  << " disabled_unsupported_skipped=yes"
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_collision_gpu_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
