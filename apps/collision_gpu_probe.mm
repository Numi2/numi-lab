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
    };
    scene.exclusions = {
        {2u, 1u},
        {1u, 2u},
        {12u, 12u},
        {99u, 100u},
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
        data != nullptr && count > 0u,
        "attempted to allocate an empty Metal buffer"
    );
    require(
        count <= std::numeric_limits<NSUInteger>::max() / sizeof(T),
        "Metal buffer byte count overflow"
    );
    const NSUInteger byteCount =
        static_cast<NSUInteger>(count * sizeof(T));
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:data
                           length:byteCount
                          options:MTLResourceStorageModeShared];
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

        std::cout << std::scientific << std::setprecision(6)
                  << "device=\"" << first.deviceName << "\""
                  << " broadphase=metal_o_n2_baseline"
                  << " shapes=" << scene.shapes.size()
                  << " pairs=" << first.status.requiredPairs
                  << " raw_contacts="
                  << first.status.requiredContacts
                  << " max_witness_error="
                  << maximumWitnessError
                  << " canonical_filters=yes"
                  << " stable_features=yes"
                  << " deterministic_replay=yes"
                  << " overflow_transactional=yes"
                  << " finite_validation=yes"
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_collision_gpu_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
