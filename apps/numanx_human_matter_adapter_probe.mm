#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanMatter.hpp"
#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "numi/matter/matter.hpp"
#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Reuse the owner probe's production-sized 160-DoF Human fixture inside this
// translation unit.  Its program callbacks are replaced below by the real
// adapter; the fixture supplies only the exact EngineModel, MyoSim sidecar and
// owner point stream.
namespace owner_fixture {
#pragma push_macro("main")
#undef main
#define main numanx_human_matter_candidate_probe_embedded_main
#include "numanx_human_matter_candidate_probe.mm"
#undef main
#pragma pop_macro("main")
} // namespace owner_fixture

#ifndef NUMI_MATTER_METALLIB
#error "NUMI_MATTER_METALLIB must name the build-tree Matter metallib"
#endif

#ifndef NUMI_MATTER_MATERIAL
#error "NUMI_MATTER_MATERIAL must name a production Matter material"
#endif

#ifndef NUMANX_ADAPTER_METALLIB
#error "NUMANX_ADAPTER_METALLIB must name the MetalRobo metallib"
#endif

namespace adapter_fixture {

constexpr std::uint32_t kEnvironmentCount = 1u;
constexpr std::uint32_t kDofs = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
constexpr std::uint32_t kQ = MR_NUMANX_COUPLED_HUMAN_MAX_Q;
constexpr std::uint32_t kBodies = 1u;
constexpr std::uint32_t kPoints = 1u;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string stringValue(NSString* value) {
    return value == nil || value.UTF8String == nullptr
        ? std::string{} : std::string(value.UTF8String);
}

std::string errorValue(NSError* error) {
    return error == nil ? "unknown Metal error" :
        stringValue(error.localizedDescription);
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::span<const T> values,
    NSString* label
) {
    require(device != nil && !values.empty(), "invalid Metal buffer input");
    id<MTLBuffer> result = [device
        newBufferWithBytes:values.data()
                    length:values.size_bytes()
                   options:MTLResourceStorageModeShared];
    require(result != nil && result.gpuAddress != 0u,
        "failed to allocate " + stringValue(label));
    result.label = label;
    return result;
}

template <typename T, std::size_t Count>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::array<T, Count>& values,
    NSString* label
) {
    return makeBuffer<T>(device, std::span<const T>(values), label);
}

template <typename T>
id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const T& value,
    NSString* label
) {
    return makeBuffer<T>(device, std::span<const T>(&value, 1u), label);
}

id<MTLBuffer> makeZeroBuffer(
    id<MTLDevice> device,
    const std::size_t bytes,
    NSString* label
) {
    require(bytes != 0u && bytes <= std::numeric_limits<NSUInteger>::max(),
        "invalid zero-buffer size");
    id<MTLBuffer> result = [device
        newBufferWithLength:static_cast<NSUInteger>(bytes)
                   options:MTLResourceStorageModeShared];
    require(result != nil && result.gpuAddress != 0u,
        "failed to allocate " + stringValue(label));
    std::memset(result.contents, 0, result.length);
    result.label = label;
    return result;
}

template <typename T>
T value(id<MTLBuffer> buffer) {
    require(buffer != nil && buffer.length >= sizeof(T),
        "readback buffer is undersized");
    T result{};
    std::memcpy(&result, buffer.contents, sizeof(result));
    return result;
}

void finish(id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    require(commandBuffer.status == MTLCommandBufferStatusCompleted,
        "Metal command failed: " + errorValue(commandBuffer.error));
}

void waitForSharedEventValue(
    id<MTLSharedEvent> event,
    const std::uint64_t expected,
    const char* message
) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while (event != nil && event.signaledValue < expected &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(event != nil && event.signaledValue >= expected, message);
}

numi::matter::CompiledWorld compileAttachedWorld(
    const std::uint32_t attachmentBodyIndex = 0u
) {
    const auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "Matter material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = kEnvironmentCount;
    source.frameTimestep = 1.0 / 240.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.articulatedDofCapacity = kDofs;
    source.articulatedQCapacity = kQ;
    source.materials.push_back(parsed.material);
    numi::matter::ObjectSource object;
    object.name = "numanx_adapter_attached_fem";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.characteristicLength = 0.01;
    object.mixedFEM = false;
    object.femNodes = {
        {-0.005, -0.005, -0.005}, {0.005, -0.005, -0.005},
        {-0.005, 0.005, -0.005}, {-0.005, -0.005, 0.005},
    };
    object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    numi::matter::FEMHumanAttachmentSource attachment;
    attachment.node = 0u;
    attachment.bodyIndex = attachmentBodyIndex;
    attachment.stableIdentifier = 0x4e584831u;
    attachment.localPoint = {-0.005, -0.005, -0.005};
    object.femHumanAttachments.push_back(attachment);
    source.objects.push_back(std::move(object));
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "attached Matter world did not compile");
    std::string error;
    require(numi::matter::validateCompiledWorldLayout(compiled.world, &error),
        "attached Matter layout failed: " + error);
    return std::move(compiled.world);
}

struct ExactCandidateService {
    id<MTLBuffer> q = nil;
    id<MTLBuffer> body = nil;
    id<MTLBuffer> jacobian = nil;
    std::uint64_t calls = 0u;
};

bool copy(
    id<MTLBlitCommandEncoder> blit,
    id<MTLBuffer> source,
    void* rawDestination,
    const NSUInteger bytes
) {
    id<MTLBuffer> destination = rawDestination == nullptr
        ? nil : (__bridge id<MTLBuffer>)rawDestination;
    if (blit == nil || source == nil || destination == nil ||
        source.length < bytes || destination.length < bytes ||
        source.device.registryID != destination.device.registryID) return false;
    [blit copyFromBuffer:source sourceOffset:0u
                toBuffer:destination destinationOffset:0u size:bytes];
    return true;
}

bool exactCandidate(
    void* context,
    const metalrobo::MetalNumanXHumanMatterPass& pass,
    const metalrobo::MetalNumanXHumanMatterCandidateQuery& query
) noexcept {
    auto* service = static_cast<ExactCandidateService*>(context);
    if (service == nullptr || query.deltaVelocity == nullptr ||
        query.deltaVelocityStride != kDofs || query.candidateQStride != kQ ||
        query.candidateBodyStride < kBodies ||
        query.substepIndex != pass.substepIndex ||
        query.transactionSlot != pass.transactionSlot ||
        query.physicsSubstepCount != pass.physicsSubstepCount ||
        query.controlStep != pass.controlStep ||
        pass.phase !=
            metalrobo::MetalNumanXHumanMatterPhase::preDynamics) return false;
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit == nil) return false;
    bool valid = copy(blit, service->q, query.candidateQ,
                      kQ * sizeof(float)) &&
        copy(blit, service->body, query.candidateBodies,
             query.candidateBodyStride * sizeof(MRBodyStateGPU));
    if (valid && query.pointCount != 0u) {
        valid = query.pointCount == kPoints && query.pointWorld == nullptr &&
            query.pointWorldGPUAddress == 0u &&
            query.pointWorldStride == 0u &&
            query.pointJacobianStride == 3u * kDofs &&
            copy(blit, service->jacobian, query.pointJacobians,
                 3u * kDofs * sizeof(float));
    }
    [blit endEncoding];
    if (valid) ++service->calls;
    return valid;
}

struct OwnerArenas {
    id<MTLBuffer> q = nil;
    id<MTLBuffer> v = nil;
    id<MTLBuffer> mujoco = nil;
    id<MTLBuffer> forces = nil;
    id<MTLBuffer> poses = nil;
    id<MTLBuffer> points = nil;
    id<MTLBuffer> pointWorld = nil;
    id<MTLBuffer> jacobian = nil;
    id<MTLBuffer> stand = nil;
    id<MTLBuffer> qCheckpoint = nil;
    id<MTLBuffer> vCheckpoint = nil;
    id<MTLBuffer> mujocoCheckpoint = nil;
    id<MTLBuffer> factor = nil;
    id<MTLBuffer> ownerStatus = nil;
};

OwnerArenas makeOwnerArenas(id<MTLDevice> device) {
    OwnerArenas result;
    std::array<float, kQ> q{};
    q[6] = 1.0f;
    std::array<float, kDofs> v{};
    std::vector<float> factor(kDofs * kDofs, 0.0f);
    for (std::uint32_t index = 0u; index < kDofs; ++index) {
        factor[index * kDofs + index] = 1.0f;
    }
    MRArticulatedBodyPoseGPU pose{};
    pose.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    MRArticulatedPointImpulseGPU point{};
    point.bodyIndex = 0u;
    MRNumiHumanStandStatusGPU stand{};
    stand.code = MR_NUMI_HUMAN_STAND_SUCCESS;
    stand.environment = 0u;
    stand.completedSteps = 1u;
    result.q = makeBuffer(device, q, @"owner q");
    result.v = makeBuffer(device, v, @"owner v");
    result.mujoco = makeZeroBuffer(
        device, sizeof(MRMujocoMuscleStateGPU), @"owner MyoSim");
    result.forces = makeZeroBuffer(
        device, kDofs * sizeof(float), @"owner generalized force");
    result.poses = makeBuffer(device, pose, @"owner body pose");
    result.points = makeBuffer(device, point, @"owner point query");
    result.pointWorld = makeZeroBuffer(
        device, sizeof(MRArticulatedPointWorldGPU), @"owner point world");
    result.jacobian = makeZeroBuffer(
        device, 3u * kDofs * sizeof(float), @"owner point Jacobian");
    result.stand = makeBuffer(device, stand, @"owner stand status");
    result.qCheckpoint = makeBuffer(device, q, @"owner q checkpoint");
    result.vCheckpoint = makeBuffer(device, v, @"owner v checkpoint");
    result.mujocoCheckpoint = makeZeroBuffer(
        device, sizeof(MRMujocoMuscleStateGPU), @"owner MyoSim checkpoint");
    result.factor = makeBuffer<float>(
        device, std::span<const float>(factor), @"owner frozen A0 factor");
    result.ownerStatus = makeZeroBuffer(
        device, sizeof(MRNumanXHumanMatterOwnerStatusGPU),
        @"owner transaction status");
    return result;
}

metalrobo::MetalNumanXHumanMatterPass makePass(
    const metalrobo::MetalNumanXHumanMatterProgram& program,
    const OwnerArenas& arena,
    ExactCandidateService& exact,
    id<MTLCommandBuffer> commandBuffer,
    const metalrobo::MetalNumanXHumanMatterPhase phase,
    const float timestep
) {
    using namespace metalrobo;
    MetalNumanXHumanMatterPass pass;
    pass.accessFlags = program.accessFlags;
    pass.capabilities = program.capabilities;
    pass.phase = phase;
    pass.stepIndex = 0u;
    pass.stepCount = 1u;
    pass.transactionSlot = program.transactionSlot;
    pass.substepIndex = program.substepIndex;
    pass.physicsSubstepCount = program.physicsSubstepCount;
    pass.controlStep = program.controlStep;
    pass.commandBuffer = (__bridge void*)commandBuffer;
    pass.q = (__bridge void*)arena.q;
    pass.v = (__bridge void*)arena.v;
    pass.mujocoStates = (__bridge void*)arena.mujoco;
    pass.mujocoGeneralizedForceArena = (__bridge void*)arena.forces;
    pass.bodyPoses = (__bridge void*)arena.poses;
    pass.pointQueries = (__bridge void*)arena.points;
    pass.pointWorld = (__bridge void*)arena.pointWorld;
    pass.pointJacobians = (__bridge void*)arena.jacobian;
    pass.standStatuses = (__bridge void*)arena.stand;
    pass.qCheckpoint = (__bridge void*)arena.qCheckpoint;
    pass.vCheckpoint = (__bridge void*)arena.vCheckpoint;
    pass.mujocoStateCheckpoint = (__bridge void*)arena.mujocoCheckpoint;
    pass.sourceEffectiveTangentFactor = (__bridge void*)arena.factor;
    pass.ownerStatuses = (__bridge void*)arena.ownerStatus;
    pass.matterGeneralizedReaction = program.matterGeneralizedReaction;
    pass.jointStatuses = program.jointStatuses;
    pass.acceptedPhysicsStateTokens = program.acceptedPhysicsStateTokens;
    if (phase ==
            metalrobo::MetalNumanXHumanMatterPhase::preDynamics) {
        pass.exactCandidateContext = &exact;
        pass.encodeExactCandidate = &exactCandidate;
    }
    pass.qGPUAddress = arena.q.gpuAddress;
    pass.vGPUAddress = arena.v.gpuAddress;
    pass.mujocoStatesGPUAddress = arena.mujoco.gpuAddress;
    pass.mujocoGeneralizedForceArenaGPUAddress = arena.forces.gpuAddress;
    pass.bodyPosesGPUAddress = arena.poses.gpuAddress;
    pass.pointQueriesGPUAddress = arena.points.gpuAddress;
    pass.pointWorldGPUAddress = arena.pointWorld.gpuAddress;
    pass.pointJacobiansGPUAddress = arena.jacobian.gpuAddress;
    pass.standStatusesGPUAddress = arena.stand.gpuAddress;
    pass.qCheckpointGPUAddress = arena.qCheckpoint.gpuAddress;
    pass.vCheckpointGPUAddress = arena.vCheckpoint.gpuAddress;
    pass.mujocoStateCheckpointGPUAddress = arena.mujocoCheckpoint.gpuAddress;
    pass.sourceEffectiveTangentFactorGPUAddress = arena.factor.gpuAddress;
    pass.ownerStatusesGPUAddress = arena.ownerStatus.gpuAddress;
    pass.matterGeneralizedReactionGPUAddress =
        program.matterGeneralizedReactionGPUAddress;
    pass.jointStatusesGPUAddress = program.jointStatusesGPUAddress;
    pass.acceptedPhysicsStateTokensGPUAddress =
        program.acceptedPhysicsStateTokensGPUAddress;
    pass.environmentCount = kEnvironmentCount;
    pass.qCoordinateCount = kQ;
    pass.dofCount = kDofs;
    pass.bodyCount = kBodies;
    pass.pointCount = kPoints;
    pass.mujocoStateCount = 1u;
    pass.qStride = kQ;
    pass.vStride = kDofs;
    pass.bodyPoseStride = kBodies;
    pass.pointStride = kPoints;
    pass.pointWorldStride = kPoints;
    pass.pointJacobianStride = 3u * kDofs;
    pass.mujocoStateStride = 1u;
    pass.factorStride = kDofs * kDofs;
    pass.generalizedForceOffset = 0u;
    pass.generalizedForceStride = kDofs;
    pass.generalizedForceArenaElementCount = kDofs;
    pass.reactionStride = program.reactionStride;
    pass.jointStatusStride = program.jointStatusStride;
    pass.acceptedTokenStrideBytes = program.acceptedTokenStrideBytes;
    pass.timestepSeconds = timestep;
    pass.articulationIndex = 0u;
    pass.articulationFirstBody = 0u;
    pass.programFingerprint = program.fingerprint;
    pass.transactionFingerprint = program.transactionFingerprint;
    pass.linearizationEpoch = program.linearizationEpoch;
    pass.slotGeneration = program.slotGeneration;
    return pass;
}

void verifyDistinctHeapAliasRejected(
    const metalrobo::MetalNumanXHumanMatterProgram& program,
    const OwnerArenas& arenas,
    ExactCandidateService& exact,
    id<MTLCommandQueue> queue,
    const float timestep
) {
    constexpr NSUInteger qBytes = kQ * sizeof(float);
    constexpr MTLResourceOptions options =
        MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
    const MTLSizeAndAlign sizeAndAlign =
        [queue.device heapBufferSizeAndAlignWithLength:qBytes
                                               options:options];
    require(sizeAndAlign.size >= qBytes && sizeAndAlign.align != 0u,
        "could not size alias-negative placement heap");
    MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    descriptor.size = sizeAndAlign.size;
    id<MTLHeap> heap = [queue.device newHeapWithDescriptor:descriptor];
    require(heap != nil, "could not allocate alias-negative placement heap");
    id<MTLBuffer> live = [heap newBufferWithLength:qBytes
                                           options:options
                                            offset:0u];
    require(live != nil, "could not allocate first overlapping heap buffer");
    [live makeAliasable];
    id<MTLBuffer> checkpoint = [heap newBufferWithLength:qBytes
                                                 options:options
                                                  offset:0u];
    require(checkpoint != nil && live != checkpoint &&
            live.gpuAddress != 0u &&
            live.gpuAddress == checkpoint.gpuAddress,
        "Metal did not produce distinct overlapping heap resources");

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    require(commandBuffer != nil,
        "failed to allocate alias-negative command buffer");
    auto pass = makePass(
        program, arenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::beginStep, timestep);
    pass.q = (__bridge void*)live;
    pass.qGPUAddress = live.gpuAddress;
    pass.qCheckpoint = (__bridge void*)checkpoint;
    pass.qCheckpointGPUAddress = checkpoint.gpuAddress;
    require(!program.encode(program.context, pass),
        "distinct MTLBuffers with overlapping GPU intervals were accepted");
}

void mixByte(std::uint64_t& hash, const std::uint8_t byte) {
    hash ^= byte;
    hash *= 1099511628211ull;
}

void mixU32(std::uint64_t& hash, const std::uint32_t value) {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        mixByte(hash, static_cast<std::uint8_t>(value >> (8u * byte)));
    }
}

void mixU64(std::uint64_t& hash, const std::uint64_t value) {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        mixByte(hash, static_cast<std::uint8_t>(value >> (8u * byte)));
    }
}

std::uint64_t tokenFingerprint(
    const MRNumanXAcceptedPhysicsStateTokenGPU& token
) {
    std::uint64_t hash = 14695981039346656037ull;
    mixU32(hash, 1u);
    mixU64(hash, token.transactionFingerprint);
    mixU64(hash, token.substepFingerprint);
    mixU64(hash, token.physicsStateFingerprint);
    mixU64(hash, token.acceptedTimestampMicroseconds);
    mixU64(hash, token.physicsGeneration);
    mixU32(hash, token.environmentIdentifier);
    mixU32(hash, token.flags);
    mixU64(hash, token.reserved);
    return hash;
}

template <typename T>
std::uint64_t recordFingerprint(const T& record) {
    static_assert(sizeof(T) == 128u);
    std::uint64_t hash = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&record);
    for (std::size_t index = 0u; index < 120u; ++index) {
        mixByte(hash, bytes[index]);
    }
    return hash == 0u ? 14695981039346656037ull : hash;
}

struct SensorPublicationService {
    enum class Stage : std::uint32_t {
        provisional,
        rootReserved,
        released,
        rejected,
        terminalNoTouch,
    };

    metalrobo::MetalNumanXHumanIOCandidatePublicationProgram program{};
    metalrobo::MetalNumanXHumanIOCandidatePublicationBinding binding{};
    Stage stage = Stage::provisional;
    std::uint32_t reserveCalls = 0u;
    std::uint32_t publishCalls = 0u;
    std::uint32_t rejectCalls = 0u;
    std::uint32_t terminalCalls = 0u;

    static bool reserve(
        void* opaque,
        const std::uint64_t candidatePublicationFingerprint,
        const metalrobo::MetalNumanXHumanIOCandidatePublicationBinding&
            candidateBinding
    ) noexcept {
        auto* service = static_cast<SensorPublicationService*>(opaque);
        if (service == nullptr || !service->program.valid() ||
            service->stage != Stage::provisional ||
            candidatePublicationFingerprint !=
                service->program.candidatePublicationFingerprint ||
            candidateBinding.abiVersion !=
                metalrobo::kMetalNumanXHumanIOPublicationABIVersion ||
            candidateBinding.structSize != sizeof(candidateBinding) ||
            candidateBinding.environmentCount != 1u ||
            candidateBinding.stepIndex != 0u ||
            candidateBinding.transactionFingerprint !=
                service->program.transactionFingerprint ||
            candidateBinding.candidateKeyFingerprint !=
                service->program.candidateKeyFingerprint ||
            candidateBinding.acceptedBrainGeneration !=
                service->program.acceptedBrainGeneration ||
            candidateBinding.sensorGeneration !=
                service->program.sensorGeneration ||
            candidateBinding.humanIOProgramFingerprint !=
                service->program.humanIOProgramFingerprint ||
            candidateBinding.sensorFingerprint !=
                service->program.sensorFingerprint ||
            candidateBinding.transactionInstanceFingerprint !=
                service->program.transactionInstanceFingerprint ||
            candidateBinding.candidatePublicationFingerprint !=
                service->program.candidatePublicationFingerprint ||
            candidateBinding.deviceRegistryID !=
                service->program.deviceRegistryID ||
            candidateBinding.humanIOIdentityFingerprint !=
                service->program.identityFingerprint ||
            candidateBinding.physicsTokenFingerprint == 0u ||
            candidateBinding.proposalFingerprint == 0u ||
            candidateBinding.ackFingerprint == 0u ||
            candidateBinding.appliedDecisionFingerprint == 0u ||
            candidateBinding.jointCommitFingerprint == 0u ||
            candidateBinding.brainGeneration == 0u ||
            candidateBinding.bindingFingerprint == 0u ||
            candidateBinding.bindingFingerprint != metalrobo::
                metalNumanXHumanIOPublicationBindingFingerprint(
                    candidateBinding)) {
            return false;
        }
        service->binding = candidateBinding;
        service->stage = Stage::rootReserved;
        ++service->reserveCalls;
        return true;
    }

    static metalrobo::MetalNumanXHumanIOCandidatePublicationDisposition
    publish(
        void* opaque,
        const std::uint64_t candidatePublicationFingerprint,
        const metalrobo::MetalNumanXHumanIOCandidatePublicationCommit& commit
    ) noexcept {
        using Disposition = metalrobo::
            MetalNumanXHumanIOCandidatePublicationDisposition;
        auto* service = static_cast<SensorPublicationService*>(opaque);
        if (service == nullptr) return Disposition::terminalNoTouch;
        const bool valid = service->program.valid() &&
            service->stage == Stage::rootReserved &&
            candidatePublicationFingerprint ==
                service->program.candidatePublicationFingerprint &&
            commit.abiVersion ==
                metalrobo::kMetalNumanXHumanIOPublicationABIVersion &&
            commit.structSize == sizeof(commit) &&
            commit.status == metalrobo::
                MetalNumanXHumanIOCandidatePublicationCommitStatus::
                    committed &&
            commit.reserved0 == 0u &&
            commit.candidatePublicationFingerprint ==
                service->program.candidatePublicationFingerprint &&
            commit.bindingFingerprint ==
                service->binding.bindingFingerprint &&
            commit.jointCommitFingerprint ==
                service->binding.jointCommitFingerprint &&
            commit.brainGeneration == service->binding.brainGeneration &&
            commit.fenceFingerprint != 0u;
        if (!valid) {
            service->stage = Stage::terminalNoTouch;
            ++service->terminalCalls;
            return Disposition::terminalNoTouch;
        }
        service->stage = Stage::released;
        ++service->publishCalls;
        return Disposition::released;
    }

    static metalrobo::MetalNumanXHumanIOCandidatePublicationDisposition
    reject(
        void* opaque,
        const std::uint64_t candidatePublicationFingerprint
    ) noexcept {
        using Disposition = metalrobo::
            MetalNumanXHumanIOCandidatePublicationDisposition;
        auto* service = static_cast<SensorPublicationService*>(opaque);
        if (service == nullptr) return Disposition::terminalNoTouch;
        if (service->program.valid() &&
            service->stage == Stage::provisional &&
            candidatePublicationFingerprint ==
                service->program.candidatePublicationFingerprint) {
            service->stage = Stage::rejected;
            ++service->rejectCalls;
            return Disposition::rejected;
        }
        service->stage = Stage::terminalNoTouch;
        ++service->terminalCalls;
        return Disposition::terminalNoTouch;
    }
};

std::unique_ptr<SensorPublicationService> makeSensorPublicationService(
    id<MTLDevice> device,
    const std::uint64_t transactionFingerprint,
    const std::uint64_t generation
) {
    require(device != nil && device.registryID != 0u &&
            transactionFingerprint != 0u && generation != 0u,
        "invalid HumanIO candidate service identity");
    auto result = std::make_unique<SensorPublicationService>();
    auto& program = result->program;
    program.context = result.get();
    program.reservePublishedRoot = &SensorPublicationService::reserve;
    program.publishCandidate = &SensorPublicationService::publish;
    program.rejectCandidate = &SensorPublicationService::reject;
    program.candidateKeyFingerprint = 0x48494f1000000000ull ^ generation;
    program.transactionFingerprint = transactionFingerprint;
    program.acceptedBrainGeneration = 0x48494f2000000000ull ^ generation;
    program.sensorGeneration = 0x48494f3000000000ull ^ generation;
    program.humanIOProgramFingerprint = 0x48494f4000000000ull ^ generation;
    program.sensorFingerprint = 0x48494f5000000000ull ^ generation;
    program.transactionInstanceFingerprint =
        0x48494f6000000000ull ^ generation;
    program.candidatePublicationFingerprint =
        0x48494f7000000000ull ^ generation;
    program.deviceRegistryID = device.registryID;
    program.identityFingerprint = program.computedIdentityFingerprint();
    require(program.valid(), "HumanIO candidate service is invalid");
    return result;
}

bool encodeRuntimeProof(
    void* context,
    const metalrobo::MetalNumanXHumanMatterStateProofPass& source
) noexcept {
    auto* runtime = static_cast<numi::matter::Runtime*>(context);
    if (runtime == nullptr ||
        source.abiVersion != MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION ||
        source.structSize !=
            sizeof(metalrobo::MetalNumanXHumanMatterStateProofPass)) {
        return false;
    }
    numi::matter::AcceptedStateProofPass pass{};
    pass.environmentCount = source.environmentCount;
    pass.environmentIdentifierBase = source.environmentIdentifierBase;
    pass.commandBuffer = source.commandBuffer;
    pass.q = source.q;
    pass.v = source.v;
    pass.mujocoStates = source.mujocoStates;
    pass.matterGeneralizedReaction = source.matterGeneralizedReaction;
    pass.environmentStatuses = source.environmentStatuses;
    pass.matterStatuses = source.matterStatuses;
    pass.acceptedStateProofs = source.acceptedStateProofs;
    pass.qGPUAddress = source.qGPUAddress;
    pass.vGPUAddress = source.vGPUAddress;
    pass.mujocoStatesGPUAddress = source.mujocoStatesGPUAddress;
    pass.matterGeneralizedReactionGPUAddress =
        source.matterGeneralizedReactionGPUAddress;
    pass.environmentStatusesGPUAddress =
        source.environmentStatusesGPUAddress;
    pass.matterStatusesGPUAddress = source.matterStatusesGPUAddress;
    pass.acceptedStateProofsGPUAddress =
        source.acceptedStateProofsGPUAddress;
    pass.qElementCount = source.qElementCount;
    pass.vElementCount = source.vElementCount;
    pass.mujocoStateCount = source.mujocoStateCount;
    pass.matterGeneralizedReactionElementCount =
        source.matterGeneralizedReactionElementCount;
    pass.environmentStatusElementCount =
        source.environmentStatusElementCount;
    pass.matterStatusElementCount = source.matterStatusElementCount;
    pass.acceptedStateProofElementCount =
        source.acceptedStateProofElementCount;
    pass.qStride = source.qStride;
    pass.vStride = source.vStride;
    pass.mujocoStateStride = source.mujocoStateStride;
    pass.reactionStride = source.reactionStride;
    pass.environmentStatusStride = source.environmentStatusStride;
    pass.matterStatusStride = source.matterStatusStride;
    pass.acceptedStateProofStride = source.acceptedStateProofStride;
    pass.qCoordinateCount = source.qCoordinateCount;
    pass.dofCount = source.dofCount;
    pass.transactionSlot = source.transactionSlot;
    pass.programFingerprint = source.programFingerprint;
    pass.stateProofProgramFingerprint =
        source.stateProofProgramFingerprint;
    pass.transactionFingerprint = source.transactionFingerprint;
    pass.substepFingerprint = source.substepFingerprint;
    pass.acceptedTimestampMicroseconds =
        source.acceptedTimestampMicroseconds;
    pass.physicsGeneration = source.physicsGeneration;
    pass.linearizationEpoch = source.linearizationEpoch;
    pass.slotGeneration = source.slotGeneration;
    pass.matterSourcePhysicsFingerprint =
        source.matterSourcePhysicsFingerprint;
    pass.matterDeviceProgramFingerprint =
        source.matterDeviceProgramFingerprint;
    return runtime->encodeAcceptedStateProof(pass);
}

template <typename T>
bool equalBytes(const std::vector<T>& left, const std::vector<T>& right) {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(
            left.data(), right.data(), left.size() * sizeof(T)) == 0);
}

bool equalAcceptedAuthority(
    const numi::matter::RuntimeStateSnapshot& left,
    const numi::matter::RuntimeStateSnapshot& right
) {
    return left.available && right.available &&
        equalBytes(left.particles, right.particles) &&
        equalBytes(left.femNodes, right.femNodes) &&
        equalBytes(left.femFields, right.femFields) &&
        equalBytes(left.femTopologyNodes, right.femTopologyNodes) &&
        equalBytes(left.femTopologyTetrahedra,
                   right.femTopologyTetrahedra) &&
        equalBytes(left.cohesiveFaces, right.cohesiveFaces) &&
        equalBytes(left.punctureChannels, right.punctureChannels) &&
        equalBytes(left.topologyStates, right.topologyStates) &&
        equalBytes(left.rigidGeneralizedCandidate,
                   right.rigidGeneralizedCandidate) &&
        equalBytes(left.learnedWeights, right.learnedWeights) &&
        left.learnedWeightRevision == right.learnedWeightRevision &&
        equalBytes(left.adaptive, right.adaptive) &&
        equalBytes(left.schedulers, right.schedulers) &&
        equalBytes(left.reactions, right.reactions) &&
        equalBytes(left.rigidStates, right.rigidStates) &&
        equalBytes(left.contactHistories, right.contactHistories) &&
        equalBytes(left.deformableContactHistories,
                   right.deformableContactHistories);
}

void checkpointHuman(const OwnerArenas& arenas) {
    std::memcpy(arenas.qCheckpoint.contents, arenas.q.contents,
                arenas.q.length);
    std::memcpy(arenas.vCheckpoint.contents, arenas.v.contents,
                arenas.v.length);
    std::memcpy(arenas.mujocoCheckpoint.contents, arenas.mujoco.contents,
                arenas.mujoco.length);
    std::memset(arenas.forces.contents, 0, arenas.forces.length);
}

std::vector<std::uint8_t> ownerAuthorityBytes(
    const OwnerArenas& arenas
) {
    const std::array<id<MTLBuffer>, 14u> buffers{{
        arenas.q, arenas.v, arenas.mujoco, arenas.forces, arenas.poses,
        arenas.points, arenas.pointWorld, arenas.jacobian, arenas.stand,
        arenas.qCheckpoint, arenas.vCheckpoint, arenas.mujocoCheckpoint,
        arenas.factor, arenas.ownerStatus,
    }};
    std::vector<std::uint8_t> result;
    for (id<MTLBuffer> buffer : buffers) {
        require(buffer != nil && buffer.contents != nullptr,
            "owner authority snapshot requires host-visible buffers");
        const auto* begin = static_cast<const std::uint8_t*>(buffer.contents);
        result.insert(result.end(), begin, begin + buffer.length);
    }
    return result;
}

void encodeHumanMutation(
    id<MTLDevice> device,
    id<MTLCommandBuffer> commandBuffer,
    const OwnerArenas& arenas,
    const std::uint64_t generation
) {
    std::array<float, kQ> q{};
    std::array<float, kDofs> v{};
    MRMujocoMuscleStateGPU mujoco{};
    std::memcpy(q.data(), arenas.q.contents, sizeof(q));
    std::memcpy(v.data(), arenas.v.contents, sizeof(v));
    std::memcpy(&mujoco, arenas.mujoco.contents, sizeof(mujoco));
    q[0] += 0.001f * static_cast<float>(generation);
    v[0] += 0.002f * static_cast<float>(generation);
    mujoco.excitationAndActivation.x +=
        0.01f * static_cast<float>(generation);
    id<MTLBuffer> qSource = makeBuffer(device, q, @"candidate Human q");
    id<MTLBuffer> vSource = makeBuffer(device, v, @"candidate Human v");
    id<MTLBuffer> mujocoSource = makeBuffer(
        device, mujoco, @"candidate Human MyoSim");
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    require(blit != nil &&
            copy(blit, qSource, (__bridge void*)arenas.q, arenas.q.length) &&
            copy(blit, vSource, (__bridge void*)arenas.v, arenas.v.length) &&
            copy(blit, mujocoSource, (__bridge void*)arenas.mujoco,
                 arenas.mujoco.length),
        "failed to encode candidate Human mutation");
    [blit endEncoding];
}

void encodeHumanStandFailure(
    id<MTLDevice> device,
    id<MTLCommandBuffer> commandBuffer,
    const OwnerArenas& arenas
) {
    MRNumiHumanStandStatusGPU failure{};
    failure.code = MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED;
    failure.environment = 0u;
    failure.completedSteps = 0u;
    failure.failingIndex = 7u;
    id<MTLBuffer> source = makeBuffer(
        device, failure, @"failed Human stand result");
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    require(blit != nil && copy(
            blit, source, (__bridge void*)arenas.stand,
            sizeof(MRNumiHumanStandStatusGPU)),
        "failed to encode the authoritative Human stand failure");
    [blit endEncoding];
}

metalrobo::MetalNumanXHumanMatterPrepareLease makeLease(
    const metalrobo::MetalNumanXHumanMatterProgram& program,
    id<MTLBuffer> proposals,
    id<MTLBuffer> proposedTokens,
    id<MTLBuffer> applyActions,
    id<MTLBuffer> appliedOutcomes,
    id<MTLBuffer> finalTokens,
    id<MTLBuffer> publicationFences,
    id<MTLSharedEvent> event,
    const std::uint64_t eventValue
) {
    metalrobo::MetalNumanXHumanMatterPrepareLease lease{};
    lease.environmentCount = 1u;
    lease.transactionSlot = program.transactionSlot;
    lease.stepIndex = 0u;
    lease.substepIndex = program.substepIndex;
    lease.preparedTokenStrideBytes = program.acceptedTokenStrideBytes;
    lease.proposalStride = 1u;
    lease.proposedTokenStrideBytes =
        MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
    lease.applyActionStride = 1u;
    lease.matterApplyOutcomeStride = program.matterApplyOutcomeStride;
    lease.appliedOutcomeStride = 1u;
    lease.finalTokenStrideBytes = MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
    lease.publicationFenceStride = 1u;
    lease.physicsSubstepCount = program.physicsSubstepCount;
    lease.qCoordinateCount = program.qCoordinateCount;
    lease.dofCount = program.dofCount;
    lease.dofLayoutVersion = program.dofLayoutVersion;
    lease.preparedPhysicsStateTokens = program.acceptedPhysicsStateTokens;
    lease.proposals = (__bridge void*)proposals;
    lease.proposedPhysicsStateTokens = (__bridge void*)proposedTokens;
    lease.applyActions = (__bridge void*)applyActions;
    lease.matterApplyOutcomes = program.matterApplyOutcomes;
    lease.appliedOutcomes = (__bridge void*)appliedOutcomes;
    lease.finalAcceptedPhysicsStateTokens = (__bridge void*)finalTokens;
    lease.publicationFences = (__bridge void*)publicationFences;
    lease.physicalPreparedEvent = (__bridge void*)event;
    lease.preparedPhysicsStateTokensGPUAddress =
        program.acceptedPhysicsStateTokensGPUAddress;
    lease.proposalsGPUAddress = proposals.gpuAddress;
    lease.proposedPhysicsStateTokensGPUAddress = proposedTokens.gpuAddress;
    lease.applyActionsGPUAddress = applyActions.gpuAddress;
    lease.matterApplyOutcomesGPUAddress =
        program.matterApplyOutcomesGPUAddress;
    lease.appliedOutcomesGPUAddress = appliedOutcomes.gpuAddress;
    lease.finalAcceptedPhysicsStateTokensGPUAddress = finalTokens.gpuAddress;
    lease.publicationFencesGPUAddress = publicationFences.gpuAddress;
    lease.preparedPhysicsStateTokenByteCount =
        program.acceptedPhysicsStateTokenByteCount;
    lease.proposalElementCount = 1u;
    lease.proposedPhysicsStateTokenByteCount =
        MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
    lease.applyActionElementCount = 1u;
    lease.matterApplyOutcomeElementCount =
        program.matterApplyOutcomeElementCount;
    lease.appliedOutcomeElementCount = 1u;
    lease.finalAcceptedPhysicsStateTokenByteCount =
        MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
    lease.publicationFenceElementCount = 1u;
    lease.physicalPreparedEventValue = eventValue;
    lease.proposalEventValue = eventValue + 1u;
    lease.appliedEventValue = eventValue + 2u;
    lease.controlStep = program.controlStep;
    lease.programFingerprint = program.fingerprint;
    lease.transactionFingerprint = program.transactionFingerprint;
    lease.linearizationEpoch = program.linearizationEpoch;
    lease.slotGeneration = program.slotGeneration;
    return lease;
}

numi::matter::PreparedStateDispositionIdentity dispositionIdentity(
    const metalrobo::MetalNumanXHumanMatterTransaction& transaction,
    const metalrobo::MetalNumanXHumanMatterProgram& program
) {
    numi::matter::PreparedStateDispositionIdentity identity{};
    identity.controlStep = transaction.controlStep;
    identity.physicsSubstep = transaction.physicsSubstep;
    identity.physicsSubstepCount = transaction.physicsSubsteps;
    identity.transactionSlot = transaction.transactionSlot;
    identity.ownerProgramFingerprint = program.fingerprint;
    identity.transactionFingerprint = transaction.transactionFingerprint;
    identity.linearizationEpoch = transaction.linearizationEpoch;
    identity.slotGeneration = transaction.slotGeneration;
    return identity;
}

struct PreparedTransaction {
    metalrobo::MetalNumanXHumanMatterTransaction transaction{};
    metalrobo::MetalNumanXHumanMatterProgram program{};
    metalrobo::MetalNumanXHumanMatterPrepareLease lease{};
    numi::matter::RuntimeStateSnapshot before{};
    id<MTLBuffer> proposals = nil;
    id<MTLBuffer> proposedTokens = nil;
    id<MTLBuffer> applyActions = nil;
    id<MTLBuffer> appliedOutcomes = nil;
    id<MTLBuffer> finalTokens = nil;
    id<MTLBuffer> publicationFences = nil;
    id<MTLSharedEvent> event = nil;
    // Borrowed Brain application resources remain strongly owned through the
    // exact terminal disposition; terminal quarantine never permits Metal to
    // recycle their objects or GPU intervals into another slot.
    id<MTLBuffer> brainPreflights = nil;
    id<MTLSharedEvent> preflightEvent = nil;
    id<MTLBuffer> brainAcks = nil;
    id<MTLSharedEvent> brainAckEvent = nil;
    std::unique_ptr<SensorPublicationService> humanIO;
    MRNumanXCoupledHumanStatusGPU joint{};
    MRNumanXAcceptedPhysicsStateTokenGPU preparedToken{};
    std::uintptr_t physicalCommandBufferAddress = 0u;
    bool applyReusedPhysicalCommandBufferAddress = false;
    bool physicalRejected = false;
    std::uint64_t exactCandidateCalls = 0u;
    std::uint32_t reactionConsumptions = 0u;
};

void verifyFirstCommandAbort(
    metalrobo::MetalNumanXHumanMatterContext& adapter,
    numi::matter::Runtime& matter,
    id<MTLCommandQueue> queue,
    const OwnerArenas& arenas,
    ExactCandidateService& exact
) {
    const auto before = matter.snapshot();
    require(before.available,
        "Matter accepted authority was unavailable before abort probe");
    metalrobo::MetalNumanXHumanMatterTransaction transaction{};
    transaction.environmentCount = 1u;
    transaction.transactionSlot = 0u;
    transaction.controlStep = 36u;
    transaction.physicsSubstep = 0u;
    transaction.physicsSubsteps = 1u;
    transaction.expectedMatterCompletedMicrosteps = 1u;
    transaction.seed = 0x8fffu;
    transaction.transactionFingerprint = 0xff01u;
    transaction.substepFingerprint = 0xff02u;
    transaction.acceptedTimestampMicroseconds = 999999u;
    transaction.physicsGeneration = 1u;
    transaction.linearizationEpoch = 0xff03u;
    transaction.slotGeneration = 1u;
    const auto program = adapter.program(transaction);
    require(program.valid(), "first-command abort program is invalid");
    id<MTLBuffer> proposals = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_PROPOSAL_BYTES,
        @"abort owner proposal");
    id<MTLBuffer> proposedTokens = makeZeroBuffer(
        queue.device, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
        @"abort proposed token");
    id<MTLBuffer> applyActions = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_APPLY_ACTION_BYTES,
        @"abort apply action");
    id<MTLBuffer> appliedOutcomes = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_APPLIED_OUTCOME_BYTES,
        @"abort applied outcome");
    id<MTLBuffer> finalTokens = makeZeroBuffer(
        queue.device, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
        @"abort owner final token");
    id<MTLBuffer> publicationFences = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES,
        @"abort publication fence");
    id<MTLSharedEvent> event = [queue.device newSharedEvent];
    require(event != nil, "failed to allocate abort lease event");
    const auto lease = makeLease(
        program, proposals, proposedTokens, applyActions, appliedOutcomes,
        finalTokens, publicationFences, event, 1u);
    require(program.acquirePrepareLease(program.context, lease),
        "first-command abort lease was rejected");

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    require(commandBuffer != nil,
        "failed to allocate first-command abort CB");
    const auto begin = makePass(
        program, arenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::beginStep,
        matter.timestepSeconds());
    const auto pre = makePass(
        program, arenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::preDynamics,
        matter.timestepSeconds());
    require(program.encode(program.context, begin),
        "first-command abort probe did not begin");
    require(program.encode(program.context, pre),
        "first-command abort probe did not open real Matter preDynamics");
    require(program.releasePrepareLease(
                program.context, lease, nullptr, false) ==
                metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
                    released,
        "unqueued first-command unwind did not release scalar lease state");
    program.abort(program.context, (__bridge void*)commandBuffer);
    const auto after = matter.snapshot();
    require(equalAcceptedAuthority(before, after),
        "first-command abort changed accepted Matter authority");
}

void verifyCrossSlotAuthorityRejected(
    metalrobo::MetalNumanXHumanMatterContext& adapter,
    numi::matter::Runtime& matter,
    id<MTLCommandQueue> queue,
    const OwnerArenas& retainedArenas,
    ExactCandidateService& exact,
    const std::uint64_t generation,
    const std::uint32_t controlStep,
    const std::uint32_t transactionSlot
) {
    const auto beforeMatter = matter.snapshot();
    const auto beforeOwner = ownerAuthorityBytes(retainedArenas);
    require(beforeMatter.available,
        "Matter accepted authority was unavailable before cross-slot alias probe");

    metalrobo::MetalNumanXHumanMatterTransaction transaction{};
    transaction.environmentCount = 1u;
    transaction.transactionSlot = transactionSlot;
    transaction.controlStep = controlStep;
    transaction.physicsSubstep = 0u;
    transaction.physicsSubsteps = 1u;
    transaction.expectedMatterCompletedMicrosteps = 1u;
    transaction.seed = 0xa000u + generation;
    transaction.transactionFingerprint = 0xa100u + generation;
    transaction.substepFingerprint = 0xa200u + generation;
    transaction.acceptedTimestampMicroseconds = 2000000u + generation;
    transaction.physicsGeneration = generation;
    transaction.linearizationEpoch = 0xa300u + generation;
    transaction.slotGeneration = generation;
    const auto program = adapter.program(transaction);
    require(program.valid(),
        "cross-slot alias probe could not acquire a fresh slot program");

    id<MTLBuffer> proposals = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_PROPOSAL_BYTES,
        @"cross-slot owner proposal");
    id<MTLBuffer> proposedTokens = makeZeroBuffer(
        queue.device, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
        @"cross-slot proposed token");
    id<MTLBuffer> applyActions = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_APPLY_ACTION_BYTES,
        @"cross-slot apply action");
    id<MTLBuffer> appliedOutcomes = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_APPLIED_OUTCOME_BYTES,
        @"cross-slot applied outcome");
    id<MTLBuffer> finalTokens = makeZeroBuffer(
        queue.device, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
        @"cross-slot final token");
    id<MTLBuffer> publicationFences = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES,
        @"cross-slot publication fence");
    id<MTLSharedEvent> event = [queue.device newSharedEvent];
    require(event != nil, "failed to allocate cross-slot lease event");
    const auto lease = makeLease(
        program, proposals, proposedTokens, applyActions, appliedOutcomes,
        finalTokens, publicationFences, event, generation * 4u + 1u);

    auto aliasedLease = lease;
    aliasedLease.proposals = (__bridge void*)retainedArenas.q;
    aliasedLease.proposalsGPUAddress = retainedArenas.q.gpuAddress;
    require(!program.acquirePrepareLease(program.context, aliasedLease),
        "slot-1 lease aliased terminal slot-0 physical authority");
    require(program.acquirePrepareLease(program.context, lease),
        "disjoint cross-slot control lease was rejected");

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    require(commandBuffer != nil,
        "failed to allocate cross-slot physical alias command buffer");
    const auto begin = makePass(
        program, retainedArenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::beginStep,
        matter.timestepSeconds());
    require(!program.encode(program.context, begin),
        "slot-1 physical pass reused terminal slot-0 authority");
    require(program.releasePrepareLease(
                program.context, lease, nullptr, false) ==
                metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
                    released,
        "rejected cross-slot physical alias did not release untouched slot-1 state");
    require(equalAcceptedAuthority(beforeMatter, matter.snapshot()) &&
            beforeOwner == ownerAuthorityBytes(retainedArenas),
        "cross-slot alias rejection mutated retained authority");
}

PreparedTransaction prepareTransaction(
    metalrobo::MetalNumanXHumanMatterContext& adapter,
    numi::matter::Runtime& matter,
    id<MTLCommandQueue> queue,
    const OwnerArenas& arenas,
    ExactCandidateService& exact,
    const std::uint64_t generation,
    const std::uint32_t controlStep,
    const bool exerciseMalformed,
    const bool physicalReject = false,
    const std::uint32_t transactionSlot = 0u
) {
    PreparedTransaction result;
    result.before = matter.snapshot();
    require(result.before.available,
        "Matter accepted authority was not idle before prepare");
    checkpointHuman(arenas);
    MRNumiHumanStandStatusGPU initialStand{};
    initialStand.code = MR_NUMI_HUMAN_STAND_SUCCESS;
    initialStand.environment = 0u;
    initialStand.completedSteps = 1u;
    std::memcpy(
        arenas.stand.contents, &initialStand, sizeof(initialStand));
    result.physicalRejected = physicalReject;
    result.transaction.environmentCount = 1u;
    result.transaction.transactionSlot = transactionSlot;
    result.transaction.controlStep = controlStep;
    result.transaction.physicsSubstep = 0u;
    result.transaction.physicsSubsteps = 1u;
    result.transaction.expectedMatterCompletedMicrosteps = 1u;
    result.transaction.seed = 0x9000u + generation;
    result.transaction.transactionFingerprint = 0x1000u + generation;
    result.transaction.substepFingerprint = 0x2000u + generation;
    result.transaction.acceptedTimestampMicroseconds = 1000000u + generation;
    result.transaction.physicsGeneration = generation;
    result.transaction.linearizationEpoch = 0x3000u + generation;
    result.transaction.slotGeneration = generation;
    result.program = adapter.program(result.transaction);
    require(result.program.valid(),
        "adapter rejected a fresh prepared-state program");

    result.proposals = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_PROPOSAL_BYTES,
        @"owner proposal");
    result.proposedTokens = makeZeroBuffer(
        queue.device, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
        @"owner immutable proposed token");
    result.applyActions = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_APPLY_ACTION_BYTES,
        @"owner apply action");
    result.appliedOutcomes = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_APPLIED_OUTCOME_BYTES,
        @"owner applied outcome");
    result.finalTokens = makeZeroBuffer(
        queue.device, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
        @"owner final accepted token");
    result.publicationFences = makeZeroBuffer(
        queue.device, MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES,
        @"owner publication fence");
    result.event = [queue.device newSharedEvent];
    require(result.event != nil, "failed to allocate owner shared event");
    result.lease = makeLease(
        result.program, result.proposals, result.proposedTokens,
        result.applyActions, result.appliedOutcomes, result.finalTokens,
        result.publicationFences, result.event, generation * 4u + 1u);

    if (exerciseMalformed) {
        auto malformed = result.lease;
        malformed.transactionSlot = 1u;
        require(!result.program.acquirePrepareLease(
            result.program.context, malformed),
            "wrong-slot prepare lease was accepted");
        malformed = result.lease;
        ++malformed.slotGeneration;
        require(!result.program.acquirePrepareLease(
            result.program.context, malformed),
            "stale-generation prepare lease was accepted");
        malformed = result.lease;
        ++malformed.controlStep;
        require(!result.program.acquirePrepareLease(
            result.program.context, malformed),
            "wrong-control-step prepare lease was accepted");
        malformed = result.lease;
        ++malformed.proposalsGPUAddress;
        require(!result.program.acquirePrepareLease(
            result.program.context, malformed),
            "wrong-address prepare lease was accepted");
    }
    require(result.program.acquirePrepareLease(
                result.program.context, result.lease),
        "exact owner prepare lease was rejected");
    require(!result.program.acquirePrepareLease(
                result.program.context, result.lease),
        "duplicate owner prepare lease was accepted");

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    require(commandBuffer != nil, "failed to create physical prepare CB");
    result.physicalCommandBufferAddress =
        reinterpret_cast<std::uintptr_t>((__bridge void*)commandBuffer);
    exact.calls = 0u;
    if (exerciseMalformed) {
        verifyDistinctHeapAliasRejected(
            result.program, arenas, exact, queue, matter.timestepSeconds());
    }
    auto pre = makePass(
        result.program, arenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::preDynamics,
        matter.timestepSeconds());
    require(!result.program.encode(result.program.context, pre),
        "preDynamics was accepted before beginStep");
    auto begin = makePass(
        result.program, arenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::beginStep,
        matter.timestepSeconds());
    auto wrongControl = begin;
    ++wrongControl.controlStep;
    require(!result.program.encode(result.program.context, wrongControl),
        "mismatched pass controlStep was accepted");
    require(result.program.encode(result.program.context, begin),
        "beginStep rejected real prepared Matter encoding");
    require(!result.program.encode(result.program.context, begin),
        "duplicate beginStep was accepted");
    auto post = makePass(
        result.program, arenas, exact, commandBuffer,
        metalrobo::MetalNumanXHumanMatterPhase::postDynamics,
        matter.timestepSeconds());
    require(!result.program.encode(result.program.context, post),
        "postDynamics was accepted before preDynamics");
    require(result.program.encode(result.program.context, pre),
        "preDynamics rejected real prepared Matter encoding");
    require(!result.program.encode(result.program.context, pre),
        "duplicate preDynamics was accepted");
    id<MTLBuffer> preJointRead = makeZeroBuffer(
        queue.device, sizeof(MRNumanXCoupledHumanStatusGPU),
        @"preDynamics joint readback");
    id<MTLBlitCommandEncoder> preReadback =
        [commandBuffer blitCommandEncoder];
    require(preReadback != nil && copy(
            preReadback,
            (__bridge id<MTLBuffer>)result.program.jointStatuses,
            (__bridge void*)preJointRead,
            sizeof(MRNumanXCoupledHumanStatusGPU)),
        "failed to stage preDynamics joint status");
    [preReadback endEncoding];

    // The owner contract consumes the staged A0*dv/h force exactly once.
    id<MTLBlitCommandEncoder> consume = [commandBuffer blitCommandEncoder];
    require(consume != nil && copy(
        consume,
        (__bridge id<MTLBuffer>)result.program.matterGeneralizedReaction,
        (__bridge void*)arenas.forces,
        kDofs * sizeof(float)),
        "failed to consume staged generalized reaction");
    [consume endEncoding];
    ++result.reactionConsumptions;
    encodeHumanMutation(
        queue.device, commandBuffer, arenas, generation);
    if (physicalReject) {
        // This is the actual first-CB Human authority, not a host-fabricated
        // proposal. The same-CB adapter maps it into MRMetalWorldStatusGPU;
        // Matter's proof gate must therefore emit REJECTED/zero binding.
        encodeHumanStandFailure(queue.device, commandBuffer, arenas);
    }
    require(result.program.encode(result.program.context, post),
        "postDynamics rejected prepared proof/token encoding");
    require(!result.program.encode(result.program.context, post),
        "duplicate postDynamics was accepted");

    id<MTLBuffer> jointRead = makeZeroBuffer(
        queue.device, sizeof(MRNumanXCoupledHumanStatusGPU),
        @"prepared joint readback");
    id<MTLBuffer> tokenRead = makeZeroBuffer(
        queue.device, sizeof(MRNumanXAcceptedPhysicsStateTokenGPU),
        @"prepared token readback");
    id<MTLBlitCommandEncoder> readback =
        [commandBuffer blitCommandEncoder];
    require(readback != nil && copy(
            readback,
            (__bridge id<MTLBuffer>)result.program.jointStatuses,
            (__bridge void*)jointRead,
            sizeof(MRNumanXCoupledHumanStatusGPU)) && copy(
            readback,
            (__bridge id<MTLBuffer>)
                result.program.acceptedPhysicsStateTokens,
            (__bridge void*)tokenRead,
            sizeof(MRNumanXAcceptedPhysicsStateTokenGPU)),
        "failed to stage prepared result readback");
    [readback endEncoding];
    finish(commandBuffer);
    result.joint = value<MRNumanXCoupledHumanStatusGPU>(jointRead);
    const auto preJoint = value<MRNumanXCoupledHumanStatusGPU>(preJointRead);
    result.preparedToken =
        value<MRNumanXAcceptedPhysicsStateTokenGPU>(tokenRead);
    result.exactCandidateCalls = exact.calls;
    require(preJoint.decision == MR_NUMANX_COUPLED_HUMAN_PENDING &&
            preJoint.humanCode == MR_NUMI_HUMAN_STAND_SUCCESS &&
            preJoint.matterCode == NM_STATUS_SUCCESS &&
            preJoint.humanCompletedSteps == 0u &&
            preJoint.matterCompletedMicrosteps == 0u,
        "pre-publication status did not preserve exact SUCCESS/0 PENDING "
        "boundary: decision=" + std::to_string(preJoint.decision) +
            " humanCode=" + std::to_string(preJoint.humanCode) +
            " matterCode=" + std::to_string(preJoint.matterCode) +
            " humanSteps=" +
                std::to_string(preJoint.humanCompletedSteps) +
            " matterSteps=" +
                std::to_string(preJoint.matterCompletedMicrosteps));
    require((physicalReject
                ? result.joint.decision ==
                      MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN &&
                    result.joint.humanCode ==
                      MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED &&
                    result.joint.humanCompletedSteps == 0u
                : result.joint.decision ==
                      MR_NUMANX_COUPLED_HUMAN_ACCEPT) &&
            result.joint.matterCode == NM_STATUS_SUCCESS,
        "prepared Human/Matter joint candidate had the wrong decision: decision=" +
            std::to_string(result.joint.decision) +
            " humanCode=" + std::to_string(result.joint.humanCode) +
            " matterCode=" + std::to_string(result.joint.matterCode) +
            " humanSteps=" +
                std::to_string(result.joint.humanCompletedSteps) +
            " matterSteps=" +
                std::to_string(result.joint.matterCompletedMicrosteps) +
            " preDecision=" + std::to_string(preJoint.decision) +
            " preMatterCode=" + std::to_string(preJoint.matterCode) +
            " preMatterSteps=" +
                std::to_string(preJoint.matterCompletedMicrosteps));
    if (physicalReject) {
        const MRNumanXAcceptedPhysicsStateTokenGPU zero{};
        require(std::memcmp(
                    &result.preparedToken, &zero, sizeof(zero)) == 0,
            "physically rejected first CB produced a prepared token");
    } else {
        require(result.preparedToken.transactionFingerprint ==
                    result.transaction.transactionFingerprint &&
                result.preparedToken.substepFingerprint ==
                    result.transaction.substepFingerprint &&
                result.preparedToken.physicsStateFingerprint != 0u &&
                result.preparedToken.acceptedTimestampMicroseconds ==
                    result.transaction.acceptedTimestampMicroseconds &&
                result.preparedToken.physicsGeneration == generation &&
                result.preparedToken.reserved == 0u &&
                result.preparedToken.tokenFingerprint != 0u &&
                result.preparedToken.tokenFingerprint ==
                    tokenFingerprint(result.preparedToken),
            "prepared token violated the canonical NumiBrain/FNV relation");
    }
    require(result.exactCandidateCalls != 0u &&
            result.reactionConsumptions == 1u,
        "real coupled graph skipped an operation or duplicated reaction consumption");
    require(matter.preparedStateDisposition(dispositionIdentity(
                result.transaction, result.program)) ==
            numi::matter::PreparedStateDisposition::prepared,
        "Matter did not retain prepared authority after the first CB");

    result.humanIO = makeSensorPublicationService(
        queue.device, result.transaction.transactionFingerprint,
        generation);
    auto stagedLease = result.lease;
    stagedLease.humanIOCandidate = result.humanIO->program;
    if (exerciseMalformed) {
        auto wrongTransaction = result.humanIO->program;
        ++wrongTransaction.transactionFingerprint;
        wrongTransaction.identityFingerprint =
            wrongTransaction.computedIdentityFingerprint();
        auto wrongTransactionLease = result.lease;
        wrongTransactionLease.humanIOCandidate = wrongTransaction;
        require(!result.program.bindHumanIOCandidatePublication(
                    result.program.context, wrongTransactionLease,
                    wrongTransaction),
            "wrong-transaction HumanIO candidate was bound");

        auto wrongDevice = result.humanIO->program;
        ++wrongDevice.deviceRegistryID;
        wrongDevice.identityFingerprint =
            wrongDevice.computedIdentityFingerprint();
        auto wrongDeviceLease = result.lease;
        wrongDeviceLease.humanIOCandidate = wrongDevice;
        require(!result.program.bindHumanIOCandidatePublication(
                    result.program.context, wrongDeviceLease, wrongDevice),
            "wrong-device HumanIO candidate was bound");

        require(!result.program.bindHumanIOCandidatePublication(
                    result.program.context, result.lease,
                    result.humanIO->program),
            "HumanIO candidate missing from staged lease was bound");
    }
    require(result.program.bindHumanIOCandidatePublication(
                result.program.context, stagedLease,
                result.humanIO->program),
        "exact post-physical HumanIO candidate was rejected");
    result.lease = stagedLease;
    require(!result.program.bindHumanIOCandidatePublication(
                result.program.context, result.lease,
                result.humanIO->program),
        "duplicate HumanIO candidate bind was accepted");
    return result;
}

enum class ApplyMode {
    accept,
    physicalReject,
    fabricatedPhysicalReject,
    fabricatedAcceptOnRejected,
    reject,
    pending,
};

bool physicalProposal(const ApplyMode mode) {
    return mode == ApplyMode::physicalReject ||
        mode == ApplyMode::fabricatedPhysicalReject;
}

bool invalidBindingCross(const ApplyMode mode) {
    return mode == ApplyMode::fabricatedPhysicalReject ||
        mode == ApplyMode::fabricatedAcceptOnRejected;
}

MRNumanXAcceptedPhysicsStateTokenGPU proposedTokenFor(
    const PreparedTransaction& prepared,
    const ApplyMode mode
) {
    if (mode != ApplyMode::fabricatedAcceptOnRejected) {
        return prepared.preparedToken;
    }
    MRNumanXAcceptedPhysicsStateTokenGPU token{};
    token.transactionFingerprint =
        prepared.transaction.transactionFingerprint;
    token.substepFingerprint = prepared.transaction.substepFingerprint;
    token.physicsStateFingerprint = 0xfa110001u;
    token.acceptedTimestampMicroseconds =
        prepared.transaction.acceptedTimestampMicroseconds;
    token.physicsGeneration = prepared.transaction.physicsGeneration;
    token.environmentIdentifier =
        prepared.transaction.environmentIdentifierBase;
    token.tokenFingerprint = tokenFingerprint(token);
    require(token.tokenFingerprint != 0u,
        "fabricated binding-cross token fingerprint is zero");
    return token;
}

MRNumanXHumanMatterProposalGPU makeProposal(
    const PreparedTransaction& prepared,
    const ApplyMode mode
) {
    const bool physicalReject = physicalProposal(mode);
    const auto proposedToken = proposedTokenFor(prepared, mode);
    MRNumanXHumanMatterProposalGPU proposal{};
    proposal.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    proposal.status = MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY;
    proposal.decision = physicalReject
        ? MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
        : MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    proposal.code = physicalReject
        ? MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT
        : MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS;
    proposal.programFingerprint = prepared.program.fingerprint;
    proposal.transactionFingerprint =
        prepared.program.transactionFingerprint;
    proposal.linearizationEpoch = prepared.program.linearizationEpoch;
    proposal.slotGeneration = prepared.program.slotGeneration;
    proposal.physicsTokenFingerprint = physicalReject
        ? 0u : proposedToken.tokenFingerprint;
    proposal.brainProgramFingerprint = physicalReject ? 0u : 0xb1000001u;
    proposal.brainShadowStateFingerprint = physicalReject ? 0u : 0xb1000002u;
    proposal.brainWitnessFingerprint = physicalReject ? 0u : 0xb1000003u;
    require(prepared.humanIO != nullptr &&
            prepared.humanIO->program.valid(),
        "proposal has no exact HumanIO candidate");
    proposal.candidatePublicationFingerprint =
        prepared.humanIO->program.candidatePublicationFingerprint;
    proposal.humanIOIdentityFingerprint =
        prepared.humanIO->program.identityFingerprint;
    proposal.environment = 0u;
    proposal.stepIndex = 0u;
    proposal.substepIndex = prepared.program.substepIndex;
    proposal.transactionSlot = prepared.program.transactionSlot;
    proposal.physicsSubstepCount = prepared.program.physicsSubstepCount;
    proposal.controlStep = prepared.program.controlStep;
    proposal.proposalFingerprint = recordFingerprint(proposal);
    return proposal;
}

MRNumanXHumanMatterBrainCommitPreflightGPU makePreflight(
    const PreparedTransaction& prepared,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const ApplyMode mode
) {
    if (physicalProposal(mode)) return {};
    MRNumanXHumanMatterBrainCommitPreflightGPU preflight{};
    preflight.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION;
    preflight.structBytes =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES;
    preflight.status = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS;
    preflight.environment = 0u;
    preflight.controlStep = prepared.program.controlStep;
    preflight.substepIndex = prepared.program.substepIndex;
    preflight.physicsSubstepCount = prepared.program.physicsSubstepCount;
    preflight.transactionSlot = prepared.program.transactionSlot;
    preflight.ownerProgramFingerprint = prepared.program.fingerprint;
    preflight.transactionFingerprint =
        prepared.program.transactionFingerprint;
    preflight.linearizationEpoch = prepared.program.linearizationEpoch;
    preflight.slotGeneration = prepared.program.slotGeneration;
    preflight.substepFingerprint = prepared.transaction.substepFingerprint;
    preflight.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    preflight.fastTargetGeneration = 1u;
    preflight.cognitiveTargetGeneration = 1u;
    preflight.jointReceiptFingerprint = 0xb1000010u;
    preflight.fastProgramFingerprint = 0xb1000011u;
    preflight.brainProgramFingerprint = proposal.brainProgramFingerprint;
    preflight.preflightFingerprint = recordFingerprint(preflight);
    return preflight;
}

MRNumanXHumanMatterBrainAckGPU makeAck(
    const PreparedTransaction& prepared,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterBrainCommitPreflightGPU& preflight,
    const ApplyMode mode
) {
    const bool physicalReject = physicalProposal(mode);
    MRNumanXHumanMatterBrainAckGPU ack{};
    ack.abiVersion = MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION;
    ack.status = mode == ApplyMode::pending
        ? MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PENDING
        : physicalReject
            ? MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_REJECT
            : MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ACCEPT;
    ack.decision = mode == ApplyMode::pending
        ? MR_NUMANX_HUMAN_MATTER_ROOT_PENDING
        : physicalReject
            ? MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
            : MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    ack.code = physicalReject
        ? MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_PROPOSAL_REJECT
        : MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_SUCCESS;
    ack.programFingerprint = prepared.program.fingerprint;
    ack.transactionFingerprint = prepared.program.transactionFingerprint;
    ack.linearizationEpoch = prepared.program.linearizationEpoch;
    ack.slotGeneration = prepared.program.slotGeneration;
    ack.physicsTokenFingerprint = physicalReject
        ? 0u : proposal.physicsTokenFingerprint;
    ack.proposalFingerprint = proposal.proposalFingerprint;
    ack.preflightFingerprint = physicalReject
        ? 0u : preflight.preflightFingerprint;
    ack.fastGateFingerprint = physicalReject ? 0u : 0xb1000020u;
    ack.brainWitnessFingerprint = physicalReject
        ? 0u : proposal.brainWitnessFingerprint;
    ack.brainProgramFingerprint = physicalReject
        ? 0xb1000001u : proposal.brainProgramFingerprint;
    ack.environment = 0u;
    ack.stepIndex = 0u;
    ack.substepIndex = prepared.program.substepIndex;
    ack.transactionSlot = prepared.program.transactionSlot;
    ack.physicsSubstepCount = prepared.program.physicsSubstepCount;
    ack.controlStep = prepared.program.controlStep;
    ack.ackFingerprint = recordFingerprint(ack);
    return ack;
}

MRNumanXHumanMatterApplyActionGPU makeAction(
    const PreparedTransaction& prepared,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterBrainAckGPU& ack,
    const ApplyMode mode
) {
    const bool physicalReject = physicalProposal(mode);
    const bool forcedReject = mode == ApplyMode::reject;
    MRNumanXHumanMatterApplyActionGPU action{};
    action.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    action.status = mode == ApplyMode::pending
        ? MR_NUMANX_HUMAN_MATTER_APPLY_PENDING
        : (physicalReject || forcedReject)
            ? MR_NUMANX_HUMAN_MATTER_APPLY_REJECT
            : MR_NUMANX_HUMAN_MATTER_APPLY_ACCEPT;
    action.decision = mode == ApplyMode::pending
        ? MR_NUMANX_HUMAN_MATTER_ROOT_PENDING
        : (physicalReject || forcedReject)
            ? MR_NUMANX_HUMAN_MATTER_ROOT_REJECT
            : MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT;
    action.code = physicalReject
        ? MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT
        : forcedReject
            ? MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT
            : MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS;
    action.programFingerprint = prepared.program.fingerprint;
    action.transactionFingerprint = prepared.program.transactionFingerprint;
    action.linearizationEpoch = prepared.program.linearizationEpoch;
    action.slotGeneration = prepared.program.slotGeneration;
    action.physicsTokenFingerprint = (physicalReject || forcedReject)
        ? 0u : proposal.physicsTokenFingerprint;
    action.proposalFingerprint = proposal.proposalFingerprint;
    action.ackFingerprint = forcedReject ? 0u : ack.ackFingerprint;
    action.preflightFingerprint = forcedReject
        ? 0u : ack.preflightFingerprint;
    action.fastGateFingerprint = forcedReject
        ? 0u : ack.fastGateFingerprint;
    action.brainWitnessFingerprint = forcedReject
        ? 0u : ack.brainWitnessFingerprint;
    action.environment = 0u;
    action.stepIndex = 0u;
    action.substepIndex = prepared.program.substepIndex;
    action.transactionSlot = prepared.program.transactionSlot;
    action.physicsSubstepCount = prepared.program.physicsSubstepCount;
    action.controlStep = prepared.program.controlStep;
    action.actionFingerprint = recordFingerprint(action);
    return action;
}

metalrobo::MetalNumanXHumanMatterProposalView proposalView(
    const PreparedTransaction& prepared
) {
    metalrobo::MetalNumanXHumanMatterProposalView view{};
    view.proposals = prepared.lease.proposals;
    view.proposedPhysicsStateTokens =
        prepared.lease.proposedPhysicsStateTokens;
    view.proposalsGPUAddress = prepared.lease.proposalsGPUAddress;
    view.proposedPhysicsStateTokensGPUAddress =
        prepared.lease.proposedPhysicsStateTokensGPUAddress;
    view.proposalElementCount = prepared.lease.proposalElementCount;
    view.proposedPhysicsStateTokenByteCount =
        prepared.lease.proposedPhysicsStateTokenByteCount;
    view.proposalStride = prepared.lease.proposalStride;
    view.proposedTokenStrideBytes = prepared.lease.proposedTokenStrideBytes;
    view.proposalEventValue = prepared.lease.proposalEventValue;
    return view;
}

metalrobo::MetalNumanXHumanMatterPublicationFenceView fenceView(
    const PreparedTransaction& prepared
) {
    metalrobo::MetalNumanXHumanMatterPublicationFenceView view{};
    view.publicationFences = prepared.lease.publicationFences;
    view.publicationFencesGPUAddress =
        prepared.lease.publicationFencesGPUAddress;
    view.publicationFenceElementCount =
        prepared.lease.publicationFenceElementCount;
    view.publicationFenceStride = prepared.lease.publicationFenceStride;
    view.environmentCount = 1u;
    view.transactionSlot = prepared.program.transactionSlot;
    view.stepIndex = 0u;
    view.substepIndex = prepared.program.substepIndex;
    view.physicsSubstepCount = prepared.program.physicsSubstepCount;
    view.controlStep = prepared.program.controlStep;
    view.programFingerprint = prepared.program.fingerprint;
    view.transactionFingerprint = prepared.program.transactionFingerprint;
    view.linearizationEpoch = prepared.program.linearizationEpoch;
    view.slotGeneration = prepared.program.slotGeneration;
    return view;
}

struct ApplyResult {
    MRNumanXAcceptedPhysicsStateTokenGPU token{};
    numi::matter::RuntimeStateSnapshot after{};
};

ApplyResult applyTransaction(
    numi::matter::Runtime& matter,
    id<MTLCommandQueue> queue,
    const OwnerArenas& arenas,
    PreparedTransaction& prepared,
    const ApplyMode mode,
    const bool abortAndRetry
) {
    const auto identity = dispositionIdentity(
        prepared.transaction, prepared.program);
    const auto proposedToken = proposedTokenFor(prepared, mode);
    const auto proposal = makeProposal(prepared, mode);
    const auto preflight = makePreflight(prepared, proposal, mode);
    std::memcpy(prepared.proposals.contents, &proposal, sizeof(proposal));
    if (physicalProposal(mode)) {
        std::memset(
            prepared.proposedTokens.contents,
            0,
            prepared.proposedTokens.length
        );
    } else {
        std::memcpy(
            prepared.proposedTokens.contents, &proposedToken,
            sizeof(proposedToken));
    }
    prepared.event.signaledValue = prepared.lease.proposalEventValue;
    prepared.brainPreflights = makeBuffer(
        queue.device, preflight, @"Brain commit preflight");
    prepared.preflightEvent = [queue.device newSharedEvent];
    require(prepared.preflightEvent != nil,
        "failed to allocate preflight event");
    prepared.preflightEvent.signaledValue = 1u;
    auto proposalResource = proposalView(prepared);
    metalrobo::MetalNumanXHumanMatterBrainPreflightView preflightView{};
    preflightView.brainCommitPreflights =
        (__bridge void*)prepared.brainPreflights;
    preflightView.preflightReadyEvent =
        (__bridge void*)prepared.preflightEvent;
    preflightView.brainCommitPreflightsGPUAddress =
        prepared.brainPreflights.gpuAddress;
    preflightView.brainCommitPreflightElementCount = 1u;
    preflightView.preflightReadyEventValue = 1u;
    preflightView.brainCommitPreflightStride = 1u;
    preflightView.environmentCount = 1u;
    preflightView.transactionSlot = prepared.program.transactionSlot;
    preflightView.stepIndex = 0u;
    preflightView.substepIndex = prepared.program.substepIndex;
    preflightView.physicsSubstepCount =
        prepared.program.physicsSubstepCount;
    preflightView.controlStep = prepared.program.controlStep;
    preflightView.programFingerprint = prepared.program.fingerprint;
    preflightView.transactionFingerprint =
        prepared.program.transactionFingerprint;
    preflightView.linearizationEpoch = prepared.program.linearizationEpoch;
    preflightView.slotGeneration = prepared.program.slotGeneration;
    if (abortAndRetry) {
        auto stale = prepared.lease;
        ++stale.slotGeneration;
        require(!prepared.program.reservePreparedApplication(
                    prepared.program.context, stale, proposalResource,
                    preflightView),
            "stale application reservation was accepted");
        auto wrongControl = preflightView;
        ++wrongControl.controlStep;
        require(!prepared.program.reservePreparedApplication(
                    prepared.program.context, prepared.lease,
                    proposalResource, wrongControl),
            "wrong-control-step preflight was accepted");
    }
    require(prepared.program.reservePreparedApplication(
                prepared.program.context, prepared.lease, proposalResource,
                preflightView),
        "exact proposal/preflight reservation was rejected");
    require(!prepared.program.reservePreparedApplication(
                prepared.program.context, prepared.lease, proposalResource,
                preflightView),
        "duplicate application reservation was accepted");

    const auto ack = makeAck(prepared, proposal, preflight, mode);
    const auto action = makeAction(prepared, proposal, ack, mode);
    prepared.brainAcks = makeBuffer(
        queue.device, ack, @"Brain ACK");
    prepared.brainAckEvent = [queue.device newSharedEvent];
    require(prepared.brainAckEvent != nil,
        "failed to allocate Brain ACK event");
    prepared.brainAckEvent.signaledValue = 1u;
    std::memcpy(
        prepared.applyActions.contents, &action, sizeof(action));

    const bool forceReject = mode == ApplyMode::reject;
    const auto makeApplyPass = [&](id<MTLCommandBuffer> commandBuffer) {
        metalrobo::MetalNumanXHumanMatterApplyPass pass{};
        pass.mode = forceReject
            ? metalrobo::MetalNumanXHumanMatterApplyMode::forceReject
            : metalrobo::MetalNumanXHumanMatterApplyMode::validateBrainAck;
        pass.commandBuffer = (__bridge void*)commandBuffer;
        if (!forceReject) {
            pass.brainAcks = (__bridge void*)prepared.brainAcks;
            pass.brainAckEvent = (__bridge void*)prepared.brainAckEvent;
            pass.brainAcksGPUAddress = prepared.brainAcks.gpuAddress;
            pass.brainAckElementCount = 1u;
            pass.brainAckEventValue = 1u;
            pass.brainAckStride = 1u;
        }
        pass.environmentCount = 1u;
        pass.transactionSlot = prepared.program.transactionSlot;
        pass.stepIndex = 0u;
        pass.substepIndex = prepared.program.substepIndex;
        pass.physicsSubstepCount = prepared.program.physicsSubstepCount;
        pass.controlStep = prepared.program.controlStep;
        pass.programFingerprint = prepared.program.fingerprint;
        pass.transactionFingerprint =
            prepared.program.transactionFingerprint;
        pass.linearizationEpoch = prepared.program.linearizationEpoch;
        pass.slotGeneration = prepared.program.slotGeneration;
        return pass;
    };
    const auto encodeAttempt = [&](const bool testMalformed) {
        id<MTLCommandBuffer> commandBuffer = nil;
        // A completed borrowed command buffer is intentionally not retained.
        // Exercise the real Objective-C address-reuse case that previously
        // confused an unretained scalar pointer with a lifetime identity.
        // Every candidate remains untouched and NotEnqueued until selected.
        for (std::uint32_t attempt = 0u;
             attempt < 64u && commandBuffer == nil; ++attempt) {
            @autoreleasepool {
                id<MTLCommandBuffer> candidate = [queue commandBuffer];
                require(candidate != nil, "failed to allocate apply CB");
                const bool reused = reinterpret_cast<std::uintptr_t>(
                    (__bridge void*)candidate) ==
                    prepared.physicalCommandBufferAddress;
                if (reused || attempt == 63u) {
                    commandBuffer = candidate;
                    prepared.applyReusedPhysicalCommandBufferAddress |=
                        reused;
                } else {
                    // Retire this otherwise untouched candidate so Metal may
                    // recycle its Objective-C command-buffer wrapper.
                    finish(candidate);
                }
            }
        }
        require(commandBuffer != nil, "failed to allocate apply CB");
        auto applyPass = makeApplyPass(commandBuffer);
        if (testMalformed) {
            auto stale = prepared.lease;
            ++stale.slotGeneration;
            require(!prepared.program.encodePreparedApply(
                        prepared.program.context, stale, applyPass),
                "stale-generation apply was accepted");
            auto wrongControl = applyPass;
            ++wrongControl.controlStep;
            require(!prepared.program.encodePreparedApply(
                        prepared.program.context, prepared.lease,
                        wrongControl),
                "wrong-control-step apply was accepted");
        }
        require(prepared.program.encodePreparedApply(
                    prepared.program.context, prepared.lease, applyPass),
            "exact prepared apply was rejected");
        require(!prepared.program.encodePreparedApply(
                    prepared.program.context, prepared.lease, applyPass),
            "duplicate prepared apply was accepted");
        require(matter.preparedStateDisposition(identity) ==
                numi::matter::PreparedStateDisposition::applying,
            "Matter did not enter FINALIZING for the exact apply CB");
        return std::pair{commandBuffer, applyPass};
    };

    if (abortAndRetry) {
        auto [abandoned, abandonedPass] = encodeAttempt(true);
        prepared.program.abortPreparedApply(
            prepared.program.context, prepared.lease, abandonedPass);
        require(matter.preparedStateDisposition(identity) ==
                numi::matter::PreparedStateDisposition::prepared,
            "unqueued apply abort poisoned retry");
    }

    auto [commandBuffer, applyPass] = encodeAttempt(!abortAndRetry);
    id<MTLBuffer> outcomeRead = makeZeroBuffer(
        queue.device,
        sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU),
        @"Matter apply outcome readback");
    id<MTLBlitCommandEncoder> outcomeBlit =
        [commandBuffer blitCommandEncoder];
    require(outcomeBlit != nil && copy(
            outcomeBlit,
            (__bridge id<MTLBuffer>)prepared.program.matterApplyOutcomes,
            (__bridge void*)outcomeRead,
            sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU)),
        "failed to stage Matter apply outcome");
    [outcomeBlit endEncoding];
    finish(commandBuffer);
    const auto matterOutcome = value<
        MRNumanXHumanMatterMatterApplyOutcomeGPU>(outcomeRead);
    require(matterOutcome.outcomeFingerprint ==
                recordFingerprint(matterOutcome),
        "Matter apply outcome FNV is invalid");
    if (mode == ApplyMode::physicalReject) {
        require(matterOutcome.status ==
                    MR_NUMANX_HUMAN_MATTER_APPLY_REJECT &&
                matterOutcome.decision ==
                    MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
                matterOutcome.code ==
                    MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT &&
                matterOutcome.physicsTokenFingerprint == 0u,
            "Matter lost the exact zero-gate physical-reject outcome");
    } else if (invalidBindingCross(mode)) {
        require(matterOutcome.status ==
                    MR_NUMANX_HUMAN_MATTER_APPLY_REJECT &&
                matterOutcome.decision ==
                    MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
                matterOutcome.code ==
                    MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_MATTER_OUTCOME &&
                matterOutcome.physicsTokenFingerprint ==
                    (physicalProposal(mode)
                        ? 0u : proposal.physicsTokenFingerprint),
            "prepared binding cross-substitution did not produce "
            "INVALID_OUTCOME");
    }

    if (mode != ApplyMode::pending) {
        const bool accept = mode == ApplyMode::accept;
        const bool physicalReject = mode == ApplyMode::physicalReject;
        const bool invalidCross = invalidBindingCross(mode);
        MRNumanXHumanMatterAppliedOutcomeGPU applied{};
        applied.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
        applied.status = invalidCross
            ? MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH
            : accept
                ? MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED
                : MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED;
        applied.decision = invalidCross
            ? MR_NUMANX_HUMAN_MATTER_ROOT_PENDING
            : accept
                ? MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT
                : MR_NUMANX_HUMAN_MATTER_ROOT_REJECT;
        applied.code = invalidCross
            ? MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_MATTER_OUTCOME
            : accept
                ? MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS
                : physicalReject
                    ? MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT
                    : MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT;
        applied.programFingerprint = prepared.program.fingerprint;
        applied.transactionFingerprint =
            prepared.program.transactionFingerprint;
        applied.linearizationEpoch = prepared.program.linearizationEpoch;
        applied.slotGeneration = prepared.program.slotGeneration;
        applied.physicsTokenFingerprint = accept
            ? proposal.physicsTokenFingerprint : 0u;
        applied.proposalFingerprint = proposal.proposalFingerprint;
        if (accept || physicalReject || invalidCross) {
            applied.ackFingerprint = ack.ackFingerprint;
            applied.preflightFingerprint = preflight.preflightFingerprint;
            applied.fastGateFingerprint = ack.fastGateFingerprint;
        }
        applied.matterApplyFingerprint = invalidCross
            ? 0u : matterOutcome.outcomeFingerprint;
        applied.environment = 0u;
        applied.stepIndex = 0u;
        applied.substepIndex = prepared.program.substepIndex;
        applied.transactionSlot = prepared.program.transactionSlot;
        applied.physicsSubstepCount = prepared.program.physicsSubstepCount;
        applied.controlStep = prepared.program.controlStep;
        applied.appliedFingerprint = recordFingerprint(applied);
        std::memcpy(
            prepared.appliedOutcomes.contents, &applied, sizeof(applied));
        if (accept) {
            std::memcpy(
                prepared.finalTokens.contents,
                prepared.proposedTokens.contents,
                MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES);
        } else {
            std::memset(
                prepared.finalTokens.contents, 0,
                prepared.finalTokens.length);
            if (!invalidCross) {
                std::memcpy(arenas.q.contents, arenas.qCheckpoint.contents,
                            arenas.q.length);
                std::memcpy(arenas.v.contents, arenas.vCheckpoint.contents,
                            arenas.v.length);
                std::memcpy(
                    arenas.mujoco.contents, arenas.mujocoCheckpoint.contents,
                    arenas.mujoco.length);
            }
        }
    }
    auto stale = prepared.lease;
    ++stale.slotGeneration;
    require(prepared.program.releasePrepareLease(
                prepared.program.context, stale,
                (__bridge void*)commandBuffer, true) ==
                metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
                    terminalNoTouch,
        "stale release did not fail closed");
    const auto exactDisposition = prepared.program.releasePrepareLease(
                prepared.program.context, prepared.lease,
                (__bridge void*)commandBuffer, true);
    const auto expectedDisposition = mode == ApplyMode::accept
        ? metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
            acceptedPendingPublication
        : (mode == ApplyMode::reject ||
           mode == ApplyMode::physicalReject)
            ? metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::released
            : metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
                terminalNoTouch;
    require(exactDisposition == expectedDisposition,
        "apply completion returned the wrong adapter disposition: actual=" +
            std::to_string(static_cast<std::uint32_t>(exactDisposition)) +
            " expected=" +
            std::to_string(static_cast<std::uint32_t>(expectedDisposition)) +
            " mode=" + std::to_string(static_cast<std::uint32_t>(mode)));

    ApplyResult result;
    result.token = value<MRNumanXAcceptedPhysicsStateTokenGPU>(
        prepared.finalTokens);
    if (mode == ApplyMode::accept) {
        const auto applied = value<MRNumanXHumanMatterAppliedOutcomeGPU>(
            prepared.appliedOutcomes);
        numi::matter::PreparedStatePublicationBinding expectedBinding{};
        expectedBinding.physicsTokenFingerprint =
            proposal.physicsTokenFingerprint;
        expectedBinding.brainProgramFingerprint =
            proposal.brainProgramFingerprint;
        expectedBinding.brainShadowStateFingerprint =
            proposal.brainShadowStateFingerprint;
        expectedBinding.brainWitnessFingerprint =
            proposal.brainWitnessFingerprint;
        expectedBinding.matterApplyFingerprint =
            applied.matterApplyFingerprint;
        expectedBinding.appliedDecisionFingerprint =
            applied.appliedFingerprint;
        expectedBinding.jointCommitFingerprint = 0xd002u;
        expectedBinding.brainGeneration = 19u;
        MRNumanXHumanMatterJointPublicationFenceGPU fence{};
        fence.abiVersion =
            MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION;
        fence.structBytes = MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES;
        fence.status = MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING;
        fence.controlStep = prepared.program.controlStep;
        fence.substepIndex = prepared.program.substepIndex;
        fence.physicsSubstepCount = prepared.program.physicsSubstepCount;
        fence.ownerProgramFingerprint = prepared.program.fingerprint;
        fence.transactionFingerprint =
            prepared.program.transactionFingerprint;
        fence.linearizationEpoch = prepared.program.linearizationEpoch;
        fence.slotGeneration = prepared.program.slotGeneration;
        fence.physicsTokenFingerprint =
            expectedBinding.physicsTokenFingerprint;
        fence.brainProgramFingerprint =
            expectedBinding.brainProgramFingerprint;
        fence.brainShadowStateFingerprint =
            expectedBinding.brainShadowStateFingerprint;
        fence.brainWitnessFingerprint =
            expectedBinding.brainWitnessFingerprint;
        fence.appliedDecisionFingerprint =
            expectedBinding.appliedDecisionFingerprint;
        fence.jointCommitFingerprint =
            expectedBinding.jointCommitFingerprint;
        fence.brainGeneration = expectedBinding.brainGeneration;
        fence.fenceFingerprint = recordFingerprint(fence);
        std::memcpy(
            prepared.publicationFences.contents, &fence, sizeof(fence));
        metalrobo::MetalNumanXHumanMatterPublicationReservationView
            publication{};
        publication.proposal = proposalResource;
        publication.fence = fenceView(prepared);
        publication.appliedOutcomes = prepared.lease.appliedOutcomes;
        publication.finalAcceptedPhysicsStateTokens =
            prepared.lease.finalAcceptedPhysicsStateTokens;
        publication.appliedOutcomesGPUAddress =
            prepared.lease.appliedOutcomesGPUAddress;
        publication.finalAcceptedPhysicsStateTokensGPUAddress =
            prepared.lease.finalAcceptedPhysicsStateTokensGPUAddress;
        publication.appliedOutcomeElementCount = 1u;
        publication.finalAcceptedPhysicsStateTokenByteCount =
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
        publication.appliedOutcomeStride = 1u;
        publication.finalTokenStrideBytes =
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
        publication.jointCommitFingerprint =
            expectedBinding.jointCommitFingerprint;
        publication.brainGeneration = expectedBinding.brainGeneration;
        auto& humanBinding = publication.humanIOBinding;
        humanBinding.environmentCount = 1u;
        humanBinding.transactionSlot = prepared.program.transactionSlot;
        humanBinding.stepIndex = 0u;
        humanBinding.substepIndex = prepared.program.substepIndex;
        humanBinding.physicsSubstepCount =
            prepared.program.physicsSubstepCount;
        humanBinding.controlStep = prepared.program.controlStep;
        humanBinding.ownerProgramFingerprint = prepared.program.fingerprint;
        humanBinding.transactionFingerprint =
            prepared.program.transactionFingerprint;
        humanBinding.linearizationEpoch = prepared.program.linearizationEpoch;
        humanBinding.slotGeneration = prepared.program.slotGeneration;
        humanBinding.physicsTokenFingerprint =
            proposal.physicsTokenFingerprint;
        humanBinding.proposalFingerprint = proposal.proposalFingerprint;
        humanBinding.ackFingerprint = applied.ackFingerprint;
        humanBinding.appliedDecisionFingerprint =
            applied.appliedFingerprint;
        humanBinding.jointCommitFingerprint =
            expectedBinding.jointCommitFingerprint;
        humanBinding.brainGeneration = expectedBinding.brainGeneration;
        require(prepared.humanIO != nullptr,
            "publication lost the bound HumanIO candidate");
        const auto& humanProgram = prepared.humanIO->program;
        humanBinding.candidateKeyFingerprint =
            humanProgram.candidateKeyFingerprint;
        humanBinding.acceptedBrainGeneration =
            humanProgram.acceptedBrainGeneration;
        humanBinding.sensorGeneration = humanProgram.sensorGeneration;
        humanBinding.humanIOProgramFingerprint =
            humanProgram.humanIOProgramFingerprint;
        humanBinding.sensorFingerprint = humanProgram.sensorFingerprint;
        humanBinding.transactionInstanceFingerprint =
            humanProgram.transactionInstanceFingerprint;
        humanBinding.candidatePublicationFingerprint =
            humanProgram.candidatePublicationFingerprint;
        humanBinding.deviceRegistryID = humanProgram.deviceRegistryID;
        humanBinding.humanIOIdentityFingerprint =
            humanProgram.identityFingerprint;
        humanBinding.bindingFingerprint = metalrobo::
            metalNumanXHumanIOPublicationBindingFingerprint(humanBinding);
        require(prepared.humanIO->stage ==
                    SensorPublicationService::Stage::provisional &&
                prepared.humanIO->reserveCalls == 0u &&
                prepared.humanIO->publishCalls == 0u,
            "HumanIO candidate became visible before root reservation");
        require(prepared.program.reservePublishedRoot(
                    prepared.program.context, prepared.lease, publication),
            "exact publication reservation was rejected");
        require(prepared.humanIO->stage ==
                    SensorPublicationService::Stage::rootReserved &&
                prepared.humanIO->reserveCalls == 1u &&
                prepared.humanIO->publishCalls == 0u,
            "HumanIO root reservation published or lost the candidate");
        require(!prepared.program.reservePublishedRoot(
                    prepared.program.context, prepared.lease, publication),
            "duplicate publication reservation was accepted");
        require(prepared.humanIO->reserveCalls == 1u &&
                prepared.humanIO->publishCalls == 0u,
            "duplicate publication reservation touched HumanIO");
        fence.status = MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED;
        fence.fenceFingerprint = recordFingerprint(fence);
        std::memcpy(
            prepared.publicationFences.contents, &fence, sizeof(fence));
        auto staleFence = publication.fence;
        ++staleFence.controlStep;
        require(prepared.program.releasePublishedRoot(
                    prepared.program.context, prepared.lease, staleFence) ==
                    metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
                        terminalNoTouch,
            "stale publication fence view was accepted");
        require(prepared.program.releasePublishedRoot(
                    prepared.program.context, prepared.lease,
                    publication.fence) ==
                    metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::
                        released,
            "exact COMMITTED publication fence was rejected");
        require(prepared.humanIO->stage ==
                    SensorPublicationService::Stage::released &&
                prepared.humanIO->reserveCalls == 1u &&
                prepared.humanIO->publishCalls == 1u &&
                prepared.humanIO->rejectCalls == 0u &&
                prepared.humanIO->terminalCalls == 0u,
            "COMMITTED root did not publish HumanIO exactly once");
        require(std::memcmp(
                    &result.token, &prepared.preparedToken,
                    sizeof(result.token)) == 0,
            "ACCEPT did not preserve the immutable proposed token");
    } else if (mode == ApplyMode::reject ||
               mode == ApplyMode::physicalReject) {
        result.after = matter.snapshot();
        const MRNumanXAcceptedPhysicsStateTokenGPU zero{};
        require(std::memcmp(&result.token, &zero, sizeof(zero)) == 0,
            "REJECT published a root token");
        require(equalAcceptedAuthority(prepared.before, result.after),
            "REJECT did not byte-restore Matter accepted authority");
        require(std::memcmp(arenas.q.contents, arenas.qCheckpoint.contents,
                            arenas.q.length) == 0 &&
                std::memcmp(arenas.v.contents, arenas.vCheckpoint.contents,
                            arenas.v.length) == 0 &&
                std::memcmp(arenas.mujoco.contents,
                            arenas.mujocoCheckpoint.contents,
                            arenas.mujoco.length) == 0,
            "REJECT did not byte-restore Human q/v/MyoSim");
        require(prepared.humanIO != nullptr &&
                prepared.humanIO->stage ==
                    SensorPublicationService::Stage::rejected &&
                prepared.humanIO->reserveCalls == 0u &&
                prepared.humanIO->publishCalls == 0u &&
                prepared.humanIO->rejectCalls == 1u,
            "REJECT did not reject the unpublished HumanIO candidate once");
    } else if (invalidBindingCross(mode)) {
        result.after = matter.snapshot();
        const MRNumanXAcceptedPhysicsStateTokenGPU zero{};
        require(std::memcmp(&result.token, &zero, sizeof(zero)) == 0 &&
                equalAcceptedAuthority(prepared.before, result.after),
            "invalid binding cross-substitution changed Matter authority");
        require(std::memcmp(
                    arenas.q.contents, arenas.qCheckpoint.contents,
                    arenas.q.length) != 0 &&
                prepared.humanIO != nullptr &&
                prepared.humanIO->stage ==
                    SensorPublicationService::Stage::provisional &&
                prepared.humanIO->reserveCalls == 0u &&
                prepared.humanIO->publishCalls == 0u &&
                prepared.humanIO->rejectCalls == 0u,
            "invalid binding cross-substitution escaped terminal quarantine");
    } else {
        require(!matter.snapshot().available,
            "PENDING terminal state became snapshot-readable");
        require(prepared.humanIO != nullptr &&
                prepared.humanIO->stage ==
                    SensorPublicationService::Stage::provisional &&
                prepared.humanIO->reserveCalls == 0u &&
                prepared.humanIO->publishCalls == 0u &&
                prepared.humanIO->rejectCalls == 0u,
            "PENDING terminal no-touch consumed the HumanIO candidate");
    }
    if (!result.after.available && mode != ApplyMode::pending) {
        result.after = matter.snapshot();
    }
    require(matter.preparedStateDisposition(identity) ==
            (mode == ApplyMode::pending
                ? numi::matter::PreparedStateDisposition::terminalNoTouch
                : numi::matter::PreparedStateDisposition::resolved),
        "apply/publication lost exact Matter disposition identity");
    return result;
}

void verifyTerminalPending(
    const numi::matter::CompiledWorld& world,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue
) {
    numi::matter::RuntimeConfiguration runtimeConfig;
    runtimeConfig.metallib = NUMI_MATTER_METALLIB;
    runtimeConfig.environmentCount = 1u;
    runtimeConfig.captureEvents = false;
    runtimeConfig.captureDiagnostics = true;
    runtimeConfig.adaptiveTransfer = false;
    runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        sizeof(MRMujocoMuscleStateGPU);
    auto* matter = new numi::matter::Runtime;
    const auto matterInit = matter->initialize(world, runtimeConfig);
    require(matterInit.encoded,
        "terminal Matter Runtime initialization failed: " +
            matterInit.message);
    metalrobo::MetalNumanXHumanMatterConfig config;
    config.matterRuntime = matter;
    config.coupledHumanMetallibPath = NUMANX_ADAPTER_METALLIB;
    config.adapterMetallibPath = NUMANX_ADAPTER_METALLIB;
    config.environmentCapacity = 1u;
    config.pointCapacity = 1u;
    config.transactionSlotCount = 1u;
    config.stateProofProgram.context = matter;
    config.stateProofProgram.encode = &encodeRuntimeProof;
    config.stateProofProgram.fingerprint =
        matter->acceptedStateProofProgramFingerprint();
    auto* adapter = new metalrobo::MetalNumanXHumanMatterContext(config);
    const auto initialized = adapter->initialize();
    require(initialized.succeeded(),
        "terminal adapter initialization failed: " + initialized.message);
    OwnerArenas arenas = makeOwnerArenas(device);
    std::array<float, kQ> q{};
    q[6] = 1.0f;
    MRBodyStateGPU body{};
    body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    body.linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
    body.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
    body.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    body.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
    body.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    std::array<float, 3u * kDofs> jacobian{};
    jacobian[0u] = 1.0f;
    jacobian[kDofs + 1u] = 1.0f;
    jacobian[2u * kDofs + 2u] = 1.0f;
    ExactCandidateService exact;
    exact.q = makeBuffer(device, q, @"terminal candidate q");
    exact.body = makeBuffer(device, body, @"terminal candidate body");
    exact.jacobian = makeBuffer(
        device, jacobian, @"terminal candidate Jacobian");
    auto prepared = prepareTransaction(
        *adapter, *matter, queue, arenas, exact, 100u, 137u, false);
    std::memset(prepared.finalTokens.contents, 0xa5,
                prepared.finalTokens.length);
    (void)applyTransaction(
        *matter, queue, arenas, prepared, ApplyMode::pending, false);
    const std::array<std::uint8_t,
        MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES> sentinel = [] {
            std::array<std::uint8_t,
                MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES> value{};
            value.fill(0xa5u);
            return value;
        }();
    require(std::memcmp(prepared.finalTokens.contents, sentinel.data(),
                        sentinel.size()) == 0,
        "PENDING touched the owner final token");
    require(matter->preparedStateDisposition(dispositionIdentity(
                prepared.transaction, prepared.program)) ==
            numi::matter::PreparedStateDisposition::terminalNoTouch,
        "PENDING did not enter terminal no-touch disposition");
    auto newer = prepared.transaction;
    ++newer.slotGeneration;
    ++newer.transactionFingerprint;
    require(!adapter->program(newer).valid(),
        "terminal no-touch generation admitted slot reuse");
    require(!matter->snapshot().available,
        "terminal no-touch Matter authority became readable/reusable");
    // Terminal quarantine intentionally owns these objects for process
    // lifetime; deleting either context would manufacture unsafe recovery.
}

struct OwnerApplyCompletionCapture {
    std::atomic<std::uint32_t> count{0u};
    std::atomic<std::uint32_t> status{0u};
    std::atomic<std::uint64_t> slotGeneration{0u};
};

void captureOwnerApplyCompletion(
    void* opaque,
    const metalrobo::MetalNumanXHumanMatterApplyTerminalStatus status,
    const std::uint64_t slotGeneration
) noexcept {
    auto* capture = static_cast<OwnerApplyCompletionCapture*>(opaque);
    if (capture == nullptr) return;
    capture->status.store(
        static_cast<std::uint32_t>(status), std::memory_order_release);
    capture->slotGeneration.store(
        slotGeneration, std::memory_order_release);
    capture->count.fetch_add(1u, std::memory_order_acq_rel);
}

void waitForOwnerApplyCompletion(OwnerApplyCompletionCapture& capture) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while (capture.count.load(std::memory_order_acquire) == 0u &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(capture.count.load(std::memory_order_acquire) == 1u,
        "owner apply completion did not arrive exactly once");
}

void verifyOwnerAdapterMatterIntegration(id<MTLDevice> device) {
    constexpr std::uint32_t attachmentBody =
        owner_fixture::kFirstBody + owner_fixture::kBodyCount - 1u;
    const auto world = compileAttachedWorld(attachmentBody);
    numi::matter::RuntimeConfiguration runtimeConfig;
    runtimeConfig.metallib = NUMI_MATTER_METALLIB;
    runtimeConfig.environmentCount = 1u;
    runtimeConfig.captureEvents = false;
    runtimeConfig.captureDiagnostics = true;
    runtimeConfig.adaptiveTransfer = false;
    runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        sizeof(MRMujocoMuscleStateGPU);
    numi::matter::Runtime matter;
    const auto matterInit = matter.initialize(world, runtimeConfig);
    require(matterInit.encoded,
        "integrated owner Matter initialization failed: " +
            matterInit.message);

    metalrobo::MetalNumanXHumanMatterConfig adapterConfig;
    adapterConfig.matterRuntime = &matter;
    adapterConfig.coupledHumanMetallibPath = NUMANX_ADAPTER_METALLIB;
    adapterConfig.adapterMetallibPath = NUMANX_ADAPTER_METALLIB;
    adapterConfig.environmentCapacity = 1u;
    adapterConfig.pointCapacity = 1u;
    adapterConfig.transactionSlotCount = 1u;
    adapterConfig.stateProofProgram.context = &matter;
    adapterConfig.stateProofProgram.encode = &encodeRuntimeProof;
    adapterConfig.stateProofProgram.fingerprint =
        matter.acceptedStateProofProgramFingerprint();
    metalrobo::MetalNumanXHumanMatterContext adapter(adapterConfig);
    const auto adapterInit = adapter.initialize();
    require(adapterInit.succeeded() &&
            adapterInit.acceptedStateProofAvailable,
        "integrated owner adapter initialization failed: " +
            adapterInit.message);

    metalrobo::MetalNumanXHumanMatterTransaction transaction{};
    transaction.environmentCount = 1u;
    transaction.transactionSlot = 0u;
    transaction.controlStep = 47u;
    transaction.physicsSubstep = 0u;
    transaction.physicsSubsteps = 1u;
    transaction.expectedMatterCompletedMicrosteps = 1u;
    transaction.seed = 0x4701u;
    transaction.transactionFingerprint = 0x4702u;
    transaction.substepFingerprint = 0x4703u;
    transaction.acceptedTimestampMicroseconds = 4704u;
    transaction.physicsGeneration = 1u;
    transaction.linearizationEpoch = 0x4705u;
    transaction.slotGeneration = 1u;
    const auto program = adapter.program(transaction);
    require(program.valid(), "integrated owner adapter program is invalid");

    auto model = owner_fixture::makeHumanModel();
    const auto points = owner_fixture::bodyProbes();
    owner_fixture::CandidateAudit ownerFixture;
    owner_fixture::initializeAudit(ownerFixture, device);
    auto input = owner_fixture::makeInput(model, points, ownerFixture);
    input.stand.numanXHumanMatterProgram = program;
    const metalrobo::MetalArticulatedOperatorConfig ownerConfig{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = matter.timestepSeconds(),
        .metallibPath = NUMANX_ADAPTER_METALLIB,
    };
    metalrobo::MetalArticulatedOperatorContext owner(ownerConfig);
    const auto before = matter.snapshot();
    require(before.available,
        "integrated owner Matter baseline is unavailable");
    metalrobo::MetalArticulatedOperatorSubmission submission;
    const auto submitted = owner.submit(model, input, submission);
    require(submitted.succeeded() && submitted.dispatched &&
            submission.valid(),
        "real owner rejected adapter+Matter prepare: " + submitted.message);
    metalrobo::MetalNumanXHumanMatterPrepared prepared;
    require(submission.extractPreparedHumanMatter(prepared) &&
            prepared.valid(),
        "real owner did not expose the prepared capability");
    metalrobo::MetalNumanXHumanMatterPreparedView view{};
    require(prepared.view(view) && view.environmentCount == 1u &&
            view.transactionSlot == transaction.transactionSlot &&
            view.stepIndex == 0u && view.substepIndex == 0u &&
            view.physicsSubstepCount == 1u &&
            view.controlStep == transaction.controlStep &&
            view.programFingerprint == program.fingerprint &&
            view.transactionFingerprint ==
                transaction.transactionFingerprint &&
            view.linearizationEpoch == transaction.linearizationEpoch &&
            view.slotGeneration == transaction.slotGeneration,
        "real owner prepared view lost adapter identity");

    id<MTLCommandQueue> applyQueue = [device newCommandQueue];
    require(applyQueue != nil,
        "real owner apply queue allocation failed");
    __unsafe_unretained id<MTLSharedEvent> ownerEvent =
        (__bridge id<MTLSharedEvent>)view.physicalPreparedEvent;
    waitForSharedEventValue(
        ownerEvent, view.physicalPreparedEventValue,
        "real owner physical completion event did not advance");

    auto humanIO = makeSensorPublicationService(
        device, transaction.transactionFingerprint,
        transaction.slotGeneration + 0x4700u);
    auto wrongTransactionCandidate = humanIO->program;
    ++wrongTransactionCandidate.transactionFingerprint;
    wrongTransactionCandidate.identityFingerprint =
        wrongTransactionCandidate.computedIdentityFingerprint();
    require(!prepared.bindHumanIOCandidatePublication(
                wrongTransactionCandidate),
        "real owner admitted a wrong-transaction HumanIO candidate");
    auto wrongDeviceCandidate = humanIO->program;
    ++wrongDeviceCandidate.deviceRegistryID;
    wrongDeviceCandidate.identityFingerprint =
        wrongDeviceCandidate.computedIdentityFingerprint();
    require(!prepared.bindHumanIOCandidatePublication(wrongDeviceCandidate),
        "real owner admitted a wrong-device HumanIO candidate");
    require(prepared.bindHumanIOCandidatePublication(humanIO->program),
        "real owner rejected exact post-physical HumanIO candidate");
    require(!prepared.bindHumanIOCandidatePublication(humanIO->program),
        "real owner admitted a duplicate HumanIO candidate bind");
    require(prepared.view(view) &&
            view.humanIOCandidateKeyFingerprint ==
                humanIO->program.candidateKeyFingerprint &&
            view.acceptedBrainGeneration ==
                humanIO->program.acceptedBrainGeneration &&
            view.humanIOSensorGeneration ==
                humanIO->program.sensorGeneration &&
            view.humanIOProgramFingerprint ==
                humanIO->program.humanIOProgramFingerprint &&
            view.humanIOSensorFingerprint ==
                humanIO->program.sensorFingerprint &&
            view.humanIOTransactionInstanceFingerprint ==
                humanIO->program.transactionInstanceFingerprint &&
            view.humanIOCandidatePublicationFingerprint ==
                humanIO->program.candidatePublicationFingerprint &&
            view.humanIODeviceRegistryID == device.registryID &&
            view.humanIOIdentityFingerprint ==
                humanIO->program.identityFingerprint,
        "real owner prepared view lost bound HumanIO identity");

    id<MTLCommandBuffer> proposalCommand = [applyQueue commandBuffer];
    metalrobo::MetalNumanXHumanMatterProposalRequest proposalRequest{};
    proposalRequest.mode =
        metalrobo::MetalNumanXHumanMatterProposalMode::forceReject;
    proposalRequest.commandBuffer = (__bridge void*)proposalCommand;
    proposalRequest.environmentCount = view.environmentCount;
    proposalRequest.transactionSlot = view.transactionSlot;
    proposalRequest.stepIndex = view.stepIndex;
    proposalRequest.substepIndex = view.substepIndex;
    proposalRequest.physicsSubstepCount = view.physicsSubstepCount;
    proposalRequest.controlStep = view.controlStep;
    proposalRequest.programFingerprint = view.programFingerprint;
    proposalRequest.transactionFingerprint = view.transactionFingerprint;
    proposalRequest.linearizationEpoch = view.linearizationEpoch;
    proposalRequest.slotGeneration = view.slotGeneration;
    const auto proposed = prepared.proposePrepared(proposalRequest);
    require(proposed.succeeded() && proposed.encoded,
        "real owner rejected mutation-free force-reject proposal: " +
            proposed.message);
    require(!prepared.proposePrepared(proposalRequest).succeeded(),
        "real owner admitted a duplicate proposal");
    finish(proposalCommand);
    waitForSharedEventValue(
        ownerEvent, view.proposalEventValue,
        "real owner proposal completion event did not advance");

    const auto proposal = value<MRNumanXHumanMatterProposalGPU>(
        (__bridge id<MTLBuffer>)view.proposals);
    require(proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
            proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
            proposal.candidatePublicationFingerprint ==
                humanIO->program.candidatePublicationFingerprint &&
            proposal.humanIOIdentityFingerprint ==
                humanIO->program.identityFingerprint &&
            proposal.proposalFingerprint == recordFingerprint(proposal),
        "real owner force-reject proposal is malformed");
    MRNumanXHumanMatterBrainCommitPreflightGPU preflight{};
    preflight.abiVersion =
        MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_ABI_VERSION;
    preflight.structBytes = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES;
    preflight.status = MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_SUCCESS;
    preflight.environment = 0u;
    preflight.controlStep = view.controlStep;
    preflight.substepIndex = view.substepIndex;
    preflight.physicsSubstepCount = view.physicsSubstepCount;
    preflight.transactionSlot = view.transactionSlot;
    preflight.ownerProgramFingerprint = view.programFingerprint;
    preflight.transactionFingerprint = view.transactionFingerprint;
    preflight.linearizationEpoch = view.linearizationEpoch;
    preflight.slotGeneration = view.slotGeneration;
    preflight.substepFingerprint = transaction.substepFingerprint;
    preflight.physicsTokenFingerprint = proposal.physicsTokenFingerprint;
    preflight.fastTargetGeneration = 1u;
    preflight.cognitiveTargetGeneration = 1u;
    preflight.jointReceiptFingerprint = 0x4706u;
    preflight.fastProgramFingerprint = 0x4707u;
    preflight.brainProgramFingerprint = 0x4708u;
    preflight.preflightFingerprint = recordFingerprint(preflight);
    id<MTLBuffer> preflightBuffer = makeBuffer(
        device, preflight, @"integrated Brain commit preflight");
    id<MTLSharedEvent> preflightReady = [device newSharedEvent];
    require(preflightReady != nil,
        "integrated Brain preflight event allocation failed");
    preflightReady.signaledValue = 1u;
    metalrobo::MetalNumanXHumanMatterBrainPreflightView preflightView{};
    preflightView.brainCommitPreflights =
        (__bridge void*)preflightBuffer;
    preflightView.preflightReadyEvent = (__bridge void*)preflightReady;
    preflightView.brainCommitPreflightsGPUAddress =
        preflightBuffer.gpuAddress;
    preflightView.brainCommitPreflightElementCount = 1u;
    preflightView.preflightReadyEventValue = 1u;
    preflightView.brainCommitPreflightStride = 1u;
    preflightView.environmentCount = view.environmentCount;
    preflightView.transactionSlot = view.transactionSlot;
    preflightView.stepIndex = view.stepIndex;
    preflightView.substepIndex = view.substepIndex;
    preflightView.physicsSubstepCount = view.physicsSubstepCount;
    preflightView.controlStep = view.controlStep;
    preflightView.programFingerprint = view.programFingerprint;
    preflightView.transactionFingerprint = view.transactionFingerprint;
    preflightView.linearizationEpoch = view.linearizationEpoch;
    preflightView.slotGeneration = view.slotGeneration;
    auto stalePreflight = preflightView;
    ++stalePreflight.controlStep;
    require(!prepared.reservePreparedApplication(stalePreflight),
        "real owner admitted a stale preflight reservation");
    require(prepared.reservePreparedApplication(preflightView),
        "real owner rejected exact force-reject reservation");
    require(!prepared.reservePreparedApplication(preflightView),
        "real owner admitted duplicate application reservation");

    OwnerApplyCompletionCapture completion;
    id<MTLCommandBuffer> applyCommand = [applyQueue commandBuffer];
    metalrobo::MetalNumanXHumanMatterApplyRequest applyRequest{};
    applyRequest.mode =
        metalrobo::MetalNumanXHumanMatterApplyMode::forceReject;
    applyRequest.commandBuffer = (__bridge void*)applyCommand;
    applyRequest.completionContext = &completion;
    applyRequest.completion = &captureOwnerApplyCompletion;
    applyRequest.environmentCount = view.environmentCount;
    applyRequest.transactionSlot = view.transactionSlot;
    applyRequest.stepIndex = view.stepIndex;
    applyRequest.substepIndex = view.substepIndex;
    applyRequest.physicsSubstepCount = view.physicsSubstepCount;
    applyRequest.controlStep = view.controlStep;
    applyRequest.programFingerprint = view.programFingerprint;
    applyRequest.transactionFingerprint = view.transactionFingerprint;
    applyRequest.linearizationEpoch = view.linearizationEpoch;
    applyRequest.slotGeneration = view.slotGeneration;
    auto staleApply = applyRequest;
    ++staleApply.slotGeneration;
    require(!prepared.applyPrepared(staleApply).succeeded(),
        "real owner admitted a stale apply generation");
    const auto applied = prepared.applyPrepared(applyRequest);
    require(applied.succeeded() && applied.encoded,
        "real owner rejected forced root apply: " + applied.message);
    require(!prepared.applyPrepared(applyRequest).succeeded(),
        "real owner admitted duplicate apply");
    finish(applyCommand);
    waitForOwnerApplyCompletion(completion);
    require(completion.status.load(std::memory_order_acquire) ==
                static_cast<std::uint32_t>(metalrobo::
                    MetalNumanXHumanMatterApplyTerminalStatus::
                        rejectedReleased) &&
            completion.slotGeneration.load(std::memory_order_acquire) ==
                view.slotGeneration,
        "real owner apply completion lost reject/generation identity");

    const auto disposition = matter.preparedStateDisposition(
        dispositionIdentity(transaction, program));
    const auto after = matter.snapshot();
    require(disposition == numi::matter::PreparedStateDisposition::resolved &&
            !prepared.valid() && equalAcceptedAuthority(before, after) &&
            humanIO->stage == SensorPublicationService::Stage::rejected &&
            humanIO->reserveCalls == 0u &&
            humanIO->publishCalls == 0u &&
            humanIO->rejectCalls == 1u,
        "owner+adapter+Matter reject did not resolve and byte-restore");
}

} // namespace adapter_fixture

int main() {
    using namespace adapter_fixture;
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device is available");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "failed to create probe command queue");
            auto world = compileAttachedWorld();
            numi::matter::RuntimeConfiguration runtimeConfig;
            runtimeConfig.metallib = NUMI_MATTER_METALLIB;
            runtimeConfig.environmentCount = 1u;
            runtimeConfig.captureEvents = false;
            runtimeConfig.captureDiagnostics = true;
            runtimeConfig.adaptiveTransfer = false;
            runtimeConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
                sizeof(MRMujocoMuscleStateGPU);
            numi::matter::Runtime matter;
            const auto matterInit = matter.initialize(world, runtimeConfig);
            require(matterInit.encoded,
                "Matter Runtime initialization failed: " + matterInit.message);

            metalrobo::MetalNumanXHumanMatterConfig config;
            config.matterRuntime = &matter;
            config.coupledHumanMetallibPath = NUMANX_ADAPTER_METALLIB;
            config.adapterMetallibPath = NUMANX_ADAPTER_METALLIB;
            config.environmentCapacity = 1u;
            config.pointCapacity = 1u;
            config.transactionSlotCount = 2u;
            config.stateProofProgram.context = &matter;
            config.stateProofProgram.encode = &encodeRuntimeProof;
            config.stateProofProgram.fingerprint =
                matter.acceptedStateProofProgramFingerprint();
            metalrobo::MetalNumanXHumanMatterContext adapter(config);
            const auto initialized = adapter.initialize();
            require(initialized.succeeded(),
                "adapter initialization failed: " + initialized.message);
            require(initialized.acceptedStateProofAvailable,
                "adapter did not acquire real Matter proof authority");

            OwnerArenas arenas = makeOwnerArenas(device);
            std::array<float, kQ> candidateQ{};
            candidateQ[6] = 1.0f;
            MRBodyStateGPU body{};
            body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            body.linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
            body.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
            body.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
            body.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
            body.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
            std::array<float, 3u * kDofs> jacobian{};
            jacobian[0u * kDofs + 0u] = 1.0f;
            jacobian[1u * kDofs + 1u] = 1.0f;
            jacobian[2u * kDofs + 2u] = 1.0f;
            ExactCandidateService exact;
            exact.q = makeBuffer(device, candidateQ, @"candidate q source");
            exact.body = makeBuffer(device, body, @"candidate body source");
            exact.jacobian = makeBuffer(
                device, jacobian, @"candidate Jacobian source");

            verifyFirstCommandAbort(
                adapter, matter, queue, arenas, exact);
            auto accepted = prepareTransaction(
                adapter, matter, queue, arenas, exact, 2u, 37u, true);
            const auto acceptedApply = applyTransaction(
                matter, queue, arenas, accepted,
                ApplyMode::accept, true);
            const auto acceptedToken = acceptedApply.token;
            require(accepted.applyReusedPhysicalCommandBufferAddress,
                "bounded apply allocation did not reproduce completed-CB "
                "Objective-C address reuse");

            auto rejected = prepareTransaction(
                adapter, matter, queue, arenas, exact, 3u, 38u, true);
            const auto rejectedApply = applyTransaction(
                matter, queue, arenas, rejected,
                ApplyMode::reject, false);
            require(std::memcmp(
                        accepted.finalTokens.contents, &acceptedToken,
                        sizeof(acceptedToken)) == 0,
                "later REJECT changed a prior accepted root token");

            auto physicalRejected = prepareTransaction(
                adapter, matter, queue, arenas, exact, 4u, 39u, false,
                true);
            const auto physicalRejectedApply = applyTransaction(
                matter, queue, arenas, physicalRejected,
                ApplyMode::physicalReject, false);
            require(equalAcceptedAuthority(
                        physicalRejected.before,
                        physicalRejectedApply.after) &&
                    std::memcmp(
                        accepted.finalTokens.contents, &acceptedToken,
                        sizeof(acceptedToken)) == 0,
                "zero-gate physical REJECT changed accepted authority");

            auto replay = prepareTransaction(
                adapter, matter, queue, arenas, exact, 5u, 40u, false);
            const auto replayApply = applyTransaction(
                matter, queue, arenas, replay,
                ApplyMode::reject, false);
            require(equalAcceptedAuthority(
                        rejected.before, replay.before) &&
                    equalAcceptedAuthority(
                        rejectedApply.after, replayApply.after) &&
                    equalAcceptedAuthority(
                        rejected.before, replayApply.after),
                "REJECT replay did not preserve byte-identical Matter authority");
            require(std::memcmp(
                        accepted.finalTokens.contents, &acceptedToken,
                        sizeof(acceptedToken)) == 0,
                "REJECT replay changed a prior accepted root token");

            // Cross-substitution negative: this first CB was genuinely
            // ACCEPT/VALID, but the later immutable records falsely claim a
            // physical proposal reject. Matter must restore with the exact
            // INVALID_OUTCOME code and the adapter must quarantine HumanIO
            // rather than releasing the mismatched joint authority.
            auto fabricatedPhysical = prepareTransaction(
                adapter, matter, queue, arenas, exact, 6u, 41u, false);
            const auto fabricatedPhysicalApply = applyTransaction(
                matter, queue, arenas, fabricatedPhysical,
                ApplyMode::fabricatedPhysicalReject, false);
            require(equalAcceptedAuthority(
                        fabricatedPhysical.before,
                        fabricatedPhysicalApply.after) &&
                    std::memcmp(
                        accepted.finalTokens.contents, &acceptedToken,
                        sizeof(acceptedToken)) == 0,
                "fabricated physical reject changed a published root");

            // Slot 0 is now terminally quarantined. Neither a later slot's
            // lease nor its physical pass may reuse any of slot 0's retained
            // full-buffer authority. The failed admission must remain a
            // no-touch operation and leave slot 1 reusable at a newer
            // generation.
            verifyCrossSlotAuthorityRejected(
                adapter, matter, queue, arenas, exact, 7u, 42u, 1u);

            OwnerArenas disjointArenas = makeOwnerArenas(device);
            ExactCandidateService disjointExact;
            disjointExact.q = makeBuffer(
                device, candidateQ, @"slot-1 candidate q source");
            disjointExact.body = makeBuffer(
                device, body, @"slot-1 candidate body source");
            disjointExact.jacobian = makeBuffer(
                device, jacobian, @"slot-1 candidate Jacobian source");
            auto fabricatedAccept = prepareTransaction(
                adapter, matter, queue, disjointArenas, disjointExact,
                8u, 43u, false,
                true, 1u);
            const auto fabricatedAcceptApply = applyTransaction(
                matter, queue, disjointArenas, fabricatedAccept,
                ApplyMode::fabricatedAcceptOnRejected, false);
            require(equalAcceptedAuthority(
                        fabricatedAccept.before,
                        fabricatedAcceptApply.after) &&
                    std::memcmp(
                        accepted.finalTokens.contents, &acceptedToken,
                        sizeof(acceptedToken)) == 0,
                "fabricated ACCEPT over REJECTED binding changed a root");

            verifyTerminalPending(world, device, queue);
            verifyOwnerAdapterMatterIntegration(device);
            std::cout
                << "numanx_human_matter_adapter_probe=pass\n"
                << "matter_runtime=real_attached\n"
                << "control_step=37_nonzero_and_mismatch_rejected\n"
                << "callback_operations=all_four_exact_host_gated\n"
                << "reaction_consumption=exactly_once\n"
                << "heap_alias=distinct_object_overlap_rejected\n"
                << "cross_slot_authority=lease_and_physical_alias_rejected\n"
                << "first_command_abort=runtime_cancel_and_slot_reuse\n"
                << "command_buffer_address_reuse=admitted_by_live_apply\n"
                << "prepared_accept=canonical_token_fnv_reserved_zero\n"
                << "prepared_reject=human_matter_byte_restore\n"
                << "physical_reject=zero_gate_exact_code_and_restore\n"
                << "final_abort_retry=pass\n"
                << "replay_and_prior_token_stability=pass\n"
                << "terminal_pending=no_touch_quarantine\n"
                << "slot_reuse=resolved_generation_only\n"
                << "owner_adapter_matter=integrated_force_reject\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "numanx_human_matter_adapter_probe=fail\n"
                      << "reason=" << exception.what() << '\n';
            return 1;
        }
    }
}
