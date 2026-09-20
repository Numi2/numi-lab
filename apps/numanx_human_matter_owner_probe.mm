#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"
#include "metalrobo/numanx_human_matter_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kEnvironmentCount = 1u;
constexpr std::uint32_t kNq = 161u;
constexpr std::uint32_t kNv = 160u;
constexpr std::uint32_t kBodyCount = 1u;
constexpr std::uint32_t kPointCount = 4u;
constexpr std::uint32_t kPointJacobianStride =
    kPointCount * 3u * kNv;
constexpr std::uint32_t kMujocoStateCount = 3u;
constexpr float kTimestep = 0.01f;
constexpr std::uint64_t kProgramFingerprint = 0x484d4f574e455231ull;
constexpr std::uint64_t kTransactionFingerprint = 0x5452414e53414354ull;
constexpr std::uint64_t kLinearizationEpoch = 0x4c494e4541523031ull;
constexpr std::uint64_t kSlotGeneration = 1u;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

MRCompensatedRootTranslationGPU initialTranslation() {
    MRCompensatedRootTranslationGPU root{
        {1.0f, -2.0f, 0.5f, 0.0f},
        {0x1p-26f, -0x1p-26f, 0x1p-27f, 0.0f},
        {0x1p-52f, 0x1p-52f, -0x1p-53f, 0.0f}};
    require(mrCompensatedTranslationValid(root), "noncanonical root fixture");
    const auto projected = mrCompensatedTranslationProjection(root);
    require(std::memcmp(&projected, &root.reference, sizeof(projected)) == 0,
        "fixture low root state must be hidden by q projection");
    return root;
}

std::uint64_t rootHash(const MRCompensatedRootTranslationGPU& root) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&root);
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0u; i < sizeof(root); ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash;
}

void mixTokenU32(std::uint64_t& hash, const std::uint32_t value) {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        hash = (hash ^ static_cast<std::uint8_t>(value >> (8u * byte))) *
            1099511628211ull;
    }
}

void mixTokenU64(std::uint64_t& hash, const std::uint64_t value) {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash = (hash ^ static_cast<std::uint8_t>(value >> (8u * byte))) *
            1099511628211ull;
    }
}

std::uint64_t tokenFingerprint(
    const MRNumanXAcceptedPhysicsStateTokenGPU& token
) {
    std::uint64_t hash = 14695981039346656037ull;
    mixTokenU32(hash, 1u);
    mixTokenU64(hash, token.transactionFingerprint);
    mixTokenU64(hash, token.substepFingerprint);
    mixTokenU64(hash, token.physicsStateFingerprint);
    mixTokenU64(hash, token.acceptedTimestampMicroseconds);
    mixTokenU64(hash, token.physicsGeneration);
    mixTokenU32(hash, token.environmentIdentifier);
    mixTokenU32(hash, token.flags);
    mixTokenU64(hash, token.reserved);
    return hash;
}

void initializeRootQ(std::vector<float>& q) {
    const auto projected = mrCompensatedTranslationProjection(initialTranslation());
    q[0] = projected.x; q[1] = projected.y; q[2] = projected.z;
}

template <typename T>
id<MTLBuffer> bufferWithValues(
    id<MTLDevice> device,
    const std::vector<T>& values,
    NSString* label
) {
    const NSUInteger bytes = std::max<NSUInteger>(
        values.size() * sizeof(T), sizeof(T));
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(buffer != nil && buffer.contents != nullptr,
            "failed to allocate shared Metal buffer");
    buffer.label = label;
    std::memset(buffer.contents, 0, bytes);
    if (!values.empty()) {
        std::memcpy(buffer.contents, values.data(), values.size() * sizeof(T));
    }
    return buffer;
}

template <typename T>
id<MTLBuffer> zeroBuffer(
    id<MTLDevice> device,
    const std::size_t count,
    NSString* label
) {
    return bufferWithValues<T>(device, std::vector<T>(count), label);
}

template <typename T>
std::vector<T> snapshot(id<MTLBuffer> buffer, const std::size_t count) {
    require(buffer != nil && buffer.length >= count * sizeof(T),
            "snapshot buffer is too small");
    const T* values = static_cast<const T*>(buffer.contents);
    return std::vector<T>(values, values + count);
}

void encodeEnvironmentGroups(
    id<MTLComputeCommandEncoder> encoder,
    const NSUInteger width = 256u
) {
    [encoder dispatchThreadgroups:MTLSizeMake(kEnvironmentCount, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

struct Pipelines {
    id<MTLComputePipelineState> begin = nil;
    id<MTLComputePipelineState> factor = nil;
    id<MTLComputePipelineState> consume = nil;
    id<MTLComputePipelineState> preparePhysical = nil;
    id<MTLComputePipelineState> markPhysicalComplete = nil;
    id<MTLComputePipelineState> stand = nil;
};

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    require(function != nil, "missing Metal function " +
        std::string{name.UTF8String});
    NSError* error = nil;
    id<MTLComputePipelineState> result = [device
        newComputePipelineStateWithFunction:function error:&error];
    const char* pipelineError = error.localizedDescription.UTF8String;
    require(result != nil, "pipeline creation failed: " +
        std::string{pipelineError != nullptr ? pipelineError : "unknown"});
    return result;
}

struct Fixture {
    id<MTLDevice> device = nil;
    Pipelines pipelines{};

    id<MTLBuffer> worlds = nil;
    id<MTLBuffer> articulations = nil;
    id<MTLBuffer> dofs = nil;
    id<MTLBuffer> bodies = nil;
    id<MTLBuffer> bodyPoses = nil;
    id<MTLBuffer> bodyPositionLow = nil;
    id<MTLBuffer> pointWorld = nil;
    id<MTLBuffer> pointPositionLow = nil;
    id<MTLBuffer> pointJacobians = nil;
    id<MTLBuffer> contacts = nil;
    id<MTLBuffer> spatial = nil;
    id<MTLBuffer> bodyMotion = nil;
    id<MTLBuffer> vectorScratch = nil;
    id<MTLBuffer> response = nil;
    id<MTLBuffer> standStatuses = nil;
    id<MTLBuffer> tendonBindings = nil;
    id<MTLBuffer> tendonTransfers = nil;
    id<MTLBuffer> equalities = nil;
    id<MTLBuffer> passiveJointProgram = nil;

    MRNumiHumanStandDispatchGPU standDispatch{};
    MRNumanXHumanMatterDispatchGPU ownerDispatch{};

    Fixture(id<MTLDevice> selected, id<MTLLibrary> library)
        : device(selected) {
        pipelines.begin = pipeline(
            device, library, @"mr_numanx_human_matter_begin");
        pipelines.factor = pipeline(
            device, library,
            @"mr_numanx_human_matter_source_factor");
        pipelines.consume = pipeline(
            device, library,
            @"mr_numanx_human_matter_consume_reaction");
        pipelines.preparePhysical = pipeline(
            device, library, @"mr_numanx_human_matter_prepare_physical");
        pipelines.markPhysicalComplete = pipeline(
            device, library,
            @"mr_numanx_human_matter_mark_physical_complete");
        pipelines.stand = pipeline(
            device, library, @"mr_numi_human_stand_step");

        MRWorldGPU world{};
        world.articulationCount = 1u;
        world.bodyCount = kBodyCount;
        world.nq = kNq;
        world.nv = kNv;
        world.gravityAndTimestep = {0.0f, 0.0f, 0.0f, kTimestep};
        worlds = bufferWithValues<MRWorldGPU>(device, {world}, @"world");

        MRArticulationGPU articulation{};
        articulation.firstBody = 0u;
        articulation.bodyCount = kBodyCount;
        articulation.qOffset = 0u;
        articulation.nq = kNq;
        articulation.vOffset = 0u;
        articulation.nv = kNv;
        articulation.rootType = MR_ROOT_FLOATING;
        articulations = bufferWithValues<MRArticulationGPU>(
            device, {articulation}, @"articulation");

        std::vector<MRDofPropertiesGPU> dofValues(kNv);
        for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
            MRDofPropertiesGPU value{};
            value.qIndex = dof < 6u ? MR_INVALID_INDEX : dof + 1u;
            value.drive.y = 0.2f + 0.002f * static_cast<float>(dof);
            value.drive.z = 0.5f + 0.001f * static_cast<float>(dof);
            value.flags = 0u;
            dofValues[dof] = value;
        }
        dofs = bufferWithValues<MRDofPropertiesGPU>(
            device, dofValues, @"dofs");

        MRBodyPropertiesGPU body{};
        body.massAndInverseMass = {3.0f, 1.0f / 3.0f, 0.0f, 0.0f};
        body.inertiaRow0 = {4.0f, 0.0f, 0.0f, 0.0f};
        body.inertiaRow1 = {0.0f, 5.0f, 0.0f, 0.0f};
        body.inertiaRow2 = {0.0f, 0.0f, 6.0f, 0.0f};
        bodies = bufferWithValues<MRBodyPropertiesGPU>(
            device, {body}, @"bodies");

        MRArticulatedBodyPoseGPU pose{};
        const auto translation = initialTranslation();
        const auto bodyPosition = mrCompensatedTranslationPosition(translation, {0.0f, 0.0f, 0.0f, 0.0f});
        pose.position = bodyPosition.high;
        pose.position.w = 1.0f;
        pose.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        bodyPoses = bufferWithValues<MRArticulatedBodyPoseGPU>(
            device, {pose}, @"body poses");
        bodyPositionLow = bufferWithValues<mr_float4>(
            device, {bodyPosition.low}, @"body position low");

        std::vector<MRArticulatedPointWorldGPU> worldsAtPoints(kPointCount);
        worldsAtPoints[1].position = {1.0f, 0.0f, 0.0f, 1.0f};
        worldsAtPoints[2].position = {0.0f, 1.0f, 0.0f, 1.0f};
        worldsAtPoints[3].position = {0.0f, 0.0f, 1.0f, 1.0f};
        std::vector<mr_float4> pointLows(kPointCount);
        for (std::uint32_t point = 0u; point < kPointCount; ++point) {
            const auto paired = mrCompensatedTranslationPosition(translation,
                worldsAtPoints[point].position);
            worldsAtPoints[point].position = paired.high;
            worldsAtPoints[point].position.w = 1.0f;
            pointLows[point] = paired.low;
        }
        pointWorld = bufferWithValues<MRArticulatedPointWorldGPU>(
            device, worldsAtPoints, @"point world");
        pointPositionLow = bufferWithValues<mr_float4>(
            device, pointLows, @"point position low");

        std::vector<float> jacobians(kPointJacobianStride, 0.0f);
        const std::array<std::array<float, 3u>, 4u> localPoints{{
            {{0.0f, 0.0f, 0.0f}},
            {{1.0f, 0.0f, 0.0f}},
            {{0.0f, 1.0f, 0.0f}},
            {{0.0f, 0.0f, 1.0f}},
        }};
        for (std::uint32_t point = 0u; point < kPointCount; ++point) {
            const auto r = localPoints[point];
            for (std::uint32_t dof = 0u; dof < 6u; ++dof) {
                std::array<float, 3u> linear{{0.0f, 0.0f, 0.0f}};
                if (dof < 3u) {
                    linear[dof] = 1.0f;
                } else {
                    std::array<float, 3u> omega{{0.0f, 0.0f, 0.0f}};
                    omega[dof - 3u] = 1.0f;
                    linear = {{
                        omega[1] * r[2] - omega[2] * r[1],
                        omega[2] * r[0] - omega[0] * r[2],
                        omega[0] * r[1] - omega[1] * r[0],
                    }};
                }
                for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                    jacobians[point * 3u * kNv + axis * kNv + dof] =
                        linear[axis];
                }
            }
        }
        pointJacobians = bufferWithValues<float>(
            device, jacobians, @"point Jacobians");

        contacts = zeroBuffer<MRNumiHumanStandContactGPU>(
            device, 1u, @"contacts");
        spatial = zeroBuffer<float>(
            device,
            kBodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * kNv,
            @"spatial Jacobian");
        bodyMotion = zeroBuffer<mr_float4>(
            device, kBodyCount * 2u, @"body motion");
        vectorScratch = zeroBuffer<float>(
            device, 4u * kNv, @"stand vectors");
        response = zeroBuffer<float>(
            device, kNv * kNv, @"response");
        standStatuses = zeroBuffer<MRNumiHumanStandStatusGPU>(
            device, kEnvironmentCount, @"stand statuses");
        tendonBindings = zeroBuffer<MRNumiHumanTendonBindingGPU>(
            device, 1u, @"tendon bindings");
        tendonTransfers = zeroBuffer<MRNumiHumanTendonTransferResultGPU>(
            device, 1u, @"tendon transfers");
        equalities = zeroBuffer<MRNumiHumanJointEqualityGPU>(
            device, 1u, @"equalities");
        passiveJointProgram = zeroBuffer<float>(
            device, 1u, @"empty passive joint program");

        standDispatch.abiVersion = MR_NUMI_HUMAN_STAND_ABI_VERSION;
        standDispatch.environmentCount = kEnvironmentCount;
        standDispatch.articulationIndex = 0u;
        standDispatch.stepIndex = 0u;
        standDispatch.stepCount = 1u;
        standDispatch.bodyJacobianPointOffset = 0u;
        standDispatch.supportContactCount = 0u;
        standDispatch.flags = 0u;
        standDispatch.qStride = kNq;
        standDispatch.vStride = kNv;
        standDispatch.pointWorldStride = kPointCount;
        standDispatch.pointJacobianStride = kPointJacobianStride;
        standDispatch.bodyPoseStride = kBodyCount;
        standDispatch.generalizedForceStride = kNv;
        standDispatch.generalizedForceOffset = 0u;
        standDispatch.contactIterationCount = 1u;
        standDispatch.tendonTransferStride = 0u;
        standDispatch.groundPointAndTimestep = {
            0.0f, 0.0f, 0.0f, kTimestep};
        standDispatch.groundNormal = {0.0f, 1.0f, 0.0f, 0.0f};
        standDispatch.targetRootOrientation = {0.0f, 0.0f, 0.0f, 1.0f};

        ownerDispatch.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
        ownerDispatch.environmentCount = kEnvironmentCount;
        ownerDispatch.stepIndex = 0u;
        ownerDispatch.flags = MR_NUMANX_HUMAN_MATTER_HAS_PREPARED_TOKEN;
        ownerDispatch.nq = kNq;
        ownerDispatch.nv = kNv;
        ownerDispatch.qStride = kNq;
        ownerDispatch.vStride = kNv;
        ownerDispatch.mujocoStateStride = kMujocoStateCount;
        ownerDispatch.mujocoStateCount = kMujocoStateCount;
        ownerDispatch.generalizedForceStride = kNv;
        ownerDispatch.generalizedForceOffset = 0u;
        ownerDispatch.reactionStride = kNv;
        ownerDispatch.jointStatusStride = 1u;
        ownerDispatch.acceptedTokenStrideBytes =
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
        ownerDispatch.ownerStatusStride = 1u;
        ownerDispatch.physicsSubstepCount =
            MR_NUMANX_HUMAN_MATTER_PHYSICS_SUBSTEP_COUNT;
        ownerDispatch.controlStep = 37u;
        ownerDispatch.timestepAndInverse = {
            kTimestep, 1.0f / kTimestep, 0.0f, 0.0f};
        ownerDispatch.programFingerprint = kProgramFingerprint;
        ownerDispatch.transactionFingerprint = kTransactionFingerprint;
        ownerDispatch.linearizationEpoch = kLinearizationEpoch;
        ownerDispatch.slotGeneration = kSlotGeneration;
    }
};

enum class Outcome { accept, reject, duplicateConsume };

struct RunResult {
    MRCompensatedRootTranslationGPU root{};
    std::vector<float> q;
    std::vector<float> v;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::vector<float> forces;
    std::vector<float> sourceFactor;
    std::vector<float> standFactor;
    std::array<std::uint8_t, 64u> token{};
    MRNumanXHumanMatterOwnerStatusGPU owner{};
    MRNumiHumanStandStatusGPU stand{};
};

RunResult run(Fixture& fixture, const Outcome outcome) {
    std::vector<float> initialQ(kNq, 0.0f);
    initializeRootQ(initialQ);
    initialQ[6u] = 1.0f;
    for (std::uint32_t coordinate = 7u; coordinate < kNq; ++coordinate) {
        initialQ[coordinate] = 0.001f * static_cast<float>(coordinate);
    }
    std::vector<float> initialV(kNv, 0.0f);
    for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
        initialV[dof] = 0.01f * std::sin(static_cast<float>(dof));
    }
    std::vector<MRMujocoMuscleStateGPU> initialStates(kMujocoStateCount);
    for (std::uint32_t index = 0u; index < kMujocoStateCount; ++index) {
        initialStates[index].excitationAndActivation = {
            0.1f * index, 0.2f * index, 0.0f, 0.0f};
    }
    std::vector<MRMujocoMuscleStateGPU> mutatedStates = initialStates;
    for (auto& state : mutatedStates) {
        state.excitationAndActivation.y += 0.25f;
    }
    std::vector<float> reactions(kNv, 0.0f);
    for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
        reactions[dof] = 0.05f + 0.0001f * static_cast<float>(dof);
    }

    id<MTLBuffer> q = bufferWithValues<float>(
        fixture.device, initialQ, @"q");
    const auto initialRoot = initialTranslation();
    id<MTLBuffer> root = bufferWithValues<MRCompensatedRootTranslationGPU>(
        fixture.device, {initialRoot}, @"compensated root");
    id<MTLBuffer> rootCheckpoint = zeroBuffer<MRCompensatedRootTranslationGPU>(
        fixture.device, kEnvironmentCount, @"compensated root checkpoint");
    id<MTLBuffer> v = bufferWithValues<float>(
        fixture.device, initialV, @"v");
    id<MTLBuffer> states = bufferWithValues<MRMujocoMuscleStateGPU>(
        fixture.device, initialStates, @"MyoSim states");
    id<MTLBuffer> mutated = bufferWithValues<MRMujocoMuscleStateGPU>(
        fixture.device, mutatedStates, @"mutated MyoSim states");
    id<MTLBuffer> qCheckpoint = zeroBuffer<float>(
        fixture.device, kNq, @"q checkpoint");
    id<MTLBuffer> vCheckpoint = zeroBuffer<float>(
        fixture.device, kNv, @"v checkpoint");
    id<MTLBuffer> stateCheckpoint = zeroBuffer<MRMujocoMuscleStateGPU>(
        fixture.device, kMujocoStateCount, @"state checkpoint");
    id<MTLBuffer> factor = zeroBuffer<float>(
        fixture.device, kNv * kNv, @"factor");
    id<MTLBuffer> factorSnapshot = zeroBuffer<float>(
        fixture.device, kNv * kNv, @"factor snapshot");
    id<MTLBuffer> forces = zeroBuffer<float>(
        fixture.device, kNv, @"forces");
    id<MTLBuffer> reaction = bufferWithValues<float>(
        fixture.device, reactions, @"reaction");
    id<MTLBuffer> joint = zeroBuffer<MRNumanXCoupledHumanStatusGPU>(
        fixture.device, 1u, @"joint status");
    id<MTLBuffer> postJoint = zeroBuffer<MRNumanXCoupledHumanStatusGPU>(
        fixture.device, 1u, @"post joint status");
    id<MTLBuffer> token = zeroBuffer<std::uint8_t>(
        fixture.device, 64u, @"accepted token");
    id<MTLBuffer> postToken = zeroBuffer<std::uint8_t>(
        fixture.device, 64u, @"post accepted token");
    id<MTLBuffer> owner = zeroBuffer<MRNumanXHumanMatterOwnerStatusGPU>(
        fixture.device, 1u, @"owner status");
    id<MTLBuffer> applied = zeroBuffer<MRNumanXHumanMatterAppliedOutcomeGPU>(
        fixture.device, 1u, @"applied outcome");
    id<MTLBuffer> finalToken = zeroBuffer<std::uint8_t>(
        fixture.device, 64u, @"final accepted token");

    auto* pending = static_cast<MRNumanXCoupledHumanStatusGPU*>(
        joint.contents);
    pending->abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    pending->decision = MR_NUMANX_COUPLED_HUMAN_PENDING;
    pending->environment = 0u;
    pending->stepIndex = 0u;
    pending->humanCode = 0u;
    pending->matterCode = 0u;
    pending->matterCompletedMicrosteps = 0u;

    auto* resolved = static_cast<MRNumanXCoupledHumanStatusGPU*>(
        postJoint.contents);
    *resolved = *pending;
    resolved->decision = outcome == Outcome::accept
        ? MR_NUMANX_COUPLED_HUMAN_ACCEPT
        : MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER;
    resolved->humanCode = MR_NUMI_HUMAN_STAND_SUCCESS;
    resolved->humanCompletedSteps = 1u;
    resolved->matterCompletedMicrosteps = 1u;
    MRNumanXAcceptedPhysicsStateTokenGPU preparedToken{};
    preparedToken.transactionFingerprint = kTransactionFingerprint;
    preparedToken.substepFingerprint = 0x5355425354455031ull;
    preparedToken.physicsStateFingerprint = 0x5048595349435331ull;
    preparedToken.acceptedTimestampMicroseconds = 10'000u;
    preparedToken.physicsGeneration = 1u;
    preparedToken.environmentIdentifier = 0u;
    preparedToken.flags = 0u;
    preparedToken.reserved = 0u;
    preparedToken.tokenFingerprint = tokenFingerprint(preparedToken);
    std::memcpy(
        postToken.contents, &preparedToken, sizeof(preparedToken));

    id<MTLCommandQueue> queue = [fixture.device newCommandQueue];
    require(queue != nil, "failed to create command queue");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "failed to create command buffer");

    id<MTLBlitCommandEncoder> checkpoint = [command blitCommandEncoder];
    [checkpoint copyFromBuffer:root sourceOffset:0u toBuffer:rootCheckpoint
             destinationOffset:0u size:sizeof(MRCompensatedRootTranslationGPU)];
    [checkpoint copyFromBuffer:q sourceOffset:0u toBuffer:qCheckpoint
             destinationOffset:0u size:kNq * sizeof(float)];
    [checkpoint copyFromBuffer:v sourceOffset:0u toBuffer:vCheckpoint
             destinationOffset:0u size:kNv * sizeof(float)];
    [checkpoint copyFromBuffer:states sourceOffset:0u toBuffer:stateCheckpoint
             destinationOffset:0u
                          size:kMujocoStateCount *
                              sizeof(MRMujocoMuscleStateGPU)];
    [checkpoint endEncoding];

    id<MTLComputeCommandEncoder> begin = [command computeCommandEncoder];
    [begin setComputePipelineState:fixture.pipelines.begin];
    [begin setBytes:&fixture.ownerDispatch
             length:sizeof(fixture.ownerDispatch) atIndex:0u];
    [begin setBuffer:token offset:0u atIndex:1u];
    [begin setBuffer:owner offset:0u atIndex:2u];
    [begin setBuffer:applied offset:0u atIndex:3u];
    [begin setBuffer:finalToken offset:0u atIndex:4u];
    [begin dispatchThreads:MTLSizeMake(1u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [begin endEncoding];

    id<MTLBlitCommandEncoder> mutate = [command blitCommandEncoder];
    [mutate copyFromBuffer:mutated sourceOffset:0u toBuffer:states
          destinationOffset:0u
                       size:kMujocoStateCount *
                           sizeof(MRMujocoMuscleStateGPU)];
    [mutate endEncoding];

    id<MTLComputeCommandEncoder> factorEncoder =
        [command computeCommandEncoder];
    [factorEncoder setComputePipelineState:fixture.pipelines.factor];
    [factorEncoder setBuffer:fixture.worlds offset:0u atIndex:0u];
    [factorEncoder setBuffer:fixture.articulations offset:0u atIndex:1u];
    [factorEncoder setBuffer:fixture.dofs offset:0u atIndex:2u];
    [factorEncoder setBuffer:fixture.bodies offset:0u atIndex:3u];
    [factorEncoder setBytes:&fixture.standDispatch
                     length:sizeof(fixture.standDispatch) atIndex:4u];
    [factorEncoder setBytes:&fixture.ownerDispatch
                     length:sizeof(fixture.ownerDispatch) atIndex:5u];
    [factorEncoder setBuffer:fixture.bodyPoses offset:0u atIndex:6u];
    [factorEncoder setBuffer:fixture.pointJacobians offset:0u atIndex:7u];
    [factorEncoder setBuffer:fixture.spatial offset:0u atIndex:8u];
    [factorEncoder setBuffer:factor offset:0u atIndex:9u];
    [factorEncoder setBuffer:owner offset:0u atIndex:10u];
    encodeEnvironmentGroups(factorEncoder);
    [factorEncoder endEncoding];

    id<MTLBlitCommandEncoder> saveFactor = [command blitCommandEncoder];
    [saveFactor copyFromBuffer:factor sourceOffset:0u toBuffer:factorSnapshot
             destinationOffset:0u size:kNv * kNv * sizeof(float)];
    [saveFactor endEncoding];

    const auto encodeConsume = [&] {
        id<MTLComputeCommandEncoder> consume =
            [command computeCommandEncoder];
        [consume setComputePipelineState:fixture.pipelines.consume];
        [consume setBytes:&fixture.ownerDispatch
                   length:sizeof(fixture.ownerDispatch) atIndex:0u];
        [consume setBuffer:reaction offset:0u atIndex:1u];
        [consume setBuffer:joint offset:0u atIndex:2u];
        [consume setBuffer:forces offset:0u atIndex:3u];
        [consume setBuffer:owner offset:0u atIndex:4u];
        // Metal dynamic threadgroup allocations are 16-byte granular, even
        // though this kernel uses only one uint flag.
        [consume setThreadgroupMemoryLength:16u atIndex:0u];
        encodeEnvironmentGroups(consume);
        [consume endEncoding];
    };
    encodeConsume();
    if (outcome == Outcome::duplicateConsume) encodeConsume();

    id<MTLComputeCommandEncoder> stand = [command computeCommandEncoder];
    [stand setComputePipelineState:fixture.pipelines.stand];
    [stand setBuffer:fixture.worlds offset:0u atIndex:0u];
    [stand setBuffer:fixture.articulations offset:0u atIndex:1u];
    [stand setBuffer:fixture.dofs offset:0u atIndex:2u];
    [stand setBuffer:fixture.bodies offset:0u atIndex:3u];
    [stand setBytes:&fixture.standDispatch
             length:sizeof(fixture.standDispatch) atIndex:4u];
    [stand setBuffer:q offset:0u atIndex:5u];
    [stand setBuffer:v offset:0u atIndex:6u];
    [stand setBuffer:fixture.bodyPoses offset:0u atIndex:7u];
    [stand setBuffer:fixture.pointWorld offset:0u atIndex:8u];
    [stand setBuffer:fixture.pointJacobians offset:0u atIndex:9u];
    [stand setBuffer:forces offset:0u atIndex:10u];
    [stand setBuffer:fixture.contacts offset:0u atIndex:11u];
    [stand setBuffer:fixture.spatial offset:0u atIndex:12u];
    [stand setBuffer:fixture.bodyMotion offset:0u atIndex:13u];
    [stand setBuffer:factor offset:0u atIndex:14u];
    [stand setBuffer:fixture.vectorScratch offset:0u atIndex:15u];
    [stand setBuffer:fixture.response offset:0u atIndex:16u];
    [stand setBuffer:fixture.standStatuses offset:0u atIndex:17u];
    [stand setBuffer:fixture.tendonBindings offset:0u atIndex:18u];
    [stand setBuffer:fixture.tendonTransfers offset:0u atIndex:19u];
    [stand setBuffer:fixture.equalities offset:0u atIndex:20u];
    [stand setBuffer:root offset:0u atIndex:21u];
    [stand setBuffer:fixture.bodyPositionLow offset:0u atIndex:22u];
    [stand setBuffer:fixture.pointPositionLow offset:0u atIndex:23u];
    [stand setBuffer:fixture.passiveJointProgram offset:0u atIndex:24u];
    encodeEnvironmentGroups(stand);
    [stand endEncoding];

    id<MTLBlitCommandEncoder> resolve = [command blitCommandEncoder];
    [resolve copyFromBuffer:postJoint sourceOffset:0u toBuffer:joint
              destinationOffset:0u
                           size:sizeof(MRNumanXCoupledHumanStatusGPU)];
    [resolve copyFromBuffer:postToken sourceOffset:0u toBuffer:token
              destinationOffset:0u size:64u];
    [resolve endEncoding];

    id<MTLComputeCommandEncoder> preparePhysical =
        [command computeCommandEncoder];
    [preparePhysical setComputePipelineState:fixture.pipelines.preparePhysical];
    [preparePhysical setBytes:&fixture.ownerDispatch
                       length:sizeof(fixture.ownerDispatch) atIndex:0u];
    [preparePhysical setBuffer:joint offset:0u atIndex:1u];
    [preparePhysical setBuffer:token offset:0u atIndex:2u];
    [preparePhysical setBuffer:q offset:0u atIndex:3u];
    [preparePhysical setBuffer:v offset:0u atIndex:4u];
    [preparePhysical setBuffer:states offset:0u atIndex:5u];
    [preparePhysical setBuffer:qCheckpoint offset:0u atIndex:6u];
    [preparePhysical setBuffer:vCheckpoint offset:0u atIndex:7u];
    [preparePhysical setBuffer:stateCheckpoint offset:0u atIndex:8u];
    [preparePhysical setBuffer:owner offset:0u atIndex:9u];
    [preparePhysical setBuffer:root offset:0u atIndex:10u];
    [preparePhysical setBuffer:rootCheckpoint offset:0u atIndex:11u];
    // Match the same dynamic-threadgroup allocation granularity as consume.
    [preparePhysical setThreadgroupMemoryLength:16u atIndex:0u];
    encodeEnvironmentGroups(preparePhysical);
    [preparePhysical endEncoding];

    id<MTLComputeCommandEncoder> markPhysicalComplete =
        [command computeCommandEncoder];
    [markPhysicalComplete setComputePipelineState:
        fixture.pipelines.markPhysicalComplete];
    [markPhysicalComplete setBytes:&fixture.ownerDispatch
                            length:sizeof(fixture.ownerDispatch) atIndex:0u];
    [markPhysicalComplete setBuffer:owner offset:0u atIndex:1u];
    [markPhysicalComplete dispatchThreads:MTLSizeMake(1u, 1u, 1u)
                         threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [markPhysicalComplete endEncoding];

    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "owner transaction command buffer failed");

    RunResult result{};
    result.root = *static_cast<const MRCompensatedRootTranslationGPU*>(root.contents);
    require(std::memcmp(rootCheckpoint.contents, &initialRoot, sizeof(initialRoot)) == 0 &&
        rootHash(*static_cast<const MRCompensatedRootTranslationGPU*>(rootCheckpoint.contents)) == rootHash(initialRoot),
        "48-byte root checkpoint was mutated");
    result.q = snapshot<float>(q, kNq);
    result.v = snapshot<float>(v, kNv);
    result.states = snapshot<MRMujocoMuscleStateGPU>(
        states, kMujocoStateCount);
    result.forces = snapshot<float>(forces, kNv);
    result.sourceFactor = snapshot<float>(factorSnapshot, kNv * kNv);
    result.standFactor = snapshot<float>(factor, kNv * kNv);
    std::memcpy(result.token.data(), token.contents, result.token.size());
    result.owner = *static_cast<MRNumanXHumanMatterOwnerStatusGPU*>(
        owner.contents);
    result.stand = *static_cast<MRNumiHumanStandStatusGPU*>(
        fixture.standStatuses.contents);
    return result;
}

bool sameBytes(const auto& left, const auto& right) {
    return left.size() == right.size() &&
        std::memcmp(
            left.data(), right.data(), left.size() * sizeof(left[0])) == 0;
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: numanx_human_matter_owner_probe <metallib>");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");
            NSString* path = [NSString stringWithUTF8String:argv[1]];
            NSError* error = nil;
            id<MTLLibrary> library = [device
                newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            const char* libraryError = error.localizedDescription.UTF8String;
            require(library != nil, "failed to load metallib: " +
                std::string{libraryError != nullptr
                    ? libraryError
                    : "unknown"});
            Fixture fixture{device, library};

            const RunResult accepted = run(fixture, Outcome::accept);
            const RunResult rejected = run(fixture, Outcome::reject);
            const RunResult replay = run(fixture, Outcome::reject);
            const RunResult duplicate = run(
                fixture, Outcome::duplicateConsume);

            require(accepted.owner.stage ==
                        MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_PREPARED &&
                    accepted.owner.reactionConsumed == 1u &&
                    accepted.owner.restored == 0u &&
                    accepted.owner.preparedTokenPreserved == 1u &&
                    accepted.owner.physicalCommandStatus ==
                        MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE,
                    "accepted owner witness is wrong: stage=" +
                        std::to_string(accepted.owner.stage) +
                        " reaction=" +
                        std::to_string(accepted.owner.reactionConsumed) +
                        " restored=" +
                        std::to_string(accepted.owner.restored) +
                        " token=" +
                        std::to_string(
                            accepted.owner.preparedTokenPreserved) +
                        " physical=" +
                        std::to_string(
                            accepted.owner.physicalCommandStatus) +
                        " stand_code=" +
                        std::to_string(accepted.stand.code) +
                        " stand_steps=" +
                        std::to_string(accepted.stand.completedSteps));
            require(accepted.stand.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                    accepted.stand.completedSteps == 1u,
                    "stand did not complete accepted step");
            require(sameBytes(accepted.sourceFactor, accepted.standFactor),
                    "owner A0 factor bytes differ from stand factor bytes");
            for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
                const float inertial = dof < 3u ? 3.0f
                    : dof == 3u ? 4.0f
                    : dof == 4u ? 5.0f
                    : dof == 5u ? 6.0f : 0.0f;
                const float expected = std::sqrt(
                    inertial + 0.5f + 0.001f * dof +
                    kTimestep * (0.2f + 0.002f * dof));
                require(std::abs(
                            accepted.sourceFactor[dof * kNv + dof] -
                            expected) < 3.0e-6f,
                        "A0 diagonal does not include exact armature+hD");
                require(std::abs(
                            accepted.forces[dof] -
                            (0.05f + 0.0001f * dof)) < 1.0e-7f,
                        "reaction was not consumed exactly once");
            }
            require(accepted.token[56u] != 0u,
                    "accepted token was not preserved");

            std::vector<float> initialQ(kNq, 0.0f);
            initializeRootQ(initialQ);
            initialQ[6u] = 1.0f;
            for (std::uint32_t coordinate = 7u;
                 coordinate < kNq; ++coordinate) {
                initialQ[coordinate] = 0.001f * coordinate;
            }
            std::vector<float> initialV(kNv, 0.0f);
            for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
                initialV[dof] = 0.01f * std::sin(static_cast<float>(dof));
            }
            require(sameBytes(rejected.q, initialQ) &&
                    sameBytes(rejected.v, initialV),
                    "rejected q/v were not restored byte-for-byte");
            const auto initialRoot = initialTranslation();
            const auto projectedOnly = mrCompensatedTranslationFromProjection(initialRoot.reference);
            require(rootHash(initialRoot) != rootHash(projectedOnly),
                "root hash failed to distinguish hidden state with identical q");
            for (const auto* restored : {&rejected.root, &replay.root, &duplicate.root})
                require(std::memcmp(restored, &initialRoot, sizeof(initialRoot)) == 0 &&
                    rootHash(*restored) == rootHash(initialRoot),
                    "rejected full 48-byte root restore/hash mismatch");
            require(mrCompensatedTranslationValid(accepted.root) &&
                std::memcmp(&accepted.root.reference, &initialRoot.reference, sizeof(mr_float4)) == 0 &&
                rootHash(accepted.root) != rootHash(initialRoot),
                "accepted compensated root did not advance with immutable origin");
            const auto acceptedProjection = mrCompensatedTranslationProjection(accepted.root);
            require(std::memcmp(&acceptedProjection, accepted.q.data(), 3u * sizeof(float)) == 0,
                "accepted root projection differs from q bytes");
            require(rejected.owner.stage ==
                        MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_REJECT_RESTORED &&
                    rejected.owner.restored == 1u &&
                    rejected.owner.physicalCommandStatus ==
                        MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE,
                    "rejected owner witness is wrong");
            require(std::all_of(
                        rejected.token.begin(), rejected.token.end(),
                        [](const std::uint8_t value) { return value == 0u; }),
                    "rejected accepted-physics token was not cleared");
            require(sameBytes(rejected.q, replay.q) &&
                    sameBytes(rejected.v, replay.v) &&
                    sameBytes(rejected.states, replay.states),
                    "rejected replay is not byte-identical");
            for (std::uint32_t dof = 0u; dof < kNv; ++dof) {
                require(std::abs(
                            duplicate.forces[dof] -
                            (0.05f + 0.0001f * dof)) < 1.0e-7f,
                        "duplicate consume added reaction twice");
            }
            require(duplicate.owner.stage ==
                        MR_NUMANX_HUMAN_MATTER_STAGE_PHYSICAL_REJECT_RESTORED &&
                    duplicate.owner.restored == 1u &&
                    duplicate.owner.physicalCommandStatus ==
                        MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE,
                    "duplicate consume did not fail closed and restore");

            std::cout
                << "PASS device=" << device.name.UTF8String
                << " dofs=160 q=161"
                << " factor=A0_M_armature_hD_byte_exact"
                << " reaction=prestand_once"
                << " accept=physical_prepared_quarantined"
                << " reject=restored"
                << " replay=byte_identical"
                << " root_state=48_byte_restore_hash_exact"
                << " duplicate_consume=fail_closed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}
