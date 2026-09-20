#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXCoupledHuman.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using metalrobo::MetalNumanXCoupledHumanArenaView;
using metalrobo::MetalNumanXCoupledHumanConfig;
using metalrobo::MetalNumanXCoupledHumanContext;
using metalrobo::MetalNumanXCoupledHumanOperation;
using metalrobo::MetalNumanXCoupledHumanPass;
using metalrobo::MetalNumanXCoupledHumanPhase;
using metalrobo::MetalNumanXCoupledHumanProgram;
using metalrobo::MetalNumanXCoupledHumanQuery;
using metalrobo::MetalNumanXCoupledHumanResolveQuery;

constexpr std::uint32_t kEnvironmentCount = 3u;
constexpr std::uint32_t kDofCount = 160u;
constexpr std::uint32_t kQCount = 161u;
constexpr std::uint32_t kBodyCount = 1u;
constexpr std::uint32_t kArticulationIndex = 2u;
constexpr std::uint32_t kFactorStride = kDofCount * kDofCount;
constexpr std::uint32_t kExpectedMatterMicrosteps = 5u;
constexpr std::uint32_t kMatterSuccess = 0u;
constexpr std::uint32_t kMatterFailure = 77u;
constexpr std::uint64_t kTransactionFingerprint =
    0x7a96b4e1d85c210full;
constexpr std::uint64_t kLinearizationEpoch = 0x31c02a93ull;
constexpr float kTimestepSeconds = 0.25f;
constexpr std::uint8_t kMarkerByte = 0xa5u;

constexpr std::size_t kQElementCount =
    static_cast<std::size_t>(kEnvironmentCount) * kQCount;
constexpr std::size_t kVectorElementCount =
    static_cast<std::size_t>(kEnvironmentCount) * kDofCount;
constexpr std::size_t kFactorElementCount =
    static_cast<std::size_t>(kEnvironmentCount) * kFactorStride;
constexpr std::size_t kBodyElementCount =
    static_cast<std::size_t>(kEnvironmentCount) * kBodyCount;
constexpr std::size_t kInverseStatusElementCount =
    static_cast<std::size_t>(kArticulationIndex + 1u) * kEnvironmentCount;

struct ProbeBuffers {
    id<MTLBuffer> sourceQ = nil;
    id<MTLBuffer> sourceV = nil;
    id<MTLBuffer> factorSeed = nil;
    id<MTLBuffer> sourceEffectiveTangentFactor = nil;
    id<MTLBuffer> tangentProjector = nil;
    id<MTLBuffer> bodySeed = nil;
    id<MTLBuffer> candidateQ = nil;
    id<MTLBuffer> candidateBodies = nil;
    id<MTLBuffer> massInput = nil;
    id<MTLBuffer> massOutput = nil;
    id<MTLBuffer> inverseInput = nil;
    id<MTLBuffer> inverseOutput = nil;
    id<MTLBuffer> inverseStatuses = nil;
    id<MTLBuffer> deltaV = nil;
    id<MTLBuffer> generalizedImpulse = nil;
    id<MTLBuffer> publishedMatterOutcomes = nil;
    id<MTLBuffer> matterOutcomes = nil;
    id<MTLBuffer> resolvedMatterOutcomes = nil;
    id<MTLBuffer> standStatuses = nil;
    id<MTLBuffer> callbackMarker = nil;
    id<MTLBuffer> pendingStatuses = nil;
    id<MTLBuffer> stagedReaction = nil;
    id<MTLBuffer> finalStatuses = nil;
    id<MTLBuffer> finalReaction = nil;
};

struct ExactCallbackContext {
    id<MTLBuffer> factorSeed = nil;
    id<MTLBuffer> bodySeed = nil;
    id<MTLBuffer> marker = nil;
    std::uint32_t invocationCount = 0u;
    std::uint32_t hostSequence = 0u;
    std::uint32_t callbackOrder = 0u;
    std::uint32_t encodeReturnOrder = 0u;
};

struct RunCapture {
    std::vector<std::uint8_t> bytes;
};

[[nodiscard]] int fail(const int code, const std::string& message) {
    std::fprintf(stderr, "numanx_coupled_human_probe: %s\n", message.c_str());
    return code;
}

[[nodiscard]] void* rawBuffer(id<MTLBuffer> buffer) noexcept {
    return (__bridge void*)buffer;
}

[[nodiscard]] std::uint64_t gpuAddress(id<MTLBuffer> buffer) noexcept {
    return static_cast<std::uint64_t>(buffer.gpuAddress);
}

[[nodiscard]] id<MTLBuffer> makeBuffer(
    id<MTLDevice> device,
    const std::size_t byteCount,
    const MTLResourceOptions options,
    NSString* label
) {
    id<MTLBuffer> buffer = [device
        newBufferWithLength:static_cast<NSUInteger>(byteCount)
                   options:options];
    if (buffer != nil) buffer.label = label;
    return buffer;
}

[[nodiscard]] bool allBuffersValid(const ProbeBuffers& buffers) noexcept {
    return buffers.sourceQ != nil && buffers.sourceV != nil &&
        buffers.factorSeed != nil &&
        buffers.sourceEffectiveTangentFactor != nil &&
        buffers.tangentProjector != nil &&
        buffers.bodySeed != nil && buffers.candidateQ != nil &&
        buffers.candidateBodies != nil && buffers.massInput != nil &&
        buffers.massOutput != nil && buffers.inverseInput != nil &&
        buffers.inverseOutput != nil && buffers.inverseStatuses != nil &&
        buffers.deltaV != nil && buffers.generalizedImpulse != nil &&
        buffers.publishedMatterOutcomes != nil &&
        buffers.matterOutcomes != nil &&
        buffers.resolvedMatterOutcomes != nil &&
        buffers.standStatuses != nil &&
        buffers.callbackMarker != nil && buffers.pendingStatuses != nil &&
        buffers.stagedReaction != nil && buffers.finalStatuses != nil &&
        buffers.finalReaction != nil;
}

[[nodiscard]] ProbeBuffers makeBuffers(id<MTLDevice> device) {
    constexpr MTLResourceOptions shared = MTLResourceStorageModeShared;
    constexpr MTLResourceOptions privateStorage =
        MTLResourceStorageModePrivate;
    constexpr std::size_t qBytes = kQElementCount * sizeof(float);
    constexpr std::size_t vectorBytes =
        kVectorElementCount * sizeof(float);
    constexpr std::size_t factorBytes =
        kFactorElementCount * sizeof(float);
    constexpr std::size_t bodyBytes =
        kBodyElementCount * sizeof(MRBodyStateGPU);
    constexpr std::size_t inverseStatusBytes =
        kInverseStatusElementCount * sizeof(MRInverseMassStatusGPU);
    constexpr std::size_t outcomeBytes =
        kEnvironmentCount * sizeof(MRNumanXCoupledMatterOutcomeGPU);
    constexpr std::size_t standStatusBytes =
        kEnvironmentCount * sizeof(MRNumiHumanStandStatusGPU);
    constexpr std::size_t jointStatusBytes =
        kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU);
    constexpr std::size_t reactionBytes =
        kEnvironmentCount * MR_NUMANX_COUPLED_HUMAN_MAX_DOFS *
        sizeof(float);

    ProbeBuffers result{};
    result.sourceQ = makeBuffer(device, qBytes, shared, @"source q");
    result.sourceV = makeBuffer(device, vectorBytes, shared, @"source v");
    result.factorSeed = makeBuffer(
        device, factorBytes, shared, @"exact factor seed"
    );
    result.sourceEffectiveTangentFactor = makeBuffer(
        device, factorBytes, privateStorage, @"source effective-tangent factor"
    );
    result.tangentProjector = makeBuffer(
        device, factorBytes, shared, @"noncommuting tangent projector"
    );
    result.bodySeed = makeBuffer(
        device, bodyBytes, shared, @"exact body seed"
    );
    result.candidateQ = makeBuffer(
        device, qBytes, shared, @"candidate q"
    );
    result.candidateBodies = makeBuffer(
        device, bodyBytes, shared, @"candidate bodies"
    );
    result.massInput = makeBuffer(
        device, vectorBytes, shared, @"mass input"
    );
    result.massOutput = makeBuffer(
        device, vectorBytes, shared, @"mass output"
    );
    result.inverseInput = makeBuffer(
        device, vectorBytes, shared, @"inverse input"
    );
    result.inverseOutput = makeBuffer(
        device, vectorBytes, shared, @"inverse output"
    );
    result.inverseStatuses = makeBuffer(
        device, inverseStatusBytes, shared, @"inverse statuses"
    );
    result.deltaV = makeBuffer(
        device, vectorBytes, shared, @"Matter delta v"
    );
    result.generalizedImpulse = makeBuffer(
        device, vectorBytes, shared, @"staged generalized impulse"
    );
    result.publishedMatterOutcomes = makeBuffer(
        device, outcomeBytes, shared, @"published Matter outcomes"
    );
    result.matterOutcomes = makeBuffer(
        device, outcomeBytes, shared, @"Matter outcomes"
    );
    result.resolvedMatterOutcomes = makeBuffer(
        device, outcomeBytes, shared, @"resolved Matter outcomes"
    );
    result.standStatuses = makeBuffer(
        device, standStatusBytes, shared, @"Human stand statuses"
    );
    result.callbackMarker = makeBuffer(
        device, sizeof(std::uint32_t), shared, @"exact callback marker"
    );
    result.pendingStatuses = makeBuffer(
        device, jointStatusBytes, shared, @"pending status snapshot"
    );
    result.stagedReaction = makeBuffer(
        device, reactionBytes, shared, @"pre-resolve reaction snapshot"
    );
    result.finalStatuses = makeBuffer(
        device, jointStatusBytes, shared, @"final status snapshot"
    );
    result.finalReaction = makeBuffer(
        device, reactionBytes, shared, @"final reaction snapshot"
    );
    return result;
}

void initializeImmutableInputs(ProbeBuffers& buffers) {
    auto* sourceQ = static_cast<float*>(buffers.sourceQ.contents);
    auto* sourceV = static_cast<float*>(buffers.sourceV.contents);
    auto* factor = static_cast<float*>(buffers.factorSeed.contents);
    auto* projector = static_cast<float*>(
        buffers.tangentProjector.contents
    );
    auto* bodies = static_cast<MRBodyStateGPU*>(buffers.bodySeed.contents);
    auto* massInput = static_cast<float*>(buffers.massInput.contents);
    auto* inverseInput = static_cast<float*>(buffers.inverseInput.contents);
    auto* deltaV = static_cast<float*>(buffers.deltaV.contents);
    auto* publishedOutcomes =
        static_cast<MRNumanXCoupledMatterOutcomeGPU*>(
            buffers.publishedMatterOutcomes.contents);
    auto* outcomes = static_cast<MRNumanXCoupledMatterOutcomeGPU*>(
        buffers.matterOutcomes.contents);
    auto* resolvedOutcomes =
        static_cast<MRNumanXCoupledMatterOutcomeGPU*>(
            buffers.resolvedMatterOutcomes.contents);
    auto* standStatuses = static_cast<MRNumiHumanStandStatusGPU*>(
        buffers.standStatuses.contents
    );

    std::memset(factor, 0, kFactorElementCount * sizeof(float));
    std::memset(projector, 0, kFactorElementCount * sizeof(float));
    std::memset(bodies, 0, kBodyElementCount * sizeof(MRBodyStateGPU));
    std::memset(
        standStatuses,
        0,
        kEnvironmentCount * sizeof(MRNumiHumanStandStatusGPU)
    );
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        const std::size_t qBase =
            static_cast<std::size_t>(environment) * kQCount;
        const std::size_t vectorBase =
            static_cast<std::size_t>(environment) * kDofCount;
        const std::size_t factorBase =
            static_cast<std::size_t>(environment) * kFactorStride;
        for (std::uint32_t coordinate = 0u;
             coordinate < kQCount;
             ++coordinate) {
            sourceQ[qBase + coordinate] =
                static_cast<float>((environment + 1u) * 1000u + coordinate) /
                1024.0f;
        }
        for (std::uint32_t dof = 0u; dof < kDofCount; ++dof) {
            const std::size_t vectorIndex = vectorBase + dof;
            sourceV[vectorIndex] =
                -static_cast<float>((environment + 1u) * (dof + 1u)) /
                512.0f;
            massInput[vectorIndex] =
                static_cast<float>((environment + 1u) * (dof + 1u)) /
                256.0f;
            inverseInput[vectorIndex] =
                static_cast<float>((environment + 2u) * (dof + 1u)) /
                128.0f;
            deltaV[vectorIndex] =
                static_cast<float>((environment + 1u) *
                                   ((dof % 7u) + 1u)) /
                128.0f;
            factor[factorBase +
                   static_cast<std::size_t>(dof) * kDofCount + dof] =
                dof == 1u ? 3.0f : 2.0f;
            projector[factorBase +
                      static_cast<std::size_t>(dof) * kDofCount + dof] =
                1.0f;
        }
        // Symmetric idempotent projector onto span((1,1)) for the first two
        // coordinates. It deliberately does not commute with the source
        // effective tangent A0=diag(4,9).
        projector[factorBase] = 0.5f;
        projector[factorBase + 1u] = 0.5f;
        projector[factorBase + kDofCount] = 0.5f;
        projector[factorBase + kDofCount + 1u] = 0.5f;

        MRBodyStateGPU& body = bodies[environment];
        body.position = {
            static_cast<float>(environment),
            1.0f,
            -2.0f,
            0.0f,
        };
        body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        body.linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
        body.angularVelocity = {0.0f, 0.0f, 0.0f, 0.0f};
        body.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
        body.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
        body.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
        body.flagsAndIndices[0] = 0u;
        body.flagsAndIndices[1] = 0u;
        body.flagsAndIndices[2] = 0u;
        body.flagsAndIndices[3] = 0u;

        publishedOutcomes[environment] = {};
        publishedOutcomes[environment].code = environment == 2u
            ? kMatterFailure
            : kMatterSuccess;
        publishedOutcomes[environment].environment = environment;
        publishedOutcomes[environment].completedMicrosteps = 0u;
        outcomes[environment] = publishedOutcomes[environment];
        resolvedOutcomes[environment] = publishedOutcomes[environment];
        resolvedOutcomes[environment].completedMicrosteps =
            kExpectedMatterMicrosteps;

        standStatuses[environment].code = environment == 1u
            ? MR_NUMI_HUMAN_STAND_CONTACT_FAILED
            : MR_NUMI_HUMAN_STAND_SUCCESS;
        standStatuses[environment].environment = environment;
        standStatuses[environment].completedSteps = 1u;
        standStatuses[environment].failingIndex = environment == 1u
            ? 12u
            : MR_INVALID_INDEX;
    }
}

void poisonWritableBuffers(ProbeBuffers& buffers) {
    constexpr int poison = 0xcd;
    std::memset(
        buffers.candidateQ.contents,
        poison,
        kQElementCount * sizeof(float)
    );
    std::memset(
        buffers.candidateBodies.contents,
        poison,
        kBodyElementCount * sizeof(MRBodyStateGPU)
    );
    std::memset(
        buffers.massOutput.contents,
        poison,
        kVectorElementCount * sizeof(float)
    );
    std::memset(
        buffers.inverseOutput.contents,
        poison,
        kVectorElementCount * sizeof(float)
    );
    std::memset(
        buffers.inverseStatuses.contents,
        poison,
        kInverseStatusElementCount * sizeof(MRInverseMassStatusGPU)
    );
    std::memset(
        buffers.generalizedImpulse.contents,
        poison,
        kVectorElementCount * sizeof(float)
    );
    std::memset(buffers.callbackMarker.contents, 0, sizeof(std::uint32_t));
    std::memset(
        buffers.pendingStatuses.contents,
        poison,
        kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU)
    );
    std::memset(
        buffers.stagedReaction.contents,
        poison,
        kVectorElementCount * sizeof(float)
    );
    std::memset(
        buffers.finalStatuses.contents,
        poison,
        kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU)
    );
    std::memset(
        buffers.finalReaction.contents,
        poison,
        kVectorElementCount * sizeof(float)
    );
}

[[nodiscard]] bool encodeExactCandidate(
    void* opaque,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    @autoreleasepool {
        auto* context = static_cast<ExactCallbackContext*>(opaque);
        if (context == nullptr || context->factorSeed == nil ||
            context->bodySeed == nil || context->marker == nil ||
            pass.phase != MetalNumanXCoupledHumanPhase::preDynamics ||
            query.operation !=
                MetalNumanXCoupledHumanOperation::candidateKinematics ||
            pass.commandBuffer == nullptr || pass.sourceQ == nullptr ||
            pass.sourceEffectiveTangentFactor == nullptr ||
            query.candidateQ == nullptr ||
            query.candidateBodies == nullptr ||
            pass.sourceQElementCount != kQElementCount ||
            pass.sourceEffectiveTangentFactorElementCount !=
                kFactorElementCount ||
            query.candidateQStride != kQCount ||
            query.candidateBodyStride != kBodyCount) {
            return false;
        }

        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        __unsafe_unretained id<MTLBuffer> sourceQ =
            (__bridge id<MTLBuffer>)pass.sourceQ;
        __unsafe_unretained id<MTLBuffer> effectiveTangentFactor =
            (__bridge id<MTLBuffer>)pass.sourceEffectiveTangentFactor;
        __unsafe_unretained id<MTLBuffer> candidateQ =
            (__bridge id<MTLBuffer>)query.candidateQ;
        __unsafe_unretained id<MTLBuffer> candidateBodies =
            (__bridge id<MTLBuffer>)query.candidateBodies;
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit == nil) return false;
        blit.label = @"probe exact Human candidate";
        [blit fillBuffer:context->marker
                   range:NSMakeRange(0u, sizeof(std::uint32_t))
                   value:kMarkerByte];
        [blit copyFromBuffer:sourceQ
                sourceOffset:0u
                    toBuffer:candidateQ
           destinationOffset:0u
                        size:kQElementCount * sizeof(float)];
        [blit copyFromBuffer:context->bodySeed
                sourceOffset:0u
                    toBuffer:candidateBodies
           destinationOffset:0u
                        size:kBodyElementCount * sizeof(MRBodyStateGPU)];
        [blit copyFromBuffer:context->factorSeed
                sourceOffset:0u
                    toBuffer:effectiveTangentFactor
           destinationOffset:0u
                        size:kFactorElementCount * sizeof(float)];
        [blit endEncoding];

        ++context->invocationCount;
        context->callbackOrder = ++context->hostSequence;
        return true;
    }
}

[[nodiscard]] MetalNumanXCoupledHumanPass makePass(
    id<MTLCommandBuffer> commandBuffer,
    const ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanPhase phase,
    const std::uint32_t standFlags,
    const std::uint64_t slotGeneration,
    const bool useProjector = false
) {
    MetalNumanXCoupledHumanPass pass{};
    pass.accessFlags =
        metalrobo::MetalNumanXCoupledHumanReadSourceState |
        metalrobo::MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor |
        metalrobo::MetalNumanXCoupledHumanEncodeExactCandidate |
        metalrobo::MetalNumanXCoupledHumanWriteReaction |
        metalrobo::MetalNumanXCoupledHumanWriteJointStatus;
    if (phase == MetalNumanXCoupledHumanPhase::postDynamics) {
        pass.accessFlags |=
            metalrobo::MetalNumanXCoupledHumanReadStandStatus;
    }
    pass.capabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
    if (useProjector) {
        pass.accessFlags |=
            metalrobo::MetalNumanXCoupledHumanReadTangentProjector;
        pass.capabilities |=
            MR_NUMANX_COUPLED_HUMAN_CAP_TANGENT_PROJECTOR;
    }
    pass.commandBuffer = (__bridge void*)commandBuffer;
    pass.sourceQ = rawBuffer(buffers.sourceQ);
    pass.sourceV = rawBuffer(buffers.sourceV);
    pass.sourceEffectiveTangentFactor =
        rawBuffer(buffers.sourceEffectiveTangentFactor);
    pass.tangentProjector = useProjector
        ? rawBuffer(buffers.tangentProjector)
        : nullptr;
    pass.standStatuses = rawBuffer(buffers.standStatuses);
    pass.exactKinematicsContext = &callback;
    pass.encodeExactKinematics = &encodeExactCandidate;
    pass.sourceQGPUAddress = gpuAddress(buffers.sourceQ);
    pass.sourceVGPUAddress = gpuAddress(buffers.sourceV);
    pass.sourceEffectiveTangentFactorGPUAddress =
        gpuAddress(buffers.sourceEffectiveTangentFactor);
    pass.tangentProjectorGPUAddress = useProjector
        ? gpuAddress(buffers.tangentProjector)
        : 0u;
    pass.standStatusesGPUAddress = gpuAddress(buffers.standStatuses);
    pass.sourceQElementCount = kQElementCount;
    pass.sourceVElementCount = kVectorElementCount;
    pass.sourceEffectiveTangentFactorElementCount = kFactorElementCount;
    pass.tangentProjectorElementCount = useProjector
        ? kFactorElementCount
        : 0u;
    pass.standStatusElementCount = kEnvironmentCount;
    pass.phase = phase;
    pass.environmentCount = kEnvironmentCount;
    pass.qCoordinateCount = kQCount;
    pass.dofCount = kDofCount;
    pass.bodyCount = kBodyCount;
    pass.qStride = kQCount;
    pass.vStride = kDofCount;
    pass.factorStride = kFactorStride;
    pass.projectorStride = useProjector ? kFactorStride : 0u;
    pass.standStatusStride = 1u;
    pass.articulationIndex = kArticulationIndex;
    pass.standFlags = standFlags;
    pass.stepIndex = 0u;
    pass.stepCount = 1u;
    pass.transactionSlot = 0u;
    pass.timestepSeconds = kTimestepSeconds;
    pass.programFingerprint = program.fingerprint;
    pass.transactionFingerprint = kTransactionFingerprint;
    pass.linearizationEpoch = kLinearizationEpoch;
    pass.slotGeneration = slotGeneration;
    return pass;
}

[[nodiscard]] MetalNumanXCoupledHumanQuery commonQuery(
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanOperation operation,
    const std::uint64_t slotGeneration
) {
    MetalNumanXCoupledHumanQuery query{};
    query.operation = operation;
    query.programFingerprint = program.fingerprint;
    query.transactionFingerprint = kTransactionFingerprint;
    query.linearizationEpoch = kLinearizationEpoch;
    query.transactionSlot = 0u;
    query.slotGeneration = slotGeneration;
    return query;
}

[[nodiscard]] MetalNumanXCoupledHumanQuery candidateQuery(
    const MetalNumanXCoupledHumanProgram& program,
    const ProbeBuffers& buffers,
    const std::uint64_t slotGeneration
) {
    MetalNumanXCoupledHumanQuery query = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::candidateKinematics,
        slotGeneration
    );
    query.input = rawBuffer(buffers.deltaV);
    query.candidateQ = rawBuffer(buffers.candidateQ);
    query.candidateBodies = rawBuffer(buffers.candidateBodies);
    query.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteCandidateQ |
        metalrobo::MetalNumanXCoupledHumanQueryWriteCandidateBodies;
    query.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
    query.inputGPUAddress = gpuAddress(buffers.deltaV);
    query.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    query.candidateBodiesGPUAddress = gpuAddress(buffers.candidateBodies);
    query.generalizedVectorStride = kDofCount;
    query.candidateQStride = kQCount;
    query.candidateBodyStride = kBodyCount;
    return query;
}

[[nodiscard]] MetalNumanXCoupledHumanQuery massQuery(
    const MetalNumanXCoupledHumanProgram& program,
    const ProbeBuffers& buffers,
    const std::uint64_t slotGeneration
) {
    MetalNumanXCoupledHumanQuery query = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::massAction,
        slotGeneration
    );
    query.input = rawBuffer(buffers.massInput);
    query.output = rawBuffer(buffers.massOutput);
    query.candidateQ = rawBuffer(buffers.candidateQ);
    query.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput;
    query.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION;
    query.inputGPUAddress = gpuAddress(buffers.massInput);
    query.outputGPUAddress = gpuAddress(buffers.massOutput);
    query.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    query.generalizedVectorStride = kDofCount;
    query.candidateQStride = kQCount;
    return query;
}

[[nodiscard]] MetalNumanXCoupledHumanQuery inverseQuery(
    const MetalNumanXCoupledHumanProgram& program,
    const ProbeBuffers& buffers,
    const std::uint64_t slotGeneration
) {
    MetalNumanXCoupledHumanQuery query = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::inverseMassPreconditioner,
        slotGeneration
    );
    query.input = rawBuffer(buffers.inverseInput);
    query.output = rawBuffer(buffers.inverseOutput);
    query.statuses = rawBuffer(buffers.inverseStatuses);
    query.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteInverseStatus;
    query.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER;
    query.inputGPUAddress = gpuAddress(buffers.inverseInput);
    query.outputGPUAddress = gpuAddress(buffers.inverseOutput);
    query.statusesGPUAddress = gpuAddress(buffers.inverseStatuses);
    query.generalizedVectorStride = kDofCount;
    query.statusStride = kEnvironmentCount;
    return query;
}

[[nodiscard]] MetalNumanXCoupledHumanQuery publishQuery(
    const MetalNumanXCoupledHumanProgram& program,
    const ProbeBuffers& buffers,
    const std::uint64_t slotGeneration
) {
    MetalNumanXCoupledHumanQuery query = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::publishCandidate,
        slotGeneration
    );
    query.input = rawBuffer(buffers.deltaV);
    query.output = rawBuffer(buffers.generalizedImpulse);
    query.candidateQ = rawBuffer(buffers.candidateQ);
    query.matterOutcomes = rawBuffer(buffers.matterOutcomes);
    query.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput |
        metalrobo::MetalNumanXCoupledHumanQueryReadMatterOutcome;
    query.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH;
    query.inputGPUAddress = gpuAddress(buffers.deltaV);
    query.outputGPUAddress = gpuAddress(buffers.generalizedImpulse);
    query.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    query.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcomes);
    query.generalizedVectorStride = kDofCount;
    query.candidateQStride = kQCount;
    query.matterOutcomeStride = 1u;
    query.matterSuccessCode = kMatterSuccess;
    query.expectedMatterCompletedMicrosteps = 0u;
    return query;
}

[[nodiscard]] MetalNumanXCoupledHumanResolveQuery resolveQuery(
    const MetalNumanXCoupledHumanProgram& program,
    const ProbeBuffers& buffers,
    const std::uint64_t slotGeneration
) {
    MetalNumanXCoupledHumanResolveQuery query{};
    query.matterOutcomes = rawBuffer(buffers.matterOutcomes);
    query.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcomes);
    query.matterOutcomeElementCount = kEnvironmentCount;
    query.matterOutcomeStride = 1u;
    query.matterSuccessCode = kMatterSuccess;
    query.expectedMatterCompletedMicrosteps = kExpectedMatterMicrosteps;
    query.transactionSlot = 0u;
    query.programFingerprint = program.fingerprint;
    query.transactionFingerprint = kTransactionFingerprint;
    query.linearizationEpoch = kLinearizationEpoch;
    query.slotGeneration = slotGeneration;
    return query;
}

[[nodiscard]] bool copyBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> source,
    id<MTLBuffer> destination,
    const std::size_t byteCount,
    NSString* label
) {
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit == nil) return false;
    blit.label = label;
    [blit copyFromBuffer:source
            sourceOffset:0u
                toBuffer:destination
       destinationOffset:0u
                    size:byteCount];
    [blit endEncoding];
    return true;
}

void appendBufferBytes(
    std::vector<std::uint8_t>& destination,
    id<MTLBuffer> buffer
) {
    const auto* begin = static_cast<const std::uint8_t*>(buffer.contents);
    destination.insert(destination.end(), begin, begin + buffer.length);
}

[[nodiscard]] bool closeEnough(const float actual, const float expected) {
    const float scale = std::fmax(1.0f, std::fabs(expected));
    return std::fabs(actual - expected) <= 2.0e-5f * scale;
}

[[nodiscard]] constexpr float massDiagonal(const std::uint32_t dof) {
    return dof == 1u ? 9.0f : 4.0f;
}

[[nodiscard]] bool validateOutputs(
    const ProbeBuffers& buffers,
    const MetalNumanXCoupledHumanArenaView& arena,
    std::string& error,
    const bool useProjector
) {
    if (arena.matterGeneralizedReaction ==
            rawBuffer(buffers.generalizedImpulse) ||
        arena.reactionGPUAddress == gpuAddress(buffers.generalizedImpulse)) {
        error = "reaction arena aliases the generalized-impulse output";
        return false;
    }

    if (std::memcmp(
            buffers.candidateQ.contents,
            buffers.sourceQ.contents,
            kQElementCount * sizeof(float)) != 0) {
        error = "exact callback did not materialize candidate q";
        return false;
    }
    if (std::memcmp(
            buffers.candidateBodies.contents,
            buffers.bodySeed.contents,
            kBodyElementCount * sizeof(MRBodyStateGPU)) != 0) {
        error = "exact callback did not materialize candidate bodies";
        return false;
    }
    if (*static_cast<const std::uint32_t*>(
            buffers.callbackMarker.contents) != 0xa5a5a5a5u) {
        error = "exact callback GPU marker was not ordered on the transaction";
        return false;
    }

    const auto* massInput = static_cast<const float*>(
        buffers.massInput.contents
    );
    const auto* massOutput = static_cast<const float*>(
        buffers.massOutput.contents
    );
    const auto* inverseInput = static_cast<const float*>(
        buffers.inverseInput.contents
    );
    const auto* inverseOutput = static_cast<const float*>(
        buffers.inverseOutput.contents
    );
    const auto* deltaV = static_cast<const float*>(buffers.deltaV.contents);
    const auto* impulse = static_cast<const float*>(
        buffers.generalizedImpulse.contents
    );
    const auto* stagedReaction = static_cast<const float*>(
        buffers.stagedReaction.contents
    );
    const auto* finalReaction = static_cast<const float*>(
        buffers.finalReaction.contents
    );

    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        const std::size_t base =
            static_cast<std::size_t>(environment) * kDofCount;
        for (std::uint32_t dof = 0u; dof < kDofCount; ++dof) {
            const std::size_t index = base + dof;
            const float diagonal = massDiagonal(dof);
            const float expectedMass = useProjector && dof < 2u
                ? 3.25f * (massInput[base] + massInput[base + 1u])
                : diagonal * massInput[index];
            if (!closeEnough(
                    massOutput[index], expectedMass)) {
                error = "160-DoF projected effective-tangent action mismatch";
                return false;
            }
            const float expectedInverse = useProjector && dof < 2u
                ? (13.0f / 144.0f) *
                    (inverseInput[base] + inverseInput[base + 1u])
                : inverseInput[index] / diagonal;
            if (!closeEnough(
                    inverseOutput[index], expectedInverse)) {
                error = "160-DoF effective-tangent preconditioner mismatch";
                return false;
            }

            const float projectedImpulse = useProjector && dof < 2u
                ? 3.25f * (deltaV[base] + deltaV[base + 1u])
                : diagonal * deltaV[index];
            const float expectedImpulse = environment == 2u
                ? 0.0f
                : projectedImpulse;
            const float expectedStagedReaction = environment == 2u
                ? 0.0f
                : expectedImpulse / kTimestepSeconds;
            const float expectedFinalReaction = environment == 0u
                ? expectedStagedReaction
                : 0.0f;
            if (!closeEnough(impulse[index], expectedImpulse) ||
                !closeEnough(
                    stagedReaction[index], expectedStagedReaction) ||
                !closeEnough(finalReaction[index], expectedFinalReaction)) {
                error = "staged/separate/rolled-back reaction mismatch";
                return false;
            }
        }
    }

    const auto* inverseStatuses =
        static_cast<const MRInverseMassStatusGPU*>(
            buffers.inverseStatuses.contents
        );
    const auto* pending =
        static_cast<const MRNumanXCoupledHumanStatusGPU*>(
            buffers.pendingStatuses.contents
        );
    const auto* finalStatuses =
        static_cast<const MRNumanXCoupledHumanStatusGPU*>(
            buffers.finalStatuses.contents
        );
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        const std::size_t inverseIndex =
            static_cast<std::size_t>(kArticulationIndex) *
                kEnvironmentCount + environment;
        const MRInverseMassStatusGPU& inverse =
            inverseStatuses[inverseIndex];
        if (inverse.code != MR_INVERSE_MASS_SUCCESS ||
            inverse.environment != environment ||
            inverse.articulationIndex != kArticulationIndex ||
            inverse.bodyCount != kBodyCount || inverse.nq != kQCount ||
            inverse.nv != kDofCount || inverse.rhsCount != 1u) {
            error = "inverse-mass device status is not successful/exact-sized";
            return false;
        }
        if (pending[environment].abiVersion !=
                MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
            pending[environment].decision !=
                MR_NUMANX_COUPLED_HUMAN_PENDING ||
            pending[environment].environment != environment ||
            pending[environment].stepIndex != 0u) {
            error = "begin did not expose a pending 32-byte joint status";
            return false;
        }
    }

    const auto* inverseStatusBytes =
        static_cast<const std::uint8_t*>(buffers.inverseStatuses.contents);
    const std::size_t untouchedBytes =
        static_cast<std::size_t>(kArticulationIndex) * kEnvironmentCount *
        sizeof(MRInverseMassStatusGPU);
    for (std::size_t index = 0u; index < untouchedBytes; ++index) {
        if (inverseStatusBytes[index] != 0xcdu) {
            error = "inverse solve corrupted a different articulation block";
            return false;
        }
    }

    if (finalStatuses[0].decision != MR_NUMANX_COUPLED_HUMAN_ACCEPT ||
        finalStatuses[0].humanCode != MR_NUMI_HUMAN_STAND_SUCCESS ||
        finalStatuses[0].matterCode != kMatterSuccess ||
        finalStatuses[0].humanCompletedSteps != 1u ||
        finalStatuses[0].matterCompletedMicrosteps !=
            kExpectedMatterMicrosteps) {
        error = "pending-to-accept resolution failed";
        return false;
    }
    if (finalStatuses[1].decision !=
            MR_NUMANX_COUPLED_HUMAN_REJECT_HUMAN ||
        finalStatuses[1].humanCode !=
            MR_NUMI_HUMAN_STAND_CONTACT_FAILED) {
        error = "Human rejection did not win joint resolution";
        return false;
    }
    if (finalStatuses[2].decision !=
            MR_NUMANX_COUPLED_HUMAN_REJECT_MATTER ||
        finalStatuses[2].matterCode != kMatterFailure ||
        finalStatuses[2].matterCompletedMicrosteps !=
            kExpectedMatterMicrosteps) {
        error = "Matter rejection did not resolve jointly";
        return false;
    }
    return true;
}

[[nodiscard]] bool runTransaction(
    id<MTLCommandQueue> queue,
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanArenaView& arena,
    ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    RunCapture& capture,
    std::string& error,
    const std::uint64_t slotGeneration,
    const bool useProjector = false
) {
    poisonWritableBuffers(buffers);
    callback.hostSequence = 0u;
    callback.callbackOrder = 0u;
    callback.encodeReturnOrder = 0u;
    const std::uint32_t invocationBefore = callback.invocationCount;

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        error = "failed to create command buffer";
        return false;
    }
    commandBuffer.label = @"NumanX coupled Human production-slice probe";

    id<MTLBlitCommandEncoder> poison = [commandBuffer blitCommandEncoder];
    if (poison == nil) {
        error = "failed to create factor poison encoder";
        return false;
    }
    [poison fillBuffer:buffers.sourceEffectiveTangentFactor
                 range:NSMakeRange(
                     0u, buffers.sourceEffectiveTangentFactor.length)
                 value:0xffu];
    [poison copyFromBuffer:buffers.publishedMatterOutcomes sourceOffset:0u
                  toBuffer:buffers.matterOutcomes destinationOffset:0u
                       size:kEnvironmentCount *
                           sizeof(MRNumanXCoupledMatterOutcomeGPU)];
    [poison endEncoding];

    MetalNumanXCoupledHumanPass pre = makePass(
        commandBuffer,
        buffers,
        callback,
        program,
        MetalNumanXCoupledHumanPhase::preDynamics,
        useProjector ? MR_NUMI_HUMAN_STAND_ENABLE_CONTACT : 0u,
        slotGeneration,
        useProjector
    );
    if (!program.begin(program.context, pre)) {
        error = "begin rejected a valid unconstrained 160-DoF pass";
        return false;
    }

    __unsafe_unretained id<MTLBuffer> arenaStatuses =
        (__bridge id<MTLBuffer>)arena.jointStatuses;
    __unsafe_unretained id<MTLBuffer> arenaReaction =
        (__bridge id<MTLBuffer>)arena.matterGeneralizedReaction;
    if (!copyBuffer(
            commandBuffer,
            arenaStatuses,
            buffers.pendingStatuses,
            kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU),
            @"snapshot pending joint status")) {
        error = "failed to snapshot pending status";
        return false;
    }

    MetalNumanXCoupledHumanQuery candidate = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::candidateKinematics,
        slotGeneration
    );
    candidate.input = rawBuffer(buffers.deltaV);
    candidate.candidateQ = rawBuffer(buffers.candidateQ);
    candidate.candidateBodies = rawBuffer(buffers.candidateBodies);
    candidate.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteCandidateQ |
        metalrobo::MetalNumanXCoupledHumanQueryWriteCandidateBodies;
    candidate.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
    candidate.inputGPUAddress = gpuAddress(buffers.deltaV);
    candidate.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    candidate.candidateBodiesGPUAddress =
        gpuAddress(buffers.candidateBodies);
    candidate.generalizedVectorStride = kDofCount;
    candidate.candidateQStride = kQCount;
    candidate.candidateBodyStride = kBodyCount;
    if (!program.encode(program.context, pre, candidate)) {
        error = "exact candidate encode was rejected";
        return false;
    }
    callback.encodeReturnOrder = ++callback.hostSequence;
    if (callback.invocationCount != invocationBefore + 1u ||
        callback.callbackOrder != 1u || callback.encodeReturnOrder != 2u) {
        error = "exact kinematics callback host order is not singular/ordered";
        return false;
    }

    MetalNumanXCoupledHumanQuery mass = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::massAction,
        slotGeneration
    );
    mass.input = rawBuffer(buffers.massInput);
    mass.output = rawBuffer(buffers.massOutput);
    mass.candidateQ = rawBuffer(buffers.candidateQ);
    mass.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput;
    mass.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION;
    mass.inputGPUAddress = gpuAddress(buffers.massInput);
    mass.outputGPUAddress = gpuAddress(buffers.massOutput);
    mass.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    mass.generalizedVectorStride = kDofCount;
    mass.candidateQStride = kQCount;
    if (!program.encode(program.context, pre, mass)) {
        error = "unconstrained 160-DoF mass action was rejected";
        return false;
    }

    MetalNumanXCoupledHumanQuery inverse = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::inverseMassPreconditioner,
        slotGeneration
    );
    inverse.input = rawBuffer(buffers.inverseInput);
    inverse.output = rawBuffer(buffers.inverseOutput);
    inverse.statuses = rawBuffer(buffers.inverseStatuses);
    inverse.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteInverseStatus;
    inverse.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER;
    inverse.inputGPUAddress = gpuAddress(buffers.inverseInput);
    inverse.outputGPUAddress = gpuAddress(buffers.inverseOutput);
    inverse.statusesGPUAddress = gpuAddress(buffers.inverseStatuses);
    inverse.generalizedVectorStride = kDofCount;
    inverse.statusStride = kEnvironmentCount;
    if (!program.encode(program.context, pre, inverse)) {
        error = "unconstrained 160-DoF inverse solve was rejected";
        return false;
    }

    MetalNumanXCoupledHumanQuery publish = commonQuery(
        program,
        MetalNumanXCoupledHumanOperation::publishCandidate,
        slotGeneration
    );
    publish.input = rawBuffer(buffers.deltaV);
    publish.output = rawBuffer(buffers.generalizedImpulse);
    publish.candidateQ = rawBuffer(buffers.candidateQ);
    publish.matterOutcomes = rawBuffer(buffers.matterOutcomes);
    publish.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput |
        metalrobo::MetalNumanXCoupledHumanQueryReadMatterOutcome;
    publish.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH;
    publish.inputGPUAddress = gpuAddress(buffers.deltaV);
    publish.outputGPUAddress = gpuAddress(buffers.generalizedImpulse);
    publish.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    publish.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcomes);
    publish.generalizedVectorStride = kDofCount;
    publish.candidateQStride = kQCount;
    publish.matterOutcomeStride = 1u;
    publish.matterSuccessCode = kMatterSuccess;
    publish.expectedMatterCompletedMicrosteps = 0u;
    if (!program.encode(program.context, pre, publish)) {
        error = "staged reaction publish was rejected";
        return false;
    }
    if (program.encode(program.context, pre, publish)) {
        error = "duplicate staged reaction publish was accepted";
        return false;
    }

    if (!copyBuffer(
            commandBuffer,
            arenaReaction,
            buffers.stagedReaction,
            kVectorElementCount * sizeof(float),
            @"snapshot staged Matter reaction")) {
        error = "failed to snapshot pre-resolve reaction";
        return false;
    }
    if (!copyBuffer(
            commandBuffer,
            buffers.resolvedMatterOutcomes,
            buffers.matterOutcomes,
            kEnvironmentCount * sizeof(MRNumanXCoupledMatterOutcomeGPU),
            @"advance Matter outcome to finalized microsteps")) {
        error = "failed to stage finalized Matter outcome";
        return false;
    }

    MetalNumanXCoupledHumanPass post = pre;
    post.phase = MetalNumanXCoupledHumanPhase::postDynamics;
    post.accessFlags |= metalrobo::MetalNumanXCoupledHumanReadStandStatus;
    MetalNumanXCoupledHumanResolveQuery resolve{};
    resolve.matterOutcomes = rawBuffer(buffers.matterOutcomes);
    resolve.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcomes);
    resolve.matterOutcomeElementCount = kEnvironmentCount;
    resolve.matterOutcomeStride = 1u;
    resolve.matterSuccessCode = kMatterSuccess;
    resolve.expectedMatterCompletedMicrosteps =
        kExpectedMatterMicrosteps;
    resolve.transactionSlot = 0u;
    resolve.programFingerprint = program.fingerprint;
    resolve.transactionFingerprint = kTransactionFingerprint;
    resolve.linearizationEpoch = kLinearizationEpoch;
    resolve.slotGeneration = slotGeneration;
    if (!program.resolve(program.context, post, resolve)) {
        error = "joint post-dynamics resolution was rejected";
        return false;
    }

    if (!copyBuffer(
            commandBuffer,
            arenaStatuses,
            buffers.finalStatuses,
            kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU),
            @"snapshot final joint status") ||
        !copyBuffer(
            commandBuffer,
            arenaReaction,
            buffers.finalReaction,
            kVectorElementCount * sizeof(float),
            @"snapshot final Matter reaction")) {
        error = "failed to snapshot final transaction state";
        return false;
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        const char* metalError =
            commandBuffer.error.localizedDescription.UTF8String;
        error = "Metal command buffer failed: " + std::string(
            metalError != nullptr ? metalError : "unknown"
        );
        return false;
    }

    // Every host readback and all validation below occur only after completion.
    if (!validateOutputs(buffers, arena, error, useProjector)) return false;

    capture.bytes.clear();
    appendBufferBytes(capture.bytes, buffers.candidateQ);
    appendBufferBytes(capture.bytes, buffers.candidateBodies);
    appendBufferBytes(capture.bytes, buffers.massOutput);
    appendBufferBytes(capture.bytes, buffers.inverseOutput);
    appendBufferBytes(capture.bytes, buffers.inverseStatuses);
    appendBufferBytes(capture.bytes, buffers.generalizedImpulse);
    appendBufferBytes(capture.bytes, buffers.callbackMarker);
    appendBufferBytes(capture.bytes, buffers.pendingStatuses);
    appendBufferBytes(capture.bytes, buffers.stagedReaction);
    appendBufferBytes(capture.bytes, buffers.finalStatuses);
    appendBufferBytes(capture.bytes, buffers.finalReaction);
    return true;
}

[[nodiscard]] bool snapshotPendingZero(
    id<MTLCommandBuffer> commandBuffer,
    const MetalNumanXCoupledHumanArenaView& arena,
    ProbeBuffers& buffers,
    std::string& error,
    NSString* label
) {
    __unsafe_unretained id<MTLBuffer> arenaStatuses =
        (__bridge id<MTLBuffer>)arena.jointStatuses;
    __unsafe_unretained id<MTLBuffer> arenaReaction =
        (__bridge id<MTLBuffer>)arena.matterGeneralizedReaction;
    if (!copyBuffer(
            commandBuffer,
            arenaStatuses,
            buffers.finalStatuses,
            kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU),
            label) ||
        !copyBuffer(
            commandBuffer,
            arenaReaction,
            buffers.finalReaction,
            kVectorElementCount * sizeof(float),
            label)) {
        error = "failed to snapshot fail-closed transaction state";
        return false;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        error = "fail-closed transaction command buffer failed";
        return false;
    }
    const auto* statuses =
        static_cast<const MRNumanXCoupledHumanStatusGPU*>(
            buffers.finalStatuses.contents
        );
    const auto* reactions =
        static_cast<const float*>(buffers.finalReaction.contents);
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        if (statuses[environment].abiVersion !=
                MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
            statuses[environment].decision !=
                MR_NUMANX_COUPLED_HUMAN_PENDING ||
            statuses[environment].environment != environment ||
            statuses[environment].stepIndex != 0u) {
            error = "invalid host operation modified pending joint status";
            return false;
        }
    }
    for (std::size_t index = 0u; index < kVectorElementCount; ++index) {
        if (reactions[index] != 0.0f) {
            error = "invalid host operation modified staged reaction";
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool staleGenerationPreservesArena(
    id<MTLCommandQueue> queue,
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanArenaView& arena,
    ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    const std::uint64_t staleGeneration,
    std::string& error
) {
    const std::vector<std::uint8_t> expectedStatuses(
        static_cast<const std::uint8_t*>(buffers.finalStatuses.contents),
        static_cast<const std::uint8_t*>(buffers.finalStatuses.contents) +
            buffers.finalStatuses.length
    );
    const std::vector<std::uint8_t> expectedReaction(
        static_cast<const std::uint8_t*>(buffers.finalReaction.contents),
        static_cast<const std::uint8_t*>(buffers.finalReaction.contents) +
            buffers.finalReaction.length
    );
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        error = "failed to create stale-generation command buffer";
        return false;
    }
    const MetalNumanXCoupledHumanPass pass = makePass(
        commandBuffer,
        buffers,
        callback,
        program,
        MetalNumanXCoupledHumanPhase::preDynamics,
        0u,
        staleGeneration
    );
    if (program.begin(program.context, pass)) {
        error = "slot reuse without a newer generation was accepted";
        return false;
    }
    __unsafe_unretained id<MTLBuffer> arenaStatuses =
        (__bridge id<MTLBuffer>)arena.jointStatuses;
    __unsafe_unretained id<MTLBuffer> arenaReaction =
        (__bridge id<MTLBuffer>)arena.matterGeneralizedReaction;
    if (!copyBuffer(
            commandBuffer,
            arenaStatuses,
            buffers.finalStatuses,
            expectedStatuses.size(),
            @"snapshot stale-generation status") ||
        !copyBuffer(
            commandBuffer,
            arenaReaction,
            buffers.finalReaction,
            expectedReaction.size(),
            @"snapshot stale-generation reaction")) {
        error = "failed to snapshot stale-generation state";
        return false;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        error = "stale-generation snapshot command buffer failed";
        return false;
    }
    if (std::memcmp(
            expectedStatuses.data(),
            buffers.finalStatuses.contents,
            expectedStatuses.size()) != 0 ||
        std::memcmp(
            expectedReaction.data(),
            buffers.finalReaction.contents,
            expectedReaction.size()) != 0) {
        error = "stale-generation begin modified prior public arena state";
        return false;
    }
    return true;
}

[[nodiscard]] bool invalidOrderingFailsClosed(
    id<MTLCommandQueue> queue,
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanArenaView& arena,
    ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    std::string& error
) {
    const auto beginPass = [&] (
        const std::uint64_t generation,
        id<MTLCommandBuffer> __strong& commandBuffer,
        MetalNumanXCoupledHumanPass& pass
    ) -> bool {
        commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) return false;
        pass = makePass(
            commandBuffer,
            buffers,
            callback,
            program,
            MetalNumanXCoupledHumanPhase::preDynamics,
            0u,
            generation
        );
        return program.begin(program.context, pass);
    };

    id<MTLCommandBuffer> commandBuffer = nil;
    MetalNumanXCoupledHumanPass pass{};
    if (!beginPass(4u, commandBuffer, pass) ||
        program.begin(program.context, pass)) {
        error = "duplicate begin was not rejected";
        return false;
    }
    if (!snapshotPendingZero(
            commandBuffer, arena, buffers, error,
            @"duplicate-begin snapshot")) return false;

    if (!beginPass(5u, commandBuffer, pass) ||
        program.encode(
            program.context, pass, publishQuery(program, buffers, 5u))) {
        error = "publish-before-candidate was not rejected";
        return false;
    }
    if (!snapshotPendingZero(
            commandBuffer, arena, buffers, error,
            @"publish-before-candidate snapshot")) return false;

    if (!beginPass(6u, commandBuffer, pass) ||
        program.encode(
            program.context, pass, massQuery(program, buffers, 6u)) ||
        program.encode(
            program.context, pass, inverseQuery(program, buffers, 6u))) {
        error = "mass/inverse-before-candidate was not rejected";
        return false;
    }
    if (!snapshotPendingZero(
            commandBuffer, arena, buffers, error,
            @"operator-before-candidate snapshot")) return false;

    if (!beginPass(7u, commandBuffer, pass)) {
        error = "mixed-identity test begin was rejected";
        return false;
    }
    MetalNumanXCoupledHumanPass mixedPass = pass;
    mixedPass.transactionFingerprint ^= 0x101u;
    mixedPass.linearizationEpoch += 1u;
    MetalNumanXCoupledHumanQuery mixedQuery =
        candidateQuery(program, buffers, 7u);
    mixedQuery.transactionFingerprint = mixedPass.transactionFingerprint;
    mixedQuery.linearizationEpoch = mixedPass.linearizationEpoch;
    const std::uint32_t invocationsBefore = callback.invocationCount;
    if (program.encode(program.context, mixedPass, mixedQuery) ||
        callback.invocationCount != invocationsBefore) {
        error = "mixed transaction/epoch reached exact candidate encoding";
        return false;
    }
    if (!snapshotPendingZero(
            commandBuffer, arena, buffers, error,
            @"mixed-identity snapshot")) return false;

    if (!beginPass(8u, commandBuffer, pass)) {
        error = "resolve-before-publish test begin was rejected";
        return false;
    }
    MetalNumanXCoupledHumanPass post = pass;
    post.phase = MetalNumanXCoupledHumanPhase::postDynamics;
    post.accessFlags |= metalrobo::MetalNumanXCoupledHumanReadStandStatus;
    if (program.resolve(
            program.context, post, resolveQuery(program, buffers, 8u))) {
        error = "resolve-before-publish was not rejected";
        return false;
    }
    if (!snapshotPendingZero(
        commandBuffer,
        arena,
        buffers,
        error,
        @"resolve-before-publish snapshot"
    )) return false;

    if (!beginPass(9u, commandBuffer, pass) ||
        !program.encode(
            program.context, pass, candidateQuery(program, buffers, 9u))) {
        error = "pre-progress count test candidate was rejected";
        return false;
    }
    auto wrongPreCount = publishQuery(program, buffers, 9u);
    wrongPreCount.expectedMatterCompletedMicrosteps = 1u;
    if (program.encode(program.context, pass, wrongPreCount)) {
        error = "publish admitted a nonzero pre-progress microstep count";
        return false;
    }
    if (!snapshotPendingZero(
            commandBuffer, arena, buffers, error,
            @"nonzero pre-progress count snapshot")) return false;

    if (!beginPass(10u, commandBuffer, pass) ||
        !program.encode(
            program.context, pass, candidateQuery(program, buffers, 10u))) {
        error = "post-zero count test candidate was rejected";
        return false;
    }
    __unsafe_unretained id<MTLCommandBuffer> phaseCommand =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBuffer> phaseOutcome = makeBuffer(
        phaseCommand.device,
        kEnvironmentCount * sizeof(MRNumanXCoupledMatterOutcomeGPU),
        MTLResourceStorageModeShared,
        @"phase-explicit zero-progress Matter outcome"
    );
    if (phaseOutcome == nil) {
        error = "phase-explicit outcome allocation failed";
        return false;
    }
    auto* phaseRecords =
        static_cast<MRNumanXCoupledMatterOutcomeGPU*>(phaseOutcome.contents);
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        phaseRecords[environment] = {};
        phaseRecords[environment].environment = environment;
        phaseRecords[environment].code = kMatterSuccess;
        phaseRecords[environment].completedMicrosteps = 0u;
    }
    auto zeroPublish = publishQuery(program, buffers, 10u);
    zeroPublish.matterOutcomes = rawBuffer(phaseOutcome);
    zeroPublish.matterOutcomeGPUAddress = gpuAddress(phaseOutcome);
    if (!program.encode(program.context, pass, zeroPublish)) {
        error = "publish rejected exact zero pre-progress microsteps";
        return false;
    }
    post = pass;
    post.phase = MetalNumanXCoupledHumanPhase::postDynamics;
    post.accessFlags |= metalrobo::MetalNumanXCoupledHumanReadStandStatus;
    auto wrongPostCount = resolveQuery(program, buffers, 10u);
    wrongPostCount.matterOutcomes = rawBuffer(phaseOutcome);
    wrongPostCount.matterOutcomeGPUAddress = gpuAddress(phaseOutcome);
    wrongPostCount.expectedMatterCompletedMicrosteps = 0u;
    if (program.resolve(program.context, post, wrongPostCount)) {
        error = "resolve admitted a zero finalized microstep count";
        return false;
    }
    __unsafe_unretained id<MTLBuffer> arenaStatuses =
        (__bridge id<MTLBuffer>)arena.jointStatuses;
    if (!copyBuffer(
            commandBuffer,
            arenaStatuses,
            buffers.finalStatuses,
            kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU),
            @"zero finalized-count status snapshot")) {
        error = "zero finalized-count snapshot failed";
        return false;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        error = "phase-explicit count command failed";
        return false;
    }
    const auto* finalStatuses =
        static_cast<const MRNumanXCoupledHumanStatusGPU*>(
            buffers.finalStatuses.contents);
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        if (finalStatuses[environment].decision !=
                MR_NUMANX_COUPLED_HUMAN_PENDING ||
            finalStatuses[environment].matterCompletedMicrosteps != 0u) {
            error = "zero pre-progress publish did not remain pending";
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool constrainedPassFailsClosed(
    id<MTLCommandQueue> queue,
    const MetalNumanXCoupledHumanProgram& program,
    const ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    const std::uint32_t standFlag,
    const bool supplyProjector
) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) return false;
    const MetalNumanXCoupledHumanPass pass = makePass(
        commandBuffer,
        buffers,
        callback,
        program,
        MetalNumanXCoupledHumanPhase::preDynamics,
        standFlag,
        1u,
        supplyProjector
    );
    const bool accepted = program.begin(program.context, pass);
    if (accepted) {
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return false;
    }
    return commandBuffer.status == MTLCommandBufferStatusNotEnqueued;
}

[[nodiscard]] bool gpuIntervalAliasesFailClosed(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanArenaView& arena,
    ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    std::string& error
) {
    constexpr std::uint64_t generation = 11u;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        error = "failed to create interval-alias command buffer";
        return false;
    }
    const MetalNumanXCoupledHumanPass pass = makePass(
        commandBuffer,
        buffers,
        callback,
        program,
        MetalNumanXCoupledHumanPhase::preDynamics,
        0u,
        generation
    );
    if (!program.begin(program.context, pass) ||
        !program.encode(
            program.context,
            pass,
            candidateQuery(program, buffers, generation))) {
        error = "interval-alias transaction failed before query admission";
        return false;
    }

    const auto aliasesFactor = [&] (
        MetalNumanXCoupledHumanQuery query
    ) noexcept -> bool {
        query.output = pass.sourceEffectiveTangentFactor;
        query.outputGPUAddress =
            pass.sourceEffectiveTangentFactorGPUAddress;
        return program.encode(program.context, pass, query);
    };
    if (aliasesFactor(massQuery(program, buffers, generation)) ||
        aliasesFactor(inverseQuery(program, buffers, generation)) ||
        aliasesFactor(publishQuery(program, buffers, generation))) {
        error = "same-object query output/source-factor alias was admitted";
        return false;
    }

    constexpr NSUInteger vectorBytes =
        kVectorElementCount * sizeof(float);
    constexpr MTLResourceOptions options =
        MTLResourceStorageModePrivate |
        MTLResourceHazardTrackingModeTracked;
    const MTLSizeAndAlign sizeAndAlign =
        [device heapBufferSizeAndAlignWithLength:vectorBytes
                                         options:options];
    if (sizeAndAlign.size < vectorBytes || sizeAndAlign.align == 0u) {
        error = "could not size CoupledHuman alias-negative heap";
        return false;
    }
    MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    descriptor.size = sizeAndAlign.size;
    id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
    if (heap == nil) {
        error = "failed to allocate CoupledHuman alias-negative heap";
        return false;
    }
    id<MTLBuffer> first = [heap newBufferWithLength:vectorBytes
                                           options:options
                                            offset:0u];
    if (first == nil) {
        error = "failed to allocate first CoupledHuman heap alias";
        return false;
    }
    [first makeAliasable];
    id<MTLBuffer> second = [heap newBufferWithLength:vectorBytes
                                            options:options
                                             offset:0u];
    if (second == nil || first == second || first.gpuAddress == 0u ||
        first.gpuAddress != second.gpuAddress) {
        error = "Metal did not expose distinct overlapping heap buffers";
        return false;
    }
    const auto distinctHeapAliasAccepted = [&] (
        MetalNumanXCoupledHumanQuery query
    ) noexcept -> bool {
        query.input = rawBuffer(first);
        query.inputGPUAddress = gpuAddress(first);
        query.output = rawBuffer(second);
        query.outputGPUAddress = gpuAddress(second);
        return program.encode(program.context, pass, query);
    };
    if (distinctHeapAliasAccepted(
            massQuery(program, buffers, generation)) ||
        distinctHeapAliasAccepted(
            inverseQuery(program, buffers, generation)) ||
        distinctHeapAliasAccepted(
            publishQuery(program, buffers, generation))) {
        error = "distinct overlapping heap query buffers were admitted";
        return false;
    }
    return snapshotPendingZero(
        commandBuffer,
        arena,
        buffers,
        error,
        @"GPU interval-alias rejection snapshot"
    );
}

[[nodiscard]] bool missingPublishFailsClosed(
    id<MTLCommandQueue> queue,
    const MetalNumanXCoupledHumanProgram& program,
    const MetalNumanXCoupledHumanArenaView& arena,
    ProbeBuffers& buffers,
    ExactCallbackContext& callback,
    std::string& error,
    const std::uint64_t slotGeneration
) {
    std::memset(
        buffers.finalStatuses.contents,
        0xcd,
        kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU)
    );
    std::memset(
        buffers.finalReaction.contents,
        0xcd,
        kVectorElementCount * sizeof(float)
    );
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        error = "failed to create missing-publish command buffer";
        return false;
    }
    MetalNumanXCoupledHumanPass pre = makePass(
        commandBuffer,
        buffers,
        callback,
        program,
        MetalNumanXCoupledHumanPhase::preDynamics,
        0u,
        slotGeneration
    );
    if (!program.begin(program.context, pre)) {
        error = "missing-publish begin was rejected on the host";
        return false;
    }
    MetalNumanXCoupledHumanPass post = pre;
    post.phase = MetalNumanXCoupledHumanPhase::postDynamics;
    post.accessFlags |= metalrobo::MetalNumanXCoupledHumanReadStandStatus;
    MetalNumanXCoupledHumanResolveQuery resolve{};
    resolve.matterOutcomes = rawBuffer(buffers.matterOutcomes);
    resolve.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcomes);
    resolve.matterOutcomeElementCount = kEnvironmentCount;
    resolve.matterOutcomeStride = 1u;
    resolve.matterSuccessCode = kMatterSuccess;
    resolve.expectedMatterCompletedMicrosteps =
        kExpectedMatterMicrosteps;
    resolve.transactionSlot = 0u;
    resolve.programFingerprint = program.fingerprint;
    resolve.transactionFingerprint = kTransactionFingerprint;
    resolve.linearizationEpoch = kLinearizationEpoch;
    resolve.slotGeneration = slotGeneration;
    if (program.resolve(program.context, post, resolve)) {
        error = "missing-publish resolve was accepted on the host";
        return false;
    }

    __unsafe_unretained id<MTLBuffer> arenaStatuses =
        (__bridge id<MTLBuffer>)arena.jointStatuses;
    __unsafe_unretained id<MTLBuffer> arenaReaction =
        (__bridge id<MTLBuffer>)arena.matterGeneralizedReaction;
    if (!copyBuffer(
            commandBuffer,
            arenaStatuses,
            buffers.finalStatuses,
            kEnvironmentCount * sizeof(MRNumanXCoupledHumanStatusGPU),
            @"snapshot missing-publish status") ||
        !copyBuffer(
            commandBuffer,
            arenaReaction,
            buffers.finalReaction,
            kVectorElementCount * sizeof(float),
            @"snapshot missing-publish reaction")) {
        error = "failed to snapshot missing-publish rejection";
        return false;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        error = "missing-publish command buffer failed";
        return false;
    }

    const auto* statuses =
        static_cast<const MRNumanXCoupledHumanStatusGPU*>(
            buffers.finalStatuses.contents
        );
    const auto* reactions =
        static_cast<const float*>(buffers.finalReaction.contents);
    if (statuses[0].decision != MR_NUMANX_COUPLED_HUMAN_PENDING) {
        error = "rejected missing-publish resolve modified pending status";
        return false;
    }
    for (std::size_t index = 0u; index < kVectorElementCount; ++index) {
        if (reactions[index] != 0.0f) {
            error = "missing-publish rejection retained a reaction";
            return false;
        }
    }
    return true;
}

constexpr std::uint32_t kLogicalDofCount = 128u;
constexpr std::uint32_t kLogicalQCount = 129u;
constexpr std::uint32_t kMatterVectorCapacity =
    MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
constexpr std::uint32_t kMatterQCapacity =
    MR_NUMANX_COUPLED_HUMAN_MAX_Q;
constexpr std::uint8_t kLogicalTailMarker = 0x5au;
constexpr std::uint64_t kLogicalTransactionFingerprint =
    0x4f5a361d8c2b9071ull;
constexpr std::uint64_t kLogicalLinearizationEpoch =
    0x97ad30c5ull;

struct LogicalDofBuffers {
    id<MTLBuffer> sourceQ = nil;
    id<MTLBuffer> sourceV = nil;
    id<MTLBuffer> factorSeed = nil;
    id<MTLBuffer> sourceEffectiveTangentFactor = nil;
    id<MTLBuffer> bodySeed = nil;
    id<MTLBuffer> candidateQ = nil;
    id<MTLBuffer> candidateBodies = nil;
    id<MTLBuffer> massInput = nil;
    id<MTLBuffer> massOutput = nil;
    id<MTLBuffer> inverseInput = nil;
    id<MTLBuffer> inverseOutput = nil;
    id<MTLBuffer> inverseStatus = nil;
    id<MTLBuffer> deltaV = nil;
    id<MTLBuffer> generalizedImpulse = nil;
    id<MTLBuffer> matterOutcome = nil;
    id<MTLBuffer> resolvedMatterOutcome = nil;
    id<MTLBuffer> standStatus = nil;
    id<MTLBuffer> finalStatus = nil;
    id<MTLBuffer> finalReaction = nil;
};

struct LogicalDofCallbackContext {
    id<MTLBuffer> factorSeed = nil;
    id<MTLBuffer> bodySeed = nil;
    std::uint32_t invocationCount = 0u;
};

[[nodiscard]] bool logicalDofBuffersValid(
    const LogicalDofBuffers& buffers
) noexcept {
    return buffers.sourceQ != nil && buffers.sourceV != nil &&
        buffers.factorSeed != nil &&
        buffers.sourceEffectiveTangentFactor != nil &&
        buffers.bodySeed != nil && buffers.candidateQ != nil &&
        buffers.candidateBodies != nil && buffers.massInput != nil &&
        buffers.massOutput != nil && buffers.inverseInput != nil &&
        buffers.inverseOutput != nil && buffers.inverseStatus != nil &&
        buffers.deltaV != nil && buffers.generalizedImpulse != nil &&
        buffers.matterOutcome != nil &&
        buffers.resolvedMatterOutcome != nil &&
        buffers.standStatus != nil && buffers.finalStatus != nil &&
        buffers.finalReaction != nil;
}

[[nodiscard]] bool encodeLogicalDofCandidate(
    void* opaque,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    @autoreleasepool {
        auto* context = static_cast<LogicalDofCallbackContext*>(opaque);
        if (context == nullptr || context->factorSeed == nil ||
            context->bodySeed == nil ||
            pass.phase != MetalNumanXCoupledHumanPhase::preDynamics ||
            pass.environmentCount != 1u ||
            pass.qCoordinateCount != kLogicalQCount ||
            pass.dofCount != kLogicalDofCount ||
            pass.qStride != kLogicalQCount ||
            pass.vStride != kLogicalDofCount ||
            pass.factorStride !=
                kLogicalDofCount * kLogicalDofCount ||
            pass.sourceQElementCount != kLogicalQCount ||
            pass.sourceVElementCount != kLogicalDofCount ||
            pass.sourceEffectiveTangentFactorElementCount !=
                kLogicalDofCount * kLogicalDofCount ||
            query.operation !=
                MetalNumanXCoupledHumanOperation::candidateKinematics ||
            query.generalizedVectorStride != kMatterVectorCapacity ||
            query.candidateQStride != kMatterQCapacity ||
            query.candidateBodyStride != 1u ||
            query.pointCount != 0u || pass.commandBuffer == nullptr ||
            pass.sourceQ == nullptr ||
            pass.sourceEffectiveTangentFactor == nullptr ||
            query.candidateQ == nullptr ||
            query.candidateBodies == nullptr) {
            return false;
        }

        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        __unsafe_unretained id<MTLBuffer> sourceQ =
            (__bridge id<MTLBuffer>)pass.sourceQ;
        __unsafe_unretained id<MTLBuffer> factor =
            (__bridge id<MTLBuffer>)pass.sourceEffectiveTangentFactor;
        __unsafe_unretained id<MTLBuffer> candidateQ =
            (__bridge id<MTLBuffer>)query.candidateQ;
        __unsafe_unretained id<MTLBuffer> candidateBodies =
            (__bridge id<MTLBuffer>)query.candidateBodies;
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit == nil) return false;
        blit.label = @"128-DoF exact Human candidate";
        [blit copyFromBuffer:sourceQ
                sourceOffset:0u
                    toBuffer:candidateQ
           destinationOffset:0u
                        size:kLogicalQCount * sizeof(float)];
        [blit copyFromBuffer:context->bodySeed
                sourceOffset:0u
                    toBuffer:candidateBodies
           destinationOffset:0u
                        size:sizeof(MRBodyStateGPU)];
        [blit copyFromBuffer:context->factorSeed
                sourceOffset:0u
                    toBuffer:factor
           destinationOffset:0u
                        size:static_cast<NSUInteger>(kLogicalDofCount) *
                            kLogicalDofCount * sizeof(float)];
        [blit endEncoding];
        ++context->invocationCount;
        return true;
    }
}

[[nodiscard]] bool markerBytesPreserved(
    id<MTLBuffer> buffer,
    const std::size_t byteOffset,
    const std::size_t byteCount
) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
    for (std::size_t index = 0u; index < byteCount; ++index) {
        if (bytes[byteOffset + index] != kLogicalTailMarker) return false;
    }
    return true;
}

[[nodiscard]] bool runLogicalCapacityTransaction(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const char* metallibPath,
    std::string& error
) {
    constexpr MTLResourceOptions shared = MTLResourceStorageModeShared;
    constexpr MTLResourceOptions privateStorage =
        MTLResourceStorageModePrivate;
    constexpr std::size_t logicalQBytes =
        static_cast<std::size_t>(kLogicalQCount) * sizeof(float);
    constexpr std::size_t logicalVBytes =
        static_cast<std::size_t>(kLogicalDofCount) * sizeof(float);
    constexpr std::size_t logicalFactorBytes =
        static_cast<std::size_t>(kLogicalDofCount) * kLogicalDofCount *
        sizeof(float);
    constexpr std::size_t vectorCapacityBytes =
        static_cast<std::size_t>(kMatterVectorCapacity) * sizeof(float);
    constexpr std::size_t qCapacityBytes =
        static_cast<std::size_t>(kMatterQCapacity) * sizeof(float);

    MetalNumanXCoupledHumanConfig config{};
    config.metallibPath = metallibPath;
    config.environmentCapacity = 1u;
    config.pointCapacity = 0u;
    config.transactionSlotCount = 1u;
    MetalNumanXCoupledHumanContext context(std::move(config));
    const auto initialization = context.initialize();
    if (!initialization.succeeded()) {
        error = std::string("128-DoF context initialization failed: ") +
            initialization.message;
        return false;
    }
    const MetalNumanXCoupledHumanProgram program = context.program();
    MetalNumanXCoupledHumanArenaView arena{};
    if (!program.valid() ||
        !context.arenaView(0u, arena).succeeded() ||
        arena.environmentCapacity != 1u ||
        arena.reactionStride != kMatterVectorCapacity ||
        arena.reactionByteCount != vectorCapacityBytes) {
        error = "128-DoF context exposed a non-capacity transport arena";
        return false;
    }

    LogicalDofBuffers buffers{};
    buffers.sourceQ = makeBuffer(
        device, logicalQBytes, shared, @"128-DoF source q");
    buffers.sourceV = makeBuffer(
        device, logicalVBytes, shared, @"128-DoF source v");
    buffers.factorSeed = makeBuffer(
        device, logicalFactorBytes, shared, @"128-DoF factor seed");
    buffers.sourceEffectiveTangentFactor = makeBuffer(
        device,
        logicalFactorBytes,
        privateStorage,
        @"128-DoF effective-tangent factor"
    );
    buffers.bodySeed = makeBuffer(
        device, sizeof(MRBodyStateGPU), shared, @"128-DoF body seed");
    buffers.candidateQ = makeBuffer(
        device, qCapacityBytes, shared, @"161-capacity candidate q");
    buffers.candidateBodies = makeBuffer(
        device,
        sizeof(MRBodyStateGPU),
        shared,
        @"128-DoF candidate body"
    );
    buffers.massInput = makeBuffer(
        device, vectorCapacityBytes, shared, @"160-capacity mass input");
    buffers.massOutput = makeBuffer(
        device, vectorCapacityBytes, shared, @"160-capacity mass output");
    buffers.inverseInput = makeBuffer(
        device, vectorCapacityBytes, shared, @"160-capacity inverse input");
    buffers.inverseOutput = makeBuffer(
        device, vectorCapacityBytes, shared, @"160-capacity inverse output");
    buffers.inverseStatus = makeBuffer(
        device,
        sizeof(MRInverseMassStatusGPU),
        shared,
        @"128-DoF inverse status"
    );
    buffers.deltaV = makeBuffer(
        device, vectorCapacityBytes, shared, @"160-capacity delta v");
    buffers.generalizedImpulse = makeBuffer(
        device,
        vectorCapacityBytes,
        shared,
        @"160-capacity generalized impulse"
    );
    buffers.matterOutcome = makeBuffer(
        device,
        sizeof(MRNumanXCoupledMatterOutcomeGPU),
        shared,
        @"128-DoF Matter outcome"
    );
    buffers.resolvedMatterOutcome = makeBuffer(
        device,
        sizeof(MRNumanXCoupledMatterOutcomeGPU),
        shared,
        @"128-DoF resolved Matter outcome"
    );
    buffers.standStatus = makeBuffer(
        device,
        sizeof(MRNumiHumanStandStatusGPU),
        shared,
        @"128-DoF Human status"
    );
    buffers.finalStatus = makeBuffer(
        device,
        sizeof(MRNumanXCoupledHumanStatusGPU),
        shared,
        @"128-DoF final joint status"
    );
    buffers.finalReaction = makeBuffer(
        device,
        vectorCapacityBytes,
        shared,
        @"128-DoF final reaction"
    );
    if (!logicalDofBuffersValid(buffers)) {
        error = "128-DoF buffer allocation failed";
        return false;
    }

    auto* sourceQ = static_cast<float*>(buffers.sourceQ.contents);
    auto* sourceV = static_cast<float*>(buffers.sourceV.contents);
    auto* factorSeed = static_cast<float*>(buffers.factorSeed.contents);
    auto* body = static_cast<MRBodyStateGPU*>(buffers.bodySeed.contents);
    auto* massInput = static_cast<float*>(buffers.massInput.contents);
    auto* inverseInput = static_cast<float*>(buffers.inverseInput.contents);
    auto* deltaV = static_cast<float*>(buffers.deltaV.contents);
    std::memset(factorSeed, 0, logicalFactorBytes);
    std::memset(body, 0, sizeof(MRBodyStateGPU));
    std::memset(
        buffers.candidateQ.contents, kLogicalTailMarker, qCapacityBytes);
    std::memset(
        buffers.massOutput.contents,
        kLogicalTailMarker,
        vectorCapacityBytes
    );
    std::memset(
        buffers.inverseOutput.contents,
        kLogicalTailMarker,
        vectorCapacityBytes
    );
    std::memset(
        buffers.generalizedImpulse.contents,
        kLogicalTailMarker,
        vectorCapacityBytes
    );
    for (std::uint32_t coordinate = 0u;
         coordinate < kLogicalQCount;
         ++coordinate) {
        sourceQ[coordinate] =
            static_cast<float>(coordinate + 1u) / 256.0f;
    }
    for (std::uint32_t dof = 0u; dof < kLogicalDofCount; ++dof) {
        sourceV[dof] = -static_cast<float>(dof + 1u) / 512.0f;
        massInput[dof] = static_cast<float>(dof + 1u) / 256.0f;
        inverseInput[dof] = static_cast<float>(dof + 1u) / 128.0f;
        deltaV[dof] = static_cast<float>((dof % 5u) + 1u) / 128.0f;
        factorSeed[static_cast<std::size_t>(dof) * kLogicalDofCount + dof] =
            2.0f;
    }
    for (std::uint32_t dof = kLogicalDofCount;
         dof < kMatterVectorCapacity;
         ++dof) {
        massInput[dof] = 0.0f;
        inverseInput[dof] = 0.0f;
        deltaV[dof] = 0.0f;
    }
    body[0].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    body[0].linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
    body[0].inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
    body[0].inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
    body[0].inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};

    auto* matter = static_cast<MRNumanXCoupledMatterOutcomeGPU*>(
        buffers.matterOutcome.contents
    );
    auto* resolved = static_cast<MRNumanXCoupledMatterOutcomeGPU*>(
        buffers.resolvedMatterOutcome.contents
    );
    matter[0] = {};
    matter[0].code = kMatterSuccess;
    matter[0].environment = 0u;
    matter[0].completedMicrosteps = 0u;
    resolved[0] = matter[0];
    resolved[0].completedMicrosteps = kExpectedMatterMicrosteps;
    auto* stand = static_cast<MRNumiHumanStandStatusGPU*>(
        buffers.standStatus.contents
    );
    stand[0] = {};
    stand[0].code = MR_NUMI_HUMAN_STAND_SUCCESS;
    stand[0].environment = 0u;
    stand[0].completedSteps = 1u;
    stand[0].failingIndex = MR_INVALID_INDEX;

    LogicalDofCallbackContext callback{};
    callback.factorSeed = buffers.factorSeed;
    callback.bodySeed = buffers.bodySeed;

    const auto makeLogicalPass = [&] (
        id<MTLCommandBuffer> commandBuffer,
        const MetalNumanXCoupledHumanPhase phase
    ) {
        MetalNumanXCoupledHumanPass pass{};
        pass.accessFlags =
            metalrobo::MetalNumanXCoupledHumanReadSourceState |
            metalrobo::MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor |
            metalrobo::MetalNumanXCoupledHumanEncodeExactCandidate |
            metalrobo::MetalNumanXCoupledHumanWriteReaction |
            metalrobo::MetalNumanXCoupledHumanWriteJointStatus;
        if (phase == MetalNumanXCoupledHumanPhase::postDynamics) {
            pass.accessFlags |=
                metalrobo::MetalNumanXCoupledHumanReadStandStatus;
        }
        pass.capabilities =
            MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
        pass.commandBuffer = (__bridge void*)commandBuffer;
        pass.sourceQ = rawBuffer(buffers.sourceQ);
        pass.sourceV = rawBuffer(buffers.sourceV);
        pass.sourceEffectiveTangentFactor =
            rawBuffer(buffers.sourceEffectiveTangentFactor);
        pass.standStatuses = rawBuffer(buffers.standStatus);
        pass.exactKinematicsContext = &callback;
        pass.encodeExactKinematics = &encodeLogicalDofCandidate;
        pass.sourceQGPUAddress = gpuAddress(buffers.sourceQ);
        pass.sourceVGPUAddress = gpuAddress(buffers.sourceV);
        pass.sourceEffectiveTangentFactorGPUAddress =
            gpuAddress(buffers.sourceEffectiveTangentFactor);
        pass.standStatusesGPUAddress = gpuAddress(buffers.standStatus);
        pass.sourceQElementCount = kLogicalQCount;
        pass.sourceVElementCount = kLogicalDofCount;
        pass.sourceEffectiveTangentFactorElementCount =
            kLogicalDofCount * kLogicalDofCount;
        pass.standStatusElementCount = 1u;
        pass.phase = phase;
        pass.environmentCount = 1u;
        pass.qCoordinateCount = kLogicalQCount;
        pass.dofCount = kLogicalDofCount;
        pass.bodyCount = 1u;
        pass.qStride = kLogicalQCount;
        pass.vStride = kLogicalDofCount;
        pass.factorStride = kLogicalDofCount * kLogicalDofCount;
        pass.standStatusStride = 1u;
        pass.articulationIndex = 0u;
        pass.stepIndex = 0u;
        pass.stepCount = 1u;
        pass.transactionSlot = 0u;
        pass.timestepSeconds = kTimestepSeconds;
        pass.programFingerprint = program.fingerprint;
        pass.transactionFingerprint = kLogicalTransactionFingerprint;
        pass.linearizationEpoch = kLogicalLinearizationEpoch;
        pass.slotGeneration = 1u;
        return pass;
    };

    const auto rejectedBegin = [&] (
        MetalNumanXCoupledHumanPass pass
    ) -> bool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) return false;
        pass.commandBuffer = (__bridge void*)commandBuffer;
        return !program.begin(program.context, pass) &&
            commandBuffer.status == MTLCommandBufferStatusNotEnqueued;
    };
    MetalNumanXCoupledHumanPass malformed = makeLogicalPass(
        nil, MetalNumanXCoupledHumanPhase::preDynamics);
    malformed.qCoordinateCount = kLogicalDofCount;
    if (!rejectedBegin(malformed)) {
        error = "nq != nv + 1 was admitted";
        return false;
    }
    malformed = makeLogicalPass(
        nil, MetalNumanXCoupledHumanPhase::preDynamics);
    malformed.qStride = kLogicalQCount + 1u;
    malformed.sourceQElementCount = malformed.qStride;
    if (!rejectedBegin(malformed)) {
        error = "live q capacity was admitted as a logical stride";
        return false;
    }
    malformed = makeLogicalPass(
        nil, MetalNumanXCoupledHumanPhase::preDynamics);
    malformed.vStride = kLogicalDofCount + 1u;
    malformed.sourceVElementCount = malformed.vStride;
    if (!rejectedBegin(malformed)) {
        error = "live v capacity was admitted as a logical stride";
        return false;
    }
    malformed = makeLogicalPass(
        nil, MetalNumanXCoupledHumanPhase::preDynamics);
    ++malformed.factorStride;
    malformed.sourceEffectiveTangentFactorElementCount =
        malformed.factorStride;
    if (!rejectedBegin(malformed)) {
        error = "non-logical factor stride was admitted";
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        error = "128-DoF transaction command-buffer allocation failed";
        return false;
    }
    commandBuffer.label = @"NumanX 128 logical / 160 transport probe";
    __unsafe_unretained id<MTLBuffer> reaction =
        (__bridge id<MTLBuffer>)arena.matterGeneralizedReaction;
    id<MTLBlitCommandEncoder> initialize =
        [commandBuffer blitCommandEncoder];
    if (initialize == nil) {
        error = "128-DoF initialization encoder allocation failed";
        return false;
    }
    [initialize fillBuffer:buffers.sourceEffectiveTangentFactor
                     range:NSMakeRange(0u, logicalFactorBytes)
                     value:0xffu];
    [initialize fillBuffer:reaction
                     range:NSMakeRange(0u, vectorCapacityBytes)
                     value:kLogicalTailMarker];
    [initialize endEncoding];

    MetalNumanXCoupledHumanPass pre = makeLogicalPass(
        commandBuffer, MetalNumanXCoupledHumanPhase::preDynamics);
    if (!program.begin(program.context, pre)) {
        error = "valid 128-DoF begin was rejected";
        return false;
    }
    const auto commonLogicalQuery = [&] (
        const MetalNumanXCoupledHumanOperation operation
    ) {
        MetalNumanXCoupledHumanQuery query{};
        query.operation = operation;
        query.programFingerprint = program.fingerprint;
        query.transactionFingerprint = kLogicalTransactionFingerprint;
        query.linearizationEpoch = kLogicalLinearizationEpoch;
        query.transactionSlot = 0u;
        query.slotGeneration = 1u;
        return query;
    };
    const auto makeLogicalCandidateQuery = [&] () {
        MetalNumanXCoupledHumanQuery query = commonLogicalQuery(
            MetalNumanXCoupledHumanOperation::candidateKinematics);
        query.input = rawBuffer(buffers.deltaV);
        query.candidateQ = rawBuffer(buffers.candidateQ);
        query.candidateBodies = rawBuffer(buffers.candidateBodies);
        query.accessFlags =
            metalrobo::MetalNumanXCoupledHumanQueryReadInput |
            metalrobo::MetalNumanXCoupledHumanQueryWriteCandidateQ |
            metalrobo::MetalNumanXCoupledHumanQueryWriteCandidateBodies;
        query.requiredCapabilities =
            MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
        query.inputGPUAddress = gpuAddress(buffers.deltaV);
        query.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
        query.candidateBodiesGPUAddress =
            gpuAddress(buffers.candidateBodies);
        query.generalizedVectorStride = kMatterVectorCapacity;
        query.candidateQStride = kMatterQCapacity;
        query.candidateBodyStride = 1u;
        return query;
    };

    MetalNumanXCoupledHumanQuery candidate = makeLogicalCandidateQuery();
    candidate.generalizedVectorStride = kLogicalDofCount - 1u;
    if (program.encode(program.context, pre, candidate)) {
        error = "Matter vector capacity below logical nv was admitted";
        return false;
    }
    candidate = makeLogicalCandidateQuery();
    candidate.generalizedVectorStride = kMatterVectorCapacity + 1u;
    if (program.encode(program.context, pre, candidate)) {
        error = "Matter vector capacity above ABI maximum was admitted";
        return false;
    }
    candidate = makeLogicalCandidateQuery();
    candidate.candidateQStride = kLogicalQCount - 1u;
    if (program.encode(program.context, pre, candidate)) {
        error = "candidate q capacity below logical nq was admitted";
        return false;
    }
    candidate = makeLogicalCandidateQuery();
    candidate.candidateQStride = kMatterQCapacity + 1u;
    if (program.encode(program.context, pre, candidate)) {
        error = "candidate q capacity above ABI maximum was admitted";
        return false;
    }
    if (callback.invocationCount != 0u) {
        error = "invalid logical/capacity query reached owner callback";
        return false;
    }
    candidate = makeLogicalCandidateQuery();
    if (!program.encode(program.context, pre, candidate) ||
        callback.invocationCount != 1u) {
        error = "valid 128/129 logical candidate on 160/161 arenas failed";
        return false;
    }

    MetalNumanXCoupledHumanQuery mass = commonLogicalQuery(
        MetalNumanXCoupledHumanOperation::massAction);
    mass.input = rawBuffer(buffers.massInput);
    mass.output = rawBuffer(buffers.massOutput);
    mass.candidateQ = rawBuffer(buffers.candidateQ);
    mass.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput;
    mass.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION;
    mass.inputGPUAddress = gpuAddress(buffers.massInput);
    mass.outputGPUAddress = gpuAddress(buffers.massOutput);
    mass.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    mass.generalizedVectorStride = kMatterVectorCapacity;
    mass.candidateQStride = kMatterQCapacity;
    if (!program.encode(program.context, pre, mass)) {
        error = "128-DoF mass action on capacity arena was rejected";
        return false;
    }

    MetalNumanXCoupledHumanQuery inverse = commonLogicalQuery(
        MetalNumanXCoupledHumanOperation::inverseMassPreconditioner);
    inverse.input = rawBuffer(buffers.inverseInput);
    inverse.output = rawBuffer(buffers.inverseOutput);
    inverse.statuses = rawBuffer(buffers.inverseStatus);
    inverse.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteInverseStatus;
    inverse.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER;
    inverse.inputGPUAddress = gpuAddress(buffers.inverseInput);
    inverse.outputGPUAddress = gpuAddress(buffers.inverseOutput);
    inverse.statusesGPUAddress = gpuAddress(buffers.inverseStatus);
    inverse.generalizedVectorStride = kMatterVectorCapacity;
    inverse.statusStride = 1u;
    if (!program.encode(program.context, pre, inverse)) {
        error = "128-DoF inverse action on capacity arena was rejected";
        return false;
    }

    MetalNumanXCoupledHumanQuery publish = commonLogicalQuery(
        MetalNumanXCoupledHumanOperation::publishCandidate);
    publish.input = rawBuffer(buffers.deltaV);
    publish.output = rawBuffer(buffers.generalizedImpulse);
    publish.candidateQ = rawBuffer(buffers.candidateQ);
    publish.matterOutcomes = rawBuffer(buffers.matterOutcome);
    publish.accessFlags =
        metalrobo::MetalNumanXCoupledHumanQueryReadInput |
        metalrobo::MetalNumanXCoupledHumanQueryWriteOutput |
        metalrobo::MetalNumanXCoupledHumanQueryReadMatterOutcome;
    publish.requiredCapabilities =
        MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH;
    publish.inputGPUAddress = gpuAddress(buffers.deltaV);
    publish.outputGPUAddress = gpuAddress(buffers.generalizedImpulse);
    publish.candidateQGPUAddress = gpuAddress(buffers.candidateQ);
    publish.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcome);
    publish.generalizedVectorStride = kMatterVectorCapacity;
    publish.candidateQStride = kMatterQCapacity;
    publish.matterOutcomeStride = 1u;
    publish.matterSuccessCode = kMatterSuccess;
    publish.expectedMatterCompletedMicrosteps = 0u;
    if (!program.encode(program.context, pre, publish)) {
        error = "128-DoF staged reaction on capacity arena was rejected";
        return false;
    }
    if (!copyBuffer(
            commandBuffer,
            buffers.resolvedMatterOutcome,
            buffers.matterOutcome,
            sizeof(MRNumanXCoupledMatterOutcomeGPU),
            @"128-DoF advance Matter outcome")) {
        error = "128-DoF resolved-outcome copy failed";
        return false;
    }

    MetalNumanXCoupledHumanPass post = makeLogicalPass(
        commandBuffer, MetalNumanXCoupledHumanPhase::postDynamics);
    MetalNumanXCoupledHumanResolveQuery resolve{};
    resolve.matterOutcomes = rawBuffer(buffers.matterOutcome);
    resolve.matterOutcomeGPUAddress = gpuAddress(buffers.matterOutcome);
    resolve.matterOutcomeElementCount = 1u;
    resolve.matterOutcomeStride = 1u;
    resolve.matterSuccessCode = kMatterSuccess;
    resolve.expectedMatterCompletedMicrosteps =
        kExpectedMatterMicrosteps;
    resolve.transactionSlot = 0u;
    resolve.programFingerprint = program.fingerprint;
    resolve.transactionFingerprint = kLogicalTransactionFingerprint;
    resolve.linearizationEpoch = kLogicalLinearizationEpoch;
    resolve.slotGeneration = 1u;
    if (!program.resolve(program.context, post, resolve)) {
        error = "128-DoF joint resolution was rejected";
        return false;
    }
    __unsafe_unretained id<MTLBuffer> statuses =
        (__bridge id<MTLBuffer>)arena.jointStatuses;
    if (!copyBuffer(
            commandBuffer,
            statuses,
            buffers.finalStatus,
            sizeof(MRNumanXCoupledHumanStatusGPU),
            @"128-DoF final status") ||
        !copyBuffer(
            commandBuffer,
            reaction,
            buffers.finalReaction,
            vectorCapacityBytes,
            @"128-DoF final reaction")) {
        error = "128-DoF final snapshot encode failed";
        return false;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        const char* description =
            commandBuffer.error.localizedDescription.UTF8String;
        error = std::string("128-DoF Metal execution failed: ") +
            (description != nullptr ? description : "unknown");
        return false;
    }

    const auto* candidateQ =
        static_cast<const float*>(buffers.candidateQ.contents);
    const auto* massOutput =
        static_cast<const float*>(buffers.massOutput.contents);
    const auto* inverseOutput =
        static_cast<const float*>(buffers.inverseOutput.contents);
    const auto* impulse =
        static_cast<const float*>(buffers.generalizedImpulse.contents);
    const auto* reactionOutput =
        static_cast<const float*>(buffers.finalReaction.contents);
    for (std::uint32_t coordinate = 0u;
         coordinate < kLogicalQCount;
         ++coordinate) {
        if (candidateQ[coordinate] != sourceQ[coordinate]) {
            error = "128-DoF candidate q did not preserve logical source";
            return false;
        }
    }
    for (std::uint32_t dof = 0u; dof < kLogicalDofCount; ++dof) {
        const float expectedMass = 4.0f * massInput[dof];
        const float expectedInverse = 0.25f * inverseInput[dof];
        const float expectedImpulse = 4.0f * deltaV[dof];
        const float expectedReaction =
            expectedImpulse / kTimestepSeconds;
        if (std::fabs(massOutput[dof] - expectedMass) > 1.0e-6f ||
            std::fabs(inverseOutput[dof] - expectedInverse) > 1.0e-6f ||
            std::fabs(impulse[dof] - expectedImpulse) > 1.0e-6f ||
            std::fabs(reactionOutput[dof] - expectedReaction) > 1.0e-6f) {
            error = "128-DoF operator output did not match logical A0";
            return false;
        }
    }
    const std::size_t logicalVectorTailOffset = logicalVBytes;
    const std::size_t logicalVectorTailBytes =
        vectorCapacityBytes - logicalVectorTailOffset;
    if (!markerBytesPreserved(
            buffers.candidateQ,
            logicalQBytes,
            qCapacityBytes - logicalQBytes) ||
        !markerBytesPreserved(
            buffers.massOutput,
            logicalVectorTailOffset,
            logicalVectorTailBytes) ||
        !markerBytesPreserved(
            buffers.inverseOutput,
            logicalVectorTailOffset,
            logicalVectorTailBytes) ||
        !markerBytesPreserved(
            buffers.generalizedImpulse,
            logicalVectorTailOffset,
            logicalVectorTailBytes) ||
        !markerBytesPreserved(
            buffers.finalReaction,
            logicalVectorTailOffset,
            logicalVectorTailBytes)) {
        error = "logical kernels touched capacity-only tail storage";
        return false;
    }
    const auto* inverseStatus =
        static_cast<const MRInverseMassStatusGPU*>(
            buffers.inverseStatus.contents);
    const auto* finalStatus =
        static_cast<const MRNumanXCoupledHumanStatusGPU*>(
            buffers.finalStatus.contents);
    if (inverseStatus[0].code != MR_INVERSE_MASS_SUCCESS ||
        inverseStatus[0].nq != kLogicalQCount ||
        inverseStatus[0].nv != kLogicalDofCount ||
        finalStatus[0].decision != MR_NUMANX_COUPLED_HUMAN_ACCEPT ||
        finalStatus[0].humanCode != MR_NUMI_HUMAN_STAND_SUCCESS ||
        finalStatus[0].matterCode != kMatterSuccess ||
        finalStatus[0].matterCompletedMicrosteps !=
            kExpectedMatterMicrosteps) {
        error = "128-DoF final status lost logical identity or acceptance";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            return fail(2, "usage: numanx_coupled_human_probe <metallib>");
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return fail(3, "no system-default Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return fail(4, "failed to create Metal queue");

        MetalNumanXCoupledHumanConfig config{};
        config.metallibPath = argv[1];
        config.environmentCapacity = kEnvironmentCount;
        config.pointCapacity = 0u;
        config.transactionSlotCount = 1u;
        MetalNumanXCoupledHumanContext context(std::move(config));
        const auto initialization = context.initialize();
        if (!initialization.succeeded()) {
            return fail(
                5,
                std::string("context initialization failed [") +
                    metalrobo::metalNumanXCoupledHumanHostStatusName(
                        initialization.status) +
                    "]: " + initialization.message
            );
        }
        const MetalNumanXCoupledHumanProgram program = context.program();
        if (!program.valid()) return fail(6, "program is not valid");
        if ((program.capabilities &
             MR_NUMANX_COUPLED_HUMAN_CAP_TANGENT_PROJECTOR) != 0u ||
            (program.capabilities &
             MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS) != 0u) {
            return fail(
                7,
                "baseline program overclaims per-pass owner capabilities"
            );
        }

        MetalNumanXCoupledHumanArenaView arena{};
        const auto arenaDiagnostics = context.arenaView(0u, arena);
        if (!arenaDiagnostics.succeeded() ||
            arena.reactionStride != kDofCount ||
            arena.jointStatusStride != 1u ||
            arena.environmentCapacity != kEnvironmentCount ||
            arena.reactionByteCount !=
                kVectorElementCount * sizeof(float) ||
            arena.jointStatusByteCount !=
                kEnvironmentCount *
                    sizeof(MRNumanXCoupledHumanStatusGPU)) {
            return fail(8, "private arena view is not exact-sized/versioned");
        }

        ProbeBuffers buffers = makeBuffers(device);
        if (!allBuffersValid(buffers)) {
            return fail(9, "failed to allocate probe buffers");
        }
        initializeImmutableInputs(buffers);
        std::vector<std::uint8_t> originalQ(
            static_cast<const std::uint8_t*>(buffers.sourceQ.contents),
            static_cast<const std::uint8_t*>(buffers.sourceQ.contents) +
                buffers.sourceQ.length
        );
        std::vector<std::uint8_t> originalV(
            static_cast<const std::uint8_t*>(buffers.sourceV.contents),
            static_cast<const std::uint8_t*>(buffers.sourceV.contents) +
                buffers.sourceV.length
        );

        ExactCallbackContext callback{};
        callback.factorSeed = buffers.factorSeed;
        callback.bodySeed = buffers.bodySeed;
        callback.marker = buffers.callbackMarker;

        if (!constrainedPassFailsClosed(
                queue,
                program,
                buffers,
                callback,
                MR_NUMI_HUMAN_STAND_ENABLE_CONTACT,
                false) ||
            !constrainedPassFailsClosed(
                queue,
                program,
                buffers,
                callback,
                MR_NUMI_HUMAN_STAND_ENABLE_CONTACT,
                true) ||
            !constrainedPassFailsClosed(
                queue,
                program,
                buffers,
                callback,
                MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES,
                false) ||
            !constrainedPassFailsClosed(
                queue,
                program,
                buffers,
                callback,
                MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES,
                true)) {
            return fail(
                10,
                "contact/equality pass did not fail closed with and without explicit P"
            );
        }

        RunCapture first{};
        RunCapture replay{};
        std::string runError;
        if (!runTransaction(
                queue,
                program,
                arena,
                buffers,
                callback,
                first,
                runError,
                1u)) {
            return fail(11, "first transaction: " + runError);
        }
        if (std::memcmp(
                originalQ.data(),
                buffers.sourceQ.contents,
                originalQ.size()) != 0 ||
            std::memcmp(
                originalV.data(),
                buffers.sourceV.contents,
                originalV.size()) != 0) {
            return fail(12, "candidate service mutated live source q/v");
        }

        if (!runTransaction(
                queue,
                program,
                arena,
                buffers,
                callback,
                replay,
                runError,
                2u)) {
            return fail(13, "replay transaction: " + runError);
        }
        if (first.bytes != replay.bytes) {
            return fail(14, "same-input replay was not byte-identical");
        }
        if (callback.invocationCount != 2u) {
            return fail(15, "exact callback invocation count is not two");
        }
        if (!staleGenerationPreservesArena(
                queue,
                program,
                arena,
                buffers,
                callback,
                2u,
                runError)) {
            return fail(16, "stale-generation transaction: " + runError);
        }
        if (!missingPublishFailsClosed(
                queue,
                program,
                arena,
                buffers,
                callback,
                runError,
                3u)) {
            return fail(17, "missing-publish transaction: " + runError);
        }
        if (!invalidOrderingFailsClosed(
                queue,
                program,
                arena,
                buffers,
                callback,
                runError)) {
            return fail(18, "ordered-witness transaction: " + runError);
        }
        if (!gpuIntervalAliasesFailClosed(
                device,
                queue,
                program,
                arena,
                buffers,
                callback,
                runError)) {
            return fail(19, "GPU interval-alias transaction: " + runError);
        }
        if (!runLogicalCapacityTransaction(
                device, queue, argv[1], runError)) {
            return fail(20, "variable logical-DoF transaction: " + runError);
        }

        std::printf(
            "numanx_coupled_human_probe: PASS device=%s dofs=%u q=%u "
            "logical_dofs=%u logical_q=%u transport_dofs=%u transport_q=%u "
            "environments=%u replay_bytes=%zu decisions=pending/accept/"
            "reject_human/reject_matter constrained_with_or_without_P=rejected "
            "stale_generation=rejected ordered_witnesses=rejected "
            "same_object_alias=rejected distinct_heap_alias=rejected "
            "logical_capacity_mismatch=rejected capacity_tail=untouched "
            "operator=source_effective_tangent_A0 generalized_reaction=staged\n",
            device.name.UTF8String,
            kDofCount,
            kQCount,
            kLogicalDofCount,
            kLogicalQCount,
            kMatterVectorCapacity,
            kMatterQCapacity,
            kEnvironmentCount,
            first.bytes.size()
        );
        return 0;
    }
}
