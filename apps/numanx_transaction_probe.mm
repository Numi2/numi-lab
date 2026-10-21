#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#error "METALROBO_DEFAULT_METALLIB must name the build-tree Metal library"
#endif

namespace {

constexpr std::uint32_t kStepCount = 2u;
constexpr float kTimestepSeconds = 1.0e-4f;
constexpr std::uint64_t kProgramFingerprint = 0x4e554d414e585458ull;
constexpr std::uint8_t kMarkerSentinel = 0xa5u;
constexpr std::size_t kPhaseCount = 3u;
constexpr std::size_t kMarkerCount = kStepCount * kPhaseCount;

using Phase = metalrobo::MetalNumanXTransactionPhase;
using Pass = metalrobo::MetalNumanXTransactionPass;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct BorrowedResources {
    void* q = nullptr;
    void* v = nullptr;
    void* bodyPoses = nullptr;
    void* pointWorld = nullptr;
    void* pointJacobians = nullptr;
    void* mujocoMuscles = nullptr;
    void* mujocoStates = nullptr;
    void* mujocoSites = nullptr;
    void* mujocoWraps = nullptr;
    void* mujocoRouteNodes = nullptr;
    void* mujocoResults = nullptr;
    void* mujocoGeneralizedForceArena = nullptr;
    void* tendonBindings = nullptr;
    void* tendonEnvelopes = nullptr;
    void* tendonTransfers = nullptr;
    void* tendonGeneralizedCorrections = nullptr;
    void* standStatuses = nullptr;
    void* standContacts = nullptr;
    void* standVectorWorkspace = nullptr;
};

BorrowedResources borrowedResources(const Pass& pass) noexcept {
    return {
        .q = pass.q,
        .v = pass.v,
        .bodyPoses = pass.bodyPoses,
        .pointWorld = pass.pointWorld,
        .pointJacobians = pass.pointJacobians,
        .mujocoMuscles = pass.mujocoMuscles,
        .mujocoStates = pass.mujocoStates,
        .mujocoSites = pass.mujocoSites,
        .mujocoWraps = pass.mujocoWraps,
        .mujocoRouteNodes = pass.mujocoRouteNodes,
        .mujocoResults = pass.mujocoResults,
        .mujocoGeneralizedForceArena = pass.mujocoGeneralizedForceArena,
        .tendonBindings = pass.tendonBindings,
        .tendonEnvelopes = pass.tendonEnvelopes,
        .tendonTransfers = pass.tendonTransfers,
        .tendonGeneralizedCorrections = pass.tendonGeneralizedCorrections,
        .standStatuses = pass.standStatuses,
        .standContacts = pass.standContacts,
        .standVectorWorkspace = pass.standVectorWorkspace,
    };
}

bool sameResources(
    const BorrowedResources& lhs,
    const BorrowedResources& rhs
) noexcept {
    return lhs.q == rhs.q && lhs.v == rhs.v &&
        lhs.bodyPoses == rhs.bodyPoses && lhs.pointWorld == rhs.pointWorld &&
        lhs.pointJacobians == rhs.pointJacobians &&
        lhs.mujocoMuscles == rhs.mujocoMuscles &&
        lhs.mujocoStates == rhs.mujocoStates &&
        lhs.mujocoSites == rhs.mujocoSites &&
        lhs.mujocoWraps == rhs.mujocoWraps &&
        lhs.mujocoRouteNodes == rhs.mujocoRouteNodes &&
        lhs.mujocoResults == rhs.mujocoResults &&
        lhs.mujocoGeneralizedForceArena == rhs.mujocoGeneralizedForceArena &&
        lhs.tendonBindings == rhs.tendonBindings &&
        lhs.tendonEnvelopes == rhs.tendonEnvelopes &&
        lhs.tendonTransfers == rhs.tendonTransfers &&
        lhs.tendonGeneralizedCorrections ==
            rhs.tendonGeneralizedCorrections &&
        lhs.standStatuses == rhs.standStatuses &&
        lhs.standContacts == rhs.standContacts &&
        lhs.standVectorWorkspace == rhs.standVectorWorkspace;
}

bool borrowedBufferOnDevice(
    void* opaque,
    const std::uint64_t registryID
) noexcept {
    if (opaque == nullptr) {
        return false;
    }
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)opaque;
    return buffer != nil && buffer.device != nil &&
        buffer.device.registryID == registryID && buffer.length > 0u;
}

bool validBorrowedResources(
    const BorrowedResources& resources,
    const std::uint64_t registryID
) noexcept {
    return borrowedBufferOnDevice(resources.q, registryID) &&
        borrowedBufferOnDevice(resources.v, registryID) &&
        borrowedBufferOnDevice(resources.bodyPoses, registryID) &&
        borrowedBufferOnDevice(resources.pointWorld, registryID) &&
        borrowedBufferOnDevice(resources.pointJacobians, registryID) &&
        borrowedBufferOnDevice(resources.mujocoMuscles, registryID) &&
        borrowedBufferOnDevice(resources.mujocoStates, registryID) &&
        borrowedBufferOnDevice(resources.mujocoSites, registryID) &&
        borrowedBufferOnDevice(resources.mujocoWraps, registryID) &&
        borrowedBufferOnDevice(resources.mujocoRouteNodes, registryID) &&
        borrowedBufferOnDevice(resources.mujocoResults, registryID) &&
        borrowedBufferOnDevice(
            resources.mujocoGeneralizedForceArena, registryID
        ) &&
        borrowedBufferOnDevice(resources.tendonBindings, registryID) &&
        borrowedBufferOnDevice(resources.tendonEnvelopes, registryID) &&
        borrowedBufferOnDevice(resources.tendonTransfers, registryID) &&
        borrowedBufferOnDevice(
            resources.tendonGeneralizedCorrections, registryID
        ) &&
        borrowedBufferOnDevice(resources.standStatuses, registryID) &&
        borrowedBufferOnDevice(resources.standContacts, registryID) &&
        borrowedBufferOnDevice(resources.standVectorWorkspace, registryID);
}

struct PhaseRecord {
    Phase phase = Phase::beginStep;
    std::uint32_t stepIndex = 0u;
};

struct CandidateSample {
    id<MTLBuffer> deltaVelocity = nil;
    id<MTLBuffer> candidateQ = nil;
    id<MTLBuffer> candidateBodies = nil;
    id<MTLBuffer> pointQueries = nil;
    id<MTLBuffer> candidateRootTranslation = nil;
    id<MTLBuffer> candidateBodyPositionLow = nil;
    id<MTLBuffer> pointJacobians = nil;
};

struct TransactionAudit {
    bool rollback = false;
    bool candidateKinematics = false;
    bool sourceMassOperations = false;
    bool candidateEncoded = false;
    bool sourceMassOperationsEncoded = false;
    id<MTLBuffer> sourceMassInput = nil;
    id<MTLBuffer> sourceMassAction = nil;
    id<MTLBuffer> sourceMassSolve = nil;
    id<MTLBuffer> sourceMassActionStatus = nil;
    id<MTLBuffer> sourceMassSolveStatus = nil;
    std::vector<float> sourceMassExpected;
    const char* failure = nullptr;
    void* commandBuffer = nullptr;
    BorrowedResources resources{};
    std::array<PhaseRecord, kMarkerCount> records{};
    std::size_t recordCount = 0u;
    std::uint32_t rejectionCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t candidateQueryCount = 0u;
    std::array<CandidateSample, 2u> candidateSamples{};
    id<MTLBuffer> markers = nil;

    bool fail(const char* message) noexcept {
        if (failure == nullptr) failure = message;
        return false;
    }
};

Phase expectedPhase(const std::size_t ordinal) noexcept {
    switch (ordinal % kPhaseCount) {
    case 0u:
        return Phase::beginStep;
    case 1u:
        return Phase::preDynamics;
    default:
        return Phase::postDynamics;
    }
}

bool validatePassMetadata(const Pass& pass, const TransactionAudit& audit) noexcept {
    const std::uint32_t expectedAccessFlags =
        metalrobo::MetalNumanXTransactionReadBorrowedState |
        (audit.candidateKinematics && pass.phase == Phase::preDynamics
             ? metalrobo::MetalNumanXTransactionEncodeExactCandidateKinematics
             : 0u) |
        (audit.sourceMassOperations && pass.phase == Phase::preDynamics
             ? metalrobo::MetalNumanXTransactionEncodeSourceMassOperations
             : 0u) |
        (pass.phase == Phase::beginStep
             ? metalrobo::MetalNumanXTransactionWriteMujocoExcitation
             : 0u) |
        (pass.phase == Phase::postDynamics
             ? metalrobo::MetalNumanXTransactionWriteStandFailure
             : 0u);
    const std::uint32_t expectedABI = audit.sourceMassOperations
        ? metalrobo::kMetalNumanXTransactionSourceMassABIVersion
        : (audit.candidateKinematics
            ? metalrobo::kMetalNumanXTransactionCandidateKinematicsABIVersion
            : metalrobo::kMetalNumanXTransactionABIVersion);
    const bool candidatePassValid = !audit.candidateKinematics ||
        (pass.phase != Phase::preDynamics ||
         (pass.exactCandidateKinematicsContext != nullptr &&
          pass.encodeExactCandidateKinematics != nullptr &&
          pass.candidatePhysicsSubstepCount == 1u &&
          pass.candidateTransactionFingerprint != 0u &&
          pass.candidateLinearizationEpoch != 0u &&
          pass.candidateSlotGeneration != 0u));
    const bool sourceMassPassValid = !audit.sourceMassOperations ||
        (pass.phase != Phase::preDynamics ||
         (pass.sourceMassOperationContext != nullptr &&
          pass.encodeSourceMassOperation != nullptr &&
          pass.sourceMassMatrix != nullptr &&
          pass.sourceMassAssemblyStatuses != nullptr &&
          pass.sourceMassMatrixElementCount ==
              pass.environmentCount * pass.sourceMassMatrixStride));
    return pass.abiVersion == expectedABI && candidatePassValid &&
        sourceMassPassValid &&
        pass.structSize == metalrobo::metalNumanXTransactionPassStructSizeForABI(expectedABI) &&
        pass.accessFlags == expectedAccessFlags && pass.reserved0 == 0u &&
        pass.programFingerprint == kProgramFingerprint &&
        pass.stepCount == kStepCount &&
        pass.timestepSeconds == kTimestepSeconds &&
        pass.articulationFirstBody == 1u &&
        pass.bodyJacobianPointOffset == 0u &&
        pass.environmentCount == 1u &&
        pass.qCoordinateCount == 7u && pass.qElementCount == 7u &&
        pass.qStride == 7u && pass.dofCount == 6u &&
        pass.vElementCount == 6u && pass.vStride == 6u &&
        pass.bodyCount == 1u && pass.bodyPoseElementCount == 1u &&
        pass.bodyPoseStride == 1u && pass.pointCount == 4u &&
        pass.pointWorldElementCount == 4u && pass.pointWorldStride == 4u &&
        pass.pointJacobianElementCount == 72u &&
        pass.pointJacobianStride == 72u &&
        pass.mujocoMuscleCount == 1u &&
        pass.mujocoStateElementCount == 1u &&
        pass.mujocoStateStride == 1u && pass.mujocoSiteCount == 2u &&
        pass.mujocoWrapCount == 0u && pass.mujocoRouteNodeCount == 2u &&
        pass.mujocoResultElementCount == 1u &&
        pass.mujocoResultStride == 1u &&
        pass.mujocoMuscleGeneralizedForceElementCount == 6u &&
        pass.mujocoMuscleGeneralizedForceRowStride == 6u &&
        pass.mujocoMuscleGeneralizedForceEnvironmentStride == 6u &&
        pass.mujocoGeneralizedForceElementCount == 6u &&
        pass.mujocoGeneralizedForceOffset == 6u &&
        pass.mujocoGeneralizedForceStride == 6u &&
        pass.mujocoGeneralizedForceArenaElementCount == 12u &&
        pass.tendonBindingCount == 0u && pass.tendonEnvelopeCount == 0u &&
        pass.tendonTransferElementCount == 0u &&
        pass.tendonTransferStride == 0u &&
        pass.tendonCorrectionElementCount == 0u &&
        pass.tendonCorrectionStride == 0u &&
        pass.standStatusElementCount == 1u && pass.standStatusStride == 1u &&
        pass.standContactCount == 0u &&
        pass.standVectorElementCount == 24u && pass.standVectorStride == 24u &&
        pass.standContactImpulseOffset == 24u &&
        pass.standContactImpulseSampleStepIndex ==
            (pass.phase == Phase::postDynamics ? pass.stepIndex : MR_INVALID_INDEX) &&
        pass.standContactEnabled == 0u &&
        pass.standGroundPoint.x == 0.0f && pass.standGroundPoint.y == 0.0f &&
        pass.standGroundPoint.z == 0.0f && pass.standGroundPoint.w == 0.0f &&
        pass.standGroundNormal.x == 0.0f && pass.standGroundNormal.y == 1.0f &&
        pass.standGroundNormal.z == 0.0f && pass.standGroundNormal.w == 0.0f;
}

bool encodeCandidateKinematics(TransactionAudit& audit, const Pass& pass) noexcept {
    if (audit.candidateEncoded) return true;
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    if (commandBuffer == nil || commandBuffer.device == nil ||
        pass.exactCandidateKinematicsContext == nullptr ||
        pass.encodeExactCandidateKinematics == nullptr ||
        pass.environmentCount != 1u || pass.qCoordinateCount == 0u ||
        pass.dofCount < 4u || pass.pointCount != 4u) {
        return audit.fail("candidate FK fixture received invalid source metadata");
    }
    id<MTLDevice> device = commandBuffer.device;
    const NSUInteger qBytes = static_cast<NSUInteger>(pass.qCoordinateCount) * sizeof(float);
    const NSUInteger vBytes = static_cast<NSUInteger>(pass.dofCount) * sizeof(float);
    const NSUInteger bodyStride =
        static_cast<NSUInteger>(pass.articulationFirstBody + pass.bodyCount);
    const NSUInteger bodyBytes = bodyStride * sizeof(MRBodyStateGPU);
    const NSUInteger lowBytes = bodyStride * sizeof(mr_float4);
    const NSUInteger jacobianCount = 3u * static_cast<NSUInteger>(pass.dofCount);
    const NSUInteger jacobianBytes = jacobianCount * sizeof(float);
    if (bodyStride == 0u || qBytes == 0u || vBytes == 0u ||
        bodyBytes == 0u || lowBytes == 0u || jacobianBytes == 0u) {
        return audit.fail("candidate FK fixture dimensions are empty");
    }
    MRArticulatedPointImpulseGPU point{};
    point.bodyIndex = pass.articulationFirstBody;
    point.localPoint = {0.137f, -0.083f, 0.219f, 0.0f};
    for (std::size_t sampleIndex = 0u;
         sampleIndex < audit.candidateSamples.size();
         ++sampleIndex) {
        CandidateSample& sample = audit.candidateSamples[sampleIndex];
        sample.deltaVelocity = [device newBufferWithLength:vBytes
            options:MTLResourceStorageModeShared];
        sample.candidateQ = [device newBufferWithLength:qBytes
            options:MTLResourceStorageModeShared];
        sample.candidateBodies = [device newBufferWithLength:bodyBytes
            options:MTLResourceStorageModeShared];
        sample.pointQueries = [device newBufferWithLength:sizeof(point)
            options:MTLResourceStorageModeShared];
        sample.candidateRootTranslation = [device newBufferWithLength:
            sizeof(MRCompensatedRootTranslationGPU)
            options:MTLResourceStorageModeShared];
        sample.candidateBodyPositionLow = [device newBufferWithLength:lowBytes
            options:MTLResourceStorageModeShared];
        sample.pointJacobians = [device newBufferWithLength:jacobianBytes
            options:MTLResourceStorageModeShared];
        if (sample.deltaVelocity == nil || sample.candidateQ == nil ||
            sample.candidateBodies == nil || sample.pointQueries == nil ||
            sample.candidateRootTranslation == nil ||
            sample.candidateBodyPositionLow == nil ||
            sample.pointJacobians == nil) {
            return audit.fail("candidate FK fixture could not allocate buffers");
        }
        std::memset(sample.deltaVelocity.contents, 0, vBytes);
        std::memset(sample.candidateQ.contents, 0xa5, qBytes);
        std::memset(sample.candidateBodies.contents, 0xa5, bodyBytes);
        std::memcpy(sample.pointQueries.contents, &point, sizeof(point));
        std::memset(sample.candidateRootTranslation.contents, 0xa5,
                    sizeof(MRCompensatedRootTranslationGPU));
        std::memset(sample.candidateBodyPositionLow.contents, 0xa5, lowBytes);
        std::memset(sample.pointJacobians.contents, 0xa5, jacobianBytes);
        if (sampleIndex == 1u) {
            static_cast<float*>(sample.deltaVelocity.contents)[3u] = 1.0f;
        }

        metalrobo::MetalNumanXHumanMatterCandidateQuery query{};
        query.accessFlags =
            metalrobo::MetalNumanXHumanMatterCandidateReadDeltaVelocity |
            metalrobo::MetalNumanXHumanMatterCandidateWriteQ |
            metalrobo::MetalNumanXHumanMatterCandidateWriteBodies |
            metalrobo::MetalNumanXHumanMatterCandidateReadPointQueries |
            metalrobo::MetalNumanXHumanMatterCandidateWritePointJacobians;
        query.deltaVelocity = (__bridge void*)sample.deltaVelocity;
        query.candidateQ = (__bridge void*)sample.candidateQ;
        query.candidateBodies = (__bridge void*)sample.candidateBodies;
        query.pointQueries = (__bridge void*)sample.pointQueries;
        query.pointJacobians = (__bridge void*)sample.pointJacobians;
        query.candidateRootTranslation =
            (__bridge void*)sample.candidateRootTranslation;
        query.candidateBodyPositionLow =
            (__bridge void*)sample.candidateBodyPositionLow;
        query.deltaVelocityGPUAddress = sample.deltaVelocity.gpuAddress;
        query.candidateQGPUAddress = sample.candidateQ.gpuAddress;
        query.candidateBodiesGPUAddress = sample.candidateBodies.gpuAddress;
        query.pointQueriesGPUAddress = sample.pointQueries.gpuAddress;
        query.pointJacobiansGPUAddress = sample.pointJacobians.gpuAddress;
        query.candidateRootTranslationGPUAddress =
            sample.candidateRootTranslation.gpuAddress;
        query.candidateBodyPositionLowGPUAddress =
            sample.candidateBodyPositionLow.gpuAddress;
        query.deltaVelocityStride = static_cast<std::uint32_t>(pass.dofCount);
        query.candidateQStride = static_cast<std::uint32_t>(pass.qCoordinateCount);
        query.candidateBodyStride = static_cast<std::uint32_t>(bodyStride);
        query.pointCount = 1u;
        query.pointStride = 1u;
        query.pointJacobianStride = static_cast<std::uint32_t>(jacobianCount);
        query.substepIndex = pass.candidateSubstepIndex;
        query.transactionSlot = pass.candidateTransactionSlot;
        query.physicsSubstepCount = pass.candidatePhysicsSubstepCount;
        query.controlStep = pass.candidateControlStep;
        query.programFingerprint = pass.programFingerprint;
        query.transactionFingerprint = pass.candidateTransactionFingerprint;
        query.linearizationEpoch = pass.candidateLinearizationEpoch;
        query.slotGeneration = pass.candidateSlotGeneration;
        query.candidateRootTranslationElementCount = 1u;
        query.candidateBodyPositionLowElementCount = bodyStride;
        if (!pass.encodeExactCandidateKinematics(
                pass.exactCandidateKinematicsContext, pass, query)) {
            return audit.fail("exact candidate FK callback rejected the valid fixture query");
        }
        ++audit.candidateQueryCount;
    }
    audit.candidateEncoded = true;
    return true;
}

bool encodeTransaction(void* context, const Pass& pass) noexcept {
    auto& audit = *static_cast<TransactionAudit*>(context);
    if (pass.commandBuffer == nullptr || audit.markers == nil) {
        return audit.fail("transaction pass lacks its command buffer or marker");
    }
    if (audit.recordCount >= audit.records.size()) {
        return audit.fail("transaction offered more than three phases per step");
    }

    const std::size_t ordinal = audit.recordCount;
    const Phase phase = expectedPhase(ordinal);
    const std::uint32_t stepIndex = static_cast<std::uint32_t>(
        ordinal / kPhaseCount
    );
    if (pass.phase != phase || pass.stepIndex != stepIndex) {
        return audit.fail("transaction phases are not in canonical step order");
    }
    if (!validatePassMetadata(pass, audit)) {
        return audit.fail("transaction pass metadata disagrees with the fixture");
    }

    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    if (commandBuffer == nil || commandBuffer.device == nil) {
        return audit.fail("transaction command buffer is not a Metal object");
    }
    const std::uint64_t registryID = commandBuffer.device.registryID;
    const BorrowedResources resources = borrowedResources(pass);
    if (!validBorrowedResources(resources, registryID)) {
        return audit.fail("transaction did not borrow every expected Metal buffer");
    }
    if (audit.commandBuffer == nullptr) {
        audit.commandBuffer = pass.commandBuffer;
        audit.resources = resources;
    } else if (audit.commandBuffer != pass.commandBuffer ||
               !sameResources(audit.resources, resources)) {
        return audit.fail("transaction changed command buffer or resources mid-horizon");
    }

    audit.records[ordinal] = {.phase = pass.phase, .stepIndex = pass.stepIndex};
    ++audit.recordCount;
    if (audit.candidateKinematics && pass.phase == Phase::preDynamics &&
        !encodeCandidateKinematics(audit, pass)) {
        return false;
    }
    if (audit.sourceMassOperations && pass.phase == Phase::preDynamics) {
        if (pass.sourceMassOperationContext == nullptr ||
            pass.encodeSourceMassOperation == nullptr ||
            pass.sourceMassMatrix == nullptr ||
            pass.sourceMassAssemblyStatuses == nullptr ||
            pass.sourceMassMatrixStride != pass.dofCount * pass.dofCount ||
            audit.sourceMassInput == nil || audit.sourceMassAction == nil ||
            audit.sourceMassSolve == nil ||
            audit.sourceMassActionStatus == nil ||
            audit.sourceMassSolveStatus == nil) {
            return audit.fail("ABI 8 source-M fields or fixture buffers are incomplete");
        }
        metalrobo::MetalNumanXTransactionSourceMassQuery query{};
        query.operation = MR_ARTICULATED_SOURCE_MASS_ACTION;
        query.input = (__bridge void*)audit.sourceMassInput;
        query.output = (__bridge void*)audit.sourceMassAction;
        query.statuses = (__bridge void*)audit.sourceMassActionStatus;
        query.inputGPUAddress = audit.sourceMassInput.gpuAddress;
        query.outputGPUAddress = audit.sourceMassAction.gpuAddress;
        query.statusesGPUAddress = audit.sourceMassActionStatus.gpuAddress;
        query.inputElementCount = audit.sourceMassInput.length / sizeof(float);
        query.outputElementCount = audit.sourceMassAction.length / sizeof(float);
        query.statusElementCount = audit.sourceMassActionStatus.length /
            sizeof(std::uint32_t);
        query.generalizedVectorStride = static_cast<std::uint32_t>(pass.dofCount);
        query.statusStride = 1u;
        auto unsupportedCandidate = query;
        unsupportedCandidate.candidateQ = pass.q;
        __unsafe_unretained id<MTLBuffer> sourceQ =
            (__bridge id<MTLBuffer>)pass.q;
        unsupportedCandidate.candidateQGPUAddress = sourceQ.gpuAddress;
        if (pass.encodeSourceMassOperation(
                pass.sourceMassOperationContext, pass,
                unsupportedCandidate)) {
            return audit.fail("source-M operation accepted unsupported candidate q");
        }
        if (!pass.encodeSourceMassOperation(
                pass.sourceMassOperationContext, pass, query)) {
            return audit.fail("source-M action rejected valid same-command-buffer query");
        }
        query.operation = MR_ARTICULATED_SOURCE_MASS_SOLVE;
        query.input = (__bridge void*)audit.sourceMassAction;
        query.output = (__bridge void*)audit.sourceMassSolve;
        query.statuses = (__bridge void*)audit.sourceMassSolveStatus;
        query.inputGPUAddress = audit.sourceMassAction.gpuAddress;
        query.outputGPUAddress = audit.sourceMassSolve.gpuAddress;
        query.statusesGPUAddress = audit.sourceMassSolveStatus.gpuAddress;
        query.inputElementCount = audit.sourceMassAction.length / sizeof(float);
        query.outputElementCount = audit.sourceMassSolve.length / sizeof(float);
        query.statusElementCount = audit.sourceMassSolveStatus.length /
            sizeof(std::uint32_t);
        if (!pass.encodeSourceMassOperation(
                pass.sourceMassOperationContext, pass, query)) {
            return audit.fail("source-M factor solve rejected valid same-command-buffer query");
        }
        audit.sourceMassOperationsEncoded = true;
    }
    if (audit.rollback && pass.phase == Phase::preDynamics &&
        pass.stepIndex == 0u) {
        ++audit.rejectionCount;
        return false;
    }

    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
        return audit.fail("transaction could not create its Metal blit encoder");
    }
    [encoder fillBuffer:audit.markers
                  range:NSMakeRange(ordinal, 1u)
                  value:static_cast<std::uint8_t>(0x20u + ordinal)];
    [encoder endEncoding];
    return true;
}

void abortTransaction(void* context, void* commandBuffer) noexcept {
    auto& audit = *static_cast<TransactionAudit*>(context);
    ++audit.abortCount;
    if (audit.commandBuffer == nullptr ||
        audit.commandBuffer != commandBuffer) {
        audit.fail("transaction abort did not return the borrowed command buffer");
    }
}

std::vector<MRArticulatedPointImpulseGPU> makeBodyProbes() {
    constexpr std::array<mr_float4, 4u> points{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    std::vector<MRArticulatedPointImpulseGPU> result;
    result.reserve(points.size());
    for (const mr_float4 point : points) {
        MRArticulatedPointImpulseGPU query{};
        query.bodyIndex = 1u;
        query.localPoint = point;
        result.push_back(query);
    }
    return result;
}

struct SyntheticHumanFixture {
    metalrobo::EngineModel model = metalrobo::makeFreeSphereEngineModel();
    std::vector<float> q = model.defaultQ;
    std::vector<float> v = model.defaultV;
    std::vector<MRArticulatedPointImpulseGPU> points = makeBodyProbes();
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleRouteNodeGPU> routeNodes;

    SyntheticHumanFixture() {
        MRMujocoMuscleSiteGPU origin{};
        origin.bodyIndex = 1u;
        origin.localPoint = {-0.05f, 0.0f, 0.0f, 0.0f};
        sites.push_back(origin);
        MRMujocoMuscleSiteGPU insertion{};
        insertion.bodyIndex = 1u;
        insertion.localPoint = {0.05f, 0.0f, 0.0f, 0.0f};
        sites.push_back(insertion);

        MRMujocoMuscleRouteNodeGPU first{};
        first.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        first.targetIndex = 0u;
        first.sideSiteIndex = MR_INVALID_INDEX;
        routeNodes.push_back(first);
        MRMujocoMuscleRouteNodeGPU second{};
        second.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
        second.targetIndex = 1u;
        second.sideSiteIndex = MR_INVALID_INDEX;
        routeNodes.push_back(second);

        MRMujocoMuscleGPU muscle{};
        muscle.route = {0u, 2u, 0u, 0u};
        muscle.lengthRangeAndAcceleration = {0.05f, 0.15f, 1.0f, 0.0f};
        muscle.controlRange = {0.0f, 1.0f, 0.0f, 0.0f};
        muscles.push_back(muscle);

        MRMujocoMuscleStateGPU state{};
        state.excitationAndActivation = {0.0f, 0.0f, 0.0f, 0.0f};
        states.push_back(state);
    }

    metalrobo::MetalArticulatedOperatorInput input(
        TransactionAudit& audit
    ) const {
        metalrobo::MetalArticulatedOperatorInput result{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = points.size(),
            .q = q,
            .v = v,
            .points = points,
            .mujoco = {
                .muscles = muscles,
                .states = states,
                .sites = sites,
                .wraps = {},
                .routeNodes = routeNodes,
                .bodyJacobianPointOffset = 0u,
            },
            .stand = {
                .v = v,
                .contacts = {},
                .jointEqualities = {},
                .tendonBindings = {},
                .tendonEnvelopes = {},
                .tendonLoadProgram = {},
                .numanXTransactionProgram = {
                    .abiVersion = audit.sourceMassOperations
                        ? metalrobo::kMetalNumanXTransactionSourceMassABIVersion
                        : (audit.candidateKinematics
                            ? metalrobo::kMetalNumanXTransactionCandidateKinematicsABIVersion
                            : metalrobo::kMetalNumanXTransactionABIVersion),
                    .context = &audit,
                    .encode = &encodeTransaction,
                    .abort = &abortTransaction,
                    .fingerprint = kProgramFingerprint,
                    .writeExactCandidateKinematics = audit.candidateKinematics,
                    .candidatePointCapacity = audit.candidateKinematics ? 1u : 0u,
                },
                .stepCount = kStepCount,
                .contactIterationCount = 1u,
                .enableContact = false,
                .enableRootAssistance = false,
            },
        };
        return result;
    }
};

template <typename T>
void seed(std::vector<T>& values, const std::uint8_t byte) {
    static_assert(std::is_trivially_copyable_v<T>);
    values.resize(2u);
    std::memset(values.data(), byte, values.size() * sizeof(T));
}

void seedResult(metalrobo::MetalArticulatedOperatorResult& result) {
    result.layout.dispatch.flags = 0x13579bdfu;
    result.layout.qElements = 17u;
    result.layout.totalAllocatedBytes = 0x2468u;
    seed(result.bodyPoses, 0x10u);
    seed(result.pointWorld, 0x11u);
    seed(result.diagnosticMassMatrix, 0x12u);
    seed(result.pointJacobians, 0x13u);
    seed(result.generalizedImpulse, 0x14u);
    seed(result.deltaVelocity, 0x15u);
    seed(result.statuses, 0x16u);
    seed(result.millardResults, 0x17u);
    seed(result.millardGeneralizedForces, 0x18u);
    seed(result.mujocoResults, 0x19u);
    seed(result.mujocoActivationStates, 0x1au);
    seed(result.mujocoMuscleGeneralizedForces, 0x1bu);
    seed(result.mujocoGeneralizedForces, 0x1cu);
    seed(result.standQ, 0x1du);
    seed(result.standV, 0x1eu);
    seed(result.standStatuses, 0x1fu);
    seed(result.standTendonTransfers, 0x20u);
    seed(result.standTendonGeneralizedCorrections, 0x21u);
}

template <typename T>
void appendObject(std::vector<std::byte>& bytes, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* first = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(T));
}

template <typename T>
void appendVector(std::vector<std::byte>& bytes, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    appendObject(bytes, values.size());
    const auto* first = reinterpret_cast<const std::byte*>(values.data());
    bytes.insert(bytes.end(), first, first + values.size() * sizeof(T));
}

std::vector<std::byte> resultBytes(
    const metalrobo::MetalArticulatedOperatorResult& result
) {
    std::vector<std::byte> bytes;
    appendObject(bytes, result.layout);
    appendVector(bytes, result.bodyPoses);
    appendVector(bytes, result.pointWorld);
    appendVector(bytes, result.diagnosticMassMatrix);
    appendVector(bytes, result.pointJacobians);
    appendVector(bytes, result.generalizedImpulse);
    appendVector(bytes, result.deltaVelocity);
    appendVector(bytes, result.statuses);
    appendVector(bytes, result.millardResults);
    appendVector(bytes, result.millardGeneralizedForces);
    appendVector(bytes, result.mujocoResults);
    appendVector(bytes, result.mujocoActivationStates);
    appendVector(bytes, result.mujocoMuscleGeneralizedForces);
    appendVector(bytes, result.mujocoGeneralizedForces);
    appendVector(bytes, result.standQ);
    appendVector(bytes, result.standV);
    appendVector(bytes, result.standStatuses);
    appendVector(bytes, result.standTendonTransfers);
    appendVector(bytes, result.standTendonGeneralizedCorrections);
    return bytes;
}

TransactionAudit makeAudit(const bool rollback) {
    TransactionAudit audit;
    audit.rollback = rollback;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "NumanX transaction probe requires a local Metal device");
    audit.markers = [device newBufferWithLength:kMarkerCount
                                       options:MTLResourceStorageModeShared];
    require(audit.markers != nil, "could not allocate transaction marker buffer");
    std::memset(audit.markers.contents, kMarkerSentinel, kMarkerCount);
    return audit;
}

metalrobo::MetalArticulatedOperatorContext makeContext() {
    const metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = kTimestepSeconds,
        .metallibPath = METALROBO_DEFAULT_METALLIB,
    };
    return metalrobo::MetalArticulatedOperatorContext(config);
}

void verifyMarkers(const TransactionAudit& audit, const bool committed) {
    const auto* markers = static_cast<const std::uint8_t*>(
        audit.markers.contents
    );
    for (std::size_t index = 0u; index < kMarkerCount; ++index) {
        const std::uint8_t expected = committed
            ? static_cast<std::uint8_t>(0x20u + index)
            : kMarkerSentinel;
        require(markers[index] == expected,
                "transaction marker publication disagrees with commit state");
    }
}

void runAccept() {
    SyntheticHumanFixture fixture;
    TransactionAudit audit = makeAudit(false);
    auto context = makeContext();
    const auto input = fixture.input(audit);
    metalrobo::MetalArticulatedOperatorResult result;
    const auto diagnostics = context.run(fixture.model, input, result);

    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(diagnostics.succeeded() && diagnostics.dispatched &&
                diagnostics.published &&
                diagnostics.completedStandSteps == kStepCount &&
                diagnostics.numanXProgramFingerprint ==
                    kProgramFingerprint &&
                diagnostics.commandBufferIdentity ==
                    reinterpret_cast<std::uintptr_t>(audit.commandBuffer),
            "accepted NumanX transaction did not dispatch and publish: " +
                diagnostics.message);
    require(audit.recordCount == kMarkerCount && audit.rejectionCount == 0u &&
                audit.abortCount == 0u,
            "accepted NumanX transaction did not offer exactly three phases per step");
    require(result.standQ.size() == 7u && result.standV.size() == 6u &&
                result.standStatuses.size() == 1u &&
                result.standStatuses.front().code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                result.standStatuses.front().completedSteps == kStepCount &&
                result.mujocoResults.size() == 1u &&
                result.mujocoResults.front().status ==
                    MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS,
            "accepted NumanX transaction published an incoherent Human result");
    verifyMarkers(audit, true);
    std::cout << "numanx human transaction accept passed device=\""
              << diagnostics.deviceName << "\" phases=" << audit.recordCount
              << " steps=" << diagnostics.completedStandSteps << '\n';
}

void runRollback() {
    SyntheticHumanFixture fixture;
    TransactionAudit audit = makeAudit(true);
    auto context = makeContext();
    const auto input = fixture.input(audit);
    metalrobo::MetalArticulatedOperatorResult result;
    seedResult(result);
    const std::vector<std::byte> sentinel = resultBytes(result);
    const auto diagnostics = context.run(fixture.model, input, result);

    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                !diagnostics.published &&
                diagnostics.status ==
                    metalrobo::MetalArticulatedOperatorHostStatus::
                        externalProgramFailure &&
                std::string(
                    metalrobo::metalArticulatedOperatorHostStatusName(
                        diagnostics.status
                    )
                ) == "external_program_failure",
            "rejected NumanX transaction was dispatched or published");
    require(audit.recordCount == 2u &&
                audit.records[0].phase == Phase::beginStep &&
                audit.records[1].phase == Phase::preDynamics &&
                audit.rejectionCount == 1u && audit.abortCount == 1u,
            "rejected NumanX transaction did not abort exactly once");
    require(resultBytes(result) == sentinel,
            "rejected NumanX transaction changed the caller result sentinel");
    verifyMarkers(audit, false);
    std::cout << "numanx human transaction rollback passed phases="
              << audit.recordCount << " aborts=" << audit.abortCount << '\n';
}

float maxAbsoluteDifference(const float* lhs, const float* rhs,
                            const std::size_t count) {
    float maximum = 0.0f;
    for (std::size_t index = 0u; index < count; ++index) {
        maximum = std::max(maximum, std::abs(lhs[index] - rhs[index]));
    }
    return maximum;
}

void verifyCandidateSampleAgainstDirectFK(
    const SyntheticHumanFixture& fixture,
    const CandidateSample& sample,
    const std::uint32_t dofCount
) {
    const std::size_t nq = fixture.model.articulations[0].nq;
    std::vector<float> candidateQ(
        static_cast<const float*>(sample.candidateQ.contents),
        static_cast<const float*>(sample.candidateQ.contents) + nq);
    const auto* candidateRoot =
        static_cast<const MRCompensatedRootTranslationGPU*>(
            sample.candidateRootTranslation.contents);
    const auto* candidateLow =
        static_cast<const mr_float4*>(sample.candidateBodyPositionLow.contents);
    std::vector<MRCompensatedRootTranslationGPU> roots{candidateRoot[0]};
    std::vector<MRArticulatedPointImpulseGPU> points = fixture.points;
    MRArticulatedPointImpulseGPU candidatePoint{};
    candidatePoint.bodyIndex = fixture.model.articulations[0].firstBody;
    candidatePoint.localPoint = {0.137f, -0.083f, 0.219f, 0.0f};
    points.push_back(candidatePoint);
    const metalrobo::MetalArticulatedOperatorInput referenceInput{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = points.size(),
        .q = candidateQ,
        .rootTranslations = roots,
        .points = points,
    };
    metalrobo::MetalArticulatedOperatorContext directContext({
        .pointJacobiansOnly = true,
        .metallibPath = METALROBO_DEFAULT_METALLIB,
    });
    metalrobo::MetalArticulatedOperatorResult direct;
    const auto diagnostics = directContext.run(fixture.model, referenceInput, direct);
    require(diagnostics.succeeded() && diagnostics.published,
            "direct candidate-FK reference failed: " + diagnostics.message);
    const std::size_t jacobianCount = 3u * dofCount;
    const std::size_t queryOffset = fixture.points.size() * jacobianCount;
    const std::uint32_t localBodyIndex =
        candidatePoint.bodyIndex -
        fixture.model.articulations[0].firstBody;
    require(direct.pointJacobians.size() >= queryOffset + jacobianCount &&
                direct.bodyPositionLow.size() > localBodyIndex,
            "direct candidate-FK reference omitted queried geometry");
    const float jacobianError = maxAbsoluteDifference(
        static_cast<const float*>(sample.pointJacobians.contents),
        direct.pointJacobians.data() + queryOffset, jacobianCount);
    const mr_float4 outputLow = candidateLow[candidatePoint.bodyIndex];
    const mr_float4 referenceLow = direct.bodyPositionLow[localBodyIndex];
    const float lowError = std::max({
        std::abs(outputLow.x - referenceLow.x),
        std::abs(outputLow.y - referenceLow.y),
        std::abs(outputLow.z - referenceLow.z)});
    require(std::isfinite(jacobianError) && jacobianError <= 2.0e-4f &&
                std::isfinite(lowError) && lowError <= 2.0e-4f,
            "transaction candidate FK differs from direct FK: jacobian=" +
                std::to_string(jacobianError) + " body_low=" +
                std::to_string(lowError));
}

void runCandidateKinematics(const bool rollback) {
    SyntheticHumanFixture fixture;
    TransactionAudit audit = makeAudit(rollback);
    audit.candidateKinematics = true;
    const auto originalQ = fixture.q;
    const auto originalV = fixture.v;
    const auto input = fixture.input(audit);
    auto context = makeContext();
    metalrobo::MetalArticulatedOperatorResult result;
    if (rollback) seedResult(result);
    const std::vector<std::byte> sentinel = resultBytes(result);
    const auto diagnostics = context.run(fixture.model, input, result);
    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(audit.candidateQueryCount == 2u && audit.candidateEncoded,
            "transaction did not encode zero and nonzero candidate FK");
    require(fixture.q == originalQ && fixture.v == originalV,
            "candidate FK modified source q or v");
    if (rollback) {
        require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                    !diagnostics.published && audit.abortCount == 1u &&
                    audit.rejectionCount == 1u && resultBytes(result) == sentinel,
                "candidate rejection changed transaction publication state");
        std::cout << "numanx exact candidate FK rollback passed queries="
                  << audit.candidateQueryCount << " aborts=" << audit.abortCount << '\n';
        return;
    }
    require(diagnostics.succeeded() && diagnostics.dispatched &&
                diagnostics.published && audit.abortCount == 0u,
            "candidate FK transaction did not complete: " + diagnostics.message);
    const auto dofCount =
        static_cast<std::uint32_t>(fixture.model.articulations[0].nv);
    for (const CandidateSample& sample : audit.candidateSamples) {
        require(sample.candidateQ != nil &&
                    sample.candidateRootTranslation != nil &&
                    sample.candidateBodyPositionLow != nil &&
                    sample.pointJacobians != nil,
                "candidate FK output buffer was not retained");
        verifyCandidateSampleAgainstDirectFK(fixture, sample, dofCount);
    }
    const std::size_t nq = fixture.model.articulations[0].nq;
    const float qDifference = maxAbsoluteDifference(
        static_cast<const float*>(audit.candidateSamples[0].candidateQ.contents),
        static_cast<const float*>(audit.candidateSamples[1].candidateQ.contents), nq);
    const bool rootDifference = std::memcmp(
        audit.candidateSamples[0].candidateRootTranslation.contents,
        audit.candidateSamples[1].candidateRootTranslation.contents,
        sizeof(MRCompensatedRootTranslationGPU)) != 0;
    const float jacobianDifference = maxAbsoluteDifference(
        static_cast<const float*>(audit.candidateSamples[0].pointJacobians.contents),
        static_cast<const float*>(audit.candidateSamples[1].pointJacobians.contents),
        3u * dofCount);
    require(qDifference > 1.0e-7f || rootDifference ||
                jacobianDifference > 1.0e-7f,
            "nonzero generalized candidate produced no kinematic change");
    std::cout << "numanx exact candidate FK passed queries="
              << audit.candidateQueryCount << " q_delta=" << qDifference
              << " jacobian_delta=" << jacobianDifference << '\n';
}

void runSourceMass(const bool rollback) {
    SyntheticHumanFixture fixture;
    TransactionAudit audit = makeAudit(rollback);
    audit.candidateKinematics = true;
    audit.sourceMassOperations = true;
    const std::uint32_t dofCount = static_cast<std::uint32_t>(
        fixture.model.articulations[0].nv);
    require(dofCount > 0u, "source-M fixture has no generalized coordinates");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "source-M fixture requires a Metal device");
    const NSUInteger vectorBytes =
        static_cast<NSUInteger>(dofCount) * sizeof(float);
    audit.sourceMassInput = [device newBufferWithLength:vectorBytes
        options:MTLResourceStorageModeShared];
    audit.sourceMassAction = [device newBufferWithLength:vectorBytes
        options:MTLResourceStorageModeShared];
    audit.sourceMassSolve = [device newBufferWithLength:vectorBytes
        options:MTLResourceStorageModeShared];
    audit.sourceMassActionStatus = [device newBufferWithLength:sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
    audit.sourceMassSolveStatus = [device newBufferWithLength:sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
    require(audit.sourceMassInput != nil && audit.sourceMassAction != nil &&
                audit.sourceMassSolve != nil &&
                audit.sourceMassActionStatus != nil &&
                audit.sourceMassSolveStatus != nil,
            "source-M fixture could not allocate action/solve buffers");
    auto* input = static_cast<float*>(audit.sourceMassInput.contents);
    audit.sourceMassExpected.resize(dofCount);
    for (std::uint32_t dof = 0u; dof < dofCount; ++dof) {
        const float value = 0.25f + 0.125f * static_cast<float>(dof + 1u);
        input[dof] = value;
        audit.sourceMassExpected[dof] = value;
    }
    std::memset(audit.sourceMassAction.contents, 0xa5, vectorBytes);
    std::memset(audit.sourceMassSolve.contents, 0xa5, vectorBytes);
    std::memset(audit.sourceMassActionStatus.contents, 0xa5,
                sizeof(std::uint32_t));
    std::memset(audit.sourceMassSolveStatus.contents, 0xa5,
                sizeof(std::uint32_t));

    const auto originalQ = fixture.q;
    const auto originalV = fixture.v;
    const auto inputData = fixture.input(audit);
    auto context = makeContext();
    metalrobo::MetalArticulatedOperatorResult result;
    if (rollback) seedResult(result);
    const std::vector<std::byte> sentinel = resultBytes(result);
    const auto diagnostics = context.run(fixture.model, inputData, result);
    require(audit.failure == nullptr,
            audit.failure == nullptr ? "" : audit.failure);
    require(audit.sourceMassOperationsEncoded,
            "transaction did not exercise source-M action and solve");
    require(fixture.q == originalQ && fixture.v == originalV,
            "source-M operations modified accepted source q or v");
    if (rollback) {
        require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                    !diagnostics.published && audit.abortCount == 1u &&
                    audit.rejectionCount == 1u && resultBytes(result) == sentinel,
                "source-M rollback changed transaction publication state");
        std::cout << "numanx source physical mass rollback passed\n";
        return;
    }
    require(diagnostics.succeeded() && diagnostics.dispatched &&
                diagnostics.published && audit.abortCount == 0u,
            "source-M transaction did not complete: " + diagnostics.message);
    const auto actionStatus = *static_cast<const std::uint32_t*>(
        audit.sourceMassActionStatus.contents);
    const auto solveStatus = *static_cast<const std::uint32_t*>(
        audit.sourceMassSolveStatus.contents);
    require(actionStatus == MR_ARTICULATED_SOURCE_MASS_SUCCESS &&
                solveStatus == MR_ARTICULATED_SOURCE_MASS_SUCCESS,
            "source-M operation status rejected finite SPD fixture");
    const float error = maxAbsoluteDifference(
        static_cast<const float*>(audit.sourceMassSolve.contents),
        audit.sourceMassExpected.data(), dofCount);
    float scale = 0.0f;
    for (float value : audit.sourceMassExpected) scale = std::max(scale, std::abs(value));
    require(std::isfinite(error) && error <= 2.0e-4f * std::max(scale, 1.0e-6f),
            "M^-1(Mx) did not recover the source vector: " +
                std::to_string(error));
    std::cout << "numanx source physical mass action/solve passed dofs="
              << dofCount << " roundtrip_error=" << error << '\n';
}

void runABIContract() {
    metalrobo::MetalNumanXTransactionProgram empty;
    require(!empty.configured() && !empty.valid(),
            "default NumanX program is not an empty configuration");

    TransactionAudit audit = makeAudit(false);
    metalrobo::MetalNumanXTransactionProgram malformed{
        .context = &audit,
        .encode = &encodeTransaction,
        .abort = &abortTransaction,
        .fingerprint = kProgramFingerprint,
    };
    require(malformed.configured() && malformed.valid(),
            "well-formed NumanX program did not validate");
    malformed.abiVersion =
        metalrobo::kMetalNumanXTransactionSourceMassABIVersion + 1u;
    require(malformed.configured() && !malformed.valid(),
            "unknown NumanX program ABI did not fail closed");

    SyntheticHumanFixture fixture;
    auto input = fixture.input(audit);
    input.stand.numanXTransactionProgram = malformed;
    auto context = makeContext();
    metalrobo::MetalArticulatedOperatorResult result;
    seedResult(result);
    const std::vector<std::byte> sentinel = resultBytes(result);
    const auto diagnostics = context.run(fixture.model, input, result);
    require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                !diagnostics.published && audit.recordCount == 0u &&
                audit.abortCount == 0u && resultBytes(result) == sentinel,
            "unknown NumanX program ABI reached encoding or publication");

    malformed.abiVersion = metalrobo::kMetalNumanXTransactionABIVersion;
    malformed.structSize = sizeof(malformed) - 1u;
    require(malformed.configured() && !malformed.valid(),
            "truncated NumanX program ABI did not fail closed");
    std::cout << "numanx human transaction ABI contract passed version="
              << metalrobo::kMetalNumanXTransactionABIVersion
              << " pass_bytes=" << sizeof(Pass)
              << " program_bytes=" << sizeof(malformed) << '\n';
}

} // namespace

int main(const int argc, const char* const argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2,
                    "usage: metalrobo_numanx_transaction_probe --accept|--rollback|--abi-contract|--candidate-fk|--candidate-fk-rollback|--source-mass|--source-mass-rollback");
            const std::string mode = argv[1];
            if (mode == "--accept") {
                runAccept();
            } else if (mode == "--rollback") {
                runRollback();
            } else if (mode == "--abi-contract") {
                runABIContract();
            } else if (mode == "--candidate-fk") {
                runCandidateKinematics(false);
            } else if (mode == "--candidate-fk-rollback") {
                runCandidateKinematics(true);
            } else if (mode == "--source-mass") {
                runSourceMass(false);
            } else if (mode == "--source-mass-rollback") {
                runSourceMass(true);
            } else {
                throw std::runtime_error("unknown NumanX transaction probe mode");
            }
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "numanx transaction probe failed: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
