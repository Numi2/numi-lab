#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXCoupledHuman.hpp"
#include "NumanXProgramIdentity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace metalrobo {
namespace detail {

enum class MetalNumanXCoupledHumanHostStage : std::uint32_t {
    uninitialized = MR_NUMANX_COUPLED_HUMAN_STAGE_UNINITIALIZED,
    begun = MR_NUMANX_COUPLED_HUMAN_STAGE_BEGUN,
    candidateValidated =
        MR_NUMANX_COUPLED_HUMAN_STAGE_CANDIDATE_VALIDATED,
    published = MR_NUMANX_COUPLED_HUMAN_STAGE_PUBLISHED,
    resolved = MR_NUMANX_COUPLED_HUMAN_STAGE_RESOLVED,
};

struct MetalNumanXCoupledHumanSlot {
    id<MTLBuffer> reaction = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> metadata = nil;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t generation = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t reactionStride = 0u;
    MetalNumanXCoupledHumanHostStage stage =
        MetalNumanXCoupledHumanHostStage::uninitialized;
    bool bound = false;
    bool busy = false;
};

struct MetalNumanXCoupledHumanState {
    explicit MetalNumanXCoupledHumanState(
        MetalNumanXCoupledHumanConfig value
    ) : config(std::move(value)) {}

    mutable std::mutex mutex;
    MetalNumanXCoupledHumanConfig config;
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> beginPipeline = nil;
    id<MTLComputePipelineState> validateCandidatePipeline = nil;
    id<MTLComputePipelineState> massActionPipeline = nil;
    id<MTLComputePipelineState> inverseMassPipeline = nil;
    id<MTLComputePipelineState> publishPipeline = nil;
    id<MTLComputePipelineState> resolvePipeline = nil;
    id<MTLComputePipelineState> advancePipeline = nil;
    std::vector<MetalNumanXCoupledHumanSlot> slots;
    NumanXExecutableImageIdentity metallibIdentity{};
    std::uint64_t fingerprint = 0u;
    std::uint64_t retainedBytes = 0u;
    bool initialized = false;
};

} // namespace detail

namespace {

using State = detail::MetalNumanXCoupledHumanState;
using Slot = detail::MetalNumanXCoupledHumanSlot;
using HostStage = detail::MetalNumanXCoupledHumanHostStage;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kProgramCapabilities =
    MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION |
    MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER |
    MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH |
    MR_NUMANX_COUPLED_HUMAN_CAP_JOINT_DECISION;
constexpr std::uint32_t kConstraintFlags =
    MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
    MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES;
constexpr std::uint32_t kKnownStandFlags =
    MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
    MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE |
    MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS |
    MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES;

[[nodiscard]] bool checkedMultiply(
    const std::uint64_t a,
    const std::uint64_t b,
    std::uint64_t& result
) noexcept {
    if (a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

[[nodiscard]] bool checkedAdd(
    const std::uint64_t a,
    const std::uint64_t b,
    std::uint64_t& result
) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    result = a + b;
    return true;
}

[[nodiscard]] std::uint64_t hashValue(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (8u * byte)) & 0xffu;
        hash *= kFnvPrime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t hashString(
    std::uint64_t hash,
    const std::string& value
) noexcept {
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= kFnvPrime;
    }
    return hashValue(hash, value.size());
}

[[nodiscard]] std::uint64_t nonzeroHash(
    const std::uint64_t value
) noexcept {
    return value == 0u ? 0x9e3779b97f4a7c15ull : value;
}

[[nodiscard]] std::string fromNSString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) return {};
    return value.UTF8String;
}

[[nodiscard]] bool sameDevice(
    id<MTLDevice> first,
    id<MTLDevice> second
) noexcept {
    return first != nil && second != nil &&
        (first == second || first.registryID == second.registryID);
}

[[nodiscard]] MetalNumanXCoupledHumanDiagnostics diagnostics(
    const State* state,
    const MetalNumanXCoupledHumanHostStatus status,
    std::string message = {}
) {
    MetalNumanXCoupledHumanDiagnostics result{};
    result.status = status;
    result.message = std::move(message);
    if (state != nullptr) {
        result.programFingerprint = state->fingerprint;
        result.retainedBytes = state->retainedBytes;
        if (state->device != nil)
            result.deviceName = fromNSString(state->device.name);
    }
    return result;
}

[[nodiscard]] bool validCommandBuffer(
    const State& state,
    void* raw
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)raw;
    return commandBuffer != nil && commandBuffer.commandQueue != nil &&
        sameDevice(commandBuffer.commandQueue.device, state.device) &&
        commandBuffer.status == MTLCommandBufferStatusNotEnqueued;
}

[[nodiscard]] bool requiredBytes(
    const std::uint64_t elementCount,
    const std::uint64_t elementSize,
    std::uint64_t& bytes
) noexcept {
    return checkedMultiply(elementCount, elementSize, bytes) &&
        bytes <= std::numeric_limits<NSUInteger>::max();
}

[[nodiscard]] bool metalIndexedElements(
    const std::uint64_t elementCount
) noexcept {
    return elementCount != 0u &&
        elementCount <= std::numeric_limits<std::uint32_t>::max();
}

[[nodiscard]] bool metalIndexedProduct(
    const std::uint64_t first,
    const std::uint64_t second,
    std::uint64_t& elementCount
) noexcept {
    return checkedMultiply(first, second, elementCount) &&
        metalIndexedElements(elementCount);
}

[[nodiscard]] bool metalIndexedStridedLast(
    const std::uint64_t count,
    const std::uint64_t stride,
    std::uint64_t& elementCount
) noexcept {
    if (count == 0u || stride == 0u) return false;
    std::uint64_t lastBase = 0u;
    return checkedMultiply(count - 1u, stride, lastBase) &&
        checkedAdd(lastBase, 1u, elementCount) &&
        metalIndexedElements(elementCount);
}

[[nodiscard]] bool validBuffer(
    const State& state,
    void* raw,
    const std::uint64_t expectedGPUAddress,
    const std::uint64_t requiredByteCount
) noexcept {
    if (raw == nullptr || expectedGPUAddress == 0u ||
        requiredByteCount == 0u ||
        requiredByteCount > std::numeric_limits<NSUInteger>::max()) {
        return false;
    }
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)raw;
    return buffer != nil && sameDevice(buffer.device, state.device) &&
        static_cast<std::uint64_t>(buffer.length) >= requiredByteCount &&
        static_cast<std::uint64_t>(buffer.gpuAddress) ==
            expectedGPUAddress;
}

[[nodiscard]] bool emptyBuffer(
    void* raw,
    const std::uint64_t expectedGPUAddress
) noexcept {
    return raw == nullptr && expectedGPUAddress == 0u;
}

struct BufferRegion {
    std::uint64_t begin = 0u;
    std::uint64_t end = 0u;

    [[nodiscard]] bool active() const noexcept {
        return begin != 0u && end > begin;
    }
};

[[nodiscard]] bool makeBufferRegion(
    const State& state,
    void* raw,
    const std::uint64_t address,
    const std::uint64_t byteCount,
    BufferRegion& result
) noexcept {
    result = {};
    std::uint64_t end = 0u;
    if (!validBuffer(state, raw, address, byteCount) ||
        !checkedAdd(
            address,
            static_cast<std::uint64_t>(
                ((__bridge id<MTLBuffer>)raw).length),
            end) || end <= address) {
        return false;
    }
    result.begin = address;
    result.end = end;
    return true;
}

[[nodiscard]] bool makeOwnedBufferRegion(
    id<MTLBuffer> buffer,
    BufferRegion& result
) noexcept {
    result = {};
    if (buffer == nil || buffer.gpuAddress == 0u || buffer.length == 0u) {
        return false;
    }
    const std::uint64_t address = buffer.gpuAddress;
    std::uint64_t end = 0u;
    if (!checkedAdd(
            address,
            static_cast<std::uint64_t>(buffer.length),
            end) || end <= address) return false;
    result.begin = address;
    result.end = end;
    return true;
}

[[nodiscard]] bool regionsOverlap(
    const BufferRegion& first,
    const BufferRegion& second
) noexcept {
    return first.active() && second.active() &&
        first.begin < second.end && second.begin < first.end;
}

template <std::size_t Count>
[[nodiscard]] bool pairwiseDisjoint(
    const std::array<BufferRegion, Count>& regions,
    const std::size_t activeCount
) noexcept {
    if (activeCount > regions.size()) return false;
    for (std::size_t first = 0u; first < activeCount; ++first) {
        if (!regions[first].active()) return false;
        for (std::size_t second = first + 1u;
             second < activeCount;
             ++second) {
            if (regionsOverlap(regions[first], regions[second])) return false;
        }
    }
    return true;
}

[[nodiscard]] bool vectorByteCount(
    const MetalNumanXCoupledHumanPass& pass,
    const std::uint32_t stride,
    std::uint64_t& result
) noexcept {
    std::uint64_t elements = 0u;
    return metalIndexedProduct(pass.environmentCount, stride, elements) &&
        requiredBytes(elements, sizeof(float), result);
}

[[nodiscard]] bool queryRegionsIsolated(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const std::initializer_list<BufferRegion> queryRegions
) noexcept {
    if (pass.transactionSlot >= state.slots.size()) return false;
    std::array<
        BufferRegion,
        4u + 3u * MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS
    > protectedRegions{};
    std::size_t protectedCount = 0u;
    const auto appendExternal = [&] (
        void* raw,
        const std::uint64_t address,
        const std::uint64_t byteCount
    ) noexcept -> bool {
        if (protectedCount >= protectedRegions.size()) return false;
        return makeBufferRegion(
            state,
            raw,
            address,
            byteCount,
            protectedRegions[protectedCount++]);
    };
    const auto appendOwned = [&] (id<MTLBuffer> buffer) noexcept -> bool {
        if (protectedCount >= protectedRegions.size()) return false;
        return makeOwnedBufferRegion(
            buffer, protectedRegions[protectedCount++]);
    };
    std::uint64_t qBytes = 0u;
    std::uint64_t vBytes = 0u;
    std::uint64_t factorBytes = 0u;
    if (!requiredBytes(
            pass.sourceQElementCount, sizeof(float), qBytes) ||
        !requiredBytes(
            pass.sourceVElementCount, sizeof(float), vBytes) ||
        !requiredBytes(
            pass.sourceEffectiveTangentFactorElementCount,
            sizeof(float),
            factorBytes) ||
        !appendExternal(
            pass.sourceQ, pass.sourceQGPUAddress, qBytes) ||
        !appendExternal(
            pass.sourceV, pass.sourceVGPUAddress, vBytes) ||
        !appendExternal(
            pass.sourceEffectiveTangentFactor,
            pass.sourceEffectiveTangentFactorGPUAddress,
            factorBytes)) {
        return false;
    }
    if (pass.tangentProjector != nullptr) {
        std::uint64_t projectorBytes = 0u;
        if (!requiredBytes(
                pass.tangentProjectorElementCount,
                sizeof(float),
                projectorBytes) ||
            !appendExternal(
                pass.tangentProjector,
                pass.tangentProjectorGPUAddress,
                projectorBytes)) return false;
    }
    for (const Slot& slot : state.slots) {
        if (!appendOwned(slot.reaction) || !appendOwned(slot.statuses) ||
            !appendOwned(slot.metadata)) return false;
    }
    if (!pairwiseDisjoint(protectedRegions, protectedCount)) return false;

    std::array<BufferRegion, 8u> externalRegions{};
    if (queryRegions.size() > externalRegions.size()) return false;
    std::size_t externalCount = 0u;
    for (const BufferRegion& region : queryRegions) {
        if (!region.active()) return false;
        externalRegions[externalCount++] = region;
    }
    if (!pairwiseDisjoint(externalRegions, externalCount)) return false;
    for (std::size_t external = 0u;
         external < externalCount;
         ++external) {
        for (std::size_t protectedIndex = 0u;
             protectedIndex < protectedCount;
             ++protectedIndex) {
            if (regionsOverlap(
                    externalRegions[external],
                    protectedRegions[protectedIndex])) return false;
        }
    }
    return true;
}

[[nodiscard]] bool validOperation(
    const MetalNumanXCoupledHumanOperation operation
) noexcept {
    switch (operation) {
    case MetalNumanXCoupledHumanOperation::candidateKinematics:
    case MetalNumanXCoupledHumanOperation::massAction:
    case MetalNumanXCoupledHumanOperation::inverseMassPreconditioner:
    case MetalNumanXCoupledHumanOperation::publishCandidate:
        return true;
    }
    return false;
}

[[nodiscard]] bool validPassHeader(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanPhase expectedPhase
) noexcept {
    if (!state.initialized ||
        pass.abiVersion != MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
        pass.structSize != sizeof(MetalNumanXCoupledHumanPass) ||
        (pass.accessFlags & ~kMetalNumanXCoupledHumanKnownAccess) != 0u ||
        (pass.capabilities &
         ~MR_NUMANX_COUPLED_HUMAN_KNOWN_CAPABILITIES) != 0u ||
        pass.phase != expectedPhase || pass.reserved0 != 0u ||
        pass.programFingerprint != state.fingerprint ||
        pass.transactionFingerprint == 0u ||
        pass.linearizationEpoch == 0u ||
        pass.slotGeneration == 0u ||
        pass.transactionSlot >= state.config.transactionSlotCount ||
        pass.environmentCount == 0u ||
        pass.environmentCount > state.config.environmentCapacity ||
        pass.dofCount == 0u ||
        pass.dofCount > MR_NUMANX_COUPLED_HUMAN_MAX_DOFS ||
        pass.qCoordinateCount != pass.dofCount + 1u ||
        pass.qCoordinateCount > MR_NUMANX_COUPLED_HUMAN_MAX_Q ||
        pass.bodyCount == 0u ||
        pass.bodyCount > MR_NUMI_HUMAN_STAND_MAX_BODIES ||
        pass.qStride != pass.qCoordinateCount ||
        pass.vStride != pass.dofCount ||
        pass.factorStride != pass.dofCount * pass.dofCount ||
        pass.stepCount == 0u || pass.stepIndex >= pass.stepCount ||
        (pass.standFlags & ~kKnownStandFlags) != 0u ||
        !(pass.timestepSeconds > 0.0f) ||
        !std::isfinite(pass.timestepSeconds) ||
        !validCommandBuffer(state, pass.commandBuffer)) {
        return false;
    }

    std::uint64_t expectedQ = 0u;
    std::uint64_t expectedV = 0u;
    std::uint64_t reactionElements = 0u;
    std::uint64_t qBytes = 0u;
    std::uint64_t vBytes = 0u;
    if (!metalIndexedProduct(
            pass.environmentCount, pass.qStride, expectedQ) ||
        !metalIndexedProduct(
            pass.environmentCount, pass.vStride, expectedV) ||
        !metalIndexedProduct(
            pass.environmentCount,
            MR_NUMANX_COUPLED_HUMAN_MAX_DOFS,
            reactionElements) ||
        pass.sourceQElementCount != expectedQ ||
        pass.sourceVElementCount != expectedV ||
        !requiredBytes(expectedQ, sizeof(float), qBytes) ||
        !requiredBytes(expectedV, sizeof(float), vBytes) ||
        !validBuffer(
            state, pass.sourceQ, pass.sourceQGPUAddress, qBytes) ||
        !validBuffer(
            state, pass.sourceV, pass.sourceVGPUAddress, vBytes)) {
        return false;
    }

    const bool constrained = (pass.standFlags & kConstraintFlags) != 0u;
    const bool hasProjector =
        (pass.capabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_TANGENT_PROJECTOR) != 0u;
    // A full-space P does not make P^T*A0*P invertible and its projected
    // preconditioner is not that operator's inverse for general A0/P. Until a
    // nullspace or KKT ABI exists, every Stand-constrained pass fails closed.
    if (constrained) return false;
    if (hasProjector) {
        std::uint64_t projectorElements = 0u;
        std::uint64_t projectorBytes = 0u;
        if ((pass.accessFlags &
             MetalNumanXCoupledHumanReadTangentProjector) == 0u ||
            pass.projectorStride != pass.dofCount * pass.dofCount ||
            !metalIndexedProduct(
                pass.environmentCount,
                pass.projectorStride,
                projectorElements) ||
            pass.tangentProjectorElementCount != projectorElements ||
            !requiredBytes(
                projectorElements, sizeof(float), projectorBytes) ||
            !validBuffer(
                state,
                pass.tangentProjector,
                pass.tangentProjectorGPUAddress,
                projectorBytes)) {
            return false;
        }
    } else if (pass.projectorStride != 0u ||
               pass.tangentProjectorElementCount != 0u ||
               !emptyBuffer(
                   pass.tangentProjector,
                   pass.tangentProjectorGPUAddress)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool validFactor(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass
) noexcept {
    std::uint64_t elements = 0u;
    std::uint64_t bytes = 0u;
    return (pass.accessFlags &
            MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor) != 0u &&
        metalIndexedProduct(
            pass.environmentCount, pass.factorStride, elements) &&
        pass.sourceEffectiveTangentFactorElementCount == elements &&
        requiredBytes(elements, sizeof(float), bytes) &&
        validBuffer(
            state,
            pass.sourceEffectiveTangentFactor,
            pass.sourceEffectiveTangentFactorGPUAddress,
            bytes);
}

[[nodiscard]] bool validStandStatuses(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass
) noexcept {
    if ((pass.accessFlags &
         MetalNumanXCoupledHumanReadStandStatus) == 0u ||
        pass.standStatusStride == 0u) return false;
    std::uint64_t requiredElements = 0u;
    std::uint64_t bytes = 0u;
    if (!metalIndexedStridedLast(
            pass.environmentCount,
            pass.standStatusStride,
            requiredElements) ||
        pass.standStatusElementCount < requiredElements ||
        !requiredBytes(
            requiredElements,
            sizeof(MRNumiHumanStandStatusGPU),
            bytes)) return false;
    return validBuffer(
        state,
        pass.standStatuses,
        pass.standStatusesGPUAddress,
        bytes
    );
}

[[nodiscard]] bool sameTransaction(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    return query.abiVersion == MR_NUMANX_COUPLED_HUMAN_ABI_VERSION &&
        query.structSize == sizeof(MetalNumanXCoupledHumanQuery) &&
        (query.accessFlags &
         ~kMetalNumanXCoupledHumanKnownQueryAccess) == 0u &&
        (query.requiredCapabilities &
         ~MR_NUMANX_COUPLED_HUMAN_KNOWN_CAPABILITIES) == 0u &&
        (query.requiredCapabilities &
         ~(kProgramCapabilities | pass.capabilities)) == 0u &&
        query.programFingerprint == state.fingerprint &&
        query.transactionFingerprint == pass.transactionFingerprint &&
        query.linearizationEpoch == pass.linearizationEpoch &&
        query.slotGeneration == pass.slotGeneration &&
        query.transactionSlot == pass.transactionSlot &&
        validOperation(query.operation);
}

[[nodiscard]] bool slotMatches(
    const Slot& slot,
    const MetalNumanXCoupledHumanPass& pass
) noexcept {
    return slot.bound &&
        slot.transactionFingerprint == pass.transactionFingerprint &&
        slot.linearizationEpoch == pass.linearizationEpoch &&
        slot.generation == pass.slotGeneration &&
        slot.stepIndex == pass.stepIndex &&
        slot.environmentCount == pass.environmentCount &&
        slot.qCoordinateCount == pass.qCoordinateCount &&
        slot.dofCount == pass.dofCount &&
        slot.reactionStride == MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
}

[[nodiscard]] bool validVectorBuffer(
    const State& state,
    void* raw,
    const std::uint64_t address,
    const MetalNumanXCoupledHumanPass& pass,
    const std::uint32_t stride
) noexcept {
    std::uint64_t elements = 0u;
    std::uint64_t bytes = 0u;
    return stride >= pass.dofCount &&
        stride <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        metalIndexedProduct(pass.environmentCount, stride, elements) &&
        requiredBytes(elements, sizeof(float), bytes) &&
        validBuffer(state, raw, address, bytes);
}

[[nodiscard]] bool distinct(
    void* first,
    void* second
) noexcept {
    return first == nullptr || second == nullptr || first != second;
}

[[nodiscard]] bool validCandidateQuery(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    const std::uint32_t requiredAccess =
        MetalNumanXCoupledHumanQueryReadInput |
        MetalNumanXCoupledHumanQueryWriteCandidateQ |
        MetalNumanXCoupledHumanQueryWriteCandidateBodies |
        (query.pointCount == 0u
             ? 0u
             : MetalNumanXCoupledHumanQueryReadPointQueries |
                 MetalNumanXCoupledHumanQueryWritePointJacobians);
    if (query.accessFlags != requiredAccess ||
        (query.requiredCapabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS) == 0u ||
        (pass.capabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS) == 0u ||
        (pass.accessFlags &
         (MetalNumanXCoupledHumanReadSourceState |
          MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor |
          MetalNumanXCoupledHumanEncodeExactCandidate)) !=
            (MetalNumanXCoupledHumanReadSourceState |
             MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor |
             MetalNumanXCoupledHumanEncodeExactCandidate) ||
        pass.exactKinematicsContext == nullptr ||
        pass.encodeExactKinematics == nullptr ||
        !validFactor(state, pass) ||
        !validVectorBuffer(
            state,
            query.input,
            query.inputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        query.output != nullptr || query.outputGPUAddress != 0u ||
        query.statuses != nullptr || query.statusesGPUAddress != 0u ||
        query.matterOutcomes != nullptr ||
        query.matterOutcomeGPUAddress != 0u ||
        query.matterOutcomeStride != 0u ||
        query.expectedMatterCompletedMicrosteps != 0u ||
        query.candidateQStride < pass.qCoordinateCount ||
        query.candidateQStride > MR_NUMANX_COUPLED_HUMAN_MAX_Q ||
        query.candidateBodyStride < pass.bodyCount ||
        query.pointCount > state.config.pointCapacity ||
        query.pointCount > MR_NUMANX_COUPLED_HUMAN_MAX_POINTS ||
        !distinct(query.input, query.candidateQ) ||
        !distinct(query.input, query.candidateBodies) ||
        !distinct(query.candidateQ, query.candidateBodies) ||
        !distinct(query.candidateQ, pass.sourceQ) ||
        !distinct(
            query.candidateBodies,
            pass.sourceEffectiveTangentFactor)) {
        return false;
    }

    std::uint64_t qElements = 0u;
    std::uint64_t qBytes = 0u;
    std::uint64_t bodyElements = 0u;
    std::uint64_t bodyBytes = 0u;
    std::uint64_t inputBytes = 0u;
    BufferRegion inputRegion{};
    BufferRegion qRegion{};
    BufferRegion bodyRegion{};
    if (!metalIndexedProduct(
            pass.environmentCount,
            query.candidateQStride,
            qElements) ||
        !requiredBytes(qElements, sizeof(float), qBytes) ||
        !validBuffer(
            state,
            query.candidateQ,
            query.candidateQGPUAddress,
            qBytes) ||
        !metalIndexedProduct(
            pass.environmentCount,
            query.candidateBodyStride,
            bodyElements) ||
        !requiredBytes(
            bodyElements, sizeof(MRBodyStateGPU), bodyBytes) ||
        !validBuffer(
            state,
            query.candidateBodies,
            query.candidateBodiesGPUAddress,
            bodyBytes) ||
        !vectorByteCount(
            pass, query.generalizedVectorStride, inputBytes) ||
        !makeBufferRegion(
            state,
            query.input,
            query.inputGPUAddress,
            inputBytes,
            inputRegion) ||
        !makeBufferRegion(
            state,
            query.candidateQ,
            query.candidateQGPUAddress,
            qBytes,
            qRegion) ||
        !makeBufferRegion(
            state,
            query.candidateBodies,
            query.candidateBodiesGPUAddress,
            bodyBytes,
            bodyRegion)) {
        return false;
    }

    if (query.pointCount == 0u) {
        return query.pointStride == 0u &&
            query.pointJacobianStride == 0u &&
            emptyBuffer(query.pointQueries, query.pointQueriesGPUAddress) &&
            emptyBuffer(
                query.pointJacobians,
                query.pointJacobiansGPUAddress) &&
            queryRegionsIsolated(
                state, pass, {inputRegion, qRegion, bodyRegion});
    }
    std::uint64_t pointJacobianMinimum = 0u;
    if (query.pointStride < query.pointCount ||
        !checkedMultiply(
            query.pointCount, 3u, pointJacobianMinimum) ||
        !checkedMultiply(
            pointJacobianMinimum,
            query.generalizedVectorStride,
            pointJacobianMinimum) ||
        pointJacobianMinimum >
            std::numeric_limits<std::uint32_t>::max() ||
        query.pointJacobianStride < pointJacobianMinimum) {
        return false;
    }
    std::uint64_t pointElements = 0u;
    std::uint64_t pointBytes = 0u;
    std::uint64_t jacobianElements = 0u;
    std::uint64_t jacobianBytes = 0u;
    BufferRegion pointRegion{};
    BufferRegion jacobianRegion{};
    return metalIndexedProduct(
            pass.environmentCount,
            query.pointStride,
            pointElements) &&
        requiredBytes(
            pointElements,
            sizeof(MRArticulatedPointImpulseGPU),
            pointBytes) &&
        validBuffer(
            state,
            query.pointQueries,
            query.pointQueriesGPUAddress,
            pointBytes) &&
        metalIndexedProduct(
            pass.environmentCount,
            query.pointJacobianStride,
            jacobianElements) &&
        requiredBytes(
            jacobianElements,
            sizeof(float),
            jacobianBytes) &&
        validBuffer(
            state,
            query.pointJacobians,
            query.pointJacobiansGPUAddress,
            jacobianBytes) &&
        makeBufferRegion(
            state,
            query.pointQueries,
            query.pointQueriesGPUAddress,
            pointBytes,
            pointRegion) &&
        makeBufferRegion(
            state,
            query.pointJacobians,
            query.pointJacobiansGPUAddress,
            jacobianBytes,
            jacobianRegion) &&
        queryRegionsIsolated(
            state,
            pass,
            {inputRegion,
             qRegion,
             bodyRegion,
             pointRegion,
             jacobianRegion});
}

[[nodiscard]] bool validMassQuery(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    if (query.accessFlags !=
            (MetalNumanXCoupledHumanQueryReadInput |
             MetalNumanXCoupledHumanQueryWriteOutput) ||
        (query.requiredCapabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION) == 0u ||
        !validFactor(state, pass) ||
        !validVectorBuffer(
            state,
            query.input,
            query.inputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        !validVectorBuffer(
            state,
            query.output,
            query.outputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        query.input == query.output ||
        query.candidateBodies != nullptr ||
        query.statuses != nullptr || query.pointQueries != nullptr ||
        query.pointJacobians != nullptr || query.matterOutcomes != nullptr ||
        query.candidateBodiesGPUAddress != 0u ||
        query.statusesGPUAddress != 0u ||
        query.pointQueriesGPUAddress != 0u ||
        query.pointJacobiansGPUAddress != 0u ||
        query.matterOutcomeGPUAddress != 0u ||
        query.matterOutcomeStride != 0u ||
        query.expectedMatterCompletedMicrosteps != 0u ||
        query.candidateBodyStride != 0u || query.statusStride != 0u ||
        query.pointCount != 0u || query.pointStride != 0u ||
        query.pointJacobianStride != 0u) {
        return false;
    }
    std::uint64_t vectorBytes = 0u;
    BufferRegion inputRegion{};
    BufferRegion outputRegion{};
    if (!vectorByteCount(
            pass, query.generalizedVectorStride, vectorBytes) ||
        !makeBufferRegion(
            state,
            query.input,
            query.inputGPUAddress,
            vectorBytes,
            inputRegion) ||
        !makeBufferRegion(
            state,
            query.output,
            query.outputGPUAddress,
            vectorBytes,
            outputRegion)) {
        return false;
    }
    if (query.candidateQ == nullptr) {
        return query.candidateQGPUAddress == 0u &&
            query.candidateQStride == 0u &&
            queryRegionsIsolated(
                state, pass, {inputRegion, outputRegion});
    }
    std::uint64_t elements = 0u;
    std::uint64_t bytes = 0u;
    BufferRegion qRegion{};
    return query.candidateQStride >= pass.qCoordinateCount &&
        query.candidateQStride <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        metalIndexedProduct(
            pass.environmentCount,
            query.candidateQStride,
            elements) &&
        requiredBytes(elements, sizeof(float), bytes) &&
        validBuffer(
            state,
            query.candidateQ,
            query.candidateQGPUAddress,
            bytes) &&
        makeBufferRegion(
            state,
            query.candidateQ,
            query.candidateQGPUAddress,
            bytes,
            qRegion) &&
        queryRegionsIsolated(
            state, pass, {inputRegion, outputRegion, qRegion});
}

[[nodiscard]] bool validInverseQuery(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    if (query.accessFlags !=
            (MetalNumanXCoupledHumanQueryReadInput |
             MetalNumanXCoupledHumanQueryWriteOutput |
             MetalNumanXCoupledHumanQueryWriteInverseStatus) ||
        (query.requiredCapabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER) == 0u ||
        !validFactor(state, pass) ||
        !validVectorBuffer(
            state,
            query.input,
            query.inputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        !validVectorBuffer(
            state,
            query.output,
            query.outputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        query.input == query.output || query.statusStride < pass.environmentCount ||
        query.candidateQ != nullptr || query.candidateBodies != nullptr ||
        query.pointQueries != nullptr || query.pointJacobians != nullptr ||
        query.matterOutcomes != nullptr ||
        query.candidateQGPUAddress != 0u ||
        query.candidateBodiesGPUAddress != 0u ||
        query.pointQueriesGPUAddress != 0u ||
        query.pointJacobiansGPUAddress != 0u ||
        query.matterOutcomeGPUAddress != 0u ||
        query.matterOutcomeStride != 0u ||
        query.expectedMatterCompletedMicrosteps != 0u ||
        query.candidateQStride != 0u ||
        query.candidateBodyStride != 0u ||
        query.pointCount != 0u || query.pointStride != 0u ||
        query.pointJacobianStride != 0u) {
        return false;
    }
    std::uint64_t statusBase = 0u;
    std::uint64_t requiredElements = 0u;
    std::uint64_t bytes = 0u;
    std::uint64_t vectorBytes = 0u;
    BufferRegion inputRegion{};
    BufferRegion outputRegion{};
    BufferRegion statusRegion{};
    return checkedMultiply(
            pass.articulationIndex,
            query.statusStride,
            statusBase) &&
        checkedAdd(
            statusBase,
            pass.environmentCount,
            requiredElements) &&
        requiredElements <= std::numeric_limits<std::uint32_t>::max() &&
        requiredBytes(
            requiredElements,
            sizeof(MRInverseMassStatusGPU),
            bytes) &&
        validBuffer(
            state,
            query.statuses,
            query.statusesGPUAddress,
            bytes) &&
        vectorByteCount(
            pass, query.generalizedVectorStride, vectorBytes) &&
        makeBufferRegion(
            state,
            query.input,
            query.inputGPUAddress,
            vectorBytes,
            inputRegion) &&
        makeBufferRegion(
            state,
            query.output,
            query.outputGPUAddress,
            vectorBytes,
            outputRegion) &&
        makeBufferRegion(
            state,
            query.statuses,
            query.statusesGPUAddress,
            bytes,
            statusRegion) &&
        queryRegionsIsolated(
            state, pass, {inputRegion, outputRegion, statusRegion});
}

[[nodiscard]] bool validPublishQuery(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    if (query.accessFlags !=
            (MetalNumanXCoupledHumanQueryReadInput |
             MetalNumanXCoupledHumanQueryWriteOutput |
             MetalNumanXCoupledHumanQueryReadMatterOutcome) ||
        (query.requiredCapabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH) == 0u ||
        !validFactor(state, pass) ||
        !validVectorBuffer(
            state,
            query.input,
            query.inputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        !validVectorBuffer(
            state,
            query.output,
            query.outputGPUAddress,
            pass,
            query.generalizedVectorStride) ||
        query.input == query.output || query.matterOutcomeStride == 0u ||
        query.expectedMatterCompletedMicrosteps != 0u ||
        query.candidateBodies != nullptr ||
        query.statuses != nullptr || query.pointQueries != nullptr ||
        query.pointJacobians != nullptr ||
        query.candidateBodiesGPUAddress != 0u ||
        query.statusesGPUAddress != 0u ||
        query.pointQueriesGPUAddress != 0u ||
        query.pointJacobiansGPUAddress != 0u ||
        query.candidateBodyStride != 0u || query.statusStride != 0u ||
        query.pointCount != 0u || query.pointStride != 0u ||
        query.pointJacobianStride != 0u) {
        return false;
    }
    std::uint64_t vectorBytes = 0u;
    BufferRegion inputRegion{};
    BufferRegion outputRegion{};
    BufferRegion qRegion{};
    if (!vectorByteCount(
            pass, query.generalizedVectorStride, vectorBytes) ||
        !makeBufferRegion(
            state,
            query.input,
            query.inputGPUAddress,
            vectorBytes,
            inputRegion) ||
        !makeBufferRegion(
            state,
            query.output,
            query.outputGPUAddress,
            vectorBytes,
            outputRegion)) {
        return false;
    }
    const bool hasCandidateQ = query.candidateQ != nullptr;
    if (query.candidateQ == nullptr) {
        if (query.candidateQGPUAddress != 0u ||
            query.candidateQStride != 0u) return false;
    } else {
        std::uint64_t candidateQElements = 0u;
        std::uint64_t candidateQBytes = 0u;
        if (query.candidateQStride < pass.qCoordinateCount ||
            query.candidateQStride > MR_NUMANX_COUPLED_HUMAN_MAX_Q ||
            !metalIndexedProduct(
                pass.environmentCount,
                query.candidateQStride,
                candidateQElements) ||
            !requiredBytes(
                candidateQElements, sizeof(float), candidateQBytes) ||
            !validBuffer(
                state,
                query.candidateQ,
                query.candidateQGPUAddress,
                candidateQBytes) ||
            !makeBufferRegion(
                state,
                query.candidateQ,
                query.candidateQGPUAddress,
                candidateQBytes,
                qRegion)) {
            return false;
        }
    }
    std::uint64_t outcomeElements = 0u;
    std::uint64_t outcomeBytes = 0u;
    BufferRegion outcomeRegion{};
    if (!metalIndexedStridedLast(
            pass.environmentCount,
            query.matterOutcomeStride,
            outcomeElements) ||
        !requiredBytes(
            outcomeElements,
            sizeof(MRNumanXCoupledMatterOutcomeGPU),
            outcomeBytes) ||
        !makeBufferRegion(
            state,
            query.matterOutcomes,
            query.matterOutcomeGPUAddress,
            outcomeBytes,
            outcomeRegion)) {
        return false;
    }
    return hasCandidateQ
        ? queryRegionsIsolated(
            state,
            pass,
            {inputRegion, outputRegion, qRegion, outcomeRegion})
        : queryRegionsIsolated(
            state, pass, {inputRegion, outputRegion, outcomeRegion});
}

[[nodiscard]] MRNumanXCoupledHumanDispatchGPU makeDispatch(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanOperation operation,
    const MetalNumanXCoupledHumanQuery* query
) noexcept {
    MRNumanXCoupledHumanDispatchGPU dispatch{};
    dispatch.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    dispatch.operation = static_cast<std::uint32_t>(operation);
    dispatch.environmentCount = pass.environmentCount;
    dispatch.flags =
        (pass.capabilities &
         MR_NUMANX_COUPLED_HUMAN_CAP_TANGENT_PROJECTOR) != 0u
        ? MR_NUMANX_COUPLED_HUMAN_DISPATCH_USE_PROJECTOR
        : 0u;
    dispatch.nq = pass.qCoordinateCount;
    dispatch.nv = pass.dofCount;
    dispatch.sourceQStride = pass.qStride;
    dispatch.sourceVStride = pass.vStride;
    dispatch.generalizedVectorStride = query != nullptr
        ? query->generalizedVectorStride
        : pass.dofCount;
    dispatch.candidateQStride = query != nullptr
        ? query->candidateQStride
        : pass.qCoordinateCount;
    dispatch.candidateBodyStride = query != nullptr
        ? query->candidateBodyStride
        : pass.bodyCount;
    dispatch.statusStride = query != nullptr
        ? query->statusStride
        : 1u;
    dispatch.pointCount = query != nullptr ? query->pointCount : 0u;
    dispatch.pointStride = query != nullptr ? query->pointStride : 0u;
    dispatch.pointJacobianStride = query != nullptr
        ? query->pointJacobianStride
        : 0u;
    dispatch.factorStride = pass.factorStride;
    dispatch.projectorStride = pass.projectorStride;
    dispatch.reactionStride = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
    dispatch.articulationIndex = pass.articulationIndex;
    dispatch.transactionSlot = pass.transactionSlot;
    dispatch.stepIndex = pass.stepIndex;
    dispatch.bodyCount = pass.bodyCount;
    dispatch.standFlags = pass.standFlags;
    dispatch.timestepAndInverse = {
        pass.timestepSeconds,
        1.0f / pass.timestepSeconds,
        0.0f,
        0.0f,
    };
    dispatch.programFingerprint = state.fingerprint;
    dispatch.transactionFingerprint = pass.transactionFingerprint;
    dispatch.linearizationEpoch = pass.linearizationEpoch;
    dispatch.slotGeneration = pass.slotGeneration;
    return dispatch;
}

[[nodiscard]] MRNumanXCoupledHumanAdvanceDispatchGPU makeAdvanceDispatch(
    const State& state,
    const MetalNumanXCoupledHumanPass& pass,
    const HostStage expectedStage,
    const HostStage nextStage,
    const std::uint32_t witness,
    const std::uint32_t flags
) noexcept {
    MRNumanXCoupledHumanAdvanceDispatchGPU dispatch{};
    dispatch.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
    dispatch.expectedStage = static_cast<std::uint32_t>(expectedStage);
    dispatch.nextStage = static_cast<std::uint32_t>(nextStage);
    dispatch.witness = witness;
    dispatch.flags = flags;
    dispatch.environmentCount = pass.environmentCount;
    dispatch.transactionSlot = pass.transactionSlot;
    dispatch.stepIndex = pass.stepIndex;
    dispatch.programFingerprint = state.fingerprint;
    dispatch.transactionFingerprint = pass.transactionFingerprint;
    dispatch.linearizationEpoch = pass.linearizationEpoch;
    dispatch.slotGeneration = pass.slotGeneration;
    dispatch.qCoordinateCount = pass.qCoordinateCount;
    dispatch.dofCount = pass.dofCount;
    dispatch.reactionStride = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
    return dispatch;
}

[[nodiscard]] bool encodeAdvance(
    State& state,
    Slot& slot,
    id<MTLCommandBuffer> commandBuffer,
    const MetalNumanXCoupledHumanPass& pass,
    const HostStage expectedStage,
    const HostStage nextStage,
    const std::uint32_t witness,
    const std::uint32_t flags,
    NSString* label
) noexcept {
    const MRNumanXCoupledHumanAdvanceDispatchGPU dispatch =
        makeAdvanceDispatch(
            state, pass, expectedStage, nextStage, witness, flags
        );
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) return false;
    encoder.label = label;
    [encoder setComputePipelineState:state.advancePipeline];
    [encoder setBuffer:slot.metadata offset:0u atIndex:0u];
    [encoder setBuffer:slot.statuses offset:0u atIndex:1u];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:2u];
    [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [encoder endEncoding];
    return true;
}

void dispatchEnvironments(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const std::uint32_t environmentCount
) {
    const NSUInteger width = std::min<NSUInteger>(
        std::max<NSUInteger>(environmentCount, 1u),
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 64u)
    );
    [encoder dispatchThreads:MTLSizeMake(environmentCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

[[nodiscard]] bool beginCallback(
    void* opaque,
    const MetalNumanXCoupledHumanPass& pass
) noexcept {
    @autoreleasepool {
        auto* state = static_cast<State*>(opaque);
        if (state == nullptr ||
            !validPassHeader(
                *state, pass, MetalNumanXCoupledHumanPhase::preDynamics) ||
            !validFactor(*state, pass) ||
            !queryRegionsIsolated(*state, pass, {}) ||
            (pass.accessFlags &
             (MetalNumanXCoupledHumanWriteReaction |
              MetalNumanXCoupledHumanWriteJointStatus)) !=
                (MetalNumanXCoupledHumanWriteReaction |
                 MetalNumanXCoupledHumanWriteJointStatus)) {
            return false;
        }
        std::lock_guard lock(state->mutex);
        Slot& slot = state->slots[pass.transactionSlot];
        if (slot.busy ||
            (slot.bound && pass.slotGeneration <= slot.generation)) {
            return false;
        }
        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        const MRNumanXCoupledHumanDispatchGPU dispatch = makeDispatch(
            *state,
            pass,
            MetalNumanXCoupledHumanOperation::candidateKinematics,
            nullptr
        );
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        encoder.label = @"NumanX coupled Human begin";
        [encoder setComputePipelineState:state->beginPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBuffer:slot.reaction offset:0u atIndex:1u];
        [encoder setBuffer:slot.statuses offset:0u atIndex:2u];
        [encoder setBuffer:slot.metadata offset:0u atIndex:3u];
        dispatchEnvironments(
            encoder, state->beginPipeline, pass.environmentCount
        );
        [encoder endEncoding];
        slot.transactionFingerprint = pass.transactionFingerprint;
        slot.linearizationEpoch = pass.linearizationEpoch;
        slot.generation = pass.slotGeneration;
        slot.stepIndex = pass.stepIndex;
        slot.environmentCount = pass.environmentCount;
        slot.qCoordinateCount = pass.qCoordinateCount;
        slot.dofCount = pass.dofCount;
        slot.reactionStride = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
        slot.stage = HostStage::begun;
        slot.bound = true;
        return true;
    }
}

[[nodiscard]] bool encodeCallback(
    void* opaque,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    @autoreleasepool {
        auto* state = static_cast<State*>(opaque);
        if (state == nullptr ||
            !validPassHeader(
                *state, pass, MetalNumanXCoupledHumanPhase::preDynamics) ||
            !sameTransaction(*state, pass, query) ||
            (pass.accessFlags &
             (MetalNumanXCoupledHumanWriteReaction |
              MetalNumanXCoupledHumanWriteJointStatus)) !=
                (MetalNumanXCoupledHumanWriteReaction |
                 MetalNumanXCoupledHumanWriteJointStatus)) {
            return false;
        }
        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        __unsafe_unretained id<MTLBuffer> factor =
            (__bridge id<MTLBuffer>)pass.sourceEffectiveTangentFactor;
        __unsafe_unretained id<MTLBuffer> projector =
            pass.tangentProjector != nullptr
                ? (__bridge id<MTLBuffer>)pass.tangentProjector
                : factor;
        const MRNumanXCoupledHumanDispatchGPU dispatch = makeDispatch(
            *state, pass, query.operation, &query
        );

        if (query.operation ==
            MetalNumanXCoupledHumanOperation::candidateKinematics) {
            if (!validCandidateQuery(*state, pass, query)) return false;
            HostStage expectedStage = HostStage::uninitialized;
            {
                std::lock_guard lock(state->mutex);
                Slot& slot = state->slots[pass.transactionSlot];
                if (!slotMatches(slot, pass) || slot.busy ||
                    (slot.stage != HostStage::begun &&
                     slot.stage != HostStage::candidateValidated)) {
                    return false;
                }
                expectedStage = slot.stage;
                slot.busy = true;
            }

            const auto releaseBusy = [&] () noexcept {
                std::lock_guard lock(state->mutex);
                Slot& slot = state->slots[pass.transactionSlot];
                if (slotMatches(slot, pass)) slot.busy = false;
            };
            if (!pass.encodeExactKinematics(
                    pass.exactKinematicsContext, pass, query)) {
                releaseBusy();
                return false;
            }

            Slot& slot = state->slots[pass.transactionSlot];
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                releaseBusy();
                return false;
            }
            encoder.label = @"NumanX coupled Human candidate validation";
            [encoder setComputePipelineState:
                state->validateCandidatePipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.candidateQ
                         offset:0u atIndex:1u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.candidateBodies
                         offset:0u atIndex:2u];
            [encoder setBuffer:query.pointCount == 0u
                    ? (__bridge id<MTLBuffer>)query.candidateQ
                    : (__bridge id<MTLBuffer>)query.pointJacobians
                         offset:0u atIndex:3u];
            [encoder setBuffer:slot.statuses offset:0u atIndex:4u];
            [encoder setBuffer:slot.metadata offset:0u atIndex:5u];
            dispatchEnvironments(
                encoder,
                state->validateCandidatePipeline,
                pass.environmentCount
            );
            [encoder endEncoding];
            if (!encodeAdvance(
                    *state,
                    slot,
                    commandBuffer,
                    pass,
                    expectedStage,
                    HostStage::candidateValidated,
                    MR_NUMANX_COUPLED_HUMAN_WITNESS_CANDIDATE,
                    MR_NUMANX_COUPLED_HUMAN_ADVANCE_REQUIRE_PENDING,
                    @"NumanX coupled Human candidate witness")) {
                releaseBusy();
                return false;
            }
            {
                std::lock_guard lock(state->mutex);
                Slot& current = state->slots[pass.transactionSlot];
                if (!slotMatches(current, pass) || !current.busy ||
                    current.stage != expectedStage) {
                    if (slotMatches(current, pass)) current.busy = false;
                    return false;
                }
                current.stage = HostStage::candidateValidated;
                current.busy = false;
            }
            return true;
        }

        if (query.operation == MetalNumanXCoupledHumanOperation::massAction) {
            if (!validMassQuery(*state, pass, query)) return false;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[pass.transactionSlot];
            if (!slotMatches(slot, pass) || slot.busy ||
                slot.stage != HostStage::candidateValidated) return false;
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) return false;
            encoder.label = @"NumanX coupled Human effective-tangent action";
            [encoder setComputePipelineState:state->massActionPipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [encoder setBuffer:factor offset:0u atIndex:1u];
            [encoder setBuffer:projector offset:0u atIndex:2u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.input
                         offset:0u atIndex:3u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.output
                         offset:0u atIndex:4u];
            [encoder setBuffer:slot.statuses offset:0u atIndex:5u];
            [encoder setBuffer:slot.metadata offset:0u atIndex:6u];
            dispatchEnvironments(
                encoder, state->massActionPipeline, pass.environmentCount
            );
            [encoder endEncoding];
            return true;
        }

        if (query.operation ==
            MetalNumanXCoupledHumanOperation::inverseMassPreconditioner) {
            if (!validInverseQuery(*state, pass, query)) return false;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[pass.transactionSlot];
            if (!slotMatches(slot, pass) || slot.busy ||
                slot.stage != HostStage::candidateValidated) return false;
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) return false;
            encoder.label = @"NumanX coupled Human effective-tangent preconditioner";
            [encoder setComputePipelineState:state->inverseMassPipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [encoder setBuffer:factor offset:0u atIndex:1u];
            [encoder setBuffer:projector offset:0u atIndex:2u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.input
                         offset:0u atIndex:3u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.output
                         offset:0u atIndex:4u];
            [encoder setBuffer:(__bridge id<MTLBuffer>)query.statuses
                         offset:0u atIndex:5u];
            [encoder setBuffer:slot.statuses offset:0u atIndex:6u];
            [encoder setBuffer:slot.metadata offset:0u atIndex:7u];
            dispatchEnvironments(
                encoder, state->inverseMassPipeline, pass.environmentCount
            );
            [encoder endEncoding];
            return true;
        }

        if (!validPublishQuery(*state, pass, query)) return false;
        std::lock_guard lock(state->mutex);
        Slot& slot = state->slots[pass.transactionSlot];
        if (!slotMatches(slot, pass) || slot.busy ||
            slot.stage != HostStage::candidateValidated) return false;
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        encoder.label = @"NumanX coupled Human staged generalized reaction";
        [encoder setComputePipelineState:state->publishPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBuffer:factor offset:0u atIndex:1u];
        [encoder setBuffer:projector offset:0u atIndex:2u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)query.input
                     offset:0u atIndex:3u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)query.output
                     offset:0u atIndex:4u];
        [encoder setBuffer:slot.reaction offset:0u atIndex:5u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)query.matterOutcomes
                     offset:0u atIndex:6u];
        [encoder setBytes:&query.matterOutcomeStride
                   length:sizeof(query.matterOutcomeStride) atIndex:7u];
        [encoder setBytes:&query.matterSuccessCode
                   length:sizeof(query.matterSuccessCode) atIndex:8u];
        [encoder setBytes:&query.expectedMatterCompletedMicrosteps
                   length:sizeof(query.expectedMatterCompletedMicrosteps)
                  atIndex:9u];
        [encoder setBuffer:slot.statuses offset:0u atIndex:10u];
        [encoder setBuffer:slot.metadata offset:0u atIndex:11u];
        dispatchEnvironments(
            encoder, state->publishPipeline, pass.environmentCount
        );
        [encoder endEncoding];
        if (!encodeAdvance(
                *state,
                slot,
                commandBuffer,
                pass,
                HostStage::candidateValidated,
                HostStage::published,
                MR_NUMANX_COUPLED_HUMAN_WITNESS_PUBLISH,
                0u,
                @"NumanX coupled Human publish witness")) return false;
        slot.stage = HostStage::published;
        return true;
    }
}

[[nodiscard]] bool resolveCallback(
    void* opaque,
    const MetalNumanXCoupledHumanPass& pass,
    const MetalNumanXCoupledHumanResolveQuery& query
) noexcept {
    @autoreleasepool {
        auto* state = static_cast<State*>(opaque);
        if (state == nullptr ||
            !validPassHeader(
                *state, pass, MetalNumanXCoupledHumanPhase::postDynamics) ||
            !validStandStatuses(*state, pass) ||
            (pass.accessFlags &
             (MetalNumanXCoupledHumanWriteReaction |
              MetalNumanXCoupledHumanWriteJointStatus)) !=
                (MetalNumanXCoupledHumanWriteReaction |
                 MetalNumanXCoupledHumanWriteJointStatus) ||
            query.abiVersion != MR_NUMANX_COUPLED_HUMAN_ABI_VERSION ||
            query.structSize != sizeof(MetalNumanXCoupledHumanResolveQuery) ||
            query.accessFlags !=
                MetalNumanXCoupledHumanQueryReadMatterOutcome ||
            query.reserved0 != 0u || query.matterOutcomeStride == 0u ||
            query.expectedMatterCompletedMicrosteps == 0u ||
            query.transactionSlot != pass.transactionSlot ||
            query.programFingerprint != state->fingerprint ||
            query.transactionFingerprint != pass.transactionFingerprint ||
            query.linearizationEpoch != pass.linearizationEpoch ||
            query.slotGeneration != pass.slotGeneration) {
            return false;
        }
        std::uint64_t requiredElements = 0u;
        std::uint64_t requiredByteCount = 0u;
        if (!metalIndexedStridedLast(
                pass.environmentCount,
                query.matterOutcomeStride,
                requiredElements) ||
            query.matterOutcomeElementCount < requiredElements ||
            !requiredBytes(
                requiredElements,
                sizeof(MRNumanXCoupledMatterOutcomeGPU),
                requiredByteCount) ||
            !validBuffer(
                *state,
                query.matterOutcomes,
                query.matterOutcomeGPUAddress,
                requiredByteCount)) {
            return false;
        }
        std::uint64_t standElements = 0u;
        std::uint64_t standBytes = 0u;
        BufferRegion standRegion{};
        BufferRegion outcomeRegion{};
        if (!metalIndexedStridedLast(
                pass.environmentCount,
                pass.standStatusStride,
                standElements) ||
            !requiredBytes(
                standElements,
                sizeof(MRNumiHumanStandStatusGPU),
                standBytes) ||
            !makeBufferRegion(
                *state,
                pass.standStatuses,
                pass.standStatusesGPUAddress,
                standBytes,
                standRegion) ||
            !makeBufferRegion(
                *state,
                query.matterOutcomes,
                query.matterOutcomeGPUAddress,
                requiredByteCount,
                outcomeRegion) ||
            !queryRegionsIsolated(
                *state, pass, {standRegion, outcomeRegion})) {
            return false;
        }

        MRNumanXCoupledHumanResolveDispatchGPU dispatch{};
        dispatch.abiVersion = MR_NUMANX_COUPLED_HUMAN_ABI_VERSION;
        dispatch.environmentCount = pass.environmentCount;
        dispatch.stepIndex = pass.stepIndex;
        dispatch.dofCount = pass.dofCount;
        dispatch.statusStride = 1u;
        dispatch.standStatusStride = pass.standStatusStride;
        dispatch.matterOutcomeStride = query.matterOutcomeStride;
        dispatch.reactionStride = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
        dispatch.expectedHumanCompletedSteps = pass.stepIndex + 1u;
        dispatch.expectedMatterCompletedMicrosteps =
            query.expectedMatterCompletedMicrosteps;
        dispatch.matterSuccessCode = query.matterSuccessCode;
        dispatch.transactionSlot = pass.transactionSlot;
        dispatch.programFingerprint = state->fingerprint;
        dispatch.transactionFingerprint = pass.transactionFingerprint;
        dispatch.linearizationEpoch = pass.linearizationEpoch;
        dispatch.slotGeneration = pass.slotGeneration;

        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        std::lock_guard lock(state->mutex);
        Slot& slot = state->slots[pass.transactionSlot];
        if (!slotMatches(slot, pass) || slot.busy ||
            slot.stage != HostStage::published) return false;
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        encoder.label = @"NumanX coupled Human joint decision";
        [encoder setComputePipelineState:state->resolvePipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)pass.standStatuses
                     offset:0u atIndex:1u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)query.matterOutcomes
                     offset:0u atIndex:2u];
        [encoder setBuffer:slot.reaction offset:0u atIndex:3u];
        [encoder setBuffer:slot.statuses offset:0u atIndex:4u];
        [encoder setBuffer:slot.metadata offset:0u atIndex:5u];
        dispatchEnvironments(
            encoder, state->resolvePipeline, pass.environmentCount
        );
        [encoder endEncoding];
        if (!encodeAdvance(
                *state,
                slot,
                commandBuffer,
                pass,
                HostStage::published,
                HostStage::resolved,
                MR_NUMANX_COUPLED_HUMAN_WITNESS_RESOLVE,
                0u,
                @"NumanX coupled Human resolve witness")) return false;
        slot.stage = HostStage::resolved;
        return true;
    }
}

} // namespace

MetalNumanXCoupledHumanContext::MetalNumanXCoupledHumanContext(
    MetalNumanXCoupledHumanConfig config
) : state_(std::make_unique<detail::MetalNumanXCoupledHumanState>(
        std::move(config))) {}

MetalNumanXCoupledHumanContext::~MetalNumanXCoupledHumanContext() = default;

MetalNumanXCoupledHumanContext::MetalNumanXCoupledHumanContext(
    MetalNumanXCoupledHumanContext&& other
) noexcept = default;

MetalNumanXCoupledHumanContext&
MetalNumanXCoupledHumanContext::operator=(
    MetalNumanXCoupledHumanContext&& other
) noexcept = default;

MetalNumanXCoupledHumanDiagnostics
MetalNumanXCoupledHumanContext::initialize() {
    if (state_ == nullptr) {
        return diagnostics(
            nullptr,
            MetalNumanXCoupledHumanHostStatus::invalidConfiguration,
            "context was moved from"
        );
    }
    std::lock_guard lock(state_->mutex);
    State& state = *state_;
    if (state.initialized) {
        return diagnostics(
            &state,
            MetalNumanXCoupledHumanHostStatus::success
        );
    }
    const auto& config = state.config;
    if (config.metallibPath.empty() ||
        config.environmentCapacity == 0u ||
        config.pointCapacity > MR_NUMANX_COUPLED_HUMAN_MAX_POINTS ||
        config.transactionSlotCount == 0u ||
        config.transactionSlotCount >
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS ||
        config.reserved0 != 0u || config.maximumRetainedBytes == 0u) {
        return diagnostics(
            &state,
            MetalNumanXCoupledHumanHostStatus::invalidConfiguration,
            "metallibPath, capacities, reserved fields, or retained-byte ceiling are invalid"
        );
    }

    std::uint64_t reactionElements = 0u;
    std::uint64_t reactionBytes = 0u;
    std::uint64_t statusBytes = 0u;
    std::uint64_t metadataBytes = 0u;
    std::uint64_t perSlotBytes = 0u;
    std::uint64_t publicArenaBytes = 0u;
    std::uint64_t totalBytes = 0u;
    if (!metalIndexedProduct(
            config.environmentCapacity,
            MR_NUMANX_COUPLED_HUMAN_MAX_DOFS,
            reactionElements) ||
        !requiredBytes(reactionElements, sizeof(float), reactionBytes) ||
        !requiredBytes(
            config.environmentCapacity,
            sizeof(MRNumanXCoupledHumanStatusGPU),
            statusBytes) ||
        !requiredBytes(
            1u,
            sizeof(MRNumanXCoupledHumanSlotMetadataGPU),
            metadataBytes) ||
        !checkedAdd(reactionBytes, statusBytes, publicArenaBytes) ||
        !checkedAdd(publicArenaBytes, metadataBytes, perSlotBytes) ||
        !checkedMultiply(
            perSlotBytes, config.transactionSlotCount, totalBytes) ||
        totalBytes > config.maximumRetainedBytes) {
        return diagnostics(
            &state,
            MetalNumanXCoupledHumanHostStatus::arithmeticOverflow,
            "reaction/status/metadata arena sizes overflow or exceed maximumRetainedBytes"
        );
    }

    @autoreleasepool {
        state.device = MTLCreateSystemDefaultDevice();
        if (state.device == nil) {
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::metalDeviceUnavailable,
                "no system-default Metal device is available"
            );
        }
        NSString* path = [NSString
            stringWithUTF8String:config.metallibPath.c_str()];
        if (path == nil) {
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::invalidConfiguration,
                "metallibPath is not valid UTF-8"
            );
        }
        BOOL isDirectory = NO;
        if (![[NSFileManager defaultManager]
                fileExistsAtPath:path
                     isDirectory:&isDirectory] || isDirectory) {
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::metallibUnavailable,
                "metallibPath does not name a readable regular file"
            );
        }
        NSError* imageError = nil;
        NSData* image = [NSData dataWithContentsOfFile:path
                                               options:NSDataReadingMappedIfSafe
                                                 error:&imageError];
        detail::NumanXExecutableImageIdentity imageIdentity{};
        if (image == nil ||
            !numanXExecutableImageIdentity(
                image.bytes, image.length, imageIdentity)) {
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::metallibUnavailable,
                "failed to read the explicit coupled-Human metallib image: " +
                    fromNSString(imageError.localizedDescription)
            );
        }
        dispatch_data_t libraryImage = dispatch_data_create(
            image.bytes,
            image.length,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT
        );
        if (libraryImage == nullptr) {
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::metallibUnavailable,
                "failed to retain the coupled-Human metallib image"
            );
        }
        NSError* libraryError = nil;
        state.library = [state.device
            newLibraryWithData:libraryImage error:&libraryError];
        if (state.library == nil) {
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::metalLibraryFailure,
                "failed to load explicit coupled-Human metallib: " +
                    fromNSString(libraryError.localizedDescription)
            );
        }
        state.metallibIdentity = imageIdentity;

        const auto makePipeline = [&] (
            NSString* name,
            __strong id<MTLComputePipelineState>& pipeline
        ) -> MetalNumanXCoupledHumanDiagnostics {
            id<MTLFunction> function =
                [state.library newFunctionWithName:name];
            if (function == nil) {
                return diagnostics(
                    &state,
                    MetalNumanXCoupledHumanHostStatus::metalPipelineFailure,
                    "explicit metallib is missing kernel " +
                        fromNSString(name)
                );
            }
            NSError* pipelineError = nil;
            pipeline = [state.device
                newComputePipelineStateWithFunction:function
                                               error:&pipelineError];
            if (pipeline == nil) {
                return diagnostics(
                    &state,
                    MetalNumanXCoupledHumanHostStatus::metalPipelineFailure,
                    "failed to create pipeline for " + fromNSString(name) +
                        ": " +
                        fromNSString(pipelineError.localizedDescription)
                );
            }
            return diagnostics(
                &state,
                MetalNumanXCoupledHumanHostStatus::success
            );
        };

        MetalNumanXCoupledHumanDiagnostics result = makePipeline(
            @"numanx_coupled_human_begin", state.beginPipeline
        );
        if (!result.succeeded()) return result;
        result = makePipeline(
            @"numanx_coupled_human_validate_candidate",
            state.validateCandidatePipeline
        );
        if (!result.succeeded()) return result;
        result = makePipeline(
            @"numanx_coupled_human_effective_tangent_action",
            state.massActionPipeline
        );
        if (!result.succeeded()) return result;
        result = makePipeline(
            @"numanx_coupled_human_effective_tangent_preconditioner",
            state.inverseMassPipeline
        );
        if (!result.succeeded()) return result;
        result = makePipeline(
            @"numanx_coupled_human_stage_generalized_reaction",
            state.publishPipeline
        );
        if (!result.succeeded()) return result;
        result = makePipeline(
            @"numanx_coupled_human_resolve", state.resolvePipeline
        );
        if (!result.succeeded()) return result;
        result = makePipeline(
            @"numanx_coupled_human_advance", state.advancePipeline
        );
        if (!result.succeeded()) return result;

        std::vector<Slot> slots(config.transactionSlotCount);
        for (std::uint32_t index = 0u;
             index < config.transactionSlotCount;
             ++index) {
            slots[index].reaction = [state.device
                newBufferWithLength:static_cast<NSUInteger>(reactionBytes)
                           options:MTLResourceStorageModePrivate];
            slots[index].statuses = [state.device
                newBufferWithLength:static_cast<NSUInteger>(statusBytes)
                           options:MTLResourceStorageModePrivate];
            slots[index].metadata = [state.device
                newBufferWithLength:static_cast<NSUInteger>(metadataBytes)
                           options:MTLResourceStorageModePrivate];
            if (slots[index].reaction == nil ||
                slots[index].statuses == nil ||
                slots[index].metadata == nil ||
                slots[index].reaction.gpuAddress == 0u ||
                slots[index].statuses.gpuAddress == 0u ||
                slots[index].metadata.gpuAddress == 0u) {
                return diagnostics(
                    &state,
                    MetalNumanXCoupledHumanHostStatus::metalBufferFailure,
                    "failed to allocate exact private reaction/status/metadata arena"
                );
            }
            slots[index].reaction.label = [NSString stringWithFormat:
                @"NumanX Human Matter reaction slot %u", index];
            slots[index].statuses.label = [NSString stringWithFormat:
                @"NumanX Human joint status slot %u", index];
            slots[index].metadata.label = [NSString stringWithFormat:
                @"NumanX Human private metadata slot %u", index];
        }
        state.slots = std::move(slots);
    }

    std::uint64_t fingerprint = hashString(
        kFnvOffset,
        "metalrobo.numanx-coupled-human.program.v4"
    );
    fingerprint = hashValue(
        fingerprint, state.metallibIdentity.byteFingerprint
    );
    fingerprint = hashValue(
        fingerprint, state.metallibIdentity.byteCount
    );
    fingerprint = hashValue(
        fingerprint, MR_NUMANX_COUPLED_HUMAN_ABI_VERSION
    );
    fingerprint = hashValue(fingerprint, kProgramCapabilities);
    fingerprint = hashValue(fingerprint, config.environmentCapacity);
    fingerprint = hashValue(fingerprint, config.pointCapacity);
    fingerprint = hashValue(fingerprint, config.transactionSlotCount);
    state.fingerprint = nonzeroHash(fingerprint);
    state.retainedBytes = totalBytes;
    state.initialized = true;
    return diagnostics(
        &state,
        MetalNumanXCoupledHumanHostStatus::success
    );
}

MetalNumanXCoupledHumanProgram
MetalNumanXCoupledHumanContext::program() const noexcept {
    MetalNumanXCoupledHumanProgram result{};
    if (state_ == nullptr || !state_->initialized) return result;
    result.capabilities = kProgramCapabilities;
    result.environmentCapacity = state_->config.environmentCapacity;
    result.pointCapacity = state_->config.pointCapacity;
    result.transactionSlotCount = state_->config.transactionSlotCount;
    result.context = state_.get();
    result.begin = &beginCallback;
    result.encode = &encodeCallback;
    result.resolve = &resolveCallback;
    result.fingerprint = state_->fingerprint;
    return result;
}

MetalNumanXCoupledHumanDiagnostics
MetalNumanXCoupledHumanContext::arenaView(
    const std::uint32_t transactionSlot,
    MetalNumanXCoupledHumanArenaView& view
) const {
    view = {};
    if (state_ == nullptr || !state_->initialized) {
        return diagnostics(
            state_.get(),
            MetalNumanXCoupledHumanHostStatus::uninitialized,
            "initialize() must succeed before requesting an arena view"
        );
    }
    std::lock_guard lock(state_->mutex);
    if (transactionSlot >= state_->slots.size()) {
        return diagnostics(
            state_.get(),
            MetalNumanXCoupledHumanHostStatus::invalidSlot,
            "transactionSlot exceeds configured slot capacity"
        );
    }
    const Slot& slot = state_->slots[transactionSlot];
    view.matterGeneralizedReaction = (__bridge void*)slot.reaction;
    view.jointStatuses = (__bridge void*)slot.statuses;
    view.transactionMetadata = (__bridge void*)slot.metadata;
    view.reactionGPUAddress = slot.reaction.gpuAddress;
    view.jointStatusGPUAddress = slot.statuses.gpuAddress;
    view.transactionMetadataGPUAddress = slot.metadata.gpuAddress;
    view.reactionByteCount = slot.reaction.length;
    view.jointStatusByteCount = slot.statuses.length;
    view.transactionMetadataByteCount = slot.metadata.length;
    view.environmentCapacity = state_->config.environmentCapacity;
    view.reactionStride = MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
    view.jointStatusStride = 1u;
    view.transactionSlot = transactionSlot;
    view.programFingerprint = state_->fingerprint;
    return diagnostics(
        state_.get(),
        MetalNumanXCoupledHumanHostStatus::success
    );
}

const char* metalNumanXCoupledHumanHostStatusName(
    const MetalNumanXCoupledHumanHostStatus status
) noexcept {
    switch (status) {
    case MetalNumanXCoupledHumanHostStatus::success:
        return "success";
    case MetalNumanXCoupledHumanHostStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalNumanXCoupledHumanHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalNumanXCoupledHumanHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalNumanXCoupledHumanHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalNumanXCoupledHumanHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalNumanXCoupledHumanHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalNumanXCoupledHumanHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalNumanXCoupledHumanHostStatus::incompatibleDevice:
        return "incompatible_device";
    case MetalNumanXCoupledHumanHostStatus::uninitialized:
        return "uninitialized";
    case MetalNumanXCoupledHumanHostStatus::invalidSlot:
        return "invalid_slot";
    }
    return "unknown";
}

} // namespace metalrobo
