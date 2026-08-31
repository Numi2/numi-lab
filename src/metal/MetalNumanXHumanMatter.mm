#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanMatter.hpp"
#include "metalrobo/MetalNumanXHumanIO.hpp"

#include "metalrobo/MetalNumanXCoupledHuman.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "numi/matter/matter.hpp"
#include "numi/matter/accepted_state_apply_gpu.h"
#include "numi/matter/shared.h"
#include "NumanXProgramIdentity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION ==
              NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION);
static_assert(sizeof(MRNumanXAcceptedStateProofGPU) ==
              sizeof(NMAcceptedStateProofGPU));
static_assert(alignof(MRNumanXAcceptedStateProofGPU) ==
              alignof(NMAcceptedStateProofGPU));
#define MR_ASSERT_PROOF_FIELD_LAYOUT(field) \
    static_assert(offsetof(MRNumanXAcceptedStateProofGPU, field) == \
                  offsetof(NMAcceptedStateProofGPU, field))
MR_ASSERT_PROOF_FIELD_LAYOUT(abiVersion);
MR_ASSERT_PROOF_FIELD_LAYOUT(structSize);
MR_ASSERT_PROOF_FIELD_LAYOUT(status);
MR_ASSERT_PROOF_FIELD_LAYOUT(environment);
MR_ASSERT_PROOF_FIELD_LAYOUT(transactionFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(substepFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(acceptedTimestampMicroseconds);
MR_ASSERT_PROOF_FIELD_LAYOUT(physicsGeneration);
MR_ASSERT_PROOF_FIELD_LAYOUT(humanStateFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(matterStateFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(physicsStateFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(matterSourcePhysicsFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(matterDeviceProgramFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(linearizationEpoch);
MR_ASSERT_PROOF_FIELD_LAYOUT(slotGeneration);
MR_ASSERT_PROOF_FIELD_LAYOUT(proofFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(adapterProgramFingerprint);
MR_ASSERT_PROOF_FIELD_LAYOUT(transactionPolicyFingerprint);
#undef MR_ASSERT_PROOF_FIELD_LAYOUT

static_assert(sizeof(MRNumanXHumanMatterProposalGPU) ==
              sizeof(NMOwnerProposalGPU));
static_assert(sizeof(MRNumanXHumanMatterBrainAckGPU) ==
              sizeof(NMOwnerBrainAckGPU));
static_assert(sizeof(MRNumanXHumanMatterApplyActionGPU) ==
              sizeof(NMOwnerApplyActionGPU));
static_assert(sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU) ==
              sizeof(NMMatterApplyOutcomeGPU));
static_assert(sizeof(MRNumanXHumanMatterAppliedOutcomeGPU) ==
              sizeof(NMOwnerAppliedOutcomeGPU));
static_assert(sizeof(MRNumanXHumanMatterJointPublicationFenceGPU) ==
              sizeof(NMJointPublicationFenceGPU));

namespace metalrobo::detail {

enum class HumanMatterSlotStage : std::uint32_t {
    empty = 0u,
    prepared,
    leased,
    begun,
    preEncoded,
    postEncoded,
    applicationReserved,
    applyEncoded,
    acceptedPendingPublication,
    restoreRequired,
    terminalNoTouch,
    released,
};

struct MetalNumanXHumanMatterCallbackFrame;

// Scalar identity only. The adapter never retains owner buffers/events; the
// owner is required to keep every lease resource alive until release.
struct MetalNumanXHumanMatterLeaseIdentity {
    std::uintptr_t preparedTokens = 0u;
    std::uintptr_t proposals = 0u;
    std::uintptr_t proposedTokens = 0u;
    std::uintptr_t applyActions = 0u;
    std::uintptr_t matterApplyOutcomes = 0u;
    std::uintptr_t appliedOutcomes = 0u;
    std::uintptr_t finalTokens = 0u;
    std::uintptr_t publicationFences = 0u;
    std::uintptr_t physicalEvent = 0u;
    std::uint64_t preparedTokensGPUAddress = 0u;
    std::uint64_t proposalsGPUAddress = 0u;
    std::uint64_t proposedTokensGPUAddress = 0u;
    std::uint64_t applyActionsGPUAddress = 0u;
    std::uint64_t matterApplyOutcomesGPUAddress = 0u;
    std::uint64_t appliedOutcomesGPUAddress = 0u;
    std::uint64_t finalTokensGPUAddress = 0u;
    std::uint64_t publicationFencesGPUAddress = 0u;
    std::uint64_t preparedTokenBytes = 0u;
    std::uint64_t proposalElements = 0u;
    std::uint64_t proposedTokenBytes = 0u;
    std::uint64_t applyActionElements = 0u;
    std::uint64_t matterApplyOutcomeElements = 0u;
    std::uint64_t appliedOutcomeElements = 0u;
    std::uint64_t finalTokenBytes = 0u;
    std::uint64_t publicationFenceElements = 0u;
    std::uint64_t physicalEventValue = 0u;
    std::uint64_t proposalEventValue = 0u;
    std::uint64_t appliedEventValue = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t stepIndex = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t preparedTokenStrideBytes = 0u;
    std::uint32_t proposalStride = 0u;
    std::uint32_t proposedTokenStrideBytes = 0u;
    std::uint32_t applyActionStride = 0u;
    std::uint32_t matterApplyOutcomeStride = 0u;
    std::uint32_t appliedOutcomeStride = 0u;
    std::uint32_t finalTokenStrideBytes = 0u;
    std::uint32_t publicationFenceStride = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t qCoordinateCount = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t dofLayoutVersion = 0u;
    std::uint32_t reservedDofLayout = 0u;
};

struct MetalNumanXHumanMatterAuthorityRegion {
    std::uint64_t address = 0u;
    std::uint64_t bytes = 0u;
};

struct MetalNumanXHumanMatterApplicationReservation {
    std::uintptr_t preflights = 0u;
    std::uintptr_t preflightEvent = 0u;
    std::uint64_t preflightsGPUAddress = 0u;
    std::uint64_t preflightElements = 0u;
    std::uint64_t preflightEventValue = 0u;
    std::uintptr_t brainAcks = 0u;
    std::uintptr_t brainAckEvent = 0u;
    std::uint64_t brainAcksGPUAddress = 0u;
    std::uint64_t brainAckElements = 0u;
    std::uint64_t brainAckEventValue = 0u;
    std::uint32_t preflightStride = 0u;
    std::uint32_t brainAckStride = 0u;
    MetalNumanXHumanMatterApplyMode applyMode =
        MetalNumanXHumanMatterApplyMode::validateBrainAck;
    MRNumanXHumanMatterProposalGPU proposal{};
    bool active = false;
    bool applyActive = false;
};

struct MetalNumanXHumanMatterSlot {
    MetalNumanXCoupledHumanArenaView coupledArena{};
    // CoupledHuman's transient 16-byte post-commit status. It is not the
    // ABI4 Matter application result exposed through the owner lease.
    __strong id<MTLBuffer> matterOutcomes = nil;
    // Exact 128-byte ABI4 result written by Matter on the later apply CB.
    __strong id<MTLBuffer> matterApplyOutcomes = nil;
    __strong id<MTLBuffer> worldStatuses = nil;
    __strong id<MTLBuffer> acceptedTokens = nil;
    __strong id<MTLBuffer> acceptedStateProofs = nil;
    MetalNumanXHumanMatterTransaction transaction{};
    MetalNumanXHumanMatterLeaseIdentity lease{};
    HumanMatterSlotStage stage = HumanMatterSlotStage::empty;
    std::uintptr_t commandBufferIdentity = 0u;
    std::uintptr_t applyCommandBufferIdentity = 0u;
    std::uint64_t passSignature = 0u;
    std::uint64_t applyAttempt = 0u;
    std::array<MetalNumanXHumanMatterAuthorityRegion, 14u>
        physicalAuthorityRegions{};
    std::size_t physicalAuthorityRegionCount = 0u;
    MetalNumanXHumanMatterApplicationReservation applicationReservation{};
    numi::matter::PreparedStatePublicationBinding publicationBinding{};
    numi::matter::PreparedStatePublicationReservation
        publicationReservation{};
    MRNumanXHumanMatterAppliedOutcomeGPU appliedOutcome{};
    MetalNumanXHumanIOCandidatePublicationProgram humanIOCandidate{};
    MetalNumanXHumanIOCandidatePublicationBinding humanIOBinding{};
    bool publicationReserved = false;
    bool humanIOCandidateBound = false;
    bool humanIORootReserved = false;
    bool matterOpened = false;
    bool cancelIssued = false;
    bool completionArmed = false;
    bool leaseAcquired = false;
    bool physicalCommandCompleted = false;
    bool physicalCommandFailed = false;
    bool applyEncoded = false;
    bool applyCommandCompleted = false;
    bool applyCommandSucceeded = false;
    bool restoreRequired = false;
    bool awaitingFirstAbort = false;
    MetalNumanXHumanMatterCallbackFrame* callbackFrame = nullptr;
};

struct MetalNumanXHumanMatterState {
    explicit MetalNumanXHumanMatterState(
        MetalNumanXHumanMatterConfig value
    ) : config(std::move(value)), coupledContext([&] {
            MetalNumanXCoupledHumanConfig result;
            result.metallibPath = config.coupledHumanMetallibPath;
            result.environmentCapacity = config.environmentCapacity;
            result.pointCapacity = config.pointCapacity;
            result.transactionSlotCount = config.transactionSlotCount;
            result.maximumRetainedBytes = config.maximumRetainedBytes;
            return result;
        }()) {}

    MetalNumanXHumanMatterConfig config;
    MetalNumanXCoupledHumanContext coupledContext;
    MetalNumanXCoupledHumanProgram coupledProgram{};
    __strong id<MTLDevice> device = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> prepareWorldStatusPipeline = nil;
    __strong id<MTLComputePipelineState> mapHumanStatusPipeline = nil;
    __strong id<MTLComputePipelineState> captureOutcomePipeline = nil;
    __strong id<MTLComputePipelineState> preparedTokenPipeline = nil;
    std::vector<MetalNumanXHumanMatterSlot> slots;
    std::mutex mutex;
    NumanXExecutableImageIdentity metallibIdentity{};
    std::uint64_t fingerprint = 0u;
    std::uint64_t matterSourceFingerprint = 0u;
    std::uint64_t matterDeviceFingerprint = 0u;
    std::uint64_t retainedBytes = 0u;
    bool initialized = false;
    std::weak_ptr<MetalNumanXHumanMatterState> self;
    // Armed while any owner lease can still call the raw callback context.
    // A terminal quarantine intentionally retains this state indefinitely;
    // freeing its buffers would be less safe than leaking failed authority.
    std::shared_ptr<MetalNumanXHumanMatterState> lifetimeHold;
};

struct MetalNumanXHumanMatterCallbackFrame {
    MetalNumanXHumanMatterState* state = nullptr;
    MetalNumanXHumanMatterSlot* slot = nullptr;
    const MetalNumanXHumanMatterPass* ownerPass = nullptr;
    MetalNumanXCoupledHumanPass coupledPass{};
    std::array<std::uint32_t, 4u> operationCounts{};
};

} // namespace metalrobo::detail

namespace metalrobo {
namespace {

using State = detail::MetalNumanXHumanMatterState;
using Slot = detail::MetalNumanXHumanMatterSlot;
using SlotStage = detail::HumanMatterSlotStage;
using Frame = detail::MetalNumanXHumanMatterCallbackFrame;

constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

[[nodiscard]] bool checkedMultiply(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& result
) noexcept {
    if (left != 0u && right >
            std::numeric_limits<std::uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

[[nodiscard]] bool checkedAdd(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& result
) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool narrowU32(
    const std::uint64_t value,
    std::uint32_t& result
) noexcept {
    if (value > std::numeric_limits<std::uint32_t>::max()) return false;
    result = static_cast<std::uint32_t>(value);
    return true;
}

[[nodiscard]] bool sameDevice(id<MTLDevice> left, id<MTLDevice> right) {
    return left != nil && right != nil && left.registryID == right.registryID;
}

[[nodiscard]] bool exactBuffer(
    id<MTLDevice> device,
    void* raw,
    const std::uint64_t address,
    const std::uint64_t requiredBytes
) noexcept {
    if (device == nil || raw == nullptr || address == 0u ||
        requiredBytes == 0u ||
        requiredBytes > std::numeric_limits<NSUInteger>::max()) return false;
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)raw;
    return buffer != nil && sameDevice(device, buffer.device) &&
        static_cast<std::uint64_t>(buffer.length) >= requiredBytes &&
        static_cast<std::uint64_t>(buffer.gpuAddress) == address;
}

[[nodiscard]] bool exactHostVisibleBuffer(
    id<MTLDevice> device,
    void* raw,
    const std::uint64_t address,
    const std::uint64_t requiredBytes
) noexcept {
    if (!exactBuffer(device, raw, address, requiredBytes)) return false;
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)raw;
    return buffer.storageMode == MTLStorageModeShared &&
        buffer.contents != nullptr;
}

struct BufferRegion {
    std::uint64_t address = 0u;
    std::uint64_t bytes = 0u;
};

[[nodiscard]] bool bufferRegion(
    id<MTLDevice> device,
    void* raw,
    const std::uint64_t address,
    const std::uint64_t bytes,
    BufferRegion& result
) noexcept {
    if (!exactBuffer(device, raw, address, bytes) ||
        bytes > std::numeric_limits<std::uint64_t>::max() - address) {
        return false;
    }
    result = {address, bytes};
    return true;
}

[[nodiscard]] bool regionsOverlap(
    const BufferRegion& left,
    const BufferRegion& right
) noexcept {
    return left.address < right.address + right.bytes &&
        right.address < left.address + left.bytes;
}

[[nodiscard]] bool sameRegion(
    const BufferRegion& left,
    const BufferRegion& right
) noexcept {
    return left.address == right.address && left.bytes == right.bytes;
}

[[nodiscard]] bool scalarRegion(
    const std::uint64_t address,
    const std::uint64_t bytes,
    BufferRegion& result
) noexcept {
    if (address == 0u || bytes == 0u ||
        bytes > std::numeric_limits<std::uint64_t>::max() - address) {
        return false;
    }
    result = {address, bytes};
    return true;
}

constexpr std::size_t kAdapterPrivateRegionCount = 8u;
constexpr std::size_t kAcceptedTokenPrivateRegion = 6u;
constexpr std::size_t kMatterApplyPrivateRegion = 4u;

[[nodiscard]] bool adapterPrivateRegions(
    State& state,
    const Slot& slot,
    std::array<BufferRegion, kAdapterPrivateRegionCount>& regions
) noexcept {
    const std::array<std::tuple<void*, std::uint64_t, std::uint64_t>,
                     kAdapterPrivateRegionCount> buffers{{
        {slot.coupledArena.matterGeneralizedReaction,
         slot.coupledArena.reactionGPUAddress,
         slot.coupledArena.reactionByteCount},
        {slot.coupledArena.jointStatuses,
         slot.coupledArena.jointStatusGPUAddress,
         slot.coupledArena.jointStatusByteCount},
        {slot.coupledArena.transactionMetadata,
         slot.coupledArena.transactionMetadataGPUAddress,
         slot.coupledArena.transactionMetadataByteCount},
        {(__bridge void*)slot.matterOutcomes,
         static_cast<std::uint64_t>(slot.matterOutcomes.gpuAddress),
         static_cast<std::uint64_t>(slot.matterOutcomes.length)},
        {(__bridge void*)slot.matterApplyOutcomes,
         static_cast<std::uint64_t>(slot.matterApplyOutcomes.gpuAddress),
         static_cast<std::uint64_t>(slot.matterApplyOutcomes.length)},
        {(__bridge void*)slot.worldStatuses,
         static_cast<std::uint64_t>(slot.worldStatuses.gpuAddress),
         static_cast<std::uint64_t>(slot.worldStatuses.length)},
        {(__bridge void*)slot.acceptedTokens,
         static_cast<std::uint64_t>(slot.acceptedTokens.gpuAddress),
         static_cast<std::uint64_t>(slot.acceptedTokens.length)},
        {(__bridge void*)slot.acceptedStateProofs,
         static_cast<std::uint64_t>(slot.acceptedStateProofs.gpuAddress),
         static_cast<std::uint64_t>(slot.acceptedStateProofs.length)},
    }};
    for (std::size_t index = 0u; index < buffers.size(); ++index) {
        const auto& [raw, address, bytes] = buffers[index];
        if (!bufferRegion(state.device, raw, address, bytes, regions[index])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool storedLeaseRegions(
    const Slot& slot,
    std::array<BufferRegion, 8u>& regions
) noexcept {
    const auto& lease = slot.lease;
    std::uint64_t proposalBytes = 0u;
    std::uint64_t actionBytes = 0u;
    std::uint64_t outcomeBytes = 0u;
    std::uint64_t appliedBytes = 0u;
    std::uint64_t fenceBytes = 0u;
    return checkedMultiply(lease.proposalElements, 128u, proposalBytes) &&
        checkedMultiply(lease.applyActionElements, 128u, actionBytes) &&
        checkedMultiply(
            lease.matterApplyOutcomeElements, 128u, outcomeBytes) &&
        checkedMultiply(lease.appliedOutcomeElements, 128u, appliedBytes) &&
        checkedMultiply(
            lease.publicationFenceElements, 128u, fenceBytes) &&
        scalarRegion(
            lease.preparedTokensGPUAddress, lease.preparedTokenBytes,
            regions[0]) &&
        scalarRegion(
            lease.proposalsGPUAddress, proposalBytes, regions[1]) &&
        scalarRegion(
            lease.proposedTokensGPUAddress, lease.proposedTokenBytes,
            regions[2]) &&
        scalarRegion(
            lease.applyActionsGPUAddress, actionBytes, regions[3]) &&
        scalarRegion(
            lease.matterApplyOutcomesGPUAddress, outcomeBytes, regions[4]) &&
        scalarRegion(
            lease.appliedOutcomesGPUAddress, appliedBytes, regions[5]) &&
        scalarRegion(
            lease.finalTokensGPUAddress, lease.finalTokenBytes, regions[6]) &&
        scalarRegion(
            lease.publicationFencesGPUAddress, fenceBytes, regions[7]);
}

[[nodiscard]] bool storedApplicationRegions(
    const Slot& slot,
    std::array<BufferRegion, 2u>& regions,
    std::size_t& count
) noexcept {
    count = 0u;
    const auto& reservation = slot.applicationReservation;
    if (!reservation.active) return true;
    std::uint64_t preflightBytes = 0u;
    if (!checkedMultiply(
            reservation.preflightElements, 128u, preflightBytes) ||
        !scalarRegion(
            reservation.preflightsGPUAddress, preflightBytes,
            regions[count])) {
        return false;
    }
    ++count;
    const bool hasAck = reservation.brainAcks != 0u ||
        reservation.brainAcksGPUAddress != 0u ||
        reservation.brainAckElements != 0u;
    if (!hasAck) {
        return reservation.brainAcks == 0u &&
            reservation.brainAcksGPUAddress == 0u &&
            reservation.brainAckElements == 0u;
    }
    std::uint64_t ackBytes = 0u;
    if (reservation.brainAcks == 0u ||
        !checkedMultiply(reservation.brainAckElements, 128u, ackBytes) ||
        !scalarRegion(
            reservation.brainAcksGPUAddress, ackBytes, regions[count])) {
        return false;
    }
    ++count;
    return true;
}

[[nodiscard]] bool slotRetainsBorrowedAuthority(
    const Slot& slot
) noexcept {
    return slot.stage != SlotStage::empty &&
        slot.stage != SlotStage::released;
}

// Caller-controlled buffers are admitted as one interval transaction. Every
// new physical, lease, preflight or ACK range is checked against all private
// adapter/CoupledHuman arenas and against every borrowed authority retained by
// every unresolved slot. This includes terminal quarantine: those bytes are
// immutable forever and can never become another slot's writable storage.
[[nodiscard]] bool authorityRegionsIsolated(
    State& state,
    const Slot& target,
    const BufferRegion* candidates,
    const std::size_t candidateCount,
    const bool skipTargetPhysical,
    const bool skipTargetLease,
    const bool skipTargetApplication,
    const bool allowTargetLeasePrivateAliases
) noexcept {
    if (candidates == nullptr || candidateCount == 0u) return false;
    for (std::size_t left = 0u; left < candidateCount; ++left) {
        if (candidates[left].address == 0u || candidates[left].bytes == 0u ||
            candidates[left].bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    candidates[left].address) {
            return false;
        }
        for (std::size_t right = left + 1u;
             right < candidateCount; ++right) {
            if (regionsOverlap(candidates[left], candidates[right])) {
                return false;
            }
        }
    }

    for (const Slot& owned : state.slots) {
        std::array<BufferRegion, kAdapterPrivateRegionCount> privateRegions{};
        if (!adapterPrivateRegions(state, owned, privateRegions)) return false;
        for (const BufferRegion& candidate :
             std::span<const BufferRegion>(candidates, candidateCount)) {
            for (std::size_t index = 0u;
                 index < privateRegions.size(); ++index) {
                if (!regionsOverlap(candidate, privateRegions[index])) {
                    continue;
                }
                const bool exactOwnedLeaseAlias =
                    allowTargetLeasePrivateAliases && &owned == &target &&
                    (index == kAcceptedTokenPrivateRegion ||
                     index == kMatterApplyPrivateRegion) &&
                    sameRegion(candidate, privateRegions[index]);
                if (!exactOwnedLeaseAlias) return false;
            }
        }
    }
    __unsafe_unretained id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)state.config.matterRuntime->statusBuffer();
    BufferRegion matterStatusRegion{};
    if (matterStatuses == nil || !bufferRegion(
            state.device, (__bridge void*)matterStatuses,
            matterStatuses.gpuAddress, matterStatuses.length,
            matterStatusRegion)) {
        return false;
    }
    for (const BufferRegion& candidate :
         std::span<const BufferRegion>(candidates, candidateCount)) {
        if (regionsOverlap(candidate, matterStatusRegion)) return false;
    }

    const auto overlapsCandidates = [&] (
        const BufferRegion* retained, const std::size_t retainedCount
    ) noexcept {
        for (const BufferRegion& candidate :
             std::span<const BufferRegion>(candidates, candidateCount)) {
            for (const BufferRegion& authority :
                 std::span<const BufferRegion>(retained, retainedCount)) {
                if (regionsOverlap(candidate, authority)) return true;
            }
        }
        return false;
    };
    for (const Slot& retained : state.slots) {
        if (!slotRetainsBorrowedAuthority(retained)) continue;
        const bool targetSlot = &retained == &target;
        if (!(targetSlot && skipTargetPhysical) &&
            retained.physicalAuthorityRegionCount != 0u) {
            if (retained.physicalAuthorityRegionCount !=
                    retained.physicalAuthorityRegions.size()) {
                return false;
            }
            std::array<BufferRegion, 14u> physical{};
            for (std::size_t index = 0u; index < physical.size(); ++index) {
                if (!scalarRegion(
                        retained.physicalAuthorityRegions[index].address,
                        retained.physicalAuthorityRegions[index].bytes,
                        physical[index])) {
                    return false;
                }
            }
            if (overlapsCandidates(physical.data(), physical.size())) {
                return false;
            }
        }
        if (!(targetSlot && skipTargetLease) && retained.leaseAcquired) {
            std::array<BufferRegion, 8u> leaseRegions{};
            if (!storedLeaseRegions(retained, leaseRegions) ||
                overlapsCandidates(leaseRegions.data(), leaseRegions.size())) {
                return false;
            }
        }
        if (!(targetSlot && skipTargetApplication) &&
            retained.applicationReservation.active) {
            std::array<BufferRegion, 2u> applicationRegions{};
            std::size_t applicationRegionCount = 0u;
            if (!storedApplicationRegions(
                    retained, applicationRegions, applicationRegionCount) ||
                overlapsCandidates(
                    applicationRegions.data(), applicationRegionCount)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool fixedAuthoritiesPairwiseDisjoint(State& state) noexcept {
    constexpr std::size_t maximumRegions =
        kAdapterPrivateRegionCount *
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS + 1u;
    std::array<BufferRegion, maximumRegions> regions{};
    std::size_t count = 0u;
    for (const Slot& slot : state.slots) {
        std::array<BufferRegion, kAdapterPrivateRegionCount> slotRegions{};
        if (!adapterPrivateRegions(state, slot, slotRegions)) return false;
        for (const BufferRegion& region : slotRegions) regions[count++] = region;
    }
    __unsafe_unretained id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)state.config.matterRuntime->statusBuffer();
    if (matterStatuses == nil || count >= regions.size() || !bufferRegion(
            state.device, (__bridge void*)matterStatuses,
            matterStatuses.gpuAddress, matterStatuses.length,
            regions[count])) {
        return false;
    }
    ++count;
    for (std::size_t left = 0u; left < count; ++left) {
        for (std::size_t right = left + 1u; right < count; ++right) {
            if (regionsOverlap(regions[left], regions[right])) return false;
        }
    }
    return true;
}

[[nodiscard]] bool exactSharedEvent(
    id<MTLDevice> device,
    void* raw
) noexcept {
    if (device == nil || raw == nullptr) return false;
    __unsafe_unretained id<MTLSharedEvent> event =
        (__bridge id<MTLSharedEvent>)raw;
    if (event == nil) return false;
    MTLSharedEventHandle* handle = [event newSharedEventHandle];
    id<MTLSharedEvent> imported = handle == nil
        ? nil : [device newSharedEventWithHandle:handle];
    return imported != nil;
}

[[nodiscard]] std::uint64_t gpuAddress(void* raw) noexcept {
    if (raw == nullptr) return 0u;
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)raw;
    return buffer == nil ? 0u : static_cast<std::uint64_t>(buffer.gpuAddress);
}

[[nodiscard]] bool validCommandBuffer(
    id<MTLDevice> device,
    void* raw
) noexcept {
    if (device == nil || raw == nullptr) return false;
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)raw;
    return commandBuffer != nil && commandBuffer.commandQueue != nil &&
        sameDevice(device, commandBuffer.commandQueue.device) &&
        commandBuffer.status == MTLCommandBufferStatusNotEnqueued;
}

void mixByte(std::uint64_t& hash, const std::uint8_t value) noexcept {
    hash ^= value;
    hash *= kFNVPrime;
}

void mixValue(
    std::uint64_t& hash,
    const std::uint64_t value
) noexcept {
    for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
        mixByte(hash, static_cast<std::uint8_t>(value >> (byte * 8u)));
    }
}

template <typename T>
    requires std::is_integral_v<T>
void mixValue(std::uint64_t& hash, const T value) noexcept {
    mixValue(hash, static_cast<std::uint64_t>(value));
}

template <typename T>
    requires std::is_enum_v<T>
void mixValue(std::uint64_t& hash, const T value) noexcept {
    mixValue(hash, static_cast<std::uint64_t>(value));
}

[[nodiscard]] std::uint64_t nonzeroHash(std::uint64_t hash) noexcept {
    return hash == 0u ? 1u : hash;
}

// ABI4 owner records are exactly 128 bytes with their terminal fingerprint at
// byte 120. This byte-wise helper is deliberately separate from mixValue,
// whose eight-byte widening is part of the adapter program fingerprint ABI.
[[nodiscard]] std::uint64_t humanMatterRecordFingerprint(
    const void* record
) noexcept {
    if (record == nullptr) return 0u;
    std::uint64_t hash = kFNVOffset;
    const auto* bytes = static_cast<const std::uint8_t*>(record);
    for (std::size_t index = 0u; index < 120u; ++index) {
        mixByte(hash, bytes[index]);
    }
    return hash == 0u ? kFNVOffset : hash;
}

[[nodiscard]] std::uint64_t acceptedPhysicsTokenFingerprint(
    const void* raw
) noexcept {
    if (raw == nullptr) return 0u;
    const auto mixLittleEndian = [] (
        std::uint64_t hash,
        const std::uint64_t value,
        const std::size_t byteCount
    ) noexcept {
        for (std::size_t index = 0u; index < byteCount; ++index) {
            hash = (hash ^ static_cast<std::uint8_t>(
                value >> (8u * index))) * kFNVPrime;
        }
        return hash;
    };
    std::array<std::uint64_t, 8u> words{};
    std::array<std::uint32_t, 16u> words32{};
    std::memcpy(words.data(), raw, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES);
    std::memcpy(words32.data(), raw, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES);
    std::uint64_t hash = kFNVOffset;
    hash = mixLittleEndian(hash, 1u, sizeof(std::uint32_t));
    for (std::size_t index = 0u; index < 5u; ++index) {
        hash = mixLittleEndian(hash, words[index], sizeof(std::uint64_t));
    }
    hash = mixLittleEndian(hash, words32[10], sizeof(std::uint32_t));
    hash = mixLittleEndian(hash, words32[11], sizeof(std::uint32_t));
    hash = mixLittleEndian(hash, words[6], sizeof(std::uint64_t));
    return hash;
}

[[nodiscard]] bool zeroBytes(
    const void* raw,
    const std::size_t byteCount
) noexcept {
    if (raw == nullptr) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    return std::all_of(
        bytes, bytes + byteCount,
        [](const std::uint8_t value) noexcept { return value == 0u; });
}

[[nodiscard]] bool validAcceptedPhysicsToken(
    const void* raw,
    const std::uint64_t transactionFingerprint,
    const std::uint64_t expectedFingerprint
) noexcept {
    if (raw == nullptr || transactionFingerprint == 0u ||
        expectedFingerprint == 0u) return false;
    std::array<std::uint64_t, 8u> words{};
    std::memcpy(words.data(), raw, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES);
    return words[0] == transactionFingerprint && words[6] == 0u &&
        words[7] == expectedFingerprint &&
        acceptedPhysicsTokenFingerprint(raw) == expectedFingerprint;
}

[[nodiscard]] NSString* nsString(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

[[nodiscard]] std::string fromNSString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) return {};
    return std::string(value.UTF8String);
}

[[nodiscard]] MetalNumanXHumanMatterDiagnostics diagnostics(
    const State* state,
    const MetalNumanXHumanMatterHostStatus status,
    std::string message = {}
) {
    MetalNumanXHumanMatterDiagnostics result;
    result.status = status;
    result.message = std::move(message);
    if (state != nullptr) {
        result.programFingerprint = state->fingerprint;
        result.matterSourcePhysicsFingerprint =
            state->matterSourceFingerprint;
        result.matterDeviceProgramFingerprint =
            state->matterDeviceFingerprint;
        result.retainedBytes = state->retainedBytes;
        result.acceptedStateProofAvailable =
            state->config.stateProofProgram.valid();
        result.deviceName = state->device == nil
            ? std::string{}
            : fromNSString(state->device.name);
    }
    if (result.message.empty()) {
        result.message = status == MetalNumanXHumanMatterHostStatus::success
            ? "NumanX Human/Matter adapter initialized"
            : "NumanX Human/Matter adapter failed";
    }
    return result;
}

[[nodiscard]] bool transactionValid(
    const State& state,
    const MetalNumanXHumanMatterTransaction& transaction
) noexcept {
    std::uint64_t lastEnvironment = 0u;
    // NumiBrain root, substep, motor-candidate, and accepted-physics records
    // are single-environment authorities.  A batched Human/Matter root needs
    // a distinct canonical ABI and a pre-publication all-environment reduce;
    // admitting it here would permit split Human/Matter decisions.
    return transaction.environmentCount == 1u &&
        transaction.environmentCount <= state.config.environmentCapacity &&
        transaction.transactionSlot < state.slots.size() &&
        transaction.physicsSubsteps == 1u &&
        transaction.physicsSubstep == 0u &&
        transaction.expectedMatterCompletedMicrosteps != 0u &&
        transaction.dofLayoutVersion ==
            kMetalNumanXHumanMatterDofLayoutVersion &&
        transaction.dofCount != 0u &&
        transaction.dofCount <= MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
        transaction.qCoordinateCount == transaction.dofCount + 1u &&
        transaction.qCoordinateCount <= MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
        transaction.reserved0 == 0u &&
        transaction.transactionFingerprint != 0u &&
        transaction.substepFingerprint != 0u &&
        transaction.acceptedTimestampMicroseconds != 0u &&
        transaction.physicsGeneration != 0u &&
        transaction.linearizationEpoch != 0u &&
        transaction.slotGeneration != 0u &&
        checkedAdd(
            transaction.environmentIdentifierBase,
            transaction.environmentCount - 1u,
            lastEnvironment) &&
        lastEnvironment <= std::numeric_limits<std::uint32_t>::max();
}

[[nodiscard]] MRNumanXHumanMatterAdapterDispatchGPU makeDispatch(
    const State& state,
    const Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    const auto& transaction = slot.transaction;
    MRNumanXHumanMatterAdapterDispatchGPU dispatch{};
    dispatch.abiVersion = MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION;
    dispatch.environmentCount = transaction.environmentCount;
    dispatch.expectedMatterCompletedMicrosteps =
        transaction.expectedMatterCompletedMicrosteps;
    dispatch.matterSuccessCode = NM_STATUS_SUCCESS;
    dispatch.jointStatusStride = slot.coupledArena.jointStatusStride;
    narrowU32(pass.jointStatusStride, dispatch.jointStatusStride);
    dispatch.standStatusStride = 1u;
    dispatch.matterOutcomeStride = 1u;
    dispatch.acceptedTokenStrideBytes =
        MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
    dispatch.worldStatusStride = 1u;
    dispatch.acceptedStateProofStride = 1u;
    dispatch.environmentIdentifierBase =
        transaction.environmentIdentifierBase;
    dispatch.controlStep = transaction.controlStep;
    dispatch.physicsSubstep = transaction.physicsSubstep;
    dispatch.physicsSubsteps = transaction.physicsSubsteps;
    dispatch.programFingerprint = state.fingerprint;
    dispatch.transactionFingerprint =
        transaction.transactionFingerprint;
    dispatch.substepFingerprint = transaction.substepFingerprint;
    dispatch.acceptedTimestampMicroseconds =
        transaction.acceptedTimestampMicroseconds;
    dispatch.physicsGeneration = transaction.physicsGeneration;
    dispatch.linearizationEpoch = transaction.linearizationEpoch;
    dispatch.slotGeneration = transaction.slotGeneration;
    dispatch.matterSourcePhysicsFingerprint =
        state.matterSourceFingerprint;
    dispatch.matterDeviceProgramFingerprint =
        state.matterDeviceFingerprint;
    dispatch.stateProofProgramFingerprint =
        state.config.stateProofProgram.valid()
            ? state.config.stateProofProgram.fingerprint
            : 0u;
    return dispatch;
}

void dispatchEnvironments(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const std::uint32_t environmentCount
) {
    const NSUInteger width = std::min<NSUInteger>(
        std::max<NSUInteger>(environmentCount, 1u),
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 64u));
    [encoder dispatchThreads:MTLSizeMake(environmentCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

[[nodiscard]] bool encodePrepareWorldStatus(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    const auto dispatch = makeDispatch(state, slot, pass);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) return false;
    encoder.label = @"NumanX Human/Matter prepare world status";
    [encoder setComputePipelineState:state.prepareWorldStatusPipeline];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBuffer:slot.worldStatuses offset:0u atIndex:1u];
    dispatchEnvironments(
        encoder, state.prepareWorldStatusPipeline,
        slot.transaction.environmentCount);
    [encoder endEncoding];
    return true;
}

[[nodiscard]] bool encodeMapHumanStatus(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    const auto dispatch = makeDispatch(state, slot, pass);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) return false;
    encoder.label = @"NumanX Human/Matter map Human status";
    [encoder setComputePipelineState:state.mapHumanStatusPipeline];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.standStatuses
                 offset:0u atIndex:1u];
    [encoder setBuffer:slot.worldStatuses offset:0u atIndex:2u];
    dispatchEnvironments(
        encoder, state.mapHumanStatusPipeline,
        slot.transaction.environmentCount);
    [encoder endEncoding];
    return true;
}

[[nodiscard]] bool encodeCaptureOutcome(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    __unsafe_unretained id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)state.config.matterRuntime->statusBuffer();
    const auto dispatch = makeDispatch(state, slot, pass);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil || matterStatuses == nil) return false;
    encoder.label = @"NumanX Human/Matter capture Matter outcome";
    [encoder setComputePipelineState:state.captureOutcomePipeline];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBuffer:matterStatuses offset:0u atIndex:1u];
    [encoder setBuffer:slot.matterOutcomes offset:0u atIndex:2u];
    dispatchEnvironments(
        encoder, state.captureOutcomePipeline,
        slot.transaction.environmentCount);
    [encoder endEncoding];
    return true;
}

[[nodiscard]] bool encodePreparedToken(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    const auto dispatch = makeDispatch(state, slot, pass);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) return false;
    encoder.label = @"NumanX Human/Matter accepted-state proof gate";
    [encoder setComputePipelineState:state.preparedTokenPipeline];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)pass.jointStatuses
                 offset:0u atIndex:1u];
    [encoder setBuffer:slot.acceptedStateProofs offset:0u atIndex:2u];
    [encoder setBuffer:slot.acceptedTokens offset:0u atIndex:3u];
    dispatchEnvironments(
        encoder, state.preparedTokenPipeline,
        slot.transaction.environmentCount);
    [encoder endEncoding];
    return true;
}

[[nodiscard]] std::uint64_t passSignature(
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    std::uint64_t hash = kFNVOffset;
    const std::array<void*, 18u> pointers{{
        pass.commandBuffer, pass.q, pass.v, pass.mujocoStates,
        pass.mujocoGeneralizedForceArena, pass.bodyPoses, pass.pointQueries,
        pass.pointWorld, pass.pointJacobians, pass.standStatuses,
        pass.qCheckpoint, pass.vCheckpoint, pass.mujocoStateCheckpoint,
        pass.sourceEffectiveTangentFactor, pass.ownerStatuses,
        pass.matterGeneralizedReaction, pass.jointStatuses,
        pass.acceptedPhysicsStateTokens,
    }};
    for (void* pointer : pointers) {
        const std::uintptr_t value = reinterpret_cast<std::uintptr_t>(pointer);
        mixValue(hash, static_cast<std::uint64_t>(value));
    }
    const std::array<std::uint64_t, 38u> values{{
        pass.qGPUAddress, pass.vGPUAddress,
        pass.mujocoStatesGPUAddress,
        pass.mujocoGeneralizedForceArenaGPUAddress,
        pass.bodyPosesGPUAddress, pass.pointQueriesGPUAddress,
        pass.pointWorldGPUAddress, pass.pointJacobiansGPUAddress,
        pass.standStatusesGPUAddress, pass.qCheckpointGPUAddress,
        pass.vCheckpointGPUAddress, pass.mujocoStateCheckpointGPUAddress,
        pass.sourceEffectiveTangentFactorGPUAddress,
        pass.ownerStatusesGPUAddress,
        pass.matterGeneralizedReactionGPUAddress,
        pass.jointStatusesGPUAddress,
        pass.acceptedPhysicsStateTokensGPUAddress,
        pass.environmentCount, pass.qCoordinateCount, pass.dofCount,
        pass.bodyCount, pass.pointCount, pass.mujocoStateCount,
        pass.qStride, pass.vStride, pass.bodyPoseStride, pass.pointStride,
        pass.pointWorldStride, pass.pointJacobianStride,
        pass.mujocoStateStride, pass.factorStride,
        pass.generalizedForceOffset, pass.generalizedForceStride,
        pass.generalizedForceArenaElementCount, pass.reactionStride,
        pass.jointStatusStride, pass.acceptedTokenStrideBytes,
        pass.bodyJacobianPointOffset,
    }};
    for (const auto value : values) mixValue(hash, value);
    mixValue(hash, pass.articulationIndex);
    mixValue(hash, pass.articulationFirstBody);
    mixValue(hash, pass.substepIndex);
    mixValue(hash, pass.physicsSubstepCount);
    mixValue(hash, pass.controlStep);
    mixValue(hash, pass.reserved0);
    mixValue(hash, pass.programFingerprint);
    mixValue(hash, pass.transactionFingerprint);
    mixValue(hash, pass.linearizationEpoch);
    mixValue(hash, pass.slotGeneration);
    return nonzeroHash(hash);
}

[[nodiscard]] bool validatePass(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass,
    const MetalNumanXHumanMatterPhase expectedPhase,
    const bool compareSignature
) noexcept {
    constexpr std::uint32_t capabilities =
        MetalNumanXHumanMatterExactCandidateKinematics |
        MetalNumanXHumanMatterSourceEffectiveTangent |
        MetalNumanXHumanMatterStagedReaction |
        MetalNumanXHumanMatterJointDecision |
        MetalNumanXHumanMatterPreparedPhysicsGate;
    constexpr std::uint32_t access =
        MetalNumanXHumanMatterReadLiveHumanState |
        MetalNumanXHumanMatterReadHumanCheckpoints |
        MetalNumanXHumanMatterReadSourceEffectiveTangent |
        MetalNumanXHumanMatterMayEncodeExactCandidate |
        MetalNumanXHumanMatterWriteStagedReaction |
        MetalNumanXHumanMatterWriteJointStatus |
        MetalNumanXHumanMatterWritePreparedPhysicsToken;
    const bool exactCallbackValid =
        expectedPhase == MetalNumanXHumanMatterPhase::preDynamics
            ? pass.encodeExactCandidate != nullptr &&
                pass.exactCandidateContext != nullptr
            : pass.encodeExactCandidate == nullptr &&
                pass.exactCandidateContext == nullptr;
    if (pass.abiVersion != kMetalNumanXHumanMatterABIVersion ||
        pass.structSize != sizeof(MetalNumanXHumanMatterPass) ||
        pass.accessFlags != access || pass.capabilities != capabilities ||
        pass.phase != expectedPhase || pass.stepIndex != 0u ||
        pass.stepCount != 1u ||
        pass.transactionSlot != slot.transaction.transactionSlot ||
        pass.substepIndex != slot.transaction.physicsSubstep ||
        pass.physicsSubstepCount != slot.transaction.physicsSubsteps ||
        pass.controlStep != slot.transaction.controlStep ||
        pass.reserved0 != 0u ||
        pass.environmentCount != slot.transaction.environmentCount ||
        pass.qCoordinateCount != slot.transaction.qCoordinateCount ||
        pass.dofCount != slot.transaction.dofCount ||
        pass.qStride != pass.qCoordinateCount ||
        pass.vStride != pass.dofCount ||
        pass.factorStride != pass.dofCount * pass.dofCount ||
        pass.reactionStride < pass.dofCount ||
        pass.reactionStride != slot.coupledArena.reactionStride ||
        pass.jointStatusStride != slot.coupledArena.jointStatusStride ||
        pass.acceptedTokenStrideBytes !=
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES ||
        !(pass.timestepSeconds > 0.0f) ||
        !std::isfinite(pass.timestepSeconds) ||
        pass.programFingerprint != state.fingerprint ||
        pass.transactionFingerprint !=
            slot.transaction.transactionFingerprint ||
        pass.linearizationEpoch != slot.transaction.linearizationEpoch ||
        pass.slotGeneration != slot.transaction.slotGeneration ||
        pass.matterGeneralizedReaction !=
            slot.coupledArena.matterGeneralizedReaction ||
        pass.jointStatuses != slot.coupledArena.jointStatuses ||
        pass.acceptedPhysicsStateTokens !=
            (__bridge void*)slot.acceptedTokens ||
        pass.matterGeneralizedReactionGPUAddress !=
            slot.coupledArena.reactionGPUAddress ||
        pass.jointStatusesGPUAddress !=
            slot.coupledArena.jointStatusGPUAddress ||
        pass.acceptedPhysicsStateTokensGPUAddress !=
            slot.acceptedTokens.gpuAddress ||
        !exactCallbackValid ||
        !validCommandBuffer(state.device, pass.commandBuffer)) return false;

    std::uint64_t qElements = 0u, vElements = 0u, bodyElements = 0u;
    std::uint64_t pointElements = 0u, pointWorldElements = 0u;
    std::uint64_t jacobianElements = 0u;
    std::uint64_t stateElements = 0u, factorElements = 0u;
    std::uint64_t standElements = 0u, ownerElements = 0u;
    if (!checkedMultiply(pass.environmentCount, pass.qStride, qElements) ||
        !checkedMultiply(pass.environmentCount, pass.vStride, vElements) ||
        !checkedMultiply(
            pass.environmentCount, pass.bodyPoseStride, bodyElements) ||
        !checkedMultiply(
            pass.environmentCount, pass.pointStride, pointElements) ||
        !checkedMultiply(
            pass.environmentCount, pass.pointWorldStride,
            pointWorldElements) ||
        !checkedMultiply(
            pass.environmentCount, pass.pointJacobianStride,
            jacobianElements) ||
        !checkedMultiply(
            pass.environmentCount, pass.mujocoStateStride, stateElements) ||
        !checkedMultiply(
            pass.environmentCount, pass.factorStride, factorElements) ||
        !checkedMultiply(
            pass.environmentCount - 1u, 1u,
            standElements) ||
        !checkedAdd(standElements, 1u, standElements) ||
        !checkedMultiply(pass.environmentCount, 1u, ownerElements)) {
        return false;
    }
    // Object inequality is not an alias proof: distinct MTLBuffer resources
    // created from one heap may name overlapping GPU virtual-address ranges.
    // Admit every range only after checked byte arithmetic, then reject every
    // overlap.  This is deliberately stricter than the individual kernel
    // access modes: checkpoints, live destinations, owner status, staged
    // reaction, joint status, proof scratch and the prepared token form one
    // rollback/proof authority and must never share bytes.
    std::array<BufferRegion, 21u> regions{};
    std::size_t regionCount = 0u;
    const auto appendBuffer = [&] (
        void* raw, const std::uint64_t address,
        const std::uint64_t elements, const std::uint64_t elementBytes
    ) -> bool {
        std::uint64_t bytes = 0u;
        if (regionCount >= regions.size() ||
            !checkedMultiply(elements, elementBytes, bytes) ||
            !bufferRegion(
                state.device, raw, address, bytes, regions[regionCount])) {
            return false;
        }
        ++regionCount;
        return true;
    };
    const auto appendExactBytes = [&] (
        void* raw, const std::uint64_t address, const std::uint64_t bytes
    ) -> bool {
        if (regionCount >= regions.size() ||
            !bufferRegion(
                state.device, raw, address, bytes, regions[regionCount])) {
            return false;
        }
        ++regionCount;
        return true;
    };
    const auto appendOwned = [&] (id<MTLBuffer> buffer) -> bool {
        if (buffer == nil) return false;
        return appendExactBytes(
            (__bridge void*)buffer,
            static_cast<std::uint64_t>(buffer.gpuAddress),
            static_cast<std::uint64_t>(buffer.length));
    };
    if (!appendBuffer(pass.q, pass.qGPUAddress, qElements, sizeof(float)) ||
        !appendBuffer(pass.v, pass.vGPUAddress, vElements, sizeof(float)) ||
        !appendBuffer(pass.mujocoStates, pass.mujocoStatesGPUAddress,
                stateElements, sizeof(MRMujocoMuscleStateGPU)) ||
        !appendBuffer(pass.mujocoGeneralizedForceArena,
                pass.mujocoGeneralizedForceArenaGPUAddress,
                pass.generalizedForceArenaElementCount, sizeof(float)) ||
        !appendBuffer(pass.bodyPoses, pass.bodyPosesGPUAddress,
                bodyElements, sizeof(MRArticulatedBodyPoseGPU)) ||
        !appendBuffer(pass.pointQueries, pass.pointQueriesGPUAddress,
                pointElements, sizeof(MRArticulatedPointImpulseGPU)) ||
        !appendBuffer(pass.pointWorld, pass.pointWorldGPUAddress,
                pointWorldElements,
                sizeof(MRArticulatedPointWorldGPU)) ||
        !appendBuffer(pass.pointJacobians, pass.pointJacobiansGPUAddress,
                jacobianElements, sizeof(float)) ||
        !appendBuffer(pass.standStatuses, pass.standStatusesGPUAddress,
                standElements, sizeof(MRNumiHumanStandStatusGPU)) ||
        !appendBuffer(pass.qCheckpoint, pass.qCheckpointGPUAddress,
                qElements, sizeof(float)) ||
        !appendBuffer(pass.vCheckpoint, pass.vCheckpointGPUAddress,
                vElements, sizeof(float)) ||
        !appendBuffer(pass.mujocoStateCheckpoint,
                pass.mujocoStateCheckpointGPUAddress,
                stateElements, sizeof(MRMujocoMuscleStateGPU)) ||
        !appendBuffer(pass.sourceEffectiveTangentFactor,
                pass.sourceEffectiveTangentFactorGPUAddress,
                factorElements, sizeof(float)) ||
        !appendBuffer(pass.ownerStatuses, pass.ownerStatusesGPUAddress,
                ownerElements, sizeof(MRNumanXHumanMatterOwnerStatusGPU)) ||
        !appendExactBytes(
            pass.matterGeneralizedReaction,
            pass.matterGeneralizedReactionGPUAddress,
            slot.coupledArena.reactionByteCount) ||
        !appendExactBytes(
            pass.jointStatuses, pass.jointStatusesGPUAddress,
            slot.coupledArena.jointStatusByteCount) ||
        !appendExactBytes(
            pass.acceptedPhysicsStateTokens,
            pass.acceptedPhysicsStateTokensGPUAddress,
            static_cast<std::uint64_t>(slot.acceptedTokens.length)) ||
        !appendOwned(slot.matterOutcomes) ||
        !appendOwned(slot.worldStatuses) ||
        !appendOwned(slot.acceptedStateProofs) ||
        !appendOwned((__bridge id<MTLBuffer>)
            state.config.matterRuntime->statusBuffer())) {
        return false;
    }
    for (std::size_t left = 0u; left < regionCount; ++left) {
        for (std::size_t right = left + 1u; right < regionCount; ++right) {
            if (regionsOverlap(regions[left], regions[right])) return false;
        }
    }
    if (!compareSignature) {
        if (!authorityRegionsIsolated(
                state, slot, regions.data(),
                slot.physicalAuthorityRegions.size(),
                true, false, false, false)) {
            return false;
        }
        // Freeze only the owner-supplied physical authority ranges. The
        // adapter-owned ranges that follow are validated independently and
        // must never be mistaken for borrowable owner storage.
        static_assert(
            std::tuple_size_v<decltype(slot.physicalAuthorityRegions)> ==
            14u);
        for (std::size_t index = 0u;
             index < slot.physicalAuthorityRegions.size(); ++index) {
            slot.physicalAuthorityRegions[index] = {
                regions[index].address,
                regions[index].bytes,
            };
        }
        slot.physicalAuthorityRegionCount =
            slot.physicalAuthorityRegions.size();
        const std::array<BufferRegion, 8u> leaseRegions{{
            {slot.lease.preparedTokensGPUAddress,
             slot.lease.preparedTokenBytes},
            {slot.lease.proposalsGPUAddress,
             slot.lease.proposalElements * 128u},
            {slot.lease.proposedTokensGPUAddress,
             slot.lease.proposedTokenBytes},
            {slot.lease.applyActionsGPUAddress,
             slot.lease.applyActionElements * 128u},
            {slot.lease.matterApplyOutcomesGPUAddress,
             slot.lease.matterApplyOutcomeElements * 128u},
            {slot.lease.appliedOutcomesGPUAddress,
             slot.lease.appliedOutcomeElements * 128u},
            {slot.lease.finalTokensGPUAddress,
             slot.lease.finalTokenBytes},
            {slot.lease.publicationFencesGPUAddress,
             slot.lease.publicationFenceElements * 128u},
        }};
        for (const auto& authority : slot.physicalAuthorityRegions) {
            const BufferRegion physical{authority.address, authority.bytes};
            for (const BufferRegion& leaseRegion : leaseRegions) {
                if (physical.address == 0u || physical.bytes == 0u ||
                    leaseRegion.address == 0u || leaseRegion.bytes == 0u ||
                    regionsOverlap(physical, leaseRegion)) {
                    return false;
                }
            }
        }
    }
    const std::uint64_t signature = passSignature(pass);
    return !compareSignature || signature == slot.passSignature;
}

[[nodiscard]] detail::MetalNumanXHumanMatterLeaseIdentity leaseIdentity(
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    detail::MetalNumanXHumanMatterLeaseIdentity result{};
    result.preparedTokens = reinterpret_cast<std::uintptr_t>(
        lease.preparedPhysicsStateTokens);
    result.proposals = reinterpret_cast<std::uintptr_t>(lease.proposals);
    result.proposedTokens = reinterpret_cast<std::uintptr_t>(
        lease.proposedPhysicsStateTokens);
    result.applyActions = reinterpret_cast<std::uintptr_t>(lease.applyActions);
    result.matterApplyOutcomes = reinterpret_cast<std::uintptr_t>(
        lease.matterApplyOutcomes);
    result.appliedOutcomes = reinterpret_cast<std::uintptr_t>(
        lease.appliedOutcomes);
    result.finalTokens = reinterpret_cast<std::uintptr_t>(
        lease.finalAcceptedPhysicsStateTokens);
    result.publicationFences = reinterpret_cast<std::uintptr_t>(
        lease.publicationFences);
    result.physicalEvent = reinterpret_cast<std::uintptr_t>(
        lease.physicalPreparedEvent);
    result.preparedTokensGPUAddress =
        lease.preparedPhysicsStateTokensGPUAddress;
    result.proposalsGPUAddress = lease.proposalsGPUAddress;
    result.proposedTokensGPUAddress =
        lease.proposedPhysicsStateTokensGPUAddress;
    result.applyActionsGPUAddress = lease.applyActionsGPUAddress;
    result.matterApplyOutcomesGPUAddress =
        lease.matterApplyOutcomesGPUAddress;
    result.appliedOutcomesGPUAddress = lease.appliedOutcomesGPUAddress;
    result.finalTokensGPUAddress =
        lease.finalAcceptedPhysicsStateTokensGPUAddress;
    result.publicationFencesGPUAddress = lease.publicationFencesGPUAddress;
    result.preparedTokenBytes =
        lease.preparedPhysicsStateTokenByteCount;
    result.proposalElements = lease.proposalElementCount;
    result.proposedTokenBytes = lease.proposedPhysicsStateTokenByteCount;
    result.applyActionElements = lease.applyActionElementCount;
    result.matterApplyOutcomeElements =
        lease.matterApplyOutcomeElementCount;
    result.appliedOutcomeElements = lease.appliedOutcomeElementCount;
    result.finalTokenBytes =
        lease.finalAcceptedPhysicsStateTokenByteCount;
    result.publicationFenceElements = lease.publicationFenceElementCount;
    result.physicalEventValue = lease.physicalPreparedEventValue;
    result.proposalEventValue = lease.proposalEventValue;
    result.appliedEventValue = lease.appliedEventValue;
    result.programFingerprint = lease.programFingerprint;
    result.transactionFingerprint = lease.transactionFingerprint;
    result.linearizationEpoch = lease.linearizationEpoch;
    result.slotGeneration = lease.slotGeneration;
    result.environmentCount = lease.environmentCount;
    result.transactionSlot = lease.transactionSlot;
    result.stepIndex = lease.stepIndex;
    result.substepIndex = lease.substepIndex;
    result.preparedTokenStrideBytes = lease.preparedTokenStrideBytes;
    result.proposalStride = lease.proposalStride;
    result.proposedTokenStrideBytes = lease.proposedTokenStrideBytes;
    result.applyActionStride = lease.applyActionStride;
    result.matterApplyOutcomeStride = lease.matterApplyOutcomeStride;
    result.appliedOutcomeStride = lease.appliedOutcomeStride;
    result.finalTokenStrideBytes = lease.finalTokenStrideBytes;
    result.publicationFenceStride = lease.publicationFenceStride;
    result.physicsSubstepCount = lease.physicsSubstepCount;
    result.controlStep = lease.controlStep;
    result.qCoordinateCount = lease.qCoordinateCount;
    result.dofCount = lease.dofCount;
    result.dofLayoutVersion = lease.dofLayoutVersion;
    result.reservedDofLayout = lease.reservedDofLayout;
    return result;
}

[[nodiscard]] bool sameHumanIOCandidate(
    const MetalNumanXHumanIOCandidatePublicationProgram& expected,
    const MetalNumanXHumanIOCandidatePublicationProgram& actual
) noexcept {
    return expected.abiVersion == actual.abiVersion &&
        expected.structSize == actual.structSize &&
        expected.context == actual.context &&
        expected.reservePublishedRoot == actual.reservePublishedRoot &&
        expected.publishCandidate == actual.publishCandidate &&
        expected.rejectCandidate == actual.rejectCandidate &&
        expected.candidateKeyFingerprint ==
            actual.candidateKeyFingerprint &&
        expected.transactionFingerprint == actual.transactionFingerprint &&
        expected.acceptedBrainGeneration ==
            actual.acceptedBrainGeneration &&
        expected.sensorGeneration == actual.sensorGeneration &&
        expected.humanIOProgramFingerprint ==
            actual.humanIOProgramFingerprint &&
        expected.sensorFingerprint == actual.sensorFingerprint &&
        expected.transactionInstanceFingerprint ==
            actual.transactionInstanceFingerprint &&
        expected.candidatePublicationFingerprint ==
            actual.candidatePublicationFingerprint &&
        expected.deviceRegistryID == actual.deviceRegistryID &&
        expected.identityFingerprint == actual.identityFingerprint;
}

[[nodiscard]] bool sameLease(
    const detail::MetalNumanXHumanMatterLeaseIdentity& expected,
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    const auto actual = leaseIdentity(lease);
#define MR_SAME_LEASE_FIELD(field) expected.field == actual.field
    return
        MR_SAME_LEASE_FIELD(preparedTokens) &&
        MR_SAME_LEASE_FIELD(proposals) &&
        MR_SAME_LEASE_FIELD(proposedTokens) &&
        MR_SAME_LEASE_FIELD(applyActions) &&
        MR_SAME_LEASE_FIELD(matterApplyOutcomes) &&
        MR_SAME_LEASE_FIELD(appliedOutcomes) &&
        MR_SAME_LEASE_FIELD(finalTokens) &&
        MR_SAME_LEASE_FIELD(publicationFences) &&
        MR_SAME_LEASE_FIELD(physicalEvent) &&
        MR_SAME_LEASE_FIELD(preparedTokensGPUAddress) &&
        MR_SAME_LEASE_FIELD(proposalsGPUAddress) &&
        MR_SAME_LEASE_FIELD(proposedTokensGPUAddress) &&
        MR_SAME_LEASE_FIELD(applyActionsGPUAddress) &&
        MR_SAME_LEASE_FIELD(matterApplyOutcomesGPUAddress) &&
        MR_SAME_LEASE_FIELD(appliedOutcomesGPUAddress) &&
        MR_SAME_LEASE_FIELD(finalTokensGPUAddress) &&
        MR_SAME_LEASE_FIELD(publicationFencesGPUAddress) &&
        MR_SAME_LEASE_FIELD(preparedTokenBytes) &&
        MR_SAME_LEASE_FIELD(proposalElements) &&
        MR_SAME_LEASE_FIELD(proposedTokenBytes) &&
        MR_SAME_LEASE_FIELD(applyActionElements) &&
        MR_SAME_LEASE_FIELD(matterApplyOutcomeElements) &&
        MR_SAME_LEASE_FIELD(appliedOutcomeElements) &&
        MR_SAME_LEASE_FIELD(finalTokenBytes) &&
        MR_SAME_LEASE_FIELD(publicationFenceElements) &&
        MR_SAME_LEASE_FIELD(physicalEventValue) &&
        MR_SAME_LEASE_FIELD(proposalEventValue) &&
        MR_SAME_LEASE_FIELD(appliedEventValue) &&
        MR_SAME_LEASE_FIELD(programFingerprint) &&
        MR_SAME_LEASE_FIELD(transactionFingerprint) &&
        MR_SAME_LEASE_FIELD(linearizationEpoch) &&
        MR_SAME_LEASE_FIELD(slotGeneration) &&
        MR_SAME_LEASE_FIELD(environmentCount) &&
        MR_SAME_LEASE_FIELD(transactionSlot) &&
        MR_SAME_LEASE_FIELD(stepIndex) &&
        MR_SAME_LEASE_FIELD(substepIndex) &&
        MR_SAME_LEASE_FIELD(preparedTokenStrideBytes) &&
        MR_SAME_LEASE_FIELD(proposalStride) &&
        MR_SAME_LEASE_FIELD(proposedTokenStrideBytes) &&
        MR_SAME_LEASE_FIELD(applyActionStride) &&
        MR_SAME_LEASE_FIELD(matterApplyOutcomeStride) &&
        MR_SAME_LEASE_FIELD(appliedOutcomeStride) &&
        MR_SAME_LEASE_FIELD(finalTokenStrideBytes) &&
        MR_SAME_LEASE_FIELD(publicationFenceStride) &&
        MR_SAME_LEASE_FIELD(physicsSubstepCount) &&
        MR_SAME_LEASE_FIELD(controlStep) &&
        MR_SAME_LEASE_FIELD(qCoordinateCount) &&
        MR_SAME_LEASE_FIELD(dofCount) &&
        MR_SAME_LEASE_FIELD(dofLayoutVersion) &&
        MR_SAME_LEASE_FIELD(reservedDofLayout);
#undef MR_SAME_LEASE_FIELD
}

[[nodiscard]] bool leaseBufferRegions(
    State& state,
    const MetalNumanXHumanMatterPrepareLease& lease,
    std::array<BufferRegion, 8u>& regions
) noexcept {
    std::uint64_t tokenBytes = 0u;
    std::uint64_t recordBytes = 0u;
    return checkedMultiply(
            lease.environmentCount,
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
            tokenBytes) &&
        checkedMultiply(lease.environmentCount, 128u, recordBytes) &&
        bufferRegion(
            state.device, lease.preparedPhysicsStateTokens,
            lease.preparedPhysicsStateTokensGPUAddress,
            lease.preparedPhysicsStateTokenByteCount, regions[0]) &&
        bufferRegion(
            state.device, lease.proposals, lease.proposalsGPUAddress,
            recordBytes, regions[1]) &&
        bufferRegion(
            state.device, lease.proposedPhysicsStateTokens,
            lease.proposedPhysicsStateTokensGPUAddress,
            tokenBytes, regions[2]) &&
        bufferRegion(
            state.device, lease.applyActions, lease.applyActionsGPUAddress,
            recordBytes, regions[3]) &&
        bufferRegion(
            state.device, lease.matterApplyOutcomes,
            lease.matterApplyOutcomesGPUAddress,
            recordBytes, regions[4]) &&
        bufferRegion(
            state.device, lease.appliedOutcomes,
            lease.appliedOutcomesGPUAddress,
            recordBytes, regions[5]) &&
        bufferRegion(
            state.device, lease.finalAcceptedPhysicsStateTokens,
            lease.finalAcceptedPhysicsStateTokensGPUAddress,
            tokenBytes, regions[6]) &&
        bufferRegion(
            state.device, lease.publicationFences,
            lease.publicationFencesGPUAddress,
            recordBytes, regions[7]);
}

[[nodiscard]] bool aliasesLeaseOrAdapterAuthority(
    State& state,
    const Slot& slot,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const BufferRegion& candidate
) noexcept {
    if (!authorityRegionsIsolated(
            state, slot, &candidate, 1u,
            false, false, false, false)) {
        return true;
    }
    std::array<BufferRegion, 8u> leaseRegions{};
    if (!leaseBufferRegions(state, lease, leaseRegions)) return true;
    for (const BufferRegion& region : leaseRegions) {
        if (regionsOverlap(candidate, region)) return true;
    }
    for (const auto& authority : slot.physicalAuthorityRegions) {
        if (authority.bytes != 0u && regionsOverlap(
                candidate, {authority.address, authority.bytes})) {
            return true;
        }
    }
    __unsafe_unretained id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)state.config.matterRuntime->statusBuffer();
    const std::array<id<MTLBuffer>, 4u> protectedBuffers{{
        slot.matterOutcomes,
        slot.worldStatuses,
        slot.acceptedStateProofs,
        matterStatuses,
    }};
    for (id<MTLBuffer> buffer : protectedBuffers) {
        BufferRegion region{};
        if (buffer == nil || !bufferRegion(
                state.device, (__bridge void*)buffer, buffer.gpuAddress,
                buffer.length, region) || regionsOverlap(candidate, region)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool validLeaseResources(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    std::uint64_t tokenBytes = 0u;
    std::uint64_t recordBytes = 0u;
    if (lease.abiVersion != kMetalNumanXHumanMatterABIVersion ||
        lease.structSize != sizeof(MetalNumanXHumanMatterPrepareLease) ||
        lease.environmentCount != slot.transaction.environmentCount ||
        lease.environmentCount != 1u ||
        lease.transactionSlot != slot.transaction.transactionSlot ||
        lease.stepIndex != 0u ||
        lease.substepIndex != slot.transaction.physicsSubstep ||
        lease.physicsSubstepCount != slot.transaction.physicsSubsteps ||
        lease.controlStep != slot.transaction.controlStep ||
        lease.qCoordinateCount != slot.transaction.qCoordinateCount ||
        lease.dofCount != slot.transaction.dofCount ||
        lease.dofLayoutVersion !=
            kMetalNumanXHumanMatterDofLayoutVersion ||
        lease.reservedDofLayout != 0u ||
        lease.preparedTokenStrideBytes !=
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES ||
        lease.proposalStride != 1u ||
        lease.proposedTokenStrideBytes !=
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES ||
        lease.applyActionStride != 1u ||
        lease.matterApplyOutcomeStride != 1u ||
        lease.appliedOutcomeStride != 1u ||
        lease.finalTokenStrideBytes !=
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES ||
        lease.publicationFenceStride != 1u ||
        lease.preparedPhysicsStateTokens !=
            (__bridge void*)slot.acceptedTokens ||
        lease.preparedPhysicsStateTokensGPUAddress !=
            slot.acceptedTokens.gpuAddress ||
        lease.preparedPhysicsStateTokenByteCount !=
            slot.acceptedTokens.length ||
        lease.matterApplyOutcomes !=
            (__bridge void*)slot.matterApplyOutcomes ||
        lease.matterApplyOutcomesGPUAddress !=
            slot.matterApplyOutcomes.gpuAddress ||
        lease.matterApplyOutcomeElementCount != 1u ||
        lease.programFingerprint != state.fingerprint ||
        lease.transactionFingerprint !=
            slot.transaction.transactionFingerprint ||
        lease.linearizationEpoch != slot.transaction.linearizationEpoch ||
        lease.slotGeneration != slot.transaction.slotGeneration ||
        lease.physicalPreparedEventValue == 0u ||
        lease.physicalPreparedEventValue >
            std::numeric_limits<std::uint64_t>::max() - 2u ||
        lease.proposalEventValue !=
            lease.physicalPreparedEventValue + 1u ||
        lease.appliedEventValue != lease.proposalEventValue + 1u ||
        lease.proposalElementCount != 1u ||
        lease.applyActionElementCount != 1u ||
        lease.appliedOutcomeElementCount != 1u ||
        lease.publicationFenceElementCount != 1u ||
        !checkedMultiply(
            lease.environmentCount,
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES, tokenBytes) ||
        lease.finalAcceptedPhysicsStateTokenByteCount != tokenBytes ||
        lease.proposedPhysicsStateTokenByteCount != tokenBytes ||
        !exactSharedEvent(state.device, lease.physicalPreparedEvent)) {
        return false;
    }
    if (slot.humanIOCandidateBound) {
        if (!slot.humanIOCandidate.valid() ||
            !sameHumanIOCandidate(
                slot.humanIOCandidate,
                lease.humanIOCandidate)) {
            return false;
        }
    } else if (lease.humanIOCandidate.configured()) {
        return false;
    }

    if (!checkedMultiply(lease.environmentCount, 128u, recordBytes)) {
        return false;
    }
    std::array<BufferRegion, 8u> leaseRegions{};
    if (!bufferRegion(
            state.device, lease.preparedPhysicsStateTokens,
            lease.preparedPhysicsStateTokensGPUAddress,
            lease.preparedPhysicsStateTokenByteCount, leaseRegions[0]) ||
        !bufferRegion(
            state.device, lease.proposals, lease.proposalsGPUAddress,
            recordBytes, leaseRegions[1]) ||
        !bufferRegion(
            state.device, lease.proposedPhysicsStateTokens,
            lease.proposedPhysicsStateTokensGPUAddress,
            tokenBytes, leaseRegions[2]) ||
        !bufferRegion(
            state.device, lease.applyActions, lease.applyActionsGPUAddress,
            recordBytes, leaseRegions[3]) ||
        !bufferRegion(
            state.device, lease.matterApplyOutcomes,
            lease.matterApplyOutcomesGPUAddress,
            recordBytes, leaseRegions[4]) ||
        !bufferRegion(
            state.device, lease.appliedOutcomes,
            lease.appliedOutcomesGPUAddress,
            recordBytes, leaseRegions[5]) ||
        !bufferRegion(
            state.device, lease.finalAcceptedPhysicsStateTokens,
            lease.finalAcceptedPhysicsStateTokensGPUAddress,
            tokenBytes, leaseRegions[6]) ||
        !bufferRegion(
            state.device, lease.publicationFences,
            lease.publicationFencesGPUAddress,
            recordBytes, leaseRegions[7]) ||
        !exactHostVisibleBuffer(
            state.device, lease.proposals, lease.proposalsGPUAddress,
            recordBytes) ||
        !exactHostVisibleBuffer(
            state.device, lease.proposedPhysicsStateTokens,
            lease.proposedPhysicsStateTokensGPUAddress, tokenBytes) ||
        !exactHostVisibleBuffer(
            state.device, lease.appliedOutcomes,
            lease.appliedOutcomesGPUAddress, recordBytes) ||
        !exactHostVisibleBuffer(
            state.device, lease.finalAcceptedPhysicsStateTokens,
            lease.finalAcceptedPhysicsStateTokensGPUAddress, tokenBytes) ||
        !exactHostVisibleBuffer(
            state.device, lease.publicationFences,
            lease.publicationFencesGPUAddress, recordBytes)) {
        return false;
    }
    for (std::size_t left = 0u; left < leaseRegions.size(); ++left) {
        for (std::size_t right = left + 1u;
             right < leaseRegions.size(); ++right) {
            if (regionsOverlap(leaseRegions[left], leaseRegions[right])) {
                return false;
            }
        }
    }
    if (!authorityRegionsIsolated(
            state, slot, leaseRegions.data(), leaseRegions.size(),
            false, true, false, true)) {
        return false;
    }
    for (const auto& authority : slot.physicalAuthorityRegions) {
        if (authority.bytes == 0u) continue;
        const BufferRegion region{authority.address, authority.bytes};
        for (const BufferRegion& leaseRegion : leaseRegions) {
            // The prepared token and Matter outcome are exact adapter-owned
            // arenas already present in the physical authority table.
            if ((leaseRegion.address == leaseRegions[0].address &&
                 leaseRegion.bytes == leaseRegions[0].bytes) ||
                (leaseRegion.address == leaseRegions[4].address &&
                 leaseRegion.bytes == leaseRegions[4].bytes)) {
                continue;
            }
            if (regionsOverlap(region, leaseRegion)) return false;
        }
    }
    std::array<BufferRegion, 4u> adapterProtected{};
    __unsafe_unretained id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)state.config.matterRuntime->statusBuffer();
    const std::array<id<MTLBuffer>, 4u> protectedBuffers{{
        slot.matterOutcomes,
        slot.worldStatuses,
        slot.acceptedStateProofs,
        matterStatuses,
    }};
    for (std::size_t index = 0u; index < protectedBuffers.size(); ++index) {
        id<MTLBuffer> buffer = protectedBuffers[index];
        if (buffer == nil || !bufferRegion(
                state.device, (__bridge void*)buffer, buffer.gpuAddress,
                buffer.length, adapterProtected[index])) {
            return false;
        }
        for (const BufferRegion& leaseRegion : leaseRegions) {
            if (regionsOverlap(adapterProtected[index], leaseRegion)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool exactProposalView(
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterProposalView& proposal
) noexcept {
    return proposal.abiVersion == kMetalNumanXHumanMatterABIVersion &&
        proposal.structSize == sizeof(proposal) &&
        proposal.proposals == lease.proposals &&
        proposal.proposedPhysicsStateTokens ==
            lease.proposedPhysicsStateTokens &&
        proposal.proposalsGPUAddress == lease.proposalsGPUAddress &&
        proposal.proposedPhysicsStateTokensGPUAddress ==
            lease.proposedPhysicsStateTokensGPUAddress &&
        proposal.proposalElementCount == lease.proposalElementCount &&
        proposal.proposedPhysicsStateTokenByteCount ==
            lease.proposedPhysicsStateTokenByteCount &&
        proposal.proposalStride == lease.proposalStride &&
        proposal.proposedTokenStrideBytes ==
            lease.proposedTokenStrideBytes &&
        proposal.proposalEventValue == lease.proposalEventValue;
}

[[nodiscard]] bool exactFenceView(
    const Slot& slot,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterPublicationFenceView& fence
) noexcept {
    return fence.abiVersion == kMetalNumanXHumanMatterABIVersion &&
        fence.structSize == sizeof(fence) &&
        fence.publicationFences == lease.publicationFences &&
        fence.publicationFencesGPUAddress ==
            lease.publicationFencesGPUAddress &&
        fence.publicationFenceElementCount ==
            lease.publicationFenceElementCount &&
        fence.publicationFenceStride == lease.publicationFenceStride &&
        fence.environmentCount == slot.transaction.environmentCount &&
        fence.transactionSlot == slot.transaction.transactionSlot &&
        fence.stepIndex == 0u &&
        fence.substepIndex == slot.transaction.physicsSubstep &&
        fence.physicsSubstepCount == slot.transaction.physicsSubsteps &&
        fence.controlStep == slot.transaction.controlStep &&
        fence.programFingerprint == lease.programFingerprint &&
        fence.transactionFingerprint == lease.transactionFingerprint &&
        fence.linearizationEpoch == lease.linearizationEpoch &&
        fence.slotGeneration == lease.slotGeneration;
}

[[nodiscard]] bool readyProposalRejectCode(
    const std::uint32_t code
) noexcept {
    return code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT ||
        code ==
            MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT ||
        code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT;
}

[[nodiscard]] std::uint32_t appliedCodeForProposalReject(
    const std::uint32_t code
) noexcept {
    switch (code) {
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_PHYSICAL_REJECT:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_PHYSICAL_REJECT;
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_BRAIN_WITNESS:
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_BRAIN_REJECT:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_BRAIN_REJECT;
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_TOKEN_MISMATCH:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_TOKEN_MISMATCH;
    case MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCED_REJECT:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT;
    default:
        return MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER;
    }
}

[[nodiscard]] bool validProposalRecord(
    const State& state,
    const Slot& slot,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const void* proposedToken
) noexcept {
    const bool accept =
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
        proposal.code == MR_NUMANX_HUMAN_MATTER_PROPOSAL_SUCCESS &&
        proposal.physicsTokenFingerprint != 0u &&
        proposal.brainProgramFingerprint != 0u &&
        proposal.brainShadowStateFingerprint != 0u &&
        proposal.brainWitnessFingerprint != 0u;
    const bool reject =
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
        readyProposalRejectCode(proposal.code) &&
        proposal.physicsTokenFingerprint == 0u &&
        proposal.brainProgramFingerprint == 0u &&
        proposal.brainShadowStateFingerprint == 0u &&
        proposal.brainWitnessFingerprint == 0u;
    return proposal.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        proposal.status == MR_NUMANX_HUMAN_MATTER_PROPOSAL_READY &&
        (accept || reject) &&
        proposal.programFingerprint == state.fingerprint &&
        proposal.transactionFingerprint ==
            slot.transaction.transactionFingerprint &&
        proposal.linearizationEpoch == slot.transaction.linearizationEpoch &&
        proposal.slotGeneration == slot.transaction.slotGeneration &&
        proposal.environment == 0u && proposal.stepIndex == 0u &&
        proposal.substepIndex == slot.transaction.physicsSubstep &&
        proposal.transactionSlot == slot.transaction.transactionSlot &&
        proposal.physicsSubstepCount == slot.transaction.physicsSubsteps &&
        proposal.controlStep == slot.transaction.controlStep &&
        slot.humanIOCandidateBound && slot.humanIOCandidate.valid() &&
        proposal.candidatePublicationFingerprint ==
            slot.humanIOCandidate.candidatePublicationFingerprint &&
        proposal.humanIOIdentityFingerprint ==
            slot.humanIOCandidate.identityFingerprint &&
        proposal.proposalFingerprint != 0u &&
        proposal.proposalFingerprint ==
            humanMatterRecordFingerprint(&proposal) &&
        (accept
             ? validAcceptedPhysicsToken(
                   proposedToken,
                   slot.transaction.transactionFingerprint,
                   proposal.physicsTokenFingerprint)
             : zeroBytes(
                   proposedToken,
                   MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES));
}

[[nodiscard]] bool validAppliedRecord(
    const State& state,
    const Slot& slot,
    const MRNumanXHumanMatterAppliedOutcomeGPU& applied,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const void* proposedToken,
    const void* finalToken
) noexcept {
    const bool accept = applied.status ==
            MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED &&
        applied.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
        applied.code == MR_NUMANX_HUMAN_MATTER_APPLIED_SUCCESS &&
        applied.physicsTokenFingerprint != 0u;
    const bool forcedNoAckClosure =
        (applied.code == MR_NUMANX_HUMAN_MATTER_APPLIED_FORCED_REJECT &&
         applied.physicsTokenFingerprint == 0u &&
         applied.ackFingerprint == 0u &&
         applied.preflightFingerprint == 0u &&
         applied.fastGateFingerprint == 0u);
    const bool proposalRejectClosure =
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
        readyProposalRejectCode(proposal.code) &&
        applied.code == appliedCodeForProposalReject(proposal.code) &&
        applied.physicsTokenFingerprint == 0u &&
        applied.ackFingerprint != 0u &&
        applied.preflightFingerprint == 0u &&
        applied.fastGateFingerprint == 0u;
    const bool matterRejectClosure =
        proposal.decision == MR_NUMANX_HUMAN_MATTER_ROOT_ACCEPT &&
        applied.code == MR_NUMANX_HUMAN_MATTER_APPLIED_MATTER_REJECT &&
        applied.physicsTokenFingerprint ==
            proposal.physicsTokenFingerprint &&
        applied.ackFingerprint != 0u &&
         applied.preflightFingerprint != 0u &&
        applied.fastGateFingerprint != 0u;
    const bool reject = applied.status ==
            MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED &&
        applied.decision == MR_NUMANX_HUMAN_MATTER_ROOT_REJECT &&
        (forcedNoAckClosure || proposalRejectClosure ||
         matterRejectClosure);
    const bool closureBound = accept
        ? applied.ackFingerprint != 0u &&
            applied.preflightFingerprint != 0u &&
            applied.fastGateFingerprint != 0u
        : reject;
    return validProposalRecord(state, slot, proposal, proposedToken) &&
        applied.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        (accept || reject) &&
        applied.programFingerprint == state.fingerprint &&
        applied.transactionFingerprint ==
            slot.transaction.transactionFingerprint &&
        applied.linearizationEpoch == slot.transaction.linearizationEpoch &&
        applied.slotGeneration == slot.transaction.slotGeneration &&
        applied.proposalFingerprint == proposal.proposalFingerprint &&
        closureBound &&
        applied.matterApplyFingerprint != 0u &&
        applied.environment == 0u && applied.stepIndex == 0u &&
        applied.substepIndex == slot.transaction.physicsSubstep &&
        applied.transactionSlot == slot.transaction.transactionSlot &&
        applied.physicsSubstepCount == slot.transaction.physicsSubsteps &&
        applied.controlStep == slot.transaction.controlStep &&
        applied.appliedFingerprint != 0u &&
        applied.appliedFingerprint == humanMatterRecordFingerprint(&applied) &&
        (accept
             ? (applied.physicsTokenFingerprint ==
                    proposal.physicsTokenFingerprint &&
                validAcceptedPhysicsToken(
                    finalToken,
                    slot.transaction.transactionFingerprint,
                    applied.physicsTokenFingerprint) &&
                std::memcmp(
                    proposedToken, finalToken,
                    MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES) == 0)
             : zeroBytes(
                   finalToken,
                   MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES));
}

[[nodiscard]] bool validPublicationFenceRecord(
    const State& state,
    const Slot& slot,
    const MRNumanXHumanMatterJointPublicationFenceGPU& fence,
    const numi::matter::PreparedStatePublicationBinding& binding,
    const std::uint32_t expectedStatus
) noexcept {
    return fence.abiVersion ==
            MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION &&
        fence.structBytes ==
            MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES &&
        fence.status == expectedStatus && fence.environment == 0u &&
        fence.controlStep == slot.transaction.controlStep &&
        fence.substepIndex == slot.transaction.physicsSubstep &&
        fence.physicsSubstepCount == slot.transaction.physicsSubsteps &&
        fence.reserved0 == 0u &&
        fence.ownerProgramFingerprint == state.fingerprint &&
        fence.transactionFingerprint ==
            slot.transaction.transactionFingerprint &&
        fence.linearizationEpoch == slot.transaction.linearizationEpoch &&
        fence.slotGeneration == slot.transaction.slotGeneration &&
        fence.physicsTokenFingerprint == binding.physicsTokenFingerprint &&
        fence.brainProgramFingerprint == binding.brainProgramFingerprint &&
        fence.brainShadowStateFingerprint ==
            binding.brainShadowStateFingerprint &&
        fence.brainWitnessFingerprint ==
            binding.brainWitnessFingerprint &&
        fence.appliedDecisionFingerprint ==
            binding.appliedDecisionFingerprint &&
        fence.jointCommitFingerprint == binding.jointCommitFingerprint &&
        fence.brainGeneration == binding.brainGeneration &&
        fence.fenceFingerprint != 0u &&
        fence.fenceFingerprint == humanMatterRecordFingerprint(&fence);
}

[[nodiscard]] bool validHumanIOPublicationBinding(
    const State& state,
    const Slot& slot,
    const MetalNumanXHumanIOCandidatePublicationBinding& binding,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterAppliedOutcomeGPU& applied,
    const std::uint64_t jointCommitFingerprint,
    const std::uint64_t brainGeneration
) noexcept {
    const auto& candidate = slot.humanIOCandidate;
    return slot.humanIOCandidateBound && candidate.valid() &&
        binding.abiVersion == kMetalNumanXHumanIOPublicationABIVersion &&
        binding.structSize == sizeof(binding) &&
        binding.environmentCount == slot.transaction.environmentCount &&
        binding.environmentCount == 1u &&
        binding.transactionSlot == slot.transaction.transactionSlot &&
        binding.stepIndex == 0u &&
        binding.substepIndex == slot.transaction.physicsSubstep &&
        binding.physicsSubstepCount == slot.transaction.physicsSubsteps &&
        binding.controlStep == slot.transaction.controlStep &&
        binding.ownerProgramFingerprint == state.fingerprint &&
        binding.transactionFingerprint ==
            slot.transaction.transactionFingerprint &&
        binding.linearizationEpoch == slot.transaction.linearizationEpoch &&
        binding.slotGeneration == slot.transaction.slotGeneration &&
        binding.physicsTokenFingerprint ==
            proposal.physicsTokenFingerprint &&
        binding.proposalFingerprint == proposal.proposalFingerprint &&
        binding.ackFingerprint == applied.ackFingerprint &&
        binding.appliedDecisionFingerprint == applied.appliedFingerprint &&
        binding.jointCommitFingerprint == jointCommitFingerprint &&
        binding.brainGeneration == brainGeneration &&
        binding.candidateKeyFingerprint ==
            candidate.candidateKeyFingerprint &&
        binding.acceptedBrainGeneration ==
            candidate.acceptedBrainGeneration &&
        binding.sensorGeneration == candidate.sensorGeneration &&
        binding.humanIOProgramFingerprint ==
            candidate.humanIOProgramFingerprint &&
        binding.sensorFingerprint == candidate.sensorFingerprint &&
        binding.transactionInstanceFingerprint ==
            candidate.transactionInstanceFingerprint &&
        binding.candidatePublicationFingerprint ==
            candidate.candidatePublicationFingerprint &&
        binding.deviceRegistryID == candidate.deviceRegistryID &&
        binding.deviceRegistryID == state.device.registryID &&
        binding.humanIOIdentityFingerprint ==
            candidate.identityFingerprint &&
        binding.bindingFingerprint != 0u &&
        binding.bindingFingerprint ==
            metalNumanXHumanIOPublicationBindingFingerprint(binding);
}

void maybeReleaseLifetimeHold(State& state) noexcept {
    const bool active = std::any_of(
        state.slots.begin(), state.slots.end(), [](const Slot& slot) {
            return slot.leaseAcquired || slot.awaitingFirstAbort ||
                (slot.stage != SlotStage::empty &&
                 slot.stage != SlotStage::released);
        });
    if (!active) state.lifetimeHold.reset();
}

[[nodiscard]] MetalNumanXCoupledHumanPass makeCoupledPass(
    State& state,
    const MetalNumanXHumanMatterPass& pass,
    Frame* frame,
    const MetalNumanXCoupledHumanPhase phase
) noexcept {
    MetalNumanXCoupledHumanPass result{};
    result.accessFlags =
        MetalNumanXCoupledHumanReadSourceState |
        MetalNumanXCoupledHumanReadSourceEffectiveTangentFactor |
        MetalNumanXCoupledHumanReadStandStatus |
        MetalNumanXCoupledHumanEncodeExactCandidate |
        MetalNumanXCoupledHumanWriteReaction |
        MetalNumanXCoupledHumanWriteJointStatus;
    // Exact candidate authority belongs to this owner pass, not to the
    // reusable frozen-A0 service program.
    result.capabilities = state.coupledProgram.capabilities |
        MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
    result.commandBuffer = pass.commandBuffer;
    result.sourceQ = pass.q;
    result.sourceV = pass.v;
    result.sourceEffectiveTangentFactor =
        pass.sourceEffectiveTangentFactor;
    result.standStatuses = pass.standStatuses;
    result.exactKinematicsContext = frame;
    result.encodeExactKinematics = nullptr;
    result.sourceQGPUAddress = pass.qGPUAddress;
    result.sourceVGPUAddress = pass.vGPUAddress;
    result.sourceEffectiveTangentFactorGPUAddress =
        pass.sourceEffectiveTangentFactorGPUAddress;
    result.standStatusesGPUAddress = pass.standStatusesGPUAddress;
    result.sourceQElementCount = pass.environmentCount * pass.qStride;
    result.sourceVElementCount = pass.environmentCount * pass.vStride;
    result.sourceEffectiveTangentFactorElementCount =
        pass.environmentCount * pass.factorStride;
    result.standStatusElementCount =
        pass.environmentCount;
    result.phase = phase;
    narrowU32(pass.environmentCount, result.environmentCount);
    narrowU32(pass.qCoordinateCount, result.qCoordinateCount);
    narrowU32(pass.dofCount, result.dofCount);
    narrowU32(pass.bodyCount, result.bodyCount);
    narrowU32(pass.qStride, result.qStride);
    narrowU32(pass.vStride, result.vStride);
    narrowU32(pass.factorStride, result.factorStride);
    result.standStatusStride = 1u;
    result.articulationIndex = pass.articulationIndex;
    result.stepIndex = pass.stepIndex;
    result.stepCount = pass.stepCount;
    result.transactionSlot = pass.transactionSlot;
    result.timestepSeconds = pass.timestepSeconds;
    result.programFingerprint = state.coupledProgram.fingerprint;
    result.transactionFingerprint = pass.transactionFingerprint;
    result.linearizationEpoch = pass.linearizationEpoch;
    result.slotGeneration = pass.slotGeneration;
    return result;
}

[[nodiscard]] bool exactCandidateCallback(
    void* opaque,
    const MetalNumanXCoupledHumanPass&,
    const MetalNumanXCoupledHumanQuery& query
) noexcept {
    auto* frame = static_cast<Frame*>(opaque);
    if (frame == nullptr || frame->state == nullptr || frame->slot == nullptr ||
        frame->ownerPass == nullptr ||
        query.operation !=
            MetalNumanXCoupledHumanOperation::candidateKinematics) {
        return false;
    }
    const auto& owner = *frame->ownerPass;
    MetalNumanXHumanMatterCandidateQuery candidate{};
    candidate.accessFlags =
        MetalNumanXHumanMatterCandidateReadDeltaVelocity |
        MetalNumanXHumanMatterCandidateWriteQ |
        MetalNumanXHumanMatterCandidateWriteBodies;
    if (query.pointCount != 0u) {
        candidate.accessFlags |=
            MetalNumanXHumanMatterCandidateReadPointQueries |
            MetalNumanXHumanMatterCandidateWritePointJacobians;
    }
    candidate.deltaVelocity = query.input;
    candidate.candidateQ = query.candidateQ;
    candidate.candidateBodies = query.candidateBodies;
    candidate.pointQueries = query.pointQueries;
    candidate.pointJacobians = query.pointJacobians;
    candidate.deltaVelocityGPUAddress = query.inputGPUAddress;
    candidate.candidateQGPUAddress = query.candidateQGPUAddress;
    candidate.candidateBodiesGPUAddress = query.candidateBodiesGPUAddress;
    candidate.pointQueriesGPUAddress = query.pointQueriesGPUAddress;
    candidate.pointJacobiansGPUAddress = query.pointJacobiansGPUAddress;
    candidate.deltaVelocityStride = query.generalizedVectorStride;
    candidate.candidateQStride = query.candidateQStride;
    candidate.candidateBodyStride = query.candidateBodyStride;
    candidate.pointCount = query.pointCount;
    candidate.pointStride = query.pointStride;
    candidate.pointJacobianStride = query.pointJacobianStride;
    candidate.substepIndex = owner.substepIndex;
    candidate.transactionSlot = owner.transactionSlot;
    candidate.physicsSubstepCount = owner.physicsSubstepCount;
    candidate.controlStep = owner.controlStep;
    candidate.programFingerprint = owner.programFingerprint;
    candidate.transactionFingerprint = owner.transactionFingerprint;
    candidate.linearizationEpoch = owner.linearizationEpoch;
    candidate.slotGeneration = owner.slotGeneration;
    const bool encoded = owner.encodeExactCandidate(
        owner.exactCandidateContext, owner, candidate);
    return encoded;
}

[[nodiscard]] bool coupledCandidateCallback(
    void* opaque,
    const numi::matter::CoupledCandidateQuery& matter
) {
    auto* frame = static_cast<Frame*>(opaque);
    if (frame == nullptr || frame->state == nullptr || frame->slot == nullptr ||
        frame->ownerPass == nullptr) return false;
    auto& state = *frame->state;
    auto& slot = *frame->slot;
    const std::uint32_t operation =
        static_cast<std::uint32_t>(matter.operation);
    if (operation >= frame->operationCounts.size()) return false;
    ++frame->operationCounts[operation];

    MetalNumanXCoupledHumanQuery query{};
    query.input = matter.input;
    query.output = matter.output;
    query.candidateQ = matter.candidateQ;
    query.candidateBodies = matter.candidateBodies;
    query.statuses = matter.statuses;
    // Matter may use non-null dummy buffers for an empty attachment suffix.
    // CoupledHuman's frozen ABI represents zero points canonically: every
    // point resource, address, and stride is null/zero.  The count is the
    // authority here; never infer work from a dummy resource identity.
    query.pointQueries = matter.pointCount == 0u ? nullptr : matter.pointQueries;
    query.pointJacobians =
        matter.pointCount == 0u ? nullptr : matter.pointJacobians;
    query.operation = static_cast<MetalNumanXCoupledHumanOperation>(operation);
    query.generalizedVectorStride = matter.generalizedVectorStride;
    query.candidateQStride = matter.candidateQStride;
    query.candidateBodyStride = matter.candidateBodyStride;
    query.statusStride = matter.statusStride;
    query.pointCount = matter.pointCount;
    query.pointStride = matter.pointCount == 0u ? 0u : matter.pointStride;
    query.pointJacobianStride =
        matter.pointCount == 0u ? 0u : matter.pointJacobianStride;
    query.programFingerprint = state.coupledProgram.fingerprint;
    query.transactionFingerprint = slot.transaction.transactionFingerprint;
    query.linearizationEpoch = slot.transaction.linearizationEpoch;
    query.inputGPUAddress = gpuAddress(query.input);
    query.outputGPUAddress = gpuAddress(query.output);
    query.candidateQGPUAddress = gpuAddress(query.candidateQ);
    query.candidateBodiesGPUAddress = gpuAddress(query.candidateBodies);
    query.statusesGPUAddress = gpuAddress(query.statuses);
    query.pointQueriesGPUAddress = gpuAddress(query.pointQueries);
    query.pointJacobiansGPUAddress = gpuAddress(query.pointJacobians);
    query.transactionSlot = slot.transaction.transactionSlot;
    query.slotGeneration = slot.transaction.slotGeneration;

    switch (query.operation) {
    case MetalNumanXCoupledHumanOperation::candidateKinematics:
        query.accessFlags =
            MetalNumanXCoupledHumanQueryReadInput |
            MetalNumanXCoupledHumanQueryWriteCandidateQ |
            MetalNumanXCoupledHumanQueryWriteCandidateBodies |
            (query.pointCount == 0u ? 0u :
                MetalNumanXCoupledHumanQueryReadPointQueries |
                MetalNumanXCoupledHumanQueryWritePointJacobians);
        query.requiredCapabilities =
            MR_NUMANX_COUPLED_HUMAN_CAP_EXACT_CANDIDATE_KINEMATICS;
        break;
    case MetalNumanXCoupledHumanOperation::massAction:
        query.accessFlags =
            MetalNumanXCoupledHumanQueryReadInput |
            MetalNumanXCoupledHumanQueryWriteOutput;
        query.requiredCapabilities =
            MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_ACTION;
        break;
    case MetalNumanXCoupledHumanOperation::inverseMassPreconditioner:
        query.accessFlags =
            MetalNumanXCoupledHumanQueryReadInput |
            MetalNumanXCoupledHumanQueryWriteOutput |
            MetalNumanXCoupledHumanQueryWriteInverseStatus;
        query.requiredCapabilities =
            MR_NUMANX_COUPLED_HUMAN_CAP_PROJECTED_EFFECTIVE_TANGENT_PRECONDITIONER;
        break;
    case MetalNumanXCoupledHumanOperation::publishCandidate:
        if (!encodeCaptureOutcome(state, slot, *frame->ownerPass)) {
            return false;
        }
        query.accessFlags =
            MetalNumanXCoupledHumanQueryReadInput |
            MetalNumanXCoupledHumanQueryWriteOutput |
            MetalNumanXCoupledHumanQueryReadMatterOutcome;
        query.requiredCapabilities =
            MR_NUMANX_COUPLED_HUMAN_CAP_STAGED_GENERALIZED_REACTION_PUBLISH;
        query.matterOutcomes = (__bridge void*)slot.matterOutcomes;
        query.matterOutcomeGPUAddress = slot.matterOutcomes.gpuAddress;
        query.matterOutcomeStride = 1u;
        query.matterSuccessCode = NM_STATUS_SUCCESS;
        // preDynamics publishes the candidate reaction before Matter advances
        // the success-surviving prepared microstep.  At this boundary the
        // exact status count is the current substep (zero in ABI v1), and the
        // CoupledHuman status must remain PENDING.  postDynamics separately
        // validates expectedMatterCompletedMicrosteps after reconciliation.
        query.expectedMatterCompletedMicrosteps =
            slot.transaction.physicsSubstep;
        break;
    }
    const bool encoded = state.coupledProgram.encode(
        state.coupledProgram.context, frame->coupledPass, query);
    return encoded;
}

[[nodiscard]] numi::matter::EncodeRequest makeMatterRequest(
    State&,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass,
    Frame& frame,
    const numi::matter::EncodePhase phase
) noexcept {
    numi::matter::EncodeRequest request;
    request.commandBuffer = pass.commandBuffer;
    request.phase = phase;
    request.rigid.q = pass.q;
    request.rigid.v = pass.v;
    request.rigid.currentBodies = nullptr;
    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(pass.articulationFirstBody) +
        pass.bodyCount;
    narrowU32(bodyEnd, request.rigid.currentBodyCount);
    request.rigid.currentBodyStride = request.rigid.currentBodyCount;
    narrowU32(pass.qStride, request.rigid.qStride);
    narrowU32(pass.vStride, request.rigid.vStride);
    request.environmentStatuses = (__bridge void*)slot.worldStatuses;
    request.coupledCandidateContext = &frame;
    request.encodeCoupledCandidate = &coupledCandidateCallback;
    request.articulationRootBody = pass.articulationFirstBody;
    request.controlStep = slot.transaction.controlStep;
    request.physicsSubstep = slot.transaction.physicsSubstep;
    request.physicsSubsteps = slot.transaction.physicsSubsteps;
    request.seed = slot.transaction.seed;
    request.timestepSeconds = pass.timestepSeconds;
    request.runIdentification = false;
    request.runAdaptiveTransfer = false;
    request.enablePreparedState = true;
    return request;
}

void cancelSlot(State& state, Slot& slot) noexcept {
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        slot.commandBufferIdentity == 0u ? nil :
        (__bridge id<MTLCommandBuffer>)reinterpret_cast<void*>(
            slot.commandBufferIdentity);
    if (commandBuffer != nil &&
        commandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
        slot.callbackFrame = nullptr;
        slot.physicalCommandFailed = true;
        slot.stage = SlotStage::terminalNoTouch;
        return;
    }
    if (!slot.cancelIssued && slot.commandBufferIdentity != 0u) {
        state.config.matterRuntime->cancel(
            reinterpret_cast<void*>(slot.commandBufferIdentity));
        slot.cancelIssued = true;
    }
    slot.callbackFrame = nullptr;
    slot.matterOpened = false;
    slot.completionArmed = false;
    slot.stage = SlotStage::released;
}

[[nodiscard]] bool beginPhase(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    if (slot.stage != SlotStage::leased || !slot.leaseAcquired ||
        !validatePass(
            state, slot, pass,
            MetalNumanXHumanMatterPhase::beginStep, false)) return false;
    slot.commandBufferIdentity =
        reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
    slot.passSignature = passSignature(pass);
    slot.cancelIssued = false;
    slot.matterOpened = false;
    slot.completionArmed = false;
    slot.stage = SlotStage::begun;

    const std::shared_ptr<State> ownedState = state.self.lock();
    if (!ownedState) {
        cancelSlot(state, slot);
        return false;
    }
    __unsafe_unretained id<MTLCommandBuffer> completionCommand =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    const std::uint32_t completionSlot = pass.transactionSlot;
    const std::uint64_t completionGeneration = pass.slotGeneration;
    const std::uint64_t completionTransaction = pass.transactionFingerprint;
    const std::uintptr_t completionIdentity =
        reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
    [completionCommand addCompletedHandler:^(id<MTLCommandBuffer> completedCB) {
        std::lock_guard completionLock(ownedState->mutex);
        if (completionSlot >= ownedState->slots.size()) return;
        Slot& completed = ownedState->slots[completionSlot];
        if (completed.transaction.slotGeneration == completionGeneration &&
            completed.transaction.transactionFingerprint ==
                completionTransaction && completed.completionArmed &&
            completed.commandBufferIdentity == completionIdentity) {
            completed.callbackFrame = nullptr;
            completed.completionArmed = false;
            if (completedCB.status == MTLCommandBufferStatusCompleted) {
                completed.physicalCommandCompleted = true;
            } else {
                // Checkpoint writes on a failed first CB may be partial. Keep
                // every adapter/Matter authority immutable forever; neither a
                // later apply nor a forced restore is safe.
                completed.physicalCommandFailed = true;
                completed.stage = SlotStage::terminalNoTouch;
            }
        }
    }];
    slot.completionArmed = true;

    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBlitCommandEncoder> clear = [commandBuffer blitCommandEncoder];
    if (clear == nil) {
        cancelSlot(state, slot);
        return false;
    }
    [clear fillBuffer:slot.matterOutcomes
                 range:NSMakeRange(0u, slot.matterOutcomes.length) value:0u];
    [clear fillBuffer:slot.acceptedTokens
                 range:NSMakeRange(0u, slot.acceptedTokens.length) value:0u];
    [clear fillBuffer:slot.acceptedStateProofs
                 range:NSMakeRange(0u, slot.acceptedStateProofs.length)
                 value:0u];
    [clear endEncoding];
    if (!encodePrepareWorldStatus(state, slot, pass)) {
        cancelSlot(state, slot);
        return false;
    }
    Frame frame;
    frame.state = &state;
    frame.slot = &slot;
    frame.ownerPass = &pass;
    frame.coupledPass = makeCoupledPass(
        state, pass, &frame,
        MetalNumanXCoupledHumanPhase::preDynamics);
    frame.coupledPass.encodeExactKinematics = &exactCandidateCallback;
    if (!state.coupledProgram.begin(
            state.coupledProgram.context, frame.coupledPass)) {
        cancelSlot(state, slot);
        return false;
    }
    return true;
}

[[nodiscard]] bool prePhase(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    if (slot.stage != SlotStage::begun) return false;
    if (!validatePass(
            state, slot, pass,
            MetalNumanXHumanMatterPhase::preDynamics, true)) return false;
    Frame frame;
    frame.state = &state;
    frame.slot = &slot;
    frame.ownerPass = &pass;
    frame.coupledPass = makeCoupledPass(
        state, pass, &frame,
        MetalNumanXCoupledHumanPhase::preDynamics);
    frame.coupledPass.encodeExactKinematics = &exactCandidateCallback;
    slot.callbackFrame = &frame;
    auto request = makeMatterRequest(
        state, slot, pass, frame, numi::matter::EncodePhase::preDynamics);
    const auto encoded = state.config.matterRuntime->encode(request);
    slot.callbackFrame = nullptr;
    slot.matterOpened = encoded.encoded;
    const bool completeCallbacks =
        frame.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::candidateKinematics)] != 0u &&
        frame.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::massAction)] != 0u &&
        frame.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::inverseMassPreconditioner)] != 0u &&
        frame.operationCounts[static_cast<std::uint32_t>(
            numi::matter::CoupledCandidateOperation::publishCandidate)] == 1u;
    if (!encoded.encoded || !completeCallbacks) {
        cancelSlot(state, slot);
        return false;
    }
    slot.stage = SlotStage::preEncoded;
    return true;
}

[[nodiscard]] bool encodeStateProof(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    const auto& program = state.config.stateProofProgram;
    if (!program.valid()) return true;
    __unsafe_unretained id<MTLBuffer> matterStatuses =
        (__bridge id<MTLBuffer>)state.config.matterRuntime->statusBuffer();
    MetalNumanXHumanMatterStateProofPass proof;
    proof.environmentCount = slot.transaction.environmentCount;
    proof.environmentIdentifierBase =
        slot.transaction.environmentIdentifierBase;
    proof.commandBuffer = pass.commandBuffer;
    proof.q = pass.q;
    proof.v = pass.v;
    proof.mujocoStates = pass.mujocoStates;
    proof.matterGeneralizedReaction = pass.matterGeneralizedReaction;
    proof.environmentStatuses = (__bridge void*)slot.worldStatuses;
    proof.matterStatuses = (__bridge void*)matterStatuses;
    proof.acceptedStateProofs = (__bridge void*)slot.acceptedStateProofs;
    proof.qGPUAddress = pass.qGPUAddress;
    proof.vGPUAddress = pass.vGPUAddress;
    proof.mujocoStatesGPUAddress = pass.mujocoStatesGPUAddress;
    proof.matterGeneralizedReactionGPUAddress =
        pass.matterGeneralizedReactionGPUAddress;
    proof.environmentStatusesGPUAddress = slot.worldStatuses.gpuAddress;
    proof.matterStatusesGPUAddress = matterStatuses.gpuAddress;
    proof.acceptedStateProofsGPUAddress =
        slot.acceptedStateProofs.gpuAddress;
    proof.qElementCount = pass.environmentCount * pass.qStride;
    proof.vElementCount = pass.environmentCount * pass.vStride;
    proof.mujocoStateCount =
        pass.environmentCount * pass.mujocoStateStride;
    proof.matterGeneralizedReactionElementCount =
        pass.environmentCount * pass.reactionStride;
    proof.environmentStatusElementCount = pass.environmentCount;
    proof.matterStatusElementCount = pass.environmentCount;
    proof.acceptedStateProofElementCount = pass.environmentCount;
    narrowU32(pass.qStride, proof.qStride);
    narrowU32(pass.vStride, proof.vStride);
    narrowU32(pass.mujocoStateStride, proof.mujocoStateStride);
    narrowU32(pass.reactionStride, proof.reactionStride);
    proof.environmentStatusStride = 1u;
    proof.matterStatusStride = 1u;
    proof.acceptedStateProofStride = 1u;
    narrowU32(pass.qCoordinateCount, proof.qCoordinateCount);
    narrowU32(pass.dofCount, proof.dofCount);
    proof.transactionSlot = slot.transaction.transactionSlot;
    proof.programFingerprint = state.fingerprint;
    proof.stateProofProgramFingerprint = program.fingerprint;
    proof.transactionFingerprint = slot.transaction.transactionFingerprint;
    proof.substepFingerprint = slot.transaction.substepFingerprint;
    proof.acceptedTimestampMicroseconds =
        slot.transaction.acceptedTimestampMicroseconds;
    proof.physicsGeneration = slot.transaction.physicsGeneration;
    proof.linearizationEpoch = slot.transaction.linearizationEpoch;
    proof.slotGeneration = slot.transaction.slotGeneration;
    proof.matterSourcePhysicsFingerprint = state.matterSourceFingerprint;
    proof.matterDeviceProgramFingerprint = state.matterDeviceFingerprint;
    return program.encode(program.context, proof);
}

[[nodiscard]] bool postPhase(
    State& state,
    Slot& slot,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    if (slot.stage != SlotStage::preEncoded || !slot.matterOpened ||
        !validatePass(
            state, slot, pass,
            MetalNumanXHumanMatterPhase::postDynamics, true) ||
        !encodeMapHumanStatus(state, slot, pass)) return false;
    Frame frame;
    frame.state = &state;
    frame.slot = &slot;
    frame.ownerPass = &pass;
    frame.coupledPass = makeCoupledPass(
        state, pass, &frame,
        MetalNumanXCoupledHumanPhase::postDynamics);
    frame.coupledPass.encodeExactKinematics = &exactCandidateCallback;
    auto request = makeMatterRequest(
        state, slot, pass, frame, numi::matter::EncodePhase::postCommit);
    request.coupledCandidateContext = nullptr;
    request.encodeCoupledCandidate = nullptr;
    const auto encoded =
        state.config.matterRuntime->prepareAcceptedState(request);
    if (!encoded.encoded || !encodeStateProof(state, slot, pass) ||
        !encodeCaptureOutcome(state, slot, pass)) {
        cancelSlot(state, slot);
        return false;
    }
    MetalNumanXCoupledHumanResolveQuery resolve;
    resolve.matterOutcomes = (__bridge void*)slot.matterOutcomes;
    resolve.matterOutcomeGPUAddress = slot.matterOutcomes.gpuAddress;
    resolve.matterOutcomeElementCount = slot.transaction.environmentCount;
    resolve.matterOutcomeStride = 1u;
    resolve.matterSuccessCode = NM_STATUS_SUCCESS;
    resolve.expectedMatterCompletedMicrosteps =
        slot.transaction.expectedMatterCompletedMicrosteps;
    resolve.transactionSlot = slot.transaction.transactionSlot;
    resolve.programFingerprint = state.coupledProgram.fingerprint;
    resolve.transactionFingerprint = slot.transaction.transactionFingerprint;
    resolve.linearizationEpoch = slot.transaction.linearizationEpoch;
    resolve.slotGeneration = slot.transaction.slotGeneration;
    if (!state.coupledProgram.resolve(
            state.coupledProgram.context, frame.coupledPass, resolve) ||
        !encodePreparedToken(state, slot, pass)) {
        cancelSlot(state, slot);
        return false;
    }
    slot.callbackFrame = nullptr;
    slot.stage = SlotStage::postEncoded;
    return true;
}

[[nodiscard]] bool encodeCallback(
    void* opaque,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    @autoreleasepool {
        auto* state = static_cast<State*>(opaque);
        const auto ownedState = state == nullptr
            ? std::shared_ptr<State>{} : state->self.lock();
        if (!ownedState || !state->initialized ||
            pass.transactionSlot >= state->slots.size()) return false;
        std::lock_guard lock(state->mutex);
        Slot& slot = state->slots[pass.transactionSlot];
        switch (pass.phase) {
        case MetalNumanXHumanMatterPhase::beginStep:
            return beginPhase(*state, slot, pass);
        case MetalNumanXHumanMatterPhase::preDynamics:
            return prePhase(*state, slot, pass);
        case MetalNumanXHumanMatterPhase::postDynamics:
            return postPhase(*state, slot, pass);
        }
        return false;
    }
}

void abortCallback(void* opaque, void* commandBuffer) noexcept {
    @autoreleasepool {
        auto* state = static_cast<State*>(opaque);
        const auto ownedState = state == nullptr
            ? std::shared_ptr<State>{} : state->self.lock();
        if (!ownedState || !state->initialized || commandBuffer == nullptr) {
            return;
        }
        const std::uintptr_t identity =
            reinterpret_cast<std::uintptr_t>(commandBuffer);
        std::lock_guard lock(state->mutex);
        for (auto& slot : state->slots) {
            if (slot.commandBufferIdentity == identity &&
                slot.stage != SlotStage::empty &&
                slot.stage != SlotStage::prepared) {
                if (slot.stage != SlotStage::released) {
                    cancelSlot(*state, slot);
                }
                slot.awaitingFirstAbort = false;
                slot.commandBufferIdentity = 0u;
            }
        }
        maybeReleaseLifetimeHold(*state);
    }
}

[[nodiscard]] bool acquirePrepareLeaseCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) return false;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            if (slot.stage != SlotStage::prepared || slot.leaseAcquired ||
                !validLeaseResources(*state, slot, lease)) return false;
            slot.lease = leaseIdentity(lease);
            slot.leaseAcquired = true;
            slot.stage = SlotStage::leased;
            return true;
        } catch (...) {
            return false;
        }
    }
}

[[nodiscard]] bool bindHumanIOCandidatePublicationCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& stagedLease,
    const MetalNumanXHumanIOCandidatePublicationProgram& candidate
) noexcept {
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                stagedLease.transactionSlot >= state->slots.size() ||
                !candidate.valid() ||
                !sameHumanIOCandidate(
                    stagedLease.humanIOCandidate,
                    candidate)) {
                return false;
            }
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[stagedLease.transactionSlot];
            auto baseLease = stagedLease;
            baseLease.humanIOCandidate = {};
            if (!slot.leaseAcquired ||
                slot.stage != SlotStage::postEncoded ||
                slot.humanIOCandidateBound ||
                slot.physicalCommandFailed ||
                slot.applicationReservation.active ||
                !sameLease(slot.lease, baseLease) ||
                !validLeaseResources(*state, slot, baseLease) ||
                candidate.transactionFingerprint !=
                    slot.transaction.transactionFingerprint ||
                candidate.deviceRegistryID != state->device.registryID ||
                candidate.identityFingerprint !=
                    candidate.computedIdentityFingerprint()) {
                return false;
            }
            __unsafe_unretained id<MTLCommandBuffer> physical =
                slot.commandBufferIdentity == 0u ? nil :
                (__bridge id<MTLCommandBuffer>)reinterpret_cast<void*>(
                    slot.commandBufferIdentity);
            if (physical == nil ||
                physical.status != MTLCommandBufferStatusCompleted ||
                physical.error != nil) {
                if (physical != nil &&
                    physical.status == MTLCommandBufferStatusError) {
                    slot.physicalCommandFailed = true;
                    slot.stage = SlotStage::terminalNoTouch;
                }
                return false;
            }
            slot.physicalCommandCompleted = true;
            slot.humanIOCandidate = candidate;
            slot.humanIOBinding = {};
            slot.humanIOCandidateBound = true;
            slot.humanIORootReserved = false;
            return true;
        } catch (...) {
            return false;
        }
    }
}

[[nodiscard]] numi::matter::PreparedStateDispositionIdentity
makeDispositionIdentity(
    const Slot& slot,
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept;

[[nodiscard]] bool reservePreparedApplicationCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterProposalView& proposalView,
    const MetalNumanXHumanMatterBrainPreflightView& preflight
) noexcept {
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) return false;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            if (!slot.leaseAcquired || slot.stage != SlotStage::postEncoded ||
                slot.physicalCommandFailed ||
                slot.applicationReservation.active ||
                !sameLease(slot.lease, lease) ||
                !validLeaseResources(*state, slot, lease) ||
                !exactProposalView(lease, proposalView) ||
                preflight.abiVersion != kMetalNumanXHumanMatterABIVersion ||
                preflight.structSize != sizeof(preflight) ||
                preflight.environmentCount != 1u ||
                preflight.environmentCount !=
                    slot.transaction.environmentCount ||
                preflight.transactionSlot !=
                    slot.transaction.transactionSlot ||
                preflight.stepIndex != 0u ||
                preflight.substepIndex != slot.transaction.physicsSubstep ||
                preflight.physicsSubstepCount !=
                    slot.transaction.physicsSubsteps ||
                preflight.controlStep != slot.transaction.controlStep ||
                preflight.programFingerprint != state->fingerprint ||
                preflight.transactionFingerprint !=
                    slot.transaction.transactionFingerprint ||
                preflight.linearizationEpoch !=
                    slot.transaction.linearizationEpoch ||
                preflight.slotGeneration !=
                    slot.transaction.slotGeneration ||
                preflight.brainCommitPreflightStride != 1u ||
                preflight.brainCommitPreflightElementCount != 1u ||
                preflight.preflightReadyEventValue == 0u ||
                preflight.preflightReadyEvent ==
                    lease.physicalPreparedEvent ||
                !exactSharedEvent(
                    state->device, preflight.preflightReadyEvent) ||
                !exactHostVisibleBuffer(
                    state->device,
                    preflight.brainCommitPreflights,
                    preflight.brainCommitPreflightsGPUAddress,
                    MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES)) {
                return false;
            }
            __unsafe_unretained id<MTLSharedEvent> ownerEvent =
                (__bridge id<MTLSharedEvent>)lease.physicalPreparedEvent;
            if (ownerEvent == nil ||
                ownerEvent.signaledValue < lease.proposalEventValue) {
                return false;
            }
            BufferRegion preflightRegion{};
            if (!bufferRegion(
                    state->device, preflight.brainCommitPreflights,
                    preflight.brainCommitPreflightsGPUAddress,
                    MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES,
                    preflightRegion) ||
                aliasesLeaseOrAdapterAuthority(
                    *state, slot, lease, preflightRegion)) {
                return false;
            }
            const auto matterDisposition =
                state->config.matterRuntime->preparedStateDisposition(
                    makeDispositionIdentity(slot, lease));
            if (matterDisposition !=
                numi::matter::PreparedStateDisposition::prepared) {
                if (matterDisposition ==
                        numi::matter::PreparedStateDisposition::
                            terminalNoTouch ||
                    matterDisposition ==
                        numi::matter::PreparedStateDisposition::
                            restoreRequired) {
                    slot.stage = SlotStage::terminalNoTouch;
                }
                return false;
            }
            __unsafe_unretained id<MTLBuffer> proposals =
                (__bridge id<MTLBuffer>)lease.proposals;
            __unsafe_unretained id<MTLBuffer> proposedTokens =
                (__bridge id<MTLBuffer>)lease.proposedPhysicsStateTokens;
            if (proposals == nil || proposedTokens == nil ||
                proposals.contents == nullptr ||
                proposedTokens.contents == nullptr) return false;
            const auto* proposal = static_cast<const
                MRNumanXHumanMatterProposalGPU*>(proposals.contents);
            if (!validProposalRecord(
                    *state, slot, *proposal, proposedTokens.contents)) {
                slot.stage = SlotStage::terminalNoTouch;
                return false;
            }
            auto& reservation = slot.applicationReservation;
            reservation.preflights = reinterpret_cast<std::uintptr_t>(
                preflight.brainCommitPreflights);
            reservation.preflightEvent = reinterpret_cast<std::uintptr_t>(
                preflight.preflightReadyEvent);
            reservation.preflightsGPUAddress =
                preflight.brainCommitPreflightsGPUAddress;
            reservation.preflightElements =
                preflight.brainCommitPreflightElementCount;
            reservation.preflightEventValue =
                preflight.preflightReadyEventValue;
            reservation.preflightStride =
                preflight.brainCommitPreflightStride;
            reservation.proposal = *proposal;
            reservation.active = true;
            reservation.applyActive = false;
            slot.stage = SlotStage::applicationReserved;
            return true;
        } catch (...) {
            return false;
        }
    }
}

[[nodiscard]] bool sameApplyPass(
    const detail::MetalNumanXHumanMatterApplicationReservation& reservation,
    const MetalNumanXHumanMatterApplyPass& pass
) noexcept {
    return reservation.applyActive &&
        reservation.applyMode == pass.mode &&
        reservation.brainAcks ==
            reinterpret_cast<std::uintptr_t>(pass.brainAcks) &&
        reservation.brainAckEvent ==
            reinterpret_cast<std::uintptr_t>(pass.brainAckEvent) &&
        reservation.brainAcksGPUAddress == pass.brainAcksGPUAddress &&
        reservation.brainAckElements == pass.brainAckElementCount &&
        reservation.brainAckEventValue == pass.brainAckEventValue &&
        reservation.brainAckStride == pass.brainAckStride;
}

[[nodiscard]] bool encodePreparedApplyCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterApplyPass& ownerPass
) noexcept {
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) return false;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            auto& reservation = slot.applicationReservation;
            const bool validateAck = ownerPass.mode ==
                MetalNumanXHumanMatterApplyMode::validateBrainAck;
            if (!slot.leaseAcquired ||
                slot.stage != SlotStage::applicationReserved ||
                !reservation.active || reservation.applyActive ||
                slot.physicalCommandFailed || slot.applyEncoded ||
                slot.applyCommandBufferIdentity != 0u ||
                !sameLease(slot.lease, lease) ||
                !validLeaseResources(*state, slot, lease) ||
                ownerPass.abiVersion !=
                    kMetalNumanXHumanMatterABIVersion ||
                ownerPass.structSize != sizeof(ownerPass) ||
                ownerPass.reserved0 != 0u ||
                (!validateAck && ownerPass.mode !=
                    MetalNumanXHumanMatterApplyMode::forceReject) ||
                ownerPass.environmentCount != 1u ||
                ownerPass.environmentCount !=
                    slot.transaction.environmentCount ||
                ownerPass.transactionSlot !=
                    slot.transaction.transactionSlot ||
                ownerPass.stepIndex != 0u ||
                ownerPass.substepIndex != slot.transaction.physicsSubstep ||
                ownerPass.physicsSubstepCount !=
                    slot.transaction.physicsSubsteps ||
                ownerPass.controlStep != slot.transaction.controlStep ||
                ownerPass.programFingerprint != state->fingerprint ||
                ownerPass.transactionFingerprint !=
                    slot.transaction.transactionFingerprint ||
                ownerPass.linearizationEpoch !=
                    slot.transaction.linearizationEpoch ||
                ownerPass.slotGeneration !=
                    slot.transaction.slotGeneration ||
                !validCommandBuffer(
                    state->device, ownerPass.commandBuffer)) {
                return false;
            }
            if (!exactHostVisibleBuffer(
                    state->device,
                    reinterpret_cast<void*>(reservation.preflights),
                    reservation.preflightsGPUAddress,
                    MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES) ||
                !exactSharedEvent(
                    state->device,
                    reinterpret_cast<void*>(reservation.preflightEvent))) {
                slot.stage = SlotStage::terminalNoTouch;
                return false;
            }
            BufferRegion preflightRegion{
                reservation.preflightsGPUAddress,
                MR_NUMANX_HUMAN_MATTER_BRAIN_PREFLIGHT_BYTES,
            };
            BufferRegion ackRegion{};
            if (validateAck) {
                if (ownerPass.brainAckStride != 1u ||
                    ownerPass.brainAckElementCount != 1u ||
                    ownerPass.brainAckEventValue == 0u ||
                    ownerPass.brainAckEvent == lease.physicalPreparedEvent ||
                    ownerPass.brainAckEvent ==
                        reinterpret_cast<void*>(
                            reservation.preflightEvent) ||
                    !exactSharedEvent(
                        state->device, ownerPass.brainAckEvent) ||
                    !exactHostVisibleBuffer(
                        state->device, ownerPass.brainAcks,
                        ownerPass.brainAcksGPUAddress,
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_BYTES) ||
                    !bufferRegion(
                        state->device, ownerPass.brainAcks,
                        ownerPass.brainAcksGPUAddress,
                        MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_BYTES,
                        ackRegion) ||
                    regionsOverlap(ackRegion, preflightRegion) ||
                    aliasesLeaseOrAdapterAuthority(
                        *state, slot, lease, ackRegion)) {
                    return false;
                }
            } else if (ownerPass.brainAcks != nullptr ||
                       ownerPass.brainAckEvent != nullptr ||
                       ownerPass.brainAcksGPUAddress != 0u ||
                       ownerPass.brainAckElementCount != 0u ||
                       ownerPass.brainAckEventValue != 0u ||
                       ownerPass.brainAckStride != 0u) {
                return false;
            }
            __unsafe_unretained id<MTLBuffer> proposals =
                (__bridge id<MTLBuffer>)lease.proposals;
            __unsafe_unretained id<MTLBuffer> proposedTokens =
                (__bridge id<MTLBuffer>)lease.proposedPhysicsStateTokens;
            if (proposals == nil || proposedTokens == nil ||
                proposals.contents == nullptr ||
                proposedTokens.contents == nullptr ||
                std::memcmp(
                    proposals.contents, &reservation.proposal,
                    sizeof(reservation.proposal)) != 0 ||
                !validProposalRecord(
                    *state, slot, reservation.proposal,
                    proposedTokens.contents)) {
                slot.stage = SlotStage::terminalNoTouch;
                return false;
            }
            numi::matter::AcceptedStateApplyPass matterPass{};
            matterPass.mode = validateAck
                ? numi::matter::PreparedStateApplyMode::validateBrainAck
                : numi::matter::PreparedStateApplyMode::forceReject;
            matterPass.environmentCount = slot.transaction.environmentCount;
            matterPass.environmentIdentifierBase =
                slot.transaction.environmentIdentifierBase;
            matterPass.controlStep = slot.transaction.controlStep;
            matterPass.physicsSubstep = slot.transaction.physicsSubstep;
            matterPass.physicsSubstepCount =
                slot.transaction.physicsSubsteps;
            matterPass.transactionSlot = slot.transaction.transactionSlot;
            matterPass.commandBuffer = ownerPass.commandBuffer;
            matterPass.proposals = lease.proposals;
            matterPass.brainAcks = ownerPass.brainAcks;
            matterPass.applyActions = lease.applyActions;
            matterPass.matterApplyOutcomes = lease.matterApplyOutcomes;
            matterPass.proposedPhysicsStateTokens =
                lease.proposedPhysicsStateTokens;
            matterPass.proposalsGPUAddress = lease.proposalsGPUAddress;
            matterPass.brainAcksGPUAddress = ownerPass.brainAcksGPUAddress;
            matterPass.applyActionsGPUAddress = lease.applyActionsGPUAddress;
            matterPass.matterApplyOutcomesGPUAddress =
                lease.matterApplyOutcomesGPUAddress;
            matterPass.proposedPhysicsStateTokensGPUAddress =
                lease.proposedPhysicsStateTokensGPUAddress;
            matterPass.proposalElementCount = lease.proposalElementCount;
            matterPass.brainAckElementCount = ownerPass.brainAckElementCount;
            matterPass.applyActionElementCount =
                lease.applyActionElementCount;
            matterPass.matterApplyOutcomeElementCount =
                lease.matterApplyOutcomeElementCount;
            matterPass.proposedPhysicsStateTokenBytes =
                lease.proposedPhysicsStateTokenByteCount;
            matterPass.proposalStride = lease.proposalStride;
            matterPass.brainAckStride = ownerPass.brainAckStride;
            matterPass.applyActionStride = lease.applyActionStride;
            matterPass.matterApplyOutcomeStride =
                lease.matterApplyOutcomeStride;
            matterPass.proposedPhysicsStateTokenStrideBytes =
                lease.proposedTokenStrideBytes;
            matterPass.ownerProgramFingerprint = lease.programFingerprint;
            matterPass.transactionFingerprint =
                lease.transactionFingerprint;
            matterPass.linearizationEpoch = lease.linearizationEpoch;
            matterPass.slotGeneration = lease.slotGeneration;
            if (!state->config.matterRuntime->applyPreparedState(matterPass)) {
                return false;
            }
            reservation.brainAcks = reinterpret_cast<std::uintptr_t>(
                ownerPass.brainAcks);
            reservation.brainAckEvent = reinterpret_cast<std::uintptr_t>(
                ownerPass.brainAckEvent);
            reservation.brainAcksGPUAddress =
                ownerPass.brainAcksGPUAddress;
            reservation.brainAckElements =
                ownerPass.brainAckElementCount;
            reservation.brainAckEventValue =
                ownerPass.brainAckEventValue;
            reservation.brainAckStride = ownerPass.brainAckStride;
            reservation.applyMode = ownerPass.mode;
            reservation.applyActive = true;
            slot.applyCommandBufferIdentity =
                reinterpret_cast<std::uintptr_t>(ownerPass.commandBuffer);
            slot.applyEncoded = true;
            slot.applyCommandCompleted = false;
            slot.applyCommandSucceeded = false;
            ++slot.applyAttempt;
            slot.stage = SlotStage::applyEncoded;
            return true;
        } catch (...) {
            return false;
        }
    }
}

void abortPreparedApplyCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterApplyPass& ownerPass
) noexcept {
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) return;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
                (__bridge id<MTLCommandBuffer>)ownerPass.commandBuffer;
            if (!slot.leaseAcquired ||
                slot.stage != SlotStage::applyEncoded ||
                !slot.applyEncoded ||
                slot.applyCommandBufferIdentity !=
                    reinterpret_cast<std::uintptr_t>(
                        ownerPass.commandBuffer) ||
                !sameLease(slot.lease, lease) ||
                !validLeaseResources(*state, slot, lease) ||
                !sameApplyPass(slot.applicationReservation, ownerPass) ||
                commandBuffer == nil ||
                commandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
                return;
            }
            state->config.matterRuntime->cancel(ownerPass.commandBuffer);
            const auto disposition =
                state->config.matterRuntime->preparedStateDisposition(
                    makeDispositionIdentity(slot, lease));
            if (disposition !=
                numi::matter::PreparedStateDisposition::prepared) {
                slot.stage = SlotStage::terminalNoTouch;
                return;
            }
            auto& reservation = slot.applicationReservation;
            reservation.brainAcks = 0u;
            reservation.brainAckEvent = 0u;
            reservation.brainAcksGPUAddress = 0u;
            reservation.brainAckElements = 0u;
            reservation.brainAckEventValue = 0u;
            reservation.brainAckStride = 0u;
            reservation.applyActive = false;
            slot.applyEncoded = false;
            slot.applyCommandCompleted = false;
            slot.applyCommandSucceeded = false;
            slot.applyCommandBufferIdentity = 0u;
            ++slot.applyAttempt;
            slot.stage = SlotStage::applicationReserved;
        } catch (...) {
        }
    }
}

[[nodiscard]] numi::matter::PreparedStateDispositionIdentity
makeDispositionIdentity(
    const Slot& slot,
    const MetalNumanXHumanMatterPrepareLease& lease
) noexcept {
    numi::matter::PreparedStateDispositionIdentity identity{};
    identity.controlStep = slot.transaction.controlStep;
    identity.physicsSubstep = slot.transaction.physicsSubstep;
    identity.physicsSubstepCount = slot.transaction.physicsSubsteps;
    identity.transactionSlot = slot.transaction.transactionSlot;
    identity.ownerProgramFingerprint = lease.programFingerprint;
    identity.transactionFingerprint = lease.transactionFingerprint;
    identity.linearizationEpoch = lease.linearizationEpoch;
    identity.slotGeneration = lease.slotGeneration;
    return identity;
}

[[nodiscard]] MetalNumanXHumanMatterPrepareLeaseDisposition
releasePrepareLeaseCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease,
    void* applyCommandBuffer,
    const bool completed
) noexcept {
    using Disposition = MetalNumanXHumanMatterPrepareLeaseDisposition;
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) {
                return Disposition::terminalNoTouch;
            }
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            if (!slot.leaseAcquired || !sameLease(slot.lease, lease) ||
                !validLeaseResources(*state, slot, lease)) {
                return Disposition::terminalNoTouch;
            }
            if (slot.physicalCommandFailed ||
                slot.stage == SlotStage::terminalNoTouch) {
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }

            if (applyCommandBuffer == nullptr) {
                if (completed) {
                    slot.stage = SlotStage::terminalNoTouch;
                    return Disposition::terminalNoTouch;
                }
                if (slot.commandBufferIdentity == 0u &&
                    (slot.stage == SlotStage::leased ||
                     slot.stage == SlotStage::released)) {
                    slot.leaseAcquired = false;
                    slot.lease = {};
                    slot.stage = SlotStage::released;
                    maybeReleaseLifetimeHold(*state);
                    return Disposition::released;
                }
                __unsafe_unretained id<MTLCommandBuffer> firstCommand =
                    slot.commandBufferIdentity == 0u ? nil :
                    (__bridge id<MTLCommandBuffer>)reinterpret_cast<void*>(
                        slot.commandBufferIdentity);
                if (firstCommand != nil &&
                    firstCommand.status ==
                        MTLCommandBufferStatusNotEnqueued) {
                    if (!slot.cancelIssued) {
                        state->config.matterRuntime->cancel(
                            reinterpret_cast<void*>(
                                slot.commandBufferIdentity));
                        slot.cancelIssued = true;
                    }
                    slot.callbackFrame = nullptr;
                    slot.matterOpened = false;
                    slot.completionArmed = false;
                    slot.leaseAcquired = false;
                    slot.lease = {};
                    slot.stage = SlotStage::released;
                    // The owner's abort guard runs after its lease guard on
                    // this failure path. Keep the raw callback context alive
                    // until that exact abort is observed.
                    slot.awaitingFirstAbort = true;
                    return Disposition::released;
                }
                // Teardown after submission cannot prove checkpoint
                // completeness or final outcome. Retain all authority as a
                // terminal quarantine.
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }

            __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
                (__bridge id<MTLCommandBuffer>)applyCommandBuffer;
            if (commandBuffer == nil ||
                slot.applyCommandBufferIdentity !=
                    reinterpret_cast<std::uintptr_t>(applyCommandBuffer)) {
                // Proposal failure and every stale/non-apply completion are
                // terminal without touching Matter's retained checkpoint.
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
            const bool commandSucceeded = completed &&
                commandBuffer.status == MTLCommandBufferStatusCompleted;
            if ((completed && !commandSucceeded) ||
                (!completed &&
                 commandBuffer.status != MTLCommandBufferStatusError)) {
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }

            const auto matterDisposition =
                state->config.matterRuntime->preparedStateDisposition(
                    makeDispositionIdentity(slot, lease));
            slot.applicationReservation.applyActive = false;
            slot.applyEncoded = false;
            slot.applyCommandCompleted = true;
            slot.applyCommandSucceeded = commandSucceeded;
            slot.applyCommandBufferIdentity = 0u;
            if (!commandSucceeded) {
                // Owner ABI4 overwrites AppliedOutcome with an exact terminal
                // record and zero token before advancing liveness. A failed
                // apply CB may have partially mutated either physical owner;
                // no forced restore is safe.
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
            switch (matterDisposition) {
            case numi::matter::PreparedStateDisposition::resolved:
            {
                __unsafe_unretained id<MTLBuffer> proposals =
                    (__bridge id<MTLBuffer>)lease.proposals;
                __unsafe_unretained id<MTLBuffer> proposedTokens =
                    (__bridge id<MTLBuffer>)
                        lease.proposedPhysicsStateTokens;
                __unsafe_unretained id<MTLBuffer> appliedOutcomes =
                    (__bridge id<MTLBuffer>)lease.appliedOutcomes;
                __unsafe_unretained id<MTLBuffer> finalTokens =
                    (__bridge id<MTLBuffer>)
                        lease.finalAcceptedPhysicsStateTokens;
                if (proposals == nil || proposedTokens == nil ||
                    appliedOutcomes == nil || finalTokens == nil ||
                    proposals.contents == nullptr ||
                    proposedTokens.contents == nullptr ||
                    appliedOutcomes.contents == nullptr ||
                    finalTokens.contents == nullptr) {
                    slot.stage = SlotStage::terminalNoTouch;
                    return Disposition::terminalNoTouch;
                }
                const auto* proposal = static_cast<const
                    MRNumanXHumanMatterProposalGPU*>(proposals.contents);
                const auto* applied = static_cast<const
                    MRNumanXHumanMatterAppliedOutcomeGPU*>(
                        appliedOutcomes.contents);
                if (!validAppliedRecord(
                        *state, slot, *applied, *proposal,
                        proposedTokens.contents, finalTokens.contents) ||
                    applied->status !=
                        MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED) {
                    // Matter may have safely restored its private checkpoint,
                    // but an invalid/terminal owner completion cannot release
                    // Human or sensor authority. Preserve adapter quarantine.
                    slot.stage = SlotStage::terminalNoTouch;
                    return Disposition::terminalNoTouch;
                }
                if (!slot.humanIOCandidateBound ||
                    !slot.humanIOCandidate.valid() ||
                    slot.humanIORootReserved ||
                    slot.humanIOCandidate.rejectCandidate(
                        slot.humanIOCandidate.context,
                        slot.humanIOCandidate.
                            candidatePublicationFingerprint) !=
                        MetalNumanXHumanIOCandidatePublicationDisposition::
                            rejected) {
                    slot.stage = SlotStage::terminalNoTouch;
                    return Disposition::terminalNoTouch;
                }
                slot.leaseAcquired = false;
                slot.lease = {};
                slot.commandBufferIdentity = 0u;
                slot.applyCommandBufferIdentity = 0u;
                slot.matterOpened = false;
                slot.cancelIssued = false;
                slot.completionArmed = false;
                slot.physicalCommandCompleted = false;
                slot.physicalCommandFailed = false;
                slot.applyEncoded = false;
                slot.applyCommandCompleted = false;
                slot.applyCommandSucceeded = false;
                slot.restoreRequired = false;
                slot.applicationReservation = {};
                slot.publicationBinding = {};
                slot.publicationReservation = {};
                slot.appliedOutcome = {};
                slot.humanIOCandidate = {};
                slot.humanIOBinding = {};
                slot.publicationReserved = false;
                slot.humanIOCandidateBound = false;
                slot.humanIORootReserved = false;
                slot.stage = SlotStage::released;
                maybeReleaseLifetimeHold(*state);
                return Disposition::released;
            }
            case numi::matter::PreparedStateDisposition::
                    acceptedPendingPublication:
                slot.stage = SlotStage::acceptedPendingPublication;
                return Disposition::acceptedPendingPublication;
            case numi::matter::PreparedStateDisposition::restoreRequired:
                slot.restoreRequired = true;
                slot.stage = SlotStage::restoreRequired;
                return Disposition::restoreRequired;
            case numi::matter::PreparedStateDisposition::terminalNoTouch:
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            case numi::matter::PreparedStateDisposition::unknown:
            case numi::matter::PreparedStateDisposition::prepared:
            case numi::matter::PreparedStateDisposition::applying:
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
        } catch (...) {
            return Disposition::terminalNoTouch;
        }
    }
    return Disposition::terminalNoTouch;
}

[[nodiscard]] bool reservePublishedRootCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterPublicationReservationView& reservation
) noexcept {
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) return false;
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            if (!slot.leaseAcquired ||
                slot.stage != SlotStage::acceptedPendingPublication ||
                !slot.applicationReservation.active ||
                slot.publicationReserved || slot.physicalCommandFailed ||
                !sameLease(slot.lease, lease) ||
                !validLeaseResources(*state, slot, lease) ||
                reservation.abiVersion !=
                    kMetalNumanXHumanMatterABIVersion ||
                reservation.structSize != sizeof(reservation) ||
                !exactProposalView(lease, reservation.proposal) ||
                !exactFenceView(slot, lease, reservation.fence) ||
                reservation.appliedOutcomes != lease.appliedOutcomes ||
                reservation.finalAcceptedPhysicsStateTokens !=
                    lease.finalAcceptedPhysicsStateTokens ||
                reservation.appliedOutcomesGPUAddress !=
                    lease.appliedOutcomesGPUAddress ||
                reservation.finalAcceptedPhysicsStateTokensGPUAddress !=
                    lease.finalAcceptedPhysicsStateTokensGPUAddress ||
                reservation.appliedOutcomeElementCount !=
                    lease.appliedOutcomeElementCount ||
                reservation.finalAcceptedPhysicsStateTokenByteCount !=
                    lease.finalAcceptedPhysicsStateTokenByteCount ||
                reservation.appliedOutcomeStride !=
                    lease.appliedOutcomeStride ||
                reservation.finalTokenStrideBytes !=
                    lease.finalTokenStrideBytes ||
                reservation.jointCommitFingerprint == 0u ||
                reservation.brainGeneration == 0u ||
                state->config.matterRuntime->preparedStateDisposition(
                    makeDispositionIdentity(slot, lease)) !=
                    numi::matter::PreparedStateDisposition::
                        acceptedPendingPublication) {
                return false;
            }
            __unsafe_unretained id<MTLBuffer> proposals =
                (__bridge id<MTLBuffer>)lease.proposals;
            __unsafe_unretained id<MTLBuffer> proposedTokens =
                (__bridge id<MTLBuffer>)lease.proposedPhysicsStateTokens;
            __unsafe_unretained id<MTLBuffer> appliedOutcomes =
                (__bridge id<MTLBuffer>)lease.appliedOutcomes;
            __unsafe_unretained id<MTLBuffer> finalTokens =
                (__bridge id<MTLBuffer>)
                    lease.finalAcceptedPhysicsStateTokens;
            __unsafe_unretained id<MTLBuffer> fences =
                (__bridge id<MTLBuffer>)lease.publicationFences;
            if (proposals == nil || proposedTokens == nil ||
                appliedOutcomes == nil || finalTokens == nil ||
                fences == nil || proposals.contents == nullptr ||
                proposedTokens.contents == nullptr ||
                appliedOutcomes.contents == nullptr ||
                finalTokens.contents == nullptr ||
                fences.contents == nullptr) {
                return false;
            }
            const auto* proposal = static_cast<const
                MRNumanXHumanMatterProposalGPU*>(proposals.contents);
            const auto* applied = static_cast<const
                MRNumanXHumanMatterAppliedOutcomeGPU*>(
                    appliedOutcomes.contents);
            const auto* fence = static_cast<const
                MRNumanXHumanMatterJointPublicationFenceGPU*>(
                    fences.contents);
            if (std::memcmp(
                    proposal, &slot.applicationReservation.proposal,
                    sizeof(*proposal)) != 0 ||
                !validAppliedRecord(
                    *state, slot, *applied, *proposal,
                    proposedTokens.contents, finalTokens.contents) ||
                applied->status !=
                    MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED) {
                slot.stage = SlotStage::terminalNoTouch;
                return false;
            }
            if (!validHumanIOPublicationBinding(
                    *state,
                    slot,
                    reservation.humanIOBinding,
                    *proposal,
                    *applied,
                    reservation.jointCommitFingerprint,
                    reservation.brainGeneration)) {
                return false;
            }
            numi::matter::PreparedStatePublicationBinding binding{};
            binding.physicsTokenFingerprint =
                proposal->physicsTokenFingerprint;
            binding.brainProgramFingerprint =
                proposal->brainProgramFingerprint;
            binding.brainShadowStateFingerprint =
                proposal->brainShadowStateFingerprint;
            binding.brainWitnessFingerprint =
                proposal->brainWitnessFingerprint;
            binding.matterApplyFingerprint =
                applied->matterApplyFingerprint;
            binding.appliedDecisionFingerprint =
                applied->appliedFingerprint;
            binding.jointCommitFingerprint =
                reservation.jointCommitFingerprint;
            binding.brainGeneration = reservation.brainGeneration;
            if (!validPublicationFenceRecord(
                    *state, slot, *fence, binding,
                    MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING)) {
                return false;
            }
            if (slot.humanIORootReserved) {
                if (std::memcmp(
                        &slot.humanIOBinding,
                        &reservation.humanIOBinding,
                        sizeof(slot.humanIOBinding)) != 0) {
                    slot.stage = SlotStage::terminalNoTouch;
                    return false;
                }
            } else {
                if (!slot.humanIOCandidate.reservePublishedRoot(
                        slot.humanIOCandidate.context,
                        slot.humanIOCandidate.
                            candidatePublicationFingerprint,
                        reservation.humanIOBinding)) {
                    return false;
                }
                slot.humanIOBinding = reservation.humanIOBinding;
                slot.humanIORootReserved = true;
            }
            numi::matter::PreparedStatePublicationReservation
                matterReservation{};
            if (!state->config.matterRuntime->reservePublishedRoot(
                    makeDispositionIdentity(slot, lease), binding,
                    matterReservation)) {
                return false;
            }
            slot.publicationBinding = binding;
            slot.publicationReservation = matterReservation;
            slot.appliedOutcome = *applied;
            slot.publicationReserved = true;
            return true;
        } catch (...) {
            return false;
        }
    }
}

[[nodiscard]] MetalNumanXHumanMatterPrepareLeaseDisposition
releasePublishedRootCallback(
    void* opaque,
    const MetalNumanXHumanMatterPrepareLease& lease,
    const MetalNumanXHumanMatterPublicationFenceView& fenceView
) noexcept {
    using Disposition = MetalNumanXHumanMatterPrepareLeaseDisposition;
    @autoreleasepool {
        try {
            auto* state = static_cast<State*>(opaque);
            const auto ownedState = state == nullptr
                ? std::shared_ptr<State>{} : state->self.lock();
            if (!ownedState || !state->initialized ||
                lease.transactionSlot >= state->slots.size()) {
                return Disposition::terminalNoTouch;
            }
            std::lock_guard lock(state->mutex);
            Slot& slot = state->slots[lease.transactionSlot];
            if (!slot.leaseAcquired ||
                slot.stage != SlotStage::acceptedPendingPublication ||
                !slot.publicationReserved ||
                !sameLease(slot.lease, lease) ||
                !validLeaseResources(*state, slot, lease) ||
                !exactFenceView(slot, lease, fenceView)) {
                return Disposition::terminalNoTouch;
            }
            __unsafe_unretained id<MTLBuffer> fences =
                (__bridge id<MTLBuffer>)lease.publicationFences;
            if (fences == nil || fences.contents == nullptr) {
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
            const auto* ownerFence = static_cast<const
                MRNumanXHumanMatterJointPublicationFenceGPU*>(
                    fences.contents);
            numi::matter::PreparedStatePublicationFence matterFence{};
            static_assert(sizeof(matterFence) == sizeof(*ownerFence));
            std::memcpy(&matterFence, ownerFence, sizeof(matterFence));
            const bool localValid = validPublicationFenceRecord(
                *state, slot, *ownerFence, slot.publicationBinding,
                MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED);
            const bool humanValid = slot.humanIOCandidateBound &&
                slot.humanIORootReserved &&
                slot.humanIOCandidate.valid() &&
                slot.humanIOBinding.bindingFingerprint != 0u &&
                slot.humanIOBinding.jointCommitFingerprint ==
                    ownerFence->jointCommitFingerprint &&
                slot.humanIOBinding.brainGeneration ==
                    ownerFence->brainGeneration &&
                slot.humanIOBinding.candidatePublicationFingerprint ==
                    slot.humanIOCandidate.
                        candidatePublicationFingerprint &&
                slot.humanIOBinding.deviceRegistryID ==
                    state->device.registryID;
            if (!localValid || !humanValid) {
                // The exact owning release follows the nonthrow Brain pointer
                // flip. Any fence mismatch is one-shot terminal; the valid
                // reservation and all prepared authority remain quarantined.
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
            const bool matterReleased =
                state->config.matterRuntime->releasePublishedRoot(
                    slot.publicationReservation, matterFence);
            if (!matterReleased) {
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
            MetalNumanXHumanIOCandidatePublicationCommit humanCommit{};
            humanCommit.candidatePublicationFingerprint =
                slot.humanIOCandidate.candidatePublicationFingerprint;
            humanCommit.bindingFingerprint =
                slot.humanIOBinding.bindingFingerprint;
            humanCommit.jointCommitFingerprint =
                ownerFence->jointCommitFingerprint;
            humanCommit.brainGeneration = ownerFence->brainGeneration;
            humanCommit.fenceFingerprint = ownerFence->fenceFingerprint;
            if (slot.humanIOCandidate.publishCandidate(
                    slot.humanIOCandidate.context,
                    slot.humanIOCandidate.candidatePublicationFingerprint,
                    humanCommit) !=
                MetalNumanXHumanIOCandidatePublicationDisposition::released) {
                // The cross-runtime bridge still holds its aggregate reader
                // gate, so this internal split can never become a public root.
                // Retain the owner generation as terminal-no-touch.
                slot.stage = SlotStage::terminalNoTouch;
                return Disposition::terminalNoTouch;
            }
            slot.leaseAcquired = false;
            slot.lease = {};
            slot.commandBufferIdentity = 0u;
            slot.applyCommandBufferIdentity = 0u;
            slot.matterOpened = false;
            slot.cancelIssued = false;
            slot.completionArmed = false;
            slot.physicalCommandCompleted = false;
            slot.physicalCommandFailed = false;
            slot.applyEncoded = false;
            slot.applyCommandCompleted = false;
            slot.applyCommandSucceeded = false;
            slot.restoreRequired = false;
            slot.applicationReservation = {};
            slot.publicationBinding = {};
            slot.publicationReservation = {};
            slot.appliedOutcome = {};
            slot.humanIOCandidate = {};
            slot.humanIOBinding = {};
            slot.publicationReserved = false;
            slot.humanIOCandidateBound = false;
            slot.humanIORootReserved = false;
            slot.stage = SlotStage::released;
            maybeReleaseLifetimeHold(*state);
            return Disposition::released;
        } catch (...) {
            return Disposition::terminalNoTouch;
        }
    }
}

} // namespace

MetalNumanXHumanMatterContext::MetalNumanXHumanMatterContext(
    MetalNumanXHumanMatterConfig config
) : state_(std::make_shared<State>(std::move(config))) {
    state_->self = state_;
}

MetalNumanXHumanMatterContext::~MetalNumanXHumanMatterContext() = default;

MetalNumanXHumanMatterContext::MetalNumanXHumanMatterContext(
    MetalNumanXHumanMatterContext&& other
) noexcept = default;

MetalNumanXHumanMatterContext& MetalNumanXHumanMatterContext::operator=(
    MetalNumanXHumanMatterContext&& other
) noexcept = default;

MetalNumanXHumanMatterDiagnostics
MetalNumanXHumanMatterContext::initialize() {
    if (state_ == nullptr) {
        return diagnostics(
            nullptr, MetalNumanXHumanMatterHostStatus::invalidConfiguration,
            "context was moved from");
    }
    std::lock_guard lock(state_->mutex);
    State& state = *state_;
    if (state.initialized) {
        return diagnostics(&state, MetalNumanXHumanMatterHostStatus::success);
    }
    const auto& config = state.config;
    if (config.matterRuntime == nullptr ||
        config.coupledHumanMetallibPath.empty() ||
        config.adapterMetallibPath.empty() ||
        config.environmentCapacity == 0u ||
        config.pointCapacity > MR_NUMANX_COUPLED_HUMAN_MAX_POINTS ||
        config.transactionSlotCount == 0u ||
        config.transactionSlotCount >
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS ||
        config.reserved0 != 0u || config.maximumRetainedBytes == 0u ||
        (config.stateProofProgram.configured() &&
         !config.stateProofProgram.valid())) {
        return diagnostics(
            &state, MetalNumanXHumanMatterHostStatus::invalidConfiguration,
            "runtime, metallibs, capacities, proof program, or reserved fields are invalid");
    }
    if (!config.matterRuntime->valid()) {
        return diagnostics(
            &state, MetalNumanXHumanMatterHostStatus::matterRuntimeUnavailable,
            "Matter Runtime must be initialized before the adapter");
    }
    if (!config.matterRuntime->requiresCoupledCandidate() ||
        config.pointCapacity <
            config.matterRuntime->coupledCandidatePointCapacity()) {
        return diagnostics(
            &state,
            MetalNumanXHumanMatterHostStatus::matterRuntimeIncompatible,
            "Matter Runtime does not expose the required attached coupled-candidate graph or point capacity");
    }
    state.matterSourceFingerprint =
        config.matterRuntime->sourcePhysicsFingerprint();
    state.matterDeviceFingerprint =
        config.matterRuntime->deviceProgramFingerprint();
    const std::uint64_t matterProofFingerprint =
        config.matterRuntime->acceptedStateProofProgramFingerprint();
    if (state.matterSourceFingerprint == 0u ||
        state.matterDeviceFingerprint == 0u ||
        matterProofFingerprint == 0u ||
        (config.stateProofProgram.valid() &&
         config.stateProofProgram.fingerprint != matterProofFingerprint)) {
        return diagnostics(
            &state,
            MetalNumanXHumanMatterHostStatus::matterRuntimeIncompatible,
            "Matter Runtime fingerprints or exact prepared-state proof identity are unavailable");
    }
    @autoreleasepool {
        __unsafe_unretained id<MTLBuffer> matterStatuses =
            (__bridge id<MTLBuffer>)config.matterRuntime->statusBuffer();
        if (matterStatuses == nil || matterStatuses.device == nil ||
            matterStatuses.gpuAddress == 0u ||
            matterStatuses.length <
                static_cast<NSUInteger>(config.environmentCapacity) *
                    sizeof(NMMatterStatusGPU)) {
            return diagnostics(
                &state,
                MetalNumanXHumanMatterHostStatus::matterRuntimeIncompatible,
                "Matter Runtime status authority is missing or undersized");
        }
        state.device = matterStatuses.device;
        NSString* path = nsString(config.adapterMetallibPath);
        BOOL directory = NO;
        if (path == nil || ![[NSFileManager defaultManager]
                fileExistsAtPath:path isDirectory:&directory] || directory) {
            return diagnostics(
                &state, MetalNumanXHumanMatterHostStatus::metallibUnavailable,
                "adapterMetallibPath does not name a readable regular file");
        }
        NSError* imageError = nil;
        NSData* image = [NSData dataWithContentsOfFile:path
                                               options:NSDataReadingMappedIfSafe
                                                 error:&imageError];
        detail::NumanXExecutableImageIdentity imageIdentity{};
        if (image == nil ||
            !detail::numanXExecutableImageIdentity(
                image.bytes, image.length, imageIdentity)) {
            return diagnostics(
                &state, MetalNumanXHumanMatterHostStatus::metallibUnavailable,
                "failed to read the explicit Human/Matter adapter metallib image: " +
                    fromNSString(imageError.localizedDescription));
        }
        dispatch_data_t libraryImage = dispatch_data_create(
            image.bytes,
            image.length,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT
        );
        if (libraryImage == nullptr) {
            return diagnostics(
                &state, MetalNumanXHumanMatterHostStatus::metallibUnavailable,
                "failed to retain the Human/Matter adapter metallib image");
        }
        NSError* libraryError = nil;
        state.library = [state.device
            newLibraryWithData:libraryImage error:&libraryError];
        if (state.library == nil) {
            return diagnostics(
                &state, MetalNumanXHumanMatterHostStatus::metalLibraryFailure,
                "failed to load explicit adapter metallib: " +
                    fromNSString(libraryError.localizedDescription));
        }
        state.metallibIdentity = imageIdentity;
        const auto pipeline = [&] (
            NSString* name,
            __strong id<MTLComputePipelineState>& destination
        ) -> bool {
            id<MTLFunction> function =
                [state.library newFunctionWithName:name];
            if (function == nil) return false;
            NSError* error = nil;
            destination = [state.device
                newComputePipelineStateWithFunction:function error:&error];
            return destination != nil;
        };
        if (!pipeline(@"numanx_human_matter_prepare_world_status",
                      state.prepareWorldStatusPipeline) ||
            !pipeline(@"numanx_human_matter_map_human_status",
                      state.mapHumanStatusPipeline) ||
            !pipeline(@"numanx_human_matter_capture_outcome",
                      state.captureOutcomePipeline) ||
            !pipeline(@"numanx_human_matter_write_prepared_token",
                      state.preparedTokenPipeline)) {
            return diagnostics(
                &state, MetalNumanXHumanMatterHostStatus::metalPipelineFailure,
                "adapter metallib is missing a required NumanX Human/Matter kernel");
        }
    }

    const auto coupled = state.coupledContext.initialize();
    if (!coupled.succeeded()) {
        return diagnostics(
            &state, MetalNumanXHumanMatterHostStatus::coupledHumanFailure,
            "frozen-A0 coupled Human service failed: " + coupled.message);
    }
    state.coupledProgram = state.coupledContext.program();
    if (!state.coupledProgram.valid()) {
        return diagnostics(
            &state, MetalNumanXHumanMatterHostStatus::coupledHumanFailure,
            "frozen-A0 coupled Human service program is invalid");
    }

    std::uint64_t outcomesBytes = 0u, applyOutcomeBytes = 0u;
    std::uint64_t worldBytes = 0u;
    std::uint64_t tokenBytes = 0u, proofBytes = 0u, perSlot = 0u;
    if (!checkedMultiply(config.environmentCapacity,
                         sizeof(MRNumanXCoupledMatterOutcomeGPU),
                         outcomesBytes) ||
        !checkedMultiply(config.environmentCapacity,
                         sizeof(MRMetalWorldStatusGPU), worldBytes) ||
        !checkedMultiply(config.environmentCapacity,
                         sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU),
                         applyOutcomeBytes) ||
        !checkedMultiply(config.environmentCapacity,
                         MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES,
                         tokenBytes) ||
        !checkedMultiply(config.environmentCapacity,
                         MR_NUMANX_ACCEPTED_STATE_PROOF_BYTES,
                         proofBytes) ||
        !checkedAdd(outcomesBytes, applyOutcomeBytes, perSlot) ||
        !checkedAdd(perSlot, worldBytes, perSlot) ||
        !checkedAdd(perSlot, tokenBytes, perSlot) ||
        !checkedAdd(perSlot, proofBytes, perSlot) ||
        !checkedMultiply(perSlot, config.transactionSlotCount,
                         state.retainedBytes) ||
        !checkedAdd(state.retainedBytes, coupled.retainedBytes,
                    state.retainedBytes) ||
        state.retainedBytes > config.maximumRetainedBytes ||
        perSlot > std::numeric_limits<NSUInteger>::max()) {
        return diagnostics(
            &state, MetalNumanXHumanMatterHostStatus::arithmeticOverflow,
            "adapter slot arenas overflow or exceed maximumRetainedBytes");
    }

    std::vector<Slot> slots(config.transactionSlotCount);
    for (std::uint32_t index = 0u; index < config.transactionSlotCount;
         ++index) {
        const auto arena = state.coupledContext.arenaView(
            index, slots[index].coupledArena);
        if (!arena.succeeded()) {
            return diagnostics(
                &state,
                MetalNumanXHumanMatterHostStatus::coupledHumanFailure,
                "failed to resolve coupled Human slot arena: " +
                    arena.message);
        }
        @autoreleasepool {
            slots[index].matterOutcomes = [state.device
                newBufferWithLength:static_cast<NSUInteger>(outcomesBytes)
                           options:MTLResourceStorageModePrivate];
            slots[index].matterApplyOutcomes = [state.device
                newBufferWithLength:static_cast<NSUInteger>(applyOutcomeBytes)
                           options:MTLResourceStorageModePrivate];
            slots[index].worldStatuses = [state.device
                newBufferWithLength:static_cast<NSUInteger>(worldBytes)
                           options:MTLResourceStorageModePrivate];
            slots[index].acceptedTokens = [state.device
                newBufferWithLength:static_cast<NSUInteger>(tokenBytes)
                           options:MTLResourceStorageModePrivate];
            slots[index].acceptedStateProofs = [state.device
                newBufferWithLength:static_cast<NSUInteger>(proofBytes)
                           options:MTLResourceStorageModePrivate];
            if (slots[index].matterOutcomes == nil ||
                slots[index].matterApplyOutcomes == nil ||
                slots[index].worldStatuses == nil ||
                slots[index].acceptedTokens == nil ||
                slots[index].acceptedStateProofs == nil ||
                slots[index].matterOutcomes.gpuAddress == 0u ||
                slots[index].matterApplyOutcomes.gpuAddress == 0u ||
                slots[index].worldStatuses.gpuAddress == 0u ||
                slots[index].acceptedTokens.gpuAddress == 0u ||
                slots[index].acceptedStateProofs.gpuAddress == 0u ||
                !exactBuffer(state.device,
                    slots[index].coupledArena.matterGeneralizedReaction,
                    slots[index].coupledArena.reactionGPUAddress,
                    slots[index].coupledArena.reactionByteCount) ||
                !exactBuffer(state.device,
                    slots[index].coupledArena.jointStatuses,
                    slots[index].coupledArena.jointStatusGPUAddress,
                    slots[index].coupledArena.jointStatusByteCount) ||
                !exactBuffer(state.device,
                    slots[index].coupledArena.transactionMetadata,
                    slots[index].coupledArena.
                        transactionMetadataGPUAddress,
                    slots[index].coupledArena.
                        transactionMetadataByteCount)) {
                return diagnostics(
                    &state,
                    MetalNumanXHumanMatterHostStatus::metalBufferFailure,
                    "failed to allocate exact same-device adapter arenas");
            }
            slots[index].matterOutcomes.label = [NSString stringWithFormat:
                @"NumanX coupled Matter status slot %u", index];
            slots[index].matterApplyOutcomes.label =
                [NSString stringWithFormat:
                    @"NumanX Matter apply outcome slot %u", index];
            slots[index].worldStatuses.label = [NSString stringWithFormat:
                @"NumanX world status slot %u", index];
            slots[index].acceptedTokens.label = [NSString stringWithFormat:
                @"NumanX accepted physics token slot %u", index];
            slots[index].acceptedStateProofs.label = [NSString stringWithFormat:
                @"NumanX accepted state proof slot %u", index];
        }
    }
    state.slots = std::move(slots);
    if (!fixedAuthoritiesPairwiseDisjoint(state)) {
        return diagnostics(
            &state, MetalNumanXHumanMatterHostStatus::metalBufferFailure,
            "adapter/CoupledHuman private authority arenas overlap");
    }
    std::uint64_t fingerprint = kFNVOffset;
    mixValue(fingerprint,
        static_cast<std::uint64_t>(
            MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION));
    mixValue(fingerprint, state.matterSourceFingerprint);
    mixValue(fingerprint, state.matterDeviceFingerprint);
    mixValue(fingerprint, state.coupledProgram.fingerprint);
    mixValue(fingerprint, config.stateProofProgram.valid()
        ? config.stateProofProgram.fingerprint : 0u);
    mixValue(fingerprint,
        static_cast<std::uint64_t>(config.environmentCapacity));
    mixValue(fingerprint,
        static_cast<std::uint64_t>(config.pointCapacity));
    mixValue(fingerprint,
        static_cast<std::uint64_t>(config.transactionSlotCount));
    mixValue(fingerprint, state.metallibIdentity.byteFingerprint);
    mixValue(fingerprint, state.metallibIdentity.byteCount);
    state.fingerprint = nonzeroHash(fingerprint);
    state.initialized = true;
    return diagnostics(&state, MetalNumanXHumanMatterHostStatus::success);
}

MetalNumanXHumanMatterProgram MetalNumanXHumanMatterContext::program(
    const MetalNumanXHumanMatterTransaction& transaction
) noexcept {
    MetalNumanXHumanMatterProgram result{};
    if (state_ == nullptr || !state_->initialized ||
        !transactionValid(*state_, transaction)) return result;
    std::lock_guard lock(state_->mutex);
    Slot& slot = state_->slots[transaction.transactionSlot];
    if ((slot.stage != SlotStage::empty &&
         slot.stage != SlotStage::released) ||
        slot.leaseAcquired || slot.awaitingFirstAbort ||
        (slot.transaction.slotGeneration != 0u &&
         transaction.slotGeneration <=
            slot.transaction.slotGeneration)) return result;
    slot.transaction = transaction;
    slot.lease = {};
    slot.stage = SlotStage::prepared;
    slot.commandBufferIdentity = 0u;
    slot.applyCommandBufferIdentity = 0u;
    slot.passSignature = 0u;
    slot.applyAttempt = 0u;
    slot.physicalAuthorityRegions = {};
    slot.physicalAuthorityRegionCount = 0u;
    slot.applicationReservation = {};
    slot.publicationBinding = {};
    slot.publicationReservation = {};
    slot.appliedOutcome = {};
    slot.humanIOCandidate = {};
    slot.humanIOBinding = {};
    slot.publicationReserved = false;
    slot.humanIOCandidateBound = false;
    slot.humanIORootReserved = false;
    slot.matterOpened = false;
    slot.cancelIssued = false;
    slot.completionArmed = false;
    slot.leaseAcquired = false;
    slot.physicalCommandCompleted = false;
    slot.physicalCommandFailed = false;
    slot.applyEncoded = false;
    slot.applyCommandCompleted = false;
    slot.applyCommandSucceeded = false;
    slot.restoreRequired = false;
    slot.awaitingFirstAbort = false;
    slot.callbackFrame = nullptr;
    state_->lifetimeHold = state_;

    result.capabilities =
        MetalNumanXHumanMatterExactCandidateKinematics |
        MetalNumanXHumanMatterSourceEffectiveTangent |
        MetalNumanXHumanMatterStagedReaction |
        MetalNumanXHumanMatterJointDecision |
        MetalNumanXHumanMatterPreparedPhysicsGate;
    result.accessFlags =
        MetalNumanXHumanMatterReadLiveHumanState |
        MetalNumanXHumanMatterReadHumanCheckpoints |
        MetalNumanXHumanMatterReadSourceEffectiveTangent |
        MetalNumanXHumanMatterMayEncodeExactCandidate |
        MetalNumanXHumanMatterWriteStagedReaction |
        MetalNumanXHumanMatterWriteJointStatus |
        MetalNumanXHumanMatterWritePreparedPhysicsToken;
    result.context = state_.get();
    result.encode = &encodeCallback;
    result.abort = &abortCallback;
    result.acquirePrepareLease = &acquirePrepareLeaseCallback;
    result.bindHumanIOCandidatePublication =
        &bindHumanIOCandidatePublicationCallback;
    result.releasePrepareLease = &releasePrepareLeaseCallback;
    result.reservePreparedApplication =
        &reservePreparedApplicationCallback;
    result.encodePreparedApply = &encodePreparedApplyCallback;
    result.abortPreparedApply = &abortPreparedApplyCallback;
    result.reservePublishedRoot = &reservePublishedRootCallback;
    result.releasePublishedRoot = &releasePublishedRootCallback;
    result.fingerprint = state_->fingerprint;
    result.matterGeneralizedReaction =
        slot.coupledArena.matterGeneralizedReaction;
    result.jointStatuses = slot.coupledArena.jointStatuses;
    result.acceptedPhysicsStateTokens =
        (__bridge void*)slot.acceptedTokens;
    result.matterApplyOutcomes =
        (__bridge void*)slot.matterApplyOutcomes;
    result.matterGeneralizedReactionGPUAddress =
        slot.coupledArena.reactionGPUAddress;
    result.jointStatusesGPUAddress =
        slot.coupledArena.jointStatusGPUAddress;
    result.acceptedPhysicsStateTokensGPUAddress =
        slot.acceptedTokens.gpuAddress;
    result.matterApplyOutcomesGPUAddress =
        slot.matterApplyOutcomes.gpuAddress;
    result.matterGeneralizedReactionElementCount =
        slot.coupledArena.reactionByteCount / sizeof(float);
    result.jointStatusElementCount =
        slot.coupledArena.jointStatusByteCount /
            sizeof(MRNumanXCoupledHumanStatusGPU);
    result.acceptedPhysicsStateTokenByteCount = slot.acceptedTokens.length;
    result.matterApplyOutcomeElementCount = transaction.environmentCount;
    result.environmentCount = transaction.environmentCount;
    result.reactionStride = slot.coupledArena.reactionStride;
    result.jointStatusStride = slot.coupledArena.jointStatusStride;
    result.acceptedTokenStrideBytes =
        MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
    result.matterApplyOutcomeStride = 1u;
    result.transactionSlot = transaction.transactionSlot;
    result.substepIndex = transaction.physicsSubstep;
    result.physicsSubstepCount = transaction.physicsSubsteps;
    // The owner uses this otherwise-reserved word as the exact suffix point
    // capacity for candidate attachment queries.
    result.candidatePointCapacity = state_->config.pointCapacity;
    result.controlStep = transaction.controlStep;
    result.qCoordinateCount = transaction.qCoordinateCount;
    result.dofCount = transaction.dofCount;
    result.dofLayoutVersion = transaction.dofLayoutVersion;
    result.transactionFingerprint = transaction.transactionFingerprint;
    result.linearizationEpoch = transaction.linearizationEpoch;
    result.slotGeneration = transaction.slotGeneration;
    if (!result.valid()) {
        slot.stage = SlotStage::released;
        maybeReleaseLifetimeHold(*state_);
        return {};
    }
    return result;
}

const char* metalNumanXHumanMatterHostStatusName(
    const MetalNumanXHumanMatterHostStatus status
) noexcept {
    switch (status) {
    case MetalNumanXHumanMatterHostStatus::success: return "success";
    case MetalNumanXHumanMatterHostStatus::invalidConfiguration:
        return "invalid_configuration";
    case MetalNumanXHumanMatterHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalNumanXHumanMatterHostStatus::matterRuntimeUnavailable:
        return "matter_runtime_unavailable";
    case MetalNumanXHumanMatterHostStatus::matterRuntimeIncompatible:
        return "matter_runtime_incompatible";
    case MetalNumanXHumanMatterHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalNumanXHumanMatterHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalNumanXHumanMatterHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalNumanXHumanMatterHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalNumanXHumanMatterHostStatus::coupledHumanFailure:
        return "coupled_human_failure";
    case MetalNumanXHumanMatterHostStatus::uninitialized:
        return "uninitialized";
    case MetalNumanXHumanMatterHostStatus::invalidTransaction:
        return "invalid_transaction";
    case MetalNumanXHumanMatterHostStatus::slotBusy: return "slot_busy";
    }
    return "unknown";
}

} // namespace metalrobo
