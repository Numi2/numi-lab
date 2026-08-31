#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/accepted_state_apply_gpu.h"
#include "accepted_state_proof_gpu.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"
#include "metalrobo/numanx_human_matter_gpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_DEFAULT_METALLIB
#define NUMI_MATTER_DEFAULT_METALLIB ""
#endif

namespace numi::matter {
namespace {

static_assert(
    NM_MATTER_MAX_HUMAN_ATTACHMENT_POINTS ==
        MR_ARTICULATED_OPERATOR_MAX_POINTS
);
static_assert(
    NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION ==
        MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION
);
static_assert(sizeof(NMAcceptedStateProofGPU) ==
              sizeof(MRNumanXAcceptedStateProofGPU));
static_assert(alignof(NMAcceptedStateProofGPU) ==
              alignof(MRNumanXAcceptedStateProofGPU));
#define NM_ASSERT_PROOF_FIELD_LAYOUT(field) \
    static_assert(offsetof(NMAcceptedStateProofGPU, field) == \
                  offsetof(MRNumanXAcceptedStateProofGPU, field))
NM_ASSERT_PROOF_FIELD_LAYOUT(abiVersion);
NM_ASSERT_PROOF_FIELD_LAYOUT(structSize);
NM_ASSERT_PROOF_FIELD_LAYOUT(status);
NM_ASSERT_PROOF_FIELD_LAYOUT(environment);
NM_ASSERT_PROOF_FIELD_LAYOUT(transactionFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(substepFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(acceptedTimestampMicroseconds);
NM_ASSERT_PROOF_FIELD_LAYOUT(physicsGeneration);
NM_ASSERT_PROOF_FIELD_LAYOUT(humanStateFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(matterStateFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(physicsStateFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(matterSourcePhysicsFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(matterDeviceProgramFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(linearizationEpoch);
NM_ASSERT_PROOF_FIELD_LAYOUT(slotGeneration);
NM_ASSERT_PROOF_FIELD_LAYOUT(proofFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(adapterProgramFingerprint);
NM_ASSERT_PROOF_FIELD_LAYOUT(transactionPolicyFingerprint);
#undef NM_ASSERT_PROOF_FIELD_LAYOUT
static_assert(static_cast<std::uint32_t>(NM_ACCEPTED_STATE_PROOF_PENDING) ==
              static_cast<std::uint32_t>(
                  MR_NUMANX_ACCEPTED_STATE_PROOF_PENDING));
static_assert(static_cast<std::uint32_t>(NM_ACCEPTED_STATE_PROOF_VALID) ==
              static_cast<std::uint32_t>(
                  MR_NUMANX_ACCEPTED_STATE_PROOF_VALID));
static_assert(static_cast<std::uint32_t>(NM_ACCEPTED_STATE_PROOF_REJECTED) ==
              static_cast<std::uint32_t>(
                  MR_NUMANX_ACCEPTED_STATE_PROOF_REJECTED));
static_assert(NM_MATTER_OWNER_APPLY_ABI_VERSION ==
              MR_NUMANX_HUMAN_MATTER_ABI_VERSION);
static_assert(NM_MATTER_OWNER_BRAIN_ACK_ABI_VERSION ==
              MR_NUMANX_HUMAN_MATTER_BRAIN_ACK_ABI_VERSION);
static_assert(sizeof(NMOwnerProposalGPU) ==
              sizeof(MRNumanXHumanMatterProposalGPU));
static_assert(sizeof(NMOwnerBrainAckGPU) ==
              sizeof(MRNumanXHumanMatterBrainAckGPU));
static_assert(sizeof(NMOwnerApplyActionGPU) ==
              sizeof(MRNumanXHumanMatterApplyActionGPU));
static_assert(sizeof(NMMatterApplyOutcomeGPU) ==
              sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU));
static_assert(sizeof(NMOwnerAppliedOutcomeGPU) ==
              sizeof(MRNumanXHumanMatterAppliedOutcomeGPU));
static_assert(sizeof(NMJointPublicationFenceGPU) ==
              sizeof(MRNumanXHumanMatterJointPublicationFenceGPU));
#define NM_ASSERT_ABI4_FIELD_LAYOUT(nmType, mrType, field) \
    static_assert(offsetof(nmType, field) == offsetof(mrType, field))
NM_ASSERT_ABI4_FIELD_LAYOUT(NMOwnerProposalGPU,
    MRNumanXHumanMatterProposalGPU, candidatePublicationFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMOwnerProposalGPU,
    MRNumanXHumanMatterProposalGPU, humanIOIdentityFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMOwnerProposalGPU,
    MRNumanXHumanMatterProposalGPU, proposalFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMOwnerBrainAckGPU,
    MRNumanXHumanMatterBrainAckGPU, ackFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMOwnerApplyActionGPU,
    MRNumanXHumanMatterApplyActionGPU, actionFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMMatterApplyOutcomeGPU,
    MRNumanXHumanMatterMatterApplyOutcomeGPU, outcomeFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMOwnerAppliedOutcomeGPU,
    MRNumanXHumanMatterAppliedOutcomeGPU, appliedFingerprint);
NM_ASSERT_ABI4_FIELD_LAYOUT(NMJointPublicationFenceGPU,
    MRNumanXHumanMatterJointPublicationFenceGPU, fenceFingerprint);
#undef NM_ASSERT_ABI4_FIELD_LAYOUT

const char kImageAnchor = 0;

[[nodiscard]] std::string nsString(NSString* value) {
    return value != nil && value.UTF8String != nullptr
        ? std::string(value.UTF8String)
        : std::string{};
}

[[nodiscard]] std::string errorString(NSError* error) {
    return error == nil
        ? std::string{"unknown Metal error"}
        : nsString(error.localizedDescription);
}

[[nodiscard]] bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

[[nodiscard]] std::filesystem::path defaultMetallib() {
    Dl_info image{};
    if (dladdr(&kImageAnchor, &image) != 0 && image.dli_fname != nullptr) {
        const std::filesystem::path directory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            directory / "numi-matter/NumiMatter.metallib",
            directory.parent_path() / "shaders/NumiMatter.metallib",
            directory / "NumiMatter.metallib",
        };
        for (const auto& candidate : candidates) {
            if (regularFile(candidate)) {
                return candidate;
            }
        }
    }
    const std::filesystem::path configured{NUMI_MATTER_DEFAULT_METALLIB};
    return regularFile(configured) ? configured : std::filesystem::path{};
}

[[nodiscard]] std::uint64_t mixFingerprint(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (8u * byte)) & 0xffu;
        hash *= prime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t mixFingerprint32(
    std::uint64_t hash,
    const std::uint32_t value
) noexcept {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        hash ^= (value >> (8u * byte)) & 0xffu;
        hash *= prime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t publicationFenceFingerprint(
    const PreparedStatePublicationFence& fence
) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = mixFingerprint32(hash, fence.abiVersion);
    hash = mixFingerprint32(hash, fence.structBytes);
    hash = mixFingerprint32(hash, fence.status);
    hash = mixFingerprint32(hash, fence.environment);
    hash = mixFingerprint32(hash, fence.controlStep);
    hash = mixFingerprint32(hash, fence.substepIndex);
    hash = mixFingerprint32(hash, fence.physicsSubstepCount);
    hash = mixFingerprint32(hash, fence.reserved0);
    hash = mixFingerprint(hash, fence.ownerProgramFingerprint);
    hash = mixFingerprint(hash, fence.transactionFingerprint);
    hash = mixFingerprint(hash, fence.linearizationEpoch);
    hash = mixFingerprint(hash, fence.slotGeneration);
    hash = mixFingerprint(hash, fence.physicsTokenFingerprint);
    hash = mixFingerprint(hash, fence.brainProgramFingerprint);
    hash = mixFingerprint(hash, fence.brainShadowStateFingerprint);
    hash = mixFingerprint(hash, fence.brainWitnessFingerprint);
    hash = mixFingerprint(hash, fence.appliedDecisionFingerprint);
    hash = mixFingerprint(hash, fence.jointCommitFingerprint);
    hash = mixFingerprint(hash, fence.brainGeneration);
    return hash == 0u ? 14695981039346656037ull : hash;
}

[[nodiscard]] std::uint64_t publicationFactsFingerprint(
    const NMPreparedStatePublicationFactsGPU& facts
) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = mixFingerprint32(hash, facts.abiVersion);
    hash = mixFingerprint32(hash, facts.status);
    hash = mixFingerprint32(hash, facts.reserved0);
    hash = mixFingerprint32(hash, facts.reserved1);
    hash = mixFingerprint(hash, facts.physicsTokenFingerprint);
    hash = mixFingerprint(hash, facts.brainProgramFingerprint);
    hash = mixFingerprint(hash, facts.brainShadowStateFingerprint);
    hash = mixFingerprint(hash, facts.brainWitnessFingerprint);
    hash = mixFingerprint(hash, facts.matterApplyFingerprint);
    return hash == 0u ? 14695981039346656037ull : hash;
}

[[nodiscard]] std::uint64_t publicationReservationFingerprint(
    const PreparedStatePublicationReservation& reservation
) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = mixFingerprint32(hash, reservation.abiVersion);
    hash = mixFingerprint32(hash, reservation.structSize);
    hash = mixFingerprint32(hash, reservation.transactionSlot);
    hash = mixFingerprint32(hash, reservation.reserved0);
    hash = mixFingerprint(hash, reservation.transactionFingerprint);
    hash = mixFingerprint(hash, reservation.slotGeneration);
    hash = mixFingerprint(hash, reservation.reservationNonce);
    hash = mixFingerprint(hash, reservation.reserved1);
    hash = mixFingerprint(hash, reservation.reserved2);
    return hash == 0u ? 14695981039346656037ull : hash;
}

[[nodiscard]] std::uint64_t makeDeviceProgramFingerprint(
    const std::uint64_t worldFingerprint,
    const std::uint64_t metallibFingerprint,
    const std::uint64_t metallibByteCount,
    const RuntimeConfiguration& configuration
) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = mixFingerprint(hash, worldFingerprint);
    hash = mixFingerprint(hash, NM_MATTER_ABI_VERSION);
    hash = mixFingerprint(hash, metallibFingerprint);
    hash = mixFingerprint(hash, metallibByteCount);
    hash = mixFingerprint(hash, configuration.captureEvents ? 1u : 0u);
    hash = mixFingerprint(hash, configuration.captureDiagnostics ? 1u : 0u);
    hash = mixFingerprint(hash, configuration.automaticIdentification ? 1u : 0u);
    hash = mixFingerprint(hash, configuration.adaptiveTransfer ? 1u : 0u);
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] std::uint64_t acceptedStateProofSourceFingerprint() noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = mixFingerprint(hash, 0x4e584d50524f4f46ull); // "NXMPROOF"
    hash = mixFingerprint(
        hash, detail::kAcceptedStateProofSchemaVersion);
    hash = mixFingerprint(
        hash, detail::kAcceptedStateProofManifestVersion);
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] std::uint64_t makeAcceptedStateProofProgramFingerprint(
    const std::uint64_t executionFingerprint,
    const NMMatterDispatchGPU& dispatch,
    const std::uint32_t identificationDistributionCount,
    const std::uint32_t femNodeIncidenceStride,
    const std::uint32_t femNodeRangeStride,
    const std::uint64_t mujocoBytesPerEnvironmentCapacity
) noexcept {
    std::uint64_t hash = acceptedStateProofSourceFingerprint();
    hash = mixFingerprint(
        hash, NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION);
    hash = mixFingerprint(hash, sizeof(NMAcceptedStateProofGPU));
    hash = mixFingerprint(hash, executionFingerprint);
    hash = mixFingerprint(hash, dispatch.environmentCount);
    hash = mixFingerprint(hash, dispatch.objectCount);
    hash = mixFingerprint(hash, dispatch.parameterCount);
    hash = mixFingerprint(hash, dispatch.particleCount);
    hash = mixFingerprint(hash, dispatch.femNodeCount);
    hash = mixFingerprint(hash, dispatch.tetrahedronCount);
    hash = mixFingerprint(hash, dispatch.topologyNodeCapacity);
    hash = mixFingerprint(hash, dispatch.cohesiveFaceCount);
    hash = mixFingerprint(hash, dispatch.punctureChannelCount);
    hash = mixFingerprint(hash, dispatch.learnedWeightCount);
    hash = mixFingerprint(hash, dispatch.contactPairCount);
    hash = mixFingerprint(hash, dispatch.deformableContactCapacity);
    hash = mixFingerprint(hash, dispatch.rigidGeneralizedCapacity);
    hash = mixFingerprint(hash, dispatch.rigidProxyCount);
    hash = mixFingerprint(hash, dispatch.materialStateStride);
    hash = mixFingerprint(hash, identificationDistributionCount);
    hash = mixFingerprint(hash, femNodeIncidenceStride);
    hash = mixFingerprint(hash, femNodeRangeStride);
    hash = mixFingerprint(hash, mujocoBytesPerEnvironmentCapacity);
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] NSUInteger checkedBytes(
    const std::size_t count,
    const std::size_t elementSize,
    bool& valid
) {
    if (count != 0u && elementSize >
        std::numeric_limits<std::size_t>::max() / count) {
        valid = false;
        return 0u;
    }
    const std::size_t bytes = count * elementSize;
    if (bytes > std::numeric_limits<NSUInteger>::max()) {
        valid = false;
        return 0u;
    }
    // Metal validates a typed device pointer against at least one complete
    // pointee even when the logical dispatch width is zero. A generic
    // 16-byte placeholder is therefore insufficient for empty bindings such
    // as NMParticleStateGPU (160 bytes), and validation correctly rejects the
    // command before the kernel's zero-work guard can run. Preserve the
    // logical zero count while allocating one fully typed sentinel.
    return static_cast<NSUInteger>(std::max({
        bytes,
        elementSize,
        std::size_t{16u},
    }));
}

class UploadPlan {
public:
    UploadPlan(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : device_(device), commandBuffer_([queue commandBuffer]),
          staging_([NSMutableArray array]) {
        blit_ = [commandBuffer_ blitCommandEncoder];
    }

    template <typename T>
    [[nodiscard]] id<MTLBuffer> repeated(
        const std::span<const T> values,
        const std::size_t repeat,
        bool& valid,
        std::size_t& residentBytes
    ) {
        const NSUInteger oneBytes = checkedBytes(values.size(), sizeof(T), valid);
        const NSUInteger totalBytes = checkedBytes(values.size() * repeat, sizeof(T), valid);
        if (!valid) {
            return nil;
        }
        id<MTLBuffer> output = [device_
            newBufferWithLength:totalBytes
                        options:MTLResourceStorageModePrivate];
        if (output == nil) {
            valid = false;
            return nil;
        }
        residentBytes += totalBytes;
        if (values.empty() || repeat == 0u) {
            return output;
        }
        id<MTLBuffer> staging = [device_
            newBufferWithBytes:values.data()
                       length:oneBytes
                      options:MTLResourceStorageModeShared];
        if (staging == nil) {
            valid = false;
            return nil;
        }
        [staging_ addObject:staging];
        const NSUInteger logicalBytes = values.size_bytes();
        for (std::size_t index = 0u; index < repeat; ++index) {
            [blit_ copyFromBuffer:staging
                    sourceOffset:0u
                        toBuffer:output
               destinationOffset:static_cast<NSUInteger>(index) * logicalBytes
                            size:logicalBytes];
        }
        return output;
    }

    template <typename T>
    [[nodiscard]] id<MTLBuffer> one(
        const std::span<const T> values,
        bool& valid,
        std::size_t& residentBytes
    ) {
        return repeated(values, 1u, valid, residentBytes);
    }

    [[nodiscard]] bool finish(std::string& error) {
        [blit_ endEncoding];
        [commandBuffer_ commit];
        [commandBuffer_ waitUntilCompleted];
        if (commandBuffer_.status == MTLCommandBufferStatusError) {
            error = errorString(commandBuffer_.error);
            return false;
        }
        return true;
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandBuffer> commandBuffer_ = nil;
    id<MTLBlitCommandEncoder> blit_ = nil;
    NSMutableArray<id<MTLBuffer>>* staging_ = nil;
};

template <typename T>
[[nodiscard]] id<MTLBuffer> privateScratch(
    id<MTLDevice> device,
    const std::size_t count,
    bool& valid,
    std::size_t& residentBytes
) {
    const NSUInteger bytes = checkedBytes(count, sizeof(T), valid);
    if (!valid) {
        return nil;
    }
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes
                    options:MTLResourceStorageModePrivate];
    if (buffer == nil) {
        valid = false;
        return nil;
    }
    residentBytes += bytes;
    return buffer;
}

template <typename T>
[[nodiscard]] id<MTLBuffer> sharedScratch(
    id<MTLDevice> device,
    const std::size_t count,
    bool& valid,
    std::size_t& residentBytes
) {
    const NSUInteger bytes = checkedBytes(count, sizeof(T), valid);
    if (!valid) {
        return nil;
    }
    id<MTLBuffer> buffer = [device
        newBufferWithLength:bytes
                    options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        valid = false;
        return nil;
    }
    std::memset(buffer.contents, 0, bytes);
    residentBytes += bytes;
    return buffer;
}


} // namespace

struct Runtime::State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    id<MTLComputePipelineState> acceptedStateProofBegin = nil;
    id<MTLComputePipelineState> acceptedStateProofChunks = nil;
    id<MTLComputePipelineState> acceptedStateProofReduce = nil;
    id<MTLComputePipelineState> acceptedStateProofFold = nil;
    id<MTLComputePipelineState> acceptedStateProofFinalize = nil;
    id<MTLComputePipelineState> preparedStateValidateApplication = nil;
    id<MTLComputePipelineState> preparedStateNormalizeApplication = nil;
    id<MTLComputePipelineState> preparedStateMaterializeRestoreStatuses = nil;
    id<MTLComputePipelineState> preparedStateRestoreBytes = nil;
    id<MTLBuffer> acceptedStateProofHashes = nil;
    id<MTLBuffer> acceptedStateProofScratchA = nil;
    id<MTLBuffer> acceptedStateProofScratchB = nil;
    id<MTLBuffer> preparedStateBindings = nil;
    id<MTLBuffer> preparedStateActions = nil;
    id<MTLBuffer> preparedStateRestoreStatuses = nil;
    id<MTLBuffer> preparedStateApplyOutcome = nil;
    id<MTLBuffer> preparedStatePublicationFacts = nil;
    std::uint32_t acceptedStateProofScratchStride = 0u;

    NMMatterDispatchGPU dispatch{};
    NMMixedSolverGPU mixedSolverValue{};
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t executionFingerprint = 0u;
    std::uint64_t acceptedStateProofProgramFingerprint = 0u;
    std::uint64_t acceptedStateProofMujocoBytesPerEnvironmentCapacity = 0u;
    std::size_t acceptedStateProofResidentByteCount = 0u;
    std::size_t residentBytes = 0u;
    std::uint32_t identificationDistributionCount = 0u;
    std::uint32_t femNodeIncidenceStride = 0u;
    std::uint32_t femNodeRangeStride = 0u;
    std::uint32_t requiredCurrentBodyCount = 0u;
    std::uint32_t requiredBodyWrenchCount = 0u;
    std::uint32_t requiredSceneBodyCount = 0u;
    std::uint32_t requiredRodNodeCount = 0u;
    std::uint32_t requiredCandidateBodyCount = 0u;
    std::uint32_t reactionBodyCount = 0u;
    struct CommandOwnership {
        std::mutex mutex;
        void* activeCommandBuffer = nullptr;
        bool preDynamicsOpen = false;
        std::uint32_t controlStep = 0u;
        std::uint32_t physicsSubstep = 0u;
        std::uint32_t identificationGeneration = 0u;
        std::uint32_t identificationCheckpoint = 0u;
        bool identificationAdvanced = false;
        bool acceptedStateProofEncoded = false;
        bool acceptedStateProofEligible = false;
        std::uint64_t transactionPolicyFingerprint = 0u;
        std::uint32_t physicsSubsteps = 1u;
        bool preparedStateOpen = false;
        bool applyEncoded = false;
        bool restoreRequired = false;
        bool preparedStateRequested = false;
        bool preparedCommandCompleted = false;
        bool preparedCommandFailed = false;
        bool terminalNoTouch = false;
        bool applyCommandCompleted = false;
        bool publicationReserved = false;
        // Scalar, non-retaining identity of the provisional Human status
        // stream used by the prepared post-commit pass. The proof pass must
        // consume this exact object/range before the borrowed CB is queued.
        void* preparedEnvironmentStatuses = nullptr;
        std::uint64_t preparedEnvironmentStatusesGPUAddress = 0u;
        PreparedStatePublicationBinding publicationBinding{};
        PreparedStatePublicationReservation publicationReservation{};
        NMPreparedStatePublicationFactsGPU publicationFacts{};
        std::uint64_t publicationReservationCounter = 0u;
        void* applyCommandBuffer = nullptr;
        void* preparedCommandBuffer = nullptr;
        PreparedStateDispositionIdentity dispositionIdentity{};
        PreparedStateDisposition disposition =
            PreparedStateDisposition::unknown;

        [[nodiscard]] bool matchesDispositionIdentity(
            const PreparedStateDispositionIdentity& identity
        ) const noexcept {
            return identity.abiVersion == 1u &&
                identity.structSize == sizeof(PreparedStateDispositionIdentity) &&
                identity.reserved0 == 0u && identity.reserved1 == 0u &&
                dispositionIdentity.abiVersion == identity.abiVersion &&
                dispositionIdentity.structSize == identity.structSize &&
                dispositionIdentity.controlStep == identity.controlStep &&
                dispositionIdentity.physicsSubstep == identity.physicsSubstep &&
                dispositionIdentity.physicsSubstepCount ==
                    identity.physicsSubstepCount &&
                dispositionIdentity.transactionSlot == identity.transactionSlot &&
                dispositionIdentity.reserved0 == identity.reserved0 &&
                dispositionIdentity.reserved1 == identity.reserved1 &&
                dispositionIdentity.ownerProgramFingerprint ==
                    identity.ownerProgramFingerprint &&
                dispositionIdentity.transactionFingerprint ==
                    identity.transactionFingerprint &&
                dispositionIdentity.linearizationEpoch ==
                    identity.linearizationEpoch &&
                dispositionIdentity.slotGeneration == identity.slotGeneration;
        }

        // Must be called with mutex held. The completion handler and exact
        // disposition query share this transition so callback ordering cannot
        // change the lifecycle result.
        void settleApplication(
            void* identity,
            const MTLCommandBufferStatus status,
            const std::uint32_t outcome,
            const NMPreparedStatePublicationFactsGPU* facts
        ) noexcept {
            if (applyCommandBuffer != identity) return;
            applyCommandBuffer = nullptr;
            applyEncoded = false;
            if (status != MTLCommandBufferStatusCompleted) {
                publicationReserved = false;
                publicationBinding = {};
                publicationReservation = {};
                publicationFacts = {};
                acceptedStateProofEligible = false;
                // Owner completion publishes a terminal Applied record and
                // zero token before liveness advances. A failed apply CB may
                // have partially mutated both owners, so neither checkpoint
                // may be restored.
                terminalNoTouch = true;
                restoreRequired = true;
                disposition = PreparedStateDisposition::terminalNoTouch;
                return;
            }
            if (outcome == NM_PREPARED_STATE_APPLY_TERMINAL_NO_TOUCH) {
                publicationReserved = false;
                publicationBinding = {};
                publicationReservation = {};
                publicationFacts = {};
                terminalNoTouch = true;
                restoreRequired = true;
                acceptedStateProofEligible = false;
                disposition = PreparedStateDisposition::terminalNoTouch;
                return;
            }
            if ((outcome != NM_PREPARED_STATE_APPLY_RESOLVED &&
                 outcome !=
                    NM_PREPARED_STATE_APPLY_ACCEPTED_PENDING_PUBLICATION) ||
                preparedCommandFailed) {
                publicationReserved = false;
                publicationBinding = {};
                publicationReservation = {};
                publicationFacts = {};
                restoreRequired = true;
                acceptedStateProofEligible = false;
                disposition = PreparedStateDisposition::restoreRequired;
                return;
            }
            if (outcome ==
                    NM_PREPARED_STATE_APPLY_ACCEPTED_PENDING_PUBLICATION &&
                !preparedCommandFailed) {
                if (facts == nullptr ||
                    facts->abiVersion != NM_MATTER_OWNER_APPLY_ABI_VERSION ||
                    facts->status !=
                        NM_PREPARED_STATE_PUBLICATION_FACTS_VALID ||
                    facts->reserved0 != 0u || facts->reserved1 != 0u ||
                    facts->physicsTokenFingerprint == 0u ||
                    facts->brainProgramFingerprint == 0u ||
                    facts->brainShadowStateFingerprint == 0u ||
                    facts->brainWitnessFingerprint == 0u ||
                    facts->matterApplyFingerprint == 0u ||
                    facts->factsFingerprint == 0u ||
                    facts->factsFingerprint !=
                        publicationFactsFingerprint(*facts)) {
                    restoreRequired = true;
                    acceptedStateProofEligible = false;
                    disposition =
                        PreparedStateDisposition::restoreRequired;
                    return;
                }
                disposition =
                    PreparedStateDisposition::acceptedPendingPublication;
                publicationReserved = false;
                publicationBinding = {};
                publicationReservation = {};
                publicationFacts = *facts;
                restoreRequired = false;
                applyCommandCompleted = false;
                return;
            }
            applyCommandCompleted = true;
            disposition = PreparedStateDisposition::resolved;
            preparedStateOpen = false;
            restoreRequired = false;
            preparedStateRequested = false;
            preparedCommandBuffer = nullptr;
            preparedEnvironmentStatuses = nullptr;
            preparedEnvironmentStatusesGPUAddress = 0u;
            acceptedStateProofEncoded = false;
            acceptedStateProofEligible = false;
            transactionPolicyFingerprint = 0u;
            identificationAdvanced = false;
            preparedCommandCompleted = false;
            preparedCommandFailed = false;
            terminalNoTouch = false;
            applyCommandCompleted = false;
            publicationReserved = false;
            publicationBinding = {};
            publicationReservation = {};
            publicationFacts = {};
        }
    };
    struct GrowthOwnership {
        std::mutex mutex;
        TopologyGrowthRequest pending;
        id<MTLBuffer> readback = nil;
    };

    bool captureEvents = true;
    bool captureDiagnostics = true;
    bool automaticIdentification = false;
    bool adaptiveTransfer = true;
    bool requiresCurrentBodies = false;
    bool requiresBodyWrenches = false;
    bool requiresCoupledCandidate = false;
    bool requiresSceneBodies = false;
    bool requiresRodNodes = false;
    bool hasAdaptive = false;
    std::vector<NMRigidProxyGPU> rigidProxyLayout;
    std::vector<std::uint32_t> sutureProxyIndices;
    std::vector<std::uint32_t> sutureProxyEdges;
    std::uint64_t sutureProxyBindingRevision = 0u;
    std::atomic<std::uint32_t> coupledTimestepMultiplier{1u};
    std::atomic<std::uint32_t> coupledTimestepDivisor{1u};
    // Zero canonically selects mixedSolverValue.nonlinearIterations.z.
    // A nonzero phase-boundary override changes encoded work only; the
    // restarted basis allocation and every convergence tolerance stay fixed.
    std::atomic<std::uint32_t> fgmresIterationBudgetOverride{0u};
    // Zero canonically selects mixedSolverValue.nonlinearIterations.x. This
    // completion-boundary override changes only the number of encoded outer
    // residual reassemblies/corrections; device arenas and tolerances stay fixed.
    std::atomic<std::uint32_t> newtonIterationBudgetOverride{0u};
    std::shared_ptr<CommandOwnership> commandOwnership =
        std::make_shared<CommandOwnership>();
    std::shared_ptr<GrowthOwnership> growthOwnership =
        std::make_shared<GrowthOwnership>();
    std::vector<NMContinuumObjectGPU> objectLayout;
    std::vector<NMFEMCapacityGPU> capacityLayout;

    id<MTLBuffer> dispatchBuffer = nil;
    id<MTLBuffer> mixedSolver = nil;
    id<MTLBuffer> materials = nil;
    id<MTLBuffer> mixedMaterials = nil;
    id<MTLBuffer> learnedMaterials = nil;
    id<MTLBuffer> learnedLayers = nil;
    id<MTLBuffer> parameterDefaults = nil;
    id<MTLBuffer> instructions = nil;
    id<MTLBuffer> scalarPrograms = nil;
    id<MTLBuffer> objects = nil;
    id<MTLBuffer> mpmGrids = nil;
    id<MTLBuffer> mpmBlocks = nil;
    id<MTLBuffer> mpmBlockLookup = nil;
    id<MTLBuffer> mpmStencils = nil;
    id<MTLBuffer> mpmNodeIncidence = nil;
    id<MTLBuffer> mpmNodeRanges = nil;
    id<MTLBuffer> femTetrahedra = nil;
    id<MTLBuffer> femTetrahedraAccepted = nil;
    id<MTLBuffer> femTetrahedraCandidate = nil;
    id<MTLBuffer> femTetrahedraCheckpoint = nil;
    id<MTLBuffer> continuumSurfacePrimitives = nil;
    id<MTLBuffer> femSurfaceSortKeysA = nil;
    id<MTLBuffer> femSurfaceSortKeysB = nil;
    id<MTLBuffer> femSurfaceSortIndicesA = nil;
    id<MTLBuffer> femSurfaceSortIndicesB = nil;
    id<MTLBuffer> femSurfaceActiveCounts = nil;
    id<MTLBuffer> deformableContactCandidates = nil;
    id<MTLBuffer> deformableContactCandidateCounts = nil;
    id<MTLBuffer> deformableContacts = nil;
    id<MTLBuffer> deformableContactHistoriesAccepted = nil;
    id<MTLBuffer> deformableContactHistoriesCandidate = nil;
    id<MTLBuffer> deformableContactHistoriesCheckpoint = nil;
    id<MTLBuffer> deformableContactActiveIndices = nil;
    id<MTLBuffer> deformableContactActiveCounts = nil;
    id<MTLBuffer> deformableContactNodeIncidence = nil;
    id<MTLBuffer> deformableContactNodeRanges = nil;
    id<MTLBuffer> deformableContactActiveOffsets = nil;
    id<MTLBuffer> deformableContactGlobalActiveIndices = nil;
    id<MTLBuffer> deformableContactActiveDispatch = nil;
    id<MTLBuffer> femTopologyNodesAccepted = nil;
    id<MTLBuffer> femTopologyNodesCandidate = nil;
    id<MTLBuffer> femTopologyNodesCheckpoint = nil;
    id<MTLBuffer> cohesiveFacesAccepted = nil;
    id<MTLBuffer> cohesiveFacesCandidate = nil;
    id<MTLBuffer> cohesiveFacesCheckpoint = nil;
    id<MTLBuffer> punctureChannelsAccepted = nil;
    id<MTLBuffer> punctureChannelsCandidate = nil;
    id<MTLBuffer> punctureChannelsCheckpoint = nil;
    id<MTLBuffer> topologyStatesAccepted = nil;
    id<MTLBuffer> topologyStatesCandidate = nil;
    id<MTLBuffer> topologyStatesCheckpoint = nil;
    id<MTLBuffer> femCapacities = nil;
    id<MTLBuffer> cookedMutationCommands = nil;
    id<MTLBuffer> triggeredMutationCommands = nil;
    id<MTLBuffer> femNodeIncidence = nil;
    id<MTLBuffer> femNodeRanges = nil;
    id<MTLBuffer> femNodeIncidenceCheckpoint = nil;
    id<MTLBuffer> femNodeRangesCheckpoint = nil;
    id<MTLBuffer> topologyIncidenceCursors = nil;
    id<MTLBuffer> fieldBoundaries = nil;
    id<MTLBuffer> rigidProxies = nil;
    id<MTLBuffer> contactPairs = nil;
    id<MTLBuffer> contactNodeIncidence = nil;
    id<MTLBuffer> contactNodeRanges = nil;
    id<MTLBuffer> rigidIncidence = nil;
    id<MTLBuffer> rigidRanges = nil;
    // Runtime-derived body-owned proxy incidence. It permits multiple contact
    // proxies per rigid body while retaining one deterministic wrench writer.
    id<MTLBuffer> bodyProxyIncidence = nil;
    id<MTLBuffer> bodyProxyRanges = nil;
    id<MTLBuffer> contactActivePairs = nil;
    id<MTLBuffer> contactActiveSlotsByPair = nil;
    id<MTLBuffer> contactActiveCounts = nil;
    std::uint32_t contactActiveCapacity = 0u;

    id<MTLBuffer> environmentParameters = nil;
    id<MTLBuffer> environmentParametersPreparedCheckpoint = nil;
    id<MTLBuffer> particleDefaults = nil;
    id<MTLBuffer> particleMaterialStateDefaults = nil;
    id<MTLBuffer> femDefaults = nil;
    id<MTLBuffer> femMaterialStateDefaults = nil;
    id<MTLBuffer> adaptiveDefaults = nil;
    id<MTLBuffer> schedulerDefaults = nil;
    id<MTLBuffer> particleAccepted = nil;
    id<MTLBuffer> particleCandidate = nil;
    id<MTLBuffer> particleCheckpoint = nil;
    id<MTLBuffer> particleMaterialStateAccepted = nil;
    id<MTLBuffer> particleMaterialStateCandidate = nil;
    id<MTLBuffer> particleMaterialStateCheckpoint = nil;
    id<MTLBuffer> gridNodes = nil;
    id<MTLBuffer> mpmNodeGenerations = nil;
    id<MTLBuffer> mpmActiveNodeIndices = nil;
    id<MTLBuffer> mpmNodeToActive = nil;
    id<MTLBuffer> mpmActiveNodeCounts = nil;
    id<MTLBuffer> mpmParticleKeys = nil;
    id<MTLBuffer> mpmBlockCounts = nil;
    id<MTLBuffer> mpmBlockOffsets = nil;
    id<MTLBuffer> mpmBlockCursors = nil;
    id<MTLBuffer> mpmBlockActiveFlags = nil;
    id<MTLBuffer> mpmBlockActiveLocalOffsets = nil;
    id<MTLBuffer> mpmEnvironmentActiveCounts = nil;
    id<MTLBuffer> mpmEnvironmentActiveOffsets = nil;
    id<MTLBuffer> mpmActiveBlocks = nil;
    id<MTLBuffer> mpmSortedParticleIndices = nil;
    id<MTLBuffer> mpmActiveDispatch = nil;
    id<MTLBuffer> femAccepted = nil;
    id<MTLBuffer> femCandidate = nil;
    id<MTLBuffer> femCheckpoint = nil;
    id<MTLBuffer> femMaterialStateAccepted = nil;
    id<MTLBuffer> femMaterialStateCandidate = nil;
    id<MTLBuffer> femMaterialStateCheckpoint = nil;
    id<MTLBuffer> femFieldsAccepted = nil;
    id<MTLBuffer> femFieldsCandidate = nil;
    id<MTLBuffer> femFieldsCheckpoint = nil;
    id<MTLBuffer> femFieldsWork = nil;
    id<MTLBuffer> mixedFieldSolution = nil;
    id<MTLBuffer> mixedFieldResidual = nil;
    id<MTLBuffer> mixedFieldPreconditioned = nil;
    id<MTLBuffer> mixedFieldDirection = nil;
    id<MTLBuffer> mixedFieldInverseDiagonal = nil;
    id<MTLBuffer> mixedFieldOperator = nil;
    id<MTLBuffer> solverCertificates = nil;
    id<MTLBuffer> learnedWeightDefaults = nil;
    id<MTLBuffer> learnedWeightsAccepted = nil;
    id<MTLBuffer> learnedWeightsCandidate = nil;
    id<MTLBuffer> learnedWeightsCheckpoint = nil;
    id<MTLBuffer> learnedRevisionAccepted = nil;
    id<MTLBuffer> learnedRevisionCandidate = nil;
    id<MTLBuffer> learnedRevisionCheckpoint = nil;
    std::uint64_t mutationFingerprint = 0u;
    std::uint64_t learnedFingerprint = 0u;
    id<MTLBuffer> rigidStates = nil;
    id<MTLBuffer> rigidStatesPreparedCheckpoint = nil;
    id<MTLBuffer> contactSamples = nil;
    id<MTLBuffer> contactHistoriesAccepted = nil;
    id<MTLBuffer> contactHistoriesCandidate = nil;
    id<MTLBuffer> contactHistoriesCheckpoint = nil;
    id<MTLBuffer> articulatedPointQueries = nil;
    id<MTLBuffer> coupledGeneralizedInput = nil;
    id<MTLBuffer> coupledGeneralizedOutput = nil;
    id<MTLBuffer> coupledGeneralizedCandidate = nil;
    id<MTLBuffer> coupledGeneralizedPreparedCheckpoint = nil;
    id<MTLBuffer> coupledPointJacobians = nil;
    id<MTLBuffer> coupledInverseStatuses = nil;
    id<MTLBuffer> coupledCandidateQ = nil;
    id<MTLBuffer> coupledCandidateBodies = nil;
    id<MTLBuffer> femHumanAttachments = nil;
    id<MTLBuffer> femHumanAttachmentPointQueries = nil;
    id<MTLBuffer> femHumanAttachmentPointJacobians = nil;
    id<MTLBuffer> femHumanAttachmentResidualImpulses = nil;
    id<MTLBuffer> femHumanAttachmentRawReactions = nil;
    id<MTLBuffer> femHumanAttachmentOperator = nil;
    std::uint32_t coupledQStride = 0u;
    std::uint32_t coupledVStride = 0u;
    std::uint32_t coupledBodyStride = 0u;
    id<MTLBuffer> microstepReactions = nil;
    id<MTLBuffer> frameReactions = nil;
    id<MTLBuffer> frameReactionsPreparedCheckpoint = nil;
    id<MTLBuffer> adaptive = nil;
    id<MTLBuffer> adaptivePreparedCheckpoint = nil;
    id<MTLBuffer> schedulers = nil;
    // Per-substep snapshot used only for event edge detection.
    id<MTLBuffer> schedulerPrevious = nil;
    id<MTLBuffer> schedulerPreviousPreparedCheckpoint = nil;
    // Immutable control-step-start snapshot used for transactional rollback.
    id<MTLBuffer> schedulerCheckpoint = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> events = nil;

    id<MTLBuffer> elementForces = nil;
    id<MTLBuffer> elementOperator = nil;
    id<MTLBuffer> femSolution = nil;
    id<MTLBuffer> femResidual = nil;
    id<MTLBuffer> femPreconditioned = nil;
    id<MTLBuffer> femDirection = nil;
    id<MTLBuffer> femOperatorValue = nil;
    id<MTLBuffer> femConstraintReactions = nil;
    id<MTLBuffer> femLineSearch = nil;
    id<MTLBuffer> fgmresBasis = nil;
    id<MTLBuffer> fgmresPreconditionedBasis = nil;
    id<MTLBuffer> fgmresHessenberg = nil;
    id<MTLBuffer> fgmresRotations = nil;
    id<MTLBuffer> fgmresLeastSquares = nil;
    id<MTLBuffer> fgmresRestartCoefficients = nil;
    id<MTLBuffer> fgmresStates = nil;
    id<MTLBuffer> primalContactArguments = nil;

    id<MTLBuffer> identificationDistributions = nil;
    id<MTLBuffer> identificationPreparedCheckpoint = nil;
    id<MTLBuffer> identificationCandidates = nil;
    id<MTLBuffer> identificationLosses = nil;
    id<MTLBuffer> dummy = nil;

    [[nodiscard]] id<MTLComputePipelineState> pipeline(
        const std::string_view name
    ) const {
        const auto iterator = pipelines.find(std::string(name));
        return iterator == pipelines.end() ? nil : iterator->second;
    }
};

Runtime::Runtime() = default;
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

RuntimeDiagnostics Runtime::initialize(
    const CompiledWorld& world,
    const RuntimeConfiguration& configuration
) {
    @autoreleasepool {
        RuntimeDiagnostics diagnostics;
        std::string layoutError;
        if (!validateCompiledWorldLayout(world, &layoutError)) {
            diagnostics.message =
                "compiled Matter world has an invalid fixed-capacity layout: " +
                layoutError;
            return diagnostics;
        }
        if (configuration.environmentCount != 0u &&
            configuration.environmentCount != world.dispatch.environmentCount) {
            diagnostics.message = "runtime environment count must match the cooked matter package";
            return diagnostics;
        }
        if (world.dispatch.femHumanAttachmentCount >
                NM_MATTER_MAX_HUMAN_ATTACHMENT_POINTS ||
            !detail::femHumanAttachmentPointJacobianStrideFits(
                world.dispatch.femHumanAttachmentCount,
                world.dispatch.rigidGeneralizedCapacity) ||
            world.dispatch.femHumanAttachmentPointJacobianStride !=
                detail::femHumanAttachmentPointJacobianStride(
                    world.dispatch.femHumanAttachmentCount,
                    world.dispatch.rigidGeneralizedCapacity) ||
            (world.dispatch.femHumanAttachmentCount != 0u &&
             (world.dispatch.rigidGeneralizedCapacity == 0u ||
              world.dispatch.rigidQCapacity == 0u))) {
            diagnostics.message =
                "FEM Human attachments exceed the exact coupled-candidate point/Jacobian capacity";
            return diagnostics;
        }
        auto candidate = std::make_unique<State>();
        candidate->dispatch = world.dispatch;
        candidate->objectLayout = world.objects;
        candidate->capacityLayout = world.fem.capacities;
        if (world.fem.nodeIncidence.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            world.fem.nodeRanges.size() >
                std::numeric_limits<std::uint32_t>::max()) {
            diagnostics.message =
                "Matter topology incidence exceeds accepted-state proof ABI capacity";
            return diagnostics;
        }
        candidate->femNodeIncidenceStride = static_cast<std::uint32_t>(
            world.fem.nodeIncidence.size());
        candidate->femNodeRangeStride = static_cast<std::uint32_t>(
            world.fem.nodeRanges.size());
        candidate->mixedSolverValue = world.mixedSolver;
        candidate->mutationFingerprint = detail::hashBytes(
            world.fem.mutationCommands.data(),
            world.fem.mutationCommands.size() * sizeof(NMMutationCommandGPU)
        );
        candidate->learnedFingerprint = detail::hashBytes(
            world.learnedWeights.data(),
            world.learnedWeights.size() * sizeof(float)
        );
        candidate->sourcePhysicsFingerprint = world.physicsFingerprint;
        candidate->worldFingerprint = world.fingerprint;
        candidate->captureEvents = configuration.captureEvents;
        candidate->captureDiagnostics = configuration.captureDiagnostics;
        candidate->automaticIdentification =
            configuration.automaticIdentification;
        candidate->adaptiveTransfer = configuration.adaptiveTransfer;
        candidate->identificationDistributionCount =
            static_cast<std::uint32_t>(world.identification.size());
        candidate->hasAdaptive =
            (world.dispatch.flags & NM_MATTER_ADAPTIVE) != 0u;
        candidate->device = MTLCreateSystemDefaultDevice();
        if (candidate->device == nil) {
            diagnostics.message = "no Metal device is available";
            return diagnostics;
        }
        candidate->queue = [candidate->device newCommandQueue];
        if (candidate->queue == nil) {
            diagnostics.message = "failed to create Numi Matter command queue";
            return diagnostics;
        }
        std::filesystem::path metallib = configuration.metallib;
        if (metallib.empty()) {
            metallib = defaultMetallib();
        }
        if (!regularFile(metallib)) {
            diagnostics.message = "NumiMatter.metallib is unavailable";
            return diagnostics;
        }
        NSString* metallibPath = [NSString
            stringWithUTF8String:metallib.string().c_str()];
        NSError* imageError = nil;
        NSData* metallibImage = metallibPath == nil ? nil :
            [NSData dataWithContentsOfFile:metallibPath
                                  options:NSDataReadingMappedIfSafe
                                    error:&imageError];
        if (metallibImage == nil || metallibImage.length == 0u ||
            metallibImage.bytes == nullptr) {
            diagnostics.message =
                "failed to read one immutable NumiMatter.metallib image: " +
                errorString(imageError);
            return diagnostics;
        }
        const std::uint64_t metallibFingerprint = detail::hashBytes(
            metallibImage.bytes,
            static_cast<std::size_t>(metallibImage.length));
        const std::uint64_t metallibByteCount =
            static_cast<std::uint64_t>(metallibImage.length);
        if (metallibFingerprint == 0u || metallibByteCount == 0u) {
            diagnostics.message =
                "NumiMatter.metallib executable image identity is invalid";
            return diagnostics;
        }
        candidate->executionFingerprint = makeDeviceProgramFingerprint(
            world.fingerprint,
            metallibFingerprint,
            metallibByteCount,
            configuration
        );
        candidate->acceptedStateProofMujocoBytesPerEnvironmentCapacity =
            configuration
                .acceptedStateProofMujocoBytesPerEnvironmentCapacity;
        if (candidate->acceptedStateProofMujocoBytesPerEnvironmentCapacity >
                std::numeric_limits<NSUInteger>::max()) {
            diagnostics.message =
                "accepted-state MyoSim proof capacity exceeds Metal address space";
            return diagnostics;
        }
        candidate->acceptedStateProofProgramFingerprint =
            candidate->acceptedStateProofMujocoBytesPerEnvironmentCapacity ==
                    0u
            ? 0u
            : makeAcceptedStateProofProgramFingerprint(
                candidate->executionFingerprint,
                candidate->dispatch,
                candidate->identificationDistributionCount,
                candidate->femNodeIncidenceStride,
                candidate->femNodeRangeStride,
                candidate
                    ->acceptedStateProofMujocoBytesPerEnvironmentCapacity);
        dispatch_data_t libraryImage = dispatch_data_create(
            metallibImage.bytes,
            metallibImage.length,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (libraryImage == nullptr) {
            diagnostics.message =
                "failed to retain the exact NumiMatter.metallib image";
            return diagnostics;
        }
        NSError* libraryError = nil;
        candidate->library = [candidate->device
            newLibraryWithData:libraryImage error:&libraryError];
        if (candidate->library == nil) {
            diagnostics.message = "failed to load Numi Matter Metal library: " +
                errorString(libraryError);
            return diagnostics;
        }

        const auto proofPipeline = [&](NSString* name) {
            NSString* qualified = [@"numi_matter_metal::"
                stringByAppendingString:name];
            id<MTLFunction> function =
                [candidate->library newFunctionWithName:qualified];
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline = function == nil
                ? nil
                : [candidate->device
                    newComputePipelineStateWithFunction:function
                                                   error:&pipelineError];
            if (pipeline == nil && diagnostics.message.empty()) {
                diagnostics.message =
                    "failed to create Matter accepted-state proof pipeline " +
                    nsString(name) + ": " + errorString(pipelineError);
            }
            return pipeline;
        };
        candidate->acceptedStateProofBegin = proofPipeline(
            @"nm_accepted_state_proof_begin");
        candidate->acceptedStateProofChunks = proofPipeline(
            @"nm_accepted_state_proof_chunks");
        candidate->acceptedStateProofReduce = proofPipeline(
            @"nm_accepted_state_proof_reduce");
        candidate->acceptedStateProofFold = proofPipeline(
            @"nm_accepted_state_proof_fold");
        candidate->acceptedStateProofFinalize = proofPipeline(
            @"nm_accepted_state_proof_finalize");
        candidate->preparedStateValidateApplication = proofPipeline(
            @"nm_prepared_state_validate_application");
        candidate->preparedStateNormalizeApplication = proofPipeline(
            @"nm_prepared_state_normalize_application");
        candidate->preparedStateMaterializeRestoreStatuses = proofPipeline(
            @"nm_prepared_state_materialize_restore_statuses");
        candidate->preparedStateRestoreBytes = proofPipeline(
            @"nm_prepared_state_restore_bytes");
        if (candidate->acceptedStateProofBegin == nil ||
            candidate->acceptedStateProofChunks == nil ||
            candidate->acceptedStateProofReduce == nil ||
            candidate->acceptedStateProofFold == nil ||
            candidate->acceptedStateProofFinalize == nil ||
            candidate->preparedStateValidateApplication == nil ||
            candidate->preparedStateNormalizeApplication == nil ||
            candidate->preparedStateMaterializeRestoreStatuses == nil ||
            candidate->preparedStateRestoreBytes == nil) {
            if (diagnostics.message.empty()) {
                diagnostics.message =
                    "Matter accepted-state proof authority is incomplete";
            }
            return diagnostics;
        }

        const char* kernelNames[]{
            "nm_prepare_status",
            "nm_prepare_events",
            "nm_prepare_reactions",
            "nm_checkpoint_scheduler",
            "nm_prepare_scheduler",
            "nm_apply_episode_resets",
            "nm_identification_update",
            "nm_publish_identification_means",
            "nm_identification_sample",
            "nm_mpm_checkpoint",
            "nm_mpm_prepare_sparse",
            "nm_mpm_classify_particles",
            "nm_mpm_scan_environment_blocks",
            "nm_mpm_scan_environment_counts",
            "nm_mpm_scatter_active_blocks",
            "nm_mpm_scatter_particles",
            "nm_mpm_sort_block_particles",
            "nm_fem_checkpoint",
            "nm_fem_prepare_candidate",
            "nm_fem_apply_kinematic_targets",
            "nm_fem_human_attachment_drive_candidate",
            "nm_topology_checkpoint",
            "nm_topology_detect_puncture",
            "nm_topology_execute_transaction",
            "nm_topology_rebuild_clear",
            "nm_topology_rebuild_count_mass",
            "nm_topology_rebuild_prefix",
            "nm_topology_rebuild_scatter",
            "nm_topology_rebuild_finalize",
            "nm_topology_conserve_transaction",
            "nm_topology_certify_transaction",
            "nm_topology_validate_growth_references",
            "nm_topology_mark_growth_generation",
            "nm_topology_publish_growth_request",
            "nm_topology_commit",
            "nm_topology_rollback",
            "nm_mixed_checkpoint",
            "nm_mixed_prepare_microstep",
            "nm_mixed_prepare_residual",
            "nm_mixed_apply_operator_elements",
            "nm_mixed_apply_operator_nodes",
            "nm_mixed_build_residual_diagonal",
            "nm_mixed_apply_kkt_solution",
            "nm_mixed_certify",
            "nm_mixed_commit",
            "nm_mixed_rollback",
            "nm_learned_stage_weights",
            "nm_learned_checkpoint",
            "nm_learned_commit",
            "nm_learned_rollback",
            "nm_project_rigid_states",
            "nm_project_primal_free_rigid_candidate",
            "nm_mask_primal_rigid_candidate",
            "nm_publish_primal_free_rigid_candidate",
            "nm_publish_adaptive_rigid_ownership",
            "nm_mpm_p2g",
            "nm_mpm_compact_active_nodes",
            "nm_fem_internal_forces",
            "nm_fem_build_mechanical_residual",
            "nm_fem_apply_solution",
            "nm_fem_select_backtracking",
            "nm_fem_synchronize_environment_line_search",
            "nm_fgmres_begin",
            "nm_fem_human_attachment_map_direction",
            "nm_fem_human_attachment_eliminate_direction",
            "nm_fem_human_attachment_capture_residual",
            "nm_fem_human_attachment_scatter_residual",
            "nm_fem_human_attachment_capture_operator",
            "nm_fem_human_attachment_scatter_operator",
            "nm_fem_human_attachment_mask_reactions",
            "nm_fgmres_measure_correction",
            "nm_fgmres_build_preconditioner",
            "nm_fgmres_precondition",
            "nm_fgmres_precondition_patches",
            "nm_fgmres_precondition_coarse",
            "nm_fgmres_import_field_residual",
            "nm_fgmres_clear_rigid_residual",
            "nm_rigid_initialize_feasible_candidate",
            "nm_rigid_apply_candidate_solution",
            "nm_fgmres_field_smoother_initialize",
            "nm_fgmres_field_smoother_export",
            "nm_fgmres_field_smooth",
            "nm_fgmres_precondition_cross",
            "nm_fgmres_export_rigid",
            "nm_fgmres_import_rigid",
            "nm_fgmres_precondition_free_rigid",
            "nm_fgmres_apply_free_rigid",
            "nm_fgmres_apply_primal_rigid_contacts",
            "nm_contact_accumulate_rigid_residual",
            "nm_contact_subtract_rigid_inertia_residual",
            "nm_mpm_build_implicit_residual",
            "nm_mpm_build_constitutive_residual",
            "nm_fgmres_precondition_mpm",
            "nm_fgmres_precondition_mpm_patches",
            "nm_fgmres_precondition_mpm_objects",
            "nm_fgmres_apply_mpm",
            "nm_fgmres_apply_mpm_constitutive",
            "nm_fgmres_accumulate_mpm",
            "nm_fgmres_restart_residual_mpm",
            "nm_mpm_limit_implicit_line_search",
            "nm_mpm_apply_implicit_solution",
            "nm_fem_apply_operator_elements",
            "nm_fgmres_gather_nodes",
            "nm_fgmres_orthogonalize_and_finish_column",
            "nm_fgmres_finalize_cycle",
            "nm_fgmres_accumulate",
            "nm_fgmres_accumulate_fields",
            "nm_fgmres_accumulate_rigid",
            "nm_fgmres_restart_residual_nodes",
            "nm_fgmres_restart_residual_rigid",
            "nm_contact_clear_samples",
            "nm_contact_build_surface_primitives",
            "nm_contact_sort_surface_primitives",
            "nm_contact_build_deformable_candidates",
            "nm_contact_narrowphase_deformable",
            "nm_contact_compact_deformable",
            "nm_contact_scan_deformable_active_counts",
            "nm_contact_scatter_deformable_active_work",
            "nm_contact_count_deformable_node_incidence",
            "nm_contact_scan_deformable_node_incidence",
            "nm_contact_scatter_deformable_node_incidence",
            "nm_contact_evaluate",
            "nm_contact_certify_post_commit_rigid",
            "nm_contact_compact_active",
            "nm_contact_checkpoint_histories",
            "nm_contact_commit_histories",
            "nm_contact_rollback_histories",
            "nm_contact_clear_remapped_suture_histories",
            "nm_contact_checkpoint_deformable_contact_histories",
            "nm_contact_commit_deformable_contact_histories",
            "nm_contact_rollback_deformable_contact_histories",
            "nm_contact_prepare_articulated_queries",
            "nm_contact_accumulate_fem_residual",
            "nm_contact_accumulate_deformable_fem_residual",
            "nm_contact_publish_primal_history",
            "nm_contact_limit_deformable_line_search",
            "nm_contact_limit_rigid_line_search",
            "nm_contact_publish_deformable_history",
            "nm_contact_reduce_rigid",
            "nm_accumulate_rigid_reactions",
            "nm_apply_suture_strand_reactions",
            "nm_mpm_g2p",
            "nm_fem_integrate",
            "nm_fem_validate",
            "nm_mpm_commit_microstep",
            "nm_fem_commit_microstep",
            "nm_scheduler_observe",
            "nm_complete_microstep",
            "nm_mpm_rollback_frame",
            "nm_fem_rollback_frame",
            "nm_latch_matter_status_into_rigid_world",
            "nm_reconcile_rigid_world_status",
            "nm_scheduler_reconcile",
            "nm_adaptive_measure",
            "nm_adaptive_observe_rigid_contacts",
            "nm_adaptive_decide",
            "nm_adaptive_demote_to_rigid",
        };
        const std::array<const char*, 5> finalKernelNames{
            "nm_adaptive_promote_mpm",
            "nm_adaptive_promote_fem",
            "nm_adaptive_finish_promotion",
            "nm_scheduler_finalize",
            "nm_finalize_status",
        };
        const auto kernel = [&](const char* name) {
            const std::string qualified =
                std::string("numi_matter_metal::") + name;
            return [candidate->library newFunctionWithName:
                [NSString stringWithUTF8String:qualified.c_str()]];
        };
        for (const char* name : kernelNames) {
            id<MTLFunction> function = kernel(name);
            if (function == nil) {
                diagnostics.message = std::string("missing Metal function ") + name;
                return diagnostics;
            }
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline = [candidate->device
                newComputePipelineStateWithFunction:function
                                               error:&pipelineError];
            if (pipeline == nil) {
                diagnostics.message = std::string("failed to compile pipeline ") + name +
                    ": " + errorString(pipelineError);
                return diagnostics;
            }
            candidate->pipelines.emplace(name, pipeline);
        }
        for (const char* name : finalKernelNames) {
            id<MTLFunction> function = kernel(name);
            NSError* pipelineError = nil;
            id<MTLComputePipelineState> pipeline = function == nil
                ? nil
                : [candidate->device newComputePipelineStateWithFunction:function
                                                                   error:&pipelineError];
            if (pipeline == nil) {
                diagnostics.message = std::string("failed to compile pipeline ") + name +
                    ": " + errorString(pipelineError);
                return diagnostics;
            }
            candidate->pipelines.emplace(name, pipeline);
        }
        for (const char* name : {
                "nm_mpm_scan_environment_blocks",
                "nm_mpm_scan_environment_counts",
                "nm_scheduler_observe",
                "nm_adaptive_measure",
            }) {
            if (candidate->pipeline(name).threadExecutionWidth != 32u) {
                diagnostics.message = std::string(name) +
                    " requires the Apple SIMD32 execution contract";
                return diagnostics;
            }
        }

        id<MTLFunction> primalContactFunction = [candidate->library
            newFunctionWithName:
                @"numi_matter_metal::nm_primal_contact_argument_layout"];
        id<MTLArgumentEncoder> primalContactEncoder =
            primalContactFunction == nil
            ? nil
            : [primalContactFunction newArgumentEncoderWithBufferIndex:0u];
        if (primalContactEncoder == nil) {
            diagnostics.message =
                "failed to create monolithic primal-contact argument encoder";
            return diagnostics;
        }

        bool valid = true;
        candidate->growthOwnership->readback =
            sharedScratch<NMTopologyGrowthRequestGPU>(
                candidate->device, 1u, valid, candidate->residentBytes);
        const std::size_t contactPairCount = world.contact.pairs.size();
        candidate->contactActiveCapacity = std::min<std::uint32_t>(
            candidate->dispatch.reservedMixed1,
            candidate->dispatch.contactPairCount
        );
        if (contactPairCount != 0u && candidate->contactActiveCapacity == 0u) {
            diagnostics.message =
                "Matter active contact capacity is zero for a contact world";
            return diagnostics;
        }
        UploadPlan uploads(candidate->device, candidate->queue);
        const std::size_t environments = world.dispatch.environmentCount;
        candidate->dispatchBuffer = uploads.one(
            std::span<const NMMatterDispatchGPU>(&candidate->dispatch, 1u),
            valid, candidate->residentBytes);
        candidate->mixedSolver = uploads.one(
            std::span<const NMMixedSolverGPU>(&world.mixedSolver, 1u),
            valid, candidate->residentBytes);
        candidate->materials = uploads.one(
            std::span<const NMMaterialGPU>(world.materials),
            valid, candidate->residentBytes);
        candidate->mixedMaterials = uploads.one(
            std::span<const NMMixedMaterialGPU>(world.mixedMaterials),
            valid, candidate->residentBytes);
        candidate->learnedMaterials = uploads.one(
            std::span<const NMLearnedMaterialGPU>(world.learnedMaterials),
            valid, candidate->residentBytes);
        candidate->learnedLayers = uploads.one(
            std::span<const NMLearnedLayerGPU>(world.learnedLayers),
            valid, candidate->residentBytes);
        candidate->parameterDefaults = uploads.one(
            std::span<const NMParameterRangeGPU>(world.parameters),
            valid, candidate->residentBytes);
        candidate->instructions = uploads.one(
            std::span<const NMExpressionInstructionGPU>(world.instructions),
            valid, candidate->residentBytes);
        candidate->scalarPrograms = uploads.one(
            std::span<const NMScalarProgramGPU>(world.scalarPrograms),
            valid, candidate->residentBytes);
        candidate->objects = uploads.one(
            std::span<const NMContinuumObjectGPU>(world.objects),
            valid, candidate->residentBytes);
        candidate->mpmGrids = uploads.one(
            std::span<const NMMPMGridGPU>(world.mpm.grids),
            valid, candidate->residentBytes);
        candidate->mpmBlocks = uploads.one(
            std::span<const NMMPMBlockGPU>(world.mpm.blocks),
            valid, candidate->residentBytes);
        candidate->mpmBlockLookup = uploads.one(
            std::span<const std::uint32_t>(world.mpm.blockLookup),
            valid, candidate->residentBytes);
        candidate->mpmStencils = uploads.one(
            std::span<const NMMPMStencilGPU>(world.mpm.stencils),
            valid, candidate->residentBytes);
        candidate->mpmNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(world.mpm.nodeIncidence),
            valid, candidate->residentBytes);
        candidate->mpmNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.mpm.nodeRanges),
            valid, candidate->residentBytes);
        candidate->femTetrahedra = uploads.one(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            valid, candidate->residentBytes);
        candidate->femCapacities = uploads.one(
            std::span<const NMFEMCapacityGPU>(world.fem.capacities),
            valid, candidate->residentBytes);
        candidate->femHumanAttachments = uploads.one(
            std::span<const NMFEMHumanAttachmentGPU>(
                world.fem.humanAttachments),
            valid, candidate->residentBytes);
        std::vector<MRArticulatedPointImpulseGPU>
            femHumanAttachmentQueries;
        femHumanAttachmentQueries.reserve(
            world.fem.humanAttachments.size());
        for (const NMFEMHumanAttachmentGPU& attachment :
             world.fem.humanAttachments) {
            MRArticulatedPointImpulseGPU query{};
            query.bodyIndex = attachment.identity.y;
            query.localPoint = {
                attachment.localPoint.x,
                attachment.localPoint.y,
                attachment.localPoint.z,
                0.0f,
            };
            query.worldImpulse = {0.0f, 0.0f, 0.0f, 0.0f};
            femHumanAttachmentQueries.push_back(query);
        }
        candidate->femHumanAttachmentPointQueries = uploads.repeated(
            std::span<const MRArticulatedPointImpulseGPU>(
                femHumanAttachmentQueries),
            environments, valid, candidate->residentBytes);
        candidate->cookedMutationCommands = uploads.one(
            std::span<const NMMutationCommandGPU>(world.fem.mutationCommands),
            valid, candidate->residentBytes);
        std::vector<NMFEMTopologyStateGPU> initialTopologyStates(
            world.objects.size()
        );
        for (std::size_t objectIndex = 0u;
             objectIndex < world.objects.size(); ++objectIndex) {
            const NMContinuumObjectGPU& object = world.objects[objectIndex];
            if (object.representation != NM_REPRESENTATION_FEM) continue;
            NMFEMTopologyStateGPU topology{};
            double conservedMass = 0.0;
            for (std::uint32_t local = 0u; local < object.stateCount; ++local) {
                const std::size_t node = object.stateOffset + local;
                if ((world.fem.topologyNodes[node].identity.w &
                     NM_TOPOLOGY_ACTIVE) != 0u) {
                    ++topology.counts.x;
                    conservedMass += world.fem.nodes[node].positionAndMass.w;
                }
            }
            topology.accounting.x = static_cast<float>(conservedMass);
            for (std::uint32_t local = 0u; local < object.elementCount; ++local)
                if ((world.fem.tetrahedra[object.elementOffset + local]
                     .identity.w & NM_OBJECT_ACTIVE) != 0u)
                    ++topology.counts.y;
            for (const NMCohesiveFaceGPU& face : world.fem.cohesiveFaces)
                if (face.adjacency.y == objectIndex &&
                    (face.adjacency.w & NM_TOPOLOGY_ACTIVE) != 0u)
                    ++topology.counts.z;
            for (const NMPunctureChannelGPU& channel : world.fem.punctureChannels)
                if (channel.identity.x == objectIndex &&
                    (channel.identity.w & NM_TOPOLOGY_ACTIVE) != 0u)
                    ++topology.counts.w;
            topology.roles = {0u, 1u, 2u, object.topologyGeneration};
            initialTopologyStates[objectIndex] = topology;
        }
        candidate->topologyStatesAccepted = uploads.repeated(
            std::span<const NMFEMTopologyStateGPU>(initialTopologyStates),
            environments, valid, candidate->residentBytes);
        candidate->topologyStatesCandidate = uploads.repeated(
            std::span<const NMFEMTopologyStateGPU>(initialTopologyStates),
            environments, valid, candidate->residentBytes);
        candidate->topologyStatesCheckpoint = uploads.repeated(
            std::span<const NMFEMTopologyStateGPU>(initialTopologyStates),
            environments, valid, candidate->residentBytes);
        std::vector<std::uint32_t> environmentFEMIncidence;
        std::vector<NMIncidenceRangeGPU> environmentFEMRanges;
        environmentFEMIncidence.reserve(
            world.fem.nodeIncidence.size() * environments
        );
        environmentFEMRanges.reserve(
            world.fem.nodeRanges.size() * environments
        );
        for (std::size_t environment = 0u; environment < environments;
             ++environment) {
            environmentFEMIncidence.insert(
                environmentFEMIncidence.end(),
                world.fem.nodeIncidence.begin(),
                world.fem.nodeIncidence.end()
            );
            const std::uint32_t incidenceBase = static_cast<std::uint32_t>(
                environment * world.fem.nodeIncidence.size()
            );
            for (NMIncidenceRangeGPU range : world.fem.nodeRanges) {
                range.first += incidenceBase;
                environmentFEMRanges.push_back(range);
            }
        }
        candidate->femNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(environmentFEMIncidence),
            valid, candidate->residentBytes);
        candidate->femNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(environmentFEMRanges),
            valid, candidate->residentBytes);
        candidate->femNodeIncidenceCheckpoint = privateScratch<std::uint32_t>(
            candidate->device, environmentFEMIncidence.size(),
            valid, candidate->residentBytes);
        candidate->femNodeRangesCheckpoint = privateScratch<NMIncidenceRangeGPU>(
            candidate->device, environmentFEMRanges.size(),
            valid, candidate->residentBytes);
        candidate->topologyIncidenceCursors = privateScratch<std::uint32_t>(
            candidate->device,
            static_cast<std::size_t>(environments) * world.dispatch.femNodeCount,
            valid, candidate->residentBytes);
        candidate->fieldBoundaries = uploads.one(
            std::span<const NMFieldBoundaryGPU>(world.fem.fieldBoundaries),
            valid, candidate->residentBytes);
        candidate->rigidProxies = uploads.one(
            std::span<const NMRigidProxyGPU>(world.contact.rigidProxies),
            valid, candidate->residentBytes);
        candidate->rigidProxyLayout = world.contact.rigidProxies;
        for (std::uint32_t proxyIndex = 0u;
             proxyIndex < world.contact.rigidProxies.size();
             ++proxyIndex) {
            const NMRigidProxyGPU& proxy =
                world.contact.rigidProxies[proxyIndex];
            if ((proxy.flags & NM_RIGID_SUTURE_STRAND) == 0u) continue;
            candidate->sutureProxyIndices.push_back(proxyIndex);
            candidate->sutureProxyEdges.push_back(proxy.bodyIndex);
        }
        candidate->contactPairs = uploads.one(
            std::span<const NMContactPairGPU>(world.contact.pairs),
            valid, candidate->residentBytes);
        candidate->contactNodeIncidence = uploads.one(
            std::span<const std::uint32_t>(world.contact.nodeIncidence),
            valid, candidate->residentBytes);
        candidate->contactNodeRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.contact.nodeRanges),
            valid, candidate->residentBytes);
        candidate->rigidIncidence = uploads.one(
            std::span<const std::uint32_t>(world.contact.rigidIncidence),
            valid, candidate->residentBytes);
        candidate->rigidRanges = uploads.one(
            std::span<const NMIncidenceRangeGPU>(world.contact.rigidRanges),
            valid, candidate->residentBytes);
        std::vector<float> defaultParameters;
        defaultParameters.reserve(world.parameters.size());
        for (const NMParameterRangeGPU parameter : world.parameters) {
            defaultParameters.push_back(parameter.valueAndBounds.x);
        }
        const std::size_t materialStateStride =
            world.dispatch.materialStateStride;
        std::vector<float> particleMaterialStateDefaults(
            world.mpm.particles.size() * materialStateStride,
            0.0f
        );
        for (std::size_t particleIndex = 0u;
             particleIndex < world.mpm.particles.size();
             ++particleIndex) {
            const std::uint32_t materialIndex =
                world.mpm.particles[particleIndex].identity.y;
            if (materialIndex >= world.materials.size()) {
                diagnostics.message =
                    "MPM particle references an invalid stateful material";
                return diagnostics;
            }
            const NMMaterialGPU& material = world.materials[materialIndex];
            for (std::uint32_t stateIndex = 0u;
                 stateIndex < material.stateCount;
                 ++stateIndex) {
                particleMaterialStateDefaults[
                    particleIndex * materialStateStride + stateIndex
                ] = world.stateInitials[
                    material.stateInitialOffset + stateIndex
                ];
            }
        }
        std::vector<float> femMaterialStateDefaults(
            world.fem.tetrahedra.size() * materialStateStride,
            0.0f
        );
        for (std::size_t tetrahedronIndex = 0u;
             tetrahedronIndex < world.fem.tetrahedra.size();
             ++tetrahedronIndex) {
            const std::uint32_t materialIndex =
                world.fem.tetrahedra[tetrahedronIndex].identity.x;
            if (materialIndex >= world.materials.size()) {
                diagnostics.message =
                    "FEM element references an invalid stateful material";
                return diagnostics;
            }
            const NMMaterialGPU& material = world.materials[materialIndex];
            for (std::uint32_t stateIndex = 0u;
                 stateIndex < material.stateCount;
                 ++stateIndex) {
                femMaterialStateDefaults[
                    tetrahedronIndex * materialStateStride + stateIndex
                ] = world.stateInitials[
                    material.stateInitialOffset + stateIndex
                ];
            }
        }
        candidate->environmentParameters = uploads.repeated(
            std::span<const float>(defaultParameters),
            environments, valid, candidate->residentBytes);
        candidate->environmentParametersPreparedCheckpoint = uploads.repeated(
            std::span<const float>(defaultParameters),
            environments, valid, candidate->residentBytes);
        candidate->particleDefaults = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateDefaults = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femDefaults = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateDefaults = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->adaptiveDefaults = uploads.repeated(
            std::span<const NMAdaptiveStateGPU>(world.adaptive),
            environments, valid, candidate->residentBytes);
        candidate->schedulerDefaults = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->particleAccepted = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleCandidate = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleCheckpoint = uploads.repeated(
            std::span<const NMParticleStateGPU>(world.mpm.particles),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateAccepted = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateCandidate = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->particleMaterialStateCheckpoint = uploads.repeated(
            std::span<const float>(particleMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->gridNodes = uploads.repeated(
            std::span<const NMGridNodeStateGPU>(world.mpm.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femAccepted = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femCandidate = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femCheckpoint = uploads.repeated(
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes),
            environments, valid, candidate->residentBytes);
        candidate->femTetrahedraAccepted = uploads.repeated(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            environments, valid, candidate->residentBytes);
        candidate->femTetrahedraCandidate = uploads.repeated(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            environments, valid, candidate->residentBytes);
        candidate->femTetrahedraCheckpoint = uploads.repeated(
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra),
            environments, valid, candidate->residentBytes);
        candidate->femTopologyNodesAccepted = uploads.repeated(
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes),
            environments, valid, candidate->residentBytes);
        candidate->femTopologyNodesCandidate = uploads.repeated(
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes),
            environments, valid, candidate->residentBytes);
        candidate->femTopologyNodesCheckpoint = uploads.repeated(
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes),
            environments, valid, candidate->residentBytes);
        candidate->cohesiveFacesAccepted = uploads.repeated(
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces),
            environments, valid, candidate->residentBytes);
        candidate->cohesiveFacesCandidate = uploads.repeated(
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces),
            environments, valid, candidate->residentBytes);
        candidate->cohesiveFacesCheckpoint = uploads.repeated(
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces),
            environments, valid, candidate->residentBytes);
        candidate->punctureChannelsAccepted = uploads.repeated(
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels),
            environments, valid, candidate->residentBytes);
        candidate->punctureChannelsCandidate = uploads.repeated(
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels),
            environments, valid, candidate->residentBytes);
        candidate->punctureChannelsCheckpoint = uploads.repeated(
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateAccepted = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateCandidate = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femMaterialStateCheckpoint = uploads.repeated(
            std::span<const float>(femMaterialStateDefaults),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsAccepted = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsCandidate = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsCheckpoint = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        candidate->femFieldsWork = uploads.repeated(
            std::span<const NMFEMFieldStateGPU>(world.fem.fields),
            environments, valid, candidate->residentBytes);
        const std::size_t mixedNodeTotal =
            environments * world.dispatch.femNodeCount;
        candidate->mixedFieldSolution = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldResidual = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldPreconditioned = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldDirection = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldInverseDiagonal = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->mixedFieldOperator = privateScratch<nm_float4>(
            candidate->device, mixedNodeTotal,
            valid, candidate->residentBytes);
        candidate->learnedWeightDefaults = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        candidate->learnedWeightsAccepted = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        candidate->learnedWeightsCandidate = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        candidate->learnedWeightsCheckpoint = uploads.one(
            std::span<const float>(world.learnedWeights),
            valid, candidate->residentBytes);
        const std::uint32_t initialLearnedRevision = 0u;
        candidate->learnedRevisionAccepted = uploads.one(
            std::span<const std::uint32_t>(&initialLearnedRevision, 1u),
            valid, candidate->residentBytes);
        candidate->learnedRevisionCandidate = uploads.one(
            std::span<const std::uint32_t>(&initialLearnedRevision, 1u),
            valid, candidate->residentBytes);
        candidate->learnedRevisionCheckpoint = uploads.one(
            std::span<const std::uint32_t>(&initialLearnedRevision, 1u),
            valid, candidate->residentBytes);
        candidate->adaptive = uploads.repeated(
            std::span<const NMAdaptiveStateGPU>(world.adaptive),
            environments, valid, candidate->residentBytes);
        candidate->adaptivePreparedCheckpoint = uploads.repeated(
            std::span<const NMAdaptiveStateGPU>(world.adaptive),
            environments, valid, candidate->residentBytes);
        candidate->schedulers = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->schedulerPrevious = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->schedulerPreviousPreparedCheckpoint = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->schedulerCheckpoint = uploads.repeated(
            std::span<const NMSchedulerStateGPU>(world.schedulers),
            environments, valid, candidate->residentBytes);
        candidate->identificationDistributions = uploads.one(
            std::span<const NMIdentificationDistributionGPU>(world.identification),
            valid, candidate->residentBytes);
        candidate->identificationPreparedCheckpoint = uploads.one(
            std::span<const NMIdentificationDistributionGPU>(
                world.identification),
            valid, candidate->residentBytes);
        const std::vector<NMRigidStateGPU> initialRigidStates(
            world.dispatch.rigidProxyCount
        );
        candidate->rigidStates = uploads.repeated(
            std::span<const NMRigidStateGPU>(initialRigidStates),
            environments, valid, candidate->residentBytes);
        candidate->rigidStatesPreparedCheckpoint = uploads.repeated(
            std::span<const NMRigidStateGPU>(initialRigidStates),
            environments, valid, candidate->residentBytes);
        const std::vector<nm_float4> initialContactHistories(
            world.dispatch.contactPairCount
        );
        candidate->contactHistoriesAccepted = uploads.repeated(
            std::span<const nm_float4>(initialContactHistories),
            environments, valid, candidate->residentBytes);
        const std::vector<NMDeformableContactHistoryGPU>
            initialDeformableContactHistories(
                world.dispatch.deformableContactCapacity);
        candidate->deformableContactHistoriesAccepted = uploads.repeated(
            std::span<const NMDeformableContactHistoryGPU>(
                initialDeformableContactHistories),
            environments, valid, candidate->residentBytes);
        std::string uploadError;
        if (!valid || !uploads.finish(uploadError)) {
            diagnostics.message = valid
                ? "failed to upload immutable matter state: " + uploadError
                : "matter buffer size or allocation overflow";
            return diagnostics;
        }

        const auto multiplied = [&](const std::size_t perEnvironment) {
            return environments * perEnvironment;
        };
        candidate->continuumSurfacePrimitives =
            privateScratch<NMContinuumSurfacePrimitiveGPU>(
                candidate->device,
                multiplied(4u * world.dispatch.tetrahedronCount +
                           world.dispatch.gridNodeCount),
                valid,
                candidate->residentBytes);
        const std::size_t surfacePrimitiveTotal =
            multiplied(4u * world.dispatch.tetrahedronCount +
                       world.dispatch.gridNodeCount);
        candidate->femSurfaceSortKeysA = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceSortKeysB = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceSortIndicesA = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceSortIndicesB = privateScratch<std::uint32_t>(
            candidate->device, surfacePrimitiveTotal,
            valid, candidate->residentBytes);
        candidate->femSurfaceActiveCounts = privateScratch<std::uint32_t>(
            candidate->device, environments,
            valid, candidate->residentBytes);
        candidate->deformableContactCandidates =
            privateScratch<NMDeformableContactCandidateGPU>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactCandidateCounts =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->deformableContacts = privateScratch<NMDeformableContactGPU>(
            candidate->device,
            multiplied(world.dispatch.deformableContactCapacity),
            valid,
            candidate->residentBytes);
        candidate->deformableContactHistoriesCandidate =
            privateScratch<NMDeformableContactHistoryGPU>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactHistoriesCheckpoint =
            privateScratch<NMDeformableContactHistoryGPU>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveIndices =
            privateScratch<std::uint32_t>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveCounts =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->deformableContactNodeIncidence =
            privateScratch<std::uint32_t>(
                candidate->device,
                4u * multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactNodeRanges =
            privateScratch<NMIncidenceRangeGPU>(
                candidate->device,
                multiplied(world.dispatch.gridNodeCount +
                           world.dispatch.femNodeCount),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveOffsets =
            privateScratch<std::uint32_t>(
                candidate->device,
                environments,
                valid,
                candidate->residentBytes);
        candidate->deformableContactGlobalActiveIndices =
            privateScratch<std::uint32_t>(
                candidate->device,
                multiplied(world.dispatch.deformableContactCapacity),
                valid,
                candidate->residentBytes);
        candidate->deformableContactActiveDispatch =
            privateScratch<NMIndirectDispatchGPU>(
                candidate->device,
                1u,
                valid,
                candidate->residentBytes);
        candidate->mpmNodeGenerations = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.gridNodeCount),
            valid, candidate->residentBytes);
        candidate->mpmParticleKeys = privateScratch<NMMPMParticleKeyGPU>(
            candidate->device, multiplied(world.dispatch.particleCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockCounts = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockOffsets = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockCursors = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockActiveFlags = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmBlockActiveLocalOffsets =
            privateScratch<std::uint32_t>(
                candidate->device,
                multiplied(world.dispatch.mpmBlockCount),
                valid,
                candidate->residentBytes
            );
        candidate->mpmEnvironmentActiveCounts =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->mpmEnvironmentActiveOffsets =
            privateScratch<std::uint32_t>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->mpmActiveBlocks = privateScratch<NMActiveMPMBlockGPU>(
            candidate->device, multiplied(world.dispatch.mpmBlockCount),
            valid, candidate->residentBytes);
        candidate->mpmSortedParticleIndices = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.particleCount),
            valid, candidate->residentBytes);
        candidate->mpmActiveDispatch = privateScratch<NMIndirectDispatchGPU>(
            candidate->device, 1u, valid, candidate->residentBytes);
        candidate->mpmActiveNodeIndices = privateScratch<std::uint32_t>(
            candidate->device,
            multiplied(world.dispatch.mpmActiveNodeCapacity),
            valid, candidate->residentBytes);
        candidate->mpmNodeToActive = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.gridNodeCount),
            valid, candidate->residentBytes);
        candidate->mpmActiveNodeCounts = privateScratch<std::uint32_t>(
            candidate->device, environments,
            valid, candidate->residentBytes);
        candidate->contactSamples = privateScratch<NMContactSampleGPU>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->contactActivePairs = privateScratch<std::uint32_t>(
            candidate->device, multiplied(candidate->contactActiveCapacity),
            valid, candidate->residentBytes);
        candidate->contactActiveSlotsByPair = privateScratch<std::uint32_t>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->contactActiveCounts = privateScratch<std::uint32_t>(
            candidate->device, environments,
            valid, candidate->residentBytes);
        candidate->triggeredMutationCommands =
            privateScratch<NMMutationCommandGPU>(
                candidate->device,
                multiplied(world.dispatch.objectCount),
                valid,
                candidate->residentBytes
            );
        candidate->contactHistoriesCandidate = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->contactHistoriesCheckpoint = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.contactPairCount),
            valid, candidate->residentBytes);
        candidate->articulatedPointQueries =
            privateScratch<MRArticulatedPointImpulseGPU>(
                candidate->device,
                multiplied(candidate->contactActiveCapacity),
                valid,
                candidate->residentBytes
            );
        candidate->coupledGeneralizedInput = privateScratch<float>(
            candidate->device,
            std::max<std::size_t>(
                multiplied(world.dispatch.rigidGeneralizedCapacity), 1u),
            valid,
            candidate->residentBytes
        );
        candidate->coupledGeneralizedOutput = privateScratch<float>(
            candidate->device,
            std::max<std::size_t>(
                multiplied(world.dispatch.rigidGeneralizedCapacity), 1u),
            valid,
            candidate->residentBytes
        );
        candidate->coupledGeneralizedCandidate = privateScratch<float>(
            candidate->device,
            std::max<std::size_t>(
                multiplied(world.dispatch.rigidGeneralizedCapacity), 1u),
            valid,
            candidate->residentBytes
        );
        candidate->coupledGeneralizedPreparedCheckpoint =
            privateScratch<float>(
                candidate->device,
                std::max<std::size_t>(
                    multiplied(world.dispatch.rigidGeneralizedCapacity), 1u),
                valid,
                candidate->residentBytes
            );
        candidate->coupledPointJacobians = privateScratch<float>(
            candidate->device,
            std::max<std::size_t>(
                multiplied(candidate->contactActiveCapacity) * 3u *
                    world.dispatch.rigidGeneralizedCapacity,
                1u),
            valid,
            candidate->residentBytes
        );
        candidate->femHumanAttachmentPointJacobians = privateScratch<float>(
            candidate->device,
            std::max<std::size_t>(
                multiplied(
                    world.dispatch.femHumanAttachmentPointJacobianStride),
                1u),
            valid,
            candidate->residentBytes
        );
        candidate->femHumanAttachmentResidualImpulses =
            privateScratch<nm_float4>(
                candidate->device,
                multiplied(world.dispatch.femHumanAttachmentCount),
                valid,
                candidate->residentBytes
            );
        candidate->femHumanAttachmentRawReactions =
            privateScratch<nm_float4>(
                candidate->device,
                multiplied(world.dispatch.femHumanAttachmentCount),
                valid,
                candidate->residentBytes
            );
        candidate->femHumanAttachmentOperator =
            privateScratch<nm_float4>(
                candidate->device,
                multiplied(world.dispatch.femHumanAttachmentCount),
                valid,
                candidate->residentBytes
            );
        candidate->coupledInverseStatuses =
            privateScratch<MRInverseMassStatusGPU>(
                candidate->device,
                multiplied(std::max<std::size_t>(
                    world.contact.rigidProxies.size(), 1u)),
                valid,
                candidate->residentBytes
            );
        candidate->microstepReactions = privateScratch<NMRigidReactionGPU>(
            candidate->device, multiplied(world.dispatch.rigidProxyCount),
            valid, candidate->residentBytes);
        candidate->frameReactions = privateScratch<NMRigidReactionGPU>(
            candidate->device, multiplied(world.dispatch.rigidProxyCount),
            valid, candidate->residentBytes);
        candidate->frameReactionsPreparedCheckpoint =
            privateScratch<NMRigidReactionGPU>(
                candidate->device,
                multiplied(world.dispatch.rigidProxyCount),
                valid,
                candidate->residentBytes);
        candidate->elementForces = privateScratch<NMFEMElementVectorGPU>(
            candidate->device, multiplied(world.dispatch.tetrahedronCount),
            valid, candidate->residentBytes);
        candidate->elementOperator = privateScratch<NMFEMElementVectorGPU>(
            candidate->device, multiplied(world.dispatch.tetrahedronCount),
            valid, candidate->residentBytes);
        const std::size_t mixedUnknownWidth =
            2u * static_cast<std::size_t>(world.dispatch.femNodeCount) +
            world.dispatch.mpmActiveNodeCapacity +
            world.dispatch.rigidGeneralizedCapacity;
        const std::size_t mixedUnknownTotal = multiplied(mixedUnknownWidth);
        candidate->femSolution = privateScratch<nm_float4>(
            candidate->device, mixedUnknownTotal,
            valid, candidate->residentBytes);
        candidate->femResidual = privateScratch<nm_float4>(
            candidate->device, mixedUnknownTotal,
            valid, candidate->residentBytes);
        candidate->femPreconditioned = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femDirection = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femOperatorValue = privateScratch<nm_float4>(
            candidate->device, mixedUnknownTotal,
            valid, candidate->residentBytes);
        candidate->femConstraintReactions = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.femNodeCount),
            valid, candidate->residentBytes);
        candidate->femLineSearch = privateScratch<nm_float4>(
            candidate->device, multiplied(world.dispatch.objectCount),
            valid, candidate->residentBytes);
        const std::size_t fgmresUnknownTotal = mixedUnknownTotal;
        const std::size_t fgmresSystemTotal = multiplied(1u);
        const std::size_t configuredRestart = std::clamp<std::size_t>(
            world.mixedSolver.nonlinearIterations.y,
            1u,
            NM_MIXED_FGMRES_RESTART);
        candidate->fgmresBasis = privateScratch<nm_float4>(
            candidate->device,
            fgmresUnknownTotal * (configuredRestart + 1u),
            valid, candidate->residentBytes);
        candidate->fgmresPreconditionedBasis = privateScratch<nm_float4>(
            candidate->device,
            fgmresUnknownTotal * configuredRestart,
            valid, candidate->residentBytes);
        candidate->fgmresHessenberg = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * (NM_MIXED_FGMRES_RESTART + 1u) *
                NM_MIXED_FGMRES_RESTART,
            valid, candidate->residentBytes);
        candidate->fgmresRotations = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * NM_MIXED_FGMRES_RESTART * 2u,
            valid, candidate->residentBytes);
        candidate->fgmresLeastSquares = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * (NM_MIXED_FGMRES_RESTART + 1u),
            valid, candidate->residentBytes);
        candidate->fgmresRestartCoefficients = privateScratch<float>(
            candidate->device,
            fgmresSystemTotal * (NM_MIXED_FGMRES_RESTART + 1u),
            valid, candidate->residentBytes);
        candidate->fgmresStates = privateScratch<NMFGMRESStateGPU>(
            candidate->device, fgmresSystemTotal,
            valid, candidate->residentBytes);
        candidate->solverCertificates = privateScratch<NMSolverCertificateGPU>(
            candidate->device, multiplied(world.dispatch.objectCount),
            valid, candidate->residentBytes);
        candidate->identificationCandidates =
            privateScratch<NMIdentificationCandidateGPU>(
                candidate->device,
                static_cast<std::size_t>(world.dispatch.identificationCandidateCount) *
                    world.identification.size(),
                valid,
                candidate->residentBytes
            );
        candidate->identificationLosses = sharedScratch<float>(
            candidate->device,
            world.dispatch.identificationCandidateCount,
            valid,
            candidate->residentBytes
        );
        candidate->statuses = sharedScratch<NMMatterStatusGPU>(
            candidate->device,
            environments,
            valid,
            candidate->residentBytes
        );
        const auto proofBytes = [&](const std::uint64_t count,
                                    const std::uint64_t elementBytes) {
            if (count != 0u && elementBytes >
                    std::numeric_limits<std::uint64_t>::max() / count) {
                valid = false;
                return std::uint64_t{0u};
            }
            return count * elementBytes;
        };
        const std::array proofArenaBytes{
            proofBytes(candidate->dispatch.rigidQCapacity, sizeof(float)),
            proofBytes(
                candidate->dispatch.rigidGeneralizedCapacity, sizeof(float)),
            candidate->acceptedStateProofMujocoBytesPerEnvironmentCapacity,
            proofBytes(
                candidate->dispatch.rigidGeneralizedCapacity, sizeof(float)),
            proofBytes(candidate->dispatch.particleCount,
                       sizeof(NMParticleStateGPU)),
            proofBytes(
                proofBytes(candidate->dispatch.particleCount,
                           candidate->dispatch.materialStateStride),
                sizeof(float)),
            proofBytes(candidate->dispatch.femNodeCount,
                       sizeof(NMFEMNodeStateGPU)),
            proofBytes(
                proofBytes(candidate->dispatch.tetrahedronCount,
                           candidate->dispatch.materialStateStride),
                sizeof(float)),
            proofBytes(candidate->dispatch.femNodeCount,
                       sizeof(NMFEMFieldStateGPU)),
            proofBytes(candidate->dispatch.tetrahedronCount,
                       sizeof(NMTetrahedronGPU)),
            proofBytes(candidate->dispatch.topologyNodeCapacity,
                       sizeof(NMFEMTopologyNodeGPU)),
            proofBytes(candidate->dispatch.cohesiveFaceCount,
                       sizeof(NMCohesiveFaceGPU)),
            proofBytes(candidate->dispatch.punctureChannelCount,
                       sizeof(NMPunctureChannelGPU)),
            proofBytes(candidate->dispatch.objectCount,
                       sizeof(NMFEMTopologyStateGPU)),
            proofBytes(candidate->dispatch.learnedWeightCount, sizeof(float)),
            proofBytes(candidate->dispatch.contactPairCount, sizeof(nm_float4)),
            proofBytes(candidate->dispatch.deformableContactCapacity,
                       sizeof(NMDeformableContactHistoryGPU)),
            proofBytes(candidate->dispatch.rigidGeneralizedCapacity,
                       sizeof(float)),
            proofBytes(candidate->dispatch.rigidProxyCount,
                       sizeof(NMRigidReactionGPU)),
            proofBytes(candidate->dispatch.rigidProxyCount,
                       sizeof(NMRigidStateGPU)),
            proofBytes(candidate->dispatch.objectCount,
                       sizeof(NMAdaptiveStateGPU)),
            proofBytes(candidate->dispatch.objectCount,
                       sizeof(NMSchedulerStateGPU)),
            proofBytes(candidate->identificationDistributionCount,
                       sizeof(NMIdentificationDistributionGPU)),
            proofBytes(candidate->dispatch.parameterCount, sizeof(float)),
            proofBytes(candidate->femNodeIncidenceStride,
                       sizeof(std::uint32_t)),
            proofBytes(candidate->femNodeRangeStride,
                       sizeof(NMIncidenceRangeGPU)),
        };
        const std::uint64_t maximumProofArenaBytes =
            *std::max_element(proofArenaBytes.begin(), proofArenaBytes.end());
        const std::uint64_t proofScratchStride64 = std::max<std::uint64_t>(
            1u,
            maximumProofArenaBytes /
                    detail::kAcceptedStateProofChunkBytes +
                (maximumProofArenaBytes %
                     detail::kAcceptedStateProofChunkBytes != 0u));
        if (proofScratchStride64 >
                std::numeric_limits<std::uint32_t>::max()) {
            valid = false;
        } else {
            candidate->acceptedStateProofScratchStride =
                static_cast<std::uint32_t>(proofScratchStride64);
        }
        const std::uint64_t proofScratchCount64 = valid
            ? static_cast<std::uint64_t>(environments) *
                candidate->acceptedStateProofScratchStride
            : 0u;
        const std::uint64_t proofHashCount64 =
            static_cast<std::uint64_t>(environments) * 2u;
        if (proofScratchCount64 >
                std::numeric_limits<std::uint32_t>::max() ||
            proofScratchCount64 >
                std::numeric_limits<std::size_t>::max() ||
            proofHashCount64 >
                std::numeric_limits<std::uint32_t>::max() ||
            proofHashCount64 >
                std::numeric_limits<std::size_t>::max()) {
            valid = false;
        }
        const std::size_t proofScratchCount = valid
            ? static_cast<std::size_t>(proofScratchCount64)
            : 0u;
        const std::size_t proofResidentBefore = candidate->residentBytes;
        candidate->acceptedStateProofHashes = privateScratch<std::uint64_t>(
            candidate->device,
            valid ? static_cast<std::size_t>(proofHashCount64) : 0u,
            valid,
            candidate->residentBytes
        );
        candidate->acceptedStateProofScratchA = privateScratch<std::uint64_t>(
            candidate->device,
            proofScratchCount,
            valid,
            candidate->residentBytes
        );
        candidate->acceptedStateProofScratchB = privateScratch<std::uint64_t>(
            candidate->device,
            proofScratchCount,
            valid,
            candidate->residentBytes
        );
        candidate->preparedStateBindings =
            privateScratch<NMPreparedStateBindingGPU>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->preparedStateActions = privateScratch<std::uint32_t>(
            candidate->device, environments,
            valid, candidate->residentBytes);
        candidate->preparedStateRestoreStatuses =
            privateScratch<NMMatterStatusGPU>(
                candidate->device, environments,
                valid, candidate->residentBytes);
        candidate->preparedStateApplyOutcome =
            sharedScratch<std::uint32_t>(
                candidate->device, 1u,
                valid, candidate->residentBytes);
        candidate->preparedStatePublicationFacts =
            sharedScratch<NMPreparedStatePublicationFactsGPU>(
                candidate->device, 1u,
                valid, candidate->residentBytes);
        candidate->acceptedStateProofResidentByteCount =
            candidate->residentBytes - proofResidentBefore;
        candidate->events = sharedScratch<NMEventTokenGPU>(
            candidate->device,
            multiplied(world.dispatch.eventStride),
            valid,
            candidate->residentBytes
        );
        candidate->dummy = sharedScratch<std::uint8_t>(
            candidate->device, 256u, valid, candidate->residentBytes);
        if (!valid) {
            diagnostics.message = "failed to allocate persistent matter runtime state";
            return diagnostics;
        }

        candidate->primalContactArguments = [candidate->device
            newBufferWithLength:primalContactEncoder.encodedLength
                         options:MTLResourceStorageModeShared];
        if (candidate->primalContactArguments == nil) {
            diagnostics.message =
                "failed to allocate monolithic primal-contact argument table";
            return diagnostics;
        }
        [primalContactEncoder
            setArgumentBuffer:candidate->primalContactArguments offset:0u];
        [primalContactEncoder setBuffer:candidate->contactPairs
                                 offset:0u atIndex:0u];
        [primalContactEncoder setBuffer:candidate->contactSamples
                                 offset:0u atIndex:1u];
        [primalContactEncoder setBuffer:candidate->contactNodeIncidence
                                 offset:0u atIndex:2u];
        [primalContactEncoder setBuffer:candidate->contactNodeRanges
                                 offset:0u atIndex:3u];
        [primalContactEncoder setBuffer:candidate->contactActivePairs
                                 offset:0u atIndex:4u];
        [primalContactEncoder setBuffer:candidate->contactActiveSlotsByPair
                                 offset:0u atIndex:5u];
        [primalContactEncoder setBuffer:candidate->contactActiveCounts
                                 offset:0u atIndex:6u];
        [primalContactEncoder setBuffer:candidate->deformableContacts
                                 offset:0u atIndex:7u];
        [primalContactEncoder
            setBuffer:candidate->deformableContactNodeIncidence
               offset:0u atIndex:8u];
        [primalContactEncoder
            setBuffer:candidate->deformableContactNodeRanges
               offset:0u atIndex:9u];
        [primalContactEncoder setBuffer:candidate->mpmNodeToActive
                                 offset:0u atIndex:10u];
        [primalContactEncoder setBuffer:candidate->mpmActiveNodeCounts
                                 offset:0u atIndex:11u];
        [primalContactEncoder setBuffer:candidate->rigidProxies
                                 offset:0u atIndex:12u];
        [primalContactEncoder setBuffer:candidate->rigidStates
                                 offset:0u atIndex:13u];
        [primalContactEncoder
            setBuffer:candidate->deformableContactActiveCounts
               offset:0u atIndex:14u];
        candidate->residentBytes += primalContactEncoder.encodedLength;

        for (const NMRigidProxyGPU& proxy : world.contact.rigidProxies) {
            if ((proxy.flags & NM_RIGID_SUTURE_STRAND) != 0u) {
                candidate->requiresRodNodes = true;
                candidate->requiredRodNodeCount = std::max({
                    candidate->requiredRodNodeCount,
                    proxy.bodyIndex + 1u,
                    proxy.sceneBodyIndex + 1u,
                });
                continue;
            }
            if (proxy.bodyIndex != NM_INVALID_INDEX) {
                candidate->requiresCurrentBodies = true;
                candidate->requiredCurrentBodyCount = std::max(
                    candidate->requiredCurrentBodyCount,
                    proxy.bodyIndex + 1u
                );
            }
            if ((proxy.flags &
                 (NM_RIGID_ARTICULATED | NM_RIGID_DYNAMIC)) != 0u) {
                candidate->requiresBodyWrenches = true;
                candidate->requiredBodyWrenchCount = std::max(
                    candidate->requiredBodyWrenchCount,
                    proxy.bodyIndex + 1u
                );
            }
            if ((proxy.flags & NM_RIGID_ARTICULATED) != 0u) {
                candidate->requiresCoupledCandidate = true;
            }
            if ((proxy.flags & NM_RIGID_DYNAMIC) != 0u) {
                candidate->requiresSceneBodies = true;
                candidate->requiredSceneBodyCount = std::max(
                    candidate->requiredSceneBodyCount,
                    proxy.sceneBodyIndex + 1u
                );
                if (proxy.sceneBodyIndex == NM_INVALID_INDEX) {
                    diagnostics.message =
                        "dynamic rigid proxies require a scene-body index";
                    return diagnostics;
                }
            }
        }

        if (!world.fem.humanAttachments.empty()) {
            candidate->requiresCoupledCandidate = true;
            for (const NMFEMHumanAttachmentGPU& attachment :
                 world.fem.humanAttachments) {
                candidate->requiredCandidateBodyCount = std::max(
                    candidate->requiredCandidateBodyCount,
                    attachment.identity.y + 1u
                );
            }
        }

        candidate->reactionBodyCount =
            candidate->requiredBodyWrenchCount;
        std::vector<std::vector<std::uint32_t>> bodyProxyOwners(
            candidate->reactionBodyCount
        );
        for (std::uint32_t proxyIndex = 0u;
             proxyIndex < world.contact.rigidProxies.size();
             ++proxyIndex) {
            const NMRigidProxyGPU& proxy =
                world.contact.rigidProxies[proxyIndex];
            if (proxy.bodyIndex != NM_INVALID_INDEX &&
                (proxy.flags &
                 (NM_RIGID_ARTICULATED | NM_RIGID_DYNAMIC)) != 0u) {
                bodyProxyOwners[proxy.bodyIndex].push_back(proxyIndex);
            }
        }

        std::set<std::uint32_t> adaptiveProxyBindings;
        std::set<std::uint32_t> adaptiveBodyBindings;
        std::set<std::uint32_t> adaptiveSceneBindings;
        for (const NMContinuumObjectGPU& object : world.objects) {
            if ((object.flags & NM_OBJECT_ADAPTIVE) == 0u) {
                continue;
            }
            if (object.rigidBinding >= world.contact.rigidProxies.size()) {
                diagnostics.message =
                    "adaptive Matter object has no valid rigid proxy";
                return diagnostics;
            }
            const NMRigidProxyGPU& proxy =
                world.contact.rigidProxies[object.rigidBinding];
            if ((proxy.flags & NM_RIGID_DYNAMIC) == 0u ||
                (proxy.flags & NM_RIGID_ARTICULATED) != 0u ||
                proxy.bodyIndex == NM_INVALID_INDEX ||
                proxy.sceneBodyIndex == NM_INVALID_INDEX ||
                !adaptiveProxyBindings.insert(object.rigidBinding).second ||
                !adaptiveBodyBindings.insert(proxy.bodyIndex).second ||
                !adaptiveSceneBindings.insert(proxy.sceneBodyIndex).second) {
                diagnostics.message =
                    "adaptive Matter objects require unique free-dynamic proxy and scene-body bindings";
                return diagnostics;
            }
        }

        std::vector<std::uint32_t> bodyProxyIncidence;
        std::vector<NMIncidenceRangeGPU> bodyProxyRanges(
            candidate->reactionBodyCount
        );
        for (std::uint32_t bodyIndex = 0u;
             bodyIndex < candidate->reactionBodyCount;
             ++bodyIndex) {
            NMIncidenceRangeGPU range{};
            range.first = static_cast<nm_u32>(bodyProxyIncidence.size());
            range.count = static_cast<nm_u32>(bodyProxyOwners[bodyIndex].size());
            range.objectIndex = bodyIndex;
            bodyProxyIncidence.insert(
                bodyProxyIncidence.end(),
                bodyProxyOwners[bodyIndex].begin(),
                bodyProxyOwners[bodyIndex].end()
            );
            bodyProxyRanges[bodyIndex] = range;
        }
        UploadPlan bridgeUploads(candidate->device, candidate->queue);
        candidate->bodyProxyIncidence = bridgeUploads.one(
            std::span<const std::uint32_t>(bodyProxyIncidence),
            valid,
            candidate->residentBytes
        );
        candidate->bodyProxyRanges = bridgeUploads.one(
            std::span<const NMIncidenceRangeGPU>(bodyProxyRanges),
            valid,
            candidate->residentBytes
        );
        std::string bridgeUploadError;
        if (!valid || !bridgeUploads.finish(bridgeUploadError)) {
            diagnostics.message = valid
                ? "failed to upload body-owned Matter coupling incidence: " +
                    bridgeUploadError
                : "failed to allocate body-owned Matter coupling incidence";
            return diagnostics;
        }

        diagnostics.encoded = true;
        diagnostics.residentBytes = candidate->residentBytes;
        diagnostics.device = nsString(candidate->device.name);
        diagnostics.message = "Numi Matter runtime initialized";
        state_ = std::move(candidate);
        return diagnostics;
    }
}

RuntimeDiagnostics Runtime::encode(const EncodeRequest& request) {
    return encodeImpl(request, false);
}

RuntimeDiagnostics Runtime::prepareAcceptedState(
    const EncodeRequest& postCommitRequest
) {
    if (postCommitRequest.phase != EncodePhase::postCommit) {
        RuntimeDiagnostics diagnostics;
        diagnostics.message =
            "prepared-state reconciliation requires a postCommit request";
        return diagnostics;
    }
    if (!postCommitRequest.enablePreparedState ||
        postCommitRequest.physicsSubsteps != 1u ||
        postCommitRequest.physicsSubstep != 0u) {
        RuntimeDiagnostics diagnostics;
        diagnostics.message =
            "prepared-state v1 requires an explicitly opted-in single-substep transaction";
        return diagnostics;
    }
    return encodeImpl(postCommitRequest, true);
}

RuntimeDiagnostics Runtime::encodeImpl(
    const EncodeRequest& request,
    const bool retainPreparedState
) {
    @autoreleasepool {
        RuntimeDiagnostics diagnostics;
        if (!state_) {
            diagnostics.message = "Numi Matter runtime is not initialized";
            return diagnostics;
        }
        if (request.commandBuffer == nullptr) {
            diagnostics.message = "encode requires a borrowed Metal command buffer";
            return diagnostics;
        }
        State& state = *state_;
        if (request.phase != EncodePhase::preDynamics &&
            request.phase != EncodePhase::postCommit) {
            diagnostics.message = "unknown Matter encode phase";
            return diagnostics;
        }
        if (request.physicsSubsteps == 0u ||
            request.physicsSubstep >= request.physicsSubsteps) {
            diagnostics.message = "matter physics-substep coordinates are invalid";
            return diagnostics;
        }
        if (request.enablePreparedState &&
            (request.physicsSubsteps != 1u ||
             request.physicsSubstep != 0u)) {
            diagnostics.message =
                "prepared-state v1 supports exactly one physics substep";
            return diagnostics;
        }
        if (request.enablePreparedState &&
            (request.runIdentification ||
             request.resetMaskStepStride != 0u)) {
            diagnostics.message =
                "prepared-state v1 rejects identification and episode reset until their control-plane authority is covered by ACK-gated apply/restore";
            return diagnostics;
        }
        if (request.enablePreparedState &&
            (state.hasAdaptive || request.runAdaptiveTransfer)) {
            diagnostics.message =
                "prepared-state v1 rejects adaptive worlds/transfers until borrowed current-body and scene-body rollback authority is covered by ACK-gated apply/restore";
            return diagnostics;
        }
        if (state.requiresCurrentBodies &&
            request.rigid.currentBodies == nullptr) {
            diagnostics.message =
                "body-backed matter proxies require the current body arena";
            return diagnostics;
        }
        if (request.phase == EncodePhase::preDynamics &&
            state.requiresBodyWrenches &&
            request.rigid.bodyWrenches == nullptr) {
            diagnostics.message =
                "two-way matter coupling requires the global body-wrench arena";
            return diagnostics;
        }
        if (request.phase == EncodePhase::preDynamics &&
            state.requiresCoupledCandidate &&
            (request.encodeCoupledCandidate == nullptr ||
             request.coupledCandidateContext == nullptr ||
             request.rigid.q == nullptr || request.rigid.v == nullptr ||
             request.rigid.qStride == 0u || request.rigid.vStride == 0u ||
             request.rigid.qStride > state.dispatch.rigidQCapacity ||
             request.rigid.vStride >
                 state.dispatch.rigidGeneralizedCapacity ||
             (state.dispatch.femHumanAttachmentCount != 0u &&
              (request.rigid.currentBodyCount <
                   state.requiredCandidateBodyCount ||
               request.rigid.currentBodyStride <
                   state.requiredCandidateBodyCount)))) {
            diagnostics.message =
                "articulated Matter IPC requires a compatible primal coupled-candidate service plus exact q/v and candidate-body capacity";
            return diagnostics;
        }
        if (state.requiresSceneBodies &&
            request.rigid.sceneBodies == nullptr) {
            diagnostics.message =
                "dynamic matter coupling requires the scene-body arena";
            return diagnostics;
        }
        if (state.requiresRodNodes &&
            (request.rigid.rodNodes == nullptr ||
             request.rigid.rodInverseMasses == nullptr)) {
            diagnostics.message =
                "suture-strand matter proxies require live DER node and inverse-mass arenas";
            return diagnostics;
        }
        if (request.phase == EncodePhase::postCommit &&
            request.environmentStatuses == nullptr) {
            diagnostics.message =
                "post-commit matter reconciliation requires MetalWorld status";
            return diagnostics;
        }
        if (request.environmentStatuses == nullptr) {
            diagnostics.message =
                "coupled Matter execution requires MetalWorld environment status";
            return diagnostics;
        }
        const std::uint64_t requiredFEMExternalForceCount =
            static_cast<std::uint64_t>(state.dispatch.environmentCount) *
            state.dispatch.femNodeCount;
        if ((request.femExternalForces == nullptr) !=
                (request.femExternalForceCount == 0u) ||
            (request.femExternalForces != nullptr &&
             request.femExternalForceCount != requiredFEMExternalForceCount)) {
            diagnostics.message =
                "borrowed FEM external-force field must exactly cover every environment and cooked node";
            return diagnostics;
        }
        if ((request.femKinematicTargets == nullptr) !=
                (request.femKinematicTargetCount == 0u) ||
            (request.femKinematicTargets != nullptr &&
             request.femKinematicTargetCount != requiredFEMExternalForceCount)) {
            diagnostics.message =
                "borrowed FEM kinematic-target field must exactly cover every environment and cooked node";
            return diagnostics;
        }
        if (request.rigid.currentBodyCount >
                request.rigid.currentBodyStride ||
            request.rigid.bodyWrenchCount >
                request.rigid.bodyWrenchStride ||
            request.rigid.sceneBodyCount >
                request.rigid.sceneStride ||
            request.rigid.rodNodeCount > request.rigid.rodNodeStride) {
            diagnostics.message =
                "borrowed rigid-world buffer strides are invalid";
            return diagnostics;
        }
        if (request.rigid.currentBodyCount <
                state.requiredCurrentBodyCount ||
            request.rigid.bodyWrenchCount <
                state.requiredBodyWrenchCount ||
            request.rigid.sceneBodyCount <
                state.requiredSceneBodyCount ||
            request.rigid.rodNodeCount < state.requiredRodNodeCount) {
            diagnostics.message =
                "borrowed rigid-world buffers do not cover every compiled Matter proxy";
            return diagnostics;
        }
        if (request.runIdentification &&
            (request.phase != EncodePhase::preDynamics ||
             request.physicsSubstep != 0u)) {
            diagnostics.message =
                "inverse material identification may run only in the first pre-dynamics substep";
            return diagnostics;
        }
        if (request.runIdentification && state.requiresCoupledCandidate) {
            diagnostics.message =
                "accepted-state attached Matter transactions reject in-place identification until device rollback authority is available";
            return diagnostics;
        }
        if (request.runAdaptiveTransfer &&
            (request.phase != EncodePhase::postCommit ||
             request.physicsSubstep + 1u != request.physicsSubsteps)) {
            diagnostics.message =
                "adaptive representation transfer may run only after the final rigid substep commits";
            return diagnostics;
        }
        if (request.resetMaskStepStride != 0u &&
            (request.resetMasks == nullptr ||
             request.resetMaskStepStride < state.dispatch.environmentCount)) {
            diagnostics.message =
                "matter reset-mask stream is missing or undersized";
            return diagnostics;
        }
        if (request.learnedWeightUpdate != nullptr &&
            request.phase == EncodePhase::preDynamics &&
            (request.physicsSubstep != 0u ||
             request.learnedWeightCount != state.dispatch.learnedWeightCount ||
             request.learnedWeightRevision == 0u ||
             (request.expectedLearnedFingerprint != 0u &&
              request.expectedLearnedFingerprint != state.learnedFingerprint))) {
            diagnostics.message =
                "learned weights require a complete, newer first-substep candidate";
            return diagnostics;
        }
        if (request.learnedWeightUpdate == nullptr &&
            request.learnedWeightCount != 0u) {
            diagnostics.message = "learned weight count has no borrowed buffer";
            return diagnostics;
        }
        if (request.mutationCommandCount != 0u &&
            request.phase == EncodePhase::preDynamics &&
            (request.mutationCommands == nullptr ||
             request.physicsSubstep != 0u ||
             request.mutationCommandStride < sizeof(NMMutationCommandGPU))) {
            diagnostics.message =
                "mutation commands require a fixed-stride first-substep device stream";
            return diagnostics;
        }
        if (request.mutationCommandCount != 0u &&
            request.expectedMutationFingerprint != 0u &&
            request.expectedMutationFingerprint != state.mutationFingerprint) {
            diagnostics.message = "borrowed mutation fingerprint is stale";
            return diagnostics;
        }
        const float cookedTimestep =
            state.dispatch.gravityAndTimestep.w;
        const float activeTimestep = cookedTimestep *
            static_cast<float>(
                state.coupledTimestepMultiplier.load(
                    std::memory_order_acquire
                )
            ) /
            static_cast<float>(
                state.coupledTimestepDivisor.load(
                    std::memory_order_acquire
                )
            );
        const float frameTimestep = request.timestepSeconds > 0.0f
            ? request.timestepSeconds
            : activeTimestep;
        if (!std::isfinite(frameTimestep) || !(frameTimestep > 0.0f)) {
            diagnostics.message =
                "matter frame timestep must be finite and positive";
            return diagnostics;
        }
        if (frameTimestep != activeTimestep) {
            diagnostics.message =
                "Matter runtime timestep differs from the active coupled cadence";
            return diagnostics;
        }

        id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)request.commandBuffer;
        id<MTLBuffer> femExternalForces = request.femExternalForces == nullptr
            ? state.dummy
            : (__bridge id<MTLBuffer>)request.femExternalForces;
        const std::uint32_t hasFEMExternalForces =
            request.femExternalForces == nullptr ? 0u : 1u;
        if (request.femExternalForces != nullptr &&
            femExternalForces.length <
                static_cast<NSUInteger>(request.femExternalForceCount) *
                    4u * sizeof(float)) {
            diagnostics.message = "borrowed FEM external-force Metal buffer is undersized";
            return diagnostics;
        }
        id<MTLBuffer> femKinematicTargets =
            request.femKinematicTargets == nullptr
                ? state.dummy
                : (__bridge id<MTLBuffer>)request.femKinematicTargets;
        const std::uint32_t hasFEMKinematicTargets =
            request.femKinematicTargets == nullptr ? 0u : 1u;
        if (request.femKinematicTargets != nullptr &&
            femKinematicTargets.length <
                static_cast<NSUInteger>(request.femKinematicTargetCount) *
                    4u * sizeof(float)) {
            diagnostics.message =
                "borrowed FEM kinematic-target Metal buffer is undersized";
            return diagnostics;
        }
        if (commandBuffer.commandQueue == nil ||
            commandBuffer.commandQueue.device.registryID !=
                state.device.registryID) {
            diagnostics.message =
                "borrowed command buffer does not belong to the initialized Matter device";
            return diagnostics;
        }
        if (state.requiresCoupledCandidate) {
            id<MTLBuffer> q = (__bridge id<MTLBuffer>)request.rigid.q;
            id<MTLBuffer> v = (__bridge id<MTLBuffer>)request.rigid.v;
            const std::uint64_t requiredQBytes =
                static_cast<std::uint64_t>(state.dispatch.environmentCount) *
                request.rigid.qStride * sizeof(float);
            const std::uint64_t requiredVBytes =
                static_cast<std::uint64_t>(state.dispatch.environmentCount) *
                request.rigid.vStride * sizeof(float);
            if (q == nil || v == nil ||
                q.device.registryID != state.device.registryID ||
                v.device.registryID != state.device.registryID ||
                q.length < requiredQBytes || v.length < requiredVBytes) {
                diagnostics.message =
                    "borrowed coupled q/v buffers have wrong device provenance or byte capacity";
                return diagnostics;
            }
        }

        const auto ownership = state.commandOwnership;
        std::unique_lock ownershipLock(ownership->mutex);
        if (ownership->preparedStateOpen || ownership->applyEncoded ||
            ownership->restoreRequired) {
            diagnostics.message =
                "a prepared Matter transaction still requires apply, publication, or restore";
            return diagnostics;
        }
        if (ownership->activeCommandBuffer != nullptr &&
            ownership->activeCommandBuffer != request.commandBuffer) {
            diagnostics.message =
                "Numi Matter runtime already has another command buffer in flight";
            return diagnostics;
        }
        if (request.phase == EncodePhase::preDynamics &&
            ownership->preDynamicsOpen) {
            diagnostics.message =
                "a Matter pre-dynamics pass is still awaiting post-commit reconciliation";
            return diagnostics;
        }
        if (request.phase == EncodePhase::postCommit &&
            (!ownership->preDynamicsOpen ||
             ownership->controlStep != request.controlStep ||
             ownership->physicsSubstep != request.physicsSubstep)) {
            diagnostics.message =
                "Matter post-commit pass does not match its pre-dynamics transaction";
            return diagnostics;
        }
        if (retainPreparedState && !ownership->preparedStateRequested) {
            diagnostics.message =
                "prepared-state postCommit was not opted in at preDynamics";
            return diagnostics;
        }
        if (request.phase == EncodePhase::postCommit &&
            ownership->acceptedStateProofEncoded &&
            request.runAdaptiveTransfer) {
            diagnostics.message =
                "accepted-state proof cannot precede adaptive post-commit mutation";
            return diagnostics;
        }
        if (request.runIdentification &&
            ownership->identificationAdvanced) {
            diagnostics.message =
                "inverse material identification may advance only once per command buffer";
            return diagnostics;
        }
        const bool firstEncodeForCommandBuffer =
            ownership->activeCommandBuffer == nullptr;

        if (request.phase == EncodePhase::preDynamics &&
            state.requiresCoupledCandidate) {
            const std::uint32_t requiredQStride = request.rigid.qStride;
            const std::uint32_t requiredVStride = request.rigid.vStride;
            const std::uint32_t requiredBodyStride =
                request.rigid.currentBodyStride;
            if ((state.coupledQStride != 0u &&
                 state.coupledQStride != requiredQStride) ||
                (state.coupledVStride != 0u &&
                 state.coupledVStride != requiredVStride) ||
                (state.coupledBodyStride != 0u &&
                 state.coupledBodyStride != requiredBodyStride)) {
                diagnostics.message =
                    "MetalWorld coupled candidate strides changed after runtime allocation";
                return diagnostics;
            }
            if (state.coupledCandidateQ == nil) {
                bool allocationValid = true;
                const NSUInteger qBytes = checkedBytes(
                    static_cast<std::size_t>(state.dispatch.environmentCount) *
                        requiredQStride,
                    sizeof(float),
                    allocationValid);
                const NSUInteger bodyBytes = checkedBytes(
                    static_cast<std::size_t>(state.dispatch.environmentCount) *
                        requiredBodyStride,
                    sizeof(MRBodyStateGPU),
                    allocationValid);
                if (!allocationValid) {
                    diagnostics.message =
                        "coupled candidate allocation exceeds host address space";
                    return diagnostics;
                }
                state.coupledCandidateQ = [state.device
                    newBufferWithLength:qBytes
                                options:MTLResourceStorageModePrivate];
                state.coupledCandidateBodies = [state.device
                    newBufferWithLength:bodyBytes
                                options:MTLResourceStorageModePrivate];
                if (state.coupledCandidateQ == nil ||
                    state.coupledCandidateBodies == nil) {
                    state.coupledCandidateQ = nil;
                    state.coupledCandidateBodies = nil;
                    diagnostics.message =
                        "failed to allocate private coupled candidate kinematics";
                    return diagnostics;
                }
                state.coupledQStride = requiredQStride;
                state.coupledVStride = requiredVStride;
                state.coupledBodyStride = requiredBodyStride;
                state.residentBytes += qBytes + bodyBytes;
            }
        }

        const bool firstPrePass =
            request.phase == EncodePhase::preDynamics &&
            request.physicsSubstep == 0u;
        if (firstPrePass && request.enablePreparedState) {
            id<MTLBlitCommandEncoder> checkpointBlit =
                [commandBuffer blitCommandEncoder];
            if (checkpointBlit == nil) {
                diagnostics.message =
                    "failed to create prepared-state checkpoint blit";
                return diagnostics;
            }
            const std::array checkpointPairs{
                std::pair{id<MTLBuffer>(state.adaptive),
                          id<MTLBuffer>(state.adaptivePreparedCheckpoint)},
                std::pair{id<MTLBuffer>(state.schedulerPrevious),
                          id<MTLBuffer>(
                              state.schedulerPreviousPreparedCheckpoint)},
                std::pair{id<MTLBuffer>(state.frameReactions),
                          id<MTLBuffer>(
                              state.frameReactionsPreparedCheckpoint)},
                std::pair{id<MTLBuffer>(state.coupledGeneralizedCandidate),
                          id<MTLBuffer>(
                              state.coupledGeneralizedPreparedCheckpoint)},
                std::pair{id<MTLBuffer>(state.rigidStates),
                          id<MTLBuffer>(
                              state.rigidStatesPreparedCheckpoint)},
                std::pair{id<MTLBuffer>(state.identificationDistributions),
                          id<MTLBuffer>(
                              state.identificationPreparedCheckpoint)},
                std::pair{id<MTLBuffer>(state.environmentParameters),
                          id<MTLBuffer>(
                              state.environmentParametersPreparedCheckpoint)},
            };
            bool checkpointValid = true;
            for (const auto& [source, destination] : checkpointPairs) {
                if (source == nil || destination == nil ||
                    destination.length < source.length) {
                    checkpointValid = false;
                    break;
                }
                if (source.length != 0u) {
                    [checkpointBlit copyFromBuffer:source sourceOffset:0u
                                         toBuffer:destination
                                destinationOffset:0u size:source.length];
                }
            }
            if (checkpointValid) {
                [checkpointBlit fillBuffer:state.preparedStateBindings
                                     range:NSMakeRange(
                                         0u,
                                         state.preparedStateBindings.length)
                                     value:0u];
                [checkpointBlit fillBuffer:state.preparedStateActions
                                     range:NSMakeRange(
                                         0u,
                                         state.preparedStateActions.length)
                                     value:0u];
            }
            [checkpointBlit endEncoding];
            if (!checkpointValid) {
                diagnostics.message =
                    "prepared-state checkpoint arena is unavailable";
                return diagnostics;
            }
        }

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            diagnostics.message =
                "failed to create borrowed matter compute encoder";
            return diagnostics;
        }
        [encoder setLabel:request.phase == EncodePhase::preDynamics
            ? @"Numi Matter pre-dynamics"
            : @"Numi Matter post-commit"];

        const auto buffer = [&](void* value) -> id<MTLBuffer> {
            return value == nullptr
                ? state.dummy
                : (__bridge id<MTLBuffer>)value;
        };
        id<MTLBuffer> currentBodies = buffer(request.rigid.currentBodies);
        id<MTLBuffer> bodyWrenches = buffer(request.rigid.bodyWrenches);
        id<MTLBuffer> sceneBodies = buffer(request.rigid.sceneBodies);
        id<MTLBuffer> rodNodes = buffer(request.rigid.rodNodes);
        id<MTLBuffer> rodInverseMasses =
            buffer(request.rigid.rodInverseMasses);
        id<MTLBuffer> resetMasks = buffer(request.resetMasks);
        id<MTLBuffer> worldStatuses = buffer(request.environmentStatuses);
        const std::uint64_t requiredWorldStatusBytes =
            static_cast<std::uint64_t>(state.dispatch.environmentCount) *
            sizeof(MRMetalWorldStatusGPU);
        if (worldStatuses == nil || worldStatuses.device == nil ||
            worldStatuses.device.registryID != state.device.registryID ||
            worldStatuses.gpuAddress == 0u ||
            requiredWorldStatusBytes >
                static_cast<std::uint64_t>(worldStatuses.length)) {
            diagnostics.message =
                "borrowed MetalWorld status has wrong device provenance or byte capacity";
            return diagnostics;
        }

        NMMatterDispatchGPU frameDispatch = state.dispatch;
        frameDispatch.gravityAndTimestep.w = frameTimestep;
        NMBridgeDispatchGPU bridge{};
        bridge.environmentCount = state.dispatch.environmentCount;
        bridge.rigidProxyCount = state.dispatch.rigidProxyCount;
        bridge.currentBodyCount = request.rigid.currentBodyCount;
        bridge.currentBodyStride = request.rigid.currentBodyStride;
        bridge.bodyWrenchCount = request.rigid.bodyWrenchCount;
        bridge.sceneBodyCount = request.rigid.sceneBodyCount;
        bridge.bodyWrenchStride = request.rigid.bodyWrenchStride;
        bridge.sceneStride = request.rigid.sceneStride;
        bridge.reactionStride = state.dispatch.rigidProxyCount;
        bridge.reactionBodyCount = state.reactionBodyCount;
        bridge.rodNodeCount = request.rigid.rodNodeCount;
        bridge.rodNodeStride = request.rigid.rodNodeStride;
        bridge.time = {
            1.0f / frameTimestep,
            frameTimestep,
            0.0f,
            0.0f,
        };
        const std::uint32_t coupledArticulatedNv =
            state.requiresCoupledCandidate ? request.rigid.vStride : 0u;

        const auto dispatchThreads = [&](
            const char* name,
            const NSUInteger count,
            const auto& bind
        ) {
            if (count == 0u) {
                return;
            }
            ++diagnostics.threadDispatchCount;
            diagnostics.requestedThreadCount +=
                static_cast<std::uint64_t>(count);
            id<MTLComputePipelineState> pipeline = state.pipeline(name);
            [encoder setComputePipelineState:pipeline];
            bind();
            const NSUInteger width = std::min<NSUInteger>(
                256u,
                pipeline.maxTotalThreadsPerThreadgroup
            );
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        };
        const auto dispatchGroups32 = [&](
            const char* name,
            const NSUInteger groupCount,
            const auto& bind
        ) {
            if (groupCount == 0u) {
                return;
            }
            ++diagnostics.simdgroupDispatchCount;
            diagnostics.requestedThreadgroupCount +=
                static_cast<std::uint64_t>(groupCount);
            [encoder setComputePipelineState:state.pipeline(name)];
            bind();
            [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        };
        const auto dispatchIndirect = [&](
            const char* name,
            id<MTLBuffer> indirect,
            const NSUInteger width,
            const auto& bind
        ) {
            id<MTLComputePipelineState> pipeline = state.pipeline(name);
            if (pipeline == nil || indirect == nil ||
                width == 0u ||
                width > pipeline.maxTotalThreadsPerThreadgroup) {
                return false;
            }
            ++diagnostics.indirectDispatchCount;
            [encoder setComputePipelineState:pipeline];
            bind();
            [encoder
                dispatchThreadgroupsWithIndirectBuffer:indirect
                indirectBufferOffset:0u
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
            return true;
        };
        const auto setDispatch = [&]() {
            [encoder setBytes:&frameDispatch
                       length:sizeof(frameDispatch)
                      atIndex:0u];
        };
        const NSUInteger environments = state.dispatch.environmentCount;
        const NSUInteger objects = state.dispatch.objectCount;
        const NSUInteger particleTotal =
            environments * state.dispatch.particleCount;
        const NSUInteger mpmActiveNodeTotal =
            environments * state.dispatch.mpmActiveNodeCapacity;
        const NSUInteger femNodeTotal =
            environments * state.dispatch.femNodeCount;
        const NSUInteger continuumNodeTotal = environments *
            (state.dispatch.gridNodeCount + state.dispatch.femNodeCount);
        const NSUInteger surfacePrimitiveTotal = environments *
            (4u * state.dispatch.tetrahedronCount +
             state.dispatch.gridNodeCount);
        const NSUInteger tetrahedronTotal =
            environments * state.dispatch.tetrahedronCount;
        const NSUInteger femTransactionalTotal = environments * std::max(
            state.dispatch.femNodeCount,
            state.dispatch.tetrahedronCount
        );
        const NSUInteger pairTotal =
            environments * state.dispatch.contactPairCount;
        const NSUInteger femHumanAttachmentTotal =
            environments * state.dispatch.femHumanAttachmentCount;
        const NSUInteger deformableContactHistoryTotal =
            environments * state.dispatch.deformableContactCapacity;
        const NSUInteger proxyTotal =
            environments * state.dispatch.rigidProxyCount;
        const NSUInteger bodyWrenchTotal =
            environments * state.reactionBodyCount;
        const NSUInteger objectTotal = environments * objects;
        const NSUInteger mixedTransactionalTotal = std::max(
            femNodeTotal, objectTotal
        );
        const NSUInteger topologyTransactionalTotal = environments * std::max({
            static_cast<NSUInteger>(state.dispatch.femNodeCount),
            static_cast<NSUInteger>(state.dispatch.tetrahedronCount),
            static_cast<NSUInteger>(state.dispatch.cohesiveFaceCount),
            static_cast<NSUInteger>(state.dispatch.punctureChannelCount),
            static_cast<NSUInteger>(state.dispatch.objectCount),
            static_cast<NSUInteger>(state.dispatch.tetrahedronCount) * 4u
        });

        const auto encodeFEMHumanAttachmentKinematics = [&]() -> bool {
            if (state.dispatch.femHumanAttachmentCount == 0u) return true;
            [encoder endEncoding];
            encoder = nil;
            const std::uint32_t pointCount =
                state.dispatch.femHumanAttachmentCount;
            const CoupledCandidateQuery query{
                .input = (__bridge void*)
                    state.coupledGeneralizedCandidate,
                .candidateQ = (__bridge void*)state.coupledCandidateQ,
                .candidateBodies = (__bridge void*)
                    state.coupledCandidateBodies,
                .pointQueries = (__bridge void*)
                    state.femHumanAttachmentPointQueries,
                .pointJacobians = (__bridge void*)
                    state.femHumanAttachmentPointJacobians,
                .operation =
                    CoupledCandidateOperation::candidateKinematics,
                .generalizedVectorStride =
                    state.dispatch.rigidGeneralizedCapacity,
                .candidateQStride = state.coupledQStride,
                .candidateBodyStride = state.coupledBodyStride,
                .pointCount = pointCount,
                .pointStride = pointCount,
                .pointJacobianStride =
                    state.dispatch.femHumanAttachmentPointJacobianStride,
            };
            if (!request.encodeCoupledCandidate(
                    request.coupledCandidateContext, query)) {
                diagnostics.message =
                    "Human candidate service rejected exact FEM attachment kinematics";
                return false;
            }
            encoder = [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                diagnostics.message =
                    "failed to resume Matter after FEM attachment kinematics";
                return false;
            }
            [encoder setLabel:@"Numi Matter Human/FEM attachment continuation"];
            dispatchThreads(
                "nm_fem_human_attachment_drive_candidate",
                femHumanAttachmentTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&request.rigid.currentBodyCount
                               length:sizeof(request.rigid.currentBodyCount)
                              atIndex:1u];
                    [encoder setBytes:&request.rigid.currentBodyStride
                               length:sizeof(request.rigid.currentBodyStride)
                              atIndex:2u];
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:3u];
                    [encoder setBuffer:state.femHumanAttachments
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.coupledCandidateBodies
                                 offset:0u atIndex:5u];
                    [encoder setBuffer:state.femHumanAttachmentPointJacobians
                                 offset:0u atIndex:6u];
                    [encoder setBuffer:state.coupledGeneralizedCandidate
                                 offset:0u atIndex:7u];
                    [encoder setBuffer:state.femAccepted
                                 offset:0u atIndex:8u];
                    [encoder setBuffer:state.femCandidate
                                 offset:0u atIndex:9u];
                    [encoder setBuffer:state.femNodeRanges
                                 offset:0u atIndex:10u];
                    [encoder setBuffer:state.statuses
                                 offset:0u atIndex:11u];
                }
            );
            return true;
        };
        const auto captureFEMHumanAttachmentResidual = [&]() {
            dispatchThreads(
                "nm_fem_human_attachment_capture_residual",
                femHumanAttachmentTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.femHumanAttachments
                                 offset:0u atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femNodeRanges
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                    [encoder setBuffer:state.femCandidate
                                 offset:0u atIndex:5u];
                    [encoder setBuffer:state.femResidual
                                 offset:0u atIndex:6u];
                    [encoder setBuffer:state.femPreconditioned
                                 offset:0u atIndex:7u];
                    [encoder setBuffer:state.femDirection
                                 offset:0u atIndex:8u];
                    [encoder
                        setBuffer:state.femHumanAttachmentResidualImpulses
                           offset:0u atIndex:9u];
                    [encoder setBuffer:state.femHumanAttachmentRawReactions
                                 offset:0u atIndex:10u];
                    [encoder setBuffer:state.statuses
                                 offset:0u atIndex:11u];
                }
            );
        };
        const auto scatterFEMHumanAttachmentResidual = [&]() {
            dispatchThreads(
                "nm_fem_human_attachment_scatter_residual",
                environments * state.dispatch.rigidGeneralizedCapacity,
                [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:1u];
                    [encoder setBuffer:state.femHumanAttachmentPointJacobians
                                 offset:0u atIndex:2u];
                    [encoder
                        setBuffer:state.femHumanAttachmentResidualImpulses
                           offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses
                                 offset:0u atIndex:5u];
                }
            );
        };

        if (request.phase == EncodePhase::postCommit) {
            dispatchThreads(
                "nm_reconcile_rigid_world_status",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:worldStatuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:2u];
                }
            );
            // MetalWorld has now published the realized free/articulated/DER
            // state.  It may differ from Matter's accepted primal candidate
            // after rigid jaw contact.  Re-project and certify the resulting
            // continuum/proxy geometry before either side retains this frame.
            dispatchThreads("nm_project_rigid_states", proxyTotal, [&] {
                setDispatch();
                [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
                [encoder setBuffer:currentBodies offset:0u atIndex:3u];
                [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                [encoder setBuffer:rodNodes offset:0u atIndex:5u];
                [encoder setBuffer:rodInverseMasses offset:0u atIndex:6u];
            });
            const std::uint32_t internalMicroticks =
                1u << state.dispatch.maximumRateExponent;
            std::uint64_t finalGeneration64 =
                (static_cast<std::uint64_t>(request.controlStep) *
                     request.physicsSubsteps +
                 request.physicsSubstep + 1u) * internalMicroticks;
            std::uint32_t finalMPMGeneration =
                static_cast<std::uint32_t>(finalGeneration64);
            if (finalMPMGeneration == 0u) finalMPMGeneration = 1u;
            dispatchThreads(
                "nm_contact_certify_post_commit_rigid",
                pairTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&finalMPMGeneration
                               length:sizeof(finalMPMGeneration)
                              atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:3u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:5u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:6u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:7u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:8u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:9u];
                    [encoder setBuffer:state.mpmNodeGenerations
                                 offset:0u atIndex:10u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:11u];
                    [encoder setBuffer:state.punctureChannelsAccepted
                                 offset:0u atIndex:12u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:13u];
                }
            );
            const std::uint32_t rigidWorldPhysicsSubstep =
                request.rigidWorldPhysicsSubstep == NM_INVALID_INDEX
                ? request.physicsSubstep
                : request.rigidWorldPhysicsSubstep;
            dispatchThreads(
                "nm_latch_matter_status_into_rigid_world",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder setBuffer:worldStatuses offset:0u atIndex:2u];
                    [encoder setBytes:&rigidWorldPhysicsSubstep
                               length:sizeof(rigidWorldPhysicsSubstep)
                              atIndex:3u];
                }
            );
            dispatchThreads("nm_mpm_rollback_frame", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:5u];
            });
            dispatchThreads("nm_fem_rollback_frame", femTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:5u];
            });
            dispatchThreads("nm_topology_rollback", topologyTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:5u];
                [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:7u];
                [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
                [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:9u];
                [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
                [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:11u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:12u];
                [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:13u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:14u];
                [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:15u];
            });
            dispatchThreads("nm_mixed_rollback", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
            });
            dispatchThreads("nm_learned_rollback", state.dispatch.learnedWeightCount, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:5u];
            });
            dispatchThreads("nm_contact_rollback_histories", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.contactHistoriesAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.contactHistoriesCheckpoint offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_contact_rollback_deformable_contact_histories",
                deformableContactHistoryTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableContactHistoriesAccepted
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableContactHistoriesCandidate
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.deformableContactHistoriesCheckpoint
                                 offset:0u atIndex:4u];
                });
            dispatchThreads("nm_scheduler_reconcile", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulerCheckpoint offset:0u atIndex:2u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                [encoder setBuffer:state.events offset:0u atIndex:4u];
            });
            if (request.runAdaptiveTransfer && state.hasAdaptive) {
                dispatchGroups32("nm_adaptive_measure", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:4u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                });
                if (request.rigidContactConstraints != nullptr &&
                    request.rigidContactStatuses != nullptr &&
                    request.rigidContactConstraintStride != 0u) {
                    const std::uint32_t contactConstraintStride =
                        request.rigidContactConstraintStride;
                    dispatchThreads(
                        "nm_adaptive_observe_rigid_contacts",
                        objectTotal,
                        [&] {
                            setDispatch();
                            [encoder setBuffer:state.objects offset:0u atIndex:1u];
                            [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
                            [encoder
                                setBuffer:(__bridge id<MTLBuffer>)request.rigidContactConstraints
                                offset:0u atIndex:3u];
                            [encoder
                                setBuffer:(__bridge id<MTLBuffer>)request.rigidContactStatuses
                                offset:0u atIndex:4u];
                            [encoder setBytes:&contactConstraintStride
                                       length:sizeof(contactConstraintStride)
                                      atIndex:5u];
                            [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                        }
                    );
                }
                dispatchThreads("nm_adaptive_decide", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:2u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:3u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:4u];
                });
                dispatchThreads("nm_adaptive_demote_to_rigid", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                    [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                });
                dispatchThreads("nm_adaptive_promote_mpm", particleTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:3u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                });
                dispatchThreads("nm_adaptive_promote_fem", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:5u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                });
                dispatchThreads("nm_adaptive_finish_promotion", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                });
            }

            if (state.hasAdaptive) {
                dispatchThreads(
                    "nm_publish_adaptive_rigid_ownership",
                    objectTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                        [encoder setBuffer:currentBodies offset:0u atIndex:5u];
                        [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                    }
                );
            }

            NMMicrostepGPU eventFrame{};
            eventFrame.controlStep = request.controlStep;
            eventFrame.microtick = request.physicsSubstep;
            eventFrame.microtickCount = request.physicsSubsteps;
            eventFrame.flags = state.captureEvents
                ? NM_MICROSTEP_CAPTURE_EVENTS
                : 0u;
            const std::uint64_t completedSubsteps =
                static_cast<std::uint64_t>(request.controlStep) *
                    request.physicsSubsteps +
                request.physicsSubstep;
            eventFrame.time = {
                frameTimestep,
                1.0f / frameTimestep,
                static_cast<float>(completedSubsteps) * frameTimestep,
                frameTimestep,
            };
            dispatchThreads("nm_scheduler_finalize", objectTotal, [&] {
                setDispatch();
                [encoder setBytes:&eventFrame length:sizeof(eventFrame) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                [encoder setBuffer:state.events offset:0u atIndex:6u];
            });
            dispatchThreads("nm_finalize_status", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.schedulers offset:0u atIndex:1u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:2u];
                [encoder setBuffer:state.statuses offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_fem_human_attachment_mask_reactions",
                femHumanAttachmentTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder
                        setBuffer:state.femHumanAttachmentResidualImpulses
                           offset:0u atIndex:2u];
                    [encoder setBuffer:state.femHumanAttachmentRawReactions
                                 offset:0u atIndex:3u];
                }
            );

            [encoder endEncoding];
            ownership->preDynamicsOpen = false;
            if (retainPreparedState) {
                ownership->transactionPolicyFingerprint = mixFingerprint(
                    ownership->transactionPolicyFingerprint,
                    0x4e5854504f535432ull); // "NXTPPOST2"
                ownership->transactionPolicyFingerprint = mixFingerprint(
                    ownership->transactionPolicyFingerprint,
                    request.runAdaptiveTransfer ? 1u : 0u);
                ownership->transactionPolicyFingerprint = mixFingerprint(
                    ownership->transactionPolicyFingerprint,
                    request.rigidContactConstraintStride);
                ownership->transactionPolicyFingerprint = mixFingerprint(
                    ownership->transactionPolicyFingerprint,
                    request.rigidWorldPhysicsSubstep);
                if (ownership->transactionPolicyFingerprint == 0u) {
                    ownership->transactionPolicyFingerprint =
                        14695981039346656037ull;
                }
                ownership->preparedStateOpen = true;
                ownership->preparedCommandBuffer = request.commandBuffer;
                ownership->preparedEnvironmentStatuses =
                    request.environmentStatuses;
                ownership->preparedEnvironmentStatusesGPUAddress =
                    worldStatuses.gpuAddress;
                ownership->applyEncoded = false;
                ownership->restoreRequired = false;
                ownership->preparedCommandCompleted = false;
                ownership->preparedCommandFailed = false;
                ownership->terminalNoTouch = false;
                ownership->applyCommandCompleted = false;
            } else {
                ownership->acceptedStateProofEncoded = false;
                ownership->acceptedStateProofEligible = false;
                ownership->transactionPolicyFingerprint = 0u;
                ownership->preparedStateRequested = false;
            }
            diagnostics.encoded = true;
            diagnostics.residentBytes = state.residentBytes;
            diagnostics.device = nsString(state.device.name);
            diagnostics.message = retainPreparedState
                ? "Numi Matter prepared-state reconciliation encoded"
                : "Numi Matter post-commit reconciliation encoded";
            return diagnostics;
        }

        if (firstPrePass && request.resetMaskStepStride != 0u) {
            const std::uint32_t stateWidth = std::max({
                state.dispatch.particleCount,
                state.dispatch.femNodeCount,
                state.dispatch.tetrahedronCount,
                state.dispatch.objectCount,
                state.dispatch.parameterCount,
                state.dispatch.rigidProxyCount,
            });
            NMResetPassGPU reset{};
            reset.controlStep = request.controlStep;
            reset.resetMaskStepStride = request.resetMaskStepStride;
            reset.stateWidth = stateWidth;
            reset.flags = NM_RESET_ENABLED | NM_RESET_PARAMETERS;
            dispatchThreads(
                "nm_apply_episode_resets",
                environments * stateWidth,
                [&] {
                    setDispatch();
                    [encoder setBytes:&reset length:sizeof(reset) atIndex:1u];
                    [encoder setBuffer:resetMasks offset:0u atIndex:2u];
                    [encoder setBuffer:state.particleDefaults offset:0u atIndex:3u];
                    [encoder setBuffer:state.femDefaults offset:0u atIndex:4u];
                    [encoder setBuffer:state.adaptiveDefaults offset:0u atIndex:5u];
                    [encoder setBuffer:state.schedulerDefaults offset:0u atIndex:6u];
                    [encoder setBuffer:state.parameterDefaults offset:0u atIndex:7u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:8u];
                    [encoder setBuffer:state.particleCandidate offset:0u atIndex:9u];
                    [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:10u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:11u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:12u];
                    [encoder setBuffer:state.femCheckpoint offset:0u atIndex:13u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:14u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:15u];
                    [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:16u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:17u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:18u];
                    [encoder setBuffer:state.particleMaterialStateDefaults offset:0u atIndex:19u];
                    [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:20u];
                    [encoder setBuffer:state.particleMaterialStateCandidate offset:0u atIndex:21u];
                    [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:22u];
                    [encoder setBuffer:state.femMaterialStateDefaults offset:0u atIndex:23u];
                    [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:24u];
                    [encoder setBuffer:state.femMaterialStateCandidate offset:0u atIndex:25u];
                    [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:26u];
                }
            );
        }

        if (firstPrePass) {
            // MetalWorld's rollback authority is the control-step checkpoint,
            // not the current physics substep. Keep this snapshot immutable
            // until the final post-commit reconciliation has completed.
            dispatchThreads("nm_checkpoint_scheduler", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.schedulers offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulerCheckpoint offset:0u atIndex:2u];
            });
        }

        dispatchThreads("nm_prepare_status", environments, [&] {
            setDispatch();
            [encoder setBuffer:worldStatuses offset:0u atIndex:1u];
            [encoder setBuffer:state.statuses offset:0u atIndex:2u];
        });
        if (firstPrePass && state.captureEvents) {
            dispatchThreads(
                "nm_prepare_events",
                environments * state.dispatch.eventStride,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.events offset:0u atIndex:1u];
                }
            );
        }
        dispatchThreads("nm_prepare_reactions", proxyTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.frameReactions offset:0u atIndex:1u];
        });
        dispatchThreads("nm_prepare_scheduler", objectTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.schedulers offset:0u atIndex:1u];
            [encoder setBuffer:state.schedulerPrevious offset:0u atIndex:2u];
        });
        if (state.hasAdaptive) {
            dispatchThreads(
                "nm_publish_adaptive_rigid_ownership",
                objectTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:3u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:4u];
                    [encoder setBuffer:currentBodies offset:0u atIndex:5u];
                    [encoder setBuffer:sceneBodies offset:0u atIndex:6u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                }
            );
        }

        if (firstPrePass && state.identificationDistributionCount != 0u) {
            NMIdentificationPassGPU identification{};
            identification.candidateCount =
                state.dispatch.identificationCandidateCount;
            identification.distributionCount =
                state.identificationDistributionCount;
            identification.generation =
                ownership->identificationGeneration;
            identification.seedLo = static_cast<std::uint32_t>(request.seed);
            identification.seedHi =
                static_cast<std::uint32_t>(request.seed >> 32u);
            if (request.runIdentification &&
                ownership->identificationGeneration != 0u) {
                dispatchThreads(
                    "nm_identification_update",
                    identification.distributionCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&identification
                                   length:sizeof(identification)
                                  atIndex:1u];
                        [encoder setBuffer:state.identificationDistributions offset:0u atIndex:2u];
                        [encoder setBuffer:state.identificationCandidates offset:0u atIndex:3u];
                        [encoder setBuffer:state.identificationLosses offset:0u atIndex:4u];
                    }
                );
            }
            dispatchThreads(
                "nm_publish_identification_means",
                environments * identification.distributionCount,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.identificationDistributions offset:0u atIndex:1u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:2u];
                    [encoder setBytes:&identification.distributionCount
                               length:sizeof(identification.distributionCount)
                              atIndex:3u];
                }
            );
            if (request.runIdentification &&
                identification.candidateCount != 0u) {
                dispatchThreads(
                    "nm_identification_sample",
                    static_cast<NSUInteger>(identification.candidateCount) *
                        identification.distributionCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&identification
                                   length:sizeof(identification)
                                  atIndex:1u];
                        [encoder setBuffer:state.identificationDistributions offset:0u atIndex:2u];
                        [encoder setBuffer:state.identificationCandidates offset:0u atIndex:3u];
                        [encoder setBuffer:state.environmentParameters offset:0u atIndex:4u];
                    }
                );
                ownership->identificationCheckpoint =
                    ownership->identificationGeneration;
                ownership->identificationAdvanced = true;
                ++ownership->identificationGeneration;
            }
        }

        if (firstPrePass) {
            dispatchThreads("nm_mpm_checkpoint", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:2u];
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:4u];
            });
            dispatchThreads("nm_fem_checkpoint", femTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femCheckpoint offset:0u atIndex:2u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:4u];
            });
            dispatchThreads("nm_topology_checkpoint", topologyTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:6u];
                [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:9u];
                [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:10u];
                [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:11u];
                [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:12u];
                [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:13u];
                [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:14u];
                [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:15u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:16u];
                [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:17u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:18u];
                [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:19u];
            });
            dispatchThreads("nm_mixed_checkpoint", mixedTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.solverCertificates offset:0u atIndex:4u];
            });
            dispatchThreads("nm_learned_checkpoint", state.dispatch.learnedWeightCount, [&] {
                setDispatch();
                [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.learnedRevisionCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:6u];
            });
            dispatchThreads("nm_contact_checkpoint_histories", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.contactHistoriesAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.contactHistoriesCheckpoint offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_contact_checkpoint_deformable_contact_histories",
                deformableContactHistoryTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.deformableContactHistoriesAccepted
                                 offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableContactHistoriesCandidate
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableContactHistoriesCheckpoint
                                 offset:0u atIndex:3u];
                });
            if (request.learnedWeightUpdate != nullptr) {
                id<MTLBuffer> learnedUpdate =
                    (__bridge id<MTLBuffer>)request.learnedWeightUpdate;
                dispatchThreads("nm_learned_stage_weights", state.dispatch.learnedWeightCount, [&] {
                    setDispatch();
                    [encoder setBuffer:learnedUpdate offset:0u atIndex:1u];
                    [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                    [encoder setBuffer:state.learnedLayers offset:0u atIndex:4u];
                    [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:5u];
                    [encoder setBuffer:state.learnedRevisionCandidate offset:0u atIndex:6u];
                    [encoder setBytes:&request.learnedWeightCount
                               length:sizeof(request.learnedWeightCount)
                              atIndex:7u];
                    [encoder setBytes:&request.learnedWeightRevision
                               length:sizeof(request.learnedWeightRevision)
                              atIndex:8u];
                });
            }
        }
        dispatchThreads("nm_project_rigid_states", proxyTotal, [&] {
            setDispatch();
            [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
            [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
            [encoder setBuffer:currentBodies offset:0u atIndex:3u];
            [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
            [encoder setBuffer:rodNodes offset:0u atIndex:5u];
            [encoder setBuffer:rodInverseMasses offset:0u atIndex:6u];
        });

        const std::uint32_t microtickCount =
            1u << state.dispatch.maximumRateExponent;
        id<MTLBuffer> borrowedMutations = request.mutationCommands == nullptr
            ? state.dummy
            : (__bridge id<MTLBuffer>)request.mutationCommands;
        for (std::uint32_t microtick = 0u;
             microtick < microtickCount;
             ++microtick) {
            NMMicrostepGPU micro{};
            micro.controlStep = request.controlStep;
            micro.microtick = microtick;
            micro.microtickCount = microtickCount;
            micro.seedLo = static_cast<std::uint32_t>(request.seed);
            micro.seedHi = static_cast<std::uint32_t>(request.seed >> 32u);
            micro.flags = 0u;
            const std::uint64_t generation64 =
                (static_cast<std::uint64_t>(request.controlStep) *
                     request.physicsSubsteps +
                 request.physicsSubstep) *
                    microtickCount +
                microtick + 1u;
            micro.reserved = static_cast<std::uint32_t>(generation64);
            if (micro.reserved == 0u) {
                micro.reserved = 1u;
            }
            const float globalDt =
                frameTimestep / float(microtickCount);
            micro.time = {
                globalDt,
                1.0f / globalDt,
                float(request.physicsSubstep) * frameTimestep +
                float(microtick) * globalDt,
                0.0f,
            };

            const auto encodeTopologyRebuild = [&](id<MTLBuffer> nodes) {
                dispatchThreads("nm_topology_rebuild_clear", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:nodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.topologyIncidenceCursors offset:0u atIndex:3u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:4u];
                });
                dispatchThreads("nm_topology_rebuild_count_mass", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.materials offset:0u atIndex:1u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:nodes offset:0u atIndex:3u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                });
                dispatchGroups32("nm_topology_rebuild_prefix", environments, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.topologyIncidenceCursors offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                });
                dispatchThreads("nm_topology_rebuild_scatter", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.topologyIncidenceCursors offset:0u atIndex:3u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                });
                dispatchThreads("nm_topology_rebuild_finalize", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:nodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                });
            };
            const auto encodeTopologyCertificate = [&](
                id<MTLBuffer> candidateNodes,
                id<MTLBuffer> candidateFields
            ) {
                dispatchThreads(
                    "nm_topology_conserve_transaction", objectTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.objects offset:0u atIndex:1u];
                        [encoder setBuffer:state.femCheckpoint
                                     offset:0u atIndex:2u];
                        [encoder setBuffer:candidateNodes offset:0u atIndex:3u];
                        [encoder setBuffer:state.femFieldsCheckpoint
                                     offset:0u atIndex:4u];
                        [encoder setBuffer:candidateFields offset:0u atIndex:5u];
                        [encoder setBuffer:state.topologyStatesCheckpoint
                                     offset:0u atIndex:6u];
                        [encoder setBuffer:state.topologyStatesCandidate
                                     offset:0u atIndex:7u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                    });
                dispatchThreads("nm_topology_certify_transaction", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femCheckpoint offset:0u atIndex:2u];
                    [encoder setBuffer:candidateNodes offset:0u atIndex:3u];
                    [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:4u];
                    [encoder setBuffer:candidateFields offset:0u atIndex:5u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:6u];
                    [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:7u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:8u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:9u];
                });
            };

            if (firstPrePass && microtick == 0u) {
                dispatchThreads("nm_topology_detect_puncture", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&request.controlStep
                               length:sizeof(request.controlStep) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                    [encoder setBuffer:state.femTetrahedraAccepted offset:0u
                               atIndex:7u];
                    [encoder setBuffer:state.punctureChannelsAccepted offset:0u
                               atIndex:8u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:9u];
                    [encoder setBuffer:state.contactHistoriesAccepted offset:0u
                               atIndex:10u];
                    [encoder setBuffer:state.contactSamples offset:0u
                               atIndex:11u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u
                               atIndex:12u];
                });
                const std::uint32_t triggeredMutationCount =
                    state.dispatch.objectCount;
                const std::uint32_t cookedMutationCount =
                    state.dispatch.mutationCommandCount;
                dispatchThreads("nm_topology_execute_transaction", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.materials offset:0u atIndex:4u];
                    [encoder setBuffer:state.cookedMutationCommands offset:0u atIndex:5u];
                    [encoder setBuffer:borrowedMutations offset:0u atIndex:6u];
                    [encoder setBytes:&request.mutationCommandCount
                               length:sizeof(request.mutationCommandCount) atIndex:7u];
                    [encoder setBytes:&request.mutationCommandStride
                               length:sizeof(request.mutationCommandStride) atIndex:8u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:9u];
                    [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:10u];
                    [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:11u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:12u];
                    [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:13u];
                    [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:14u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:15u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:16u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:17u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:18u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u
                               atIndex:19u];
                    [encoder setBytes:&triggeredMutationCount
                               length:sizeof(triggeredMutationCount) atIndex:20u];
                    [encoder setBytes:&cookedMutationCount
                               length:sizeof(cookedMutationCount) atIndex:21u];
                    [encoder setBuffer:state.femMaterialStateAccepted
                                 offset:0u atIndex:22u];
                    [encoder setBuffer:state.topologyStatesCheckpoint
                                 offset:0u atIndex:23u];
                });
                encodeTopologyRebuild(state.femAccepted);
                encodeTopologyCertificate(
                    state.femAccepted, state.femFieldsAccepted);
            }

            dispatchThreads("nm_mixed_prepare_microstep", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:2u];
            });

            const NSUInteger sparsePrepareCount = std::max<NSUInteger>(
                environments * state.dispatch.mpmBlockCount,
                environments
            );
            dispatchThreads("nm_mpm_prepare_sparse", sparsePrepareCount, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmBlockCursors offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmBlockActiveLocalOffsets offset:0u atIndex:5u];
                [encoder setBuffer:state.mpmEnvironmentActiveCounts offset:0u atIndex:6u];
                [encoder setBuffer:state.mpmEnvironmentActiveOffsets offset:0u atIndex:7u];
                [encoder setBuffer:state.mpmActiveDispatch offset:0u atIndex:8u];
            });
            dispatchThreads("nm_mpm_classify_particles", particleTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmGrids offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:4u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:6u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:7u];
                [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                [encoder setBuffer:state.mpmParticleKeys offset:0u atIndex:9u];
                [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:10u];
                [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:11u];
            });
            dispatchGroups32("nm_mpm_scan_environment_blocks", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmBlockActiveLocalOffsets offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmEnvironmentActiveCounts offset:0u atIndex:5u];
                [encoder setBuffer:state.statuses offset:0u atIndex:6u];
            });
            dispatchGroups32("nm_mpm_scan_environment_counts", 1u, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmEnvironmentActiveCounts offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmEnvironmentActiveOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmActiveDispatch offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_mpm_scatter_active_blocks",
                environments * state.dispatch.mpmBlockCount,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.mpmBlockActiveFlags offset:0u atIndex:1u];
                    [encoder setBuffer:state.mpmBlockActiveLocalOffsets offset:0u atIndex:2u];
                    [encoder setBuffer:state.mpmEnvironmentActiveOffsets offset:0u atIndex:3u];
                    [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:4u];
                    [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:5u];
                    [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:6u];
                }
            );
            dispatchThreads("nm_mpm_scatter_particles", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mpmParticleKeys offset:0u atIndex:1u];
                [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmBlockCursors offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:4u];
            });
            if (!dispatchIndirect(
                    "nm_mpm_sort_block_particles",
                    state.mpmActiveDispatch,
                    NM_MPM_MAX_PARTICLES_PER_BLOCK,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:1u];
                        [encoder setBuffer:state.mpmParticleKeys offset:0u atIndex:2u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:3u];
                    }
                ) ||
                !dispatchIndirect(
                    "nm_mpm_p2g",
                    state.mpmActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                        [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                        [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                        [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                        [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:8u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:9u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:10u];
                        [encoder setBuffer:state.mpmBlocks offset:0u atIndex:11u];
                        [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:12u];
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:13u];
                        [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:14u];
                        [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:15u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:16u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:17u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:18u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:19u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:20u];
                    }
                )) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                diagnostics.message =
                    "failed to encode sparse MPM indirect work";
                return diagnostics;
            }
            dispatchGroups32("nm_mpm_compact_active_nodes", environments, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:3u];
                [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:5u];
                [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:6u];
                [encoder setBuffer:state.statuses offset:0u atIndex:7u];
            });
            dispatchThreads("nm_fem_prepare_candidate", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
            });
            if (hasFEMKinematicTargets != 0u) {
                dispatchThreads("nm_fem_apply_kinematic_targets", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                    [encoder setBuffer:femKinematicTargets offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                });
            }
            const NSUInteger rigidCandidateTotal = environments *
                state.dispatch.rigidGeneralizedCapacity;
            dispatchThreads(
                "nm_rigid_initialize_feasible_candidate",
                rigidCandidateTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:1u];
                    [encoder setBytes:&bridge
                               length:sizeof(bridge)
                              atIndex:2u];
                    [encoder setBuffer:state.rigidProxies
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:currentBodies offset:0u atIndex:4u];
                    [encoder setBuffer:buffer(request.rigid.v)
                                 offset:0u atIndex:5u];
                    [encoder setBuffer:state.coupledGeneralizedCandidate
                                 offset:0u atIndex:6u];
                });
            dispatchThreads("nm_contact_clear_samples", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.contactSamples offset:0u atIndex:1u];
                [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:2u];
            });
            // Contact is a block of the nonlinear variational residual, not a
            // post-FEM correction. Re-evaluate candidate geometry and stream
            // inverse-ABA columns on every Newton candidate, then accumulate
            // the primal barrier gradient before the fused FGMRES solve. The
            // callback only encodes into the borrowed command buffer; it
            // never commits, waits, or opens another queue.
            const auto encodeCoupledPrimalContact = [&] (
                const bool certify,
                id<MTLBuffer> histories,
                id<MTLBuffer> deformableContactHistories
            ) -> bool {
                if (state.requiresCoupledCandidate) {
                    [encoder endEncoding];
                    encoder = nil;
                    const CoupledCandidateQuery kinematicsQuery{
                        .input = (__bridge void*)
                            state.coupledGeneralizedCandidate,
                        .candidateQ = (__bridge void*)state.coupledCandidateQ,
                        .candidateBodies = (__bridge void*)
                            state.coupledCandidateBodies,
                        .operation =
                            CoupledCandidateOperation::candidateKinematics,
                        .generalizedVectorStride =
                            state.dispatch.rigidGeneralizedCapacity,
                        .candidateQStride = state.coupledQStride,
                        .candidateBodyStride = state.coupledBodyStride,
                    };
                    if (!request.encodeCoupledCandidate(
                            request.coupledCandidateContext,
                            kinematicsQuery)) {
                        diagnostics.message =
                            "MetalWorld failed to materialize Matter's articulated Newton candidate";
                        return false;
                    }
                    encoder = [commandBuffer computeCommandEncoder];
                    if (encoder == nil) {
                        diagnostics.message =
                            "failed to resume Matter after candidate kinematics";
                        return false;
                    }
                    [encoder setLabel:@"Numi Matter candidate-contact continuation"];
                    dispatchThreads("nm_project_rigid_states", proxyTotal, [&] {
                        setDispatch();
                        [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                        [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
                        [encoder setBuffer:state.coupledCandidateBodies
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.rigidStates offset:0u atIndex:4u];
                        [encoder setBuffer:rodNodes offset:0u atIndex:5u];
                        [encoder setBuffer:rodInverseMasses offset:0u atIndex:6u];
                    });
                }
                dispatchThreads(
                    "nm_project_primal_free_rigid_candidate",
                    proxyTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                        [encoder setBytes:&coupledArticulatedNv
                                   length:sizeof(coupledArticulatedNv)
                                  atIndex:2u];
                        [encoder setBuffer:state.rigidProxies
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:currentBodies offset:0u atIndex:4u];
                        [encoder setBuffer:state.coupledGeneralizedCandidate
                                   offset:0u atIndex:5u];
                        [encoder setBuffer:state.rigidStates
                                   offset:0u atIndex:6u];
                        [encoder setBuffer:state.statuses
                                   offset:0u atIndex:7u];
                    });
                dispatchThreads(
                    "nm_contact_build_surface_primitives",
                    surfacePrimitiveTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:5u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:6u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:7u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:8u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:9u];
                        [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:10u];
                        [encoder setBuffer:state.femNodeRanges offset:0u atIndex:11u];
                        [encoder setBuffer:state.continuumSurfacePrimitives offset:0u atIndex:12u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_sort_surface_primitives",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.continuumSurfacePrimitives offset:0u atIndex:1u];
                        [encoder setBuffer:state.femSurfaceSortKeysA offset:0u atIndex:2u];
                        [encoder setBuffer:state.femSurfaceSortKeysB offset:0u atIndex:3u];
                        [encoder setBuffer:state.femSurfaceSortIndicesA offset:0u atIndex:4u];
                        [encoder setBuffer:state.femSurfaceSortIndicesB offset:0u atIndex:5u];
                        [encoder setBuffer:state.femSurfaceActiveCounts offset:0u atIndex:6u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_build_deformable_candidates",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.continuumSurfacePrimitives offset:0u atIndex:1u];
                        [encoder setBuffer:state.femSurfaceSortIndicesA offset:0u atIndex:2u];
                        [encoder setBuffer:state.femSurfaceActiveCounts offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactCandidates offset:0u atIndex:4u];
                        [encoder setBuffer:state.deformableContactCandidateCounts offset:0u atIndex:5u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                        [encoder setBuffer:state.femTopologyNodesCandidate
                                     offset:0u atIndex:7u];
                        [encoder setBuffer:state.objects offset:0u atIndex:8u];
                    }
                );
                dispatchThreads(
                    "nm_contact_narrowphase_deformable",
                    environments * state.dispatch.deformableContactCapacity,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:5u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:6u];
                        [encoder setBuffer:state.continuumSurfacePrimitives offset:0u atIndex:7u];
                        [encoder setBuffer:state.deformableContactCandidates offset:0u atIndex:8u];
                        [encoder setBuffer:state.deformableContactCandidateCounts offset:0u atIndex:9u];
                        [encoder setBuffer:deformableContactHistories offset:0u atIndex:10u];
                        [encoder setBuffer:state.deformableContacts offset:0u atIndex:11u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:12u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_compact_deformable",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:3u];
                    }
                );
                dispatchGroups32(
                    "nm_contact_scan_deformable_active_counts",
                    1u,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveOffsets
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveDispatch
                                   offset:0u atIndex:3u];
                    });
                dispatchThreads(
                    "nm_contact_scatter_deformable_active_work",
                    environments * state.dispatch.deformableContactCapacity,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveOffsets
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactGlobalActiveIndices
                                   offset:0u atIndex:4u];
                    });
                dispatchThreads(
                    "nm_contact_count_deformable_node_incidence",
                    continuumNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:4u];
                    });
                dispatchGroups32(
                    "nm_contact_scan_deformable_node_incidence",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.statuses
                                   offset:0u atIndex:2u];
                    });
                dispatchThreads(
                    "nm_contact_scatter_deformable_node_incidence",
                    continuumNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactActiveIndices
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactActiveCounts
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:4u];
                        [encoder setBuffer:state.deformableContactNodeIncidence
                                   offset:0u atIndex:5u];
                    });
                dispatchThreads("nm_contact_evaluate", pairTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.materials offset:0u atIndex:3u];
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:4u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:5u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:6u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:7u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:8u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:11u];
                    [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:12u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                    [encoder setBuffer:histories offset:0u atIndex:14u];
                    [encoder setBytes:&bridge length:sizeof(bridge) atIndex:15u];
                    [encoder setBuffer:currentBodies offset:0u atIndex:16u];
                    [encoder setBuffer:state.punctureChannelsCandidate
                                 offset:0u atIndex:17u];
                });
                dispatchGroups32("nm_contact_compact_active", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&state.contactActiveCapacity
                               length:sizeof(state.contactActiveCapacity)
                              atIndex:1u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:2u];
                    [encoder setBuffer:state.contactActivePairs offset:0u atIndex:3u];
                    [encoder setBuffer:state.contactActiveSlotsByPair offset:0u atIndex:4u];
                    [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:5u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                });
                if (state.requiresCoupledCandidate) {
                    dispatchThreads(
                        "nm_contact_prepare_articulated_queries",
                        environments * state.contactActiveCapacity,
                        [&] {
                            setDispatch();
                            [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                            [encoder setBytes:&request.articulationRootBody
                                       length:sizeof(request.articulationRootBody)
                                      atIndex:2u];
                            [encoder setBytes:&state.contactActiveCapacity
                                       length:sizeof(state.contactActiveCapacity)
                                      atIndex:3u];
                            [encoder setBuffer:state.contactActivePairs offset:0u atIndex:4u];
                            [encoder setBuffer:state.contactActiveCounts offset:0u atIndex:5u];
                            [encoder setBuffer:state.rigidProxies offset:0u atIndex:6u];
                            [encoder setBuffer:state.coupledCandidateBodies
                                         offset:0u atIndex:7u];
                            [encoder setBuffer:state.contactPairs offset:0u atIndex:8u];
                            [encoder setBuffer:state.contactSamples offset:0u atIndex:9u];
                            [encoder setBuffer:state.articulatedPointQueries offset:0u atIndex:10u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:11u];
                        }
                    );
                }
                if (state.requiresCoupledCandidate) {
                    [encoder endEncoding];
                    encoder = nil;
                    const CoupledCandidateQuery candidateJacobianQuery{
                        .input = (__bridge void*)
                            state.coupledGeneralizedCandidate,
                        .candidateQ = (__bridge void*)state.coupledCandidateQ,
                        .candidateBodies = (__bridge void*)
                            state.coupledCandidateBodies,
                        .pointQueries = (__bridge void*)
                            state.articulatedPointQueries,
                        .pointJacobians = (__bridge void*)
                            state.coupledPointJacobians,
                        .operation =
                            CoupledCandidateOperation::candidateKinematics,
                        .generalizedVectorStride =
                            state.dispatch.rigidGeneralizedCapacity,
                        .candidateQStride = state.coupledQStride,
                        .candidateBodyStride = state.coupledBodyStride,
                        .pointCount = state.contactActiveCapacity,
                        .pointStride = state.contactActiveCapacity,
                        .pointJacobianStride =
                            state.contactActiveCapacity * 3u *
                                state.dispatch.rigidGeneralizedCapacity,
                    };
                    if (!request.encodeCoupledCandidate(
                            request.coupledCandidateContext,
                            candidateJacobianQuery)) {
                        diagnostics.message =
                            "MetalWorld failed to encode candidate point Jacobians";
                        return false;
                    }
                    id<MTLBlitCommandEncoder> clearCoupledMass =
                        [commandBuffer blitCommandEncoder];
                    if (clearCoupledMass == nil) {
                        diagnostics.message =
                            "failed to clear coupled candidate mass residual";
                        return false;
                    }
                    const NSUInteger coupledScalarBytes =
                        static_cast<NSUInteger>(environments) *
                        state.dispatch.rigidGeneralizedCapacity *
                        sizeof(float);
                    [clearCoupledMass
                        fillBuffer:state.coupledGeneralizedOutput
                             range:NSMakeRange(0u, coupledScalarBytes)
                             value:0u];
                    [clearCoupledMass endEncoding];
                    const CoupledCandidateQuery candidateMassQuery{
                        .input = (__bridge void*)
                            state.coupledGeneralizedCandidate,
                        .output = (__bridge void*)
                            state.coupledGeneralizedOutput,
                        .candidateQ = (__bridge void*)state.coupledCandidateQ,
                        .operation = CoupledCandidateOperation::massAction,
                        .generalizedVectorStride =
                            state.dispatch.rigidGeneralizedCapacity,
                        .candidateQStride = state.coupledQStride,
                    };
                    if (!request.encodeCoupledCandidate(
                            request.coupledCandidateContext,
                            candidateMassQuery)) {
                        diagnostics.message =
                            "MetalWorld failed to encode candidate mass residual";
                        return false;
                    }
                    encoder = [commandBuffer computeCommandEncoder];
                    if (encoder == nil) {
                        diagnostics.message =
                            "failed to resume Matter KKT solve after candidate mass action";
                        return false;
                    }
                    [encoder setLabel:@"Numi Matter monolithic KKT continuation"];
                }
                dispatchThreads("nm_contact_accumulate_fem_residual", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.contactNodeIncidence offset:0u atIndex:1u];
                    [encoder setBuffer:state.contactNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                });
                dispatchThreads(
                    "nm_contact_accumulate_deformable_fem_residual",
                    femNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.deformableContacts
                                   offset:0u atIndex:1u];
                        [encoder setBuffer:state.deformableContactNodeIncidence
                                   offset:0u atIndex:2u];
                        [encoder setBuffer:state.deformableContactNodeRanges
                                   offset:0u atIndex:3u];
                        [encoder setBuffer:state.femResidual
                                   offset:0u atIndex:4u];
                    });
                (void)certify;
                return true;
            };
            const NSUInteger rigidGeneralizedTotalForResidual = environments *
                state.dispatch.rigidGeneralizedCapacity;
            const std::uint32_t nonlinearBudgetOverride =
                state.newtonIterationBudgetOverride.load(
                    std::memory_order_acquire
                );
            const std::uint32_t nonlinearIterationBudget =
                nonlinearBudgetOverride == 0u
                    ? state.mixedSolverValue.nonlinearIterations.x
                    : nonlinearBudgetOverride;
            for (std::uint32_t nonlinearIteration = 0u;
                 nonlinearIteration < nonlinearIterationBudget;
                 ++nonlinearIteration) {
            micro.solverIteration = nonlinearIteration;
            if (!encodeFEMHumanAttachmentKinematics()) {
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
            // Reassemble the backward-Euler field residual at this Newton
            // candidate. These kernels no longer iterate or publish a field
            // solution; they provide the field residual and block diagonal to
            // the single generalized KKT solve below.
            dispatchThreads("nm_mixed_prepare_residual", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:3u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:5u];
            });
            dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:8u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
            });
            dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:7u];
            });
            dispatchThreads("nm_mixed_build_residual_diagonal", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:8u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:10u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:11u];
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedFieldPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedFieldInverseDiagonal offset:0u atIndex:14u];
            });
            // Mechanical pressure is a KKT unknown. Its normalized Schur
            // diagonal is applied only by nm_fgmres_precondition; there is no
            // standalone pressure correction or independently failing solve.
            dispatchThreads("nm_fem_internal_forces", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:10u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:11u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:14u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:15u];
                [encoder setBuffer:state.learnedMaterials offset:0u atIndex:16u];
                [encoder setBuffer:state.learnedLayers offset:0u atIndex:17u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:18u];
            });
            dispatchThreads("nm_fem_build_mechanical_residual", femNodeTotal, [&] {
                const std::uint32_t preserveSolution = 0u;
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:7u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:11u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.femPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:14u];
                [encoder setBuffer:state.femOperatorValue offset:0u atIndex:15u];
                [encoder setBuffer:state.statuses offset:0u atIndex:16u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:17u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:18u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:19u];
                [encoder setBytes:&preserveSolution
                           length:sizeof(preserveSolution) atIndex:20u];
                [encoder setBuffer:femExternalForces offset:0u atIndex:21u];
                [encoder setBytes:&hasFEMExternalForces
                           length:sizeof(hasFEMExternalForces) atIndex:22u];
                [encoder setBuffer:state.femConstraintReactions
                            offset:0u atIndex:23u];
            });
            dispatchThreads("nm_fgmres_clear_rigid_residual",
                rigidGeneralizedTotalForResidual, [&] {
                setDispatch();
                [encoder setBuffer:state.femResidual offset:0u atIndex:1u];
            });

            id<MTLBuffer> nonlinearHistories = nonlinearIteration == 0u
                ? state.contactHistoriesAccepted
                : state.contactHistoriesCandidate;
            id<MTLBuffer> nonlinearDeformableContactHistories =
                nonlinearIteration == 0u
                ? state.deformableContactHistoriesAccepted
                : state.deformableContactHistoriesCandidate;
            if (!encodeCoupledPrimalContact(
                    false,
                    nonlinearHistories,
                    nonlinearDeformableContactHistories)) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
            captureFEMHumanAttachmentResidual();
            dispatchThreads("nm_contact_subtract_rigid_inertia_residual",
                rigidGeneralizedTotalForResidual, [&] {
                setDispatch();
                [encoder setBytes:&coupledArticulatedNv
                           length:sizeof(coupledArticulatedNv) atIndex:1u];
                [encoder setBuffer:state.coupledGeneralizedOutput
                             offset:0u atIndex:2u];
                [encoder setBuffer:state.coupledGeneralizedCandidate
                             offset:0u atIndex:3u];
                [encoder setBuffer:state.primalContactArguments
                             offset:0u atIndex:4u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:5u];
            });
            dispatchThreads("nm_contact_accumulate_rigid_residual",
                rigidGeneralizedTotalForResidual, [&] {
                setDispatch();
                [encoder setBytes:&coupledArticulatedNv
                           length:sizeof(coupledArticulatedNv) atIndex:1u];
                [encoder setBuffer:state.primalContactArguments
                             offset:0u atIndex:2u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:3u];
                [encoder setBuffer:state.coupledPointJacobians
                             offset:0u atIndex:4u];
            });
            scatterFEMHumanAttachmentResidual();

            dispatchThreads(
                "nm_mpm_build_implicit_residual", mpmActiveNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:3u];
                [encoder setBuffer:state.contactNodeIncidence offset:0u atIndex:4u];
                [encoder setBuffer:state.contactNodeRanges offset:0u atIndex:5u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:6u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:7u];
                [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:8u];
                [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:9u];
                [encoder setBuffer:state.primalContactArguments offset:0u atIndex:10u];
            });
            if (!dispatchIndirect(
                    "nm_mpm_build_constitutive_residual",
                    state.mpmActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                        [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                        [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                        [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                        [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:8u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:9u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:10u];
                        [encoder setBuffer:state.mpmBlocks offset:0u atIndex:11u];
                        [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:12u];
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:13u];
                        [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:14u];
                        [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:15u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:16u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:17u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:18u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:19u];
                        [encoder setBuffer:state.femDirection offset:0u atIndex:20u];
                        [encoder setBuffer:state.femResidual offset:0u atIndex:21u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:22u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:23u];
                        [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:24u];
                    })) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                diagnostics.message =
                    "failed to encode implicit MPM residual work";
                return diagnostics;
            }
            dispatchThreads("nm_fgmres_import_field_residual", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:1u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:2u];
            });

            // Staged, GPU-resident FGMRES: expensive constitutive and contact
            // operator work is distributed over nodes/active contacts. The
            // SIMD32 object kernels only perform bounded reductions and the
            // tiny Hessenberg solve.
            const NSUInteger monolithicUnknownTotal =
                2u * femNodeTotal +
                mpmActiveNodeTotal +
                environments * state.dispatch.rigidGeneralizedCapacity;
            const NSUInteger rigidGeneralizedTotal = environments *
                state.dispatch.rigidGeneralizedCapacity;
            const NSUInteger vectorBytes =
                monolithicUnknownTotal * sizeof(nm_float4);
            [encoder useResource:state.contactPairs usage:MTLResourceUsageRead];
            [encoder useResource:state.contactSamples usage:MTLResourceUsageRead];
            [encoder useResource:state.contactNodeIncidence usage:MTLResourceUsageRead];
            [encoder useResource:state.contactNodeRanges usage:MTLResourceUsageRead];
            [encoder useResource:state.contactActivePairs usage:MTLResourceUsageRead];
            [encoder useResource:state.contactActiveSlotsByPair usage:MTLResourceUsageRead];
            [encoder useResource:state.contactActiveCounts usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContacts
                         usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [encoder useResource:state.deformableContactActiveIndices
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactActiveCounts
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactNodeIncidence
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactNodeRanges
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactGlobalActiveIndices
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.deformableContactActiveDispatch
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.mpmNodeToActive
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.mpmActiveNodeCounts
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.rigidProxies
                         usage:MTLResourceUsageRead];
            [encoder useResource:state.rigidStates
                         usage:MTLResourceUsageRead];
            const std::uint32_t restart = std::min(
                state.mixedSolverValue.nonlinearIterations.y,
                static_cast<std::uint32_t>(NM_MIXED_FGMRES_RESTART));
            const std::uint32_t budgetOverride =
                state.fgmresIterationBudgetOverride.load(
                    std::memory_order_acquire
                );
            const std::uint32_t linearIterationBudget = std::max(
                restart,
                budgetOverride == 0u
                    ? state.mixedSolverValue.nonlinearIterations.z
                    : budgetOverride
            );
            const std::uint32_t restartCycleCount =
                (linearIterationBudget + restart - 1u) / restart;
            const float nonlinearTolerance = std::max(
                state.mixedSolverValue.residualTolerances.x, 0.0f);
            const float forcingFloor = std::sqrt(nonlinearTolerance);
            const float scheduledForcing = std::ldexp(
                0.25f,
                -static_cast<int>(std::min(nonlinearIteration, 4u)));
            const float linearForcing = std::clamp(
                std::max(forcingFloor, scheduledForcing),
                nonlinearTolerance,
                0.25f);
            NMMicrostepGPU operatorMicro = micro;
            operatorMicro.flags |= NM_MICROSTEP_FGMRES_OPERATOR;
            for (std::uint32_t restartCycle = 0u;
                 restartCycle < restartCycleCount;
                 ++restartCycle) {
            const std::uint32_t iterationOffset = restartCycle * restart;
            const std::uint32_t columnsThisCycle = std::min(
                restart, linearIterationBudget - iterationOffset);
            dispatchGroups32("nm_fgmres_begin", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresBasis offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresHessenberg offset:0u atIndex:5u];
                [encoder setBuffer:state.fgmresRotations offset:0u atIndex:6u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:7u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:8u];
                [encoder setBuffer:state.statuses offset:0u atIndex:9u];
                [encoder setBytes:&micro length:sizeof(micro) atIndex:10u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:12u];
                [encoder setBuffer:state.primalContactArguments offset:0u atIndex:13u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:14u];
                [encoder setBytes:&linearForcing
                           length:sizeof(linearForcing) atIndex:15u];
                [encoder setBytes:&nonlinearIteration
                           length:sizeof(nonlinearIteration) atIndex:16u];
                [encoder setBuffer:state.femCandidate
                             offset:0u atIndex:17u];
                [encoder setBuffer:state.coupledGeneralizedCandidate
                             offset:0u atIndex:18u];
            });
            dispatchThreads("nm_fgmres_build_preconditioner", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:5u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:6u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:7u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:8u];
                [encoder setBuffer:state.femPreconditioned offset:0u atIndex:9u];
                [encoder setBuffer:state.primalContactArguments
                             offset:0u atIndex:10u];
            });
            for (std::uint32_t column = 0u; column < columnsThisCycle; ++column) {
                const NSUInteger columnOffset = vectorBytes * column;
                dispatchThreads("nm_fgmres_precondition", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:3u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:6u];
                    [encoder setBuffer:state.objects offset:0u atIndex:7u];
                });
                dispatchThreads("nm_fgmres_precondition_patches", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femTetrahedraCandidate
                                 offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeIncidence
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.femNodeRanges
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresBasis
                                 offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.femPreconditioned
                                 offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:6u];
                    [encoder setBuffer:state.fgmresStates
                                 offset:0u atIndex:7u];
                });
                dispatchGroups32("nm_fgmres_precondition_coarse", objectTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis
                                 offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.femPreconditioned
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.fgmresStates
                                 offset:0u atIndex:5u];
                });
                dispatchThreads("nm_fgmres_field_smoother_initialize", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.mixedFieldInverseDiagonal offset:0u atIndex:3u];
                    [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:4u];
                    [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:5u];
                    [encoder setBuffer:state.mixedFieldPreconditioned offset:0u atIndex:6u];
                    [encoder setBuffer:state.mixedFieldDirection offset:0u atIndex:7u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:8u];
                });
                // Apply a fixed matrix-free polynomial smoother to the four
                // packed transport blocks. The former nested field solve launched up
                // to seven kernels per inner iteration for every Arnoldi
                // column. This bounded right preconditioner keeps the outer
                // FGMRES authoritative while removing reductions, scalar
                // dependencies and most command traffic from the hot loop.
                NMMicrostepGPU fieldPreconditionerMicro = micro;
                fieldPreconditionerMicro.flags |=
                    NM_MICROSTEP_FIELD_PRECONDITIONER;
                constexpr std::array<float, 3> fieldSmootherDamping{
                    0.72f, 0.86f, 0.72f
                };
                const std::uint32_t fieldSmoothingPasses = std::min(
                    state.mixedSolverValue.executionBudgets.x,
                    static_cast<std::uint32_t>(fieldSmootherDamping.size()));
                for (std::uint32_t fieldPass = 0u;
                     fieldPass < fieldSmoothingPasses; ++fieldPass) {
                    fieldPreconditionerMicro.solverIteration = fieldPass;
                    dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                        setDispatch();
                        [encoder setBytes:&fieldPreconditionerMicro
                                   length:sizeof(fieldPreconditionerMicro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                        [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                        [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                        [encoder setBuffer:state.femDirection offset:0u atIndex:8u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
                    });
                    dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                        [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                        [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                        [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                        [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                        [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:7u];
                    });
                    dispatchThreads("nm_fgmres_field_smooth", femNodeTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                        [encoder setBuffer:state.fgmresBasis
                                     offset:columnOffset atIndex:2u];
                        [encoder setBuffer:state.mixedFieldInverseDiagonal
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.mixedFieldOperator
                                     offset:0u atIndex:4u];
                        [encoder setBuffer:state.mixedFieldSolution
                                     offset:0u atIndex:5u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:6u];
                        const float damping = fieldSmootherDamping[fieldPass];
                        [encoder setBytes:&damping length:sizeof(damping)
                                   atIndex:7u];
                    });
                }
                dispatchThreads("nm_fgmres_field_smoother_export", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:3u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                });
                dispatchThreads("nm_fgmres_precondition_cross", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                    [encoder setBuffer:state.femPreconditioned offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:5u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:6u];
                    [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:7u];
                    [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:8u];
                });
                dispatchThreads(
                    "nm_fgmres_precondition_mpm", mpmActiveNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:2u];
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:3u];
                    [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:4u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:5u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:6u];
                    [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:7u];
                    [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:8u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:9u];
                    [encoder setBuffer:state.femOperatorValue
                                 offset:0u atIndex:10u];
                });
                dispatchThreads(
                    "nm_fgmres_precondition_mpm_patches",
                    mpmActiveNodeTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&operatorMicro
                                   length:sizeof(operatorMicro) atIndex:1u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:3u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:4u];
                        [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:5u];
                        [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:6u];
                        [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:7u];
                        [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:8u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:9u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:11u];
                    });
                dispatchGroups32(
                    "nm_fgmres_precondition_mpm_objects",
                    objectTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.objects offset:0u atIndex:1u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:2u];
                        [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:4u];
                        [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:5u];
                        [encoder setBuffer:state.fgmresBasis offset:columnOffset atIndex:6u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:7u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:8u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:9u];
                    });
                if (state.requiresCoupledCandidate) {
                    dispatchThreads("nm_fgmres_export_rigid", rigidGeneralizedTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.fgmresBasis
                                     offset:columnOffset atIndex:1u];
                        [encoder setBuffer:state.coupledGeneralizedInput
                                     offset:0u atIndex:2u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:3u];
                    });
                    [encoder endEncoding];
                    id<MTLBlitCommandEncoder> clear =
                        [commandBuffer blitCommandEncoder];
                    if (clear == nil) {
                        ownership->preDynamicsOpen = false;
                        diagnostics.message =
                            "failed to clear coupled inverse-mass output";
                        return diagnostics;
                    }
                    [clear fillBuffer:state.coupledGeneralizedOutput
                                range:NSMakeRange(
                                    0u,
                                    rigidGeneralizedTotal * sizeof(float))
                                value:0u];
                    [clear endEncoding];
                    const CoupledCandidateQuery inverseQuery{
                        .input = (__bridge void*)state.coupledGeneralizedInput,
                        .output = (__bridge void*)state.coupledGeneralizedOutput,
                        .statuses = (__bridge void*)state.coupledInverseStatuses,
                        .operation =
                            CoupledCandidateOperation::inverseMassPreconditioner,
                        .generalizedVectorStride =
                            state.dispatch.rigidGeneralizedCapacity,
                        .statusStride =
                            static_cast<std::uint32_t>(environments),
                    };
                    if (!request.encodeCoupledCandidate(
                            request.coupledCandidateContext,
                            inverseQuery)) {
                        ownership->preDynamicsOpen = false;
                        diagnostics.message =
                            "MetalWorld failed to encode coupled inverse-mass preconditioning";
                        return diagnostics;
                    }
                    encoder = [commandBuffer computeCommandEncoder];
                    if (encoder == nil) {
                        ownership->preDynamicsOpen = false;
                        diagnostics.message =
                            "failed to resume Matter after coupled inverse mass";
                        return diagnostics;
                    }
                    [encoder setLabel:@"Numi Matter coupled FGMRES continuation"];
                    dispatchThreads("nm_fgmres_import_rigid", rigidGeneralizedTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.coupledGeneralizedOutput
                                     offset:0u atIndex:1u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:2u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:3u];
                    });
                }
                dispatchThreads("nm_fgmres_precondition_free_rigid",
                    rigidGeneralizedTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv) atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis
                                 offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:3u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                });
                dispatchThreads(
                    "nm_fem_human_attachment_map_direction",
                    femHumanAttachmentTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&coupledArticulatedNv
                                   length:sizeof(coupledArticulatedNv)
                                  atIndex:1u];
                        [encoder setBuffer:state.femHumanAttachments
                                     offset:0u atIndex:2u];
                        [encoder
                            setBuffer:state.femHumanAttachmentPointJacobians
                               offset:0u atIndex:3u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:4u];
                        [encoder setBuffer:state.statuses
                                     offset:0u atIndex:5u];
                    }
                );
                dispatchThreads("nm_fem_apply_operator_elements", tetrahedronTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.materials offset:0u atIndex:3u];
                    [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                    [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                    [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:10u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:12u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:13u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:14u];
                    [encoder setBuffer:state.mixedMaterials offset:0u atIndex:16u];
                    [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:17u];
                    [encoder setBuffer:state.learnedMaterials offset:0u atIndex:18u];
                    [encoder setBuffer:state.learnedLayers offset:0u atIndex:19u];
                    [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:20u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:21u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:22u];
                });
                dispatchThreads("nm_fgmres_gather_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:5u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:8u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:10u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:11u];
                    [encoder setBuffer:state.primalContactArguments offset:0u atIndex:12u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:13u];
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:14u];
                    [encoder setBuffer:state.coupledPointJacobians
                                 offset:0u atIndex:15u];
                });
                dispatchThreads(
                    "nm_fem_human_attachment_capture_operator",
                    femHumanAttachmentTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.femHumanAttachments
                                     offset:0u atIndex:1u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:2u];
                        [encoder setBuffer:state.femHumanAttachmentOperator
                                     offset:0u atIndex:3u];
                    }
                );
                const NSUInteger fieldVectorOffset =
                    columnOffset + femNodeTotal * sizeof(nm_float4);
                const NSUInteger fieldWorkOffset =
                    femNodeTotal * sizeof(nm_float4);
                dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                    [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:fieldVectorOffset atIndex:6u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:8u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
                });
                dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                    [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                    [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:fieldVectorOffset atIndex:6u];
                    [encoder setBuffer:state.femOperatorValue
                                 offset:fieldWorkOffset atIndex:7u];
                });
                dispatchThreads(
                    "nm_fgmres_apply_mpm", mpmActiveNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:1u];
                    [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:2u];
                    [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:3u];
                    [encoder setBuffer:state.contactNodeIncidence offset:0u atIndex:4u];
                    [encoder setBuffer:state.contactNodeRanges offset:0u atIndex:5u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:6u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:7u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:8u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:9u];
                    [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:10u];
                    [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:11u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:12u];
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:13u];
                    [encoder setBuffer:state.coupledPointJacobians
                                 offset:0u atIndex:14u];
                });
                if (!dispatchIndirect(
                        "nm_fgmres_apply_mpm_constitutive",
                        state.mpmActiveDispatch,
                        256u,
                        [&] {
                            setDispatch();
                            [encoder setBytes:&operatorMicro length:sizeof(operatorMicro) atIndex:1u];
                            [encoder setBuffer:state.objects offset:0u atIndex:2u];
                            [encoder setBuffer:state.materials offset:0u atIndex:3u];
                            [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                            [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                            [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                            [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                            [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:8u];
                            [encoder setBuffer:state.gridNodes offset:0u atIndex:9u];
                            [encoder setBuffer:state.mpmGrids offset:0u atIndex:10u];
                            [encoder setBuffer:state.mpmBlocks offset:0u atIndex:11u];
                            [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:12u];
                            [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:13u];
                            [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:14u];
                            [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:15u];
                            [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:16u];
                            [encoder setBuffer:state.schedulers offset:0u atIndex:17u];
                            [encoder setBuffer:state.adaptive offset:0u atIndex:18u];
                            [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:19u];
                            [encoder setBuffer:state.fgmresPreconditionedBasis offset:columnOffset atIndex:20u];
                            [encoder setBuffer:state.femOperatorValue offset:0u atIndex:21u];
                            [encoder setBuffer:state.fgmresStates offset:0u atIndex:22u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:23u];
                            [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:24u];
                        })) {
                    [encoder endEncoding];
                    ownership->preDynamicsOpen = false;
                    diagnostics.message =
                        "failed to encode implicit MPM constitutive action";
                    return diagnostics;
                }
                if (state.requiresCoupledCandidate) {
                    dispatchThreads("nm_fgmres_export_rigid", rigidGeneralizedTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:1u];
                        [encoder setBuffer:state.coupledGeneralizedInput
                                     offset:0u atIndex:2u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:3u];
                    });
                    [encoder endEncoding];
                    id<MTLBlitCommandEncoder> clear =
                        [commandBuffer blitCommandEncoder];
                    if (clear == nil) {
                        ownership->preDynamicsOpen = false;
                        diagnostics.message =
                            "failed to clear coupled mass-action output";
                        return diagnostics;
                    }
                    [clear fillBuffer:state.coupledGeneralizedOutput
                                range:NSMakeRange(
                                    0u,
                                    rigidGeneralizedTotal * sizeof(float))
                                value:0u];
                    [clear endEncoding];
                    const CoupledCandidateQuery massQuery{
                        .input = (__bridge void*)state.coupledGeneralizedInput,
                        .output = (__bridge void*)state.coupledGeneralizedOutput,
                        .operation = CoupledCandidateOperation::massAction,
                        .generalizedVectorStride =
                            state.dispatch.rigidGeneralizedCapacity,
                    };
                    if (!request.encodeCoupledCandidate(
                            request.coupledCandidateContext,
                            massQuery)) {
                        ownership->preDynamicsOpen = false;
                        diagnostics.message =
                            "MetalWorld failed to encode coupled articulated mass action";
                        return diagnostics;
                    }
                    encoder = [commandBuffer computeCommandEncoder];
                    if (encoder == nil) {
                        ownership->preDynamicsOpen = false;
                        diagnostics.message =
                            "failed to resume Matter after coupled mass action";
                        return diagnostics;
                    }
                    [encoder setLabel:@"Numi Matter coupled FGMRES continuation"];
                    dispatchThreads("nm_fgmres_import_rigid", rigidGeneralizedTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.coupledGeneralizedOutput
                                     offset:0u atIndex:1u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:2u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:3u];
                    });
                }
                dispatchThreads("nm_fgmres_apply_free_rigid",
                    rigidGeneralizedTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv) atIndex:1u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.femOperatorValue
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                });
                dispatchThreads("nm_fgmres_apply_primal_rigid_contacts",
                    rigidGeneralizedTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv) atIndex:1u];
                    [encoder setBuffer:state.fgmresPreconditionedBasis
                                 offset:columnOffset atIndex:2u];
                    [encoder setBuffer:state.femOperatorValue
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.coupledPointJacobians
                                 offset:0u atIndex:6u];
                });
                dispatchThreads(
                    "nm_fem_human_attachment_scatter_operator",
                    rigidGeneralizedTotal,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&coupledArticulatedNv
                                   length:sizeof(coupledArticulatedNv)
                                  atIndex:1u];
                        [encoder
                            setBuffer:state.femHumanAttachmentPointJacobians
                               offset:0u atIndex:2u];
                        [encoder setBuffer:state.femHumanAttachmentOperator
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:4u];
                        [encoder setBuffer:state.statuses
                                     offset:0u atIndex:5u];
                    }
                );
                dispatchThreads(
                    "nm_fem_human_attachment_eliminate_direction",
                    femHumanAttachmentTotal,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.femHumanAttachments
                                     offset:0u atIndex:1u];
                        [encoder setBuffer:state.fgmresPreconditionedBasis
                                     offset:columnOffset atIndex:2u];
                    }
                );
                dispatchGroups32(
                    "nm_fgmres_orthogonalize_and_finish_column",
                    environments,
                    [&] {
                        setDispatch();
                        [encoder setBuffer:state.mixedSolver
                                     offset:0u atIndex:1u];
                        [encoder setBytes:&column
                                   length:sizeof(column) atIndex:2u];
                        [encoder setBuffer:state.fgmresBasis
                                     offset:0u atIndex:3u];
                        [encoder setBuffer:state.femOperatorValue
                                     offset:0u atIndex:4u];
                        [encoder setBuffer:state.fgmresBasis
                                     offset:vectorBytes * (column + 1u)
                                    atIndex:5u];
                        [encoder setBuffer:state.fgmresHessenberg
                                     offset:0u atIndex:6u];
                        [encoder setBuffer:state.fgmresRotations
                                     offset:0u atIndex:7u];
                        [encoder setBuffer:state.fgmresLeastSquares
                                     offset:0u atIndex:8u];
                        [encoder setBuffer:state.fgmresStates
                                     offset:0u atIndex:9u];
                        [encoder setBytes:&linearForcing
                                   length:sizeof(linearForcing) atIndex:10u];
                    });
            }
            const std::uint32_t finalRestartCycle =
                restartCycle + 1u == restartCycleCount ? 1u : 0u;
            dispatchThreads("nm_fgmres_finalize_cycle", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.fgmresHessenberg offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.statuses offset:0u atIndex:5u];
                [encoder setBytes:&iterationOffset
                           length:sizeof(iterationOffset) atIndex:6u];
                [encoder setBytes:&finalRestartCycle
                           length:sizeof(finalRestartCycle) atIndex:7u];
                [encoder setBytes:&linearForcing
                           length:sizeof(linearForcing) atIndex:8u];
                [encoder setBuffer:state.fgmresRotations
                             offset:0u atIndex:9u];
                [encoder setBuffer:state.fgmresRestartCoefficients
                             offset:0u atIndex:10u];
            });
            dispatchThreads("nm_fgmres_accumulate", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:6u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:7u];
            });
            dispatchThreads("nm_fgmres_accumulate_fields", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:4u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:6u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:7u];
            });
            dispatchThreads(
                "nm_fgmres_accumulate_mpm", mpmActiveNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.fgmresPreconditionedBasis offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresLeastSquares offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:5u];
                [encoder setBytes:&restartCycle length:sizeof(restartCycle) atIndex:6u];
            });
            dispatchThreads("nm_fgmres_accumulate_rigid", rigidGeneralizedTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.fgmresPreconditionedBasis
                             offset:0u atIndex:2u];
                [encoder setBuffer:state.fgmresLeastSquares
                             offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:5u];
                [encoder setBytes:&restartCycle
                           length:sizeof(restartCycle) atIndex:6u];
            });
            if (finalRestartCycle == 0u) {
                dispatchThreads("nm_fgmres_restart_residual_nodes", femNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresBasis offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresRestartCoefficients
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:5u];
                });
                dispatchThreads(
                    "nm_fgmres_restart_residual_mpm",
                    mpmActiveNodeTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.fgmresBasis offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresRestartCoefficients offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                });
                dispatchThreads("nm_fgmres_restart_residual_rigid",
                    rigidGeneralizedTotal, [&] {
                    setDispatch();
                    [encoder setBuffer:state.fgmresBasis offset:0u atIndex:1u];
                    [encoder setBuffer:state.fgmresRestartCoefficients
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:3u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:4u];
                });
            }
            }

            dispatchGroups32("nm_fgmres_measure_correction", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:3u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.primalContactArguments offset:0u atIndex:6u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:7u];
                [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:8u];
                [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:9u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:10u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:11u];
            });

            // Line search evaluates the physical full-space displacement.
            // Lift the accepted reduced Human correction only after reduced
            // convergence metrics have been measured, then keep the FEM
            // update kernel from treating it as an independent nodal DOF.
            dispatchThreads(
                "nm_fem_human_attachment_map_direction",
                femHumanAttachmentTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:1u];
                    [encoder setBuffer:state.femHumanAttachments
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.femHumanAttachmentPointJacobians
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.femSolution
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.statuses
                                 offset:0u atIndex:5u];
                }
            );

            micro.solverIteration = nonlinearIteration;
            dispatchThreads("nm_fem_select_backtracking", objectTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:2u];
                [encoder setBuffer:state.objects offset:0u atIndex:3u];
                [encoder setBuffer:state.materials offset:0u atIndex:4u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:6u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:11u];
                [encoder setBuffer:state.statuses offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:13u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:14u];
            });
            dispatchGroups32(
                "nm_mpm_limit_implicit_line_search",
                objectTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:2u];
                    [encoder setBuffer:state.objects offset:0u atIndex:3u];
                    [encoder setBuffer:state.materials offset:0u atIndex:4u];
                    [encoder setBuffer:state.scalarPrograms offset:0u atIndex:5u];
                    [encoder setBuffer:state.instructions offset:0u atIndex:6u];
                    [encoder setBuffer:state.environmentParameters offset:0u atIndex:7u];
                    [encoder setBuffer:state.particleAccepted offset:0u atIndex:8u];
                    [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:9u];
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:10u];
                    [encoder setBuffer:state.mpmGrids offset:0u atIndex:11u];
                    [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:12u];
                    [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:13u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:14u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:15u];
                    [encoder setBuffer:state.adaptive offset:0u atIndex:16u];
                    [encoder setBuffer:state.femLineSearch offset:0u atIndex:17u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:18u];
                });
            dispatchGroups32(
                "nm_fem_synchronize_environment_line_search",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.femLineSearch
                                 offset:0u atIndex:1u];
                });
            dispatchGroups32(
                "nm_contact_limit_deformable_line_search",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                    [encoder setBuffer:state.deformableContacts offset:0u atIndex:4u];
                    [encoder setBuffer:state.deformableContactActiveIndices offset:0u atIndex:5u];
                    [encoder setBuffer:state.deformableContactActiveCounts offset:0u atIndex:6u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:8u];
                    [encoder setBuffer:state.femLineSearch offset:0u atIndex:9u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:11u];
                    [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:12u];
                    [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:13u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:14u];
                });
            dispatchGroups32(
                "nm_contact_limit_rigid_line_search",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.objects offset:0u atIndex:1u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                    [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:5u];
                    [encoder setBuffer:state.femLineSearch offset:0u atIndex:6u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                    [encoder setBuffer:state.gridNodes offset:0u atIndex:8u];
                    [encoder setBuffer:state.primalContactArguments offset:0u atIndex:9u];
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:10u];
                    [encoder setBuffer:state.coupledPointJacobians
                                 offset:0u atIndex:11u];
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:12u];
                });
            dispatchGroups32(
                "nm_fem_synchronize_environment_line_search",
                environments,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.femLineSearch
                                 offset:0u atIndex:1u];
                });
            dispatchThreads("nm_rigid_apply_candidate_solution",
                rigidCandidateTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femSolution offset:0u atIndex:1u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:2u];
                [encoder setBuffer:state.coupledGeneralizedCandidate
                             offset:0u atIndex:3u];
            });
            dispatchThreads("nm_fem_apply_solution", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:7u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:9u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:10u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:11u];
            });
            dispatchThreads(
                "nm_mpm_apply_implicit_solution", mpmActiveNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:2u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:3u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:5u];
                [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:6u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:7u];
            });
            dispatchThreads("nm_mixed_apply_kkt_solution", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:1u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:2u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:3u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.statuses offset:0u atIndex:6u];
            });
            dispatchThreads("nm_contact_publish_primal_history", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.contactSamples offset:0u atIndex:1u];
                [encoder setBuffer:state.contactHistoriesCandidate
                             offset:0u atIndex:2u];
            });
            (void)dispatchIndirect(
                "nm_contact_publish_deformable_history",
                state.deformableContactActiveDispatch,
                256u,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.deformableContacts offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableContactGlobalActiveIndices offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableContactActiveDispatch offset:0u atIndex:3u];
                    [encoder setBuffer:state.deformableContactHistoriesCandidate
                                 offset:0u atIndex:4u];
                });
            // Physics-triggered puncture is an active-set change of this same
            // nonlinear transaction. Apply it to candidate topology/state,
            // then let the following Newton iteration rebuild contact and the
            // KKT operator. No intermediate commit is permitted.
            if (nonlinearIteration <
                state.mixedSolverValue.executionBudgets.y) {
                dispatchThreads("nm_topology_detect_puncture", objectTotal, [&] {
                    setDispatch();
                    [encoder setBytes:&request.controlStep
                               length:sizeof(request.controlStep) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.rigidProxies offset:0u atIndex:4u];
                    [encoder setBuffer:state.rigidStates offset:0u atIndex:5u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:6u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:7u];
                    [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:8u];
                    [encoder setBuffer:state.contactPairs offset:0u atIndex:9u];
                    [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:10u];
                    [encoder setBuffer:state.contactSamples offset:0u atIndex:11u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u atIndex:12u];
                });
                const std::uint32_t triggeredMutationCount =
                    state.dispatch.objectCount;
                const std::uint32_t zeroMutationCount = 0u;
                dispatchThreads("nm_topology_execute_transaction", environments, [&] {
                    setDispatch();
                    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                    [encoder setBuffer:state.objects offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCapacities offset:0u atIndex:3u];
                    [encoder setBuffer:state.materials offset:0u atIndex:4u];
                    [encoder setBuffer:state.cookedMutationCommands offset:0u atIndex:5u];
                    [encoder setBuffer:borrowedMutations offset:0u atIndex:6u];
                    [encoder setBytes:&zeroMutationCount
                               length:sizeof(zeroMutationCount) atIndex:7u];
                    [encoder setBytes:&request.mutationCommandStride
                               length:sizeof(request.mutationCommandStride) atIndex:8u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                    [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:10u];
                    [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:11u];
                    [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:12u];
                    [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:13u];
                    [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:14u];
                    [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:15u];
                    [encoder setBuffer:state.femNodeRanges offset:0u atIndex:16u];
                    [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:17u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:18u];
                    [encoder setBuffer:state.triggeredMutationCommands offset:0u atIndex:19u];
                    [encoder setBytes:&triggeredMutationCount
                               length:sizeof(triggeredMutationCount) atIndex:20u];
                    [encoder setBytes:&zeroMutationCount
                               length:sizeof(zeroMutationCount) atIndex:21u];
                    [encoder setBuffer:state.femMaterialStateCandidate
                                 offset:0u atIndex:22u];
                    [encoder setBuffer:state.topologyStatesCheckpoint
                                 offset:0u atIndex:23u];
                });
                encodeTopologyRebuild(state.femCandidate);
                encodeTopologyCertificate(
                    state.femCandidate, state.femFieldsCandidate);
            }
            }
            // Reassemble the accepted Newton candidate once, then certify the
            // same generalized residual. This is a certificate pass only: no
            // post-contact FEM correction or second linear solve is encoded.
            if (!encodeFEMHumanAttachmentKinematics()) {
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
            dispatchThreads("nm_mixed_prepare_residual", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.objects offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:3u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:5u];
            });
            dispatchThreads("nm_mixed_apply_operator_elements", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:7u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:8u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:9u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
            });
            dispatchThreads("nm_mixed_apply_operator_nodes", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:1u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.elementOperator offset:0u atIndex:4u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:5u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:6u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:7u];
            });
            dispatchThreads("nm_mixed_build_residual_diagonal", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:7u];
                [encoder setBuffer:state.fieldBoundaries offset:0u atIndex:8u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.mixedFieldSolution offset:0u atIndex:10u];
                [encoder setBuffer:state.mixedFieldOperator offset:0u atIndex:11u];
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedFieldPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedFieldInverseDiagonal offset:0u atIndex:14u];
            });
            dispatchThreads("nm_fem_internal_forces", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:10u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:11u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:14u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:15u];
                [encoder setBuffer:state.learnedMaterials offset:0u atIndex:16u];
                [encoder setBuffer:state.learnedLayers offset:0u atIndex:17u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:18u];
            });
            dispatchThreads("nm_fem_build_mechanical_residual", femNodeTotal, [&] {
                const std::uint32_t preserveSolution = 1u;
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:5u];
                [encoder setBuffer:state.elementForces offset:0u atIndex:6u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:7u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:11u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.femPreconditioned offset:0u atIndex:13u];
                [encoder setBuffer:state.femDirection offset:0u atIndex:14u];
                [encoder setBuffer:state.femOperatorValue offset:0u atIndex:15u];
                [encoder setBuffer:state.statuses offset:0u atIndex:16u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:17u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:18u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:19u];
                [encoder setBytes:&preserveSolution
                           length:sizeof(preserveSolution) atIndex:20u];
                [encoder setBuffer:femExternalForces offset:0u atIndex:21u];
                [encoder setBytes:&hasFEMExternalForces
                           length:sizeof(hasFEMExternalForces) atIndex:22u];
                [encoder setBuffer:state.femConstraintReactions
                            offset:0u atIndex:23u];
            });
            if (!encodeCoupledPrimalContact(
                    true,
                    state.contactHistoriesCandidate,
                    state.deformableContactHistoriesCandidate
                )) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
            captureFEMHumanAttachmentResidual();
            // Candidate contact materialization also refreshes articulated
            // mass action. Rebuild the shared rigid rows after that final
            // contact pass so certification observes the accepted rigid,
            // FEM, and MPM equilibrium rather than the preceding Newton row.
            dispatchThreads("nm_fgmres_clear_rigid_residual",
                rigidGeneralizedTotalForResidual, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femResidual offset:0u atIndex:1u];
                });
            dispatchThreads("nm_contact_subtract_rigid_inertia_residual",
                rigidGeneralizedTotalForResidual, [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv) atIndex:1u];
                    [encoder setBuffer:state.coupledGeneralizedOutput
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.coupledGeneralizedCandidate
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:4u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:5u];
                });
            dispatchThreads("nm_contact_accumulate_rigid_residual",
                rigidGeneralizedTotalForResidual, [&] {
                    setDispatch();
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv) atIndex:1u];
                    [encoder setBuffer:state.primalContactArguments
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.femResidual offset:0u atIndex:3u];
                    [encoder setBuffer:state.coupledPointJacobians
                                 offset:0u atIndex:4u];
                });
            scatterFEMHumanAttachmentResidual();
            // The accepted Newton correction changes MPM grid velocities and
            // the final primal-contact rebuild changes their barrier forces.
            // Reassemble the MPM block once at that accepted candidate so the
            // certificate reports the actual nonlinear residual rather than
            // the residual from the preceding Newton iterate.
            dispatchThreads(
                "nm_mpm_build_implicit_residual", mpmActiveNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:2u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:3u];
                [encoder setBuffer:state.contactNodeIncidence offset:0u atIndex:4u];
                [encoder setBuffer:state.contactNodeRanges offset:0u atIndex:5u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:6u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:7u];
                [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:8u];
                [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:9u];
                [encoder setBuffer:state.primalContactArguments offset:0u atIndex:10u];
            });
            if (!dispatchIndirect(
                    "nm_mpm_build_constitutive_residual",
                    state.mpmActiveDispatch,
                    256u,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                        [encoder setBuffer:state.objects offset:0u atIndex:2u];
                        [encoder setBuffer:state.materials offset:0u atIndex:3u];
                        [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                        [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                        [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                        [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                        [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:8u];
                        [encoder setBuffer:state.gridNodes offset:0u atIndex:9u];
                        [encoder setBuffer:state.mpmGrids offset:0u atIndex:10u];
                        [encoder setBuffer:state.mpmBlocks offset:0u atIndex:11u];
                        [encoder setBuffer:state.mpmBlockLookup offset:0u atIndex:12u];
                        [encoder setBuffer:state.mpmActiveBlocks offset:0u atIndex:13u];
                        [encoder setBuffer:state.mpmBlockCounts offset:0u atIndex:14u];
                        [encoder setBuffer:state.mpmBlockOffsets offset:0u atIndex:15u];
                        [encoder setBuffer:state.mpmSortedParticleIndices offset:0u atIndex:16u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:17u];
                        [encoder setBuffer:state.adaptive offset:0u atIndex:18u];
                        [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:19u];
                        [encoder setBuffer:state.femDirection offset:0u atIndex:20u];
                        [encoder setBuffer:state.femResidual offset:0u atIndex:21u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:22u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:23u];
                        [encoder setBuffer:state.mpmNodeToActive offset:0u atIndex:24u];
                    })) {
                [encoder endEncoding];
                ownership->preDynamicsOpen = false;
                diagnostics.message =
                    "failed to encode final implicit MPM certificate residual";
                return diagnostics;
            }
            dispatchThreads("nm_contact_reduce_rigid", proxyTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.rigidIncidence offset:0u atIndex:1u];
                [encoder setBuffer:state.rigidRanges offset:0u atIndex:2u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                [encoder setBuffer:state.microstepReactions offset:0u atIndex:4u];
            });
            dispatchThreads("nm_accumulate_rigid_reactions", proxyTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.microstepReactions offset:0u atIndex:1u];
                [encoder setBuffer:state.frameReactions offset:0u atIndex:2u];
            });
            if (state.requiresRodNodes) {
                dispatchThreads(
                    "nm_apply_suture_strand_reactions",
                    static_cast<NSUInteger>(state.dispatch.environmentCount) *
                        request.rigid.rodNodeCount,
                    [&] {
                        setDispatch();
                        [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                        [encoder setBuffer:state.rigidProxies offset:0u atIndex:2u];
                        [encoder setBuffer:state.rigidIncidence offset:0u atIndex:3u];
                        [encoder setBuffer:state.rigidRanges offset:0u atIndex:4u];
                        [encoder setBuffer:state.contactSamples offset:0u atIndex:5u];
                        [encoder setBuffer:rodInverseMasses offset:0u atIndex:6u];
                        [encoder setBuffer:rodNodes offset:0u atIndex:7u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:8u];
                    }
                );
            }
            dispatchThreads("nm_mpm_g2p", particleTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.particleCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:9u];
                [encoder setBuffer:state.particleMaterialStateCandidate offset:0u atIndex:10u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:11u];
                [encoder setBuffer:state.mpmGrids offset:0u atIndex:12u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:13u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:14u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:15u];
                [encoder setBuffer:state.statuses offset:0u atIndex:16u];
            });
            dispatchThreads("nm_fem_integrate", femNodeTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:3u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:4u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:5u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
            });
            dispatchThreads("nm_fem_validate", tetrahedronTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.materials offset:0u atIndex:3u];
                [encoder setBuffer:state.scalarPrograms offset:0u atIndex:4u];
                [encoder setBuffer:state.instructions offset:0u atIndex:5u];
                [encoder setBuffer:state.environmentParameters offset:0u atIndex:6u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:8u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:9u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:10u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:11u];
                [encoder setBuffer:state.femMaterialStateCandidate offset:0u atIndex:12u];
                [encoder setBuffer:state.statuses offset:0u atIndex:13u];
            });
            dispatchGroups32("nm_mixed_certify", objectTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:3u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:5u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:6u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:7u];
                [encoder setBuffer:state.solverCertificates offset:0u atIndex:8u];
                [encoder setBuffer:state.statuses offset:0u atIndex:9u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:10u];
                [encoder setBuffer:state.fgmresStates offset:0u atIndex:11u];
                [encoder setBuffer:state.mixedFieldResidual offset:0u atIndex:12u];
                [encoder setBuffer:state.mixedMaterials offset:0u atIndex:13u];
                [encoder setBuffer:state.femSolution offset:0u atIndex:14u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:15u];
                [encoder setBuffer:state.mpmActiveNodeIndices offset:0u atIndex:16u];
                [encoder setBuffer:state.mpmActiveNodeCounts offset:0u atIndex:17u];
                [encoder setBuffer:state.femLineSearch offset:0u atIndex:18u];
                [encoder setBuffer:state.primalContactArguments offset:0u atIndex:19u];
                [encoder setBuffer:state.coupledGeneralizedCandidate
                             offset:0u atIndex:20u];
            });
            dispatchThreads(
                "nm_mask_primal_rigid_candidate",
                rigidCandidateTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.coupledGeneralizedCandidate
                                 offset:0u atIndex:2u];
                });
            dispatchThreads(
                "nm_fem_human_attachment_mask_reactions",
                femHumanAttachmentTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder
                        setBuffer:state.femHumanAttachmentResidualImpulses
                           offset:0u atIndex:2u];
                    [encoder setBuffer:state.femHumanAttachmentRawReactions
                                 offset:0u atIndex:3u];
                }
            );
            if (state.requiresCoupledCandidate) {
                [encoder endEncoding];
                const CoupledCandidateQuery publishQuery{
                    .input = (__bridge void*)
                        state.coupledGeneralizedCandidate,
                    .output = (__bridge void*)state.coupledGeneralizedOutput,
                    .candidateQ = (__bridge void*)state.coupledCandidateQ,
                    .operation = CoupledCandidateOperation::publishCandidate,
                    .generalizedVectorStride =
                        state.dispatch.rigidGeneralizedCapacity,
                    .candidateQStride = state.coupledQStride,
                };
                if (!request.encodeCoupledCandidate(
                        request.coupledCandidateContext, publishQuery)) {
                    ownership->preDynamicsOpen = false;
                    diagnostics.message =
                        "MetalWorld failed to publish Matter's accepted articulated candidate";
                    return diagnostics;
                }
                encoder = [commandBuffer computeCommandEncoder];
                if (encoder == nil) {
                    ownership->preDynamicsOpen = false;
                    diagnostics.message =
                        "failed to resume Matter after articulated candidate publication";
                    return diagnostics;
                }
                [encoder setLabel:@"Numi Matter primal publication continuation"];
            }
            dispatchThreads(
                "nm_publish_primal_free_rigid_candidate",
                bodyWrenchTotal,
                [&] {
                    setDispatch();
                    [encoder setBytes:&bridge length:sizeof(bridge) atIndex:1u];
                    [encoder setBytes:&coupledArticulatedNv
                               length:sizeof(coupledArticulatedNv)
                              atIndex:2u];
                    [encoder setBuffer:state.rigidProxies
                                 offset:0u atIndex:3u];
                    [encoder setBuffer:currentBodies offset:0u atIndex:4u];
                    [encoder setBuffer:state.coupledGeneralizedCandidate
                                 offset:0u atIndex:5u];
                    [encoder setBuffer:bodyWrenches offset:0u atIndex:6u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                    [encoder setBuffer:state.bodyProxyIncidence
                                 offset:0u atIndex:8u];
                    [encoder setBuffer:state.bodyProxyRanges
                                 offset:0u atIndex:9u];
                });
            dispatchThreads("nm_mpm_commit_microstep", particleTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.particleCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.particleMaterialStateCandidate offset:0u atIndex:5u];
            });
            dispatchThreads("nm_fem_commit_microstep", femTransactionalTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femMaterialStateCandidate offset:0u atIndex:5u];
            });
            dispatchThreads("nm_mixed_commit", femNodeTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCandidate offset:0u atIndex:3u];
            });
            dispatchThreads("nm_learned_commit", state.dispatch.learnedWeightCount, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.learnedWeightsCandidate offset:0u atIndex:3u];
                [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.learnedRevisionCandidate offset:0u atIndex:5u];
            });
            dispatchThreads("nm_contact_commit_histories", pairTotal, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.contactHistoriesAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:3u];
            });
            dispatchThreads(
                "nm_contact_commit_deformable_contact_histories",
                deformableContactHistoryTotal,
                [&] {
                    setDispatch();
                    [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                    [encoder setBuffer:state.deformableContactHistoriesAccepted
                                 offset:0u atIndex:2u];
                    [encoder setBuffer:state.deformableContactHistoriesCandidate
                                 offset:0u atIndex:3u];
                });
            dispatchGroups32("nm_scheduler_observe", objectTotal, [&] {
                setDispatch();
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBuffer:state.objects offset:0u atIndex:2u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:3u];
                [encoder setBuffer:state.gridNodes offset:0u atIndex:4u];
                [encoder setBuffer:state.mpmNodeRanges offset:0u atIndex:5u];
                [encoder setBuffer:state.mpmNodeGenerations offset:0u atIndex:6u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:7u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:8u];
                [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:9u];
                [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:11u];
            });
            dispatchThreads("nm_complete_microstep", environments, [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            });
        }

        dispatchThreads("nm_topology_commit", topologyTransactionalTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:3u];
            [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.femTopologyNodesCandidate offset:0u atIndex:5u];
            [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
            [encoder setBuffer:state.cohesiveFacesCandidate offset:0u atIndex:7u];
            [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
            [encoder setBuffer:state.punctureChannelsCandidate offset:0u atIndex:9u];
            [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
            [encoder setBuffer:state.topologyStatesCandidate offset:0u atIndex:11u];
        });

        dispatchThreads("nm_mpm_rollback_frame", particleTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:5u];
        });
        dispatchThreads("nm_fem_rollback_frame", femTransactionalTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:5u];
        });
        dispatchThreads("nm_topology_rollback", topologyTransactionalTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:5u];
            [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
            [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:7u];
            [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
            [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:9u];
            [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
            [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:11u];
            [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:12u];
            [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:13u];
            [encoder setBuffer:state.femNodeRanges offset:0u atIndex:14u];
            [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:15u];
        });
        dispatchThreads("nm_mixed_rollback", femNodeTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
        });
        dispatchThreads("nm_learned_rollback", state.dispatch.learnedWeightCount, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
            [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
            [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:5u];
        });
        dispatchThreads("nm_contact_rollback_histories", pairTotal, [&] {
            setDispatch();
            [encoder setBuffer:state.statuses offset:0u atIndex:1u];
            [encoder setBuffer:state.contactHistoriesAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.contactHistoriesCheckpoint offset:0u atIndex:3u];
        });
        dispatchThreads(
            "nm_contact_rollback_deformable_contact_histories",
            deformableContactHistoryTotal,
            [&] {
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:state.deformableContactHistoriesAccepted
                             offset:0u atIndex:2u];
                [encoder setBuffer:state.deformableContactHistoriesCandidate
                             offset:0u atIndex:3u];
                [encoder setBuffer:state.deformableContactHistoriesCheckpoint
                             offset:0u atIndex:4u];
            });

        dispatchThreads(
            "nm_latch_matter_status_into_rigid_world",
            environments,
            [&] {
                const std::uint32_t rigidWorldPhysicsSubstep =
                    request.rigidWorldPhysicsSubstep == NM_INVALID_INDEX
                    ? request.physicsSubstep
                    : request.rigidWorldPhysicsSubstep;
                setDispatch();
                [encoder setBuffer:state.statuses offset:0u atIndex:1u];
                [encoder setBuffer:worldStatuses offset:0u atIndex:2u];
                [encoder setBytes:&rigidWorldPhysicsSubstep
                           length:sizeof(rigidWorldPhysicsSubstep)
                          atIndex:3u];
            }
        );

        dispatchThreads("nm_topology_publish_growth_request", 1u, [&] {
            setDispatch();
            [encoder setBuffer:state.femCapacities offset:0u atIndex:1u];
            [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:2u];
            [encoder setBuffer:state.statuses offset:0u atIndex:3u];
            [encoder setBuffer:state.growthOwnership->readback offset:0u atIndex:4u];
        });

        [encoder endEncoding];
        if (firstEncodeForCommandBuffer) {
            ownership->activeCommandBuffer = request.commandBuffer;
            const std::weak_ptr<State::CommandOwnership> weakOwnership =
                ownership;
            const std::weak_ptr<State::GrowthOwnership> weakGrowth =
                state.growthOwnership;
            void* const borrowedIdentity = request.commandBuffer;
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                if (completed.status == MTLCommandBufferStatusCompleted) {
                    if (const auto growth = weakGrowth.lock()) {
                        const auto* gpu = static_cast<
                            const NMTopologyGrowthRequestGPU*>(
                                growth->readback.contents);
                        if (gpu != nullptr) {
                            TopologyGrowthRequest pending;
                            pending.required = gpu->identity.x != 0u;
                            pending.allocationGeneration = gpu->identity.y;
                            pending.firstObject = gpu->identity.z;
                            pending.reason = gpu->identity.w;
                            pending.nodes = gpu->topology.x;
                            pending.tetrahedra = gpu->topology.y;
                            pending.cohesiveFaces = gpu->topology.z;
                            pending.punctureChannels = gpu->topology.w;
                            pending.incidence = gpu->work.x;
                            pending.mutationCommands = gpu->work.y;
                            pending.rigidContacts = gpu->work.z;
                            pending.deformableContacts = gpu->work.w;
                            const std::lock_guard growthLock(growth->mutex);
                            growth->pending = pending;
                        }
                    }
                }
                if (const auto locked = weakOwnership.lock()) {
                    const std::lock_guard lock(locked->mutex);
                    if (locked->activeCommandBuffer == borrowedIdentity) {
                        locked->activeCommandBuffer = nullptr;
                        locked->preDynamicsOpen = false;
                        if (locked->preparedStateOpen) {
                            if (completed.status !=
                                    MTLCommandBufferStatusCompleted) {
                                locked->restoreRequired = true;
                                locked->preparedCommandFailed = true;
                                locked->terminalNoTouch = true;
                                locked->acceptedStateProofEligible = false;
                                locked->disposition =
                                    PreparedStateDisposition::terminalNoTouch;
                            } else {
                                locked->preparedCommandCompleted = true;
                                if (locked->applyCommandCompleted &&
                                    !locked->preparedCommandFailed) {
                                    locked->disposition =
                                        PreparedStateDisposition::resolved;
                                    locked->preparedStateOpen = false;
                                    locked->restoreRequired = false;
                                    locked->preparedStateRequested = false;
                                    locked->preparedCommandBuffer = nullptr;
                                    locked->preparedEnvironmentStatuses =
                                        nullptr;
                                    locked->preparedEnvironmentStatusesGPUAddress =
                                        0u;
                                    locked->acceptedStateProofEncoded = false;
                                    locked->acceptedStateProofEligible = false;
                                    locked->transactionPolicyFingerprint = 0u;
                                    locked->identificationAdvanced = false;
                                    locked->preparedCommandCompleted = false;
                                    locked->preparedCommandFailed = false;
                                    locked->terminalNoTouch = false;
                                    locked->applyCommandCompleted = false;
                                }
                            }
                            return;
                        }
                        if (completed.status !=
                                MTLCommandBufferStatusCompleted &&
                            locked->identificationAdvanced) {
                            locked->identificationGeneration =
                                locked->identificationCheckpoint;
                        }
                        locked->identificationAdvanced = false;
                        locked->acceptedStateProofEncoded = false;
                        locked->acceptedStateProofEligible = false;
                        locked->transactionPolicyFingerprint = 0u;
                        locked->preparedStateRequested = false;
                    }
                }
            }];
        }
        std::uint64_t transactionPolicyFingerprint =
            14695981039346656037ull;
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            0x4e5854504f4c4943ull); // "NXTPOLIC"
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            detail::kAcceptedStateProofSchemaVersion);
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint, state.executionFingerprint);
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            state.acceptedStateProofProgramFingerprint);
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint, state.sutureProxyBindingRevision);
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint, state.sutureProxyEdges.size());
        for (const std::uint32_t edge : state.sutureProxyEdges) {
            transactionPolicyFingerprint = mixFingerprint(
                transactionPolicyFingerprint, edge);
        }
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            state.coupledTimestepMultiplier.load(std::memory_order_acquire));
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            state.coupledTimestepDivisor.load(std::memory_order_acquire));
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            state.fgmresIterationBudgetOverride.load(
                std::memory_order_acquire));
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            state.newtonIterationBudgetOverride.load(
                std::memory_order_acquire));
        transactionPolicyFingerprint = mixFingerprint(
            transactionPolicyFingerprint,
            ownership->identificationGeneration);
        const std::array<std::uint64_t, 30u> transactionPolicyValues{{
            request.controlStep,
            request.physicsSubstep,
            request.physicsSubsteps,
            request.rigidWorldPhysicsSubstep,
            request.seed,
            std::bit_cast<std::uint32_t>(request.timestepSeconds),
            std::bit_cast<std::uint32_t>(frameTimestep),
            request.runIdentification ? 1u : 0u,
            request.runAdaptiveTransfer ? 1u : 0u,
            request.resetMaskStepStride,
            request.mutationCommandCount,
            request.mutationCommandStride,
            request.expectedMutationFingerprint,
            request.learnedWeightCount,
            request.learnedWeightRevision,
            request.expectedLearnedFingerprint,
            request.femExternalForceCount,
            request.rigidContactConstraintStride,
            request.articulationRootBody,
            request.rigid.currentBodyCount,
            request.rigid.currentBodyStride,
            request.rigid.bodyWrenchCount,
            request.rigid.bodyWrenchStride,
            request.rigid.sceneBodyCount,
            request.rigid.sceneStride,
            request.rigid.rodNodeCount,
            request.rigid.rodNodeStride,
            request.rigid.qStride,
            request.rigid.vStride,
            (state.captureEvents ? 1ull : 0ull) |
                (state.captureDiagnostics ? 2ull : 0ull) |
                (state.automaticIdentification ? 4ull : 0ull) |
                (state.adaptiveTransfer ? 8ull : 0ull) |
                (request.enablePreparedState ? 16ull : 0ull),
        }};
        for (const std::uint64_t value : transactionPolicyValues) {
            transactionPolicyFingerprint = mixFingerprint(
                transactionPolicyFingerprint, value);
        }
        if (transactionPolicyFingerprint == 0u) {
            transactionPolicyFingerprint = 14695981039346656037ull;
        }
        ownership->preDynamicsOpen = true;
        ownership->dispositionIdentity = {};
        ownership->disposition = PreparedStateDisposition::unknown;
        ownership->acceptedStateProofEncoded = false;
        ownership->acceptedStateProofEligible =
            !request.runIdentification &&
            request.resetMaskStepStride == 0u &&
            request.enablePreparedState;
        ownership->transactionPolicyFingerprint =
            transactionPolicyFingerprint;
        ownership->controlStep = request.controlStep;
        ownership->physicsSubstep = request.physicsSubstep;
        ownership->physicsSubsteps = request.physicsSubsteps;
        ownership->preparedStateRequested = request.enablePreparedState;
        diagnostics.encoded = true;
        diagnostics.residentBytes = state.residentBytes;
        diagnostics.device = nsString(state.device.name);
        diagnostics.message =
            "Numi Matter pre-dynamics graph encoded into borrowed command buffer";
        return diagnostics;
    }
}

bool Runtime::encodeAcceptedStateProof(
    const AcceptedStateProofPass& pass
) noexcept {
    @autoreleasepool {
        if (state_ == nullptr ||
            pass.abiVersion != NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION ||
            pass.structSize != sizeof(AcceptedStateProofPass) ||
            pass.commandBuffer == nullptr || pass.q == nullptr ||
            pass.v == nullptr || pass.mujocoStates == nullptr ||
            pass.matterGeneralizedReaction == nullptr ||
            pass.environmentStatuses == nullptr ||
            pass.matterStatuses == nullptr ||
            pass.acceptedStateProofs == nullptr) {
            return false;
        }
        State& state = *state_;
        if (pass.environmentCount == 0u ||
            pass.environmentCount != state.dispatch.environmentCount ||
            pass.qStride == 0u || pass.vStride == 0u ||
            pass.mujocoStateStride == 0u || pass.reactionStride == 0u ||
            pass.environmentStatusStride != 1u ||
            pass.matterStatusStride != 1u ||
            pass.acceptedStateProofStride != 1u ||
            pass.qCoordinateCount == 0u || pass.dofCount == 0u ||
            pass.qCoordinateCount != pass.dofCount + 1u ||
            pass.qCoordinateCount != pass.qStride ||
            pass.dofCount != pass.vStride ||
            pass.qStride != state.coupledQStride ||
            pass.vStride != state.coupledVStride ||
            pass.reactionStride != state.dispatch.rigidGeneralizedCapacity ||
            pass.programFingerprint == 0u ||
            pass.stateProofProgramFingerprint !=
                state.acceptedStateProofProgramFingerprint ||
            pass.transactionFingerprint == 0u ||
            pass.substepFingerprint == 0u ||
            pass.acceptedTimestampMicroseconds == 0u ||
            pass.physicsGeneration == 0u ||
            pass.linearizationEpoch == 0u || pass.slotGeneration == 0u ||
            pass.matterSourcePhysicsFingerprint !=
                state.sourcePhysicsFingerprint ||
            pass.matterDeviceProgramFingerprint !=
                state.executionFingerprint) {
            return false;
        }
        const std::uint64_t lastEnvironment =
            static_cast<std::uint64_t>(pass.environmentIdentifierBase) +
            pass.environmentCount - 1u;
        if (lastEnvironment > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        const auto exactElements = [&](const std::uint32_t stride,
                                       const std::uint64_t supplied) {
            return supplied ==
                static_cast<std::uint64_t>(pass.environmentCount) * stride;
        };
        if (!exactElements(pass.qStride, pass.qElementCount) ||
            !exactElements(pass.vStride, pass.vElementCount) ||
            !exactElements(pass.mujocoStateStride, pass.mujocoStateCount) ||
            !exactElements(
                pass.reactionStride,
                pass.matterGeneralizedReactionElementCount) ||
            pass.environmentStatusElementCount != pass.environmentCount ||
            pass.matterStatusElementCount != pass.environmentCount ||
            pass.acceptedStateProofElementCount != pass.environmentCount) {
            return false;
        }
        const auto byteCount = [](
            const std::uint64_t count,
            const std::uint64_t elementBytes,
            std::uint64_t& result
        ) {
            if (count != 0u && elementBytes >
                    std::numeric_limits<std::uint64_t>::max() / count) {
                return false;
            }
            result = count * elementBytes;
            return result <= std::numeric_limits<NSUInteger>::max();
        };
        std::uint64_t qBytes = 0u;
        std::uint64_t vBytes = 0u;
        std::uint64_t mujocoBytes = 0u;
        std::uint64_t reactionBytes = 0u;
        std::uint64_t environmentStatusBytes = 0u;
        std::uint64_t statusBytes = 0u;
        std::uint64_t outputBytes = 0u;
        if (!byteCount(pass.qElementCount, sizeof(float), qBytes) ||
            !byteCount(pass.vElementCount, sizeof(float), vBytes) ||
            !byteCount(pass.mujocoStateCount,
                       sizeof(MRMujocoMuscleStateGPU), mujocoBytes) ||
            !byteCount(pass.matterGeneralizedReactionElementCount,
                       sizeof(float), reactionBytes) ||
            !byteCount(pass.environmentStatusElementCount,
                       sizeof(MRMetalWorldStatusGPU),
                       environmentStatusBytes) ||
            !byteCount(pass.matterStatusElementCount,
                       sizeof(NMMatterStatusGPU), statusBytes) ||
            !byteCount(pass.acceptedStateProofElementCount,
                       sizeof(NMAcceptedStateProofGPU), outputBytes) ||
            mujocoBytes / pass.environmentCount >
                state
                    .acceptedStateProofMujocoBytesPerEnvironmentCapacity ||
            state.acceptedStateProofMujocoBytesPerEnvironmentCapacity == 0u) {
            return false;
        }

        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        __unsafe_unretained id<MTLBuffer> q =
            (__bridge id<MTLBuffer>)pass.q;
        __unsafe_unretained id<MTLBuffer> v =
            (__bridge id<MTLBuffer>)pass.v;
        __unsafe_unretained id<MTLBuffer> mujocoStates =
            (__bridge id<MTLBuffer>)pass.mujocoStates;
        __unsafe_unretained id<MTLBuffer> generalizedReaction =
            (__bridge id<MTLBuffer>)pass.matterGeneralizedReaction;
        __unsafe_unretained id<MTLBuffer> environmentStatuses =
            (__bridge id<MTLBuffer>)pass.environmentStatuses;
        __unsafe_unretained id<MTLBuffer> matterStatuses =
            (__bridge id<MTLBuffer>)pass.matterStatuses;
        __unsafe_unretained id<MTLBuffer> acceptedStateProofs =
            (__bridge id<MTLBuffer>)pass.acceptedStateProofs;
        const auto validBorrowedBuffer = [&](id<MTLBuffer> buffer,
                                             const std::uint64_t address,
                                             const std::uint64_t bytes) {
            return buffer != nil && buffer.device != nil &&
                buffer.device.registryID == state.device.registryID &&
                address != 0u && buffer.gpuAddress == address &&
                buffer.length >= static_cast<NSUInteger>(bytes);
        };
        const auto rangesOverlap = [](
            const std::uint64_t leftAddress,
            const std::uint64_t leftBytes,
            const std::uint64_t rightAddress,
            const std::uint64_t rightBytes
        ) {
            if (leftAddress == 0u || rightAddress == 0u ||
                leftBytes == 0u || rightBytes == 0u ||
                leftBytes > std::numeric_limits<std::uint64_t>::max() -
                    leftAddress ||
                rightBytes > std::numeric_limits<std::uint64_t>::max() -
                    rightAddress) {
                return true;
            }
            const std::uint64_t leftEnd = leftAddress + leftBytes;
            const std::uint64_t rightEnd = rightAddress + rightBytes;
            return leftAddress < rightEnd && rightAddress < leftEnd;
        };
        if (commandBuffer == nil || commandBuffer.commandQueue == nil ||
            commandBuffer.commandQueue.device.registryID !=
                state.device.registryID ||
            commandBuffer.status != MTLCommandBufferStatusNotEnqueued ||
            !validBorrowedBuffer(q, pass.qGPUAddress, qBytes) ||
            !validBorrowedBuffer(v, pass.vGPUAddress, vBytes) ||
            !validBorrowedBuffer(
                mujocoStates, pass.mujocoStatesGPUAddress, mujocoBytes) ||
            !validBorrowedBuffer(
                generalizedReaction,
                pass.matterGeneralizedReactionGPUAddress,
                reactionBytes) ||
            !validBorrowedBuffer(
                environmentStatuses,
                pass.environmentStatusesGPUAddress,
                environmentStatusBytes) ||
            !validBorrowedBuffer(
                matterStatuses, pass.matterStatusesGPUAddress, statusBytes) ||
            !validBorrowedBuffer(
                acceptedStateProofs,
                pass.acceptedStateProofsGPUAddress,
                outputBytes) ||
            matterStatuses != state.statuses) {
            return false;
        }

        const id<MTLBuffer> protectedProofArenas[] = {
            state.dispatchBuffer,
            state.particleAccepted,
            state.particleCheckpoint,
            state.particleMaterialStateAccepted,
            state.particleMaterialStateCheckpoint,
            state.femAccepted,
            state.femCheckpoint,
            state.femMaterialStateAccepted,
            state.femMaterialStateCheckpoint,
            state.femFieldsAccepted,
            state.femFieldsCheckpoint,
            state.femTetrahedraAccepted,
            state.femTetrahedraCheckpoint,
            state.femTopologyNodesAccepted,
            state.femTopologyNodesCheckpoint,
            state.cohesiveFacesAccepted,
            state.cohesiveFacesCheckpoint,
            state.punctureChannelsAccepted,
            state.punctureChannelsCheckpoint,
            state.topologyStatesAccepted,
            state.topologyStatesCheckpoint,
            state.learnedWeightsAccepted,
            state.learnedWeightsCheckpoint,
            state.learnedRevisionAccepted,
            state.learnedRevisionCheckpoint,
            state.contactHistoriesAccepted,
            state.contactHistoriesCheckpoint,
            state.deformableContactHistoriesAccepted,
            state.deformableContactHistoriesCandidate,
            state.deformableContactHistoriesCheckpoint,
            state.coupledGeneralizedCandidate,
            state.coupledGeneralizedPreparedCheckpoint,
            state.frameReactions,
            state.frameReactionsPreparedCheckpoint,
            state.rigidStates,
            state.rigidStatesPreparedCheckpoint,
            state.adaptive,
            state.adaptivePreparedCheckpoint,
            state.schedulers,
            state.schedulerCheckpoint,
            state.schedulerPrevious,
            state.schedulerPreviousPreparedCheckpoint,
            state.events,
            state.identificationDistributions,
            state.identificationPreparedCheckpoint,
            state.environmentParameters,
            state.environmentParametersPreparedCheckpoint,
            state.femNodeIncidence,
            state.femNodeIncidenceCheckpoint,
            state.femNodeRanges,
            state.femNodeRangesCheckpoint,
            state.acceptedStateProofHashes,
            state.acceptedStateProofScratchA,
            state.acceptedStateProofScratchB,
            state.preparedStateBindings,
            state.preparedStateActions,
            state.preparedStateRestoreStatuses,
            state.preparedStateApplyOutcome,
            state.preparedStatePublicationFacts,
        };
        struct BorrowedRange {
            id<MTLBuffer> buffer = nil;
            std::uint64_t address = 0u;
            std::uint64_t bytes = 0u;
        };
        const std::array<BorrowedRange, 7u> borrowedRanges{{
            {q, pass.qGPUAddress, qBytes},
            {v, pass.vGPUAddress, vBytes},
            {mujocoStates, pass.mujocoStatesGPUAddress, mujocoBytes},
            {generalizedReaction,
             pass.matterGeneralizedReactionGPUAddress, reactionBytes},
            {environmentStatuses,
             pass.environmentStatusesGPUAddress, environmentStatusBytes},
            {matterStatuses, pass.matterStatusesGPUAddress, statusBytes},
            {acceptedStateProofs,
             pass.acceptedStateProofsGPUAddress, outputBytes},
        }};
        for (std::size_t left = 0u; left < borrowedRanges.size(); ++left) {
            for (std::size_t right = left + 1u;
                 right < borrowedRanges.size(); ++right) {
                if (rangesOverlap(
                        borrowedRanges[left].address,
                        borrowedRanges[left].bytes,
                        borrowedRanges[right].address,
                        borrowedRanges[right].bytes)) {
                    return false;
                }
            }
        }
        for (id<MTLBuffer> protectedArena : protectedProofArenas) {
            if (protectedArena == nil) return false;
            for (const BorrowedRange& borrowed : borrowedRanges) {
                if (rangesOverlap(
                        borrowed.address,
                        borrowed.bytes,
                        protectedArena.gpuAddress,
                        static_cast<std::uint64_t>(protectedArena.length))) {
                    return false;
                }
            }
        }
        for (const BorrowedRange& borrowed : borrowedRanges) {
            if (borrowed.buffer == nil) {
                return false;
            }
        }

        const auto ownership = state.commandOwnership;
        std::unique_lock ownershipLock(ownership->mutex);
        if (ownership->activeCommandBuffer != pass.commandBuffer ||
            !ownership->preparedStateOpen ||
            ownership->preparedEnvironmentStatuses !=
                pass.environmentStatuses ||
            ownership->preparedEnvironmentStatusesGPUAddress !=
                pass.environmentStatusesGPUAddress ||
            ownership->acceptedStateProofEncoded ||
            !ownership->acceptedStateProofEligible ||
            ownership->applyEncoded || ownership->restoreRequired ||
            ownership->transactionPolicyFingerprint == 0u ||
            ownership->physicsSubstep + 1u != ownership->physicsSubsteps ||
            state.acceptedStateProofBegin == nil ||
            state.acceptedStateProofChunks == nil ||
            state.acceptedStateProofReduce == nil ||
            state.acceptedStateProofFold == nil ||
            state.acceptedStateProofFinalize == nil ||
            state.acceptedStateProofHashes == nil ||
            state.acceptedStateProofScratchA == nil ||
            state.acceptedStateProofScratchB == nil ||
            state.acceptedStateProofScratchStride == 0u) {
            return false;
        }

        struct ProofArena {
            void* buffer = nullptr;
            detail::AcceptedStateProofSource source{};
            std::uint32_t target = 0u;
            std::uint32_t flags = 0u;
            std::uint64_t bytesPerEnvironment = 0u;
            std::uint64_t sharedBytes = 0u;
        };
        const auto perEnvironmentBytes = [](
            const std::uint64_t count,
            const std::uint64_t elementBytes
        ) {
            return count * elementBytes;
        };
        const auto owned = [](id<MTLBuffer> buffer) {
            return (__bridge void*)buffer;
        };
        const std::uint64_t particleMaterialScalars =
            static_cast<std::uint64_t>(state.dispatch.particleCount) *
            state.dispatch.materialStateStride;
        const std::uint64_t femMaterialScalars =
            static_cast<std::uint64_t>(state.dispatch.tetrahedronCount) *
            state.dispatch.materialStateStride;
        const std::array<ProofArena, 26u> arenas{{
            {pass.q, detail::AcceptedStateProofSource::humanQ,
             detail::kAcceptedStateProofTargetHuman, 0u,
             perEnvironmentBytes(pass.qStride, sizeof(float)), 0u},
            {pass.v, detail::AcceptedStateProofSource::humanV,
             detail::kAcceptedStateProofTargetHuman, 0u,
             perEnvironmentBytes(pass.vStride, sizeof(float)), 0u},
            {pass.mujocoStates,
             detail::AcceptedStateProofSource::humanMujoco,
             detail::kAcceptedStateProofTargetHuman, 0u,
             perEnvironmentBytes(
                 pass.mujocoStateStride, sizeof(MRMujocoMuscleStateGPU)), 0u},
            {owned(state.particleAccepted),
             detail::AcceptedStateProofSource::matterParticles,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.particleCount, sizeof(NMParticleStateGPU)), 0u},
            {owned(state.particleMaterialStateAccepted),
             detail::AcceptedStateProofSource::matterParticleMaterialState,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(particleMaterialScalars, sizeof(float)), 0u},
            {owned(state.femAccepted),
             detail::AcceptedStateProofSource::matterFEMNodes,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.femNodeCount, sizeof(NMFEMNodeStateGPU)), 0u},
            {owned(state.femMaterialStateAccepted),
             detail::AcceptedStateProofSource::matterFEMMaterialState,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(femMaterialScalars, sizeof(float)), 0u},
            {owned(state.femFieldsAccepted),
             detail::AcceptedStateProofSource::matterFEMFields,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.femNodeCount, sizeof(NMFEMFieldStateGPU)), 0u},
            {owned(state.femTetrahedraAccepted),
             detail::AcceptedStateProofSource::matterFEMTetrahedra,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.tetrahedronCount, sizeof(NMTetrahedronGPU)), 0u},
            {owned(state.femTopologyNodesAccepted),
             detail::AcceptedStateProofSource::matterFEMTopologyNodes,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.topologyNodeCapacity,
                 sizeof(NMFEMTopologyNodeGPU)), 0u},
            {owned(state.cohesiveFacesAccepted),
             detail::AcceptedStateProofSource::matterCohesiveFaces,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.cohesiveFaceCount,
                 sizeof(NMCohesiveFaceGPU)), 0u},
            {owned(state.punctureChannelsAccepted),
             detail::AcceptedStateProofSource::matterPunctureChannels,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.punctureChannelCount,
                 sizeof(NMPunctureChannelGPU)), 0u},
            {owned(state.topologyStatesAccepted),
             detail::AcceptedStateProofSource::matterTopologyStates,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.objectCount, sizeof(NMFEMTopologyStateGPU)), 0u},
            {owned(state.learnedWeightsAccepted),
             detail::AcceptedStateProofSource::matterLearnedWeights,
             detail::kAcceptedStateProofTargetMatter,
             detail::kAcceptedStateProofSourceShared, 0u,
             perEnvironmentBytes(
                 state.dispatch.learnedWeightCount, sizeof(float))},
            {owned(state.learnedRevisionAccepted),
             detail::AcceptedStateProofSource::matterLearnedRevision,
             detail::kAcceptedStateProofTargetMatter,
             detail::kAcceptedStateProofSourceShared, 0u,
             sizeof(std::uint32_t)},
            {owned(state.contactHistoriesAccepted),
             detail::AcceptedStateProofSource::matterContactHistories,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.contactPairCount, sizeof(nm_float4)), 0u},
            {owned(state.deformableContactHistoriesAccepted),
             detail::AcceptedStateProofSource::matterDeformableContactHistories,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.deformableContactCapacity,
                 sizeof(NMDeformableContactHistoryGPU)), 0u},
            {owned(state.coupledGeneralizedCandidate),
             detail::AcceptedStateProofSource::matterGeneralizedCandidate,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.rigidGeneralizedCapacity, sizeof(float)), 0u},
            {owned(state.frameReactions),
             detail::AcceptedStateProofSource::matterFrameReactions,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.rigidProxyCount,
                 sizeof(NMRigidReactionGPU)), 0u},
            {owned(state.rigidStates),
             detail::AcceptedStateProofSource::matterRigidStates,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.rigidProxyCount,
                 sizeof(NMRigidStateGPU)), 0u},
            {owned(state.adaptive),
             detail::AcceptedStateProofSource::matterAdaptiveState,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.objectCount, sizeof(NMAdaptiveStateGPU)), 0u},
            {owned(state.schedulers),
             detail::AcceptedStateProofSource::matterSchedulers,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.dispatch.objectCount, sizeof(NMSchedulerStateGPU)), 0u},
            {owned(state.identificationDistributions),
             detail::AcceptedStateProofSource::matterIdentification,
             detail::kAcceptedStateProofTargetMatter,
             detail::kAcceptedStateProofSourceShared, 0u,
             perEnvironmentBytes(
                 state.identificationDistributionCount,
                 sizeof(NMIdentificationDistributionGPU))},
            {owned(state.environmentParameters),
             detail::AcceptedStateProofSource::matterEnvironmentParameters,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(state.dispatch.parameterCount, sizeof(float)),
             0u},
            {owned(state.femNodeIncidence),
             detail::AcceptedStateProofSource::matterFEMNodeIncidence,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.femNodeIncidenceStride, sizeof(std::uint32_t)), 0u},
            {owned(state.femNodeRanges),
             detail::AcceptedStateProofSource::matterFEMNodeRanges,
             detail::kAcceptedStateProofTargetMatter, 0u,
             perEnvironmentBytes(
                 state.femNodeRangeStride, sizeof(NMIncidenceRangeGPU)), 0u},
        }};
        for (const ProofArena& arena : arenas) {
            __unsafe_unretained id<MTLBuffer> source =
                (__bridge id<MTLBuffer>)arena.buffer;
            const std::uint64_t bytes =
                (arena.flags & detail::kAcceptedStateProofSourceShared) != 0u
                    ? arena.sharedBytes
                    : arena.bytesPerEnvironment * pass.environmentCount;
            const std::uint64_t logicalPerEnvironment =
                (arena.flags & detail::kAcceptedStateProofSourceShared) != 0u
                    ? arena.sharedBytes : arena.bytesPerEnvironment;
            const std::uint64_t chunkCount64 = std::max<std::uint64_t>(
                1u,
                logicalPerEnvironment /
                        detail::kAcceptedStateProofChunkBytes +
                    (logicalPerEnvironment %
                         detail::kAcceptedStateProofChunkBytes != 0u));
            if (source == nil || source.device == nil ||
                source.device.registryID != state.device.registryID ||
                source.length < bytes ||
                chunkCount64 > state.acceptedStateProofScratchStride) {
                return false;
            }
        }

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        [encoder setLabel:@"Numi Matter accepted-state content proof"];
        const auto dispatch = [&](id<MTLComputePipelineState> pipeline,
                                  const NSUInteger count) {
            [encoder setComputePipelineState:pipeline];
            const NSUInteger width = std::min<NSUInteger>(
                std::max<NSUInteger>(count, 1u),
                std::min<NSUInteger>(
                    pipeline.maxTotalThreadsPerThreadgroup, 256u));
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                  threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        };
        const auto barrier = [&]() {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
        detail::AcceptedStateProofBeginGPU begin{
            .environmentCount = pass.environmentCount,
            .schemaVersion = detail::kAcceptedStateProofSchemaVersion,
            .transactionPolicyFingerprint =
                ownership->transactionPolicyFingerprint,
        };
        [encoder setBytes:&begin length:sizeof(begin) atIndex:0u];
        [encoder setBuffer:state.acceptedStateProofHashes offset:0u atIndex:1u];
        dispatch(state.acceptedStateProofBegin, pass.environmentCount);
        barrier();

        for (const ProofArena& arena : arenas) {
            const std::uint64_t logicalPerEnvironment =
                (arena.flags & detail::kAcceptedStateProofSourceShared) != 0u
                    ? arena.sharedBytes : arena.bytesPerEnvironment;
            const std::uint32_t chunkCount = static_cast<std::uint32_t>(
                std::max<std::uint64_t>(
                    1u,
                    logicalPerEnvironment /
                            detail::kAcceptedStateProofChunkBytes +
                        (logicalPerEnvironment %
                             detail::kAcceptedStateProofChunkBytes != 0u)));
            detail::AcceptedStateProofChunkGPU chunks{
                .environmentCount = pass.environmentCount,
                .source = static_cast<std::uint32_t>(arena.source),
                .flags = arena.flags,
                .chunkBytes = detail::kAcceptedStateProofChunkBytes,
                .chunkCount = chunkCount,
                .scratchStride = state.acceptedStateProofScratchStride,
                .bytesPerEnvironment = arena.bytesPerEnvironment,
                .sharedBytes = arena.sharedBytes,
            };
            [encoder setBuffer:(__bridge id<MTLBuffer>)arena.buffer
                         offset:0u atIndex:0u];
            [encoder setBuffer:state.acceptedStateProofScratchA
                         offset:0u atIndex:1u];
            [encoder setBytes:&chunks length:sizeof(chunks) atIndex:2u];
            dispatch(
                state.acceptedStateProofChunks,
                static_cast<NSUInteger>(pass.environmentCount) * chunkCount);
            barrier();

            id<MTLBuffer> reductionInput = state.acceptedStateProofScratchA;
            id<MTLBuffer> reductionOutput = state.acceptedStateProofScratchB;
            std::uint32_t inputCount = chunkCount;
            std::uint32_t level = 0u;
            while (inputCount > 1u) {
                const std::uint32_t outputCount =
                    inputCount / 2u + inputCount % 2u;
                detail::AcceptedStateProofReduceGPU reduce{
                    .environmentCount = pass.environmentCount,
                    .source = static_cast<std::uint32_t>(arena.source),
                    .inputCount = inputCount,
                    .outputCount = outputCount,
                    .scratchStride = state.acceptedStateProofScratchStride,
                    .level = level,
                };
                [encoder setBuffer:reductionInput offset:0u atIndex:0u];
                [encoder setBuffer:reductionOutput offset:0u atIndex:1u];
                [encoder setBytes:&reduce length:sizeof(reduce) atIndex:2u];
                dispatch(
                    state.acceptedStateProofReduce,
                    static_cast<NSUInteger>(pass.environmentCount) *
                        outputCount);
                barrier();
                std::swap(reductionInput, reductionOutput);
                inputCount = outputCount;
                ++level;
            }
            detail::AcceptedStateProofFoldGPU fold{
                .environmentCount = pass.environmentCount,
                .source = static_cast<std::uint32_t>(arena.source),
                .target = arena.target,
                .flags = arena.flags,
                .scratchStride = state.acceptedStateProofScratchStride,
                .bytesPerEnvironment = arena.bytesPerEnvironment,
                .sharedBytes = arena.sharedBytes,
            };
            [encoder setBuffer:reductionInput offset:0u atIndex:0u];
            [encoder setBuffer:state.acceptedStateProofHashes
                         offset:0u atIndex:1u];
            [encoder setBytes:&fold length:sizeof(fold) atIndex:2u];
            dispatch(state.acceptedStateProofFold, pass.environmentCount);
            barrier();
        }

        detail::AcceptedStateProofFinalizeGPU finalize{
            .abiVersion = NM_MATTER_ACCEPTED_STATE_PROOF_ABI_VERSION,
            .structSize = NM_MATTER_ACCEPTED_STATE_PROOF_BYTES,
            .environmentCount = pass.environmentCount,
            .environmentIdentifierBase = pass.environmentIdentifierBase,
            .matterStatusStride = pass.matterStatusStride,
            .acceptedStateProofStride = pass.acceptedStateProofStride,
            .controlStep = ownership->controlStep,
            .physicsSubstep = ownership->physicsSubstep,
            .transactionSlot = pass.transactionSlot,
            .environmentStatusStride = pass.environmentStatusStride,
            .transactionFingerprint = pass.transactionFingerprint,
            .substepFingerprint = pass.substepFingerprint,
            .acceptedTimestampMicroseconds =
                pass.acceptedTimestampMicroseconds,
            .physicsGeneration = pass.physicsGeneration,
            .matterSourcePhysicsFingerprint =
                pass.matterSourcePhysicsFingerprint,
            .matterDeviceProgramFingerprint =
                pass.matterDeviceProgramFingerprint,
            .stateProofProgramFingerprint =
                pass.stateProofProgramFingerprint,
            .adapterProgramFingerprint = pass.programFingerprint,
            .transactionPolicyFingerprint =
                ownership->transactionPolicyFingerprint,
            .linearizationEpoch = pass.linearizationEpoch,
            .slotGeneration = pass.slotGeneration,
        };
        [encoder setBytes:&finalize length:sizeof(finalize) atIndex:0u];
        [encoder setBuffer:state.acceptedStateProofHashes offset:0u atIndex:1u];
        [encoder setBuffer:matterStatuses offset:0u atIndex:2u];
        [encoder setBuffer:acceptedStateProofs offset:0u atIndex:3u];
        [encoder setBuffer:state.preparedStateBindings offset:0u atIndex:4u];
        [encoder setBuffer:environmentStatuses offset:0u atIndex:5u];
        dispatch(state.acceptedStateProofFinalize, pass.environmentCount);
        [encoder endEncoding];
        ownership->acceptedStateProofEncoded = true;
        ownership->dispositionIdentity = PreparedStateDispositionIdentity{
            .abiVersion = 1u,
            .structSize = sizeof(PreparedStateDispositionIdentity),
            .controlStep = ownership->controlStep,
            .physicsSubstep = ownership->physicsSubstep,
            .physicsSubstepCount = ownership->physicsSubsteps,
            .transactionSlot = pass.transactionSlot,
            .reserved0 = 0u,
            .reserved1 = 0u,
            .ownerProgramFingerprint = pass.programFingerprint,
            .transactionFingerprint = pass.transactionFingerprint,
            .linearizationEpoch = pass.linearizationEpoch,
            .slotGeneration = pass.slotGeneration,
        };
        ownership->disposition = PreparedStateDisposition::prepared;
        return true;
    }
}

bool Runtime::applyPreparedState(
    const AcceptedStateApplyPass& pass
) noexcept {
    return applyPreparedStateImpl(pass);
}

bool Runtime::applyPreparedStateImpl(
    const AcceptedStateApplyPass& input
) noexcept {
    struct CommonPass {
        std::uint32_t environmentCount = 0u;
        std::uint32_t environmentIdentifierBase = 0u;
        void* commandBuffer = nullptr;
        void* finalPhysicsStateTokens = nullptr;
        std::uint64_t finalPhysicsStateTokensGPUAddress = 0u;
        std::uint64_t finalPhysicsStateTokenBytes = 0u;
        std::uint32_t finalPhysicsStateTokenStrideBytes = 0u;
        std::uint32_t controlStep = 0u;
        std::uint32_t physicsSubstep = 0u;
        std::uint32_t physicsSubstepCount = 0u;
        std::uint32_t transactionSlot = 0u;
        std::uint64_t ownerProgramFingerprint = 0u;
        std::uint64_t transactionFingerprint = 0u;
        std::uint64_t linearizationEpoch = 0u;
        std::uint64_t slotGeneration = 0u;
    } common{
        .environmentCount = input.environmentCount,
        .environmentIdentifierBase = input.environmentIdentifierBase,
        .commandBuffer = input.commandBuffer,
        .finalPhysicsStateTokens = input.proposedPhysicsStateTokens,
        .finalPhysicsStateTokensGPUAddress =
            input.proposedPhysicsStateTokensGPUAddress,
        .finalPhysicsStateTokenBytes =
            input.proposedPhysicsStateTokenBytes,
        .finalPhysicsStateTokenStrideBytes =
            input.proposedPhysicsStateTokenStrideBytes,
        .controlStep = input.controlStep,
        .physicsSubstep = input.physicsSubstep,
        .physicsSubstepCount = input.physicsSubstepCount,
        .transactionSlot = input.transactionSlot,
        .ownerProgramFingerprint = input.ownerProgramFingerprint,
        .transactionFingerprint = input.transactionFingerprint,
        .linearizationEpoch = input.linearizationEpoch,
        .slotGeneration = input.slotGeneration,
    };
    const CommonPass* pass = &common;
    const AcceptedStateApplyPass* applyPass = &input;
    void* const commandBufferPointer = input.commandBuffer;
    const bool forceRestore =
        input.mode == PreparedStateApplyMode::forceReject;
    const bool brainAckProvided = input.brainAcks != nullptr &&
        input.brainAcksGPUAddress != 0u &&
        input.brainAckElementCount != 0u && input.brainAckStride != 0u;
    const bool brainAckAbsent = input.brainAcks == nullptr &&
        input.brainAcksGPUAddress == 0u &&
        input.brainAckElementCount == 0u && input.brainAckStride == 0u;
    @autoreleasepool {
        if (state_ == nullptr || commandBufferPointer == nullptr ||
            pass->commandBuffer != commandBufferPointer ||
            pass->finalPhysicsStateTokens == nullptr ||
            pass->finalPhysicsStateTokensGPUAddress == 0u ||
            pass->finalPhysicsStateTokenStrideBytes !=
                MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES ||
            pass->physicsSubstep != 0u ||
            pass->physicsSubstepCount != 1u) {
            return false;
        }
        if (applyPass->abiVersion != 1u ||
             applyPass->structSize != sizeof(AcceptedStateApplyPass) ||
             (applyPass->mode != PreparedStateApplyMode::validateBrainAck &&
              applyPass->mode != PreparedStateApplyMode::forceReject) ||
             applyPass->reserved0 != 0u || applyPass->reserved1 != 0u ||
             applyPass->reserved2 != 0u || applyPass->reserved3 != 0u ||
             applyPass->environmentCount != 1u ||
             applyPass->proposals == nullptr ||
             applyPass->applyActions == nullptr ||
             applyPass->matterApplyOutcomes == nullptr ||
             applyPass->proposalsGPUAddress == 0u ||
             applyPass->applyActionsGPUAddress == 0u ||
             applyPass->matterApplyOutcomesGPUAddress == 0u ||
             applyPass->proposalStride == 0u ||
             applyPass->applyActionStride == 0u ||
             applyPass->matterApplyOutcomeStride == 0u ||
             applyPass->proposalElementCount !=
                 static_cast<std::uint64_t>(applyPass->environmentCount) *
                     applyPass->proposalStride ||
             (!forceRestore && !brainAckProvided) ||
             (forceRestore && !brainAckProvided && !brainAckAbsent) ||
             (brainAckProvided && applyPass->brainAckElementCount !=
                 static_cast<std::uint64_t>(applyPass->environmentCount) *
                     applyPass->brainAckStride) ||
             applyPass->applyActionElementCount !=
                 static_cast<std::uint64_t>(applyPass->environmentCount) *
                     applyPass->applyActionStride ||
             applyPass->matterApplyOutcomeElementCount !=
                 static_cast<std::uint64_t>(applyPass->environmentCount) *
                     applyPass->matterApplyOutcomeStride ||
             applyPass->ownerProgramFingerprint == 0u ||
             applyPass->transactionFingerprint == 0u ||
             applyPass->linearizationEpoch == 0u ||
             applyPass->slotGeneration == 0u) {
            return false;
        }
        State& state = *state_;
        // A uint32 environment count times the fixed 64-byte token is exact
        // in uint64; the following NSUInteger gate covers host addressability.
        const std::uint64_t expectedTokenBytes =
            static_cast<std::uint64_t>(pass->environmentCount) *
            MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_BYTES;
        if (pass->environmentCount == 0u ||
            pass->environmentCount != state.dispatch.environmentCount ||
            pass->finalPhysicsStateTokenBytes != expectedTokenBytes ||
            pass->finalPhysicsStateTokenBytes >
                std::numeric_limits<NSUInteger>::max()) {
            return false;
        }
        const std::uint64_t lastEnvironment =
            static_cast<std::uint64_t>(pass->environmentIdentifierBase) +
            pass->environmentCount - 1u;
        if (lastEnvironment > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }

        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)commandBufferPointer;
        __unsafe_unretained id<MTLBuffer> finalTokens =
            (__bridge id<MTLBuffer>)pass->finalPhysicsStateTokens;
        if (commandBuffer == nil || commandBuffer.commandQueue == nil ||
            commandBuffer.commandQueue.device.registryID !=
                state.device.registryID ||
            commandBuffer.status != MTLCommandBufferStatusNotEnqueued ||
            finalTokens == nil || finalTokens.device == nil ||
            finalTokens.device.registryID != state.device.registryID ||
            finalTokens.gpuAddress !=
                pass->finalPhysicsStateTokensGPUAddress ||
            finalTokens.length < pass->finalPhysicsStateTokenBytes) {
            return false;
        }

        __unsafe_unretained id<MTLBuffer> proposals = nil;
        __unsafe_unretained id<MTLBuffer> brainAcks = nil;
        __unsafe_unretained id<MTLBuffer> applyActions = nil;
        __unsafe_unretained id<MTLBuffer> matterApplyOutcomes = nil;
        std::uint64_t proposalBytes = 0u;
        std::uint64_t brainAckBytes = 0u;
        std::uint64_t applyActionBytes = 0u;
        std::uint64_t matterApplyOutcomeBytes = 0u;
        const auto recordBytes = [](const std::uint64_t count,
                                    const std::uint64_t size,
                                    std::uint64_t& bytes) noexcept {
            if (count == 0u || count >
                std::numeric_limits<std::uint64_t>::max() / size) {
                return false;
            }
            bytes = count * size;
            return bytes <= std::numeric_limits<NSUInteger>::max();
        };
        if (!recordBytes(applyPass->proposalElementCount,
                         sizeof(NMOwnerProposalGPU), proposalBytes) ||
            (brainAckProvided &&
             !recordBytes(applyPass->brainAckElementCount,
                          sizeof(NMOwnerBrainAckGPU), brainAckBytes)) ||
            !recordBytes(applyPass->applyActionElementCount,
                         sizeof(NMOwnerApplyActionGPU), applyActionBytes) ||
            !recordBytes(applyPass->matterApplyOutcomeElementCount,
                         sizeof(NMMatterApplyOutcomeGPU),
                         matterApplyOutcomeBytes)) {
            return false;
        }
        proposals = (__bridge id<MTLBuffer>)applyPass->proposals;
        brainAcks = brainAckProvided
            ? (__bridge id<MTLBuffer>)applyPass->brainAcks
            : state.dummy;
        applyActions = (__bridge id<MTLBuffer>)applyPass->applyActions;
        matterApplyOutcomes =
            (__bridge id<MTLBuffer>)applyPass->matterApplyOutcomes;
        const auto exactBuffer = [&](id<MTLBuffer> buffer,
                                     const std::uint64_t address,
                                     const std::uint64_t bytes) {
            return buffer != nil && buffer.device != nil &&
                buffer.device.registryID == state.device.registryID &&
                buffer.gpuAddress == address && buffer.length >= bytes;
        };
        if (!exactBuffer(proposals, applyPass->proposalsGPUAddress,
                         proposalBytes) ||
            (brainAckProvided &&
             !exactBuffer(brainAcks, applyPass->brainAcksGPUAddress,
                          brainAckBytes)) ||
            !exactBuffer(applyActions, applyPass->applyActionsGPUAddress,
                         applyActionBytes) ||
            !exactBuffer(matterApplyOutcomes,
                         applyPass->matterApplyOutcomesGPUAddress,
                         matterApplyOutcomeBytes)) {
            return false;
        }

        const auto ownership = state.commandOwnership;
        std::unique_lock ownershipLock(ownership->mutex);
        if (!ownership->preparedStateOpen || ownership->applyEncoded ||
            ownership->applyCommandCompleted ||
            ownership->applyCommandBuffer != nullptr ||
            ownership->controlStep != pass->controlStep ||
            ownership->physicsSubstep != pass->physicsSubstep ||
            ((forceRestore && ownership->disposition !=
                    PreparedStateDisposition::prepared &&
                ownership->disposition !=
                    PreparedStateDisposition::restoreRequired) ||
             (!forceRestore && ownership->disposition !=
                    PreparedStateDisposition::prepared)) ||
            (!forceRestore && ownership->restoreRequired) ||
            !ownership->preparedCommandCompleted ||
            ownership->preparedCommandFailed ||
            ownership->terminalNoTouch ||
            (!forceRestore && !ownership->acceptedStateProofEncoded) ||
            ownership->dispositionIdentity.transactionSlot !=
                pass->transactionSlot ||
            ownership->dispositionIdentity.ownerProgramFingerprint !=
                pass->ownerProgramFingerprint ||
            ownership->dispositionIdentity.transactionFingerprint !=
                pass->transactionFingerprint ||
            ownership->dispositionIdentity.linearizationEpoch !=
                pass->linearizationEpoch ||
            ownership->dispositionIdentity.slotGeneration !=
                pass->slotGeneration ||
            ownership->dispositionIdentity.physicsSubstepCount !=
                pass->physicsSubstepCount) {
            return false;
        }

        const id<MTLBuffer> protectedArenas[] = {
            state.dispatchBuffer,
            state.statuses,
            state.particleAccepted,
            state.particleCheckpoint,
            state.particleMaterialStateAccepted,
            state.particleMaterialStateCheckpoint,
            state.femAccepted,
            state.femCheckpoint,
            state.femMaterialStateAccepted,
            state.femMaterialStateCheckpoint,
            state.femTetrahedraAccepted,
            state.femTetrahedraCheckpoint,
            state.femTopologyNodesAccepted,
            state.femTopologyNodesCheckpoint,
            state.cohesiveFacesAccepted,
            state.cohesiveFacesCheckpoint,
            state.punctureChannelsAccepted,
            state.punctureChannelsCheckpoint,
            state.topologyStatesAccepted,
            state.topologyStatesCheckpoint,
            state.femNodeIncidence,
            state.femNodeIncidenceCheckpoint,
            state.femNodeRanges,
            state.femNodeRangesCheckpoint,
            state.femFieldsAccepted,
            state.femFieldsCheckpoint,
            state.learnedWeightsAccepted,
            state.learnedWeightsCheckpoint,
            state.learnedRevisionAccepted,
            state.learnedRevisionCheckpoint,
            state.contactHistoriesAccepted,
            state.contactHistoriesCheckpoint,
            state.deformableContactHistoriesAccepted,
            state.deformableContactHistoriesCandidate,
            state.deformableContactHistoriesCheckpoint,
            state.schedulers,
            state.schedulerCheckpoint,
            state.schedulerPrevious,
            state.schedulerPreviousPreparedCheckpoint,
            state.events,
            state.preparedStateBindings,
            state.preparedStateActions,
            state.preparedStateRestoreStatuses,
            state.preparedStateApplyOutcome,
            state.preparedStatePublicationFacts,
            state.acceptedStateProofHashes,
            state.acceptedStateProofScratchA,
            state.acceptedStateProofScratchB,
            state.adaptive,
            state.adaptivePreparedCheckpoint,
            state.frameReactions,
            state.frameReactionsPreparedCheckpoint,
            state.coupledGeneralizedCandidate,
            state.coupledGeneralizedPreparedCheckpoint,
            state.rigidStates,
            state.rigidStatesPreparedCheckpoint,
            state.identificationDistributions,
            state.identificationPreparedCheckpoint,
            state.environmentParameters,
            state.environmentParametersPreparedCheckpoint,
        };
        const auto overlaps = [](
            const std::uint64_t left,
            const std::uint64_t leftBytes,
            const std::uint64_t right,
            const std::uint64_t rightBytes
        ) {
            if (left == 0u || right == 0u || leftBytes == 0u ||
                rightBytes == 0u ||
                leftBytes > std::numeric_limits<std::uint64_t>::max() - left ||
                rightBytes >
                    std::numeric_limits<std::uint64_t>::max() - right) {
                return true;
            }
            return left < right + rightBytes && right < left + leftBytes;
        };
        struct BorrowedRegion {
            std::uint64_t address = 0u;
            std::uint64_t bytes = 0u;
        };
        const std::array<BorrowedRegion, 5u> applicationRegions{{
            {proposals.gpuAddress, proposalBytes},
            {brainAckProvided ? brainAcks.gpuAddress : 0u,
             brainAckProvided ? brainAckBytes : 0u},
            {applyActions.gpuAddress, applyActionBytes},
            {matterApplyOutcomes.gpuAddress, matterApplyOutcomeBytes},
            {finalTokens.gpuAddress, pass->finalPhysicsStateTokenBytes},
        }};
        for (std::size_t left = 0u;
             left < applicationRegions.size(); ++left) {
            for (std::size_t right = left + 1u;
                 right < applicationRegions.size(); ++right) {
                if (applicationRegions[left].bytes == 0u ||
                    applicationRegions[right].bytes == 0u) {
                    continue;
                }
                if (overlaps(applicationRegions[left].address,
                             applicationRegions[left].bytes,
                             applicationRegions[right].address,
                             applicationRegions[right].bytes)) {
                    return false;
                }
            }
        }
        for (id<MTLBuffer> arena : protectedArenas) {
            if (arena == nil || overlaps(
                    finalTokens.gpuAddress,
                    pass->finalPhysicsStateTokenBytes,
                    arena.gpuAddress,
                    arena.length)) {
                return false;
            }
            for (const BorrowedRegion& region : applicationRegions) {
                if (region.bytes == 0u) continue;
                if (overlaps(region.address, region.bytes,
                             arena.gpuAddress, arena.length)) {
                    return false;
                }
            }
        }

        // Reserve the exact NotEnqueued command before creating its encoder.
        // If any later encode step fails, the caller must discard that command
        // and Runtime::cancel can prove that no partial rollback executed,
        // clear only this reservation, and leave the prepared proof retryable.
        ownership->applyEncoded = true;
        ownership->applyCommandCompleted = false;
        ownership->applyCommandBuffer = commandBufferPointer;
        ownership->disposition = PreparedStateDisposition::applying;

        auto* applyOutcome = static_cast<std::uint32_t*>(
            state.preparedStateApplyOutcome.contents);
        auto* publicationFacts =
            static_cast<NMPreparedStatePublicationFactsGPU*>(
                state.preparedStatePublicationFacts.contents);
        if (applyOutcome == nullptr || publicationFacts == nullptr) {
            return false;
        }
        *applyOutcome = NM_PREPARED_STATE_APPLY_PENDING;
        *publicationFacts = {};

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        [encoder setLabel:@"Numi Matter apply prepared state"];
        const auto dispatch = [&](id<MTLComputePipelineState> pipeline,
                                  const NSUInteger count,
                                  const auto& bind) {
            if (count == 0u) return true;
            if (pipeline == nil) return false;
            [encoder setComputePipelineState:pipeline];
            bind();
            const NSUInteger width = std::min<NSUInteger>(
                256u, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
            return true;
        };
        const auto runtimeDispatch = [&](const char* name,
                                         const NSUInteger count,
                                         const auto& bind) {
            return dispatch(state.pipeline(name), count, [&] {
                [encoder setBuffer:state.dispatchBuffer offset:0u atIndex:0u];
                bind();
            });
        };

        NMPreparedStateApplyGPU applicationDispatch{
            .abiVersion = NM_MATTER_OWNER_APPLY_ABI_VERSION,
            .environmentCount = pass->environmentCount,
            .proposalStride = applyPass->proposalStride,
            .brainAckStride = brainAckProvided
                ? applyPass->brainAckStride : 1u,
            .applyActionStride = applyPass->applyActionStride,
            .matterApplyOutcomeStride =
                applyPass->matterApplyOutcomeStride,
            .proposedTokenStrideBytes =
                applyPass->proposedPhysicsStateTokenStrideBytes,
            .stepIndex = 0u,
            .substepIndex = pass->physicsSubstep,
            .transactionSlot = pass->transactionSlot,
            .physicsSubstepCount = pass->physicsSubstepCount,
            .controlStep = pass->controlStep,
            .forceRestore = forceRestore ? 1u : 0u,
            .reserved0 = 0u,
            .reserved1 = 0u,
            .reserved2 = 0u,
            .ownerProgramFingerprint = pass->ownerProgramFingerprint,
            .transactionFingerprint = pass->transactionFingerprint,
            .linearizationEpoch = pass->linearizationEpoch,
            .slotGeneration = pass->slotGeneration,
            .matterProgramFingerprint =
                state.acceptedStateProofProgramFingerprint,
            .reserved3 = 0u,
        };
        bool encoded = dispatch(
            state.preparedStateValidateApplication,
            pass->environmentCount,
            [&] {
                [encoder setBytes:&applicationDispatch
                           length:sizeof(applicationDispatch) atIndex:0u];
                [encoder setBuffer:proposals offset:0u atIndex:1u];
                [encoder setBuffer:brainAcks offset:0u atIndex:2u];
                [encoder setBuffer:applyActions offset:0u atIndex:3u];
                [encoder setBuffer:state.preparedStateBindings
                             offset:0u atIndex:4u];
                [encoder setBuffer:state.preparedStateActions
                             offset:0u atIndex:5u];
                [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                [encoder setBuffer:matterApplyOutcomes
                             offset:0u atIndex:7u];
                [encoder setBuffer:finalTokens offset:0u atIndex:8u];
                [encoder setBuffer:state.preparedStatePublicationFacts
                             offset:0u atIndex:9u];
            });
        if (!encoded) {
            [encoder endEncoding];
            return false;
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encoded = dispatch(
            state.preparedStateNormalizeApplication, 1u, [&] {
                [encoder setBytes:&applicationDispatch
                           length:sizeof(applicationDispatch) atIndex:0u];
                [encoder setBuffer:state.preparedStateActions
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.preparedStateApplyOutcome
                             offset:0u atIndex:2u];
            });
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        const std::uint32_t preparedEnvironmentCount =
            pass->environmentCount;
        encoded = encoded && dispatch(
            state.preparedStateMaterializeRestoreStatuses,
            pass->environmentCount,
            [&] {
                [encoder setBytes:&preparedEnvironmentCount
                           length:sizeof(preparedEnvironmentCount)
                          atIndex:0u];
                [encoder setBuffer:state.preparedStateActions
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.statuses offset:0u atIndex:2u];
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:3u];
            });
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        const NSUInteger environments = state.dispatch.environmentCount;
        const NSUInteger particleTotal =
            environments * state.dispatch.particleCount;
        const NSUInteger femNodeTotal =
            environments * state.dispatch.femNodeCount;
        const NSUInteger objectTotal =
            environments * state.dispatch.objectCount;
        const NSUInteger femTransactionalTotal = environments * std::max(
            state.dispatch.femNodeCount,
            state.dispatch.tetrahedronCount);
        const NSUInteger topologyTransactionalTotal = environments * std::max({
            static_cast<NSUInteger>(state.dispatch.femNodeCount),
            static_cast<NSUInteger>(state.dispatch.tetrahedronCount),
            static_cast<NSUInteger>(state.dispatch.cohesiveFaceCount),
            static_cast<NSUInteger>(state.dispatch.punctureChannelCount),
            static_cast<NSUInteger>(state.dispatch.objectCount),
            static_cast<NSUInteger>(state.dispatch.tetrahedronCount) * 4u,
        });
        const NSUInteger mixedTransactionalTotal =
            std::max(femNodeTotal, objectTotal);
        const NSUInteger pairTotal =
            environments * state.dispatch.contactPairCount;
        const NSUInteger deformableTotal =
            environments * state.dispatch.deformableContactCapacity;

        encoded = encoded && runtimeDispatch(
            "nm_mpm_rollback_frame", particleTotal, [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.particleAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.particleCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.particleMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.particleMaterialStateCheckpoint offset:0u atIndex:5u];
            });
        encoded = encoded && runtimeDispatch(
            "nm_fem_rollback_frame", femTransactionalTotal, [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femMaterialStateAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femMaterialStateCheckpoint offset:0u atIndex:5u];
            });
        encoded = encoded && runtimeDispatch(
            "nm_topology_rollback", topologyTransactionalTotal, [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.femTetrahedraAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femTetrahedraCheckpoint offset:0u atIndex:3u];
                [encoder setBuffer:state.femTopologyNodesAccepted offset:0u atIndex:4u];
                [encoder setBuffer:state.femTopologyNodesCheckpoint offset:0u atIndex:5u];
                [encoder setBuffer:state.cohesiveFacesAccepted offset:0u atIndex:6u];
                [encoder setBuffer:state.cohesiveFacesCheckpoint offset:0u atIndex:7u];
                [encoder setBuffer:state.punctureChannelsAccepted offset:0u atIndex:8u];
                [encoder setBuffer:state.punctureChannelsCheckpoint offset:0u atIndex:9u];
                [encoder setBuffer:state.topologyStatesAccepted offset:0u atIndex:10u];
                [encoder setBuffer:state.topologyStatesCheckpoint offset:0u atIndex:11u];
                [encoder setBuffer:state.femNodeIncidence offset:0u atIndex:12u];
                [encoder setBuffer:state.femNodeIncidenceCheckpoint offset:0u atIndex:13u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:14u];
                [encoder setBuffer:state.femNodeRangesCheckpoint offset:0u atIndex:15u];
            });
        encoded = encoded && runtimeDispatch(
            "nm_mixed_rollback", mixedTransactionalTotal, [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.femFieldsAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.femFieldsCheckpoint offset:0u atIndex:3u];
            });
        if (state.dispatch.learnedWeightCount != 0u) {
            encoded = encoded && runtimeDispatch(
                "nm_learned_rollback",
                state.dispatch.learnedWeightCount,
                [&] {
                    [encoder setBuffer:state.preparedStateRestoreStatuses
                                 offset:0u atIndex:1u];
                    [encoder setBuffer:state.learnedWeightsAccepted offset:0u atIndex:2u];
                    [encoder setBuffer:state.learnedWeightsCheckpoint offset:0u atIndex:3u];
                    [encoder setBuffer:state.learnedRevisionAccepted offset:0u atIndex:4u];
                    [encoder setBuffer:state.learnedRevisionCheckpoint offset:0u atIndex:5u];
                });
        }
        encoded = encoded && runtimeDispatch(
            "nm_contact_rollback_histories", pairTotal, [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.contactHistoriesAccepted offset:0u atIndex:2u];
                [encoder setBuffer:state.contactHistoriesCheckpoint offset:0u atIndex:3u];
            });
        encoded = encoded && runtimeDispatch(
            "nm_contact_rollback_deformable_contact_histories",
            deformableTotal,
            [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.deformableContactHistoriesAccepted
                             offset:0u atIndex:2u];
                [encoder setBuffer:state.deformableContactHistoriesCandidate
                             offset:0u atIndex:3u];
                [encoder setBuffer:state.deformableContactHistoriesCheckpoint
                             offset:0u atIndex:4u];
            });
        encoded = encoded && runtimeDispatch(
            "nm_scheduler_reconcile", objectTotal, [&] {
                [encoder setBuffer:state.preparedStateRestoreStatuses
                             offset:0u atIndex:1u];
                [encoder setBuffer:state.schedulerCheckpoint offset:0u atIndex:2u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:3u];
                [encoder setBuffer:state.events offset:0u atIndex:4u];
            });

        struct RestoreArena {
            id<MTLBuffer> checkpoint = nil;
            id<MTLBuffer> destination = nil;
            std::uint64_t bytesPerEnvironment = 0u;
            std::uint64_t byteCount = 0u;
            bool shared = false;
        };
        const auto perEnvironmentBytes = [&](const std::uint64_t count,
                                             const std::uint64_t bytes) {
            return count * bytes;
        };
        const std::array<RestoreArena, 7u> restoreArenas{{
            {state.adaptivePreparedCheckpoint, state.adaptive,
             perEnvironmentBytes(state.dispatch.objectCount,
                                 sizeof(NMAdaptiveStateGPU)),
             perEnvironmentBytes(environments * state.dispatch.objectCount,
                                 sizeof(NMAdaptiveStateGPU)), false},
            {state.schedulerPreviousPreparedCheckpoint,
             state.schedulerPrevious,
             perEnvironmentBytes(state.dispatch.objectCount,
                                 sizeof(NMSchedulerStateGPU)),
             perEnvironmentBytes(environments * state.dispatch.objectCount,
                                 sizeof(NMSchedulerStateGPU)), false},
            {state.frameReactionsPreparedCheckpoint, state.frameReactions,
             perEnvironmentBytes(state.dispatch.rigidProxyCount,
                                 sizeof(NMRigidReactionGPU)),
             perEnvironmentBytes(environments * state.dispatch.rigidProxyCount,
                                 sizeof(NMRigidReactionGPU)), false},
            {state.coupledGeneralizedPreparedCheckpoint,
             state.coupledGeneralizedCandidate,
             perEnvironmentBytes(state.dispatch.rigidGeneralizedCapacity,
                                 sizeof(float)),
             perEnvironmentBytes(
                 environments * state.dispatch.rigidGeneralizedCapacity,
                 sizeof(float)), false},
            {state.rigidStatesPreparedCheckpoint, state.rigidStates,
             perEnvironmentBytes(state.dispatch.rigidProxyCount,
                                 sizeof(NMRigidStateGPU)),
             perEnvironmentBytes(environments * state.dispatch.rigidProxyCount,
                                 sizeof(NMRigidStateGPU)), false},
            {state.identificationPreparedCheckpoint,
             state.identificationDistributions, 0u,
             perEnvironmentBytes(state.identificationDistributionCount,
                                 sizeof(NMIdentificationDistributionGPU)),
             true},
            {state.environmentParametersPreparedCheckpoint,
             state.environmentParameters,
             perEnvironmentBytes(state.dispatch.parameterCount, sizeof(float)),
             perEnvironmentBytes(environments * state.dispatch.parameterCount,
                                 sizeof(float)), false},
        }};
        for (const RestoreArena& arena : restoreArenas) {
            if (arena.byteCount == 0u) continue;
            if (arena.checkpoint == nil || arena.destination == nil ||
                arena.checkpoint.length < arena.byteCount ||
                arena.destination.length < arena.byteCount) {
                encoded = false;
                break;
            }
            const std::uint64_t chunkCount64 =
                arena.byteCount / 16u + (arena.byteCount % 16u != 0u);
            if (chunkCount64 > std::numeric_limits<std::uint32_t>::max()) {
                encoded = false;
                break;
            }
            NMPreparedStateRestoreGPU restore{
                .environmentCount = pass->environmentCount,
                .actionStride = 1u,
                .flags = arena.shared
                    ? NM_PREPARED_STATE_RESTORE_SHARED : 0u,
                .bytesPerEnvironment = arena.bytesPerEnvironment,
                .byteCount = arena.byteCount,
            };
            encoded = encoded && dispatch(
                state.preparedStateRestoreBytes,
                static_cast<NSUInteger>(chunkCount64),
                [&] {
                    [encoder setBytes:&restore
                               length:sizeof(restore) atIndex:0u];
                    [encoder setBuffer:arena.checkpoint offset:0u atIndex:1u];
                    [encoder setBuffer:arena.destination offset:0u atIndex:2u];
                    [encoder setBuffer:state.preparedStateActions
                                 offset:0u atIndex:3u];
                });
        }
        [encoder endEncoding];
        if (!encoded) {
            // The reservation deliberately remains until cancel(NotEnqueued).
            // The borrowed command contains partial encoding but has executed
            // nothing, so its immutable proof/checkpoints remain retryable.
            return false;
        }

        const std::weak_ptr<State::CommandOwnership> weakOwnership = ownership;
        void* const applyIdentity = commandBufferPointer;
        id<MTLBuffer> const retainedApplyOutcome =
            state.preparedStateApplyOutcome;
        id<MTLBuffer> const retainedPublicationFacts =
            state.preparedStatePublicationFacts;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (const auto locked = weakOwnership.lock()) {
                const std::lock_guard lock(locked->mutex);
                const auto* outcome = static_cast<const std::uint32_t*>(
                    retainedApplyOutcome.contents);
                const auto* facts = static_cast<
                    const NMPreparedStatePublicationFactsGPU*>(
                        retainedPublicationFacts.contents);
                locked->settleApplication(
                    applyIdentity,
                    completed.status,
                    outcome != nullptr
                        ? *outcome : NM_PREPARED_STATE_APPLY_PENDING,
                    facts);
            }
        }];
        return true;
    }
}

std::uint64_t Runtime::acceptedStateProofProgramFingerprint() const noexcept {
    return state_ != nullptr
        ? state_->acceptedStateProofProgramFingerprint
        : 0u;
}

std::size_t Runtime::acceptedStateProofResidentBytes() const noexcept {
    return state_ != nullptr
        ? state_->acceptedStateProofResidentByteCount
        : 0u;
}

PreparedStateDisposition Runtime::preparedStateDisposition(
    const PreparedStateDispositionIdentity& identity
) const noexcept {
    @autoreleasepool {
        if (state_ == nullptr) return PreparedStateDisposition::unknown;
        try {
            const auto ownership = state_->commandOwnership;
            const std::lock_guard lock(ownership->mutex);
            if (!ownership->matchesDispositionIdentity(identity)) {
                return PreparedStateDisposition::unknown;
            }
            if (ownership->disposition ==
                    PreparedStateDisposition::applying &&
                ownership->applyCommandBuffer != nullptr) {
                void* const applyIdentity =
                    ownership->applyCommandBuffer;
                __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
                    (__bridge id<MTLCommandBuffer>)applyIdentity;
                const auto* facts = static_cast<const
                    NMPreparedStatePublicationFactsGPU*>(
                        state_->preparedStatePublicationFacts.contents);
                if (commandBuffer == nil) {
                    ownership->settleApplication(
                        applyIdentity,
                        MTLCommandBufferStatusError,
                        NM_PREPARED_STATE_APPLY_PENDING,
                        facts);
                } else if (commandBuffer.status ==
                           MTLCommandBufferStatusCompleted) {
                    const auto* outcome = static_cast<const std::uint32_t*>(
                        state_->preparedStateApplyOutcome.contents);
                    ownership->settleApplication(
                        applyIdentity,
                        commandBuffer.status,
                        outcome != nullptr
                            ? *outcome
                            : NM_PREPARED_STATE_APPLY_PENDING,
                        facts);
                } else if (commandBuffer.status ==
                           MTLCommandBufferStatusError) {
                    ownership->settleApplication(
                        applyIdentity,
                        commandBuffer.status,
                        NM_PREPARED_STATE_APPLY_PENDING,
                        facts);
                }
            }
            return ownership->disposition;
        } catch (...) {
            return PreparedStateDisposition::unknown;
        }
    }
}

bool Runtime::reservePublishedRoot(
    const PreparedStateDispositionIdentity& identity,
    const PreparedStatePublicationBinding& binding,
    PreparedStatePublicationReservation& reservation
) noexcept {
    reservation = {};
    if (state_ == nullptr || identity.abiVersion != 1u ||
        identity.structSize != sizeof(PreparedStateDispositionIdentity) ||
        identity.reserved0 != 0u || identity.reserved1 != 0u ||
        binding.abiVersion != 1u ||
        binding.structSize != sizeof(PreparedStatePublicationBinding) ||
        binding.reserved0 != 0u || binding.reserved1 != 0u ||
        binding.physicsTokenFingerprint == 0u ||
        binding.brainProgramFingerprint == 0u ||
        binding.brainShadowStateFingerprint == 0u ||
        binding.brainWitnessFingerprint == 0u ||
        binding.matterApplyFingerprint == 0u ||
        binding.appliedDecisionFingerprint == 0u ||
        binding.jointCommitFingerprint == 0u ||
        binding.brainGeneration == 0u) {
        return false;
    }
    try {
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (!ownership->matchesDispositionIdentity(identity) ||
            ownership->disposition !=
                PreparedStateDisposition::acceptedPendingPublication ||
            !ownership->preparedStateOpen || ownership->restoreRequired ||
            ownership->terminalNoTouch || ownership->applyEncoded ||
            ownership->applyCommandBuffer != nullptr ||
            !ownership->preparedCommandCompleted ||
            ownership->preparedCommandFailed ||
            ownership->publicationReserved) {
            return false;
        }
        const NMPreparedStatePublicationFactsGPU& facts =
            ownership->publicationFacts;
        if (facts.abiVersion != NM_MATTER_OWNER_APPLY_ABI_VERSION ||
            facts.status != NM_PREPARED_STATE_PUBLICATION_FACTS_VALID ||
            facts.reserved0 != 0u || facts.reserved1 != 0u ||
            facts.factsFingerprint == 0u ||
            facts.factsFingerprint != publicationFactsFingerprint(facts) ||
            binding.physicsTokenFingerprint !=
                facts.physicsTokenFingerprint ||
            binding.brainProgramFingerprint !=
                facts.brainProgramFingerprint ||
            binding.brainShadowStateFingerprint !=
                facts.brainShadowStateFingerprint ||
            binding.brainWitnessFingerprint !=
                facts.brainWitnessFingerprint ||
            binding.matterApplyFingerprint !=
                facts.matterApplyFingerprint) {
            return false;
        }
        ownership->publicationBinding = binding;
        PreparedStatePublicationReservation capability{};
        capability.transactionSlot = identity.transactionSlot;
        capability.transactionFingerprint = identity.transactionFingerprint;
        capability.slotGeneration = identity.slotGeneration;
        ++ownership->publicationReservationCounter;
        if (ownership->publicationReservationCounter == 0u) {
            ++ownership->publicationReservationCounter;
        }
        std::uint64_t nonce = ownership->publicationReservationCounter;
        nonce = mixFingerprint(nonce, identity.ownerProgramFingerprint);
        nonce = mixFingerprint(nonce, identity.linearizationEpoch);
        nonce = mixFingerprint(nonce, binding.appliedDecisionFingerprint);
        nonce = mixFingerprint(nonce, binding.jointCommitFingerprint);
        capability.reservationNonce = nonce == 0u
            ? 14695981039346656037ull : nonce;
        capability.reservationFingerprint =
            publicationReservationFingerprint(capability);
        ownership->publicationReservation = capability;
        ownership->publicationReserved = true;
        reservation = capability;
        return true;
    } catch (...) {
        return false;
    }
}

bool Runtime::releasePublishedRoot(
    const PreparedStatePublicationReservation& reservation,
    const PreparedStatePublicationFence& fence
) noexcept {
    if (state_ == nullptr || reservation.abiVersion != 1u ||
        reservation.structSize !=
            sizeof(PreparedStatePublicationReservation) ||
        reservation.reserved0 != 0u || reservation.reserved1 != 0u ||
        reservation.reserved2 != 0u ||
        reservation.transactionFingerprint == 0u ||
        reservation.slotGeneration == 0u ||
        reservation.reservationNonce == 0u ||
        reservation.reservationFingerprint == 0u ||
        reservation.reservationFingerprint !=
            publicationReservationFingerprint(reservation)) {
        return false;
    }
    try {
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        // A stale capability is not the owning release attempt. It cannot
        // consume or poison the current exact reservation.
        if (std::memcmp(
                &reservation, &ownership->publicationReservation,
                sizeof(reservation)) != 0) {
            return false;
        }
        const PreparedStateDispositionIdentity& identity =
            ownership->dispositionIdentity;
        if (ownership->disposition !=
                PreparedStateDisposition::acceptedPendingPublication ||
            !ownership->preparedStateOpen || ownership->restoreRequired ||
            ownership->terminalNoTouch || ownership->applyEncoded ||
            ownership->applyCommandBuffer != nullptr ||
            !ownership->preparedCommandCompleted ||
            ownership->preparedCommandFailed ||
            !ownership->publicationReserved) {
            return false;
        }
        const PreparedStatePublicationBinding expected =
            ownership->publicationBinding;
        // An exact-identity release is one shot. Any unexpected fence after
        // Brain publication cannot be retried or restored safely; preserve
        // the reservation itself as immutable quarantine evidence.
        const bool validFence =
            fence.abiVersion == NM_MATTER_PUBLICATION_FENCE_ABI_VERSION &&
            fence.structBytes == sizeof(PreparedStatePublicationFence) &&
            fence.status == NM_JOINT_PUBLICATION_COMMITTED &&
            fence.environment == 0u && fence.reserved0 == 0u &&
            fence.controlStep == identity.controlStep &&
            fence.substepIndex == identity.physicsSubstep &&
            fence.physicsSubstepCount == identity.physicsSubstepCount &&
            fence.ownerProgramFingerprint ==
                identity.ownerProgramFingerprint &&
            fence.transactionFingerprint == identity.transactionFingerprint &&
            fence.linearizationEpoch == identity.linearizationEpoch &&
            fence.slotGeneration == identity.slotGeneration &&
            fence.physicsTokenFingerprint ==
                expected.physicsTokenFingerprint &&
            fence.brainProgramFingerprint ==
                expected.brainProgramFingerprint &&
            fence.brainShadowStateFingerprint ==
                expected.brainShadowStateFingerprint &&
            fence.brainWitnessFingerprint ==
                expected.brainWitnessFingerprint &&
            fence.appliedDecisionFingerprint ==
                expected.appliedDecisionFingerprint &&
            fence.jointCommitFingerprint ==
                expected.jointCommitFingerprint &&
            fence.brainGeneration == expected.brainGeneration &&
            fence.fenceFingerprint != 0u &&
            fence.fenceFingerprint == publicationFenceFingerprint(fence);
        if (!validFence) {
            ownership->terminalNoTouch = true;
            ownership->restoreRequired = true;
            ownership->acceptedStateProofEligible = false;
            ownership->disposition =
                PreparedStateDisposition::terminalNoTouch;
            return false;
        }
        ownership->publicationReserved = false;
        ownership->publicationBinding = {};
        ownership->publicationReservation = {};
        ownership->publicationFacts = {};
        ownership->disposition = PreparedStateDisposition::resolved;
        ownership->preparedStateOpen = false;
        ownership->restoreRequired = false;
        ownership->preparedStateRequested = false;
        ownership->preparedCommandBuffer = nullptr;
        ownership->preparedEnvironmentStatuses = nullptr;
        ownership->preparedEnvironmentStatusesGPUAddress = 0u;
        ownership->acceptedStateProofEncoded = false;
        ownership->acceptedStateProofEligible = false;
        ownership->transactionPolicyFingerprint = 0u;
        ownership->identificationAdvanced = false;
        ownership->preparedCommandCompleted = false;
        ownership->preparedCommandFailed = false;
        ownership->terminalNoTouch = false;
        ownership->applyCommandCompleted = false;
        return true;
    } catch (...) {
        return false;
    }
}

void Runtime::cancel(void* commandBuffer) noexcept {
    if (state_ == nullptr || commandBuffer == nullptr) {
        return;
    }
    try {
        __unsafe_unretained id<MTLCommandBuffer> borrowed =
            (__bridge id<MTLCommandBuffer>)commandBuffer;
        if (borrowed == nil ||
            borrowed.status != MTLCommandBufferStatusNotEnqueued) {
            return;
        }
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (ownership->applyCommandBuffer == commandBuffer) {
            ownership->applyCommandBuffer = nullptr;
            ownership->applyEncoded = false;
            ownership->applyCommandCompleted = false;
            ownership->disposition = PreparedStateDisposition::prepared;
            // `borrowed` is still NotEnqueued and the caller's cancel
            // contract abandons it permanently. No encoded apply can
            // have executed, so the immutable prepared checkpoints/proof
            // remain valid for a later witness-validation attempt.  A GPU
            // error after commit follows the completion-handler path instead
            // and still requires a fresh forced restore.
            return;
        }
        if (ownership->activeCommandBuffer == commandBuffer) {
            if (ownership->identificationAdvanced) {
                ownership->identificationGeneration =
                    ownership->identificationCheckpoint;
            }
            ownership->identificationAdvanced = false;
            ownership->activeCommandBuffer = nullptr;
            ownership->preDynamicsOpen = false;
            ownership->acceptedStateProofEncoded = false;
            ownership->acceptedStateProofEligible = false;
            ownership->transactionPolicyFingerprint = 0u;
            ownership->preparedStateRequested = false;
            ownership->preparedStateOpen = false;
            ownership->restoreRequired = false;
            ownership->preparedCommandBuffer = nullptr;
            ownership->preparedEnvironmentStatuses = nullptr;
            ownership->preparedEnvironmentStatusesGPUAddress = 0u;
            ownership->preparedCommandCompleted = false;
            ownership->preparedCommandFailed = false;
            ownership->terminalNoTouch = false;
            ownership->applyCommandCompleted = false;
            ownership->controlStep = 0u;
            ownership->physicsSubstep = 0u;
            ownership->dispositionIdentity = {};
            ownership->disposition = PreparedStateDisposition::unknown;
        }
    } catch (...) {
    }
}

TopologyGrowthRequest Runtime::pendingTopologyGrowth() const noexcept {
    if (state_ == nullptr) return {};
    try {
        const auto growth = state_->growthOwnership;
        const std::lock_guard lock(growth->mutex);
        return growth->pending;
    } catch (...) {
        return {};
    }
}

RuntimeDiagnostics Runtime::encodeTopologyGrowth(
    void* commandBufferPointer,
    const Runtime& source
) {
    @autoreleasepool {
        RuntimeDiagnostics diagnostics;
        if (state_ == nullptr || source.state_ == nullptr ||
            commandBufferPointer == nullptr) {
            diagnostics.message =
                "topology growth migration requires two initialized runtimes and a borrowed command buffer";
            return diagnostics;
        }
        State& destination = *state_;
        const State& previous = *source.state_;
        TopologyGrowthRequest growthRequest;
        {
            const std::lock_guard lock(previous.growthOwnership->mutex);
            growthRequest = previous.growthOwnership->pending;
        }
        if (!growthRequest.required) {
            diagnostics.message =
                "topology growth migration requires a completed source growth request";
            return diagnostics;
        }
        if (destination.sourcePhysicsFingerprint == 0u ||
            destination.sourcePhysicsFingerprint !=
                previous.sourcePhysicsFingerprint) {
            diagnostics.message =
                "topology growth destination does not match source authored physics";
            return diagnostics;
        }
        if (destination.device.registryID != previous.device.registryID ||
            destination.dispatch.environmentCount !=
                previous.dispatch.environmentCount ||
            destination.objectLayout.size() != previous.objectLayout.size() ||
            destination.dispatch.materialStateStride !=
                previous.dispatch.materialStateStride) {
            diagnostics.message =
                "topology growth destination is not layout-compatible with the source runtime";
            return diagnostics;
        }
        for (std::size_t object = 0u;
             object < previous.objectLayout.size(); ++object) {
            if (destination.objectLayout[object].representation !=
                    previous.objectLayout[object].representation ||
                destination.objectLayout[object].stateCount <
                    previous.objectLayout[object].stateCount ||
                destination.objectLayout[object].elementCount <
                    previous.objectLayout[object].elementCount ||
                destination.capacityLayout[object].topology.z <
                    previous.capacityLayout[object].topology.z ||
                destination.capacityLayout[object].topology.w <
                    previous.capacityLayout[object].topology.w) {
                diagnostics.message =
                    "topology growth destination did not geometrically contain every source arena";
                return diagnostics;
            }
        }
        if (destination.dispatch.femNodeCount < growthRequest.nodes ||
            destination.dispatch.tetrahedronCount < growthRequest.tetrahedra ||
            destination.dispatch.cohesiveFaceCount < growthRequest.cohesiveFaces ||
            destination.dispatch.punctureChannelCount < growthRequest.punctureChannels ||
            destination.dispatch.mutationCommandCount < growthRequest.mutationCommands ||
            destination.dispatch.contactPairCount < growthRequest.rigidContacts ||
            destination.dispatch.deformableContactCapacity <
                growthRequest.deformableContacts) {
            diagnostics.message =
                "topology growth destination does not satisfy the completed growth request";
            return diagnostics;
        }
        std::uint32_t previousIdentificationGeneration = 0u;
        std::uint32_t previousIdentificationCheckpoint = 0u;
        {
            std::scoped_lock lock(
                destination.commandOwnership->mutex,
                previous.commandOwnership->mutex);
            if (destination.commandOwnership->activeCommandBuffer != nullptr ||
                previous.commandOwnership->activeCommandBuffer != nullptr ||
                destination.commandOwnership->preparedStateOpen ||
                previous.commandOwnership->preparedStateOpen ||
                destination.commandOwnership->restoreRequired ||
                previous.commandOwnership->restoreRequired) {
                diagnostics.message =
                    "topology growth migration must be encoded between completed submissions";
                return diagnostics;
            }
            previousIdentificationGeneration =
                previous.commandOwnership->identificationGeneration;
            previousIdentificationCheckpoint =
                previous.commandOwnership->identificationCheckpoint;
            destination.commandOwnership->activeCommandBuffer =
                commandBufferPointer;
            previous.commandOwnership->activeCommandBuffer =
                commandBufferPointer;
        }
        id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)commandBufferPointer;
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit == nil) {
            std::scoped_lock lock(
                destination.commandOwnership->mutex,
                previous.commandOwnership->mutex);
            destination.commandOwnership->activeCommandBuffer = nullptr;
            previous.commandOwnership->activeCommandBuffer = nullptr;
            diagnostics.message =
                "failed to create topology growth migration encoder";
            return diagnostics;
        }
        [blit setLabel:@"Numi Matter topology growth migration"];
        const auto copyBytes = ^(
            id<MTLBuffer> input,
            const NSUInteger inputOffset,
            id<MTLBuffer> output,
            const NSUInteger outputOffset,
            const NSUInteger bytes
        ) {
            if (bytes != 0u)
                [blit copyFromBuffer:input sourceOffset:inputOffset
                            toBuffer:output destinationOffset:outputOffset
                                size:bytes];
        };
        const auto copyWholeAccepted = ^(
            id<MTLBuffer> input,
            id<MTLBuffer> accepted,
            id<MTLBuffer> candidate,
            id<MTLBuffer> checkpoint
        ) {
            const NSUInteger bytes = std::min(input.length, accepted.length);
            copyBytes(input, 0u, accepted, 0u, bytes);
            copyBytes(input, 0u, candidate, 0u,
                std::min(input.length, candidate.length));
            copyBytes(input, 0u, checkpoint, 0u,
                std::min(input.length, checkpoint.length));
        };
        copyWholeAccepted(previous.particleAccepted,
            destination.particleAccepted, destination.particleCandidate,
            destination.particleCheckpoint);
        copyWholeAccepted(previous.particleMaterialStateAccepted,
            destination.particleMaterialStateAccepted,
            destination.particleMaterialStateCandidate,
            destination.particleMaterialStateCheckpoint);
        copyBytes(previous.adaptive, 0u, destination.adaptive, 0u,
            std::min(previous.adaptive.length, destination.adaptive.length));
        copyBytes(previous.schedulers, 0u, destination.schedulers, 0u,
            std::min(previous.schedulers.length,
                     destination.schedulers.length));
        copyBytes(previous.environmentParameters, 0u,
            destination.environmentParameters, 0u,
            std::min(previous.environmentParameters.length,
                     destination.environmentParameters.length));
        copyBytes(previous.identificationDistributions, 0u,
            destination.identificationDistributions, 0u,
            std::min(previous.identificationDistributions.length,
                     destination.identificationDistributions.length));
        copyWholeAccepted(previous.learnedWeightsAccepted,
            destination.learnedWeightsAccepted,
            destination.learnedWeightsCandidate,
            destination.learnedWeightsCheckpoint);
        copyWholeAccepted(previous.learnedRevisionAccepted,
            destination.learnedRevisionAccepted,
            destination.learnedRevisionCandidate,
            destination.learnedRevisionCheckpoint);

        const NSUInteger environments = previous.dispatch.environmentCount;
        const NSUInteger stateStride = previous.dispatch.materialStateStride;
        for (NSUInteger environment = 0u; environment < environments;
             ++environment) {
            NSUInteger sourceFaceOffset = 0u;
            NSUInteger destinationFaceOffset = 0u;
            NSUInteger sourceChannelOffset = 0u;
            NSUInteger destinationChannelOffset = 0u;
            for (std::size_t objectIndex = 0u;
                 objectIndex < previous.objectLayout.size(); ++objectIndex) {
                const NMContinuumObjectGPU sourceObject =
                    previous.objectLayout[objectIndex];
                const NMContinuumObjectGPU destinationObject =
                    destination.objectLayout[objectIndex];
                const NSUInteger sourceNode =
                    environment * previous.dispatch.femNodeCount +
                    sourceObject.stateOffset;
                const NSUInteger destinationNode =
                    environment * destination.dispatch.femNodeCount +
                    destinationObject.stateOffset;
                const NSUInteger nodeCount = sourceObject.stateCount;
                const NSUInteger sourceTet =
                    environment * previous.dispatch.tetrahedronCount +
                    sourceObject.elementOffset;
                const NSUInteger destinationTet =
                    environment * destination.dispatch.tetrahedronCount +
                    destinationObject.elementOffset;
                const NSUInteger tetCount = sourceObject.elementCount;
                const auto copyNodeArena = [&](id<MTLBuffer> input,
                                               id<MTLBuffer> output,
                                               const NSUInteger elementSize) {
                    copyBytes(input, sourceNode * elementSize, output,
                        destinationNode * elementSize, nodeCount * elementSize);
                };
                for (id<MTLBuffer> output : @[
                         destination.femAccepted,
                         destination.femCandidate,
                         destination.femCheckpoint])
                    copyNodeArena(previous.femAccepted, output,
                        sizeof(NMFEMNodeStateGPU));
                for (id<MTLBuffer> output : @[
                         destination.femFieldsAccepted,
                         destination.femFieldsCandidate,
                         destination.femFieldsCheckpoint])
                    copyNodeArena(previous.femFieldsAccepted, output,
                        sizeof(NMFEMFieldStateGPU));
                for (id<MTLBuffer> output : @[
                         destination.femTopologyNodesAccepted,
                         destination.femTopologyNodesCandidate,
                         destination.femTopologyNodesCheckpoint])
                    copyNodeArena(previous.femTopologyNodesAccepted, output,
                        sizeof(NMFEMTopologyNodeGPU));
                const auto copyTetArena = [&](id<MTLBuffer> input,
                                              id<MTLBuffer> output,
                                              const NSUInteger elementSize) {
                    copyBytes(input, sourceTet * elementSize, output,
                        destinationTet * elementSize, tetCount * elementSize);
                };
                for (id<MTLBuffer> output : @[
                         destination.femTetrahedraAccepted,
                         destination.femTetrahedraCandidate,
                         destination.femTetrahedraCheckpoint])
                    copyTetArena(previous.femTetrahedraAccepted, output,
                        sizeof(NMTetrahedronGPU));
                if (stateStride != 0u) {
                    const NSUInteger scalar = sizeof(float);
                    for (id<MTLBuffer> output : @[
                             destination.femMaterialStateAccepted,
                             destination.femMaterialStateCandidate,
                             destination.femMaterialStateCheckpoint])
                        copyBytes(previous.femMaterialStateAccepted,
                            sourceTet * stateStride * scalar, output,
                            destinationTet * stateStride * scalar,
                            tetCount * stateStride * scalar);
                }
                const NSUInteger oldFaces =
                    previous.capacityLayout[objectIndex].topology.z;
                const NSUInteger newFaces =
                    destination.capacityLayout[objectIndex].topology.z;
                const NSUInteger oldChannels =
                    previous.capacityLayout[objectIndex].topology.w;
                const NSUInteger newChannels =
                    destination.capacityLayout[objectIndex].topology.w;
                const NSUInteger sourceFace = environment *
                    previous.dispatch.cohesiveFaceCount + sourceFaceOffset;
                const NSUInteger destinationFace = environment *
                    destination.dispatch.cohesiveFaceCount +
                    destinationFaceOffset;
                for (id<MTLBuffer> output : @[
                         destination.cohesiveFacesAccepted,
                         destination.cohesiveFacesCandidate,
                         destination.cohesiveFacesCheckpoint])
                    copyBytes(previous.cohesiveFacesAccepted,
                        sourceFace * sizeof(NMCohesiveFaceGPU), output,
                        destinationFace * sizeof(NMCohesiveFaceGPU),
                        oldFaces * sizeof(NMCohesiveFaceGPU));
                const NSUInteger sourceChannel = environment *
                    previous.dispatch.punctureChannelCount +
                    sourceChannelOffset;
                const NSUInteger destinationChannel = environment *
                    destination.dispatch.punctureChannelCount +
                    destinationChannelOffset;
                for (id<MTLBuffer> output : @[
                         destination.punctureChannelsAccepted,
                         destination.punctureChannelsCandidate,
                         destination.punctureChannelsCheckpoint])
                    copyBytes(previous.punctureChannelsAccepted,
                        sourceChannel * sizeof(NMPunctureChannelGPU), output,
                        destinationChannel * sizeof(NMPunctureChannelGPU),
                        oldChannels * sizeof(NMPunctureChannelGPU));
                sourceFaceOffset += oldFaces;
                destinationFaceOffset += newFaces;
                sourceChannelOffset += oldChannels;
                destinationChannelOffset += newChannels;
            }
        }
        copyWholeAccepted(previous.topologyStatesAccepted,
            destination.topologyStatesAccepted,
            destination.topologyStatesCandidate,
            destination.topologyStatesCheckpoint);
        [blit endEncoding];

        id<MTLComputeCommandEncoder> growthEncoder =
            [commandBuffer computeCommandEncoder];
        if (growthEncoder == nil) {
            std::scoped_lock lock(
                destination.commandOwnership->mutex,
                previous.commandOwnership->mutex);
            destination.commandOwnership->activeCommandBuffer = nullptr;
            previous.commandOwnership->activeCommandBuffer = nullptr;
            diagnostics.message =
                "failed to create topology growth rebuild encoder";
            return diagnostics;
        }
        [growthEncoder setLabel:@"Numi Matter topology growth rebuild"];
        bool growthEncodingValid = true;
        const auto growthDispatch = &destination.dispatch;
        const auto dispatchGrowthThreads = [&](
            const char* name,
            const NSUInteger count,
            const auto& bind
        ) {
            if (count == 0u || !growthEncodingValid) return;
            id<MTLComputePipelineState> pipeline = destination.pipeline(name);
            if (pipeline == nil) {
                growthEncodingValid = false;
                return;
            }
            [growthEncoder setComputePipelineState:pipeline];
            bind();
            const NSUInteger width = std::min<NSUInteger>(
                256u, pipeline.maxTotalThreadsPerThreadgroup);
            [growthEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                     threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        };
        const auto setGrowthDispatch = [&]() {
            [growthEncoder setBytes:growthDispatch
                             length:sizeof(*growthDispatch)
                            atIndex:0u];
        };
        const NSUInteger destinationEnvironments =
            destination.dispatch.environmentCount;
        const NSUInteger destinationNodes = destinationEnvironments *
            destination.dispatch.femNodeCount;
        const NSUInteger topologyWidth = destinationEnvironments * std::max({
            static_cast<NSUInteger>(destination.dispatch.femNodeCount),
            static_cast<NSUInteger>(destination.dispatch.tetrahedronCount),
            static_cast<NSUInteger>(destination.dispatch.cohesiveFaceCount),
            static_cast<NSUInteger>(destination.dispatch.objectCount)});
        const NSUInteger topologyReferenceWidth =
            destinationEnvironments * std::max(
                static_cast<NSUInteger>(
                    destination.dispatch.tetrahedronCount),
                static_cast<NSUInteger>(
                    destination.dispatch.cohesiveFaceCount));
        const std::uint32_t allocationGeneration =
            growthRequest.allocationGeneration;
        dispatchGrowthThreads(
            "nm_topology_validate_growth_references",
            topologyReferenceWidth, [&] {
                setGrowthDispatch();
                [growthEncoder setBuffer:previous.objects
                                  offset:0u atIndex:1u];
                [growthEncoder setBuffer:destination.femTetrahedraAccepted
                                  offset:0u atIndex:2u];
                [growthEncoder setBuffer:destination.cohesiveFacesAccepted
                                  offset:0u atIndex:3u];
                [growthEncoder setBuffer:destination.statuses
                                  offset:0u atIndex:4u];
            });
        dispatchGrowthThreads(
            "nm_topology_mark_growth_generation", topologyWidth, [&] {
                setGrowthDispatch();
                [growthEncoder setBytes:&allocationGeneration
                                 length:sizeof(allocationGeneration)
                                atIndex:1u];
                [growthEncoder setBuffer:destination.femTopologyNodesAccepted
                                  offset:0u atIndex:2u];
                [growthEncoder setBuffer:destination.femTetrahedraAccepted
                                  offset:0u atIndex:3u];
                [growthEncoder setBuffer:destination.topologyStatesAccepted
                                  offset:0u atIndex:4u];
                [growthEncoder setBuffer:previous.objects
                                  offset:0u atIndex:5u];
                [growthEncoder setBuffer:destination.objects
                                  offset:0u atIndex:6u];
                [growthEncoder setBuffer:destination.cohesiveFacesAccepted
                                  offset:0u atIndex:7u];
                [growthEncoder setBuffer:destination.statuses
                                  offset:0u atIndex:8u];
            });
        dispatchGrowthThreads("nm_topology_rebuild_clear", destinationNodes, [&] {
            setGrowthDispatch();
            [growthEncoder setBuffer:destination.femAccepted offset:0u atIndex:1u];
            [growthEncoder setBuffer:destination.femNodeRanges offset:0u atIndex:2u];
            [growthEncoder setBuffer:destination.topologyIncidenceCursors
                              offset:0u atIndex:3u];
            [growthEncoder setBuffer:destination.statuses
                              offset:0u atIndex:4u];
        });
        dispatchGrowthThreads(
            "nm_topology_rebuild_count_mass", destinationNodes, [&] {
                setGrowthDispatch();
                [growthEncoder setBuffer:destination.materials offset:0u atIndex:1u];
                [growthEncoder setBuffer:destination.femTetrahedraAccepted
                                  offset:0u atIndex:2u];
                [growthEncoder setBuffer:destination.femAccepted offset:0u atIndex:3u];
                [growthEncoder setBuffer:destination.femNodeRanges offset:0u atIndex:4u];
                [growthEncoder setBuffer:destination.statuses
                                  offset:0u atIndex:5u];
            });
        if (destinationEnvironments != 0u && growthEncodingValid) {
            id<MTLComputePipelineState> prefix =
                destination.pipeline("nm_topology_rebuild_prefix");
            if (prefix == nil) {
                growthEncodingValid = false;
            } else {
                [growthEncoder setComputePipelineState:prefix];
                setGrowthDispatch();
                [growthEncoder setBuffer:destination.femNodeRanges
                                  offset:0u atIndex:1u];
                [growthEncoder setBuffer:destination.topologyIncidenceCursors
                                  offset:0u atIndex:2u];
                [growthEncoder setBuffer:destination.statuses
                                  offset:0u atIndex:3u];
                [growthEncoder dispatchThreadgroups:
                    MTLSizeMake(destinationEnvironments, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            }
        }
        dispatchGrowthThreads(
            "nm_topology_rebuild_scatter", destinationNodes, [&] {
                setGrowthDispatch();
                [growthEncoder setBuffer:destination.femTetrahedraAccepted
                                  offset:0u atIndex:1u];
                [growthEncoder setBuffer:destination.femNodeRanges
                                  offset:0u atIndex:2u];
                [growthEncoder setBuffer:destination.topologyIncidenceCursors
                                  offset:0u atIndex:3u];
                [growthEncoder setBuffer:destination.femNodeIncidence
                                  offset:0u atIndex:4u];
                [growthEncoder setBuffer:destination.statuses
                                  offset:0u atIndex:5u];
            });
        dispatchGrowthThreads(
            "nm_topology_rebuild_finalize", destinationNodes, [&] {
                setGrowthDispatch();
                [growthEncoder setBuffer:destination.femAccepted offset:0u atIndex:1u];
                [growthEncoder setBuffer:destination.femTopologyNodesAccepted
                                  offset:0u atIndex:2u];
                [growthEncoder setBuffer:destination.femNodeRanges
                                  offset:0u atIndex:3u];
                [growthEncoder setBuffer:destination.topologyStatesAccepted
                                  offset:0u atIndex:4u];
                [growthEncoder setBuffer:destination.statuses
                                  offset:0u atIndex:5u];
            });
        [growthEncoder endEncoding];
        if (!growthEncodingValid) {
            std::scoped_lock lock(
                destination.commandOwnership->mutex,
                previous.commandOwnership->mutex);
            destination.commandOwnership->activeCommandBuffer = nullptr;
            previous.commandOwnership->activeCommandBuffer = nullptr;
            diagnostics.message =
                "topology growth rebuild pipeline is unavailable";
            return diagnostics;
        }

        id<MTLBlitCommandEncoder> finalizeBlit =
            [commandBuffer blitCommandEncoder];
        if (finalizeBlit == nil) {
            std::scoped_lock lock(
                destination.commandOwnership->mutex,
                previous.commandOwnership->mutex);
            destination.commandOwnership->activeCommandBuffer = nullptr;
            previous.commandOwnership->activeCommandBuffer = nullptr;
            diagnostics.message =
                "failed to create topology growth finalization encoder";
            return diagnostics;
        }
        [finalizeBlit setLabel:@"Numi Matter topology growth finalize"];
        const auto mirrorAccepted = ^(
            id<MTLBuffer> accepted,
            id<MTLBuffer> candidate,
            id<MTLBuffer> checkpoint
        ) {
            const NSUInteger candidateBytes = std::min(
                accepted.length, candidate.length);
            const NSUInteger checkpointBytes = std::min(
                accepted.length, checkpoint.length);
            if (candidateBytes != 0u)
                [finalizeBlit copyFromBuffer:accepted sourceOffset:0u
                                    toBuffer:candidate destinationOffset:0u
                                        size:candidateBytes];
            if (checkpointBytes != 0u)
                [finalizeBlit copyFromBuffer:accepted sourceOffset:0u
                                    toBuffer:checkpoint destinationOffset:0u
                                        size:checkpointBytes];
        };
        mirrorAccepted(destination.femAccepted,
            destination.femCandidate, destination.femCheckpoint);
        mirrorAccepted(destination.femTopologyNodesAccepted,
            destination.femTopologyNodesCandidate,
            destination.femTopologyNodesCheckpoint);
        mirrorAccepted(destination.femTetrahedraAccepted,
            destination.femTetrahedraCandidate,
            destination.femTetrahedraCheckpoint);
        mirrorAccepted(destination.cohesiveFacesAccepted,
            destination.cohesiveFacesCandidate,
            destination.cohesiveFacesCheckpoint);
        mirrorAccepted(destination.topologyStatesAccepted,
            destination.topologyStatesCandidate,
            destination.topologyStatesCheckpoint);
        const NSUInteger incidenceBytes = std::min(
            destination.femNodeIncidence.length,
            destination.femNodeIncidenceCheckpoint.length);
        if (incidenceBytes != 0u)
            [finalizeBlit copyFromBuffer:destination.femNodeIncidence
                            sourceOffset:0u
                                toBuffer:destination.femNodeIncidenceCheckpoint
                       destinationOffset:0u size:incidenceBytes];
        const NSUInteger rangeBytes = std::min(
            destination.femNodeRanges.length,
            destination.femNodeRangesCheckpoint.length);
        if (rangeBytes != 0u)
            [finalizeBlit copyFromBuffer:destination.femNodeRanges
                            sourceOffset:0u
                                toBuffer:destination.femNodeRangesCheckpoint
                       destinationOffset:0u size:rangeBytes];
        [finalizeBlit endEncoding];
        const std::weak_ptr<State::CommandOwnership> weakDestination =
            destination.commandOwnership;
        const std::weak_ptr<State::CommandOwnership> weakSource =
            previous.commandOwnership;
        const std::weak_ptr<State::GrowthOwnership> weakDestinationGrowth =
            destination.growthOwnership;
        const std::weak_ptr<State::GrowthOwnership> weakSourceGrowth =
            previous.growthOwnership;
        void* const borrowedIdentity = commandBufferPointer;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (const auto owner = weakDestination.lock()) {
                const std::lock_guard lock(owner->mutex);
                if (owner->activeCommandBuffer == borrowedIdentity) {
                    if (completed.status == MTLCommandBufferStatusCompleted) {
                        owner->identificationGeneration =
                            previousIdentificationGeneration;
                        owner->identificationCheckpoint =
                            previousIdentificationCheckpoint;
                        owner->identificationAdvanced = false;
                    }
                    owner->activeCommandBuffer = nullptr;
                }
            }
            if (const auto owner = weakSource.lock()) {
                const std::lock_guard lock(owner->mutex);
                if (owner->activeCommandBuffer == borrowedIdentity)
                    owner->activeCommandBuffer = nullptr;
            }
            if (completed.status == MTLCommandBufferStatusCompleted) {
                if (const auto growth = weakDestinationGrowth.lock()) {
                    const std::lock_guard lock(growth->mutex);
                    growth->pending = {};
                }
                if (const auto growth = weakSourceGrowth.lock()) {
                    const std::lock_guard lock(growth->mutex);
                    if (growth->pending.allocationGeneration ==
                        allocationGeneration)
                        growth->pending = {};
                }
            }
        }];
        diagnostics.encoded = true;
        diagnostics.residentBytes = destination.residentBytes;
        diagnostics.device = nsString(destination.device.name);
        diagnostics.message =
            "Numi Matter topology growth migration encoded into borrowed command buffer";
        return diagnostics;
    }
}

RuntimeDiagnostics Runtime::encodeTopologyGrowth(
    void* commandBufferPointer,
    const Runtime& source,
    const CompiledWorld& expandedWorld,
    const RuntimeConfiguration& configuration
) {
    RuntimeDiagnostics diagnostics;
    if (state_ != nullptr) {
        diagnostics.message =
            "allocation-owning topology growth requires an empty destination runtime";
        return diagnostics;
    }
    if (source.state_ == nullptr || commandBufferPointer == nullptr) {
        diagnostics.message =
            "topology growth allocation requires an initialized source and borrowed command buffer";
        return diagnostics;
    }
    RuntimeDiagnostics allocation = initialize(expandedWorld, configuration);
    if (!allocation.encoded) return allocation;
    diagnostics = encodeTopologyGrowth(commandBufferPointer, source);
    if (!diagnostics.encoded && diagnostics.device.empty()) {
        diagnostics.device = allocation.device;
        diagnostics.residentBytes = allocation.residentBytes;
    }
    return diagnostics;
}

bool Runtime::valid() const noexcept {
    return state_ != nullptr;
}

std::uint64_t Runtime::fingerprint() const noexcept {
    return state_ ? state_->worldFingerprint : 0u;
}

std::uint64_t Runtime::sourcePhysicsFingerprint() const noexcept {
    return state_ ? state_->sourcePhysicsFingerprint : 0u;
}

std::uint64_t Runtime::deviceProgramFingerprint() const noexcept {
    return state_ ? state_->executionFingerprint : 0u;
}

bool Runtime::automaticIdentificationEnabled() const noexcept {
    return state_ != nullptr && state_->automaticIdentification;
}

bool Runtime::adaptiveTransferEnabled() const noexcept {
    return state_ != nullptr && state_->adaptiveTransfer;
}

bool Runtime::requiresBodyWrenches() const noexcept {
    return state_ != nullptr && state_->requiresBodyWrenches;
}

bool Runtime::requiresRodNodes() const noexcept {
    return state_ != nullptr && state_->requiresRodNodes;
}

bool Runtime::requiresCoupledCandidate() const noexcept {
    return state_ != nullptr && state_->requiresCoupledCandidate;
}

std::uint32_t Runtime::coupledCandidatePointCapacity() const noexcept {
    return state_ != nullptr && state_->requiresCoupledCandidate
        ? std::max(
              state_->contactActiveCapacity,
              state_->dispatch.femHumanAttachmentCount
          )
        : 0u;
}

bool Runtime::requiresRigidContactEvidence() const noexcept {
    return state_ != nullptr && state_->adaptiveTransfer &&
        state_->hasAdaptive;
}

RuntimeDiagnostics Runtime::setSutureProxyEdges(
    const std::span<const std::uint32_t> firstRodEdges,
    const std::uint32_t rodNodeCount
) {
    RuntimeDiagnostics diagnostics;
    if (state_ == nullptr) {
        diagnostics.message = "Matter runtime is not initialized";
        return diagnostics;
    }
    State& state = *state_;
    const auto ownership = state.commandOwnership;
    std::unique_lock lock(ownership->mutex);
    if (ownership->activeCommandBuffer != nullptr ||
        ownership->preDynamicsOpen || ownership->preparedStateOpen ||
        ownership->applyEncoded || ownership->restoreRequired) {
        diagnostics.message =
            "suture proxy bindings require a completed command boundary";
        return diagnostics;
    }
    if (state.sutureProxyIndices.empty() ||
        firstRodEdges.size() != state.sutureProxyIndices.size() ||
        rodNodeCount < 2u) {
        diagnostics.message =
            "suture proxy binding dimensions do not match the cooked window";
        return diagnostics;
    }

    std::vector<std::uint32_t> sortedEdges(
        firstRodEdges.begin(), firstRodEdges.end());
    std::ranges::sort(sortedEdges);
    for (std::size_t slot = 0u; slot < sortedEdges.size(); ++slot) {
        if (sortedEdges[slot] >= rodNodeCount - 1u ||
            (slot != 0u && sortedEdges[slot] == sortedEdges[slot - 1u])) {
            diagnostics.message =
                "suture proxy edges must form one unique in-range set";
            return diagnostics;
        }
    }

    std::vector<NMRigidProxyGPU> updated = state.rigidProxyLayout;
    std::vector<std::uint32_t> remapped(
        state.dispatch.rigidProxyCount, 0u);
    std::uint32_t changedCount = 0u;
    std::uint32_t requiredRodNodes = state.requiredRodNodeCount;
    for (std::size_t slot = 0u; slot < firstRodEdges.size(); ++slot) {
        const std::uint32_t proxyIndex = state.sutureProxyIndices[slot];
        const std::uint32_t edge = firstRodEdges[slot];
        requiredRodNodes = std::max(requiredRodNodes, edge + 2u);
        if (edge == state.sutureProxyEdges[slot]) continue;
        ++changedCount;
        remapped[proxyIndex] = 1u;
        updated[proxyIndex].bodyIndex = edge;
        updated[proxyIndex].sceneBodyIndex = edge + 1u;
    }
    if (changedCount > 1u) {
        diagnostics.message =
            "suture proxy transition must retain stable overlapping slots";
        return diagnostics;
    }
    if (changedCount == 0u) {
        diagnostics.encoded = true;
        diagnostics.residentBytes = state.residentBytes;
        diagnostics.device = nsString(state.device.name);
        diagnostics.message = "suture proxy bindings already match";
        return diagnostics;
    }

    @autoreleasepool {
        id<MTLBuffer> proxyStaging = [state.device
            newBufferWithBytes:updated.data()
                        length:updated.size() * sizeof(NMRigidProxyGPU)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> remappedStaging = [state.device
            newBufferWithBytes:remapped.data()
                        length:remapped.size() * sizeof(std::uint32_t)
                       options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commandBuffer = [state.queue commandBuffer];
        if (proxyStaging == nil || remappedStaging == nil ||
            commandBuffer == nil) {
            diagnostics.message =
                "failed to allocate suture proxy maintenance resources";
            return diagnostics;
        }
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit == nil) {
            diagnostics.message =
                "failed to encode suture proxy binding upload";
            return diagnostics;
        }
        [blit copyFromBuffer:proxyStaging sourceOffset:0u
                   toBuffer:state.rigidProxies destinationOffset:0u
                       size:updated.size() * sizeof(NMRigidProxyGPU)];
        [blit endEncoding];

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        id<MTLComputePipelineState> pipeline = state.pipeline(
            "nm_contact_clear_remapped_suture_histories");
        if (encoder == nil || pipeline == nil) {
            diagnostics.message =
                "failed to encode suture proxy history reset";
            return diagnostics;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:state.dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:state.contactPairs offset:0u atIndex:1u];
        [encoder setBuffer:remappedStaging offset:0u atIndex:2u];
        [encoder setBuffer:state.contactHistoriesAccepted offset:0u atIndex:3u];
        [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:4u];
        [encoder setBuffer:state.contactHistoriesCheckpoint offset:0u atIndex:5u];
        const NSUInteger total =
            static_cast<NSUInteger>(state.dispatch.environmentCount) *
            state.dispatch.contactPairCount;
        const NSUInteger width = std::min<NSUInteger>(
            256u, pipeline.maxTotalThreadsPerThreadgroup);
        if (total != 0u) {
            [encoder dispatchThreads:MTLSizeMake(total, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        }
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            diagnostics.message =
                "suture proxy maintenance command failed: " +
                errorString(commandBuffer.error);
            return diagnostics;
        }
    }

    state.rigidProxyLayout = std::move(updated);
    state.sutureProxyEdges.assign(
        firstRodEdges.begin(), firstRodEdges.end());
    state.requiredRodNodeCount = requiredRodNodes;
    ++state.sutureProxyBindingRevision;
    diagnostics.encoded = true;
    diagnostics.residentBytes = state.residentBytes;
    diagnostics.device = nsString(state.device.name);
    diagnostics.threadDispatchCount = 1u;
    diagnostics.requestedThreadCount =
        static_cast<std::uint64_t>(state.dispatch.environmentCount) *
        state.dispatch.contactPairCount;
    diagnostics.message = "suture proxy bindings advanced with stable ownership";
    return diagnostics;
}

bool Runtime::setCoupledTimestepMultiplier(
    const std::uint32_t multiplier
) noexcept {
    if (state_ == nullptr || multiplier == 0u ||
        (multiplier & (multiplier - 1u)) != 0u ||
        multiplier > (1u << NM_MAX_RATE_EXPONENT) ||
        !state_->requiresRodNodes ||
        state_->dispatch.maximumRateExponent != 0u) {
        return false;
    }
    try {
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr ||
            ownership->preDynamicsOpen || ownership->preparedStateOpen ||
            ownership->applyEncoded || ownership->restoreRequired) {
            return false;
        }
        const float activeTimestep =
            state_->dispatch.gravityAndTimestep.w *
            static_cast<float>(multiplier);
        if (!std::isfinite(activeTimestep) || !(activeTimestep > 0.0f)) {
            return false;
        }
        state_->coupledTimestepMultiplier.store(
            multiplier,
            std::memory_order_release
        );
        state_->coupledTimestepDivisor.store(
            1u,
            std::memory_order_release
        );
        return true;
    } catch (...) {
        return false;
    }
}

std::uint32_t Runtime::coupledTimestepMultiplier() const noexcept {
    return state_
        ? state_->coupledTimestepMultiplier.load(std::memory_order_acquire)
        : 0u;
}

bool Runtime::setCoupledTimestepDivisor(
    const std::uint32_t divisor
) noexcept {
    if (state_ == nullptr || divisor == 0u ||
        (divisor & (divisor - 1u)) != 0u ||
        divisor > (1u << NM_MAX_RATE_EXPONENT) ||
        !state_->requiresRodNodes ||
        state_->dispatch.maximumRateExponent != 0u) {
        return false;
    }
    try {
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr ||
            ownership->preDynamicsOpen || ownership->preparedStateOpen ||
            ownership->applyEncoded || ownership->restoreRequired) {
            return false;
        }
        const float activeTimestep =
            state_->dispatch.gravityAndTimestep.w /
            static_cast<float>(divisor);
        if (!std::isfinite(activeTimestep) || !(activeTimestep > 0.0f)) {
            return false;
        }
        state_->coupledTimestepMultiplier.store(
            1u,
            std::memory_order_release
        );
        state_->coupledTimestepDivisor.store(
            divisor,
            std::memory_order_release
        );
        return true;
    } catch (...) {
        return false;
    }
}

std::uint32_t Runtime::coupledTimestepDivisor() const noexcept {
    return state_
        ? state_->coupledTimestepDivisor.load(std::memory_order_acquire)
        : 0u;
}

float Runtime::timestepSeconds() const noexcept {
    return state_
        ? state_->dispatch.gravityAndTimestep.w *
            static_cast<float>(state_->coupledTimestepMultiplier.load(
                std::memory_order_acquire
            )) /
            static_cast<float>(state_->coupledTimestepDivisor.load(
                std::memory_order_acquire
            ))
        : 0.0f;
}

bool Runtime::setFGMRESIterationBudget(
    const std::uint32_t iterations
) noexcept {
    if (state_ == nullptr || iterations == 0u) {
        return false;
    }
    try {
        const std::uint32_t restart = std::min(
            state_->mixedSolverValue.nonlinearIterations.y,
            static_cast<std::uint32_t>(NM_MIXED_FGMRES_RESTART)
        );
        if (restart == 0u || iterations < restart ||
            iterations >
                std::numeric_limits<std::uint32_t>::max() - restart + 1u) {
            return false;
        }
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr ||
            ownership->preDynamicsOpen || ownership->preparedStateOpen ||
            ownership->applyEncoded || ownership->restoreRequired) {
            return false;
        }
        const std::uint32_t cooked = std::max(
            restart,
            state_->mixedSolverValue.nonlinearIterations.z
        );
        state_->fgmresIterationBudgetOverride.store(
            iterations == cooked ? 0u : iterations,
            std::memory_order_release
        );
        return true;
    } catch (...) {
        return false;
    }
}

std::uint32_t Runtime::fgmresIterationBudget() const noexcept {
    if (state_ == nullptr) {
        return 0u;
    }
    const std::uint32_t restart = std::min(
        state_->mixedSolverValue.nonlinearIterations.y,
        static_cast<std::uint32_t>(NM_MIXED_FGMRES_RESTART)
    );
    const std::uint32_t budgetOverride =
        state_->fgmresIterationBudgetOverride.load(
            std::memory_order_acquire
        );
    return std::max(
        restart,
        budgetOverride == 0u
            ? state_->mixedSolverValue.nonlinearIterations.z
            : budgetOverride
    );
}

bool Runtime::setNewtonIterationBudget(
    const std::uint32_t iterations
) noexcept {
    if (state_ == nullptr || iterations == 0u) {
        return false;
    }
    try {
        const auto ownership = state_->commandOwnership;
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr ||
            ownership->preDynamicsOpen || ownership->preparedStateOpen ||
            ownership->applyEncoded || ownership->restoreRequired) {
            return false;
        }
        const std::uint32_t cooked =
            state_->mixedSolverValue.nonlinearIterations.x;
        state_->newtonIterationBudgetOverride.store(
            iterations == cooked ? 0u : iterations,
            std::memory_order_release
        );
        return true;
    } catch (...) {
        return false;
    }
}

std::uint32_t Runtime::newtonIterationBudget() const noexcept {
    if (state_ == nullptr) {
        return 0u;
    }
    const std::uint32_t budgetOverride =
        state_->newtonIterationBudgetOverride.load(
            std::memory_order_acquire
        );
    return budgetOverride == 0u
        ? state_->mixedSolverValue.nonlinearIterations.x
        : budgetOverride;
}

RuntimeDiagnostics Runtime::restore(const RuntimeStateSnapshot& snapshot) {
    RuntimeDiagnostics diagnostics;
    if (state_ == nullptr) {
        diagnostics.message = "Matter runtime is not initialized";
        return diagnostics;
    }
    State& state = *state_;
    const auto ownership = state.commandOwnership;
    std::unique_lock lock(ownership->mutex);
    if (ownership->activeCommandBuffer != nullptr ||
        ownership->preDynamicsOpen || ownership->preparedStateOpen ||
        ownership->applyEncoded || ownership->restoreRequired) {
        diagnostics.message =
            "Matter snapshot restore requires a completed command boundary";
        return diagnostics;
    }
    {
        const std::lock_guard growthLock(state.growthOwnership->mutex);
        if (state.growthOwnership->pending.required) {
            diagnostics.message =
                "Matter snapshot restore cannot bypass pending topology growth";
            return diagnostics;
        }
    }
    if (!snapshot.available ||
        snapshot.sourcePhysicsFingerprint != state.sourcePhysicsFingerprint ||
        snapshot.deviceProgramFingerprint != state.executionFingerprint ||
        snapshot.materialStateStride != state.dispatch.materialStateStride) {
        diagnostics.message =
            "Matter snapshot does not match the initialized device program";
        return diagnostics;
    }
    // Incidence is derived state rather than independent snapshot authority.
    // A topology mutation can advance the accepted generation without changing
    // the compiled arena sizes, so reconstruct that derived state below from
    // the exact accepted node/tetrahedron arrays instead of pairing restored
    // topology with the initialized world's stale incidence.
    std::uint32_t cookedAllocationGeneration = 0u;
    for (const NMContinuumObjectGPU& object : state.objectLayout) {
        if (object.representation != NM_REPRESENTATION_FEM) {
            continue;
        }
        cookedAllocationGeneration = std::max(
            cookedAllocationGeneration,
            object.topologyGeneration
        );
    }
    const bool rebuildTopologyIncidence =
        snapshot.allocationGeneration != cookedAllocationGeneration;
    const auto powerOfTwo = [](const std::uint32_t value) {
        return value != 0u && (value & (value - 1u)) == 0u;
    };
    if (!powerOfTwo(snapshot.coupledTimestepMultiplier) ||
        !powerOfTwo(snapshot.coupledTimestepDivisor) ||
        (snapshot.coupledTimestepMultiplier != 1u &&
         snapshot.coupledTimestepDivisor != 1u) ||
        snapshot.coupledTimestepMultiplier >
            (1u << NM_MAX_RATE_EXPONENT) ||
        snapshot.coupledTimestepDivisor >
            (1u << NM_MAX_RATE_EXPONENT) ||
        (!state.requiresRodNodes &&
         (snapshot.coupledTimestepMultiplier != 1u ||
          snapshot.coupledTimestepDivisor != 1u))) {
        diagnostics.message =
            "Matter snapshot has an invalid coupled timestep cadence";
        return diagnostics;
    }
    const std::uint32_t fgmresRestart = std::min(
        state.mixedSolverValue.nonlinearIterations.y,
        static_cast<std::uint32_t>(NM_MIXED_FGMRES_RESTART)
    );
    const std::uint32_t fgmresBudget =
        snapshot.fgmresIterationBudgetOverride == 0u
            ? state.mixedSolverValue.nonlinearIterations.z
            : snapshot.fgmresIterationBudgetOverride;
    if (fgmresRestart == 0u || fgmresBudget < fgmresRestart ||
        fgmresBudget >
            std::numeric_limits<std::uint32_t>::max() - fgmresRestart + 1u) {
        diagnostics.message =
            "Matter snapshot has an invalid FGMRES iteration budget";
        return diagnostics;
    }
    const std::uint32_t newtonBudget =
        snapshot.newtonIterationBudgetOverride == 0u
            ? state.mixedSolverValue.nonlinearIterations.x
            : snapshot.newtonIterationBudgetOverride;
    if (newtonBudget == 0u) {
        diagnostics.message =
            "Matter snapshot has an invalid Newton iteration budget";
        return diagnostics;
    }

    bool dimensionsValid = true;
    const auto exactArena = [&](const auto& values,
                                id<MTLBuffer> target,
                                const char* label) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        const std::size_t bytes = values.size() * sizeof(Value);
        if (target == nil || bytes != target.length) {
            dimensionsValid = false;
            if (diagnostics.message.empty()) {
                diagnostics.message = std::string{"Matter snapshot "} +
                    label + " arena size changed";
            }
        }
    };
    const auto boundedArena = [&](const auto& values,
                                  id<MTLBuffer> target,
                                  const char* label) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        const std::size_t bytes = values.size() * sizeof(Value);
        if (target == nil || bytes > target.length) {
            dimensionsValid = false;
            if (diagnostics.message.empty()) {
                diagnostics.message = std::string{"Matter snapshot "} +
                    label + " logical size exceeds its arena";
            }
        }
    };
    exactArena(snapshot.particles, state.particleAccepted, "particle");
    exactArena(snapshot.femNodes, state.femAccepted, "FEM-node");
    exactArena(
        snapshot.particleMaterialState,
        state.particleMaterialStateAccepted,
        "particle-material-state"
    );
    exactArena(
        snapshot.femMaterialState,
        state.femMaterialStateAccepted,
        "FEM-material-state"
    );
    exactArena(snapshot.femFields, state.femFieldsAccepted, "FEM-field");
    exactArena(
        snapshot.solverCertificates,
        state.solverCertificates,
        "solver-certificate"
    );
    exactArena(snapshot.learnedWeights, state.learnedWeightsAccepted,
        "learned-weight");
    exactArena(snapshot.femTopologyNodes,
        state.femTopologyNodesAccepted, "topology-node");
    exactArena(snapshot.femTopologyTetrahedra,
        state.femTetrahedraAccepted, "topology-tetrahedron");
    exactArena(snapshot.cohesiveFaces,
        state.cohesiveFacesAccepted, "cohesive-face");
    exactArena(snapshot.punctureChannels,
        state.punctureChannelsAccepted, "puncture-channel");
    exactArena(snapshot.topologyStates,
        state.topologyStatesAccepted, "topology-state");
    exactArena(snapshot.adaptive, state.adaptive, "adaptive");
    exactArena(snapshot.schedulers, state.schedulers, "scheduler");
    exactArena(snapshot.reactions, state.frameReactions, "reaction");
    exactArena(snapshot.identification,
        state.identificationDistributions, "identification");
    exactArena(snapshot.environmentParameters,
        state.environmentParameters, "environment-parameter");
    boundedArena(snapshot.statuses, state.statuses, "status");
    boundedArena(snapshot.mpmActiveNodeIndices,
        state.mpmActiveNodeIndices, "MPM-active-index");
    boundedArena(snapshot.mpmNodeToActive,
        state.mpmNodeToActive, "MPM-node-map");
    boundedArena(snapshot.mpmActiveNodeCounts,
        state.mpmActiveNodeCounts, "MPM-active-count");
    boundedArena(snapshot.rigidGeneralizedCandidate,
        state.coupledGeneralizedCandidate, "rigid-generalized-candidate");
    boundedArena(snapshot.contactHistories,
        state.contactHistoriesAccepted, "contact-history");
    boundedArena(snapshot.deformableContactHistories,
        state.deformableContactHistoriesAccepted,
        "deformable-contact-history");
    boundedArena(snapshot.rigidStates, state.rigidStates, "rigid-state");
    if (state.captureDiagnostics) {
        boundedArena(
            snapshot.contactSamples,
            state.contactSamples,
            "contact-sample"
        );
    } else if (!snapshot.contactSamples.empty()) {
        dimensionsValid = false;
        diagnostics.message =
            "Matter snapshot carries diagnostics for a non-diagnostic runtime";
    }
    if (!dimensionsValid) {
        return diagnostics;
    }

    std::vector<std::uint32_t> restoredFEMIncidence;
    std::vector<NMIncidenceRangeGPU> restoredFEMRanges;
    if (rebuildTopologyIncidence) {
        try {
            const std::size_t environments = state.dispatch.environmentCount;
            const std::size_t nodesPerEnvironment =
                state.dispatch.femNodeCount;
            const std::size_t tetrahedraPerEnvironment =
                state.dispatch.tetrahedronCount;
            const std::size_t nodeCount =
                environments * nodesPerEnvironment;
            const std::size_t tetrahedronCount =
                environments * tetrahedraPerEnvironment;
            const std::size_t incidenceCapacity =
                state.femNodeIncidence.length / sizeof(std::uint32_t);
            const std::size_t incidencePerEnvironment = environments == 0u
                ? 0u : incidenceCapacity / environments;
            const bool arenaShapeValid =
                environments != 0u &&
                state.femNodeIncidence.length % sizeof(std::uint32_t) == 0u &&
                incidenceCapacity % environments == 0u &&
                incidenceCapacity <=
                    std::numeric_limits<std::uint32_t>::max() &&
                state.femNodeIncidenceCheckpoint.length ==
                    state.femNodeIncidence.length &&
                state.femNodeRanges.length %
                        sizeof(NMIncidenceRangeGPU) ==
                    0u &&
                state.femNodeRanges.length /
                        sizeof(NMIncidenceRangeGPU) ==
                    nodeCount &&
                state.femNodeRangesCheckpoint.length ==
                    state.femNodeRanges.length &&
                snapshot.femTopologyNodes.size() == nodeCount &&
                snapshot.femTopologyTetrahedra.size() == tetrahedronCount;
            if (!arenaShapeValid) {
                diagnostics.message =
                    "Matter snapshot topology incidence arena shape changed: "
                    "incidence_bytes=" +
                    std::to_string(state.femNodeIncidence.length) +
                    " incidence_checkpoint_bytes=" +
                    std::to_string(
                        state.femNodeIncidenceCheckpoint.length
                    ) +
                    " incidence_capacity=" +
                    std::to_string(incidenceCapacity) +
                    " incidence_per_environment=" +
                    std::to_string(incidencePerEnvironment) +
                    " tetrahedron_slot_incidence=" +
                    std::to_string(tetrahedraPerEnvironment * 4u) +
                    " range_bytes=" +
                    std::to_string(state.femNodeRanges.length) +
                    " range_checkpoint_bytes=" +
                    std::to_string(state.femNodeRangesCheckpoint.length) +
                    " expected_nodes=" + std::to_string(nodeCount) +
                    " snapshot_nodes=" +
                    std::to_string(snapshot.femTopologyNodes.size()) +
                    " expected_tetrahedra=" +
                    std::to_string(tetrahedronCount) +
                    " snapshot_tetrahedra=" +
                    std::to_string(
                        snapshot.femTopologyTetrahedra.size()
                    );
                return diagnostics;
            }

            restoredFEMIncidence.assign(incidenceCapacity, 0u);
            restoredFEMRanges.assign(nodeCount, NMIncidenceRangeGPU{});
            std::vector<std::uint32_t> counts(nodeCount, 0u);
            std::vector<std::uint32_t> cursors(nodeCount, 0u);
            bool topologyValid = true;
            const auto visitActiveTetrahedronNodes = [&] (
                const std::size_t environment,
                const std::uint32_t localTetrahedron,
                const auto& visitor
            ) {
                const NMTetrahedronGPU& tetrahedron =
                    snapshot.femTopologyTetrahedra[
                        environment * tetrahedraPerEnvironment +
                        localTetrahedron
                    ];
                if ((tetrahedron.identity.w & NM_OBJECT_ACTIVE) == 0u) {
                    return;
                }
                const std::array<std::uint32_t, 4u> nodes{
                    tetrahedron.nodes.x,
                    tetrahedron.nodes.y,
                    tetrahedron.nodes.z,
                    tetrahedron.nodes.w,
                };
                for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
                    const std::uint32_t node = nodes[slot];
                    bool duplicate = false;
                    for (std::uint32_t earlier = 0u;
                         earlier < slot; ++earlier) {
                        duplicate = duplicate || nodes[earlier] == node;
                    }
                    if (duplicate) {
                        continue;
                    }
                    if (node >= nodesPerEnvironment) {
                        topologyValid = false;
                        continue;
                    }
                    const std::size_t globalNode =
                        environment * nodesPerEnvironment + node;
                    const NMFEMTopologyNodeGPU& topologyNode =
                        snapshot.femTopologyNodes[globalNode];
                    if ((topologyNode.identity.w & NM_TOPOLOGY_ACTIVE) == 0u ||
                        topologyNode.identity.y != tetrahedron.identity.x) {
                        topologyValid = false;
                        continue;
                    }
                    visitor(
                        globalNode,
                        localTetrahedron
                    );
                }
            };
            for (std::size_t environment = 0u;
                 environment < environments; ++environment) {
                for (std::uint32_t tetrahedron = 0u;
                     tetrahedron < tetrahedraPerEnvironment;
                     ++tetrahedron) {
                    visitActiveTetrahedronNodes(
                        environment,
                        tetrahedron,
                        [&] (const std::size_t node, const std::uint32_t) {
                            if (counts[node] ==
                                std::numeric_limits<std::uint32_t>::max()) {
                                topologyValid = false;
                                return;
                            }
                            ++counts[node];
                        }
                    );
                }
            }
            for (std::size_t environment = 0u;
                 environment < environments; ++environment) {
                const std::size_t nodeBase =
                    environment * nodesPerEnvironment;
                const std::size_t incidenceBase =
                    environment * incidencePerEnvironment;
                std::size_t running = 0u;
                for (std::size_t localNode = 0u;
                     localNode < nodesPerEnvironment; ++localNode) {
                    const std::size_t node = nodeBase + localNode;
                    if (incidenceBase + running + counts[node] >
                        restoredFEMIncidence.size()) {
                        topologyValid = false;
                        break;
                    }
                    NMIncidenceRangeGPU& range = restoredFEMRanges[node];
                    range.first = static_cast<std::uint32_t>(
                        incidenceBase + running
                    );
                    range.count = counts[node];
                    range.objectIndex = counts[node] == 0u
                        ? NM_INVALID_INDEX
                        : snapshot.femTopologyNodes[node].identity.y;
                    range.reserved = 0u;
                    running += counts[node];
                }
                topologyValid = topologyValid &&
                    running <= incidencePerEnvironment;
            }
            if (topologyValid) {
                for (std::size_t environment = 0u;
                     environment < environments; ++environment) {
                    for (std::uint32_t tetrahedron = 0u;
                         tetrahedron < tetrahedraPerEnvironment;
                         ++tetrahedron) {
                        visitActiveTetrahedronNodes(
                            environment,
                            tetrahedron,
                            [&] (
                                const std::size_t node,
                                const std::uint32_t localTetrahedron
                            ) {
                                const NMIncidenceRangeGPU& range =
                                    restoredFEMRanges[node];
                                if (cursors[node] >= range.count ||
                                    static_cast<std::size_t>(range.first) +
                                            cursors[node] >=
                                        restoredFEMIncidence.size()) {
                                    topologyValid = false;
                                    return;
                                }
                                restoredFEMIncidence[
                                    range.first + cursors[node]++
                                ] = localTetrahedron;
                            }
                        );
                    }
                }
            }
            topologyValid = topologyValid && std::ranges::equal(
                counts,
                cursors
            );
            if (!topologyValid) {
                diagnostics.message =
                    "Matter snapshot topology incidence is invalid or exceeds "
                    "the exact device arena";
                return diagnostics;
            }
        } catch (const std::bad_alloc&) {
            diagnostics.message =
                "failed to allocate Matter topology incidence restore staging";
            return diagnostics;
        }
    }

    std::vector<NMRigidProxyGPU> restoredProxyLayout =
        state.rigidProxyLayout;
    if (snapshot.sutureProxyEdges.size() !=
        state.sutureProxyIndices.size()) {
        diagnostics.message =
            "Matter snapshot suture-proxy set width changed";
        return diagnostics;
    }
    std::uint32_t restoredRequiredRodNodeCount =
        state.requiredRodNodeCount;
    if (!snapshot.sutureProxyEdges.empty()) {
        std::vector<std::uint32_t> sorted = snapshot.sutureProxyEdges;
        std::ranges::sort(sorted);
        for (std::size_t slot = 0u; slot < sorted.size(); ++slot) {
            if (sorted[slot] >=
                    std::numeric_limits<std::uint32_t>::max() - 1u ||
                (slot != 0u && sorted[slot] == sorted[slot - 1u])) {
                diagnostics.message =
                    "Matter snapshot suture-proxy set is invalid";
                return diagnostics;
            }
            restoredRequiredRodNodeCount = std::max(
                restoredRequiredRodNodeCount,
                sorted[slot] + 2u
            );
            const std::uint32_t proxy = state.sutureProxyIndices[slot];
            restoredProxyLayout[proxy].bodyIndex =
                snapshot.sutureProxyEdges[slot];
            restoredProxyLayout[proxy].sceneBodyIndex =
                snapshot.sutureProxyEdges[slot] + 1u;
        }
    }

    @autoreleasepool {
        bool stagingValid = true;
        const auto stage = [&](const auto& values,
                               const char* label) -> id<MTLBuffer> {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const NSUInteger bytes = static_cast<NSUInteger>(
                values.size() * sizeof(Value)
            );
            if (bytes == 0u) {
                return nil;
            }
            id<MTLBuffer> result = [state.device
                newBufferWithBytes:values.data()
                            length:bytes
                           options:MTLResourceStorageModeShared];
            if (result == nil) {
                stagingValid = false;
                if (diagnostics.message.empty()) {
                    diagnostics.message = std::string{
                        "failed to allocate Matter snapshot "
                    } + label + " staging";
                }
            }
            return result;
        };
        id<MTLBuffer> particles = stage(snapshot.particles, "particle");
        id<MTLBuffer> femNodes = stage(snapshot.femNodes, "FEM-node");
        id<MTLBuffer> particleMaterialState = stage(
            snapshot.particleMaterialState,
            "particle-material-state"
        );
        id<MTLBuffer> femMaterialState = stage(
            snapshot.femMaterialState,
            "FEM-material-state"
        );
        id<MTLBuffer> femFields = stage(snapshot.femFields, "FEM-field");
        id<MTLBuffer> solverCertificates = stage(
            snapshot.solverCertificates,
            "solver-certificate"
        );
        id<MTLBuffer> statuses = stage(snapshot.statuses, "status");
        id<MTLBuffer> activeNodeIndices = stage(
            snapshot.mpmActiveNodeIndices,
            "MPM-active-index"
        );
        id<MTLBuffer> nodeToActive = stage(
            snapshot.mpmNodeToActive,
            "MPM-node-map"
        );
        id<MTLBuffer> activeNodeCounts = stage(
            snapshot.mpmActiveNodeCounts,
            "MPM-active-count"
        );
        id<MTLBuffer> rigidGeneralized = stage(
            snapshot.rigidGeneralizedCandidate,
            "rigid-generalized-candidate"
        );
        id<MTLBuffer> learnedWeights = stage(
            snapshot.learnedWeights,
            "learned-weight"
        );
        const std::array learnedRevision{snapshot.learnedWeightRevision};
        id<MTLBuffer> learnedRevisionBuffer = stage(
            learnedRevision,
            "learned-revision"
        );
        id<MTLBuffer> topologyNodes = stage(
            snapshot.femTopologyNodes,
            "topology-node"
        );
        id<MTLBuffer> topologyTetrahedra = stage(
            snapshot.femTopologyTetrahedra,
            "topology-tetrahedron"
        );
        id<MTLBuffer> cohesiveFaces = stage(
            snapshot.cohesiveFaces,
            "cohesive-face"
        );
        id<MTLBuffer> punctureChannels = stage(
            snapshot.punctureChannels,
            "puncture-channel"
        );
        id<MTLBuffer> topologyStates = stage(
            snapshot.topologyStates,
            "topology-state"
        );
        id<MTLBuffer> femNodeIncidence = rebuildTopologyIncidence
            ? stage(restoredFEMIncidence, "FEM-node-incidence") : nil;
        id<MTLBuffer> femNodeRanges = rebuildTopologyIncidence
            ? stage(restoredFEMRanges, "FEM-node-range") : nil;
        id<MTLBuffer> contactHistories = stage(
            snapshot.contactHistories,
            "contact-history"
        );
        id<MTLBuffer> deformableContactHistories = stage(
            snapshot.deformableContactHistories,
            "deformable-contact-history"
        );
        id<MTLBuffer> adaptive = stage(snapshot.adaptive, "adaptive");
        id<MTLBuffer> schedulers = stage(snapshot.schedulers, "scheduler");
        id<MTLBuffer> reactions = stage(snapshot.reactions, "reaction");
        id<MTLBuffer> rigidStates = stage(
            snapshot.rigidStates,
            "rigid-state"
        );
        id<MTLBuffer> contactSamples = state.captureDiagnostics
            ? stage(snapshot.contactSamples, "contact-sample")
            : nil;
        id<MTLBuffer> identification = stage(
            snapshot.identification,
            "identification"
        );
        id<MTLBuffer> environmentParameters = stage(
            snapshot.environmentParameters,
            "environment-parameter"
        );
        id<MTLBuffer> rigidProxies = stage(
            restoredProxyLayout,
            "rigid-proxy"
        );
        if (!stagingValid) {
            return diagnostics;
        }

        id<MTLCommandBuffer> commandBuffer = [state.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (commandBuffer == nil || blit == nil) {
            diagnostics.message =
                "failed to encode Matter snapshot restore";
            return diagnostics;
        }
        [blit setLabel:@"Numi Matter snapshot restore"];
        const auto copy = [&](id<MTLBuffer> source,
                              id<MTLBuffer> destination) {
            if (destination.length != 0u) {
                [blit fillBuffer:destination
                           range:NSMakeRange(0u, destination.length)
                           value:0u];
            }
            if (source != nil && source.length != 0u) {
                [blit copyFromBuffer:source sourceOffset:0u
                            toBuffer:destination destinationOffset:0u
                                size:source.length];
            }
        };
        const auto mirror = [&](id<MTLBuffer> source,
                                id<MTLBuffer> accepted,
                                id<MTLBuffer> candidate,
                                id<MTLBuffer> checkpoint) {
            copy(source, accepted);
            copy(source, candidate);
            copy(source, checkpoint);
        };
        mirror(particles, state.particleAccepted,
            state.particleCandidate, state.particleCheckpoint);
        mirror(particleMaterialState,
            state.particleMaterialStateAccepted,
            state.particleMaterialStateCandidate,
            state.particleMaterialStateCheckpoint);
        mirror(femNodes, state.femAccepted,
            state.femCandidate, state.femCheckpoint);
        mirror(femMaterialState,
            state.femMaterialStateAccepted,
            state.femMaterialStateCandidate,
            state.femMaterialStateCheckpoint);
        mirror(femFields, state.femFieldsAccepted,
            state.femFieldsCandidate, state.femFieldsCheckpoint);
        copy(femFields, state.femFieldsWork);
        mirror(learnedWeights, state.learnedWeightsAccepted,
            state.learnedWeightsCandidate, state.learnedWeightsCheckpoint);
        mirror(learnedRevisionBuffer, state.learnedRevisionAccepted,
            state.learnedRevisionCandidate, state.learnedRevisionCheckpoint);
        mirror(topologyNodes, state.femTopologyNodesAccepted,
            state.femTopologyNodesCandidate,
            state.femTopologyNodesCheckpoint);
        mirror(topologyTetrahedra, state.femTetrahedraAccepted,
            state.femTetrahedraCandidate,
            state.femTetrahedraCheckpoint);
        mirror(cohesiveFaces, state.cohesiveFacesAccepted,
            state.cohesiveFacesCandidate, state.cohesiveFacesCheckpoint);
        mirror(punctureChannels, state.punctureChannelsAccepted,
            state.punctureChannelsCandidate,
            state.punctureChannelsCheckpoint);
        mirror(topologyStates, state.topologyStatesAccepted,
            state.topologyStatesCandidate, state.topologyStatesCheckpoint);
        if (rebuildTopologyIncidence) {
            copy(femNodeIncidence, state.femNodeIncidence);
            copy(femNodeIncidence, state.femNodeIncidenceCheckpoint);
            copy(femNodeRanges, state.femNodeRanges);
            copy(femNodeRanges, state.femNodeRangesCheckpoint);
            if (state.topologyIncidenceCursors.length != 0u) {
                [blit fillBuffer:state.topologyIncidenceCursors
                           range:NSMakeRange(
                               0u,
                               state.topologyIncidenceCursors.length
                           )
                           value:0u];
            }
        }
        mirror(contactHistories, state.contactHistoriesAccepted,
            state.contactHistoriesCandidate,
            state.contactHistoriesCheckpoint);
        mirror(deformableContactHistories,
            state.deformableContactHistoriesAccepted,
            state.deformableContactHistoriesCandidate,
            state.deformableContactHistoriesCheckpoint);
        copy(solverCertificates, state.solverCertificates);
        copy(statuses, state.statuses);
        copy(activeNodeIndices, state.mpmActiveNodeIndices);
        copy(nodeToActive, state.mpmNodeToActive);
        copy(activeNodeCounts, state.mpmActiveNodeCounts);
        copy(rigidGeneralized, state.coupledGeneralizedCandidate);
        copy(adaptive, state.adaptive);
        copy(schedulers, state.schedulers);
        copy(schedulers, state.schedulerPrevious);
        copy(schedulers, state.schedulerCheckpoint);
        copy(reactions, state.frameReactions);
        copy(rigidStates, state.rigidStates);
        if (state.captureDiagnostics) {
            copy(contactSamples, state.contactSamples);
        }
        copy(identification, state.identificationDistributions);
        copy(environmentParameters, state.environmentParameters);
        copy(rigidProxies, state.rigidProxies);
        [blit endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            diagnostics.message =
                "Matter snapshot restore command failed: " +
                errorString(commandBuffer.error);
            return diagnostics;
        }
    }

    state.rigidProxyLayout = std::move(restoredProxyLayout);
    state.sutureProxyEdges = snapshot.sutureProxyEdges;
    state.sutureProxyBindingRevision = snapshot.sutureProxyBindingRevision;
    state.requiredRodNodeCount = restoredRequiredRodNodeCount;
    state.coupledTimestepMultiplier.store(
        snapshot.coupledTimestepMultiplier,
        std::memory_order_release
    );
    state.coupledTimestepDivisor.store(
        snapshot.coupledTimestepDivisor,
        std::memory_order_release
    );
    state.fgmresIterationBudgetOverride.store(
        snapshot.fgmresIterationBudgetOverride,
        std::memory_order_release
    );
    state.newtonIterationBudgetOverride.store(
        snapshot.newtonIterationBudgetOverride,
        std::memory_order_release
    );
    ownership->controlStep = snapshot.controlStep;
    ownership->physicsSubstep = snapshot.physicsSubstep;
    ownership->identificationGeneration =
        snapshot.identificationGeneration;
    ownership->identificationCheckpoint =
        snapshot.identificationCheckpoint;
    ownership->identificationAdvanced = snapshot.identificationAdvanced;
    diagnostics.encoded = true;
    diagnostics.residentBytes = state.residentBytes;
    diagnostics.device = nsString(state.device.name);
    diagnostics.message =
        "Numi Matter completion-boundary snapshot restored";
    return diagnostics;
}

RuntimeStateSnapshot Runtime::snapshot() const {
    RuntimeStateSnapshot snapshot;
    if (state_ == nullptr) {
        snapshot.message = "Matter runtime is not initialized";
        return snapshot;
    }
    snapshot.sourcePhysicsFingerprint = state_->sourcePhysicsFingerprint;
    snapshot.deviceProgramFingerprint = state_->executionFingerprint;
    const auto ownership = state_->commandOwnership;
    {
        const std::lock_guard lock(ownership->mutex);
        if (ownership->activeCommandBuffer != nullptr ||
            ownership->preparedStateOpen || ownership->applyEncoded ||
            ownership->restoreRequired) {
            snapshot.message =
                "Matter state cannot be read while a borrowed transaction is active";
            return snapshot;
        }
        snapshot.sutureProxyEdges = state_->sutureProxyEdges;
        snapshot.sutureProxyBindingRevision =
            state_->sutureProxyBindingRevision;
        snapshot.coupledTimestepMultiplier =
            state_->coupledTimestepMultiplier.load(
                std::memory_order_acquire
            );
        snapshot.coupledTimestepDivisor =
            state_->coupledTimestepDivisor.load(
                std::memory_order_acquire
            );
        snapshot.fgmresIterationBudgetOverride =
            state_->fgmresIterationBudgetOverride.load(
                std::memory_order_acquire
            );
        snapshot.newtonIterationBudgetOverride =
            state_->newtonIterationBudgetOverride.load(
                std::memory_order_acquire
            );
        snapshot.controlStep = ownership->controlStep;
        snapshot.physicsSubstep = ownership->physicsSubstep;
        snapshot.identificationGeneration =
            ownership->identificationGeneration;
        snapshot.identificationCheckpoint =
            ownership->identificationCheckpoint;
        snapshot.identificationAdvanced = ownership->identificationAdvanced;
    }
    @autoreleasepool {
        const auto copy = [&](id<MTLBuffer> source) -> id<MTLBuffer> {
            return [state_->device newBufferWithLength:source.length
                                               options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> particles = copy(state_->particleAccepted);
        id<MTLBuffer> femNodes = copy(state_->femAccepted);
        id<MTLBuffer> particleMaterialState =
            copy(state_->particleMaterialStateAccepted);
        id<MTLBuffer> femMaterialState =
            copy(state_->femMaterialStateAccepted);
        id<MTLBuffer> femFields = copy(state_->femFieldsAccepted);
        id<MTLBuffer> solverCertificates = copy(state_->solverCertificates);
        id<MTLBuffer> statuses = copy(state_->statuses);
        id<MTLBuffer> mpmActiveNodeIndices =
            copy(state_->mpmActiveNodeIndices);
        id<MTLBuffer> mpmNodeToActive = copy(state_->mpmNodeToActive);
        id<MTLBuffer> mpmActiveNodeCounts =
            copy(state_->mpmActiveNodeCounts);
        id<MTLBuffer> rigidGeneralizedCandidate =
            copy(state_->coupledGeneralizedCandidate);
        id<MTLBuffer> learnedWeights = copy(state_->learnedWeightsAccepted);
        id<MTLBuffer> learnedRevision = copy(state_->learnedRevisionAccepted);
        id<MTLBuffer> topologyNodes = copy(state_->femTopologyNodesAccepted);
        id<MTLBuffer> topologyTetrahedra = copy(state_->femTetrahedraAccepted);
        id<MTLBuffer> cohesiveFaces = copy(state_->cohesiveFacesAccepted);
        id<MTLBuffer> punctureChannels = copy(state_->punctureChannelsAccepted);
        id<MTLBuffer> topologyStates = copy(state_->topologyStatesAccepted);
        id<MTLBuffer> contactHistories =
            copy(state_->contactHistoriesAccepted);
        id<MTLBuffer> deformableContactHistories =
            copy(state_->deformableContactHistoriesAccepted);
        id<MTLBuffer> adaptive = copy(state_->adaptive);
        id<MTLBuffer> schedulers = copy(state_->schedulers);
        id<MTLBuffer> reactions = copy(state_->frameReactions);
        id<MTLBuffer> rigidStates = copy(state_->rigidStates);
        id<MTLBuffer> contactSamples = state_->captureDiagnostics
            ? copy(state_->contactSamples)
            : nil;
        id<MTLBuffer> identification =
            copy(state_->identificationDistributions);
        id<MTLBuffer> environmentParameters =
            copy(state_->environmentParameters);
        if (particles == nil || femNodes == nil ||
            particleMaterialState == nil || femMaterialState == nil ||
            femFields == nil || solverCertificates == nil ||
            statuses == nil ||
            mpmActiveNodeIndices == nil || mpmNodeToActive == nil ||
            mpmActiveNodeCounts == nil ||
            rigidGeneralizedCandidate == nil ||
            learnedWeights == nil || learnedRevision == nil ||
            topologyNodes == nil || topologyTetrahedra == nil ||
            cohesiveFaces == nil || punctureChannels == nil ||
            topologyStates == nil || contactHistories == nil ||
            deformableContactHistories == nil ||
            adaptive == nil || schedulers == nil || reactions == nil ||
            rigidStates == nil ||
            (state_->captureDiagnostics && contactSamples == nil) ||
            identification == nil || environmentParameters == nil) {
            snapshot.message = "failed to allocate Matter diagnostic readback";
            return snapshot;
        }
        id<MTLCommandBuffer> commandBuffer = [state_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (commandBuffer == nil || blit == nil) {
            snapshot.message = "failed to create Matter diagnostic readback command";
            return snapshot;
        }
        const auto encodeCopy = [&](id<MTLBuffer> source, id<MTLBuffer> target) {
            if (source.length != 0u) {
                [blit copyFromBuffer:source sourceOffset:0u
                              toBuffer:target destinationOffset:0u
                                  size:source.length];
            }
        };
        encodeCopy(state_->particleAccepted, particles);
        encodeCopy(state_->femAccepted, femNodes);
        encodeCopy(
            state_->particleMaterialStateAccepted,
            particleMaterialState
        );
        encodeCopy(state_->femMaterialStateAccepted, femMaterialState);
        encodeCopy(state_->femFieldsAccepted, femFields);
        encodeCopy(state_->solverCertificates, solverCertificates);
        encodeCopy(state_->statuses, statuses);
        encodeCopy(state_->mpmActiveNodeIndices, mpmActiveNodeIndices);
        encodeCopy(state_->mpmNodeToActive, mpmNodeToActive);
        encodeCopy(state_->mpmActiveNodeCounts, mpmActiveNodeCounts);
        encodeCopy(
            state_->coupledGeneralizedCandidate,
            rigidGeneralizedCandidate);
        encodeCopy(state_->learnedWeightsAccepted, learnedWeights);
        encodeCopy(state_->learnedRevisionAccepted, learnedRevision);
        encodeCopy(state_->femTopologyNodesAccepted, topologyNodes);
        encodeCopy(state_->femTetrahedraAccepted, topologyTetrahedra);
        encodeCopy(state_->cohesiveFacesAccepted, cohesiveFaces);
        encodeCopy(state_->punctureChannelsAccepted, punctureChannels);
        encodeCopy(state_->topologyStatesAccepted, topologyStates);
        encodeCopy(state_->contactHistoriesAccepted, contactHistories);
        encodeCopy(
            state_->deformableContactHistoriesAccepted,
            deformableContactHistories);
        encodeCopy(state_->adaptive, adaptive);
        encodeCopy(state_->schedulers, schedulers);
        encodeCopy(state_->frameReactions, reactions);
        encodeCopy(state_->rigidStates, rigidStates);
        if (state_->captureDiagnostics) {
            encodeCopy(state_->contactSamples, contactSamples);
        }
        encodeCopy(state_->identificationDistributions, identification);
        encodeCopy(state_->environmentParameters, environmentParameters);
        [blit endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            snapshot.message = "Matter diagnostic readback command failed: " +
                errorString(commandBuffer.error);
            return snapshot;
        }
        const auto read = [](id<MTLBuffer> buffer, auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            values.resize(buffer.length / sizeof(Value));
            if (!values.empty()) {
                std::memcpy(
                    values.data(),
                    buffer.contents,
                    values.size() * sizeof(Value)
                );
            }
        };
        const auto readCount = [](id<MTLBuffer> buffer, auto& values,
                                  const std::size_t count) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            values.resize(count);
            if (count != 0u) {
                const std::size_t bytes = count * sizeof(Value);
                if (bytes > buffer.length) {
                    values.clear();
                    return;
                }
                std::memcpy(values.data(), buffer.contents, bytes);
            }
        };
        read(particles, snapshot.particles);
        read(femNodes, snapshot.femNodes);
        snapshot.materialStateStride = state_->dispatch.materialStateStride;
        read(particleMaterialState, snapshot.particleMaterialState);
        read(femMaterialState, snapshot.femMaterialState);
        read(femFields, snapshot.femFields);
        read(solverCertificates, snapshot.solverCertificates);
        readCount(
            statuses,
            snapshot.statuses,
            state_->dispatch.environmentCount
        );
        readCount(
            mpmActiveNodeIndices, snapshot.mpmActiveNodeIndices,
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
                state_->dispatch.mpmActiveNodeCapacity);
        readCount(
            mpmNodeToActive, snapshot.mpmNodeToActive,
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
                state_->dispatch.gridNodeCount);
        readCount(
            mpmActiveNodeCounts, snapshot.mpmActiveNodeCounts,
            state_->dispatch.environmentCount);
        readCount(
            rigidGeneralizedCandidate,
            snapshot.rigidGeneralizedCandidate,
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
                state_->dispatch.rigidGeneralizedCapacity);
        read(learnedWeights, snapshot.learnedWeights);
        snapshot.learnedWeightRevision =
            *static_cast<const std::uint32_t*>(learnedRevision.contents);
        read(topologyNodes, snapshot.femTopologyNodes);
        read(topologyTetrahedra, snapshot.femTopologyTetrahedra);
        read(cohesiveFaces, snapshot.cohesiveFaces);
        read(punctureChannels, snapshot.punctureChannels);
        read(topologyStates, snapshot.topologyStates);
        for (const NMFEMTopologyStateGPU& topology : snapshot.topologyStates)
            snapshot.allocationGeneration = std::max(
                snapshot.allocationGeneration, topology.roles.w);
        const std::size_t logicalContactCount =
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
            state_->dispatch.contactPairCount;
        readCount(contactHistories, snapshot.contactHistories,
                  logicalContactCount);
        const std::size_t logicalDeformableContactCount =
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
            state_->dispatch.deformableContactCapacity;
        readCount(
            deformableContactHistories,
            snapshot.deformableContactHistories,
            logicalDeformableContactCount);
        read(adaptive, snapshot.adaptive);
        read(schedulers, snapshot.schedulers);
        read(reactions, snapshot.reactions);
        readCount(
            rigidStates,
            snapshot.rigidStates,
            static_cast<std::size_t>(state_->dispatch.environmentCount) *
                state_->dispatch.rigidProxyCount
        );
        if (state_->captureDiagnostics) {
            readCount(contactSamples, snapshot.contactSamples,
                      logicalContactCount);
        }
        read(identification, snapshot.identification);
        read(environmentParameters, snapshot.environmentParameters);
        snapshot.available = true;
    }
    return snapshot;
}

void* Runtime::eventBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->events : nullptr;
}

void* Runtime::statusBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->statuses : nullptr;
}

void* Runtime::femConstraintReactionBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->femConstraintReactions : nullptr;
}

void* Runtime::parameterBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->environmentParameters : nullptr;
}

void* Runtime::identificationLossBuffer() const noexcept {
    return state_ ? (__bridge void*)state_->identificationLosses : nullptr;
}

} // namespace numi::matter
