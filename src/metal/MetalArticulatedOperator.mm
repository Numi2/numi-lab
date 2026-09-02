#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <utility>

#ifndef METALROBO_DEFAULT_METALLIB
#define METALROBO_DEFAULT_METALLIB ""
#endif

namespace metalrobo {
namespace {

// The final stream carries one canonical packed OpenSim SpatialTransform per
// global joint. Non-FunctionBased slots are all-zero and are never consumed.
// Shared kernel ABI. The first sixteen slots are the established articulated
// operator stream; the Millard sidecar consumes those private outputs and
// occupies slots 16..23 in the same command buffer. The MyoSim sidecar owns
// slots 24..30 and consumes the same private pose/Jacobian output directly.
constexpr std::size_t kRawBufferCount = 31u;
constexpr std::size_t kStandBufferCount = 13u;
constexpr std::size_t kHumanMatterBufferCount = 16u;
constexpr std::size_t kHumanMatterQCheckpointBuffer = 0u;
constexpr std::size_t kHumanMatterVCheckpointBuffer = 1u;
constexpr std::size_t kHumanMatterMujocoCheckpointBuffer = 2u;
constexpr std::size_t kHumanMatterOwnerStatusBuffer = 3u;
constexpr std::size_t kHumanMatterCandidatePointBuffer = 4u;
constexpr std::size_t kHumanMatterCandidateBodyPoseBuffer = 5u;
constexpr std::size_t kHumanMatterCandidatePointWorldBuffer = 6u;
constexpr std::size_t kHumanMatterCandidatePointJacobianBuffer = 7u;
constexpr std::size_t kHumanMatterCandidateOperatorScratchBuffer = 8u;
constexpr std::size_t kHumanMatterCandidateOperatorStatusBuffer = 9u;
constexpr std::size_t kHumanMatterAppliedOutcomeBuffer = 10u;
constexpr std::size_t kHumanMatterFinalAcceptedTokenBuffer = 11u;
constexpr std::size_t kHumanMatterProposalBuffer = 12u;
constexpr std::size_t kHumanMatterProposedTokenBuffer = 13u;
constexpr std::size_t kHumanMatterApplyActionBuffer = 14u;
constexpr std::size_t kHumanMatterPublicationFenceBuffer = 15u;
constexpr std::size_t kStandVelocityBuffer = 0u;
constexpr std::size_t kStandContactsBuffer = 1u;
constexpr std::size_t kStandSpatialJacobianBuffer = 2u;
constexpr std::size_t kStandBodyMotionBuffer = 3u;
constexpr std::size_t kStandFactorBuffer = 4u;
constexpr std::size_t kStandVectorBuffer = 5u;
constexpr std::size_t kStandResponseBuffer = 6u;
constexpr std::size_t kStandStatusBuffer = 7u;
constexpr std::size_t kStandTendonBindingsBuffer = 8u;
constexpr std::size_t kStandTendonEnvelopesBuffer = 9u;
constexpr std::size_t kStandTendonTransfersBuffer = 10u;
constexpr std::size_t kStandTendonCorrectionsBuffer = 11u;
constexpr std::size_t kStandJointEqualitiesBuffer = 12u;
constexpr std::size_t kMillardDispatchBuffer = 16u;
constexpr std::size_t kMillardMusclesBuffer = 17u;
constexpr std::size_t kMillardStatesBuffer = 18u;
constexpr std::size_t kMillardPathPointsBuffer = 19u;
constexpr std::size_t kMillardCurvesBuffer = 20u;
constexpr std::size_t kMillardWrapsBuffer = 21u;
constexpr std::size_t kMillardResultsBuffer = 22u;
constexpr std::size_t kMillardForcesBuffer = 23u;
constexpr std::size_t kMujocoDispatchBuffer = 24u;
constexpr std::size_t kMujocoMusclesBuffer = 25u;
constexpr std::size_t kMujocoStatesBuffer = 26u;
constexpr std::size_t kMujocoSitesBuffer = 27u;
constexpr std::size_t kMujocoWrapsBuffer = 28u;
constexpr std::size_t kMujocoRoutesBuffer = 29u;
constexpr std::size_t kMujocoResultsBuffer = 30u;
constexpr NSUInteger kThreadsPerThreadgroup = 32u;
constexpr NSUInteger kStandThreadsPerThreadgroup = 256u;
constexpr float kQuaternionHostTolerance = 1.9e-5f;
constexpr std::uint64_t kShaderAddressableElements =
    static_cast<std::uint64_t>(
        std::numeric_limits<mr_u32>::max()
    ) + 1u;
const char kMetalRoboImageAnchor = 0;

struct TendonLoadAbortGuard {
    const MetalNumiHumanTendonLoadProgram* program = nullptr;
    void* commandBuffer = nullptr;
    bool armed = false;

    ~TendonLoadAbortGuard() {
        if (armed && program != nullptr && program->abort != nullptr) {
            program->abort(program->context, commandBuffer);
        }
    }
};

[[nodiscard]] bool knownNumanXTransactionPhase(
    const MetalNumanXTransactionPhase phase
) noexcept {
    switch (phase) {
    case MetalNumanXTransactionPhase::beginStep:
    case MetalNumanXTransactionPhase::preDynamics:
    case MetalNumanXTransactionPhase::postDynamics:
        return true;
    }
    return false;
}

struct NumanXTransactionAbortGuard {
    void* context = nullptr;
    MetalNumanXTransactionAbort abort = nullptr;
    void* commandBuffer = nullptr;
    bool armed = false;

    ~NumanXTransactionAbortGuard() noexcept {
        if (armed && abort != nullptr) {
            armed = false;
            abort(context, commandBuffer);
        }
    }
};

[[nodiscard]] bool knownNumanXHumanMatterPhase(
    const MetalNumanXHumanMatterPhase phase
) noexcept {
    switch (phase) {
    case MetalNumanXHumanMatterPhase::beginStep:
    case MetalNumanXHumanMatterPhase::preDynamics:
    case MetalNumanXHumanMatterPhase::postDynamics:
        return true;
    }
    return false;
}

struct NumanXHumanMatterAbortGuard {
    void* context = nullptr;
    MetalNumanXHumanMatterAbort abort = nullptr;
    void* commandBuffer = nullptr;
    bool armed = false;

    ~NumanXHumanMatterAbortGuard() noexcept {
        if (armed && abort != nullptr) {
            armed = false;
            abort(context, commandBuffer);
        }
    }
};

struct NumanXHumanMatterPrepareLeaseGuard {
    void* context = nullptr;
    MetalNumanXHumanMatterReleasePrepareLease release = nullptr;
    MetalNumanXHumanMatterPrepareLease lease{};
    bool armed = false;

    ~NumanXHumanMatterPrepareLeaseGuard() noexcept {
        if (armed && release != nullptr) {
            armed = false;
            release(context, lease, nullptr, false);
        }
    }
};

struct BufferRequirement {
    const char* label = "";
    std::size_t logicalElements = 0u;
    std::size_t logicalBytes = 0u;
    std::size_t allocationBytes = 0u;
};

struct RequiredBuffers {
    std::array<BufferRequirement, kRawBufferCount> entries{};
    std::array<BufferRequirement, kStandBufferCount> standEntries{};
    std::array<BufferRequirement, kHumanMatterBufferCount>
        humanMatterEntries{};
};

} // namespace

namespace detail {

struct MetalArticulatedOperatorContextState {
    explicit MetalArticulatedOperatorContextState(
        MetalArticulatedOperatorConfig configured
    )
        : config(std::move(configured)) {}
    ~MetalArticulatedOperatorContextState();

    MetalArticulatedOperatorConfig config;
    mutable std::mutex mutex;
    bool initialized = false;
    bool inFlight = false;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> pipeline = nil;
    __strong id<MTLComputePipelineState> millardPipeline = nil;
    __strong id<MTLComputePipelineState> mujocoPipeline = nil;
    __strong id<MTLComputePipelineState> mujocoActiveForcePipeline = nil;
    __strong id<MTLComputePipelineState> mujocoReducePipeline = nil;
    __strong id<MTLComputePipelineState> mujocoActivationPipeline = nil;
    __strong id<MTLComputePipelineState> standPipeline = nil;
    __strong id<MTLComputePipelineState> tendonPipeline = nil;
    __strong id<MTLComputePipelineState> humanMatterBeginPipeline = nil;
    __strong id<MTLComputePipelineState> humanMatterFactorPipeline = nil;
    __strong id<MTLComputePipelineState> humanMatterConsumePipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterPreparePhysicalPipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterMarkPhysicalCompletePipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterProposePreparedPipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterValidateApplyPipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterCompleteApplyPipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterPrepareCandidatePipeline = nil;
    __strong id<MTLComputePipelineState>
        humanMatterMaterializeCandidatePipeline = nil;
    __strong id<MTLSharedEvent> humanMatterTimelineEvent = nil;
    std::uint64_t humanMatterNextEventValue = 0u;
    __strong id<MTLBuffer> buffers[kRawBufferCount] = {};
    __strong id<MTLBuffer> standBuffers[kStandBufferCount] = {};
    __strong id<MTLBuffer> humanMatterBuffers[kHumanMatterBufferCount] = {};
    std::array<std::size_t, kRawBufferCount> capacities{};
    std::array<std::size_t, kStandBufferCount> standCapacities{};
    std::array<std::size_t, kHumanMatterBufferCount> humanMatterCapacities{};
    struct PublishedResidentState {
        bool active = false;
        const EngineModel* model = nullptr;
        std::uint64_t transactionFingerprint = 0u;
        std::uint64_t physicsGeneration = 0u;
        std::uint64_t acceptedTokenFingerprint = 0u;
        std::size_t qBytes = 0u;
        std::size_t velocityBytes = 0u;
        std::size_t mujocoStateBytes = 0u;
    } publishedResident{};
    struct HumanMatterPreparedRuntimeState {
        bool active = false;
        bool firstSubmissionReleased = false;
        bool physicalFailure = false;
        bool physicalComplete = false;
        bool physicalCompletionRegistered = false;
        bool physicalCompletionDelivered = false;
        void* physicalCompletionContext = nullptr;
        MetalNumanXHumanMatterPhysicalCompletion physicalCompletion = nullptr;
        bool terminalNoTouch = false;
        bool proposalInFlight = false;
        bool proposalComplete = false;
        bool proposalFailed = false;
        bool proposalCompletionRegistered = false;
        bool proposalCompletionDelivered = false;
        void* proposalCompletionContext = nullptr;
        MetalNumanXHumanMatterProposalCompletion proposalCompletion = nullptr;
        bool humanIOBindInFlight = false;
        bool applicationReservationInFlight = false;
        bool applicationReserved = false;
        bool applyInFlight = false;
        bool applyComplete = false;
        bool applyFailed = false;
        bool appliedAcceptPendingPublication = false;
        bool publicationReserved = false;
        bool publicationReleaseInFlight = false;
        std::uint64_t jointCommitFingerprint = 0u;
        std::uint64_t brainGeneration = 0u;
        std::uint64_t proposalAttempt = 0u;
        std::uintptr_t proposalCommandBufferIdentity = 0u;
        std::uint64_t applyAttempt = 0u;
        std::uintptr_t applyCommandBufferIdentity = 0u;
        std::uint64_t physicalPreparedEventValue = 0u;
        std::uint64_t proposalEventValue = 0u;
        std::uint64_t appliedEventValue = 0u;
        MRNumanXHumanMatterDispatchGPU dispatch{};
        std::uint32_t transactionSlot = 0u;
        std::uint64_t preparedTokenByteCount = 0u;
        std::uint64_t proposalElementCount = 0u;
        std::uint64_t proposedTokenByteCount = 0u;
        std::uint64_t applyActionElementCount = 0u;
        std::uint64_t appliedOutcomeElementCount = 0u;
        std::uint64_t finalTokenByteCount = 0u;
        std::uint64_t publicationFenceElementCount = 0u;
        const EngineModel* residentModel = nullptr;
        std::size_t residentQBytes = 0u;
        std::size_t residentVelocityBytes = 0u;
        std::size_t residentMujocoStateBytes = 0u;
        __strong id<MTLBuffer> preparedTokens = nil;
        void* leaseContext = nullptr;
        MetalNumanXHumanMatterBindHumanIOCandidatePublication
            bindHumanIOCandidatePublication = nullptr;
        MetalNumanXHumanMatterReleasePrepareLease releasePrepareLease =
            nullptr;
        MetalNumanXHumanMatterReservePreparedApplication
            reservePreparedApplication = nullptr;
        MetalNumanXHumanMatterEncodePreparedApply encodePreparedApply = nullptr;
        MetalNumanXHumanMatterAbortPreparedApply abortPreparedApply = nullptr;
        MetalNumanXHumanMatterReservePublishedRoot reservePublishedRoot = nullptr;
        MetalNumanXHumanMatterReleasePublishedRoot releasePublishedRoot = nullptr;
        MetalNumanXHumanMatterBrainPreflightView preflight{};
        MetalNumanXHumanMatterApplyPass applyPass{};
        bool adapterApplyEncoded = false;
        MetalNumanXHumanMatterPrepareLease lease{};
        std::weak_ptr<MetalNumanXHumanMatterPreparedState> capability{};
    } humanMatterPrepared{};
    MetalArticulatedOperatorContextStats stats{};
};

MetalArticulatedOperatorContextState::
    ~MetalArticulatedOperatorContextState() {
    auto& prepared = humanMatterPrepared;
    if (prepared.active && prepared.releasePrepareLease != nullptr) {
        prepared.releasePrepareLease(
            prepared.leaseContext,
            prepared.lease,
            nullptr,
            false
        );
    }
}

struct MetalArticulatedOperatorSubmissionState {
    [[nodiscard]] bool reapTerminalWithoutWaiting() noexcept {
        if (!ownsInFlight || context == nullptr) {
            return true;
        }
        const MTLCommandBufferStatus status = commandBuffer == nil
            ? MTLCommandBufferStatusError
            : commandBuffer.status;
        if (status != MTLCommandBufferStatusCompleted &&
            status != MTLCommandBufferStatusError) {
            return false;
        }
        try {
            const std::lock_guard lock(context->mutex);
            auto& prepared = context->humanMatterPrepared;
            const bool stillQuarantined = hasPreparedHumanMatter &&
                prepared.active &&
                prepared.dispatch.slotGeneration ==
                    preparedHumanMatterGeneration;
            if (stillQuarantined) {
                prepared.firstSubmissionReleased = true;
                context->stats.hasInFlightSubmission = true;
            } else {
                context->inFlight = false;
                context->stats.hasInFlightSubmission = false;
            }
            ++context->stats.completedSubmissionCount;
            ++context->stats.terminalSubmissionNonwaitingReapCount;
        } catch (...) {
            // Metal is already terminal. Never turn a host bookkeeping
            // failure into a wait or leave ownership armed for the destructor.
        }
        ownsInFlight = false;
        return true;
    }

    ~MetalArticulatedOperatorSubmissionState() {
        if (!ownsInFlight || context == nullptr) {
            return;
        }
        if (reapTerminalWithoutWaiting()) {
            return;
        }
        @autoreleasepool {
            [commandBuffer waitUntilCompleted];
        }
        try {
            const std::lock_guard lock(context->mutex);
            auto& prepared = context->humanMatterPrepared;
            const bool stillQuarantined = hasPreparedHumanMatter &&
                prepared.active &&
                prepared.dispatch.slotGeneration ==
                    preparedHumanMatterGeneration;
            if (stillQuarantined) {
                prepared.firstSubmissionReleased = true;
                context->stats.hasInFlightSubmission = true;
            } else {
                context->inFlight = false;
                context->stats.hasInFlightSubmission = false;
            }
            ++context->stats.completedSubmissionCount;
            ++context->stats.submissionDestructorWaitCount;
        } catch (...) {
            // Destructors must not throw. The command is complete, so even a
            // platform mutex failure cannot leave GPU work accessing memory.
        }
        ownsInFlight = false;
    }

    std::shared_ptr<MetalArticulatedOperatorContextState> context;
    __strong id<MTLCommandBuffer> commandBuffer = nil;
    MetalArticulatedOperatorDiagnostics diagnostics{};
    std::chrono::steady_clock::time_point start{};
    MRArticulationGPU articulation{};
    std::uint32_t articulationIndex = 0u;
    std::size_t pointCount = 0u;
    bool hasMillardReference = false;
    std::size_t millardMuscleCount = 0u;
    bool hasMujocoReference = false;
    std::size_t mujocoMuscleCount = 0u;
    bool hasStandHorizon = false;
    bool hasPreparedHumanMatter = false;
    std::uint64_t preparedHumanMatterGeneration = 0u;
    std::uint32_t standStepCount = 0u;
    std::size_t standTendonBindingCount = 0u;
    std::size_t standTendonEnvelopeBindingCount = 0u;
    std::size_t standJointEqualityCount = 0u;
    bool ownsInFlight = false;
};

// The move-only public prepared capability adopts the original submission so
// its destructor cannot synchronously wait while a valid two-phase root is in
// flight. The final-command completion handler holds this shared state and
// reaps the submission only after the event-ordered final command completes.
struct MetalNumanXHumanMatterPreparedState {
    std::unique_ptr<MetalArticulatedOperatorSubmissionState> submission;
};

} // namespace detail

namespace {

struct HumanMatterPhysicalCompletionInvocation {
    MetalNumanXHumanMatterPhysicalCompletion completion = nullptr;
    void* context = nullptr;
    MetalNumanXHumanMatterPhysicalCompletionStatus status =
        MetalNumanXHumanMatterPhysicalCompletionStatus::terminalNoTouch;
    std::uint64_t slotGeneration = 0u;
};

[[nodiscard]] HumanMatterPhysicalCompletionInvocation
takeHumanMatterPhysicalCompletionLocked(
    detail::MetalArticulatedOperatorContextState::
        HumanMatterPreparedRuntimeState& prepared
) noexcept {
    HumanMatterPhysicalCompletionInvocation invocation{};
    if (!prepared.physicalCompletionRegistered ||
        prepared.physicalCompletionDelivered ||
        prepared.physicalCompletion == nullptr ||
        prepared.physicalCompletionContext == nullptr ||
        !prepared.physicalComplete) {
        return invocation;
    }
    invocation.completion = prepared.physicalCompletion;
    invocation.context = prepared.physicalCompletionContext;
    invocation.status = !prepared.physicalFailure &&
            !prepared.terminalNoTouch
        ? MetalNumanXHumanMatterPhysicalCompletionStatus::ready
        : MetalNumanXHumanMatterPhysicalCompletionStatus::terminalNoTouch;
    invocation.slotGeneration = prepared.dispatch.slotGeneration;
    prepared.physicalCompletionRegistered = false;
    prepared.physicalCompletionDelivered = true;
    prepared.physicalCompletionContext = nullptr;
    prepared.physicalCompletion = nullptr;
    return invocation;
}

void invokeHumanMatterPhysicalCompletion(
    const HumanMatterPhysicalCompletionInvocation& invocation
) noexcept {
    if (invocation.completion != nullptr && invocation.context != nullptr) {
        invocation.completion(
            invocation.context,
            invocation.status,
            invocation.slotGeneration);
    }
}

struct HumanMatterProposalCompletionInvocation {
    MetalNumanXHumanMatterProposalCompletion completion = nullptr;
    void* context = nullptr;
    MetalNumanXHumanMatterProposalCompletionStatus status =
        MetalNumanXHumanMatterProposalCompletionStatus::terminalNoTouch;
    std::uint64_t slotGeneration = 0u;
};

[[nodiscard]] HumanMatterProposalCompletionInvocation
takeHumanMatterProposalCompletionLocked(
    detail::MetalArticulatedOperatorContextState::
        HumanMatterPreparedRuntimeState& prepared
) noexcept {
    HumanMatterProposalCompletionInvocation invocation{};
    if (!prepared.proposalCompletionRegistered ||
        prepared.proposalCompletionDelivered ||
        prepared.proposalCompletion == nullptr ||
        prepared.proposalCompletionContext == nullptr ||
        prepared.proposalInFlight ||
        (!prepared.proposalComplete && !prepared.proposalFailed &&
         !prepared.terminalNoTouch)) {
        return invocation;
    }
    invocation.completion = prepared.proposalCompletion;
    invocation.context = prepared.proposalCompletionContext;
    invocation.status = prepared.proposalComplete &&
            !prepared.proposalFailed && !prepared.physicalFailure &&
            !prepared.terminalNoTouch
        ? MetalNumanXHumanMatterProposalCompletionStatus::ready
        : MetalNumanXHumanMatterProposalCompletionStatus::terminalNoTouch;
    invocation.slotGeneration = prepared.dispatch.slotGeneration;
    prepared.proposalCompletionRegistered = false;
    prepared.proposalCompletionDelivered = true;
    prepared.proposalCompletionContext = nullptr;
    prepared.proposalCompletion = nullptr;
    return invocation;
}

void invokeHumanMatterProposalCompletion(
    const HumanMatterProposalCompletionInvocation& invocation
) noexcept {
    if (invocation.completion != nullptr && invocation.context != nullptr) {
        invocation.completion(
            invocation.context,
            invocation.status,
            invocation.slotGeneration);
    }
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
    std::string result = nsString(error.localizedDescription);
    if (result.empty()) {
        result = nsString(error.description);
    }
    return result.empty() ? "unknown Metal error" : result;
}

[[nodiscard]] bool exactMetalBuffer(
    id<MTLDevice> device,
    void* raw,
    const std::uint64_t expectedGPUAddress,
    const std::uint64_t minimumBytes
) noexcept {
    if (device == nil || raw == nullptr || expectedGPUAddress == 0u ||
        minimumBytes == 0u ||
        minimumBytes > std::numeric_limits<NSUInteger>::max()) {
        return false;
    }
    __unsafe_unretained id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)raw;
    return buffer != nil && buffer.device == device &&
        buffer.gpuAddress == expectedGPUAddress &&
        static_cast<std::uint64_t>(buffer.length) >= minimumBytes;
}

[[nodiscard]] bool ownedMetalBuffer(
    id<MTLDevice> device,
    id<MTLBuffer> buffer,
    const std::uint64_t minimumBytes
) noexcept {
    return device != nil && buffer != nil && buffer.device == device &&
        buffer.gpuAddress != 0u && minimumBytes != 0u &&
        minimumBytes <= std::numeric_limits<NSUInteger>::max() &&
        static_cast<std::uint64_t>(buffer.length) >= minimumBytes;
}

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
        !error;
}

std::string defaultMetallibPath() {
    Dl_info image{};
    if (dladdr(&kMetalRoboImageAnchor, &image) != 0 &&
        image.dli_fname != nullptr) {
        const std::filesystem::path libraryDirectory =
            std::filesystem::path(image.dli_fname).parent_path();
        const std::array candidates{
            libraryDirectory /
                "metalrobo/MetalRobo.metallib",
            libraryDirectory.parent_path() /
                "shaders/MetalRobo.metallib",
        };
        for (const std::filesystem::path& candidate :
             candidates) {
            if (regularFile(candidate)) {
                return candidate.string();
            }
        }
    }

    const std::filesystem::path configured{
        METALROBO_DEFAULT_METALLIB
    };
    return regularFile(configured)
        ? configured.string()
        : std::string{};
}

MetalArticulatedOperatorDiagnostics reject(
    MetalArticulatedOperatorDiagnostics diagnostics,
    const MetalArticulatedOperatorHostStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checkedAdd(
    const std::size_t left,
    const std::size_t right,
    std::size_t& result
) {
    if (right >
        std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

template <typename T>
bool makeRequirement(
    const char* label,
    const std::size_t logicalElements,
    BufferRequirement& result
) {
    std::size_t logicalBytes = 0u;
    if (!checkedMultiply(
            logicalElements,
            sizeof(T),
            logicalBytes
        )) {
        return false;
    }
    result.label = label;
    result.logicalElements = logicalElements;
    result.logicalBytes = logicalBytes;
    result.allocationBytes =
        logicalBytes == 0u ? sizeof(T) : logicalBytes;
    return result.allocationBytes <=
        std::numeric_limits<NSUInteger>::max();
}

bool finite(const mr_float4 value) {
    return std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool finite(const MRMujocoMuscleResultGPU& value) {
    return finite(value.pathForceAndActivationDerivative) &&
        finite(value.endpointLengthGradients[0]) &&
        finite(value.endpointLengthGradients[1]) &&
        finite(value.activeForceAndReserved) &&
        finite(value.fiberStateTendonForceResidual);
}

bool zero(const mr_float4 value) {
    return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f &&
        value.w == 0.0f;
}

bool isZeroInertiaTransformCarrier(const MRBodyPropertiesGPU& body) {
    return body.motionType == MR_MOTION_STATIC &&
        body.massAndInverseMass.x == 0.0f &&
        body.massAndInverseMass.y == 0.0f &&
        zero(body.inertiaRow0) && zero(body.inertiaRow1) &&
        zero(body.inertiaRow2) && zero(body.inverseInertiaRow0) &&
        zero(body.inverseInertiaRow1) && zero(body.inverseInertiaRow2);
}

bool supportedTopology(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    const bool pointJacobiansOnly,
    std::string& reason
) {
    const std::uint32_t maximumBodies = pointJacobiansOnly
        ? MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_BODIES
        : MR_ARTICULATED_OPERATOR_MAX_BODIES;
    const std::uint32_t maximumDofs = pointJacobiansOnly
        ? MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_DOFS
        : MR_ARTICULATED_OPERATOR_MAX_DOFS;
    if (articulation.bodyCount == 0u ||
        articulation.bodyCount > maximumBodies ||
        articulation.nv == 0u ||
        articulation.nv > maximumDofs) {
        reason =
            pointJacobiansOnly
                ? "articulation exceeds the Metal kinematics/Jacobian "
                  "body/DoF bucket"
                : "articulation exceeds the dense Metal operator "
                  "body/DoF bucket";
        return false;
    }
    const std::size_t jointEnd =
        static_cast<std::size_t>(articulation.firstJoint) +
        articulation.jointCount;
    for (std::size_t jointIndex = articulation.firstJoint;
         jointIndex < jointEnd;
         ++jointIndex) {
        const MRJointDescriptorGPU& joint = model.joints[jointIndex];
        if (joint.flags != 0u) {
            reason =
                "Metal articulated joints require zero reserved flags";
            return false;
        }
        if (joint.jointType != MR_JOINT_REVOLUTE &&
            joint.jointType != MR_JOINT_CONTINUOUS &&
            joint.jointType != MR_JOINT_PRISMATIC &&
            joint.jointType != MR_JOINT_FIXED &&
            joint.jointType != MR_JOINT_FUNCTION_BASED) {
            reason =
                "Metal articulated operator supports revolute, prismatic, "
                "continuous, fixed, and FunctionBased joints";
            return false;
        }
    }
    const std::size_t bodyEnd =
        static_cast<std::size_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (std::size_t bodyIndex = articulation.firstBody;
         bodyIndex < bodyEnd;
         ++bodyIndex) {
        const MRBodyPropertiesGPU& body = model.bodies[bodyIndex];
        if (body.motionType != MR_MOTION_DYNAMIC &&
            !isZeroInertiaTransformCarrier(body)) {
            reason =
                "Metal articulations require dynamic bodies or exact "
                "zero-inertia transform carriers";
            return false;
        }
    }
    return true;
}

bool packFunctionPrograms(
    const EngineModel& model,
    std::vector<MROpenSimSpatialTransformGPU>& packed,
    std::string* reason = nullptr
) {
    packed.assign(std::max<std::size_t>(model.joints.size(), 1u), {});
    for (const FunctionBasedJointProgram& program :
         model.functionBasedJointPrograms) {
        if (program.jointIndex >= model.joints.size()) {
            if (reason != nullptr) {
                *reason = "FunctionBased program joint index is outside model";
            }
            return false;
        }
        const OpenSimSpatialTransformStatus status =
            packOpenSimSpatialTransformGPU(
                program.transform,
                packed[program.jointIndex]
            );
        if (status != OpenSimSpatialTransformStatus::success) {
            if (reason != nullptr) {
                *reason = std::string("FunctionBased program packing failed: ") +
                    openSimSpatialTransformStatusName(status);
            }
            return false;
        }
    }
    return true;
}

bool validQ(
    const MRArticulationGPU& articulation,
    const std::size_t environmentCount,
    const std::span<const float> q
) {
    for (const float value : q) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    if (articulation.rootType != MR_ROOT_FLOATING) {
        return true;
    }
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        const std::size_t base =
            environment * articulation.nq + 3u;
        const float x = q[base + 0u];
        const float y = q[base + 1u];
        const float z = q[base + 2u];
        const float w = q[base + 3u];
        const float normSquared =
            x * x + y * y + z * z + w * w;
        if (!(normSquared > 1.0e-12f) ||
            !std::isfinite(normSquared)) {
            return false;
        }
        const float norm = std::sqrt(normSquared);
        if (!std::isfinite(norm) ||
            std::abs(norm - 1.0f) >
                kQuaternionHostTolerance) {
            return false;
        }
    }
    return true;
}

bool validPoints(
    const MRArticulationGPU& articulation,
    const std::span<const MRArticulatedPointImpulseGPU> points
) {
    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (const MRArticulatedPointImpulseGPU& point : points) {
        if (point.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(point.bodyIndex) >= bodyEnd ||
            point.flags != 0u ||
            point.reserved0 != 0u ||
            point.reserved1 != 0u ||
            !finite(point.localPoint) ||
            !finite(point.worldImpulse) ||
            point.localPoint.w != 0.0f ||
            point.worldImpulse.w != 0.0f) {
            return false;
        }
    }
    return true;
}

bool validMillardReference(
    const MRArticulationGPU& articulation,
    const std::size_t environmentCount,
    const std::span<const MRArticulatedPointImpulseGPU> operatorPoints,
    const MetalMillardReferenceInput& millard,
    std::string& reason
) {
    const bool enabled = millard.enabled();
    if (!enabled) {
        if (!millard.states.empty() || !millard.pathPoints.empty() ||
            !millard.curves.empty() || !millard.cylinderWraps.empty()) {
            reason = "Millard sidecar has data but no muscle definitions";
            return false;
        }
        return true;
    }
    if (millard.muscles.size() > std::numeric_limits<mr_u32>::max() ||
        millard.pathPoints.size() > std::numeric_limits<mr_u32>::max() ||
        millard.cylinderWraps.size() > std::numeric_limits<mr_u32>::max() ||
        millard.curves.size() != millard.muscles.size()) {
        reason = "Millard source dimensions do not fit the device ABI";
        return false;
    }
    std::size_t expectedStateCount = 0u;
    if (!checkedMultiply(
            environmentCount,
            millard.muscles.size(),
            expectedStateCount
        ) || millard.states.size() != expectedStateCount) {
        reason = "Millard state stream is not environment-major";
        return false;
    }
    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (std::size_t index = 0u; index < millard.muscles.size(); ++index) {
        const MRMillardMuscleGPU& muscle = millard.muscles[index];
        if (!finite(muscle.forceAndLengths) ||
            !finite(muscle.dampingAndActivation) ||
            muscle.dampingAndActivation.w != 0.0f ||
            muscle.pathAndWrap.y < 2u ||
            muscle.pathAndWrap.x > millard.pathPoints.size() ||
            muscle.pathAndWrap.y >
                millard.pathPoints.size() - muscle.pathAndWrap.x ||
            muscle.pathAndWrap.z > millard.cylinderWraps.size() ||
            muscle.pathAndWrap.w >
                millard.cylinderWraps.size() - muscle.pathAndWrap.z ||
            muscle.pathAndWrap.w >
                MR_MILLARD_REFERENCE_MAX_WRAPS_PER_MUSCLE ||
            muscle.flags.y != 0u || muscle.flags.z != 0u ||
            muscle.flags.w != 0u) {
            reason = "Millard muscle definition is malformed";
            return false;
        }
        const MRMillardSourceCurveGPU& curve = millard.curves[index];
        for (const mr_float4 block : curve.values) {
            if (!finite(block)) {
                reason = "Millard source curves contain non-finite values";
                return false;
            }
        }
        if (curve.values[5u].z != 0.0f || curve.values[5u].w != 0.0f) {
            reason = "Millard source curve padding is nonzero";
            return false;
        }
        for (std::size_t wrapOffset = 0u;
             wrapOffset < muscle.pathAndWrap.w; ++wrapOffset) {
            const MRMillardCylinderWrapGPU& wrap = millard.cylinderWraps[
                static_cast<std::size_t>(muscle.pathAndWrap.z) + wrapOffset
            ];
            const auto validEndpoint = [&muscle](const mr_i32 endpoint) {
                return endpoint == -1 ||
                    (endpoint >= 1 && static_cast<mr_u32>(endpoint) <=
                        muscle.pathAndWrap.y);
            };
            if (!validEndpoint(wrap.startPoint) || !validEndpoint(wrap.endPoint) ||
                (wrap.startPoint != -1 && wrap.endPoint != -1 &&
                    wrap.startPoint > wrap.endPoint) ||
                wrap.method > MR_MILLARD_PATH_WRAP_AXIAL) {
                reason = "Millard PathWrap range or method is malformed";
                return false;
            }
        }
    }
    for (const MRMillardMuscleStateGPU& state : millard.states) {
        if (!finite(state.activationAndVelocity) ||
            state.activationAndVelocity.z != 0.0f ||
            state.activationAndVelocity.w != 0.0f) {
            reason = "Millard state stream is non-finite or noncanonical";
            return false;
        }
    }
    for (const MRMillardPathPointGPU& point : millard.pathPoints) {
        if (point.pointQueryIndex >= operatorPoints.size() ||
            point.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(point.bodyIndex) >= bodyEnd ||
            point.bodyIndex != operatorPoints[point.pointQueryIndex].bodyIndex ||
            point.reserved0 != 0u || point.reserved1 != 0u) {
            reason = "Millard path point does not match an operator query";
            return false;
        }
    }
    for (const MRMillardCylinderWrapGPU& wrap : millard.cylinderWraps) {
        if (wrap.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(wrap.bodyIndex) >= bodyEnd ||
            !finite(wrap.center) ||
            !finite(wrap.rotationAndRadius) || !finite(wrap.length) ||
            wrap.center.w != 0.0f || wrap.length.y != 0.0f ||
            wrap.length.z != 0.0f || wrap.length.w != 0.0f ||
            !(wrap.rotationAndRadius.w > 0.0f) || !(wrap.length.x > 0.0f)) {
            reason = "Millard cylinder wrap is malformed";
            return false;
        }
    }
    return true;
}

bool validMujocoReference(
    const MRArticulationGPU& articulation,
    const std::size_t environmentCount,
    const std::size_t pointCount,
    const MetalMujocoMuscleReferenceInput& mujoco,
    std::string& reason
) {
    if (!mujoco.enabled()) {
        if (!mujoco.states.empty() || !mujoco.sites.empty() ||
            !mujoco.wraps.empty() || !mujoco.routeNodes.empty()) {
            reason = "MyoSim sidecar has data but no muscle definitions";
            return false;
        }
        return true;
    }
    if (mujoco.bodyJacobianPointOffset == MR_INVALID_INDEX ||
        mujoco.bodyJacobianPointOffset > pointCount ||
        articulation.bodyCount >
            (pointCount - mujoco.bodyJacobianPointOffset) / 4u) {
        reason = "MyoSim sidecar lacks four body-Jacobian probes per body";
        return false;
    }
    if (mujoco.muscles.size() > std::numeric_limits<mr_u32>::max() ||
        mujoco.sites.size() > std::numeric_limits<mr_u32>::max() ||
        mujoco.wraps.size() > std::numeric_limits<mr_u32>::max() ||
        mujoco.routeNodes.size() > std::numeric_limits<mr_u32>::max()) {
        reason = "MyoSim source dimensions do not fit the device ABI";
        return false;
    }
    std::size_t expectedStateCount = 0u;
    if (!checkedMultiply(
            environmentCount,
            mujoco.muscles.size(),
            expectedStateCount
        ) || mujoco.states.size() != expectedStateCount) {
        reason = "MyoSim state stream is not environment-major";
        return false;
    }
    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(articulation.firstBody) +
        articulation.bodyCount;
    for (const MRMujocoMuscleSiteGPU& site : mujoco.sites) {
        if (site.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(site.bodyIndex) >= bodyEnd ||
            site.reserved0 != 0u || site.reserved1 != 0u ||
            site.reserved2 != 0u || !finite(site.localPoint) ||
            site.localPoint.w != 0.0f) {
            reason = "MyoSim site is malformed";
            return false;
        }
    }
    for (const MRMujocoMuscleWrapGPU& wrap : mujoco.wraps) {
        if (wrap.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(wrap.bodyIndex) >= bodyEnd ||
            (wrap.type != MR_MUJOCO_MUSCLE_ROUTE_SPHERE &&
             wrap.type != MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) ||
            wrap.reserved0 != 0u || wrap.reserved1 != 0u ||
            !finite(wrap.localCenter) || !finite(wrap.rotationRow0) ||
            !finite(wrap.rotationRow1) || !finite(wrap.rotationRow2) ||
            !finite(wrap.radius) || wrap.localCenter.w != 0.0f ||
            wrap.rotationRow0.w != 0.0f || wrap.rotationRow1.w != 0.0f ||
            wrap.rotationRow2.w != 0.0f || !(wrap.radius.x > 0.0f) ||
            wrap.radius.y != 0.0f || wrap.radius.z != 0.0f ||
            wrap.radius.w != 0.0f) {
            reason = "MyoSim wrap geometry is malformed";
            return false;
        }
    }
    for (const MRMujocoMuscleRouteNodeGPU& route : mujoco.routeNodes) {
        if ((route.type != MR_MUJOCO_MUSCLE_ROUTE_SITE &&
             route.type != MR_MUJOCO_MUSCLE_ROUTE_SPHERE &&
             route.type != MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) ||
            route.reserved0 != 0u ||
            (route.type == MR_MUJOCO_MUSCLE_ROUTE_SITE
                ? route.targetIndex >= mujoco.sites.size()
                : route.targetIndex >= mujoco.wraps.size()) ||
            (route.sideSiteIndex != MR_INVALID_INDEX &&
             route.sideSiteIndex >= mujoco.sites.size())) {
            reason = "MyoSim route node is malformed";
            return false;
        }
    }
    for (const MRMujocoMuscleGPU& muscle : mujoco.muscles) {
        if (muscle.route.y < 2u || muscle.route.x > mujoco.routeNodes.size() ||
            muscle.route.y > mujoco.routeNodes.size() - muscle.route.x ||
            muscle.route.z != 0u || muscle.route.w != 0u ||
            !finite(muscle.lengthRangeAndAcceleration) ||
            !finite(muscle.controlRange) ||
            !finite(muscle.compliantArchitecture0) ||
            !finite(muscle.compliantArchitecture1) ||
            muscle.lengthRangeAndAcceleration.w != 0.0f ||
            muscle.controlRange.z != 0.0f || muscle.controlRange.w != 0.0f) {
            reason = "MyoSim muscle definition is malformed";
            return false;
        }
    }
    for (const MRMujocoMuscleStateGPU& state : mujoco.states) {
        if (!finite(state.excitationAndActivation) ||
            state.excitationAndActivation.z < 0.0f) {
            reason = "MyoSim muscle state is malformed";
            return false;
        }
    }
    return true;
}

bool validNumiHumanStand(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorConfig& config,
    std::string& reason
) {
    const MetalNumiHumanStandInput& stand = input.stand;
    if (stand.numanXHumanMatterProgram.configured() &&
        !stand.numanXHumanMatterProgram.valid()) {
        reason = "NumanX Human/Matter program is only partially configured";
        return false;
    }
    if (stand.numanXTransactionProgram.configured() &&
        !stand.numanXTransactionProgram.valid()) {
        reason = "NumanX transaction program is only partially configured";
        return false;
    }
    if (!stand.enabled()) {
        if (!stand.v.empty() || !stand.contacts.empty() ||
            !stand.jointEqualities.empty() ||
            !stand.tendonBindings.empty() || !stand.tendonEnvelopes.empty() ||
            stand.tendonLoadProgram.configured() ||
            stand.numanXTransactionProgram.configured() ||
            stand.numanXHumanMatterProgram.configured()) {
            reason = "stand sidecar data or transaction program requires a nonzero stand horizon";
            return false;
        }
        return true;
    }
    if (!config.pointJacobiansOnly || !input.mujoco.enabled() ||
        !(config.mujocoActivationTimestepSeconds > 0.0f)) {
        reason = "stand horizon requires point-Jacobian-only MyoSim with activation stepping";
        return false;
    }
    if (articulation.rootType != MR_ROOT_FLOATING ||
        articulation.bodyCount == 0u ||
        articulation.bodyCount > MR_NUMI_HUMAN_STAND_MAX_BODIES ||
        articulation.nv < 6u || articulation.nv > MR_NUMI_HUMAN_STAND_MAX_DOFS ||
        articulation.nq < 7u || articulation.nq > MR_NUMI_HUMAN_STAND_MAX_Q) {
        reason = "stand horizon requires a supported floating large-state articulation";
        return false;
    }
    if (stand.numanXHumanMatterProgram.valid()) {
        if (stand.stepCount != 1u ||
            articulation.nv !=
                stand.numanXHumanMatterProgram.dofCount ||
            articulation.nq !=
                stand.numanXHumanMatterProgram.qCoordinateCount ||
            stand.numanXHumanMatterProgram.dofLayoutVersion !=
                kMetalNumanXHumanMatterDofLayoutVersion ||
            stand.numanXHumanMatterProgram.environmentCount !=
                input.environmentCount) {
            reason = "NumanX Human/Matter requires one stand step whose exact logical nq/nv identity matches the program and environments";
            return false;
        }
        const bool exactCandidate =
            (stand.numanXHumanMatterProgram.capabilities &
             MetalNumanXHumanMatterExactCandidateKinematics) != 0u;
        const std::uint64_t bodyProbeCount =
            static_cast<std::uint64_t>(articulation.bodyCount) * 4u;
        if (exactCandidate &&
            (bodyProbeCount > MR_ARTICULATED_OPERATOR_MAX_POINTS ||
             stand.numanXHumanMatterProgram.candidatePointCapacity >
                 MR_ARTICULATED_OPERATOR_MAX_POINTS - bodyProbeCount)) {
            reason = "NumanX Human/Matter exact-candidate body probes and attachment capacity exceed the generic operator ceiling";
            return false;
        }
        // A Euclidean square projector is not an exact inverse of the
        // projected effective tangent in general. v1 therefore admits only
        // the unconstrained Human block; a future ABI must carry an exact
        // nullspace basis or KKT/Schur operator.
        if (stand.enableContact || !stand.contacts.empty() ||
            !stand.jointEqualities.empty()) {
            reason = "NumanX Human/Matter v1 rejects contact and joint-equality constrained modes";
            return false;
        }
    }
    if (stand.stepCount > MR_NUMI_HUMAN_STAND_MAX_STEPS ||
        stand.contactIterationCount == 0u ||
        stand.contactIterationCount > 64u ||
        stand.contacts.size() > MR_NUMI_HUMAN_STAND_MAX_CONTACTS ||
        stand.jointEqualities.size() > articulation.nv) {
        reason = "stand step, contact, or iteration count exceeds the device ABI";
        return false;
    }
    std::size_t expectedVelocityCount = 0u;
    if (!checkedMultiply(input.environmentCount, articulation.nv,
                         expectedVelocityCount) ||
        stand.v.size() != expectedVelocityCount ||
        !std::all_of(stand.v.begin(), stand.v.end(), [](const float value) {
            return std::isfinite(value);
        })) {
        reason = "stand velocity stream is not finite environment-major nv state";
        return false;
    }
    if (!finite(stand.groundPoint) || !finite(stand.groundNormal) ||
        !finite(stand.targetRootPosition) ||
        !finite(stand.targetRootOrientation) ||
        !finite(stand.assistanceGains) || stand.groundPoint.w != 0.0f ||
        stand.groundNormal.w != 0.0f || stand.targetRootPosition.w != 0.0f ||
        stand.assistanceGains.x < 0.0f || stand.assistanceGains.y < 0.0f ||
        stand.assistanceGains.z < 0.0f || stand.assistanceGains.w < 0.0f) {
        reason = "stand plane, target, or assistance gains are malformed";
        return false;
    }
    const double normalNormSquared =
        static_cast<double>(stand.groundNormal.x) * stand.groundNormal.x +
        static_cast<double>(stand.groundNormal.y) * stand.groundNormal.y +
        static_cast<double>(stand.groundNormal.z) * stand.groundNormal.z;
    const double targetNormSquared =
        static_cast<double>(stand.targetRootOrientation.x) *
            stand.targetRootOrientation.x +
        static_cast<double>(stand.targetRootOrientation.y) *
            stand.targetRootOrientation.y +
        static_cast<double>(stand.targetRootOrientation.z) *
            stand.targetRootOrientation.z +
        static_cast<double>(stand.targetRootOrientation.w) *
            stand.targetRootOrientation.w;
    if (std::abs(normalNormSquared - 1.0) > 2.0e-4 ||
        (stand.enableRootAssistance &&
         std::abs(targetNormSquared - 1.0) > kQuaternionHostTolerance)) {
        reason = "stand ground normal or assisted target quaternion is not normalized";
        return false;
    }

    std::vector<std::uint8_t> dependentDofs(articulation.nv, 0u);
    for (std::size_t index = 0u;
         index < stand.jointEqualities.size(); ++index) {
        const MRNumiHumanJointEqualityGPU& equality =
            stand.jointEqualities[index];
        const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
            equality.indices.w == MR_INVALID_INDEX;
        const bool coupled = equality.indices.z < articulation.nq &&
            equality.indices.w < articulation.nv;
        const bool dependentMapping =
            equality.indices.x < articulation.nq &&
            equality.indices.y < articulation.nv &&
            model.dofs[articulation.vOffset + equality.indices.y].qIndex ==
                articulation.qOffset + equality.indices.x;
        const bool masterMapping = fixed ||
            (coupled &&
             model.dofs[articulation.vOffset + equality.indices.w].qIndex ==
                 articulation.qOffset + equality.indices.z);
        if (!dependentMapping || (!fixed && !coupled) || !masterMapping ||
            (coupled && (equality.indices.x == equality.indices.z ||
                         equality.indices.y == equality.indices.w)) ||
            !finite(equality.referencesAndCoefficients0) ||
            !finite(equality.coefficients1) || !finite(equality.solref) ||
            !finite(equality.solimp0) || !finite(equality.solimp1) ||
            equality.coefficients1.w != 0.0f || equality.solref.z != 0.0f ||
            equality.solref.w != 0.0f || equality.solimp1.y != 0.0f ||
            equality.solimp1.z != 0.0f || equality.solimp1.w != 0.0f ||
            dependentDofs[equality.indices.y] != 0u) {
            reason = "stand joint equality is malformed, duplicated, or not scalar-owned";
            return false;
        }
        dependentDofs[equality.indices.y] = 1u;
    }
    for (const MRNumiHumanJointEqualityGPU& equality :
         stand.jointEqualities) {
        if (equality.indices.w != MR_INVALID_INDEX &&
            dependentDofs[equality.indices.w] != 0u) {
            reason = "stand joint equality has a chained dependent master";
            return false;
        }
    }

    const std::uint64_t bodyEnd =
        static_cast<std::uint64_t>(articulation.firstBody) +
        articulation.bodyCount;
    if (stand.tendonLoadProgram.configured() &&
        !stand.tendonLoadProgram.valid()) {
        reason = "stand tendon-load consumer is only partially configured";
        return false;
    }
    if (!stand.tendonBindings.empty()) {
        if ((stand.tendonBindings.size() % 2u) != 0u ||
            stand.tendonBindings.size() / 2u != input.mujoco.muscles.size() ||
            stand.tendonBindings.size() >
                std::numeric_limits<mr_u32>::max() ||
            stand.tendonBindings.size() >
                std::numeric_limits<mr_u32>::max() / stand.stepCount ||
            stand.tendonEnvelopes.size() >
                std::numeric_limits<mr_u32>::max()) {
            reason = "stand tendon program does not cover exactly two endpoints per muscle";
            return false;
        }
        for (std::size_t index = 0u;
             index < stand.tendonEnvelopes.size(); ++index) {
            const MRNumiHumanTendonEnvelopeGPU& envelope =
                stand.tendonEnvelopes[index];
            if (envelope.bodyIndex < articulation.firstBody ||
                static_cast<std::uint64_t>(envelope.bodyIndex) >= bodyEnd ||
                envelope.boneStableId == 0u || envelope.nodeCount != 4u ||
                !finite(envelope.metrics) || !(envelope.metrics.y > 0.0f) ||
                !(envelope.metrics.z > 0.0f) || !(envelope.metrics.w > 0.0f)) {
                reason = "stand tendon envelope is malformed or outside the articulation";
                return false;
            }
            for (const mr_float4 node : envelope.localNodes) {
                if (!finite(node) || node.w != 0.0f) {
                    reason = "stand tendon envelope node is malformed";
                    return false;
                }
            }
            for (const mr_float4 row : envelope.forceMapRows) {
                if (!finite(row) || row.w != 0.0f) {
                    reason = "stand tendon envelope force map is malformed";
                    return false;
                }
            }
        }
        for (std::size_t index = 0u;
             index < stand.tendonBindings.size(); ++index) {
            const MRNumiHumanTendonBindingGPU& binding =
                stand.tendonBindings[index];
            if (binding.muscleIndex != index / 2u ||
                binding.endpointOrdinal != index % 2u ||
                binding.bodyIndex < articulation.firstBody ||
                static_cast<std::uint64_t>(binding.bodyIndex) >= bodyEnd ||
                binding.reserved0 != 0u || binding.reserved1 != 0u ||
                !finite(binding.sourceLocalPoint) ||
                binding.sourceLocalPoint.w != 0.0f) {
                reason = "stand tendon binding is malformed, reordered, or outside the articulation";
                return false;
            }
            if (binding.mode == MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT) {
                if (binding.envelopeIndex != MR_INVALID_INDEX ||
                    binding.boneStableId != 0u) {
                    reason = "stand tendon point fallback carries an envelope reference";
                    return false;
                }
            } else if (binding.mode ==
                       MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE) {
                if (binding.envelopeIndex >= stand.tendonEnvelopes.size() ||
                    binding.boneStableId == 0u) {
                    reason = "stand tendon distributed binding has no valid envelope";
                    return false;
                }
                const MRNumiHumanTendonEnvelopeGPU& envelope =
                    stand.tendonEnvelopes[binding.envelopeIndex];
                if (envelope.bodyIndex != binding.bodyIndex ||
                    envelope.boneStableId != binding.boneStableId) {
                    reason = "stand tendon binding and envelope identity disagree";
                    return false;
                }
            } else {
                reason = "stand tendon binding uses an unknown transfer mode";
                return false;
            }
        }
    } else if (!stand.tendonEnvelopes.empty() ||
               stand.tendonLoadProgram.configured()) {
        reason = "stand tendon envelopes or consumer require endpoint bindings";
        return false;
    }
    const std::array<mr_float4, 4u> expectedProbePoints{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    for (std::size_t environment = 0u;
         environment < input.environmentCount; ++environment) {
        const std::size_t environmentBase = environment * input.pointCount;
        for (std::uint32_t localBody = 0u;
             localBody < articulation.bodyCount; ++localBody) {
            for (std::size_t probe = 0u; probe < expectedProbePoints.size(); ++probe) {
                const std::size_t queryIndex =
                    input.mujoco.bodyJacobianPointOffset +
                    4u * localBody + probe;
                if (queryIndex >= input.pointCount) {
                    reason = "stand body-probe query index is outside the point stream";
                    return false;
                }
                const MRArticulatedPointImpulseGPU& query =
                    input.points[environmentBase + queryIndex];
                const mr_float4 expected = expectedProbePoints[probe];
                if (query.bodyIndex != articulation.firstBody + localBody ||
                    query.flags != 0u || query.localPoint.x != expected.x ||
                    query.localPoint.y != expected.y ||
                    query.localPoint.z != expected.z ||
                    query.localPoint.w != expected.w) {
                    reason = "stand body-probe block is not canonical COM/+axis order";
                    return false;
                }
            }
        }
    }
    for (std::size_t contactIndex = 0u;
         contactIndex < stand.contacts.size(); ++contactIndex) {
        const MRNumiHumanStandContactGPU& contact = stand.contacts[contactIndex];
        if (contact.bodyIndex < articulation.firstBody ||
            static_cast<std::uint64_t>(contact.bodyIndex) >= bodyEnd ||
            contact.pointQueryIndex >= input.pointCount ||
            contact.reserved0 != 0u ||
            !finite(contact.frictionSlopAndStabilization) ||
            contact.frictionSlopAndStabilization.x < 0.0f ||
            contact.frictionSlopAndStabilization.y < 0.0f ||
            contact.frictionSlopAndStabilization.z < 0.0f ||
            contact.frictionSlopAndStabilization.z > 1.0f ||
            contact.frictionSlopAndStabilization.w != 0.0f) {
            reason = "stand support-contact record is malformed";
            return false;
        }
        for (std::size_t environment = 0u;
             environment < input.environmentCount; ++environment) {
            const MRArticulatedPointImpulseGPU& query = input.points[
                environment * input.pointCount + contact.pointQueryIndex
            ];
            if (query.bodyIndex != contact.bodyIndex || query.flags != 0u) {
                reason = "stand support contact does not match its active point query";
                return false;
            }
        }
    }
    for (std::uint32_t localDof = 6u;
         localDof < articulation.nv; ++localDof) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + localDof];
        if (dof.qIndex == MR_INVALID_INDEX ||
            dof.qIndex < articulation.qOffset ||
            dof.qIndex >= articulation.qOffset + articulation.nq) {
            reason = "stand horizon requires scalar configuration ownership after the root";
            return false;
        }
    }
    return true;
}

bool buildRequirements(
    const EngineModel& model,
    const MetalArticulatedOperatorLayout& layout,
    RequiredBuffers& requirements,
    std::size_t& totalAllocatedBytes
) {
    std::size_t jointElements =
        std::max<std::size_t>(model.joints.size(), 1u);
    if (!makeRequirement<MRWorldGPU>(
            "world",
            1u,
            requirements.entries[0]
        ) ||
        !makeRequirement<MRArticulationGPU>(
            "articulations",
            model.articulations.size(),
            requirements.entries[1]
        ) ||
        !makeRequirement<MRJointDescriptorGPU>(
            "joints",
            jointElements,
            requirements.entries[2]
        ) ||
        !makeRequirement<MRDofPropertiesGPU>(
            "DoF properties",
            model.dofs.size(),
            requirements.entries[3]
        ) ||
        !makeRequirement<MRBodyPropertiesGPU>(
            "body properties",
            model.bodies.size(),
            requirements.entries[4]
        ) ||
        !makeRequirement<MRArticulatedOperatorDispatchGPU>(
            "dispatch",
            1u,
            requirements.entries[5]
        ) ||
        !makeRequirement<float>(
            "q",
            layout.qElements,
            requirements.entries[6]
        ) ||
        !makeRequirement<MRArticulatedPointImpulseGPU>(
            "point queries",
            layout.pointElements,
            requirements.entries[7]
        ) ||
        !makeRequirement<MRArticulatedBodyPoseGPU>(
            "body poses",
            layout.bodyPoseElements,
            requirements.entries[8]
        ) ||
        !makeRequirement<MRArticulatedPointWorldGPU>(
            "point world",
            layout.pointWorldElements,
            requirements.entries[9]
        ) ||
        !makeRequirement<float>(
            "diagnostic mass",
            layout.massMatrixElements,
            requirements.entries[10]
        ) ||
        !makeRequirement<float>(
            "point Jacobians",
            layout.pointJacobianElements,
            requirements.entries[11]
        ) ||
        !makeRequirement<float>(
            "generalized impulse",
            layout.generalizedElements,
            requirements.entries[12]
        ) ||
        !makeRequirement<float>(
            "delta velocity",
            layout.generalizedElements,
            requirements.entries[13]
        ) ||
        !makeRequirement<MRArticulatedOperatorStatusGPU>(
            "statuses",
            layout.statusElements,
            requirements.entries[14]
        ) ||
        !makeRequirement<MROpenSimSpatialTransformGPU>(
            "FunctionBased programs",
            jointElements,
            requirements.entries[15]
        ) ||
        !makeRequirement<MRMillardReferenceDispatchGPU>(
            "Millard dispatch",
            1u,
            requirements.entries[kMillardDispatchBuffer]
        ) ||
        !makeRequirement<MRMillardMuscleGPU>(
            "Millard muscles",
            layout.millardMuscleElements,
            requirements.entries[kMillardMusclesBuffer]
        ) ||
        !makeRequirement<MRMillardMuscleStateGPU>(
            "Millard states",
            layout.millardStateElements,
            requirements.entries[kMillardStatesBuffer]
        ) ||
        !makeRequirement<MRMillardPathPointGPU>(
            "Millard path points",
            layout.millardPathPointElements,
            requirements.entries[kMillardPathPointsBuffer]
        ) ||
        !makeRequirement<MRMillardSourceCurveGPU>(
            "Millard source curves",
            layout.millardCurveElements,
            requirements.entries[kMillardCurvesBuffer]
        ) ||
        !makeRequirement<MRMillardCylinderWrapGPU>(
            "Millard cylinder wraps",
            layout.millardWrapElements,
            requirements.entries[kMillardWrapsBuffer]
        ) ||
        !makeRequirement<MRMillardMuscleResultGPU>(
            "Millard results",
            layout.millardResultElements,
            requirements.entries[kMillardResultsBuffer]
        ) ||
        !makeRequirement<float>(
            "source-muscle generalized-force workspace",
            std::max(
                layout.millardGeneralizedForceElements,
                layout.mujocoForceWorkspaceElements
            ),
            requirements.entries[kMillardForcesBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleReferenceDispatchGPU>(
            "MyoSim dispatch",
            1u,
            requirements.entries[kMujocoDispatchBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleGPU>(
            "MyoSim muscles",
            layout.mujocoMuscleElements,
            requirements.entries[kMujocoMusclesBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleStateGPU>(
            "MyoSim states",
            layout.mujocoStateElements,
            requirements.entries[kMujocoStatesBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleSiteGPU>(
            "MyoSim sites",
            layout.mujocoSiteElements,
            requirements.entries[kMujocoSitesBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleWrapGPU>(
            "MyoSim wraps",
            layout.mujocoWrapElements,
            requirements.entries[kMujocoWrapsBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleRouteNodeGPU>(
            "MyoSim route nodes",
            layout.mujocoRouteNodeElements,
            requirements.entries[kMujocoRoutesBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleResultGPU>(
            "MyoSim results",
            layout.mujocoResultElements,
            requirements.entries[kMujocoResultsBuffer]
        ) ||
        !makeRequirement<float>(
            "Numi Human stand velocity",
            layout.standVelocityElements,
            requirements.standEntries[kStandVelocityBuffer]
        ) ||
        !makeRequirement<MRNumiHumanStandContactGPU>(
            "Numi Human stand contacts",
            layout.standContactElements,
            requirements.standEntries[kStandContactsBuffer]
        ) ||
        !makeRequirement<float>(
            "Numi Human stand spatial Jacobians",
            layout.standSpatialJacobianElements,
            requirements.standEntries[kStandSpatialJacobianBuffer]
        ) ||
        !makeRequirement<mr_float4>(
            "Numi Human stand body motion",
            layout.standBodyMotionElements,
            requirements.standEntries[kStandBodyMotionBuffer]
        ) ||
        !makeRequirement<float>(
            "Numi Human stand factor",
            layout.standFactorElements,
            requirements.standEntries[kStandFactorBuffer]
        ) ||
        !makeRequirement<float>(
            "Numi Human stand vectors",
            layout.standVectorElements,
            requirements.standEntries[kStandVectorBuffer]
        ) ||
        !makeRequirement<float>(
            "Numi Human stand constraint response",
            layout.standResponseElements,
            requirements.standEntries[kStandResponseBuffer]
        ) ||
        !makeRequirement<MRNumiHumanStandStatusGPU>(
            "Numi Human stand statuses",
            layout.standStatusElements,
            requirements.standEntries[kStandStatusBuffer]
        ) ||
        !makeRequirement<MRNumiHumanTendonBindingGPU>(
            "Numi Human tendon bindings",
            layout.standTendonBindingElements,
            requirements.standEntries[kStandTendonBindingsBuffer]
        ) ||
        !makeRequirement<MRNumiHumanTendonEnvelopeGPU>(
            "Numi Human tendon envelopes",
            layout.standTendonEnvelopeElements,
            requirements.standEntries[kStandTendonEnvelopesBuffer]
        ) ||
        !makeRequirement<MRNumiHumanTendonTransferResultGPU>(
            "Numi Human tendon transfer results",
            layout.standTendonTransferElements,
            requirements.standEntries[kStandTendonTransfersBuffer]
        ) ||
        !makeRequirement<float>(
            "Numi Human tendon generalized corrections",
            layout.standTendonCorrectionElements,
            requirements.standEntries[kStandTendonCorrectionsBuffer]
        ) ||
        !makeRequirement<MRNumiHumanJointEqualityGPU>(
            "Numi Human joint equalities",
            layout.standJointEqualityElements,
            requirements.standEntries[kStandJointEqualitiesBuffer]
        ) ||
        !makeRequirement<float>(
            "NumanX Human/Matter q checkpoint",
            layout.humanMatterQCheckpointElements,
            requirements.humanMatterEntries[
                kHumanMatterQCheckpointBuffer]
        ) ||
        !makeRequirement<float>(
            "NumanX Human/Matter v checkpoint",
            layout.humanMatterVCheckpointElements,
            requirements.humanMatterEntries[
                kHumanMatterVCheckpointBuffer]
        ) ||
        !makeRequirement<MRMujocoMuscleStateGPU>(
            "NumanX Human/Matter MyoSim checkpoint",
            layout.humanMatterMujocoCheckpointElements,
            requirements.humanMatterEntries[
                kHumanMatterMujocoCheckpointBuffer]
        ) ||
        !makeRequirement<MRNumanXHumanMatterOwnerStatusGPU>(
            "NumanX Human/Matter owner statuses",
            layout.humanMatterOwnerStatusElements,
            requirements.humanMatterEntries[
                kHumanMatterOwnerStatusBuffer]
        ) ||
        !makeRequirement<MRArticulatedPointImpulseGPU>(
            "NumanX Human/Matter candidate points",
            layout.humanMatterCandidatePointElements,
            requirements.humanMatterEntries[
                kHumanMatterCandidatePointBuffer]
        ) ||
        !makeRequirement<MRArticulatedBodyPoseGPU>(
            "NumanX Human/Matter candidate body poses",
            layout.humanMatterCandidateBodyPoseElements,
            requirements.humanMatterEntries[
                kHumanMatterCandidateBodyPoseBuffer]
        ) ||
        !makeRequirement<MRArticulatedPointWorldGPU>(
            "NumanX Human/Matter candidate point world",
            layout.humanMatterCandidatePointWorldElements,
            requirements.humanMatterEntries[
                kHumanMatterCandidatePointWorldBuffer]
        ) ||
        !makeRequirement<float>(
            "NumanX Human/Matter candidate point Jacobians",
            layout.humanMatterCandidatePointJacobianElements,
            requirements.humanMatterEntries[
                kHumanMatterCandidatePointJacobianBuffer]
        ) ||
        !makeRequirement<float>(
            "NumanX Human/Matter candidate operator scratch",
            layout.humanMatterCandidateOperatorScratchElements,
            requirements.humanMatterEntries[
                kHumanMatterCandidateOperatorScratchBuffer]
        ) ||
        !makeRequirement<MRArticulatedOperatorStatusGPU>(
            "NumanX Human/Matter candidate operator statuses",
            layout.humanMatterCandidateOperatorStatusElements,
            requirements.humanMatterEntries[
                kHumanMatterCandidateOperatorStatusBuffer]
        ) ||
        !makeRequirement<MRNumanXHumanMatterAppliedOutcomeGPU>(
            "NumanX Human/Matter applied outcomes",
            layout.humanMatterAppliedOutcomeElements,
            requirements.humanMatterEntries[
                kHumanMatterAppliedOutcomeBuffer]
        ) ||
        !makeRequirement<std::uint8_t>(
            "NumanX Human/Matter final accepted tokens",
            layout.humanMatterFinalAcceptedTokenElements,
            requirements.humanMatterEntries[
                kHumanMatterFinalAcceptedTokenBuffer]
        ) ||
        !makeRequirement<MRNumanXHumanMatterProposalGPU>(
            "NumanX Human/Matter proposals",
            layout.humanMatterProposalElements,
            requirements.humanMatterEntries[kHumanMatterProposalBuffer]
        ) ||
        !makeRequirement<std::uint8_t>(
            "NumanX Human/Matter proposed tokens",
            layout.humanMatterProposedTokenElements,
            requirements.humanMatterEntries[kHumanMatterProposedTokenBuffer]
        ) ||
        !makeRequirement<MRNumanXHumanMatterApplyActionGPU>(
            "NumanX Human/Matter apply actions",
            layout.humanMatterApplyActionElements,
            requirements.humanMatterEntries[kHumanMatterApplyActionBuffer]
        ) ||
        !makeRequirement<MRNumanXHumanMatterJointPublicationFenceGPU>(
            "NumanX Human/Matter publication fences",
            layout.humanMatterPublicationFenceElements,
            requirements.humanMatterEntries[
                kHumanMatterPublicationFenceBuffer]
        )) {
        return false;
    }

    // The historical operator does not bind the separate stand arena. Keep
    // its cold allocation/stats contract unchanged unless a horizon exists.
    if (layout.standStatusElements == 0u) {
        for (std::size_t index = kStandContactsBuffer;
             index < kStandBufferCount; ++index) {
            requirements.standEntries[index].allocationBytes = 0u;
        }
    }

    totalAllocatedBytes = 0u;
    for (const BufferRequirement& requirement :
         requirements.entries) {
        if (!checkedAdd(
                totalAllocatedBytes,
                requirement.allocationBytes,
                totalAllocatedBytes
            )) {
            return false;
        }
    }
    for (const BufferRequirement& requirement :
         requirements.standEntries) {
        if (!checkedAdd(
                totalAllocatedBytes,
                requirement.allocationBytes,
                totalAllocatedBytes
            )) {
            return false;
        }
    }
    for (const BufferRequirement& requirement :
         requirements.humanMatterEntries) {
        if (!checkedAdd(
                totalAllocatedBytes,
                requirement.allocationBytes,
                totalAllocatedBytes
            )) {
            return false;
        }
    }
    return true;
}

MetalArticulatedOperatorDiagnostics validateAndBuildLayout(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorConfig& config,
    RequiredBuffers& requirements
) {
    MetalArticulatedOperatorDiagnostics diagnostics{};

    if (!std::isfinite(config.mujocoActivationTimestepSeconds) ||
        config.mujocoActivationTimestepSeconds < 0.0f ||
        config.mujocoActivationTimestepSeconds > 0.01f) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "MyoSim activation timestep must be finite and within [0, 0.01] seconds"
        );
    }

    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidModel,
            "invalid EngineModel: " + modelReason
        );
    }
    std::vector<MROpenSimSpatialTransformGPU> packedPrograms;
    if (!packFunctionPrograms(model, packedPrograms, &modelReason)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidModel,
            "invalid FunctionBased program stream: " + modelReason
        );
    }
    if (input.articulationIndex >=
        model.articulations.size()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "articulation index is outside the canonical model"
        );
    }
    if (input.environmentCount == 0u) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "environmentCount must be greater than zero"
        );
    }
    if (input.environmentCount >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "environmentCount does not fit the GPU dispatch ABI"
        );
    }
    if (input.pointCount >
        std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "pointCount does not fit the GPU dispatch ABI"
        );
    }
    if (input.pointCount >
        MR_ARTICULATED_OPERATOR_MAX_POINTS) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::capacityOverflow,
            "pointCount exceeds the compiled operator capacity"
        );
    }

    const MRArticulationGPU& articulation =
        model.articulations[input.articulationIndex];
    std::string topologyReason;
    if (!supportedTopology(
            model,
            articulation,
            config.pointJacobiansOnly,
            topologyReason
        )) {
        const std::uint32_t maximumBodies = config.pointJacobiansOnly
            ? MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_BODIES
            : MR_ARTICULATED_OPERATOR_MAX_BODIES;
        const std::uint32_t maximumDofs = config.pointJacobiansOnly
            ? MR_ARTICULATED_OPERATOR_KINEMATICS_MAX_DOFS
            : MR_ARTICULATED_OPERATOR_MAX_DOFS;
        const bool capacity = articulation.bodyCount > maximumBodies ||
            articulation.nv > maximumDofs;
        return reject(
            std::move(diagnostics),
            capacity
                ? MetalArticulatedOperatorHostStatus::
                      capacityOverflow
                : MetalArticulatedOperatorHostStatus::
                      unsupportedTopology,
            std::move(topologyReason)
        );
    }

    MetalArticulatedOperatorLayout layout{};
    MRArticulatedOperatorDispatchGPU& dispatch =
        layout.dispatch;
    dispatch.articulationIndex = input.articulationIndex;
    dispatch.environmentCount =
        static_cast<mr_u32>(input.environmentCount);
    dispatch.pointCount =
        static_cast<mr_u32>(input.pointCount);
    dispatch.flags = 0u;
    if (config.writeDiagnosticMassMatrix) {
        dispatch.flags |=
            MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS;
    }
    if (config.pointJacobiansOnly) {
        dispatch.flags |=
            MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY;
    }
    if (config.writeDiagnosticMassMatrix &&
        config.pointJacobiansOnly) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                invalidDimensions,
            "point-Jacobian-only mode cannot request a "
            "diagnostic mass matrix"
        );
    }
    if (input.millard.enabled() && input.mujoco.enabled()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "Millard and MyoSim source-muscle sidecars cannot share one "
            "operator submission"
        );
    }
    dispatch.qStride = articulation.nq;
    dispatch.pointStride = dispatch.pointCount;
    dispatch.bodyPoseStride = articulation.bodyCount;
    dispatch.pointWorldStride = dispatch.pointCount;
    dispatch.generalizedStride = articulation.nv;

    std::size_t massStride = 0u;
    std::size_t jacobianStride = 0u;
    if (!checkedMultiply(
            articulation.nv,
            articulation.nv,
            massStride
        ) ||
        !checkedMultiply(
            input.pointCount,
            3u,
            jacobianStride
        ) ||
        !checkedMultiply(
            jacobianStride,
            articulation.nv,
            jacobianStride
        ) ||
        massStride > std::numeric_limits<mr_u32>::max() ||
        jacobianStride > std::numeric_limits<mr_u32>::max()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "derived GPU stride overflow"
        );
    }
    dispatch.massMatrixStride =
        config.writeDiagnosticMassMatrix
            ? static_cast<mr_u32>(massStride)
            : 0u;
    dispatch.pointJacobianStride =
        static_cast<mr_u32>(jacobianStride);

    if (!checkedMultiply(
            input.environmentCount,
            dispatch.qStride,
            layout.qElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.pointStride,
            layout.pointElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.bodyPoseStride,
            layout.bodyPoseElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.pointWorldStride,
            layout.pointWorldElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.massMatrixStride,
            layout.massMatrixElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.pointJacobianStride,
            layout.pointJacobianElements
        ) ||
        !checkedMultiply(
            input.environmentCount,
            dispatch.generalizedStride,
            layout.generalizedElements
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "derived GPU element-count overflow"
        );
    }
    layout.statusElements = input.environmentCount;

    if (input.millard.enabled()) {
        layout.millardMuscleElements = input.millard.muscles.size();
        layout.millardStateElements = input.millard.states.size();
        layout.millardPathPointElements = input.millard.pathPoints.size();
        layout.millardCurveElements = input.millard.curves.size();
        layout.millardWrapElements = input.millard.cylinderWraps.size();
        if (!checkedMultiply(
                input.environmentCount,
                layout.millardMuscleElements,
                layout.millardResultElements
            ) || !checkedMultiply(
                layout.millardResultElements,
                articulation.nv,
                layout.millardGeneralizedForceElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "derived Millard output element-count overflow"
            );
        }
    }
    if (input.mujoco.enabled()) {
        layout.mujocoMuscleElements = input.mujoco.muscles.size();
        layout.mujocoStateElements = input.mujoco.states.size();
        layout.mujocoSiteElements = input.mujoco.sites.size();
        layout.mujocoWrapElements = input.mujoco.wraps.size();
        layout.mujocoRouteNodeElements = input.mujoco.routeNodes.size();
        if (!checkedMultiply(
                input.environmentCount,
                layout.mujocoMuscleElements,
                layout.mujocoResultElements
            ) || !checkedMultiply(
                layout.mujocoResultElements,
                articulation.nv,
                layout.mujocoMuscleGeneralizedForceElements
            ) || !checkedMultiply(
                input.environmentCount,
                articulation.nv,
                layout.mujocoGeneralizedForceElements
            ) || !checkedAdd(
                layout.mujocoMuscleGeneralizedForceElements,
                layout.mujocoGeneralizedForceElements,
                layout.mujocoForceWorkspaceElements
            )) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "derived MyoSim output element-count overflow"
            );
        }
    }
    if (input.stand.enabled()) {
        layout.standVelocityElements = input.stand.v.size();
        layout.standContactElements = input.stand.contacts.size();
        layout.standJointEqualityElements =
            input.stand.jointEqualities.size();
        layout.standStatusElements = input.environmentCount;
        layout.standTendonBindingElements =
            input.stand.tendonBindings.size();
        layout.standTendonEnvelopeElements =
            input.stand.tendonEnvelopes.size();
        std::size_t bodyDofs = 0u;
        std::size_t environmentBodyDofs = 0u;
        std::size_t bodyMotionPerEnvironment = 0u;
        std::size_t factorPerEnvironment = 0u;
        std::size_t vectorPerEnvironment = 0u;
        std::size_t constraintVectorElements = 0u;
        std::size_t responsePerEnvironment = 0u;
        if (!checkedMultiply(
                input.environmentCount,
                layout.standTendonBindingElements,
                layout.standTendonTransferElements
            ) ||
            !checkedMultiply(
                layout.standTendonTransferElements,
                articulation.nv,
                layout.standTendonCorrectionElements
            ) ||
            !checkedMultiply(articulation.bodyCount, articulation.nv, bodyDofs) ||
            !checkedMultiply(bodyDofs, 6u, bodyDofs) ||
            !checkedMultiply(input.environmentCount, bodyDofs,
                             layout.standSpatialJacobianElements) ||
            !checkedMultiply(articulation.bodyCount, 2u,
                             bodyMotionPerEnvironment) ||
            !checkedMultiply(input.environmentCount, bodyMotionPerEnvironment,
                             layout.standBodyMotionElements) ||
            !checkedMultiply(articulation.nv, articulation.nv,
                             factorPerEnvironment) ||
            !checkedMultiply(input.environmentCount, factorPerEnvironment,
                             layout.standFactorElements) ||
            !checkedMultiply(input.stand.contacts.size(), 12u,
                             constraintVectorElements) ||
            !checkedAdd(constraintVectorElements,
                        input.stand.jointEqualities.size(),
                        constraintVectorElements) ||
            !checkedMultiply(articulation.nv, 3u,
                             vectorPerEnvironment) ||
            !checkedAdd(vectorPerEnvironment, constraintVectorElements,
                        vectorPerEnvironment) ||
            !checkedMultiply(input.environmentCount, vectorPerEnvironment,
                             layout.standVectorElements) ||
            !checkedMultiply(input.stand.contacts.size(), 3u,
                             responsePerEnvironment) ||
            !checkedAdd(responsePerEnvironment,
                        input.stand.jointEqualities.size(),
                        responsePerEnvironment) ||
            !checkedMultiply(responsePerEnvironment, articulation.nv,
                             responsePerEnvironment) ||
            !checkedMultiply(input.environmentCount, responsePerEnvironment,
                             layout.standResponseElements)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "derived Numi Human stand or tendon element-count overflow"
            );
        }
        if (input.stand.numanXHumanMatterProgram.valid()) {
            layout.humanMatterQCheckpointElements = layout.qElements;
            layout.humanMatterVCheckpointElements =
                layout.standVelocityElements;
            layout.humanMatterMujocoCheckpointElements =
                layout.mujocoStateElements;
            layout.humanMatterSourceFactorElements =
                layout.standFactorElements;
            layout.humanMatterOwnerStatusElements = input.environmentCount;
            layout.humanMatterProposalElements =
                input.environmentCount;
            layout.humanMatterApplyActionElements =
                input.environmentCount;
            layout.humanMatterAppliedOutcomeElements =
                input.environmentCount;
            layout.humanMatterPublicationFenceElements =
                input.environmentCount;
            if (!checkedMultiply(
                    input.environmentCount,
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
                    layout.humanMatterFinalAcceptedTokenElements) ||
                !checkedMultiply(
                    input.environmentCount,
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
                    layout.humanMatterProposedTokenElements)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                    "derived NumanX final-token arena overflow"
                );
            }
            const MetalNumanXHumanMatterProgram& program =
                input.stand.numanXHumanMatterProgram;
            if ((program.capabilities &
                 MetalNumanXHumanMatterExactCandidateKinematics) != 0u) {
                std::size_t combinedPointCapacity = 0u;
                std::size_t pointJacobianCapacity = 0u;
                if (!checkedMultiply(
                        articulation.bodyCount, 4u,
                        combinedPointCapacity) ||
                    !checkedAdd(
                        combinedPointCapacity,
                        program.candidatePointCapacity,
                        combinedPointCapacity) ||
                    combinedPointCapacity >
                        MR_ARTICULATED_OPERATOR_MAX_POINTS ||
                    !checkedMultiply(
                        input.environmentCount,
                        combinedPointCapacity,
                        layout.humanMatterCandidatePointElements) ||
                    !checkedMultiply(
                        input.environmentCount,
                        articulation.bodyCount,
                        layout.humanMatterCandidateBodyPoseElements) ||
                    !checkedMultiply(
                        combinedPointCapacity, 3u,
                        pointJacobianCapacity) ||
                    !checkedMultiply(
                        pointJacobianCapacity, articulation.nv,
                        pointJacobianCapacity) ||
                    !checkedMultiply(
                        input.environmentCount,
                        pointJacobianCapacity,
                        layout.humanMatterCandidatePointJacobianElements) ||
                    !checkedMultiply(
                        input.environmentCount,
                        articulation.nv,
                        layout.humanMatterCandidateOperatorScratchElements)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                        "derived NumanX exact-candidate arena overflow"
                    );
                }
                layout.humanMatterCandidatePointWorldElements =
                    layout.humanMatterCandidatePointElements;
                layout.humanMatterCandidateOperatorStatusElements =
                    input.environmentCount;
            }
        }
        (void)environmentBodyDofs;
    }
    if (!input.stand.enabled() && input.mujoco.enabled()) {
        layout.standVelocityElements = layout.generalizedElements;
    }

    const auto exceedsShaderAddressing =
        [](const std::size_t elements) {
            return static_cast<std::uint64_t>(elements) >
                kShaderAddressableElements;
        };
    if (exceedsShaderAddressing(layout.qElements) ||
        exceedsShaderAddressing(layout.pointElements) ||
        exceedsShaderAddressing(layout.bodyPoseElements) ||
        exceedsShaderAddressing(layout.pointWorldElements) ||
        exceedsShaderAddressing(layout.massMatrixElements) ||
        exceedsShaderAddressing(
            layout.pointJacobianElements
        ) ||
        exceedsShaderAddressing(layout.generalizedElements) ||
        exceedsShaderAddressing(layout.statusElements) ||
        exceedsShaderAddressing(layout.millardMuscleElements) ||
        exceedsShaderAddressing(layout.millardStateElements) ||
        exceedsShaderAddressing(layout.millardPathPointElements) ||
        exceedsShaderAddressing(layout.millardCurveElements) ||
        exceedsShaderAddressing(layout.millardWrapElements) ||
        exceedsShaderAddressing(layout.millardResultElements) ||
        exceedsShaderAddressing(layout.millardGeneralizedForceElements) ||
        exceedsShaderAddressing(layout.mujocoMuscleElements) ||
        exceedsShaderAddressing(layout.mujocoStateElements) ||
        exceedsShaderAddressing(layout.mujocoSiteElements) ||
        exceedsShaderAddressing(layout.mujocoWrapElements) ||
        exceedsShaderAddressing(layout.mujocoRouteNodeElements) ||
        exceedsShaderAddressing(layout.mujocoResultElements) ||
        exceedsShaderAddressing(
            layout.mujocoMuscleGeneralizedForceElements
        ) ||
        exceedsShaderAddressing(layout.mujocoGeneralizedForceElements) ||
        exceedsShaderAddressing(layout.standVelocityElements) ||
        exceedsShaderAddressing(layout.standContactElements) ||
        exceedsShaderAddressing(layout.standJointEqualityElements) ||
        exceedsShaderAddressing(layout.standSpatialJacobianElements) ||
        exceedsShaderAddressing(layout.standBodyMotionElements) ||
        exceedsShaderAddressing(layout.standFactorElements) ||
        exceedsShaderAddressing(layout.standVectorElements) ||
        exceedsShaderAddressing(layout.standResponseElements) ||
        exceedsShaderAddressing(layout.standStatusElements) ||
        exceedsShaderAddressing(layout.standTendonBindingElements) ||
        exceedsShaderAddressing(layout.standTendonEnvelopeElements) ||
        exceedsShaderAddressing(layout.standTendonTransferElements) ||
        exceedsShaderAddressing(layout.standTendonCorrectionElements) ||
        exceedsShaderAddressing(
            layout.humanMatterQCheckpointElements) ||
        exceedsShaderAddressing(
            layout.humanMatterVCheckpointElements) ||
        exceedsShaderAddressing(
            layout.humanMatterMujocoCheckpointElements) ||
        exceedsShaderAddressing(
            layout.humanMatterSourceFactorElements) ||
        exceedsShaderAddressing(
            layout.humanMatterOwnerStatusElements) ||
        exceedsShaderAddressing(
            layout.humanMatterCandidatePointElements) ||
        exceedsShaderAddressing(
            layout.humanMatterCandidateBodyPoseElements) ||
        exceedsShaderAddressing(
            layout.humanMatterCandidatePointWorldElements) ||
        exceedsShaderAddressing(
            layout.humanMatterCandidatePointJacobianElements) ||
        exceedsShaderAddressing(
            layout.humanMatterCandidateOperatorScratchElements) ||
        exceedsShaderAddressing(
            layout.humanMatterCandidateOperatorStatusElements) ||
        exceedsShaderAddressing(
            layout.humanMatterProposalElements) ||
        exceedsShaderAddressing(
            layout.humanMatterProposedTokenElements) ||
        exceedsShaderAddressing(
            layout.humanMatterApplyActionElements) ||
        exceedsShaderAddressing(
            layout.humanMatterAppliedOutcomeElements) ||
        exceedsShaderAddressing(
            layout.humanMatterPublicationFenceElements) ||
        exceedsShaderAddressing(
            layout.humanMatterFinalAcceptedTokenElements)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "compact buffer exceeds the shader's 32-bit element "
            "addressing contract"
        );
    }

    std::size_t totalAllocatedBytes = 0u;
    if (!buildRequirements(
            model,
            layout,
            requirements,
            totalAllocatedBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "required Metal buffer byte-count overflow"
        );
    }
    layout.qBytes = requirements.entries[6].logicalBytes;
    layout.pointBytes = requirements.entries[7].logicalBytes;
    layout.bodyPoseBytes =
        requirements.entries[8].logicalBytes;
    layout.pointWorldBytes =
        requirements.entries[9].logicalBytes;
    layout.massMatrixBytes =
        requirements.entries[10].logicalBytes;
    layout.pointJacobianBytes =
        requirements.entries[11].logicalBytes;
    layout.generalizedBytes =
        requirements.entries[12].logicalBytes;
    layout.statusBytes =
        requirements.entries[14].logicalBytes;
    layout.millardMuscleBytes =
        requirements.entries[kMillardMusclesBuffer].logicalBytes;
    layout.millardStateBytes =
        requirements.entries[kMillardStatesBuffer].logicalBytes;
    layout.millardPathPointBytes =
        requirements.entries[kMillardPathPointsBuffer].logicalBytes;
    layout.millardCurveBytes =
        requirements.entries[kMillardCurvesBuffer].logicalBytes;
    layout.millardWrapBytes =
        requirements.entries[kMillardWrapsBuffer].logicalBytes;
    layout.millardResultBytes =
        requirements.entries[kMillardResultsBuffer].logicalBytes;
    layout.millardGeneralizedForceBytes =
        requirements.entries[kMillardForcesBuffer].logicalBytes;
    layout.mujocoMuscleBytes =
        requirements.entries[kMujocoMusclesBuffer].logicalBytes;
    layout.mujocoStateBytes =
        requirements.entries[kMujocoStatesBuffer].logicalBytes;
    layout.mujocoSiteBytes =
        requirements.entries[kMujocoSitesBuffer].logicalBytes;
    layout.mujocoWrapBytes =
        requirements.entries[kMujocoWrapsBuffer].logicalBytes;
    layout.mujocoRouteNodeBytes =
        requirements.entries[kMujocoRoutesBuffer].logicalBytes;
    layout.mujocoResultBytes =
        requirements.entries[kMujocoResultsBuffer].logicalBytes;
    layout.standVelocityBytes =
        requirements.standEntries[kStandVelocityBuffer].logicalBytes;
    layout.standContactBytes =
        requirements.standEntries[kStandContactsBuffer].logicalBytes;
    layout.standJointEqualityBytes =
        requirements.standEntries[kStandJointEqualitiesBuffer].logicalBytes;
    layout.standStatusBytes =
        requirements.standEntries[kStandStatusBuffer].logicalBytes;
    layout.standTendonBindingBytes =
        requirements.standEntries[kStandTendonBindingsBuffer].logicalBytes;
    layout.standTendonEnvelopeBytes =
        requirements.standEntries[kStandTendonEnvelopesBuffer].logicalBytes;
    layout.standTendonTransferBytes =
        requirements.standEntries[kStandTendonTransfersBuffer].logicalBytes;
    layout.standTendonCorrectionBytes =
        requirements.standEntries[kStandTendonCorrectionsBuffer].logicalBytes;
    layout.humanMatterQCheckpointBytes =
        requirements.humanMatterEntries[
            kHumanMatterQCheckpointBuffer].logicalBytes;
    layout.humanMatterVCheckpointBytes =
        requirements.humanMatterEntries[
            kHumanMatterVCheckpointBuffer].logicalBytes;
    layout.humanMatterMujocoCheckpointBytes =
        requirements.humanMatterEntries[
            kHumanMatterMujocoCheckpointBuffer].logicalBytes;
    layout.humanMatterSourceFactorBytes =
        requirements.standEntries[kStandFactorBuffer].logicalBytes;
    layout.humanMatterOwnerStatusBytes =
        requirements.humanMatterEntries[
            kHumanMatterOwnerStatusBuffer].logicalBytes;
    layout.humanMatterCandidatePointBytes =
        requirements.humanMatterEntries[
            kHumanMatterCandidatePointBuffer].logicalBytes;
    layout.humanMatterCandidateBodyPoseBytes =
        requirements.humanMatterEntries[
            kHumanMatterCandidateBodyPoseBuffer].logicalBytes;
    layout.humanMatterCandidatePointWorldBytes =
        requirements.humanMatterEntries[
            kHumanMatterCandidatePointWorldBuffer].logicalBytes;
    layout.humanMatterCandidatePointJacobianBytes =
        requirements.humanMatterEntries[
            kHumanMatterCandidatePointJacobianBuffer].logicalBytes;
    layout.humanMatterCandidateOperatorScratchBytes =
        requirements.humanMatterEntries[
            kHumanMatterCandidateOperatorScratchBuffer].logicalBytes;
    layout.humanMatterCandidateOperatorStatusBytes =
        requirements.humanMatterEntries[
            kHumanMatterCandidateOperatorStatusBuffer].logicalBytes;
    layout.humanMatterAppliedOutcomeBytes =
        requirements.humanMatterEntries[
            kHumanMatterAppliedOutcomeBuffer].logicalBytes;
    layout.humanMatterFinalAcceptedTokenBytes =
        requirements.humanMatterEntries[
            kHumanMatterFinalAcceptedTokenBuffer].logicalBytes;
    layout.humanMatterProposalBytes = requirements.humanMatterEntries[
        kHumanMatterProposalBuffer].logicalBytes;
    layout.humanMatterProposedTokenBytes =
        requirements.humanMatterEntries[
            kHumanMatterProposedTokenBuffer].logicalBytes;
    layout.humanMatterApplyActionBytes = requirements.humanMatterEntries[
        kHumanMatterApplyActionBuffer].logicalBytes;
    layout.humanMatterPublicationFenceBytes =
        requirements.humanMatterEntries[
            kHumanMatterPublicationFenceBuffer].logicalBytes;
    layout.standScratchBytes = 0u;
    for (std::size_t index = kStandSpatialJacobianBuffer;
         index <= kStandResponseBuffer; ++index) {
        if (!checkedAdd(
                layout.standScratchBytes,
                requirements.standEntries[index].logicalBytes,
                layout.standScratchBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "derived Numi Human stand scratch byte-count overflow"
            );
        }
    }
    if (!checkedMultiply(
            layout.mujocoMuscleGeneralizedForceElements,
            sizeof(float),
            layout.mujocoMuscleGeneralizedForceBytes
        ) || !checkedMultiply(
            layout.mujocoGeneralizedForceElements,
            sizeof(float),
            layout.mujocoGeneralizedForceBytes
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::arithmeticOverflow,
            "derived MyoSim force-output byte-count overflow"
        );
    }
    layout.totalAllocatedBytes = totalAllocatedBytes;
    diagnostics.layout = layout;

    if (input.q.size() != layout.qElements ||
        input.points.size() != layout.pointElements) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "packed q or point-query span has the wrong element count"
        );
    }
    if (!input.v.empty() && input.v.size() != layout.generalizedElements) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "optional MyoSim velocity span has the wrong element count"
        );
    }
    if (!std::all_of(input.v.begin(), input.v.end(), [](const float value) {
            return std::isfinite(value);
        })) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::nonfiniteInput,
            "optional MyoSim velocity span contains a non-finite value"
        );
    }
    if (!validQ(
            articulation,
            input.environmentCount,
            input.q
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::nonfiniteInput,
            "q contains a non-finite value or invalid root quaternion"
        );
    }
    if (!validPoints(articulation, input.points)) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidPointQuery,
            "point query is non-finite, reserved, or outside articulation"
        );
    }
    std::string millardReason;
    if (!validMillardReference(
            articulation,
            input.environmentCount,
            input.points,
            input.millard,
            millardReason
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "invalid Millard reference program: " + millardReason
        );
    }
    std::string mujocoReason;
    if (!validMujocoReference(
            articulation,
            input.environmentCount,
            input.pointCount,
            input.mujoco,
            mujocoReason
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "invalid MyoSim reference program: " + mujocoReason
        );
    }
    std::string standReason;
    if (!validNumiHumanStand(
            model,
            articulation,
            input,
            config,
            standReason
        )) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "invalid Numi Human stand horizon: " + standReason
        );
    }
    return diagnostics;
}

NSString* bufferLabel(const std::size_t index) {
    switch (index) {
    case 0u:
        return @"articulated world";
    case 1u:
        return @"articulation descriptors";
    case 2u:
        return @"joint descriptors";
    case 3u:
        return @"DoF properties";
    case 4u:
        return @"body properties";
    case 5u:
        return @"articulated dispatch";
    case 6u:
        return @"articulated q";
    case 7u:
        return @"point impulses";
    case 8u:
        return @"body pose output";
    case 9u:
        return @"point world output";
    case 10u:
        return @"diagnostic mass output";
    case 11u:
        return @"point Jacobian output";
    case 12u:
        return @"generalized impulse output";
    case 13u:
        return @"delta velocity output";
    case 14u:
        return @"articulated status output";
    case 15u:
        return @"FunctionBased programs";
    case kMillardDispatchBuffer:
        return @"Millard dispatch";
    case kMillardMusclesBuffer:
        return @"Millard muscles";
    case kMillardStatesBuffer:
        return @"Millard states";
    case kMillardPathPointsBuffer:
        return @"Millard path points";
    case kMillardCurvesBuffer:
        return @"Millard source curves";
    case kMillardWrapsBuffer:
        return @"Millard cylinder wraps";
    case kMillardResultsBuffer:
        return @"Millard results";
    case kMillardForcesBuffer:
        return @"Millard generalized forces";
    case kMujocoDispatchBuffer:
        return @"MyoSim dispatch";
    case kMujocoMusclesBuffer:
        return @"MyoSim muscles";
    case kMujocoStatesBuffer:
        return @"MyoSim states";
    case kMujocoSitesBuffer:
        return @"MyoSim sites";
    case kMujocoWrapsBuffer:
        return @"MyoSim wraps";
    case kMujocoRoutesBuffer:
        return @"MyoSim route nodes";
    case kMujocoResultsBuffer:
        return @"MyoSim results";
    default:
        return @"articulated buffer";
    }
}

MetalArticulatedOperatorDiagnostics initializeContext(
    detail::MetalArticulatedOperatorContextState& context,
    MetalArticulatedOperatorDiagnostics diagnostics
) {
    if (context.initialized) {
        diagnostics.deviceName = nsString(context.device.name);
        return diagnostics;
    }

    std::string metallibPath = context.config.metallibPath;
    if (metallibPath.empty()) {
        metallibPath = defaultMetallibPath();
    }
    if (metallibPath.empty()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metallibUnavailable,
            "no articulated-operator metallib path is available"
        );
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnavailable,
            "no Metal-capable device is available"
        );
    }
    diagnostics.deviceName = nsString(device.name);
    if (!device.hasUnifiedMemory) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnsupported,
            "articulated operator requires unified-memory Metal"
        );
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnavailable,
            "failed to create a Metal command queue"
        );
    }
    queue.label = @"MetalRobo articulated operator queue";

    NSString* path = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    if (path == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metallibUnavailable,
            "metallib path is not valid UTF-8"
        );
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalLibraryFailure,
            "failed to load metallib: " +
                describeError(error)
        );
    }
    id<MTLFunction> function = [library
        newFunctionWithName:@"mr_articulated_operator"];
    if (function == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalLibraryFailure,
            "metallib does not contain the articulated operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                       error:&error];
    if (pipeline == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalPipelineFailure,
            "failed to create articulated pipeline: " +
                describeError(error)
        );
    }
    if (pipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup ||
        pipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalDeviceUnsupported,
            "device cannot execute the articulated threadgroup"
        );
    }
    id<MTLFunction> millardFunction = [library
        newFunctionWithName:@"mr_millard_reference"];
    if (millardFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the Millard reference operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> millardPipeline = [device
        newComputePipelineStateWithFunction:millardFunction
                                       error:&error];
    if (millardPipeline == nil ||
        millardPipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create Millard reference pipeline: " +
                describeError(error)
        );
    }
    id<MTLFunction> mujocoFunction = [library
        newFunctionWithName:@"mr_mujoco_muscle_reference"];
    if (mujocoFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the MyoSim reference operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> mujocoPipeline = [device
        newComputePipelineStateWithFunction:mujocoFunction
                                       error:&error];
    if (mujocoPipeline == nil ||
        mujocoPipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create MyoSim reference pipeline: " +
                describeError(error)
        );
    }
    id<MTLFunction> mujocoReduceFunction = [library
        newFunctionWithName:@"mr_mujoco_muscle_reduce"];
    if (mujocoReduceFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the MyoSim force-reduction operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> mujocoReducePipeline = [device
        newComputePipelineStateWithFunction:mujocoReduceFunction
                                       error:&error];
    if (mujocoReducePipeline == nil ||
        mujocoReducePipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create MyoSim force-reduction pipeline: " +
                describeError(error)
        );
    }
    id<MTLFunction> mujocoActiveForceFunction = [library
        newFunctionWithName:@"mr_mujoco_muscle_active_force_rows"];
    if (mujocoActiveForceFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the MyoSim active-force operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> mujocoActiveForcePipeline = [device
        newComputePipelineStateWithFunction:mujocoActiveForceFunction
                                       error:&error];
    if (mujocoActiveForcePipeline == nil ||
        mujocoActiveForcePipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create MyoSim active-force pipeline: " +
                describeError(error)
        );
    }
    id<MTLFunction> mujocoActivationFunction = [library
        newFunctionWithName:@"mr_mujoco_muscle_activation_step"];
    if (mujocoActivationFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the MyoSim activation-step operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> mujocoActivationPipeline = [device
        newComputePipelineStateWithFunction:mujocoActivationFunction
                                       error:&error];
    if (mujocoActivationPipeline == nil ||
        mujocoActivationPipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create MyoSim activation-step pipeline: " +
                describeError(error)
        );
    }
    id<MTLFunction> standFunction = [library
        newFunctionWithName:@"mr_numi_human_stand_step"];
    if (standFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the Numi Human stand operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> standPipeline = [device
        newComputePipelineStateWithFunction:standFunction
                                       error:&error];
    if (standPipeline == nil ||
        standPipeline.maxTotalThreadsPerThreadgroup <
            kStandThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create Numi Human stand pipeline: " +
                describeError(error)
        );
    }
    id<MTLFunction> tendonFunction = [library
        newFunctionWithName:@"mr_numi_human_tendon_transfer"];
    if (tendonFunction == nil) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
            "metallib does not contain the Numi Human tendon-transfer operator"
        );
    }
    error = nil;
    id<MTLComputePipelineState> tendonPipeline = [device
        newComputePipelineStateWithFunction:tendonFunction
                                       error:&error];
    if (tendonPipeline == nil ||
        tendonPipeline.maxTotalThreadsPerThreadgroup <
            kThreadsPerThreadgroup) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
            "failed to create Numi Human tendon-transfer pipeline: " +
                describeError(error)
        );
    }

    context.device = device;
    context.queue = queue;
    context.library = library;
    context.pipeline = pipeline;
    context.millardPipeline = millardPipeline;
    context.mujocoPipeline = mujocoPipeline;
    context.mujocoActiveForcePipeline = mujocoActiveForcePipeline;
    context.mujocoReducePipeline = mujocoReducePipeline;
    context.mujocoActivationPipeline = mujocoActivationPipeline;
    context.standPipeline = standPipeline;
    context.tendonPipeline = tendonPipeline;
    context.initialized = true;
    ++context.stats.pipelineCreationCount;
    return diagnostics;
}

MetalArticulatedOperatorDiagnostics initializeHumanMatterPipelines(
    detail::MetalArticulatedOperatorContextState& context,
    MetalArticulatedOperatorDiagnostics diagnostics
) {
    if (context.humanMatterBeginPipeline != nil &&
        context.humanMatterFactorPipeline != nil &&
        context.humanMatterConsumePipeline != nil &&
        context.humanMatterPreparePhysicalPipeline != nil &&
        context.humanMatterMarkPhysicalCompletePipeline != nil &&
        context.humanMatterProposePreparedPipeline != nil &&
        context.humanMatterValidateApplyPipeline != nil &&
        context.humanMatterCompleteApplyPipeline != nil &&
        context.humanMatterPrepareCandidatePipeline != nil &&
        context.humanMatterMaterializeCandidatePipeline != nil &&
        context.humanMatterTimelineEvent != nil) {
        return diagnostics;
    }
    struct PipelineSpec {
        NSString* name;
        __strong id<MTLComputePipelineState>* destination;
        NSUInteger minimumThreads;
    };
    const std::array<PipelineSpec, 10u> specs{{
        {@"mr_numanx_human_matter_begin",
         &context.humanMatterBeginPipeline, kThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_source_factor",
         &context.humanMatterFactorPipeline, kStandThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_consume_reaction",
         &context.humanMatterConsumePipeline, kStandThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_prepare_physical",
         &context.humanMatterPreparePhysicalPipeline,
         kStandThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_mark_physical_complete",
         &context.humanMatterMarkPhysicalCompletePipeline,
         kThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_propose_prepared",
         &context.humanMatterProposePreparedPipeline,
         kThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_validate_apply",
         &context.humanMatterValidateApplyPipeline,
         kThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_complete_apply",
         &context.humanMatterCompleteApplyPipeline,
         kStandThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_prepare_candidate",
         &context.humanMatterPrepareCandidatePipeline,
         kThreadsPerThreadgroup},
        {@"mr_numanx_human_matter_materialize_candidate",
         &context.humanMatterMaterializeCandidatePipeline,
         kStandThreadsPerThreadgroup},
    }};
    for (const PipelineSpec& spec : specs) {
        id<MTLFunction> function = [context.library
            newFunctionWithName:spec.name];
        if (function == nil) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalLibraryFailure,
                "metallib does not contain required NumanX Human/Matter owner kernels"
            );
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [context.device
            newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil ||
            pipeline.maxTotalThreadsPerThreadgroup < spec.minimumThreads) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalPipelineFailure,
                "failed to create NumanX Human/Matter owner pipeline: " +
                    describeError(error)
            );
        }
        *spec.destination = pipeline;
    }
    if (context.humanMatterTimelineEvent == nil) {
        context.humanMatterTimelineEvent = [context.device newSharedEvent];
        if (context.humanMatterTimelineEvent == nil) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalDeviceUnsupported,
                "failed to create NumanX Human/Matter cross-queue shared event"
            );
        }
        context.humanMatterTimelineEvent.label =
            @"NumanX Human/Matter physical prepare timeline";
    }
    return diagnostics;
}

std::size_t growthCapacity(
    const std::size_t current,
    const std::size_t required,
    const std::size_t maximum
) {
    if (current >= required) {
        return current;
    }
    if (current == 0u) {
        return required;
    }
    const std::size_t half = current / 2u;
    const std::size_t grown =
        half <= maximum - current
            ? current + half
            : maximum;
    return std::max(required, grown);
}

MetalArticulatedOperatorDiagnostics ensureBufferArena(
    detail::MetalArticulatedOperatorContextState& context,
    const RequiredBuffers& requirements,
    MetalArticulatedOperatorDiagnostics diagnostics
) {
    const std::size_t maximumBufferLength =
        static_cast<std::size_t>(
            context.device.maxBufferLength
        );
    std::array<std::size_t, kRawBufferCount> proposed =
        context.capacities;
    std::array<std::size_t, kStandBufferCount> standProposed =
        context.standCapacities;
    std::array<std::size_t, kHumanMatterBufferCount> humanMatterProposed =
        context.humanMatterCapacities;
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        const BufferRequirement& requirement =
            requirements.entries[index];
        if (requirement.allocationBytes >
            maximumBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    metalBufferFailure,
                std::string(requirement.label) +
                    " exceeds device.maxBufferLength"
            );
        }
        proposed[index] = growthCapacity(
            context.capacities[index],
            requirement.allocationBytes,
            maximumBufferLength
        );
    }
    for (std::size_t index = 0u;
         index < kStandBufferCount; ++index) {
        const BufferRequirement& requirement =
            requirements.standEntries[index];
        if (requirement.allocationBytes > maximumBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalBufferFailure,
                std::string(requirement.label) +
                    " exceeds device.maxBufferLength"
            );
        }
        standProposed[index] = growthCapacity(
            context.standCapacities[index],
            requirement.allocationBytes,
            maximumBufferLength
        );
    }
    for (std::size_t index = 0u;
         index < kHumanMatterBufferCount; ++index) {
        const BufferRequirement& requirement =
            requirements.humanMatterEntries[index];
        if (requirement.allocationBytes > maximumBufferLength) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalBufferFailure,
                std::string(requirement.label) +
                    " exceeds device.maxBufferLength"
            );
        }
        humanMatterProposed[index] = growthCapacity(
            context.humanMatterCapacities[index],
            requirement.allocationBytes,
            maximumBufferLength
        );
    }

    std::size_t projectedBytes = 0u;
    for (const std::size_t capacity : proposed) {
        if (!checkedAdd(
                projectedBytes,
                capacity,
                projectedBytes
            )) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    arithmeticOverflow,
                "persistent Metal arena byte-count overflow"
            );
        }
    }
    for (const std::size_t capacity : standProposed) {
        if (!checkedAdd(projectedBytes, capacity, projectedBytes)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "persistent Metal arena byte-count overflow"
            );
        }
    }
    for (const std::size_t capacity : humanMatterProposed) {
        if (!checkedAdd(projectedBytes, capacity, projectedBytes)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                "persistent Metal arena byte-count overflow"
            );
        }
    }
    const std::uint64_t recommendedWorkingSet =
        context.device.recommendedMaxWorkingSetSize;
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        // Geometric slack is optional. Retry at the smallest safe retained
        // capacities before rejecting a batch near the working-set budget.
        projectedBytes = 0u;
        for (std::size_t index = 0u;
             index < kRawBufferCount;
             ++index) {
            proposed[index] = std::max(
                context.capacities[index],
                requirements.entries[index].allocationBytes
            );
            if (!checkedAdd(
                    projectedBytes,
                    proposed[index],
                    projectedBytes
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        arithmeticOverflow,
                    "persistent Metal arena byte-count overflow"
                );
            }
        }
        for (std::size_t index = 0u;
             index < kStandBufferCount; ++index) {
            standProposed[index] = std::max(
                context.standCapacities[index],
                requirements.standEntries[index].allocationBytes
            );
            if (!checkedAdd(
                    projectedBytes,
                    standProposed[index],
                    projectedBytes
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                    "persistent Metal arena byte-count overflow"
                );
            }
        }
        for (std::size_t index = 0u;
             index < kHumanMatterBufferCount; ++index) {
            humanMatterProposed[index] = std::max(
                context.humanMatterCapacities[index],
                requirements.humanMatterEntries[index].allocationBytes
            );
            if (!checkedAdd(
                    projectedBytes,
                    humanMatterProposed[index],
                    projectedBytes
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                    "persistent Metal arena byte-count overflow"
                );
            }
        }
    }
    if (recommendedWorkingSet != 0u &&
        static_cast<std::uint64_t>(projectedBytes) >
            recommendedWorkingSet) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalBufferFailure,
            "persistent articulated buffer arena exceeds "
            "device.recommendedMaxWorkingSetSize"
        );
    }

    __strong id<MTLBuffer> replacements[kRawBufferCount] = {};
    __strong id<MTLBuffer> standReplacements[kStandBufferCount] = {};
    __strong id<MTLBuffer>
        humanMatterReplacements[kHumanMatterBufferCount] = {};
    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        if (proposed[index] == context.capacities[index]) {
            continue;
        }
        replacements[index] = [context.device
            newBufferWithLength:static_cast<NSUInteger>(
                proposed[index]
            )
                       options:MTLResourceStorageModeShared];
        if (replacements[index] == nil ||
            replacements[index].contents == nullptr ||
            replacements[index].length < proposed[index]) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    metalBufferFailure,
                std::string("persistent Metal buffer growth failed for ") +
                    requirements.entries[index].label
            );
        }
        replacements[index].label = bufferLabel(index);
    }
    for (std::size_t index = 0u;
         index < kStandBufferCount; ++index) {
        if (standProposed[index] == context.standCapacities[index]) {
            continue;
        }
        standReplacements[index] = [context.device
            newBufferWithLength:static_cast<NSUInteger>(standProposed[index])
                       options:MTLResourceStorageModeShared];
        if (standReplacements[index] == nil ||
            standReplacements[index].contents == nullptr ||
            standReplacements[index].length < standProposed[index]) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalBufferFailure,
                std::string("persistent Metal buffer growth failed for ") +
                    requirements.standEntries[index].label
            );
        }
        standReplacements[index].label = [NSString stringWithUTF8String:
            requirements.standEntries[index].label];
    }
    for (std::size_t index = 0u;
         index < kHumanMatterBufferCount; ++index) {
        if (humanMatterProposed[index] ==
            context.humanMatterCapacities[index]) {
            continue;
        }
        humanMatterReplacements[index] = [context.device
            newBufferWithLength:static_cast<NSUInteger>(
                humanMatterProposed[index])
                       options:MTLResourceStorageModeShared];
        if (humanMatterReplacements[index] == nil ||
            humanMatterReplacements[index].contents == nullptr ||
            humanMatterReplacements[index].length <
                humanMatterProposed[index]) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::metalBufferFailure,
                std::string("persistent Metal buffer growth failed for ") +
                    requirements.humanMatterEntries[index].label
            );
        }
        humanMatterReplacements[index].label =
            [NSString stringWithUTF8String:
                requirements.humanMatterEntries[index].label];
    }

    for (std::size_t index = 0u;
         index < kRawBufferCount;
         ++index) {
        if (replacements[index] == nil) {
            continue;
        }
        if (context.capacities[index] != 0u) {
            ++context.stats.bufferGrowthCount;
        }
        ++context.stats.bufferAllocationCount;
        context.buffers[index] = replacements[index];
        context.capacities[index] = proposed[index];
    }
    for (std::size_t index = 0u;
         index < kStandBufferCount; ++index) {
        if (standReplacements[index] == nil) {
            continue;
        }
        if (context.standCapacities[index] != 0u) {
            ++context.stats.bufferGrowthCount;
        }
        ++context.stats.bufferAllocationCount;
        context.standBuffers[index] = standReplacements[index];
        context.standCapacities[index] = standProposed[index];
    }
    for (std::size_t index = 0u;
         index < kHumanMatterBufferCount; ++index) {
        if (humanMatterReplacements[index] == nil) {
            continue;
        }
        if (context.humanMatterCapacities[index] != 0u) {
            ++context.stats.bufferGrowthCount;
        }
        ++context.stats.bufferAllocationCount;
        context.humanMatterBuffers[index] =
            humanMatterReplacements[index];
        context.humanMatterCapacities[index] =
            humanMatterProposed[index];
    }
    context.stats.retainedBufferBytes = projectedBytes;
    return diagnostics;
}

void copyToBuffer(
    id<MTLBuffer> destination,
    const void* source,
    const BufferRequirement& requirement
) {
    if (requirement.logicalBytes == 0u) {
        std::memset(
            destination.contents,
            0,
            requirement.allocationBytes
        );
        return;
    }
    std::memcpy(
        destination.contents,
        source,
        requirement.logicalBytes
    );
}

void uploadBatch(
    detail::MetalArticulatedOperatorContextState& context,
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorLayout& layout,
    const RequiredBuffers& requirements,
    const bool reusePublishedResidentState
) {
    MRJointDescriptorGPU emptyJoint{};
    MRArticulatedPointImpulseGPU emptyPoint{};
    const std::array<const void*, 8u> sources{
        &model.world,
        model.articulations.data(),
        model.joints.empty()
            ? static_cast<const void*>(&emptyJoint)
            : static_cast<const void*>(model.joints.data()),
        model.dofs.data(),
        model.bodies.data(),
        &layout.dispatch,
        input.q.data(),
        input.points.empty()
            ? static_cast<const void*>(&emptyPoint)
            : static_cast<const void*>(input.points.data()),
    };
    for (std::size_t index = 0u; index < sources.size(); ++index) {
        if (reusePublishedResidentState && index == 6u) {
            continue;
        }
        copyToBuffer(
            context.buffers[index],
            sources[index],
            requirements.entries[index]
        );
    }
    for (std::size_t index = sources.size();
         index < 15u;
         ++index) {
        std::memset(
            context.buffers[index].contents,
            0,
            requirements.entries[index].allocationBytes
        );
    }
    std::vector<MROpenSimSpatialTransformGPU> packedPrograms;
    const bool packed = packFunctionPrograms(model, packedPrograms);
    // validateAndBuildLayout() has already validated this exact immutable
    // stream. Retaining the defensive branch makes a future model mutation
    // between validation and upload fail closed rather than dispatch zeros.
    if (!packed) {
        std::memset(
            context.buffers[15u].contents,
            0,
            requirements.entries[15u].allocationBytes
        );
        return;
    }
    copyToBuffer(
        context.buffers[15u],
        packedPrograms.data(),
        requirements.entries[15u]
    );

    MRMillardReferenceDispatchGPU millardDispatch{};
    if (input.millard.enabled()) {
        const MRArticulationGPU& articulation =
            model.articulations[input.articulationIndex];
        millardDispatch.abiVersion = MR_MILLARD_REFERENCE_GPU_ABI_VERSION;
        millardDispatch.muscleCount = static_cast<mr_u32>(
            input.millard.muscles.size()
        );
        millardDispatch.pathPointCount = static_cast<mr_u32>(
            input.millard.pathPoints.size()
        );
        millardDispatch.wrapCount = static_cast<mr_u32>(
            input.millard.cylinderWraps.size()
        );
        millardDispatch.environmentCount = static_cast<mr_u32>(
            input.environmentCount
        );
        millardDispatch.dofCount = articulation.nv;
        millardDispatch.pointWorldStride = layout.dispatch.pointWorldStride;
        millardDispatch.pointJacobianStride =
            layout.dispatch.pointJacobianStride;
        millardDispatch.bodyPoseStride = layout.dispatch.bodyPoseStride;
        millardDispatch.articulationFirstBody = articulation.firstBody;
    }
    copyToBuffer(
        context.buffers[kMillardDispatchBuffer],
        &millardDispatch,
        requirements.entries[kMillardDispatchBuffer]
    );
    const auto uploadMillard = [&](const std::size_t index, const void* source) {
        copyToBuffer(
            context.buffers[index],
            source,
            requirements.entries[index]
        );
    };
    uploadMillard(
        kMillardMusclesBuffer,
        input.millard.muscles.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.muscles.data())
    );
    uploadMillard(
        kMillardStatesBuffer,
        input.millard.states.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.states.data())
    );
    uploadMillard(
        kMillardPathPointsBuffer,
        input.millard.pathPoints.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.pathPoints.data())
    );
    uploadMillard(
        kMillardCurvesBuffer,
        input.millard.curves.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.curves.data())
    );
    uploadMillard(
        kMillardWrapsBuffer,
        input.millard.cylinderWraps.empty()
            ? nullptr
            : static_cast<const void*>(input.millard.cylinderWraps.data())
    );
    std::memset(
        context.buffers[kMillardResultsBuffer].contents,
        0,
        requirements.entries[kMillardResultsBuffer].allocationBytes
    );
    std::memset(
        context.buffers[kMillardForcesBuffer].contents,
        0,
        requirements.entries[kMillardForcesBuffer].allocationBytes
    );

    MRMujocoMuscleReferenceDispatchGPU mujocoDispatch{};
    if (input.mujoco.enabled()) {
        const MRArticulationGPU& articulation =
            model.articulations[input.articulationIndex];
        mujocoDispatch.abiVersion = MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION;
        mujocoDispatch.muscleCount = static_cast<mr_u32>(
            input.mujoco.muscles.size()
        );
        mujocoDispatch.siteCount = static_cast<mr_u32>(
            input.mujoco.sites.size()
        );
        mujocoDispatch.wrapCount = static_cast<mr_u32>(
            input.mujoco.wraps.size()
        );
        mujocoDispatch.routeNodeCount = static_cast<mr_u32>(
            input.mujoco.routeNodes.size()
        );
        mujocoDispatch.environmentCount = static_cast<mr_u32>(
            input.environmentCount
        );
        mujocoDispatch.bodyPoseStride = layout.dispatch.bodyPoseStride;
        mujocoDispatch.articulationFirstBody = articulation.firstBody;
        mujocoDispatch.dofCount = articulation.nv;
        mujocoDispatch.pointJacobianStride =
            layout.dispatch.pointJacobianStride;
        mujocoDispatch.bodyJacobianPointOffset =
            input.mujoco.bodyJacobianPointOffset;
        mujocoDispatch.bodyJacobianPointStride = 4u;
        mujocoDispatch.timestepSecondsAndReserved = {
            context.config.mujocoActivationTimestepSeconds, 0.0f, 0.0f, 0.0f,
        };
    }
    copyToBuffer(
        context.buffers[kMujocoDispatchBuffer],
        &mujocoDispatch,
        requirements.entries[kMujocoDispatchBuffer]
    );
    const auto uploadMujoco = [&](const std::size_t index, const void* source) {
        copyToBuffer(
            context.buffers[index],
            source,
            requirements.entries[index]
        );
    };
    uploadMujoco(
        kMujocoMusclesBuffer,
        input.mujoco.muscles.empty()
            ? nullptr
            : static_cast<const void*>(input.mujoco.muscles.data())
    );
    if (!reusePublishedResidentState) {
        uploadMujoco(
            kMujocoStatesBuffer,
            input.mujoco.states.empty()
                ? nullptr
                : static_cast<const void*>(input.mujoco.states.data())
        );
    }
    uploadMujoco(
        kMujocoSitesBuffer,
        input.mujoco.sites.empty()
            ? nullptr
            : static_cast<const void*>(input.mujoco.sites.data())
    );
    uploadMujoco(
        kMujocoWrapsBuffer,
        input.mujoco.wraps.empty()
            ? nullptr
            : static_cast<const void*>(input.mujoco.wraps.data())
    );
    uploadMujoco(
        kMujocoRoutesBuffer,
        input.mujoco.routeNodes.empty()
            ? nullptr
            : static_cast<const void*>(input.mujoco.routeNodes.data())
    );
    std::memset(
        context.buffers[kMujocoResultsBuffer].contents,
        0,
        requirements.entries[kMujocoResultsBuffer].allocationBytes
    );
    if (reusePublishedResidentState) {
        // The prior accepted velocity is authoritative for both MyoSim and
        // the stand horizon. It remains in this context-owned sidecar.
    } else if (input.stand.enabled()) {
        copyToBuffer(
            context.standBuffers[kStandVelocityBuffer],
            input.stand.v.data(),
            requirements.standEntries[kStandVelocityBuffer]
        );
    } else if (!input.v.empty()) {
        copyToBuffer(
            context.standBuffers[kStandVelocityBuffer],
            input.v.data(),
            requirements.standEntries[kStandVelocityBuffer]
        );
    } else {
        std::memset(
            context.standBuffers[kStandVelocityBuffer].contents,
            0,
            requirements.standEntries[kStandVelocityBuffer].allocationBytes
        );
    }

    if (input.stand.enabled()) {
        copyToBuffer(
            context.standBuffers[kStandContactsBuffer],
            input.stand.contacts.empty()
                ? nullptr
                : static_cast<const void*>(input.stand.contacts.data()),
            requirements.standEntries[kStandContactsBuffer]
        );
        for (std::size_t index = kStandSpatialJacobianBuffer;
             index < kStandBufferCount; ++index) {
            std::memset(
                context.standBuffers[index].contents,
                0,
                requirements.standEntries[index].allocationBytes
            );
        }
        copyToBuffer(
            context.standBuffers[kStandTendonBindingsBuffer],
            input.stand.tendonBindings.empty()
                ? nullptr
                : static_cast<const void*>(
                      input.stand.tendonBindings.data()
                  ),
            requirements.standEntries[kStandTendonBindingsBuffer]
        );
        copyToBuffer(
            context.standBuffers[kStandTendonEnvelopesBuffer],
            input.stand.tendonEnvelopes.empty()
                ? nullptr
                : static_cast<const void*>(
                      input.stand.tendonEnvelopes.data()
                  ),
            requirements.standEntries[kStandTendonEnvelopesBuffer]
        );
        copyToBuffer(
            context.standBuffers[kStandJointEqualitiesBuffer],
            input.stand.jointEqualities.empty()
                ? nullptr
                : static_cast<const void*>(
                      input.stand.jointEqualities.data()
                  ),
            requirements.standEntries[kStandJointEqualitiesBuffer]
        );
    }
    for (std::size_t index = 0u;
         index < kHumanMatterBufferCount; ++index) {
        if (context.humanMatterBuffers[index] != nil) {
            std::memset(
                context.humanMatterBuffers[index].contents,
                0,
                requirements.humanMatterEntries[index].allocationBytes
            );
        }
    }
}

[[nodiscard]] bool validHumanMatterArena(
    const detail::MetalArticulatedOperatorContextState& context,
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorLayout& layout,
    std::string& reason
) noexcept {
    const MetalNumanXHumanMatterProgram& program =
        input.stand.numanXHumanMatterProgram;
    if (!program.valid()) return true;
    const auto byteCount = [](
        const std::uint64_t elements,
        const std::uint64_t elementBytes,
        std::uint64_t& bytes
    ) noexcept {
        if (elements != 0u &&
            elementBytes >
                std::numeric_limits<std::uint64_t>::max() / elements) {
            return false;
        }
        bytes = elements * elementBytes;
        return bytes != 0u;
    };
    std::uint64_t reactionBytes = 0u;
    std::uint64_t statusBytes = 0u;
    std::uint64_t matterOutcomeBytes = 0u;
    if (program.transactionSlot >=
            MR_NUMANX_COUPLED_HUMAN_MAX_TRANSACTION_SLOTS ||
        !byteCount(
            program.matterGeneralizedReactionElementCount,
            sizeof(float), reactionBytes) ||
        !byteCount(
            program.jointStatusElementCount,
            sizeof(MRNumanXCoupledHumanStatusGPU), statusBytes) ||
        !byteCount(
            program.matterApplyOutcomeElementCount,
            sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU),
            matterOutcomeBytes) ||
        !exactMetalBuffer(
            context.device,
            program.matterGeneralizedReaction,
            program.matterGeneralizedReactionGPUAddress,
            reactionBytes) ||
        !exactMetalBuffer(
            context.device,
            program.jointStatuses,
            program.jointStatusesGPUAddress,
            statusBytes) ||
        !exactMetalBuffer(
            context.device,
            program.acceptedPhysicsStateTokens,
            program.acceptedPhysicsStateTokensGPUAddress,
            program.acceptedPhysicsStateTokenByteCount) ||
        !exactMetalBuffer(
            context.device,
            program.matterApplyOutcomes,
            program.matterApplyOutcomesGPUAddress,
            matterOutcomeBytes)) {
        reason = "NumanX Human/Matter adapter arenas are not exact same-device Metal buffers";
        return false;
    }

    const std::array<id<MTLBuffer>, 20u> ownerBuffers{{
        context.buffers[6u],
        context.standBuffers[kStandVelocityBuffer],
        context.buffers[kMujocoStatesBuffer],
        context.buffers[kMillardForcesBuffer],
        context.buffers[7u],
        context.buffers[8u],
        context.buffers[9u],
        context.buffers[11u],
        context.standBuffers[kStandFactorBuffer],
        context.standBuffers[kStandStatusBuffer],
        context.humanMatterBuffers[kHumanMatterQCheckpointBuffer],
        context.humanMatterBuffers[kHumanMatterVCheckpointBuffer],
        context.humanMatterBuffers[kHumanMatterMujocoCheckpointBuffer],
        context.humanMatterBuffers[kHumanMatterOwnerStatusBuffer],
        context.humanMatterBuffers[kHumanMatterAppliedOutcomeBuffer],
        context.humanMatterBuffers[kHumanMatterFinalAcceptedTokenBuffer],
        context.humanMatterBuffers[kHumanMatterProposalBuffer],
        context.humanMatterBuffers[kHumanMatterProposedTokenBuffer],
        context.humanMatterBuffers[kHumanMatterApplyActionBuffer],
        context.humanMatterBuffers[kHumanMatterPublicationFenceBuffer],
    }};
    const std::array<std::uint64_t, 20u> ownerMinimumBytes{{
        layout.qBytes,
        layout.standVelocityBytes,
        layout.mujocoStateBytes,
        layout.millardGeneralizedForceBytes,
        layout.pointBytes,
        layout.bodyPoseBytes,
        layout.pointWorldBytes,
        layout.pointJacobianBytes,
        layout.humanMatterSourceFactorBytes,
        layout.standStatusBytes,
        layout.humanMatterQCheckpointBytes,
        layout.humanMatterVCheckpointBytes,
        layout.humanMatterMujocoCheckpointBytes,
        layout.humanMatterOwnerStatusBytes,
        layout.humanMatterAppliedOutcomeBytes,
        layout.humanMatterFinalAcceptedTokenBytes,
        layout.humanMatterProposalBytes,
        layout.humanMatterProposedTokenBytes,
        layout.humanMatterApplyActionBytes,
        layout.humanMatterPublicationFenceBytes,
    }};
    for (std::size_t index = 0u; index < ownerBuffers.size(); ++index) {
        if (!ownedMetalBuffer(
                context.device,
                ownerBuffers[index],
                ownerMinimumBytes[index])) {
            reason = "NumanX Human/Matter owner arena is not an exact same-device Metal buffer";
            return false;
        }
    }
    if ((program.capabilities &
         MetalNumanXHumanMatterExactCandidateKinematics) != 0u) {
        const std::array<id<MTLBuffer>, 6u> candidateBuffers{{
            context.humanMatterBuffers[kHumanMatterCandidatePointBuffer],
            context.humanMatterBuffers[
                kHumanMatterCandidateBodyPoseBuffer],
            context.humanMatterBuffers[
                kHumanMatterCandidatePointWorldBuffer],
            context.humanMatterBuffers[
                kHumanMatterCandidatePointJacobianBuffer],
            context.humanMatterBuffers[
                kHumanMatterCandidateOperatorScratchBuffer],
            context.humanMatterBuffers[
                kHumanMatterCandidateOperatorStatusBuffer],
        }};
        const std::array<std::uint64_t, 6u> candidateMinimumBytes{{
            layout.humanMatterCandidatePointBytes,
            layout.humanMatterCandidateBodyPoseBytes,
            layout.humanMatterCandidatePointWorldBytes,
            layout.humanMatterCandidatePointJacobianBytes,
            layout.humanMatterCandidateOperatorScratchBytes,
            layout.humanMatterCandidateOperatorStatusBytes,
        }};
        for (std::size_t index = 0u;
             index < candidateBuffers.size(); ++index) {
            if (!ownedMetalBuffer(
                    context.device,
                    candidateBuffers[index],
                    candidateMinimumBytes[index])) {
                reason = "NumanX Human/Matter exact-candidate private arena is not an exact same-device Metal buffer";
                return false;
            }
        }
    }
    const std::array<void*, 4u> adapterBuffers{{
        program.matterGeneralizedReaction,
        program.jointStatuses,
        program.acceptedPhysicsStateTokens,
        program.matterApplyOutcomes,
    }};
    for (std::size_t left = 0u; left < adapterBuffers.size(); ++left) {
        for (std::size_t right = left + 1u;
             right < adapterBuffers.size(); ++right) {
            if (adapterBuffers[left] == adapterBuffers[right]) {
                reason = "NumanX Human/Matter adapter arenas alias each other";
                return false;
            }
        }
        for (id<MTLBuffer> owner : ownerBuffers) {
            if (adapterBuffers[left] == (__bridge void*)owner) {
                reason = "NumanX Human/Matter adapter arena aliases owner state or checkpoint storage";
                return false;
            }
        }
    }
    const auto rangesAlias = [](
        id<MTLBuffer> left,
        id<MTLBuffer> right
    ) noexcept {
        if (left == nil || right == nil) return false;
        const std::uint64_t leftBegin = left.gpuAddress;
        const std::uint64_t rightBegin = right.gpuAddress;
        const std::uint64_t leftBytes = left.length;
        const std::uint64_t rightBytes = right.length;
        if (leftBegin == 0u || rightBegin == 0u ||
            leftBegin >
                std::numeric_limits<std::uint64_t>::max() - leftBytes ||
            rightBegin >
                std::numeric_limits<std::uint64_t>::max() - rightBytes) {
            return true;
        }
        return leftBegin < rightBegin + rightBytes &&
            rightBegin < leftBegin + leftBytes;
    };
    for (std::size_t left = 0u; left < adapterBuffers.size(); ++left) {
        for (std::size_t right = left + 1u;
             right < adapterBuffers.size(); ++right) {
            if (rangesAlias(
                    (__bridge id<MTLBuffer>)adapterBuffers[left],
                    (__bridge id<MTLBuffer>)adapterBuffers[right])) {
                reason = "NumanX Human/Matter adapter buffer ranges alias each other";
                return false;
            }
        }
    }
    for (void* rawAdapter : adapterBuffers) {
        __unsafe_unretained id<MTLBuffer> adapter =
            (__bridge id<MTLBuffer>)rawAdapter;
        for (id<MTLBuffer> owner : context.buffers) {
            if (rangesAlias(adapter, owner)) {
                reason = "NumanX Human/Matter adapter arena aliases an operator-owned buffer range";
                return false;
            }
        }
        for (id<MTLBuffer> owner : context.standBuffers) {
            if (rangesAlias(adapter, owner)) {
                reason = "NumanX Human/Matter adapter arena aliases a stand-owned buffer range";
                return false;
            }
        }
        for (id<MTLBuffer> owner : context.humanMatterBuffers) {
            if (rangesAlias(adapter, owner)) {
                reason = "NumanX Human/Matter adapter arena aliases a private owner-transaction buffer range";
                return false;
            }
        }
    }
    const std::array<std::uint64_t, 13u> compactValues{{
        input.environmentCount,
        layout.dispatch.qStride,
        static_cast<std::uint64_t>(
            context.buffers[kMujocoStatesBuffer].length /
            sizeof(MRMujocoMuscleStateGPU)),
        layout.dispatch.bodyPoseStride,
        layout.dispatch.pointStride,
        layout.dispatch.pointWorldStride,
        layout.dispatch.pointJacobianStride,
        layout.standFactorElements,
        layout.mujocoMuscleGeneralizedForceElements,
        layout.mujocoGeneralizedForceElements,
        program.reactionStride,
        program.jointStatusStride,
        program.candidatePointCapacity,
    }};
    if (std::any_of(
            compactValues.begin(), compactValues.end(),
            [](const std::uint64_t value) {
                return value > std::numeric_limits<std::uint32_t>::max();
            })) {
        reason = "NumanX Human/Matter count or stride exceeds the fixed-width device ABI";
        return false;
    }
    return true;
}

struct NumanXHumanMatterCandidateEncoderContext {
    detail::MetalArticulatedOperatorContextState* state = nullptr;
    void* commandBuffer = nullptr;
    void* matterReaction = nullptr;
    void* jointStatuses = nullptr;
    void* acceptedTokens = nullptr;
    std::uint64_t matterReactionGPUAddress = 0u;
    std::uint64_t jointStatusesGPUAddress = 0u;
    std::uint64_t acceptedTokensGPUAddress = 0u;
    std::uint64_t matterReactionBytes = 0u;
    std::uint64_t jointStatusBytes = 0u;
    std::uint64_t acceptedTokenBytes = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t articulationFirstBody = 0u;
    std::uint32_t sourcePointStride = 0u;
    std::uint32_t sourceBodyProbeOffset = 0u;
    std::uint32_t combinedPointStride = 0u;
    std::uint32_t candidatePointCapacity = 0u;
    std::uint32_t substepIndex = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint32_t physicsSubstepCount = 0u;
    float timestepSeconds = 0.0f;
    std::uint32_t controlStep = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    std::uint64_t slotGeneration = 0u;
};

struct MetalBufferRegion {
    std::uint64_t begin = 0u;
    std::uint64_t end = 0u;
};

[[nodiscard]] bool logicalBufferBytes(
    const std::uint64_t environmentCount,
    const std::uint64_t stride,
    const std::uint64_t width,
    const std::uint64_t elementBytes,
    std::uint64_t& bytes
) noexcept {
    if (environmentCount == 0u || width == 0u || stride < width ||
        elementBytes == 0u) return false;
    const std::uint64_t environmentOffset = environmentCount - 1u;
    if (stride != 0u && environmentOffset >
            (std::numeric_limits<std::uint64_t>::max() - width) / stride) {
        return false;
    }
    const std::uint64_t elements = environmentOffset * stride + width;
    if (elements >
        std::numeric_limits<std::uint64_t>::max() / elementBytes) {
        return false;
    }
    bytes = elements * elementBytes;
    return bytes != 0u;
}

[[nodiscard]] bool exactBufferRegion(
    id<MTLDevice> device,
    void* raw,
    const std::uint64_t gpuAddress,
    const std::uint64_t bytes,
    MetalBufferRegion& region
) noexcept {
    if (!exactMetalBuffer(device, raw, gpuAddress, bytes) ||
        gpuAddress > std::numeric_limits<std::uint64_t>::max() - bytes) {
        return false;
    }
    region = {gpuAddress, gpuAddress + bytes};
    return true;
}

[[nodiscard]] bool ownedBufferRegion(
    id<MTLBuffer> buffer,
    MetalBufferRegion& region
) noexcept {
    if (buffer == nil || buffer.gpuAddress == 0u || buffer.length == 0u) {
        return false;
    }
    const std::uint64_t begin = buffer.gpuAddress;
    const std::uint64_t bytes = buffer.length;
    if (begin > std::numeric_limits<std::uint64_t>::max() - bytes) {
        return false;
    }
    region = {begin, begin + bytes};
    return true;
}

[[nodiscard]] bool regionsOverlap(
    const MetalBufferRegion& left,
    const MetalBufferRegion& right
) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] std::uint64_t humanMatterRecordFingerprint(
    const void* record
) noexcept {
    constexpr std::uint64_t offset = 14695981039346656037ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    if (record == nullptr) return 0u;
    std::uint64_t hash = offset;
    const auto* bytes = static_cast<const std::uint8_t*>(record);
    for (std::size_t index = 0u; index < 120u; ++index) {
        hash = (hash ^ bytes[index]) * prime;
    }
    return hash == 0u ? offset : hash;
}

[[nodiscard]] std::uint64_t humanIOPublicationBindingFingerprint(
    const MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept {
    return metalNumanXHumanIOPublicationBindingFingerprint(binding);
}

[[nodiscard]] std::uint64_t acceptedPhysicsTokenFingerprint(
    const void* raw
) noexcept {
    constexpr std::uint64_t offset = 14695981039346656037ull;
    if (raw == nullptr) return 0u;
    const auto mix = [](std::uint64_t hash, const std::uint64_t value,
                        const std::size_t bytes) noexcept {
        constexpr std::uint64_t localPrime = 1099511628211ull;
        for (std::size_t index = 0u; index < bytes; ++index) {
            hash = (hash ^ static_cast<std::uint8_t>(value >> (8u * index))) *
                localPrime;
        }
        return hash;
    };
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    std::uint64_t words[8]{};
    std::memcpy(words, bytes, sizeof(words));
    std::uint32_t words32[16]{};
    std::memcpy(words32, bytes, sizeof(words32));
    std::uint64_t hash = offset;
    hash = mix(hash, 1u, sizeof(std::uint32_t));
    for (std::size_t index = 0u; index < 5u; ++index) {
        hash = mix(hash, words[index], sizeof(std::uint64_t));
    }
    hash = mix(hash, words32[10], sizeof(std::uint32_t));
    hash = mix(hash, words32[11], sizeof(std::uint32_t));
    hash = mix(hash, words[6], sizeof(std::uint64_t));
    return hash;
}

[[nodiscard]] bool zeroAcceptedPhysicsToken(const void* raw) noexcept {
    if (raw == nullptr) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    return std::all_of(
        bytes,
        bytes + MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES,
        [](const std::uint8_t value) noexcept { return value == 0u; });
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

[[nodiscard]] bool validAcceptedPhysicsToken(
    const void* raw,
    const std::uint64_t transactionFingerprint,
    const std::uint64_t expectedFingerprint
) noexcept {
    if (raw == nullptr || transactionFingerprint == 0u ||
        expectedFingerprint == 0u) return false;
    std::uint64_t words[8]{};
    std::memcpy(words, raw, sizeof(words));
    return words[0] == transactionFingerprint && words[6] == 0u &&
        words[7] == expectedFingerprint &&
        acceptedPhysicsTokenFingerprint(raw) == expectedFingerprint;
}

[[nodiscard]] bool importableSharedEvent(
    id<MTLDevice> device,
    id<MTLSharedEvent> event
) noexcept {
    if (device == nil || event == nil) return false;
    @autoreleasepool {
        MTLSharedEventHandle* handle = [event newSharedEventHandle];
        if (handle == nil) return false;
        id<MTLSharedEvent> imported =
            [device newSharedEventWithHandle:handle];
        return imported != nil;
    }
}

[[nodiscard]] bool validProposalRecord(
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterDispatchGPU& dispatch,
    const void* proposedToken,
    const MetalNumanXHumanIOCandidatePublicationProgram& humanIOCandidate
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
        proposal.programFingerprint == dispatch.programFingerprint &&
        proposal.transactionFingerprint == dispatch.transactionFingerprint &&
        proposal.linearizationEpoch == dispatch.linearizationEpoch &&
        proposal.slotGeneration == dispatch.slotGeneration &&
        proposal.environment == 0u &&
        proposal.stepIndex == dispatch.stepIndex &&
        proposal.substepIndex == dispatch.substepIndex &&
        proposal.transactionSlot == dispatch.transactionSlot &&
        proposal.physicsSubstepCount == dispatch.physicsSubstepCount &&
        proposal.controlStep == dispatch.controlStep &&
        humanIOCandidate.valid() &&
        humanIOCandidate.transactionFingerprint ==
            dispatch.transactionFingerprint &&
        proposal.candidatePublicationFingerprint ==
            humanIOCandidate.candidatePublicationFingerprint &&
        proposal.humanIOIdentityFingerprint ==
            humanIOCandidate.identityFingerprint &&
        proposal.proposalFingerprint != 0u &&
        proposal.proposalFingerprint == humanMatterRecordFingerprint(&proposal) &&
        (accept
             ? validAcceptedPhysicsToken(
                   proposedToken,
                   dispatch.transactionFingerprint,
                   proposal.physicsTokenFingerprint)
             : zeroAcceptedPhysicsToken(proposedToken));
}

[[nodiscard]] bool validAppliedRecord(
    const MRNumanXHumanMatterAppliedOutcomeGPU& applied,
    const MRNumanXHumanMatterProposalGPU& proposal,
    const MRNumanXHumanMatterDispatchGPU& dispatch,
    const void* proposedToken,
    const void* finalToken,
    const MetalNumanXHumanIOCandidatePublicationProgram& humanIOCandidate
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
    return validProposalRecord(
            proposal, dispatch, proposedToken, humanIOCandidate) &&
        applied.abiVersion == MR_NUMANX_HUMAN_MATTER_ABI_VERSION &&
        (accept || reject) &&
        applied.programFingerprint == dispatch.programFingerprint &&
        applied.transactionFingerprint == dispatch.transactionFingerprint &&
        applied.linearizationEpoch == dispatch.linearizationEpoch &&
        applied.slotGeneration == dispatch.slotGeneration &&
        applied.proposalFingerprint == proposal.proposalFingerprint &&
        closureBound &&
        applied.matterApplyFingerprint != 0u &&
        applied.environment == 0u && applied.stepIndex == dispatch.stepIndex &&
        applied.substepIndex == dispatch.substepIndex &&
        applied.transactionSlot == dispatch.transactionSlot &&
        applied.physicsSubstepCount == dispatch.physicsSubstepCount &&
        applied.controlStep == dispatch.controlStep &&
        applied.appliedFingerprint != 0u &&
        applied.appliedFingerprint == humanMatterRecordFingerprint(&applied) &&
        (accept
             ? (applied.physicsTokenFingerprint ==
                    proposal.physicsTokenFingerprint &&
                validAcceptedPhysicsToken(
                    finalToken,
                    dispatch.transactionFingerprint,
                    applied.physicsTokenFingerprint) &&
                std::memcmp(
                    proposedToken,
                    finalToken,
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES) == 0)
             : zeroAcceptedPhysicsToken(finalToken));
}

[[nodiscard]] bool matchingHumanMatterCandidatePass(
    const NumanXHumanMatterCandidateEncoderContext& context,
    const MetalNumanXHumanMatterPass& pass
) noexcept {
    if (context.state == nullptr ||
        pass.abiVersion != kMetalNumanXHumanMatterABIVersion ||
        pass.structSize != sizeof(MetalNumanXHumanMatterPass) ||
        pass.phase != MetalNumanXHumanMatterPhase::preDynamics ||
        pass.stepIndex != 0u || pass.stepCount != 1u ||
        pass.substepIndex != context.substepIndex ||
        pass.transactionSlot != context.transactionSlot ||
        pass.physicsSubstepCount != context.physicsSubstepCount ||
        pass.controlStep != context.controlStep ||
        pass.commandBuffer != context.commandBuffer ||
        pass.environmentCount != context.environmentCount ||
        pass.qCoordinateCount != context.nq ||
        pass.dofCount != context.nv ||
        pass.bodyCount != context.bodyCount ||
        pass.qStride != context.nq || pass.vStride != context.nv ||
        pass.articulationIndex != context.articulationIndex ||
        pass.articulationFirstBody != context.articulationFirstBody ||
        pass.pointStride != context.sourcePointStride ||
        pass.bodyJacobianPointOffset !=
            context.sourceBodyProbeOffset ||
        pass.timestepSeconds != context.timestepSeconds ||
        pass.programFingerprint != context.programFingerprint ||
        pass.transactionFingerprint != context.transactionFingerprint ||
        pass.linearizationEpoch != context.linearizationEpoch ||
        pass.slotGeneration != context.slotGeneration ||
        (pass.capabilities &
         MetalNumanXHumanMatterExactCandidateKinematics) == 0u ||
        (pass.accessFlags &
         MetalNumanXHumanMatterMayEncodeExactCandidate) == 0u) {
        return false;
    }
    const detail::MetalArticulatedOperatorContextState& state =
        *context.state;
    return pass.q == (__bridge void*)state.buffers[6u] &&
        pass.v == (__bridge void*)state.standBuffers[
            kStandVelocityBuffer] &&
        pass.pointQueries == (__bridge void*)state.buffers[7u] &&
        pass.qCheckpoint == (__bridge void*)state.humanMatterBuffers[
            kHumanMatterQCheckpointBuffer] &&
        pass.vCheckpoint == (__bridge void*)state.humanMatterBuffers[
            kHumanMatterVCheckpointBuffer] &&
        pass.sourceEffectiveTangentFactor ==
            (__bridge void*)state.standBuffers[kStandFactorBuffer] &&
        pass.matterGeneralizedReaction == context.matterReaction &&
        pass.jointStatuses == context.jointStatuses &&
        pass.acceptedPhysicsStateTokens == context.acceptedTokens &&
        pass.qGPUAddress == state.buffers[6u].gpuAddress &&
        pass.vGPUAddress == state.standBuffers[
            kStandVelocityBuffer].gpuAddress &&
        pass.pointQueriesGPUAddress == state.buffers[7u].gpuAddress &&
        pass.qCheckpointGPUAddress == state.humanMatterBuffers[
            kHumanMatterQCheckpointBuffer].gpuAddress &&
        pass.vCheckpointGPUAddress == state.humanMatterBuffers[
            kHumanMatterVCheckpointBuffer].gpuAddress &&
        pass.sourceEffectiveTangentFactorGPUAddress ==
            state.standBuffers[kStandFactorBuffer].gpuAddress &&
        pass.matterGeneralizedReactionGPUAddress ==
            context.matterReactionGPUAddress &&
        pass.jointStatusesGPUAddress == context.jointStatusesGPUAddress &&
        pass.acceptedPhysicsStateTokensGPUAddress ==
            context.acceptedTokensGPUAddress;
}

[[nodiscard]] bool encodeExactHumanMatterCandidate(
    void* rawContext,
    const MetalNumanXHumanMatterPass& pass,
    const MetalNumanXHumanMatterCandidateQuery& query
) noexcept {
    if (rawContext == nullptr) return false;
    auto& context = *static_cast<
        NumanXHumanMatterCandidateEncoderContext*>(rawContext);
    if (!matchingHumanMatterCandidatePass(context, pass) ||
        pass.exactCandidateContext != rawContext ||
        pass.encodeExactCandidate != &encodeExactHumanMatterCandidate ||
        query.abiVersion != kMetalNumanXHumanMatterABIVersion ||
        query.structSize != sizeof(MetalNumanXHumanMatterCandidateQuery) ||
        query.reserved0 != 0u ||
        (query.accessFlags &
         ~kMetalNumanXHumanMatterCandidateKnownAccess) != 0u ||
        query.programFingerprint != context.programFingerprint ||
        query.transactionFingerprint != context.transactionFingerprint ||
        query.linearizationEpoch != context.linearizationEpoch ||
        query.slotGeneration != context.slotGeneration ||
        query.substepIndex != context.substepIndex ||
        query.transactionSlot != context.transactionSlot ||
        query.physicsSubstepCount != context.physicsSubstepCount ||
        query.controlStep != context.controlStep ||
        query.pointCount > context.candidatePointCapacity ||
        query.deltaVelocityStride < context.nv ||
        query.deltaVelocityStride > MR_NUMANX_COUPLED_HUMAN_MAX_DOFS ||
        query.candidateQStride < context.nq ||
        query.candidateQStride > MR_NUMANX_COUPLED_HUMAN_MAX_Q ||
        query.candidateBodyStride <
            context.articulationFirstBody + context.bodyCount) {
        return false;
    }

    std::uint32_t requiredAccess =
        MetalNumanXHumanMatterCandidateReadDeltaVelocity |
        MetalNumanXHumanMatterCandidateWriteQ |
        MetalNumanXHumanMatterCandidateWriteBodies;
    if (query.pointCount != 0u) {
        requiredAccess |=
            MetalNumanXHumanMatterCandidateReadPointQueries |
            MetalNumanXHumanMatterCandidateWritePointJacobians;
    }
    if (query.pointWorld != nullptr) {
        requiredAccess |= MetalNumanXHumanMatterCandidateWritePointWorld;
    }
    if (query.accessFlags != requiredAccess ||
        query.deltaVelocity == nullptr || query.candidateQ == nullptr ||
        query.candidateBodies == nullptr) {
        return false;
    }
    if (query.pointCount == 0u) {
        if (query.pointQueries != nullptr || query.pointWorld != nullptr ||
            query.pointJacobians != nullptr ||
            query.pointQueriesGPUAddress != 0u ||
            query.pointWorldGPUAddress != 0u ||
            query.pointJacobiansGPUAddress != 0u ||
            query.pointStride != 0u || query.pointWorldStride != 0u ||
            query.pointJacobianStride != 0u) {
            return false;
        }
    } else if (query.pointQueries == nullptr ||
               query.pointJacobians == nullptr ||
               query.pointStride < query.pointCount ||
               query.pointJacobianStride <
                   static_cast<std::uint64_t>(query.pointCount) * 3u *
                       query.deltaVelocityStride ||
               (query.pointWorld == nullptr
                    ? (query.pointWorldGPUAddress != 0u ||
                       query.pointWorldStride != 0u)
                    : (query.pointWorldGPUAddress == 0u ||
                       query.pointWorldStride < query.pointCount))) {
        return false;
    }

    std::array<MetalBufferRegion, 6u> queryRegions{};
    std::size_t queryRegionCount = 0u;
    const auto addQueryRegion = [&] (
        void* raw,
        const std::uint64_t address,
        const std::uint64_t environments,
        const std::uint64_t stride,
        const std::uint64_t width,
        const std::uint64_t elementBytes
    ) noexcept {
        std::uint64_t bytes = 0u;
        if (!logicalBufferBytes(
                environments, stride, width, elementBytes, bytes) ||
            queryRegionCount >= queryRegions.size() ||
            !exactBufferRegion(
                context.state->device, raw, address, bytes,
                queryRegions[queryRegionCount])) {
            return false;
        }
        ++queryRegionCount;
        return true;
    };
    if (!addQueryRegion(
            query.deltaVelocity, query.deltaVelocityGPUAddress,
            context.environmentCount, query.deltaVelocityStride,
            context.nv, sizeof(float)) ||
        !addQueryRegion(
            query.candidateQ, query.candidateQGPUAddress,
            context.environmentCount, query.candidateQStride,
            context.nq, sizeof(float)) ||
        !addQueryRegion(
            query.candidateBodies, query.candidateBodiesGPUAddress,
            context.environmentCount, query.candidateBodyStride,
            context.articulationFirstBody + context.bodyCount,
            sizeof(MRBodyStateGPU))) {
        return false;
    }
    if (query.pointCount != 0u) {
        if (!addQueryRegion(
                query.pointQueries, query.pointQueriesGPUAddress,
                context.environmentCount, query.pointStride,
                query.pointCount, sizeof(MRArticulatedPointImpulseGPU)) ||
            !addQueryRegion(
                query.pointJacobians, query.pointJacobiansGPUAddress,
                context.environmentCount, query.pointJacobianStride,
                query.pointJacobianStride,
                sizeof(float))) {
            return false;
        }
        if (query.pointWorld != nullptr &&
            !addQueryRegion(
                query.pointWorld, query.pointWorldGPUAddress,
                context.environmentCount, query.pointWorldStride,
                query.pointCount,
                sizeof(MRArticulatedPointWorldGPU))) {
            return false;
        }
    }
    for (std::size_t left = 0u; left < queryRegionCount; ++left) {
        for (std::size_t right = left + 1u;
             right < queryRegionCount; ++right) {
            if (regionsOverlap(queryRegions[left], queryRegions[right])) {
                return false;
            }
        }
    }

    std::array<MetalBufferRegion,
        kRawBufferCount + kStandBufferCount + kHumanMatterBufferCount + 3u>
        protectedRegions{};
    std::size_t protectedCount = 0u;
    const auto addProtected = [&] (id<MTLBuffer> buffer) noexcept {
        if (buffer == nil) return true;
        return protectedCount < protectedRegions.size() &&
            ownedBufferRegion(buffer, protectedRegions[protectedCount++]);
    };
    for (id<MTLBuffer> buffer : context.state->buffers) {
        if (!addProtected(buffer)) return false;
    }
    for (id<MTLBuffer> buffer : context.state->standBuffers) {
        if (!addProtected(buffer)) return false;
    }
    for (id<MTLBuffer> buffer : context.state->humanMatterBuffers) {
        if (!addProtected(buffer)) return false;
    }
    const auto addExternalProtected = [&] (
        void* raw,
        const std::uint64_t address,
        const std::uint64_t bytes
    ) noexcept {
        if (protectedCount >= protectedRegions.size()) return false;
        return exactBufferRegion(
            context.state->device, raw, address, bytes,
            protectedRegions[protectedCount++]);
    };
    if (!addExternalProtected(
            context.matterReaction, context.matterReactionGPUAddress,
            context.matterReactionBytes) ||
        !addExternalProtected(
            context.jointStatuses, context.jointStatusesGPUAddress,
            context.jointStatusBytes) ||
        !addExternalProtected(
            context.acceptedTokens, context.acceptedTokensGPUAddress,
            context.acceptedTokenBytes)) {
        return false;
    }
    for (std::size_t queryIndex = 0u;
         queryIndex < queryRegionCount; ++queryIndex) {
        for (std::size_t protectedIndex = 0u;
             protectedIndex < protectedCount; ++protectedIndex) {
            if (regionsOverlap(
                    queryRegions[queryIndex],
                    protectedRegions[protectedIndex])) {
                return false;
            }
        }
    }

    const std::uint64_t combinedCount64 =
        static_cast<std::uint64_t>(context.bodyCount) * 4u +
        query.pointCount;
    const std::uint64_t privateJacobianStride64 =
        static_cast<std::uint64_t>(context.combinedPointStride) * 3u *
        context.nv;
    if (combinedCount64 > std::numeric_limits<std::uint32_t>::max() ||
        privateJacobianStride64 >
            std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }

    MRNumanXHumanMatterCandidateDispatchGPU candidate{};
    candidate.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
    candidate.flags = query.pointWorld != nullptr
        ? MR_NUMANX_HUMAN_MATTER_CANDIDATE_HAS_POINT_WORLD
        : 0u;
    candidate.environmentCount = context.environmentCount;
    candidate.articulationIndex = context.articulationIndex;
    candidate.nq = context.nq;
    candidate.nv = context.nv;
    candidate.bodyCount = context.bodyCount;
    candidate.articulationFirstBody = context.articulationFirstBody;
    candidate.sourcePointStride = context.sourcePointStride;
    candidate.sourceBodyProbeOffset = context.sourceBodyProbeOffset;
    candidate.bodyProbeCount = 4u * context.bodyCount;
    candidate.candidatePointCount = query.pointCount;
    candidate.combinedPointStride = context.combinedPointStride;
    candidate.candidateQStride = query.candidateQStride;
    candidate.candidateBodyStride = query.candidateBodyStride;
    candidate.candidatePointStride = query.pointStride;
    candidate.candidatePointWorldStride = query.pointWorldStride;
    candidate.candidatePointJacobianStride =
        query.pointJacobianStride;
    candidate.privateBodyPoseStride = context.bodyCount;
    candidate.privatePointWorldStride = context.combinedPointStride;
    candidate.privatePointJacobianStride = static_cast<mr_u32>(
        privateJacobianStride64);
    candidate.privateOperatorScratchStride = context.nv;
    candidate.deltaVelocityStride = query.deltaVelocityStride;
    candidate.substepIndex = context.substepIndex;
    candidate.transactionSlot = context.transactionSlot;
    candidate.physicsSubstepCount = context.physicsSubstepCount;
    candidate.controlStep = context.controlStep;
    candidate.timestepAndInverse = {
        context.timestepSeconds,
        1.0f / context.timestepSeconds,
        0.0f,
        0.0f,
    };
    candidate.programFingerprint = context.programFingerprint;
    candidate.transactionFingerprint = context.transactionFingerprint;
    candidate.linearizationEpoch = context.linearizationEpoch;
    candidate.slotGeneration = context.slotGeneration;

    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    __unsafe_unretained id<MTLBuffer> deltaVelocity =
        (__bridge id<MTLBuffer>)query.deltaVelocity;
    __unsafe_unretained id<MTLBuffer> candidateQ =
        (__bridge id<MTLBuffer>)query.candidateQ;
    __unsafe_unretained id<MTLBuffer> candidateBodies =
        (__bridge id<MTLBuffer>)query.candidateBodies;
    __unsafe_unretained id<MTLBuffer> candidatePoints =
        query.pointCount == 0u
            ? context.state->humanMatterBuffers[
                  kHumanMatterCandidatePointBuffer]
            : (__bridge id<MTLBuffer>)query.pointQueries;
    __unsafe_unretained id<MTLBuffer> candidatePointWorld =
        query.pointWorld == nullptr
            ? context.state->humanMatterBuffers[
                  kHumanMatterCandidatePointWorldBuffer]
            : (__bridge id<MTLBuffer>)query.pointWorld;
    __unsafe_unretained id<MTLBuffer> candidatePointJacobians =
        query.pointJacobians == nullptr
            ? context.state->humanMatterBuffers[
                  kHumanMatterCandidatePointJacobianBuffer]
            : (__bridge id<MTLBuffer>)query.pointJacobians;

    id<MTLComputeCommandEncoder> prepare =
        [commandBuffer computeCommandEncoder];
    if (prepare == nil) return false;
    prepare.label = @"NumanX Human/Matter exact candidate prepare";
    [prepare setComputePipelineState:
        context.state->humanMatterPrepareCandidatePipeline];
    [prepare setBytes:&candidate length:sizeof(candidate) atIndex:0u];
    [prepare setBuffer:context.state->buffers[1u] offset:0u atIndex:1u];
    [prepare setBuffer:context.state->buffers[3u] offset:0u atIndex:2u];
    [prepare setBuffer:context.state->humanMatterBuffers[
                           kHumanMatterQCheckpointBuffer]
                  offset:0u atIndex:3u];
    [prepare setBuffer:context.state->humanMatterBuffers[
                           kHumanMatterVCheckpointBuffer]
                  offset:0u atIndex:4u];
    [prepare setBuffer:deltaVelocity offset:0u atIndex:5u];
    [prepare setBuffer:candidateQ offset:0u atIndex:6u];
    [prepare setBuffer:context.state->buffers[7u] offset:0u atIndex:7u];
    [prepare setBuffer:candidatePoints offset:0u atIndex:8u];
    [prepare setBuffer:context.state->humanMatterBuffers[
                           kHumanMatterCandidatePointBuffer]
                  offset:0u atIndex:9u];
    [prepare dispatchThreads:MTLSizeMake(context.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            std::min<std::uint32_t>(
                context.environmentCount, kThreadsPerThreadgroup),
            1u, 1u)];
    [prepare endEncoding];

    MRArticulatedOperatorDispatchGPU operatorDispatch{};
    operatorDispatch.articulationIndex = context.articulationIndex;
    operatorDispatch.environmentCount = context.environmentCount;
    operatorDispatch.pointCount = static_cast<mr_u32>(combinedCount64);
    operatorDispatch.flags =
        MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY;
    operatorDispatch.qStride = query.candidateQStride;
    operatorDispatch.pointStride = context.combinedPointStride;
    operatorDispatch.bodyPoseStride = context.bodyCount;
    operatorDispatch.pointWorldStride = context.combinedPointStride;
    operatorDispatch.pointJacobianStride =
        candidate.privatePointJacobianStride;
    operatorDispatch.generalizedStride = context.nv;
    id<MTLComputeCommandEncoder> kinematics =
        [commandBuffer computeCommandEncoder];
    if (kinematics == nil) return false;
    kinematics.label = @"NumanX Human/Matter exact candidate kinematics";
    [kinematics setComputePipelineState:context.state->pipeline];
    for (NSUInteger index = 0u; index < 5u; ++index) {
        [kinematics setBuffer:context.state->buffers[index]
                       offset:0u atIndex:index];
    }
    [kinematics setBytes:&operatorDispatch
                   length:sizeof(operatorDispatch) atIndex:5u];
    [kinematics setBuffer:candidateQ offset:0u atIndex:6u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidatePointBuffer]
                     offset:0u atIndex:7u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidateBodyPoseBuffer]
                     offset:0u atIndex:8u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidatePointWorldBuffer]
                     offset:0u atIndex:9u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidateOperatorScratchBuffer]
                     offset:0u atIndex:10u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidatePointJacobianBuffer]
                     offset:0u atIndex:11u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidateOperatorScratchBuffer]
                     offset:0u atIndex:12u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidateOperatorScratchBuffer]
                     offset:0u atIndex:13u];
    [kinematics setBuffer:context.state->humanMatterBuffers[
                              kHumanMatterCandidateOperatorStatusBuffer]
                     offset:0u atIndex:14u];
    [kinematics setBuffer:context.state->buffers[15u]
                     offset:0u atIndex:15u];
    [kinematics setThreadgroupMemoryLength:
        detail::articulatedOperatorThreadgroupBytes(
            context.bodyCount, context.nv, false)
                                   atIndex:0u];
    [kinematics dispatchThreadgroups:
        MTLSizeMake(context.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            kThreadsPerThreadgroup, 1u, 1u)];
    [kinematics endEncoding];

    id<MTLComputeCommandEncoder> materialize =
        [commandBuffer computeCommandEncoder];
    if (materialize == nil) return false;
    materialize.label =
        @"NumanX Human/Matter exact candidate materialize";
    [materialize setComputePipelineState:
        context.state->humanMatterMaterializeCandidatePipeline];
    [materialize setBytes:&candidate length:sizeof(candidate) atIndex:0u];
    [materialize setBuffer:context.state->buffers[1u]
                     offset:0u atIndex:1u];
    [materialize setBuffer:context.state->buffers[4u]
                     offset:0u atIndex:2u];
    [materialize setBuffer:context.state->humanMatterBuffers[
                               kHumanMatterVCheckpointBuffer]
                      offset:0u atIndex:3u];
    [materialize setBuffer:deltaVelocity offset:0u atIndex:4u];
    [materialize setBuffer:candidateQ offset:0u atIndex:5u];
    [materialize setBuffer:context.state->humanMatterBuffers[
                               kHumanMatterCandidateBodyPoseBuffer]
                      offset:0u atIndex:6u];
    [materialize setBuffer:context.state->humanMatterBuffers[
                               kHumanMatterCandidatePointWorldBuffer]
                      offset:0u atIndex:7u];
    [materialize setBuffer:context.state->humanMatterBuffers[
                               kHumanMatterCandidatePointJacobianBuffer]
                      offset:0u atIndex:8u];
    [materialize setBuffer:context.state->humanMatterBuffers[
                               kHumanMatterCandidateOperatorStatusBuffer]
                      offset:0u atIndex:9u];
    [materialize setBuffer:candidateBodies offset:0u atIndex:10u];
    [materialize setBuffer:candidatePointWorld offset:0u atIndex:11u];
    [materialize setBuffer:candidatePointJacobians
                      offset:0u atIndex:12u];
    [materialize setThreadgroupMemoryLength:sizeof(std::uint32_t)
                                      atIndex:0u];
    [materialize dispatchThreadgroups:
        MTLSizeMake(context.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(
            kStandThreadsPerThreadgroup, 1u, 1u)];
    [materialize endEncoding];
    return true;
}

[[nodiscard]] MRNumiHumanStandDispatchGPU makeNumiHumanStandDispatch(
    const MetalArticulatedOperatorInput& input,
    const MetalArticulatedOperatorLayout& layout,
    const MRArticulationGPU& articulation,
    const float timestepSeconds,
    const std::uint32_t stepIndex
) noexcept {
    MRNumiHumanStandDispatchGPU dispatch{};
    dispatch.abiVersion = MR_NUMI_HUMAN_STAND_ABI_VERSION;
    dispatch.environmentCount = static_cast<mr_u32>(
        input.environmentCount
    );
    dispatch.articulationIndex = input.articulationIndex;
    dispatch.stepIndex = stepIndex;
    dispatch.stepCount = input.stand.stepCount;
    dispatch.bodyJacobianPointOffset =
        input.mujoco.bodyJacobianPointOffset;
    dispatch.supportContactCount = static_cast<mr_u32>(
        input.stand.contacts.size()
    );
    if (input.stand.enableContact) {
        dispatch.flags |= MR_NUMI_HUMAN_STAND_ENABLE_CONTACT;
    }
    if (input.stand.enableRootAssistance) {
        dispatch.flags |= MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE;
    }
    dispatch.qStride = articulation.nq;
    dispatch.vStride = articulation.nv;
    dispatch.pointWorldStride = layout.dispatch.pointWorldStride;
    dispatch.pointJacobianStride = layout.dispatch.pointJacobianStride;
    dispatch.bodyPoseStride = articulation.bodyCount;
    dispatch.generalizedForceStride = articulation.nv;
    dispatch.generalizedForceOffset = static_cast<mr_u32>(
        layout.mujocoMuscleGeneralizedForceElements
    );
    dispatch.contactIterationCount = input.stand.contactIterationCount;
    dispatch.tendonEndpointCount = static_cast<mr_u32>(
        input.stand.tendonBindings.size()
    );
    dispatch.tendonEnvelopeCount = static_cast<mr_u32>(
        input.stand.tendonEnvelopes.size()
    );
    dispatch.tendonTransferStride = static_cast<mr_u32>(
        input.stand.tendonBindings.size()
    );
    dispatch.jointEqualityCount = static_cast<mr_u32>(
        input.stand.jointEqualities.size()
    );
    if (!input.stand.tendonBindings.empty()) {
        dispatch.flags |= MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS;
    }
    if (!input.stand.jointEqualities.empty()) {
        dispatch.flags |= MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES;
    }
    dispatch.groundPointAndTimestep = {
        input.stand.groundPoint.x,
        input.stand.groundPoint.y,
        input.stand.groundPoint.z,
        timestepSeconds,
    };
    dispatch.groundNormal = input.stand.groundNormal;
    dispatch.targetRootPosition = input.stand.targetRootPosition;
    dispatch.targetRootOrientation = input.stand.targetRootOrientation;
    dispatch.assistanceGains = input.stand.assistanceGains;
    return dispatch;
}

// The standalone compatibility entry point still uses an isolated arena so
// that its historical lifetime and transaction semantics remain unchanged.
// Reusable callers should use MetalArticulatedOperatorContext.
id<MTLBuffer> makeInputBuffer(
    id<MTLDevice> device,
    const void* source,
    const BufferRequirement& requirement,
    NSString* label
) {
    id<MTLBuffer> result = [device
        newBufferWithLength:static_cast<NSUInteger>(
            requirement.allocationBytes
        )
                   options:MTLResourceStorageModeShared];
    if (result != nil && result.contents != nullptr) {
        copyToBuffer(result, source, requirement);
    }
    result.label = label;
    return result;
}

id<MTLBuffer> makeOutputBuffer(
    id<MTLDevice> device,
    const BufferRequirement& requirement,
    NSString* label
) {
    id<MTLBuffer> result = [device
        newBufferWithLength:static_cast<NSUInteger>(
            requirement.allocationBytes
        )
                   options:MTLResourceStorageModeShared];
    if (result != nil && result.contents != nullptr) {
        std::memset(
            result.contents,
            0,
            requirement.allocationBytes
        );
    }
    result.label = label;
    return result;
}

bool validBuffer(
    id<MTLBuffer> buffer,
    const BufferRequirement& requirement
) {
    return buffer != nil &&
        buffer.contents != nullptr &&
        buffer.length >= requirement.allocationBytes;
}

template <typename T>
void copyOutput(
    std::vector<T>& destination,
    id<MTLBuffer> source
) {
    if (!destination.empty()) {
        std::memcpy(
            destination.data(),
            source.contents,
            destination.size() * sizeof(T)
        );
    }
}

bool finitePayload(
    const MetalArticulatedOperatorResult& result
) {
    return
        std::all_of(
            result.bodyPoses.begin(),
            result.bodyPoses.end(),
            [](const MRArticulatedBodyPoseGPU& pose) {
                return finite(pose.position) &&
                    finite(pose.orientation);
            }
        ) &&
        std::all_of(
            result.pointWorld.begin(),
            result.pointWorld.end(),
            [](const MRArticulatedPointWorldGPU& point) {
                return finite(point.position);
            }
        ) &&
        std::all_of(
            result.diagnosticMassMatrix.begin(),
            result.diagnosticMassMatrix.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.pointJacobians.begin(),
            result.pointJacobians.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.generalizedImpulse.begin(),
            result.generalizedImpulse.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.deltaVelocity.begin(),
            result.deltaVelocity.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.millardResults.begin(),
            result.millardResults.end(),
            [](const MRMillardMuscleResultGPU& value) {
                return finite(value.pathFiberTendonResidual);
            }
        ) &&
        std::all_of(
            result.millardGeneralizedForces.begin(),
            result.millardGeneralizedForces.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.mujocoResults.begin(),
            result.mujocoResults.end(),
            [](const MRMujocoMuscleResultGPU& value) {
                return finite(value);
            }
        ) &&
        std::all_of(
            result.mujocoActivationStates.begin(),
            result.mujocoActivationStates.end(),
            [](const MRMujocoMuscleStateGPU& value) {
                return finite(value.excitationAndActivation) &&
                    value.excitationAndActivation.z >= 0.0f;
            }
        ) &&
        std::all_of(
            result.mujocoMuscleGeneralizedForces.begin(),
            result.mujocoMuscleGeneralizedForces.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.mujocoGeneralizedForces.begin(),
            result.mujocoGeneralizedForces.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        ) &&
        std::all_of(
            result.standQ.begin(), result.standQ.end(),
            [](const float value) { return std::isfinite(value); }
        ) &&
        std::all_of(
            result.standV.begin(), result.standV.end(),
            [](const float value) { return std::isfinite(value); }
        ) &&
        std::all_of(
            result.standTendonTransfers.begin(),
            result.standTendonTransfers.end(),
            [](const MRNumiHumanTendonTransferResultGPU& value) {
                return finite(value.terminalWorldForce) &&
                    finite(value.residualsAndForce) &&
                    std::all_of(
                        std::begin(value.nodalWorldForces),
                        std::end(value.nodalWorldForces),
                        [](const mr_float4 force) { return finite(force); }
                    );
            }
        ) &&
        std::all_of(
            result.standTendonGeneralizedCorrections.begin(),
            result.standTendonGeneralizedCorrections.end(),
            [](const float value) { return std::isfinite(value); }
        );
}

} // namespace

MetalArticulatedOperatorSubmission::
    MetalArticulatedOperatorSubmission() noexcept = default;

MetalArticulatedOperatorSubmission::
    ~MetalArticulatedOperatorSubmission() = default;

MetalArticulatedOperatorSubmission::
    MetalArticulatedOperatorSubmission(
        MetalArticulatedOperatorSubmission&& other
    ) noexcept = default;

MetalArticulatedOperatorSubmission&
MetalArticulatedOperatorSubmission::operator=(
    MetalArticulatedOperatorSubmission&& other
) noexcept = default;

bool MetalArticulatedOperatorSubmission::valid() const noexcept {
    return state_ != nullptr;
}

bool MetalArticulatedOperatorSubmission::extractPreparedHumanMatter(
    MetalNumanXHumanMatterPrepared& output
) noexcept {
    if (state_ == nullptr || output.state_ != nullptr ||
        output.capability_ != nullptr || output.slotGeneration_ != 0u) {
        return false;
    }
    try {
        const std::shared_ptr<detail::MetalArticulatedOperatorContextState>
            context = state_->context;
        if (context == nullptr) return false;
        auto capability = std::make_shared<
            detail::MetalNumanXHumanMatterPreparedState>();
        const std::lock_guard lock(context->mutex);
        auto& prepared = context->humanMatterPrepared;
        if (!state_->hasPreparedHumanMatter || !state_->ownsInFlight ||
            state_->preparedHumanMatterGeneration == 0u ||
            !prepared.active || prepared.capability.lock() != nullptr ||
            prepared.dispatch.slotGeneration !=
                state_->preparedHumanMatterGeneration) {
            return false;
        }
        const std::uint64_t generation =
            state_->preparedHumanMatterGeneration;
        capability->submission = std::move(state_);
        prepared.capability = capability;
        output.state_ = context;
        output.capability_ = std::move(capability);
        output.slotGeneration_ = generation;
        return true;
    } catch (...) {
        return false;
    }
}

MetalNumanXHumanMatterPrepared::
    MetalNumanXHumanMatterPrepared() noexcept = default;

MetalNumanXHumanMatterPrepared::
    ~MetalNumanXHumanMatterPrepared() = default;

MetalNumanXHumanMatterPrepared::
    MetalNumanXHumanMatterPrepared(
        MetalNumanXHumanMatterPrepared&& other
    ) noexcept = default;

MetalNumanXHumanMatterPrepared&
MetalNumanXHumanMatterPrepared::operator=(
    MetalNumanXHumanMatterPrepared&& other
) noexcept = default;

bool MetalNumanXHumanMatterPrepared::valid() const noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) return false;
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->humanMatterPrepared.active &&
            state_->humanMatterPrepared.dispatch.slotGeneration ==
                slotGeneration_ &&
            state_->humanMatterPrepared.capability.lock() == capability_;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::view(
    MetalNumanXHumanMatterPreparedView& output
) const noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) return false;
    try {
        const std::lock_guard lock(state_->mutex);
        const auto& prepared = state_->humanMatterPrepared;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_ ||
            prepared.preparedTokens == nil ||
            state_->humanMatterBuffers[kHumanMatterProposalBuffer] == nil ||
            state_->humanMatterBuffers[kHumanMatterProposedTokenBuffer] == nil ||
            state_->humanMatterBuffers[kHumanMatterApplyActionBuffer] == nil ||
            state_->humanMatterBuffers[kHumanMatterAppliedOutcomeBuffer] == nil ||
            state_->humanMatterBuffers[
                kHumanMatterFinalAcceptedTokenBuffer] == nil ||
            state_->humanMatterBuffers[
                kHumanMatterPublicationFenceBuffer] == nil ||
            state_->humanMatterTimelineEvent == nil) {
            return false;
        }
        MetalNumanXHumanMatterPreparedView staged{};
        staged.environmentCount = prepared.dispatch.environmentCount;
        staged.transactionSlot = prepared.transactionSlot;
        staged.stepIndex = prepared.dispatch.stepIndex;
        staged.substepIndex = prepared.dispatch.substepIndex;
        staged.physicsSubstepCount =
            prepared.dispatch.physicsSubstepCount;
        staged.finalTokenStrideBytes =
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
        staged.proposalStride = 1u;
        staged.proposedTokenStrideBytes =
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
        staged.applyActionStride = 1u;
        staged.matterApplyOutcomeStride =
            prepared.lease.matterApplyOutcomeStride;
        staged.appliedOutcomeStride = 1u;
        staged.publicationFenceStride = 1u;
        staged.preparedPhysicsStateTokens =
            (__bridge void*)prepared.preparedTokens;
        staged.finalAcceptedPhysicsStateTokens =
            (__bridge void*)state_->humanMatterBuffers[
                kHumanMatterFinalAcceptedTokenBuffer];
        staged.proposals = (__bridge void*)state_->humanMatterBuffers[
            kHumanMatterProposalBuffer];
        staged.proposedPhysicsStateTokens =
            (__bridge void*)state_->humanMatterBuffers[
                kHumanMatterProposedTokenBuffer];
        staged.applyActions = (__bridge void*)state_->humanMatterBuffers[
            kHumanMatterApplyActionBuffer];
        staged.matterApplyOutcomes = prepared.lease.matterApplyOutcomes;
        staged.appliedOutcomes = (__bridge void*)state_->humanMatterBuffers[
            kHumanMatterAppliedOutcomeBuffer];
        staged.publicationFences = (__bridge void*)state_->humanMatterBuffers[
            kHumanMatterPublicationFenceBuffer];
        staged.physicalPreparedEvent =
            (__bridge void*)state_->humanMatterTimelineEvent;
        staged.preparedPhysicsStateTokensGPUAddress =
            prepared.preparedTokens.gpuAddress;
        staged.finalAcceptedPhysicsStateTokensGPUAddress =
            state_->humanMatterBuffers[
                kHumanMatterFinalAcceptedTokenBuffer].gpuAddress;
        staged.proposalsGPUAddress = state_->humanMatterBuffers[
            kHumanMatterProposalBuffer].gpuAddress;
        staged.proposedPhysicsStateTokensGPUAddress =
            state_->humanMatterBuffers[
                kHumanMatterProposedTokenBuffer].gpuAddress;
        staged.applyActionsGPUAddress = state_->humanMatterBuffers[
            kHumanMatterApplyActionBuffer].gpuAddress;
        staged.matterApplyOutcomesGPUAddress =
            prepared.lease.matterApplyOutcomesGPUAddress;
        staged.appliedOutcomesGPUAddress = state_->humanMatterBuffers[
            kHumanMatterAppliedOutcomeBuffer].gpuAddress;
        staged.publicationFencesGPUAddress = state_->humanMatterBuffers[
            kHumanMatterPublicationFenceBuffer].gpuAddress;
        staged.preparedPhysicsStateTokenByteCount =
            prepared.preparedTokenByteCount;
        staged.finalAcceptedPhysicsStateTokenByteCount =
            prepared.finalTokenByteCount;
        staged.proposalElementCount = prepared.proposalElementCount;
        staged.proposedPhysicsStateTokenByteCount =
            prepared.proposedTokenByteCount;
        staged.applyActionElementCount = prepared.applyActionElementCount;
        staged.matterApplyOutcomeElementCount =
            prepared.lease.matterApplyOutcomeElementCount;
        staged.appliedOutcomeElementCount =
            prepared.appliedOutcomeElementCount;
        staged.publicationFenceElementCount =
            prepared.publicationFenceElementCount;
        staged.physicalPreparedEventValue =
            prepared.physicalPreparedEventValue;
        staged.proposalEventValue = prepared.proposalEventValue;
        staged.appliedEventValue = prepared.appliedEventValue;
        staged.controlStep = prepared.dispatch.controlStep;
        staged.qCoordinateCount = prepared.dispatch.nq;
        staged.dofCount = prepared.dispatch.nv;
        staged.dofLayoutVersion =
            kMetalNumanXHumanMatterDofLayoutVersion;
        staged.programFingerprint = prepared.dispatch.programFingerprint;
        staged.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        staged.linearizationEpoch = prepared.dispatch.linearizationEpoch;
        staged.slotGeneration = prepared.dispatch.slotGeneration;
        const auto& humanIO = prepared.lease.humanIOCandidate;
        if (humanIO.valid()) {
            staged.humanIOCandidateKeyFingerprint =
                humanIO.candidateKeyFingerprint;
            staged.acceptedBrainGeneration = humanIO.acceptedBrainGeneration;
            staged.humanIOSensorGeneration = humanIO.sensorGeneration;
            staged.humanIOProgramFingerprint =
                humanIO.humanIOProgramFingerprint;
            staged.humanIOSensorFingerprint = humanIO.sensorFingerprint;
            staged.humanIOTransactionInstanceFingerprint =
                humanIO.transactionInstanceFingerprint;
            staged.humanIOCandidatePublicationFingerprint =
                humanIO.candidatePublicationFingerprint;
            staged.humanIODeviceRegistryID = humanIO.deviceRegistryID;
            staged.humanIOIdentityFingerprint = humanIO.identityFingerprint;
        }
        output = staged;
        return true;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::encodeWaitForPhysicalPrepare(
    void* commandBuffer
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u ||
        commandBuffer == nullptr) return false;
    try {
        const std::lock_guard lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        __unsafe_unretained id<MTLCommandBuffer> borrowed =
            (__bridge id<MTLCommandBuffer>)commandBuffer;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_ ||
            prepared.physicalPreparedEventValue == 0u ||
            state_->humanMatterTimelineEvent == nil || borrowed == nil ||
            borrowed.device != state_->device ||
            borrowed.status != MTLCommandBufferStatusNotEnqueued) {
            return false;
        }
        [borrowed encodeWaitForEvent:state_->humanMatterTimelineEvent
                              value:prepared.physicalPreparedEventValue];
        return true;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::registerPhysicalCompletion(
    void* completionContext,
    const MetalNumanXHumanMatterPhysicalCompletion completion
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u || completionContext == nullptr ||
        completion == nullptr) {
        return false;
    }
    try {
        HumanMatterPhysicalCompletionInvocation invocation{};
        {
            const std::lock_guard lock(state_->mutex);
            auto& prepared = state_->humanMatterPrepared;
            if (!prepared.active ||
                prepared.dispatch.slotGeneration != slotGeneration_ ||
                prepared.capability.lock() != capability_ ||
                prepared.physicalCompletionRegistered ||
                prepared.physicalCompletionDelivered) {
                return false;
            }
            prepared.physicalCompletionRegistered = true;
            prepared.physicalCompletionContext = completionContext;
            prepared.physicalCompletion = completion;
            invocation = takeHumanMatterPhysicalCompletionLocked(prepared);
        }
        invokeHumanMatterPhysicalCompletion(invocation);
        return true;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::bindHumanIOCandidatePublication(
    const MetalNumanXHumanIOCandidatePublicationProgram& candidate
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u || !candidate.valid()) return false;
    try {
        std::unique_lock lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        const auto exactGeneration = [&]() noexcept {
            return prepared.active &&
                prepared.dispatch.slotGeneration == slotGeneration_ &&
                prepared.capability.lock() == capability_;
        };
        if (!exactGeneration() || prepared.humanIOBindInFlight ||
            prepared.lease.humanIOCandidate.configured() ||
            prepared.proposalInFlight || prepared.proposalComplete ||
            prepared.proposalFailed ||
            prepared.applicationReservationInFlight ||
            prepared.applicationReserved || prepared.applyInFlight ||
            prepared.applyComplete || prepared.applyFailed ||
            prepared.physicalFailure || prepared.terminalNoTouch ||
            prepared.bindHumanIOCandidatePublication == nullptr ||
            candidate.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            candidate.deviceRegistryID != state_->device.registryID ||
            capability_->submission == nullptr ||
            capability_->submission->commandBuffer == nil ||
            capability_->submission->commandBuffer.device != state_->device ||
            capability_->submission->commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
            return false;
        }
        const auto* owner = static_cast<const
            MRNumanXHumanMatterOwnerStatusGPU*>(
            state_->humanMatterBuffers[
                kHumanMatterOwnerStatusBuffer].contents);
        if (owner == nullptr ||
            owner[0u].abiVersion != MR_NUMANX_HUMAN_MATTER_ABI_VERSION ||
            owner[0u].environment != 0u ||
            owner[0u].transactionSlot != prepared.transactionSlot ||
            owner[0u].controlStep != prepared.dispatch.controlStep ||
            owner[0u].qCoordinateCount != prepared.dispatch.nq ||
            owner[0u].dofCount != prepared.dispatch.nv ||
            owner[0u].programFingerprint !=
                prepared.dispatch.programFingerprint ||
            owner[0u].transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            owner[0u].linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            owner[0u].slotGeneration != prepared.dispatch.slotGeneration ||
            owner[0u].physicalCommandStatus !=
                MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_COMPLETE) {
            return false;
        }

        MetalNumanXHumanMatterPrepareLease stagedLease = prepared.lease;
        stagedLease.humanIOCandidate = candidate;
        const auto callback = prepared.bindHumanIOCandidatePublication;
        void* callbackContext = prepared.leaseContext;
        const std::uint64_t generation = prepared.dispatch.slotGeneration;
        prepared.humanIOBindInFlight = true;
        lock.unlock();
        const bool bound = callback(callbackContext, stagedLease, candidate);
        lock.lock();
        auto& current = state_->humanMatterPrepared;
        if (!current.active ||
            current.dispatch.slotGeneration != generation ||
            current.capability.lock() != capability_ ||
            !current.humanIOBindInFlight) {
            return false;
        }
        current.humanIOBindInFlight = false;
        if (!bound) return false;
        if (current.lease.humanIOCandidate.configured() ||
            current.proposalInFlight || current.proposalComplete ||
            current.proposalFailed || current.physicalFailure ||
            current.terminalNoTouch) {
            current.terminalNoTouch = true;
            return false;
        }
        current.lease = stagedLease;
        return true;
    } catch (...) {
        try {
            const std::lock_guard lock(state_->mutex);
            auto& prepared = state_->humanMatterPrepared;
            if (prepared.active &&
                prepared.dispatch.slotGeneration == slotGeneration_ &&
                prepared.capability.lock() == capability_) {
                prepared.humanIOBindInFlight = false;
            }
        } catch (...) {
        }
        return false;
    }
}


MetalNumanXHumanMatterOperationDiagnostics
MetalNumanXHumanMatterPrepared::proposePrepared(
    const MetalNumanXHumanMatterProposalRequest& request
) noexcept {
    MetalNumanXHumanMatterOperationDiagnostics diagnostics{};
    const auto fail = [&diagnostics](
        const MetalNumanXHumanMatterOperationStatus status,
        const char* message
    ) {
        diagnostics.status = status;
        diagnostics.message = message;
        return diagnostics;
    };
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) {
        return fail(
            MetalNumanXHumanMatterOperationStatus::invalidHandle,
            "prepared Human/Matter handle is empty");
    }
    try {
        const std::lock_guard lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidHandle,
                "prepared Human/Matter generation is no longer active");
        }
        if (prepared.proposalInFlight || prepared.proposalComplete ||
            prepared.humanIOBindInFlight ||
            prepared.applicationReservationInFlight ||
            prepared.applicationReserved || prepared.applyInFlight ||
            prepared.applyComplete) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::alreadyFinalizing,
                "prepared Human/Matter generation already has a proposal or apply attempt");
        }
        if (prepared.terminalNoTouch || prepared.proposalFailed ||
            prepared.applyFailed) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::terminalNoTouch,
                "prepared Human/Matter generation is terminally quarantined");
        }
        if (!prepared.lease.humanIOCandidate.valid() ||
            prepared.lease.humanIOCandidate.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "proposal requires the exact post-physical HumanIO candidate publication binding");
        }
        const bool validateWitness = request.mode ==
            MetalNumanXHumanMatterProposalMode::validateBrainWitness;
        if (request.abiVersion != kMetalNumanXHumanMatterABIVersion ||
            request.structSize != sizeof(request) || request.reserved1 != 0u ||
            (!validateWitness && request.mode !=
                MetalNumanXHumanMatterProposalMode::forceReject) ||
            request.environmentCount != prepared.dispatch.environmentCount ||
            request.transactionSlot != prepared.transactionSlot ||
            request.stepIndex != prepared.dispatch.stepIndex ||
            request.substepIndex != prepared.dispatch.substepIndex ||
            request.physicsSubstepCount !=
                prepared.dispatch.physicsSubstepCount ||
            request.controlStep != prepared.dispatch.controlStep ||
            request.programFingerprint != prepared.dispatch.programFingerprint ||
            request.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            request.linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            request.slotGeneration != prepared.dispatch.slotGeneration ||
            request.commandBuffer == nullptr) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "proposal metadata does not match the quarantined generation");
        }
        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)request.commandBuffer;
        if (commandBuffer == nil || commandBuffer.device != state_->device ||
            commandBuffer.status != MTLCommandBufferStatusNotEnqueued ||
            state_->humanMatterTimelineEvent == nil ||
            prepared.physicalPreparedEventValue == 0u ||
            prepared.proposalEventValue == 0u) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "proposal requires an uncommitted same-device borrowed command buffer and live owner timeline");
        }

        __unsafe_unretained id<MTLSharedEvent> brainEvent = nil;
        std::uint64_t witnessBytes = 0u;
        MetalBufferRegion witnessRegion{};
        if (validateWitness) {
            brainEvent = (__bridge id<MTLSharedEvent>)
                request.brainPrepareCompleteEvent;
            const std::uint64_t required =
                static_cast<std::uint64_t>(request.environmentCount - 1u) *
                    request.brainCommitWitnessStride + 1u;
            if (request.brainCommitWitnesses == nullptr || brainEvent == nil ||
                brainEvent == state_->humanMatterTimelineEvent ||
                request.brainPrepareCompleteEventValue == 0u ||
                request.brainCommitWitnessStride == 0u ||
                request.brainCommitWitnessElementCount < required ||
                request.brainCommitWitnessElementCount >
                    std::numeric_limits<std::uint64_t>::max() /
                        sizeof(MRNumanXHumanMatterBrainCommitWitnessGPU)) {
                return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain commit witness layout is undersized or malformed");
            }
            witnessBytes = request.brainCommitWitnessElementCount *
                sizeof(MRNumanXHumanMatterBrainCommitWitnessGPU);
            if (!importableSharedEvent(state_->device, brainEvent) ||
                !exactBufferRegion(
                    state_->device,
                    request.brainCommitWitnesses,
                    request.brainCommitWitnessesGPUAddress,
                    witnessBytes,
                    witnessRegion)) {
                return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain witness/event is not an exact importable same-device resource");
            }
            const auto aliases = [&](id<MTLBuffer> buffer) noexcept {
                if (buffer == nil) return false;
                MetalBufferRegion region{};
                return !ownedBufferRegion(buffer, region) ||
                    regionsOverlap(witnessRegion, region);
            };
            for (id<MTLBuffer> buffer : state_->buffers) {
                if (aliases(buffer)) return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain witness aliases owner state");
            }
            for (id<MTLBuffer> buffer : state_->standBuffers) {
                if (aliases(buffer)) return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain witness aliases stand state");
            }
            for (id<MTLBuffer> buffer : state_->humanMatterBuffers) {
                if (aliases(buffer)) return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain witness aliases owner transaction state");
            }
            if (aliases(prepared.preparedTokens) ||
                aliases((__bridge id<MTLBuffer>)
                    prepared.lease.matterApplyOutcomes)) {
                return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain witness aliases adapter transaction state");
            }
        } else if (request.brainCommitWitnesses != nullptr ||
                   request.brainPrepareCompleteEvent != nullptr ||
                   request.brainPrepareCompleteEventValue != 0u ||
                   request.brainCommitWitnessesGPUAddress != 0u ||
                   request.brainCommitWitnessElementCount != 0u ||
                   request.brainCommitWitnessStride != 0u) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "forced proposal reject must not carry Brain resources");
        }
        const bool exactLease =
            prepared.lease.preparedPhysicsStateTokens ==
                (__bridge void*)prepared.preparedTokens &&
            prepared.lease.preparedPhysicsStateTokensGPUAddress ==
                prepared.preparedTokens.gpuAddress &&
            exactMetalBuffer(
                state_->device,
                prepared.lease.preparedPhysicsStateTokens,
                prepared.lease.preparedPhysicsStateTokensGPUAddress,
                prepared.lease.preparedPhysicsStateTokenByteCount) &&
            prepared.lease.proposals == (__bridge void*)
                state_->humanMatterBuffers[kHumanMatterProposalBuffer] &&
            prepared.lease.proposalsGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterProposalBuffer].gpuAddress &&
            prepared.lease.proposedPhysicsStateTokens == (__bridge void*)
                state_->humanMatterBuffers[kHumanMatterProposedTokenBuffer] &&
            prepared.lease.proposedPhysicsStateTokensGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterProposedTokenBuffer].gpuAddress;
        if (!exactLease) {
            prepared.terminalNoTouch = true;
            return fail(
                MetalNumanXHumanMatterOperationStatus::terminalNoTouch,
                "prepared lease resources changed before proposal");
        }

        MRNumanXHumanMatterProposalDispatchGPU dispatch{};
        dispatch.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
        dispatch.flags = validateWitness
            ? MR_NUMANX_HUMAN_MATTER_PROPOSAL_VALIDATE_BRAIN_WITNESS
            : MR_NUMANX_HUMAN_MATTER_PROPOSAL_FORCE_REJECT;
        dispatch.environmentCount = prepared.dispatch.environmentCount;
        dispatch.stepIndex = prepared.dispatch.stepIndex;
        dispatch.substepIndex = prepared.dispatch.substepIndex;
        dispatch.transactionSlot = prepared.transactionSlot;
        dispatch.ownerStatusStride = prepared.dispatch.ownerStatusStride;
        dispatch.brainWitnessStride = validateWitness
            ? request.brainCommitWitnessStride : 0u;
        dispatch.preparedTokenStrideBytes =
            prepared.dispatch.acceptedTokenStrideBytes;
        dispatch.proposalStride = 1u;
        dispatch.proposedTokenStrideBytes =
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
        dispatch.physicsSubstepCount = prepared.dispatch.physicsSubstepCount;
        dispatch.controlStep = prepared.dispatch.controlStep;
        dispatch.programFingerprint = prepared.dispatch.programFingerprint;
        dispatch.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        dispatch.linearizationEpoch = prepared.dispatch.linearizationEpoch;
        dispatch.slotGeneration = prepared.dispatch.slotGeneration;
        dispatch.candidatePublicationFingerprint =
            prepared.lease.humanIOCandidate.candidatePublicationFingerprint;
        dispatch.humanIOIdentityFingerprint =
            prepared.lease.humanIOCandidate.identityFingerprint;

        @autoreleasepool {
            [commandBuffer encodeWaitForEvent:state_->humanMatterTimelineEvent
                                       value:prepared.physicalPreparedEventValue];
            if (validateWitness) {
                [commandBuffer encodeWaitForEvent:brainEvent
                                           value:
                    request.brainPrepareCompleteEventValue];
            }
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) return fail(
                MetalNumanXHumanMatterOperationStatus::metalEncoderFailure,
                "failed to create Human/Matter proposal encoder");
            encoder.label = @"NumanX Human/Matter mutation-free proposal";
            [encoder setComputePipelineState:
                state_->humanMatterProposePreparedPipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [encoder setBuffer:validateWitness
                    ? (__bridge id<MTLBuffer>)request.brainCommitWitnesses
                    : state_->humanMatterBuffers[kHumanMatterOwnerStatusBuffer]
                        offset:0u atIndex:1u];
            [encoder setBuffer:prepared.preparedTokens offset:0u atIndex:2u];
            [encoder setBuffer:state_->humanMatterBuffers[
                                   kHumanMatterOwnerStatusBuffer]
                         offset:0u atIndex:3u];
            [encoder setBuffer:state_->humanMatterBuffers[
                                   kHumanMatterProposalBuffer]
                         offset:0u atIndex:4u];
            [encoder setBuffer:state_->humanMatterBuffers[
                                   kHumanMatterProposedTokenBuffer]
                         offset:0u atIndex:5u];
            [encoder dispatchThreads:MTLSizeMake(
                dispatch.environmentCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
            [encoder endEncoding];
            prepared.proposalInFlight = true;
            prepared.proposalCommandBufferIdentity =
                reinterpret_cast<std::uintptr_t>(request.commandBuffer);
            const std::uint64_t attempt = ++prepared.proposalAttempt;
            const std::uint64_t generation = slotGeneration_;
            const auto retainedState = state_;
            const auto retainedCapability = capability_;
            __strong id<MTLSharedEvent> retainedTimeline =
                state_->humanMatterTimelineEvent;
            const std::uint64_t eventValue = prepared.proposalEventValue;
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                bool terminal = false;
                bool recordSettled = false;
                MetalNumanXHumanMatterReleasePrepareLease release = nullptr;
                void* releaseContext = nullptr;
                MetalNumanXHumanMatterPrepareLease lease{};
                HumanMatterProposalCompletionInvocation invocation{};
                try {
                    const std::lock_guard completionLock(retainedState->mutex);
                    auto& current = retainedState->humanMatterPrepared;
                    if (!current.active ||
                        current.dispatch.slotGeneration != generation ||
                        current.capability.lock() != retainedCapability ||
                        !current.proposalInFlight ||
                        current.proposalAttempt != attempt) return;
                    auto* proposal = static_cast<
                        MRNumanXHumanMatterProposalGPU*>(
                        retainedState->humanMatterBuffers[
                            kHumanMatterProposalBuffer].contents);
                    void* token = retainedState->humanMatterBuffers[
                        kHumanMatterProposedTokenBuffer].contents;
                    if (completed.status != MTLCommandBufferStatusCompleted) {
                        MRNumanXHumanMatterProposalGPU failed{};
                        failed.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
                        failed.status =
                            MR_NUMANX_HUMAN_MATTER_PROPOSAL_TERMINAL_NO_TOUCH;
                        failed.decision = MR_NUMANX_HUMAN_MATTER_ROOT_PENDING;
                        failed.code =
                            MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_OWNER;
                        failed.programFingerprint =
                            current.dispatch.programFingerprint;
                        failed.transactionFingerprint =
                            current.dispatch.transactionFingerprint;
                        failed.linearizationEpoch =
                            current.dispatch.linearizationEpoch;
                        failed.slotGeneration =
                            current.dispatch.slotGeneration;
                        failed.candidatePublicationFingerprint =
                            current.lease.humanIOCandidate.
                                candidatePublicationFingerprint;
                        failed.humanIOIdentityFingerprint =
                            current.lease.humanIOCandidate.identityFingerprint;
                        failed.environment = 0u;
                        failed.stepIndex = current.dispatch.stepIndex;
                        failed.substepIndex = current.dispatch.substepIndex;
                        failed.transactionSlot = current.transactionSlot;
                        failed.physicsSubstepCount =
                            current.dispatch.physicsSubstepCount;
                        failed.controlStep = current.dispatch.controlStep;
                        failed.proposalFingerprint =
                            humanMatterRecordFingerprint(&failed);
                        *proposal = failed;
                        std::memset(
                            token,
                            0,
                            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
                    }
                    const bool validRecord = completed.status ==
                            MTLCommandBufferStatusCompleted &&
                        validProposalRecord(
                            *proposal,
                            current.dispatch,
                            token,
                            current.lease.humanIOCandidate);
                    current.proposalInFlight = false;
                    current.proposalCommandBufferIdentity = 0u;
                    current.proposalComplete = validRecord;
                    current.proposalFailed = !validRecord;
                    recordSettled = true;
                    terminal = !validRecord || current.physicalFailure;
                    if (terminal) {
                        current.terminalNoTouch = true;
                        release = current.releasePrepareLease;
                        releaseContext = current.leaseContext;
                        lease = current.lease;
                    }
                    invocation =
                        takeHumanMatterProposalCompletionLocked(current);
                } catch (...) {
                    terminal = true;
                    // The completion block is a noexcept boundary. Retry an
                    // allocation-free terminal settlement under the owner
                    // mutex so a callback is never skipped by diagnostics or
                    // validation exceptions.
                    try {
                        const std::lock_guard completionLock(
                            retainedState->mutex);
                        auto& current = retainedState->humanMatterPrepared;
                        if (current.active &&
                            current.dispatch.slotGeneration == generation &&
                            current.capability.lock() == retainedCapability &&
                            current.proposalInFlight &&
                            current.proposalAttempt == attempt) {
                            auto* proposal = static_cast<
                                MRNumanXHumanMatterProposalGPU*>(
                                retainedState->humanMatterBuffers[
                                    kHumanMatterProposalBuffer].contents);
                            void* token = retainedState->humanMatterBuffers[
                                kHumanMatterProposedTokenBuffer].contents;
                            if (proposal != nullptr && token != nullptr) {
                                MRNumanXHumanMatterProposalGPU failed{};
                                failed.abiVersion =
                                    MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
                                failed.status =
                                    MR_NUMANX_HUMAN_MATTER_PROPOSAL_TERMINAL_NO_TOUCH;
                                failed.decision =
                                    MR_NUMANX_HUMAN_MATTER_ROOT_PENDING;
                                failed.code =
                                    MR_NUMANX_HUMAN_MATTER_PROPOSAL_INVALID_OWNER;
                                failed.programFingerprint =
                                    current.dispatch.programFingerprint;
                                failed.transactionFingerprint =
                                    current.dispatch.transactionFingerprint;
                                failed.linearizationEpoch =
                                    current.dispatch.linearizationEpoch;
                                failed.slotGeneration =
                                    current.dispatch.slotGeneration;
                                failed.candidatePublicationFingerprint =
                                    current.lease.humanIOCandidate.
                                        candidatePublicationFingerprint;
                                failed.humanIOIdentityFingerprint =
                                    current.lease.humanIOCandidate.
                                        identityFingerprint;
                                failed.environment = 0u;
                                failed.stepIndex = current.dispatch.stepIndex;
                                failed.substepIndex =
                                    current.dispatch.substepIndex;
                                failed.transactionSlot =
                                    current.transactionSlot;
                                failed.physicsSubstepCount =
                                    current.dispatch.physicsSubstepCount;
                                failed.controlStep =
                                    current.dispatch.controlStep;
                                failed.proposalFingerprint =
                                    humanMatterRecordFingerprint(&failed);
                                *proposal = failed;
                                std::memset(
                                    token,
                                    0,
                                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
                                recordSettled = true;
                            }
                            current.proposalInFlight = false;
                            current.proposalCommandBufferIdentity = 0u;
                            current.proposalComplete = false;
                            current.proposalFailed = true;
                            current.terminalNoTouch = true;
                            release = current.releasePrepareLease;
                            releaseContext = current.leaseContext;
                            lease = current.lease;
                            invocation =
                                takeHumanMatterProposalCompletionLocked(
                                    current);
                        }
                    } catch (...) {
                        // Never allow an exception to escape a Metal handler.
                    }
                }
                if (recordSettled && retainedTimeline != nil &&
                    retainedTimeline.signaledValue < eventValue) {
                    retainedTimeline.signaledValue = eventValue;
                }
                if (terminal && release != nullptr) {
                    (void)release(
                        releaseContext,
                        lease,
                        (__bridge void*)completed,
                        completed.status == MTLCommandBufferStatusCompleted);
                }
                invokeHumanMatterProposalCompletion(invocation);
            }];
        }
        diagnostics.encoded = true;
        return diagnostics;
    } catch (const std::exception& exception) {
        diagnostics.status =
            MetalNumanXHumanMatterOperationStatus::invalidRequest;
        diagnostics.message = exception.what();
        return diagnostics;
    } catch (...) {
        return fail(
            MetalNumanXHumanMatterOperationStatus::invalidRequest,
            "unexpected failure while encoding Human/Matter proposal");
    }
}

bool MetalNumanXHumanMatterPrepared::registerProposalCompletion(
    void* completionContext,
    const MetalNumanXHumanMatterProposalCompletion completion
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u || completionContext == nullptr ||
        completion == nullptr) {
        return false;
    }
    try {
        // Both locals deliberately survive the callback. It is legal for a
        // synchronous/reentrant callback to destroy or move the public
        // Prepared handle, so nothing may be accessed through `this` after
        // invocation.
        const auto retainedState = state_;
        const auto retainedCapability = capability_;
        const std::uint64_t generation = slotGeneration_;
        HumanMatterProposalCompletionInvocation invocation{};
        bool registered = false;
        {
            const std::lock_guard lock(retainedState->mutex);
            auto& prepared = retainedState->humanMatterPrepared;
            if (!prepared.active ||
                prepared.dispatch.slotGeneration != generation ||
                prepared.capability.lock() != retainedCapability ||
                (!prepared.proposalInFlight &&
                 !prepared.proposalComplete &&
                 !prepared.proposalFailed &&
                 !prepared.terminalNoTouch) ||
                prepared.proposalCompletionRegistered ||
                prepared.proposalCompletionDelivered) {
                return false;
            }
            prepared.proposalCompletionRegistered = true;
            prepared.proposalCompletionContext = completionContext;
            prepared.proposalCompletion = completion;
            invocation =
                takeHumanMatterProposalCompletionLocked(prepared);
            registered = true;
        }
        invokeHumanMatterProposalCompletion(invocation);
        return registered;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::abortProposal(
    void* commandBuffer
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u || commandBuffer == nullptr) return false;
    try {
        const std::lock_guard lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        __unsafe_unretained id<MTLCommandBuffer> borrowed =
            (__bridge id<MTLCommandBuffer>)commandBuffer;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_ ||
            !prepared.proposalInFlight ||
            prepared.proposalCommandBufferIdentity !=
                reinterpret_cast<std::uintptr_t>(commandBuffer) ||
            borrowed == nil ||
            borrowed.status != MTLCommandBufferStatusNotEnqueued) {
            return false;
        }
        prepared.proposalInFlight = false;
        prepared.proposalCommandBufferIdentity = 0u;
        ++prepared.proposalAttempt;
        prepared.proposalCompletionRegistered = false;
        prepared.proposalCompletionDelivered = false;
        prepared.proposalCompletionContext = nullptr;
        prepared.proposalCompletion = nullptr;
        std::memset(
            state_->humanMatterBuffers[kHumanMatterProposalBuffer].contents,
            0,
            static_cast<std::size_t>(prepared.proposalElementCount) *
                sizeof(MRNumanXHumanMatterProposalGPU));
        std::memset(
            state_->humanMatterBuffers[kHumanMatterProposedTokenBuffer].contents,
            0,
            static_cast<std::size_t>(prepared.proposedTokenByteCount));
        return true;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::reservePreparedApplication(
    const MetalNumanXHumanMatterBrainPreflightView& preflight
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) return false;
    try {
        std::unique_lock lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_ ||
            !prepared.proposalComplete || prepared.proposalInFlight ||
            prepared.proposalFailed || prepared.physicalFailure ||
            prepared.terminalNoTouch ||
            prepared.applicationReservationInFlight ||
            prepared.applicationReserved || prepared.applyInFlight ||
            prepared.applyComplete ||
            prepared.reservePreparedApplication == nullptr) {
            return false;
        }
        if (preflight.abiVersion != kMetalNumanXHumanMatterABIVersion ||
            preflight.structSize != sizeof(preflight) ||
            preflight.brainCommitPreflights == nullptr ||
            preflight.preflightReadyEvent == nullptr ||
            preflight.preflightReadyEventValue == 0u ||
            preflight.brainCommitPreflightStride == 0u ||
            preflight.environmentCount != prepared.dispatch.environmentCount ||
            preflight.transactionSlot != prepared.transactionSlot ||
            preflight.stepIndex != prepared.dispatch.stepIndex ||
            preflight.substepIndex != prepared.dispatch.substepIndex ||
            preflight.physicsSubstepCount !=
                prepared.dispatch.physicsSubstepCount ||
            preflight.controlStep != prepared.dispatch.controlStep ||
            preflight.programFingerprint !=
                prepared.dispatch.programFingerprint ||
            preflight.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            preflight.linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            preflight.slotGeneration != prepared.dispatch.slotGeneration) {
            return false;
        }
        const std::uint64_t required =
            static_cast<std::uint64_t>(preflight.environmentCount - 1u) *
                preflight.brainCommitPreflightStride + 1u;
        if (preflight.brainCommitPreflightElementCount < required ||
            preflight.brainCommitPreflightElementCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU)) {
            return false;
        }
        const std::uint64_t preflightBytes =
            preflight.brainCommitPreflightElementCount *
                sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU);
        __unsafe_unretained id<MTLSharedEvent> readyEvent =
            (__bridge id<MTLSharedEvent>)preflight.preflightReadyEvent;
        MetalBufferRegion preflightRegion{};
        if (readyEvent == nil || readyEvent == state_->humanMatterTimelineEvent ||
            !importableSharedEvent(state_->device, readyEvent) ||
            !exactBufferRegion(
                state_->device,
                preflight.brainCommitPreflights,
                preflight.brainCommitPreflightsGPUAddress,
                preflightBytes,
                preflightRegion)) {
            return false;
        }
        const auto aliases = [&](id<MTLBuffer> buffer) noexcept {
            if (buffer == nil) return false;
            MetalBufferRegion region{};
            return !ownedBufferRegion(buffer, region) ||
                regionsOverlap(preflightRegion, region);
        };
        for (id<MTLBuffer> buffer : state_->buffers) {
            if (aliases(buffer)) return false;
        }
        for (id<MTLBuffer> buffer : state_->standBuffers) {
            if (aliases(buffer)) return false;
        }
        for (id<MTLBuffer> buffer : state_->humanMatterBuffers) {
            if (aliases(buffer)) return false;
        }
        if (aliases(prepared.preparedTokens) ||
            aliases((__bridge id<MTLBuffer>)prepared.lease.matterApplyOutcomes)) {
            return false;
        }

        const auto* proposal = static_cast<const
            MRNumanXHumanMatterProposalGPU*>(
            state_->humanMatterBuffers[kHumanMatterProposalBuffer].contents);
        const void* proposedToken = state_->humanMatterBuffers[
            kHumanMatterProposedTokenBuffer].contents;
        if (proposal == nullptr ||
            !validProposalRecord(
                *proposal,
                prepared.dispatch,
                proposedToken,
                prepared.lease.humanIOCandidate)) {
            prepared.terminalNoTouch = true;
            return false;
        }

        MetalNumanXHumanMatterProposalView proposalView{};
        proposalView.proposals = prepared.lease.proposals;
        proposalView.proposedPhysicsStateTokens =
            prepared.lease.proposedPhysicsStateTokens;
        proposalView.proposalsGPUAddress = prepared.lease.proposalsGPUAddress;
        proposalView.proposedPhysicsStateTokensGPUAddress =
            prepared.lease.proposedPhysicsStateTokensGPUAddress;
        proposalView.proposalElementCount =
            prepared.lease.proposalElementCount;
        proposalView.proposedPhysicsStateTokenByteCount =
            prepared.lease.proposedPhysicsStateTokenByteCount;
        proposalView.proposalStride = prepared.lease.proposalStride;
        proposalView.proposedTokenStrideBytes =
            prepared.lease.proposedTokenStrideBytes;
        proposalView.proposalEventValue = prepared.proposalEventValue;
        const auto callback = prepared.reservePreparedApplication;
        void* callbackContext = prepared.leaseContext;
        const auto lease = prepared.lease;
        const std::uint64_t generation = prepared.dispatch.slotGeneration;
        prepared.applicationReservationInFlight = true;
        lock.unlock();
        const bool reserved = callback(
            callbackContext, lease, proposalView, preflight);
        lock.lock();
        auto& current = state_->humanMatterPrepared;
        if (!current.active ||
            current.dispatch.slotGeneration != generation ||
            current.capability.lock() != capability_ ||
            !current.applicationReservationInFlight) {
            return false;
        }
        current.applicationReservationInFlight = false;
        if (!reserved) return false;
        current.applicationReserved = true;
        current.preflight = preflight;
        return true;
    } catch (...) {
        return false;
    }
}

MetalNumanXHumanMatterOperationDiagnostics
MetalNumanXHumanMatterPrepared::applyPrepared(
    const MetalNumanXHumanMatterApplyRequest& request
) noexcept {
    MetalNumanXHumanMatterOperationDiagnostics diagnostics{};
    const auto fail = [&diagnostics](
        const MetalNumanXHumanMatterOperationStatus status,
        const char* message
    ) {
        diagnostics.status = status;
        diagnostics.message = message;
        return diagnostics;
    };
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) {
        return fail(
            MetalNumanXHumanMatterOperationStatus::invalidHandle,
            "prepared Human/Matter handle is empty");
    }
    try {
        const std::lock_guard lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidHandle,
                "prepared Human/Matter generation is no longer active");
        }
        if (prepared.applyInFlight || prepared.applyComplete) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::alreadyFinalizing,
                "prepared Human/Matter generation already has an apply command");
        }
        if (!prepared.proposalComplete || !prepared.applicationReserved ||
            prepared.applicationReservationInFlight ||
            prepared.physicalFailure || prepared.proposalFailed ||
            prepared.applyFailed || prepared.terminalNoTouch) {
            return fail(
                prepared.terminalNoTouch || prepared.physicalFailure ||
                        prepared.proposalFailed || prepared.applyFailed
                    ? MetalNumanXHumanMatterOperationStatus::terminalNoTouch
                    : MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "prepared Human/Matter generation is not reserved for apply");
        }
        const bool validateAck = request.mode ==
            MetalNumanXHumanMatterApplyMode::validateBrainAck;
        if (request.abiVersion != kMetalNumanXHumanMatterABIVersion ||
            request.structSize != sizeof(request) || request.reserved0 != 0u ||
            (!validateAck && request.mode !=
                MetalNumanXHumanMatterApplyMode::forceReject) ||
            request.environmentCount != prepared.dispatch.environmentCount ||
            request.transactionSlot != prepared.transactionSlot ||
            request.stepIndex != prepared.dispatch.stepIndex ||
            request.substepIndex != prepared.dispatch.substepIndex ||
            request.physicsSubstepCount !=
                prepared.dispatch.physicsSubstepCount ||
            request.controlStep != prepared.dispatch.controlStep ||
            request.programFingerprint != prepared.dispatch.programFingerprint ||
            request.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            request.linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            request.slotGeneration != prepared.dispatch.slotGeneration ||
            request.commandBuffer == nullptr ||
            request.completionContext == nullptr ||
            request.completion == nullptr) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "apply metadata does not match the quarantined generation");
        }
        __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)request.commandBuffer;
        if (commandBuffer == nil || commandBuffer.device != state_->device ||
            commandBuffer.status != MTLCommandBufferStatusNotEnqueued ||
            state_->humanMatterTimelineEvent == nil ||
            prepared.proposalEventValue == 0u ||
            prepared.appliedEventValue == 0u) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "apply requires an uncommitted same-device borrowed command buffer and live owner timeline");
        }

        __unsafe_unretained id<MTLSharedEvent> ackEvent = nil;
        MetalBufferRegion ackRegion{};
        std::uint64_t ackBytes = 0u;
        if (validateAck) {
            ackEvent = (__bridge id<MTLSharedEvent>)request.brainAckEvent;
            const std::uint64_t required =
                static_cast<std::uint64_t>(request.environmentCount - 1u) *
                    request.brainAckStride + 1u;
            if (request.brainAcks == nullptr || ackEvent == nil ||
                ackEvent == state_->humanMatterTimelineEvent ||
                ackEvent == (__bridge id<MTLSharedEvent>)
                    prepared.preflight.preflightReadyEvent ||
                request.brainAckEventValue == 0u ||
                request.brainAckStride == 0u ||
                request.brainAckElementCount < required ||
                request.brainAckElementCount >
                    std::numeric_limits<std::uint64_t>::max() /
                        sizeof(MRNumanXHumanMatterBrainAckGPU)) {
                return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain ACK layout/event is undersized or malformed");
            }
            ackBytes = request.brainAckElementCount *
                sizeof(MRNumanXHumanMatterBrainAckGPU);
            if (!importableSharedEvent(state_->device, ackEvent) ||
                !exactBufferRegion(
                    state_->device,
                    request.brainAcks,
                    request.brainAcksGPUAddress,
                    ackBytes,
                    ackRegion)) {
                return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain ACK/event is not an exact importable same-device resource");
            }
            const auto aliases = [&](id<MTLBuffer> buffer) noexcept {
                if (buffer == nil) return false;
                MetalBufferRegion region{};
                return !ownedBufferRegion(buffer, region) ||
                    regionsOverlap(ackRegion, region);
            };
            for (id<MTLBuffer> buffer : state_->buffers) {
                if (aliases(buffer)) return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain ACK aliases owner state");
            }
            for (id<MTLBuffer> buffer : state_->standBuffers) {
                if (aliases(buffer)) return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain ACK aliases stand state");
            }
            for (id<MTLBuffer> buffer : state_->humanMatterBuffers) {
                if (aliases(buffer)) return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain ACK aliases owner transaction state");
            }
            if (aliases(prepared.preparedTokens) ||
                aliases((__bridge id<MTLBuffer>)
                    prepared.lease.matterApplyOutcomes) ||
                aliases((__bridge id<MTLBuffer>)
                    prepared.preflight.brainCommitPreflights)) {
                return fail(
                    MetalNumanXHumanMatterOperationStatus::invalidRequest,
                    "Brain ACK aliases adapter or preflight state");
            }
        } else if (request.brainAcks != nullptr ||
                   request.brainAckEvent != nullptr ||
                   request.brainAckEventValue != 0u ||
                   request.brainAcksGPUAddress != 0u ||
                   request.brainAckElementCount != 0u ||
                   request.brainAckStride != 0u) {
            return fail(
                MetalNumanXHumanMatterOperationStatus::invalidRequest,
                "forced apply reject must not carry Brain ACK resources");
        }

        if (prepared.lease.matterApplyOutcomeElementCount == 0u ||
            prepared.lease.matterApplyOutcomeElementCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU)) {
            prepared.terminalNoTouch = true;
            return fail(
                MetalNumanXHumanMatterOperationStatus::terminalNoTouch,
                "Matter apply-outcome layout is no longer valid");
        }
        const std::uint64_t matterOutcomeBytes =
            prepared.lease.matterApplyOutcomeElementCount *
                sizeof(MRNumanXHumanMatterMatterApplyOutcomeGPU);
        const bool exactLease =
            prepared.lease.physicalPreparedEvent ==
                (__bridge void*)state_->humanMatterTimelineEvent &&
            prepared.lease.proposals == (__bridge void*)
                state_->humanMatterBuffers[kHumanMatterProposalBuffer] &&
            prepared.lease.proposedPhysicsStateTokens == (__bridge void*)
                state_->humanMatterBuffers[kHumanMatterProposedTokenBuffer] &&
            prepared.lease.applyActions == (__bridge void*)
                state_->humanMatterBuffers[kHumanMatterApplyActionBuffer] &&
            prepared.lease.appliedOutcomes == (__bridge void*)
                state_->humanMatterBuffers[kHumanMatterAppliedOutcomeBuffer] &&
            prepared.lease.finalAcceptedPhysicsStateTokens == (__bridge void*)
                state_->humanMatterBuffers[
                    kHumanMatterFinalAcceptedTokenBuffer] &&
            prepared.lease.publicationFences == (__bridge void*)
                state_->humanMatterBuffers[
                    kHumanMatterPublicationFenceBuffer] &&
            prepared.lease.proposalsGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterProposalBuffer].gpuAddress &&
            prepared.lease.proposedPhysicsStateTokensGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterProposedTokenBuffer].gpuAddress &&
            prepared.lease.applyActionsGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterApplyActionBuffer].gpuAddress &&
            prepared.lease.appliedOutcomesGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterAppliedOutcomeBuffer].gpuAddress &&
            prepared.lease.finalAcceptedPhysicsStateTokensGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterFinalAcceptedTokenBuffer].gpuAddress &&
            prepared.lease.publicationFencesGPUAddress ==
                state_->humanMatterBuffers[
                    kHumanMatterPublicationFenceBuffer].gpuAddress &&
            exactMetalBuffer(
                state_->device,
                prepared.lease.matterApplyOutcomes,
                prepared.lease.matterApplyOutcomesGPUAddress,
                matterOutcomeBytes);
        if (!exactLease) {
            prepared.terminalNoTouch = true;
            return fail(
                MetalNumanXHumanMatterOperationStatus::terminalNoTouch,
                "prepared lease resources changed before apply");
        }

        const auto* proposal = static_cast<const
            MRNumanXHumanMatterProposalGPU*>(
            state_->humanMatterBuffers[kHumanMatterProposalBuffer].contents);
        const void* proposedToken = state_->humanMatterBuffers[
            kHumanMatterProposedTokenBuffer].contents;
        if (proposal == nullptr ||
            !validProposalRecord(
                *proposal,
                prepared.dispatch,
                proposedToken,
                prepared.lease.humanIOCandidate)) {
            prepared.terminalNoTouch = true;
            return fail(
                MetalNumanXHumanMatterOperationStatus::terminalNoTouch,
                "immutable owner proposal or token failed validation");
        }
        const MRNumanXHumanMatterProposalGPU proposalSnapshot = *proposal;

        MRNumanXHumanMatterApplyDispatchGPU dispatch{};
        dispatch.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
        dispatch.flags = validateAck
            ? MR_NUMANX_HUMAN_MATTER_APPLY_VALIDATE_BRAIN_ACK
            : MR_NUMANX_HUMAN_MATTER_APPLY_FORCE_REJECT;
        dispatch.environmentCount = prepared.dispatch.environmentCount;
        dispatch.stepIndex = prepared.dispatch.stepIndex;
        dispatch.substepIndex = prepared.dispatch.substepIndex;
        dispatch.transactionSlot = prepared.transactionSlot;
        dispatch.nq = prepared.dispatch.nq;
        dispatch.nv = prepared.dispatch.nv;
        dispatch.qStride = prepared.dispatch.qStride;
        dispatch.vStride = prepared.dispatch.vStride;
        dispatch.mujocoStateStride = prepared.dispatch.mujocoStateStride;
        dispatch.mujocoStateCount = prepared.dispatch.mujocoStateCount;
        dispatch.ownerStatusStride = prepared.dispatch.ownerStatusStride;
        dispatch.brainAckStride = validateAck ? request.brainAckStride : 0u;
        dispatch.proposedTokenStrideBytes =
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
        dispatch.applyActionStride = 1u;
        dispatch.matterOutcomeStride =
            prepared.lease.matterApplyOutcomeStride;
        dispatch.appliedOutcomeStride = 1u;
        dispatch.finalTokenStrideBytes =
            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
        dispatch.physicsSubstepCount = prepared.dispatch.physicsSubstepCount;
        dispatch.controlStep = prepared.dispatch.controlStep;
        dispatch.programFingerprint = prepared.dispatch.programFingerprint;
        dispatch.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        dispatch.linearizationEpoch = prepared.dispatch.linearizationEpoch;
        dispatch.slotGeneration = prepared.dispatch.slotGeneration;

        MetalNumanXHumanMatterApplyPass pass{};
        pass.mode = request.mode;
        pass.commandBuffer = request.commandBuffer;
        pass.brainAcks = request.brainAcks;
        pass.brainAckEvent = request.brainAckEvent;
        pass.brainAcksGPUAddress = request.brainAcksGPUAddress;
        pass.brainAckElementCount = request.brainAckElementCount;
        pass.brainAckEventValue = request.brainAckEventValue;
        pass.brainAckStride = request.brainAckStride;
        pass.environmentCount = request.environmentCount;
        pass.transactionSlot = request.transactionSlot;
        pass.stepIndex = request.stepIndex;
        pass.substepIndex = request.substepIndex;
        pass.physicsSubstepCount = request.physicsSubstepCount;
        pass.controlStep = request.controlStep;
        pass.programFingerprint = request.programFingerprint;
        pass.transactionFingerprint = request.transactionFingerprint;
        pass.linearizationEpoch = request.linearizationEpoch;
        pass.slotGeneration = request.slotGeneration;

        @autoreleasepool {
            [commandBuffer encodeWaitForEvent:state_->humanMatterTimelineEvent
                                       value:prepared.proposalEventValue];
            if (validateAck) {
                [commandBuffer encodeWaitForEvent:ackEvent
                                           value:request.brainAckEventValue];
            }
            id<MTLComputeCommandEncoder> validate =
                [commandBuffer computeCommandEncoder];
            if (validate == nil) return fail(
                MetalNumanXHumanMatterOperationStatus::metalEncoderFailure,
                "failed to create Human/Matter apply-validation encoder");
            validate.label = @"NumanX Human/Matter validate apply action";
            [validate setComputePipelineState:
                state_->humanMatterValidateApplyPipeline];
            [validate setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [validate setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterProposalBuffer]
                           offset:0u atIndex:1u];
            [validate setBuffer:validateAck
                    ? (__bridge id<MTLBuffer>)request.brainAcks
                    : state_->humanMatterBuffers[kHumanMatterApplyActionBuffer]
                         offset:0u atIndex:2u];
            [validate setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterProposedTokenBuffer]
                           offset:0u atIndex:3u];
            [validate setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterOwnerStatusBuffer]
                           offset:0u atIndex:4u];
            [validate setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterApplyActionBuffer]
                           offset:0u atIndex:5u];
            [validate dispatchThreads:MTLSizeMake(
                dispatch.environmentCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
            [validate endEncoding];

            if (prepared.encodePreparedApply == nullptr ||
                prepared.abortPreparedApply == nullptr ||
                !prepared.encodePreparedApply(
                    prepared.leaseContext,
                    prepared.lease,
                    pass)) {
                if (prepared.abortPreparedApply != nullptr) {
                    prepared.abortPreparedApply(
                        prepared.leaseContext, prepared.lease, pass);
                }
                return fail(
                    MetalNumanXHumanMatterOperationStatus::metalEncoderFailure,
                    "Matter adapter rejected prepared apply encoding");
            }

            id<MTLComputeCommandEncoder> complete =
                [commandBuffer computeCommandEncoder];
            if (complete == nil) {
                prepared.abortPreparedApply(
                    prepared.leaseContext, prepared.lease, pass);
                return fail(
                    MetalNumanXHumanMatterOperationStatus::metalEncoderFailure,
                    "failed to create Human/Matter apply-completion encoder");
            }
            complete.label = @"NumanX Human/Matter complete apply";
            [complete setComputePipelineState:
                state_->humanMatterCompleteApplyPipeline];
            [complete setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterProposalBuffer]
                           offset:0u atIndex:1u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterApplyActionBuffer]
                           offset:0u atIndex:2u];
            [complete setBuffer:(__bridge id<MTLBuffer>)
                    prepared.lease.matterApplyOutcomes
                           offset:0u atIndex:3u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterProposedTokenBuffer]
                           offset:0u atIndex:4u];
            [complete setBuffer:state_->buffers[6u] offset:0u atIndex:5u];
            [complete setBuffer:state_->standBuffers[kStandVelocityBuffer]
                           offset:0u atIndex:6u];
            [complete setBuffer:state_->buffers[kMujocoStatesBuffer]
                           offset:0u atIndex:7u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterQCheckpointBuffer]
                           offset:0u atIndex:8u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterVCheckpointBuffer]
                           offset:0u atIndex:9u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterMujocoCheckpointBuffer]
                           offset:0u atIndex:10u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterOwnerStatusBuffer]
                           offset:0u atIndex:11u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterAppliedOutcomeBuffer]
                           offset:0u atIndex:12u];
            [complete setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterFinalAcceptedTokenBuffer]
                           offset:0u atIndex:13u];
            [complete setThreadgroupMemoryLength:
                sizeof(MRNumanXHumanMatterAppliedOutcomeGPU) atIndex:0u];
            [complete setThreadgroupMemoryLength:sizeof(std::uint32_t)
                                         atIndex:1u];
            [complete dispatchThreadgroups:MTLSizeMake(
                dispatch.environmentCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    kStandThreadsPerThreadgroup, 1u, 1u)];
            [complete endEncoding];
            prepared.applyPass = pass;
            prepared.adapterApplyEncoded = true;
            prepared.applyInFlight = true;
            prepared.applyCommandBufferIdentity =
                reinterpret_cast<std::uintptr_t>(request.commandBuffer);
            const std::uint64_t attempt = ++prepared.applyAttempt;
            const std::uint64_t generation = slotGeneration_;
            const auto retainedState = state_;
            const auto retainedCapability = capability_;
            __strong id<MTLSharedEvent> retainedTimeline =
                state_->humanMatterTimelineEvent;
            const std::uint64_t eventValue = prepared.appliedEventValue;
            const auto completionCallback = request.completion;
            void* completionContext = request.completionContext;
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                MetalNumanXHumanMatterReleasePrepareLease release = nullptr;
                void* releaseContext = nullptr;
                MetalNumanXHumanMatterPrepareLease lease{};
                bool validRecord = false;
                bool commandSucceeded = completed.status ==
                    MTLCommandBufferStatusCompleted;
                bool recordSettled = false;
                try {
                    const std::lock_guard completionLock(retainedState->mutex);
                    auto& current = retainedState->humanMatterPrepared;
                    if (!current.active ||
                        current.dispatch.slotGeneration != generation ||
                        current.capability.lock() != retainedCapability ||
                        !current.applyInFlight ||
                        current.applyAttempt != attempt) return;
                    auto* applied = static_cast<
                        MRNumanXHumanMatterAppliedOutcomeGPU*>(
                        retainedState->humanMatterBuffers[
                            kHumanMatterAppliedOutcomeBuffer].contents);
                    void* finalToken = retainedState->humanMatterBuffers[
                        kHumanMatterFinalAcceptedTokenBuffer].contents;
                    const void* immutableToken =
                        retainedState->humanMatterBuffers[
                            kHumanMatterProposedTokenBuffer].contents;
                    if (!commandSucceeded) {
                        MRNumanXHumanMatterAppliedOutcomeGPU failed{};
                        failed.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
                        failed.status =
                            MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH;
                        failed.decision = MR_NUMANX_HUMAN_MATTER_ROOT_PENDING;
                        failed.code =
                            MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER;
                        failed.programFingerprint =
                            current.dispatch.programFingerprint;
                        failed.transactionFingerprint =
                            current.dispatch.transactionFingerprint;
                        failed.linearizationEpoch =
                            current.dispatch.linearizationEpoch;
                        failed.slotGeneration =
                            current.dispatch.slotGeneration;
                        failed.physicsTokenFingerprint =
                            proposalSnapshot.physicsTokenFingerprint;
                        failed.proposalFingerprint =
                            proposalSnapshot.proposalFingerprint;
                        failed.environment = 0u;
                        failed.stepIndex = current.dispatch.stepIndex;
                        failed.substepIndex = current.dispatch.substepIndex;
                        failed.transactionSlot = current.transactionSlot;
                        failed.physicsSubstepCount =
                            current.dispatch.physicsSubstepCount;
                        failed.controlStep = current.dispatch.controlStep;
                        failed.appliedFingerprint =
                            humanMatterRecordFingerprint(&failed);
                        *applied = failed;
                        std::memset(
                            finalToken,
                            0,
                            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
                    } else {
                        const auto* currentProposal = static_cast<const
                            MRNumanXHumanMatterProposalGPU*>(
                            retainedState->humanMatterBuffers[
                                kHumanMatterProposalBuffer].contents);
                        validRecord = currentProposal != nullptr &&
                            validAppliedRecord(
                                *applied,
                                *currentProposal,
                                current.dispatch,
                                immutableToken,
                                finalToken,
                                current.lease.humanIOCandidate);
                    }
                    current.applyInFlight = false;
                    current.applyCommandBufferIdentity = 0u;
                    current.adapterApplyEncoded = false;
                    current.applyComplete = commandSucceeded && validRecord;
                    current.applyFailed = !current.applyComplete;
                    recordSettled = true;
                    if (!current.applyComplete || current.physicalFailure) {
                        current.terminalNoTouch = true;
                    }
                    release = current.releasePrepareLease;
                    releaseContext = current.leaseContext;
                    lease = current.lease;
                } catch (...) {
                    commandSucceeded = false;
                    validRecord = false;
                    auto* applied = static_cast<
                        MRNumanXHumanMatterAppliedOutcomeGPU*>(
                        retainedState->humanMatterBuffers[
                            kHumanMatterAppliedOutcomeBuffer].contents);
                    void* finalToken = retainedState->humanMatterBuffers[
                        kHumanMatterFinalAcceptedTokenBuffer].contents;
                    if (applied != nullptr && finalToken != nullptr) {
                        MRNumanXHumanMatterAppliedOutcomeGPU failed{};
                        failed.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
                        failed.status =
                            MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH;
                        failed.decision = MR_NUMANX_HUMAN_MATTER_ROOT_PENDING;
                        failed.code =
                            MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER;
                        failed.programFingerprint = dispatch.programFingerprint;
                        failed.transactionFingerprint =
                            dispatch.transactionFingerprint;
                        failed.linearizationEpoch = dispatch.linearizationEpoch;
                        failed.slotGeneration = dispatch.slotGeneration;
                        failed.physicsTokenFingerprint =
                            proposalSnapshot.physicsTokenFingerprint;
                        failed.proposalFingerprint =
                            proposalSnapshot.proposalFingerprint;
                        failed.environment = 0u;
                        failed.stepIndex = dispatch.stepIndex;
                        failed.substepIndex = dispatch.substepIndex;
                        failed.transactionSlot = dispatch.transactionSlot;
                        failed.physicsSubstepCount =
                            dispatch.physicsSubstepCount;
                        failed.controlStep = dispatch.controlStep;
                        failed.appliedFingerprint =
                            humanMatterRecordFingerprint(&failed);
                        *applied = failed;
                        std::memset(
                            finalToken,
                            0,
                            MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
                        recordSettled = true;
                    }
                }
                if (recordSettled && retainedTimeline != nil &&
                    retainedTimeline.signaledValue < eventValue) {
                    retainedTimeline.signaledValue = eventValue;
                }
                auto disposition =
                    MetalNumanXHumanMatterPrepareLeaseDisposition::
                        terminalNoTouch;
                if (release != nullptr) {
                    disposition = release(
                        releaseContext,
                        lease,
                        (__bridge void*)completed,
                        commandSucceeded);
                }
                bool reapSubmission = false;
                MetalNumanXHumanMatterApplyTerminalStatus terminalStatus =
                    MetalNumanXHumanMatterApplyTerminalStatus::terminalNoTouch;
                try {
                    const std::lock_guard dispositionLock(retainedState->mutex);
                    auto& current = retainedState->humanMatterPrepared;
                    if (!current.active ||
                        current.dispatch.slotGeneration != generation ||
                        current.capability.lock() != retainedCapability ||
                        current.applyAttempt != attempt) return;
                    const auto* applied = static_cast<const
                        MRNumanXHumanMatterAppliedOutcomeGPU*>(
                        retainedState->humanMatterBuffers[
                            kHumanMatterAppliedOutcomeBuffer].contents);
                    const bool accepted = commandSucceeded && validRecord &&
                        applied != nullptr && applied->status ==
                            MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED;
                    const bool rejected = commandSucceeded && validRecord &&
                        applied != nullptr && applied->status ==
                            MR_NUMANX_HUMAN_MATTER_APPLIED_REJECT_RESTORED;
                    if (accepted && disposition ==
                            MetalNumanXHumanMatterPrepareLeaseDisposition::
                                acceptedPendingPublication) {
                        current.appliedAcceptPendingPublication = true;
                        terminalStatus =
                            MetalNumanXHumanMatterApplyTerminalStatus::
                                acceptedPendingPublication;
                    } else if (rejected && disposition ==
                                   MetalNumanXHumanMatterPrepareLeaseDisposition::
                                       released) {
                        current.active = false;
                        current.bindHumanIOCandidatePublication = nullptr;
                        current.releasePrepareLease = nullptr;
                        current.reservePreparedApplication = nullptr;
                        current.encodePreparedApply = nullptr;
                        current.abortPreparedApply = nullptr;
                        current.releasePublishedRoot = nullptr;
                        current.leaseContext = nullptr;
                        current.preparedTokens = nil;
                        current.capability.reset();
                        reapSubmission = true;
                        terminalStatus =
                            MetalNumanXHumanMatterApplyTerminalStatus::
                                rejectedReleased;
                    } else {
                        current.terminalNoTouch = true;
                    }
                } catch (...) {
                    // The exact generation remains quarantined.
                }
                if (reapSubmission && retainedCapability != nullptr) {
                    if (retainedCapability->submission != nullptr) {
                        (void)retainedCapability->submission->
                            reapTerminalWithoutWaiting();
                    }
                }
                // A released REJECT is reusable from the completion callback.
                // Keep the capability/submission object alive, but settle its
                // terminal nonwaiting ownership before invoking user code.
                completionCallback(
                    completionContext, terminalStatus, generation);
                if (reapSubmission && retainedCapability != nullptr) {
                    retainedCapability->submission.reset();
                }
            }];
        }
        diagnostics.encoded = true;
        return diagnostics;
    } catch (const std::exception& exception) {
        diagnostics.status =
            MetalNumanXHumanMatterOperationStatus::invalidRequest;
        diagnostics.message = exception.what();
        return diagnostics;
    } catch (...) {
        return fail(
            MetalNumanXHumanMatterOperationStatus::invalidRequest,
            "unexpected failure while encoding Human/Matter apply");
    }
}

bool MetalNumanXHumanMatterPrepared::abortApply(
    void* commandBuffer
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u || commandBuffer == nullptr) return false;
    try {
        const std::lock_guard lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        __unsafe_unretained id<MTLCommandBuffer> borrowed =
            (__bridge id<MTLCommandBuffer>)commandBuffer;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_ ||
            !prepared.applyInFlight ||
            prepared.applyCommandBufferIdentity !=
                reinterpret_cast<std::uintptr_t>(commandBuffer) ||
            borrowed == nil ||
            borrowed.status != MTLCommandBufferStatusNotEnqueued) {
            return false;
        }
        if (prepared.adapterApplyEncoded &&
            prepared.abortPreparedApply != nullptr) {
            prepared.abortPreparedApply(
                prepared.leaseContext,
                prepared.lease,
                prepared.applyPass);
        }
        prepared.applyInFlight = false;
        prepared.adapterApplyEncoded = false;
        prepared.applyCommandBufferIdentity = 0u;
        ++prepared.applyAttempt;
        std::memset(
            state_->humanMatterBuffers[kHumanMatterApplyActionBuffer].contents,
            0,
            static_cast<std::size_t>(prepared.applyActionElementCount) *
                sizeof(MRNumanXHumanMatterApplyActionGPU));
        std::memset(
            state_->humanMatterBuffers[kHumanMatterAppliedOutcomeBuffer].contents,
            0,
            static_cast<std::size_t>(prepared.appliedOutcomeElementCount) *
                sizeof(MRNumanXHumanMatterAppliedOutcomeGPU));
        std::memset(
            state_->humanMatterBuffers[
                kHumanMatterFinalAcceptedTokenBuffer].contents,
            0,
            static_cast<std::size_t>(prepared.finalTokenByteCount));
        return true;
    } catch (...) {
        return false;
    }
}

bool MetalNumanXHumanMatterPrepared::reservePublishedRoot(
    const MetalNumanXHumanMatterPublicationReservationRequest& request
) noexcept {
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) return false;
    try {
        std::unique_lock lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_ ||
            prepared.applyInFlight || !prepared.applyComplete ||
            prepared.applyFailed || !prepared.appliedAcceptPendingPublication ||
            prepared.physicalFailure || prepared.terminalNoTouch ||
            prepared.publicationReserved ||
            prepared.publicationReleaseInFlight ||
            prepared.reservePublishedRoot == nullptr ||
            !prepared.lease.humanIOCandidate.valid()) {
            return false;
        }
        if (request.abiVersion != kMetalNumanXHumanMatterABIVersion ||
            request.structSize != sizeof(request) ||
            request.environmentCount != prepared.dispatch.environmentCount ||
            request.transactionSlot != prepared.transactionSlot ||
            request.stepIndex != prepared.dispatch.stepIndex ||
            request.substepIndex != prepared.dispatch.substepIndex ||
            request.physicsSubstepCount !=
                prepared.dispatch.physicsSubstepCount ||
            request.controlStep != prepared.dispatch.controlStep ||
            request.programFingerprint != prepared.dispatch.programFingerprint ||
            request.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            request.linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            request.slotGeneration != prepared.dispatch.slotGeneration ||
            request.jointCommitFingerprint == 0u ||
            request.brainGeneration == 0u) {
            return false;
        }
        id<MTLBuffer> proposalBuffer = state_->humanMatterBuffers[
            kHumanMatterProposalBuffer];
        id<MTLBuffer> proposedTokenBuffer = state_->humanMatterBuffers[
            kHumanMatterProposedTokenBuffer];
        id<MTLBuffer> appliedBuffer = state_->humanMatterBuffers[
            kHumanMatterAppliedOutcomeBuffer];
        id<MTLBuffer> finalTokenBuffer = state_->humanMatterBuffers[
            kHumanMatterFinalAcceptedTokenBuffer];
        id<MTLBuffer> fenceBuffer = state_->humanMatterBuffers[
            kHumanMatterPublicationFenceBuffer];
        if (prepared.proposalElementCount == 0u ||
            prepared.proposalElementCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRNumanXHumanMatterProposalGPU) ||
            prepared.appliedOutcomeElementCount == 0u ||
            prepared.appliedOutcomeElementCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRNumanXHumanMatterAppliedOutcomeGPU) ||
            prepared.publicationFenceElementCount == 0u ||
            prepared.publicationFenceElementCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRNumanXHumanMatterJointPublicationFenceGPU)) {
            prepared.terminalNoTouch = true;
            return false;
        }
        const bool exactLease =
            prepared.lease.proposals == (__bridge void*)proposalBuffer &&
            prepared.lease.proposalsGPUAddress == proposalBuffer.gpuAddress &&
            prepared.lease.proposedPhysicsStateTokens ==
                (__bridge void*)proposedTokenBuffer &&
            prepared.lease.proposedPhysicsStateTokensGPUAddress ==
                proposedTokenBuffer.gpuAddress &&
            prepared.lease.appliedOutcomes == (__bridge void*)appliedBuffer &&
            prepared.lease.appliedOutcomesGPUAddress ==
                appliedBuffer.gpuAddress &&
            prepared.lease.finalAcceptedPhysicsStateTokens ==
                (__bridge void*)finalTokenBuffer &&
            prepared.lease.finalAcceptedPhysicsStateTokensGPUAddress ==
                finalTokenBuffer.gpuAddress &&
            prepared.lease.publicationFences == (__bridge void*)fenceBuffer &&
            prepared.lease.publicationFencesGPUAddress ==
                fenceBuffer.gpuAddress &&
            ownedMetalBuffer(
                state_->device,
                proposalBuffer,
                prepared.proposalElementCount *
                    sizeof(MRNumanXHumanMatterProposalGPU)) &&
            ownedMetalBuffer(
                state_->device,
                proposedTokenBuffer,
                prepared.proposedTokenByteCount) &&
            ownedMetalBuffer(
                state_->device,
                appliedBuffer,
                prepared.appliedOutcomeElementCount *
                    sizeof(MRNumanXHumanMatterAppliedOutcomeGPU)) &&
            ownedMetalBuffer(
                state_->device,
                finalTokenBuffer,
                prepared.finalTokenByteCount) &&
            ownedMetalBuffer(
                state_->device,
                fenceBuffer,
                prepared.publicationFenceElementCount *
                    sizeof(MRNumanXHumanMatterJointPublicationFenceGPU));
        if (!exactLease) {
            prepared.terminalNoTouch = true;
            return false;
        }
        const std::array<id<MTLBuffer>, 5u> buffers{{
            proposalBuffer,
            proposedTokenBuffer,
            appliedBuffer,
            finalTokenBuffer,
            fenceBuffer,
        }};
        std::array<MetalBufferRegion, 5u> regions{};
        for (std::size_t index = 0u; index < buffers.size(); ++index) {
            if (!ownedBufferRegion(buffers[index], regions[index])) return false;
            for (std::size_t prior = 0u; prior < index; ++prior) {
                if (regionsOverlap(regions[index], regions[prior])) return false;
            }
        }
        const auto* proposal = static_cast<const
            MRNumanXHumanMatterProposalGPU*>(proposalBuffer.contents);
        const auto* applied = static_cast<const
            MRNumanXHumanMatterAppliedOutcomeGPU*>(appliedBuffer.contents);
        const void* proposedToken = proposedTokenBuffer.contents;
        const void* finalToken = finalTokenBuffer.contents;
        if (proposal == nullptr || applied == nullptr ||
            !validAppliedRecord(
                *applied,
                *proposal,
                prepared.dispatch,
                proposedToken,
                finalToken,
                prepared.lease.humanIOCandidate) ||
            applied->status !=
                MR_NUMANX_HUMAN_MATTER_APPLIED_ACCEPT_QUARANTINED) {
            prepared.terminalNoTouch = true;
            return false;
        }

        MRNumanXHumanMatterJointPublicationFenceGPU fence{};
        fence.abiVersion =
            MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION;
        fence.structBytes = MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES;
        fence.status = MR_NUMANX_HUMAN_MATTER_PUBLICATION_PENDING;
        fence.environment = 0u;
        fence.controlStep = prepared.dispatch.controlStep;
        fence.substepIndex = prepared.dispatch.substepIndex;
        fence.physicsSubstepCount = prepared.dispatch.physicsSubstepCount;
        fence.ownerProgramFingerprint = prepared.dispatch.programFingerprint;
        fence.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        fence.linearizationEpoch = prepared.dispatch.linearizationEpoch;
        fence.slotGeneration = prepared.dispatch.slotGeneration;
        fence.physicsTokenFingerprint = applied->physicsTokenFingerprint;
        fence.brainProgramFingerprint = proposal->brainProgramFingerprint;
        fence.brainShadowStateFingerprint =
            proposal->brainShadowStateFingerprint;
        fence.brainWitnessFingerprint = proposal->brainWitnessFingerprint;
        fence.appliedDecisionFingerprint = applied->appliedFingerprint;
        fence.jointCommitFingerprint = request.jointCommitFingerprint;
        fence.brainGeneration = request.brainGeneration;
        fence.fenceFingerprint = humanMatterRecordFingerprint(&fence);
        *static_cast<MRNumanXHumanMatterJointPublicationFenceGPU*>(
            fenceBuffer.contents) = fence;

        MetalNumanXHumanMatterPublicationReservationView reservation{};
        reservation.proposal.proposals = prepared.lease.proposals;
        reservation.proposal.proposedPhysicsStateTokens =
            prepared.lease.proposedPhysicsStateTokens;
        reservation.proposal.proposalsGPUAddress =
            prepared.lease.proposalsGPUAddress;
        reservation.proposal.proposedPhysicsStateTokensGPUAddress =
            prepared.lease.proposedPhysicsStateTokensGPUAddress;
        reservation.proposal.proposalElementCount =
            prepared.lease.proposalElementCount;
        reservation.proposal.proposedPhysicsStateTokenByteCount =
            prepared.lease.proposedPhysicsStateTokenByteCount;
        reservation.proposal.proposalStride = prepared.lease.proposalStride;
        reservation.proposal.proposedTokenStrideBytes =
            prepared.lease.proposedTokenStrideBytes;
        reservation.proposal.proposalEventValue = prepared.proposalEventValue;
        reservation.fence.publicationFences =
            prepared.lease.publicationFences;
        reservation.fence.publicationFencesGPUAddress =
            prepared.lease.publicationFencesGPUAddress;
        reservation.fence.publicationFenceElementCount =
            prepared.lease.publicationFenceElementCount;
        reservation.fence.publicationFenceStride =
            prepared.lease.publicationFenceStride;
        reservation.fence.environmentCount = prepared.dispatch.environmentCount;
        reservation.fence.transactionSlot = prepared.transactionSlot;
        reservation.fence.stepIndex = prepared.dispatch.stepIndex;
        reservation.fence.substepIndex = prepared.dispatch.substepIndex;
        reservation.fence.physicsSubstepCount =
            prepared.dispatch.physicsSubstepCount;
        reservation.fence.controlStep = prepared.dispatch.controlStep;
        reservation.fence.programFingerprint =
            prepared.dispatch.programFingerprint;
        reservation.fence.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        reservation.fence.linearizationEpoch =
            prepared.dispatch.linearizationEpoch;
        reservation.fence.slotGeneration = prepared.dispatch.slotGeneration;
        reservation.appliedOutcomes = prepared.lease.appliedOutcomes;
        reservation.finalAcceptedPhysicsStateTokens =
            prepared.lease.finalAcceptedPhysicsStateTokens;
        reservation.appliedOutcomesGPUAddress =
            prepared.lease.appliedOutcomesGPUAddress;
        reservation.finalAcceptedPhysicsStateTokensGPUAddress =
            prepared.lease.finalAcceptedPhysicsStateTokensGPUAddress;
        reservation.appliedOutcomeElementCount =
            prepared.lease.appliedOutcomeElementCount;
        reservation.finalAcceptedPhysicsStateTokenByteCount =
            prepared.lease.finalAcceptedPhysicsStateTokenByteCount;
        reservation.appliedOutcomeStride =
            prepared.lease.appliedOutcomeStride;
        reservation.finalTokenStrideBytes =
            prepared.lease.finalTokenStrideBytes;
        reservation.jointCommitFingerprint = request.jointCommitFingerprint;
        reservation.brainGeneration = request.brainGeneration;
        const auto& humanIO = prepared.lease.humanIOCandidate;
        auto& humanIOBinding = reservation.humanIOBinding;
        humanIOBinding.environmentCount = prepared.dispatch.environmentCount;
        humanIOBinding.transactionSlot = prepared.transactionSlot;
        humanIOBinding.stepIndex = prepared.dispatch.stepIndex;
        humanIOBinding.substepIndex = prepared.dispatch.substepIndex;
        humanIOBinding.physicsSubstepCount =
            prepared.dispatch.physicsSubstepCount;
        humanIOBinding.controlStep = prepared.dispatch.controlStep;
        humanIOBinding.ownerProgramFingerprint =
            prepared.dispatch.programFingerprint;
        humanIOBinding.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        humanIOBinding.linearizationEpoch =
            prepared.dispatch.linearizationEpoch;
        humanIOBinding.slotGeneration = prepared.dispatch.slotGeneration;
        humanIOBinding.physicsTokenFingerprint =
            applied->physicsTokenFingerprint;
        humanIOBinding.proposalFingerprint = proposal->proposalFingerprint;
        humanIOBinding.ackFingerprint = applied->ackFingerprint;
        humanIOBinding.appliedDecisionFingerprint =
            applied->appliedFingerprint;
        humanIOBinding.jointCommitFingerprint =
            request.jointCommitFingerprint;
        humanIOBinding.brainGeneration = request.brainGeneration;
        humanIOBinding.candidateKeyFingerprint =
            humanIO.candidateKeyFingerprint;
        humanIOBinding.acceptedBrainGeneration =
            humanIO.acceptedBrainGeneration;
        humanIOBinding.sensorGeneration = humanIO.sensorGeneration;
        humanIOBinding.humanIOProgramFingerprint =
            humanIO.humanIOProgramFingerprint;
        humanIOBinding.sensorFingerprint = humanIO.sensorFingerprint;
        humanIOBinding.transactionInstanceFingerprint =
            humanIO.transactionInstanceFingerprint;
        humanIOBinding.candidatePublicationFingerprint =
            humanIO.candidatePublicationFingerprint;
        humanIOBinding.deviceRegistryID = humanIO.deviceRegistryID;
        humanIOBinding.humanIOIdentityFingerprint =
            humanIO.identityFingerprint;
        humanIOBinding.bindingFingerprint =
            humanIOPublicationBindingFingerprint(humanIOBinding);
        if (humanIOBinding.bindingFingerprint == 0u ||
            proposal->candidatePublicationFingerprint !=
                humanIOBinding.candidatePublicationFingerprint ||
            proposal->humanIOIdentityFingerprint !=
                humanIOBinding.humanIOIdentityFingerprint) {
            prepared.terminalNoTouch = true;
            return false;
        }

        const auto callback = prepared.reservePublishedRoot;
        void* callbackContext = prepared.leaseContext;
        const auto lease = prepared.lease;
        const std::uint64_t generation = prepared.dispatch.slotGeneration;
        prepared.publicationReleaseInFlight = true;
        lock.unlock();
        const bool reserved = callback(callbackContext, lease, reservation);
        lock.lock();
        auto& current = state_->humanMatterPrepared;
        if (!current.active ||
            current.dispatch.slotGeneration != generation ||
            current.capability.lock() != capability_ ||
            !current.publicationReleaseInFlight) {
            return false;
        }
        current.publicationReleaseInFlight = false;
        if (!reserved) {
            std::memset(
                fenceBuffer.contents,
                0,
                sizeof(MRNumanXHumanMatterJointPublicationFenceGPU));
            return false;
        }
        current.publicationReserved = true;
        current.jointCommitFingerprint = request.jointCommitFingerprint;
        current.brainGeneration = request.brainGeneration;
        return true;
    } catch (...) {
        return false;
    }
}

MetalNumanXHumanMatterPrepareLeaseDisposition
MetalNumanXHumanMatterPrepared::releasePublishedRoot(
    const MetalNumanXHumanMatterPublicationReleaseRequest& request
) noexcept {
    constexpr auto terminal =
        MetalNumanXHumanMatterPrepareLeaseDisposition::terminalNoTouch;
    if (state_ == nullptr || capability_ == nullptr ||
        slotGeneration_ == 0u) return terminal;
    try {
        std::unique_lock lock(state_->mutex);
        auto& prepared = state_->humanMatterPrepared;
        if (!prepared.active ||
            prepared.dispatch.slotGeneration != slotGeneration_ ||
            prepared.capability.lock() != capability_) {
            return terminal;
        }
        if (!prepared.applyComplete ||
            !prepared.appliedAcceptPendingPublication ||
            !prepared.publicationReserved ||
            prepared.publicationReleaseInFlight ||
            prepared.terminalNoTouch ||
            prepared.releasePublishedRoot == nullptr ||
            request.abiVersion != kMetalNumanXHumanMatterABIVersion ||
            request.structSize != sizeof(request) ||
            request.publicationFences != prepared.lease.publicationFences ||
            request.publicationFencesGPUAddress !=
                prepared.lease.publicationFencesGPUAddress ||
            request.publicationFenceElementCount !=
                prepared.lease.publicationFenceElementCount ||
            request.publicationFenceStride !=
                prepared.lease.publicationFenceStride ||
            request.environmentCount != prepared.dispatch.environmentCount ||
            request.transactionSlot != prepared.transactionSlot ||
            request.stepIndex != prepared.dispatch.stepIndex ||
            request.substepIndex != prepared.dispatch.substepIndex ||
            request.physicsSubstepCount !=
                prepared.dispatch.physicsSubstepCount ||
            request.controlStep != prepared.dispatch.controlStep ||
            request.programFingerprint != prepared.dispatch.programFingerprint ||
            request.transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            request.linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            request.slotGeneration != prepared.dispatch.slotGeneration ||
            request.jointCommitFingerprint !=
                prepared.jointCommitFingerprint ||
            request.brainGeneration != prepared.brainGeneration ||
            request.publicationFenceElementCount == 0u ||
            request.publicationFenceElementCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRNumanXHumanMatterJointPublicationFenceGPU) ||
            !exactMetalBuffer(
                state_->device,
                request.publicationFences,
                request.publicationFencesGPUAddress,
                request.publicationFenceElementCount *
                    sizeof(MRNumanXHumanMatterJointPublicationFenceGPU))) {
            prepared.terminalNoTouch = true;
            return terminal;
        }
        const auto* proposal = static_cast<const
            MRNumanXHumanMatterProposalGPU*>(
            state_->humanMatterBuffers[kHumanMatterProposalBuffer].contents);
        const auto* applied = static_cast<const
            MRNumanXHumanMatterAppliedOutcomeGPU*>(
            state_->humanMatterBuffers[kHumanMatterAppliedOutcomeBuffer].contents);
        const auto* fence = static_cast<const
            MRNumanXHumanMatterJointPublicationFenceGPU*>(
            state_->humanMatterBuffers[
                kHumanMatterPublicationFenceBuffer].contents);
        const void* proposedToken = state_->humanMatterBuffers[
            kHumanMatterProposedTokenBuffer].contents;
        const void* finalToken = state_->humanMatterBuffers[
            kHumanMatterFinalAcceptedTokenBuffer].contents;
        const auto* acceptedToken = static_cast<const
            MRNumanXAcceptedPhysicsStateTokenGPU*>(finalToken);
        const bool fenceValid = proposal != nullptr && applied != nullptr &&
            fence != nullptr && acceptedToken != nullptr &&
            acceptedToken->transactionFingerprint ==
                prepared.dispatch.transactionFingerprint &&
            acceptedToken->physicsGeneration != 0u &&
            acceptedToken->environmentIdentifier == 0u &&
            acceptedToken->flags == 0u &&
            acceptedToken->reserved == 0u &&
            acceptedToken->tokenFingerprint != 0u &&
            validAppliedRecord(
                *applied,
                *proposal,
                prepared.dispatch,
                proposedToken,
                finalToken,
                prepared.lease.humanIOCandidate) &&
            fence->abiVersion ==
                MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION &&
            fence->structBytes ==
                MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_BYTES &&
            fence->status == MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED &&
            fence->environment == 0u &&
            fence->controlStep == prepared.dispatch.controlStep &&
            fence->substepIndex == prepared.dispatch.substepIndex &&
            fence->physicsSubstepCount ==
                prepared.dispatch.physicsSubstepCount &&
            fence->reserved0 == 0u &&
            fence->ownerProgramFingerprint ==
                prepared.dispatch.programFingerprint &&
            fence->transactionFingerprint ==
                prepared.dispatch.transactionFingerprint &&
            fence->linearizationEpoch ==
                prepared.dispatch.linearizationEpoch &&
            fence->slotGeneration == prepared.dispatch.slotGeneration &&
            fence->physicsTokenFingerprint ==
                applied->physicsTokenFingerprint &&
            fence->brainProgramFingerprint ==
                proposal->brainProgramFingerprint &&
            fence->brainShadowStateFingerprint ==
                proposal->brainShadowStateFingerprint &&
            fence->brainWitnessFingerprint ==
                proposal->brainWitnessFingerprint &&
            fence->appliedDecisionFingerprint == applied->appliedFingerprint &&
            fence->jointCommitFingerprint ==
                request.jointCommitFingerprint &&
            fence->brainGeneration == request.brainGeneration &&
            fence->fenceFingerprint != 0u &&
            fence->fenceFingerprint == humanMatterRecordFingerprint(fence);
        if (!fenceValid) {
            prepared.terminalNoTouch = true;
            return terminal;
        }
        auto* ownerStatuses = static_cast<
            MRNumanXHumanMatterOwnerStatusGPU*>(
            state_->humanMatterBuffers[
                kHumanMatterOwnerStatusBuffer].contents);
        if (ownerStatuses == nullptr ||
            ownerStatuses[0u].abiVersion !=
                MR_NUMANX_HUMAN_MATTER_ABI_VERSION ||
            ownerStatuses[0u].stage !=
                MR_NUMANX_HUMAN_MATTER_STAGE_APPLIED_ACCEPT_QUARANTINED ||
            ownerStatuses[0u].qCoordinateCount != prepared.dispatch.nq ||
            ownerStatuses[0u].dofCount != prepared.dispatch.nv ||
            ownerStatuses[0u].programFingerprint !=
                prepared.dispatch.programFingerprint ||
            ownerStatuses[0u].transactionFingerprint !=
                prepared.dispatch.transactionFingerprint ||
            ownerStatuses[0u].linearizationEpoch !=
                prepared.dispatch.linearizationEpoch ||
            ownerStatuses[0u].slotGeneration !=
                prepared.dispatch.slotGeneration) {
            prepared.terminalNoTouch = true;
            return terminal;
        }

        MetalNumanXHumanMatterPublicationFenceView fenceView{};
        fenceView.publicationFences = prepared.lease.publicationFences;
        fenceView.publicationFencesGPUAddress =
            prepared.lease.publicationFencesGPUAddress;
        fenceView.publicationFenceElementCount =
            prepared.lease.publicationFenceElementCount;
        fenceView.publicationFenceStride =
            prepared.lease.publicationFenceStride;
        fenceView.environmentCount = prepared.dispatch.environmentCount;
        fenceView.transactionSlot = prepared.transactionSlot;
        fenceView.stepIndex = prepared.dispatch.stepIndex;
        fenceView.substepIndex = prepared.dispatch.substepIndex;
        fenceView.physicsSubstepCount =
            prepared.dispatch.physicsSubstepCount;
        fenceView.controlStep = prepared.dispatch.controlStep;
        fenceView.programFingerprint = prepared.dispatch.programFingerprint;
        fenceView.transactionFingerprint =
            prepared.dispatch.transactionFingerprint;
        fenceView.linearizationEpoch = prepared.dispatch.linearizationEpoch;
        fenceView.slotGeneration = prepared.dispatch.slotGeneration;
        const auto callback = prepared.releasePublishedRoot;
        void* callbackContext = prepared.leaseContext;
        const auto lease = prepared.lease;
        const std::uint64_t generation = prepared.dispatch.slotGeneration;
        prepared.publicationReleaseInFlight = true;
        lock.unlock();
        const auto disposition = callback(
            callbackContext, lease, fenceView);
        lock.lock();
        auto& current = state_->humanMatterPrepared;
        if (!current.active ||
            current.dispatch.slotGeneration != generation ||
            current.capability.lock() != capability_ ||
            !current.publicationReleaseInFlight) {
            return terminal;
        }
        current.publicationReleaseInFlight = false;
        if (disposition !=
            MetalNumanXHumanMatterPrepareLeaseDisposition::released) {
            current.terminalNoTouch = true;
            return terminal;
        }
        state_->publishedResident = {
            .active = true,
            .model = current.residentModel,
            .transactionFingerprint =
                acceptedToken->transactionFingerprint,
            .physicsGeneration = acceptedToken->physicsGeneration,
            .acceptedTokenFingerprint =
                acceptedToken->tokenFingerprint,
            .qBytes = current.residentQBytes,
            .velocityBytes = current.residentVelocityBytes,
            .mujocoStateBytes = current.residentMujocoStateBytes,
        };
        auto& owner = ownerStatuses[0u];
        owner.stage = MR_NUMANX_HUMAN_MATTER_STAGE_ROOT_PUBLISHED;
        owner.code = MR_NUMANX_HUMAN_MATTER_OWNER_SUCCESS;
        owner.restored = 0u;
        owner.preparedTokenPreserved = 1u;
        owner.appliedOutcomePublished = 1u;
        current.active = false;
        current.bindHumanIOCandidatePublication = nullptr;
        current.releasePrepareLease = nullptr;
        current.reservePreparedApplication = nullptr;
        current.encodePreparedApply = nullptr;
        current.abortPreparedApply = nullptr;
        current.reservePublishedRoot = nullptr;
        current.releasePublishedRoot = nullptr;
        current.leaseContext = nullptr;
        current.preparedTokens = nil;
        current.capability.reset();
        auto retainedCapability = capability_;
        lock.unlock();
        if (retainedCapability != nullptr) {
            if (retainedCapability->submission != nullptr) {
                (void)retainedCapability->submission->
                    reapTerminalWithoutWaiting();
            }
            retainedCapability->submission.reset();
        }
        return disposition;
    } catch (...) {
        return terminal;
    }
}

MetalArticulatedOperatorDiagnostics
MetalArticulatedOperatorSubmission::wait(
    MetalArticulatedOperatorResult& result
) {
    if (state_ == nullptr) {
        return reject(
            {},
            MetalArticulatedOperatorHostStatus::
                metalCommandFailure,
            "submission is empty or has already been consumed"
        );
    }
    if (state_->hasPreparedHumanMatter) {
        return reject(
            state_->diagnostics,
            MetalArticulatedOperatorHostStatus::externalProgramFailure,
            "NumanX Human/Matter ABI4 submissions must be moved into extractPreparedHumanMatter before asynchronous root close"
        );
    }

    std::unique_ptr<
        detail::MetalArticulatedOperatorSubmissionState
    > pending = std::move(state_);
    MetalArticulatedOperatorDiagnostics diagnostics =
        pending->diagnostics;
    try {
        MetalArticulatedOperatorResult staged{};
        @autoreleasepool {
            [pending->commandBuffer waitUntilCompleted];
            const auto end =
                std::chrono::steady_clock::now();
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - pending->start
                ).count();
            if (pending->commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "Metal articulated command failed: " +
                        describeError(
                            pending->commandBuffer.error
                        )
                );
            }

            const MetalArticulatedOperatorLayout& layout =
                diagnostics.layout;
            staged.layout = layout;
            staged.bodyPoses.resize(layout.bodyPoseElements);
            staged.pointWorld.resize(layout.pointWorldElements);
            staged.diagnosticMassMatrix.resize(
                layout.massMatrixElements
            );
            staged.pointJacobians.resize(
                layout.pointJacobianElements
            );
            staged.generalizedImpulse.resize(
                layout.generalizedElements
            );
            staged.deltaVelocity.resize(
                layout.generalizedElements
            );
            staged.statuses.resize(layout.statusElements);
            staged.millardResults.resize(layout.millardResultElements);
            staged.millardGeneralizedForces.resize(
                layout.millardGeneralizedForceElements
            );
            staged.mujocoResults.resize(layout.mujocoResultElements);
            staged.mujocoActivationStates.resize(
                layout.mujocoStateElements
            );
            staged.mujocoMuscleGeneralizedForces.resize(
                layout.mujocoMuscleGeneralizedForceElements
            );
            staged.mujocoGeneralizedForces.resize(
                layout.mujocoGeneralizedForceElements
            );
            if (pending->hasStandHorizon) {
                staged.standQ.resize(layout.qElements);
                staged.standV.resize(layout.standVelocityElements);
                staged.standStatuses.resize(layout.standStatusElements);
                staged.standTendonTransfers.resize(
                    layout.standTendonTransferElements
                );
                staged.standTendonGeneralizedCorrections.resize(
                    layout.standTendonCorrectionElements
                );
            }

            const auto& buffers = pending->context->buffers;
            copyOutput(staged.bodyPoses, buffers[8]);
            copyOutput(staged.pointWorld, buffers[9]);
            copyOutput(
                staged.diagnosticMassMatrix,
                buffers[10]
            );
            copyOutput(staged.pointJacobians, buffers[11]);
            copyOutput(
                staged.generalizedImpulse,
                buffers[12]
            );
            copyOutput(staged.deltaVelocity, buffers[13]);
            copyOutput(staged.statuses, buffers[14]);
            copyOutput(
                staged.millardResults,
                buffers[kMillardResultsBuffer]
            );
            copyOutput(
                staged.millardGeneralizedForces,
                buffers[kMillardForcesBuffer]
            );
            copyOutput(
                staged.mujocoResults,
                buffers[kMujocoResultsBuffer]
            );
            copyOutput(
                staged.mujocoActivationStates,
                buffers[kMujocoStatesBuffer]
            );
            copyOutput(
                staged.mujocoMuscleGeneralizedForces,
                buffers[kMillardForcesBuffer]
            );
            if (!staged.mujocoGeneralizedForces.empty()) {
                const auto* source = static_cast<const float*>(
                    buffers[kMillardForcesBuffer].contents
                ) + layout.mujocoMuscleGeneralizedForceElements;
                std::copy_n(
                    source,
                    staged.mujocoGeneralizedForces.size(),
                    staged.mujocoGeneralizedForces.begin()
                );
            }
            if (pending->hasStandHorizon) {
                copyOutput(staged.standQ, buffers[6u]);
                copyOutput(
                    staged.standV,
                    pending->context->standBuffers[kStandVelocityBuffer]
                );
                copyOutput(
                    staged.standStatuses,
                    pending->context->standBuffers[kStandStatusBuffer]
                );
                copyOutput(
                    staged.standTendonTransfers,
                    pending->context->standBuffers[
                        kStandTendonTransfersBuffer
                    ]
                );
                copyOutput(
                    staged.standTendonGeneralizedCorrections,
                    pending->context->standBuffers[
                        kStandTendonCorrectionsBuffer
                    ]
                );
            }
        }

        for (std::size_t environment = 0u;
             environment < staged.statuses.size();
             ++environment) {
            const MRArticulatedOperatorStatusGPU& status =
                staged.statuses[environment];
            if (status.environment != environment ||
                status.articulationIndex !=
                    pending->articulationIndex ||
                status.code >
                    MR_ARTICULATED_OPERATOR_ACCURACY_FAILED) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        internalFailure,
                    "GPU returned a malformed articulated status record"
                );
            }
            if (status.code ==
                MR_ARTICULATED_OPERATOR_SUCCESS) {
                if (status.bodyCount !=
                        pending->articulation.bodyCount ||
                    status.nq != pending->articulation.nq ||
                    status.nv != pending->articulation.nv ||
                    status.pointCount != pending->pointCount ||
                    !finite(status.diagnostics)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            internalFailure,
                        "GPU success status has invalid dimensions or "
                        "diagnostics"
                    );
                }
                ++diagnostics.successfulEnvironmentCount;
            } else {
                if (diagnostics.failedEnvironmentCount == 0u) {
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(
                            environment
                        );
                    diagnostics.firstGPUStatusCode =
                        status.code;
                }
                ++diagnostics.failedEnvironmentCount;
            }
        }
        if (pending->hasMillardReference) {
            for (std::size_t index = 0u;
                 index < staged.millardResults.size();
                 ++index) {
                const MRMillardMuscleResultGPU& millard =
                    staged.millardResults[index];
                const std::size_t environment =
                    index / pending->millardMuscleCount;
                const std::size_t muscle =
                    index - environment * pending->millardMuscleCount;
                if (millard.status != MR_MILLARD_REFERENCE_SUCCESS ||
                    millard.environment != environment ||
                    millard.muscleIndex != muscle ||
                    !finite(millard.pathFiberTendonResidual)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected a Millard source-reference muscle"
                    );
                }
            }
        }
        if (pending->hasMujocoReference) {
            for (std::size_t index = 0u;
                 index < staged.mujocoResults.size();
                 ++index) {
                const MRMujocoMuscleResultGPU& mujoco =
                    staged.mujocoResults[index];
                const std::size_t environment =
                    index / pending->mujocoMuscleCount;
                const std::size_t muscle =
                    index - environment * pending->mujocoMuscleCount;
                if (mujoco.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
                    mujoco.environment != environment ||
                    mujoco.muscleIndex != muscle ||
                    !finite(mujoco)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected a MyoSim source-reference muscle"
                    );
                }
            }
        }
        if (pending->hasStandHorizon) {
            for (std::size_t environment = 0u;
                 environment < staged.standStatuses.size(); ++environment) {
                const MRNumiHumanStandStatusGPU& stand =
                    staged.standStatuses[environment];
                if (stand.environment != environment ||
                    stand.code > MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED ||
                    (stand.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                     stand.completedSteps != pending->standStepCount)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::internalFailure,
                        "GPU returned a malformed Numi Human stand status"
                    );
                }
                diagnostics.completedStandSteps = std::min(
                    diagnostics.completedStandSteps == 0u
                        ? stand.completedSteps
                        : diagnostics.completedStandSteps,
                    stand.completedSteps
                );
                if (stand.code != MR_NUMI_HUMAN_STAND_SUCCESS) {
                    diagnostics.firstStandGPUStatusCode = stand.code;
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected the Numi Human stand horizon: code=" +
                            std::to_string(stand.code) +
                            " completed_steps=" +
                            std::to_string(stand.completedSteps) +
                            " failing_index=" +
                            std::to_string(stand.failingIndex)
                    );
                }
                if (!finite(stand.contactAndAcceleration) ||
                    !finite(stand.factorAndAssistance) ||
                    !finite(stand.tendonDiagnostics) ||
                    !finite(stand.jointEqualityDiagnostics)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::internalFailure,
                        "GPU Numi Human stand diagnostics are non-finite"
                    );
                }
                if (stand.jointEqualityCounts.x !=
                        pending->standJointEqualityCount ||
                    stand.jointEqualityCounts.y !=
                        pending->standJointEqualityCount ||
                    stand.jointEqualityCounts.z != 0u ||
                    stand.jointEqualityCounts.w != 0u) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::internalFailure,
                        "GPU Numi Human joint-equality accounting is malformed"
                    );
                }
                const std::size_t expectedTransfers =
                    pending->standTendonBindingCount *
                    pending->standStepCount;
                const std::size_t expectedEnvelopeTransfers =
                    pending->standTendonEnvelopeBindingCount *
                    pending->standStepCount;
                if (stand.tendonTransferCount != expectedTransfers ||
                    stand.tendonEnvelopeTransferCount !=
                        expectedEnvelopeTransfers ||
                    stand.tendonPointTransferCount !=
                        expectedTransfers - expectedEnvelopeTransfers ||
                    stand.tendonFailureCount != 0u) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::internalFailure,
                        "GPU Numi Human stand tendon transaction counts disagree"
                    );
                }
            }
            for (std::size_t index = 0u;
                 index < staged.standTendonTransfers.size(); ++index) {
                const MRNumiHumanTendonTransferResultGPU& transfer =
                    staged.standTendonTransfers[index];
                const std::size_t environment =
                    index / pending->standTendonBindingCount;
                const std::size_t binding =
                    index - environment * pending->standTendonBindingCount;
                if (transfer.status !=
                        MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS ||
                    transfer.environment != environment ||
                    transfer.bindingIndex != binding) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::internalFailure,
                        "GPU Numi Human final tendon transfer is malformed"
                    );
                }
            }
        }
        if (!finitePayload(staged)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    internalFailure,
                "GPU batch contained non-finite typed payload"
            );
        }

        result = std::move(staged);
        diagnostics.published = true;
        if (diagnostics.failedEnvironmentCount != 0u) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    gpuEnvironmentFailure,
                "one or more GPU environments rejected execution"
            );
        }
        diagnostics.status =
            MetalArticulatedOperatorHostStatus::success;
        diagnostics.message.clear();
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalBufferFailure,
            "host allocation failed while publishing Metal results"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedOperatorContext::
    MetalArticulatedOperatorContext(
        MetalArticulatedOperatorConfig config
    )
    : state_(std::make_shared<
          detail::MetalArticulatedOperatorContextState
      >(std::move(config))) {}

MetalArticulatedOperatorContext::
    ~MetalArticulatedOperatorContext() = default;

MetalArticulatedOperatorContext::
    MetalArticulatedOperatorContext(
        MetalArticulatedOperatorContext&& other
    ) noexcept = default;

MetalArticulatedOperatorContext&
MetalArticulatedOperatorContext::operator=(
    MetalArticulatedOperatorContext&& other
) noexcept = default;

MetalArticulatedOperatorDiagnostics
MetalArticulatedOperatorContext::submit(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorSubmission& submission
) {
    MetalArticulatedOperatorDiagnostics diagnostics{};
    if (state_ == nullptr) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                internalFailure,
            "operator context was moved from"
        );
    }
    if (submission.valid()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::contextBusy,
            "submission output already owns an in-flight batch"
        );
    }

    RequiredBuffers requirements{};
    try {
        diagnostics = validateAndBuildLayout(
            model,
            input,
            state_->config,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        const std::lock_guard lock(state_->mutex);
        if (state_->inFlight) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::contextBusy,
                "operator context already has an in-flight batch"
            );
        }

        const auto& continuation = input.residentContinuation;
        if (continuation.configured() && !continuation.valid()) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::invalidDimensions,
                "device-resident continuation identity is partial"
            );
        }
        const bool reusePublishedResidentState =
            state_->publishedResident.active;
        if (reusePublishedResidentState) {
            const auto& resident = state_->publishedResident;
            const bool stateArenaValid =
                continuation.valid() &&
                input.stand.numanXHumanMatterProgram.valid() &&
                resident.model == &model &&
                continuation.previousTransactionFingerprint ==
                    resident.transactionFingerprint &&
                continuation.previousPhysicsGeneration ==
                    resident.physicsGeneration &&
                resident.acceptedTokenFingerprint != 0u &&
                resident.qBytes == requirements.entries[6u].logicalBytes &&
                resident.velocityBytes == requirements.standEntries[
                    kStandVelocityBuffer].logicalBytes &&
                resident.mujocoStateBytes == requirements.entries[
                    kMujocoStatesBuffer].logicalBytes &&
                state_->buffers[6u] != nil &&
                state_->buffers[kMujocoStatesBuffer] != nil &&
                state_->standBuffers[kStandVelocityBuffer] != nil &&
                state_->capacities[6u] >=
                    requirements.entries[6u].allocationBytes &&
                state_->capacities[kMujocoStatesBuffer] >=
                    requirements.entries[
                        kMujocoStatesBuffer].allocationBytes &&
                state_->standCapacities[kStandVelocityBuffer] >=
                    requirements.standEntries[
                        kStandVelocityBuffer].allocationBytes;
            if (!stateArenaValid) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::invalidDimensions,
                    "device-resident continuation does not name the exact "
                    "published Human state"
                );
            }
        } else if (continuation.configured()) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::invalidDimensions,
                "device-resident continuation has no published predecessor"
            );
        }

        @autoreleasepool {
            diagnostics = initializeContext(
                *state_,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }
            if (input.stand.numanXHumanMatterProgram.valid()) {
                diagnostics = initializeHumanMatterPipelines(
                    *state_, std::move(diagnostics)
                );
                if (!diagnostics.succeeded()) {
                    return diagnostics;
                }
            }
            diagnostics = ensureBufferArena(
                *state_,
                requirements,
                std::move(diagnostics)
            );
            if (!diagnostics.succeeded()) {
                return diagnostics;
            }

            std::string humanMatterArenaReason;
            if (!validHumanMatterArena(
                    *state_, input, diagnostics.layout,
                    humanMatterArenaReason
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::invalidDimensions,
                    std::move(humanMatterArenaReason)
                );
            }

            uploadBatch(
                *state_,
                model,
                input,
                diagnostics.layout,
                requirements,
                reusePublishedResidentState
            );

            id<MTLCommandBuffer> commandBuffer =
                [state_->queue commandBuffer];
            if (commandBuffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo persistent articulated operator";
            diagnostics.numanXProgramFingerprint =
                input.stand.numanXTransactionProgram.valid()
                    ? input.stand.numanXTransactionProgram.fingerprint
                    : 0u;
            diagnostics.commandBufferIdentity =
                reinterpret_cast<std::uintptr_t>(
                    (__bridge void*)commandBuffer
                );
            TendonLoadAbortGuard tendonLoadAbort{
                input.stand.tendonLoadProgram.valid()
                    ? &input.stand.tendonLoadProgram
                    : nullptr,
                (__bridge void*)commandBuffer,
                false,
            };
            NumanXTransactionAbortGuard numanXTransactionAbort{
                input.stand.numanXTransactionProgram.valid()
                    ? input.stand.numanXTransactionProgram.context
                    : nullptr,
                input.stand.numanXTransactionProgram.valid()
                    ? input.stand.numanXTransactionProgram.abort
                    : nullptr,
                (__bridge void*)commandBuffer,
                false,
            };
            NumanXHumanMatterAbortGuard humanMatterAbort{
                input.stand.numanXHumanMatterProgram.valid()
                    ? input.stand.numanXHumanMatterProgram.context
                    : nullptr,
                input.stand.numanXHumanMatterProgram.valid()
                    ? input.stand.numanXHumanMatterProgram.abort
                    : nullptr,
                (__bridge void*)commandBuffer,
                false,
            };
            std::uint64_t humanMatterPreparedEventValue = 0u;
            std::uint64_t humanMatterProposalEventValue = 0u;
            std::uint64_t humanMatterAppliedEventValue = 0u;
            NumanXHumanMatterPrepareLeaseGuard humanMatterLease{};
            if (input.stand.numanXHumanMatterProgram.valid()) {
                if (state_->humanMatterNextEventValue >
                    std::numeric_limits<std::uint64_t>::max() - 3u) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::arithmeticOverflow,
                        "NumanX Human/Matter shared-event value exhausted"
                    );
                }
                humanMatterPreparedEventValue =
                    state_->humanMatterNextEventValue + 1u;
                humanMatterProposalEventValue =
                    humanMatterPreparedEventValue + 1u;
                humanMatterAppliedEventValue =
                    humanMatterProposalEventValue + 1u;
                const MetalNumanXHumanMatterProgram& program =
                    input.stand.numanXHumanMatterProgram;
                MetalNumanXHumanMatterPrepareLease lease{};
                lease.environmentCount = static_cast<std::uint32_t>(
                    input.environmentCount);
                lease.transactionSlot = program.transactionSlot;
                lease.stepIndex = 0u;
                lease.substepIndex = program.substepIndex;
                lease.physicsSubstepCount = program.physicsSubstepCount;
                lease.controlStep = program.controlStep;
                lease.qCoordinateCount = program.qCoordinateCount;
                lease.dofCount = program.dofCount;
                lease.dofLayoutVersion = program.dofLayoutVersion;
                lease.preparedTokenStrideBytes =
                    program.acceptedTokenStrideBytes;
                lease.proposalStride = 1u;
                lease.proposedTokenStrideBytes =
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
                lease.applyActionStride = 1u;
                lease.matterApplyOutcomeStride =
                    program.matterApplyOutcomeStride;
                lease.appliedOutcomeStride = 1u;
                lease.finalTokenStrideBytes =
                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES;
                lease.publicationFenceStride = 1u;
                lease.preparedPhysicsStateTokens =
                    program.acceptedPhysicsStateTokens;
                lease.proposals = (__bridge void*)
                    state_->humanMatterBuffers[
                        kHumanMatterProposalBuffer];
                lease.proposedPhysicsStateTokens = (__bridge void*)
                    state_->humanMatterBuffers[
                        kHumanMatterProposedTokenBuffer];
                lease.applyActions = (__bridge void*)
                    state_->humanMatterBuffers[kHumanMatterApplyActionBuffer];
                lease.matterApplyOutcomes = program.matterApplyOutcomes;
                lease.appliedOutcomes = (__bridge void*)
                    state_->humanMatterBuffers[
                        kHumanMatterAppliedOutcomeBuffer];
                lease.finalAcceptedPhysicsStateTokens = (__bridge void*)
                    state_->humanMatterBuffers[
                        kHumanMatterFinalAcceptedTokenBuffer];
                lease.publicationFences = (__bridge void*)
                    state_->humanMatterBuffers[
                        kHumanMatterPublicationFenceBuffer];
                lease.physicalPreparedEvent =
                    (__bridge void*)state_->humanMatterTimelineEvent;
                lease.preparedPhysicsStateTokensGPUAddress =
                    program.acceptedPhysicsStateTokensGPUAddress;
                lease.proposalsGPUAddress =
                    state_->humanMatterBuffers[
                        kHumanMatterProposalBuffer].gpuAddress;
                lease.proposedPhysicsStateTokensGPUAddress =
                    state_->humanMatterBuffers[
                        kHumanMatterProposedTokenBuffer].gpuAddress;
                lease.applyActionsGPUAddress = state_->humanMatterBuffers[
                    kHumanMatterApplyActionBuffer].gpuAddress;
                lease.matterApplyOutcomesGPUAddress =
                    program.matterApplyOutcomesGPUAddress;
                lease.appliedOutcomesGPUAddress =
                    state_->humanMatterBuffers[
                        kHumanMatterAppliedOutcomeBuffer].gpuAddress;
                lease.finalAcceptedPhysicsStateTokensGPUAddress =
                    state_->humanMatterBuffers[
                        kHumanMatterFinalAcceptedTokenBuffer].gpuAddress;
                lease.publicationFencesGPUAddress =
                    state_->humanMatterBuffers[
                        kHumanMatterPublicationFenceBuffer].gpuAddress;
                lease.preparedPhysicsStateTokenByteCount =
                    program.acceptedPhysicsStateTokenByteCount;
                lease.proposalElementCount =
                    diagnostics.layout.humanMatterProposalElements;
                lease.proposedPhysicsStateTokenByteCount =
                    diagnostics.layout.humanMatterProposedTokenBytes;
                lease.applyActionElementCount =
                    diagnostics.layout.humanMatterApplyActionElements;
                lease.matterApplyOutcomeElementCount =
                    program.matterApplyOutcomeElementCount;
                lease.appliedOutcomeElementCount =
                    diagnostics.layout.humanMatterAppliedOutcomeElements;
                lease.finalAcceptedPhysicsStateTokenByteCount =
                    diagnostics.layout.humanMatterFinalAcceptedTokenBytes;
                lease.publicationFenceElementCount =
                    diagnostics.layout.humanMatterPublicationFenceElements;
                lease.physicalPreparedEventValue =
                    humanMatterPreparedEventValue;
                lease.proposalEventValue = humanMatterProposalEventValue;
                lease.appliedEventValue = humanMatterAppliedEventValue;
                lease.programFingerprint = program.fingerprint;
                lease.transactionFingerprint =
                    program.transactionFingerprint;
                lease.linearizationEpoch = program.linearizationEpoch;
                lease.slotGeneration = program.slotGeneration;
                if (!program.acquirePrepareLease(
                        program.context, lease)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::externalProgramFailure,
                        "NumanX Human/Matter adapter refused the prepared-slot lease"
                    );
                }
                humanMatterLease.context = program.context;
                humanMatterLease.release = program.releasePrepareLease;
                humanMatterLease.lease = lease;
                humanMatterLease.armed = true;
            }
            const std::uint32_t horizonStepCount = input.stand.enabled()
                ? input.stand.stepCount
                : 1u;
            const MRArticulationGPU& articulation =
                model.articulations[input.articulationIndex];
            NumanXHumanMatterCandidateEncoderContext
                humanMatterCandidateContext{};
            if (input.stand.numanXHumanMatterProgram.valid() &&
                (input.stand.numanXHumanMatterProgram.capabilities &
                 MetalNumanXHumanMatterExactCandidateKinematics) != 0u) {
                const MetalNumanXHumanMatterProgram& program =
                    input.stand.numanXHumanMatterProgram;
                humanMatterCandidateContext.state = state_.get();
                humanMatterCandidateContext.commandBuffer =
                    (__bridge void*)commandBuffer;
                humanMatterCandidateContext.matterReaction =
                    program.matterGeneralizedReaction;
                humanMatterCandidateContext.jointStatuses =
                    program.jointStatuses;
                humanMatterCandidateContext.acceptedTokens =
                    program.acceptedPhysicsStateTokens;
                humanMatterCandidateContext.matterReactionGPUAddress =
                    program.matterGeneralizedReactionGPUAddress;
                humanMatterCandidateContext.jointStatusesGPUAddress =
                    program.jointStatusesGPUAddress;
                humanMatterCandidateContext.acceptedTokensGPUAddress =
                    program.acceptedPhysicsStateTokensGPUAddress;
                humanMatterCandidateContext.matterReactionBytes =
                    program.matterGeneralizedReactionElementCount *
                    sizeof(float);
                humanMatterCandidateContext.jointStatusBytes =
                    program.jointStatusElementCount *
                    sizeof(MRNumanXCoupledHumanStatusGPU);
                humanMatterCandidateContext.acceptedTokenBytes =
                    program.acceptedPhysicsStateTokenByteCount;
                humanMatterCandidateContext.environmentCount =
                    static_cast<std::uint32_t>(input.environmentCount);
                humanMatterCandidateContext.articulationIndex =
                    input.articulationIndex;
                humanMatterCandidateContext.nq = articulation.nq;
                humanMatterCandidateContext.nv = articulation.nv;
                humanMatterCandidateContext.bodyCount =
                    articulation.bodyCount;
                humanMatterCandidateContext.articulationFirstBody =
                    articulation.firstBody;
                humanMatterCandidateContext.sourcePointStride =
                    diagnostics.layout.dispatch.pointStride;
                humanMatterCandidateContext.sourceBodyProbeOffset =
                    input.mujoco.bodyJacobianPointOffset;
                humanMatterCandidateContext.combinedPointStride =
                    4u * articulation.bodyCount +
                    program.candidatePointCapacity;
                humanMatterCandidateContext.candidatePointCapacity =
                    program.candidatePointCapacity;
                humanMatterCandidateContext.substepIndex =
                    program.substepIndex;
                humanMatterCandidateContext.transactionSlot =
                    program.transactionSlot;
                humanMatterCandidateContext.physicsSubstepCount =
                    program.physicsSubstepCount;
                humanMatterCandidateContext.timestepSeconds =
                    state_->config.mujocoActivationTimestepSeconds;
                humanMatterCandidateContext.controlStep = program.controlStep;
                humanMatterCandidateContext.programFingerprint =
                    program.fingerprint;
                humanMatterCandidateContext.transactionFingerprint =
                    program.transactionFingerprint;
                humanMatterCandidateContext.linearizationEpoch =
                    program.linearizationEpoch;
                humanMatterCandidateContext.slotGeneration =
                    program.slotGeneration;
            }
            const auto encodeNumanXTransactionPhase = [&](
                const MetalNumanXTransactionPhase phase,
                const std::uint32_t stepIndex
            ) -> bool {
                if (!knownNumanXTransactionPhase(phase)) {
                    return false;
                }
                const MetalNumanXTransactionProgram& program =
                    input.stand.numanXTransactionProgram;
                if (!program.valid()) {
                    return true;
                }

                const MetalArticulatedOperatorLayout& layout =
                    diagnostics.layout;
                MetalNumanXTransactionPass pass{};
                pass.abiVersion = kMetalNumanXTransactionABIVersion;
                pass.structSize = sizeof(MetalNumanXTransactionPass);
                pass.accessFlags =
                    MetalNumanXTransactionReadBorrowedState;
                if (phase == MetalNumanXTransactionPhase::beginStep) {
                    pass.accessFlags |=
                        MetalNumanXTransactionWriteMujocoExcitation;
                } else if (
                    phase == MetalNumanXTransactionPhase::postDynamics
                ) {
                    pass.accessFlags |=
                        MetalNumanXTransactionWriteStandFailure;
                }
                pass.reserved0 = 0u;
                pass.commandBuffer = (__bridge void*)commandBuffer;
                pass.q = (__bridge void*)state_->buffers[6u];
                pass.v = (__bridge void*)state_->standBuffers[
                    kStandVelocityBuffer
                ];
                pass.bodyPoses = (__bridge void*)state_->buffers[8u];
                pass.pointWorld = (__bridge void*)state_->buffers[9u];
                pass.pointJacobians = (__bridge void*)state_->buffers[11u];
                pass.mujocoMuscles = (__bridge void*)state_->buffers[
                    kMujocoMusclesBuffer
                ];
                pass.mujocoStates = (__bridge void*)state_->buffers[
                    kMujocoStatesBuffer
                ];
                pass.mujocoSites = (__bridge void*)state_->buffers[
                    kMujocoSitesBuffer
                ];
                pass.mujocoWraps = (__bridge void*)state_->buffers[
                    kMujocoWrapsBuffer
                ];
                pass.mujocoRouteNodes = (__bridge void*)state_->buffers[
                    kMujocoRoutesBuffer
                ];
                pass.mujocoResults = (__bridge void*)state_->buffers[
                    kMujocoResultsBuffer
                ];
                pass.mujocoGeneralizedForceArena =
                    (__bridge void*)state_->buffers[kMillardForcesBuffer];
                pass.tendonBindings = (__bridge void*)state_->standBuffers[
                    kStandTendonBindingsBuffer
                ];
                pass.tendonEnvelopes = (__bridge void*)state_->standBuffers[
                    kStandTendonEnvelopesBuffer
                ];
                pass.tendonTransfers = (__bridge void*)state_->standBuffers[
                    kStandTendonTransfersBuffer
                ];
                pass.tendonGeneralizedCorrections =
                    (__bridge void*)state_->standBuffers[
                        kStandTendonCorrectionsBuffer
                    ];
                pass.standStatuses = (__bridge void*)state_->standBuffers[
                    kStandStatusBuffer
                ];

                pass.phase = phase;
                pass.programFingerprint = program.fingerprint;
                pass.stepIndex = stepIndex;
                pass.stepCount = input.stand.stepCount;
                pass.timestepSeconds =
                    state_->config.mujocoActivationTimestepSeconds;
                pass.articulationFirstBody = articulation.firstBody;
                pass.bodyJacobianPointOffset =
                    input.mujoco.bodyJacobianPointOffset;

                pass.environmentCount = input.environmentCount;
                pass.qCoordinateCount = articulation.nq;
                pass.qElementCount = layout.qElements;
                pass.qStride = layout.dispatch.qStride;
                pass.dofCount = articulation.nv;
                pass.vElementCount = layout.standVelocityElements;
                pass.vStride = articulation.nv;
                pass.bodyCount = articulation.bodyCount;
                pass.bodyPoseElementCount = layout.bodyPoseElements;
                pass.bodyPoseStride = layout.dispatch.bodyPoseStride;
                pass.pointCount = input.pointCount;
                pass.pointWorldElementCount = layout.pointWorldElements;
                pass.pointWorldStride = layout.dispatch.pointWorldStride;
                pass.pointJacobianElementCount =
                    layout.pointJacobianElements;
                pass.pointJacobianStride =
                    layout.dispatch.pointJacobianStride;

                pass.mujocoMuscleCount = layout.mujocoMuscleElements;
                pass.mujocoStateElementCount = layout.mujocoStateElements;
                pass.mujocoStateStride =
                    layout.mujocoStateElements / input.environmentCount;
                pass.mujocoSiteCount = layout.mujocoSiteElements;
                pass.mujocoWrapCount = layout.mujocoWrapElements;
                pass.mujocoRouteNodeCount = layout.mujocoRouteNodeElements;
                pass.mujocoResultElementCount = layout.mujocoResultElements;
                pass.mujocoResultStride =
                    layout.mujocoResultElements / input.environmentCount;
                pass.mujocoMuscleGeneralizedForceElementCount =
                    layout.mujocoMuscleGeneralizedForceElements;
                pass.mujocoMuscleGeneralizedForceRowStride = articulation.nv;
                pass.mujocoMuscleGeneralizedForceEnvironmentStride =
                    layout.mujocoMuscleGeneralizedForceElements /
                    input.environmentCount;
                pass.mujocoGeneralizedForceElementCount =
                    layout.mujocoGeneralizedForceElements;
                pass.mujocoGeneralizedForceOffset =
                    layout.mujocoMuscleGeneralizedForceElements;
                pass.mujocoGeneralizedForceStride = articulation.nv;
                pass.mujocoGeneralizedForceArenaElementCount =
                    layout.mujocoForceWorkspaceElements;

                pass.tendonBindingCount =
                    layout.standTendonBindingElements;
                pass.tendonEnvelopeCount =
                    layout.standTendonEnvelopeElements;
                pass.tendonTransferElementCount =
                    layout.standTendonTransferElements;
                pass.tendonTransferStride =
                    layout.standTendonTransferElements /
                    input.environmentCount;
                pass.tendonCorrectionElementCount =
                    layout.standTendonCorrectionElements;
                pass.tendonCorrectionStride =
                    layout.standTendonCorrectionElements /
                    input.environmentCount;
                pass.standStatusElementCount = layout.standStatusElements;
                pass.standStatusStride =
                    layout.standStatusElements / input.environmentCount;

                if (!numanXTransactionAbort.armed) {
                    numanXTransactionAbort.armed = true;
                }
                return program.encode(program.context, pass);
            };
            const auto makeHumanMatterDispatch = [&]() {
                const MetalArticulatedOperatorLayout& layout =
                    diagnostics.layout;
                const MetalNumanXHumanMatterProgram& program =
                    input.stand.numanXHumanMatterProgram;
                MRNumanXHumanMatterDispatchGPU dispatch{};
                dispatch.abiVersion = MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
                dispatch.environmentCount = static_cast<mr_u32>(
                    input.environmentCount
                );
                dispatch.stepIndex = 0u;
                dispatch.substepIndex = program.substepIndex;
                dispatch.flags =
                    MR_NUMANX_HUMAN_MATTER_HAS_PREPARED_TOKEN;
                dispatch.transactionSlot = program.transactionSlot;
                dispatch.nq = articulation.nq;
                dispatch.nv = articulation.nv;
                dispatch.qStride = layout.dispatch.qStride;
                dispatch.vStride = articulation.nv;
                dispatch.mujocoStateStride = static_cast<mr_u32>(
                    layout.mujocoStateElements / input.environmentCount
                );
                dispatch.mujocoStateCount = dispatch.mujocoStateStride;
                dispatch.generalizedForceStride = articulation.nv;
                dispatch.generalizedForceOffset = static_cast<mr_u32>(
                    layout.mujocoMuscleGeneralizedForceElements
                );
                dispatch.reactionStride = program.reactionStride;
                dispatch.jointStatusStride = program.jointStatusStride;
                dispatch.acceptedTokenStrideBytes =
                    program.acceptedTokenStrideBytes;
                dispatch.ownerStatusStride = 1u;
                dispatch.physicsSubstepCount =
                    program.physicsSubstepCount;
                dispatch.timestepAndInverse = {
                    state_->config.mujocoActivationTimestepSeconds,
                    1.0f /
                        state_->config.mujocoActivationTimestepSeconds,
                    0.0f,
                    0.0f,
                };
                dispatch.controlStep = program.controlStep;
                dispatch.programFingerprint = program.fingerprint;
                dispatch.transactionFingerprint =
                    program.transactionFingerprint;
                dispatch.linearizationEpoch = program.linearizationEpoch;
                dispatch.slotGeneration = program.slotGeneration;
                return dispatch;
            };
            const auto makeHumanMatterPass = [&] (
                const MetalNumanXHumanMatterPhase phase
            ) {
                const MetalArticulatedOperatorLayout& layout =
                    diagnostics.layout;
                const MetalNumanXHumanMatterProgram& program =
                    input.stand.numanXHumanMatterProgram;
                MetalNumanXHumanMatterPass pass{};
                pass.abiVersion = kMetalNumanXHumanMatterABIVersion;
                pass.structSize = sizeof(MetalNumanXHumanMatterPass);
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
                pass.q = (__bridge void*)state_->buffers[6u];
                pass.v = (__bridge void*)state_->standBuffers[
                    kStandVelocityBuffer];
                pass.mujocoStates = (__bridge void*)state_->buffers[
                    kMujocoStatesBuffer];
                pass.mujocoGeneralizedForceArena =
                    (__bridge void*)state_->buffers[kMillardForcesBuffer];
                pass.bodyPoses = (__bridge void*)state_->buffers[8u];
                pass.pointQueries = (__bridge void*)state_->buffers[7u];
                pass.pointWorld = (__bridge void*)state_->buffers[9u];
                pass.pointJacobians = (__bridge void*)state_->buffers[11u];
                pass.standStatuses = (__bridge void*)state_->standBuffers[
                    kStandStatusBuffer];
                pass.qCheckpoint = (__bridge void*)state_->humanMatterBuffers[
                    kHumanMatterQCheckpointBuffer];
                pass.vCheckpoint = (__bridge void*)state_->humanMatterBuffers[
                    kHumanMatterVCheckpointBuffer];
                pass.mujocoStateCheckpoint =
                    (__bridge void*)state_->humanMatterBuffers[
                        kHumanMatterMujocoCheckpointBuffer];
                pass.sourceEffectiveTangentFactor =
                    (__bridge void*)state_->standBuffers[kStandFactorBuffer];
                pass.ownerStatuses =
                    (__bridge void*)state_->humanMatterBuffers[
                        kHumanMatterOwnerStatusBuffer];
                pass.matterGeneralizedReaction =
                    program.matterGeneralizedReaction;
                pass.jointStatuses = program.jointStatuses;
                pass.acceptedPhysicsStateTokens =
                    program.acceptedPhysicsStateTokens;
                if (phase ==
                        MetalNumanXHumanMatterPhase::preDynamics &&
                    (program.capabilities &
                     MetalNumanXHumanMatterExactCandidateKinematics) != 0u) {
                    pass.exactCandidateContext =
                        &humanMatterCandidateContext;
                    pass.encodeExactCandidate =
                        &encodeExactHumanMatterCandidate;
                }

                pass.qGPUAddress = state_->buffers[6u].gpuAddress;
                pass.vGPUAddress = state_->standBuffers[
                    kStandVelocityBuffer].gpuAddress;
                pass.mujocoStatesGPUAddress = state_->buffers[
                    kMujocoStatesBuffer].gpuAddress;
                pass.mujocoGeneralizedForceArenaGPUAddress =
                    state_->buffers[kMillardForcesBuffer].gpuAddress;
                pass.bodyPosesGPUAddress = state_->buffers[8u].gpuAddress;
                pass.pointQueriesGPUAddress = state_->buffers[7u].gpuAddress;
                pass.pointWorldGPUAddress = state_->buffers[9u].gpuAddress;
                pass.pointJacobiansGPUAddress =
                    state_->buffers[11u].gpuAddress;
                pass.standStatusesGPUAddress = state_->standBuffers[
                    kStandStatusBuffer].gpuAddress;
                pass.qCheckpointGPUAddress = state_->humanMatterBuffers[
                    kHumanMatterQCheckpointBuffer].gpuAddress;
                pass.vCheckpointGPUAddress = state_->humanMatterBuffers[
                    kHumanMatterVCheckpointBuffer].gpuAddress;
                pass.mujocoStateCheckpointGPUAddress =
                    state_->humanMatterBuffers[
                        kHumanMatterMujocoCheckpointBuffer].gpuAddress;
                pass.sourceEffectiveTangentFactorGPUAddress =
                    state_->standBuffers[kStandFactorBuffer].gpuAddress;
                pass.ownerStatusesGPUAddress = state_->humanMatterBuffers[
                    kHumanMatterOwnerStatusBuffer].gpuAddress;
                pass.matterGeneralizedReactionGPUAddress =
                    program.matterGeneralizedReactionGPUAddress;
                pass.jointStatusesGPUAddress =
                    program.jointStatusesGPUAddress;
                pass.acceptedPhysicsStateTokensGPUAddress =
                    program.acceptedPhysicsStateTokensGPUAddress;

                pass.environmentCount = input.environmentCount;
                pass.qCoordinateCount = articulation.nq;
                pass.dofCount = articulation.nv;
                pass.bodyCount = articulation.bodyCount;
                pass.pointCount = input.pointCount;
                pass.mujocoStateCount =
                    layout.mujocoStateElements / input.environmentCount;
                pass.qStride = layout.dispatch.qStride;
                pass.vStride = articulation.nv;
                pass.bodyPoseStride = layout.dispatch.bodyPoseStride;
                pass.pointStride = layout.dispatch.pointStride;
                pass.pointWorldStride = layout.dispatch.pointWorldStride;
                pass.pointJacobianStride =
                    layout.dispatch.pointJacobianStride;
                pass.mujocoStateStride = pass.mujocoStateCount;
                pass.factorStride = static_cast<std::uint64_t>(
                    articulation.nv) * articulation.nv;
                pass.generalizedForceOffset =
                    layout.mujocoMuscleGeneralizedForceElements;
                pass.generalizedForceStride = articulation.nv;
                pass.generalizedForceArenaElementCount =
                    layout.mujocoForceWorkspaceElements;
                pass.reactionStride = program.reactionStride;
                pass.jointStatusStride = program.jointStatusStride;
                pass.acceptedTokenStrideBytes =
                    program.acceptedTokenStrideBytes;
                pass.bodyJacobianPointOffset =
                    input.mujoco.bodyJacobianPointOffset;
                pass.timestepSeconds =
                    state_->config.mujocoActivationTimestepSeconds;
                pass.articulationIndex = input.articulationIndex;
                pass.articulationFirstBody = articulation.firstBody;
                pass.programFingerprint = program.fingerprint;
                pass.transactionFingerprint =
                    program.transactionFingerprint;
                pass.linearizationEpoch = program.linearizationEpoch;
                pass.slotGeneration = program.slotGeneration;
                return pass;
            };
            const auto encodeHumanMatterPhase = [&] (
                const MetalNumanXHumanMatterPhase phase
            ) -> bool {
                const MetalNumanXHumanMatterProgram& program =
                    input.stand.numanXHumanMatterProgram;
                if (!program.valid()) return true;
                if (!knownNumanXHumanMatterPhase(phase)) return false;
                std::string reason;
                if (!validHumanMatterArena(
                        *state_, input, diagnostics.layout, reason)) {
                    return false;
                }
                if (!humanMatterAbort.armed) {
                    humanMatterAbort.armed = true;
                }
                const MetalNumanXHumanMatterPass pass =
                    makeHumanMatterPass(phase);
                return program.encode(program.context, pass);
            };
            for (std::uint32_t horizonStep = 0u;
                 horizonStep < horizonStepCount; ++horizonStep) {
            const auto tendonLoadPass = [&]() {
                MetalNumiHumanTendonLoadPass pass{};
                pass.commandBuffer = (__bridge void*)commandBuffer;
                pass.mujocoMuscles = (__bridge void*)state_->buffers[
                    kMujocoMusclesBuffer
                ];
                pass.mujocoSites = (__bridge void*)state_->buffers[
                    kMujocoSitesBuffer
                ];
                pass.mujocoWraps = (__bridge void*)state_->buffers[
                    kMujocoWrapsBuffer
                ];
                pass.mujocoRouteNodes = (__bridge void*)state_->buffers[
                    kMujocoRoutesBuffer
                ];
                pass.mujocoResults = (__bridge void*)state_->buffers[
                    kMujocoResultsBuffer
                ];
                pass.bindings = (__bridge void*)state_->standBuffers[
                    kStandTendonBindingsBuffer
                ];
                pass.envelopes = (__bridge void*)state_->standBuffers[
                    kStandTendonEnvelopesBuffer
                ];
                pass.transfers = (__bridge void*)state_->standBuffers[
                    kStandTendonTransfersBuffer
                ];
                pass.generalizedCorrections =
                    (__bridge void*)state_->standBuffers[
                        kStandTendonCorrectionsBuffer
                    ];
                pass.generalizedForces =
                    (__bridge void*)state_->buffers[kMillardForcesBuffer];
                pass.bodyPoses = (__bridge void*)state_->buffers[8u];
                pass.pointJacobians = (__bridge void*)state_->buffers[11u];
                pass.standStatuses = (__bridge void*)state_->standBuffers[
                    kStandStatusBuffer
                ];
                pass.stepIndex = horizonStep;
                pass.environmentCount = static_cast<std::uint32_t>(
                    input.environmentCount
                );
                pass.endpointCount = static_cast<std::uint32_t>(
                    input.stand.tendonBindings.size()
                );
                pass.envelopeCount = static_cast<std::uint32_t>(
                    input.stand.tendonEnvelopes.size()
                );
                pass.dofCount = articulation.nv;
                pass.muscleCount = static_cast<std::uint32_t>(
                    input.mujoco.muscles.size()
                );
                pass.siteCount = static_cast<std::uint32_t>(
                    input.mujoco.sites.size()
                );
                pass.wrapCount = static_cast<std::uint32_t>(
                    input.mujoco.wraps.size()
                );
                pass.routeNodeCount = static_cast<std::uint32_t>(
                    input.mujoco.routeNodes.size()
                );
                pass.generalizedForceStride = articulation.nv;
                pass.generalizedForceOffset = static_cast<std::uint32_t>(
                    diagnostics.layout.mujocoMuscleGeneralizedForceElements
                );
                pass.pointJacobianStride =
                    diagnostics.layout.dispatch.pointJacobianStride;
                pass.bodyJacobianPointOffset =
                    input.mujoco.bodyJacobianPointOffset;
                pass.bodyPoseStride = articulation.bodyCount;
                pass.articulationFirstBody = articulation.firstBody;
                return pass;
            };
            if (input.stand.numanXHumanMatterProgram.valid()) {
                id<MTLBlitCommandEncoder> checkpoint =
                    [commandBuffer blitCommandEncoder];
                if (checkpoint == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create NumanX Human/Matter checkpoint encoder"
                    );
                }
                checkpoint.label = @"NumanX Human/Matter checkpoint";
                [checkpoint copyFromBuffer:state_->buffers[6u]
                              sourceOffset:0u
                                  toBuffer:state_->humanMatterBuffers[
                                      kHumanMatterQCheckpointBuffer]
                         destinationOffset:0u
                                      size:diagnostics.layout.
                                          humanMatterQCheckpointBytes];
                [checkpoint copyFromBuffer:state_->standBuffers[
                                          kStandVelocityBuffer]
                              sourceOffset:0u
                                  toBuffer:state_->humanMatterBuffers[
                                      kHumanMatterVCheckpointBuffer]
                         destinationOffset:0u
                                      size:diagnostics.layout.
                                          humanMatterVCheckpointBytes];
                [checkpoint copyFromBuffer:state_->buffers[
                                          kMujocoStatesBuffer]
                              sourceOffset:0u
                                  toBuffer:state_->humanMatterBuffers[
                                      kHumanMatterMujocoCheckpointBuffer]
                         destinationOffset:0u
                                      size:diagnostics.layout.
                                          humanMatterMujocoCheckpointBytes];
                [checkpoint endEncoding];

                const MRNumanXHumanMatterDispatchGPU humanMatterDispatch =
                    makeHumanMatterDispatch();
                id<MTLComputeCommandEncoder> begin =
                    [commandBuffer computeCommandEncoder];
                if (begin == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create NumanX Human/Matter begin encoder"
                    );
                }
                begin.label = @"NumanX Human/Matter owner begin";
                [begin setComputePipelineState:
                    state_->humanMatterBeginPipeline];
                [begin setBytes:&humanMatterDispatch
                         length:sizeof(humanMatterDispatch)
                        atIndex:0u];
                [begin setBuffer:(__bridge id<MTLBuffer>)input.stand.
                                     numanXHumanMatterProgram.
                                     acceptedPhysicsStateTokens
                          offset:0u atIndex:1u];
                [begin setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterOwnerStatusBuffer]
                          offset:0u atIndex:2u];
                [begin setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterAppliedOutcomeBuffer]
                          offset:0u atIndex:3u];
                [begin setBuffer:state_->humanMatterBuffers[
                                     kHumanMatterFinalAcceptedTokenBuffer]
                          offset:0u atIndex:4u];
                [begin dispatchThreads:MTLSizeMake(
                                           input.environmentCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(
                        std::min<std::size_t>(
                            input.environmentCount,
                            kThreadsPerThreadgroup),
                        1u, 1u)];
                [begin endEncoding];

                if (!encodeHumanMatterPhase(
                        MetalNumanXHumanMatterPhase::beginStep)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::externalProgramFailure,
                        "NumanX Human/Matter begin phase rejected encoding"
                    );
                }
            }
            if (!encodeNumanXTransactionPhase(
                    MetalNumanXTransactionPhase::beginStep,
                    horizonStep
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::externalProgramFailure,
                    "NumanX begin-step transaction rejected encoding"
                );
            }
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal compute encoder"
                );
            }
            [encoder setComputePipelineState:state_->pipeline];
            for (NSUInteger index = 0u;
                 index < kRawBufferCount;
                 ++index) {
                [encoder
                    setBuffer:state_->buffers[index]
                       offset:0u
                      atIndex:index];
            }
            [encoder
                setThreadgroupMemoryLength:
                    detail::articulatedOperatorThreadgroupBytes(
                        articulation.bodyCount,
                        articulation.nv,
                        !state_->config.pointJacobiansOnly
                    )
                atIndex:0u];
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                    static_cast<NSUInteger>(
                        input.environmentCount
                    ),
                    1u,
                    1u
                )
                threadsPerThreadgroup:MTLSizeMake(
                    kThreadsPerThreadgroup,
                    1u,
                    1u
                )];
            [encoder endEncoding];

            if (input.millard.enabled()) {
                id<MTLComputeCommandEncoder> millardEncoder =
                    [commandBuffer computeCommandEncoder];
                if (millardEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create Millard reference encoder"
                    );
                }
                [millardEncoder setComputePipelineState:state_->millardPipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [millardEncoder
                        setBuffer:state_->buffers[index]
                           offset:0u
                          atIndex:index];
                }
                const std::size_t threadCount =
                    diagnostics.layout.millardResultElements;
                [millardEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (threadCount + kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [millardEncoder endEncoding];
            }

            if (input.mujoco.enabled()) {
                id<MTLComputeCommandEncoder> mujocoEncoder =
                    [commandBuffer computeCommandEncoder];
                if (mujocoEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create MyoSim reference encoder"
                    );
                }
                [mujocoEncoder setComputePipelineState:state_->mujocoPipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [mujocoEncoder
                        setBuffer:state_->buffers[index]
                           offset:0u
                          atIndex:index];
                }
                [mujocoEncoder setBuffer:state_->standBuffers[
                    kStandVelocityBuffer] offset:0u atIndex:7u];
                const std::size_t threadCount =
                    diagnostics.layout.mujocoResultElements;
                [mujocoEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (threadCount + kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [mujocoEncoder endEncoding];
                if (input.stand.enabled()) {
                    MRMujocoMuscleActiveForceDispatchGPU activeDispatch{};
                    activeDispatch.abiVersion =
                        MR_MUJOCO_MUSCLE_ACTIVE_FORCE_GPU_ABI_VERSION;
                    activeDispatch.muscleCount = static_cast<mr_u32>(
                        input.mujoco.muscles.size()
                    );
                    activeDispatch.environmentCount = static_cast<mr_u32>(
                        input.environmentCount
                    );
                    activeDispatch.dofCount = articulation.nv;
                    id<MTLComputeCommandEncoder> activeForceEncoder =
                        [commandBuffer computeCommandEncoder];
                    if (activeForceEncoder == nil) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalCommandFailure,
                            "failed to create MyoSim active-force encoder"
                        );
                    }
                    [activeForceEncoder setComputePipelineState:
                        state_->mujocoActiveForcePipeline];
                    [activeForceEncoder setBuffer:state_->buffers[kMujocoMusclesBuffer]
                                          offset:0u atIndex:0u];
                    [activeForceEncoder setBuffer:state_->buffers[kMujocoStatesBuffer]
                                          offset:0u atIndex:1u];
                    [activeForceEncoder setBuffer:state_->buffers[kMujocoResultsBuffer]
                                          offset:0u atIndex:2u];
                    [activeForceEncoder setBuffer:state_->buffers[kMillardForcesBuffer]
                                          offset:0u atIndex:3u];
                    [activeForceEncoder setBytes:&activeDispatch
                                           length:sizeof(activeDispatch)
                                          atIndex:4u];
                    const std::size_t activeThreadCount =
                        diagnostics.layout.mujocoResultElements;
                    [activeForceEncoder
                        dispatchThreadgroups:MTLSizeMake(
                            static_cast<NSUInteger>(
                                (activeThreadCount + kThreadsPerThreadgroup - 1u) /
                                    kThreadsPerThreadgroup
                            ),
                            1u,
                            1u
                        )
                        threadsPerThreadgroup:MTLSizeMake(
                            kThreadsPerThreadgroup,
                            1u,
                            1u
                        )];
                    [activeForceEncoder endEncoding];
                }
                id<MTLComputeCommandEncoder> mujocoReduceEncoder =
                    [commandBuffer computeCommandEncoder];
                if (mujocoReduceEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create MyoSim force-reduction encoder"
                    );
                }
                [mujocoReduceEncoder
                    setComputePipelineState:state_->mujocoReducePipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [mujocoReduceEncoder
                        setBuffer:state_->buffers[index]
                           offset:0u
                          atIndex:index];
                }
                const std::size_t reducedThreadCount =
                    diagnostics.layout.mujocoGeneralizedForceElements;
                [mujocoReduceEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (reducedThreadCount +
                             kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [mujocoReduceEncoder endEncoding];
                if (!input.stand.tendonBindings.empty()) {
                    MRNumiHumanTendonTransferDispatchGPU tendonDispatch{};
                    tendonDispatch.abiVersion =
                        MR_NUMI_HUMAN_TENDON_TRANSFER_GPU_ABI_VERSION;
                    tendonDispatch.endpointCount = static_cast<mr_u32>(
                        input.stand.tendonBindings.size()
                    );
                    tendonDispatch.envelopeCount = static_cast<mr_u32>(
                        input.stand.tendonEnvelopes.size()
                    );
                    tendonDispatch.muscleCount = static_cast<mr_u32>(
                        input.mujoco.muscles.size()
                    );
                    tendonDispatch.environmentCount = static_cast<mr_u32>(
                        input.environmentCount
                    );
                    tendonDispatch.dofCount = articulation.nv;
                    tendonDispatch.bodyPoseStride = articulation.bodyCount;
                    tendonDispatch.articulationFirstBody =
                        articulation.firstBody;
                    tendonDispatch.pointJacobianStride =
                        diagnostics.layout.dispatch.pointJacobianStride;
                    tendonDispatch.bodyJacobianPointOffset =
                        input.mujoco.bodyJacobianPointOffset;
                    tendonDispatch.bodyJacobianPointStride = 4u;
                    id<MTLComputeCommandEncoder> tendonEncoder =
                        [commandBuffer computeCommandEncoder];
                    if (tendonEncoder == nil) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalCommandFailure,
                            "failed to create Numi Human tendon-transfer encoder"
                        );
                    }
                    [tendonEncoder setComputePipelineState:state_->tendonPipeline];
                    [tendonEncoder setBytes:&tendonDispatch
                                      length:sizeof(tendonDispatch)
                                     atIndex:0u];
                    [tendonEncoder setBuffer:state_->standBuffers[
                        kStandTendonBindingsBuffer] offset:0u atIndex:1u];
                    [tendonEncoder setBuffer:state_->standBuffers[
                        kStandTendonEnvelopesBuffer] offset:0u atIndex:2u];
                    [tendonEncoder setBuffer:state_->buffers[kMujocoResultsBuffer]
                                      offset:0u atIndex:3u];
                    [tendonEncoder setBuffer:state_->buffers[8u]
                                      offset:0u atIndex:4u];
                    [tendonEncoder setBuffer:state_->buffers[11u]
                                      offset:0u atIndex:5u];
                    [tendonEncoder setBuffer:state_->standBuffers[
                        kStandTendonTransfersBuffer] offset:0u atIndex:6u];
                    [tendonEncoder setBuffer:state_->standBuffers[
                        kStandTendonCorrectionsBuffer] offset:0u atIndex:7u];
                    const std::size_t transferThreadCount =
                        diagnostics.layout.standTendonTransferElements;
                    [tendonEncoder
                        dispatchThreadgroups:MTLSizeMake(
                            static_cast<NSUInteger>(
                                (transferThreadCount +
                                 kThreadsPerThreadgroup - 1u) /
                                    kThreadsPerThreadgroup
                            ),
                            1u,
                            1u
                        )
                        threadsPerThreadgroup:MTLSizeMake(
                            kThreadsPerThreadgroup, 1u, 1u
                        )];
                    [tendonEncoder endEncoding];

                    if (input.stand.tendonLoadProgram.valid()) {
                        tendonLoadAbort.armed = true;
                        const MetalNumiHumanTendonLoadPass pass =
                            tendonLoadPass();
                        if (!input.stand.tendonLoadProgram.encodePreDynamics(
                                input.stand.tendonLoadProgram.context,
                                pass
                            )) {
                            return reject(
                                std::move(diagnostics),
                                MetalArticulatedOperatorHostStatus::metalCommandFailure,
                                "Numi Human pre-dynamics tendon-load consumer rejected encoding"
                            );
                        }
                    }
                }
                if (state_->config.mujocoActivationTimestepSeconds > 0.0f) {
                    MRMujocoMuscleActivationDispatchGPU activationDispatch{};
                    activationDispatch.abiVersion =
                        MR_MUJOCO_MUSCLE_ACTIVATION_GPU_ABI_VERSION;
                    activationDispatch.stateCount = static_cast<mr_u32>(
                        diagnostics.layout.mujocoStateElements
                    );
                    activationDispatch.timestepSecondsAndReserved = {
                        state_->config.mujocoActivationTimestepSeconds,
                        0.0f,
                        0.0f,
                        0.0f,
                    };
                    id<MTLComputeCommandEncoder> activationEncoder =
                        [commandBuffer computeCommandEncoder];
                    if (activationEncoder == nil) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalCommandFailure,
                            "failed to create MyoSim activation-step encoder"
                        );
                    }
                    [activationEncoder
                        setComputePipelineState:state_->mujocoActivationPipeline];
                    [activationEncoder setBuffer:state_->buffers[kMujocoStatesBuffer]
                                       offset:0u
                                      atIndex:0u];
                    [activationEncoder setBuffer:state_->buffers[kMujocoResultsBuffer]
                                       offset:0u
                                      atIndex:1u];
                    [activationEncoder setBytes:&activationDispatch
                                          length:sizeof(activationDispatch)
                                         atIndex:2u];
                    const std::size_t activationThreadCount =
                        diagnostics.layout.mujocoStateElements;
                    [activationEncoder
                        dispatchThreadgroups:MTLSizeMake(
                            static_cast<NSUInteger>(
                                (activationThreadCount +
                                 kThreadsPerThreadgroup - 1u) /
                                    kThreadsPerThreadgroup
                            ),
                            1u,
                            1u
                        )
                        threadsPerThreadgroup:MTLSizeMake(
                            kThreadsPerThreadgroup,
                            1u,
                            1u
                        )];
                    [activationEncoder endEncoding];
                }
            }

            if (!encodeNumanXTransactionPhase(
                    MetalNumanXTransactionPhase::preDynamics,
                    horizonStep
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::externalProgramFailure,
                    "NumanX pre-dynamics transaction rejected encoding"
                );
            }

            if (input.stand.numanXHumanMatterProgram.valid()) {
                const MRNumanXHumanMatterDispatchGPU humanMatterDispatch =
                    makeHumanMatterDispatch();
                const MRNumiHumanStandDispatchGPU sourceStandDispatch =
                    makeNumiHumanStandDispatch(
                        input,
                        diagnostics.layout,
                        articulation,
                        state_->config.mujocoActivationTimestepSeconds,
                        horizonStep
                    );
                id<MTLComputeCommandEncoder> factor =
                    [commandBuffer computeCommandEncoder];
                if (factor == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create NumanX Human/Matter source-factor encoder"
                    );
                }
                factor.label =
                    @"NumanX Human/Matter frozen source effective tangent";
                [factor setComputePipelineState:
                    state_->humanMatterFactorPipeline];
                [factor setBuffer:state_->buffers[0u] offset:0u atIndex:0u];
                [factor setBuffer:state_->buffers[1u] offset:0u atIndex:1u];
                [factor setBuffer:state_->buffers[3u] offset:0u atIndex:2u];
                [factor setBuffer:state_->buffers[4u] offset:0u atIndex:3u];
                [factor setBytes:&sourceStandDispatch
                          length:sizeof(sourceStandDispatch)
                         atIndex:4u];
                [factor setBytes:&humanMatterDispatch
                          length:sizeof(humanMatterDispatch)
                         atIndex:5u];
                [factor setBuffer:state_->buffers[8u] offset:0u atIndex:6u];
                [factor setBuffer:state_->buffers[11u] offset:0u atIndex:7u];
                [factor setBuffer:state_->standBuffers[
                                      kStandSpatialJacobianBuffer]
                               offset:0u atIndex:8u];
                [factor setBuffer:state_->standBuffers[kStandFactorBuffer]
                               offset:0u atIndex:9u];
                [factor setBuffer:state_->humanMatterBuffers[
                                      kHumanMatterOwnerStatusBuffer]
                               offset:0u atIndex:10u];
                [factor dispatchThreadgroups:MTLSizeMake(
                                                 input.environmentCount,
                                                 1u, 1u)
                     threadsPerThreadgroup:MTLSizeMake(
                         kStandThreadsPerThreadgroup, 1u, 1u)];
                [factor endEncoding];

                if (!encodeHumanMatterPhase(
                        MetalNumanXHumanMatterPhase::preDynamics)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::externalProgramFailure,
                        "NumanX Human/Matter pre-dynamics phase rejected encoding"
                    );
                }

                id<MTLComputeCommandEncoder> consume =
                    [commandBuffer computeCommandEncoder];
                if (consume == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create NumanX Human/Matter reaction encoder"
                    );
                }
                consume.label =
                    @"NumanX Human/Matter consume reaction once";
                [consume setComputePipelineState:
                    state_->humanMatterConsumePipeline];
                [consume setBytes:&humanMatterDispatch
                           length:sizeof(humanMatterDispatch)
                          atIndex:0u];
                [consume setBuffer:(__bridge id<MTLBuffer>)input.stand.
                                       numanXHumanMatterProgram.
                                       matterGeneralizedReaction
                            offset:0u atIndex:1u];
                [consume setBuffer:(__bridge id<MTLBuffer>)input.stand.
                                       numanXHumanMatterProgram.jointStatuses
                            offset:0u atIndex:2u];
                [consume setBuffer:state_->buffers[kMillardForcesBuffer]
                            offset:0u atIndex:3u];
                [consume setBuffer:state_->humanMatterBuffers[
                                       kHumanMatterOwnerStatusBuffer]
                            offset:0u atIndex:4u];
                [consume setThreadgroupMemoryLength:sizeof(std::uint32_t)
                                            atIndex:0u];
                [consume dispatchThreadgroups:MTLSizeMake(
                                                  input.environmentCount,
                                                  1u, 1u)
                      threadsPerThreadgroup:MTLSizeMake(
                          kStandThreadsPerThreadgroup, 1u, 1u)];
                [consume endEncoding];
            }

            if (input.stand.enabled()) {
                const MRNumiHumanStandDispatchGPU standDispatch =
                    makeNumiHumanStandDispatch(
                        input,
                        diagnostics.layout,
                        articulation,
                        state_->config.mujocoActivationTimestepSeconds,
                        horizonStep
                    );

                id<MTLComputeCommandEncoder> standEncoder =
                    [commandBuffer computeCommandEncoder];
                if (standEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create Numi Human stand encoder"
                    );
                }
                [standEncoder setComputePipelineState:state_->standPipeline];
                [standEncoder setBuffer:state_->buffers[0u] offset:0u atIndex:0u];
                [standEncoder setBuffer:state_->buffers[1u] offset:0u atIndex:1u];
                [standEncoder setBuffer:state_->buffers[3u] offset:0u atIndex:2u];
                [standEncoder setBuffer:state_->buffers[4u] offset:0u atIndex:3u];
                [standEncoder setBytes:&standDispatch
                                 length:sizeof(standDispatch)
                                atIndex:4u];
                [standEncoder setBuffer:state_->buffers[6u] offset:0u atIndex:5u];
                [standEncoder setBuffer:state_->standBuffers[kStandVelocityBuffer]
                                  offset:0u atIndex:6u];
                [standEncoder setBuffer:state_->buffers[8u] offset:0u atIndex:7u];
                [standEncoder setBuffer:state_->buffers[9u] offset:0u atIndex:8u];
                [standEncoder setBuffer:state_->buffers[11u] offset:0u atIndex:9u];
                [standEncoder setBuffer:state_->buffers[kMillardForcesBuffer]
                                  offset:0u atIndex:10u];
                for (NSUInteger index = kStandContactsBuffer;
                     index <= kStandStatusBuffer; ++index) {
                    [standEncoder setBuffer:state_->standBuffers[index]
                                      offset:0u
                                     atIndex:10u + index];
                }
                [standEncoder setBuffer:state_->standBuffers[
                    kStandTendonBindingsBuffer] offset:0u atIndex:18u];
                [standEncoder setBuffer:state_->standBuffers[
                    kStandTendonTransfersBuffer] offset:0u atIndex:19u];
                [standEncoder setBuffer:state_->standBuffers[
                    kStandJointEqualitiesBuffer] offset:0u atIndex:20u];
                [standEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(input.environmentCount),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kStandThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [standEncoder endEncoding];

                if (input.stand.numanXHumanMatterProgram.valid()) {
                    if (!encodeHumanMatterPhase(
                            MetalNumanXHumanMatterPhase::postDynamics)) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::externalProgramFailure,
                            "NumanX Human/Matter post-dynamics phase rejected encoding"
                        );
                    }
                    const MRNumanXHumanMatterDispatchGPU
                        humanMatterDispatch = makeHumanMatterDispatch();
                    id<MTLComputeCommandEncoder> preparePhysical =
                        [commandBuffer computeCommandEncoder];
                    if (preparePhysical == nil) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalCommandFailure,
                            "failed to create NumanX Human/Matter physical-prepare encoder"
                        );
                    }
                    preparePhysical.label =
                        @"NumanX Human/Matter physical prepare or restore";
                    [preparePhysical setComputePipelineState:
                        state_->humanMatterPreparePhysicalPipeline];
                    [preparePhysical setBytes:&humanMatterDispatch
                                length:sizeof(humanMatterDispatch)
                               atIndex:0u];
                    [preparePhysical setBuffer:(__bridge id<MTLBuffer>)input.stand.
                                            numanXHumanMatterProgram.
                                            jointStatuses
                                 offset:0u atIndex:1u];
                    [preparePhysical setBuffer:(__bridge id<MTLBuffer>)input.stand.
                                            numanXHumanMatterProgram.
                                            acceptedPhysicsStateTokens
                                 offset:0u atIndex:2u];
                    [preparePhysical setBuffer:state_->buffers[6u]
                                 offset:0u atIndex:3u];
                    [preparePhysical setBuffer:state_->standBuffers[
                                            kStandVelocityBuffer]
                                 offset:0u atIndex:4u];
                    [preparePhysical setBuffer:state_->buffers[kMujocoStatesBuffer]
                                 offset:0u atIndex:5u];
                    [preparePhysical setBuffer:state_->humanMatterBuffers[
                                            kHumanMatterQCheckpointBuffer]
                                 offset:0u atIndex:6u];
                    [preparePhysical setBuffer:state_->humanMatterBuffers[
                                            kHumanMatterVCheckpointBuffer]
                                 offset:0u atIndex:7u];
                    [preparePhysical setBuffer:state_->humanMatterBuffers[
                                            kHumanMatterMujocoCheckpointBuffer]
                                 offset:0u atIndex:8u];
                    [preparePhysical setBuffer:state_->humanMatterBuffers[
                                            kHumanMatterOwnerStatusBuffer]
                                 offset:0u atIndex:9u];
                    [preparePhysical setThreadgroupMemoryLength:
                                  sizeof(std::uint32_t)
                                                 atIndex:0u];
                    [preparePhysical dispatchThreadgroups:MTLSizeMake(
                                                        input.environmentCount,
                                                        1u, 1u)
                           threadsPerThreadgroup:MTLSizeMake(
                               kStandThreadsPerThreadgroup, 1u, 1u)];
                    [preparePhysical endEncoding];
                }

                if (!encodeNumanXTransactionPhase(
                        MetalNumanXTransactionPhase::postDynamics,
                        horizonStep
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::externalProgramFailure,
                        "NumanX post-dynamics transaction rejected encoding"
                    );
                }

                if (input.stand.tendonLoadProgram.valid()) {
                    const MetalNumiHumanTendonLoadPass pass =
                        tendonLoadPass();
                    tendonLoadAbort.armed = true;
                    if (!input.stand.tendonLoadProgram.encodePostValidation(
                            input.stand.tendonLoadProgram.context,
                            pass
                        )) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalCommandFailure,
                            "Numi Human post-validation tendon-load consumer rejected encoding"
                        );
                    }
                }
            }
            }

            if (input.stand.numanXHumanMatterProgram.valid()) {
                const MRNumanXHumanMatterDispatchGPU completionDispatch =
                    makeHumanMatterDispatch();
                id<MTLComputeCommandEncoder> completionEncoder =
                    [commandBuffer computeCommandEncoder];
                if (completionEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create NumanX Human/Matter physical-completion encoder"
                    );
                }
                completionEncoder.label =
                    @"NumanX Human/Matter physical completion witness";
                [completionEncoder setComputePipelineState:
                    state_->humanMatterMarkPhysicalCompletePipeline];
                [completionEncoder setBytes:&completionDispatch
                                      length:sizeof(completionDispatch)
                                     atIndex:0u];
                [completionEncoder setBuffer:state_->humanMatterBuffers[
                    kHumanMatterOwnerStatusBuffer] offset:0u atIndex:1u];
                [completionEncoder dispatchThreads:MTLSizeMake(
                    input.environmentCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(
                        std::min<std::size_t>(
                            input.environmentCount,
                            kThreadsPerThreadgroup),
                        1u,
                        1u)];
                [completionEncoder endEncoding];
            }

            auto pending = std::make_unique<
                detail::MetalArticulatedOperatorSubmissionState
            >();
            diagnostics.dispatched = true;
            pending->context = state_;
            pending->commandBuffer = commandBuffer;
            pending->diagnostics = diagnostics;
            pending->articulation =
                model.articulations[input.articulationIndex];
            pending->articulationIndex =
                input.articulationIndex;
            pending->pointCount = input.pointCount;
            pending->hasMillardReference = input.millard.enabled();
            pending->millardMuscleCount = input.millard.muscles.size();
            pending->hasMujocoReference = input.mujoco.enabled();
            pending->mujocoMuscleCount = input.mujoco.muscles.size();
            pending->hasStandHorizon = input.stand.enabled();
            pending->standStepCount = input.stand.stepCount;
            pending->standTendonBindingCount =
                input.stand.tendonBindings.size();
            pending->standTendonEnvelopeBindingCount =
                static_cast<std::size_t>(std::count_if(
                    input.stand.tendonBindings.begin(),
                    input.stand.tendonBindings.end(),
                    [](const MRNumiHumanTendonBindingGPU& binding) {
                        return binding.mode ==
                            MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE;
                    }
                ));
            pending->standJointEqualityCount =
                input.stand.jointEqualities.size();
            if (input.stand.numanXHumanMatterProgram.valid()) {
                const MetalNumanXHumanMatterProgram& program =
                    input.stand.numanXHumanMatterProgram;
                auto& prepared = state_->humanMatterPrepared;
                prepared = {};
                prepared.active = true;
                prepared.dispatch = makeHumanMatterDispatch();
                prepared.transactionSlot = program.transactionSlot;
                prepared.preparedTokenByteCount =
                    program.acceptedPhysicsStateTokenByteCount;
                prepared.proposalElementCount =
                    diagnostics.layout.humanMatterProposalElements;
                prepared.proposedTokenByteCount =
                    diagnostics.layout.humanMatterProposedTokenBytes;
                prepared.applyActionElementCount =
                    diagnostics.layout.humanMatterApplyActionElements;
                prepared.appliedOutcomeElementCount =
                    diagnostics.layout.humanMatterAppliedOutcomeElements;
                prepared.finalTokenByteCount =
                    diagnostics.layout.humanMatterFinalAcceptedTokenBytes;
                prepared.publicationFenceElementCount =
                    diagnostics.layout.humanMatterPublicationFenceElements;
                prepared.residentModel = &model;
                prepared.residentQBytes =
                    requirements.entries[6u].logicalBytes;
                prepared.residentVelocityBytes = requirements.standEntries[
                    kStandVelocityBuffer].logicalBytes;
                prepared.residentMujocoStateBytes = requirements.entries[
                    kMujocoStatesBuffer].logicalBytes;
                prepared.physicalPreparedEventValue =
                    humanMatterPreparedEventValue;
                prepared.proposalEventValue = humanMatterProposalEventValue;
                prepared.appliedEventValue = humanMatterAppliedEventValue;
                prepared.preparedTokens =
                    (__bridge id<MTLBuffer>)
                        program.acceptedPhysicsStateTokens;
                prepared.leaseContext = program.context;
                prepared.bindHumanIOCandidatePublication =
                    program.bindHumanIOCandidatePublication;
                prepared.releasePrepareLease =
                    program.releasePrepareLease;
                prepared.reservePreparedApplication =
                    program.reservePreparedApplication;
                prepared.encodePreparedApply = program.encodePreparedApply;
                prepared.abortPreparedApply = program.abortPreparedApply;
                prepared.reservePublishedRoot = program.reservePublishedRoot;
                prepared.releasePublishedRoot = program.releasePublishedRoot;
                prepared.lease = humanMatterLease.lease;
                pending->hasPreparedHumanMatter = true;
                pending->preparedHumanMatterGeneration =
                    program.slotGeneration;
                state_->humanMatterNextEventValue =
                    humanMatterAppliedEventValue;
            }
            pending->start =
                std::chrono::steady_clock::now();
            pending->ownsInFlight = true;

            state_->inFlight = true;
            state_->stats.hasInFlightSubmission = true;
            ++state_->stats.submissionCount;
            if (reusePublishedResidentState) {
                ++state_->stats.residentContinuationSubmissionCount;
            }
            if (input.stand.numanXHumanMatterProgram.valid()) {
                const std::uint64_t generation =
                    input.stand.numanXHumanMatterProgram.slotGeneration;
                const MRNumanXHumanMatterDispatchGPU completionDispatch =
                    state_->humanMatterPrepared.dispatch;
                const std::uint64_t preparedEventValue =
                    humanMatterPreparedEventValue;
                const std::shared_ptr<
                    detail::MetalArticulatedOperatorContextState>
                    retainedState = state_;
                __strong id<MTLSharedEvent> retainedTimelineEvent =
                    state_->humanMatterTimelineEvent;
                __strong id<MTLBuffer> retainedOwnerStatuses =
                    state_->humanMatterBuffers[
                        kHumanMatterOwnerStatusBuffer];
                __strong id<MTLBuffer> retainedAppliedOutcomes =
                    state_->humanMatterBuffers[
                        kHumanMatterAppliedOutcomeBuffer];
                __strong id<MTLBuffer> retainedFinalTokens =
                    state_->humanMatterBuffers[
                        kHumanMatterFinalAcceptedTokenBuffer];
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    HumanMatterPhysicalCompletionInvocation invocation{};
                    const bool failed = completed.status !=
                        MTLCommandBufferStatusCompleted;
                    if (failed) {
                        auto* statuses = static_cast<
                            MRNumanXHumanMatterOwnerStatusGPU*>(
                                retainedOwnerStatuses.contents);
                        if (statuses != nullptr) {
                            for (std::uint32_t environment = 0u;
                                 environment <
                                     completionDispatch.environmentCount;
                                 ++environment) {
                                statuses[
                                    environment *
                                    completionDispatch.ownerStatusStride].
                                        physicalCommandStatus =
                                    MR_NUMANX_HUMAN_MATTER_PHYSICAL_COMMAND_FAILED;
                            }
                        }
                        if (retainedAppliedOutcomes.contents != nullptr) {
                            auto* applied = static_cast<
                                MRNumanXHumanMatterAppliedOutcomeGPU*>(
                                retainedAppliedOutcomes.contents);
                            for (std::uint32_t environment = 0u;
                                 environment <
                                     completionDispatch.environmentCount;
                                 ++environment) {
                                MRNumanXHumanMatterAppliedOutcomeGPU failed{};
                                failed.abiVersion =
                                    MR_NUMANX_HUMAN_MATTER_ABI_VERSION;
                                failed.status =
                                    MR_NUMANX_HUMAN_MATTER_APPLIED_TERMINAL_NO_TOUCH;
                                failed.decision =
                                    MR_NUMANX_HUMAN_MATTER_ROOT_PENDING;
                                failed.code =
                                    MR_NUMANX_HUMAN_MATTER_APPLIED_INVALID_OWNER;
                                failed.programFingerprint =
                                    completionDispatch.programFingerprint;
                                failed.transactionFingerprint =
                                    completionDispatch.transactionFingerprint;
                                failed.linearizationEpoch =
                                    completionDispatch.linearizationEpoch;
                                failed.slotGeneration =
                                    completionDispatch.slotGeneration;
                                failed.environment = environment;
                                failed.stepIndex = completionDispatch.stepIndex;
                                failed.substepIndex =
                                    completionDispatch.substepIndex;
                                failed.transactionSlot =
                                    completionDispatch.transactionSlot;
                                failed.physicsSubstepCount =
                                    completionDispatch.physicsSubstepCount;
                                failed.controlStep =
                                    completionDispatch.controlStep;
                                failed.appliedFingerprint =
                                    humanMatterRecordFingerprint(&failed);
                                applied[environment] = failed;
                            }
                        }
                        if (retainedFinalTokens.contents != nullptr) {
                            std::memset(
                                retainedFinalTokens.contents,
                                0,
                                static_cast<std::size_t>(
                                    completionDispatch.environmentCount) *
                                    MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES);
                        }
                    }
                    try {
                        const std::lock_guard completionLock(
                            retainedState->mutex);
                        auto& completionPrepared =
                            retainedState->humanMatterPrepared;
                        if (completionPrepared.active &&
                            completionPrepared.dispatch.slotGeneration ==
                                generation) {
                            completionPrepared.physicalComplete = true;
                            if (failed) {
                                // Checkpoint writes on a failed first command
                                // may themselves be partial. This is a
                                // terminal quarantine: no restore/reuse.
                                completionPrepared.physicalFailure = true;
                                completionPrepared.terminalNoTouch = true;
                            }
                            invocation =
                                takeHumanMatterPhysicalCompletionLocked(
                                    completionPrepared);
                        }
                    } catch (...) {
                        // The device FAILED witness above still prevents a
                        // later kernel from touching quarantined Human state.
                    }
                    if (retainedTimelineEvent != nil &&
                        retainedTimelineEvent.signaledValue <
                            preparedEventValue) {
                        retainedTimelineEvent.signaledValue =
                            preparedEventValue;
                    }
                    invokeHumanMatterPhysicalCompletion(invocation);
                }];
            }
            humanMatterAbort.armed = false;
            humanMatterLease.armed = false;
            numanXTransactionAbort.armed = false;
            tendonLoadAbort.armed = false;
            [commandBuffer commit];
            submission.state_ = std::move(pending);
        }
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                metalBufferFailure,
            "host allocation failed while preparing Metal submission"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::
                internalFailure,
            exception.what()
        );
    }
}

MetalArticulatedOperatorDiagnostics
MetalArticulatedOperatorContext::run(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorResult& result
) {
    MetalArticulatedOperatorSubmission submission;
    MetalArticulatedOperatorDiagnostics diagnostics = submit(
        model,
        input,
        submission
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    return submission.wait(result);
}

MetalArticulatedOperatorContextStats
MetalArticulatedOperatorContext::stats() const noexcept {
    if (state_ == nullptr) {
        return {};
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->stats;
    } catch (...) {
        return {};
    }
}

MetalArticulatedOperatorDiagnostics runMetalArticulatedOperator(
    const EngineModel& model,
    const MetalArticulatedOperatorInput& input,
    MetalArticulatedOperatorResult& result,
    const MetalArticulatedOperatorConfig& config
) {
    RequiredBuffers requirements{};
    MetalArticulatedOperatorDiagnostics diagnostics{};
    if (input.stand.enabled()) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::invalidDimensions,
            "Numi Human stand horizons require MetalArticulatedOperatorContext"
        );
    }
    try {
        diagnostics = validateAndBuildLayout(
            model,
            input,
            config,
            requirements
        );
        if (!diagnostics.succeeded()) {
            return diagnostics;
        }

        const MetalArticulatedOperatorLayout layout =
            diagnostics.layout;

        MetalArticulatedOperatorResult staged{};
        MRJointDescriptorGPU emptyJoint{};
        MRArticulatedPointImpulseGPU emptyPoint{};

        @autoreleasepool {
            std::string metallibPath = config.metallibPath;
            if (metallibPath.empty()) {
                metallibPath = defaultMetallibPath();
            }
            if (metallibPath.empty()) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metallibUnavailable,
                    "no articulated-operator metallib path is available"
                );
            }

            id<MTLDevice> device =
                MTLCreateSystemDefaultDevice();
            if (device == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnavailable,
                    "no Metal-capable device is available"
                );
            }
            diagnostics.deviceName = nsString(device.name);
            if (!device.hasUnifiedMemory) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnsupported,
                    "articulated operator requires unified-memory Metal"
                );
            }
            for (const BufferRequirement& requirement :
                 requirements.entries) {
                if (requirement.allocationBytes >
                    device.maxBufferLength) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            metalBufferFailure,
                        std::string(requirement.label) +
                            " exceeds device.maxBufferLength"
                    );
                }
            }
            const std::uint64_t recommendedWorkingSet =
                device.recommendedMaxWorkingSetSize;
            if (recommendedWorkingSet != 0u &&
                static_cast<std::uint64_t>(
                    layout.totalAllocatedBytes
                ) > recommendedWorkingSet) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalBufferFailure,
                    "aggregate articulated buffers exceed "
                    "device.recommendedMaxWorkingSetSize"
                );
            }

            id<MTLCommandQueue> queue =
                [device newCommandQueue];
            if (queue == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnavailable,
                    "failed to create a Metal command queue"
                );
            }
            queue.label =
                @"MetalRobo articulated operator queue";

            NSString* path = [NSString
                stringWithUTF8String:metallibPath.c_str()];
            if (path == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metallibUnavailable,
                    "metallib path is not valid UTF-8"
                );
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device
                newLibraryWithURL:[NSURL fileURLWithPath:path]
                            error:&error];
            if (library == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalLibraryFailure,
                    "failed to load metallib: " +
                        describeError(error)
                );
            }
            id<MTLFunction> function = [library
                newFunctionWithName:@"mr_articulated_operator"];
            if (function == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalLibraryFailure,
                    "metallib does not contain the articulated operator"
                );
            }
            error = nil;
            id<MTLComputePipelineState> pipeline = [device
                newComputePipelineStateWithFunction:function
                                               error:&error];
            if (pipeline == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalPipelineFailure,
                    "failed to create articulated pipeline: " +
                        describeError(error)
                );
            }
            if (pipeline.maxTotalThreadsPerThreadgroup <
                    kThreadsPerThreadgroup ||
                pipeline.staticThreadgroupMemoryLength >
                    device.maxThreadgroupMemoryLength) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalDeviceUnsupported,
                        "device cannot execute the articulated threadgroup"
                );
            }

            id<MTLComputePipelineState> millardPipeline = nil;
            if (input.millard.enabled()) {
                id<MTLFunction> millardFunction = [library
                    newFunctionWithName:@"mr_millard_reference"];
                if (millardFunction == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalLibraryFailure,
                        "metallib does not contain the Millard reference operator"
                    );
                }
                error = nil;
                millardPipeline = [device
                    newComputePipelineStateWithFunction:millardFunction
                                                   error:&error];
                if (millardPipeline == nil ||
                    millardPipeline.maxTotalThreadsPerThreadgroup <
                        kThreadsPerThreadgroup) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalPipelineFailure,
                        "failed to create Millard reference pipeline: " +
                            describeError(error)
                    );
                }
            }
            id<MTLComputePipelineState> mujocoPipeline = nil;
            id<MTLComputePipelineState> mujocoReducePipeline = nil;
            id<MTLComputePipelineState> mujocoActivationPipeline = nil;
            if (input.mujoco.enabled()) {
                id<MTLFunction> mujocoFunction = [library
                    newFunctionWithName:@"mr_mujoco_muscle_reference"];
                if (mujocoFunction == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalLibraryFailure,
                        "metallib does not contain the MyoSim reference operator"
                    );
                }
                error = nil;
                mujocoPipeline = [device
                    newComputePipelineStateWithFunction:mujocoFunction
                                                   error:&error];
                if (mujocoPipeline == nil ||
                    mujocoPipeline.maxTotalThreadsPerThreadgroup <
                        kThreadsPerThreadgroup) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalPipelineFailure,
                        "failed to create MyoSim reference pipeline: " +
                            describeError(error)
                        );
                }
                id<MTLFunction> mujocoReduceFunction = [library
                    newFunctionWithName:@"mr_mujoco_muscle_reduce"];
                if (mujocoReduceFunction == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalLibraryFailure,
                        "metallib does not contain the MyoSim force-reduction operator"
                    );
                }
                error = nil;
                mujocoReducePipeline = [device
                    newComputePipelineStateWithFunction:mujocoReduceFunction
                                                   error:&error];
                if (mujocoReducePipeline == nil ||
                    mujocoReducePipeline.maxTotalThreadsPerThreadgroup <
                        kThreadsPerThreadgroup) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalPipelineFailure,
                        "failed to create MyoSim force-reduction pipeline: " +
                            describeError(error)
                        );
                }
                if (config.mujocoActivationTimestepSeconds > 0.0f) {
                    id<MTLFunction> mujocoActivationFunction = [library
                        newFunctionWithName:@"mr_mujoco_muscle_activation_step"];
                    if (mujocoActivationFunction == nil) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalLibraryFailure,
                            "metallib does not contain the MyoSim activation-step operator"
                        );
                    }
                    error = nil;
                    mujocoActivationPipeline = [device
                        newComputePipelineStateWithFunction:mujocoActivationFunction
                                                       error:&error];
                    if (mujocoActivationPipeline == nil ||
                        mujocoActivationPipeline.maxTotalThreadsPerThreadgroup <
                            kThreadsPerThreadgroup) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalPipelineFailure,
                            "failed to create MyoSim activation-step pipeline: " +
                                describeError(error)
                        );
                    }
                }
            }

            id<MTLBuffer> buffers[kRawBufferCount] = {};
            buffers[0] = makeInputBuffer(
                device,
                &model.world,
                requirements.entries[0],
                @"articulated world"
            );
            buffers[1] = makeInputBuffer(
                device,
                model.articulations.data(),
                requirements.entries[1],
                @"articulation descriptors"
            );
            buffers[2] = makeInputBuffer(
                device,
                model.joints.empty()
                    ? static_cast<const void*>(&emptyJoint)
                    : static_cast<const void*>(
                          model.joints.data()
                      ),
                requirements.entries[2],
                @"joint descriptors"
            );
            buffers[3] = makeInputBuffer(
                device,
                model.dofs.data(),
                requirements.entries[3],
                @"DoF properties"
            );
            buffers[4] = makeInputBuffer(
                device,
                model.bodies.data(),
                requirements.entries[4],
                @"body properties"
            );
            buffers[5] = makeInputBuffer(
                device,
                &layout.dispatch,
                requirements.entries[5],
                @"articulated dispatch"
            );
            buffers[6] = makeInputBuffer(
                device,
                input.q.data(),
                requirements.entries[6],
                @"articulated q"
            );
            buffers[7] = makeInputBuffer(
                device,
                input.points.empty()
                    ? static_cast<const void*>(&emptyPoint)
                    : static_cast<const void*>(input.points.data()),
                requirements.entries[7],
                @"point impulses"
            );
            buffers[8] = makeOutputBuffer(
                device,
                requirements.entries[8],
                @"body pose output"
            );
            buffers[9] = makeOutputBuffer(
                device,
                requirements.entries[9],
                @"point world output"
            );
            buffers[10] = makeOutputBuffer(
                device,
                requirements.entries[10],
                @"diagnostic mass output"
            );
            buffers[11] = makeOutputBuffer(
                device,
                requirements.entries[11],
                @"point Jacobian output"
            );
            buffers[12] = makeOutputBuffer(
                device,
                requirements.entries[12],
                @"generalized impulse output"
            );
            buffers[13] = makeOutputBuffer(
                device,
                requirements.entries[13],
                @"delta velocity output"
            );
            buffers[14] = makeOutputBuffer(
                device,
                requirements.entries[14],
                @"articulated status output"
            );
            std::vector<MROpenSimSpatialTransformGPU> packedPrograms;
            if (!packFunctionPrograms(model, packedPrograms)) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::invalidModel,
                    "FunctionBased program packing failed before Metal allocation"
                );
            }
            buffers[15] = makeInputBuffer(
                device,
                packedPrograms.data(),
                requirements.entries[15],
                @"FunctionBased programs"
            );
            MRMillardReferenceDispatchGPU millardDispatch{};
            if (input.millard.enabled()) {
                const MRArticulationGPU& millardArticulation =
                    model.articulations[input.articulationIndex];
                millardDispatch.abiVersion =
                    MR_MILLARD_REFERENCE_GPU_ABI_VERSION;
                millardDispatch.muscleCount = static_cast<mr_u32>(
                    input.millard.muscles.size()
                );
                millardDispatch.pathPointCount = static_cast<mr_u32>(
                    input.millard.pathPoints.size()
                );
                millardDispatch.wrapCount = static_cast<mr_u32>(
                    input.millard.cylinderWraps.size()
                );
                millardDispatch.environmentCount = static_cast<mr_u32>(
                    input.environmentCount
                );
                millardDispatch.dofCount = millardArticulation.nv;
                millardDispatch.pointWorldStride =
                    layout.dispatch.pointWorldStride;
                millardDispatch.pointJacobianStride =
                    layout.dispatch.pointJacobianStride;
                millardDispatch.bodyPoseStride =
                    layout.dispatch.bodyPoseStride;
                millardDispatch.articulationFirstBody =
                    millardArticulation.firstBody;
            }
            buffers[kMillardDispatchBuffer] = makeInputBuffer(
                device,
                &millardDispatch,
                requirements.entries[kMillardDispatchBuffer],
                @"Millard dispatch"
            );
            buffers[kMillardMusclesBuffer] = makeInputBuffer(
                device,
                input.millard.muscles.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.muscles.data()),
                requirements.entries[kMillardMusclesBuffer],
                @"Millard muscles"
            );
            buffers[kMillardStatesBuffer] = makeInputBuffer(
                device,
                input.millard.states.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.states.data()),
                requirements.entries[kMillardStatesBuffer],
                @"Millard states"
            );
            buffers[kMillardPathPointsBuffer] = makeInputBuffer(
                device,
                input.millard.pathPoints.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.pathPoints.data()),
                requirements.entries[kMillardPathPointsBuffer],
                @"Millard path points"
            );
            buffers[kMillardCurvesBuffer] = makeInputBuffer(
                device,
                input.millard.curves.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.curves.data()),
                requirements.entries[kMillardCurvesBuffer],
                @"Millard source curves"
            );
            buffers[kMillardWrapsBuffer] = makeInputBuffer(
                device,
                input.millard.cylinderWraps.empty()
                    ? nullptr
                    : static_cast<const void*>(input.millard.cylinderWraps.data()),
                requirements.entries[kMillardWrapsBuffer],
                @"Millard cylinder wraps"
            );
            buffers[kMillardResultsBuffer] = makeOutputBuffer(
                device,
                requirements.entries[kMillardResultsBuffer],
                @"Millard results"
            );
            buffers[kMillardForcesBuffer] = makeOutputBuffer(
                device,
                requirements.entries[kMillardForcesBuffer],
                @"Millard generalized forces"
            );

            MRMujocoMuscleReferenceDispatchGPU mujocoDispatch{};
            if (input.mujoco.enabled()) {
                const MRArticulationGPU& mujocoArticulation =
                    model.articulations[input.articulationIndex];
                mujocoDispatch.abiVersion =
                    MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION;
                mujocoDispatch.muscleCount = static_cast<mr_u32>(
                    input.mujoco.muscles.size()
                );
                mujocoDispatch.siteCount = static_cast<mr_u32>(
                    input.mujoco.sites.size()
                );
                mujocoDispatch.wrapCount = static_cast<mr_u32>(
                    input.mujoco.wraps.size()
                );
                mujocoDispatch.routeNodeCount = static_cast<mr_u32>(
                    input.mujoco.routeNodes.size()
                );
                mujocoDispatch.environmentCount = static_cast<mr_u32>(
                    input.environmentCount
                );
                mujocoDispatch.bodyPoseStride = layout.dispatch.bodyPoseStride;
                mujocoDispatch.articulationFirstBody =
                    mujocoArticulation.firstBody;
                mujocoDispatch.dofCount = mujocoArticulation.nv;
                mujocoDispatch.pointJacobianStride =
                    layout.dispatch.pointJacobianStride;
                mujocoDispatch.bodyJacobianPointOffset =
                    input.mujoco.bodyJacobianPointOffset;
                mujocoDispatch.bodyJacobianPointStride = 4u;
                mujocoDispatch.timestepSecondsAndReserved = {
                    config.mujocoActivationTimestepSeconds, 0.0f, 0.0f, 0.0f,
                };
            }
            buffers[kMujocoDispatchBuffer] = makeInputBuffer(
                device,
                &mujocoDispatch,
                requirements.entries[kMujocoDispatchBuffer],
                @"MyoSim dispatch"
            );
            buffers[kMujocoMusclesBuffer] = makeInputBuffer(
                device,
                input.mujoco.muscles.empty()
                    ? nullptr
                    : static_cast<const void*>(input.mujoco.muscles.data()),
                requirements.entries[kMujocoMusclesBuffer],
                @"MyoSim muscles"
            );
            buffers[kMujocoStatesBuffer] = makeInputBuffer(
                device,
                input.mujoco.states.empty()
                    ? nullptr
                    : static_cast<const void*>(input.mujoco.states.data()),
                requirements.entries[kMujocoStatesBuffer],
                @"MyoSim states"
            );
            buffers[kMujocoSitesBuffer] = makeInputBuffer(
                device,
                input.mujoco.sites.empty()
                    ? nullptr
                    : static_cast<const void*>(input.mujoco.sites.data()),
                requirements.entries[kMujocoSitesBuffer],
                @"MyoSim sites"
            );
            buffers[kMujocoWrapsBuffer] = makeInputBuffer(
                device,
                input.mujoco.wraps.empty()
                    ? nullptr
                    : static_cast<const void*>(input.mujoco.wraps.data()),
                requirements.entries[kMujocoWrapsBuffer],
                @"MyoSim wraps"
            );
            buffers[kMujocoRoutesBuffer] = makeInputBuffer(
                device,
                input.mujoco.routeNodes.empty()
                    ? nullptr
                    : static_cast<const void*>(input.mujoco.routeNodes.data()),
                requirements.entries[kMujocoRoutesBuffer],
                @"MyoSim route nodes"
            );
            buffers[kMujocoResultsBuffer] = makeOutputBuffer(
                device,
                requirements.entries[kMujocoResultsBuffer],
                @"MyoSim results"
            );
            std::vector<float> zeroMujocoVelocity;
            if (input.v.empty()) {
                zeroMujocoVelocity.assign(layout.generalizedElements, 0.0f);
            }
            id<MTLBuffer> mujocoVelocityBuffer = makeInputBuffer(
                device,
                input.v.empty()
                    ? static_cast<const void*>(zeroMujocoVelocity.data())
                    : static_cast<const void*>(input.v.data()),
                requirements.standEntries[kStandVelocityBuffer],
                @"MyoSim generalized velocities"
            );
            if (!validBuffer(
                    mujocoVelocityBuffer,
                    requirements.standEntries[kStandVelocityBuffer]
                )) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::metalBufferFailure,
                    "Metal MyoSim velocity buffer allocation failed"
                );
            }

            for (std::size_t index = 0u;
                 index < kRawBufferCount;
                 ++index) {
                if (!validBuffer(
                        buffers[index],
                        requirements.entries[index]
                    )) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            metalBufferFailure,
                        std::string("Metal buffer allocation/length "
                                    "check failed for ") +
                            requirements.entries[index].label
                    );
                }
            }

            id<MTLCommandBuffer> commandBuffer =
                [queue commandBuffer];
            if (commandBuffer == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal command buffer"
                );
            }
            commandBuffer.label =
                @"MetalRobo articulated operator";
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "failed to create Metal compute encoder"
                );
            }
            [encoder setComputePipelineState:pipeline];
            for (NSUInteger index = 0u;
                 index < kRawBufferCount;
                 ++index) {
                [encoder
                    setBuffer:buffers[index]
                       offset:0u
                      atIndex:index];
            }
            const MRArticulationGPU& articulation =
                model.articulations[input.articulationIndex];
            [encoder
                setThreadgroupMemoryLength:
                    detail::articulatedOperatorThreadgroupBytes(
                        articulation.bodyCount,
                        articulation.nv,
                        !config.pointJacobiansOnly
                    )
                atIndex:0u];
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                    static_cast<NSUInteger>(
                        input.environmentCount
                    ),
                    1u,
                    1u
                )
                threadsPerThreadgroup:MTLSizeMake(
                    kThreadsPerThreadgroup,
                    1u,
                    1u
                )];
            [encoder endEncoding];

            if (input.millard.enabled()) {
                id<MTLComputeCommandEncoder> millardEncoder =
                    [commandBuffer computeCommandEncoder];
                if (millardEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create Millard reference encoder"
                    );
                }
                [millardEncoder setComputePipelineState:millardPipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [millardEncoder setBuffer:buffers[index] offset:0u atIndex:index];
                }
                const std::size_t threadCount = layout.millardResultElements;
                [millardEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (threadCount + kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [millardEncoder endEncoding];
            }

            if (input.mujoco.enabled()) {
                id<MTLComputeCommandEncoder> mujocoEncoder =
                    [commandBuffer computeCommandEncoder];
                if (mujocoEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create MyoSim reference encoder"
                    );
                }
                [mujocoEncoder setComputePipelineState:mujocoPipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [mujocoEncoder setBuffer:buffers[index] offset:0u atIndex:index];
                }
                [mujocoEncoder setBuffer:mujocoVelocityBuffer offset:0u atIndex:7u];
                const std::size_t threadCount = layout.mujocoResultElements;
                [mujocoEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (threadCount + kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [mujocoEncoder endEncoding];
                id<MTLComputeCommandEncoder> mujocoReduceEncoder =
                    [commandBuffer computeCommandEncoder];
                if (mujocoReduceEncoder == nil) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::metalCommandFailure,
                        "failed to create MyoSim force-reduction encoder"
                    );
                }
                [mujocoReduceEncoder
                    setComputePipelineState:mujocoReducePipeline];
                for (NSUInteger index = 0u;
                     index < kRawBufferCount;
                     ++index) {
                    [mujocoReduceEncoder
                        setBuffer:buffers[index]
                           offset:0u
                          atIndex:index];
                }
                const std::size_t reducedThreadCount =
                    layout.mujocoGeneralizedForceElements;
                [mujocoReduceEncoder
                    dispatchThreadgroups:MTLSizeMake(
                        static_cast<NSUInteger>(
                            (reducedThreadCount +
                             kThreadsPerThreadgroup - 1u) /
                                kThreadsPerThreadgroup
                        ),
                        1u,
                        1u
                    )
                    threadsPerThreadgroup:MTLSizeMake(
                        kThreadsPerThreadgroup,
                        1u,
                        1u
                    )];
                [mujocoReduceEncoder endEncoding];
                if (config.mujocoActivationTimestepSeconds > 0.0f) {
                    MRMujocoMuscleActivationDispatchGPU activationDispatch{};
                    activationDispatch.abiVersion =
                        MR_MUJOCO_MUSCLE_ACTIVATION_GPU_ABI_VERSION;
                    activationDispatch.stateCount = static_cast<mr_u32>(
                        layout.mujocoStateElements
                    );
                    activationDispatch.timestepSecondsAndReserved = {
                        config.mujocoActivationTimestepSeconds,
                        0.0f,
                        0.0f,
                        0.0f,
                    };
                    id<MTLComputeCommandEncoder> activationEncoder =
                        [commandBuffer computeCommandEncoder];
                    if (activationEncoder == nil) {
                        return reject(
                            std::move(diagnostics),
                            MetalArticulatedOperatorHostStatus::metalCommandFailure,
                            "failed to create MyoSim activation-step encoder"
                        );
                    }
                    [activationEncoder
                        setComputePipelineState:mujocoActivationPipeline];
                    [activationEncoder setBuffer:buffers[kMujocoStatesBuffer]
                                       offset:0u
                                      atIndex:0u];
                    [activationEncoder setBuffer:buffers[kMujocoResultsBuffer]
                                       offset:0u
                                      atIndex:1u];
                    [activationEncoder setBytes:&activationDispatch
                                          length:sizeof(activationDispatch)
                                         atIndex:2u];
                    const std::size_t activationThreadCount =
                        layout.mujocoStateElements;
                    [activationEncoder
                        dispatchThreadgroups:MTLSizeMake(
                            static_cast<NSUInteger>(
                                (activationThreadCount +
                                 kThreadsPerThreadgroup - 1u) /
                                    kThreadsPerThreadgroup
                            ),
                            1u,
                            1u
                        )
                        threadsPerThreadgroup:MTLSizeMake(
                            kThreadsPerThreadgroup,
                            1u,
                            1u
                        )];
                    [activationEncoder endEncoding];
                }
            }

            const auto start =
                std::chrono::steady_clock::now();
            diagnostics.dispatched = true;
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            const auto end =
                std::chrono::steady_clock::now();
            diagnostics.elapsedMilliseconds =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();
            if (commandBuffer.status !=
                MTLCommandBufferStatusCompleted) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        metalCommandFailure,
                    "Metal articulated command failed: " +
                        describeError(commandBuffer.error)
                );
            }

            // Allocate the host-side transactional snapshot only after the
            // device has accepted every individual and aggregate buffer and
            // the command has completed. Until this point result is untouched.
            staged.layout = layout;
            staged.bodyPoses.resize(layout.bodyPoseElements);
            staged.pointWorld.resize(layout.pointWorldElements);
            staged.diagnosticMassMatrix.resize(
                layout.massMatrixElements
            );
            staged.pointJacobians.resize(
                layout.pointJacobianElements
            );
            staged.generalizedImpulse.resize(
                layout.generalizedElements
            );
            staged.deltaVelocity.resize(
                layout.generalizedElements
            );
            staged.statuses.resize(layout.statusElements);
            staged.millardResults.resize(layout.millardResultElements);
            staged.millardGeneralizedForces.resize(
                layout.millardGeneralizedForceElements
            );
            staged.mujocoResults.resize(layout.mujocoResultElements);
            staged.mujocoActivationStates.resize(
                layout.mujocoStateElements
            );
            staged.mujocoMuscleGeneralizedForces.resize(
                layout.mujocoMuscleGeneralizedForceElements
            );
            staged.mujocoGeneralizedForces.resize(
                layout.mujocoGeneralizedForceElements
            );
            copyOutput(staged.bodyPoses, buffers[8]);
            copyOutput(staged.pointWorld, buffers[9]);
            copyOutput(
                staged.diagnosticMassMatrix,
                buffers[10]
            );
            copyOutput(staged.pointJacobians, buffers[11]);
            copyOutput(
                staged.generalizedImpulse,
                buffers[12]
            );
            copyOutput(staged.deltaVelocity, buffers[13]);
            copyOutput(staged.statuses, buffers[14]);
            copyOutput(
                staged.millardResults,
                buffers[kMillardResultsBuffer]
            );
            copyOutput(
                staged.millardGeneralizedForces,
                buffers[kMillardForcesBuffer]
            );
            copyOutput(
                staged.mujocoResults,
                buffers[kMujocoResultsBuffer]
            );
            copyOutput(
                staged.mujocoActivationStates,
                buffers[kMujocoStatesBuffer]
            );
            copyOutput(
                staged.mujocoMuscleGeneralizedForces,
                buffers[kMillardForcesBuffer]
            );
            if (!staged.mujocoGeneralizedForces.empty()) {
                const auto* source = static_cast<const float*>(
                    buffers[kMillardForcesBuffer].contents
                ) + layout.mujocoMuscleGeneralizedForceElements;
                std::copy_n(
                    source,
                    staged.mujocoGeneralizedForces.size(),
                    staged.mujocoGeneralizedForces.begin()
                );
            }
        }

        for (std::size_t environment = 0u;
             environment < staged.statuses.size();
             ++environment) {
            const MRArticulatedOperatorStatusGPU& status =
                staged.statuses[environment];
            const MRArticulationGPU& articulation =
                model.articulations[input.articulationIndex];
            if (status.environment != environment ||
                status.articulationIndex !=
                    input.articulationIndex ||
                status.code >
                    MR_ARTICULATED_OPERATOR_ACCURACY_FAILED) {
                return reject(
                    std::move(diagnostics),
                    MetalArticulatedOperatorHostStatus::
                        internalFailure,
                    "GPU returned a malformed articulated status record"
                );
            }
            if (status.code ==
                MR_ARTICULATED_OPERATOR_SUCCESS) {
                if (status.bodyCount !=
                        articulation.bodyCount ||
                    status.nq != articulation.nq ||
                    status.nv != articulation.nv ||
                    status.pointCount != input.pointCount ||
                    !finite(status.diagnostics)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::
                            internalFailure,
                        "GPU success status has invalid dimensions or "
                        "diagnostics"
                    );
                }
                ++diagnostics.successfulEnvironmentCount;
            } else {
                if (diagnostics.failedEnvironmentCount == 0u) {
                    diagnostics.firstFailingEnvironment =
                        static_cast<std::uint32_t>(environment);
                    diagnostics.firstGPUStatusCode = status.code;
                }
                ++diagnostics.failedEnvironmentCount;
            }
        }
        if (input.millard.enabled()) {
            for (std::size_t index = 0u;
                 index < staged.millardResults.size();
                 ++index) {
                const MRMillardMuscleResultGPU& millard =
                    staged.millardResults[index];
                const std::size_t environment =
                    index / input.millard.muscles.size();
                const std::size_t muscle =
                    index - environment * input.millard.muscles.size();
                if (millard.status != MR_MILLARD_REFERENCE_SUCCESS ||
                    millard.environment != environment ||
                    millard.muscleIndex != muscle ||
                    !finite(millard.pathFiberTendonResidual)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected a Millard source-reference muscle"
                    );
                }
            }
        }
        if (input.mujoco.enabled()) {
            for (std::size_t index = 0u;
                 index < staged.mujocoResults.size();
                 ++index) {
                const MRMujocoMuscleResultGPU& mujoco =
                    staged.mujocoResults[index];
                const std::size_t environment =
                    index / input.mujoco.muscles.size();
                const std::size_t muscle =
                    index - environment * input.mujoco.muscles.size();
                if (mujoco.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
                    mujoco.environment != environment ||
                    mujoco.muscleIndex != muscle ||
                    !finite(mujoco)) {
                    return reject(
                        std::move(diagnostics),
                        MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure,
                        "GPU rejected a MyoSim source-reference muscle"
                    );
                }
            }
        }
        if (!finitePayload(staged)) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    internalFailure,
                "GPU batch contained non-finite typed payload"
            );
        }

        result = std::move(staged);
        diagnostics.published = true;
        if (diagnostics.failedEnvironmentCount != 0u) {
            return reject(
                std::move(diagnostics),
                MetalArticulatedOperatorHostStatus::
                    gpuEnvironmentFailure,
                "one or more GPU environments rejected execution"
            );
        }
        diagnostics.status =
            MetalArticulatedOperatorHostStatus::success;
        diagnostics.message.clear();
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::metalBufferFailure,
            "host allocation failed while staging Metal buffers"
        );
    } catch (const std::exception& exception) {
        return reject(
            std::move(diagnostics),
            MetalArticulatedOperatorHostStatus::internalFailure,
            exception.what()
        );
    }
}

const char* metalArticulatedOperatorHostStatusName(
    const MetalArticulatedOperatorHostStatus status
) noexcept {
    switch (status) {
    case MetalArticulatedOperatorHostStatus::success:
        return "success";
    case MetalArticulatedOperatorHostStatus::invalidModel:
        return "invalid_model";
    case MetalArticulatedOperatorHostStatus::unsupportedTopology:
        return "unsupported_topology";
    case MetalArticulatedOperatorHostStatus::invalidDimensions:
        return "invalid_dimensions";
    case MetalArticulatedOperatorHostStatus::capacityOverflow:
        return "capacity_overflow";
    case MetalArticulatedOperatorHostStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case MetalArticulatedOperatorHostStatus::nonfiniteInput:
        return "nonfinite_input";
    case MetalArticulatedOperatorHostStatus::invalidPointQuery:
        return "invalid_point_query";
    case MetalArticulatedOperatorHostStatus::metallibUnavailable:
        return "metallib_unavailable";
    case MetalArticulatedOperatorHostStatus::metalDeviceUnavailable:
        return "metal_device_unavailable";
    case MetalArticulatedOperatorHostStatus::metalDeviceUnsupported:
        return "metal_device_unsupported";
    case MetalArticulatedOperatorHostStatus::metalLibraryFailure:
        return "metal_library_failure";
    case MetalArticulatedOperatorHostStatus::metalPipelineFailure:
        return "metal_pipeline_failure";
    case MetalArticulatedOperatorHostStatus::metalBufferFailure:
        return "metal_buffer_failure";
    case MetalArticulatedOperatorHostStatus::metalCommandFailure:
        return "metal_command_failure";
    case MetalArticulatedOperatorHostStatus::gpuEnvironmentFailure:
        return "gpu_environment_failure";
    case MetalArticulatedOperatorHostStatus::internalFailure:
        return "internal_failure";
    case MetalArticulatedOperatorHostStatus::contextBusy:
        return "context_busy";
    case MetalArticulatedOperatorHostStatus::externalProgramFailure:
        return "external_program_failure";
    }
    return "unknown";
}

} // namespace metalrobo
