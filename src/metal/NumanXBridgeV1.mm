#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "NumanXBridgeV1Internal.hpp"

#include "metalrobo/numanx_human_matter_gpu.h"

#include <atomic>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <utility>

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kFullBodyMuscleCount = 416u;
constexpr std::uint32_t kFeatureCount =
    MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
constexpr std::uint32_t kInteroceptionFeatureCount =
    MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
constexpr std::uint32_t kCandidateChannelCount = 2u;

[[nodiscard]] std::uint64_t fnvU64(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t fnvCString(
    std::uint64_t hash,
    const char* value
) noexcept {
    std::uint64_t size = 0u;
    if (value != nullptr) {
        while (value[size] != '\0') {
            hash ^= static_cast<unsigned char>(value[size]);
            hash *= kFnvPrime;
            ++size;
        }
    }
    return fnvU64(hash, size);
}

[[nodiscard]] std::uint64_t nonzero(
    const std::uint64_t hash
) noexcept {
    return hash == 0u ? 0x9e3779b97f4a7c15ull : hash;
}

[[nodiscard]] std::uint64_t candidateKeyFingerprint(
    const metalrobo::MetalNumanXHumanIOTransactionKey& key
) noexcept {
    std::uint64_t hash = fnvCString(
        kFnvOffset,
        "metalrobo.numanx-human-io.candidate-key.v1");
    hash = fnvU64(hash, key.transactionFingerprint);
    hash = fnvU64(hash, key.programFingerprint);
    hash = fnvU64(hash, key.sensorFingerprint);
    hash = fnvU64(hash, key.transactionInstanceFingerprint);
    hash = fnvU64(hash, key.sensorGeneration);
    hash = fnvU64(hash, key.commandBufferIdentity);
    return nonzero(hash);
}

[[nodiscard]] bool checkedEnd(
    const std::uint64_t start,
    const std::uint64_t count,
    std::uint64_t& end
) noexcept {
    if (start == 0u || count == 0u ||
        count > std::numeric_limits<std::uint64_t>::max() - start) {
        return false;
    }
    end = start + count;
    return true;
}

[[nodiscard]] bool disjoint(
    const std::uint64_t firstStart,
    const std::uint64_t firstCount,
    const std::uint64_t secondStart,
    const std::uint64_t secondCount
) noexcept {
    std::uint64_t firstEnd = 0u;
    std::uint64_t secondEnd = 0u;
    return checkedEnd(firstStart, firstCount, firstEnd) &&
        checkedEnd(secondStart, secondCount, secondEnd) &&
        (firstEnd <= secondStart || secondEnd <= firstStart);
}

[[nodiscard]] bool bufferObject(
    void* raw,
    __unsafe_unretained id<MTLBuffer>& buffer
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLBuffer)]) return false;
    buffer = (__bridge id<MTLBuffer>)raw;
    return buffer != nil;
}

[[nodiscard]] bool eventObject(
    void* raw,
    __unsafe_unretained id<MTLSharedEvent>& event
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLSharedEvent)]) return false;
    event = (__bridge id<MTLSharedEvent>)raw;
    return event != nil;
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

[[nodiscard]] mrnx_root_v1 makeRoot(
    const metalrobo::MetalNumanXHumanMatterPreparedView& view,
    const std::uint64_t deviceRegistryID
) noexcept {
    mrnx_root_v1 root{};
    root.abi_version = MRNX_BRIDGE_ABI_V1;
    root.struct_size = sizeof(root);
    root.owner_wire_abi_version = MRNX_OWNER_WIRE_ABI_V4;
    root.environment_count = view.environmentCount;
    root.environment = 0u;
    root.transaction_slot = view.transactionSlot;
    root.step_index = view.stepIndex;
    root.control_step = view.controlStep;
    root.substep_index = view.substepIndex;
    root.physics_substep_count = view.physicsSubstepCount;
    root.q_coordinate_count = view.qCoordinateCount;
    root.dof_count = view.dofCount;
    root.dof_layout_version = view.dofLayoutVersion;
    root.program_fingerprint = view.programFingerprint;
    root.transaction_fingerprint = view.transactionFingerprint;
    root.linearization_epoch = view.linearizationEpoch;
    root.slot_generation = view.slotGeneration;
    root.device_registry_id = deviceRegistryID;
    return root;
}

[[nodiscard]] mrnx_metal_range_v1 makeRange(
    id<MTLBuffer> buffer,
    const std::uint64_t address,
    const std::uint64_t byteCount,
    const std::uint32_t elementType,
    const std::uint32_t elementBytes
) noexcept {
    mrnx_metal_range_v1 range{};
    range.abi_version = MRNX_BRIDGE_ABI_V1;
    range.struct_size = sizeof(range);
    range.metal_buffer = (__bridge void*)buffer;
    range.gpu_address = address;
    range.byte_count = byteCount;
    range.element_type = elementType;
    range.element_byte_count = elementBytes;
    return range;
}

[[nodiscard]] mrnx_event_point_v1 makeEvent(
    id<MTLSharedEvent> event,
    const std::uint64_t value,
    const std::uint64_t deviceRegistryID
) noexcept {
    mrnx_event_point_v1 point{};
    point.abi_version = MRNX_BRIDGE_ABI_V1;
    point.struct_size = sizeof(point);
    point.shared_event = (__bridge void*)event;
    point.value = value;
    point.device_registry_id = deviceRegistryID;
    return point;
}

template <typename T>
void retainHandle(T* handle) noexcept {
    if (handle != nullptr) {
        handle->references.fetch_add(1u, std::memory_order_relaxed);
    }
}

template <typename T>
void releaseHandle(T* handle) noexcept {
    if (handle != nullptr && handle->references.fetch_sub(
            1u, std::memory_order_acq_rel) == 1u) {
        delete handle;
    }
}

template <typename T>
class HandleHold {
public:
    explicit HandleHold(T* handle) noexcept : handle_(handle) {
        retainHandle(handle_);
    }
    ~HandleHold() { releaseHandle(handle_); }
    HandleHold(const HandleHold&) = delete;
    HandleHold& operator=(const HandleHold&) = delete;
private:
    T* handle_ = nullptr;
};

template <typename T>
[[nodiscard]] bool writableOutput(T* output) noexcept {
    return output != nullptr &&
        output->abi_version == MRNX_BRIDGE_ABI_V1 &&
        output->struct_size == sizeof(T);
}

} // namespace

namespace metalrobo::numanx_bridge_v1 {

struct Domain {
    explicit Domain(id<MTLDevice> value) noexcept : device(value) {
        deviceRegistryID = value != nil ? value.registryID : 0u;
    }

    __strong id<MTLDevice> device = nil;
    std::uint64_t deviceRegistryID = 0u;
    // Production readers and the final Brain latch/release writer use this
    // sole gate. Lock order is Brain runtime -> publicGate -> owner adapter ->
    // Matter -> HumanIO. Code under publicGate never calls a Brain getter.
    mutable std::shared_mutex publicGate;
    std::atomic<std::uint64_t> publicationEpoch{0u};
    std::atomic<std::uint64_t> latchedBrainGeneration{0u};
    std::atomic<std::uint64_t> sensorGeneration{0u};
    // Set between a successful Brain latch and completion of the exact native
    // Matter/HumanIO release. It is cleared only by a fully released root. A
    // post-latch native failure leaves this sticky so the future aggregate
    // reader facade can fail closed rather than expose a split tuple.
    std::atomic<bool> publicationPoisoned{false};
};

} // namespace metalrobo::numanx_bridge_v1

struct mrnx_candidate_v1 {
    std::atomic<std::uint32_t> references{2u}; // external + lifecycle
    mutable std::mutex mutex;
    metalrobo::numanx_bridge_v1::DomainPtr domain;
    metalrobo::MetalNumanXHumanIOCandidatePublicationLease lease;
    metalrobo::MetalNumanXHumanIOCandidatePublicationProgram program{};
    mrnx_candidate_view_v1 view{};
    mrnx_candidate_channel_v1 channels[kCandidateChannelCount]{};
    __strong id<MTLBuffer> values[kCandidateChannelCount]{nil, nil};
    __strong id<MTLBuffer> validity[kCandidateChannelCount]{nil, nil};
    bool bound = false;
    bool terminal = false;
    bool lifecycleHeld = true;
};

enum class BridgePreparedPhase : std::uint32_t {
    adopted = 0u,
    candidateBound,
    proposalInFlight,
    proposalReady,
    applicationReserved,
    applyInFlight,
    acceptedPendingPublication,
    rejectedReleased,
    published,
    terminalNoTouch,
};

struct mrnx_prepared_v1 {
    std::atomic<std::uint32_t> references{2u}; // external + lifecycle
    mutable std::mutex mutex;
    metalrobo::numanx_bridge_v1::DomainPtr domain;
    metalrobo::MetalNumanXHumanMatterPrepared prepared;
    mrnx_root_v1 root{};
    mrnx_wire_lease_v1 physicalGate{};
    mrnx_proposal_view_v1 proposal{};
    mrnx_applied_view_v1 applied{};
    __strong id<MTLBuffer> preparedToken = nil;
    __strong id<MTLBuffer> proposalRecord = nil;
    __strong id<MTLBuffer> proposedToken = nil;
    __strong id<MTLBuffer> appliedRecord = nil;
    __strong id<MTLBuffer> finalToken = nil;
    __strong id<MTLBuffer> publicationFence = nil;
    __strong id<MTLSharedEvent> timeline = nil;
    __strong id<MTLCommandBuffer> proposalCommandBuffer = nil;
    __strong id<MTLBuffer> brainWitness = nil;
    __strong id<MTLSharedEvent> brainWitnessEvent = nil;
    __strong id<MTLBuffer> brainPreflight = nil;
    __strong id<MTLSharedEvent> brainPreflightEvent = nil;
    __strong id<MTLCommandBuffer> applyCommandBuffer = nil;
    __strong id<MTLBuffer> brainAck = nil;
    __strong id<MTLSharedEvent> brainAckEvent = nil;
    mrnx_candidate_v1* candidate = nullptr;
    mrnx_proposal_settled_callback_v1 proposalCompletion = nullptr;
    void* proposalCompletionContext = nullptr;
    mrnx_apply_settled_callback_v1 applyCompletion = nullptr;
    void* applyCompletionContext = nullptr;
    mrnx_publication_v1 publication{};
    std::shared_ptr<void> runtimeOwner;
    void* terminalCompletionContext = nullptr;
    metalrobo::numanx_bridge_v1::PreparedTerminalCompletion
        terminalCompletion = nullptr;
    BridgePreparedPhase phase = BridgePreparedPhase::adopted;
    bool proposalCompletionDelivered = false;
    bool applyCompletionDelivered = false;
    bool proposalForcedReject = false;
    bool applyForcedReject = false;
    bool rejectedObserved = false;
    bool timeoutQuarantined = false;
    bool terminal = false;
    bool lifecycleHeld = true;
    bool terminalCompletionDelivered = false;

    ~mrnx_prepared_v1() {
        if (candidate != nullptr) {
            releaseHandle(candidate);
            candidate = nullptr;
        }
    }
};

namespace {

void releaseCandidateLifecycle(mrnx_candidate_v1* candidate) noexcept {
    if (candidate == nullptr) return;
    bool release = false;
    {
        const std::lock_guard lock(candidate->mutex);
        if (candidate->lifecycleHeld) {
            candidate->lifecycleHeld = false;
            release = true;
        }
    }
    if (release) releaseHandle(candidate);
}

void releasePreparedLifecycle(mrnx_prepared_v1* prepared) noexcept {
    if (prepared == nullptr) return;
    bool release = false;
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->lifecycleHeld) {
            prepared->lifecycleHeld = false;
            release = true;
        }
    }
    if (release) releaseHandle(prepared);
}

void notifyPreparedTerminal(
    mrnx_prepared_v1* prepared,
    const metalrobo::numanx_bridge_v1::PreparedTerminalDisposition
        disposition
) noexcept {
    if (prepared == nullptr) return;
    metalrobo::numanx_bridge_v1::PreparedTerminalCompletion completion =
        nullptr;
    void* context = nullptr;
    mrnx_root_v1 root{};
    mrnx_candidate_view_v1 candidate{};
    mrnx_candidate_channel_v1 channels[kCandidateChannelCount]{};
    bool hasCandidate = false;
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->terminalCompletionDelivered ||
            prepared->terminalCompletion == nullptr) {
            return;
        }
        prepared->terminalCompletionDelivered = true;
        completion = prepared->terminalCompletion;
        context = prepared->terminalCompletionContext;
        root = prepared->root;
        if (prepared->candidate != nullptr) {
            const std::lock_guard candidateLock(prepared->candidate->mutex);
            candidate = prepared->candidate->view;
            for (std::uint32_t index = 0u;
                 index < kCandidateChannelCount; ++index) {
                channels[index] = prepared->candidate->channels[index];
            }
            hasCandidate = true;
        }
    }
    completion(
        context,
        disposition,
        root,
        hasCandidate ? &candidate : nullptr,
        hasCandidate ? channels : nullptr,
        hasCandidate ? kCandidateChannelCount : 0u);
}

[[nodiscard]] bool commandBufferObject(
    void* raw,
    __unsafe_unretained id<MTLCommandBuffer>& commandBuffer
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLCommandBuffer)]) return false;
    commandBuffer = (__bridge id<MTLCommandBuffer>)raw;
    return commandBuffer != nil;
}

[[nodiscard]] bool sameRoot(
    const mrnx_root_v1& supplied,
    const mrnx_root_v1& expected
) noexcept {
    return supplied.abi_version == MRNX_BRIDGE_ABI_V1 &&
        supplied.struct_size == sizeof(supplied) &&
        supplied.owner_wire_abi_version == MRNX_OWNER_WIRE_ABI_V4 &&
        supplied.reserved0 == 0u &&
        supplied.environment_count == expected.environment_count &&
        supplied.environment == expected.environment &&
        supplied.transaction_slot == expected.transaction_slot &&
        supplied.step_index == expected.step_index &&
        supplied.control_step == expected.control_step &&
        supplied.substep_index == expected.substep_index &&
        supplied.physics_substep_count == expected.physics_substep_count &&
        supplied.q_coordinate_count == expected.q_coordinate_count &&
        supplied.dof_count == expected.dof_count &&
        supplied.dof_layout_version == expected.dof_layout_version &&
        supplied.program_fingerprint == expected.program_fingerprint &&
        supplied.transaction_fingerprint ==
            expected.transaction_fingerprint &&
        supplied.linearization_epoch == expected.linearization_epoch &&
        supplied.slot_generation == expected.slot_generation &&
        supplied.device_registry_id == expected.device_registry_id;
}

struct ImportedWire {
    __strong id<MTLBuffer> record = nil;
    __strong id<MTLSharedEvent> event = nil;
};

[[nodiscard]] bool importWire(
    const metalrobo::numanx_bridge_v1::DomainPtr& domain,
    const mrnx_root_v1& expectedRoot,
    const mrnx_wire_lease_v1* wire,
    const std::uint64_t recordBytes,
    ImportedWire& imported
) noexcept {
    if (domain == nullptr || domain->device == nil || wire == nullptr ||
        wire->abi_version != MRNX_BRIDGE_ABI_V1 ||
        wire->struct_size != sizeof(*wire) ||
        !sameRoot(wire->root, expectedRoot) ||
        wire->record.abi_version != MRNX_BRIDGE_ABI_V1 ||
        wire->record.struct_size != sizeof(wire->record) ||
        wire->record.metal_buffer == nullptr ||
        wire->record.gpu_address == 0u ||
        wire->record.byte_offset != 0u ||
        wire->record.byte_count != recordBytes ||
        wire->record.element_type != MRNX_ELEMENT_RAW_BYTES_V1 ||
        wire->record.element_byte_count != 1u ||
        wire->ready.abi_version != MRNX_BRIDGE_ABI_V1 ||
        wire->ready.struct_size != sizeof(wire->ready) ||
        wire->ready.shared_event == nullptr || wire->ready.value == 0u ||
        wire->ready.device_registry_id != domain->deviceRegistryID) {
        return false;
    }
    __unsafe_unretained id<MTLBuffer> record = nil;
    __unsafe_unretained id<MTLSharedEvent> event = nil;
    if (!bufferObject(wire->record.metal_buffer, record) ||
        record.device != domain->device ||
        record.gpuAddress != wire->record.gpu_address ||
        static_cast<std::uint64_t>(record.length) != recordBytes ||
        !eventObject(wire->ready.shared_event, event) ||
        !importableSharedEvent(domain->device, event)) {
        return false;
    }
    imported.record = record;
    imported.event = event;
    return true;
}

[[nodiscard]] bool wireDisjointFromPrepared(
    const mrnx_prepared_v1& prepared,
    id<MTLBuffer> buffer,
    const std::uint64_t byteCount
) noexcept {
    if (buffer == nil || byteCount == 0u) return false;
    const std::uint64_t address = buffer.gpuAddress;
    const mrnx_metal_range_v1 ranges[] = {
        prepared.proposal.proposal,
        prepared.proposal.proposed_token,
        prepared.proposal.publication_fence,
        prepared.applied.applied,
        prepared.applied.final_token,
    };
    for (const auto& range : ranges) {
        if (range.metal_buffer == (__bridge void*)buffer ||
            !disjoint(address, byteCount,
                      range.gpu_address, range.byte_count)) {
            return false;
        }
    }
    if (prepared.preparedToken == buffer ||
        !disjoint(address, byteCount,
                  prepared.preparedToken.gpuAddress,
                  MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES)) {
        return false;
    }
    if (prepared.candidate != nullptr) {
        for (std::uint32_t index = 0u;
             index < kCandidateChannelCount; ++index) {
            if (prepared.candidate->values[index] == buffer ||
                prepared.candidate->validity[index] == buffer ||
                !disjoint(
                    address,
                    byteCount,
                    prepared.candidate->channels[index].values.gpu_address,
                    prepared.candidate->channels[index].values.byte_count) ||
                !disjoint(
                    address,
                    byteCount,
                    prepared.candidate->channels[index].validity.gpu_address,
                    prepared.candidate->channels[index].validity.byte_count)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool exactPreparedRange(
    id<MTLDevice> device,
    void* raw,
    const std::uint64_t address,
    const std::uint64_t byteCount,
    __strong id<MTLBuffer>& retained
) noexcept {
    __unsafe_unretained id<MTLBuffer> buffer = nil;
    if (!bufferObject(raw, buffer) ||
        device == nil || buffer == nil || buffer.device != device ||
        address == 0u || byteCount == 0u ||
        buffer.gpuAddress != address ||
        static_cast<std::uint64_t>(buffer.length) < byteCount) {
        return false;
    }
    retained = buffer;
    return true;
}

[[nodiscard]] mrnx_completion_v1 makeCompletion(
    const std::uint32_t status,
    const std::uint32_t metalStatus,
    const std::uint64_t generation
) noexcept {
    mrnx_completion_v1 completion{};
    completion.abi_version = MRNX_BRIDGE_ABI_V1;
    completion.struct_size = sizeof(completion);
    completion.status = status;
    completion.metal_status = metalStatus;
    completion.slot_generation = generation;
    return completion;
}

void proposalCompletionCallback(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterProposalCompletionStatus status,
    const std::uint64_t slotGeneration
) noexcept {
    auto* prepared = static_cast<mrnx_prepared_v1*>(raw);
    if (prepared == nullptr) return;
    retainHandle(prepared);
    mrnx_proposal_settled_callback_v1 callback = nullptr;
    void* callbackContext = nullptr;
    mrnx_completion_v1 completion{};
    mrnx_proposal_view_v1 proposal{};
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->proposalCompletionDelivered ||
            prepared->phase != BridgePreparedPhase::proposalInFlight ||
            slotGeneration != prepared->root.slot_generation) {
            releaseHandle(prepared);
            return;
        }
        const std::uint32_t metalStatus = prepared->proposalCommandBuffer != nil
            ? static_cast<std::uint32_t>(
                  prepared->proposalCommandBuffer.status)
            : 0u;
        const bool ready = status ==
            metalrobo::MetalNumanXHumanMatterProposalCompletionStatus::ready;
        const bool commandFailed = !ready && metalStatus !=
            static_cast<std::uint32_t>(MTLCommandBufferStatusCompleted);
        prepared->proposalCompletionDelivered = true;
        prepared->phase = ready
            ? BridgePreparedPhase::proposalReady
            : BridgePreparedPhase::terminalNoTouch;
        prepared->terminal = !ready;
        callback = prepared->proposalCompletion;
        callbackContext = prepared->proposalCompletionContext;
        prepared->proposalCompletion = nullptr;
        prepared->proposalCompletionContext = nullptr;
        completion = makeCompletion(
            prepared->timeoutQuarantined
                ? MRNX_COMPLETION_TIMEOUT_QUARANTINED_V1
                : (commandFailed
                       ? MRNX_COMPLETION_COMMAND_BUFFER_FAILURE_V1
                       : (ready ? MRNX_COMPLETION_READY_V1
                                : MRNX_COMPLETION_TERMINAL_NO_TOUCH_V1)),
            metalStatus,
            slotGeneration);
        proposal = prepared->proposal;
        prepared->proposalCommandBuffer = nil;
        prepared->brainWitness = nil;
        prepared->brainWitnessEvent = nil;
        prepared->proposalForcedReject = false;
        if (!ready && prepared->candidate != nullptr) {
            const std::lock_guard candidateLock(
                prepared->candidate->mutex);
            prepared->candidate->terminal = true;
        }
    }
    if (callback != nullptr) {
        callback(callbackContext, &completion, &proposal);
    }
    if (status !=
        metalrobo::MetalNumanXHumanMatterProposalCompletionStatus::ready) {
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
    }
    // Terminal-no-touch retains its lifecycle authority indefinitely. The
    // temporary callback hold alone is released here.
    releaseHandle(prepared);
}

void applyCompletionCallback(
    void* raw,
    const metalrobo::MetalNumanXHumanMatterApplyTerminalStatus status,
    const std::uint64_t slotGeneration
) noexcept {
    auto* prepared = static_cast<mrnx_prepared_v1*>(raw);
    if (prepared == nullptr) return;
    retainHandle(prepared);
    mrnx_apply_settled_callback_v1 callback = nullptr;
    void* callbackContext = nullptr;
    mrnx_completion_v1 completion{};
    mrnx_applied_view_v1 applied{};
    bool terminalNoTouch = false;
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->applyCompletionDelivered ||
            prepared->phase != BridgePreparedPhase::applyInFlight ||
            slotGeneration != prepared->root.slot_generation) {
            releaseHandle(prepared);
            return;
        }
        const std::uint32_t metalStatus = prepared->applyCommandBuffer != nil
            ? static_cast<std::uint32_t>(prepared->applyCommandBuffer.status)
            : 0u;
        const bool timedOut = prepared->timeoutQuarantined;
        const bool nativeAccepted = status ==
            metalrobo::MetalNumanXHumanMatterApplyTerminalStatus::
                acceptedPendingPublication;
        const bool rejected = status ==
            metalrobo::MetalNumanXHumanMatterApplyTerminalStatus::
                rejectedReleased;
        const bool accepted = nativeAccepted && !timedOut;
        const bool commandFailed = !nativeAccepted && !rejected &&
            metalStatus !=
                static_cast<std::uint32_t>(MTLCommandBufferStatusCompleted);
        prepared->applyCompletionDelivered = true;
        prepared->phase = accepted
            ? BridgePreparedPhase::acceptedPendingPublication
            : (rejected ? BridgePreparedPhase::rejectedReleased
                        : BridgePreparedPhase::terminalNoTouch);
        prepared->terminal = !accepted && !rejected;
        terminalNoTouch = prepared->terminal;
        callback = prepared->applyCompletion;
        callbackContext = prepared->applyCompletionContext;
        prepared->applyCompletion = nullptr;
        prepared->applyCompletionContext = nullptr;
        const std::uint32_t completionStatus = timedOut
            ? MRNX_COMPLETION_TIMEOUT_QUARANTINED_V1
            : (commandFailed
                   ? MRNX_COMPLETION_COMMAND_BUFFER_FAILURE_V1
                   : (accepted
                          ? MRNX_COMPLETION_ACCEPTED_PENDING_PUBLICATION_V1
                          : (rejected
                                 ? MRNX_COMPLETION_REJECTED_RELEASED_V1
                                 : MRNX_COMPLETION_TERMINAL_NO_TOUCH_V1)));
        completion = makeCompletion(
            completionStatus, metalStatus, slotGeneration);
        prepared->applied.command_disposition = accepted
            ? MRNX_COMMAND_ACCEPTED_PENDING_PUBLICATION_V1
            : (rejected ? MRNX_COMMAND_REJECTED_RELEASED_V1
                        : MRNX_COMMAND_TERMINAL_NO_TOUCH_V1);
        applied = prepared->applied;
        prepared->applyCommandBuffer = nil;
        prepared->applyForcedReject = false;
        if (!accepted) {
            if (prepared->candidate != nullptr) {
                const std::lock_guard candidateLock(
                    prepared->candidate->mutex);
                prepared->candidate->terminal = true;
            }
        }
    }
    if (callback != nullptr) {
        callback(callbackContext, &completion, &applied);
    }
    if (terminalNoTouch) {
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
    }
    releaseHandle(prepared);
}

} // namespace

namespace metalrobo::numanx_bridge_v1 {

DomainPtr makeDomain(void* metalDevice) noexcept {
    @autoreleasepool {
        if (metalDevice == nullptr) return {};
        __unsafe_unretained id object = (__bridge id)metalDevice;
        if (![object conformsToProtocol:@protocol(MTLDevice)]) return {};
        __unsafe_unretained id<MTLDevice> device =
            (__bridge id<MTLDevice>)metalDevice;
        if (device == nil || device.registryID == 0u) return {};
        try {
            return std::make_shared<Domain>(device);
        } catch (...) {
            return {};
        }
    }
}

mrnx_candidate_v1* adoptCandidate(
    const DomainPtr& domain,
    MetalNumanXHumanIOCandidatePublicationLease&& lease
) noexcept {
    @autoreleasepool {
        if (domain == nullptr || domain->device == nil || !lease.valid()) {
            return nullptr;
        }
        const auto program = lease.program();
        const auto native = lease.view();
        const auto& sensor = native.sensor;
        __unsafe_unretained id<MTLBuffer> proprioceptionValues = nil;
        __unsafe_unretained id<MTLBuffer> proprioceptionValidity = nil;
        __unsafe_unretained id<MTLBuffer> interoceptionValues = nil;
        __unsafe_unretained id<MTLBuffer> interoceptionValidity = nil;
        const auto key = metalrobo::MetalNumanXHumanIOTransactionKey{
            .transactionFingerprint = sensor.transactionFingerprint,
            .programFingerprint = sensor.programFingerprint,
            .sensorFingerprint = sensor.sensorFingerprint,
            .transactionInstanceFingerprint =
                sensor.transactionInstanceFingerprint,
            .sensorGeneration = sensor.sensorGeneration,
            .commandBufferIdentity = sensor.commandBufferIdentity,
        };
        const std::uint64_t keyFingerprint = candidateKeyFingerprint(key);
        const std::uint64_t receptorMicros =
            sensor.receptorTimestampMicroseconds;
        if (!program.valid() || !key.valid() || keyFingerprint == 0u ||
            program.candidateKeyFingerprint != keyFingerprint ||
            program.transactionFingerprint != key.transactionFingerprint ||
            program.acceptedBrainGeneration !=
                sensor.acceptedBrainGeneration ||
            program.sensorGeneration != key.sensorGeneration ||
            program.humanIOProgramFingerprint != key.programFingerprint ||
            program.sensorFingerprint != key.sensorFingerprint ||
            program.transactionInstanceFingerprint !=
                key.transactionInstanceFingerprint ||
            program.candidatePublicationFingerprint == 0u ||
            program.candidatePublicationFingerprint !=
                native.candidatePublicationFingerprint ||
            program.deviceRegistryID != domain->deviceRegistryID ||
            program.identityFingerprint !=
                metalNumanXHumanIOCandidatePublicationIdentityFingerprint(
                    program) ||
            native.abiVersion != kMetalNumanXHumanIOPublicationABIVersion ||
            native.structSize != sizeof(native) ||
            native.deviceRegistryID != domain->deviceRegistryID ||
            native.proprioception.metalBuffer !=
                sensor.proprioceptionMetalBuffer ||
            native.validity.metalBuffer != sensor.validityMetalBuffer ||
            native.interoception.metalBuffer !=
                sensor.interoceptionMetalBuffer ||
            native.interoceptionValidity.metalBuffer !=
                sensor.interoceptionValidityMetalBuffer ||
            native.proprioception.gpuAddress !=
                sensor.proprioceptionGPUAddress ||
            native.validity.gpuAddress != sensor.validityGPUAddress ||
            native.interoception.gpuAddress !=
                sensor.interoceptionGPUAddress ||
            native.interoceptionValidity.gpuAddress !=
                sensor.interoceptionValidityGPUAddress ||
            native.proprioception.byteOffset != 0u ||
            native.validity.byteOffset != 0u ||
            native.interoception.byteOffset != 0u ||
            native.interoceptionValidity.byteOffset != 0u ||
            native.proprioception.byteCount !=
                sensor.proprioceptionByteCount ||
            native.validity.byteCount != sensor.validityByteCount ||
            native.interoception.byteCount !=
                sensor.interoceptionByteCount ||
            native.interoceptionValidity.byteCount !=
                sensor.interoceptionValidityByteCount ||
            native.proprioception.elementType !=
                MetalNumanXHumanIOSensorElementType::float32 ||
            native.proprioception.elementByteCount != sizeof(float) ||
            native.validity.elementType !=
                MetalNumanXHumanIOSensorElementType::uint32 ||
            native.validity.elementByteCount != sizeof(std::uint32_t) ||
            native.interoception.elementType !=
                MetalNumanXHumanIOSensorElementType::float32 ||
            native.interoception.elementByteCount != sizeof(float) ||
            native.interoceptionValidity.elementType !=
                MetalNumanXHumanIOSensorElementType::uint32 ||
            native.interoceptionValidity.elementByteCount !=
                sizeof(std::uint32_t) ||
            sensor.environmentCount != 1u || sensor.stepCount != 1u ||
            sensor.receptorCount != kFullBodyMuscleCount ||
            sensor.featureCount != kFeatureCount ||
            sensor.proprioceptionEnvironmentStrideElements !=
                kFullBodyMuscleCount * kFeatureCount ||
            sensor.proprioceptionStepStrideElements !=
                kFullBodyMuscleCount * kFeatureCount ||
            sensor.proprioceptionReceptorStrideElements != kFeatureCount ||
            sensor.validityEnvironmentStrideElements !=
                kFullBodyMuscleCount ||
            sensor.validityStepStrideElements != kFullBodyMuscleCount ||
            sensor.validityReceptorStrideElements != 1u ||
            sensor.interoceptionEnvironmentStrideElements !=
                kFullBodyMuscleCount * kInteroceptionFeatureCount ||
            sensor.interoceptionStepStrideElements !=
                kFullBodyMuscleCount * kInteroceptionFeatureCount ||
            sensor.interoceptionReceptorStrideElements !=
                kInteroceptionFeatureCount ||
            sensor.interoceptionValidityEnvironmentStrideElements !=
                kFullBodyMuscleCount ||
            sensor.interoceptionValidityStepStrideElements !=
                kFullBodyMuscleCount ||
            sensor.interoceptionValidityReceptorStrideElements != 1u ||
            sensor.proprioceptionByteCount !=
                static_cast<std::size_t>(kFullBodyMuscleCount) *
                    kFeatureCount * sizeof(float) ||
            sensor.validityByteCount !=
                static_cast<std::size_t>(kFullBodyMuscleCount) *
                    sizeof(std::uint32_t) ||
            sensor.interoceptionByteCount !=
                static_cast<std::size_t>(kFullBodyMuscleCount) *
                    kInteroceptionFeatureCount * sizeof(float) ||
            sensor.interoceptionValidityByteCount !=
                static_cast<std::size_t>(kFullBodyMuscleCount) *
                    sizeof(std::uint32_t) ||
            !bufferObject(
                native.proprioception.metalBuffer, proprioceptionValues) ||
            !bufferObject(
                native.validity.metalBuffer, proprioceptionValidity) ||
            !bufferObject(
                native.interoception.metalBuffer, interoceptionValues) ||
            !bufferObject(
                native.interoceptionValidity.metalBuffer,
                interoceptionValidity) ||
            proprioceptionValues.device != domain->device ||
            proprioceptionValidity.device != domain->device ||
            interoceptionValues.device != domain->device ||
            interoceptionValidity.device != domain->device ||
            proprioceptionValues.gpuAddress !=
                native.proprioception.gpuAddress ||
            proprioceptionValidity.gpuAddress !=
                native.validity.gpuAddress ||
            interoceptionValues.gpuAddress !=
                native.interoception.gpuAddress ||
            interoceptionValidity.gpuAddress !=
                native.interoceptionValidity.gpuAddress ||
            static_cast<std::uint64_t>(proprioceptionValues.length) !=
                native.proprioception.byteCount ||
            static_cast<std::uint64_t>(proprioceptionValidity.length) !=
                native.validity.byteCount ||
            static_cast<std::uint64_t>(interoceptionValues.length) !=
                native.interoception.byteCount ||
            static_cast<std::uint64_t>(interoceptionValidity.length) !=
                native.interoceptionValidity.byteCount) {
            return nullptr;
        }

        struct ChannelRange {
            __unsafe_unretained id<MTLBuffer> buffer;
            std::uint64_t address;
            std::uint64_t count;
        };
        const ChannelRange ranges[] = {
            {proprioceptionValues, proprioceptionValues.gpuAddress,
             native.proprioception.byteCount},
            {proprioceptionValidity, proprioceptionValidity.gpuAddress,
             native.validity.byteCount},
            {interoceptionValues, interoceptionValues.gpuAddress,
             native.interoception.byteCount},
            {interoceptionValidity, interoceptionValidity.gpuAddress,
             native.interoceptionValidity.byteCount},
        };
        for (std::size_t first = 0u; first < std::size(ranges); ++first) {
            for (std::size_t second = first + 1u;
                 second < std::size(ranges); ++second) {
                if (ranges[first].buffer == ranges[second].buffer ||
                    !disjoint(
                        ranges[first].address,
                        ranges[first].count,
                        ranges[second].address,
                        ranges[second].count)) {
                    return nullptr;
                }
            }
        }

        auto* handle = new (std::nothrow) mrnx_candidate_v1;
        if (handle == nullptr) return nullptr;
        handle->domain = domain;
        handle->lease = std::move(lease);
        handle->program = program;
        handle->values[0] = proprioceptionValues;
        handle->validity[0] = proprioceptionValidity;
        handle->values[1] = interoceptionValues;
        handle->validity[1] = interoceptionValidity;
        handle->view.abi_version = MRNX_BRIDGE_ABI_V1;
        handle->view.struct_size = sizeof(handle->view);
        handle->view.key.abi_version = MRNX_BRIDGE_ABI_V1;
        handle->view.key.struct_size = sizeof(handle->view.key);
        handle->view.key.transaction_fingerprint =
            key.transactionFingerprint;
        handle->view.key.program_fingerprint = key.programFingerprint;
        handle->view.key.sensor_fingerprint = key.sensorFingerprint;
        handle->view.key.transaction_instance_fingerprint =
            key.transactionInstanceFingerprint;
        handle->view.key.sensor_generation = key.sensorGeneration;
        handle->view.key.command_buffer_identity =
            key.commandBufferIdentity;
        handle->view.key.fingerprint = keyFingerprint;
        handle->view.accepted_brain_generation =
            program.acceptedBrainGeneration;
        handle->view.candidate_publication_fingerprint =
            program.candidatePublicationFingerprint;
        handle->view.candidate_identity_fingerprint =
            program.identityFingerprint;
        handle->view.device_registry_id = domain->deviceRegistryID;
        handle->view.channel_count = kCandidateChannelCount;
        handle->channels[0].abi_version = MRNX_BRIDGE_ABI_V1;
        handle->channels[0].struct_size = sizeof(handle->channels[0]);
        handle->channels[0].modality =
            MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1;
        handle->channels[0].flags =
            MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1;
        handle->channels[0].receptor_timestamp_microseconds = receptorMicros;
        handle->channels[0].receptor_count = kFullBodyMuscleCount;
        handle->channels[0].feature_dimension = kFeatureCount;
        handle->channels[0].values = makeRange(
            proprioceptionValues,
            proprioceptionValues.gpuAddress,
            proprioceptionValues.length,
            MRNX_ELEMENT_FLOAT32_V1,
            sizeof(float));
        handle->channels[0].validity = makeRange(
            proprioceptionValidity,
            proprioceptionValidity.gpuAddress,
            proprioceptionValidity.length,
            MRNX_ELEMENT_UINT32_V1,
            sizeof(std::uint32_t));
        handle->channels[1].abi_version = MRNX_BRIDGE_ABI_V1;
        handle->channels[1].struct_size = sizeof(handle->channels[1]);
        handle->channels[1].modality =
            MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1;
        handle->channels[1].flags =
            MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1;
        handle->channels[1].receptor_timestamp_microseconds = receptorMicros;
        handle->channels[1].receptor_count = kFullBodyMuscleCount;
        handle->channels[1].feature_dimension = kInteroceptionFeatureCount;
        handle->channels[1].values = makeRange(
            interoceptionValues,
            interoceptionValues.gpuAddress,
            interoceptionValues.length,
            MRNX_ELEMENT_FLOAT32_V1,
            sizeof(float));
        handle->channels[1].validity = makeRange(
            interoceptionValidity,
            interoceptionValidity.gpuAddress,
            interoceptionValidity.length,
            MRNX_ELEMENT_UINT32_V1,
            sizeof(std::uint32_t));
        return handle;
    }
}

mrnx_prepared_v1* adoptPrepared(
    const DomainPtr& domain,
    MetalNumanXHumanMatterPrepared&& prepared,
    std::shared_ptr<void> runtimeOwner,
    void* terminalContext,
    const PreparedTerminalCompletion terminalCompletion
) noexcept {
    @autoreleasepool {
        if (domain == nullptr || domain->device == nil || !prepared.valid()) {
            return nullptr;
        }
        MetalNumanXHumanMatterPreparedView view{};
        if (!prepared.view(view) ||
            view.abiVersion != kMetalNumanXHumanMatterABIVersion ||
            view.structSize != sizeof(view) || view.environmentCount != 1u ||
            view.stepIndex != 0u || view.substepIndex != 0u ||
            view.physicsSubstepCount != 1u ||
            view.qCoordinateCount != MRNX_FULL_BODY_NQ ||
            view.dofCount != MRNX_FULL_BODY_NV ||
            view.dofLayoutVersion !=
                kMetalNumanXHumanMatterDofLayoutVersion ||
            view.programFingerprint == 0u ||
            view.transactionFingerprint == 0u ||
            view.linearizationEpoch == 0u || view.slotGeneration == 0u ||
            view.preparedPhysicsStateTokenByteCount !=
                MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES ||
            view.finalAcceptedPhysicsStateTokenByteCount !=
                MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES ||
            view.proposedPhysicsStateTokenByteCount !=
                MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES ||
            view.proposalElementCount != 1u ||
            view.appliedOutcomeElementCount != 1u ||
            view.publicationFenceElementCount != 1u ||
            view.proposalStride != 1u ||
            view.appliedOutcomeStride != 1u ||
            view.publicationFenceStride != 1u ||
            view.proposedTokenStrideBytes !=
                MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES ||
            view.finalTokenStrideBytes !=
                MR_NUMANX_HUMAN_MATTER_ACCEPTED_TOKEN_BYTES) {
            return nullptr;
        }
        __unsafe_unretained id<MTLBuffer> preparedBuffer = nil;
        if (!bufferObject(view.preparedPhysicsStateTokens, preparedBuffer) ||
            preparedBuffer.device != domain->device) {
            return nullptr;
        }
        __unsafe_unretained id<MTLSharedEvent> event = nil;
        if (!eventObject(view.physicalPreparedEvent, event) ||
            !importableSharedEvent(domain->device, event) ||
            view.physicalPreparedEventValue == 0u ||
            view.proposalEventValue <= view.physicalPreparedEventValue ||
            view.appliedEventValue <= view.proposalEventValue) {
            return nullptr;
        }
        auto* handle = new (std::nothrow) mrnx_prepared_v1;
        if (handle == nullptr) return nullptr;
        handle->domain = domain;
        const bool exact = exactPreparedRange(
                domain->device,
                view.preparedPhysicsStateTokens,
                view.preparedPhysicsStateTokensGPUAddress,
                view.preparedPhysicsStateTokenByteCount,
                handle->preparedToken) &&
            exactPreparedRange(
                domain->device,
                view.proposals,
                view.proposalsGPUAddress,
                sizeof(MRNumanXHumanMatterProposalGPU),
                handle->proposalRecord) &&
            exactPreparedRange(
                domain->device,
                view.proposedPhysicsStateTokens,
                view.proposedPhysicsStateTokensGPUAddress,
                view.proposedPhysicsStateTokenByteCount,
                handle->proposedToken) &&
            exactPreparedRange(
                domain->device,
                view.appliedOutcomes,
                view.appliedOutcomesGPUAddress,
                sizeof(MRNumanXHumanMatterAppliedOutcomeGPU),
                handle->appliedRecord) &&
            exactPreparedRange(
                domain->device,
                view.finalAcceptedPhysicsStateTokens,
                view.finalAcceptedPhysicsStateTokensGPUAddress,
                view.finalAcceptedPhysicsStateTokenByteCount,
                handle->finalToken) &&
            exactPreparedRange(
                domain->device,
                view.publicationFences,
                view.publicationFencesGPUAddress,
                sizeof(MRNumanXHumanMatterJointPublicationFenceGPU),
                handle->publicationFence);
        struct RetainedRange {
            __unsafe_unretained id<MTLBuffer> buffer;
            std::uint64_t address;
            std::uint64_t byteCount;
        };
        const RetainedRange ranges[] = {
            {handle->preparedToken,
             view.preparedPhysicsStateTokensGPUAddress,
             view.preparedPhysicsStateTokenByteCount},
            {handle->proposalRecord,
             view.proposalsGPUAddress,
             sizeof(MRNumanXHumanMatterProposalGPU)},
            {handle->proposedToken,
             view.proposedPhysicsStateTokensGPUAddress,
             view.proposedPhysicsStateTokenByteCount},
            {handle->appliedRecord,
             view.appliedOutcomesGPUAddress,
             sizeof(MRNumanXHumanMatterAppliedOutcomeGPU)},
            {handle->finalToken,
             view.finalAcceptedPhysicsStateTokensGPUAddress,
             view.finalAcceptedPhysicsStateTokenByteCount},
            {handle->publicationFence,
             view.publicationFencesGPUAddress,
             sizeof(MRNumanXHumanMatterJointPublicationFenceGPU)},
        };
        constexpr std::size_t rangeCount =
            sizeof(ranges) / sizeof(ranges[0]);
        bool isolated = exact;
        for (std::size_t first = 0u; isolated && first < rangeCount;
             ++first) {
            for (std::size_t second = first + 1u;
                 isolated && second < rangeCount; ++second) {
                isolated = ranges[first].buffer != ranges[second].buffer &&
                    disjoint(
                        ranges[first].address,
                        ranges[first].byteCount,
                        ranges[second].address,
                        ranges[second].byteCount);
            }
        }
        if (!isolated) {
            delete handle;
            return nullptr;
        }
        handle->timeline = event;
        handle->prepared = std::move(prepared);
        handle->runtimeOwner = std::move(runtimeOwner);
        handle->terminalCompletionContext = terminalContext;
        handle->terminalCompletion = terminalCompletion;
        handle->root = makeRoot(view, domain->deviceRegistryID);
        handle->physicalGate.abi_version = MRNX_BRIDGE_ABI_V1;
        handle->physicalGate.struct_size = sizeof(handle->physicalGate);
        handle->physicalGate.root = handle->root;
        handle->physicalGate.record = makeRange(
            handle->preparedToken,
            view.preparedPhysicsStateTokensGPUAddress,
            view.preparedPhysicsStateTokenByteCount,
            MRNX_ELEMENT_RAW_BYTES_V1,
            1u);
        handle->physicalGate.ready = makeEvent(
            event,
            view.physicalPreparedEventValue,
            domain->deviceRegistryID);
        handle->proposal.abi_version = MRNX_BRIDGE_ABI_V1;
        handle->proposal.struct_size = sizeof(handle->proposal);
        handle->proposal.root = handle->root;
        handle->proposal.proposal = makeRange(
            handle->proposalRecord,
            view.proposalsGPUAddress,
            sizeof(MRNumanXHumanMatterProposalGPU),
            MRNX_ELEMENT_RAW_BYTES_V1,
            1u);
        handle->proposal.proposed_token = makeRange(
            handle->proposedToken,
            view.proposedPhysicsStateTokensGPUAddress,
            view.proposedPhysicsStateTokenByteCount,
            MRNX_ELEMENT_RAW_BYTES_V1,
            1u);
        handle->proposal.publication_fence = makeRange(
            handle->publicationFence,
            view.publicationFencesGPUAddress,
            sizeof(MRNumanXHumanMatterJointPublicationFenceGPU),
            MRNX_ELEMENT_RAW_BYTES_V1,
            1u);
        handle->proposal.ready = makeEvent(
            event, view.proposalEventValue, domain->deviceRegistryID);
        handle->applied.abi_version = MRNX_BRIDGE_ABI_V1;
        handle->applied.struct_size = sizeof(handle->applied);
        handle->applied.root = handle->root;
        handle->applied.applied = makeRange(
            handle->appliedRecord,
            view.appliedOutcomesGPUAddress,
            sizeof(MRNumanXHumanMatterAppliedOutcomeGPU),
            MRNX_ELEMENT_RAW_BYTES_V1,
            1u);
        handle->applied.final_token = makeRange(
            handle->finalToken,
            view.finalAcceptedPhysicsStateTokensGPUAddress,
            view.finalAcceptedPhysicsStateTokenByteCount,
            MRNX_ELEMENT_RAW_BYTES_V1,
            1u);
        handle->applied.ready = makeEvent(
            event, view.appliedEventValue, domain->deviceRegistryID);
        return handle;
    }
}

namespace {
void preparedPhysicalCompletionCallback(
    void* raw,
    const MetalNumanXHumanMatterPhysicalCompletionStatus status,
    const std::uint64_t slotGeneration
) noexcept {
    auto* invocation = static_cast<
        std::pair<void*, PreparedPhysicalCompletion>*>(raw);
    if (invocation == nullptr || invocation->second == nullptr) return;
    const auto completion = invocation->second;
    void* context = invocation->first;
    delete invocation;
    completion(
        context,
        status == MetalNumanXHumanMatterPhysicalCompletionStatus::ready,
        slotGeneration);
}
} // namespace

bool registerPreparedPhysicalCompletion(
    mrnx_prepared_v1* prepared,
    void* completionContext,
    const PreparedPhysicalCompletion completion
) noexcept {
    if (prepared == nullptr || completion == nullptr) return false;
    auto* invocation = new (std::nothrow)
        std::pair<void*, PreparedPhysicalCompletion>(
            completionContext, completion);
    if (invocation == nullptr) return false;
    if (!prepared->prepared.registerPhysicalCompletion(
            invocation, &preparedPhysicalCompletionCallback)) {
        delete invocation;
        return false;
    }
    return true;
}

void markPreparedPhysicalTerminal(mrnx_prepared_v1* prepared) noexcept {
    if (prepared == nullptr) return;
    {
        const std::lock_guard lock(prepared->mutex);
        prepared->terminal = true;
        prepared->phase = BridgePreparedPhase::terminalNoTouch;
        if (prepared->candidate != nullptr) {
            const std::lock_guard candidateLock(prepared->candidate->mutex);
            prepared->candidate->terminal = true;
        }
    }
    notifyPreparedTerminal(
        prepared,
        PreparedTerminalDisposition::terminalNoTouch);
}

} // namespace metalrobo::numanx_bridge_v1

extern "C" {

uint32_t mrnx_bridge_v1_abi_version(void) {
    return MRNX_BRIDGE_ABI_V1;
}

void mrnx_bridge_v1_prepared_retain(mrnx_prepared_v1* prepared) {
    retainHandle(prepared);
}

void mrnx_bridge_v1_prepared_drop(mrnx_prepared_v1* prepared) {
    releaseHandle(prepared);
}

void mrnx_bridge_v1_candidate_retain(mrnx_candidate_v1* candidate) {
    retainHandle(candidate);
}

void mrnx_bridge_v1_candidate_drop(mrnx_candidate_v1* candidate) {
    releaseHandle(candidate);
}

bool mrnx_bridge_v1_prepared_copy_root(
    const mrnx_prepared_v1* prepared,
    mrnx_root_v1* output
) {
    if (prepared == nullptr || !writableOutput(output)) return false;
    const std::lock_guard lock(prepared->mutex);
    *output = prepared->root;
    return true;
}

bool mrnx_bridge_v1_prepared_copy_physical_gate(
    const mrnx_prepared_v1* prepared,
    mrnx_wire_lease_v1* output
) {
    if (prepared == nullptr || !writableOutput(output)) return false;
    const std::lock_guard lock(prepared->mutex);
    if (prepared->terminal ||
        prepared->phase == BridgePreparedPhase::acceptedPendingPublication ||
        prepared->phase == BridgePreparedPhase::rejectedReleased ||
        prepared->phase == BridgePreparedPhase::published ||
        prepared->phase == BridgePreparedPhase::terminalNoTouch) {
        return false;
    }
    *output = prepared->physicalGate;
    return true;
}

bool mrnx_bridge_v1_candidate_copy_view(
    const mrnx_candidate_v1* candidate,
    mrnx_candidate_view_v1* output
) {
    if (candidate == nullptr || !writableOutput(output)) return false;
    const std::lock_guard lock(candidate->mutex);
    if (candidate->terminal) return false;
    *output = candidate->view;
    return true;
}

bool mrnx_bridge_v1_candidate_copy_channel(
    const mrnx_candidate_v1* candidate,
    const uint32_t channelIndex,
    mrnx_candidate_channel_v1* output
) {
    if (candidate == nullptr || channelIndex >= kCandidateChannelCount ||
        !writableOutput(output)) {
        return false;
    }
    const std::lock_guard lock(candidate->mutex);
    if (candidate->terminal) return false;
    *output = candidate->channels[channelIndex];
    return true;
}

bool mrnx_bridge_v1_bind_candidate(
    mrnx_prepared_v1* prepared,
    mrnx_candidate_v1* candidate
) {
    if (prepared == nullptr || candidate == nullptr) return false;
    bool succeeded = false;
    bool becameTerminal = false;
    {
        std::scoped_lock lock(prepared->mutex, candidate->mutex);
        if (prepared->terminal || candidate->terminal ||
            prepared->candidate != nullptr || candidate->bound ||
            prepared->domain != candidate->domain ||
            prepared->root.device_registry_id !=
                candidate->view.device_registry_id ||
            prepared->root.transaction_fingerprint !=
                candidate->view.key.transaction_fingerprint ||
            !prepared->prepared.valid() || !candidate->lease.valid()) {
            return false;
        }
        // Retain before the native one-shot bind. If the native callback
        // consumes the program and a subsequent reflected-view check fails,
        // the exact HumanIO lease/context remains quarantined with this root.
        retainHandle(candidate);
        prepared->candidate = candidate;
        candidate->bound = true;
        if (!prepared->prepared.bindHumanIOCandidatePublication(
                candidate->program)) {
            // Native bool failure is deliberately not interpreted as
            // "unconsumed": the adapter callback may have installed the raw
            // program before an owner revalidation/race failed.
            becameTerminal = true;
        } else {
            metalrobo::MetalNumanXHumanMatterPreparedView refreshed{};
            if (!prepared->prepared.view(refreshed) ||
                refreshed.humanIOCandidateKeyFingerprint !=
                    candidate->view.key.fingerprint ||
                refreshed.acceptedBrainGeneration !=
                    candidate->view.accepted_brain_generation ||
                refreshed.humanIOSensorGeneration !=
                    candidate->view.key.sensor_generation ||
                refreshed.humanIOProgramFingerprint !=
                    candidate->view.key.program_fingerprint ||
                refreshed.humanIOSensorFingerprint !=
                    candidate->view.key.sensor_fingerprint ||
                refreshed.humanIOTransactionInstanceFingerprint !=
                    candidate->view.key.transaction_instance_fingerprint ||
                refreshed.humanIOCandidatePublicationFingerprint !=
                    candidate->view.candidate_publication_fingerprint ||
                refreshed.humanIODeviceRegistryID !=
                    candidate->view.device_registry_id ||
                refreshed.humanIOIdentityFingerprint !=
                    candidate->view.candidate_identity_fingerprint) {
                // The owner has already consumed the one-shot bind. Any
                // disagreement is terminal; never substitute a candidate.
                becameTerminal = true;
            } else {
                prepared->phase = BridgePreparedPhase::candidateBound;
                succeeded = true;
            }
        }
        if (becameTerminal) {
            prepared->terminal = true;
            prepared->phase = BridgePreparedPhase::terminalNoTouch;
            candidate->terminal = true;
        }
    }
    if (becameTerminal) {
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
    }
    return succeeded;
}

static bool submitProposalImpl(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw,
    const mrnx_wire_lease_v1* brainWitness,
    void* completionContext,
    mrnx_proposal_settled_callback_v1 completion,
    const bool timeoutReject
) {
    if (prepared == nullptr || commandBufferRaw == nullptr ||
        completion == nullptr) {
        return false;
    }
    ImportedWire imported{};
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer = nil;
    bool becameTerminal = false;
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->terminal || prepared->candidate == nullptr ||
            prepared->timeoutQuarantined != timeoutReject ||
            prepared->phase != BridgePreparedPhase::candidateBound ||
            !prepared->prepared.valid() ||
            !commandBufferObject(commandBufferRaw, commandBuffer) ||
            commandBuffer.device != prepared->domain->device ||
            commandBuffer.status != MTLCommandBufferStatusNotEnqueued ||
            (!timeoutReject &&
             (!importWire(
                  prepared->domain,
                  prepared->root,
                  brainWitness,
                  sizeof(MRNumanXHumanMatterBrainCommitWitnessGPU),
                  imported) ||
              imported.event == prepared->timeline ||
              !wireDisjointFromPrepared(
                  *prepared,
                  imported.record,
                  sizeof(MRNumanXHumanMatterBrainCommitWitnessGPU))))) {
            return false;
        }
        metalrobo::MetalNumanXHumanMatterProposalRequest request{};
        request.mode = timeoutReject
            ? metalrobo::MetalNumanXHumanMatterProposalMode::forceReject
            : metalrobo::MetalNumanXHumanMatterProposalMode::
                  validateBrainWitness;
        request.commandBuffer = commandBufferRaw;
        if (!timeoutReject) {
            request.brainCommitWitnesses = brainWitness->record.metal_buffer;
            request.brainPrepareCompleteEvent =
                brainWitness->ready.shared_event;
            request.brainPrepareCompleteEventValue =
                brainWitness->ready.value;
            request.brainCommitWitnessesGPUAddress =
                brainWitness->record.gpu_address;
            request.brainCommitWitnessElementCount = 1u;
            request.brainCommitWitnessStride = 1u;
        }
        request.environmentCount = prepared->root.environment_count;
        request.transactionSlot = prepared->root.transaction_slot;
        request.stepIndex = prepared->root.step_index;
        request.substepIndex = prepared->root.substep_index;
        request.physicsSubstepCount =
            prepared->root.physics_substep_count;
        request.controlStep = prepared->root.control_step;
        request.programFingerprint = prepared->root.program_fingerprint;
        request.transactionFingerprint =
            prepared->root.transaction_fingerprint;
        request.linearizationEpoch = prepared->root.linearization_epoch;
        request.slotGeneration = prepared->root.slot_generation;
        const auto diagnostics = prepared->prepared.proposePrepared(request);
        if (!diagnostics.succeeded() || !diagnostics.encoded) return false;
        prepared->proposalCommandBuffer = commandBuffer;
        prepared->brainWitness = imported.record;
        prepared->brainWitnessEvent = imported.event;
        prepared->proposalCompletion = completion;
        prepared->proposalCompletionContext = completionContext;
        prepared->proposalCompletionDelivered = false;
        prepared->proposalForcedReject = timeoutReject;
        prepared->phase = BridgePreparedPhase::proposalInFlight;
    }
    // Registration may invoke synchronously if command completion won the
    // race. It is therefore the last operation through `prepared` on success.
    if (prepared->prepared.registerProposalCompletion(
            prepared, &proposalCompletionCallback)) {
        return true;
    }
    const bool aborted = prepared->prepared.abortProposal(commandBufferRaw);
    {
        const std::lock_guard lock(prepared->mutex);
        if (aborted &&
            prepared->phase == BridgePreparedPhase::proposalInFlight) {
            prepared->phase = BridgePreparedPhase::candidateBound;
            prepared->proposalCommandBuffer = nil;
            prepared->brainWitness = nil;
            prepared->brainWitnessEvent = nil;
            prepared->proposalCompletion = nullptr;
            prepared->proposalCompletionContext = nullptr;
            prepared->proposalCompletionDelivered = false;
            prepared->proposalForcedReject = false;
        } else {
            prepared->terminal = true;
            prepared->phase = BridgePreparedPhase::terminalNoTouch;
            becameTerminal = true;
            if (prepared->candidate != nullptr) {
                const std::lock_guard candidateLock(
                    prepared->candidate->mutex);
                prepared->candidate->terminal = true;
            }
        }
    }
    if (becameTerminal) {
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
    }
    return false;
}

bool mrnx_bridge_v1_submit_proposal(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw,
    const mrnx_wire_lease_v1* brainWitness,
    void* completionContext,
    mrnx_proposal_settled_callback_v1 completion
) {
    return submitProposalImpl(
        prepared,
        commandBufferRaw,
        brainWitness,
        completionContext,
        completion,
        false);
}

bool mrnx_bridge_v1_submit_timeout_reject_proposal(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw,
    void* completionContext,
    mrnx_proposal_settled_callback_v1 completion
) {
    return submitProposalImpl(
        prepared,
        commandBufferRaw,
        nullptr,
        completionContext,
        completion,
        true);
}

bool mrnx_bridge_v1_abort_proposal(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw
) {
    if (prepared == nullptr || commandBufferRaw == nullptr) return false;
    const std::lock_guard lock(prepared->mutex);
    if (prepared->terminal ||
        prepared->phase != BridgePreparedPhase::proposalInFlight ||
        (prepared->timeoutQuarantined &&
         !prepared->proposalForcedReject) ||
        prepared->proposalCommandBuffer == nil ||
        (__bridge void*)prepared->proposalCommandBuffer != commandBufferRaw ||
        !prepared->prepared.abortProposal(commandBufferRaw)) {
        return false;
    }
    prepared->phase = BridgePreparedPhase::candidateBound;
    prepared->proposalCommandBuffer = nil;
    prepared->brainWitness = nil;
    prepared->brainWitnessEvent = nil;
    prepared->proposalCompletion = nullptr;
    prepared->proposalCompletionContext = nullptr;
    prepared->proposalCompletionDelivered = false;
    prepared->proposalForcedReject = false;
    return true;
}

static bool reserveApplicationImpl(
    mrnx_prepared_v1* prepared,
    const mrnx_wire_lease_v1* brainPreflight,
    const bool timeoutReject
) {
    if (prepared == nullptr) return false;
    ImportedWire imported{};
    const std::lock_guard lock(prepared->mutex);
    if (prepared->terminal ||
        prepared->timeoutQuarantined != timeoutReject ||
        prepared->phase != BridgePreparedPhase::proposalReady ||
        !importWire(
            prepared->domain,
            prepared->root,
            brainPreflight,
            sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU),
            imported) ||
        imported.event == prepared->timeline ||
        !wireDisjointFromPrepared(
            *prepared,
            imported.record,
            sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU))) {
        return false;
    }
    metalrobo::MetalNumanXHumanMatterBrainPreflightView preflight{};
    preflight.brainCommitPreflights =
        brainPreflight->record.metal_buffer;
    preflight.preflightReadyEvent = brainPreflight->ready.shared_event;
    preflight.brainCommitPreflightsGPUAddress =
        brainPreflight->record.gpu_address;
    preflight.brainCommitPreflightElementCount = 1u;
    preflight.preflightReadyEventValue = brainPreflight->ready.value;
    preflight.brainCommitPreflightStride = 1u;
    preflight.environmentCount = prepared->root.environment_count;
    preflight.transactionSlot = prepared->root.transaction_slot;
    preflight.stepIndex = prepared->root.step_index;
    preflight.substepIndex = prepared->root.substep_index;
    preflight.physicsSubstepCount = prepared->root.physics_substep_count;
    preflight.controlStep = prepared->root.control_step;
    preflight.programFingerprint = prepared->root.program_fingerprint;
    preflight.transactionFingerprint =
        prepared->root.transaction_fingerprint;
    preflight.linearizationEpoch = prepared->root.linearization_epoch;
    preflight.slotGeneration = prepared->root.slot_generation;
    if (!prepared->prepared.reservePreparedApplication(preflight)) {
        return false;
    }
    prepared->brainPreflight = imported.record;
    prepared->brainPreflightEvent = imported.event;
    prepared->phase = BridgePreparedPhase::applicationReserved;
    return true;
}

bool mrnx_bridge_v1_reserve_application(
    mrnx_prepared_v1* prepared,
    const mrnx_wire_lease_v1* brainPreflight
) {
    return reserveApplicationImpl(prepared, brainPreflight, false);
}

bool mrnx_bridge_v1_reserve_timeout_reject_application(
    mrnx_prepared_v1* prepared,
    const mrnx_wire_lease_v1* brainPreflight
) {
    return reserveApplicationImpl(prepared, brainPreflight, true);
}

static bool submitApplyImpl(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw,
    const mrnx_wire_lease_v1* brainAck,
    void* completionContext,
    mrnx_apply_settled_callback_v1 completion,
    const bool timeoutReject
) {
    if (prepared == nullptr || commandBufferRaw == nullptr ||
        completion == nullptr) {
        return false;
    }
    ImportedWire imported{};
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer = nil;
    const std::lock_guard lock(prepared->mutex);
    if (prepared->terminal ||
        prepared->timeoutQuarantined != timeoutReject ||
        prepared->phase != BridgePreparedPhase::applicationReserved ||
        !commandBufferObject(commandBufferRaw, commandBuffer) ||
        commandBuffer.device != prepared->domain->device ||
        commandBuffer.status != MTLCommandBufferStatusNotEnqueued ||
        prepared->brainPreflight == nil ||
        (!timeoutReject &&
         (!importWire(
              prepared->domain,
              prepared->root,
              brainAck,
              sizeof(MRNumanXHumanMatterBrainAckGPU),
              imported) ||
          imported.event == prepared->timeline ||
          imported.event == prepared->brainPreflightEvent ||
          !wireDisjointFromPrepared(
              *prepared,
              imported.record,
              sizeof(MRNumanXHumanMatterBrainAckGPU)) ||
          imported.record == prepared->brainPreflight ||
          !disjoint(
              imported.record.gpuAddress,
              sizeof(MRNumanXHumanMatterBrainAckGPU),
              prepared->brainPreflight.gpuAddress,
              sizeof(MRNumanXHumanMatterBrainCommitPreflightGPU))))) {
        return false;
    }
    metalrobo::MetalNumanXHumanMatterApplyRequest request{};
    request.mode = timeoutReject
        ? metalrobo::MetalNumanXHumanMatterApplyMode::forceReject
        : metalrobo::MetalNumanXHumanMatterApplyMode::validateBrainAck;
    request.commandBuffer = commandBufferRaw;
    request.completionContext = prepared;
    request.completion = &applyCompletionCallback;
    if (!timeoutReject) {
        request.brainAcks = brainAck->record.metal_buffer;
        request.brainAckEvent = brainAck->ready.shared_event;
        request.brainAckEventValue = brainAck->ready.value;
        request.brainAcksGPUAddress = brainAck->record.gpu_address;
        request.brainAckElementCount = 1u;
        request.brainAckStride = 1u;
    }
    request.environmentCount = prepared->root.environment_count;
    request.transactionSlot = prepared->root.transaction_slot;
    request.stepIndex = prepared->root.step_index;
    request.substepIndex = prepared->root.substep_index;
    request.physicsSubstepCount = prepared->root.physics_substep_count;
    request.controlStep = prepared->root.control_step;
    request.programFingerprint = prepared->root.program_fingerprint;
    request.transactionFingerprint =
        prepared->root.transaction_fingerprint;
    request.linearizationEpoch = prepared->root.linearization_epoch;
    request.slotGeneration = prepared->root.slot_generation;
    const auto diagnostics = prepared->prepared.applyPrepared(request);
    if (!diagnostics.succeeded() || !diagnostics.encoded) return false;
    prepared->applyCommandBuffer = commandBuffer;
    prepared->brainAck = imported.record;
    prepared->brainAckEvent = imported.event;
    prepared->applyCompletion = completion;
    prepared->applyCompletionContext = completionContext;
    prepared->applyCompletionDelivered = false;
    prepared->applyForcedReject = timeoutReject;
    prepared->phase = BridgePreparedPhase::applyInFlight;
    return true;
}

bool mrnx_bridge_v1_submit_apply(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw,
    const mrnx_wire_lease_v1* brainAck,
    void* completionContext,
    mrnx_apply_settled_callback_v1 completion
) {
    return submitApplyImpl(
        prepared,
        commandBufferRaw,
        brainAck,
        completionContext,
        completion,
        false);
}

bool mrnx_bridge_v1_submit_timeout_reject_apply(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw,
    void* completionContext,
    mrnx_apply_settled_callback_v1 completion
) {
    return submitApplyImpl(
        prepared,
        commandBufferRaw,
        nullptr,
        completionContext,
        completion,
        true);
}

bool mrnx_bridge_v1_abort_apply(
    mrnx_prepared_v1* prepared,
    void* commandBufferRaw
) {
    if (prepared == nullptr || commandBufferRaw == nullptr) return false;
    const std::lock_guard lock(prepared->mutex);
    if (prepared->terminal ||
        prepared->phase != BridgePreparedPhase::applyInFlight ||
        (prepared->timeoutQuarantined && !prepared->applyForcedReject) ||
        prepared->applyCommandBuffer == nil ||
        (__bridge void*)prepared->applyCommandBuffer != commandBufferRaw ||
        !prepared->prepared.abortApply(commandBufferRaw)) {
        return false;
    }
    prepared->phase = BridgePreparedPhase::applicationReserved;
    prepared->applyCommandBuffer = nil;
    prepared->brainAck = nil;
    prepared->brainAckEvent = nil;
    prepared->applyCompletion = nullptr;
    prepared->applyCompletionContext = nullptr;
    prepared->applyCompletionDelivered = false;
    prepared->applyForcedReject = false;
    return true;
}

bool mrnx_bridge_v1_reserve_publication(
    mrnx_prepared_v1* prepared,
    const mrnx_publication_v1* publication
) {
    if (prepared == nullptr || publication == nullptr ||
        publication->abi_version != MRNX_BRIDGE_ABI_V1 ||
        publication->struct_size != sizeof(*publication) ||
        publication->joint_commit_fingerprint == 0u ||
        publication->brain_generation == 0u) {
        return false;
    }
    const std::lock_guard lock(prepared->mutex);
    if (prepared->terminal ||
        prepared->phase !=
            BridgePreparedPhase::acceptedPendingPublication ||
        prepared->timeoutQuarantined ||
        prepared->domain->publicationPoisoned.load(
            std::memory_order_acquire) ||
        prepared->publication.joint_commit_fingerprint != 0u ||
        prepared->candidate == nullptr) {
        return false;
    }
    metalrobo::MetalNumanXHumanMatterPublicationReservationRequest request{};
    request.environmentCount = prepared->root.environment_count;
    request.transactionSlot = prepared->root.transaction_slot;
    request.stepIndex = prepared->root.step_index;
    request.substepIndex = prepared->root.substep_index;
    request.physicsSubstepCount = prepared->root.physics_substep_count;
    request.controlStep = prepared->root.control_step;
    request.programFingerprint = prepared->root.program_fingerprint;
    request.transactionFingerprint =
        prepared->root.transaction_fingerprint;
    request.linearizationEpoch = prepared->root.linearization_epoch;
    request.slotGeneration = prepared->root.slot_generation;
    request.jointCommitFingerprint =
        publication->joint_commit_fingerprint;
    request.brainGeneration = publication->brain_generation;
    if (!prepared->prepared.reservePublishedRoot(request)) return false;
    prepared->publication = *publication;
    return true;
}

uint32_t mrnx_bridge_v1_release_accepted(
    mrnx_prepared_v1* prepared,
    const mrnx_publication_v1* publication,
    void* latchContext,
    mrnx_brain_generation_latch_v1 generationLatch
) {
    if (prepared == nullptr) {
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }
    HandleHold hold(prepared);
    std::unique_lock publicWriter(prepared->domain->publicGate);
    metalrobo::MetalNumanXHumanMatterPublicationReleaseRequest request{};
    mrnx_candidate_v1* candidate = nullptr;
    std::uint64_t nextEpoch = 0u;
    bool invalidReservation = false;
    {
        const std::lock_guard lock(prepared->mutex);
        const std::uint64_t epoch =
            prepared->domain->publicationEpoch.load(std::memory_order_acquire);
        if (publication == nullptr || generationLatch == nullptr ||
            publication->abi_version != MRNX_BRIDGE_ABI_V1 ||
            publication->struct_size != sizeof(*publication) ||
            prepared->terminal || prepared->timeoutQuarantined ||
            prepared->phase !=
                BridgePreparedPhase::acceptedPendingPublication ||
            prepared->publication.abi_version != MRNX_BRIDGE_ABI_V1 ||
            prepared->publication.struct_size !=
                sizeof(prepared->publication) ||
            prepared->publication.joint_commit_fingerprint !=
                publication->joint_commit_fingerprint ||
            prepared->publication.brain_generation !=
                publication->brain_generation ||
            publication->joint_commit_fingerprint == 0u ||
            publication->brain_generation == 0u ||
            prepared->candidate == nullptr ||
            prepared->domain->publicationPoisoned.load(
                std::memory_order_acquire) ||
            epoch == std::numeric_limits<std::uint64_t>::max()) {
            prepared->terminal = true;
            prepared->phase = BridgePreparedPhase::terminalNoTouch;
            invalidReservation = true;
            if (prepared->candidate != nullptr) {
                const std::lock_guard candidateLock(
                    prepared->candidate->mutex);
                prepared->candidate->terminal = true;
            }
        } else {
            nextEpoch = epoch + 1u;
            candidate = prepared->candidate;
            request.publicationFences =
                prepared->proposal.publication_fence.metal_buffer;
            request.publicationFencesGPUAddress =
                prepared->proposal.publication_fence.gpu_address;
            request.publicationFenceElementCount = 1u;
            request.publicationFenceStride = 1u;
            request.environmentCount = prepared->root.environment_count;
            request.transactionSlot = prepared->root.transaction_slot;
            request.stepIndex = prepared->root.step_index;
            request.substepIndex = prepared->root.substep_index;
            request.physicsSubstepCount =
                prepared->root.physics_substep_count;
            request.controlStep = prepared->root.control_step;
            request.programFingerprint = prepared->root.program_fingerprint;
            request.transactionFingerprint =
                prepared->root.transaction_fingerprint;
            request.linearizationEpoch = prepared->root.linearization_epoch;
            request.slotGeneration = prepared->root.slot_generation;
            request.jointCommitFingerprint =
                publication->joint_commit_fingerprint;
            request.brainGeneration = publication->brain_generation;
        }
    }
    if (invalidReservation) {
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }

    // The borrowed latch is called exactly once and before any native release.
    // All potentially fallible staging is complete above.
    bool brainLatched = false;
    try {
        brainLatched = generationLatch(
            latchContext, publication->brain_generation);
    } catch (...) {
        // A foreign callback must not unwind across the C boundary. It may
        // have changed Brain state before throwing, so poison the aggregate
        // publication domain even though native visibility was not attempted.
        prepared->domain->publicationPoisoned.store(
            true, std::memory_order_release);
    }
    if (!brainLatched) {
        {
            const std::lock_guard lock(prepared->mutex);
            prepared->terminal = true;
            prepared->phase = BridgePreparedPhase::terminalNoTouch;
            if (candidate != nullptr) {
                const std::lock_guard candidateLock(candidate->mutex);
                candidate->terminal = true;
            }
        }
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }
    // From this point until the complete native release is known-good, any
    // future aggregate reader must fail closed. The writer gate prevents a
    // current aggregate reader from observing this transient state.
    prepared->domain->publicationPoisoned.store(
        true, std::memory_order_release);
    const auto disposition = prepared->prepared.releasePublishedRoot(request);
    if (disposition !=
        metalrobo::MetalNumanXHumanMatterPrepareLeaseDisposition::released) {
        {
            const std::lock_guard lock(prepared->mutex);
            prepared->terminal = true;
            prepared->phase = BridgePreparedPhase::terminalNoTouch;
            if (candidate != nullptr) {
                const std::lock_guard candidateLock(candidate->mutex);
                candidate->terminal = true;
            }
        }
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }
    {
        const std::lock_guard lock(prepared->mutex);
        prepared->phase = BridgePreparedPhase::published;
        prepared->brainPreflight = nil;
        prepared->brainPreflightEvent = nil;
        prepared->brainAck = nil;
        prepared->brainAckEvent = nil;
        if (candidate != nullptr) {
            const std::lock_guard candidateLock(candidate->mutex);
            candidate->terminal = true;
        }
    }
    prepared->domain->latchedBrainGeneration.store(
        publication->brain_generation, std::memory_order_release);
    prepared->domain->sensorGeneration.store(
        candidate->view.key.sensor_generation, std::memory_order_release);
    prepared->domain->publicationEpoch.store(
        nextEpoch, std::memory_order_release);
    prepared->domain->publicationPoisoned.store(
        false, std::memory_order_release);
    notifyPreparedTerminal(
        prepared,
        metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::published);
    releaseCandidateLifecycle(candidate);
    releasePreparedLifecycle(prepared);
    return MRNX_PUBLICATION_RELEASED_V1;
}

uint32_t mrnx_bridge_v1_release_rejected(mrnx_prepared_v1* prepared) {
    if (prepared == nullptr) {
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }
    HandleHold hold(prepared);
    std::unique_lock publicWriter(prepared->domain->publicGate);
    mrnx_candidate_v1* candidate = nullptr;
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->terminal ||
            prepared->phase != BridgePreparedPhase::rejectedReleased ||
            prepared->rejectedObserved) {
            return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
        }
        prepared->rejectedObserved = true;
        candidate = prepared->candidate;
        prepared->brainPreflight = nil;
        prepared->brainPreflightEvent = nil;
        prepared->brainAck = nil;
        prepared->brainAckEvent = nil;
        if (candidate != nullptr) {
            const std::lock_guard candidateLock(candidate->mutex);
            candidate->terminal = true;
        }
    }
    notifyPreparedTerminal(
        prepared,
        metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::rejected);
    releaseCandidateLifecycle(candidate);
    releasePreparedLifecycle(prepared);
    return MRNX_PUBLICATION_REJECTED_V1;
}

bool mrnx_bridge_v1_quarantine_timeout(mrnx_prepared_v1* prepared) {
    if (prepared == nullptr) return false;
    bool becameTerminal = false;
    {
        const std::lock_guard lock(prepared->mutex);
        if (prepared->timeoutQuarantined ||
            prepared->phase == BridgePreparedPhase::rejectedReleased ||
            prepared->phase == BridgePreparedPhase::published ||
            prepared->phase == BridgePreparedPhase::terminalNoTouch) {
            return false;
        }
        prepared->timeoutQuarantined = true;
        if (prepared->phase ==
                BridgePreparedPhase::acceptedPendingPublication) {
            // The apply has already made native state accepted-but-private. It
            // may neither be restored nor published after timeout wins.
            prepared->terminal = true;
            prepared->phase = BridgePreparedPhase::terminalNoTouch;
            becameTerminal = true;
            if (prepared->candidate != nullptr) {
                const std::lock_guard candidateLock(
                    prepared->candidate->mutex);
                prepared->candidate->terminal = true;
            }
        }
    }
    if (becameTerminal) {
        notifyPreparedTerminal(
            prepared,
            metalrobo::numanx_bridge_v1::PreparedTerminalDisposition::
                terminalNoTouch);
    }
    return true;
}

uint32_t mrnx_bridge_v1_reject_unbound_candidate(
    mrnx_candidate_v1* candidate
) {
    if (candidate == nullptr) {
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }
    HandleHold hold(candidate);
    metalrobo::MetalNumanXHumanIOCandidatePublicationProgram program{};
    {
        const std::lock_guard lock(candidate->mutex);
        if (candidate->bound || candidate->terminal ||
            !candidate->lease.valid() || !candidate->program.valid()) {
            return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
        }
        candidate->terminal = true;
        program = candidate->program;
    }
    const auto disposition = program.rejectCandidate(
        program.context, program.candidatePublicationFingerprint);
    if (disposition !=
        metalrobo::MetalNumanXHumanIOCandidatePublicationDisposition::rejected) {
        return MRNX_PUBLICATION_TERMINAL_NO_TOUCH_V1;
    }
    releaseCandidateLifecycle(candidate);
    return MRNX_PUBLICATION_REJECTED_V1;
}

} // extern "C"
