#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "NumanXProgramIdentity.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <utility>

namespace metalrobo {
namespace detail {

struct MetalNumanXHumanIOBufferSlot {
    id<MTLBuffer> proprioception = nil;
    id<MTLBuffer> validity = nil;
    id<MTLBuffer> interoception = nil;
    id<MTLBuffer> interoceptionValidity = nil;
    id<MTLBuffer> motorValidation = nil;
    id<MTLBuffer> motorHeaderValidation = nil;
    id<MTLBuffer> environmentGate = nil;

    std::size_t proprioceptionByteCount = 0u;
    std::size_t validityByteCount = 0u;
    std::size_t interoceptionByteCount = 0u;
    std::size_t interoceptionValidityByteCount = 0u;
    std::size_t motorValidationByteCount = 0u;
    std::size_t motorHeaderValidationByteCount = 0u;
    std::size_t environmentGateByteCount = 0u;
    std::size_t proprioceptionEnvironmentStride = 0u;
    std::size_t proprioceptionStepStride = 0u;
    std::size_t validityEnvironmentStride = 0u;
    std::size_t validityStepStride = 0u;

    std::uint32_t environmentCount = 0u;
    std::uint32_t stepCount = 0u;
    std::uint32_t receptorCount = 0u;
    float timestepSeconds = 0.0f;
    std::uint64_t receptorTimestampMicroseconds = 0u;

    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t motorCandidateFingerprint = 0u;
    std::uint64_t acceptedBrainGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint64_t sensorFingerprint = 0u;
    std::uint64_t transactionInstanceFingerprint = 0u;
    std::uint64_t excitationGPUAddress = 0u;
    std::uint64_t motorOutputHeaderGPUAddress = 0u;
    std::uintptr_t commandBufferIdentity = 0u;

    [[nodiscard]] std::size_t retainedBytes() const noexcept {
        return proprioceptionByteCount + validityByteCount +
            interoceptionByteCount + interoceptionValidityByteCount +
            motorValidationByteCount + motorHeaderValidationByteCount +
            environmentGateByteCount;
    }
};

struct MetalNumanXHumanIOState final
    : std::enable_shared_from_this<MetalNumanXHumanIOState> {
    explicit MetalNumanXHumanIOState(MetalNumanXHumanIOConfig value)
        : config(std::move(value)) {}

    mutable std::mutex mutex;
    MetalNumanXHumanIOConfig config;
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> admitPipeline = nil;
    id<MTLComputePipelineState> validateMotorHeaderPipeline = nil;
    id<MTLComputePipelineState> gatePipeline = nil;
    id<MTLComputePipelineState> writePipeline = nil;
    NumanXExecutableImageIdentity metallibIdentity{};
    bool initialized = false;

    MetalNumanXHumanIOBufferSlot slots[2];
    int publishedSlot = -1;
    int candidateSlot = -1;
    MetalNumanXHumanIOInput candidateInput{};

    bool candidatePrepared = false;
    bool encodingStarted = false;
    bool phasesComplete = false;
    bool completionHandlerInstalled = false;
    bool commandBufferCompleted = false;
    bool commandBufferSucceeded = false;
    bool candidateCompletionRegistered = false;
    bool candidateCompletionDelivered = false;
    void* candidateCompletionContext = nullptr;
    MetalNumanXHumanIOCandidateCompletion candidateCompletion = nullptr;
    bool candidateQuarantined = false;
    bool candidatePublicationLeased = false;
    bool rootPublicationReserved = false;
    bool candidatePublicationTerminal = false;
    bool abortHandled = false;
    std::uint64_t candidatePublicationFingerprint = 0u;
    MetalNumanXHumanIOCandidatePublicationBinding publicationBinding{};
    std::uint32_t expectedStep = 0u;
    MetalNumanXTransactionPhase expectedPhase =
        MetalNumanXTransactionPhase::beginStep;
    std::uintptr_t activeCommandBufferIdentity = 0u;
    std::uintptr_t stateBufferIdentity = 0u;
    std::uintptr_t resultBufferIdentity = 0u;
    std::uintptr_t standStatusBufferIdentity = 0u;

    MetalNumanXHumanIOStatus lastStatus =
        MetalNumanXHumanIOStatus::success;
    std::string lastMessage;
};

} // namespace detail

namespace {

using State = detail::MetalNumanXHumanIOState;
using Slot = detail::MetalNumanXHumanIOBufferSlot;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr NSUInteger kPreferredThreadsPerThreadgroup = 256u;

[[nodiscard]] bool checkedMultiply(
    const std::size_t a,
    const std::size_t b,
    std::size_t& result
) noexcept {
    if (a != 0u && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

[[nodiscard]] bool checkedAdd(
    const std::size_t a,
    const std::size_t b,
    std::size_t& result
) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

[[nodiscard]] std::uint64_t hashValue(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t hashU32(
    std::uint64_t hash,
    const std::uint32_t value
) noexcept {
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t hashU64(
    const std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    return hashValue(hash, value);
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

[[nodiscard]] std::uint64_t hashCString(
    std::uint64_t hash,
    const char* value
) noexcept {
    if (value == nullptr) {
        return hashValue(hash, 0u);
    }
    std::uint64_t size = 0u;
    while (value[size] != '\0') {
        hash ^= static_cast<unsigned char>(value[size]);
        hash *= kFnvPrime;
        ++size;
    }
    return hashValue(hash, size);
}

[[nodiscard]] std::uint64_t nonzeroHash(std::uint64_t hash) noexcept {
    return hash == 0u ? 0x9e3779b97f4a7c15ull : hash;
}

[[nodiscard]] std::string fromNSString(NSString* value) {
    if (value == nil || value.UTF8String == nullptr) {
        return {};
    }
    return value.UTF8String;
}

[[nodiscard]] bool sameDevice(
    id<MTLDevice> first,
    id<MTLDevice> second
) noexcept {
    if (first == nil || second == nil) {
        return false;
    }
    return first == second || first.registryID == second.registryID;
}

[[nodiscard]] MetalNumanXHumanIODiagnostics diagnosticsLocked(
    const State& state,
    const MetalNumanXHumanIOStatus status,
    std::string message = {}
) {
    MetalNumanXHumanIODiagnostics result{};
    result.status = status;
    result.message = std::move(message);
    if (state.device != nil) {
        result.deviceName = fromNSString(state.device.name);
    }
    if (state.candidateSlot >= 0) {
        const Slot& slot = state.slots[state.candidateSlot];
        result.encoded = state.phasesComplete;
        result.commandBufferCompleted = state.commandBufferCompleted;
        result.commandBufferSucceeded = state.commandBufferSucceeded;
        result.transactionFingerprint = slot.transactionFingerprint;
        result.programFingerprint = slot.programFingerprint;
        result.sensorGeneration = slot.sensorGeneration;
        result.commandBufferIdentity = slot.commandBufferIdentity;
    } else if (state.publishedSlot >= 0) {
        const Slot& slot = state.slots[state.publishedSlot];
        result.transactionFingerprint = slot.transactionFingerprint;
        result.programFingerprint = slot.programFingerprint;
        result.sensorGeneration = slot.sensorGeneration;
    }
    return result;
}

void rememberFailureLocked(
    State& state,
    const MetalNumanXHumanIOStatus status,
    std::string message
) {
    state.lastStatus = status;
    state.lastMessage = std::move(message);
}

[[nodiscard]] MetalNumanXHumanIOSensorView makeView(
    const Slot& slot,
    const MetalNumanXHumanIOViewState state
) noexcept {
    MetalNumanXHumanIOSensorView view{};
    view.proprioceptionMetalBuffer =
        (__bridge void*)slot.proprioception;
    view.validityMetalBuffer = (__bridge void*)slot.validity;
    view.interoceptionMetalBuffer =
        (__bridge void*)slot.interoception;
    view.interoceptionValidityMetalBuffer =
        (__bridge void*)slot.interoceptionValidity;
    view.proprioceptionGPUAddress = slot.proprioception != nil
        ? static_cast<std::uint64_t>(slot.proprioception.gpuAddress)
        : 0u;
    view.validityGPUAddress = slot.validity != nil
        ? static_cast<std::uint64_t>(slot.validity.gpuAddress)
        : 0u;
    view.interoceptionGPUAddress = slot.interoception != nil
        ? static_cast<std::uint64_t>(slot.interoception.gpuAddress)
        : 0u;
    view.interoceptionValidityGPUAddress =
        slot.interoceptionValidity != nil
        ? static_cast<std::uint64_t>(slot.interoceptionValidity.gpuAddress)
        : 0u;
    view.proprioceptionByteCount = slot.proprioceptionByteCount;
    view.validityByteCount = slot.validityByteCount;
    view.interoceptionByteCount = slot.interoceptionByteCount;
    view.interoceptionValidityByteCount =
        slot.interoceptionValidityByteCount;
    view.environmentCount = slot.environmentCount;
    view.stepCount = slot.stepCount;
    view.receptorCount = slot.receptorCount;
    view.featureCount =
        MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
    view.proprioceptionEnvironmentStrideElements =
        slot.proprioceptionEnvironmentStride;
    view.proprioceptionStepStrideElements =
        slot.proprioceptionStepStride;
    view.proprioceptionReceptorStrideElements =
        MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
    view.validityEnvironmentStrideElements =
        slot.validityEnvironmentStride;
    view.validityStepStrideElements = slot.validityStepStride;
    view.validityReceptorStrideElements = 1u;
    view.interoceptionEnvironmentStrideElements =
        slot.validityEnvironmentStride;
    view.interoceptionStepStrideElements = slot.validityStepStride;
    view.interoceptionReceptorStrideElements =
        MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
    view.interoceptionValidityEnvironmentStrideElements =
        slot.validityEnvironmentStride;
    view.interoceptionValidityStepStrideElements =
        slot.validityStepStride;
    view.interoceptionValidityReceptorStrideElements = 1u;
    view.transactionFingerprint = slot.transactionFingerprint;
    view.motorCandidateFingerprint = slot.motorCandidateFingerprint;
    view.acceptedBrainGeneration = slot.acceptedBrainGeneration;
    view.sensorGeneration = slot.sensorGeneration;
    view.programFingerprint = slot.programFingerprint;
    view.sensorFingerprint = slot.sensorFingerprint;
    view.transactionInstanceFingerprint =
        slot.transactionInstanceFingerprint;
    view.excitationGPUAddress = slot.excitationGPUAddress;
    view.motorOutputHeaderGPUAddress = slot.motorOutputHeaderGPUAddress;
    view.commandBufferIdentity = slot.commandBufferIdentity;
    view.receptorTimestampMicroseconds =
        slot.receptorTimestampMicroseconds;
    view.receptorTimeSeconds = static_cast<double>(
        slot.receptorTimestampMicroseconds
    ) / 1'000'000.0;
    view.deliveryTimeSeconds =
        view.receptorTimeSeconds + slot.timestepSeconds;
    view.latencySeconds = slot.timestepSeconds;
    view.stepTimeStrideSeconds = slot.timestepSeconds;
    view.state = state;
    return view;
}

[[nodiscard]] MetalNumanXHumanIOTransactionKey makeKey(
    const Slot& slot
) noexcept {
    MetalNumanXHumanIOTransactionKey key{};
    key.transactionFingerprint = slot.transactionFingerprint;
    key.programFingerprint = slot.programFingerprint;
    key.sensorFingerprint = slot.sensorFingerprint;
    key.transactionInstanceFingerprint =
        slot.transactionInstanceFingerprint;
    key.sensorGeneration = slot.sensorGeneration;
    key.commandBufferIdentity = slot.commandBufferIdentity;
    return key;
}

struct CandidateCompletionInvocation {
    MetalNumanXHumanIOCandidateCompletion completion = nullptr;
    void* context = nullptr;
    MetalNumanXHumanIOCandidateCompletionStatus status =
        MetalNumanXHumanIOCandidateCompletionStatus::commandBufferFailure;
    MetalNumanXHumanIOTransactionKey key{};
    MetalNumanXHumanIOSensorView view{};
};

[[nodiscard]] CandidateCompletionInvocation
takeCandidateCompletionLocked(State& state) noexcept {
    CandidateCompletionInvocation invocation{};
    if (!state.candidateCompletionRegistered ||
        state.candidateCompletionDelivered ||
        state.candidateCompletion == nullptr ||
        state.candidateCompletionContext == nullptr ||
        !state.commandBufferCompleted || !state.candidatePrepared ||
        state.candidateSlot < 0) {
        return invocation;
    }
    const Slot& slot = state.slots[state.candidateSlot];
    invocation.completion = state.candidateCompletion;
    invocation.context = state.candidateCompletionContext;
    invocation.status = state.commandBufferSucceeded &&
            !state.candidateQuarantined
        ? MetalNumanXHumanIOCandidateCompletionStatus::succeeded
        : MetalNumanXHumanIOCandidateCompletionStatus::commandBufferFailure;
    invocation.key = makeKey(slot);
    invocation.view = makeView(
        slot, MetalNumanXHumanIOViewState::candidate);
    state.candidateCompletionRegistered = false;
    state.candidateCompletionDelivered = true;
    state.candidateCompletionContext = nullptr;
    state.candidateCompletion = nullptr;
    return invocation;
}

void invokeCandidateCompletion(
    const CandidateCompletionInvocation& invocation
) noexcept {
    if (invocation.completion != nullptr && invocation.context != nullptr) {
        invocation.completion(
            invocation.context,
            invocation.status,
            invocation.key,
            invocation.view);
    }
}

[[nodiscard]] bool keyMatches(
    const MetalNumanXHumanIOTransactionKey& key,
    const Slot& slot
) noexcept {
    return key.valid() &&
        key.transactionFingerprint == slot.transactionFingerprint &&
        key.programFingerprint == slot.programFingerprint &&
        key.sensorFingerprint == slot.sensorFingerprint &&
        key.transactionInstanceFingerprint ==
            slot.transactionInstanceFingerprint &&
        key.sensorGeneration == slot.sensorGeneration &&
        key.commandBufferIdentity == slot.commandBufferIdentity;
}

void clearCandidateOwnershipLocked(State& state) noexcept;

[[nodiscard]] std::uint64_t candidateKeyFingerprint(
    const MetalNumanXHumanIOTransactionKey& key
) noexcept {
    std::uint64_t hash = hashCString(
        kFnvOffset,
        "metalrobo.numanx-human-io.candidate-key.v1"
    );
    hash = hashU64(hash, key.transactionFingerprint);
    hash = hashU64(hash, key.programFingerprint);
    hash = hashU64(hash, key.sensorFingerprint);
    hash = hashU64(hash, key.transactionInstanceFingerprint);
    hash = hashU64(hash, key.sensorGeneration);
    hash = hashU64(hash, key.commandBufferIdentity);
    return nonzeroHash(hash);
}

[[nodiscard]] std::uint64_t candidatePublicationFingerprint(
    const State& state,
    const Slot& slot,
    const MetalNumanXHumanIOTransactionKey& key
) noexcept {
    std::uint64_t hash = hashCString(
        kFnvOffset,
        "metalrobo.numanx-human-io.candidate-publication.v1"
    );
    hash = hashU32(
        hash,
        kMetalNumanXHumanIOPublicationABIVersion
    );
    hash = hashU64(hash, state.device != nil ? state.device.registryID : 0u);
    hash = hashU64(hash, key.transactionFingerprint);
    hash = hashU64(hash, key.programFingerprint);
    hash = hashU64(hash, key.sensorFingerprint);
    hash = hashU64(hash, key.transactionInstanceFingerprint);
    hash = hashU64(hash, key.sensorGeneration);
    hash = hashU64(hash, key.commandBufferIdentity);
    hash = hashU64(hash, slot.motorCandidateFingerprint);
    hash = hashU64(hash, slot.acceptedBrainGeneration);
    hash = hashU64(
        hash,
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)slot.proprioception
        )
    );
    hash = hashU64(hash, slot.proprioception.gpuAddress);
    hash = hashU64(hash, 0u);
    hash = hashU64(hash, slot.proprioceptionByteCount);
    hash = hashU32(
        hash,
        static_cast<std::uint32_t>(
            MetalNumanXHumanIOSensorElementType::float32
        )
    );
    hash = hashU32(hash, sizeof(float));
    hash = hashU64(
        hash,
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)slot.validity
        )
    );
    hash = hashU64(hash, slot.validity.gpuAddress);
    hash = hashU64(hash, 0u);
    hash = hashU64(hash, slot.validityByteCount);
    hash = hashU32(
        hash,
        static_cast<std::uint32_t>(
            MetalNumanXHumanIOSensorElementType::uint32
        )
    );
    hash = hashU32(hash, sizeof(std::uint32_t));
    hash = hashU64(
        hash,
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)slot.interoception
        )
    );
    hash = hashU64(hash, slot.interoception.gpuAddress);
    hash = hashU64(hash, 0u);
    hash = hashU64(hash, slot.interoceptionByteCount);
    hash = hashU32(
        hash,
        static_cast<std::uint32_t>(
            MetalNumanXHumanIOSensorElementType::float32
        )
    );
    hash = hashU32(hash, sizeof(float));
    hash = hashU64(
        hash,
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)slot.interoceptionValidity
        )
    );
    hash = hashU64(hash, slot.interoceptionValidity.gpuAddress);
    hash = hashU64(hash, 0u);
    hash = hashU64(hash, slot.interoceptionValidityByteCount);
    hash = hashU32(
        hash,
        static_cast<std::uint32_t>(
            MetalNumanXHumanIOSensorElementType::uint32
        )
    );
    hash = hashU32(hash, sizeof(std::uint32_t));
    hash = hashU32(hash, slot.environmentCount);
    hash = hashU32(hash, slot.stepCount);
    hash = hashU32(hash, slot.receptorCount);
    hash = hashU32(
        hash,
        MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT
    );
    hash = hashU64(hash, slot.proprioceptionEnvironmentStride);
    hash = hashU64(hash, slot.proprioceptionStepStride);
    hash = hashU64(
        hash,
        MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT
    );
    hash = hashU64(hash, slot.validityEnvironmentStride);
    hash = hashU64(hash, slot.validityStepStride);
    hash = hashU64(hash, 1u);
    hash = hashU32(
        hash,
        MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT
    );
    hash = hashU64(hash, slot.validityEnvironmentStride);
    hash = hashU64(hash, slot.validityStepStride);
    hash = hashU64(
        hash,
        MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT
    );
    hash = hashU64(hash, slot.validityEnvironmentStride);
    hash = hashU64(hash, slot.validityStepStride);
    hash = hashU64(hash, 1u);
    hash = hashU32(hash, std::bit_cast<std::uint32_t>(slot.timestepSeconds));
    hash = hashU64(hash, slot.receptorTimestampMicroseconds);
    hash = hashU64(hash, slot.excitationGPUAddress);
    hash = hashU64(hash, slot.motorOutputHeaderGPUAddress);
    return nonzeroHash(hash);
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationView
makeCandidatePublicationView(
    const State& state,
    const Slot& slot,
    const std::uint64_t fingerprint
) noexcept {
    MetalNumanXHumanIOCandidatePublicationView view{};
    view.sensor = makeView(
        slot,
        MetalNumanXHumanIOViewState::candidate
    );
    view.proprioception.metalBuffer =
        (__bridge void*)slot.proprioception;
    view.proprioception.gpuAddress = slot.proprioception.gpuAddress;
    view.proprioception.byteOffset = 0u;
    view.proprioception.byteCount = slot.proprioceptionByteCount;
    view.proprioception.elementType =
        MetalNumanXHumanIOSensorElementType::float32;
    view.proprioception.elementByteCount = sizeof(float);
    view.validity.metalBuffer = (__bridge void*)slot.validity;
    view.validity.gpuAddress = slot.validity.gpuAddress;
    view.validity.byteOffset = 0u;
    view.validity.byteCount = slot.validityByteCount;
    view.validity.elementType =
        MetalNumanXHumanIOSensorElementType::uint32;
    view.validity.elementByteCount = sizeof(std::uint32_t);
    view.interoception.metalBuffer =
        (__bridge void*)slot.interoception;
    view.interoception.gpuAddress = slot.interoception.gpuAddress;
    view.interoception.byteOffset = 0u;
    view.interoception.byteCount = slot.interoceptionByteCount;
    view.interoception.elementType =
        MetalNumanXHumanIOSensorElementType::float32;
    view.interoception.elementByteCount = sizeof(float);
    view.interoceptionValidity.metalBuffer =
        (__bridge void*)slot.interoceptionValidity;
    view.interoceptionValidity.gpuAddress =
        slot.interoceptionValidity.gpuAddress;
    view.interoceptionValidity.byteOffset = 0u;
    view.interoceptionValidity.byteCount =
        slot.interoceptionValidityByteCount;
    view.interoceptionValidity.elementType =
        MetalNumanXHumanIOSensorElementType::uint32;
    view.interoceptionValidity.elementByteCount = sizeof(std::uint32_t);
    view.deviceRegistryID = state.device != nil
        ? state.device.registryID
        : 0u;
    view.candidatePublicationFingerprint = fingerprint;
    return view;
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationProgram
makeCandidatePublicationProgram(
    State& state,
    const Slot& slot,
    std::uint64_t fingerprint
) noexcept;

[[nodiscard]] bool validPublicationBinding(
    const MetalNumanXHumanIOCandidatePublicationBinding& binding,
    const Slot& slot,
    const MetalNumanXHumanIOCandidatePublicationProgram& program
) noexcept {
    return binding.abiVersion ==
            kMetalNumanXHumanIOPublicationABIVersion &&
        binding.structSize == sizeof(binding) &&
        binding.environmentCount == slot.environmentCount &&
        binding.environmentCount == 1u &&
        binding.stepIndex == 0u && binding.substepIndex == 0u &&
        binding.physicsSubstepCount == 1u &&
        binding.ownerProgramFingerprint != 0u &&
        binding.transactionFingerprint == slot.transactionFingerprint &&
        binding.linearizationEpoch != 0u &&
        binding.slotGeneration != 0u &&
        binding.physicsTokenFingerprint != 0u &&
        binding.proposalFingerprint != 0u &&
        binding.ackFingerprint != 0u &&
        binding.appliedDecisionFingerprint != 0u &&
        binding.jointCommitFingerprint != 0u &&
        binding.brainGeneration == slot.acceptedBrainGeneration &&
        binding.candidateKeyFingerprint == program.candidateKeyFingerprint &&
        binding.acceptedBrainGeneration ==
            program.acceptedBrainGeneration &&
        binding.humanIOProgramFingerprint ==
            program.humanIOProgramFingerprint &&
        binding.sensorFingerprint == program.sensorFingerprint &&
        binding.transactionInstanceFingerprint ==
            program.transactionInstanceFingerprint &&
        binding.candidatePublicationFingerprint ==
            program.candidatePublicationFingerprint &&
        binding.deviceRegistryID == program.deviceRegistryID &&
        binding.deviceRegistryID != 0u &&
        binding.sensorGeneration == slot.sensorGeneration &&
        binding.humanIOIdentityFingerprint == program.identityFingerprint &&
        binding.bindingFingerprint != 0u &&
        binding.bindingFingerprint ==
            metalNumanXHumanIOPublicationBindingFingerprint(binding);
}

[[nodiscard]] bool reservePublishedRootCallback(
    void* context,
    const std::uint64_t candidateFingerprint,
    const MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept {
    if (context == nullptr || candidateFingerprint == 0u) {
        return false;
    }
    State& state = *static_cast<State*>(context);
    try {
        const std::lock_guard lock(state.mutex);
        if (!state.candidatePrepared || state.candidateSlot < 0 ||
            !state.candidatePublicationLeased ||
            state.candidatePublicationTerminal ||
            state.candidatePublicationFingerprint != candidateFingerprint ||
            state.rootPublicationReserved) {
            return false;
        }
        const Slot& slot = state.slots[state.candidateSlot];
        const auto program = makeCandidatePublicationProgram(
            state,
            slot,
            candidateFingerprint
        );
        if (!state.commandBufferCompleted ||
            !state.commandBufferSucceeded || state.candidateQuarantined ||
            !validPublicationBinding(
                binding,
                slot,
                program
            )) {
            return false;
        }
        state.publicationBinding = binding;
        state.rootPublicationReserved = true;
        return true;
    } catch (...) {
        return false;
    }
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationDisposition
publishCandidateCallback(
    void* context,
    const std::uint64_t candidateFingerprint,
    const MetalNumanXHumanIOCandidatePublicationCommit& commit
) noexcept {
    using Disposition =
        MetalNumanXHumanIOCandidatePublicationDisposition;
    if (context == nullptr || candidateFingerprint == 0u) {
        return Disposition::terminalNoTouch;
    }
    State& state = *static_cast<State*>(context);
    try {
        const std::lock_guard lock(state.mutex);
        if (state.candidatePublicationTerminal) {
            return Disposition::terminalNoTouch;
        }
        const bool exact = state.candidatePrepared &&
            state.candidateSlot >= 0 &&
            state.candidatePublicationLeased &&
            state.rootPublicationReserved &&
            state.candidatePublicationFingerprint == candidateFingerprint &&
            commit.abiVersion ==
                kMetalNumanXHumanIOPublicationABIVersion &&
            commit.structSize == sizeof(commit) &&
            commit.status ==
                MetalNumanXHumanIOCandidatePublicationCommitStatus::committed &&
            commit.reserved0 == 0u &&
            commit.candidatePublicationFingerprint == candidateFingerprint &&
            commit.bindingFingerprint ==
                state.publicationBinding.bindingFingerprint &&
            commit.jointCommitFingerprint ==
                state.publicationBinding.jointCommitFingerprint &&
            commit.brainGeneration ==
                state.publicationBinding.brainGeneration &&
            commit.fenceFingerprint != 0u;
        if (!exact) {
            state.candidatePublicationTerminal = true;
            state.candidateQuarantined = true;
            state.lastStatus =
                MetalNumanXHumanIOStatus::incompatibleTransaction;
            return Disposition::terminalNoTouch;
        }
        state.publishedSlot = state.candidateSlot;
        clearCandidateOwnershipLocked(state);
        state.lastStatus = MetalNumanXHumanIOStatus::success;
        state.lastMessage.clear();
        return Disposition::released;
    } catch (...) {
        // No mutation is authorized after an unexpected host failure. The
        // enclosing joint owner must retain physical visibility quarantine.
        return Disposition::terminalNoTouch;
    }
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationDisposition
rejectCandidateCallback(
    void* context,
    const std::uint64_t candidateFingerprint
) noexcept {
    using Disposition =
        MetalNumanXHumanIOCandidatePublicationDisposition;
    if (context == nullptr || candidateFingerprint == 0u) {
        return Disposition::terminalNoTouch;
    }
    State& state = *static_cast<State*>(context);
    try {
        const std::lock_guard lock(state.mutex);
        if (!state.candidatePrepared || state.candidateSlot < 0 ||
            !state.candidatePublicationLeased ||
            state.candidatePublicationTerminal ||
            state.rootPublicationReserved ||
            state.candidatePublicationFingerprint != candidateFingerprint ||
            !state.commandBufferCompleted) {
            return Disposition::terminalNoTouch;
        }
        clearCandidateOwnershipLocked(state);
        state.lastStatus = MetalNumanXHumanIOStatus::success;
        state.lastMessage.clear();
        return Disposition::rejected;
    } catch (...) {
        return Disposition::terminalNoTouch;
    }
}

[[nodiscard]] MetalNumanXHumanIOCandidatePublicationProgram
makeCandidatePublicationProgram(
    State& state,
    const Slot& slot,
    const std::uint64_t fingerprint
) noexcept {
    MetalNumanXHumanIOCandidatePublicationProgram program{};
    program.context = &state;
    program.reservePublishedRoot = &reservePublishedRootCallback;
    program.publishCandidate = &publishCandidateCallback;
    program.rejectCandidate = &rejectCandidateCallback;
    program.candidateKeyFingerprint = candidateKeyFingerprint(makeKey(slot));
    program.transactionFingerprint = slot.transactionFingerprint;
    program.acceptedBrainGeneration = slot.acceptedBrainGeneration;
    program.sensorGeneration = slot.sensorGeneration;
    program.humanIOProgramFingerprint = slot.programFingerprint;
    program.sensorFingerprint = slot.sensorFingerprint;
    program.transactionInstanceFingerprint =
        slot.transactionInstanceFingerprint;
    program.candidatePublicationFingerprint = fingerprint;
    program.deviceRegistryID = state.device != nil
        ? state.device.registryID
        : 0u;
    program.identityFingerprint =
        metalNumanXHumanIOCandidatePublicationIdentityFingerprint(program);
    return program;
}

void clearCandidateOwnershipLocked(State& state) noexcept {
    state.candidateSlot = -1;
    state.candidateInput = {};
    state.candidatePrepared = false;
    state.encodingStarted = false;
    state.phasesComplete = false;
    state.completionHandlerInstalled = false;
    state.commandBufferCompleted = false;
    state.commandBufferSucceeded = false;
    state.candidateCompletionRegistered = false;
    state.candidateCompletionDelivered = false;
    state.candidateCompletionContext = nullptr;
    state.candidateCompletion = nullptr;
    state.candidateQuarantined = false;
    state.candidatePublicationLeased = false;
    state.rootPublicationReserved = false;
    state.candidatePublicationTerminal = false;
    state.abortHandled = false;
    state.candidatePublicationFingerprint = 0u;
    state.publicationBinding = {};
    state.expectedStep = 0u;
    state.expectedPhase = MetalNumanXTransactionPhase::beginStep;
    state.activeCommandBufferIdentity = 0u;
    state.stateBufferIdentity = 0u;
    state.resultBufferIdentity = 0u;
    state.standStatusBufferIdentity = 0u;
}

[[nodiscard]] MetalNumanXHumanIODiagnostics initializeLocked(State& state) {
    if (state.initialized) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::success
        );
    }
    if (state.config.metallibPath.empty()) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidConfiguration,
            "MetalNumanXHumanIOConfig.metallibPath must be explicit and nonempty"
        );
    }
    if (state.config.maximumRetainedBytes == 0u) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidConfiguration,
            "MetalNumanXHumanIOConfig.maximumRetainedBytes must be nonzero"
        );
    }

    @autoreleasepool {
        state.device = MTLCreateSystemDefaultDevice();
        if (state.device == nil) {
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metalDeviceUnavailable,
                "no system-default Metal device is available"
            );
        }

        NSString* path = [NSString
            stringWithUTF8String:state.config.metallibPath.c_str()];
        if (path == nil) {
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::invalidConfiguration,
                "metallibPath is not valid UTF-8"
            );
        }
        BOOL isDirectory = NO;
        if (![[NSFileManager defaultManager]
                fileExistsAtPath:path
                     isDirectory:&isDirectory] || isDirectory) {
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metallibUnavailable,
                "metallibPath does not name a readable regular file: " +
                    state.config.metallibPath
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
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metallibUnavailable,
                "failed to read the explicit NumanX Human IO metallib image: " +
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
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metallibUnavailable,
                "failed to retain the NumanX Human IO metallib image"
            );
        }
        NSError* error = nil;
        state.library = [state.device
            newLibraryWithData:libraryImage error:&error];
        if (state.library == nil) {
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metalLibraryFailure,
                "failed to load explicit NumanX Human IO metallib: " +
                    fromNSString(error.localizedDescription)
            );
        }
        state.metallibIdentity = imageIdentity;

        const auto makePipeline = [&](
            NSString* functionName,
            __strong id<MTLComputePipelineState>& pipeline
        ) -> MetalNumanXHumanIODiagnostics {
            id<MTLFunction> function =
                [state.library newFunctionWithName:functionName];
            if (function == nil) {
                return diagnosticsLocked(
                    state,
                    MetalNumanXHumanIOStatus::metalPipelineFailure,
                    "explicit metallib is missing required kernel " +
                        fromNSString(functionName)
                );
            }
            NSError* pipelineError = nil;
            pipeline = [state.device
                newComputePipelineStateWithFunction:function
                                               error:&pipelineError];
            if (pipeline == nil) {
                return diagnosticsLocked(
                    state,
                    MetalNumanXHumanIOStatus::metalPipelineFailure,
                    "failed to create pipeline for " +
                        fromNSString(functionName) + ": " +
                        fromNSString(pipelineError.localizedDescription)
                );
            }
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::success
            );
        };

        MetalNumanXHumanIODiagnostics result = makePipeline(
            @"numanx_human_validate_motor_output",
            state.validateMotorHeaderPipeline
        );
        if (!result.succeeded()) {
            return result;
        }
        result = makePipeline(
            @"numanx_human_admit_excitations",
            state.admitPipeline
        );
        if (!result.succeeded()) {
            return result;
        }
        result = makePipeline(
            @"numanx_human_gate_proprioception",
            state.gatePipeline
        );
        if (!result.succeeded()) {
            return result;
        }
        result = makePipeline(
            @"numanx_human_write_proprioception",
            state.writePipeline
        );
        if (!result.succeeded()) {
            return result;
        }
    }

    MetalNumanXHumanIODiagnostics result = diagnosticsLocked(
        state, MetalNumanXHumanIOStatus::success);
    state.initialized = true;
    return result;
}

[[nodiscard]] bool bufferObject(
    void* raw,
    __unsafe_unretained id<MTLBuffer>& buffer,
    std::string& reason,
    const char* label
) {
    if (raw == nullptr) {
        reason = std::string(label) + " is null";
        return false;
    }
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLBuffer)]) {
        reason = std::string(label) + " is not an id<MTLBuffer> object";
        return false;
    }
    buffer = (__bridge id<MTLBuffer>)raw;
    return true;
}

[[nodiscard]] bool validSharedEventLease(
    const State& state,
    void* raw,
    const std::uint64_t value,
    std::string& reason
) {
    if (raw == nullptr || value == 0u || state.device == nil) {
        reason = "motor-ready shared event or value is missing";
        return false;
    }
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLSharedEvent)]) {
        reason = "motor-ready object is not an id<MTLSharedEvent>";
        return false;
    }
    __unsafe_unretained id<MTLSharedEvent> event =
        (__bridge id<MTLSharedEvent>)raw;
    MTLSharedEventHandle* handle = [event newSharedEventHandle];
    if (handle == nil ||
        [state.device newSharedEventWithHandle:handle] == nil) {
        reason = "motor-ready event is not importable on the Human device";
        return false;
    }
    return true;
}

[[nodiscard]] bool commandBufferObject(
    void* raw,
    __unsafe_unretained id<MTLCommandBuffer>& commandBuffer,
    std::string& reason
) {
    if (raw == nullptr) {
        reason = "transaction pass commandBuffer is null";
        return false;
    }
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLCommandBuffer)]) {
        reason = "transaction pass commandBuffer is not an id<MTLCommandBuffer> object";
        return false;
    }
    commandBuffer = (__bridge id<MTLCommandBuffer>)raw;
    return true;
}

[[nodiscard]] bool validJointTransaction(
    const MRNumanXBrainJointTransactionToken& root,
    std::string& reason
) noexcept {
    if (root.formatVersion != MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION) {
        reason = "NumiBrain root transaction format version is incompatible";
        return false;
    }
    if (root.parameterVersionFingerprint == 0u) {
        reason = "NumiBrain root parameter-version identity is zero";
        return false;
    }
    if (root.targetTimestampMicroseconds <=
            root.committedTimestampMicroseconds) {
        reason = "NumiBrain root timestamp order is invalid";
        return false;
    }
    if (root.baseBrainGeneration ==
            std::numeric_limits<std::uint64_t>::max() ||
        root.shadowGeneration != root.baseBrainGeneration + 1u) {
        reason = "NumiBrain root brain generations are not consecutive";
        return false;
    }
    if (root.flags != 0u || root.reserved != 0u) {
        reason = "NumiBrain root flags or reserved field are nonzero";
        return false;
    }
    if (root.transactionFingerprint == 0u ||
        root.transactionFingerprint !=
            metalNumanXBrainJointTransactionFingerprint(root)) {
        reason = "NumiBrain root transaction fingerprint is invalid";
        return false;
    }
    return true;
}

[[nodiscard]] bool validJointSubstep(
    const MRNumanXBrainJointTransactionToken& root,
    const MRNumanXBrainJointSubstepToken& substep,
    std::string& reason
) noexcept {
    if (!validJointTransaction(root, reason)) return false;
    if (substep.transactionFingerprint != root.transactionFingerprint ||
        substep.shadowGeneration != root.shadowGeneration ||
        substep.randomCounterGeneration != root.randomCounterGeneration) {
        reason = "NumiBrain substep does not belong to the exact root generation";
        return false;
    }
    if (substep.durationMicroseconds == 0u ||
        substep.startTimestampMicroseconds <
            root.committedTimestampMicroseconds ||
        substep.startTimestampMicroseconds >=
            root.targetTimestampMicroseconds ||
        substep.durationMicroseconds >
            std::numeric_limits<std::uint64_t>::max() -
                substep.startTimestampMicroseconds ||
        substep.candidateTimestampMicroseconds !=
            substep.startTimestampMicroseconds +
                substep.durationMicroseconds ||
        substep.candidateTimestampMicroseconds >
            root.targetTimestampMicroseconds) {
        reason = "NumiBrain substep timing relation is invalid";
        return false;
    }
    if (substep.flags != 0u || substep.reserved != 0u) {
        reason = "NumiBrain substep flags or reserved field are nonzero";
        return false;
    }
    if (substep.substepFingerprint == 0u ||
        substep.substepFingerprint !=
            metalNumanXBrainJointSubstepFingerprint(substep)) {
        reason = "NumiBrain substep fingerprint is invalid";
        return false;
    }
    return true;
}

[[nodiscard]] bool validMotorCandidate(
    const MRNumanXBrainJointTransactionToken& root,
    const MRNumanXBrainJointSubstepToken& substep,
    const MRNumanXBrainMotorCandidate& candidate,
    std::string& reason
) noexcept {
    if (candidate.formatVersion !=
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION) {
        reason = "NumiBrain motor candidate format version is incompatible";
        return false;
    }
    constexpr std::uint32_t knownCandidateFlags =
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
        MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
    if ((candidate.flags & MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID) == 0u ||
        (candidate.flags & ~knownCandidateFlags) != 0u) {
        reason = "NumiBrain motor candidate flags are invalid";
        return false;
    }
    if (!validJointSubstep(root, substep, reason)) return false;
    if (candidate.transactionFingerprint != root.transactionFingerprint ||
        candidate.substepFingerprint != substep.substepFingerprint ||
        candidate.acceptedBrainTimestampMicroseconds !=
            substep.startTimestampMicroseconds ||
        candidate.randomCounterGeneration !=
            substep.randomCounterGeneration ||
        candidate.environmentIdentifier != root.environmentIdentifier ||
        candidate.motorProfileFingerprint == 0u ||
        candidate.speciesTemplateFingerprint == 0u ||
        candidate.compiledSpeciesTemplateFingerprint == 0u ||
        candidate.actuatorCommandKind < 1u ||
        candidate.actuatorCommandKind > 7u || candidate.reserved != 0u) {
        reason = "NumiBrain motor candidate identity/provenance relation is invalid";
        return false;
    }
    const bool decisionShadow =
        (candidate.flags &
         MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW) != 0u;
    if (decisionShadow && substep.substepIndex != 0u) {
        reason = "NumiBrain decision-shadow candidate is not the sole root substep";
        return false;
    }
    const std::uint64_t expectedGeneration = decisionShadow
        ? root.shadowGeneration
        : (substep.substepIndex == 0u
               ? root.baseBrainGeneration
               : root.shadowGeneration);
    if (candidate.brainGeneration != expectedGeneration) {
        reason = "NumiBrain motor candidate brain generation is invalid";
        return false;
    }
    if (candidate.motorOutputHeaderGPUAddress == 0u ||
        candidate.muscleExcitationGPUAddress == 0u ||
        candidate.autonomicCommandGPUAddress == 0u ||
        candidate.activeSensingCommandGPUAddress == 0u ||
        candidate.motorOutputHeaderGPUAddress % 8u != 0u ||
        candidate.muscleExcitationGPUAddress % 4u != 0u ||
        candidate.autonomicCommandGPUAddress % 4u != 0u ||
        candidate.activeSensingCommandGPUAddress % 4u != 0u) {
        reason = "NumiBrain motor candidate GPU address contract is invalid";
        return false;
    }
    const std::uint64_t excitationBytes =
        static_cast<std::uint64_t>(candidate.muscleCount) * sizeof(float);
    const std::uint64_t autonomicBytes =
        static_cast<std::uint64_t>(candidate.autonomicCommandCount) *
            MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT;
    const std::uint64_t activeSensingBytes =
        static_cast<std::uint64_t>(candidate.activeSensingCommandCount) *
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
    if (candidate.motorOutputHeaderByteCount !=
            MR_NUMANX_BRAIN_MOTOR_OUTPUT_HEADER_BYTE_COUNT ||
        candidate.muscleCount == 0u || excitationBytes > UINT32_MAX ||
        candidate.muscleExcitationByteCount != excitationBytes ||
        candidate.autonomicCommandCount == 0u ||
        autonomicBytes > UINT32_MAX ||
        candidate.autonomicCommandByteCount != autonomicBytes ||
        activeSensingBytes > UINT32_MAX ||
        candidate.activeSensingCommandByteCount != activeSensingBytes) {
        reason = "NumiBrain motor candidate resource sizes are invalid";
        return false;
    }
    if (candidate.candidateFingerprint == 0u ||
        candidate.candidateFingerprint !=
            metalNumanXBrainMotorCandidateFingerprint(candidate)) {
        reason = "NumiBrain motor candidate fingerprint is invalid";
        return false;
    }
    return true;
}

[[nodiscard]] bool validBorrowedLease(
    const State& state,
    void* raw,
    const std::size_t byteOffset,
    const std::size_t byteCount,
    const std::uint64_t expectedGPUAddress,
    const std::size_t alignment,
    const char* label,
    std::string& reason
) {
    __unsafe_unretained id<MTLBuffer> buffer = nil;
    if (!bufferObject(raw, buffer, reason, label)) return false;
    std::size_t end = 0u;
    const std::uint64_t base = static_cast<std::uint64_t>(buffer.gpuAddress);
    if (!sameDevice(buffer.device, state.device) ||
        alignment == 0u || byteOffset % alignment != 0u ||
        expectedGPUAddress % alignment != 0u ||
        !checkedAdd(byteOffset, byteCount, end) || end > buffer.length ||
        base == 0u ||
        byteOffset > std::numeric_limits<std::uint64_t>::max() - base ||
        base + byteOffset != expectedGPUAddress) {
        reason = std::string(label) +
            " device, alignment, slice, or exact GPU address is invalid";
        return false;
    }
    return true;
}

[[nodiscard]] bool slicesOverlap(
    void* first,
    const std::size_t firstOffset,
    const std::size_t firstCount,
    void* second,
    const std::size_t secondOffset,
    const std::size_t secondCount
) noexcept {
    if (first == nullptr || second == nullptr || first != second ||
        firstCount == 0u || secondCount == 0u) return false;
    return firstOffset < secondOffset + secondCount &&
        secondOffset < firstOffset + firstCount;
}

[[nodiscard]] bool gpuRangesDisjoint(
    const std::uint64_t firstAddress,
    const std::size_t firstCount,
    const std::uint64_t secondAddress,
    const std::size_t secondCount
) noexcept {
    if (firstAddress == 0u || secondAddress == 0u || firstCount == 0u ||
        secondCount == 0u ||
        firstCount > std::numeric_limits<std::uint64_t>::max() -
            firstAddress ||
        secondCount > std::numeric_limits<std::uint64_t>::max() -
            secondAddress) {
        return false;
    }
    const std::uint64_t firstEnd = firstAddress + firstCount;
    const std::uint64_t secondEnd = secondAddress + secondCount;
    return firstEnd <= secondAddress || secondEnd <= firstAddress;
}

[[nodiscard]] bool validCandidateLeases(
    const State& state,
    const MetalNumanXHumanIOInput& input,
    std::string& reason
) {
    const auto& candidate = input.candidate;
    const bool decisionShadow =
        (candidate.flags &
         MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW) != 0u;
    if (input.motorOutputHeaderByteCount !=
            candidate.motorOutputHeaderByteCount ||
        input.excitationByteCount != candidate.muscleExcitationByteCount ||
        input.autonomicCommandByteCount !=
            candidate.autonomicCommandByteCount ||
        input.activeSensingCommandByteCount !=
            candidate.activeSensingCommandByteCount ||
        input.expectedMotorOutputHeaderGPUAddress !=
            candidate.motorOutputHeaderGPUAddress ||
        input.expectedExcitationGPUAddress !=
            candidate.muscleExcitationGPUAddress ||
        input.expectedAutonomicCommandGPUAddress !=
            candidate.autonomicCommandGPUAddress ||
        input.expectedActiveSensingCommandGPUAddress !=
            candidate.activeSensingCommandGPUAddress) {
        reason = "borrowed lease slices do not exactly match motor-candidate addresses and byte counts";
        return false;
    }
    if (!validBorrowedLease(
            state, input.motorOutputHeaderMetalBuffer,
            input.motorOutputHeaderByteOffset,
            input.motorOutputHeaderByteCount,
            input.expectedMotorOutputHeaderGPUAddress,
            alignof(MRNumanXBrainMotorOutputHeaderGPU),
            "motorOutputHeaderMetalBuffer", reason) ||
        !validBorrowedLease(
            state, input.excitationMetalBuffer,
            input.excitationByteOffset, input.excitationByteCount,
            input.expectedExcitationGPUAddress, alignof(float),
            "excitationMetalBuffer", reason) ||
        !validBorrowedLease(
            state, input.autonomicCommandMetalBuffer,
            input.autonomicCommandByteOffset,
            input.autonomicCommandByteCount,
            input.expectedAutonomicCommandGPUAddress, alignof(float),
            "autonomicCommandMetalBuffer", reason) ||
        !validBorrowedLease(
            state, input.activeSensingCommandMetalBuffer,
            input.activeSensingCommandByteOffset,
            input.activeSensingCommandByteCount,
            input.expectedActiveSensingCommandGPUAddress, alignof(float),
            "activeSensingCommandMetalBuffer", reason)) {
        return false;
    }
    if (decisionShadow) {
        if (input.motorReadyGateByteCount !=
                MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT ||
            !validBorrowedLease(
                state,
                input.motorReadyGateMetalBuffer,
                input.motorReadyGateByteOffset,
                input.motorReadyGateByteCount,
                input.expectedMotorReadyGateGPUAddress,
                alignof(MRNumanXBrainMotorReadyGateGPU),
                "motorReadyGateMetalBuffer",
                reason) ||
            !validSharedEventLease(
                state,
                input.motorReadySharedEvent,
                input.motorReadySharedEventValue,
                reason)) {
            return false;
        }
    } else if (input.motorReadyGateMetalBuffer != nullptr ||
        input.motorReadyGateByteOffset != 0u ||
        input.motorReadyGateByteCount != 0u ||
        input.expectedMotorReadyGateGPUAddress != 0u ||
        input.motorReadySharedEvent != nullptr ||
        input.motorReadySharedEventValue != 0u) {
        reason = "legacy motor candidate carries decision-shadow gate authority";
        return false;
    }
    struct Slice {
        void* object;
        std::size_t offset;
        std::size_t count;
        std::uint64_t address;
    };
    const Slice slices[] = {
        {input.motorOutputHeaderMetalBuffer,
         input.motorOutputHeaderByteOffset,
         input.motorOutputHeaderByteCount,
         input.expectedMotorOutputHeaderGPUAddress},
        {input.excitationMetalBuffer, input.excitationByteOffset,
         input.excitationByteCount,
         input.expectedExcitationGPUAddress},
        {input.autonomicCommandMetalBuffer,
         input.autonomicCommandByteOffset,
         input.autonomicCommandByteCount,
         input.expectedAutonomicCommandGPUAddress},
        {input.activeSensingCommandMetalBuffer,
         input.activeSensingCommandByteOffset,
         input.activeSensingCommandByteCount,
         input.expectedActiveSensingCommandGPUAddress},
        {input.motorReadyGateMetalBuffer,
         input.motorReadyGateByteOffset,
         input.motorReadyGateByteCount,
         input.expectedMotorReadyGateGPUAddress},
    };
    for (std::size_t first = 0u; first < std::size(slices); ++first) {
        for (std::size_t second = first + 1u;
             second < std::size(slices); ++second) {
            if ((slices[first].count != 0u &&
                 slices[second].count != 0u &&
                 !gpuRangesDisjoint(
                     slices[first].address,
                     slices[first].count,
                     slices[second].address,
                     slices[second].count)) ||
                slicesOverlap(
                    slices[first].object, slices[first].offset,
                    slices[first].count, slices[second].object,
                    slices[second].offset, slices[second].count)) {
                reason = "borrowed motor-candidate lease slices overlap";
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] MetalNumanXHumanIODiagnostics validateInputLocked(
    State& state,
    const MetalNumanXHumanIOInput& input,
    std::size_t& proprioceptionByteCount,
    std::size_t& validityByteCount,
    std::size_t& interoceptionByteCount,
    std::size_t& interoceptionValidityByteCount,
    std::size_t& motorValidationByteCount,
    std::size_t& motorHeaderValidationByteCount,
    std::size_t& environmentGateByteCount,
    std::size_t& proprioceptionEnvironmentStride,
    std::size_t& proprioceptionStepStride,
    std::size_t& validityEnvironmentStride,
    std::size_t& validityStepStride
) {
    if (input.environmentCount != 1u || input.muscleCount == 0u ||
        input.stepCount == 0u ||
        input.stepCount > MR_NUMI_HUMAN_STAND_MAX_STEPS) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "the authoritative motor-candidate ABI is single-environment; muscleCount and stepCount must be nonzero and stepCount must fit Human"
        );
    }
    const double receptorTimeSeconds = static_cast<double>(
        input.receptorTimestampMicroseconds
    ) / 1'000'000.0;
    if (!(input.timestepSeconds > 0.0f) ||
        !std::isfinite(input.timestepSeconds) ||
        !std::isfinite(receptorTimeSeconds) ||
        !std::isfinite(
            receptorTimeSeconds +
                static_cast<double>(input.stepCount) *
                    input.timestepSeconds
        )) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "timestepSeconds must be finite and positive, and every horizon receptor/delivery time must be finite"
        );
    }
    if (input.candidateSensorGeneration == 0u) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "candidateSensorGeneration must be nonzero"
        );
    }
    std::string reason;
    if (!validMotorCandidate(
            input.root, input.substep, input.candidate, reason)) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            std::move(reason)
        );
    }
    if (input.candidate.actuatorCommandKind !=
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION ||
        input.candidate.muscleCount != input.muscleCount) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "Human accepts only candidate actuator kind 1 and an exact candidate/input muscle count"
        );
    }
    const double stepMicroseconds =
        static_cast<double>(input.timestepSeconds) * 1'000'000.0;
    if (!std::isfinite(stepMicroseconds) || stepMicroseconds < 1.0 ||
        stepMicroseconds >
            static_cast<double>(std::numeric_limits<long long>::max())) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "Human timestep cannot be represented as integral microseconds"
        );
    }
    const auto roundedStepMicroseconds =
        static_cast<std::uint64_t>(std::llround(stepMicroseconds));
    const float canonicalTimestepSeconds = static_cast<float>(
        roundedStepMicroseconds
    ) / 1'000'000.0f;
    if (input.timestepSeconds != canonicalTimestepSeconds ||
        roundedStepMicroseconds >
            std::numeric_limits<std::uint64_t>::max() / input.stepCount ||
        roundedStepMicroseconds * input.stepCount !=
            input.substep.durationMicroseconds ||
        input.receptorTimestampMicroseconds !=
            input.candidate.acceptedBrainTimestampMicroseconds) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "Human timestep must be the canonical integral-microsecond float, and its horizon/receptor time must exactly bind the validated NumiBrain substep"
        );
    }
    if (state.publishedSlot >= 0 &&
        input.candidateSensorGeneration <=
            state.slots[state.publishedSlot].sensorGeneration) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "candidateSensorGeneration must be strictly newer than the published generation"
        );
    }
    if (input.excitationEnvironmentStride != input.muscleCount) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "excitationEnvironmentStride must equal muscleCount exactly for an environment-major dense slice"
        );
    }
    if (input.motorOutputHeaderEnvironmentStride !=
            sizeof(MRNumanXBrainMotorOutputHeaderGPU)) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "motorOutputHeaderEnvironmentStride must equal the exact 80-byte brain header ABI"
        );
    }
    std::size_t excitationElementCount = 0u;
    std::size_t excitationByteCount = 0u;
    if (!checkedMultiply(
            input.environmentCount,
            input.muscleCount,
            excitationElementCount
        ) || !checkedMultiply(
            excitationElementCount,
            sizeof(float),
            excitationByteCount
        )) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::arithmeticOverflow,
            "excitation slice element/byte count overflow"
        );
    }
    if (excitationElementCount >
            std::numeric_limits<std::uint32_t>::max() ||
        input.excitationByteCount != excitationByteCount) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "excitationByteCount must equal the validated candidate muscleCount * sizeof(float) exactly"
        );
    }
    if (!validCandidateLeases(state, input, reason)) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            std::move(reason)
        );
    }

    if (!checkedMultiply(
            input.muscleCount,
            MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT,
            proprioceptionStepStride
        ) || !checkedMultiply(
            input.stepCount,
            proprioceptionStepStride,
            proprioceptionEnvironmentStride
        ) || !checkedMultiply(
            input.stepCount,
            input.muscleCount,
            validityEnvironmentStride
        )) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::arithmeticOverflow,
            "proprioception tensor stride overflow"
        );
    }
    validityStepStride = input.muscleCount;
    if (proprioceptionStepStride >
            std::numeric_limits<std::uint32_t>::max() ||
        proprioceptionEnvironmentStride >
            std::numeric_limits<std::uint32_t>::max() ||
        validityStepStride > std::numeric_limits<std::uint32_t>::max() ||
        validityEnvironmentStride >
            std::numeric_limits<std::uint32_t>::max()) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::invalidInput,
            "proprioception or validity stride exceeds the UInt32 GPU ABI"
        );
    }

    std::size_t proprioceptionElementCount = 0u;
    std::size_t validityElementCount = 0u;
    if (!checkedMultiply(
            input.environmentCount,
            proprioceptionEnvironmentStride,
            proprioceptionElementCount
        ) || !checkedMultiply(
            proprioceptionElementCount,
            sizeof(float),
            proprioceptionByteCount
        ) || !checkedMultiply(
            input.environmentCount,
            validityEnvironmentStride,
            validityElementCount
        ) || !checkedMultiply(
            validityElementCount,
            sizeof(std::uint32_t),
            validityByteCount
        ) || !checkedMultiply(
            validityElementCount,
            sizeof(float),
            interoceptionByteCount
        ) || !checkedMultiply(
            validityElementCount,
            sizeof(std::uint32_t),
            interoceptionValidityByteCount
        ) || !checkedMultiply(
            excitationElementCount,
            sizeof(std::uint32_t),
            motorValidationByteCount
        ) || !checkedMultiply(
            input.environmentCount,
            sizeof(std::uint32_t),
            motorHeaderValidationByteCount
        ) || !checkedMultiply(
            input.environmentCount,
            sizeof(std::uint32_t),
            environmentGateByteCount
        )) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::arithmeticOverflow,
            "output or scratch buffer byte-count overflow"
        );
    }
    return diagnosticsLocked(state, MetalNumanXHumanIOStatus::success);
}

[[nodiscard]] MetalNumanXHumanIODiagnostics ensureSlotLocked(
    State& state,
    const int slotIndex,
    const std::size_t proprioceptionByteCount,
    const std::size_t validityByteCount,
    const std::size_t interoceptionByteCount,
    const std::size_t interoceptionValidityByteCount,
    const std::size_t motorValidationByteCount,
    const std::size_t motorHeaderValidationByteCount,
    const std::size_t environmentGateByteCount
) {
    Slot& slot = state.slots[slotIndex];
    const bool exact = slot.proprioception != nil &&
        slot.validity != nil && slot.interoception != nil &&
        slot.interoceptionValidity != nil && slot.motorValidation != nil &&
        slot.motorHeaderValidation != nil &&
        slot.environmentGate != nil &&
        slot.proprioception.length == proprioceptionByteCount &&
        slot.validity.length == validityByteCount &&
        slot.interoception.length == interoceptionByteCount &&
        slot.interoceptionValidity.length ==
            interoceptionValidityByteCount &&
        slot.motorValidation.length == motorValidationByteCount &&
        slot.motorHeaderValidation.length ==
            motorHeaderValidationByteCount &&
        slot.environmentGate.length == environmentGateByteCount;
    if (exact) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::success
        );
    }

    std::size_t desiredBytes = 0u;
    std::size_t temporary = 0u;
    if (!checkedAdd(
            proprioceptionByteCount,
            validityByteCount,
            temporary
        ) || !checkedAdd(
            temporary,
            interoceptionByteCount,
            temporary
        ) || !checkedAdd(
            temporary,
            interoceptionValidityByteCount,
            temporary
        ) || !checkedAdd(
            temporary,
            motorValidationByteCount,
            temporary
        ) || !checkedAdd(
            temporary,
            motorHeaderValidationByteCount,
            temporary
        ) || !checkedAdd(
            temporary,
            environmentGateByteCount,
            desiredBytes
        )) {
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::arithmeticOverflow,
            "candidate slot retained-byte count overflow"
        );
    }
    const std::size_t otherBytes = state.slots[1 - slotIndex].retainedBytes();
    std::size_t proposedRetainedBytes = 0u;
    if (!checkedAdd(otherBytes, desiredBytes, proposedRetainedBytes) ||
        proposedRetainedBytes > state.config.maximumRetainedBytes) {
        std::ostringstream stream;
        stream << "candidate/private buffer request would retain "
               << proposedRetainedBytes << " bytes, exceeding configured "
               << state.config.maximumRetainedBytes << " bytes";
        return diagnosticsLocked(
            state,
            MetalNumanXHumanIOStatus::metalBufferFailure,
            stream.str()
        );
    }

    @autoreleasepool {
        id<MTLBuffer> proprioception = [state.device
            newBufferWithLength:proprioceptionByteCount
                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> validity = [state.device
            newBufferWithLength:validityByteCount
                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> interoception = [state.device
            newBufferWithLength:interoceptionByteCount
                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> interoceptionValidity = [state.device
            newBufferWithLength:interoceptionValidityByteCount
                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> motorValidation = [state.device
            newBufferWithLength:motorValidationByteCount
                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> motorHeaderValidation = [state.device
            newBufferWithLength:motorHeaderValidationByteCount
                       options:MTLResourceStorageModePrivate];
        id<MTLBuffer> environmentGate = [state.device
            newBufferWithLength:environmentGateByteCount
                       options:MTLResourceStorageModePrivate];
        if (proprioception == nil || validity == nil ||
            interoception == nil || interoceptionValidity == nil ||
            motorValidation == nil || motorHeaderValidation == nil ||
            environmentGate == nil) {
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metalBufferFailure,
                "failed to allocate exact private Metal buffers for the candidate sensor slot"
            );
        }
        if (proprioception.gpuAddress == 0u ||
            validity.gpuAddress == 0u ||
            interoception.gpuAddress == 0u ||
            interoceptionValidity.gpuAddress == 0u ||
            motorValidation.gpuAddress == 0u ||
            motorHeaderValidation.gpuAddress == 0u ||
            environmentGate.gpuAddress == 0u) {
            return diagnosticsLocked(
                state,
                MetalNumanXHumanIOStatus::metalBufferFailure,
                "Metal returned a zero GPU address for an adapter-owned private buffer"
            );
        }
        proprioception.label = @"NumanX Human candidate proprioception";
        validity.label = @"NumanX Human candidate validity";
        interoception.label = @"NumanX Human candidate interoception";
        interoceptionValidity.label =
            @"NumanX Human candidate interoception validity";
        motorValidation.label = @"NumanX Human motor validation";
        motorHeaderValidation.label =
            @"NumanX Human motor header validation";
        environmentGate.label = @"NumanX Human environment gate";

        MetalNumanXHumanIODiagnostics result = diagnosticsLocked(
            state, MetalNumanXHumanIOStatus::success);

        slot.proprioception = proprioception;
        slot.validity = validity;
        slot.interoception = interoception;
        slot.interoceptionValidity = interoceptionValidity;
        slot.motorValidation = motorValidation;
        slot.motorHeaderValidation = motorHeaderValidation;
        slot.environmentGate = environmentGate;
        slot.proprioceptionByteCount = proprioceptionByteCount;
        slot.validityByteCount = validityByteCount;
        slot.interoceptionByteCount = interoceptionByteCount;
        slot.interoceptionValidityByteCount =
            interoceptionValidityByteCount;
        slot.motorValidationByteCount = motorValidationByteCount;
        slot.motorHeaderValidationByteCount =
            motorHeaderValidationByteCount;
        slot.environmentGateByteCount = environmentGateByteCount;
        return result;
    }
}

[[nodiscard]] std::uint64_t programFingerprint(
    const State& state,
    const Slot& slot,
    const MetalNumanXHumanIOInput& input
) noexcept {
    std::uint64_t hash = hashString(
        kFnvOffset,
        "metalrobo.numanx-human-io.program.v5"
    );
    hash = hashValue(hash, state.metallibIdentity.byteFingerprint);
    hash = hashValue(hash, state.metallibIdentity.byteCount);
    hash = hashValue(hash, MR_NUMANX_HUMAN_IO_ABI_VERSION);
    hash = hashValue(hash, input.root.transactionFingerprint);
    hash = hashValue(hash, input.substep.substepFingerprint);
    hash = hashValue(hash, input.candidate.candidateFingerprint);
    hash = hashValue(hash, input.candidateSensorGeneration);
    hash = hashValue(hash, input.expectedExcitationGPUAddress);
    hash = hashValue(hash, input.expectedMotorOutputHeaderGPUAddress);
    hash = hashValue(hash, input.expectedAutonomicCommandGPUAddress);
    hash = hashValue(hash, input.expectedActiveSensingCommandGPUAddress);
    hash = hashValue(hash, input.expectedMotorReadyGateGPUAddress);
    hash = hashValue(hash, input.motorOutputHeaderByteCount);
    hash = hashValue(hash, input.excitationByteCount);
    hash = hashValue(hash, input.autonomicCommandByteCount);
    hash = hashValue(hash, input.activeSensingCommandByteCount);
    hash = hashValue(hash, input.motorReadyGateByteCount);
    hash = hashValue(
        hash,
        reinterpret_cast<std::uintptr_t>(input.motorReadySharedEvent));
    hash = hashValue(hash, input.motorReadySharedEventValue);
    hash = hashValue(hash, input.environmentCount);
    hash = hashValue(hash, input.muscleCount);
    hash = hashValue(hash, input.stepCount);
    hash = hashValue(hash, std::bit_cast<std::uint32_t>(input.timestepSeconds));
    hash = hashValue(hash, input.receptorTimestampMicroseconds);
    hash = hashValue(hash, slot.proprioception.gpuAddress);
    hash = hashValue(hash, slot.validity.gpuAddress);
    hash = hashValue(hash, slot.interoception.gpuAddress);
    hash = hashValue(hash, slot.interoceptionValidity.gpuAddress);
    return nonzeroHash(hash);
}

[[nodiscard]] std::uint64_t sensorFingerprint(
    const Slot& slot
) noexcept {
    std::uint64_t hash = hashString(
        kFnvOffset,
        "metalrobo.numanx-human-io.sensor.v4"
    );
    hash = hashValue(hash, slot.programFingerprint);
    hash = hashValue(hash, slot.transactionFingerprint);
    hash = hashValue(hash, slot.motorCandidateFingerprint);
    hash = hashValue(hash, slot.acceptedBrainGeneration);
    hash = hashValue(hash, slot.sensorGeneration);
    hash = hashValue(hash, slot.excitationGPUAddress);
    hash = hashValue(hash, slot.motorOutputHeaderGPUAddress);
    hash = hashValue(hash, slot.proprioception.gpuAddress);
    hash = hashValue(hash, slot.validity.gpuAddress);
    hash = hashValue(hash, slot.interoception.gpuAddress);
    hash = hashValue(hash, slot.interoceptionValidity.gpuAddress);
    hash = hashValue(hash, slot.proprioceptionByteCount);
    hash = hashValue(hash, slot.validityByteCount);
    hash = hashValue(hash, slot.interoceptionByteCount);
    hash = hashValue(hash, slot.interoceptionValidityByteCount);
    return nonzeroHash(hash);
}

[[nodiscard]] std::uint64_t transactionInstanceFingerprint(
    const Slot& slot,
    const std::uintptr_t commandBufferIdentity
) noexcept {
    std::uint64_t hash = hashString(
        kFnvOffset,
        "metalrobo.numanx-human-io.instance.v4"
    );
    hash = hashValue(hash, slot.sensorFingerprint);
    hash = hashValue(hash, slot.programFingerprint);
    hash = hashValue(hash, commandBufferIdentity);
    return nonzeroHash(hash);
}

[[nodiscard]] bool validatePassLocked(
    State& state,
    const MetalNumanXTransactionPass& pass,
    std::string& reason
) {
    const MetalNumanXHumanIOInput& input = state.candidateInput;
    const std::uint32_t expectedAccessFlags =
        MetalNumanXTransactionReadBorrowedState |
        (pass.phase == MetalNumanXTransactionPhase::beginStep
             ? MetalNumanXTransactionWriteMujocoExcitation
             : 0u) |
        (pass.phase == MetalNumanXTransactionPhase::postDynamics
             ? MetalNumanXTransactionWriteStandFailure
             : 0u);
    if (pass.abiVersion != kMetalNumanXTransactionABIVersion ||
        pass.structSize != sizeof(MetalNumanXTransactionPass) ||
        pass.reserved0 != 0u || pass.accessFlags != expectedAccessFlags) {
        reason = "transaction pass ABI, size, reserved field, or phase access contract is incompatible";
        return false;
    }
    if (pass.programFingerprint !=
        state.slots[state.candidateSlot].programFingerprint) {
        reason = "transaction pass programFingerprint does not match the prepared adapter program";
        return false;
    }
    if (pass.phase != state.expectedPhase ||
        pass.stepIndex != state.expectedStep) {
        std::ostringstream stream;
        stream << "transaction phase/order mismatch: expected phase "
               << static_cast<std::uint32_t>(state.expectedPhase)
               << " step " << state.expectedStep << ", received phase "
               << static_cast<std::uint32_t>(pass.phase)
               << " step " << pass.stepIndex;
        reason = stream.str();
        return false;
    }
    if (pass.stepCount != input.stepCount ||
        pass.environmentCount != input.environmentCount ||
        pass.mujocoMuscleCount != input.muscleCount ||
        pass.mujocoStateStride != input.muscleCount ||
        pass.mujocoResultStride != input.muscleCount ||
        pass.standStatusStride != 1u ||
        pass.timestepSeconds != input.timestepSeconds) {
        reason = "transaction pass environment/muscle/horizon/timestep or dense stride differs from the prepared exact layout";
        return false;
    }
    std::size_t expectedMuscleElements = 0u;
    if (!checkedMultiply(
            input.environmentCount,
            input.muscleCount,
            expectedMuscleElements
        ) || pass.mujocoStateElementCount != expectedMuscleElements ||
        pass.mujocoResultElementCount != expectedMuscleElements ||
        pass.standStatusElementCount != input.environmentCount) {
        reason = "transaction pass reports an inexact MyoSim state/result or stand-status logical length";
        return false;
    }

    __unsafe_unretained id<MTLCommandBuffer> commandBuffer = nil;
    if (!commandBufferObject(pass.commandBuffer, commandBuffer, reason)) {
        return false;
    }
    if (!sameDevice(commandBuffer.commandQueue.device, state.device)) {
        reason = "transaction command buffer belongs to a different Metal device";
        return false;
    }
    if (commandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
        reason = "transaction command buffer is already enqueued, committed, completed, or reused";
        return false;
    }

    struct BorrowedBuffer {
        const char* name;
        void* object;
    };
    const BorrowedBuffer borrowed[] = {
        {"q", pass.q},
        {"v", pass.v},
        {"bodyPoses", pass.bodyPoses},
        {"pointWorld", pass.pointWorld},
        {"pointJacobians", pass.pointJacobians},
        {"mujocoMuscles", pass.mujocoMuscles},
        {"mujocoStates", pass.mujocoStates},
        {"mujocoSites", pass.mujocoSites},
        {"mujocoWraps", pass.mujocoWraps},
        {"mujocoRouteNodes", pass.mujocoRouteNodes},
        {"mujocoResults", pass.mujocoResults},
        {"mujocoGeneralizedForceArena", pass.mujocoGeneralizedForceArena},
        {"tendonBindings", pass.tendonBindings},
        {"tendonEnvelopes", pass.tendonEnvelopes},
        {"tendonTransfers", pass.tendonTransfers},
        {"tendonGeneralizedCorrections", pass.tendonGeneralizedCorrections},
        {"standStatuses", pass.standStatuses},
    };
    for (const BorrowedBuffer& entry : borrowed) {
        if (entry.object == nullptr) {
            continue;
        }
        __unsafe_unretained id<MTLBuffer> buffer = nil;
        if (!bufferObject(
                entry.object,
                buffer,
                reason,
                entry.name
            )) {
            return false;
        }
        if (!sameDevice(buffer.device, state.device)) {
            reason = std::string(entry.name) +
                " belongs to a different Metal device";
            return false;
        }
    }

    __unsafe_unretained id<MTLBuffer> states = nil;
    __unsafe_unretained id<MTLBuffer> results = nil;
    __unsafe_unretained id<MTLBuffer> statuses = nil;
    if (!bufferObject(pass.mujocoStates, states, reason, "mujocoStates") ||
        !bufferObject(pass.mujocoResults, results, reason, "mujocoResults") ||
        !bufferObject(pass.standStatuses, statuses, reason, "standStatuses")) {
        return false;
    }
    std::size_t stateBytes = 0u;
    std::size_t resultBytes = 0u;
    std::size_t statusBytes = 0u;
    if (!checkedMultiply(
            expectedMuscleElements,
            sizeof(MRMujocoMuscleStateGPU),
            stateBytes
        ) || !checkedMultiply(
            expectedMuscleElements,
            sizeof(MRMujocoMuscleResultGPU),
            resultBytes
        ) || !checkedMultiply(
            input.environmentCount,
            sizeof(MRNumiHumanStandStatusGPU),
            statusBytes
        ) || states.length < stateBytes || results.length < resultBytes ||
        statuses.length < statusBytes) {
        reason = "borrowed MyoSim/status MTLBuffer capacity is smaller than its exact reported logical layout";
        return false;
    }

    const std::uintptr_t stateIdentity =
        reinterpret_cast<std::uintptr_t>(pass.mujocoStates);
    const std::uintptr_t resultIdentity =
        reinterpret_cast<std::uintptr_t>(pass.mujocoResults);
    const std::uintptr_t statusIdentity =
        reinterpret_cast<std::uintptr_t>(pass.standStatuses);
    if (state.stateBufferIdentity == 0u) {
        state.stateBufferIdentity = stateIdentity;
        state.resultBufferIdentity = resultIdentity;
        state.standStatusBufferIdentity = statusIdentity;
    } else if (state.stateBufferIdentity != stateIdentity ||
        state.resultBufferIdentity != resultIdentity ||
        state.standStatusBufferIdentity != statusIdentity) {
        reason = "borrowed MyoSim/status buffer identity changed within one command-buffer transaction";
        return false;
    }

    return validMotorCandidate(
            input.root, input.substep, input.candidate, reason) &&
        validCandidateLeases(state, input, reason);
}

[[nodiscard]] NSUInteger threadgroupWidth(
    id<MTLComputePipelineState> pipeline
) noexcept {
    return std::max<NSUInteger>(
        1u,
        std::min<NSUInteger>(
            kPreferredThreadsPerThreadgroup,
            pipeline.maxTotalThreadsPerThreadgroup
        )
    );
}

void dispatchOneDimensional(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const NSUInteger count
) {
    const NSUInteger width = threadgroupWidth(pipeline);
    [encoder
        dispatchThreadgroups:MTLSizeMake(
            (count + width - 1u) / width,
            1u,
            1u
        )
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

[[nodiscard]] bool encodeBeginLocked(
    State& state,
    const MetalNumanXTransactionPass& pass,
    std::string& reason
) {
    Slot& slot = state.slots[state.candidateSlot];
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    __unsafe_unretained id<MTLBuffer> excitation =
        (__bridge id<MTLBuffer>)state.candidateInput.excitationMetalBuffer;
    __unsafe_unretained id<MTLBuffer> motorHeaders =
        (__bridge id<MTLBuffer>)
            state.candidateInput.motorOutputHeaderMetalBuffer;
    const bool decisionShadow =
        (state.candidateInput.candidate.flags &
         MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW) != 0u;
    __unsafe_unretained id<MTLBuffer> motorReadyGate = decisionShadow
        ? (__bridge id<MTLBuffer>)
              state.candidateInput.motorReadyGateMetalBuffer
        : motorHeaders;
    __unsafe_unretained id<MTLBuffer> states =
        (__bridge id<MTLBuffer>)pass.mujocoStates;

    if (decisionShadow) {
        __unsafe_unretained id<MTLSharedEvent> motorReadyEvent =
            (__bridge id<MTLSharedEvent>)
                state.candidateInput.motorReadySharedEvent;
        [commandBuffer
            encodeWaitForEvent:motorReadyEvent
                         value:state.candidateInput.motorReadySharedEventValue];
    }

    MRNumanXHumanMotorDispatchGPU dispatch{};
    dispatch.abiVersion = MR_NUMANX_HUMAN_IO_ABI_VERSION;
    dispatch.environmentCount = state.candidateInput.environmentCount;
    dispatch.muscleCount = state.candidateInput.muscleCount;
    dispatch.stateStride = state.candidateInput.muscleCount;
    dispatch.excitationEnvironmentStride = static_cast<mr_u32>(
        state.candidateInput.excitationEnvironmentStride
    );
    dispatch.stepIndex = pass.stepIndex;
    dispatch.stepCount = state.candidateInput.stepCount;
    dispatch.flags = state.candidateInput.candidate.flags;
    dispatch.motorOutputFormatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION;
    dispatch.actuatorCommandKind =
        state.candidateInput.candidate.actuatorCommandKind;
    dispatch.environmentIdentifierBase =
        state.candidateInput.candidate.environmentIdentifier;
    dispatch.headerEnvironmentStride = static_cast<mr_u32>(
        state.candidateInput.motorOutputHeaderEnvironmentStride /
            sizeof(MRNumanXBrainMotorOutputHeaderGPU)
    );
    dispatch.transactionFingerprint = slot.transactionFingerprint;
    dispatch.motorCandidateFingerprint = slot.motorCandidateFingerprint;
    dispatch.acceptedBrainGeneration = slot.acceptedBrainGeneration;
    dispatch.candidateSensorGeneration = slot.sensorGeneration;
    dispatch.expectedExcitationGPUAddress = slot.excitationGPUAddress;
    dispatch.expectedMotorOutputHeaderGPUAddress =
        slot.motorOutputHeaderGPUAddress;
    dispatch.acceptedBrainTimestampMicroseconds =
        state.candidateInput.candidate.acceptedBrainTimestampMicroseconds;
    dispatch.motorProfileFingerprint =
        state.candidateInput.candidate.motorProfileFingerprint;
    dispatch.programFingerprint = slot.programFingerprint;

    id<MTLComputeCommandEncoder> headerEncoder =
        [commandBuffer computeCommandEncoder];
    if (headerEncoder == nil) {
        reason = "failed to create beginStep motor-header validation encoder";
        return false;
    }
    headerEncoder.label = @"NumanX Human motor header validation";
    [headerEncoder
        setComputePipelineState:state.validateMotorHeaderPipeline];
    [headerEncoder
        setBuffer:motorHeaders
           offset:state.candidateInput.motorOutputHeaderByteOffset
          atIndex:0u];
    [headerEncoder
        setBuffer:excitation
           offset:state.candidateInput.excitationByteOffset
          atIndex:1u];
    [headerEncoder
        setBuffer:slot.motorHeaderValidation
           offset:0u
          atIndex:2u];
    [headerEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:3u];
    [headerEncoder
        setBuffer:motorReadyGate
           offset:decisionShadow
               ? state.candidateInput.motorReadyGateByteOffset
               : state.candidateInput.motorOutputHeaderByteOffset
          atIndex:4u];
    dispatchOneDimensional(
        headerEncoder,
        state.validateMotorHeaderPipeline,
        state.candidateInput.environmentCount
    );
    [headerEncoder endEncoding];

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        reason = "failed to create beginStep excitation admission encoder";
        return false;
    }
    encoder.label = @"NumanX Human excitation admission";
    [encoder setComputePipelineState:state.admitPipeline];
    [encoder
        setBuffer:excitation
           offset:state.candidateInput.excitationByteOffset
          atIndex:0u];
    [encoder setBuffer:states offset:0u atIndex:1u];
    [encoder setBuffer:slot.motorValidation offset:0u atIndex:2u];
    [encoder setBuffer:slot.motorHeaderValidation offset:0u atIndex:3u];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:4u];

    const NSUInteger count = static_cast<NSUInteger>(
        state.candidateInput.environmentCount
    ) * state.candidateInput.muscleCount;
    dispatchOneDimensional(encoder, state.admitPipeline, count);
    [encoder endEncoding];
    return true;
}

[[nodiscard]] MRNumanXHumanProprioceptionDispatchGPU sensorDispatch(
    const State& state,
    const Slot& slot,
    const MetalNumanXTransactionPass& pass
) noexcept {
    MRNumanXHumanProprioceptionDispatchGPU dispatch{};
    dispatch.abiVersion = MR_NUMANX_HUMAN_IO_ABI_VERSION;
    dispatch.environmentCount = state.candidateInput.environmentCount;
    dispatch.muscleCount = state.candidateInput.muscleCount;
    dispatch.featureCount =
        MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
    dispatch.stepIndex = pass.stepIndex;
    dispatch.stepCount = state.candidateInput.stepCount;
    dispatch.stateStride = state.candidateInput.muscleCount;
    dispatch.resultStride = state.candidateInput.muscleCount;
    dispatch.proprioceptionEnvironmentStride = static_cast<mr_u32>(
        slot.proprioceptionEnvironmentStride
    );
    dispatch.proprioceptionStepStride = static_cast<mr_u32>(
        slot.proprioceptionStepStride
    );
    dispatch.validityEnvironmentStride = static_cast<mr_u32>(
        slot.validityEnvironmentStride
    );
    dispatch.validityStepStride = static_cast<mr_u32>(
        slot.validityStepStride
    );
    dispatch.timestepSecondsAndReserved.x =
        state.candidateInput.timestepSeconds;
    dispatch.transactionFingerprint = slot.transactionFingerprint;
    dispatch.motorCandidateFingerprint = slot.motorCandidateFingerprint;
    dispatch.acceptedBrainGeneration = slot.acceptedBrainGeneration;
    dispatch.candidateSensorGeneration = slot.sensorGeneration;
    dispatch.expectedExcitationGPUAddress = slot.excitationGPUAddress;
    dispatch.programFingerprint = slot.programFingerprint;
    return dispatch;
}

[[nodiscard]] bool encodePostLocked(
    State& state,
    const MetalNumanXTransactionPass& pass,
    std::string& reason
) {
    Slot& slot = state.slots[state.candidateSlot];
    __unsafe_unretained id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    __unsafe_unretained id<MTLBuffer> states =
        (__bridge id<MTLBuffer>)pass.mujocoStates;
    __unsafe_unretained id<MTLBuffer> results =
        (__bridge id<MTLBuffer>)pass.mujocoResults;
    __unsafe_unretained id<MTLBuffer> statuses =
        (__bridge id<MTLBuffer>)pass.standStatuses;
    const MRNumanXHumanProprioceptionDispatchGPU dispatch =
        sensorDispatch(state, slot, pass);

    id<MTLComputeCommandEncoder> gateEncoder =
        [commandBuffer computeCommandEncoder];
    if (gateEncoder == nil) {
        reason = "failed to create postDynamics environment gate encoder";
        return false;
    }
    gateEncoder.label = @"NumanX Human proprioception acceptance gate";
    [gateEncoder setComputePipelineState:state.gatePipeline];
    [gateEncoder setBuffer:states offset:0u atIndex:0u];
    [gateEncoder setBuffer:results offset:0u atIndex:1u];
    [gateEncoder setBuffer:slot.motorValidation offset:0u atIndex:2u];
    [gateEncoder setBuffer:statuses offset:0u atIndex:3u];
    [gateEncoder setBuffer:slot.environmentGate offset:0u atIndex:4u];
    [gateEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:5u];
    dispatchOneDimensional(
        gateEncoder,
        state.gatePipeline,
        state.candidateInput.environmentCount
    );
    [gateEncoder endEncoding];

    // A separate encoder establishes an explicit device ordering boundary
    // between the environment-level gate/status writes and receptor rows.
    id<MTLComputeCommandEncoder> writeEncoder =
        [commandBuffer computeCommandEncoder];
    if (writeEncoder == nil) {
        reason = "failed to create postDynamics proprioception writer encoder";
        return false;
    }
    writeEncoder.label = @"NumanX Human proprioception candidate writer";
    [writeEncoder setComputePipelineState:state.writePipeline];
    [writeEncoder setBuffer:states offset:0u atIndex:0u];
    [writeEncoder setBuffer:results offset:0u atIndex:1u];
    [writeEncoder setBuffer:slot.environmentGate offset:0u atIndex:2u];
    [writeEncoder setBuffer:slot.proprioception offset:0u atIndex:3u];
    [writeEncoder setBuffer:slot.validity offset:0u atIndex:4u];
    [writeEncoder setBuffer:slot.interoception offset:0u atIndex:5u];
    [writeEncoder
        setBuffer:slot.interoceptionValidity offset:0u atIndex:6u];
    [writeEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:7u];
    const NSUInteger count = static_cast<NSUInteger>(
        state.candidateInput.environmentCount
    ) * state.candidateInput.muscleCount;
    dispatchOneDimensional(writeEncoder, state.writePipeline, count);
    [writeEncoder endEncoding];
    return true;
}

void installCompletionHandlerLocked(
    State& state,
    id<MTLCommandBuffer> commandBuffer
) {
    const std::shared_ptr<State> retainedState = state.shared_from_this();
    const int slotIndex = state.candidateSlot;
    const std::uintptr_t identity =
        state.slots[slotIndex].commandBufferIdentity;
    const std::uint64_t instanceFingerprint =
        state.slots[slotIndex].transactionInstanceFingerprint;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        const std::uintptr_t completedIdentity =
            reinterpret_cast<std::uintptr_t>((__bridge void*)completed);
        const bool succeeded =
            completed.status == MTLCommandBufferStatusCompleted &&
            completed.error == nil;
        CandidateCompletionInvocation invocation{};
        try {
            const std::lock_guard lock(retainedState->mutex);
            if (!retainedState->candidatePrepared ||
                retainedState->candidateSlot != slotIndex) {
                return;
            }
            Slot& completedSlot = retainedState->slots[slotIndex];
            if (completedIdentity != identity ||
                completedSlot.commandBufferIdentity != identity ||
                completedSlot.transactionInstanceFingerprint !=
                    instanceFingerprint) {
                return;
            }
            retainedState->commandBufferCompleted = true;
            retainedState->commandBufferSucceeded = succeeded;
            retainedState->candidateQuarantined = !succeeded;
            retainedState->activeCommandBufferIdentity = 0u;
            retainedState->encodingStarted = false;
            if (!succeeded) {
                // Diagnostics are best effort. Allocation or NSString
                // conversion failure must not escape the Metal callback and
                // skip the exact terminal-state transition/notification.
                try {
                    std::string failureMessage =
                        "NumanX Human IO command buffer terminated with Metal status " +
                        std::to_string(static_cast<unsigned long>(
                            completed.status));
                    if (completed.error != nil) {
                        failureMessage += ": " + fromNSString(
                            completed.error.localizedDescription);
                    }
                    rememberFailureLocked(
                        *retainedState,
                        MetalNumanXHumanIOStatus::commandBufferFailure,
                        std::move(failureMessage));
                } catch (...) {
                    retainedState->lastStatus =
                        MetalNumanXHumanIOStatus::commandBufferFailure;
                    retainedState->lastMessage.clear();
                }
            }
            invocation = takeCandidateCompletionLocked(*retainedState);
        } catch (...) {
            // The handler itself is a noexcept boundary. A second, allocation-
            // free settlement attempt preserves completion delivery if a
            // best-effort diagnostic path unexpectedly threw.
            try {
                const std::lock_guard lock(retainedState->mutex);
                if (retainedState->candidatePrepared &&
                    retainedState->candidateSlot == slotIndex) {
                    Slot& completedSlot =
                        retainedState->slots[slotIndex];
                    if (completedIdentity == identity &&
                        completedSlot.commandBufferIdentity == identity &&
                        completedSlot.transactionInstanceFingerprint ==
                            instanceFingerprint) {
                        retainedState->commandBufferCompleted = true;
                        retainedState->commandBufferSucceeded = succeeded;
                        retainedState->candidateQuarantined = !succeeded;
                        retainedState->activeCommandBufferIdentity = 0u;
                        retainedState->encodingStarted = false;
                        if (!succeeded) {
                            retainedState->lastStatus =
                                MetalNumanXHumanIOStatus::
                                    commandBufferFailure;
                            retainedState->lastMessage.clear();
                        }
                        invocation =
                            takeCandidateCompletionLocked(*retainedState);
                    }
                }
            } catch (...) {
                // A mutex failure is not recoverable here. Never allow it to
                // escape an Objective-C completion block.
            }
        }
        invokeCandidateCompletion(invocation);
    }];
    state.completionHandlerInstalled = true;
}

[[nodiscard]] bool encodeTransaction(
    void* context,
    const MetalNumanXTransactionPass& pass
) noexcept {
    if (context == nullptr) {
        return false;
    }
    State& state = *static_cast<State*>(context);
    try {
        const std::lock_guard lock(state.mutex);
        if (!state.candidatePrepared || state.candidateSlot < 0) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::candidateUnavailable,
                "transaction callback arrived without a prepared candidate"
            );
            return false;
        }
        if (state.phasesComplete) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "transaction program was reused after its final postDynamics phase"
            );
            return false;
        }
        Slot& slot = state.slots[state.candidateSlot];
        if (pass.programFingerprint != slot.programFingerprint) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "transaction callback program fingerprint is stale or belongs to another candidate"
            );
            return false;
        }
        if (pass.commandBuffer == nullptr) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "transaction callback supplied a null command buffer"
            );
            return false;
        }

        const std::uintptr_t identity =
            reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
        if (slot.commandBufferIdentity == 0u) {
            slot.commandBufferIdentity = identity;
            slot.transactionInstanceFingerprint =
                transactionInstanceFingerprint(slot, identity);
            state.activeCommandBufferIdentity = identity;
        } else if (slot.commandBufferIdentity != identity ||
            state.activeCommandBufferIdentity != identity) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "concurrent or reused command-buffer identity was offered to one prepared candidate"
            );
            return false;
        }

        std::string reason;
        __unsafe_unretained id<MTLCommandBuffer> commandBuffer = nil;
        if (!commandBufferObject(
                pass.commandBuffer,
                commandBuffer,
                reason
            )) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                std::move(reason)
            );
            return false;
        }
        if (!state.completionHandlerInstalled) {
            installCompletionHandlerLocked(state, commandBuffer);
        }
        state.encodingStarted = true;

        if (!validatePassLocked(state, pass, reason)) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                std::move(reason)
            );
            return false;
        }

        bool encoded = true;
        switch (pass.phase) {
            case MetalNumanXTransactionPhase::beginStep:
                encoded = encodeBeginLocked(state, pass, reason);
                break;
            case MetalNumanXTransactionPhase::preDynamics:
                // The adapter has no preDynamics write. Keeping the explicit
                // callback in the state machine proves ordering and prevents
                // phase omission/reuse from being accepted.
                break;
            case MetalNumanXTransactionPhase::postDynamics:
                encoded = encodePostLocked(state, pass, reason);
                break;
            default:
                encoded = false;
                reason = "transaction callback supplied an unknown phase";
                break;
        }
        if (!encoded) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::commandBufferFailure,
                std::move(reason)
            );
            return false;
        }

        if (pass.phase == MetalNumanXTransactionPhase::beginStep) {
            state.expectedPhase = MetalNumanXTransactionPhase::preDynamics;
        } else if (pass.phase ==
            MetalNumanXTransactionPhase::preDynamics) {
            state.expectedPhase = MetalNumanXTransactionPhase::postDynamics;
        } else if (pass.stepIndex + 1u < pass.stepCount) {
            ++state.expectedStep;
            state.expectedPhase = MetalNumanXTransactionPhase::beginStep;
        } else {
            state.phasesComplete = true;
        }
        return true;
    } catch (const std::exception& exception) {
        try {
            const std::lock_guard lock(state.mutex);
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::internalFailure,
                std::string("transaction callback threw: ") + exception.what()
            );
        } catch (...) {
        }
        return false;
    } catch (...) {
        try {
            const std::lock_guard lock(state.mutex);
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::internalFailure,
                "transaction callback threw an unknown exception"
            );
        } catch (...) {
        }
        return false;
    }
}

void abortTransaction(void* context, void* commandBuffer) noexcept {
    if (context == nullptr || commandBuffer == nullptr) {
        return;
    }
    State& state = *static_cast<State*>(context);
    try {
        const std::lock_guard lock(state.mutex);
        if (!state.candidatePrepared || state.candidateSlot < 0 ||
            state.commandBufferCompleted) {
            return;
        }
        Slot& slot = state.slots[state.candidateSlot];
        const std::uintptr_t identity =
            reinterpret_cast<std::uintptr_t>(commandBuffer);
        if (slot.commandBufferIdentity == 0u ||
            slot.commandBufferIdentity != identity || state.abortHandled) {
            return;
        }
        state.abortHandled = true;
        if (state.lastStatus == MetalNumanXHumanIOStatus::success) {
            rememberFailureLocked(
                state,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "enclosing Human command buffer was abandoned before commit; candidate was quarantined"
            );
        }
        // No command was committed, so the candidate slot can be made
        // reusable immediately. The prior published slot is untouched.
        clearCandidateOwnershipLocked(state);
    } catch (...) {
        // The ABI is noexcept/void. The enclosing owner still abandons the
        // uncommitted command buffer, so no candidate can be published.
    }
}

} // namespace

std::uint64_t metalNumanXBrainJointTransactionFingerprint(
    const MRNumanXBrainJointTransactionToken& token
) noexcept {
    std::uint64_t hash = kFnvOffset;
    hash = hashU32(hash, MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION);
    hash = hashU32(hash, token.formatVersion);
    hash = hashU32(hash, token.environmentIdentifier);
    hash = hashU64(hash, token.episodeIdentifier);
    hash = hashU64(hash, token.controlStepIdentifier);
    hash = hashU64(hash, token.parameterVersionFingerprint);
    hash = hashU64(hash, token.baseBrainGeneration);
    hash = hashU64(hash, token.basePhysicsGeneration);
    hash = hashU64(hash, token.committedTimestampMicroseconds);
    hash = hashU64(hash, token.targetTimestampMicroseconds);
    hash = hashU64(hash, token.shadowGeneration);
    hash = hashU64(hash, token.randomCounterGeneration);
    hash = hashU32(hash, token.flags);
    hash = hashU32(hash, token.reserved);
    return hash;
}

std::uint64_t metalNumanXBrainJointSubstepFingerprint(
    const MRNumanXBrainJointSubstepToken& token
) noexcept {
    std::uint64_t hash = kFnvOffset;
    hash = hashU32(hash, MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION);
    hash = hashU64(hash, token.transactionFingerprint);
    hash = hashU32(hash, token.substepIndex);
    hash = hashU32(hash, token.attemptIndex);
    hash = hashU64(hash, token.startTimestampMicroseconds);
    hash = hashU64(hash, token.durationMicroseconds);
    hash = hashU64(hash, token.candidateTimestampMicroseconds);
    hash = hashU64(hash, token.shadowGeneration);
    hash = hashU64(hash, token.randomCounterGeneration);
    hash = hashU32(hash, token.flags);
    hash = hashU32(hash, token.reserved);
    return hash;
}

std::uint64_t metalNumanXBrainMotorCandidateFingerprint(
    const MRNumanXBrainMotorCandidate& candidate
) noexcept {
    std::uint64_t hash = kFnvOffset;
    hash = hashU32(hash, MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION);
    hash = hashU32(hash, candidate.formatVersion);
    hash = hashU32(hash, candidate.flags);
    hash = hashU64(hash, candidate.transactionFingerprint);
    hash = hashU64(hash, candidate.substepFingerprint);
    hash = hashU64(hash, candidate.acceptedBrainTimestampMicroseconds);
    hash = hashU64(hash, candidate.brainGeneration);
    hash = hashU64(hash, candidate.motorProfileFingerprint);
    hash = hashU64(hash, candidate.motorOutputHeaderGPUAddress);
    hash = hashU64(hash, candidate.muscleExcitationGPUAddress);
    hash = hashU64(hash, candidate.randomCounterGeneration);
    hash = hashU32(hash, candidate.motorOutputHeaderByteCount);
    hash = hashU32(hash, candidate.muscleExcitationByteCount);
    hash = hashU32(hash, candidate.muscleCount);
    hash = hashU32(hash, candidate.environmentIdentifier);
    hash = hashU64(hash, candidate.autonomicCommandGPUAddress);
    hash = hashU32(hash, candidate.autonomicCommandByteCount);
    hash = hashU32(hash, candidate.autonomicCommandCount);
    hash = hashU64(hash, candidate.activeSensingCommandGPUAddress);
    hash = hashU32(hash, candidate.activeSensingCommandByteCount);
    hash = hashU32(hash, candidate.activeSensingCommandCount);
    hash = hashU32(hash, candidate.actuatorCommandKind);
    hash = hashU32(hash, candidate.reserved);
    hash = hashU64(hash, candidate.speciesTemplateFingerprint);
    hash = hashU64(hash, candidate.compiledSpeciesTemplateFingerprint);
    return hash;
}

std::uint64_t metalNumanXHumanIOPublicationBindingFingerprint(
    const MetalNumanXHumanIOCandidatePublicationBinding& binding
) noexcept {
    std::uint64_t hash = hashCString(
        kFnvOffset,
        "metalrobo.numanx-human-io.publication-binding.v1"
    );
    hash = hashU32(hash, binding.abiVersion);
    hash = hashU32(hash, binding.structSize);
    hash = hashU32(hash, binding.environmentCount);
    hash = hashU32(hash, binding.transactionSlot);
    hash = hashU32(hash, binding.stepIndex);
    hash = hashU32(hash, binding.substepIndex);
    hash = hashU32(hash, binding.physicsSubstepCount);
    hash = hashU32(hash, binding.controlStep);
    hash = hashU64(hash, binding.ownerProgramFingerprint);
    hash = hashU64(hash, binding.transactionFingerprint);
    hash = hashU64(hash, binding.linearizationEpoch);
    hash = hashU64(hash, binding.slotGeneration);
    hash = hashU64(hash, binding.physicsTokenFingerprint);
    hash = hashU64(hash, binding.proposalFingerprint);
    hash = hashU64(hash, binding.ackFingerprint);
    hash = hashU64(hash, binding.appliedDecisionFingerprint);
    hash = hashU64(hash, binding.jointCommitFingerprint);
    hash = hashU64(hash, binding.brainGeneration);
    hash = hashU64(hash, binding.candidateKeyFingerprint);
    hash = hashU64(hash, binding.acceptedBrainGeneration);
    hash = hashU64(hash, binding.sensorGeneration);
    hash = hashU64(hash, binding.humanIOProgramFingerprint);
    hash = hashU64(hash, binding.sensorFingerprint);
    hash = hashU64(hash, binding.transactionInstanceFingerprint);
    hash = hashU64(hash, binding.candidatePublicationFingerprint);
    hash = hashU64(hash, binding.deviceRegistryID);
    hash = hashU64(hash, binding.humanIOIdentityFingerprint);
    return nonzeroHash(hash);
}

std::uint64_t metalNumanXHumanIOCandidatePublicationIdentityFingerprint(
    const MetalNumanXHumanIOCandidatePublicationProgram& program
) noexcept {
    return program.computedIdentityFingerprint();
}

bool MetalNumanXHumanIOCandidatePublicationLease::valid() const noexcept {
    if (state_ == nullptr || !program_.valid() ||
        view_.candidatePublicationFingerprint !=
            program_.candidatePublicationFingerprint) {
        return false;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        return state_->candidatePrepared && state_->candidateSlot >= 0 &&
            state_->candidatePublicationLeased &&
            !state_->candidatePublicationTerminal &&
            state_->candidatePublicationFingerprint ==
                program_.candidatePublicationFingerprint;
    } catch (...) {
        return false;
    }
}

MetalNumanXHumanIOContext::MetalNumanXHumanIOContext(
    MetalNumanXHumanIOConfig config
)
    : state_(std::make_shared<detail::MetalNumanXHumanIOState>(
          std::move(config)
      )) {}

MetalNumanXHumanIOContext::~MetalNumanXHumanIOContext() = default;

MetalNumanXHumanIOContext::MetalNumanXHumanIOContext(
    MetalNumanXHumanIOContext&& other
) noexcept = default;

MetalNumanXHumanIOContext& MetalNumanXHumanIOContext::operator=(
    MetalNumanXHumanIOContext&& other
) noexcept = default;

MetalNumanXHumanIODiagnostics MetalNumanXHumanIOContext::prepare(
    const MetalNumanXHumanIOInput& input,
    MetalNumanXTransactionProgram& program,
    MetalNumanXHumanIOSensorView& candidateView
) {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (state_->candidatePrepared) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::contextBusy,
                "a candidate transaction is already prepared or awaiting explicit accept/reject"
            );
        }
        MetalNumanXHumanIODiagnostics result = initializeLocked(*state_);
        if (!result.succeeded()) {
            return result;
        }

        std::size_t proprioceptionByteCount = 0u;
        std::size_t validityByteCount = 0u;
        std::size_t interoceptionByteCount = 0u;
        std::size_t interoceptionValidityByteCount = 0u;
        std::size_t motorValidationByteCount = 0u;
        std::size_t motorHeaderValidationByteCount = 0u;
        std::size_t environmentGateByteCount = 0u;
        std::size_t proprioceptionEnvironmentStride = 0u;
        std::size_t proprioceptionStepStride = 0u;
        std::size_t validityEnvironmentStride = 0u;
        std::size_t validityStepStride = 0u;
        result = validateInputLocked(
            *state_,
            input,
            proprioceptionByteCount,
            validityByteCount,
            interoceptionByteCount,
            interoceptionValidityByteCount,
            motorValidationByteCount,
            motorHeaderValidationByteCount,
            environmentGateByteCount,
            proprioceptionEnvironmentStride,
            proprioceptionStepStride,
            validityEnvironmentStride,
            validityStepStride
        );
        if (!result.succeeded()) {
            return result;
        }

        const int slotIndex = state_->publishedSlot >= 0
            ? 1 - state_->publishedSlot
            : 0;
        Slot& slot = state_->slots[slotIndex];
        if (input.excitationMetalBuffer ==
                (__bridge void*)slot.proprioception ||
            input.excitationMetalBuffer == (__bridge void*)slot.validity ||
            input.excitationMetalBuffer ==
                (__bridge void*)slot.interoception ||
            input.excitationMetalBuffer ==
                (__bridge void*)slot.interoceptionValidity ||
            input.excitationMetalBuffer ==
                (__bridge void*)slot.motorValidation ||
            input.excitationMetalBuffer ==
                (__bridge void*)slot.motorHeaderValidation ||
            input.excitationMetalBuffer ==
                (__bridge void*)slot.environmentGate ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.proprioception ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.validity ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.interoception ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.interoceptionValidity ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.motorValidation ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.motorHeaderValidation ||
            input.motorOutputHeaderMetalBuffer ==
                (__bridge void*)slot.environmentGate ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.proprioception ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.validity ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.interoception ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.interoceptionValidity ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.motorValidation ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.motorHeaderValidation ||
            input.autonomicCommandMetalBuffer ==
                (__bridge void*)slot.environmentGate ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.proprioception ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.validity ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.interoception ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.interoceptionValidity ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.motorValidation ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.motorHeaderValidation ||
            input.activeSensingCommandMetalBuffer ==
                (__bridge void*)slot.environmentGate) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::invalidInput,
                "borrowed motor input buffers may not alias adapter-owned candidate output or scratch storage"
            );
        }
        result = ensureSlotLocked(
            *state_,
            slotIndex,
            proprioceptionByteCount,
            validityByteCount,
            interoceptionByteCount,
            interoceptionValidityByteCount,
            motorValidationByteCount,
            motorHeaderValidationByteCount,
            environmentGateByteCount
        );
        if (!result.succeeded()) {
            return result;
        }

        slot.proprioceptionEnvironmentStride =
            proprioceptionEnvironmentStride;
        slot.proprioceptionStepStride = proprioceptionStepStride;
        slot.validityEnvironmentStride = validityEnvironmentStride;
        slot.validityStepStride = validityStepStride;
        slot.environmentCount = input.environmentCount;
        slot.stepCount = input.stepCount;
        slot.receptorCount = input.muscleCount;
        slot.timestepSeconds = input.timestepSeconds;
        slot.receptorTimestampMicroseconds =
            input.receptorTimestampMicroseconds;
        slot.transactionFingerprint = input.root.transactionFingerprint;
        slot.motorCandidateFingerprint = input.candidate.candidateFingerprint;
        slot.acceptedBrainGeneration = input.candidate.brainGeneration;
        slot.sensorGeneration = input.candidateSensorGeneration;
        slot.excitationGPUAddress = input.expectedExcitationGPUAddress;
        slot.motorOutputHeaderGPUAddress =
            input.expectedMotorOutputHeaderGPUAddress;
        slot.commandBufferIdentity = 0u;
        slot.transactionInstanceFingerprint = 0u;
        slot.programFingerprint = programFingerprint(
            *state_,
            slot,
            input
        );
        slot.sensorFingerprint = sensorFingerprint(slot);

        MetalNumanXHumanIODiagnostics success = diagnosticsLocked(
            *state_, MetalNumanXHumanIOStatus::success);
        success.encoded = false;
        success.commandBufferCompleted = false;
        success.commandBufferSucceeded = false;
        success.published = false;
        success.transactionFingerprint = slot.transactionFingerprint;
        success.programFingerprint = slot.programFingerprint;
        success.sensorGeneration = slot.sensorGeneration;
        success.commandBufferIdentity = 0u;

        state_->candidateSlot = slotIndex;
        state_->candidateInput = input;
        state_->candidatePrepared = true;
        state_->encodingStarted = false;
        state_->phasesComplete = false;
        state_->completionHandlerInstalled = false;
        state_->commandBufferCompleted = false;
        state_->commandBufferSucceeded = false;
        state_->candidateCompletionRegistered = false;
        state_->candidateCompletionDelivered = false;
        state_->candidateCompletionContext = nullptr;
        state_->candidateCompletion = nullptr;
        state_->candidateQuarantined = false;
        state_->candidatePublicationLeased = false;
        state_->rootPublicationReserved = false;
        state_->candidatePublicationTerminal = false;
        state_->abortHandled = false;
        state_->candidatePublicationFingerprint = 0u;
        state_->publicationBinding = {};
        state_->expectedStep = 0u;
        state_->expectedPhase = MetalNumanXTransactionPhase::beginStep;
        state_->activeCommandBufferIdentity = 0u;
        state_->stateBufferIdentity = 0u;
        state_->resultBufferIdentity = 0u;
        state_->standStatusBufferIdentity = 0u;
        state_->lastStatus = MetalNumanXHumanIOStatus::success;
        state_->lastMessage.clear();

        MetalNumanXTransactionProgram stagedProgram{};
        stagedProgram.context = state_.get();
        stagedProgram.encode = &encodeTransaction;
        stagedProgram.abort = &abortTransaction;
        stagedProgram.fingerprint = slot.programFingerprint;
        MetalNumanXHumanIOSensorView stagedView = makeView(
            slot,
            MetalNumanXHumanIOViewState::candidate
        );
        program = stagedProgram;
        candidateView = stagedView;
        return success;
    } catch (const std::bad_alloc&) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::metalBufferFailure;
        result.message = "host allocation failed while preparing the NumanX Human IO adapter";
        return result;
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    }
}

MetalNumanXHumanIODiagnostics
MetalNumanXHumanIOContext::pendingCandidate(
    MetalNumanXHumanIOTransactionKey& key,
    MetalNumanXHumanIOSensorView& candidateView
) const {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!state_->candidatePrepared || state_->candidateSlot < 0) {
            const MetalNumanXHumanIOStatus status =
                state_->lastStatus == MetalNumanXHumanIOStatus::success
                ? MetalNumanXHumanIOStatus::candidateUnavailable
                : state_->lastStatus;
            return diagnosticsLocked(
                *state_,
                status,
                state_->lastMessage.empty()
                    ? "no candidate transaction is pending"
                    : state_->lastMessage
            );
        }
        if (!state_->phasesComplete) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateNotEncoded,
                "candidate has not encoded every beginStep/preDynamics/postDynamics phase"
            );
        }
        const Slot& slot = state_->slots[state_->candidateSlot];
        const MetalNumanXHumanIOTransactionKey stagedKey = makeKey(slot);
        if (!stagedKey.valid()) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::internalFailure,
                "encoded candidate does not have a complete command-buffer-bound identity key"
            );
        }
        key = stagedKey;
        candidateView = makeView(
            slot,
            MetalNumanXHumanIOViewState::candidate
        );
        if (state_->commandBufferCompleted &&
            !state_->commandBufferSucceeded) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::commandBufferFailure,
                state_->lastMessage.empty()
                    ? "candidate command buffer failed and remains quarantined until reject()"
                    : state_->lastMessage
            );
        }
        if (state_->candidateQuarantined) {
            return diagnosticsLocked(
                *state_,
                state_->lastStatus == MetalNumanXHumanIOStatus::success
                    ? MetalNumanXHumanIOStatus::humanDiagnosticsRejected
                    : state_->lastStatus,
                state_->lastMessage.empty()
                    ? "candidate is quarantined and requires explicit reject()"
                    : state_->lastMessage
            );
        }
        return diagnosticsLocked(
            *state_,
            MetalNumanXHumanIOStatus::success
        );
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    }
}

MetalNumanXHumanIODiagnostics
MetalNumanXHumanIOContext::registerCandidateCompletion(
    const MetalNumanXHumanIOTransactionKey& key,
    void* completionContext,
    const MetalNumanXHumanIOCandidateCompletion completion
) noexcept {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    if (completionContext == nullptr || completion == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::invalidInput;
        result.message = "candidate completion requires a callback and context";
        return result;
    }
    try {
        const std::shared_ptr<State> retainedState = state_;
        CandidateCompletionInvocation invocation{};
        MetalNumanXHumanIODiagnostics result{};
        {
            const std::lock_guard lock(retainedState->mutex);
            if (!retainedState->candidatePrepared ||
                retainedState->candidateSlot < 0) {
                return diagnosticsLocked(
                    *retainedState,
                    MetalNumanXHumanIOStatus::candidateUnavailable,
                    "no candidate is available for completion registration");
            }
            const Slot& slot =
                retainedState->slots[retainedState->candidateSlot];
            if (!keyMatches(key, slot)) {
                return diagnosticsLocked(
                    *retainedState,
                    MetalNumanXHumanIOStatus::incompatibleTransaction,
                    "candidate completion key does not match the active command-buffer generation");
            }
            if (!retainedState->phasesComplete ||
                !retainedState->completionHandlerInstalled) {
                return diagnosticsLocked(
                    *retainedState,
                    MetalNumanXHumanIOStatus::candidateNotEncoded,
                    "candidate completion cannot register before all phases and the owning handler are installed");
            }
            if (retainedState->candidateCompletionRegistered ||
                retainedState->candidateCompletionDelivered) {
                return diagnosticsLocked(
                    *retainedState,
                    MetalNumanXHumanIOStatus::contextBusy,
                    "candidate completion is already registered or delivered");
            }
            // Build every fallible diagnostic string before transferring the
            // raw callback/context authority. Once armed, the remaining
            // commit/take path is allocation-free and the function must
            // return success even when it invokes synchronously.
            result = diagnosticsLocked(
                *retainedState, MetalNumanXHumanIOStatus::success);
            retainedState->candidateCompletionRegistered = true;
            retainedState->candidateCompletionContext = completionContext;
            retainedState->candidateCompletion = completion;
            invocation = takeCandidateCompletionLocked(*retainedState);
        }
        invokeCandidateCompletion(invocation);
        return result;
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    } catch (...) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "unexpected candidate completion registration failure";
        return result;
    }
}

MetalNumanXHumanIODiagnostics MetalNumanXHumanIOContext::cancelPrepared(
    const std::uint64_t transactionFingerprint,
    const std::uint64_t programFingerprint
) {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!state_->candidatePrepared || state_->candidateSlot < 0) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateUnavailable,
                "no prepared candidate is available for cancellation"
            );
        }
        const Slot& slot = state_->slots[state_->candidateSlot];
        if (transactionFingerprint == 0u || programFingerprint == 0u ||
            slot.transactionFingerprint != transactionFingerprint ||
            slot.programFingerprint != programFingerprint) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "cancelPrepared fingerprints do not match the prepared candidate"
            );
        }
        if (state_->encodingStarted || state_->phasesComplete ||
            slot.commandBufferIdentity != 0u) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateInFlight,
                "cancelPrepared cannot cancel a candidate after any command-buffer phase is offered"
            );
        }
        MetalNumanXHumanIODiagnostics result = diagnosticsLocked(
            *state_,
            MetalNumanXHumanIOStatus::success
        );
        clearCandidateOwnershipLocked(*state_);
        state_->lastStatus = MetalNumanXHumanIOStatus::success;
        state_->lastMessage.clear();
        return result;
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    }
}

MetalNumanXHumanIODiagnostics
MetalNumanXHumanIOContext::reserveCandidatePublication(
    const MetalNumanXHumanIOTransactionKey& key,
    MetalNumanXHumanIOCandidatePublicationLease& lease
) {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!state_->candidatePrepared || state_->candidateSlot < 0) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateUnavailable,
                "no candidate is available for publication reservation"
            );
        }
        Slot& slot = state_->slots[state_->candidateSlot];
        if (!keyMatches(key, slot)) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "publication key does not match the candidate transaction, command-buffer identity, generation, and fingerprints"
            );
        }
        if (lease.state_ != nullptr) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::contextBusy,
                "output publication lease already owns an active candidate capability"
            );
        }
        if (!state_->phasesComplete ||
            !state_->completionHandlerInstalled) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateNotEncoded,
                "candidate cannot be reserved before every transaction phase is encoded"
            );
        }
        if (!state_->commandBufferCompleted) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateInFlight,
                "candidate cannot be reserved before command-buffer completion"
            );
        }
        if (!state_->commandBufferSucceeded || state_->candidateQuarantined) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::commandBufferFailure,
                "failed Metal command-buffer transport cannot reserve a sensor generation"
            );
        }
        if (state_->candidatePublicationLeased ||
            state_->rootPublicationReserved ||
            state_->candidatePublicationTerminal) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::contextBusy,
                "candidate publication capability is already leased or quarantined"
            );
        }

        const std::uint64_t fingerprint =
            candidatePublicationFingerprint(*state_, slot, key);
        MetalNumanXHumanIOCandidatePublicationProgram stagedProgram =
            makeCandidatePublicationProgram(*state_, slot, fingerprint);
        MetalNumanXHumanIOCandidatePublicationView stagedView =
            makeCandidatePublicationView(*state_, slot, fingerprint);
        if (fingerprint == 0u || !stagedProgram.valid() ||
            stagedView.deviceRegistryID == 0u ||
            stagedView.proprioception.metalBuffer == nullptr ||
            stagedView.validity.metalBuffer == nullptr ||
            stagedView.interoception.metalBuffer == nullptr ||
            stagedView.interoceptionValidity.metalBuffer == nullptr) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::internalFailure,
                "failed to construct an exact candidate-publication capability"
            );
        }
        MetalNumanXHumanIODiagnostics result = diagnosticsLocked(
            *state_, MetalNumanXHumanIOStatus::success);
        result.published = false;
        result.transactionFingerprint = slot.transactionFingerprint;
        result.programFingerprint = slot.programFingerprint;
        result.sensorGeneration = slot.sensorGeneration;
        result.commandBufferIdentity = slot.commandBufferIdentity;
        result.encoded = true;
        result.commandBufferCompleted = true;
        result.commandBufferSucceeded = true;
        state_->candidatePublicationFingerprint = fingerprint;
        state_->candidatePublicationLeased = true;
        state_->rootPublicationReserved = false;
        state_->candidatePublicationTerminal = false;
        state_->publicationBinding = {};
        state_->lastStatus = MetalNumanXHumanIOStatus::success;
        state_->lastMessage.clear();
        lease = {};
        lease.state_ = state_;
        lease.program_ = stagedProgram;
        lease.view_ = stagedView;
        return result;
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    }
}

MetalNumanXHumanIODiagnostics MetalNumanXHumanIOContext::reject(
    const MetalNumanXHumanIOTransactionKey& key
) {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (!state_->candidatePrepared || state_->candidateSlot < 0) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateUnavailable,
                "no candidate is available for rejection"
            );
        }
        const Slot& slot = state_->slots[state_->candidateSlot];
        if (!keyMatches(key, slot)) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::incompatibleTransaction,
                "rejection key does not match the pending candidate"
            );
        }
        if (!state_->commandBufferCompleted) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateInFlight,
                "candidate cannot be recycled while its command buffer is in flight"
            );
        }
        if (state_->candidatePublicationLeased ||
            state_->rootPublicationReserved ||
            state_->candidatePublicationTerminal) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::contextBusy,
                "an exact candidate-publication lease owns this generation; use its reject callback"
            );
        }

        MetalNumanXHumanIODiagnostics result = diagnosticsLocked(
            *state_,
            MetalNumanXHumanIOStatus::success
        );
        clearCandidateOwnershipLocked(*state_);
        state_->lastStatus = MetalNumanXHumanIOStatus::success;
        state_->lastMessage.clear();
        result.published = false;
        return result;
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    }
}

MetalNumanXHumanIODiagnostics MetalNumanXHumanIOContext::publishedView(
    MetalNumanXHumanIOSensorView& view
) const {
    if (state_ == nullptr) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = "MetalNumanXHumanIOContext is moved-from";
        return result;
    }
    try {
        const std::lock_guard lock(state_->mutex);
        if (state_->publishedSlot < 0) {
            return diagnosticsLocked(
                *state_,
                MetalNumanXHumanIOStatus::candidateUnavailable,
                "no sensor generation has been accepted and published"
            );
        }
        const Slot& slot = state_->slots[state_->publishedSlot];
        MetalNumanXHumanIODiagnostics result = diagnosticsLocked(
            *state_,
            MetalNumanXHumanIOStatus::success
        );
        result.published = true;
        result.transactionFingerprint = slot.transactionFingerprint;
        result.programFingerprint = slot.programFingerprint;
        result.sensorGeneration = slot.sensorGeneration;
        result.commandBufferIdentity = slot.commandBufferIdentity;
        result.encoded = true;
        result.commandBufferCompleted = true;
        result.commandBufferSucceeded = true;
        view = makeView(slot, MetalNumanXHumanIOViewState::published);
        return result;
    } catch (const std::exception& exception) {
        MetalNumanXHumanIODiagnostics result{};
        result.status = MetalNumanXHumanIOStatus::internalFailure;
        result.message = exception.what();
        return result;
    }
}

const char* metalNumanXHumanIOStatusName(
    const MetalNumanXHumanIOStatus status
) noexcept {
    switch (status) {
        case MetalNumanXHumanIOStatus::success:
            return "success";
        case MetalNumanXHumanIOStatus::invalidConfiguration:
            return "invalid_configuration";
        case MetalNumanXHumanIOStatus::invalidInput:
            return "invalid_input";
        case MetalNumanXHumanIOStatus::arithmeticOverflow:
            return "arithmetic_overflow";
        case MetalNumanXHumanIOStatus::metalDeviceUnavailable:
            return "metal_device_unavailable";
        case MetalNumanXHumanIOStatus::metallibUnavailable:
            return "metallib_unavailable";
        case MetalNumanXHumanIOStatus::metalLibraryFailure:
            return "metal_library_failure";
        case MetalNumanXHumanIOStatus::metalPipelineFailure:
            return "metal_pipeline_failure";
        case MetalNumanXHumanIOStatus::metalBufferFailure:
            return "metal_buffer_failure";
        case MetalNumanXHumanIOStatus::incompatibleDevice:
            return "incompatible_device";
        case MetalNumanXHumanIOStatus::incompatibleTransaction:
            return "incompatible_transaction";
        case MetalNumanXHumanIOStatus::contextBusy:
            return "context_busy";
        case MetalNumanXHumanIOStatus::candidateUnavailable:
            return "candidate_unavailable";
        case MetalNumanXHumanIOStatus::candidateNotEncoded:
            return "candidate_not_encoded";
        case MetalNumanXHumanIOStatus::candidateInFlight:
            return "candidate_in_flight";
        case MetalNumanXHumanIOStatus::commandBufferFailure:
            return "command_buffer_failure";
        case MetalNumanXHumanIOStatus::humanDiagnosticsRejected:
            return "human_diagnostics_rejected";
        case MetalNumanXHumanIOStatus::internalFailure:
            return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
