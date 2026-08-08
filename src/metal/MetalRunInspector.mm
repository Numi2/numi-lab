#include "metalrobo/MetalRunInspector.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <utility>

namespace metalrobo {
namespace detail {

constexpr std::uint32_t kInspectionSlotCount = 3u;

enum class InspectionSlotState : std::uint32_t {
    free = 0u,
    writing,
    ready,
    acquired,
};

struct InspectionSlot {
    __strong id<MTLBuffer> rgb = nil;
    __strong id<MTLBuffer> depth = nil;
    __strong id<MTLBuffer> validity = nil;
    std::atomic<InspectionSlotState> state{InspectionSlotState::free};
    std::uint64_t frameIndex = 0u;
    std::uint64_t submissionIndex = 0u;
};

struct MetalRunInspectorState
    : std::enable_shared_from_this<MetalRunInspectorState> {
    explicit MetalRunInspectorState(MetalRunInspectorConfig value)
        : config(std::move(value)), renderer({
              .metallibPath = config.metallibPath,
              .width = config.width,
              .height = config.height,
              .retainObservationBuffers = false,
              .geometricObservationsOnly = false,
              .maximumRetainedBytes = config.maximumRetainedBytes,
          }) {}

    MetalRunInspectorConfig config;
    MetalHybridRenderer renderer;
    const MetalWorldFamilyContext* worlds = nullptr;
    std::array<InspectionSlot, kInspectionSlotCount> slots{};
    std::atomic_uint32_t droppedFrames{0u};
    std::atomic_uint64_t nextFrameIndex{1u};
    std::mutex mutex;
    bool compiled = false;
    std::uint32_t usableSlotCount = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;

    [[nodiscard]] bool ensureBuffers(id<MTLDevice> device) {
        if (device == nil || width == 0u || height == 0u) {
            return false;
        }
        const std::size_t pixels =
            static_cast<std::size_t>(width) * height;
        if (pixels == 0u ||
            pixels > std::numeric_limits<std::size_t>::max() /
                sizeof(mr_float4)) {
            return false;
        }
        const std::size_t rgbBytes = pixels * sizeof(mr_float4);
        const std::size_t depthBytes = pixels * sizeof(float);
        const std::size_t validityBytes = pixels * sizeof(std::uint32_t);
        for (std::uint32_t index = 0u; index < usableSlotCount; ++index) {
            InspectionSlot& slot = slots[index];
            if (slot.rgb != nil && slot.depth != nil &&
                slot.validity != nil) {
                continue;
            }
            slot.rgb = [device newBufferWithLength:rgbBytes
                                           options:MTLResourceStorageModePrivate];
            slot.depth = [device newBufferWithLength:depthBytes
                                             options:MTLResourceStorageModePrivate];
            slot.validity = [device newBufferWithLength:validityBytes
                                                options:MTLResourceStorageModePrivate];
            if (slot.rgb == nil || slot.depth == nil ||
                slot.validity == nil) {
                slot.rgb = nil;
                slot.depth = nil;
                slot.validity = nil;
                return false;
            }
        }
        return true;
    }

    void completeSlot(
        const std::uint32_t slotIndex,
        const bool completed
    ) noexcept {
        if (slotIndex >= usableSlotCount) {
            return;
        }
        slots[slotIndex].state.store(
            completed ? InspectionSlotState::ready : InspectionSlotState::free,
            std::memory_order_release
        );
    }
};

struct EncoderContext {
    __unsafe_unretained id<MTLComputeCommandEncoder> encoder = nil;
};

void setLabel(void* context, const char* label) {
    auto* source = static_cast<EncoderContext*>(context);
    source->encoder.label =
        label == nullptr ? nil : [NSString stringWithUTF8String:label];
}

void useHeap(void* context, void* heap) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder useHeap:(__bridge id<MTLHeap>)heap];
}

void setPipeline(void* context, void* pipeline) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)pipeline];
}

void setBuffer(
    void* context,
    void* buffer,
    const std::size_t offset,
    const std::uint32_t index
) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder setBuffer:(__bridge id<MTLBuffer>)buffer
                         offset:offset
                        atIndex:index];
}

void setBytes(
    void* context,
    const void* bytes,
    const std::size_t length,
    const std::uint32_t index
) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder setBytes:bytes length:length atIndex:index];
}

void dispatchThreads(
    void* context,
    const std::size_t threadCount,
    const std::size_t threadsPerThreadgroup
) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder dispatchThreads:MTLSizeMake(threadCount, 1u, 1u)
               threadsPerThreadgroup:MTLSizeMake(
                   threadsPerThreadgroup,
                   1u,
                   1u
               )];
}

void dispatchThreadgroups(
    void* context,
    const std::size_t threadgroupCount,
    const std::size_t threadsPerThreadgroup
) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder dispatchThreadgroups:MTLSizeMake(
                   threadgroupCount,
                   1u,
                   1u
               )
               threadsPerThreadgroup:MTLSizeMake(
                   threadsPerThreadgroup,
                   1u,
                   1u
               )];
}

void dispatchThreadgroupsIndirect(
    void* context,
    void* arguments,
    const std::size_t offset,
    const std::size_t threadsPerThreadgroup
) {
    auto* source = static_cast<EncoderContext*>(context);
    [source->encoder dispatchThreadgroupsWithIndirectBuffer:
         (__bridge id<MTLBuffer>)arguments
                                    indirectBufferOffset:offset
                             threadsPerThreadgroup:MTLSizeMake(
                                 threadsPerThreadgroup,
                                 1u,
                                 1u
                             )];
}

} // namespace detail

MetalRunInspector::MetalRunInspector(MetalRunInspectorConfig config)
    : state_(std::make_shared<detail::MetalRunInspectorState>(
          std::move(config)
      )) {}

MetalRunInspector::~MetalRunInspector() = default;
MetalRunInspector::MetalRunInspector(MetalRunInspector&& other) noexcept = default;
MetalRunInspector& MetalRunInspector::operator=(
    MetalRunInspector&& other
) noexcept = default;

MetalHybridRendererDiagnostics MetalRunInspector::compile(
    VisualRenderSceneV3&& scene,
    const VisualRendererProfileV1& profile,
    const MetalWorldFamilyContext& worlds
) {
    if (state_ == nullptr ||
        state_->config.width == 0u || state_->config.height == 0u ||
        state_->config.maximumFramesInFlight == 0u ||
        state_->config.maximumFramesInFlight > detail::kInspectionSlotCount ||
        profile.rayQueryVisibility) {
        MetalHybridRendererDiagnostics result;
        result.status = MetalHybridRendererStatus::invalidConfiguration;
        result.message =
            "inspection requires sensor_fast and one to three frame slots";
        return result;
    }
    std::lock_guard lock(state_->mutex);
    MetalHybridRendererDiagnostics result = state_->renderer.compile(
        std::move(scene),
        profile,
        1u
    );
    if (!result.succeeded()) {
        return result;
    }
    state_->worlds = &worlds;
    state_->width = result.layout.width;
    state_->height = result.layout.height;
    state_->usableSlotCount = state_->config.maximumFramesInFlight;
    state_->compiled = true;
    return result;
}

MetalWorldInspectionProgram MetalRunInspector::inspectionProgram() noexcept {
    return state_ != nullptr && state_->compiled
        ? MetalWorldInspectionProgram{
              .context = state_.get(),
              .encode = &MetalRunInspector::encodeInspection,
          }
        : MetalWorldInspectionProgram{};
}

bool MetalRunInspector::encodeInspection(
    void* context,
    const MetalWorldInspectionPass& pass
) {
    auto* state = static_cast<detail::MetalRunInspectorState*>(context);
    if (state == nullptr || !state->compiled || state->worlds == nullptr ||
        pass.commandBuffer == nullptr || pass.currentBodies == nullptr ||
        pass.environmentCount == 0u || pass.bodyCount == 0u ||
        state->config.environmentIndex >= pass.environmentCount) {
        return false;
    }

    std::uint32_t slotIndex = MR_INVALID_INDEX;
    for (std::uint32_t index = 0u; index < state->usableSlotCount; ++index) {
        auto expected = detail::InspectionSlotState::free;
        if (state->slots[index].state.compare_exchange_strong(
                expected,
                detail::InspectionSlotState::writing,
                std::memory_order_acq_rel
            )) {
            slotIndex = index;
            break;
        }
    }
    if (slotIndex == MR_INVALID_INDEX) {
        state->droppedFrames.fetch_add(1u, std::memory_order_relaxed);
        return true;
    }

    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBuffer> currentBodies =
        (__bridge id<MTLBuffer>)pass.currentBodies;
    if (command == nil || currentBodies == nil ||
        !state->ensureBuffers(currentBodies.device)) {
        state->completeSlot(slotIndex, false);
        return false;
    }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) {
        state->completeSlot(slotIndex, false);
        return false;
    }

    detail::EncoderContext contextState{encoder};
    MetalHybridComputeEncoderCallbacks callbacks{
        .context = &contextState,
        .setLabel = &detail::setLabel,
        .useHeap = &detail::useHeap,
        .useResidencySet = nullptr,
        .setPipeline = &detail::setPipeline,
        .setBuffer = &detail::setBuffer,
        .setBytes = &detail::setBytes,
        .dispatchThreads = &detail::dispatchThreads,
        .dispatchThreadgroups = &detail::dispatchThreadgroups,
        .dispatchThreadgroupsIndirect = &detail::dispatchThreadgroupsIndirect,
    };
    const std::size_t bodyOffset =
        static_cast<std::size_t>(state->config.environmentIndex) *
        pass.bodyCount * sizeof(MRBodyStateGPU);
    HybridDeviceStateBatch liveState;
    liveState.currentBodyStates = pass.currentBodies;
    liveState.currentBodyOffset = bodyOffset;
    liveState.environmentCount = 1u;
    liveState.bodyCount = pass.bodyCount;
    liveState.frameIndex = state->nextFrameIndex.fetch_add(
        1u,
        std::memory_order_relaxed
    );
    liveState.sensorSequence = static_cast<std::uint32_t>(
        liveState.frameIndex
    );
    liveState.source = MR_VISUAL_SOURCE_SIMULATION;
    HybridDeviceObservationBuffers outputs;
    outputs.rgb = (__bridge void*)state->slots[slotIndex].rgb;
    outputs.depth = (__bridge void*)state->slots[slotIndex].depth;
    outputs.validity = (__bridge void*)state->slots[slotIndex].validity;
    outputs.outputMask = 0u;
    const MetalHybridRendererDiagnostics rendered = state->renderer.encodeGraph(
        *state->worlds,
        liveState,
        0u,
        callbacks,
        outputs
    );
    [encoder endEncoding];
    if (!rendered.succeeded()) {
        state->completeSlot(slotIndex, false);
        return false;
    }
    state->slots[slotIndex].frameIndex = liveState.frameIndex;
    state->slots[slotIndex].submissionIndex = pass.submissionIndex;
    const std::shared_ptr<detail::MetalRunInspectorState> retained =
        state->shared_from_this();
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        retained->completeSlot(
            slotIndex,
            completed.status == MTLCommandBufferStatusCompleted
        );
    }];
    return true;
}

bool MetalRunInspector::acquireLatestFrame(
    MetalRunInspectorFrame& output
) noexcept {
    if (state_ == nullptr || !state_->compiled) {
        return false;
    }
    std::uint32_t selected = MR_INVALID_INDEX;
    std::uint64_t newest = 0u;
    for (std::uint32_t index = 0u; index < state_->usableSlotCount; ++index) {
        const detail::InspectionSlot& slot = state_->slots[index];
        if (slot.state.load(std::memory_order_acquire) ==
                detail::InspectionSlotState::ready &&
            slot.frameIndex >= newest) {
            newest = slot.frameIndex;
            selected = index;
        }
    }
    if (selected == MR_INVALID_INDEX) {
        return false;
    }
    auto expected = detail::InspectionSlotState::ready;
    if (!state_->slots[selected].state.compare_exchange_strong(
            expected,
            detail::InspectionSlotState::acquired,
            std::memory_order_acq_rel
        )) {
        return false;
    }
    const detail::InspectionSlot& slot = state_->slots[selected];
    output.rgb = (__bridge void*)slot.rgb;
    output.slotIndex = selected;
    output.width = state_->width;
    output.height = state_->height;
    output.frameIndex = slot.frameIndex;
    output.submissionIndex = slot.submissionIndex;
    output.environmentIndex = state_->config.environmentIndex;
    output.droppedFrames = state_->droppedFrames.load(
        std::memory_order_relaxed
    );
    return true;
}

void MetalRunInspector::releaseFrame(const std::uint32_t slotIndex) noexcept {
    if (state_ == nullptr || slotIndex >= state_->usableSlotCount) {
        return;
    }
    auto expected = detail::InspectionSlotState::acquired;
    state_->slots[slotIndex].state.compare_exchange_strong(
        expected,
        detail::InspectionSlotState::free,
        std::memory_order_release
    );
}

} // namespace metalrobo
