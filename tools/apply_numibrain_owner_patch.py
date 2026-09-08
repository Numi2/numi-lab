#!/usr/bin/env python3
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:80]!r}")
    if text.count(old) != 1:
        raise SystemExit(f"anchor not unique in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


h = "include/metalrobo/MetalWorld.hpp"
old = """    [[nodiscard]] MetalWorldDiagnostics run(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldResult& result
    );

    [[nodiscard]] MetalWorldContextStats stats() const noexcept;"""
new = """    [[nodiscard]] MetalWorldDiagnostics run(
        const CompiledWorld& world,
        const MetalWorldBatch& batch,
        const MetalWorldStepConfig& config,
        MetalWorldResult& result
    );

    // Canonical digest of the complete device-resident continuation state.
    // The call is inspection-only, requires an initialized idle resident
    // slot, and includes every persistent arena buffer plus resident metadata.
    [[nodiscard]] MetalWorldDiagnostics residentStateFingerprint(
        const MetalWorldResidentState& state,
        std::uint64_t& fingerprint
    );

    [[nodiscard]] MetalWorldContextStats stats() const noexcept;"""
replace_once(h, old, new)

mm = "src/metal/MetalWorld.mm"
marker = """const char* metalWorldHostStatusName(
    const MetalWorldHostStatus status
) noexcept {"""
impl = r'''MetalWorldDiagnostics MetalWorldContext::residentStateFingerprint(
    const MetalWorldResidentState& state,
    std::uint64_t& fingerprint
) {
    fingerprint = 0u;
    const auto resident = state.state_;
    if (resident == nullptr) {
        return reject({}, MetalWorldHostStatus::invalidModel,
            "resident-state fingerprint requires a valid state");
    }
    const auto owner = resident->ownerPool.lock();
    if (owner == nullptr || owner.get() != pool_.get()) {
        return reject({}, MetalWorldHostStatus::invalidModel,
            "resident-state fingerprint owner mismatch");
    }

    const std::lock_guard residentLock(resident->mutex);
    const auto context = resident->context;
    if (!resident->initialized || resident->pending || context == nullptr) {
        return reject({}, MetalWorldHostStatus::contextBusy,
            "resident-state fingerprint requires an accepted idle state");
    }
    const std::lock_guard contextLock(context->mutex);
    if (context->inFlight ||
        resident->stateArenaGeneration != context->stateArenaGeneration) {
        return reject({}, MetalWorldHostStatus::contextBusy,
            "resident-state fingerprint cannot inspect an in-flight or stale arena");
    }

    std::array<__strong id<MTLBuffer>, kRawBufferCount> staged{};
    id<MTLCommandBuffer> command = [context->queue commandBuffer];
    if (command == nil) {
        return reject({}, MetalWorldHostStatus::metalCommandFailure,
            "resident-state fingerprint could not allocate a command buffer");
    }
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) {
        return reject({}, MetalWorldHostStatus::metalCommandFailure,
            "resident-state fingerprint could not allocate a blit encoder");
    }

    for (std::size_t index = 0u; index < kRawBufferCount; ++index) {
        if (!privatePersistentBuffer(index)) {
            continue;
        }
        id<MTLBuffer> source = context->buffers[index];
        if (source == nil || source.length == 0u ||
            source.storageMode != MTLStorageModePrivate) {
            continue;
        }
        id<MTLBuffer> copy = [context->device
            newBufferWithLength:source.length
                       options:MTLResourceStorageModeShared];
        if (copy == nil || copy.contents == nullptr) {
            [blit endEncoding];
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                "resident-state fingerprint could not allocate readback storage");
        }
        staged[index] = copy;
        [blit copyFromBuffer:source
                sourceOffset:0u
                    toBuffer:copy
           destinationOffset:0u
                        size:source.length];
    }
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return reject({}, MetalWorldHostStatus::metalCommandFailure,
            "resident-state fingerprint readback command failed");
    }

    std::uint64_t hash = kFNVOffset;
    const auto append = [&](const void* data, const std::size_t size) {
        const auto* bytes = static_cast<const std::byte*>(data);
        for (std::size_t i = 0u; i < size; ++i) {
            hash ^= std::to_integer<std::uint8_t>(bytes[i]);
            hash *= kFNVPrime;
        }
    };
    const auto scalar = [&](const auto& value) { append(&value, sizeof(value)); };

    scalar(resident->worldFingerprint);
    scalar(resident->taskFingerprint);
    scalar(resident->taskSeed);
    scalar(resident->stateArenaGeneration);
    scalar(resident->environmentCount);
    scalar(resident->qBuffer);
    scalar(resident->vBuffer);
    scalar(resident->sceneBuffer);
    scalar(resident->manifoldHeaderBuffer);
    scalar(resident->manifoldPointBuffer);
    scalar(resident->manifoldCountBuffer);
    scalar(resident->rodNodeBuffer);
    scalar(resident->rodEdgeBuffer);
    scalar(resident->rodWitnessBuffer);
    scalar(context->boundModelFingerprint);
    scalar(context->boundTaskFingerprint);
    scalar(context->boundPolicyFingerprint);
    scalar(context->boundMulticopterFingerprint);

    for (std::size_t index = 0u; index < kRawBufferCount; ++index) {
        if (!privatePersistentBuffer(index)) {
            continue;
        }
        id<MTLBuffer> source = context->buffers[index];
        const std::uint64_t index64 = static_cast<std::uint64_t>(index);
        scalar(index64);
        const std::uint64_t length = source == nil
            ? 0u : static_cast<std::uint64_t>(source.length);
        scalar(length);
        if (length == 0u) {
            continue;
        }
        id<MTLBuffer> readable = source.storageMode == MTLStorageModePrivate
            ? staged[index] : source;
        if (readable == nil || readable.contents == nullptr ||
            readable.length < source.length) {
            return reject({}, MetalWorldHostStatus::metalBufferFailure,
                "resident-state fingerprint encountered unreadable persistent storage");
        }
        append(readable.contents, static_cast<std::size_t>(source.length));
    }
    fingerprint = hash == 0u ? 1u : hash;
    return {};
}

'''
replace_once(mm, marker, impl + marker)

ch = "include/metalrobo/c_api.h"
layout_anchor = "typedef struct MRTaskRolloutLayoutC {\n"
binding_struct = """typedef struct MRTaskActionBindingC {
    uint32_t action_index;
    uint32_t dof_index;
    uint32_t q_index;
    uint32_t v_index;
    float normalized_scale;
    float lower_target;
    float upper_target;
    float response_time_seconds;
    float drive_stiffness;
    float drive_damping;
    uint32_t interaction_motion;
    uint32_t reserved0;
    uint32_t actuator_kind;
    uint32_t resolved_component;
    uint32_t component_lane;
    uint32_t flags;
} MRTaskActionBindingC;

"""
replace_once(ch, layout_anchor, binding_struct + layout_anchor)
proto_anchor = """MR_API MRTaskRolloutLayoutC mr_task_rollout_layout(
    const MRTaskRolloutHandle* handle
);"""
protos = """MR_API size_t mr_task_rollout_action_binding_count(
    const MRTaskRolloutHandle* handle
);
MR_API int mr_task_rollout_copy_action_bindings(
    const MRTaskRolloutHandle* handle,
    MRTaskActionBindingC* output,
    size_t output_count
);
// Returns zero unless the rollout owns an initialized, accepted, idle resident
// state. The digest covers the complete persistent continuation arena.
MR_API uint64_t mr_task_rollout_resident_state_fingerprint(
    MRTaskRolloutHandle* handle
);

"""
replace_once(ch, proto_anchor, protos + proto_anchor)

cpp = "src/c_api.cpp"
cpp_anchor = """MRTaskRolloutLayoutC mr_task_rollout_layout(
    const MRTaskRolloutHandle* handle
) {"""
cpp_impl = r'''size_t mr_task_rollout_action_binding_count(
    const MRTaskRolloutHandle* handle
) {
    return handle == nullptr ? 0u : handle->taskProgram.actionBindings().size();
}

int mr_task_rollout_copy_action_bindings(
    const MRTaskRolloutHandle* handle,
    MRTaskActionBindingC* output,
    const size_t output_count
) {
    if (handle == nullptr) {
        return -1;
    }
    const auto bindings = handle->taskProgram.actionBindings();
    if ((bindings.size() != 0u && output == nullptr) ||
        output_count < bindings.size()) {
        return -1;
    }
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        const MRTaskActionBindingGPU& source = bindings[index];
        MRTaskActionBindingC& target = output[index];
        target.action_index = source.indices.x;
        target.dof_index = source.indices.y;
        target.q_index = source.indices.z;
        target.v_index = source.indices.w;
        target.normalized_scale = source.parameters.x;
        target.lower_target = source.parameters.y;
        target.upper_target = source.parameters.z;
        target.response_time_seconds = source.parameters.w;
        target.drive_stiffness = source.drive.x;
        target.drive_damping = source.drive.y;
        target.interaction_motion = source.drive.z != 0.0f ? 1u : 0u;
        target.reserved0 = 0u;
        target.actuator_kind = source.actuator.x;
        target.resolved_component = source.actuator.y;
        target.component_lane = source.actuator.z;
        target.flags = source.actuator.w;
    }
    return 0;
}

uint64_t mr_task_rollout_resident_state_fingerprint(
    MRTaskRolloutHandle* handle
) {
    if (handle == nullptr || !handle->residentState.valid()) {
        return 0u;
    }
    std::uint64_t fingerprint = 0u;
    const metalrobo::MetalWorldDiagnostics status =
        handle->context.residentStateFingerprint(
            handle->residentState,
            fingerprint
        );
    return status.succeeded() ? fingerprint : 0u;
}

'''
replace_once(cpp, cpp_anchor, cpp_impl + cpp_anchor)
