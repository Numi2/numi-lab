#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/rod_gpu_shared.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct MetalDiscreteElasticRodInput {
    std::span<const DiscreteElasticRodState> states{};
    // Fixed count per environment, packed environment-major. Targets remain
    // explicit inputs so PSM/needle kinematics can update them on-device.
    std::size_t attachmentCount = 0u;
    std::span<const DiscreteRodAttachment> attachments{};
    // Optional environment-major rigid candidate state. When bindings are
    // supplied, attachment targets are projected from these bodies on-device
    // and reaction impulses are applied back to disjoint bodies in the same
    // command graph.
    std::size_t rigidBodyCount = 0u;
    std::span<const MRBodyStateGPU> rigidBodies{};
    // Immutable homogeneous-world mapping, one slot per local attachment.
    // Coupled body indices must be unique so physical writes stay atomic-free.
    std::span<const DiscreteRodRigidAttachmentBinding> rigidBindings{};
};

struct MetalDiscreteElasticRodConfig {
    DiscreteElasticRodStepConfig step{};
    std::string metallibPath;
};

enum class MetalDiscreteElasticRodHostStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidConfiguration,
    invalidState,
    invalidAttachment,
    capacityOverflow,
    arithmeticOverflow,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    gpuEnvironmentFailure,
    internalFailure,
};

struct MetalDiscreteElasticRodResult {
    std::vector<DiscreteElasticRodState> states;
    std::vector<MRRodGPUStatus> statuses;
    // Environment-major, fixed attachmentCount.
    std::vector<DiscreteRodAttachmentReaction> reactions;
    // Environment-major, fixed rigidBodyCount. Empty for an uncoupled solve.
    std::vector<MRBodyStateGPU> rigidBodies;
};

struct MetalDiscreteElasticRodDiagnostics {
    MetalDiscreteElasticRodHostStatus status =
        MetalDiscreteElasticRodHostStatus::success;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingEnvironment = 0xffffffffu;
    std::uint32_t firstGPUStatusCode = MR_ROD_GPU_SUCCESS;
    std::size_t allocatedBytes = 0u;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MetalDiscreteElasticRodHostStatus::success;
    }
};

// SIMD32-cohort implicit XPBD/DER solve. One threadgroup owns one complete
// rod environment; 2/3-color phases make all physical writes disjoint.
// Optional non-adjacent capsule self-contact is refreshed within each
// nonlinear sweep. Body-anchor projection and equal/opposite reaction
// application are encoded in the same command buffer. Publication of rod and
// rigid candidates is transactional across the batch.
[[nodiscard]] MetalDiscreteElasticRodDiagnostics
runMetalDiscreteElasticRod(
    const DiscreteElasticRodModel& model,
    const MetalDiscreteElasticRodInput& input,
    MetalDiscreteElasticRodResult& output,
    const MetalDiscreteElasticRodConfig& config = {}
);

[[nodiscard]] const char* metalDiscreteElasticRodHostStatusName(
    MetalDiscreteElasticRodHostStatus status
) noexcept;

} // namespace metalrobo
