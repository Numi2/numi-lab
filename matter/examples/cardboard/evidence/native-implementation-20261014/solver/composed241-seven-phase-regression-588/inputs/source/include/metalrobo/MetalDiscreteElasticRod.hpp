#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/EngineModel.hpp"
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
    // Optional immutable collision topology for thread/tool contact. Body
    // indices in this model address rigidBodies. Rod-edge/tool-shape pairs
    // are cooked deterministically when toolPairs is empty.
    const EngineModel* toolModel = nullptr;
    std::span<const MRRodToolPairGPU> toolPairs{};
    // Optional explicit persistent cache packed
    // [environment][tool pair][four canonical feature slots].
    std::span<const MRRodToolWitnessGPU> previousToolContacts{};
};

struct MetalDiscreteElasticRodToolConfig {
    bool enabled = false;
    bool warmStart = true;
    bool enableCCD = false;
    std::uint32_t rodMaterialIndex = 0u;
    std::uint32_t collisionGroup = 1u;
    std::uint32_t collisionMask = ~0u;
    std::uint32_t outerIterations = 2u;
    float contactOffset = 2.0e-5f;
    float restOffset = 0.0f;
    float compliance = 0.0f;
    float damping = 0.0f;
    float restitution = 0.0f;
    float restitutionThreshold = 0.2f;
    float frictionScale = 1.0f;
    float maximumDepenetrationVelocity = 2.0f;
};

struct MetalDiscreteElasticRodConfig {
    DiscreteElasticRodStepConfig step{};
    MetalDiscreteElasticRodToolConfig tool{};
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
    // Fixed pair-owned evidence. Counts are packed [environment][tool pair],
    // witnesses are packed [environment][tool pair][four canonical slots].
    std::vector<std::uint32_t> toolContactCounts;
    std::vector<MRRodToolWitnessGPU> toolContacts;
};

struct MetalDiscreteElasticRodDiagnostics {
    MetalDiscreteElasticRodHostStatus status =
        MetalDiscreteElasticRodHostStatus::success;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingEnvironment = 0xffffffffu;
    std::uint32_t firstGPUStatusCode = MR_ROD_GPU_SUCCESS;
    std::uint32_t toolPairCount = 0u;
    std::uint32_t toolContactCount = 0u;
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
