#pragma once

#include "metalrobo/MeasuredSurfaceRobot.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace metalrobo {

struct MetalMeasuredSurfaceStats {
    std::uint64_t encodedPrepareCount = 0u;
    std::uint64_t encodedCommitCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::size_t immutableBytes = 0u;
    std::size_t persistentBytes = 0u;
    std::size_t transientBytes = 0u;
    std::size_t controlStepCheckpointBytes = 0u;
    std::size_t threadgroupBytes = 0u;
    std::uint32_t environmentCapacity = 0u;
    std::uint32_t threadgroupWidth = 0u;
};

struct MetalMeasuredSurfaceInspection {
    std::vector<MRMeasuredSurfaceStateGPU> acceptedStates;
    std::vector<MRMeasuredSurfaceEvidenceGPU> acceptedEvidence;
};

struct MetalMeasuredSurfacePresentationStyle {
    std::uint32_t semanticId = 0u;
    std::uint32_t instanceId = 0u;
    mr_float4 baseColorAndOpacity{0.18f, 0.42f, 0.85f, 1.0f};
    float perceptualRoughness = 0.42f;
    float metallic = 0.05f;
};

// Materializes one compiled measured-surface binding on the Metal device used
// by MetalWorld. It owns only its immutable payload and persistent generalized
// surface state; command submission, q/v, actions, body wrench, solver status,
// reset, and publication remain MetalWorld-owned.
class MetalMeasuredSurfaceMechanics {
public:
    struct State;

    explicit MetalMeasuredSurfaceMechanics(
        CompiledMeasuredSurfaceBinding binding);
    ~MetalMeasuredSurfaceMechanics();

    MetalMeasuredSurfaceMechanics(
        MetalMeasuredSurfaceMechanics&&) noexcept;
    MetalMeasuredSurfaceMechanics& operator=(
        MetalMeasuredSurfaceMechanics&&) noexcept;
    MetalMeasuredSurfaceMechanics(
        const MetalMeasuredSurfaceMechanics&) = delete;
    MetalMeasuredSurfaceMechanics& operator=(
        const MetalMeasuredSurfaceMechanics&) = delete;

    [[nodiscard]] MetalWorldDeviceMechanicsProgram program() noexcept;
    [[nodiscard]] const CompiledMeasuredSurfaceBinding& binding()
        const noexcept;
    [[nodiscard]] MetalMeasuredSurfaceStats stats() const noexcept;
    [[nodiscard]] MetalHybridDevicePresentationProgram
    presentationProgram(
        MetalMeasuredSurfacePresentationStyle style
    ) noexcept;
    // Explicit post-submission inspection boundary. This performs one private
    // to shared blit and wait; it is never used by rollout or training.
    [[nodiscard]] MetalMeasuredSurfaceInspection inspectAccepted() const;

private:
    std::unique_ptr<State> state_;
};

} // namespace metalrobo
