#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kHeterogeneousWorldFormatVersion = 1u;

struct HeterogeneousWorldComponent {
    const EngineModel* model = nullptr;
    std::string_view instanceId;
    // One state for every source body whose articulationIndex is invalid,
    // ordered by source body index.
    std::span<const MRBodyStateGPU> defaultSceneBodies{};
};

struct HeterogeneousRodProgram {
    std::string instanceId;
    DiscreteElasticRodModel model;
    DiscreteElasticRodState defaultState;
    std::vector<DiscreteRodAttachment> attachments;
    // Body indices address HeterogeneousWorld::defaultSceneBodies, not global
    // EngineModel body indices. This is the exact Metal rod runtime packing.
    std::vector<DiscreteRodRigidAttachmentBinding> rigidBindings;
};

struct HeterogeneousWorld {
    std::uint32_t formatVersion =
        kHeterogeneousWorldFormatVersion;
    EngineModel model;
    // Global EngineModel body indices in the exact order used by
    // defaultSceneBodies and MetalWorldBatch::initialSceneBodies.
    std::vector<std::uint32_t> sceneBodyIndices;
    std::vector<MRBodyStateGPU> defaultSceneBodies;
    std::vector<HeterogeneousRodProgram> rods;
    std::vector<std::string> componentInstanceIds;
    std::uint64_t fingerprint = 0u;

    [[nodiscard]] bool valid(
        std::string* reason = nullptr
    ) const;
};

enum class HeterogeneousWorldComposeStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidComponent,
    invalidSceneState,
    invalidRodProgram,
    modelCompositionFailure,
    invalidWorld,
    allocationFailure,
};

struct HeterogeneousWorldComposeDiagnostics {
    HeterogeneousWorldComposeStatus status =
        HeterogeneousWorldComposeStatus::success;
    std::uint32_t componentCount = 0u;
    std::uint32_t articulationCount = 0u;
    std::uint32_t sceneBodyCount = 0u;
    std::uint32_t rodCount = 0u;
    std::uint32_t firstFailingComponent = MR_INVALID_INDEX;
    std::uint64_t fingerprint = 0u;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == HeterogeneousWorldComposeStatus::success;
    }
};

// Owns every immutable topology and reset stream needed to instantiate a
// heterogeneous world. EngineModel composition, scene-state rebasing, rod
// binding validation, and fingerprint publication are transactional.
[[nodiscard]] HeterogeneousWorldComposeDiagnostics
composeHeterogeneousWorld(
    std::span<const HeterogeneousWorldComponent> components,
    std::span<const HeterogeneousRodProgram> rods,
    HeterogeneousWorld& output,
    const EngineModelComposeConfig& config = {}
);

[[nodiscard]] std::uint64_t heterogeneousWorldFingerprint(
    const HeterogeneousWorld& world
) noexcept;

// Canonical surgical bundle used by R2S/S2R world compilation. The model owns
// two PSM articulations plus the dynamic needle's rigid topology; the reset
// streams own the needle pose and the geometry-derived swage/thread program.
[[nodiscard]] HeterogeneousWorldComposeDiagnostics
makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
    HeterogeneousWorld& output,
    const DualPsmNeedleThreadWorldConfig& config = {}
);

[[nodiscard]] const char* heterogeneousWorldComposeStatusName(
    HeterogeneousWorldComposeStatus status
) noexcept;

} // namespace metalrobo
