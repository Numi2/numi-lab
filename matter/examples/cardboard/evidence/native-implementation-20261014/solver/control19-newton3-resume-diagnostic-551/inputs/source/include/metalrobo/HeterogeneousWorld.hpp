#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kHeterogeneousWorldFormatVersion = 7u;

struct HeterogeneousWorldComponent {
    const EngineModel* model = nullptr;
    std::string_view instanceId;
    // One state for every source body whose articulationIndex is invalid,
    // ordered by source body index.
    std::span<const MRBodyStateGPU> defaultSceneBodies{};
};

struct HeterogeneousRodCollisionConfig {
    // When present, composition appends this rod-owned material to the
    // common EngineModel and resolves materialIndex transactionally. This
    // prevents procedural rods from silently inheriting an unrelated
    // component material merely because it occupies slot zero.
    std::optional<MRMaterialGPU> ownedMaterial;
    std::uint32_t materialIndex = 0u;
    std::uint32_t collisionGroup = 1u;
    std::uint32_t collisionMask = ~0u;
    std::uint32_t topologyGeneration = 1u;
    double contactOffset = 0.0;
    double restOffset = 0.0;
    bool enableToolCollision = true;
    bool enableCCD = true;
};

struct HeterogeneousRodProgram {
    std::string instanceId;
    DiscreteElasticRodModel model;
    DiscreteElasticRodState defaultState;
    // World-owned default mechanics, including whether capsule self-contact
    // participates in every rod transaction.
    DiscreteElasticRodStepConfig stepConfig;
    // Generic-world procedural capsule and pair-filtering semantics. These
    // records are consumed by the same collision graph as rigid colliders.
    HeterogeneousRodCollisionConfig collision;
    std::vector<DiscreteRodAttachment> attachments;
    // Body indices address HeterogeneousWorld::defaultSceneBodies, not global
    // EngineModel body indices. This is the exact Metal rod runtime packing.
    std::vector<DiscreteRodRigidAttachmentBinding> rigidBindings;
    // One line/swing relation per clamped rod edge. Each record cooks two
    // transverse scalar rows while leaving axial stretch to the rod.
    std::vector<DiscreteRodRigidTangentAttachmentBinding> tangentBindings;
    // One scalar material-frame relation per permanently swaged edge. These
    // are cooked into the same typed ConstraintIR/island as the translational
    // bindings, so the rigid body receives the equal-and-opposite torque.
    std::vector<DiscreteRodRigidTwistAttachmentBinding> twistBindings;
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

struct DualPsmNeedleThreadNeutralZoneConfig {
    DualPsmNeedleThreadWorldConfig surgical =
        makeBowelAnastomosisNeedleThreadWorldConfig();
    SurgicalNeutralZonePadSpec pad{};
    SurgicalBasePose padPose{};
    // The packaged monofilament is placed as a separated planar coil rather
    // than an unphysical 25 cm cantilever. Values are explicit training-scene
    // layout parameters, not product-package dimensions.
    double threadCoilPitchM = 0.003;
    double threadCoilInnerRadiusM = 0.005;
    // The reset laydown must leave every non-neighbour edge capsule visibly
    // separated before the live per-substep DER self-contact projector starts;
    // this geometry certificate complements rather than replaces projection.
    double threadMinimumNonNeighbourSurfaceClearanceM = 5.0e-5;
    // Four exact-length chords continue the swage tangent through a smooth
    // quarter-turn while descending to the support plane. One edge created a
    // 90-degree material kink at the first free node.
    std::uint32_t threadDescentEdgeCount = 4u;
    double threadContactOffsetM = 2.0e-5;
    double threadRestOffsetM = 1.0e-5;
    // Explicit research damping and bounded nonlinear certificate for the
    // packaged monofilament. These are training-scene parameters pending
    // package-specific dynamic calibration, not product claims.
    // The bound is intentionally above the usual resting solve. Early exit
    // preserves the common path, while measured acquisition/lift transients
    // have enough nonlinear work to satisfy the same 20 um certificate. A
    // 512-iteration ceiling rolled back a still-bilateral 128-node grasp at a
    // 32.7 um correction; increasing work does not weaken acceptance.
    std::uint32_t threadSolverIterations = 2048u;
    double threadConstraintToleranceM = 2.0e-5;
    double threadLinearDampingRate = 8.0;
    double threadTwistDampingRate = 8.0;
    // A micrometre-scale initial overlap makes the supported contact set
    // active at reset and avoids a synthetic free-fall impact.
    double threadSupportPenetrationM = 1.0e-6;
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

// Canonical bimanual pickup/handoff world. The needle and attached DER thread
// are dynamic, both PSMs are independently articulated, and the neutral-zone
// pad is an ordinary static collider in the same MetalWorld contact graph.
[[nodiscard]] HeterogeneousWorldComposeDiagnostics
makeDualDvrkPsmNeedleThreadNeutralZoneHeterogeneousWorld(
    HeterogeneousWorld& output,
    const DualPsmNeedleThreadNeutralZoneConfig& config = {}
);

[[nodiscard]] const char* heterogeneousWorldComposeStatusName(
    HeterogeneousWorldComposeStatus status
) noexcept;

} // namespace metalrobo
