#pragma once

#include "metalrobo/ArticulatedCollision.hpp"
#include "metalrobo/Collision.hpp"
#include "metalrobo/CoupledArticulatedRigidContact.hpp"
#include "metalrobo/RigidBodyWorld.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

enum class ArticulatedRigidCollisionStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidDimensions,
    invalidRigidBody,
    invalidRigidShape,
    invalidWarmStart,
    kinematicsFailure,
    collisionFailure,
    contactAssemblyFailure,
    contactAdaptationFailure,
    nonfiniteResult,
};

// Stable identity for one cross-system contact. pairKey packs the source
// EngineModel shape index in its high word and the source rigid-shape span
// index in its low word. featureKey uses the same articulated/rigid order.
// Slot generations and articulationIndex prevent stale warm starts from
// surviving model, asset, or slot replacement.
struct ArticulatedRigidContactKey {
    std::uint64_t pairKey = 0u;
    std::uint64_t featureKey = 0u;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t articulatedGeneration = 0u;
    std::uint32_t rigidGeneration = 0u;

    bool operator==(const ArticulatedRigidContactKey&) const = default;
};

// A world-space impulse acting on the rigid endpoint. World storage avoids
// treating last frame's tangent coordinates as current-frame coordinates.
// The adapter projects a matched impulse into its newly generated contact
// frame. Stale, unmatched keys are permitted; duplicate keys are rejected.
struct ArticulatedRigidContactWarmStart {
    ArticulatedRigidContactKey key{};
    std::array<double, 3> worldImpulseOnRigid{};
};

// Metadata is exactly index-aligned with the output contact span.
struct ArticulatedRigidContactMetadata {
    ArticulatedRigidContactKey key{};
    std::uint32_t articulatedShapeIndex = 0u;
    std::uint32_t rigidShapeIndex = 0u;
    std::uint32_t articulatedBodyIndex = 0u;
    std::uint32_t rigidBodyIndex = 0u;
    std::uint32_t manifoldIndex = 0u;
    std::uint32_t manifoldPointIndex = 0u;
    std::uint32_t lifetime = 0u;
    // Collision-local source records remain useful for debugging and cache
    // correlation. The public key above does not depend on these rebased
    // collider indices.
    std::uint64_t collisionPairKey = 0u;
    std::uint64_t collisionFeatureKey = 0u;
    std::array<double, 3> contactPointWorld{};
    std::array<double, 3> localWitnessArticulated{};
    std::array<double, 3> localWitnessRigid{};
    double effectiveSeparation = 0.0;
};

struct ArticulatedRigidCollisionConfig {
    CollisionConfig collision{
        .environment = 0u,
        .capacities = {
            .pairCapacity = 2048u,
            .rawContactCapacity = 4096u,
            .manifoldCapacity = 2048u,
        },
        .manifoldBreakingSeparation = 0.003,
        .manifoldBreakingTangential = 0.003,
        .manifoldMergeDistance = 0.0001,
        .manifoldNormalCosine = 0.95,
    };
    ArticulatedCollisionAdapterConfig contact{};
    ArticulatedDynamicsConfig dynamics{};
};

struct ArticulatedRigidCollisionDiagnostics {
    ArticulatedRigidCollisionStatus status =
        ArticulatedRigidCollisionStatus::success;
    MRStepStatusCode code = MR_STEP_SUCCESS;
    ArticulatedDynamicsDiagnostics kinematics{};
    CollisionDiagnostics collision{};
    ContactAssemblyDiagnostics assembly{};
    ArticulatedCollisionDiagnostics adaptation{};
    std::uint32_t articulationIndex = 0u;
    std::uint32_t articulatedShapeCount = 0u;
    std::uint32_t rigidShapeCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t suppliedWarmStartCount = 0u;
    std::uint32_t matchedWarmStartCount = 0u;
    double maximumPenetration = 0.0;
    std::string failure;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedRigidCollisionStatus::success;
    }
};

struct ArticulatedRigidCollisionResult {
    ArticulatedRigidCollisionDiagnostics diagnostics{};
    std::vector<CoupledArticulatedRigidContact> contacts;
    std::vector<ArticulatedRigidContactMetadata> metadata;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

enum class ArticulatedRigidIslandPairClass : std::uint32_t {
    articulatedDynamicScene = 0u,
    articulatedPrescribedScene,
    dynamicSceneDynamicScene,
    dynamicScenePrescribedScene,
};

// Canonical source identity for one endpoint of a full-scene contact.
// Articulated shape/body indices address EngineModel; scene indices address
// the caller's external spans. Motion type is repeated deliberately so a
// dynamic/prescribed role change cannot inherit an incompatible warm start.
struct ArticulatedRigidIslandContactEndpointKey {
    CoupledContactEndpointKind kind =
        CoupledContactEndpointKind::sceneBody;
    std::uint32_t bodyIndex = 0u;
    std::uint32_t shapeIndex = 0u;
    std::uint32_t feature = 0u;
    std::uint32_t slotGeneration = 0u;
    std::uint32_t motionType = MR_MOTION_STATIC;

    bool operator==(
        const ArticulatedRigidIslandContactEndpointKey&
    ) const = default;
};

// Endpoint order follows the deterministic combined collision stream:
// selected articulation shapes in EngineModel order, followed by scene
// shapes in caller order. It never depends on broadphase traversal order.
struct ArticulatedRigidIslandContactKey {
    std::uint32_t articulationIndex = 0u;
    ArticulatedRigidIslandContactEndpointKey endpointA{};
    ArticulatedRigidIslandContactEndpointKey endpointB{};

    bool operator==(
        const ArticulatedRigidIslandContactKey&
    ) const = default;
};

// World-space impulse acting on canonical endpoint B. Persisting a physical
// vector rather than tangent coordinates makes warm starts frame-rotation
// safe. Duplicate keys and non-finite impulses are rejected transactionally.
struct ArticulatedRigidIslandContactWarmStart {
    ArticulatedRigidIslandContactKey key{};
    std::array<double, 3> worldImpulseOnB{};
};

// Exactly index-aligned with ArticulatedRigidIslandCollisionResult::contacts.
struct ArticulatedRigidIslandContactMetadata {
    ArticulatedRigidIslandContactKey key{};
    ArticulatedRigidIslandPairClass pairClass =
        ArticulatedRigidIslandPairClass::articulatedDynamicScene;
    std::uint32_t manifoldIndex = 0u;
    std::uint32_t manifoldPointIndex = 0u;
    std::uint32_t lifetime = 0u;
    bool warmStartMatched = false;
    std::uint64_t collisionPairKey = 0u;
    std::uint64_t collisionFeatureKey = 0u;
    std::array<double, 3> contactPointWorld{};
    std::array<double, 3> localWitnessA{};
    std::array<double, 3> localWitnessB{};
    double effectiveSeparation = 0.0;
};

struct ArticulatedRigidIslandCollisionDiagnostics {
    ArticulatedRigidCollisionStatus status =
        ArticulatedRigidCollisionStatus::success;
    MRStepStatusCode code = MR_STEP_SUCCESS;
    ArticulatedDynamicsDiagnostics kinematics{};
    CollisionDiagnostics collision{};
    ContactAssemblyDiagnostics assembly{};
    std::uint32_t articulationIndex = 0u;
    std::uint32_t articulatedShapeCount = 0u;
    std::uint32_t sceneShapeCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t articulatedDynamicContactCount = 0u;
    std::uint32_t articulatedPrescribedContactCount = 0u;
    std::uint32_t dynamicDynamicContactCount = 0u;
    std::uint32_t dynamicPrescribedContactCount = 0u;
    std::uint32_t suppliedWarmStartCount = 0u;
    std::uint32_t matchedWarmStartCount = 0u;
    double maximumPenetration = 0.0;
    std::string failure;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedRigidCollisionStatus::success;
    }
};

struct ArticulatedRigidIslandCollisionResult {
    ArticulatedRigidIslandCollisionDiagnostics diagnostics{};
    std::vector<CoupledArticulatedRigidIslandContact> contacts;
    std::vector<ArticulatedRigidIslandContactMetadata> metadata;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

// Generates collision contacts between one articulation and independent
// dynamic rigid bodies:
//
//  1. evaluate actual FP64 articulated q/v kinematics;
//  2. project selected articulated and rigid shapes into one collision call;
//  3. assemble the shared material/manifold contact ABI;
//  4. reuse the articulated collision adapter for canonical target,
//     regularization, frame, and articulated-local point semantics;
//  5. restore the independent rigid endpoint for the coupled solver.
//
// rigidShapes index rigidBodies and rigidMaterials locally. Every rigid body
// must satisfy the coupled operator's independent dynamic-body contract.
// Only articulation-rigid pairs are evaluated; articulation self-collision
// and rigid-rigid collision are outside this adapter.
//
// The coupled exact-cone operator has one Coulomb coefficient, so this
// adapter deliberately selects the common ABI's mixed dynamic coefficient.
// Rolling and torsional friction remain unsupported and fail explicitly.
//
// Publication is transactional. On every failure contacts and metadata are
// empty and manifoldCache is unchanged. Empty, valid cross-collision results
// are successful and commit the collision cache.
[[nodiscard]] ArticulatedRigidCollisionResult
collideArticulatedRigidContactsCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const MRShapeGPU> rigidShapes,
    std::span<const MRMaterialGPU> rigidMaterials,
    std::span<const MRBodyStateGPU> rigidBodies,
    PersistentManifoldCache& manifoldCache,
    const ArticulatedRigidCollisionConfig& config = {},
    std::span<const ArticulatedRigidContactWarmStart> warmStarts = {}
);

// Generates every executable contact in one scene containing exactly one
// selected articulation plus independent external bodies. Scene bodies may be
// dynamic, static, or kinematic. Only articulation-articulation pairs are
// explicitly excluded; the shared collision oracle rejects same-body and
// prescribed-prescribed pairs, leaving these four supported classes:
//
//   articulation-dynamic, articulation-prescribed,
//   dynamic-dynamic, and dynamic-prescribed.
//
// Each output uses the generic mixed-island endpoint contract. targetVelocity
// is the desired physical A-to-B relative velocity; prescribed scene velocity
// is intentionally not pre-compensated because the mixed solver subtracts it
// exactly once. Dynamic friction is selected for the current one-coefficient
// exact cone. Rolling and torsional friction fail explicitly.
//
// sceneShapes index sceneBodies and sceneMaterials locally. Combined collider
// order and all public source keys are deterministic. Publication is fully
// transactional: on failure contacts and metadata are empty and
// manifoldCache is unchanged.
[[nodiscard]] ArticulatedRigidIslandCollisionResult
collideArticulatedRigidIslandContactsCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const MRShapeGPU> sceneShapes,
    std::span<const MRMaterialGPU> sceneMaterials,
    std::span<const MRBodyStateGPU> sceneBodies,
    PersistentManifoldCache& manifoldCache,
    const ArticulatedRigidCollisionConfig& config = {},
    std::span<const ArticulatedRigidIslandContactWarmStart> warmStarts = {}
);

} // namespace metalrobo
