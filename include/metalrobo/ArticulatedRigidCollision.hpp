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

} // namespace metalrobo
