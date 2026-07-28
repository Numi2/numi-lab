#pragma once

#include "metalrobo/ArticulatedActuation.hpp"
#include "metalrobo/ArticulatedRigidCollision.hpp"
#include "metalrobo/FreeBodyDynamics.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace metalrobo {

enum class ArticulatedRigidWorldFailure : std::uint32_t {
    none = 0u,
    invalidConfiguration,
    actuation,
    articulatedFreeDynamics,
    rigidFreeDynamics,
    collision,
    jointLimitCompilation,
    coupledSolve,
    velocityPublication,
    articulatedIntegration,
    rigidIntegration,
    graspEvidence,
};

// Optional physics-derived grasp classifier. It never creates a weld or
// changes dynamics. A body qualifies only after both configured jaw bodies
// carry compressive impulses with opposing normals and bounded post-solve
// tangential slip for the requested dwell.
struct ArticulatedRigidGraspConfig {
    bool enabled = false;
    std::uint32_t jawBodyA = MR_INVALID_INDEX;
    std::uint32_t jawBodyB = MR_INVALID_INDEX;
    double minimumNormalImpulse = 1.0e-8;
    double minimumFriction = 0.05;
    double maximumTangentialSlipSpeed = 0.02;
    double maximumOpposingNormalDot = -0.2;
    std::uint32_t requiredConsecutiveSteps = 3u;
};

struct ArticulatedRigidWorldConfig {
    // The articulation timestep and gravity are authoritative. The composed
    // step copies them into rigid prediction, collision stabilization, and
    // joint-limit compilation so those phases cannot drift.
    ArticulatedDynamicsConfig dynamics{};
    ArticulatedActuationConfig actuation{};
    FreeBodyIntegratorConfig rigidFreeMotion{
        .timestep = 1.0 / 1000.0,
        .gravity = {0.0f, 0.0f, -9.81f, 0.0f},
        .integrator = FreeBodyIntegrator::symplecticEuler,
        .nonlinearIterations = 12u,
        .nonlinearTolerance = 1.0e-12,
    };
    ArticulatedJointLimitConfig jointLimits{};
    ArticulatedRigidCollisionConfig collision{};
    QualityContactSolverConfig quality{};
    ArticulatedRigidGraspConfig grasp{};
    // Optional deterministic deepest-point conditioning across compound
    // primitive witnesses belonging to the same articulated/rigid body pair.
    // The physical default preserves every assembled witness. Small,
    // explicitly characterized compound assets may opt into a lower cap.
    std::uint32_t maximumContactsPerBodyPair =
        std::numeric_limits<std::uint32_t>::max();
    std::uint64_t contactCacheMaximumAge = 8u;
    std::uint64_t jointLimitCacheMaximumAge = 8u;
};

struct ArticulatedRigidContactCacheEntry {
    ArticulatedRigidContactWarmStart warmStart{};
    std::uint64_t lastSeenStep = 0u;
};

struct ArticulatedRigidJointLimitCacheEntry {
    std::uint64_t stableKey = 0u;
    double impulse = 0.0;
    std::uint64_t lastSeenStep = 0u;
};

struct ArticulatedRigidGraspCacheEntry {
    std::uint32_t rigidBody = MR_INVALID_INDEX;
    // Hash of the model/jaws, grasp thresholds, rigid slot, and all
    // participating shape generations. Dwell never transfers across a
    // replaced object or changed grasp configuration.
    std::uint64_t identity = 0u;
    std::uint32_t consecutiveQualifiedSteps = 0u;
    bool grasped = false;
    std::uint64_t lastSeenStep = 0u;
};

struct ArticulatedRigidWorldCache {
    PersistentManifoldCache manifolds;
    std::vector<ArticulatedRigidContactCacheEntry> contactImpulses;
    std::vector<ArticulatedRigidJointLimitCacheEntry>
        jointLimitImpulses;
    std::vector<ArticulatedRigidGraspCacheEntry> graspEvidence;
    std::uint64_t step = 0u;
};

struct ArticulatedRigidGraspEvidence {
    std::uint32_t rigidBody = MR_INVALID_INDEX;
    bool jawAContact = false;
    bool jawBContact = false;
    bool qualifiedThisStep = false;
    bool grasped = false;
    std::uint32_t consecutiveQualifiedSteps = 0u;
    double jawANormalImpulse = 0.0;
    double jawBNormalImpulse = 0.0;
    double jawAFriction = 0.0;
    double jawBFriction = 0.0;
    double normalDot = 1.0;
    double maximumTangentialSlipSpeed = 0.0;
};

struct ArticulatedRigidWorldStepDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    ArticulatedRigidWorldFailure failure =
        ArticulatedRigidWorldFailure::none;
    ArticulatedActuationDiagnostics actuation{};
    ArticulatedDynamicsDiagnostics articulatedFreeDynamics{};
    FreeBodyIntegratorDiagnostics rigidFreeDynamics{};
    ArticulatedRigidCollisionDiagnostics collision{};
    ArticulatedJointLimitDiagnostics jointLimitCompilation{};
    CoupledArticulatedRigidContactDiagnostics coupledSolve{};
    ArticulatedDynamicsDiagnostics articulatedIntegration{};
    FreeBodyIntegratorDiagnostics rigidIntegration{};
    std::uint32_t contactCount = 0u;
    std::uint32_t jointLimitCount = 0u;
    std::uint32_t matchedContactWarmStarts = 0u;
    std::uint32_t matchedJointLimitWarmStarts = 0u;
    std::uint32_t graspedBodyCount = 0u;
    double maximumPenetration = 0.0;
    double maximumNormalImpulse = 0.0;
    double maximumJointLimitImpulse = 0.0;
    double maximumVelocityCorrection = 0.0;
    std::vector<ArticulatedRigidGraspEvidence> graspEvidence;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

// One transactional semi-implicit step for exactly one articulation and one
// or more independent dynamic rigid bodies:
//
//   articulated + rigid free-velocity prediction
//   -> collision-generated cross-system contacts
//   -> simultaneous exact-cone contact + joint-limit solve
//   -> one articulation and rigid configuration integration
//
// rigidShapes index rigidBodies and rigidMaterials locally. q, v, rigidBodies,
// every cache stream, and the cache step are unchanged on any failure.
[[nodiscard]] ArticulatedRigidWorldStepDiagnostics
stepArticulatedRigidWorldCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<double> q,
    std::span<double> v,
    std::span<const double> generalizedForce,
    std::span<const ArticulatedBodyWrench> articulatedWrenches,
    std::span<const MRBodyPropertiesGPU> rigidProperties,
    std::span<MRBodyStateGPU> rigidBodies,
    std::span<const MRShapeGPU> rigidShapes,
    std::span<const MRMaterialGPU> rigidMaterials,
    std::span<const BodyWrench> rigidWrenches,
    const ArticulatedRigidWorldConfig& config,
    ArticulatedRigidWorldCache& cache
);

[[nodiscard]] ArticulatedRigidWorldStepDiagnostics
stepControlledArticulatedRigidWorldCpu(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<double> q,
    std::span<double> v,
    std::span<const ArticulatedDofCommand> commands,
    std::span<const ArticulatedBodyWrench> articulatedWrenches,
    std::span<const MRBodyPropertiesGPU> rigidProperties,
    std::span<MRBodyStateGPU> rigidBodies,
    std::span<const MRShapeGPU> rigidShapes,
    std::span<const MRMaterialGPU> rigidMaterials,
    std::span<const BodyWrench> rigidWrenches,
    const ArticulatedRigidWorldConfig& config,
    ArticulatedRigidWorldCache& cache
);

} // namespace metalrobo
