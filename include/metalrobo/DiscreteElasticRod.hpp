#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kDiscreteRodNoRigidBody =
    0xffffffffu;

struct DiscreteRodMaterial {
    double radius = 1.5e-4;
    double density = 1100.0;
    double youngModulus = 5.0e8;
    double poissonRatio = 0.35;
};

struct DiscreteElasticRodModel {
    std::vector<std::array<double, 3>> restPositions;
    std::vector<double> restTwists;
    std::vector<double> restLengths;
    std::vector<double> nodeMasses;
    std::vector<double> edgeRotationalInertias;
    std::vector<double> stretchStiffness;
    std::vector<double> bendStiffness;
    std::vector<double> twistStiffness;
    double radius = 0.0;
    std::string name;
    std::string fidelityBoundary;

    [[nodiscard]] bool valid(
        std::string* reason = nullptr
    ) const;
};

struct DiscreteElasticRodState {
    std::vector<std::array<double, 3>> positions;
    std::vector<std::array<double, 3>> velocities;
    // One scalar material-frame rotation and rate per edge.
    std::vector<double> twists;
    std::vector<double> twistRates;
};

struct DiscreteRodAttachment {
    std::uint32_t nodeIndex = 0u;
    std::array<double, 3> targetPosition{};
    std::array<double, 3> targetVelocity{};
    // Zero is a hard attachment; positive values are XPBD compliance.
    double compliance = 0.0;
};

// Fixed-slot evidence for one attachment. impulseOnTarget is the equal and
// opposite impulse exerted by the rod on the attachment target over the
// complete step; averageForceOnTarget is impulseOnTarget / timestep. This is
// the force-transfer boundary used by rigid needle and tool coupling.
struct DiscreteRodAttachmentReaction {
    std::uint32_t nodeIndex = 0u;
    std::uint32_t bodyIndex = kDiscreteRodNoRigidBody;
    std::array<double, 3> impulseOnTarget{};
    std::array<double, 3> averageForceOnTarget{};
    double finalPositionError = 0.0;
};

struct DiscreteRodRigidAttachmentBinding {
    // Local body index inside each environment. UINT32_MAX leaves the
    // attachment target explicit.
    std::uint32_t bodyIndex = kDiscreteRodNoRigidBody;
    std::array<double, 3> localAnchor{};
};

struct DiscreteElasticRodEnergy {
    double stretch = 0.0;
    double bend = 0.0;
    double twist = 0.0;

    [[nodiscard]] double total() const noexcept {
        return stretch + bend + twist;
    }
};

enum class DiscreteElasticRodStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidModel,
    invalidState,
    invalidAttachment,
    degenerateGeometry,
    didNotConverge,
    nonfiniteResult,
};

struct DiscreteElasticRodStepConfig {
    double timestep = 1.0 / 1000.0;
    std::array<double, 3> gravity{0.0, 0.0, -9.81};
    std::uint32_t solverIterations = 24u;
    double constraintTolerance = 1.0e-7;
    double linearDamping = 0.02;
    double twistDamping = 0.02;
    // Relative perturbation for local bend-constraint derivatives.
    double derivativeStep = 2.0e-6;
    // Non-adjacent edges are treated as capsules with the model radius.
    // Contact is regenerated and relinearized on every nonlinear sweep.
    bool enableSelfCollision = false;
    double selfCollisionMargin = 0.0;
    // Zero is a hard unilateral contact; positive values soften it through
    // the same XPBD compliance convention as attachments.
    double selfCollisionCompliance = 0.0;
};

struct DiscreteElasticRodDiagnostics {
    DiscreteElasticRodStatus status =
        DiscreteElasticRodStatus::success;
    std::uint32_t iterations = 0u;
    std::uint32_t projectedStretchConstraints = 0u;
    std::uint32_t projectedBendConstraints = 0u;
    std::uint32_t projectedTwistConstraints = 0u;
    std::uint32_t projectedAttachments = 0u;
    std::uint32_t projectedSelfContacts = 0u;
    double maximumConstraintError = 0.0;
    double maximumPositionCorrection = 0.0;
    double maximumSelfPenetration = 0.0;
    DiscreteElasticRodEnergy before{};
    DiscreteElasticRodEnergy after{};
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == DiscreteElasticRodStatus::success;
    }
};

// Creates a straight, physically parameterized rod. Stiffnesses are derived
// from EA, EI, and GJ for a circular cross-section. Defaults are explicit
// research values, not calibrated suture-package data.
[[nodiscard]] DiscreteElasticRodModel makeStraightSutureRod(
    std::uint32_t nodeCount,
    double length,
    const DiscreteRodMaterial& material = {}
);

[[nodiscard]] DiscreteElasticRodState
makeDiscreteElasticRodDefaultState(
    const DiscreteElasticRodModel& model
);

// Evaluates stretch plus curvature-binormal bend and material-frame twist
// energy. This is the authoritative rod semantic oracle shared by task
// evidence and later Metal solvers.
[[nodiscard]] DiscreteElasticRodDiagnostics
evaluateDiscreteElasticRodEnergy(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state,
    DiscreteElasticRodEnergy& output
);

// Deterministic implicit XPBD/DER reference step. Bend gradients are local
// numerical derivatives of the exact curvature-binormal constraints; stretch
// and twist use analytic gradients. State publication is transactional.
[[nodiscard]] DiscreteElasticRodDiagnostics
stepDiscreteElasticRodCpu(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    std::span<const DiscreteRodAttachment> attachments = {},
    const DiscreteElasticRodStepConfig& config = {},
    std::span<DiscreteRodAttachmentReaction> reactions = {}
);

[[nodiscard]] const char* discreteElasticRodStatusName(
    DiscreteElasticRodStatus status
) noexcept;

} // namespace metalrobo
