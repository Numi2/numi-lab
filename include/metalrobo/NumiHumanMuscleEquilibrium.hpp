#pragma once

#include "metalrobo/MujocoMuscleReference.hpp"
#include "metalrobo/NumiHumanJointEquality.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// Offline compiler for a source-faithful, muscle-supported Human state. This
// is deliberately outside the dynamics hot loop: it resolves one bounded
// posture/recruitment program which the persistent Metal transaction then
// consumes without host-side control or per-step force restaging.
enum class NumiHumanMuscleEquilibriumStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidArticulation,
    invalidDimensions,
    nonfiniteInput,
    invalidSelection,
    unsupportedMuscleArchitecture,
    equalityFailure,
    kinematicsFailure,
    muscleFailure,
    dynamicsFailure,
    nonfiniteResult,
};

struct NumiHumanMuscleEquilibriumConfig {
    // Used only by the compliant-muscle reference call. Zero-state fibres are
    // first initialized at zero-velocity equilibrium, so a stationary compile
    // is timestep independent within numerical tolerance.
    double timestep = 1.0e-4;
    double activationLimit = 1.0;
    std::uint32_t activationSamples = 9u;
    std::uint32_t activationSweeps = 160u;
    double activationRegularization = 2.5e-4;
    double activationConvergence = 1.0e-7;
    // Recruitment is evaluated in constrained acceleration space, not raw
    // generalized-force units. This prevents small distal-joint torque
    // errors from disappearing beside pelvis/hip loads merely because their
    // effective inertias differ by orders of magnitude.
    double minimumGeneralizedAccelerationScale = 1.0;
    double balanceTolerance = 5.0e-2;

    // Deterministic bounded coordinate search. Only scalar, authoritative
    // position-limited internal DoFs are candidates; root coordinates and
    // quaternion-rate coordinates are never altered. Each accepted posture
    // update is followed by a complete recruitment recompile.
    std::uint32_t poseSweeps = 4u;
    std::uint32_t poseCandidateCount = 8u;
    double poseStepFraction = 0.025;
    double maximumPoseStep = 0.08;
    double positionLimitMarginFraction = 0.01;
    double poseRegularization = 2.5e-3;
    double poseImprovementTolerance = 1.0e-8;
    // A static unilateral reaction exists only at the stop, within this
    // numerical tolerance. The broader runtime activation distance belongs
    // to the velocity-level complementarity solve, not this equilibrium
    // certificate.
    double positionLimitTolerance = 1.0e-7;
    // Optional static support reactions are optimized as nonnegative normal
    // forces. The cap is an admission bound, not a prescribed load.
    double maximumSupportForceNewtons = 5000.0;
    double supportForceRegularization = 1.0e-14;
    std::uint32_t supportForceSweeps = 4096u;
    double supportForceConvergence = 1.0e-10;
};

struct NumiHumanStaticSupportContact {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::array<double, 3> localPoint{};
    // World-space force direction applied to the Human, normally the outward
    // ground-plane normal. Static v1 intentionally admits no adhesion and no
    // tangential force variable.
    std::array<double, 3> normal{0.0, 0.0, 1.0};
};

struct NumiHumanMuscleEquilibriumDiagnostics {
    NumiHumanMuscleEquilibriumStatus status =
        NumiHumanMuscleEquilibriumStatus::success;
    MujocoMuscleReferenceStatus muscleStatus =
        MujocoMuscleReferenceStatus::success;
    ArticulatedDynamicsStatus dynamicsStatus =
        ArticulatedDynamicsStatus::success;
    NumiHumanJointEqualityStatus equalityStatus =
        NumiHumanJointEqualityStatus::success;
    std::uint32_t failingIndex = MR_INVALID_INDEX;
    std::uint32_t muscleCount = 0u;
    std::uint32_t recruitedMuscleCount = 0u;
    std::uint32_t activeMuscleCount = 0u;
    std::uint32_t activationSweeps = 0u;
    std::uint32_t acceptedPoseSteps = 0u;
    std::uint32_t activePositionLimitCount = 0u;
    std::uint32_t jointEqualityCount = 0u;
    std::uint32_t supportContactCount = 0u;
    std::uint32_t activeSupportContactCount = 0u;
    std::uint32_t maximumNormalizedResidualDof = MR_INVALID_INDEX;
    std::uint32_t maximumAccelerationResidualDof = MR_INVALID_INDEX;
    double initialNormalizedResidualRms = 0.0;
    double normalizedResidualRms = 0.0;
    double maximumGeneralizedForceResidual = 0.0;
    double maximumNormalizedAccelerationResidual = 0.0;
    double maximumGeneralizedAccelerationResidual = 0.0;
    double maximumActivation = 0.0;
    double minimumNormalizedPositionLimitMargin = 1.0;
    double maximumPositionLimitReaction = 0.0;
    double maximumJointEqualityReaction = 0.0;
    double maximumInitialEqualityProjection = 0.0;
    double maximumJointEqualityError = 0.0;
    double totalSupportForceNewtons = 0.0;
    double maximumSupportForceNewtons = 0.0;
    double maximumFloatingRootForceResidual = 0.0;
    double maximumFloatingRootAccelerationResidual = 0.0;
    bool balanced = false;
    bool floatingRootIncluded = false;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanMuscleEquilibriumStatus::success;
    }
};

struct NumiHumanMuscleEquilibriumResult {
    NumiHumanMuscleEquilibriumDiagnostics diagnostics{};
    // Articulation-local q and activation in source muscle order.
    std::vector<double> q;
    std::vector<double> activation;
    // Static accepted fibre state matching q/activation. These values are an
    // FP64 oracle; the Metal transaction may independently initialize its
    // typed state from the zero sentinel and parity-check the result.
    std::vector<double> fiberLength;
    std::vector<double> generalizedMuscleForce;
    // Signed articulation-local unilateral reaction. Lower stops contribute
    // positive generalized force and upper stops negative generalized force.
    std::vector<double> generalizedPositionLimitForce;
    std::vector<double> generalizedJointEqualityForce;
    std::vector<double> supportNormalForce;
    std::vector<double> generalizedSupportForce;
    std::vector<double> gravityTarget;
    std::vector<double> generalizedForceResidual;
};

// selectedMuscleIndices controls which muscles may recruit. Empty means all.
// Passive force is always retained for every supplied muscle. The destination
// is published only after a complete finite compile.
[[nodiscard]] NumiHumanMuscleEquilibriumDiagnostics
compileNumiHumanMuscleEquilibrium(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> initialQ,
    std::span<const MujocoMuscleSite> sites,
    std::span<const MujocoWrapGeometry> wraps,
    std::span<const MujocoMuscleDefinition> muscles,
    std::span<const MujocoCompliantMuscleArchitecture> architectures,
    std::span<const MRNumiHumanJointEqualityGPU> jointEqualities,
    std::span<const std::uint32_t> selectedMuscleIndices,
    NumiHumanMuscleEquilibriumResult& result,
    const NumiHumanMuscleEquilibriumConfig& config = {}
);

// Supported overload. Ground reactions and muscle activation are optimized
// in one constrained acceleration-space problem, including the floating root.
// An empty support span is exactly equivalent to the legacy overload above.
[[nodiscard]] NumiHumanMuscleEquilibriumDiagnostics
compileNumiHumanMuscleEquilibrium(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> initialQ,
    std::span<const MujocoMuscleSite> sites,
    std::span<const MujocoWrapGeometry> wraps,
    std::span<const MujocoMuscleDefinition> muscles,
    std::span<const MujocoCompliantMuscleArchitecture> architectures,
    std::span<const MRNumiHumanJointEqualityGPU> jointEqualities,
    std::span<const std::uint32_t> selectedMuscleIndices,
    std::span<const NumiHumanStaticSupportContact> supportContacts,
    NumiHumanMuscleEquilibriumResult& result,
    const NumiHumanMuscleEquilibriumConfig& config = {}
);

[[nodiscard]] const char* numiHumanMuscleEquilibriumStatusName(
    NumiHumanMuscleEquilibriumStatus status
) noexcept;

} // namespace metalrobo
