#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

// Actuation is explicit per generalized-velocity coordinate. Floating-root
// coordinates must remain disabled. Model PD consumes the immutable gains in
// MRDofPropertiesGPU; it therefore means the explicitly selected model preset,
// not intrinsic motor constants. In the canonical G1 model this is the pinned,
// named Unitree RL Lab training preset documented by G1ModelMetadata. Custom PD
// consumes the gains in the command; effort consumes feedForward only.
enum class ArticulatedActuationMode : std::uint32_t {
    disabled = 0u,
    modelPD = 1u,
    customPD = 2u,
    effort = 3u,
};

struct ArticulatedDofCommand {
    ArticulatedActuationMode mode =
        ArticulatedActuationMode::disabled;
    double feedForward = 0.0;
    double desiredPosition = 0.0;
    double desiredVelocity = 0.0;
    double stiffness = 0.0;
    double damping = 0.0;
};

struct ArticulatedActuationConfig {
    // At and below this speed, dry friction is treated as stiction and
    // cancels the actuator load up to drive.w. This is deliberately a
    // controller-local approximation: gravity, bias, external, and contact
    // loads are not available to this evaluator, so complete set-valued
    // stiction belongs in the coupled dynamics/constraint solve. Above the
    // threshold, Coulomb friction has magnitude drive.w and strictly opposes
    // velocity.
    double stictionVelocityThreshold = 1.0e-4;
};

struct ActuatorProfile {
    double jointTorqueConstant = 0.0;
    double currentLimit = 0.0;
    double noLoadSpeed = 0.0;
    double efficiency = 0.0;
    double backlash = 0.0;
    double commandDelaySeconds = 0.0;
    bool calibrated = false;
};

// Transactionally cooks authored SI-unit parameters into the fixed Metal
// record and derives stall torque = torque constant * current limit.
[[nodiscard]] bool cookActuatorProfile(
    const ActuatorProfile& source,
    std::uint32_t globalVIndex,
    MRActuatorProfileGPU& output,
    std::string* reason = nullptr
);

enum class ArticulatedActuationStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidArticulation,
    invalidDimensions,
    nonfiniteInput,
    invalidDofMetadata,
    invalidCommandMode,
    invalidCommandSemantics,
    rootActuationForbidden,
    unactuatedDof,
    missingEffortLimit,
    missingModelDrive,
    positionCoordinateUnavailable,
    nonfiniteResult,
};

struct ArticulatedActuationDiagnostics {
    ArticulatedActuationStatus status =
        ArticulatedActuationStatus::success;
    std::uint32_t articulationIndex = 0u;
    std::uint32_t dofCount = 0u;
    std::uint32_t rejectedLocalDof = MR_INVALID_INDEX;
    std::uint32_t saturatedDofCount = 0u;
    std::uint32_t frictionActiveDofCount = 0u;
    std::uint32_t movingFrictionDofCount = 0u;
    std::uint32_t stictionDofCount = 0u;
    double maximumUnclampedActuatorEffort = 0.0;
    double maximumActuatorEffort = 0.0;
    double maximumPassiveFrictionEffort = 0.0;
    double maximumGeneralizedEffort = 0.0;
    double maximumSaturationExcess = 0.0;
    double movingFrictionDissipation = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ArticulatedActuationStatus::success;
    }
};

struct ArticulatedActuationResult {
    // All arrays are articulation-local and contain exactly articulation.nv
    // entries. generalizedEffort = actuatorEffort + passiveFrictionEffort.
    std::vector<double> actuatorEffort;
    std::vector<double> passiveFrictionEffort;
    std::vector<double> generalizedEffort;
};

// Branch-free scalar definitions mirrored by the native and MLX Metal
// prepare kernels. The profile must already have passed EngineModel::valid.
[[nodiscard]] double actuatorTorqueEnvelope(
    const MRActuatorProfileGPU& profile,
    double jointVelocity,
    double authoredEffortLimit
) noexcept;

// Deterministic play operator for an explicit position-target state.
// Zero play returns the incoming command exactly.
[[nodiscard]] double updateActuatorBacklashTarget(
    double previousEffectiveTarget,
    double commandedTarget,
    double backlashPlay
) noexcept;

// Evaluates one immutable articulation's control and passive dry friction.
//
// q and v are articulation-local despite MRDofPropertiesGPU carrying global
// q/v indices. commands are in articulation-local v order. PD modes evaluate
// exactly:
//   tau = feedForward
//       + stiffness * positionError
//       + damping * (desiredVelocity - v)
// where positionError is desiredPosition - q for bounded coordinates and its
// shortest signed modulo-2pi representative for continuous joints.
//
// Inactive fields are rejected instead of silently ignored:
//   disabled: every scalar must be zero
//   modelPD: command stiffness and damping must be zero
//   customPD: command stiffness and damping must be non-negative
//   effort: desired values, stiffness, and damping must be zero
//
// Every active actuator must be authored as actuated and effort-limited.
// Model PD additionally requires MR_DOF_FLAG_DRIVE. The actuator contribution
// is clamped before passive friction is applied. result is published only
// after every coordinate and every produced value has passed validation.
[[nodiscard]] ArticulatedActuationDiagnostics
evaluateArticulatedActuation(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> q,
    std::span<const double> v,
    std::span<const ArticulatedDofCommand> commands,
    ArticulatedActuationResult& result,
    const ArticulatedActuationConfig& config = {}
);

[[nodiscard]] const char* articulatedActuationStatusName(
    ArticulatedActuationStatus status
) noexcept;

} // namespace metalrobo
